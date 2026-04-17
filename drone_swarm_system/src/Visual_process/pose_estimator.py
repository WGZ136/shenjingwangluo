from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

from Visual_process.types import EgoMotion, Array

# Try to import any torch dependencies, but we don't need them for pose_estimator
# This file uses cv2 and numpy only, so it should work without torch


@dataclass
class PoseEstimatorConfig:
    background_label: int = 0
    ransac_thresh: float = 0.5  # 降低阈值提高鲁棒性
    min_points: int = 15  # 降低最小点数要求
    depth_min: float = 0.3
    depth_max: float = 80.0
    focal_length_px: float = 320.0
    principal_point: Tuple[float, float] = (320.0, 240.0)
    smooth_velocity: float = 0.6
    min_inlier_ratio: float = 0.3  # 最小内点比例


class BackgroundPoseEstimator:
    """Estimate ego pose using points belonging to background clusters."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = PoseEstimatorConfig()
        if config:
            for k, v in config.items():
                setattr(cfg, k, v)
        self.config = cfg
        self.prev_translation = np.zeros(3, dtype=np.float32)
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.prev_timestamp = None

    def _filter_background(
        self, neural_output
    ) -> Optional[Tuple[Array, Array, Array]]:
        points = neural_output.feature_points.get("points_t")
        points_next = neural_output.feature_points.get("points_t_plus_1")
        flows = neural_output.feature_points.get("flow_vectors")
        labels = neural_output.segmentation.get("labels")
        if (
            points is None
            or points_next is None
            or flows is None
            or labels is None
            or len(points) == 0
        ):
            return None
        mask = labels == self.config.background_label
        if mask.sum() < self.config.min_points:
            return None
        return points[mask], points_next[mask], flows[mask]

    def _sample_depths(self, depth_map: Array, coords: Array) -> Array:
        h, w = depth_map.shape
        xs = np.clip(coords[:, 0], 0, w - 1).astype(np.int32)
        ys = np.clip(coords[:, 1], 0, h - 1).astype(np.int32)
        depths = depth_map[ys, xs]
        valid = (depths > self.config.depth_min) & (depths < self.config.depth_max)
        return depths, valid

    def estimate(self, neural_output, intrinsics: Optional[Dict] = None) -> EgoMotion:
        """
        Estimate ego motion from neural output.

        Args:
            neural_output: NeuralOutput containing features and depths
            intrinsics: Camera intrinsics {'fx', 'fy', 'cx', 'cy'} (optional, uses config if None)

        Returns:
            EgoMotion: Estimated ego motion
        """
        samples = self._filter_background(neural_output)
        if samples is None:
            return EgoMotion.identity()

        pts0, pts1, flows = samples
        depth_map = neural_output.depth_maps.get("depth_t")
        ts = neural_output.timestamp
        if depth_map is None:
            return EgoMotion.identity()

        depths, valid = self._sample_depths(depth_map, pts0)
        pts0 = pts0[valid]
        pts1 = pts1[valid]
        depths = depths[valid]
        if len(pts0) < self.config.min_points:
            return EgoMotion.identity()

        # Use provided intrinsics or fall back to config
        if intrinsics:
            fx, fy = intrinsics['fx'], intrinsics['fy']
            cx, cy = intrinsics['cx'], intrinsics['cy']
        else:
            fx = fy = self.config.focal_length_px
            cx, cy = self.config.principal_point

        cam_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            cam_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.config.ransac_thresh,
        )
        if E is None:
            return EgoMotion.identity()

        _, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, cam_matrix)
        inlier_mask = mask_pose.astype(bool).ravel()
        n_inliers = int(inlier_mask.sum())
        
        # 计算内点比例，检查是否满足最小要求
        inlier_ratio = n_inliers / max(len(pts0), 1)
        if n_inliers < self.config.min_points or inlier_ratio < self.config.min_inlier_ratio:
            return EgoMotion.identity()

        translation = t[:, 0].astype(np.float32)
        translation *= np.median(depths[inlier_mask])
        velocity = self._estimate_velocity(translation, ts)

        return EgoMotion(
            rotation=R.astype(np.float32),
            translation=translation,
            velocity=velocity,
            confidence=float(n_inliers / max(len(pts0), 1)),
            inlier_ratio=float(n_inliers / max(len(pts0), 1)),
            n_inliers=n_inliers,
        )

    def _estimate_velocity(self, translation: Array, timestamp: float) -> Array:
        """
        Estimate velocity with smoothing.

        Args:
            translation: Current translation vector
            timestamp: Current timestamp

        Returns:
            velocity: Smoothed velocity vector
        """
        if self.prev_timestamp is None or timestamp <= self.prev_timestamp:
            self.prev_translation = translation.copy()
            self.prev_timestamp = timestamp
            return np.zeros(3, dtype=np.float32)

        dt = max(timestamp - self.prev_timestamp, 1e-3)
        raw_velocity = (translation - self.prev_translation) / dt

        # Exponential smoothing
        smoothed_velocity = (
            self.config.smooth_velocity * self.prev_velocity
            + (1.0 - self.config.smooth_velocity) * raw_velocity
        )

        self.prev_velocity = smoothed_velocity.copy()
        self.prev_translation = translation.copy()
        self.prev_timestamp = timestamp
        return smoothed_velocity.astype(np.float32)

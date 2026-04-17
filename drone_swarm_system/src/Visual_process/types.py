from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


Array = np.ndarray


@dataclass
class EgoMotion:
    rotation: Array = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    translation: Array = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: Array = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    transform: Array = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    confidence: float = 0.0
    inlier_ratio: float = 0.0
    n_inliers: int = 0
    covariance: Optional[Array] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.transform is None or self.transform.shape != (4, 4):
            self.transform = np.eye(4, dtype=np.float32)
        self.transform[:3, :3] = self.rotation
        self.transform[:3, 3] = self.translation

    def __repr__(self) -> str:
        return (f"EgoMotion(translation={self.translation.round(3)}, "
                f"confidence={self.confidence:.3f}, n_inliers={self.n_inliers})")

    @classmethod
    def identity(cls) -> "EgoMotion":
        return cls()

    def as_dict(self) -> Dict[str, Array]:
        return {
            "rotation": self.rotation,
            "translation": self.translation,
            "velocity": self.velocity,
            "transform": self.transform,
            "confidence": self.confidence,
            "inlier_ratio": self.inlier_ratio,
            "n_inliers": self.n_inliers,
        }


@dataclass
class ObstaclePolarFrame:
    angles: Array
    depths: Array
    safety_mask: Array
    forbidden_mask: Array
    closest_angle: float
    closest_depth: float
    coverage_ratio: float
    n_clusters: int = 0
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def empty(cls, num_bins: int, max_depth: float = np.inf) -> "ObstaclePolarFrame":
        angles = np.linspace(-np.pi, np.pi, num_bins, endpoint=False).astype(np.float32)
        depths = np.full(num_bins, max_depth, dtype=np.float32)
        mask = np.zeros(num_bins, dtype=bool)
        return cls(
            angles=angles,
            depths=depths,
            safety_mask=mask.copy(),
            forbidden_mask=mask.copy(),
            closest_angle=float(angles[0]),
            closest_depth=float(max_depth),
            coverage_ratio=0.0,
            warnings=["empty"],
        )

    def as_dict(self) -> Dict[str, Array]:
        return {
            "angles": self.angles,
            "depths": self.depths,
            "safety_mask": self.safety_mask,
            "forbidden_mask": self.forbidden_mask,
            "closest_angle": self.closest_angle,
            "closest_depth": self.closest_depth,
            "coverage_ratio": self.coverage_ratio,
            "n_clusters": self.n_clusters,
        }


@dataclass
class PolarHistorySnapshot:
    timestamp: float
    frame_idx: int
    angles: Array
    depths: Array
    weights: Array


@dataclass
class PolarHistoryBuffer:
    num_bins: int
    max_length: int = 6
    decay: float = 0.7
    _buffer: Deque[PolarHistorySnapshot] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.max_length)

    def push(
        self,
        timestamp: float,
        frame_idx: int,
        angles: Array,
        depths: Array,
        quality: float = 1.0,
    ) -> None:
        if len(angles) != self.num_bins or len(depths) != self.num_bins:
            raise ValueError("角度/深度向量必须与配置的num_bins匹配")
        weights = np.full(self.num_bins, quality, dtype=np.float32)
        self._buffer.append(
            PolarHistorySnapshot(
                timestamp=timestamp,
                frame_idx=frame_idx,
                angles=angles.astype(np.float32),
                depths=depths.astype(np.float32),
                weights=weights,
            )
        )

    def fused_depths(self) -> Optional[Array]:
        if not self._buffer:
            return None
        accum = np.zeros(self.num_bins, dtype=np.float32)
        weight_sum = np.zeros(self.num_bins, dtype=np.float32)
        weight = 1.0
        for snapshot in reversed(self._buffer):
            accum += snapshot.depths * weight
            weight_sum += snapshot.weights * weight
            weight *= self.decay
        weight_sum[weight_sum == 0.0] = 1.0
        return accum / weight_sum

    def latest(self) -> Optional[PolarHistorySnapshot]:
        return self._buffer[-1] if self._buffer else None

    def __len__(self) -> int:
        return len(self._buffer)


@dataclass
class VisualState:
    timestamp: float = 0.0
    frame_idx: int = -1
    ego_motion: EgoMotion = field(default_factory=EgoMotion)
    obstacle_frame: Optional[ObstaclePolarFrame] = None
    history_angles: Optional[Array] = None
    history_depths: Optional[Array] = None
    history_valid_ratio: float = 0.0
    quality: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> Dict:
        payload = {
            "timestamp": self.timestamp,
            "frame_idx": self.frame_idx,
            "ego_motion": self.ego_motion.as_dict(),
            "quality": self.quality,
            "warnings": self.warnings,
        }
        if self.obstacle_frame is not None:
            payload["obstacle_polar"] = self.obstacle_frame.as_dict()
        if self.history_angles is not None and self.history_depths is not None:
            payload["history"] = {
                "angles": self.history_angles,
                "depths": self.history_depths,
                "valid_ratio": self.history_valid_ratio,
            }
        if self.debug:
            payload["debug"] = self.debug
        return payload

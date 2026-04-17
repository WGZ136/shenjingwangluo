"""
多摄像头障碍物融合模块

功能：
- 融合多个摄像头的障碍物信息
- 生成360°全覆盖的障碍物极坐标表示

支持配置：
- 4摄像头方案：前、后、左、右
- 角度偏移和FOV配置
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from Visual_process.types import ObstaclePolarFrame


@dataclass
class CameraConfig:
    name: str
    fov_deg: float  # 视场角（度）
    angle_offset_deg: float  # 相对于机头的角度偏移（度）
    
    def __post_init__(self):
        self.fov_rad = np.radians(self.fov_deg)
        self.angle_offset_rad = np.radians(self.angle_offset_deg)


class MultiCameraFusion:
    """
    多摄像头障碍物融合器
    
    将多个摄像头的ObstaclePolarFrame融合为360°全覆盖的障碍物信息
    """
    
    DEFAULT_CAMERAS = [
        CameraConfig(name="front", fov_deg=90, angle_offset_deg=0),
        CameraConfig(name="right", fov_deg=90, angle_offset_deg=90),
        CameraConfig(name="back", fov_deg=90, angle_offset_deg=180),
        CameraConfig(name="left", fov_deg=90, angle_offset_deg=-90),
    ]
    
    def __init__(
        self,
        num_bins: int = 72,
        max_depth: float = 20.0,
        cameras: Optional[List[CameraConfig]] = None,
    ):
        """
        初始化多摄像头融合器
        
        Args:
            num_bins: 极坐标向量 bins 数量（360°全覆盖）
            max_depth: 最大感知距离（米）
            cameras: 摄像头配置列表
        """
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.cameras = cameras or self.DEFAULT_CAMERAS
        
        self.global_angles = np.linspace(-np.pi, np.pi, num_bins, endpoint=False).astype(np.float32)
        self.global_depths = np.full(num_bins, max_depth, dtype=np.float32)
    
    def fuse(
        self,
        camera_frames: Dict[str, ObstaclePolarFrame],
        quality_weights: Optional[Dict[str, float]] = None,
    ) -> ObstaclePolarFrame:
        """
        融合多个摄像头的障碍物信息
        
        Args:
            camera_frames: 摄像头名称到ObstaclePolarFrame的映射
            quality_weights: 各摄像头质量权重（可选）
            
        Returns:
            融合后的ObstaclePolarFrame（360°覆盖）
        """
        if quality_weights is None:
            quality_weights = {cam.name: 1.0 for cam in self.cameras}
        
        fused_depths = np.full(self.num_bins, self.max_depth, dtype=np.float32)
        coverage_count = np.zeros(self.num_bins, dtype=np.float32)
        safety_mask = np.zeros(self.num_bins, dtype=bool)
        forbidden_mask = np.zeros(self.num_bins, dtype=bool)
        
        for camera in self.cameras:
            if camera.name not in camera_frames:
                continue
            
            frame = camera_frames[camera.name]
            weight = quality_weights.get(camera.name, 1.0)
            
            camera_depths, camera_coverage = self._map_camera_to_global(
                frame.depths,
                frame.angles,
                camera.angle_offset_rad,
                weight
            )
            
            valid_mask = (camera_depths < self.max_depth) & (camera_depths > 0)
            fused_depths = np.where(
                valid_mask & (camera_depths < fused_depths),
                camera_depths,
                fused_depths
            )
            coverage_count += camera_coverage
            
            if frame.safety_mask is not None:
                mapped_safety = self._map_mask_to_global(
                    frame.safety_mask,
                    frame.angles,
                    camera.angle_offset_rad
                )
                safety_mask |= mapped_safety
            
            if frame.forbidden_mask is not None:
                mapped_forbidden = self._map_mask_to_global(
                    frame.forbidden_mask,
                    frame.angles,
                    camera.angle_offset_rad
                )
                forbidden_mask |= mapped_forbidden
        
        coverage_ratio = np.mean(coverage_count > 0)
        
        closest_idx = np.argmin(fused_depths)
        closest_angle = float(np.degrees(self.global_angles[closest_idx]))
        closest_depth = float(fused_depths[closest_idx])
        
        return ObstaclePolarFrame(
            angles=self.global_angles,
            depths=fused_depths,
            safety_mask=safety_mask,
            forbidden_mask=forbidden_mask,
            closest_angle=closest_angle,
            closest_depth=closest_depth,
            coverage_ratio=coverage_ratio,
            n_clusters=0,
        )
    
    def _map_camera_to_global(
        self,
        camera_depths: np.ndarray,
        camera_angles: np.ndarray,
        angle_offset_rad: float,
        weight: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将摄像头角度映射到全局360°坐标系
        
        Args:
            camera_depths: 摄像头深度数据
            camera_angles: 摄像头角度向量（弧度）
            angle_offset_rad: 摄像头相对于机头的角度偏移
            weight: 质量权重
            
        Returns:
            (global_depths, coverage): 全局深度和覆盖标记
        """
        global_depths = np.full(self.num_bins, self.max_depth, dtype=np.float32)
        coverage = np.zeros(self.num_bins, dtype=np.float32)
        
        if len(camera_angles) == 0 or len(camera_depths) == 0:
            return global_depths, coverage
        
        camera_angles_shifted = camera_angles + angle_offset_rad
        
        for i, global_angle in enumerate(self.global_angles):
            angle_diff = np.abs(camera_angles_shifted - global_angle)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            
            min_idx = np.argmin(angle_diff)
            if angle_diff[min_idx] < np.pi / len(camera_angles):
                global_depths[i] = camera_depths[min_idx]
                coverage[i] = weight
        
        return global_depths, coverage
    
    def _map_mask_to_global(
        self,
        camera_mask: np.ndarray,
        camera_angles: np.ndarray,
        angle_offset_rad: float,
    ) -> np.ndarray:
        """将摄像头的mask映射到全局坐标系"""
        global_mask = np.zeros(self.num_bins, dtype=bool)
        
        if len(camera_angles) == 0 or len(camera_mask) == 0:
            return global_mask
        
        camera_angles_shifted = camera_angles + angle_offset_rad
        
        for i, global_angle in enumerate(self.global_angles):
            angle_diff = np.abs(camera_angles_shifted - global_angle)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            
            min_idx = np.argmin(angle_diff)
            if angle_diff[min_idx] < np.pi / len(camera_angles):
                global_mask[i] = camera_mask[min_idx]
        
        return global_mask
    
    @classmethod
    def create_quad_camera(cls, num_bins: int = 72, max_depth: float = 20.0) -> "MultiCameraFusion":
        """创建标准的4摄像头配置（前/后/左/右）"""
        return cls(
            num_bins=num_bins,
            max_depth=max_depth,
            cameras=cls.DEFAULT_CAMERAS
        )


def fuse_single_camera(
    frame: ObstaclePolarFrame,
    camera_fov: float,
    camera_offset: float,
    num_bins: int = 72,
    max_depth: float = 20.0,
) -> ObstaclePolarFrame:
    """
    辅助函数：将单个摄像头的障碍物扩展到全局坐标系
    
    用于需要模拟单摄像头场景时
    """
    fusion = MultiCameraFusion(
        num_bins=num_bins,
        max_depth=max_depth,
        cameras=[
            CameraConfig(name="virtual", fov_deg=camera_fov, angle_offset_deg=camera_offset)
        ]
    )
    
    return fusion.fuse({"virtual": frame})

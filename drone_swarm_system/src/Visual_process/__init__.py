"""视觉处理模块包

基于AirSim项目的视觉处理模块，包含：
- 位姿估计 (pose_estimator)
- 障碍物检测 (obstacle_detect)
- 多相机融合 (multi_camera_fusion)
- 视觉中心 (visual_center)
- 数据类型定义 (types)
"""

from .types import (
    EgoMotion,
    ObstaclePolarFrame,
    PolarHistorySnapshot,
    PolarHistoryBuffer,
    VisualState
)
from .pose_estimator import BackgroundPoseEstimator
from .obstacle_detect import ObstacleProcessor
from .multi_camera_fusion import MultiCameraFusion, CameraConfig
from .visual_center import VisualPerception

__all__ = [
    "EgoMotion",
    "ObstaclePolarFrame",
    "PolarHistorySnapshot",
    "PolarHistoryBuffer",
    "VisualState",
    "BackgroundPoseEstimator",
    "ObstacleProcessor",
    "MultiCameraFusion",
    "CameraConfig",
    "VisualPerception",
]

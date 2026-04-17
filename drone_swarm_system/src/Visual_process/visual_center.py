from __future__ import annotations

import numpy as np
from typing import Optional, Dict

# Import existing modules
try:
    from .obstacle_detect import ObstacleProcessor
    from .pose_estimator import BackgroundPoseEstimator
    from .types import VisualState, EgoMotion, ObstaclePolarFrame
except Exception:
    # Fallback imports for local development
    from Visual_process.obstacle_detect import ObstacleProcessor
    from Visual_process.pose_estimator import BackgroundPoseEstimator
    from Visual_process.types import VisualState, EgoMotion, ObstaclePolarFrame

class VisualPerception:
    """
    VisualPerception hub.
    Receives NeuralOutput (via NeuralPerception) and optional camera intrinsics,
    processes ego-motion and obstacle information, and returns a VisualState object.
    """

    def __init__(self, config: Optional[Dict] = None):
        # Initialize sub-modules with sensible defaults
        self.intrinsics = None

        # Obstacle processing (polar mapping + history fusion)
        self.obstacle_processor = ObstacleProcessor(
            num_angles=config.get('num_angles', 360) if config else 360,
            max_distance=config.get('max_distance', 10.0) if config else 10.0,
            fov_horizontal=config.get('fov_horizontal', 90.0) if config else 90.0,
            history_size=config.get('history_size', 5) if config else 5,
            decay_factor=config.get('decay_factor', 0.7) if config else 0.7
        )

        # Ego motion estimator (background pose)
        self.pose_estimator = BackgroundPoseEstimator()

        # Optional: allow external config to tweak behavior
        self.config = config or {}
        
        # 保存上一帧的ego_motion用于时空融合
        self._prev_ego_motion = None

    def _load_intrinsics(self, intrinsics: Optional[Dict] = None) -> Dict:
        """
        Return a camera intrinsics dict with keys: fx, fy, cx, cy
        If intrinsics provided, use it. Otherwise, synthesize defaults based on settings.
        """
        if intrinsics:
            fx = intrinsics.get('fx')
            fy = intrinsics.get('fy')
            cx = intrinsics.get('cx')
            cy = intrinsics.get('cy')
            if fx is None or fy is None or cx is None or cy is None:
                # fall back to defaults if partial
                fx = intrinsics.get('fx', 320.0)
                fy = intrinsics.get('fy', 320.0)
                cx = intrinsics.get('cx', 320.0)
                cy = intrinsics.get('cy', 240.0)
            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        else:
            # default intrinsics (assume 640x480, fx=fy=320, cx=320, cy=240)
            return {'fx': 320.0, 'fy': 320.0, 'cx': 320.0, 'cy': 240.0}

    def process(self, neural_output, intrinsics: Optional[Dict] = None) -> VisualState:
        """
        Process a NeuralOutput to produce a VisualState.
        Returns an instance of VisualState (with to_payload for serialization).
        """
        # Load intrinsics
        cam_intrinsics = self._load_intrinsics(intrinsics)

        # 1) ego motion estimation
        ego_motion = self.pose_estimator.estimate(neural_output, intrinsics=cam_intrinsics)

        # 2) obstacle detection (polar)
        polar_frame = self.obstacle_processor.update(neural_output, cam_intrinsics)

        # 3) build VisualState
        state = VisualState(
            timestamp=getattr(neural_output, 'timestamp', 0.0),
            frame_idx=getattr(neural_output, 'frame_idx', -1),
            ego_motion=ego_motion,
            obstacle_frame=polar_frame,
            history_angles=None,
            history_depths=None,
            history_valid_ratio=0.0,
            quality=getattr(neural_output, 'quality_metrics', {}),
            warnings=getattr(neural_output, 'warnings', []),
            debug=getattr(neural_output, 'debug', {})
        )

        return state


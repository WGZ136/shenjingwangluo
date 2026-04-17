"""
神经网络感知输出数据类

定义神经网络处理后的统一输出格式，供视觉处理模块使用
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class NeuralOutput:
    """
    神经网络输出的统一数据容器
    
    整合深度估计、光流估计、特征点检测、分割等神经网络的输出结果
    """
    
    timestamp: float = 0.0
    frame_idx: int = -1
    
    depth_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    """深度图字典，包含 'depth_t' 等"""
    
    feature_points: Dict[str, np.ndarray] = field(default_factory=dict)
    """特征点字典，包含:
        - 'points_t': t时刻特征点坐标 [N, 2]
        - 'points_t_plus_1': t+1时刻特征点坐标 [N, 2]
        - 'flow_vectors': 光流向量 [N, 2]
    """
    
    segmentation: Dict[str, np.ndarray] = field(default_factory=dict)
    """分割结果字典，包含:
        - 'labels': 分割标签 [H, W]
    """
    
    optical_flow: Dict[str, np.ndarray] = field(default_factory=dict)
    """光流结果字典，包含:
        - 'flow': 光流场 [H, W, 2]
    """
    
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    """质量评估指标，如 overall_confidence"""
    
    warnings: List[str] = field(default_factory=list)
    """处理过程中的警告信息"""
    
    debug: Dict[str, Any] = field(default_factory=dict)
    """调试信息"""
    
    @classmethod
    def from_processing_result(cls, result, timestamp: float = 0.0, frame_idx: int = -1):
        """
        从 ProcessingResult 创建 NeuralOutput
        
        Args:
            result: DroneSwarmSystem 的 ProcessingResult
            timestamp: 时间戳
            frame_idx: 帧索引
            
        Returns:
            NeuralOutput 实例
        """
        neural_output = cls(
            timestamp=timestamp,
            frame_idx=frame_idx,
            quality_metrics={'overall_confidence': 1.0 if result.success else 0.0},
            warnings=[] if result.success else ['Processing failed']
        )
        
        if result.depth_map is not None:
            neural_output.depth_maps['depth_t'] = result.depth_map
            
        if result.flow_vectors is not None:
            neural_output.optical_flow['flow'] = result.flow_vectors
            
        if result.segmentation_labels is not None:
            neural_output.segmentation['labels'] = result.segmentation_labels
            
        return neural_output
    
    @classmethod
    def create_mock(cls, 
                    image_shape: Tuple[int, int] = (480, 640),
                    num_features: int = 100,
                    timestamp: float = 0.0):
        """
        创建模拟数据用于测试
        
        Args:
            image_shape: 图像尺寸 (H, W)
            num_features: 特征点数量
            timestamp: 时间戳
            
        Returns:
            NeuralOutput 实例（模拟数据）
        """
        h, w = image_shape
        
        depth_map = np.random.uniform(0.5, 10.0, (h, w)).astype(np.float32)
        
        points_t = np.random.rand(num_features, 2) * [w, h]
        points_t_plus_1 = points_t + np.random.randn(num_features, 2) * 2
        flow_vectors = points_t_plus_1 - points_t
        
        labels = np.random.randint(0, 3, (h, w)).astype(np.int32)
        
        return cls(
            timestamp=timestamp,
            frame_idx=0,
            depth_maps={'depth_t': depth_map},
            feature_points={
                'points_t': points_t.astype(np.float32),
                'points_t_plus_1': points_t_plus_1.astype(np.float32),
                'flow_vectors': flow_vectors.astype(np.float32)
            },
            segmentation={'labels': labels},
            quality_metrics={'overall_confidence': 0.85}
        )

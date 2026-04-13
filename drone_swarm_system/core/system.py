"""
无人机集群系统核心类

包含主系统类和处理结果数据类
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

# 从配置模块导入
from .config import (
    CUDA_AVAILABLE, 
    DEFAULT_DEVICE, 
    DEFAULT_CONFIG,
    merge_config,
    check_cuda_available,
    PROJECT_ROOT,
    CORE_ALGORITHMS_DIR
)

# 添加路径
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_ALGORITHMS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src"))

# 导入项目模块
try:
    from modules.depth_estimator import Monodepth2Estimator
    from modules.flow_processor import FlowProcessor
    from modules.clustering import TraditionalSegmenter
    from modules.pose_estimator import PoseEstimator
    from modules.geometry_utils import GeometryProcessor
    from utils.visualization import Visualizer
    MODULES_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"部分模块导入失败: {e}")
    MODULES_AVAILABLE = False


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    depth_map: Optional[np.ndarray] = None
    flow_vectors: Optional[np.ndarray] = None
    segmentation_labels: Optional[np.ndarray] = None
    pose_transform: Optional[Dict] = None
    cluster_centers: Optional[List] = None
    processing_time: float = 0.0
    success: bool = False
    message: str = ""


class DroneSwarmSystem:
    """
    无人机集群系统主类
    
    集成以下功能模块:
    - 深度估计 (Monodepth2)
    - 光流估计 (RAFT)
    - 聚类分割 (K-means/DBSCAN)
    - 位姿估计 (RANSAC/SVD)
    - 可视化
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化无人机集群系统
        
        Args:
            config: 自定义配置字典，会覆盖默认配置
        """
        self.config = merge_config(config)
        self._initialized = False
        
        # 模块实例
        self.depth_estimator = None
        self.flow_processor = None
        self.segmenter = None
        self.pose_estimator = None
        self.geometry_processor = None
        self.visualizer = None
        
        # 状态
        self.frame_count = 0
        self.last_frame = None
        
        print("=" * 60)
        print("无人机集群系统初始化")
        print("=" * 60)
        self._print_device_info()
    
    def _print_device_info(self):
        """打印设备信息"""
        print("\n📊 系统配置信息:")
        print(f"   深度估计设备: {self.config['depth']['device']}")
        print(f"   光流使用GPU: {self.config['flow']['use_gpu']}")
        if CUDA_AVAILABLE:
            import torch
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   运行模式: CPU")
    
    def initialize(self) -> bool:
        """初始化所有模块"""
        if self._initialized:
            return True
        
        try:
            if self.config['depth']['enabled']:
                print("\n[1/5] 初始化深度估计器...")
                self.depth_estimator = Monodepth2Estimator(
                    model_path=self.config['depth']['model_path'],
                    input_width=self.config['depth']['input_width'],
                    input_height=self.config['depth']['input_height'],
                    min_depth=self.config['depth']['min_depth'],
                    max_depth=self.config['depth']['max_depth'],
                    device=self.config['depth']['device']
                )
            
            if self.config['flow']['enabled']:
                print("\n[2/5] 初始化光流处理器...")
                self.flow_processor = FlowProcessor(config=self.config['flow'])
            
            if self.config['clustering']['enabled']:
                print("\n[3/5] 初始化聚类分割器...")
                self.segmenter = TraditionalSegmenter(
                    method=self.config['clustering']['method'],
                    n_clusters=self.config['clustering']['n_clusters'],
                    eps=self.config['clustering']['eps'],
                    min_samples=self.config['clustering']['min_samples']
                )
            
            if self.config['pose']['enabled']:
                print("\n[4/5] 初始化位姿估计器...")
                self.pose_estimator = PoseEstimator(config=self.config['pose'])
            
            print("\n[5/5] 初始化几何处理器...")
            self.geometry_processor = GeometryProcessor()
            
            if self.config['visualization']['enabled']:
                self.visualizer = Visualizer()
                output_dir = self.config['visualization']['output_dir']
                if self.config['visualization']['save_results']:
                    os.makedirs(output_dir, exist_ok=True)
            
            self._initialized = True
            print("\n" + "=" * 60)
            print("✅ 系统初始化完成")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_frame(self, 
                     current_frame: np.ndarray,
                     previous_frame: Optional[np.ndarray] = None) -> ProcessingResult:
        """处理单帧图像"""
        start_time = time.time()
        
        if not self._initialized:
            if not self.initialize():
                return ProcessingResult(success=False, message="系统初始化失败")
        
        result = ProcessingResult()
        
        try:
            if self.depth_estimator:
                result.depth_map = self.depth_estimator.estimate(current_frame)
            
            if self.flow_processor and previous_frame is not None:
                result.flow_vectors = self.flow_processor.compute_flow(previous_frame, current_frame)
            
            if self.segmenter and result.flow_vectors is not None:
                labels, centers = self.segment_clusters(result.flow_vectors)
                result.segmentation_labels = labels
                result.cluster_centers = centers
            
            if self.pose_estimator and result.depth_map is not None and self.last_frame is not None:
                result.pose_transform = self.estimate_pose(self.last_frame, current_frame, result.depth_map)
            
            self.last_frame = current_frame.copy()
            self.frame_count += 1
            
            result.processing_time = time.time() - start_time
            result.success = True
            result.message = "处理成功"
            
        except Exception as e:
            result.success = False
            result.message = f"处理失败: {str(e)}"
            import traceback
            traceback.print_exc()
        
        return result
    
    def segment_clusters(self, 
                        flow_vectors: np.ndarray,
                        points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List]:
        """对光流向量进行聚类分割"""
        if self.segmenter is None:
            raise RuntimeError("聚类分割器未初始化")
        
        if len(flow_vectors.shape) == 3:
            h, w = flow_vectors.shape[:2]
            flow_flat = flow_vectors.reshape(-1, 2)
            if points is None:
                y, x = np.mgrid[0:h, 0:w]
                points = np.stack([x, y], axis=-1).reshape(-1, 2)
        else:
            flow_flat = flow_vectors
        
        labels = self.segmenter.segment(points, flow_flat)
        
        centers = []
        for label in np.unique(labels):
            if label == -1:
                continue
            mask = labels == label
            center = np.mean(points[mask], axis=0)
            centers.append(center)
        
        return labels, centers
    
    def estimate_pose(self, image1, image2, depth_map=None):
        """估计两帧之间的位姿变换"""
        return {
            'rotation': np.eye(3),
            'translation': np.zeros(3),
            'success': False,
            'message': '位姿估计需要更多实现'
        }
    
    def visualize(self, image, result, save_path=None):
        """可视化处理结果"""
        if self.visualizer is None:
            return image
        return image  # 简化版本
    
    def process_video(self, video_path, output_path=None, max_frames=None):
        """处理视频文件"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        results = []
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame, prev_frame)
            results.append(result)
            prev_frame = frame
            
            if max_frames and len(results) >= max_frames:
                break
        
        cap.release()
        return results
    
    def get_status(self):
        """获取系统状态"""
        return {
            'initialized': self._initialized,
            'frame_count': self.frame_count,
            'modules': {
                'depth': self.depth_estimator is not None,
                'flow': self.flow_processor is not None,
                'clustering': self.segmenter is not None,
                'pose': self.pose_estimator is not None,
            },
            'config': self.config
        }
    
    def reset(self):
        """重置系统状态"""
        self.frame_count = 0
        self.last_frame = None
    
    def release(self):
        """释放资源"""
        if self.depth_estimator:
            del self.depth_estimator
        if self.flow_processor:
            del self.flow_processor
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# 便捷函数
def create_system(config=None):
    """创建系统实例的便捷函数"""
    return DroneSwarmSystem(config)


def quick_process(image, prev_image=None, config=None):
    """快速处理单帧的便捷函数"""
    with DroneSwarmSystem(config) as system:
        return system.process_frame(image, prev_image)

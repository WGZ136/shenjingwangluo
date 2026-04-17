"""
无人机集群系统核心类

包含主系统类和处理结果数据类
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
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
    CORE_ALGORITHMS_DIR,
    SystemConfig
)

# 添加路径
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_ALGORITHMS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src"))

# 导入项目模块
try:
    try:
        from modules.depth_estimator import Monodepth2Estimator
    except ImportError:
        try:
            from depth_estimator import Monodepth2Estimator
        except ImportError:
            Monodepth2Estimator = None
    from modules.flow_processor import FlowProcessor
    from modules.clustering import TraditionalSegmenter
    from modules.pose_estimator import PoseEstimator
    from modules.geometry_utils import GeometryProcessor
    from utils.visualization import Visualizer
    MODULES_AVAILABLE = True
except ImportError as e:
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src" / "utils"))
        from visualization import Visualizer
        MODULES_AVAILABLE = True
    except ImportError:
        warnings.warn(f"部分模块导入失败: {e}")
        MODULES_AVAILABLE = False

# 尝试导入视觉整合模块
try:
    from visual_integration import VisualIntegration, MotionMatrix
    VISUAL_INTEGRATION_AVAILABLE = True
except ImportError:
    VISUAL_INTEGRATION_AVAILABLE = False

# 尝试导入ESP32接收器
try:
    from esp32_receiver import DualCameraReceiver
    ESP32_AVAILABLE = True
except ImportError:
    ESP32_AVAILABLE = False


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
    # 新增字段
    motion_matrix: Optional['MotionMatrix'] = None
    point_cloud: Any = None
    cluster_info: Optional[Dict] = None


class DroneSwarmSystem:
    """
    无人机集群系统主类
    
    集成以下功能模块:
    - 深度估计 (Monodepth2)
    - 光流估计 (RAFT)
    - 聚类分割 (K-means/DBSCAN)
    - 位姿估计 (RANSAC/SVD)
    - ESP32图传接收
    - 视觉整合 (点云、运动矩阵)
    - 可视化
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化无人机集群系统
        
        Args:
            config: 配置对象 (SystemConfig 或 dict) 或 None 使用默认配置
        """
        self._initialized = False
        
        # 处理配置
        if config is None:
            self.config = merge_config()
            self.system_config = None
        elif isinstance(config, SystemConfig):
            self.system_config = config
            self.config = merge_config(config.to_dict())
        else:
            self.config = merge_config(config)
            self.system_config = None
        
        # 模块实例
        self.depth_estimator = None
        self.flow_processor = None
        self.segmenter = None
        self.pose_estimator = None
        self.geometry_processor = None
        self.visualizer = None
        
        # 视觉整合模块
        self.visual_integration: Optional[VisualIntegration] = None
        
        # ESP32接收器
        self.esp32_receiver: Optional[DualCameraReceiver] = None
        
        # 状态
        self.frame_count = 0
        self.last_frame = None
        self._current_frame = None
        
        print("=" * 60)
        print("无人机集群系统初始化")
        print("=" * 60)
        self._print_device_info()
    
    def _print_device_info(self):
        """打印设备信息"""
        print("\n📊 系统配置信息:")
        print(f"   深度估计设备: {self.config['depth']['device']}")
        print(f"   光流使用GPU: {self.config['flow']['use_gpu']}")
        print(f"   视觉整合: {'可用' if VISUAL_INTEGRATION_AVAILABLE else '不可用'}")
        print(f"   ESP32接收: {'可用' if ESP32_AVAILABLE else '不可用'}")
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
            
            # 初始化视觉整合模块
            if VISUAL_INTEGRATION_AVAILABLE and self.system_config:
                print("\n[6/6] 初始化视觉整合模块...")
                self.visual_integration = VisualIntegration(
                    config=self.config,
                    enable_3d_visualization=self.system_config.enable_3d_visualization,
                    camera_intrinsics=self.system_config.camera_intrinsics,
                    enable_esp32_receiver=self.system_config.enable_esp32_receiver
                )
                
                # 如果启用了ESP32接收，保存接收器引用
                if self.system_config.enable_esp32_receiver and self.visual_integration:
                    self.esp32_receiver = self.visual_integration.esp32_receiver
            
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
                     current_frame: Optional[np.ndarray] = None,
                     previous_frame: Optional[np.ndarray] = None,
                     color_image: Optional[np.ndarray] = None,
                     timestamp: float = 0.0) -> ProcessingResult:
        """
        处理单帧图像
        
        Args:
            current_frame: 当前帧图像 (如果为None，会尝试从ESP32获取)
            previous_frame: 上一帧图像
            color_image: 彩色图像（用于点云着色）
            timestamp: 时间戳
            
        Returns:
            ProcessingResult 处理结果
        """
        start_time = time.time()
        
        if not self._initialized:
            if not self.initialize():
                return ProcessingResult(success=False, message="系统初始化失败")
        
        # 如果没有提供帧，尝试从ESP32获取
        if current_frame is None and self.esp32_receiver:
            frames = self.esp32_receiver.read_both()
            if frames[0] is not None:
                current_frame = frames[0]
                if color_image is None:
                    color_image = current_frame
        
        if current_frame is None:
            return ProcessingResult(success=False, message="没有可用图像")
        
        result = ProcessingResult()
        
        try:
            # 基础处理
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
            
            # 视觉整合处理（如果有）
            if self.visual_integration:
                integration_result = self.visual_integration.process_frame(
                    processing_result=result,
                    color_image=color_image or current_frame,
                    timestamp=timestamp if timestamp else time.time(),
                    frame_idx=self.frame_count
                )
                
                result.motion_matrix = integration_result.get('motion_matrix')
                result.point_cloud = integration_result.get('point_cloud')
                result.cluster_info = integration_result.get('cluster_info')
            
            self.last_frame = current_frame.copy()
            self._current_frame = current_frame
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
    
    def visualize(self, image=None, result=None, save_path=None):
        """可视化处理结果"""
        if image is None:
            image = self._current_frame
        
        if image is None:
            return None
        
        # 使用视觉整合的可视化
        if self.visual_integration:
            self.visual_integration.display_esp32_feed()
            return image
        
        # 基础可视化
        if self.visualizer:
            return self.visualizer.visualize(image)
        
        return image
    
    def visualize_esp32_feed(self):
        """显示ESP32图传画面"""
        if self.visual_integration:
            self.visual_integration.display_esp32_feed()
        elif self.esp32_receiver:
            self.esp32_receiver.display()
    
    def save_pointcloud(self, filename: str):
        """保存点云"""
        if self.visual_integration:
            self.visual_integration.save_pointcloud(filename)
        else:
            print("视觉整合模块不可用，无法保存点云")
    
    def simulate_multiple_drones(self, frame_results: List[ProcessingResult], positions: List[np.ndarray]) -> List[Dict]:
        """
        模拟多无人机（单机多位置拍摄）
        
        Args:
            frame_results: 多帧处理结果
            positions: 各帧对应的无人机位置
            
        Returns:
            各无人机的集群控制信息列表
        """
        if self.visual_integration:
            return self.visual_integration.simulate_multiple_drones(
                [r.__dict__ for r in frame_results], positions
            )
        else:
            print("视觉整合模块不可用")
            return []
    
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
                'visual_integration': self.visual_integration is not None,
                'esp32_receiver': self.esp32_receiver is not None,
            },
            'config': self.config
        }
    
    def reset(self):
        """重置系统状态"""
        self.frame_count = 0
        self.last_frame = None
        self._current_frame = None
    
    def release(self):
        """释放资源"""
        if self.visual_integration:
            self.visual_integration.release()
            self.visual_integration = None
        
        if self.esp32_receiver:
            self.esp32_receiver.stop()
            self.esp32_receiver = None
        
        if self.depth_estimator:
            del self.depth_estimator
        if self.flow_processor:
            del self.flow_processor
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False
        print("系统资源已释放")
    
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

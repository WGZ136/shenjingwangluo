"""
视觉处理模块整合适配器

将 Visual_process 模块整合到 DroneSwarmSystem 中
提供：
1. 从 ProcessingResult 到 NeuralOutput 的转换
2. 视觉感知处理流程
3. 点云生成和3D可视化
4. 运动矩阵提取（供集群控制使用）
5. ESP32-CAM 图传接收
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
import warnings
import time

# 类型检查导入 (用于Pylance静态分析)
if TYPE_CHECKING:
    try:
        import open3d as o3d
    except ImportError:
        pass

# 尝试导入Open3D
o3d = None
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    from neural_processing.neural_perception import NeuralOutput
    from Visual_process.visual_center import VisualPerception
    from Visual_process.types import VisualState, EgoMotion, ObstaclePolarFrame
    from open3d_utils.pointcloud_processor import PointCloudProcessor, PointCloudConfig
    from open3d_utils.visualization_3d import PointCloudVisualizer, DronePose
    from esp32_receiver.camera_receiver import DualCameraReceiver, CameraFrame
    VISUAL_PROCESS_AVAILABLE = True
except ImportError as e:
    VISUAL_PROCESS_AVAILABLE = False
    warnings.warn(f"Visual_process 模块导入失败: {e}")


@dataclass
class MotionMatrix:
    """
    运动矩阵数据结构
    供集群控制使用
    """
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    transformation: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    timestamp: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist(),
            'transformation': self.transformation.tolist(),
            'velocity': self.velocity.tolist(),
            'timestamp': self.timestamp,
            'confidence': self.confidence
        }


@dataclass
class ClusterControlInfo:
    """
    集群控制所需的信息
    """
    drone_id: str = "drone_0"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    nearest_obstacle_distance: float = float('inf')
    nearest_obstacle_angle: float = 0.0
    surrounding_obstacles: List[Tuple[float, float]] = field(default_factory=list)
    """障碍物列表: [(angle, distance), ...]"""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'drone_id': self.drone_id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'nearest_obstacle_distance': self.nearest_obstacle_distance,
            'nearest_obstacle_angle': self.nearest_obstacle_angle,
            'surrounding_obstacles': self.surrounding_obstacles
        }


class VisualIntegration:
    """
    视觉处理整合器
    
    整合 Visual_process 模块到 DroneSwarmSystem，提供：
    - 深度图转3D点云
    - 位姿估计和运动矩阵提取
    - 障碍物检测
    - 3D可视化
    - ESP32-CAM 图传接收
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 enable_3d_visualization: bool = False,
                 camera_intrinsics: Optional[Dict] = None,
                 enable_esp32_receiver: bool = False,
                 frame_callback: Optional[Callable] = None):
        """
        初始化视觉整合器
        
        Args:
            config: 配置字典
            enable_3d_visualization: 是否启用3D可视化
            camera_intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
            enable_esp32_receiver: 是否启用ESP32图传接收
            frame_callback: 新帧回调函数(frame_dict: Dict[str, np.ndarray])
        """
        if not VISUAL_PROCESS_AVAILABLE:
            raise ImportError("Visual_process 模块不可用，无法初始化 VisualIntegration")
        
        self.config = config or {}
        self.camera_intrinsics = camera_intrinsics or {
            'fx': 320.0, 'fy': 320.0,
            'cx': 320.0, 'cy': 240.0
        }
        
        # 初始化视觉感知中心
        self.visual_perception = VisualPerception(config=self.config)
        
        # 初始化点云处理器
        self.pointcloud_processor = PointCloudProcessor(
            PointCloudConfig(
                max_depth=self.config.get('max_depth', 20.0),
                min_depth=self.config.get('min_depth', 0.1),
                downsample_voxel_size=self.config.get('voxel_size', 0.05)
            )
        )
        
        # 初始化3D可视化器
        self.visualizer_3d: Optional[PointCloudVisualizer] = None
        if enable_3d_visualization:
            self.visualizer_3d = PointCloudVisualizer()
            self.visualizer_3d.start()
        
        # 历史记录
        self._motion_history: List[MotionMatrix] = []
        self._position_history: Dict[str, List[np.ndarray]] = {"drone_0": []}
        self._current_position = np.zeros(3, dtype=np.float32)
        self._current_rotation = np.eye(3, dtype=np.float32)
        
        # 无人机ID（单机模拟多机时用于区分）
        self._drone_counter = 0
        
        # 初始化ESP32接收器
        self.esp32_receiver: Optional[DualCameraReceiver] = None
        self._enable_esp32 = enable_esp32_receiver
        self._frame_callback = frame_callback
        self._latest_frames: Dict[str, np.ndarray] = {}
        
        if enable_esp32_receiver:
            self._init_esp32_receiver()
        
        print("VisualIntegration 初始化完成")
        if enable_3d_visualization:
            print("3D可视化已启用")
        if enable_esp32_receiver:
            print("ESP32图传接收已启用")
    
    def _init_esp32_receiver(self):
        """初始化ESP32接收器"""
        try:
            def front_callback(frame: CameraFrame):
                self._latest_frames['front'] = frame.image
                self._on_new_frame('front', frame.image)
            
            def down_callback(frame: CameraFrame):
                self._latest_frames['down'] = frame.image
                self._on_new_frame('down', frame.image)
            
            self.esp32_receiver = DualCameraReceiver(
                front_callback=front_callback,
                down_callback=down_callback
            )
            self.esp32_receiver.start()
            
        except Exception as e:
            print(f"ESP32接收器初始化失败: {e}")
            self._enable_esp32 = False
    
    def _on_new_frame(self, camera_name: str, frame: np.ndarray):
        """新帧回调"""
        if self._frame_callback:
            self._frame_callback({
                'camera': camera_name,
                'frame': frame,
                'timestamp': time.time()
            })
    
    def get_esp32_frames(self) -> Dict[str, np.ndarray]:
        """
        获取ESP32摄像头的最新帧
        
        Returns:
            {'front': 前视图像, 'down': 下视图像}
        """
        if self.esp32_receiver:
            return {
                'front': self.esp32_receiver.read('front'),
                'down': self.esp32_receiver.read('down')
            }
        return self._latest_frames
    
    def display_esp32_feed(self, window_name: str = "ESP32 Cameras"):
        """
        显示ESP32摄像头画面（调试用）
        
        Args:
            window_name: 窗口名称
        """
        import cv2
        
        if self.esp32_receiver:
            self.esp32_receiver.display(window_name)
        else:
            # 使用缓存帧显示
            frames = []
            for name in ['front', 'down']:
                if name in self._latest_frames:
                    frame = self._latest_frames[name].copy()
                    cv2.putText(frame, name, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frames.append(frame)
            
            if len(frames) == 2:
                combined = np.vstack(frames)
                cv2.imshow(window_name, combined)
    
    def process_frame(self, 
                     processing_result,
                     color_image: Optional[np.ndarray] = None,
                     timestamp: float = 0.0,
                     frame_idx: int = -1) -> Dict[str, Any]:
        """
        处理一帧数据
        
        Args:
            processing_result: DroneSwarmSystem 的 ProcessingResult
            color_image: 彩色图像（用于点云着色）
            timestamp: 时间戳
            frame_idx: 帧索引
            
        Returns:
            处理结果字典，包含：
            - visual_state: VisualState
            - point_cloud: Open3D 点云
            - motion_matrix: MotionMatrix
            - cluster_info: ClusterControlInfo
        """
        # 转换为 NeuralOutput
        neural_output = NeuralOutput.from_processing_result(
            processing_result, timestamp, frame_idx
        )
        
        # 视觉感知处理
        visual_state = self.visual_perception.process(
            neural_output, 
            intrinsics=self.camera_intrinsics
        )
        
        # 生成点云
        point_cloud = None
        if processing_result.depth_map is not None:
            point_cloud = self.pointcloud_processor.depth_to_pointcloud(
                processing_result.depth_map,
                self.camera_intrinsics,
                color_image
            )
            point_cloud = self.pointcloud_processor.filter_pointcloud(point_cloud)
            self.pointcloud_processor.add_to_history(point_cloud)
        
        # 提取运动矩阵
        motion_matrix = self._extract_motion_matrix(visual_state)
        self._update_position(motion_matrix)
        
        # 提取集群控制信息
        cluster_info = self._extract_cluster_control_info(visual_state)
        
        # 更新3D可视化
        if self.visualizer_3d is not None and point_cloud is not None:
            self._update_3d_visualization(point_cloud, motion_matrix)
        
        return {
            'visual_state': visual_state,
            'point_cloud': point_cloud,
            'motion_matrix': motion_matrix,
            'cluster_info': cluster_info,
            'timestamp': timestamp,
            'frame_idx': frame_idx
        }
    
    def _extract_motion_matrix(self, visual_state: 'VisualState') -> MotionMatrix:
        """从 VisualState 提取运动矩阵"""
        ego_motion = visual_state.ego_motion
        
        motion = MotionMatrix(
            rotation=ego_motion.rotation,
            translation=ego_motion.translation,
            transformation=ego_motion.transform,
            velocity=ego_motion.velocity,
            timestamp=visual_state.timestamp,
            confidence=ego_motion.confidence
        )
        
        self._motion_history.append(motion)
        if len(self._motion_history) > 100:
            self._motion_history.pop(0)
        
        return motion
    
    def _update_position(self, motion_matrix: MotionMatrix):
        """更新当前位置（通过累加位移）"""
        # 简化的位置更新：累加位移
        self._current_position += motion_matrix.translation
        self._current_rotation = motion_matrix.rotation @ self._current_rotation
        
        # 记录历史
        drone_id = f"drone_{self._drone_counter}"
        if drone_id not in self._position_history:
            self._position_history[drone_id] = []
        self._position_history[drone_id].append(self._current_position.copy())
    
    def _extract_cluster_control_info(self, visual_state: 'VisualState') -> ClusterControlInfo:
        """提取集群控制所需信息"""
        info = ClusterControlInfo(
            drone_id=f"drone_{self._drone_counter}",
            position=self._current_position.copy(),
            velocity=visual_state.ego_motion.velocity.copy()
        )
        
        # 提取障碍物信息
        if visual_state.obstacle_frame is not None:
            obstacle = visual_state.obstacle_frame
            info.nearest_obstacle_distance = obstacle.closest_depth
            info.nearest_obstacle_angle = obstacle.closest_angle
            
            # 提取周围障碍物
            angles = obstacle.angles
            depths = obstacle.depths
            for i in range(0, len(angles), 10):  # 每10个角度采样一个
                if depths[i] < self.config.get('max_depth', 20.0):
                    info.surrounding_obstacles.append(
                        (float(angles[i]), float(depths[i]))
                    )
        
        return info
    
    def _update_3d_visualization(self, 
                                 point_cloud: 'o3d.geometry.PointCloud',
                                 motion_matrix: MotionMatrix):
        """更新3D可视化"""
        if self.visualizer_3d is None:
            return
        
        # 更新点云
        self.visualizer_3d.update_point_cloud(point_cloud)
        
        # 更新无人机位姿
        drone_pose = DronePose(
            position=self._current_position,
            rotation=self._current_rotation,
            drone_id=f"drone_{self._drone_counter}"
        )
        self.visualizer_3d.update_drone_pose(drone_pose)
        
        # 绘制轨迹
        self.visualizer_3d.draw_trajectory(f"drone_{self._drone_counter}")
    
    def simulate_multiple_drones(self, 
                                 frame_results: List[Dict],
                                 positions: List[np.ndarray]) -> List[ClusterControlInfo]:
        """
        模拟多无人机（单机多位置拍摄）
        
        Args:
            frame_results: 多帧处理结果
            positions: 各帧对应的无人机位置
            
        Returns:
            各无人机的集群控制信息列表
        """
        cluster_infos = []
        
        for i, (result, position) in enumerate(zip(frame_results, positions)):
            self._drone_counter = i
            self._current_position = position
            
            # 更新集群信息
            info = ClusterControlInfo(
                drone_id=f"drone_{i}",
                position=position,
                velocity=result['cluster_info'].velocity,
                nearest_obstacle_distance=result['cluster_info'].nearest_obstacle_distance,
                nearest_obstacle_angle=result['cluster_info'].nearest_obstacle_angle
            )
            cluster_infos.append(info)
            
            # 在3D可视化中显示多无人机
            if self.visualizer_3d is not None and 'point_cloud' in result:
                # 变换点云到世界坐标系
                pcd = result['point_cloud']
                if pcd is not None:
                    pcd_transformed = pcd.translate(position)
                    
                    drone_pose = DronePose(
                        position=position,
                        rotation=np.eye(3),
                        drone_id=f"drone_{i}"
                    )
                    self.visualizer_3d.update_drone_pose(drone_pose)
        
        return cluster_infos
    
    def get_fused_pointcloud(self) -> Optional['o3d.geometry.PointCloud']:
        """获取融合的历史点云"""
        return self.pointcloud_processor.get_merged_history()
    
    def save_pointcloud(self, filename: str):
        """保存当前点云"""
        import open3d as o3d
        pcd = self.get_fused_pointcloud()
        if pcd is not None:
            o3d.io.write_point_cloud(filename, pcd)
            print(f"点云已保存: {filename}")
    
    def get_motion_trajectory(self) -> List[np.ndarray]:
        """获取运动轨迹"""
        drone_id = f"drone_{self._drone_counter}"
        return self._position_history.get(drone_id, [])
    
    def reset(self):
        """重置状态"""
        self._motion_history.clear()
        self._position_history.clear()
        self._current_position = np.zeros(3, dtype=np.float32)
        self._current_rotation = np.eye(3, dtype=np.float32)
        self._drone_counter = 0
        
        if self.pointcloud_processor is not None:
            self.pointcloud_processor._point_cloud_history.clear()
    
    def release(self):
        """释放资源"""
        if self.visualizer_3d is not None:
            self.visualizer_3d.stop()
            self.visualizer_3d = None
        
        if self.esp32_receiver is not None:
            self.esp32_receiver.stop()
            self.esp32_receiver = None
        
        print("VisualIntegration 资源已释放")


# 便捷函数
def create_visual_integration(enable_3d: bool = False,
                              intrinsics: Optional[Dict] = None,
                              enable_esp32: bool = False,
                              frame_callback: Optional[Callable] = None) -> Optional[VisualIntegration]:
    """
    创建视觉整合器的便捷函数
    
    Args:
        enable_3d: 是否启用3D可视化
        intrinsics: 相机内参
        enable_esp32: 是否启用ESP32图传接收
        frame_callback: 新帧回调函数
        
    Returns:
        VisualIntegration 实例，如果不可用则返回 None
    """
    if not VISUAL_PROCESS_AVAILABLE:
        print("Visual_process 模块不可用，无法创建 VisualIntegration")
        return None
    
    try:
        return VisualIntegration(
            enable_3d_visualization=enable_3d,
            camera_intrinsics=intrinsics,
            enable_esp32_receiver=enable_esp32,
            frame_callback=frame_callback
        )
    except Exception as e:
        print(f"创建 VisualIntegration 失败: {e}")
        return None


def create_esp32_receiver_only(frame_callback: Optional[Callable] = None):
    """
    仅创建ESP32接收器的便捷函数
    
    Args:
        frame_callback: 新帧回调函数
        
    Returns:
        DualCameraReceiver 实例，如果不可用则返回 None
    """
    if not VISUAL_PROCESS_AVAILABLE:
        print("Visual_process 模块不可用，无法创建 ESP32 接收器")
        return None
    
    try:
        from esp32_receiver import DualCameraReceiver
        receiver = DualCameraReceiver(
            front_callback=lambda f: frame_callback({'camera': 'front', 'frame': f.image, 'timestamp': f.timestamp}) if frame_callback else None,
            down_callback=lambda f: frame_callback({'camera': 'down', 'frame': f.image, 'timestamp': f.timestamp}) if frame_callback else None
        )
        receiver.start()
        return receiver
    except Exception as e:
        print(f"创建 ESP32 接收器失败: {e}")
        return None

"""
3D点云可视化模块

用于实时显示3D点云、无人机位置、障碍物等信息
"""

import numpy as np
import threading
import time
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
import warnings

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    # 创建占位符类型
    class MockO3D:
        class geometry:
            class PointCloud:
                pass
            class TriangleMesh:
                @staticmethod
                def create_coordinate_frame(size=1.0):
                    pass
                @staticmethod
                def create_sphere(radius=1.0):
                    pass
            class LineSet:
                pass
            class KDTreeSearchParamHybrid:
                def __init__(self, radius=1.0, max_nn=30):
                    pass
        class utility:
            @staticmethod
            def Vector3dVector(x):
                pass
            @staticmethod
            def Vector3iVector(x):
                pass
            @staticmethod
            def Vector2iVector(x):
                pass
        class visualization:
            class Visualizer:
                def create_window(self, **kwargs):
                    pass
                def add_geometry(self, geom, **kwargs):
                    pass
                def remove_geometry(self, geom, **kwargs):
                    pass
                def update_geometry(self, geom):
                    pass
                def poll_events(self):
                    pass
                def update_renderer(self):
                    pass
                def destroy_window(self):
                    pass
                def get_render_option(self):
                    return MockRenderOption()
                def reset_view_point(self, flag):
                    pass
                def capture_screen_image(self, filename):
                    pass
            class rendering:
                pass
        class pipelines:
            class registration:
                class TransformationEstimationPointToPlane:
                    pass
                @staticmethod
                def registration_icp(*args, **kwargs):
                    class Result:
                        transformation = np.eye(4)
                    return Result()
        class io:
            @staticmethod
            def write_point_cloud(filename, pcd):
                pass
    
    class MockRenderOption:
        background_color = [0, 0, 0]
        point_size = 1.0
    
    o3d = MockO3D()
    warnings.warn("Open3D 未安装，3D可视化功能将不可用")


@dataclass
class DronePose:
    """无人机位姿"""
    position: np.ndarray = None
    rotation: np.ndarray = None
    drone_id: str = "drone_0"
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.rotation is None:
            self.rotation = np.eye(3)


class PointCloudVisualizer:
    """
    3D点云可视化器
    
    功能：
    1. 实时显示3D点云
    2. 显示无人机位置和朝向
    3. 显示障碍物信息
    4. 支持多无人机（集群模拟）
    """
    
    def __init__(self, 
                 window_name: str = "Drone Swarm 3D Visualization",
                 width: int = 1280,
                 height: int = 720,
                 enable_ui: bool = True):
        """
        初始化3D可视化器
        
        Args:
            window_name: 窗口名称
            width: 窗口宽度
            height: 窗口高度
            enable_ui: 是否启用UI界面
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D 未安装，无法使用3D可视化功能")
        
        self.window_name = window_name
        self.width = width
        self.height = height
        self.enable_ui = enable_ui
        
        # 可视化对象
        self._point_cloud = None
        self._drone_meshes: Dict[str, o3d.geometry.TriangleMesh] = {}
        self._obstacle_geometries: List = []
        self._trajectory_lines = None
        
        # 轨迹历史
        self._trajectory_history: Dict[str, List[np.ndarray]] = {}
        self._max_trajectory_points = 1000
        
        # 可视化窗口
        self._vis = None
        self._vis_thread = None
        self._is_running = False
        
        # 更新锁
        self._lock = threading.Lock()
        
        print("PointCloudVisualizer 初始化完成")
    
    def create_drone_mesh(self, drone_id: str = "drone_0") -> o3d.geometry.TriangleMesh:
        """
        创建无人机3D模型（简化的四面体）
        
        Args:
            drone_id: 无人机ID
            
        Returns:
            无人机网格模型
        """
        # 创建简化的无人机模型（四面体）
        mesh = o3d.geometry.TriangleMesh()
        
        # 顶点：机身 + 四个旋翼臂
        size = 0.15  # 无人机尺寸（米）
        vertices = [
            [0, 0, 0],  # 中心
            [size, 0, 0],  # 前
            [-size, 0, 0],  # 后
            [0, size, 0],  # 左
            [0, -size, 0],  # 右
            [0, 0, size * 0.3],  # 上（机身）
        ]
        
        # 面
        triangles = [
            [0, 1, 5], [0, 5, 2], [0, 3, 5], [0, 5, 4],  # 机身
            [1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],  # 旋翼连接
        ]
        
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 计算法线
        mesh.compute_vertex_normals()
        
        # 设置颜色（根据无人机ID）
        colors = {
            "drone_0": [1.0, 0.0, 0.0],  # 红色
            "drone_1": [0.0, 1.0, 0.0],  # 绿色
            "drone_2": [0.0, 0.0, 1.0],  # 蓝色
            "drone_3": [1.0, 1.0, 0.0],  # 黄色
        }
        color = colors.get(drone_id, [0.5, 0.5, 0.5])
        mesh.paint_uniform_color(color)
        
        return mesh
    
    def start(self):
        """启动可视化窗口（在新线程中）"""
        if self._is_running:
            return
        
        self._is_running = True
        self._vis_thread = threading.Thread(target=self._visualization_loop)
        self._vis_thread.daemon = True
        self._vis_thread.start()
        
        print("3D可视化窗口已启动")
    
    def _visualization_loop(self):
        """可视化主循环"""
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(
            window_name=self.window_name,
            width=self.width,
            height=self.height
        )
        
        # 设置渲染选项
        render_option = self._vis.get_render_option()
        render_option.background_color = [0.1, 0.1, 0.1]  # 深灰色背景
        render_option.point_size = 2.0
        
        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self._vis.add_geometry(coord_frame)
        
        # 主循环
        while self._is_running:
            with self._lock:
                self._update_geometries()
            
            self._vis.poll_events()
            self._vis.update_renderer()
            time.sleep(0.033)  # ~30 FPS
        
        self._vis.destroy_window()
    
    def _update_geometries(self):
        """更新几何体"""
        if self._vis is None:
            return
        
        # 更新点云
        if self._point_cloud is not None:
            self._vis.update_geometry(self._point_cloud)
        
        # 更新无人机模型
        for drone_id, mesh in self._drone_meshes.items():
            self._vis.update_geometry(mesh)
    
    def stop(self):
        """停止可视化"""
        self._is_running = False
        if self._vis_thread is not None:
            self._vis_thread.join(timeout=2.0)
        print("3D可视化窗口已关闭")
    
    def update_point_cloud(self, pcd: o3d.geometry.PointCloud):
        """
        更新点云显示
        
        Args:
            pcd: 新的点云
        """
        with self._lock:
            if self._point_cloud is None:
                self._point_cloud = pcd
                if self._vis is not None:
                    self._vis.add_geometry(pcd)
            else:
                self._point_cloud.points = pcd.points
                if pcd.has_colors():
                    self._point_cloud.colors = pcd.colors
                if pcd.has_normals():
                    self._point_cloud.normals = pcd.normals
    
    def update_drone_pose(self, pose: DronePose):
        """
        更新无人机位姿
        
        Args:
            pose: 无人机位姿
        """
        with self._lock:
            drone_id = pose.drone_id
            
            # 创建或获取无人机模型
            if drone_id not in self._drone_meshes:
                mesh = self.create_drone_mesh(drone_id)
                self._drone_meshes[drone_id] = mesh
                if self._vis is not None:
                    self._vis.add_geometry(mesh)
            else:
                mesh = self._drone_meshes[drone_id]
            
            # 应用变换
            transform = np.eye(4)
            transform[:3, :3] = pose.rotation
            transform[:3, 3] = pose.position
            
            mesh.transform(transform)
            
            # 更新轨迹
            if drone_id not in self._trajectory_history:
                self._trajectory_history[drone_id] = []
            self._trajectory_history[drone_id].append(pose.position.copy())
            
            # 限制轨迹长度
            if len(self._trajectory_history[drone_id]) > self._max_trajectory_points:
                self._trajectory_history[drone_id].pop(0)
    
    def update_obstacles(self, 
                        obstacle_centers: np.ndarray,
                        obstacle_radii: Optional[np.ndarray] = None,
                        colors: Optional[np.ndarray] = None):
        """
        更新障碍物显示
        
        Args:
            obstacle_centers: 障碍物中心位置 [N, 3]
            obstacle_radii: 障碍物半径 [N]
            colors: 障碍物颜色 [N, 3]
        """
        with self._lock:
            # 清除旧障碍物
            for geom in self._obstacle_geometries:
                if self._vis is not None:
                    self._vis.remove_geometry(geom, reset_bounding_box=False)
            self._obstacle_geometries.clear()
            
            # 创建新障碍物
            if obstacle_radii is None:
                obstacle_radii = np.full(len(obstacle_centers), 0.2)
            
            for i, (center, radius) in enumerate(zip(obstacle_centers, obstacle_radii)):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(center)
                
                if colors is not None and i < len(colors):
                    sphere.paint_uniform_color(colors[i])
                else:
                    sphere.paint_uniform_color([1.0, 0.5, 0.0])  # 橙色
                
                sphere.compute_vertex_normals()
                self._obstacle_geometries.append(sphere)
                
                if self._vis is not None:
                    self._vis.add_geometry(sphere, reset_bounding_box=False)
    
    def draw_trajectory(self, drone_id: str = "drone_0"):
        """
        绘制无人机轨迹
        
        Args:
            drone_id: 无人机ID
        """
        with self._lock:
            if drone_id not in self._trajectory_history:
                return
            
            points = self._trajectory_history[drone_id]
            if len(points) < 2:
                return
            
            # 创建线段
            lines = [[i, i + 1] for i in range(len(points) - 1)]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.0, 1.0, 1.0])  # 青色
            
            if self._trajectory_lines is not None and self._vis is not None:
                self._vis.remove_geometry(self._trajectory_lines, reset_bounding_box=False)
            
            self._trajectory_lines = line_set
            if self._vis is not None:
                self._vis.add_geometry(line_set, reset_bounding_box=False)
    
    def reset_view(self):
        """重置视角"""
        if self._vis is not None:
            self._vis.reset_view_point(True)
    
    def save_screenshot(self, filename: str):
        """
        保存截图
        
        Args:
            filename: 保存路径
        """
        if self._vis is not None:
            self._vis.capture_screen_image(filename)
            print(f"截图已保存: {filename}")
    
    def save_point_cloud(self, filename: str):
        """
        保存当前点云
        
        Args:
            filename: 保存路径（.ply格式）
        """
        if self._point_cloud is not None:
            o3d.io.write_point_cloud(filename, self._point_cloud)
            print(f"点云已保存: {filename}")


def create_simple_visualization(pcd: o3d.geometry.PointCloud,
                                drone_poses: Optional[List[DronePose]] = None,
                                blocking: bool = True):
    """
    简单的点云可视化函数（非交互式）
    
    Args:
        pcd: 点云
        drone_poses: 无人机位姿列表
        blocking: 是否阻塞等待窗口关闭
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D 未安装，无法显示3D可视化")
        return
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualization", width=1024, height=768)
    
    # 添加点云
    vis.add_geometry(pcd)
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)
    
    # 添加无人机
    if drone_poses:
        for pose in drone_poses:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            transform = np.eye(4)
            transform[:3, :3] = pose.rotation
            transform[:3, 3] = pose.position
            mesh.transform(transform)
            vis.add_geometry(mesh)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = [0.1, 0.1, 0.1]
    render_option.point_size = 2.0
    
    vis.run()
    
    if blocking:
        vis.destroy_window()
    else:
        return vis

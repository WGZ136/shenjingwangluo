"""
点云处理模块

将深度图和特征点转换为3D点云，支持数字孪生建模
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
import warnings

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    # 创建占位符类型用于类型注解
    class MockO3D:
        class geometry:
            class PointCloud:
                pass
        class utility:
            @staticmethod
            def Vector3dVector(x):
                pass
    o3d = MockO3D()
    warnings.warn("Open3D 未安装，点云功能将不可用。请运行: pip install open3d")


@dataclass
class PointCloudConfig:
    """点云处理配置"""
    max_depth: float = 20.0
    min_depth: float = 0.1
    downsample_voxel_size: float = 0.05
    remove_outliers: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    estimate_normals: bool = True
    normal_radius: float = 0.1
    normal_max_nn: int = 30


class PointCloudProcessor:
    """
    点云处理器
    
    功能：
    1. 将深度图转换为3D点云
    2. 特征点3D重建
    3. 点云滤波和预处理
    4. 多帧点云融合
    """
    
    def __init__(self, config: Optional[PointCloudConfig] = None):
        self.config = config or PointCloudConfig()
        
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D 未安装，无法使用点云功能")
        
        self._point_cloud_history: List = []
        self._max_history_size = 10
        
        print("PointCloudProcessor 初始化完成")
    
    def depth_to_pointcloud(self,
                           depth_map: np.ndarray,
                           intrinsics: Dict[str, float],
                           color_image: Optional[np.ndarray] = None):
        """
        将深度图转换为3D点云
        
        Args:
            depth_map: 深度图 [H, W]，单位为米
            intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
            color_image: 可选的彩色图像 [H, W, 3] (RGB)
            
        Returns:
            Open3D 点云对象
        """
        h, w = depth_map.shape
        fx = intrinsics.get('fx', 320.0)
        fy = intrinsics.get('fy', 320.0)
        cx = intrinsics.get('cx', w / 2.0)
        cy = intrinsics.get('cy', h / 2.0)
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 根据深度图过滤有效深度
        valid_mask = (depth_map > self.config.min_depth) & (depth_map < self.config.max_depth)
        
        # 反投影到3D空间
        z = depth_map[valid_mask]
        x = (u[valid_mask] - cx) * z / fx
        y = (v[valid_mask] - cy) * z / fy
        
        # 创建点云
        points = np.stack([x, y, z], axis=-1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 添加颜色信息（如果有）
        if color_image is not None:
            colors = color_image[valid_mask] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def features_to_pointcloud(self,
                               points_2d: np.ndarray,
                               depths: np.ndarray,
                               intrinsics: Dict[str, float],
                               colors: Optional[np.ndarray] = None):
        """
        将2D特征点和深度转换为3D点云
        
        Args:
            points_2d: 2D特征点坐标 [N, 2] (u, v)
            depths: 对应深度值 [N]
            intrinsics: 相机内参
            colors: 可选的颜色信息 [N, 3]
            
        Returns:
            Open3D 点云对象
        """
        fx = intrinsics.get('fx', 320.0)
        fy = intrinsics.get('fy', 320.0)
        cx = intrinsics.get('cx', 320.0)
        cy = intrinsics.get('cy', 240.0)
        
        # 过滤有效深度
        valid_mask = (depths > self.config.min_depth) & (depths < self.config.max_depth)
        points_2d_valid = points_2d[valid_mask]
        depths_valid = depths[valid_mask]
        
        u = points_2d_valid[:, 0]
        v = points_2d_valid[:, 1]
        z = depths_valid
        
        # 反投影
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points_3d = np.stack([x, y, z], axis=-1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        if colors is not None and len(colors) == len(points_2d):
            colors_valid = colors[valid_mask] / 255.0 if colors.max() > 1 else colors[valid_mask]
            pcd.colors = o3d.utility.Vector3dVector(colors_valid)
        
        return pcd
    
    def filter_pointcloud(self, pcd):
        """
        对点云进行滤波处理
        
        Args:
            pcd: 输入点云
            
        Returns:
            滤波后的点云
        """
        # 下采样
        if self.config.downsample_voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.config.downsample_voxel_size)
        
        # 去除离群点
        if self.config.remove_outliers and len(pcd.points) > self.config.outlier_nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio
            )
        
        # 估计法线
        if self.config.estimate_normals and len(pcd.points) > 0:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius,
                    max_nn=self.config.normal_max_nn
                )
            )
        
        return pcd
    
    def merge_pointclouds(self, pointclouds: List):
        """
        合并多个点云
        
        Args:
            pointclouds: 点云列表
            
        Returns:
            合并后的点云
        """
        if not pointclouds:
            return o3d.geometry.PointCloud()
        
        merged = o3d.geometry.PointCloud()
        for pcd in pointclouds:
            merged += pcd
        
        return self.filter_pointcloud(merged)
    
    def register_pointclouds(self,
                            source,
                            target,
                            initial_transform: Optional[np.ndarray] = None) -> Tuple:
        """
        点云配准（ICP算法）
        
        Args:
            source: 源点云
            target: 目标点云
            initial_transform: 初始变换矩阵 [4, 4]
            
        Returns:
            (配准后的点云, 变换矩阵)
        """
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # ICP配准
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=0.05,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        # 变换源点云
        source_transformed = source.transform(result.transformation)
        
        return source_transformed, result.transformation
    
    def add_to_history(self, pcd):
        """添加点云到历史记录"""
        self._point_cloud_history.append(pcd)
        if len(self._point_cloud_history) > self._max_history_size:
            self._point_cloud_history.pop(0)
    
    def get_merged_history(self):
        """获取合并的历史点云"""
        return self.merge_pointclouds(self._point_cloud_history)
    
    def estimate_ground_plane(self, 
                             pcd,
                             distance_threshold: float = 0.1) -> Tuple:
        """
        估计地面平面（RANSAC）
        
        Args:
            pcd: 输入点云
            distance_threshold: RANSAC距离阈值
            
        Returns:
            (非地面点云, 平面参数 [a, b, c, d])
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        # 提取非地面点
        non_ground = pcd.select_by_index(inliers, invert=True)
        
        return non_ground, plane_model

"""
相机工具模块 - 处理相机标定、坐标转换和相机模型
"""

import numpy as np
import cv2
import yaml
from typing import Tuple, Optional, Union
import json

class Camera:
    """
    相机类，封装相机内参、外参和基本操作
    """
    def __init__(self, intrinsic_matrix=None, distortion_coeffs=None, 
                 image_size=None, camera_name="default"):
        """
        初始化相机
        
        参数:
            intrinsic_matrix: 3x3 内参矩阵
            distortion_coeffs: 畸变系数 [k1, k2, p1, p2, k3]
            image_size: 图像尺寸 (width, height)
            camera_name: 相机名称
        """
        self.intrinsic = intrinsic_matrix
        self.distortion = distortion_coeffs
        self.image_size = image_size
        self.name = camera_name
        
        # 如果没有提供内参，使用默认值（假设标准相机）
        if self.intrinsic is None:
            self.intrinsic = np.array([
                [600.0, 0.0, 320.0],
                [0.0, 600.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        
        if self.distortion is None:
            self.distortion = np.zeros(5, dtype=np.float32)
        
        if self.image_size is None:
            self.image_size = (640, 480)
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        从YAML配置文件加载相机参数
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        camera_config = config.get('camera', {})
        intrinsic_config = camera_config.get('intrinsic', {})
        
        intrinsic = np.array([
            [intrinsic_config.get('fx', 600.0), 0.0, intrinsic_config.get('cx', 320.0)],
            [0.0, intrinsic_config.get('fy', 600.0), intrinsic_config.get('cy', 240.0)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        distortion = np.array(camera_config.get('distortion', [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        image_size = camera_config.get('resolution', [640, 480])
        
        return cls(intrinsic, distortion, image_size, camera_config.get('name', 'from_yaml'))
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        从字典加载相机参数
        """
        intrinsic = np.array(config_dict.get('intrinsic_matrix', [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
        ]), dtype=np.float32)
        
        distortion = np.array(config_dict.get('distortion_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        image_size = tuple(config_dict.get('image_size', (640, 480)))
        
        return cls(intrinsic, distortion, image_size, config_dict.get('name', 'from_dict'))
    
    def project_3d_to_2d(self, points_3d: np.ndarray, 
                          extrinsic_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        将3D点投影到2D图像平面
        
        参数:
            points_3d: 3D点坐标 (N, 3) 或 (N, 4) 齐次坐标
            extrinsic_matrix: 4x4 外参矩阵（世界到相机）
            
        返回:
            points_2d: 2D图像坐标 (N, 2)
        """
        if points_3d.shape[1] == 3:
            # 转换为齐次坐标
            points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        elif points_3d.shape[1] == 4:
            points_3d_homo = points_3d
        else:
            raise ValueError("points_3d must have shape (N, 3) or (N, 4)")
        
        if extrinsic_matrix is not None:
            # 应用外参变换：世界坐标 -> 相机坐标
            points_camera_homo = (extrinsic_matrix @ points_3d_homo.T).T
        else:
            points_camera_homo = points_3d_homo
        
        # 投影到图像平面
        points_image_homo = (self.intrinsic @ points_camera_homo[:, :3].T).T
        
        # 转换为非齐次坐标
        points_2d = points_image_homo[:, :2] / points_image_homo[:, 2, np.newaxis]
        
        # 应用畸变校正（如果需要）
        if np.any(self.distortion != 0):
            points_2d = self.undistort_points(points_2d)
        
        return points_2d
    
    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        校正图像点的畸变
        """
        points_2d = points_2d.reshape(-1, 1, 2).astype(np.float32)
        undistorted_points = cv2.undistortPoints(
            points_2d, self.intrinsic, self.distortion, 
            None, self.intrinsic
        )
        return undistorted_points.reshape(-1, 2)
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        校正图像畸变
        """
        return cv2.undistort(image, self.intrinsic, self.distortion)
    
    def back_project_2d_to_3d(self, points_2d: np.ndarray, depths: Union[float, np.ndarray]) -> np.ndarray:
        """
        将2D图像点反投影到3D空间（相机坐标系）
        
        参数:
            points_2d: 2D图像坐标 (N, 2)
            depths: 深度值，可以是标量或与points_2d同长度的数组
            
        返回:
            points_3d: 3D相机坐标 (N, 3)
        """
        # 将像素坐标转换为归一化相机坐标
        intrinsic_inv = np.linalg.inv(self.intrinsic)
        
        if points_2d.shape[1] == 2:
            # 转换为齐次坐标
            points_2d_homo = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])
        else:
            points_2d_homo = points_2d
        
        # 转换为归一化相机坐标
        points_normalized = (intrinsic_inv @ points_2d_homo.T).T
        
        # 乘以深度
        if np.isscalar(depths):
            points_3d = points_normalized * depths
        else:
            points_3d = points_normalized * depths.reshape(-1, 1)
        
        return points_3d[:, :3]
    
    def get_fov(self) -> Tuple[float, float]:
        """
        计算水平和垂直视场角（FOV）
        
        返回:
            hfov, vfov: 水平和垂直视场角（度）
        """
        fx = self.intrinsic[0, 0]
        fy = self.intrinsic[1, 1]
        width, height = self.image_size
        
        hfov = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
        vfov = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
        
        return hfov, vfov
    
    def save_to_yaml(self, yaml_path: str):
        """
        保存相机参数到YAML文件
        """
        config = {
            'camera': {
                'name': self.name,
                'intrinsic': {
                    'fx': float(self.intrinsic[0, 0]),
                    'fy': float(self.intrinsic[1, 1]),
                    'cx': float(self.intrinsic[0, 2]),
                    'cy': float(self.intrinsic[1, 2])
                },
                'distortion': self.distortion.tolist(),
                'resolution': list(self.image_size)
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def __repr__(self):
        return f"Camera(name={self.name}, size={self.image_size}, fx={self.intrinsic[0,0]:.1f})"


def estimate_camera_pose_from_points(points_3d: np.ndarray, points_2d: np.ndarray, 
                                     camera_matrix: np.ndarray, 
                                     dist_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用PnP算法估计相机位姿
    
    参数:
        points_3d: 3D世界点 (N, 3)
        points_2d: 2D图像点 (N, 2)
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        
    返回:
        success: 是否成功
        rvec: 旋转向量
        tvec: 平移向量
        inliers: 内点索引
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)
    
    # 转换为OpenCV需要的格式
    points_3d = points_3d.astype(np.float32)
    points_2d = points_2d.astype(np.float32).reshape(-1, 1, 2)
    
    # 使用RANSAC的PnP算法
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, camera_matrix, dist_coeffs,
        iterationsCount=100, reprojectionError=8.0, confidence=0.99
    )
    
    if success:
        inliers = inliers.flatten()
        return True, rvec, tvec, inliers
    else:
        return False, None, None, None


def rotation_matrix_to_euler(r_matrix: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为欧拉角（ZYX顺序，弧度）
    """
    sy = np.sqrt(r_matrix[0, 0] ** 2 + r_matrix[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(r_matrix[2, 1], r_matrix[2, 2])
        y = np.arctan2(-r_matrix[2, 0], sy)
        z = np.arctan2(r_matrix[1, 0], r_matrix[0, 0])
    else:
        x = np.arctan2(-r_matrix[1, 2], r_matrix[1, 1])
        y = np.arctan2(-r_matrix[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])


def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """
    将欧拉角（ZYX顺序，弧度）转换为旋转矩阵
    """
    x, y, z = euler_angles
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx


def create_default_camera():
    """
    创建默认相机（适用于测试）
    """
    return Camera()


# 测试代码
if __name__ == "__main__":
    # 创建默认相机
    cam = create_default_camera()
    print(f"相机: {cam}")
    
    # 测试投影
    points_3d = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]], dtype=np.float32)
    points_2d = cam.project_3d_to_2d(points_3d)
    print(f"3D点: {points_3d}")
    print(f"投影到2D: {points_2d}")
    
    # 测试反投影
    depths = np.array([5, 5, 5])
    points_3d_reconstructed = cam.back_project_2d_to_3d(points_2d, depths)
    print(f"反投影3D点: {points_3d_reconstructed}")
    
    # 计算FOV
    hfov, vfov = cam.get_fov()
    print(f"水平FOV: {hfov:.1f}°, 垂直FOV: {vfov:.1f}°")
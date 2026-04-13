"""
几何处理工具
3D投影、位姿估计等
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import cv2

class GeometryProcessor:
    """几何处理器"""
    
    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        初始化几何处理器
        
        Args:
            camera_matrix: 相机内参矩阵 [3, 3]
            dist_coeffs: 畸变系数
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
    def project_2d_to_3d(self, points_2d, depth_map):
        """
        将2D像素坐标反投影到3D相机坐标系
        
        Args:
            points_2d: 2D像素坐标 [N, 2]
            depth_map: 深度图 [H, W]
            
        Returns:
            points_3d: 3D点坐标 [N, 3]
        """
        N = points_2d.shape[0]
        points_3d = np.zeros((N, 3))
        
        # 相机内参逆矩阵
        K_inv = np.linalg.inv(self.camera_matrix)
        
        for i in range(N):
            x, y = points_2d[i]
            
            # 转换为整数索引
            x_int = int(np.clip(x, 0, depth_map.shape[1]-1))
            y_int = int(np.clip(y, 0, depth_map.shape[0]-1))
            
            # 获取深度值
            depth = depth_map[y_int, x_int]
            
            # 齐次像素坐标
            pixel_homogeneous = np.array([x, y, 1.0])
            
            # 反投影: P_3d = depth * (K^{-1} * [u, v, 1]^T)
            point_3d = depth * (K_inv @ pixel_homogeneous)
            points_3d[i] = point_3d
        
        return points_3d
    
    def project_3d_to_2d(self, points_3d):
        """
        将3D相机坐标投影到2D像素坐标
        
        Args:
            points_3d: 3D点坐标 [N, 3]
            
        Returns:
            points_2d: 2D像素坐标 [N, 2]
        """
        # 投影: [u, v, 1]^T = K * [X/Z, Y/Z, 1]^T
        points_homogeneous = points_3d / points_3d[:, 2:3]
        points_2d_homogeneous = (self.camera_matrix @ points_homogeneous.T).T
        
        # 去除齐次坐标
        points_2d = points_2d_homogeneous[:, :2]
        
        return points_2d
    
    def estimate_rigid_transform(self, points_src, points_dst):
        """
        估计刚体变换 (R, t)
        使用SVD方法求解最小二乘问题
        
        Args:
            points_src: 源点云 [N, 3]
            points_dst: 目标点云 [N, 3]
            
        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
            rmse: 均方根误差
        """
        # 检查输入
        if len(points_src) < 3 or len(points_dst) < 3:
            return None, None, float('inf')
        
        # 计算质心
        centroid_src = np.mean(points_src, axis=0)
        centroid_dst = np.mean(points_dst, axis=0)
        
        # 去中心化
        src_centered = points_src - centroid_src
        dst_centered = points_dst - centroid_dst
        
        # 计算协方差矩阵
        H = src_centered.T @ dst_centered
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R_mat = Vt.T @ U.T
        
        # 确保右手系 (det(R) = 1)
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
        
        # 计算平移向量
        t_vec = centroid_dst - R_mat @ centroid_src
        
        # 计算误差
        transformed = (R_mat @ points_src.T).T + t_vec
        errors = np.linalg.norm(transformed - points_dst, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        return R_mat, t_vec, rmse
    
    def estimate_pose_ransac(self, points_src, points_dst, 
                            max_iterations=1000, threshold=0.01):
        """
        使用RANSAC估计刚体变换
        去除异常点影响
        
        Args:
            points_src: 源点云 [N, 3]
            points_dst: 目标点云 [N, 3]
            max_iterations: 最大迭代次数
            threshold: 内点阈值
            
        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
            inlier_mask: 内点掩码 [N]
        """
        n_points = len(points_src)
        best_inliers = 0
        best_R = None
        best_t = None
        best_inlier_mask = None
        
        for _ in range(max_iterations):
            # 随机选择3个点
            indices = np.random.choice(n_points, 3, replace=False)
            
            # 用这3个点估计变换
            R, t, _ = self.estimate_rigid_transform(
                points_src[indices], points_dst[indices]
            )
            
            if R is None:
                continue
            
            # 计算所有点的误差
            transformed = (R @ points_src.T).T + t
            errors = np.linalg.norm(transformed - points_dst, axis=1)
            
            # 统计内点
            inlier_mask = errors < threshold
            n_inliers = np.sum(inlier_mask)
            
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_R, best_t = R, t
                best_inlier_mask = inlier_mask
        
        # 使用所有内点重新估计变换
        if best_inliers >= 3:
            R_refined, t_refined, rmse = self.estimate_rigid_transform(
                points_src[best_inlier_mask], 
                points_dst[best_inlier_mask]
            )
            
            if R_refined is not None:
                best_R, best_t = R_refined, t_refined
        
        return best_R, best_t, best_inlier_mask
    
    def rotation_matrix_to_euler(self, rotation_matrix):
        """旋转矩阵转换为欧拉角 (ZYX顺序)"""
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('zyx', degrees=True)
        return euler_angles
    
    def euler_to_rotation_matrix(self, euler_angles):
        """欧拉角转换为旋转矩阵 (ZYX顺序)"""
        rotation = R.from_euler('zyx', euler_angles, degrees=True)
        return rotation.as_matrix()
    
    def calculate_relative_position(self, R1, t1, R2, t2):
        """
        计算两个位姿之间的相对位置
        
        Args:
            R1, t1: 第一个位姿
            R2, t2: 第二个位姿
            
        Returns:
            relative_R: 相对旋转
            relative_t: 相对平移
            distance: 距离
        """
        # 相对变换: T_relative = T2 * T1^{-1}
        R1_inv = R1.T
        t1_inv = -R1_inv @ t1
        
        relative_R = R2 @ R1_inv
        relative_t = R2 @ t1_inv + t2
        
        # 计算距离
        distance = np.linalg.norm(relative_t)
        
        return relative_R, relative_t, distance
"""
位姿估计模块
使用RANSAC和SVD估计刚体变换
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import cv2

class PoseEstimator:
    """
    位姿估计器
    用于估计刚体变换（旋转矩阵和平移向量）
    """
    
    def __init__(self, config=None):
        """
        初始化位姿估计器
        
        Args:
            config: 配置字典
        """
        self.config = self._load_config(config)
        print("位姿估计器初始化完成")
    
    def _load_config(self, config):
        """加载配置"""
        default_config = {
            'method': 'ransac',  # 'svd', 'ransac', 'teaser', 'umeyama'
            'ransac_threshold': 0.01,
            'ransac_iterations': 1000,
            'min_inlier_ratio': 0.5,
            'min_points': 3,
            'use_scale': False,  # 是否估计尺度（对于非刚体变换）
            'verbose': True
        }
        
        if config is None:
            return default_config
        elif isinstance(config, dict):
            default_config.update(config)
            return default_config
        else:
            raise ValueError("配置必须是字典或None")
    
    def estimate_transform(self, points_src, points_dst, method=None):
        """
        估计两点集之间的变换
        
        Args:
            points_src: 源点集 [N, 3]
            points_dst: 目标点集 [N, 3]
            method: 估计方法（覆盖配置）
            
        Returns:
            R: 旋转矩阵 [3, 3] (None如果失败)
            t: 平移向量 [3] (None如果失败)
            s: 尺度因子 (None如果不估计尺度)
            inlier_mask: 内点掩码 [N] (None如果不使用RANSAC)
            info: 附加信息字典
        """
        if method is None:
            method = self.config['method']
        
        if len(points_src) < self.config['min_points']:
            if self.config.get('verbose', True):
                print(f"警告: 点数不足 ({len(points_src)} < {self.config['min_points']})")
            return None, None, None, None, {}
        
        # 选择估计方法
        if method == 'svd':
            return self._estimate_svd(points_src, points_dst)
        elif method == 'ransac':
            return self._estimate_ransac(points_src, points_dst)
        elif method == 'teaser':
            return self._estimate_teaser(points_src, points_dst)
        elif method == 'umeyama':
            return self._estimate_umeyama(points_src, points_dst)
        else:
            raise ValueError(f"不支持的估计方法: {method}")
    
    def _estimate_svd(self, points_src, points_dst):
        """
        使用SVD估计刚体变换（最小二乘解）
        
        Args:
            points_src: 源点集 [N, 3]
            points_dst: 目标点集 [N, 3]
            
        Returns:
            R: 旋转矩阵
            t: 平移向量
            s: 尺度因子 (始终为1.0)
            inlier_mask: None (所有点都视为内点)
            info: 附加信息
        """
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
        
        # 确保是右手系旋转（det(R) = 1）
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
        
        # 计算平移向量
        t_vec = centroid_dst - R_mat @ centroid_src
        
        # 计算误差
        transformed = (R_mat @ points_src.T).T + t_vec
        errors = np.linalg.norm(transformed - points_dst, axis=1)
        
        info = {
            'method': 'svd',
            'n_points': len(points_src),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors),
            'centroid_src': centroid_src,
            'centroid_dst': centroid_dst,
            'singular_values': S
        }
        
        return R_mat, t_vec, 1.0, None, info
    
    def _estimate_ransac(self, points_src, points_dst):
        """
        使用RANSAC估计刚体变换
        """
        n_points = len(points_src)
        min_points = 3  # 最小样本数
        
        best_R = None
        best_t = None
        best_inliers = 0
        best_inlier_mask = None
        best_error = float('inf')
        
        threshold = self.config['ransac_threshold']
        max_iterations = self.config['ransac_iterations']
        
        for _ in range(max_iterations):
            # 随机选择最小样本点
            indices = np.random.choice(n_points, min_points, replace=False)
            
            # 使用这些点计算初始变换
            R_candidate, t_candidate, _, _, _ = self._estimate_svd(
                points_src[indices], points_dst[indices]
            )
            
            if R_candidate is None:
                continue
            
            # 计算所有点的误差
            transformed = (R_candidate @ points_src.T).T + t_candidate
            errors = np.linalg.norm(transformed - points_dst, axis=1)
            
            # 统计内点
            inlier_mask = errors < threshold
            n_inliers = np.sum(inlier_mask)
            
            # 如果内点更多，或者内点相同但误差更小
            if n_inliers > best_inliers or \
               (n_inliers == best_inliers and np.mean(errors[inlier_mask]) < best_error):
                best_inliers = n_inliers
                best_error = np.mean(errors[inlier_mask]) if n_inliers > 0 else float('inf')
                best_R = R_candidate
                best_t = t_candidate
                best_inlier_mask = inlier_mask
        
        # 如果找到了足够的内点，使用所有内点重新估计
        min_inlier_ratio = self.config['min_inlier_ratio']
        if best_R is not None and best_inliers >= max(min_points, min_inlier_ratio * n_points):
            inlier_src = points_src[best_inlier_mask]
            inlier_dst = points_dst[best_inlier_mask]
            
            best_R, best_t, _, _, svd_info = self._estimate_svd(inlier_src, inlier_dst)
            
            # 计算最终误差
            transformed = (best_R @ points_src.T).T + best_t
            errors = np.linalg.norm(transformed - points_dst, axis=1)
            
            info = {
                'method': 'ransac',
                'n_points': n_points,
                'n_inliers': best_inliers,
                'inlier_ratio': best_inliers / n_points,
                'rmse': np.sqrt(np.mean(errors**2)),
                'max_error': np.max(errors),
                'mean_error': np.mean(errors),
                'ransac_threshold': threshold,
                'ransac_iterations': max_iterations
            }
            info.update(svd_info)
            
            return best_R, best_t, 1.0, best_inlier_mask, info
        
        return None, None, None, None, {'method': 'ransac', 'error': 'insufficient_inliers'}
    
    def _estimate_teaser(self, points_src, points_dst):
        """
        使用TEASER++算法估计变换（鲁棒性更好）
        简化实现，完整TEASER++更复杂
        """
        # 这里实现一个简化的TEASER++版本
        # 实际应用中建议使用完整的TEASER++库
        
        # 使用RANSAC获取初始内点
        R_init, t_init, _, inlier_mask, _ = self._estimate_ransac(points_src, points_dst)
        
        if R_init is None:
            return None, None, None, None, {'method': 'teaser', 'error': 'ransac_failed'}
        
        # 使用内点进行精确估计
        inlier_src = points_src[inlier_mask]
        inlier_dst = points_dst[inlier_mask]
        
        # 再次使用SVD进行精确估计
        R_final, t_final, _, _, svd_info = self._estimate_svd(inlier_src, inlier_dst)
        
        # 计算最终误差
        transformed = (R_final @ points_src.T).T + t_final
        errors = np.linalg.norm(transformed - points_dst, axis=1)
        
        info = {
            'method': 'teaser',
            'n_points': len(points_src),
            'n_inliers': np.sum(inlier_mask),
            'inlier_ratio': np.sum(inlier_mask) / len(points_src),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors)
        }
        info.update(svd_info)
        
        return R_final, t_final, 1.0, inlier_mask, info
    
    def _estimate_umeyama(self, points_src, points_dst):
        """
        使用Umeyama算法估计相似变换（包括尺度）
        S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns"
        """
        n = len(points_src)
        
        # 计算质心
        mu_src = np.mean(points_src, axis=0)
        mu_dst = np.mean(points_dst, axis=0)
        
        # 去中心化
        src_centered = points_src - mu_src
        dst_centered = points_dst - mu_dst
        
        # 计算协方差矩阵
        sigma_src = np.sum(np.square(src_centered)) / n
        sigma_dst = np.sum(np.square(dst_centered)) / n
        
        cov_matrix = dst_centered.T @ src_centered / n
        
        # SVD分解
        U, D, Vt = np.linalg.svd(cov_matrix)
        
        # 计算旋转矩阵
        S = np.eye(3)
        if np.linalg.det(cov_matrix) < 0:
            S[2, 2] = -1
        
        R_mat = U @ S @ Vt
        
        # 计算尺度
        scale = np.trace(np.diag(D) @ S) / sigma_src
        
        # 计算平移
        t_vec = mu_dst - scale * R_mat @ mu_src
        
        # 计算误差
        transformed = scale * (R_mat @ points_src.T).T + t_vec
        errors = np.linalg.norm(transformed - points_dst, axis=1)
        
        info = {
            'method': 'umeyama',
            'n_points': n,
            'scale': scale,
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors),
            'mu_src': mu_src,
            'mu_dst': mu_dst,
            'sigma_src': sigma_src,
            'sigma_dst': sigma_dst
        }
        
        return R_mat, t_vec, scale, None, info
    
    def compute_velocity(self, translation, delta_time):
        """
        根据平移向量计算速度
        
        Args:
            translation: 平移向量 [3]
            delta_time: 时间间隔 (秒)
            
        Returns:
            velocity: 速度向量 [3] (米/秒)
        """
        if delta_time <= 0:
            raise ValueError("时间间隔必须大于0")
        
        return translation / delta_time
    
    def compute_angular_velocity(self, rotation_matrix1, rotation_matrix2, delta_time):
        """
        根据旋转矩阵计算角速度
        
        Args:
            rotation_matrix1: t时刻旋转矩阵
            rotation_matrix2: t+1时刻旋转矩阵
            delta_time: 时间间隔
            
        Returns:
            angular_velocity: 角速度向量 [3] (弧度/秒)
        """
        # 计算相对旋转
        R_rel = rotation_matrix2 @ rotation_matrix1.T
        
        # 将旋转矩阵转换为轴角表示
        rotation = R.from_matrix(R_rel)
        axis_angle = rotation.as_rotvec()  # 旋转向量 (方向=轴, 大小=角度)
        
        # 计算角速度
        angular_velocity = axis_angle / delta_time
        
        return angular_velocity
    
    def rotation_matrix_to_euler(self, rotation_matrix, degrees=True):
        """
        旋转矩阵转换为欧拉角
        
        Args:
            rotation_matrix: 旋转矩阵 [3, 3]
            degrees: 是否返回角度制
            
        Returns:
            euler_angles: 欧拉角 (roll, pitch, yaw)
        """
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=degrees)
        
        return euler_angles
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw, degrees=True):
        """
        欧拉角转换为旋转矩阵
        
        Args:
            roll, pitch, yaw: 欧拉角
            degrees: 输入是否为角度制
            
        Returns:
            rotation_matrix: 旋转矩阵 [3, 3]
        """
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
        return rotation.as_matrix()
    
    def quaternion_to_rotation_matrix(self, quaternion):
        """
        四元数转换为旋转矩阵
        
        Args:
            quaternion: 四元数 [w, x, y, z] 或 [x, y, z, w]
            
        Returns:
            rotation_matrix: 旋转矩阵
        """
        quaternion = np.asarray(quaternion)
        if quaternion.shape == (4,):
            # 假设是 [x, y, z, w]
            rotation = R.from_quat(quaternion)
        elif quaternion.shape == (4, 1) or quaternion.shape == (1, 4):
            # 展平
            quaternion = quaternion.flatten()
            rotation = R.from_quat(quaternion)
        else:
            raise ValueError(f"四元数形状错误: {quaternion.shape}")
        
        return rotation.as_matrix()
    
    def compute_relative_pose(self, pose1, pose2):
        """
        计算两个位姿之间的相对位姿
        
        Args:
            pose1: 第一个位姿 (R1, t1)
            pose2: 第二个位姿 (R2, t2)
            
        Returns:
            R_rel: 相对旋转矩阵
            t_rel: 相对平移向量
        """
        R1, t1 = pose1
        R2, t2 = pose2
        
        # 相对旋转
        R_rel = R2 @ R1.T
        
        # 相对平移
        t_rel = t2 - R_rel @ t1
        
        return R_rel, t_rel
    
    def estimate_multiple_transforms(self, point_sequence):
        """
        估计点序列中的多个变换
        
        Args:
            point_sequence: 点序列列表，每个元素是 [N_i, 3] 的点集
            
        Returns:
            transforms: 变换列表
            info_list: 信息列表
        """
        transforms = []
        info_list = []
        
        for i in range(len(point_sequence) - 1):
            R, t, s, inliers, info = self.estimate_transform(
                point_sequence[i], point_sequence[i+1]
            )
            
            transforms.append((R, t, s))
            info['frame_pair'] = (i, i+1)
            info_list.append(info)
        
        return transforms, info_list


def test_pose_estimator():
    """测试位姿估计器"""
    print("测试位姿估计器...")
    
    # 创建测试数据
    np.random.seed(42)
    
    # 生成源点云
    n_points = 100
    points_src = np.random.randn(n_points, 3) * 0.5
    
    # 定义真实变换
    true_R = R.from_euler('xyz', [15, 30, 45], degrees=True).as_matrix()
    true_t = np.array([1.0, 2.0, 3.0])
    true_scale = 1.0  # 刚体变换，尺度为1
    
    # 应用变换
    points_dst = true_scale * (true_R @ points_src.T).T + true_t
    
    # 添加噪声（部分点）
    noise_mask = np.random.rand(n_points) < 0.3  # 30%的点添加噪声
    points_dst_noisy = points_dst.copy()
    points_dst_noisy[noise_mask] += np.random.randn(np.sum(noise_mask), 3) * 0.5
    
    # 初始化估计器
    estimator = PoseEstimator({
        'method': 'ransac',
        'ransac_threshold': 0.1,
        'ransac_iterations': 1000,
        'verbose': True
    })
    
    # 估计变换
    print("\n使用RANSAC估计变换...")
    R_est, t_est, s_est, inliers, info = estimator.estimate_transform(
        points_src, points_dst_noisy
    )
    
    if R_est is not None:
        print(f"真实旋转矩阵:\n{true_R}")
        print(f"估计旋转矩阵:\n{R_est}")
        print(f"旋转误差: {np.linalg.norm(true_R - R_est):.6f}")
        
        print(f"\n真实平移: {true_t}")
        print(f"估计平移: {t_est}")
        print(f"平移误差: {np.linalg.norm(true_t - t_est):.6f}")
        
        print(f"\n内点数: {info['n_inliers']}/{n_points} ({info['inlier_ratio']*100:.1f}%)")
        print(f"RMSE: {info['rmse']:.6f}")
        
        # 计算角速度示例
        delta_time = 0.1  # 假设0.1秒
        angular_velocity = estimator.compute_angular_velocity(
            np.eye(3), R_est, delta_time
        )
        print(f"\n角速度: {angular_velocity} rad/s")
        
        # 转换为欧拉角
        euler_angles = estimator.rotation_matrix_to_euler(R_est, degrees=True)
        print(f"欧拉角 (roll, pitch, yaw): {euler_angles}°")
    
    return estimator, points_src, points_dst_noisy

if __name__ == "__main__":
    # 运行测试
    estimator, points_src, points_dst = test_pose_estimator()
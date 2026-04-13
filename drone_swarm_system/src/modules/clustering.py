import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2

class TraditionalSegmenter:
    """
    传统聚类分割器
    使用K-means、DBSCAN等算法对特征点进行分割
    """
    
    def __init__(self, method='kmeans', n_clusters=3, eps=0.5, min_samples=5):
        """
        初始化分割器
        
        Args:
            method: 聚类方法 ('kmeans', 'dbscan', 'gmm')
            n_clusters: K-means聚类数
            eps: DBSCAN邻域半径
            min_samples: DBSCAN最小样本数
        """
        self.method = method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        
    def extract_features(self, points, flow_vectors):
        """
        提取聚类特征
        
        Args:
            points: 像素坐标 [N, 2]
            flow_vectors: 光流向量 [N, 2]
            
        Returns:
            features: 特征矩阵 [N, n_features]
        """
        N = points.shape[0]
        
        # 基本特征：光流幅度和方向
        flow_magnitude = np.linalg.norm(flow_vectors, axis=1, keepdims=True)
        flow_angle = np.arctan2(flow_vectors[:, 1:2], flow_vectors[:, 0:1])
        
        # 空间位置特征（归一化）
        points_normalized = points / np.max(points, axis=0)
        
        # 组合特征
        features = np.hstack([
            flow_magnitude,      # 光流大小
            flow_angle,          # 光流方向
            points_normalized,   # 归一化位置
            np.sin(flow_angle),  # 方向正弦
            np.cos(flow_angle),  # 方向余弦
        ])
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        return features_scaled
    
    def segment_by_kmeans(self, features):
        """K-means聚类分割"""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(features)
        return labels
    
    def segment_by_dbscan(self, features):
        """DBSCAN密度聚类分割"""
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        labels = dbscan.fit_predict(features)
        return labels
    
    def segment(self, points, flow_vectors):
        """
        执行分割
        
        Args:
            points: 像素坐标 [N, 2]
            flow_vectors: 光流向量 [N, 2]
            
        Returns:
            labels: 分割标签 [N]
            segment_info: 分割统计信息
        """
        # 提取特征
        features = self.extract_features(points, flow_vectors)
        
        # 执行聚类
        if self.method == 'kmeans':
            labels = self.segment_by_kmeans(features)
        elif self.method == 'dbscan':
            labels = self.segment_by_dbscan(features)
        else:
            raise ValueError(f"不支持的聚类方法: {self.method}")
        
        # 统计信息
        segment_info = self._analyze_segments(labels, points, flow_vectors)
        
        return labels, segment_info
    
    def _analyze_segments(self, labels, points, flow_vectors):
        """分析分割结果"""
        unique_labels = np.unique(labels)
        segment_info = {}
        
        for label in unique_labels:
            if label == -1:  # DBSCAN噪声点
                continue
                
            mask = labels == label
            n_points = np.sum(mask)
            
            if n_points < 10:  # 跳过点数太少的段
                continue
            
            segment_points = points[mask]
            segment_flow = flow_vectors[mask]
            
            # 计算统计信息
            centroid = np.mean(segment_points, axis=0)
            avg_flow = np.mean(segment_flow, axis=0)
            flow_magnitude = np.mean(np.linalg.norm(segment_flow, axis=1))
            
            # 边界框
            min_x, min_y = np.min(segment_points, axis=0)
            max_x, max_y = np.max(segment_points, axis=0)
            bbox = [min_x, min_y, max_x, max_y]
            
            segment_info[f'segment_{label}'] = {
                'label': int(label),
                'n_points': int(n_points),
                'centroid': centroid.tolist(),
                'avg_flow': avg_flow.tolist(),
                'flow_magnitude': float(flow_magnitude),
                'bbox': bbox,
                'points': segment_points,
                'flow_vectors': segment_flow
            }
        
        return segment_info
    
    def visualize_segmentation(self, image, points, labels, save_path=None):
        """
        可视化分割结果
        
        Args:
            image: 背景图像 [H, W, 3]
            points: 像素坐标 [N, 2]
            labels: 分割标签 [N]
            save_path: 保存路径
        """
        # 创建可视化图像
        vis_image = image.copy()
        
        # 定义颜色映射
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
        ]
        
        # 绘制点
        for i, (point, label) in enumerate(zip(points, labels)):
            if label == -1:  # 噪声点用白色
                color = (255, 255, 255)
            elif label < len(colors):
                color = colors[label % len(colors)]
            else:
                color = (128, 128, 128)  # 灰色
            
            # 绘制点
            x, y = int(point[0]), int(point[1])
            cv2.circle(vis_image, (x, y), 2, color, -1)
        
        # 绘制标签
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = labels == label
            if np.sum(mask) > 0:
                segment_points = points[mask]
                centroid = np.mean(segment_points, axis=0).astype(int)
                color = colors[label % len(colors)] if label < len(colors) else (128, 128, 128)
                
                # 绘制标签文本
                cv2.putText(vis_image, f'C{label}', 
                           (centroid[0] + 10, centroid[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
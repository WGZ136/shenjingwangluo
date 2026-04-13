"""
光流处理模块
封装RAFT光流算法，提供统一接口
"""

import torch
import numpy as np
import cv2
import sys
import os

# 添加RAFT路径
sys.path.append(os.path.join(os.path.dirname(__file__), 
                            '../../../core_algorithms/raft'))

try:
    from core.raft import RAFT
    from utils.utils import InputPadder
    from utils import flow_viz
except ImportError as e:
    print(f"导入RAFT失败: {e}")
    print("请确保RAFT已正确放置在core_algorithms目录中")
    raise

class FlowProcessor:
    """
    光流处理器
    封装RAFT算法，提供光流计算、特征点提取等功能
    """
    
    def __init__(self, config=None):
        """
        初始化光流处理器
        
        Args:
            config: 配置字典
        """
        self.config = self._load_config(config)
        self.device = self._setup_device()
        
        # 加载RAFT模型
        self.raft_model = None
        self._load_raft_model()
        
        print(f"光流处理器初始化完成 (设备: {self.device})")
    
    def _load_config(self, config):
        """加载配置"""
        default_config = {
            'model_path': 'data/model_weights/raft-things.pth',
            'iterations': 20,
            'mixed_precision': False,
            'small_model': False,
            'alternate_corr': False,
            'use_gpu': torch.cuda.is_available(),
            'verbose': True
        }
        
        if config is None:
            return default_config
        elif isinstance(config, dict):
            default_config.update(config)
            return default_config
        else:
            raise ValueError("配置必须是字典或None")
    
    def _setup_device(self):
        """设置设备，处理CUDA不可用的情况"""
        use_gpu = self.config.get('use_gpu', False)
        
        if use_gpu:
            if torch.cuda.is_available():
                try:
                    # 测试CUDA是否真的可用
                    test_tensor = torch.tensor([1.0]).cuda()
                    _ = test_tensor + 1
                    del test_tensor
                    
                    device = torch.device('cuda')
                    if self.config.get('verbose', True):
                        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
                    return device
                except RuntimeError as e:
                    print(f"⚠️  CUDA测试失败: {e}")
                    print("   自动切换到CPU模式")
            else:
                print("⚠️  请求使用GPU但CUDA不可用，自动切换到CPU模式")
        
        device = torch.device('cpu')
        if self.config.get('verbose', True):
            print("使用CPU")
        return device
    
    def _load_raft_model(self):
        """加载RAFT模型"""
        print("加载RAFT模型...")
        
        # 创建模型参数对象
        class Args:
            def __init__(self):
                self.small = self.config.get('small_model', False)
                self.mixed_precision = self.config.get('mixed_precision', False)
                self.alternate_corr = self.config.get('alternate_corr', False)
        
        args = Args()
        
        # 创建RAFT模型
        self.raft_model = torch.nn.DataParallel(RAFT(args))
        
        # 加载权重
        model_path = self.config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RAFT权重文件不存在: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.raft_model.load_state_dict(state_dict)
        
        # 设置模型为评估模式
        self.raft_model = self.raft_model.module
        self.raft_model.to(self.device)
        self.raft_model.eval()
        
        print("RAFT模型加载完成")
    
    def preprocess_image(self, image):
        """
        预处理图像为RAFT输入格式
        
        Args:
            image: numpy数组 [H, W, 3] (RGB或BGR)
            
        Returns:
            torch.Tensor: 预处理后的图像张量 [1, 3, H, W]
        """
        # 确保是RGB格式
        if len(image.shape) == 2:
            # 灰度图转RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # BGR转RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # RGBA转RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        img_tensor = img_tensor[None].to(self.device)  # 添加batch维度
        
        return img_tensor
    
    def compute_flow(self, image1, image2, return_type='numpy'):
        """
        计算两帧图像之间的光流
        
        Args:
            image1: t时刻图像
            image2: t+1时刻图像
            return_type: 返回类型 ('numpy', 'tensor', 'both')
            
        Returns:
            根据return_type返回光流数据
        """
        # 预处理图像
        img1_tensor = self.preprocess_image(image1)
        img2_tensor = self.preprocess_image(image2)
        
        # 对齐图像尺寸
        padder = InputPadder(img1_tensor.shape)
        img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)
        
        with torch.no_grad():
            # RAFT推理
            flow_low, flow_up = self.raft_model(
                img1_padded, img2_padded,
                iters=self.config['iterations'],
                test_mode=True
            )
            
            # 移除填充
            flow_nopad = padder.unpad(flow_up[0])
            
            # 获取numpy数组
            flow_np = flow_nopad.permute(1, 2, 0).cpu().numpy()
        
        if return_type == 'numpy':
            return flow_np
        elif return_type == 'tensor':
            return flow_nopad
        elif return_type == 'both':
            return flow_np, flow_nopad
        else:
            raise ValueError(f"不支持的返回类型: {return_type}")
    
    def extract_feature_points(self, flow, grid_step=10, threshold=0.5):
        """
        从光流场中提取特征点对
        
        Args:
            flow: 光流场 [H, W, 2]
            grid_step: 网格采样步长
            threshold: 光流大小阈值（过滤静止点）
            
        Returns:
            points_t: t时刻的像素坐标 [N, 2]
            points_t1: t+1时刻的像素坐标 [N, 2]
            flow_vectors: 光流向量 [N, 2]
            mask: 有效的点掩码 [N]
        """
        H, W = flow.shape[:2]
        
        # 创建采样网格
        y_coords = np.arange(0, H, grid_step)
        x_coords = np.arange(0, W, grid_step)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords)
        
        # 展平并组合为点
        points_t = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
        
        # 提取光流向量
        flow_vectors = flow[grid_y.flatten(), grid_x.flatten()]
        
        # 计算光流大小
        flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
        
        # 过滤静止点（可选）
        if threshold > 0:
            mask = flow_magnitudes > threshold
            points_t = points_t[mask]
            flow_vectors = flow_vectors[mask]
            flow_magnitudes = flow_magnitudes[mask]
        else:
            mask = np.ones(len(points_t), dtype=bool)
        
        # 计算t+1时刻的点
        points_t1 = points_t + flow_vectors
        
        return points_t, points_t1, flow_vectors, mask
    
    def compute_flow_statistics(self, flow):
        """
        计算光流统计信息
        
        Args:
            flow: 光流场 [H, W, 2]
            
        Returns:
            stats: 统计信息字典
        """
        # 分离x和y方向
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        
        # 计算光流大小
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # 统计信息
        stats = {
            'mean_x': np.mean(flow_x),
            'mean_y': np.mean(flow_y),
            'std_x': np.std(flow_x),
            'std_y': np.std(flow_y),
            'max_magnitude': np.max(flow_magnitude),
            'min_magnitude': np.min(flow_magnitude),
            'mean_magnitude': np.mean(flow_magnitude),
            'median_magnitude': np.median(flow_magnitude),
            'total_pixels': flow.shape[0] * flow.shape[1],
            'moving_pixels': np.sum(flow_magnitude > 0.5),  # 阈值0.5像素
            'flow_angle_histogram': self._compute_angle_histogram(flow_x, flow_y)
        }
        
        return stats
    
    def _compute_angle_histogram(self, flow_x, flow_y, n_bins=8):
        """计算光流角度直方图"""
        # 计算角度
        angles = np.arctan2(flow_y, flow_x)  # 弧度 [-π, π]
        
        # 转换为度数 [0, 360)
        angles_deg = np.degrees(angles) % 360
        
        # 计算直方图
        hist, bin_edges = np.histogram(angles_deg, bins=n_bins, range=(0, 360))
        
        # 归一化
        hist_normalized = hist / np.sum(hist)
        
        return {
            'counts': hist.tolist(),
            'normalized': hist_normalized.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def visualize_flow(self, flow, image=None, method='hsv'):
        """
        可视化光流
        
        Args:
            flow: 光流场 [H, W, 2]
            image: 原始图像（可选）
            method: 可视化方法 ('hsv', 'rgb', 'quiver')
            
        Returns:
            flow_image: 可视化图像
        """
        if method == 'hsv':
            # HSV色彩空间可视化（RAFT风格）
            flow_image = flow_viz.flow_to_image(flow)
            
            # 如果需要叠加到原始图像
            if image is not None:
                # 调整光流图像透明度
                alpha = 0.7
                overlay = cv2.addWeighted(image, 1-alpha, flow_image, alpha, 0)
                return overlay
            
            return flow_image
        
        elif method == 'quiver':
            # 矢量箭头可视化
            if image is None:
                image = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.uint8) * 255
            
            # 采样点
            H, W = flow.shape[:2]
            grid_step = max(H // 20, W // 20, 1)
            
            y, x = np.mgrid[0:H:grid_step, 0:W:grid_step]
            u = flow[y, x, 0]
            v = flow[y, x, 1]
            
            # 创建可视化图像
            vis_image = image.copy()
            
            # 绘制箭头
            for i in range(len(x)):
                for j in range(len(y)):
                    px = int(x[j, i])
                    py = int(y[j, i])
                    dx = int(u[j, i])
                    dy = int(v[j, i])
                    
                    # 只绘制有显著运动的点
                    if abs(dx) > 1 or abs(dy) > 1:
                        cv2.arrowedLine(vis_image, 
                                       (px, py), 
                                       (px + dx, py + dy), 
                                       (0, 0, 255),  # 红色箭头
                                       1, 
                                       tipLength=0.3)
            
            return vis_image
        
        else:
            raise ValueError(f"不支持的的可视化方法: {method}")
    
    def save_flow_data(self, flow, filepath, format='npy'):
        """
        保存光流数据
        
        Args:
            flow: 光流场
            filepath: 文件路径
            format: 保存格式 ('npy', 'png', 'flo')
        """
        if format == 'npy':
            np.save(filepath, flow)
        elif format == 'png':
            # 可视化后保存
            flow_image = self.visualize_flow(flow)
            cv2.imwrite(filepath, flow_image)
        elif format == 'flo':
            # Middlebury .flo 格式
            self._save_flo_file(flow, filepath)
        else:
            raise ValueError(f"不支持的保存格式: {format}")
    
    def _save_flo_file(self, flow, filepath):
        """保存为Middlebury .flo格式"""
        TAG_FLOAT = 202021.25
        
        with open(filepath, 'wb') as f:
            # 写入标签
            np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
            
            # 写入尺寸
            np.array(flow.shape[1], dtype=np.int32).tofile(f)  # 宽度
            np.array(flow.shape[0], dtype=np.int32).tofile(f)  # 高度
            
            # 写入数据 (逐像素)
            for y in range(flow.shape[0]):
                for x in range(flow.shape[1]):
                    np.array(flow[y, x, 0], dtype=np.float32).tofile(f)  # u
                    np.array(flow[y, x, 1], dtype=np.float32).tofile(f)  # v
    
    def batch_process(self, image_sequence, return_statistics=True):
        """
        批量处理图像序列
        
        Args:
            image_sequence: 图像序列列表
            return_statistics: 是否返回统计信息
            
        Returns:
            flow_results: 光流结果列表
            statistics: 统计信息列表（如果return_statistics为True）
        """
        flow_results = []
        statistics_list = [] if return_statistics else None
        
        for i in range(len(image_sequence) - 1):
            print(f"处理帧 {i+1}/{len(image_sequence)-1}")
            
            # 计算光流
            flow = self.compute_flow(image_sequence[i], image_sequence[i+1])
            flow_results.append(flow)
            
            # 计算统计信息
            if return_statistics:
                stats = self.compute_flow_statistics(flow)
                stats['frame_pair'] = (i, i+1)
                statistics_list.append(stats)
        
        if return_statistics:
            return flow_results, statistics_list
        else:
            return flow_results


# 简单测试函数
def test_flow_processor():
    """测试光流处理器"""
    import matplotlib.pyplot as plt
    
    print("测试光流处理器...")
    
    # 创建测试图像
    H, W = 480, 640
    img1 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    # 创建第二帧（轻微平移）
    img2 = np.roll(img1, shift=5, axis=1)  # 水平平移5像素
    
    # 初始化处理器
    processor = FlowProcessor({
        'model_path': 'data/model_weights/raft-things.pth',
        'use_gpu': False  # 测试时使用CPU
    })
    
    # 计算光流
    flow = processor.compute_flow(img1, img2)
    
    print(f"光流形状: {flow.shape}")
    print(f"光流范围: [{flow.min():.2f}, {flow.max():.2f}]")
    
    # 提取特征点
    points_t, points_t1, flow_vectors, mask = processor.extract_feature_points(
        flow, grid_step=20
    )
    
    print(f"提取到 {len(points_t)} 个特征点")
    
    # 可视化
    flow_image = processor.visualize_flow(flow, img1, method='hsv')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Frame t")
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB))
    plt.title("Optical Flow")
    
    plt.subplot(133)
    plt.scatter(points_t[:, 0], points_t[:, 1], s=1, c='r', label='Frame t')
    plt.scatter(points_t1[:, 0], points_t1[:, 1], s=1, c='b', label='Frame t+1')
    for i in range(0, len(points_t), 10):  # 每10个点画一条线
        plt.plot([points_t[i, 0], points_t1[i, 0]], 
                [points_t[i, 1], points_t1[i, 1]], 
                'g-', alpha=0.3, linewidth=0.5)
    plt.title("Feature Points")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return processor, flow

if __name__ == "__main__":
    # 运行测试
    processor, flow = test_flow_processor()
"""
Monodepth2深度估计器封装
"""

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# 添加Monodepth2路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../core_algorithms/monodepth2'))

try:
    from monodepth2.networks import ResnetEncoder, DepthDecoder
    from monodepth2.layers import disp_to_depth
    from monodepth2.utils import download_model_if_doesnt_exist
except ImportError as e:
    print(f"导入Monodepth2失败: {e}")
    print("请确保Monodepth2已正确安装")
    sys.exit(1)

class Monodepth2Estimator:
    """
    Monodepth2深度估计器
    为无人机集群系统提供深度估计功能
    """
    
    def __init__(self, model_path='models/mono_640x192', 
                 input_width=640, input_height=192,
                 min_depth=0.1, max_depth=100.0, scale_factor=5.4,
                 device='cuda'):
        """
        初始化深度估计器
        
        Args:
            model_path: 模型路径
            input_width: 输入宽度
            input_height: 输入高度
            min_depth: 最小深度值
            max_depth: 最大深度值
            scale_factor: 深度缩放因子
            device: 计算设备
        """
        # 智能设备选择：如果指定了cuda但不可用，自动切换到cpu
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  请求使用CUDA但不可用，自动切换到CPU模式")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # 参数设置
        self.input_width = input_width
        self.input_height = input_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale_factor = scale_factor
        
        # 确保模型存在
        download_model_if_doesnt_exist(model_path)
        
        # 加载模型
        self.encoder, self.depth_decoder = self._load_model(model_path)
        
        # 预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((self.input_height, self.input_width), 
                             interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Monodepth2深度估计器初始化完成 (设备: {self.device})")
    
    def _load_model(self, model_path):
        """加载Monodepth2模型"""
        print(f"加载Monodepth2模型: {model_path}")
        
        # 加载编码器
        encoder = ResnetEncoder(18, False)
        encoder_path = os.path.join(model_path, "encoder.pth")
        encoder_weights = torch.load(encoder_path, map_location=self.device)
        encoder.load_state_dict(
            {k: v for k, v in encoder_weights.items() 
             if k in encoder.state_dict()}
        )
        encoder.to(self.device)
        encoder.eval()
        
        # 加载深度解码器
        depth_decoder = DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, 
            scales=range(4)
        )
        decoder_path = os.path.join(model_path, "depth.pth")
        decoder_weights = torch.load(decoder_path, map_location=self.device)
        depth_decoder.load_state_dict(decoder_weights)
        depth_decoder.to(self.device)
        depth_decoder.eval()
        
        return encoder, depth_decoder
    
    def estimate_depth(self, image, input_format='RGB'):
        """
        估计单张图像的深度
        
        Args:
            image: 输入图像 (numpy数组、PIL Image或文件路径)
            input_format: 输入格式 'RGB' 或 'BGR'
            
        Returns:
            depth_map: 深度图 [H, W]
        """
        # 预处理图像
        if isinstance(image, str):
            # 文件路径
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # numpy数组
            if input_format == 'BGR':
                # BGR转RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            # PIL Image
            pil_image = image.convert('RGB')
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        original_size = pil_image.size  # (width, height)
        
        # 应用变换
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 前向传播
            features = self.encoder(input_tensor)
            outputs = self.depth_decoder(features)
            
            # 获取视差图并转换为深度
            disp = outputs[("disp", 0)]
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            depth = depth * self.scale_factor
            
            # 裁剪深度范围
            depth = torch.clamp(depth, self.min_depth, self.max_depth)
            
            # 转换为numpy并调整到原始尺寸
            depth_np = depth.squeeze().cpu().numpy()
            
            if original_size[0] != self.input_width or original_size[1] != self.input_height:
                depth_resized = cv2.resize(
                    depth_np, 
                    original_size, 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                depth_resized = depth_np
        
        return depth_resized
    
    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_JET):
        """
        可视化深度图
        
        Returns:
            color_depth: 彩色深度图
        """
        # 归一化到0-255
        depth_normalized = cv2.normalize(
            depth_map, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_normalized = np.uint8(depth_normalized)
        
        # 应用色彩映射
        color_depth = cv2.applyColorMap(depth_normalized, colormap)
        
        return color_depth
    
    def save_depth(self, depth_map, filename, format='npy'):
        """保存深度图"""
        if format == 'npy':
            np.save(f"{filename}.npy", depth_map)
        elif format == 'png':
            depth_normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)
            cv2.imwrite(f"{filename}.png", depth_normalized)
        else:
            raise ValueError(f"不支持的格式: {format}")
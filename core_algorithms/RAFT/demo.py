import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

class RAFTProcessor:
    """RAFT光流处理器封装"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        """加载RAFT模型"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--small', action='store_true')
        parser.add_argument('--mixed_precision', action='store_true')
        parser.add_argument('--alternate_corr', action='store_true')
        args = parser.parse_args([])  # 空列表避免参数冲突
        
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(model_path))
        return model.module.to(self.device)
    
    def load_image(self, image_path):
        """加载图像"""
        if isinstance(image_path, str):
            img = np.array(Image.open(image_path)).astype(np.uint8)
        elif isinstance(image_path, np.ndarray):
            img = image_path.astype(np.uint8)
        else:
            raise ValueError("不支持的图像输入类型")
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)
    
    def compute_flow(self, image1, image2, iters=20):
        """计算两帧图像之间的光流"""
        # 如果是文件路径，加载图像
        if isinstance(image1, str):
            image1 = self.load_image(image1)
        if isinstance(image2, str):
            image2 = self.load_image(image2)
        
        # 确保是tensor
        if isinstance(image1, np.ndarray):
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        if isinstance(image2, np.ndarray):
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        
        # 填充到合适尺寸
        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)
        
        with torch.no_grad():
            flow_low, flow_up = self.model(image1_pad, image2_pad, iters=iters, test_mode=True)
        
        # 移除填充并返回
        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_np = padder.unpad(torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0))
        flow_np = flow_np[0].permute(1, 2, 0).numpy()
        
        return flow_np
    
    def visualize_flow(self, flow, save_path=None):
        """可视化光流"""
        flow_img = flow_viz.flow_to_image(flow)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))
        
        return flow_img

def main():
    """主函数（保持向后兼容）"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="RAFT模型路径")
    parser.add_argument('--path', default='../data/test_images', help="图像序列路径")
    parser.add_argument('--output', default='../data/experiment_results/flow', help="输出目录")
    args = parser.parse_args()
    
    # 创建处理器
    processor = RAFTProcessor(args.model)
    
    # 处理图像序列
    images = sorted(glob.glob(os.path.join(args.path, '*.png')) + 
                   glob.glob(os.path.join(args.path, '*.jpg')))
    
    os.makedirs(args.output, exist_ok=True)
    
    for i, (img1_path, img2_path) in enumerate(zip(images[:-1], images[1:])):
        print(f"处理帧对 {i+1}: {os.path.basename(img1_path)} -> {os.path.basename(img2_path)}")
        
        # 计算光流
        flow = processor.compute_flow(img1_path, img2_path)
        
        # 保存结果
        base_name = f"{os.path.splitext(os.path.basename(img1_path))[0]}_{os.path.splitext(os.path.basename(img2_path))[0]}"
        np.save(os.path.join(args.output, f"{base_name}_flow.npy"), flow)
        
        # 可视化
        flow_img = processor.visualize_flow(flow)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_flow.png"), 
                   cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))
        
        print(f"  已保存: {base_name}_flow.npy, {base_name}_flow.png")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Monodepth2 AirSim数据训练脚本
使用您的AirSim数据进行有监督深度估计训练
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'monodepth2'))

from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.networks.depth_decoder import DepthDecoder
from monodepth2.datasets.airsim_dataset import AirSimDataset

def create_model(device):
    """创建Monodepth2模型"""
    # 创建编码器
    encoder = ResnetEncoder(18, False)
    encoder.num_ch_enc = [64, 64, 128, 256, 512]
    
    # 创建解码器
    decoder = DepthDecoder(encoder.num_ch_enc)
    
    # 将模型移动到设备
    encoder.to(device)
    decoder.to(device)
    
    return encoder, decoder

def compute_loss(pred_depth, gt_depth, mask=None):
    """计算深度估计损失"""
    
    # 确保深度值有效
    valid_mask = (gt_depth > 0) & (gt_depth < 80)  # 假设深度范围0-80米
    if mask is not None:
        valid_mask = valid_mask & mask
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    
    # 展平张量并使用布尔掩码索引
    pred_flat = pred_depth.view(-1)
    gt_flat = gt_depth.view(-1)
    valid_flat = valid_mask.view(-1)
    
    pred_valid = pred_flat[valid_flat]
    gt_valid = gt_flat[valid_flat]
    
    # L1损失
    l1_loss = torch.abs(pred_valid - gt_valid).mean()
    
    # 尺度不变损失
    diff_log = torch.log(pred_valid) - torch.log(gt_valid)
    scale_inv_loss = torch.sqrt((diff_log ** 2).mean() - (diff_log.mean()) ** 2)
    
    # 总损失
    total_loss = l1_loss + 0.1 * scale_inv_loss
    
    return total_loss

def train_model(args):
    """训练模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    encoder, decoder = create_model(device)
    
    # 创建优化器
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建数据集
    train_dataset = AirSimDataset(
        data_root=args.data_path,
        split='train',
        height=args.height,
        width=args.width,
        is_training=True
    )
    
    val_dataset = AirSimDataset(
        data_root=args.data_path,
        split='val',
        height=args.height,
        width=args.width,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(os.path.join(args.log_dir, args.model_name))
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # 训练阶段
        encoder.train()
        decoder.train()
        
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            images = batch['image'].to(device)
            gt_depths = batch['depth'].to(device)
            
            # 前向传播
            features = encoder(images)
            outputs = decoder(features)
            
            # 使用尺度0的深度图（最高分辨率）
            pred_depths = outputs[("disp", 0)]
            
            # 计算损失
            loss = compute_loss(pred_depths, gt_depths)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印进度
            if batch_idx % args.log_frequency == 0:
                print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f}')
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        
        # 验证阶段
        encoder.eval()
        decoder.eval()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                gt_depths = batch['depth'].to(device)
                
                features = encoder(images)
                outputs = decoder(features)
                
                # 使用尺度0的深度图（最高分辨率）
                pred_depths = outputs[("disp", 0)]
                
                loss = compute_loss(pred_depths, gt_depths)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # 保存模型
            save_path = os.path.join(args.save_dir, args.model_name)
            os.makedirs(save_path, exist_ok=True)
            
            torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.pth'))
            
            print(f'保存最佳模型到: {save_path}')
        
        # 保存检查点
        if epoch % args.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(args.save_dir, args.model_name, f'checkpoint_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
    
    writer.close()
    print("训练完成!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Monodepth2 AirSim训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='airsim_data',
                       help='数据目录路径')
    parser.add_argument('--model_name', type=str, default='airsim_monodepth2',
                       help='模型名称')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    
    # 图像参数
    parser.add_argument('--height', type=int, default=192,
                       help='输入图像高度')
    parser.add_argument('--width', type=int, default=640,
                       help='输入图像宽度')
    
    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作线程数')
    parser.add_argument('--no_cuda', action='store_true',
                       help='不使用CUDA')
    
    # 日志和保存参数
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志目录')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--log_frequency', type=int, default=10,
                       help='日志记录频率')
    parser.add_argument('--checkpoint_frequency', type=int, default=10,
                       help='检查点保存频率')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Monodepth2 AirSim训练")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"模型名称: {args.model_name}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print("=" * 60)
    
    # 开始训练
    train_model(args)

if __name__ == "__main__":
    main()
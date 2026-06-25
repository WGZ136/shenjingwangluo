"""
U-Net 图像分割训练脚本
用于无人机集群感知任务，将图像分割为背景和多个无人机类别

【依赖安装说明】
以下是经过兼容性验证的完整 requirements.txt 文件，可直接使用：

# ====== 深度学习核心框架 ======
# PyTorch 2.7.0 — 首个原生支持 CUDA 12.8 的稳定版（使用预编译 cu128 wheel）
# torch==2.7.0+cu128
# torchvision==0.22.0+cu128

# ====== 分割模型库 ======
# segmentation-models-pytorch==0.5.0

# ====== 科学计算与图像处理 ======
# numpy==2.2.4
# opencv-python==4.10.0.84
# pillow==11.3.0
# matplotlib==3.9.4

# ====== 训练辅助工具 ======
# tqdm==4.67.1
# tensorboard==2.18.0
# scikit-learn==1.5.2

【安装命令】
# 安装 PyTorch（CUDA 12.8）
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install segmentation-models-pytorch==0.5.0 numpy==2.2.4 opencv-python==4.10.0.84 pillow==11.3.0 matplotlib==3.9.4 tqdm==4.67.1 tensorboard==2.18.0 scikit-learn==1.5.2

作者: AI Assistant
日期: 2026-06-10
"""

import os
import time
import json
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
import torchvision.transforms.functional as TF

# 尝试导入 segmentation-models-pytorch，如果失败则使用手动实现的 U-Net
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("警告: segmentation-models-pytorch 未安装，将使用手动实现的 U-Net")

from sklearn.utils.class_weight import compute_class_weight


# ==================== 配置参数区 ====================
class Config:
    """训练配置参数，集中管理所有超参数"""
    
    # 【数据集路径配置】
    # 支持多个数据集合并使用，按顺序依次加载
    # 格式: [(图像目录, 标签目录), ...]
    DATASET_DIRS = [
        # 主数据集（5000帧，推荐）
        (r"d:\图片生成\airsim_seg_dataset_multi_v4\images", 
         r"d:\图片生成\airsim_seg_dataset_multi_v4\labels"),
        # 历史数据集（可选，取消注释以合并使用）
        # (r"d:\图片生成\airsim_seg_dataset_multi\images", 
        #  r"d:\图片生成\airsim_seg_dataset_multi\labels"),
        # (r"d:\图片生成\airsim_seg_dataset_multi_v2\images", 
        #  r"d:\图片生成\airsim_seg_dataset_multi_v2\labels"),
        # (r"d:\图片生成\airsim_seg_dataset_multi_v3\images", 
        #  r"d:\图片生成\airsim_seg_dataset_multi_v3\labels"),
    ]
    
    # 数据集划分比例 (训练集, 验证集, 测试集)
    SPLIT_RATIOS = (0.8, 0.1, 0.1)  # 80% / 10% / 10%
    
    # 如果数据已按 train/val/test 划分好，可取消下方注释并注释掉 DATASET_DIRS
    # DATA_ROOT = r"d:\图片生成\airsim_seg_dataset_multi_v4"
    # TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train", "images")
    # TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, "train", "labels")
    # VAL_IMAGE_DIR = os.path.join(DATA_ROOT, "val", "images")
    # VAL_LABEL_DIR = os.path.join(DATA_ROOT, "val", "labels")
    # TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test", "images")
    # TEST_LABEL_DIR = os.path.join(DATA_ROOT, "test", "labels")
    
    # 输出路径配置
    OUTPUT_DIR = "outputs"  # 输出目录
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # 模型配置
    USE_SMP = True  # 是否使用 segmentation-models-pytorch，False 则使用手动实现的 U-Net
    BACKBONE = "resnet34"  # 编码器 backbone (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
    ENCODER_WEIGHTS = "imagenet"  # 编码器预训练权重
    
    # 类别配置
    BINARY_MODE = True  # 是否使用二分类模式（背景 vs 无人机）
    # 二分类模式说明：
    #   True:  将所有非背景像素（1~5）合并为"无人机"类（类别1）
    #   False: 保持原始6分类（背景 + 5架独立无人机）
    NUM_CLASSES = 2 if BINARY_MODE else 6
    CLASS_NAMES = ["background", "drone"] if BINARY_MODE else ["background", "Drone1", "Drone2", "Drone3", "Drone4", "Drone5"]
    
    # 输入尺寸配置
    # 原始图像尺寸为 640x480，可保持原尺寸或缩小以加速训练
    INPUT_WIDTH = 640   # 输入图像宽度
    INPUT_HEIGHT = 480  # 输入图像高度
    # 若需缩小训练，可改为 320x240 或 256x256
    
    # 训练超参数
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # 优化器配置
    OPTIMIZER = "adam"  # "adam" 或 "sgd"
    
    # 学习率调度
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "step"  # "step", "cosine", "plateau"
    STEP_SIZE = 30  # StepLR 的步长
    GAMMA = 0.1  # StepLR 的衰减系数
    
    # 损失函数配置
    USE_CLASS_WEIGHTS = True  # 是否使用类别权重缓解类别不平衡
    CLASS_WEIGHTS = None  # 如果为 None，则自动计算；否则使用指定权重
    
    # 数据增强配置
    AUGMENTATION_PROB = 0.5  # 数据增强概率
    ROTATION_DEGREES = 10  # 随机旋转角度范围
    TRANSLATION_RATIO = 0.1  # 随机平移比例
    SCALE_RANGE = (0.9, 1.1)  # 随机缩放范围
    COLOR_JITTER = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    }
    
    # 训练控制
    NUM_WORKERS = 4  # 数据加载线程数
    PIN_MEMORY = True  # 是否使用锁页内存加速 GPU 传输
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 20  # 早停耐心值
    
    # 验证和保存频率
    VALIDATION_INTERVAL = 1  # 每隔多少个 epoch 验证一次
    SAVE_INTERVAL = 10  # 每隔多少个 epoch 保存一次检查点
    
    # 随机种子
    SEED = 42


# ==================== 手动 U-Net 实现 ====================
class DoubleConv(nn.Module):
    """双卷积块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块: Upsample/ConvTranspose -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # 上采样方式选择
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # 处理尺寸不匹配问题
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        
        # 跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net 实现
    编码器-解码器结构，带跳跃连接
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 6, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出层
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        logits = self.outc(x)
        return logits


# ==================== 数据集划分工具 ====================
def split_dataset(dataset_dirs: List[Tuple[str, str]], 
                  ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                  seed: int = 42) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    将多个数据集按比例划分为训练集、验证集、测试集
    
    Args:
        dataset_dirs: 数据集目录列表，每个元素为 (image_dir, label_dir)
        ratios: 划分比例 (train, val, test)
        seed: 随机种子
    
    Returns:
        (train_files, val_files, test_files)
        每个文件为 (image_path, label_path) 元组
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # 收集所有样本
    all_files = []
    for image_dir, label_dir in dataset_dirs:
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"警告: 目录不存在，跳过: {image_dir}")
            continue
        
        image_files = sorted([f for f in os.listdir(image_dir) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_name in image_files:
            img_path = os.path.join(image_dir, img_name)
            # 标签文件名与图像相同（都是 .png）
            label_path = os.path.join(label_dir, img_name)
            
            # 如果标签不存在，跳过
            if not os.path.exists(label_path):
                # 尝试替换扩展名
                label_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                label_path = os.path.join(label_dir, label_name)
                if not os.path.exists(label_path):
                    continue
            
            all_files.append((img_path, label_path))
    
    if len(all_files) == 0:
        raise ValueError("未找到任何有效的图像-标签对！")
    
    # 随机打乱
    random.shuffle(all_files)
    
    # 按比例划分
    n_total = len(all_files)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    # 测试集取剩余部分，避免舍入误差
    n_test = n_total - n_train - n_val
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    print(f"数据集划分完成:")
    print(f"  总样本数: {n_total}")
    print(f"  训练集: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    print(f"  验证集: {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
    print(f"  测试集: {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    
    return train_files, val_files, test_files


# ==================== 数据集类 ====================
class SegmentationDataset(Dataset):
    """
    语义分割数据集类
    支持同步数据增强（图像和标签同时进行相同的变换）
    支持从文件列表加载或从目录加载
    """
    
    def __init__(self, image_dir: Optional[str] = None, label_dir: Optional[str] = None,
                 file_list: Optional[List[Tuple[str, str]]] = None,
                 input_width: int = 640, input_height: int = 480,
                 is_training: bool = True,
                 augmentation_config: Optional[Dict] = None):
        """
        Args:
            image_dir: 图像目录路径（与 file_list 二选一）
            label_dir: 标签目录路径（与 file_list 二选一）
            file_list: 文件列表，每个元素为 (image_path, label_path)，优先使用
            input_width: 输入图像宽度
            input_height: 输入图像高度
            is_training: 是否为训练模式（决定是否进行数据增强）
            augmentation_config: 数据增强配置字典
        """
        self.input_width = input_width
        self.input_height = input_height
        self.is_training = is_training
        self.augmentation_config = augmentation_config or {}
        self.file_list = file_list
        
        if file_list is not None:
            # 从文件列表加载
            self.samples = file_list
        elif image_dir is not None and label_dir is not None:
            # 从目录加载
            self.image_dir = image_dir
            self.label_dir = label_dir
            
            image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            self.samples = []
            for img_name in image_files:
                img_path = os.path.join(image_dir, img_name)
                label_path = os.path.join(label_dir, img_name)
                if not os.path.exists(label_path):
                    label_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                    label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    self.samples.append((img_path, label_path))
        else:
            raise ValueError("必须提供 file_list 或 image_dir + label_dir")
        
        if len(self.samples) == 0:
            raise ValueError("未找到任何样本")
        
        print(f"{'训练' if is_training else '验证/测试'}集: {len(self.samples)} 个样本")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从 samples 中获取图像和标签的完整路径
        img_path, label_path = self.samples[idx]
        
        # 读取图像 (RGB)
        image = Image.open(img_path).convert('RGB')
        
        # 读取标签 (单通道灰度图，像素值为类别索引)
        label = Image.open(label_path).convert('L')
        
        # 转换为 numpy 数组
        image_np = np.array(image)
        label_np = np.array(label)
        
        # 数据增强（训练模式）
        if self.is_training:
            image_np, label_np = self._apply_augmentation(image_np, label_np)
        
        # 缩放到目标尺寸
        image_np = cv2.resize(image_np, (self.input_width, self.input_height), 
                              interpolation=cv2.INTER_LINEAR)
        label_np = cv2.resize(label_np, (self.input_width, self.input_height), 
                              interpolation=cv2.INTER_NEAREST)
        
        # 二分类映射：将所有非背景像素合并为类别1
        if Config.BINARY_MODE:
            label_np = np.where(label_np > 0, 1, 0).astype(np.uint8)
        
        # 归一化并转换为 tensor
        image = self._to_tensor(image_np)
        label = torch.from_numpy(label_np).long()
        
        # 确保标签值在有效范围内
        label = torch.clamp(label, 0, Config.NUM_CLASSES - 1)
        
        return image, label
    
    def _apply_augmentation(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用同步数据增强"""
        h, w = image.shape[:2]
        
        # 随机水平翻转
        if np.random.random() < self.augmentation_config.get('flip_prob', 0.5):
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        
        # 随机旋转
        angle = self.augmentation_config.get('rotation_degrees', 10)
        if angle > 0 and np.random.random() < 0.5:
            angle = np.random.uniform(-angle, angle)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # 随机缩放和平移
        scale_range = self.augmentation_config.get('scale_range', (0.9, 1.1))
        translation_ratio = self.augmentation_config.get('translation_ratio', 0.1)
        
        if np.random.random() < 0.5:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            tx = np.random.uniform(-translation_ratio, translation_ratio) * w
            ty = np.random.uniform(-translation_ratio, translation_ratio) * h
            
            M = np.float32([[scale, 0, tx], [0, scale, ty]])
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # 颜色抖动（仅应用于图像）
        color_jitter = self.augmentation_config.get('color_jitter', {})
        if color_jitter:
            image = self._apply_color_jitter(image, color_jitter)
        
        return image, label
    
    def _apply_color_jitter(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """应用颜色抖动"""
        image = image.astype(np.float32) / 255.0
        
        # 亮度调整
        if 'brightness' in config and np.random.random() < 0.5:
            factor = np.random.uniform(1 - config['brightness'], 1 + config['brightness'])
            image = np.clip(image * factor, 0, 1)
        
        # 对比度调整
        if 'contrast' in config and np.random.random() < 0.5:
            factor = np.random.uniform(1 - config['contrast'], 1 + config['contrast'])
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 1)
        
        # 饱和度调整
        if 'saturation' in config and np.random.random() < 0.5:
            factor = np.random.uniform(1 - config['saturation'], 1 + config['saturation'])
            gray = image.mean(axis=2, keepdims=True)
            image = np.clip((image - gray) * factor + gray, 0, 1)
        
        # 色相调整
        if 'hue' in config and np.random.random() < 0.5:
            hue_shift = np.random.uniform(-config['hue'], config['hue']) * 180
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0
        
        return (image * 255).astype(np.uint8)
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """将图像转换为 tensor 并归一化"""
        # ImageNet 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image = (image - mean) / std
        
        return image


# ==================== 模型创建函数 ====================
def create_model(config: Config) -> nn.Module:
    """
    创建 U-Net 模型
    
    Args:
        config: 配置对象
    
    Returns:
        创建的模型
    """
    if config.USE_SMP and SMP_AVAILABLE:
        # 使用 segmentation-models-pytorch
        model = smp.Unet(
            encoder_name=config.BACKBONE,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=config.NUM_CLASSES,
        )
        print(f"使用 smp.Unet，backbone: {config.BACKBONE}")
    else:
        # 使用手动实现的 U-Net
        model = UNet(n_channels=3, n_classes=config.NUM_CLASSES, bilinear=True)
        print("使用手动实现的 U-Net")
    
    return model


# ==================== 损失函数 ====================
def create_criterion(config: Config, train_loader: DataLoader) -> nn.Module:
    """
    创建损失函数，支持类别权重
    
    Args:
        config: 配置对象
        train_loader: 训练数据加载器，用于计算类别权重
    
    Returns:
        损失函数
    """
    if config.USE_CLASS_WEIGHTS:
        if config.CLASS_WEIGHTS is not None:
            # 使用预定义的类别权重
            class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32)
        else:
            # 自动计算类别权重（基于像素计数，避免内存溢出）
            print("正在计算类别权重...")
            class_pixel_counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
            total_pixels = 0
            
            # 最多统计 50 个批次即可收敛，避免遍历整个数据集
            max_batches = min(50, len(train_loader))
            for i, (_, labels) in enumerate(tqdm(train_loader, desc="统计类别分布", total=max_batches)):
                if i >= max_batches:
                    break
                labels_flat = labels.flatten().numpy()
                # 使用 bincount 高效统计
                counts = np.bincount(labels_flat, minlength=config.NUM_CLASSES)
                class_pixel_counts += counts
                total_pixels += len(labels_flat)
            
            # 使用 balanced 策略计算权重: weight = total / (n_classes * count)
            # 为避免除零，给未出现类别赋予中等权重
            weights = np.zeros(config.NUM_CLASSES, dtype=np.float32)
            for c in range(config.NUM_CLASSES):
                if class_pixel_counts[c] > 0:
                    weights[c] = total_pixels / (config.NUM_CLASSES * class_pixel_counts[c])
                else:
                    weights[c] = 1.0  # 未出现类别赋予默认权重
            
            class_weights = torch.tensor(weights, dtype=torch.float32)
            print(f"自动计算的类别权重: {weights}")
            print(f"各类别像素数: {class_pixel_counts}")
        
        class_weights = class_weights.to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion


# ==================== 评价指标 ====================
class MetricsCalculator:
    """语义分割评价指标计算器"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        更新混淆矩阵
        
        Args:
            pred: 预测结果，形状为 (H, W) 或 (N, H, W)
            target: 真实标签，形状与 pred 相同
        """
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype(np.int64) + pred[mask].astype(np.int64)
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
    
    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """
        计算每个类别的 IoU 和平均 IoU
        
        Returns:
            ious: 每个类别的 IoU
            miou: 平均 IoU
        """
        # 计算每个类别的 IoU
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                 self.confusion_matrix.sum(axis=0) - 
                 np.diag(self.confusion_matrix))
        
        # 避免除零
        ious = np.where(union > 0, intersection / union, 0)
        miou = np.mean(ious)
        
        return ious, miou
    
    def compute_pixel_accuracy(self) -> float:
        """计算总体像素精度"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0
    
    def compute_class_accuracy(self) -> Tuple[np.ndarray, float]:
        """计算每个类别的精度和平均精度"""
        correct = np.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(axis=1)
        accs = np.where(total > 0, correct / total, 0)
        macc = np.mean(accs)
        return accs, macc


def evaluate_model(model: nn.Module, dataloader: DataLoader, 
                   device: str, num_classes: int) -> Dict[str, float]:
    """
    在验证集/测试集上评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数
    
    Returns:
        包含各项指标的字典
    """
    model.eval()
    metrics = MetricsCalculator(num_classes)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="验证中"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            # 获取预测结果
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            
            # 更新指标
            for pred, target in zip(preds, targets):
                metrics.update(pred, target)
    
    # 计算指标
    ious, miou = metrics.compute_iou()
    pixel_acc = metrics.compute_pixel_accuracy()
    class_accs, macc = metrics.compute_class_accuracy()
    avg_loss = total_loss / len(dataloader.dataset)
    
    results = {
        'loss': avg_loss,
        'miou': miou,
        'pixel_acc': pixel_acc,
        'macc': macc,
        'ious': ious,
        'class_accs': class_accs
    }
    
    return results


# ==================== 训练循环 ====================
def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: str) -> float:
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    
    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="训练中")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def train_model(config: Config):
    """
    完整的训练流程
    
    Args:
        config: 配置对象
    """
    # 设置随机种子
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # 创建输出目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # 保存配置
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(os.path.join(config.OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print("=" * 60)
    print("U-Net 语义分割训练")
    print("=" * 60)
    print(f"设备: {config.DEVICE}")
    print(f"类别数: {config.NUM_CLASSES}")
    print(f"输入尺寸: {config.INPUT_WIDTH}x{config.INPUT_HEIGHT}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"学习率: {config.LEARNING_RATE}")
    print("=" * 60)
    
    # 数据增强配置
    aug_config = {
        'flip_prob': 0.5,
        'rotation_degrees': config.ROTATION_DEGREES,
        'translation_ratio': config.TRANSLATION_RATIO,
        'scale_range': config.SCALE_RANGE,
        'color_jitter': config.COLOR_JITTER
    }
    
    # 划分数据集
    if hasattr(config, 'DATASET_DIRS') and len(config.DATASET_DIRS) > 0:
        # 使用 DATASET_DIRS 自动划分
        train_files, val_files, test_files = split_dataset(
            config.DATASET_DIRS, 
            ratios=config.SPLIT_RATIOS, 
            seed=config.SEED
        )
        
        train_dataset = SegmentationDataset(
            file_list=train_files,
            input_width=config.INPUT_WIDTH,
            input_height=config.INPUT_HEIGHT,
            is_training=True,
            augmentation_config=aug_config
        )
        
        val_dataset = SegmentationDataset(
            file_list=val_files,
            input_width=config.INPUT_WIDTH,
            input_height=config.INPUT_HEIGHT,
            is_training=False
        )
        
        test_dataset = None
        if len(test_files) > 0:
            test_dataset = SegmentationDataset(
                file_list=test_files,
                input_width=config.INPUT_WIDTH,
                input_height=config.INPUT_HEIGHT,
                is_training=False
            )
    else:
        # 使用预划分的 train/val/test 目录
        train_dataset = SegmentationDataset(
            image_dir=config.TRAIN_IMAGE_DIR,
            label_dir=config.TRAIN_LABEL_DIR,
            input_width=config.INPUT_WIDTH,
            input_height=config.INPUT_HEIGHT,
            is_training=True,
            augmentation_config=aug_config
        )
        
        val_dataset = SegmentationDataset(
            image_dir=config.VAL_IMAGE_DIR,
            label_dir=config.VAL_LABEL_DIR,
            input_width=config.INPUT_WIDTH,
            input_height=config.INPUT_HEIGHT,
            is_training=False
        )
        test_dataset = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 创建模型
    model = create_model(config)
    model = model.to(config.DEVICE)
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建损失函数
    criterion = create_criterion(config, train_loader)
    
    # 创建优化器
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    
    # 创建学习率调度器
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA
            )
        elif config.SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS
            )
        elif config.SCHEDULER_TYPE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )
    
    # 训练记录
    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_pixel_acc': []
    }
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch [{epoch}/{config.NUM_EPOCHS}]")
        print("-" * 60)
        
        # 训练阶段
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # 验证阶段
        if epoch % config.VALIDATION_INTERVAL == 0:
            val_results = evaluate_model(model, val_loader, config.DEVICE, config.NUM_CLASSES)
            val_loss = val_results['loss']
            val_miou = val_results['miou']
            val_pixel_acc = val_results['pixel_acc']
            
            # 更新学习率
            if scheduler is not None:
                if config.SCHEDULER_TYPE == 'plateau':
                    scheduler.step(val_miou)
                else:
                    scheduler.step()
            
            # 记录到 TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/mIoU', val_miou, epoch)
            writer.add_scalar('Metrics/pixel_acc', val_pixel_acc, epoch)
            
            # 记录每个类别的 IoU
            for i, iou in enumerate(val_results['ious']):
                writer.add_scalar(f'IoU/class_{i}_{config.CLASS_NAMES[i]}', iou, epoch)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证 mIoU: {val_miou:.4f}")
            print(f"验证像素精度: {val_pixel_acc:.4f}")
            print(f"各类别 IoU:")
            for i, (name, iou) in enumerate(zip(config.CLASS_NAMES, val_results['ious'])):
                print(f"  {name}: {iou:.4f}")
            
            # 保存历史记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_miou'].append(val_miou)
            history['val_pixel_acc'].append(val_pixel_acc)
            
            # 保存最佳模型
            if val_miou > best_miou:
                best_miou = val_miou
                best_epoch = epoch
                patience_counter = 0
                
                best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_iou_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': val_miou,
                    'config': config_dict
                }, best_model_path)
                print(f"✓ 保存最佳模型 (mIoU: {val_miou:.4f}) 到 {best_model_path}")
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if epoch % config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': val_miou,
                    'config': config_dict
                }, checkpoint_path)
                print(f"✓ 保存检查点到 {checkpoint_path}")
            
            # 早停检查
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n早停触发！连续 {config.EARLY_STOPPING_PATIENCE} 个 epoch 未提升")
                break
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch 耗时: {epoch_time:.2f}s")
    
    # 训练结束
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"总耗时: {total_time / 3600:.2f} 小时")
    print(f"最佳 mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    print("=" * 60)
    
    # 保存训练历史
    history_path = os.path.join(config.OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 关闭 TensorBoard writer
    writer.close()
    
    return model, history, test_dataset


# ==================== 测试函数 ====================
def test_model(config: Config, model_path: Optional[str] = None, 
               test_dataset: Optional[SegmentationDataset] = None):
    """
    在测试集上评估模型
    
    Args:
        config: 配置对象
        model_path: 模型权重路径，如果为 None 则使用最佳模型
        test_dataset: 测试数据集（可选），如果为 None 则尝试从配置创建
    """
    if model_path is None:
        model_path = os.path.join(config.CHECKPOINT_DIR, 'best_iou_model.pth')
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    print("=" * 60)
    print("测试模型")
    print("=" * 60)
    
    # 创建测试数据集
    if test_dataset is None:
        if hasattr(config, 'DATASET_DIRS') and len(config.DATASET_DIRS) > 0:
            _, _, test_files = split_dataset(
                config.DATASET_DIRS, 
                ratios=config.SPLIT_RATIOS, 
                seed=config.SEED
            )
            if len(test_files) == 0:
                print("测试集为空，跳过测试")
                return
            test_dataset = SegmentationDataset(
                file_list=test_files,
                input_width=config.INPUT_WIDTH,
                input_height=config.INPUT_HEIGHT,
                is_training=False
            )
        else:
            if not os.path.exists(config.TEST_IMAGE_DIR):
                print(f"测试集不存在: {config.TEST_IMAGE_DIR}")
                return
            test_dataset = SegmentationDataset(
                image_dir=config.TEST_IMAGE_DIR,
                label_dir=config.TEST_LABEL_DIR,
                input_width=config.INPUT_WIDTH,
                input_height=config.INPUT_HEIGHT,
                is_training=False
            )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"测试样本数: {len(test_dataset)}")
    
    # 创建模型
    model = create_model(config)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    
    print(f"加载模型: {model_path}")
    
    # 评估
    results = evaluate_model(model, test_loader, config.DEVICE, config.NUM_CLASSES)
    
    print("\n测试结果:")
    print(f"损失: {results['loss']:.4f}")
    print(f"mIoU: {results['miou']:.4f}")
    print(f"像素精度: {results['pixel_acc']:.4f}")
    print(f"平均类别精度: {results['macc']:.4f}")
    print("\n各类别 IoU:")
    for i, (name, iou, acc) in enumerate(zip(config.CLASS_NAMES, results['ious'], results['class_accs'])):
        print(f"  {name}: IoU={iou:.4f}, Acc={acc:.4f}")
    
    # 保存测试结果
    test_results = {
        'model_path': model_path,
        'loss': results['loss'],
        'miou': results['miou'],
        'pixel_acc': results['pixel_acc'],
        'macc': results['macc'],
        'class_results': [
            {'name': name, 'iou': float(iou), 'acc': float(acc)}
            for name, iou, acc in zip(config.CLASS_NAMES, results['ious'], results['class_accs'])
        ]
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n测试结果已保存到: {results_path}")


# ==================== 主函数 ====================
def main():
    """主函数"""
    config = Config()
    
    # 训练模型
    model, history, test_dataset = train_model(config)
    
    # 测试模型（如果测试集存在）
    if test_dataset is not None and len(test_dataset) > 0:
        test_model(config, test_dataset=test_dataset)
    
    print("\n全部完成!")


if __name__ == "__main__":
    main()

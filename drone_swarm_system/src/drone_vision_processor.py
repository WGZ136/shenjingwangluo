"""
无人机视觉处理器

封装功能:
- ESP32图像获取（0.1秒两帧）
- 缓存4组图片（4*2=8帧）
- 深度估计（每组后一张或立体匹配）
- 光流计算（每组两帧之间）
- 点云生成

适用于实体ESP32无人机:
    - 两个ESP32-CAM模块（前后分布）
    - 每个ESP32两个摄像头（0°和180°）
    - 共4个摄像头，每个采集2帧（间隔0.1秒）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import time
from datetime import datetime
import json
import sys
import os
from typing import TYPE_CHECKING

import numpy as np
import cv2

# 类型检查导入 (用于Pylance静态分析)
if TYPE_CHECKING:
    try:
        from monodepth2.networks import ResnetEncoder, DepthDecoder
        from monodepth2.layers import disp_to_depth
        from monodepth2.utils import download_model_if_doesnt_exist
        import open3d as o3d
    except ImportError:
        pass

# 尝试导入Monodepth2
MONODEPTH2_AVAILABLE = False
ResnetEncoder = None
DepthDecoder = None
disp_to_depth = None
download_model_if_doesnt_exist = None

try:
    # 添加Monodepth2路径
    monodepth_path = Path(__file__).parent.parent.parent / "core_algorithms"
    sys.path.insert(0, str(monodepth_path))
    
    from monodepth2.networks import ResnetEncoder, DepthDecoder
    from monodepth2.layers import disp_to_depth
    from monodepth2.utils import download_model_if_doesnt_exist
    
    MONODEPTH2_AVAILABLE = True
    print("[OK] Monodepth2 模块可用")
except ImportError as e:
    print(f"[WARN]  Monodepth2 模块不可用: {e}")
    print("   将使用传统算法作为备用")

# 尝试导入 Open3D
OPEN3D_AVAILABLE = False
o3d = None

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("[WARN] Open3D 未安装，3D可视化功能不可用")
    print("   安装命令: pip install open3d==0.18.0")


class OutputManager:
    """管理输出文件的有序存储"""
    
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / self.session_id
        
        # 创建目录结构
        self.dirs = {
            'raw': self.session_dir / "01_raw",           # 原始图像
            'front': self.session_dir / "02_front",       # 前视处理
            'back': self.session_dir / "03_back",         # 后视处理
            'merged': self.session_dir / "04_merged",     # 合并全景
            'flow': self.session_dir / "05_flow",         # 光流结果
            'depth': self.session_dir / "06_depth",       # 深度图
            'report': self.session_dir / "07_report",     # 报告
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[DIR] 输出目录: {self.session_dir}")
    
    def get_path(self, category: str, filename: str) -> Path:
        """获取文件路径"""
        return self.dirs[category] / filename
    
    def save_json(self, category: str, filename: str, data: dict):
        """保存JSON数据"""
        filepath = self.dirs[category] / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath


@dataclass
class FrameGroup:
    """
    单组帧数据（一个摄像头的一次采集）
    
    Attributes:
        camera_id: 摄像头ID (0-3)
        timestamp: 采集时间戳
        frame_0: 第一帧（t=0）
        frame_1: 第二帧（t=0.1s）
        flow: 光流场 (frame_0 -> frame_1)
        depth: 深度图（从frame_1计算）
        motion_vector: 整体运动向量 [dx, dy]
    """
    camera_id: int
    timestamp: float
    frame_0: Optional[np.ndarray] = None
    frame_1: Optional[np.ndarray] = None
    flow: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    motion_vector: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def is_complete(self) -> bool:
        """检查组是否完整"""
        return self.frame_0 is not None and self.frame_1 is not None
    
    def get_size(self) -> Tuple[int, int]:
        """获取图像尺寸"""
        if self.frame_0 is not None:
            h, w = self.frame_0.shape[:2]
            return (w, h)
        return (0, 0)


@dataclass
class DroneVisionProcessor:
    """
    无人机视觉处理器
    
    管理4个摄像头的图像采集和处理:
    - Camera 0: ESP32_Front 摄像头0 (0°方向)
    - Camera 1: ESP32_Front 摄像头1 (180°方向)
    - Camera 2: ESP32_Back 摄像头0 (0°方向)
    - Camera 3: ESP32_Back 摄像头1 (180°方向)
    
    每个摄像头采集2帧（间隔0.1秒），共8帧
    """
    
    # 配置参数
    num_cameras: int = 4
    frames_per_camera: int = 2
    capture_interval: float = 0.1  # 帧间隔（秒）
    
    # 数据存储
    groups: List[FrameGroup] = field(default_factory=list)
    point_cloud: Optional[np.ndarray] = None
    
    # 处理状态
    is_initialized: bool = False
    session_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # 光流参数
    flow_params: Dict = field(default_factory=lambda: {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    })
    
    # 深度估计参数
    stereo_params: Dict = field(default_factory=lambda: {
        'num_disparities': 16 * 8,  # 视差范围
        'block_size': 11,           # 匹配块大小
        'min_disparity': 0
    })
    
    # 神经网络开关（默认启用）
    use_neural_network: bool = True
    force_neural_network: bool = True  # 强制使用神经网络（失败时报错而非回退）
    depth_model = None  # 神经网络深度模型
    neural_network_loaded: bool = False  # 模型是否成功加载
    
    def __post_init__(self):
        """初始化后创建空组"""
        if not self.groups:
            self.groups = [
                FrameGroup(camera_id=i, timestamp=time.time())
                for i in range(self.num_cameras)
            ]
        self.is_initialized = True
        print(f"[OK] DroneVisionProcessor 初始化完成")
        print(f"   配置: {self.num_cameras}摄像头 × {self.frames_per_camera}帧")
        print(f"   神经网络: {'启用' if self.use_neural_network else '禁用'} (按 Q 切换)")
        
        # 如果启用神经网络，自动加载模型
        if self.use_neural_network:
            print("\n[NN] 正在加载神经网络模型...")
            success = self._load_depth_model()
            if not success and self.force_neural_network:
                raise RuntimeError("[ERR] 神经网络加载失败且强制启用模式开启，无法继续")
            if success:
                print("[OK] 神经网络深度估计已就绪")
    
    # ==================== 神经网络控制 ====================
    
    def toggle_neural_network(self) -> bool:
        """
        切换神经网络开关
        
        Returns:
            当前开关状态
        """
        self.use_neural_network = not self.use_neural_network
        
        if self.use_neural_network:
            self._load_depth_model()
            print("[NN] 神经网络已启用")
        else:
            self._unload_depth_model()
            print("[TRAD] 传统算法模式")
        
        return self.use_neural_network
    
    def _load_depth_model(self) -> bool:
        """
        加载Monodepth2神经网络深度估计模型
        
        Returns:
            bool: 是否成功加载
        """
        if self.depth_model is not None:
            return True
        
        if not MONODEPTH2_AVAILABLE:
            error_msg = "Monodepth2 模块不可用"
            print(f"  [ERR] {error_msg}")
            print(f"     请检查 core_algorithms/monodepth2 目录是否存在")
            if self.force_neural_network:
                raise RuntimeError(error_msg)
            self.use_neural_network = False
            return False
        
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            import traceback
            
            print("  正在加载 Monodepth2 模型...")
            
            # 模型配置（使用本地已下载的模型）
            # 模型文件位于 core_algorithms/mono+stereo_640x192/
            model_path = str(Path(__file__).parent.parent.parent / "core_algorithms" / "mono+stereo_640x192")
            self.model_path = model_path
            self.input_width = 640
            self.input_height = 192
            self.min_depth = 0.1
            self.max_depth = 100.0
            self.scale_factor = 5.4
            print(f"    模型路径: {model_path}")
            
            # 设备选择
            self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"    使用设备: {self.torch_device}")
            
            # 检查模型文件是否存在
            print(f"    检查模型文件...")
            encoder_path = Path(model_path) / "encoder.pth"
            decoder_path = Path(model_path) / "depth.pth"
            
            if not encoder_path.exists():
                raise FileNotFoundError(f"编码器权重不存在: {encoder_path}")
            if not decoder_path.exists():
                raise FileNotFoundError(f"解码器权重不存在: {decoder_path}")
            
            print(f"    [v] 找到 encoder.pth")
            print(f"    [v] 找到 depth.pth")
            
            # 加载编码器
            print("    加载编码器...")
            encoder = ResnetEncoder(18, False)
            encoder_path = Path(model_path) / "encoder.pth"
            if not encoder_path.exists():
                raise FileNotFoundError(f"编码器权重文件不存在: {encoder_path}")
            encoder_weights = torch.load(encoder_path, map_location=self.torch_device)
            encoder.load_state_dict({k: v for k, v in encoder_weights.items() 
                                     if k in encoder.state_dict()})
            encoder.to(self.torch_device)
            encoder.eval()
            print("    [v] 编码器加载成功")
            
            # 加载深度解码器
            print("    加载解码器...")
            depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
            decoder_path = Path(model_path) / "depth.pth"
            if not decoder_path.exists():
                raise FileNotFoundError(f"解码器权重文件不存在: {decoder_path}")
            decoder_weights = torch.load(decoder_path, map_location=self.torch_device)
            depth_decoder.load_state_dict(decoder_weights)
            depth_decoder.to(self.torch_device)
            depth_decoder.eval()
            print("    [v] 解码器加载成功")
            
            # 保存模型
            self.depth_model = {
                'encoder': encoder,
                'decoder': depth_decoder
            }
            
            # 预处理变换
            self.depth_transform = transforms.Compose([
                transforms.Resize((self.input_height, self.input_width), 
                                interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.neural_network_loaded = True
            print("  [OK] Monodepth2 模型加载成功")
            return True
            
        except Exception as e:
            error_msg = f"模型加载失败: {e}"
            print(f"  [ERR] {error_msg}")
            print(f"\n{'='*60}")
            print("详细错误信息:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            
            if self.force_neural_network:
                raise RuntimeError(error_msg) from e
            
            print("  [WARN]  将回退到传统算法")
            self.use_neural_network = False
            self.depth_model = None
            self.neural_network_loaded = False
            return False
    
    def _unload_depth_model(self):
        """卸载神经网络模型"""
        if self.depth_model is not None:
            print("  正在释放模型资源...")
            import torch
            if isinstance(self.depth_model, dict):
                self.depth_model['encoder'].cpu()
                self.depth_model['decoder'].cpu()
            self.depth_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def set_neural_network(self, enabled: bool):
        """设置神经网络开关状态"""
        if enabled != self.use_neural_network:
            self.toggle_neural_network()
    
    # ==================== 图像采集 ====================
    
    def capture_from_esp32(self, camera_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从指定ESP32摄像头采集2帧
        
        Args:
            camera_id: 摄像头ID (0-3)
            
        Returns:
            (frame_0, frame_1): 两帧图像，间隔0.1秒
            
        TODO: 替换为实际的ESP32 HTTP请求
        """
        # 模拟：使用本地摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"[ERR] 摄像头{camera_id}无法打开")
            return None, None
        
        # 第一帧
        ret0, frame_0 = cap.read()
        if not ret0:
            cap.release()
            return None, None
        
        # 等待0.1秒
        time.sleep(self.capture_interval)
        
        # 第二帧
        ret1, frame_1 = cap.read()
        
        cap.release()
        
        if ret0 and ret1:
            print(f"  [v] Camera {camera_id}: 采集2帧 ({frame_0.shape[1]}x{frame_0.shape[0]})")
            return frame_0, frame_1
        else:
            print(f"  ✗ Camera {camera_id}: 采集失败")
            return None, None
    
    def capture_all_cameras(self) -> bool:
        """
        采集所有4个摄像头的图像
        
        Returns:
            True if all cameras captured successfully
        """
        print(f"\n📸 开始采集 ({self.num_cameras}摄像头 × {self.frames_per_camera}帧)")
        print(f"   帧间隔: {self.capture_interval}s")
        print("-" * 50)
        
        success_count = 0
        
        for i in range(self.num_cameras):
            frame_0, frame_1 = self.capture_from_esp32(i)
            
            if frame_0 is not None and frame_1 is not None:
                self.groups[i].frame_0 = frame_0
                self.groups[i].frame_1 = frame_1
                self.groups[i].timestamp = time.time()
                success_count += 1
            
            # 摄像头间短暂间隔
            time.sleep(0.05)
        
        print("-" * 50)
        print(f"[OK] 采集完成: {success_count}/{self.num_cameras} 摄像头")
        
        return success_count == self.num_cameras
    
    # ==================== 光流计算 ====================
    
    def compute_optical_flow(self, group_idx: int) -> Optional[np.ndarray]:
        """
        计算指定组的光流
        
        Args:
            group_idx: 组索引 (0-3)
            
        Returns:
            flow: 光流场 (H, W, 2)
        """
        group = self.groups[group_idx]
        
        if not group.is_complete():
            print(f"[WARN]  Camera {group_idx}: 帧不完整，无法计算光流")
            return None
        
        # 转换为灰度
        gray_0 = cv2.cvtColor(group.frame_0, cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.cvtColor(group.frame_1, cv2.COLOR_BGR2GRAY)
        
        # 计算Farneback光流
        flow = cv2.calcOpticalFlowFarneback(
            gray_0, gray_1, None,
            self.flow_params['pyr_scale'],
            self.flow_params['levels'],
            self.flow_params['winsize'],
            self.flow_params['iterations'],
            self.flow_params['poly_n'],
            self.flow_params['poly_sigma'],
            self.flow_params['flags']
        )
        
        group.flow = flow
        
        # 计算运动向量
        group.motion_vector = np.median(flow.reshape(-1, 2), axis=0)
        
        print(f"  [v] Camera {group_idx}: 光流计算完成")
        print(f"    运动向量: [{group.motion_vector[0]:.2f}, {group.motion_vector[1]:.2f}]")
        
        return flow
    
    def compute_all_flows(self) -> bool:
        """计算所有组的光流"""
        print("\n🌊 计算光流...")
        
        success = 0
        for i in range(self.num_cameras):
            if self.compute_optical_flow(i) is not None:
                success += 1
        
        print(f"[OK] 光流计算: {success}/{self.num_cameras} 完成")
        return success > 0
    
    # ==================== 深度估计 ====================
    
    def estimate_depth_single(self, group_idx: int) -> Optional[np.ndarray]:
        """
        单目深度估计
        
        根据 use_neural_network 开关选择:
        - True: 使用神经网络 (Monodepth2)
        - False: 使用传统算法 (梯度/立体匹配)
        
        Args:
            group_idx: 组索引
            
        Returns:
            depth: 深度图 (H, W)
        """
        group = self.groups[group_idx]
        
        if group.frame_1 is None:
            return None
        
        if self.use_neural_network and self.depth_model is not None:
            # 使用神经网络
            depth = self._estimate_depth_neural(group.frame_1)
            method = "神经网络"
        else:
            # 使用传统算法
            depth = self._estimate_depth_traditional(group.frame_1)
            method = "传统算法"
        
        group.depth = depth
        
        print(f"  [v] Camera {group_idx}: 深度估计完成 ({method})")
        return depth
    
    def _estimate_depth_traditional(self, frame: np.ndarray) -> np.ndarray:
        """传统深度估计算法（基于梯度）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # 反转：梯度大（边缘）= 距离近 = 深度小
        depth = 255 - cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 应用高斯模糊平滑
        depth = cv2.GaussianBlur(depth, (5, 5), 0)
        
        return depth
    
    def _estimate_depth_neural(self, frame: np.ndarray) -> np.ndarray:
        """
        使用 Monodepth2 进行神经网络深度估计
        
        Args:
            frame: BGR格式的输入图像
            
        Returns:
            depth: 深度图 (H, W)
            
        Raises:
            RuntimeError: 如果强制启用神经网络但模型未加载或推理失败
        """
        if self.depth_model is None:
            error_msg = "神经网络模型未加载"
            if self.force_neural_network:
                raise RuntimeError(f"[ERR] {error_msg}，且强制启用模式开启")
            print(f"    ({error_msg}，使用传统算法)")
            return self._estimate_depth_traditional(frame)
        
        try:
            import torch
            from PIL import Image
            
            # 保存原始尺寸
            original_h, original_w = frame.shape[:2]
            
            # BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 预处理
            input_tensor = self.depth_transform(pil_image).unsqueeze(0).to(self.torch_device)
            
            # 前向传播
            with torch.no_grad():
                encoder = self.depth_model['encoder']
                decoder = self.depth_model['decoder']
                
                features = encoder(input_tensor)
                outputs = decoder(features)
                
                # 获取视差图并转换为深度
                disp = outputs[("disp", 0)]
                _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
                depth = depth * self.scale_factor
                depth = torch.clamp(depth, self.min_depth, self.max_depth)
                
                # 转换为 numpy
                depth_np = depth.squeeze().cpu().numpy()
            
            # 调整回原始尺寸
            if original_w != self.input_width or original_h != self.input_height:
                depth_resized = cv2.resize(depth_np, (original_w, original_h), 
                                          interpolation=cv2.INTER_LINEAR)
            else:
                depth_resized = depth_np
            
            # 归一化到 0-255 用于显示
            depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            return depth_uint8
            
        except Exception as e:
            error_msg = f"神经网络推理失败: {e}"
            if self.force_neural_network:
                import traceback
                print(f"\n{'='*60}")
                print("神经网络推理错误详情:")
                traceback.print_exc()
                print(f"{'='*60}\n")
                raise RuntimeError(f"[ERR] {error_msg}") from e
            
            print(f"    ({error_msg}，使用传统算法)")
            return self._estimate_depth_traditional(frame)
    
    def estimate_stereo_depth(self, left_idx: int = 0, right_idx: int = 2) -> Optional[np.ndarray]:
        """
        立体深度估计（使用前视和后视摄像头）
        
        使用StereoBM计算视差图
        
        Args:
            left_idx: 左摄像头索引（默认Camera 0，前视0°）
            right_idx: 右摄像头索引（默认Camera 2，后视0°）
            
        Returns:
            depth: 深度图
        """
        left_group = self.groups[left_idx]
        right_group = self.groups[right_idx]
        
        if left_group.frame_1 is None or right_group.frame_1 is None:
            print(f"[WARN]  立体深度: 帧不完整")
            return None
        
        # 转换为灰度
        left_gray = cv2.cvtColor(left_group.frame_1, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_group.frame_1, cv2.COLOR_BGR2GRAY)
        
        # 确保尺寸相同
        if left_gray.shape != right_gray.shape:
            h, w = left_gray.shape
            right_gray = cv2.resize(right_gray, (w, h))
        
        # 创建StereoBM
        stereo = cv2.StereoBM_create(
            numDisparities=self.stereo_params['num_disparities'],
            blockSize=self.stereo_params['block_size']
        )
        stereo.setMinDisparity(self.stereo_params['min_disparity'])
        
        # 计算视差
        disparity = stereo.compute(left_gray, right_gray)
        
        # 归一化
        disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 转换为深度（视差大 = 深度小）
        depth = 255 - disparity_vis
        
        print(f"[OK] 立体深度估计完成 (Camera {left_idx} & {right_idx})")
        
        return depth
    
    def estimate_all_depths(self) -> bool:
        """估计所有摄像头的深度"""
        print("\n📏 深度估计...")
        
        success = 0
        for i in range(self.num_cameras):
            if self.estimate_depth_single(i) is not None:
                success += 1
        
        # 尝试立体深度
        stereo_depth = self.estimate_stereo_depth(0, 2)
        if stereo_depth is not None:
            print("  [v] 立体深度计算成功")
        
        print(f"[OK] 深度估计: {success}/{self.num_cameras} 单目 + 立体")
        return success > 0
    
    # ==================== 点云生成 ====================
    
    def generate_point_cloud(self, camera_idx: int = 0) -> Optional[np.ndarray]:
        """
        从深度图生成点云
        
        Args:
            camera_idx: 使用哪个摄像头的深度图
            
        Returns:
            point_cloud: N×3数组 (x, y, z)
        """
        group = self.groups[camera_idx]
        
        if group.depth is None or group.frame_1 is None:
            print(f"[WARN]  无法生成点云: Camera {camera_idx} 数据不完整")
            return None
        
        h, w = group.depth.shape
        color = group.frame_1
        
        # 相机内参（默认值）
        fx = fy = 500.0  # 焦距
        cx, cy = w / 2, h / 2  # 主点
        
        # 创建网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 深度值归一化到0-10米
        z = group.depth.astype(np.float32) / 255.0 * 10.0 + 0.1
        
        # 反投影到3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # 堆叠为N×3
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # 添加颜色信息
        colors = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        
        # 过滤掉无效点（太远或太近）
        valid_mask = (z.flatten() > 0.2) & (z.flatten() < 10.0)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # 下采样（减少点数）
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # 合并点和颜色
        point_cloud = np.hstack([points, colors])
        
        self.point_cloud = point_cloud
        
        print(f"[OK] 点云生成完成: {len(point_cloud)} 点 (Camera {camera_idx})")
        
        return point_cloud
    
    def visualize_point_cloud_3d(self, camera_idx: int = 0, window_name: str = "Point Cloud") -> bool:
        """
        使用 Open3D 可视化3D点云
        
        Args:
            camera_idx: 要可视化的摄像头索引
            window_name: 窗口标题
            
        Returns:
            bool: 是否成功显示
        """
        if not OPEN3D_AVAILABLE:
            print("[ERR] Open3D 未安装，无法显示3D视图")
            print("   安装命令: pip install open3d==0.18.0")
            return False
        
        group = self.groups[camera_idx]
        
        if group.depth is None or group.frame_1 is None:
            print(f"[ERR] 无法可视化: Camera {camera_idx} 数据不完整")
            return False
        
        print(f"\n[3D] 正在生成3D可视化 (Camera {camera_idx})...")
        print("   按 'Q' 或关闭窗口退出")
        
        try:
            import open3d as o3d
            
            # 生成点云
            self.generate_point_cloud(camera_idx)
            
            if self.point_cloud is None or len(self.point_cloud) == 0:
                print("[ERR] 点云为空")
                return False
            
            # 提取点和颜色
            points = self.point_cloud[:, :3]
            colors = self.point_cloud[:, 3:6] / 255.0  # 归一化到0-1
            
            # 创建 Open3D 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 创建坐标系
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            
            # 可视化
            o3d.visualization.draw_geometries(
                [pcd, coordinate_frame],
                window_name=window_name,
                width=1024,
                height=768,
                point_show_normal=False
            )
            
            print("[OK] 3D可视化完成")
            return True
            
        except Exception as e:
            print(f"[ERR] 3D可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== 可视化与保存 ====================
    
    def visualize_flow(self, group_idx: int) -> Optional[np.ndarray]:
        """可视化光流"""
        group = self.groups[group_idx]
        
        if group.flow is None or group.frame_1 is None:
            return None
        
        h, w = group.flow.shape[:2]
        
        # 计算幅值和角度
        mag, ang = cv2.cartToPolar(group.flow[..., 0], group.flow[..., 1])
        
        # 创建HSV
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 转换到BGR
        flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 叠加
        overlay = cv2.addWeighted(group.frame_1, 0.7, flow_color, 0.3, 0)
        
        return overlay
    
    def save_all(self, output_dir: Path):
        """保存所有结果到指定目录"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 保存结果到: {output_dir}")
        
        # 创建子目录
        (output_dir / "frames").mkdir(exist_ok=True)
        (output_dir / "flow").mkdir(exist_ok=True)
        (output_dir / "depth").mkdir(exist_ok=True)
        (output_dir / "pointcloud").mkdir(exist_ok=True)
        
        saved_count = 0
        
        for i, group in enumerate(self.groups):
            if not group.is_complete():
                continue
            
            # 保存原始帧
            cv2.imwrite(str(output_dir / "frames" / f"cam{i}_frame0.jpg"), group.frame_0)
            cv2.imwrite(str(output_dir / "frames" / f"cam{i}_frame1.jpg"), group.frame_1)
            saved_count += 2
            
            # 保存光流可视化
            if group.flow is not None:
                flow_vis = self.visualize_flow(i)
                if flow_vis is not None:
                    cv2.imwrite(str(output_dir / "flow" / f"cam{i}_flow.jpg"), flow_vis)
                    saved_count += 1
            
            # 保存深度图
            if group.depth is not None:
                cv2.imwrite(str(output_dir / "depth" / f"cam{i}_depth.jpg"), group.depth)
                saved_count += 1
        
        # 保存点云
        if self.point_cloud is not None:
            pc_path = output_dir / "pointcloud" / "pointcloud.txt"
            np.savetxt(str(pc_path), self.point_cloud, fmt='%.4f')
            saved_count += 1
            print(f"  [v] 点云已保存: {pc_path}")
        
        # 保存元数据
        metadata = {
            'session_id': self.session_timestamp,
            'num_cameras': self.num_cameras,
            'frames_per_camera': self.frames_per_camera,
            'capture_interval': self.capture_interval,
            'groups': [
                {
                    'camera_id': g.camera_id,
                    'timestamp': g.timestamp,
                    'has_flow': g.flow is not None,
                    'has_depth': g.depth is not None,
                    'motion_vector': g.motion_vector.tolist()
                }
                for g in self.groups
            ],
            'point_cloud_size': len(self.point_cloud) if self.point_cloud is not None else 0
        }
        
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] 保存完成: {saved_count} 文件")
        return output_dir
    
    # ==================== 处理流程 ====================
    
    def process_full_pipeline(self, output_dir: Optional[Path] = None) -> bool:
        """
        执行完整的处理流程
        
        1. 采集图像
        2. 计算光流
        3. 估计深度
        4. 生成点云
        5. 保存结果
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("🚀 开始完整视觉处理流程")
        print("="*60)
        
        # 1. 采集
        if not self.capture_all_cameras():
            print("[ERR] 图像采集失败")
            return False
        
        # 2. 光流
        self.compute_all_flows()
        
        # 3. 深度
        self.estimate_all_depths()
        
        # 4. 点云（使用Camera 0）
        self.generate_point_cloud(0)
        
        # 5. 保存
        if output_dir is not None:
            self.save_all(output_dir)
        
        print("\n" + "="*60)
        print("[OK] 处理流程完成")
        print("="*60)
        
        return True
    
    def get_summary(self) -> Dict:
        """获取处理摘要"""
        return {
            'session_id': self.session_timestamp,
            'num_cameras': self.num_cameras,
            'frames_per_camera': self.frames_per_camera,
            'complete_groups': sum(1 for g in self.groups if g.is_complete()),
            'has_flow': sum(1 for g in self.groups if g.flow is not None),
            'has_depth': sum(1 for g in self.groups if g.depth is not None),
            'has_pointcloud': self.point_cloud is not None,
            'point_cloud_size': len(self.point_cloud) if self.point_cloud is not None else 0
        }


# ==================== 便捷函数 ====================

def create_vision_processor(num_cameras: int = 4, 
                            frames_per_camera: int = 2,
                            capture_interval: float = 0.1,
                            use_neural_network: bool = True,
                            force_neural_network: bool = True) -> DroneVisionProcessor:
    """
    创建视觉处理器实例
    
    Args:
        num_cameras: 摄像头数量
        frames_per_camera: 每个摄像头的帧数
        capture_interval: 帧间隔（秒）
        use_neural_network: 是否使用神经网络
        force_neural_network: 是否强制使用神经网络（失败时报错）
    """
    processor = DroneVisionProcessor(
        num_cameras=num_cameras,
        frames_per_camera=frames_per_camera,
        capture_interval=capture_interval,
        use_neural_network=use_neural_network,
        force_neural_network=force_neural_network
    )
    return processor


# 测试代码
if __name__ == "__main__":
    print("测试 DroneVisionProcessor")
    print("-" * 60)
    
    # 创建处理器
    processor = create_vision_processor(num_cameras=2)  # 测试用2个摄像头
    
    # 运行完整流程
    output_path = Path("test_output")
    success = processor.process_full_pipeline(output_path)
    
    if success:
        print("\n处理摘要:")
        summary = processor.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

"""
系统配置模块

包含默认配置和配置管理功能
"""

import os
import sys
from pathlib import Path

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
CORE_ALGORITHMS_DIR = PROJECT_ROOT / "core_algorithms"
SRC_DIR = PROJECT_ROOT / "drone_swarm_system" / "src"

# 添加路径到sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_ALGORITHMS_DIR))
sys.path.insert(0, str(SRC_DIR))

# 检测CUDA是否可用
def check_cuda_available():
    """
    检查CUDA是否可用，处理各种异常情况
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "cpu"
        
        try:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            test_tensor = torch.tensor([1.0]).cuda()
            _ = test_tensor + 1
            del test_tensor
            return True, "cuda"
        except RuntimeError:
            return False, "cpu"
            
    except ImportError:
        return False, "cpu"
    except Exception:
        return False, "cpu"

CUDA_AVAILABLE, DEFAULT_DEVICE = check_cuda_available()

# 默认配置
DEFAULT_CONFIG = {
    'depth': {
        'enabled': True,
        'model_path': 'models/mono_640x192',
        'input_width': 640,
        'input_height': 192,
        'min_depth': 0.1,
        'max_depth': 100.0,
        'device': DEFAULT_DEVICE
    },
    'flow': {
        'enabled': True,
        'model_path': 'data/model_weights/raft-things.pth',
        'iterations': 20,
        'use_gpu': CUDA_AVAILABLE
    },
    'clustering': {
        'enabled': True,
        'method': 'kmeans',
        'n_clusters': 3,
        'eps': 0.5,
        'min_samples': 5
    },
    'pose': {
        'enabled': True,
        'method': 'ransac',
        'ransac_threshold': 0.01,
        'ransac_iterations': 1000
    },
    'visualization': {
        'enabled': True,
        'save_results': False,
        'output_dir': 'output',
        'show_display': False
    }
}


def merge_config(user_config=None):
    """合并用户配置和默认配置"""
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if user_config:
        deep_update(config, user_config)
    return config


def deep_update(base_dict, update_dict):
    """深度更新字典"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


class SystemConfig:
    """
    系统配置类
    
    用于配置DroneSwarmSystem的各项参数
    """
    
    def __init__(self,
                 enable_esp32_receiver: bool = False,
                 enable_3d_visualization: bool = False,
                 camera_intrinsics: dict = None,
                 depth_config: dict = None,
                 flow_config: dict = None):
        """
        初始化系统配置
        
        Args:
            enable_esp32_receiver: 是否启用ESP32图传接收
            enable_3d_visualization: 是否启用3D可视化
            camera_intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
            depth_config: 深度估计配置
            flow_config: 光流估计配置
        """
        self.enable_esp32_receiver = enable_esp32_receiver
        self.enable_3d_visualization = enable_3d_visualization
        
        # 相机内参
        self.camera_intrinsics = camera_intrinsics or {
            'fx': 320.0,
            'fy': 320.0,
            'cx': 320.0,
            'cy': 240.0
        }
        
        # 深度估计配置
        self.depth_config = depth_config or DEFAULT_CONFIG['depth']
        
        # 光流估计配置
        self.flow_config = flow_config or DEFAULT_CONFIG['flow']
        
        # 其他配置
        self.clustering_config = DEFAULT_CONFIG['clustering']
        self.pose_config = DEFAULT_CONFIG['pose']
        self.visualization_config = DEFAULT_CONFIG['visualization']
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'enable_esp32_receiver': self.enable_esp32_receiver,
            'enable_3d_visualization': self.enable_3d_visualization,
            'camera_intrinsics': self.camera_intrinsics,
            'depth': self.depth_config,
            'flow': self.flow_config,
            'clustering': self.clustering_config,
            'pose': self.pose_config,
            'visualization': self.visualization_config
        }

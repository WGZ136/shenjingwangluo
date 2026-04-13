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

"""
基于视觉感知的无人机集群协同演化系统

这是一个集成了深度估计、光流估计、聚类分割和位姿估计的无人机集群系统。

主要模块:
    - core: 核心系统类
    - modules: 功能模块（深度估计、光流、聚类等）
    - utils: 工具函数
    - scripts: 运行脚本
    - examples: 使用示例

使用示例:
    >>> from drone_swarm_system import DroneSwarmSystem
    >>> 
    >>> # 方式1: 使用主类
    >>> system = DroneSwarmSystem()
    >>> system.initialize()
    >>> result = system.process_frame(image)
    >>> 
    >>> # 方式2: 使用上下文管理器
    >>> with DroneSwarmSystem() as system:
    ...     result = system.process_frame(image)
"""

__version__ = "1.0.0"
__author__ = "Drone Swarm Team"

# 导入核心类和函数
from .core.system import (
    DroneSwarmSystem,
    ProcessingResult,
    create_system,
    quick_process,
    check_cuda_available,
    CUDA_AVAILABLE,
    DEFAULT_DEVICE,
)

# 导入配置
from .core.config import DEFAULT_CONFIG

# 定义公开接口
__all__ = [
    'DroneSwarmSystem',
    'ProcessingResult',
    'create_system',
    'quick_process',
    'check_cuda_available',
    'CUDA_AVAILABLE',
    'DEFAULT_DEVICE',
    'DEFAULT_CONFIG',
]

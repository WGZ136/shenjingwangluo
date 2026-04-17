"""
基于视觉感知的无人机集群协同演化系统

项目入口点 - 从drone_swarm_system包导入所有功能

使用示例:
    >>> from drone_swarm_system import DroneSwarmSystem
    >>> 
    >>> with DroneSwarmSystem() as system:
    ...     result = system.process_frame(image)
"""

__version__ = "1.0.0"

# 从包中导入所有公开接口
from drone_swarm_system import (
    DroneSwarmSystem,
    ProcessingResult,
    create_system,
    quick_process,
    check_cuda_available,
    CUDA_AVAILABLE,
    DEFAULT_DEVICE,
    DEFAULT_CONFIG,
)

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

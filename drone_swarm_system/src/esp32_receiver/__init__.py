"""
ESP32-CAM 图传接收模块

用于接收无人机双摄像头的视频流
- 前视摄像头 (端口 80)
- 下视摄像头 (端口 81)

基于另一位同学的代码整合

使用示例:
    from esp32_receiver import DualCameraReceiver
    
    receiver = DualCameraReceiver()
    receiver.start()
    
    # 获取帧
    front_frame = receiver.read("front")
    down_frame = receiver.read("down")
    
    # 显示
    receiver.display()
"""

from .camera_receiver import (
    CameraStream, 
    DualCameraReceiver, 
    CameraFrame,
    test_receiver
)
from .config import ESP32_CONFIG, STREAM_CONFIG, IMAGE_CONFIG

__all__ = [
    'CameraStream',
    'DualCameraReceiver',
    'CameraFrame',
    'ESP32_CONFIG',
    'STREAM_CONFIG',
    'IMAGE_CONFIG',
    'test_receiver'
]

__version__ = '1.0.0'

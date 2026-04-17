"""
ESP32-CAM 配置
"""

# 摄像头配置
ESP32_CONFIG = {
    "front_camera": {
        "name": "前视摄像头",
        "ip": "192.168.4.1",
        "port": 80,
        "stream_url": "http://192.168.4.1/stream",
        "capture_url": "http://192.168.4.1/capture",
        "wifi_ssid": "Drone_Cam_Front",
        "wifi_password": "12345678"
    },
    "down_camera": {
        "name": "下视摄像头", 
        "ip": "192.168.4.1",
        "port": 81,
        "stream_url": "http://192.168.4.1:81/stream",
        "capture_url": "http://192.168.4.1:81/capture",
        "wifi_ssid": "Drone_Cam_Rear",
        "wifi_password": "12345678"
    }
}

# 视频流配置
STREAM_CONFIG = {
    "chunk_size": 1024,
    "timeout": 5,
    "reconnect_delay": 2,
    "frame_rate": 10  # 控制帧率
}

# 图像配置
IMAGE_CONFIG = {
    "frame_size": (320, 240),  # QVGA
    "jpeg_quality": 12,        # 0-63, 越小越清晰
    "display_size": (640, 480)  # 显示大小
}

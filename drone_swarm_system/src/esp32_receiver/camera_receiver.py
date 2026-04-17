"""
ESP32-CAM 摄像头接收器

支持双摄像头视频流接收
"""

import cv2
import numpy as np
import requests
from threading import Thread, Lock
import time
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass

from .config import ESP32_CONFIG, STREAM_CONFIG


@dataclass
class CameraFrame:
    """摄像头帧数据"""
    image: np.ndarray
    timestamp: float
    camera_name: str
    is_valid: bool = True


class CameraStream:
    """
    单摄像头视频流接收器
    
    使用多线程接收ESP32-CAM的MJPEG视频流
    """
    
    def __init__(self, name: str, config: Dict, callback: Optional[Callable] = None):
        """
        初始化摄像头流
        
        Args:
            name: 摄像头名称
            config: 配置字典
            callback: 新帧回调函数(frame: CameraFrame)
        """
        self.name = name
        self.stream_url = config["stream_url"]
        self.capture_url = config["capture_url"]
        self.callback = callback
        
        # 帧数据
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._lock = Lock()
        
        # 控制标志
        self._stopped = True
        self._connected = False
        self._thread: Optional[Thread] = None
        
        # 统计信息
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        
    def start(self) -> 'CameraStream':
        """启动视频流接收"""
        if not self._stopped:
            return self
            
        self._stopped = False
        self._thread = Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return self
    
    def _update_loop(self):
        """视频流接收主循环"""
        print(f"🔌 [{self.name}] 正在连接...")
        bytes_data = bytes()
        
        while not self._stopped:
            try:
                response = requests.get(
                    self.stream_url, 
                    stream=True, 
                    timeout=STREAM_CONFIG["timeout"]
                )
                
                if response.status_code == 200:
                    self._connected = True
                    print(f"✅ [{self.name}] 连接成功")
                    
                    for chunk in response.iter_content(
                        chunk_size=STREAM_CONFIG["chunk_size"]
                    ):
                        if self._stopped:
                            break
                            
                        bytes_data += chunk
                        
                        # 查找JPEG帧边界
                        start_idx = bytes_data.find(b'\xff\xd8')  # JPEG开始
                        end_idx = bytes_data.find(b'\xff\xd9')    # JPEG结束
                        
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            # 提取完整JPEG图像
                            jpg_data = bytes_data[start_idx:end_idx+2]
                            bytes_data = bytes_data[end_idx+2:]
                            
                            # 解码图像
                            img_array = np.frombuffer(jpg_data, dtype=np.uint8)
                            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                with self._lock:
                                    self._frame = frame
                                    self._timestamp = time.time()
                                    self._frame_count += 1
                                
                                # 计算FPS
                                current_time = time.time()
                                if current_time - self._last_fps_time >= 1.0:
                                    self._fps = self._frame_count
                                    self._frame_count = 0
                                    self._last_fps_time = current_time
                                
                                # 触发回调
                                if self.callback:
                                    camera_frame = CameraFrame(
                                        image=frame,
                                        timestamp=self._timestamp,
                                        camera_name=self.name
                                    )
                                    self.callback(camera_frame)
                                    
                else:
                    print(f"❌ [{self.name}] 连接失败: {response.status_code}")
                    self._connected = False
                    time.sleep(STREAM_CONFIG["reconnect_delay"])
                    
            except Exception as e:
                if self._connected:
                    print(f"⚠️ [{self.name}] 连接错误: {str(e)}")
                self._connected = False
                time.sleep(STREAM_CONFIG["reconnect_delay"])
    
    def read(self) -> Optional[np.ndarray]:
        """
        读取最新帧
        
        Returns:
            最新图像帧，如果没有则返回None
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
    
    def get_frame_info(self) -> Dict:
        """获取帧信息"""
        with self._lock:
            return {
                "connected": self._connected,
                "fps": self._fps,
                "frame_count": self._frame_count,
                "timestamp": self._timestamp
            }
    
    def capture_image(self) -> Optional[np.ndarray]:
        """
        拍摄单张照片
        
        Returns:
            捕获的图像
        """
        try:
            response = requests.get(
                self.capture_url, 
                timeout=STREAM_CONFIG["timeout"]
            )
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ [{self.name}] 拍照失败: {e}")
        return None
    
    def stop(self):
        """停止视频流接收"""
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._connected = False
        print(f"⏹️ [{self.name}] 已停止")
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected


class DualCameraReceiver:
    """
    双摄像头接收器
    
    同时管理前视和下视两个摄像头
    """
    
    def __init__(self, 
                 front_callback: Optional[Callable] = None,
                 down_callback: Optional[Callable] = None):
        """
        初始化双摄像头接收器
        
        Args:
            front_callback: 前视摄像头帧回调
            down_callback: 下视摄像头帧回调
        """
        self.cameras: Dict[str, CameraStream] = {}
        
        # 创建摄像头流
        self.cameras["front"] = CameraStream(
            ESP32_CONFIG["front_camera"]["name"],
            ESP32_CONFIG["front_camera"],
            front_callback
        )
        
        self.cameras["down"] = CameraStream(
            ESP32_CONFIG["down_camera"]["name"],
            ESP32_CONFIG["down_camera"],
            down_callback
        )
        
        print("🚁 双摄像头接收器初始化完成")
    
    def start(self):
        """启动所有摄像头"""
        for cam in self.cameras.values():
            cam.start()
        print("🎥 双摄像头接收已启动")
    
    def stop(self):
        """停止所有摄像头"""
        for cam in self.cameras.values():
            cam.stop()
        print("⏹️ 双摄像头接收已停止")
    
    def read(self, camera: str = "front") -> Optional[np.ndarray]:
        """
        读取指定摄像头的最新帧
        
        Args:
            camera: "front" 或 "down"
            
        Returns:
            图像帧
        """
        if camera in self.cameras:
            return self.cameras[camera].read()
        return None
    
    def read_both(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        同时读取两个摄像头的帧
        
        Returns:
            (前视帧, 下视帧)
        """
        return (
            self.cameras["front"].read(),
            self.cameras["down"].read()
        )
    
    def get_status(self) -> Dict:
        """获取所有摄像头状态"""
        return {
            name: cam.get_frame_info()
            for name, cam in self.cameras.items()
        }
    
    def capture_images(self) -> Dict[str, Optional[np.ndarray]]:
        """
        同时拍摄两个摄像头的照片
        
        Returns:
            {"front": 前视图像, "down": 下视图像}
        """
        return {
            "front": self.cameras["front"].capture_image(),
            "down": self.cameras["down"].capture_image()
        }
    
    def display(self, window_name: str = "Drone Cameras", scale: float = 1.0):
        """
        显示双摄像头画面（调试用）
        
        Args:
            window_name: 窗口名称
            scale: 显示缩放比例
        """
        front_frame = self.cameras["front"].read()
        down_frame = self.cameras["down"].read()
        
        frames = []
        labels = []
        
        for name, frame in [("前视", front_frame), ("下视", down_frame)]:
            if frame is not None:
                # 添加标签
                labeled = frame.copy()
                cv2.putText(labeled, name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(labeled)
            else:
                # 显示空白帧
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, f"{name} - 连接中...", (50, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frames.append(blank)
        
        # 合并显示
        if len(frames) == 2:
            # 上下排列
            combined = np.vstack(frames)
            
            # 缩放
            if scale != 1.0:
                new_size = (int(combined.shape[1] * scale), 
                           int(combined.shape[0] * scale))
                combined = cv2.resize(combined, new_size)
            
            cv2.imshow(window_name, combined)
            return combined
        
        return None


def test_receiver():
    """测试接收器"""
    print("🧪 测试双摄像头接收器")
    print("=" * 40)
    
    receiver = DualCameraReceiver()
    receiver.start()
    
    print("\n快捷键:")
    print("  S - 保存截图")
    print("  Q/ESC - 退出")
    print("=" * 40)
    
    try:
        while True:
            receiver.display()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                images = receiver.capture_images()
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                for name, img in images.items():
                    if img is not None:
                        filename = f"capture_{name}_{timestamp}.jpg"
                        cv2.imwrite(filename, img)
                        print(f"📸 已保存: {filename}")
    
    finally:
        receiver.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_receiver()

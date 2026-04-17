"""
ESP32接收器单元测试

运行测试:
    python -m pytest drone_swarm_system/tests/test_esp32_receiver.py -v
    或
    python drone_swarm_system/tests/test_esp32_receiver.py
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src"))

import numpy as np


class TestESP32Config(unittest.TestCase):
    """测试ESP32配置"""
    
    def test_config_import(self):
        """测试配置导入"""
        from esp32_receiver import ESP32_CONFIG, STREAM_CONFIG, IMAGE_CONFIG
        
        self.assertIn("front_camera", ESP32_CONFIG)
        self.assertIn("down_camera", ESP32_CONFIG)
        
    def test_camera_config(self):
        """测试摄像头配置"""
        from esp32_receiver import ESP32_CONFIG
        
        front = ESP32_CONFIG["front_camera"]
        self.assertIn("ip", front)
        self.assertIn("port", front)
        self.assertIn("stream_url", front)
        
        down = ESP32_CONFIG["down_camera"]
        self.assertIn("ip", down)
        self.assertIn("port", down)


class TestCameraFrame(unittest.TestCase):
    """测试CameraFrame数据类"""
    
    def test_frame_creation(self):
        """测试创建帧"""
        from esp32_receiver import CameraFrame
        
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = CameraFrame(
            image=image,
            timestamp=12345.0,
            camera_name="test_camera"
        )
        
        self.assertEqual(frame.camera_name, "test_camera")
        self.assertEqual(frame.timestamp, 12345.0)
        self.assertTrue(frame.is_valid)


class TestDualCameraReceiver(unittest.TestCase):
    """测试双摄像头接收器"""
    
    def test_receiver_creation(self):
        """测试创建接收器"""
        from esp32_receiver import DualCameraReceiver
        
        receiver = DualCameraReceiver()
        self.assertIsNotNone(receiver)
        
    def test_camera_access(self):
        """测试摄像头访问"""
        from esp32_receiver import DualCameraReceiver
        
        receiver = DualCameraReceiver()
        
        # 检查摄像头是否存在
        self.assertIn("front", receiver.cameras)
        self.assertIn("down", receiver.cameras)


def run_tests():
    """运行测试"""
    print("=" * 60)
    print("🧪 ESP32接收器单元测试")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

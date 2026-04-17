#!/usr/bin/env python3
"""
无人机集群视觉感知系统 - ESP32-CAM 版本

从 ESP32-CAM 双摄像头获取图像，进行视觉处理
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "drone_swarm_system" / "src"))

from drone_vision_processor import DroneVisionProcessor, OutputManager, create_vision_processor

# 尝试导入 ESP32 接收器 (通过直接加载模块文件)
try:
    import importlib.util
    
    # 加载 config 模块
    config_path = project_root / "drone_swarm_system" / "src" / "esp32_receiver" / "config.py"
    spec = importlib.util.spec_from_file_location("esp32_config", config_path)
    esp32_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(esp32_config)
    ESP32_CONFIG = esp32_config.ESP32_CONFIG
    STREAM_CONFIG = esp32_config.STREAM_CONFIG
    
    # 加载 camera_receiver 模块
    receiver_path = project_root / "drone_swarm_system" / "src" / "esp32_receiver" / "camera_receiver.py"
    spec = importlib.util.spec_from_file_location("camera_receiver", receiver_path)
    camera_receiver = importlib.util.module_from_spec(spec)
    # 将 config 模块添加到 camera_receiver 的命名空间
    sys.modules['config'] = esp32_config
    spec.loader.exec_module(camera_receiver)
    
    DualCameraReceiver = camera_receiver.DualCameraReceiver
    CameraStream = camera_receiver.CameraStream
    CameraFrame = camera_receiver.CameraFrame
    
    ESP32_AVAILABLE = True
    print("[OK] ESP32 接收器模块可用")
except Exception as e:
    ESP32_AVAILABLE = False
    print(f"[WARN] ESP32 接收器模块不可用: {e}")
    ESP32_CONFIG = None
    STREAM_CONFIG = None


class ESP32VisionSystem:
    """ESP32-CAM 视觉感知系统主类"""
    
    def __init__(self, output_base: str = "captures"):
        self.output_base = output_base
        self.processor: Optional[DroneVisionProcessor] = None
        self.output_mgr: Optional[OutputManager] = None
        self.esp32_receiver: Optional[DualCameraReceiver] = None
        
        # 状态
        self.is_running = False
        self.session_active = False
        
        print("\n" + "="*70)
        print("[DRONE] 无人机集群视觉感知系统 (ESP32-CAM 版本)")
        print("="*70)
        print("   摄像头: ESP32-CAM 双摄像头 (前视 + 后视)")
        print(f"   输出目录: {output_base}/")
        print("="*70)
        
        if not ESP32_AVAILABLE:
            print("\n[WARN] ESP32 接收器模块不可用，请检查:")
            print("   - drone_swarm_system/src/esp32_receiver/ 目录是否存在")
            print("   - config.py 和 camera_receiver.py 是否存在")
    
    # ==================== 初始化与配置 ====================
    
    def initialize_session(self, force_neural: bool = True) -> bool:
        """
        初始化新的处理会话
        
        Args:
            force_neural: 是否强制使用神经网络（失败时报错）
        """
        if not ESP32_AVAILABLE:
            print("\n[ERR] ESP32 模块不可用，无法初始化")
            return False
            
        try:
            print("\n[DIR] 初始化 ESP32-CAM 会话...")
            
            # 创建输出管理器
            self.output_mgr = OutputManager(self.output_base)
            
            # 创建 ESP32 接收器 (2个摄像头 - 前视和后视)
            print("\n[CAM] 正在连接 ESP32-CAM...")
            print("   前视摄像头: " + ESP32_CONFIG["front_camera"]["stream_url"])
            print("   后视摄像头(180度): " + ESP32_CONFIG["down_camera"]["stream_url"])
            
            self.esp32_receiver = DualCameraReceiver()
            
            # 启动接收器
            self.esp32_receiver.start()
            
            # 等待连接建立
            import time
            print("\n   等待连接建立...")
            time.sleep(3)
            
            # 检查连接状态
            front_status = self.esp32_receiver.cameras["front"].is_connected()
            down_status = self.esp32_receiver.cameras["down"].is_connected()
            
            if front_status:
                print("   [OK] 前视摄像头已连接")
            else:
                print("   [WARN] 前视摄像头未连接")
                
            if down_status:
                print("   [OK] 后视摄像头(180度)已连接")
            else:
                print("   [WARN] 后视摄像头(180度)未连接")
            
            if not front_status and not down_status:
                print("\n[ERR] 两个摄像头都未连接，请检查:")
                print("   1. ESP32-CAM 是否已上电")
                print("   2. 电脑是否已连接到 ESP32-CAM 的 WiFi")
                print("   3. ESP32 的 IP 地址是否正确")
                return False
            
            # 创建视觉处理器 (2个摄像头)
            print("\n[NN] 正在初始化视觉处理器...")
            self.processor = create_vision_processor(
                num_cameras=2,  # 前视 + 后视
                frames_per_camera=2,
                capture_interval=0.1,
                use_neural_network=True,
                force_neural_network=force_neural
            )
            
            self.session_active = True
            print("[OK] 会话初始化完成")
            return True
            
        except RuntimeError as e:
            print(f"\n{'='*70}")
            print("[ERR] 神经网络加载失败")
            print(f"{'='*70}")
            print(f"\n错误信息: {e}")
            print("\n可能的解决方案:")
            print("  1. 运行诊断脚本检查环境:")
            print("     .\\venv\\Scripts\\python.exe test_neural_network.py")
            print("  2. 安装缺失的依赖:")
            print("     .\\venv\\Scripts\\python.exe -m pip install torch torchvision pillow")
            print("  3. 使用传统算法模式（禁用强制NN）:")
            print("     修改 drone_vision_processor.py 中的 force_neural_network = False")
            print(f"{'='*70}\n")
            return False
            
        except Exception as e:
            print(f"[ERR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset_session(self):
        """重置当前会话"""
        if self.esp32_receiver:
            self.esp32_receiver.stop()
            self.esp32_receiver = None
        self.processor = None
        self.output_mgr = None
        self.session_active = False
        print("\n[RESET] 会话已重置")
    
    # ==================== 图像采集 ====================
    
    def capture_from_esp32(self) -> bool:
        """从 ESP32-CAM 采集图像"""
        if not self.session_active or not self.esp32_receiver:
            print("[ERR] 会话未初始化")
            return False
        
        print("\n" + "="*70)
        print("[CAM] 从 ESP32-CAM 采集图像")
        print("="*70)
        
        # 采集前视摄像头 (Camera 0)
        print("\n[1/2] 采集前视摄像头...")
        front_frame = self.esp32_receiver.cameras["front"].capture_image()
        if front_frame is not None:
            # 存入处理器
            if self.processor.groups[0].frame_1 is None:
                self.processor.groups[0].frame_1 = front_frame
            else:
                self.processor.groups[0].frame_2 = front_frame
            print(f"   [OK] 前视: {front_frame.shape}")
        else:
            print("   [ERR] 前视摄像头采集失败")
        
        # 等待一小段时间
        import time
        time.sleep(0.1)
        
        # 采集后视摄像头 (Camera 1, 180度)
        print("\n[2/2] 采集后视摄像头(180度)...")
        down_frame = self.esp32_receiver.cameras["down"].capture_image()
        if down_frame is not None:
            # 存入处理器
            if self.processor.groups[1].frame_1 is None:
                self.processor.groups[1].frame_1 = down_frame
            else:
                self.processor.groups[1].frame_2 = down_frame
            print(f"   [OK] 后视(180度): {down_frame.shape}")
        else:
            print("   [ERR] 后视摄像头(180度)采集失败")
        
        # 检查是否采集成功
        success_count = sum([
            1 if self.processor.groups[i].frame_1 is not None else 0
            for i in range(2)
        ])
        
        print(f"\n[OK] 采集完成: {success_count}/2 摄像头")
        
        # 保存原始图像
        if self.output_mgr:
            for i in range(2):
                group = self.processor.groups[i]
                if group.frame_1 is not None:
                    self.output_mgr.save_image(group.frame_1, f"camera_{i}_frame_1_raw.png")
                if group.frame_2 is not None:
                    self.output_mgr.save_image(group.frame_2, f"camera_{i}_frame_2_raw.png")
        
        return success_count > 0
    
    def show_live_preview(self, duration: int = 10):
        """显示实时预览"""
        if not self.esp32_receiver:
            print("[ERR] ESP32 接收器未初始化")
            return
        
        print(f"\n[CAM] 显示实时预览 ({duration}秒，按 Q 退出)")
        print("="*70)
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 获取帧
            front_frame = self.esp32_receiver.cameras["front"].read()
            down_frame = self.esp32_receiver.cameras["down"].read()
            
            # 创建显示画面
            display_frames = []
            
            if front_frame is not None:
                # 添加标签
                front_label = front_frame.copy()
                cv2.putText(front_label, "FRONT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                display_frames.append(front_label)
            else:
                # 创建空白帧
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, "FRONT - NO SIGNAL", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                display_frames.append(blank)
            
            if down_frame is not None:
                down_label = down_frame.copy()
                cv2.putText(down_label, "REAR(180)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                display_frames.append(down_label)
            else:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, "REAR(180) - NO SIGNAL", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                display_frames.append(blank)
            
            # 水平拼接
            if len(display_frames) == 2:
                combined = np.hstack(display_frames)
            else:
                combined = display_frames[0]
            
            # 显示
            cv2.imshow("ESP32-CAM Preview (Q to exit)", combined)
            
            # 检查按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("[OK] 预览结束")
    
    # ==================== 视觉处理 ====================
    
    def process_images(self) -> bool:
        """处理已采集的图像"""
        if not self.processor:
            print("[ERR] 处理器未初始化")
            return False
        
        print("\n" + "="*70)
        print("[NN] 处理图像")
        print("="*70)
        
        # 计算光流
        print("\n[1/4] 计算光流...")
        self.processor.compute_optical_flow_all()
        
        # 深度估计
        print("\n[2/4] 深度估计...")
        self.processor.estimate_depth_all()
        
        # 立体深度
        print("\n[3/4] 立体深度估计...")
        self.processor.estimate_stereo_depth(left_idx=0, right_idx=1)
        
        # 特征提取
        print("\n[4/4] 提取特征点...")
        self.processor.extract_features_all()
        
        print("\n[OK] 处理完成")
        return True
    
    def generate_point_cloud(self) -> bool:
        """生成点云"""
        if not self.processor:
            print("[ERR] 处理器未初始化")
            return False
        
        print("\n" + "="*70)
        print("[PC] 生成点云")
        print("="*70)
        
        for i in range(2):  # 2个摄像头
            print(f"\n生成 Camera {i} 的点云...")
            self.processor.generate_point_cloud(i)
        
        print("\n[OK] 点云生成完成")
        return True
    
    def save_results(self):
        """保存处理结果"""
        if not self.output_mgr or not self.processor:
            print("[ERR] 无法保存")
            return
        
        print("\n[DIR] 保存结果...")
        self.processor.save_all(self.output_mgr.session_dir)
        print(f"[OK] 结果保存到: {self.output_mgr.session_dir}")
    
    # ==================== 完整流程 ====================
    
    def run_full_pipeline(self):
        """执行完整处理流程"""
        print("\n" + "="*70)
        print("[TARGET] 执行完整流程")
        print("="*70)
        
        # 1. 采集图像
        if not self.capture_from_esp32():
            print("[ERR] 图像采集失败")
            return False
        
        # 2. 处理图像
        if not self.process_images():
            print("[ERR] 图像处理失败")
            return False
        
        # 3. 生成点云
        if not self.generate_point_cloud():
            print("[ERR] 点云生成失败")
            return False
        
        # 4. 保存结果
        self.save_results()
        
        print("\n" + "="*70)
        print("[OK] 完整流程执行完毕")
        print("="*70)
        
        return True
    
    # ==================== 菜单系统 ====================
    
    def show_menu(self):
        """显示交互式菜单"""
        # 获取神经网络状态
        nn_status = "[NN] ON" if (self.processor and self.processor.neural_network_loaded) else "[TRAD] OFF"
        
        # 获取连接状态
        front_status = "OK" if (self.esp32_receiver and self.esp32_receiver.cameras.get("front") and self.esp32_receiver.cameras["front"].is_connected()) else "--"
        down_status = "OK" if (self.esp32_receiver and self.esp32_receiver.cameras.get("down") and self.esp32_receiver.cameras["down"].is_connected()) else "--"
        
        print("\n" + "="*70)
        print("[MENU] 主菜单")
        print("="*70)
        print(f"  连接状态: 前视(0度)[{front_status}] 后视(180度)[{down_status}]")
        print(f"  神经网络: {nn_status}")
        print("-"*70)
        print("  1. 初始化 ESP32 会话")
        print("  2. 显示实时预览")
        print("  3. 采集图像")
        print("  4. 处理图像 (光流/深度)")
        print("  5. 生成点云")
        print("  6. 3D点云可视化(Open3D)")
        print("  7. 执行完整流程")
        print("  8. 保存结果")
        print("  9. 切换神经网络")
        print("  10. 重置会话")
        print("  0. 退出程序")
        print("="*70)
        print("提示: 确保电脑已连接到 ESP32-CAM 的 WiFi 热点")
    
    def run(self):
        """运行主循环"""
        self.is_running = True
        
        while self.is_running:
            self.show_menu()
            
            try:
                choice = input("\n请选择操作 [0-10]: ").strip()
                
                if choice == '0':
                    self.is_running = False
                    print("\n退出程序...")
                    
                elif choice == '1':
                    self.initialize_session()
                    
                elif choice == '2':
                    duration = input("预览时长(秒，默认10): ").strip()
                    duration = int(duration) if duration else 10
                    self.show_live_preview(duration)
                    
                elif choice == '3':
                    self.capture_from_esp32()
                    
                elif choice == '4':
                    self.process_images()
                    
                elif choice == '5':
                    self.generate_point_cloud()
                    
                elif choice == '6':
                    # 3D点云可视化
                    if self.processor:
                        camera_idx = input("选择摄像头 [0-1] (默认0): ").strip()
                        camera_idx = int(camera_idx) if camera_idx else 0
                        self.processor.visualize_point_cloud_3d(camera_idx)
                    else:
                        print("[ERR] 请先初始化会话并生成点云")
                    
                elif choice == '7':
                    self.run_full_pipeline()
                    
                elif choice == '8':
                    self.save_results()
                    
                elif choice == '9':
                    if self.processor:
                        new_state = self.processor.toggle_neural_network()
                        status = "ON" if new_state else "OFF"
                        print(f"\n[NN] 神经网络: {status}")
                    else:
                        print("[ERR] 处理器未初始化")
                        
                elif choice == '10':
                    self.reset_session()
                    
                else:
                    print("[ERR] 无效选择")
                    
            except KeyboardInterrupt:
                print("\n\n[ERR] 用户中断")
                self.is_running = False
            except Exception as e:
                print(f"\n[ERR] 错误: {e}")


def main():
    """主函数"""
    app = ESP32VisionSystem(output_base="captures_esp32")
    
    try:
        app.run()
    except Exception as e:
        print(f"\n[ERR] 程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if app.esp32_receiver:
            app.esp32_receiver.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

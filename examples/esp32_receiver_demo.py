#!/usr/bin/env python3
"""
ESP32-CAM 接收器演示

演示如何接收和显示ESP32双摄像头的视频流

使用方法:
    python examples/esp32_receiver_demo.py

快捷键:
    S - 保存当前画面截图
    R - 开始/停止录制视频
    Q/ESC - 退出程序
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src"))

import cv2
import numpy as np
import time

from esp32_receiver import DualCameraReceiver


def main():
    print("=" * 60)
    print("🚁 ESP32-CAM 双摄像头接收演示")
    print("=" * 60)
    
    # 创建接收器
    receiver = DualCameraReceiver()
    
    # 启动接收
    print("\n正在启动摄像头接收...")
    receiver.start()
    
    print("\n✅ 系统已启动")
    print("快捷键:")
    print("  S - 保存截图")
    print("  R - 开始/停止录制")
    print("  Q/ESC - 退出")
    print("-" * 60)
    
    # 录制相关
    recording = False
    video_writer = None
    recording_start_time = None
    
    try:
        while True:
            # 获取帧
            frames = receiver.read_both()
            front_frame, down_frame = frames
            
            # 创建显示画面
            display_frames = []
            
            for name, frame in [("前视", front_frame), ("下视", down_frame)]:
                if frame is not None:
                    # 添加标签
                    labeled = frame.copy()
                    cv2.putText(labeled, name, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    display_frames.append(labeled)
                else:
                    # 显示连接中
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(blank, f"{name} - 连接中...", (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    display_frames.append(blank)
            
            # 合并显示
            if len(display_frames) == 2:
                combined = np.vstack(display_frames)
                
                # 添加状态信息
                status = receiver.get_status()
                status_text = f"前视: {'已连接' if status['front']['connected'] else '连接中'} | "
                status_text += f"下视: {'已连接' if status['down']['connected'] else '连接中'}"
                cv2.putText(combined, status_text, (10, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 添加录制状态
                if recording:
                    cv2.putText(combined, "● 录制中", (500, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if video_writer:
                        video_writer.write(combined)
                
                cv2.imshow("ESP32-CAM 双摄像头", combined)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                # 保存截图
                images = receiver.capture_images()
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                for name, img in images.items():
                    if img is not None:
                        filename = f"capture_{name}_{timestamp}.jpg"
                        cv2.imwrite(filename, img)
                        print(f"📸 已保存: {filename}")
            elif key == ord('r'):
                # 开始/停止录制
                recording = not recording
                if recording:
                    # 开始录制
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"recording_{timestamp}.avi"
                    video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))
                    recording_start_time = time.time()
                    print(f"🎬 开始录制: {filename}")
                else:
                    # 停止录制
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                        duration = time.time() - recording_start_time
                        print(f"⏹️ 停止录制，时长: {duration:.1f}秒")
    
    except KeyboardInterrupt:
        print("\n\n正在停止...")
    finally:
        receiver.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    main()

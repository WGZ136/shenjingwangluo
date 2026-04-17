#!/usr/bin/env python3
"""
视觉处理流程演示

演示从ESP32图传到深度估计、光流计算、点云生成的完整流程

使用方法:
    python examples/visual_process_demo.py

功能:
    - 接收ESP32图传
    - 实时深度估计
    - 光流计算
    - 点云生成
    - 3D可视化
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src"))

import cv2
import numpy as np
import time

from drone_swarm_system.core.system import DroneSwarmSystem
from drone_swarm_system.core.config import SystemConfig


def main():
    print("=" * 60)
    print("🚁 视觉处理流程演示")
    print("=" * 60)
    
    # 配置系统
    config = SystemConfig(
        enable_esp32_receiver=True,
        enable_3d_visualization=True,  # 启用3D可视化
        camera_intrinsics={
            'fx': 320.0, 'fy': 320.0,
            'cx': 320.0, 'cy': 240.0
        }
    )
    
    # 创建系统
    system = DroneSwarmSystem(config=config)
    
    print("\n✅ 系统已启动")
    print("正在接收ESP32图传并进行视觉处理...")
    print("按 Q 退出，按 S 保存点云")
    print("-" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 处理一帧
            result = system.process_frame()
            
            if result.success:
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # 显示结果
                system.visualize()
                
                # 显示状态
                status_text = f"FPS: {fps:.1f} | 帧数: {frame_count}"
                print(f"\r{status_text}", end="")
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存点云
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"pointcloud_{timestamp}.ply"
                    system.save_pointcloud(filename)
                    print(f"\n💾 点云已保存: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n正在停止...")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n\n处理完成:")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {elapsed:.1f}s")
        print(f"  平均FPS: {frame_count/elapsed:.1f}")


if __name__ == "__main__":
    main()

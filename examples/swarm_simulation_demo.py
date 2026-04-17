#!/usr/bin/env python3
"""
集群模拟演示

演示单机多位置拍摄，模拟多机集群控制

使用方法:
    python examples/swarm_simulation_demo.py

操作流程:
    1. 将无人机移动到位置1，按 S 拍摄
    2. 将无人机移动到位置2，按 S 拍摄
    3. 重复直到拍摄足够的位置
    4. 按 Q 结束拍摄，查看模拟结果
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src"))

import cv2
import numpy as np
import time

from visual_integration import create_visual_integration


def main():
    print("=" * 60)
    print("🚁 无人机集群模拟演示")
    print("=" * 60)
    print("\n这将模拟多架无人机的视觉感知")
    print("你需要移动单架无人机到不同位置拍摄")
    print("-" * 60)
    
    # 获取模拟无人机数量
    num_drones = int(input("请输入模拟无人机数量 (默认3): ") or "3")
    
    # 创建视觉整合器
    visual = create_visual_integration(
        enable_3d=True,  # 启用3D可视化
        enable_esp32=True
    )
    
    if visual is None:
        print("❌ 视觉整合器创建失败")
        return
    
    print(f"\n✅ 系统已启动，将模拟 {num_drones} 架无人机")
    print("\n操作说明:")
    print("  移动无人机到新位置 → 按 S 拍摄")
    print("  拍摄完成后 → 按 Q 结束")
    print("-" * 60)
    
    positions = []
    frame_results = []
    
    try:
        while len(positions) < num_drones:
            # 显示当前画面
            visual.display_esp32_feed("集群模拟 - 拍摄中")
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if len(positions) < 2:
                    print("❌ 至少需要2个位置才能模拟")
                    continue
                break
            elif key == ord('s'):
                # 获取当前帧
                frames = visual.get_esp32_frames()
                
                if frames['front'] is not None:
                    print(f"\n📸 拍摄位置 {len(positions) + 1}/{num_drones}")
                    
                    # 保存当前帧（简单处理）
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"sim_position_{len(positions)}_{timestamp}.jpg", frames['front'])
                    
                    # 记录位置（这里使用简单的计数，实际应该从运动估计获取）
                    position = np.array([len(positions) * 2.0, 0.0, 1.0])  # 简化示例
                    positions.append(position)
                    
                    print(f"✅ 已记录位置 {len(positions)}: {position}")
                    
                    if len(positions) < num_drones:
                        print(f"请移动无人机到下一个位置...")
                else:
                    print("⚠️ 未获取到图像，请检查连接")
        
        # 模拟多机处理
        print("\n" + "=" * 60)
        print("📊 模拟结果")
        print("=" * 60)
        
        for i, pos in enumerate(positions):
            print(f"\n无人机 {i}:")
            print(f"  位置: {pos}")
            print(f"  相对距离: {np.linalg.norm(pos):.2f}m")
        
        # 计算队形参数
        if len(positions) >= 2:
            distances = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            
            print(f"\n队形信息:")
            print(f"  平均间距: {np.mean(distances):.2f}m")
            print(f"  最小间距: {np.min(distances):.2f}m")
            print(f"  最大间距: {np.max(distances):.2f}m")
        
        print("\n按任意键退出...")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        visual.release()
        cv2.destroyAllWindows()
        print("\n演示结束")


if __name__ == "__main__":
    main()

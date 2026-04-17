#!/usr/bin/env python3
"""
无人机集群视觉感知系统 - 主程序

基于 DroneVisionProcessor 的交互式控制界面

运行方式:
    python main.py                    # 启动交互式菜单
    python main.py --auto             # 自动运行完整流程
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

# 类型检查导入 (用于Pylance静态分析)
if TYPE_CHECKING:
    try:
        from drone_swarm_system.src.drone_vision_processor import (
            DroneVisionProcessor, 
            FrameGroup,
            OutputManager,
            create_vision_processor
        )
    except ImportError:
        pass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "drone_swarm_system" / "src"))

import cv2
import numpy as np

from drone_vision_processor import (
    DroneVisionProcessor, 
    FrameGroup,
    OutputManager,
    create_vision_processor
)


class DroneSwarmApp:
    """
    无人机集群视觉感知应用程序
    
    提供交互式界面管理视觉处理流程
    """
    
    def __init__(self, num_cameras: int = 4, output_base: str = "captures"):
        """
        初始化应用程序
        
        Args:
            num_cameras: 摄像头数量（默认4个）
            output_base: 输出目录基名
        """
        self.num_cameras = num_cameras
        self.output_base = output_base
        
        # 核心处理器
        self.processor: Optional[DroneVisionProcessor] = None
        self.output_mgr: Optional[OutputManager] = None
        
        # 状态
        self.is_running = False
        self.session_active = False
        
        print("\n" + "="*70)
        print("[DRONE] 无人机集群视觉感知系统")
        print("="*70)
        print(f"   摄像头数量: {num_cameras}")
        print(f"   输出目录: {output_base}/")
        print("="*70)
    
    # ==================== 初始化与配置 ====================
    
    def initialize_session(self, force_neural: bool = True) -> bool:
        """
        初始化新的处理会话
        
        Args:
            force_neural: 是否强制使用神经网络（失败时报错）
        """
        try:
            print("\n[DIR] 初始化会话...")
            
            # 创建输出管理器
            self.output_mgr = OutputManager(self.output_base)
            
            # 创建视觉处理器
            print("\n[NN] 正在初始化视觉处理器...")
            self.processor = create_vision_processor(
                num_cameras=self.num_cameras,
                frames_per_camera=2,
                capture_interval=0.1,
                use_neural_network=True,
                force_neural_network=force_neural
            )
            
            self.session_active = True
            print("[OK] 会话初始化完成")
            return True
            
        except RuntimeError as e:
            # 神经网络加载失败
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
        self.processor = None
        self.output_mgr = None
        self.session_active = False
        print("\n[RESET] 会话已重置")
    
    # ==================== 键盘控制 ====================
    
    def _check_key_press(self, wait_time: int = 1) -> Optional[str]:
        """
        检测键盘按键
        
        Returns:
            按键字符或 None
        """
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 255:
            return None
        return chr(key) if 32 <= key < 127 else f"key_{key}"
    
    def _handle_key(self, key: str) -> bool:
        """
        处理按键
        
        Args:
            key: 按键字符
            
        Returns:
            True 表示继续运行, False 表示退出
        """
        if key is None:
            return True
        
        if key.lower() == 'q':
            # Q键: 切换神经网络
            new_state = self.processor.toggle_neural_network()
            print(f"\n🔘 神经网络已{'启用' if new_state else '禁用'}")
            return True
        
        elif key == ' ' or key == 'key_32':  # 空格键
            # 空格: 立即采集
            print("\n⏹  立即执行采集")
            return True
        
        elif key == 'key_27':  # ESC键
            # ESC: 退出
            print("\n⛔ 用户取消")
            return False
        
        return True
    
    # ==================== 核心功能（合并步骤） ====================
    
    def capture_and_process(self, camera_id: int) -> bool:
        """
        采集并处理单个摄像头（合并步骤）
        
        流程:
            1. 采集2帧（间隔0.1秒）
            2. 计算光流
            3. 估计深度
            
        支持按键:
            Q - 切换神经网络开关
            ESC - 退出
        """
        print(f"\n📸 Camera {camera_id}: 采集+处理")
        print("-" * 50)
        
        # 步骤1: 采集2帧
        print("  [1/3] 采集2帧...")
        frame_0, frame_1 = self.processor.capture_from_esp32(camera_id)
        
        if frame_0 is None or frame_1 is None:
            print(f"  ✗ Camera {camera_id}: 采集失败")
            return False
        
        # 保存到组
        group = self.processor.groups[camera_id]
        group.frame_0 = frame_0
        group.frame_1 = frame_1
        group.timestamp = time.time()
        
        # 保存原始图像
        cv2.imwrite(
            str(self.output_mgr.get_path('raw', f"cam{camera_id}_frame0.jpg")),
            frame_0
        )
        cv2.imwrite(
            str(self.output_mgr.get_path('raw', f"cam{camera_id}_frame1.jpg")),
            frame_1
        )
        
        # 步骤2: 计算光流
        print("  [2/3] 计算光流...")
        flow = self.processor.compute_optical_flow(camera_id)
        
        if flow is not None:
            # 保存光流可视化
            flow_vis = self.processor.visualize_flow(camera_id)
            if flow_vis is not None:
                cv2.imwrite(
                    str(self.output_mgr.get_path('flow', f"cam{camera_id}_flow.jpg")),
                    flow_vis
                )
            print(f"    运动: [{group.motion_vector[0]:.2f}, {group.motion_vector[1]:.2f}]")
        
        # 步骤3: 估计深度
        print("  [3/3] 估计深度...")
        depth = self.processor.estimate_depth_single(camera_id)
        
        if depth is not None:
            # 保存深度图
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            cv2.imwrite(
                str(self.output_mgr.get_path('depth', f"cam{camera_id}_depth.jpg")),
                depth_color
            )
        
        # 实时显示（支持按键）
        self._show_live_preview(camera_id)
        
        print(f"  ✓ Camera {camera_id}: 完成")
        return True
    
    def _show_live_preview(self, camera_id: int):
        """显示实时预览窗口（支持按键检测）"""
        group = self.processor.groups[camera_id]
        if not group.is_complete():
            return
        
        h, w = group.frame_1.shape[:2]
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # 左: 原始图像
        canvas[:, 0:w] = group.frame_1
        cv2.putText(canvas, f"Cam{camera_id} Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 中: 光流
        if group.flow is not None:
            flow_vis = self.processor.visualize_flow(camera_id)
            if flow_vis is not None:
                canvas[:, w:2*w] = cv2.resize(flow_vis, (w, h))
                motion = group.motion_vector
                cv2.putText(canvas, f"Flow: [{motion[0]:.1f}, {motion[1]:.1f}]",
                           (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 右: 深度
        if group.depth is not None:
            depth_color = cv2.applyColorMap(group.depth, cv2.COLORMAP_JET)
            canvas[:, 2*w:3*w] = cv2.resize(depth_color, (w, h))
            nn_status = "NN:ON" if self.processor.use_neural_network else "NN:OFF"
            cv2.putText(canvas, f"Depth ({nn_status})", (2*w+10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 底部提示
        hint = "Press Q:Toggle NN  ESC:Exit"
        cv2.putText(canvas, hint, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示
        win_name = f"Camera {camera_id} - Press Q to toggle Neural Network"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, canvas)
        
        # 检测按键（短暂等待）
        key = self._check_key_press(100)  # 等待100ms
        if key == 'key_27':  # ESC
            cv2.destroyWindow(win_name)
            raise KeyboardInterrupt("用户退出")
        elif key and key.lower() == 'q':
            self._handle_key(key)
            # 重新显示以更新NN状态
            self._show_live_preview(camera_id)
    
    def capture_all_with_processing(self) -> bool:
        """
        采集所有摄像头并实时处理（合并步骤）
        
        支持按键:
            Q - 切换神经网络开关
            ESC - 退出程序
        """
        print("\n" + "="*70)
        print("📸 采集+处理模式（按 Q 切换神经网络，ESC 退出）")
        print("="*70)
        print(f"   神经网络状态: {'[NN] 启用' if self.processor.use_neural_network else '[TRAD] 禁用'}")
        print("   按 Q 键切换神经网络开关\n")
        
        success_count = 0
        
        for i in range(self.num_cameras):
            try:
                if self.capture_and_process(i):
                    success_count += 1
            except KeyboardInterrupt:
                print("\n⛔ 采集中断")
                break
        
        print("\n" + "="*70)
        print(f"[OK] 采集完成: {success_count}/{self.num_cameras} 摄像头")
        print("="*70)
        
        return success_count > 0
    
    # ==================== 兼容旧方法 ====================
    
    def capture_images(self) -> bool:
        """执行图像采集（兼容旧代码）"""
        return self.capture_all_with_processing()
    
    def compute_optical_flow(self) -> bool:
        """计算光流（已在采集时完成）"""
        print("[OK] 光流已在采集时计算完成")
        return True
    
    def estimate_depth(self) -> bool:
        """估计深度"""
        if not self.session_active:
            print("[WARN]  请先初始化会话")
            return False
        
        print("\n" + "-"*70)
        print("📏 步骤 3: 深度估计")
        print("-"*70)
        
        success = self.processor.estimate_all_depths()
        
        if success:
            # 保存深度图
            for i, group in enumerate(self.processor.groups):
                if group.depth is not None:
                    # 伪彩色深度图
                    depth_color = cv2.applyColorMap(group.depth, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        str(self.output_mgr.get_path('depth', f"cam{i}_depth.jpg")),
                        depth_color
                    )
            print("[OK] 深度图已保存")
        
        return success
    
    def generate_pointcloud(self) -> bool:
        """生成点云"""
        if not self.session_active:
            print("[WARN]  请先初始化会话")
            return False
        
        print("\n" + "-"*70)
        print("☁️  步骤 4: 点云生成")
        print("-"*70)
        
        # 使用Camera 0生成点云
        point_cloud = self.processor.generate_point_cloud(camera_idx=0)
        
        if point_cloud is not None:
            # 保存点云
            pc_path = self.output_mgr.get_path('depth', "pointcloud.txt")
            np.savetxt(str(pc_path), point_cloud, fmt='%.4f')
            print(f"[OK] 点云已保存: {pc_path}")
            return True
        
        return False
    
    def run_full_pipeline(self) -> bool:
        """运行完整处理流程（合并步骤）"""
        print("\n" + "="*70)
        print("🚀 执行完整处理流程")
        print("="*70)
        print("提示: 采集过程中按 Q 切换神经网络")
        
        # 检查或初始化会话
        if not self.session_active:
            if not self.initialize_session():
                return False
        
        # 执行各步骤（合并采集+处理）
        steps = [
            ("采集+处理", self.capture_all_with_processing),  # 包含光流+深度
            ("点云生成", self.generate_pointcloud),
        ]
        
        results = []
        for name, step_func in steps:
            try:
                success = step_func()
                results.append((name, success))
                if not success:
                    print(f"[WARN]  {name} 步骤未完成")
            except Exception as e:
                print(f"[ERR] {name} 出错: {e}")
                results.append((name, False))
        
        # 保存报告
        self._save_report()
        
        # 显示结果摘要
        print("\n" + "="*70)
        print("📊 处理结果摘要")
        print("="*70)
        for name, success in results:
            status = "[OK] 成功" if success else "[ERR] 失败"
            print(f"  {name}: {status}")
        
        # 显示NN状态
        nn_state = "[NN] 启用" if self.processor.use_neural_network else "[TRAD] 禁用"
        print(f"  神经网络: {nn_state}")
        
        all_success = all(s for _, s in results)
        if all_success:
            print(f"\n[OK] 所有步骤完成!")
            print(f"[DIR] 输出目录: {self.output_mgr.session_dir}")
        
        return all_success
    
    def _save_report(self):
        """保存处理报告"""
        if not self.session_active or self.processor is None:
            return
        
        summary = self.processor.get_summary()
        
        report = {
            'session_id': summary['session_id'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'configuration': {
                'num_cameras': self.num_cameras,
                'frames_per_camera': summary['frames_per_camera'],
            },
            'results': summary,
            'output_directory': str(self.output_mgr.session_dir)
        }
        
        report_path = self.output_mgr.save_json('report', 'processing_report.json', report)
        print(f"\n📝 报告已保存: {report_path}")
    
    # ==================== 可视化 ====================
    
    def visualize_results(self):
        """可视化处理结果"""
        if not self.session_active or self.processor is None:
            print("[WARN]  没有可可视化的结果")
            return
        
        print("\n" + "-"*70)
        print("🎨 可视化结果")
        print("-"*70)
        print("按 'Q' 或 ESC 关闭窗口\n")
        
        windows_created = []
        
        for i, group in enumerate(self.processor.groups):
            if not group.is_complete():
                continue
            
            h, w = group.frame_1.shape[:2]
            
            # 创建2×2布局
            canvas = np.zeros((h*2, w*2, 3), dtype=np.uint8)
            
            # 左上: 原始图像
            canvas[0:h, 0:w] = group.frame_1
            cv2.putText(canvas, f"Camera {i} - Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 右上: 光流
            if group.flow is not None:
                flow_vis = self.processor.visualize_flow(i)
                if flow_vis is not None:
                    canvas[0:h, w:2*w] = cv2.resize(flow_vis, (w, h))
                    motion = group.motion_vector
                    cv2.putText(canvas, f"Flow: [{motion[0]:.1f}, {motion[1]:.1f}]",
                               (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 左下: 深度图
            if group.depth is not None:
                depth_color = cv2.applyColorMap(group.depth, cv2.COLORMAP_JET)
                canvas[h:2*h, 0:w] = cv2.resize(depth_color, (w, h))
                cv2.putText(canvas, "Depth", (10, h+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # 右下: 信息面板
            info = np.zeros((h, w, 3), dtype=np.uint8)
            y = 30
            cv2.putText(info, f"Camera ID: {i}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(info, f"Complete: {group.is_complete()}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            if group.flow is not None:
                y += 25
                cv2.putText(info, "Flow: ✓", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            if group.depth is not None:
                y += 25
                cv2.putText(info, "Depth: ✓", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            canvas[h:2*h, w:2*w] = info
            
            # 显示窗口
            win_name = f"Camera {i} Results"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 800, 600)
            cv2.imshow(win_name, canvas)
            windows_created.append(win_name)
        
        if windows_created:
            # 等待按键
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key in [ord('q'), 27]:  # Q 或 ESC
                    break
            
            # 关闭所有窗口
            for win in windows_created:
                cv2.destroyWindow(win)
            cv2.destroyAllWindows()
        else:
            print("[WARN]  没有可显示的结果")
    
    # ==================== 交互式菜单 ====================
    
    def show_menu(self):
        """显示交互式菜单"""
        # 获取神经网络状态
        nn_status = "[NN] ON" if (self.processor and self.processor.use_neural_network) else "[TRAD] OFF"
        
        print("\n" + "="*70)
        print("[MENU] 主菜单")
        print("="*70)
        print("  1. 初始化新会话")
        print(f"  2. 采集+处理 (含光流/深度) [当前NN:{nn_status}]")
        print("  3. 生成点云")
        print("  4. 执行完整流程")
        print("  5. 可视化结果(图像)")
        print("  6. 3D点云可视化(Open3D)")
        print("  7. 切换神经网络 (Q键)")
        print("  8. 重置会话")
        print("  9. 显示状态")
        print("  0. 退出程序")
        print("="*70)
        print("提示: 采集时按 Q 键可实时切换神经网络")
    
    def show_status(self):
        """显示当前状态"""
        nn_status = f"{'[NN] 启用' if self.processor.use_neural_network else '[TRAD] 禁用'}" if self.processor else "N/A"
        
        print("\n" + "-"*70)
        print("📊 当前状态")
        print("-"*70)
        print(f"  会话状态: {'[OK] 活跃' if self.session_active else '[ERR] 未初始化'}")
        print(f"  摄像头数: {self.num_cameras}")
        print(f"  神经网络: {nn_status}")
        
        if self.processor:
            summary = self.processor.get_summary()
            print(f"  完成组数: {summary['complete_groups']}/{summary['num_cameras']}")
            print(f"  光流计算: {summary['has_flow']}")
            print(f"  深度估计: {summary['has_depth']}")
            print(f"  点云点数: {summary['point_cloud_size']}")
        
        if self.output_mgr:
            print(f"  输出目录: {self.output_mgr.session_dir}")
        print("-"*70)
    
    def run_interactive(self):
        """运行交互式模式"""
        self.is_running = True
        
        # 自动初始化
        self.initialize_session()
        
        while self.is_running:
            self.show_menu()
            
            try:
                choice = input("\n请选择操作 [0-8]: ").strip()
                
                if choice == '1':
                    self.initialize_session()
                    
                elif choice == '2':
                    # 采集+处理（含光流和深度）
                    self.capture_all_with_processing()
                    
                elif choice == '3':
                    self.generate_pointcloud()
                    
                elif choice == '4':
                    self.run_full_pipeline()
                    
                elif choice == '5':
                    self.visualize_results()
                    
                elif choice == '6':
                    # 3D点云可视化
                    if self.processor:
                        camera_idx = input("选择摄像头 [0-3] (默认0): ").strip()
                        camera_idx = int(camera_idx) if camera_idx else 0
                        self.processor.visualize_point_cloud_3d(camera_idx)
                    else:
                        print("[ERR] 请先初始化会话并生成点云")
                    
                elif choice == '7':
                    # 切换神经网络
                    if self.processor:
                        new_state = self.processor.toggle_neural_network()
                        status = "ON" if new_state else "OFF"
                        print(f"\n[NN] 神经网络: {status}")
                    else:
                        print("[WARN] 请先初始化会话")
                    
                elif choice == '8':
                    self.reset_session()
                    
                elif choice == '9':
                    self.show_status()
                    
                elif choice == '0':
                    print("\n[BYE] 退出程序...")
                    self.is_running = False
                    
                else:
                    print("[WARN]  无效选项，请重新选择")
                    
            except KeyboardInterrupt:
                print("\n\n⛔ 操作被中断")
                self.is_running = False
            except Exception as e:
                print(f"\n[ERR] 错误: {e}")
        
        # 清理
        cv2.destroyAllWindows()
        print("[OK] 程序已退出")


def main():
    parser = argparse.ArgumentParser(
        description='无人机集群视觉感知系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
运行方式:
    python main.py              # 交互式菜单模式
    python main.py --auto       # 自动执行完整流程
    python main.py --cameras 2  # 使用2个摄像头
        '''
    )
    
    parser.add_argument('--auto', '-a',
                       action='store_true',
                       help='自动模式（跳过菜单，直接执行完整流程）')
    parser.add_argument('--cameras', '-c',
                       type=int, default=4,
                       help='摄像头数量 (默认: 4)')
    parser.add_argument('--output', '-o',
                       default='captures',
                       help='输出目录基名 (默认: captures)')
    
    args = parser.parse_args()
    
    try:
        # 创建应用程序
        app = DroneSwarmApp(
            num_cameras=args.cameras,
            output_base=args.output
        )
        
        if args.auto:
            # 自动模式
            app.run_full_pipeline()
            app.visualize_results()
        else:
            # 交互式模式
            app.run_interactive()
            
    except Exception as e:
        print(f"\n[ERR] 程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

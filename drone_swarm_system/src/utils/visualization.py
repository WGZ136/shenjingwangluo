"""
可视化工具
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class Visualizer:
    """可视化器"""
    
    def __init__(self):
        pass
    
    def plot_trajectory(self, poses_3d, save_path=None):
        """
        绘制3D轨迹
        
        Args:
            poses_3d: 位姿列表，每个元素是[x, y, z]
            save_path: 保存路径
        """
        if len(poses_3d) < 2:
            return
        
        poses_array = np.array(poses_3d)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(poses_array[:, 0], poses_array[:, 1], poses_array[:, 2], 
                'b-', linewidth=2, label='轨迹')
        
        # 绘制起点和终点
        ax.scatter(poses_array[0, 0], poses_array[0, 1], poses_array[0, 2], 
                  c='g', s=100, marker='o', label='起点')
        ax.scatter(poses_array[-1, 0], poses_array[-1, 1], poses_array[-1, 2], 
                  c='r', s=100, marker='s', label='终点')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('无人机轨迹')
        ax.legend()
        ax.grid(True)
        
        # 设置等比例
        max_range = np.array([
            poses_array[:, 0].max() - poses_array[:, 0].min(),
            poses_array[:, 1].max() - poses_array[:, 1].min(),
            poses_array[:, 2].max() - poses_array[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (poses_array[:, 0].max() + poses_array[:, 0].min()) * 0.5
        mid_y = (poses_array[:, 1].max() + poses_array[:, 1].min()) * 0.5
        mid_z = (poses_array[:, 2].max() + poses_array[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comparison_figure(self, images, titles, save_path=None):
        """
        创建对比图
        
        Args:
            images: 图像列表
            titles: 标题列表
            save_path: 保存路径
        """
        n_images = len(images)
        
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            if len(img.shape) == 3 and img.shape[2] == 3:
                axes[i].imshow(img)
            else:
                axes[i].imshow(img, cmap='gray')
            
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def animate_results(self, image_dir, output_video_path, fps=10):
        """
        创建结果动画
        
        Args:
            image_dir: 图像目录
            output_video_path: 输出视频路径
            fps: 帧率
        """
        # 获取所有图像
        image_files = sorted([f for f in os.listdir(image_dir) 
                             if f.endswith(('.png', '.jpg'))])
        
        if not image_files:
            print(f"在目录 {image_dir} 中未找到图像")
            return
        
        # 读取第一张图像获取尺寸
        first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
        height, width = first_image.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 写入所有图像
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)
        
        video_writer.release()
        print(f"视频已保存到: {output_video_path}")
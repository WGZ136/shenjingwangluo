# 基于视觉感知的无人机集群协同演化系统

## 项目简介

本项目是一个集成了深度估计、光流估计、聚类分割和位姿估计的无人机集群视觉感知系统。

### 核心功能

- **深度估计**：基于Monodepth2的单目深度估计
- **光流估计**：基于RAFT的光流计算
- **聚类分割**：K-means/DBSCAN聚类算法
- **位姿估计**：RANSAC/SVD位姿估计
- **可视化**：结果可视化与保存

## 快速开始

### 安装依赖

```bash
# 1. 安装PyTorch（根据CUDA版本选择）
# CUDA 12.1:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# CPU版本:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# 2. 安装其他依赖
pip install -r requirements.txt
```

### 基本使用

```python
from drone_swarm_system import DroneSwarmSystem

# 方式1：使用上下文管理器（推荐）
with DroneSwarmSystem() as system:
    result = system.process_frame(image)
    if result.success:
        print(f"处理成功，耗时: {result.processing_time:.3f}秒")
        print(f"深度图: {result.depth_map is not None}")
        print(f"光流向量: {result.flow_vectors is not None}")

# 方式2：手动管理生命周期
system = DroneSwarmSystem()
system.initialize()
result = system.process_frame(image)
system.release()
```

## API文档

### 主类：DroneSwarmSystem

#### 构造函数

```python
DroneSwarmSystem(config=None)
```

**参数：**
- `config` (dict, optional): 自定义配置，覆盖默认配置

**默认配置：**
```python
{
    'depth': {
        'enabled': True,           # 是否启用深度估计
        'model_path': 'models/mono_640x192',
        'device': 'cuda'           # 或 'cpu'
    },
    'flow': {
        'enabled': True,           # 是否启用光流估计
        'use_gpu': True            # 是否使用GPU
    },
    'clustering': {
        'enabled': True,
        'method': 'kmeans',        # 或 'dbscan'
        'n_clusters': 3
    },
    'pose': {
        'enabled': True,
        'method': 'ransac'
    }
}
```

#### 主要方法

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `initialize()` | 初始化所有模块 | bool: 是否成功 |
| `process_frame(current_frame, previous_frame=None)` | 处理单帧图像 | ProcessingResult |
| `process_video(video_path, output_path=None)` | 处理视频文件 | List[ProcessingResult] |
| `get_status()` | 获取系统状态 | dict |
| `reset()` | 重置系统状态 | None |
| `release()` | 释放资源 | None |

#### 上下文管理器支持

```python
with DroneSwarmSystem(config) as system:
    # 自动调用initialize()和release()
    result = system.process_frame(image)
```

### 便捷函数

```python
from drone_swarm_system import create_system, quick_process

# 快速创建系统实例
system = create_system(config={'depth': {'enabled': True}})

# 快速处理单帧（自动管理生命周期）
result = quick_process(image, prev_image=None, config=None)
```

### 处理结果：ProcessingResult

```python
@dataclass
class ProcessingResult:
    depth_map: np.ndarray          # 深度图 (H, W)
    flow_vectors: np.ndarray       # 光流向量 (H, W, 2)
    segmentation_labels: np.ndarray # 分割标签
    pose_transform: dict           # 位姿变换 {'rotation': ..., 'translation': ...}
    cluster_centers: list          # 聚类中心
    processing_time: float         # 处理时间（秒）
    success: bool                  # 是否成功
    message: str                   # 状态消息
```

### CUDA检测

```python
from drone_swarm_system import (
    check_cuda_available,  # 函数：检测CUDA
    CUDA_AVAILABLE,        # bool: CUDA是否可用
    DEFAULT_DEVICE         # str: 默认设备 ('cuda' 或 'cpu')
)

# 检测CUDA
cuda_available, device = check_cuda_available()
print(f"CUDA可用: {cuda_available}, 设备: {device}")
```

## 项目结构

```
drone_swarm_system/
├── core/
│   ├── system.py      # 主系统类
│   └── config.py      # 配置管理
└── src/
    ├── modules/       # 功能模块
    │   ├── depth_estimator.py
    │   ├── flow_processor.py
    │   ├── clustering.py
    │   └── pose_estimator.py
    └── utils/         # 工具函数
        ├── visualization.py
        └── camera_utils.py
```

## 使用示例

### 示例1：基本图像处理

```python
import cv2
from drone_swarm_system import DroneSwarmSystem

# 读取图像
image = cv2.imread('test.jpg')

# 处理图像
with DroneSwarmSystem() as system:
    result = system.process_frame(image)
    
    if result.success:
        # 获取深度图
        depth = result.depth_map
        
        # 保存结果
        if result.depth_map is not None:
            cv2.imwrite('depth.png', result.depth_map)
```

### 示例2：视频处理

```python
from drone_swarm_system import DroneSwarmSystem

with DroneSwarmSystem() as system:
    results = system.process_video(
        video_path='input.mp4',
        output_path='output.mp4',
        max_frames=100
    )
    print(f"处理了 {len(results)} 帧")
```

### 示例3：自定义配置

```python
from drone_swarm_system import DroneSwarmSystem

# 只启用深度估计，使用CPU
config = {
    'depth': {'enabled': True, 'device': 'cpu'},
    'flow': {'enabled': False},
    'clustering': {'enabled': False},
    'pose': {'enabled': False}
}

with DroneSwarmSystem(config) as system:
    result = system.process_frame(image)
```

## 注意事项

1. **PyTorch安装**：必须根据CUDA版本手动安装PyTorch
2. **模型文件**：确保模型文件存在于 `data/model_weights/` 目录
3. **CUDA兼容性**：程序会自动检测CUDA，不可用时切换到CPU
4. **内存管理**：使用上下文管理器或手动调用 `release()` 释放显存

## 依赖项

- Python >= 3.8
- PyTorch >= 2.0
- OpenCV
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Pillow

详见 `requirements.txt`

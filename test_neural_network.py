#!/usr/bin/env python3
"""
神经网络加载诊断脚本

用于排查 Monodepth2 加载问题
"""

import sys
from pathlib import Path

print("="*70)
print("🧠 神经网络加载诊断")
print("="*70)

# 检查 Python 版本
print(f"\n1. Python 版本: {sys.version}")

# 检查 PyTorch
print("\n2. 检查 PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch 已安装: {torch.__version__}")
    print(f"   📍 CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   📍 CUDA 版本: {torch.version.cuda}")
except ImportError as e:
    print(f"   ❌ PyTorch 未安装: {e}")
    sys.exit(1)

# 检查 torchvision
print("\n3. 检查 torchvision...")
try:
    import torchvision
    print(f"   ✅ torchvision 已安装: {torchvision.__version__}")
except ImportError as e:
    print(f"   ❌ torchvision 未安装: {e}")

# 检查 PIL
print("\n4. 检查 PIL...")
try:
    from PIL import Image
    print(f"   ✅ PIL 已安装")
except ImportError as e:
    print(f"   ❌ PIL 未安装: {e}")

# 检查路径
print("\n5. 检查 Monodepth2 路径...")
project_root = Path(__file__).parent.absolute()
monodepth_path = project_root / "core_algorithms" / "monodepth2"
print(f"   项目根目录: {project_root}")
print(f"   Monodepth2 路径: {monodepth_path}")
print(f"   路径存在: {monodepth_path.exists()}")

# 检查关键文件
print("\n6. 检查关键文件...")
key_files = [
    "networks/__init__.py",
    "networks/resnet_encoder.py",
    "networks/depth_decoder.py",
    "layers.py",
    "utils.py"
]
for file in key_files:
    file_path = monodepth_path / file
    status = "✅" if file_path.exists() else "❌"
    print(f"   {status} {file}")

# 尝试导入
print("\n7. 尝试导入 Monodepth2 模块...")
sys.path.insert(0, str(project_root / "core_algorithms"))
try:
    from monodepth2.networks import ResnetEncoder, DepthDecoder
    print("   ✅ ResnetEncoder 导入成功")
    print("   ✅ DepthDecoder 导入成功")
except ImportError as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)

try:
    from monodepth2.layers import disp_to_depth
    print("   ✅ disp_to_depth 导入成功")
except ImportError as e:
    print(f"   ❌ 导入失败: {e}")

try:
    from monodepth2.utils import download_model_if_doesnt_exist
    print("   ✅ download_model_if_doesnt_exist 导入成功")
except ImportError as e:
    print(f"   ❌ 导入失败: {e}")

# 检查模型文件
print("\n8. 检查模型文件...")
model_path = Path("models/mono_640x192")
encoder_path = model_path / "encoder.pth"
decoder_path = model_path / "depth.pth"

print(f"   模型目录: {model_path.absolute()}")
print(f"   目录存在: {model_path.exists()}")
if model_path.exists():
    print(f"   encoder.pth: {encoder_path.exists()}")
    print(f"   depth.pth: {decoder_path.exists()}")
else:
    print("   ⚠️  模型文件不存在，将尝试自动下载")

# 尝试加载模型
print("\n9. 尝试加载模型...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")
    
    # 下载模型（如果不存在）
    print("   检查/下载模型...")
    download_model_if_doesnt_exist(str(model_path))
    
    # 加载编码器
    print("   加载编码器...")
    encoder = ResnetEncoder(18, False)
    if encoder_path.exists():
        encoder_weights = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict({k: v for k, v in encoder_weights.items() 
                                 if k in encoder.state_dict()})
        encoder.to(device)
        encoder.eval()
        print("   ✅ 编码器加载成功")
    else:
        print(f"   ❌ 编码器权重不存在: {encoder_path}")
    
    # 加载解码器
    print("   加载解码器...")
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    if decoder_path.exists():
        decoder_weights = torch.load(decoder_path, map_location=device)
        depth_decoder.load_state_dict(decoder_weights)
        depth_decoder.to(device)
        depth_decoder.eval()
        print("   ✅ 解码器加载成功")
    else:
        print(f"   ❌ 解码器权重不存在: {decoder_path}")
    
    print("\n" + "="*70)
    print("✅ 所有检查通过！神经网络可以正常使用")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n现在可以运行主程序:")
print("  .\\venv\\Scripts\\python.exe main.py")

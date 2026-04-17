#!/usr/bin/env python3
"""
环境设置脚本

检查并安装项目所需的依赖

使用方法:
    python scripts/setup_env.py
"""

import sys
import subprocess
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# 必需的依赖
REQUIRED_PACKAGES = {
    "numpy": "NumPy",
    "cv2": "OpenCV-Python",
    "torch": "PyTorch",
    "PIL": "Pillow",
    "requests": "Requests",
    "yaml": "PyYAML"
}

# 可选依赖
OPTIONAL_PACKAGES = {
    "open3d": "Open3D (用于3D可视化)",
    "matplotlib": "Matplotlib (用于绘图)",
    "sklearn": "scikit-learn (用于聚类)"
}


def check_package(package_name: str) -> bool:
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str) -> bool:
    """安装包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return True
    except subprocess.CalledProcessError:
        return False


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    print("✅ Python版本符合要求")
    return True


def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            return False
    except:
        print("⚠️ 无法检查CUDA状态")
        return False


def main():
    print("=" * 60)
    print("🔧 环境设置工具")
    print("=" * 60)
    
    # 检查Python版本
    print("\n[1/4] 检查Python版本...")
    if not check_python_version():
        return
    
    # 检查CUDA
    print("\n[2/4] 检查CUDA...")
    check_cuda()
    
    # 检查必需依赖
    print("\n[3/4] 检查必需依赖...")
    missing_required = []
    
    for package, name in REQUIRED_PACKAGES.items():
        if check_package(package):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} (未安装)")
            missing_required.append((package, name))
    
    # 安装缺失的必需依赖
    if missing_required:
        print(f"\n将安装 {len(missing_required)} 个缺失的依赖...")
        for package, name in missing_required:
            print(f"\n  安装 {name}...")
            if install_package(name.lower().replace(" ", "-")):
                print(f"  ✅ {name} 安装成功")
            else:
                print(f"  ❌ {name} 安装失败")
    else:
        print("\n✅ 所有必需依赖已安装")
    
    # 检查可选依赖
    print("\n[4/4] 检查可选依赖...")
    missing_optional = []
    
    for package, name in OPTIONAL_PACKAGES.items():
        if check_package(package):
            print(f"  ✅ {name}")
        else:
            print(f"  ⚠️ {name} (未安装)")
            missing_optional.append((package, name))
    
    # 询问是否安装可选依赖
    if missing_optional:
        print(f"\n发现 {len(missing_optional)} 个可选依赖未安装")
        response = input("是否安装可选依赖? (y/N): ")
        
        if response.lower() == 'y':
            for package, name in missing_optional:
                # 特殊处理包名
                install_name = package
                if package == "sklearn":
                    install_name = "scikit-learn"
                
                print(f"\n  安装 {name}...")
                if install_package(install_name):
                    print(f"  ✅ {name} 安装成功")
                else:
                    print(f"  ❌ {name} 安装失败")
    
    # 检查项目路径
    print("\n检查项目路径...")
    if (PROJECT_ROOT / "drone_swarm_system").exists():
        print("  ✅ 项目路径正确")
    else:
        print("  ⚠️ 项目路径可能不正确")
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 环境设置完成")
    print("=" * 60)
    print("\n现在可以运行:")
    print("  python main.py --help")
    print("  python examples/esp32_receiver_demo.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
模型下载脚本

自动下载Monodepth2和RAFT的预训练模型

使用方法:
    python scripts/download_models.py

将下载到 data/models/ 目录
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import tarfile

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models"

# 模型下载链接
MODELS = {
    "monodepth2": {
        "url": "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
        "filename": "mono+stereo_640x192.zip",
        "extract_dir": "monodepth2"
    },
    "raft": {
        "url": "https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth",
        "filename": "raft-things.pth",
        "extract_dir": "raft"
    }
}


def download_file(url: str, destination: Path, show_progress: bool = True):
    """下载文件"""
    print(f"下载: {url}")
    print(f"保存到: {destination}")
    
    def progress_hook(count, block_size, total_size):
        if show_progress and total_size > 0:
            percent = min(int(count * block_size * 100 / total_size), 100)
            print(f"\r进度: {percent}%", end="")
    
    urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
    print("\n✅ 下载完成")


def extract_archive(archive_path: Path, extract_dir: Path):
    """解压压缩包"""
    print(f"解压: {archive_path}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    
    print("✅ 解压完成")


def download_model(name: str, model_info: dict):
    """下载单个模型"""
    print(f"\n{'=' * 60}")
    print(f"📦 下载模型: {name}")
    print(f"{'=' * 60}")
    
    # 创建模型目录
    model_dir = MODEL_DIR / model_info["extract_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载文件
    download_path = model_dir / model_info["filename"]
    
    if download_path.exists():
        print(f"⚠️ 文件已存在: {download_path}")
        response = input("是否重新下载? (y/N): ")
        if response.lower() != 'y':
            print("跳过下载")
            return
    
    try:
        download_file(model_info["url"], download_path)
        
        # 解压（如果是压缩包）
        if download_path.suffix in ['.zip', '.tar', '.gz', '.tgz']:
            extract_archive(download_path, model_dir)
            # 删除压缩包
            download_path.unlink()
        
        print(f"✅ {name} 模型准备完成")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")


def main():
    print("=" * 60)
    print("🤖 模型下载工具")
    print("=" * 60)
    print(f"\n模型将保存到: {MODEL_DIR}")
    
    # 创建模型目录
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 选择要下载的模型
    print("\n可用模型:")
    for i, name in enumerate(MODELS.keys(), 1):
        print(f"  {i}. {name}")
    print("  0. 下载全部")
    
    choice = input("\n请选择 (0-2): ").strip()
    
    if choice == "0":
        for name, info in MODELS.items():
            download_model(name, info)
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(MODELS):
            name = list(MODELS.keys())[idx]
            download_model(name, MODELS[name])
        else:
            print("❌ 无效选择")
    else:
        print("❌ 无效输入")
    
    print("\n" + "=" * 60)
    print("✅ 模型下载完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

# HAJIMI 项目设置脚本
# 安装依赖并验证环境

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_cmd(cmd, check=True):
    """运行命令"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        print(f"命令失败: {cmd}")
        return False
    return True


def install_dependencies():
    """安装依赖"""
    print("=" * 50)
    print("安装依赖")
    print("=" * 50)

    # 基础依赖
    basic_deps = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cu121",
        "transformers>=4.45.0",
        "accelerate>=0.27.0",
        "peft>=0.13.0",
        "ultralytics>=8.1.0",
        "gradio>=4.0.0",
        "plotly>=5.18.0",
        "pillow",
        "pyyaml",
        "requests",
        "qwen-vl-utils",
    ]

    for dep in basic_deps:
        run_cmd(f"{sys.executable} -m pip install {dep}", check=False)

    # CLIP (从 GitHub)
    run_cmd(f"{sys.executable} -m pip install git+https://github.com/openai/CLIP.git", check=False)

    # bitsandbytes (仅 Linux 可用)
    if sys.platform == "linux":
        run_cmd(f"{sys.executable} -m pip install bitsandbytes>=0.42.0", check=False)
    else:
        print("注意: bitsandbytes 仅在 Linux 上完全支持")
        print("Windows 用户可能需要使用 FP16 而非 4-bit 量化")


def create_directories():
    """创建必要的目录"""
    print("\n" + "=" * 50)
    print("创建目录结构")
    print("=" * 50)

    dirs = [
        "data/cat_images",
        "data/eval_set",
        "data/train",
        "models/monologue_lora",
    ]

    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {d}")


def verify_environment():
    """验证环境"""
    print("\n" + "=" * 50)
    print("验证环境")
    print("=" * 50)

    checks = []

    # PyTorch
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"✓ PyTorch {torch.__version__} (CUDA: {cuda})")
        if cuda:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        checks.append(True)
    except ImportError:
        print("✗ PyTorch 未安装")
        checks.append(False)

    # Transformers
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Transformers 未安装")
        checks.append(False)

    # CLIP
    try:
        import clip
        print("✓ CLIP")
        checks.append(True)
    except ImportError:
        print("✗ CLIP 未安装")
        checks.append(False)

    # Ultralytics (YOLO)
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics (YOLO)")
        checks.append(True)
    except ImportError:
        print("✗ Ultralytics 未安装")
        checks.append(False)

    # Gradio
    try:
        import gradio
        print(f"✓ Gradio {gradio.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Gradio 未安装")
        checks.append(False)

    return all(checks)


def main():
    print("HAJIMI 项目设置")
    print("=" * 50)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="安装依赖")
    parser.add_argument("--verify", action="store_true", help="验证环境")
    args = parser.parse_args()

    if args.install:
        install_dependencies()

    create_directories()

    if args.verify or not args.install:
        success = verify_environment()
        print("\n" + "=" * 50)
        if success:
            print("✓ 环境验证通过!")
            print("\n下一步:")
            print("  1. 运行 python test_modules.py a  测试 YOLO")
            print("  2. 运行 python test_modules.py all 测试全部模块")
            print("  3. 运行 python app.py 启动 Demo")
        else:
            print("✗ 环境验证失败，请安装缺少的依赖")
            print("  运行: python setup.py --install")


if __name__ == "__main__":
    main()

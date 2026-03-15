"""
ComfyUI RunningHub Agent 插件自动依赖安装脚本
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def sync_models_registry():
    """
    同步 models_registry.json，使插件无需依赖 ComfyUI_RH_OpenAPI 即可独立运行。

    优先级：
      1. 从同级 ComfyUI_RH_OpenAPI 目录复制（保持最新版本）
      2. 本地已有副本（直接使用）
    """
    plugin_root = Path(__file__).resolve().parent
    local_path  = plugin_root / "models_registry.json"
    rh_path     = plugin_root.parent / "ComfyUI_RH_OpenAPI" / "models_registry.json"

    if rh_path.exists():
        try:
            shutil.copy2(str(rh_path), str(local_path))
            print(f"[RH_Agent] 已从 ComfyUI_RH_OpenAPI 同步 models_registry.json")
            return True
        except Exception as e:
            print(f"[RH_Agent] 从 ComfyUI_RH_OpenAPI 复制失败: {e}")

    if local_path.exists():
        print(f"[RH_Agent] 使用本地 models_registry.json")
        return True

    print(f"[RH_Agent] 警告：未找到 models_registry.json，请将 ComfyUI_RH_OpenAPI 与本插件安装在同一 custom_nodes 目录下")
    return False


def install_dependencies():
    """安装插件所需的依赖"""

    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if not os.path.exists(requirements_file):
        print("[RH_Agent] requirements.txt 文件不存在")
        return False

    print("[RH_Agent] 检查依赖...")

    try:
        import openai
        import requests
        from PIL import Image
        print("[RH_Agent] 所有依赖已安装")
        return True
    except ImportError:
        print("[RH_Agent] 发现缺失依赖，开始安装...")

    try:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file,
            "--quiet"
        ])
        print("[RH_Agent] 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[RH_Agent] 依赖安装失败: {e}")
        print("[RH_Agent] 请手动运行: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    install_dependencies()

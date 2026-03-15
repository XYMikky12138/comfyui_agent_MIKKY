"""
ComfyUI RunningHub Agent 插件自动依赖安装脚本
"""

import subprocess
import sys
import os


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

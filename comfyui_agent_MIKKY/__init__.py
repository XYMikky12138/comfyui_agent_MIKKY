"""
ComfyUI RunningHub 批量生图 Agent 插件
通过 RunningHub OpenAPI 进行异步并发图像批量生成
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

WEB_DIRECTORY = "./web"

try:
    from .install import install_dependencies
    install_dependencies()
except Exception as e:
    print(f"[RH_Agent] 依赖检查失败: {e}")
    print("[RH_Agent] 请手动安装依赖: pip install -r requirements.txt")

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
    print("[RH_Agent] [OK] 节点加载成功")
    print(f"[RH_Agent] [OK] Web 目录注册: {WEB_DIRECTORY}")
except ImportError as e:
    print(f"[RH_Agent] [FAIL] 节点加载失败: {e}")
    if "torch" in str(e):
        print("[RH_Agent] 注意: 此插件需要在 ComfyUI 环境中运行")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

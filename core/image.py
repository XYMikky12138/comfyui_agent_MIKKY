"""
图像工具函数 - ComfyUI tensor ↔ PIL Image ↔ bytes。
"""

from io import BytesIO
from typing import Union
import numpy as np


def tensor_to_png_bytes(tensor) -> bytes:
    """
    将 ComfyUI IMAGE tensor 转换为 PNG bytes。

    Args:
        tensor: shape=(H,W,C) 或 (1,H,W,C)，值范围 [0,1]，dtype=float32

    Returns:
        PNG 格式的 bytes
    """
    import torch
    from PIL import Image

    if isinstance(tensor, torch.Tensor):
        # 如果是 batch，取第一帧
        if tensor.dim() == 4:
            tensor = tensor[0]
        # 转换为 numpy，[0,1] → [0,255]
        img_np = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    elif isinstance(tensor, np.ndarray):
        if tensor.ndim == 4:
            tensor = tensor[0]
        img_np = (tensor * 255).clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"不支持的 tensor 类型: {type(tensor)}")

    pil_image = Image.fromarray(img_np, mode="RGB")
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def url_to_tensor(url: str, timeout: int = 30):
    """
    从 URL 下载图片并转换为 ComfyUI IMAGE tensor。

    Args:
        url: 图片 URL
        timeout: 下载超时（秒）

    Returns:
        tensor: shape=(1,H,W,C)，值范围 [0,1]，dtype=float32
    """
    import requests
    import torch
    from PIL import Image

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1,H,W,C)
    return tensor


def tensors_to_batch(tensors: list):
    """
    将多个 ComfyUI IMAGE tensor 合并为一个 batch。

    自动将所有图片 resize 到第一张图片的尺寸。

    Returns:
        tensor: shape=(N,H,W,C)
    """
    import torch
    import torch.nn.functional as F

    if not tensors:
        return None

    if len(tensors) == 1:
        t = tensors[0]
        if t.dim() == 3:
            return t.unsqueeze(0)
        return t

    # 统一使用第一张图片的尺寸
    first = tensors[0]
    if first.dim() == 4:
        first = first[0]
    target_h, target_w = first.shape[0], first.shape[1]

    resized = []
    for t in tensors:
        if t.dim() == 4:
            t = t[0]
        if t.shape[0] != target_h or t.shape[1] != target_w:
            # (H,W,C) → (1,C,H,W) → resize → (1,H,W,C)
            t_4d = t.permute(2, 0, 1).unsqueeze(0)
            t_4d = F.interpolate(t_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)
            t = t_4d.squeeze(0).permute(1, 2, 0)
        resized.append(t)

    return torch.stack(resized, dim=0)  # (N,H,W,C)

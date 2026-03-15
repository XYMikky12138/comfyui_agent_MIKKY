"""
URL 转 IMAGE 工具 - 从 URL 下载图片并转换为 ComfyUI IMAGE 格式
"""

import requests
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from typing import Optional

from .config import IMAGE_DOWNLOAD_TIMEOUT, SUPPORTED_IMAGE_FORMATS


def url_to_comfyui_image(url: str, timeout: int = IMAGE_DOWNLOAD_TIMEOUT) -> Optional[torch.Tensor]:
    """
    从 URL 下载图片并转换为 ComfyUI IMAGE 格式
    
    Args:
        url: 图片 URL
        timeout: 下载超时时间（秒）
        
    Returns:
        ComfyUI IMAGE tensor [1, H, W, C]，值范围 [0, 1]
        失败返回 None
    """
    try:
        print(f"[URLConverter] 下载图片: {url[:50]}...")
        
        # 下载图片
        response = requests.get(url, timeout=timeout)
        
        if not response.ok:
            print(f"[URLConverter] ❌ 下载失败: HTTP {response.status_code}")
            return None
        
        # 读取图片
        img_bytes = BytesIO(response.content)
        pil_image = Image.open(img_bytes)
        
        # 转换为 RGB（确保 3 通道）
        if pil_image.mode != 'RGB':
            print(f"[URLConverter] 转换图片模式: {pil_image.mode} → RGB")
            pil_image = pil_image.convert('RGB')
        
        # 转为 numpy 数组
        img_np = np.array(pil_image).astype(np.float32)
        
        # 归一化到 [0, 1]
        img_np = img_np / 255.0
        
        # 转为 torch tensor 并添加 batch 维度
        # ComfyUI 格式: [batch, height, width, channels]
        img_tensor = torch.from_numpy(img_np)[None, ]
        
        print(f"[URLConverter] ✅ 图片转换成功: {img_tensor.shape}")
        return img_tensor
        
    except requests.exceptions.Timeout:
        print(f"[URLConverter] ❌ 下载超时（{timeout}秒）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[URLConverter] ❌ 下载失败: {str(e)}")
        return None
    except Exception as e:
        print(f"[URLConverter] ❌ 图片转换失败: {str(e)}")
        return None


def batch_url_to_comfyui_images(
    urls: list,
    timeout: int = IMAGE_DOWNLOAD_TIMEOUT
) -> list:
    """
    批量转换 URLs 为 ComfyUI IMAGEs
    
    Args:
        urls: 图片 URL 列表
        timeout: 下载超时时间（秒）
        
    Returns:
        ComfyUI IMAGE tensor 列表，失败的为 None
    """
    results = []
    total = len(urls)
    
    print(f"[URLConverter] 批量转换: 共 {total} 张图片")
    
    for i, url in enumerate(urls):
        print(f"[URLConverter] 转换 {i+1}/{total}...")
        img_tensor = url_to_comfyui_image(url, timeout)
        results.append(img_tensor)
    
    success_count = sum(1 for img in results if img is not None)
    print(f"[URLConverter] ✅ 批量转换完成: 成功 {success_count}/{total}")
    
    return results


def comfyui_images_to_batch(images: list) -> Optional[torch.Tensor]:
    """
    合并多个 ComfyUI IMAGE 为一个 batch tensor
    
    Args:
        images: ComfyUI IMAGE tensor 列表 [1, H, W, C]
        
    Returns:
        合并后的 tensor [N, H, W, C]，失败返回 None
    """
    # 过滤掉 None
    valid_images = [img for img in images if img is not None]
    
    if not valid_images:
        print("[URLConverter] ❌ 没有有效图片可合并")
        return None
    
    try:
        # 检查所有图片尺寸是否一致
        shapes = [img.shape for img in valid_images]
        if len(set(shapes)) > 1:
            print(f"[URLConverter] ⚠️ 图片尺寸不一致: {shapes}")
            # 尝试统一尺寸到第一张图片的尺寸
            target_shape = shapes[0]
            target_h, target_w = target_shape[1], target_shape[2]
            
            resized_images = []
            for img in valid_images:
                if img.shape != target_shape:
                    # 需要 resize
                    img_np = img[0].numpy()  # [H, W, C]
                    pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
                    pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    img_np = np.array(pil_img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np)[None, ]
                    resized_images.append(img_tensor)
                else:
                    resized_images.append(img)
            
            valid_images = resized_images
            print(f"[URLConverter] ✅ 已统一图片尺寸: {target_shape}")
        
        # 合并
        batch_tensor = torch.cat(valid_images, dim=0)
        print(f"[URLConverter] ✅ 合并完成: {batch_tensor.shape}")
        return batch_tensor
        
    except Exception as e:
        print(f"[URLConverter] ❌ 合并失败: {str(e)}")
        return None


def create_placeholder_image(width: int = 512, height: int = 512, color: tuple = (128, 128, 128)) -> torch.Tensor:
    """
    创建占位图片（用于失败情况）
    
    Args:
        width: 宽度
        height: 高度
        color: RGB 颜色值 (0-255)
        
    Returns:
        ComfyUI IMAGE tensor
    """
    # 创建纯色图片
    img_np = np.full((height, width, 3), color, dtype=np.uint8)
    img_np = img_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None, ]
    
    print(f"[URLConverter] 创建占位图片: {width}x{height}, 颜色={color}")
    return img_tensor


# 便捷函数：带占位符的批量转换
def batch_url_to_comfyui_images_safe(
    urls: list,
    use_placeholder: bool = True,
    placeholder_color: tuple = (64, 64, 64)
) -> torch.Tensor:
    """
    批量转换 URLs 为 ComfyUI IMAGEs（失败时使用占位图）
    
    Args:
        urls: 图片 URL 列表
        use_placeholder: 是否使用占位图代替失败的图片
        placeholder_color: 占位图颜色
        
    Returns:
        ComfyUI IMAGE batch tensor
    """
    results = batch_url_to_comfyui_images(urls)
    
    if use_placeholder:
        # 替换失败的图片为占位图
        # 确定目标尺寸（使用第一张成功的图片）
        target_shape = None
        for img in results:
            if img is not None:
                target_shape = img.shape
                break
        
        if target_shape:
            target_h, target_w = target_shape[1], target_shape[2]
        else:
            # 所有图片都失败，使用默认尺寸
            target_h, target_w = 512, 512
        
        # 替换 None
        for i in range(len(results)):
            if results[i] is None:
                print(f"[URLConverter] 使用占位图代替失败的图片 #{i+1}")
                results[i] = create_placeholder_image(target_w, target_h, placeholder_color)
    
    # 合并
    return comfyui_images_to_batch(results)


# 向后兼容的别名（供 nodes.py 使用）
url_to_image = url_to_comfyui_image
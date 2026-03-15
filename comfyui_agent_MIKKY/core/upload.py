"""
图片上传 - 将 ComfyUI IMAGE tensor 上传到 RunningHub OSS。

POST {base_url}/media/upload/binary
"""

import time
import requests
from io import BytesIO
from typing import List, Union


def _log(msg: str):
    print(f"[RH_Agent_Upload] {msg}")


def upload_file(
    file_content: Union[bytes, BytesIO],
    filename: str,
    mime_type: str,
    api_key: str,
    base_url: str,
    timeout: int = 60,
    max_retries: int = 3,
) -> str:
    """
    上传单个文件到 RunningHub /media/upload/binary。

    Returns:
        download_url
    """
    url = f"{base_url.rstrip('/')}/media/upload/binary"
    headers = {"Authorization": f"Bearer {api_key}"}

    if isinstance(file_content, BytesIO):
        file_content = file_content.getvalue()

    files = {"file": (filename, file_content, mime_type)}

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = min(2 ** attempt, 30)
                _log(f"上传重试 {attempt + 1}/{max_retries}，等待 {wait}s...")
                time.sleep(wait)

            response = requests.post(url, headers=headers, files=files, timeout=timeout)
            data = response.json() if response.text else {}

            if response.status_code != 200:
                err_msg = data.get("message", response.text[:200])
                last_error = RuntimeError(f"HTTP {response.status_code}: {err_msg}")
                if response.status_code >= 500 or response.status_code == 429:
                    _log(f"第 {attempt + 1} 次 HTTP {response.status_code}，重试中...")
                    continue
                raise last_error

            if data.get("code") != 0:
                err_msg = data.get("message", "上传失败")
                last_error = RuntimeError(err_msg)
                if "server" in err_msg.lower() or "internal" in err_msg.lower():
                    _log(f"第 {attempt + 1} 次服务器错误，重试中...")
                    continue
                raise last_error

            download_url = (data.get("data") or {}).get("download_url")
            if not download_url:
                raise RuntimeError("响应中无 download_url")

            return download_url

        except requests.exceptions.RequestException as e:
            last_error = RuntimeError(f"网络错误: {type(e).__name__}: {e}")
            _log(f"第 {attempt + 1} 次网络错误: {type(e).__name__}")
            continue
        except RuntimeError:
            raise
        except Exception as e:
            last_error = RuntimeError(f"意外错误: {e}")
            _log(f"第 {attempt + 1} 次意外错误: {e}")
            continue

    raise RuntimeError(f"上传失败（已重试 {max_retries} 次）: {last_error}")


def upload_tensor(
    tensor,
    api_key: str,
    base_url: str,
    timeout: int = 60,
    index: int = 0,
) -> str:
    """
    将 ComfyUI IMAGE tensor 上传到 RH OSS。

    Args:
        tensor: ComfyUI IMAGE tensor，shape=(H,W,C)，值范围 [0,1]，float32
        api_key: RunningHub API Key
        base_url: RunningHub API 基础地址
        timeout: 上传超时（秒）
        index: 图片序号（用于文件命名）

    Returns:
        上传后的图片 URL
    """
    from .image import tensor_to_png_bytes

    png_bytes = tensor_to_png_bytes(tensor)
    filename = f"input_{index}.png"
    return upload_file(png_bytes, filename, "image/png", api_key, base_url, timeout)


def upload_tensors(
    tensors: list,
    api_key: str,
    base_url: str,
    timeout: int = 60,
) -> List[str]:
    """
    批量上传多个 tensor，返回 URL 列表。
    """
    urls = []
    for i, tensor in enumerate(tensors):
        _log(f"上传图片 {i + 1}/{len(tensors)}...")
        url = upload_tensor(tensor, api_key, base_url, timeout, index=i)
        urls.append(url)
        _log(f"图片 {i + 1} 上传成功: {url[:60]}...")
    return urls

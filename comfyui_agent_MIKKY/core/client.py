"""
RunningHub API 客户端 - 任务提交与轮询。

submit: POST {base_url}/{endpoint}
poll:   POST {base_url}/query
"""

import time
import json
import requests
from typing import Optional, List, Callable, Dict, Tuple, Any

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_CANCEL = "CANCEL"
STATUS_RUNNING = "RUNNING"
STATUS_QUEUED = "QUEUED"
STATUS_CREATE = "CREATE"

MAX_CONSECUTIVE_POLL_FAILURES = 5
MAX_SUBMIT_RETRIES = 3


def _log(msg: str):
    print(f"[RH_Agent] {msg}")


def _is_retryable_error(error_msg: str, status_code: int = 0) -> bool:
    err_lower = str(error_msg).lower()
    non_retryable = [
        "violation", "illegal", "forbidden", "nsfw",
        "content policy", "unauthorized", "bad request",
        "content verification failed", "moderation",
        "invalid parameter", "parameter error",
        "balance", "insufficient", "quota",
    ]
    if any(kw in err_lower for kw in non_retryable):
        return False
    if status_code and 400 <= status_code < 500 and status_code != 429:
        return False
    return True


def submit_task(
    endpoint: str,
    payload: dict,
    api_key: str,
    base_url: str,
    timeout: int = 60,
    max_retries: int = MAX_SUBMIT_RETRIES,
) -> str:
    """
    提交任务到 RunningHub API。

    Returns:
        task_id
    """
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            wait = min(2 ** attempt + 1, 15)
            _log(f"提交重试 {attempt + 1}/{max_retries}，等待 {wait}s...")
            time.sleep(wait)

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_error = RuntimeError(f"提交失败：网络错误 ({type(e).__name__}: {e})")
            _log(f"提交网络错误（第 {attempt + 1} 次）: {type(e).__name__}")
            continue

        try:
            data = response.json() if response.text else {}
        except json.JSONDecodeError:
            if response.status_code != 200:
                last_error = RuntimeError(
                    f"提交失败：HTTP {response.status_code} [{response.text[:200]}]"
                )
                if _is_retryable_error("", response.status_code):
                    _log(f"提交 HTTP {response.status_code}（第 {attempt + 1} 次），重试中...")
                    continue
                raise last_error
            last_error = RuntimeError("提交失败：响应 JSON 格式无效")
            continue

        if response.status_code != 200:
            err_code = str(data.get("errorCode", ""))
            err_msg = data.get("errorMessage", response.text[:200]) or f"HTTP {response.status_code}"
            last_error = RuntimeError(f"提交失败：{err_msg} [errorCode: {err_code}]")
            if _is_retryable_error(err_msg, response.status_code):
                _log(f"提交错误（第 {attempt + 1} 次）: {err_msg[:100]}")
                continue
            raise last_error

        err_code = data.get("errorCode") or data.get("error_code") or ""
        err_msg = data.get("errorMessage") or data.get("error_message") or ""
        if err_code or err_msg:
            last_error = RuntimeError(f"提交失败：{err_msg or f'错误码 {err_code}'} [errorCode: {err_code}]")
            if _is_retryable_error(err_msg):
                _log(f"提交 API 错误（第 {attempt + 1} 次）: {err_msg[:100]}")
                continue
            raise last_error

        task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            raise RuntimeError("提交失败：响应中无任务 ID")

        return str(task_id)

    raise last_error or RuntimeError(f"提交失败：已重试 {max_retries} 次")


def poll_task(
    task_id: str,
    api_key: str,
    base_url: str,
    polling_interval: float = 5,
    max_polling_time: int = 600,
    on_progress: Optional[Callable[[int], None]] = None,
) -> Tuple[List[str], Dict]:
    """
    轮询任务结果。

    Returns:
        (result_urls, full_response) - URL 列表和完整响应字典
    """
    url = f"{base_url.rstrip('/')}/query"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {"taskId": task_id}

    start_time = time.time()
    consecutive_failures = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_polling_time:
            raise RuntimeError(
                f"轮询超时（{max_polling_time}s）[taskId: {task_id}]"
            )

        if on_progress:
            progress = min(int(30 + elapsed / max_polling_time * 55), 85)
            try:
                on_progress(progress)
            except Exception:
                pass

        time.sleep(polling_interval)

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            consecutive_failures += 1
            _log(f"轮询失败（{consecutive_failures}/{MAX_CONSECUTIVE_POLL_FAILURES}）: {type(e).__name__}")
            if consecutive_failures >= MAX_CONSECUTIVE_POLL_FAILURES:
                raise RuntimeError(f"多次网络错误后轮询失败 [taskId: {task_id}]")
            time.sleep(min(consecutive_failures * 2, 10))
            continue

        if response.status_code != 200:
            consecutive_failures += 1
            _log(f"轮询 HTTP {response.status_code}（{consecutive_failures}/{MAX_CONSECUTIVE_POLL_FAILURES}）")
            if consecutive_failures >= MAX_CONSECUTIVE_POLL_FAILURES:
                raise RuntimeError(
                    f"服务器连续返回 HTTP {response.status_code}，轮询失败 [taskId: {task_id}]"
                )
            time.sleep(min(consecutive_failures * 2, 10))
            continue

        try:
            data = response.json()
        except json.JSONDecodeError:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_POLL_FAILURES:
                raise RuntimeError(f"多次 JSON 解析失败 [taskId: {task_id}]")
            continue

        consecutive_failures = 0

        err_code = data.get("errorCode") or data.get("error_code") or ""
        err_msg = data.get("errorMessage") or data.get("error_message") or ""
        if err_code or err_msg:
            raise RuntimeError(
                f"任务失败：{err_msg or f'错误码 {err_code}'} [errorCode: {err_code}, taskId: {task_id}]"
            )

        status = (data.get("status") or "").strip().upper()

        if status == STATUS_SUCCESS:
            results = data.get("results") or []
            if not results:
                raise RuntimeError(f"响应中无结果 [taskId: {task_id}]")

            urls = []
            texts = []
            for r in results:
                u = r.get("url") or r.get("outputUrl")
                if u:
                    urls.append(u)
                t = r.get("text") or r.get("content") or r.get("output")
                if t:
                    texts.append(t)

            if not urls and not texts:
                raise RuntimeError(f"结果中无 URL 或文本 [taskId: {task_id}]")

            result_items = urls if urls else texts

            if on_progress:
                try:
                    on_progress(100)
                except Exception:
                    pass
            return result_items, data

        if status == STATUS_FAILED:
            raise RuntimeError(
                f"任务执行失败：{err_msg or '未知错误'} [errorCode: {err_code}, taskId: {task_id}]"
            )

        if status == STATUS_CANCEL:
            raise RuntimeError(f"任务已取消 [taskId: {task_id}]")

        if status and status not in (STATUS_CREATE, STATUS_QUEUED, STATUS_RUNNING):
            raise RuntimeError(f"未知任务状态：{status} [taskId: {task_id}]")


class RHClient:
    """RunningHub API 客户端，封装提交和轮询流程。"""

    def __init__(self, api_key: str, base_url: str, config: Dict[str, Any] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.config = config or {}
        self.timeout = self.config.get("timeout", 60)
        self.polling_interval = self.config.get("polling_interval", 5.0)
        self.max_polling_time = self.config.get("max_polling_time", 600)

    def generate(
        self,
        endpoint: str,
        payload: dict,
        task_name: str = "任务",
    ) -> Dict[str, Any]:
        """
        提交任务并等待结果。

        Returns:
            {
                "success": bool,
                "task_name": str,
                "task_id": str,
                "result_urls": [url, ...],
                "response": dict,
                "error": str  # 仅失败时
            }
        """
        try:
            _log(f"[{task_name}] 提交任务...")
            task_id = submit_task(
                endpoint=endpoint,
                payload=payload,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            _log(f"[{task_name}] taskId={task_id}，开始轮询...")

            result_urls, full_response = poll_task(
                task_id=task_id,
                api_key=self.api_key,
                base_url=self.base_url,
                polling_interval=self.polling_interval,
                max_polling_time=self.max_polling_time,
            )

            _log(f"[{task_name}] 完成，获得 {len(result_urls)} 个结果")
            return {
                "success": True,
                "task_name": task_name,
                "task_id": task_id,
                "result_urls": result_urls,
                "response": full_response,
            }

        except Exception as e:
            _log(f"[{task_name}] 失败: {e}")
            return {
                "success": False,
                "task_name": task_name,
                "task_id": None,
                "result_urls": [],
                "response": {},
                "error": str(e),
            }

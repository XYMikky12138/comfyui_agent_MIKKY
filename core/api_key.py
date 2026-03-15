"""
Config and API key resolution for RH Agent.

Priority for RH API key:
  1. Settings node (RH_OPENAPI_CONFIG connected)
  2. Direct node input (rh_api_key parameter)
  3. Environment variable RH_API_KEY
  4. config/.env file

Priority for VLM API key:
  1. Direct node input (vlm_api_key parameter)
  2. Environment variable VLM_API_KEY
  3. config/.env file
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

DEFAULT_TIMEOUT = 60
DEFAULT_POLLING_INTERVAL = 5.0
DEFAULT_MAX_POLLING_TIME = 600
DEFAULT_UPLOAD_TIMEOUT = 60
DEFAULT_BASE_URL = "https://www.runninghub.cn/openapi/v2"
DEFAULT_VLM_BASE_URL = "https://api.siliconflow.cn/v1"


def _get_plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_env_file() -> Dict[str, str]:
    """Load key=value pairs from config/.env."""
    env_path = _get_plugin_root() / "config" / ".env"
    result = {}
    if not env_path.exists():
        return result
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        result[key] = value
    except Exception:
        pass
    return result


def _extract_settings_config(api_config: Any) -> Optional[Dict]:
    """Extract config dict from RH_OPENAPI_CONFIG Settings node output."""
    if api_config is None:
        return None
    if isinstance(api_config, list) and len(api_config) > 0:
        item = api_config[0]
        if isinstance(item, dict):
            return item
    if isinstance(api_config, dict):
        return api_config
    return None


def _get_shared_api_key() -> Optional[str]:
    """Try to get shared API key from PromptServer (RunningHub system default)."""
    try:
        from server import PromptServer
        instance = PromptServer.instance
        api_key = getattr(instance, 'shared_api_key', None)
        if api_key and isinstance(api_key, str) and api_key.strip() and api_key != 'unknown':
            return api_key.strip()
    except Exception:
        pass
    return None


def _build_config(base_url: str, api_key: str, env_data: Dict[str, str]) -> Dict[str, Any]:
    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "timeout": int(env_data.get("RH_API_TIMEOUT", DEFAULT_TIMEOUT)),
        "polling_interval": float(env_data.get("RH_API_POLLING_INTERVAL", DEFAULT_POLLING_INTERVAL)),
        "max_polling_time": int(env_data.get("RH_API_MAX_POLLING_TIME", DEFAULT_MAX_POLLING_TIME)),
        "upload_timeout": int(env_data.get("RH_UPLOAD_TIMEOUT", DEFAULT_UPLOAD_TIMEOUT)),
    }


def get_rh_config(
    rh_settings: Any = None,
    rh_api_key_input: str = "",
) -> Dict[str, Any]:
    """
    Resolve RunningHub API config.

    Priority:
      1. rh_settings (RH_OPENAPI_CONFIG from Settings node)
      2. rh_api_key_input (direct node input)
      3. PromptServer shared_api_key (RunningHub system default)
      4. Environment variable RH_API_KEY
      5. config/.env file
    """
    env_data = _load_env_file()

    # Priority 1: Settings node
    if rh_settings is not None:
        c = _extract_settings_config(rh_settings)
        if c:
            base_url = (c.get("base_url") or "").strip()
            api_key = (c.get("apiKey") or c.get("api_key") or "").strip()
            if base_url and api_key:
                return _build_config(base_url, api_key, env_data)
            raise RuntimeError("RH Settings 节点：base_url 和 apiKey 都不能为空。")

    # Priority 2: Direct node input
    if rh_api_key_input and rh_api_key_input.strip():
        api_key = rh_api_key_input.strip()
        base_url = (
            os.environ.get("RH_API_BASE_URL", "").strip()
            or env_data.get("RH_API_BASE_URL", "").strip()
            or DEFAULT_BASE_URL
        )
        return _build_config(base_url, api_key, env_data)

    # Priority 3: PromptServer shared_api_key
    shared_key = _get_shared_api_key()
    if shared_key:
        print(f"[RH_Agent] 使用系统共享 API Key: ...{shared_key[-6:]}")
        base_url = (
            os.environ.get("RH_API_BASE_URL", "").strip()
            or env_data.get("RH_API_BASE_URL", "").strip()
            or DEFAULT_BASE_URL
        )
        return _build_config(base_url, shared_key, env_data)

    # Priority 4: Environment variables
    env_base_url = os.environ.get("RH_API_BASE_URL", "").strip()
    env_api_key = os.environ.get("RH_API_KEY", "").strip()
    if env_api_key:
        base_url = env_base_url or env_data.get("RH_API_BASE_URL", "").strip() or DEFAULT_BASE_URL
        return _build_config(base_url, env_api_key, env_data)

    # Priority 5: config/.env file
    file_api_key = env_data.get("RH_API_KEY", "").strip()
    file_base_url = env_data.get("RH_API_BASE_URL", "").strip()

    if not file_api_key:
        raise RuntimeError(
            "未找到 RunningHub API Key。\n"
            "请通过以下任一方式提供：\n"
            "  1. 连接 RH OpenAPI Settings 节点\n"
            "  2. 在节点的 rh_api_key 输入框中填写\n"
            "  3. 设置环境变量 RH_API_KEY\n"
            "  4. 在插件目录的 config/.env 文件中设置 RH_API_KEY"
        )

    base_url = file_base_url or DEFAULT_BASE_URL
    return _build_config(base_url, file_api_key, env_data)


def get_vlm_api_key(vlm_api_key_input: str = "") -> str:
    """
    Resolve SiliconFlow VLM API key.

    Priority:
      1. Direct node input (vlm_api_key_input)
      2. Environment variable VLM_API_KEY
      3. config/.env file
    """
    # Priority 1: Direct node input
    if vlm_api_key_input and vlm_api_key_input.strip():
        return vlm_api_key_input.strip()

    # Priority 2: Environment variable
    env_key = os.environ.get("VLM_API_KEY", "").strip()
    if env_key:
        return env_key

    # Priority 3: config/.env
    env_data = _load_env_file()
    file_key = env_data.get("VLM_API_KEY", "").strip()
    if file_key:
        return file_key

    return ""

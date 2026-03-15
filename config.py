"""
配置文件 - RunningHub Agent 插件常量
"""

# ==================== RunningHub API 配置 ====================

# RunningHub API 基础地址（实际值由 core/api_key.py 从 config/.env 或环境变量读取）
RH_DEFAULT_BASE_URL = "https://www.runninghub.cn/openapi/v2"

# ==================== VLM（SiliconFlow）配置 ====================

# SiliconFlow API 基础地址（OpenAI 兼容接口）
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# VLM API 超时时间（秒）
VLM_TIMEOUT = 120

# VLM API 重试次数
VLM_MAX_RETRIES = 2

# VLM 最大 tokens
VLM_MAX_TOKENS = 8888

# VLM 温度参数
VLM_TEMPERATURE = 0.7

# 可用的 VLM 模型列表
# 标注 [视觉] 表示支持图片输入（多模态），无标注为纯文本模型
AVAILABLE_VLM_MODELS = [
    "Pro/moonshotai/Kimi-K2.5",          # 视觉模型，推荐用于图生图场景
    "Qwen/Qwen3-VL-32B-Instruct",        # 视觉模型
    "Qwen/Qwen2-VL-72B-Instruct",        # 视觉模型
    "Pro/MiniMaxAI/MiniMax-M2.5",        # 纯文本，不支持视觉输入
    "Pro/zai-org/GLM-4.7",               # 纯文本
]

# 默认 VLM 模型（选用支持视觉的模型，以便参考图功能正常工作）
DEFAULT_VLM_MODEL = "Pro/moonshotai/Kimi-K2.5"

# ==================== 并发配置 ====================

# 默认最大并发数
DEFAULT_MAX_CONCURRENT = 50

# 默认最大队列大小
DEFAULT_MAX_QUEUE_SIZE = 10

# ==================== 图像配置 ====================

# 图片下载超时时间（秒）
IMAGE_DOWNLOAD_TIMEOUT = 60

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = ['PNG', 'JPEG', 'JPG', 'WEBP']

# ==================== 错误消息 ====================

ERROR_MESSAGES = {
    "no_api_key": "未提供 RH API Key，请在节点参数中填写或在 config/.env 中配置",
    "invalid_json": "JSON 格式错误，请检查语法",
    "validation_failed": "VLM 验证失败",
    "upload_failed": "图片上传到 RunningHub 失败，请检查 API Key 或网络连接",
    "task_creation_failed": "RunningHub 任务创建失败",
    "task_timeout": "任务执行超时",
    "download_failed": "生成图片下载失败",
}

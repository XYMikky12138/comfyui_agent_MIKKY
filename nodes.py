"""
ComfyUI RunningHub 批量生图 Agent 节点
支持从 models_registry.json 选择模型、批量并发调用 RunningHub OpenAPI、VLM 智能验证
"""

import json
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

from .config import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_VLM_MODEL,
    AVAILABLE_VLM_MODELS,
    ERROR_MESSAGES,
)
from .core.api_key import get_rh_config, get_vlm_api_key
from .core.client import RHClient
from .core.upload import upload_tensors
from .core.image import url_to_tensor, tensors_to_batch
from .vlm_validator import VLMValidator
from .task_manager import TaskManager


# ==================== 模型注册表加载 ====================

_PROMPT_KEYWORDS = ("prompt", "text", "description", "caption")


def _is_prompt_field(field_key: str, field_type: str) -> bool:
    """判断该字段是否为提示词字段（由 prompts_json 处理，不出现在 model_params 中）。"""
    if field_type.upper() != "STRING":
        return False
    return any(kw in field_key.lower() for kw in _PROMPT_KEYWORDS)


def _load_models_registry() -> List[Dict]:
    """
    加载 models_registry.json。
    优先从相邻的 ComfyUI_RH_OpenAPI 插件目录读取，若不存在则回退到本插件目录。
    从 ComfyUI_RH_OpenAPI 加载成功后，自动保存本地副本以支持独立运行。
    """
    import shutil

    plugin_root = Path(__file__).resolve().parent
    local_path  = plugin_root / "models_registry.json"
    rh_path     = plugin_root.parent / "ComfyUI_RH_OpenAPI" / "models_registry.json"

    candidates = [rh_path, local_path]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[RH_Agent] 加载模型注册表: {path}（共 {len(data)} 个模型）")
                # 从 ComfyUI_RH_OpenAPI 加载时，自动保存/更新本地副本
                if path == rh_path and path != local_path:
                    try:
                        shutil.copy2(str(rh_path), str(local_path))
                        print(f"[RH_Agent] 已自动同步本地副本: {local_path}")
                    except Exception as copy_err:
                        print(f"[RH_Agent] 自动同步本地副本失败（不影响运行）: {copy_err}")
                return data
            except Exception as e:
                print(f"[RH_Agent] 读取 {path} 失败: {e}")
    print("[RH_Agent] 警告：未找到 models_registry.json，模型列表为空")
    return []


def _build_model_lookup(registry: List[Dict]) -> Dict[str, Dict]:
    return {entry["display_name"]: entry for entry in registry if entry.get("display_name")}


def _generate_web_params(registry: List[Dict]):
    """
    根据 models_registry.json 生成 web/models_params.json，供前端 JS 读取。

    每个模型的数据结构：
    {
        "imageCount": int,           # IMAGE 类型参数数量（决定 inputcount 默认值）
        "params": {
            "fieldKey": {
                "type": "LIST"|"STRING"|"INT"|"FLOAT"|"BOOLEAN",
                "defaultValue": "...",  # 可选
                "options": ["..."],     # LIST 类型时提供
                "required": true        # 可选
            }
        }
    }
    """
    web_dir = Path(__file__).resolve().parent / "web"
    web_dir.mkdir(exist_ok=True)

    params_data = {}
    for entry in registry:
        name = entry.get("display_name")
        if not name:
            continue

        raw_params = entry.get("params", [])
        image_count = sum(1 for p in raw_params if p.get("type", "").upper() == "IMAGE")

        model_params = {}
        for p in raw_params:
            fk = p.get("fieldKey", "")
            pt = p.get("type", "").upper()

            if pt == "IMAGE":
                continue  # 图片通过 images_N 端口处理
            if _is_prompt_field(fk, pt):
                continue  # 提示词通过 prompts_json 处理

            param_info: Dict[str, Any] = {"type": pt}
            dv = p.get("defaultValue")
            if dv is not None:
                param_info["defaultValue"] = dv
            options = p.get("options", [])
            if options:
                param_info["options"] = [
                    str(o.get("value", "")) for o in options if o.get("value") is not None
                ]
            if p.get("required"):
                param_info["required"] = True

            model_params[fk] = param_info

        params_data[name] = {
            "imageCount": image_count,
            "endpoint": entry.get("endpoint", ""),
            "params": model_params,
        }

    out_path = web_dir / "models_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params_data, f, ensure_ascii=False)
    print(f"[RH_Agent] 生成 web/models_params.json（{len(params_data)} 个模型）")


# 模块级缓存
_REGISTRY: List[Dict] = []
_MODEL_LOOKUP: Dict[str, Dict] = {}
_MODEL_NAMES: List[str] = []


def _ensure_registry():
    global _REGISTRY, _MODEL_LOOKUP, _MODEL_NAMES
    if not _REGISTRY:
        _REGISTRY = _load_models_registry()
        _MODEL_LOOKUP = _build_model_lookup(_REGISTRY)
        _MODEL_NAMES = list(_MODEL_LOOKUP.keys()) or ["（未找到模型注册表）"]
        if _REGISTRY:
            try:
                _generate_web_params(_REGISTRY)
            except Exception as e:
                print(f"[RH_Agent] 生成 models_params.json 失败: {e}")


# ==================== 参数推断工具 ====================

def _find_prompt_field(params: List[Dict]) -> Optional[str]:
    candidates = []
    for p in params:
        fk = p.get("fieldKey", "")
        pt = p.get("type", "").upper()
        if pt == "STRING":
            candidates.append((p["fieldKey"], fk.lower()))
    for fieldKey, fk_lower in candidates:
        if any(kw in fk_lower for kw in _PROMPT_KEYWORDS):
            return fieldKey
    return candidates[0][0] if candidates else None


def _find_image_field_defs(params: List[Dict]) -> List[Dict]:
    """返回所有 IMAGE 类型参数的完整定义（含 multipleInputs 标志）。"""
    return [p for p in params if p.get("type", "").upper() == "IMAGE"]


def _assign_image_urls(payload: Dict, image_field_defs: List[Dict], image_urls: List[str]):
    """
    将上传后的图片 URL 填入 payload。

    规则：
    - 只有一个 IMAGE 字段且 multipleInputs=true → 所有 URL 作为列表传入
    - 只有一个 IMAGE 字段且 multipleInputs=false → 只传第一张 URL（字符串）
    - 多个 IMAGE 字段 → 逐一对应（各字段视自身 multipleInputs 决定列表或字符串）
    """
    if not image_urls or not image_field_defs:
        return

    if len(image_field_defs) == 1:
        field = image_field_defs[0]
        fk = field.get("fieldKey", "")
        if field.get("multipleInputs", False):
            payload[fk] = image_urls          # 列表
        else:
            payload[fk] = image_urls[0]       # 字符串
    else:
        for field, url in zip(image_field_defs, image_urls):
            fk = field.get("fieldKey", "")
            if field.get("multipleInputs", False):
                payload[fk] = [url]
            else:
                payload[fk] = url


def _build_payload(
    model_entry: Dict,
    prompt_text: str,
    prompt_obj: Dict,
    image_urls: List[str],
    model_params: Dict,
) -> Dict:
    """
    构建单次 API 请求 payload。

    优先级（从低到高）：
      1. 模型参数 defaultValue
      2. prompt_obj 中与 fieldKey 匹配的字段
      3. prompt_text → 推断的 prompt 字段
      4. image_urls → IMAGE 类型字段（自动处理单图/多图）
      5. model_params（用户指定，最终覆盖）
    """
    params = model_entry.get("params", [])
    payload = {}

    # 1. 模型默认值
    for p in params:
        fk = p.get("fieldKey")
        dv = p.get("defaultValue")
        if fk and dv is not None:
            payload[fk] = dv

    # 2. prompt_obj 中的匹配字段
    fk_lower_map = {p.get("fieldKey", "").lower(): p.get("fieldKey") for p in params}
    for obj_key, obj_val in prompt_obj.items():
        matched_fk = fk_lower_map.get(obj_key.lower())
        if matched_fk and isinstance(obj_val, (str, int, float, bool)):
            payload[matched_fk] = str(obj_val) if isinstance(obj_val, (int, float)) else obj_val

    # 3. prompt_text
    prompt_field = _find_prompt_field(params)
    if prompt_field and prompt_text:
        payload[prompt_field] = prompt_text

    # 4. 图片 URL（自动判断单图/多图字段）
    image_field_defs = _find_image_field_defs(params)
    _assign_image_urls(payload, image_field_defs, image_urls)

    # 5. model_params 最终覆盖
    for k, v in model_params.items():
        payload[k] = v

    return payload


# ==================== 主节点 ====================

class RHAgentNode:
    """RunningHub 批量生图 Agent - 并发调用 RH OpenAPI，支持 VLM 智能验证和多图输入"""

    def __init__(self):
        _ensure_registry()

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_registry()
        model_list = _MODEL_NAMES if _MODEL_NAMES else ["（请先安装 ComfyUI_RH_OpenAPI 插件）"]

        return {
            "required": {
                "model": (model_list, {
                    "default": "RH 全能图片PRO-图生图" if "RH 全能图片PRO-图生图" in model_list else model_list[0]
                }),

                "prompts_json": ("STRING", {
                    "multiline": True,
                    "default": '{"prompts": [{"prompt": "a beautiful landscape, golden hour"}, {"prompt": "a futuristic city at night"}]}'
                }),

                # 切换模型时由 JS 自动填入该模型的默认参数，用户可手动修改
                "model_params": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "模型参数 JSON（切换模型时自动刷新为该模型的默认值，可手动修改）"
                }),

                "max_concurrent": ("INT", {
                    "default": DEFAULT_MAX_CONCURRENT,
                    "min": 1,
                    "max": 200,
                    "tooltip": "最大并发任务数"
                }),

                "vlm_model": (AVAILABLE_VLM_MODELS, {
                    "default": DEFAULT_VLM_MODEL,
                    "tooltip": "VLM 智能验证使用的模型（需要 SiliconFlow API Key）"
                }),

                "enable_validation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用 VLM 智能验证（需要填写 vlm_api_key 或在 config/.env 配置 VLM_API_KEY）"
                }),

                "enable_prompt_optimization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用提示词优化（需要 vlm_api_key，在 VLM 验证前对每条 prompt 进行专业化优化）"
                }),

                "inputcount": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "显示的图片端口数量（切换模型时自动同步）。端口均为可选：不连接任何图片时自动切换为纯文生图模式。"
                }),
            },
            "optional": {
                "rh_settings": ("RH_OPENAPI_CONFIG", {
                    "tooltip": "连接 RH OpenAPI Settings 节点以使用其 API Key"
                }),
                "rh_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "RunningHub API Key（可选，优先于 config/.env）"
                }),
                "vlm_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "SiliconFlow API Key（可选，优先于 config/.env 中的 VLM_API_KEY）"
                }),
                # 静态声明 0-9，更多端口由 JS 动态添加
                "images_0": ("IMAGE",),
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "images_3": ("IMAGE",),
                "images_4": ("IMAGE",),
                "images_5": ("IMAGE",),
                "images_6": ("IMAGE",),
                "images_7": ("IMAGE",),
                "images_8": ("IMAGE",),
                "images_9": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "generation_log")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "RunningHub"

    def generate(
        self,
        model: str,
        prompts_json: str,
        model_params: str,
        max_concurrent: int,
        vlm_model: str,
        enable_validation: bool,
        enable_prompt_optimization: bool,
        inputcount: int,
        rh_settings=None,
        rh_api_key: str = "",
        vlm_api_key: str = "",
        **kwargs,
    ) -> Tuple[torch.Tensor, str]:

        print("\n" + "=" * 60)
        print("RunningHub 批量生图 Agent - 开始执行")
        print("=" * 60)

        generation_log = []

        # ── 步骤 1：解析 API 配置 ──────────────────────────────
        print("\n[步骤 1/6] 解析 API 配置...")
        rh_config = get_rh_config(rh_settings, rh_api_key)
        api_key = rh_config["api_key"]
        base_url = rh_config["base_url"]
        upload_timeout = rh_config.get("upload_timeout", 60)
        print(f"[OK] base_url={base_url}")

        # ── 步骤 2：查找模型定义 ──────────────────────────────
        print(f"\n[步骤 2/6] 查找模型: {model}")
        _ensure_registry()
        model_entry = _MODEL_LOOKUP.get(model)
        if not model_entry:
            raise ValueError(f"未找到模型 '{model}'，请检查 models_registry.json 是否存在")

        endpoint = model_entry.get("endpoint", "")
        if not endpoint:
            raise ValueError(f"模型 '{model}' 缺少 endpoint 定义")
        print(f"[OK] endpoint={endpoint}")

        # ── 步骤 3：解析 model_params ─────────────────────────
        try:
            model_params_dict = json.loads(model_params.strip() or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"model_params JSON 解析失败: {e}")

        # ── 收集输入图片 tensor（最早收集，VLM验证和优化都需要）─
        input_tensors = []
        for i in range(inputcount):
            img = kwargs.get(f"images_{i}")
            if img is not None:
                if img.dim() == 4:
                    img = img[0]
                input_tensors.append(img)

        # ── 步骤 4：VLM 验证 / 解析 prompts_json ─────────────
        print(f"\n[步骤 3/6] 解析 prompts_json...")
        parsed_prompts, validation_used_vlm = self._parse_prompts(
            prompts_json, enable_validation, vlm_model, vlm_api_key,
            image_tensors=input_tensors or None,
        )
        if not parsed_prompts:
            raise ValueError("提示词列表为空")
        print(f"[OK] 共 {len(parsed_prompts)} 个任务")

        # 将 VLM 验证解析后的提示词写入 log
        if enable_validation and validation_used_vlm:
            generation_log.append("=== VLM 验证解析结果 ===")
            for i, obj in enumerate(parsed_prompts):
                task_label = (
                    obj.get("pose_name_cn")
                    or obj.get("scene_name_cn")
                    or obj.get("name_cn")
                    or f"任务{i + 1}"
                )
                generation_log.append(f"[{i+1}/{len(parsed_prompts)}] {task_label}")
                generation_log.append(f"  提示词: {obj.get('prompt', '')}")
            generation_log.append("")

        # ── 提示词优化（可选，在任务提交前逐条优化）─────────
        vlm_key = get_vlm_api_key(vlm_api_key) if enable_prompt_optimization else ""
        if enable_prompt_optimization:
            if vlm_key:
                ref_info = f"，参考图 {len(input_tensors)} 张" if input_tensors else ""
                print(f"\n[提示词优化] 开始优化 {len(parsed_prompts)} 条提示词{ref_info}...")
                original_prompts = [obj.get("prompt", "") for obj in parsed_prompts]
                optimizer = VLMValidator(vlm_key, vlm_model)
                parsed_prompts = optimizer.optimize_prompts(parsed_prompts, input_tensors or None)
                print(f"[提示词优化] 优化完成")
                # 记录优化前后对比到 generation_log
                generation_log.append("=== 提示词优化结果 ===")
                for i, (orig, obj) in enumerate(zip(original_prompts, parsed_prompts)):
                    optimized = obj.get("prompt", "")
                    task_label = (
                        obj.get("pose_name_cn")
                        or obj.get("scene_name_cn")
                        or obj.get("name_cn")
                        or f"任务{i + 1}"
                    )
                    generation_log.append(f"[{i+1}/{len(parsed_prompts)}] {task_label}")
                    generation_log.append(f"  原始: {orig}")
                    generation_log.append(f"  优化: {optimized}")
                generation_log.append("")
            else:
                print(f"\n[提示词优化] ⚠️ enable_prompt_optimization=True 但未找到 VLM_API_KEY，跳过优化")

        # ── 步骤 5：上传图片到 RH OSS ─────────────────────────
        input_image_urls = []

        if input_tensors:
            print(f"\n[步骤 4/6] 上传 {len(input_tensors)} 张图片（inputcount={inputcount}，已连接 {len(input_tensors)} 张）...")
            try:
                input_image_urls = upload_tensors(
                    input_tensors, api_key, base_url, upload_timeout
                )
                print(f"[OK] 上传完成，共 {len(input_image_urls)} 张")
            except Exception as e:
                raise RuntimeError(f"图片上传失败: {e}")
        else:
            if inputcount > 0:
                print(f"\n[步骤 4/6] 图片端口已显示 {inputcount} 个，但均未连接 → 纯文生图模式")
            else:
                print(f"\n[步骤 4/6] 跳过图片上传（inputcount=0，纯文生图模式）")

        # ── 步骤 6：并发提交任务 ─────────────────────────────
        print(f"\n[步骤 5/6] 并发提交（最大并发={max_concurrent}）...")
        client = RHClient(api_key, base_url, rh_config)
        manager = TaskManager(max_concurrent, max(max_concurrent, DEFAULT_MAX_QUEUE_SIZE))

        tasks = []
        for i, prompt_obj in enumerate(parsed_prompts):
            prompt_text = prompt_obj.get("prompt", "")
            task_name = (
                prompt_obj.get("pose_name_cn")
                or prompt_obj.get("scene_name_cn")
                or prompt_obj.get("name_cn")
                or f"任务{i + 1}"
            )
            payload = _build_payload(
                model_entry, prompt_text, prompt_obj, input_image_urls, model_params_dict
            )
            tasks.append({
                "task_id": i,
                "name": task_name,
                "task_name": task_name,
                "endpoint": endpoint,
                "payload": payload,
            })

        results = manager.submit_batch(
            tasks,
            lambda task: self._run_single_task(client, task)
        )

        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        print(f"\n[进度] 成功 {len(successful)}/{len(results)}")

        generation_log.append("=== 生成结果 ===")
        for f in failed:
            msg = f"FAIL {f.get('task_name', '?')}: {f.get('error', '未知错误')}"
            print(f"  {msg}")
            generation_log.append(msg)

        # ── 步骤 7：下载并合并结果 ───────────────────────────
        print(f"\n[步骤 6/6] 下载生成结果...")
        generated_tensors = []
        for i, r in enumerate(successful):
            urls = r.get("result_urls", [])
            if not urls:
                generation_log.append(f"SKIP {r.get('task_name', f'任务{i+1}')}: 无结果 URL")
                continue
            for url in urls:
                try:
                    tensor = url_to_tensor(url)
                    generated_tensors.append(tensor)
                    generation_log.append(f"OK {r.get('task_name', f'任务{i+1}')}")
                    print(f"  [OK] {r.get('task_name')} 下载成功")
                    break
                except Exception as e:
                    err = f"DOWNLOAD_FAIL {r.get('task_name', f'任务{i+1}')}: {e}"
                    print(f"  [FAIL] {err}")
                    generation_log.append(err)

        if not generated_tensors:
            raise RuntimeError("所有任务均失败，未能生成任何图片")

        output_batch = tensors_to_batch(generated_tensors)
        print(f"\n[OK] 完成！{len(generated_tensors)} 张图片，shape={output_batch.shape}")
        print("=" * 60)

        return (output_batch, "\n".join(generation_log))

    def _parse_prompts(
        self,
        prompts_json: str,
        enable_validation: bool,
        vlm_model: str,
        vlm_api_key_input: str,
        image_tensors: Optional[List] = None,
    ) -> Tuple[List[Dict], bool]:
        """
        返回 (prompts, vlm_used)。
        vlm_used=True 表示本次解析实际调用了 VLM（而非本地 fallback）。
        """
        # ── 基础检查：prompts_json 不能为空 ──────────────────
        cleaned = (prompts_json or "").strip()
        if not cleaned:
            raise ValueError(
                "prompts_json 为空，请填写提示词 JSON。\n"
                "示例：{\"prompts\": [{\"prompt\": \"a beautiful landscape\"}]}"
            )

        vlm_key = get_vlm_api_key(vlm_api_key_input) if enable_validation else ""

        if enable_validation and vlm_key:
            print("  使用 VLM 验证模式...")
            try:
                validator = VLMValidator(vlm_key, vlm_model)
                # 将所有参考图转为 base64 传给验证器
                ref_images_b64 = []
                for t in (image_tensors or []):
                    b64 = validator._tensor_to_base64(t)
                    if b64:
                        ref_images_b64.append(b64)
                if ref_images_b64:
                    print(f"  [OK] {len(ref_images_b64)} 张参考图已传入 VLM 验证器")
                result = validator.validate_json(cleaned, ref_images_b64=ref_images_b64 or None)
                if result.get("valid"):
                    prompts = result.get("parsed_data", {}).get("prompts", [])
                    print(f"  [OK] VLM 验证通过，{len(prompts)} 个提示词")
                    return prompts, True
                else:
                    print(f"  [WARN] VLM 验证失败: {result.get('error')}，降级到本地解析")
            except Exception as e:
                print(f"  [WARN] VLM 调用异常: {e}，降级到本地解析")
        elif enable_validation and not vlm_key:
            print("  [WARN] enable_validation=True 但未找到 VLM_API_KEY，降级到本地解析")

        # ── 本地解析 ─────────────────────────────────────────
        print("  使用本地解析模式...")

        # 自然语言输入（不以 { 或 [ 开头）无法本地解析
        if not cleaned.startswith(('{', '[')):
            raise ValueError(
                "检测到自然语言输入，但 VLM 功能未启用或 VLM_API_KEY 未配置。\n"
                "自然语言转提示词需要有效的 VLM_API_KEY。\n"
                "解决方案：\n"
                "  1. 在 config/.env 中填写 VLM_API_KEY（SiliconFlow API Key）\n"
                "  2. 或改用 JSON 格式，例如：\n"
                "     {\"prompts\": [{\"prompt\": \"side view, white background\"}]}"
            )

        validator = VLMValidator("dummy", vlm_model)
        result = validator.validate_json(cleaned)
        if result.get("valid"):
            return result.get("parsed_data", {}).get("prompts", []), False
        raise ValueError(
            f"prompts_json 解析失败：{result.get('error', '未知错误')}\n"
            f"{result.get('suggestion', '')}"
        )

    def _run_single_task(self, client: RHClient, task: Dict) -> Dict:
        return client.generate(
            endpoint=task["endpoint"],
            payload=task["payload"],
            task_name=task.get("task_name", f"任务{task.get('task_id', 0) + 1}"),
        )


# ==================== 节点注册 ====================

NODE_CLASS_MAPPINGS = {
    "RHAgentNode": RHAgentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RHAgentNode": "RH 批量生图 Agent",
}

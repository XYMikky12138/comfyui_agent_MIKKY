"""
VLM 调控中心 - 智能验证 JSON 格式和提示词质量
支持自动识别并转换多种 JSON 格式
基于 SiliconFlow API（https://api.siliconflow.cn），使用 OpenAI 兼容接口
VLM API Key 通过 core/api_key.get_vlm_api_key() 解析，优先级：节点输入 > 环境变量 > config/.env
"""

import json
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List
from openai import OpenAI

# 支持相对导入和绝对导入
try:
    from .config import (
        SILICONFLOW_BASE_URL,
        VLM_TIMEOUT,
        VLM_MAX_TOKENS,
        VLM_TEMPERATURE,
        DEFAULT_VLM_MODEL,
        VLM_MAX_RETRIES
    )
except ImportError:
    from config import (
        SILICONFLOW_BASE_URL,
        VLM_TIMEOUT,
        VLM_MAX_TOKENS,
        VLM_TEMPERATURE,
        DEFAULT_VLM_MODEL,
        VLM_MAX_RETRIES
    )


# VLM 系统提示词（JSON 输入模式）
VLM_SYSTEM_PROMPT = """你是一个专业的 JSON 验证和智能转换助手，专门处理图像生成任务的提示词数据。

## 核心任务

1. **验证 JSON 格式**：确保输入的 JSON 格式正确
2. **智能识别结构**：自动识别任何包含提示词的数组结构，无论字段名是什么
3. **智能转换**：将复杂的描述转换为适合 AI 图像生成的英文提示词

## 智能识别机制

**重要**: 不要硬编码字段名！系统会自动扫描 JSON 中的所有数组，找到包含 `prompt` 字段或可转换字段的数组。

支持的字段名示例（但不限于）:
- `prompts`, `scenes`, `storyboard`, `tasks`, `items`, `list`, `data`
- 或任何其他包含提示词数组的字段名

## 支持的 JSON 格式

### 格式 1: 简单格式（直接使用）
```json
{
  "prompts": [
    {
      "prompt": "Front standing pose, white background"
    }
  ]
}
```

**可选字段**：
- `pose_name_cn`: 中文名称（可选）
- `pose_name_en`: 英文名称（可选）

完整示例：
```json
{
  "prompts": [
    {
      "pose_name_cn": "正面站立",
      "pose_name_en": "Front Standing",
      "prompt": "Front standing pose, white background"
    }
  ]
}
```

### 格式 2: 详细格式（需要转换）
```json
{
  "total_poses": 8,
  "model_gender": "female",
  "poses": [
    {
      "priority": 1,
      "name_cn": "正面全身站立",
      "name_en": "Front Full-Body Standing",
      "description": "自然直立，双脚与肩同宽...",
      "focus_points": ["整体比例", "正面效果"],
      "camera_angle": "正面平视"
    }
  ]
}
```

### 格式 3: 视频分镜格式（自动提取）
```json
{
  "video_info": {
    "product_type": "Product Name",
    "platform": "TikTok"
  },
  "storyboard": [
    {
      "scene_number": 1,
      "scene_name_cn": "开场镜头",
      "scene_name_en": "Opening Shot",
      "prompt": "Close-up shot of product, white background, soft lighting",
      "duration": "2 seconds",
      "shot_type": "Close-up"
    }
  ]
}
```

**格式 3 说明**：
- 系统会自动从 `storyboard` 数组中提取每个场景的 `prompt` 字段
- `scene_name_cn` 和 `scene_name_en` 会作为提示词名称
- `scene_number` 会作为优先级
- 其他字段（duration, shot_type, focus_point, text_overlay, transition）会被保留但不影响生成

## 转换规则

### 对于格式 2（姿势描述格式）

你需要将每个 pose 转换为标准的 prompt 格式：

1. **提取关键信息**：
   - name_cn → pose_name_cn（可选）
   - name_en → pose_name_en（可选）
   - description + focus_points + camera_angle → prompt（必需）

2. **生成英文 prompt**：
   - 结合 description（姿势描述）、focus_points（关注点）、camera_angle（机位角度）
   - 转换为简洁的英文提示词（50-200 字符）
   - 格式：[姿势动作], [关注点], [机位], [背景和光线]
   - 示例：`Transform to front full-body standing pose, natural straight posture, feet shoulder-width apart, arms naturally down, facing camera, white background, soft lighting, 3:4 portrait`

3. **保留优先级**：按 priority 排序

### 对于格式 3（视频分镜格式）

直接提取已有的 prompt 字段，无需转换：

1. **提取关键信息**：
   - scene_name_cn → pose_name_cn
   - scene_name_en → pose_name_en
   - prompt → prompt（直接使用，已经是英文提示词）
   - scene_number → priority

2. **无需生成**：格式 3 的 prompt 字段已经是完整的英文提示词，直接使用即可

3. **保留场景信息**：duration, shot_type, focus_point, text_overlay, transition 等字段会被保留

## 返回格式

**必须严格返回以下 JSON 格式（不要添加任何其他文字或 markdown 标记）：**

```json
{
  "valid": true,
  "parsed_data": {
    "total_prompts": 8,
    "model_gender": "female",
    "model_body_type": "standard",
    "prompts": [
      {
        "prompt": "完整的英文提示词",
        "pose_name_cn": "中文名称（可选）",
        "pose_name_en": "English Name（可选）",
        "priority": 1
      }
    ]
  }
}
```

如果验证失败：
```json
{
  "valid": false,
  "error": "错误信息",
  "suggestion": "修复建议"
}
```

## 重要提示

- 返回的 JSON 必须可以直接被 Python json.loads() 解析
- 不要使用 markdown 代码块标记（```json）
- 不要添加任何注释或额外文字
- 确保所有字符串正确转义
- **prompt 字段为必需，pose_name_cn 和 pose_name_en 为可选**
"""

# 提示词优化器系统提示词（基于 Gemini 图片提示词优化助手）
PROMPT_OPTIMIZER_SYSTEM_PROMPT = """你是一位专业的 Gemini 图片生成/编辑提示词优化师，专为 Nano Banana 系列模型（Gemini Flash Image）优化提示词。

## 你的核心任务

将用户输入的模糊、口语化或 AI 难以理解的图片指令，转化为结构清晰、表意精确、符合 Gemini 图片模型理解习惯的英文提示词。

**铁律：不得改变用户原始指令的核心要求。只优化表达方式，不替换用户意图。**

## 工作流程

### 第一步：理解原始意图

仔细分析用户输入，提取以下信息：
- **核心要求**：用户最想实现什么？（这是不可更改的）
- **任务类型**：文本生图 / 图片编辑 / 风格转换 / 背景替换 / 其他
- **模糊点**：哪些描述 AI 可能误解或无法执行？

### 第二步：识别模糊问题

常见模糊问题清单：
| 问题类型 | 举例 | 优化方向 |
|---------|------|---------|
| 主观感受词 | "好看一点"、"更自然" | 转化为具体视觉描述 |
| 中文口语指令 | "把背景搞掉" | 转化为标准英文动作词 |
| 缺少保留说明 | "换个背景" | 补充"keep the subject unchanged" |
| 方向/位置模糊 | "放左边一点" | 量化或具体化位置 |
| 风格描述不足 | "弄得专业点" | 补充具体风格关键词 |
| 缺少技术参数 | 未指定比例/质量 | 根据用途推断并补充 |

### 第三步：按结构重写提示词

根据任务类型选择对应结构：

**文本生图结构：**
[主体描述] + [动作/状态] + [环境/背景] + [光线/氛围] + [风格] + [技术参数]

**图片编辑结构：**
[编辑动作] + [编辑对象] + [目标效果] + [保留内容（重要！）]

**风格转换结构：**
Convert/Transform this [原内容] into [目标风格], maintaining [保留元素], with [具体风格特征].

**背景替换结构：**
Replace the background with [新背景]. Keep [主体] exactly unchanged, including all details. Ensure natural edge blending.

### 第四步：输出格式

直接输出优化后的英文提示词，不附加任何解释、标签、说明或确认文字。

## 优化规则详解

### 规则一：核心要求绝对不变

用户说"把背景换成森林"，核心要求是"森林背景"。
- 允许：将"森林"细化为"dense green forest with dappled sunlight"
- 禁止：擅自改为"草地"、"公园"或任何用户未指定的场景

### 规则二：口语转专业术语

| 用户口语 | 优化为 |
|---------|--------|
| "背景搞掉" / "去掉背景" | "Remove the background, replace with pure white background" |
| "人不要动" / "不要改人" | "Keep the subject/person exactly unchanged, including all details of clothing, face, and posture" |
| "亮一点" | "Increase overall brightness, add soft even lighting" |
| "高级感" | "luxury aesthetic, professional commercial photography style, premium quality" |
| "清楚一点" | "sharp details, crisp edges, high resolution, no blur" |
| "颜色鲜艳" | "vibrant saturated colors, rich color palette" |
| "朦胧感" | "soft dreamy atmosphere, shallow depth of field, slight bokeh effect" |
| "复古感" | "vintage aesthetic, film grain texture, muted warm tones, retro color grading" |

### 规则三：必须补充保留说明

图片编辑类指令，若用户未说明"保留什么"，必须根据常识推断并补充：
- 换背景 → 补充：Keep the [subject] exactly unchanged
- 调光线 → 补充：maintain the subject's composition and position
- 改风格 → 补充：preserve the original subject, composition, and key elements
- 局部修改 → 补充：do not change any other parts of the image

### 规则四：补充质量控制词（除非用户明确说不要）

根据使用场景自动判断并补充：
- 电商/产品图 → professional product photography, high resolution, commercial quality
- 社交媒体 → social media ready, visually engaging
- 印刷品 → print-ready quality, high resolution, sharp details
- 快速预览 → clean clear result

### 规则五：复杂指令加思考前缀

若用户指令涉及多个要素同时修改，在提示词开头加：
- 中等复杂（2-3个修改点）：Think carefully about this request before generating:
- 高度复杂（4个以上修改点/精确构图要求）：Use high-detail reasoning to ensure accuracy:

### 规则六：负向约束

若原指令包含"不要"、"别"、"不能"等否定词，转化为英文负向指令并置于提示词末尾：
- "不要有文字" → no text overlays, no watermarks
- "脸不要变形" → accurate facial features, no face distortion
- "不要改衣服" → do not alter the clothing in any way
- "不要模糊" → no blurry areas, sharp throughout

## 重要提示

- 直接输出优化后的英文提示词文本，不要添加任何解释、标签、说明文字
- 不要使用 markdown 代码块标记
- 不要在开头或结尾添加任何额外内容
- 输出内容只有优化后的英文提示词本身

## 参考图使用规则（当用户提供参考图时）

若消息中包含参考图片，必须：
1. **优先分析参考图**：提取主体的性别、年龄、肤色、体型、发型、服装颜色/款式等视觉特征
2. **以参考图为准**：优化后的提示词必须如实描述参考图中主体的特征，不得与图片内容矛盾
3. **补充特征描述**：将从图中提取的特征自然融入优化后的提示词（如 male, white t-shirt, dark jogger pants 等）

## 性别保留规则（铁律）

- 参考图中可见性别 → 以参考图为准，不得更改
- 原始提示词中有明确性别词（female/male/woman/man/女/男）且与参考图一致 → 保留
- 原始提示词中**没有**任何性别词，且无参考图 → **严禁**擅自添加 female、male、woman、man 等任何性别词，使用中性词 person/model
- 场景（健身房/室外/街道）、服装、动作等不构成推断性别的依据
"""

# VLM 自然语言转换系统提示词
VLM_NATURAL_LANGUAGE_PROMPT = """你是一个专业的图像生成提示词助手。

## 任务

用户会用中文或英文描述他们想生成的图片内容，可能包含：
- 数量信息（"生成3张"、"三个视角"）
- 内容描述（"侧面视角"、"背面视角"、"穿红色连衣裙"）
- 风格要求（"写实风格"、"卡通风格"）

你需要将用户的自然语言描述**智能拆分**成对应数量的图像生成提示词列表。

## 拆分规则

1. **识别数量**：从描述中提取要生成的图片数量
   - "生成两张" → 2 个 prompt
   - "三个视角" → 3 个 prompt
   - "侧面和背面" → 2 个 prompt（侧面 + 背面）
   - 未指定数量时 → 根据内容合理推断（通常为 1 个）

2. **拆分维度**：
   - 视角变化（正面/侧面/背面/俯视/仰视）→ 每个视角一个 prompt
   - 动作变化（站立/坐姿/行走）→ 每个动作一个 prompt
   - 场景变化（室内/室外/不同背景）→ 每个场景一个 prompt
   - 服装变化（不同服装）→ 每套服装一个 prompt

3. **补充共同描述**：用户提到的整体要求（性别、风格、背景等）要加入每个 prompt

4. **prompt 语言**：所有 prompt 必须使用**英文**，简洁精准（50~200字符为宜）

5. **性别处理（重要）**：
   - 用户明确说明性别（"男性"、"女性"、"male"、"female"）→ 使用对应性别词
   - 用户**未提及性别** → 使用中性词 `person` 或 `model`，**严禁**擅自添加 `female`、`male`、`woman`、`man` 等性别词
   - 场景词（健身房、室外、街道等）不影响性别推断，不得因为场景而假设性别

## 返回格式

**只返回以下 JSON，不要添加任何其他文字或 markdown 标记：**

{
  "valid": true,
  "parsed_data": {
    "total_prompts": <数量>,
    "prompts": [
      {
        "prompt": "英文提示词",
        "pose_name_cn": "中文名称（可选）",
        "priority": 1
      }
    ]
  }
}

## 示例

用户输入：生成两张图片，换成侧面视角和背面视角
（未指定性别 → 使用中性词 person）

返回：
{
  "valid": true,
  "parsed_data": {
    "total_prompts": 2,
    "prompts": [
      {"prompt": "side view, full body portrait, person standing, white background, soft lighting", "pose_name_cn": "侧面视角", "priority": 1},
      {"prompt": "back view, full body portrait, person standing, white background, soft lighting", "pose_name_cn": "背面视角", "priority": 2}
    ]
  }
}

用户输入：生成一张红色连衣裙女性站立正面图

返回：
{
  "valid": true,
  "parsed_data": {
    "total_prompts": 1,
    "prompts": [
      {"prompt": "female model wearing red dress, standing pose, front view, full body, white background, soft lighting", "pose_name_cn": "红色连衣裙正面站立", "priority": 1}
    ]
  }
}
"""


class VLMValidator:
    """VLM 调控中心 - 验证输入格式和提示词质量"""
    
    def __init__(self, api_key: str, model: str = DEFAULT_VLM_MODEL):
        """
        初始化 VLM 验证器

        Args:
            api_key: SiliconFlow VLM API Key（通过 core/api_key.get_vlm_api_key() 获取，
                     传空字符串或 "dummy" 则进入仅本地模式）
            model: VLM 模型名称
        """
        if not api_key or api_key == "dummy":
            self.client = None
            self.api_key = None
            self.model = model
            print(f"[VLM] 初始化（仅本地模式）")
        else:
            self.api_key = api_key
            self.model = model
            self.client = OpenAI(
                api_key=api_key,
                base_url=SILICONFLOW_BASE_URL
            )
            print(f"[VLM] 初始化成功，模型: {model}")
    
    def _auto_fix_common_errors(self, json_string: str) -> str:
        """
        自动修复JSON中常见的语法错误
        
        Args:
            json_string: 原始JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        # 修复1: 移除无效的单引号转义 \' -> '
        # JSON标准中，双引号内的单引号不需要转义
        if "\\'" in json_string:
            print(f"[VLM] 🔧 自动修复: 移除无效的单引号转义符")
            json_string = json_string.replace("\\'", "'")
        
        # 修复2: 修复常见的Unicode转义错误
        # 例如: \u后面必须跟4个十六进制数字
        
        return json_string
    
    def _is_natural_language(self, text: str) -> bool:
        """判断输入是否为自然语言（而非 JSON 或格式错误的 JSON）。"""
        stripped = text.strip()
        # JSON 必然以 { 或 [ 开头；不以这两个字符开头的一定是自然语言
        return not stripped.startswith(('{', '['))

    def _convert_natural_language(self, text: str, ref_image_b64: Optional[str] = None) -> Dict:
        """
        使用 VLM 将自然语言描述转换为标准提示词列表。
        仅在 self.client 不为 None 时调用。

        Args:
            text: 用户自然语言描述
            ref_image_b64: 参考图片 base64（可选）；提供后模型可识别主体特征，避免性别误推断
        """
        print(f"[VLM] 检测到自然语言输入，调用 VLM 进行智能转换...")

        # 尝试顺序：1) 带图（若有）→ 2) 纯文本兜底（模型不支持视觉时自动降级）
        attempts_plan = []
        if ref_image_b64:
            attempts_plan.append(("带参考图", ref_image_b64))
        attempts_plan.append(("纯文本", None))

        for mode_label, current_image_b64 in attempts_plan:
            last_error = None
            for attempt in range(VLM_MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"[VLM] 重试第 {attempt + 1}/{VLM_MAX_RETRIES} 次（{mode_label}）...")
                    else:
                        print(f"[VLM] 调用模式: {mode_label}")

                    # 有参考图时使用多模态格式，模型可从图中识别主体特征
                    if current_image_b64:
                        user_content = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{current_image_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "这是用户上传的参考图片，请先观察图中主体的性别、外貌、服装等特征，"
                                    "在生成所有 prompt 时必须保留这些特征（尤其是性别），不得更改或假设。\n\n"
                                    f"用户的描述：{text}"
                                ),
                            },
                        ]
                    else:
                        user_content = text

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": VLM_NATURAL_LANGUAGE_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=VLM_TEMPERATURE,
                        max_tokens=VLM_MAX_TOKENS,
                        timeout=VLM_TIMEOUT,
                    )
                    raw = response.choices[0].message.content.strip()
                    print(f"[VLM] VLM 返回: {raw[:200]}...")

                    # 清理 markdown 包裹
                    if raw.startswith("```json"):
                        raw = raw[7:]
                    if raw.startswith("```"):
                        raw = raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()

                    result = json.loads(raw)
                    if result.get("valid", False):
                        cnt = result.get("parsed_data", {}).get("total_prompts", 0)
                        print(f"[VLM] [OK] 自然语言转换成功（{mode_label}），共 {cnt} 个提示词")
                    else:
                        print(f"[VLM] [FAIL] 转换失败: {result.get('error', '未知错误')}")
                    return result

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    print(f"[VLM] ❌ {mode_label} 转换失败 (第 {attempt + 1} 次): {err_str}")
                    # 检测到视觉不支持错误时，不再重试，直接切换到纯文本模式
                    vision_err_keywords = ("does not support", "image", "vision", "multimodal",
                                           "400", "unsupported")
                    if current_image_b64 and any(k in err_str.lower() for k in vision_err_keywords):
                        print(f"[VLM] ⚠️ 当前模型不支持视觉输入，自动降级为纯文本模式")
                        break
                    if attempt < VLM_MAX_RETRIES - 1:
                        continue

        return {
            "valid": False,
            "error": f"VLM 自然语言转换失败: {last_error}",
            "suggestion": "请改用 JSON 格式输入，或检查 VLM API Key 是否有效。",
        }

    def validate_json(self, json_string: str, ref_image_b64: Optional[str] = None) -> Dict:
        """
        验证/转换输入为标准提示词列表。

        支持三种输入形式：
          1. 标准 JSON（直接验证）
          2. 格式错误的 JSON（尝试修复）
          3. 自然语言描述（调用 VLM 转换为提示词列表）

        Args:
            json_string: 用户输入（JSON 或自然语言）
            ref_image_b64: 参考图片 base64（可选）；传给自然语言转换器，
                           使模型能从图中识别主体特征（性别、外貌等）

        Returns:
            验证结果字典:
            {
                "valid": bool,
                "parsed_data": dict (如果验证通过),
                "error": str (如果验证失败),
                "message": str
            }
        """
        print(f"[VLM] 开始验证输入...")

        # ── 自然语言输入：直接交给 VLM 转换 ──────────────────────
        if self._is_natural_language(json_string):
            if self.client:
                return self._convert_natural_language(json_string, ref_image_b64=ref_image_b64)
            else:
                return {
                    "valid": False,
                    "error": "检测到自然语言输入，但 VLM 客户端不可用（未配置 VLM_API_KEY）",
                    "suggestion": (
                        "请在 config/.env 中填写 VLM_API_KEY，或改用 JSON 格式输入。\n"
                        "示例：{\"prompts\": [{\"prompt\": \"side view, white background\"}]}"
                    ),
                }

        # ── JSON 输入：先修复再验证 ────────────────────────────────
        print(f"[VLM] 开始验证 JSON 输入...")
        
        # 预处理：自动修复常见的JSON语法错误
        original_json_string = json_string
        json_string = self._auto_fix_common_errors(json_string)
        
        # 预处理：尝试直接解析 JSON
        json_parse_error = None
        try:
            parsed_json = json.loads(json_string)
            if json_string != original_json_string:
                print(f"[VLM] ✅ JSON语法验证通过（已自动修复转义字符）")
            else:
                print(f"[VLM] JSON 语法验证通过")
        except json.JSONDecodeError as e:
            json_parse_error = e
            error_msg = f"JSON 语法错误: {str(e)}"
            print(f"[VLM] {error_msg}")
            
            # 如果有VLM客户端，尝试让VLM修复JSON
            if self.client:
                print(f"[VLM] 💡 尝试使用VLM智能修复JSON语法错误...")
                fixed_result = self._try_fix_json_with_vlm(original_json_string, error_msg)
                if fixed_result:
                    return fixed_result
            
            # 没有VLM或VLM修复失败，返回错误
            return {
                "valid": False,
                "error": error_msg,
                "suggestion": "请检查 JSON 格式，确保所有括号、引号正确闭合，转义字符正确（JSON中双引号内的单引号不需要转义）"
            }
        
        # 智能格式检查 - 自动识别任何包含 prompt 的数组
        task_field = None
        task_array = None
        
        # 扫描 JSON 中的所有字段，找到包含 prompt 的数组
        for field_name, field_value in parsed_json.items():
            # 跳过非数组字段
            if not isinstance(field_value, list):
                continue
            
            # 跳过空数组
            if len(field_value) == 0:
                continue
            
            # 检查数组的第一个元素
            first_item = field_value[0]
            if not isinstance(first_item, dict):
                continue
            
            # 检查是否包含 prompt 字段或可转换字段
            if "prompt" in first_item:
                # 找到包含 prompt 的数组（格式 1）
                task_field = field_name
                task_array = field_value
                print(f"[VLM] 智能识别: 找到包含 'prompt' 的数组字段 '{field_name}'")
                break
            elif "description" in first_item or "name_cn" in first_item:
                # 找到需要转换的姿势数组（格式 2）
                task_field = field_name
                task_array = field_value
                print(f"[VLM] 智能识别: 找到可转换的姿势数组字段 '{field_name}'")
                break
        
        # 如果没有找到任何可用的数组
        if task_field is None:
            return {
                "valid": False,
                "error": "未找到包含 'prompt' 字段的数组",
                "suggestion": "请确保 JSON 中包含一个数组，且数组中的每个对象都有 'prompt' 字段。示例:\n{\n  \"任意字段名\": [\n    {\"prompt\": \"提示词1\"},\n    {\"prompt\": \"提示词2\"}\n  ]\n}"
            }
        
        # 如果没有 VLM 客户端，直接使用本地验证
        if not self.client:
            print(f"[VLM] 使用本地验证模式...")
            return self._local_validate(parsed_json, task_field, task_array)
        
        # 调用 VLM 进行深度验证（带重试机制）
        last_error = None
        for attempt in range(VLM_MAX_RETRIES):
            try:
                if attempt > 0:
                    print(f"[VLM] 重试第 {attempt + 1}/{VLM_MAX_RETRIES} 次...")
                else:
                    print(f"[VLM] 调用 VLM 模型进行深度验证...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": VLM_SYSTEM_PROMPT},
                        {"role": "user", "content": f"请验证并转换以下 JSON 数据:\n\n{json_string}"}
                    ],
                    temperature=VLM_TEMPERATURE,
                    max_tokens=VLM_MAX_TOKENS,
                    timeout=VLM_TIMEOUT
                )
                
                # 提取 VLM 返回内容
                vlm_response = response.choices[0].message.content.strip()
                print(f"[VLM] VLM 返回: {vlm_response[:200]}...")
                
                # 清理可能的 markdown 标记
                if vlm_response.startswith("```json"):
                    vlm_response = vlm_response[7:]
                if vlm_response.startswith("```"):
                    vlm_response = vlm_response[3:]
                if vlm_response.endswith("```"):
                    vlm_response = vlm_response[:-3]
                vlm_response = vlm_response.strip()
                
                # 解析 VLM 返回的 JSON
                try:
                    vlm_result = json.loads(vlm_response)
                    
                    if vlm_result.get("valid", False):
                        print(f"[VLM] [OK] 验证通过，共 {vlm_result.get('parsed_data', {}).get('total_prompts', 0)} 个提示词")
                        return vlm_result
                    else:
                        print(f"[VLM] [FAIL] 验证失败: {vlm_result.get('error', '未知错误')}")
                        return vlm_result
                        
                except json.JSONDecodeError:
                    # VLM 返回的不是标准 JSON，降级到本地验证
                    print(f"[VLM] ⚠️ VLM 返回非标准 JSON，降级到本地验证...")
                    return self._local_validate(parsed_json, task_field, task_array)
            
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # 判断错误类型
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                    print(f"[VLM] ⏱️ 第 {attempt + 1} 次尝试超时 (当前超时限制: {VLM_TIMEOUT}秒)")
                    if attempt < VLM_MAX_RETRIES - 1:
                        print(f"[VLM] 准备重试...")
                        continue
                else:
                    print(f"[VLM] ❌ API调用失败: {error_msg}")
                    if attempt < VLM_MAX_RETRIES - 1:
                        print(f"[VLM] 准备重试...")
                        continue
        
        # 所有重试都失败，降级到本地验证
        print(f"[VLM] ❌ {VLM_MAX_RETRIES} 次尝试均失败: {str(last_error)}")
        print(f"[VLM] 💡 自动降级到本地验证模式（功能正常，但无VLM智能转换）")
        return self._local_validate(parsed_json, task_field, task_array)
    
    def _local_validate(self, parsed_json: Dict, task_field: str, task_array: List) -> Dict:
        """
        本地基础验证和转换（VLM 不可用时的降级方案）
        
        Args:
            parsed_json: 已解析的 JSON 对象
            task_field: 智能识别到的任务数组字段名
            task_array: 任务数组
            
        Returns:
            验证结果
        """
        # 检查第一个元素，判断是否需要转换
        first_item = task_array[0]
        
        # 情况 1: 已经包含 prompt 字段（直接使用）
        if "prompt" in first_item:
            print(f"[VLM] 格式类型: 简单格式（已包含 prompt）")
            return self._validate_simple_format(parsed_json, task_field, task_array)
        
        # 情况 2: 包含 description/name_cn 等字段（需要转换）
        elif "description" in first_item or "name_cn" in first_item:
            print(f"[VLM] 格式类型: 复杂格式（需要转换为 prompt）")
            return self._convert_complex_format(parsed_json, task_field, task_array)
        
        else:
            return {
                "valid": False,
                "error": f"数组 '{task_field}' 中的对象缺少必需字段",
                "suggestion": "每个对象必须包含 'prompt' 字段，或包含可转换的字段（如 'description', 'name_cn' 等）"
            }
    
    def _validate_simple_format(self, parsed_json: Dict, task_field: str, task_array: List) -> Dict:
        """
        验证简单格式（已包含 prompt 字段）
        
        Args:
            parsed_json: 完整的 JSON 对象
            task_field: 任务数组的字段名
            task_array: 任务数组
        """
        validated_prompts = []
        
        print(f"[VLM] 验证数组 '{task_field}'，共 {len(task_array)} 个任务...")
        
        # 验证每个提示词对象
        for i, prompt_obj in enumerate(task_array):
            if not isinstance(prompt_obj, dict):
                return {
                    "valid": False,
                    "error": f"任务 #{i+1} 不是对象类型",
                    "suggestion": "每个任务必须是 JSON 对象格式"
                }
            
            # 检查必需字段：只有 prompt 是必需的
            prompt = prompt_obj.get("prompt")
            
            if not prompt:
                return {
                    "valid": False,
                    "error": f"任务 #{i+1} 缺少 'prompt' 字段",
                    "suggestion": "请为每个任务添加 'prompt' 字段"
                }
            
            if not isinstance(prompt, str) or not prompt.strip():
                return {
                    "valid": False,
                    "error": f"任务 #{i+1} 的 'prompt' 为空",
                    "suggestion": "请提供有效的提示词内容"
                }
            
            # 智能提取名称字段（兼容多种命名）
            name_cn = (
                prompt_obj.get("pose_name_cn") or 
                prompt_obj.get("name_cn") or 
                prompt_obj.get("scene_name_cn") or 
                prompt_obj.get("title_cn") or 
                f"任务 {i+1}"
            )
            
            name_en = (
                prompt_obj.get("pose_name_en") or 
                prompt_obj.get("name_en") or 
                prompt_obj.get("scene_name_en") or 
                prompt_obj.get("title_en") or 
                f"Task {i+1}"
            )
            
            # 智能提取优先级/场景编号
            priority = (
                prompt_obj.get("priority") or 
                prompt_obj.get("scene_number") or 
                prompt_obj.get("order") or 
                i + 1
            )
            
            # 构建标准化的提示词对象
            validated_prompt = {
                "prompt": prompt,
                "pose_name_cn": name_cn,
                "pose_name_en": name_en,
                "priority": priority
            }
            
            # 保留所有其他字段
            for key, value in prompt_obj.items():
                if key not in ["prompt", "pose_name_cn", "pose_name_en", "name_cn", "name_en", "scene_name_cn", "scene_name_en", "title_cn", "title_en", "priority", "scene_number", "order"]:
                    validated_prompt[key] = value
            
            validated_prompts.append(validated_prompt)
            print(f"[VLM]   [OK] 任务 #{i+1}: {name_cn}")
        
        # 保留原始 JSON 中的其他顶层字段
        extra_data = {}
        for key, value in parsed_json.items():
            if key != task_field:
                extra_data[key] = value
        
        result = {
            "valid": True,
            "parsed_data": {
                "total_prompts": len(validated_prompts),
                "prompts": validated_prompts,
                "source_field": task_field,  # 记录源字段名
                **extra_data  # 保留其他字段
            },
            "message": f"本地验证通过（智能识别: 数组字段 '{task_field}'）"
        }
        
        print(f"[VLM] [OK] 验证完成，共 {len(validated_prompts)} 个提示词")
        
        return result
    
    def _convert_complex_format(self, parsed_json: Dict, task_field: str, task_array: List) -> Dict:
        """
        转换复杂格式为标准格式（通用转换器）
        
        适用于包含 description、name_cn 等描述性字段但缺少 prompt 的格式
        
        Args:
            parsed_json: 完整的 JSON 对象
            task_field: 任务数组的字段名
            task_array: 任务数组
        """
        converted_prompts = []
        
        print(f"[VLM] 开始转换复杂格式，共 {len(task_array)} 个任务...")
        
        for i, item in enumerate(task_array):
            if not isinstance(item, dict):
                return {
                    "valid": False,
                    "error": f"任务 #{i+1} 不是对象类型",
                    "suggestion": "每个任务必须是 JSON 对象格式"
                }
            
            # 智能提取名称
            name_cn = (
                item.get("name_cn") or 
                item.get("pose_name_cn") or 
                item.get("scene_name_cn") or 
                item.get("title_cn") or 
                f"任务 {i+1}"
            )
            
            name_en = (
                item.get("name_en") or 
                item.get("pose_name_en") or 
                item.get("scene_name_en") or 
                item.get("title_en") or 
                f"Task {i+1}"
            )
            
            # 智能提取描述/提示词
            description = item.get("description", "")
            focus_points = item.get("focus_points", [])
            camera_angle = item.get("camera_angle", "")
            category = item.get("category", "")
            priority = item.get("priority") or item.get("scene_number") or item.get("order") or (i + 1)
            composition = item.get("composition", "")
            
            # 生成英文 prompt
            prompt = self._generate_prompt_from_pose(
                name_en, description, focus_points, camera_angle, 
                category, composition, parsed_json.get("execution_notes", {})
            )
            
            # 构建标准化对象
            converted_prompt = {
                "prompt": prompt,
                "pose_name_cn": name_cn,
                "pose_name_en": name_en,
                "priority": priority,
                "category": category or "converted"
            }
            
            # 保留所有其他字段
            for key, value in item.items():
                if key not in ["prompt", "pose_name_cn", "pose_name_en", "name_cn", "name_en", "scene_name_cn", "scene_name_en", "priority", "scene_number", "order"]:
                    converted_prompt[key] = value
            
            converted_prompts.append(converted_prompt)
            print(f"[VLM]   [OK] 转换 #{i+1}: {name_cn} -> {len(prompt)} 字符")
        
        # 按优先级排序
        converted_prompts.sort(key=lambda x: x.get("priority", 999))
        
        # 保留原始 JSON 中的其他顶层字段
        extra_data = {}
        for key, value in parsed_json.items():
            if key != task_field:
                extra_data[key] = value
        
        print(f"[VLM] [OK] 转换完成，共 {len(converted_prompts)} 个提示词")
        
        return {
            "valid": True,
            "parsed_data": {
                "total_prompts": len(converted_prompts),
                "prompts": converted_prompts,
                "source_field": task_field,  # 记录源字段名
                **extra_data  # 保留其他字段
            },
            "message": f"本地验证通过（智能转换: 数组字段 '{task_field}'）"
        }
    
    def _convert_detailed_format(self, parsed_json: Dict) -> Dict:
        """转换详细格式（格式 2）为标准格式"""
        poses = parsed_json["poses"]
        converted_prompts = []
        
        print(f"[VLM] 开始转换详细格式，共 {len(poses)} 个姿势...")
        
        for i, pose in enumerate(poses):
            if not isinstance(pose, dict):
                return {
                    "valid": False,
                    "error": f"姿势 #{i+1} 不是对象类型",
                    "suggestion": "每个姿势必须是 JSON 对象格式"
                }
            
            # 提取关键信息
            name_cn = pose.get("name_cn", f"姿势 {i+1}")
            name_en = pose.get("name_en", f"Pose {i+1}")
            description = pose.get("description", "")
            focus_points = pose.get("focus_points", [])
            camera_angle = pose.get("camera_angle", "")
            category = pose.get("category", "")
            priority = pose.get("priority", i + 1)
            composition = pose.get("composition", "")
            
            # 智能生成英文 prompt
            prompt = self._generate_prompt_from_pose(
                name_en, description, focus_points, camera_angle, 
                category, composition, parsed_json.get("execution_notes", {})
            )
            
            converted_prompts.append({
                "pose_name_cn": name_cn,
                "pose_name_en": name_en,
                "prompt": prompt,
                "priority": priority,
                "category": category
            })
            
            print(f"[VLM]   [OK] 转换 #{i+1}: {name_cn} -> {len(prompt)} 字符")
        
        # 按优先级排序
        converted_prompts.sort(key=lambda x: x.get("priority", 999))
        
        print(f"[VLM] [OK] 转换完成，共 {len(converted_prompts)} 个提示词")
        
        return {
            "valid": True,
            "parsed_data": {
                "total_prompts": len(converted_prompts),
                "model_gender": parsed_json.get("model_gender", ""),
                "model_body_type": parsed_json.get("model_body_type", ""),
                "prompts": converted_prompts
            },
            "message": "本地验证通过（格式 2 - 已转换）"
        }
    
    def _generate_prompt_from_pose(
        self, 
        name_en: str, 
        description: str, 
        focus_points: List[str],
        camera_angle: str,
        category: str,
        composition: str,
        execution_notes: Dict
    ) -> str:
        """
        从姿势描述生成英文 prompt
        
        Args:
            name_en: 英文名称
            description: 姿势描述（中文）
            focus_points: 关注点列表
            camera_angle: 机位角度
            category: 类别
            composition: 构图说明
            execution_notes: 执行说明
            
        Returns:
            生成的英文 prompt
        """
        prompt_parts = []
        
        # 1. 基础指令
        prompt_parts.append(f"Transform person to {name_en}")
        
        # 2. 解析描述中的关键姿势元素
        desc_lower = description.lower()
        
        # 身体姿态
        if "站立" in description or "standing" in desc_lower:
            prompt_parts.append("standing pose")
        if "双脚" in description and "与肩同宽" in description:
            prompt_parts.append("feet shoulder-width apart")
        if "双脚" in description and ("并拢" in description or "错开" in description):
            prompt_parts.append("feet together or slightly staggered")
        
        # 手臂位置
        if "双手" in description:
            if "下垂" in description or "naturally" in desc_lower:
                prompt_parts.append("arms naturally down")
            elif "叉腰" in description:
                prompt_parts.append("hands on hips")
            elif "伸展" in description or "extend" in desc_lower:
                prompt_parts.append("arms extended")
        
        # 头部和视线
        if "面向镜头" in description or "facing" in desc_lower:
            prompt_parts.append("facing camera")
        if "头部" in description and "转" in description:
            prompt_parts.append("head turned towards camera")
        
        # 背部姿态
        if "背对" in description or "back" in desc_lower:
            prompt_parts.append("back facing camera")
        if "侧身" in description or "side" in desc_lower:
            if "45" in description or "45" in camera_angle:
                prompt_parts.append("45-degree side angle")
            elif "90" in description or "完全侧" in description:
                prompt_parts.append("90-degree side profile")
        
        # 动态姿势
        if "前倾" in description or "lean" in desc_lower:
            prompt_parts.append("leaning forward")
        if "运动" in description or "athletic" in desc_lower or "运动" in category:
            prompt_parts.append("athletic dynamic pose")
        if "慢跑" in description or "拉伸" in description:
            prompt_parts.append("running preparation or stretching motion")
        
        # 特写和细节
        if "半身" in description or "half-body" in name_en.lower():
            prompt_parts.append("half-body shot from waist up")
        if "特写" in description or "close-up" in name_en.lower():
            prompt_parts.append("close-up detail shot")
        
        # 3. 添加机位信息
        if camera_angle:
            if "正面" in camera_angle:
                prompt_parts.append("front view")
            elif "背面" in camera_angle:
                prompt_parts.append("back view")
            elif "侧" in camera_angle:
                if "45" in camera_angle:
                    prompt_parts.append("45-degree angle")
                else:
                    prompt_parts.append("side view")
            elif "仰拍" in camera_angle or "略微仰" in camera_angle:
                prompt_parts.append("slightly low angle")
        
        # 4. 添加构图信息
        if "3:4" in composition or "竖版" in composition:
            prompt_parts.append("3:4 vertical portrait")
        
        # 5. 添加通用设置（从 execution_notes）
        if execution_notes:
            # 背景
            bg = execution_notes.get("background", "")
            if "white" in bg.lower() or "白" in bg:
                prompt_parts.append("pure white background")
            
            # 光线
            lighting = execution_notes.get("lighting", "")
            if lighting:
                if "soft" in lighting.lower():
                    prompt_parts.append("soft even lighting")
        
        # 6. 添加默认设置（如果没有指定）
        if not any("background" in p for p in prompt_parts):
            prompt_parts.append("white background")
        if not any("light" in p for p in prompt_parts):
            prompt_parts.append("professional studio lighting")
        
        # 7. 添加质量和风格描述
        prompt_parts.append("high quality")
        prompt_parts.append("natural look")
        
        # 组合为最终 prompt
        prompt = ", ".join(prompt_parts)
        
        return prompt
    
    def _convert_storyboard_format(self, parsed_json: Dict) -> Dict:
        """
        转换视频分镜格式（格式 3）为标准格式
        
        Args:
            parsed_json: 包含 storyboard 字段的 JSON 对象
            
        Returns:
            验证结果
        """
        storyboard = parsed_json["storyboard"]
        converted_prompts = []
        
        print(f"[VLM] 开始转换视频分镜格式，共 {len(storyboard)} 个场景...")
        
        for i, scene in enumerate(storyboard):
            if not isinstance(scene, dict):
                return {
                    "valid": False,
                    "error": f"场景 #{i+1} 不是对象类型",
                    "suggestion": "每个场景必须是 JSON 对象格式"
                }
            
            # 提取关键信息
            scene_number = scene.get("scene_number", i + 1)
            scene_name_cn = scene.get("scene_name_cn", f"场景 {scene_number}")
            scene_name_en = scene.get("scene_name_en", f"Scene {scene_number}")
            prompt = scene.get("prompt", "")
            
            # 检查 prompt 是否存在
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                return {
                    "valid": False,
                    "error": f"场景 #{scene_number} ({scene_name_cn}) 缺少或提示词为空",
                    "suggestion": "每个场景必须包含有效的 'prompt' 字段"
                }
            
            # 提取其他有用信息
            duration = scene.get("duration", "")
            shot_type = scene.get("shot_type", "")
            focus_point = scene.get("focus_point", "")
            text_overlay = scene.get("text_overlay", "")
            transition = scene.get("transition", "")
            
            converted_prompts.append({
                "prompt": prompt,
                "pose_name_cn": scene_name_cn,
                "pose_name_en": scene_name_en,
                "scene_number": scene_number,
                "priority": scene_number,  # 使用场景编号作为优先级
                "category": "storyboard",
                # 保留额外的场景信息（可选）
                "duration": duration,
                "shot_type": shot_type,
                "focus_point": focus_point,
                "text_overlay": text_overlay,
                "transition": transition
            })
            
            print(f"[VLM]   [OK] 转换场景 #{scene_number}: {scene_name_cn} -> {len(prompt)} 字符")
        
        print(f"[VLM] [OK] 转换完成，共 {len(converted_prompts)} 个场景提示词")
        
        # 提取视频信息（如果有）
        video_info = parsed_json.get("video_info", {})
        
        return {
            "valid": True,
            "parsed_data": {
                "total_prompts": len(converted_prompts),
                "video_info": video_info,
                "product_features": parsed_json.get("product_features", []),
                "prompts": converted_prompts
            },
            "message": "本地验证通过（格式 3 - 视频分镜格式已转换）"
        }
    
    def _try_fix_json_with_vlm(self, json_string: str, error_msg: str) -> Optional[Dict]:
        """
        尝试使用VLM智能修复JSON语法错误
        
        Args:
            json_string: 有语法错误的JSON字符串
            error_msg: 错误信息
            
        Returns:
            修复后的验证结果，如果修复失败返回None
        """
        fix_prompt = f"""你是一个JSON修复专家。用户提供了一个有语法错误的JSON，请修复它并验证内容。

**原始JSON（有错误）：**
```
{json_string}
```

**错误信息：**
{error_msg}

**常见问题和修复方法：**
1. 无效转义: JSON中双引号内的单引号不需要转义，应该将 `\\'` 改为 `'`
2. 多余转义: 删除不必要的反斜杠
3. 缺失逗号/括号: 补全缺失的语法元素

**你的任务：**
1. 修复JSON语法错误
2. 保持所有内容不变，只修复语法
3. 验证修复后的JSON
4. 按照标准格式返回验证结果

**必须严格返回以下JSON格式（不要添加markdown标记）：**
```json
{{
  "valid": true,
  "fixed": true,
  "original_error": "{error_msg}",
  "parsed_data": {{
    "total_prompts": 数量,
    "prompts": [
      {{
        "prompt": "提示词内容",
        "pose_name_cn": "中文名称",
        "pose_name_en": "英文名称",
        "priority": 优先级
      }}
    ]
  }}
}}
```"""

        try:
            print(f"[VLM] 调用VLM进行JSON智能修复...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的JSON修复和验证助手"},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.3,  # 降低温度，更加精确
                max_tokens=VLM_MAX_TOKENS,
                timeout=VLM_TIMEOUT
            )
            
            vlm_response = response.choices[0].message.content.strip()
            
            # 清理markdown标记
            if vlm_response.startswith("```json"):
                vlm_response = vlm_response[7:]
            if vlm_response.startswith("```"):
                vlm_response = vlm_response[3:]
            if vlm_response.endswith("```"):
                vlm_response = vlm_response[:-3]
            vlm_response = vlm_response.strip()
            
            # 解析VLM修复结果
            vlm_result = json.loads(vlm_response)
            
            if vlm_result.get("valid", False):
                print(f"[VLM] ✅ JSON修复成功！共 {vlm_result.get('parsed_data', {}).get('total_prompts', 0)} 个提示词")
                return vlm_result
            else:
                print(f"[VLM] ❌ VLM无法修复此JSON: {vlm_result.get('error', '未知')}")
                return None
                
        except Exception as e:
            print(f"[VLM] ❌ VLM修复失败: {str(e)}")
            return None
    
    def _tensor_to_base64(self, tensor) -> Optional[str]:
        """
        将 ComfyUI IMAGE tensor 转换为 JPEG base64 字符串（压缩至长边 768px）。

        Args:
            tensor: shape=(H,W,C) 或 (1,H,W,C)，值范围 [0,1]

        Returns:
            base64 编码字符串；失败时返回 None
        """
        try:
            import numpy as np
            from PIL import Image

            # 兼容 torch.Tensor 和 np.ndarray
            try:
                import torch
                if isinstance(tensor, torch.Tensor):
                    if tensor.dim() == 4:
                        tensor = tensor[0]
                    img_np = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                else:
                    img_np = (tensor * 255).clip(0, 255).astype(np.uint8)
            except ImportError:
                img_np = (tensor * 255).clip(0, 255).astype(np.uint8)

            pil_img = Image.fromarray(img_np, mode="RGB")

            # 压缩：长边限制 768px，节省 token
            max_side = 768
            w, h = pil_img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"[Optimizer] 图片转 base64 失败: {e}")
            return None

    def optimize_single_prompt(self, prompt_text: str, ref_image_b64: Optional[str] = None) -> str:
        """
        使用提示词优化器对单条 prompt 进行优化。

        Args:
            prompt_text: 原始提示词文本（中文或英文均可）
            ref_image_b64: 参考图片的 JPEG base64 字符串（可选）；
                           提供后模型可识别主体的性别、外貌、服装等特征

        Returns:
            优化后的英文提示词；若 API 调用失败则返回原始文本
        """
        if not self.client:
            print(f"[Optimizer] VLM 客户端不可用，跳过优化")
            return prompt_text

        if not prompt_text or not prompt_text.strip():
            return prompt_text

        print(f"[Optimizer] 优化提示词: {prompt_text[:80]}...")

        # 尝试顺序：1) 带图（若有）→ 2) 纯文本兜底（模型不支持视觉时自动降级）
        attempts_plan = []
        if ref_image_b64:
            attempts_plan.append(("带参考图", ref_image_b64))
        attempts_plan.append(("纯文本", None))

        for mode_label, current_image_b64 in attempts_plan:
            try:
                # 构造 user 内容：有参考图时使用多模态格式
                if current_image_b64:
                    user_content = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{current_image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "这是用户上传的参考图片，请先分析图中主体的性别、年龄、外貌、"
                                "服装、体型等特征，在优化提示词时必须完整保留这些特征，不得更改。\n\n"
                                f"需要优化的提示词：{prompt_text}"
                            ),
                        },
                    ]
                else:
                    user_content = prompt_text

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PROMPT_OPTIMIZER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.7,
                    max_tokens=512,
                    timeout=60,
                )
                optimized = response.choices[0].message.content.strip()
                # 去掉可能残留的 markdown 代码块标记
                if optimized.startswith("```"):
                    lines = optimized.splitlines()
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    optimized = "\n".join(lines).strip()
                print(f"[Optimizer] 优化完成（{mode_label}）: {optimized[:80]}...")
                return optimized if optimized else prompt_text

            except Exception as e:
                err_str = str(e)
                print(f"[Optimizer] {mode_label} 失败: {err_str}")
                # 检测到视觉不支持错误，降级为纯文本继续
                vision_err_keywords = ("does not support", "image", "vision", "multimodal",
                                       "400", "unsupported")
                if current_image_b64 and any(k in err_str.lower() for k in vision_err_keywords):
                    print(f"[Optimizer] ⚠️ 当前模型不支持视觉输入，自动降级为纯文本模式")
                    continue
                # 非视觉错误，直接保留原文
                print(f"[Optimizer] 优化失败，保留原文")
                return prompt_text

        return prompt_text

    def optimize_prompts(
        self,
        prompt_objects: List[Dict],
        image_tensors: Optional[List] = None,
        max_concurrent: int = 20,
    ) -> List[Dict]:
        """
        并发批量优化 prompt 列表中每个对象的 "prompt" 字段。

        Args:
            prompt_objects: 标准 prompt 对象列表，每个对象必须含 "prompt" 字段
            image_tensors: 用户上传的参考图片 tensor 列表（可选）；
                           取第一张转为 base64 传入优化器，用于提取主体特征
            max_concurrent: 最大并发数（默认 5，避免 API 限流）

        Returns:
            优化后的 prompt 对象列表，顺序与输入一致
        """
        if not self.client:
            print(f"[Optimizer] VLM 客户端不可用，跳过批量优化")
            return prompt_objects

        # 提取参考图 base64（仅取第一张，所有并发任务共享）
        ref_image_b64: Optional[str] = None
        if image_tensors:
            print(f"[Optimizer] 正在将参考图转换为 base64...")
            ref_image_b64 = self._tensor_to_base64(image_tensors[0])
            if ref_image_b64:
                print(f"[Optimizer] 参考图已就绪，优化时将以此图为准")
            else:
                print(f"[Optimizer] 参考图转换失败，将使用纯文本优化")

        total = len(prompt_objects)
        print(f"[Optimizer] 开始并发优化，共 {total} 条提示词，最大并发 {max_concurrent}...")

        # 用于按原始索引还原顺序的结果字典
        results_dict: Dict[int, Dict] = {}

        def _optimize_one(idx: int, obj: Dict) -> tuple:
            new_obj = dict(obj)
            original = obj.get("prompt", "")
            if original:
                optimized = self.optimize_single_prompt(original, ref_image_b64)
                new_obj["prompt"] = optimized
                print(f"[Optimizer] [{idx+1}/{total}] 完成")
            else:
                print(f"[Optimizer] [{idx+1}/{total}] 跳过（prompt 为空）")
            return idx, new_obj

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(_optimize_one, i, obj): i
                for i, obj in enumerate(prompt_objects)
            }
            for future in as_completed(futures):
                try:
                    idx, new_obj = future.result()
                    results_dict[idx] = new_obj
                except Exception as e:
                    idx = futures[future]
                    print(f"[Optimizer] [{idx+1}/{total}] 异常，保留原文: {e}")
                    results_dict[idx] = dict(prompt_objects[idx])

        # 按原始顺序组装结果
        result = [results_dict[i] for i in range(total)]
        print(f"[Optimizer] 并发优化完成")
        return result

    def test_connection(self) -> bool:
        """
        测试 VLM API 连接
        
        Returns:
            是否连接成功
        """
        if not self.client:
            print(f"[VLM] 本地模式，无需连接测试")
            return True
            
        try:
            print(f"[VLM] 测试 API 连接...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个测试助手"},
                    {"role": "user", "content": "请回复：连接成功"}
                ],
                temperature=0.7,
                max_tokens=50,
                timeout=10
            )
            
            content = response.choices[0].message.content
            print(f"[VLM] [OK] API 连接成功，响应: {content}")
            return True
            
        except Exception as e:
            print(f"[VLM] [FAIL] API 连接失败: {str(e)}")
            return False


# 便捷函数
def validate_prompts_json(json_string: str, api_key: Optional[str] = None, model: str = DEFAULT_VLM_MODEL) -> Dict:
    """
    验证提示词 JSON（便捷函数）
    
    Args:
        json_string: JSON 字符串
        api_key: VLM API Key（如果为 None，只进行本地验证）
        model: VLM 模型名称
        
    Returns:
        验证结果
    """
    validator = VLMValidator(api_key or "dummy", model)
    return validator.validate_json(json_string)

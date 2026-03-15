# RH 批量生图 Agent

ComfyUI 插件，通过 RunningHub OpenAPI 实现批量并发图像生成。

rh_api_key填入runninghub的API https://www.runninghub.cn/enterprise-api/sharedApi（仅限企业级API，需要钱包充值）

vlm_api_key填入硅基流动（siliconflow）的API  https://cloud.siliconflow.cn/playground/chat（因为调用了PRO模型，需要充值一定的金额才能使用，不需要充值很多，充一次能用很久）

---

## 输入

| 参数 | 类型 | 说明 |
|---|---|---|
| `model` | 下拉选择 | 目标模型（自动从 models_registry.json 加载） |
| `prompts_json` | 文本 | 批量提示词，支持 JSON 格式或直接输入自然语言描述 |
| `model_params` | 文本 | 模型参数 JSON，对所有批次生效，如 `{"aspectRatio": "16:9"}` |
| `max_concurrent` | 整数 | 最大并发任务数（默认 50） |
| `vlm_model` | 下拉选择 | VLM 模型（用于验证和提示词优化，需支持视觉的模型） |
| `enable_validation` | 布尔值 | 开启后用 VLM 自动验证/转换提示词格式，支持自然语言输入（默认开启） |
| `enable_prompt_optimization` | 布尔值 | 开启后在生图前对每条 prompt 进行专业化优化，有参考图时自动读取图中主体特征（默认关闭） |
| `inputcount` | 整数 | 图片输入端口数量（0 = 纯文生图；设置后点击 Update inputs 刷新端口） |
| `rh_settings` | 可选连接 | RH OpenAPI Settings 节点（API Key 优先级最高） |
| `rh_api_key` | 可选文本 | RunningHub API Key |
| `vlm_api_key` | 可选文本 | SiliconFlow API Key（启用 VLM 功能时需要） |
| `images_0..N` | 可选图片 | 参考图片输入端口（由 inputcount 控制数量） |

### prompts_json 支持的格式

**JSON 格式：**
```json
{"prompts": [{"prompt": "a beautiful landscape"}, {"prompt": "a city at night"}]}
```

**自然语言（需开启 enable_validation）：**
```
生成3张图片，正面、侧面、背面，白色背景
```

---

## 输出

| 输出 | 类型 | 说明 |
|---|---|---|
| `generated_images` | IMAGE | 所有成功生成的图片合并为一个 batch |
| `generation_log` | STRING | 每个任务的执行状态（OK / FAIL / SKIP） |

---

## API Key 配置

编辑 `config/.env`：

```ini
RH_API_KEY=your_runninghub_api_key
VLM_API_KEY=your_siliconflow_api_key
```

优先级：Settings 节点 > 节点直接输入 > 环境变量 > `config/.env`

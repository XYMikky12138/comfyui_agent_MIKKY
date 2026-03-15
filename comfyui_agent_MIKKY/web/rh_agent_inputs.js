/**
 * RH Agent 动态参数扩展
 *
 * 功能：
 * 1. 切换模型时，按 models_params.json 定义动态生成参数控件
 *    - LIST 类型 → 下拉（combo）
 *    - INT/FLOAT → 数字输入（number）
 *    - BOOLEAN   → 开关（toggle）
 *    - STRING    → 文本框（text）
 * 2. 参数控件值实时同步到隐藏的 model_params JSON 字段（Python 读取此字段）
 * 3. 图片输入端口标记为可选（optional: true）
 */

import { app } from "../../scripts/app.js";

const NODE_NAME  = "RHAgentNode";
const PARAMS_URL = "/extensions/comfyui_agent/models_params.json";

// ── 模型参数数据（异步加载一次）────────────────────────────
let _modelsData  = null;
let _loadPromise = null;

function ensureModelsData() {
    if (_modelsData)  return Promise.resolve(_modelsData);
    if (_loadPromise) return _loadPromise;

    _loadPromise = fetch(PARAMS_URL)
        .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
        .then(data => {
            _modelsData = data;
            console.log(`[RH_Agent] 模型参数数据已加载：${Object.keys(data).length} 个模型`);
            return data;
        })
        .catch(e => {
            console.warn("[RH_Agent] 加载模型参数数据失败:", e);
            _loadPromise = null;
            return null;
        });

    return _loadPromise;
}

ensureModelsData(); // 预加载

// ── 工具 ────────────────────────────────────────────────────
function findWidget(node, name) {
    return node.widgets?.find(w => w.name === name) ?? null;
}

// 节点内置 widget 名，防止参数名与它们冲突
const RESERVED_NAMES = new Set([
    "model", "prompts_json", "model_params",
    "max_concurrent", "vlm_model", "enable_validation",
    "inputcount", "rh_api_key", "vlm_api_key"
]);

// ── 注册扩展 ────────────────────────────────────────────────
app.registerExtension({
    name: "Comfy.RHAgent.DynamicParams",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        ensureModelsData();

        // ── onNodeCreated ──────────────────────────────────
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            this._rhParamKeys   = new Set();
            this._rhConfigured  = false;   // 防止与 onConfigure 竞争
            this._rhInit();
            return r;
        };

        // ── onConfigure（从已保存工作流加载时触发）─────────
        //
        // 注意：LiteGraph 在调用 onConfigure 之前就已经把 widgets_values 按位置
        // 赋给各 widget，所以这里无法通过修改 data 来影响赋值结果。
        // 正确做法是：检测到旧格式后，直接按名字覆写各 widget 的 value。
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            this._rhConfigured = true;
            const self = this;

            const r = origConfigure?.apply(this, arguments);

            // ── 兼容旧格式：按名字覆写被错位的静态 widget 值 ─
            // 旧代码将动态参数控件（如 aspectRatio、resolution）插入 model 之后
            // 并参与了序列化，导致 widgets_values 比静态 widget 数量多出 N 项。
            // LiteGraph 按位置赋值，所以静态 widget 全部接收到错误的值。
            // 此处通过检测数组长度来识别旧格式并直接修正。
            //
            // 静态 widget 固定 9 个（nodes.py INPUT_TYPES 顺序）：
            //   [0]model [1]prompts_json [2]model_params [3]max_concurrent
            //   [4]vlm_model [5]enable_validation [6]inputcount
            //   [7]rh_api_key [8]vlm_api_key
            const STATIC_NAMES = [
                "model", "prompts_json", "model_params", "max_concurrent",
                "vlm_model", "enable_validation", "inputcount",
                "rh_api_key", "vlm_api_key",
            ];
            const STATIC_COUNT = STATIC_NAMES.length; // 9

            if (data?.widgets_values?.length > STATIC_COUNT) {
                const N = data.widgets_values.length - STATIC_COUNT;
                const v = data.widgets_values;
                // 动态参数值在 index 1..N；静态值：model=v[0], 其余从 v[N+1] 起
                const corrected = [
                    v[0],         // model
                    v[N + 1],     // prompts_json
                    v[N + 2],     // model_params
                    v[N + 3],     // max_concurrent
                    v[N + 4],     // vlm_model
                    v[N + 5],     // enable_validation
                    v[N + 6],     // inputcount
                    v[N + 7],     // rh_api_key
                    v[N + 8],     // vlm_api_key
                ];
                // 直接按名字覆写，绕过 LiteGraph 已经完成的错误赋值
                for (let i = 0; i < STATIC_NAMES.length; i++) {
                    const w = findWidget(self, STATIC_NAMES[i]);
                    if (w && corrected[i] !== undefined) w.value = corrected[i];
                }
                console.log(
                    `[RH_Agent] 检测到旧格式工作流（widgets_values 多出 ${N} 项），` +
                    `已按名字覆写静态 widget 值`
                );
            }

            // onConfigure + 值修正完成后，model_params 已正确还原，重建动态参数控件
            ensureModelsData().then(() => {
                self._rhRemoveParamWidgets();
                const mw  = findWidget(self, "model");
                const mpw = findWidget(self, "model_params");
                if (!mw) return;
                let saved = {};
                try { saved = JSON.parse(mpw?.value || "{}"); } catch (_) {}
                self._rhAddParamWidgets(mw.value, saved);
                // 只在内容增加导致需要更多空间时扩展，不缩小用户已调整的尺寸
                const needed = self.computeSize();
                self.setSize([
                    Math.max(self.size?.[0] ?? 0, needed[0]),
                    Math.max(self.size?.[1] ?? 0, needed[1]),
                ]);
                self.setDirtyCanvas(true, true);
            });
            return r;
        };

        // ── 初始化（新建节点时）─────────────────────────────
        nodeType.prototype._rhInit = function () {
            const self = this;

            // 隐藏 model_params 原始 JSON 框（Python 仍可读取，前端无需展示）
            const mpw = findWidget(this, "model_params");
            if (mpw) {
                mpw.computeSize = () => [0, -4];
                mpw.hidden = true;
            }

            // 添加 Update inputs 按钮
            this._rhAddButton();

            // 监听 model 下拉变化
            const modelWidget = findWidget(this, "model");
            if (modelWidget) {
                const origCb = modelWidget.callback;
                modelWidget.callback = async function (v, canvas, node, pos, evt) {
                    if (origCb) origCb.call(this, v, canvas, node, pos, evt);
                    await self._rhOnModelChange(v);
                };
            }

            // 新建节点：等数据加载完后用默认值填充参数控件
            // （_rhConfigured 为 false 表示 onConfigure 还未运行，即真正的新建）
            ensureModelsData().then(() => {
                if (self._rhConfigured) return; // onConfigure 已处理，跳过
                const mw = findWidget(self, "model");
                if (!mw) return;
                self._rhAddParamWidgets(mw.value, {});
                const needed = self.computeSize();
                self.setSize([
                    Math.max(self.size?.[0] ?? 0, needed[0]),
                    Math.max(self.size?.[1] ?? 0, needed[1]),
                ]);
                self.setDirtyCanvas(true, true);
            });

            // 初始化图片端口
            const icw = findWidget(this, "inputcount");
            if (icw) this._rhUpdateImageInputs(icw.value ?? 5);
        };

        // ── 用户切换模型 ────────────────────────────────────
        nodeType.prototype._rhOnModelChange = async function (modelName) {
            await ensureModelsData();
            this._rhRemoveParamWidgets();
            this._rhAddParamWidgets(modelName, {}); // 用模型默认值
            this._rhSyncJson();

            // 同步 inputcount（模型切换时强制重建端口，清除旧连接）
            const icw = findWidget(this, "inputcount");
            if (icw && _modelsData?.[modelName] != null) {
                const ic = _modelsData[modelName].imageCount ?? 0;
                icw.value = ic;
                this._rhUpdateImageInputs(ic, true); // forceReset
            }

            const needed = this.computeSize();
            this.setSize([
                Math.max(this.size?.[0] ?? 0, needed[0]),
                Math.max(this.size?.[1] ?? 0, needed[1]),
            ]);
            this.setDirtyCanvas(true, true);
        };

        // ── 为指定模型添加参数控件 ──────────────────────────
        nodeType.prototype._rhAddParamWidgets = function (modelName, savedValues) {
            if (!_modelsData?.[modelName]) return;
            const { params = {} } = _modelsData[modelName];
            const self = this;

            for (const [key, info] of Object.entries(params)) {
                if (RESERVED_NAMES.has(key)) continue;
                this._rhParamKeys.add(key);
                const saved = savedValues[key];
                let w = null;

                switch (info.type) {
                    case "LIST": {
                        const opts = info.options ?? [];
                        if (!opts.length) break;
                        const defVal = opts.includes(saved ?? info.defaultValue)
                            ? (saved ?? info.defaultValue)
                            : opts[0];
                        w = this.addWidget("combo", key, defVal,
                            () => self._rhSyncJson(),
                            { values: opts }
                        );
                        break;
                    }
                    case "INT": {
                        const v = parseInt(saved ?? info.defaultValue ?? 0, 10);
                        w = this.addWidget("number", key, isNaN(v) ? 0 : v,
                            () => self._rhSyncJson(),
                            { precision: 0, step: 1 }
                        );
                        break;
                    }
                    case "FLOAT": {
                        const v = parseFloat(saved ?? info.defaultValue ?? 0);
                        w = this.addWidget("number", key, isNaN(v) ? 0 : v,
                            () => self._rhSyncJson(),
                            { precision: 2, step: 0.01 }
                        );
                        break;
                    }
                    case "BOOLEAN": {
                        const raw = saved ?? info.defaultValue ?? false;
                        const bv  = raw === true || String(raw).toLowerCase() === "true";
                        w = this.addWidget("toggle", key, bv, () => self._rhSyncJson());
                        break;
                    }
                    default: {
                        // STRING 或其他
                        const defVal = String(saved ?? info.defaultValue ?? "");
                        w = this.addWidget("text", key, defVal, () => self._rhSyncJson());
                        break;
                    }
                }

                // 动态控件值已存入 model_params JSON，不需要再单独序列化
                // 必须设置 w.options.serialize（LiteGraph 检查的是 options，不是 w.serialize）
                if (w) {
                    w.options = w.options ?? {};
                    w.options.serialize = false;
                }
            }

            // 将参数控件移到 model 下拉之后（视觉上紧跟模型选择）
            this._rhReorder();
        };

        // ── 将模型参数控件排列到 model 之后、prompts_json 之前 ─
        // 静态参数（max_concurrent 及以下）始终保持 Python 定义的顺序不变
        nodeType.prototype._rhReorder = function () {
            if (!this.widgets || !this._rhParamKeys?.size) return;
            const params = this.widgets.filter(w =>  this._rhParamKeys.has(w.name));
            const others = this.widgets.filter(w => !this._rhParamKeys.has(w.name));
            // 插入到 model 之后、prompts_json 之前，保持静态参数不被动态控件夹在中间
            const modelIdx = others.findIndex(w => w.name === "model");
            const insertAt = modelIdx >= 0 ? modelIdx + 1 : 0;
            others.splice(insertAt, 0, ...params);
            this.widgets = others;
        };

        // ── 移除所有参数控件 ────────────────────────────────
        nodeType.prototype._rhRemoveParamWidgets = function () {
            if (!this.widgets || !this._rhParamKeys?.size) return;
            for (let i = this.widgets.length - 1; i >= 0; i--) {
                if (this._rhParamKeys.has(this.widgets[i].name)) {
                    this.widgets.splice(i, 1);
                }
            }
            this._rhParamKeys = new Set();
        };

        // ── 将控件值同步到隐藏的 model_params JSON ───────────
        nodeType.prototype._rhSyncJson = function () {
            const mpw = findWidget(this, "model_params");
            if (!mpw || !this.widgets) return;
            const values = {};
            for (const w of this.widgets) {
                if (this._rhParamKeys?.has(w.name)) values[w.name] = w.value;
            }
            mpw.value = Object.keys(values).length
                ? JSON.stringify(values, null, 2)
                : "{}";
        };

        // ── 动态图片输入端口（差量更新，保留已有连接）───────
        nodeType.prototype._rhUpdateImageInputs = function (count, forceReset) {
            // 统计当前已有的 images_* 端口数量
            const current = (this.inputs || []).filter(
                inp => inp.name?.startsWith("images_")
            ).length;

            if (!forceReset && current === count) return; // 数量不变则跳过

            if (forceReset) {
                // 强制重置（仅在切换模型时使用）：全部删除后重建
                if (this.inputs) {
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        if (this.inputs[i].name?.startsWith("images_")) this.removeInput(i);
                    }
                }
                for (let i = 0; i < count; i++) {
                    this.addInput(`images_${i}`, "IMAGE");
                    const inp = this.inputs[this.inputs.length - 1];
                    if (inp) inp.optional = true;
                }
            } else if (count > current) {
                // 只追加新端口，不删除已有端口（保留现有连接）
                for (let i = current; i < count; i++) {
                    this.addInput(`images_${i}`, "IMAGE");
                    const inp = this.inputs[this.inputs.length - 1];
                    if (inp) inp.optional = true;
                }
            } else {
                // 从末尾移除多余的端口
                for (let i = current - 1; i >= count; i--) {
                    const idx = (this.inputs || []).findIndex(
                        inp => inp.name === `images_${i}`
                    );
                    if (idx >= 0) this.removeInput(idx);
                }
            }

            // 尺寸只扩大不缩小，避免重置用户手动调整的节点大小
            const needed = this.computeSize();
            this.setSize([
                Math.max(this.size?.[0] ?? 0, needed[0]),
                Math.max(this.size?.[1] ?? 0, needed[1]),
            ]);
            this.setDirtyCanvas(true, true);
        };

        // ── Update inputs 按钮 ──────────────────────────────
        nodeType.prototype._rhAddButton = function () {
            const self = this;
            try {
                const btn = this.addWidget("button", "Update inputs", "update", () => {
                    const w = findWidget(self, "inputcount");
                    self._rhUpdateImageInputs(w?.value ?? 0);
                });
                if (btn) btn.serialize = false;
            } catch (e) {
                console.error("[RH_Agent] 添加按钮失败:", e);
            }
        };

        // ── 右键菜单 ────────────────────────────────────────
        const origMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const r    = origMenuOptions?.apply(this, arguments);
            const self = this;
            options.unshift(
                {
                    content: "Update inputs",
                    callback: () => {
                        const w = findWidget(self, "inputcount");
                        self._rhUpdateImageInputs(w?.value ?? 0);
                    }
                },
                {
                    content: "Reset model params",
                    callback: async () => {
                        await ensureModelsData();
                        const mw = findWidget(self, "model");
                        if (mw) await self._rhOnModelChange(mw.value);
                    }
                },
                null
            );
            return r;
        };
    }
});

console.log("[RH_Agent] 动态参数扩展已加载");

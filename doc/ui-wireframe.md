# LLM-Vis M0 低保真 UI Wireframe

- 状态：M0 可核验交互规格
- 日期：2026-08-29
- 最后更新：2026-08-31；最后完整浏览器验收：2026-08-30
- 实现状态：M0–M3.7 与 M3.10 自包含离线 UI 已实现并验收；M3.8/M3.9 功能与自动化已通过、最终 `file://` 页面待手工确认。M3.10 已把同一中央 DAG 切换为从上到下阅读、单击结构化解释与去抖双击下钻；直属母图上下文、按需 Browse/Inspector/精确 Layers/Supporting analysis 保持不变。官方 Model Explorer consumer 仍为 Partial，M4 timeline 未实现

## 1. 目标

这份 wireframe 固定 LLM-Vis 的信息层级和关键状态，使前端 spike、静态报告与后续 UI 可以围绕同一组可验收任务实现。它必须同时回答：

- 当前查看的是 L0、L1 还是 L2；
- 当前 Scenario 是什么；
- 节点来自 config 语义投影还是受限 capture；
- 指标是 Exact、Formula、Estimated、Measured 还是 Unknown；
- 覆盖率是多少，缺失原因在哪里；
- 热点结论依赖哪些公式和假设；
- 没有 trace 时为何不存在 Runtime 数值。

本规格遵循 [Model Map IR](model-map-ir.md)、[Adapter 指南](adapter-guide.md)、[测量协议](measurement-protocol.md)和 [DR-0007](decisions/DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md)。

## 2. 全局框架

```text
┌ LLM-Vis / Model · rev ───────── [Scenario ▾] [Search graph/layer/ID] [Report status] ┐
├─────────────────────────────────────────────────────────────────────────────────────┤
│ Model graph · L0→L1              [Theory heat ▾] [Browse] [Inspector] [Analysis] │
│ › Text decoder layers · [L×3 → A] ×16 · 64 exact layers · L/A 图例 · 非循环│
│   （精确 64/78 层默认隐藏；首次展开后单行横向滚动）                           │
│ ┌ Back · View · Breadcrumb ─────────────────────────────── −  +  Fit · Read-only ┐ │
│ │                                                                               │ │
│ │                         CENTRAL DAG CANVAS                                    │ │
│ │                                  Input                                        │ │
│ │                                    ↓                                          │ │
│ │                               Embedding                                       │ │
│ │                                    ↓                                          │ │
│ │                 state/route → Decoder Pattern → optional branches             │ │
│ │                                    ↓                                          │ │
│ │                              Norm → Head → Logits                              │ │
│ │                                                     minimap                   │ │
│ └───────────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ › Supporting analysis（默认折叠：Capture / Structure / Cost / Roofline / Runtime） │
└─────────────────────────────────────────────────────────────────────────────────────┘

[Browse]    → 左侧按需抽屉：Definitions；Diagnostics 默认二次折叠
[Inspector] → 右侧按需抽屉：Explain/Tensors/Cost/Runtime/Provenance/Coverage
```

不可隐藏的全局上下文：模型与 revision、Scenario、当前 L0/L1 层级。Metric origin、Coverage 与零执行状态可收纳到 Report status/Inspector，但不能删除或改写。默认首屏不得同时常驻 Browse、Inspector、Layer 全量按钮和分析表格；信息完整性通过按需抽屉与折叠区保留。

### 2.1 M3.9 One-Input CLI 与报告证据状态

M3.9 的启动面是 `llm-vis view INPUT`，不是浏览器表单。CLI 已统一接受 HF Model ID（单段 `model` 或 `owner/model`）、HF 模型页 URL、HF `config.json` blob/resolve URL、本地 config 文件/目录和 JSON 文本/stdin 五类语义输入。当前不实现浏览器启动页、粘贴框或文件选择器；浏览器只打开生成后的自包含离线报告。

用户不指定 `--output` 时，CLI 只在确认当前工作目录位于 LLM Vis Git checkout（Git marker 与项目身份均匹配）后，才在仓库根目录的 `artifacts/generated/<model-slug>-<config-hash8>` 生成 artifact；同名目录已存在时依次使用 `-2`、`-3`，不覆盖旧结果。仓库外省略 `--output` 会返回带修复提示的错误，不向无关 cwd 写文件。CLI 输出 artifact/report 绝对路径并默认调用系统浏览器；`--no-open` 可关闭打开。打开失败只产生 warning，不丢失已生成 artifact。用户显式传入的 `--output` 路径及其既有行为不变，并可在仓库外使用。

离线报告沿用现有 graph-first 页面，主图标题下增加紧凑证据栏：`Target input: config.json only`、source kind/requested+resolved revision、config SHA-256、`Known adapter` 或 `Unsupported · opaque`、当前 view evidence，以及 `no target weights · no model code · no full forward`。证据栏不增加第二张图：

- `Known adapter`：显示 adapter 名、requested/resolved revision、config hash 与 coverage，中央位置直接显示已有同一 L0→L1→operator DAG；
- `Partial`：已知区域继续使用同一 DAG，证据不足的节点为 opaque/Unknown，页头显示范围而不用绿色“完成”掩盖；
- `Unsupported (opaque)`：保留可读 config inventory、source/hash/Diagnostic 和所需 adapter/remediation，中央画布显示一个有证据范围说明的 opaque model card，不伪造内部 DAG；
- `Unsupported (opaque)` 的最小形态：语法有效的 JSON object 即使没有 `model_type` 或模型维度证据，也生成 C0 报告，中央图只有一个 coverage 0 的 opaque architecture 节点；
- `Invalid / blocked`：只有 malformed JSON、顶层非 object、非 HF 远程 host/redirect、非法显式 local revision，或超出 2 MiB、64 层嵌套/100,000 项安全限额时，CLI 显示带 code/hint 的错误并不生成伪报告；私有/gated 仓库显示 `HF_AUTH_REQUIRED` 与本地 config 修复路径。

远程 source 显示已固定的 Hugging Face commit；本地/inline source 默认显示 `sha256:<digest>` 内容身份，不冒充 HF commit。不可信 `_commit_hash`、本机绝对源路径、HF token 或请求头都不得出现在可分享页面。

## 3. L0：模型宏观视图

L0 默认目标为 10–50 个节点，不平铺 64 层或数百个专家。

```text
Input
  ├─▶ Vision Tower × 27 ─▶ Projector ─┐
  └─▶ Token Embedding ────────────────┼─▶ Decoder Pattern × 16 ─▶ Final Norm ─▶ LM Head
                                      │      [L L L A]
                                      └─▶ MTP [conditional]

Decoder instances: 64
State summary: Linear layers → recurrent state (48); Full Attention → KV cache (16)
```

L0 节点至少显示：Definition label、instance count、semantic kind、是否 conditional/opaque、来源和 Coverage badge。GLM 的专家显示为：

```text
Dense Block × 3 → Sparse Block × 75
                     └─ Router → Expert Pool [256 total · top-8 active] + Shared Expert × 1
```

Expert Pool 默认不展开 256 个专家。

## 4. L1：代表性 Block

从 L0 的 pattern、Layer Strip 或搜索结果进入 L1。Breadcrumb 始终保留原实例位置。

```text
Model › Decoder pattern › layer.3 [Full Attention]

hidden ─▶ RMSNorm ─▶ Q/K/V projection ─▶ Attention semantic core ─▶ O projection ─┐
   └──────────────────────────────── residual ────────────────────────────────────┼─▶ hidden'
KV state: read [B,N_KV,L,D] · write [B,N_KV,T,D]                                 │
                                                  RMSNorm ─▶ FFN ────────────────┘
```

边使用独立样式或文字编码 `data`、`state_read`、`state_write`、`route` 和 `control`；颜色不能同时承载多个含义。节点选择后，Inspector 显示 Definition 和具体 Instance，不把代表层误认为全部层的实测结果。

### 4.1 M3.5 主画布交互契约

M3.5 的中间区域由结构列表升级为 Netron/ComfyUI 风格的端口级 DAG。画布消费渲染中立 `graph-view.json`，不直接重新解释 config，也不将前端布局字段写回 Model Map IR。

```text
Top bar: [Scenario ▾] [Global search…] [Report status]
Canvas:  [Back] [View ▾] [Breadcrumb]                         [−] [+] [Fit]
Panels:  [Browse] [Inspector] [Analysis]                         Minimap

                 input ports
                      ↓
┌ Node label · semantic kind ─ origin/coverage ─┐
│ core attributes                 N nodes ↓      │
│ click: Explain · double-click/Enter: child DAG │
└────────────────────────────────────────────────┘
                      ↓
                 output ports
```

节点卡只放理解数据流必需的核心信息：`label`、`kind`、代表的 instance/group、主要 I/O、origin/evidence、coverage 与 opaque/conditional 状态。完整 config 字段、公式和诊断仍在 Inspector，不把节点卡做成长表单。

| 连线 kind | 画布语义 | 非颜色编码 |
|---|---|---|
| `data` | 当前 token/hidden activation 从 output port 到 input port | 实线 + Tensor 标签 |
| `route` | Router 选择到 Expert Pool/Shared Expert | 带 `route` 文字的虚线 |
| `control` | config 可证明的显式路由/调度控制关系 | 点线 + `control` |
| `state_read` | KV/recurrent state 从独立 state rail 进入 Block | 轨道线 + `read` |
| `state_write` | Block 更新 KV/recurrent state rail | 轨道线 + `write` |

每条线必须连接明确的 `source_port_id → target_port_id`，固定以三行显示 kind/Tensor 名、紧凑 shape 和 dtype；不可证明的 shape/dtype 显示 `Unknown` 与原因，不猜测。Weight 边保留在 Model Map 事实层中，默认不进入主数据流画布。state rail 表达“本步读/写外部状态”，不用跨 token 回边破坏单步 DAG。不得为 MTP 伪造 decoder control Tensor；已验收的支路是 `decoder hidden → MTP → MTP Draft Logits`，conditional 作为节点属性保留。

只有 `drilldown_view_id` 指向实际存在 view 的节点才显示 `N nodes ↓` 角标。角标有至少 28×28 的触摸/点击区，单击角标直接进入 primary child；M3.10 起整张卡片单击自动打开结构化 Explain，双击或键盘 Enter 下钻，叶节点的 Enter/Space 只解释。节点的可访问名称和 `aria-keyshortcuts` 必须说明该差异，画布状态栏同时显示当前可下钻节点数。叶节点不显示角标且双击不跳转，避免让用户猜哪些节点可以打开。

必须可操作的动作为：拖拽平移、Ctrl/Cmd+滚轮或按钮缩放、Fit to view、minimap 定位、角标/节点双击/键盘 Enter 下钻、Back/breadcrumb 返回、搜索命中定位，以及选中节点后的上游/下游路径高亮。普通滚轮应继续滚动页面，避免画布形成滚动陷阱。一次只绘制当前 view；64/78 层和 256 个专家以 pattern/group 折叠，默认每级不超过 200 个可见节点。

Qwen 的最小可验收路径是 `Input → Token Embedding → Decoder Pattern → Final Norm → LM Head → Logits`，Vision/MTP 以 config 可追溯分支显示。Decoder Pattern 可下钻到 Linear Attention 和 Full Attention 代表视图，两者分别使用 recurrent-state 和 KV state rail。GLM L0 保留 Dense×3 与 Sparse×75 的折叠阶段；Sparse L1 显示 `Router → Expert Pool (top-8/256) + Shared Expert → Reduce`，不展开 256 个专家，DSA 内部未知部分保持 opaque。

### 4.2 M3.6 递归算子披露契约

M3.6 不增加 Beginner/Expert 开关。新手和熟练用户看到同一份 DAG 与同一组稳定节点身份；区别只在于用户是否继续展开复合节点、是否打开 Inspector。M3.10 将原 `N ops ›` 升级为 `N nodes ↓`，数量来自 primary child view 中除 boundary 外的真实节点数。

```text
Block DAG
  Full Attention [28 nodes ↓]
        │ double-click / Enter / badge
        ▼
Full Attention operators
  Q/K/V GEMM → Q/K Norm → RoPE → KV append
       → QKᵀ MatMul → Scale → Mask → Softmax → P×V MatMul
       → Merge → Gate SiLU × Context → O GEMM
        │
        └── [← Collapse] 回到同一个父节点与父 view viewport/selection
```

交互约束：

- 任意深度都复用当前中央画布、pan/zoom/Fit/minimap、搜索与六页 Inspector；operator child views 不作为顶部 View 下拉框的常驻选项，避免控制区重新变杂。进入 child 后只增加一个带 `↳` 的当前临时项，用户仍可辨认当前位置；其他 child 只能从复合节点、breadcrumb 或搜索进入。
- 每个 view 保存自己的 viewport 与 selection；展开/折叠恢复相应状态。Scenario 只刷新 Metric/heat，不重建 GraphView，不改变 view/node/port/edge ID。
- 基础算子显示 `primitive_kind` 与核心 shape；不显示下钻角标。Opaque 显示明确原因与 coverage，也不显示假入口。
- child boundary 是可见的输入/输出/state rail；其端口与父 compound port 由正式 binding 对应。父 input 对应 child boundary output，父 output 对应 child boundary input，shape/dtype/role/TensorSpec 必须相同。
- 热图统计集合是当前 view 的 `cost_frontier_node_ids`。边界/container 不进入分母；父节点在 child view 中不可见且不参与合计；Unknown/Excluded 仍显示灰色原因，但不作为 0 参与排序。
- Full Attention/FFN 的 Cost Inspector 显示 parent total、known child subtotal、unattributed remainder、coverage 与公式范围。`unattributed=0` 只有在已知子项确实与父公式相等时显示；未知项永不通过减法或默认值伪造成 0。
- GLM Router/TopK/Expert 只显示静态结构。`runtime_route_known=false`、`selected_expert_ids=null` 与 `experts_materialized=0` 必须可从卡片/Inspector 读取；不得绘制某个 token 实际去了哪个专家。

## 5. L2：两种必须区分的状态

### 5.1 M1 config 语义投影

```text
┌ CONFIG SEMANTIC PROJECTION ─ LogicalOps not captured ─ Coverage: semantic only ┐
│ Norm → Full Attention → Residual → Norm → FFN → Residual                      │
│                                                                                │
│ No aten.* names · no execution order claim · no fusion/kernel claim            │
└────────────────────────────────────────────────────────────────────────────────┘
```

该状态只显示 `SemanticNode`。必须展示 `LOGICAL_OPS_NOT_CAPTURED`，禁止为视觉完整而生成虚构 ATen/custom op。

### 5.2 M2 受限代表块 capture

```text
┌ CAPTURED: tiny/reduced Full Attention · FakeTensor/meta · not full-model perf ┐
│ aten.rms_norm → aten.linear → ... → [opaque custom op] → aten.linear          │
│ Coverage: 87% · Opaque: 1 · Source: bounded torch.export                       │
└────────────────────────────────────────────────────────────────────────────────┘
```

该状态只能来自 Tiny/语义同类缩小代表块。页头和 Inspector 固定显示 `architecture_equivalence=semantic-kind-only`、`valid_for_full_model_performance=false`；不能把它当成 27B/GLM 的 latency、fusion 或 Kernel 证据。

## 6. Layer Strip

默认关闭态本身必须可读，不再只显示 `Layers · 64`：

```text
› Text decoder layers  [L×3 → A] ×16  64 exact layers
  L Linear Attention · recurrent state   A Full Attention · KV cache
  one-way DAG · repeat ≠ loop / shared weights
```

这里的 `×16` 是结构摘要，不是回边。紧凑可见文案使用 `repeat ≠ loop / shared weights`，完整无障碍说明为 `repeat notation is not a cycle and does not imply weight sharing`。页面不得画 `layer.63 → layer.0`，也不得暗示 64 层共享权重。Qwen 摘要必须严格来自完整有序 `layerStrip`；GLM 使用 `D×3 → M×75`，Tiny 使用 `D×4`。如果实际结构 tuple 包含 tail/偏差，摘要按实际序列回退；如果结构仍重复但 `anomaly=true`，摘要必须附加 `N deviation(s)`，不能静默显示成无偏差周期。

用户首次展开后，页面惰性创建全部精确层按钮：

```text
Layer       0  1  2  3  4  5  6  7 ... 63
Attention   L  L  L  A  L  L  L  A ...  A
State       S  S  S  K  S  S  S  K ...  K
MLP         D  D  D  D  D  D  D  D ...  D
Coverage    ●  ●  ●  ●  ●  ●  ●  ● ...  ●
```

Legend 不能采用脱离模型语义的全局字母表：Qwen `L/A` 由 Attention/state 解释，GLM `D/M` 由 Dense/MoE FFN、DSA Attention 和 KV cache 共同解释；文字必须始终可见，颜色只作辅助。Coverage 使用独立符号：complete、partial、opaque、unknown。精确 Layer Strip 默认折叠，展开后只占一行并横向滚动，不再把 64/78 层换行铺满首屏；每个短标签都有包含 layer index、Attention、MLP、state、expected pattern、coverage/anomaly 的完整 `aria-label`。

首次展开前 `.layer` 数为 0，展开后为 64/78；关闭再打开不得重复创建。点击 layer 继续更新 Inspector，保留原始 layer index、Instance、captured/config/opaque 与异常状态，并同步到对应代表 L1 GraphView。单纯展开/关闭层栏不得改变当前 GraphView、breadcrumb、selection、viewport 或 Scenario。

## 7. Inspector

| Tab | 必须显示 |
|---|---|
| Explain | 语义用途、Definition/Instance、简化数据流；候选解释不得冒充事实 |
| Tensors | 输入输出、symbolic/concrete shape、dtype/storage dtype、state/alias；缺失为 Unknown |
| Cost | parameters、FLOPs、logical bytes、KV/state、arithmetic intensity、理论瓶颈/下界、HardwareProfile provenance、单位、origin、公式和假设 |
| Runtime | 仅导入 trace 后显示 measured 数据；无 trace 固定显示 Unknown 与导入提示 |
| Provenance | revision、SourceArtifact、config JSON Pointer、adapter/capture rule、工具版本 |
| Coverage | status、coverage 数值、opaque/unmapped/failed 原因及 Diagnostic |

Inspector 不能用 `0` 代替缺失；也不能将 Formula 的 roofline lower bound放进 Runtime latency 字段。

M3.5 画布的节点、端口和边都是可选对象。M3.10 起，真实鼠标/键盘节点选择自动打开 Explain；搜索定位、返回时恢复 selection、Scenario/heat 刷新只更新 Inspector 的待查看状态，不主动遮住主图。Explain 显示用途、教学简式公式、I/O 契约、child views 与证据边界；Tensors 显示所有 ports 及可用 TensorSpec。选中 port 或 edge 时仍只更新待查看状态，Tensors 优先定位其 `tensor_spec_id`。Provenance 显示 config JSON Pointer/adapter rule/capture source，Coverage 显示该对象的 coverage/opaque/Unknown 原因。Cost 和 Runtime 仅显示能通过 `subject_ids` 可追溯关联的现有 Metric，不为画布节点伪造数值。

### 4.3 M3.10 从上到下与节点解释契约

- GraphView `layout_direction=DOWN`；每个 rank 水平排列并围绕最宽行居中，输入端口在卡片顶部、输出端口在底部，主边整体从上到下。相邻 rank 使用行间正交走廊；跨 rank/back edge 使用画布左右外侧、按稳定 edge interval 分配的独立 lane，Tensor 标签锚定在无节点的行间段，禁止穿过其他卡片。
- current minimap 与 Parent context 同样使用底部到顶部的节点间连线，不保留旧的右侧到左侧缩略线。
- 单击卡片使用 320 ms 去抖后选择节点并打开 Explain；双击、角标、键盘和 view 切换先统一取消 mount 级 timer，再对有效 primary child 下钻。回调还会核对原 view、node 与 DOM 连接状态，leaf/opaque 双击不改变 view，只显示 Explain。
- Explain 的 `Simplified equation` 由受控 semantic kind/primitive kind 映射生成，并明确标记为教学方程，不是 Cost 页的 FLOPs/bytes AST、kernel 实现或 measured runtime。
- Qwen Hybrid Decoder 的 Explain 同时列出 Linear Attention 和 Full Attention 两个实际 child view；双击只进入稳定 primary，alternate 通过明确按钮进入。任何 child 列表都从可解析 GraphView 关系派生，不从 Model Map `children/child_ids` 猜测。
- opaque 区域必须显示 `Unknown — opaque evidence boundary`；GLM DSA、actual expert route/IDs/weights 与 Linear delta core 不因解释 UI 而获得虚构公式。

## 8. Scenario 与 workload diff

Scenario selector 显示：

```text
phase · batch B · new tokens T · past tokens L · S_KV=L+T · output length
activation dtype · weight format · KV dtype · backend · hardware profile/Unknown
```

若没有 Scenario：

```text
No workload selected — workload-dependent metrics are Unknown.
```

同模型 workload diff 以稳定 `(subject_id, metric name)` 匹配：`added`、`removed`、`changed`、`unknown`。左右任一 Metric 为 Unknown 时，不计算 0 基准 delta。跨模型/revision diff 明确显示“Deferred to M5”。

## 9. Hotspots 与 roofline

热点表每行显示：rank、subject、metric、value/unit、origin、coverage、Scenario。排序规则：

1. 仅已知且同单位的值参与数值排名；
2. partial value 保留 Coverage badge；
3. Unknown 行放入未排名区，不按 0 排在末尾；
4. Formula、Estimated 和 Measured 不静默混成同一排行榜；
5. M0–M3 点击热点定位对应 Instance 与 Inspector 证据；M3.5 进一步定位对应 GraphNode 并高亮其上下游，该联动已验收。

Roofline 面板只显示：arithmetic intensity、compute lower bound、bandwidth lower bound、二者的 max lower bound 和 bottleneck class，并固定文字：

```text
Theoretical lower bounds — not a latency estimate.
```

没有用户提供且 dtype 匹配的 HardwareProfile 时，lower bounds 和 bottleneck 为 Unknown；不得填入默认 GPU。

主图提供 `Pressure / Compute / Memory / Off` 理论热力层：Compute 使用可归因的公式 FLOPs，Memory 使用 logical minimum bytes，Pressure 使用 `max(FLOPs/peak, logical bytes/bandwidth)`。强度只在“当前 Scenario + 当前 view + 当前 mode”内按 `raw / hottest known node` 归一化，不允许跨 view、跨场景比较；legend 固定显示公式、HardwareProfile 名称/ID/provenance、known/total coverage、`Unknown is not zero` 和 `not measured latency`。切换 Scenario 只重算热力值，不改变 view/node ID 或选择。

热力层必须复用 Cost Metric 的显式归属规则，而不是直接按 `subject_ids` 盲连：L0 仅 decoder pattern 聚合成员 Instance；L1 仅 Attention 对应 `*.attention`、FFN 对应 `*.ffn`。Norm、Residual、输入输出等无独立子项指标的节点显示灰色 `not attributable`；metric 缺失、opaque 或硬件不匹配显示灰色斜纹 `Unknown` 并给出原因。Qwen 当前可显示公式热度；GLM DSA/MoE executed FLOPs/bytes 未知时全图保持 Unknown，禁止用 active parameters、weights storage 或 0 代替。颜色不能覆盖白色选择描边或上下游路径描边，tooltip/Inspector 必须同时给出数值、origin、coverage 和公式证据。

## 10. Unknown 与 Coverage 视觉规则

| 数据状态 | 展示 | 禁止行为 |
|---|---|---|
| Unknown | `Unknown` + 原因 + coverage 0/unknown | 显示 `0`、空白或灰色数值 |
| Partial | 已知部分值 + `Partial 62%` | 当成完整值排名而不标注 |
| Opaque | Opaque node + 已知端口 | 删除该区域或虚构内部 op |
| Unmapped runtime | `Unattributed/Unmapped` bucket | 静默丢弃 dispatch 时间 |
| Formula | 值 + Formula badge + 公式/假设 | 标记为 Measured |
| Measured | 值 + trace/run provenance | 与 counter replay wall time混用 |

## 11. 可核验任务

| ID | 操作 | 通过条件 | M0–M3 状态 |
|---|---|---|---|
| UI-01 | 打开 artifact | 模型/revision、Scenario、层级可见；Coverage 与零执行状态可从 Report status 一步打开 | PASS |
| UI-02 | 查看 Qwen L0 | 64 层压缩为 pattern，KV/recurrent state 分开 | PASS |
| UI-03 | 展开 Layer Strip 并点击 layer.3 | Inspector 待查看状态保留 layer.3 Instance，并显示 Full Attention、captured/config coverage 与异常状态 | PASS |
| UI-04 | 打开 M1 L2 | 显示 config semantic projection 与 `LogicalOps not captured` | PASS |
| UI-05 | 打开 M2 L2 | 显示 tiny/reduced、FakeTensor/meta 和非性能证据告警 | PASS |
| UI-06 | 选择未知 GLM cost | Value 为 Unknown，Coverage 为 0/partial，原因可见 | PASS |
| UI-07 | 查看 Runtime 且无 trace | 所有 runtime 值为 Unknown/null，不出现 0 ms | PASS |
| UI-08 | 切换 prefill/decode | Scenario 全字段更新，结构 ID 不变，成本按 Scenario 更新 | PASS |
| UI-09 | 打开热点 | 只排名可比较已知值；点击后 Inspector 显示 Instance、Metric 公式、Symbol 绑定与假设 | PASS |
| UI-10 | 断网打开静态 HTML | 核心结构、Layer Strip、Inspector 和数据不发起外部请求 | PASS |

M3.5 的独立退出任务如下，Qwen/GLM 已在 2026-08-30 完成实际浏览器验收：

| ID | 操作 | 通过条件 | M3.5 状态 |
|---|---|---|---|
| DAG-01 | 打开 Qwen L0 | 从 Input 到 Logits 存在端口级可达主路径，Vision/MTP 分支有 config evidence | PASS |
| DAG-02 | 双击 Qwen Decoder Pattern | 进入 Linear/Full Attention L1，breadcrumb/Back 可返回，recurrent/KV state rail 不混用 | PASS |
| DAG-03 | 打开 GLM L0/L1 | Dense/Sparse pattern 可下钻，Router/Expert Pool/Shared Expert/Reduce 不展开 256 专家 | PASS |
| DAG-04 | 点击节点、port 和 edge | 六页 Inspector 定位关联 subject/Tensor/Metric/evidence，Unknown 不补零 | PASS |
| DAG-05 | 搜索节点/选层/点击热点 | 画布打开正确 view、定位节点并高亮上下游 | PASS |
| DAG-06 | 平移、缩放、Fit、minimap | 操作可逆，线端持续锚定具体 port，当前 view 不丢失 | PASS |
| DAG-07 | 切换 Scenario | GraphView/node/port/edge 结构 ID 不变，只更新可关联的指标 | PASS |
| DAG-08 | 断网打开 Qwen/GLM HTML | DAG 不请求外部资源，console 无 warning/error | PASS |
| DAG-09 | 默认打开 Qwen/GLM 报告 | 首屏只有紧凑全局栏、中央主图和折叠入口；Browse/Inspector/Analysis 不抢占主图宽度 | PASS |
| DAG-10 | 打开/关闭 Browse 与 Inspector | 抽屉按需出现，关闭后主图状态、当前 view、选择和路径高亮不丢失；Escape/关闭按钮可返回触发点 | PASS |
| DAG-11 | 展开 Layers/Supporting analysis | Layer 保持单行滚动，分析内容在折叠区内；两者默认不占主视觉 | PASS |
| DAG-12 | 查找并打开可下钻节点 | Qwen L0 恰有 1 个、GLM L0 恰有 2 个 `L1 ›`；角标点击、双击、Enter 可下钻，Space 只选择，叶节点无角标 | PASS |
| DAG-13 | 切换 Pressure/Compute/Memory 与 Scenario | Qwen 仅可归因节点着色并保留选择/路径；legend 显示 synthetic provenance 与非实测声明；GLM 全部 Unknown 且不补 0；console 为空 | PASS |

M3.8 Layer Disclosure 使用以下浏览器退出任务；完整事实与边界见主计划 LAY-01～LAY-10：

| ID | 操作 | 通过条件 | M3.8 状态 |
|---|---|---|---|
| LAY-UI-01 | 初始打开 Qwen | 关闭态显示 `[L×3 → A] ×16`、`64 exact layers`、L/A 文字图例和非循环说明；DOM 中尚无 `.layer` | 静态契约 PASS；待浏览器 |
| LAY-UI-02 | 键盘展开 Layers | 恰生成 64 个可聚焦按钮，索引 `0..63`、标签 `LLLA ×16`，逐层条单行横向滚动 | 实现/数据 PASS；待浏览器 |
| LAY-UI-03 | 点击 layer 0、3、63 | 分别进入 Linear/Full/Full 代表 L1；Inspector 保留精确实际 layer index/Instance | 待验收 |
| LAY-UI-04 | 关闭、重开 Layers | 仍为 64 个按钮且当前 view/selection/viewport/Scenario 不变，不重复生成 | 待验收 |
| LAY-UI-05 | 初始打开 GLM/Tiny | 分别显示 `D×3 → M×75`/`D×4` 与模型相关图例，不出现 Qwen 专用文案 | 自动化/报告 PASS；待浏览器 |
| LAY-UI-06 | 640px 窄屏查看并操作 | 摘要、图例与非循环说明可换行，中央 DAG 不被横向撑开，summary/button 均可键盘操作 | CSS/原生交互契约 PASS；待浏览器 |

M3.9 的 CLI/报告 UI 退出状态如下；真实 `file://` 页面是唯一仍待手工确认的 UI 项：

| ID | 操作 | 页面通过条件 | M3.9 状态 |
|---|---|---|---|
| IMP-UI-01 | 通过 CLI 传入 Model ID/模型页/config URL | 固定 revision/hash 后生成报告，不要求用户选 adapter；不声称存在浏览器输入页 | **PASS：CLI/resolver 自动化** |
| IMP-UI-02 | 通过 CLI 传入本地 config、inline JSON 或 stdin | 报告显示 hash 内容身份而不泄露本机源绝对路径 | **PASS：CLI/artifact 自动化** |
| IMP-UI-03 | 导入 Qwen/GLM 已知 config | `Known adapter` 与 source/revision/hash/coverage/零执行状态已进入 HTML，中央主视觉仍是现有同一 DAG | **静态/报告自动化 PASS；`file://` 手工待验** |
| IMP-UI-04 | 导入未知或证据不足的 JSON object | 显示 `Unsupported · opaque`、C0/coverage 证据；缺乏模型证据时只有单个 opaque 节点，无伪造 child view/成本 | **静态/GraphView 自动化 PASS；`file://` 手工待验** |
| IMP-UI-05 | 导入 malformed/非 object/超限/非 HF 远程 JSON | CLI 显示类型/安全限额 code 与 hint，不生成伪报告；普通有效 object 则降级而非拒绝 | **PASS：resolver 正/负自动化** |
| IMP-UI-06 | 默认打开、`--no-open` 或模拟打开失败 | 仅在已确认的 LLM Vis Git checkout 中默认写入仓库根 `artifacts/generated/<slug>-<hash8>`，冲突时使用 `-2`/`-3` 且不覆盖；仓库外省略 `--output` 给出可操作错误；显式 `--output` 仍可用且不变；关闭时只给路径；打开失败不丢 artifact | **PASS（2026-09-08 maintenance）**：当前环境与 Python 3.9 隔离定向各 19 项、全量 Python 276 项与真实 Chromium 17 项通过；真实 CLI 连续生成 base/`-2` 且均被 Git ignore |

M3.6 直接采用以下退出编号：

| ID | 操作 | 通过条件 | M3.6 状态 |
|---|---|---|---|
| OP-01 | 展开 Qwen Full Attention | 同一画布出现 Q/K/V/O GEMM、QKᵀ/P×V MatMul、Softmax 及完整可达路径 | **PASS** |
| OP-02 | 展开 Qwen FFN | 显示恰好三个 GEMM、SiLU 与 Multiply，Gate/Up→Multiply→Down 连接正确 | **PASS** |
| OP-03 | 检查展开边界 | 父子 port 一一绑定，role/dtype/shape/TensorSpec 与输入输出可达性不变 | **PASS** |
| OP-04 | 打开 Cost reconciliation | parent、known child subtotal、unattributed、coverage 可核对；Unknown/partial 不补 0 | **PASS** |
| OP-05 | 展开/折叠并观察热图 legend | 只统计当前 visible frontier，父子不同时进入 known/total、排名或合计 | **PASS** |
| OP-06 | 查看 Qwen Full/Linear state | KV read/write 接 K/V append，conv/delta recurrent state 接各自计算核心，二者不混用 | **PASS** |
| OP-07 | 查看 GLM Sparse MoE | Router GEMM→TopK→Expert Pool/Shared Expert→Combine；不出现实际 expert route | **PASS** |
| OP-08 | 尝试展开 GLM DSA | 节点保持 opaque/coverage 0 且无下钻入口、无虚构内部 op | **PASS** |
| OP-09 | 展开、折叠、搜索、选择、切 Scenario | 当前稳定结构 ID 不变；各 view viewport/selection 可恢复 | **PASS** |
| OP-10 | 检查 Report status/Provenance | weights/model construction/full forward/remote code 仍全为 false | **PASS** |

## 12. 当前实现证据与缺口

当前离线 HTML 已收敛为 graph-first 结构：顶栏只保留模型/revision、Scenario、唯一全局搜索与 Report status；中央 DAG 默认占满可用宽度；Definition/Diagnostic 位于 Browse 抽屉，六页 Inspector 位于右抽屉，64/78 层 Layer Strip 和所有 Supporting analysis 默认折叠。Definition、Semantic/LogicalOp、capture、逐层 captured/config/opaque、异常层、成本、热点、roofline、workload diff 与 Runtime Unknown 证据均仍保留，但不再同时铺满首屏。2026-08-30 的最终 M0–M3 浏览器验收确认：Qwen layer.0 整层可同时看到 Dense 与 Linear State 两种代表体，但具体 Dense FFN 节点只显示 Dense capture SourceArtifact，Linear Attention 节点只显示 Linear State source；layer.1 config-only 只显示 config source，不串入任何 Tiny capture。捕获 Tensor 逐项显示 `origin=capture`、`materialized=false` 和唯一 source。layer.3 仍显示 `full_attention · captured · anomaly=false`；Cost 可读取离线 Symbol 绑定，Provenance 固定显示零权重/零完整 forward。Report status 把结构范围与成本指标可用性分开，例如 GLM 显示 `structure 78/82 · 95.1% | cost 3/8 known`，不再用平均 metric coverage 冒充 decoder layer 覆盖率。GLM 的 executed FLOPs/bytes、roofline 与 Runtime 均显示 Unknown/null/0% coverage，不参与热点排名，也不以 0 代替。对应自动化见 [离线 artifact 集成测试](../tests/integration/test_inspect_artifact.py)、[M3 报告测试](../tests/integration/test_m3_report.py)、[capture 报告测试](../tests/integration/test_capture_cli_report.py)和 [Model Explorer adapter 集成测试](../tests/integration/test_model_explorer_adapter.py)。

M3.5 浏览器验收进一步确认：Qwen L0 为 11 node/19 port/10 edge，Linear/Full L1 各为 10/24/13；GLM L0 为 9/15/8，Dense L1 为 10/20/11，Sparse DSA+MoE L1 为 13/30/17。默认 70% 可读视图、27% Fit 总览、真实拖拽平移及 minimap 同步、缩放/Back、节点/port/edge 选择、Linear/KV state rail、route/control 样式、三行 Tensor 名/shape/dtype、Layer Strip、跨 view 搜索、热点定位、上下游高亮和 Scenario 指标刷新均通过。graph-first 复测进一步确认 1280×720 首屏主图全宽、两个抽屉默认关闭、搜索命中 Sparse L1 时 Inspector 不自动遮挡、Browse/Inspector 可开关且 view/selection 不丢失。DAG-12/13 复测确认 Qwen/GLM 的 `L1 ›` 数量分别为 1/2，角标、双击与键盘契约可用；Qwen L1 只有 Attention/FFN 获得公式热度，Scenario 切换保持 view/selection，GLM 全图为 Unknown 而不是冷色 0，synthetic HardwareProfile 与非 latency 声明在 legend/Inspector 可见。GLM DSA 保持 opaque，未知 KV shape 显示 `Unknown` 与明确原因；L0/L1 不串入 Tiny L2 capture。Qwen/GLM 两页 console 均无 warning/error，24/24 机器退出项及 Python 3.9/3.12 各 150 项测试通过。

M3.6 浏览器验收确认：Qwen Full Attention 展开为 32 node/36 edge，包含 Q/K/V/O GEMM、Q/K/V reshape+transpose、4→24 GQA KV-head Broadcast、QKᵀ/P×V MatMul、Softmax、RMSNorm、RoPE、cache append、context transpose 与 output gate；FFN 展开为三个 GEMM、SiLU、Multiply；Linear Attention 的 4 条 recurrent-state 边接入 Conv/opaque delta core。Softmax 搜索与 prefill→decode 切换保持相同 node ID；端口选择在 `Collapse`→重新展开后恢复，随后选择 node 会清除 port 描边；operator views 不作为 View 下拉框常驻项，当前 child 只显示一个 `↳` 临时项。Cost Inspector 显示 FLOPs `complete/signed_remainder=0/inconsistent=false` 与 logical bytes `partial/signed_remainder>0/unknown_is_zero=false`；热图只显示当前 frontier（Pressure `4/28 known`，Compute `6/28 known`）。GLM Sparse view 为 14 node/19 edge，显示 Router GEMM、TopK indices、`[B,T,8] float32` routing weights→Combine、route/control、virtual Expert/Shared Expert；Expert child view 显示 Gather/Scatter 与三个 GEMM，且不产生 runtime route/weight value；DSA 没有下钻入口，GLM frontier 保持 `0/10 known`。跨 view Inspector 残留已修复；两页 console 均无 warning/error。27/27 checked-in 资产检查和 Python 3.9/3.12 各 192 项通过。

M3.7 的直属母图、current minimap、per-view viewport/selection、结构 parent focus 和 700px 折叠路径已完成 CTX-01～CTX-10 浏览器验收。M3.8 功能与自动化已完成：Python 3.9/3.12 各 198、Ruff、27/27 verifier、golden current；重生成的 Qwen/GLM 报告分别内嵌 `[L×3 → A] ×16`/`D×3 → M×75`，JavaScript syntax 与 safety flags 通过。自动浏览器因 URL policy 拒绝重载本地 `file://` 页面，因此没有绕过策略；LAY-UI-01～06 的最终目视/交互状态保留为用户手动刷新退出项。M3.9 One-Input CLI、generic C0/opaque GraphView 和 Source/Evidence HTML 已有自动化证据；最终 `file://` 页面的实际展示仍保留为手工退出项，且不声称有浏览器启动页。M3.10 EXP-01～EXP-10 已完成：所有 view 从上到下，单击 Explain、去抖双击、真实 child 选择、Softmax 教学公式、DSA opaque 边界及 Parent/current minimap 经 Qwen/GLM 浏览器验证；跨 rank 边使用稳定左右 lane，Qwen 7 个与 GLM 6 个 view 的几何采样均为 0 条 edge/label 穿过非端点卡片；700×900 窄屏无横向溢出，console 0 warning/error；Python 3.9/3.12 各 260、Ruff、27/27 verifier 与两页 JavaScript syntax 通过。

仍未完成的还有 M3.8 最终手动浏览器退出、M3.9 IMP-08 真实 `file://` 页面手工退出、本里程碑外的官方 Model Explorer consumer 真实加载/交互验收、面向 10k 原始 op 图的性能测试，以及 M4 trace timeline。官方 consumer 与超大 raw-op 图继续按 [DR-0003](decisions/DR-0003-model-explorer-bounded-spike.md) 标为 Partial；它们不应与已经通过的 M0–M3.7 自包含离线 DAG、已经通过自动化的 M3.8/M3.9 功能混为一谈。

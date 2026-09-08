# LLM-Vis Adapter 指南

状态：M0–M3.6 已实现事实层与递归投影接入契约；M3.9 One-Input Model Import 功能与自动化已完成，`file://` 真实页面待手工验收；M5 扩展项显式标注
日期：2026-08-29

Adapter 将外部模型事实转换成 [Model Map IR v0.1](model-map-ir.md)。它负责声明自己知道什么、证据来自哪里和哪些区域仍未知；它不负责替用户执行不可信代码，也不能把 config 推导伪装成真实 forward 图。

## 1. 首批 adapter

| Adapter | M1 目标 | 不在 M1 范围内 |
|---|---|---|
| Tiny Dense | 通用 Dense baseline、Definition/Instance、参数与 state 契约 | 复杂动态控制流 |
| Qwen3.8-27B | 已知 config adapter、64 层周期、Vision/Text/Projector、KV/recurrent state、MTP | 完整多模态 forward、真实 Logical Ops |
| GLM-5.3-BF16 | 78 层 Dense→MoE 宏观图、虚拟 Expert Pool、total/active 参数口径 | 动态 DSA/MoE 路径、executed-cost |
| Local Torch（M5 候选） | 尚未实现；未来只允许显式、可审计的本地入口 | M0–M3.6 不读取或执行用户 Python 模型代码 |

M1 的共同出口是离线可验证的静态 artifact。M0–M3.6 全程默认零权重、零完整模型 forward，完整 meta tree 默认禁用。受限的 `torch.export`、Logical Ops 和 opaque capture fallback 属于 M2；成本公式属于 M3；M3.5/M3.6 GraphView 仍是 config 语义投影；M4a/M4b 只导入外部 runtime trace。

## 2. 输入与信任分类

| 输入 | MVP 是否可读 | MVP 是否可执行 | 说明 |
|---|---:|---:|---|
| 本地/远程 `config.json` | 是 | 不适用 | 使用数据解析器，不导入模型仓库 Python |
| 固定 revision 的模型 metadata/许可证 | 是 | **否** | 远程 resolver 只取仓库 metadata 与 `config.json`；不取 Python 源码 |
| Hugging Face `auto_map` 目标 | 是 | **否** | 记录目标并产生 remote-code Diagnostic |
| 标准 Transformers 已安装实现 | 否 | 否 | 当前 adapter 不导入 Transformers 模型实现 |
| 用户显式指定的本地 import path | 否 | 否 | Local Torch adapter 延后 M5 |
| safetensors/权重文件 | M0–M3.6 默认不读 | 否 | 参数 inventory 优先由 config/允许的 meta shape 推导 |
| 模型权重 | **不下载、不加载** | 否 | 不属于 M0–M3.6 或 M4 trace importer 输入 |

详细策略见 [DR-0002](decisions/DR-0002-remote-code-disabled-in-mvp.md)。MVP 接口中不得提供能绕过该策略的 `trust_remote_code=True` 路径。普通子进程不是安全边界。

M3.9 不改变这张信任表：“可读”仍只表示将 config 当作不可信纯数据解析，不表示会执行 config 中声明的 class、`auto_map` 或模型代码。

## 3. 解析流程

Adapter runner 按以下顺序工作：

1. 规范化输入：Model ID、本地 `config.json` 或包含它的目录；
2. 解析并固定完整 revision，记录本地缓存命中、网络使用和文件哈希；
3. 用纯数据方式读取 `config.json` 的 `model_type`、`architectures`、`auto_map` 和嵌套 config；
4. 选择已知 config adapter；未知架构停留在 C0，并返回可操作的 Diagnostic；
5. config-first 生成惰性的 Definition、Instance、SemanticNode、state 类别和参数公式；
6. M2 只构造项目自有 Tiny 语义代表 Block，并使用 FakeTensor/meta；完整 meta tree 默认禁用；
7. 折叠重复层、保留例外实例和 provenance；
8. 校验 IR，写入 manifest、SourceArtifact、diagnostics 和离线报告。

### 3.1 M3.9 One-Input 规范化契约

> 实施状态：**功能与自动化完成，IMP-08 真实 `file://` 页面待手工验收**。Resolver 和 `llm-vis view` 已接受 Hugging Face Model ID（单段 `model` 或 `owner/model`）、HF 模型页 URL、HF config URL、本地 config 文件/目录以及 JSON 文本/stdin；本里程碑不包含浏览器启动页或文件选择器。

Input normalizer 统一接受五类用户输入：

| Input kind | 示例 | 规范化结果 |
|---|---|---|
| HF Model ID | `gpt2` 或 `Qwen/Qwen3.8-27B` | 单段或 namespaced repo ID + requested revision，再固定 commit/config |
| HF 模型页 URL | `https://huggingface.co/gpt2` 或 `https://huggingface.co/Qwen/Qwen3.8-27B` | 提取单段或 namespaced repo ID/revision，不解析 HTML 作 config |
| HF config URL | `/blob/<rev>/config.json` 或 `/resolve/<rev>/config.json` | 规范化 repo/revision/path，固定 commit 后再读 JSON |
| 本地 config | `./config.json` 或包含它的目录 | 本地 JSON object + canonical hash；报告默认脱敏源绝对路径 |
| JSON 文本 | CLI 参数中的 JSON object 或 stdin `-` | JSON object + canonical hash + `sha256:<digest>` 内容身份 |

所有形式都产生统一 `ResolvedConfig` 语义，但不把来源差异抹掉：manifest 保留 input kind、repo/identifier、requested/resolved revision、source URI、canonical JSON SHA-256、cache/network/license 证据。远程 branch/tag/`main` 在取 config 前固定为 Hugging Face immutable commit；本地/inline JSON 默认使用 `sha256:<digest>` 内容身份，不冒充 HF commit。嵌入 revision 只有严格 40–64 位十六进制 commit 才接受；显式 revision 只可使用 immutable commit 或完全匹配内容的 `sha256:<digest>`。路径、token-like 字符串和其他 `_commit_hash` 不进入 artifact，非法显式值也不会在错误中回显。

远程输入只接受精确 `https://huggingface.co` host 与可规范化的 model/config 路径，不实现任意 URL fetcher。解析器对远程、本地与 inline JSON 统一限制为 2 MiB、嵌套深度 64 和 100,000 个 container items，并执行超时、redirect/final-host 校验。Malformed JSON、非 object 顶层、HTML 伪 JSON 和超限输入被拒绝；语法有效的 JSON object 会进入 adapter/fallback 判定，不因缺少已知 `model_type` 而被拒绝。HF token/请求头不得进入 manifest、log 或报告；私有/gated 仓库在未支持认证时返回 `HF_AUTH_REQUIRED`，提示用户提供已自行下载的本地 `config.json`。

规范化后按证据分流：

1. **Known adapter**：registry 只按可消费的顶层 `model_type` 命中并生成当前同一 Model Map 与 L0→L1→operator GraphView；显式非法字段报错，不得被 generic fallback 吞掉，未知 wrapper 不能只凭 nested type 路由；输入形式不得改变稳定身份、shape、state 或 opaque 边界；
2. **Partial**：只编译 adapter/config 能证明的区域，其余保持 opaque/Unknown，报告显示范围和 coverage；
3. **Unsupported (C0)**：未注册 `model_type` 或有效 JSON object 仍生成可读证据报告，保留 config inventory/source/hash/Diagnostic 并给出 adapter remediation，不伪造内部 DAG 或成本。如果 JSON 缺乏任何可识别的文本模型证据，GraphView 只有一个 coverage 0 的 opaque architecture 节点；若存在 config 可证明的通用维度/causal-LM 声明，可显示保守的 unverified skeleton，内部仍 opaque；
4. **Invalid input**：只有 malformed JSON、顶层非 object、超出安全限额或不受信的 URL/redirect 返回可操作错误。字段很少但语法有效的 JSON object 不被伪装成 C1，也不被粗暴拒绝。

M3.9 默认路径仅调用 config-first `inspect`，不隐式触发 `capture`、下载/读取权重、remote code、目标/完整模型构造、target/full forward、compile 或 profiler。One-Input 是可用性与 provenance 增强，不是新 adapter 证据源。

当前 registry 直接按 `model_type` 选择已知 config adapter，并把 `architectures`/`auto_map` 保留为 metadata/安全证据；不会导入或实例化入口 class。以下优先级仅是 M5 通用 adapter 的候选设计，不是 M0–M3.6 已实现行为：

1. 用户显式指定的 class/task；
2. config `architectures` 指向的精确实现类；
3. 与任务匹配的 task-specific AutoClass；
4. base `AutoModel` fallback。

当前 adapter 选择和 metadata 写入 manifest。未来若实现通用 class resolution，每次选择和 fallback 也必须写入 manifest；包含 Vision Tower、LM Head 或 MTP 的入口 class 不能被视为可互换。

## 4. Adapter 能力声明

每个 adapter 必须暴露可机读的能力摘要。下面是语义示例，不限定最终序列化字段：

```json
{
  "adapter_id": "qwen3_5_config",
  "adapter_version": "0.1.0",
  "model_types": ["qwen3_5"],
  "structure": "config",
  "meta_validation": "tiny_or_reduced_block_only",
  "full_meta_tree": false,
  "weights": false,
  "full_model_forward": false,
  "semantic_projection": true,
  "logical_ops": false,
  "runtime": false,
  "remote_code_execution": false
}
```

能力声明用于选择 adapter 和生成 Coverage，不能用来宣称某个具体 artifact 已经达到对应能力；artifact 的 C0–C5 等级由实际证据决定。

## 5. M1 输出契约

M0–M3.6 当前 config adapter 输出：

- Model 与完整 source revision；
- Definition/Instance 层级、module/semantic path 和层索引；
- layer pattern、重复组、周期和异常层；
- config 可证明的参数量、KV cache、recurrent state、MoE route 等类别；
- config 推导的 SemanticNode、识别 rule、evidence、confidence；
- artifact-local ID、canonical semantic key；
- unsupported/opaque/unknown 区域及结构化 Diagnostic；
- adapter 名称/metadata、零执行 safety、fallback Diagnostic 和 metric coverage。

真实 parameter/buffer 名称清单、共享权重对象关系和完整 data/state edge 需要模型 module 证据，M0–M3.6 不通过读取权重或执行完整模型来补齐。M2 只为项目自有代表块写入 Tensor/LogicalOp/Lowering。

### 5.1 M1 的 L2 限制

Adapter 可以为了 UI 钻取生成 L2 风格的“语义契约”，但必须满足：

- 对象类型仍为 `SemanticNode`，不是 `LogicalOp`；
- `evidence` 指向 config/adapter rule，manifest 记录 `capability.level=C1` 与 `representative_block_captured=false`；
- 只声明 config/官方 adapter 能证明的组件和端口；
- 不生成 `aten.*`、custom op、op ordinal、真实执行顺序、fusion 或 Kernel；
- artifact 中 `logical_ops=[]`，并显示 `LOGICAL_OPS_NOT_CAPTURED`。

M2 捕获成功后才由 capture backend 写入 Logical Ops，并通过 Lowering 关联语义节点。M1 config adapter 不应直接依赖 `torch.export`，否则无法保持静态回退路径。

## 6. Config-first 与 meta 预算

已知模型优先直接从 config 生成 Definition/Instance，不创建完整模型的数万 Python module 或 parameter 对象。完整 meta tree 在 M0–M3.6 默认禁用，也没有诊断性 CLI 开关。只允许项目自有 Tiny 语义代表 Block 使用 FakeTensor/meta 验证：

- module class 与端口；
- 参数和 buffer shape；
- shared/tied parameter；
- 语义类别代表块的 op/Tensor contract。

仓库保留预算隔离 helper 及超时/失败回退测试，供未来可选 probe 使用；当前 CLI 不暴露完整 meta 构造，也不在 manifest 声称执行了该 probe。代表块构造或 export 失败时：

1. 立即停止该验证路径；
2. 保留已生成的 config/template IR；
3. 降低 capability/coverage，而不是令整个分析失败；
4. 产生含原因、阶段和 remediation 的 Diagnostic。

代表性 meta Block/export 不能用于真实性能结论，缩小模型也不能替代完整模型 runtime benchmark。Adapter 或 capture backend 必须拒绝真实权重输入、完整模型目标和不带 FakeTensor/meta 的 M2 默认 capture。

## 7. Folding 与稳定身份

重复识别优先级：

1. config 中的 `layer_types`、interval 或显式层表；
2. module scope 和 class；
3. op 类型、端口、符号 shape、局部拓扑与 state 副作用构成的结构哈希。

具有不同 state、routing、副作用或符号契约的实例不得折叠。Adapter 为每个 Definition 提供 structural signature，为每个 Instance 保留 `module_path` 和 `layer_index`。稳定 ID 规则遵循 [Model Map IR](model-map-ir.md#5-稳定-id)，不得使用随机 UUID 破坏 diff 与缓存。

## 8. 首批模型验收契约

### 8.1 Qwen3.8-27B

- 官方仓库：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- 固定 revision：[1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- 固定 config：[config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
- config SHA-256：`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`

M1 golden 必须识别：

- 顶层 `model_type=qwen3_5` 和 `Qwen3_5ForConditionalGeneration` 入口；
- Vision、Text、Projector/merge 级别的多模态宏观结构，不执行完整多模态 forward；
- 64 个文本层；
- `linear_attention, linear_attention, linear_attention, full_attention` 的 4 层周期，共 16 个周期；
- Full Attention 层维护 KV cache，Linear Attention/Gated DeltaNet 层维护 recurrent state，不能统一标成 KV；
- 1 个 MTP hidden layer，作为 conditional path；
- 代表性 Linear 和 Full Attention Definition 及 64 个实例映射。

Qwen config adapter 在 M1 不承诺 Linear Attention 内部真实 op、custom op 或 fusion。M3.6 可按冻结 config/公式规则递归投影 Full Attention 与 Dense Gated FFN 的 semantic primitives，并把 KV/recurrent state 接到对应 child 算子；Full Attention 显式保留 Q/K/V 的 reshape/transpose、GQA KV-head broadcast 和 context transpose，`attn_output_gate` 必须是布尔值，关闭时不得保留门控分支或双倍 Q projection 成本。这仍不是一次目标模型 forward。Linear Attention 的 Gated Delta rule core 在没有更强证据时保持 opaque，不能为追求“展开到底”而编造其内部顺序。

### 8.2 GLM-5.3-BF16

- 官方仓库：[zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)
- 固定 revision：[304b8051cfb2b260b61ce0cbe330e02a98e73639](https://huggingface.co/zai-org/GLM-5.3-BF16/tree/304b8051cfb2b260b61ce0cbe330e02a98e73639)
- 固定 config：[config.json](https://huggingface.co/zai-org/GLM-5.3-BF16/blob/304b8051cfb2b260b61ce0cbe330e02a98e73639/config.json)
- config SHA-256：`ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca`

M1 golden 必须识别：

- `model_type=glm_moe_dsa` 和 `GlmMoeDsaForCausalLM` 入口；
- 78 个层实例；
- 前 3 层 Dense，后 75 层 MoE；
- 256 个 routed experts、top-8 和 1 个 shared expert；
- 一个虚拟 Expert Pool Definition，而不是默认展开 256 份完整子图；
- resident/active 参数口径分离，缺少公式输入的成本保持 unknown。

M3.6 可显示静态 `Router GEMM → TopK → virtual Expert Pool/Shared Expert → Combine`，以及 symbolic selected-expert FFN template；Router scores 与 TopK routing weights 保留 config 声明的 `moe_router_dtype=float32`，weights 显式连接到 Combine。必须保留 `runtime_route_known=false`、`routing_weight_values_known=false`、`experts_materialized=0`，不产生实际 expert IDs、权重值、tokens/expert 或 route histogram。DSA 内部、动态 MoE route、grouped-GEMM padding 和 executed-cost 仍保持 opaque/Unknown 并延后到 M5。

### 8.3 Tiny Dense

Tiny Dense fixture 是 CI 和公式的独立 baseline。它应覆盖 embedding、重复 Decoder Block、Norm、GQA/MHA、FFN、KV cache、LM Head 和可选 weight tying。fixture 必须足够小、完全离线且不依赖 remote code 或下载权重。

## 9. M2 capture adapter 边界

M2 capture backend 不是完整模型执行器。它只接受：

- Tiny fixture；或
- 项目自有、与目标类别只具语义同类关系的 Tiny 代表 Block；
- FakeTensor/meta 输入和显式 symbolic constraint；
- 不包含真实模型权重的 module/ExportedProgram。

输出 manifest 记录原始模型 revision、代表 Block 类型、`capture_backend=torch.export.strict`、tensor mode 与 PyTorch 版本，并固定声明 `architecture_equivalence=semantic-kind-only`、`valid_for_full_model_performance=false`。捕获失败时保留 config IR，写入 opaque node 与 Diagnostic。

禁止将 27B/GLM 完整模型、完整多模态 forward、真实 MoE 路由或真实权重 capture 作为 M2 默认、验收或 fallback 路径。

## 10. Runtime importer 边界

M4 adapter 只读取用户在外部受控环境生成的 PyTorch Profiler、rocprofv3 或 ROCm Compute Profiler artifact。Importer 不启动模型、不执行 backend compile、不调用 profiler、不发起 counter replay。

若没有 trace：

- `trace_events=[]`；
- `runtime.measured_latency`、`runtime.kernel_dispatch_count`、`runtime.measured_hbm_traffic` 使用 `origin=unknown, value=null`；
- coverage 为 0、`coverage_status=unmapped`；
- 报告显示 `RUNTIME_TRACE_MISSING`，不能用 static/formula 指标填充 runtime 字段。

有 trace 时也必须保存原始 source artifact、run/tool/version、Scenario 和采集条件。M4a 在 AMD reference stack 冻结前仍被阻塞，详见 [DR-0006](decisions/DR-0006-amd-reference-stack-blocks-m4a.md)。

## 11. Diagnostic 最小集合

当前 artifact 会按可达路径使用以下代码；实现可以扩展，但不能只返回自由文本：

| Code | 触发条件 | 恢复策略 |
|---|---|---|
| `CONFIG_FIRST_NO_EXECUTION` | config-first 分析 | 声明零权重/零 forward |
| `REMOTE_CODE_EXECUTION_DISABLED` | `auto_map` 或实现需要 remote code | 静态检查，停留 C0/adapter fallback |
| `FULL_META_TREE_DISABLED` | config-first artifact | 声明完整 meta tree 未启用 |
| `LOGICAL_OPS_NOT_CAPTURED` | M1 config-only artifact | 提示 M2 capture |
| `CAPTURE_SCOPE_REPRESENTATIVE_ONLY` | M2 capture 成功 | 声明只来自 Tiny 语义代表块 |
| `CAPTURE_*` | PyTorch 不可用、fixture/export/转换失败 | opaque/unavailable，不回退完整模型 |
| `CAPTURE_DEFERRED_TO_M5` | GLM 请求 capture | 保留 config-first IR |
| `CAPTURE_KIND_NOT_APPLICABLE` | 请求的 Tiny kind 与 adapter 无匹配区域 | 不执行 capture，保留 config-first IR |
| `RUNTIME_TRACE_MISSING` | 未导入外部 trace | Runtime Metric 全部 Unknown/null |

`--local-files-only` 缓存缺失发生在 artifact 创建前，CLI 以带 `OFFLINE_CACHE_MISS` 前缀的 `ResolutionError` 退出，因此它不是 artifact 内 Diagnostic。未知成本主要由 `Metric(value=null, origin=unknown, coverage_status=partial|unmapped)` 表达，不额外虚构 `METRIC_UNKNOWN` 节点。

## 12. 离线与可复现性

`--local-files-only` 下 adapter 不得进行 DNS、HTTP、git fetch 或包安装。所有 source URI 仍保留官方位置，但报告使用本地复制的派生 metadata，且 HTML/Markdown 不需要访问这些链接才能显示核心内容。

当前 `analysis_id` 至少包含完整模型 revision/config hash、schema version、adapter name、工具版本和 cost-model version；工具版本承担当前内置 adapter 实现版本的身份，不另设独立 `adapter_version` 字段。相同输入和版本必须生成相同层级、artifact-local ID、canonical key 和静态成本。报告只保存必要的 config 派生事实、哈希和链接，不重分发模型权重或受限制源码。manifest 的 `safety` 还要固定 `weights_loaded=false`、`full_model_constructed=false`、`full_forward_executed=false`、`full_meta_tree_enabled=false`，以及存在时的代表 capture/trace 来源。

未提供 `--output` 时，CLI 只在确认当前目录处于 LLM Vis Git checkout 后使用仓库根的 `artifacts/generated/`；checkout 必须同时具有 Git marker（`.git` 目录或 worktree 文件）与 LLM Vis 项目身份，避免误认只有相似源码布局的无关目录。单次目录名为 `<model-slug>-<config-hash8>`；若已存在则分配 `-2`、`-3` 等后缀，禁止静默覆盖。仓库外省略 `--output` 必须返回可操作错误且不创建默认目录。显式 `--output` 仍完全由调用者控制，可在仓库内外使用，不受该默认路径策略影响。

## 13. 测试要求

每个 adapter 需要：

- 固定 revision 的 config fixture；
- 人工可审查的 L0/L1/operator recursive GraphView golden；
- Definition/Instance 数量、周期和异常层测试；
- config 可证明的参数/state 与 GLM resident/active 测试；
- deterministic ID 测试；
- remote code 禁用和 offline cache miss 测试；
- Tiny 代表块 meta/FakeTensor 成功、预算 helper 与失败 fallback 测试；
- 权重加载 API 不被调用、完整模型没有 fallback 的测试；
- 无 trace 时 Runtime Unknown/null 测试；
- M4 runtime importer 实现后补充“不启动模型/profiler”的测试；
- Unknown 不聚合为 0 的测试；
- 输出 IR 的 schema validation。

M3.9 输入层另需要：

- 五类 input kind 的等价 config/hash 正测，以及 model-page/blob/resolve/revision 规范化测试；
- branch/tag/`main` 固定 immutable commit 和 requested/resolved provenance 测试；
- 未知 `model_type`、缺失必需字段、非模型 JSON 与 known adapter 构建失败的分类降级测试；
- 非 HF URL、redirect/final-host 偏离、HTML、超过 2 MiB、深度 64、超时和超项数输入的拒绝测试；
- Qwen/GLM 不同 input kind 生成同一核心 Model Map/GraphView 身份与零权重/代码/forward 回归；
- 默认 repo-local output、Git/project 身份校验、冲突后缀、仓库外可操作错误、显式 `--output` 优先、auto-open、`--no-open`、打开失败保留 artifact、路径/token 脱敏与 Source/Evidence status 页面测试。当前页面测试是静态/生成报告自动化，真实 `file://` 打开仍是手工退出项。

新增 adapter 不得通过修改核心 IR 含义来“适配”单一模型。确有通用字段缺口时，先提出 schema migration 和 Decision Record，再更新 adapter。

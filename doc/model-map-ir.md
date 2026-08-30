# Model Map IR v0.1

状态：M0 语义与 wire-format 基线  
日期：2026-08-29  
适用范围：M0–M3 事实层；M3.5 的 `graph-view.json` 是其渲染中立投影；M4a/M4b 只导入外部 trace

Model Map IR 是 LLM-Vis 的框架中立事实层。它将模型定义、实例、Tensor 契约、语义节点、捕获到的逻辑算子、成本和导入的运行时证据放在同一个可版本化 artifact 中，同时保留未知与覆盖缺口。

本文件解释 [`schemas/model-map.schema.json`](../schemas/model-map.schema.json) 的语义。Schema 是 v0.1 的机器可执行 wire contract；分析 ID、工具版本、能力等级、执行策略和更丰富的 provenance 放在同目录的 `manifest.json`。任何不兼容字段改动都需要 schema migration 和 Decision Record，不能由单个 adapter 私自扩展。

## 1. 不变量

1. Definition 与 Instance 分离。重复 64 次的 Block 只有一个或少量 Definition，但保留每个 Instance 与例外覆盖。
2. Shape 使用可验证的符号 AST，不是只供展示的字符串。
3. Edge 明确区分 `data`、`weight`、`state_read`、`state_write`、`control` 和 `route`。
4. SemanticNode、LogicalOp、后端实现与 Kernel 之间允许多对多关系。
5. 结构身份与 Scenario 身份分离；改变 batch/context 不改变节点结构 ID。
6. Metric 携带单位、来源、Scenario（适用时）、公式/假设和覆盖状态。
7. `origin=unknown` 时 `value=null`；Unknown 绝不能编码或聚合成 `0`。
8. 无法解析的区域保留 opaque 对象和 Diagnostic；已有可靠信息不能因局部失败被丢弃。
9. v0.1 通过 `SourceArtifact`、`SemanticNode.evidence`、Metric 和 Diagnostic 回溯事实；完整运行上下文保存在 manifest。
10. Model Explorer 等前端只是消费者，不能把其内部 ID/schema 写成核心 IR 身份。
11. M0–M3.5 不读取权重、不运行完整模型 forward，完整 meta tree 默认禁用。
12. M4 只导入外部 trace；没有 trace 时所有 Runtime Metric 必须为 Unknown/null。
13. `graph-view.json` 是 `model-map.json` 的只读显示投影，不能成为新的事实来源，也不能把前端布局、搜索状态或选中状态写回 Model Map IR。
14. M3.5 L0/L1 只是 config-first 语义 DAG；生成或浏览该 DAG 不读取权重、不构造目标模型、不执行目标模型 forward。

执行与资源边界由 [DR-0007](decisions/DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md) 冻结。

## 2. L0–L4 与 M1/M2 边界

| 层级 | v0.1 对象 | 最早可信来源 |
|---|---|---|
| L0 定义层 | `Model`、`SourceArtifact`、`Definition`、`Instance` | config/source inventory |
| L1 语义层 | `SemanticNode`、state/route edge、Tensor contract | config adapter；可选缩小代表块验证 |
| L2 逻辑算子层 | `LogicalOp`、捕获 Tensor dataflow | M2 Tiny/同构缩小代表块的 FakeTensor/meta `torch.export` |
| L3 后端层 | `Lowering` 与 backend-domain op（预留） | 导入的 backend IR/编译 metadata |
| L4 运行层 | `TraceEvent`、measured Metric | M4 导入的 profiler/rocprof/counter artifact |

M1 可以在 UI 中提供“L2 语义投影”，例如依据 Qwen config 展示 Full Attention 预期组件。但它仍然是 `SemanticNode`，并遵守：

- `evidence` 指向 config JSON Pointer 和 adapter rule；
- `logical_ops=[]`；
- Diagnostic 包含 `LOGICAL_OPS_NOT_CAPTURED`；
- manifest 记录 `capability.level=C1`、`representative_block_captured=false` 与 `safety.*`；
- 不编造 `aten.*`、custom op、op ordinal、真实 Tensor dataflow、执行顺序、源码行号、fusion 或 Kernel。

M2 只有在受限代表块 export 成功后才写入 `LogicalOp`，并以 `Lowering(kind=expand|decompose)` 关联 SemanticNode。它不授权完整模型 capture。config 推导的 state/参数可以是 exact；workload FLOPs/bytes 只有公式与 Scenario 完整时才是 formula，否则仍为 unknown。

M3.5 不在 L1 和 L2 之间新增事实层。它把 L0/L1 已知对象投影为适合下钻的 GraphView，并且必须显示 `CONFIG SEMANTIC PROJECTION`。Qwen/GLM 的 activation `data`、MoE `route` 和 `control` 边是 adapter 声明的可解释语义契约，不是从目标模型 forward 观测到的 Tensor dataflow。L2 仍只在项目自有 Tiny 代表块成功 capture 时展示真实 `LogicalOp`；GraphView 不得为目标模型补造 L2。

## 3. 顶层 wire format

`model-map.json` 的 v0.1 顶层字段为：

```text
schema_version = "0.1"
model
source_artifacts[]
symbols[]
scenarios[]
definitions[]
instances[]
tensors[]
edges[]
semantic_nodes[]
logical_ops[]
lowerings[]
metrics[]
trace_events[]
diagnostics[]
```

除 `schema_version` 和 `model` 外的集合可以为空。Schema 禁止未知字段；跨阶段的补充信息放入 manifest，而不是利用未声明字段绕过校验。

M0–M3.5 当前 manifest 记录 analysis ID、LLM-Vis 版本、adapter 名称/metadata、模型 revision/config hash、输入 URI 与 cache/remote 状态、C0–C5 能力等级、`safety.*`、成本模型版本、显式 HardwareProfile、capture 摘要和产物清单。未来 M4 importer 再加入 trace source/run 上下文；当前不伪造尚不存在的 trace 或依赖清单字段。

| 能力 | 已获得证据 |
|---|---|
| C0 | config、源码索引或参数 inventory |
| C1 | config-first 宏观结构经过允许的结构验证 |
| C2 | 至少一种受限代表 Block capture |
| C3 | 指定缩小 fixture/workload 的完整逻辑路径；不等于完整真实模型 forward |
| C4 | 导入 trace 后成功关联 runtime op 与 Kernel |
| C5 | 导入并关联 hardware counter |

能力描述实际证据，不随目标里程碑自动提升。

### 3.1 M3.5 GraphView 产物边界

`graph-view.json` 与 `model-map.json` 同目录产生，并通过 `source_model_map_id` 关联唯一事实层。该 ID 不是 `Model.id`：producer 必须对完整 canonical Model Map 内容确定性计算 `mmap_*`，并在 evidence 中分别记录 `model:art_*` 与 `model_map:mmap_*`。它的机器契约由独立的 `schemas/graph-view.schema.json` 定义，不扩大或污染 Model Map v0.1 schema。它保存跨渲染器一致的语义视图和导航关系，不保存 SVG/canvas 坐标、缩放倍率、搜索词或选中态。自包含 HTML 嵌入与独立产物同源的 GraphView，不另造一套节点身份。该 schema、Qwen/GLM golden 与浏览器退出项已在 M3.5 验收通过。

```text
GraphViewDocument
  schema_version, id, model_id, revision, adapter_name
  source_model_map_id, provenance, views[]

GraphView
  id, key, level=L0|L1, label, parent_view_id, breadcrumb[]
  layer_index, root_group_id, nodes[], ports[], edges[], groups[], metadata

GraphNode
  id, key, label, kind, group_id
  input_port_ids[], output_port_ids[], subject_ids[], evidence[]
  coverage, opaque, drilldown_view_id, attributes

GraphPort
  id, key, node_id, name, direction=input|output, role
  shape, shape_known, dtype, tensor_spec_id, evidence[]

GraphEdge
  id, key, source_port_id, target_port_id
  kind=data|state_read|state_write|route|control
  tensor_spec_id, origin, evidence[], coverage

GraphGroup
  id, key, label, kind, member_node_ids[], parent_group_id
  collapsed, attributes
```

关键约束：

- `GraphNode.subject_ids[]` 只引用 Model Map 中已有的 Model/Definition/Instance/SemanticNode/LogicalOp 等 subject，供 Inspector 反查 Metric、Diagnostic 和 provenance；GraphNode 本身不复制或升格这些事实。
- 每个 `GraphPort` 只属于一个节点且方向明确。当 `tensor_spec_id` 存在时，必须指向 Model Map 的 TensorSpec；config 无法证明 shape/dtype 时保留 `shape_known=false`/`dtype=unknown`。
- 每个 `GraphEdge` 必须从 output port 指向 input port；每个 input port 至多有一个 producer，除非未来 schema 显式增加 merge 语义。绑定 TensorSpec 时，边与两端 port 的 `tensor_spec_id` 必须一致。`origin`、`evidence` 和 `coverage` 说明这条可见线为何成立。
- GraphView 默认不投影 `weight` 边。`data|route|control` 子图必须可拓扑排序；KV/recurrent state 使用 `state_read|state_write` 连接独立 state rail，不把跨 token 反馈回边混入单步主 DAG。
- L0 的每个模型 output 必须能沿非 control 数据路径从输入到达。MTP 不得只由伪造 bool 条件产生 logits；当前合同是 `decoder hidden → MTP → MTP Draft Logits`，主 LM Head Logits 与 Draft Logits 使用不同 input port/output boundary。
- L0 的折叠 Decoder/Dense/Sparse pattern 可用 `drilldown_view_id` 指向 L1；L1 用 `parent_view_id` 和 `breadcrumb` 返回上层，`layer_index` 支持 Layer Strip 定位。`GraphGroup.collapsed` 描述应折叠的 pattern/Expert Pool，不要在 L0 物化 64/78 层或 256 专家。
- 搜索、上下游高亮和 Inspector 联动使用上述稳定 `id/key`、`subject_ids`、`tensor_spec_id` 和 `evidence`；交互临时状态不进入 artifact，Scenario 切换也不改变这些结构 ID。

`GraphViewDocument.provenance` 必须声明 `projection=config-first-graph-view`、投影版本、源 artifact/evidence，并将 `weights_loaded`、`target_model_constructed`、`target_model_forward`、`remote_code_executed` 全部固定为 `false`。这一层不能通过“只是渲染”绕过 manifest 的零执行安全声明。

## 4. 核心对象

### 4.1 Model 与 SourceArtifact

```text
Model
  id, name, family, revision, framework, source_artifact_ids[]

SourceArtifact
  id, kind, uri, path, revision, sha256, license, trusted
```

Hugging Face `revision` 必须解析为完整 commit SHA，不能只保存 `main` 或短 SHA。远程源码即使固定哈希也使用 `trusted=false`；MVP 的 `inspect_only` 执行策略记录在 manifest。`uri` 是 provenance 链接，不表示离线打开报告时必须访问网络。

### 4.2 SymbolicExpression 与 Symbol

`Symbol` 记录 `id`、`name`、`description`、`unit`、上下界和默认值。常用符号包括 `B`、`T`、`L`、`S_KV`、`H`、`N_Q`、`N_KV` 和 `D_H`。

表达式使用带 `op` 的 AST：

```json
{
  "op": "multiply",
  "args": [
    {"op": "symbol", "symbol": "B"},
    {
      "op": "add",
      "args": [
        {"op": "symbol", "symbol": "L"},
        {"op": "symbol", "symbol": "T"}
      ]
    },
    {"op": "symbol", "symbol": "H"}
  ]
}
```

允许的 op 由 Schema 枚举。`concrete_shape` 可来自 config 中全部为字面量的可证明 shape，也可来自绑定后的 Scenario 或 FakeTensor/meta capture；一次 decode 的 `T=1` 不能覆盖通用 symbolic contract。config 无法证明 layout 时使用 `shape_known=false`，不得补猜测维度。

### 4.3 Definition 与 Instance

```text
Definition
  id, kind, label, class_name, ports[], children[], semantic_pattern

Instance
  id, definition_id, parent_id, module_path, layer_index, overrides
```

Definition 描述复用结构，Instance 描述具体位置。`overrides` 只保存实例差异。具有不同 state、routing、副作用或 Tensor contract 的实例不得仅因 class 相同而折叠。

虚拟 `expert_pool` 是合法 Definition：它保留专家总数、top-k、shared expert 和单专家模板，不要求展开数百份完整子树。

### 4.4 TensorSpec 与 Edge

顶层集合名为 `tensors`，每个元素是 TensorSpec：

```text
TensorSpec
  id, semantic_name, owner_id, symbolic_shape, concrete_shape, shape_known
  dtype, storage_dtype, layout, stride, device, role, alias_group
  origin, materialized, source_artifact_ids[], evidence[]

Edge
  id, source_id, target_id, tensor_id, kind
```

logical dtype、storage dtype、compute dtype 和 accumulator dtype 是不同概念。v0.1 直接保存前两者；后两者在 metric assumptions/manifest 中记录，直到 schema 扩展。M1 inventory Tensor 固定 `origin=config`、`materialized=false`，`semantic_name` 不冒充真实 `named_parameters()` 名称；M1 Edge 是 adapter 声明的 weight/state 契约，不得称为一次真实 forward 的 dataflow。

### 4.5 SemanticNode 与 LogicalOp

```text
SemanticNode
  id, kind, label, definition_id, instance_ids[]
  input_tensor_ids[], output_tensor_ids[], child_ids[]
  confidence, evidence[], opaque

LogicalOp
  id, domain, op_type, inputs[], outputs[], attrs, scope_instance_id

M2 的代表体捕获对象必须可独立追溯：每个 FakeTensor 元数据使用
`origin=capture`、`materialized=false`，且只绑定本次捕获的一个
`source_artifact_id`；每个 LogicalOp 的 `attrs` 同步记录 `capture_id`、
捕获后端、tensor mode 与同一 source artifact，并将 `scope_instance_id`
指向被代表的 decoder layer。这里的 capture 来源始终是项目自有 Tiny
fixture，不是目标模型实例。

Tiny capture SourceArtifact 的 `sha256` 是规范化捕获内容（capture contract、
TensorSpec、LogicalOp 与 SemanticNode）的确定性摘要，不是 fixture 版本字符串的
简单摘要，更不是目标模型权重摘要。
```

`SemanticNode.kind` 可表示 Full/Linear Attention、Norm、FFN、Router、Expert Pool、KV cache、recurrent state、Vision Tower、MTP 等。M1 使用 `evidence` 保存如 `config:/text_config/layer_types/3` 和 `adapter:qwen3_5_config:<rule>` 的证据。

LogicalOp 只来自 M2 capture；domain 可为 `aten`、`prims`、`torch`、`custom`、`python`、`backend` 或 `opaque`。不支持的 custom op 可保留为 opaque LogicalOp，并附 Diagnostic。

语义识别可信度按官方/项目 adapter、config/model class、module 属性、source symbol、op pattern 依次下降。LLM 解释只能作为候选文案，不能把未证实事实写成高置信度结构。

### 4.6 Lowering

```text
Lowering
  id, source_ids[], target_ids[]
  kind = expand | decompose | fuse | split | copy | fallback
```

Lowering 支持 N:M。没有导入 backend IR/trace 时 L2→L3/L4 未知，不能由 ATen 名称猜 fusion 或 Kernel。融合 Kernel 的 exclusive time 只归 fusion group；参与融合的语义节点只做 shared/inclusive attribution，不能重复累计。

### 4.7 Scenario

```text
Scenario
  id, phase, batch, new_tokens, past_tokens, total_visible_length
  output_length, activation_dtype, weight_format, kv_dtype
  backend, hardware
```

统一口径：

```text
T = new_tokens
L = past_tokens
S_KV = total_visible_length = L + T
```

`prefill` 要求 `T>1, L=0`；`chunked_prefill` 要求 `T>1, L>0`；`decode` 要求 `T=1`。Scenario ID 从 canonical payload 确定性生成。Ragged/paged batch 若没有逐序列信息，相关指标保持 estimated/unknown。

### 4.8 Metric 与 Coverage

```text
Metric
  id, subject_id, scenario_id, name, value, unit
  origin, formula, assumptions[], confidence
  coverage, coverage_status

origin = exact | formula | estimated | measured | inferred | unknown
coverage_status = complete | partial | opaque | unmapped | failed | unknown
```

`coverage` 是 `[0,1]` 数值。聚合规则：

- 只聚合同一 Scenario、同一单位和同一 metric 语义；
- unknown 不进入数值求和，但进入覆盖缺口；
- 部分覆盖仍标记 `partial`，同时展示已覆盖值和 coverage；
- estimated、formula 和 measured 不在没有显式规则时混成一个值；
- useful/executed MoE FLOPs、occupied/allocated KV、logical/measured HBM bytes 分开保存；
- 没有导入 trace 时，三个占位 Runtime Metric 均为 `origin=unknown, value=null, coverage=0, coverage_status=unmapped`。

M0–M3 的占位名是 `runtime.measured_latency`、`runtime.kernel_dispatch_count`、`runtime.measured_hbm_traffic`。静态 FLOPs、logical bytes 或 roofline 不能填入这些 Runtime 字段，也不能产生 Kernel/counter 实测结论。

### 4.9 TraceEvent 与 Diagnostic

```text
TraceEvent
  id, parent_id, correlation_id, category, start, duration
  device, stream, kernel_name, counters

Diagnostic
  id, severity, code, subject_id, message, evidence[]
```

TraceEvent 只由 M4 importer 从外部 artifact 产生。run/dispatch 原始身份、工具版本和采集条件保存在 source artifact 与 manifest；`correlation_id` 只在对应 run 内有效。无法归属的事件进入 Unattributed 分组并使用 `unmapped` coverage，不允许静默丢弃。

Diagnostic 的 stage、是否可恢复和 remediation 在 v0.1 编码于标准化 `code`、`message` 与 `evidence`；未来若升级为独立字段需要 schema migration。

## 5. 稳定 ID

IR 使用两类稳定身份：

- artifact-local ID：在固定 artifact 内唯一，输入包含模型 revision、framework domain、module path、op ordinal（存在时）和 capture variant；
- canonical semantic key：不包含 revision，输入包含规范化 semantic path、Definition signature、符号契约和 structural signature，用于跨 artifact 匹配。

v0.1 wire 使用对象 `id` 作为 artifact-local 身份，并把 `canonical_semantic_key` 保存在 Instance `overrides`。生成器采用版本化 canonical JSON 与确定性哈希，禁止随机 UUID 作为唯一长期标识。相同输入、adapter 和 schema/cost 版本必须生成相同 ID。

M0–M3 的 `diff` 只比较同一 Model Map artifact 内的两个 Scenario。跨 revision diff 延后 M5；届时才能评估是否先按 canonical key 精确匹配、再用结构相似度处理 rename/move，不能把这套候选算法描述为当前已交付行为。

## 6. Provenance

v0.1 的 provenance 由下列部分组合：

- `SourceArtifact`：URI/path、完整 revision、SHA-256、license 和 trusted；
- `SemanticNode.evidence`：config JSON Pointer、adapter rule、source symbol 或 capture rule；
- `Metric.origin/formula/assumptions/confidence/coverage`：数值真实性与覆盖；
- `Diagnostic.evidence`：失败、fallback 和缺口；
- `manifest.json`：工具版本、执行边界、capability、成本/HardwareProfile 与 capture 摘要；trace run 上下文属于 M4。

Metric 的 `origin` 和事实来源不是同一概念。例如固定 config 的层数可以是 exact 且 evidence 指向 config；Attention FLOPs 可以是 formula 且 assumptions 指向 Scenario。

## 7. M1 config-only 示例

以下是可被 v0.1 Schema 接受的最小边界示例：

```json
{
  "schema_version": "0.1",
  "model": {
    "id": "model:qwen38-27b:1d4bf0f",
    "name": "Qwen/Qwen3.8-27B",
    "family": "qwen3_5",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "framework": "transformers",
    "source_artifact_ids": ["source:qwen-config"]
  },
  "source_artifacts": [
    {
      "id": "source:qwen-config",
      "kind": "huggingface_config",
      "uri": "https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json",
      "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
      "trusted": false
    }
  ],
  "definitions": [
    {
      "id": "def:full-attention-block",
      "kind": "block",
      "label": "Full Attention Block",
      "semantic_pattern": "full_attention"
    }
  ],
  "instances": [
    {
      "id": "instance:layer-3",
      "definition_id": "def:full-attention-block",
      "module_path": "model.layers.3",
      "layer_index": 3
    }
  ],
  "semantic_nodes": [
    {
      "id": "semantic:layer-3-attention",
      "kind": "full_attention",
      "label": "Full Attention",
      "definition_id": "def:full-attention-block",
      "instance_ids": ["instance:layer-3"],
      "confidence": 1.0,
      "evidence": [
        "config:/text_config/layer_types/3",
        "adapter:qwen3_5_config"
      ],
      "opaque": false
    }
  ],
  "logical_ops": [],
  "metrics": [
    {
      "id": "metric:model-runtime-latency",
      "subject_id": "model:qwen38-27b:1d4bf0f",
      "name": "runtime.measured_latency",
      "value": null,
      "unit": "seconds",
      "origin": "unknown",
      "assumptions": ["no trace imported"],
      "coverage": 0.0,
      "coverage_status": "unmapped"
    }
  ],
  "trace_events": [],
  "diagnostics": [
    {
      "id": "diagnostic:logical-ops-not-captured",
      "severity": "info",
      "code": "LOGICAL_OPS_NOT_CAPTURED",
      "message": "Config-only M1 artifact; no LogicalOp has been captured."
    },
    {
      "id": "diagnostic:runtime-trace-missing",
      "severity": "info",
      "code": "RUNTIME_TRACE_MISSING",
      "message": "No external runtime trace was imported; Runtime metrics remain Unknown."
    }
  ]
}
```

允许的缩小 meta/FakeTensor 验证可以提高 manifest capability，但不会自动生成 LogicalOp；只有 M2 export 结果才会填充 `logical_ops`。

## 8. 执行与 Runtime 边界

- M0–M3.5 Model Map/GraphView 不引用或读取模型权重 artifact。
- 完整 meta tree 默认禁用；其缺失不是 C0/M1 失败。
- M2 export 的持久化 `SourceArtifact.kind=generated`，URI 指向项目自有 Tiny representative；manifest/captures 标明 `capture_backend=torch.export.strict`、meta/FakeTensor tensor mode、PyTorch 版本、`architecture_equivalence=semantic-kind-only` 与 `valid_for_full_model_performance=false`。内存中的 `ExportedProgram` 不写入 artifact。
- M4 importer 是 trace_events/measured Metric 的唯一生产路径；LLM-Vis 不启动完整模型、compile、profiler 或 counter replay。
- `trace_events=[]` 时 runtime 缺失是正常状态，必须以 Unknown Metric/Diagnostic 呈现。
- M4a 仍被未冻结的 AMD reference stack 阻塞，见 [DR-0006](decisions/DR-0006-amd-reference-stack-blocks-m4a.md)。

## 9. 校验与迁移

Schema 与 producer-policy 检查至少覆盖：

- 所有引用 ID 存在且全局不复用；
- Definition/Instance 父子引用合法；
- Scenario 满足 `S_KV=L+T`；
- `origin=unknown` 时 `value=null`，非 unknown 时 value 为数值；
- config-only artifact 的 `logical_ops` 为空；
- unknown 不参与数值求和；
- remote source 使用 `trusted=false`，manifest `safety.remote_code_executed=false`；
- lowering source/target 合法且支持多对多；
- 稳定 ID 可确定性重算；
- 无 trace 时 Runtime Metric 为 Unknown/null；
- M0–M3.5 manifest 记录零权重、零完整 forward、完整 meta tree disabled；
- export artifact 记录项目自有 Tiny 代表块、`torch.export.strict`、PyTorch 版本与 FakeTensor/meta mode。
- GraphView 的内容寻址 `source_model_map_id=mmap_*`、Model Map subject/TensorSpec 引用、view parent/drilldown/group/port 引用全部有效；相同输入生成稳定的 document/view/node/port/edge/group ID。
- GraphView 中所有边从 output port 指向 input port，每个 input 至多一个 producer，`data|route|control` 主子图无环，L0 outputs 沿数据边可达，state rail 只用 `state_read|state_write`。
- GraphView provenance 与 manifest 一致声明未读取权重、未构造或 forward 目标模型。

v0.x 允许快速演进，但每次不兼容改动都需要新 schema version、迁移器、golden 更新和 DR。旧 artifact 必须可离线验证；无法迁移的内容保留原始 artifact 并产生 Diagnostic，不能静默丢弃。

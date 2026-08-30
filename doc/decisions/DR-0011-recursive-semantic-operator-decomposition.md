# DR-0011：递归语义算子分解与可见成本前沿

- 状态：Accepted
- 日期：2026-08-30
- 影响范围：M3.6 GraphView、Qwen/GLM adapter projection、成本归因、离线 DAG UI

## 背景

M3.5 已能把模型与代表 Block 显示为端口级 DAG，但 Attention、FFN、MoE Expert 等复合节点仍是黑盒。把用户分成“新手/专家模式”会产生两套图和两套事实；直接运行 27B/超大 MoE 模型又违反本项目的资源与证据边界。

## 决策

1. 产品只保留一份递归 DAG。复合语义节点在同一中央画布中打开局部 GraphView，breadcrumb 表示展开路径，`Collapse` 返回父图；递归深度不叫 L2，也不代表用户等级。
2. 冻结 semantic primitive ontology v1：GEMM、MatMul、RMSNorm/LayerNorm、Softmax/TopK、SiLU/GELU、Add/Multiply/Scale/Mask、Reshape/Transpose/Broadcast/Split/Concat、RoPE、Gather/Scatter/Reduce、Convolution、state read/write 与 Opaque。它不是 ATen、GPU Kernel 或融合边界。
3. 每个 decomposition view 必须声明被分解的父节点，并用显式 boundary binding 将父节点每个 input/output/state port 恰好映射到一个 child boundary port。role、dtype、shape、TensorSpec 与输入到输出的可达性必须保持。
4. 每个 primitive 可声明精确 Metric 名，或显式标为 `unknown`/`excluded` 并给出原因。每个 decomposition view 声明 parent metric、known child metrics、unattributed nodes、excluded nodes 与 complete/partial/unknown 状态。Unknown/partial 不得补零。
5. 热图只统计当前 GraphView 的 `cost_frontier_node_ids`。父 view 与已展开 child view 不同时进入分母、排名或合计，避免父子重复计算。
6. Qwen Full Attention 与 Dense Gated FFN 是首个完整参考实现；Full Attention 显式展示 layout reshape/transpose 与 GQA Broadcast，且 `attn_output_gate=false` 时删除门控分支并同步成本。Linear Attention 必须把 recurrent state read/write 接到对应算子核心。GLM sparse MoE 只展示 config 可证明的 Router GEMM、TopK、虚拟 Expert Pool、Shared Expert 与 Combine；Router/TopK 权重保留声明 dtype 和 shape，但不生成实际 selected expert、routing-weight value、route histogram 或 tokens-per-expert。
7. GLM DSA 和其他证据不足区域保持 opaque 且无展开入口。
8. `_PROJECTION_VERSION` 暂保持 `0.1`，避免仅因增加 child views 而改变既有 L0/L1 view/node/port ID；新 child view 使用独立稳定 key。GraphView 文档内容 ID 随新增内容变化是预期行为。
9. M3.6 继续只读取 config/已有 Model Map 与公式，不下载或读取权重，不构造目标模型，不执行完整模型 forward，也不把语义分解描述为实际调度、融合或 Kernel 图。

## 成本口径

- Qwen Full Attention 父 FLOPs 公式的已知范围严格分解为 Q/K/V/O GEMM、QKᵀ MatMul 与 P×V MatMul；Norm、RoPE、Scale、Mask、Softmax、gate activation/Multiply 仍显示，但明确在现有父公式范围外。
- Qwen FFN 父 FLOPs 与 logical bytes 分解为 Gate/Up/Down 三个 GEMM；SiLU 与 Multiply 明确在现有父公式范围外。
- Full Attention logical bytes 当前只精确归因 projection；core/state remainder 保持 partial/unattributed。
- GLM executed FLOPs/traffic 继续 Unknown；静态 top-k 只能用于结构与 active-parameter 口径，不能伪装 runtime route。

## 后果

- 用户从同一张 DAG 逐步看到更多细节，基础算子叶节点不再显示虚假下钻入口。
- GraphView schema/golden 会增加 boundary、frontier、reconciliation 与 primitive 字段，但既有节点身份保持稳定。
- 报告的 Compute/Memory/Pressure 热图在 primitive view 中可能出现灰色 Unknown/Not applicable；这是证据边界，不是缺陷。
- SVG 内嵌 compound group 可作为未来表现层优化，但不得改变本 DR 的 GraphView 与成本契约。

## 验收

M3.6 退出项使用 `OP-01`～`OP-10`，详见主计划、UI wireframe 与 measurement protocol。任何一项未通过时，M3.6 保持“实施中”。最终证据为 checked-in 结构子集 27/27 PASS、Python 3.9/3.12 各 192 passed，以及两份自包含报告的 OP-01～OP-10 真实交互与空 console。

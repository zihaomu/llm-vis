# DR-0004：MVP 范围与 M1/M2 的 L2 边界

- 状态：Accepted
- 日期：2026-08-29
- 决策者角色：Product / Model / Tech Lead
- 复审点：M2 入口评审

## 背景

规划中的 L2 指逻辑算子层，但 M1 是 config-first 静态纵切。若 M1 为了界面完整而把 config 中的架构类型标成 ATen/custom op，会把“模型应该有的语义组件”误报成“一次真实 forward 已执行的算子图”。

## 决策

公开 MVP 是 M0–M3：

- M0：IR、golden、决策和可行性 spike；
- M1：config-first L0/L1 和 L2 语义投影；
- M2：单一 `torch.export` 代表性 Block 捕获、真实 Logical Ops 和 opaque fallback；
- M3：prefill/decode Scenario、基础成本、roofline 和同模型 workload diff。

M1 的“L2”固定解释为 config adapter 推导的语义契约：

- 可以显示 Attention、Linear Attention、Norm、FFN、Router、Expert Pool 等预期组件；
- 证据必须指向 config 字段或已测试 adapter rule；
- `logical_ops` 保持空；
- 不生成 `aten.*`、custom op、op ordinal、真实数据依赖、fusion 或 Kernel；
- artifact 显示 `LOGICAL_OPS_NOT_CAPTURED`；
- 未知内容使用 opaque/unknown，不用猜测补齐。

M2 捕获代表性 Block 后才写入 `LogicalOp` 和 Tensor dataflow，并用 `Lowering` 连接语义节点。捕获边界同时受 [DR-0007](DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md) 约束。

## 模型深度边界

- Tiny Dense 与 Qwen 是 M0–M3 端到端纵切对象。
- Qwen 多模态在 MVP 只展示 Vision/Text/Projector 宏观结构，不执行完整多模态 forward。
- GLM-5.3 在 MVP 只做 config-level 超大 MoE 宏观验收；动态 DSA/MoE Block 和 executed-cost 延后 M5。
- MVP diff 只比较同一模型的 workload；跨模型/revision diff 延后 M5。
- TorchLens/hooks 不是 MVP 主后端；第二捕获 backend 延后 M5。

## 后果

- UI 必须区分“semantic projection”和“captured logical ops”。
- M1 达到 C0/C1 不意味着 C2/C3；meta 结构验证也不产生 Logical Ops。
- M2 捕获失败时保留 M1 结构，并以 opaque/Diagnostic 表示覆盖缺口。
- M4a 之前，产品是结构与理论成本工具，不能称为完整 runtime 性能诊断器。


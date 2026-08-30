# LLM-Vis Decision Records

本目录记录已经冻结的产品、数据和实施边界。Decision Record（DR）一旦标记为 Accepted，代码、测试、示例和报告必须遵守；需要改变时新建替代 DR，不直接改写历史理由。

| DR | 状态 | 决策摘要 |
|---|---|---|
| [DR-0001](DR-0001-local-tool-and-offline-reports.md) | Accepted | 本地工具 + 可分享的离线静态 HTML/Markdown/JSON；MVP 无托管服务 |
| [DR-0002](DR-0002-remote-code-disabled-in-mvp.md) | Accepted | MVP 禁止执行 Hugging Face remote code |
| [DR-0003](DR-0003-model-explorer-bounded-spike.md) | Accepted（路径）；Exit Partial | M0/M1 用独立 custom adapter 评估 Model Explorer；离线 adapter 已验证，真实 consumer/UI 退出项仍未完成 |
| [DR-0004](DR-0004-milestone-and-l2-boundary.md) | Accepted | MVP=M0–M3；M1 L2 仅 config 语义投影，真实 Logical Ops 在 M2 |
| [DR-0005](DR-0005-frozen-model-revisions.md) | Accepted | 冻结 Qwen3.8-27B 与 GLM-5.3-BF16 的 Hugging Face revision |
| [DR-0006](DR-0006-amd-reference-stack-blocks-m4a.md) | Blocked | AMD reference stack 未冻结，M4a 不得开始验收 |
| [DR-0007](DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md) | Accepted | M0–M3 零权重/零完整 forward；完整 meta tree 默认禁用；M4 只导入 trace |
| [DR-0008](DR-0008-m3-cost-int4-and-roofline-boundary.md) | Accepted | M3 成本口径、首种 INT4 存储格式与理论 roofline 边界 |
| [DR-0009](DR-0009-model-map-ir-v0x-internal.md) | Accepted | Model Map IR v0.x 可导出但仍为实验性内部规范，暂不承诺稳定第三方 ABI |
| [DR-0010](DR-0010-renderer-neutral-dag-canvas.md) | Accepted | M3.5 使用 renderer-neutral GraphView 与自包含 L0→L1 DAG 主画布；不运行目标模型、不伪造完整算子图 |

## M0 书面退出物

- [`ui-wireframe.md`](../ui-wireframe.md)：可核验的低保真界面与交互契约，覆盖 L0/L1/L2、Layer Strip、Inspector、Scenario、热点、Unknown 与 Coverage。
- [`measurement-protocol.md`](../measurement-protocol.md)：M0–M3 理论指标、M4 trace-import-only、timing/timeline/counter 分离及退出命令模板。
- [DR-0003 退出审计](DR-0003-model-explorer-bounded-spike.md#退出审计与证据)：当前结论为 Partial，并逐项记录已通过和缺失的证据。

## 尚未冻结的 Gate

以下项目仍需新的 DR，不能从现有文本中推断答案：

- 首个 AMD GPU/gfx、ROCm、PyTorch、Transformers、backend、profiler 和 workload 组合；
- M5 是否构建经审计的 remote-code sandbox；未通过审计时继续禁止执行；
- Model Explorer bounded spike 的真实 consumer 加载、完整 Instance 导航、shape/Coverage/Diagnostic overlay 及搜索/展开交互证据；补齐后才能把 Exit 从 Partial 改为 Pass，失败时再决定外围适配或自研主画布。

这些未决项不阻塞 M0–M3 的 config-first 静态 MVP，但 AMD reference stack 直接阻塞 M4a，见 [DR-0006](DR-0006-amd-reference-stack-blocks-m4a.md)。

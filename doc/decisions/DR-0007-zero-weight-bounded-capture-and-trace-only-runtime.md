# DR-0007：零权重、受限捕获与 trace-only Runtime

- 状态：Accepted
- 日期：2026-08-29
- 决策者角色：Tech / Model / Performance Lead
- 复审点：M5；任何放宽都需要新的安全与资源评审

## 背景

首批模型包含 27B 与超大 MoE。加载权重、构造完整 module tree 或运行完整 forward 会显著扩大下载、内存、设备、执行安全和可复现性范围，也不是验证 config-first IR 所必需。Runtime 数据应来自用户在受控环境外部采集的 trace，而不是由本地可视化工具擅自启动大模型。

## 决策

### M0–M3

- 默认且基线验收路径为**零权重**：不下载、不 mmap、不反序列化模型权重；
- 不执行完整模型 forward，包括完整文本、多模态或 MoE forward；
- 完整 meta module tree 默认禁用；M0–M3 的标准 CLI 不自动开启它；
- config adapter 直接生成惰性 Definition/Instance 和语义契约；
- `torch.export` 只允许 Tiny fixture 或同构缩小的代表性 Block，并使用 FakeTensor/meta 输入；
- 不允许以真实 27B/GLM 权重或完整模型 capture 作为 M2 的必需或默认路径；
- 代表性 Block 只验证结构、Tensor contract 和公式，不能作为完整模型 latency 或 runtime 真值。

完整 meta tree 即使技术上可构造，也不属于 M0–M3 默认支持面。未来的实验性 opt-in 必须另行记录预算、输入信任和失败策略，不得改变默认关闭状态。

### M4a/M4b

- Runtime 固定为 **trace-import-only**：LLM-Vis 读取外部生成的 PyTorch Profiler、rocprofv3 或 ROCm Compute Profiler artifact；
- LLM-Vis 不负责启动完整模型、采集生产 trace、执行 backend compile 或发起 hardware-counter replay；
- importer 必须保留原始 run/source 标识、版本、Scenario 和采集条件；
- timing、timeline 和 counter 来自独立外部运行，不能混用 wall time；
- 没有导入 trace 时，Runtime latency、Kernel、dispatch、copy/sync、带宽、occupancy、MFMA/VALU 和 counter 指标一律 `origin=unknown, value=null`；
- 静态 FLOPs、logical bytes 或 roofline 不能自动填充 Runtime 字段，也不能推断 Kernel/fusion。

## 后果

- M0–M3 可以在没有 GPU、权重和完整模型依赖的环境中离线完成。
- M2 capture 产物必须在 manifest 中标记 fixture/block、缩小方式、FakeTensor/meta 和“不可用于完整模型性能结论”。
- `inspect` 与静态 `report` 不隐式触发下载权重、完整 meta tree、forward 或 profiler。
- M4 importer 可以独立开发，但 [DR-0006](DR-0006-amd-reference-stack-blocks-m4a.md) 解除前不能宣称 AMD reference 验收通过。
- 报告的 Runtime 面板在无 trace 时显示 Unknown 和导入提示，而不是空表、0 ms 或估算 Kernel。

## 验证

自动化检查至少证明：

- M0–M3 默认路径没有权重文件读取和完整 forward 调用；
- 完整 meta tree 默认关闭；
- export 拒绝非 Tiny/缩小代表块或真实权重输入；
- 无 trace artifact 时 Runtime Metric 的值为 null 且 origin 为 unknown；
- importer 只读外部 artifact，不启动模型或 profiler；
- 报告清楚区分 static/formula 与 runtime/measured。


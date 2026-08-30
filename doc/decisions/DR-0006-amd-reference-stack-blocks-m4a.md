# DR-0006：AMD reference stack 未冻结，M4a 阻塞

- 状态：Blocked
- 日期：2026-08-29
- Owner 角色：Performance Lead
- 复审点：完整 stack tuple 被提议并可在目标机器验证时

## 背景

M4a 的 marker、correlation、dispatch、compile/fusion 和覆盖率都依赖具体 GPU、ROCm、PyTorch 和 backend。没有固定环境就无法定义可复现 trace、分母、兼容字段或验收命令。

## 当前结论

首个 AMD reference stack 尚未冻结，因此：

- M4a 处于阻塞状态；
- 不宣称 eager 95% 或 compile/fused 90% 的映射覆盖率已经可实现或通过；
- 不把任意开发机 profiler 输出当作 reference artifact；
- M4b hardware counter 继续依赖 M4a，不单独启动验收；
- M0–M3 的静态结构、理论成本和离线报告可以继续。

此外，runtime 的产品边界已由 [DR-0007](DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md) 冻结为 trace-import-only；reference stack 冻结后也不授权 LLM-Vis 启动完整模型 forward。

## 解锁所需 tuple

新的 Accepted DR 必须同时冻结：

1. AMD GPU 型号、显存容量和 gfx target；
2. driver 与 ROCm 完整版本；
3. PyTorch build/version；
4. Transformers 和必要 adapter 版本；
5. eager/compile 或明确推理 backend 及版本；
6. PyTorch Profiler、rocprofv3/rocprofiler-sdk 版本；
7. 固定模型或 tiny fixture revision；
8. prefill/decode workload、dtype、输入、随机种子和 warmup/repeat；
9. timing、timeline 和 counter 的独立采集协议；
10. reference machine、exit command、预期摘要和覆盖率分母。

GPU/gfx/ROCm/PyTorch/backend 任一项缺失都不能解除阻塞。

## 阻塞期间允许的工作

可以设计 trace schema、编写纯 importer fixture 和离线关联算法，但其状态必须标为未在 reference stack 验证。没有实际 trace 时所有 Runtime latency、dispatch、bandwidth、counter 和 Kernel 指标均为 Unknown，不能由静态成本推断。


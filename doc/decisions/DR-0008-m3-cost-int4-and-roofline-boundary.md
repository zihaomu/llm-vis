# DR-0008：M3 成本、INT4 与 roofline 边界

- 状态：Accepted
- 日期：2026-08-29
- 决策者角色：Model / Performance / Tech Lead
- 复审点：M4 导入实测 trace，或 M5 增加新的量化布局时

## 背景

M3 必须在不加载模型权重、不运行完整模型的前提下比较 prefill、chunked prefill 与 decode。为避免把理论 useful work、逻辑最小流量、物理权重存储和实测 GPU 流量混为一谈，需要冻结公式来源、首种量化格式、未知值与 roofline 的解释边界。

## 决策

### 成本口径

- Dense/GQA/Full Attention/SwiGLU/KV cache 使用 config 尺寸与 Scenario 的确定性公式；
- Qwen Linear Attention 的投影、卷积与 state 尺寸由 config 精确计算，Gated DeltaNet core useful-work FLOPs 与最小 state traffic 标记为 `estimated`；
- `logical_bytes` 是理想化的逻辑最小流量，不是 HBM/L2 counter，不模拟融合、cache、workspace、padding、调度或 allocator；
- 参数、FLOPs、logical bytes、KV/state 与 arithmetic intensity 均携带 formula、assumptions、origin 和 coverage；
- 聚合时 `Unknown` 绝不替换为 `0`。只有已知成员可形成部分和，并必须降低 coverage；全部成员未知时结果保持 `value=null`。

GLM-5.3 在 M3 仅计算 config 可证明的 decoder resident/active 参数和存储。DSA 与 executed MoE 的 FLOPs、logical bytes、KV/state、arithmetic intensity 保持 `Unknown`；没有 route histogram 时 top-k 只用于 active 参数上界口径，不能冒充真实路由分布。

### 冻结的首种量化格式

除 BF16/FP16 外，M3 只支持：

`int4-groupwise-symmetric-g128-fp16-scale-v1`

其物理存储规则为：

1. 每个二维权重 tensor 独立按 128 个值分组；尾组零填充到完整分组；
2. 每个值使用一个 4-bit nibble，两个值打包为一个字节；
3. 每组保存一个 2-byte FP16 scale；
4. 对称量化不保存 zero point；
5. Norm、bias 与其他明确未量化的 side parameter 按 Scenario activation dtype 单列；
6. artifact 必须分别保留 packed bytes、scale metadata、zero-point metadata、padding values 与非量化 bytes，不能只使用“参数量 ÷ 2”。

本格式只是静态存储约定，不声称与任一 runtime 的实际 kernel layout、对齐、分片或量化精度完全一致。

### Roofline

- LLM-Vis 不内置或猜测真实 GPU 峰值；HardwareProfile 必须由调用方显式提供并带 provenance；
- M3 只计算独立的 compute lower bound 与 bandwidth lower bound，以及两者的最大值；
- 结果固定标记为 `theoretical_lower_bounds_not_latency_estimate`；
- dtype 不匹配、峰值缺失、FLOPs/bytes 未知时保留 Unknown，不套用其他 dtype 或默认设备；
- 项目提供的 synthetic profile 只用于测试和可复现示例，不代表任何真实硬件。

## 后果

- M3 报告可以排序理论热点，但不能声称预测端到端 latency、Kernel 数、HBM 实测流量或 AMD 利用率；
- 同模型 workload diff 比较完整 Metric 事实，包括 formula、origin、assumptions 与 coverage；未知项没有数值 delta；
- M4 导入 trace 后，measured runtime 指标使用独立命名与来源，不覆盖 M3 静态指标；
- 新量化格式、非对称 zero point、双量化或 backend-specific packing 需要新的格式名、公式和 golden，不能改变本格式的含义。

## 验证

- property tests 验证 causal/sliding attention 计数、attention core FLOPs 与 INT4 分组边界；
- fixed Scenario golden 覆盖 Tiny BF16 prefill 与 Qwen INT4 decode；
- GLM 测试断言 executed cost 为 `null/unknown` 且聚合不补零；
- synthetic HardwareProfile 测试断言所有 roofline 值只是理论下界，且 dtype/输入缺失时保留 Unknown。

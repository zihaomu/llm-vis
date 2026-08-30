# LLM-Vis 示例

本目录给出已经实现并通过 M0–M3 验收的 CLI 与 artifact 示例，不包含模型权重，也不执行完整模型。以 `llm-vis --help` 为最终接口事实来源。

## 场景文件

- [`scenarios/prefill-2x2048.json`](scenarios/prefill-2x2048.json)：batch 2、2048 个新 token、无 past context；
- [`scenarios/chunked-prefill-2x512-l2048.json`](scenarios/chunked-prefill-2x512-l2048.json)：chunked prefill；
- [`scenarios/decode-8x1-l1024.json`](scenarios/decode-8x1-l1024.json)：BF16 decode；
- [`scenarios/decode-1x1-l4096-int4.json`](scenarios/decode-1x1-l4096-int4.json)：冻结 INT4 口径的 decode；
- [`scenarios/m3-prefill-decode.json`](scenarios/m3-prefill-decode.json)：用于同 artifact workload diff 的两场景组合。

Scenario 只用于绑定静态/公式指标。它不会触发模型 forward、权重加载、profiler 或 trace 采集。

## 模型示例

- [`qwen38_27b/README.md`](qwen38_27b/README.md)：固定 Qwen revision 的 config-first inspect；
- [`glm53/README.md`](glm53/README.md)：固定 GLM revision 的超大 MoE 宏观 inspect；
- [`tiny_dense/README.md`](tiny_dense/README.md)：允许的 Tiny/FakeTensor `torch.export` 边界。

M0–M3 的所有示例遵循 [DR-0007](../doc/decisions/DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md)：零权重、零完整模型 forward、完整 meta tree 默认关闭。M4 只导入外部 trace；本目录不提供会启动完整模型或 profiler 的命令。

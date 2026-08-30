# Qwen3.8-27B：config-first 示例

本示例固定：

- 模型：[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- revision：[`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- config SHA-256：`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`

已实现且仓库内可复现的 M1 离线调用：

```bash
llm-vis inspect \
  --model tests/fixtures/configs/qwen3_8_27b.json \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-files-only \
  --output artifacts/qwen38-27b
```

也可把 `--model` 改成官方模型 ID；此时必须先把固定 revision 的 config 放入本地 Hugging Face 缓存。`--local-files-only` 缓存缺失时直接报错，不得访问网络。

## 预期静态结果

- Vision/Text/Projector 级别宏观结构；
- 64 个文本层；
- `[Linear Attention × 3 → Full Attention] × 16`；
- Full Attention 的 KV cache 与 Linear Attention 的 recurrent state 分轨；
- 1 个 MTP conditional path；
- Definition 与 64 个 Instance 分离；
- M1 L2 仅包含 config 推导 SemanticNode，`logical_ops=[]`。

本命令不下载/读取权重，不构造完整 meta tree，不运行完整多模态或文本 forward。若未导入外部 trace，`runtime.measured_latency`、`runtime.kernel_dispatch_count` 和 `runtime.measured_hbm_traffic` 都是 Unknown/null；Kernel、dispatch、bandwidth 和 counter 实测信息不存在。

M2 若验证 Qwen 代表结构，只能构造项目自有的语义同类缩小 Block 并使用 FakeTensor/meta；该结果不能代表 27B 完整模型性能。M2+M3 的组合验收命令为：

```bash
llm-vis capture \
  --model tests/fixtures/configs/qwen3_8_27b.json \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --scenario examples/scenarios/m3-prefill-decode.json \
  --hardware-profile examples/hardware/synthetic-bf16.json \
  --local-files-only \
  --kind dense \
  --kind full_attention \
  --kind linear_state \
  --output artifacts/qwen38-27b-m3
```

生成的 `reports/report.html` 是自包含离线页面；示例 HardwareProfile 是虚构测试 profile，只用于验证 roofline lower-bound 流程，不代表真实 GPU。

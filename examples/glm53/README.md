# GLM-5.3-BF16：config-first MoE 示例

本示例固定：

- 模型：[`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16)
- revision：[`304b8051cfb2b260b61ce0cbe330e02a98e73639`](https://huggingface.co/zai-org/GLM-5.3-BF16/tree/304b8051cfb2b260b61ce0cbe330e02a98e73639)
- config SHA-256：`ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca`

已实现且仓库内可复现的 M1/M3 config-first 调用：

```bash
llm-vis inspect \
  --model tests/fixtures/configs/glm_5_3_bf16.json \
  --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
  --scenario examples/scenarios/m3-prefill-decode.json \
  --local-files-only \
  --output artifacts/glm53-bf16
```

也可把 `--model` 改成官方模型 ID；此时固定 revision 的 config 必须已经位于本地 Hugging Face 缓存。

## 预期静态结果

- 78 个层实例；
- 前 3 层 Dense，后 75 层 MoE；
- 256 routed experts、top-8、1 shared expert；
- 默认显示虚拟 Expert Pool，不展开 256 份完整子树；
- resident/active 参数口径分开；缺少证据的动态成本为 Unknown；
- `logical_ops=[]`，因为 M1 没有捕获真实 DSA/MoE 路径。

本示例不下载或加载 GLM 权重，不构造完整 meta tree，不运行完整 DSA/MoE forward。M3 只输出 config 可证明的 resident/active/storage 部分成本；DSA、executed MoE FLOPs/bytes、KV/state、AI、roofline 和动态热点保持 Unknown/null。动态 route histogram、tokens/expert、grouped-GEMM padding、executed-cost 和 Kernel 映射不属于 M0–M3 结果。没有外部 trace 时 Runtime 也全部为 Unknown/null。

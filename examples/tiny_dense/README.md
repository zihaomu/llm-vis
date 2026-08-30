# Tiny Dense：允许的代表块捕获示例

Tiny Dense 是 M0–M3 的完全离线 synthetic fixture，用于验证 IR、公式和 M2 `torch.export` 路径。它不包含下载的模型权重，也不代表任一真实大模型的 latency。

已实现的 M2 调用（从仓库根目录执行）：

```bash
llm-vis capture \
  --model tests/fixtures/configs/tiny_dense.json \
  --local-files-only \
  --kind dense \
  --kind full_attention \
  --output artifacts/tiny-dense-capture
```

允许的输入是 Tiny module 或同构缩小 Block，加 FakeTensor/meta 与符号约束。输出可以包含真实捕获到的 LogicalOps、Tensor dataflow、Lowering 和 opaque Diagnostic，但 manifest 必须标明：

```text
safety.weights_loaded = false
safety.full_model_constructed = false
safety.full_forward_executed = false
safety.full_meta_tree_enabled = false
captures[].capture_source = project-owned-tiny-fixture
captures[].capture_backend = torch.export.strict
captures[].tensor_mode = meta-parameters-and-inputs/faketensor-trace
captures[].architecture_equivalence = semantic-kind-only
captures[].valid_for_full_model_performance = false
```

当前 CLI 只暴露项目自有代表块的闭合注册表；Tiny Dense config 可映射 `dense` 与 `full_attention`，Qwen config 另可映射 `linear_state`。CLI 不接受用户传入真实模型 module 或权重，因此不会把捕获范围扩大成完整 27B/GLM forward。捕获失败时保留 config-first IR，并以 opaque Diagnostic 表达未知区域。

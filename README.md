# LLM-Vis

LLM-Vis 是一个本地优先的 LLM 结构与性能分析工具。M0–M3.6 围绕 Hugging Face 配置或本地 `config.json`，再加项目自有 Tiny 语义代表块捕获到的逻辑图，构建统一的 Model Map IR、递归 GraphView，并生成可离线查看、可分享的静态报告。MVP 不建设托管服务，也不依赖 GGUF、大模型权重或用户 Python 模型代码。

M0–M3.6 的本地静态 MVP 已完成并通过机器退出清单。Qwen/GLM 可操作语义 DAG 能在同一中央画布把 Full/Linear Attention、FFN 与静态 MoE 递归展开到 semantic primitives；复合节点以 `N ops ›` 明示，`Collapse` 返回父图。主图可切换 Scenario 联动的 Pressure/Compute/Memory 理论热力层，并只统计当前 visible cost frontier。默认 UI 仍是 graph-first 布局：中央主图占据首屏，Browse/Inspector 为按需抽屉，Layers/Supporting analysis 默认折叠。独立 `graph-view.json` 是 Model Map 证据的渲染中立投影，不是运行目标模型得到的执行图；热力层也只是 formula/estimated Metric 的报告投影，不是实际模型执行或 latency 测量。当前事实来源是 [完整规划](doc/llm-visualization-plan.md)、[Model Map IR v0.1](doc/model-map-ir.md)、[Adapter 指南](doc/adapter-guide.md)、[UI wireframe](doc/ui-wireframe.md)、[measurement protocol](doc/measurement-protocol.md)和[决策记录](doc/decisions/README.md)。M4 尚未开始，Runtime 继续是 trace-import-only 的边界约定。

## 已冻结的产品边界

- 分发形态是本地命令行工具加离线静态 HTML/Markdown/JSON，不建设远程托管分析服务。
- M0–M3.6 的默认与验收路径是零权重、零完整模型 forward：不下载/读取模型权重，不运行完整文本、多模态或 MoE 模型。
- 完整 meta module tree 默认禁用。M2 的 `torch.export` 只用于 Tiny 或同构缩小的代表性 Block，并使用 FakeTensor/meta 输入。
- MVP **禁止执行 Hugging Face remote code**。存在 `auto_map` 或需要 `trust_remote_code=True` 时，只能读取 JSON、文本源码和文件清单；不得导入远程 `configuration_*.py`、`modeling_*.py` 或其他 Python 模块。
- M1 的 L2 只表示 adapter 从 config 推导出的语义契约或占位，不是一次真实 forward 的算子图。真实 `LogicalOp`、Tensor 数据依赖和 ATen/custom op 来源由 M2 的 `torch.export` 代表性 Block 捕获产生。
- M3.5 的 L0/L1 标记为 `CONFIG SEMANTIC PROJECTION`：端口和 activation/route/control 边只能来自 config/adapter 可追溯规则，不声称是目标模型的实际算子调度、精确执行顺序或 Kernel 图。
- M3.6 递归 view 继续是同一 `CONFIG SEMANTIC PROJECTION`：semantic primitive 不是 ATen/Kernel 边界；每个 parent/child port 有显式无损 binding，Unknown/partial 成本不补 0，GLM route/DSA 不伪造。
- `graph-view.json` 不取代 `model-map.json`：前者仅组织可见 view/node/port/edge/group 与下钻关系，后者仍是 TensorSpec、SemanticNode、LogicalOp、Metric 和 provenance 的权威事实层。
- Model Explorer 只作为独立 custom-adapter JSON spike。其确定性 snapshot 已通过；官方 consumer/UI 的完整审计仍为 Partial。Model Map IR、成本、provenance 和离线报告不依赖它的内部 schema。
- 首个 AMD reference stack 尚未冻结，因此 M4a 当前处于阻塞状态；现阶段不能承诺 op-to-kernel 覆盖率或 AMD runtime 性能结论。
- M4 Runtime 只导入用户在外部环境生成的 trace；LLM-Vis 不启动完整模型或 profiler。没有 trace 时，所有 Runtime 指标都是 `Unknown/null`。
- `Unknown` 不是 `0`。所有指标必须携带来源、假设、Scenario 和覆盖状态。

对应理由和复审条件见 [决策索引](doc/decisions/README.md)，其中 [DR-0007](doc/decisions/DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md)冻结零权重/trace-only 边界，[DR-0008](doc/decisions/DR-0008-m3-cost-int4-and-roofline-boundary.md)冻结 M3 成本、INT4 和 roofline 口径。

## 已完成的 M0–M3.6

当前纵切包括：

1. 冻结 Model Map IR v0.1、Scenario、Metric、Diagnostic、稳定 ID 与 provenance 规则；
2. 解析 Hugging Face Model ID、本地目录和精确 revision，优先使用本地缓存；
3. 为 Tiny Dense、Qwen3.8-27B 和 GLM-5.3-BF16 生成 Definition/Instance、重复层、Layer Type Strip、config 可证明的参数量与状态类别；
4. 对 Qwen 构建已知 config adapter，对 GLM 只构建可信的 config-level 宏观图；
5. 只用 Tiny/同构缩小代表块的 FakeTensor/meta 验证结构；完整 meta tree 默认关闭，失败时退回 config/template 并生成 Diagnostic；
6. 生成确定性的 JSON/Markdown 与无外部网络依赖的静态 HTML；HTML 支持 Scenario 切换、搜索、Inspector、成本、热点、roofline、diff 与 Runtime Unknown 面板；
7. 用项目自有 Tiny Dense/Full Attention/Linear State 的 meta/FakeTensor `torch.export` 生成真实 LogicalOps/Tensor/Lowering；失败转为 opaque，绝不回退完整模型；
8. 计算 prefill/chunked prefill/decode 下的 BF16/FP16/冻结 INT4 成本、KV/state、理论热点、显式 HardwareProfile roofline lower bounds 和同 artifact workload diff。
9. 生成 renderer-neutral GraphView 与完全离线的端口级 SVG DAG：Qwen L0 可下钻 Linear/Full Attention L1，GLM L0 可下钻 Dense/Sparse DSA+MoE L1；支持 state/route/control 样式、三行 Tensor 边标签、可读缩放、pan/zoom/Fit/minimap、Layer Strip、搜索/热点定位、上下游高亮和六页 Inspector。
10. M3.6 已完成递归 operator views、semantic primitive ontology、boundary binding、cost reconciliation 与 visible frontier；覆盖 Qwen Full/Linear Attention、Dense FFN、KV/recurrent state，以及 GLM Router/TopK/Expert/Shared Expert 与 opaque DSA，OP-01～OP-10 全部通过。

GLM-5.3 的动态 DSA/MoE 捕获、跨模型/revision diff、第二捕获后端和 runtime trace importer 仍按计划延后。M0–M3.6 始终不加载权重、不运行完整模型 forward。

## 固定验证模型

| 模型 | 固定 revision | 首批用途 |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 64 层、`Linear × 3 + Full × 1`、KV/recurrent state、多模态宏观结构和 MTP |
| [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16/tree/304b8051cfb2b260b61ce0cbe330e02a98e73639) | `304b8051cfb2b260b61ce0cbe330e02a98e73639` | 78 层、前 3 层 Dense、后 75 层 MoE、256 routed experts、top-8 和 shared expert 的惰性宏观图 |

固定 revision 的范围和 config 校验摘要见 [DR-0005](doc/decisions/DR-0005-frozen-model-revisions.md)。示例入口见 [examples](examples/README.md)。

## CLI

以下命令已实现：

```bash
# M1：完全离线读取仓库内 config fixture，不读取权重
llm-vis inspect \
  --model tests/fixtures/configs/qwen3_8_27b.json \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-files-only \
  --output artifacts/qwen38-27b-structure

# M2+M3：捕获项目自有 Tiny 语义代表 Block，并绑定两种成本 Scenario
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

# M3：只比较同一 artifact 内的两个 Scenario
llm-vis diff artifacts/qwen38-27b-m3 \
  --left-scenario <prefill-scenario-id> \
  --right-scenario <decode-scenario-id> \
  --output artifacts/qwen38-27b-m3/workload-diff.json

# 机器退出清单
uv run python scripts/verify_milestones.py --milestone all
```

`--local-files-only` 的语义是禁止网络访问并要求所有输入已经位于本地；缓存缺失直接报错，不能悄悄访问网络。`inspect` 与 `capture` 会直接生成 JSON、Markdown 和自包含 HTML，不存在单独的 `report` 或 `serve` 命令。打开 HTML 时不会向第三方发送请求。

## 安全边界

MVP 中，`--trust-remote-code` 不属于有效接口。普通 Python 子进程只是故障隔离，不是安全沙箱，不能作为执行 remote code 的理由。未来只有在经过审计的 OS/container/VM sandbox 同时实现网络、凭据、宿主文件、CPU、内存和超时限制后，才可在新的 Decision Record 中讨论显式 opt-in。

本地 Python import path adapter 尚未实现，M0–M3.6 对目标模型只读取配置数据。未来若加入本地代码入口，只能把用户明确选择的代码按“用户信任代码”处理，并在 manifest 记录路径、commit/哈希和执行方式。模型源码文本、trace、IR 字符串和 node metadata 均按不可信数据处理；静态 HTML 必须转义内容，禁止把模型提供的文本当作脚本或 HTML 注入报告。

`inspect`、`capture` 和其他 M0–M3.6 命令不得隐式触发权重读取、完整 meta tree、完整 forward、backend compile 或 profiler。代表性 Block 的捕获只证明 Tiny fixture 的结构/Tensor 契约，不能作为 27B/GLM 完整模型的 latency 或 Kernel 证据。M4 也只读取外部 trace；无 trace 的 Runtime 面板必须显式显示 Unknown，不能用静态 FLOPs、logical bytes 或 roofline 填充实测字段。

## Artifact 约定

一个分析目录预期包含：

```text
artifacts/<analysis-id>/
├── manifest.json
├── model-map.json
├── graph-view.json
├── scenarios.json
├── metrics.json
├── diagnostics.json
├── layer-strip.json
├── captures.json
├── hardware-profile.json
├── hotspots.json
├── roofline.json
├── workload-diffs.json
├── model-explorer.json
├── reports/
│   ├── report.html
│   └── report.md
```

Artifact 记录模型 revision/config hash、工具版本、adapter 名称、输入来源/缓存状态、Scenario、`safety.*` 零执行声明、代表块的 backend/tensor mode/PyTorch 版本、Metric 来源与覆盖率。`graph-view.json` 与报告内嵌的 DAG 数据同源，并用 canonical Model Map 内容生成的 `mmap_*` ID 关联事实层：L0 保留端到端主路径和折叠 pattern，L1 保留 Qwen Linear/Full Attention 或 GLM Dense/Sparse Block 的具体输入输出端口，operator views 保存递归分解、boundary binding、cost reconciliation 与 visible frontier；KV/recurrent state 用独立 state rail 表达，weight 边默认不进入主画布。报告根据完整 Metric、当前 Scenario 和显式 HardwareProfile 临时生成热力 overlay，不修改 GraphView：Qwen 只给当前 frontier 中可归因的节点着色，GLM executed cost 未知时显示 Unknown 而不补 0。该产物已通过 M3.5/M3.6 schema、golden、引用、可达性与浏览器验收；manifest 的 `recursive_operator_decomposition` 明确说明当前 artifact 是否实际包含 operator child views。

报告可以在没有原模型仓库和网络的环境中重新打开；报告本身不包含模型权重。生成 GraphView 不会构造目标模型或执行 forward。没有导入 trace 时，三个 `runtime.*` Metric 必须是 `origin=unknown, value=null, coverage_status=unmapped`，并带 `RUNTIME_TRACE_MISSING` Diagnostic。

## 里程碑状态

| 里程碑 | 当前状态 | 边界 |
|---|---|---|
| M0 | **完成** | IR、golden、决策、wireframe、measurement protocol 与 spike 结论 |
| M1 | **完成** | config-first 静态结构；L2 config 语义投影；三模型 golden |
| M2 | **完成** | Tiny/缩小语义代表 Block 的 FakeTensor/meta `torch.export`、Logical Ops 与 opaque fallback |
| M3 | **完成** | 零权重 Scenario 成本、冻结 INT4、热点、roofline 和同 artifact workload diff |
| M3.5 | **完成** | Qwen/GLM L0→L1 config-first GraphView、`L1 ›` 可发现下钻、graph-first 中央主图、Scenario 理论热力层、按需 Browse/Inspector、折叠 Layers/Analysis、state rail、完整 Tensor 边标签与 Inspector 联动；24/24、Python 3.9/3.12 各 150 项、浏览器 DAG-01–DAG-13 通过；不执行目标模型 |
| M3.6 | **完成** | 同一 DAG 递归算子分解、`N ops ›`/Collapse、boundary binding、cost reconciliation、visible frontier、Qwen Attention/FFN/state 与 GLM static MoE/opaque DSA；27/27 checked-in 资产检查、Python 3.9/3.12 各 192 项及 Qwen/GLM 浏览器 OP-01～OP-10 验收通过 |
| M4a | **Blocked** | trace-import-only；AMD reference stack 尚未冻结 |
| M4b | Blocked by M4a | hardware counter 与 replay 验证 |
| M5 | 延后 | GLM 动态 MoE、第二 backend、跨 artifact diff 与硬化 |

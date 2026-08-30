# LLM-Vis Measurement Protocol v0.1

- 状态：M0 测量与退出基线
- 日期：2026-08-29
- 适用范围：M0–M4；M4 当前仍被未冻结的 AMD reference stack 阻塞

## 1. 核心声明

M0–M3.5 不产生、预测或承诺完整模型 latency：

- M0/M1 只有 config-first 结构、确定性 inventory 和 Coverage；
- M2 只有 Tiny/同构缩小代表块的 FakeTensor/meta capture，用于结构与 LogicalOp 验证；
- M3 只有公式成本、logical bytes、arithmetic intensity 和理论 roofline lower bounds；
- M4 只导入用户在外部环境生成的 trace，不启动完整模型、backend compile、profiler 或 counter replay；
- 没有外部 trace 时，所有 Runtime Metric 都是 `origin=unknown, value=null`。

M4 的 trace-import-only 边界见 [DR-0007](decisions/DR-0007-zero-weight-bounded-capture-and-trace-only-runtime.md)。首个 AMD reference stack 尚未冻结，因此只阻塞 M4a/M4b，不阻塞 M0–M3.5；见 [DR-0006](decisions/DR-0006-amd-reference-stack-blocks-m4a.md)。

## 2. Metric 真实性

| Origin | 可接受来源 | 可以声称 | 不可以声称 |
|---|---|---|---|
| Exact | 固定 config、参数 shape、已验证 IR | 对应结构事实精确 | 完整 forward 已发生 |
| Formula | 明确公式 + Scenario | 理论算法工作量 | 实测 latency/HBM traffic |
| Estimated | route/cache/layout 等显式假设 | 假设下的估算值 | 确定或实测值 |
| Measured | 导入的外部 timing/trace/counter | 对应 run 与采集条件的实测值 | 其他硬件/workload 的结论 |
| Inferred | marker/correlation 等规则关联 | 带置信度的归属 | 无证据的精确映射 |
| Unknown | 信息缺失或不适用 | 当前没有足够证据 | `0`、空白或默认硬件值 |

Unknown 永远不参加数值求和或 delta 基准；partial 聚合必须同时展示覆盖率。

## 3. M0–M3 公式协议

每个公式 Metric 至少记录：

- `subject_id` 与 Scenario ID；
- value、unit 和 `origin=formula|estimated`；
- 公式 AST 或可核验公式 ID；
- 符号绑定，如 `B/T/L/S_KV/H/N_Q/N_KV/D_H`；
- dtype、weight format 和 KV dtype；
- assumptions、confidence、coverage/status；
- adapter 与 cost-model version。

Scenario 统一使用：

```text
T = 本次调用的新 token 数
L = 调用前已有 context
S_KV = L + T
```

prefill、chunked prefill 和 decode 分开保存。改变 Scenario 重算 Metric，不改变结构身份。

### 3.1 理论 roofline

只有用户显式提供、dtype 匹配并带 provenance 的 HardwareProfile 才能计算：

```text
compute lower bound   = formula FLOPs / peak FLOPs per second
bandwidth lower bound = logical bytes / memory bandwidth bytes per second
max lower bound       = max(compute lower bound, bandwidth lower bound)
arithmetic intensity  = formula FLOPs / logical bytes
```

输出仅用于理论下界和 bottleneck class。报告固定声明：

```text
Theoretical lower bounds — not a latency estimate.
```

这些下界不包含 launch、cache、fusion、layout、occupancy、同步、调度或物理 HBM traffic。缺少 FLOPs、logical bytes、匹配 dtype 的峰值或带宽时，对应项和 max/bottleneck 保持 Unknown。项目不提供或猜测默认硬件；[`synthetic-bf16.json`](../examples/hardware/synthetic-bf16.json) 仅用于虚构自含测试。

M3.5 的主图热力层只是上述既有 Metric 和公式下界的 Scenario/view 投影，不产生新测量：`Compute=可归因 FLOPs`、`Memory=可归因 logical bytes`、`Pressure=max(compute lower bound, bandwidth lower bound)`。颜色强度在当前 view 内按 `raw / max(raw_known)` 归一化，只表示相对理论压力，不能解释为节点 latency、实际 GPU 利用率或跨 view/Scenario 可比的绝对尺度。节点归属必须按明确组件指标选择：L0 decoder pattern 使用成员 Instance 汇总，L1 Attention/FFN 分别使用 `*.attention`/`*.ffn`；不能因共享 `subject_ids` 就把整层成本复制到 Norm、Residual 或边界节点。

Unknown、opaque、not attributable 和 partial 必须以文字/纹理/coverage 与颜色共同编码。Unknown 不进入归一化分母，也不能当作冷色 0；partial 只能显示已知 subtotal 并保留 Coverage。GLM DSA/MoE 在 config-only 能力下缺少 executed FLOPs/logical bytes 时，整张计算热力图保持 Unknown，禁止用 active parameters、weight storage 或 top-k 参数上界代替。legend 与 Inspector 必须显示 HardwareProfile 名称/ID/dtype/provenance、公式、origin、coverage 和 `not measured latency`；synthetic profile 必须显著标记为虚构测试值。

## 4. M2 capture 协议

M2 capture 只接受 Tiny fixture 或同构缩小代表块，使用 FakeTensor/meta，且不加载真实权重。artifact 记录：

```text
weights_loaded = false
full_model_constructed = false
full_forward_executed = false
full_meta_tree_enabled = false
capture_source = project-owned-tiny-fixture
architecture_equivalence = semantic-kind-only
valid_for_full_model_performance = false
```

capture 用于验证 LogicalOp、Tensor contract、lowering 和 opaque fallback。它不能产生或校准 27B/GLM 完整模型 latency。

## 5. M4 外部测量输入

LLM-Vis 只导入外部测量。外部生产者必须将 timing、timeline 和 counter 分成三个独立 run/artifact；不能把任一 run 的 wall time冒充另一类数据。

### 5.1 Timing run

用途：外部环境中的 steady-state model-only 或明确服务边界 latency。

协议：

- compile、权重预处理和首次 Kernel 初始化不计入 steady-state；
- warmup 后至少 10 次，短任务建议 30 次以上；
- 保存原始 samples，并报告 median、p95、标准差和变异系数；
- 变异系数超过 3% 标记 unstable；
- 固定输入、随机种子和 MoE route；
- cold-cache 与 warm-cache 分开；
- 记录 instrumentation overhead；
- model-only prefill 只能叫 model-only prefill latency/prompt throughput；若不含 tokenizer、scheduler、queue、sampling 和网络，不得称为端到端 TTFT。

### 5.2 Timeline run

用途：module/op/dispatch/copy/sync 的时间线和映射覆盖。

协议：

- profiler/ROCTx instrumentation 与 timing run 分开；
- 保存 raw trace、工具版本、clock domain、stream、correlation/dispatch ID；
- instrumentation 可能改变 wall time，因此 timeline duration 不自动覆盖 timing 统计；
- copy 单列，idle 不进入 kernel mapping 分母；
- 重叠 dispatch 各自按 duration 计数，但 exclusive attribution 不重复归属；
- 无法映射的事件进入 `Unattributed`，不能丢弃。

### 5.3 Counter run

用途：HBM/L2/LDS、occupancy、MFMA/VALU 等硬件 counter。

协议：

- 与 timing/timeline run 分开；
- 保存 counter set、replay 次数、序列化模式和 capability discovery；
- replay 前后验证路径/输入/route hash 一致；
- counter replay 或 dispatch serialization 的 wall time永远不进入生产 latency；
- unsupported counter 保持 Unknown，不用其他 GPU 的默认值替代。

## 6. 外部 artifact 最小 provenance

每类外部 run 至少记录：

```text
run_id, run_kind = timing | timeline | counter
model_id, full revision, code revision
input/route hash, Scenario, dtype/quantization/cache policy
GPU model, gfx, memory, driver, ROCm
PyTorch, Transformers, backend, profiler/counter tool versions
warmup/repeat, cache state, seed, clock/power mode, temperature
command/launcher identity, raw artifact SHA-256
```

缺少关键 provenance 时 importer 可以保留原始数据，但必须降低 confidence/coverage 并产生 Diagnostic。

## 7. M4 映射与 Coverage

稳定关联链为：

```text
stable semantic_id
  → per-run marker/range
  → per-run correlation_id
  → per-run dispatch_id
```

不得只用 Kernel 名称模糊匹配。融合 Kernel 的 exclusive time 归 fusion group；参与融合的语义节点只使用 shared/inclusive attribution。Coverage 报告必须声明分母：duration-weighted GPU kernel dispatch time；copy 单列、idle 不计、重叠 dispatch 不重复归属。

M4a 的 95% eager / 90% compile-fused 目标只有在 [DR-0006](decisions/DR-0006-amd-reference-stack-blocks-m4a.md) 解锁且 reference stack 固定后才可验收。当前不能用任意开发机 trace 宣称达标。

## 8. 退出命令模板

命令从项目根目录执行。`<artifact-dir>`、`<external-trace>` 等为调用方提供的明确路径；模板不授权下载权重或运行完整模型。

### 8.1 M0：IR、golden、adapter spike 与离线报告

```bash
.venv/bin/python scripts/update_structure_goldens.py --check

.venv/bin/pytest -q \
  tests/unit/test_ir_models.py \
  tests/golden/test_structure_goldens.py \
  tests/integration/test_model_explorer_adapter.py \
  tests/integration/test_inspect_artifact.py
```

期望：golden current；全部测试通过。Model Explorer spike 的剩余人工/消费者验证仍按 [DR-0003](decisions/DR-0003-model-explorer-bounded-spike.md) 记录，不能因上述测试通过而自动关闭。

### 8.2 M1：config-first artifact

```bash
.venv/bin/llm-vis inspect \
  --model tests/fixtures/configs/tiny_dense.json \
  --local-files-only \
  --output <artifact-dir>

.venv/bin/llm-vis validate <artifact-dir>/model-map.json

jq -e '
  .safety.weights_loaded == false and
  .safety.full_model_constructed == false and
  .safety.full_forward_executed == false and
  .safety.remote_code_executed == false and
  .safety.full_meta_tree_enabled == false
' <artifact-dir>/manifest.json

jq -e 'all(.[]; if .origin == "unknown" then .value == null else true end)' \
  <artifact-dir>/metrics.json
```

Qwen/GLM 使用固定本地 config fixture 和 [DR-0005](decisions/DR-0005-frozen-model-revisions.md) revision 重复同一模板。

### 8.3 M2：受限 representative capture

```bash
.venv/bin/llm-vis capture \
  --model tests/fixtures/configs/tiny_dense.json \
  --local-files-only \
  --kind dense \
  --kind full_attention \
  --output <artifact-dir>

.venv/bin/llm-vis validate <artifact-dir>/model-map.json

.venv/bin/pytest -q tests/unit/test_capture.py tests/integration/test_capture_merge.py
```

期望：Tiny capture 产生 LogicalOps/Lowering；GLM 明确 deferred；故障注入产生 opaque/Diagnostic；manifest 仍声明零权重、零完整 forward。

### 8.4 M3：公式、roofline 与同模型 workload diff

```bash
.venv/bin/pytest -q \
  tests/unit/test_cost_formulas.py \
  tests/unit/test_cost_properties.py \
  tests/unit/test_cost_expressions.py \
  tests/unit/test_cost_storage.py \
  tests/unit/test_cost_engine.py \
  tests/unit/test_roofline.py \
  tests/unit/test_diff_scenarios.py \
  tests/integration/test_cost_pipeline.py \
  tests/integration/test_m3_report.py \
  tests/integration/test_m3_cli.py
```

期望：固定公式/golden 一致；roofline 只输出 theoretical lower bounds 且无硬件默认；Unknown 不补 0；diff 只允许同一 ModelMap 的两个 Scenario。

统一机器入口：

```bash
.venv/bin/python scripts/verify_milestones.py --milestone all
.venv/bin/pytest -q
```

2026-08-30 的历史 M0–M3 退出结果为静态资产 20/20 PASS（包含 M2 Qwen 三类代表块与 opaque/no-full-model fallback contract）。加入 M3.5 后的当前累计结果为 24/24 PASS，Python 3.9 与 3.12 各 150 项测试通过。

### 8.5 M3.5：GraphView 与离线 DAG

```bash
.venv/bin/python scripts/update_graph_view_goldens.py --check
.venv/bin/python scripts/verify_milestones.py --milestone all
.venv/bin/pytest -q \
  tests/unit/test_graph_view.py \
  tests/unit/test_dag_canvas.py \
  tests/integration/test_inspect_artifact.py \
  tests/integration/test_m3_report.py \
  tests/integration/test_capture_cli_report.py
```

机器检查必须验证：GraphView schema 与 Qwen/GLM golden；canonical Model Map
内容寻址 `mmap_*`；所有 subject/TensorSpec/view/group/port 引用；output→input、
单 input 单 producer、主图无环、L0 output 数据可达；`weights_loaded`、
`target_model_constructed`、`target_model_forward`、`remote_code_executed` 全为 false。

浏览器检查使用自包含 Qwen/GLM `reports/report.html`，覆盖 DAG-01–DAG-13：
L0→L1 下钻与 Back、节点/port/edge Inspector、三行 Tensor 名/shape/dtype、
state/route/control、Layer Strip、跨 view 搜索、热点与上下游高亮、真实拖拽平移、
zoom/Fit/minimap、Scenario 刷新、Unknown 原因和空 console；同时验证 graph-first 默认
状态只常驻中央主图，Browse/Inspector 可开关且不丢 view/selection，Layers 与 Supporting
analysis 默认折叠、展开后不挤压或替换当前图。通过 localhost 只为浏览器读取本地 artifact；
页面自身必须不请求任何外部资源。

DAG-12 必须验证可下钻节点的显式 `L1 ›` 数量和角标点击/双击/Enter/Space 契约；DAG-13 必须验证 Pressure/Compute/Memory、Scenario 重算、Qwen L1 仅 Attention/FFN 可归因、selection/upstream/downstream 描边不被热色覆盖，以及 GLM 全图 Unknown 不补 0。浏览器还需确认 legend 与 Inspector 可见 synthetic provenance、相对归一化、logical bytes 非物理 HBM traffic 和非 latency 声明，console 无 warning/error。

2026-08-30 结果：24/24 PASS；Python 3.9 与 3.12 各 150 passed；Qwen
L0/Linear L1/Full L1 为 `11/19/10`、`10/24/13`、`10/24/13`
（node/port/edge），GLM L0/Dense L1/Sparse L1 为 `9/15/8`、`10/20/11`、
`13/30/17`；两份页面 console 均无 warning/error。测试期间没有读取模型权重、
构造 Qwen/GLM 模型或运行其 forward。graph-first 复测在 1280×720 下确认中央 DAG
占满首屏可用宽度，GLM 搜索 `Expert Pool` 后进入 Sparse L1 但不自动打开 Inspector，
Browse/Inspector 显式开关与 Layers/Analysis 折叠状态均符合 UI wireframe。

### 8.6 M4：计划模板，当前禁止作为退出证据

以下命令是 reference stack 冻结后的接口模板，当前 CLI 未实现 `import-trace`，不得执行或据此宣称 M4 完成：

```bash
llm-vis import-trace \
  --model-map <artifact-dir>/model-map.json \
  --trace <external-trace> \
  --format <pytorch-profiler|rocprofv3|rocm-compute-profiler> \
  --run-kind <timeline|counter> \
  --output <artifact-dir-with-imported-trace>
```

M4 退出命令只有在新的 Accepted reference-stack DR、外部 trace fixture、importer 测试和固定预期摘要存在后才可转为可执行命令。

## 9. 退出报告模板

每个里程碑保存一份摘要：

```text
milestone, command, exit_code
tool/schema/adapter/cost-model versions
fixture/model revision and input SHA-256
Scenario IDs
test/golden summary
Metric origins and coverage summary
safety flags
known failures/deferred diagnostics
hardware/reference stack = Unknown（M0–M3.5）或 fixed DR id（M4）
```

退出报告不因测试数量多而隐去未覆盖项。任何 Unknown、Partial、Opaque、Deferred 或 Blocked 都必须显式列出。

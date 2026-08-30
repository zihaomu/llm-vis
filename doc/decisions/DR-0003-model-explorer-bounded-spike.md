# DR-0003：M0/M1 使用 Model Explorer 的 bounded spike

- 状态：Accepted（bounded spike 路径）
- Spike 退出审计：Partial（2026-08-29）
- 日期：2026-08-29
- 决策者角色：Frontend / Tech Lead
- 复审点：M1 退出评审；达到自研门槛时提前复审

## 背景

M0/M1 需要尽快验证层级展开、重复结构、搜索和节点 metadata。Model Explorer 已提供层级图、custom adapter、custom node data 和 WebGL 渲染能力，其官方仓库采用 Apache-2.0 许可证：

- [Model Explorer 官方仓库](https://github.com/google-ai-edge/model-explorer)
- [官方 custom adapter 指南](https://github.com/google-ai-edge/model-explorer/wiki/6.-Develop-Adapter-Extension)
- [官方 API / custom node data 指南](https://github.com/google-ai-edge/model-explorer/wiki/4.-API-Guide)
- [Apache-2.0 LICENSE](https://github.com/google-ai-edge/model-explorer/blob/main/LICENSE)

这些能力与 LLM-Vis 的 M0/M1 图形验证相符，但 workload、lowering、timeline、diff 和静态离线打包仍需验证。

## 决策

M0/M1 采用一个有边界的 Model Explorer spike：

- LLM-Vis 先生成自己的 Model Map IR；
- 独立 custom adapter 将 IR 转换成 Model Explorer graph/node data；
- 核心 resolver、adapter、IR、成本和 provenance 不导入 Model Explorer schema；
- Model Explorer 只负责图渲染和基础交互；外围 UI/报告拥有 workload、Coverage 和后续 runtime 视图；
- spike 必须验证离线打包，不能让报告依赖远程 Model Explorer 服务或 CDN；
- 这项决策不是永久冻结前端，也不是声明 spike 已经通过性能验收。

## Spike 退出条件

M0/M1 至少证明：

1. 同一份 IR 可以渲染 Tiny Dense、Qwen 和 GLM 的 L0/L1 层级；
2. Definition/Instance 折叠后可定位到具体层实例；
3. 节点可以显示 stable ID、shape/state、provenance、Coverage 和 Diagnostic；
4. 搜索、展开/折叠和 metadata overlay 不要求修改核心 IR；
5. adapter 输出可以确定性测试；
6. 静态报告或本地浏览方式在断网环境可用；
7. 许可证和分发依赖被记录并满足项目分发要求。

未通过的条件必须有书面结果，不能以“之后再做”视为 spike 成功。

## 退出审计与证据

本次审计把“采用 bounded spike 的技术路径”和“spike 已满足全部退出条件”分开判断。前者维持 Accepted；后者目前是 Partial，不能据此声明 Model Explorer 已成为最终前端。

| # | 结果 | 当前可核验证据 | 尚缺证据 |
|---|---|---|---|
| 1 | Partial | [`test_model_explorer_adapter.py`](../../tests/integration/test_model_explorer_adapter.py) 对 Tiny、Qwen、GLM 使用同一转换器，验证 L0/L1 图、节点上限和确定性输出。 | 还没有由官方 Model Explorer consumer 实际加载产物的 smoke test，因此只能证明适配数据生成，不能证明三种模型已经在官方 UI 中完成渲染。 |
| 2 | Partial | [`model_explorer.py`](../../src/llm_vis/report/model_explorer.py) 输出 Definition/Semantic/LogicalOp 图并保留 representative path、实例计数及 source IDs；自包含离线 HTML 已通过 Layer Strip/搜索定位具体 Qwen Instance。 | 官方 Model Explorer consumer 内从折叠 Definition 定位任意 Instance 的交互仍未验收。 |
| 3 | Partial | 适配器测试验证 stable ID、state、provenance/evidence；离线 Inspector 已联动 Instance、Tensor、Metric formula/origin/Coverage 与 Diagnostic，GLM Unknown 也有浏览器证据。 | 这些 overlay 尚未在官方 Model Explorer consumer 中验收。 |
| 4 | Partial | adapter 通过 namespace/attrs 承载元数据，不要求核心 IR 采用 Model Explorer schema；离线 HTML 的搜索、Scenario 切换与 metadata Inspector 已通过实际页面验收。 | 官方 Model Explorer UI 中的搜索、展开/折叠和 overlay 仍缺真实 consumer 证据。 |
| 5 | Pass | [`test_model_explorer_adapter.py`](../../tests/integration/test_model_explorer_adapter.py) 对多种模型重复生成并断言结果相等；结构 golden 由 [`test_structure_goldens.py`](../../tests/golden/test_structure_goldens.py) 固定。 | — |
| 6 | Pass | [`test_inspect_artifact.py`](../../tests/integration/test_inspect_artifact.py) 验证离线 artifact 可重复、HTML 内嵌 JSON 且不含 `http://` 或 `https://` 远程依赖；[`html.py`](../../src/llm_vis/report/html.py) 生成无第三方运行时依赖的静态 HTML。 | — |
| 7 | Pass（当前纯 JSON spike） | 本 DR 记录官方仓库、adapter/API 指南与 Apache-2.0 LICENSE；当前代码只输出自有 JSON/HTML，没有打包或再分发 Model Explorer 依赖。 | 若后续把 Model Explorer UI 或其依赖打包进报告，必须重新审核版本、NOTICE、许可证与离线分发物。 |

审计复现命令：

```bash
.venv/bin/python scripts/update_structure_goldens.py --check
.venv/bin/pytest -q \
  tests/golden/test_structure_goldens.py \
  tests/integration/test_model_explorer_adapter.py \
  tests/integration/test_inspect_artifact.py \
  tests/integration/test_capture_cli_report.py \
  tests/integration/test_m3_report.py
```

2026-08-30 的最终 M0–M3 回归为 Python 3.9/3.12 各 `139 passed`，structure goldens current；Qwen/GLM 自包含离线页面另经过实际浏览器交互且无脚本告警。这足以接受 adapter/离线报告基线和 M0–M3 自包含 UI，但不足以把官方 Model Explorer 退出项 1–4 判为 Pass。剩余工作必须补充真实 consumer 加载及其交互证据。对应界面契约见 [`ui-wireframe.md`](../ui-wireframe.md)，测量声明边界见 [`measurement-protocol.md`](../measurement-protocol.md)。

## 自研主画布复审门槛

下列问题满足任意两项时启动自研 WebGL/canvas 主画布评估：

- workload 切换必须重载整图；
- 无法稳定联动 Layer Strip、主图、热点表和 timeline；
- 无法表达多对多 lowering；
- 大于 10k 原始 op 时达不到交互目标；
- diff、expert heatmap 或 state track 需要大量绕过；
- custom adapter 维护成本高于自研渲染层。

## 后果

- Model Explorer 版本和 adapter 输出要固定并测试。
- 不把 Model Explorer 对 PyTorch ExportedProgram 的支持当作 M1 已有真实 Logical Ops 的证据。
- spike 失败不会阻塞 IR、CLI 和离线报告；它只会触发前端路径复审。
- 在退出审计从 Partial 变为 Pass 之前，对外只能描述为“Model Explorer-compatible adapter spike”，不能描述为“完整 Model Explorer UI 已验收”。

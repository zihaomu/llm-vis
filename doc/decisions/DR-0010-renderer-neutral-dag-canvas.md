# DR-0010：M3.5 使用 renderer-neutral GraphView 与自包含 DAG 主画布

- 状态：Accepted
- 日期：2026-08-30
- 决策者角色：Product / Frontend / Tech Lead
- 复审：M3.5 Qwen/GLM 浏览器退出验收已于 2026-08-30 通过；10k raw-op 性能路线在 M5 前复审

## 背景

M0–M3 的自包含 HTML 已能展示结构列表、Layer Strip、成本和六页 Inspector，
但尚未实现 Netron/ComfyUI 风格的端到端节点、端口和连线主画布。主计划虽然
描述了多尺度画布，UI 规格也明确把独立主画布留给后续 consumer/自研评估。
当前官方 Model Explorer consumer 退出仍为 Partial，且现有 config IR 的 Edge
主要是 weight/state ownership，不能直接冒充完整 activation forward graph。

## 决策

- 新增 M3.5，在 M4 runtime 之前交付 Qwen/GLM L0→L1 可操作语义 DAG；
- 新增 renderer-neutral GraphView artifact/schema，使用稳定 Node/Port/Edge/Group，
  不把 Model Map IR 绑定到任何前端库；
- L0/L1 只绘制 adapter/config 证据支持的语义数据流，并显著标记
  `CONFIG SEMANTIC PROJECTION`；L2 只有项目自有 Tiny 代表块捕获的 LogicalOp
  DAG，未捕获目标模型内部保持 config-only/opaque；
- KV/recurrent state 作为 state-in/state-out rail 表达，避免把跨 token 反馈回路
  混入单步严格 DAG；weight 边默认隐藏；
- 首个画布保持完全离线和只读，支持 node card、input/output port、Tensor edge、
  pan/zoom/fit/minimap、折叠下钻、路径高亮和 Inspector 联动；
- 可下钻节点必须有显式 `L1 ›` 入口；Scenario/Hardware 相关的理论热力作为报告
  overlay 计算，不写回渲染中立 GraphView，也不把公式下界冒充实测 latency；
- 64/78 层、256 experts 与未来 raw op 图采用 pattern/group 和按需展开，默认
  每级不超过 200 个可见节点；不把“一次绘制全部节点”作为理解模型的默认交互；
- 官方 Model Explorer-compatible JSON 继续保留为次级输出，consumer Partial
  不阻塞自包含主画布；未来渲染器可以更换，GraphView contract 保持独立；
- M3.5 延续 DR-0007：不读取权重、不构造或执行完整目标模型、不运行 profiler。

## 后果

- M3.5 首先需要补可信的 semantic activation/route/control 投影，而不只是更换 CSS；
- 画布可以从同一 GraphView 支持不同 renderer，也能对稳定 ID、port 引用、无环主
  data flow 和 byte-identical artifact 做机器验收；
- config 语义 DAG 不等同真实目标模型算子调度；报告、节点和 Inspector 必须持续
  展示 origin、coverage 和 Unknown/opaque 边界；
- 10k raw-op 的 WebGL/LOD 仍需专项性能基准，不能由 M3.5 的折叠视图结果外推。

## 验证

- Qwen/GLM GraphView schema/golden、引用完整性与主 data/route/control DAG 检查；
- Qwen Input→Embedding→Decoder→Norm→Head→Logits 可达，Vision/MTP 分支可追溯；
- 每个 input port 至多一个 producer；MTP 使用 decoder hidden 数据输入并输出独立 Draft Logits，不伪造 control Tensor；
- Qwen Linear/Full Attention 与 GLM Dense/Sparse MoE L1 port/edge 语义正确；
- 断网报告完成缩放、下钻、搜索/Layer Strip 定位、节点/端口/边 Inspector 联动；
- 默认 UI 使用 graph-first 布局：中央 DAG 常驻，Browse/Inspector 为按需抽屉，
  Layers/Supporting analysis 默认折叠；搜索命中只更新 Inspector 待查看状态，
  不自动遮挡当前图；
- Qwen 只对明确归属的 Decoder/Attention/FFN 展示公式热度；GLM executed FLOPs/
  logical bytes 未知时保持 Unknown，禁止以 active parameters、存储量或 0 代替；
- manifest 持续断言 weights/full-model/full-forward 均为 false。

验收结果：`scripts/verify_milestones.py` 24/24 PASS，Python 3.9/3.12
各 150 passed；Qwen/GLM 自包含报告的 DAG-01–DAG-13 真实浏览器操作通过，
console 无 warning/error。该结果只覆盖折叠 L0/L1 语义图，不外推到 10k raw-op。

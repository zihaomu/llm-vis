# LLM Vis 项目定位

> 状态：Accepted v1.0  
> 日期：2026-08-31  
> 项目名：**LLM Vis**  
> GitHub 仓库名：`llm-vis`  
> 产品类别：**Weight-Free Recursive LLM Graph Explorer**

本文档定义 LLM Vis 的产品定位、目标用户、核心价值、对外表述和能力边界。它用于指导 README、GitHub About、首页文案、UI 信息架构和后续功能取舍；具体实现计划、数据契约与测量口径仍分别以[完整规划](llm-visualization-plan.md)、[Model Map IR](model-map-ir.md)和[测量协议](measurement-protocol.md)为准。

## 1. 定位结论

LLM Vis 不是泛化的“AI 图表工具”，也不是模型运行器或运行时 profiler。它的核心定位是：

> **一个无需加载权重、无需运行完整模型的 LLM 架构与计算图浏览器。用户可以从模型级 DAG 递归下钻到 Attention、FFN，再到 GEMM、MatMul、Softmax、RMSNorm 等基础算子，并沿途查看 Tensor、State、Shape、证据来源与理论计算瓶颈。**

英文定位：

> **LLM Vis is a weight-free LLM architecture and compute graph explorer. Unfold a model from its top-level DAG to primitive operators, follow tensor and state flows, and inspect theoretical compute and memory pressure—without running the full model.**

品牌口号：

> **See the model. Follow the tensors. Unfold the operators. No weights required.**

中文口号：

> **看见模型，追踪 Tensor，展开算子，无需加载权重。**

## 2. 用户问题

大型 LLM 的结构分析存在四个同时发生的问题：

1. 模型过大，普通开发环境无法方便地下载权重、构造完整模型或执行 forward；
2. 固定教学图过度简化，无法表达 Qwen Linear/Full Attention、GLM MoE/DSA、KV cache 和 recurrent state 等真实差异；
3. 原始算子图节点过多，缺少从模型语义到基础算子的可理解层次；
4. 真实性边界容易混淆：config 推导、代表块捕获、公式估算、外部 trace 和实测指标常被错误地放在同一可信等级。

LLM Vis 需要帮助用户回答：

- 这个模型由哪些语义组件组成，输入到输出如何连接？
- Decoder 层是完全重复、周期变化，还是包含结构例外？
- Attention、FFN、MoE 和 state 内部分别有哪些基础算子？
- 每条边传递什么 Tensor，它的 shape、dtype 和 state 角色是什么？
- 指定 prefill/decode Scenario 下，哪些区域理论上更偏 compute-bound 或 memory-bound？
- 当前结论来自 config、代表块 capture、公式、外部 trace，还是仍然 Unknown？

## 3. 目标用户

| 用户 | 主要任务 | LLM Vis 提供的价值 |
|---|---|---|
| AI/LLM 学习者 | 从 GEMM、Attention 基础知识继续理解真实模型 | 在同一 DAG 中逐步披露说明、Tensor 和基础算子，不要求先理解框架内部实现 |
| 模型与推理工程师 | 快速审查陌生 Qwen、GLM、Llama 类架构 | 查看层模式、端口、KV/state、MoE 结构、shape 和理论成本 |
| 框架与后端工程师 | 建立语义模块、逻辑算子和未来 Kernel/trace 的关联 | 使用稳定身份、provenance、Unknown 边界和可导入的外部运行证据 |
| 研究者与评审者 | 在无法运行超大模型时核查结构声明 | 获得离线、可分享、可复现且不包含权重的结构报告 |

第一阶段的主要用户是**具备 GEMM、Attention 基础知识的学习者和模型/推理工程师**。专家能力继续存在于同一份证据 DAG 中，但不另外创建一套“专家图”。

## 4. 核心用户路径

```text
Hugging Face config / 本地 config.json
                    ↓
        模型级递归 DAG（L0）
                    ↓
       Decoder / Attention / FFN（L1）
                    ↓
  GEMM / MatMul / Softmax / RMSNorm 等基础算子
                    ↓
       Tensor / Shape / dtype / KV 或 state 流
                    ↓
 Scenario 公式成本与理论 compute/memory pressure
                    ↓
      可选外部 trace overlay（未来，不由本工具运行模型）
```

用户始终操作同一个 DAG 系统：复合节点可以展开，基础算子是叶节点，证据不足的区域保持 opaque。下钻只改变当前可见前沿，不改变节点身份、输入输出契约或事实来源。

## 5. 核心价值主张

### 5.1 无需运行超大模型

默认分析路径只读取配置和受控静态输入，不下载或读取目标模型权重，不构造完整目标模型，也不执行完整 forward。大模型的规模不应成为查看其宏观结构的前提条件。

### 5.2 从模型语义递归展开到基础算子

LLM Vis 不把几万节点一次性铺开。它先展示可理解的模型主路径，再允许用户把 Attention、FFN、MoE 等复合节点递归展开到基础算子，同时保留 breadcrumb、直属母图和稳定身份。

### 5.3 Tensor 与 State 是图的一等公民

节点之间不仅显示“有连接”，还显示输入输出端口、Tensor 名称、shape、dtype，以及 KV cache、recurrent state、route/control 等语义。展开前后边界 Tensor 契约必须保持一致。

### 5.4 理论瓶颈可解释但不冒充实测

热力层只根据显式 Scenario、公式 Metric、HardwareProfile 和当前可见成本前沿生成。它用于提示潜在 compute/memory pressure，不声称是实际 latency、Kernel 时间或硬件利用率。

### 5.5 证据和未知边界始终可见

所有事实和指标都要区分 config、capture、formula/estimated、measured 与 Unknown。Unknown 不能补成 0；GLM DSA、动态 MoE route 等证据不足区域不能通过猜测变成确定结构或运行结果。

### 5.6 本地优先、离线可分享

核心输出是自包含 HTML、Markdown 和 JSON artifact。报告可以断网打开，不向外部服务发送模型信息，也不包含模型权重。

## 6. 产品边界

### 6.1 可以对外宣称

- 可视化 config/adapter 能证明的 LLM 架构与语义计算图；
- 从模型 DAG 下钻到 Attention、FFN 和 semantic primitive；
- 展示 Tensor、shape、dtype、state、route/control 和证据来源；
- 根据明确 Scenario 和 HardwareProfile 展示理论 FLOPs、logical bytes、roofline lower bound 和潜在瓶颈；
- 在不加载目标权重、不运行完整模型的条件下生成离线报告；
- 未来导入用户在目标环境外部生成的 trace，并与稳定图身份关联。

### 6.2 不可以对外宣称

- 当前图等于目标模型一次真实 forward 的精确执行图；
- 理论热力图等于实测 latency、Kernel 时间、HBM 流量或 GPU 利用率；
- config 无法证明的 Tensor shape、融合、调度或 MoE 实际 route 已知；
- Tiny/同构缩小代表块 capture 能证明完整 27B/GLM 模型的性能；
- 没有外部 trace 时已经完成 runtime profiling；
- LLM Vis 能解释模型为什么生成某个答案或展示真实 activation/attention value。

### 6.3 非目标

- 不提供聊天、文本生成或模型托管服务；
- 不以执行工作流为目标，不成为 ComfyUI 类生成管线编辑器；
- 不替代 Nsight、PyTorch Profiler、rocprof 等真实运行时分析器；
- 不默认导入 Hugging Face remote code；
- 不为了生成漂亮图而伪造 Unknown 数据。

## 7. 差异化定位

| 相邻工具类别 | 主要关注点 | LLM Vis 的差异 |
|---|---|---|
| Netron/通用模型文件查看器 | 查看 ONNX、TorchScript 等文件中的静态算子和参数 | 理解 LLM 特有的层模式、Attention、FFN、MoE、KV/state，并提供语义递归下钻 |
| Attention/activation 可视化 | 展示一次真实推理产生的 Token、Head 或激活值 | 默认不执行目标模型，重点是架构、Tensor 契约和证据边界 |
| Runtime profiler | 测量 Kernel、时间线、显存和硬件 counter | 先提供零执行的结构与理论分析；未来只导入外部 trace，不冒充实测 |
| ComfyUI 类 Node 工作流 | 编辑并执行用户定义的生成工作流 | 借鉴节点交互，但展示的是模型自身的计算结构，不是工作流编排 |
| 固定 Transformer 教学图 | 用一张简化图片解释通用结构 | 使用真实模型配置、稳定身份和可操作 DAG 表达模型差异与例外 |

Netron 和 ComfyUI 是交互与视觉参考，不是 LLM Vis 的产品类别。

## 8. UX 定位原则

1. **Graph first**：页面中心始终是当前主图，控制项和补充信息按需出现；
2. **同一证据 DAG**：新手和专家查看同一份图，通过递归展开和 Inspector 获得不同深度的信息；
3. **渐进披露**：复合节点明确显示可展开入口，基础算子和 opaque 节点没有虚假下钻；
4. **核心信息在节点内**：节点优先展示用途、关键 shape/state、成本与证据状态，完整细节进入 Inspector；
5. **Tensor 连接可读**：边显示端口方向、Tensor 名、shape 和 dtype，state/route/control 使用可区分的语义；
6. **上下文不丢失**：breadcrumb、直属母图缩略图和 current minimap 分别承担路径、母图位置和当前视口定位；
7. **重复不是循环**：Layer pattern 可以压缩显示，但不得绘制假回边或暗示共享权重；
8. **热力不重复计费**：只统计当前可见成本前沿，展开父节点后不能同时累计父子成本；
9. **Unknown 显式可见**：缺失证据必须显示原因，不能用冷色或 0 伪装为低成本；
10. **无障碍与键盘可操作**：颜色只作辅助，文字图例、focus、Enter/Space、稳定身份和窄屏路径都属于产品契约。

## 9. 品牌与对外文案

### 9.1 名称

- 正式项目名：**LLM Vis**
- 仓库名：`llm-vis`
- CLI：`llm-vis`
- 完整副标题：**LLM Vis — Weight-Free Recursive LLM Graph Explorer**
- 中文副标题：**无需运行模型的递归 LLM 架构与计算图浏览器**

`LLM Vis` 保持直观、可搜索，并与现有代码、CLI 和 artifact 命名一致；通过副标题与口号解决名称较通用的问题。

### 9.2 GitHub About

> **Weight-free LLM architecture explorer with recursive DAGs, tensor flows, operator decomposition, and theoretical bottleneck heatmaps.**

### 9.3 README/Home Hero

标题：

> **See inside an LLM—without loading it.**

说明：

> **Explore the architecture as a recursive DAG, unfold complex blocks into primitive operators, follow tensor and state flows, and inspect theoretical compute and memory pressure.**

### 9.4 推荐关键词

`llm`、`model-visualization`、`transformer`、`compute-graph`、`dag`、`attention`、`mixture-of-experts`、`tensor`、`performance-modeling`、`offline-first`

### 9.5 推荐用语与避免用语

| 推荐 | 避免 |
|---|---|
| theoretical bottleneck / potential pressure | measured bottleneck（没有 trace 时） |
| config semantic projection | exact runtime graph |
| weight-free / zero full forward | model profiling（没有外部 trace 时） |
| recursive operator decomposition | exact Kernel schedule |
| Unknown with reason | zero / cold / free（数据未知时） |
| external trace import | LLM Vis runs the model |

## 10. 功能取舍准则

一个新功能只有在至少强化以下一项且不破坏真实性边界时，才符合当前定位：

1. 让用户更快理解一个真实 LLM 的整体架构；
2. 让复合节点到基础算子的下钻更完整、可验证；
3. 让 Tensor/state/route 的输入输出关系更清晰；
4. 让理论成本、覆盖率和 Unknown 原因更可解释；
5. 让离线 artifact 更稳定、可分享、可复现；
6. 让未来外部 runtime 证据能安全关联到现有稳定身份。

若功能要求默认下载权重、运行完整模型、执行 remote code，或者用猜测填补未知数据，则不属于当前核心产品路径，必须通过新的 Decision Record 明确复审。

## 11. 定位成功标准

一个首次使用 LLM Vis、只具备 GEMM 和 Attention 基础知识的用户，应能在一份报告中完成以下任务：

- 说出模型从输入到输出的主要组件；
- 理解 Decoder 层模式及 L/A/D/M 等符号的具体含义；
- 点击进入 Attention 或 FFN，并继续看到基础算子；
- 沿边解释关键 Tensor 的 shape、dtype 和 state 角色；
- 找到理论计算或访存压力较高的区域，并理解这不是实测 latency；
- 区分 config 事实、代表块 capture、公式估算、外部实测和 Unknown；
- 在全过程中不需要加载目标模型权重或执行完整 forward。

当以上路径成立时，LLM Vis 才不仅是一张模型图，而是一个可信、可操作、适合大型模型的结构理解工作台。

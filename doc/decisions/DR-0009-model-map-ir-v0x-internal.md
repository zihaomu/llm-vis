# DR-0009：Model Map IR v0.x 暂不作为公开稳定规范

- 状态：Accepted
- 日期：2026-08-29
- 决策者角色：Tech Lead
- 复审点：M3 MVP 完成后的首个兼容性评审，最晚在 M5 adapter API 冻结前

## 背景

Model Map IR v0.1 已承担 config 结构、代表块 capture、Scenario、Metric、Coverage 与 Diagnostic 的统一交换格式，但 M4 trace correlation、M5 adapter 插件和跨 revision diff 尚未验证。现在承诺长期公开兼容会过早冻结字段和身份语义。

## 决策

- v0.x schema、JSON artifact 和 Python model 对项目用户可见、可验证、可导出，但标记为实验性内部规范；
- 同一个 v0.x schema version 内，golden、稳定 ID、canonical key 和 JSON Schema 必须保持确定性；
- 破坏性变化必须提升 schema version、提供迁移说明，并保留旧 artifact 的只读验证路径；
- 不宣称第三方 adapter ABI、长期向后兼容或独立标准已经冻结；
- M3 后使用真实 artifact、M4 trace importer 设计与至少一个外部 adapter spike 评估是否发布 v1 候选。

## 后果

- M0–M3 可以围绕严格 v0.1 schema 完成端到端 MVP，而不把未验证的 runtime 字段永久化；
- README、报告和 schema 的版本必须可见，消费者不得假定 `0.1` 等同稳定 v1；
- 任何跨版本 diff 先经过显式迁移，不能只凭字段名猜测等价性。

## 验证

- checked-in Draft 2020-12 schema 与 Pydantic model 一致；
- 三份结构 golden 和固定 Scenario golden 均能被当前 IR 解析；
- 同一输入产生 byte-identical artifact；
- schema version 变化会改变 analysis provenance，并在 Decision Record 中说明。

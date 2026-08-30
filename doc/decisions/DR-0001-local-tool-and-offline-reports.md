# DR-0001：本地工具与离线静态报告

- 状态：Accepted
- 日期：2026-08-29
- 决策者角色：Product / Tech Lead
- 复审点：M5 或出现明确的托管服务需求时

## 背景

LLM-Vis 需要处理模型源码、revision、结构派生数据和后续 profiler trace。MVP 的核心价值是可复现的本地分析，不需要账户、多租户、远程队列或托管存储。离线环境也是模型与基础设施团队的常见约束。

## 决策

MVP 的分发形态固定为：

1. 在用户机器上运行的本地 CLI/Python 工具；
2. 可离线打开、可复制分享的静态 HTML、Markdown 和 JSON artifact；
3. 可选的本机只读浏览入口，默认只绑定 loopback；
4. 不建设 LLM-Vis 托管分析服务。

静态报告使用内嵌或相对资源。打开报告不得依赖 CDN、远程字体、第三方脚本、遥测或网络 API；外部官方链接仅作为可点击 provenance，不影响核心内容显示。

## 后果

- M1 先形成确定性的 JSON/Markdown，MVP 内补齐静态 HTML；报告能力随 M2/M3 的证据增加。
- artifact 是主交付物；UI 和 Model Explorer 都是它的消费者。
- 不实现账户、鉴权、多租户、远程上传、后台任务队列或服务端持久化。
- `serve` 若存在，只是查看本地 artifact，不是远程服务，也不得默认监听公网接口。
- 报告必须包含 revision、工具/adapter/schema 版本、Scenario、Metric 来源、Coverage、Diagnostic 和安全策略。
- 报告不得内嵌模型权重，也不得为了显示而复制受限制的完整模型源码。

## 被拒绝的方案

- MVP 优先建设远程 SaaS：扩大安全、隐私和运维范围，不能提高 config-first 纵切的验证质量。
- 只提供在线交互页面：不满足可复现和离线分享要求。
- 只导出截图：丢失结构化 provenance、搜索和机器校验能力。

## 验证

在断网环境中打开 artifact 的 HTML/Markdown/JSON，核心图、表、告警和 provenance 摘要必须可用；浏览器网络面板不应出现外部请求。


# DR-0002：MVP 禁止执行 Hugging Face remote code

- 状态：Accepted
- 日期：2026-08-29
- 决策者角色：Security / Tech Lead
- 复审点：M5；只有审计通过的 sandbox 才能触发复审

## 背景

Hugging Face config 可能通过 `auto_map` 指向模型仓库中的 Python 实现，并要求 `trust_remote_code=True`。导入 configuration/modeling 模块等同于执行不可信 Python。普通子进程可以隔离崩溃，却不能阻止读取凭据、访问宿主文件或联网，因此不是安全边界。

## 决策

MVP 无条件禁止执行 remote code：

- 不设置或透传 `trust_remote_code=True`；
- 不导入远程 `configuration_*.py`、`modeling_*.py`、`modular_*.py` 或其依赖；
- 不通过 `exec`、`eval`、动态 import、普通 Python worker 或 notebook 绕过限制；
- `auto_map` 只作为数据读取并写入 manifest/Diagnostic；
- 可以读取固定 revision 的 JSON、模型卡、许可证、Python 源码文本和文件清单，计算哈希并做静态索引；
- 需要 remote code 才能解析的模型停留在 C0，或使用经过人工实现和测试的纯数据 config adapter。

CLI 不提供有效的 `--trust-remote-code` 开关。若用户传入同名未知参数，必须失败，而不是静默忽略后继续执行。

## 本地代码边界

用户显式提供的本地 Python import path 可被视为“用户选择执行的本地代码”，但必须明确记录路径、commit/哈希、入口 class 和执行方式。工具不得把刚下载的远程仓库自动改名为“本地代码”来绕过本 DR。

## 数据内容安全

模型源码、config 字符串、trace、IR label 和 metadata 均按不可信数据处理。报告必须转义 HTML；不得将模型提供的文本作为脚本、模板表达式或 HTML 直接注入。

## 后果

- 某些新架构只能得到 config/source inventory 或手写 adapter 的静态结构，这是可接受的降级。
- remote-code 拒绝必须产生结构化 `REMOTE_CODE_EXECUTION_DISABLED` Diagnostic，并保留已获得的事实。
- 哈希和固定 revision 提供可复现性，不提供执行安全性。
- 不承诺通过普通 subprocess、虚拟环境或容器默认配置获得安全隔离。

## 未来复审门槛

只有新的 DR 证明并审计以下能力后，才可讨论显式 opt-in：OS/container/VM 级隔离、默认禁网、无宿主凭据、受限文件系统、CPU/内存/进程/超时上限、依赖供应链控制、输出序列化边界、销毁策略和逃逸测试。任一项缺失则继续禁止执行。


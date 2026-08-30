# DR-0005：冻结首批真实模型 revision

- 状态：Accepted
- 日期：2026-08-29
- 决策者角色：Model / Tech Lead
- 复审点：只有新增 fixture 或显式 cross-revision 测试时

## 决策

首批 golden、示例和 adapter 验收固定使用以下 Hugging Face commit，禁止使用浮动 `main`：

| 模型 | 固定 revision | 官方链接 | 固定 config SHA-256 |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | [revision tree](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) · [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `zai-org/GLM-5.3-BF16` | `304b8051cfb2b260b61ce0cbe330e02a98e73639` | [revision tree](https://huggingface.co/zai-org/GLM-5.3-BF16/tree/304b8051cfb2b260b61ce0cbe330e02a98e73639) · [config.json](https://huggingface.co/zai-org/GLM-5.3-BF16/blob/304b8051cfb2b260b61ce0cbe330e02a98e73639/config.json) | `ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca` |

SHA-256 是对固定 revision 解析得到的 `config.json` 原始字节计算的值。它用于检测 fixture/缓存漂移，不替代 revision，也不表示允许执行仓库代码。

## 范围

本 DR 冻结模型仓库 config/source snapshot 身份，不下载或冻结模型权重。M0–M3 的分析遵循零权重边界。Transformers、PyTorch、ROCm 和 backend 版本是独立依赖；其中 AMD reference stack 尚未冻结，见 [DR-0006](DR-0006-amd-reference-stack-blocks-m4a.md)。

## 验收摘要

Qwen golden 以该 revision 的 64 个文本层、`Linear × 3 + Full × 1` 周期、Vision/Text、KV/recurrent state 和 MTP 为准。GLM golden 以 78 层、前 3 层 Dense、后 75 层 MoE、256 routed experts、top-8 和 1 个 shared expert 为准。

## 后果

- resolver 必须把 branch/tag 解析成完整 commit 并写入 artifact；golden 不接受 `main`。
- fixture 只能保存必要 config、哈希和派生 metadata，不重分发权重。
- 上游更新不会静默改变测试；支持新 revision 需要新 fixture 和显式兼容性结果。


# Security Policy

## Supported versions

LLM Vis is currently preparing its first public release. Security fixes are
applied to the `main` branch and to the latest published release once release
tags exist. Older development snapshots are not maintained separately.

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/zihaomu/llm-vis/security/advisories/new).
Include the affected version or commit, the input needed to reproduce the
problem, its impact, and any suggested mitigation. Do not include access
tokens, private model configuration, proprietary traces, or other secrets.

If private advisories are unavailable, open a minimal public issue that asks
the maintainer to establish a private contact channel. Do not publish exploit
details in that issue.

## Security boundary

LLM Vis treats model configuration, source metadata, trace strings, node
labels, and report content as untrusted data. Reports must escape these values
and must not interpret them as executable HTML or JavaScript.

The supported target-model path:

- does not download or read target-model weights;
- does not construct or execute the complete target model;
- does not execute Hugging Face remote code or enable `trust_remote_code`;
- restricts remote configuration fetches to trusted Hugging Face HTTPS hosts;
- pins remote branches/tags to immutable revisions and records the config hash;
- bounds remote, local, and inline JSON by size, nesting depth, and item count;
- writes self-contained reports that make no third-party request when opened.

The optional representative-capture dependency executes only project-owned
Tiny/meta blocks. It is not authorization to import or run arbitrary model
code. A Python subprocess is fault isolation, not a security sandbox.

Theoretical cost and pressure values are not runtime measurements. Without an
explicit external trace, runtime metrics must remain `Unknown`.

## Out of scope

- Vulnerabilities in third-party model repositories or Hugging Face itself.
- Claims that require LLM Vis to execute untrusted remote Python code.
- Performance-estimate accuracy reports that do not cross a documented
  evidence or safety boundary.

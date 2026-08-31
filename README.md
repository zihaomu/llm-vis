# LLM Vis

[![CI](https://github.com/zihaomu/llm-vis/actions/workflows/ci.yml/badge.svg)](https://github.com/zihaomu/llm-vis/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)

**Weight-Free Recursive LLM Graph Explorer**

> See the model. Follow the tensors. Unfold the operators. No weights required.

LLM Vis turns a Hugging Face `config.json` into an interactive, top-to-bottom model DAG. Start with the model architecture, open Attention, FFN, or MoE blocks, follow tensor and state edges, and inspect theoretical compute and memory pressure—all without loading the target model's weights or running a full forward pass.

## Why LLM Vis?

Very large models are difficult to inspect directly: their weights may not fit locally, generic operator graphs are overwhelming, and static Transformer diagrams hide model-specific structure. LLM Vis provides a middle ground between a teaching diagram and a runtime profiler.

- **Graph first.** Read the architecture from input to output in a Netron-style vertical DAG.
- **Recursive, not flattened.** Double-click a compound node to enter a real child graph, from Attention or FFN down to semantic primitives such as GEMM, MatMul, Softmax, RMSNorm, activation, and Multiply.
- **Explanations in context.** Single-click any node to see its purpose, a simplified equation, input/output shape and dtype, child graphs, and evidence boundary.
- **Tensor and state flow.** Edges distinguish data, KV cache, recurrent state, route, and control semantics with explicit ports.
- **Theoretical bottleneck views.** Compare Pressure, Compute, and Memory heatmaps for a selected prefill or decode scenario. Heat is formula-derived and never presented as measured latency.
- **Evidence-aware by design.** Unsupported or unverifiable regions remain `Unknown` or opaque instead of being guessed or silently treated as zero.
- **Local and shareable.** Each run produces a self-contained offline HTML report plus deterministic JSON and Markdown artifacts.

## Quick start

LLM Vis is currently installed from source. It requires Python 3.9 or newer; [uv](https://docs.astral.sh/uv/) is the recommended environment manager.

```bash
git clone https://github.com/zihaomu/llm-vis.git
cd llm-vis
uv sync --locked

# Fetch config metadata only, then open the generated local report.
uv run llm-vis view Qwen/Qwen3.8-27B
```

The command resolves the requested Hugging Face revision to an immutable commit, records the `config.json` SHA-256, writes an artifact to a unique temporary directory, and opens its self-contained report. It does **not** download model weights.

For a deterministic demo that requires no network access:

```bash
uv run llm-vis view tests/fixtures/configs/qwen3_8_27b.json \
  --local-files-only \
  --output artifacts/qwen-demo
```

To install with standard Python tooling instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
llm-vis view Qwen/Qwen3.8-27B
```

## One input, one report

`llm-vis view` accepts a model ID, model page URL, `config.json` URL, local file or directory, inline JSON, or JSON from stdin.

```bash
# Hugging Face model IDs and URLs
uv run llm-vis view gpt2
uv run llm-vis view https://huggingface.co/Qwen/Qwen3.8-27B
uv run llm-vis view \
  https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json

# Local config file or model directory
uv run llm-vis view ./config.json
uv run llm-vis view ./downloaded-model-directory

# Inline JSON or stdin; --no-open is useful in CI/headless environments
uv run llm-vis view '{"model_type":"future_model"}' --no-open
printf '%s' '{"model_type":"future_model"}' | uv run llm-vis view - --no-open
```

Known adapters produce the detailed recursive DAG. A valid but unsupported configuration still produces a conservative C0/opaque report, with its limited evidence coverage shown explicitly; it does not crash or invent internals.

## Reading the graph

| Action | Result |
|---|---|
| Single-click a node | Open **Explain** with purpose, simplified equation, I/O contracts, child graphs, cost coverage, and evidence notes |
| Double-click a compound node | Enter its child DAG while preserving stable node identity and parent context |
| Double-click a primitive or opaque node | Keep the current view and explain why no child graph exists |
| Select Pressure / Compute / Memory | Apply a scenario-aware theoretical heatmap to the current visible cost frontier |
| Search or choose a hotspot | Focus and highlight the matching node without changing its identity |
| Use Parent context / minimap | See where the current child graph sits in its parent and where the viewport sits in the current graph |

Layer summaries such as `[L×3 → A] ×16` compress repeated structure for readability; they are not loop edges and do not imply weight sharing. Exact layer instances remain available in the report.

## Evidence, not guesswork

LLM Vis keeps structural facts, theoretical estimates, and future runtime evidence separate.

| Evidence class | What it can establish |
|---|---|
| Config semantic projection | Model blocks, repeated-layer patterns, config-provable shapes/state, and adapter-backed connectivity |
| Bounded representative capture | Logical operations and tensor contracts for project-owned Tiny or shape-compatible representative blocks using FakeTensor/meta inputs |
| Formula / hardware profile | Scenario-dependent FLOPs, logical bytes, storage, pressure, and roofline lower bounds |
| External runtime trace | Reserved for imported measurements; without a trace, runtime values remain `Unknown` |

The DAG is therefore an explainable model projection, not a claim that LLM Vis reproduced the target model's exact kernel schedule or executed a real inference.

## Current architecture coverage

| Configuration family | Current visualization |
|---|---|
| Qwen hybrid models (`qwen3_5`, `qwen3_5_text`) | Detailed Linear Attention / Full Attention pattern, KV cache, recurrent state, Attention and FFN operator views |
| GLM MoE + DSA (`glm_moe_dsa`) | Dense and static MoE structure, Router / TopK / Expert / Shared Expert views; unverifiable DSA regions stay opaque and no actual expert route is fabricated |
| Dense Llama / Qwen2 (`llama`, `qwen2`) | Config-first dense decoder structure |
| Other valid JSON configs | Generic C0 skeleton with opaque architecture and explicit partial/unknown coverage |

Adapters are intentionally conservative. See the [adapter guide](doc/adapter-guide.md) for the evidence required to add another architecture family.

## Safety and privacy boundary

The default and acceptance paths are designed for large models that should not be executed merely to inspect their structure.

- No target-model weight download or read.
- No construction of the complete target model and no full model forward.
- No Hugging Face remote-code execution or `trust_remote_code=True` path.
- Remote inputs are restricted to trusted HTTPS Hugging Face hosts; branches and tags are pinned to immutable revisions.
- Remote, local, and inline JSON are limited to 2 MiB and a maximum nesting depth of 64.
- Generated HTML is self-contained and makes no third-party requests when opened.
- Static pressure and roofline values are theoretical. Without an imported external trace, runtime metrics remain `Unknown/null`.

The optional `capture` extra installs PyTorch only for bounded project-owned representative blocks; it does not broaden the target-model execution boundary.

## Artifacts

An analysis directory contains the report and the evidence used to render it:

```text
<artifact>/
├── manifest.json          # input identity, revision, hashes, safety declaration
├── model-map.json         # canonical model facts and stable IDs
├── graph-view.json        # renderer-neutral recursive DAG projection
├── scenarios.json         # prefill/decode workload assumptions
├── metrics.json           # values, origins, coverage, and Unknowns
├── diagnostics.json       # limitations and failures that remain visible
├── layer-strip.json       # exact ordered layer identities
└── reports/
    ├── report.html        # self-contained interactive UI
    └── report.md          # portable text summary
```

Additional files such as captures, hotspots, roofline results, and workload diffs appear when the corresponding evidence is available. Reports never contain model weights.

## Development

Install the locked development environment, including the optional representative-capture dependencies:

```bash
uv sync --locked --group dev --extra capture
```

Run the same core checks used by CI:

```bash
uv run --frozen --group dev --extra capture ruff check .
uv run --frozen --group dev --extra capture pytest -q
uv run --frozen --group dev --extra capture \
  python scripts/verify_milestones.py --milestone all
```

CI covers Python 3.9 and 3.12. The optional capture path uses Torch; config-first `view` and `inspect` require only the core package dependencies.

## Documentation

- [Project positioning](doc/project-positioning.md) — audience, product promise, terminology, and non-goals
- [Implementation plan](doc/llm-visualization-plan.md) — milestone status and acceptance criteria
- [Model Map IR](doc/model-map-ir.md) — canonical schema, stable identity, and provenance
- [UI wireframe](doc/ui-wireframe.md) — graph interaction and responsive behavior
- [Measurement protocol](doc/measurement-protocol.md) — theoretical versus measured metrics
- [Decision records](doc/decisions/README.md) — frozen boundaries and their review conditions

## Project status

LLM Vis is at `0.1.0` and under active development. Config-first import, recursive Qwen/GLM DAGs, explainable node details, scenario-based theoretical heatmaps, and offline reports are implemented. Runtime trace import and hardware-backed operator-to-kernel mappings remain roadmap work; the [implementation plan](doc/llm-visualization-plan.md) is the source of truth.

Focused bug reports and architecture-adapter contributions are welcome through [GitHub Issues](https://github.com/zihaomu/llm-vis/issues).

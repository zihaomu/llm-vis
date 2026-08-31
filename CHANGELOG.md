# Changelog

All notable changes to LLM Vis are recorded here. The project follows
[Semantic Versioning](https://semver.org/) once public release tags begin.

## Unreleased

The first public release is being prepared as `v0.1.0`.

### Added

- One-input model import from a Hugging Face model ID or URL, a local
  `config.json`/directory, inline JSON, or stdin.
- Config-first adapters for Qwen hybrid attention, GLM MoE/DSA, and dense
  Llama/Qwen2-style models, with a conservative opaque fallback for unknown
  configurations.
- Renderer-neutral recursive GraphView artifacts and self-contained HTML,
  Markdown, and JSON reports.
- Top-to-bottom DAG rendering with tensor/state/route/control edges, stable
  ports, recursive child graphs, parent context, and a current-view minimap.
- Single-click node explanations, simplified semantic equations, I/O
  contracts, evidence boundaries, and explicit child-graph choices.
- Scenario-based formula cost, visible-frontier Compute/Memory heatmaps, and
  HardwareProfile-gated theoretical Pressure.
- Python 3.9 and 3.12 CI, schema/golden verification, and real Chromium report
  interaction coverage.

### Security

- Target-model analysis remains zero-weight, zero remote-code execution, and
  zero complete-model forward.
- Remote configuration resolution is restricted to trusted Hugging Face HTTPS
  hosts and immutable revisions; JSON size, depth, and item counts are bounded.

### Known limitations

- The detailed recursive graph is adapter-backed. Unknown architectures remain
  partial and opaque instead of receiving inferred internals.
- Formula heatmaps are theoretical and are not measured latency, physical HBM
  traffic, or runtime profiling.
- Runtime trace import and AMD kernel/counter correlation are not part of the
  first public release.

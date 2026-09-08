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
- Accessible Dark/Light report themes with system-first selection, local
  preference persistence, and state-preserving DAG/heatmap repainting.
- Python 3.9 and 3.12 CI, schema/golden verification, and real Chromium report
  interaction coverage.

### Changed

- Default report output is now kept in
  `<llm-vis-checkout>/artifacts/generated/<model-slug>-<config-hash8>` when the
  command runs inside a verified LLM Vis Git checkout, with `-2`, `-3`, and
  later suffixes used to avoid overwriting an existing artifact. Outside that
  checkout, omitting `--output` returns an actionable error rather than writing
  into an unrelated directory; an explicit `--output` remains available and
  unchanged.

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

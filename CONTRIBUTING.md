# Contributing to LLM Vis

Thanks for helping make large-model architecture easier to understand without
requiring model weights or a complete forward pass.

## Development setup

LLM Vis requires Python 3.9 or newer. The repository uses
[uv](https://docs.astral.sh/uv/) for locked development environments.

```bash
git clone https://github.com/zihaomu/llm-vis.git
cd llm-vis
uv sync --locked --group dev --extra capture
```

Run the same core checks used by CI:

```bash
uv run --frozen --group dev --extra capture ruff check .
uv run --frozen --group dev --extra capture pytest -q
uv run --frozen --group dev --extra capture \
  python scripts/update_structure_goldens.py --check
uv run --frozen --group dev --extra capture \
  python scripts/verify_milestones.py --milestone all
```

Browser tests use Chromium and are kept in a separate CI job. Install its
binary and run the focused suite with:

```bash
uv run playwright install chromium
LLM_VIS_BROWSER_E2E=1 uv run pytest -q tests/browser
```

## Pull requests

1. Open an issue for large behavior, schema, or evidence-boundary changes.
2. Keep each pull request focused and explain the user-visible outcome.
3. Add tests for success, fallback, and safety behavior.
4. Update schemas/goldens only through the checked-in update scripts, then
   review the semantic diff before committing it.
5. Update the implementation plan and relevant decision record when changing a
   frozen product, evidence, or execution boundary.
6. Confirm that Ruff, pytest, goldens, milestone verification, and relevant
   browser tests pass.

## Architecture adapters

An adapter may expose only facts supported by configuration, a documented
semantic contract, or bounded representative capture. It must:

- use a fixed config fixture/revision and record provenance;
- preserve stable node, port, edge, and view identity;
- keep unsupported internals opaque and `Unknown` rather than guessing;
- include GraphView/structure goldens and invalid-config/fallback tests;
- preserve zero target weights, zero remote code, and zero complete-model
  forward;
- document Tensor/state shapes, route/control semantics, and cost coverage.

See the [adapter guide](doc/adapter-guide.md) and
[Model Map IR](doc/model-map-ir.md) before adding a model family.

## Reporting bugs

Include the LLM Vis commit/version, input kind (never secrets), config hash when
available, the command used, the generated diagnostic, and expected behavior.
Use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities.

Unless explicitly marked otherwise, contributions intentionally submitted for
inclusion are accepted under the repository's Apache License 2.0 terms.

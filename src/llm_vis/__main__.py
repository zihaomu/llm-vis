"""Allow ``python -m llm_vis`` to run the command line interface."""

from __future__ import annotations

from llm_vis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

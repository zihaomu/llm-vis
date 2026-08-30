from __future__ import annotations

import sys

from llm_vis.meta import MetaBuildBudget, run_budgeted_command


def test_budgeted_worker_returns_json() -> None:
    result = run_budgeted_command(
        [sys.executable, "-c", "import json; print(json.dumps({'modules': 3}))"],
        budget=MetaBuildBudget(timeout_seconds=2, memory_mib=256),
    )
    assert result.succeeded
    assert result.payload == {"modules": 3}


def test_timeout_is_a_fallback_result_not_an_exception() -> None:
    result = run_budgeted_command(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        budget=MetaBuildBudget(timeout_seconds=0.05, memory_mib=256),
    )
    assert result.status == "timed_out"
    assert result.diagnostic_code == "META_BUDGET_TIMEOUT"


def test_invalid_worker_output_is_diagnostic() -> None:
    result = run_budgeted_command(
        [sys.executable, "-c", "print('not json')"],
        budget=MetaBuildBudget(timeout_seconds=2, memory_mib=256),
    )
    assert result.status == "failed"
    assert result.diagnostic_code == "META_WORKER_INVALID_OUTPUT"

"""Run optional meta-model probes behind time and memory budgets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class MetaBuildBudget:
    timeout_seconds: float = 30.0
    memory_mib: int = 2048

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.memory_mib <= 0:
            raise ValueError("memory_mib must be greater than zero")


@dataclass(frozen=True)
class BudgetedCommandResult:
    status: str
    returncode: Optional[int]
    payload: Optional[Dict[str, Any]]
    stderr: str
    diagnostic_code: Optional[str]

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


def _memory_limiter(memory_mib: int):  # type: ignore[no-untyped-def]
    # RLIMIT_AS is reliable for this purpose on Linux. macOS exposes the symbol
    # but rejects or inconsistently applies it to a freshly exec'd interpreter,
    # so time isolation remains active there and memory capability is reported
    # separately by higher-level manifests.
    if os.name != "posix" or not sys.platform.startswith("linux"):
        return None

    def apply_limit() -> None:
        import resource

        byte_limit = memory_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))

    return apply_limit


def run_budgeted_command(
    command: Sequence[str], *, budget: Optional[MetaBuildBudget] = None
) -> BudgetedCommandResult:
    """Run a JSON-producing worker and turn failures into structured fallback states.

    The command is executed directly, never through a shell. Timeout or worker failure
    is a normal analysis outcome; callers should continue with config-first IR.
    """

    if budget is None:
        budget = MetaBuildBudget()
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain non-empty string arguments")
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=budget.timeout_seconds,
            preexec_fn=_memory_limiter(budget.memory_mib),
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return BudgetedCommandResult(
            status="timed_out",
            returncode=None,
            payload=None,
            stderr=stderr,
            diagnostic_code="META_BUDGET_TIMEOUT",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return BudgetedCommandResult(
            status="failed",
            returncode=None,
            payload=None,
            stderr=str(exc),
            diagnostic_code="META_WORKER_START_FAILED",
        )

    if completed.returncode != 0:
        return BudgetedCommandResult(
            status="failed",
            returncode=completed.returncode,
            payload=None,
            stderr=completed.stderr,
            diagnostic_code="META_WORKER_FAILED",
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return BudgetedCommandResult(
            status="failed",
            returncode=completed.returncode,
            payload=None,
            stderr=f"Worker did not return a JSON object: {exc}",
            diagnostic_code="META_WORKER_INVALID_OUTPUT",
        )
    if not isinstance(payload, dict):
        return BudgetedCommandResult(
            status="failed",
            returncode=completed.returncode,
            payload=None,
            stderr="Worker JSON output must be an object",
            diagnostic_code="META_WORKER_INVALID_OUTPUT",
        )
    return BudgetedCommandResult(
        status="success",
        returncode=completed.returncode,
        payload=payload,
        stderr=completed.stderr,
        diagnostic_code=None,
    )

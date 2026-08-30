# ruff: noqa: UP045
"""Deterministic identifiers for Model Map IR artifacts.

Identifiers are hashes of canonical JSON rather than process-local UUIDs.  This
makes generated artifacts reproducible and gives callers a stable primitive for
caching and cross-artifact matching.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Mapping, Optional

from pydantic import BaseModel

DEFAULT_DIGEST_LENGTH = 24


def _json_compatible(value: Any) -> Any:
    """Return a JSON-compatible representation without losing key ordering.

    Sorting happens in :func:`canonical_json`.  This normalization step keeps
    the public hashing functions useful with Pydantic models and enums while
    deliberately rejecting opaque Python objects.
    """

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__!r} is not JSON-compatible")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` into the canonical JSON used by all IR hashes.

    The representation is UTF-8 friendly, has sorted object keys, contains no
    insignificant whitespace, and rejects NaN/Infinity.
    """

    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deterministic_id(
    namespace: str,
    payload: Any,
    *,
    digest_length: int = DEFAULT_DIGEST_LENGTH,
) -> str:
    """Return ``<namespace>_<truncated sha256>`` for a canonical JSON payload."""

    if not namespace or not namespace.replace("-", "_").isalnum():
        raise ValueError("namespace must contain only letters, digits, '-' or '_'")
    if not 8 <= digest_length <= 64:
        raise ValueError("digest_length must be between 8 and 64 hexadecimal characters")
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{namespace}_{digest[:digest_length]}"


def artifact_local_id(
    *,
    model_revision: str,
    framework_domain: str,
    module_path: str,
    op_ordinal: Optional[int] = None,
    capture_variant: Optional[str] = None,
) -> str:
    """Build an ID that is unique and deterministic inside one artifact."""

    if op_ordinal is not None and op_ordinal < 0:
        raise ValueError("op_ordinal must be non-negative")
    return deterministic_id(
        "art",
        {
            "capture_variant": capture_variant,
            "framework_domain": framework_domain,
            "model_revision": model_revision,
            "module_path": module_path,
            "op_ordinal": op_ordinal,
        },
    )


def canonical_semantic_key(
    *,
    semantic_path: str,
    definition_signature: Any,
    symbolic_contract: Any,
    structural_signature: Any,
) -> str:
    """Build the revision-independent key used for semantic matching."""

    return deterministic_id(
        "sem",
        {
            "definition_signature": definition_signature,
            "semantic_path": semantic_path,
            "structural_signature": structural_signature,
            "symbolic_contract": symbolic_contract,
        },
    )


def scenario_id(scenario: Any) -> str:
    """Build a deterministic ID from Scenario fields or a scenario mapping.

    A supplied top-level ``id`` field is intentionally excluded so round-trips
    do not change the identity payload.
    """

    payload = _json_compatible(scenario)
    if not isinstance(payload, dict):
        raise TypeError("scenario must be a mapping or a Pydantic model")
    payload = dict(payload)
    payload.pop("id", None)
    return deterministic_id("scn", payload)

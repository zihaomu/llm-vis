"""Resolve local and Hugging Face model configurations safely.

Only JSON metadata is read. This module never imports model code and never downloads
weight files. A Hugging Face revision is resolved to an immutable commit before the
configuration is fetched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ResolutionError(RuntimeError):
    """Raised when a model configuration cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedConfig:
    """A resolved configuration and the provenance needed to reproduce it."""

    identifier: str
    revision: str
    config: Mapping[str, Any]
    source_uri: str
    sha256: str
    is_remote: bool
    cache_hit: bool
    license: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _config_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResolutionError(f"Unable to read model config at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResolutionError(f"Model config at {path} must contain a JSON object")
    return value


def _default_hf_cache() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cache_repo_dir(model_id: str, cache_dir: Optional[Path]) -> Path:
    owner, name = model_id.split("/", 1)
    return (cache_dir or _default_hf_cache()) / f"models--{owner}--{name}"


def _resolve_cached_revision(repo_dir: Path, revision: str) -> Optional[str]:
    snapshot = repo_dir / "snapshots" / revision
    if snapshot.is_dir():
        return revision
    ref = repo_dir / "refs" / revision
    if ref.is_file():
        try:
            value = ref.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if value and (repo_dir / "snapshots" / value).is_dir():
            return value
    return None


def _from_hf_cache(
    model_id: str, revision: str, cache_dir: Optional[Path]
) -> Optional[ResolvedConfig]:
    repo_dir = _cache_repo_dir(model_id, cache_dir)
    commit = _resolve_cached_revision(repo_dir, revision)
    if commit is None:
        return None
    config_path = repo_dir / "snapshots" / commit / "config.json"
    if not config_path.is_file():
        return None
    config = _read_json_file(config_path)
    return ResolvedConfig(
        identifier=model_id,
        revision=commit,
        config=config,
        source_uri=f"hf://{model_id}@{commit}/config.json",
        sha256=_config_digest(config),
        is_remote=True,
        cache_hit=True,
        metadata={"requested_revision": revision, "cache_path": str(config_path)},
    )


def _request_json(url: str, timeout_seconds: float) -> Tuple[Dict[str, Any], Mapping[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "llm-vis/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
            headers = dict(response.headers.items())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ResolutionError(f"Unable to fetch metadata from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResolutionError(f"Expected a JSON object from {url}")
    return payload, headers


def _resolve_remote_model(model_id: str, revision: str, timeout_seconds: float) -> ResolvedConfig:
    encoded_model = urllib.parse.quote(model_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    api_url = f"https://huggingface.co/api/models/{encoded_model}/revision/{encoded_revision}"
    metadata, _ = _request_json(api_url, timeout_seconds)
    commit = metadata.get("sha")
    if not isinstance(commit, str) or not commit:
        raise ResolutionError(f"Hugging Face did not return an immutable revision for {model_id}")

    encoded_commit = urllib.parse.quote(commit, safe="")
    config_url = f"https://huggingface.co/{encoded_model}/resolve/{encoded_commit}/config.json"
    config, _ = _request_json(config_url, timeout_seconds)
    card_data = metadata.get("cardData")
    license_name = card_data.get("license") if isinstance(card_data, dict) else None
    return ResolvedConfig(
        identifier=model_id,
        revision=commit,
        config=config,
        source_uri=f"hf://{model_id}@{commit}/config.json",
        sha256=_config_digest(config),
        is_remote=True,
        cache_hit=False,
        license=license_name if isinstance(license_name, str) else None,
        metadata={
            "requested_revision": revision,
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
        },
    )


def _resolve_local(source: Path, revision: Optional[str]) -> ResolvedConfig:
    config_path = source / "config.json" if source.is_dir() else source
    if config_path.name != "config.json" and source.is_dir():
        raise ResolutionError(f"No config.json found below {source}")
    if not config_path.is_file():
        raise ResolutionError(f"Local model config does not exist: {config_path}")
    config = _read_json_file(config_path)
    resolved_revision = revision or str(config.get("_commit_hash") or "local")
    return ResolvedConfig(
        identifier=str(source.resolve()),
        revision=resolved_revision,
        config=config,
        source_uri=str(config_path.resolve()),
        sha256=_config_digest(config),
        is_remote=False,
        cache_hit=True,
        license=config.get("license") if isinstance(config.get("license"), str) else None,
        metadata={
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
        },
    )


def resolve_config(
    source: str,
    *,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[Path] = None,
    timeout_seconds: float = 20.0,
) -> ResolvedConfig:
    """Resolve ``source`` into a config without loading weights or executing Python.

    ``source`` may be a local directory, a local ``config.json`` path, or a Hugging
    Face model ID. Remote requests fetch only repository metadata and ``config.json``.
    """

    local_path = Path(source).expanduser()
    if local_path.exists():
        return _resolve_local(local_path, revision)
    if not _HF_ID_RE.fullmatch(source):
        raise ResolutionError(
            f"{source!r} is neither an existing local path nor a valid owner/model ID"
        )

    requested_revision = revision or "main"
    cached = _from_hf_cache(source, requested_revision, cache_dir)
    if cached is not None:
        return cached
    if local_files_only:
        raise ResolutionError(
            "OFFLINE_CACHE_MISS: "
            f"{source}@{requested_revision} is not present in the local Hugging Face cache"
        )
    return _resolve_remote_model(source, requested_revision, timeout_seconds)

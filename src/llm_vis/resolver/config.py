"""Resolve local and Hugging Face model configurations safely.

Only bounded JSON metadata is read. This module never imports model code and never
downloads weight files. A Hugging Face revision is resolved to an immutable commit
before the configuration is fetched.
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

_HF_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$"
)
_HF_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_HF_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_LOCAL_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})
_HF_RESERVED_OWNERS = frozenset(
    {
        "api",
        "datasets",
        "docs",
        "join",
        "login",
        "models",
        "new",
        "organizations",
        "pricing",
        "search",
        "settings",
        "spaces",
        "tasks",
    }
)

_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_CONTAINER_ITEMS = 100_000


class ResolutionError(RuntimeError):
    """A structured error raised when a config cannot be resolved safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RESOLUTION_ERROR",
        hint: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.hint = hint
        self.details = dict(details or {})
        rendered = f"{code}: {message}"
        if hint:
            rendered = f"{rendered} Hint: {hint}"
        super().__init__(rendered)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready diagnostic."""

        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class ResolvedConfig:
    """A resolved configuration and its reproducible provenance."""

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


def _json_constant_is_invalid(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _validate_json_shape(value: Mapping[str, Any], source_label: str) -> None:
    stack = [(value, 1)]
    total_items = 0
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise ResolutionError(
                f"JSON from {source_label} exceeds the maximum nesting depth "
                f"of {_MAX_JSON_DEPTH}",
                code="JSON_DEPTH_LIMIT",
                hint="Use a regular Hugging Face config.json without deeply nested payloads.",
                details={"limit": _MAX_JSON_DEPTH},
            )
        if isinstance(current, dict):
            total_items += len(current)
            children = current.values()
        elif isinstance(current, list):
            total_items += len(current)
            children = current
        else:
            continue
        if total_items > _MAX_JSON_CONTAINER_ITEMS:
            raise ResolutionError(
                f"JSON from {source_label} contains more than "
                f"{_MAX_JSON_CONTAINER_ITEMS} container items",
                code="JSON_ITEM_LIMIT",
                hint="Use config.json rather than tokenizer data or model weights.",
                details={"limit": _MAX_JSON_CONTAINER_ITEMS},
            )
        stack.extend(
            (child, depth + 1) for child in children if isinstance(child, (dict, list))
        )


def _decode_json_object(raw: bytes, source_label: str) -> Dict[str, Any]:
    if len(raw) > _MAX_JSON_BYTES:
        raise ResolutionError(
            f"JSON from {source_label} is larger than {_MAX_JSON_BYTES} bytes",
            code="JSON_SIZE_LIMIT",
            hint=(
                "Provide config.json only; weight, tokenizer, and generation files "
                "are not accepted."
            ),
            details={"limit_bytes": _MAX_JSON_BYTES, "actual_bytes": len(raw)},
        )
    try:
        text = raw.decode("utf-8-sig")
        value = json.loads(text, parse_constant=_json_constant_is_invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ResolutionError(
            f"Unable to parse JSON from {source_label}: {exc}",
            code="JSON_INVALID",
            hint="Provide a UTF-8 JSON object containing the model configuration.",
        ) from exc
    if not isinstance(value, dict):
        raise ResolutionError(
            f"Model config from {source_label} must contain a JSON object",
            code="JSON_ROOT_NOT_OBJECT",
            hint="Provide config.json, whose top-level value is an object.",
        )
    _validate_json_shape(value, source_label)
    return value


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        declared_size = path.stat().st_size
        if declared_size > _MAX_JSON_BYTES:
            raise ResolutionError(
                f"JSON file {path} is larger than {_MAX_JSON_BYTES} bytes",
                code="JSON_SIZE_LIMIT",
                hint="Provide config.json only; weight and tokenizer files are not accepted.",
                details={"limit_bytes": _MAX_JSON_BYTES, "actual_bytes": declared_size},
            )
        with path.open("rb") as handle:
            raw = handle.read(_MAX_JSON_BYTES + 1)
    except ResolutionError:
        raise
    except OSError as exc:
        raise ResolutionError(
            f"Unable to read model config at {path}: {exc}",
            code="LOCAL_CONFIG_READ_FAILED",
            hint="Check that the path points to a readable JSON file.",
        ) from exc
    return _decode_json_object(raw, str(path))


def _default_hf_cache() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cache_repo_dir(model_id: str, cache_dir: Optional[Path]) -> Path:
    cache_name = model_id.replace("/", "--")
    return (cache_dir or _default_hf_cache()) / f"models--{cache_name}"


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or not _HF_REVISION_RE.fullmatch(revision):
        raise ResolutionError(
            f"Invalid Hugging Face revision {revision!r}",
            code="HF_INVALID_REVISION",
            hint="Use a branch, tag, or commit with letters, digits, '.', '_', '-', and '/'.",
        )
    if any(segment in {"", ".", ".."} for segment in revision.split("/")):
        raise ResolutionError(
            f"Invalid Hugging Face revision {revision!r}",
            code="HF_INVALID_REVISION",
            hint="Revision path segments cannot be empty, '.' or '..'.",
        )
    return revision


def _resolve_cached_revision(repo_dir: Path, revision: str) -> Optional[str]:
    if _HF_COMMIT_RE.fullmatch(revision):
        commit = revision.lower()
        snapshot = repo_dir / "snapshots" / commit
        if snapshot.is_dir():
            return commit
    ref = repo_dir / "refs" / revision
    if ref.is_file():
        try:
            if ref.stat().st_size > 256:
                return None
            value = ref.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        if _HF_COMMIT_RE.fullmatch(value or ""):
            commit = value.lower()
            if (repo_dir / "snapshots" / commit).is_dir():
                return commit
    return None


def _from_hf_cache(
    model_id: str,
    revision: str,
    cache_dir: Optional[Path],
    input_kind: str,
) -> Optional[ResolvedConfig]:
    repo_dir = _cache_repo_dir(model_id, cache_dir)
    commit = _resolve_cached_revision(repo_dir, revision)
    if commit is None:
        return None
    config_path = repo_dir / "snapshots" / commit / "config.json"
    if not config_path.is_file():
        return None
    try:
        config_path.resolve().relative_to(repo_dir.resolve())
    except (OSError, ValueError):
        raise ResolutionError(
            f"Cached config path escapes the Hugging Face repository cache: {config_path}",
            code="HF_CACHE_UNSAFE",
            hint="Remove the unsafe cache entry and fetch config.json again.",
        ) from None
    config = _read_json_file(config_path)
    return ResolvedConfig(
        identifier=model_id,
        revision=commit,
        config=config,
        source_uri=f"hf://{model_id}@{commit}/config.json",
        sha256=_config_digest(config),
        is_remote=True,
        cache_hit=True,
        metadata={
            "input_kind": input_kind,
            "requested_revision": revision,
            "resolved_commit": commit,
        },
    )


def _trusted_hf_url_parts(url: str) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ResolutionError(
            f"Invalid Hugging Face URL: {exc}",
            code="HF_URL_INVALID",
            hint="Paste an https://huggingface.co/owner/model or /model URL.",
        ) from exc
    if parsed.scheme.lower() != "https":
        raise ResolutionError(
            "Only HTTPS Hugging Face URLs are accepted",
            code="HF_URL_UNTRUSTED",
            hint="Use an https://huggingface.co/owner/model or /model URL.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise ResolutionError(
            "Hugging Face URLs cannot contain user information",
            code="HF_URL_UNTRUSTED",
            hint="Remove credentials or userinfo from the URL.",
        )
    host = (parsed.hostname or "").lower()
    if host not in _HF_HOSTS:
        raise ResolutionError(
            f"Untrusted Hugging Face URL host {host!r}",
            code="HF_URL_UNTRUSTED",
            hint="Only huggingface.co and www.huggingface.co are accepted.",
        )
    if port not in {None, 443}:
        raise ResolutionError(
            f"Non-standard Hugging Face URL port {port!r} is not accepted",
            code="HF_URL_UNTRUSTED",
            hint="Remove the custom port from the URL.",
        )
    return parsed


def _parse_hf_input_url(url: str) -> Tuple[str, Optional[str], str]:
    parsed = _trusted_hf_url_parts(url)
    if parsed.query or parsed.fragment:
        raise ResolutionError(
            "Hugging Face input URLs cannot contain a query string or fragment",
            code="HF_URL_QUERY_UNSUPPORTED",
            hint="Copy the canonical model or config.json URL without '?' or '#'.",
        )
    path = parsed.path[:-1] if parsed.path.endswith("/") else parsed.path
    raw_segments = path.split("/")[1:] if path.startswith("/") else []
    if any(not segment for segment in raw_segments):
        raw_segments = []

    if len(raw_segments) in {1, 2}:
        model_segments = [urllib.parse.unquote(segment) for segment in raw_segments]
        url_revision: Optional[str] = None
        input_kind = "hf_model_url"
    elif raw_segments[-1:] == ["config.json"]:
        marker_indices = [
            index
            for index, segment in enumerate(raw_segments[:-1])
            if segment in {"blob", "resolve"}
        ]
        if len(marker_indices) != 1 or marker_indices[0] not in {1, 2}:
            marker_indices = []
        if not marker_indices:
            raise ResolutionError(
                f"Unsupported Hugging Face URL path {parsed.path!r}",
                code="HF_URL_PATH_UNSUPPORTED",
                hint=(
                    "Use a model page URL or a blob/resolve URL ending exactly in "
                    "config.json; weight and arbitrary repository paths are not accepted."
                ),
            )
        marker_index = marker_indices[0]
        model_segments = [
            urllib.parse.unquote(segment) for segment in raw_segments[:marker_index]
        ]
        url_revision = urllib.parse.unquote(
            "/".join(raw_segments[marker_index + 1 : -1])
        )
        _validate_revision(url_revision)
        input_kind = "hf_config_url"
    else:
        raise ResolutionError(
            f"Unsupported Hugging Face URL path {parsed.path!r}",
            code="HF_URL_PATH_UNSUPPORTED",
            hint=(
                "Use a model page URL or a blob/resolve URL ending exactly in config.json; "
                "weight and arbitrary repository paths are not accepted."
            ),
        )

    model_id = "/".join(model_segments)
    owner_or_repo = model_segments[0].lower() if model_segments else ""
    if not _HF_ID_RE.fullmatch(model_id) or owner_or_repo in _HF_RESERVED_OWNERS:
        raise ResolutionError(
            f"Hugging Face URL does not identify a valid model repository: {model_id!r}",
            code="HF_URL_PATH_UNSUPPORTED",
            hint="Use a model URL shaped like https://huggingface.co/owner/model or /model.",
        )
    return model_id, url_revision, input_kind


class _TrustedHFRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject a redirect before urllib can contact an untrusted target."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        _trusted_hf_url_parts(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_trusted_url(request: urllib.request.Request, timeout_seconds: float) -> Any:
    opener = urllib.request.build_opener(_TrustedHFRedirectHandler())
    return opener.open(request, timeout=timeout_seconds)


def _request_json(url: str, timeout_seconds: float) -> Tuple[Dict[str, Any], Mapping[str, str]]:
    _trusted_hf_url_parts(url)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "llm-vis/0.1"},
    )
    try:
        with _open_trusted_url(request, timeout_seconds) as response:
            final_url_getter = getattr(response, "geturl", None)
            final_url = final_url_getter() if callable(final_url_getter) else url
            _trusted_hf_url_parts(final_url)
            declared_length = response.headers.get("Content-Length")
            if declared_length is not None:
                try:
                    length = int(declared_length)
                except (TypeError, ValueError):
                    length = None
                if length is not None and length > _MAX_JSON_BYTES:
                    raise ResolutionError(
                        f"Remote JSON from {url} is larger than {_MAX_JSON_BYTES} bytes",
                        code="JSON_SIZE_LIMIT",
                        hint="The resolver fetches config.json only; verify the model URL.",
                        details={"limit_bytes": _MAX_JSON_BYTES, "actual_bytes": length},
                    )
            raw = response.read(_MAX_JSON_BYTES + 1)
            headers = dict(response.headers.items())
    except ResolutionError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ResolutionError(
                "The Hugging Face repository is private, gated, or not accessible",
                code="HF_AUTH_REQUIRED",
                hint=(
                    "M3.9 does not accept Hugging Face credentials; use a public model or "
                    "download config.json yourself and pass the local file."
                ),
                details={"status": exc.code},
            ) from exc
        raise ResolutionError(
            f"Hugging Face returned HTTP {exc.code} while fetching model metadata",
            code="REMOTE_FETCH_FAILED",
            hint="Confirm that the model, revision, and config.json exist.",
            details={"status": exc.code},
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ResolutionError(
            f"Unable to fetch metadata from {url}: {exc}",
            code="REMOTE_FETCH_FAILED",
            hint="Check network access and confirm that the model and revision are public.",
        ) from exc
    return _decode_json_object(raw, url), headers


def _resolve_remote_model(
    model_id: str,
    revision: str,
    timeout_seconds: float,
    input_kind: str,
) -> ResolvedConfig:
    encoded_model = urllib.parse.quote(model_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    api_url = f"https://huggingface.co/api/models/{encoded_model}/revision/{encoded_revision}"
    metadata, _ = _request_json(api_url, timeout_seconds)
    commit = metadata.get("sha")
    if not isinstance(commit, str) or not _HF_COMMIT_RE.fullmatch(commit):
        raise ResolutionError(
            f"Hugging Face did not return an immutable commit for {model_id}",
            code="HF_IMMUTABLE_REVISION_MISSING",
            hint="Verify that the model and requested branch/tag exist.",
        )

    commit = commit.lower()
    encoded_model = urllib.parse.quote(model_id, safe="/")
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
            "input_kind": input_kind,
            "requested_revision": revision,
            "resolved_commit": commit,
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
        },
    )


def _safe_local_label(config: Mapping[str, Any], source: Path, digest: str) -> str:
    configured_name = config.get("_name_or_path")
    if isinstance(configured_name, str):
        candidate = configured_name.strip()
        if _HF_ID_RE.fullmatch(candidate) or _LOCAL_LABEL_RE.fullmatch(candidate):
            return candidate

    if source.is_dir():
        candidate = source.name
    elif source.name == "config.json":
        candidate = source.parent.name
    else:
        candidate = source.stem
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate).strip("._-")
    return candidate[:128] or f"config-{digest[:12]}"


def _content_identity_label(config: Mapping[str, Any], digest: str) -> str:
    """Choose a path-independent label so local-file and pasted JSON IDs agree."""

    candidates = [config.get("_name_or_path"), config.get("model_type")]
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        candidates.append(architectures[0])
    for value in candidates:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if _HF_ID_RE.fullmatch(candidate) or _LOCAL_LABEL_RE.fullmatch(candidate):
            return candidate
    return f"config-{digest[:12]}"


def _content_revision(
    config: Mapping[str, Any],
    explicit_revision: Optional[str],
    digest: str,
) -> str:
    """Return a non-secret immutable identity for local or pasted config content."""

    content_revision = f"sha256:{digest}"
    if explicit_revision is not None:
        if _HF_COMMIT_RE.fullmatch(explicit_revision):
            return explicit_revision.lower()
        if explicit_revision == content_revision:
            return content_revision
        raise ResolutionError(
            "Local and inline --revision must be an immutable Hugging Face commit or "
            "the config's exact sha256:<digest> identity",
            code="LOCAL_REVISION_NOT_IMMUTABLE",
            hint="Omit --revision to use the config content hash automatically.",
        )

    embedded_revision = config.get("_commit_hash")
    if isinstance(embedded_revision, str) and _HF_COMMIT_RE.fullmatch(embedded_revision):
        return embedded_revision.lower()
    return content_revision


def _resolve_local(source: Path, revision: Optional[str]) -> ResolvedConfig:
    config_path = source / "config.json" if source.is_dir() else source
    if not config_path.is_file():
        raise ResolutionError(
            f"Local model config does not exist: {config_path}",
            code="LOCAL_CONFIG_NOT_FOUND",
            hint="Provide a readable JSON file or a directory containing config.json.",
        )
    config = _read_json_file(config_path)
    digest = _config_digest(config)
    label = _safe_local_label(config, source, digest)
    identity_label = _content_identity_label(config, digest)
    encoded_label = urllib.parse.quote(label, safe="")
    resolved_revision = _content_revision(config, revision, digest)
    return ResolvedConfig(
        identifier=f"config:{identity_label}@{digest[:12]}",
        revision=resolved_revision,
        config=config,
        source_uri=f"local://{encoded_label}/{digest}/config.json",
        sha256=digest,
        is_remote=False,
        cache_hit=True,
        license=config.get("license") if isinstance(config.get("license"), str) else None,
        metadata={
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
            "input_kind": "local_directory" if source.is_dir() else "local_file",
            "local_label": label,
        },
    )


def _resolve_inline_json(source: str, revision: Optional[str]) -> ResolvedConfig:
    config = _decode_json_object(source.encode("utf-8"), "pasted JSON")
    digest = _config_digest(config)
    identity_label = _content_identity_label(config, digest)
    resolved_revision = _content_revision(config, revision, digest)
    return ResolvedConfig(
        identifier=f"config:{identity_label}@{digest[:12]}",
        revision=resolved_revision,
        config=config,
        source_uri=f"inline://sha256/{digest}/config.json",
        sha256=digest,
        is_remote=False,
        cache_hit=False,
        license=config.get("license") if isinstance(config.get("license"), str) else None,
        metadata={
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
            "input_kind": "inline_json",
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
    """Resolve a config without loading weights or executing Python."""

    if not isinstance(source, str) or not source.strip():
        raise ResolutionError(
            "Model config source cannot be empty",
            code="SOURCE_EMPTY",
            hint="Provide a Hugging Face model ID/URL, a local path, or pasted JSON.",
        )

    stripped = source.strip()
    json_candidate = stripped.lstrip("\ufeff")
    if json_candidate.startswith(("{", "[")):
        return _resolve_inline_json(stripped, revision)

    is_url_candidate = "://" in stripped or stripped.lower().startswith(("http:", "https:"))
    url_revision: Optional[str] = None
    if is_url_candidate:
        model_id, url_revision, input_kind = _parse_hf_input_url(stripped)
    else:
        try:
            local_path = Path(source).expanduser()
            is_local = local_path.exists()
        except (OSError, ValueError):
            is_local = False
            local_path = Path(".")
        if is_local:
            return _resolve_local(local_path, revision)
        model_id = stripped
        if not _HF_ID_RE.fullmatch(model_id):
            raise ResolutionError(
                f"{source!r} is neither an existing local path nor a valid model ID",
                code="SOURCE_UNRECOGNIZED",
                hint="Provide model or owner/model, an HTTPS Hugging Face URL, or config.json.",
            )
        input_kind = "hf_model_id"

    if revision is not None:
        requested_revision = _validate_revision(revision)
        if url_revision is not None and requested_revision != url_revision:
            raise ResolutionError(
                f"URL revision {url_revision!r} conflicts with --revision {revision!r}",
                code="HF_REVISION_CONFLICT",
                hint="Remove the explicit revision or make it match the config URL.",
            )
    else:
        requested_revision = url_revision or "main"
    requested_revision = _validate_revision(requested_revision)

    cached = _from_hf_cache(model_id, requested_revision, cache_dir, input_kind)
    if cached is not None:
        return cached
    if local_files_only:
        raise ResolutionError(
            f"{model_id}@{requested_revision} is not present in the local Hugging Face cache",
            code="OFFLINE_CACHE_MISS",
            hint="Populate the cache first or omit local-files-only to fetch config.json.",
        )
    return _resolve_remote_model(model_id, requested_revision, timeout_seconds, input_kind)

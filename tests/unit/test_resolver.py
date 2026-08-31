from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import llm_vis.resolver.config as resolver_config
from llm_vis.resolver import ResolutionError, resolve_config


class _StubResponse:
    def __init__(
        self,
        payload: Any,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._raw = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        self._url = url
        self.headers = (
            headers if headers is not None else {"Content-Length": str(len(self._raw))}
        )

    def __enter__(self) -> _StubResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, limit: int = -1) -> bytes:
        return self._raw if limit < 0 else self._raw[:limit]


def _stub_hf_requests(
    monkeypatch: pytest.MonkeyPatch,
    payloads: List[Any],
    *,
    final_urls: Optional[List[str]] = None,
    headers: Optional[List[Optional[Dict[str, str]]]] = None,
) -> List[str]:
    captured: List[str] = []
    remaining = iter(payloads)
    remaining_urls = iter(final_urls) if final_urls is not None else None
    remaining_headers = iter(headers) if headers is not None else None

    def fake_open(request: Any, timeout_seconds: float) -> _StubResponse:
        del timeout_seconds
        url = request.full_url
        captured.append(url)
        final_url = next(remaining_urls) if remaining_urls is not None else url
        response_headers = next(remaining_headers) if remaining_headers is not None else None
        return _StubResponse(next(remaining), final_url, headers=response_headers)

    monkeypatch.setattr(resolver_config, "_open_trusted_url", fake_open)
    return captured


def test_resolves_local_directory_without_importing_code(tmp_path: Path) -> None:
    config = {
        "model_type": "fixture",
        "architectures": ["UntrustedModel"],
        "auto_map": {"AutoModel": "modeling_untrusted.UntrustedModel"},
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "modeling_untrusted.py").write_text(
        "raise RuntimeError('must never be imported')\n", encoding="utf-8"
    )

    resolved = resolve_config(str(model_dir))

    assert resolved.config == config
    assert resolved.revision == f"sha256:{resolved.sha256}"
    assert resolved.cache_hit is True
    assert resolved.is_remote is False
    assert len(resolved.sha256) == 64
    assert resolved.identifier.startswith("config:fixture@")
    assert resolved.source_uri.startswith("local://model/")
    assert str(tmp_path) not in resolved.identifier
    assert str(tmp_path) not in resolved.source_uri
    assert str(tmp_path) not in json.dumps(resolved.metadata)
    assert resolved.metadata["input_kind"] == "local_directory"


def test_rejects_unknown_non_path_identifier() -> None:
    with pytest.raises(ResolutionError, match="neither an existing local path"):
        resolve_config("not a model id!", local_files_only=True)


def test_local_files_only_reports_cache_miss(tmp_path: Path) -> None:
    with pytest.raises(ResolutionError, match="OFFLINE_CACHE_MISS:.*not present"):
        resolve_config(
            "owner/model",
            revision="missing",
            local_files_only=True,
            cache_dir=tmp_path,
        )


def test_resolves_hugging_face_cache_ref(tmp_path: Path) -> None:
    commit = "a" * 40
    repo = tmp_path / "models--owner--model"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(f"{commit}\n", encoding="utf-8")
    (snapshot / "config.json").write_text(json.dumps({"model_type": "cached"}), encoding="utf-8")

    resolved = resolve_config("owner/model", local_files_only=True, cache_dir=tmp_path)

    assert resolved.revision == commit
    assert resolved.config["model_type"] == "cached"
    assert resolved.cache_hit is True
    assert resolved.source_uri == f"hf://owner/model@{commit}/config.json"
    assert resolved.metadata == {
        "input_kind": "hf_model_id",
        "requested_revision": "main",
        "resolved_commit": commit,
    }


def test_resolves_single_segment_hugging_face_model_id_from_cache(tmp_path: Path) -> None:
    commit = "c" * 40
    repo = tmp_path / "models--gpt2"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(f"{commit}\n", encoding="utf-8")
    (snapshot / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")

    resolved = resolve_config("gpt2", local_files_only=True, cache_dir=tmp_path)

    assert resolved.identifier == "gpt2"
    assert resolved.revision == commit
    assert resolved.source_uri == f"hf://gpt2@{commit}/config.json"
    assert resolved.metadata["input_kind"] == "hf_model_id"


def test_resolves_pasted_json_with_content_addressed_provenance() -> None:
    source = """
    {
      "model_type": "inline",
      "architectures": ["NeverImported"],
      "auto_map": {"AutoModel": "untrusted.RemoteModel"}
    }
    """

    first = resolve_config(source)
    second = resolve_config(json.dumps(first.config, sort_keys=True))

    assert first.config["model_type"] == "inline"
    assert first.identifier.startswith("config:inline@")
    assert first.revision == f"sha256:{first.sha256}"
    assert first.source_uri == f"inline://sha256/{first.sha256}/config.json"
    assert first.metadata["input_kind"] == "inline_json"
    assert first.sha256 == second.sha256
    assert first.cache_hit is False
    assert first.is_remote is False


@pytest.mark.parametrize(
    "embedded_revision",
    [123, "main", "/Users/private-user/secret/model", "token-like-secret"],
)
def test_local_and_inline_ignore_untrusted_embedded_commit_hash(
    tmp_path: Path, embedded_revision: object
) -> None:
    config = {"model_type": "fixture", "_commit_hash": embedded_revision}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    local = resolve_config(str(config_path))
    inline = resolve_config(json.dumps(config))

    assert local.revision == inline.revision == f"sha256:{local.sha256}"
    assert local.identifier == inline.identifier


def test_local_and_inline_accept_only_immutable_embedded_commit_hash(tmp_path: Path) -> None:
    commit = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    config = {"model_type": "fixture", "_commit_hash": commit}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    local = resolve_config(str(config_path))
    inline = resolve_config(json.dumps(config))

    assert local.revision == inline.revision == commit.lower()


@pytest.mark.parametrize(
    "secret_revision",
    [
        "/Users/private-user/secret/model",
        "hf_very_secret_token",
        "token-like-secret",
    ],
)
def test_rejects_non_immutable_explicit_revision_without_echoing_it(
    tmp_path: Path, secret_revision: str
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model_type":"fixture"}', encoding="utf-8")

    with pytest.raises(ResolutionError) as raised:
        resolve_config(str(config_path), revision=secret_revision)

    assert raised.value.code == "LOCAL_REVISION_NOT_IMMUTABLE"
    assert secret_revision not in str(raised.value)
    assert secret_revision not in json.dumps(raised.value.as_dict())


@pytest.mark.parametrize(
    ("source", "requested_revision", "input_kind"),
    [
        ("https://www.huggingface.co:443/owner/model/", "main", "hf_model_url"),
        (
            "https://huggingface.co/owner/model/blob/release%2Fv1/config.json",
            "release/v1",
            "hf_config_url",
        ),
        (
            "https://huggingface.co/owner/model/resolve/"
            "0123456789abcdef0123456789abcdef01234567/config.json",
            "0123456789abcdef0123456789abcdef01234567",
            "hf_config_url",
        ),
    ],
)
def test_hf_urls_pin_revision_then_fetch_only_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    requested_revision: str,
    input_kind: str,
) -> None:
    commit = "abcdef0123456789abcdef0123456789abcdef01"
    captured = _stub_hf_requests(
        monkeypatch,
        [
            {"sha": commit.upper(), "cardData": {"license": "apache-2.0"}},
            {"model_type": "remote", "architectures": ["RemoteModel"]},
        ],
    )

    resolved = resolve_config(source, cache_dir=tmp_path)

    encoded_revision = resolver_config.urllib.parse.quote(requested_revision, safe="")
    assert captured == [
        f"https://huggingface.co/api/models/owner/model/revision/{encoded_revision}",
        f"https://huggingface.co/owner/model/resolve/{commit}/config.json",
    ]
    assert resolved.identifier == "owner/model"
    assert resolved.revision == commit
    assert resolved.metadata["requested_revision"] == requested_revision
    assert resolved.metadata["resolved_commit"] == commit
    assert resolved.metadata["input_kind"] == input_kind
    assert resolved.source_uri == f"hf://owner/model@{commit}/config.json"
    assert resolved.license == "apache-2.0"
    assert len(resolved.sha256) == 64


def test_hf_model_url_can_resolve_from_existing_cache(tmp_path: Path) -> None:
    commit = "b" * 40
    repo = tmp_path / "models--owner--model"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(f"{commit}\n", encoding="utf-8")
    (snapshot / "config.json").write_text('{"model_type":"cached-url"}', encoding="utf-8")

    resolved = resolve_config(
        "https://huggingface.co/owner/model",
        local_files_only=True,
        cache_dir=tmp_path,
    )

    assert resolved.revision == commit
    assert resolved.config["model_type"] == "cached-url"
    assert resolved.metadata["input_kind"] == "hf_model_url"


@pytest.mark.parametrize(
    ("source", "input_kind"),
    [
        ("https://huggingface.co/gpt2", "hf_model_url"),
        ("https://huggingface.co/gpt2/blob/main/config.json", "hf_config_url"),
    ],
)
def test_single_segment_hf_urls_pin_revision_and_fetch_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    input_kind: str,
) -> None:
    commit = "d" * 40
    captured = _stub_hf_requests(
        monkeypatch,
        [{"sha": commit}, {"model_type": "gpt2", "n_layer": 12}],
    )

    resolved = resolve_config(source, cache_dir=tmp_path)

    assert captured == [
        "https://huggingface.co/api/models/gpt2/revision/main",
        f"https://huggingface.co/gpt2/resolve/{commit}/config.json",
    ]
    assert resolved.identifier == "gpt2"
    assert resolved.metadata["input_kind"] == input_kind


def test_equivalent_hf_inputs_have_identical_config_hash_and_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "a" * 40
    config = {"model_type": "same", "hidden_size": 32}
    sources = [
        "owner/model",
        "https://huggingface.co/owner/model",
        "https://huggingface.co/owner/model/blob/main/config.json",
    ]
    results = []
    for source in sources:
        _stub_hf_requests(monkeypatch, [{"sha": commit}, config])
        results.append(resolve_config(source, cache_dir=tmp_path))

    assert {result.revision for result in results} == {commit}
    assert {result.sha256 for result in results} == {results[0].sha256}
    assert all(result.config == config for result in results)
    assert [result.metadata["input_kind"] for result in results] == [
        "hf_model_id",
        "hf_model_url",
        "hf_config_url",
    ]


def test_local_provenance_is_stable_and_redacts_absolute_name_or_path(tmp_path: Path) -> None:
    config = {
        "model_type": "fixture",
        "_name_or_path": "/Users/private-user/secret/model",
    }
    model_dirs = [tmp_path / "left" / "model", tmp_path / "right" / "model"]
    resolved = []
    for model_dir in model_dirs:
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        resolved.append(resolve_config(str(model_dir)))

    assert resolved[0].identifier == resolved[1].identifier
    assert resolved[0].source_uri == resolved[1].source_uri
    assert resolved[0].identifier.startswith("config:fixture@")
    serialized = json.dumps(
        [
            {
                "identifier": item.identifier,
                "source_uri": item.source_uri,
                "metadata": item.metadata,
            }
            for item in resolved
        ]
    )
    assert str(tmp_path) not in serialized
    assert "/Users/private-user" not in serialized


def test_local_json_file_records_file_input_kind(tmp_path: Path) -> None:
    config_path = tmp_path / "named-model.json"
    config_path.write_text('{"model_type":"file"}', encoding="utf-8")

    resolved = resolve_config(str(config_path))

    assert resolved.metadata["input_kind"] == "local_file"
    assert resolved.metadata["local_label"] == "named-model"
    assert resolved.source_uri.startswith("local://named-model/")


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("http://huggingface.co/owner/model", "HF_URL_UNTRUSTED"),
        ("https://example.com/owner/model", "HF_URL_UNTRUSTED"),
        ("https://huggingface.co.example.com/owner/model", "HF_URL_UNTRUSTED"),
        ("https://user@huggingface.co/owner/model", "HF_URL_UNTRUSTED"),
        ("https://huggingface.co:8443/owner/model", "HF_URL_UNTRUSTED"),
        (
            "https://huggingface.co/owner/model?tab=files",
            "HF_URL_QUERY_UNSUPPORTED",
        ),
        (
            "https://huggingface.co/owner/model#files",
            "HF_URL_QUERY_UNSUPPORTED",
        ),
        ("https://huggingface.co/datasets/owner/model", "HF_URL_PATH_UNSUPPORTED"),
        (
            "https://huggingface.co/owner/model/blob/main/model.safetensors",
            "HF_URL_PATH_UNSUPPORTED",
        ),
        ("https://huggingface.co/owner/model/tree/main", "HF_URL_PATH_UNSUPPORTED"),
    ],
)
def test_rejects_untrusted_or_non_config_urls(source: str, code: str) -> None:
    with pytest.raises(ResolutionError) as raised:
        resolve_config(source)

    assert raised.value.code == code
    assert raised.value.hint


def test_rejects_conflicting_url_and_explicit_revisions() -> None:
    with pytest.raises(ResolutionError) as raised:
        resolve_config(
            "https://huggingface.co/owner/model/blob/main/config.json",
            revision="other",
        )

    assert raised.value.code == "HF_REVISION_CONFLICT"


@pytest.mark.parametrize("revision", ["../outside", "main/../../outside", "/main", "main//x"])
def test_rejects_revision_path_traversal(revision: str, tmp_path: Path) -> None:
    with pytest.raises(ResolutionError) as raised:
        resolve_config(
            "owner/model",
            revision=revision,
            local_files_only=True,
            cache_dir=tmp_path,
        )

    assert raised.value.code == "HF_INVALID_REVISION"


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("[1, 2, 3]", "JSON_ROOT_NOT_OBJECT"),
        ('{"model_type":', "JSON_INVALID"),
        ('{"value": NaN}', "JSON_INVALID"),
    ],
)
def test_rejects_invalid_pasted_config(source: str, code: str) -> None:
    with pytest.raises(ResolutionError) as raised:
        resolve_config(source)

    assert raised.value.code == code
    assert raised.value.as_dict()["hint"]


def test_limits_inline_json_nesting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver_config, "_MAX_JSON_DEPTH", 2)

    with pytest.raises(ResolutionError) as raised:
        resolve_config('{"nested":[[0]]}')

    assert raised.value.code == "JSON_DEPTH_LIMIT"
    assert raised.value.details == {"limit": 2}


def test_limits_inline_json_container_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver_config, "_MAX_JSON_CONTAINER_ITEMS", 3)

    with pytest.raises(ResolutionError) as raised:
        resolve_config('{"values":[1,2,3]}')

    assert raised.value.code == "JSON_ITEM_LIMIT"


def test_limits_local_json_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resolver_config, "_MAX_JSON_BYTES", 32)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"padding": "x" * 40}), encoding="utf-8")

    with pytest.raises(ResolutionError) as raised:
        resolve_config(str(config_path))

    assert raised.value.code == "JSON_SIZE_LIMIT"
    assert raised.value.details["limit_bytes"] == 32


def test_limits_remote_json_before_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resolver_config, "_MAX_JSON_BYTES", 32)
    captured = _stub_hf_requests(
        monkeypatch,
        [{"sha": "a" * 40}],
        headers=[{"Content-Length": "33"}],
    )

    with pytest.raises(ResolutionError) as raised:
        resolve_config("owner/model", cache_dir=tmp_path)

    assert raised.value.code == "JSON_SIZE_LIMIT"
    assert len(captured) == 1


def test_limits_remote_body_when_content_length_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resolver_config, "_MAX_JSON_BYTES", 32)
    _stub_hf_requests(
        monkeypatch,
        [b'{"padding":"' + (b"x" * 40) + b'"}'],
        headers=[{}],
    )

    with pytest.raises(ResolutionError) as raised:
        resolve_config("owner/model", cache_dir=tmp_path)

    assert raised.value.code == "JSON_SIZE_LIMIT"
    assert raised.value.details["actual_bytes"] == 33


@pytest.mark.parametrize("status", [401, 403])
def test_private_or_gated_hf_repository_returns_capability_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
) -> None:
    def denied(request: Any, timeout_seconds: float) -> None:
        del timeout_seconds
        raise resolver_config.urllib.error.HTTPError(
            request.full_url,
            status,
            "access denied",
            {},
            None,
        )

    monkeypatch.setattr(resolver_config, "_open_trusted_url", denied)

    with pytest.raises(ResolutionError) as raised:
        resolve_config("owner/private-model", cache_dir=tmp_path)

    assert raised.value.code == "HF_AUTH_REQUIRED"
    assert raised.value.details == {"status": status}
    assert "does not accept Hugging Face credentials" in str(raised.value)


def test_rejects_cached_config_symlink_outside_repository(tmp_path: Path) -> None:
    commit = "c" * 40
    repo = tmp_path / "models--owner--model"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(f"{commit}\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text('{"model_type":"outside"}', encoding="utf-8")
    (snapshot / "config.json").symlink_to(outside)

    with pytest.raises(ResolutionError) as raised:
        resolve_config(
            "owner/model",
            local_files_only=True,
            cache_dir=tmp_path,
        )

    assert raised.value.code == "HF_CACHE_UNSAFE"


def test_non_commit_cache_ref_is_not_treated_as_immutable(tmp_path: Path) -> None:
    repo = tmp_path / "models--owner--model"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (snapshot / "config.json").write_text('{"model_type":"unsafe"}', encoding="utf-8")

    with pytest.raises(ResolutionError) as raised:
        resolve_config(
            "owner/model",
            local_files_only=True,
            cache_dir=tmp_path,
        )

    assert raised.value.code == "OFFLINE_CACHE_MISS"


def test_rejects_redirect_away_from_hugging_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    followed = False

    def must_not_follow(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        nonlocal followed
        followed = True

    monkeypatch.setattr(
        resolver_config.urllib.request.HTTPRedirectHandler,
        "redirect_request",
        must_not_follow,
    )
    handler = resolver_config._TrustedHFRedirectHandler()
    request = resolver_config.urllib.request.Request(
        "https://huggingface.co/owner/model/resolve/main/config.json"
    )
    with pytest.raises(ResolutionError) as raised:
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://example.com/model.safetensors",
        )

    assert raised.value.code == "HF_URL_UNTRUSTED"
    assert followed is False


def test_requires_remote_metadata_to_return_immutable_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_hf_requests(monkeypatch, [{"sha": "main"}])

    with pytest.raises(ResolutionError) as raised:
        resolve_config("owner/model", cache_dir=tmp_path)

    assert raised.value.code == "HF_IMMUTABLE_REVISION_MISSING"

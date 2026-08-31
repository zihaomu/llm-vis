from __future__ import annotations

import json
from pathlib import Path

from llm_vis.analysis import inspect_model
from llm_vis.report import write_analysis

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def test_local_artifacts_do_not_serialize_absolute_input_paths(tmp_path: Path) -> None:
    config = json.loads((FIXTURE_DIR / "tiny_dense.json").read_text(encoding="utf-8"))
    config["_name_or_path"] = "/Users/private-user/secret/model"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    written = write_analysis(inspect_model(str(model_dir)), tmp_path / "artifact")

    manifest = json.loads(written["manifest.json"].read_text(encoding="utf-8"))
    assert manifest["source"]["uri"].startswith("local://model/")
    for path in written.values():
        serialized = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in serialized
        assert "/Users/private-user" not in serialized


def test_local_file_and_pasted_json_keep_the_same_graph_identity(tmp_path: Path) -> None:
    config_text = (FIXTURE_DIR / "qwen3_8_27b.json").read_text(encoding="utf-8")
    config_path = tmp_path / "downloaded-config.json"
    config_path.write_text(config_text, encoding="utf-8")

    local = inspect_model(str(config_path))
    pasted = inspect_model(config_text)

    assert local.resolved.identifier == pasted.resolved.identifier
    assert local.resolved.revision == pasted.resolved.revision
    assert local.analysis_id == pasted.analysis_id
    local_graph = local.graph_view
    pasted_graph = pasted.graph_view
    assert [view.id for view in local_graph.views] == [view.id for view in pasted_graph.views]
    assert [
        [node.id for node in view.nodes] for view in local_graph.views
    ] == [
        [node.id for node in view.nodes] for view in pasted_graph.views
    ]


def test_untrusted_commit_hash_is_not_serialized_to_any_artifact(tmp_path: Path) -> None:
    secret = "/Users/private-user/secret/model"
    config = {
        "model_type": "future_model",
        "_commit_hash": secret,
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "vocab_size": 256,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    written = write_analysis(inspect_model(str(config_path)), tmp_path / "artifact-secret")

    for path in written.values():
        assert secret not in path.read_text(encoding="utf-8")

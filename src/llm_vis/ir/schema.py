"""JSON Schema export for the public Model Map and GraphView artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Type, Union

from pydantic import BaseModel

from .models import ModelMap, Scenario, TraceDocument

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

SCHEMA_MODELS: Mapping[str, Type[BaseModel]] = {
    "model-map.schema.json": ModelMap,
    "scenario.schema.json": Scenario,
    "trace.schema.json": TraceDocument,
}


def schema_documents() -> Dict[str, Dict[str, Any]]:
    """Return all version 0.1 schemas keyed by their distribution filename."""

    # Import lazily so ``llm_vis.ir`` can finish initializing before GraphView loads
    # its Model Map types.  GraphView remains a renderer-neutral companion artifact,
    # not a field added to the stable Model Map v0.1 wire contract.
    from llm_vis.graph_view.models import GraphViewDocument

    schema_models = dict(SCHEMA_MODELS)
    schema_models["graph-view.schema.json"] = GraphViewDocument
    documents: Dict[str, Dict[str, Any]] = {}
    for filename, model_type in schema_models.items():
        schema = model_type.model_json_schema(mode="validation")
        schema["$schema"] = SCHEMA_DIALECT
        schema["$id"] = f"https://llm-vis.dev/schemas/v0.1/{filename}"
        documents[filename] = schema
    return documents


def export_schemas(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Write deterministic, human-reviewable JSON Schema files.

    Returns a mapping from schema filename to the path written.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for filename, schema in schema_documents().items():
        path = destination / filename
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written[filename] = path
    return written

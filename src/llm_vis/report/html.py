"""Dependency-free offline HTML report for M0-M3.10 analysis artifacts."""

from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional

from llm_vis.analysis.service import AnalysisBundle
from llm_vis.graph_view import GraphViewDocument
from llm_vis.report.dag_canvas import render_dag_canvas, render_dag_canvas_assets
from llm_vis.report.summaries import workload_diff_summaries

_ATTENTION_LABELS = {
    "linear_attention": "Linear Attention",
    "full_attention": "Full Attention",
    "dsa": "DSA Attention",
}
_MLP_LABELS = {"dense": "Dense FFN", "sparse": "MoE FFN"}
_STATE_LABELS = {"recurrent_state": "recurrent state", "kv_cache": "KV cache"}

_PRIMITIVE_EXPLANATIONS: Dict[str, tuple[str, str]] = {
    "gemm": (
        "Dense matrix projection. Each output feature is a weighted sum of input features.",
        "Y = X · W (+ b)",
    ),
    "matmul": (
        "Matrix multiplication between activation tensors, such as attention scores or values.",
        "Y = A · B",
    ),
    "rms_norm": (
        "Normalizes each token by its root-mean-square magnitude, then applies a learned scale.",
        "RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)",
    ),
    "layer_norm": (
        "Centers and scales each token using its feature mean and variance.",
        "LayerNorm(x) = γ ⊙ (x − mean(x)) / √(var(x) + ε) + β",
    ),
    "softmax": (
        "Turns scores into a normalized probability distribution along one axis.",
        "softmax(x)ᵢ = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x))",
    ),
    "top_k": (
        "Keeps the k largest scores and their indices; it does not prove a runtime route here.",
        "(values, indices) = TopK(x, k)",
    ),
    "silu": (
        "Smooth elementwise activation used by gated feed-forward networks.",
        "SiLU(x) = x · sigmoid(x)",
    ),
    "gelu": (
        "Smooth elementwise activation that gates values by a Gaussian-shaped factor.",
        "GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))",
    ),
    "add": ("Adds two tensors element by element.", "Y = A + B"),
    "multiply": ("Multiplies two tensors element by element.", "Y = A ⊙ B"),
    "scale": ("Multiplies every value by a scalar or broadcast scale.", "Y = α · X"),
    "mask": (
        "Applies an attention or validity mask before normalization.",
        "Yᵢ = Xᵢ if visible, otherwise −∞",
    ),
    "reshape": ("Changes tensor shape without changing logical values.", "Y = reshape(X)"),
    "transpose": ("Reorders tensor axes without changing logical values.", "Y = permute(X)"),
    "broadcast": (
        "Logically repeats compatible dimensions, for example KV heads across query heads.",
        "Y = broadcast(X, target_shape)",
    ),
    "split": ("Splits one tensor into named slices or branches.", "(Y₁,…,Yₙ) = split(X)"),
    "concat": ("Joins tensors along one dimension.", "Y = concat(X₁,…,Xₙ, axis)"),
    "rope": (
        "Rotates query or key feature pairs using position-dependent angles.",
        "RoPE(x, p) = rotate_pairs(x, θ(p))",
    ),
    "gather": ("Selects rows or elements using indices.", "Y = X[index]"),
    "scatter": ("Writes indexed values back into an output tensor.", "Y[index] = X"),
    "reduce": ("Combines values along an axis, commonly by sum.", "Y = Σ_axis X"),
    "convolution": (
        "Applies a local learned filter over a sequence or spatial neighborhood.",
        "Y[t] = Σₖ X[t−k] · W[k]",
    ),
    "state_read": ("Reads persistent cache or recurrent state.", "S = read(state)"),
    "state_write": ("Writes an updated cache or recurrent state.", "state′ = write(S′)"),
    "opaque": (
        "The available config evidence does not justify exposing internal operators.",
        "Unknown — opaque evidence boundary",
    ),
}

_COMPOUND_EXPLANATIONS: Dict[str, tuple[str, str]] = {
    "input": ("Introduces model inputs into this graph view.", "Y := input (no arithmetic)"),
    "output": ("Exposes a graph result to the parent view or caller.", "output := X"),
    "boundary": (
        "Preserves a tensor contract across a parent/child graph boundary.",
        "Y := X (boundary; no arithmetic)",
    ),
    "embedding": (
        "Looks up one learned vector for each token id.",
        "H[b,t,:] = E[token_id[b,t],:]",
    ),
    "decoder_pattern": (
        "Represents the ordered decoder-layer sequence; instances are not weight-shared loops.",
        "hₗ₊₁ = DecoderLayerₗ(hₗ, stateₗ)",
    ),
    "full_attention": (
        "Builds content-based token mixing from query, key and value projections.",
        "Attention(Q,K,V) = softmax(QKᵀ / √d + mask) · V",
    ),
    "linear_attention": (
        "Updates recurrent attention state and reads it to mix the current token sequence.",
        "Sₜ = update(Sₜ₋₁, xₜ); yₜ = read(Sₜ, xₜ)",
    ),
    "ffn": (
        "Applies a gated feature expansion and projects it back to hidden size.",
        "FFN(x) = (SiLU(xW_gate) ⊙ xW_up) · W_down",
    ),
    "residual": ("Adds a block branch back to its input.", "Y = X + branch(X)"),
    "norm": (
        "Normalizes token features before or after a model component.",
        "Y = Norm(X)",
    ),
    "lm_head": (
        "Projects hidden features to one score per vocabulary token.",
        "logits = H · W_vocabᵀ",
    ),
    "state_boundary_in": (
        "Reads the cache or recurrent state entering this view.",
        "S = read(parent_state)",
    ),
    "state_boundary_out": (
        "Returns the updated cache or recurrent state to the parent view.",
        "parent_state′ = write(S′)",
    ),
    "dsa": (
        "Attention region retained as opaque because config evidence does not prove its internals.",
        "Unknown — DSA internals are not inferred",
    ),
    "expert_pool": (
        "Static mixture-of-experts template; the report does not invent the runtime expert route.",
        "Y = Σₑ wₑ · Expertₑ(X), e ∈ TopK(router(X))",
    ),
    "shared_expert": ("Expert branch applied to every token.", "Y = Expert_shared(X)"),
    "moe_combine": (
        "Combines routed expert outputs using router weights.",
        "Y = Σₑ wₑ · Yₑ",
    ),
    "projector": ("Projects features into the language-model hidden space.", "Y = X · W"),
    "mtp": ("Optional multi-token prediction branch shown from config evidence.", "Y = MTP(H)"),
    "vision_tower": (
        "Vision encoder represented only to the depth supported by configuration evidence.",
        "Y = VisionEncoder(image)",
    ),
    "opaque": (
        "The available config evidence does not justify exposing internal operators.",
        "Unknown — opaque evidence boundary",
    ),
}


def _graph_node_detail(node: Dict[str, Any]) -> Dict[str, Any]:
    """Return novice-readable semantics without claiming an executed model graph."""

    primitive = str(node.get("primitive_kind") or "")
    kind = str(node.get("kind") or "")
    if node.get("opaque") or primitive == "opaque" or kind == "opaque":
        description, formula = _COMPOUND_EXPLANATIONS["opaque"]
        formula_scope = "unknown"
    elif primitive in _PRIMITIVE_EXPLANATIONS:
        description, formula = _PRIMITIVE_EXPLANATIONS[primitive]
        formula_scope = "conceptual operator equation"
    elif kind in _COMPOUND_EXPLANATIONS:
        description, formula = _COMPOUND_EXPLANATIONS[kind]
        formula_scope = "conceptual component equation"
    else:
        description = "Passes data through this semantic model component."
        formula = "Y = component(X)"
        formula_scope = "conceptual component equation"
    evidence_note = (
        "Config evidence does not justify an internal equation; Unknown is not zero, "
        "and no behavior is inferred. This is not a runtime trace, kernel equation, "
        "or measured latency."
        if formula_scope == "unknown"
        else (
            "Educational equation derived from the semantic node kind; "
            "it is not a runtime trace, kernel equation, or measured latency."
        )
    )
    return {
        "description": description,
        "formula": formula,
        "formula_scope": formula_scope,
        "evidence_note": evidence_note,
    }


def _layer_signature(item: Dict[str, Any]) -> tuple[str, ...]:
    """Return the macro semantic identity used only for UI pattern folding."""

    return tuple(
        str(item.get(key, ""))
        for key in (
            "label",
            "attention_kind",
            "mlp_kind",
            "state_kind",
        )
    )


def _layer_runs(items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    runs: list[Dict[str, Any]] = []
    for item in items:
        if runs and runs[-1]["signature"] == _layer_signature(item):
            runs[-1]["count"] += 1
            continue
        runs.append(
            {
                "signature": _layer_signature(item),
                "label": str(item.get("label", "?")),
                "count": 1,
                "first_layer_index": int(item.get("layer_index", 0)),
                "attention_kind": str(item.get("attention_kind", "unknown")),
                "mlp_kind": str(item.get("mlp_kind", "unknown")),
                "state_kind": str(item.get("state_kind", "unknown")),
            }
        )
    for run in runs:
        run.pop("signature")
    return runs


def _layer_run_text(run: Dict[str, Any]) -> str:
    count = int(run["count"])
    return str(run["label"]) if count == 1 else f"{run['label']}×{count}"


def _layer_legend(layer_strip: list[Dict[str, Any]]) -> list[Dict[str, str]]:
    legend: list[Dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    attention_kinds = {str(item.get("attention_kind", "unknown")) for item in layer_strip}
    mlp_kinds = {str(item.get("mlp_kind", "unknown")) for item in layer_strip}
    for item in layer_strip:
        signature = _layer_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        attention_kind = str(item.get("attention_kind", "unknown"))
        mlp_kind = str(item.get("mlp_kind", "unknown"))
        state_kind = str(item.get("state_kind", "unknown"))
        attention = _ATTENTION_LABELS.get(
            attention_kind, attention_kind.replace("_", " ").title()
        )
        mlp = _MLP_LABELS.get(mlp_kind, mlp_kind.replace("_", " ").title())
        state = _STATE_LABELS.get(state_kind, state_kind.replace("_", " "))
        label = str(item.get("label", "?"))
        if len(mlp_kinds) > 1 and len(attention_kinds) == 1:
            description = mlp
        elif len(attention_kinds) > 1 and len(mlp_kinds) == 1:
            description = f"{attention} · {state}"
        else:
            description = f"{mlp} · {attention} · {state}"
        legend.append(
            {
                "label": label,
                "description": description,
                "attention_kind": attention_kind,
                "mlp_kind": mlp_kind,
                "state_kind": state_kind,
            }
        )
    return legend


def _layer_legend_context(layer_strip: list[Dict[str, Any]]) -> Optional[str]:
    attention_kinds = {str(item.get("attention_kind", "unknown")) for item in layer_strip}
    mlp_kinds = {str(item.get("mlp_kind", "unknown")) for item in layer_strip}
    state_kinds = {str(item.get("state_kind", "unknown")) for item in layer_strip}
    if len(mlp_kinds) <= 1 or len(attention_kinds) != 1 or len(state_kinds) != 1:
        return None
    attention_kind = next(iter(attention_kinds))
    state_kind = next(iter(state_kinds))
    attention = _ATTENTION_LABELS.get(
        attention_kind, attention_kind.replace("_", " ").title()
    )
    state = _STATE_LABELS.get(state_kind, state_kind.replace("_", " "))
    return f"all layers · {attention} · {state}"


def _layer_pattern_payload(layer_strip: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a truthful compact pattern without replacing exact layer instances."""

    total = len(layer_strip)
    if not total:
        return {
            "summary": "No text decoder layers",
            "total_layers": 0,
            "period_length": 0,
            "repeat_count": 0,
            "segments": [],
            "tail_segments": [],
            "legend": [],
            "legend_context": None,
            "anomaly_count": 0,
        }

    signatures = [_layer_signature(item) for item in layer_strip]
    period_length = total
    repeat_count = 1
    tail_length = 0
    for candidate in range(1, total // 2 + 1):
        matches_candidate = all(
            signature == signatures[index % candidate]
            for index, signature in enumerate(signatures)
        )
        if matches_candidate:
            candidate_repeats, candidate_tail = divmod(total, candidate)
            if candidate_repeats >= 2:
                period_length = candidate
                repeat_count = candidate_repeats
                tail_length = candidate_tail
                break

    repeat_unit = layer_strip[:period_length]
    tail = layer_strip[period_length * repeat_count :] if tail_length else []
    segments = _layer_runs(repeat_unit)
    tail_segments = _layer_runs(tail)
    anomaly_count = sum(bool(item.get("anomaly")) for item in layer_strip)

    if len(segments) > 6:
        if repeat_count > 1:
            summary = f"Mixed {period_length}-layer pattern ×{repeat_count}"
            if tail_length:
                summary = f"{summary} + {tail_length}-layer tail"
        else:
            summary = f"Mixed sequence · {total} layers · {len(segments)} runs"
        display_segments: list[Dict[str, Any]] = []
    else:
        core = " → ".join(_layer_run_text(run) for run in segments)
        if repeat_count > 1 and len(segments) == 1 and not tail_segments:
            summary = f"{segments[0]['label']}×{total}"
        elif repeat_count > 1:
            summary = f"[{core}] ×{repeat_count}"
        else:
            summary = core
        if tail_segments:
            tail_text = " → ".join(_layer_run_text(run) for run in tail_segments)
            summary = f"{summary} → {tail_text} tail"
        display_segments = segments
    if anomaly_count:
        suffix = "deviation" if anomaly_count == 1 else "deviations"
        summary = f"{summary} · {anomaly_count} {suffix}"

    return {
        "summary": summary,
        "total_layers": total,
        "period_length": period_length,
        "repeat_count": repeat_count,
        "segments": display_segments,
        "tail_segments": tail_segments,
        "legend": _layer_legend(layer_strip),
        "legend_context": _layer_legend_context(layer_strip),
        "anomaly_count": anomaly_count,
    }


def _capture_summary(bundle: AnalysisBundle) -> str:
    items = []
    for capture in bundle.captures:
        status = str(capture.get("status", "unknown"))
        items.append(
            '<button class="capture-card" data-status="{}" type="button">'
            '<span class="capture-kind">{}</span><span>{}</span>'
            "<span>{:.1%} coverage</span><span>{} ops</span>"
            "<span>full-model perf: false</span></button>".format(
                html.escape(status),
                html.escape(str(capture.get("kind", "unknown"))),
                html.escape(status),
                float(capture.get("coverage", 0.0)),
                int(capture.get("logical_op_count", 0)),
            )
        )
    return "".join(items) or (
        '<div class="empty">No bounded representative capture was requested.</div>'
    )


def _renderer_graph_document(document: GraphViewDocument) -> Dict[str, Any]:
    """Enrich the neutral flat graph contract for the dependency-free SVG renderer."""

    payload = document.model_dump(mode="json")
    views_by_id = {view["id"]: view for view in payload["views"]}
    for view in payload["views"]:
        ports_by_id = {port["id"]: port for port in view["ports"]}
        for port in view["ports"]:
            if not port["shape_known"]:
                port["unknown_reason"] = (
                    "The config inventory does not provide a trustworthy shape for this port."
                )
        for node in view["nodes"]:
            port_ids = [*node["input_port_ids"], *node["output_port_ids"]]
            node["ports"] = [ports_by_id[port_id] for port_id in port_ids]
            node["origin"] = "config"
            detail = _graph_node_detail(node)
            alternate_map = node.get("attributes", {}).get("drilldown_view_ids", {})
            alternate_ids = (
                list(alternate_map.values()) if isinstance(alternate_map, dict) else []
            )
            candidate_ids = [node.get("drilldown_view_id"), *alternate_ids]
            child_view_ids = list(dict.fromkeys(item for item in candidate_ids if item))
            detail["child_views"] = [
                {
                    "id": child_id,
                    "label": views_by_id[child_id]["label"],
                    "primary": child_id == node.get("drilldown_view_id"),
                    "node_count": sum(
                        item.get("kind") != "boundary"
                        for item in views_by_id[child_id]["nodes"]
                    ),
                }
                for child_id in child_view_ids
                if child_id in views_by_id
            ]
            node["detail"] = detail
            node["coverage_status"] = (
                "opaque"
                if node["opaque"]
                else "config-evidenced"
                if node["coverage"] > 0
                else "unknown"
            )
            core_info = dict(node["attributes"])
            representative_port = next(
                (
                    ports_by_id[port_id]
                    for port_id in (*node["output_port_ids"], *node["input_port_ids"])
                ),
                None,
            )
            if representative_port is not None:
                core_info.setdefault(
                    "shape",
                    representative_port["shape"]
                    if representative_port["shape_known"]
                    else "Unknown",
                )
                core_info.setdefault("dtype", representative_port["dtype"])
                if not representative_port["shape_known"]:
                    core_info.setdefault(
                        "shape_unknown_reason",
                        representative_port["unknown_reason"],
                    )
            node["core_info"] = core_info
        for edge in view["edges"]:
            source = ports_by_id[edge["source_port_id"]]
            target = ports_by_id[edge["target_port_id"]]
            edge["source_node_id"] = source["node_id"]
            edge["target_node_id"] = target["node_id"]
            shape = (
                "[" + ",".join(str(dimension) for dimension in source["shape"]) + "]"
                if source["shape_known"]
                else "Unknown"
            )
            edge["tensor_name"] = source["name"]
            edge["shape_label"] = shape
            edge["dtype_label"] = source["dtype"]
            edge["label"] = f"{source['name']} {shape} {source['dtype']}"
            edge["shape_known"] = source["shape_known"]
            if not source["shape_known"]:
                edge["unknown_reason"] = source["unknown_reason"]
    return payload


def render_html(
    bundle: AnalysisBundle,
    *,
    graph_view: GraphViewDocument | None = None,
) -> str:
    graph_view = graph_view or bundle.graph_view
    graph_payload = graph_view.model_dump(mode="json")
    dag_canvas = render_dag_canvas(
        _renderer_graph_document(graph_view),
        element_id="model-dag",
        initial_view_id=graph_view.views[0].id,
    )
    layer_strip = list(bundle.layer_strip)
    payload = {
        "manifest": bundle.manifest(),
        "modelMap": bundle.model_map.model_dump(mode="json"),
        "graphView": graph_payload,
        "layerStrip": layer_strip,
        "layerPattern": _layer_pattern_payload(layer_strip),
        "captures": [dict(item) for item in bundle.captures],
        "hotspots": [dict(item) for item in bundle.hotspot_summaries],
        "roofline": [dict(item) for item in bundle.roofline_summaries],
        "workloadDiffs": list(workload_diff_summaries(bundle.model_map)),
        "hardwareProfile": (
            bundle.hardware_profile.model_dump(mode="json")
            if bundle.hardware_profile is not None
            else None
        ),
    }
    embedded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    source_title = html.escape(bundle.model_map.model.name)
    model_definition = next(
        (definition for definition in bundle.model_map.definitions if definition.kind == "model"),
        None,
    )
    display_title = html.escape(
        model_definition.label if model_definition else bundle.model_map.model.family
    )
    model_revision = str(bundle.model_map.model.revision or "Unknown")
    revision_title = html.escape(model_revision)
    revision_short = html.escape(
        model_revision if model_revision == "Unknown" else model_revision[:8]
    )
    config_sha256 = str(bundle.resolved.sha256)
    config_hash_title = html.escape(config_sha256)
    config_hash_short = html.escape(config_sha256[:8])
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
<title>LLM-Vis — __DISPLAY_TITLE__</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; --bg:#080d18; --panel:#11192a; --panel-strong:#0d1627; --line:#27344c; --muted:#8fa3c4; --text:#e9eff8; --blue:#6ea8fe; --green:#5ee0ad; --amber:#f6ba58; --red:#ff7b86; }
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; background:var(--bg); color:var(--text); }
header { position:sticky; top:0; z-index:8; display:flex; align-items:center; justify-content:space-between; gap:16px; min-height:62px; padding:10px 16px; border-bottom:1px solid var(--line); background:rgba(8,13,24,.97); backdrop-filter:blur(12px); }
.identity,.primary-controls,.badges,.workspace-actions,.drawer-heading { display:flex; align-items:center; gap:9px; }
.identity { min-width:0; flex:1 1 auto; }
.brand { color:var(--blue); font-size:12px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; white-space:nowrap; }
.identity-divider { color:#52627d; }
h1 { font-size:16px; margin:0; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.revision-label { flex:0 0 auto; color:#8fa3c4; border:1px solid #293952; border-radius:999px; padding:3px 7px; font:10px/1 ui-monospace,monospace; }
h2 { font-size:14px; letter-spacing:.02em; margin:0; color:#c7d5ea; }
h3 { font-size:13px; margin:18px 0 8px; color:#9fb7dc; }
.badge { border:1px solid var(--line); background:#16213a; color:#c9d8ef; border-radius:999px; padding:4px 9px; font-size:11px; }
.badge.safe { border-color:#276b58; color:var(--green); }
.badge.warn { border-color:#805d25; color:var(--amber); }
.primary-controls { flex:0 1 auto; justify-content:flex-end; }
.compact-control { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:11px; white-space:nowrap; }
.compact-control > span { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
select,input { border:1px solid var(--line); background:#0c1424; color:var(--text); border-radius:7px; padding:7px 9px; }
.global-search { width:min(300px,26vw); min-width:180px; }
.status-menu { position:relative; }
.status-menu summary,.workspace-action,.drawer-close { border:1px solid var(--line); background:#111c30; color:var(--text); border-radius:7px; padding:7px 10px; font:12px/1.2 inherit; cursor:pointer; list-style:none; }
.status-menu summary::-webkit-details-marker { display:none; }
.status-menu[open] summary,.workspace-action[aria-expanded="true"] { border-color:var(--blue); background:#192c49; }
.status-popover { position:absolute; right:0; top:calc(100% + 9px); display:grid; gap:7px; width:min(420px,88vw); padding:12px; border:1px solid var(--line); border-radius:10px; background:#0e1829; box-shadow:0 18px 50px rgba(0,0,0,.45); }
.status-popover .badge { display:block; width:max-content; max-width:100%; }
main { padding:12px; }
section { min-width:0; background:var(--panel); border:1px solid var(--line); border-radius:11px; padding:14px; }
.dag-panel { position:relative; padding:12px; min-height:calc(100vh - 86px); }
.workspace-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin:0 2px 10px; }
.workspace-title { display:flex; align-items:center; gap:8px; }
.level-pill { border:1px solid #35547b; border-radius:999px; padding:2px 7px; color:#9fc7ff; font:10px/1.3 ui-monospace,monospace; }
.dag-panel > .llm-dag { padding:0; box-shadow:none; }
.dag-panel #model-dag .llm-dag-search { display:none; }
.dag-panel .llm-dag-stage { height:clamp(460px,calc(100vh - 230px),820px); }
.dag-help { color:var(--muted); font-size:11px; margin:5px 0 0; }
.dag-evidence { display:flex; align-items:center; flex-wrap:wrap; gap:6px; margin:7px 0 0; color:var(--muted); font-size:10px; }
.dag-evidence .badge { padding:3px 7px; }
.workspace-action { min-height:32px; }
.workspace-actions { flex-wrap:wrap; justify-content:flex-end; }
.heat-control { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:11px; white-space:nowrap; }
.heat-control select { min-height:32px; padding:5px 8px; font-size:11px; }
.workspace-action:hover,.drawer-close:hover { border-color:#4f6f9a; background:#162742; }
.workspace-action:focus-visible,.drawer-close:focus-visible,.status-menu summary:focus-visible,.inspector-tab:focus-visible,.layer-panel > summary:focus-visible,.layer:focus-visible { outline:2px solid #9bc7ff; outline-offset:2px; }
.layer-panel { margin-bottom:8px; border:1px solid var(--line); border-radius:8px; background:#0e1728; }
.layer-panel > summary { display:flex; align-items:center; flex-wrap:wrap; gap:7px 10px; padding:8px 10px; color:#afbfda; font-size:11px; cursor:pointer; user-select:none; list-style:none; }
.layer-panel > summary::-webkit-details-marker { display:none; }
.layer-chevron { color:#7187a8; font-size:16px; line-height:1; transform-origin:center; transition:transform .15s ease; }
.layer-panel[open] .layer-chevron { transform:rotate(90deg); }
.layer-summary-main,.layer-legend,.layer-legend-item { display:flex; align-items:center; gap:7px; }
.layer-summary-main { flex:0 1 auto; min-width:0; }
.layer-panel-title { color:#d2deef; font-weight:700; white-space:nowrap; }
.layer-pattern-summary { max-width:360px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; border:1px solid #35547b; border-radius:999px; background:#14243b; color:#e9f2ff; padding:3px 8px; font:11px/1.2 ui-monospace,monospace; }
.layer-count { color:#8296b7; white-space:nowrap; }
.layer-legend { flex:1 1 360px; flex-wrap:wrap; min-width:220px; }
.layer-legend-item { gap:5px; color:#9fb1cc; white-space:nowrap; }
.layer-legend-context { color:#7187a8; white-space:nowrap; }
.layer-key { display:grid; place-items:center; width:20px; height:20px; border-radius:5px; background:#285d9d; color:#fff; font:700 10px/1 ui-monospace,monospace; }
.layer-key[data-state="recurrent_state"] { background:#7045a5; }
.layer-key[data-mlp="sparse"] { outline:2px solid var(--amber); outline-offset:-2px; }
.layer-summary-note { margin-left:auto; color:#7187a8; font-size:10px; white-space:nowrap; }
.layer-summary-note[data-anomalies="true"] { color:var(--red); }
.layer-detail-body { border-top:1px solid var(--line); padding:8px 10px 10px; }
.layer-detail-help { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:7px; color:#8296b7; font-size:10px; }
.layer-detail-key { color:#7187a8; text-align:right; }
.structure-index .list { max-height:320px; }
.split { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.strip { display:flex; flex-wrap:nowrap; gap:4px; overflow-x:auto; overscroll-behavior-inline:contain; padding-bottom:2px; }
button { color:inherit; }
.layer { flex:0 0 27px; width:27px; height:27px; border:0; border-radius:6px; display:grid; place-items:center; font:12px ui-monospace,monospace; cursor:pointer; background:#285d9d; }
.layer[data-state="recurrent_state"] { background:#7045a5; }
.layer[data-mlp="sparse"] { outline:2px solid var(--amber); }
.layer[data-coverage="captured"] { box-shadow:inset 0 0 0 2px var(--green); }
.layer[data-coverage="config"] { opacity:.72; }
.layer[data-coverage="opaque"] { box-shadow:inset 0 0 0 2px var(--amber); }
.layer[data-anomaly="true"] { outline:2px solid var(--red); }
.layer:hover,.item:hover,.capture-card:hover { filter:brightness(1.16); }
.list { max-height:420px; overflow:auto; padding-right:4px; }
.item { width:100%; text-align:left; padding:8px 9px; margin:5px 0; border:0; border-left:3px solid #4f8edc; background:#16223a; border-radius:5px; cursor:pointer; }
.item small { display:block; color:var(--muted); margin-top:3px; overflow:hidden; text-overflow:ellipsis; }
.capture-grid { display:grid; gap:6px; }
.capture-card { display:grid; grid-template-columns:1.2fr 1fr 1fr .7fr 1.3fr; gap:7px; width:100%; border:1px solid var(--line); border-left:3px solid var(--blue); border-radius:7px; background:#121f35; padding:8px; text-align:left; font-size:11px; cursor:pointer; }
.capture-card[data-status="captured"] { border-left-color:var(--green); }
.capture-card[data-status="opaque"],.capture-card[data-status="unavailable"] { border-left-color:var(--amber); }
.capture-kind { font-weight:700; }
.empty,.muted { color:var(--muted); }
.empty { padding:9px; border:1px dashed var(--line); border-radius:7px; font-size:12px; }
.notice { padding:9px 11px; border:1px solid #775a28; border-radius:7px; background:#2b2317; color:#ffd99c; font-size:12px; margin-bottom:10px; }
.unknown { color:var(--amber); }
table { width:100%; border-collapse:collapse; font-size:12px; }
th,td { text-align:left; padding:7px 8px; border-bottom:1px solid #23304a; vertical-align:top; }
th { color:#9fb4d8; font-weight:600; position:sticky; top:0; background:#111b2e; }
tr.clickable { cursor:pointer; }
tr.clickable:hover { background:#17253e; }
pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; color:#cbd8ed; max-height:660px; overflow:auto; }
.workspace-drawer { position:fixed; z-index:21; top:62px; bottom:0; display:flex; flex-direction:column; width:min(390px,92vw); padding:0; border:0; border-radius:0; background:#0d1728; box-shadow:0 0 60px rgba(0,0,0,.52); transition:transform .18s ease,visibility .18s ease; visibility:hidden; }
.workspace-drawer.navigator { left:0; transform:translateX(-102%); }
.workspace-drawer.inspector { right:0; transform:translateX(102%); }
.workspace-drawer.is-open { transform:translateX(0); visibility:visible; }
.drawer-header { display:flex; align-items:center; justify-content:space-between; gap:12px; min-height:62px; padding:11px 14px; border-bottom:1px solid var(--line); }
.drawer-heading { min-width:0; }
.drawer-kicker { color:var(--blue); font:10px/1.2 ui-monospace,monospace; letter-spacing:.08em; text-transform:uppercase; }
.drawer-close { width:32px; height:32px; padding:0; font-size:18px; }
.drawer-body { flex:1 1 auto; min-height:0; overflow:auto; padding:12px 14px 18px; }
.drawer-body > h3:first-child { margin-top:2px; }
.workspace-drawer .list { max-height:none; overflow:visible; }
.navigator-group { border-bottom:1px solid var(--line); }
.navigator-group > summary { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 2px; color:#b8c8e1; font-size:12px; cursor:pointer; }
.navigator-group > summary span { color:var(--muted); font:10px/1.2 ui-monospace,monospace; }
.navigator-group .list { padding-bottom:10px; }
.workspace-drawer.inspector .drawer-body { display:flex; flex-direction:column; overflow:hidden; }
.inspector-content { flex:1 1 auto; min-height:0; overflow:auto; }
.inspector-json { min-height:100%; max-height:none; padding:10px; border:1px solid #253653; border-radius:8px; background:#091322; }
.inspector-explain { display:grid; gap:12px; }
.inspector-node-kind { display:flex; flex-wrap:wrap; gap:6px; }
.inspector-node-kind span { border:1px solid #35547b; border-radius:999px; padding:3px 7px; color:#aacbfa; font:10px/1.2 ui-monospace,monospace; }
.inspector-section { display:grid; gap:6px; padding:10px; border:1px solid #263754; border-radius:8px; background:#101b2e; }
.inspector-section h3 { margin:0; color:#a9c4ea; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
.inspector-section p { margin:0; color:#d5e0f0; font-size:12px; line-height:1.5; }
.inspector-formula { display:block; padding:10px; border:1px solid #3b5680; border-radius:7px; background:#081323; color:#eaf3ff; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }
.inspector-port-list { display:grid; gap:5px; }
.inspector-port { display:grid; grid-template-columns:minmax(72px,1fr) auto; gap:8px; padding:6px 7px; border-radius:6px; background:#0b1526; color:#c8d8ec; font:10px/1.35 ui-monospace,monospace; }
.inspector-port small { color:#8399ba; text-align:right; }
.inspector-interaction { border-color:#3d6190; background:#12243d; }
.inspector-child-list { display:grid; gap:6px; }
.inspector-child { width:100%; display:flex; align-items:center; justify-content:space-between; gap:8px; padding:8px 9px; border:1px solid #40648f; border-radius:7px; background:#132844; color:#e3efff; text-align:left; cursor:pointer; }
.inspector-child:hover,.inspector-child:focus-visible { border-color:#8abaff; background:#1a365a; }
.inspector-child small { color:#93acd0; }
.inspector-evidence { color:#8fa3c4!important; font-size:10px!important; }
.drawer-scrim { position:fixed; z-index:20; inset:62px 0 0; width:100%; height:calc(100% - 62px); border:0; padding:0; background:rgba(2,7,15,.62); backdrop-filter:blur(2px); cursor:pointer; }
.drawer-scrim[hidden] { display:none; }
.inspector-tabs { display:grid; grid-template-columns:repeat(3,1fr); gap:5px; margin-bottom:9px; }
.inspector-tab { border:1px solid var(--line); border-radius:6px; background:#101a2d; padding:6px; font-size:11px; cursor:pointer; }
.inspector-tab.active { border-color:var(--blue); color:#fff; background:#1b3152; }
.inspector-title { color:var(--muted); font-size:11px; margin:0 0 8px; min-height:16px; overflow-wrap:anywhere; }
.scenario-card { display:grid; grid-template-columns:repeat(6,minmax(80px,1fr)); gap:7px; margin-bottom:10px; }
.scenario-field { background:#0d1728; border:1px solid #22314b; border-radius:7px; padding:7px; }
.scenario-field span { display:block; color:var(--muted); font-size:10px; }
.scenario-field strong { display:block; margin-top:3px; font-size:12px; overflow:hidden; text-overflow:ellipsis; }
.diag { border-left-color:var(--amber); }
.analysis-shell { margin-top:12px; border:1px solid var(--line); border-radius:11px; background:#0c1423; }
.analysis-shell > summary { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:13px 15px; color:#b9c9e2; cursor:pointer; }
.analysis-shell > summary small { color:var(--muted); font-size:11px; font-weight:400; }
.analysis-shell[open] > summary { border-bottom:1px solid var(--line); }
.analysis-grid { display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:12px; padding:12px; }
.analysis-grid > section { grid-column:span 4; overflow-x:auto; }
.analysis-grid > .wide { grid-column:1/-1; }
@media(max-width:900px) {
  header { align-items:flex-start; flex-direction:column; gap:8px; }
  .primary-controls { width:100%; justify-content:flex-start; flex-wrap:wrap; }
  .global-search { flex:1 1 240px; width:auto; }
  .dag-panel { min-height:auto; }
  .analysis-grid > section { grid-column:1/-1; }
  .scenario-card { grid-template-columns:repeat(3,1fr); }
  .workspace-drawer { top:0; }
  .drawer-scrim { inset:0; height:100%; }
}
@media(min-width:901px) { .drawer-scrim { display:none; } }
@media(max-width:640px) {
  header { padding:9px 10px; }
  .identity-divider { display:none; }
  .identity { align-items:flex-start; flex-direction:column; gap:2px; }
  h1 { max-width:88vw; }
  main { padding:8px; }
  .workspace-heading { align-items:stretch; flex-direction:column; }
  .workspace-actions { display:grid; grid-template-columns:1fr 1fr 1fr; }
  .heat-control { grid-column:1/-1; justify-content:space-between; }
  .workspace-action { padding-inline:6px; }
  .layer-panel > summary { align-items:flex-start; }
  .layer-summary-main { flex-wrap:wrap; }
  .layer-pattern-summary { max-width:70vw; }
  .layer-legend { flex-basis:calc(100% - 28px); margin-left:26px; }
  .layer-summary-note { flex:0 1 calc(100% - 26px); margin-left:26px; white-space:normal; }
  .layer-detail-help { align-items:flex-start; flex-direction:column; }
  .layer-detail-key { text-align:left; }
  .dag-panel .llm-dag-stage { height:68vh; min-height:430px; }
  .split { grid-template-columns:1fr; }
  .scenario-card { grid-template-columns:repeat(2,1fr); }
  .capture-card { grid-template-columns:1fr 1fr; }
  .analysis-shell > summary { align-items:flex-start; flex-direction:column; }
}
@media(max-width:480px) {
  .layer-panel > summary { gap:5px 7px; padding:7px 8px; }
  .layer-summary-main { gap:5px; }
  .layer-legend { gap:5px 7px; margin-left:22px; min-width:0; }
  .layer-legend-item { gap:4px; }
  .layer-key { width:18px; height:18px; }
  .layer-summary-note { flex-basis:calc(100% - 22px); margin-left:22px; font-size:9px; }
}
</style>
__DAG_ASSETS__
</head>
<body>
<header>
  <div class="identity"><span class="brand">LLM-Vis</span><span class="identity-divider">/</span><h1 title="__SOURCE_TITLE__">__DISPLAY_TITLE__</h1><span class="revision-label" title="Revision __REVISION_TITLE__">rev __REVISION_SHORT__</span></div>
  <div class="primary-controls">
    <label class="compact-control"><span>Scenario</span><select id="scenario-select" aria-label="Scenario"></select></label>
    <label class="compact-control"><span>Find graph or structure</span><input class="global-search" id="search" type="search" placeholder="Search graph, layer or ID" aria-label="Find graph or structure"></label>
    <details class="status-menu"><summary>Report status</summary><div class="status-popover"><div class="badges"><span class="badge" id="capability-badge"></span><span class="badge" id="coverage-badge"></span><span class="badge safe">target config.json only · zero target weights · zero target forward</span><span class="badge warn">runtime trace: not imported</span></div></div></details>
  </div>
</header>
<main>
  <section class="dag-panel" aria-labelledby="graph-heading">
    <div class="workspace-heading">
      <div><div class="workspace-title"><h2 id="graph-heading">Model graph</h2><span class="level-pill">Top → bottom DAG</span></div><p class="dag-help">Follow tensors from top to bottom. Single-click a node for its explanation and equation; double-click a node marked “N nodes ↓” to enter its child graph. Collapse returns to the parent.</p><p class="dag-evidence"><span class="badge safe">Target input: config.json only</span><span class="badge" id="source-evidence-badge"></span><span class="badge" title="Config SHA-256 __CONFIG_HASH_TITLE__">config __CONFIG_HASH_SHORT__</span><span class="badge" id="graph-evidence-badge"></span><span>no target weights · no model code · no full forward</span></p></div>
      <div class="workspace-actions" aria-label="Workspace panels">
        <label class="heat-control"><span>Theory heat</span><select id="heatmap-mode" aria-label="Theoretical bottleneck heatmap"><option value="pressure">Pressure</option><option value="compute">Compute</option><option value="memory">Memory</option><option value="off">Off</option></select></label>
        <button class="workspace-action" type="button" data-drawer-target="navigator-drawer" aria-controls="navigator-drawer" aria-expanded="false">Browse</button>
        <button class="workspace-action" type="button" data-drawer-target="inspector-drawer" aria-controls="inspector-drawer" aria-expanded="false">Inspector</button>
        <button class="workspace-action" id="analysis-toggle" type="button">Analysis</button>
      </div>
    </div>
    <details class="layer-panel" id="layer-panel">
      <summary>
        <span class="layer-chevron" aria-hidden="true">›</span>
        <span class="layer-summary-main"><span class="layer-panel-title">Text decoder layers</span><span class="layer-pattern-summary" id="layer-pattern-summary"></span><span class="layer-count" id="layer-count"></span></span>
        <span class="layer-legend" id="layer-legend" role="list" aria-label="Layer symbol legend"></span>
        <span class="layer-summary-note" id="layer-summary-note"></span>
      </summary>
      <div class="layer-detail-body">
        <div class="layer-detail-help"><span id="layer-detail-label">Exact layer positions</span><span class="layer-detail-key">green inner = bounded capture · amber inner = opaque · amber outer = MoE · red outer = anomaly</span></div>
        <div class="strip" id="strip" role="group" aria-label="Exact text decoder layers"></div>
      </div>
    </details>
    __DAG_CANVAS__

    <button class="drawer-scrim" id="drawer-scrim" type="button" aria-label="Close side panel" hidden></button>
    <aside class="workspace-drawer navigator" id="navigator-drawer" aria-hidden="true" aria-label="Architecture navigator">
      <div class="drawer-header"><div class="drawer-heading"><span class="drawer-kicker">Browse</span><h2>Architecture navigator</h2></div><button class="drawer-close" type="button" data-drawer-close aria-label="Close architecture navigator">×</button></div>
      <div class="drawer-body"><details class="navigator-group" open><summary>Definitions <span id="definition-count"></span></summary><div class="list" id="definitions"></div></details><details class="navigator-group"><summary>Diagnostics <span id="diagnostic-count"></span></summary><div class="list" id="diagnostics"></div></details></div>
    </aside>
    <aside class="workspace-drawer inspector" id="inspector-drawer" aria-hidden="true" aria-label="Selection inspector">
      <div class="drawer-header"><div class="drawer-heading"><span class="drawer-kicker">Selection</span><h2>Inspector</h2></div><button class="drawer-close" type="button" data-drawer-close aria-label="Close inspector">×</button></div>
      <div class="drawer-body"><div class="inspector-tabs" role="tablist" aria-label="Inspector views"><button class="inspector-tab active" id="inspector-tab-explain" data-inspector-tab="explain" type="button" role="tab" aria-controls="inspector" aria-selected="true">Explain</button><button class="inspector-tab" id="inspector-tab-tensors" data-inspector-tab="tensors" type="button" role="tab" aria-controls="inspector" aria-selected="false">Tensors</button><button class="inspector-tab" id="inspector-tab-cost" data-inspector-tab="cost" type="button" role="tab" aria-controls="inspector" aria-selected="false">Cost</button><button class="inspector-tab" id="inspector-tab-runtime" data-inspector-tab="runtime" type="button" role="tab" aria-controls="inspector" aria-selected="false">Runtime</button><button class="inspector-tab" id="inspector-tab-provenance" data-inspector-tab="provenance" type="button" role="tab" aria-controls="inspector" aria-selected="false">Provenance</button><button class="inspector-tab" id="inspector-tab-coverage" data-inspector-tab="coverage" type="button" role="tab" aria-controls="inspector" aria-selected="false">Coverage</button></div><div class="inspector-title" id="inspector-title" aria-live="polite">Nothing selected</div><div class="inspector-content" id="inspector" role="tabpanel" aria-labelledby="inspector-tab-explain">Select a graph node, port, edge, definition, layer, logical op, hotspot or diagnostic.</div></div>
    </aside>
  </section>
  <details class="analysis-shell" id="supporting-analysis">
    <summary><strong>Supporting analysis</strong><small>Capture summary, structure index, formula cost, roofline, workload diff and imported runtime</small></summary>
    <div class="analysis-grid">
      <section class="wide"><h2>Capture summary</h2><div class="capture-grid" id="captures">__CAPTURES__</div></section>
      <section class="wide structure-index"><h2>Structure index</h2><div class="split"><div><h3>Semantic structure</h3><div class="list" id="semantic-nodes"></div></div><div id="logical-block"><h3>Captured logical ops</h3><div class="notice">Tiny/meta semantic representative only — not full-model performance evidence.</div><div class="list" id="logical-ops"></div></div></div></section>
      <section class="wide"><h2>Scenario and formula cost</h2><div class="scenario-card" id="scenario-card"></div><div class="notice">Formula/estimated work and logical minimum traffic. These values are not measured latency or physical HBM traffic.</div><div class="split"><div><h3>Model-level decoder aggregates</h3><div id="cost-table"></div></div><div><h3>Theoretical hotspots</h3><div id="hotspot-table"></div></div></div></section>
      <section><h2>Roofline lower bounds</h2><div class="notice">Theoretical lower bounds — not a latency estimate.</div><div id="roofline-table"></div></section>
      <section><h2>Same-model workload diff</h2><div id="diff-table"></div></section>
      <section><h2>Runtime · external trace only</h2><div class="notice">No trace was imported. Unknown is never replaced by zero or a formula estimate.</div><div id="runtime-table"></div></section>
    </div>
  </details>
</main>
<script type="application/json" id="llm-vis-data">__EMBEDDED__</script>
<script>
const data=JSON.parse(document.getElementById('llm-vis-data').textContent);
const map=data.modelMap, graphView=data.graphView, manifest=data.manifest,hardwareProfile=data.hardwareProfile||null;
const byId=id=>document.getElementById(id);
const fmt=value=>value===null||value===undefined?'Unknown':(typeof value==='number'?value.toLocaleString(undefined,{maximumSignificantDigits:7}):String(value));
const pct=value=>`${(Number(value||0)*100).toFixed(1)}%`;
const clear=node=>{ while(node.firstChild) node.removeChild(node.firstChild); };
const emptyInspectorSelection={message:'Select a graph node, port, edge, definition, layer, logical op, hotspot or diagnostic.'};
let inspectorSelection=emptyInspectorSelection,inspectorTab='explain',lastGraphDetail=null;
const graphDetailByView=new Map();
let activeDrawerId=null,lastDrawerTrigger=null;
function closeWorkspaceDrawers({restoreFocus=false}={}){
  document.querySelectorAll('.workspace-drawer').forEach(drawer=>{drawer.classList.remove('is-open');drawer.setAttribute('aria-hidden','true');});
  document.querySelectorAll('[data-drawer-target]').forEach(button=>button.setAttribute('aria-expanded','false'));
  byId('drawer-scrim').hidden=true;activeDrawerId=null;
  if(restoreFocus&&lastDrawerTrigger)lastDrawerTrigger.focus();
}
function openWorkspaceDrawer(id,trigger=null){
  const drawer=byId(id);if(!drawer)return;
  closeWorkspaceDrawers();drawer.classList.add('is-open');drawer.setAttribute('aria-hidden','false');byId('drawer-scrim').hidden=false;activeDrawerId=id;
  document.querySelectorAll(`[data-drawer-target="${id}"]`).forEach(button=>button.setAttribute('aria-expanded','true'));
  lastDrawerTrigger=trigger||document.querySelector(`[data-drawer-target="${id}"]`);
  requestAnimationFrame(()=>drawer.querySelector('.drawer-close')?.focus());
}
document.querySelectorAll('[data-drawer-target]').forEach(button=>{button.onclick=()=>activeDrawerId===button.dataset.drawerTarget?closeWorkspaceDrawers({restoreFocus:true}):openWorkspaceDrawer(button.dataset.drawerTarget,button);});
document.querySelectorAll('[data-drawer-close]').forEach(button=>{button.onclick=()=>closeWorkspaceDrawers({restoreFocus:true});});
byId('drawer-scrim').onclick=()=>closeWorkspaceDrawers({restoreFocus:true});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&activeDrawerId)closeWorkspaceDrawers({restoreFocus:true});});
byId('analysis-toggle').onclick=()=>{const panel=byId('supporting-analysis');panel.open=true;panel.scrollIntoView({behavior:'smooth',block:'start'});};
const compact=items=>items.filter(item=>item!==null&&item!==undefined);
const graphInspectLimit=48;
const graphPreview=items=>(items||[]).slice(0,graphInspectLimit);
function graphNodeSummary(node){if(!node)return null;return {...node,subject_ids:graphPreview(node.subject_ids),subject_id_count:(node.subject_ids||[]).length};}
function inspectorLabel(value){const subject=value.graphPort||value.graphEdge||value.graphNode||value.node||value.op||value.instance||value.definition||value.metric||value.diagnostic||value.capture||value.layer||value.hotspot||value.workload_diff||value.roofline||value;return subject.label||subject.name||subject.code||subject.op_type||subject.module_path||subject.metric_name||subject.phase||subject.kind||subject.key||subject.id||'Selection';}
function inspectorMetrics(value){return compact([...(value.metrics||[]),value.metric]);}
function inspectorDiagnostics(value){return compact([...(value.diagnostics||[]),value.diagnostic]);}
function inspectorPayload(value,tab){
  const metrics=inspectorMetrics(value),diagnostics=inspectorDiagnostics(value),tensors=value.tensors||[],ports=value.ports||[],lowerings=value.lowerings||[],sources=value.sourceArtifacts||[];
  if(tab==='tensors')return tensors.length||ports.length?{ports,logical_ops:value.ops||[],tensors}:{status:'Unknown',reason:'No Tensor or port contract is attached to this selection.'};
  if(tab==='cost'){const costs=metrics.filter(item=>!item.name?.startsWith('runtime.')),theory=value.theoretical_bottleneck||value.heat_overlay,reconciliation=value.decomposition_cost_reconciliation;return costs.length||theory||reconciliation?{theoretical_bottleneck:value.theoretical_bottleneck||null,current_heat_overlay:value.heat_overlay||null,decomposition_cost_reconciliation:reconciliation||null,metric_scope:value.inspection_scope?.metric_scope||null,metrics:costs,symbols:map.symbols}:{status:'Unknown',reason:value.inspection_scope?.metric_unknown_reason||'No Scenario cost Metric is attached to this selection.'};}
  if(tab==='runtime'){const runtime=metrics.filter(item=>item.name?.startsWith('runtime.'));return runtime.length?runtime:{status:'Unknown',reason:'No external trace was imported; measured Runtime is unavailable.'};}
  if(tab==='provenance')return {model:manifest.model,tool:manifest.tool,safety:manifest.safety,graph_view:value.graphView||null,source_artifacts:sources,evidence:compact([...(value.graphNode?.evidence||[]),...(value.graphPort?.evidence||[]),...(value.graphEdge?.evidence||[]),...(value.node?.evidence||[]),...(value.metric?.assumptions||[]),...(value.diagnostic?.evidence||[])]),lowerings,captures:value.captures||[]};
  if(tab==='coverage')return {selection_coverage:value.coverage_status??'config-only-or-not-applicable',inspection_scope:value.inspection_scope||null,metrics:metrics.map(item=>({name:item.name,value:item.value,origin:item.origin,coverage:item.coverage,coverage_status:item.coverage_status,assumptions:item.assumptions})),diagnostics,capture_scope:value.captures||[],unknown_is_zero:false};
  return value;
}
function appendInspectorSection(parent,title,className=''){
  const section=document.createElement('section');section.className=`inspector-section ${className}`.trim();
  const heading=document.createElement('h3');heading.textContent=title;section.append(heading);parent.append(section);return section;
}
function renderInspectorJson(panel,value){
  const pre=document.createElement('pre');pre.className='inspector-json';pre.textContent=JSON.stringify(value,null,2);panel.append(pre);
}
function graphPortContract(port){
  const shape=port.shape_known===false||!Array.isArray(port.shape)?'shape Unknown':`[${port.shape.join(',')}]`;
  return `${shape} · ${port.dtype||'dtype Unknown'}`;
}
function renderNodeExplanation(panel,value){
  const node=value.graphNode,detail=node?.detail||{},ports=value.ports||[];
  if(!node){renderInspectorJson(panel,inspectorPayload(value,'explain'));return;}
  const root=document.createElement('div');root.className='inspector-explain';
  const meta=document.createElement('div');meta.className='inspector-node-kind';
  for(const label of compact([node.primitive_kind||node.kind,node.decomposition_status,node.opaque?'opaque':null])){const badge=document.createElement('span');badge.textContent=label;meta.append(badge);}root.append(meta);
  const purpose=appendInspectorSection(root,'What this node does');const purposeText=document.createElement('p');purposeText.textContent=detail.description||'Semantic graph node; no additional explanation is available.';purpose.append(purposeText);
  const formula=appendInspectorSection(root,detail.formula_scope==='unknown'?'Formula status':'Simplified equation');const equation=document.createElement('code');equation.className='inspector-formula';equation.textContent=detail.formula||'Not applicable — this node does not perform arithmetic.';formula.append(equation);
  const inputs=ports.filter(port=>port.direction==='input'),outputs=ports.filter(port=>port.direction==='output');
  if(inputs.length||outputs.length){const io=appendInspectorSection(root,'Inputs and outputs');const list=document.createElement('div');list.className='inspector-port-list';for(const [direction,items] of [['IN',inputs],['OUT',outputs]])for(const port of items){const row=document.createElement('div');row.className='inspector-port';const name=document.createElement('span');name.textContent=`${direction} · ${port.name||port.key||port.id}`;const contract=document.createElement('small');contract.textContent=graphPortContract(port);row.append(name,contract);list.append(row);}io.append(list);}
  const children=Array.isArray(detail.child_views)?detail.child_views:[];
  const interaction=appendInspectorSection(root,'Interaction','inspector-interaction');const interactionText=document.createElement('p');interactionText.textContent=children.length?`This compound node has ${children.length} child graph${children.length===1?'':'s'}. Double-click enters the primary child; choose a child below when alternatives exist.`:'This is a leaf or opaque node. Double-click keeps the current graph and shows this explanation.';interaction.append(interactionText);
  if(children.length){const childList=document.createElement('div');childList.className='inspector-child-list';for(const child of children){const button=document.createElement('button');button.type='button';button.className='inspector-child';const label=document.createElement('span');label.textContent=child.label||child.id;const role=document.createElement('small');role.textContent=`${child.primary?'Primary':'Alternate'} · ${child.node_count||'?'} nodes · open ↓`;button.append(label,role);button.onclick=()=>{if(window.LLMVisDAG?.openView('model-dag',child.id))closeWorkspaceDrawers({restoreFocus:false});};childList.append(button);}interaction.append(childList);}
  const evidence=appendInspectorSection(root,'Evidence boundary');const evidenceText=document.createElement('p');evidenceText.className='inspector-evidence';evidenceText.textContent=detail.evidence_note||'Config-first semantic projection; not measured runtime.';evidence.append(evidenceText);
  panel.append(root);
}
function renderInspector(){
  document.querySelectorAll('.inspector-tab').forEach(button=>{const active=button.dataset.inspectorTab===inspectorTab;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;});
  const hasSelection=!inspectorSelection.message,label=hasSelection?inspectorLabel(inspectorSelection):'Nothing selected',panel=byId('inspector');byId('inspector-title').textContent=hasSelection?`${label} · ${inspectorTab}`:label;panel.setAttribute('aria-labelledby',`inspector-tab-${inspectorTab}`);clear(panel);if(!hasSelection)panel.textContent=inspectorSelection.message;else if(inspectorTab==='explain')renderNodeExplanation(panel,inspectorSelection);else renderInspectorJson(panel,inspectorPayload(inspectorSelection,inspectorTab));
  document.querySelectorAll('[data-drawer-target="inspector-drawer"]').forEach(button=>{button.textContent=hasSelection?'Inspector •':'Inspector';button.title=hasSelection?`Selected: ${label}`:'Open selection inspector';});
}
const setInspector=(value,{reveal=false}={})=>{inspectorSelection=value||{status:'Unknown'};renderInspector();if(reveal)openWorkspaceDrawer('inspector-drawer');};
const inspect=value=>{lastGraphDetail=null;setInspector(value,{reveal:true});};
function empty(node,message){ clear(node); const el=document.createElement('div'); el.className='empty'; el.textContent=message; node.append(el); }
function makeTable(headers,rows,onClick){
  const table=document.createElement('table'),head=document.createElement('thead'),hr=document.createElement('tr');
  for(const label of headers){const th=document.createElement('th');th.textContent=label;hr.append(th);} head.append(hr);table.append(head);
  const body=document.createElement('tbody');
  rows.forEach((row,index)=>{const tr=document.createElement('tr');if(onClick){tr.className='clickable';tr.onclick=()=>onClick(index);}for(const value of row){const td=document.createElement('td');td.textContent=fmt(value);if(value===null||value===undefined)td.className='unknown';tr.append(td);}body.append(tr);});
  table.append(body);return table;
}
function buttonItem(label,detail,value,className='item'){
  const el=document.createElement('button');el.type='button';el.className=className;const title=document.createElement('span');title.textContent=label;el.append(title);if(detail){const small=document.createElement('small');small.textContent=detail;el.append(small);}el.onclick=()=>inspect(value);return el;
}
byId('capability-badge').textContent=`${manifest.capability.level} · ${manifest.capability.cost_analyzed?'cost':'structure'}`;
const captureMode=manifest.captures.length?'bounded Tiny capture':'config only';
byId('capability-badge').title=`${captureMode}; full model not executed`;
const sourceSupport=manifest.capability.family_adapter_available?'Known adapter':'Unsupported · opaque';
const requestedRevision=manifest.source.requested_revision,resolvedRevision=manifest.source.resolved_revision||manifest.model.revision;
const revisionFlow=requestedRevision&&requestedRevision!==resolvedRevision?` · ${requestedRevision}→${String(resolvedRevision).slice(0,8)}`:'';
byId('source-evidence-badge').textContent=`${sourceSupport} · ${manifest.adapter.name}${revisionFlow}`;
byId('source-evidence-badge').title=`Input ${manifest.source.input_kind}; requested revision ${requestedRevision||'not declared'}; resolved revision ${resolvedRevision||'Unknown'}; config SHA-256 ${manifest.source.config_sha256}`;

function uniqueById(items){const seen=new Set();return items.filter(item=>{if(!item)return false;const key=item.id||item.capture_id||JSON.stringify(item);if(seen.has(key))return false;seen.add(key);return true;});}
function metricsFor(ids){const selected=new Set(ids);return map.metrics.filter(item=>selected.has(item.subject_id));}
function graphMetricsFor(ids){const scenarioId=byId('scenario-select')?.value||null;return metricsFor(ids).filter(item=>item.scenario_id===null||item.scenario_id===scenarioId);}
function scopedGraphMetrics(detail,node,view,subjectIds){
  const candidates=graphMetricsFor(subjectIds),type=detail?.type||'node';
  if(type==='edge')return {metrics:[],scope:'not-applicable-to-edge',unknown_reason:'A connection does not own component cost metrics.'};
  if(!node)return {metrics:[],scope:'unknown-selection',unknown_reason:'No owning graph node was resolved.'};
  const explicit=(node.metric_bindings||[]).filter(binding=>binding.status==='known'&&binding.metric_name);
  if(explicit.length){const names=new Set(explicit.map(binding=>binding.metric_name));return {metrics:candidates.filter(item=>names.has(item.name)),scope:'explicit-node-metric-binding',unknown_reason:null};}
  const unavailable=(node.metric_bindings||[]).find(binding=>binding.status==='unknown'||binding.status==='excluded');
  if(unavailable)return {metrics:[],scope:`explicit-${unavailable.status}`,unknown_reason:unavailable.reason};
  if(view?.level==='L0'){
    if(node.kind==='decoder_pattern')return {metrics:candidates,scope:'member-instance-estimates',unknown_reason:null};
    return {metrics:[],scope:'not-attached-to-L0-boundary',unknown_reason:'No independently attributable cost metric exists for this L0 boundary node.'};
  }
  const kind=String(node.kind||''),key=String(node.key||''),attentionKind=String(view?.metadata?.attention_kind||kind);
  if(kind==='linear_attention'||kind==='full_attention'){
    const names=attentionKind==='linear_attention'
      ?new Set(['state.bytes','recurrent_state.bytes','conv_state.bytes'])
      :new Set(['state.bytes','kv_cache.bytes','kv_cache.bytes_per_sequence_token']);
    return {metrics:candidates.filter(item=>item.name.endsWith('.attention')||names.has(item.name)),scope:'component-estimate',unknown_reason:null};
  }
  if(key==='ffn'&&candidates.some(item=>item.name.endsWith('.ffn'))){
    return {metrics:candidates.filter(item=>item.name.endsWith('.ffn')),scope:'component-estimate',unknown_reason:null};
  }
  if(kind==='state_boundary_in'||kind==='state_boundary_out'){
    const relevant=attentionKind==='linear_attention'
      ?new Set(['state.bytes','recurrent_state.bytes','conv_state.bytes'])
      :new Set(['state.bytes','kv_cache.bytes','kv_cache.bytes_per_sequence_token']);
    return {metrics:candidates.filter(item=>relevant.has(item.name)),scope:'state-contract-estimate',unknown_reason:null};
  }
  return {metrics:[],scope:'not-attributable',unknown_reason:node.opaque?'Opaque internals prevent trustworthy component attribution.':'Available estimates are block aggregates and are not attributed to this synthetic semantic node.'};
}
function currentScenario(){const id=byId('scenario-select')?.value||'';return map.scenarios.find(item=>item.id===id)||null;}
function graphMetricBinding(node,baseName){return (node?.metric_bindings||[]).find(binding=>binding.dimension===baseName)||null;}
function metricNameForGraphNode(node,view,baseName){
  const binding=graphMetricBinding(node,baseName);if(binding?.status==='known')return binding.metric_name;
  if(view?.level==='L0'&&node?.kind==='decoder_pattern')return baseName;
  const kind=String(node?.kind||''),key=String(node?.key||'');
  if(view?.level==='L1'&&(kind==='linear_attention'||kind==='full_attention'))return `${baseName}.attention`;
  if(view?.level==='L1'&&key==='ffn')return `${baseName}.ffn`;
  return null;
}
function graphMetricRollup(node,view,scenario,baseName,unit){
  const binding=graphMetricBinding(node,baseName);
  if(binding&&binding.status!=='known')return {known:false,status:binding.status==='excluded'?'not-applicable':'unknown',name:binding.parent_metric_name||baseName,value:null,unit,origin:'unknown',coverage:0,reason:binding.reason||`${baseName} is ${binding.status}.`,metric_binding:binding};
  const metricName=metricNameForGraphNode(node,view,baseName);
  if(!metricName){
    return {known:false,status:node?.opaque?'unknown':'not-applicable',name:baseName,value:null,unit,origin:'unknown',coverage:0,reason:node?.opaque?'Opaque internals prevent trustworthy component attribution.':'No independently attributable component metric exists for this semantic node.'};
  }
  const subjectIds=[...new Set(node?.subject_ids||[])];
  const scoped=scopedGraphMetrics({type:'node'},node,view,subjectIds);
  const matching=scoped.metrics.filter(metric=>metric.scenario_id===scenario?.id&&metric.name===metricName&&metric.unit===unit);
  const known=matching.filter(metric=>metric.value!==null&&metric.value!==undefined&&metric.origin!=='unknown'&&Number.isFinite(Number(metric.value)));
  const instanceIds=new Set(map.instances.map(instance=>instance.id));
  const contributorIds=subjectIds.filter(id=>instanceIds.has(id));
  const expected=Math.max(1,contributorIds.length||matching.length);
  const assumptions=[...new Set(matching.flatMap(metric=>metric.assumptions||[]))];
  if(!known.length){
    const evidenceReason=assumptions.filter(item=>/unknown|required|opaque|unavailable/i.test(item)).slice(0,2).join(' ');
    return {known:false,status:'unknown',name:metricName,value:null,unit,origin:'unknown',coverage:0,contributor_count:0,expected_contributor_count:expected,source_metric_ids:matching.map(metric=>metric.id),reason:evidenceReason||scoped.unknown_reason||`${metricName} is Unknown for this node and Scenario.`};
  }
  const value=known.reduce((total,metric)=>total+Number(metric.value),0);
  const coverage=known.reduce((total,metric)=>total+Number(metric.coverage||0),0)/expected;
  const origins=[...new Set(known.map(metric=>metric.origin))].sort();
  const partial=known.length<expected||coverage<1;
  return {known:true,status:partial?'partial':'known',name:metricName,value,unit,origin:origins.length===1?origins[0]:`mixed(${origins.join(',')})`,origins,coverage:Math.min(1,coverage),contributor_count:known.length,expected_contributor_count:expected,source_metric_ids:known.map(metric=>metric.id),assumptions,partial};
}
function hardwareCompatibility(scenario){
  if(!hardwareProfile)return {known:false,reason:'No explicit HardwareProfile was supplied.'};
  if(!scenario)return {known:false,reason:'No Scenario is selected.'};
  if(String(hardwareProfile.dtype)!==String(scenario.activation_dtype))return {known:false,reason:`HardwareProfile dtype ${hardwareProfile.dtype} does not match Scenario activation dtype ${scenario.activation_dtype}.`};
  return {known:true,reason:null};
}
function profileEvidence(){
  if(!hardwareProfile)return null;
  return {id:hardwareProfile.id,name:hardwareProfile.name,dtype:hardwareProfile.dtype,peak_flops_per_second:hardwareProfile.peak_flops_per_second,memory_bandwidth_bytes_per_second:hardwareProfile.memory_bandwidth_bytes_per_second,provenance:hardwareProfile.provenance};
}
function compactMagnitude(value,unit){
  if(value===null||value===undefined||!Number.isFinite(Number(value)))return 'Unknown';
  const numeric=Number(value),number=(scaled,suffix)=>`${scaled.toLocaleString(undefined,{maximumSignificantDigits:4})} ${suffix}`;
  if(unit==='seconds'){
    if(Math.abs(numeric)<1e-6)return number(numeric*1e9,'ns');
    if(Math.abs(numeric)<1e-3)return number(numeric*1e6,'µs');
    if(Math.abs(numeric)<1)return number(numeric*1e3,'ms');
    return number(numeric,'s');
  }
  if(unit==='FLOPs'){
    const scales=[[1e15,'PFLOPs'],[1e12,'TFLOPs'],[1e9,'GFLOPs'],[1e6,'MFLOPs'],[1e3,'kFLOPs']];
    const scale=scales.find(([threshold])=>Math.abs(numeric)>=threshold);return scale?number(numeric/scale[0],scale[1]):number(numeric,'FLOPs');
  }
  if(unit==='bytes'){
    const scales=[[2**40,'TiB'],[2**30,'GiB'],[2**20,'MiB'],[2**10,'KiB']];
    const scale=scales.find(([threshold])=>Math.abs(numeric)>=threshold);return scale?number(numeric/scale[0],scale[1]):number(numeric,'bytes');
  }
  return number(numeric,unit);
}
function theoreticalNodeHeat(node,view,scenario,mode='pressure'){
  if(mode==='off')return {known:false,status:'off',mode,reason:'Heat overlay is off.'};
  const compute=graphMetricRollup(node,view,scenario,'flops','FLOPs');
  const memory=graphMetricRollup(node,view,scenario,'logical_bytes','bytes');
  const profile=profileEvidence(),compatibility=hardwareCompatibility(scenario);
  if(mode==='compute'){
    if(!compute.known)return {...compute,mode,label:'Theoretical compute',hardware_profile:profile};
    const peak=Number(hardwareProfile?.peak_flops_per_second),lowerBound=compatibility.known&&Number.isFinite(peak)?compute.value/peak:null;
    return {...compute,mode,label:'Theoretical compute',raw_value:compute.value,value_label:compactMagnitude(compute.value,'FLOPs'),compute_lower_bound_seconds:lowerBound,hardware_profile:profile,is_latency_estimate:false};
  }
  if(mode==='memory'){
    if(!memory.known)return {...memory,mode,label:'Theoretical memory traffic',hardware_profile:profile};
    const bandwidth=Number(hardwareProfile?.memory_bandwidth_bytes_per_second),lowerBound=compatibility.known&&Number.isFinite(bandwidth)?memory.value/bandwidth:null;
    return {...memory,mode,label:'Theoretical memory traffic',raw_value:memory.value,value_label:compactMagnitude(memory.value,'bytes'),bandwidth_lower_bound_seconds:lowerBound,hardware_profile:profile,traffic_scope:'logical minimum bytes; not physical HBM traffic',is_latency_estimate:false};
  }
  if(!compute.known||!memory.known)return {known:false,status:compute.status==='not-applicable'&&memory.status==='not-applicable'?'not-applicable':'unknown',mode,label:'Theoretical pressure',value:null,unit:'seconds',compute,memory,hardware_profile:profile,reason:!compute.known?compute.reason:memory.reason,is_latency_estimate:false};
  if(!compatibility.known)return {known:false,status:'unknown',mode,label:'Theoretical pressure',value:null,unit:'seconds',compute,memory,hardware_profile:profile,reason:compatibility.reason,is_latency_estimate:false};
  const peak=Number(hardwareProfile.peak_flops_per_second),bandwidth=Number(hardwareProfile.memory_bandwidth_bytes_per_second);
  if(!Number.isFinite(peak)||!Number.isFinite(bandwidth))return {known:false,status:'unknown',mode,label:'Theoretical pressure',value:null,unit:'seconds',compute,memory,hardware_profile:profile,reason:'HardwareProfile peak FLOPs/s or memory bandwidth is Unknown.',is_latency_estimate:false};
  const computeBound=compute.value/peak,bandwidthBound=memory.value/bandwidth,value=Math.max(computeBound,bandwidthBound),scale=Math.max(computeBound,bandwidthBound,Number.EPSILON);
  const balanced=Math.abs(computeBound-bandwidthBound)<=scale*1e-12;
  const bottleneck=balanced?'balanced':computeBound>bandwidthBound?'compute':'bandwidth';
  const partial=Boolean(compute.partial||memory.partial);
  return {known:true,status:partial?'partial':'known',mode,label:'Theoretical pressure',raw_value:value,value,value_label:compactMagnitude(value,'seconds'),unit:'seconds',compute_lower_bound_seconds:computeBound,bandwidth_lower_bound_seconds:bandwidthBound,bottleneck_class:bottleneck,arithmetic_intensity_flops_per_byte:memory.value?compute.value/memory.value:null,coverage:Math.min(compute.coverage,memory.coverage),origin:`max(${compute.origin},${memory.origin})`,compute,memory,hardware_profile:profile,partial,is_latency_estimate:false};
}
function heatmapForView(view,scenario,mode){
  if(!view||mode==='off')return {mode:'off',nodes:{},title:'Theoretical heat',basis:'Heat overlay is off.',coverageLabel:''};
  const frontierIds=new Set(view.cost_frontier_node_ids||[]),frontier=(view.nodes||[]).filter(node=>frontierIds.has(node.id));
  const frontierSet=new Set(frontier.map(node=>node.id));
  const results=frontier.map(node=>({node,result:theoreticalNodeHeat(node,view,scenario,mode)}));
  const known=results.filter(item=>item.result.known&&Number.isFinite(Number(item.result.raw_value)));
  const hottest=Math.max(0,...known.map(item=>Number(item.result.raw_value))),total=known.reduce((sum,item)=>sum+Number(item.result.raw_value),0);
  known.sort((left,right)=>Number(right.result.raw_value)-Number(left.result.raw_value)).forEach((item,index)=>{item.result.rank=index+1;});
  const nodes={};
  for(const {node,result} of results){
    const normalized=result.known?(hottest===0?0:Number(result.raw_value)/hottest):null;
    const contribution=result.known&&total>0?Number(result.raw_value)/total:null;
    const dominant=result.bottleneck_class?` · ${result.bottleneck_class}-bound`:'';
    const relative=result.known?` · ${Math.round(normalized*100)}% of hottest known node`:'';
    const coverage=result.known?` · ${pct(result.coverage)}`:'';
    const valueLabel=result.value_label||compactMagnitude(result.value,result.unit);
    const accessibilityLabel=result.known?`${result.label}: ${valueLabel}${dominant}${relative}${coverage}. Formula lower bound; not measured latency.`:`${result.label||'Theoretical heat'}: Unknown. ${result.reason||'No trustworthy attribution.'} Unknown is not zero.`;
    nodes[node.id]={...result,normalized,contribution,valueLabel,accessibilityLabel};
  }
  for(const node of view.nodes||[]){if(!frontierSet.has(node.id))nodes[node.id]={known:false,status:'not-applicable',mode,label:'Outside visible cost frontier',normalized:null,contribution:null,valueLabel:'Not applicable',reason:'Boundary/container node is visible but excluded from the current cost frontier.',accessibilityLabel:'Outside the current visible cost frontier; not included in heat statistics.'};}
  const title=mode==='compute'?'Theoretical compute':mode==='memory'?'Theoretical memory':'Theoretical pressure';
  const provenance=hardwareProfile?`${hardwareProfile.name} (${hardwareProfile.id}; ${hardwareProfile.provenance?.kind||'unknown provenance'}: ${hardwareProfile.provenance?.source||'source Unknown'})`:'no HardwareProfile';
  const formula=mode==='pressure'?`max(FLOPs / peak, logical bytes / bandwidth) using ${provenance}`:mode==='compute'?'formula/estimated FLOPs':'logical minimum bytes (not physical HBM traffic)';
  const partial=known.filter(item=>item.result.partial).length;
  return {mode,nodes,title,basis:`${formula}; current visible cost frontier only; raw ÷ hottest known frontier node; no parent/child double count; Unknown is not zero; not measured latency.`,coverageLabel:`frontier ${known.length}/${results.length} known${partial?` · ${partial} partial`:''} · ${scenario?.phase||'no Scenario'}`};
}
function currentGraphView(){return window.LLMVisDAG?.get('model-dag')?.view||graphView.views[0]||null;}
function renderHeatmap(scenario=currentScenario(),view=currentGraphView()){
  const mode=byId('heatmap-mode')?.value||'pressure';
  const config=heatmapForView(view,scenario,mode);
  window.LLMVisDAG?.setHeatmap('model-dag',config);
  return config;
}
function diagnosticsFor(ids){const selected=new Set(ids);return map.diagnostics.filter(item=>item.subject_id===null||selected.has(item.subject_id));}
function tensorsFor(ids){const selected=new Set(ids);return map.tensors.filter(item=>item.owner_id&&selected.has(item.owner_id));}
function loweringsFor(ids){const selected=new Set(ids);return map.lowerings.filter(item=>item.source_ids.some(id=>selected.has(id))||item.target_ids.some(id=>selected.has(id)));}
function graphForLowerings(lowerings,extraOps=[]){
  const opIds=new Set(lowerings.flatMap(item=>item.target_ids));
  const ops=uniqueById([...extraOps,...map.logical_ops.filter(item=>opIds.has(item.id))]);
  const tensorIds=new Set(ops.flatMap(item=>[...item.inputs,...item.outputs]));
  return {ops,tensors:map.tensors.filter(item=>tensorIds.has(item.id))};
}
function capturesForLayer(layerIndex){return data.captures.filter(item=>item.representative_layer_index===layerIndex);}
function capturesForLowerings(lowerings){
  const opIds=new Set(lowerings.flatMap(item=>item.target_ids));
  const captureIds=new Set(map.logical_ops.filter(item=>opIds.has(item.id)).map(item=>item.attrs?.capture_id).filter(Boolean));
  return data.captures.filter(item=>captureIds.has(item.capture_id));
}
function sourceArtifactsFor(captures=[],tensors=[]){
  const allCaptureSourceIds=new Set(data.captures.flatMap(item=>compact([item.source_artifact_id,...(item.source_artifact_ids||[])])));
  const ids=new Set(map.model.source_artifact_ids.filter(id=>!allCaptureSourceIds.has(id)));
  captures.forEach(item=>{if(item.source_artifact_id)ids.add(item.source_artifact_id);(item.source_artifact_ids||[]).forEach(id=>ids.add(id));});
  tensors.forEach(item=>(item.source_artifact_ids||[]).forEach(id=>ids.add(id)));
  return map.source_artifacts.filter(item=>ids.has(item.id));
}
function decompositionCostSummary(node,view,scenario){
  const child=graphView.views.find(candidate=>candidate.id===node?.drilldown_view_id);if(!child||!(child.cost_reconciliations||[]).length)return null;
  const childIds=new Set(child.cost_frontier_node_ids||[]),children=(child.nodes||[]).filter(item=>childIds.has(item.id));
  return {child_view_id:child.id,visible_frontier_node_ids:children.map(item=>item.id),dimensions:(child.cost_reconciliations||[]).map(contract=>{
    const unit=contract.dimension==='flops'?'FLOPs':'bytes',parent=graphMetricRollup(node,view,scenario,contract.dimension,unit);
    const childResults=children.map(item=>({node_id:item.id,label:item.label,result:graphMetricRollup(item,child,scenario,contract.dimension,unit)}));
    const known=childResults.filter(item=>item.result.known),knownSubtotal=known.reduce((sum,item)=>sum+Number(item.result.value),0);
    const expectedKnownNames=new Set(contract.known_child_metric_names||[]),observedKnownNames=new Set(known.map(item=>item.result.name).filter(Boolean));
    const unexpectedKnown=known.filter(item=>!expectedKnownNames.has(item.result.name));
    const missingExpectedKnownNames=[...expectedKnownNames].filter(name=>!observedKnownNames.has(name));
    const signedRemainder=parent.known?Number(parent.value)-knownSubtotal:null;
    const tolerance=parent.known?Math.max(1,Math.abs(Number(parent.value)),Math.abs(knownSubtotal))*1e-12:0;
    const inconsistencyReasons=[];
    if(unexpectedKnown.length)inconsistencyReasons.push('Known child metrics appeared outside the explicit reconciliation contract.');
    if(parent.known&&contract.status==='complete'&&missingExpectedKnownNames.length)inconsistencyReasons.push('Expected known child metrics are missing from a complete reconciliation.');
    if(parent.known&&signedRemainder < -tolerance)inconsistencyReasons.push('Known child subtotal exceeds the parent metric.');
    if(parent.known&&contract.status==='complete'&&Math.abs(signedRemainder)>tolerance)inconsistencyReasons.push('Complete reconciliation does not sum to the parent metric.');
    const remainderIsZero=parent.known&&Math.abs(signedRemainder)<=tolerance;
    return {dimension:contract.dimension,status:contract.status,parent,known_child_subtotal:known.length?knownSubtotal:null,known_child_count:known.length,unknown_child_count:childResults.filter(item=>item.result.status==='unknown').length,excluded_child_count:childResults.filter(item=>item.result.status==='not-applicable').length,signed_remainder:signedRemainder,signed_remainder_is_zero:remainderIsZero,unattributed:signedRemainder,unattributed_is_zero:remainderIsZero,coverage:parent.known&&Number(parent.value)>0?knownSubtotal/Number(parent.value):null,unexpected_known_child_ids:unexpectedKnown.map(item=>item.node_id),unexpected_known_metric_names:unexpectedKnown.map(item=>item.result.name),missing_expected_known_metric_names:missingExpectedKnownNames,inconsistent:inconsistencyReasons.length>0,inconsistency_reasons:inconsistencyReasons,contract,children:childResults,unknown_is_zero:false};
  })};
}
function graphSelectionContext(detail){
  const item=detail?.item||{},view=detail?.view||{},viewPorts=view.ports||[],portsById=new Map(viewPorts.map(port=>[port.id,port]));
  const sourcePort=portsById.get(item.source_port_id),targetPort=portsById.get(item.target_port_id);
  const selectedNode=detail?.type==='node'?item:(detail?.node||null);
  const endpointNodes=(view.nodes||[]).filter(node=>node.id===sourcePort?.node_id||node.id===targetPort?.node_id);
  const subjectIds=uniqueById([...(selectedNode?.subject_ids||[]),...endpointNodes.flatMap(node=>node.subject_ids||[])]);
  const ports=detail?.type==='node'?(item.ports||viewPorts.filter(port=>port.node_id===item.id)):detail?.type==='port'?[item]:compact([sourcePort,targetPort]);
  const tensorIds=new Set(ports.map(port=>port.tensor_spec_id).filter(Boolean));
  if(item.tensor_spec_id)tensorIds.add(item.tensor_spec_id);
  const tensors=map.tensors.filter(tensor=>tensorIds.has(tensor.id));
  const captures=[]; // Recursive GraphView is config semantic projection; bounded capture remains separate evidence.
  const metricSelection=scopedGraphMetrics(detail,selectedNode,view,subjectIds),selectedMetrics=metricSelection.metrics,selectedDiagnostics=diagnosticsFor(subjectIds),selectedSemantics=map.semantic_nodes.filter(node=>subjectIds.includes(node.id)),selectedInstances=map.instances.filter(instance=>subjectIds.includes(instance.id)),selectedDefinitions=map.definitions.filter(definition=>subjectIds.includes(definition.id));
  const context={
    graphView:{id:view.id,key:view.key,level:view.level,label:view.label,breadcrumb:view.breadcrumb,layer_index:view.layer_index,decomposes_node_id:view.decomposes_node_id,boundary_bindings:view.boundary_bindings,cost_frontier_node_ids:view.cost_frontier_node_ids,cost_reconciliations:view.cost_reconciliations,metadata:view.metadata},
    ports,tensors,metrics:graphPreview(selectedMetrics),diagnostics:graphPreview(selectedDiagnostics),captures,
    semantic_nodes:graphPreview(selectedSemantics),instances:graphPreview(selectedInstances),definitions:graphPreview(selectedDefinitions),
    inspection_scope:{projection:'config-first-recursive-semantic-DAG',capture_scope:'not-applicable',metric_scope:metricSelection.scope,metric_unknown_reason:metricSelection.unknown_reason,shape_unknown_reason:item.shape_known===false?item.unknown_reason:null,subject_count:subjectIds.length,preview_limit:graphInspectLimit,metrics_total:selectedMetrics.length,diagnostics_total:selectedDiagnostics.length,semantic_nodes_total:selectedSemantics.length,instances_total:selectedInstances.length,definitions_total:selectedDefinitions.length},
    coverage_status:item.opaque?'opaque':(item.shape_known===false?'unknown':(item.coverage_status??(item.coverage===0?'unknown':item.coverage))),
    sourceArtifacts:sourceArtifactsFor(captures,tensors),
    upstream_ids:detail?.upstreamIds||[],downstream_ids:detail?.downstreamIds||[],
  };
  if(selectedNode){
    const scenario=currentScenario(),mode=byId('heatmap-mode')?.value||'pressure';
    context.theoretical_bottleneck=theoreticalNodeHeat(selectedNode,view,scenario,'pressure');
    context.heat_overlay=theoreticalNodeHeat(selectedNode,view,scenario,mode==='off'?'pressure':mode);
    context.decomposition_cost_reconciliation=decompositionCostSummary(selectedNode,view,scenario);
  }
  if(detail?.type==='node')context.graphNode=graphNodeSummary(item);
  else if(detail?.type==='port'){context.graphPort=item;context.graphNode=graphNodeSummary(selectedNode);}
  else if(detail?.type==='edge'){context.graphEdge=item;context.endpoint_nodes=endpointNodes.map(graphNodeSummary);}
  return context;
}
window.llmVisInspect=detail=>{
  lastGraphDetail=detail;
  if(detail?.view?.id)graphDetailByView.set(detail.view.id,detail);
  setInspector(graphSelectionContext(detail),{reveal:detail?.reveal===true});
};
function semanticContext(item){
  const instances=map.instances.filter(instance=>item.instance_ids.includes(instance.id));
  const semanticIds=[item.id,...item.child_ids];
  const lowerings=loweringsFor(semanticIds),graph=graphForLowerings(lowerings);
  const subjectIds=[...semanticIds,...instances.map(instance=>instance.id)];
  const directTensors=map.tensors.filter(tensor=>item.input_tensor_ids.includes(tensor.id)||item.output_tensor_ids.includes(tensor.id));
  const tensors=uniqueById([...directTensors,...tensorsFor(subjectIds),...graph.tensors]);
  const captures=capturesForLowerings(lowerings);
  return {node:item,children:map.semantic_nodes.filter(node=>item.child_ids.includes(node.id)),instances,lowerings,ops:graph.ops,tensors,metrics:metricsFor(subjectIds),diagnostics:diagnosticsFor(subjectIds),captures,sourceArtifacts:sourceArtifactsFor(captures,tensors)};
}
function definitionContext(item){
  const instances=map.instances.filter(instance=>instance.definition_id===item.id);
  const semantics=map.semantic_nodes.filter(node=>node.definition_id===item.id||node.instance_ids.some(id=>instances.some(instance=>instance.id===id)));
  const lowerings=loweringsFor(semantics.map(node=>node.id)),graph=graphForLowerings(lowerings);
  const captures=capturesForLowerings(lowerings);
  const subjectIds=[item.id,...instances.map(instance=>instance.id),...semantics.map(node=>node.id)];
  const tensors=uniqueById([...tensorsFor(subjectIds),...graph.tensors]);
  return {definition:item,instances,semantic_nodes:semantics,lowerings,ops:graph.ops,tensors,metrics:metricsFor(subjectIds),diagnostics:diagnosticsFor(subjectIds),captures,sourceArtifacts:sourceArtifactsFor(captures,tensors)};
}
function logicalContext(item){
  const lowerings=loweringsFor([item.id]),graph=graphForLowerings(lowerings,[item]),captures=capturesForLowerings(lowerings);
  const sourceNodes=map.semantic_nodes.filter(node=>lowerings.some(lowering=>lowering.source_ids.includes(node.id)));
  return {op:item,semantic_nodes:sourceNodes,lowerings,ops:graph.ops,tensors:graph.tensors,diagnostics:diagnosticsFor([item.id,...sourceNodes.map(node=>node.id)]),captures,sourceArtifacts:sourceArtifactsFor(captures,graph.tensors)};
}
function metricContext(metric,extra={}){return {metric,...extra,diagnostics:diagnosticsFor([metric.subject_id]),sourceArtifacts:sourceArtifactsFor()};}
function captureContext(capture){
  const instance=map.instances.find(item=>item.layer_index===capture.representative_layer_index),semantics=map.semantic_nodes.filter(node=>node.instance_ids.includes(instance?.id));
  const representativeIds=new Set(map.semantic_nodes.filter(node=>(node.evidence||[]).includes(`representative_kind=${capture.kind}`)).map(node=>node.id));
  const lowerings=map.lowerings.filter(item=>item.source_ids.some(id=>representativeIds.has(id))),graph=graphForLowerings(lowerings);
  const subjectIds=[instance?.id,...semantics.map(node=>node.id)].filter(Boolean);
  return {capture,instance,semantic_nodes:semantics,lowerings,ops:graph.ops,tensors:graph.tensors,diagnostics:diagnosticsFor(subjectIds),captures:[capture],coverage_status:capture.status,sourceArtifacts:sourceArtifactsFor([capture],graph.tensors)};
}

function renderDefinitions(query=''){
  const root=byId('definitions');clear(root);const q=query.toLowerCase();
  map.definitions.filter(item=>`${item.label} ${item.id} ${item.kind}`.toLowerCase().includes(q)).forEach(item=>root.append(buttonItem(item.label,`${item.kind} · ${item.id}`,definitionContext(item))));
  byId('definition-count').textContent=`${root.children.length}/${map.definitions.length}`;
  if(!root.children.length)empty(root,'No matching definitions.');
}
function renderSemantics(query=''){
  const root=byId('semantic-nodes');clear(root);const q=query.toLowerCase();
  map.semantic_nodes.filter(item=>`${item.label} ${item.id} ${item.kind} ${(item.evidence||[]).join(' ')}`.toLowerCase().includes(q)).forEach(item=>root.append(buttonItem(item.label,`${item.kind} · coverage evidence ${item.evidence.length}`,semanticContext(item))));
  if(!root.children.length)empty(root,'No matching semantic nodes.');
}
function renderLogical(query=''){
  const block=byId('logical-block'),root=byId('logical-ops');
  if(!map.logical_ops.length){block.hidden=true;return;}block.hidden=false;clear(root);const q=query.toLowerCase();
  map.logical_ops.filter(item=>`${item.op_type} ${item.id} ${item.domain}`.toLowerCase().includes(q)).forEach(item=>root.append(buttonItem(item.op_type,`${item.domain} · ${item.inputs.length} inputs · ${item.outputs.length} outputs`,logicalContext(item))));
  if(!root.children.length)empty(root,'No matching logical ops.');
}
function renderDiagnostics(){const root=byId('diagnostics');clear(root);map.diagnostics.forEach(item=>root.append(buttonItem(item.code,`${item.severity} · ${item.message}`,{diagnostic:item,sourceArtifacts:sourceArtifactsFor()},'item diag')));byId('diagnostic-count').textContent=String(map.diagnostics.length);if(!root.children.length)empty(root,'No diagnostics.');}
function graphViewForLayer(layer){return graphView.views.find(view=>view.layer_index===layer.layer_index)||graphView.views.find(view=>view.level==='L1'&&view.metadata?.attention_kind===layer.attention_kind&&view.metadata?.mlp_kind===layer.mlp_kind);}
function graphTargetForSubject(subjectId){for(const view of graphView.views){const node=view.nodes.find(candidate=>(candidate.subject_ids||[]).includes(subjectId));if(node)return {view,node};}return null;}
function locateGraphSubject(subjectId){const target=graphTargetForSubject(subjectId);if(!target)return null;window.LLMVisDAG?.openView('model-dag',target.view.id);window.LLMVisDAG?.focusNode('model-dag',target.node.id);return {view_id:target.view.id,view_key:target.view.key,node_id:target.node.id,node_key:target.node.key};}
function layerInspection(item){
  const instance=map.instances.find(value=>value.module_path===item.instance_path),semantics=map.semantic_nodes.filter(node=>node.instance_ids.includes(instance?.id));
  const lowerings=loweringsFor(semantics.map(node=>node.id)),graph=graphForLowerings(lowerings),captures=uniqueById([...capturesForLowerings(lowerings),...capturesForLayer(item.layer_index)]);
  const opaque=captures.some(capture=>capture.status!=='captured'),coverageStatus=lowerings.length?'captured':(opaque?'opaque':'config');
  const subjectIds=[instance?.id,...semantics.map(node=>node.id)].filter(Boolean),diagnostics=diagnosticsFor(subjectIds),tensors=uniqueById([...tensorsFor(subjectIds),...graph.tensors]);
  const targetView=graphViewForLayer(item);
  return {targetView,context:{layer:item,graphView:targetView?{id:targetView.id,key:targetView.key,level:targetView.level,label:targetView.label,metadata:targetView.metadata}:null,instance,semantic_nodes:semantics,lowerings,ops:graph.ops,tensors,metrics:metricsFor(subjectIds),diagnostics,captures,coverage_status:coverageStatus,sourceArtifacts:sourceArtifactsFor(captures,tensors)}};
}
function openLayer(item){const selection=layerInspection(item);if(selection.targetView)window.LLMVisDAG?.openView('model-dag',selection.targetView.id);inspect(selection.context);}
function renderLayerSummary(){
  const pattern=data.layerPattern||{summary:`${data.layerStrip.length} layers`,total_layers:data.layerStrip.length,legend:[],legend_context:null,anomaly_count:0};
  const patternSummary=byId('layer-pattern-summary');patternSummary.textContent=pattern.summary;patternSummary.title=pattern.summary;
  byId('layer-count').textContent=`${pattern.total_layers} exact layers`;
  byId('layer-detail-label').textContent=`All ${pattern.total_layers} exact positions · select a layer to inspect it`;
  const note=byId('layer-summary-note'),anomalyCount=Number(pattern.anomaly_count||0);note.dataset.anomalies=String(anomalyCount>0);note.textContent=anomalyCount?`${anomalyCount} pattern deviations · repeat ≠ loop / shared weights`:'one-way DAG · repeat ≠ loop / shared weights';note.setAttribute('aria-label',anomalyCount?`${anomalyCount} pattern deviations; repeat notation is still not a cycle and does not imply weight sharing`:'one-way DAG; repeat notation is not a cycle and does not imply weight sharing');
  const legend=byId('layer-legend');clear(legend);
  for(const item of pattern.legend||[]){const entry=document.createElement('span');entry.className='layer-legend-item';entry.setAttribute('role','listitem');entry.setAttribute('aria-label',`${item.label}: ${item.description}`);const key=document.createElement('span');key.className='layer-key';key.dataset.state=item.state_kind;key.dataset.mlp=item.mlp_kind;key.textContent=item.label;key.setAttribute('aria-hidden','true');const description=document.createElement('span');description.textContent=item.description;entry.append(key,description);legend.append(entry);}
  if(pattern.legend_context){const context=document.createElement('span');context.className='layer-legend-context';context.setAttribute('role','listitem');context.textContent=pattern.legend_context;legend.append(context);}
  const panel=byId('layer-panel');panel.addEventListener('toggle',()=>{if(panel.open)renderStrip();});if(panel.open)renderStrip();
}
function renderStrip(){
  const root=byId('strip');if(root.dataset.rendered==='true')return;clear(root);
  for(const item of data.layerStrip){
    const selection=layerInspection(item),targetView=selection.targetView,coverageStatus=selection.context.coverage_status,el=document.createElement('button');el.type='button';el.className='layer';el.dataset.layerIndex=String(item.layer_index);el.dataset.state=item.state_kind;el.dataset.mlp=item.mlp_kind;el.dataset.coverage=coverageStatus;el.dataset.anomaly=String(Boolean(item.anomaly));if(targetView)el.dataset.graphViewId=targetView.id;el.textContent=item.label;
    el.title=`layer ${item.layer_index} · ${item.attention_kind} · ${item.mlp_kind} · ${item.state_kind} · expected ${item.expected_pattern} · ${coverageStatus}${item.anomaly?` · anomaly: ${item.reason}`:''}`;
    el.setAttribute('aria-label',el.title);
    el.onclick=()=>openLayer(item);root.append(el);
  }
  root.dataset.rendered='true';
}
function scenarioFields(scenario){return [['phase',scenario.phase],['batch',scenario.batch],['new tokens',scenario.new_tokens],['past tokens',scenario.past_tokens],['activation',scenario.activation_dtype],['weights',scenario.weight_format],['KV dtype',scenario.kv_dtype],['backend',scenario.backend],['hardware',scenario.hardware]];}
function renderScenarioCard(scenario){const root=byId('scenario-card');clear(root);if(!scenario){empty(root,'No workload selected.');return;}for(const [label,value] of scenarioFields(scenario)){const box=document.createElement('div');box.className='scenario-field';const key=document.createElement('span');key.textContent=label;const val=document.createElement('strong');val.textContent=fmt(value);box.append(key,val);root.append(box);}}
const costNames=['parameters.resident','parameters.active','weights.storage_bytes','flops','logical_bytes','kv_cache.bytes','state.bytes','arithmetic_intensity'];
function renderCosts(scenario){const root=byId('cost-table');if(!scenario){empty(root,'No Scenario cost analysis.');return;}const metrics=map.metrics.filter(metric=>metric.subject_id===map.model.id&&metric.scenario_id===scenario.id&&costNames.includes(metric.name));if(!metrics.length){empty(root,'No formula costs for this Scenario.');return;}clear(root);root.append(makeTable(['Metric','Value','Unit','Origin','Coverage'],metrics.map(metric=>[metric.name,metric.value,metric.unit,metric.origin,pct(metric.coverage)]),index=>inspect(metricContext(metrics[index],{scenario,aggregate_scope:(manifest.cost.aggregate_scopes||[]).find(item=>item.scenario_id===scenario.id)}))));}
function renderHotspots(scenario){const root=byId('hotspot-table');if(!scenario){empty(root,'No Scenario hotspots.');return;}const summaries=data.hotspots.filter(item=>item.scenario_id===scenario.id),entries=[];for(const summary of summaries){if(!summary.hotspots.length)entries.push({metric_name:summary.metric_name,rank:'–',module_path:'Not ranked',value:summary.aggregate.value,unit:summary.aggregate.unit,origin:summary.aggregate.origin,coverage:summary.aggregate.coverage,aggregate:summary.aggregate,aggregate_scope:summary.aggregate_scope});else for(const item of summary.hotspots)entries.push({...item,metric_name:summary.metric_name,aggregate_scope:summary.aggregate_scope});}if(!entries.length){empty(root,'No hotspot analysis.');return;}clear(root);root.append(makeTable(['Metric','Rank','Module','Value','Unit','Origin','Coverage'],entries.map(item=>[item.metric_name,item.rank,item.module_path,item.value,item.unit,item.origin,pct(item.coverage)]),index=>{const item=entries[index];const instance=map.instances.find(value=>value.id===item.subject_id);const metric=map.metrics.find(value=>value.subject_id===item.subject_id&&value.scenario_id===scenario.id&&value.name===item.metric_name);const graphTarget=locateGraphSubject(item.subject_id);const context={hotspot:item,graph_target:graphTarget,instance,scenario,aggregate_scope:item.aggregate_scope,sourceArtifacts:sourceArtifactsFor(),diagnostics:diagnosticsFor([item.subject_id].filter(Boolean))};if(metric)context.metric=metric;inspect(context);}));}
function renderRoofline(scenario){const root=byId('roofline-table');const result=data.roofline.find(item=>item.scenario_id===scenario?.id);if(!result){empty(root,'No explicit matching HardwareProfile; bounds unavailable.');return;}clear(root);const rows=[['Arithmetic intensity',result.arithmetic_intensity_flops_per_byte,'FLOP/byte'],['Compute lower bound',result.compute_lower_bound_seconds,'seconds'],['Bandwidth lower bound',result.bandwidth_lower_bound_seconds,'seconds'],['Max lower bound',result.max_lower_bound_seconds,'seconds'],['Bottleneck class',result.bottleneck_class,'']];root.append(makeTable(['Metric','Value','Unit'],rows));const button=document.createElement('button');button.className='item';button.textContent='Inspect assumptions and Unknown reasons';button.onclick=()=>inspect({roofline:result,scenario,sourceArtifacts:sourceArtifactsFor(),coverage_status:result.source_scope?.name||'decoder-scope'});root.append(button);}
function renderDiff(scenario){const root=byId('diff-table');if(!data.workloadDiffs.length){empty(root,'At least two Scenarios are required.');return;}const result=data.workloadDiffs.find(item=>item.right_scenario_id===scenario?.id)||data.workloadDiffs[0];const entries=result.entries.filter(item=>item.subject_id===map.model.id);if(!entries.length){empty(root,'No changed model-level metrics.');return;}clear(root);root.append(makeTable(['Metric','Status','Left','Right','Delta'],entries.map(item=>[item.name,item.status,item.left?.value,item.right?.value,item.value_delta]),index=>inspect({workload_diff:entries[index],comparison:{left_scenario_id:result.left_scenario_id,right_scenario_id:result.right_scenario_id},sourceArtifacts:sourceArtifactsFor()})));}
function renderRuntime(scenario){const root=byId('runtime-table');const metrics=map.metrics.filter(metric=>metric.subject_id===map.model.id&&metric.scenario_id===(scenario?.id||null)&&metric.name.startsWith('runtime.'));if(!metrics.length){empty(root,'Runtime is Unknown; no trace artifact was imported.');return;}clear(root);root.append(makeTable(['Metric','Value','Unit','Origin','Coverage'],metrics.map(metric=>[metric.name,metric.value,metric.unit,metric.origin,pct(metric.coverage)]),index=>inspect(metricContext(metrics[index],{scenario}))));}
function renderCoverageBadge(scenario){
  const badge=byId('coverage-badge');
  if(!scenario){const captured=data.captures.filter(item=>item.status==='captured').length;badge.textContent=`${captured} bounded captures`;badge.title='Representative captures only; config structure remains available outside captured regions.';return;}
  const metrics=map.metrics.filter(metric=>metric.subject_id===map.model.id&&metric.scenario_id===scenario.id&&costNames.includes(metric.name));
  if(!metrics.length){badge.textContent='cost coverage · Unknown';badge.title='No formula cost metrics for this scenario.';return;}
  const known=metrics.filter(item=>item.value!==null&&item.value!==undefined&&item.origin!=='unknown').length,average=metrics.reduce((total,item)=>total+Number(item.coverage||0),0)/metrics.length,scope=(manifest.cost.aggregate_scopes||[]).find(item=>item.scenario_id===scenario.id);
  if(scope){const total=scope.included_instance_count+scope.excluded_instance_count;badge.textContent=`structure ${scope.included_instance_count}/${total} · ${pct(scope.structural_coverage)} | cost ${known}/${metrics.length} known`;badge.title=`Structural instance scope; cost metric availability is separate. Mean metric coverage: ${pct(average)}. ${scope.coverage_basis}.`;}
  else{badge.textContent=`cost ${known}/${metrics.length} known`;badge.title=`No aggregate structural scope; mean metric coverage: ${pct(average)}.`;}
}
function renderGraphEvidence(view){
  const badge=byId('graph-evidence-badge'),nodes=view?.nodes||[],total=nodes.length;
  const opaque=nodes.filter(node=>node.opaque).length;
  const evidenced=nodes.filter(node=>!node.opaque&&Number(node.coverage||0)>0).length;
  const partial=nodes.filter(node=>!node.opaque&&Number(node.coverage||0)>0&&Number(node.coverage||0)<1).length;
  badge.textContent=`Current view evidence ${evidenced}/${total}${opaque?` · ${opaque} opaque`:''}${partial?` · ${partial} partial`:''}`;
  badge.title='Evidence coverage for nodes in the currently visible DAG view; opaque and partial regions are never filled by inference.';
}
function renderScenario(){const id=byId('scenario-select').value,scenario=map.scenarios.find(item=>item.id===id);renderScenarioCard(scenario);renderCosts(scenario);renderHotspots(scenario);renderRoofline(scenario);renderDiff(scenario);renderRuntime(scenario);renderCoverageBadge(scenario);renderHeatmap(scenario);if(lastGraphDetail)setInspector(graphSelectionContext(lastGraphDetail),{reveal:false});}
function setupScenarios(){const select=byId('scenario-select');clear(select);if(!map.scenarios.length){const option=document.createElement('option');option.textContent='No workload';option.value='';select.append(option);select.disabled=true;}else for(const scenario of map.scenarios){const option=document.createElement('option');option.value=scenario.id;option.textContent=`${scenario.phase} · B${scenario.batch} T${scenario.new_tokens} L${scenario.past_tokens} · ${scenario.weight_format}`;select.append(option);}select.onchange=renderScenario;renderScenario();}
byId('heatmap-mode').onchange=()=>{renderHeatmap();if(lastGraphDetail)setInspector(graphSelectionContext(lastGraphDetail),{reveal:false});};
byId('model-dag').addEventListener('llm-vis:dag-view-change',event=>{
  const view=event.detail.view,saved=graphDetailByView.get(view.id)||null;
  lastGraphDetail=saved;
  setInspector(saved?graphSelectionContext(saved):emptyInspectorSelection,{reveal:false});
  renderGraphEvidence(view);
  renderHeatmap(currentScenario(),view);
});
byId('search').addEventListener('input',event=>{const query=event.target.value;renderDefinitions(query);renderSemantics(query);renderLogical(query);window.LLMVisDAG?.search('model-dag',query);});
document.querySelectorAll('.inspector-tab').forEach(button=>{button.onclick=()=>{inspectorTab=button.dataset.inspectorTab;renderInspector();};button.onkeydown=event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const tabs=[...document.querySelectorAll('.inspector-tab')],index=tabs.indexOf(button),next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[next].click();tabs[next].focus();};});
document.querySelectorAll('.capture-card').forEach((element,index)=>{element.onclick=()=>inspect(captureContext(data.captures[index]));});
renderDefinitions();renderSemantics();renderLogical();renderDiagnostics();renderLayerSummary();renderGraphEvidence(graphView.views[0]);setupScenarios();renderInspector();
</script>
</body>
</html>
"""
    return (
        template.replace("__DISPLAY_TITLE__", display_title)
        .replace("__SOURCE_TITLE__", source_title)
        .replace("__REVISION_TITLE__", revision_title)
        .replace("__REVISION_SHORT__", revision_short)
        .replace("__CONFIG_HASH_TITLE__", config_hash_title)
        .replace("__CONFIG_HASH_SHORT__", config_hash_short)
        .replace("__DAG_ASSETS__", render_dag_canvas_assets())
        .replace("__DAG_CANVAS__", dag_canvas)
        .replace("__CAPTURES__", _capture_summary(bundle))
        .replace("__EMBEDDED__", embedded)
    )

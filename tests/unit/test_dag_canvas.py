from __future__ import annotations

from llm_vis.report.dag_canvas import (
    DAG_CANVAS_CSS,
    DAG_CANVAS_JS,
    render_dag_canvas,
    render_dag_canvas_assets,
)


def _graph_document() -> dict[str, object]:
    return {
        "views": [
            {
                "id": "l0",
                "label": "Model L0",
                "nodes": [
                    {
                        "id": "tokens",
                        "label": "Tokens",
                        "kind": "input",
                        "origin": "config",
                        "coverage": "config",
                        "ports": [{"id": "hidden", "direction": "output"}],
                    },
                    {
                        "id": "block",
                        "label": "Decoder block",
                        "kind": "decoder",
                        "origin": "config",
                        "coverage": "captured",
                        "drilldown_view_id": "l1-block",
                        "ports": [
                            {"id": "in", "direction": "input"},
                            {"id": "out", "direction": "output"},
                        ],
                        "core_info": {"shape": "[B,T,H]", "dtype": "bf16"},
                    },
                ],
                "edges": [
                    {
                        "id": "hidden-edge",
                        "source_node_id": "tokens",
                        "source_port_id": "hidden",
                        "target_node_id": "block",
                        "target_port_id": "in",
                        "kind": "data",
                        "label": "hidden [B,T,H] bf16",
                    }
                ],
            },
            {
                "id": "l1-block",
                "label": "Block L1",
                "parent_view_id": "l0",
                "nodes": [],
                "edges": [],
            },
        ]
    }


def test_render_dag_canvas_embeds_payload_and_read_only_controls() -> None:
    fragment = render_dag_canvas(_graph_document(), element_id="qwen-dag", initial_view_id="l0")

    assert 'id="qwen-dag"' in fragment
    assert 'data-initial-view-id="l0"' in fragment
    assert 'data-readonly="true"' in fragment
    assert 'data-dag-action="back"' in fragment
    assert 'data-dag-action="fit"' in fragment
    assert 'data-dag-action="zoom-in"' in fragment
    assert 'class="llm-dag-nav-controls"' in fragment
    assert 'aria-label="Graph navigation controls"' in fragment
    assert 'class="llm-dag-view-controls"' in fragment
    assert 'aria-label="Canvas view controls"' in fragment
    assert 'class="llm-dag-view-select"' in fragment
    assert 'class="llm-dag-search"' in fragment
    assert 'class="llm-dag-breadcrumb"' in fragment
    assert 'class="llm-dag-minimap"' in fragment
    assert 'class="llm-dag-heat-legend"' in fragment
    assert "gray = Unknown / not attributable" in fragment
    assert 'role="group" aria-label="Interactive model dataflow graph"' in fragment
    assert '"drilldown_view_id":"l1-block"' in fragment
    assert '"source_port_id":"hidden"' in fragment


def test_assets_expose_offline_interaction_api_and_distinct_edge_patterns() -> None:
    assets = render_dag_canvas_assets()

    assert "window.LLMVisDAG" in assets
    assert "llmVisInspect" in assets
    assert "llm-vis:dag-select" in assets
    assert "llm-vis:dag-view-change" in assets
    assert "openView" in assets
    assert "search" in assets
    assert "searchViews" in assets
    assert "focusNode" in assets
    assert "setHeatmap" in assets
    assert "centerNode" in assets
    assert "frameReadable" in assets
    assert "llm-dag-edge-label-name" in assets
    assert "llm-dag-edge-label-shape" in assets
    assert "llm-dag-edge-label-dtype" in assets
    assert "stableLayout" in assets
    assert "JSON.stringify(value)" in assets
    assert "linear_attention_layers" in assets
    assert "routed_experts" in assets
    assert "normalizeView" in assets
    assert "ownerByPort" in assets
    assert "shape Unknown" in assets
    assert '[data-edge-kind="state"]' in DAG_CANVAS_CSS
    assert '[data-edge-kind="route"]' in DAG_CANVAS_CSS
    assert '[data-edge-kind="control"]' in DAG_CANVAS_CSS
    assert "stroke-dasharray: 9 5" in DAG_CANVAS_CSS
    assert "stroke-dasharray: 2 5" in DAG_CANVAS_CSS
    assert "11 4 2 4" in DAG_CANVAS_CSS
    assert "dblclick" in DAG_CANVAS_JS
    assert "has-drilldown" in DAG_CANVAS_JS
    assert "llm-dag-drill-badge" in DAG_CANVAS_JS
    assert "L1 ›" in DAG_CANVAS_JS
    assert "viewsById.has(drilldownId)" in DAG_CANVAS_JS
    assert "'aria-keyshortcuts': 'Enter'" in DAG_CANVAS_JS
    assert "source: 'node-affordance'" in DAG_CANVAS_JS
    assert "event.key === 'Enter' && canDrill" in DAG_CANVAS_JS
    assert "event.key === 'Enter' || event.key === ' '" in DAG_CANVAS_JS
    assert "heatKnown" in DAG_CANVAS_JS
    assert 'data-heat-known="false"' in DAG_CANVAS_CSS
    assert "llm-dag-heat-legend" in DAG_CANVAS_CSS
    assert "stroke-dasharray: none" in DAG_CANVAS_CSS
    assert "parent_view_id" in DAG_CANVAS_JS
    assets_without_svg_namespace = assets.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in assets_without_svg_namespace
    assert "https://" not in assets_without_svg_namespace
    assert "<script src=" not in assets
    assert "@import" not in assets


def test_payload_is_deterministic_and_script_closing_sequence_is_safe() -> None:
    first = render_dag_canvas({"views": [], "label": "</script><b>unsafe</b>"})
    second = render_dag_canvas({"label": "</script><b>unsafe</b>", "views": []})

    assert first == second
    assert "</script><b>unsafe" not in first
    assert "<\\/script><b>unsafe" in first
    assert "http://" not in first
    assert "https://" not in first

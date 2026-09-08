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
                "parent_node_id": "block",
                "decomposes_node_id": "block",
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
    assert 'data-layout-direction="DOWN"' in fragment
    assert 'data-dag-action="back"' in fragment
    assert "← Collapse" in fragment
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
    assert 'aria-label="Current view minimap"' in fragment
    assert 'class="llm-dag-current-title">Current view</div>' in fragment
    assert 'class="llm-dag-parent-context"' in fragment
    assert 'aria-label="Parent context navigator"' in fragment
    assert 'data-dag-action="parent-context-back"' in fragment
    assert 'data-dag-action="toggle-parent-context"' in fragment
    assert 'class="llm-dag-parent-map" role="button" tabindex="0"' in fragment
    assert 'aria-keyshortcuts="Enter Space"' in fragment
    assert 'class="llm-dag-heat-legend"' in fragment
    assert '<span class="llm-dag-readonly">Flow ↓</span>' in fragment
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
    assert "nodes ↓" in DAG_CANVAS_JS
    assert "childNodeCount" in DAG_CANVAS_JS
    assert "viewsById.has(drilldownId)" in DAG_CANVAS_JS
    assert "'aria-keyshortcuts': 'Enter'" in DAG_CANVAS_JS
    assert "source: 'node-affordance'" in DAG_CANVAS_JS
    assert "event.key === 'Enter' && canDrill" in DAG_CANVAS_JS
    assert "source: 'single-click', reveal: true" in DAG_CANVAS_JS
    assert "cancelPendingNodeClick()" in DAG_CANVAS_JS
    assert "pending.viewId !== viewId(currentView)" in DAG_CANVAS_JS
    assert "!group.isConnected" in DAG_CANVAS_JS
    assert "}, 320);" in DAG_CANVAS_JS
    assert "double-click-leaf" in DAG_CANVAS_JS
    assert "event.key === 'Enter' || event.key === ' '" in DAG_CANVAS_JS
    assert "selectionByView" in DAG_CANVAS_JS
    assert "viewportByView" in DAG_CANVAS_JS
    assert "syncViewSelector" in DAG_CANVAS_JS
    assert "data-current-only" in DAG_CANVAS_JS
    assert "option[data-current-only]" in DAG_CANVAS_JS
    assert "option.textContent = `↳ " in DAG_CANVAS_JS
    assert "const restoredSelection = selected ? { ...selected } : null" in DAG_CANVAS_JS
    assert "const restoredViewId = viewId(next)" in DAG_CANVAS_JS
    assert "viewId(currentView) === restoredViewId" in DAG_CANVAS_JS
    assert "type: 'port'" in DAG_CANVAS_JS
    assert "root.querySelectorAll('.llm-dag-port-group')" in DAG_CANVAS_JS
    assert (
        "root.querySelectorAll('.llm-dag-node, .llm-dag-port-group, "
        ".llm-dag-edge-group')" in DAG_CANVAS_JS
    )
    assert "selector_visible" in DAG_CANVAS_JS
    assert "heatKnown" in DAG_CANVAS_JS
    assert 'data-heat-known="false"' in DAG_CANVAS_CSS
    assert "llm-dag-heat-legend" in DAG_CANVAS_CSS
    assert "stroke-dasharray: none" in DAG_CANVAS_CSS
    assert "parent_view_id" in DAG_CANVAS_JS
    assert "immediateParentContext" in DAG_CANVAS_JS
    assert "viewsById.get(String(view?.parent_view_id || ''))" in DAG_CANVAS_JS
    assert "view?.parent_node_id" in DAG_CANVAS_JS
    assert "view?.decomposes_node_id" in DAG_CANVAS_JS
    assert "String(node.drilldown_view_id || '') === viewId(view)" in DAG_CANVAS_JS
    assert "return expandedNode ? { parentView, expandedNodeId, expandedNode } : null" in (
        DAG_CANVAS_JS
    )
    assert "drawParentContext" in DAG_CANVAS_JS
    assert "stableLayout(parentView)" in DAG_CANVAS_JS
    assert "direction: 'DOWN'" in DAG_CANVAS_JS
    assert "root.dataset.layoutDirection = 'DOWN'" in DAG_CANVAS_JS
    assert "direction === 'input' ? pos.y : pos.y + pos.height" in DAG_CANVAS_JS
    assert "V ${middle} H ${target.x} V ${target.y}" in DAG_CANVAS_JS
    assert "rankDelta === 1" in DAG_CANVAS_JS
    assert "activeLayout.contentLeft" in DAG_CANVAS_JS
    assert "edgeRouteLanes" in DAG_CANVAS_JS
    assert "availableLane" in DAG_CANVAS_JS
    assert "const side = routeHint.side" in DAG_CANVAS_JS
    assert "data-route-lane" in DAG_CANVAS_JS
    assert "labelY: labelFromTarget ? targetEntry : sourceExit" in DAG_CANVAS_JS
    assert DAG_CANVAS_JS.count("d: route.d") >= 2
    assert "source.x + source.width / 2" in DAG_CANVAS_JS
    assert "target.x + target.width / 2" in DAG_CANVAS_JS
    assert "is-expanded-parent" in DAG_CANVAS_JS
    assert "parentContextCollapsed" in DAG_CANVAS_JS
    assert "returnToParent('parent-context')" in DAG_CANVAS_JS
    assert "returnToParent('parent-context-map')" in DAG_CANVAS_JS
    assert "returnToParent('breadcrumb')" in DAG_CANVAS_JS
    assert "focusNodeId: context?.expandedNodeId || null" in DAG_CANVAS_JS
    assert "focusRenderedNode(requestedFocusNodeId)" in DAG_CANVAS_JS
    assert "window.matchMedia('(max-width: 720px)')" in DAG_CANVAS_JS
    assert "event.key !== 'Escape' || parentContextCollapsed" in DAG_CANVAS_JS
    assert "compactParentContext.addEventListener('change'" in DAG_CANVAS_JS
    assert "root.classList.add('has-parent-context')" in DAG_CANVAS_JS
    assert ".llm-dag-current-context" in DAG_CANVAS_CSS
    assert "'.llm-dag-parent-context, .llm-dag-current-context, '" in DAG_CANVAS_JS
    assert ".llm-dag-parent-context" in DAG_CANVAS_CSS
    assert ".llm-dag-parent-node.is-expanded-parent" in DAG_CANVAS_CSS
    assets_without_svg_namespace = assets.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in assets_without_svg_namespace
    assert "https://" not in assets_without_svg_namespace
    assert "<script src=" not in assets
    assert "@import" not in assets


def test_assets_expose_real_dark_light_palettes_and_repaint_without_navigation() -> None:
    assert '.llm-dag[data-theme="dark"]' in DAG_CANVAS_CSS
    assert '[data-theme="light"] .llm-dag:not([data-theme="dark"])' in DAG_CANVAS_CSS
    assert '.llm-dag[data-theme="light"]' in DAG_CANVAS_CSS
    assert ":root:not([data-theme]) .llm-dag:not([data-theme])" in DAG_CANVAS_CSS
    assert "@media (prefers-color-scheme: light)" in DAG_CANVAS_CSS
    for token in (
        "--dag-bg",
        "--dag-panel",
        "--dag-text",
        "--dag-focus",
        "--dag-node",
        "--dag-node-header",
        "--dag-node-title",
        "--dag-port-input",
        "--dag-port-output",
        "--dag-edge-data",
        "--dag-edge-state",
        "--dag-edge-route",
        "--dag-edge-control",
        "--dag-heat-low-rgb",
        "--dag-heat-mid-rgb",
        "--dag-heat-high-rgb",
        "--dag-heat-unknown",
        "--dag-heat-not-applicable",
        "--dag-minimap-node",
        "--dag-minimap-node-outline",
        "--dag-minimap-edge",
        "--dag-minimap-viewport-fill",
        "--dag-minimap-viewport-stroke",
        "--dag-minimap-heat-unknown",
        "--dag-parent-map-bg",
        "--dag-parent-expanded",
    ):
        assert DAG_CANVAS_CSS.count(token) >= 2, token

    assert ".llm-dag-arrow { fill: var(--dag-edge-data); }" in DAG_CANVAS_CSS
    for kind in ("state", "route", "control"):
        assert f'.llm-dag-arrow[data-edge-kind="{kind}"]' in DAG_CANVAS_CSS
    assert "`--dag-edge-${kind}`" in DAG_CANVAS_JS

    assert "window.addEventListener('llm-vis:theme-change', handleThemeChange)" in (
        DAG_CANVAS_JS
    )
    assert "new MutationObserver(handleThemeChange)" in DAG_CANVAS_JS
    assert "attributeFilter: ['data-theme']" in DAG_CANVAS_JS
    assert "function syncThemePaint()" in DAG_CANVAS_JS
    assert "function refreshHeatPalette()" in DAG_CANVAS_JS
    assert "const stops = heatPalette" in DAG_CANVAS_JS
    assert "syncThemePaint();" in DAG_CANVAS_JS
    assert "window.removeEventListener('llm-vis:theme-change', handleThemeChange)" in (
        DAG_CANVAS_JS
    )
    assert "stroke: var(--dag-minimap-node-outline);" in DAG_CANVAS_CSS
    assert "vector-effect: non-scaling-stroke;" in DAG_CANVAS_CSS


def test_media_query_adapter_handles_legacy_and_throwing_factories() -> None:
    assert "const safeMatchMedia = (query, nativeFactory) =>" in DAG_CANVAS_JS
    assert "nativeMedia = nativeFactory()" in DAG_CANVAS_JS
    assert "catch (_error) {\n      return mediaFallback(query);\n    }" in DAG_CANVAS_JS
    assert "typeof nativeMedia.addListener === 'function'" in DAG_CANVAS_JS
    assert "nativeMedia.addListener(listener)" in DAG_CANVAS_JS
    assert "typeof nativeMedia.removeListener === 'function'" in DAG_CANVAS_JS
    assert "nativeMedia.removeListener(listener)" in DAG_CANVAS_JS
    assert "get matches()" in DAG_CANVAS_JS
    assert "catch (_error) { return false; }" in DAG_CANVAS_JS


def test_payload_is_deterministic_and_script_closing_sequence_is_safe() -> None:
    first = render_dag_canvas({"views": [], "label": "</script><b>unsafe</b>"})
    second = render_dag_canvas({"label": "</script><b>unsafe</b>", "views": []})

    assert first == second
    assert "</script><b>unsafe" not in first
    assert "<\\/script><b>unsafe" in first
    assert "http://" not in first
    assert "https://" not in first

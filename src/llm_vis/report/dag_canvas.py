"""Offline, dependency-free SVG DAG canvas used by the M3.5 report UI.

The renderer deliberately emits a read-only shell plus an embedded GraphView document.  Add
``render_dag_canvas_assets()`` once to the page and ``render_dag_canvas(document)`` wherever
the canvas belongs.  The JavaScript auto-mounts every rendered shell.

The GraphView document is expected to contain ``views``.  Each view contains ``nodes`` and
``edges``; nodes may contain ``ports`` and an optional ``drilldown_view_id``.  Views may link
back with ``parent_view_id``.  Selecting an item calls ``window.llmVisInspect(detail)`` when
that hook exists and always dispatches ``llm-vis:dag-select``.  View changes dispatch
``llm-vis:dag-view-change``.  Both events bubble from the canvas root.

``window.LLMVisDAG`` exposes ``get``, ``fit``, ``search``, ``openView`` and ``setHeatmap``
for report-level controls.  No graph editing or model execution capability is included.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any, Optional

DAG_CANVAS_CSS = r"""
.llm-dag {
  --dag-bg: #08101d;
  --dag-panel: #101b2d;
  --dag-node: #172741;
  --dag-line: #31415d;
  --dag-text: #e8effa;
  --dag-muted: #91a7c7;
  --dag-accent: #70adff;
  --dag-upstream: #f1bd64;
  --dag-downstream: #6be0b3;
  min-width: 0;
  color: var(--dag-text);
  background: var(--dag-bg);
  border: 1px solid var(--dag-line);
  border-radius: 12px;
  overflow: hidden;
}
.llm-dag * { box-sizing: border-box; }
.llm-dag-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border-bottom: 1px solid var(--dag-line);
  background: #0e192a;
}
.llm-dag-nav-controls,
.llm-dag-view-controls {
  display: flex;
  align-items: center;
  gap: 6px;
}
.llm-dag-nav-controls { flex: 0 1 auto; min-width: 0; }
.llm-dag-view-controls { flex: 0 0 auto; margin-left: auto; }
.llm-dag-toolbar button,
.llm-dag-toolbar select,
.llm-dag-toolbar input {
  color: var(--dag-text);
  background: #111f35;
  border: 1px solid #354967;
  border-radius: 6px;
  min-height: 30px;
  padding: 5px 8px;
  font: 12px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.llm-dag-toolbar button { cursor: pointer; }
.llm-dag-toolbar button:hover:not(:disabled) { border-color: var(--dag-accent); }
.llm-dag-toolbar button:disabled { opacity: .42; cursor: default; }
.llm-dag-toolbar label {
  display: flex;
  align-items: center;
  gap: 5px;
  color: var(--dag-muted);
  font-size: 11px;
}
.llm-dag-search { width: min(220px, 30vw); }
.llm-dag-breadcrumb {
  display: flex;
  align-items: center;
  gap: 3px;
  min-width: 100px;
  flex: 1 1 180px;
  overflow: hidden;
}
.llm-dag-crumb {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 180px;
}
.llm-dag-separator { color: var(--dag-muted); }
.llm-dag-readonly {
  color: var(--dag-muted);
  border: 1px solid #354967;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 10px;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.llm-dag-stage {
  position: relative;
  height: clamp(440px, 64vh, 820px);
  min-height: 360px;
  overflow: hidden;
  background-color: var(--dag-bg);
  background-image:
    linear-gradient(rgba(94, 124, 164, .07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(94, 124, 164, .07) 1px, transparent 1px);
  background-size: 24px 24px;
  touch-action: pan-y;
  cursor: grab;
}
.llm-dag-stage.is-panning { cursor: grabbing; }
.llm-dag-surface { width: 100%; height: 100%; display: block; user-select: none; }
.llm-dag-world { transform-origin: 0 0; }
.llm-dag-edge {
  fill: none;
  stroke: #91bfff;
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
  pointer-events: none;
}
.llm-dag-edge[data-edge-kind="state"] { stroke: #e3a7ff; stroke-dasharray: 9 5; }
.llm-dag-edge[data-edge-kind="route"] { stroke: #ffd172; stroke-dasharray: 2 5; }
.llm-dag-edge[data-edge-kind="control"] {
  stroke: #a8b5c8;
  stroke-width: 1.5;
  stroke-dasharray: 11 4 2 4;
}
.llm-dag-edge-hit {
  fill: none;
  stroke: transparent;
  stroke-width: 14;
  vector-effect: non-scaling-stroke;
  cursor: pointer;
  pointer-events: stroke;
}
.llm-dag-edge-label {
  fill: #c8d6ec;
  stroke: var(--dag-bg);
  stroke-width: 4px;
  paint-order: stroke;
  font: 10px/1 ui-monospace, monospace;
  pointer-events: none;
}
.llm-dag-node { cursor: pointer; }
.llm-dag-node-card {
  fill: var(--dag-node-heat, var(--dag-node));
  stroke: #3e5577;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.llm-dag-node.has-drilldown { cursor: pointer; }
.llm-dag-node-header { fill: #20385b; }
.llm-dag-node-title { fill: #fff; font: 600 13px/1 ui-sans-serif, system-ui; }
.llm-dag-node-meta { fill: #a9bdd9; font: 10px/1 ui-monospace, monospace; }
.llm-dag-node-core { fill: #ced9e9; font: 10px/1 ui-monospace, monospace; }
.llm-dag-drill-badge { cursor: pointer; }
.llm-dag-drill-hit { fill: transparent; }
.llm-dag-drill-pill { fill: #0e1b30; stroke: #6ea8fe; stroke-width: 1; }
.llm-dag-drill-badge:hover .llm-dag-drill-pill,
.llm-dag-drill-badge:focus .llm-dag-drill-pill { fill: #1d3c64; stroke: #b9d7ff; }
.llm-dag-drill-label {
  fill: #b9d7ff;
  font: 700 8px/1 ui-sans-serif, system-ui;
  letter-spacing: .04em;
  pointer-events: none;
}
.llm-dag-port-label { fill: #b7c7de; font: 9px/1 ui-monospace, monospace; }
.llm-dag-port { stroke: #07101e; stroke-width: 2; pointer-events:none; }
.llm-dag-port-hit { fill: transparent; pointer-events: all; cursor: pointer; }
.llm-dag-port[data-port-direction="input"] { fill: #8ab7f2; }
.llm-dag-port[data-port-direction="output"] { fill: #69ddb1; }
.llm-dag-node:focus-visible .llm-dag-node-card,
.llm-dag-port-group:focus-visible .llm-dag-port,
.llm-dag-edge-group:focus-visible .llm-dag-edge {
  stroke: #fff;
  stroke-width: 4;
}
.llm-dag-node.is-selected .llm-dag-node-card { stroke: #fff; stroke-width: 3; }
.llm-dag-node.is-upstream .llm-dag-node-card { stroke: var(--dag-upstream); }
.llm-dag-node.is-downstream .llm-dag-node-card { stroke: var(--dag-downstream); }
.llm-dag-node.is-match .llm-dag-node-card { filter: brightness(1.35); stroke: #fff0a6; }
.llm-dag-node.is-dimmed,
.llm-dag-edge-group.is-dimmed { opacity: .18; }
.llm-dag-edge-group.is-selected .llm-dag-edge { stroke-width: 4; }
.llm-dag[data-heatmap]:not([data-heatmap="off"])
  .llm-dag-node[data-heat-known="false"] .llm-dag-node-card {
  fill: #1c2533;
  stroke-dasharray: 6 4;
}
.llm-dag[data-heatmap]:not([data-heatmap="off"])
  .llm-dag-node[data-heat-status="not-applicable"] .llm-dag-node-card {
  fill: #111a27;
  stroke-dasharray: 2 5;
}
.llm-dag[data-heatmap]:not([data-heatmap="off"])
  .llm-dag-node[data-heat-status="partial"] .llm-dag-node-card {
  stroke-dasharray: 8 3;
}
.llm-dag[data-heatmap]:not([data-heatmap="off"])
  .llm-dag-node.is-selected .llm-dag-node-card,
.llm-dag[data-heatmap]:not([data-heatmap="off"])
  .llm-dag-node.is-upstream .llm-dag-node-card,
.llm-dag[data-heatmap]:not([data-heatmap="off"])
  .llm-dag-node.is-downstream .llm-dag-node-card { stroke-dasharray: none; }
.llm-dag[data-heatmap]:not([data-heatmap="off"])
  .llm-dag-minimap-node[data-heat-known="false"] { fill: #343d4d; }
.llm-dag-heat-legend {
  display: flex;
  align-items: center;
  gap: 9px;
  min-height: 36px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--dag-line);
  background: #0b1525;
  color: var(--dag-muted);
  font: 10px/1.25 ui-sans-serif, system-ui, sans-serif;
}
.llm-dag-heat-legend[hidden] { display: none; }
.llm-dag-heat-title { color: var(--dag-text); font-weight: 700; white-space: nowrap; }
.llm-dag-heat-scale {
  width: 112px;
  height: 8px;
  border: 1px solid #41516a;
  border-radius: 999px;
  background: linear-gradient(90deg,#173a58,#806722,#87353c);
}
.llm-dag-heat-basis {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.llm-dag-heat-coverage { margin-left: auto; white-space: nowrap; }
.llm-dag-heat-unknown { white-space: nowrap; }
.llm-dag-minimap {
  position: absolute;
  right: 12px;
  bottom: 12px;
  width: 190px;
  height: 112px;
  border: 1px solid #49617f;
  border-radius: 7px;
  background: rgba(7, 14, 25, .92);
  box-shadow: 0 5px 18px rgba(0, 0, 0, .35);
}
.llm-dag-minimap-node { fill: #4f719e; }
.llm-dag-minimap-edge { fill: none; stroke: #536887; stroke-width: 1; }
.llm-dag-minimap-viewport { fill: rgba(112, 173, 255, .09); stroke: #84b8ff; }
.llm-dag-status {
  position: absolute;
  left: 11px;
  bottom: 9px;
  color: var(--dag-muted);
  background: rgba(7, 14, 25, .84);
  border-radius: 5px;
  padding: 4px 7px;
  font: 10px/1.2 ui-monospace, monospace;
  pointer-events: none;
}
.llm-dag-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--dag-muted);
  font-size: 13px;
}
.llm-dag-empty[hidden] { display: none; }
@media (max-width: 720px) {
  .llm-dag-toolbar { align-items: stretch; flex-wrap: wrap; }
  .llm-dag-nav-controls { flex: 1 1 100%; }
  .llm-dag-nav-controls label { flex: 1 1 auto; }
  .llm-dag-view-select { width: 100%; min-width: 0; }
  .llm-dag-breadcrumb { flex-basis: 100%; order: 3; }
  .llm-dag-view-controls { flex: 1 1 100%; margin-left: 0; }
  .llm-dag-search { flex: 1 1 auto; width: auto; }
  .llm-dag-stage { height: 66vh; }
  .llm-dag-minimap { width: 140px; height: 88px; }
}
@media (max-width: 480px) {
  .llm-dag-minimap { display: none; }
}
"""


DAG_CANVAS_JS = r"""
(() => {
  'use strict';
  if (window.LLMVisDAG) return;

  const NS = 'http://www.w3.org/2000/svg';
  const WIDTH = 224;
  const BASE_HEIGHT = 142;
  const X_GAP = 116;
  const Y_GAP = 42;
  const PAD = 58;
  const instances = new Map();
  const svg = (tag, attrs = {}) => {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    return el;
  };
  const text = (tag, value, attrs = {}) => {
    const el = svg(tag, attrs);
    el.textContent = value == null ? '' : String(value);
    return el;
  };
  const clip = (value, size = 29) => {
    const raw = String(value == null ? '' : value);
    return raw.length > size ? `${raw.slice(0, size - 1)}…` : raw;
  };
  const nodeId = node => String(node.id || node.node_id || node.label || 'node');
  const portId = port => String(port.id || port.port_id || port.name || port.label || 'port');
  const edgeId = (edge, index) => String(edge.id || edge.edge_id || `edge-${index}`);
  const edgeKind = edge => {
    const kind = String(edge.kind || 'data').toLowerCase();
    if (kind.includes('state') || kind.includes('cache')) return 'state';
    if (kind.includes('route') || kind.includes('expert')) return 'route';
    if (kind.includes('control')) return 'control';
    return 'data';
  };
  const viewId = view => String(view.id || view.view_id || view.label || 'view');

  function displayShape(port) {
    if (port.shape_known === false || !Array.isArray(port.shape)) return 'shape Unknown';
    return `[${port.shape.join(',')}]`;
  }

  function normalizeView(rawView) {
    // GraphView keeps ports at view scope and edges connect port-to-port.  The SVG layout
    // also needs the owning nodes, so derive that renderer-only information without
    // changing the serialized renderer-neutral artifact.
    const ports = [...(rawView.ports || [])];
    const portById = new Map(ports.map(port => [portId(port), port]));
    const ownerByPort = new Map();
    const nodes = [...(rawView.nodes || [])].map(rawNode => {
      const declared = [
        ...(rawNode.input_port_ids || []),
        ...(rawNode.output_port_ids || [])
      ].map(String);
      const nodePorts = rawNode.ports || declared.map(id => portById.get(id)).filter(Boolean);
      for (const port of nodePorts) ownerByPort.set(portId(port), nodeId(rawNode));
      const numericCoverage = Number(rawNode.coverage);
      const coverage = rawNode.coverage_status || (rawNode.opaque ? 'opaque'
        : Number.isFinite(numericCoverage) ? `${Math.round(numericCoverage * 100)}% config`
          : 'config');
      return {
        ...rawNode,
        ports: nodePorts,
        origin: rawNode.origin || 'config',
        coverage_status: coverage,
        core_info: rawNode.core_info || rawNode.attributes || {}
      };
    });
    const edges = [...(rawView.edges || [])].map(rawEdge => {
      const sourcePort = portById.get(String(rawEdge.source_port_id));
      const targetPort = portById.get(String(rawEdge.target_port_id));
      const contractPort = sourcePort || targetPort || {};
      const tensorName = rawEdge.tensor_name || contractPort.name || contractPort.key
        || rawEdge.tensor_spec_id || 'tensor';
      const shapeLabel = rawEdge.shape_label || displayShape(contractPort).replace(/^shape /, '');
      const dtypeLabel = rawEdge.dtype_label || contractPort.dtype || 'dtype Unknown';
      const label = rawEdge.label
        || `${tensorName} ${shapeLabel} ${dtypeLabel}`;
      return {
        ...rawEdge,
        source_node_id: rawEdge.source_node_id
          || ownerByPort.get(String(rawEdge.source_port_id)),
        target_node_id: rawEdge.target_node_id
          || ownerByPort.get(String(rawEdge.target_port_id)),
        tensor_name: tensorName,
        shape_label: shapeLabel,
        dtype_label: dtypeLabel,
        label
      };
    });
    return { ...rawView, nodes, edges };
  }

  function stableLayout(view) {
    const nodes = [...(view.nodes || [])].sort((a, b) => nodeId(a).localeCompare(nodeId(b)));
    const ids = new Set(nodes.map(nodeId));
    const edges = [...(view.edges || [])]
      .filter(edge => ids.has(String(edge.source_node_id)) && ids.has(String(edge.target_node_id)))
      .sort((a, b) => {
        const ak = `${a.source_node_id}|${a.target_node_id}|${a.id || ''}`;
        const bk = `${b.source_node_id}|${b.target_node_id}|${b.id || ''}`;
        return ak.localeCompare(bk);
      });
    const incoming = new Map(nodes.map(node => [nodeId(node), 0]));
    const outgoing = new Map(nodes.map(node => [nodeId(node), []]));
    const ranks = new Map(nodes.map(node => [nodeId(node), 0]));
    for (const edge of edges) {
      const source = String(edge.source_node_id);
      const target = String(edge.target_node_id);
      if (source === target) continue;
      outgoing.get(source).push(target);
      incoming.set(target, incoming.get(target) + 1);
    }
    for (const targets of outgoing.values()) targets.sort();
    const queue = nodes.map(nodeId).filter(id => incoming.get(id) === 0).sort();
    const visited = new Set();
    while (queue.length) {
      const id = queue.shift();
      if (visited.has(id)) continue;
      visited.add(id);
      for (const target of outgoing.get(id)) {
        ranks.set(target, Math.max(ranks.get(target), ranks.get(id) + 1));
        incoming.set(target, incoming.get(target) - 1);
        if (incoming.get(target) === 0) {
          queue.push(target);
          queue.sort();
        }
      }
    }
    const remaining = nodes.map(nodeId).filter(id => !visited.has(id)).sort();
    let cycleRank = Math.max(0, ...ranks.values());
    for (const id of remaining) ranks.set(id, cycleRank++);
    const columns = new Map();
    for (const node of nodes) {
      const rank = ranks.get(nodeId(node));
      if (!columns.has(rank)) columns.set(rank, []);
      columns.get(rank).push(node);
    }
    const positions = new Map();
    let maxX = PAD + WIDTH;
    let maxY = PAD + BASE_HEIGHT;
    for (const [rank, column] of [...columns.entries()].sort((a, b) => a[0] - b[0])) {
      column.sort((a, b) => {
        const ak = `${a.label || ''}|${nodeId(a)}`;
        const bk = `${b.label || ''}|${nodeId(b)}`;
        return ak.localeCompare(bk);
      });
      let columnY = PAD;
      column.forEach(node => {
        const ports = [...(node.ports || [])];
        const inputs = ports.filter(port => portDirection(port, node, edges) === 'input');
        const outputs = ports.filter(port => portDirection(port, node, edges) === 'output');
        const portCount = Math.max(1, inputs.length, outputs.length);
        const height = Math.max(BASE_HEIGHT, 112 + portCount * 17);
        const x = PAD + rank * (WIDTH + X_GAP);
        const y = columnY;
        positions.set(nodeId(node), { x, y, width: WIDTH, height, node });
        maxX = Math.max(maxX, x + WIDTH + PAD);
        maxY = Math.max(maxY, y + height + PAD);
        columnY += height + Y_GAP;
      });
    }
    return { nodes, edges, positions, width: maxX, height: maxY };
  }

  function portDirection(port, node, edges) {
    const declared = String(port.direction || port.io || '').toLowerCase();
    if (declared === 'input' || declared === 'in') return 'input';
    if (declared === 'output' || declared === 'out') return 'output';
    const id = portId(port);
    const owner = nodeId(node);
    if (edges.some(edge => String(edge.target_node_id) === owner
      && String(edge.target_port_id || '') === id)) return 'input';
    return 'output';
  }

  function coreLines(node) {
    const source = node.core_info || node.core || node.metrics || node.attrs || {};
    const lines = [];
    const display = value => {
      if (value && typeof value === 'object') return JSON.stringify(value);
      return String(value);
    };
    if (Array.isArray(source)) {
      for (const item of source.slice(0, 3)) {
        lines.push(typeof item === 'object'
          ? `${item.label || item.name}: ${display(item.value)}` : item);
      }
    } else if (source && typeof source === 'object') {
      const priority = ['shape', 'dtype', 'layer_count', 'linear_attention_layers',
        'full_attention_layers', 'repeat_count', 'routed_experts', 'top_k',
        'shared_experts', 'intermediate_size', 'state_kind', 'internals',
        'parameters', 'parameter_count', 'flops', 'logical_bytes', 'state', 'cache'];
      const keys = [...priority.filter(key => key in source),
        ...Object.keys(source).filter(key => !priority.includes(key)).sort()];
      for (const key of keys.slice(0, 3)) lines.push(`${key}: ${display(source[key])}`);
    }
    return lines.map(item => clip(item, 34));
  }

  function buildPath(source, target) {
    const middle = source.x + Math.max(44, (target.x - source.x) / 2);
    if (target.x >= source.x + 40) {
      return `M ${source.x} ${source.y} H ${middle} V ${target.y} H ${target.x}`;
    }
    const detour = Math.max(source.y, target.y) + 52;
    return `M ${source.x} ${source.y} H ${source.x + 38} V ${detour}`
      + ` H ${target.x - 38} V ${target.y} H ${target.x}`;
  }

  function mount(root, graphDocument, options = {}) {
    if (!root || root.dataset.dagMounted === 'true') return instances.get(root && root.id);
    root.dataset.dagMounted = 'true';
    const views = [...((graphDocument && graphDocument.views) || [])].map(normalizeView);
    const viewsById = new Map(views.map(view => [viewId(view), view]));
    const stage = root.querySelector('.llm-dag-stage');
    const surface = root.querySelector('.llm-dag-surface');
    const world = root.querySelector('.llm-dag-world');
    const edgeLayer = root.querySelector('.llm-dag-edges');
    const nodeLayer = root.querySelector('.llm-dag-nodes');
    const minimap = root.querySelector('.llm-dag-minimap');
    const miniWorld = root.querySelector('.llm-dag-minimap-world');
    const miniViewport = root.querySelector('.llm-dag-minimap-viewport');
    const selector = root.querySelector('.llm-dag-view-select');
    const breadcrumb = root.querySelector('.llm-dag-breadcrumb');
    const back = root.querySelector('[data-dag-action="back"]');
    const empty = root.querySelector('.llm-dag-empty');
    const status = root.querySelector('.llm-dag-status');
    const searchInput = root.querySelector('.llm-dag-search');
    const heatLegend = root.querySelector('.llm-dag-heat-legend');
    const heatTitle = root.querySelector('.llm-dag-heat-title');
    const heatBasis = root.querySelector('.llm-dag-heat-basis');
    const heatCoverage = root.querySelector('.llm-dag-heat-coverage');
    let currentView = null;
    let layout = null;
    let scale = 1;
    let tx = 0;
    let ty = 0;
    let drag = null;
    let selected = null;
    let searchQuery = '';
    let heatmap = { mode: 'off', nodes: {} };

    function applyTransform() {
      world.setAttribute('transform', `translate(${tx} ${ty}) scale(${scale})`);
      updateMinimapViewport();
      const drillable = currentView
        ? (currentView.nodes || []).filter(node => node.drilldown_view_id
          && viewsById.has(String(node.drilldown_view_id))).length : 0;
      status.textContent = currentView
        ? `${currentView.nodes?.length || 0} nodes · ${currentView.edges?.length || 0} edges`
          + ` · ${Math.round(scale * 100)}%`
          + (drillable ? ` · ${drillable} × L1 › details` : '')
        : 'No graph view';
    }

    function fit() {
      if (!layout || !layout.width || !layout.height) return;
      const box = stage.getBoundingClientRect();
      scale = Math.min(1.3, Math.max(.12,
        Math.min((box.width - 30) / layout.width, (box.height - 30) / layout.height)));
      tx = (box.width - layout.width * scale) / 2;
      ty = (box.height - layout.height * scale) / 2;
      applyTransform();
    }

    function frameReadable() {
      if (!layout || !layout.width || !layout.height) return;
      const box = stage.getBoundingClientRect();
      const overviewScale = Math.min(
        (box.width - 30) / layout.width,
        (box.height - 30) / layout.height
      );
      scale = Math.min(1.15, Math.max(.7, overviewScale));
      tx = layout.width * scale <= box.width
        ? (box.width - layout.width * scale) / 2 : 22;
      ty = layout.height * scale <= box.height
        ? (box.height - layout.height * scale) / 2 : 22;
      applyTransform();
    }

    function centerNode(id) {
      const pos = layout && layout.positions.get(String(id));
      if (!pos) return false;
      const box = stage.getBoundingClientRect();
      tx = box.width / 2 - (pos.x + pos.width / 2) * scale;
      ty = box.height / 2 - (pos.y + pos.height / 2) * scale;
      applyTransform();
      return true;
    }

    function zoom(factor, clientX, clientY) {
      const box = stage.getBoundingClientRect();
      const px = clientX == null ? box.width / 2 : clientX - box.left;
      const py = clientY == null ? box.height / 2 : clientY - box.top;
      const next = Math.min(2.5, Math.max(.12, scale * factor));
      tx = px - ((px - tx) / scale) * next;
      ty = py - ((py - ty) / scale) * next;
      scale = next;
      applyTransform();
    }

    function chainFor(view) {
      const result = [];
      const seen = new Set();
      let cursor = view;
      while (cursor && !seen.has(viewId(cursor))) {
        seen.add(viewId(cursor));
        result.unshift(cursor);
        cursor = viewsById.get(String(cursor.parent_view_id || ''));
      }
      return result;
    }

    function renderBreadcrumb() {
      breadcrumb.replaceChildren();
      const chain = chainFor(currentView);
      chain.forEach((view, index) => {
        if (index) breadcrumb.append(text('span', '/', { class: 'llm-dag-separator' }));
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'llm-dag-crumb';
        button.textContent = view.label || view.name || viewId(view);
        button.title = `Open ${button.textContent}`;
        button.addEventListener('click', () => openView(viewId(view)));
        breadcrumb.append(button);
      });
      back.disabled = !currentView || !currentView.parent_view_id;
    }

    function dispatchSelection(type, item, extra = {}) {
      const detail = { type, item, view: currentView, ...extra };
      if (typeof options.onInspect === 'function') options.onInspect(detail);
      else if (typeof window.llmVisInspect === 'function') window.llmVisInspect(detail);
      root.dispatchEvent(new CustomEvent('llm-vis:dag-select', { bubbles: true, detail }));
    }

    function relationSets(id) {
      const upstream = new Set();
      const downstream = new Set();
      const walk = (start, direction, result) => {
        const queue = [start];
        while (queue.length) {
          const cursor = queue.shift();
          for (const edge of currentView.edges || []) {
            const source = String(edge.source_node_id);
            const target = String(edge.target_node_id);
            const next = direction === 'up' && target === cursor ? source
              : direction === 'down' && source === cursor ? target : null;
            if (next && next !== id && !result.has(next)) {
              result.add(next);
              queue.push(next);
            }
          }
        }
      };
      walk(id, 'up', upstream);
      walk(id, 'down', downstream);
      return { upstream, downstream };
    }

    function selectNode(node) {
      const id = nodeId(node);
      selected = { type: 'node', id };
      const relations = relationSets(id);
      root.querySelectorAll('.llm-dag-node').forEach(el => {
        const itemId = el.dataset.nodeId;
        el.classList.toggle('is-selected', itemId === id);
        el.classList.toggle('is-upstream', relations.upstream.has(itemId));
        el.classList.toggle('is-downstream', relations.downstream.has(itemId));
        el.classList.toggle('is-dimmed', itemId !== id && !relations.upstream.has(itemId)
          && !relations.downstream.has(itemId));
      });
      root.querySelectorAll('.llm-dag-edge-group').forEach(el => {
        const source = el.dataset.sourceNodeId;
        const target = el.dataset.targetNodeId;
        const related = source === id || target === id
          || (relations.upstream.has(source) && (relations.upstream.has(target) || target === id))
          || (relations.downstream.has(target)
            && (relations.downstream.has(source) || source === id));
        el.classList.toggle('is-dimmed', !related);
        el.classList.remove('is-selected');
      });
      dispatchSelection('node', node, {
        upstreamIds: [...relations.upstream], downstreamIds: [...relations.downstream]
      });
    }

    function selectEdge(edge, id) {
      selected = { type: 'edge', id };
      root.querySelectorAll('.llm-dag-node, .llm-dag-edge-group')
        .forEach(el => el.classList.remove('is-selected', 'is-upstream', 'is-downstream',
          'is-dimmed'));
      const target = root.querySelector(`.llm-dag-edge-group[data-edge-id="${CSS.escape(id)}"]`);
      if (target) target.classList.add('is-selected');
      dispatchSelection('edge', edge);
    }

    function portPoint(node, port, direction, index, count) {
      const pos = layout.positions.get(nodeId(node));
      const spacing = Math.min(18, (pos.height - 54) / Math.max(count, 1));
      return {
        x: direction === 'input' ? pos.x : pos.x + pos.width,
        y: pos.y + 106 + index * spacing
      };
    }

    function drawNode(node) {
      const pos = layout.positions.get(nodeId(node));
      const drilldownId = node.drilldown_view_id && String(node.drilldown_view_id);
      const canDrill = Boolean(drilldownId && viewsById.has(drilldownId));
      const label = node.label || nodeId(node);
      const baseAria = canDrill
        ? `${label} has L1 details. Click the L1 badge, double-click, or press Enter to open;`
          + ' Space inspects.'
        : `Inspect ${label}. Enter or Space selects.`;
      const baseTooltip = canDrill
        ? `Open L1 details for ${label} — click L1 ›, double-click, or press Enter`
        : `Inspect ${label}`;
      const group = svg('g', {
        class: `llm-dag-node${canDrill ? ' has-drilldown' : ''}`,
        tabindex: '0', role: 'button', 'aria-label': baseAria,
        'data-base-aria-label': baseAria, 'data-base-tooltip': baseTooltip,
        'data-node-id': nodeId(node),
        ...(canDrill ? {
          'aria-keyshortcuts': 'Enter', 'data-drilldown-view-id': drilldownId
        } : {})
      });
      group.setAttribute('transform', `translate(${pos.x} ${pos.y})`);
      group.append(text('title', baseTooltip, { class: 'llm-dag-node-tooltip' }));
      group.append(svg('rect', {
        class: 'llm-dag-node-card', width: pos.width, height: pos.height, rx: 9
      }));
      group.append(svg('path', {
        class: 'llm-dag-node-header', d: `M 9 0 H ${pos.width - 9} Q ${pos.width} 0`
          + ` ${pos.width} 9 V 38 H 0 V 9 Q 0 0 9 0 Z`
      }));
      group.append(text('text', clip(node.label || node.name || nodeId(node), 28), {
        class: 'llm-dag-node-title', x: 12, y: 17
      }));
      const kind = node.kind || node.type || 'node';
      const origin = node.origin || 'unknown-origin';
      const coverage = node.coverage_status ?? node.coverage ?? 'unknown-coverage';
      group.append(text('text', clip(`${kind} · ${origin} · ${coverage}`, 35), {
        class: 'llm-dag-node-meta', x: 12, y: 32
      }));
      coreLines(node).forEach((line, index) => {
        group.append(text('text', line, {
          class: 'llm-dag-node-core', x: 12, y: 54 + index * 14
        }));
      });
      if (canDrill) {
        const badge = svg('g', {
          class: 'llm-dag-drill-badge',
          transform: `translate(${pos.width - 34} -10)`,
          'aria-hidden': 'true'
        });
        badge.append(svg('rect', {
          class: 'llm-dag-drill-hit', x: -2, y: -4, width: 44, height: 28, rx: 12
        }));
        badge.append(svg('rect', {
          class: 'llm-dag-drill-pill', width: 40, height: 20, rx: 10
        }));
        badge.append(text('text', 'L1 ›', {
          class: 'llm-dag-drill-label', x: 20, y: 13, 'text-anchor': 'middle'
        }));
        badge.addEventListener('click', event => {
          event.stopPropagation();
          openView(drilldownId, { source: 'node-affordance' });
        });
        group.append(badge);
      }
      const ports = [...(node.ports || [])];
      const inputs = ports.filter(port => portDirection(port, node, layout.edges) === 'input');
      const outputs = ports.filter(port => portDirection(port, node, layout.edges) === 'output');
      for (const [direction, list] of [['input', inputs], ['output', outputs]]) {
        list.forEach((port, index) => {
          const point = portPoint(node, port, direction, index, list.length);
          const cx = direction === 'input' ? 0 : pos.width;
          const cy = point.y - pos.y;
          const portGroup = svg('g', {
            class: 'llm-dag-port-group', tabindex: '0', role: 'button',
            'aria-label': `Inspect ${direction} port ${port.label || port.name || portId(port)}`,
            'data-port-id': portId(port), 'data-port-direction': direction
          });
          portGroup.append(svg('circle', { class: 'llm-dag-port-hit', cx, cy, r: 12 }));
          portGroup.append(svg('circle', {
            class: 'llm-dag-port', cx, cy, r: 5, 'data-port-direction': direction
          }));
          const selectPort = event => {
            event.stopPropagation();
            if (event.type === 'keydown') event.preventDefault();
            dispatchSelection('port', port, { node, direction });
          };
          portGroup.addEventListener('click', selectPort);
          portGroup.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') selectPort(event);
          });
          group.append(portGroup);
          const name = clip(port.label || port.name || portId(port), 15);
          group.append(text('text', name, {
            class: 'llm-dag-port-label', x: direction === 'input' ? 10 : pos.width - 10,
            y: cy + 3, 'text-anchor': direction === 'input' ? 'start' : 'end'
          }));
        });
      }
      group.addEventListener('click', event => {
        event.stopPropagation();
        selectNode(node);
      });
      group.addEventListener('keydown', event => {
        if (event.key === 'Enter' && canDrill) {
          event.preventDefault();
          openView(drilldownId, { source: 'keyboard' });
        } else if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectNode(node);
        }
      });
      group.addEventListener('dblclick', event => {
        event.stopPropagation();
        if (canDrill) openView(drilldownId, { source: 'double-click' });
      });
      nodeLayer.append(group);
    }

    function edgeEndpoint(node, requestedPort, direction) {
      const ports = [...(node.ports || [])];
      const matching = ports.filter(port => portDirection(port, node, layout.edges) === direction);
      const index = Math.max(0, matching.findIndex(port => portId(port) === String(requestedPort)));
      const port = matching[index] || matching[0] || { id: requestedPort || direction };
      return portPoint(node, port, direction, index, matching.length || 1);
    }

    function drawEdges() {
      const nodeById = new Map(layout.nodes.map(node => [nodeId(node), node]));
      layout.edges.forEach((edge, index) => {
        const sourceNode = nodeById.get(String(edge.source_node_id));
        const targetNode = nodeById.get(String(edge.target_node_id));
        if (!sourceNode || !targetNode) return;
        const source = edgeEndpoint(sourceNode, edge.source_port_id, 'output');
        const target = edgeEndpoint(targetNode, edge.target_port_id, 'input');
        const id = edgeId(edge, index);
        const kind = edgeKind(edge);
        const path = buildPath(source, target);
        const group = svg('g', {
          class: 'llm-dag-edge-group', 'data-edge-id': id, tabindex: '0', role: 'button',
          'aria-label': `Inspect ${kind} edge ${edge.label || id}`,
          'data-source-node-id': nodeId(sourceNode),
          'data-target-node-id': nodeId(targetNode)
        });
        group.append(svg('path', {
          class: 'llm-dag-edge', d: path, 'data-edge-kind': kind,
          'marker-end': `url(#${root.id}-arrow-${kind})`
        }));
        const hit = svg('path', {
          class: 'llm-dag-edge-hit', d: path
        });
        group.addEventListener('click', event => {
          event.stopPropagation();
          selectEdge(edge, id);
        });
        group.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            selectEdge(edge, id);
          }
        });
        group.append(hit);
        const labelX = (source.x + target.x) / 2;
        const labelY = (source.y + target.y) / 2;
        const tensorName = edge.tensor_name || edge.label || 'tensor';
        const shapeLabel = edge.shape_label || 'Unknown';
        const dtypeLabel = edge.dtype_label || 'dtype Unknown';
        group.append(text('title', `${kind} · ${tensorName} · ${shapeLabel} · ${dtypeLabel}`));
        for (const [line, className, offset] of [
          [`[${kind}] ${tensorName}`, 'llm-dag-edge-label-name', -18],
          [shapeLabel, 'llm-dag-edge-label-shape', -7],
          [dtypeLabel, 'llm-dag-edge-label-dtype', 4]
        ]) {
          group.append(text('text', line, {
            class: `llm-dag-edge-label ${className}`, x: labelX,
            y: labelY + offset, 'text-anchor': 'middle'
          }));
        }
        edgeLayer.append(group);
      });
    }

    function drawMinimap() {
      miniWorld.replaceChildren();
      if (!layout || !layout.width || !layout.height) return;
      minimap.setAttribute('viewBox', `0 0 ${layout.width} ${layout.height}`);
      for (const edge of layout.edges) {
        const source = layout.positions.get(String(edge.source_node_id));
        const target = layout.positions.get(String(edge.target_node_id));
        if (!source || !target) continue;
        miniWorld.append(svg('path', {
          class: 'llm-dag-minimap-edge',
          d: `M ${source.x + source.width} ${source.y + source.height / 2}`
            + ` L ${target.x} ${target.y + target.height / 2}`
        }));
      }
      for (const pos of layout.positions.values()) {
        miniWorld.append(svg('rect', {
          class: 'llm-dag-minimap-node', x: pos.x, y: pos.y,
          width: pos.width, height: pos.height, rx: 5,
          'data-node-id': nodeId(pos.node)
        }));
      }
      miniWorld.append(miniViewport);
      updateMinimapViewport();
    }

    function updateMinimapViewport() {
      if (!layout || !layout.width || !layout.height) return;
      const box = stage.getBoundingClientRect();
      miniViewport.setAttribute('x', Math.max(0, -tx / scale));
      miniViewport.setAttribute('y', Math.max(0, -ty / scale));
      miniViewport.setAttribute('width', Math.min(layout.width, box.width / scale));
      miniViewport.setAttribute('height', Math.min(layout.height, box.height / scale));
    }

    function heatColor(value) {
      const normalized = Math.max(0, Math.min(1, Number(value) || 0));
      const stops = [[23, 58, 88], [128, 103, 34], [135, 53, 60]];
      const position = normalized * 2;
      const index = Math.min(1, Math.floor(position));
      const mix = position - index;
      const rgb = stops[index].map((channel, offset) =>
        Math.round(channel + (stops[index + 1][offset] - channel) * mix));
      return `rgb(${rgb.join(',')})`;
    }

    function applyHeatmap() {
      const mode = String(heatmap.mode || 'off');
      const entries = heatmap.nodes || {};
      root.dataset.heatmap = mode;
      heatLegend.hidden = mode === 'off';
      heatTitle.textContent = heatmap.title || 'Theoretical heat';
      heatBasis.textContent = heatmap.basis || 'Relative within this view; not measured latency.';
      heatBasis.title = heatmap.basis || '';
      heatCoverage.textContent = heatmap.coverageLabel || '';
      root.querySelectorAll('.llm-dag-node').forEach(element => {
        const entry = entries[element.dataset.nodeId] || null;
        const normalized = Number(entry?.normalized);
        const known = mode !== 'off' && entry?.known === true
          && Number.isFinite(normalized);
        const statusName = mode === 'off' ? 'off'
          : String(entry?.status || (known ? 'known' : 'unknown'));
        element.dataset.heatKnown = String(known);
        element.dataset.heatStatus = statusName;
        if (known) element.style.setProperty('--dag-node-heat', heatColor(normalized));
        else element.style.removeProperty('--dag-node-heat');
        const heatText = mode === 'off' ? '' : (entry?.accessibilityLabel
          || (known ? `${heatmap.title || 'Theoretical heat'}: ${entry.valueLabel}`
            : `${heatmap.title || 'Theoretical heat'}: Unknown. ${entry?.reason || ''}`));
        const baseAria = element.dataset.baseAriaLabel || '';
        const baseTooltip = element.dataset.baseTooltip || '';
        element.setAttribute('aria-label', heatText ? `${baseAria} ${heatText}` : baseAria);
        const tooltip = element.querySelector('.llm-dag-node-tooltip');
        if (tooltip) tooltip.textContent = heatText
          ? `${baseTooltip} · ${heatText}` : baseTooltip;
      });
      root.querySelectorAll('.llm-dag-minimap-node').forEach(element => {
        const entry = entries[element.dataset.nodeId] || null;
        const normalized = Number(entry?.normalized);
        const known = mode !== 'off' && entry?.known === true
          && Number.isFinite(normalized);
        element.dataset.heatKnown = String(known);
        element.dataset.heatStatus = mode === 'off' ? 'off'
          : String(entry?.status || (known ? 'known' : 'unknown'));
        if (known) element.style.fill = heatColor(normalized);
        else element.style.removeProperty('fill');
      });
    }

    function setHeatmap(next = {}) {
      heatmap = {
        mode: String(next.mode || 'off'),
        nodes: next.nodes && typeof next.nodes === 'object' ? next.nodes : {},
        title: next.title || '', basis: next.basis || '',
        coverageLabel: next.coverageLabel || ''
      };
      applyHeatmap();
      return true;
    }

    function applySearch(query) {
      searchQuery = String(query || '').trim().toLowerCase();
      let count = 0;
      root.querySelectorAll('.llm-dag-node').forEach(el => {
        const node = layout && layout.positions.get(el.dataset.nodeId)?.node;
        const content = node ? JSON.stringify(node).toLowerCase() : '';
        const match = Boolean(searchQuery && content.includes(searchQuery));
        el.classList.toggle('is-match', match);
        if (match) count += 1;
      });
      return count;
    }

    function searchViews(query) {
      searchQuery = String(query || '').trim().toLowerCase();
      searchInput.value = query || '';
      if (!searchQuery) return applySearch('');
      const matchingNode = view => (view.nodes || []).find(node =>
        JSON.stringify(node).toLowerCase().includes(searchQuery));
      let node = currentView && matchingNode(currentView);
      if (!node) {
        const targetView = views.find(view => matchingNode(view));
        if (targetView) {
          node = matchingNode(targetView);
          if (targetView !== currentView) {
            openView(viewId(targetView), { source: 'search' });
          }
        }
      }
      const count = applySearch(searchQuery);
      if (node) requestAnimationFrame(() => {
        selectNode(node);
        centerNode(nodeId(node));
      });
      return count;
    }

    function focusNode(id) {
      const node = layout && layout.positions.get(String(id))?.node;
      if (!node) return false;
      selectNode(node);
      requestAnimationFrame(() => centerNode(id));
      return true;
    }

    function openView(id, detail = {}) {
      const next = viewsById.get(String(id));
      if (!next) return false;
      currentView = next;
      selected = null;
      selector.value = viewId(next);
      edgeLayer.replaceChildren();
      nodeLayer.replaceChildren();
      layout = stableLayout(next);
      drawEdges();
      layout.nodes.forEach(drawNode);
      empty.hidden = layout.nodes.length > 0;
      renderBreadcrumb();
      drawMinimap();
      applyHeatmap();
      applySearch(searchQuery);
      requestAnimationFrame(frameReadable);
      const eventDetail = { view: next, viewId: viewId(next), ...detail };
      root.dispatchEvent(new CustomEvent('llm-vis:dag-view-change', {
        bubbles: true, detail: eventDetail
      }));
      return true;
    }

    selector.replaceChildren();
    views.forEach(view => {
      const option = document.createElement('option');
      option.value = viewId(view);
      option.textContent = view.label || view.name || viewId(view);
      selector.append(option);
    });
    selector.addEventListener('change', () => openView(selector.value, { source: 'selector' }));
    searchInput.addEventListener('input', () => searchViews(searchInput.value));
    root.querySelectorAll('[data-dag-action]').forEach(button => {
      button.addEventListener('click', () => {
        const action = button.dataset.dagAction;
        if (action === 'fit') fit();
        else if (action === 'zoom-in') zoom(1.2);
        else if (action === 'zoom-out') zoom(1 / 1.2);
        else if (action === 'back' && currentView?.parent_view_id) {
          openView(String(currentView.parent_view_id), { source: 'back' });
        }
      });
    });
    stage.addEventListener('wheel', event => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12, event.clientX, event.clientY);
    }, { passive: false });
    stage.addEventListener('pointerdown', event => {
      if (event.target.closest('.llm-dag-node, .llm-dag-edge-hit')) return;
      stage.setPointerCapture(event.pointerId);
      stage.classList.add('is-panning');
      drag = { x: event.clientX, y: event.clientY, tx, ty };
      selected = null;
      root.querySelectorAll('.is-selected, .is-upstream, .is-downstream, .is-dimmed')
        .forEach(el => el.classList.remove('is-selected', 'is-upstream',
          'is-downstream', 'is-dimmed'));
    });
    stage.addEventListener('pointermove', event => {
      if (!drag) return;
      tx = drag.tx + event.clientX - drag.x;
      ty = drag.ty + event.clientY - drag.y;
      applyTransform();
    });
    const endPan = () => { drag = null; stage.classList.remove('is-panning'); };
    stage.addEventListener('pointerup', endPan);
    stage.addEventListener('pointercancel', endPan);
    window.addEventListener('resize', frameReadable);

    const controller = {
      root,
      get view() { return currentView; },
      get selection() { return selected; },
      fit,
      search: searchViews,
      focusNode,
      openView,
      setHeatmap,
      destroy() {
        window.removeEventListener('resize', frameReadable);
        instances.delete(root.id);
        root.dataset.dagMounted = 'false';
      }
    };
    instances.set(root.id, controller);
    if (views.length) openView(options.initialViewId || viewId(views[0]), { source: 'initial' });
    else {
      empty.hidden = false;
      selector.disabled = true;
      applyTransform();
    }
    return controller;
  }

  function controllerFor(target) {
    const id = typeof target === 'string' ? target : target && target.id;
    return instances.get(id);
  }

  function autoMount() {
    document.querySelectorAll('[data-llm-vis-dag]').forEach(root => {
      if (root.dataset.dagMounted === 'true') return;
      const payload = document.getElementById(root.dataset.dagPayloadId);
      if (!payload) return;
      try {
        mount(root, JSON.parse(payload.textContent), {
          initialViewId: root.dataset.initialViewId || undefined
        });
      } catch (error) {
        const empty = root.querySelector('.llm-dag-empty');
        empty.hidden = false;
        empty.textContent = `GraphView could not be rendered: ${error.message}`;
      }
    });
  }

  window.LLMVisDAG = {
    mount,
    get: controllerFor,
    fit: target => controllerFor(target)?.fit(),
    search: (target, query) => controllerFor(target)?.search(query),
    focusNode: (target, id) => controllerFor(target)?.focusNode(id),
    openView: (target, id) => controllerFor(target)?.openView(id),
    setHeatmap: (target, config) => controllerFor(target)?.setHeatmap(config)
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoMount, { once: true });
  } else {
    queueMicrotask(autoMount);
  }
})();
"""


def render_dag_canvas_assets() -> str:
    """Return the one-per-page offline CSS and JavaScript runtime."""

    return f"<style>\n{DAG_CANVAS_CSS}\n</style>\n<script>\n{DAG_CANVAS_JS}\n</script>"


def render_dag_canvas(
    document: Mapping[str, Any],
    *,
    element_id: str = "llm-vis-dag",
    payload_id: Optional[str] = None,
    initial_view_id: Optional[str] = None,
) -> str:
    """Render a read-only DAG canvas shell and its embedded GraphView JSON document.

    The returned fragment needs :func:`render_dag_canvas_assets` on the same page.  IDs are
    escaped for HTML, while the payload is encoded deterministically and protects the closing
    ``script`` sequence so arbitrary labels cannot terminate the JSON element.
    """

    payload_id = payload_id or f"{element_id}-data"
    safe_element_id = html.escape(element_id, quote=True)
    safe_payload_id = html.escape(payload_id, quote=True)
    initial_attr = ""
    if initial_view_id is not None:
        initial_attr = f' data-initial-view-id="{html.escape(initial_view_id, quote=True)}"'
    embedded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).replace("</", "<\\/")
    return f"""<section class="llm-dag" id="{safe_element_id}" data-llm-vis-dag
  data-dag-payload-id="{safe_payload_id}" data-readonly="true"{initial_attr}>
  <div class="llm-dag-toolbar" role="toolbar" aria-label="DAG canvas controls">
    <div class="llm-dag-nav-controls" aria-label="Graph navigation controls">
      <button type="button" data-dag-action="back" aria-label="Back to parent view">← Back</button>
      <label>View <select class="llm-dag-view-select" aria-label="Graph view"></select></label>
    </div>
    <nav class="llm-dag-breadcrumb" aria-label="Graph breadcrumb"></nav>
    <div class="llm-dag-view-controls" aria-label="Canvas view controls">
      <input class="llm-dag-search" type="search" placeholder="Find in graph"
        aria-label="Find graph item">
      <button type="button" data-dag-action="zoom-out" aria-label="Zoom out">−</button>
      <button type="button" data-dag-action="zoom-in" aria-label="Zoom in">+</button>
      <button type="button" data-dag-action="fit">Fit</button>
      <span class="llm-dag-readonly">Read-only</span>
    </div>
  </div>
  <div class="llm-dag-heat-legend" role="status" aria-live="polite" hidden>
    <strong class="llm-dag-heat-title">Theoretical heat</strong>
    <span>low</span><span class="llm-dag-heat-scale" aria-hidden="true"></span><span>high</span>
    <span class="llm-dag-heat-unknown">gray = Unknown / not attributable</span>
    <span class="llm-dag-heat-basis"></span>
    <span class="llm-dag-heat-coverage"></span>
  </div>
  <div class="llm-dag-stage">
    <svg class="llm-dag-surface" role="group" aria-label="Interactive model dataflow graph">
      <defs>
        <marker id="{safe_element_id}-arrow-data" markerWidth="8" markerHeight="8"
          refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#91bfff"/></marker>
        <marker id="{safe_element_id}-arrow-state" markerWidth="8" markerHeight="8"
          refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#e3a7ff"/></marker>
        <marker id="{safe_element_id}-arrow-route" markerWidth="8" markerHeight="8"
          refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#ffd172"/></marker>
        <marker id="{safe_element_id}-arrow-control" markerWidth="8" markerHeight="8"
          refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#a8b5c8"/></marker>
      </defs>
      <g class="llm-dag-world">
        <g class="llm-dag-edges"></g>
        <g class="llm-dag-nodes"></g>
      </g>
    </svg>
    <svg class="llm-dag-minimap" aria-label="Graph minimap">
      <g class="llm-dag-minimap-world"></g>
      <rect class="llm-dag-minimap-viewport" x="0" y="0" width="0" height="0"/>
    </svg>
    <div class="llm-dag-status" aria-live="polite"></div>
    <div class="llm-dag-empty" hidden>No nodes in this view.</div>
  </div>
</section>
<script type="application/json" id="{safe_payload_id}">{embedded}</script>"""

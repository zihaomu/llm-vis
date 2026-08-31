"""Real-browser acceptance tests for the self-contained offline report.

The suite is opt-in locally because Playwright's Chromium binary is intentionally
not part of the core package.  CI enables it in a dedicated job after installing
Chromium::

    LLM_VIS_BROWSER_E2E=1 uv run pytest -q -m browser tests/browser
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Browser, Page, expect, sync_playwright

from llm_vis.cli import main

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_FIXTURES = REPOSITORY_ROOT / "tests" / "fixtures" / "configs"
_BROWSER_ENABLED = os.environ.get("LLM_VIS_BROWSER_E2E") == "1"

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(
        not _BROWSER_ENABLED,
        reason="set LLM_VIS_BROWSER_E2E=1 to run real Chromium acceptance tests",
    ),
]


def _generate_view(
    source: Path, output: Path, *, extra_args: tuple[str, ...] = ()
) -> Path:
    assert (
        main(
            [
                "view",
                str(source),
                "--output",
                str(output),
                "--no-open",
                *extra_args,
            ]
        )
        == 0
    )
    report = output / "reports" / "report.html"
    assert report.is_file()
    return report


@pytest.fixture(scope="session")
def qwen_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _generate_view(
        CONFIG_FIXTURES / "qwen3_8_27b.json",
        tmp_path_factory.mktemp("qwen-browser-report"),
    )


@pytest.fixture(scope="session")
def glm_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _generate_view(
        CONFIG_FIXTURES / "glm_5_3_bf16.json",
        tmp_path_factory.mktemp("glm-browser-report"),
    )


@pytest.fixture(scope="session")
def qwen_profiled_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _generate_view(
        CONFIG_FIXTURES / "qwen3_8_27b.json",
        tmp_path_factory.mktemp("qwen-profiled-browser-report"),
        extra_args=(
            "--hardware-profile",
            str(REPOSITORY_ROOT / "examples" / "hardware" / "synthetic-bf16.json"),
        ),
    )


@pytest.fixture(scope="session")
def generic_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("generic-browser-report")
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "future_wrapper",
                "text_config": {
                    "model_type": "future_text",
                    "num_hidden_layers": 3,
                    "hidden_size": 96,
                    "vocab_size": 1024,
                },
            }
        ),
        encoding="utf-8",
    )
    return _generate_view(config, root / "artifact")


@pytest.fixture(scope="session")
def chromium() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@contextmanager
def _open_report(
    browser: Browser,
    report: Path,
    *,
    width: int = 1280,
    height: int = 900,
) -> Iterator[tuple[Page, list[str]]]:
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    browser_errors: list[str] = []

    def record_console(message: object) -> None:
        message_type = getattr(message, "type", "")
        if message_type == "error":
            browser_errors.append(f"console.error: {getattr(message, 'text', message)}")

    page.on("console", record_console)
    page.on("pageerror", lambda error: browser_errors.append(f"pageerror: {error}"))
    try:
        page.goto(
            report.resolve().as_uri(),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        page.wait_for_function(
            "() => document.querySelector('#model-dag')?.dataset.dagMounted === 'true'"
        )
        expect(page.locator("#model-dag .llm-dag-node").first).to_be_visible()
        yield page, browser_errors
    finally:
        context.close()


def _node(page: Page, label: str):  # type: ignore[no-untyped-def]
    return page.locator("#model-dag .llm-dag-node", has_text=label).first


def _selected_view(page: Page):  # type: ignore[no-untyped-def]
    return page.locator("#model-dag .llm-dag-view-select option:checked")


def _center_node(page: Page, node) -> None:  # type: ignore[no-untyped-def]
    node_id = node.get_attribute("data-node-id")
    assert node_id is not None
    assert page.evaluate(
        "nodeId => window.LLMVisDAG.focusNode('model-dag', nodeId)", node_id
    )
    page.wait_for_timeout(100)


def _assert_browser_clean(browser_errors: list[str]) -> None:
    assert browser_errors == []


def test_qwen_explain_drilldown_search_and_formula_heat(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    with _open_report(chromium, qwen_report) as (page, browser_errors):
        source_badge = page.locator("#source-evidence-badge")
        expect(source_badge).to_contain_text("Known adapter · qwen3-5")
        source_title = source_badge.get_attribute("title") or ""
        assert "Input local_file" in source_title
        assert "resolved revision sha256:" in source_title
        assert "config SHA-256" in source_title

        scenarios = page.locator("#scenario-select option")
        expect(scenarios).to_have_count(2)
        expect(scenarios.nth(0)).to_have_text(
            "Default preset · Prefill · B1 T512 L0 · bfloat16"
        )
        expect(scenarios.nth(1)).to_have_text(
            "Default preset · Decode · B1 T1 L512 · bfloat16"
        )

        heatmap = page.locator("#heatmap-mode")
        pressure = heatmap.locator('option[value="pressure"]')
        expect(pressure).to_have_attribute("disabled", "")
        expect(pressure).to_have_text("Pressure (requires HardwareProfile)")
        expect(page.locator("#heatmap-note")).to_have_text(
            "Pressure requires an explicit HardwareProfile."
        )
        expect(heatmap).to_have_value("compute")
        expect(page.locator("#model-dag .llm-dag-heat-title")).to_have_text(
            "Theoretical compute"
        )
        expect(
            page.locator('#model-dag .llm-dag-node[data-heat-known="true"]').first
        ).to_be_visible()

        heatmap.select_option("memory")
        expect(page.locator("#model-dag .llm-dag-heat-title")).to_have_text(
            "Theoretical memory"
        )
        expect(
            page.locator('#model-dag .llm-dag-node[data-heat-known="true"]').first
        ).to_be_visible()

        hybrid_decoder = _node(page, "Hybrid Decoder")
        expect(hybrid_decoder).to_be_visible()
        hybrid_decoder_id = hybrid_decoder.get_attribute("data-node-id")
        assert hybrid_decoder_id is not None
        hybrid_decoder.click()
        inspector = page.locator("#inspector-drawer")
        expect(inspector).to_have_attribute("aria-hidden", "false")
        expect(page.locator("#inspector-title")).to_contain_text("Hybrid Decoder")
        expect(page.locator("#inspector")).to_contain_text("Simplified equation")
        expect(page.locator("#inspector .inspector-formula")).to_contain_text(
            "DecoderLayer"
        )
        child_buttons = page.locator("#inspector .inspector-child")
        expect(child_buttons).to_have_count(2)
        expect(child_buttons.nth(1)).to_contain_text("Alternate")
        child_buttons.nth(1).click()
        expect(inspector).to_have_attribute("aria-hidden", "true")
        expect(_selected_view(page)).to_have_text("Full Attention (GQA) representative")
        page.get_by_role("button", name="Collapse to parent graph").click()
        expect(_selected_view(page)).to_have_text("Qwen model DAG")
        expect(_node(page, "Hybrid Decoder")).to_have_attribute(
            "data-node-id", hybrid_decoder_id
        )

        hybrid_decoder = _node(page, "Hybrid Decoder")
        hybrid_decoder.dblclick()
        expect(_selected_view(page)).to_have_text(
            "Gated DeltaNet Linear Attention representative"
        )
        collapse = page.get_by_role("button", name="Collapse to parent graph")
        expect(collapse).to_be_enabled()
        collapse.click()
        expect(_selected_view(page)).to_have_text("Qwen model DAG")
        expect(_node(page, "Hybrid Decoder")).to_have_attribute(
            "data-node-id", hybrid_decoder_id
        )
        page.wait_for_function(
            """nodeId => document.querySelector(
                `#model-dag .llm-dag-node[data-node-id="${nodeId}"]`
            )?.classList.contains('is-selected')""",
            arg=hybrid_decoder_id,
        )

        layer_panel = page.locator("#layer-panel")
        expect(page.locator("#layer-pattern-summary")).to_have_text("[L×3 → A] ×16")
        expect(page.locator("#layer-legend")).to_contain_text("Linear Attention")
        expect(page.locator("#layer-legend")).to_contain_text("Full Attention")
        expect(page.locator("#strip .layer")).to_have_count(0)
        state_before_layer_toggle = page.evaluate(
            """() => ({
                scenario: document.querySelector('#scenario-select').value,
                heatmap: document.querySelector('#heatmap-mode').value,
                view: document.querySelector('#model-dag .llm-dag-view-select').value,
                transform: document
                    .querySelector('#model-dag .llm-dag-world').getAttribute('transform'),
                selected: document
                    .querySelector('#model-dag .llm-dag-node.is-selected')?.dataset.nodeId || null
            })"""
        )
        layer_panel.locator("summary").click()
        layers = page.locator("#strip .layer")
        expect(layers).to_have_count(64)
        layer_panel.locator("summary").click()
        layer_panel.locator("summary").click()
        expect(layers).to_have_count(64)
        state_after_layer_toggle = page.evaluate(
            """() => ({
                scenario: document.querySelector('#scenario-select').value,
                heatmap: document.querySelector('#heatmap-mode').value,
                view: document.querySelector('#model-dag .llm-dag-view-select').value,
                transform: document
                    .querySelector('#model-dag .llm-dag-world').getAttribute('transform'),
                selected: document
                    .querySelector('#model-dag .llm-dag-node.is-selected')?.dataset.nodeId || null
            })"""
        )
        assert state_after_layer_toggle == state_before_layer_toggle

        for layer_index, expected_view in (
            (0, "Gated DeltaNet Linear Attention representative"),
            (3, "Full Attention (GQA) representative"),
            (63, "Full Attention (GQA) representative"),
        ):
            layer = layers.nth(layer_index)
            expect(layer).to_have_attribute("data-layer-index", str(layer_index))
            if layer_index == 3:
                layer.focus()
                page.keyboard.press("Enter")
            else:
                layer.click()
            expect(_selected_view(page)).to_contain_text(expected_view)
            expect(page.locator("#inspector")).to_contain_text(
                f'"layer_index": {layer_index}'
            )
            page.get_by_role("button", name="Close inspector").click()

        page.get_by_role("button", name="Collapse to parent graph").click()
        expect(_selected_view(page)).to_have_text("Qwen model DAG")

        page.locator("#search").fill("Q RMSNorm")
        expect(_selected_view(page)).to_contain_text("Full Attention · primitive operators")
        match = page.locator("#model-dag .llm-dag-node.is-match", has_text="Q RMSNorm")
        expect(match).to_have_count(1)
        assert "is-selected" in (match.get_attribute("class") or "").split()

        _assert_browser_clean(browser_errors)


def test_qwen_key_controls_fit_without_horizontal_overflow_at_700px(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    with _open_report(chromium, qwen_report, width=700, height=900) as (
        page,
        browser_errors,
    ):
        for control in (
            page.locator("#scenario-select"),
            page.locator("#search"),
            page.locator("#heatmap-mode"),
            page.locator("#model-dag .llm-dag-view-select"),
            page.get_by_role("button", name="Collapse to parent graph"),
            page.get_by_role("button", name="Fit"),
        ):
            expect(control).to_be_visible()

        widths = page.evaluate(
            """() => ({
                viewport: window.innerWidth,
                document: document.documentElement.scrollWidth,
                body: document.body.scrollWidth
            })"""
        )
        assert widths["viewport"] == 700
        assert widths["document"] <= widths["viewport"]
        assert widths["body"] <= widths["viewport"]

        _node(page, "Hybrid Decoder").click()
        inspector = page.locator("#inspector-drawer")
        expect(inspector).to_have_attribute("aria-hidden", "false")
        box = inspector.bounding_box()
        assert box is not None
        assert box["x"] >= 0
        assert box["x"] + box["width"] <= 700.5
        assert page.evaluate("document.documentElement.scrollWidth") <= 700

        child_button = page.locator("#inspector .inspector-child").first
        expect(child_button).to_be_visible()
        child_button.click()
        expect(_selected_view(page)).to_have_text(
            "Gated DeltaNet Linear Attention representative"
        )
        page.get_by_role("button", name="Collapse to parent graph").click()
        expect(_selected_view(page)).to_have_text("Qwen model DAG")

        layer_panel = page.locator("#layer-panel")
        layer_panel.locator("summary").click()
        expect(layer_panel).to_have_attribute("open", "")
        expect(page.locator("#strip .layer")).to_have_count(64)
        assert page.evaluate("document.documentElement.scrollWidth") <= 700
        layer_panel.locator("summary").click()

        minimap = page.locator("#model-dag .llm-dag-minimap")
        expect(minimap).to_be_visible()
        expect(minimap.locator(".llm-dag-minimap-node")).to_have_count(11)
        minimap_viewport = minimap.locator(".llm-dag-minimap-viewport")
        viewport_width_before = float(minimap_viewport.get_attribute("width") or "0")
        for _ in range(4):
            page.get_by_role("button", name="Zoom in").click()
        viewport_width_after = float(minimap_viewport.get_attribute("width") or "0")
        assert viewport_width_after < viewport_width_before

        hybrid_decoder = _node(page, "Hybrid Decoder")
        _center_node(page, hybrid_decoder)
        hybrid_decoder.dblclick()
        expect(_selected_view(page)).to_have_text(
            "Gated DeltaNet Linear Attention representative"
        )
        parent_context = page.locator("#model-dag .llm-dag-parent-context")
        expect(parent_context).to_be_visible()
        parent_toggle = parent_context.locator(".llm-dag-parent-context-toggle")
        expect(parent_toggle).to_have_text("Show map")
        parent_toggle.click()
        expect(parent_toggle).to_have_attribute("aria-expanded", "true")
        parent_map = parent_context.locator(".llm-dag-parent-map")
        expect(parent_map).to_be_visible()
        expect(parent_map.locator(".llm-dag-parent-node")).to_have_count(11)
        parent_map.press("Enter")
        expect(_selected_view(page)).to_have_text("Qwen model DAG")
        assert page.evaluate("document.documentElement.scrollWidth") <= 700

        _assert_browser_clean(browser_errors)


def test_qwen_explicit_hardware_profile_enables_pressure(
    chromium: Browser,
    qwen_profiled_report: Path,
) -> None:
    with _open_report(chromium, qwen_profiled_report) as (page, browser_errors):
        heatmap = page.locator("#heatmap-mode")
        pressure = heatmap.locator('option[value="pressure"]')
        expect(pressure).not_to_have_attribute("disabled", "")
        expect(pressure).to_have_text("Pressure")
        expect(heatmap).to_have_value("pressure")
        expect(page.locator("#heatmap-note")).to_be_empty()
        expect(page.locator("#model-dag .llm-dag-heat-title")).to_have_text(
            "Theoretical pressure"
        )
        expect(page.locator("#model-dag .llm-dag-heat-basis")).to_contain_text(
            "Synthetic BF16 profile (test only)"
        )
        _assert_browser_clean(browser_errors)


def test_glm_preserves_static_moe_and_opaque_dsa_boundaries(
    chromium: Browser,
    glm_report: Path,
) -> None:
    with _open_report(chromium, glm_report) as (page, browser_errors):
        expect(page.locator("#layer-pattern-summary")).to_have_text("D×3 → M×75")
        expect(page.locator("#layer-legend")).to_contain_text("Dense FFN")
        expect(page.locator("#layer-legend")).to_contain_text("MoE FFN")
        expect(page.locator("#strip .layer")).to_have_count(0)
        page.locator("#layer-panel summary").click()
        glm_layers = page.locator("#strip .layer")
        expect(glm_layers).to_have_count(78)
        expect(glm_layers.nth(0)).to_have_text("D")
        expect(glm_layers.nth(2)).to_have_text("D")
        expect(glm_layers.nth(3)).to_have_text("M")
        expect(glm_layers.nth(77)).to_have_text("M")
        page.locator("#layer-panel summary").click()

        _node(page, "Sparse DSA + MoE").dblclick()
        expect(_selected_view(page)).to_have_text(
            "GLM Sparse DSA + MoE representative"
        )

        for label in ("Router GEMM", "TopK (k=8)", "Expert Pool"):
            expect(_node(page, label)).to_be_visible()

        dsa = _node(page, "DSA (internals Unknown)")
        dsa.click()
        expect(page.locator("#inspector-drawer")).to_have_attribute("aria-hidden", "false")
        expect(page.locator("#inspector-title")).to_contain_text("DSA (internals Unknown)")
        expect(page.locator("#inspector")).to_contain_text("Formula status")
        expect(page.locator("#inspector .inspector-formula")).to_have_text(
            "Unknown — opaque evidence boundary"
        )
        expect(page.locator("#inspector")).to_contain_text("Unknown is not zero")
        expect(page.locator("#inspector .inspector-child")).to_have_count(0)

        page.get_by_role("button", name="Close inspector").click()
        expert_pool = _node(page, "Expert Pool")
        _center_node(page, expert_pool)
        expert_pool.click()
        expect(page.locator("#inspector")).to_contain_text(
            "does not invent the runtime expert route"
        )
        page.get_by_role("button", name="Close inspector").click()

        expert_pool.dblclick()
        expect(_selected_view(page)).to_contain_text(
            "Symbolic routed Expert FFN · primitive operators"
        )
        for label in (
            "Static TopK route",
            "Gather selected expert inputs",
            "Scatter expert outputs",
        ):
            expect(_node(page, label)).to_be_visible()

        _assert_browser_clean(browser_errors)


def test_generic_fallback_searches_and_explains_without_children_or_cost(
    chromium: Browser,
    generic_report: Path,
) -> None:
    with _open_report(chromium, generic_report) as (page, browser_errors):
        expect(page.locator("#source-evidence-badge")).to_contain_text(
            "Unsupported · opaque · generic-config"
        )
        source_title = page.locator("#source-evidence-badge").get_attribute("title") or ""
        assert "Input local_file" in source_title
        assert "resolved revision sha256:" in source_title
        assert "config SHA-256" in source_title
        expect(_selected_view(page)).to_have_text("Config-only model skeleton")
        expect(page.locator("#model-dag .llm-dag-node.has-drilldown")).to_have_count(0)
        expect(page.locator("#model-dag .llm-dag-view-select option")).to_have_count(1)
        expect(page.locator("#heatmap-mode")).to_have_value("compute")
        expect(page.locator('#model-dag .llm-dag-node[data-heat-known="true"]')).to_have_count(
            0
        )

        page.locator("#search").fill("Architecture")
        architecture = page.locator(
            "#model-dag .llm-dag-node.is-match", has_text="Architecture"
        )
        expect(architecture).to_have_count(1)
        architecture.dblclick()

        expect(_selected_view(page)).to_have_text("Config-only model skeleton")
        expect(page.locator("#inspector-drawer")).to_have_attribute("aria-hidden", "false")
        expect(page.locator("#inspector-title")).to_contain_text("Architecture × 3 (opaque)")
        expect(page.locator("#inspector .inspector-formula")).to_have_text(
            "Unknown — opaque evidence boundary"
        )
        expect(page.locator("#inspector .inspector-child")).to_have_count(0)

        page.get_by_role("tab", name="Cost").click()
        expect(page.locator("#inspector")).to_contain_text(
            "No family adapter or operator evidence is available"
        )
        expect(page.locator("#cost-table")).to_contain_text(
            "No formula costs for this Scenario."
        )
        non_runtime_metrics = page.evaluate(
            """() => JSON.parse(document.querySelector('#llm-vis-data').textContent)
                .modelMap.metrics.filter(metric => !metric.name.startsWith('runtime.')).length"""
        )
        assert non_runtime_metrics == 0

        _assert_browser_clean(browser_errors)

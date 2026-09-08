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
    color_scheme: str = "no-preference",
    init_script: str = "",
) -> Iterator[tuple[Page, list[str]]]:
    context = browser.new_context(
        viewport={"width": width, "height": height},
        color_scheme=color_scheme,
    )
    if init_script:
        context.add_init_script(script=init_script)
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


def _theme_interaction_snapshot(page: Page) -> dict[str, object]:
    return page.evaluate(
        """() => ({
            view: document.querySelector('#model-dag .llm-dag-view-select').value,
            selected: document
                .querySelector('#model-dag .llm-dag-node.is-selected')?.dataset.nodeId || null,
            transform: document
                .querySelector('#model-dag .llm-dag-world').getAttribute('transform'),
            scenario: document.querySelector('#scenario-select').value,
            heatmap: document.querySelector('#heatmap-mode').value,
            search: document.querySelector('#search').value,
            layerOpen: document.querySelector('#layer-panel').open,
            inspectorOpen: document.querySelector('#inspector-drawer')
                .getAttribute('aria-hidden') === 'false',
            parentExpanded: document.querySelector(
                '#model-dag .llm-dag-parent-context-toggle'
            )?.getAttribute('aria-expanded') || null,
            nodeIds: [...document.querySelectorAll('#model-dag .llm-dag-node')]
                .map(node => node.dataset.nodeId),
            edgeIds: [...document.querySelectorAll('#model-dag .llm-dag-edge-group')]
                .map(edge => edge.dataset.edgeId),
            nodeHeat: [...document.querySelectorAll('#model-dag .llm-dag-node')]
                .map(node => [node.dataset.nodeId, node.dataset.heatKnown,
                    node.dataset.heatStatus]),
            minimapHeat: [...document.querySelectorAll(
                '#model-dag .llm-dag-minimap-node'
            )].map(node => [node.dataset.nodeId, node.dataset.heatKnown,
                node.dataset.heatStatus])
        })"""
    )


def _theme_computed_styles(page: Page) -> dict[str, str]:
    return page.evaluate(
        """() => {
            const value = (selector, property) => {
                const element = document.querySelector(selector);
                return element ? getComputedStyle(element)[property] : '';
            };
            return {
                bodyBackground: value('body', 'backgroundColor'),
                bodyText: value('body', 'color'),
                dagBackground: value('#model-dag', 'backgroundColor'),
                nodeHeader: value('#model-dag .llm-dag-node-header', 'fill'),
                nodeTitle: value('#model-dag .llm-dag-node-title', 'fill'),
                dataEdge: value(
                    '#model-dag .llm-dag-edge[data-edge-kind="data"]', 'stroke'
                ),
                minimapNode: value(
                    '#model-dag .llm-dag-minimap-node', 'fill'
                ),
                knownHeat: value(
                    '#model-dag .llm-dag-node[data-heat-known="true"] '
                        + '.llm-dag-node-card',
                    'fill'
                )
            };
        }"""
    )


def _heat_semantics_snapshot(page: Page) -> dict[str, object]:
    return page.evaluate(
        """() => ({
            mode: document.querySelector('#heatmap-mode').value,
            title: document.querySelector('#model-dag .llm-dag-heat-title').textContent,
            basis: document.querySelector('#model-dag .llm-dag-heat-basis').textContent,
            pressureDisabled: document.querySelector(
                '#heatmap-mode option[value="pressure"]'
            ).disabled,
            nodeHeat: [...document.querySelectorAll('#model-dag .llm-dag-node')]
                .map(node => [node.dataset.nodeId, node.dataset.heatKnown,
                    node.dataset.heatStatus]),
            minimapHeat: [...document.querySelectorAll(
                '#model-dag .llm-dag-minimap-node'
            )].map(node => [node.dataset.nodeId, node.dataset.heatKnown,
                node.dataset.heatStatus])
        })"""
    )


def _heat_palette_snapshot(page: Page) -> dict[str, object]:
    return page.evaluate(
        """() => {
            const root = document.querySelector('#model-dag');
            const styles = getComputedStyle(root);
            const node = root.querySelector('.llm-dag-node[data-heat-known="true"]')
                || root.querySelector('.llm-dag-node');
            const minimap = root.querySelector(
                `.llm-dag-minimap-node[data-node-id="${node.dataset.nodeId}"]`
            ) || root.querySelector('.llm-dag-minimap-node');
            return {
                tokens: [
                    '--dag-heat-low-rgb', '--dag-heat-mid-rgb',
                    '--dag-heat-high-rgb', '--dag-heat-unknown',
                    '--dag-heat-not-applicable'
                ].map(name => [name, styles.getPropertyValue(name).trim()]),
                nodeFill: getComputedStyle(
                    node.querySelector('.llm-dag-node-card')
                ).fill,
                minimapFill: getComputedStyle(minimap).fill,
                nodeId: node.dataset.nodeId,
                nodeKnown: node.dataset.heatKnown
            };
        }"""
    )


def _light_theme_contrast_ratios(page: Page) -> dict[str, float]:
    return page.evaluate(
        """() => {
            const parse = value => {
                const channels = String(value).match(/[0-9.]+/g)?.map(Number) || [];
                return channels.length >= 3 ? channels.slice(0, 3) : [0, 0, 0];
            };
            const luminance = value => {
                const linear = parse(value).map(channel => {
                    const normalized = channel / 255;
                    return normalized <= .04045 ? normalized / 12.92
                        : ((normalized + .055) / 1.055) ** 2.4;
                });
                return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
            };
            const ratio = (foreground, background) => {
                const first = luminance(foreground), second = luminance(background);
                return (Math.max(first, second) + .05) /
                    (Math.min(first, second) + .05);
            };
            const style = selector => getComputedStyle(document.querySelector(selector));
            const body = style('body'), control = style('#scenario-select');
            const dag = style('#model-dag');
            const nodeTitle = style('#model-dag .llm-dag-node-title');
            const nodeHeader = style('#model-dag .llm-dag-node-header');
            const edge = style('#model-dag .llm-dag-edge[data-edge-kind="data"]');
            const mainPanel = style('.dag-panel');
            const layerPanel = style('#layer-panel');
            const mutedSoft = style('#layer-summary-note');
            const root = style('html');
            return {
                bodyText: ratio(body.color, body.backgroundColor),
                controlText: ratio(control.color, control.backgroundColor),
                dagText: ratio(dag.color, dag.backgroundColor),
                nodeTitle: ratio(nodeTitle.fill, nodeHeader.fill),
                dataEdge: ratio(edge.stroke, dag.backgroundColor),
                controlBorder: ratio(
                    control.borderTopColor, control.backgroundColor
                ),
                lineStrong: ratio(
                    root.getPropertyValue('--line-strong'), mainPanel.backgroundColor
                ),
                mutedSoftText: ratio(
                    mutedSoft.color, layerPanel.backgroundColor
                )
            };
        }"""
    )


def _minimap_outline_contrast(page: Page) -> dict[str, object]:
    return page.evaluate(
        """() => {
            const parse = value => {
                const channels = String(value).match(/[0-9.]+/g)?.map(Number) || [];
                return {
                    rgb: channels.slice(0, 3),
                    alpha: channels.length > 3 ? channels[3] : 1
                };
            };
            const composite = (foreground, background) => {
                const fg = parse(foreground), bg = parse(background);
                return fg.rgb.map((channel, index) =>
                    channel * fg.alpha + bg.rgb[index] * (1 - fg.alpha));
            };
            const luminance = channels => {
                const linear = channels.map(channel => {
                    const normalized = channel / 255;
                    return normalized <= .04045 ? normalized / 12.92
                        : ((normalized + .055) / 1.055) ** 2.4;
                });
                return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
            };
            const ratio = (first, second) => {
                const left = luminance(first), right = luminance(second);
                return (Math.max(left, right) + .05) /
                    (Math.min(left, right) + .05);
            };
            const node = document.querySelector(
                '#model-dag .llm-dag-minimap-node[data-heat-known="true"]'
            );
            const minimapNode = getComputedStyle(node);
            const overlay = getComputedStyle(document.querySelector(
                '#model-dag .llm-dag-current-context'
            ));
            const dag = getComputedStyle(document.querySelector('#model-dag'));
            const background = composite(overlay.backgroundColor, dag.backgroundColor);
            const stroke = parse(minimapNode.stroke).rgb;
            return {
                ratio: ratio(stroke, background),
                stroke: minimapNode.stroke,
                strokeWidth: parseFloat(minimapNode.strokeWidth),
                heatKnown: node.dataset.heatKnown
            };
        }"""
    )


def _dag_semantic_token_contrast(page: Page) -> dict[str, object]:
    return page.evaluate(
        """() => {
            const root = document.querySelector('#model-dag');
            const styles = getComputedStyle(root);
            const token = name => styles.getPropertyValue(name).trim();
            const channels = value => {
                const text = String(value).trim();
                if (/^#[0-9a-f]{6}$/i.test(text)) {
                    return [1, 3, 5].map(index => parseInt(
                        text.slice(index, index + 2), 16
                    ));
                }
                return (text.match(/[0-9.]+/g) || []).slice(0, 3).map(Number);
            };
            const luminance = value => {
                const linear = channels(value).map(channel => {
                    const normalized = channel / 255;
                    return normalized <= .04045 ? normalized / 12.92
                        : ((normalized + .055) / 1.055) ** 2.4;
                });
                return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
            };
            const ratio = (foreground, background) => {
                const first = luminance(foreground), second = luminance(background);
                return (Math.max(first, second) + .05) /
                    (Math.min(first, second) + .05);
            };
            const sameColor = (first, second) => {
                const left = channels(first), right = channels(second);
                return left.length === 3 && right.length === 3
                    && left.every((value, index) => Math.abs(value - right[index]) < .5);
            };
            const dagBackground = token('--dag-bg');
            const parentBackground = token('--dag-parent-map-bg');
            const ratioPairs = {
                edgeData: ['--dag-edge-data', dagBackground],
                edgeState: ['--dag-edge-state', dagBackground],
                edgeRoute: ['--dag-edge-route', dagBackground],
                edgeControl: ['--dag-edge-control', dagBackground],
                portInput: ['--dag-port-input', dagBackground],
                portOutput: ['--dag-port-output', dagBackground],
                focus: ['--dag-focus', dagBackground],
                selection: ['--dag-selected', dagBackground],
                upstream: ['--dag-upstream', dagBackground],
                downstream: ['--dag-downstream', dagBackground],
                parentEdge: ['--dag-parent-edge', parentBackground],
                parentNode: ['--dag-parent-node', parentBackground],
                parentExpanded: ['--dag-parent-expanded', parentBackground]
            };
            const ratios = Object.fromEntries(Object.entries(ratioPairs).map(
                ([name, [foreground, background]]) => [
                    name, ratio(token(foreground), background)
                ]
            ));

            const surface = root.querySelector('.llm-dag-surface');
            const edgeTokenMatches = {}, markerTokenMatches = {};
            for (const kind of ['data', 'state', 'route', 'control']) {
                const edge = document.createElementNS(
                    'http://www.w3.org/2000/svg', 'path'
                );
                edge.setAttribute('class', 'llm-dag-edge');
                edge.dataset.edgeKind = kind;
                surface.append(edge);
                edgeTokenMatches[kind] = sameColor(
                    getComputedStyle(edge).stroke, token(`--dag-edge-${kind}`)
                );
                edge.remove();
                markerTokenMatches[kind] = sameColor(
                    getComputedStyle(root.querySelector(
                        `.llm-dag-arrow[data-edge-kind="${kind}"]`
                    )).fill,
                    token(`--dag-edge-${kind}`)
                );
            }
            const portTokenMatches = Object.fromEntries(
                ['input', 'output'].map(direction => {
                    const port = root.querySelector(
                        `.llm-dag-port[data-port-direction="${direction}"]`
                    );
                    return [direction, sameColor(
                        getComputedStyle(port).fill, token(`--dag-port-${direction}`)
                    )];
                })
            );
            const parentChecks = {
                edge: ['.llm-dag-parent-edge', '--dag-parent-edge', 'stroke'],
                node: ['.llm-dag-parent-node:not(.is-expanded-parent)',
                    '--dag-parent-node', 'fill'],
                expanded: ['.llm-dag-parent-node.is-expanded-parent',
                    '--dag-parent-expanded', 'fill']
            };
            const parentTokenMatches = Object.fromEntries(Object.entries(
                parentChecks
            ).map(([name, [selector, tokenName, property]]) => {
                const element = root.querySelector(selector);
                return [name, Boolean(element) && sameColor(
                    getComputedStyle(element)[property], token(tokenName)
                )];
            }));
            return {
                ratios,
                edgeTokenMatches,
                markerTokenMatches,
                portTokenMatches,
                parentTokenMatches
            };
        }"""
    )


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


@pytest.mark.parametrize("system_theme", ["light", "dark"])
def test_theme_first_use_follows_system_and_manual_choice_persists(
    chromium: Browser,
    qwen_report: Path,
    system_theme: str,
) -> None:
    with _open_report(
        chromium,
        qwen_report,
        color_scheme=system_theme,
    ) as (page, browser_errors):
        toggle = page.locator("#theme-toggle")
        expect(toggle).to_be_visible()
        expect(page.locator("html")).to_have_attribute("data-theme", system_theme)
        expect(toggle).to_have_attribute(
            "aria-pressed", str(system_theme == "light").lower()
        )
        expect(toggle).to_have_attribute(
            "aria-label", f"Use {'dark' if system_theme == 'light' else 'light'} theme"
        )
        assert page.evaluate("localStorage.getItem('llm-vis-theme')") is None

        page.evaluate(
            """() => {
                window.__llmVisThemeEvents = [];
                window.addEventListener('llm-vis:theme-change', event => {
                    window.__llmVisThemeEvents.push(event.detail);
                });
            }"""
        )
        manual_theme = "dark" if system_theme == "light" else "light"
        toggle.focus()
        page.keyboard.press("Enter")
        expect(page.locator("html")).to_have_attribute("data-theme", manual_theme)
        expect(toggle).to_have_attribute(
            "aria-pressed", str(manual_theme == "light").lower()
        )
        assert page.evaluate("localStorage.getItem('llm-vis-theme')") == manual_theme
        assert page.evaluate("window.__llmVisThemeEvents.at(-1)") == {
            "theme": manual_theme,
            "source": "user",
        }

        page.reload(wait_until="domcontentloaded")
        page.wait_for_function(
            "() => document.querySelector('#model-dag')?.dataset.dagMounted === 'true'"
        )
        expect(page.locator("html")).to_have_attribute("data-theme", manual_theme)
        expect(page.locator("#theme-toggle")).to_have_attribute(
            "aria-pressed", str(manual_theme == "light").lower()
        )
        _assert_browser_clean(browser_errors)


def test_theme_manual_choice_survives_storage_security_error_and_system_change(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    storage_denied = """
        Object.defineProperty(Storage.prototype, 'setItem', {
            configurable: true,
            value() {
                throw new DOMException('storage denied by test', 'SecurityError');
            }
        });
    """
    with _open_report(
        chromium,
        qwen_report,
        color_scheme="light",
        init_script=storage_denied,
    ) as (page, browser_errors):
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        assert page.evaluate("localStorage.getItem('llm-vis-theme')") is None

        page.emulate_media(color_scheme="dark")
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.emulate_media(color_scheme="light")
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        expect(page.locator("#theme-toggle")).to_have_attribute(
            "aria-pressed", "false"
        )
        _assert_browser_clean(browser_errors)


def test_report_mounts_with_dark_fallback_when_match_media_is_unavailable(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    without_match_media = """
        Object.defineProperty(window, 'matchMedia', {
            configurable: true,
            value: undefined
        });
    """
    with _open_report(
        chromium,
        qwen_report,
        color_scheme="light",
        init_script=without_match_media,
    ) as (page, browser_errors):
        assert page.evaluate("typeof window.matchMedia") == "undefined"
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        expect(page.locator("#model-dag")).to_have_attribute(
            "data-dag-mounted", "true"
        )
        expect(page.locator("#model-dag .llm-dag-node")).not_to_have_count(0)
        expect(page.locator("#model-dag .llm-dag-minimap")).to_be_visible()

        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        expect(page.locator("#model-dag")).to_have_attribute(
            "data-dag-mounted", "true"
        )
        _assert_browser_clean(browser_errors)


def test_report_mounts_when_match_media_factory_throws(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    throwing_match_media = """
        window.matchMedia = () => {
            throw new DOMException('media query denied by test', 'SecurityError');
        };
    """
    with _open_report(
        chromium,
        qwen_report,
        color_scheme="light",
        init_script=throwing_match_media,
    ) as (page, browser_errors):
        assert page.evaluate("typeof window.matchMedia") == "function"
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")

        toggle = page.locator("#theme-toggle")
        expect(toggle).to_be_visible()
        expect(toggle).to_have_attribute("aria-label", "Use light theme")
        scenario = page.locator("#scenario-select")
        expect(scenario).to_be_visible()
        expect(scenario.locator("option")).to_have_count(2)
        scenario.select_option(index=1)
        expect(scenario).to_have_value(
            page.locator("#scenario-select option").nth(1).get_attribute("value") or ""
        )

        expect(page.locator("#model-dag")).to_have_attribute(
            "data-dag-mounted", "true"
        )
        expect(page.locator("#model-dag .llm-dag-node")).not_to_have_count(0)
        expect(page.locator("#model-dag .llm-dag-minimap")).to_be_visible()
        toggle.click()
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        expect(page.locator("#model-dag")).to_have_attribute(
            "data-dag-mounted", "true"
        )
        _assert_browser_clean(browser_errors)


def test_report_and_dag_support_legacy_match_media_listeners(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    legacy_match_media = """
        window.__legacyMediaQueries = [];
        window.matchMedia = query => {
            const record = {
                query,
                matches: query.includes('prefers-color-scheme: light'),
                listeners: [],
                removed: 0
            };
            const result = {
                media: query,
                get matches() { return record.matches; },
                addListener(listener) { record.listeners.push(listener); },
                removeListener(listener) {
                    record.listeners = record.listeners.filter(item => item !== listener);
                    record.removed += 1;
                }
            };
            record.emit = matches => {
                record.matches = matches;
                for (const listener of [...record.listeners]) {
                    listener({matches, media: query});
                }
            };
            window.__legacyMediaQueries.push(record);
            return result;
        };
    """
    with _open_report(
        chromium,
        qwen_report,
        init_script=legacy_match_media,
    ) as (page, browser_errors):
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        expect(page.locator("#model-dag")).to_have_attribute(
            "data-dag-mounted", "true"
        )
        registered = page.evaluate(
            """() => window.__legacyMediaQueries
                .filter(item => item.listeners.length > 0)
                .map(item => item.query)"""
        )
        assert "(prefers-color-scheme: light)" in registered
        assert "(max-width: 720px)" in registered
        light_background = page.evaluate(
            "getComputedStyle(document.querySelector('#model-dag')).backgroundColor"
        )

        page.evaluate(
            """() => {
                for (const item of window.__legacyMediaQueries) {
                    if (item.query === '(prefers-color-scheme: light)') item.emit(false);
                }
            }"""
        )
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.wait_for_function(
            """light => getComputedStyle(
                document.querySelector('#model-dag')
            ).backgroundColor !== light""",
            arg=light_background,
        )

        page.evaluate("window.LLMVisDAG.get('model-dag').destroy()")
        removed = page.evaluate(
            """() => window.__legacyMediaQueries
                .reduce((total, item) => total + item.removed, 0)"""
        )
        assert removed >= 2
        _assert_browser_clean(browser_errors)


def test_dag_local_theme_override_wins_over_page_theme(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    with _open_report(
        chromium,
        qwen_report,
        color_scheme="light",
    ) as (page, browser_errors):
        dag = page.locator("#model-dag")
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        light_background = page.evaluate(
            "getComputedStyle(document.querySelector('#model-dag')).backgroundColor"
        )

        dag.evaluate("element => { element.dataset.theme = 'dark'; }")
        page.wait_for_function(
            """light => getComputedStyle(
                document.querySelector('#model-dag')
            ).backgroundColor !== light""",
            arg=light_background,
        )
        dark_background = page.evaluate(
            "getComputedStyle(document.querySelector('#model-dag')).backgroundColor"
        )
        expect(page.locator("html")).to_have_attribute("data-theme", "light")

        page.evaluate(
            """() => {
                document.documentElement.dataset.theme = 'dark';
                document.querySelector('#model-dag').dataset.theme = 'light';
            }"""
        )
        page.wait_for_function(
            """light => getComputedStyle(
                document.querySelector('#model-dag')
            ).backgroundColor === light""",
            arg=light_background,
        )
        assert dark_background != light_background
        expect(dag).to_have_attribute("data-dag-mounted", "true")
        _assert_browser_clean(browser_errors)


def test_dag_semantic_tokens_and_markers_meet_contrast_in_both_themes(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    with _open_report(
        chromium,
        qwen_report,
        color_scheme="light",
    ) as (page, browser_errors):
        _node(page, "Hybrid Decoder").dblclick()
        expect(_selected_view(page)).to_have_text(
            "Gated DeltaNet Linear Attention representative"
        )
        expect(
            page.locator("#model-dag .llm-dag-parent-node.is-expanded-parent")
        ).to_have_count(1)

        for theme in ("light", "dark"):
            expect(page.locator("html")).to_have_attribute("data-theme", theme)
            result = _dag_semantic_token_contrast(page)
            ratios = result["ratios"]
            assert isinstance(ratios, dict)
            for name, value in ratios.items():
                assert float(value) >= 3.0, f"{theme}:{name}={value}"
            for group_name in (
                "edgeTokenMatches",
                "markerTokenMatches",
                "portTokenMatches",
                "parentTokenMatches",
            ):
                matches = result[group_name]
                assert isinstance(matches, dict)
                assert all(matches.values()), f"{theme}:{group_name}={matches}"
            if theme == "light":
                marker_before = page.evaluate(
                    """getComputedStyle(document.querySelector(
                        '#model-dag .llm-dag-arrow[data-edge-kind="data"]'
                    )).fill"""
                )
                page.locator("#theme-toggle").click()
                page.wait_for_function(
                    """before => getComputedStyle(document.querySelector(
                        '#model-dag .llm-dag-arrow[data-edge-kind="data"]'
                    )).fill !== before""",
                    arg=marker_before,
                )

        _assert_browser_clean(browser_errors)


def test_qwen_theme_switch_preserves_dag_heat_and_interaction_state(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    with _open_report(
        chromium,
        qwen_report,
        color_scheme="light",
    ) as (page, browser_errors):
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        page.locator("#scenario-select").select_option(index=1)
        page.locator("#heatmap-mode").select_option("memory")

        hybrid_decoder = _node(page, "Hybrid Decoder")
        _center_node(page, hybrid_decoder)
        hybrid_decoder.dblclick()
        expect(_selected_view(page)).to_contain_text(
            "Gated DeltaNet Linear Attention representative"
        )
        page.locator("#search").fill("Q RMSNorm")
        expect(_selected_view(page)).to_contain_text(
            "Full Attention · primitive operators"
        )
        match = page.locator(
            "#model-dag .llm-dag-node.is-match", has_text="Q RMSNorm"
        )
        expect(match).to_have_count(1)
        match.click()
        expect(page.locator("#inspector-drawer")).to_have_attribute(
            "aria-hidden", "false"
        )
        page.locator("#layer-panel summary").click()
        expect(page.locator("#layer-panel")).to_have_attribute("open", "")

        state_before = _theme_interaction_snapshot(page)
        light_styles = _theme_computed_styles(page)
        light_outline = _minimap_outline_contrast(page)
        contrast = _light_theme_contrast_ratios(page)
        assert contrast["bodyText"] >= 4.5
        assert contrast["controlText"] >= 4.5
        assert contrast["dagText"] >= 4.5
        assert contrast["nodeTitle"] >= 4.5
        assert contrast["dataEdge"] >= 3.0
        assert contrast["controlBorder"] >= 3.0
        assert contrast["lineStrong"] >= 3.0
        assert contrast["mutedSoftText"] >= 4.5
        assert light_outline["heatKnown"] == "true"
        assert float(light_outline["strokeWidth"]) >= 1.0
        assert float(light_outline["ratio"]) >= 3.0

        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.wait_for_function(
            """previous => getComputedStyle(document.querySelector(
                '#model-dag .llm-dag-node[data-heat-known="true"] '
                    + '.llm-dag-node-card'
            )).fill !== previous""",
            arg=light_styles["knownHeat"],
        )
        state_after = _theme_interaction_snapshot(page)
        dark_styles = _theme_computed_styles(page)
        dark_outline = _minimap_outline_contrast(page)

        assert state_after == state_before
        assert dark_outline["heatKnown"] == "true"
        assert dark_outline["strokeWidth"] == light_outline["strokeWidth"]
        assert float(dark_outline["ratio"]) >= 3.0
        for key in (
            "bodyBackground",
            "bodyText",
            "dagBackground",
            "nodeHeader",
            "nodeTitle",
            "dataEdge",
            "minimapNode",
            "knownHeat",
        ):
            assert dark_styles[key] != light_styles[key], key
        _assert_browser_clean(browser_errors)


@pytest.mark.parametrize(
    ("report_fixture", "model_case"),
    [
        ("glm_report", "glm_moe"),
        ("qwen_profiled_report", "qwen_pressure"),
        ("generic_report", "generic_unknown"),
    ],
)
def test_model_heat_semantics_survive_light_dark_round_trip(
    chromium: Browser,
    request: pytest.FixtureRequest,
    report_fixture: str,
    model_case: str,
) -> None:
    report = request.getfixturevalue(report_fixture)
    with _open_report(
        chromium,
        report,
        color_scheme="light",
    ) as (page, browser_errors):
        if model_case == "glm_moe":
            _node(page, "Sparse DSA + MoE").dblclick()
            expect(_selected_view(page)).to_have_text(
                "GLM Sparse DSA + MoE representative"
            )
        elif model_case == "qwen_pressure":
            expect(page.locator("#heatmap-mode")).to_have_value("pressure")
            expect(page.locator("#model-dag .llm-dag-heat-title")).to_have_text(
                "Theoretical pressure"
            )
        else:
            expect(_selected_view(page)).to_have_text("Config-only model skeleton")

        state_before = _theme_interaction_snapshot(page)
        semantics_before = _heat_semantics_snapshot(page)
        palette_before = _heat_palette_snapshot(page)
        node_heat = semantics_before["nodeHeat"]
        assert isinstance(node_heat, list)
        known_count = sum(item[1] == "true" for item in node_heat)
        if model_case == "qwen_pressure":
            assert semantics_before["mode"] == "pressure"
            assert semantics_before["pressureDisabled"] is False
            assert known_count > 0
        else:
            assert semantics_before["pressureDisabled"] is True
            assert known_count == 0

        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.wait_for_function(
            """before => {
                const node = document.querySelector(
                    `#model-dag .llm-dag-node[data-node-id="${before.nodeId}"]`
                );
                return getComputedStyle(
                    node.querySelector('.llm-dag-node-card')
                ).fill !== before.nodeFill;
            }""",
            arg=palette_before,
        )
        state_dark = _theme_interaction_snapshot(page)
        semantics_dark = _heat_semantics_snapshot(page)
        palette_dark = _heat_palette_snapshot(page)
        assert state_dark == state_before
        assert semantics_dark == semantics_before
        assert palette_dark["tokens"] != palette_before["tokens"]
        assert palette_dark["nodeFill"] != palette_before["nodeFill"]
        assert palette_dark["minimapFill"] != palette_before["minimapFill"]

        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        page.wait_for_function(
            """before => {
                const node = document.querySelector(
                    `#model-dag .llm-dag-node[data-node-id="${before.nodeId}"]`
                );
                return getComputedStyle(
                    node.querySelector('.llm-dag-node-card')
                ).fill === before.nodeFill;
            }""",
            arg=palette_before,
        )
        assert _theme_interaction_snapshot(page) == state_before
        assert _heat_semantics_snapshot(page) == semantics_before
        assert _heat_palette_snapshot(page) == palette_before
        _assert_browser_clean(browser_errors)


def test_qwen_key_controls_fit_without_horizontal_overflow_at_700px(
    chromium: Browser,
    qwen_report: Path,
) -> None:
    with _open_report(
        chromium,
        qwen_report,
        width=700,
        height=900,
        color_scheme="dark",
    ) as (
        page,
        browser_errors,
    ):
        for control in (
            page.locator("#scenario-select"),
            page.locator("#search"),
            page.locator("#theme-toggle"),
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

        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
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

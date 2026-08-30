from __future__ import annotations

from llm_vis.ir import Definition, Metric, Model, ModelMap, Scenario
from llm_vis.report import workload_diff_summaries


def _scenario(phase: str, *, new_tokens: int, past_tokens: int) -> Scenario:
    return Scenario(
        phase=phase,
        batch=1,
        new_tokens=new_tokens,
        past_tokens=past_tokens,
        activation_dtype="bfloat16",
        weight_format="bfloat16",
        kv_dtype="bfloat16",
    )


def test_workload_diff_summaries_use_first_scenario_as_same_artifact_baseline() -> None:
    prefill = _scenario("prefill", new_tokens=8, past_tokens=0)
    decode = _scenario("decode", new_tokens=1, past_tokens=8)
    model_map = ModelMap(
        model=Model(
            id="model.tiny",
            name="Tiny",
            family="tiny",
            revision="r1",
            framework="generic",
        ),
        definitions=[Definition(id="subject.block", kind="block", label="Block")],
        scenarios=[prefill, decode],
        metrics=[
            Metric(
                id="metric.prefill",
                subject_id="subject.block",
                scenario_id=prefill.id,
                name="flops",
                value=100,
                unit="FLOPs",
                origin="formula",
            ),
            Metric(
                id="metric.decode",
                subject_id="subject.block",
                scenario_id=decode.id,
                name="flops",
                value=25,
                unit="FLOPs",
                origin="formula",
            ),
        ],
    )

    summaries = workload_diff_summaries(model_map)

    assert len(summaries) == 1
    assert summaries[0]["left_scenario_id"] == prefill.id
    assert summaries[0]["right_scenario_id"] == decode.id
    assert summaries[0]["entries"][0]["value_delta"] == -75.0


def test_workload_diff_summaries_require_two_scenarios() -> None:
    model_map = ModelMap(
        model=Model(
            id="model.tiny",
            name="Tiny",
            family="tiny",
            revision="r1",
            framework="generic",
        )
    )

    assert workload_diff_summaries(model_map) == ()

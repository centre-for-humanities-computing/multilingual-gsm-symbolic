from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "paper" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from numbercoverage import number_coverage_grid  # noqa: E402
from plot_config import language_order, model_sort_key, ordered_models  # noqa: E402
from qwen_compute_budget import (  # noqa: E402
    plot_qwen_compute_budget_family_transfers,
    plot_qwen_compute_budget_transfer,
    qwen_compute_budget_table,
)
from visualizegrid import (  # noqa: E402
    filter_summary_models,
    infer_model_info,
    model_order,
    plot_reasoning_delta,
)


def test_ordered_models_groups_known_families_by_size_and_keeps_unknowns() -> None:
    models = [
        "custom/Unknown-2B",
        "google/gemma-3-4b-it",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen3-4B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        "utter-project/EuroLLM-1.7B-Instruct",
    ]

    assert ordered_models(models) == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen3-4B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-3-4b-it",
        "allenai/OLMo-2-1124-7B-Instruct",
        "utter-project/EuroLLM-1.7B-Instruct",
        "custom/Unknown-2B",
    ]


def test_model_sort_key_places_distill_models_after_their_base_families() -> None:
    ordered = sorted(
        [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "Qwen/Qwen2.5-32B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        key=model_sort_key,
    )

    assert ordered == [
        "Qwen/Qwen2.5-32B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ]


def test_model_sort_key_groups_reasoning_off_before_reasoning_on_within_family() -> None:
    models = [
        "Qwen/Qwen3-4B",
        "google/gemma-3-1b-it",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B (reasoning off)",
        "Qwen/Qwen3-0.6B (reasoning off)",
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    assert ordered_models(models) == [
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen3-0.6B (reasoning off)",
        "Qwen/Qwen3-4B (reasoning off)",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
        "google/gemma-3-1b-it",
    ]


def test_language_order_retains_all_languages_with_known_first() -> None:
    assert language_order(["xxx", "eng", "eng_metric", "zho", "dan", "fra"]) == [
        "zho",
        "eng",
        "eng_metric",
        "fra",
        "dan",
        "xxx",
    ]


def test_number_coverage_grid_retains_supported_metric_language() -> None:
    models, languages, rates, samples = number_coverage_grid(
        [
            {"model": "provider/model-1B", "language": "eng", "all_prompt_numbers_present": True},
            {"model": "provider/model-1B", "language": "eng_metric", "all_prompt_numbers_present": False},
            {"model": "provider/model-1B", "language": "invalid_suffix", "all_prompt_numbers_present": False},
        ]
    )

    assert models == ["provider/model-1B"]
    assert languages == ["eng", "eng_metric"]
    assert rates.tolist() == [[1.0, 0.0]]
    assert samples.tolist() == [[1, 1]]


def test_infer_model_info_supports_new_ucloudeval_model_families() -> None:
    examples = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": ("DeepSeek-R1-Distill-Qwen", 1.5),
        "bigscience/bloomz-560m": ("BLOOMZ", 0.56),
        "bigscience/bloomz-1b1": ("BLOOMZ", 1.1),
        "bigscience/bloomz-7b1": ("BLOOMZ", 7.1),
        "EleutherAI/pythia-2.8b": ("Pythia", 2.8),
        "swiss-ai/Apertus-70B-Instruct-2509": ("Apertus", 70.0),
    }

    for model, (family, params_b) in examples.items():
        info = infer_model_info(model)
        assert info.family == family
        assert info.params_b == params_b


def test_model_order_only_uses_models_present_in_summary() -> None:
    import pandas as pd

    summary = pd.DataFrame(
        [
            {"model": "bloomz-560m", "language": "eng", "split": "synthetic", "accuracy": 0.2},
            {"model": "pythia-1b", "language": "eng", "split": "synthetic", "accuracy": 0.3},
        ]
    )

    assert model_order(summary) == ["bloomz-560m", "pythia-1b"]


def test_filter_summary_models_removes_sparse_qwen_reasoning_on_row() -> None:
    import pandas as pd

    summary = pd.DataFrame(
        [
            {"model": "Qwen3-0.6B", "language": "eng", "split": "original", "accuracy": 0.7},
            {"model": "Qwen3-0.6B (reasoning off)", "language": "eng", "split": "original", "accuracy": 0.6},
            {"model": "Qwen3-0.6B (reasoning on)", "language": "dan", "split": "original", "accuracy": 0.8},
        ]
    )

    filtered = filter_summary_models(summary)

    assert filtered["model"].tolist() == ["Qwen3-0.6B", "Qwen3-0.6B (reasoning off)"]


def test_plot_reasoning_delta_uses_off_model_name_for_on_variant(tmp_path) -> None:
    import pandas as pd

    summary = pd.DataFrame(
        [
            {
                "model_raw": "qwen3-0.6b",
                "model": "qwen3-0.6b (reasoning off)",
                "family": "Qwen3",
                "params_b": 0.6,
                "vocab_size": None,
                "language": "eng",
                "split": "synthetic",
                "accuracy": 0.55,
            },
            {
                "model_raw": "qwen3-0.6b",
                "model": "qwen3-0.6b (reasoning off)",
                "family": "Qwen3",
                "params_b": 0.6,
                "vocab_size": None,
                "language": "dan",
                "split": "synthetic",
                "accuracy": 0.45,
            },
            {
                "model_raw": "qwen3-0.6b",
                "model": "qwen3-0.6b",
                "family": "Qwen3",
                "params_b": 0.6,
                "vocab_size": None,
                "language": "eng",
                "split": "synthetic",
                "accuracy": 0.65,
            },
            {
                "model_raw": "qwen3-0.6b",
                "model": "qwen3-0.6b",
                "family": "Qwen3",
                "params_b": 0.6,
                "vocab_size": None,
                "language": "dan",
                "split": "synthetic",
                "accuracy": 0.50,
            },
        ]
    )
    out_path = tmp_path / "reasoning_delta.png"

    assert plot_reasoning_delta(summary, out_path)
    assert out_path.exists()


def test_plot_reasoning_delta_excludes_eng_metric_from_transfer_gap(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from matplotlib.axes import Axes

    plotted_y: list[list[float]] = []
    original_errorbar = Axes.errorbar

    def capture_errorbar(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        plotted_y.append(list(y))
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture_errorbar)
    rows = []
    for model, english_accuracy in [
        ("qwen3-0.6b", 1.0),
        ("qwen3-0.6b (reasoning off)", 1.0),
    ]:
        for language, accuracy in [
            ("eng", english_accuracy),
            ("eng_metric", 0.0),
            ("dan", 0.5),
        ]:
            rows.append(
                {
                    "model_raw": "qwen3-0.6b",
                    "model": model,
                    "family": "Qwen3",
                    "params_b": 0.6,
                    "vocab_size": None,
                    "language": language,
                    "split": "synthetic",
                    "accuracy": accuracy,
                }
            )

    assert plot_reasoning_delta(pd.DataFrame(rows), tmp_path / "reasoning_delta.png")
    assert plotted_y == [[0.5], [0.5]]


def test_qwen_compute_budget_table_marks_paired_qwen_variants_as_on_and_off() -> None:
    import pandas as pd

    rows = []
    for model_raw, model, family, params_b in [
        ("qwen3-4b", "qwen3-4b", "Qwen3", 4.0),
        ("qwen3-4b", "qwen3-4b (reasoning off)", "Qwen3", 4.0),
        ("qwen2.5-7b-instruct", "qwen2.5-7b-instruct", "Qwen2.5", 7.0),
        ("qwen3.5-27b", "qwen3.5-27b", "Qwen3.5", 27.0),
    ]:
        for language, accuracy in [("eng", 0.8), ("dan", 0.6)]:
            rows.append(
                {
                    "model_raw": model_raw,
                    "model": model,
                    "family": family,
                    "params_b": params_b,
                    "vocab_size": None,
                    "language": language,
                    "split": "synthetic",
                    "accuracy": accuracy,
                    "avg_generation_seconds": 5.0,
                }
            )

    table = qwen_compute_budget_table(pd.DataFrame(rows))

    qwen3 = table[table["model_raw"] == "qwen3-4b"]

    assert set(qwen3["model"]) == {"qwen3-4b", "qwen3-4b (reasoning off)"}
    assert set(qwen3["reasoning"]) == {"on", "off"}
    assert set(table["model_raw"]) == {"qwen3-4b"}


def test_qwen_compute_budget_table_includes_other_families_only_when_reasoning_is_paired() -> None:
    import pandas as pd

    rows = []
    for model_raw, model, family, params_b in [
        ("qwen3-4b", "qwen3-4b", "Qwen3", 4.0),
        ("qwen3-4b", "qwen3-4b (reasoning off)", "Qwen3", 4.0),
        ("ibm/granite-3.2-8b-instruct", "granite-3.2-8b-instruct (reasoning on)", "Granite", 8.0),
        ("ibm/granite-3.2-8b-instruct", "granite-3.2-8b-instruct (reasoning off)", "Granite", 8.0),
        ("meta-llama/llama-3.2-3b-instruct", "llama-3.2-3b-instruct", "Llama 3", 3.0),
    ]:
        for language, accuracy in [("eng", 0.8), ("dan", 0.6)]:
            rows.append(
                {
                    "model_raw": model_raw,
                    "model": model,
                    "family": family,
                    "params_b": params_b,
                    "vocab_size": None,
                    "language": language,
                    "split": "synthetic",
                    "accuracy": accuracy,
                    "avg_generation_seconds": 5.0,
                }
            )

    table = qwen_compute_budget_table(pd.DataFrame(rows))

    assert set(table["family"]) == {"Qwen3", "Granite"}
    assert set(table["reasoning"]) == {"on", "off"}


def test_qwen_compute_budget_exports_one_png_per_family_folder(tmp_path) -> None:
    import pandas as pd

    rows = []
    for model_raw, family, params_b, seconds_offset in [
        ("qwen3-4b", "Qwen3", 4.0, 0.0),
        ("ibm/granite-3.2-8b-instruct", "Granite", 8.0, 1.0),
    ]:
        base = model_raw.rsplit("/", 1)[-1]
        for reasoning, gap, seconds in [
            ("off", 0.24, params_b + seconds_offset),
            ("on", 0.12, params_b + seconds_offset + 0.5),
        ]:
            suffix = "" if family == "Qwen3" and reasoning == "on" else f" (reasoning {reasoning})"
            model = f"{base}{suffix}"
            for language, accuracy in [("eng", 0.8), ("dan", 0.8 * (1 - gap))]:
                rows.append(
                    {
                        "model_raw": model_raw,
                        "model": model,
                        "family": family,
                        "params_b": params_b,
                        "vocab_size": None,
                        "language": language,
                        "split": "synthetic",
                        "accuracy": accuracy,
                        "avg_generation_seconds": seconds,
                    }
                )

    outputs = plot_qwen_compute_budget_family_transfers(
        pd.DataFrame(rows),
        tmp_path,
    )

    assert sorted(path.relative_to(tmp_path).as_posix() for path in outputs) == [
        "qwen_compute_budget_transfer/granite/qwen_compute_budget_transfer.png",
        "qwen_compute_budget_transfer/qwen3/qwen_compute_budget_transfer.png",
    ]
    assert all(path.exists() for path in outputs)


def test_qwen_compute_budget_plot_colors_reasoning_modes_and_draws_one_frontier(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from matplotlib.axes import Axes

    scatter_colors: dict[str, str] = {}
    step_labels: list[str] = []
    original_scatter = Axes.scatter
    original_step = Axes.step

    def capture_scatter(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        scatter_colors[kwargs["label"]] = kwargs["color"]
        return original_scatter(self, x, y, *args, **kwargs)

    def capture_step(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        step_labels.append(kwargs["label"])
        return original_step(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Axes, "step", capture_step)
    rows = []
    for params_b, base_accuracy, off_gap, on_gap in [(4.0, 0.8, 0.25, 0.1), (8.0, 0.9, 0.2, 0.08)]:
        for reasoning, gap, seconds in [("off", off_gap, params_b), ("on", on_gap, params_b * 3)]:
            model = f"qwen3-{params_b:g}b" + ("" if reasoning == "on" else " (reasoning off)")
            for language, accuracy in [("eng", base_accuracy), ("dan", base_accuracy * (1 - gap))]:
                rows.append(
                    {
                        "model_raw": f"qwen3-{params_b:g}b",
                        "model": model,
                        "family": "Qwen3",
                        "params_b": params_b,
                        "vocab_size": None,
                        "language": language,
                        "split": "synthetic",
                        "accuracy": accuracy,
                        "avg_generation_seconds": seconds,
                    }
                )

    assert plot_qwen_compute_budget_transfer(
        pd.DataFrame(rows),
        tmp_path / "qwen_budget.png",
    )

    assert scatter_colors["reasoning off"] != scatter_colors["reasoning on"]
    assert step_labels == ["Pareto frontier"]


def test_qwen_compute_budget_plot_draws_one_global_pareto_frontier(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from matplotlib.axes import Axes

    step_calls: list[tuple[list[float], list[float], str]] = []
    original_step = Axes.step

    def capture_step(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        step_calls.append((list(x), list(y), kwargs["label"]))
        return original_step(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "step", capture_step)
    rows = []
    for family, sizes, seconds_offset, gaps_by_reasoning in [
        ("Qwen3", [4.0, 8.0], 0.0, {"off": [0.32, 0.21], "on": [0.14, 0.08]}),
        ("Qwen3.5", [9.0, 27.0], 10.0, {"off": [0.28, 0.18], "on": [0.11, 0.06]}),
    ]:
        for index, params_b in enumerate(sizes):
            for reasoning, gaps in gaps_by_reasoning.items():
                gap = gaps[index]
                model = f"qwen{family.replace('.', '')}-{params_b:g}b" + ("" if reasoning == "on" else " (reasoning off)")
                for language, accuracy in [("eng", 0.9), ("dan", 0.9 * (1 - gap))]:
                    rows.append(
                        {
                            "model_raw": model.replace(" (reasoning off)", ""),
                            "model": model,
                            "family": family,
                            "params_b": params_b,
                            "vocab_size": None,
                            "language": language,
                            "split": "synthetic",
                            "accuracy": accuracy,
                            "avg_generation_seconds": params_b
                            + seconds_offset
                            + (0.2 if reasoning == "on" else 0.0),
                        }
                    )

    assert plot_qwen_compute_budget_transfer(
        pd.DataFrame(rows),
        tmp_path / "qwen_budget.png",
    )

    assert len(step_calls) == 1
    x, y, label = step_calls[0]
    assert label == "Pareto frontier"
    assert len(x) == len(y)
    assert y == sorted(y, reverse=True)

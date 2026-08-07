from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "paper" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_log_utils import map_log_loader  # noqa: E402
from number_coverage_utils import extract_numbers, number_coverage_counts  # noqa: E402
from numbercoverage import number_coverage_grid, summarize_group  # noqa: E402
from plot_config import (  # noqa: E402
    HUMAN_VERIFIED_LANGUAGES,
    LANGUAGE_COLORS,
    LANGUAGE_LABELS,
    LANGUAGE_ORDER,
    LANGUAGE_SPEAKERS,
    language_order,
    model_sort_key,
    ordered_models,
)
from qwen_compute_budget import (  # noqa: E402
    plot_qwen_compute_budget_family_transfers,
    plot_qwen_compute_budget_relative_transfer,
    plot_qwen_compute_budget_transfer,
    qwen_compute_budget_table,
)
from visualizegrid import (  # noqa: E402
    filter_summary_models,
    model_order,
)


def _test_log_loader(path: Path, scorer: str | None):
    return path.name, None, scorer


def test_map_log_loader_runs_serially_and_in_parallel() -> None:
    paths = [Path("one.eval"), Path("two.eval")]
    expected = [(path.name, None, "math") for path in paths]
    assert list(map_log_loader(_test_log_loader, paths, "math", workers=1)) == expected
    assert sorted(map_log_loader(_test_log_loader, paths, "math", workers=2)) == expected


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


def test_new_languages_have_complete_paper_metadata() -> None:
    new_languages = {"mar", "hin", "ara", "nld", "est", "jpn"}

    assert new_languages <= HUMAN_VERIFIED_LANGUAGES
    assert new_languages <= LANGUAGE_LABELS.keys()
    assert new_languages <= LANGUAGE_SPEAKERS.keys()
    assert new_languages <= LANGUAGE_COLORS.keys()
    assert new_languages <= LANGUAGE_ORDER.keys()


def test_language_order_includes_new_languages() -> None:
    assert language_order(["nld", "jpn", "mar", "hin", "ara", "est"]) == [
        "hin",
        "ara",
        "jpn",
        "mar",
        "nld",
        "est",
    ]


def test_number_coverage_grid_retains_supported_metric_language() -> None:
    models, languages, rates, samples = number_coverage_grid(
        [
            {"model": "provider/model-1B", "language": "eng", "all_prompt_numbers_present": True},
            {"model": "provider/model-1B", "language": "eng", "all_prompt_numbers_present": True, "prompt_number_count": 0},
            {"model": "provider/model-1B", "language": "eng_metric", "all_prompt_numbers_present": False},
            {"model": "provider/model-1B", "language": "invalid_suffix", "all_prompt_numbers_present": False},
        ]
    )

    assert models == ["provider/model-1B"]
    assert languages == ["eng", "eng_metric"]
    assert rates.tolist() == [[1.0, 0.0]]
    assert samples.tolist() == [[1, 1]]
    assert summarize_group(
        [
            {"all_prompt_numbers_present": True, "prompt_number_count": 0},
            {"all_prompt_numbers_present": False, "prompt_number_count": 1},
        ]
    )["all_prompt_numbers_present_rate"] == 0


def test_number_coverage_counts() -> None:
    assert number_coverage_counts("Use 10, 20, and 30.", "10 + 20 = 40") == {
        "all_prompt_numbers_present": False,
        "prompt_number_count": 3,
        "retrieved_prompt_number_count": 2,
        "lhs_count": 0,
        "lhs_retrieved": 0,
        "rhs_count": 0,
        "rhs_retrieved": 0,
    }


def test_number_coverage_counts_splits_chevron_sides() -> None:
    assert number_coverage_counts(
        "No numeric prompt values.",
        "The answer uses 2 and 4.",
        "Compute <<2+2=4>>, then <<4/2=2>>.",
    ) == {
        "all_prompt_numbers_present": False,
        "prompt_number_count": 0,
        "retrieved_prompt_number_count": 0,
        "lhs_count": 1,
        "lhs_retrieved": 1,
        "rhs_count": 2,
        "rhs_retrieved": 2,
    }


def test_extract_numbers_equates_digit_and_fraction_forms() -> None:
    assert extract_numbers("5 and 5.0") == {Decimal("5")}
    assert extract_numbers("1/2, 0.5, and half") == {Decimal("0.5")}
    assert extract_numbers("10kg and １２ items") == {Decimal("10"), Decimal("12")}
    assert extract_numbers("100匹の4分の3と２倍", "jpn") == {Decimal("100"), Decimal("0.75"), Decimal("2")}
    assert extract_numbers("4分之3", "zho") == {Decimal("0.75")}
    assert extract_numbers("0,25 and 2.700.", "dan") == {Decimal("0.25"), Decimal("2700")}
    assert extract_numbers("१२ and ١٢") == {Decimal("12")}
    assert extract_numbers("five, half, and twenty-five") == set()


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
                    "avg_total_tokens": 100.0,
                }
            )

    table = qwen_compute_budget_table(pd.DataFrame(rows))

    qwen3 = table[table["model_raw"] == "qwen3-4b"]

    assert set(qwen3["model"]) == {"qwen3-4b", "qwen3-4b (reasoning off)"}
    assert set(qwen3["reasoning"]) == {"on", "off"}
    assert set(qwen3["inference_flops"]) == {8e11}
    assert set(qwen3["absolute_transfer_gap"].round(3)) == {0.2}
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
                    "avg_total_tokens": 100.0,
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
                        "avg_total_tokens": seconds * 100,
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


def test_qwen_compute_budget_plot_connects_families_and_styles_reasoning_modes(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from matplotlib.axes import Axes

    line_calls: list[tuple[str, str]] = []
    step_calls: list[str] = []
    original_plot = Axes.plot
    original_step = Axes.step

    def capture_plot(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        line_calls.append((kwargs["color"], kwargs["linestyle"]))
        return original_plot(self, x, y, *args, **kwargs)

    def capture_step(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        step_calls.append(kwargs.get("label", ""))
        return original_step(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
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
                        "avg_total_tokens": seconds * 100,
                    }
                )

    assert plot_qwen_compute_budget_transfer(
        pd.DataFrame(rows),
        tmp_path / "qwen_budget.png",
    )

    assert len(line_calls) == 2
    assert {linestyle for _, linestyle in line_calls} == {":", "-"}
    assert len({color for color, _ in line_calls}) == 2
    assert step_calls == []


def test_qwen_compute_budget_plot_assigns_each_family_its_own_color(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from matplotlib.axes import Axes

    line_calls: list[tuple[str, str]] = []
    original_plot = Axes.plot

    def capture_plot(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        line_calls.append((kwargs["color"], kwargs["linestyle"]))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
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
                            "avg_total_tokens": (params_b
                            + seconds_offset
                            + (0.2 if reasoning == "on" else 0.0)) * 100,
                        }
                    )

    assert plot_qwen_compute_budget_transfer(
        pd.DataFrame(rows),
        tmp_path / "qwen_budget.png",
    )

    solid_colors = {color for color, linestyle in line_calls if linestyle == "-"}
    dotted_colors = {color for color, linestyle in line_calls if linestyle == ":"}
    assert len(solid_colors) == 2
    assert len(dotted_colors) == 2
    assert solid_colors.isdisjoint(dotted_colors)


def test_qwen_compute_budget_relative_plot_writes_separate_png(tmp_path) -> None:
    import pandas as pd

    rows = []
    for reasoning, tokens, gap in [("off", 100.0, 0.2), ("on", 300.0, 0.1)]:
        model = "qwen3-4b" + ("" if reasoning == "on" else " (reasoning off)")
        for language, accuracy in [("eng", 0.8), ("dan", 0.8 * (1 - gap))]:
            rows.append(
                {
                    "model_raw": "qwen3-4b",
                    "model": model,
                    "family": "Qwen3",
                    "params_b": 4.0,
                    "language": language,
                    "split": "synthetic",
                    "accuracy": accuracy,
                    "sample_correct": {str(i): float(i < round(accuracy * 100)) for i in range(100)},
                    "avg_total_tokens": tokens,
                }
            )

    out = tmp_path / "relative.png"
    assert plot_qwen_compute_budget_relative_transfer(pd.DataFrame(rows), out)
    assert out.exists()

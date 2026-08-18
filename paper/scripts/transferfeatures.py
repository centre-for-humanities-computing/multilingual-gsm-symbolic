# /// script
# dependencies = [
#   "lang2vec",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "scipy",
#   "setuptools<81",
#   "transformers",
# ]
# ///
"""Collect language features and relate them to cross-lingual transfer.

The script combines ``visualize_model_grid.py``'s run summary with:

* tokenizer fertility: tokens per non-whitespace Unicode character, normalized
  to English on matched source problems for each model tokenizer;
* typological distance: cosine distance from English over concatenated
  URIEL/lang2vec ``syntax_knn`` and ``inventory_knn`` vectors;
* resource quantity: Common Crawl page count, plotted on a base-10 log scale.

Outputs include the collected feature tables, the joined transfer-analysis
table, provenance metadata, and one relationship plot per feature.

Example:
    uv run paper/scripts/transferfeatures.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import lang2vec.lang2vec as l2v
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from plot_config import FAMILY_COLORS, LANGUAGE_LABELS, PLOT_STYLE, language_order, ordered_families
from scipy.spatial import distance
from transformers import AutoTokenizer

SCRIPT_MARKERS = {
    "Latin": "o",
    "Cyrillic": "s",
    "CJK": "^",
    "Devanagari": "D",
    "Arabic": "v",
    "Thai": "p",
    "Hangul": "h",
}

LANGUAGE_SCRIPTS = {
    "eng": "Latin",
    "dan": "Latin",
    "deu": "Latin",
    "isl": "Latin",
    "uncorrected_isl": "Latin",
    "nob": "Latin",
    "rus": "Cyrillic",
    "zho": "CJK",
    "jpn": "CJK",
    "kor": "Hangul",
    "mar": "Devanagari",
    "hin": "Devanagari",
    "ara": "Arabic",
    "nld": "Latin",
    "est": "Latin",
    "tha": "Thai",
}

def relationship_plot(
    data: pd.DataFrame,
    x_column: str,
    xlabel: str,
    out: Path,
    use_script_shapes: bool = False,
    footnote: str | None = None,
    label_languages: bool = True,
) -> list[Path]:
    """Plot a descriptive feature relationship for the synthetic split only."""
    plot_data = data[
        (data["language"] != "eng_metric") & (data["family"] != "OpenAI")
    ].dropna(
        subset=[x_column, "performance_recovered"]
    )
    if plot_data.empty:
        return []

    # Synthetic split only as requested
    panel = plot_data[plot_data["split"] == "synthetic"]
    if panel.empty:
        return []

    fig, ax = plt.subplots(figsize=(6.5, 5.3 if use_script_shapes else 5))

    for family in ordered_families(panel["family"]):
        family_rows = panel[panel["family"] == family]
        if use_script_shapes:
            for row in family_rows.itertuples():
                script = LANGUAGE_SCRIPTS.get(row.language, "Latin")
                marker = SCRIPT_MARKERS.get(script, "o")
                ax.scatter(
                    getattr(row, x_column),
                    row.performance_recovered,
                    marker=marker,
                    s=42,
                    alpha=0.55,
                    color=FAMILY_COLORS.get(family, "#666666"),
                    zorder=3,
                )
        else:
            ax.scatter(
                family_rows[x_column],
                family_rows["performance_recovered"],
                marker="o",
                s=42,
                alpha=0.55,
                color=FAMILY_COLORS.get(family, "#666666"),
                label=family,
                zorder=3,
            )

    # Use a restrained neutral trendline so it does not compete with model colors.
    unique_x = panel[x_column].nunique()
    slope: float | None = None
    intercept: float | None = None
    if len(panel) >= 3 and unique_x >= 2:
        slope, intercept = np.polyfit(panel[x_column], panel["performance_recovered"], 1)
        x_line = np.linspace(panel[x_column].min(), panel[x_column].max(), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#000000",
            linestyle="-",
            linewidth=1.6,
            alpha=0.85,
            zorder=4,
            label="Trendline",
        )

    if label_languages:
        label_positions = (
            panel.groupby("language", as_index=False)
            .agg(x=(x_column, "mean"))
            .sort_values("x")
        )
        x_span = max(float(panel[x_column].max() - panel[x_column].min()), 1e-9)
        lane_last_x: list[float] = []
        for row in label_positions.itertuples(index=False):
            lane = next(
                (
                    index
                    for index, previous_x in enumerate(lane_last_x)
                    if row.x - previous_x >= 0.12 * x_span
                ),
                len(lane_last_x),
            )
            if lane == len(lane_last_x):
                lane_last_x.append(row.x)
            else:
                lane_last_x[lane] = row.x
            ax.annotate(
                LANGUAGE_LABELS.get(row.language, row.language),
                xy=(row.x, 1),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0, 6 + 14 * lane),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#1F2937",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
                annotation_clip=False,
                zorder=6,
            )

    ax.axhline(1, color="black", linewidth=0.8, alpha=0.4, zorder=2)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=16 if use_script_shapes else 8)
    ax.set_ylabel("English performance recovered", fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    if use_script_shapes:
        family_handles = [
            Line2D(
                [0],
                [0],
                color=FAMILY_COLORS.get(f, "#666666"),
                marker="o",
                linestyle="None",
                markersize=6,
                alpha=0.55,
                label=f,
            )
            for f in ordered_families(panel["family"])
        ]
        present_scripts = sorted(set(LANGUAGE_SCRIPTS.get(l, "Latin") for l in panel["language"].unique()))
        script_handles = [
            Line2D(
                [0],
                [0],
                color="#333333",
                marker=SCRIPT_MARKERS.get(s, "o"),
                linestyle="None",
                markersize=6,
                alpha=0.55,
                label=f"Script: {s}",
            )
            for s in present_scripts
        ]
        trend_handle = [Line2D([0], [0], color="#000000", lw=1.6, alpha=0.85, label="Trendline")]
        fig.legend(
            family_handles + trend_handle,
            [h.get_label() for h in family_handles + trend_handle],
            title="Model family",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.115),
            ncol=3,
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )
        fig.legend(
            script_handles,
            [h.get_label().removeprefix("Script: ") for h in script_handles],
            title="Language script",
            loc="upper right",
            bbox_to_anchor=(0.98, 0.115),
            ncol=2,
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="Model family",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.13),
                ncol=math.ceil(len(labels) / 2),
                frameon=False,
                fontsize=8,
                title_fontsize=8,
            )

    bottom_margin = 0.16 if use_script_shapes else (0.17 if footnote else 0.14)
    if footnote:
        footnote_y = 0.145 if use_script_shapes else 0.16
        fig.text(0.5, footnote_y, footnote, ha="center", fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return [out]

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "paper" / "artifacts" / "figures"
DEFAULT_OUT_DIR = ARTIFACTS_DIR / "transfer_features"

COMMON_CRAWL_LANGUAGE_CODES = {"nob": "nor"}
DEFAULT_COMMON_CRAWL_CSV = ARTIFACTS_DIR / "transfer_features" / "languages.csv"
SOURCE_METADATA = {
    "definitions": {
        "transfer_gap": "English accuracy minus target-language accuracy for the same model and split",
        "performance_recovered": "Target-language accuracy divided by English accuracy for the same model and split",
        "tokenizer_fertility": "Tokenizer tokens divided by non-whitespace Unicode characters, with no special tokens",
        "normalized_fertility": "Target-language fertility divided by English fertility on matched source_id questions",
        "typological_distance": "Cosine distance from English over concatenated URIEL/lang2vec syntax_knn and inventory_knn vectors",
        "resource_quantity": "Common Crawl page count; plots use log10(page count)",
    },
    "sources": {
        "tokenizers": "https://huggingface.co/docs/transformers/en/model_doc/auto",
        "uriel_lang2vec": "https://github.com/antonisa/lang2vec",
        "uriel_paper": "https://aclanthology.org/E17-2002/",
        "common_crawl_language_counts": str(DEFAULT_COMMON_CRAWL_CSV),
    },
}

plt.rcParams.update(PLOT_STYLE)


def load_common_crawl_pages(source: str | Path, languages: list[str]) -> pd.DataFrame:
    """Load the latest Common Crawl page count for each available language."""
    frame = pd.read_csv(source)
    required = {"crawl", "primary_language", "pages"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")
    rows = []
    for language in languages:
        common_crawl_language = COMMON_CRAWL_LANGUAGE_CODES.get(language, language)
        matches = frame.loc[frame["primary_language"] == common_crawl_language].sort_values("crawl")
        if matches.empty:
            continue
        resource = matches.iloc[-1]
        rows.append(
            {
                "language": language,
                "common_crawl_language": common_crawl_language,
                "common_crawl_crawl": resource["crawl"],
                "common_crawl_pages": int(resource["pages"]),
                "resource_source_path": str(source),
            }
        )
    return pd.DataFrame(rows)


def collect_language_features(
    languages: list[str],
    common_crawl_resources: pd.DataFrame,
) -> pd.DataFrame:
    """Collect URIEL distance and Common Crawl page counts per language."""
    vectors = l2v.get_features(sorted(set(languages) | {"eng"}), "syntax_knn")
    english = np.asarray(vectors["eng"], dtype=float)
    resources = common_crawl_resources.set_index("language").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    for language in languages:
        resource = resources[language]
        pages = int(resource["common_crawl_pages"])
        rows.append(
            {
                "language": language,
                "language_name": LANGUAGE_LABELS.get(language, language),
                "typological_distance_from_english": float(
                    distance.cosine(
                        np.asarray(vectors[language], dtype=float),
                        english,
                    )
                ),
                "typological_feature_set": "URIEL syntax_knn",
                **resource,
                "log10_common_crawl_pages": math.log10(pages) if pages > 0 else math.nan,
            }
        )

    return pd.DataFrame(rows)


def load_questions(data_dir: Path, languages: list[str]) -> dict[str, pd.DataFrame]:
    """Load original benchmark questions, indexed by matched source id."""
    questions: dict[str, pd.DataFrame] = {}
    for language in languages:
        paths = sorted((data_dir / language).glob("test_original-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No test_original parquet files found for {language} in {data_dir}")
        frame = pd.concat(pd.read_parquet(path, columns=["question", "source_id"]) for path in paths)
        frame = frame.drop_duplicates("source_id").set_index("source_id").sort_index()
        questions[language] = frame
    return questions


def tokenizer_repo(model_raw: str) -> str | None:
    raw = model_raw.rstrip("/")
    for prefix in ("vllm/", "hf/", "transformers/"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix) :]
            break
    if raw.lower().startswith("openai/"):
        return None
    # Gemma 3 checkpoints share one tokenizer. Use a single accessible/cached
    # checkpoint so artifact regeneration does not depend on every gated model
    # repository being individually authorized.
    if raw.lower() in {"google/gemma-3-1b-it", "google/gemma-3-27b-it"}:
        return "google/gemma-3-4b-it"
    return raw if "/" in raw else None


def model_repo(model: str) -> str | None:
    """Recover the public tokenizer repository from the canonical model label."""
    base = re.sub(r" \(reasoning (?:on|off)\)$", "", model, flags=re.IGNORECASE)
    lower = base.lower()
    if lower.startswith("qwen"):
        return f"Qwen/{base}"
    if lower.startswith("olmo"):
        return f"allenai/{base}"
    if lower.startswith("granite"):
        return f"ibm-granite/{base}"
    if lower.startswith("gemma"):
        return f"google/{base}"
    if lower.startswith("apertus"):
        return f"swiss-ai/{base}"
    if lower.startswith("eurollm"):
        return f"utter-project/{base}"
    if lower == "phi-4" or lower.startswith("phi-4-mini"):
        return f"microsoft/{base}"
    return None


def text_fertility(tokenizer: Any, texts: list[str]) -> tuple[float, int, int]:
    """Return corpus tokens/non-whitespace-character ratio and its totals."""
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_count = sum(len(input_ids) for input_ids in encoded["input_ids"])
    character_count = sum(sum(not character.isspace() for character in text) for text in texts)
    if character_count == 0:
        return math.nan, token_count, character_count
    return token_count / character_count, token_count, character_count


def collect_tokenizer_fertility(
    summary: pd.DataFrame,
    questions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate model-specific fertility on source-matched translated questions."""
    if "eng" not in questions:
        raise ValueError("English questions are required as the fertility reference")

    models = summary[["model", "family"]].drop_duplicates()
    models["model_raw"] = models["model"].map(model_repo)
    rows: list[dict[str, Any]] = []
    load_errors: list[str] = []

    for model_row in models.itertuples(index=False):
        repo = tokenizer_repo(model_row.model_raw)
        if repo is None:
            print(f"Skipping tokenizer fertility for {model_row.model}: no open tokenizer repository")
            continue

        print(f"Loading tokenizer for {model_row.model} from {repo}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo,
                trust_remote_code=False,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            load_errors.append(f"{model_row.model} ({repo}): {exc}")
            continue

        for language, language_questions in questions.items():
            common_ids = questions["eng"].index.intersection(language_questions.index)
            if common_ids.empty:
                print(f"Skipping {model_row.model}/{language}: no source ids match English")
                continue

            english_texts = questions["eng"].loc[common_ids, "question"].astype(str).tolist()
            language_texts = language_questions.loc[common_ids, "question"].astype(str).tolist()
            english_fertility, english_tokens, english_characters = text_fertility(
                tokenizer,
                english_texts,
            )
            fertility, token_count, character_count = text_fertility(tokenizer, language_texts)

            rows.append(
                {
                    "model_raw": model_row.model_raw,
                    "model": model_row.model,
                    "family": model_row.family,
                    "tokenizer_repo": repo,
                    "language": language,
                    "fertility_tokens_per_character": fertility,
                    "english_fertility_tokens_per_character": english_fertility,
                    "normalized_fertility": fertility / english_fertility,
                    "n_matched_questions": len(common_ids),
                    "token_count": token_count,
                    "non_whitespace_character_count": character_count,
                    "english_token_count": english_tokens,
                    "english_non_whitespace_character_count": english_characters,
                }
            )

    if load_errors:
        failures = "\n\n".join(load_errors)
        raise RuntimeError(
            "Failed to load required tokenizers; refusing to write incomplete fertility data. "
            "Set HF_TOKEN to a Hugging Face token with access to gated model repositories.\n\n"
            f"{failures}"
        )

    return pd.DataFrame(rows)


def build_transfer_table(
    summary: pd.DataFrame,
    language_features: pd.DataFrame,
    fertility: pd.DataFrame,
) -> pd.DataFrame:
    """Join English-relative accuracy penalties to collected language features."""
    index_columns = ["model_raw", "model", "family", "params_b", "language", "split"]
    values = summary[index_columns + ["accuracy", "stderr", "n_problems"]].copy()
    english = (
        values[values["language"] == "eng"]
        .drop(columns="language")
        .rename(
            columns={
                "accuracy": "english_accuracy",
                "stderr": "english_stderr",
                "n_problems": "english_n_problems",
            }
        )
    )
    transfer = values[
        ~values["language"].isin({"eng", "eng_metric"}) & (values["family"] != "OpenAI")
    ].merge(
        english,
        on=["model_raw", "model", "family", "params_b", "split"],
        how="inner",
        validate="many_to_one",
    )
    transfer["transfer_gap"] = transfer["english_accuracy"] - transfer["accuracy"]
    transfer["transfer_gap_stderr"] = np.sqrt(
        transfer["stderr"].fillna(0) ** 2 + transfer["english_stderr"].fillna(0) ** 2
    )
    english_accuracy = transfer["english_accuracy"].replace(0, np.nan)
    transfer["performance_recovered"] = transfer["accuracy"] / english_accuracy
    transfer["performance_recovered_stderr"] = np.sqrt(
        (transfer["accuracy"] / english_accuracy.pow(2) * transfer["english_stderr"].fillna(0)).pow(2)
        + (transfer["stderr"].fillna(0) / english_accuracy).pow(2)
    )

    transfer = transfer.merge(
        language_features,
        on="language",
        how="left",
        validate="many_to_one",
    )
    if not fertility.empty:
        transfer = transfer.merge(
            fertility.drop(columns=["family"]),
            on=["model_raw", "model", "language"],
            how="left",
            validate="many_to_one",
        )
    return transfer

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=REPO_ROOT / "paper" / "artifacts" / "transfer_tables" / "analysis.parquet",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "hf_dataset" / "data")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--common-crawl-csv",
        type=Path,
        default=DEFAULT_COMMON_CRAWL_CSV,
        help="Common Crawl language page-count CSV.",
    )
    args = parser.parse_args()

    samples = pd.read_parquet(args.analysis)
    samples = samples[samples["language"] != "uncorrected_isl"]
    group_cols = ["model", "family", "params_b", "vocab_size", "language", "split"]
    summary = samples.groupby(group_cols, dropna=False)["correct"].agg(accuracy="mean", n_problems="size", stderr="sem").reset_index()
    summary["model_raw"] = summary["model"].map(model_repo)
    if "eng" not in set(summary["language"]):
        raise SystemExit("English results are required to calculate recovered performance.")
    languages = language_order(summary["language"].unique())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    language_features_path = args.out_dir / "language_features.csv"
    common_crawl_resources = load_common_crawl_pages(args.common_crawl_csv, languages)
    feature_languages = common_crawl_resources["language"].tolist()

    print(f"Collecting language features for: {', '.join(feature_languages)} (Common Crawl)")
    language_features = collect_language_features(
        feature_languages,
        common_crawl_resources,
    )
    language_features.to_csv(language_features_path, index=False)

    questions = load_questions(args.data_dir, languages)
    fertility = collect_tokenizer_fertility(
        summary,
        questions,
    )
    fertility_path = args.out_dir / "tokenizer_fertility.csv"
    fertility.to_csv(fertility_path, index=False)

    transfer = build_transfer_table(summary, language_features, fertility)
    transfer_path = args.out_dir / "transfer_feature_data.csv"
    transfer.to_csv(transfer_path, index=False)

    metadata_path = args.out_dir / "feature_sources.json"
    metadata_path.write_text(
        json.dumps(SOURCE_METADATA, indent=2) + "\n",
        encoding="utf-8",
    )

    plots = [
        (
            "normalized_fertility",
            "Tokenizer fertility ratio",
            args.out_dir / "tokenizer_fertility_vs_transfer.png",
            True,
            "Ratio of target-language to English tokens per character, computed on GSM8K templates.",
            False,
        ),
        (
            "typological_distance_from_english",
            "Typological distance",
            args.out_dir / "typological_distance_vs_transfer.png",
            False,
            "Typological distance is cosine distance from English using URIEL syntax features.",
            True,
        ),
        (
            "log10_common_crawl_pages",
            "Language resources",
            args.out_dir / "resource_quantity_vs_transfer.png",
            False,
            "Language resources are measured as log10 Common Crawl page count.",
            True,
        ),
    ]
    for plot in plots:
        column, xlabel, path, use_script_shapes, footnote, label_languages = plot
        if saved_plots := relationship_plot(
            transfer,
            column,
            xlabel,
            path,
            use_script_shapes=use_script_shapes,
            footnote=footnote,
            label_languages=label_languages,
        ):
            for saved_plot in saved_plots:
                print(f"Saved {saved_plot}")
        else:
            print(f"Skipped {path.name}: no paired transfer observations with this feature.")

    print(f"Saved {language_features_path}")
    print(f"Saved {fertility_path}")
    print(f"Saved {transfer_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()

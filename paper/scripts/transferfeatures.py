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
from pathlib import Path
from typing import Any

import lang2vec.lang2vec as l2v
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from plot_config import FAMILY_COLORS, FAMILY_ORDER, LANGUAGE_LABELS, SPLIT_LABELS, language_order, ordered_families
from scipy.spatial import distance
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "paper" / "artifacts" / "figures"
DEFAULT_OUT_DIR = ARTIFACTS_DIR / "transfer_features"

COMMON_CRAWL_LANGUAGE_CODES = {"nob": "nor"}
DEFAULT_COMMON_CRAWL_CSV = ARTIFACTS_DIR / "transfer_features" / "languages.csv"
SOURCE_METADATA = {
    "definitions": {
        "transfer_gap": "English accuracy minus target-language accuracy for the same model and split",
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

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 160,
        "font.family": "sans-serif",
    }
)


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
            return raw[len(prefix) :]
    if raw.lower().startswith("openai/"):
        return None
    return raw if "/" in raw else None


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

    models = summary[["model_raw", "model", "family"]].drop_duplicates()
    rows: list[dict[str, Any]] = []

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
            )
        except Exception as exc:
            print(f"Skipping tokenizer fertility for {model_row.model}: {exc}")
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
    transfer = values[values["language"] != "eng"].merge(
        english,
        on=["model_raw", "model", "family", "params_b", "split"],
        how="inner",
        validate="many_to_one",
    )
    transfer["transfer_gap"] = transfer["english_accuracy"] - transfer["accuracy"]
    transfer["transfer_gap_stderr"] = np.sqrt(
        transfer["stderr"].fillna(0) ** 2 + transfer["english_stderr"].fillna(0) ** 2
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


def relationship_plot(
    data: pd.DataFrame,
    x_column: str,
    xlabel: str,
    title: str,
    out: Path,
) -> bool:
    """Plot a descriptive feature relationship separately for each split."""
    plot_data = data.dropna(subset=[x_column, "transfer_gap"])
    if plot_data.empty:
        return False

    splits = [split for split in ("original", "synthetic") if split in set(plot_data["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 5), squeeze=False)
    axes = axes[0]

    for ax, split in zip(axes, splits, strict=True):
        panel = plot_data[plot_data["split"] == split]
        for family in ordered_families(panel["family"]):
            family_rows = panel[panel["family"] == family]
            ax.errorbar(
                family_rows[x_column],
                family_rows["transfer_gap"],
                yerr=family_rows["transfer_gap_stderr"],
                fmt="o",
                markersize=6,
                capsize=2,
                alpha=0.8,
                color=FAMILY_COLORS.get(family, "#666666"),
                label=family,
            )

        label_positions = panel.groupby(["family", "language"], as_index=False).agg(
            x=(x_column, "mean"), y=("transfer_gap", "mean")
        )
        label_positions["family_order"] = label_positions["family"].map(FAMILY_ORDER).fillna(99)
        label_positions = label_positions.sort_values(["family_order", "language"])
        for row in label_positions.itertuples(index=False):
            ax.annotate(
                row.language,
                (row.x, row.y),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color=FAMILY_COLORS.get(row.family, "#666666"),
            )

        unique_x = panel[x_column].nunique()
        if len(panel) >= 3 and unique_x >= 2:
            slope, intercept = np.polyfit(panel[x_column], panel["transfer_gap"], 1)
            x_line = np.linspace(panel[x_column].min(), panel[x_column].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=1)
            correlation = panel[[x_column, "transfer_gap"]].corr().iloc[0, 1]
            ax.text(
                0.02,
                0.98,
                f"Descriptive Pearson r = {correlation:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.grid(alpha=0.2)
        ax.set_ylabel("Accuracy gap: English - target language")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(SPLIT_LABELS[split])

    legend_entries: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(dict(zip(labels, handles, strict=True)))
    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=len(legend_entries),
            frameon=False,
        )
    fig.supxlabel(xlabel, y=0.08)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ARTIFACTS_DIR / "model_grid" / "run_summary.csv",
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

    summary = pd.read_csv(args.summary)
    if "eng" not in set(summary["language"]):
        raise SystemExit("English results are required to calculate transfer gaps.")
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
            "TFR: target-language tokens/character divided by English tokens/character. Computed on GSM8K templates.",
            "English-minus-target-language accuracy gap versus tokenizer fertility ratio (TFR)",
            args.out_dir / "tokenizer_fertility_vs_transfer.png",
        ),
        (
            "typological_distance_from_english",
            "URIEL cosine distance from English (syntax features)",
            "English-minus-target-language accuracy gap versus typological distance from English",
            args.out_dir / "typological_distance_vs_transfer.png",
        ),
        (
            "log10_common_crawl_pages",
            "Language-resource proxy: log10 Common Crawl page count",
            "English-minus-target-language accuracy gap versus Common Crawl resource quantity",
            args.out_dir / "resource_quantity_vs_transfer.png",
        ),
    ]
    for column, xlabel, title, path in plots:
        if relationship_plot(transfer, column, xlabel, title, path):
            print(f"Saved {path}")
        else:
            print(f"Skipped {path.name}: no paired transfer observations with this feature.")

    print(f"Saved {language_features_path}")
    print(f"Saved {fertility_path}")
    print(f"Saved {transfer_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()

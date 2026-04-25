# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.24.0"]
# ///
"""Push eval logs, figures, and the dataset card to the HuggingFace Hub.

Usage:
    uv run src/scripts/hf_scripts/push_eval_artifacts.py
"""

from pathlib import Path

from huggingface_hub import HfApi

HF_REPO_ID = "danish-foundation-models/multilingual-gsm-symbolic"
REPO_TYPE = "dataset"

_HERE = Path(__file__).parent
FIGURES_DIR = _HERE / "figures"
LOGS_DIR = Path("logs")
DATASET_CARD = _HERE / "DATASET_CARD.md"


def main() -> None:
    api = HfApi()

    # Upload dataset card
    if DATASET_CARD.exists():
        api.upload_file(
            path_or_fileobj=DATASET_CARD,
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"Uploaded {DATASET_CARD} → README.md")

    # Upload figures
    for fig in sorted(FIGURES_DIR.glob("*.png")):
        api.upload_file(
            path_or_fileobj=fig,
            path_in_repo=f"figures/{fig.name}",
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"Uploaded {fig}")

    # Upload eval logs
    for log in sorted(LOGS_DIR.glob("*.eval")):
        api.upload_file(
            path_or_fileobj=log,
            path_in_repo=f"logs/{log.name}",
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"Uploaded {log}")


if __name__ == "__main__":
    main()

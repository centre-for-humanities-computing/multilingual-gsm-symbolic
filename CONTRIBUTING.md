# Contributing

## Setup

```bash
make install
```

## Testing

```bash
make test
```

## Linting

```bash
make lint
```

## Publishing

Releases are automated with [python-semantic-release](https://python-semantic-release.readthedocs.io/) (see `.github/workflows/release.yml`).
When the tests pass on `main`, the version is bumped based on the [conventional commit](https://www.conventionalcommits.org/) messages since the last release, a `vX.Y.Z` tag is pushed, and the package is published to PyPI and GitHub Releases:

- `feat: ...` bumps the minor version (0.4.14 → 0.5.0)
- `fix: ...` or `perf: ...` bumps the patch version (0.4.14 → 0.4.15)
- other prefixes (`docs:`, `ci:`, `chore:`, ...) or non-conventional messages do not trigger a release

As PRs are squash-merged, it is the PR title that decides the release, so make sure it uses the right prefix.

The package version is not stored in `pyproject.toml`; it is read from the latest git tag at build time (via `hatch-vcs`), so a release only pushes a tag and never commits to `main`.

To publish the package manually instead:

```bash
# Build and publish to PyPI
uv build
uv publish
```

Credentials can be provided via `UV_PUBLISH_TOKEN` or with `--token` / `--username` + `--password` flags.
See the [uv publish docs](https://docs.astral.sh/uv/guides/publish/) for details.

## Dataset

The generated questions are stored in `dataset/<lang>/<template>.json`, one file per template, with the original question and its synthetic variants.
When you add or change a template, regenerate its file with:

```bash
make build-dataset
```

This only regenerates files that are missing or whose template has changed. Changes elsewhere (e.g. to `replacements.json` or the generation code) are not detected: delete the affected files in `dataset/` and run it again. The tests check that `dataset/` is up to date with the templates.

Each release builds the [Hugging Face dataset](https://huggingface.co/datasets/danish-foundation-models/multilingual-gsm-symbolic) from `dataset/` (`src/scripts/build_hf_dataset.py`) and pushes it, tagged with the same version (see `.github/workflows/publish-dataset.yml`).
The dataset card is edited directly on the Hub; a release only updates the parts between its `START`/`END` markers (the metadata, the version and the language overview), and replaces `eval.yaml`, which is generated from the `instruction.toml` files.
To republish the dataset for an existing release, run the "Publish dataset" workflow manually with the release tag.

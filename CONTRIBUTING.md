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

```bash
# Build and publish to PyPI
uv build
uv publish
```

Credentials can be provided via `UV_PUBLISH_TOKEN` or with `--token` / `--username` + `--password` flags.
See the [uv publish docs](https://docs.astral.sh/uv/guides/publish/) for details.

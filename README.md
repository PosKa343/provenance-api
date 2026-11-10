# Provenance API

A Python service that will serve an AI-generated-text classifier behind an
authenticated, rate-limited REST API with background batch jobs.

**Status:** early scaffold. Only the project skeleton and CI exist so far; no
endpoints are implemented yet.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # create .venv and install dependencies
uv run pre-commit install    # run checks on every commit
uv run ruff check .          # lint
uv run ruff format --check . # format check
uv run mypy                  # type check (strict)
uv run pytest                # tests
```

## License

MIT. See [LICENSE](LICENSE).

# Provenance API

A Python service that will serve an AI-generated-text classifier behind an
authenticated, rate-limited REST API with background batch jobs.

**Status:** `POST /v1/classify` works end to end and stores each prediction in
PostgreSQL. There is no authentication, rate limiting or batch processing yet.

## Quickstart

Requires Docker.

```sh
docker compose up --build
```

This starts PostgreSQL, applies the database migrations, and serves the API on
port 8000. Then:

```sh
curl http://localhost:8000/healthz

curl http://localhost:8000/v1/classify \
  --header 'Content-Type: application/json' \
  --data '{"text": "The committee met on Tuesday. It approved the budget."}'
```

Interactive API docs are at http://localhost:8000/docs.

The service stores a SHA-256 hash of each submitted text, never the text.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # create .venv and install dependencies
uv run python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
uv run pre-commit install    # run checks on every commit
uv run ruff check .          # lint
uv run ruff format --check . # format check
uv run mypy                  # type check (strict)
uv run pytest                # tests
```

The tests run against a real PostgreSQL. By default they start a throwaway
container, so Docker must be running and your user must be allowed to use it.
To use an existing database instead, set `PROVENANCE_TEST_DATABASE_URL`.

To run the API outside Docker:

```sh
docker compose up --detach postgres
uv run alembic upgrade head
uv run uvicorn provenance_api.main:app --reload
```

## Model provenance

The classifier is a **retrain** of my earlier AI-generated-text detection
project, not the original trained model, which was never saved. The feature
extraction in `src/provenance_api/ml/features.py` is a port of the original
code and gives the same 29 feature values on every row of the dataset (largest
difference 8.7e-14). The dataset is in `data/`, with its sources and license.

Retrain with `uv run python -m provenance_api.ml.train`. It writes the model
and a `model.json` with the measured metrics to
`src/provenance_api/ml/artifacts/`.

| | 5-fold CV accuracy | Holdout accuracy (127 rows) |
|---|---|---|
| Original notebook (2025) | 0.9763 ± 0.0133 | 0.9921 |
| This retrain (2026-11-09) | 0.9731 ± 0.0095 | 0.9921 |

Limits:

- AI-text detection is unreliable. Do not use this model to accuse or
  penalise anyone.
- The data has one source per class (Wikipedia for human, Gemini Flash for
  AI), so the model may have learned those two styles and nothing more
  general. The accuracy above says nothing about other kinds of text.
- Training texts were 300 to 1000 words long. Several features are raw counts,
  so scores for much shorter or longer texts are not trustworthy.

## License

MIT. See [LICENSE](LICENSE).

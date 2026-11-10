# Provenance API

A Python service that will serve an AI-generated-text classifier behind an
authenticated, rate-limited REST API with background batch jobs.

**Status:** accounts, API keys, roles and `POST /v1/classify` work end to end
with PostgreSQL. There is no rate limiting, batch processing or billing yet.

## Quickstart

Requires Docker.

```sh
docker compose up --build
```

This starts PostgreSQL, applies the database migrations, and serves the API on
port 8000. Then create an account, log in, create an API key and classify a
text:

```sh
curl http://localhost:8000/v1/auth/register \
  --header 'Content-Type: application/json' \
  --data '{"email": "me@example.com", "password": "a long password"}'

curl http://localhost:8000/v1/auth/login \
  --header 'Content-Type: application/json' \
  --data '{"email": "me@example.com", "password": "a long password"}'
# copy "access_token" from the response

curl http://localhost:8000/v1/api-keys \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <access_token>' \
  --data '{"name": "laptop"}'
# copy "key" from the response; it is shown only once

curl http://localhost:8000/v1/classify \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: <key>' \
  --data '{"text": "The committee met on Tuesday. It approved the budget."}'
```

`python3 scripts/smoke_test.py` runs the same path in one command.
Interactive API docs are at http://localhost:8000/docs.

## Authentication

There are two kinds of credential, for two kinds of client:

- **Access tokens** (JWT, 30 minutes) are for a person using a dashboard. They
  open `/v1/me`, `/v1/api-keys` and `/v1/admin/*`.
- **API keys** are for programs. They open `/v1/classify`. The server stores
  only a SHA-256 hash of each key, so a leaked database does not leak keys.

Passwords are hashed with Argon2id. The service stores a SHA-256 hash of each
classified text, never the text.

Roles: `member` (the default) can manage their own API keys, `viewer` can only
read, and `admin` can also list all users. There is no API for changing roles.
Make the first admin from a shell:

```sh
docker compose exec api provenance-api set-role me@example.com admin
```

The server will not start without `PROVENANCE_JWT_SECRET` (at least 32
characters). `compose.yaml` sets a value for local use only.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # create .venv and install dependencies
uv run provenance-api nltk-data
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
export PROVENANCE_JWT_SECRET=local-development-secret-do-not-use-elsewhere
uv run uvicorn provenance_api.main:create_app --factory --reload
```

## Model provenance

The classifier is a **retrain** of my earlier AI-generated-text detection
project, not the original trained model, which was never saved. The feature
extraction in `src/provenance_api/ml/features.py` is a port of the original
code and gives the same 29 feature values on every row of the dataset (largest
difference 8.7e-14). The dataset is in `data/`, with its sources and license.

Retrain with `uv run provenance-api train`. It writes the model
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

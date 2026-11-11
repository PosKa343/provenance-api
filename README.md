# Provenance API

A Python service that will serve an AI-generated-text classifier behind an
authenticated, rate-limited REST API with background batch jobs.

**Status:** accounts, API keys, roles, `POST /v1/classify`, background batch
jobs, rate limits, usage metering, Stripe billing (test mode only), structured
logs and Prometheus metrics work end to end with PostgreSQL and Redis.

## Quickstart

Requires Docker.

```sh
docker compose up --build
```

This starts PostgreSQL and Redis, applies the database migrations, and runs the
API on port 8000 plus a background worker. Then create an account, log in, create an API key and classify a
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

## Plans and limits

| | Free | Pro |
|---|---|---|
| Daily quota | 100 | 10,000 |
| Burst per API key | 10, refilling 1 per second | 50, refilling 10 per second |
| Batch jobs | no | yes |

- **Daily quota** counts each single classification as 1 and each batch job as
  1 per text. It is shared by all of a user's keys and resets at 00:00 UTC.
- **Burst** is a token bucket per API key.
- Going over either returns `429` with a `Retry-After` header. The error code
  is `rate_limited` for the bucket and `quota_exceeded` for the daily quota.
- `GET /v1/me` shows today's usage and limit; `GET /v1/usage` shows usage by day.

All of these numbers are settings (`PROVENANCE_FREE_DAILY_QUOTA` and so on).
A plan changes through [billing](#billing-stripe-test-mode), or by hand:

```sh
docker compose exec api provenance-api set-plan me@example.com pro
```

Both limits live in Redis, so several API replicas share them. If Redis is
unreachable the limits are not enforced: an outage should not take the API
down. Login and registration are not rate limited yet.

## Billing (Stripe, test mode)

Billing uses Stripe in **test mode only**. No real money can move: the server
refuses to start with a live key.

- `POST /v1/billing/checkout` returns a Stripe Checkout page for the Pro plan.
  Pay with the test card `4242 4242 4242 4242`, any future date, any CVC.
- `POST /v1/billing/webhook` is called by Stripe. A completed checkout or an
  active subscription moves the user to Pro; a cancelled or unpaid one moves
  them back to Free.

How the webhook stays correct:

- **Signature.** Each request must carry a valid `Stripe-Signature` for the
  exact body received, made with the webhook secret and less than five minutes
  old. Anything else is a 400 and changes nothing.
- **Idempotency.** Stripe may deliver an event more than once. Each event id is
  stored in the same transaction as the plan change, so a repeat is
  acknowledged and skipped.
- **Ordering.** Stripe does not promise delivery order. An event older than the
  last one applied to a user is recorded and ignored.

To turn billing on, put these in a `.env` file (git-ignored) in the project
folder. Without them the billing endpoints answer 503 and everything else works.

```sh
PROVENANCE_STRIPE_SECRET_KEY=sk_test_...
PROVENANCE_STRIPE_PRICE_ID=price_...        # a recurring price for the Pro plan
PROVENANCE_STRIPE_WEBHOOK_SECRET=whsec_...  # printed by `stripe listen`
```

Stripe cannot reach `localhost`, so local webhooks go through the
[Stripe CLI](https://docs.stripe.com/stripe-cli):

```sh
stripe listen \
  --events checkout.session.completed,customer.subscription.created,customer.subscription.updated,customer.subscription.deleted \
  --forward-to localhost:8000/v1/billing/webhook
```

`stripe listen --print-secret` prints the `whsec_...` value for `.env`. Restart
the stack after changing `.env`.

There is no endpoint to cancel a subscription yet; cancel it in the Stripe
dashboard and the webhook moves the user back to Free.

## Batch jobs

Batch jobs need the Pro plan. `POST /v1/jobs` takes a list of texts and returns at once with a job id. A
separate worker process scores the texts; poll `GET /v1/jobs/{id}` for
`queued`, `running`, `succeeded` or `failed`, and read
`GET /v1/jobs/{id}/results` for the results in submission order.

```sh
curl http://localhost:8000/v1/jobs \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: <key>' \
  --header 'Idempotency-Key: my-batch-1' \
  --data '{"texts": ["First text to score.", "Second text to score."]}'
```

- **Retrying is safe.** Send the same `Idempotency-Key` again and you get the
  first job back, not a second one. The same key with different texts is a 409.
- **Failures are retried.** A job is attempted three times with a growing
  delay, then marked `failed`.
- **Work is not repeated or duplicated.** Progress is saved every ten texts, and
  a retry skips texts that already have a result.
- **Texts are not stored.** They wait in Redis until the worker has scored
  them, then are deleted (or expire after 24 hours if never picked up). They
  are kept out of the queue entry and out of logs. PostgreSQL keeps only
  hashes and results.

Known limits: if the API process dies between saving a job and queueing it,
the job stays `queued`; if a worker is killed during its last attempt, the job
stays `running`. Both need a periodic sweeper, which is not built yet.

## Operations

**Logs.** The API and the worker write one JSON object per line to stdout.
Every line written while handling a request carries that request's
`request_id`, which is also returned in the `X-Request-ID` header and in error
bodies, so a user's error report leads straight to the log lines. A caller may
send its own `X-Request-ID`. Worker lines carry the `job_id`. Request bodies and
submitted texts are never logged. Set `PROVENANCE_LOG_FORMAT=console` for
plain text while developing.

**Endpoints.**

| Endpoint | Answers | Checks |
|---|---|---|
| `GET /healthz` | Is the process alive? | Nothing else, on purpose: a database outage should not get healthy API processes restarted |
| `GET /readyz` | Can it serve requests now? | Model loaded, PostgreSQL and Redis reachable. 503 names what is down |
| `GET /metrics` | Prometheus metrics | |

**Metrics.** The API serves `http_request_duration_seconds` (by method, route
template and status), `predictions_total` and `job_queue_depth`. The worker is
a separate process and serves its own on port 9100: `jobs_finished_total`,
`job_duration_seconds` and its `predictions_total`. `/metrics` has no
authentication, so keep it off the public internet. Each container runs one
process; running several per container would need Prometheus's multiprocess mode.

**Shutdown.** On `SIGTERM` the API stops accepting connections, lets requests
in flight finish (up to 20 seconds), then closes its database and Redis
connections. The worker gives running jobs 20 seconds to finish; a job cut off
after that is picked up again later and resumes where it stopped.

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

The tests run against a real PostgreSQL and Redis. By default they start
throwaway containers, so Docker must be running and your user must be allowed to use it.
To use existing servers instead, set `PROVENANCE_TEST_DATABASE_URL` and
`PROVENANCE_TEST_REDIS_URL`.

To run the API outside Docker:

```sh
docker compose up --detach postgres redis
uv run alembic upgrade head
export PROVENANCE_JWT_SECRET=local-development-secret-do-not-use-elsewhere
uv run uvicorn provenance_api.main:create_app --factory --reload
uv run provenance-api worker   # in a second terminal
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

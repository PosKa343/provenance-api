"""Walk the main path of a running API: register, log in, create a key, classify, run a batch.

Usage: python scripts/smoke_test.py [base_url] [--upgrade-command COMMAND]

Batch jobs need the Pro plan. Pass --upgrade-command to have the script move
its user to Pro and test a batch job too, for example:

    --upgrade-command "docker compose exec -T api provenance-api set-plan {email} pro"

Uses only the standard library, so it runs anywhere Python does.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

parser = argparse.ArgumentParser(description="Smoke-test a running Provenance API.")
parser.add_argument("base_url", nargs="?", default="http://localhost:8000")
parser.add_argument("--upgrade-command", help="Shell command that moves {email} to the Pro plan.")
ARGS = parser.parse_args()
BASE_URL: str = ARGS.base_url


def call(
    method: str, path: str, body: dict[str, Any] | None = None, **headers: str
) -> tuple[int, Any]:
    request = urllib.request.Request(  # noqa: S310 - the URL is given by whoever runs the script
        BASE_URL + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    return status, json.loads(raw) if raw else None


def expect(description: str, actual: object, expected: object) -> None:
    if actual != expected:
        sys.exit(f"FAIL {description}: expected {expected!r}, got {actual!r}")
    print(f"ok   {description}")


def main() -> None:
    credentials = {
        "email": f"smoke-{uuid.uuid4().hex[:8]}@example.com",
        "password": "smoke test password",
    }
    text = {"text": "The committee met on Tuesday. It approved the budget."}

    status, _ = call("POST", "/v1/auth/register", credentials)
    expect("register", status, 201)

    status, login = call("POST", "/v1/auth/login", credentials)
    expect("login", status, 200)
    token = {"Authorization": f"Bearer {login['access_token']}"}

    status, created = call("POST", "/v1/api-keys", {"name": "smoke"}, **token)
    expect("create API key", status, 201)

    status, _ = call("POST", "/v1/classify", text)
    expect("classify without a key is rejected", status, 401)

    key = {"X-API-Key": created["key"]}
    status, prediction = call("POST", "/v1/classify", text, **key)
    expect("classify with the key", status, 200)
    expect("label is human or ai", prediction["label"] in ("human", "ai"), True)

    batch = {"texts": [text["text"], "Rain fell all night. The river rose by morning."]}
    status, _ = call("POST", "/v1/jobs", batch, **key)
    expect("a batch job on the free plan is refused", status, 403)

    status, me = call("GET", "/v1/me", **token)
    expect("today's usage counts the one classification", me["usage"]["used_today"], 1)

    if ARGS.upgrade_command:
        command = shlex.split(ARGS.upgrade_command.format(email=credentials["email"]))
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)  # noqa: S603
        run_batch_job(batch, key)

    status, _ = call("DELETE", f"/v1/api-keys/{created['id']}", **token)
    expect("revoke the key", status, 204)


def run_batch_job(batch: dict[str, Any], key: dict[str, str]) -> None:
    status, job = call("POST", "/v1/jobs", batch, **key, **{"Idempotency-Key": "smoke-1"})
    expect("submit a batch job on the Pro plan", status, 202)

    status, replay = call("POST", "/v1/jobs", batch, **key, **{"Idempotency-Key": "smoke-1"})
    expect("resubmitting with the same key returns the same job", replay["id"], job["id"])

    for _ in range(60):
        status, job = call("GET", f"/v1/jobs/{job['id']}", **key)
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    expect("the worker finishes the job", job["status"], "succeeded")

    status, results = call("GET", f"/v1/jobs/{job['id']}/results", **key)
    expect("the job has one result per text", len(results["items"]), 2)


if __name__ == "__main__":
    main()

"""Walk the main path of a running API: register, log in, create a key, classify.

Usage: python scripts/smoke_test.py [base_url]
Uses only the standard library, so it runs anywhere Python does.
"""

import json
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


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

    status, prediction = call("POST", "/v1/classify", text, **{"X-API-Key": created["key"]})
    expect("classify with the key", status, 200)
    expect("label is human or ai", prediction["label"] in ("human", "ai"), True)

    status, _ = call("DELETE", f"/v1/api-keys/{created['id']}", **token)
    expect("revoke the key", status, 204)


if __name__ == "__main__":
    main()

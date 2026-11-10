import hashlib
import uuid

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from provenance_api.ml.features import FEATURE_NAMES

pytestmark = pytest.mark.anyio

TEXT = "The committee met on Tuesday. It approved the budget after a short debate."


async def test_classify_returns_a_prediction(
    client: httpx.AsyncClient, db: AsyncEngine, api_key: dict[str, str]
) -> None:
    response = await client.post("/v1/classify", json={"text": TEXT}, headers=api_key)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "prediction_id",
        "label",
        "probability_ai",
        "top_features",
        "model_version",
    }
    assert body["label"] in {"human", "ai"}
    assert 0 <= body["probability_ai"] <= 1
    assert (body["label"] == "ai") == (body["probability_ai"] >= 0.5)
    assert len(body["top_features"]) == 5
    assert {feature["name"] for feature in body["top_features"]} <= set(FEATURE_NAMES)
    assert body["model_version"].startswith("lr-")


async def test_classify_stores_a_hash_of_the_text_and_not_the_text(
    client: httpx.AsyncClient, db: AsyncEngine, api_key: dict[str, str]
) -> None:
    response = await client.post("/v1/classify", json={"text": TEXT}, headers=api_key)
    body = response.json()

    async with db.connect() as connection:
        rows = (await connection.execute(text("SELECT * FROM predictions"))).mappings().all()

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == uuid.UUID(body["prediction_id"])
    assert row["input_sha256"] == hashlib.sha256(TEXT.encode()).hexdigest()
    assert row["label"] == body["label"]
    assert row["probability_ai"] == pytest.approx(body["probability_ai"])
    assert row["model_version"] == body["model_version"]
    assert row["created_at"] is not None
    assert TEXT not in [value for value in row.values() if isinstance(value, str)]


async def test_same_text_twice_gives_two_rows_with_the_same_hash(
    client: httpx.AsyncClient, db: AsyncEngine, api_key: dict[str, str]
) -> None:
    await client.post("/v1/classify", json={"text": TEXT}, headers=api_key)
    await client.post("/v1/classify", json={"text": TEXT}, headers=api_key)

    async with db.connect() as connection:
        hashes = (await connection.execute(text("SELECT input_sha256 FROM predictions"))).scalars()
        assert list(hashes) == [hashlib.sha256(TEXT.encode()).hexdigest()] * 2


@pytest.mark.parametrize(
    ("payload", "message_part"),
    [
        ({}, "body.text"),
        ({"text": "   \n "}, "non-whitespace"),
        ({"text": 42}, "body.text"),
        ({"text": TEXT, "unexpected": True}, "body.unexpected"),
    ],
)
async def test_invalid_body_is_rejected_with_the_error_envelope(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    api_key: dict[str, str],
    payload: dict[str, object],
    message_part: str,
) -> None:
    response = await client.post("/v1/classify", json=payload, headers=api_key)

    assert response.status_code == 422
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id"}
    assert error["code"] == "validation_error"
    assert message_part in error["message"]
    assert error["request_id"] == response.headers["X-Request-ID"]

    async with db.connect() as connection:
        count = (await connection.execute(text("SELECT count(*) FROM predictions"))).scalar_one()
    assert count == 0


async def test_text_over_the_configured_limit_is_rejected(
    client: httpx.AsyncClient, db: AsyncEngine, api_key: dict[str, str]
) -> None:
    response = await client.post("/v1/classify", json={"text": "word " * 1001}, headers=api_key)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "text_too_long"
    assert "5000" in response.json()["error"]["message"]


async def test_text_at_the_limit_is_accepted(
    client: httpx.AsyncClient, db: AsyncEngine, api_key: dict[str, str]
) -> None:
    response = await client.post("/v1/classify", json={"text": "word " * 1000}, headers=api_key)

    assert response.status_code == 200


async def test_unexpected_failure_returns_500_envelope_without_leaking_details(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    api_key: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(self: object, text: str) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("provenance_api.ml.model.Classifier.predict", explode)

    response = await client.post("/v1/classify", json={"text": TEXT}, headers=api_key)

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "Internal server error.",
            "request_id": response.headers["X-Request-ID"],
        }
    }

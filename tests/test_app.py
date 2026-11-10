import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_unknown_path_uses_the_error_envelope(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/nope")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Not Found",
            "request_id": response.headers["X-Request-ID"],
        }
    }


async def test_every_response_gets_a_fresh_request_id(client: httpx.AsyncClient) -> None:
    first = await client.get("/healthz")
    second = await client.get("/healthz")

    assert len(first.headers["X-Request-ID"]) == 32
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


async def test_openapi_documents_classify_and_its_errors(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()

    operation = schema["paths"]["/v1/classify"]["post"]
    assert set(operation["responses"]) == {"200", "401", "413", "422", "429"}
    assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorEnvelope"
    }

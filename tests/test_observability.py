import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

import anyio
import httpx
import pytest
from arq.connections import ArqRedis
from fastapi import FastAPI
from prometheus_client import REGISTRY

from provenance_api.config import Settings
from provenance_api.logs import formatter
from provenance_api.ml.model import Classifier, Prediction
from provenance_api.worker import create_worker
from tests.conftest import TEST_JWT_SECRET, Headers

pytestmark = pytest.mark.anyio

BODY = {"text": "The committee met on Tuesday. It approved the budget."}
BATCH = {"texts": ["First text to score.", "Second text to score.", "Third text to score."]}

RunWorker = Callable[[], Awaitable[None]]


@pytest.fixture
def run_worker(settings: Settings, queue: ArqRedis) -> RunWorker:
    async def run() -> None:
        worker = create_worker(settings, redis_pool=queue, burst=True, poll_delay=0.01)
        try:
            await worker.main()
        finally:
            await worker.ctx["engine"].dispose()

    return run


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def as_json(records: list[logging.LogRecord]) -> list[dict[str, object]]:
    """Render captured records the way the service writes them to stdout."""
    json_formatter = formatter("json")
    return [json.loads(json_formatter.format(record)) for record in records]


async def test_each_request_writes_one_json_log_line_with_its_request_id(
    client: httpx.AsyncClient, api_key: Headers, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    caplog.clear()

    response = await client.post("/v1/classify", json=BODY, headers=api_key)

    lines = [line for line in as_json(caplog.records) if line.get("event") == "request"]
    assert len(lines) == 1
    line = lines[0]
    assert line["request_id"] == response.headers["X-Request-ID"]
    assert (line["method"], line["route"], line["status"]) == ("POST", "/v1/classify", 200)
    assert isinstance(line["duration_ms"], float)
    assert line["level"] == "info"
    assert "timestamp" in line
    assert BODY["text"] not in json.dumps(line)


async def test_log_lines_from_inside_a_request_carry_its_request_id(
    client: httpx.AsyncClient,
    api_key: Headers,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(self: Classifier, text: str) -> Prediction:
        raise RuntimeError("boom")

    monkeypatch.setattr(Classifier, "predict", explode)
    caplog.set_level(logging.INFO)
    caplog.clear()

    response = await client.post("/v1/classify", json=BODY, headers=api_key)

    assert response.status_code == 500
    lines = as_json(caplog.records)
    error_line = next(line for line in lines if line["event"] == "Unhandled error")
    access_line = next(line for line in lines if line["event"] == "request")
    assert error_line["request_id"] == response.headers["X-Request-ID"]
    assert "RuntimeError: boom" in str(error_line["exception"])
    assert access_line["status"] == 500


async def test_a_sensible_request_id_from_the_caller_is_kept(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/nope", headers={"X-Request-ID": "trace-abc-12345"})

    assert response.headers["X-Request-ID"] == "trace-abc-12345"
    assert response.json()["error"]["request_id"] == "trace-abc-12345"


@pytest.mark.parametrize("supplied", ["short", "x" * 65, "has spaces in it", 'quote"injection'])
async def test_an_unusable_request_id_from_the_caller_is_replaced(
    client: httpx.AsyncClient, supplied: str
) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": supplied})

    assert response.headers["X-Request-ID"] != supplied
    assert len(response.headers["X-Request-ID"]) == 32


async def test_health_checks_are_not_logged(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    caplog.clear()

    for path in ("/healthz", "/readyz", "/metrics"):
        await client.get(path)

    assert [line for line in as_json(caplog.records) if line.get("event") == "request"] == []


async def test_metrics_time_requests_by_route_template_not_raw_path(
    client: httpx.AsyncClient, api_key: Headers
) -> None:
    job_id = str(uuid.uuid4())
    labels = {"method": "GET", "route": "/v1/jobs/{job_id}", "status": "404"}
    before = sample("http_request_duration_seconds_count", **labels)

    await client.get(f"/v1/jobs/{job_id}", headers=api_key)
    await client.get("/no/such/path")
    exposition = (await client.get("/metrics")).text

    assert sample("http_request_duration_seconds_count", **labels) == before + 1
    assert 'route="/v1/jobs/{job_id}"' in exposition
    assert 'route="unmatched"' in exposition
    assert job_id not in exposition
    assert "/no/such/path" not in exposition


async def test_metrics_count_single_and_batch_predictions(
    client: httpx.AsyncClient, pro_api_key: Headers, run_worker: RunWorker
) -> None:
    def predictions(source: str) -> float:
        return sum(
            sample("predictions_total", label=label, source=source) for label in ("human", "ai")
        )

    single_before, batch_before = predictions("single"), predictions("batch")
    succeeded_before = sample("jobs_finished_total", status="succeeded")

    await client.post("/v1/classify", json=BODY, headers=pro_api_key)
    await client.post("/v1/jobs", json=BATCH, headers=pro_api_key)
    queued = (await client.get("/metrics")).text
    await run_worker()
    drained = (await client.get("/metrics")).text

    assert predictions("single") == single_before + 1
    assert predictions("batch") == batch_before + 3
    assert sample("jobs_finished_total", status="succeeded") == succeeded_before + 1
    assert "job_queue_depth 1.0" in queued
    assert "job_queue_depth 0.0" in drained


async def test_failed_job_is_counted(
    client: httpx.AsyncClient,
    pro_api_key: Headers,
    run_worker: RunWorker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(self: Classifier, text: str) -> Prediction:
        raise RuntimeError("boom")

    failed_before = sample("jobs_finished_total", status="failed")
    await client.post("/v1/jobs", json=BATCH, headers=pro_api_key)
    monkeypatch.setattr(Classifier, "predict", explode)

    await run_worker()

    assert sample("jobs_finished_total", status="failed") == failed_before + 1


async def test_readyz_reports_ready_when_every_dependency_answers(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"model": True, "database": True, "redis": True},
    }


async def test_readyz_fails_and_names_the_dependency_that_is_down(
    client: httpx.AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unreachable() -> None:
        raise ConnectionError("redis is down")

    monkeypatch.setattr(app.state.queue, "ping", unreachable)

    ready = await client.get("/readyz")
    alive = await client.get("/healthz")

    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "checks": {"model": True, "database": True, "redis": False},
    }
    assert alive.status_code == 200


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def test_sigterm_lets_a_request_in_flight_finish_before_the_server_exits(
    database_url: str, redis_url: str
) -> None:
    port = _free_port()
    server = subprocess.Popen(  # noqa: S603 - fixed command, test-controlled arguments
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.support.slow_app:create_app",
            "--factory",
            "--port",
            str(port),
            "--no-access-log",
            "--timeout-graceful-shutdown",
            "10",
        ],
        env={
            **os.environ,
            "PROVENANCE_DATABASE_URL": database_url,
            "PROVENANCE_REDIS_URL": redis_url,
            "PROVENANCE_JWT_SECRET": TEST_JWT_SECRET,
        },
        # The service logs to stdout. stderr is left alone: tools in the
        # environment may write notices there that are not the service's logs.
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as http:
            for _ in range(100):
                try:
                    if (await http.get("/readyz")).status_code == 200:
                        break
                except httpx.TransportError:
                    await anyio.sleep(0.1)
            else:
                pytest.fail("server did not start")

            async with anyio.create_task_group() as tasks:
                responses: list[httpx.Response] = []

                async def slow_request() -> None:
                    responses.append(await http.get("/slow"))

                tasks.start_soon(slow_request)
                await anyio.sleep(0.3)  # the request is now being handled
                server.send_signal(signal.SIGTERM)
                sent_at = time.monotonic()

        exit_code = await anyio.to_thread.run_sync(server.wait)
        output = server.stdout.read() if server.stdout else ""
    finally:
        server.kill()

    assert responses[0].status_code == 200
    assert responses[0].json() == {"status": "finished"}
    # After shutting down cleanly uvicorn re-raises the signal, so the exit
    # status is "ended by SIGTERM". The log shows whether shutdown completed.
    assert exit_code in (0, -signal.SIGTERM)
    assert time.monotonic() - sent_at < 8
    # The server's own output is JSON lines too.
    lines = [line for line in output.splitlines() if line.strip()]
    assert [line for line in lines if not line.startswith("{")] == []
    events = [json.loads(line)["event"] for line in lines]
    assert "request" in events
    assert "Application shutdown complete." in events


async def test_worker_logs_one_line_per_finished_job_tagged_with_the_job_id(
    client: httpx.AsyncClient,
    pro_api_key: Headers,
    run_worker: RunWorker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = (await client.post("/v1/jobs", json=BATCH, headers=pro_api_key)).json()
    caplog.set_level(logging.INFO)
    caplog.clear()

    await run_worker()

    finished = [line for line in as_json(caplog.records) if line.get("event") == "job finished"]
    assert len(finished) == 1
    assert finished[0]["job_id"] == job["id"]
    assert (finished[0]["status"], finished[0]["texts"], finished[0]["attempt"]) == (
        "succeeded",
        3,
        1,
    )
    assert not any(text in json.dumps(finished[0]) for text in BATCH["texts"])

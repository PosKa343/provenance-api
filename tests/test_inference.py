import os
import signal
from collections.abc import AsyncIterator

import anyio
import pytest

from provenance_api.inference import InferencePool
from provenance_api.ml.model import Classifier

pytestmark = pytest.mark.anyio

TEXTS = [
    "The committee met on Tuesday. It approved the budget.",
    "Rain fell all night. The river rose by morning.",
    "She opened the old door slowly. Nobody was inside.",
]


@pytest.fixture(scope="module")
async def process_pool() -> AsyncIterator[InferencePool]:
    pool = InferencePool(workers=2)
    await pool.warm_up()
    yield pool
    pool.close()


async def test_worker_processes_give_the_same_predictions_as_the_classifier_itself(
    process_pool: InferencePool,
) -> None:
    expected = [Classifier.load().predict(text) for text in TEXTS]

    assert await process_pool.predict_many(TEXTS) == expected
    assert await process_pool.predict(TEXTS[1]) == expected[1]
    assert process_pool.model_version == expected[0].model_version


async def test_in_process_mode_gives_the_same_predictions(process_pool: InferencePool) -> None:
    in_process = InferencePool(workers=0)
    try:
        assert await in_process.predict_many(TEXTS) == await process_pool.predict_many(TEXTS)
    finally:
        in_process.close()


async def test_prediction_really_runs_in_another_process(process_pool: InferencePool) -> None:
    worker_pids = set(process_pool._executor._processes)  # type: ignore[attr-defined]

    assert len(worker_pids) == 2
    assert os.getpid() not in worker_pids


async def test_pool_recovers_after_a_worker_process_is_killed(process_pool: InferencePool) -> None:
    processes = process_pool._executor._processes  # type: ignore[attr-defined]
    victim_pid, victim = next(iter(processes.items()))
    os.kill(victim_pid, signal.SIGKILL)
    await anyio.to_thread.run_sync(victim.join)

    # The first call after the death may still be served by the surviving
    # worker, before the pool has noticed. Every call must succeed either way,
    # and the pool must end up without the dead process.
    expected = Classifier.load().predict(TEXTS[0])
    for _ in range(5):
        assert await process_pool.predict(TEXTS[0]) == expected
        await anyio.sleep(0.1)

    current = process_pool._executor._processes  # type: ignore[attr-defined]
    assert victim_pid not in current
    assert len(current) >= 1

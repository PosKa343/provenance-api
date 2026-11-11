"""Runs the classifier off the event loop, in worker processes.

Why processes and not threads: NLTK sends every regular expression through an
engine that releases Python's global interpreter lock on each call, several
hundred times per text. Threads then spend their time passing the lock between
CPU cores. Measured on the development machine, 16 threads classified about a
quarter as many texts per second as one thread did. Separate processes each
have their own lock and so run in parallel. See BENCHMARKS.md.
"""

import asyncio
import multiprocessing
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from provenance_api.ml.model import Classifier, Prediction

# One classifier per worker process, loaded when the process starts.
_worker_classifier: Classifier | None = None


def _load_in_worker() -> None:
    global _worker_classifier
    _worker_classifier = Classifier.load()


def _predict_in_worker(texts: list[str]) -> list[Prediction]:
    if _worker_classifier is None:
        raise RuntimeError("The inference worker has no classifier loaded.")
    return [_worker_classifier.predict(text) for text in texts]


class InferencePool:
    """Scores texts without blocking the caller's event loop.

    workers > 0: that many worker processes, for real parallelism.
    workers = 0: one thread in this process. Used by tests, and it still avoids
    the lock contention above because only one text is scored at a time.
    """

    def __init__(self, workers: int) -> None:
        # Loaded here as well: it checks the model and NLTK data at startup,
        # before any worker process exists, and provides the model version.
        self._classifier = Classifier.load()
        self._workers = workers
        self._executor = self._new_executor()

    def _new_executor(self) -> Executor:
        if self._workers == 0:
            return ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        return ProcessPoolExecutor(
            max_workers=self._workers,
            # "spawn" starts each worker as a fresh interpreter. Forking a
            # process that already runs an event loop and threads is unsafe.
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_load_in_worker,
        )

    def _predict_here(self, texts: list[str]) -> list[Prediction]:
        return [self._classifier.predict(text) for text in texts]

    @property
    def model_version(self) -> str:
        return self._classifier.model_version

    async def warm_up(self) -> None:
        """Start every worker process now, so the first real request does not pay for it."""
        if self._workers > 0:
            await asyncio.gather(*(self.predict("Warm up.") for _ in range(self._workers)))

    async def predict(self, text: str) -> Prediction:
        return (await self.predict_many([text]))[0]

    async def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Score texts in order. A batch goes to a single worker as one task."""
        task = self._predict_here if self._workers == 0 else _predict_in_worker
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, task, texts)
        except BrokenProcessPool:
            # A worker died (for example, killed for using too much memory) and
            # the pool cannot be used again. Replace it and try once more.
            self._executor.shutdown(wait=False)
            self._executor = self._new_executor()
            return await loop.run_in_executor(self._executor, task, texts)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

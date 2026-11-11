"""Load test for POST /v1/classify. Run it through loadtest/run.sh."""

import csv
import itertools
import os
from pathlib import Path

from locust import HttpUser, constant, task

DATASET = Path(__file__).parent.parent / "data" / "ai_detection_dataset.csv"


def _texts() -> list[str]:
    # All 632 dataset texts (Wikipedia extracts and Gemini outputs, 27 to 1,325
    # words, median 293), the texts the model was trained on. Latency depends
    # on text length, and sending one text over and over would let caches hide
    # the real cost, so every request sends a different one.
    with DATASET.open(newline="", encoding="utf-8") as handle:
        return [row["content"] for row in csv.DictReader(handle)]


TEXTS = _texts()
_next_text = itertools.cycle(TEXTS)


class ClassifyUser(HttpUser):
    # No think time: each simulated user sends its next request as soon as the
    # last one returns, so the run finds the service's maximum throughput.
    wait_time = constant(0)

    def on_start(self) -> None:
        self.client.headers["X-API-Key"] = os.environ["PROVENANCE_API_KEY"]

    @task
    def classify(self) -> None:
        self.client.post("/v1/classify", json={"text": next(_next_text)})

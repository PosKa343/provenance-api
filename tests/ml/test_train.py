import json
from pathlib import Path

import joblib
import pytest

from provenance_api.ml.train import METADATA_PATH, MODEL_PATH, train

DATASET = Path(__file__).parent.parent.parent / "data" / "ai_detection_dataset.csv"


def test_retraining_reproduces_the_committed_model() -> None:
    committed_model = joblib.load(MODEL_PATH)
    committed_metadata = json.loads(METADATA_PATH.read_text())

    result = train(DATASET)

    assert result.estimator.coef_[0] == pytest.approx(committed_model.coef_[0], abs=1e-6)
    assert result.metadata["metrics"] == committed_metadata["metrics"]
    assert result.metadata["dataset"] == committed_metadata["dataset"]

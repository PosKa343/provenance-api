import json
from pathlib import Path

import joblib
import pytest

from provenance_api.ml.train import METADATA_PATH, MODEL_PATH, train

DATASET = Path(__file__).parent.parent.parent / "data" / "ai_detection_dataset.csv"

# The baseline is fitted on unscaled features, so lbfgs stops at max_iter
# before fully converging and the weights it reaches differ slightly between
# CPUs (up to about 0.01 seen on CI runners). The tolerances allow for that.
WEIGHT_TOLERANCE = 0.05
ACCURACY_TOLERANCE = 0.01


def test_retraining_reproduces_the_committed_model() -> None:
    committed_model = joblib.load(MODEL_PATH)
    committed = json.loads(METADATA_PATH.read_text())

    result = train(DATASET)
    metrics = result.metadata["metrics"]

    assert result.metadata["dataset"] == committed["dataset"]
    assert result.estimator.coef_[0] == pytest.approx(
        committed_model.coef_[0], abs=WEIGHT_TOLERANCE
    )
    assert metrics["cv_accuracy_mean"] == pytest.approx(
        committed["metrics"]["cv_accuracy_mean"], abs=ACCURACY_TOLERANCE
    )
    assert metrics["holdout_accuracy"] == pytest.approx(
        committed["metrics"]["holdout_accuracy"], abs=ACCURACY_TOLERANCE
    )

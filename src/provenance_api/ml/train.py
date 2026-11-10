"""Retrain the classifier from the dataset and write the model artifact.

Usage: uv run python -m provenance_api.ml.train [path/to/dataset.csv]
"""

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from provenance_api.ml.features import FEATURE_NAMES, extract_features

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model.json"
DEFAULT_DATASET = Path("data/ai_detection_dataset.csv")

# Same settings as the original notebook, so its results can be compared.
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2
CV_FOLDS = 5


@dataclass(frozen=True)
class TrainingResult:
    estimator: LogisticRegression
    metadata: dict[str, Any]


def load_dataset(path: Path) -> tuple[list[str], NDArray[np.int64]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    texts = [row["content"] for row in rows]
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    return texts, labels


def feature_matrix(texts: list[str]) -> NDArray[np.float64]:
    # extract_features returns values in FEATURE_NAMES order.
    return np.array([list(extract_features(text).values()) for text in texts], dtype=np.float64)


def model_version(estimator: LogisticRegression) -> str:
    """Identify a model by its weights, so equal models get equal versions."""
    weights = json.dumps(
        {
            "features": FEATURE_NAMES,
            "coefficients": [round(float(c), 10) for c in estimator.coef_[0]],
            "intercept": round(float(estimator.intercept_[0]), 10),
        },
        sort_keys=True,
    )
    return "lr-" + hashlib.sha256(weights.encode()).hexdigest()[:12]


def train(dataset_path: Path) -> TrainingResult:
    texts, labels = load_dataset(dataset_path)
    features = feature_matrix(texts)

    def new_estimator() -> LogisticRegression:
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(new_estimator(), features, labels, cv=CV_FOLDS, scoring="accuracy")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=HOLDOUT_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    estimator = new_estimator().fit(x_train, y_train)

    metadata: dict[str, Any] = {
        "model_version": model_version(estimator),
        "trained_at": datetime.now(UTC).date().isoformat(),
        "algorithm": "LogisticRegression(max_iter=1000, random_state=42), unscaled features",
        "feature_names": list(FEATURE_NAMES),
        "dataset": {
            "file": dataset_path.name,
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "rows": len(texts),
            "human": int((labels == 0).sum()),
            "ai": int((labels == 1).sum()),
        },
        "metrics": {
            "cv_folds": CV_FOLDS,
            "cv_accuracy_per_fold": [round(float(score), 4) for score in cv_scores],
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "holdout_rows": len(y_test),
            "holdout_accuracy": round(float(estimator.score(x_test, y_test)), 4),
        },
        "library_versions": {
            name: version(name) for name in ("scikit-learn", "numpy", "nltk", "textstat")
        },
    }
    return TrainingResult(estimator=estimator, metadata=metadata)


def main() -> None:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    result = train(dataset_path)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(result.estimator, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(result.metadata, indent=2) + "\n")
    print(json.dumps(result.metadata["metrics"], indent=2))
    print(f"Wrote {MODEL_PATH} ({result.metadata['model_version']})")


if __name__ == "__main__":
    main()

"""Load the trained classifier once and score texts with it."""

import json
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import as_file, files
from typing import Any

import joblib
import numpy as np

from provenance_api.ml.features import FEATURE_NAMES, extract_features

AI_THRESHOLD = 0.5
TOP_FEATURE_COUNT = 5
_AI_CLASS = 1


class Label(StrEnum):
    HUMAN = "human"
    AI = "ai"


@dataclass(frozen=True)
class FeatureContribution:
    name: str
    value: float
    contribution: float


@dataclass(frozen=True)
class Prediction:
    label: Label
    probability_ai: float
    top_features: tuple[FeatureContribution, ...]
    model_version: str


class ModelArtifactError(RuntimeError):
    """The packaged model does not match the code that is trying to serve it."""


class Classifier:
    """A trained model held in memory. Build it once per process with load()."""

    def __init__(self, estimator: Any, metadata: dict[str, Any]) -> None:
        if tuple(metadata["feature_names"]) != FEATURE_NAMES:
            raise ModelArtifactError(
                "The model was trained on a different feature list than this code extracts."
            )
        self._estimator = estimator
        self._model_version: str = metadata["model_version"]
        self._coefficients = np.asarray(estimator.coef_[0], dtype=np.float64)
        self._ai_column = list(estimator.classes_).index(_AI_CLASS)

    @classmethod
    def load(cls) -> "Classifier":
        artifacts = files("provenance_api.ml") / "artifacts"
        metadata = json.loads((artifacts / "model.json").read_text(encoding="utf-8"))
        # joblib files are pickles and can run code when loaded. Only the file
        # shipped inside this package is ever loaded, never a user-supplied path.
        with as_file(artifacts / "model.joblib") as model_path:
            estimator = joblib.load(model_path)
        return cls(estimator, metadata)

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, text: str) -> Prediction:
        features = extract_features(text)
        values = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

        probability_ai = float(
            self._estimator.predict_proba(values.reshape(1, -1))[0, self._ai_column]
        )

        # For logistic regression the log-odds are intercept + sum(weight * value),
        # so weight * value is how far each feature pushed this text towards AI.
        contributions = self._coefficients * values
        strongest = np.argsort(-np.abs(contributions))[:TOP_FEATURE_COUNT]

        return Prediction(
            label=Label.AI if probability_ai >= AI_THRESHOLD else Label.HUMAN,
            probability_ai=probability_ai,
            top_features=tuple(
                FeatureContribution(
                    name=FEATURE_NAMES[index],
                    value=float(values[index]),
                    contribution=float(contributions[index]),
                )
                for index in strongest
            ),
            model_version=self._model_version,
        )

"""Load the trained classifier once and score texts with it."""

from dataclasses import dataclass
from enum import StrEnum


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


class Classifier:
    @classmethod
    def load(cls) -> "Classifier":
        raise NotImplementedError

    @property
    def model_version(self) -> str:
        raise NotImplementedError

    def predict(self, text: str) -> Prediction:
        raise NotImplementedError

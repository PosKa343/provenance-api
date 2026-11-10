import csv
import re
from pathlib import Path

import pytest

from provenance_api.ml.features import FEATURE_NAMES
from provenance_api.ml.model import Classifier, Label, Prediction

DATASET = Path(__file__).parent.parent.parent / "data" / "ai_detection_dataset.csv"


@pytest.fixture(scope="module")
def classifier() -> Classifier:
    return Classifier.load()


@pytest.fixture(scope="module")
def first_text_per_label() -> dict[str, str]:
    texts: dict[str, str] = {}
    with DATASET.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            texts.setdefault(row["label"], row["content"])
    return texts


def test_wikipedia_article_is_labelled_human(
    classifier: Classifier, first_text_per_label: dict[str, str]
) -> None:
    prediction = classifier.predict(first_text_per_label["0"])

    assert prediction.label is Label.HUMAN
    assert prediction.probability_ai < 0.5


def test_gemini_text_is_labelled_ai(
    classifier: Classifier, first_text_per_label: dict[str, str]
) -> None:
    prediction = classifier.predict(first_text_per_label["1"])

    assert prediction.label is Label.AI
    assert prediction.probability_ai >= 0.5


def test_prediction_reports_five_strongest_features_first(
    classifier: Classifier, first_text_per_label: dict[str, str]
) -> None:
    prediction = classifier.predict(first_text_per_label["1"])

    strengths = [abs(feature.contribution) for feature in prediction.top_features]
    assert len(prediction.top_features) == 5
    assert strengths == sorted(strengths, reverse=True)
    assert {feature.name for feature in prediction.top_features} <= set(FEATURE_NAMES)


def test_prediction_carries_the_model_version(classifier: Classifier) -> None:
    prediction = classifier.predict("A short text. It has two sentences.")

    assert re.fullmatch(r"lr-[0-9a-f]{12}", prediction.model_version)
    assert prediction.model_version == classifier.model_version


def test_same_text_gives_the_same_prediction(classifier: Classifier) -> None:
    text = "A short text. It has two sentences."

    first: Prediction = classifier.predict(text)

    assert classifier.predict(text) == first

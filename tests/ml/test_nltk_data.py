import nltk
import pytest

from provenance_api.ml import nltk_data
from provenance_api.ml.model import Classifier


def test_required_data_is_installed_in_the_test_environment() -> None:
    nltk_data.ensure_available()


def test_loading_the_classifier_fails_fast_when_data_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setattr(nltk.data, "path", [str(tmp_path)])

    with pytest.raises(nltk_data.MissingNltkDataError, match="cmudict"):
        Classifier.load()

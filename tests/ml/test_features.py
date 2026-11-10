import math

import pytest

from provenance_api.ml.features import FEATURE_NAMES, extract_features, preprocess

TWO_SENTENCES = "The cat sat. The cat ran!"


def test_preprocess_lowercases_and_splits_tokens_and_sentences() -> None:
    result = preprocess(TWO_SENTENCES)

    assert result.normalized == "the cat sat. the cat ran!"
    assert result.tokens == ("the", "cat", "sat", ".", "the", "cat", "ran", "!")
    assert result.sentences == ("the cat sat.", "the cat ran!")


def test_preprocess_strips_urls_and_emails() -> None:
    result = preprocess("Write to me@example.com or see https://example.com/page now.")

    assert result.normalized == "write to or see now."


def test_preprocess_expands_contractions() -> None:
    result = preprocess("I can't go and they won't stay, so we're done.")

    assert result.normalized == "i cannot go and they will not stay, so we are done."


def test_extract_features_returns_every_feature_in_training_order() -> None:
    features = extract_features(TWO_SENTENCES)

    assert tuple(features) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 29
    assert all(math.isfinite(value) for value in features.values())


def test_statistical_features_match_hand_computed_values() -> None:
    features = extract_features(TWO_SENTENCES)

    # Punctuation marks are tokens, so there are 8 tokens of total length 20.
    assert features["avg_word_length"] == pytest.approx(2.5)
    assert features["avg_sentence_length"] == pytest.approx(4.0)
    assert features["unique_word_count"] == 6
    assert features["vocabulary_richness"] == pytest.approx(0.75)
    assert features["punctuation_ratio"] == pytest.approx(2 / 25)
    assert features["period_count"] == 1
    assert features["exclamation_count"] == 1
    assert features["comma_count"] == 0
    assert features["sentence_length_std"] == pytest.approx(0.0)


def test_sentence_length_std_is_population_standard_deviation() -> None:
    features = extract_features("One two. One two three four.")

    assert features["sentence_length_std"] == pytest.approx(1.0)


def test_ngram_diversity_and_entropy_match_hand_computed_values() -> None:
    features = extract_features(TWO_SENTENCES)

    assert features["bigram_diversity"] == pytest.approx(6 / 7)
    assert features["trigram_diversity"] == pytest.approx(1.0)
    assert features["word_entropy"] == pytest.approx(2.5)


def test_part_of_speech_ratios() -> None:
    # Seven tokens: pronoun, adverb, verb, determiner, adjective, noun, full stop.
    features = extract_features("She quickly opened the old door.")

    assert features["pronoun_ratio"] == pytest.approx(1 / 7)
    assert features["adverb_ratio"] == pytest.approx(1 / 7)
    assert features["verb_ratio"] == pytest.approx(1 / 7)
    assert features["adjective_ratio"] == pytest.approx(1 / 7)
    assert features["noun_ratio"] == pytest.approx(1 / 7)


def test_readability_scores_separate_simple_from_dense_text() -> None:
    simple = extract_features("The cat sat on the mat. The dog ran to the park.")
    dense = extract_features(
        "Institutional accountability necessitates comprehensive regulatory "
        "frameworks governing multinational pharmaceutical conglomerates."
    )

    assert simple["flesch_reading_ease"] > dense["flesch_reading_ease"]
    assert simple["flesch_kincaid_grade"] < dense["flesch_kincaid_grade"]


def test_empty_text_yields_all_zero_features() -> None:
    features = extract_features("")

    assert features == dict.fromkeys(FEATURE_NAMES, 0.0)


# The next two tests pin quirks inherited from the original pipeline. The
# model was trained with them, so they stay until a retrain removes them.


def test_three_features_are_the_same_ratio() -> None:
    features = extract_features(TWO_SENTENCES)

    assert features["lexical_diversity"] == features["vocabulary_richness"]
    assert features["unique_word_ratio"] == features["vocabulary_richness"]


def test_paragraph_count_is_one_because_cleaning_collapses_blank_lines() -> None:
    features = extract_features("First paragraph.\n\nSecond paragraph.\n\nThird.")

    assert features["paragraph_count"] == 1

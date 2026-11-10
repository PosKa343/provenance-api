"""Turn a text into the numeric features the classifier was trained on."""

from dataclasses import dataclass

FEATURE_NAMES: tuple[str, ...] = (
    "avg_word_length",
    "avg_sentence_length",
    "vocabulary_richness",
    "unique_word_count",
    "lexical_diversity",
    "punctuation_ratio",
    "comma_count",
    "period_count",
    "exclamation_count",
    "question_count",
    "semicolon_count",
    "colon_count",
    "sentence_length_std",
    "paragraph_count",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "automated_readability_index",
    "dale_chall_readability",
    "gunning_fog",
    "smog_index",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "bigram_diversity",
    "trigram_diversity",
    "word_entropy",
    "unique_word_ratio",
)


@dataclass(frozen=True)
class PreprocessedText:
    normalized: str
    tokens: tuple[str, ...]
    sentences: tuple[str, ...]


def preprocess(text: str) -> PreprocessedText:
    raise NotImplementedError


def extract_features(text: str) -> dict[str, float]:
    raise NotImplementedError

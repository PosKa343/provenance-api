"""Turn a text into the numeric features the classifier was trained on."""

import inspect
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

import nltk
import textstat

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


# Order matters: the two whole-word forms must be replaced before the generic "n't".
_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    ("won't", "will not"),
    ("can't", "cannot"),
    ("n't", " not"),
    ("'re", " are"),
    ("'ve", " have"),
    ("'ll", " will"),
    ("'d", " would"),
    ("'m", " am"),
)
_PUNCTUATION_MARKS = frozenset(".,!?;:'\"")
_NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})
_VERB_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})
_ADJECTIVE_TAGS = frozenset({"JJ", "JJR", "JJS"})
_ADVERB_TAGS = frozenset({"RB", "RBR", "RBS"})
_PRONOUN_TAGS = frozenset({"PRP", "PRP$", "WP", "WP$"})


def preprocess(text: str) -> PreprocessedText:
    normalized = _normalize(_clean(text))
    return PreprocessedText(
        normalized=normalized,
        tokens=tuple(nltk.word_tokenize(normalized)),
        sentences=tuple(nltk.sent_tokenize(normalized)),
    )


def extract_features(text: str) -> dict[str, float]:
    pre = preprocess(text)
    try:
        features = {
            **_statistical_features(pre),
            **_readability_features(pre.normalized),
            **_part_of_speech_ratios(pre.tokens),
            **_ngram_features(pre.tokens),
        }
    finally:
        _forget_text()
    return {name: float(features[name]) for name in FEATURE_NAMES}


def _textstat_caches_keyed_by_text() -> list[Any]:
    """textstat's result caches whose key is a whole text.

    textstat remembers its last 128 results per function. That is useful within
    one text, where six scores share intermediate counts, but it would keep
    submitted texts in memory afterwards. Caches keyed by a single word or by
    nothing (the pronunciation dictionary) are left alone.
    """
    caches = []
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("textstat."):
            continue
        for candidate in vars(module).values():
            wrapped = getattr(candidate, "__wrapped__", None)
            if wrapped is None or not hasattr(candidate, "cache_clear"):
                continue
            if next(iter(inspect.signature(wrapped).parameters), None) == "text":
                caches.append(candidate)
    return list({id(cache): cache for cache in caches}.values())


_TEXT_CACHES = _textstat_caches_keyed_by_text()


def _forget_text() -> None:
    for cache in _TEXT_CACHES:
        cache.cache_clear()


def _clean(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s.,!?;:\'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    for contraction, expansion in _CONTRACTIONS:
        text = text.replace(contraction, expansion)
    return text


def _ratio(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _statistical_features(pre: PreprocessedText) -> dict[str, float]:
    text, tokens, sentences = pre.normalized, pre.tokens, pre.sentences
    unique_words = len(set(tokens))
    unique_ratio = _ratio(unique_words, len(tokens))
    words_per_sentence = [len(sentence.split()) for sentence in sentences]
    return {
        "avg_word_length": _ratio(sum(len(token) for token in tokens), len(tokens)),
        "avg_sentence_length": _ratio(len(tokens), len(sentences)),
        "vocabulary_richness": unique_ratio,
        "unique_word_count": unique_words,
        "lexical_diversity": unique_ratio,
        "unique_word_ratio": unique_ratio,
        "punctuation_ratio": _ratio(
            sum(1 for char in text if char in _PUNCTUATION_MARKS), len(text)
        ),
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "sentence_length_std": pstdev(words_per_sentence) if len(sentences) > 1 else 0.0,
        "paragraph_count": sum(1 for paragraph in text.split("\n\n") if paragraph.strip()),
    }


def _readability_features(text: str) -> dict[str, float]:
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability": textstat.dale_chall_readability_score(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
    }


def _part_of_speech_ratios(tokens: tuple[str, ...]) -> dict[str, float]:
    tag_counts = Counter(tag for _, tag in nltk.pos_tag(list(tokens)))

    def ratio(tags: frozenset[str]) -> float:
        return _ratio(sum(tag_counts[tag] for tag in tags), len(tokens))

    return {
        "noun_ratio": ratio(_NOUN_TAGS),
        "verb_ratio": ratio(_VERB_TAGS),
        "adjective_ratio": ratio(_ADJECTIVE_TAGS),
        "adverb_ratio": ratio(_ADVERB_TAGS),
        "pronoun_ratio": ratio(_PRONOUN_TAGS),
    }


def _ngram_features(tokens: tuple[str, ...]) -> dict[str, float]:
    return {
        "bigram_diversity": _ngram_diversity(tokens, 2),
        "trigram_diversity": _ngram_diversity(tokens, 3),
        "word_entropy": _word_entropy(tokens),
    }


def _ngram_diversity(tokens: tuple[str, ...], size: int) -> float:
    ngrams = [tokens[start : start + size] for start in range(len(tokens) - size + 1)]
    return _ratio(len(set(ngrams)), len(ngrams))


def _word_entropy(tokens: tuple[str, ...]) -> float:
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total) for count in Counter(tokens).values())

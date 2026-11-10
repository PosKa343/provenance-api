"""The NLTK data files feature extraction needs, and a command to fetch them.

Usage: python -m provenance_api.ml.nltk_data [target_directory]
"""

import sys

import nltk

# Lookup path -> downloader package name. cmudict is used by textstat to count
# syllables; textstat tries to download it on first use if it is missing.
REQUIRED_RESOURCES: dict[str, str] = {
    "tokenizers/punkt_tab": "punkt_tab",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "corpora/cmudict": "cmudict",
}


class MissingNltkDataError(RuntimeError):
    pass


def ensure_available() -> None:
    """Fail at startup, not on the first request, if any data file is missing."""
    missing = [package for path, package in REQUIRED_RESOURCES.items() if not _is_installed(path)]
    if missing:
        raise MissingNltkDataError(
            f"Missing NLTK data: {', '.join(missing)}. Run: python -m provenance_api.ml.nltk_data"
        )


def _is_installed(path: str) -> bool:
    try:
        nltk.data.find(path)
    except LookupError:
        return False
    return True


def download(target_directory: str | None = None) -> None:
    for package in REQUIRED_RESOURCES.values():
        if not nltk.download(package, download_dir=target_directory, quiet=True):
            raise MissingNltkDataError(f"Could not download NLTK package {package}.")


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else None)

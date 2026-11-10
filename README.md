# Provenance API

A Python service that will serve an AI-generated-text classifier behind an
authenticated, rate-limited REST API with background batch jobs.

**Status:** the model module (`provenance_api.ml`) works and is tested. No HTTP
endpoints are implemented yet.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # create .venv and install dependencies
uv run python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
uv run pre-commit install    # run checks on every commit
uv run ruff check .          # lint
uv run ruff format --check . # format check
uv run mypy                  # type check (strict)
uv run pytest                # tests
```

## Model provenance

The classifier is a **retrain** of my earlier AI-generated-text detection
project, not the original trained model, which was never saved. The feature
extraction in `src/provenance_api/ml/features.py` is a port of the original
code and gives the same 29 feature values on every row of the dataset (largest
difference 8.7e-14). The dataset is in `data/`, with its sources and license.

Retrain with `uv run python -m provenance_api.ml.train`. It writes the model
and a `model.json` with the measured metrics to
`src/provenance_api/ml/artifacts/`.

| | 5-fold CV accuracy | Holdout accuracy (127 rows) |
|---|---|---|
| Original notebook (2025) | 0.9763 ± 0.0133 | 0.9921 |
| This retrain (2026-11-09) | 0.9731 ± 0.0095 | 0.9921 |

Limits:

- AI-text detection is unreliable. Do not use this model to accuse or
  penalise anyone.
- The data has one source per class (Wikipedia for human, Gemini Flash for
  AI), so the model may have learned those two styles and nothing more
  general. The accuracy above says nothing about other kinds of text.
- Training texts were 300 to 1000 words long. Several features are raw counts,
  so scores for much shorter or longer texts are not trustworthy.

## License

MIT. See [LICENSE](LICENSE).

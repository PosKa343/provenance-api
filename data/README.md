# Training data

`ai_detection_dataset.csv` is the dataset from my original AI-generated-text
detection project. It is copied here unchanged so the model can be retrained
and its reported metrics checked.

- 632 rows, balanced: 316 human (`label` 0) and 316 AI (`label` 1).
- Human texts are English Wikipedia article extracts fetched through the
  Wikipedia API. Each row keeps its `title` and `url`. Wikipedia text is
  licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
  and that license, not this repository's MIT license, applies to those rows.
- AI texts were generated with Google Gemini Flash through the Gemini API.
  Each row keeps the `prompt` and `topic` used.
- SHA-256: `1eca724405d9eeb1379ef4a2fa28a94ac2a38dc265ac29091d02116c5f4415c9`

Known limit: each class comes from a single source, so a model trained on
this data may be separating "Wikipedia style" from "Gemini style" rather than
human from AI writing in general.

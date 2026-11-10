# Build stage: install dependencies and the package into a virtualenv.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so this layer is reused until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Bake the NLTK data into the image so the container needs no network at runtime.
RUN /app/.venv/bin/provenance-api nltk-data /app/nltk_data


# Runtime stage: only the virtualenv, NLTK data and migrations. No uv, no build tools.
FROM python:3.12-slim
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/nltk_data /app/nltk_data
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH="/app/.venv/bin:$PATH" NLTK_DATA=/app/nltk_data PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["uvicorn", "provenance_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

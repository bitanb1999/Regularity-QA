# One image for both services: `api` (uvicorn, default) and `ui` (streamlit), see docker-compose.yml.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/models \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Non-root user created up front so models are downloaded as that user (a later chown would copy
# the whole model directory into a new layer).
RUN useradd --create-home --uid 1000 rqa \
    && mkdir -p /app/data /opt/models \
    && chown rqa /app/data /opt/models

# Dependencies first, so code changes don't reinstall them. On Linux, torch resolves to the
# CPU-only build (see [tool.uv.sources] in pyproject.toml).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Bake the embedding and reranker models in, so containers start without network access.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
USER rqa
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANK_MODEL}')"
USER root

COPY app ./app
COPY ui ./ui
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

USER rqa
ENV HF_HUB_OFFLINE=1

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

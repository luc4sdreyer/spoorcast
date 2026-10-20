# Spoorcast prediction API.
#
#   docker build -t spoorcast-api .
#   docker run -p 8000:8000 \
#     -v "$PWD/models/latest:/models/model:ro" \
#     -v "$PWD/data/processed/feature_snapshot:/models/feature_snapshot:ro" \
#     spoorcast-api

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 spoorcast
COPY --from=build --chown=spoorcast:spoorcast /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SPOORCAST_MODEL_DIR=/models/model \
    SPOORCAST_SNAPSHOT_DIR=/models/feature_snapshot \
    SPOORCAST_DATA_DIR=/data
USER spoorcast
WORKDIR /home/spoorcast
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "spoorcast.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

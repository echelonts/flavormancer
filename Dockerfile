# Flavormancer — the on-prem prediction service.
#
# Deliberately a RUNTIME image, not a training one. Model training needs a GPU box, 32 cores and
# several hours; serving needs none of that. Trained artifacts (aroma_models/, taste_models/,
# mouthfeel_models/, tox_models/ and the parquet tables) are mounted at run time rather than baked
# in — they are ~1 GB, they change on every retrain, and burning them into a layer would make the
# image both enormous and stale the moment a head is retrained.
#
#   docker build -t flavormancer:latest .
#   docker compose up          # see docker-compose.yml for the volume wiring
#
# RDKit is the reason for the slim-bookworm base rather than alpine: it ships manylinux wheels
# that need glibc, and building it from source on musl is hours of pain for no benefit.

FROM python:3.12-slim-bookworm AS base

# libxrender/libxext are RDKit's molecule-drawing dependencies (the structure SVGs); libgomp is
# OpenMP, which scikit-learn's forests use for parallel predict_proba.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FLAVORMANCER_HOME=/app/data

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the source, so an app
# edit rebuilds in seconds instead of reinstalling RDKit.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi" "uvicorn[standard]" pydantic

COPY training/ ./training/

# Serving runs as an unprivileged user. The mounted model directory only needs to be readable.
RUN useradd --create-home --shell /usr/sbin/nologin flavormancer \
    && mkdir -p /app/data && chown -R flavormancer:flavormancer /app
USER flavormancer

WORKDIR /app/training
EXPOSE 8000

# /healthz answers before the models finish loading (the app serves a warming page meanwhile), so
# a long start-period is what keeps the container from being killed during a legitimate ~50s
# cold start. See docs/METHODS.md on why loading is serial.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=4 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

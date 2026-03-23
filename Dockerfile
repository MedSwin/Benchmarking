# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime paths (override in compose if needed)
ENV DATA_ROOT=/app/data \
    OUTPUT_ROOT=/app/runs

WORKDIR /app

RUN addgroup --system --gid 1001 app \
    && adduser --system --uid 1001 --ingroup app app

# Install CPU torch first so pip does not pull the default CUDA wheel from PyPI.
RUN pip install --no-cache-dir "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --no-cache-dir .

# Writable job output
RUN mkdir -p /app/runs && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

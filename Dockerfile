# BeevR for Legal — app/API image (doc 20 §1: Compose-first packaging).
# Base tier: API + ingestion + verification logic; runs in stub-model
# (degraded) mode without a GPU — /readyz reports models:"stub" (SRS §3.1.2).
# The model tier installs the `models` extra on the GPU node (doc 14 §4).
FROM python:3.12-slim AS base

# no root; no build tools left in the runtime layer
RUN useradd --create-home --uid 10001 beevr
WORKDIR /app

COPY pyproject.toml README.md* ./
COPY src ./src
COPY db ./db
COPY scripts ./scripts

RUN pip install --no-cache-dir . && pip install --no-cache-dir ".[db]"

USER beevr
EXPOSE 8777

# Healthcheck hits the no-customer-data probe (doc 11 §4)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8777/readyz').raise_for_status()"

CMD ["python", "-m", "uvicorn", "beevr.api:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8777"]

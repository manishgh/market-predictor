FROM python:3.11.15-slim-trixie@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

ARG SOURCE_REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/manishgh/market-predictor" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.title="market-predictor-production"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV PYTHONPATH=/app/src
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV RUNTIME_MEMORY_BUDGET_GIB=4.0
ENV RUNTIME_MEMORY_HEADROOM_GIB=0.25

WORKDIR /app

COPY requirements/production.lock /tmp/production.lock
RUN python -m pip install --require-hashes --no-deps -r /tmp/production.lock \
    && rm /tmp/production.lock \
    && groupadd --system --gid 10001 market-predictor \
    && useradd --system --uid 10001 --gid market-predictor --home-dir /app market-predictor \
    && mkdir -p /app/data /app/models \
    && chown -R market-predictor:market-predictor /app/data /app/models

COPY src ./src
COPY configs ./configs

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health/live', timeout=3).read()"]

CMD ["python", "-m", "uvicorn", "market_predictor.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

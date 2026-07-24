FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV RUNTIME_MEMORY_BUDGET_GIB=4.0
ENV RUNTIME_MEMORY_HEADROOM_GIB=0.25

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --upgrade pip \
    && pip install . \
    && apt-get purge -y --auto-remove build-essential \
    && groupadd --system --gid 10001 market-predictor \
    && useradd --system --uid 10001 --gid market-predictor --home-dir /app market-predictor \
    && mkdir -p /app/data /app/models \
    && chown -R market-predictor:market-predictor /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health/live', timeout=3).read()"]

CMD ["sh", "scripts/container-entrypoint.sh"]

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY templates ./templates
COPY site ./site
COPY scripts ./scripts
COPY README.md ./README.md

RUN mkdir -p data/cache outputs site/races /tmp/matplotlib \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /tmp/matplotlib

USER appuser

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD []

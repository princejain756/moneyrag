FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed

WORKDIR /app

# Dependency layer first, so editing source does not reinstall packages.
COPY pyproject.toml README.md ./
COPY moneyrag ./moneyrag
RUN pip install ".[server]" \
 && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
 && chmod -R a+rX /opt/fastembed \
 && useradd --create-home --uid 10001 appuser

COPY data ./data
RUN mkdir -p /app/.state && chown -R appuser:appuser /app/.state

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=90s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "moneyrag.service:app", "--host", "0.0.0.0", "--port", "8000"]

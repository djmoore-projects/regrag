FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY regrag ./regrag
RUN pip install --no-cache-dir .

# Bake the embedding + reranker ONNX models into the image so cold starts don't download.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-base-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-12-v2')"

RUN useradd --create-home app && chown -R app /app /models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "regrag.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

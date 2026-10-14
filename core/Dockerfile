FROM python:3.11-slim

WORKDIR /app

# Point specifically to core/requirements.txt — build context is the repo root (see
# docker-compose.yml's `context: .` and Railway's root directory), not core/ itself.
COPY core/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# R5.01: bake the cross-encoder reranker into the image so a live container never downloads
# it (no HuggingFace dependency at runtime, no cold first call). Must match the
# RETRIEVAL_RERANK_MODEL / _REVISION / _ONNX_FILE defaults in app/config.py — change together.
# Placed BEFORE `COPY . .` so a code change does not re-download the model on every build.
# Soft-fail on purpose: the reranker is optional, and a HuggingFace outage must not block a
# deploy. A missing model shows up as "unavailable" in GET /knowledge/reranker and retrieval
# keeps its RRF order (or, with RETRIEVAL_RERANK_ALLOW_DOWNLOAD=true, fetches it in the
# background at startup).
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
ARG RERANK_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a
ARG RERANK_ONNX_FILE=onnx/model_quint8_avx2.onnx
ENV RETRIEVAL_RERANK_CACHE_DIR=/opt/models/hf
RUN python -c "from huggingface_hub import hf_hub_download as d; \
[print('baked', d(repo_id='${RERANK_MODEL}', filename=f, revision='${RERANK_REVISION}', cache_dir='/opt/models/hf')) \
 for f in ('${RERANK_ONNX_FILE}', 'tokenizer.json', 'config.json')]" \
    || echo "WARNING: reranker model NOT baked into the image (download failed) -- see R5.01 notes in core/Dockerfile"

# Copy all code
COPY . .

WORKDIR /app/core

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face Space (Docker SDK) image for the CurriculumGPT backend.
#
# This is a sibling of backend/Dockerfile. Keep the two in sync when backend
# dependencies or the start command change. Differences from backend/Dockerfile:
#   • runs as the non-root uid 1000 that HF Spaces expects
#   • bakes BAAI/bge-large-en-v1.5 into the image so cold starts don't
#     re-download ~1.3 GB every time the Space wakes from sleep
#   • listens on 7860 (HF's default app_port; also set in .hf/README.md)
#
# The GitHub Action .github/workflows/deploy-hf-space.yml copies this file to
# the Space repo root as `Dockerfile` along with backend/, ml/, and the two
# data/ subdirs the app needs, so the COPY paths below resolve.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    HF_HOME=/home/user/.cache/huggingface \
    MODEL_CACHE_DIR=/home/user/.cache/model_cache \
    MODEL_TYPE=bge \
    BGE_MODEL_NAME=BAAI/bge-large-en-v1.5 \
    DEVICE=cpu \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc g++ && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so the -r requirements.txt step below reuses it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY ml/ ./ml/
COPY data/mappings/ ./data/mappings/
COPY data/processed/ ./data/processed/

# Pre-download the embedding model into the exact cache dir the app reads
# (AlignmentModel passes cache_dir=$MODEL_CACHE_DIR to from_pretrained).
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-large-en-v1.5', cache_dir='/home/user/.cache/model_cache')" \
    && mkdir -p /home/user/.cache/model_cache \
    && chown -R user:user /app /home/user

USER user

EXPOSE 7860
WORKDIR /app/backend
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${WORKERS:-1}"]

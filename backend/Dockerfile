FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc g++ && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to avoid downloading 2GB of CUDA/GPU packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch already satisfied, skipped)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code the backend needs
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY data/mappings/ ./data/mappings/
COPY data/processed/ ./data/processed/

# ml/ and data/ are importable from /app
ENV PYTHONPATH=/app

WORKDIR /app/backend

EXPOSE 7000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 7000 --workers 1"]

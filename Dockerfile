FROM python:3.10-slim

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch!)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pillow \
    python-multipart \
    sentencepiece \
    transformers \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy app
COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

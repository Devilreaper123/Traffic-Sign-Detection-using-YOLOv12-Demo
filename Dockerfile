# ---- base ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps required by opencv-python and pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install deps first (better layer caching)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy app code
COPY src ./src
COPY models ./models

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Optional: healthcheck using /healthz
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"


# Default: start uvicorn
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# Backend image for the N-body FastAPI service.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Only the engine and the API are needed at runtime.
COPY core ./core
COPY backend ./backend

EXPOSE 8000

# $PORT is provided by most platforms (Render, Railway, ...); default to 8000.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_MODE=real \
    DEMO_MODE=off

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
# Install deps — torch CPU-only to keep image size manageable on free plan
RUN sed '/^torch[<>=]/d' /app/backend/requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2.0' \
    && rm /tmp/requirements-no-torch.txt

COPY backend /app/backend
COPY config /app/config

# Create empty model directories so the app starts cleanly without ML artifacts.
# ML prediction endpoints will report "model unavailable" rather than crashing.
RUN mkdir -p /app/backend/models/sea_ice \
    && mkdir -p /app/backend/models/iceberg \
    && mkdir -p "/app/Real data/processed"

WORKDIR /app/backend
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]

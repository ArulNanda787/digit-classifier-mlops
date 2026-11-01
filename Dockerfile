FROM python:3.11-slim

# Cleaner logs, no .pyc files, quieter TensorFlow
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilds
# when requirements-serve.txt changes, not on every code edit.
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# App code and the trained model are baked into the image.
# Don't run as root inside the container.
RUN useradd --create-home appuser

# owned by appuser so it can read them regardless of host permissions.
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser models/digit_cnn.keras models/digit_cnn.keras

USER appuser

EXPOSE 8000

# Docker marks the container "unhealthy" if the model didn't load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import json,sys,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); sys.exit(0 if r['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
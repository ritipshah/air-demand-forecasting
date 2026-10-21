# AeroCast — Streamlit dashboard image.
# Deployment contract: Streamlit on port 8501, GET /_stcore/health -> "ok",
# PROJECT_COURSE env var selects the team shown in the app.
FROM python:3.12-slim

# Don't write .pyc files, flush logs immediately, keep pip quiet.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user (never as root inside the container).
RUN useradd --create-home --uid 1000 appuser

# Only what the app needs at runtime.
COPY --chown=appuser:appuser airdemand/ airdemand/
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser results/ results/
COPY --chown=appuser:appuser .streamlit/ .streamlit/

USER appuser

ENV PROJECT_COURSE=vcc
EXPOSE 8501

# Slim images have no curl, so use Python's urllib for the health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).read().strip()==b'ok' else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

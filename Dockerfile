FROM python:3.11-slim

LABEL maintainer="openenv-submission"
LABEL description="Email Triage & Response OpenEnv"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files from repo root
COPY app.py .
COPY models.py .
COPY environment.py .
COPY openenv.yaml .
COPY inference.py .
COPY __init__.py .

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app.py"]

# ==============================================================================
# SIH26055 — Electronic Warfare Smart Scan Strategy (V4.1 Production Container)
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and production assets
COPY app/ app/
COPY rf_environment/ rf_environment/
COPY benchmarks/ benchmarks/
COPY docs/ docs/
COPY pyproject.toml .
COPY README.md .
COPY run_tests.py .

# Create non-root system user for security compliance
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check against production /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Production startup entrypoint
CMD ["python", "-m", "app", "serve", "--host", "0.0.0.0", "--port", "8000"]

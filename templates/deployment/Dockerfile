# Production Dockerfile for RAG System

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p cache data logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${WORKER_PORT:-8000}/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "src.main"]

# Production image
FROM base as production

# Copy additional production files
COPY templates/deployment/entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER appuser

ENTRYPOINT ["/entrypoint.sh"]

# Development image
FROM base as development

# Install development dependencies
USER root
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
USER appuser

# Override command for development
CMD ["python", "-m", "src.main", "--dev"]
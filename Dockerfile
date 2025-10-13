# Multi-stage Dockerfile for FRA Diagnostics Platform
# SIH 2025 PS 25190
# syntax=docker/dockerfile:1.4

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# SECURITY: Create non-root user for build stage
RUN useradd -m -u 1000 -s /bin/bash builduser && \
    chown -R builduser:builduser /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY --chown=builduser:builduser requirements.txt .

# Switch to non-root user for dependency installation
USER builduser

# OPTIMIZATION: Create virtual environment and install dependencies with cache mount
RUN python -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"

# Use BuildKit cache mount for faster pip installs
RUN --mount=type=cache,target=/home/builduser/.cache/pip,uid=1000 \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Set labels
LABEL maintainer="SIH 2025 Team"
LABEL description="Transformer FRA Diagnostics Platform"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FRA_LOG_DIR=/app/logs \
    FRA_MODEL_DIR=/app/models \
    FRA_TEMP_DIR=/app/temp \
    FRA_DATA_DIR=/app/data \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies (minimal set for production)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for runtime
RUN useradd -m -u 1000 -s /bin/bash frauser && \
    mkdir -p /app/logs /app/models /app/temp /app/data && \
    chown -R frauser:frauser /app

# Copy virtual environment from builder (with proper ownership)
COPY --from=builder --chown=frauser:frauser /build/venv /opt/venv

# Copy application code
COPY --chown=frauser:frauser . .

# Switch to non-root user
USER frauser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

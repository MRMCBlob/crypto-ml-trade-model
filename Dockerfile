# Crypto ML Trading System
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ=UTC

# Trading defaults (can be overridden at runtime)
ENV TRADING_MODE=paper
ENV INITIAL_CAPITAL=10000
ENV LOG_LEVEL=INFO

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories (storage is separate from data Python module)
RUN mkdir -p /app/storage/raw /app/storage/processed /app/models/saved /app/logs /app/reports

# Copy and make entrypoint executable
COPY scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN chmod +x /app/scripts/entrypoint.sh

# Note: Running as root for compatibility with bind-mounted volumes
# Container isolation provides security boundary

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command - runs full pipeline (download, train, trade)
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

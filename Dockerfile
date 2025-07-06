# DocuLLaMA Dockerfile - Azure Container Apps Optimized
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    WORKERS=4 \
    TIMEOUT=300

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    unzip \
    default-jre \
    libreoffice \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Apache Tika
RUN wget -O /opt/tika-server.jar https://archive.apache.org/dist/tika/2.9.0/tika-server-standard-2.9.0.jar

# Create non-root user
RUN groupadd -r docullama && useradd -r -g docullama docullama

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/cache && \
    chown -R docullama:docullama /app

# Switch to non-root user
USER docullama

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Start command optimized for Azure Container Apps
CMD ["sh", "-c", "java -jar /opt/tika-server.jar --host=0.0.0.0 --port=9998 & python -m uvicorn app:app --host=0.0.0.0 --port=${PORT} --workers=${WORKERS} --timeout-keep-alive=${TIMEOUT}"]
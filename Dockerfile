# Neo4j Text2SQL Dockerfile
# FastAPI + LangChain

FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (including build tools for numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install Python dependencies
RUN uv pip install --system --no-cache -r pyproject.toml

# Reinstall numpy 1.x for legacy CPU compatibility (numpy 2.x requires X86_V2)
RUN pip uninstall -y numpy && pip install --no-cache-dir "numpy<2"

# Copy application code
COPY app ./app
COPY main.py .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

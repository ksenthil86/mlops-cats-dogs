FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Set environment variables
ENV MODEL_PATH=/app/models/cats_dogs_cnn.pkl
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/app:/app/src
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the inference service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

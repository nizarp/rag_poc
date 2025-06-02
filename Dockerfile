FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    beautifulsoup4 \
    faiss-cpu \
    llama-index \
    huggingface-hub \
    llama-cpp-python

# Expose port
EXPOSE 8000

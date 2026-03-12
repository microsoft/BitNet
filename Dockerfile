# syntax=docker/dockerfile:1
# BitNet Dockerfile - CPU inference for 1-bit LLMs
# Build: docker build -t bitnet .
# Run: docker run --rm -it bitnet python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "Your prompt" -cnv

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone BitNet repository (for builds from source)
# If building from local source, copy files instead
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Build the project (optional - can be done at runtime with setup_env.py)
# This creates the native extensions needed for inference
RUN git submodule update --init --recursive 2>/dev/null || true

# Create models directory
RUN mkdir -p models

# Default command shows help
CMD ["python", "run_inference.py", "--help"]

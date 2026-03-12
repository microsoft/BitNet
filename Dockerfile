# BitNet.cpp Docker Image
# For running 1-bit LLM inference on CPU

FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Initialize git submodules
RUN git submodule update --init --recursive

# Build the project
RUN cmake -B build && cmake --build build --config Release

# Set environment variables
ENV PYTHONPATH=/app/build/bin:$PYTHONPATH

# Default command - show help
CMD ["python3", "run_inference.py", "--help"]

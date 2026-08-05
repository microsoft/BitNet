FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

ARG MODEL=BitNet-b1.58-2B-4T
ARG MODEL_REPO=microsoft/${MODEL}-gguf
ARG MODEL_DIR=models/${MODEL}
ARG QUANT_TYPE=i2_s

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    gnupg \
    python3 \
    python3-pip \
    python3-venv \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" -- 18 && \
    apt-get update && apt-get install -y --no-install-recommends \
    clang-18 \
    lld-18 \
    lldb-18 \
    && ln -sf /usr/bin/clang-18 /usr/bin/clang \
    && ln -sf /usr/bin/clang++-18 /usr/bin/clang++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/BitNet

COPY . .

RUN python3 -m venv /opt/bitnet-venv

ENV PATH="/opt/bitnet-venv/bin:${PATH}" \
    CC=/usr/bin/clang-18 \
    CXX=/usr/bin/clang++-18

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -U "huggingface_hub[cli]"

# Patch the upstream source before compiling to avoid the known constness issue.
RUN sed -i 's/int8_t \* y_col = y + col \* by;/const int8_t * y_col = y + col * by;/' src/ggml-bitnet-mad.cpp && \
    hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}" && \
    python setup_env.py -md "${MODEL_DIR}" -q "${QUANT_TYPE}"

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/BitNet

COPY --from=builder /app/BitNet/build ./build
COPY --from=builder /app/BitNet/models ./models
COPY --from=builder /app/BitNet/run_inference.py ./run_inference.py
COPY --from=builder /app/BitNet/run_inference_server.py ./run_inference_server.py

CMD ["/bin/bash"]

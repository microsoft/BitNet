# ────────────────
# Stage 1: builder
# ────────────────
FROM python:3.10-slim AS builder

# install system deps: git, wget, ca-certificates, plus add LLVM repo for clang-18
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git wget ca-certificates lsb-release gnupg && \
    rm -rf /var/lib/apt/lists/* && \
    # add LLVM apt repo and install clang-18 + cmake
    wget -O - https://apt.llvm.org/llvm.sh | bash && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      clang-18 cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# clone BitNet (with submodules)
RUN git clone --recursive https://github.com/microsoft/BitNet.git . 

# install Python requirements + CLI for huggingface-hub
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface-hub[cli]"

# download the model into the repo
RUN huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
     --local-dir models/BitNet-b1.58-2B-4T

# build the C++ inference binary and set up env
RUN python setup_env.py \
      -md models/BitNet-b1.58-2B-4T \
      -q i2_s

# ────────────────
# Stage 2: runtime
# ────────────────
FROM python:3.10-slim

WORKDIR /app

# copy the compiled binary + model
COPY --from=builder /src/build       ./build
COPY --from=builder /src/models      ./models

# copy your Python inference wrapper + FastAPI app
COPY run_inference_api.py app.py        ./

# install only what’s needed at runtime
RUN pip install --no-cache-dir \
      fastapi uvicorn

# expose and launch
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


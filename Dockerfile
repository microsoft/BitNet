# ────────────────
# Stage 1: builder
# ────────────────
FROM python:3.10-slim AS builder


# install system deps: git, wget, curl, ca-certificates, lsb-release, software-properties-common, gnupg
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git \
  wget \
  curl \
  ca-certificates \
  lsb-release \
  software-properties-common \
  gnupg \
  build-essential \
  clang \
  && rm -rf /var/lib/apt/lists/*

# add LLVM apt repo and install clang-18 + cmake
RUN wget -O - https://apt.llvm.org/llvm.sh | bash -s -- 18 && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  cmake \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# clone BitNet (with submodules)
RUN git clone --recursive https://github.com/microsoft/BitNet.git . 

# install Python requirements + CLI for huggingface-hub
RUN pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir "huggingface-hub[cli]"

# download the model into the repo
#RUN huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
#  --local-dir models/BitNet-b1.58-2B-4T

COPY models      ./models
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

COPY run_inference_server.py        ./

# expose and launch
EXPOSE 11435


CMD ["python", "run_inference_server.py", "--host", "0.0.0.0", "--port", "11435"]

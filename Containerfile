FROM mcr.microsoft.com/azurelinux/base/python:3 AS builder

USER 0
RUN tdnf --setopt=install_weak_deps=False install -y ca-certificates git git-lfs make cmake gcc gcc-c++ glibc-devel libgomp-devel binutils shadow-utils kernel-headers
RUN useradd -m -d /bitnet -g nogroup bitnet

USER bitnet
RUN git clone --single-branch --depth=1 --branch=main https://github.com/microsoft/BitNet.git /bitnet/bitnet

WORKDIR /bitnet/bitnet
RUN git submodule add -f https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf models/BitNet-b1.58-2B-4T
RUN git submodule update --init --recursive
# Fixes
# Fix for https://github.com/microsoft/BitNet/pull/418
RUN curl -fsSL https://github.com/darsh7807/BitNet/commit/597508f3a01e917988f0d45302e944d5192f4852.patch | git apply
## Workarround for https://github.com/microsoft/azurelinux/issues/16192
# End of fixes

ENV PATH=/bitnet/build/bin:/bitnet/.local/bin:${PATH}
RUN mkdir logs
RUN python3 -m pip install --user -r requirements.txt
#Replacing due --user: RUN python3 -c "import setup_env; setup_env.args = setup_env.parse_args() ; setup_env.setup_gguf()"
RUN python3 -m pip install --user 3rdparty/llama.cpp/gguf-py
RUN python3 -c "import sys, setup_env; sys.argv = ['setup_env', '--hf-repo', 'microsoft/BitNet-b1.58-2B-4T'] ; setup_env.args = setup_env.parse_args() ; setup_env.gen_code()"
# Patch for GCC
RUN HEADER=include/bitnet-lut-kernels.h && \
    # Bug 1: fix vec_zero declaration type
    sed -i 's/const int8x16_t vec_zero = vdupq_n_s16/const int16x8_t vec_zero = vdupq_n_s16/g' "${HEADER}" && \
    # Bug 2: wrap .val[X] accumulations into vec_c with vreinterpretq_s16_s8
    sed -i 's/vec_c\[\([^]]*\)\] += \(vec_v_[a-z_]*[0-9]*\.val\[[01]\]\);/vec_c[\1] += vreinterpretq_s16_s8(\2);/g' "${HEADER}" && \
    echo "Patched ${HEADER} successfully" && \
    # Verify the bad patterns are gone
    ! grep -n 'const int8x16_t vec_zero' "${HEADER}" && \
    echo "Verification passed"
# End of patch
RUN set -eux; \
    ARCH="$(uname -m)"; \
    echo "Detected architecture: ${ARCH}"; \
    case "${ARCH}" in \
      x86_64) \
        cmake -B build \
          -DCMAKE_C_COMPILER=gcc \
          -DCMAKE_CXX_COMPILER=g++ \
          -DCMAKE_BUILD_TYPE=Release \
          -DBITNET_X86_TL2=ON \
	  -DGGML_CCACHE=OFF \
          -DGGML_NATIVE=OFF \
          -DGGML_AVX=ON \
          -DGGML_AVX2=ON \
          -DGGML_FMA=ON \
          -DGGML_F16C=ON \
          -DGGML_AVX512=OFF \
          -DGGML_AVX512_VBMI=OFF \
          -DGGML_AVX512_VNNI=OFF \
          -DGGML_AVX512_BF16=OFF \
          "-DCMAKE_C_FLAGS=-march=x86-64-v3 -mtune=generic" \
          "-DCMAKE_CXX_FLAGS=-march=x86-64-v3 -mtune=generic" \
        ;; \
      aarch64) \
        cmake -B build \
          -DCMAKE_C_COMPILER=gcc \
          -DCMAKE_CXX_COMPILER=g++ \
          -DCMAKE_BUILD_TYPE=Release \
          -DBITNET_ARM_TL1=ON \
	  -DGGML_CCACHE=OFF \
          -DGGML_NATIVE=OFF \
          -DGGML_SVE=OFF \
          "-DCMAKE_C_FLAGS=-march=armv8-a -mtune=generic" \
          "-DCMAKE_CXX_FLAGS=-march=armv8-a -mtune=generic" \
        ;; \
      *) \
        echo "ERROR: Unsupported architecture: ${ARCH}" >&2; exit 1 \
        ;; \
    esac

RUN cmake --build build --config Release -j"$(nproc)"
RUN python3 -c "import sys, setup_env; sys.argv = ['setup_env', '--model-dir', 'models/BitNet-b1.58-2B-4T'] ; setup_env.args = setup_env.parse_args() ; setup_env.prepare_model()" || (cat logs/convert_to_f32_gguf.log && exit 1)
RUN rm -rf logs

RUN find /bitnet/bitnet -type d -name .git -exec rm -rf {} \+
RUN rm -rf models/BitNet-b1.58-2B-4T

FROM mcr.microsoft.com/azurelinux/base/python:3 AS runtime

COPY --from=builder /bitnet /bitnet
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/shadow /etc/shadow

WORKDIR /bitnet/bitnet

USER 0
RUN tdnf --setopt=install_weak_deps=False install -y libgomp tini \
 && tdnf clean all
RUN pip install -r requirements.txt
# && pip install 3rdparty/llama.cpp/gguf-py

USER bitnet
ENV PATH=/bitnet/build/bin:/bitnet/.local/bin:${PATH}

EXPOSE 8080

ENTRYPOINT [ "/usr/bin/tini", \
             "-g", \
	     "--", \
	     "/bin/bash", "-c" \
	     ]

CMD [ "/usr/bin/python3 \
      /bitnet/bitnet/run_inference_server.py \
      -m ${MODEL:-models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf} \
      -c ${CONTEXT_WINDOW:-2048} \
      -t $(nproc --ignore=2) \
      -n ${N_PREDICT:-4096} \
      --temperature ${TEMPERATURE:-0.8} \
      --host ${LISTEN_HOST:-0.0.0.0} \
      --port ${LISTEN_PORT:-8080}" \
      ]

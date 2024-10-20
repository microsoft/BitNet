FROM python:3.9-alpine AS build

RUN apk add --no-cache build-base cmake clang git

RUN git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git /BitNet

WORKDIR /BitNet

RUN pip install --no-cache-dir -r requirements.txt

RUN python3 utils/codegen_tl2.py --model Llama3-8B-1.58-100B-tokens \
    --BM 256,128,256,128 --BK 96,96,96,96 --bm 32,32,32,32

RUN cmake -B build -DBITNET_X86_TL2=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ && \
    cmake --build build --target llama-cli --config Release

FROM python:3.9-alpine

RUN apk add --no-cache libstdc++ libgcc

WORKDIR /BitNet

COPY --from=build /BitNet /BitNet

ADD https://huggingface.co/brunopio/Llama3-8B-1.58-100B-tokens-GGUF/resolve/main/Llama3-8B-1.58-100B-tokens-TQ2_0.gguf .

RUN echo "2565559c82a1d03ecd1101f536c5e99418d07e55a88bd5e391ed734f6b3989ac Llama3-8B-1.58-100B-tokens-TQ2_0.gguf" | sha256sum -c

CMD ["python3", "run_inference.py", "-m", "Llama3-8B-1.58-100B-tokens-TQ2_0.gguf", "-p", "Water boils at 100 degrees Celsius due to the energy required to break the bonds between water molecules."]

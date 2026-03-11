# bitnet.cpp
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)

[<img src="./assets/header_model_release.png" alt="BitNet Model on Hugging Face" width="800"/>](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)

[데모](https://bitnet-demo.azurewebsites.net/)를 통해 직접 체험해 보거나, 자체 [CPU](https://github.com/microsoft/BitNet?tab=readme-ov-file#build-from-source) 또는 [GPU](https://github.com/microsoft/BitNet/blob/main/gpu/README.md)에서 빌드하고 실행해 보세요.

bitnet.cpp는 1비트 LLM(예: BitNet b1.58)을 위한 공식 추론 프레임워크입니다. CPU 및 GPU에서 1.58비트 모델의 **빠르고** **손실 없는** 추론을 지원하는 최적화된 커널 제품군을 제공합니다(NPU 지원은 곧 제공될 예정입니다).

bitnet.cpp의 첫 번째 릴리스는 CPU에서의 추론을 지원하는 것입니다. bitnet.cpp는 ARM CPU에서 **1.37배**에서 **5.07배**의 속도 향상을 달성하며, 모델이 클수록 성능 향상 폭이 커집니다. 또한 에너지 소비를 **55.4%**에서 **70.0%**까지 줄여 전반적인 효율성을 더욱 높입니다. x86 CPU에서는 **71.9%**에서 **82.2%**의 에너지 절감과 함께 **2.37배**에서 **6.17배**의 속도 향상을 보입니다. 또한 bitnet.cpp는 단일 CPU에서 100B BitNet b1.58 모델을 실행하여 사람의 읽기 속도(초당 5-7 토큰)에 필적하는 속도를 달성함으로써 로컬 장치에서 LLM을 실행할 수 있는 잠재력을 크게 향상시킵니다. 자세한 내용은 [기술 보고서](https://arxiv.org/abs/2410.16144)를 참조하십시오.

<img src="./assets/m2_performance.jpg" alt="m2_performance" width="800"/>
<img src="./assets/intel_performance.jpg" alt="intel_performance" width="800"/>

>테스트된 모델은 bitnet.cpp의 추론 성능을 보여주기 위해 연구 환경에서 사용된 더미 설정입니다.

## 데모

Apple M2에서 BitNet b1.58 3B 모델을 실행하는 bitnet.cpp 데모:

https://github.com/user-attachments/assets/7f46b736-edec-4828-b809-4be780a3e5b1

## 새로운 소식:
- 2025년 5월 20일 [BitNet 공식 GPU 추론 커널](https://github.com/microsoft/BitNet/blob/main/gpu/README.md) ![NEW](https://img.shields.io/badge/NEW-red)
- 2025년 4월 14일 [Hugging Face의 BitNet 공식 2B 파라미터 모델](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
- 2025년 2월 18일 [Bitnet.cpp: 3진 LLM을 위한 효율적인 엣지 추론](https://arxiv.org/abs/2502.11880)
- 2024년 11월 8일 [BitNet a4.8: 1비트 LLM을 위한 4비트 활성화](https://arxiv.org/abs/2411.04965)
- 2024년 10월 21일 [1비트 AI 인프라: 파트 1.1, CPU에서의 빠르고 손실 없는 BitNet b1.58 추론](https://arxiv.org/abs/2410.16144)
- 2024년 10월 17일 bitnet.cpp 1.0 출시.
- 2024년 3월 21일 [1비트 LLM의 시대__학습 팁_코드_FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
- 2024년 2월 27일 [1비트 LLM의 시대: 모든 대규모 언어 모델은 1.58비트](https://arxiv.org/abs/2402.17764)
- 2023년 10월 17일 [BitNet: 대규모 언어 모델을 위한 1비트 트랜스포머 확장](https://arxiv.org/abs/2310.11453)

## 감사의 말

이 프로젝트는 [llama.cpp](https://github.com/ggerganov/llama.cpp) 프레임워크를 기반으로 합니다. 오픈 소스 커뮤니티에 기여해주신 모든 저자분들께 감사드립니다. 또한 bitnet.cpp의 커널은 [T-MAC](https://github.com/microsoft/T-MAC/)에서 개척된 조회 테이블 방법론을 기반으로 구축되었습니다. 3진 모델을 넘어선 일반적인 저비트 LLM의 추론에는 T-MAC 사용을 권장합니다.

## 공식 모델
<table>
    </tr>
    <tr>
        <th rowspan="2">모델</th>
        <th rowspan="2">파라미터</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">커널</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-b1.58-2B-4T">BitNet-b1.58-2B-4T</a></td>
        <td rowspan="2">2.4B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>

## 지원되는 모델
❗️**bitnet.cpp의 추론 기능을 보여주기 위해 [Hugging Face](https://huggingface.co/)에서 사용 가능한 기존 1비트 LLM을 사용합니다. bitnet.cpp의 출시가 모델 크기 및 학습 토큰 측면에서 대규모 설정에서 1비트 LLM 개발에 영감을 주기를 바랍니다.**

<table>
    </tr>
    <tr>
        <th rowspan="2">모델</th>
        <th rowspan="2">파라미터</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">커널</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-large">bitnet_b1_58-large</a></td>
        <td rowspan="2">0.7B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">bitnet_b1_58-3B</a></td>
        <td rowspan="2">3.3B</td>
        <td>x86</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens">Llama3-8B-1.58-100B-tokens</a></td>
        <td rowspan="2">8.0B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026">Falcon3 제품군</a></td>
        <td rowspan="2">1B-10B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130">Falcon-E 제품군</a></td>
        <td rowspan="2">1B-3B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>

## 설치

### 요구 사항
- python>=3.9
- cmake>=3.22
- clang>=18
    - Windows 사용자의 경우 [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)를 설치하십시오. 설치 프로그램에서 최소한 다음 옵션을 켜십시오 (CMake와 같은 필요한 추가 도구도 자동으로 설치됨):
        - C++를 사용한 데스크톱 개발
        - Windows용 C++ CMake 도구
        - Windows용 Git
        - Windows용 C++ Clang 컴파일러
        - LLVM 도구 집합(clang)에 대한 MS-Build 지원
    - Debian/Ubuntu 사용자의 경우 [자동 설치 스크립트](https://apt.llvm.org/)로 다운로드할 수 있습니다.

        `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
- conda (강력 권장)

### 소스에서 빌드

> [!IMPORTANT]
> Windows를 사용하는 경우 다음 명령에 대해 항상 VS2022용 개발자 명령 프롬프트/PowerShell을 사용해야 합니다. 문제가 발생하면 아래 FAQ를 참조하십시오.

1. 저장소 복제
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
2. 종속성 설치
```bash
# (권장) 새 conda 환경 만들기
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp

pip install -r requirements.txt
```
3. 프로젝트 빌드
```bash
# 수동으로 모델을 다운로드하고 로컬 경로로 실행
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

```
<pre>
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--quant-type {i2_s,tl1}] [--quant-embd]
                    [--use-pretuned]

추론 실행을 위한 환경 설정

선택적 인수:
  -h, --help            이 도움말 메시지를 표시하고 종료합니다.
  --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}
                        추론에 사용되는 모델
  --model-dir MODEL_DIR, -md MODEL_DIR
                        모델을 저장/로드할 디렉터리
  --log-dir LOG_DIR, -ld LOG_DIR
                        로깅 정보를 저장할 디렉터리
  --quant-type {i2_s,tl1}, -q {i2_s,tl1}
                        양자화 유형
  --quant-embd          임베딩을 f16으로 양자화
  --use-pretuned, -p    미리 조정된 커널 매개변수 사용
</pre>
## 사용법
### 기본 사용법
```bash
# 양자화된 모델로 추론 실행
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "당신은 도움이 되는 어시스턴트입니다" -cnv
```
<pre>
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT [-t THREADS] [-c CTX_SIZE] [-temp TEMPERATURE] [-cnv]

추론 실행

선택적 인수:
  -h, --help            이 도움말 메시지를 표시하고 종료합니다.
  -m MODEL, --model MODEL
                        모델 파일 경로
  -n N_PREDICT, --n-predict N_PREDICT
                        텍스트 생성 시 예측할 토큰 수
  -p PROMPT, --prompt PROMPT
                        텍스트를 생성할 프롬프트
  -t THREADS, --threads THREADS
                        사용할 스레드 수
  -c CTX_SIZE, --ctx-size CTX_SIZE
                        프롬프트 컨텍스트 크기
  -temp TEMPERATURE, --temperature TEMPERATURE
                        온도, 생성된 텍스트의 무작위성을 제어하는 하이퍼파라미터
  -cnv, --conversation  채팅 모드 활성화 여부 (지시 모델용).
                        (이 옵션을 켜면 -p로 지정된 프롬프트가 시스템 프롬프트로 사용됩니다.)
</pre>

### 벤치마크
모델을 제공하여 추론 벤치마크를 실행하는 스크립트를 제공합니다.

```
usage: e2e_benchmark.py -m MODEL [-n N_TOKEN] [-p N_PROMPT] [-t THREADS]

추론 실행을 위한 환경 설정

필수 인수:
  -m MODEL, --model MODEL
                        모델 파일 경로.

선택적 인수:
  -h, --help
                        이 도움말 메시지를 표시하고 종료합니다.
  -n N_TOKEN, --n-token N_TOKEN
                        생성된 토큰 수.
  -p N_PROMPT, --n-prompt N_PROMPT
                        텍스트를 생성할 프롬프트.
  -t THREADS, --threads THREADS
                        사용할 스레드 수.
```

각 인수에 대한 간략한 설명은 다음과 같습니다.

- `-m`, `--model`: 모델 파일 경로. 스크립트 실행 시 반드시 제공해야 하는 필수 인수입니다.
- `-n`, `--n-token`: 추론 중 생성할 토큰 수. 기본값은 128인 선택적 인수입니다.
- `-p`, `--n-prompt`: 텍스트 생성을 위해 사용할 프롬프트 토큰 수. 기본값은 512인 선택적 인수입니다.
- `-t`, `--threads`: 추론 실행에 사용할 스레드 수. 기본값은 2인 선택적 인수입니다.
- `-h`, `--help`: 도움말 메시지를 표시하고 종료합니다. 사용 정보를 표시하려면 이 인수를 사용하십시오.

예:

```sh
python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4
```

이 명령은 `/path/to/model`에 있는 모델을 사용하여 추론 벤치마크를 실행하고, 256 토큰 프롬프트에서 200개의 토큰을 생성하며, 4개의 스레드를 활용합니다.

공개 모델에서 지원하지 않는 모델 레이아웃의 경우, 지정된 모델 레이아웃으로 더미 모델을 생성하고 시스템에서 벤치마크를 실행하는 스크립트를 제공합니다.

```bash
python utils/generate-dummy-bitnet-model.py models/bitnet_b1_58-large --outfile models/dummy-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M

# 생성된 모델로 벤치마크 실행, -m을 사용하여 모델 경로 지정, -p를 사용하여 처리된 프롬프트 지정, -n을 사용하여 생성할 토큰 수 지정
python utils/e2e_benchmark.py -m models/dummy-bitnet-125m.tl1.gguf -p 512 -n 128
```

### `.safetensors` 체크포인트에서 변환

```sh
# .safetensors 모델 파일 준비
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./models/bitnet-b1.58-2B-4T-bf16

# gguf 모델로 변환
python ./utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16
```

### FAQ (자주 묻는 질문)📌

#### Q1: log.cpp의 std::chrono 관련 문제로 인해 llama.cpp 빌드 중 오류가 발생하며 빌드가 중단됩니다.

**A:**
이는 최신 버전의 llama.cpp에서 발생한 문제입니다. 이 문제를 해결하려면 [토론](https://github.com/abetlen/llama-cpp-python/issues/1942)의 이 [커밋](https://github.com/tinglou/llama.cpp/commit/4e3db1e3d78cc1bcd22bcb3af54bd2a4628dd323)을 참조하십시오.

#### Q2: Windows의 conda 환경에서 clang으로 빌드하려면 어떻게 해야 합니까?

**A:**
프로젝트를 빌드하기 전에 다음을 실행하여 clang 설치 및 Visual Studio 도구에 대한 액세스를 확인하십시오.
```
clang -v
```

이 명령은 올바른 버전의 clang을 사용하고 있는지 그리고 Visual Studio 도구를 사용할 수 있는지 확인합니다. 다음과 같은 오류 메시지가 표시되는 경우:
```
'clang'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램 또는 배치 파일이 아닙니다.
```

이는 명령줄 창이 Visual Studio 도구에 대해 올바르게 초기화되지 않았음을 나타냅니다.

• 명령 프롬프트를 사용하는 경우 다음을 실행하십시오.
```
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```

• Windows PowerShell을 사용하는 경우 다음 명령을 실행하십시오.
```
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Microsoft.VisualStudio.DevShell.dll" Enter-VsDevShell 3f0e31ad -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
```

이러한 단계를 수행하면 환경이 초기화되고 올바른 Visual Studio 도구를 사용할 수 있게 됩니다.

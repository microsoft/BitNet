# 한국어 BitNet 언어 모델 사전 학습 및 튜닝 가이드

## 서론

BitNet을 이해하기 위해서는 단순히 특정 모델 아키텍처로만 한정짓기보다 더 넓은 프레임워크로 접근하는 것이 중요합니다. BitNet은 다음과 같은 핵심 요소들을 포함하는 종합적인 접근 방식을 의미합니다: (1) 가중치의 초저비트 양자화(예: 1비트 또는 1.58비트 3진 가중치) 및 활성화 값의 저비트 양자화(예: 8비트), (2) 처음부터 양자화를 고려하여 학습하는 양자화 인식 학습(Quantization-Aware Training, QAT) 방식 채택, (3) 전체 정밀도(FP16/BF16) 모델과 유사한 성능을 목표로 하면서 메모리 사용량, 추론 속도, 에너지 소비 측면에서 상당한 효율성 향상을 추구, (4) `BitLinear`와 같은 맞춤형 레이어 구성 요소 및 `bitnet.cpp`와 같은 최적화된 추론 엔진의 활용 가능성. 이러한 요소들이 결합되어 BitNet의 특징을 이룹니다.

BitNet은 최근 주목받고 있는 1비트 대규모 언어 모델(LLM) 아키텍처로, 기존 부동소수점(FP16/BF16) 모델에 비해 메모리 사용량, 에너지 소비, 추론 속도 면에서 상당한 이점을 제공하는 것을 목표로 합니다. 본 문서는 한국어 BitNet 언어 모델, 특히 BitNet b1.58 변형을 사전 학습하는 방법에 대한 핵심 개념과 일반적인 지침을 제공합니다.

BitNet 모델, 특히 BitNet b1.58은 가중치를 -1, 0, 1의 세 가지 값으로 표현하는 3진 가중치(ternary weights)를 특징으로 합니다. 이러한 접근 방식은 모델의 크기를 크게 줄이고 계산 효율성을 향상시킵니다.

**매우 중요한 점:** BitNet 모델은 기존의 FP16 또는 BF16 정밀도를 가진 사전 학습된 모델을 양자화하여 얻는 것이 아닙니다. **BitNet 모델은 처음부터(from scratch) 학습되어야 합니다.** 이는 BitNet 아키텍처의 고유한 특성을 최대한 활용하고 최적의 성능을 달성하기 위해 필수적입니다.

## BitNet b1.58 사전 학습의 핵심 특징

BitNet b1.58 모델을 효과적으로 사전 학습하기 위해 이해해야 할 몇 가지 주요 특징이 있습니다.

### 1. 3진 가중치 (Ternary Weights: -1, 0, 1)

BitNet b1.58의 핵심은 대부분의 가중치를 -1, 0, 또는 1로 제한하는 것입니다. 이는 모델의 저장 공간을 극적으로 줄이고, 곱셈 연산을 단순한 덧셈/뺄셈 또는 무시(0일 경우)로 대체하여 계산 속도를 높일 수 있는 잠재력을 가집니다.

### 2. 처음부터 학습 (Training from Scratch)

다시 한번 강조하지만, BitNet 모델은 기존에 학습된 FP16/BF16 모델을 단순히 양자화하는 방식으로 생성되지 않습니다. 대신, 모델 아키텍처와 학습 과정 자체가 1비트 또는 1.58비트 가중치를 염두에 두고 설계되어 처음부터 학습됩니다. 이를 통해 모델은 저정밀도 가중치 환경에 최적화될 수 있습니다.

### 3. 전체 정밀도 `lm_head` (Full-Precision `lm_head`)

일반적으로 BitNet 모델의 대부분의 레이어는 3진 가중치를 사용하지만, 최종 언어 모델 헤드(`lm_head` 또는 출력 레이어)는 전체 정밀도(예: FP16 또는 BF16)로 유지됩니다. 이는 모델이 미묘한 출력 분포를 더 잘 학습하고 표현하는 데 도움이 되어 전반적인 성능 저하를 최소화하는 데 기여합니다.

### 4. 목표: 효율성과 성능의 균형

BitNet 사전 학습의 궁극적인 목표는 전체 정밀도 모델과 비슷한 수준의 언어 이해 및 생성 능력을 달성하면서도, 추론 시 훨씬 낮은 지연 시간(latency), 더 적은 메모리 사용량, 그리고 감소된 에너지 소비를 실현하는 것입니다.

## 데이터 준비 (Data Preparation)

한국어 BitNet 모델을 성공적으로 사전 학습하려면 방대하고 품질 좋은 한국어 텍스트 코퍼스가 필수적입니다.

### 1. 한국어 데이터셋 유형

일반적으로 사용될 수 있는 한국어 데이터셋 유형은 다음과 같습니다:
*   **뉴스 기사:** 시사적인 내용과 정제된 문장을 제공합니다.
*   **도서:** 다양한 어휘와 문체, 깊이 있는 내용을 포함합니다.
*   **웹 텍스트:** 블로그, 포럼, 웹사이트 등에서 수집된 광범위한 주제의 텍스트입니다. 정제 과정이 중요합니다.
*   **학술 자료:** 전문 용어와 논리적인 글쓰기 스타일을 학습하는 데 도움이 됩니다.
*   **대화 데이터:** 채팅, 메신저 대화 등은 모델의 자연스러운 대화 능력을 향상시킬 수 있습니다.

데이터의 **품질과 규모**는 모델 성능에 결정적인 영향을 미칩니다. 다양한 주제와 스타일을 포괄하며, 철자 및 문법적 오류가 적고 일관성 있는 수십억 토큰 규모의 데이터셋을 목표로 해야 합니다. 데이터 수집 시 저작권 및 개인정보 보호 규정을 준수하는 것이 중요합니다.

### 2. 데이터 전처리 및 토큰화 (Data Preprocessing and Tokenization)

사전 학습을 위해서는 텍스트 데이터를 모델이 이해할 수 있는 숫자 시퀀스로 변환하는 토큰화 과정이 필요합니다. `tinyllama-bitnet` 리포지토리의 `train.py` 스크립트를 예시로 설명합니다.

*   **토크나이저 로딩**:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    ```
    위 코드는 `meta-llama/Llama-2-7b-hf` 모델의 토크나이저를 사용합니다. 이 토크나이저는 주로 영어에 최적화되어 있지만, 다국어 처리 능력을 어느 정도 갖추고 있을 수 있습니다.

*   **`tokenize` 함수 로직**:
    `train.py`의 `tokenize` 함수는 다음과 같이 동작합니다:
    1.  여러 텍스트 문서를 하나로 결합합니다 (`concatenate_texts`).
    2.  각 텍스트의 끝에 문장 종료 토큰(`eos_token_id`)을 추가합니다.
    3.  결합된 텍스트를 모델의 `context_length` (예: 2048)에 맞춰 청크(chunk)로 나눕니다. 각 청크는 `input_ids`의 시퀀스가 됩니다.

    ```python
    # train.py 내의 tokenize 함수 예시 (개념적)
    def tokenize_function(examples):
        # 텍스트들을 하나로 합치고 eos 토큰 추가
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # context_length 단위로 청크 분할
        total_length = (total_length // context_length) * context_length
        result = {
            k: [t[i : i + context_length] for i in range(0, total_length, context_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy() # CLM의 경우 labels는 input_ids와 동일
        return result
    ```

*   **사전 토큰화된 데이터셋 활용**:
    `train.py`에서는 `xz56/openwebtext-tokenized-small`와 같이 이미 토큰화된 데이터셋을 사용하는 옵션도 제공합니다. 이는 대규모 데이터셋의 경우 토큰화 시간을 절약할 수 있는 방법입니다.

*   **한국어 토큰화 고려 사항**:
    *   `tinyllama-bitnet` 예제는 Llama-2 토크나이저를 사용하지만, 한국어 모델의 성능을 최적화하기 위해서는 한국어의 특성(교착어, 띄어쓰기 등)을 잘 처리할 수 있는 한국어 전용 토크나이저를 사용하거나 직접 학습시키는 것이 이상적입니다. (예: SentencePiece BPE 토크나이저)
    *   한국어 전용 토크나이저를 사용할 경우, 어휘 크기(vocab size) 결정도 중요한 요소입니다.

### 3. 데이터 형식/구조 (Data Format/Structure)

*   **입력 데이터**: 일반적으로 `.txt` 파일이나 `.jsonl` 파일 형태로 구성된 일반 텍스트(plain text)입니다.
*   **토큰화 이후**: 텍스트는 `input_ids` (정수 시퀀스)로 변환됩니다. Hugging Face `datasets` 라이브러리는 이러한 데이터를 효율적으로 로드하고 처리하며, 메모리 매핑(memory mapping)과 같은 기술을 사용하여 대용량 데이터셋도 다룰 수 있게 합니다. 각 샘플은 보통 다음과 같은 필드를 가질 수 있습니다:
    *   `input_ids`: 토큰 ID의 리스트
    *   `attention_mask`: 어텐션 메커니즘에서 어떤 토큰에 주목해야 하는지를 나타내는 마스크 (일반적으로 CLM에서는 모든 토큰에 주목)
    *   `labels`: CLM의 경우 `input_ids`와 동일하며, 모델이 예측해야 하는 타겟 토큰 ID

## 학습 환경 설정 (Training Environment Setup)

BitNet 모델을 사전 학습하기 위한 환경 설정 과정을 `tinyllama-bitnet` 예제를 중심으로 설명합니다.

### 1. 주요 라이브러리

`train.py` 스크립트 실행에 필요한 주요 Python 라이브러리는 다음과 같습니다:
*   `transformers`: Hugging Face의 트랜스포머 모델 및 유틸리티 사용
*   `datasets`: 데이터셋 로딩 및 전처리
*   `torch`: PyTorch 딥러닝 프레임워크
*   `wandb` (선택 사항): 실험 추적 및 시각화를 위한 Weights & Biases
*   `huggingface_hub` (선택 사항): 모델 및 토크나이저를 허깅페이스 허브에 업로드/다운로드

### 2. 모델 구성 (`Tiny Llama` 예시)

`train.py`에서는 `meta-llama/Llama-2-7b-hf`의 설정을 기반으로 작은 모델(`Tiny Llama` 스타일)을 구성합니다.

```python
# AutoConfig를 사용하여 Llama-2 7B 모델 설정을 로드
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# 모델 크기 축소를 위한 설정 변경 예시
config.hidden_size = 1024  # 'dim'
config.num_attention_heads = 8 # 'n_heads'
config.num_hidden_layers = 8 # 'n_layers'
config.intermediate_size = 2048

# 새로운 모델을 AutoModelForCausalLM.from_config(config)로 생성 가능
# model = AutoModelForCausalLM.from_config(config)
```
위와 같이 `hidden_size` (차원), `num_attention_heads` (어텐션 헤드 수), `num_hidden_layers` (레이어 수), `intermediate_size` (MLP 중간 크기) 등을 조정하여 모델의 크기를 결정합니다.

### 3. BitNet 아키텍처로 변환

표준 트랜스포머 모델을 BitNet 아키텍처로 변환하는 과정은 `tinyllama-bitnet`의 `utils.py`에 정의된 `convert_to_bitnet` 함수를 통해 이루어집니다. `train.py`에서 이 함수를 호출합니다.

```python
# model = AutoModelForCausalLM.from_config(config) # 예시 모델 생성 후
# model = convert_to_bitnet(model, copy_weights=False) # BitNet으로 변환
```

`convert_to_bitnet(model, copy_weights=False)` 함수의 주요 단계는 다음과 같습니다:

1.  **`nn.Linear`를 `BitLinear`로 교체**:
    *   모델 내의 `LlamaSdpaAttention` 레이어와 `LlamaMLP` 레이어에 포함된 모든 `nn.Linear` 계층을 사용자 정의 `BitLinear` 계층으로 대체합니다. `BitLinear`는 BitNet의 핵심 구성 요소로, 가중치와 활성화 값의 양자화를 처리합니다.
2.  **`input_layernorm` 제거**:
    *   `LlamaDecoderLayer` 내의 `input_layernorm` (일반적으로 `LlamaRMSNorm`)을 제거하고, 이를 `nn.Identity()` (아무 연산도 하지 않는 항등 함수)로 대체합니다. 이는 BitNet 아키텍처의 특징 중 하나로, `BitLinear` 내부에서 정규화를 처리하기 때문일 수 있습니다.

`copy_weights=False` 인자는 모델을 처음부터 학습시킴을 의미합니다. 만약 사전 학습된 가중치를 BitNet으로 변환하는 경우라면 `True`로 설정하고 가중치 복사 로직이 추가로 필요할 수 있으나, BitNet은 일반적으로 처음부터 학습합니다.

## `BitLinear` 계층 설명 (BitLinear Layer Explanation)

`BitLinear`는 `tinyllama-bitnet/utils.py`에 정의된 사용자 정의 `nn.Linear` 계층으로, BitNet의 핵심적인 연산을 수행합니다.

```python
# class BitLinear(nn.Linear): # utils.py 내의 BitLinear 정의 (개념적)
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.norm = LlamaRMSNorm(self.in_features, eps=1e-6) # 내부 RMSNorm
#
#     def forward(self, x):
#         # 입력 정규화
#         x_norm = self.norm(x)
#
#         # 활성화 양자화 (Activation Quantization) - Per-token 8-bit
#         # STE (Straight-Through Estimator)를 사용하여 역전파 시 기울기 흐름 유지
#         x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
#
#         # 가중치 양자화 (Weight Quantization) - Per-tensor 1.58-bit (ternary)
#         # STE 사용
#         w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
#
#         # 양자화된 활성화와 가중치를 사용하여 선형 변환
#         # F.linear는 y = xW^T + b 연산을 수행
#         y = F.linear(x_quant, w_quant, self.bias)
#
#         return y
```
`BitLinear`의 `forward` 연산 과정은 다음과 같습니다:

1.  **내부 `LlamaRMSNorm` 적용**: 입력 `x`에 대해 먼저 `LlamaRMSNorm`을 적용하여 정규화합니다 (`x_norm`).
2.  **활성화 양자화 (`activation_quant`)**: 정규화된 활성화 값 `x_norm`을 토큰별(per-token)로 8비트 양자화합니다. 이 과정에서 Straight-Through Estimator (STE)가 사용됩니다: `x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()`. 순전파 시에는 양자화된 값을 사용하지만, 역전파 시에는 `detach()`를 통해 양자화 연산의 기울기를 무시하고 `x_norm`의 기울기를 그대로 전달하여 학습을 가능하게 합니다.
3.  **가중치 양자화 (`weight_quant`)**: 계층의 가중치 `self.weight`를 텐서별(per-tensor)로 1.58비트 3진(-1, 0, 1) 양자화합니다. 여기서도 STE가 사용됩니다: `w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()`.
4.  **선형 변환**: 양자화된 활성화 `x_quant`와 양자화된 가중치 `w_quant`를 사용하여 `F.linear` 연산을 수행합니다.

`utils.py`의 주석에 따르면, "**This is only for training, and kernel optimization is needed for efficiency.**" 즉, 이 `BitLinear` 구현은 학습을 위한 것이며, 실제 추론 시 효율성을 극대화하기 위해서는 최적화된 커널(예: `bitnet.cpp`에서 제공하는 커널)이 필요합니다.

## 학습 실행 방법 (Training Execution Method)

### 1. 양자화 인식 학습 (Quantization-Aware Training, QAT)

BitNet의 핵심 학습 방식은 QAT입니다. 이는 모델 학습 과정에서 양자화(가중치 및 활성화의 저비트 표현)를 시뮬레이션하여, 모델이 양자화로 인한 성능 저하에 강인해지도록 만듭니다. `BitLinear` 계층 내의 양자화 로직과 STE 사용이 QAT를 가능하게 합니다.

### 2. 데이터 콜레이터 (Data Collator)

인과적 언어 모델링(Causal Language Modeling, CLM)을 위해 `DataCollatorForLanguageModeling`을 사용합니다.
```python
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# mlm=False는 CLM을 의미 (Masked Language Modeling이 아님)
```

### 3. `TrainingArguments` 설정

Hugging Face `Trainer`를 사용하기 위한 학습 관련 인자들을 설정합니다. `train.py`의 예시는 다음과 같습니다:

```python
args = TrainingArguments(
    output_dir="output_tinyllama",
    per_device_train_batch_size=8, # 장치당 배치 크기
    gradient_accumulation_steps=16, # 그래디언트 축적 스텝
    num_train_epochs=1, # 총 학습 에포크
    weight_decay=0.1, # 가중치 감쇠
    warmup_steps=750, # 웜업 스텝 수 (또는 warmup_ratio=0.1)
    lr_scheduler_type="cosine", # 학습률 스케줄러 유형 (예: cosine, polynomial)
    learning_rate=1.5e-3, # 학습률
    save_steps=5000,
    logging_steps=10,
    fp16=True, # 혼합 정밀도 학습 활성화
    # ... 기타 인자들
)
```

*   **유효 배치 크기 (Effective Batch Size)**: `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`.
    `train.py` 예시에서는 `8 * 16 = 128` 입니다.
*   **배치당 토큰 수**: "배치당 토큰 수는 최소 ~100k (10만)"를 권장합니다. 이는 `유효 배치 크기 * context_length`로 계산할 수 있습니다. 예를 들어 유효 배치 크기가 128이고 `context_length`가 2048이라면, 배치당 토큰 수는 `128 * 2048 = 262,144` (약 262k)입니다.

### 4. 하이퍼파라미터 (Hyperparameters)

사용자 피드백 및 일반적인 BitNet 학습 경험에 기반한 하이퍼파라미터 권장 사항은 다음과 같습니다 (`train.py`의 값과 다를 수 있음):

*   **옵티마이저 (Optimizer)**: AdamW (PyTorch 기본값 사용 가능). BitNet 관련 연구에서는 Adam의 β₁=0.9, β₂=0.98을 사용하는 경우가 많습니다. `train.py`는 `transformers`의 기본 AdamW 설정을 따릅니다.
*   **학습률 스케줄 (Learning Rate Schedule)**: Polynomial decay 또는 Cosine decay (`lr_scheduler_type`으로 설정). `train.py`는 `cosine`을 사용합니다.
*   **웜업 (Warmup)**: 전체 학습 스텝의 일정 비율(예: `warmup_ratio=0.1` in `train.py`) 또는 고정된 스텝 수(예: 750 스텝).
*   **배치 크기 (Batch Size)**: 가능한 큰 배치당 토큰 수를 사용하는 것이 좋습니다 (예: 256K 토큰 이상). 이는 `per_device_train_batch_size`, `gradient_accumulation_steps`, `context_length`를 조정하여 달성합니다.
*   **드롭아웃 (Dropout)**: 일반적으로 사용하지 않습니다 (`dropout=0`).
*   **가중치 감쇠 (Weight Decay)**: 모델 크기에 따라 0.01 또는 0.05 (매우 큰 모델의 경우) 등을 사용합니다. `train.py`는 0.1을 사용합니다.
*   **학습률 (Learning Rate)**: 모델 크기에 따라 다릅니다.
    *   `tinyllama-bitnet` (~84M 모델): `1.5e-3`
    *   125M 모델: 약 `2.4e-3`
    *   13B-30B 모델: 약 `4e-4`
    *   BitNet b1.58-2B-4T 모델의 경우, 2단계 학습률 스케줄(더 높은 LR로 시작하여 중간에 낮추는 방식)을 사용했다는 고급 정보도 있습니다.

### 5. 혼합 정밀도 학습 (Mixed-Precision Training)

`TrainingArguments`에서 `fp16=True` (또는 `bf16=True` 사용 가능 시)를 설정하여 혼합 정밀도 학습을 활성화합니다. 이는 학습 속도를 높이고 메모리 사용량을 줄여줍니다.
사용자 피드백에 따르면, 옵티마이저 상태(optimizer states)와 그래디언트(gradients)는 종종 BF16 또는 FP32와 같은 더 높은 정밀도로 유지되어 학습 안정성을 돕습니다. `transformers.Trainer`는 이를 자동으로 처리할 수 있습니다.

### 6. 분산 학습 (Distributed Training)

`tinyllama-bitnet` 예제는 단일 GPU 학습에 초점을 맞추고 있지만, 더 큰 모델(수십억 파라미터 이상)을 사전 학습하려면 분산 학습 환경이 필수적입니다. DeepSpeed (ZeRO 최적화 포함), PyTorch FSDP, Megatron-LM 등의 프레임워크를 사용하여 여러 GPU 또는 여러 노드에 걸쳐 학습을 확장할 수 있습니다.

### 7. `Trainer` 실행 및 모델 저장

학습을 시작하고 완료 후 모델을 저장합니다.
```python
# Trainer 초기화
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_datasets["train"], # 학습 데이터셋
    eval_dataset=tokenized_datasets["validation"], # 평가 데이터셋 (선택 사항)
    data_collator=data_collator,
)

# 학습 실행
trainer.train()

# 모델 저장
trainer.save_model() # output_dir에 저장됨
# tokenizer.save_pretrained(args.output_dir) # 토크나이저도 함께 저장
```

## 학습 목표 (Training Objectives)

표준적인 대규모 언어 모델의 학습 목표가 BitNet 사전 학습에도 적용될 수 있습니다:

*   **인과적 언어 모델링 (Causal Language Modeling, CLM):** 다음 토큰을 예측하는 방식입니다. GPT 계열 모델에서 주로 사용되며, `tinyllama-bitnet` 예제에서도 이 방식을 사용합니다.
*   **마스크 언어 모델링 (Masked Language Modeling, MLM):** 입력 텍스트의 일부 토큰을 마스킹하고, 모델이 이를 예측하도록 학습합니다. BERT 계열 모델에서 사용됩니다.

BitNet의 경우, 특정 학습 목표나 변형이 제안될 수 있으므로 관련 연구를 참고하는 것이 좋습니다. (이 섹션은 기존 내용을 바탕으로 재구성)

## 결론 (기존 내용 유지 또는 약간 수정)

한국어 BitNet 언어 모델의 사전 학습은 기존 LLM과는 다른 독특한 고려 사항, 특히 "처음부터 학습" 원칙과 3진 가중치 시스템을 중심으로 이루어집니다. 고품질의 대규모 한국어 코퍼스, 적절한 토큰화, 그리고 BitNet 아키텍처에 맞는 학습 전략을 통해 효율적이면서도 강력한 한국어 1비트 LLM을 개발할 수 있을 것입니다.

본 문서는 개괄적인 지침을 제공하며, 실제 구현 및 최적의 성능을 위해서는 공식 BitNet 간행물에서 제공하는 구체적인 학습 방법론과 실험 결과를 참조하는 것이 중요합니다.

## 참고 자료 (References)

*   **tinyllama-bitnet GitHub Repository:** [pranavjad/tinyllama-bitnet](https://github.com/pranavjad/tinyllama-bitnet) - 본 문서에서 많은 코드 예시와 학습 절차 아이디어를 참조한 저장소입니다.
*   **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58):** [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) - 1.58비트 BitNet에 대한 핵심 논문입니다.
*   **BitNet: Scaling 1-bit Transformers for Large Language Models (Original BitNet):** [arXiv:2310.11453](https://arxiv.org/abs/2310.11453) - 초기 1비트 BitNet 아키텍처에 대한 논문입니다.
*   **The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf:** [Microsoft UniLM GitHub](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) - Microsoft에서 제공하는 BitNet 학습 팁 및 FAQ 문서입니다.
*   **microsoft/BitNet (bitnet.cpp):** [GitHub](https://github.com/microsoft/BitNet) - BitNet 모델을 위한 공식 C++ 추론 프레임워크 저장소입니다.
*   **Hugging Face TRL (Transformer Reinforcement Learning):** [GitHub](https://github.com/huggingface/trl) - DPO(Direct Preference Optimization)와 같은 선호도 튜닝 기법을 구현하는 데 유용한 라이브러리입니다. (SFT-Tuning.md에 특히 관련)

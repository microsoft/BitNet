# 한국어 BitNet 언어 모델의 지도 학습 미세조정(SFT) 가이드

## 1. 지도 학습 미세조정(SFT) 소개

BitNet은 특정 모델 아키텍처를 넘어, 가중치 및 활성화 값의 저비트 양자화(예: 1.58비트 3진 가중치, 8비트 활성화), 처음부터 양자화를 고려한 학습(QAT), 그리고 `BitLinear` 계층 및 `bitnet.cpp`와 같은 최적화된 구성 요소를 포함하는 포괄적인 프레임워크입니다. 이는 전체 정밀도 모델 성능에 근접하면서 메모리, 속도, 에너지 효율성을 극대화하는 것을 목표로 합니다.

지도 학습 미세조정(Supervised Fine-Tuning, SFT)은 사전 학습된 대규모 언어 모델(LLM)을 특정 다운스트림 작업이나 원하는 행동에 맞게 조정하는 과정입니다. 이는 일반적으로 (입력, 원하는 출력) 쌍으로 구성된 레이블이 지정된 예제 데이터셋을 사용하여 모델을 추가로 학습시킴으로써 이루어집니다. SFT를 통해 모델은 특정 지시를 따르거나, 질문에 답하거나, 텍스트를 요약하는 등 다양한 작업을 더 잘 수행하도록 특화될 수 있습니다.

## 2. BitNet 모델을 위한 SFT

BitNet 모델, 특히 BitNet b1.58과 같은 변형은 가중치를 -1, 0, 1의 세 가지 값으로 제한하는 3진 가중치(ternary weights)를 사용하여 처음부터(from scratch) 사전 학습됩니다. 이러한 고유한 아키텍처는 SFT 과정에서 몇 가지 중요한 고려 사항을 제기합니다.

### 사전 학습된 BitNet 모델에 SFT 적용

이미 BitNet b1.58 형식으로 사전 학습된 모델에 SFT를 적용하는 것을 목표로 합니다.

#### 가중치 처리 방식

*   **3진 가중치 유지:** SFT 과정에서 BitNet 모델의 핵심인 3진 가중치가 그대로 유지되는지가 중요한 질문입니다. BitNet의 주된 장점인 효율성(메모리, 속도)을 유지하려면 SFT 중에도 가중치가 -1, 0, 1로 계속 제한되어야 할 가능성이 높습니다. 만약 SFT 과정에서 가중치가 전체 정밀도로 변경된다면, BitNet의 핵심 이점이 사라질 수 있습니다. *주의: 이 부분에 대한 구체적인 공식 BitNet SFT 절차 문서는 현재 접근이 어려워, 이는 BitNet의 설계 목표에 기반한 추론입니다. 실제 적용 시에는 공식 문서를 반드시 확인해야 합니다.*
*   **학습 중 가중치 업데이트:** 가중치가 3진으로 유지된다면, SFT 중 가중치 업데이트 메커니즘은 일반적인 부동소수점 업데이트와 다를 수 있습니다. 예를 들어, 업데이트된 가중치를 다시 -1, 0, 1 중 하나로 "반올림"하거나 투영하는 과정이 포함될 수 있습니다.

#### 표준 SFT와의 잠재적 차이점

BitNet의 3진 가중치 아키텍처는 표준 SFT 절차와 비교하여 다음과 같은 부분에서 차이를 유발할 수 있습니다:

*   **옵티마이저(Optimizer) 선택:** 표준 AdamW 등이 여전히 사용될 수 있지만, 3진 가중치에 더 적합한 맞춤형 옵티마이저나 학습률 스케줄 조정이 필요할 수 있습니다. 예를 들어, 가중치의 이산적인 특성으로 인해 학습 과정이 불안정해지는 것을 방지하기 위한 기법이 필요할 수 있습니다.
*   **학습률(Learning Rate) 스케줄:** 일반적인 SFT보다 더 작거나 특수한 형태의 학습률 스케줄이 필요할 수 있습니다.
*   **정규화(Regularization):** 3진 가중치를 유지하는 데 도움이 되는 특별한 정규화 기법이 유용할 수 있습니다.

#### `lm_head` 처리

사전 학습 시 전체 정밀도(예: FP16/BF16)로 유지되었던 언어 모델 헤드(`lm_head`)는 SFT 과정에서도 계속 전체 정밀도로 학습될 가능성이 높습니다. 이는 모델이 특정 작업에 대한 미묘한 출력 분포를 더 잘 학습하도록 돕습니다.

## 3. SFT 데이터 준비 (SFT Data Preparation)

효과적인 한국어 BitNet SFT를 위해서는 고품질의 한국어 지시(instruction) 및 응답 데이터셋이 필수적입니다.

### 3.1. 한국어 SFT 데이터셋 유형

*   **지시-응답 데이터셋 (Instruction-Following Datasets):** 특정 작업을 수행하도록 지시하는 프롬프트와 그에 대한 적절한 응답으로 구성됩니다.
    *   예: 질의응답, 요약, 번역, 코드 생성, 정보 추출, 창의적 글쓰기 등.
*   **대화 데이터셋 (Dialogue Datasets):** 사용자-모델 간의 여러 턴(turn)으로 이루어진 대화 형식의 데이터입니다. 모델이 이전 대화 내용을 기억하고 문맥에 맞는 응답을 생성하도록 학습합니다.

### 3.2. 한국어 SFT 데이터 소스 예시

*   **AI Hub 데이터셋:** NIA AI Hub ([aihub.or.kr](https://aihub.or.kr))에서 제공하는 다양한 한국어 말뭉치 및 대화형 데이터셋을 활용할 수 있습니다. (예: 한국어 대화 요약, 한국어 문서요약, 질의응답 등)
*   **로컬 구축 데이터셋:**
    *   **KoAlpaca:** 스탠포드 대학의 Alpaca 프로젝트 결과를 한국어로 번역하고 개선한 데이터셋.
    *   기타 연구자들이나 기관에서 공개한 한국어 지시-응답 데이터셋.
*   **자체 제작 데이터셋:** 특정 도메인이나 작업에 특화된 SFT 데이터셋을 직접 구축할 수 있습니다. 이 경우 데이터의 품질 관리가 매우 중요합니다.

### 3.3. 데이터 형식 및 구조

SFT 데이터셋은 주로 JSONL (JSON Lines) 형식으로 구성되며, 각 줄이 하나의 JSON 객체를 나타냅니다.

*   **일반적인 지시-응답 형식:**
    ```json
    {"prompt": "질문: 한국의 수도는 어디인가요?", "response": "답변: 한국의 수도는 서울입니다."}
    {"prompt": "다음 영단어의 뜻을 한국어로 알려주세요: 'efficiency'", "response": "'efficiency'는 한국어로 '효율성'을 의미합니다."}
    ```
*   **대화 형식 예시 (간단화):**
    ```json
    {"id": "dialogue_1", "turns": ["사용자: 오늘 날씨 어때?", "모델: 오늘은 서울 기준으로 맑고 화창한 날씨가 예상됩니다."]}
    ```
    실제 대화 데이터셋은 더 복잡한 구조를 가질 수 있으며, 각 턴마다 발화자(speaker) 정보 등을 포함할 수 있습니다.

*   **데이터 품질의 중요성:**
    *   **명확한 지시:** 프롬프트는 모델이 수행해야 할 작업을 명확하게 지시해야 합니다.
    *   **고품질 응답:** 응답은 정확하고, 유용하며, 문법적으로 올바르고, 자연스러워야 합니다.
    *   **다양성:** 다양한 유형의 지시와 주제를 포함하여 모델의 일반화 성능을 높이는 것이 중요합니다.

*   **프롬프트 템플릿 활용:**
    일관성 있는 학습을 위해 프롬프트 템플릿을 사용하는 것이 일반적입니다. 예를 들어, 모든 지시 앞에 특정 시스템 메시지나 역할을 부여하는 헤더를 추가할 수 있습니다.
    ```
    ### Instruction:
    {user_query}

    ### Response:
    ```

### 3.4. SFT를 위한 데이터 전처리

1.  **토큰화 (Tokenization):**
    *   사전 학습 단계에서 사용한 것과 **동일한 토크나이저**를 사용해야 합니다.
    *   프롬프트와 응답을 각각 토큰화한 후, 일반적으로 하나의 시퀀스로 결합합니다. 특수 토큰(예: `<s>`, `</s>`, `[INST]`, `[/INST]`)을 사용하여 프롬프트와 응답, 또는 사용자 턴과 모델 턴을 구분할 수 있습니다.
    *   예시적인 결합 방식: `<s>[INST] {prompt} [/INST] {response}</s>` 또는 `<s>사용자: {prompt}</s> 모델: {response}</s>`. 이 형식은 사용하는 모델 아키텍처나 기존 학습 방식에 따라 달라질 수 있습니다.

2.  **마스킹 (Masking):**
    *   SFT의 핵심 목표는 모델이 주어진 프롬프트(또는 대화의 이전 부분)에 대해 적절한 **응답**을 생성하도록 학습하는 것입니다.
    *   따라서 손실 함수(loss function)를 계산할 때, 프롬프트에 해당하는 토큰들은 무시(마스킹)하고, **응답에 해당하는 토큰들에 대해서만 손실을 계산**하는 것이 일반적입니다. 이렇게 하면 모델이 프롬프트를 단순히 반복하는 것을 방지하고 응답 생성 능력에 집중하게 됩니다.
    *   데이터 콜레이터(Data Collator)에서 `labels`를 생성할 때, 프롬프트 부분의 토큰 ID를 특정 값(예: -100)으로 설정하여 손실 계산에서 제외합니다.

## 4. SFT 학습 환경 설정 (SFT Training Environment Setup)

SFT를 위한 학습 환경은 사전 학습 환경과 유사한 점이 많지만, 모델 로딩 및 데이터 처리 방식에서 차이가 있습니다.

### 4.1. 주요 라이브러리

사전 학습과 마찬가지로 다음과 같은 라이브러리가 주로 사용됩니다:
*   `transformers`
*   `datasets`
*   `torch`
*   `wandb` (선택 사항)
*   `huggingface_hub` (선택 사항)

### 4.2. 사전 학습된 BitNet 모델 로딩

SFT는 이미 사전 학습된 BitNet 모델에서 시작합니다. `tinyllama-bitnet`과 같은 프로젝트에서 사전 학습 후 저장된 모델 체크포인트를 로드합니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "path/to/your/pretrained_bitnet_model_checkpoint" # 사전 학습된 BitNet 모델 경로
# 모델 로딩 시, BitLinear와 같은 사용자 정의 레이어가 등록되어 있어야 할 수 있음
# 또는, 모델 저장/로드 방식이 BitNet 아키텍처를 올바르게 유지하는지 확인 필요
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path) # 사전 학습 시 사용한 토크나이저 로드

# 만약 로드된 모델이 BitNet 구성이 아니라면,
# convert_to_bitnet(model, copy_weights=True)와 유사한 함수로 변환해야 할 수 있으나,
# 일반적으로는 저장 시점의 BitNet 아키텍처 그대로 로드됩니다.
```
사전 학습된 모델이 `BitLinear`와 같은 사용자 정의 모듈을 포함하고 있다면, 해당 모듈이 현재 환경에 정의되어 있어야 `AutoModelForCausalLM.from_pretrained`가 올바르게 작동합니다.

### 4.3. SFT 스크립트 구조 (개념적)

SFT 학습 스크립트는 사전 학습 스크립트(`tinyllama-bitnet/train.py` 등)의 구조를 기반으로 수정될 수 있습니다:

*   **데이터 로딩 및 전처리**:
    *   `datasets` 라이브러리를 사용하여 SFT용 JSONL 파일을 로드합니다.
    *   위에서 설명한 토큰화 및 마스킹 전략을 적용하여 프롬프트-응답 쌍을 처리합니다.
*   **모델**: 사전 학습된 BitNet 모델을 로드합니다.
*   **토크나이저**: 사전 학습 시 사용된 것과 동일한 토크나이저를 사용합니다.
*   **데이터 콜레이터**: `DataCollatorForLanguageModeling`을 계속 사용할 수 있지만, `labels` 필드에 프롬프트 마스킹이 올바르게 적용되도록 데이터 전처리 단계에서 처리해야 합니다.

## 5. SFT 학습 실행 방법 (SFT Training Execution Method)

### 5.1. 사전 학습된 BitNet 모델 로딩

학습 시작점에 특정 BitNet 모델 체크포인트를 명시적으로 로드하는 것이 중요합니다.

### 5.2. 주요 SFT 하이퍼파라미터

SFT는 사전 학습과 비교하여 일반적으로 다른 하이퍼파라미터 값을 사용합니다:

*   **학습률 (Learning Rate)**: 사전 학습보다 훨씬 작은 학습률을 사용합니다.
    *   일반적인 범위: `1e-5` ~ `5e-5` (예: `2e-5`).
*   **에포크 (Epochs)**: 전체 데이터셋을 반복하는 횟수로, 사전 학습보다 훨씬 적습니다.
    *   일반적인 범위: 1 ~ 5 에포크. 데이터셋 크기와 품질에 따라 조절합니다.
*   **배치 크기 (Batch Size)**: GPU 메모리 및 시퀀스 길이에 따라 결정됩니다. SFT 데이터는 프롬프트와 응답을 포함하여 시퀀스 길이가 사전 학습보다 길어질 수 있으므로, 배치 크기를 줄여야 할 수 있습니다.
*   **옵티마이저 (Optimizer)**: AdamW가 여전히 일반적으로 사용됩니다. (예: β₁=0.9, β₂=0.98 또는 PyTorch 기본값)
*   **웜업 및 학습률 스케줄러 (Warmup & LR Scheduler)**: 사전 학습과 유사하게 짧은 기간의 웜업 후 코사인(cosine) 또는 선형(linear) 감쇠 스케줄러를 사용할 수 있습니다.
*   **가중치 감쇠 (Weight Decay)**: 과적합을 방지하기 위해 작은 값 (예: 0.01)을 사용할 수 있습니다.

### 5.3. 학습 루프 / `Trainer` API

Hugging Face `Trainer` API는 SFT 과정에서도 동일하게 활용될 수 있습니다. 핵심적인 차이는 입력 데이터셋의 구성(지시-응답 쌍, 프롬프트 마스킹)과 하이퍼파라미터 설정에 있습니다.

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./sft_output_bitnet_korean",
    num_train_epochs=3,
    per_device_train_batch_size=4, # SFT에서는 배치 크기를 줄일 수 있음
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.03, # 예시
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch", # 또는 save_steps
    fp16=True, # 혼합 정밀도
    # ... 기타 SFT 관련 인자들
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_train_dataset, # 전처리된 SFT 학습 데이터셋
    eval_dataset=sft_eval_dataset,  # 전처리된 SFT 평가 데이터셋 (선택 사항)
    data_collator=data_collator_sft, # SFT용 데이터 콜레이터 (labels 마스킹 처리 포함)
)

trainer.train()
trainer.save_model() # 미세조정된 모델 저장
tokenizer.save_pretrained(training_args.output_dir) # 토크나이저 저장
```

### 5.4. SFT 실행 예시 명령어 (개념적)

가상의 SFT 학습 스크립트 `train_sft.py`가 있다고 가정할 때 실행 명령어는 다음과 같을 수 있습니다:

```bash
python train_sft.py \
    --model_name_or_path "path/to/pretrained_bitnet_model_checkpoint" \
    --tokenizer_name "path/to/pretrained_bitnet_model_checkpoint" \
    --train_file "path/to/korean_sft_data_train.jsonl" \
    --output_dir "./sft_bitnet_korean_model" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --fp16 \
    # ... 기타 인자들
```

## 6. 선호도 최적화 (DPO - Direct Preference Optimization)

SFT 이후 모델의 응답을 인간의 선호도에 더욱 가깝게 정렬하기 위한 추가적인 단계로 DPO(Direct Preference Optimization)를 고려할 수 있습니다. DPO는 SFT된 모델을 기반으로 추가적인 미세조정을 수행합니다.

*   **목표**: 모델이 인간이 선호하는 응답 스타일, 내용, 안전성 등을 갖도록 학습합니다.
*   **데이터셋**: DPO는 프롬프트(prompt)와 함께 인간이 더 선호하는 응답(chosen response)과 덜 선호하는 응답(rejected response) 쌍으로 구성된 데이터셋을 사용합니다.
    *   예시 데이터 형식:
        ```json
        {"prompt": "대한민국에서 가장 높은 산은 무엇인가요?", "chosen_response": "대한민국에서 가장 높은 산은 제주도에 위치한 한라산입니다.", "rejected_response": "음... 아마 설악산일 겁니다. 확실하지는 않아요."}
        ```
*   **학습 방식**: DPO는 보상 모델(Reward Model)을 명시적으로 학습할 필요 없이, 선호도 데이터를 사용하여 직접적으로 언어 모델을 최적화합니다. 이는 RLHF(Reinforcement Learning from Human Feedback)의 복잡성을 줄이면서도 유사한 효과를 얻을 수 있는 방법으로 주목받고 있습니다.
*   **구현**: DPO 학습을 위해서는 `trl` (Transformer Reinforcement Learning) 라이브러리의 `DPOTrainer`와 같은 특화된 도구나 직접 구현이 필요할 수 있습니다.

DPO는 SFT 이후 모델의 품질을 한 단계 더 끌어올릴 수 있는 고급 튜닝 기법입니다.

## 7. 평가
(기존 섹션 5. 평가 내용을 이쪽으로 이동 또는 유지)
SFT된 한국어 BitNet 모델의 성능은 수행하려는 특정 작업에 따라 평가됩니다.

*   **자동 평가 지표:**
    *   **질의응답:** Exact Match (EM), F1 Score
    *   **요약:** ROUGE, BLEU (요약문과 참조 요약문 간의 유사도)
    *   **번역:** BLEU, METEOR
    *   **분류:** 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 Score
*   **인간 평가:** 특히 개방형 질문 답변, 창의적 텍스트 생성, 대화 능력 등과 같이 자동 평가가 어려운 작업의 경우, 인간 평가자가 모델 출력의 품질, 관련성, 유창성 등을 직접 평가하는 것이 중요합니다.
*   **벤치마크 데이터셋:** 공개된 한국어 LLM 평가 벤치마크(예: KLUE, KoBEST 등)를 사용하여 다른 모델과 성능을 비교할 수 있습니다.

## 8. 결론
(기존 섹션 6. 결론 내용을 이쪽으로 이동 또는 유지)
한국어 BitNet 언어 모델의 SFT는 사전 학습된 모델의 효율성을 유지하면서 특정 작업에 대한 성능을 극대화하는 것을 목표로 합니다. 이 과정에서는 BitNet 고유의 3진 가중치 특성을 고려한 학습 전략과 고품질의 한국어 지시-응답 데이터셋이 핵심적인 역할을 합니다.

본 문서는 일반적인 SFT 원칙과 BitNet 아키텍처의 특성을 결합하여 한국어 BitNet 모델의 SFT에 대한 개괄적인 지침을 제공합니다. 그러나 BitNet 모델을 위한 최적의 SFT 방법론과 구체적인 절차는 모델 개발자가 제공하는 공식 문서나 관련 연구 논문을 통해 확인하는 것이 가장 정확합니다. 특히 SFT 중 가중치 처리 방식과 같은 핵심적인 세부 사항은 공식적인 정보가 중요합니다.

## 참고 자료 (References)

*   **tinyllama-bitnet GitHub Repository:** [pranavjad/tinyllama-bitnet](https://github.com/pranavjad/tinyllama-bitnet) - 본 문서에서 많은 코드 예시와 학습 절차 아이디어를 참조한 저장소입니다.
*   **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58):** [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) - 1.58비트 BitNet에 대한 핵심 논문입니다.
*   **BitNet: Scaling 1-bit Transformers for Large Language Models (Original BitNet):** [arXiv:2310.11453](https://arxiv.org/abs/2310.11453) - 초기 1비트 BitNet 아키텍처에 대한 논문입니다.
*   **The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf:** [Microsoft UniLM GitHub](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) - Microsoft에서 제공하는 BitNet 학습 팁 및 FAQ 문서입니다.
*   **microsoft/BitNet (bitnet.cpp):** [GitHub](https://github.com/microsoft/BitNet) - BitNet 모델을 위한 공식 C++ 추론 프레임워크 저장소입니다.
*   **Hugging Face TRL (Transformer Reinforcement Learning):** [GitHub](https://github.com/huggingface/trl) - DPO(Direct Preference Optimization)와 같은 선호도 튜닝 기법을 구현하는 데 유용한 라이브러리입니다. (SFT-Tuning.md에 특히 관련)

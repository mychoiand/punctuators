
# SBD (Sentence Boundary Detection) 모델 상세 명세서

## 1. 모델 개요 (Model Overview)
*   **모델 ID**: `1-800-BAD-CODE/sentence_boundary_detection_multilang`
*   **목적**: 이미 구두점이 찍혀 있는 긴 텍스트를 문장 단위로 분할(Segmentation)하는 작업 수행.
*   **지원 언어**: 49개 다국어 지원 (한국어, 영어, 중국어, 일본어, 유럽 언어 등 포함).
*   **특징**: 추론 시 언어 태그(Language Tag)가 필요 없는 **Language-Agnostic** 모델입니다. 즉, 하나의 배치(Batch)에 여러 언어가 섞여 있어도 상관없이 동작합니다.

## 2. 모델 아키텍처 (Architecture) details
이 모델은 **데이터 기반(Data-driven)** 접근 방식을 사용하며, 매우 효율적인 구조를 가지고 있어 CPU 추론에 적합합니다.

### 2.1 구성 요소
1.  **Tokenizer (토크나이저)**
    *   **종류**: SentencePiece (Unigram or BPE)
    *   **Vocabulary Size**: 64,000
    *   다국어 처리를 위해 충분히 큰 어휘 사전을 사용합니다.

2.  **Encoder (인코더)**
    *   **Base**: BERT-style Transformer Encoder
    *   **Layers**: 4
    *   **Attention Heads**: 8
    *   **Hidden Dimension**: 128
    *   **Intermediate (Feed-Forward) Dimension**: 512
    *   **총 파라미터**: 약 900만 개 (9M)
        *   임베딩(Embeddings): 약 820만 개 (대부분을 차지함)
        *   연산 파라미터: 약 80만 개 (실제 연산량은 매우 적음)

3.  **Classification Head (분류기)**
    *   각 Subword 토큰에 대해 "이 토큰이 문장의 마지막인가?"를 예측하는 선형 분류기(Linear Classifier)가 부착되어 있습니다.

## 3. 학습 상세 (Training Details)
*   **프레임워크**: NVIDIA NeMo (Custom fork branch `sbd`)
*   **학습 데이터**:
    *   각 언어별 약 100만 줄(Line)의 텍스트 사용 (총 4,900만 줄).
    *   랜덤하게 언어를 샘플링하여 다국어 배치를 구성.
    *   **Batch Size**: 256
*   **학습 과정**: 수십만 Step 동안 학습됨.

## 4. 언어별 처리 규칙 (Language Specific Rules)
모델은 기본적으로 확률에 의존하지만, 일부 언어적 특성을 반영합니다.

*   **약어(Abbreviation) 처리**: "U.S.", "Dr.", "p.m." 같은 약어 뒤의 마침표는 문장의 끝이 아님을 학습합니다.
*   **다국어 문장 부호**:
    *   영어/유럽어: `.`, `?`, `!`
    *   중국어/일본어: `。`, `？`, `！`
    *   태국어: (공백이나 특정 문맥으로 구분)
    *   **예시**:
        *   영어: "dinner's at 630 p.m." -> 여기서 분리하지 않음.
        *   스페인어: "¿tiene algo de beber?" -> 문장 시작의 뒤집힌 물음표 인식.

## 5. 입력 및 출력 (Inputs and Outputs)
*   **입력**: 구두점이 포함된(Punctuated) 텍스트. (예: "Hello world. How are you?")
*   **출력**: 각 토큰별 문장 경계 확률 (Probability that token `t` is EOS).
*   **후처리**: 확률이 임계값(Threshold)을 넘는 지점을 기준으로 텍스트를 리스트로 분할합니다.

## 6. 한계점 (Limitations)
*   **구두점 의존성**: 입력 텍스트에 구두점이 없으면(예: "hello world how are you") 문장 경계를 거의 감지하지 못합니다. 구두점이 없는 텍스트는 PCS 모델을 사용해야 합니다.
*   **학습 데이터 노이즈**: 웹 크롤링 데이터 등을 사용했을 가능성이 있어, 매우 드문 케이스나 문법적으로 완벽하지 않은 문장에 대해 오작동할 수 있습니다.

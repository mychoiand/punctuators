
# PCS XLM-Roberta (High Accuracy) 모델 상세 명세서

## 1. 모델 개요
*   **모델 ID**: `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`
*   **목적**: 최고의 정확도가 필요한 환경에서 구두점 복원, 대소문자 복원, 문장 분리를 수행합니다.
*   **기반 모델**: **XLM-Roberta** (Facebook AI에서 개발한 대규모 다국어 모델)

## 2. 모델 아키텍처 (Architecture Details)
이 모델은 **Lightweight 모델과 동일한 논리적 그래프(파이프라인)**를 공유하지만, 그 기반이 되는 인코더(Backbone)가 훨씬 거대하고 강력합니다.

### 2.1 백본 인코더 (Backbone)
*   **XLM-Roberta (Base/Large)**: 100개 이상의 언어로 사전 학습된 강력한 모델입니다.
*   텍스트의 의미적 뉘앙스와 문법적 구조를 깊이 이해하므로, 모호한 문장에서도 우수한 성능을 보입니다.

### 2.2 예측 그래프 (Prediction Graph)
1.  **Tokenizer**: XLM-R Tokenizer (SentencePiece, Vocab ~250k)
2.  **Encoding**: XLM-Roberta로 문맥 벡터 생성.
3.  **Graph Logic**:
    *   **Punctuation**: 토큰 전후의 구두점을 예측합니다.
    *   **Embedding & Conditioning**: 예측된 구두점을 임베딩하여 SBD 헤드에 전달합니다. "마침표가 찍혔다는 사실"이 문장 분리의 강력한 힌트가 됩니다.
    *   **Shift for True-casing**: 문장 분리 정보를 한 칸 밀어(Shift), 다음 문장 첫 단어의 대문자화를 유도합니다.

### 2.3 대소문자 복원 (True-casing) 특징
*   이 모델은 대소문자 복원을 **Multi-label Problem**으로 접근합니다.
*   각 서브워드 토큰 내의 글자 수(N)만큼 예측을 수행합니다.
*   즉, 단순히 "첫 글자만 대문자로" 하는 것이 아니라, "NATO", "MacDonald", "mRNA" 같이 단어 중간의 대문자나 전체 대문자도 처리할 수 있는 구조입니다.

## 3. 학습 상세 (Training Details)
*   **하드웨어**: NVIDIA A100 GPU (약 7시간 학습)
*   **데이터**: WMT News Crawl (각 언어 100만 줄).
*   **샘플링 이슈**: 스페인어의 뒤집힌 물음표(`¿`)와 같은 희귀 토큰을 학습시키기 위해 해당 문장들을 Over-sampling(과잉 추출)했습니다.

## 4. 제약 사항 및 알려진 문제 (Limitations)
1.  **스페인어 물음표 과잉 예측**:
    *   희귀 토큰(`¿`) 학습을 위해 데이터를 늘린 부작용으로, 스페인어에서 물음표를 너무 자주 예측하는 경향이 있습니다. ("Over-corrected")
2.  **뉴스 데이터 편향**:
    *   뉴스 데이터로 학습되어 대화체(구어체) 데이터에서는 다소 어색할 수 있습니다.
3.  **높은 연산량**:
    *   모델 크기가 크기 때문에 CPU 환경에서는 실시간 처리가 어려울 수 있습니다. 배치 처리가 권장됩니다.

### NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models (https://arxiv.org/abs/2405.17428)
- **What's New**: 이 논문에서는 디코더-전용 대형 언어 모델(LLM)(

- **Technical Details**: {'Model Architecture': '모델 아키텍처 측면에서 NV-Embed는 잠재(attention) 계층을 통해 풀링된 임베딩을 얻는 구조를 제안합니다. 이는 평균 풀링 또는 LLM에서 마지막 <EOS> 토큰 임베딩을 사용하는 것보다 검색과 다운스트림 작업의 정확도를 지속적으로 향상시킵니다. 또한 대조 학습 중 LLM의 인과(attention) 마스크를 제거하여 표현 학습을 향상시킵니다.', 'Training Procedure': '모델 학습을 위해 두 단계의 대조적 명령(tune) 방법을 도입합니다. 첫 번째 단계에서는 인-배치 부정 사례(batch negatives) 및 엄선된 어려운 부정 사례를 사용하여 검색 데이터셋에서 명령을 통한 대조(training)를 시행합니다. 두 번째 단계에서는 다양한 비검색 데이터셋을 섞어 1단계의 학습 데이터와 통합하여, 비검색 작업의 정확도를 향상시킬 뿐 아니라 검색 성능도 향상시킵니다.'}

- **Performance Highlights**: 이러한 기술을 결합한 결과, NV-Embed 모델은 MTEB(대규모 텍스트 임베딩 벤치마크)에서 69.32의 기록적인 점수를 달성하며 56개의 작업에서 1위를 차지했습니다. 또한 MTEB 벤치마크 내의 15개 검색 작업에서 59.36의 최고 점수를 얻었습니다. 이는 E5-mistral-7b-instruct, SFR-Embedding, 그리고 Voyage-large-2-instruct 모델을 능가하는 성과입니다.



### THREAD: Thinking Deeper with Recursive Spawning (https://arxiv.org/abs/2405.17402)
- **What's New**: 새로운 연구는 언어 모델(LLMs)의 맥락 길이와 복잡성이 증가함에 따라 성능이 저하되는 문제를 해결하기 위해 'Thinking Recursively and Dynamically (ThReaD)'라는 프레임워크를 제안합니다. ThReaD는 모델 생성을 실행 스레드로 처리하며, 필요에 따라 새로운 스레드를 생성해 작업을 분담합니다. 이를 통해 모델은 작업을 점진적으로 단순한 하위 문제로 나누어 해결할 수 있게 됩니다.

- **Technical Details**: ThReaD는 모델 생성을 실행 스레드(threads of execution)로 프레임화하고, 맥락에 따라 완료되거나 새로운 스레드를 동적으로 생성할 수 있게 구성됩니다. 자식 스레드는 부모 스레드의 토큰 시퀀스를 기반으로 조건을 생성하며, 부모 스레드의 작업을 돕기 위해 필요한 정보만 반환합니다. 이로써 모델은 필요에 따라 중간 작업의 양을 동적으로 조정할 수 있게 됩니다. ThReaD는 다양한 설정에서 적용 가능하며, 자식 스레드 생성 및 동기화 메커니즘은 설정에 따라 다를 수 있습니다.

- **Performance Highlights**: ThReaD는여러 벤치마크에서 GPT-4와 GPT-3.5를 사용하여 최첨단 성능(state-of-the-art performance)을 달성했습니다. 또한 Llama-3-8b 및 CodeLlama-7b와 같은 작은 모델에서도 기존 프레임워크보다 10%에서 50%까지 성능 향상을 보여줬습니다. 주요 성능 테스트는 ALFWorld, TextCraft, WebShop, DataCommons QA, MIMIC-III ICU QA 등의 벤치마크에서 수행되었습니다.



### The Expressive Capacity of State Space Models: A Formal Language Perspectiv (https://arxiv.org/abs/2405.17394)
- **What's New**: 최근 연구들은 Linear State Space Models(SSMs)가 Transformer와 경쟁할 만한 언어 모델링 성능을 보여주고 있음을 밝혔습니다. 그러나 SSM의 이론적 능력에 대한 이해는 부족합니다. 이 논문은 SSM의 언어 모델링 능력을 Transformer 및 전통적인 RNN과 비교하는 포괄적인 이론적 연구를 제시합니다. 특히, SSM이 특정 문제를 정확하게 나타내는 데 있어 더 나은 성능을 보이지만, 현재의 SSM 디자인에 제한이 존재한다는 것을 확인했습니다.

- **Technical Details**: SSM의 단일 레이어는 입력 길이 T에 대한 반복을 기준으로 정의됩니다. SSM은 효율적인 병렬화를 허용하는 재귀적 모델로, 주기적 상태 갱신을 통해 모델을 구성합니다. SSM의 디자인에서는 Mix와 Norm(예: RMSNorm, LayerNorm)와 같은 다양한 채널 믹싱 변환과 정규화를 포함합니다. 대표적인 SSM인 Mamba를 통해 실험 결과를 확인하였습니다.

- **Performance Highlights**: 이 논문에서는 SSM과 Transformer가 겹치면서도 서로 다른 강점을 가지고 있음을 발견했습니다. 예를 들어, SSM은 flip-flop 상태 추적 문제를 쉽게 모델링하지만, Modular counting 같은 문제에서는 Transformer와 유사한 어려움을 겪습니다. 반면, 현재 SSM의 설계 선택이 표현력에 제한을 두고 있다는 점도 확인할 수 있었습니다. 이러한 결과는 SSM과 LLM 연구에 중요한 시사점을 제공하며, 미래의 언어 모델 아키텍처가 두 가지 모두의 강점을 결합해야 할 필요성을 시사합니다.



### MindMerger: Efficient Boosting LLM Reasoning in non-English Languages (https://arxiv.org/abs/2405.17386)
- **What's New**: 새로운 방법인 MindMerger를 제안하여 대형 언어 모델(LLMs)의 내장된 추론 및 언어 이해 능력을 유지하면서 다국어 모델에서 외부 언어 이해 능력을 통합하여 다국어 추론 성능을 향상시킵니다. 이는 특히 자원이 부족한 언어에서 두드러지게 성능을 향상시킵니다.

- **Technical Details**: MindMerger는 외부 다국어 모델의 언어 이해 능력을 LLM에 통합하기 위해 두 단계의 학습 스키마를 사용합니다. 첫 번째는 일반적으로 접근 가능한 이중 언어 쌍을 사용하여 매핑하여 외부 능력을 LLM에 삽입합니다. 두 번째는 번역 모델로 생성된 쿼리 번역 작업 데이터를 사용하여 내장된 능력과 외부 능력을 협력적으로 사용하는 단계를 포함합니다. LLM 및 다국어 모델의 파라미터는 동결되어 내장된 능력을 잊지 않도록 합니다.

- **Performance Highlights**: 다국어 추론 데이터셋 세 개 및 언어 이해 데이터셋에서 MindMerger는 모든 기준선을 일관되게 초과했습니다. MGSM 데이터셋에서 평균 정확도가 모든 언어에서 6.7%, 자원이 부족한 언어에서 8.0% 향상되었습니다. 또한 기존의 번역 기반 방법에 비해 평균 정확도를 6.6% 더 높였습니다.



### Unlocking the Secrets of Linear Complexity Sequence Model from A Unified Perspectiv (https://arxiv.org/abs/2405.17383)
Comments:
          Technical report. Yiran Zhong is the corresponding author

- **What's New**: 이번 연구에서는 LCSM (Linear Complexity Sequence Model)을 소개합니다. 이는 다양한 선형 복잡성 시퀀스 모델링 기술들을 하나의 프레임워크로 통합한 것입니다. 이 모델은 특히, 모델링 과정을 '확장 (Expand)', '진동 (Oscillation)', '축소 (Shrink)'의 세 단계로 구분하여 각 구성 요소의 영향을 분석합니다.

- **Technical Details**: LCSM은 입력 신호를 고차원 메모리 상태로 투영하는 '확장' 단계, 메모리 상태에 재귀적 연산을 수행하는 '진동' 단계, 그리고 메모리 상태를 저차원 공간으로 되돌리는 '축소' 단계로 구성됩니다. 다양한 시퀀스 모델링 방법론들은 이 세 단계에서 다른 설정을 활용합니다. 예를 들어, Linear Attention은 'right-product kernel trick'을 사용하여 쿼리-키 프로덕션의 계산을 회피하고, State Space Model(SSM)는 특별한 초기화 및 대각화 가정을 통해 효율성을 극대화합니다.

- **Performance Highlights**: 데이터 의존적 방법이 언어 모델링에서 중요한 역할을 한다는 것을 발견했습니다. 반면, 수작업으로 설계된 기법들이 정보 검색 작업 (retrieval tasks)에서 더 나은 성능을 보였습니다.



### Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention (https://arxiv.org/abs/2405.17381)
Comments:
          Accepted by ICML 2024. Yiran Zhong is the corresponding author. Code is released at this http URL

- **What's New**: Lightning Attention을 소개합니다. 이는 다양한 시퀀스 길이에서도 일정한 학습 속도를 유지하면서 고정된 메모리 소비를 보장하는 첫 번째 선형 주의(attention) 구현입니다. Lightning Attention은 cumsum(누적 합) 연산의 문제를 해결하며, 이를 통해 기존의 선형 주의 메커니즘들보다 더 뛰어난 성능을 이끌어냅니다. 또한, TransNormerLLM (TNL)이라는 새로운 아키텍처를 도입하여 Lightning Attention에 최적화된 정확도를 제공합니다.

- **Technical Details**: 주요 아이디어는 주의 계산을 intra-blocks과 inter-blocks으로 분할하여 실행하는 것입니다. intra-blocks에는 기존의 주의 계산(conventional attention computation)을, inter-blocks에는 선형 주의 커널 트릭(linear attention kernel tricks)을 적용합니다. 이를 통해 cumsum 연산의 필요성을 제거하였습니다. 또한, GPU 하드웨어의 성능을 최대한 활용하기 위해 타일링 기법(tiling technique)을 전방 및 후방 단계에서 채택하였습니다. TNL은 LRPE(positional embedding), 선형 주의 가속(linear attention acceleration), 게이팅 메커니즘(gating mechanism), 텐서 정규화(tensor normalization)와 같은 향상된 수정을 포함하여 구축되었습니다.

- **Performance Highlights**: TNL은 기존의 언어 모델들과 비교해 뛰어난 효율성을 자랑합니다. 사전 훈련된 데이터셋과 자체 수집 데이터셋에서 다양한 모델 크기 및 시퀀스 길이에 대해 엄격한 테스트를 실시한 결과, TNL은 최신 상태의 LLM(language models)들과 비교해도 동등하거나 더 나은 성능을 보였습니다. 또한, Lightning Attention은 FlashAttention-2와 비교하여 계산 속도와 메모리 소비 면에서 우위를 선보였으며, 다양한 크기의 모델들(44M, 385M, 1B, 7B, 15B)에서도 동일한 성과를 나타냈습니다.



### Federating Dynamic Models using Early-Exit Architectures for Automatic Speech Recognition on Heterogeneous Clients (https://arxiv.org/abs/2405.17376)
Comments:
          The paper is under review in Future Generation Computer Systems Journal

- **What's New**: 새로운 연구는 자동 음성 인식(ASR) 모델 학습에서 클라이언트 디바이스의 리소스 제약을 해결하기 위해 동적 아키텍처(dynamical architectures)를 제안합니다. 이는 입력과 운영 조건에 따라 처리 레이어를 조정할 수 있는 초기 종료 솔루션(early-exit solutions)을 사용하여, 이종 네트워크(heterogeneous networks) 환경에서도 단일 모델을 효과적으로 활용할 수 있게 합니다.

- **Technical Details**: 연구는 연방 학습(federated learning)을 활용해, 데이터 소유권 및 프라이버시 문제를 해결하면서도 로컬 데이터로 분산 학습이 가능하도록 합니다. 특히, 동적 아키텍처(EE architectures)는 입력 처리 레이어를 조정해 클라이언트 요구에 맞추어 다중 축소 버전을 제공할 수 있습니다. 이는 클라이언트 간 리소스 이질성을 관리하기 쉽게 하며, 안전한 집계(secure aggregation)와 차등 프라이버시(differential privacy)와도 자연스럽게 융합됩니다.

- **Performance Highlights**: 공공 데이터셋을 이용한 실험 결과, 제안된 방법이 기본 연방 학습 전략과 결합하여 효과적으로 작동함을 보여줍니다. 또한, EE 아키텍처들이 도입된 경우에도 성능 향상이 이루어지며, 다른 기법들에 비해 중앙 모델을 관리하는데 있어 자원 소모를 줄일 수 있음을 입증했습니다.



### A One-Layer Decoder-Only Transformer is a Two-Layer RNN: With an Application to Certified Robustness (https://arxiv.org/abs/2405.17361)
- **What's New**: 이 논문은 단일 레이어 디코더 전용 Transformer가 이중 레이어 RNN과 동등하다는 중요한 인사이트를 밝혀냅니다. 이를 바탕으로 ARC-Tran이라는 새로운 접근법을 제안하여 디코더 전용 Transformers의 로버스트니스(robustness)를 임의의 변형 공간(perturbation spaces)에서 검증합니다.

- **Technical Details**: ARC-Tran은 기존의 위치 인코딩(position encoding) 문제를 해결하기 위해 새로운 임베딩 전략을 도입하고, 단일 레이어 디코더 전용 Transformer의 주의 메커니즘(attention mechanism)을 이중 레이어 RNN으로 재해석합니다. 이를 통해, 변형된 입력의 길이 차이로 인한 검증의 복잡성을 줄이고, 정밀하고 확장 가능한 검증을 가능하게 합니다.

- **Performance Highlights**: ARC-Tran은 기존 기술보다 임의의 변형 공간에서 더 로버스트한 모델을 훈련시키며, 결과 모델의 높은 인증 정확도(certification accuracy)를 보입니다.



### DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution (https://arxiv.org/abs/2405.17357)
Comments:
          Accepted by the main conference of ACL 2024

- **What's New**: 최근 발표된 Dynamic Low-Rank Adaptation (DoRA) 방법은 Large-Scale Pre-Trained Models (대규모 사전학습 모델, 이하 PLMs)의 효율적인 파인 튜닝을 가능하게 합니다. DoRA는 각 가중치 행렬에 대한 파라미터 예산 요구를 동적으로 조정하여, 기존의 Low-Rank Adaptation (LoRA) 방법이 간과하는 최적의 파라미터 사용을 제공하고자 합니다.

- **Technical Details**: DoRA는 고계 순위의 LoRA 레이어를 단일 순위 컴포넌트로 분해하여 파라미터 예산을 동적으로 조정합니다. 이러한 방식은 특수한 작업에 따라 중요한 파라미터를 선별적으로 가지치기(pruning)하고, 제한된 파라미터 예산을 최대한 활용할 수 있도록 합니다. 기존의 LoRA 및 AdaLoRA 방식을 개선하여, PLM의 각 모듈에 맞춤형 파라미터 배분을 하는 획기적인 방법을 제안합니다.

- **Performance Highlights**: 실험 결과 DoRA는 LoRA 및 전체 모델 파인 튜닝과 비교하여 경쟁력 있는 성능을 보여주었고, 동일한 저장 파라미터 예산을 사용하면서 다양한 강력한 기준선 모델을 능가했습니다. 특히, DoRA는 전체 모델 파인 튜닝의 성능을 0.3% 이하의 학습 가능한 파라미터로 초과할 수 있음을 검증했습니다.



### Cost-efficient Knowledge-based Question Answering with Large Language Models (https://arxiv.org/abs/2405.17337)
- **What's New**: 지식 기반 질문 답변(Knowledge-based question answering, KBQA)이 여러 도메인에서 필요로 하는 지식 기반 시나리오에서 널리 사용되고 있습니다. 최근, 대형 언어 모델(Large Language Models, LLMs)은 KBQA의 성능을 향상시킬 수 있는 기회를 제공하지만, 높은 비용과 도메인 특정 지식의 부족이라는 문제가 있습니다. 이를 해결하기 위해 LLMs와 기존의 작은 지식 그래프 모델(Knowledge Graph Models, KGMs)을 결합하여 정확성과 비용 절감을 동시에 달성하려는 시도가 이루어졌습니다. 이를 위해 Coke라는 새로운 비용 효율적인 전략이 제안되었습니다.

- **Technical Details**: Coke는 제한된 예산 내에서 LLM 호출을 최소화하기 위해 다중 무장 강도 문제(Multi-Armed Bandit)를 활용하여 모델 선택을 최적화합니다. 먼저 클러스터 수준의 톰슨 샘플링(Thompson Sampling)을 사용하여 LLMs와 KGMs 중 어느 모델이 더 적합한지 예측합니다. 이후 컨텍스트 인식 정책(Context-aware policy)을 학습하여 질문의 의미에 따라 가장 적합한 전문가 모델을 선택합니다. 이 과정에서 실패로 인한 누적 비용을 기준으로 결정이 제한됩니다.

- **Performance Highlights**: 광범위한 실험 결과, Coke는 GPT-4 사용 비용을 최대 20.89% 절감하면서도 벤치마크 데이터셋에서 2.74% 더 높은 정확도를 달성하여 Pareto frontier를 개선하는 성과를 보였습니다.



### XFormParser: A Simple and Effective Multimodal Multilingual Semi-structured Form Parser (https://arxiv.org/abs/2405.17336)
Comments:
          10 pages, 3 figures, 6 tables

- **What's New**: XFormParser라는 새로운 멀티모달 및 다국어 반구조화 형태 지정 모델이 소개되었습니다. 이 모델은 다국적 상황에서 폼 문서의 핵심 정보를 추출하는 데 최적화되어 있으며, 새로운 'staged warm-up' 훈련 방식을 통해 정확도를 향상시킵니다. 또한, 다국어 폼 파싱 요건을 위한 벤치마크 데이터셋 InDFormBench도 개발되었습니다.

- **Technical Details**: XFormParser는 빌트인 사전 훈련된 언어 모델을 기반으로 하며, Semantic Entity Recognition (SER)과 Relation Extraction (RE)을 하나의 프레임워크로 통합합니다. 다국어 문서 이해에 특화된 LayoutXLM 모델을 백본으로 사용하며, Bi-LSTM과 Biaffine 디코더를 이용해 엔티티 간 관계를 학습합니다. InDFormBench는 중국어와 영어를 포함한 복합 시나리오를 위한 데이터셋으로 구성되었으며, GPT4를 통해 반자동으로 주석을 추가했습니다.

- **Performance Highlights**: XFormParser는 기존의 SOTA (state-of-the-art) 모델 대비 F1 점수가 최대 1.79% 향상되었으며, 다국어 및 제로샷 (zero-shot) 시나리오에서 뛰어난 성능을 보였습니다. InDFormBench 데이터셋을 통한 엄격한 테스트를 거쳐 신뢰성을 확인한 결과, 두 가지 작업인 SER와 RE에서 현저히 향상된 결과를 얻었습니다.



### An NLP Crosswalk Between the Common Core State Standards and NAEP Item Specifications (https://arxiv.org/abs/2405.17284)
- **What's New**: 이 논문에서는 교육 평가에서 NLP 기반 절차를 사용하여 항목 사양(item specifications)과 콘텐츠 표준(content standards)을 연결하는 방법을 제안합니다. 특히, 문장이나 텍스트의 임베딩 벡터(embedding vectors)를 사용하는 다변량 유사성(multivariate similarity)을 제시하고 이를 통해 다양한 항목 사양을 콘텐츠 표준과 일치시키는 하이브리드 회귀 절차(hybrid regression procedure)를 소개합니다. 이 방법론은 2026년 국가 교육 성취도 평가(NAEP)의 4학년 수학 항목 사양을 공통 핵심 주 교육 표준(CCSS)과 일치시키는 데 사용됩니다.

- **Technical Details**: 이 연구에서는 임베딩 벡터를 사용하여 문장의 의미적 유사성을 측정하고 이를 바탕으로 항목 사양과 콘텐츠 표준 간의 매치를 평가합니다. 회귀 모델을 통해 각 콘텐츠 표준과 여러 항목 사양 간의 유사성을 산출합니다. 논문에서는 임베딩 벡터의 통계적 특성과 코사인 유사성(cosine similarity)도 다룹니다. NLP 방법론은 객체 매핑에서 주관적 판단을 보완하고 일관성과 효율성을 향상시키기 위한 도구로 사용됩니다.

- **Performance Highlights**: 이 연구에서 제안된 절차는 CCSS 4학년 수학 표준과 2026년 NAEP 항목 사양 간의 일치를 평가하는 데 유용한 도구로 작용했습니다. 특히, 이 접근 방식은 각 CCSS 수학 표준의 77%가 적어도 하나의 NAEP 항목과 일치할 수 있음을 발견했습니다. 또한, NLP 방법론이 주제 전문가들(SME)의 작업 흐름을 개선하고 일치 연구의 시간과 자원을 절약하는 데 도움을 줄 수 있음을 보여주었습니다.



### A Library for Automatic Natural Language Generation of Spanish Texts (https://arxiv.org/abs/2405.17280)
- **What's New**: 이 논문은 최소한의 유의미한 단어(명사, 동사, 형용사 등)로부터 스페인어 문장을 자동으로 생성하는 새로운 자연어 생성(NLG) 시스템을 소개합니다. 이 시스템은 지식 기반 접근법과 통계적 접근법을 모두 활용하여 신뢰성 있는 문장 생성을 목표로 합니다. 사용자의 주요 단어 세트로부터 완전하고 일관성이 있으며 정확하게 철자된 문장을 생성할 수 있습니다. 또한, 이 시스템은 다양한 디지털 장치에 쉽게 통합될 수 있도록 설계되었습니다.

- **Technical Details**: 이 시스템은 모듈형 아키텍처로 설계되어 도메인 종속 및 비종속 컴포넌트를 분리할 수 있습니다. 시스템은 스페인어 레버시콘(aLexiS)과 문법 규칙으로 구성된 지식 기반과 인터페이스 엔진으로 구동됩니다. 텍스트 생성 과정은 내용 결정(content determination), 텍스트 구조화(text structuring), 문장 결합(sentence aggregation), 어휘화(lexicalization), 참조 표현 생성(referring expression generation), 언어적 실현(linguistic realization) 등의 여러 단계로 나뉩니다.

- **Performance Highlights**: 본 시스템은 자동 및 수동 평가(annotated evaluation)를 통해 테스트되었습니다. 생성된 NLG 라이브러리는 스페인어 SimpleNLG 라이브러리와 비교하여 우수한 성능을 보였습니다. 또한, 다양한 응용 도메인에서 활용될 잠재력이 있습니다. 예를 들어 증강 커뮤니케이션(augmentative communication) 및 행정 보고서나 뉴스의 자동 생성 등에 사용할 수 있습니다.



### On the Noise Robustness of In-Context Learning for Text Generation (https://arxiv.org/abs/2405.17264)
- **What's New**: 새로운 연구에 따르면, 대형 언어 모델(LLMs)의 인-컨텍스트 학습(In-Context Learning, ICL)이 텍스트 생성 작업에서는 노이즈가 포함된 주석으로 인해 성능이 크게 저하될 수 있습니다. 이를 해결하기 위해 연구진은 Local Perplexity Ranking (LPR)이라는 새로운 접근 방식을 제안했습니다. 이는 주석의 노이즈를 줄이고 정확한 예제를 선택하는 방식입니다.

- **Technical Details**: LPR 방법은 노이즈가 포함된 주석을 '근처 이웃'으로 대체하여 처리합니다. 노이즈 레이블이 포함된 데이터가 언어 모델의 퍼플렉시티(perplexity)를 증가시키는 현상을 분석하고, 이를 '고유 퍼플렉시티'와 '매칭 퍼플렉시티'로 분해합니다. 이를 바탕으로 주석을 선택할 때, 후보 주석들을 의미론적 공간에서 근처 이웃으로 대체하여 퍼플렉시티 순위를 계산하며, 이렇게 함으로써 노이즈에 강한 ICL을 실현합니다.

- **Performance Highlights**: LPR은 노이즈가 있는 데이터셋에서 기존 선택 방법들의 성능을 향상시켰습니다. 예를 들어, SCIQ 데이터셋에서 60%의 노이즈 레이블이 포함된 경우, TopK 방법의 정확한 매칭 점수를 29.31에서 48.06으로, 최대 18.75점 향상시켰습니다. LPR 방법은 파라미터에 민감하지 않아 실용적으로 적용이 가능하며, 다양한 LLM에 일반화될 수 있습니다.



### Assessing LLMs Suitability for Knowledge Graph Completion (https://arxiv.org/abs/2405.17249)
Comments:
          Evaluating Mixtral-8x7B-Instruct-v0.1 and gpt-3.5-turbo-0125 for Knowledge Graph Completion task with prompts formatted according to the TELeR taxonomy

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)이 지식 그래프(kg) 관련 작업, 특히 지식 그래프 완성(KGC)을 Zero- 또는 Few-Shot 상황에서 어떻게 해결할 수 있는지를 탐구하고 있습니다. Mixtral-8x7B-Instruct-v0.1와 gpt-3.5-turbo-0125 두 가지 LLM을 사용하여, 고정된 지식 그래프에서 사용될 수 있는 프롬프트들을 실험합니다. 특히 TELeR 분류법에 따라 제로 및 원샷 학습 문맥에서 프롬프트를 구성하여 테스트합니다.

- **Technical Details**: 이 연구에서는 LLM을 위한 프롬프트 엔지니어링, 데이터를 정리하는 방식, 그리고 평가 메트릭의 엄격성과 유연성 등 다양한 측면을 다룹니다. 실험에 사용된 데이터셋은 어려움의 정도가 다른 두 가지로 구성되며, 이는 TOD 시스템(training에서 추출된 구문들을 포함)에서 유래한 것입니다. 프롬프트는 직접 프롬프트(Direct Prompting; DP), 컨텍스트 학습(In-Context Learning; ICL), 그리고 Chain of Thought(COT) 기술을 활용합니다.

- **Performance Highlights**: 평가는 엄격하고 유연한 방식의 측정 지표를 통해 이루어졌습니다. 결과에 따르면, 충분한 정보를 포함한 프롬프트가 사용된다면 LLM이 KGC와 같은 작업에 적합할 수 있음을 보여줍니다. 특히, Mixtral-8x7B-Instruct-v0.1와 gpt-3.5-turbo-0125는 모두 다양한 수준의 프롬프트 복잡성에서 높은 정확도와 F1 점수를 기록했습니다.



### RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness (https://arxiv.org/abs/2405.17220)
Comments:
          Project Website: this https URL

- **What's New**: RLAIF-V라는 새로운 프레임워크가 도입되었습니다. 이 프레임워크는 완전히 오픈 소스 패러다임에서 신뢰성을 크게 향상시키며, 고품질의 피드백 데이터를 최대한 활용하고, 온라인 피드백 학습 알고리즘을 통해 MLLM(다중 모드 대형 언어 모델)들을 정렬합니다.

- **Technical Details**: RLAIF-V는 두 가지 핵심 혁신을 활용합니다. 첫째, 고품질 피드백 데이터를 생성하기 위해 새로운 비혼돈 전략(deconfounded candidate response generation)을 사용하며, 이것은 동일한 조건에서 여러 샘플링 디코딩을 통해 후보 응답을 생성하여 신뢰성 차이를 정확히 드러냅니다. 둘째, 피드백 데이터의 정확성을 높이기 위해 분할 및 정복 접근법(divide-and-conquer approach)을 채택하며, 이는 전체 응답을 클레임으로 나누어 평가하는 방식을 취합니다.

- **Performance Highlights**: RLAIF-V는 34B 모델을 레이블러로 사용하여 7B 모델에서 객체 환각(object hallucination)을 82.9% 감소시키고, 전반적인 환각(overall hallucination)을 42.1% 감소시켰습니다. 또한, 12B 모델이 스스로의 피드백을 학습해 29.5% 이하의 전반적인 환각율을 달성, GPT-4V(45.9%)를 크게 앞섰습니다.



### Efficient multi-prompt evaluation of LLMs (https://arxiv.org/abs/2405.17202)
- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 성능을 다양한 프롬프트(질의) 템플릿으로 평가하는 방법을 제시합니다. 현재의 벤치마크는 제한된 수의 프롬프트 템플릿에 의존하여, 모델의 실제 능력을 완전히 반영하지 못하며 결과의 재현성을 저해할 수 있습니다. 이를 해결하기 위해, PromptEval이라는 방법을 도입해 다양한 프롬프트 변형에 걸친 성능 분포를 추정합니다.

- **Technical Details**: PromptEval은 평가 예산을 고려하면서도 여러 프롬프트 템플릿과 예제를 통해 정확한 성능 추정을 가능하게 합니다. 이 방법은 Item Response Theory(IRT)에 기반하여 각 예제와 프롬프트 템플릿 간의 상관관계를 활용합니다. 평가 예산을 단일 프롬프트 평가의 두 배로 설정한 경우에서도 성능 분포를 정확히 추정할 수 있습니다.

- **Performance Highlights**: PromptEval은 MMLU(Massive Multitask Language Understanding), BIG-bench Hard (BBH), 및 LMentry 벤치마크에서 유효성을 입증했습니다. 예를 들어, MMLU에서는 100가지 프롬프트 템플릿에 대해 평군 성능 평가 예산의 두 배만으로 성능 분포를 정확히 추정할 수 있습니다. 또한, 새로운 평가 데이터도 제공하여 LLMs의 프롬프트 민감성을 첫 대규모로 연구했습니다.



### Stop! In the Name of Flaws: Disentangling Personal Names and Sociodemographic Attributes in NLP (https://arxiv.org/abs/2405.17159)
- **What's New**: 이 논문은 자연어 처리(NLP) 분야에서 개인의 이름을 사용하여 사회인구통계학적 특성을 추론하는 문제를 다루고 있습니다. 저자들은 다른 학문 분야에서의 이름 연구를 기반으로, 이름과 사회인구통계학적 속성을 연관짓는 과정에서 발생하는 방법론적 및 윤리적 문제를 살펴봅니다.

- **Technical Details**: 연구는 이름의 유효성 문제(예: 체계적 오류, 구성 타당성)와 윤리적 문제(예: 피해, 차별적 영향, 문화적 민감도 부족)를 다룹니다. 특히 성별과 인종과 같은 인기 있는 카테고리에 초점을 맞추고 있습니다. 이어서, NLP에서 이름을 사용한 연구에 대한 배경을 제공하며, 다양한 문제와 윤리적 우려를 검토합니다.

- **Performance Highlights**: 논문은 향후 연구자들이 유효성 및 윤리적 문제를 피할 수 있도록 안내 질문과 규범적 권고 사항을 제시합니다. 연구는 이름과 관련된 분석에서 발생할 수 있는 여러 가지 피해를 고려하며, 데이터 과학과 윤리, 철학적 배경을 가진 저자들이 기여하였습니다. 특히, 성소수자 및 다인종 저자들의 경험을 반영하여 성별과 인종 관련 문제를 상세히 다룹니다.



### TEII: Think, Explain, Interact and Iterate with Large Language Models to Solve Cross-lingual Emotion Detection (https://arxiv.org/abs/2405.17129)
Comments:
          (Under review) Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis

- **What's New**: 이번 연구는 EXALT@WASSA 2024의 트윗 감정 탐지 과제에 관한 것입니다. 우리는 다양하고 문화적인 차이점을 포함한 다국어 감정 탐지 문제를 다루기 위해, 대규모 언어 모델 (LLM)과 다국어 임베딩을 사용하는 전통적인 기계 학습 모델을 실험했습니다. 새로운 접근 방식으로 Multi-Iteration Agentic Workflow와 Multi-Binary-Classifier Agentic Workflow를 도입했습니다. 이를 통해 다양한 모델들을 결합한 앙상블 기법이 단일 모델보다 더 높은 성능을 보였습니다.

- **Technical Details**: 우리는 세 가지 주요 모델 클래스를 탐구했습니다. 하나는 다국어 임베딩을 사용한 KNN과 BiLSTM 모델, 다른 하나는 OpenAI의 GPT3.5 및 GPT4 및 Anthropics의 Claude3와 같은 대규모 언어 모델을 사용한 것입니다. 또 다른 접근 방식은 Multi-Iteration Agentic Workflow와 Multi-Binary-Classifier Agentic Workflow로 여러 LLM들을 이용하여 성능을 향상시키는 방법입니다. 주요 실험 방법에는 fine-tuning, zero-shot learning, few-shot learning이 포함되며, 모든 코드는 GitHub(https://github.com/cl-victor1/EXALT_2024)에 공개될 예정입니다.

- **Performance Highlights**: 우리 시스템은 감정 탐지 서브태스크에서 평가 셋에서 F1-score 0.6046을 달성하여 베이스라인보다 0.16 F1-score 이상 높은 성능을 보여주었습니다. 또한 앙상블 기법이 단일 모델을 사용한 경우보다 더 높은 성능을 나타냈습니다. 특히 GPT4 기반 모델이 개발용 데이터세트에서 Claude3 기반 모델보다 더 나은 성능을 보였으며, 최종 평가에서도 주 모델로 사용되었습니다.



### Mixtures of Unsupervised Lexicon Classification (https://arxiv.org/abs/2405.17116)
Comments:
          A draft on lexicon classification unsupervised learning. It shows that aggregating lexicon scores is equivalent to a finite mixture of multinomial Naive Bayes models. A very preliminary work of a few days man-hours, like a weekly report/note, but might be useful

- **What's New**: 본 논문에서는 혼합 모델 (mixture model) 방식의 모멘트 기법을 사용한 비지도 (unsupervised) 용어 분류법을 디리클레 과정 (Dirichlet process)을 도입하여 발표하였습니다.

- **Technical Details**: 이 연구는 각 클래스에 해당하는 빈도를 계산하여 추정된 분류 확률을 사용하는 용어 분류 방식을 다룹니다. 기존 최적의 베이즈 위험 분류 규칙을 기반으로, 이 논문은 디리클레 과정을 통한 확률 모델링을 도입하여 데이터를 그룹화하고 토픽 모델링을 시도하였습니다. 기본적으로 나이브 베이즈 모델 (multinomial Naïve Bayes)과 혼합 모형 (mixture models), 디리클레 분포 (Dirichlet distribution)를 사용해 더 복잡한 밀도를 모델링할 수 있는 혼합 모델로 구성됩니다.

- **Performance Highlights**: 혼합 모델 및 디리클레 과정을 통한 새로운 접근법은 단어의 발생 빈도가 아닌 출현 여부를 모델링하여 더욱 예측력이 높아졌습니다. 또한 단어의 교차 표현 빈도와 모멘트 기법을 사용해 γ 파라미터를 최적화하였습니다.



### Empowering Character-level Text Infilling by Eliminating Sub-Tokens (https://arxiv.org/abs/2405.17103)
Comments:
          Accepted to ACL 2024 (main conference)

- **What's New**: 새로운 논문에서는 문자 수준의 채워넣기(infilling) 작업을 위해 분할된 토큰(sub-token)이 예측 단계에서 모델 성능을 저하시키는 문제를 해결하기 위해 FIM-SE(Fill-In-the-Middle with both Starting and Ending character constraints) 방법을 소개합니다. 이 방법은 중간 채워넣기를 수행할 때 시작과 끝 문자를 엄격하게 통제하여 문자를 기준으로 텍스트를 채워넣습니다. 또한 두 개의 특수 토큰을 도입하여 불완전한 줄의 나머지를 나타내어, 생성 지침을 제공합니다.

- **Technical Details**: FIM-SE 방법은 Transformer 기반의 디코더만 사용하여 인필 작업을 수행합니다. 기존의 FIM(Fill-In-the-Middle) 기법을 개선하여, 문자를 기준으로 텍스트를 나누고 L-Prefix와 F-Suffix라는 두 개의 특수 토큰을 도입하여 텍스트 생성의 일관성을 확보합니다. 즉, 랜덤으로 분할된 문자의 마지막 줄과 접미사의 첫 줄을 마크하여 모델이 이들 사이를 채우는 방식입니다. 이를 통해 서브-토큰 예측을 피하고, 출력이 주어진 문맥과 일치하도록 합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법은 기존 방법보다 문자 수준을 기준으로 한 인필 작업에서 더 높은 성능을 보였습니다. 특히, Code Llama 13B 기반으로 Humaneval 랜덤 스팬 인필 작업에서 8.8% 향상, 싱글 라인과 멀티 라인 인필 작업에서 각각 11.5%와 10.7%의 성능 향상을 기록했습니다. 또한, 코드 생성 작업에서도 최소한의 성능 저하를 유지했습니다.



### Tokenization Matters! Degrading Large Language Models through Challenging Their Tokenization (https://arxiv.org/abs/2405.17067)
Comments:
          17 pages, 3 figures, this paper is submitted to neurips 2024

- **What's New**: 최근의 연구는 토큰화(tokenization) 절차의 문제점이 LLMs (Large Language Models)의 언어 이해와 생성 능력을 저해하고 있음을 밝혔습니다. 이를 보여주기 위해 다양한 오픈 소스 LLMs의 어휘를 바탕으로 구성된 적대적인 데이터셋(Adversarial Dataset for Tokenizer, ADT)을 도입했습니다. ADT는 수동으로 구성된 ADT-Human과 자동으로 생성된 ADT-Auto의 두 하위 집합으로 구성되어 있습니다.

- **Technical Details**: LLM의 토큰화 결함을 강조하기 위해 ADT는 주류 LLM들의 어휘를 바탕으로 수동 및 자동으로 데이터를 생성해 의미 없는 응답을 유도합니다. Byte-Pair Encoding (BPE), WordPiece, Unigram과 같은 주류 토큰화 알고리즘이 적용되었으며, 이들 각각이 다루는 방식에서 문제점이 파생될 수 있음을 연구했습니다. ADT-Auto는 LLM의 어휘를 추출하고 성능에 영향을 미치는 'trap words'를 식별하여 자동으로 적대 데이터를 생성합니다.

- **Performance Highlights**: 실험 결과, ADT는 GPT-4o, Llama-3, Qwen2.5-max 등 주요 LLM들의 토큰화를 훌륭히 도전해 그들의 응답 정확성을 크게 떨어뜨렸습니다. 이는 LLM의 토큰화 절차에 존재하는 근본적인 문제를 드러내며, 이를 통해 LLM 성능을 향상시키기 위한 후속 연구에 중요한 통찰을 제공합니다.



### Unifying Demonstration Selection and Compression for In-Context Learning (https://arxiv.org/abs/2405.17062)
- **What's New**: 새로운 ICL(In-context learning) 프레임워크인 UniICL을 제안합니다. 이 프레임워크는 단일 고정된 LLM을 사용하여 데모 선택, 압축, 응답 생성을 통합적으로 수행합니다. 이는 기존 방법들보다 더 효율적으로 메모리를 사용하며, OoD(Out-of-Distribution) 문제를 자연스럽게 피할 수 있습니다.

- **Technical Details**: UniICL은 실제 데모와 추론 텍스트 입력을 각각 짧은 가상 토큰으로 투영합니다. 그런 다음, 잠재 공간 내에서 의미 유사성을 측정하여 적절한 데모를 선택합니다. 마지막으로, 선택된 가상 데모와 함께 추론 텍스트 입력을 동일한 고정 LLM에 넣어 응답을 생성합니다. 이 과정에서 단순한 프로젝션 레이어로 구성된 17M의 학습 가능한 파라미터만 사용합니다.

- **Performance Highlights**: 실험 결과 UniICL은 IMDb 데이터셋에서 4-shot ICL에서 64-shot ICL로 베이스라인을 확장하면서 12배의 압축을 효율적으로 수행하였습니다. 또한, UniICL은 8%의 추가 추론 지연 시간만으로 길이 문제를 상당히 완화했습니다. 다양한 과업에서 인기 있는 dense retriever보다 더 적절한 데모를 선택하는 데 성공했습니다.



### ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation (https://arxiv.org/abs/2405.17057)
- **What's New**: ReflectionCoder는 컴파일러 피드백을 통합하여 생성한 반영 시퀀스(reflection sequences)를 활용해 일회성 코드 생성 성능을 향상시키는 혁신적인 접근 방식을 소개합니다. 이를 통해 코드 자동 완성 및 수학적 추론과 같은 다양한 작업에서의 코드 생성 성능을 강화할 수 있습니다.

- **Technical Details**: ReflectionCoder는 반영 자기 증류(reflection self-distillation) 및 동적으로 마스킹된 증류(dynamically masked distillation)라는 새로운 기법을 제안하여 반영 시퀀스를 효과적으로 활용합니다. 이 새로운 방법은 컴파일러의 피드백을 통합하여 생성된 반영 시퀀스를 통해 모델의 성능을 극대화합니다.

- **Performance Highlights**: HumanEval (+)에서 82.9 (76.8), MBPP (+)에서 84.1 (72.0)의 pass@1 성능을 달성한 ReflectionCoder-DeepSeek-Coder-33B 모델은 GPT-3.5-Turbo 및 Claude-3-opus와 동등한 수준을 보이고 있으며, 초기 GPT-4도 능가합니다. 이는 ReflectionCoder가 코드 도메인 외에도 긴 추론 경로가 필요한 다른 도메인에서도 혜택을 줄 수 있다는 가능성을 시사합니다.



### SelfCP: Compressing Long Prompt to 1/12 Using the Frozen Large Language Model Itself (https://arxiv.org/abs/2405.17052)
- **What's New**: 이 논문은 SelfCP를 소개합니다. SelfCP는 대형 언어 모델(LLM)을 사용하여 긴 프롬프트(prompt)를 컴팩트한 가상 토큰으로 압축합니다. 이 접근법은 LLM 자체를 인코더와 디코더로 사용하여 프롬프트를 압축하고 응답을 생성합니다. 이 방법은 조건부 및 무조건부 압축을 모두 지원하며 다양한 LLM 백본에서 활용할 수 있습니다.

- **Technical Details**: SelfCP는 동결된(frozen) LLM을 두 번 사용합니다. 첫 번째로 인코더로서 긴 프롬프트를 압축하고 두 번째로 디코더로서 응답을 생성합니다. 구체적으로, 긴 프롬프트에 특별한 토큰을 넣어 LLM이 가상 토큰을 생성하도록 신호를 보냅니다. 이후 가상 토큰은 압축되지 않은 프롬프트와 결합되어 동일한 LLM에 입력되어 응답을 생성합니다. SelfCP는 단순한 프로젝션 레이어와 두 개의 특별한 임베딩을 통해 17M의 학습 가능한 매개변수를 갖추고 있습니다.

- **Performance Highlights**: SelfCP는 원래 프롬프트의 길이를 1/12로 효율적으로 대체할 수 있으며, 이를 통해 약 7%의 추가 시간 비용만 필요로 합니다. 실험 결과, SelfCP는 다양한 작업 도메인에서 원래 긴 프롬프트를 대신하여 가상 토큰을 효과적으로 사용하며, 인-도메인 및 아웃-도메인 작업 모두에서 강력한 성능을 보였습니다.



### BWArea Model: Learning World Model, Inverse Dynamics, and Policy for Controllable Language Generation (https://arxiv.org/abs/2405.17039)
- **What's New**: 새로운 BWArea 모델은 Broca's와 Wernicke's 부위의 뇌신경 메커니즘에서 영감을 받아, 자연어 생성(NLP)을 결정하기 위한 작업으로 재구상합니다. 이 모델은 언어 생성 과정에서 더 높은 제어력을 제공하며, 특히 LLM 기반 모델이 갖는 완전 자회귀(fully auto-regressive)의 한계를 극복하려 합니다.

- **Technical Details**: BWArea 모델은 세 가지 주요 컴포넌트로 구성됩니다: 언어 세계 모델(language world model), 역동적 모델(inverse dynamics model), 인지 정책(cognitive policy)입니다. 언어 세계 모델은 인지적 결정을 기반으로 토큰을 생성하며, 역동적 모델은 각 토큰의 기저에 있는 인지적 의도를 추론합니다. 이를 통해 언어 시퀀스를 결정 궤적으로 변환하여, 행동 복제(behavior cloning)을 통해 학습할 수 있도록 합니다

- **Performance Highlights**: BWArea 모델은 사전 훈련 및 미세 조정(pre-training and fine-tuning)을 통해 평가한 결과, 기존 LLM들과 비교하여 경쟁력 있는 성능을 보였습니다. 특히, '더러운' 데이터가 포함되었을 때 성능 저하가 적거나 오히려 향상되었으며, 이는 데이터 정리와 레이블링의 노력을 줄이는 데 큰 이점이 됩니다. 또한, TextWorld와 BigBench Hard의 대부분의 작업에서 기존 LLM을 능가하는 성능을 보였습니다. 특히, 강화 학습 방법론을 통해 인지 정책을 조정하면 더 높은 제어력을 발휘할 수 있습니다.



### The Multi-Range Theory of Translation Quality Measurement: MQM scoring models and Statistical Quality Contro (https://arxiv.org/abs/2405.16969)
Comments:
          working paper, 20 pages

- **What's New**: 2024년은 다차원 품질 지표(MQM) 프레임워크의 10주년을 맞이합니다. 최근 MQM 평의회는 새로운 선형 및 비선형 점수 모델을 발표했습니다. 이전에는 원시 점수 모델만 사용되었으나, 이번 발표를 통해 MQM의 최신 발전 사항이 자세히 소개되었습니다.

- **Technical Details**: MQM 프레임워크는 오류 유형(error typology)과 점수 모델(scoring model) 두 가지 기둥으로 구성됩니다. 오류 유형은 사전에 정의된 오류 타입과 심각도 수준을 사용하여 번역 오류를 주석 처리합니다. 점수 모델은 이 주석 데이터를 사용해 품질 점수를 계산하며, 최근에는 선형 보정 점수 모델과 비선형 점수 모델이 추가되었습니다.

- **Performance Highlights**: MQM는 번역 품질 평가를 위해 통계적 품질 관리(Statistical Quality Control)를 활용해야 함을 강조하고 있으며, 이는 특히 열 단위의 소규모 샘플에서 중요합니다. 또한, MQM는 평가 과정에서 샘플 크기에 따라 다양한 접근 방식을 적용해야 한다고 제안합니다.



### Exploring the LLM Journey from Cognition to Expression with Linear Representations (https://arxiv.org/abs/2405.16964)
Comments:
          Published in ICML 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 인지적 및 표현적 능력의 진화와 상호작용에 대한 심도 있는 분석을 제공합니다. 특히 Baichuan-7B와 Baichuan-33B라는 고급 다국어(중국어와 영어) LLM 시리즈를 중심으로 조사합니다. 인지적 능력은 모델이 신경 출력 벡터를 통해 전달하는 정보의 양과 질로 정의되며, 표현적 능력은 모델이 단어 수준의 출력을 생성할 능력으로 정의됩니다. 인지적 능력은 주로 초기 트레이닝(Pretaining) 단계에서 확립되고, 표현적 능력은 주로 SFT(지도형 미세 조정)와 RLHF(사람 피드백을 통한 강화 학습) 단계에서 발전된다는 것을 발견했습니다.

- **Technical Details**: 연구는 인지적 및 표현적 능력을 선형 표현(linear representation)을 통해 정의하고 이를 양적으로 분석합니다. 인지적 능력은 인간 인지의 신경 신호 처리와 유사하게 네트워크 내부에서 전달되는 정보의 양과 질로 정의됩니다. 표현적 능력은 모델이 단어 수준의 출력을 생성할 수 있는 능력으로 정의됩니다. 연구에 따르면, 인지적 능력은 주로 초반 트레이닝 단계에서 확립되며, 표현적 능력은 주로 SFT와 RLHF 단계에서 발전합니다. 또한 특정 기술들(예: few-shot learning, repeated sampling)이 인지적 및 표현적 능력 간의 격차를 줄이는 데 효율적임을 발견했습니다.

- **Performance Highlights**: 1) 인지적 능력과 표현적 능력은 다른 속도로 진화하며, 인지적 능력은 주로 초기 트레이닝 단계에서 확립되고, 표현적 능력은 SFT와 RLHF 단계를 통해 발전됩니다. 2) 인지적 능력과 표현적 능력 사이에는 강한 통계적 상관관계가 있습니다. 3) 특정 접근법(예: prompt engineering)이 모델의 표현적 능력과 인지적 능력 간의 격차를 줄이는 데 효과적일 수 있습니다.



### Empowering Large Language Models to Set up a Knowledge Retrieval Indexer via Self-Learning (https://arxiv.org/abs/2405.16933)
- **What's New**: PG-RAG(Pseudo-Graph Retrieval-Augmented Generation)은 대형 언어 모델에 실시간 지식을 주입할 수 있는 효율적인 솔루션을 제공합니다. 이는 LLMs(대형 언어 모델)를 학생처럼 대하고, 자율적으로 읽고 사실을 기록하도록 격려하여 심플하고 효율적인 인덱스를 구축하는 방식입니다. PG-RAG은 공통 주제나 보완적인 사실들을 연결하여 가짜-그래프 데이터베이스를 형성합니다.

- **Technical Details**: PG-RAG은 LLMs가 제공된 원문 자료를 읽고 자체적으로 기억 인덱스를 작성하도록 유도합니다. 생성된 인덱스는 공통 주제나 보완적인 사실을 통한 연결을 통해 'pseudo-graph'를 형성합니다. 검색 단계에서는 사람이 노트를 훑어보는 것처럼 행동하여 사실 경로를 식별하고 관련 문맥을 탐색합니다. '많이 간 길이 최선의 길'이라는 원칙에 따라 가장 신뢰할 수 있는 경로를 통합하여 구조화된 서브-그래프를 제공합니다.

- **Performance Highlights**: PG-RAG은 세 가지 특수한 질문 응답 데이터셋에서 검증되었습니다. 단일 문서 작업에서, PG-RAG는 모든 주요 평가 메트릭에서 현재 최고의 기준인 KGP-LLaMA를 능가하며 평균 11.6%의 성능 향상을 보였습니다. 특히 BLEU 점수는 약 14.3%, QE-F1 메트릭은 23.7% 향상되었습니다. 다중 문서 시나리오에서 평균 메트릭은 최소 2.35% 높았고, BLEU 점수와 QE-F1 메트릭은 각각 약 7.55%와 12.75%의 안정적인 향상을 보였습니다.



### Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words? (https://arxiv.org/abs/2405.16908)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 자신의 내재된 불확실성을 자연어로 표현할 수 있어야 한다고 주장합니다. 모델이 동일한 질문에 상반된 답을 줄 확률이 동등하다면, 모델의 응답은 이를 반영하여 '확신할 수는 없지만, ~인 것 같다' 형식으로 답할 것을 제안합니다. 이를 통해 LLM의 신뢰성을 향상시키려는 시도입니다.

- **Technical Details**: 이 연구에서의 '신뢰 가능한 응답 불확실성(faithful response uncertainty)'는 모델이 표현하는 주장에 대한 내재적인 확신과 해당 주장을 전달하는 결정적인 방식 사이의 차이에 기초하고 있습니다. 연구팀은 적절한 불확실성 표현을 유도하는 여러 프롬프트 방식으로 주요 LLM들(Gemini, GPT-3.5, GPT-4)을 평가하였습니다. 결과적으로, LLM들이 자연어로 자신들의 불확실성을 신뢰할 만하게 전달하는 데에 어려움이 있다는 결론을 내렸습니다.

- **Performance Highlights**: 표준 디코딩 방법을 사용할 경우, 대부분의 모델이 상당한 내재적 불확실성이 존재함에도 불구하고 결정적으로 대답하는 경향이 나타났습니다. 불확실성을 표현하도록 유도하는 프롬프트 방식을 사용할 때, 가끔 불확실성 표현이 유도되긴 했으나, 이러한 표현이 모델의 내재적 불확실성과 잘 일치하지 않는다는 점을 발견했습니다. 이는 LLM의 신뢰성을 높이기 위해 더 나은 정렬이 필요함을 시사합니다.



### Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching (https://arxiv.org/abs/2405.16884)
Comments:
          Under revision. Code is available at this https URL

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 활용한 엔티티 매칭(Entity Matching, EM)이 단순한 이진 매칭 패러다임을 넘어 다양한 기록 간 상호작용을 포함하는 방법을 조사합니다. 연구에서는 매칭, 비교 및 선택이라는 세 가지 대표적인 전략을 비교하고, 다양한 시나리오에서 각각의 이점과 도전을 분석합니다. 이를 통해, 여러 전략과 LLMs의 조합을 활용한 새로운 Compositional Entity Matching (ComEM) 프레임워크를 제안합니다.

- **Technical Details**: 현재 LLM 기반 엔티티 매칭 방법은 각 레코드를 독립적으로 분류하는 전략을 취해 전반적인 일관성을 무시하는 경우가 많습니다. 연구에서는 매칭, 비교, 선택이라는 세 가지 전략을 통해 레코드 간의 상호작용을 포함하는 엔티티 매칭 방식을 제안합니다. 매칭 전략은 전통적인 방식으로 레코드를 분류하고, 비교 전략은 두 레코드 중 어느 것이 앵커 레코드와 더 잘 맞는지 판별하며, 선택 전략은 목록에서 가장 잘 맞는 레코드를 선택합니다. 이러한 전략들을 조합하여 ComEM 프레임워크를 구축하고, 중간 단계에서는 로컬 매칭 및 비교 전략을 사용하고, 최종 단계에서는 글로벌 선택 전략을 사용합니다.

- **Performance Highlights**: 실험결과, ComEM은 다양한 데이터 세트에서 상당한 성능 향상을 이루었으며, 기존 매칭 전략에 비해 F1 스코어가 평균 13.39% 향상되었습니다. 또한, ComEM은 단일 선택 전략 대비 평균 F1 스코어를 최대 8.46% 더 향상시키는 동시에 비용도 절감할 수 있었습니다.



### Can We Trust LLMs? Mitigate Overconfidence Bias in LLMs through Knowledge Transfer (https://arxiv.org/abs/2405.16856)
- **What's New**: 이 연구는 LLMs (Large Language Models)의 과신(過信 - Overconfidence) 편향을 완화하여 모델의 신뢰성을 향상시키는 방법을 탐구합니다. 연구진은 'Chain of Thoughts (CoT)'를 활용한 지식 전이(knowledge transfer, KT) 기법을 도입했으며, 이를 통해 '큰' LLMs가 '작은' LLMs에게 정밀하고 순차적인 사고 경로를 통해 지식을 전달합니다.

- **Technical Details**: 제안된 방법은 먼저 큰 모델(GPT-4)이 질문에 대한 CoT를 생성한 후, 작은 모델(Vicuna-7B)을 그 경로를 따라 미세 조정합니다(fine-tuning). 이 과정에는 감정 분석과 선택형 질문들에 대한 세부적인 단계별 분석이 포함됩니다. 예로, 모델이 답변을 도출하는 과정과 그 답변에 대한 확신 수준을 명시하도록 지시합니다.

- **Performance Highlights**: 실험 결과, KT 방법은 다른 미세 조정 방법들(zero-shot 및 QA fine-tuning)보다 성능이 뛰어남을 확인했습니다. LLaMA 2-7B 모델의 Truthfulqa 데이터셋에서 KT 방법은 정확도(accuracy)가 vanilla 모델보다 64.4%, QA 방법보다 47.8% 높았습니다. 또한, 과신 편향 비율(ROB)을 효과적으로 줄였으며, 기대 보정 오차(ECE)도 현저히 감소시켰습니다.



### Perturbation-Restrained Sequential Model Editing (https://arxiv.org/abs/2405.16821)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 지식을 광범위한 재교육 없이 업데이트하는 모델 편집 분야에 대해 연구했습니다. 특히, 연속적 모델 편집 시 모델의 일반적 능력을 저하시키는 문제를 해결하기 위해 새로운 프레임워크 PRUNE (Perturbation Restraint on Upper bouNd for Editing)을 제안했습니다.

- **Technical Details**: 이 연구는 먼저 행렬의 조건수(condition number)가 연속적 모델 편집에서 일반적 능력에 영향을 미치는 중요한 요인으로 작용한다는 것을 이론적으로 분석했습니다. 조건수는 수치적 민감도를 나타내므로 편집 후 대형 언어 모델에 저장된 원래 지식 연관성이 얼마나 방해받는지를 나타낼 수 있습니다. 이를 바탕으로 PRUNE 프레임워크는 편집된 행렬의 조건수를 제한하여 원래 지식 연관성의 방해를 줄이고, 모델의 일반적 능력을 유지하는 방식으로 동작합니다.

- **Performance Highlights**: 세 가지 LLMs(GPT-2 XL, LLaMA-2, LLaMA-3)와 세 가지 대표적 편집 방법(MEND, ROME, MEMIT), 네 가지 다운스트림 작업(추론, 요약, 오픈 도메인 QA, 자연어 추론)을 사용한 실험 결과, PRUNE이 연속적 모델 편집 시에도 상당한 일반적 능력을 유지하면서 거의 모든 편집 성능을 유지할 수 있음을 보여줍니다. 본 논문은 이론적 분석, PRUNE 프레임워크 제안, 그리고 광범위한 실험을 통한 검증이라는 세 가지 주요 기여를 하고 있습니다.



### Performance evaluation of Reddit Comments using Machine Learning and Natural Language Processing methods in Sentiment Analysis (https://arxiv.org/abs/2405.16810)
Comments:
          11 pages, 5 figures, to be published in Computational and Experimental Simulations in Engineering - Proceedings of ICCES 2024 - Volume 1

- **What's New**: 새로운 연구는 GoEmotions 데이터셋을 활용하여 58,000개의 댓글을 바탕으로 감성 분석(sentiment analysis) 방법을 평가했습니다. 기존 Google 팀의 연구가 두 개의 모델만을 분석한 것과는 달리, 본 연구는 다양한 모델을 평가하였으며, 특히 감성 분류(tasks)에서 RoBERTa 모델이 우수한 성능을 보였음을 밝혔습니다.

- **Technical Details**: 본 연구는 Naive Bayes, SVM(Support Vector Machines)과 같은 전통적인 분류기(classifiers)부터, 최신 트랜스포머 기반 모델들(BERT, RoBERTa, GPT)을 평가 대상에 포함했습니다. 또한 단순한 정확도(accuracy) 평가를 넘어, 다양한 감정 분류 기준과 효율성(computational efficiency)도 고려하여 종합적인 평가를 수행했습니다.

- **Performance Highlights**: RoBERTa 모델은 미세 감성 분류(fine-grained sentiment classification) 작업에서 지속적으로 베이스라인 모델들보다 우수한 정확도를 보였습니다. 이는 RoBERTa 모델이 감성 분석 기능을 진보시키는 데 중요한 역할을 할 수 있음을 보여줍니다.



### Entity Alignment with Noisy Annotations from Large Language Models (https://arxiv.org/abs/2405.16806)
- **What's New**: LLM4EA 프레임워크가 도입되어 대형 언어 모델 (Large Language Models, LLMs)을 이용한 실질적이고 자동화된 엔티티 정렬 (Entity Alignment, EA)을 가능하게 합니다. 이를 통해 자율적으로 생성된 레이블을 활용하여 복잡한 지식 그래프 (Knowledge Graphs, KGs)를 병합할 수 있습니다.

- **Technical Details**: LLM4EA는 LLM에서 생성된 레이블의 노이즈를 극복하기 위해 몇 가지 혁신적인 접근법을 사용합니다. 첫째, 능동 학습 정책을 통해 가장 가치 있는 엔티티를 우선적으로 선택하여 주석 공간을 줄입니다. 둘째, 비지도 레이블 정제기 (unsupervised label refiner)를 도입하여 확률적 추론을 통해 레이블의 정확성을 지속적으로 향상시킵니다. 이러한 최적화는 기본 EA 모델로부터 피드백을 받아 iteratively 수행됩니다.

- **Performance Highlights**: 네 개의 벤치마크 데이터셋에서 LLM4EA의 효과, 강건성, 효율성을 입증한 광범위한 실험 결과가 있습니다. 이 프레임워크는 기존의 방법들보다 큰 폭으로 성능이 뛰어나며, 각 구성 요소가 총체적인 성능 향상에 의미 있게 기여하는 시너지 효과를 보입니다.



### AutoCV: Empowering Reasoning with Automated Process Labeling via Confidence Variation (https://arxiv.org/abs/2405.16802)
Comments:
          20 pages, 1 figure, 13 tables

- **What's New**: 새로운 연구인 Automated Process Labeling via Confidence Variation (AutoCV)는 대형 언어 모델(LLMs)의 추론 능력을 향상시키기 위해 자동으로 추론 단계를 주석 처리하는 방법을 제안합니다. 이 방법은 최종 답변의 올바름에 기초한 검증 모델을 학습하여, 자동으로 과정 주석을 생성합니다.

- **Technical Details**: AutoCV는 검증 모델을 사용하여 각 추론 단계에 확률 점수를 할당하고, 이러한 점수의 변화를 통해 자동으로 추론 과정을 주석 처리합니다. 이에 따라 많은 수작업 주석이나 높은 계산 비용 없이도 정확한 추론 단계 주석을 생성할 수 있습니다.

- **Performance Highlights**: AutoCV가 생성한 주석 데이터를 사용하여 수학 및 상식 추론 작업에서 검증 모델의 성능을 유의미하게 향상시켰습니다. 이는 다섯 개의 데이터셋에서의 실험을 통해 입증되었으며, 계산 자원 및 수작업 개입을 크게 줄이면서도 높은 정확성을 유지했습니다.



### Large Scale Knowledge Washing (https://arxiv.org/abs/2405.16720)
- **What's New**: 새로운 연구에서는 대규모 언어 모델(Large Language Models, LLMs)에서 특정 사실적 지식을 '비학습(unlearn)'하는 문제를 다루고 있습니다. 이는 개인정보, 민감한 정보, 저작권 보호 콘텐츠 등을 모델이 기억하지 않도록 하기 위함입니다. 이는 기존의 모델 업데이트 방식이 모델의 유창성과 추론 능력을 손상시킬 수 있다는 단점을 해결할 필요에서 비롯되었습니다. 새로운 방법론 'LAW(Large Scale Washing)'은 디코더만을 사용하는 대규모 언어 모델에서 MLP 계층을 업데이트하여 지식 세척(knowledge washing)을 수행합니다.

- **Technical Details**: LAW는 모델 편집 기법에 영감을 받아 지식과 추론 능력이 분리 가능하다는 가설하에 설계되었습니다. 이를 위해 특정 MLP 계층의 가중치를 업데이트하는 새로운 목표를 도출했습니다. 이는 기존의 백프로파게이션(backpropagation) 방법이 모델 성능을 저하시킬 수 있다는 문제를 해결합니다. LAW는 특정 지식을 비학습하는 새로운 목적을 설정하고, 선택된 MLP 계층의 가중치를 조정하여 모델의 추론 능력을 유지하면서 목표 지식을 삭제합니다.

- **Performance Highlights**: 실험 결과, LAW는 목표 지식을 효과적으로 비학습하면서도 모델의 추론 능력을 유지함을 입증했습니다. 두 개의 소규모 데이터셋과 위키피디아 트리플릿에서 파생된 332,036개의 사실을 포함하는 대규모 데이터셋을 사용하여 평가한 결과, LAW는 가장 철저한 지식 세척을 달성했습니다. 또한, 여러 추론 작업에서 모델이 탁월한 성능을 보였습니다.



### Crafting Interpretable Embeddings by Asking LLMs Questions (https://arxiv.org/abs/2405.16714)
- **What's New**: 박사님들, 이번 아카이브 논문에서는 대형 언어 모델(LLM) 기반 텍스트 임베딩의 해석 가능성을 높이기 위한 새로운 접근법을 소개합니다. 이 연구는 질문과 답변 임베딩(QA-Emb)을 도입하여, 각 특징이 예/아니오 질문에 대한 응답을 나타내도록 합니다. 이를 통해 LLM을 이용한 뇌 신경 과학 연구에서 fMRI 보셀 반응을 예측하는 모델을 생성할 수 있습니다.

- **Technical Details**: QA-Emb는 사전 훈련된 LLM에 일련의 질문을 던져 각 질문에 대한 답을 임베딩의 요소로 사용하는 방식입니다. 이 접근법은 LLM 내부를 조작하지 않으며 자연어 프롬프트만 수정합니다. 예를 들어 '입력에 시간이 언급되었나요?'라는 질문에 대한 yes/no 응답을 1/0으로 매핑합니다. 이를 통해 여러분이 텍스트 임베딩을 설계할 때 좀 더 구체적이고 연관된 요소를 포착할 수 있습니다.

- **Performance Highlights**: QA-Emb는 기존의 해석 가능한 기준을 26% 초과하며, 불투명한(BERT) 기준모델 보다도 성능이 조금 더 뛰어납니다. 또한, 985개의 특징을 가진 기존 기준을 29개의 질문만으로 모든 것을 능가하였습니다. 요컨대, QA-Emb는 높은 정확도를 유지하면서도 해석 가능한 간결한 임베딩을 만들어냅니다.



### Accurate and Nuanced Open-QA Evaluation Through Textual Entailmen (https://arxiv.org/abs/2405.16702)
Comments:
          To appear at ACL 2024 (Findings)

- **What's New**: 오픈 도메인 질문 답변(Open-QA) 시스템 평가의 새 접근법을 제안합니다. 전통적인 평가 방법이 질문의 모호함과 의미론적 이해 부족으로 인해 비판받아왔는데, 우리는 답변의 전제 관계를 연구하여 보다 정보적이고 일반적인 시스템 답변을 식별하고 인간 판단에 더 가까운 평가를 제공합니다.

- **Technical Details**: 제안된 방법은 NaturalQuestions 및 TriviaQA 데이터셋에서 시스템 답변과 골드 표준 답변 간의 의미론적 관계를 분석하여, 더 세밀하고 공정하게 답변의 정확성을 평가합니다. 이 방법은 '텍스트 전제(textual entailment)'를 기반으로, 추가 점수 또는 부분 점수를 할당하는 방식으로 평가의 정밀도를 높이며, 기존의 0 또는 1로 평가하는 이진(prediction) 방식을 개선합니다.

- **Performance Highlights**: 전제 기반 평가 메트릭은 별도의 학습 없이도 인간 판단과 일관되며, 여러 Open-QA 시스템의 실제 능력을 더 효과적으로 포착할 수 있습니다. 특히, 제안된 평가 메트릭은 현재 방법들보다 더 높은 AUC(Area Under Curve)를 나타내며, 시스템 답변의 세밀한 순위를 매길 수 있는 장점을 가지고 있습니다.



### gzip Predicts Data-dependent Scaling Laws (https://arxiv.org/abs/2405.16684)
Comments:
          9 pages, 9 figures

- **What's New**: 최신 연구에서는 신경망 언어 모델(LM)의 성능을 예측하는 스케일링 법칙(scaling laws)이 훈련 데이터의 복잡성에 민감하다는 것을 발견했습니다. 이를 통해 gzip이라는 압축 알고리즘이 데이터 복잡성의 영향을 예측하는 데 효과적임을 보였습니다. 이 연구는 gzip 압축 가능성(gzip-compressibility)을 고려한 새로운 데이터 의존 스케일링 법칙을 제안하고 있습니다.

- **Technical Details**: 연구팀은 PCFG(Probabilistic Context-Free Grammar)을 사용하여 다양한 복잡성을 가진 6개의 훈련 데이터셋을 생성했습니다. 각 데이터셋에 대해 6가지 크기의 언어 모델(4.4M에서 1.4B 파라미터)을 훈련하고, 6가지 다른 트레인 스텝(100K에서 100M 토큰)에서 결과를 기록했습니다. 이후 각 데이터셋에 대한 스케일링 법칙을 맞추면서 데이터의 복잡성이 증가함에 따라 법칙의 매개변수에 의미 있는 변화가 나타났습니다.

- **Performance Highlights**: 실험 결과, 훈련 데이터가 덜 압축 가능할수록(더 복잡할수록) 스케일링 법칙의 계산 최적 전선(compute-optimal frontier)이 데이터셋 크기에 대한 선호도를 더 많이 증가시켰습니다. 실제 코드 및 자연어 데이터셋의 압축 가능성을 측정한 결과, 코드가 더 압축 가능하다는 것을 확인하고, 이를 통해 다르게 적용되는 스케일링 법칙을 확인할 수 있었습니다. 이를 통해 StarCoder 성능을 달성하기 위해 24% 적은 FLOPs로도 가능하다는 추정이 나왔습니다.



### Triple Preference Optimization: Achieving Better Alignment with Less Data in a Single Step Optimization (https://arxiv.org/abs/2405.16681)
- **What's New**: 이번 논문에서는 Triple Preference Optimization (TPO)라는 새로운 방법을 소개합니다. TPO는 Supervised Fine-Tuning (SFT) 단계를 생략하면서도 훨씬 적은 데이터로 LLM(큰 언어 모델)을 세 가지 선호도에 맞게 조정할 수 있는 방법입니다. 이를 통해 데이터와 리소스를 절약하면서도 높은 성능을 유지할 수 있습니다.

- **Technical Details**: TPO는 기존의 두 가지 최적화 단계를 하나로 합쳐, 입력 프롬프트(input prompt), 기준 응답(gold standard response), 선호 응답(preferred response), 그리고 덜 선호되는 응답(less-preferred response)을 포함하는 통합 데이터 형식을 사용합니다. 이러한 방식으로 하나의 정책 모델을 최적화하여 학습합니다. 또한, 파레토 프론트(Pareto Front) 개념을 기반으로 한 단일 스텝 alignment 전략을 제안합니다.

- **Performance Highlights**: Phi-2(2.7B)와 Mistral(7B) 모델에 직접 TPO를 적용하여 UltraFeedback 데이터셋에서 학습한 결과, SFT, DPO, KTO, IPO, CPO, ORPO 등 기존 방법들보다 뛰어난 성능을 보였습니다. 특히, MT-Bench 점수가 SFT 대비 +1.27, DPO 대비 +0.63 상승하였습니다. 또한, Open LLM Leaderboard 벤치마크에서도 평균 정확도가 DPO와 SFT를 각각 4.2%, 4.97% 초과했습니다.



### RLSF: Reinforcement Learning via Symbolic Feedback (https://arxiv.org/abs/2405.16661)
- **What's New**: 최근 몇 년간, 대형 언어 모델(LLMs)은 AI의 다양한 하위 분야에 큰 영향을 미쳤습니다. 그러나 논리적 추론 역량은 여전히 미약합니다. 기존의 LLM 미세 조정 접근법(예: 인간 피드백 사용)은 이 문제를 다소 해결하지만, 많은 문제를 안고 있습니다. 이를 해결하기 위해 저자들은 강화 학습 기반의 상징적 피드백(RLSF)이라는 새로운 훈련/미세 조정 패러다임을 제안했습니다.

- **Technical Details**: RLSF 설정에서는, 훈련 중인 LLM이 RL 에이전트로 간주되고, 환경에는 추론 또는 도메인 지식 도구(예: 소버, 대수 시스템)가 접근할 수 있습니다. 중요한 점은 RLSF에서 이러한 추론 도구가 LLM에게 폴리사이즈 증명서(예: 증명)를 통해 피드백을 제공할 수 있다는 것입니다. 이는 기존 보상 모델의 제한을 해결하고, LLM에게 세밀하고 정확한 보상 신호를 제공할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 평가를 통해, LLM의 RLSF 기반 미세 조정이 두 가지 다른 응용에서 전통적인 접근법보다 뛰어나다는 것을 보여주었습니다. 예를 들어, 자연어 의사 코드에서 프로그래밍 언어(C++)로의 프로그램 합성 및 Game of 24 해답. RLSF 조정을 통해, LLM은 컴파일 정확도와 기능적 정확도에서 큰 향상을 보였습니다. RLSF 조정된 모델은 기존 챗봇 모델들보다 훨씬 더 우수한 성능을 발휘했습니다.



### Compressing Lengthy Context With UltraGis (https://arxiv.org/abs/2405.16635)
- **What's New**: 새로운 방법인 UltraGist를 소개합니다. UltraGist는 혁신적인 압축 및 학습 알고리즘을 통해 긴 컨텍스트를 고품질로 압축하는 데 특화된 방법입니다. 이 방법은 문서 QA(document QA), 문서 요약(summarization), 몇 개의 샘플을 사용한 학습(few-shot learning), 다중 세션 대화(multi-session conversation) 등 다양한 작업에서 near-lossless 압축 성능을 유지합니다. 또한, 데이터, 모델 및 코드는 공개되어 연구에 기여할 예정입니다.

- **Technical Details**: UltraGist는 길고 복잡한 컨텍스트를 미세 세그먼트로 분할한 후, 이를 맞춤형 크로스 어텐션 메커니즘을 통해 점진적으로 압축합니다. 이 과정에서 각 세그먼트는 랜덤으로 샘플된 압축 비율로 처리됩니다. 이 방법은 학습 과정의 유연성을 높이고, 고품질의 미세한 압축을 가능하게 하며, 트레이닝 손실을 모든 토큰에서 얻을 수 있게 하여 샘플 효율성을 극대화합니다. 그리고 새로운 문맥이 추가될 때마다 점진적으로 업데이트할 수 있습니다.

- **Performance Highlights**: UltraGist는 기존의 방법들이 어려움을 겪는 긴 컨텍스트를 갖는 시나리오에서도 near-lossless 압축 성능을 유지합니다. 이는 문서 QA, 문서 요약, 몇 개의 샘플을 사용한 학습, 다중 세션 대화 등 다양한 작업들을 포함합니다.



### Let Silence Speak: Enhancing Fake News Detection with Generated Comments from Large Language Models (https://arxiv.org/abs/2405.16631)
Comments:
          11 pages, 5 figures, 8 tables

- **What's New**: 소셜 미디어 사용자 보호와 건강한 뉴스 생태계를 유지하기 위해 가짜 뉴스 감지가 중요합니다. 'GenFEND'라는 새로운 프레임워크를 제안하여 대규모 언어 모델(LLMs)을 사용자 시뮬레이터와 댓글 생성기로 사용합니다. 이는 다양한 사용자 프로필로 LLM을 프롬프트하여 다양한 댓글을 생성하고, 이를 통합하여 가짜 뉴스 감지 성능을 향상시킵니다.

- **Technical Details**: GenFEND는 성별, 연령, 교육 수준과 같은 속성을 결합하여 미리 정의된 다양한 사용자 프로필을 가지고 LLM을 프롬프트하여 역할놀이를 통해 포괄적인 사용자 피드백을 생성합니다. 생성된 댓글의 의미론적 특성을 추출하고, 각 인구 통계적 관점에서 여러 하위 집단 그룹으로 분할한 후, 평균화 작업을 통해 전반적인 피드백을 얻습니다. 최종 표현은 내관점 및 상관점 집계를 통해 얻습니다.

- **Performance Highlights**: 실험 결과, GenFEND는 실제 사용자 댓글이 있는 경우보다 가짜 뉴스 탐지 성능을 향상시키고 LLM 생성 댓글이 더 효과적일 수 있음을 보여주었습니다. 다양한 사용자 그룹에서 생성된 댓글이 더 포괄적인 피드백을 제공하며, 이는 정확한 뉴스 진위를 판단하는 데 도움이 됩니다.



### MentalManip: A Dataset For Fine-grained Analysis of Mental Manipulation in Conversations (https://arxiv.org/abs/2405.16584)
Comments:
          Accepted at ACL 2024

- **What's New**: 이번 연구는 정신적 조작(mental manipulation) 탐지를 위한 새로운 데이터셋인 **MentalManip**을 소개합니다. 이 데이터셋은 4,000개의 주석이 달린 영화 대화를 포함하고 있으며, 정신적 조작 기술과 피해자의 취약점을 분석할 수 있도록 합니다. 이를 통해 정신적 조작의 탐지와 분류 능력을 향상시키고자 합니다.

- **Technical Details**: MentalManip 데이터셋은 세 가지 차원에서 주석이 달렸습니다. 첫 번째는 조작 존재 여부를 나타내는 **Presence of Manipulation**, 두 번째는 사용된 조작 기술을 식별하는 **Manipulation Technique**, 세 번째는 표적으로 삼은 취약점을 나타내는 **Targeted Vulnerability**입니다. 이 데이터셋은 Cornell Movie Dialogs Corpus에서 추출되었으며, 세부적으로 11가지 조작 기술과 5가지 취약점을 포함합니다.

- **Performance Highlights**: 최신 Large Language Models (LLMs)를 다양한 설정으로 실험한 결과, 정신적 조작 대화를 잘 인식하고 분류하지 못한다는 것을 발견했습니다. 기존의 정신 건강 및 유해성 관련 데이터셋으로 미세 조정(fine-tuning)을 시도했지만 성능 향상에 큰 도움이 되지 않았습니다. 이는 MentalManip 데이터셋의 중요성을 강조하며, 미래 연구가 정신적 조작 탐지와 분석에 있어서 더 나아갈 수 있는 기회를 제공할 것입니다.



### Automatically Generating Numerous Context-Driven SFT Data for LLMs across Diverse Granularity (https://arxiv.org/abs/2405.16579)
- **What's New**: 이 논문에서는 AugCon이라는 새로운 방법을 소개했습니다. 이는 복수의 세밀도 수준에서 컨텍스트 기반의 높은 다양성과 품질, 신뢰도를 갖춘 감독된 미세 조정(SFT) 데이터를 자동으로 생성할 수 있습니다. 이러한 데이터는 도메인 특화 AI 어시스턴트나 롤플레이 에이전트를 만드는 데 매우 중요합니다.

- **Technical Details**: AugCon 방법론의 핵심 구성 요소는 컨텍스트 분할 트리(Context-Split-Tree, CST)입니다. 이는 재귀적으로 쿼리를 생성하고 컨텍스트를 분할하여 완전한 세밀도를 포괄합니다. 또한 대조 학습(Contrastive Learning)으로 학습된 점수가 CST와 협업하여 쿼리를 순위화하고 정제합니다. 마지막으로, 자가 정렬(Self-alignment)과 자가 개선(Self-improving)을 시너지 있게 통합하여 신뢰도 높은 응답을 생성합니다.

- **Performance Highlights**: 포괄적인 실험과 평가 시나리오, 영어와 중국어로 된 네 개의 널리 사용되는 벤치마크를 포함한 실험에서 AugCon의 탁월한 성능이 입증되었습니다. 이 방법은 기존 최첨단 방법들에 비해 더 높은 다양성, 품질, 신뢰도의 SFT 데이터를 생성하는 데 현저한 장점을 보여줍니다. 모든 코드, 데이터 세트, 미세 조정된 모델은 공개될 예정입니다.



### A Preliminary Empirical Study on Prompt-based Unsupervised Keyphrase Extraction (https://arxiv.org/abs/2405.16571)
Comments:
          work in progress

- **What's New**: 이 연구는 프롬프트 기법(prompt-based approach)을 사용하는 대형 언어 모델(LLM)에서 키프레이즈 추출(keyphrase extraction) 작업에 대한 다양한 프롬프트의 효과를 조사합니다. 복잡한 프롬프트가 단순한 프롬프트보다 반드시 더 효과적이지 않다는 점과, 개별 키워드의 변화가 전체 성능에 영향을 미칠 수 있다는 점을 발견하였습니다. 특히, 긴 문서의 경우 복잡한 프롬프트가 더 나은 성능을 보였습니다.

- **Technical Details**: 이 연구는 대형 사전 학습 언어 모델(encoder-decoder architecture)을 기반으로 한 키프레이즈 추출 모델에서 프롬프트의 역할을 분석합니다. T5 모델(Raffel et al., 2020)을 사용하여, 원본 문서에서 키프레이즈 후보들을 추출하고 인코더에 문서를 입력한 후 디코더에서 디자이너된 프롬프트로 후보의 생성 확률을 계산합니다. 이 확률을 통해 키프레이즈의 중요도를 평가합니다.

- **Performance Highlights**: 여섯 개의 벤치마크 키프레이즈 추출 데이터셋(Inspec, DUC2001, SemEval2010, SemEval2017, Nus, Krapivin)에서 실험을 실시한 결과, 복잡한 프롬프트 디자인이 긴 문서에서 단순한 프롬프트보다 성능이 좋다는 결론을 얻었습니다. 또한 T5와 Flan-T5 모델을 사용했으며, 실험에서 F1 점수를 중심으로 성능을 평가했습니다.



### SED: Self-Evaluation Decoding Enhances Large Language Models for Better Generation (https://arxiv.org/abs/2405.16552)
Comments:
          The relevant code will be released in subsequent versions

- **What's New**: 이 논문은 Self-Evaluation Decoding (SED)이라는 새로운 디코딩 방법을 제안하여 대형 언어 모델(LLM)의 텍스트 생성 품질을 향상시키고자 합니다. 특히 불확실한 토큰을 선택하는 문제를 해결하는 데 중점을 둡니다. 기존의 단방향 자회귀 디코딩 방식이 쉽게 서브옵티멀한 결과에 빠지는 반면, SED는 인간의 의사결정 과정을 모방하여 추측 및 평가 단계를 통해 더 신중하게 토큰을 선택할 수 있도록 합니다.

- **Technical Details**: SED는 세 가지 주요 모듈로 구성되어 있습니다: 혼란점 감지(chaotic point detection), 추측 및 평가(speculation and evaluation), 피드백 역전파(feedback backpropagation)입니다. 이는 LLM이 혼란점에서 각 토큰을 선택한 결과를 추측하고 이를 평가한 후 가장 높은 평가 점수를 가진 토큰을 선택하도록 합니다. 이를 통해 기존의 검색 기반 방법(탐욕적 검색, 빔 검색)과 샘플링 기반 방법(핵 샘플링)의 한계를 극복합니다.

- **Performance Highlights**: 다양한 작업과 다른 LLM을 사용한 실험 결과 SED의 효과가 입증되었습니다. 기존의 디코딩 방법과 비교하여, SED는 생성된 텍스트의 품질을 크게 향상시켰습니다. 특히 불확실한 토큰 선택에서 발생하는 오류를 줄이고, 전체적인 텍스트 생성의 일관성과 정확성을 높였습니다.



### Chain of Tools: Large Language Model is an Automatic Multi-tool Learner (https://arxiv.org/abs/2405.16533)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 다양한 외부 도구와 결합시켜 실질적인 작업을 자동으로 해결할 수 있는 'Automatic Tool Chain (ATC)' 프레임워크를 제안합니다. 또한, LLM이 새로운 도구 사용법을 스스로 학습할 수 있도록 하는 'black-box probing method'도 도입됩니다. 이를 통해 LLM이 다양한 도구를 활용하고 학습할 수 있는 능력을 강화했습니다.

- **Technical Details**: ATC 프레임워크는 LLM이 도구 체인을 사용하여 프로그램을 생성하고 실행할 수 있도록 합니다. 입력-출력 스키마와 데이터 흐름 의존성을 학습하여 도구 프로토콜을 기반으로 프로그램을 작성하며, 실행 중 발생하는 오류를 추적하여 수정하는 'attributable reflection mechanism'을 포함합니다. Black-box probing 방법은 새로운 도구의 입력-출력 스키마를 조사하여 LLM이 스스로 도구 사용법을 문서화하고 학습할 수 있게 해줍니다. 이 방법은 다양한 실제 도구에 대한 포괄적인 평가를 위해 'ToolFlow'라는 새로운 벤치마크를 포함합니다.

- **Performance Highlights**: 도구 학습 관련 3개의 데이터셋에서 실험한 결과, 제안된 방법은 기존의 방법들에 비해 높은 효율성을 보였습니다. 또한, LLM은 도구 프로토콜을 잘 이해하고 체계적으로 도구 체인을 계획하는 능력을 보여주었습니다. 특히 black-box probing 방법은 LLM이 도구 프로토콜을 탐구하고 새로운 도구를 마스터하는 데 효과적임을 입증했습니다.



### DarijaBanking: A New Resource for Overcoming Language Barriers in Banking Intent Detection for Moroccan Arabic Speakers (https://arxiv.org/abs/2405.16482)
- **What's New**: 이번 논문은 모로코 방언인 '다리자(Darija)'를 위한 새로운 데이터셋 'DarijaBanking'을 소개합니다. 이 데이터셋은 금융 도메인에서의 의도 분류를 향상시키기 위해 고안된 것으로, Darija, 표준 아랍어(Modren Standard Arabic, MSA), 영어, 프랑스어 등 4개 언어로 1,800개 이상의 고품질 질의를 포함하여 총 7,200개 질의로 구성되어 있습니다.

- **Technical Details**: DarijaBanking 데이터셋은 기존 영어 금융 데이터셋에서 번역과 정제를 거쳐 완성되었습니다. 세부적으로는 Banking77, Banking-Faq-Bot, Smart-Banking-Chatbot 데이터셋을 기반으로 하여 OPUS MT 모델과 Turjuman 모델을 통해 각각 프랑스어와 표준 아랍어로 번역되었습니다. 마지막으로, GPT-4를 활용하여 다리자로 번역되었습니다. 추가적으로, 5명의 원어민으로 구성된 외부 인력이 번역의 정확성과 맥락적 적절성을 검증 및 수정하였습니다.

- **Performance Highlights**: 다양한 의도 분류 방법을 통해 성능을 평가하였으며, 여기에는 모노링구얼(monolingual) 및 멀티링구얼(multilingual) 모델의 완전 미세 조정(full-fine tuning), 제로 샷 학습(zero-shot learning), 검색 기반 접근법, 대형 언어 모델(LLM) 프롬팅(prompting) 등이 포함되었습니다. 이 논문의 주된 기여 중 하나인 'BERTouch'는 다리자에 특화된 BERT 기반 언어 모델로, DarijaBanking 데이터셋에서 다리자에 대해 0.98, 표준 아랍어에 대해 0.96의 F1 점수를 기록하여 GPT-4를 포함한 최신 모델을 뛰어넘는 성능을 보였습니다.

- **Insights**: 논문은 다리자 의도 분류 시스템을 개발하는 데 있어 일반적인 대형 언어 모델과 교차 언어 전이 학습(cross-lingual transfer learning)을 사용하는 것과 도메인별 데이터 주석의 중요성 사이의 균형을 강조합니다. 또한, 충분한 데이터 라벨링을 수행한 전문화된 분류기와 예산에 맞는 검색 기반 접근법의 효율성을 제시하여 정확하고 경제적인 시스템 개발에 대한 가이드라인을 제시합니다.



### CPsyCoun: A Report-based Multi-turn Dialogue Reconstruction and Evaluation Framework for Chinese Psychological Counseling (https://arxiv.org/abs/2405.16433)
Comments:
          Appectped to Findings of ACL2024

- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)을 이용한 심리 상담을 돕기 위한 CPsyCoun 프레임워크를 제안했습니다. 이 프레임워크는 보고서를 기반으로 다중 회전 대화를 재구성 및 평가하는 시스템으로, 특히 중국어 심리 상담에 중점을 두고 있습니다. 이를 통해 고품질 대화를 생성하고 다중 회전 심리 상담의 효과적인 자동 평가를 위한 벤치마크를 제공합니다.

- **Technical Details**: CPsyCoun은 심리 상담 보고서를 활용하여 대화를 재구성하는 두 단계 접근법을 사용합니다. 먼저, 공공 웹사이트에서 익명화된 심리 상담 보고서를 수집하고, 개인정보 보호를 위해 후처리합니다. 이후, Memo2Demo라는 방법을 통해 3,134개의 고품질 다중 회전 상담 대화를 포함하는 데이터셋 CPsyCounD를 구성합니다. 또한, 다중 회전 대화의 자동 평가를 위한 심리 상담 벤치마크를 제안하고, CPsyCounD 데이터셋에서 미세 조정된 모델 CPsyCounX도 개발합니다.

- **Performance Highlights**: 실험 결과, CPsyCounX 모델은 내재적 및 외재적 평가에서 일관되게 우수한 성능을 보였습니다. 이 모델은 기존 심리 상담 모델을 능가하며, 심리 상담에서의 대화 재구성과 자동 평가 메트릭스를 통해 더 나은 상담 서비스를 제공할 수 있음을 입증했습니다.



### AI-Generated Text Detection and Classification Based on BERT Deep Learning Algorithm (https://arxiv.org/abs/2405.16422)
- **What's New**: AI 생성 텍스트 검출(AI-generated text detection)이 다양한 분야에서 점점 더 중요한 역할을 하고 있습니다. 본 연구는 BERT 알고리즘 기반으로 효율적인 AI 생성 텍스트 검출 모델을 개발하여 관련 문제 해결을 위한 새로운 아이디어와 방법을 제시합니다.

- **Technical Details**: 데이터 전처리 단계에서는 데이터 품질과 정확성을 보장하기 위해 텍스트를 소문자로 변환하기, 단어 분할(word splitting), 불용어 제거(removing stop words), 어간 추출(stemming extraction), 숫자 제거(removing digits), 불필요한 공백 제거 등의 일련의 단계를 거쳤습니다. 데이터 셋은 60%의 학습용(training set)과 40%의 시험용(test set)으로 나누었으며, 학습 과정에서의 정확도와 손실 값의 변화를 관찰하였습니다.

- **Performance Highlights**: 모델은 학습 과정에서 잘 작동했으며, 초기 정확도 94.78%에서 99.72%로 지속적으로 증가하였고, 손실 값은 0.261에서 0.021로 감소하며 점차 수렴하였습니다. 이는 BERT 모델이 AI 생성 텍스트를 높은 정확도로 검출할 수 있음을 나타냅니다. 학습 세트(training set)의 평균 정확도는 98.1%로 나타났으며 시험 세트(test set)의 평균 정확도는 97.71%로, 두 값의 차이가 크지 않아 모델의 일반화 능력이 우수함을 보여줍니다.



### M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions (https://arxiv.org/abs/2405.16420)
Comments:
          This paper has been accepted by ACL 2024

- **What's New**: 새롭게 제안된 M-RAG(Multiple partitions based Retrieval-Augmented Generation)은 기존 RAG(Retrieval-Augmented Generation)의 단점인 불필요한 노이즈와 중요한 메모리의 집중 부족 문제를 해결하고자 한다. M-RAG는 데이터베이스를 여러 파티션으로 나누어 각 파티션을 독립적인 RAG 실행 기본 단위로 사용하는 방식을 도입하였다. 이를 통해, 다양한 언어 생성 작업에 걸쳐 LLM(Large Language Model)과 Multi-Agent Reinforcement Learning을 활용하여 성능을 최적화하는 새로운 프레임워크를 제안하였다.

- **Technical Details**: 이 연구에서는 M-RAG를 통해 다음 세 가지 주요 과제를 해결한다: (1) 데이터베이스 파티셔닝 전략 및 파티션 수 결정, (2) 적절한 메모리를 찾기 위한 파티션 선택 방법 개발, (3) 메모리 품질 개선. Multi-Agent Reinforcement Learning을 기반으로, 에이전트 S와 에이전트 R을 각각 파티션 선택 및 메모리 평가/강화를 위해 도입하였다. 에이전트 S는 Multi-Armed Bandit 문제를 풀기 위해 사용되며, 에이전트 R은 여러 후보 메모리를 생성하고 가장 품질이 높은 메모리를 사용하도록 최적화된다.

- **Performance Highlights**: M-RAG는 실험 결과, 텍스트 요약, 기계 번역, 대화 생성 작업에서 각각 11%, 8%, 12%의 성능 향상을 보였다. 이는 M-RAG가 기존의 다양한 베이스라인 방법보다 일관되게 우수한 성능을 발휘했음을 의미한다. 다양한 생성 작업에서의 실험을 통해 M-RAG의 효과를 입증하였다.



### KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledg (https://arxiv.org/abs/2405.16412)
- **What's New**: 지식 그래프 임베딩(KGE) 기술은 엔티티와 관계의 압축 표현을 학습하여 효율적인 추론 및 지식 발견을 가능하게 합니다. KG-FIT은 LLM(대형 언어 모델) 유도 세분화를 통해 엔티티 클러스터의 의미적으로 일관된 계층 구조를 구축합니다. 이 계층 지식과 텍스트 정보를 미세 조정(fine-tuning) 과정에 통합함으로써, KG-FIT은 LLM의 전반적인 의미(global semantics)와 KG의 지역적 의미(local semantics)를 효과적으로 포착합니다.

- **Technical Details**: KG-FIT은 두 단계 접근법을 사용합니다. 첫 번째로, LLM을 사용해 엔티티 설명을 생성하고 의미적으로 일관된 계층 구조를 구축합니다. 두 번째로, 이 계층 구조와 텍스트 임베딩을 통합하여 KG 임베딩을 미세 조정합니다. 이를 위해 결합 그리기(agglomerative clustering)와 LLM-유도 기법을 사용하여 초기 계층 구조를 설정한 후 이를 세밀하게 조정합니다.

- **Performance Highlights**: FB15K-237, YAGO3-10 및 PrimeKG 벤치마크 데이터셋에서 KG-FIT은 링크 예측 작업의 Hits@10 메트릭에서 기존 최신 방법보다 각각 14.4%, 13.5% 및 11.9% 향상된 성능을 보였습니다. 또한, KG-FIT은 구조 기반 기본 모델에 비해 각각 12.6%, 6.7%, 17.7%의 성능 향상을 보여주었습니다. 이는 LLM에서 얻은 외부 지식을 KG 임베딩에 효과적으로 통합함으로써 그 표현력과 정보 전달 능력을 크게 향상시켰음을 강조합니다.



### Assessing Empathy in Large Language Models with Real-World Physician-Patient Interactions (https://arxiv.org/abs/2405.16402)
- **What's New**: 최근 연구에서 Mayo Clinic의 환자 메시지와 의사 응답 데이터를 수집하고, ChatGPT가 생성한 대안 응답을 비교하여 LLM 기반 챗봇의 공감력 평가를 수행했습니다. 연구 결과, LLM 챗봇이 의사보다 더 높은 공감 수준을 제공할 가능성이 있음을 발견했습니다. 이는 환자 진료를 향상시키고 전문직 소진을 줄이는 데 기여할 수 있는 중요한 발견입니다.

- **Technical Details**: 이 연구는 LLaMA 모델을 사용하여 공감 평가 메트릭 EMRank를 개발하고, 이는 자동화된 메트릭과 인간 평가를 포함합니다. 분석 대상에 대해 zero-shot, one-shot, few-shot 학습 시나리오를 적용하였으며, 모델 평가 시 독립성을 유지하기 위해 ChatGPT와 LLaMA를 각각 응답 생성 모델과 공감 평가 모델로 사용했습니다.

- **Performance Highlights**: 실험 결과, LLaMA-EMRank 메트릭은 인간 평가와 높은 일치를 보여 공감도 평가의 신뢰성을 입증했습니다. ChatGPT는 실제 의사보다 높은 공감도를 가진 응답을 생성할 수 있다는 결과가 도출되었으며, 이는 LLM 기술이 환자 케어에 기여할 수 있는 중요한 잠재력을 나타냅니다.



### Multi-Reference Preference Optimization for Large Language Models (https://arxiv.org/abs/2405.16388)
Comments:
          20 pages

- **What's New**: 이 논문에서는 대규모 언어 모델 (Large Language Models, LLMs)을 인간의 의도와 가치에 맞추기 위한 새로운 접근법인 '다중참조 선호 최적화' (Multi-Reference Preference Optimization, MRPO)를 제안합니다. 기존의 단일 참조 모델을 사용하는 방식 대신 여러 참조 모델을 활용해 선호 학습 능력을 향상시키는 기법입니다.

- **Technical Details**: MRPO는 다중 참조 모델을 사용해 패널티로 KL 발산 (Kullback-Leibler divergence)를 도입하여 최적화를 수행합니다. 이를 위해 비선형 KL 용어들을 간소화한 대수 표현을 최대화하는 방법을 사용합니다. 또한, 로그 확률을 '클리핑' 하여 참조 방침의 불일치 문제를 최소화하고, 각 참조 모델의 기여도를 동적으로 계산하는 자동화된 메커니즘인 ARWC를 도입합니다.

- **Performance Highlights**: MRPO는 다양한 선호 데이터셋에서 단일 참조 모델 기반 방법인 DPO보다 최대 7% 더 뛰어난 성능을 보였습니다. 또한, 여러 자연어 처리 작업(GSM8K, TruthfulQA 등)에서도 상대적으로 높은 퍼포먼스를 보여주며, HuggingFace Open LLM Leaderboard에서 평균 3-4% 향상된 점수를 기록했습니다. 참조 모델의 수와 LLM의 크기에 상관없이 일관된 성능 향상을 보였습니다.



### STRIDE: A Tool-Assisted LLM Agent Framework for Strategic and Interactive Decision-Making (https://arxiv.org/abs/2405.16376)
Comments:
          39 pages, 4 figures

- **What's New**: 새로운 논문에서는 대형 언어 모델(LLMs)인 GPT-4와 같은 모델들이 가진 전략적 다중 에이전트(agents) 결정 메이킹 환경에서의 한계를 극복하기 위해 메모리와 특수 도구를 장착한 새로운 LLM 에이전트 프레임워크를 제안합니다. 특히 양자 협상(bilateral bargaining)과 다중 에이전트 및 동적 메커니즘 설계(dynamic mechanism design)와 같은 경제적으로 중요한 환경에서 이 도구들을 배치합니다.

- **Technical Details**: 제안된 프레임워크는 LLM의 전략적 결정 메이킹 능력을 향상시키기 위해 메모리와 특수 도구를 통합합니다. 이는 게임 규칙, 장기 계획, 미지의 환경에서의 탐색, 그리고 상대방의 움직임을 예측하는 등의 세밀한 작업을 보다 잘 수행할 수 있게 합니다. 다양한 전략적 문제 해결에 대한 성능을 양적 지표로 평가하는 방법을 사용합니다.

- **Performance Highlights**: 결과적으로, 이 프레임워크는 LLM의 전략적 결정 메이킹 능력을 상당히 향상시키는 것을 입증했습니다. 기존 LLM 모델들이 가지고 있는 본질적인 한계에도 불구하고, 타겟 강화(targeted enhancements)를 통해 상당한 개선이 이루어졌음을 보여줍니다. 이는 인터랙티브 환경에서의 LLM 응용 분야에서 유망한 발전 방향을 제시합니다.



### Learning to Reason via Program Generation, Emulation, and Search (https://arxiv.org/abs/2405.16337)
Comments:
          16 pages, 10 figures

- **What's New**: 최근 다양한 알고리즘적 상징 조작 작업(예: 단어 연결)을 해결할 수 있는 코드 생성(Language Models)을 사용하는 연구가 증가하고 있습니다. 그러나 상식 추론, 도덕적 결정, 그리고 풍자 이해와 같은 부드러운 추론 작업을 코드로 표현하는 것은 쉽지 않습니다. 본 연구는 언어 모델(LM)의 프로그램 생성 능력을 이러한 부드러운 추론 작업으로 확장하고, Python 프로그램의 잎 함수 호출을 정의하지 않은 상태로 평가하는 것을 목표로 합니다. 이를 위해, 우리는 CoGEX(Code Generation and Emulated EXecution)라는 새로운 접근법을 제안합니다.

- **Technical Details**: CoGEX는 (1) LMs가 자체 유사 프로그램(pseudo-programs)을 생성하도록 훈련하고, (2) 생성된 프로그램의 실행을 에뮬레이트하도록 가르치며, (3) 최적의 프로그램을 찾기 위해 여러 프로그램을 검색하는 방식으로 작동합니다. 이 모델을 새로운 과제에 적응시키기 위해, 주어진 데이터세트의 모든 인스턴스에 적용될 때 최적의 성능을 내는 단일 프로그램을 찾고자 합니다. 이를 위해 우리는 CoTACS(CoGEX Task Adaptation via Code Search)라는 검색 절차를 도입하였습니다. CoTACS는 파라미터 업데이트 없이 많은 프로그램을 시도해보며 최적의 프로그램을 찾는 방법입니다.

- **Performance Highlights**: 다양한 추론 작업(commonsense QA, 텍스트 분류, 수학 데이터세트)을 평가한 결과, CoTACS를 적용하면 CoGEX 모델이 동일한 초기 체크포인트와 동일한 학습 예제를 사용한 표준 in-context 학습 접근법에 비해 상당히 더 나은 성능을 발휘했음을 보였습니다. 이는 코드 생성을 이용하여 훨씬 더 광범위한 문제를 해결할 수 있음을 시사합니다.



### Comparative Analysis of Open-Source Language Models in Summarizing Medical Text Data (https://arxiv.org/abs/2405.16295)
- **What's New**: 이번 연구에서는 Llama2와 Mistral 같은 오픈소스 LLM(Large Language Models)을 의료 요약 작업에 사용하여 그 성능을 평가하는 새로운 접근 방식을 제안합니다. GPT-4를 평가자로 사용해 각 LLM의 요약 성능을 비교하고, 이를 통해 품질 관리를 강화하며 최적의 LLM을 선택할 수 있도록 돕습니다.

- **Technical Details**: 연구의 주요 목표는 각 LLM이 바이오메디컬 요약 작업을 수행할 수 있는 파이프라인을 구축하고, 특정 요약 작업에 적합한 프롬프트를 선택하며, GPT-4를 평가자로 사용하는 평가 프레임워크를 설계 및 구현하는 것입니다. 본 연구에서는 Llama2-70B 및 Mistral-7B 모델, 그리고 MEDIQA-QS, MeQSum, MEDIQA-ANS, MEDIQA-MAS, iCliniq 같은 여러 데이터셋을 사용하여 요약 작업을 수행하였습니다.

- **Performance Highlights**: GPT-4를 평가자로 사용하여 각 LLM의 성능을 비교한 결과, 각 모델의 반응이 일관성, 유창성, 연관성을 기준으로 평가되었습니다. 이 평가 방법론은 특정 디지털 헬스 작업에 가장 적합한 LLM을 선택하는 데 중요한 기여를 합니다.



### Generating clickbait spoilers with an ensemble of large language models (https://arxiv.org/abs/2405.16284)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 활용한 클릭베이트(clickbait) 스포일러 생성을 위한 혁신적인 접근 방식을 제안합니다. 현재의 최첨단 방법들은 구절이나 패시지 형식의 스포일러 생성에 제한되어 있으나, 본 연구는 여러 비연속 구절을 포함하는 멀티파트 스포일러도 생성할 수 있는 새로운 앙상블 방법을 소개합니다.

- **Technical Details**: 제안된 접근 방식은 세 단계로 구성됩니다: 1) 클릭베이트 텍스트를 질문으로 변환, 2) 다양한 LLM에서 후보 스포일러 생성, 3) 후보 스포일러 중 최종 스포일러를 선택하는 학습된 평가 모델 적용. 클릭베이트를 질문으로 변환하기 위해 최근의 Vicuna 모델을 사용하였으며, 각 LLM 구성 요소는 LoRA라는 어댑터 기반 방식으로 미세 조정됩니다. 또한, 최종 스포일러 선택을 위해 포인트와 페어와이즈 학습 방법론을 이용한 랭킹 모델을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 앙상블 모델은 BLEU, METEOR, BERTScore 등의 평가 지표에서 기존 방법들보다 우수한 성능을 보였습니다. 특히 멀티파트 스포일러 생성을 효과적으로 수행함을 확인했습니다. 다양한 사전 학습된 모델(LLaMA, Vicuna)을 활용하여 높은 적합성을 달성하였습니다.



### Confidence Under the Hood: An Investigation into the Confidence-Probability Alignment in Large Language Models (https://arxiv.org/abs/2405.16282)
Comments:
          9 pages (excluding references), accepted to ACL 2024 Main Conference

- **What's New**: 최근 대규모 언어 모델(LLMs)의 사용이 더욱 확산됨에 따라, 모델이 생성한 응답에서 자신의 확신을 평가하는 것이 중요해졌습니다. 이 연구에서는 'Confidence-Probability Alignment'라는 새로운 개념을 소개하였습니다. 이는 모델의 내부 확신, 즉 토큰 확률(token probabilities)로 정량화된 것을, 모델이 자신의 확실성을 명시적으로 질문받았을 때의 확신과 연결하는 것입니다. 특히 OpenAI의 GPT-4는 다양한 작업에서 평균 스피어만 상관 계수(Spearman's $ho$)가 0.42로 가장 높은 정렬을 보여주었습니다.

- **Technical Details**: 이 연구에서는 내부 확신과 표현된 확신 사이의 정렬을 분석하기 위해 다양한 데이터셋과 모델 내성(introspection)을 유도하는 프롬프트 기법을 사용했습니다. 이를 위해 구조화된 평가 척도를 사용하여 확신을 평가하고, 답변 옵션을 포함한 프롬프트를 사용하였으며, 모델이 자발적으로 인식하지 못한 출력에 대한 확신 수준도 추출했습니다. 내부 확신은 출력 토큰에 할당된 확률로 정량화되며, 말로 표현된 확신은 모델의 응답에서 명시적으로 표현된 확신 수준을 나타냅니다.

- **Performance Highlights**: 주요 성과로는 OpenAI의 GPT-4가 다양한 작업에서 가장 높은 confidence-probability alignment를 보여주었다는 점입니다. 이는 모델 신뢰성을 향상시키고, 사용자가 모델의 결과물을 신뢰할 수 있도록 돕습니다. 또한, 모델의 확신 표현이 실제 응답 정확성과 어떻게 연결되는지를 분석하였으며, 온도(temperature)와 같은 모델 특성의 영향을 탐구하였습니다.



### ConStat: Performance-Based Contamination Detection in Large Language Models (https://arxiv.org/abs/2405.16281)
- **What's New**: 이번 연구에서는 데이터 오염(data contamination)이 대형 언어 모델(LLMs) 평가지에서 초래하는 영향을 새롭게 정의하고 이를 해결하기 위한 ConStat이라는 통계적 방법을 제안했습니다. 기존의 데이터 오염 탐지 방식이 한계에 봉착한 상황에서 본 연구는 인위적으로 부풀려진 비일반화 성능에 중점을 두어 오염을 정의하고, 이로 인해 모델 성능 측정의 신뢰성을 향상시키고자 합니다.

- **Technical Details**: 기존 탐지 방식이 주로 벤치마크 샘플들이 훈련 데이터에 포함되는 것을 중점으로 한 것과 달리, 본 연구의 새로운 접근법은 성능이 부풀려진 정도를 기반으로 합니다. ConStat는 대조 벤치마크(reference benchmark)와의 성능 비교를 통해 오염 여부를 감지합니다. 구체적으로, 비슷한 대조 벤치마크(Dref)와 원래 벤치마크(D)의 성능 평균을 계산하여 모델이 실제로 얼마나 성능이 부풀어졌는지 평가합니다. 이를 통해 말단 모델의 성능이 대조군 모델들에 비해 상대적으로 어느 정도로 높은지를 분석하고, 이는 오염 정도를 수치화하는 데 도움을 줍니다.

- **Performance Highlights**: ConStat는 기존 방법들보다 훨씬 더 효과적으로 오염을 탐지할 수 있습니다. 여러 모델 아키텍처와 벤치마크, 오염 시나리오를 광범위하게 평가한 결과, Mistral, Llama, Yi 및 상위 3개 Open LLM Leaderboard 모델에서 높은 오염 수준이 발견되었습니다. 예를 들어, 모델 M1은 대조 벤치마크에서 60% 성능을 보였으나 원래 벤치마크에서는 72%를 기록해, 이는 35%의 큰 오염 효과로 나타났습니다.



### Picturing Ambiguity: A Visual Twist on the Winograd Schema Challeng (https://arxiv.org/abs/2405.16277)
Comments:
          9 pages (excluding references), accepted to ACL 2024 Main Conference

- **What's New**: 최근 출시된 WinoVis 데이터셋은 텍스트와 이미지를 결합한 멀티모달 문맥에서 대명사 구별 능력을 평가하기 위해 설계되었습니다. 이는 텍스트-이미지 모델이 대명사 해소(pronoun disambiguation)에서 얼마나 뛰어난지를 테스트하는 새로운 데이터셋입니다. GPT-4를 사용한 프롬프트 생성과 Diffusion Attentive Attribution Maps (DAAM) 열맵 분석을 통해 평가 프레임워크가 개선되었습니다.

- **Technical Details**: WinoVis는 새로운 스테이블 디퓨전(Stable Diffusion) 모델 2.0을 포함한 다양한 모델을 평가합니다. 주요 기술 요소로는 GPT-4를 활용한 프롬프트 생성과 DAAM을 사용한 열맵 분석이 포함됩니다. 또한, 새로운 평가 프레임워크는 대명사 해소 외에도 다른 시각적 처리 과제를 구별할 수 있는 방법론을 도입했습니다.

- **Performance Highlights**: Stable Diffusion 2.0은 WinoVis 데이터셋에서 56.7%의 정밀도를 기록하며 무작위 추측을 약간 넘어섰습니다. 이는 현재의 모델들이 인간 수준의 성능에 도달하기에 여전히 부족함을 보여줍니다. 추가적인 오류 분석을 통해 텍스트-이미지 모델에서 해결해야 할 중요한 영역들을 식별했습니다.



### No Two Devils Alike: Unveiling Distinct Mechanisms of Fine-tuning Attacks (https://arxiv.org/abs/2405.16229)
Comments:
          work in progress

- **What's New**: 최근 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 안전 정렬(safety alignment)이 미약하여 몇 가지 해로운 예제들에 대한 미세 조정(fine-tuning)이나 생성 결과의 접두어를 조작하는 등의 전략으로 쉽게 공격될 수 있음을 밝혔습니다. 이 논문에서는 이러한 공격 전략들의 메커니즘을 분석하여, 이들의 공격 메커니즘에 강한 유사성이 있는지 확인하고자 합니다.

- **Technical Details**: LLMs가 해로운 명령을 받았을 때 보호 과정을 세 단계로 나누어 분석합니다: (1) 해로운 명령 인식 (recognizing harmful instructions), (2) 초기 거부 톤 생성 (generating an initial refusing tone), (3) 거부 응답 완료 (completing the refusal response). 이를 통해 Explicit Harmful Attack (EHA)와 Identity-Shifting Attack (ISA)의 공격이 각 단계에 미치는 영향을 조사합니다. 주요 기술로는 logit lens와 activation patching을 사용해 모델의 특정 동작을 유도하는 구성 요소를 식별하고, cross-model probing을 통해 공격 후 표현 변화(representation shifts)를 분석합니다.

- **Performance Highlights**: EHA는 주로 해로운 명령 인식 단계에서 공격을 집중하며, 이 단계에서 모델의 신호 전달 능력을 방해합니다. 반면에, ISA는 이 단계에 큰 영향을 미치지 않습니다. 두 공격 방식 모두 초기 거부 톤 생성 및 거부 응답 완료 단계에서 모델의 동작을 손상시키지만, 이들의 메커니즘과 영향을 미치는 구성 요소는 다릅니다. EHAed와 ISAed 모델 모두 거부 접두어를 유지하지 못하지만, ISAed 모델의 경우 더 심각하게 해로운 내용을 생성합니다. Llama-2 시스템 프롬프트 같은 보안 지향 프롬프트를 추가하면 이러한 문제를 부분적으로 완화할 수 있으나, 효과는 제한적입니다.



### Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection (https://arxiv.org/abs/2405.16178)
- **What's New**: Sparse RAG라는 새로운 패러다임을 제안합니다. 이는 외부 문서 검색(Retrieval)을 통해 자료를 받아오는 과정에서 발생하는 과도한 계산 비용과 지연 시간을 저감시키고자 고안된 방식입니다. Sparse RAG는 병렬처리 방식으로 문서들을 인코딩하고, LLM(Language Models)들이 자동 회귀적으로 높은 관련성을 가진 캐시들만 선택적으로 디코딩하도록 설계되었습니다. 이를 통해 문서 수를 효율적으로 조정하고, 관련성이 낮은 문맥을 배제하여 모델의 생성 품질을 향상시킵니다.

- **Technical Details**: Sparse RAG는 키-값 캐시(key-value cache)를 이용한 대규모 사전 채우기(Pre-filling)와 선택적 디코딩(Selective Decoding)을 결합합니다. Pre-filling 단계에서는 검색된 문서들이 단일 포워드 패스를 통해 인코딩되고, 선택적 디코딩 단계에서는 관련성이 높은 캐시들만을 선택하여 디코딩합니다. 이는 LLM이 메모리 사용량에 제약을 받는 상황에서 낮은 지연 시간과 더 나은 생성 품질을 가능하게 합니다. 특히, LLM은 특별한 제어 토큰을 이용해 적절한 문서를 평가하고, 이러한 문서들의 캐시만을 불러와 디코딩합니다.

- **Performance Highlights**: 두 개의 데이터셋(PopQA와 Biography)에 대한 평가 결과, Sparse RAG는 짧은 형식 및 긴 형식의 생성 작업 모두에서 유사하거나 더 나은 성능을 유지하면서도 기존의 Dense RAG나 PCW-RAG에 비해 훨씬 더 나은 지연 시간을 구현하는 데 성공했습니다. 이를 통해 Sparse RAG의 일반화 가능성을 입증했습니다.



### Bi-reachability in Petri nets with data (https://arxiv.org/abs/2405.16176)
- **What's New**: 이 논문에서는 데이터를 가진 Petri net에 대해 연구합니다. 일반적인 Petri net의 확장으로, 토큰이 무한한 데이터 도메인에서 값을 가지며, 전이(transition)의 실행 가능성은 데이터 값 간의 동일성에 의해 결정됩니다. 이 연구는 bi-reachability 문제에 대한 결정 절차를 제공합니다: 주어진 Petri net과 두 개의 구성(configuration)이 있을 때, 각각의 구성에 도달 가능성이 있는지를 묻습니다.

- **Technical Details**: 데이터를 가진 Petri net은 전통적인 Petri net의 확장으로, 토큰이 단순히 위상적 위치를 가질 뿐만 아니라 실제 데이터 값을 가질 수 있습니다. 전이의 실행 가능성은 이런 데이터 값들이 일치하는지 여부에 달려 있습니다. 연구에서 다루는 bi-reachability 문제는 두 구성의 상호 도달 가능성을 알아보는 문제로, 이는 이전에 결정 가능한 것으로 알려진 coverability 문제를 포함하고 있으며, 아직 결정 불가능한 것으로 알려진 reachability 문제에 포함됩니다.

- **Performance Highlights**: 이번 연구는 bi-reachability 문제를 해결하기 위해 새로운 결정 절차를 제공함으로써, 결정 가능성의 경계를 한 단계 더 확장했습니다. Petri net의 구성 간 상호 도달 가능성을 결정할 수 있는 방법을 제시함으로써, coverability 문제를 넘어서는 성취를 이루었으며, 이는 복잡한 시스템의 분석에 큰 도움이 될 수 있을 것입니다.



### Improving Multi-lingual Alignment Through Soft Contrastive Learning (https://arxiv.org/abs/2405.16155)
Comments:
          8 pages, 1 figures, Accepted at NAACL SRW 2024

- **What's New**: 최신 연구에서는 고품질 다국어 문장 표현(multi-lingual sentence representations)을 개선하기 위한 새로운 방법을 제안했습니다. 이 방법은 사전 학습된 모노링구얼(단일 언어) 모델의 문장 유사성을 기반으로 다국어 임베딩을 정렬합니다. 번역 문장 쌍을 주어진 상태에서, 모노링구얼 교사 모델이 측정한 문장 유사성을 다국어 모델에서 따르도록 훈련하는 방식입니다.

- **Technical Details**: 본 연구에서는 soft labels을 사용한 대비 학습(contrastive learning) 방식을 채택했습니다. soft label은 문장 간 유사성에 따라 정의됩니다. 번역 쌍 {(si, ti)}가 주어지면, 모노링구얼 모델로부터 문장 유사성 행렬(similarity matrix)을 계산하여 다국어 학생 모델의 문장 유사성 행렬을 유도합니다.

- **Performance Highlights**: 실험 결과 5개 언어에 대해 soft labels 기반의 대비 학습이 기존의 hard labels 보다 다양한 벤치마크(bitext mining, STS tasks)에서 뛰어난 성능을 보였습니다. 특히 Tatoeba 데이터셋에서는 LaBSE를 포함한 기존의 다국어 임베딩 방법들보다 우수한 성과를 나타냈습니다.



### DefSent+: Improving sentence embeddings of language models by projecting definition sentences into a quasi-isotropic or isotropic vector space of unlimited dictionary entries (https://arxiv.org/abs/2405.16153)
- **What's New**: 이번 논문은 기존의 DefSent 연구를 개선하여 새로운 문장 임베딩(context embedding) 방법인 DefSent+를 제안합니다. 이 방법은 정의 문장(definition sentences)을 사전 항목의 벡터 공간(vector space)에 투영(project)해 문장 임베딩의 품질을 향상시키는 데 중점을 둡니다. 기존의 DefSent는 단어 임베딩(word embedding)을 사용하여 사전 항목을 표현하는 데 한계가 있어 이를 개선하고자 새로운 접근 방식을 도입했습니다.

- **Technical Details**: 기존 DefSent 방법은 단어 임베딩을 사용하여 사전 항목들을 표현하는 데, 이는 단일 단어로 제한되어 있고 의미 표현이 비등방성(anisotropic)인 문제를 갖고 있습니다. DefSent+는 이러한 한계를 극복하기 위해 점진적으로 사전 항목 임베딩(embedding)을 구축하여, 정의 문장을 준등방성(quasi-isotropic) 또는 등방성(isotropic) 벡터 공간으로 투영할 수 있게 했습니다. 이를 통해 문장 임베딩의 품질을 크게 개선할 수 있습니다.

- **Performance Highlights**: DefSent+는 문장 유사도를 측정하는 작업에서 기존 DefSent보다 성능이 크게 향상되었습니다. 또한, DefSent+를 SIMCSE와 SNCSE와 같은 데이터 증강 모델을 추가 훈련하는 데 사용하면, 수작업으로 라벨링된 데이터셋을 사용하지 않는 접근 방법 중에서 최고 수준의 성능을 달성할 수 있습니다. 그리고 NLP 다운스트림 작업에서도 특성 기반 전이(feature-based transfer)에서 경쟁력을 유지합니다.



### 5W1H Extraction With Large Language Models (https://arxiv.org/abs/2405.16150)
Comments:
          IJCNN 2024

- **What's New**: 이 연구는 뉴스 기사의 5W1H 요소(What, When, Where, Why, Who, How) 추출을 위한 데이터셋을 새롭게 주석하고, 대형언어모델(LLM)들과의 성능 비교를 시도했습니다. 특히, ChatGPT와 같은 모델들이 긴 뉴스 텍스트 및 특정 맥락에서의 5W1H 질문에 대한 답변을 어려워하는 문제를 해결하고자 합니다.

- **Technical Details**: 연구팀은 CNN/DailyMail, XSum, NYT, RA-MDS와 같은 네 가지 뉴스 코퍼스에 대해 3,500개의 항목을 주석하여 고품질 5W1H 데이터셋을 구축했습니다. 또한, Zero-shot/few-shot prompting 및 효율적인 Fine-tuning 기법을 사용하여 원본 뉴스 문서에서 5W1H 요소를 추출했습니다. Prefix-Tuning, Low-Rank Adaptation (LoRA), QLoRA와 같은 최신 Fine-tuning 방법론도 적용되었습니다.

- **Performance Highlights**: 실험 결과, 주석된 데이터셋으로 Fine-tuning된 모델의 성능이 ChatGPT를 능가했음을 보여주었습니다. 또한, 소스 도메인(NYT) 모델을 타겟 도메인(CNN/DailyMail) 코퍼스에 테스트한 결과, 도메인 적응(Domain Adaptation) 능력이 우수함을 확인했습니다.



### iREL at SemEval-2024 Task 9: Improving Conventional Prompting Methods for Brain Teasers (https://arxiv.org/abs/2405.16129)
- **What's New**: 새로운 연구는 세메발-2024 Task 9: BRAINTEASER에 대한 혁신적인 접근 방식을 소개합니다. 이 과제는 라테랄 씽킹(lateral thinking) 능력을 평가하는 다중 선택형 질문 답변을 포함하며, 일반적인 상식 연관성을 피하고 독창적인 사고를 요구합니다.

- **Technical Details**: 연구진은 사전 훈련된 언어 모델의 성능을 향상시키기 위해 Gemini 1.0 Pro Model을 사용했습니다. 정적(few-shot prompting) 및 동적(few-shot prompting) 기술과 모델 생성 추론 전략을 도입하여 대규모 언어 모델(LLM)의 추론 능력을 활용했습니다. 특히, 테스트 데이터와 훈련 데이터 간의 의미적 유사성에 기초한 동적 few-shot prompting 적용을 통해 더 연관성이 높은 예제를 사용하는 방식으로 성능을 개선했습니다. 또한, 모델이 훈련 중에 올바른 선택지에 대한 설명을 생성하도록 하여 예제에 대한 이해도를 심화시켰습니다.

- **Performance Highlights**: 연구 접근 방식은 베이스라인 모델보다 상당한 성능 향상을 보여주었으나, 인간 주석자와 비교했을 때는 아직 미흡한 점이 있었습니다. 제로샷(zero-shot) 및 몇샷(few-shot) 접근 방식 모두 베이스라인 모델인 Chat-GPT 0 Shot 및 Roberta-L 모델보다 성능이 우수했습니다.



### SNOBERT: A Benchmark for clinical notes entity linking in the SNOMED CT clinical terminology (https://arxiv.org/abs/2405.16115)
- **What's New**: SNOBERT라는 새로운 방법을 제안하여, 임상 노트의 텍스트 영역을 SNOMED CT의 특정 개념과 연결하는 방식을 개발했습니다. 이 방법은 BERT 기반 모델을 사용하며 두 가지 주요 단계(후보 선택 단계와 후보 매칭 단계)로 구성됩니다.

- **Technical Details**: SNOBERT는 BERT 모델을 활용하여 임상 노트의 텍스트를 NER(Named Entity Recognition) 및 EL(Entity Linking) 과정을 통해 구조화된 형식으로 변환합니다. 이 방법은 MIMIC-IV-Note 데이터셋을 사용하여 훈련되었으며, SNOMED CT clinical terminology에 맞춰 개념을 링크합니다. 과정은 다음과 같습니다: 첫 번째 단계에서는 NER을 수행하고, 두 번째 단계에서는 각 텍스트 스팬을 SNOMED CT 용어의 개념 ID에 매칭합니다.

- **Performance Highlights**: SNOBERT는 딥러닝에 기반한 기존의 전통적인 방법들보다 뛰어난 성능을 보였으며, 'SNOMED CT Entity Linking Challenge'에서 그 효용성을 입증했습니다.



### COLT: Towards Completeness-Oriented Tool Retrieval for Large Language Models (https://arxiv.org/abs/2405.16089)
- **What's New**: 최근 외부 도구들과 대형 언어 모델(LLMs)의 통합이 급부상하고 있습니다. 하지만 다양한 도구들을 동시에 사용하는 실제 환경에서는, 모든 도구를 LLMs에 직접 통합하는 것이 현실적이지 않습니다. 이를 해결하기 위해, 연구진은 COLT라는 새로운 모델-독립 협력 학습 기반 도구 검색 방식을 제안하였습니다. COLT는 단순히 사용자 쿼리와 도구 설명 간의 의미적 유사성 뿐만 아니라 도구 간의 협력 정보도 고려합니다.

- **Technical Details**: COLT는 두가지 주요 단계로 나뉩니다: 의미 학습(semantic learning)과 협력 학습(collaborative learning)입니다. 처음에는 PLM 기반 검색 모델을 미세 조정하여 쿼리와 도구 간의 의미적 관계를 학습합니다. 그 후, 쿼리, 상황(scenes), 도구들 간의 이중 뷰 그래프 협력 학습 프레임워크를 도입하여 도구들의 복잡한 협력 관계를 포착합니다. 이를 통해 COLT는 높은 순위의 협력 정보를 보다 잘 통합합니다.

- **Performance Highlights**: 광범위한 실험을 통해 COLT는 기존의 밀집 검색 방법(dense retrieval methods)보다 우수한 성능을 보임을 입증했습니다. 특히, 11M 파라미터를 가진 BERT-mini 모델이 COLT를 통해 340M 파라미터를 가진 BERT-large 모델을 능가하는 결과를 보여줍니다. 또한, 새로운 ToolLens 데이터셋을 공개하여 도구 검색 연구를 더욱 지원할 계획입니다.



### Keypoint-based Progressive Chain-of-Thought Distillation for LLMs (https://arxiv.org/abs/2405.16064)
Comments:
          Accepted by ICML 2024

- **What's New**: 새로운 연구는 대형 언어 모델(LLMs)로부터 작은 학생 모델로 추론 능력을 이전하는 강력한 기술인 Chain-of-thought(CoT) 증류를 발전시킵니다. 기존 방법들은 LLMs가 생성한 단계별 논리를 모방하도록 학생을 요구하지만 몇 가지 문제를 겪습니다. 이번 연구에서는 KPOD라는 새롭고 통합된 프레임워크를 제안하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: KPOD는 두 가지 주요 혁신을 포함합니다. 첫째, '토큰 가중 모듈(Token Weighting Module)'을 통해 논리에서 핵심적 토큰을 정확하게 모방할 수 있도록 학습시킵니다. 둘째, '진행적 증류 전략(Progressive Distillation Strategy)'을 통해 쉬운 단계에서 어려운 단계로 점진적으로 학습하도록 설계합니다. 이 전략은 최종 추론 단계를 생성하는 것부터 시작하여 점진적으로 전체 논리에 이르기까지 학습을 확장합니다.

- **Performance Highlights**: KPOD는 네 가지 추론 벤치마크 테스트에서 큰 성능 향상을 보여주었으며, 기존 방법들보다 상당히 우수한 결과를 나타냈습니다. 토큰의 중요도를 평가하고 단계별 추론 난이도를 정확하게 측정하는 고유의 손실 함수 및 가치 함수(value function)를 통해 학생 모델의 성능을 극대화합니다.



### SPP: Sparsity-Preserved Parameter-Efficient Fine-Tuning for Large Language Models (https://arxiv.org/abs/2405.16057)
- **What's New**: 이번 논문은 대용량 언어 모델(LLMs)의 미세 조정(fine-tuning)과 배포의 어려움을 해결하기 위해 SPP(Sparsity-Preserved Parameter-efficient fine-tuning)라는 새로운 방법을 소개합니다. 이는 경량 학습 가능 매트릭스를 사용하여 희소 LLM 가중치를 최적화하고, 미세 조정 및 가중치 병합 단계에서 모델의 희소성 패턴과 비율을 유지합니다.

- **Technical Details**: 기존의 포스트 트레이닝 가지치기(post-training pruning) 방법들은 종종 원래 성능을 유지하는 데 실패하지만, SPP는 희소 패턴과 비율을 유지하면서 모델 성능을 향상시킵니다. SPP는 경량 학습 가능 매트릭스를 사용하는 방법으로, 행렬 곱셈과 잔차 추가를 통해 원래의 희소 패턴과 비율을 유지합니다. LLaMA 및 LLaMA-2 모델 군에서 SPP의 효과가 검증되었습니다.

- **Performance Highlights**: SPP는 다양한 희소 패턴(예: 비구조적 희소성 및 N:M 희소성)과 비율에서 높은 성과를 보여주었습니다. 특히 75%와 같은 높은 희소성 비율에서도 모델의 성능을 크게 향상시켰습니다. 이는 SparseGPT 및 Wanda와 같은 최신 포스트 트레이닝 가지치기 방법에서도 개선된 성능을 보였습니다.



### Incremental Comprehension of Garden-Path Sentences by Large Language Models: Semantic Interpretation, Syntactic Re-Analysis, and Attention (https://arxiv.org/abs/2405.16042)
Comments:
          Accepted by CogSci-24

- **What's New**: 이 연구는 GPT-2, LLaMA-2, Flan-T5, RoBERTa와 같은 대규모 언어 모델(LLMs)을 사용하여 정원 길(sentence)이 제시하는 일시적 모호성을 조사하고, 인간과 LLMs가 이러한 문장을 처리하는 방식과 불명확성을 어떻게 해소하는지 비교합니다.

- **Technical Details**: 연구에서는 24개의 정원 길 문장을 사용하여 LLMs의 문장 처리 방식을 측정했습니다. 각 문장마다 오해와 정확한 해석에 대한 이해 질문을 통해 모델들의 동적 해석을 평가했습니다. 세 가지 실험을 통해 질문 응답 과제를 사용하여 LLMs의 의미 해석을 측정했고, 문장 구조 해석(implicit parse tree) 변화를 추적하며, 질문 처리 중 주의 메커니즘이 불확실한 정보에 어떻게 반응하는지 시각화했습니다.

- **Performance Highlights**: 추가 구문 정보(예: 절 경계를 구분하는 쉼표)가 제공될 때, 인간과 LLMs 간의 정원 길 문장 처리 및 불명확성 해소에 대한 약속된 정렬을 보였습니다. 인간과 비교해볼 때, LLMs는 오해의 단계를 거쳐 분명한 해석 지점에서 올바른 해석으로 전환하는 경향이 있었습니다.



### Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models (https://arxiv.org/abs/2405.15984)
Comments:
          29 pages, 6 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 등장과 함께 등장한 In-Context Learning(ICL)은 효과적이고 효율적인 학습 방법으로 주목받고 있습니다. 이 논문은 ICL의 다양한 변형 방법들이 적대적 공격(adversarial attacks)에 얼마나 민감한지 탐구합니다. 특히, Retrieval-Augmented ICL 방법의 취약성을 심도 있게 분석하고, 새로운 비훈련 기반의 적대적 방어 방법인 DARD를 제안합니다.

- **Technical Details**: ICL은 주로 예제를 통해 언어 모델을 학습시키는 방법으로, 이 중 Retrieval-Augmented ICL은 의미적으로 관련된 예제를 검색하여 성능을 향상시킵니다. 그러나 이러한 방법이 테스트 샘플, 데모, 검색된 데이터에 대한 다양한 적대적 공격에 얼마나 견딜 수 있는지는 아직 명확하지 않습니다. 연구 결과, Retrieval-Augmented ICL 모델은 테스트 샘플 공격에 대해 4.87%의 공격 성공률(ASR) 감소로 더 높은 강인성을 보였지만, 데모 공격에 대해서는 2% 증가한 ASR을 보였습니다.

- **Performance Highlights**: DARD는 훈련 없이도 적대적 방어를 가능하게 하며, 15%의 ASR 감소를 달성하여 기존 방식들보다 성능과 강인성을 크게 향상시킵니다. 이 방법은 특히 적대적으로 교란된 샘플을 검색 풀에 추가함으로써 모델의 강인성을 높일 수 있습니다.



### A hierarchical Bayesian model for syntactic priming (https://arxiv.org/abs/2405.15964)
Comments:
          6 pages; accepted to CogSci 2024

- **What's New**: 새로운 연구는 문장 부호 사용 동조화(syntactic priming) 현상의 주요 세 가지 특징인 Lexical Boost, Inverse Frequency Effect, Asymmetrical Decay를 통합할 수 있는 일반적인 학습 프레임워크 계층적 베이지안 모델(Hierarchical Bayesian Model; HBM)을 제안합니다.

- **Technical Details**: 이 모델은 문법적인 지식을 문법적 통계의 계층적 구조로 나타내며, 하위 수준은 동사-특정 편향(verb-specific biases)을 나타내고, 상위 수준은 이러한 동사-특정 편향의 집합으로서 추상적 편향(abstract bias)을 나타냅니다. 교육 데이터를 기반으로 베이지안 추론(Bayesian inference)에 의해 업데이트됩니다.

- **Performance Highlights**: 시뮬레이션을 통해 계층적 베이지안 모델(HBM)은 문장 부호 사용 동조화(syntactic priming)의 세 가지 주요 특성인 Lexical Boost, Inverse Frequency Effect, Asymmetrical Decay를 잘 캡처하는 것을 보여주었습니다. 이는 문장 부호 사용 동조화가 잔여 활성화(residual activation) 설명으로만 이해되지 않으며, 암묵적 학습(implicit learning) 설명도 가능함을 시사합니다.



### Zero-Shot Spam Email Classification Using Pre-trained Large Language Models (https://arxiv.org/abs/2405.15936)
- **What's New**: 이 논문은 사전 교육된 대형 언어 모델(LLM)을 사용하여 스팸 이메일을 분류하는 방법을 탐구합니다. 특히, OpenAI의 GPT-4와 같은 독점 LLM과 Flan-T5 같은 오픈 소스 모델을 사용하여 스팸 필터링의 가능성을 평가합니다. 'zero-shot prompting' 기법을 사용하여 기존에 훈련되지 않은 상태에서도 스팸 분류가 가능한지 연구합니다.

- **Technical Details**: 이 연구에서는 두 가지 분류 접근 방식을 사용합니다. 첫 번째는 이메일 제목과 본문의 원본 내용을 잘라서 분류하는 방법이고, 두 번째는 ChatGPT로 요약된 내용을 기반으로 분류하는 방법입니다. 연구는 유명한 SpamAssassin 데이터를 사용하여 수행되었으며, 추가 훈련 없이 전체 데이터셋을 사용한 평가를 통해 성능을 분석하였습니다. Flan-T5 모델은 원본 내용 접근 방식에서 90%의 F1-score를, GPT-4 모델은 요약된 내용 접근 방식에서 95%의 F1-score를 기록했습니다.

- **Performance Highlights**: Flan-T5 모델은 90%의 F1-score를 달성하였고, GPT-4 모델은 95%의 F1-score를 달성했습니다. 이러한 결과는 단일 데이터셋에만 적용되었지만, LLM을 기반으로 한 서브태스크(예: 요약 및 분류)의 분류 파이프라인의 가능성을 시사합니다. 그러나 독점 모델의 높은 운영 비용과 LLM의 일반적인 추론 비용은 실제 환경에서의 스팸 필터링 배치를 어렵게 할 수 있습니다.



### SLIDE: A Framework Integrating Small and Large Language Models for Open-Domain Dialogues Evaluation (https://arxiv.org/abs/2405.15924)
Comments:
          Accepted by ACL2024 Findings

- **What's New**: 오픈 도메인 대화 시스템( open-domain dialogue systems)에서 표준 응답(gold standard responses)을 평가하는 문제를 해결하기 위해 새로운 프레임워크 SLIDE(Small and Large Integrated for Dialogue Evaluation)가 제안되었습니다. 이 프레임워크는 소형 특화 모델(SLM, Small and specialised model)과 대형 언어 모델(LLMs, Large Language Models)을 통합하여 평가를 수행합니다. 이는 상위 수준의 성능을 보여주며 인간 평가와 더 높은 상관성을 나타냅니다.

- **Technical Details**: SLIDE 프레임워크는 여러 기술을 도입합니다: (1) 강건한(response) 응답 임베딩과 비강건한 응답 임베딩을 구분하는 대조 학습(contrastive learning)을 사용하고, (2) 임베딩 코사인 거리(embedding cosine distances)와 유사성을 신경망( neural networks)으로 학습하여 결합한 새로운 의미 민감성 메트릭(semantic sensitivity metric)을 도입하며, (3) SLM과 LLM의 평가 결과를 통합합니다.

- **Performance Highlights**: 실험 결과에 따르면 SLIDE는 분류 및 평가 작업에서 최신 기술을 선도하는 성능(state-of-the-art performance)을 달성하며, SLIDE 평가자는 인간 평가와 더 나은 상관관계를 보입니다. 예를 들어 DailyDialog++ 데이터셋을 사용한 분류 작업에서는 SLM이 GPT-3.5 등 일부 LLM보다 뛰어난 성과를 보여줍니다.



### Enhancing Augmentative and Alternative Communication with Card Prediction and Colourful Semantics (https://arxiv.org/abs/2405.15896)
- **What's New**: 이 논문은 브라질 포르투갈어(Brazilian Portuguese)를 위한 변형 기반 언어 모델(transformer-based language models)과 컬러풀 시맨틱스(Colourful Semantics, CS)를 통합하여 보완대체 의사소통(AAC) 시스템을 향상시키는 접근 방식을 제시합니다. 적응된 BERT 모델인 BERTptCS를 소개하며, 이는 의사소통 카드의 예측 정확도와 문맥적 관련성을 향상시키기 위해 CS 프레임워크를 통합했습니다. 이는 복잡한 의사소통 필요(CCN)를 가진 개인에게 매우 중요한 요소입니다.

- **Technical Details**: 이 연구는 브라질 포르투갈어로 학습된 BERT 모델인 BERTimbau를 활용하여 PrAACT 방법을 적용했습니다. 이 방법은 코퍼스 주석(Corpus Annotation), 모델 세분화(Model Fine-Tuning), 어휘 인코딩(Vocabulary Encoding)의 세 가지 주요 단계로 구성됩니다. AACptCorpus라는 합성 코퍼스를 사용하여 BERTptCS와 BERTptAAC 두 가지 모델을 각각 CS 구조를 통합하여 세분화했습니다. CS 역할을 적용하여 문법 구조와 언어적 구조를 주석했습니다.

- **Performance Highlights**: 모델 평가 결과, CS 구조를 통합한 BERTptCS 모델이 다양한 지표(top-k 정확도, Mean Reciprocal Rank (MRR), Entropy@K)에서 BERTptAAC 모델보다 크게 우수한 성능을 보였습니다. CS 구조를 통합하면 예측의 정확도와 관련성이 향상됨은 물론, 사용자의 입력을 보다 직관적이고 문맥적으로 이해할 수 있게 되어 의사소통이 더욱 효과적으로 이루어집니다.



### DuanzAI: Slang-Enhanced LLM with Prompt for Humor Understanding (https://arxiv.org/abs/2405.15818)
- **What's New**: 언어의 복잡함은 유머와 문화적 뉘앙스를 담은 속어 표현에서 두드러집니다. 이러한 현상은 특히 디지털 소통에서 점점 더 많이 나타나고 있습니다. 기존의 AI 모델, 특히 ChatGPT-3.5는 이러한 뉘앙스를 이해하는데 어려움을 겪고 있습니다. 이 연구에서는 새로운 접근법인 DuanzAI를 소개하며, 이는 대형 언어 모델 (LLMs)이 중국어 속어를 깊이 이해할 수 있도록 향상시킵니다.

- **Technical Details**: DuanzAI는 정교하게 스트림된 데이터셋과 고급 기술들을 활용하여 인간의 표현과 AI 이해의 격차를 메웁니다. 우리의 실험은 LLMs의 성능을 중심으로 맞춤형 Punchline Entity Recognition (PER) 시스템과 비교하였습니다. 이 PER 시스템은 음성 일치 (Phonetic matching)와 pinyin2hanzi 기술을 통합합니다.

- **Performance Highlights**: 이러한 통찰을 바탕으로 우리는 ChatDAI라는 진보된 챗봇을 개발하였으며, 이를 통해 더 맥락에 맞는 응답을 제공할 수 있게 되었습니다. 또한 우리의 코드는 공개되어 있어, 접근 가능한 링크에서 확인할 수 있습니다.



### Matryoshka Multimodal Models (https://arxiv.org/abs/2405.17430)
Comments:
          Project Page: this https URL

- **What's New**: 대형 멀티모달 모델(Large Multimodal Models, LMMs)인 LLaVA 등의 모델은 시각-언어적 추론에서 탁월한 성능을 보여주고 있습니다. 하지만 이러한 모델들은 고해상도 이미지와 비디오와 같은 밀집된 시각적 시나리오에서는 비효율성을 초래할 수 있습니다. 이를 해결하기 위해 Matryoshka Dolls 개념에서 영감을 받아, M3: Matryoshka Multimodal Models를 제안합니다. 이 모델은 시각 콘텐츠를 다양한 세밀도로 캡처하는 중첩된 시각 토큰 집합으로 표현하는 방법을 학습합니다.

- **Technical Details**: M3는 이미지의 시각적 토큰을 여러 단계의 세밀도로 표현하도록 LMM을 학습시킵니다. 훈련 과정 동안 이미지는 점점 더 많은 토큰으로 인코딩되어 시각적 정보를 조정할 수 있습니다. 예를 들어, 자연 이미지의 경우 고수준의 의미론적 정보(레스토랑, 소녀 등)는 낮은 단계의 세밀도에서, 세부사항(페프시 컵, 흰색 종이 가방 등)은 높은 단계의 세밀도에서 캡처됩니다. 모든 훈련 설정은 LLaVA와 동일하게 유지됩니다.

- **Performance Highlights**: M3 모델은 시각 콘텐츠를 효율적이고 적응적으로 표현할 수 있습니다. 하나의 모델 가중치 세트 내에서 정보 밀도별로 다양한 세밀도의 중첩된 시각 토큰을 생성할 수 있습니다. 이를 통해 예측 과정에서 이미지에 필요한 최적의 정보-성능 균형을 제어할 수 있습니다. 예를 들어, 복잡한 이미지를 위해 모든 시각 토큰을 사용할 수 있고, 단순한 이미지는 적은 수의 토큰으로 표현할 수 있습니다. 또한, COCO와 같은 자연 장면 데이터셋의 경우 약 9개의 시각 토큰만 사용해도 높은 정확도를 얻을 수 있습니다.



### Privacy-Aware Visual Language Models (https://arxiv.org/abs/2405.17423)
Comments:
          preprint

- **What's New**: 이 논문은 Visual Language Models (VLMs)가 개인정보를 어떻게 처리하는지에 대한 이해를 발전시키는 것을 목표로 합니다. 이를 위해 여권이나 지문과 같은 8개의 민감 범주에서 이미지를 포함하는 새로운 벤치마크 PrivBench를 도입했습니다. 우리는 최신의 10개 VLM을 이 벤치마크로 평가하여 일반적으로 개인정보의 이해가 제한적임을 확인했습니다. 이를 바탕으로 시각적인 프라이버시에 대한 지식을 VLM에게 부여하기 위한 새로운 인스트럭션 튜닝 데이터셋 PrivTune을 소개했습니다.

- **Technical Details**: PrivTune을 사용하여 사전 훈련된 두 VLM인 TinyLLaVa와 MiniGPT-v2를 튜닝한 결과, 민감한 콘텐츠를 인식하는 능력이 크게 향상되었습니다. 이 과정에서 프라이버시 튜닝이 VQA와 같은 표준 벤치마크에서의 성과에 미치는 영향은 최소화되었습니다. 이러한 접근 방식은 실제 데이터 처리에서 안전하게 사용할 수 있는 프라이버시 인식 VLM을 구축하는 첫 단계를 제공합니다.

- **Performance Highlights**: 튜닝된 TinyLLaVa와 MiniGPT-v2는 PrivBench에서 GPT4-V를 능가하는 성과를 보여주었습니다. 이러한 강력한 개선은 모델의 표준 벤치마크 성능에 거의 영향을 미치지 않으면서도 이루어졌습니다.



### KSW: Khmer Stop Word based Dictionary for Keyword Extraction (https://arxiv.org/abs/2405.17390)
- **What's New**: 이 논문은 Khmer 언어에 특화된 Keyword Extraction(키워드 추출) 접근법인 KSW를 소개합니다. KSW는 특별히 개발된 stop word(불용어) 사전을 활용하여 키워드 추출의 정확도와 관련성을 크게 향상시키고자 합니다.

- **Technical Details**: 현재 Khmer 언어를 위한 자연어 처리(NLP) 자원이 제한적인 상황에서 효과적인 키워드 추출이 어려웠습니다. KSW는 이를 해결하기 위해 맞춤형 stop word 사전을 개발하고, 불용어 제거를 위한 전처리 방식을 구현했습니다. 이러한 전처리 방법론은 의미 있는 키워드를 더 잘 추출할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, KSW는 이전 방법들에 비해 정확도와 관련성 측면에서 상당한 향상을 보여주었습니다. 이는 Khmer 텍스트 처리와 정보 검색 분야를 크게 발전시킬 잠재력을 지니고 있습니다. 해당 논문에서 개발한 KSW 자원들은 GitHub(기탁) 저장소에서 이용 가능합니다.



### ReMoDetect: Reward Models Recognize Aligned LLM's Generations (https://arxiv.org/abs/2405.17382)
Comments:
          20 pages

- **What's New**: 최근 강력한 대형 언어 모델(LLMs)의 뛰어난 성능과 접근성이 사회적 위험을 증가시키고 있습니다. 본 연구에서는 LLM이 생성한 텍스트(LGT)를 효과적으로 탐지할 수 있는 새로운 방법을 제안합니다. 특히, 최신 강력한 LLM들이 인간이 선호할 만한 텍스트를 생성하도록 학습된다는 점에 주목하여, 이러한 텍스트는 인간이 작성한 텍스트보다도 더 높은 선호도를 가진다는 것을 발견했습니다.

- **Technical Details**: LLM들의 일반적인 특성을 이용해 LGT를 감지하기 위해 보상 모델(reward model)을 활용한 두 가지 학습 방법을 제안합니다. 첫째, 보상 모델을 지속적으로 미세 조정하여 LGT가 인간이 작성한 텍스트보다 더 높은 보상 점수를 받을 수 있도록 합니다. 둘째, 인간이 작성한 텍스트를 LLM을 사용해 부분적으로 재구성한 Human/LLM 혼합 텍스트를 새롭게 생성하여, 이 텍스트를 학습 데이터로 활용해 감지 경계(decision boundary)를 더 잘 학습할 수 있도록 합니다.

- **Performance Highlights**: ReMoDetect는 6개의 텍스트 도메인과 12개의 정렬된 LLM을 대상으로 광범위한 평가를 거쳤으며, 기존의 탐지 방법을 능가하는 성능을 입증했습니다. 특히, GPT-4와 Claude3 Opus 등의 모델에서 평균 AUROC 점수로 각각 97.9%, 98.6%를 기록하며 최고 성능을 달성했습니다. 또한, 악의적인 재작성 공격에 대한 견고성, 탐지 텍스트 길이, 새로운 분포에 대한 탐지 성능 등 다양한 측면에서 우수한 견고성을 보여주었습니다.



### Exploring and steering the moral compass of Large Language Models (https://arxiv.org/abs/2405.17345)
- **What's New**: 이 연구는 최신 대형 언어 모델(LLMs)에 대한 포괄적인 비교 분석을 통해 그들의 도덕적 프로파일을 평가하려는 시도를 제안합니다. 여러 모델을 다양한 윤리적 딜레마에 직면하게 하여 평가한 결과, 독점 모델들은 대부분 공리주의적 특징을 보였고, 개방형 모델들은 주로 가치 기반 윤리와 일치했습니다. 더불어, '도덕적 기초 질문지'(Moral Foundations Questionnaire)를 사용했을 때 대부분의 모델이 강한 자유주의적 편향을 보였으며, Llama 2를 제외한 모든 모델이 이러한 편향을 나타냈습니다.

- **Technical Details**: 이 연구는 주로 세 가지 목표를 가지고 진행되었습니다: 첫째, LLM들이 다양한 윤리적 딜레마를 해결하는 능력을 평가하고 이들의 반응이 어떤 윤리적 사상과 일치하는지 분석했습니다. 둘째, 도덕적 기초 질문지를 통해 모델들의 도덕적 프로파일을 비교하고 인간 여러 집단과의 연관성을 확인했습니다. 셋째, 모델의 도덕적 컴퍼스를 다양한 윤리 학파로 유도할 수 있는 새로운 '유사성 특정 활성화 조정 기법'(similarity-specific activation steering technique)을 제안했습니다.

- **Performance Highlights**: 연구팀은 4개의 독점 모델(Anthropic의 Claude-3-Sonnet, OpenAI의 GPT-3.5-Turbo-0613 및 GPT-4-Turbo-2024-04-09, Google의 Gemini Pro 1.5)과 4개의 개방형 가중치 모델(Google의 Gemma-2B, META의 Llama-2-7B, Nexusflow의 Starling-LM-7B-Beta)을 대상으로 실험을 진행했습니다. 각 모델에 클래식한 윤리적 딜레마 질문을 던지고, 모델들의 도덕적 정렬과 편향을 평가했습니다. 결과적으로, 독점 모델들은 흔히 공리주의적 특징을 보였고, 개방형 모델들은 가치 기반 윤리와 일치했습니다. 이로써 이미 배포된 LLM들에도 윤리적 차원이 존재한다는 점을 강조하며, 이는 일반적으로 간과되고 있는 중요한 요소입니다.



### Collage is the New Writing: Exploring the Fragmentation of Text and User Interfaces in AI Tools (https://arxiv.org/abs/2405.17217)
Comments:
          19 pages, 7 figures, 2 tables, ACM DIS 2024

- **What's New**: 이번 에세이에서는 아방가르드 문학에서 따온 '콜라주' 개념을 AI 글쓰기 도구 설계에 적용하는 방안을 제안하고 탐구합니다. 글쓰기 인터페이스에서 텍스트를 분할하고, 서로 다른 목소리를 나란히 배치하며, 다양한 출처에서 재료를 통합하고, 수동적인 글쓰기에서 편집과 구성 의사결정으로의 전환 등을 중점으로 다룹니다.

- **Technical Details**: 에세이는 네 가지 주요 측면을 통해 콜라주 개념을 설명합니다. 첫째, 글쓰기 인터페이스에서 텍스트를 조각내는 기능입니다. 둘째, 사용자들이 콘텐츠와 명령 문장을 나란히 배치하는 방식입니다. 셋째, 다양한 출처의 자료를 통합하는 방식으로, 예를 들어 AI가 제안하는 텍스트가 포함됩니다. 마지막으로, 수동적인 글쓰기에서 벗어나 생성된 텍스트 조각을 선택하고 배열하는 편집적 결정 의사결정으로의 전환입니다.

- **Performance Highlights**: 이 에세이는 콜라주를 분석 및 구성적인 렌즈로 사용하여 최근 AI 글쓰기 도구의 사용자 인터페이스 디자인을 분석하고 새로운 설계 방향을 제시합니다. 또한, 역사적으로 문학 콜라주를 통해 작가들이 표현했던 우려를 AI 글쓰기 도구에 관련시킴으로써 비판적 관점을 제공하고자 합니다. 그리고 궁극적으로 AI 글쓰기 도구 설계에 문학적 개념을 적용하여 설계 이론을 발전시키는 목표를 가지고 있습니다.



### Exploiting the Layered Intrinsic Dimensionality of Deep Models for Practical Adversarial Training (https://arxiv.org/abs/2405.17130)
- **What's New**: SMAAT(Scalable Manifold Aware Adversarial Training)이라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 매니폴드 가정(manifold conjecture)을 활용하여 오프-매니폴드 적대적 예제(OFM-AEs)는 더 높은 강인성을 제공하고 온-매니폴드 적대적 예제(ONM-AEs)는 더 나은 일반화를 제공하는 패턴을 활용합니다. 이를 통해 기존의 적대적 훈련(AT)과 비교했을 때 더 높은 확장성과 성능을 자랑합니다.

- **Technical Details**: SMAAT은 중간 레이어 중 본질적 차원(intrinsic dimension)이 가장 낮은 레이어를 섭동하여 더 많은 OFM-AEs를 생성합니다. 이는 PGD(프로젝티드 그래디언트 디센트) 체인의 길이를 줄여 계산 비용을 낮추는 효과가 있습니다. 비전 모델(vision model)과 디코더 기반 언어 모델(dec-LLMs)은 초기 레이어에서 본질적 차원이 낮고, 인코더 기반 언어 모델(enc-LLMs)은 후반 레이어에서 본질적 차원이 낮다는 것을 발견했습니다. 이러한 통찰을 바탕으로 SMAAT은 enc-LLMs의 경우 마지막 레이어까지의 그래디언트만 계산하므로 계산 효율성을 크게 향상시킵니다.

- **Performance Highlights**: SMAAT은 감정 분류, 안전 필터링, RAG 설정에서의 검색기( retriever) 강화 등 다양한 작업에서 뛰어난 성능을 발휘했습니다. 예를 들어, 감정 분류 작업에서는 AGNEWS, IMDB, YELP 데이터셋에서 BERT와 RoBERTa 모델의 강인성을 각각 8.6%, 15.7%, 28.8% 및 6.0%, 5.8%, 19.0% 개선했습니다. 안전 필터링 작업에서는 GCG 공격으로 생성된 유해 프롬프트를 97-100% 정확도로 식별했습니다. 또한, RAG 실험에서는 Contrevier 모델의 강인성을 80% 이상 향상시켰습니다. 더불어 SMAAT은 표준 AT에 비해 GPU 시간을 25-33%만 사용하면서도 뛰어난 성능을 유지했습니다.



### LLM-Optic: Unveiling the Capabilities of Large Language Models for Universal Visual Grounding (https://arxiv.org/abs/2405.17104)
- **What's New**: LLM-Optic이라는 혁신적인 방법을 제안합니다. 이 방법은 대규모 언어 모델(LLM)을 '텍스트 렌즈(Text Grounder)'로 사용하여 복잡한 텍스트 쿼리를 정확히 파악한 후, 기존 비주얼 그라운딩 모델을 활용해 후보 박스를 생성하여 이미지와 텍스트를 연결합니다. 마지막으로, 대규모 멀티모달 모델(LMM)을 '비주얼 렌즈(Visual Grounder)'로 사용하여 원래 쿼리에 가장 잘 맞는 대상 오브젝트를 선택합니다.

- **Technical Details**: LLM-Optic은 세 가지 주요 모듈로 구성됩니다. 먼저, LLM이 텍스트 렌즈(Text Grounder)로 복잡한 텍스트 쿼리를 처리합니다. 그 결과는 후보 위치 지정 및 마크 설정(Candidate Positioning and Setting Marks) 모듈로 전달되어 대상 설명에 맞는 후보 오브젝트의 바운딩 박스를 생성하고, 각 바운딩 박스를 숫자로 표시합니다. 마지막으로, LMM을 사용하는 비주얼 렌즈(Visual Grounder)는 표시된 오브젝트 중 쿼리 텍스트에 가장 잘 맞는 오브젝트를 선택합니다.

- **Performance Highlights**: 여러 어려운 벤치마크에서 LLM-Optic은 추가 학습이나 파인튜닝 없이 기존 비주얼 그라운딩 모델보다 우수한 성능을 발휘했습니다. 특히 RefCOCOg 검증 세트에서 22%의 성능 향상을 달성했습니다.



### Phase Transitions in the Output Distribution of Large Language Models (https://arxiv.org/abs/2405.17088)
Comments:
          21 pages, 4 figures

- **What's New**: 새로운 연구는 물리 시스템에서 온도 같은 매개변수를 변화시켜 위상 전이가 발생하는 현상을 대형 언어 모델(LLM)에서도 유사하게 관찰할 수 있음을 발견했습니다. 특히, 통계적 거리(statistical distances)를 사용하여 생성된 출력물의 분포 변화를 정량화함으로써 LLM의 새로운 행동 위상 및 미탐험 전이를 발견할 수 있는 방법을 제안합니다.

- **Technical Details**: 연구는 f-divergences와 같은 통계적 거리를 활용하여 LLM의 출력 텍스트 분포 변화를 측정하는 방법론을 제시합니다. 이는 온도 하이퍼파라미터와 학습 에폭(epoch), 입력 프롬프트의 정수 등 세 가지 제어 매개변수로 변화를 측정했습니다. 해당 방법론은 시스템의 미지 영역을 탐색하는 데 유용하며, LLM의 발전 과정을 객관적으로 매개변수화하는 데 기여할 수 있습니다.

- **Performance Highlights**: 시험 결과, Llama와 Mistral 모델은 정수를 무질서하게 열거하는 반면, 베이스 모델들은 그렇지 않음을 발견했습니다. 텍스트 출력에서도 Sharp 전이가 나타났으며, 온도에 따른 세 가지 명확한 행동 위상(‘동결’, ‘일관된’ 및 ‘무질서’을)도 성공적으로 매핑했습니다. 또한, 온도가 증가함에 따라 LLM의 평균 에너지가 감소하는 '음성 열용량'을 관찰했습니다. 학습 동안 다수의 프롬프트 간에 텍스트 출력을 통해 가중치 분포의 급격한 변화와 전이가 발생함도 확인했습니다.



### Leveraging small language models for Text2SPARQL tasks to improve the resilience of AI assistanc (https://arxiv.org/abs/2405.17076)
Comments:
          To appear in Proceedings of the Workshop on Linked Data-driven Resilience Research 2024 (D2R2) co-located with Extended Semantic Web Conference 2024 (ESWC 2024)

- **What's New**: 이번 연구에서는 10억 개 미만의 파라미터를 가진 언어 모델을 미세조정하면 자연어를 SPARQL 쿼리로 번역할 수 있음을 보여줍니다. 학술에서 실세계에 이르는 세 가지 다른 데이터셋을 사용하여, 성공적인 훈련을 위해 훈련 데이터가 충족해야 하는 전제 조건을 식별했습니다. 이를 통해 사용자가 저비용 하드웨어로 AI 보조 기술을 사용할 수 있도록 하고 외부 요인에 대한 탄력성을 높일 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 10억 개 미만의 파라미터를 가진 작은 언어 모델들을 대상으로 실험을 진행했습니다. GPT4와 같은 상업적 대형 언어 모델들은 데이터 보호 및 높은 비용의 관점에서 문제가 있으며, SPARQL 쿼리 생성에서도 여전히 도전을 받고 있습니다. 따라서 저자들은 저비용 하드웨어에서 호스팅 가능한 작은 모델이 이에 대응할 수 있는지를 조사합니다. 구체적으로는 자연어 질문을 SPARQL 쿼리로 번역하는 작업을 다루며, 이는 SPARQL에 익숙하지 않은 사람들이 지식 그래프에서 지식을 추출할 수 있도록 합니다.

- **Performance Highlights**: 작은 파라미터를 가진 모델들이 성능 면에서 상업적으로 큰 모델들을 능가할 수 있는지에 대한 시각을 제공합니다. 특히 SQLCoder-7B가 SQL에서 GPT4를 능가하는 사례 등을 언급하며, SPARQL 쿼리 생성에서도 작은 모델이 유의미한 성능을 발휘할 가능성을 제시합니다. 또한, 공개적으로 접근 가능한 모델들만 선택하여 실험을 진행하며, 연구 결과를 누구나 활용할 수 있도록 하는데 주안점을 두었습니다.



### Generation and human-expert evaluation of interesting research ideas using knowledge graphs and large language models (https://arxiv.org/abs/2405.17044)
Comments:
          10 pages; 5 figures

- **What's New**: 최근 논문에서는 AI 시스템을 통해 수백만 개의 연구 논문에서 새로운 연구 아이디어를 생성하는 SciMuse를 소개합니다. 이 AI 시스템은 5,800만 개 이상의 과학 논문을 활용해 지식을 그래프로 구성하고, 이를 바탕으로 GPT-4를 통해 개인 맞춤형 연구 아이디어를 제안합니다.

- **Technical Details**: SciMuse는 논문의 제목과 초록에서 과학 개념을 추출하고 이들 간의 관계를 시각화한 지식 그래프를 사용합니다. 약 244만 개의 논문을 기반으로 Rapid Automatic Key-word Extraction (RAKE) 알고리즘을 통해 후보 개념을 추출하고, GPT와 위키피디아, 인간 평가자를 통해 이를 정제했습니다. 이후 OpenAlex 데이터베이스에서 작성된 5,800만 개 이상의 논문을 활용해 최종 지식 그래프를 만들었습니다. 이 지식 그래프를 기반으로 연구자의 관심사를 파악하여 맞춤형 제안을 생성합니다.

- **Performance Highlights**: Max Planck Society의 100명 이상의 연구 그룹 리더를 대상으로 4,000개 이상의 AI-생성 맞춤형 연구 아이디어를 흥미도에 따라 평가한 결과, 높은 예측 정확도를 보였습니다. 특히, 최상위 N개의 예측된 흥미로운 제안에 대해 50% 이상의 정확도를 달성했습니다. 이는 SciMuse가 과학자들에게 매우 유망한 연구 아이디어와 협력 기회를 제안할 수 있음을 보여줍니다.



### Vision-and-Language Navigation Generative Pretrained Transformer (https://arxiv.org/abs/2405.16994)
- **What's New**: Vision-and-Language Navigation (VLN) 분야에서 새로운 접근법인 VLN-GPT 모델을 도입했습니다. 이 모델은 GPT-2를 기반으로 하는 transformer decoder 아키텍처를 채택하여 과거의 위치와 행동을 기록하는 encoder 모듈의 필요성을 없앴습니다. 이를 통해 모델의 복잡성 및 리소스 소비를 줄이며, 더 효율적인 탐색을 가능하게 합니다.

- **Technical Details**: VLN-GPT 모델은 BERT 기반의 텍스트 임베딩 모듈과 Vision Transformer (ViT) 기반의 관찰 임베딩 모듈을 포함하고 있으며, GPT-2 기반의 transformer decoder 아키텍처를 사용하여 경로 시퀀스 의존성을 모델링합니다. 이 아키텍처는 마스크드 어텐션(masked attention) 메커니즘을 통해 이전의 관찰과 행동만을 참조하도록 설계되어 있습니다. 학습 과정에서는 imitation learning을 통한 오프라인 사전 학습과 강화 학습(reinforcement learning)을 통한 온라인 미세 조정을 구분했습니다.

- **Performance Highlights**: Room-to-Room (R2R) 데이터셋을 이용한 평가에서 VLN-GPT는 기존의 복잡하고 계산 비용이 높은 transformer-encoder 모델을 능가하는 성능을 보여주었습니다. 본 연구는 VLN-GPT 모델을 통해 multi-modal decision-making을 위한 시퀀스 모델링 접근법을 개척하며, SOTA transformer-encoder 기반 접근법에 비해 더 나은 성능을 입증했습니다.



### VoCoT: Unleashing Visually Grounded Multi-Step Reasoning in Large Multi-Modal Models (https://arxiv.org/abs/2405.16919)
- **What's New**: 최근 대규모 다중 모달 모델(LMMs, Large Multi-Modal Models)이 다양한 작업에서 뛰어난 성능을 보였으나, 복잡한 작업에서는 한계가 있었다. 이를 해결하기 위해, 새로운 다중 단계 비주얼 기반 객체 중심 사고 체인(VoCoT, Visually grounded object-centric Chain-of-Thought reasoning) 프레임워크가 제안되었다. 이는 LMM의 추론 능력을 향상시키기 위해 여러 단계를 거쳐야 하는 사고 체인을 구축한다.

- **Technical Details**: VoCoT는 두 가지 주요 특징으로 구성된다: (1) 객체 중심의 추론 경로를 활용하여 모달 간 정보 격차를 줄이고, (2) 시각적으로 그라운드된 객체 개념을 사용하여 모달 간 일치된 정보를 제공한다. VoCoT는 객체의 텍스트 설명, 좌표 및 시각 표현을 포함하는 튜플을 사용하여 신뢰성 있는 정보경로를 구축한다. 또한, RefBind 메커니즘을 도입해 추가 계산 없이 객체의 시각 표현을 효율적으로 획득한다.

- **Performance Highlights**: 새롭게 개발된 모델 VolCano는 VoCoT를 도입함으로써 GPT-4V를 포함한 최신 모델(SOTA models)을 능가하는 성능을 보였다. VolCano는 70억 개의 파라미터와 336x336 입력 해상도를 가지고 있으며, 복잡한 추론 작업에서 뛰어난 능력을 보였다.



### Mixture of Modality Knowledge Experts for Robust Multi-modal Knowledge Graph Completion (https://arxiv.org/abs/2405.16869)
Comments:
          Work in progress. Code and data will be released at this https URL

- **What's New**: Multi-modal knowledge graph completion (MMKGC) 분야에서 새로운 프레임워크 'MoMoK (Mixture of Modality Knowledge experts)'를 제시했습니다. 이는 복잡한 관계적 컨텍스트에서 적응적인 멀티모달 임베딩을 학습하기 위해 설계되었습니다.

- **Technical Details**: MoMoK는 관계 유도 Modality Knowledge 전문가와의 협업을 통해, 복잡한 관계적 컨텍스트에서 다양한 멀티모달 정보의 예측을 통합하여 포괄적인 결정을 내립니다. 또한 전문가 간 상호 정보를 최소화하여 전문가들을 분리하는 기술을 사용합니다.

- **Performance Highlights**: MoMoK는 네 개의 공공 MMKG 벤치마크에서 뛰어난 성능을 입증했으며, 이는 기존의 MMKGC 모델들에 비해 복잡한 시나리오에서도 높은 성능을 보여줍니다.



### On Mesa-Optimization in Autoregressively Trained Transformers: Emergence and Capability (https://arxiv.org/abs/2405.16845)
Comments:
          37pages

- **What's New**: 최근 연구는 오토레그레시브(autoregressive) 학습된 트랜스포머(transformer)가 메사-옵티마이저(mesa-optimizer)를 학습하여 컨텍스트 안에서(inner context) 학습을 최적화한다고 주장했습니다. 이 연구는 트랜스포머가 실험적으로 이러한 메사-옵티마이저에 수렴하는지에 대한 불확실성을 해결하고자 합니다.

- **Technical Details**: 이 논문은 일레이어(linear one-layer) 선형 인과적(causal) 자기 주의(attention) 모델을 통해 비오차(non-convex) 학습 동역학을 조사합니다. $x_{t+1} = W x_t$로 정의된 AR(autoregressive) 프로세스에 의해 생성된 시퀀스를 사용해 모델을 오토레그레시브 학습하면서, 특정 데이터 분포 조건 하에 트랜스포머가 컨텍스트 안에서 일반적 최소 제곱(OLS; ordinary least squares) 문제를 최소화하는 한 걸음의 경사 하강법(gradient descent)을 수행하는 것을 증명합니다. 학습된 $\widehat{W}$을 다음 토큰 예측에 사용하여 메사-옵티마이저 가설을 검증합니다.

- **Performance Highlights**: 학습된 메사-옵티마이저의 한계를 탐구하였으며, 데이터의 모멘트(moment)에 관련된 더욱 강력한 가정이 분포를 복원하는 데 필수적인 조건임을 보여주었습니다. 또한, 첫 번째 데이터 조건을 넘어선 탐색적 분석을 통해 일반적으로 학습된 트랜스포머가 OLS 문제에 대해 순수한 경사 하강법을 수행하지 않을 것임을 증명했습니다. 마지막으로, 시뮬레이션 결과를 통해 이론적 결과를 검증했습니다.



### LLM-Based Cooperative Agents using Information Relevance and Plan Validation (https://arxiv.org/abs/2405.16751)
- **What's New**: 다중 에이전트 협력 문제를 해결하기 위해, 복잡한 부분 관찰 상태에서 분산된 에이전트들이 상호 작용하여 공동 목표를 달성하는 데 중점을 둔 새로운 연구를 제시합니다. 이를 통해 환경에 변화가 있을 때 에이전트들이 적응하고, 불필요한 통신 비용을 최소화하며, 비효율적인 정보를 효과적으로 관리할 수 있게 합니다. 주요 기여로는 REVECA(RElevance and Validation-Enhanced Cooperative Language Agent)라는 새로운 인지 아키텍처를 도입하여 효율적이고 강력한 에이전트 협력을 가능하게 합니다.

- **Technical Details**: REVECA는 GPT-3.5로 구동되며, 세 가지 주요 기능을 활용합니다: 1) 정보의 관련성 평가 및 계획 확증 (plan validation), 2) 공간 정보 통합을 통한 최적의 경로 구축, 3) 불필요한 더미 객체 관리를 통한 통신 비용 최소화. 이로써 동적이고 부분적으로 관찰 가능한 환경에서 에이전트 협력의 효율성과 견고성을 향상시킵니다.

- **Performance Highlights**: 다양한 실험을 통해 REVECA가 기존 접근방식, 특히 GPT-4.0 기반 접근방식보다 월등함을 시연했습니다. 사용자 연구에서도 REVECA의 신뢰할 수 있는 인간-AI 협력 잠재력을 강조했습니다. 이는 게임, XR 응용 프로그램, 교육 도구 및 인간형 로봇과 같은 다양한 분야에서 경제적, 상업적, 학문적 발전으로 이어질 수 있을 것으로 기대됩니다.



### Zamba: A Compact 7B SSM Hybrid Mod (https://arxiv.org/abs/2405.16712)
- **What's New**: Zamba는 7B SSM-transformer 하이브리드 모델로, 오픈 소스로 제공되는 모델들 중에서 경쟁력 있는 성능을 자랑합니다. 이 모델은 Mamba 백본과 단일 공유 어텐션 모듈을 결합하여 최소한의 파라미터 비용으로 어텐션의 이점을 얻습니다. Zamba는 두 가지 단계의 사전 훈련을 거칩니다: 첫 번째 단계는 공개 웹 데이터셋에서 훈련하고, 두 번째 단계는 고품질 지시 데이터와 합성 데이터셋을 사용하여 빠른 학습률 감소를 특징으로 합니다.

- **Technical Details**: Zamba는 transformer 모델 대신 Mamba 기반의 상태 공간 모델(SSM)로, 어텐션 연산 대신 선형 동적 시스템을 사용하여 메모리 비용을 크게 줄였습니다. 이는 Transformer 모델이 필요로 하는 메모리와 비교했을 때, 높은 수준의 성능을 유지하면서도 메모리 효율을 극대화합니다. Zamba의 독특한 아키텍처는 Mamba 백본과 글로벌 공유 자기 어텐션 레이어를 결합하여, Transformer 모델과 유사한 검색 및 인 텍스트 학습 기능을 유지합니다.

- **Performance Highlights**: Zamba는 1조개 토큰에 대해 훈련받았고, 7B Transformer 기반 모델과 동급의 성능을 자랑합니다. 특히 언어 평가 기준에서는 최첨단 7B 모델과 비슷한 수준을 유지하고 있습니다. 그러나 추론과 인 텍스트 학습 테스트에서는 다소 뒤쳐질 수 있는데, 이는 데이터 차이에 인한 것으로 보입니다. Zamba는 상용 가능한 최고 성능의 SSM 모델로, 작은 7B 모델 범위 내에서 가장 높은 성능을 보입니다.



### Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs (https://arxiv.org/abs/2405.16700)
Comments:
          Project page: this https URL. 37 Pages

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)을 텍스트 외에 이미지, 비디오, 오디오와 같은 다양한 멀티모달 입력에 노출시켜 그 내부 표현을 분석하고자 합니다. 이를 통해 LLMs가 멀티모달 입력을 일반화하는 방법에 대한 이해를 높이고 기술적 기여를 제공합니다.

- **Technical Details**: LLMs는 텍스트와 인지적 토큰(perceptual tokens)이 서로 다른 표현 공간에서 작동하지만, 유사한 가중치를 활성화합니다. 이는 암묵적 멀티모달 정렬(Implicit Multimodal Alignment, IMA)이라는 현상을 통해 이루어집니다. IMA는 모델의 아키텍처와 관련이 깊으며, 이는 LLMs가 멀티모달 입력에 대해 일반화할 수 있는 주요 이유 중 하나로 작용합니다. 또한, 인지적 토큰의 변화가 적으므로 FFN 계층 등을 건너뛰어 계산 비용을 줄이는 방법을 제안합니다.

- **Performance Highlights**: 연구 결과는 IMA 점수가 작업 성능에 긍정적인 상관관계가 있음을 보여주며, 이는 모델 평가와 선택에 중요한 지표로 사용될 수 있습니다. 반면, IMA 점수와 환각(hallucinations) 간에 음의 상관관계가 있어, 환각 문제는 내부 표현의 정렬 부족으로 인해 발생할 수 있음을 나타냅니다. 또한, LLMs의 가중치가 다양한 멀티모달 작업에 일반화할 수 있도록 하나의 서브네트워크만 유지하는 방법으로 모델을 압축했습니다.

- **Implications**: 이 연구는 LLMs의 아키텍처가 멀티모달 표현을 일반화할 수 있는 주요 요인임을 증명합니다. IMA 점수는 작업 성능 및 환각 여부를 예측하는 지표로 사용될 수 있으며, 인지적 토큰의 계산을 건너뛰어 효율적인 추론을 가능케 합니다. LLMs 모델을 하나의 서브네트워크로 압축함으로써 멀티모달 작업에 효율적으로 대응할 수 있습니다.



### A Systematic Review of Federated Generative Models (https://arxiv.org/abs/2405.16682)
Comments:
          24 Pages, 3 Figures, 5 Tables

- **What's New**: 최근 5년간의 연구를 종합하여, 분산 시스템에서 데이터 공유 없이 모델을 학습할 수 있는 연합 학습(Federated Learning, FL)과 생성 모델(Generative Models)의 연계에 대한 서베이 논문입니다. 이는 특히 FL과 생성 모델의 보안 문제와 최적 아키텍처 설계의 난제들을 다룹니다.

- **Technical Details**: 연합 학습은 여러 클라이언트가 자신들의 데이터를 중앙 서버에 공유하지 않고 모델 업데이트만을 공유함으로써 데이터의 프라이버시를 유지하게 합니다. 세 가지 주요 범주로 구분된 연구들은 분산형 생성 모델, FL 모델에 대한 공격 및 방어, 데이터 이질성과 비IID 문제를 해결하기 위한 생성 모델의 응용을 포함합니다. 사용된 주요 생성 모델로는 GANs, VAEs, 그리고 최근에 주목받는 Diffusion Models 등이 있습니다.

- **Performance Highlights**: 연합형 GANs(Federated GANs)는 임상 응용 분야에서 주목을 받으며, 특히 차별적 프라이버시(Differential Privacy, DP) 기준을 만족하여 강력한 보안을 제공합니다. 최근에는 Diffusion 기반 연합 모델이 통신 비용과 수렴 측면에서 기존 GAN 기반 FL 모델을 능가하는 성능을 보여주었습니다. 그러나 표 형식 데이터 기반 모델과 GAN 비기반 FL 모델에서는 여전히 보안과 무결성 문제가 해결되지 않은 상태입니다. 또한, One-shot FL, 사전 학습된 Diffusion Models, LLM 기반 생성 FL 등은 현재 인기를 끌고 있는 연구 주제입니다.



### Crossmodal ASR Error Correction with Discrete Speech Units (https://arxiv.org/abs/2405.16677)
- **What's New**: 이번 연구에서는 ASR Error Correction(AEC)의 새로운 접근 방식을 제안했습니다. 특히, Low-Resource Out-of-Domain(LROOD) 문제를 다루기 위해 한정된 데이터에서 crossmodal AEC를 탐구했습니다. 우리의 연구는 ASR 도메인의 불일치 현상을 발견하고, 이런 데이터를 위한 적절한 훈련 방식을 제안했습니다. 추가로, 단어 임베딩을 정렬하고 개선하기 위해 이산 음성 유닛(Discrete Speech Units, DSUs)을 통합했습니다.

- **Technical Details**: 연구는 전처리(pre-training, PT) 및 미세 조정(fine-tuning, FT) 전략을 비교하여 LROOD 데이터에서 AEC 성능을 평가했습니다. 우리는 Wav2Vec 2.0, Conformer 모델, Whisper 모델을 사용해 다양한 ASR 시스템의 오류 유형을 분석했습니다. 또한, PT 단계에서는 오디오 소스를 사용하지 않고, FT 단계에서만 DSUs를 통합하여 크로스모달 훈련의 효율성을 높였습니다.

- **Performance Highlights**: 우리의 접근 방식은 여러 코퍼스 및 평가 지표(WER, BLEU, GLEU)를 통해 LROOD 데이터에서 AEC의 타당성과 효과를 입증했으며, 대규모 데이터에서도 그 일반성과 우수성을 확인했습니다. 최종적으로, 음성 감정 인식을 통해 ASR 오류에 강인한 성과를 보였습니다.



### Low-resourced Languages and Online Knowledge Repositories: A Need-Finding Study (https://arxiv.org/abs/2405.16669)
Comments:
          In Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI 2024)

- **What's New**: 이 논문은 Wikipedia와 같은 온라인 지식 저장소(OKRs)에서 저자원 언어로 기여하는 데 따른 도전과제를 탐구합니다. 특히 에티오피아의 세 가지 언어인 Afan Oromo, Amharic, Tigrinya에 초점을 맞췄습니다. 저자들은 Wikipedia 포럼 토론의 주제 분석과 14명의 초보 기여자들과의 맥락 조사 연구를 통해 저자원 언어 기여자들이 직면하는 여러 문제를 발견했습니다.

- **Technical Details**: 연구는 두 가지 주요 방법론을 사용했습니다: (1) Wikipedia의 경험 있는 기여자들의 포럼 토론을 수집하고 분석하는 것과, (2) Afan Oromo, Amharic, Tigrinya 언어를 사용하는 초보 Wikipedia 기여자들과의 맥락 조사 연구를 수행하는 것입니다. 주요 발견 사항에는 필요한 자료를 찾고, 번역 시스템 및 철자검사와 같은 언어 기술 지원의 실패로 인한 시간 낭비가 포함되었습니다.

- **Performance Highlights**: 연구 결과, 저자원 언어 기여자들은 기술적 장애뿐만 아니라 학술 자료와 재정 자원의 부족에도 영향을 받는 것으로 나타났습니다. 이러한 분석을 통해 저자들은 포괄적인 언어 기술을 만들고자 하는 연구원과 디자이너를 위한 디자인 기회를 제안하였습니다.



### Conjunctive categorial grammars and Lambek grammars with additives (https://arxiv.org/abs/2405.16662)
Comments:
          This article is an extended version of the conference presentation "Conjunctive categorial grammars" at the Mathematics of Language 2017 meeting (London, UK, July 13-14, 2017; proceedings published in ACL Anthology, W17-3414)

- **What's New**: 이 논문은 기본 범주 문법 (categorial grammars)에 결합(conjunction) 연산을 추가하여 새로운 범주 문법 패밀리를 제안합니다. 이 확장된 범주 문법의 표현력은 결합 문법(conjunctive grammars, 결합성을 갖춘 문맥 자유 문법)과 동일하다는 것을 증명하였습니다. 또한, 결합 연산과 분리(disjunction) 연산을 포함하는 Lambek 계산(Lambek calculus) 내에서 결합 연산을 가진 범주 문법이 자연스럽게 포함될 수 있음을 보여주었습니다. 이는 특정 NP-완전 집합을 Lambek 계산 내에서 정의할 수 있음을 의미하며, 공백 문자열과 관련된 미묘한 문제를 처리하는 방법도 제시합니다.

- **Technical Details**: 이 논문은 두 가지 문법 모델 간의 연결을 확립합니다: 전통 문법을 기반으로 한 문맥 자유 문법(context-free grammars)과 Ajdukiewicz 및 Bar-Hillel이 개발한 범주 문법(categorial grammars)입니다. 범주 문법에 결합 연산을 추가한 새로운 모델은 기존의 Lambek 계산과의 관계를 통해 정의되었습니다. 또한, 이 논문은 논리적 도출을 통해 결합 문법과 범주 문법의 동등성을 입증하였습니다. 이를 위해 Okhotin과 Reitwießner의 정규 형식(normal form) 이론을 사용하였습니다.

- **Performance Highlights**: 결합 연산을 포함한 새로운 범주 문법 모델은 모든 결합 문법을 시뮬레이션할 수 있는 반면, 결합 문법 또한 이 새로운 모델로 시뮬레이션할 수 있음을 증명하여 두 모델의 동등성을 입증하였습니다. 따라서, 이 확장된 범주 문법은 실질적인 구문 분석 알고리즘을 유지하면서도 기존 문맥 자유 문법보다 더 높은 표현력을 지니고 있습니다.



### A Survey of Multimodal Large Language Model from A Data-centric Perspectiv (https://arxiv.org/abs/2405.16640)
- **What's New**: 최근 인간은 다양한 감각을 통해 세상을 인지하듯, 멀티모달 대형 언어 모델(MLLMs)은 텍스트, 비전(vision), 오디오, 비디오, 3D 환경 등 여러 모달리티를 통합해 전통적인 대형 언어 모델(LLMs)의 능력을 향상시킵니다. 이 논문은 데이터 중심 관점에서 MLLMs의 문헌을 종합적으로 검토하며, 데이터 준비 및 모델 적응 과정, 데이터셋 평가 방법 및 벤치마크에 대해 분석합니다.

- **Technical Details**: 최근 재현된 LLMs와 MLLMs는 GPT-4, Flamingo, BLIP2, X-InstructBLIP 등의 모델을 포함하며, 다양한 모달리티 정보를 통합하여 뛰어난 이해와 생성 능력을 보여줍니다. MLLMs는 시각 인식, 비디오 이해, 음성 인식, 3D 이해와 같은 전통적인 멀티모달 작업뿐만 아니라, 질문 응답, 멀티 대화, 논리적 추론 등의 텍스트 중심 작업에서도 뛰어난 성과를 보입니다. 본 연구는 데이터 수집, 선택, 관리 방법과 데이터가 모델 성능에 미치는 영향 및 데이터 평가 벤치마크에 대한 논의를 포함합니다.

- **Performance Highlights**: 기존 MLLMs는 보통 모델 아키텍처 수정에 중점을 두었으나, 데이터의 양과 질이 모델의 성공에 크게 영향을 미칩니다. 예를 들어, 데이터 양이 커지면 모델의 성능이 향상될 수 있고, 신중하게 준비된 데이터셋은 작은 모델도 큰 모델과 비슷한 성능을 낼 수 있음이 연구 결과에서 드러났습니다.



### Cocktail: A Comprehensive Information Retrieval Benchmark with LLM-Generated Documents Integration (https://arxiv.org/abs/2405.16546)
Comments:
          Accepted by Findings of ACL 2024; Datasets Link: this https URL

- **What's New**: 최근 인터넷 상에서 대형 언어 모델(LLMs)이 생성한 콘텐츠의 급증으로 정보 검색(IR) 시스템의 데이터가 인간이 작성한 콘텐츠와 AI가 생성한 콘텐츠가 공존하는 양상을 보이게 되었습니다. 이러한 혼합된 데이터 환경에서 IR 모델의 성능을 평가하기 위한 종합 벤치마크인 'Cocktail'이 소개되었습니다. 이 벤치마크는 다양한 텍스트 검색 작업과 도메인에 걸쳐 인간 작성 및 LLM 생성된 콘텐츠가 혼합된 16개 데이터셋을 포함합니다.

- **Technical Details**: Cocktail 벤치마크는 MS MARCO, TREC, BEIR와 같은 널리 사용되는 공개 데이터셋을 기반으로 하며, Llama2를 사용해 텍스트를 재작성하여 의미적 동등성을 유지하면서도 LLM 생성 콘텐츠를 도입했습니다. 또한 최신 이벤트에서 유래한 질의로 구성된 최신 데이터셋 NQ-UTD를 추가하여 편향을 방지하려 합니다. 1,000개 이상의 실험을 통해 최신 상태의 검색 모델을 평가하고, 성능과 소스 편향 간의 균형을 밝혀냈습니다.

- **Performance Highlights**: 실험 결과, 최신 신경 검색 모델들이 랭킹 성능과 소스 편향 간의 명확한 트레이드오프(Trade-off)를 보였습니다. 대부분의 모델이 LLM 생성 콘텐츠를 편향적으로 랭킹하는 경향이 파악되었습니다. 이는 향후 IR 시스템 설계 시 성능 개선과 편향 완화 간의 균형을 고려해야 할 필요성을 강조합니다. Cocktail은 이러한 연구를 촉진할 수 있는 기초 자료로 활용될 것입니다.



### LoQT: Low Rank Adapters for Quantized Training (https://arxiv.org/abs/2405.16528)
- **What's New**: 대규모 신경망(neural networks) 훈련에 필요한 높은 자원을 문제로 삼아, LoQT라는 새로운 효율적인 양자화된 모델(quantized models) 훈련 방법을 제안했습니다. LoQT는 저순위(trainable weight matrices)를 주기적으로 양자화된 고순위(full-rank) 가중치 행렬에 병합하여, 소비자급 하드웨어에서도 최대 7B 매개변수 모델을 훈련할 수 있도록 합니다.

- **Technical Details**: LoQT는 기울기 기반 텐서 분해(gradient-based tensor factorization)를 통해 저순위 가중치 행렬을 초기화하고, 이를 정기적으로 양자화된 고순위 가중치 행렬에 병합합니다. 이 방법은 모델의 사전 훈련(pretraining)과 미세 조정(fine-tuning)에 모두 적합하며, NF4 양자화 방식을 사용합니다. 또한, 소비자급 24GB GPU에서도 모델을 효율적으로 훈련할 수 있습니다.

- **Performance Highlights**: LoQT는 7B 매개변수 모델을 소비자급 하드웨어에서 효율적으로 훈련할 수 있으며, 동일한 하드웨어로 계층별 기울기 업데이트(per-layer gradient updates)를 통해 13B 매개변수 모델도 훈련할 수 있음을 보여줍니다. 또한, 이전 방식들보다 메모리 사용을 크게 줄이면서도 경쟁력 있는 성능을 입증했습니다.



### Meta-Task Planning for Language Agents (https://arxiv.org/abs/2405.16510)
- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Model, LLM) 기반의 다중 에이전트 시스템을 위한 새로운 메타 태스크 플래닝(Meta-Task Planning, MTP) 방법론을 소개했습니다. MTP는 제로-샷 방식으로 복잡한 작업을 하위 작업, 즉 메타 태스크로 분해하여 계획을 세우는 방식입니다. 이 방법론은 협력적인 LLM 다중 에이전트 시스템에서 작업 계획의 효율성을 크게 향상시키는 것이 목표입니다.

- **Technical Details**: MTP는 매니저 에이전트가 작업을 메타 태스크로 분해하고, 각각의 메타 태스크를 수행하기 위해 실행 가능한 액션 시퀀스로 맵핑하는 구조를 가지고 있습니다. 매니저 에이전트는 작업 수준의 계획을 수립하며, 각 메타 태스크는 이그젝터 에이전트들이 담당합니다. 이그젝터 에이전트들은 ReAct와 같은 기존의 플래닝 기술을 활용하여 메타 태스크를 실행합니다. 또한, MTP는 제약 조건을 로컬 및 글로벌로 구분하며, 성공률과 안정성을 높이기 위해 슈퍼바이저 에이전트와 딜리버러 에이전트를 추가로 활용합니다.

- **Performance Highlights**: 실험 결과, MTP는 TravelPlanner 벤치마크에서 약 40%의 성공률을 기록하여 기존 최첨단(SOTA) 기준(2.92%)을 크게 상회했습니다. 또한, API-Bank에서도 LLM_{api}-4 with ReAct를 약 14% 초과하여 높은 성능을 보였습니다. 이는 LLM과 다중 에이전트 시스템의 통합이 매우 유망한 가능성을 갖고 있음을 시사합니다.



### M$^3$CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Though (https://arxiv.org/abs/2405.16473)
Comments:
          Accepted at ACL2024 Main Conference

- **What's New**: 최근 주목받고 있는 다중 모달 체인 오브 사상(Multi-modal Chain-of-Thought, MCoT) 분야에서, 본 논문은 기존 MCoT 벤치마크의 한계를 극복하는 새로운 벤치마크 M$^3$CoT를 소개합니다. M$^3$CoT는 다중 도메인, 다중 단계, 다중 모달의 복합적 사고 체인을 평가하기 위한 최초의 벤치마크로, 더욱 복잡한 실제 시나리오를 반영한 평가를 가능하게 합니다.

- **Technical Details**: 기존 벤치마크의 주요 문제점인 시각 모달 추론 부재, 단일 단계의 시각 모달 추론, 그리고 도메인 부족 문제를 해결하기 위해, M$^3$CoT는 이미지를 필요로 하지 않는 샘플을 제거하고, 다중 단계를 요하는 다중 모달 샘플을 수작업으로 주석 작업을 통해 선정했습니다. 이를 통해 물리학, 수학 및 상식 도메인에서의 다중 단계 MCoT 데이터를 생성하는 LLM-guided augmentation을 탐구했습니다. 또한, Vision Large Language Models(VLLMs)에서 다양한 MCoT 접근법을 평가하여 VLLM이 10억 파라미터 이상에서 CoT가 발생하는 현상을 발견했습니다.

- **Performance Highlights**: VLLM은 다중 단계 MCoT에서 기존의 단순한 프롬프트 전략이나 도구 사용보다 강력한 미세 조정(fine-tuning) 방법이 더 효과적임을 보였습니다. 기존 벤치마크 대비 인간 성능과의 큰 격차를 확인하며 M$^3$CoT가 현재의 MCoT 성능을 과대 평가하고 있음을 강조했습니다. M$^3$CoT는 다양한 도메인에서의 복잡한 다중 단계 추론을 통해 현재의 MCoT 모델이 여전히 인간 성능에 비해 많은 개선이 필요함을 보여줍니다.



### Development of an open education resources (OER) system: a comparative analysis and implementation approach (https://arxiv.org/abs/2405.16442)
- **What's New**: 여러 기관이 협력하여 비영리 교육 목적을 위한 새로운 웹 기반 열린 교육 자원(Open Education Resources, OER) 시스템을 개발하고 있습니다. 이 이니셔티브는 다양한 사용자 프로필을 최적화한 사용자 경험을 제공하기 위해 세심하게 설계된 연구를 바탕으로 합니다.

- **Technical Details**: 이 프로젝트는 오픈소스 도구, 프레임워크, 기술을 적극 활용하여 진행됩니다. 주요 활동으로는 상위 5개의 오픈소스 학습 관리 시스템(LMS)에 대한 비교 분석이 포함되며, 이는 개발 프로세스에 중요한 통찰을 제공합니다.

- **Performance Highlights**: 프로젝트의 주 목표는 비영리 사용자를 위한 교육 자원을 공유할 수 있는 웹 기반 시스템을 구축하는 것입니다. 이를 위해 정보 통신 기술을 적극 활용하며, 연구팀과 개발팀 두 주요 팀이 협력하여 사용자 중심의 OER 시스템을 확립하는 것을 목표로 합니다.



### The Importance of Directional Feedback for LLM-based Optimizers (https://arxiv.org/abs/2405.16434)
Comments:
          Presented at Foundation Models for Decision Making at NeurIPS 2023

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 텍스트 공간에서 최대화 문제를 해결하기 위한 상호작용 최적화기로 사용하는 가능성을 연구했습니다. 특히, 방향성 피드백(directional feedback)이 주어졌을 때 LLM이 최적화에 매우 유능하다는 점을 발견했습니다. 이에 기반하여, 우리는 역사적인 최적화 추적 기록에서 방향성 피드백을 합성하여 반복을 통해 안정적이고 효율적인 성능 향상을 이루는 새로운 LLM 기반 최적화기를 설계했습니다.

- **Technical Details**: 연구에서는 자연어 피드백을 방향성(directional)과 비방향성(non-directional)으로 분류하며, 방향성 피드백이 자연어 공간에서의 1차 피드백의 일반화 버전임을 강조했습니다. 다양한 상호작용 의사결정 도메인에서 방향성 피드백의 존재 및 이용 가능성은 LLM 기반 최적화에서 매우 중요하며, 이를 통해 LLM이 수학적 함수부터 시를 최적화하는 등의 다양한 최적화 문제를 해결할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, LLM 기반 최적화기는 기존 기법과 비교하여 최적화 문제 해결 시 더 안정적이고 효율적임을 입증했습니다. 수학적 함수 최대화와 시 작성 프롬프트 최적화 등에서 뛰어난 성능을 보였으며, 방향성 피드백을 체계적으로 사용한 결과 이전에는 해결할 수 없던 문제들을 해결할 수 있음을 확인했습니다.



### Augmented Risk Prediction for the Onset of Alzheimer's Disease from Electronic Health Records with Large Language Models (https://arxiv.org/abs/2405.16413)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 소수 샷 추론(few-shot inference) 능력을 활용하여 알츠하이머 병(AD) 및 관련 치매(ADRD) 위험 예측을 개선하는 새로운 파이프라인을 제안합니다. 학습된 LLM을 사용하여 기존의 지도 학습법(SLs)이 다루기 어려운 복잡한 사례들을 예측하는 협력적 접근 방식을 소개합니다. 이를 통해 EHR 데이터로부터 더 효과적인 예측이 가능해졌습니다.

- **Technical Details**: 이 파이프라인은 오리건 건강 및 과학 대학교(OHSU) 병원의 실제 EHR 데이터 웨어하우스를 이용하여 평가되었습니다. 데이터는 250만 명이 넘는 환자와 2천만 건 이상의 환자 만남을 포함합니다. SL과 LLM을 결합하기 위한 신뢰 기반 의사 결정 메커니즘을 도입하여, SL의 명확한 사례에 대한 힘과 LLM의 복잡한 사례에 대한 추론 능력을 최대한 활용할 수 있도록 설계되었습니다. 또한 자연어 처리 (NLP)에서 사용되는 LLM 추론에서 발생할 수 있는 노이즈를 줄이기 위해 ICL 데모노이즈 전략을 포함했습니다.

- **Performance Highlights**: 이 방식은 기존의 SL 모델과 LLM을 독립적으로 사용하는 접근 방법보다 예측 성능이 크게 향상되었습니다. 특히, OHSU 건강 시스템의 실세계 데이터셋을 사용한 검증 결과, 제안한 방법이 더 우수한 ADRD 예측 성능을 보였습니다. 또한, 다양한 LLM 크기와 다양한 의료 데이터셋에 대해 미세 조정된 모델로 실험을 수행하였습니다. 그 결과, 더 큰 모델 크기나 의료 데이터에 대해 미세 조정하는 것이 일관되게 위험 예측 성능을 개선하지는 못한다는 것을 발견했습니다.



### Tensor Attention Training: Provably Efficient Learning of Higher-order Transformers (https://arxiv.org/abs/2405.16411)
- **What's New**: 이번 연구에서는 다중 모달리티에서 고차 상관관계를 포착할 수 있는 'Tensor Attention'이 제안되었습니다. 기존의 매트릭스 어텐션(matrix attention)의 표현 한계를 극복할 수 있지만, $
^3$의 시간 복잡도가 있어 실용적인 구현에 문제가 있었습니다. 연구진은 텐서 어텐션의 역방향 그래디언트 계산이 입력 시퀀스 길이에 거의 선형 $n^{1+o(1)}$ 시간 복잡도로 수행될 수 있음을 증명했습니다. 이는 텐서 어텐션의 효율적인 고차 변환기 훈련을 가능하게 할 수 있습니다.

- **Technical Details**: 연구진은 백워드 그래디언트(backward gradient) 계산을 위한 폐쇄형 솔루션을 제공하고, 다항근사법(polynomial approximation methods)과 텐서 대수적 기법(tensor algebraic tricks)을 활용한 빠른 계산 방법을 제안했습니다. 이러한 방법을 통해 텐서 어텐션의 시간 복잡도가 고차(n^3)에서 n^{1+o(1)}로 줄어들 수 있음을 보였습니다. 또한, 입실론(epsilon)을 이용해 조금만 약화시켜도 그래디언트 문제가 진정한 서브큐빅 시간에서는 해결 불가능함을 보여줌으로써 가정의 필요성과 타당성을 입증했습니다.

- **Performance Highlights**: 이 연구 결과는 텐서 어텐션 아키텍처의 효율적인 고차 훈련 가능성을 수립했으며, 이를 통해 실제 응용에서도 텐서 어텐션을 활용할 수 있는 길을 열었습니다. 특히, 블록 순차적 접근법을 통해 고차 변환기의 훈련 속도를 크게 향상시킬 수 있습니다.



### SpinQuant -- LLM quantization with learned rotations (https://arxiv.org/abs/2405.16406)
- **What's New**: SpinQuant은 대규모 언어 모델(LLM)의 메모리 사용량, 지연 시간, 전력 소모를 크게 줄이기 위해 4비트 양자화(post-training quantization)를 사용하는 혁신적인 접근법입니다. SpinQuant는 회전 행렬을 최적화하여 더 나은 양자화 성능을 사용할 수 있게 합니다.

- **Technical Details**: SpinQuant는 옵티마이제이션을 위해 Cayley optimization 방법을 작은 밸리데이션 셋을 이용해 회전 행렬(rotation matrices)을 최적화(learn)합니다. 이 기술은 활성화 행렬(activation matrices) 또는 가중치 행렬(weight matrices)을 회전하여 이상값(outliers)을 제거하고 양자화 효율성을 높입니다.

- **Performance Highlights**: SpinQuant는 4비트 양자화의 한계에도 불구하고, LLaMA-2 7B 모델에서 zero-shot reasoning 작업의 정확도 차이를 2.9 포인트로 줄이며 풀 프리시전(full-precision)에 거의 근접한 성능을 보여줍니다. 이는 LLM-QAT에 비해 19.1 포인트, SmoothQuant 보다 25.0 포인트 더 높은 성능입니다. 또한, SpinQuant는 QuaRot를 능가하며, 특히 양자화가 어려운 LLaMA-2 7B/LLaMA-3 8B 모델에서 풀 프리시전 대비 차이를 각각 30.2%/34.1% 감소시킵니다.



### AutoManual: Generating Instruction Manuals by LLM Agents via Interactive Environmental Learning (https://arxiv.org/abs/2405.16247)
- **What's New**: AutoManual이라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 LLM 기반의 에이전트가 상호작용을 통해 환경을 이해하고 새로운 환경에 적응할 수 있게 합니다. 기존의 에이전트가 특정 도메인에서 작업을 해결하기 위해 정교한 설계와 전문가의 프롬프트가 필요했던 반면, AutoManual은 자체 생성된 매뉴얼을 통해 적응성을 향상시킵니다.

- **Technical Details**: AutoManual은 두 가지 에이전트로 구성됩니다: Planner는 현재 규칙을 기반으로 실행 가능한 계획을 작성하고, Builder는 규칙을 온라인으로 업데이트하여 환경에 적응합니다. Builder는 'case-conditioned prompting' 전략을 사용하여 규칙 관리를 하며, Formulator 에이전트는 다양한 규칙을 종합하여 Markdown 형식의 매뉴얼을 작성합니다.

- **Performance Highlights**: 단 한 번의 간단한 데모만으로도 AutoManual은 ALFWorld 벤치마크 작업에서 GPT-4-turbo는 97.4%, GPT-3.5-turbo는 86.2%의 높은 성공률을 달성했습니다. 이 프레임워크는 추후에 공개될 소스 코드와 함께 탁월한 성능을 보여줍니다.



### GeneAgent: Self-verification Language Agent for Gene Set Knowledge Discovery using Domain Databases (https://arxiv.org/abs/2405.16205)
Comments:
          30 pages with 10 figures and/or tables

- **What's New**: 새로운 연구는 GeneAgent라는 최초의 언어 에이전트(language agent)를 소개합니다. 이 에이전트는 자기 검증 능력(self-verification capability)을 갖추고 있어 정확성을 높이고 환각(hallucinations) 발생을 줄일 수 있습니다. GeneAgent는 다양한 생물학적 데이터베이스와 자율적으로 상호작용하며, 해당 도메인의 지식을 활용합니다.

- **Technical Details**: GeneAgent는 GPT-4와 같은 기존의 대형 언어 모델(LLM)의 한계를 극복하기 위해 설계되었습니다. 자기 검증 모듈을 통해 정보를 확인하고, 불확실한 데이터를 걸러내는 방식으로 정확도를 높입니다. 또한, 다양한 출처의 1,106개의 유전자 세트(gene sets)를 벤치마킹하여 높은 성능을 입증하였습니다.

- **Performance Highlights**: GeneAgent는 GPT-4를 표준으로 했을 때 일관되게 더 높은 성능을 보였습니다. 세부적인 수작업 검토 결과에 따르면, GeneAgent의 자기 검증 모듈은 환각 발생을 최소화하고 보다 신뢰할 수 있는 분석 결과를 생성하는 데 효과적이었습니다. 또한, 실제 적용 사례에서 전문가 평가를 통해 GeneAgent가 유전자 기능에 대한 새로운 통찰을 제공하고 지식 발견을 촉진함을 확인했습니다.



### C3LLM: Conditional Multimodal Content Generation Using Large Language Models (https://arxiv.org/abs/2405.16136)
- **What's New**: C3LLM(Conditioned-on-Three-Modalities Large Language Models)은 비디오-오디오(video-to-audio), 오디오-텍스트(audio-to-text), 텍스트-오디오(text-to-audio) 세 가지 작업을 결합한 새로운 프레임워크입니다. 이 모델은 LLM(Large Language Model) 구조를 활용하여 서로 다른 모달리티를 정렬하고, 주어진 조건을 기반으로 합성하여 다양한 모달리티 생성을 가능하게 합니다.

- **Technical Details**: C3LLM은 사전 학습된 오디오 코드북을 사용하여 오디오 생성 작업에 계층적 구조를 적용합니다. LLM을 훈련시켜 주어진 조건에서 오디오 의미 토큰을 생성하고, 비자동회귀(non-autoregressive) 트랜스포머를 사용해 더 높은 충실도를 보장합니다. 또한, LLM의 이산(discrete) 표현 방식을 활용하여 오디오 생성을 수행하고, 오디오 의미를 오디오 토큰으로 압축합니다. 그렇게 함으로써 LLM이 오디오를 '음향 어휘'로 다룰 수 있도록 합니다.

- **Performance Highlights**: C3LLM은 다양한 자동평가 메트릭스를 통해 개선된 결과를 보여주며, 이전 방법들에 비해 더 나은 의미적 정렬을 제공합니다. 특히, 비디오-오디오, 오디오-텍스트, 텍스트-오디오 작업을 하나의 통합된 모델로 결합하여 보다 다재다능한 성능을 보여줍니다.



### How Well Do Deep Learning Models Capture Human Concepts? The Case of the Typicality Effec (https://arxiv.org/abs/2405.16128)
Comments:
          To appear at CogSci 2024

- **What's New**: 본 연구는 딥러닝 모델이 학습한 개념적 표현이 인간의 개념적 표현과 얼마나 잘 일치하는지를 평가하며, 특히 '일반성 효과(typicality effect)'에 초점을 맞추고 있습니다. 이 효과는 사람들이 어떤 범주의 일부 예시(예: 참새)를 다른 예시(예: 펭귄)보다 더 일반적이라고 인식하는 현상을 의미합니다. 기존 연구는 주로 단일 모달리티 모델에서 제한된 개념만을 조사했으나, 본 연구는 더 광범위한 언어 모델(N=8)과 비전 모델(N=10) 아키텍처를 평가합니다. 또한 언어 및 비전 모델의 결합과 멀티모달(CLIP 기반) 모델이 단일 모달리티 모델보다 인간의 일반성 판단과 더 잘 일치하는지 평가합니다.

- **Technical Details**: 본 연구는 인간의 일반성 효과 데이터를 이용하여 다양한 언어 및 비전 모델의 개념적 정렬을 평가합니다. 언어 모델은 word2vec, GloVe, RoBERTa-large, XLNet-base, MiniLM, MPNet, T5-large, GPT와 같은 수많은 아키텍처를 포함하며, 비전 모델은 다양한 CNN 및 Transformer 모델을 포함합니다. 새로운 '자연주의적' 이미지 세트를 개발하여 비전 모델의 개념적 정렬을 테스트합니다. 이를 위해 참가자들로부터 범주별 예시를 제공받아 그 일반성을 정의하고, 이 데이터를 통해 모델의 성능을 평가합니다.

- **Performance Highlights**: 첫째, 언어 모델이 비전 모델보다 인간의 일반성 판단과 더 잘 일치합니다. 둘째, 언어 모델과 비전 모델의 조합(AlexNet + MiniLM)이 최고의 언어 모델(MiniLM)이나 비전 모델(ViT-Huge) 단독으로 보다 인간의 일반성 판단을 더 잘 예측합니다. 셋째, 멀티모달 모델(CLIP ViT)을 사용하면 인간의 일반성 판단을 설명하는 데 유망한 결과를 보여줍니다. 이러한 결과는 ML 모델의 개념적 표현이 인간과 어떻게 일치할 수 있는지에 대한 최첨단 연구를 진전시킵니다.



### Prompt Optimization with EASE? Efficient Ordering-aware Automated Selection of Exemplars (https://arxiv.org/abs/2405.16122)
Comments:
          23 pages, 1 figure, 23 tables

- **What's New**: 본 논문에서는 EASE (Efficient Ordering-aware Automated Selection of Exemplars)라는 새로운 알고리즘을 제안합니다. EASE는 대형 언어 모델(LLM)의 주변 학습 성능을 최대화하기 위해 예시(exemplars)의 자동 선택 방법을 제안합니다. 주어진 임베딩을 사용하여 순서가 있는 예시 셋을 최적화하며, 테스트 시간의 계산 오버헤드를 제거하고 프라이버시 문제를 줄입니다.

- **Technical Details**: EASE는 사전에 훈련된 언어 모델의 숨겨진 임베딩(hidden embedding)을 활용하여 예시 셋을 표현하고, 신경 밴딧 알고리즘(neural bandit algorithm)을 사용하여 예시 셋의 순서를 고려하면서 최적화합니다. 이를 통해 모든 테스트 쿼리에 대해 효율적으로 성능이 좋은 예시 셋을 찾을 수 있으며, 추가적인 테스트-시간 계산이 필요 없습니다. 또한, EASE는 예시 셋 최적화뿐만 아니라 프롬프트의 명령어(instruction)까지 최적화할 수 있습니다.

- **Performance Highlights**: EASE는 광범위한 실험 평가를 통해 기존의 메서드를 능가하는 성능을 보여주었습니다. 특히, LLM이 특정 작업에 대해 지식이 적을 때 예시 선택이 더 중요하다는 흥미로운 통찰을 밝혀냈습니다. 다양한 벤치마크 과제에서 이전 기준선보다 일관되게 우수한 성능을 보여주었으며, 예시 선택의 중요성을 강조하는 새로운 실험에서도 우월한 성능을 입증했습니다.



### Theoretical Analysis of Weak-to-Strong Generalization (https://arxiv.org/abs/2405.16043)
Comments:
          36 pages, 3 figures

- **What's New**: 강력한 학생 모델이 약한 교사 모델로부터 배울 수 있는 새로운 이론적 분석을 제시합니다. 이 논문은 약한 교사의 오류를 수정하고 교사의 예측이 확실하지 않은 예제에 대해 일반화할 수 있는 학생 모델이 어떻게 동작하는지 설명합니다. 기존의 약한 지도학습 이론은 이러한 현상들을 설명하지 못했지만, 저자들은 새로운 확장 조건을 통해 이러한 효과들을 설명하는 이론적 경계를 제시합니다.

- **Technical Details**: 이 논문에서는 확장 특성에 기반한 데이터 분포와 학생 모델의 가설 클래스(hypothesis class)를 사용하여 새로운 경계(bound)를 제시합니다. 이 경계는 약한 교사의 오류를 수정(pseudo label correction)하고, 지도되지 않은 영역에서도 일반화(coverage expansion)할 수 있게 합니다. 저자들은 실험을 통해 이 확장 특성을 실제 데이터에서도 확인할 수 있다고 주장합니다.

- **Performance Highlights**: 실험 결과, 강력한 학생 모델이 약한 교사 모델의 오류를 포함한 원본 데이터보다 더 우수한 성능을 보였습니다. 또한, 교사 모델이 레이블을 제공하지 않은 데이터 포인트에서도 높은 성능을 나타냈습니다. 이러한 결과는 약한 지도 학습의 성공이 데이터 분포의 자연스러운 확장 조건에 기인함을 시사합니다.



### Enhancing Visual-Language Modality Alignment in Large Vision Language Models via Self-Improvemen (https://arxiv.org/abs/2405.15973)
Comments:
          15 pages, 8 figures

- **What's New**: SIMA는 외부 모델이나 데이터 없이 시각-언어 모달리티 정렬(visual and language modality alignment)을 개선하는 프레임워크입니다. 기존의 방법들은 외부 자원에 의존했지만, SIMA는 자가 개선(self-improvement)을 통해 이 과정을 극복합니다.

- **Technical Details**: SIMA는 기존의 vision instruction tuning 데이터셋에서 프롬프트를 활용해 응답을 자체 생성하고, in-context self-critic 메커니즘을 사용해 선호 튜닝(preference tuning)을 위한 응답 쌍을 선택합니다. 특히, in-context self-critic 과정에서 세 가지 시각 기준(vision metrics)을 도입해, 이미지 이해를 향상시키는 응답을 선택할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SIMA는 14개의 환각 및 종합 벤치마크(hallucination and comprehensive benchmarks)에서 모두 모델 성능을 향상시키며, 기존 방법들을 능가하는 모달리티 정렬 성능을 보여줍니다.



### Transformers represent belief state geometry in their residual stream (https://arxiv.org/abs/2405.15943)
- **What's New**: 새로운 연구는 대규모 언어 모델(large language models)을 훈련시킬 때 내부적으로 어떤 계산 구조가 생성되는지를 탐구합니다. 본 연구는 특히 잔여 스트림(residual stream)에서의 신념 상태(belief states)의 선형적 표현을 다루며, 이는 모델이 단순히 다음 토큰 예측을 넘어 데이터 생성 과정의 숨겨진 상태에 대한 신념을 업데이트하는 메타 다이내믹스를 학습한다는 것을 보여줍니다.

- **Technical Details**: 본 연구는 최적 예측 이론(optimal prediction)을 바탕으로 트랜스포머 모델(transformer models)의 내부 활성화 구조의 기하학을 설명하는 이론적 프레임워크를 제시합니다. 이를 테스트하기 위해 트랜스포머 모델을 숨겨진 구조를 가진 데이터로 훈련시키고, 이론을 통해 내부 활성화의 기하학적 예측을 수행합니다. 숨겨진 마르코프 모델(HMM)을 사용해 생성된 데이터 경로를 통해 토큰의 시퀀스를 생성하고, 믹스드 상태 표현(MSP)은 신념 업데이트 과정을 나타내는 계산 구조를 제시합니다.

- **Performance Highlights**: 연구 결과, 트랜스포머 모델은 신념 상태의 기하학적 정보와 더불어, 전체 미래에 대한 정보까지 포함하는 것으로 나타났습니다. 또한 신념 상태의 기하학은 최종 잔여 스트림에 나타나거나 여러 계층의 잔여 스트림에 분산되는 현상이 관찰되었으며, 이는 모델이 단순한 다음 토큰 예측을 넘어서 더 넓은 문맥에서 정보를 활용한다는 것을 시사합니다.



### Hacc-Man: An Arcade Game for Jailbreaking LLMs (https://arxiv.org/abs/2405.15902)
- **What's New**: 최근 발표된 연구인 Hacc-Man은 대형 언어 모델(LLM)을 '탈옥(jailbreak)'하는 게임을 통해 사용자가 LLM의 예기치 않은 출력을 탐구하도록 유도합니다. 이를 통해 LLM의 보안 위험성을 인식시키고, 사용자가 LLM과 더욱 효과적으로 상호작용할 수 있도록 돕습니다.

- **Technical Details**: Hacc-Man 게임은 사용자가 창의적인 문제 해결 전략을 사용해 LLM을 예상치 못한 방향으로 출력하도록 유도하는 과정에서의 행동을 탐구합니다. 이 게임은 또한 사용자가 LLM과의 상호작용에서 자신감을 높일 수 있도록 설계되었습니다. 게임은 물리적인 아케이드 머신뿐만 아니라 온라인에서도 접근 가능합니다.

- **Performance Highlights**: 연구에 따르면, Hacc-Man 게임은 사용자들에게 LLM 탈옥의 위험성과 가능성을 이해시키는 데 효과적이며, 이를 통해 LLM 보안 연구의 새로운 지평을 열 수 있습니다. 이 게임을 통해 사람들이 어떻게 창의적인 문제 해결 전략을 활용하는지에 대한 데이터도 수집됩니다.



### Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications (https://arxiv.org/abs/2405.15877)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 압축을 위한 새로운 저순위분해(低順位分解, low-rank decomposition) 접근법을 소개합니다. 이 접근법은 특정 응용 프로그램의 요구 사항에 맞춰 LLMs의 불필요한 부분을 제거하고 필요한 요소만 유지하는 방식으로 모델을 효과적으로 압축합니다.

- **Technical Details**: 저순위분해(SVD)를 사용하여 LLMs의 가중치 행렬을 기반 구성 요소(base components)의 선형 결합으로 표현합니다. 그런 다음 특정 응용 프로그램에 맞춰 중요도가 낮은 기반을 제거하고, 새로운 유용한 기반으로 모델을 강화합니다. 이 과정에서, 훈련 세트를 사용하여 특이값(singular values)을 재학습(retraining)하고 중요도가 낮은 기반을 가지치기(prune)합니다.

- **Performance Highlights**: Llama 2-7B 및 Llama 2-13B 모델을 수학적 추론과 코드 생성 작업(target applications)에서 평가한 결과, 당사의 방법이 기존의 저순위 압축 기술 대비 큰 압축 비율에서 더 나은 성능을 보임을 확인했습니다. 특히 수학적 추론 작업에서는 압축 비율이 6을 초과할 때, 코드 생성 작업에서는 4를 초과할 때 더 우수한 성능을 나타냈습니다.



### SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (https://arxiv.org/abs/2405.15793)
Comments:
          First two authors contributed equally. Code and demo at this https URL

- **What's New**: 이 논문에서는 소프트웨어 엔지니어링(SW)을 위한 자율 시스템인 SWE-agent를 소개합니다. 이 에이전트는 언어 모델(language model)을 사용하여 컴퓨터와 상호작용하며 소프트웨어 엔지니어링 작업을 해결합니다. 특히 사용자가 만든 에이전트-컴퓨터 인터페이스(ACI)가 에이전트의 코드 파일 생성 및 편집, 저장소(navigate entire repositories) 탐색 및 프로그램 실행 능력을 크게 향상시킵니다.

- **Technical Details**: SWE-agent는 맞춤형 ACI를 통해 컴퓨터와의 상호작용을 최적화하였습니다. 이를 통해 코드 파일을 만들고 수정하며, 전체 저장소(navigate entire repositories)를 탐색하고 프로그램을 실행할 수 있는 능력이 크게 향상되었습니다. 또한, ACI의 설계가 에이전트의 행동 및 성능에 어떤 영향을 미치는지 탐구하고, 효과적인 설계에 대한 통찰을 제공합니다.

- **Performance Highlights**: SWE-bench 벤치마크 테스트에서 SWE-agent는 문제 해결률 12.5%를 기록했으며, 이는 기존의 검색-증강 생성(retrieval-augmented generation, RAG) 방식이 달성한 3.8%와 비교하여 크게 향상된 결과입니다.



### Extracting chemical food safety hazards from the scientific literature automatically using large language models (https://arxiv.org/abs/2405.15787)
Comments:
          31 pages, 5 figures

- **What's New**: 이 연구에서는 식품 안전 분야에서 과학 문헌으로부터 화학적 위험(chemical hazards)을 자동 추출하는 접근법을 소개합니다. 이 접근법은 대형 언어 모델(large language models)을 사용하여 과학 논문의 초록(abstract)에서 관련 정보를 추출합니다. 추가적인 모델 학습이나 대규모 컴퓨팅 클러스터가 필요하지 않습니다.

- **Technical Details**: 세 가지 다른 스타일의 프롬프트(prompt)가 테스트되어 최적의 방법을 평가하였습니다. 모델은 외부 상자(out-of-the-box)로 사용되었으며, 구체적인 프롬프트의 문구(wording)가 결과에 큰 영향을 미쳤습니다. 작업을 작은 단계로 나누어 제시하는 프롬프트가 가장 우수한 성능을 보였습니다.

- **Performance Highlights**: 최적 프롬프트는 평균 정확도 93%를 기록했으며, 식품 모니터링 프로그램에 이미 포함된 많은 화학적 오염물질(chemical contaminants)을 성공적으로 추출해냈습니다. 이는 대형 언어 모델이 과학 문헌에서 자동으로 정보를 추출하는 작업에 얼마나 가치 있는지를 보여줍니다.



### CLARINET: Augmenting Language Models to Ask Clarification Questions for Retrieva (https://arxiv.org/abs/2405.15784)
- **What's New**: Clarinet는 정보 검색 환경에서 명확한 질문을 통해 검색 성능을 향상시키는 새로운 시스템을 소개합니다. 이 시스템은 불확실성을 줄이는 질문을 선택하여, 검색 모델의 불확실성을 자연어 질문으로 바꾸는 문제를 해결합니다.

- **Technical Details**: Clarinet는 대형 언어 모델(LLM)을 활용하여 검색 분포 기반으로 조건화하고, 질의응답을 통해 최적의 질문을 학습합니다. 특히, Dense Passage Retriever (DPR)를 사용하여 책 데이터베이스를 검색하며, 다양한 대화 이력을 통해 후속 질문을 생성하여 사용자의 의도를 정확히 파악합니다. Clarinet는 10턴의 대화 후 상대적으로 39% 향상된 정확도를 보였으며, 정보 이득(information gain)과 같은 기존 휴리스틱 방법보다 17% 더 높은 성능을 발휘합니다.

- **Performance Highlights**: Goodreads 데이터셋을 이용한 평가에서 Clarinet는 사용자가 찾는 책을 정확히 찾아내는 데 있어서 최고 1위 검색 정확도에서 전통적인 휴리스틱 접근 방식보다 17%, 일반 LLM보다 39% 더 뛰어난 성능을 보여줍니다.




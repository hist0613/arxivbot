New uploads on arXiv(cs.CL)

### The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models (https://arxiv.org/abs/2411.08870)
Comments:
          Extended version of EMNLP 2024 paper arXiv:2411.04118. Includes additional results on clinical note QA tasks and supervised fine-tuning evaluations

- **What's New**: 이번 연구는 의료 분야에 적합한 언어 모델의 성능을 재검토한 것으로, 기존의 특정 의료 데이터에 대한 적응 훈련(DAPT)이 실제로 기대한 만큼의 성능 향상을 제공하지 않는다고 주장합니다. 22.7%의 경우에서만 의료 모델이 기본 모델보다 성능이 우수하며, 40.5%의 경우에는 성능이 열악한 결과를 보였습니다.

- **Technical Details**: 10개의 공개된 "의료" LLM과 2개의 VLM을 대상으로 기본 모델과의 성능 비교를 수행했습니다. 연구에서 다양한 의료 QA 작업을 통해 의료 LLM이 기본 모델보다 일관되게 개선되지 않음을 명확히 하였습니다. 이 과정에서 개별 최적의 프롬프트(selecting the ‘best’ prompt)를 사용하여 수행하였으며 통계적 불확실성을 고려하여 분석했습니다.

- **Performance Highlights**: 의료 LLM은 의료 지식 QA 작업에 대해서는 유의미한 개선을 보였지만, 임상 메모 기반의 QA 작업에서는 개선되지 않았고, 의료 VLM은 모든 시각적 미디어 QA 작업에서 거의 개선되지 않는 결과를 보였습니다. 또한, DAPT의 성능 이점을 평가하기 위해서는 엄격한 쌍비교(observational comparison)가 필수적이라는 점을 강조했습니다.



### CamemBERT 2.0: A Smarter French Language Model Aged to Perfection (https://arxiv.org/abs/2411.08868)
- **What's New**: 본 논문에서는 CamemBERT의 두 가지 새로운 버전인 CamemBERTav2와 CamemBERTv2를 소개하며, 이는 업데이트된 모델로 현재의 언어 트렌드를 반영하기 위해 설계되었습니다. 두 모델 모두 최신 데이터셋에서 훈련되어 프랑스어 처리 성능을 극대화합니다.

- **Technical Details**: CamemBERTav2는 DeBERTaV3 아키텍처를 기반으로 하며 Replaced Token Detection (RTD) 훈련 목표를 사용하여 문맥적 이해를 향상시킵니다. CamemBERTv2는 RoBERTa 아키텍처를 기반으로 하여 Masked Language Modeling (MLM) 목표로 훈련됩니다. 두 모델은 더 많은 최신 데이터셋에서 훈련되었으며, 프랑스어에 최적화된 새로운 토크나이저를 사용합니다.

- **Performance Highlights**: 이 모델들은 일반 도메인 NLP 태스크와 의료 분야와 같은 도메인 특정 애플리케이션에서 성능을 평가하였고, 그 결과 두 모델 모두 이전 버전들보다 현저히 우수한 성능을 나타냈습니다. 모델 아티팩트와 체크포인트는 Huggingface에서 공개되어 있습니다.



### Zero-shot Cross-lingual Transfer Learning with Multiple Source and Target Languages for Information Extraction: Language Selection and Adversarial Training (https://arxiv.org/abs/2411.08785)
- **What's New**: 본 연구는 기존의 제로샷 크로스링구얼 싱글 트랜스퍼 접근 방식에서 벗어나 여러 언어 간의 정보 추출 성능을 높이기 위한 다중 트랜스퍼(Multi-Transfer) 가능성을 조사합니다. 또한, 언어 간의 거리 값을 기반으로 한 클러스터링 기법을 통해 효율적인 데이터 수집 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 언어 간 에서의 단일 전이 성능과 언어적 거리 간의 상관관계를 분석합니다. URIEL 데이터베이스를 활용하여 언어의 계통학적 및 유형론적 속성을 추출하고, 이를 바탕으로 다양한 언어 간 거리(유사성 점수)를 계산합니다. 또한, 다중 언어 비지도 데이터를 활용한 관계 전이(relational-transfer) 학습을 제안합니다.

- **Performance Highlights**: 양적 실험 및 질적 분석 결과, 제안된 방법론에 따르면 각 언어에 대한 전이 성능이 크게 향상되었습니다. 특히 연관 전이 설정(ZSCL-R)에서 다수의 언어로부터 수집된 비지도 데이터의 활용을 통해 전반적인 모델 성능이 개선되는 결과를 보여주었습니다.



### Multi-Perspective Stance Detection (https://arxiv.org/abs/2411.08752)
- **What's New**: 이번 연구는 주관적 NLP 과제가 다수의 주석자에 의해 제공되는 인간 주석에 의존한다는 점을 고찰하며, 다수의 주석을 사용하는 것이 모델의 분류 정확도에 미치는 영향을 조사합니다.

- **Technical Details**: 이 연구는 stance detection(입장 탐지) 작업에서 viewpoint-aware classification models(관점 인식 분류 모델)의 성능을 평가하며, 주석자 간의 의견 불일치가 모델 신뢰도에 미치는 영향을 분석합니다. 자연어 처리를 위한 새로운 접근법인 perspectivism(관점주의)를 채택하여 다양하고 불일치하는 다수의 주석을 데이터에 포함시킴으로써 윤리적이고 포괄적인 AI 모델 개발을 촉진합니다.

- **Performance Highlights**: 다중 관점 접근법은 단일 라벨을 사용하는 기존 방식에 비해 더 나은 분류 성능을 보였습니다. 따라서, 관점 인식 AI 모델을 설계하는 것은 책임 있는 AI 구현을 위한 필수적인 첫 단계일 뿐만 아니라 전통적인 접근 방식보다 우수한 결과를 달성할 수 있습니다.



### Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers (https://arxiv.org/abs/2411.08745)
Comments:
          12 pages, 10 figures, previously published under the title "How Do Llamas Process Multilingual Text? A Latent Exploration through Activation Patching" at the ICML 2024 mechanistic interpretability workshop this https URL

- **What's New**: 본 논문은 멀티링구얼 언어 모델링의 중심 질문인 대형 언어 모델(LLM)이 특정 언어와 무관하게 보편적인 개념 표현을 발전시키는지를 분석합니다. 특히, transformer 기반 LLM에서 단어 번역 작업 중에 숨겨진 표현(latents)을 분석하여 개념과 언어 간의 관계를 탐구합니다.

- **Technical Details**: 연구자들은 단어 번역 작업의 소스 번역 프롬프트에서 latents를 추출하고 이를 타겟 번역 프롬프트의 전방 전달 과정에 삽입합니다. 이를 통해 early layer에서 출력 언어가 latents에 인코딩되어 있고, 변환할 개념은 이후 과정에서 처리된다는 사실을 발견합니다. 두 가지 주요 실험, 즉 activations 패칭을 사용하여 언어를 변경하지 않고 개념을 변경하는 것과 다양한 언어 간의 평균 latents로 패칭하여 성능 향상을 입증합니다.

- **Performance Highlights**: 연구 결과는 Llama 2 7B 모델을 사용하여 번역 작업 성능이 평균 개념 표현을 사용하는 경우 향상되며, 이는 개념이 언어와 독립적으로 표현됨을 지지합니다. 본 분석은 Llama 2 뿐만 아니라 다양한 transformer 모델의 성능에도 일반화된다고 제안합니다.



### A Comparative Study of Discrete Speech Tokens for Semantic-Related Tasks with Large Language Models (https://arxiv.org/abs/2411.08742)
Comments:
          5 tables, 4 figures

- **What's New**: 이 연구는 Speech Large Language Models (Speech LLMs)에서 연속적인 음성 특징과 이산 음성 토큰 간의 성능 차이를 체계적으로 비교하는 첫 번째 연구입니다. 이로 인해 이산 음성 토큰의 발전 가능성을 제시합니다.

- **Technical Details**: 연속 피처와 이산 토큰의 성능 비교를 위해 K-means 클러스터링을 사용하여 이산 음성 토큰을 생성합니다. 사용된 두 가지 처리 파이프라인은 (a) K-means 클러스터링, 중복 제거 및 서브워드 모델링을 포함한 이산 토큰 파이프라인, (b) 다운샘플링 및 선형 어댑터 모듈을 포함한 연속 피처 파이프라인입니다. 이를 통해 SSL 모델에서 생성된 피처를 효과적으로 처리합니다.

- **Performance Highlights**: 연구 결과, 연속 피처가 이산 토큰보다 대부분의 작업에서 우수한 성능을 보이며, 특히 세밀한 의미 이해가 필요한 작업에서 두드러진 성능 차이를 보였습니다. 이산 토큰의 성능 저하 원인으로는 제한된 토큰의 세분화 및 비효율적인 정보 유지가 있음을 규명하였습니다.



### Dynamic Rewarding with Prompt Optimization Enables Tuning-free Self-Alignment of Language Models (https://arxiv.org/abs/2411.08733)
Comments:
          EMNLP 2024 Main

- **What's New**: 본 논문은 경제적이고 빠르게 조정할 수 있는 대체 방법으로 Dynamic Rewarding with Prompt Optimization(\

- **Technical Details**: DRPO는 검색 기반 최적화 프레임워크를 활용하여 LLM이 스스로 개선하고 최적의 정렬 지침을 제작할 수 있도록 합니다. 이 접근 방법은 기존의 튜닝이 필요하지 않으며, 동적 보상 메커니즘을 통해 모델의 특정 정렬 약점을 식별하고 수정합니다.

- **Performance Highlights**: DRPO는 8개의 최신 LLM에서 실험하여 기존의 강화 학습 기반 방법들보다 성능을 크게 향상시킵니다. 특히, SFT/RLHF 조정 모델보다 기본 모델들이 더 뛰어난 성능을 보여주는 것으로 나타났습니다.



### Analyst Reports and Stock Performance: Evidence from the Chinese Mark (https://arxiv.org/abs/2411.08726)
- **What's New**: 이 논문은 자연어 처리(NLP)를 사용하여 텍스트 정보를 추출하고 수량화하여 주식 성과를 예측하는 새로운 방법론을 제시합니다. 특히, 중국의 분석가 보고서를 이용하여 맞춤형 BERT 딥러닝 모델을 통해 보고서의 감정을 긍정적, 중립적, 부정적으로 분류합니다.

- **Technical Details**: 연구진은 627,356,273,562,735,627,35 건의 중국 금융 분석가 보고서 데이터셋을 분석하여, BERT(Bidirectional Encoder Representations from Transformers) 모델을 적용했습니다. 감정 점수는 주식의 초과 수익, 변동성 및 거래량과의 상관관계를 통해 분석됐습니다. 또한 회귀 분석을 통해 감정과 주식 행동의 관계를 규명했습니다.

- **Performance Highlights**: 연구 결과, 긍정적인 감정 점수가 높은 보고서는 그 다음 날 주식의 초과 수익을 증가시키는 경향이 있으며, 강한 긍정적 및 부정적 감정 모두 다음 날 주식의 하루 변동성을 증가시킴을 보여줍니다. 강한 긍정적 감정은 다음 날의 거래량 증가와 연관되어 있는 반면, 부정적 감정에서는 같은 효과가 관찰되지 않았습니다.



### QCG-Rerank: Chunks Graph Rerank with Query Expansion in Retrieval-Augmented LLMs for Tourism Domain (https://arxiv.org/abs/2411.08724)
- **What's New**: 이 논문은 RAG(Retrieval-Augmented Generation)의 한계를 극복하기 위해 QCG-Rerank 모델을 제안합니다. 이 모델은 처음에 쿼리와 관련된 후보 청크를 검색한 후, 세분화된 정보를 추출하여 원래 쿼리를 확장합니다. 그런 다음, 확장된 쿼리와 후보 청크를 활용하여 유사도 점수를 계산하고, 이를 바탕으로 청크 그래프를 구축합니다.

- **Technical Details**: QCG-Rerank 모델은 간단한 쿼리에서 핵심적인 정보를 추출하고 이를 바탕으로 쿼리의 의미적 복잡성을 확장합니다. 초기 검색 결과의 유사성을 기반으로 청크 간의 전이 확률을 반복적으로 계산하여 수렴할 때까지 진행하며, 최고 점수를 가진 청크를 LLM에 입력하여 결과를 생성합니다. 이 과정은 Cultour, IIRC, StrategyQA, HotpotQA, SQuAD, MuSiQue 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, QCG-Rerank 방법의 효과성과 우수성을 입증하였습니다. 이 모델은 여행 도메인에서 LLM(대형 언어 모델)의 환각(hallucination) 문제를 완화하고, 보다 정확한 응답을 생성할 수 있도록 합니다.



### Are Triggers Needed for Document-Level Event Extraction? (https://arxiv.org/abs/2411.08708)
- **What's New**: 이번 논문은 기존의 문장 수준(event extraction)에서의 트리거 개념을 확장하여, 문서 수준(document-level) 이벤트 추출의 역할을 첫 번째로 조사한 연구입니다.

- **Technical Details**: 우리는 다양한 품질의 트리거(인간 주석, LLM 생성, 키워드 기반, 무작위)를 사용하여 문서 수준 이벤트 추출의 여러 신경망 모델의 성능을 분석했습니다. 트리거는 이벤트의 존재를 나타내는 '이유(rationale)'로 간주되어 시스템이 이벤트 인수를 찾는 데 도움을 줍니다.

- **Performance Highlights**: 연구 결과는 트리거의 유용성이 추출 작업 및 데이터의 특성에 따라 다르며, 자동 생성된 낮은 품질의 트리거도 인간 주석의 대안으로 활용 가능하다는 것을 보여줍니다. 게다가 트리거 품질 저하에도 불구하고 이벤트 정보 제공이 성능의 강건성을 유지하는 데 기여합니다.



### Dynamic Subset Tuning: Expanding the Operational Range of Parameter-Efficient Training for Large Language Models (https://arxiv.org/abs/2411.08610)
Comments:
          NeurIPS 2024 Workshop on Adaptive Foundation Models

- **What's New**: 본 논문에서는 기존의 모델 파라미터 집합을 최적화하여 다운스트림(downstream) 작업에 적응할 수 있는 새로운 파라미터 효율적 훈련(PEM) 방법을 제안합니다. 이전 방법과 달리 이 파라미터 집합은 고정되지 않고 훈련 과정 중에 변하는 동적인 선택이 특징입니다.

- **Technical Details**: 제안된 방법은 '동적 서브셋 튜닝(Dynamic Subset Tuning, DST)'이라고 하며, 매 훈련 단계마다 자유롭게 조정되는 파라미터 집합을 재선택하는 방식으로 설계되었습니다. 이 방법은 사용자가 정의할 수 있는 파라미터 예산을 모델의 크기에 따라 비율로 명확히 지정할 수 있는 유연함을 제공합니다.

- **Performance Highlights**: 본 연구에서 제안하는 방법은 여러 NLP 작업(MT, QA, GSM8K, SuperGLUE)에서 기존의 프롬프트 튜닝(prompt tuning) 및 LoRA와 대등하거나 더 나은 성능을 보여주며, 또한 훨씬 더 적은 수의 파라미터(최소 0.00001%까지)를 사용하여 매우 작은 훈련 세트에서도 효과적으로 사용할 수 있음을 입증하였습니다.



### CorrSynth -- A Correlated Sampling Method for Diverse Dataset Generation from LLMs (https://arxiv.org/abs/2411.08553)
Comments:
          Published as a main conference paper at EMNLP 2024; First two authors contributed equally

- **What's New**: 이번 연구에서는 CorrSynth라는 새로운 데이터 생성 방법론을 제안하여, 합성 데이터 생성 시 다양성과 프롬프트에 대한 충실성을 높이고자 하였습니다. 이를 통해 학생 모델을 위한 고품질 데이터셋을 생성하는데 기여하고자 하였습니다.

- **Technical Details**: CorrSynth는 상관 샘플링(correlated sampling) 접근 방식을 사용하여, 다양한 텍스트 시퀀스를 병렬로 생성하며, 각 시퀀스 간의 강한 상관관계를 유지합니다. 이 방법은 기존의 Classifier Free Guidance (CFG) 방법론과 비교하여 클래스/라벨 간 로짓(logit)의 대비(constrast)를 모색하여 독창적으로 개선하였습니다.

- **Performance Highlights**: CorrSynth는 학생 모델의 메트릭과 내재적 메트릭 모두에서 4개의 데이터셋에 걸쳐 기존 최첨단 방법들과 비교하여 우수한 성능을 보였습니다. 특히, 생성된 데이터의 다양성과(Human text와의 유사성) 정확도가 향상되어, 후반 작업에서의 지속적인 활용 가능성이 강조되었습니다.



### Neural Topic Modeling with Large Language Models in the Loop (https://arxiv.org/abs/2411.08534)
- **What's New**: 이번 논문에서는 LLM-ITL이라는 새로운 프레임워크를 제안합니다. 이는 LLM(대규모 언어 모델)과 NTM(신경 주제 모델)을 통합하여 기존의 주제 모델링의 한계를 극복하고자 합니다. LLM-ITL은 전 세계 주제(global topics)와 문서 표현(document representations)을 NTM을 통해 학습하며, LLM을 이용해 주제를 정제합니다.

- **Technical Details**: LLM-ITL은 주제의 해석 가능성과 일관성을 높이기 위해 Optimal Transport (OT) 기반의 정렬 목표를 통해 LLM의 주제를 개선합니다. 또한, LLM이 생성하는 비현실적이거나 비관련된 제안을 최소화하기 위해 신뢰도 기반의 메커니즘을 도입하여 LLM의 제안 영향력을 조절합니다.

- **Performance Highlights**: 대규모 실험 결과, LLM-ITL은 주제의 해석 가능성을 현저히 향상시키면서 문서 표현 품질을 유지하는 데 있어 이전의 방법들보다 우수한 성능을 보였습니다. LLM-ITL은 NTM과 LLM의 통합으로 더 나은 주제 일관성과 문서 표현 품질을 달성하여 주제 모델링 분야에서 최신 성능을 기록하고 있습니다.



### Tree-of-Table: Unleashing the Power of LLMs for Enhanced Large-Scale Table Understanding (https://arxiv.org/abs/2411.08516)
- **What's New**: 이 논문에서는 "Tree-of-Table"이라는 새로운 접근법을 제안하여 대규모 및 복잡한 테이블에 대한 LLM의 추론 능력을 향상시킵니다. 이 방법은 Table Condensation과 Decomposition을 사용하여 관련 데이터를 관리 가능한 형식으로 정리한 후, 계층적인 Table-Tree를 구성하여 나무 구조의 추론을 용이하게 합니다.

- **Technical Details**: 제안된 Tree-of-Table 모델은 데이터의 Condensation(농축) 및 Decomposition(분해)을 통해 정보를 체계화하고, 테이블의 복잡성을 해소하여 계층적 방법론을 통해 추론 과정을 유도합니다. 이를 통해 각 노드는 특정한 목적을 갖고 LLM과 테이블 데이터 간의 상호작용을 단순화합니다.

- **Performance Highlights**: 다양한 데이터셋(WikiTQ, TableFact, FeTaQA, BIRD)에 대한 실험 결과, Tree-of-Table은 최첨단 성과를 달성하였고, 대규모 테이블 추론에서 뛰어난 효율성과 일반화 능력을 보여줍니다.



### Towards Objective and Unbiased Decision Assessments with LLM-Enhanced Hierarchical Attention Networks (https://arxiv.org/abs/2411.08504)
- **What's New**: 이 연구는 대학 입시 과정에서 인간 전문가의 결정에서 인지 편향(cognitive bias)을 식별하고, 이를 극복하기 위한 AI 증강 워크플로우(BGM-HAN)를 제안합니다. 기존의 결정 과정에서 드러나는 불일치와 편향을 분석하여 AI의 도움으로 보다 공정하고 일관된 결정을 원하는 방향으로 나아가고자 했습니다.

- **Technical Details**: 제안된 모델인 BGM-HAN은 'Byte-Pair Encoding'과 'Multi-Head Attention', 'Gated Residual Connection'을 이용하여 다층적인 세미 구조화된 데이터를 효과적으로 표현합니다. 또한 이 모델은 'Shortlist-Analyze-Recommend' (SAR) 에이전틱 워크플로우를 사용하여 기존의 인간 결정 과정을 모방하면서 일관성을 향상시키고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델과 워크플로우는 기존 인간 평가를 기준으로 F1-score와 정확도에서 9.6% 이상의 개선을 보였습니다. 이러한 결과는 계층적 학습(hierarchical learning)이 AI의 자동화된 결정-making 과정에서 내재된 공정성과 일관성을 제공하는 잠재력을 보여줍니다.



### One STEP at a time: Language Agents are Stepwise Planners (https://arxiv.org/abs/2411.08432)
- **What's New**: STEP라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 언어 에이전트의 계획 능력을 향상시키기 위해 과거 경험에서 학습하는 것을 목표로 합니다.

- **Technical Details**: STEP는 Planner, Executor, Evaluator, Memory의 4가지 상호 연결된 구성 요소로 작동합니다. Planner는 작업을 하위 작업으로 분해하고 관련 정보를 Memory에서 검색합니다. Executor는 동작 후보를 생성하고, Evaluator는 이 동작들이 이전 경험에서 학습한 규칙들과 일치하는지 확인합니다. Memory는 경험을 저장하여 미래의 결정을 알리는 역할을 합니다.

- **Performance Highlights**: ScienceWorld 벤치마크에서 STEP는 67.4의 전체 점수를 기록하며, 18개 작업 중 12개를 성공적으로 완료했다는 결과를 보여주었습니다. 이는 STEP의 언어 에이전트의 계획 능력을 향상시키기 위한 프레임워크로서의 가능성을 부각시킵니다.



### CLaSP: Learning Concepts for Time-Series Signals from Natural Language Supervision (https://arxiv.org/abs/2411.08397)
- **What's New**: 이 논문에서는 자연어 쿼리를 사용하여 시계열 신호를 검색할 수 있는 "CLaSP"라는 기반 모델을 제안합니다. 기존의 시계열 신호 데이터 자연어 표현은 여러 가지 한계가 있어, 저자들은 대조 학습 (contrastive learning) 기반의 신경망을 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: CLaSP는 TRUCE 및 SUSHI 데이터셋을 이용해 훈련된 대조 학습 기반의 신경망으로, 시계열 신호와 해당 자연어 설명 쌍으로 구성되어 있습니다. 모델의 입력은 시계열 신호 데이터와 해당 레이블이며, 두 개의 인코더(신호 인코더 및 텍스트 인코더)를 통해 처리됩니다. 이 모델은 정의된 동의어 사전 없이도 자연어 검색이 가능하며, 대규모 언어 모델(LLM)의 일반적인 지식을 활용합니다.

- **Performance Highlights**: 실험 결과에 따르면 CLaSP는 시계열 신호 데이터에 대한 자연어 검색을 가능하게 하며, 신호 데이터의 변화를 정확하게 학습할 수 있음을 보여주었습니다. 이는 기존의 시계열 데이터 처리 방법보다 효율적인 접근 방식을 제공합니다.



### Interpretable Syntactic Representations Enable Hierarchical Word Vectors (https://arxiv.org/abs/2411.08384)
- **What's New**: 본 연구는 기존의 밀집(dense)하고 해석하기 어려운 단어 표현을 줄이고 해석 가능한 구문적(syntactic) 표현으로 변환하는 방법을 제안합니다. 이 방법은 단어 벡터를 보다 명확하게 시각화하고 비교할 수 있게 하며, 인간의 판단과 일치하는 해석을 제시합니다.

- **Technical Details**: 연구팀은 기존의 단어 벡터에서 구문적 표현을 유도하기 위해 사전 훈련된 벡터를 사용하여 각 좌표가 여덟 개의 품사(part of speech) 중 하나에 해당하도록 변환합니다. 이후, 이러한 구문적 표현을 바탕으로 단계적 학습(incremental learning) 접근법을 통해 계층적 단어 벡터(hierarchical word vectors)를 생성하였습니다. 이 방법은 인간의 학습 구조를 모방하여 이전 학습된 단어 벡터를 기반으로 점차 복잡한 내용으로 확장합니다.

- **Performance Highlights**: 개발된 구문적 표현은 원래 벡터보다 더 해석이 용이하며, 계층적 벡터는 벤치마크 테스트에서 원래 벡터보다 뛰어난 성능을 보여줍니다. 특히, 본 연구는 언어 모델의 불투명성과 성능 간의 간극을 좁히기 위한 새로운 접근 방식을 제공합니다.



### Refining Translations with LLMs: A Constraint-Aware Iterative Prompting Approach (https://arxiv.org/abs/2411.08348)
- **What's New**: 대규모 언어 모델(LLMs)은 기계 번역(MT)에서 인상적인 성능을 보여주고 있으나, 드문 단어의 번역에서 여전히 어려움을 겪고 있습니다. 본 연구에서는 번역의 신뢰성을 향상시키기 위해 핵심 용어를 우선순위에 두고 다단계 프롬프트 체인을 제안합니다.

- **Technical Details**: 제안된 방법은 다음과 같습니다. 먼저, 원문에서 번역 품질에 중요한 키워드를 식별하고, 이들에 대한 번역을 이중언어 사전에서 검색하여 Retrieval-Augmented Generation(RAG) 기법을 통해 LLM의 문맥에 통합합니다. 이후, 반복적인 자기 점검 메커니즘을 통해 긴 프롬프트에서 발생할 수 있는 출력 환각(output hallucinations)을 완화합니다.

- **Performance Highlights**: FLORES-200 및 WMT 데이터셋에서의 실험 결과, 제안된 방법이 기존 접근 방식보다 유의미하게 향상된 성능을 보였으며, 특히 저자원 언어에서의 번역 신뢰성과 견고성을 크게 개선함을 입증하였습니다.



### Bangla Grammatical Error Detection Leveraging Transformer-based Token Classification (https://arxiv.org/abs/2411.08344)
- **What's New**: 본 논문은 자동화된 Bangla 문법 검사기의 개발이 미약한 분야임을 지적하며, Bangla 언어의 문법, 구두점 및 철자 오류를 효율적으로 검출하기 위한 접근 방식을 제안합니다.

- **Technical Details**: 이번 연구에서는 Bangla 문법 오류 검출 문제를 토큰 분류(token classification) 문제로 정의하고, 최신 트랜스포머 기반(transformer-based) 모델을 활용했습니다. 시스템은 ELECTRA 모델 및 BanglaBERT 모델을 사용하여 오류 클래스를 예측하고, 이후 규칙 기반(rule-based) 후처리를 통해 보다 신뢰할 수 있는 결과를 생성합니다.

- **Performance Highlights**: 제안된 모델은 25,000개 이상의 다양한 출처의 텍스트로 구성된 데이터셋에서 평가되었으며, 가장 성능이 좋은 모델은 Levenshtein 거리(Levenshtein distance) 점수 1.04를 달성했습니다.



### Are LLMs Prescient? A Continuous Evaluation using Daily News as the Orac (https://arxiv.org/abs/2411.08324)
- **What's New**: 새로운 평가 벤치마크인 Daily Oracle을 제안합니다. 이 벤치마크는 매일 뉴스에서 생성된 질문-답변(QA) 쌍을 사용하여 LLM의 시간적 일반화(temporal generalization) 및 예측 능력을 평가합니다.

- **Technical Details**: Daily Oracle은 True/False(TF) 및 Multiple Choice(MC) 형태의 질문으로 구성되어 있으며, 다양한 카테고리(예: 비즈니스, 정치, 예술)에서 LLM이 미래 사건을 예측하도록 도전합니다. 연구 결과, LLM은 2020년 1월부터 2024년 9월까지 TF 질문에서 평균 20.14%, MC 질문에서 23.26%의 성능 저하를 경험했습니다.

- **Performance Highlights**: 이 연구는 LLM의 예측 정확도가 시간이 지남에 따라 지속적으로 감소한다는 것을 증명합니다. RAG 방식을 활용한 모델이 예측 성능이 향상될 수 있지만, 여전히 성능 감소 패턴은 지속됩니다. 이는 항상 업데이트되어야 하는 모델의 필요성을 강조합니다.



### R3HF: Reward Redistribution for Enhancing Reinforcement Learning from Human Feedback (https://arxiv.org/abs/2411.08302)
- **What's New**: 본 논문은 Reinforcement Learning from Human Feedback (RLHF)에서 발생하는 기존의 보상 구조를 개선하여 각 토큰에 대해 더 세밀한 보상을 할당하는 R3HF 방법을 제안합니다.

- **Technical Details**: R3HF 방법은 보상 모델을 회귀(regression) 문제로 간주하여 각 토큰의 기여도를 분석한 후, 정확한 보상을 분배합니다. 이 과정은 Sequence-Markov Decision Process (SDP) 프레임워크를 통해 구현되며, 각 토큰에 대해서도 그 이전 시간 단계와의 비교를 통해 보상 크기를 유도하는 방법입니다.

- **Performance Highlights**: 다양한 데이터셋과 작업을 통해 실시한 실험 결과, R3HF는 기존 RLHF 기술에 비해 학습 효율성과 성능에서 개선된 결과를 보여 주었습니다. 제안된 방법이 기존의 RLHF 방법에 원활하게 통합될 수 있음을 입증하였습니다.



### Knowledge Bases in Support of Large Language Models for Processing Web News (https://arxiv.org/abs/2411.08278)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)을 활용하여 웹 뉴스 처리를 위한 지식 기반 구축을 지원하는 일반적인 프레임워크를 소개합니다. 이 프레임워크는 뉴스 정보 추출기(NewsIE)를 사용하여 뉴스 항목에서 관계형 튜플을 추출하고 이를 그래프 합성곱 네트워크(GCN)를 통해 LLM의 암묵적 지식 사실과 결합하여 분류하는 방식입니다.

- **Technical Details**: 프레임워크는 두 가지 가벼운 구성 요소로 이루어져 있습니다: 1) NewsIE: 뉴스 항목의 구조적 정보(관계형 튜플)를 추출합니다; 2) BERTGraph: NewsIE에서 추출한 관계형 튜플과 LLM에서 얻은 암묵적 지식 사실을 그래프 합성곱하여 처리합니다. BERT(양방향 인코더 표현)를 사용하여 GCN에 입력될 수 있도록 텍스트-그래프 어댑터를 설계하였습니다.

- **Performance Highlights**: BERTGraph는 N24News, SNOPES, politifact의 세 가지 공개 웹 뉴스 데이터셋에서 뉴스 카테고리 분류 성능을 평가했으며, 모든 성능 메트릭에서 기존 BERT보다 뛰어난 결과를 보였습니다. 특히 politifact에서 정확도(precision) 이외의 모든 측정 항목에서 우수한 성과를 기록했습니다.



### Deceiving Question-Answering Models: A Hybrid Word-Level Adversarial Approach (https://arxiv.org/abs/2411.08248)
- **What's New**: 본 연구는 QA 모델을 대상으로 하는 새로운 적대적 공격 전략인 QA-Attack을 제안합니다. 이 방식은 특정 단어를 표적으로 삼아 QA 모델을 속이는 워드 수준의 적대적 기법입니다.

- **Technical Details**: QA-Attack은 Hybrid Ranking Fusion (HRF) 알고리즘을 사용하여 Attention-based Ranking (ABR) 및 Removal-based Ranking (RBR) 두 가지 방법을 통합합니다. ABR은 질문 처리 중 주의 가중치를 분석하여 중요 단어를 식별하고, RBR은 특정 단어 제거 시 모델의 출력 변화를 관찰하여 단어의 중요성을 평가합니다.

- **Performance Highlights**: QA-Attack은 다양한 질문 유형에 걸쳐 효과적으로 작동하며, 여러 벤치마크 데이터셋에서 기존의 QA 모델을 성공적으로 속였습니다. 연구 결과는 성공률, 의미 변화, BLEU 점수, 유창성 및 문법 오류율 면에서 기존 적대적 기법들을 초월하는 성과를 보여주었습니다.



### Beyond the Safety Bundle: Auditing the Helpful and Harmless Datas (https://arxiv.org/abs/2411.08243)
Comments:
          Prepared for conference submission

- **What's New**: 이 연구에서는 LLMs(대형 언어 모델)의 안전성을 높이기 위한 수단으로 LHF(인간 피드백 학습)를 사용함에 있어, 널리 사용되는 Helpful and Harmless (HH) 데이터셋에 대한 품질과 효과성을 감사(감독)했습니다. HH 데이터셋의 내용을 분석하고, 해당 데이터셋이 모델의 안전성에 미치는 영향을 실험하며, 이 데이터셋을 인용한 가장 영향력 있는 연구 100편을 분석하였습니다.

- **Technical Details**: 헬퍼(Helpfulness), 정직함(Honesty), 해롭지 않음(Harmlessness) 원칙에 따라 HH 데이터셋의 품질을 평가하고 서로 다른 변형을 사용하여 모델을 훈련시키며, 이를 통해 LHF(Learning from Human Feedback) 사용의 안전성 문제를 분석했습니다.

- **Performance Highlights**: HH 데이터셋의 구조적 결함과 품질 문제로 인해 인구 그룹 간의 안전성 행동에 차이가 발생할 수 있음을 발견했습니다. 또한, 모델이 훈련된 데이터셋으로부터의 학습 차이가 특정 인구 그룹에서 과장된 안전성을 나타낼 수 있다는 점을 강조했습니다.



### Large Language Models Can Self-Improve in Long-context Reasoning (https://arxiv.org/abs/2411.08147)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 긴 맥락 추론 능력을 자가 개선할 수 있는 가능성을 탐구하며, 이를 위한 새로운 방법론인 SeaLong을 제안합니다. SeaLong은 여러 출력 샘플을 생성하여 Minimum Bayes Risk를 기반으로 평가한 후, 이를 통해 감독 학습(supervised fine-tuning)이나 선호 최적화(preference optimization)를 적용하는 접근 방식입니다.

- **Technical Details**: SeaLong은 먼저 LLM으로부터 여러 추론 경로를 샘플링하고, 각 경로를 Minimum Bayes Risk (MBR)로 점수화하여, 높은 점수를 받은 출력을 사용해 감독 학습을 진행하거나, 높은 점수와 낮은 점수의 출력을 모두 활용해 최적화를 시도합니다. 이 방식은 일관성이 높은 출력을 우선시하며, 잘못된 출력을 줄이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, SeaLong을 적용한 Llama-3.1-8B-Instruct는 50.8에서 55.0으로 4.2포인트 향상되었습니다. 또한 Qwen-2.5-14B-Instruct는 32B 변형모델보다 더 우수한 성능을 보였으며, 이는 기존의 인적 주석이나 전문가 모델에 의존하지 않고도 이루어진 결과입니다.



### On the Role of Speech Data in Reducing Toxicity Detection Bias (https://arxiv.org/abs/2411.08135)
- **What's New**: 이번 연구에서는 텍스트 기반의 독성 감지 시스템이 특정 인구 집단에 대해 불균형한 오탐지를 나타내는 것과 더불어, 음성 기반의 독성 감지 시스템에서 이러한 편견이 얼마나 완화되는지를 조사했습니다.

- **Technical Details**: 연구에서는 다국어 MuTox 데이터셋을 활용하여 고품질의 그룹 주석(annotations)을 생성하고, 텍스트 기반 독성 분류기와 음성 기반 독성 분류기를 체계적으로 비교했습니다. 그 결과 음성 데이터 사용이 불명확하거나 논쟁을 일으키는 샘플에서 그룹 언급에 대한 편향을 줄이는 데 도움이 됨을 발견했습니다.

- **Performance Highlights**: 음성 기반 시스템이 그룹 편향을 줄이는 데 더 효과적이며, 독성 감지기의 개선이 전사 파이프라인(transcription pipelines)보다 더 유용하다는 것이 밝혀졌습니다. 연구에서는 주석을 공개하고 향후 독성 데이터셋 구축에 대한 권장 사항을 제시했습니다.



### Can sparse autoencoders be used to decompose and interpret steering vectors? (https://arxiv.org/abs/2411.08790)
- **What's New**: 본 논문에서는 steering vectors의 해석을 위해 sparse autoencoders (SAEs)를 적용하는 것의 한계점을 조사했습니다. 특히 SAEs가 직접적으로 steering vectors에 적용될 경우 잘못된 결과를 나타내는 두 가지 이유를 식별했습니다: (1) steering vectors가 SAEs가 설계된 입력 분포 밖에 위치하고, (2) steering vectors가 SAEs가 수용할 수 없는 의미 있는 음수 프로젝션을 가질 수 있습니다.

- **Technical Details**: steering vectors는 모델의 행동을 조정하기 위해 중간 모델 활성화에 추가되는 벡터 표현입니다. 반면, sparse autoencoders (SAEs)는 모델 활성화의 희소한 비음수 선형 조합으로 분해하는 방법입니다. 본 연구는 SAEs 적용 시 steering vectors의 왜곡된 분해를 일으키는 원인들을 경험적 실험을 통해 검토합니다. 특히, SAEs의 입력 분포 및 음수 프로젝션 처리 능력의 한계가 문제로 지적됩니다.

- **Performance Highlights**: 이 연구에서 제안된 내용을 통해, SAEs가 steering vectors에 대한 해석에서 직접적인 사용에 있어 제약이 있음을 강조하고 있습니다. 이는 향후 steering vectors의 해석 방법 개선을 위한 기초 자료를 제공할 것으로 기대됩니다.



### Theoretical Analysis of Byte-Pair Encoding (https://arxiv.org/abs/2411.08671)
- **What's New**: 이 논문은 Byte-Pair Encoding (BPE)의 최적 압축 유틸리티를 달성하는 쌍 인코딩 문제에 대한 연구를 다룹니다. BPE가 APX-complete라는 것을 보여주며, 효율성을 입증합니다.

- **Technical Details**: BPE는 입력 텍스트에서 가장 많이 발생하는 기호 쌍을 반복적으로 식별하여 새로운 기호로 대체하는 토큰화 방법입니다. 이 과정에서 고정된 수의 기호를 생성하며, s(입력 문자열)와 k(쌍 인코딩 라운드 수)를 입력으로 받아 최적의 압축을 목표로 합니다.

- **Performance Highlights**: BPE는 최적 쌍 인코딩의 압축 유틸리티를 $0.333$에서 $0.625$ 사이의 최악의 경우 비율로 근사합니다. 이는 BPE와 다른 알고리즘에 대한 상수 근사 보장이 부족했던 문제를 해결합니다.



### XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL (https://arxiv.org/abs/2411.08599)
- **What's New**: XiYan-SQL은 자연어(Natural Language) 쿼리를 SQL(Structured Query Language)로 변환하는 기술을 혁신적으로 향상시키기 위해 다중 생성기 앙상블(multi-generator ensemble) 전략을 사용하는 새로운 프레임워크입니다. 또한 M-Schema라는 반구조적(schema representation method) 데이터베이스 구조 접근 방식을 도입하여 데이터베이스 이해를 증진합니다.

- **Technical Details**: XiYan-SQL은 ICL(In-Context Learning)의 잠재력을 활용하면서 감독학습(Supervised Fine-Tuning)으로 후보 SQL 쿼리의 품질과 다양성을 높입니다. 모델의 세련된 학습 방법으로 다단계(multi-task) 접근 방식을 통해 SQL 생성 능력과 다양한 스타일적 선호를 증진합니다. 또한, M-Schema를 통해 데이터베이스의 계층적(structural) 구조를 보다 명확하게 표현합니다.

- **Performance Highlights**: XiYan-SQL은 Spider 테스트 세트에서 89.65%, SQL-Eval에서 69.86%, NL2GQL에서 41.20%의 실행 정확도를 기록하며, Bird 개발 벤치마크에서는 72.23%의 경쟁력 있는 성능을 보여줍니다. 이러한 성과는 다양한 벤치마크에서 제안된 방법의 효과성을 입증하며, 자연어에서 SQL로의 변환 작업에 대한 보다 넓은 응용 잠재력을 보여줍니다.



### An Information Theoretic Approach to Operationalize Right to Data Protection (https://arxiv.org/abs/2411.08506)
Comments:
          First two authors contributed equally to this work

- **What's New**: 본 연구는 데이터 보호 법규(GDPR) 준수의 문제를 해결하기 위해 자연어 데이터셋에 무시할 수 있는 잡음을 주입하는 RegText라는 새로운 프레임워크를 소개합니다.

- **Technical Details**: RegText는 자연어 데이터셋에 무시할 수 있는 가짜 상관관계를 주입하여 언어 모델이 이를 학습할 수 없도록 만듭니다. 이는 기존의 이미지 기반 접근 방식과는 달리 의미론적 내용에는 영향을 미치지 않습니다.

- **Performance Highlights**: RegText를 사용했을 때 GPT-4o 및 Llama와 같은 최신 모델이 생성된 데이터에서 학습하지 못하게 하여 테스트 정확도가 하락했습니다. 이로 인해 제로-샷(Zero-shot) 성능보다 낮은 결과를 나타냈습니다.



### Towards Evaluating Large Language Models for Graph Query Generation (https://arxiv.org/abs/2411.08449)
Comments:
          Paper accepted and will be presented at CSCI2024 in December 2024, Later will be published at Springer LNCS

- **What's New**: 본 논문은 LLM(대형 언어 모델)이 그래프 데이터베이스 및 지식 그래프(KGs)에서 Cypher 쿼리 생성을 위한 도전에 대한 비교 연구를 제시합니다. 특히, 개방 접근(Large Language Models) LLM을 활용한 혁신적인 해결책들이 빠르게 등장하고 있는 가운데, 이 분야에서는 연구가 상대적으로 미비하다는 점을 강조합니다.

- **Technical Details**: 이 연구에서는 OpenAI의 ChatGPT 4o, Claude Sonnet 3.5, Google의 Gemini Pro 1.5, 그리고 로컬에 배포된 Llama 3.1 8B와 같은 여러 LLM 에이전트를 평가합니다. 이러한 평가에는 설계된 few-shot learning 프롬프트(few-shot learning prompt)와 Chain-of-Thoughts(CoT) 추론을 지원하는 Retrieval Augmented Generation(RAG)이 사용됩니다.

- **Performance Highlights**: 실증 분석 결과, Claude Sonnet 3.5가 그래프 데이터베이스에서 쿼리 생성 정확도 측면에서 다른 모델들보다 우수하다는 것을 보여주었습니다. 또한, 앞으로의 연구 방향으로는 LLM 기반의 그래프 데이터베이스 쿼리 생성에서 식별된 한계점을 해결하고 발전시킬 수 있는 가능성을 강조합니다.



### A Chinese Multi-label Affective Computing Dataset Based on Social Media Network Users (https://arxiv.org/abs/2411.08347)
- **What's New**: 이 연구는 중국어 감정 데이터세트를 수집하고, 개인의 성격 특성과 감정을 통합한 다중 라벨 감정 데이터세트를 구축했습니다. 기존 데이터세트는 감정과 성격 특성을 별도로 주석을 달아 왔으며, 미세 감정(micro-emotions)과 감정 강도의 세밀한 라벨링이 부족했습니다.

- **Technical Details**: 연구진은 중소형 SNS 플랫폼인 위보(Weibo)에서 50,000명 이상의 개인 중 11,338명의 유효 사용자를 선별하여 이들의 MBTI 성격 라벨과 함께 566,900개의 게시글을 수집했습니다. EQN 방법론을 사용하여 같은 사용자의 MBTI 성격 특성을 6가지 감정(emotions)과 미세 감정(micro-emotions)과 통합하여 주석이 달린 감정 데이터세트를 작성했습니다.

- **Performance Highlights**: 다양한 NLP(classification) 모델을 통한 검증 결과, 이 데이터세트는 복잡한 인간 감정의 기계 인식을 진전시킬 수 있는 강력한 유용성을 입증했습니다. 이 데이터세트는 심리학, 교육, 마케팅, 금융 및 정치 연구에 자료 지원을 제공하는 데 목적이 있습니다.



### A Large-Scale Study of Relevance Assessments with Large Language Models: An Initial Look (https://arxiv.org/abs/2411.08275)
- **What's New**: 이 논문은 TREC 2024 RAG Track에서의 대규모 평가 결과를 보고하며, LLMs(large language models)를 활용한 네 가지 관련성 평가 접근 방식을 비교합니다. 특히, 자동 생성된 평가 결과가 수동 평가와 높은 상관관계를 나타내는지를 분석했습니다.

- **Technical Details**: 논문에서는 UMBRELA 도구를 사용하여 LLMs가 자동으로 생성한 관련성 평가가 기존의 수동 평가와 비교할 때 nDCG@20, nDCG@100 및 Recall@100에서 높은 상관관계를 보임을 발견했습니다. 세 가지 LLM 참여 수준(완전 자동, 수동 후 편집, 수동 필터링)을 비교하여 비용과 품질 간의 상trade-offs를 분석하였습니다.

- **Performance Highlights**: 자동 생성된 UMBRELA 평가가 수동 평가로부터 유도된 시스템 순위와 높은 상관도를 보였으며, LLM의 지원이 수동 평가와의 상관관계를 증가시키지 않는 것으로 나타났습니다. 전체적으로, 인간 평가자가 UMBRELA보다 더 엄격하게 관련성 기준을 적용하는 것으로 보입니다.



### Retrieval, Reasoning, Re-ranking: A Context-Enriched Framework for Knowledge Graph Completion (https://arxiv.org/abs/2411.08165)
- **What's New**: 이번 논문에서는 Knowledge Graph Completion (KGC) 작업을 위한 새로운 프레임워크인 KGR3를 제안합니다. KGR3는 Retrieval, Reasoning, Re-ranking 세 가지 모듈로 구성되어 있으며, 기존 방법들이 가진 한계를 극복하고 더 나은 결과를 도출합니다.

- **Technical Details**: KGR3는 KGC 작업을 위해 엔티티 맥락을 활용합니다. Retrieval 모듈에서는 KG에서 지원하는 트리플을 수집하고, Reasoning 모듈에서는 대형 언어 모델(LLM)을 사용하여 잠재적 답변을 생성합니다. 마지막으로 Re-ranking 모듈에서는 후보 답변을 통합하고 LLM을 미세 조정하여 최적의 답변을 선택합니다.

- **Performance Highlights**: KGR3는 FB15k237과 WN18RR 데이터셋에서 12.3% 및 5.6%의 Hits@1을 개선하며, 다양한 KGC 방법들보다 성능이 우수함을 입증합니다. KGR3의 적용을 통해 KGC 작업의 성능이 획기적으로 향상되었습니다.



New uploads on arXiv(cs.IR)

### Rethinking negative sampling in content-based news recommendation (https://arxiv.org/abs/2411.08700)
- **What's New**: 이 논문에서는 뉴스 추천 시스템의 모델 정확성을 크게 향상시키기 위해 개인화된 부정 샘플링 기법을 제안합니다.

- **Technical Details**: 부정 샘플링 기법은 사용자 패턴을 학습하기 위한 더 나은 암묵적 부정 예제를 제공하며, 각 사용자에 대해 별도의 경량화된 신경 추천기를 훈련하는 분산 학습 전략을 제안합니다. 이는 사용자의 개인 정보 보호 및 자율성을 개선할 수 있게 합니다.

- **Performance Highlights**: MIND 데이터셋을 사용한 실험 결과, 제안된 방법의 정확성이 최첨단(State-of-the-Art) 모델과 경쟁할 수 있음을 보여줍니다. 또한, 이 샘플링 기법은 모델 복잡성을 줄이고 훈련 과정을 가속화하며, 높은 정확도를 유지할 수 있게 합니다.



### Neural Corrective Machine Unranking (https://arxiv.org/abs/2411.08562)
Comments:
          submitted to Information Sciences

- **What's New**: 본 연구에서는 신경 정보 검색(neural information retrieval, IR) 시스템에서 데이터 제거(machine unlearning)의 필요성을 다룹니다. 기존의 기법들이 검색 효과성을 저하시킬 수 있는 문제를 해결하기 위해 corrective unranking을 정식화하였으며, 새로운 teacher-student 프레임워크인 Corrective unRanking Distillation (CuRD)을 제안합니다.

- **Technical Details**: CuRD는 (1) 잊혀져야 하는 샘플의 출력 관련 점수를 순위가 낮은 비검색 샘플의 점수와 유사하게 조정하여 잊기(positional forgetting)를 촉진합니다; (2) 대체 샘플에 대한 관련 점수를 수정하여 이에 해당하는 잊혀져야 하는 샘플의 점수와 밀접하게 일치하도록 세분화합니다; (3) 잊기를 목표로 하지 않는 샘플들의 성능을 유지합니다.

- **Performance Highlights**: CuRD는 BERTcat, BERTdot, ColBERT, PARADE의 네 가지 신경 IR 모델을 사용하여 MS MARCO와 TREC CAR 데이터셋에서 평가되었습니다. 실험 결과, CuRD는 잊기와 수정 면에서 7개의 최첨단 기법을 웃도는 성과를 보였으며, 모델의 유지 및 일반화 능력을 유지했습니다.



### A Large-Scale Study of Relevance Assessments with Large Language Models: An Initial Look (https://arxiv.org/abs/2411.08275)
- **What's New**: 이 논문은 TREC 2024 RAG Track에서의 대규모 평가 결과를 보고하며, LLMs(large language models)를 활용한 네 가지 관련성 평가 접근 방식을 비교합니다. 특히, 자동 생성된 평가 결과가 수동 평가와 높은 상관관계를 나타내는지를 분석했습니다.

- **Technical Details**: 논문에서는 UMBRELA 도구를 사용하여 LLMs가 자동으로 생성한 관련성 평가가 기존의 수동 평가와 비교할 때 nDCG@20, nDCG@100 및 Recall@100에서 높은 상관관계를 보임을 발견했습니다. 세 가지 LLM 참여 수준(완전 자동, 수동 후 편집, 수동 필터링)을 비교하여 비용과 품질 간의 상trade-offs를 분석하였습니다.

- **Performance Highlights**: 자동 생성된 UMBRELA 평가가 수동 평가로부터 유도된 시스템 순위와 높은 상관도를 보였으며, LLM의 지원이 수동 평가와의 상관관계를 증가시키지 않는 것으로 나타났습니다. 전체적으로, 인간 평가자가 UMBRELA보다 더 엄격하게 관련성 기준을 적용하는 것으로 보입니다.



### Scholarly Wikidata: Population and Exploration of Conference Data in Wikidata using LLMs (https://arxiv.org/abs/2411.08696)
Comments:
          17 pages, accepted at EKAW-24

- **What's New**: 이 논문은 학술 데이터의 접근성을 높이기 위해 Wikidata의 인프라를 활용하고, 대규모 언어 모델(LLM)을 통해 학회 메타데이터를 자동으로 추출하는 방법론을 제안합니다. 이로 인해 학술 데이터를 더욱 지속 가능하게 유지할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 우리는 학술 데이터를 표현하기 위한 존재론(ontologies)을 분석하고, Wikidata의 주요 개체와 속성으로 매핑합니다. LLM을 활용하여 학회 메타데이터를 효율적으로 추출하고, OpenRefine를 통한 사람 검증(Human-in-the-loop validation) 과정을 통해 데이터를 정제하여 Wikidata에 입력합니다. 이 방법론을 통해 총 6000개 이상의 새로운 개체가 Wikidata에 추가되었습니다.

- **Performance Highlights**: 이 연구는 105개의 의미 웹(Semantic Web) 관련 학회의 데이터를 Wikidata에 추가하여 기존의 데이터를 확장하고, 시각화 도구(Scholia 및 Synia)를 개선하여 새로운 정보를 보다 효율적으로 탐색할 수 있게 합니다. LLM을 활용한 데이터 추출과 검증 과정을 통해 신뢰도를 향상시킵니다.



### Enhancing Multimodal Query Representation via Visual Dialogues for End-to-End Knowledge Retrieva (https://arxiv.org/abs/2411.08334)
- **What's New**: 본 논문에서는 기존의 분리된 모델에 의존하는 다중 모달(multi-modal) 검색 시스템의 한계를 극복하기 위해, Ret-XKnow라는 엔드 투 엔드(end-to-end) 검색 시스템을 제안합니다. 이 시스템은 텍스트 검색기에 다중 모달 쿼리를 이해하는 능력을 부여하며, 비주얼 정보와 텍스트 쿼리 간의 동적 상호작용을 통해 성능을 향상시킵니다.

- **Technical Details**: Ret-XKnow는 부분 합성곱(partial convolution) 메커니즘을 사용하여 주어진 텍스트 쿼리와 관련된 비주얼 정보에 집중합니다. 이를 통해 비주얼 임베딩(visual embeddings)을 압축하고, 텍스트 쿼리 표현과의 관련성 점수를 적응형 마스크로 활용하여 시각적 정보의 중요성을 강조합니다. 또한, Visual Dialogue-to-Retrieval (ViD2R) 데이터셋을 도입하여, 시각적 대화 데이터셋으로부터 자동으로 생성된 데이터셋을 통해 다중 모달 상호작용을 효과적으로 학습합니다.

- **Performance Highlights**: Ret-XKnow는 사전 훈련(pre-training) 데이터셋으로 ViD2R를 사용하여 네 가지 다중 모달 데이터셋에서 제로샷(zero-shot) 검색 성능의 유의미한 향상을 보여주며, 추가적인 모델 조정 없이도 높은 차세대 검색 성능을 입증합니다. 이는 Ret-XKnow 모델이 기존의 다양한 기준선(baseline) 방법들과 비교하여 우수한 결과를 도출했음을 나타냅니다.



New uploads on arXiv(cs.CV)

### 4D Gaussian Splatting in the Wild with Uncertainty-Aware Regularization (https://arxiv.org/abs/2411.08879)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 4D Gaussian Splatting (4DGS)이라는 새로운 알고리즘을 제안하여, casual monocular 동영상에서 역동적인 장면을 재구성하는 방법을 소개합니다. 이 알고리즘은 오버피팅(overfitting) 문제를 해결하기 위해 불확실성을 인지하는 정규화(regularization)를 도입합니다.

- **Technical Details**: 본 연구에서는 우선 Gaussian primitive의 불확실성을 정량화하고, α-blending 방법을 통해 2D 불확실성 맵을 작성하여 선택적으로 정규화를 적용합니다. 또한, Structure from Motion (SfM) 알고리즘의 초기화 문제를 해결하기 위해 동적 지역 밀집화(dynamic region densification) 방법을 사용하여 깊이 맵과 장면 흐름(scene flow)을 기반으로 Gaussian primitive를 초기화합니다.

- **Performance Highlights**: 제안된 방법은 handheld monocular 카메라로 촬영한 영상에서 4DGS 재구성 성능을 향상시키며, 소수의 샷(few-shot) 정적 장면 재구성에서도 유망한 결과를 보여줍니다.



### A Short Note on Evaluating RepNet for Temporal Repetition Counting in Videos (https://arxiv.org/abs/2411.08878)
- **What's New**: 이번 논문에서는 비디오 반복 카운팅 데이터셋에서 RepNet의 평가에 관한 일관된 문제를 다루고 있습니다. 이를 해결하기 위해 RepNet의 다양한 데이터셋에서의 성능 결과를 보고하고, 평가 코드 및 RepNet 체크포인트를 공개했습니다.

- **Technical Details**: RepNet 모델은 원래 32 프레임 이상을 예측할 수 있으며, 이는 모델 수정 없이 다양한 속도로 비디오를 재생하는 Multi-speed Evaluation 기술로 가능하다는 점을 강조합니다. 평가 지표로 Off-by-one Accuracy (OBOA)와 Mean Absolute Error (MAE)를 사용하며, 다양한 데이터셋인 Countix, UCFRep, RepCount-A에서 RepNet의 성능 결과를 보고합니다.

- **Performance Highlights**: RepNet은 Countix 데이터셋에서 최신 방법보다 뛰어난 성능을 보였으며, UCFRep에서도 TransRAC보다 유의미하게 더 나은 성능을 기록했습니다. RepCount-A 데이터셋에서도 반복 구간이 있는 비디오에 대해 효과적으로 작동하며, RepNet의 기본 성능이 과거 수정된 모델보다 훨씬 강력함을 보여주었습니다.



### Multimodal Instruction Tuning with Hybrid State Space Models (https://arxiv.org/abs/2411.08840)
- **What's New**: 본 논문은 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 긴 컨텍스트 처리 문제를 해결하기 위한 하이브리드 트랜스포머-마바(Transformer-MAMBA) 모델인 MMJamba를 소개합니다.

- **Technical Details**: MMJamba 모델은 고해상도 이미지와 고프레임 비디오에서 100k 이상의 토큰을 효율적으로 처리할 수 있는 기능을 갖추고 있습니다. 이 모델은 현재의 오픈소스 모델들보다 약 4배 더 빠른 추론 효율성을 보여주며, 이 해상도에서 효율성이 증가하는 것으로 입증되었습니다. 또한, MMJamba는 짧은 컨텍스트로 훈련하고 긴 컨텍스트에서 추론하는 전략을 통해 효율성과 효과성을 동시에 개선합니다.

- **Performance Highlights**: MMJamba는 18개의 벤치마크 데이터셋에서 실험한 결과, LLaVA-NeXT 및 Gemini Pro 1.0과 비교할 때 최첨단 성능을 달성하였고, 특정 경우에는 GPT-4V와 유사한 성능을 발휘합니다. 특히 고해상도 이미지와 고프레임 비디오에서 처리 효율성이 가장 뛰어난 것으로 나타났습니다.



### Sharingan: Extract User Action Sequence from Desktop Recordings (https://arxiv.org/abs/2411.08768)
- **What's New**: 본 논문은 데스크탑 비디오 기록에서 사용자 행동을 추출하기 위한 두 가지 새로운 Vision-Language Model (VLM) 기반 방법을 제안합니다: Direct Frame-Based Approach (DF)와 Differential Frame-Based Approach (DiffF)입니다.

- **Technical Details**: DF 접근법은 샘플링한 비디오 프레임을 VLM에 직접 입력하는 방식이며, DiffF는 컴퓨터 비전 기법을 통해 감지된 프레임 차이를 명시적으로 포함한 후 VLMs로 해석합니다. 이 연구는 각 접근법의 성능을 평가지원 데이터 세트를 사용하여 평가합니다.

- **Performance Highlights**: DF 접근법은 사용자 행동 식별에서 70%에서 80%의 정확도를 달성했습니다. 추출된 행동 시퀀스는 Robotic Process Automation (RPA) 프로세스를 통해 재생할 수 있습니다. 명시적인 UI 변화가 성능 저하를 가져올 수 있으므로 DF 접근법이 보다 신뢰할 수 있는 것으로 판단됩니다.



### Masked Image Modeling Boosting Semi-Supervised Semantic Segmentation (https://arxiv.org/abs/2411.08756)
Comments:
          13 pages. This work has been submitted to the IEEE for possible publication

- **What's New**: 본 연구에서는 masked image modeling (MIM) 기법을 통한 새로운 class-wise 접근법을 제안하여 세미-슈퍼바이즈드(semi-supervised) 의미 분할 성능을 향상시켰습니다. 이를 통해 각각의 클래스에 따른 이미지 영역을 독립적으로 복원하며, 각 클래스 내부의 연결성을 강화하였습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 요소로 구성됩니다. 첫째, class-wise masked image modeling을 통해 각 클래스에 따라 독립적으로 이미지를 재구성하며, 둘째, feature aggregation 전략을 통해 동일 클래스 내에서 masked 및 visible 부분 간의 거리를 최소화합니다. 이를 통해 class 간의 혼동을 줄이고 연결성을 강화하게 됩니다.

- **Performance Highlights**: PASCAL VOC 2012 및 Cityscapes 데이터셋에 대한 실험 결과, 제안하는 S4MIM 접근법이 기존의 state-of-the-art 성능을 능가하는 것을 입증하였습니다. 각 구성 요소의 효과를 검증하여 MIM이 세미-슈퍼바이즈드 의미 분할을 더욱 향상시킬 수 있음을 보여주었습니다.



### Weakly-Supervised Anomaly Detection in Surveillance Videos Based on Two-Stream I3D Convolution Network (https://arxiv.org/abs/2411.08755)
Comments:
          11 pages, 8 figures

- **What's New**: 본 논문은 두 개의 스트림 Inflated 3D (I3D) Convolutional Network를 활용하여 비디오 이상 감지(anomaly detection) 분야의 기술을 혁신적으로 발전시킨 내용을 다룹니다. 이 접근 방식은 기존의 3D Convolutional Networks (C3D)보다 공간적(spatial) 및 시간적(temporal) 특성을 더 효과적으로 추출하여 정확도를 향상시킵니다. 또한, 다중 인스턴스 학습(Multiple Instance Learning, MIL)을 바탕으로 한 약한 감독 학습(weakly supervised learning) 프레임워크를 적용하여 주목받고 있습니다.

- **Technical Details**: 제안된 접근 방식은 비디오에서 RGB(밝기)와 Optical Flow(광학 흐름) 특징을 각각 추출하는 두 스트림 I3D Convolutional Neural Network를 사용합니다. 이 모델은 비디오의 각 클립을 '가방(bag)'으로 간주하고, 클립의 이상 가능성에 따라 순위를 매기는 혁신적인 메커니즘을 적용합니다. 이러한 방법론은 모델 설정의 세심한 최적화를 통해 새로운 성능 기준을 설정합니다.

- **Performance Highlights**: UCF-Crime 데이터셋을 기반으로 한 평가에서, 제안된 모델의 성능은 기존의 방법들에 비해 뛰어난 이상 감지 능력을 보여주었습니다. 이 시스템은 Urban Surveillance와 같은 실세계 응용 프로그램에서 적용 가능하며, 수작업 주석에 대한 의존도를 크게 줄이면서 정확성 및 신뢰성 향상을 제공합니다.



### Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Videos (https://arxiv.org/abs/2411.08753)
- **What's New**: 본 논문은 다중 시점 비디오에서 인간 관찰자에게 가장 정보가 풍부한 시점을 자동으로 선택하는 새로운 약한 감독 접근 방식을 제안합니다. 기존 방법들이 비싼 감독(supervision)이나 휴리스틱에 의존하는 반면, 본 연구에서는 언어적 설명을 활용하여 가장 유익한 시점을 회복합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 요소로 구성되어 있습니다: 베스트 뷰 의사 라벨러(pseudo-labeler)와 베스트 뷰 선택기(selector)입니다. 베스트 뷰 의사 라벨러는 다중 시점 비디오에 대해 훈련 중에 의사 라벨을 자동 생성하며, 선택기는 입력으로 다중 시점 비디오를 받아 최선의 시점 라벨을 예측합니다. 또한, 선택기는 서로 다른 뷰 간의 상대적인 카메라 포즈를 예측하여 시점 민감도를 향상시킵니다.

- **Performance Highlights**: 제안된 모델은 두 개의 도전적인 데이터셋 EGO-EXO4D와 LEMMA에서 평가되었으며, 이 두 데이터셋 모두 다양한 활동 시나리오와 멀티 카메라 설정을 포함하고 있습니다. 우리의 모델은 여러 자동 및 인간 평가 지표에서 여러 베이스라인 및 최첨단 방법을 지속적으로 능가하는 성능을 보였습니다.



### Retrieval Augmented Recipe Generation (https://arxiv.org/abs/2411.08715)
Comments:
          ACCEPT on IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문은 요리 이미지를 기반으로 레시피를 생성하기 위한 처음의 검색 보강 대형 다중 모달 모델(retrieval-augmented large multimodal model)을 제안합니다. Stochastic Diversified Retrieval Augmentation (SDRA)라는 새로운 방법을 도입하여 다양한 레시피를 검색하고 갱신된 정보로 모델의 성능을 향상시킵니다.

- **Technical Details**: 제안된 구조는 검색자(retriever)와 생성자(generator)로 구성됩니다. 검색자는 크로스 모달 레시피 검색 모델을 활용하여 이미지와 의미적으로 관련된 레시피를 식별합니다. 생성자는 LLaVA 기반으로 구성되며, 다양한 참조 레시피를 사용하여 레시피를 생성합니다. 이 과정에서는 self-consistency ensemble voting 전략을 통해 생성된 후보 레시피의 일관성을 평가하며, 가장 높은 점수를 가진 레시피가 최종 출력으로 선택됩니다.

- **Performance Highlights**: Recipe1M 데이터세트에서 제안된 모델은 기존 방법과 미세 조정된 LLaVA를 능가하는 SOTA(최첨단) 성능을 보여주었습니다. 또한, 이 모델은 원재료 인식 측면에서도 뛰어난 일반성을 입증하며 현재 SOTA 결과를 초과했습니다.



### High-resolution optical and acoustic remote sensing datasets of the Puck Lagoon, Southern Baltic (https://arxiv.org/abs/2411.08712)
- **What's New**: 이 논문은 폴란드 북부 발트해의 푸크 라군(Puck Lagoon) 지역에 대한 고해상도 원격 감지 데이터의 첫 번째 활용을 다룹니다. 이러한 데이터는 이전에 이용 가능한 데이터가 없었던 이 지역의 해양 생태계와 문화유산을 보호하는 데 기초가 됩니다.

- **Technical Details**: 본 연구에서는 항공 배수 LiDAR, 멀티빔 에코사운더, 항공 사진 측량 및 위성 이미지를 조합하여 생성한 디지털 고도 모델(Digital Elevation Models, DEMs)을 설명합니다. 이 데이터 세트는 멀티빔 에코사운더의 백산란(backscatter) 및 LiDAR 강도(intensity)를 포함하여 해저의 특성과 성질을 결정할 수 있게 합니다.

- **Performance Highlights**: 푸크 라군의 하천학적(hydrological), 생태학적(ecological), 지질학적(geological), 고고학적(archaeological) 측면에서 이 지역을 이해하고 야기되는 지속 가능성 관리(sustainable management)를 위한 중요한 자료를 제공합니다.



### TRACE: Transformer-based Risk Assessment for Clinical Evaluation (https://arxiv.org/abs/2411.08701)
- **What's New**: 이번 연구에서는 TRACE(Transformer 기반 임상 평가 리스크 평가)라는 신규 방법을 제안합니다. 이는 임상 데이터를 기반으로 리스크 평가를 진행하며, self-attention 메커니즘을 활용하여 특징 상호작용과 결과 해석을 개선합니다.

- **Technical Details**: 제안된 아키텍처는 다양한 데이터 모드인 연속형, 범주형 및 체크박스 속성을 처리할 수 있으며, 각 데이터 모드의 특화된 임베딩을 통합하여 임상 데이터의 공유 표현을 얻습니다. 이를 통해 Transformer encoder 레이어를 사용하여 고위험 대상을 탐지할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 비정형 다층 perceptron(MLP)을 기반으로 한 강력한 기준선을 설정하고, 다양한 임상 리스크 평가에 널리 사용되는 여러 기준선들을 초월하며, 누락된 값을 효과적으로 처리했습니다. 성능 측면에서도 제안된 모델은 경쟁력 있는 성과를 나타내며, clinicians이 모델 결과를 쉽게 해석할 수 있는 설명 가능성을 제공합니다.



### A Survey on Vision Autoregressive Mod (https://arxiv.org/abs/2411.08666)
- **What's New**: 이 논문은 자연어 처리(NLP)에서 성공을 거둔 autoregressive (AR) 모델을 컴퓨터 비전에 적용하려는 최근의 연구를 다룹니다. 새로운 비전 과제, 특히 이미지 생성 및 이해, 다중 모달(multi-modal) 생성과 같은 다양한 비전 작업에 적용된 AR 모델에 대한 체계적인 리뷰를 제공합니다.

- **Technical Details**: AR 모델은 이미지 데이터를 비주얼 토큰(visual tokens)으로 표현하여 다음 토큰 예측(next-token prediction)을 수행합니다. 이는 비전 생성 및 이해를 통합하려는 시도로, Transformer 아키텍처를 기반으로 큰 텍스트-이미지 쌍에서 학습할 수 있게 합니다. 또한, AR 모델은 비디오 생성, 의학 이미지 분석 등 다양한 비전 작업에도 적용되고 있습니다.

- **Performance Highlights**: 논문은 AR 모델의 성능을 벤치마킹하고 다양한 평가 데이터셋에서 기존 방법을 분석합니다. 또한, AR 모델을 활용하여 비전 과제를 수행하는 데 있어 주요 도전 과제와 향후 연구 방향을 제시하며, AR 모델이 비전의 다양한 작업에서 잠재력을 가지고 있음을 강조합니다.



### OSMLoc: Single Image-Based Visual Localization in OpenStreetMap with Geometric and Semantic Guidances (https://arxiv.org/abs/2411.08665)
Comments:
          15 pages, technical report

- **What's New**: OpenStreetMap (OSM)의 데이터를 활용하여 로봇의 이미지-OSM(I2O) 매칭과 로컬라이제이션을 개선하는 OSMLoc 프레임워크를 제안합니다. 이 방법은 인간의 지각 방식을 모방하여 기하학적 및 의미론적 정보를 통합하여 정확도를 높입니다.

- **Technical Details**: OSMLoc는 DINOv2 인코더를 활용하여 강력한 이미지 특징을 추출하고, 깊이 추정 및 카메라-비즈 아이 뷰(BEV) 변환을 위한 깊이 분포 어댑터를 사용합니다. 또한, OSM 데이터의 의미론적 임베딩을 보조 가이드로 활용하여 I2O 기능 매칭을 개선합니다.

- **Performance Highlights**: 실험 결과, OSMLoc은 MGL 데이터셋, CC 검증 벤치마크, KITTI 데이터셋에서 기존 방법론을 초과하는 성능을 보였으며, 특히 보지 못한 지역에서 우수한 결과를 나타냈습니다.



### Toward Human Understanding with Controllable Synthesis (https://arxiv.org/abs/2411.08663)
- **What's New**: 논문에서는 Generative BEDLAM (Gen-B) 데이터셋을 통해 기존의 합성 이미지 데이터와 정확한 ground truth를 효과적으로 결합하는 방법을 제시합니다. 이는 3D 인간 포즈 및 형태 (HPS) 추정의 정확성을 크게 향상시킬 수 있는 새로운 접근법입니다.

- **Technical Details**: 저자들은 기존의 BEDLAM 데이터셋에 대해 변형된 Stable Diffusion 모델을 적용하여 2D 키포인트, 깊이 맵, 표면 법선 등을 조건으로 사용하여 합성 과정을 개선합니다. 다양한 노이즈 조절 전략을 통해 이미지의 시각적 현실성을 높이는 동시에 정확한 ground truth와의 정렬을 유지합니다.

- **Performance Highlights**: Gen-B 데이터셋으로 훈련된 HPS 네트워크가 실제 테스트 데이터셋에서 평균 2.37%, 4.66%, 1.95%의 향상을 보였으며, CLIFF 모델의 경우 모든 데이터셋에서 BEDLAM보다 낮은 오류를 기록했습니다. transformer 기반의 최신 아키텍처에서도 Gen-B로 훈련된 HMR2.0은 RICH 데이터셋에서 2.26%의 오류 감소를 보여주는 등 전반적인 성능 향상이 입증되었습니다.



### MikuDance: Animating Character Art with Mixed Motion Dynamics (https://arxiv.org/abs/2411.08656)
- **What's New**: 이번 논문에서는 MikuDance라는 새로운 diffusion 기반 파이프라인을 제안하며, 혼합된 동작 동역학(mixed motion dynamics)을 활용해 스타일화된 캐릭터 아트를 애니메이션할 수 있는 기능을 소개합니다. MikuDance는 캐릭터 아트 애니메이션의 고유한 도전 과제를 해결하기 위해 Mixed Motion Modeling과 Mixed-Control Diffusion의 두 가지 핵심 기술을 통합합니다.

- **Technical Details**: MikuDance는 Scene Motion Tracking 전략을 통해 픽셀 수준에서 동적 카메라를 명시적으로 모델링하여 캐릭터와 배경의 통합 동작 모델링을 가능하게 합니다. Mixed-Control Diffusion은 다양한 캐릭터의 스케일과 신체 형태를 동작 가이드와 정렬시켜 개별 캐릭터 동작에 대한 유연한 제어를 가능하게 합니다. 또한 Motion-Adaptive Normalization 모듈을 통해 전역 장면 동작을 효과적으로 주입합니다.

- **Performance Highlights**: 다양한 캐릭터 아트와 동작 가이드를 활용한 실험을 통해, MikuDance는 캐릭터의 지역 동작 일관성을 유지하면서 고품질 애니메이션을 생성하는 데 있어 우수성을 입증했습니다. MikuDance는 기존의 최첨단 방법과 비교하여 뛰어난 애니메이션 품질과 고동적인 동작 제어를 달성했습니다.



### Towards More Accurate Fake Detection on Images Generated from Advanced Generative and Neural Rendering Models (https://arxiv.org/abs/2411.08642)
Comments:
          13 pages, 8 Figures

- **What's New**: 이번 연구에서는 Neural Radiance Fields와 3D Gaussian splatting과 같은 신경망 기반의 시각 데이터 생성 기술이 GANs와 diffusion 모델의 강력한 대안이 될 수 있음을 보여줍니다. 특히, 강력한 탐지 방법의 필요성을 강조하며 자율적으로 학습할 수 있는 새로운 기술을 제안합니다.

- **Technical Details**: 제안된 접근법은 Fourier 스펙트럼의 크기에서 포괄적인 특징을 추출하는 비지도 학습(unsupervised training) 기술을 포함합니다. 스펙트럼의 중심 대칭 성질로 인한 재구축 문제를 극복하기 위해 스펙트럼 영역과 공간 영역 정보를 동적으로 결합하여 강력한 다중 모드 탐지기(multimodal detector)를 생성합니다.

- **Performance Highlights**: 이 다중 모드 탐지기는 최신 이미지 합성 기법으로 생성된 도전적인 합성 이미지들을 식별하는 데 있어 뛰어난 일반화 능력을 보입니다. 또한, 3D 신경 렌더링 기반의 가짜 이미지 데이터베이스가 부족한 문제를 해결하기 위해 다양한 신경 렌더링 기술로 생성된 이미지를 포함하는 포괄적인 데이터베이스를 개발하였습니다.



### Zero-shot capability of SAM-family models for bone segmentation in CT scans (https://arxiv.org/abs/2411.08629)
- **What's New**: 이번 연구에서는 Segment Anything Model(SAM) 및 유사 모델들이 의료 영상에서의 이미지 및 비디오 세분화에 응용되는 방안을 탐구하고 있습니다.

- **Technical Details**: 연구는 CT 스캔에서 뼈 세분화를 위한 비효율적인 non-iterative, 'optimal' prompting 전략을 평가하며, 이러한 전략은 bounding box와 points 및 그 조합으로 구성됩니다. 또한, SAM 계열 모델의 zero-shot capability를 다섯 가지 뼈 구조에서 평가하였습니다.

- **Performance Highlights**: 결과에 따르면, SAM과 SAM2 모델이 object의 모든 구성 요소에 대해 중앙점을 가진 bounding box로 prompt되는 경우가 가장 우수한 성능을 보였습니다. 성능은 모델 유형, 크기, 데이터셋 특성 및 최적화 목표에 의존합니다. 이에 따라 2D prompting에 대한 신뢰할 수 있는 의사결정을 위한 가이드를 제공합니다.



### LG-Gaze: Learning Geometry-aware Continuous Prompts for Language-Guided Gaze Estimation (https://arxiv.org/abs/2411.08606)
Comments:
          Accepted to ECCV 2024

- **What's New**: 본 논문은 시선 추정(gaze estimation) 작업을 비전-언어 정렬(vision-language alignment) 문제로 재구성하는 새로운 접근 방식을 제안합니다. 이 프레임워크는 언어 지침 기반 시선 추정(Language-Guided Gaze Estimation, LG-Gaze)이라 명명되었으며, 기존의 문제를 해결하기 위해 심층적인 시멘틱 정보(semantic information)를 활용합니다.

- **Technical Details**: LG-Gaze 프레임워크는 지속적이고 기하학적으로 민감한(sensitive) 특징들을 학습하며, 멀티모달 대조 회귀 손실(multimodal contrastive regression loss) 기능을 통해 다양한 부정 샘플(negative samples)에 대해 적응형 가중치(adaptive weights)를 customizing 합니다. 또한, 기하 인지(interpolation-aware) 방법을 통해 정밀한 시선 임베딩을 확보합니다.

- **Performance Highlights**: 다양한 크로스 도메인(cross-domain) 평가 작업에서 LG-Gaze의 효율성을 검증했으며, 기존의 최첨단(domain generalization) 시선 추정 방법들을 초월하는 성과를 달성했습니다.



### Generalized Pose Space Embeddings for Training In-the-Wild using Anaylis-by-Synthesis (https://arxiv.org/abs/2411.08603)
- **What's New**: 이번 연구에서는 pose estimation(포즈 추정) 모델의 성능을 개선하기 위해 기존의 단일 채널 중간 skeleton representation(스켈레톤 표현)을 다채널 방식으로 확장하였습니다. 이를 통해 좌우 전환 문제를 줄이고, 분석-합성(analysis-by-synthesis) 프레임워크와 결합하여 새로운 훈련 프로토콜을 적용하였습니다.

- **Technical Details**: 새로운 다채널 중간 스켈레톤 표현을 구축하여 포즈의 의미를 포착할 수 있게 하였으며, 이 모델은 synthetic data(합성 데이터)를 활용하여 사전 훈련(pre-training)을 진행합니다. 이러한 방법을 통해 저수준의 포즈 예측(flips)을 줄이고 더욱 정확한 예측을 달성할 수 있었습니다. Human3.6M 데이터셋에서 MSE(Mean Squared Error) 성능이 기존의 방법보다 우수함을 보였습니다.

- **Performance Highlights**: 우리의 접근 방식은 기존 모델보다 안정적으로 더 정확한 예측을 수행하며, Human3.6M benchmark(벤치마크)에서 MSE를 6.62로 줄이는데 성공했습니다. 이 과정에서 합성 데이터와 비라벨 실제 데이터를 조합하여 모델의 현실성 차이를 줄이는 노력을 했습니다.



### Slender Object Scene Segmentation in Remote Sensing Image Based on Learnable Morphological Skeleton with Segment Anything Mod (https://arxiv.org/abs/2411.08592)
- **What's New**: 본 논문에서는 심층 신경망에 학습 가능한 형태학적 스켈레톤(prior)을 통합하는 새로운 접근 방식을 제안합니다. 전통적인 형태학적 연산의 비미분 가능성 문제를 해결하기 위해 매끄러운 형태로 스켈레톤을 표현하고, 변분(segmentation) 모델을 설계하였습니다.

- **Technical Details**: 우리는 매끄러운 형태학적  스켈레톤을 기반으로 한 변분(segmentation) 모델을 설계하고, 이를 Threshold Dynamics 모델에 통합합니다. 이 모델은 Douglas-Rachford 분할 방법을 사용해 효율적으로 해결 가능하며, 네트워크 아키텍처에 학습 가능한 형태학적 스켈레톤(prior)을 통합할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 원래 SAM보다 슬렌더 객체 분할에서 뛰어난 성능을 보이며, 전체 기하학적 구조를 보다 잘 보존함을 입증했습니다.



### NavAgent: Multi-scale Urban Street View Fusion For UAV Embodied Vision-and-Language Navigation (https://arxiv.org/abs/2411.08579)
- **What's New**: 본 논문은 Urban UAV Vision-and-Language Navigation(VLN)의 새로운 접근 방식인 NavAgent를 제안합니다. NavAgent는 다중 환경 정보를 융합하여 UAV 에이전트가 복잡한 도시 환경에서 자연어 명령을 통해 자율적으로 탐색할 수 있도록 합니다.

- **Technical Details**: NavAgent는 GLIP를 활용하여 시각 인식을 수행하며, 동적으로 성장하는 장면 위상 맵(scene topology map)을 개발하여 환경 정보를 통합합니다. 이 모델은 글로벌 환경 정보(지형도), 중간 규모의 파노라마, 로컬 세부 랜드마크 정보를 함께 사용하여 에이전트의 탐색 능력을 향상합니다.

- **Performance Highlights**: 테스트 결과, NavAgent는 Touchdown과 Map2seq 데이터셋에서 강력한 기본 모델과 비교해 4.6% 및 2.2%의 성능 개선을 기록했습니다. 특히 NavAgent-Landmark2K 데이터셋을 통해 9.5%의 정확도를 향상시켰습니다.



### UIFormer: A Unified Transformer-based Framework for Incremental Few-Shot Object Detection and Instance Segmentation (https://arxiv.org/abs/2411.08569)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 Transformer 아키텍처를 기반으로 한 새로운 통합 증분 소수 샷 객체 탐지(Incremental Few-shot Object Detection, iFSOD) 및 인스턴스 분할(Instance Segmentation, iFSIS) 프레임워크를 소개합니다. 이 프레임워크는 새로운 객체 클래스의 예시가 몇 개만 주어질 때 최적의 솔루션을 제공하며, 기존 클래스에 대한 데이터에 접근할 수 없는 상황에서도 높은 성능을 유지합니다.

- **Technical Details**: Mask-DINO를 확장하여 두 단계의 증분 학습 프레임워크를 구성합니다. 1단계는 기본 데이터셋을 사용하여 모델을 최적화하고, 2단계는 새로운 클래스에 대해 미세 조정(fine-tuning)을 수행합니다. 또한, 인코더와 디코더의 기능에 따라 적절한 분류기를 할당하는 분류기 선택 전략을 도입하여 새로운 클래스 학습에서의 과적합을 효과적으로 완화합니다. 지식 증류(Knowledge Distillation)를 구현하여 기본 클래스의 끔찍한 망각(catastrophic forgetting)을 방지합니다.

- **Performance Highlights**: COCO와 LVIS 데이터셋에서 수행한 광범위한 평가 결과, 본 방법이 iFSOD 및 iFSIS 작업 모두에서 최신 기법(state-of-the-art)을 상당히 능가하는 성능을 보여주었습니다.



### Saliency Map-based Image Retrieval using Invariant Krawtchouk Moments (https://arxiv.org/abs/2411.08567)
- **What's New**: 최근의 이미지 검색 시스템과 이미지 특징 추출 기술의 발전으로, 본 논문에서는 saliency map 기반의 이미지 검색 접근법인 SM-IKM을 제안하여 검색 속도와 정확성을 높이고자 하였습니다.

- **Technical Details**: SM-IKM은 전역 대비 기반의 특징 영역 감지 알고리즘을 사용하여 saliency map을 만들어 배경으로부터 전경을 효과적으로 분리합니다. 이 방법은 invariant Krawtchouk moments (IKM), local binary patterns (LBPs), 색상 히스토그램을 결합하여 전경과 배경을 종합적으로 표현합니다.

- **Performance Highlights**: 공식적으로 사용 가능한 데이터셋인 Caltech 101 및 Wang에서의 광범위한 실험 결과, SM-IKM이 최근의 최첨단 검색 방법들을 초월함을 입증하였습니다.



### APDDv2: Aesthetics of Paintings and Drawings Dataset with Artist Labeled Scores and Comments (https://arxiv.org/abs/2411.08545)
- **What's New**: APDDv2 데이터셋은 10,023장의 이미지와 90,000건 이상의 주석으로 확대되었으며, 예술 이미지의 미적 속성을 효과적으로 평가하기 위한 종합적인 기준이 개발되었습니다.

- **Technical Details**: 'Aesthetics Paintings and Drawings Dataset (APDD)'는 24개 예술 장르 및 10개 미적 속성을 포함하는 예술 이미지 데이터셋으로, 정교한 주석과 함께 디자인되었습니다. 'ArtCLIP'라는 새로운 평가 네트워크가 개발되어 APDDv2에서 훈련되었습니다.

- **Performance Highlights**: 'ArtCLIP' 모델의 실험적 검증 결과, 기존 방법론보다 높은 정확도를 기록하며 미적 평가 분야에서 개선된 성능을 보였습니다.



### MLV$^2$-Net: Rater-Based Majority-Label Voting for Consistent Meningeal Lymphatic Vessel Segmentation (https://arxiv.org/abs/2411.08537)
Comments:
          ML4H 2024

- **What's New**: 이 논문에서는 meningeal lymphatic vessels (MLVs)의 세그멘테이션을 위한 새로운 rater-aware training 스킴과 ensembling 전략을 제안하고, nnU-Net 모델의 성능을 향상시키는 방법을 탐구합니다. MLV는 노화 및 뇌 질환과 관련이 있으며, 이번 연구에서는 새로운 자동 세그멘테이션 방법인 MLV$^2$-Net을 소개합니다.

- **Technical Details**: MLV$^2$-Net은 nnU-Net 기반으로, 3D FLAIR MRI를 입력으로 받아들이며, rater 인코딩을 추가적으로 적용합니다. 또한, 다양한 rater들이 제공한 세그멘테이션 스타일을 학습하고, 예측을 집계하기 위해 가중 다수 결정을 사용하는 방식으로 이루어집니다. 이 과정에서 모델의 예측 볼륨에 대한 오류 경계를 파생하는 기술적 기여도 포함됩니다.

- **Performance Highlights**: MLV$^2$-Net은 인간의 참조 표준에 대해 0.806의 Dice similarity coefficient를 달성하여 MLV의 세그멘테이션 결과가 인간 간 신뢰성과 일치합니다. 또한, MLV의 부피와 관련된 노화 관련 연관성을 잘 재현할 수 있습니다.



### Classification and Morphological Analysis of DLBCL Subtypes in H\&E-Stained Slides (https://arxiv.org/abs/2411.08531)
- **What's New**: 이 논문에서는 diffuse large B-cell lymphoma (DLBCL)의 두 주요 아형인 activated B-cell-like (ABC)와 germinal center B-cell-like (GCB)로의 자동 분류 문제를 해결하고자 합니다. 저자들은 기존의 방법론 대신, 심층 학습(deep learning) 모델을 사용하여 DLBCL의 하위 유형을 보다 정확하게 구분할 수 있는 가능성을 보였습니다.

- **Technical Details**: 제안된 모델은 weakly supervised technique(약한 지도 학습 기법)인 Clustering-constrained Attention Multiple-instance learning (CLAM)과 Multiple Instance Learning (MIL) 모델을 활용하여 whole slide images (WSIs)을 분석합니다. ResNet34, ResNet50, RegNet, ConvNeXT_Tiny, EfficientNet, Swin_Tiny와 같은 다양한 feature extractors를 사용해 각기 다른 수준의 조직학적 세부 사항을 캡처합니다.

- **Performance Highlights**: 모델은 cross-validation 중 평균 87.4%의 AUC(Area Under the Curve)를 기록하며, 높은 positive predictive value (PPV)를 교훈적으로 보여줍니다. 이는 임상 적용 가능성, 특히 분자 검사에 대한 triaging에서 중요합니다. 추가로, ABC와 GCB의 형태학적 특징을 분석하여 두 아형 사이의 미세한 시각적 차이를 발견했습니다.



### Efficient Whole Slide Image Classification through Fisher Vector Representation (https://arxiv.org/abs/2411.08530)
- **What's New**: 이번 연구에서는 Whole Slide Images (WSI)를 효과적으로 분류하기 위한 새로운 방법론을 제안합니다. 이 방법은 가장 중요하고 유용한 패치(patch)만을 자동으로 식별하고 분석하여 전체 슬라이드를 처리할 필요를 없앱니다. 이러한 접근 방식은 진단의 정확도를 높이고 계산 효율성을 극대화하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 병리학적으로 중요한 정보에 기반하여 WSI에서 몇 가지 패치만을 추출하고, 두 번째 단계에서는 Fisher vectors (FV)를 사용하여 이들 패치에서 추출한 특징들을 표현합니다. Fisher vector는 세밀한 특징을 효과적으로 캡처할 수 있어 WSI 분석의 효율성을 높이는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 성능을 평가하여 전통적인 WSI 분석 및 최신 약한 감독 학습(weakly-supervised learning) 방법론과 비교하였습니다. 결과는 선택된 패치에 대한 집중적 분석과 Fisher vector 표현이 전통적인 방법들의 분류 정확도를 초과할 수 있음을 보여주었으며, 또한 계산 부하와 자원 소모를 크게 줄일 수 있음을 강조했습니다.



### BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis (https://arxiv.org/abs/2411.08508)
- **What's New**: 이번 논문에서는 BBSplat(Billboard Splatting)라는 새로운 3D 장면 표현 방식을 제안합니다. 이는 텍스처가 적용된 기하학적 프리미티브를 기반으로 하여 RGB 텍스처와 알파 맵을 학습할 수 있는 최적화가 가능한 평면 프리미티브의 집합으로 장면을 표현합니다.

- **Technical Details**: BBSplat은 고급 JPEG 압축 기술을 활용하여 텍스처를 sparsely structure로 효율적으로 압축하여 저장 공간을 줄입니다. 이 방법은 CUDA로 구현되어 GPU에서 효율적인 텍스처 샘플링과 백 프로파게이션을 수행합니다. BBSplat은 학습 가능한 PNSR, SSIM, LPIPS 지표의 개선을 보여줍니다.

- **Performance Highlights**: BBSplat은 3DGS 및 2DGS보다 최대 2배 빠른 추론 속도를 자랑하며, Tanks&Temples, DTU와 같은 실제 실내 및 실외 장면의 표준 데이터 세트에서 실험이 진행되었습니다. 특히 소수의 프리미티브를 사용할 때 BBSplat의 성능 향상이 두드러집니다.



### Impact of Iris Pigmentation on Performance Bias in Visible Iris Verification Systems: A Comparative Study (https://arxiv.org/abs/2411.08490)
Comments:
          14 pages, 5 figures, 5 Tables

- **What's New**: 이번 연구는 청색 홍채(blue iris)와 어두운 홍채(dark iris)의 인식 시스템 성능을 비교 분석하여, 반응 차이에 따른 바이오 메트릭 시스템의 정확성에 미치는 영향을 조사합니다. 이 연구는 다양한 장치를 이용해 수집된 데이터셋을 기반으로 진행되었습니다.

- **Technical Details**: 이 연구에서는 Open-Iris, ViT-b, ResNet50 등의 전통적인 머신 러닝(machine learning) 기술과 딥 러닝(deep learning) 모델을 활용하여 시스템의 성능 메트릭(예: Equal Error Rate (EER), True Match Rate (TMR))을 평가합니다. 데이터는 P1, P2, P3 스마트폰을 통해 수집되었습니다.

- **Performance Highlights**: 우리의 분석 결과, 청색 홍채는 어두운 홍채에 비하여 일반적으로 높은 정확성을 보였습니다. 이러한 결과는 다양한 데이터셋에 대한 훈련이 인식 성능을 향상시킬 수 있지만, 특정 모델 및 장치에 따라 그 개선 정도가 다름을 보여줍니다. 또한, 기기 간의 변동성과 관련된 본질적인 편견(bias)의 존재를 확인하였습니다.



### Methodology for a Statistical Analysis of Influencing Factors on 3D Object Detection Performanc (https://arxiv.org/abs/2411.08482)
- **What's New**: 이 논문에서는 LiDAR(라이다) 및 카메라 기반 3D object detector의 탐지 성능에 영향을 미치는 다양한 요인에 대한 통계적 분석 방법론을 제안합니다. 이는 현재까지 존재하지 않았던 혁신적인 접근 방식입니다.

- **Technical Details**: 이 연구는 다양한 환경 조건과 객체 속성이 3D.object detection의 정확성에 미치는 영향을 단변량 분석(univariate analysis)을 통해 비교하며, LiDAR와 카메라 기반 감지기를 분석하여 두 센서 모달리티(modality) 간의 차이를 조사합니다.

- **Performance Highlights**: 제안된 방법론은 각 감지기의 메타 정보에 대한 의존성을 통계적으로 비교하고, LiDAR 기반과 카메라 기반 단일 센서 감지기 간의 영향을 식별하며, 클래스별 의존성을 구체적으로 강조합니다.



### A survey on Graph Deep Representation Learning for Facial Expression Recognition (https://arxiv.org/abs/2411.08472)
- **What's New**: 이 종합 리뷰는 그래프 표현 학습 (Graph Representation Learning, GRL)의 관점에서 얼굴 표정 인식 (Facial Expression Recognition, FER)에 적용되는 다양한 방법론을 깊이 탐구합니다.

- **Technical Details**: FER의 주요 접근 방식으로 그래프 확산 (graph diffusion), 시공간 그래프 (spatio-temporal graphs), 다중 스트림 아키텍처 (multi-stream architectures) 등의 기술이 소개됩니다. GRL의 발전은 그래프 이론 (graph theory)에서 시작되었으며, 그래프 신경망 (Graph Neural Networks, GNNs)과 그래프 컨볼루션 네트워크 (Graph Convolutional Networks, GCNs) 등의 현대적 기법을 포함하고 있습니다.

- **Performance Highlights**: FER 시스템의 정확도와 강인성을 향상시키는 데 있어 GRL의 가능성을 강조하며, 미래 연구 기회와 새로운 방법론 개발의 필요성을 제기합니다.



### HyperFace: Generating Synthetic Face Recognition Datasets by Exploring Face Embedding Hyperspher (https://arxiv.org/abs/2411.08470)
Comments:
          Accepted in NeurIPS 2024 Safe Generative AI Workshop

- **What's New**: 이 논문에서는 개인의 동의 없이 인터넷에서 수집된 얼굴 인식 데이터셋의 윤리적 문제를 해결하기 위해 새로운 합성 데이터셋 생성 방법인 HyperFace를 소개합니다. 이 방법은 얼굴 인식 모델의 임베딩 공간에서 포장 문제(a packing problem)로서 데이터셋 생성 과정을 포뮬레이션합니다.

- **Technical Details**: HyperFace는 얼굴 인식 모델의 하이퍼구면(hypersphere)에서 임베딩 간의 간격을 최적화하는 방법을 사용합니다. 이를 통해 충분한 클래스 간 변동성을 제공하는 참조 임베딩 세트를 찾고, 조건부(face generator 모델)를 사용하여 최적화된 임베딩으로부터 얼굴 이미지를 합성합니다. 이 과정은 경량의 그래디언트 강하 기반 접근법을 통해 이루어집니다.

- **Performance Highlights**: HyperFace로 생성된 합성 데이터셋을 사용해 학습한 얼굴 인식 모델의 성능은 다양한 기존 실제 데이터셋에서 최첨단 결과를 달성했습니다. 실험 결과는 우리가 제안한 최적화 및 포장 접근 방식이 얼굴 인식 모델의 학습에 유용하다는 것을 보여줍니다.



### Can MLLMs Guide Weakly-Supervised Temporal Action Localization Tasks? (https://arxiv.org/abs/2411.08466)
- **What's New**: 최근 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)과 비디오 기초 모델(Video Foundation Models, VFMs) 및 대형 언어 모델(Large Language Models, LLMs)의 융합을 이용하여 비디오 이해 시스템을 향상시키는 새로운 학습 패러다임인 MLLM4WTAL이 제안되었습니다.

- **Technical Details**: MLLM4WTAL은 전통적인 약한 감독 하의 시간적 행동 위치 지정(Weakly-supervised Temporal Action Localization, WTAL) 방법을 개선하기 위해 MLLM의 잠재력을 활용합니다. 주요 모듈인 키 의미 매칭(Key Semantic Matching, KSM)과 완전 의미 재구성(Complete Semantic Reconstruction, CSR)을 통합하여 비디오 내의 행동 시간적 구간을 정확하게 찾는 방법을 제시합니다.

- **Performance Highlights**: 제안한 방법은 두 가지 주요 벤치마크인 THUMOS14와 ActivityNet1.2에서 최첨단 성능을 달성하고, 기존 WTAL 모델의 성능을 향상시키는 데에 유의미한 장점을 보여주었습니다.



### Biomass phenotyping of oilseed rape through UAV multi-view oblique imaging with 3DGS and SAM mod (https://arxiv.org/abs/2411.08453)
- **What's New**: 이 연구는 UAV(무인항공기) 기반의 촬영 기술과 3D Gaussian Splatting(3DGS)를 결합하여 유채(油菜) 작물의 바이오매스(생체량) 추정의 정확도를 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 작업에 포함된 3DGS는 36개의 각도에서 수집된 UAV 멀티뷰 경사 이미지로 3D 재구성을 수행하고, Segment Anything Model(SAM) 모듈은 포인트 클라우드(point cloud) 분할을 개선합니다. 이 과정에서 선형 회귀(linear regression)을 통해 측정된 지상 바이오매스에 맞춘 포인트 클라우드 볼륨(point cloud volumes)이 생성됩니다.

- **Performance Highlights**: 3DGS는 각각 7k와 30k 반복(iterations)에서 27.43 및 29.53의 피크 신호 대 잡음 비율(PSNR)을 기록하며, 훈련 시간은 각각 7분과 49분이 소요되었습니다. SAM 모듈의 평균 교차 비율(mIoU)은 0.961, F1-score는 0.980으로 높은 분할 정확도를 보였으며, 포인트 클라우드 볼륨 모델은 R2 값 0.976, RMSE 2.92 g/plant, MAPE 6.81%로 다른 모델들보다 더 나은 성능을 나타냈습니다.



### AD-DINO: Attention-Dynamic DINO for Distance-Aware Embodied Reference Understanding (https://arxiv.org/abs/2411.08451)
- **What's New**: 이 논문에서는 사람의 의도를 바탕으로 제스처 신호와 언어 설명을 통해 참조 대상을 예측하기 위한 새로운 프레임워크인 Attention-Dynamic DINO를 제안합니다. 이 모델은 포인팅 제스처 해석의 오류를 줄이기 위해 시각적 및 텍스트 특성을 통합하여 객체의 바운딩 박스 및 주목 출처를 동시에 예측합니다.

- **Technical Details**: Attention-Dynamic DINO (AD-DINO)는 Distance-aware Visual Perspective Taking(DA-VPT) 개념을 바탕으로 하여, 객체와의 상호작용 거리와 주목 출처를 동적으로 조정하는 Attention-Dynamic Touch Line(ADTL)을 활용합니다. 이 시스템은 초기 특성 추출, 교차 모달 특성 융합 및 언어 안내 쿼리 선택을 통해 시각 및 자연어 입력을 처리하고, 예측된 주목 출처에 따라 객체의 위치를 정확하게 파악합니다.

- **Performance Highlights**: YouRefIt 데이터셋에서 진행된 실험을 통해 AD-DINO는 0.25 IoU(threshold) 기준으로 76.3%의 정확도를 달성하였으며, 특히 0.75 IoU 기준에서는 인간 성능을 초과하는 55.4%의 정확도를 기록하여 이 분야에서 중요한 이정표를 세웠습니다. 이 결과는 이전의 SOTA 방법보다 16.4% 향상되었습니다.



### A Heterogeneous Graph Neural Network Fusing Functional and Structural Connectivity for MCI Diagnosis (https://arxiv.org/abs/2411.08424)
- **What's New**: 이 연구는 기능적 연결성과 구조적 연결성을 통합하는 이질적 그래프 신경망(HGNN)을 제안하여 다양한 두 가지 모드 이미지를 활용할 수 있는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 우리의 접근 방식은 혈액 산소 수준 의존성(rs-fMRI)과 백색질 구조(DTI) 정보를 사용하여 각각 동질 메타 경로(homo-meta-path) 및 이질 메타 경로(hetero-meta-path)를 구성합니다. 이 방법은 동질적 정보와 이질적 정보를 효과적으로 혼합하고, 이질적 그래프 풀링 전략을 통해 다양한 기능 간의 혼란을 방지합니다.

- **Performance Highlights**: ADNI-3 데이터셋에서 경미한 인지 장애(MCI) 진단을 위해 평가된 결과, 우리의 방법은 평균 분류 정확도 93.3%를 달성하여 다른 알고리즘보다 우수한 성능을 보였습니다.



### V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion (https://arxiv.org/abs/2411.08402)
- **What's New**: V2X-R 데이터셋은 LiDAR, 카메라 및 4D 레이더 데이터를 포함하는 최초의 시뮬레이션된 V2X 데이터셋입니다. 이 데이터셋은 12,079개의 시나리오와 37,727개의 LiDAR 및 4D 레이더 포인트 클라우드 프레임, 150,908장의 이미지 및 170,859개의 주석된 3D 차량 바운딩 박스를 포함하고 있습니다.

- **Technical Details**: V2X-R 데이터셋을 기반으로 한 협력 LiDAR-4D 레이더 융합 파이프라인은 네 가지 단계로 구성됩니다: 1) 각 에이전트에 의해 인코딩, 2) 에이전트 융합, 3) 모달 융합, 4) 바운스 예측. 또한, Multi-modal Denoising Diffusion (MDD) 모듈을 제안하여 에이전트 융합 후 노이즈가 있는 LiDAR 특성을 정리합니다.

- **Performance Highlights**: 릴 제출된 데이터셋에서의 실험 결과, LiDAR-4D 레이더 융합 파이프라인이 3D 객체 탐지에서 우수한 성능을 보여줍니다. MDD 모듈을 활용하여 안개 및 눈이 내리는 날씨에서 기본 융합 모델의 성능을 최대 5.73%/6.70% 향상시킬 수 있었습니다.



### MambaXCTrack: Mamba-based Tracker with SSM Cross-correlation and Motion Prompt for Ultrasound Needle Tracking (https://arxiv.org/abs/2411.08395)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 논문에서는 초음파(US) 유도 바늘 삽입에 대한 새로운 트래킹 솔루션인 MambaXCTrack를 제안합니다. 이는 Mamba 기반의 최초의 초음파 바늘 트래킹 애플리케이션으로, 구조적 상태 공간 모델과 교차 상관관계(SSMX-Corr) 및 암묵적 운동 프롬프트를 활용하여 바늘의 위치 추적 문제를 해결합니다.

- **Technical Details**: MambaXCTrack은 SSMX-Corr를 통해 글로벌 스케일의 장거리 모델링을 가능하게 하며, 노이즈와 아티팩트로부터 견고한 트래킹을 보장하기 위해 의미적 피처를 방지합니다. 또한 Cross-Map Interleaved Scan(CIS)을 결합하여 국소 픽셀 상호작용을 도와줍니다. 암묵적 저수준 운동 기술을 도입하여 바늘 끝의 간헐적 가시성 문제를 해결합니다.

- **Performance Highlights**: MambaXCTrack은 모터화된 바늘 삽입에 대한 실험에서 기존의 최첨단 방법들에 비해 우수한 성능을 보여줍니다. 그 결과는 각 개별 트래킹 모듈의 효과성을 강조하며, 물리적 샘플과 조직 샘플 모두에서 효과적으로 평가되었습니다.



### EgoVid-5M: A Large-Scale Video-Action Dataset for Egocentric Video Generation (https://arxiv.org/abs/2411.08380)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문은 egocentric 비디오 생성에 특화된 최초의 고품질 데이터셋인 EgoVid-5M을 소개합니다. 500만 개의 egocentric 비디오 클립과 함께 세부적인 행동 주석이 포함되어 있으며, 이는 가상 현실(VR), 증강 현실(AR), 게임 애플리케이션 향상에 크게 기여할 것으로 기대됩니다.

- **Technical Details**: EgoVid-5M 데이터셋에는 500만 개의 egocentric 비디오 클립이 포함되어 있으며, 고해상도(1080p)로 제공됩니다. 데이터셋은 행동 설명과 비디오 내용 간의 정렬, 동작의 강도 및 프레임 간의 일관성을 유지하기 위해 정교한 데이터 정리 파이프라인을 사용하여 관리됩니다. 또한 세부적인 행동 주석은 미세 조정된 운동 제어 및 고수준의 텍스트 설명으로 구분되어 제공됩니다.

- **Performance Highlights**: EgoVid-5M 데이터셋을 이용한 다양한 비디오 생성 모델의 실험에서, 해당 데이터셋이 egocentric 비디오 생성을 위한 모델 훈련에 상당한 향상을 가져온 것으로 나타났습니다. 논문에서는 특정 파라미터들을 측정하기 위한 평가 메트릭스를 설정하고, 비주얼 품질, 프레임 일관성, 행동 준수 및 운동 정확도 등 여러 차원에서 EgoVid-5M의 품질을 검증합니다.



### Multiscale Graph Construction Using Non-local Cluster Features (https://arxiv.org/abs/2411.08371)
- **What's New**: 이 논문에서는 그래프와 신호 특성을 동시에 고려하여 멀티스케일 그래프(multi-scale graph)를 구성하는 새로운 방법을 제안합니다. 기존의 방법들은 클러스터의 유사한 특징을 감지하지 못하는 경우가 많았습니다. 본 연구에서는 최적 수송(optimal transport)을 활용하여 클러스터 특징 간의 유사성을 계산하고, 비국소(non-local) 특성을 가진 멀티스케일 그래프를 생성합니다.

- **Technical Details**: 제안된 방법은 그래프의 클러스터를 세 가지 단계로 계층적으로 병합합니다: 1) 각 클러스터의 특징 벡터 추출, 2) 최적 수송(optimal transport)을 이용한 클러스터 특성 간 유사성 계산, 3) 변수 k-최근접 이웃 그래프(variable k-nearest neighbor graph, VkNNG) 구축 후 그래프 스펙트럴 클러스터링(graph spectral clustering)을 적용합니다. 이 접근 방식은 공간적으로 분리된 노드간에도 유사한 특징을 가진 클러스터를 연결할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 멀티스케일 이미지와 포인트 클라우드(segmentation) 분할에서 효과적인 성능을 보여주었습니다. 기존의 단일 스케일 방법들과 비교했을 때, 원래 클러스터의 경계를 잘 보존하면서도 비국소적인 영역을 연결하는 데 성공했습니다.



### A Chinese Multi-label Affective Computing Dataset Based on Social Media Network Users (https://arxiv.org/abs/2411.08347)
- **What's New**: 이 연구는 중국어 감정 데이터세트를 수집하고, 개인의 성격 특성과 감정을 통합한 다중 라벨 감정 데이터세트를 구축했습니다. 기존 데이터세트는 감정과 성격 특성을 별도로 주석을 달아 왔으며, 미세 감정(micro-emotions)과 감정 강도의 세밀한 라벨링이 부족했습니다.

- **Technical Details**: 연구진은 중소형 SNS 플랫폼인 위보(Weibo)에서 50,000명 이상의 개인 중 11,338명의 유효 사용자를 선별하여 이들의 MBTI 성격 라벨과 함께 566,900개의 게시글을 수집했습니다. EQN 방법론을 사용하여 같은 사용자의 MBTI 성격 특성을 6가지 감정(emotions)과 미세 감정(micro-emotions)과 통합하여 주석이 달린 감정 데이터세트를 작성했습니다.

- **Performance Highlights**: 다양한 NLP(classification) 모델을 통한 검증 결과, 이 데이터세트는 복잡한 인간 감정의 기계 인식을 진전시킬 수 있는 강력한 유용성을 입증했습니다. 이 데이터세트는 심리학, 교육, 마케팅, 금융 및 정치 연구에 자료 지원을 제공하는 데 목적이 있습니다.



### DyConfidMatch: Dynamic Thresholding and Re-sampling for 3D Semi-supervised Learning (https://arxiv.org/abs/2411.08340)
Comments:
          Accepted by Pattern Recognition Journal

- **What's New**: 이번 연구는 3D 맥락에서 반자동 학습(Semi-supervised Learning, SSL)의 데이터 불균형 문제를 해결하기 위한 새로운 방법론을 제시합니다. 구체적으로, 클래스 수준의 신뢰도(class-level confidence)를 이용하여 학습 상태를 평가하는 방법과 다이나믹 임계값(dynamic thresholding) 기법을 사용하여 unlabeled 데이터 활용을 개선합니다.

- **Technical Details**: 제안한 방법은 작은 표본 수를 가진 클래스에 대한 데이터를 보다 효과적으로 활용하기 위해 클래스 수준의 신뢰도를 고려합니다. 또한, 재샘플링 전략(resampling strategy)을 도입하여 충분히 알려진 클래스에 대한 편향(bias)을 완화하고, 클래스 간의 공정한(representation) 비율을 보장합니다.

- **Performance Highlights**: 광범위한 3D SSL 실험을 통해 본 연구 방법이 분류(classification)와 탐지(detection) 작업에서 기존의 최첨단 기법(state-of-the-art)을 뛰어넘는 성능을 보여주며, 데이터 불균형 문제를 효과적으로 해결할 수 있음을 입증하였습니다.



### DEEGITS: Deep Learning based Framework for Measuring Heterogenous Traffic State in Challenging Traffic Scenarios (https://arxiv.org/abs/2411.08335)
Comments:
          Submitted for presentation at the 103 rd Annual Meeting of Transportation Research Board and publication in Transportation Research Record: Journal of Transportation Research Board

- **What's New**: 이번 논문에서는 DEEGITS(Deep Learning Based Heterogeneous Traffic State Measurement)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 최신 convolutional neural network (CNN) 기술을 활용하여 차량과 보행자를 정확하고 신속하게 감지하고, 혼잡 및 가려짐과 같은 어려운 상황에서도 교통 상태를 측정합니다.

- **Technical Details**: DEEGITS는 데이터 융합(data fusion)을 통해 차량과 보행자를 동시에 감지할 수 있는 훈련 데이터셋을 향상시키며, 이미지 전처리(image preprocessing) 및 증강(augmentation)을 통해 데이터셋의 품질과 양을 개선합니다. YOLOv8로 사전 훈련된 모델에 전이 학습(transfer learning)을 적용하여 다양한 차량을 식별하는 모델의 능력을 증가시키고, Grid Search 알고리즘을 통해 최적의 하이퍼파라미터를 구합니다. 여기서 Stochastic Gradient Descent (SGD) 옵티마이저가 다른 옵티마이저보다 우수한 성능을 보입니다.

- **Performance Highlights**: 광범위한 실험과 평가 결과, 검증 세트에서 0.794 mAP@0.5, 테스트 세트에서 0.786 mAP@0.5의 높은 정확도를 달성하며, 유사한 데이터셋에서 이전 벤치마크를 초월한 성과를 보여 줍니다. 또한, DeepSORT 다중 객체 추적(multi-object tracking) 알고리즘을 통합하여 감지된 차량과 보행자를 추적합니다. 다양한 교통 조성 및 혼잡 수준을 가진 두 위치에서 실험을 진행하였으며, 두 경우 모두 통계적으로 유의미하지 않은 오류를 보였고, 이질적 교통 흐름과 속도 측정에서 각각 0.99~0.88 및 0.91~0.97의 상관관계를 나타냈습니다.



### Enhancing Multimodal Query Representation via Visual Dialogues for End-to-End Knowledge Retrieva (https://arxiv.org/abs/2411.08334)
- **What's New**: 본 논문에서는 기존의 분리된 모델에 의존하는 다중 모달(multi-modal) 검색 시스템의 한계를 극복하기 위해, Ret-XKnow라는 엔드 투 엔드(end-to-end) 검색 시스템을 제안합니다. 이 시스템은 텍스트 검색기에 다중 모달 쿼리를 이해하는 능력을 부여하며, 비주얼 정보와 텍스트 쿼리 간의 동적 상호작용을 통해 성능을 향상시킵니다.

- **Technical Details**: Ret-XKnow는 부분 합성곱(partial convolution) 메커니즘을 사용하여 주어진 텍스트 쿼리와 관련된 비주얼 정보에 집중합니다. 이를 통해 비주얼 임베딩(visual embeddings)을 압축하고, 텍스트 쿼리 표현과의 관련성 점수를 적응형 마스크로 활용하여 시각적 정보의 중요성을 강조합니다. 또한, Visual Dialogue-to-Retrieval (ViD2R) 데이터셋을 도입하여, 시각적 대화 데이터셋으로부터 자동으로 생성된 데이터셋을 통해 다중 모달 상호작용을 효과적으로 학습합니다.

- **Performance Highlights**: Ret-XKnow는 사전 훈련(pre-training) 데이터셋으로 ViD2R를 사용하여 네 가지 다중 모달 데이터셋에서 제로샷(zero-shot) 검색 성능의 유의미한 향상을 보여주며, 추가적인 모델 조정 없이도 높은 차세대 검색 성능을 입증합니다. 이는 Ret-XKnow 모델이 기존의 다양한 기준선(baseline) 방법들과 비교하여 우수한 결과를 도출했음을 나타냅니다.



### SASE: A Searching Architecture for Squeeze and Excitation Operations (https://arxiv.org/abs/2411.08333)
- **What's New**: SASE는 Neural Architecture Search(NAS)를 활용하여 squeeze와 excitation 운영을 자동으로 탐색하고, 현재 알려진 attention 모듈을 넘어서는 새로운 아키텍처를 찾기 위한 최초의 시도입니다.

- **Technical Details**: SASE는 squeeze와 excitation operation을 4개의 서로 다른 집합으로 나누어 채널 및 공간 차원에서 개별적으로 검색합니다. 각 집합은 기존 attention 블록과 새로운 기술을 포함하여 다양한 원천에서 operation을 추출합니다. 이러한 방식은 directed acyclic graph(DAG)를 기반으로 하여 효율적인 검색을 가능합니다.

- **Performance Highlights**: SASE attention 모듈을 사용한 ResNet-50/101 네트워크는 현재의 state-of-the-art attention 모듈을 사용한 네트워크에 비해 최고의 성능을 기록했습니다. 이는 COCO와 ImageNet-1K 벤치마크에서 검증된 결과입니다.



### Motion Control for Enhanced Complex Action Video Generation (https://arxiv.org/abs/2411.08328)
Comments:
          Project page: this https URL

- **What's New**: 새로운 프레임워크 MVideo는 복잡한 동작을 가진 긴 동작 비디오 생성을 위한 혁신적인 방법을 제시합니다.

- **Technical Details**: MVideo는 마스크 시퀀스를 추가적인 동작 조건 입력으로 사용하여 동작 세부사항을 개선합니다. 이를 통해 GroundingDINO와 SAM2와 같은 비전 모델을 활용하여 자동으로 마스크 시퀀스를 생성하고, 효율성을 높이며 견고함을 증가시킵니다.

- **Performance Highlights**: MVideo는 텍스트 프롬프트와 동작 조건을 효과적으로 정렬하여 복잡한 비디오를 생성할 수 있으며, 동작 조건 편집 및 조합을 통해 더 다이내믹한 생성이 가능합니다.



### Choix d'un espace de repr\'esentation image adapt\'e \`a la d\'etection de r\'eseaux routiers (https://arxiv.org/abs/2411.08293)
Comments:
          in French language

- **What's New**: 최근 몇 년간 이미지의 구조와 텍스쳐 컴포넌트를 분해할 수 있는 알고리즘들이 등장했습니다. 본 논문에서는 이러한 분해 기법을 항공 사진이나 위성 이미지에서 도로 네트워크를 자동으로 검출하는 문제에 적용하는 방법을 제시합니다.

- **Technical Details**: 이 논문에서는 Y.Meyer의 연구를 기반으로 한 이미지 분해 기법을 활용하여 도로 네트워크를 더 효과적으로 감지할 수 있는 방안을 규명합니다. 이미지 분해 모델로는 Rudin-Osher-Fatemi 알고리즘을 사용하며, 이 과정에서 고유한 텍스쳐 공간을 생성합니다. 구조는 u로, 텍스쳐는 v로 분해되어 서로 다른 성분으로 나타납니다.

- **Performance Highlights**: 본 연구의 실험 결과는 텍스쳐 공간을 사용한 감지가 기존의 Canny-Deriche 방식으로 전처리한 이미지에 비해 우수한 성능을 보임을 확인하였습니다. 특히, 필라인 객체인 도로의 감지에서 성능 향상이 두드러지며, 향후 다양한 적용 가능성이 기대됩니다.



### Noisy image decomposition: a new structure, texture and noise model based on local adaptivity (https://arxiv.org/abs/2411.08292)
Comments:
          arXiv admin note: text overlap with arXiv:2411.05265

- **What's New**: 이번 논문에서는 이미지에서 구조, 텍스처, 노이즈의 세 부분으로 분해하는 새로운 모델을 제안합니다. 기존의 2부분 모델에서 더 나아가 세 가지 구성 요소를 분리하는 접근법을 통해 노이즈 문제를 해결하고자 합니다.

- **Technical Details**: 논문에서 제안한 모델은 정규화된 방법론에 기반하여 이미지를 세 부분으로 나누는 것을 목표로 합니다. 구조는 B⁢V(영역 범위 제한 함수) 공간의 기능으로 모델링되고, 텍스처는 G(dual space) 공간에 위치합니다. 추가적으로, 가우시안 노이즈가 이미지에 추가된 것으로 가정하고, 이 노이즈는 특정한 매우 진동하는 함수로 모델링합니다.

- **Performance Highlights**: 새로운 알고리즘은 기존의 Aujol과 Chambolle의 모델과 비교하여 향상된 성능을 보여줍니다. 본 모델은 텍스처가 노이즈에 의해 손상되는 문제를 극복하고 텍스처와 노이즈의 분리를 효과적으로 수행할 수 있음을 입증하며, 3부분 분해 모델의 합리성을 증명합니다.



### Restoration algorithms and system performance evaluation for active imagers (https://arxiv.org/abs/2411.08291)
- **What's New**: 이 논문은 능동 이미징 시스템(active imaging system)과 관련된 두 가지 분야를 다룹니다. 첫째, 대기 turbulence에 의한 이미지 왜곡(artefacts)들을 복원하기 위한 이미지 처리 알고리즘을 탐구하고, 둘째 이러한 시스템의 성능을 평가하기 위한 방법론을 제안합니다.

- **Technical Details**: 논문에서는 대기 turbulence의 영향을 조사하기 위해 전통적인 시간 평균 필터(temporal mean filter) 대신 시간 중위수 필터(temporal median filter)를 제안합니다. 이 필터는 객체의 가장자리를 흐리게 하지 않고, 더 나아가 한계 상황에서는 새로운 알고리즘인 warping 기법을 기반으로 한 방법을 사용하여 이미지 dancing 현상을 교정하는 방식에 대해 설명합니다. 성능 평가에 있어, 독일의 TRM3 모델에서 영감을 받아 새로운 성과 메트릭을 제안합니다.

- **Performance Highlights**: NATO-TG40 필드 시험에서 수집된 데이터베이스를 활용한 실험 결과, 제안된 시간 중위수 필터가 고주파 이미지 복원에서 시간 평균 필터보다 우수하다는 것을 보여주었습니다. 전반적으로, 이 연구는 능동 이미징 시스템의 성능을 향상시키기 위한 새로운 접근 방식을 나열하며, 특히 고가의 turbulence 상황에서 효과를 보여줍니다.



### MBA-SLAM: Motion Blur Aware Dense Visual SLAM with Radiance Fields Representation (https://arxiv.org/abs/2411.08279)
- **What's New**: 이 논문에서 제안하는 MBA-SLAM은 모션 블러(Blur) 프레임을 효과적으로 처리하기 위한 새로운 RGB-D SLAM 파이프라인입니다. 이 방법은 기존의 SLAM 시스템의 한계를 극복하고, 모션 블러로 인해 발생하는 카메라 로컬라이제이션 및 맵 재구성 성능 저하를 해결합니다.

- **Technical Details**: MBA-SLAM은 모션 블러 감지 트래커와 Neural Radiance Fields (NeRF) 또는 3D Gaussian Splatting (3DGS) 등의 맵퍼(mapper)를 통합하여 모션 블러 영향을 최소화합니다. 트래킹(tracking) 과정에서 모션 모델을 사용해 각 이미지의 카메라 모션 궤적을 정확히 묘사하고 사진메트릭(photo-metric) 일관성을 유지하여 카메라의 이동 경로를 조정합니다.

- **Performance Highlights**: 실험 결과, MBA-SLAM은 모션 블러가 있는 이미지와 선명한 이미지를 모두 사용할 때 기존의 최첨단 방법을 초월하는 성능을 보여주었습니다. 특히, 신경망 기반 SLAM 방법인 NeRF와 3DGS 기반 SLAM 방법에 비해 개선된 로컬라이제이션 및 맵 재구성 성능이 입증되었습니다.



### LBONet: Supervised Spectral Descriptors for Shape Analysis (https://arxiv.org/abs/2411.08272)
Comments:
          14 pages, 13 figure

- **What's New**: 본 논문은 Laplace-Beltrami operator (LBO)의 비선형 형상 분석에서의 유용성을 재조명하며, 이론적으로 고안된 도구들이 실제 응용에서 기능하고 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: LBO는 등거리 변형(isometric deformation)에 대해 불변성을 가지며, 오르토노멀 기저(orthonormal basis)를 형성하는 계수를 갖습니다. 본 연구에서는 매니폴드(manifold)에서 여러 연산자를 감독(supervised)하여 학습하는 방식을 탐구합니다.

- **Performance Highlights**: LBO를 최적화함으로써 heat kernel signature와 같은 기존 기술자가 다양한 작업(예: retrieval, classification, segmentation, correspondence)에서 크게 개선되었습니다. 또한 LBO 고유 기저(eigenbasis)는 전역 및 고도로 지역적인 학습에서 모두 적응할 수 있음을 입증하였습니다.



### DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection (https://arxiv.org/abs/2411.08227)
- **What's New**: 이 논문은 다중 모달(멀티모달) OOD (Out-of-Distribution) 탐지에서 intra-class variability (클래스 내 변동성)를 고려하지 않는 기존 접근 방식의 한계를 지적합니다. 이를 해결하기 위해 Dynamic Prototype Updating (DPU) 프레임워크를 제안하여 클래스 중심 표현을 동적으로 업데이트하여 견고성과 일반화를 향상시킵니다.

- **Technical Details**: DPU는 유사 샘플의 분산을 측정하여 클래스 별 중심 표현을 동적으로 조정합니다. 이는 instance-level invariant training (인스턴스 수준 불변 훈련)을 활용하여 intra-class cohesion (클래스 내 응집력)과 inter-class separation (클래스 간 분리)을 최적화합니다. 초기 Cohesive-Separate Contrastive Training 절차를 통해 클래스 내부의 변동성을 줄이고, 이후 동적 프로토타입 근사 메커니즘을 통해 prototyping representation을 정교화하여 outlier (외부 작용물)가 클러스터 진화에 미치는 부정적 영향을 완화합니다.

- **Performance Highlights**: DPU는 기존의 9가지 OOD 알고리즘을 기반으로 한 포괄적인 실험에서 Near-OOD 탐지에서는 모든 메트릭에서 약 10% 향상된 성능을, Far-OOD 탐지에서는 최대 80% 개선된 성능을 보여주며 새로운 최첨단 성능을 달성했습니다.



### GTA: Global Tracklet Association for Multi-Object Tracking in Sports (https://arxiv.org/abs/2411.08216)
Comments:
          Accepted by ACCV 2024 MLCSA Workshop

- **What's New**: 본 논문에서는 스포츠 시나리오에서 다중 객체 추적(multi-object tracking)의 성능을 향상시키기 위한 새로운 전처리 알고리즘인 Global Tracklet Association(GTA)를 제안합니다. 이 방법은 멀티 아이덴티티가 포함된 추적기를 분할하고 동일 아이덴티티로 보이는 추적기를 연결합니다.

- **Technical Details**: GTA는 두 가지 주요 모듈로 구성됩니다: Tracklet Splitter와 Tracklet Connector. Tracklet Splitter는 서로 다른 아이덴티티 인스턴스를 올바르게 분리함으로써 혼합 오류(mix-up error)를 해결하고, Tracklet Connector는 동일한 아이덴티티의 분할된 추적기를 병합하여 컷 오프 오류(cut-off error)를 수정합니다. 이 과정에서 DBSCAN 클러스터링 기법이 사용됩니다.

- **Performance Highlights**: 제안한 방법은 SportsMOT 데이터셋에서 HOTA 점수 81.04%로 새로운 최첨단 성능을 달성했습니다. SoccerNet 데이터셋에서도 HOTA 점수가 79.41%에서 83.11%로 증가하며 여러 트래커의 성능을 일관되게 향상시켰습니다.



### Latent Space Disentanglement in Diffusion Transformers Enables Precise Zero-shot Semantic Editing (https://arxiv.org/abs/2411.08196)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2408.13335

- **What's New**: 이번 연구에서는 Diffusion Transformers (DiTs)의 최신 혁신과 텍스트 기반 이미지 생성에서의 성과를 탐구합니다. 특히, 다양한 부분의 의미(subspace)에서 편집 방향을 제어할 수 있는 분리된(latent) 공간의 구조를 분석하였습니다.

- **Technical Details**: 본 논문은 DiT 모델의 latent space에 대한 심층적인 분석과, Edit-Identify-Manipulate (EIM) 프레임워크를 제안합니다. 이 프레임워크는 주어진 원본 이미지와 텍스트 프롬프트를 인코딩하여 joint latent embedding을 생성하고, Hessian Score Distillation Sampling (HSDS) 방법을 사용하여 특정 속성을 제어하는 편집 방향을 식별합니다.

- **Performance Highlights**: 이 연구는 제안된 EIM 프레임워크를 통해 기존 이미지 편집 기술에 비해 더 정밀하고 세밀한 편집을 가능하게 하여, 새로운 기준인 ZOPIE (Zero-shot Open-source Precise Image Editing) 벤치마크에서 효과적인 성과를 입증하였습니다.



### An Explainable Machine Learning Approach for Age and Gender Estimation in Living Individuals Using Dental Biometrics (https://arxiv.org/abs/2411.08195)
- **What's New**: 이 연구는 살아있는 개인에서의 나이(age)와 성별(gender) 추정에 대한 새로운 예측 시스템을 개발하고 있으며, 치아 데이터를 활용하여 개선된 정확성을 보여줍니다.

- **Technical Details**: 연구에서는 Cat Boost Classifier (Catboost), Gradient Boosting Machine (GBM), Ada Boost Classifier (AdaBoost), Random Forest (RF), eXtreme Gradient Boosting (XGB), Light Gradient Boosting Machine (LGB), Extra Trees Classifier (ETC)와 같은 다양한 머신러닝(ML) 모델을 활용하여, 862명의 생존 개인의 치아 데이터 분석을 수행했습니다. 또한, SHAP 기법을 활용하여 해석 가능한 AI 모델을 개발하였습니다.

- **Performance Highlights**: RF와 XGB 모델은 나이와 성별 추정에 있어 가장 높은 F1 스코어를 기록하였으며, 특히 XGB 모델은 나이 추정에서 73.26%, RF 모델은 성별 추정에서 77.53%의 F1 스코어를 달성하였습니다.



### TractoEmbed: Modular Multi-level Embedding framework for white matter tract segmentation (https://arxiv.org/abs/2411.08187)
Comments:
          Accepted at 27th International Conference on Pattern Recognition (ICPR), 2024 15 pages, 2 figures

- **What's New**: 이 논문은 백질(White Matter) 경로 세분화의 효율성을 높이기 위해 TractoEmbed라는 모듈 기반의 다단계 임베딩 프레임워크를 제안합니다. 이 프레임워크는 급격한 구조적 차이를 극복하고, 다양한 연령대 데이터셋에서 백질 경로 세분화의 성능을 향상시키면서도, 미래 연구에 추가적인 임베딩을 통합할 수 있는 유연성을 제공합니다.

- **Technical Details**: TractoEmbed는 각기 다른 레벨에서 지역화된 표현을 인코딩하는 학습 과제를 통해 구성된 모듈식 구조를 가지고 있습니다. 이 방법은 개별 스트림라인(streamline), 클러스터(cluster), 패치(patch) 등으로 최대한의 공간 정보를 포착하는 새로운 계층적 스트림라인 데이터 표현을 도입합니다. 또한, 이 프레임워크는 주변 스트림라인의 최소화를 통해 실질적인 임상 환경에서 견고성을 강화하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, TractoEmbed는 다양한 데이터셋에서 기존의 최첨단(SOTA) 방법들을 초월하는 성능을 보여주었으며, 특히 다양한 연령대의 데이터에서도 일반화된 성능을 발휘했습니다. 이는 백질 경로 세분화 분야의 연구와 수술 계획에 중요한 기여를 할 것입니다.



### Comprehensive and Comparative Analysis between Transfer Learning and Custom Built VGG and CNN-SVM Models for Wildfire Detection (https://arxiv.org/abs/2411.08171)
Comments:
          In Proc. of the 2024 IEEE International Conference On Intelligent Computing in Data Sciences

- **What's New**: 이번 연구는 산불 감지 분야에서 전이 학습(Transfer Learning)의 효율성과 효용성을 분석하고, 사용자 맞춤형 모델과 사전 훈련된(pretrained) 모델 간의 성능을 비교합니다.

- **Technical Details**: 세 가지 목적에 맞춘 모델(VGG-7, VGG-10, CNN-Support Vector Machine(CNN-SVM))과 세 가지 사전 훈련된 모델(VGG-16, VGG-19, Residual Neural Network(ResNet) ResNet101)을 비교하였으며, 다양한 조명 조건, 시간대 및 지형 등 복잡성을 반영하는 산불 데이터셋을 사용하여 훈련 및 평가하였습니다.

- **Performance Highlights**: 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수(F1 Score)와 같은 성능 지표를 사용한 결과, 전이 학습이 사용자 맞춤형 모델보다 효과적임을 입증하였고, 이 연구는 AI 및 ML 분야의 향후 방향성을 제시합니다.



### CameraHMR: Aligning People with Perspectiv (https://arxiv.org/abs/2411.08128)
Comments:
          3DV 2025

- **What's New**: 본 연구는 단안 이미지에서의 3D 인간 자세 및 형태 추정의 정확성을 높이기 위해 새로운 카메라 내재 파라미터 추정 기법(HumanFoV)과 밀집 표면 키포인트 탐지기를 결합했습니다. 이러한 기술들은 기존 데이터셋에서 발생하는 오류를 줄이고, 더 현실적인 인체 모델을 생성하는 데 기여합니다.

- **Technical Details**: HumanFoV는 사람을 포함한 이미지 데이터셋을 사용하여 훈련된 시야(Field of View) 예측 모델입니다. 이 모델을 통해 SMPLify의 이미지 적합 과정에서 전방위 카메라 모델을 적용하여 카메라의 내재 파라미터를 정확히 추정합니다. 추가로, BEDLAM 데이터셋을 통해 138개의 밀집 표면 키포인트를 탐지하는 모델(DenseKP)을 훈련시키고, 이 키포인트를 사용하여 기존 2D 관절과 결합하여 더욱 정교한 3D 형태 추정을 가능하게 했습니다.

- **Performance Highlights**: CameraHMR 모델은 여러 HPS 벤치마크에서 최첨단( state-of-the-art) 정확도를 달성하였습니다. 기존의 pGT보다 향상된 3D 인체 형태를 제공하며, 이는 훈련 데이터셋에 포함된 카메라 내재 파라미터와 결합하여 이뤄진 성과입니다.



### TIPO: Text to Image with Text Presampling for Prompt Optimization (https://arxiv.org/abs/2411.08127)
Comments:
          21 pages, 13 figures

- **What's New**: TIPO(텍스트-이미지 변환을 위한 텍스트 사전 샘플링을 통한 프롬프트 최적화)는 자동 프롬프트 엔지니어링을 위한 혁신적인 프레임워크입니다. 이는 사용자 제공 프롬프트를 정제하여 고품질 이미지 생성을 위한 자세한 프롬프트 요구를 충족시킵니다.

- **Technical Details**: TIPO는 훈련된 프롬프트 데이터셋의 분포를 활용하여 사용자가 제공한 프롬프트를 조정하고, 경량 모델을 통해 복잡한 실시간 비용 없이 효율적인 프롬프트 최적화를 가능하게 합니다. 실험 결과, TIPO는 미적 점수를 개선하고 이미지 손상을 줄이며, 생성된 이미지가 데이터셋 분포와 더 잘 일치하도록 돕습니다.

- **Performance Highlights**: TIPO의 성능은 미적 품질과 사용자 선호도 측면에서 이미지를 개선하는 효과를 보여주며, 다양한 아키텍처에서 이미지 생성을 개선할 수 있는 범용 프롬프트 최적화 프레임워크로 자리잡았습니다.



### LUDO: Low-Latency Understanding of Highly Deformable Objects using Point Cloud Occupancy Functions (https://arxiv.org/abs/2411.08777)
- **What's New**: LUDO는 변형 가능한 객체(Deformable Objects)의 내부 구조를 정확하게 이해하는 새로운 방법으로, 단일 관점(Angle)에서의 포인트 클라우드(Point Cloud) 관측을 통해 30ms 이내에 결과를 제공합니다.

- **Technical Details**: LUDO는 Occupancy Networks를 활용하여 변형된 상태의 객체를 재구성하며, 내부 구조까지도 포함합니다. 해당 방법은 로봇 생검(Robotic Biopsies)과 같은 의료 작업에 매우 유용합니다.

- **Performance Highlights**: 실제 로봇 실험에서 LUDO는 다양한 ROI(Regions Of Interest)를 천공하는 데 98.9%의 성공률을 기록하였으며, 변형 가능한 객체와의 상호작용에서 변형 등록 방법(Deformable Registration Methods)이 필요하지 않음을 보여주었습니다.



### UNSCT-HRNet: Modeling Anatomical Uncertainty for Landmark Detection in Total Hip Arthroplasty (https://arxiv.org/abs/2411.08488)
- **What's New**: UNSCT-HRNet (Unstructured CT - High-Resolution Net) 모델을 제안하며, 이는 Spatial Relationship Fusion (SRF) 모듈과 Uncertainty Estimation (UE) 모듈을 통합한 딥러닝 기반 프레임워크입니다.

- **Technical Details**: SRF 모듈은 coordinate convolution과 polarized attention을 활용하여 복잡한 공간적 관계를 포착하는 능력을 향상시킵니다. UE 모듈은 엔트로피 기반으로 anatomically relevant한 예측을 보장합니다.

- **Performance Highlights**: UNSCT-HRNet은 기존 방법들과 비교했을 때 unstructured data에서 60% 이상의 성능 향상을 보여주며, structured dataset에서도 우수한 성능을 유지합니다.



### Trap-MID: Trapdoor-based Defense against Model Inversion Attacks (https://arxiv.org/abs/2411.08460)
Comments:
          Accepted by Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이번 논문에서는 Trapdoor 기반 모델 역전 방어(Trap-MID)를 제안하여 모델 역전 공격(Model Inversion, MI 공격)에 대한 새로운 방어 전략을 소개합니다. 이 방어 방법은 입력에 트리거를 주입하였을 때 특정 레이블을 예측하도록 설계된 트랩도어(trapdoor)를 통합하여 MI 공격을 오도하는 방식으로 작동합니다.

- **Technical Details**: Trap-MID는 모델의 동작에 트랩도어를 삽입하여 MI 공격자가 개인 정보를 추출하기보다는 트랩도어 트리거를 추출하도록 유도합니다. 이 연구는 트랩도어의 효과성과 자연스러움이 MI 공격을 오도하는데 미치는 영향을 이론적으로 논의하며, 다양한 MI 공격에 대한 실험에서 Trap-MID가 기존 방법보다 우수한 방어 성능을 발휘함을 보여줍니다.

- **Performance Highlights**: Trap-MID는 추가 데이터나 대규모 계산 오버헤드 없이 다양한 MI 공격에 대해 최신 방어 성능을 나타내며, 실험을 통해 그 효율성과 효과성을 입증했습니다.



### Machine Unlearning on Pre-trained Models by Residual Feature Alignment Using LoRA (https://arxiv.org/abs/2411.08443)
- **What's New**: 본 논문은 기계 비학습(machne unlearning) 기술을 제안하며, 훈련된 모델에서 특정 데이터의 영향을 제거하는 새로운 방법인 Residual Feature Alignment Unlearning을 소개합니다. 이 방법은 LoRA (Low-Rank Adaptation) 기법을 활용하여 초기 중간 특징(feature)을 분해하고, 비학습 데이터 세트와 유지된 데이터 세트에서 목표를 일치시킵니다.

- **Technical Details**: Residual Feature Alignment Unlearning 방법은 다음과 같습니다: 1) 모델의 중간 특징을 사전 훈련된 특징과 잔여 특징으로 분리합니다. 사전 훈련된 특징은 고정하고, 잔여 특징을 조정하여 유지된 데이터와 비학습 데이터 세트의 목표를 일치시킵니다. 2) 잔여 특징은 제로(또는 근제로)로 초기화하여 비학습 초기 단계에서 모델이 사전 훈련된 모델과의 거리에서 너무 빨리 벗어나지 않도록 합니다.

- **Performance Highlights**: 다양한 데이터 세트에서 실시된 실험에서 제안된 방법의 효과성과 효율성이 검증되었습니다. 이 방법은 높은 계산 비용 없이 대규모 모델의 빠르고 효율적인 비학습을 가능하게 합니다.



### The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defens (https://arxiv.org/abs/2411.08410)
- **What's New**: 이 연구는 Vision Large Language Models (VLLMs)의 jailbreak 공격에 대한 방어 기제를 심층 분석하고, 기존 방어 메커니즘의 '과도한 조심성' 문제를 드러내어 모델의 유용성을 저하시킨다는 새로운 발견을 제시합니다.

- **Technical Details**: 이 논문에서는 VLLMs가 공격에 취약한 이유와, 기존 방어 기제가 benign 입력에도 반응을 꺼리는 문제를 조사합니다. 또한 두 가지 평가 방법의 상관관계가 낮아 공격 전략이나 방어 메커니즘의 효과성을 판단하는 데 혼란을 초래하는 한계를 설명합니다. 새로운 접근으로는 VLLM 응답 전 유해성을 평가하는 비전 프리(evaluator)를 제안합니다.

- **Performance Highlights**: 이 연구는 VLLMs와 방어 전략의 안전성 패러독스를 해결하기 위한 최초의 종합 연구로, 실험 결과 방어 메커니즘의 과도한 조심성을 개선하여 모델의 유용성을 증가시킬 수 있는 기회를 제공합니다.



### Robust Divergence Learning for Missing-Modality Segmentation (https://arxiv.org/abs/2411.08305)
- **What's New**: 이번 연구에서는 뇌 종양의 다중 모달 MRI( Magnetic Resonance Imaging ) 분석을 위한 새로운 세그멘테이션 패러다임을 제안합니다. 기존 방법들이 모달리티( modality )의 결여 문제에 직면한 점을 보완하여, 유연한 처리가 가능한 단일 모달리티 병렬 처리 네트워크 프레임워크를 개발했습니다.

- **Technical Details**: 이 네트워크는 Hölder divergence를 기반으로 하여, 각 모달리티 데이터를 독립적으로 입력해 병렬로 처리하는 구조를 갖습니다. 또한, 모달리티의 가용성에 따라 네트워크 파라미터를 조정하는 동적 공유 프레임워크를 도입하였습니다. 손실 함수는 모델 예측과 실제 레이블 간의 불일치를 평가하기 위해 Hölder divergence와 상호 정보(mutual information)를 사용합니다.

- **Performance Highlights**: BraTS 2018 및 BraTS 2020 데이터셋에서 폭넓은 실험을 통해 제안된 방법이 기존 기술들보다 결여된 모달리티 처리에 있어 뛰어난 성능을 보여주었고, 각 구성 요소의 효용성 또한 검증되었습니다.



### EAPCR: A Universal Feature Extractor for Scientific Data without Explicit Feature Relation Patterns (https://arxiv.org/abs/2411.08164)
- **What's New**: 본 논문에서는 명시적인 Feature Relation Patterns (FRPs)이 없는 데이터에 대한 범용 특성 추출기 EAPCR을 소개합니다. EAPCR은 기존의 전통적인 방법보다 다양한 과학적 작업에서 일관되게 뛰어난 성능을 보입니다.

- **Technical Details**: EAPCR은 FRPs가 없는 데이터에서 중요한 특성 조합을 효과적으로 식별하기 위해 모든 가능한 FRPs를 노출하고, 이 조합을 샘플링하여 넓은 범위의 특성 상호작용을 평가하도록 설계되었습니다. 이는 비 이미지 기반 데이터에도 적용 가능한 범용적인 모델입니다.

- **Performance Highlights**: EAPCR은 비 이미지 의료 진단, 시스템 비정상 탐지 및 무기 촉매 효율 예측과 같은 다양한 과학적 작업에서 전통적인 방법을 일관되게 능가하며, 다른 딥 러닝 모델들이 어려움을 겪는 환경에서도 강력한 성능을 보여줍니다.



### TomoGRAF: A Robust and Generalizable Reconstruction Network for Single-View Computed Tomography (https://arxiv.org/abs/2411.08158)
- **What's New**: TomoGRAF는 초희소(ultra-sparse) 투영에서 고품질 3D 볼륨을 재구성할 수 있도록 X선( X-ray ) 전송 물리를 통합한 새로운 프레임워크입니다. 기존의 분석적/반복적 CT 재구성 알고리즘의 한계를 극복합니다.

- **Technical Details**: TomoGRAF는 CT 이미징 기하학을 캡처하고, X-ray 캐스팅 및 트레이싱 과정을 시뮬레이션하며, 훈련 과정에서 시뮬레이션된 CT 서브볼륨과 실제 데이터 간의 차이를 페널라이즈(penalizes)합니다. 이로 인해 기계 학습을 활용한 재구성이 가능합니다.

- **Performance Highlights**: TomoGRAF은 훈련 데이터와는 다른 이미지 특성을 가진 예측 불가능한 데이터셋에서 성능을 평가했으며, 최신 딥러닝(deep learning) 및 NeRF 방법과 비교해 현저한 성능 향상을 입증했습니다. 이 방법은 이미지 유도 방사선 치료(image-guided radiotherapy) 및 중재 방사선학(interventional radiology) 분야에서 사용할 수 있습니다.



### Deep Learning 2.0: Artificial Neurons That Matter -- Reject Correlation, Embrace Orthogonality (https://arxiv.org/abs/2411.08085)
Comments:
          Submitted to CVPR 2025

- **What's New**: 이 논문에서는 Neural Matter Network (NMN)라는 새로운 인공신경망을 소개합니다. NMN은 활성화 함수(activation function) 없이 비선형 패턴 인식을 실현하는 혁신적인 딥러닝 모델입니다.

- **Technical Details**: NMN의 핵심 혁신은 yat-product와 yat-product를 활용하여 입력을 유사 메트릭 공간(pseudo-metric space)으로 투영함으로써 비선형성을 자연스럽게 유도합니다. 이 방식은 전통적인 활성화 함수를 필요로 하지 않으며, 최종 클래스 확률 분포를 위한 softmax 레이어만 유지합니다. 이는 네트워크 아키텍처를 단순화하고 결정 과정의 투명성을 제공합니다.

- **Performance Highlights**: 다양한 데이터셋에서의 포괄적인 실험 평가 결과, NMN은 기존의 MLP(Multi-Layer Perceptrons)를 일관되게 초월하는 성능을 보여주었습니다. 이 연구는 효과적인 딥러닝 모델에 별도의 활성화 함수가 필요하다는 가정을 도전합니다.



### Online Collision Risk Estimation via Monocular Depth-Aware Object Detectors and Fuzzy Inferenc (https://arxiv.org/abs/2411.08060)
Comments:
          7 pages (IEEE double column format), 5 figures, 3 tables, submitted to ICRA 2025

- **What's New**: 이 논문은 자율주행차(AV)의 충돌 위험 수준을 모니터링 할 수 있는 프레임워크를 제안합니다. 이 프레임워크는 단일 모노큘러 카메라 이미지만을 사용하여 객체 탐지기의 성능에 기반하여 위험을 추론합니다.

- **Technical Details**: 프레임워크는 주어진 두 세트의 예측값을 활용하며, 하나는 깊이 맵에서 안전 관련 2.5D 객체를 추출하고, 다른 하나는 AV의 3D 객체 탐지기로부터 얻습니다. 논문에서는 Intersection-over-Union (IoU)와 깊이 불일치를 측정하여 두 예측 세트 간의 불일치가 3D 객체 탐지기의 안전 관련 오류와 강력히 상관관계가 있다는 실험적 검증을 제공합니다. 또한 퍼지 추론 시스템 (Fuzzy Inference System, FIS)을 구축하여 이러한 불일치를 충돌 위험 지표에 매핑합니다.

- **Performance Highlights**: 이 프레임워크는 대규모 nuScenes 데이터셋을 사용하여 검증하며, AV를 보호하는 데 유용한 위험 지표를 추론할 수 있음을 보여줍니다. 또한, 제안된 방법은 객체 탐지 기능의 결과로부터 충돌 위험 추정값을 도출하는 혁신적인 시도로, 해석 가능성과 적응성을 제공합니다.



### LAuReL: Learned Augmented Residual Layer (https://arxiv.org/abs/2411.07501)
Comments:
          Accepted at the 2nd Efficient Systems for Foundation Models Workshop at the International Conference on Machine Learning (ICML) 2024

- **What's New**: 논문에서는 기존의 잔여 연결(residual connection)을 일반화한 새로운 구조인 \emph{Learned Augmented Residual Layer} (LAuReL)를 소개하고 있다. LAuReL은 기존 구조를 대체하면서도 모델의 품질과 파라미터 수에서 개선된 성능을 보인다.

- **Technical Details**: LAuReL은 잔여 연결의 기본 구조를 재구성하여 학습 가능한 스칼라 파라미터와 선형 함수를 포함한다. 이 구조는 비선형성에 노출되지 않은 정보가 흐르는 ‘잔여 스트림(residual stream)’ 개념을 활용하여 비선형 구성요소의 학습을 최적화한다. 세 가지 특정 LAuReL 버전이 실험적으로 연구되었으며, 이를 통해 모델의 크기와 속도를 최적화하였다.

- **Performance Highlights**: ResNet-50 모델을 활용한 실험에서, LAuReL을 적용했을 때 추가 레이어를 추가하지 않고도 60%의 성능 증가를 달성하였다. 오히려 0.003%의 파라미터만 증가시키면서 성능을 유지하거나 개선할 수 있었다.



New uploads on arXiv(cs.AI)

### Causal Explanations for Image Classifiers (https://arxiv.org/abs/2411.08875)
- **What's New**: 이 논문에서는 이미지 분류기의 출력을 설명하기 위한 기존의 다양한 알고리즘들이 공식적인 원인과 설명의 정의를 기반으로 하지 않음을 지적하고, 실제 인과관계를 이론으로 한 새로운 블랙박스 접근법을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 원인과 설명에 대한 공식적인 정의에 근거하여 대략적인 설명을 계산하는 방법을 다룹니다. 이 알고리즘은 종료(termination) 및 복잡성(complexity) 분석을 포함하며, 정확한 정의와 비교했을 때의 근사(approximation) 정도에 대한 논의도 포함되어 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 도구 rex는 최신 도구들과 비교할 때 가장 효율적이며, 가장 작은 설명을 생성하면서 다른 블랙박스 도구들보다 높은 품질의 성능을 보여줍니다.



### Process-aware Human Activity Recognition (https://arxiv.org/abs/2411.08814)
- **What's New**: 이 연구에서는 Human Activity Recognition (HAR) 성능을 높이기 위해 컨텍스트(context)에서 프로세스 정보를 통합하는 새로운 방법을 제안합니다. 기존의 머신러닝 기반 접근 방식은 데이터 생성의 맥락을 간과하며, 이는 정확도에 부정적인 영향을 미칠 수 있습니다.

- **Technical Details**: 본 연구는 프로세스 모델과 생성된 이벤트 간의 정렬(alignment) 방법을 채택하여 HAR 성능을 향상시키고자 합니다. 이를 위해, HAR 모델의 출력과 컨텍스트의 프로세스 정보를 결합하여 훈련 단계에서 두 소스의 정보를 적절히 가중치를 두어 최적의 HAR 정확도를 얻습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 기존의 Graph Convolutional Network 알고리즘에 비해 더 높은 정확도와 Macro F1-score를 달성했습니다. 특히 식사 행동을 모니터링하는 데이터셋에서 이러한 결과를 입증했습니다.



### Rethinking CyberSecEval: An LLM-Aided Approach to Evaluation Critiqu (https://arxiv.org/abs/2411.08813)
Comments:
          NeurIPS 2024, 2 pages

- **What's New**: 이번 연구에서는 Meta의 CyberSecEval 접근 방식에 대한 평가와 이를 통한 사이버 보안 코드 검증 방법의 한계점을 다루고 있습니다. Static analysis(정적 분석)의 활용과 LLM(대형 언어 모델) 보조를 결합하여 보다 효과적인 평가 방법을 제안하고 있습니다.

- **Technical Details**: Meta의 분석 방법은 Insecure Code Detector (ICD), Instruct Benchmark, Autocomplete Benchmark의 세 가지 주요 구성 요소로 이루어져 있습니다. 번역된 статическая анализ (static analysis) 규칙을 사용하여 총 50개 보안 약점(weaknesses)을 탐지하며, 기존의 도구보다 제한된 범위입니다. 연구 결과, LLM 모델을 통해 수집된 데이터셋에서 인간에게 한계가 있는 불안정한 코드 생성을 분석할 수 있었습니다.

- **Performance Highlights**: Meta의 방법으로는 코드의 23.5%의 프롬프트가 정적 분석 규칙을 위반할 수 밖에 없었고, LLM을 활용한 평가 시 이 수치를 10.4%까지 감소시킬 수 있었습니다. 또한, 애초에 메타의 데이터셋에 포함된 주석이나 식별자를 제거했을 때 17.7%의 코드가 안전한 것으로 표시되었으며, 이러한 결과는 메타의 접근법의 제한적인 요소를 강조합니다.



### Evaluating World Models with LLM for Decision Making (https://arxiv.org/abs/2411.08794)
- **What's New**: 이 연구는 LLM(대형 언어 모델)이 어떻게 의사결정에서 세계 모델로 기능할 수 있는지를 포괄적으로 평가하고 있으며, 이를 통해 정책 검증, 행동 제안, 정책 계획과 같은 세 가지 주요 작업을 제안합니다.

- **Technical Details**: 31개의 다양한 환경을 Leveraging하여 정책 검증, 행동 제안, 정책 계획의 세 가지 주요 작업을 통해 LLM의 성능을 평가합니다. 이 연구의 주요 점은 LLM이 복잡한 작업에서 어떻게 의사결정에 도움을 주는지를 규명합니다.

- **Performance Highlights**: GPT-4o가 GPT-4o-mini보다 세 가지 주요 작업에서 더 우수한 성능을 보여주었으며, 특히 도메인 지식이 필요한 작업에서 두드러졌습니다. 그러나 장기적 의사결정 작업에서 LLM의 성능이 감소하고, 세계 모델의 다양한 기능의 조합이 성능의 불안정을 초래합니다.



### Polymetis:Large Language Modeling for Multiple Material Domains (https://arxiv.org/abs/2411.08728)
- **What's New**: 이 논문은 다양한 재료 과학 분야를 다루는 대형 언어 모델인 Polymetis를 제안합니다. 이는 연구자들이 재료에 대한 전문적인 지식을 얻을 수 있도록 지원하고, AI 기술을 활용하여 재료 과학 연구의 효율성을 높이는 것을 목표로 합니다.

- **Technical Details**: Polymetis 모델은 약 2백만 개의 재료 지식을 포함하는 데이터셋을 활용하며, IELM (Intelligent Extraction Large Model)을 통해 과학 텍스트에서 구조화된 지식을 자동으로 추출합니다. GLM4-9B 모델을 기반으로 여러 재료 도메인에서의 추론 능력을 향상시키기 위해 Lora 기법을 적용하고, 강화된 프롬프트 전략을 도입하여 결과의 조직성과 정확성을 개선했습니다.

- **Performance Highlights**: Polymetis는 재료 과학 전문가들이 제공한 기준 답변과 비교 평가를 통해 뛰어난 복잡한 지시 사항 이해 및 다중 도메인 추론 능력을 보여주었습니다. 이는 연구자들에게 정확하고 효율적인 재료 지식 탐색을 지원하는 잠재력을 가지고 있습니다.



### Analogical Reasoning Within a Conceptual Hyperspac (https://arxiv.org/abs/2411.08684)
Comments:
          Analogy-angle workshop full paper at IJCAI 2024

- **What's New**: 이번 논문에서는 복잡 샘플링된 하이퍼차원 컴퓨팅(Hyperdimensional Computing, HDC)과 개념 공간 이론(Conceptual Spaces Theory, CST)을 결합하여 유추 추론(analogical inference) 접근 방식을 제안합니다. 이는 기존의 전통적인 구조 매핑 이론을 넘어서는 방법을 모색합니다.

- **Technical Details**: HDC는 데이터 구조를 고차원 벡터로 표현하여 상징적/symbolic 및 서브상징적/subsymbolic 표현의 간극을 해소하는 대표 및 추론(paradigm) 체계로, 개념 공간 이론을 운영화하는 방법을 제시합니다. 이 접근 방식은 감각 관찰을 처리하고, 거리 메트릭(distance metric)을 적용하여 유추를 수행하는 데 필요한 다섯 가지 기능을 지원합니다.

- **Performance Highlights**: 예비 실험 결과는 이 HDC 기반 아키텍처가 개념 간 유사한 관계를 식별하고, 카테고리 기반 및 속성 기반 유추 추론을 수행할 수 있는 가능성을 보여줍니다. 연구는 장난감 도메인(toy domain)에서 초기 개념 증명을 통해 이론적 가능성을 뒷받침합니다.



### XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL (https://arxiv.org/abs/2411.08599)
- **What's New**: XiYan-SQL은 자연어(Natural Language) 쿼리를 SQL(Structured Query Language)로 변환하는 기술을 혁신적으로 향상시키기 위해 다중 생성기 앙상블(multi-generator ensemble) 전략을 사용하는 새로운 프레임워크입니다. 또한 M-Schema라는 반구조적(schema representation method) 데이터베이스 구조 접근 방식을 도입하여 데이터베이스 이해를 증진합니다.

- **Technical Details**: XiYan-SQL은 ICL(In-Context Learning)의 잠재력을 활용하면서 감독학습(Supervised Fine-Tuning)으로 후보 SQL 쿼리의 품질과 다양성을 높입니다. 모델의 세련된 학습 방법으로 다단계(multi-task) 접근 방식을 통해 SQL 생성 능력과 다양한 스타일적 선호를 증진합니다. 또한, M-Schema를 통해 데이터베이스의 계층적(structural) 구조를 보다 명확하게 표현합니다.

- **Performance Highlights**: XiYan-SQL은 Spider 테스트 세트에서 89.65%, SQL-Eval에서 69.86%, NL2GQL에서 41.20%의 실행 정확도를 기록하며, Bird 개발 벤치마크에서는 72.23%의 경쟁력 있는 성능을 보여줍니다. 이러한 성과는 다양한 벤치마크에서 제안된 방법의 효과성을 입증하며, 자연어에서 SQL로의 변환 작업에 대한 보다 넓은 응용 잠재력을 보여줍니다.



### Optimizing Automatic Summarization of Long Clinical Records Using Dynamic Context Extension:Testing and Evaluation of the NBCE Method (https://arxiv.org/abs/2411.08586)
- **What's New**: 현재 병원에서 임상 노트를 요약하는 것이 의료진의 문서 작업 부담을 줄이기 위해 매우 중요하다는 점을 강조하며, 기존의 수동 요약 방식이 의료진에게 어려움을 주고 있다는 문제점을 다루고 있습니다. 이러한 문제를 해결하기 위해, LLMs(대형 언어 모델)를 사용한 자동 요약 방법을 제안합니다.

- **Technical Details**: 우리는 7B 모델인 open-calm-7b를 사용하였으며, Native Bayes Context Extend와 다시 설계된 디코딩 메커니즘을 덧붙여 한 번에 한 문장을 참조하는 방식으로 입력을 관리합니다. 이를 통해 2048 tokens의 문맥 윈도우 내에서 입력을 유지하여 LLM이 맥락을 잃지 않도록 합니다.

- **Performance Highlights**: 우리의 개선된 모델은 200개의 샘플에서 ROUGE-L 메트릭을 기준으로 175B 이상의 Google Gemini와 거의 동등한 성능을 달성하여, 리소스를 적게 사용하면서 자동 EMR(전자의무기록) 요약의 타당성을 높였습니다.



### Leveraging LLMs for Predictive Insights in Food Policy and Behavioral Interventions (https://arxiv.org/abs/2411.08563)
- **What's New**: 이 논문은 대형 언어 모델(LLM)을 활용하여 식품 관련 정책 개입의 결과를 예측하는 새로운 도구인 PREDICT를 소개합니다. 이 도구는 과거의 연구 데이터를 기반으로 하여 정책 개입의 효과를 예측하는 데 도움을 줄 수 있습니다.

- **Technical Details**: PREDICT는 74개의 발표된 과학 기사를 포함하는 포괄적인 데이터셋을 사용하여 GPT-3.5 Turbo 모델을 미세 조정(fine-tuning)하여 개발되었습니다. 이 모델은 총 약 220만 개의 관찰치를 기반으로 하여 약 80%의 정확도로 효과 방향을 예측할 수 있습니다.

- **Performance Highlights**: 모델의 예측 능력은 12개의 진행 중인 실험과 비교하여 검증되었으며, 데이터셋의 특정 특성과 프롬프트 스타일이 예측 신뢰도에 미치는 영향을 밝히고 있습니다.



### Deeper Insights into Learning Performance of Stochastic Configuration Networks (https://arxiv.org/abs/2411.08544)
- **What's New**: 본 논문에서는 Stochastic Configuration Networks (SCNs)의 감독 메커니즘이 학습 성능에 미치는 영향을 포괄적으로 분석하고, 새로운 Recursive Moore-Penrose Inverse-SCN (RMPI-SCN) 훈련 방법을 제안합니다.

- **Technical Details**: SCNs는 적응형 감독 메커니즘을 가지고 있으며, 이를 통해 효과적인 랜덤 기반 함수를 생성하여 오류 없는 학습을 가능하게 합니다. 본 연구에서는 새로운 감독 메커니즘을 통해 Moos-Penrose 역행렬의 계산 없이 랜덤 기반 함수의 오류 감소 가능성을 정확하게 평가하는 방법을 제안합니다.

- **Performance Highlights**: RMPI-SCN은 기존 SCN보다 학습 성능이 향상되었으며, 대규모 데이터 모델링 응용에 대한 가능성을 입증합니다.



### Explainers' Mental Representations of Explainees' Needs in Everyday Explanations (https://arxiv.org/abs/2411.08514)
- **What's New**: 이 연구는 XAI(설명 가능한 인공지능) 시스템이 사용자의 요구에 반응할 수 있도록 설명자의 정신적 표현(mental representations)을 탐구하는 데 중점을 두었습니다. 기술적 아티팩트의 일상적 설명에서 설명자가 가진 지식과 관심의 변화를 이해하는 것이 중요하다는 점을 강조합니다.

- **Technical Details**: 이 연구는 XAI에 대한 두 가지 관점을 제시합니다: 하나는 'Architecture'라는 관찰 가능하고 측정 가능한 특징, 다른 하나는 'Relevance'라는 해석 가능성입니다. 연구에서는 9명의 설명자(Explainer)와 반구조화된 사전 및 사후 인터뷰를 통해 설명자의 응답을 질적 내용 분석(qualitative content analysis) 기법으로 분석하였습니다.

- **Performance Highlights**: 조사 결과, 초기 단계에서 설명자는 수용자의 폭넓은 지식의 필요성을 예상하지 못했던 경향이 있으며, 초기 지식은 Architecture에 집중되었다가 Relevance와 관련된 지식으로 발전하는 경향을 보였습니다. 이 연구는 XAI 시스템이 사용자 모델(user models)을 통해 적응형 설명이 가능해질 수 있도록 돕는 실제적 함의를 제시합니다.



### Building Trustworthy AI: Transparent AI Systems via Large Language Models, Ontologies, and Logical Reasoning (TranspNet) (https://arxiv.org/abs/2411.08469)
- **What's New**: 이 논문은 설명 가능하고 신뢰할 수 있는 AI 시스템에 대한 필요성을 강조하며, TranspNet 파이프라인을 제안합니다. 이 파이프라인은 대규모 언어 모델(LLM)과 기호 AI(symoblic AI)를 통합하여 전문가 지식, 검색 증강 생성(retrieval-augmented generation, RAG) 및 답변 집합 프로그래밍(Answer Set Programming, ASP)과 같은 형식적 추론 프레임워크를 활용합니다.

- **Technical Details**: TranspNet는 기호 AI의 구조적 추론과 검증을 통해 LLM의 출력을 강화하여, 규제 요구 사항에 부합하는 투명하고 신뢰할 수 있는 결과를 보장합니다. 이 파이프라인은 LLM의 능력을 활용하면서도 형식적 논리를 통해 LLM의 출력의 논리적 일관성을 검증하는 메커니즘을 제공하여, AI 시스템의 해석 가능성과 신뢰성을 개선합니다.

- **Performance Highlights**: TranspNet는 AI 시스템의 신뢰성 및 해석 가능성을 중점적으로 개선하여, 의료 및 금융과 같은 고위험 분야에서의 적용 가능성을 높입니다. 본 접근 방식은 LLM의 예측 불확실성을 해결하고, 구조적이고 해석 가능한 방식으로 결과를 제공하여, 연구 및 산업 응용 모두에서 실용성을 강화합니다.



### Crystal Structure Generation Based On Material Properties (https://arxiv.org/abs/2411.08464)
- **What's New**: 본 논문에서는 예상 자료 특성을 기반으로 결정 구조를 생성하기 위한 Crystal DiT 모델을 제안합니다. 이 모델은 대규모 언어 모델이 예측한 대칭 정보와 자료 특성을 결합하여 나옵니다. 이를 통해 자료 특성과 결정 구조 간의 매핑을 효과적으로 수행할 수 있습니다.

- **Technical Details**: 저자들은 Uni-MDM이라는 보편적인 물질 구조 설계 모델을 제시합니다. 이 모델은 자료 특성과 공간군 정보를 기반으로 두 단계로 나뉘어 결정 구조 생성을 수행합니다. 첫 번째 단계는 GLM4 모델을 통해 필요한 결정 특성에 따라 공간군 정보를 생성하고, 두 번째 단계는 DiT 모델을 사용하여 자료 특성과 공간군 정보를 바탕으로 결정 구조를 생성하는 것입니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 자료 특성과 공간군의 제약 조건 하에서 기대하는 성능 요구를 충족하는 안정적인 결정 구조를 성공적으로 생성할 수 있음을 보여줍니다.



### Symbolic-AI-Fusion Deep Learning (SAIF-DL): Encoding Knowledge into Training with Answer Set Programming Loss Penalties by a Novel Loss Function Approach (https://arxiv.org/abs/2411.08463)
- **What's New**: 본 논문은 온톨로지(ontologies)와 답 집합 프로그래밍(answer set programming, ASP)을 사용하여 도메인 전문 지식을 통합함으로써 딥러닝 모델의 학습 과정을 개선하는 하이브리드 방법론을 제시합니다. 이 접근법은 기존의 딥러닝 모델의 한계를 극복하고, 성능과 신뢰성을 향상시키는 데 기여합니다.

- **Technical Details**: 하이브리드 방법론은 도메인 특정 제약과 규칙을 딥러닝 모델의 손실 함수에 직접 인코딩하여 데이터 기반 학습과 지식 기반 학습을 조화롭게 결합합니다. 이를 통해 모델이 데이터를 학습할 뿐만 아니라 도메인 특정 제약을 준수하도록 합니다. ASP 규칙의 업데이트를 통해 손실 함수의 자동화가 가능하여 다양한 분야에 쉽게 적용될 수 있습니다.

- **Performance Highlights**: 이 방법론은 회귀(regression) 및 분류(classification) 작업 모두에 적용 가능하며, 의료, 자율 시스템, 엔지니어링 및 배터리 제조와 같은 다양한 분야에서 일반화될 수 있음을 보여줍니다. 특히 배터리 제조와 같은 산업 환경에서 도메인 전문 지식의 통합을 실용적으로 지원합니다.



### Towards Optimizing a Retrieval Augmented Generation using Large Language Model on Academic Data (https://arxiv.org/abs/2411.08438)
- **What's New**: 이 논문에서는 Retrieval Augmented Generation (RAG) 방식을 도입하여 특정 도메인 데이터에서 성능을 향상시키기 위한 최첨단 모델들을 다양한 최적화 기술을 통해 평가합니다. 특히 다중 쿼리(Multi-Query), 부모-자식 검색기(Child-Parent-Retriever), 앙상블 검색기(Ensemble Retriever), 그리고 상황 학습(In-Context Learning) 최적화를 포함하여 학술 분야에서의 기능과 성능을 개선합니다.

- **Technical Details**: 우리는 독일의 대형 기술대학교의 다양한 연구 프로그램을 목표로 하는 데이터를 검색하는 데 주력하며, 200개의 질문과 답변으로 구성된 커리큘럼 QA(CurriculumQA) 데이터셋을 구축했습니다. 이 데이터셋은 LLM이 연구 프로그램에 대한 정보로 잘 훈련되지 않았음을 감안하여 RAG 시스템의 성능을 평가하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, 다중 쿼리를 검색 단계에 포함할 때 성능이 크게 향상되는 것을 관찰했습니다. 또한 RAG의 다양한 최적화 접근 방식을 실험하고 평가하여 도메인 특화 데이터에서 최적의 구성을 찾았습니다. 최종적으로 RAG 구성의 효과성을 평가하기 위해 'RAG Confusion Matrix'라는 새로운 평가 접근법을 도입하였습니다.



### Enhanced Classroom Dialogue Sequences Analysis with a Hybrid AI Agent: Merging Expert Rule-Base with Large Language Models (https://arxiv.org/abs/2411.08418)
- **What's New**: 이번 연구에서는 학생 참여 및 깊이 있는 학습을 촉진하는 교실 대화의 분석에서 기존의 이론적 틀과 실증적 기술을 통합하기 위해 포괄적인 대화 시퀀스 규칙 기반(rule-based) 시스템과 대규모 언어 모델(LLM) 기반 인공지능(AI) 에이전트를 개발했습니다.

- **Technical Details**: 개발된 에이전트는 30개 이상의 연구 결과를 종합하여 대화 분석을 위한 포괄적인 프레임워크를 구축하였으며, 전문가 지식과 자연어의 복잡성을 반영하여 교실 대화 시퀀스를 정확하고 유연하게 분류할 수 있도록 설계되었습니다.

- **Performance Highlights**: 에이전트는 인간 전문가 코드와 비교하여 높은 정밀도(precision)와 신뢰성(reliability)을 달성하였으며, 이로 인해 교실 대화 분석의 효율성과 확장성을 크게 향상시켜 교수법 개선 및 교사 전문성 개발에 기여할 수 있는 가능성을 보여주었습니다.



### DiVR: incorporating context from diverse VR scenes for human trajectory prediction (https://arxiv.org/abs/2411.08409)
- **What's New**: 본 연구는 동적인 장면에서 사용자 행동의 경로 예측을 위한 가상 환경(Virtual environments)의 이점을 살펴봅니다. 특히, 기존 연구들이 사용자의 특정 요인(user-specific factors)을 고려하지 않고 정적인 맥락(static contexts)만을 분석해왔음을 지적하며, CREATTIVE3D 데이터셋을 활용해 다양한 상황에서의 경로를 모델링합니다.

- **Technical Details**: 우리는 Perceiver 아키텍처를 기반으로 한 교차 모달(transformer based cross-modal) 기술인 Diverse Context VR Human Motion Prediction (DiVR)을 제안합니다. 이 모델은 이질적인 그래프 합성곱 네트워크(heterogeneous graph convolution network)를 사용하여 정적 및 동적 장면 컨텍스트를 통합하고, MLP, LSTM 및 gaze, point cloud 컨텍스트를 갖춘 다른 transformer 아키텍처와 비교하는 광범위한 실험을 수행하였습니다.

- **Performance Highlights**: DiVR은 높은 정확도와 적응성을 보여주며, 다양한 사용자, 작업, 장면에 걸쳐 일반화 가능성을 테스트합니다. 연구 결과는 DiVR이 다른 모델 및 정적 그래프에 비해 더 높은 성능을 달성함을 나타냅니다. 이 연구는 메타버스(metaverse)에서 사용자 경험을 향상시킬 수 있는 맥락 인식(context-aware) 인간 경로 모델링의 장점을 강조합니다.



### RLInspect: An Interactive Visual Approach to Assess Reinforcement Learning Algorithm (https://arxiv.org/abs/2411.08392)
- **What's New**: 이 연구에서 제안하는 RLInspect는 강화 학습( Reinforcement Learning ) 모델의 동작을 이해하고 잠재적인 문제를 분석하기 위해 개발된 인터랙티브 비주얼 도구입니다. 이 도구는 다양한 구성 요소(상태, 행동, 에이전트 아키텍처 및 보상)를 고려하여 사용자가 RL 훈련 과정을 깊이 있게 분석할 수 있도록 돕습니다.

- **Technical Details**: RLInspect는 상태 공간(state space), 보상(reward), 행동(action)을 평가하기 위한 다양한 인터랙티브 비주얼 플롯을 제공합니다. 이 도구는 입력 및 출력 작업을 관리하는 Data Handler, 다양한 분석을 수행하는 Analyzers, 결과를 HTML 파일로 시각화하는 Report Generator로 구성됩니다. 특히 Incremental Principal Component Analysis (IPCA)를 사용하여 고차원 상태를 저차원 공간에 시각화할 수 있습니다.

- **Performance Highlights**: RLInspect를 활용하면 모델의 동작 이해, 훈련 과정 중 문제 식별 및 효과적인 수정이 가능하여 더 견고하고 신뢰할 수 있는 RL 시스템을 구축할 수 있습니다. Cartpole 환경에서 훈련된 RL 에이전트의 상태 및 행동 분석 결과를 통해 에이전트가 적절한 상태들을 탐색하고 있는지, 훈련이 고르게 이루어졌는지를 판단할 수 있습니다.



### A Fuzzy Reinforcement LSTM-based Long-term Prediction Model for Fault Conditions in Nuclear Power Plants (https://arxiv.org/abs/2411.08370)
- **What's New**: 이 연구는 원자력 발전소(NPP)의 운영 위험을 줄이고 신뢰성을 높이기 위한 새로운 예측 모델을 제안합니다. 이 모델은 Reinforcement Learning (강화 학습)을 Long Short-Term Memory (LSTM) 신경망과 Expert Fuzzy Evaluation Method (전문 퍼지 평가 방법)와 통합하였습니다.

- **Technical Details**: 제안된 모델은 CPR1000 압수형 수조 시뮬레이션 모델의 Main Steam Line Break (MSLB) 사고 조건에서 20가지 서로 다른 파라미터 데이터로 검증되었습니다. 이 모델은 최대 128 스텝 ahead로 NPP 파라미터 변화를 예측할 수 있으며, 각 스텝의 시간 간격은 10초입니다.

- **Performance Highlights**: 이 모델은 NPP의 고장 예측을 위한 시간 진전 요구 사항을 충족하며, 비정상 탐지(anomaly detection) 및 남은 유효 수명(prediction of remaining useful life) 예측과 같은 PHM 응용 프로그램에 대한 효과적인 참고 솔루션을 제공합니다.



### Responsible AI in Construction Safety: Systematic Evaluation of Large Language Models and Prompt Engineering (https://arxiv.org/abs/2411.08320)
Comments:
          29 pages, 5 figures

- **What's New**: 이 연구는 건설 안전 관리에서 AI, 특히 Large Language Models (LLMs)의 책임 있는 통합에 대한 체계적인 평가의 필요성을 강조합니다.

- **Technical Details**: 본 연구는 Certified Safety Professionals (BCSP)에서 실시한 세 가지 표준화된 시험을 통해 두 가지 LLM인 GPT-3.5와 GPT-4o의 성능을 평가하며, 385개의 질문을 통해 안전 지식 영역을 분석합니다. 두 모델은 BCSP 기준을 초과하는 성능을 나타냈고, GPT-4o는 84.6%, GPT-3.5는 73.8%의 정확도를 기록했습니다. 모델의 강점은 안전 관리 시스템 및 위험 식별 및 통제에 있지만, 과학, 수학, 응급 대응, 화재 예방에서는 약점을 보였습니다.

- **Performance Highlights**: LLMs의 성능에는 네 가지 주요 제한 사항이 영향을 미치며, 이는 지식 부족, 추론 오류, 기억 문제 및 계산 오류로 분류됩니다. Prompt engineering 전략의 영향도 두드러지며, 정확도에서 GPT-3.5는 13.5%, GPT-4o는 7.9%의 차이를 보였습니다. 그러나 특정한 prompt 구성이 보편적으로 효과적인 것은 아닙니다. 연구 결과는 안전 관행을 지원할 수 있는 LLM의 영역을 식별하고, 인간의 감독이 여전히 필수적인 부분을 강조하며, LLM 구현 개선을 위한 실용적인 통찰을 제공합니다.



### PerceiverS: A Multi-Scale Perceiver with Effective Segmentation for Long-Term Expressive Symbolic Music Generation (https://arxiv.org/abs/2411.08307)
- **What's New**: 새로운 아키텍처인 PerceiverS(세그멘테이션 및 스케일)를 제안하여, 구조적이며 감정 표현이 풍부한 상징 음악 생성을 위한 효과적인 세그멘테이션과 다중 스케일 주목 메커니즘을 활용합니다.

- **Technical Details**: PerceiverS는 자기 주목(self-attention)과 교차 주목(cross-attention)을 결합하여 음악의 긴 구조적 의존성과 짧은 감정적 세부 사항을 동시에 학습합니다. 이러한 구조는 기초 causal masking 문제를 해결하고, 다중 스케일의 주목을 통해 고유한 생성 방식으로 음악을 만들어냅니다.

- **Performance Highlights**: Maestro 데이터셋을 사용하여 평가한 결과, PerceiverS는 원본 데이터셋에 비해 평균 40%의 Overlap Area 개선을 달성하였습니다. 이는 Perceiver AR보다 높은 품질의 상징 음악 생성을 보여줍니다.



### DNN Task Assignment in UAV Networks: A Generative AI Enhanced Multi-Agent Reinforcement Learning Approach (https://arxiv.org/abs/2411.08299)
- **What's New**: 본 논문에서는 무인항공기(UAV) 스웜을 활용하여 DNN 작업을 효과적으로 할당하고, 이 과정을 통해 지연(latency)을 최소화하는 새로운 접근 방식을 제시합니다. 제안된 알고리즘은 다중 에이전트 강화 학습(MARL)과 생성적 확산 모델(GDM)을 통합하여 UAV의 컴퓨팅 제약을 극복합니다.

- **Technical Details**: 연구진은 UAV의 비행 경로 최적화와 작업 크기를 고려하여 탐색 문제를 해결하고, GDM-MADDPG라는 새로운 DNN 작업 할당 알고리즘을 도입하였습니다. 이 알고리즘은 동적 환경 속에서 에이전트의 관찰을 바탕으로 구체적인 DNN 작업 할당 작업을 생성합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 알고리즘은 AlexNet, ResNet152, VGG-16 모델에 대해 평균적으로 각각 5.25%, 8.18%, 11.02%의 지연을 줄이며, 전반적인 시스템 성능을 높였습니다.



### RESOLVE: Relational Reasoning with Symbolic and Object-Level Features Using Vector Symbolic Processing (https://arxiv.org/abs/2411.08290)
- **What's New**: 이번 논문에서는 RELSOLVE라는 새로운 신경-벡터 상징 구조를 제안합니다. RESOLVE는 고차원 공간에서 객체 수준의 기능과 관계 표현을 결합하여 부분 관계적 추론을 가능하게 하여 트랜스포머 기반 구조의 약점을 개선합니다.

- **Technical Details**: RESOLVE는 객체 수준 특징을 고차원 공간에서 효율적으로 인코딩하기 위해 새로운 HD-attention 메커니즘을 활용합니다. 이 메커니즘은 bipolar 공간 ({−1,1})에서 빠르게 계산되며, 이전에 필요했던 추상적인 규칙에 대한 사전 지식이 필요하지 않습니다. 또한, 주목 점수(matrix multiplication)를 간소화하여 계산 비용을 크게 낮춥니다.

- **Performance Highlights**: RESOLVE는 기존의 최신 방법들과 비교하여, 순수 관계적 추론 작업과 수학 문제 해결과 같은 부분 관계적 작업에서 더 높은 정확도와 일반화 능력을 달성합니다. 또한 낮은 계산 지연(latency)과 메모리 효율성을 제공합니다.



### Challenges in Guardrailing Large Language Models for Scienc (https://arxiv.org/abs/2411.08181)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전이 자연어 처리 및 이해(NLP/NLU)의 분야에 혁신을 가져왔으나, 과학적 연구에 적용할 경우 신뢰도와 과학적 무결성과 관련된 중대한 실패 모드를 드러낸다는 점이 논문에서 새롭게 강조되었습니다. 기존의 일반-purpose LLM 가드레일이 이러한 독특한 도전에 충분하지 않음을 지적하고, 과학 분야에 맞는 LLM 가드레일 배치에 대한 포괄적인 가이드라인을 제시합니다.

- **Technical Details**: 논문에서 제안하는 가이드라인은 특정 도전 과제인 시간 민감성(time sensitivity), 지식 자문(contextualization), 갈등 해결(conflict resolution), 지적 재산권(intellectual property) 문제를 포함하여, 신뢰성(trustworthiness), 윤리 및 편향(ethics & bias), 안전(safety), 법적 측면(legal aspects) 등을 포함하는 가드레일 차원들을 정의합니다. 또한, 백박스(black-box), 화이트박스(white-box), 그레이박스(gray-box) 방법론을 활용한 구현 전략도 상세히 설명됩니다.

- **Performance Highlights**: 과학 연구에 적용되는 LLM은 신뢰할 수 있는 출력을 유지해야 하며, 사실적 정확성(factual accuracy)과 내용 수정(content moderation) 기준을 준수하는 것이 필수적입니다. 기술적인 가이드라인은 특히 민감한 과학 영역에서 우수한 성과를 달성하는 데 중요한 역할을 할 것으로 기대되며, generative AI 기술에 대한 신뢰를 유지하기 위한 필수적인 여러 측면을 다루고 있습니다.



### Retrieval, Reasoning, Re-ranking: A Context-Enriched Framework for Knowledge Graph Completion (https://arxiv.org/abs/2411.08165)
- **What's New**: 이번 논문에서는 Knowledge Graph Completion (KGC) 작업을 위한 새로운 프레임워크인 KGR3를 제안합니다. KGR3는 Retrieval, Reasoning, Re-ranking 세 가지 모듈로 구성되어 있으며, 기존 방법들이 가진 한계를 극복하고 더 나은 결과를 도출합니다.

- **Technical Details**: KGR3는 KGC 작업을 위해 엔티티 맥락을 활용합니다. Retrieval 모듈에서는 KG에서 지원하는 트리플을 수집하고, Reasoning 모듈에서는 대형 언어 모델(LLM)을 사용하여 잠재적 답변을 생성합니다. 마지막으로 Re-ranking 모듈에서는 후보 답변을 통합하고 LLM을 미세 조정하여 최적의 답변을 선택합니다.

- **Performance Highlights**: KGR3는 FB15k237과 WN18RR 데이터셋에서 12.3% 및 5.6%의 Hits@1을 개선하며, 다양한 KGC 방법들보다 성능이 우수함을 입증합니다. KGR3의 적용을 통해 KGC 작업의 성능이 획기적으로 향상되었습니다.



### Adaptive Meta-Learning for Robust Deepfake Detection: A Multi-Agent Framework to Data Drift and Model Generalization (https://arxiv.org/abs/2411.08148)
- **What's New**: 이 논문은 adversarial meta-learning 알고리즘을 제안하여 deepfake 탐지의 일반화, 견고성 및 적응성을 개선하여 효율적인 detection 시스템을 구축하는 방법을 다룹니다.

- **Technical Details**: 제안된 시스템은 task-specific adaptive sample synthesis와 consistency regularization을 포함하는 refinement phase를 사용합니다. 모델의 강점과 약점에 집중하여 robust성 및 generalization을 향상시킵니다. 또한, hierarchical multi-agent retrieval-augmented generation (RAG) workflow를 사용하여 신속하게 데이터 추세에 적응할 수 있도록 합니다.

- **Performance Highlights**: 논문에서 제안한 모델은 다양한 데이터셋에서 일관된 성능을 보여주며 비교된 모델들보다 뛰어난 성능을 발휘하는 것으로 나타났습니다.



### The Universal PDDL Domain (https://arxiv.org/abs/2411.08040)
- **What's New**: 본 논문에서는 AI 계획에서 도메인(domain)과 문제 인스턴스(problem instance)의 구분이 어떻게 이루어지는지에 대해 다루고 있습니다. 특히, 기존의 PDDL(Planning Domain Definition Language)에서 나타나는 도메인의 개념이 어떻게 일반화될 수 있는지를 보여줍니다.

- **Technical Details**: PDDL은 도메인과 문제 인스턴스를 구문적으로 분리하여 나타냅니다. 도메인 정의에는 타입, 서술(predicates), 그리고 행동 스키마(action schemata)가 포함되며, 문제 인스턴스 정의에서는 구체적인(타입화된) 객체 집합, 초기 상태(initial state), 목표 조건(goal condition)을 제공합니다. 본 연구에서는 모든 계획 문제 인스턴스가 이 '보편(universal)' 도메인의 인스턴스가 될 수 있도록 PDDL 도메인을 정의하는 것이 가능하다는 것을 보여 줍니다.

- **Performance Highlights**: 일반 도메인은 매개변수화된 PDDL 도메인으로, 이 도메인의 인스턴스는 임의의 명제적 계획 문제(propositional planning problem)가 됩니다. 모든 PDDL 도메인-문제 쌍을 보편 도메인의 인스턴스로 변환할 수는 있으나, 이를 위해서는 바닥화(grounding)가 필요하며, 이 과정에서 크기가 기하급수적으로 증가할 수 있다는 잠재적인 단점이 언급됩니다.



### 4D Gaussian Splatting in the Wild with Uncertainty-Aware Regularization (https://arxiv.org/abs/2411.08879)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 4D Gaussian Splatting (4DGS)이라는 새로운 알고리즘을 제안하여, casual monocular 동영상에서 역동적인 장면을 재구성하는 방법을 소개합니다. 이 알고리즘은 오버피팅(overfitting) 문제를 해결하기 위해 불확실성을 인지하는 정규화(regularization)를 도입합니다.

- **Technical Details**: 본 연구에서는 우선 Gaussian primitive의 불확실성을 정량화하고, α-blending 방법을 통해 2D 불확실성 맵을 작성하여 선택적으로 정규화를 적용합니다. 또한, Structure from Motion (SfM) 알고리즘의 초기화 문제를 해결하기 위해 동적 지역 밀집화(dynamic region densification) 방법을 사용하여 깊이 맵과 장면 흐름(scene flow)을 기반으로 Gaussian primitive를 초기화합니다.

- **Performance Highlights**: 제안된 방법은 handheld monocular 카메라로 촬영한 영상에서 4DGS 재구성 성능을 향상시키며, 소수의 샷(few-shot) 정적 장면 재구성에서도 유망한 결과를 보여줍니다.



### A Short Note on Evaluating RepNet for Temporal Repetition Counting in Videos (https://arxiv.org/abs/2411.08878)
- **What's New**: 이번 논문에서는 비디오 반복 카운팅 데이터셋에서 RepNet의 평가에 관한 일관된 문제를 다루고 있습니다. 이를 해결하기 위해 RepNet의 다양한 데이터셋에서의 성능 결과를 보고하고, 평가 코드 및 RepNet 체크포인트를 공개했습니다.

- **Technical Details**: RepNet 모델은 원래 32 프레임 이상을 예측할 수 있으며, 이는 모델 수정 없이 다양한 속도로 비디오를 재생하는 Multi-speed Evaluation 기술로 가능하다는 점을 강조합니다. 평가 지표로 Off-by-one Accuracy (OBOA)와 Mean Absolute Error (MAE)를 사용하며, 다양한 데이터셋인 Countix, UCFRep, RepCount-A에서 RepNet의 성능 결과를 보고합니다.

- **Performance Highlights**: RepNet은 Countix 데이터셋에서 최신 방법보다 뛰어난 성능을 보였으며, UCFRep에서도 TransRAC보다 유의미하게 더 나은 성능을 기록했습니다. RepCount-A 데이터셋에서도 반복 구간이 있는 비디오에 대해 효과적으로 작동하며, RepNet의 기본 성능이 과거 수정된 모델보다 훨씬 강력함을 보여주었습니다.



### The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models (https://arxiv.org/abs/2411.08870)
Comments:
          Extended version of EMNLP 2024 paper arXiv:2411.04118. Includes additional results on clinical note QA tasks and supervised fine-tuning evaluations

- **What's New**: 이번 연구는 의료 분야에 적합한 언어 모델의 성능을 재검토한 것으로, 기존의 특정 의료 데이터에 대한 적응 훈련(DAPT)이 실제로 기대한 만큼의 성능 향상을 제공하지 않는다고 주장합니다. 22.7%의 경우에서만 의료 모델이 기본 모델보다 성능이 우수하며, 40.5%의 경우에는 성능이 열악한 결과를 보였습니다.

- **Technical Details**: 10개의 공개된 "의료" LLM과 2개의 VLM을 대상으로 기본 모델과의 성능 비교를 수행했습니다. 연구에서 다양한 의료 QA 작업을 통해 의료 LLM이 기본 모델보다 일관되게 개선되지 않음을 명확히 하였습니다. 이 과정에서 개별 최적의 프롬프트(selecting the ‘best’ prompt)를 사용하여 수행하였으며 통계적 불확실성을 고려하여 분석했습니다.

- **Performance Highlights**: 의료 LLM은 의료 지식 QA 작업에 대해서는 유의미한 개선을 보였지만, 임상 메모 기반의 QA 작업에서는 개선되지 않았고, 의료 VLM은 모든 시각적 미디어 QA 작업에서 거의 개선되지 않는 결과를 보였습니다. 또한, DAPT의 성능 이점을 평가하기 위해서는 엄격한 쌍비교(observational comparison)가 필수적이라는 점을 강조했습니다.



### Interaction Testing in Variation Analysis (https://arxiv.org/abs/2411.08861)
- **What's New**: 이 논문은 전통적인 매개 분석(mediation analysis)의 한계를 극복하고 관찰적인(regime) 환경에서의 원인-결과 관계를 설명하는 새로운 방법론인 변동 분석(variation analysis)을 제시합니다. 이 방법은 평균 처리 효과(ATE) 대신 총 변동(TV) 측정값에 중점을 두어, 원인 X와 결과 Y 사이에서 원인을 제대로 이해할 수 있도록 돕습니다.

- **Technical Details**: 변동 분석(variation analysis)은 총 변동(TV) 측정을 통해 원인 X와 결과 Y 사이의 직접적(direct), 간접적(indirect), 그리고 혼합(confounded) 변동을 분해할 수 있는 방법입니다. 이는 피험자 관찰(interventional) 상황에 한정되지 않고, 자연 환경(natural regime)에서 원인과 결과의 관계를 설명하는 데 유용합니다. 또한, 상호작용 테스트(interaction testing)를 도입하여 다양한 경로 간의 상호작용이 통계적으로 유의미한지를 확인합니다.

- **Performance Highlights**: 새로운 변동 분석 방법은 기존의 ATE 기반 분석이 설명하지 못했던 관찰적 데이터 내의 원인-결과 관계를 효과적으로 설명할 수 있는 가능성을 제공합니다. 이를 통해 예를 들어, 암 치료를 받는 환자의 사망률이 높은 이유와 같은 복잡한 인과 관계에 대한 이해를 돕습니다.



### Data-driven Surface Solar Irradiance Estimation using Neural Operators at Global Sca (https://arxiv.org/abs/2411.08843)
- **What's New**: 이 논문은 표면 태양 복사 (SSI) 예측을 위한 혁신적인 접근 방식을 제시합니다. 최근의 수치 기상 예측 (NWP) 및 데이터 기반 기계 학습 날씨 모델의 발전을 활용하여 SSI의 6시간 평균 예측을 위한 세계적 규모의 적응형 글로벌 프레임워크를 개발했습니다.

- **Technical Details**: 논문에서는 NWP 및 AI 기반 날씨 모델에 의해 예측된 변수를 활용하여 SSI를 추정하는 유연한 모델을 설명합니다. NVIDIA Modulus를 사용하여 개발된 이 모델은 인공위성 데이터를 통해 미세 조정 가능하며, 이렇게 개선된 성능은 태양 에너지가 전력망에 통합되는 데 중대한 영향을 미칩니다.

- **Performance Highlights**: 이 개선된 SSI 예측의 정확성은 재생 가능 에너지 소스에 대한 글로벌 전환을 지원하며, 에너지 관리의 효율성을 크게 향상시키는 데 기여합니다.



### AstroM$^3$: A self-supervised multimodal model for astronomy (https://arxiv.org/abs/2411.08842)
- **What's New**: 본 논문에서는 여러 관측 모드를 활용하여 새로운 천문학적 멀티모달 데이터 세트를 구축하고, AstroM$^3$라는 자기 지도 학습(self-supervised learning) 접근 방식을 제안합니다. 이 방법은 CLIP(Contrastive Language-Image Pretraining) 모델을 확장하여 시간 시리즈 포토메트리 데이터, 스펙트럼 및 천문학적 메타데이터를 통합할 수 있도록 합니다.

- **Technical Details**: AstroM$^3$는 CLIP 모델을 trimodal 설정으로 확장하여 다양한 관측 모드에서 학습할 수 있는 기반을 제공합니다. 이는 시간 시리즈 포토메트리, 스펙트럼 및 메타데이터를 포함하여 데이터의 이질적인 조합을 효과적으로 통합하는 것을 목표로 합니다. 모델은 실제 관측 데이터로 직접 학습되며, 미분류 데이터에 대한 분류 정확도가 84.6%에서 91.5%로 향상됩니다.

- **Performance Highlights**: 실험 결과에 따르면, CLIP의 자기 지도 학습(pre-training)은 레이블이 제한적일 때 분류 정확도를 12.6%까지 향상시키며 긍정적인 결과를 보였습니다. 또한, 학습된 임베딩(embeddings)을 사용하여 잘못 분류된 사례를 식별하고 유사도 검색, 이상 탐지 등의 다른 다운스트림 작업에서도 효과적으로 활용될 수 있음을 보여줍니다. 이 모델을 통해 새로운 하위 유형의 별 (예: Mira subtype)과 두 가지 회전 변수 클래스를 재발견한 점이 주목할 만합니다.



### Offline Adaptation of Quadruped Locomotion using Diffusion Models (https://arxiv.org/abs/2411.08832)
- **What's New**: 본 논문에서는 다중 기술 간 학습 및 보행 동작의 오프라인 적응의 한계를 동시에 해결하는 확산 기반 접근 방식을 제시합니다.

- **Technical Details**: 분산 모델(difffusion model)을 활용하여 목표 조건의 행동(goal-conditioned behaviour)을 알고리즘이 훈련된 후에도 오프라인에서 최적화할 수 있는 방식으로, classifier-free guided diffusion을 사용하여 보행 궤적을 최적화합니다. 이를 통해 로봇의 온보드 CPU에서 완전히 실행할 수 있는 정책을 학습합니다.

- **Performance Highlights**: ANYmal 사족 보행 로봇에서 하드웨어 실험을 통해 접근 방식의 유효성을 검증하였으며, 빠른 샘플링 및 고품질 동작 생성이 가능함을 입증하였습니다.



### Can sparse autoencoders be used to decompose and interpret steering vectors? (https://arxiv.org/abs/2411.08790)
- **What's New**: 본 논문에서는 steering vectors의 해석을 위해 sparse autoencoders (SAEs)를 적용하는 것의 한계점을 조사했습니다. 특히 SAEs가 직접적으로 steering vectors에 적용될 경우 잘못된 결과를 나타내는 두 가지 이유를 식별했습니다: (1) steering vectors가 SAEs가 설계된 입력 분포 밖에 위치하고, (2) steering vectors가 SAEs가 수용할 수 없는 의미 있는 음수 프로젝션을 가질 수 있습니다.

- **Technical Details**: steering vectors는 모델의 행동을 조정하기 위해 중간 모델 활성화에 추가되는 벡터 표현입니다. 반면, sparse autoencoders (SAEs)는 모델 활성화의 희소한 비음수 선형 조합으로 분해하는 방법입니다. 본 연구는 SAEs 적용 시 steering vectors의 왜곡된 분해를 일으키는 원인들을 경험적 실험을 통해 검토합니다. 특히, SAEs의 입력 분포 및 음수 프로젝션 처리 능력의 한계가 문제로 지적됩니다.

- **Performance Highlights**: 이 연구에서 제안된 내용을 통해, SAEs가 steering vectors에 대한 해석에서 직접적인 사용에 있어 제약이 있음을 강조하고 있습니다. 이는 향후 steering vectors의 해석 방법 개선을 위한 기초 자료를 제공할 것으로 기대됩니다.



### Zero-shot Cross-lingual Transfer Learning with Multiple Source and Target Languages for Information Extraction: Language Selection and Adversarial Training (https://arxiv.org/abs/2411.08785)
- **What's New**: 본 연구는 기존의 제로샷 크로스링구얼 싱글 트랜스퍼 접근 방식에서 벗어나 여러 언어 간의 정보 추출 성능을 높이기 위한 다중 트랜스퍼(Multi-Transfer) 가능성을 조사합니다. 또한, 언어 간의 거리 값을 기반으로 한 클러스터링 기법을 통해 효율적인 데이터 수집 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 언어 간 에서의 단일 전이 성능과 언어적 거리 간의 상관관계를 분석합니다. URIEL 데이터베이스를 활용하여 언어의 계통학적 및 유형론적 속성을 추출하고, 이를 바탕으로 다양한 언어 간 거리(유사성 점수)를 계산합니다. 또한, 다중 언어 비지도 데이터를 활용한 관계 전이(relational-transfer) 학습을 제안합니다.

- **Performance Highlights**: 양적 실험 및 질적 분석 결과, 제안된 방법론에 따르면 각 언어에 대한 전이 성능이 크게 향상되었습니다. 특히 연관 전이 설정(ZSCL-R)에서 다수의 언어로부터 수집된 비지도 데이터의 활용을 통해 전반적인 모델 성능이 개선되는 결과를 보여주었습니다.



### Sharingan: Extract User Action Sequence from Desktop Recordings (https://arxiv.org/abs/2411.08768)
- **What's New**: 본 논문은 데스크탑 비디오 기록에서 사용자 행동을 추출하기 위한 두 가지 새로운 Vision-Language Model (VLM) 기반 방법을 제안합니다: Direct Frame-Based Approach (DF)와 Differential Frame-Based Approach (DiffF)입니다.

- **Technical Details**: DF 접근법은 샘플링한 비디오 프레임을 VLM에 직접 입력하는 방식이며, DiffF는 컴퓨터 비전 기법을 통해 감지된 프레임 차이를 명시적으로 포함한 후 VLMs로 해석합니다. 이 연구는 각 접근법의 성능을 평가지원 데이터 세트를 사용하여 평가합니다.

- **Performance Highlights**: DF 접근법은 사용자 행동 식별에서 70%에서 80%의 정확도를 달성했습니다. 추출된 행동 시퀀스는 Robotic Process Automation (RPA) 프로세스를 통해 재생할 수 있습니다. 명시적인 UI 변화가 성능 저하를 가져올 수 있으므로 DF 접근법이 보다 신뢰할 수 있는 것으로 판단됩니다.



### SANDWICH: Towards an Offline, Differentiable, Fully-Trainable Wireless Neural Ray-Tracing Surroga (https://arxiv.org/abs/2411.08767)
Comments:
          Submitted in ICASSP 2025

- **What's New**: 본 논문에서는 기존의 온라인 학습 방법에 대한 대안을 제시하는 장면 인식 신경 결정 무선 채널 레이 트레이싱 계층(SANDWICH)을 소개합니다. SANDWICH는 오프라인 방식에서 완전히 차별화 가능하며 GPU에서 훈련할 수 있는 혁신적인 접근 방식입니다.

- **Technical Details**: SANDWICH는 무선 레이 트레이싱을 시퀀스 의사결정 문제로 재정의하여 생성 모델을 활용해 각 환경 내 광학, 물리 및 신호 속성을 공동 학습합니다. 이 과정에서 변환기에 의해 시퀀스를 모사하고, 마르코프 의사결정 과정(MDP)을 통해 희소한 감독을 통합하는 방법을 사용합니다. 이 방법은 실시간 피드백 없이도 레이의 경로를 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: SANDWICH는 기존의 온라인 학습 솔루션에 비해 레이 트레이싱 정확도가 4e^-2 라디안 향상되며, 채널 게인 추정에서 0.5 dB의 성능 저하만 발생합니다. 논문에서 제안한 방법은 GPU에서 전방향 훈련이 가능하며, 완전 차별화되고 벡터화 가능하다는 장점을 갖고 있습니다.



### Flow reconstruction in time-varying geometries using graph neural networks (https://arxiv.org/abs/2411.08764)
- **What's New**: 이 논문은 매우 희소한 데이터로부터 유량을 재구성하기 위한 Graph Attention Convolutional Network (GACN)을 소개합니다. 이 모델은 이웃 노드의 정보를 활용하여 빠진 특징을 초기화하는 feature propagation 알고리즘을 전처리 단계로 포함하고 있습니다. 또한, 원래 데이터 포인트와 전파된 데이터 포인트를 구분하기 위한 유효성 마스크로서의 이진 지표가 도입되었습니다.

- **Technical Details**: GACN은 3차원 Direct Numerical Simulations (DNS) 데이터 세트에 기반하여 훈련되었으며, 이는 역동적으로 변화하는 격자와 다양한 해상도를 포함합니다. GACN은 그래프의 노드 위치를 통해 데이터의 비구조적인 성질을 포착하고, 노드 간의 엣지 거리 특성을 통해 해상도의 변화를 고려합니다. FP 알고리즘이 결합되어 결측 특징을 초기화하고 BI가 네트워크에 원본 데이터 포인트와 전파된 데이터 포인트를 알려줍니다.

- **Performance Highlights**: GACN은 학습 중에 고려되지 않았던 실험적인 Particle Image Velocimetry (PIV) 데이터 세트와 예전에 보지 못한 DNS 데이터에서 테스트 되었고, 각각의 테스트 세트에서 전통적인 Convolutional Neural Network (CNN) 및 삼차 보간 방법보다 일관되게 더 낮은 재구성 오류를 달성했습니다. 특히, GACN은 훈련 중 관찰된 영역보다 최대 14배 큰 도메인에서도 효과적으로 유동 필드를 재구성할 수 있음을 보여주었습니다.



### Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers (https://arxiv.org/abs/2411.08745)
Comments:
          12 pages, 10 figures, previously published under the title "How Do Llamas Process Multilingual Text? A Latent Exploration through Activation Patching" at the ICML 2024 mechanistic interpretability workshop this https URL

- **What's New**: 본 논문은 멀티링구얼 언어 모델링의 중심 질문인 대형 언어 모델(LLM)이 특정 언어와 무관하게 보편적인 개념 표현을 발전시키는지를 분석합니다. 특히, transformer 기반 LLM에서 단어 번역 작업 중에 숨겨진 표현(latents)을 분석하여 개념과 언어 간의 관계를 탐구합니다.

- **Technical Details**: 연구자들은 단어 번역 작업의 소스 번역 프롬프트에서 latents를 추출하고 이를 타겟 번역 프롬프트의 전방 전달 과정에 삽입합니다. 이를 통해 early layer에서 출력 언어가 latents에 인코딩되어 있고, 변환할 개념은 이후 과정에서 처리된다는 사실을 발견합니다. 두 가지 주요 실험, 즉 activations 패칭을 사용하여 언어를 변경하지 않고 개념을 변경하는 것과 다양한 언어 간의 평균 latents로 패칭하여 성능 향상을 입증합니다.

- **Performance Highlights**: 연구 결과는 Llama 2 7B 모델을 사용하여 번역 작업 성능이 평균 개념 표현을 사용하는 경우 향상되며, 이는 개념이 언어와 독립적으로 표현됨을 지지합니다. 본 분석은 Llama 2 뿐만 아니라 다양한 transformer 모델의 성능에도 일반화된다고 제안합니다.



### QCG-Rerank: Chunks Graph Rerank with Query Expansion in Retrieval-Augmented LLMs for Tourism Domain (https://arxiv.org/abs/2411.08724)
- **What's New**: 이 논문은 RAG(Retrieval-Augmented Generation)의 한계를 극복하기 위해 QCG-Rerank 모델을 제안합니다. 이 모델은 처음에 쿼리와 관련된 후보 청크를 검색한 후, 세분화된 정보를 추출하여 원래 쿼리를 확장합니다. 그런 다음, 확장된 쿼리와 후보 청크를 활용하여 유사도 점수를 계산하고, 이를 바탕으로 청크 그래프를 구축합니다.

- **Technical Details**: QCG-Rerank 모델은 간단한 쿼리에서 핵심적인 정보를 추출하고 이를 바탕으로 쿼리의 의미적 복잡성을 확장합니다. 초기 검색 결과의 유사성을 기반으로 청크 간의 전이 확률을 반복적으로 계산하여 수렴할 때까지 진행하며, 최고 점수를 가진 청크를 LLM에 입력하여 결과를 생성합니다. 이 과정은 Cultour, IIRC, StrategyQA, HotpotQA, SQuAD, MuSiQue 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, QCG-Rerank 방법의 효과성과 우수성을 입증하였습니다. 이 모델은 여행 도메인에서 LLM(대형 언어 모델)의 환각(hallucination) 문제를 완화하고, 보다 정확한 응답을 생성할 수 있도록 합니다.



### Searching Latent Program Spaces (https://arxiv.org/abs/2411.08706)
Comments:
          Code available at this https URL

- **What's New**: 본 논문에서는 Latent Program Network (LPN)라는 새로운 알고리즘을 제안합니다. LPN은 테스트 시간에 적응할 수 있는 구조를 지닌 일반적인 프로그램 유도 기법으로, 연속 공간에서 잠재 프로그램의 분포를 학습하며, 이를 통해 더 효율적인 탐색과 적응을 가능하게 합니다.

- **Technical Details**: LPN은 테스트 시간에 적응을 위한 메커니즘을 신경망 아키텍처에 직접 통합하며, 파라미터 업데이트 없이 다양한 잠재 프로그램을 모델링하기 위해 연속적인 잠재 공간을 활용합니다. 우리는 텍스트 기반의 프로그래밍이 아닌, 픽셀 단위로 직접 출력을 생성하는 디코더를 사용하여 특정 입력에 대해 프로그램을 실행합니다.

- **Performance Highlights**: LPN은 ARC-AGI와 같은 어려운 프로그램 합성 벤치마크에서 훈련 분포를 초과하여 일반화하고, 보지 못한 작업에 적응할 수 있음을 보여주며, 테스트 시간 적응 메커니즘을 갖춘 알고리즘보다 우수한 성능을 기록했습니다.



### MVKTrans: Multi-View Knowledge Transfer for Robust Multiomics Classification (https://arxiv.org/abs/2411.08703)
- **What's New**: 본 논문에서는 다중 오믹스(multiomics) 예측의 고유한 도전과제를 해결하기 위해 새로운 MVKTrans(multi-view knowledge transfer learning) 프레임워크를 제안합니다. 이는 데이터 이질성과 편향 전이를 억제하여 분류 성능을 향상시키는 방식으로, 오믹스 간의 지식을 전이하는 방법을 동적으로 조정합니다.

- **Technical Details**: MVKTrans 프레임워크는 그래프 대조 모듈을 설계하여 레이블이 없는 데이터로 학습하며, 내부 오믹스 패턴을 효과적으로 학습하고 전이하는데 초점을 맞춥니다. 비지도 사전 학습은 각 모달리티에 대한 일반적이고 편향 없는 표현을 촉진하며, 다양한 질병과 샘플에 따라 모달리티의 구분 능력이 달라지므로, 적응형 상호 오믹스 증류 모듈을 도입하여 정보가 풍부한 오믹스에서 덜 정보가 풍부한 오믹스로 지식을 동적으로 전이합니다.

- **Performance Highlights**: 네 가지 실제 생물의학 데이터 세트를 사용한 대규모 실험을 통해 MVKTrans의 우수한 성능과 강건성을 입증했습니다. 또한, 다양한 ablation(중단) 및 변형 연구를 통해 MVKTrans의 효과성과 강건함이 확인되었습니다.



### TRACE: Transformer-based Risk Assessment for Clinical Evaluation (https://arxiv.org/abs/2411.08701)
- **What's New**: 이번 연구에서는 TRACE(Transformer 기반 임상 평가 리스크 평가)라는 신규 방법을 제안합니다. 이는 임상 데이터를 기반으로 리스크 평가를 진행하며, self-attention 메커니즘을 활용하여 특징 상호작용과 결과 해석을 개선합니다.

- **Technical Details**: 제안된 아키텍처는 다양한 데이터 모드인 연속형, 범주형 및 체크박스 속성을 처리할 수 있으며, 각 데이터 모드의 특화된 임베딩을 통합하여 임상 데이터의 공유 표현을 얻습니다. 이를 통해 Transformer encoder 레이어를 사용하여 고위험 대상을 탐지할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 비정형 다층 perceptron(MLP)을 기반으로 한 강력한 기준선을 설정하고, 다양한 임상 리스크 평가에 널리 사용되는 여러 기준선들을 초월하며, 누락된 값을 효과적으로 처리했습니다. 성능 측면에서도 제안된 모델은 경쟁력 있는 성과를 나타내며, clinicians이 모델 결과를 쉽게 해석할 수 있는 설명 가능성을 제공합니다.



### Rethinking negative sampling in content-based news recommendation (https://arxiv.org/abs/2411.08700)
- **What's New**: 이 논문에서는 뉴스 추천 시스템의 모델 정확성을 크게 향상시키기 위해 개인화된 부정 샘플링 기법을 제안합니다.

- **Technical Details**: 부정 샘플링 기법은 사용자 패턴을 학습하기 위한 더 나은 암묵적 부정 예제를 제공하며, 각 사용자에 대해 별도의 경량화된 신경 추천기를 훈련하는 분산 학습 전략을 제안합니다. 이는 사용자의 개인 정보 보호 및 자율성을 개선할 수 있게 합니다.

- **Performance Highlights**: MIND 데이터셋을 사용한 실험 결과, 제안된 방법의 정확성이 최첨단(State-of-the-Art) 모델과 경쟁할 수 있음을 보여줍니다. 또한, 이 샘플링 기법은 모델 복잡성을 줄이고 훈련 과정을 가속화하며, 높은 정확도를 유지할 수 있게 합니다.



### Scholarly Wikidata: Population and Exploration of Conference Data in Wikidata using LLMs (https://arxiv.org/abs/2411.08696)
Comments:
          17 pages, accepted at EKAW-24

- **What's New**: 이 논문은 학술 데이터의 접근성을 높이기 위해 Wikidata의 인프라를 활용하고, 대규모 언어 모델(LLM)을 통해 학회 메타데이터를 자동으로 추출하는 방법론을 제안합니다. 이로 인해 학술 데이터를 더욱 지속 가능하게 유지할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 우리는 학술 데이터를 표현하기 위한 존재론(ontologies)을 분석하고, Wikidata의 주요 개체와 속성으로 매핑합니다. LLM을 활용하여 학회 메타데이터를 효율적으로 추출하고, OpenRefine를 통한 사람 검증(Human-in-the-loop validation) 과정을 통해 데이터를 정제하여 Wikidata에 입력합니다. 이 방법론을 통해 총 6000개 이상의 새로운 개체가 Wikidata에 추가되었습니다.

- **Performance Highlights**: 이 연구는 105개의 의미 웹(Semantic Web) 관련 학회의 데이터를 Wikidata에 추가하여 기존의 데이터를 확장하고, 시각화 도구(Scholia 및 Synia)를 개선하여 새로운 정보를 보다 효율적으로 탐색할 수 있게 합니다. LLM을 활용한 데이터 추출과 검증 과정을 통해 신뢰도를 향상시킵니다.



### A Survey on Vision Autoregressive Mod (https://arxiv.org/abs/2411.08666)
- **What's New**: 이 논문은 자연어 처리(NLP)에서 성공을 거둔 autoregressive (AR) 모델을 컴퓨터 비전에 적용하려는 최근의 연구를 다룹니다. 새로운 비전 과제, 특히 이미지 생성 및 이해, 다중 모달(multi-modal) 생성과 같은 다양한 비전 작업에 적용된 AR 모델에 대한 체계적인 리뷰를 제공합니다.

- **Technical Details**: AR 모델은 이미지 데이터를 비주얼 토큰(visual tokens)으로 표현하여 다음 토큰 예측(next-token prediction)을 수행합니다. 이는 비전 생성 및 이해를 통합하려는 시도로, Transformer 아키텍처를 기반으로 큰 텍스트-이미지 쌍에서 학습할 수 있게 합니다. 또한, AR 모델은 비디오 생성, 의학 이미지 분석 등 다양한 비전 작업에도 적용되고 있습니다.

- **Performance Highlights**: 논문은 AR 모델의 성능을 벤치마킹하고 다양한 평가 데이터셋에서 기존 방법을 분석합니다. 또한, AR 모델을 활용하여 비전 과제를 수행하는 데 있어 주요 도전 과제와 향후 연구 방향을 제시하며, AR 모델이 비전의 다양한 작업에서 잠재력을 가지고 있음을 강조합니다.



### Estimating unknown parameters in differential equations with a reinforcement learning based PSO method (https://arxiv.org/abs/2411.08651)
- **What's New**: 이 논문에서는 미분 방정식의 파라미터 추정 문제를 최적화 문제로 재구성하고, 강화 학습 기반의 입자 군집 최적화(Particle Swarm Optimization, PSO) 알고리즘의 새로운 방법인 DERLPSO를 제안합니다. DERLPSO는 초기 파라미터 값에 독립적이며, 높은 정확도와 강한 안정성을 제공합니다.

- **Technical Details**: DERLPSO는 여러 새로운 전략(로그 초기화, 재초기화 메커니즘, 아래에서 위로의 업데이트 전략)을 통합하여 RLLPSO를 개선하였습니다. 이 방법은 Lorenz, FitzHugh-Nagumo 및 Lotka-Volterra 방정식과 같은 세 가지 일반 미분 방정식(Ordinary Differential Equations, ODE)에 대해 테스트되었습니다.

- **Performance Highlights**: DERLPSO는 평균 제곱 오차(Mean Square Error, MSE) 1.13e-05를 달성하여 기존 방법에 비해 약 4배의 정확도 향상을 보여줍니다. 이 방법은 부분 미분 방정식(Partial Differential Equations, PDE)의 파라미터 추정에도 큰 잠재력을 보입니다.



### A System Level Performance Evaluation for Superconducting Digital Systems (https://arxiv.org/abs/2411.08645)
Comments:
          8 figures

- **What's New**: 본 논문에서는 Superconducting Digital (SCD) 기술을 활용하여 차세대 대규모 컴퓨팅 워크로드의 성능을 향상시킬 수 있는 잠재력을 제시합니다. SCD 장치가 에너지 소비를 줄이고 계산 성능을 향상시키는 방법을 다룬 교차 레이어 모델링 접근법을 통해 대형 언어 모델 (LLM)의 학습 및 추론에서 성능 이점을 평가하는 내용을 포함하고 있습니다.

- **Technical Details**: SCD 아키텍처는 NbTiN 기반의 장치와 Pulse Conserving Logic (PCL) 설계 원칙을 이용하여, 대형 언어 모델 학습 및 추론을 위한 시스템 아키텍처를 구성합니다. 스케일링이 가능한 NbTiN 기반의 SCD 빌딩 블록, JSRAM, 그리고 패키징 솔루션을 사용하며, 성능 모델링 프레임워크를 통해 SCD 시스템 아키텍처의 성능을 예측합니다. 이 기술은 기존 CMOS 기술의 메모리 및 인터커넥트 제한을 해결할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: SCD 아키텍처는 20배 이상의 주파수에서 작동할 수 있으며, 에너지 효율적인 통신을 제공하여 CMOS 기술에 비해 10000배 이상의 에너지 효율성을 지니고 있습니다. 학습 및 추론에서 SCD 시스템의 성능 향상이 GPU 시스템에 비해 크게 나타나는 것이 실험 데이터로 입증되었습니다.



### Towards More Accurate Fake Detection on Images Generated from Advanced Generative and Neural Rendering Models (https://arxiv.org/abs/2411.08642)
Comments:
          13 pages, 8 Figures

- **What's New**: 이번 연구에서는 Neural Radiance Fields와 3D Gaussian splatting과 같은 신경망 기반의 시각 데이터 생성 기술이 GANs와 diffusion 모델의 강력한 대안이 될 수 있음을 보여줍니다. 특히, 강력한 탐지 방법의 필요성을 강조하며 자율적으로 학습할 수 있는 새로운 기술을 제안합니다.

- **Technical Details**: 제안된 접근법은 Fourier 스펙트럼의 크기에서 포괄적인 특징을 추출하는 비지도 학습(unsupervised training) 기술을 포함합니다. 스펙트럼의 중심 대칭 성질로 인한 재구축 문제를 극복하기 위해 스펙트럼 영역과 공간 영역 정보를 동적으로 결합하여 강력한 다중 모드 탐지기(multimodal detector)를 생성합니다.

- **Performance Highlights**: 이 다중 모드 탐지기는 최신 이미지 합성 기법으로 생성된 도전적인 합성 이미지들을 식별하는 데 있어 뛰어난 일반화 능력을 보입니다. 또한, 3D 신경 렌더링 기반의 가짜 이미지 데이터베이스가 부족한 문제를 해결하기 위해 다양한 신경 렌더링 기술로 생성된 이미지를 포함하는 포괄적인 데이터베이스를 개발하였습니다.



### DipMe: Haptic Recognition of Granular Media for Tangible Interactive Applications (https://arxiv.org/abs/2411.08641)
Comments:
          17 pages, 10 figures

- **What's New**: DipMe는 다양한 유형의 granular material을 실시간으로 인식할 수 있는 스마트 장치로, 기존의 시각 기반 솔루션 외에도 촉각 신호를 활용하여 granular media를 인식합니다.

- **Technical Details**: 이 장치는 사용자가 granular media에 집중하하여 촉각 신호를 수집하고, 머신러닝 기법을 통해 92.78%의 정확도로 granualr media를 식별할 수 있습니다. DipMe는 프로빙(probing) 작용을 통해 force와 torque 신호를 수집하며, 이를 활용하여 다중 시간 시퀀스 분류 문제를 해결합니다.

- **Performance Highlights**: DipMe를 통해 사용자들은 다양한 granular media와 상호작용할 수 있는 여러 응용 프로그램을 제시하였으며, 이는 새로운 tangible user interface(TUI)를 개발할 수 있는 가능성을 보여줍니다.



### Precision-Focused Reinforcement Learning Model for Robotic Object Pushing (https://arxiv.org/abs/2411.08622)
- **What's New**: 이번 연구에서는 로봇이 다양한 물리적 특성을 가진 객체를 보다 정밀하게 목표 위치로 밀 수 있도록 하는 새로운 메모리 기반 비전-자기감각 (vision-proprioception) 강화 학습 (RL) 모델을 도입했습니다. 이 모델은 물체를 밀 때 필요한 수정을 줄여줍니다.

- **Technical Details**: 비전-자기감각 모델은 객체의 모든 에피소드 히스토리를 제공받아 훈련 중 객체 파라미터 샘플링을 개선하며, 제어 특성을 추출하기 위해 gated recurrent unit (GRU)를 사용합니다. 로봇은 다양한 물리적 특성을 가진 객체를 목표 위치로 밀어야 하며, 특정 threshold를 555.5cm에서 111.1cm로 줄였습니다.

- **Performance Highlights**: 이 연구의 성과는 목표 지점과 물체의 중심 사이의 거리를 줄이며, 이전 모델들에 비해 수정 동작의 필요성을 줄여 정확도를 향상시켰습니다. 실험 설정은 Franka Emika Panda 로봇을 사용하여 커스터마이즈된 시뮬레이션 환경에서 진행되었습니다.



### Lo-MARVE: A Low Cost Autonomous Underwater Vehicle for Marine Exploration (https://arxiv.org/abs/2411.08605)
Comments:
          This paper was presented at the 12th International Conference on Control, Mechatronics and Automation (ICCMA 2024), held in London, UK, from November 11-13, 2024

- **What's New**: 이 논문에서는 저비용 해양 자율 로봇 차량 탐험기(Lo-MARVE)라는 새로운 자율 수중 차량(AUV)을 소개합니다. Lo-MARVE는 얕은 수역에서 해양 탐사와 환경 모니터링을 위한 저비용 솔루션을 제공하도록 설계되었습니다. 이 AUV는 모듈형 설계와 저비용 센서, 무선 통신 기능을 갖추고 있으며, 총 비용은 약 500 유로입니다.

- **Technical Details**: Lo-MARVE는 Raspberry Pi 4B 마이크로프로세서를 사용하여 개발되었으며, 제어 소프트웨어는 Python으로 작성되었습니다. AUV는 100mm 직경과 300mm 길이의 원통형 본체를 가지고 있으며, 전면에는 투명한 아크릴 디스크가 씌워져 있습니다. 4개의 모터가 장착되어 있으며, 깊이 측정을 위한 압력 센서도 포함되어 있습니다.

- **Performance Highlights**: Lo-MARVE는 실험실 외부의 Galway, Ireland의 River Corrib에서 현장 테스트를 통해 자율 탐색 및 데이터 수집 능력을 입증했습니다. AUV의 성공적인 배치는 그 개념 증명을 확인해주는 중요한 결과입니다.



### DeepUQ: Assessing the Aleatoric Uncertainties from two Deep Learning Methods (https://arxiv.org/abs/2411.08587)
Comments:
          Accepted to the Machine Learning for Physical Sciences workshop at NeurIPS 2024; 11 pages, 2 figures, 2 tables

- **What's New**: 이 연구에서는 불확실성 정량화(Uncertainty Quantification, UQ) 기술인 Deep Ensembles (DE)와 Deep Evidential Regression (DER)을 통해 aleatoric uncertainty를 체계적으로 비교합니다. 이 연구는 0차원 (0D) 및 2차원 (2D) 데이터에서의 UQ 방법의 작동 방식을 탐구하고, 입력 및 출력 변수에 주입된 불확실성을 조사합니다.

- **Technical Details**: Deep Ensembles (DE) 및 Deep Evidential Regression (DER)은 aleatoric uncertainty의 추정치를 제공하는 두 가지 UQ 기술입니다. 이 연구에서는 0D와 2D 데이터에 대해 불확실성을 주입하고, 이러한 입력 불확실성이 출력 변수에 미치는 영향까지 분석합니다. aleatoric uncertainty는 네트워크 예측의 표준 편차의 평균으로 정의됩니다.

- **Performance Highlights**: 모든 실험에서 aleatoric uncertainty는 주입된 노이즈 수준에 비례하여 증가하였으나, DE 실험의 절반과 DER 실험의 거의 모든 경우에서 이 예측치는 실제 불확실성과의 보정이 잘 이루어지지 않았으며, 특히 2D 입력 불확실성 실험 및 고 노이즈 레벨에서 가장 부정확한 것으로 나타났습니다.



### An Empirical Examination of the Evaluative AI Framework (https://arxiv.org/abs/2411.08583)
- **What's New**: 이 연구는 'Evaluative AI' 프레임워크를 실증적으로 분석하여, AI 사용자의 의사결정 과정을 개선하는 데 중점을 두고 있습니다. 기존의 추천 기반 접근 방식에서 가설 기반 접근 방식으로 전환하여, 사용자에게 가설에 대한 찬반 증거를 제공함으로써 정보에 기반한 결정을 내릴 수 있도록 지원합니다.

- **Technical Details**: 'Evaluative AI' 프레임워크는 Miller (2023)에 의해 제안되었으며, 데이터/프레임 이론과 귀추적 요소를 기반으로 합니다. 이 프레임워크는 AI가 직접적인 추천을 제공하는 대신, 의사결정자가 요청할 경우 각 옵션에 대한 찬성과 반대의 증거를 제공합니다. 이 연구는 AI의 제안이 아닌, 사용자가 증거를 검토하는 방식으로 의사결정 과정의 성과, 효율성 및 주관적 인식에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 현재 연구에서는 'Evaluative AI' 프레임워크가 의사결정 성과를 개선하지 못했으며, 제공된 증거에 대해 참여자들이 피상적으로만 참여함을 발견했습니다. 이는 기존의 AI 시스템에서 보이는 인지적 과정과 유사한 결과로, AI의 영향에 의한 인지적 분산이 우세했음을 시사합니다.



### Intelligent Algorithms For Signature Diagnostics Of Three-Phase Motors (https://arxiv.org/abs/2411.08582)
- **What's New**: 이 연구는 세 가지 주요 혁신을 통해 전통적인 정밀 진단 방법의 한계를 극복하고 있습니다. 첫째, 전이성 결함 탐지를 위한 새로운 비지도 학습 기법을 도입하여 물리 모델을 기반으로 한 이상 징후 생성을 통합했습니다. 둘째, 생성적 신경망 및 ResNet 기반의 합성곱 신경망을 활용하여 물리적으로 정확한 합성 이상을 생성하고, 이를 통해 감독 기계 학습 모델의 학습 능력을 향상시켰습니다. 셋째, 기존의 비지도 모델인 AnomalyBERT와 VAE-LSTM을 개선하여 산업 현장에서의 실용성을 높였습니다.

- **Technical Details**: 연구에서는 비지도 기계 학습과 감독 기계 학습의 장점을 결합한 하이브리드 접근 방식을 채택했습니다. 이는 전통적인 시그니처 분석 방법을 기반으로 하여, 생성적 신경망을 통해 현실적인 합성 결함 데이터를 생성하고 이를 사용하여 머신 러닝 모델을 훈련하는 방법입니다. 데이터 부족 문제를 해결하기 위해 물리적으로 정확한 합성 이상을 생성하는 기술이 포함되었습니다.

- **Performance Highlights**: 실험 결과, 본 연구의 접근 방식이 기존의 기계 학습 및 비기계 학습 기법 모두를 뛰어넘는 우수한 성능을 보였음을 확인했습니다. 특히, 제안한 방법은 데이터 레이블이 많이 필요하지 않고도 고도의 진단 정확성을 달성하며, 실제 산업 응용에 적합한 강력하고 효율적인 솔루션을 제시합니다.



### Neural Corrective Machine Unranking (https://arxiv.org/abs/2411.08562)
Comments:
          submitted to Information Sciences

- **What's New**: 본 연구에서는 신경 정보 검색(neural information retrieval, IR) 시스템에서 데이터 제거(machine unlearning)의 필요성을 다룹니다. 기존의 기법들이 검색 효과성을 저하시킬 수 있는 문제를 해결하기 위해 corrective unranking을 정식화하였으며, 새로운 teacher-student 프레임워크인 Corrective unRanking Distillation (CuRD)을 제안합니다.

- **Technical Details**: CuRD는 (1) 잊혀져야 하는 샘플의 출력 관련 점수를 순위가 낮은 비검색 샘플의 점수와 유사하게 조정하여 잊기(positional forgetting)를 촉진합니다; (2) 대체 샘플에 대한 관련 점수를 수정하여 이에 해당하는 잊혀져야 하는 샘플의 점수와 밀접하게 일치하도록 세분화합니다; (3) 잊기를 목표로 하지 않는 샘플들의 성능을 유지합니다.

- **Performance Highlights**: CuRD는 BERTcat, BERTdot, ColBERT, PARADE의 네 가지 신경 IR 모델을 사용하여 MS MARCO와 TREC CAR 데이터셋에서 평가되었습니다. 실험 결과, CuRD는 잊기와 수정 면에서 7개의 최첨단 기법을 웃도는 성과를 보였으며, 모델의 유지 및 일반화 능력을 유지했습니다.



### LogLLM: Log-based Anomaly Detection Using Large Language Models (https://arxiv.org/abs/2411.08561)
- **What's New**: LogLLM은 대형 언어 모델(LLM)을 활용하여 로그 기반 이상 탐지 프레임워크를 제안합니다. 기존의 로그 파서 의존 방식을 개선하여 정규 표현식을 사용해 로그 메시지를 사전 처리하여 탐지 과정을 간소화합니다.

- **Technical Details**: LogLLM은 BERT를 사용하여 로그 메시지로부터 의미 벡터(semantic vectors)를 추출하고, Llama라는 트랜스포머 디코더 모델로 로그 시퀀스를 분류합니다. 또한, BERT와 Llama의 벡터 표현 공간을 정렬하는 프로젝터(projector)를 도입하여 로그의 의미를 일관되게 이해합니다.

- **Performance Highlights**: 4개의 공공 데이터셋에 대한 실험 결과, LogLLM은 최신 기술(state-of-the-art) 방법들을 초월하는 성능을 보였습니다. 불안정한 로그를 처리할 때에도 로그 메시지의 의미를 효과적으로 포착하고 이상을 정확히 탐지합니다.



### Leveraging Pre-Trained Neural Networks to Enhance Machine Learning with Variational Quantum Circuits (https://arxiv.org/abs/2411.08552)
Comments:
          In submission

- **What's New**: 새로운 연구에서는 사전 훈련된 신경망(pre-trained neural networks)을 활용하여 변분 양자 회로(Variational Quantum Circuits, VQC)의 성능을 개선하는 혁신적인 접근 방식을 소개합니다. 이 방법은 근사 오차(approximation error)를 큐비트 수(qubit count)와 분리하여 QML의 실제 적용 가능성을 높입니다.

- **Technical Details**: 이 연구는 사전 훈련된 신경망을 통해 VQC의 매개변수 최적화(parameter optimization)를 크게 향상시키고, 심층적인 이론적 분석과 양자 점(quantum dot) 분류(task)에서의 경험적 테스트를 통해 이를 입증했습니다. 특히, 높은 양자 잡음(qeantum noise) 환경에서만 수백 개의 물리적 큐비트를 사용할 수 있는 현재의 제약을 극복하게 됩니다.

- **Performance Highlights**: 이 방법을 통해 VQC 모형은 보다 적은 훈련 데이터(training data)로도 강력한 표현(representation) 및 일반화(generalization) 능력을 달성할 수 있으며, 인간 유전자 분석(human genome analysis)과 같은 다양한 응용 분야에서도 효과를 발휘합니다. 사전 훈련된 신경망의 사용은 VQC의 실제 세계 응용 가능성을 대폭 증가시키며, 기계 학습, 재료 과학, 의학 등의 분야에서 양자 컴퓨팅의 잠재력을 전체적으로 열어줍니다.



### MLV$^2$-Net: Rater-Based Majority-Label Voting for Consistent Meningeal Lymphatic Vessel Segmentation (https://arxiv.org/abs/2411.08537)
Comments:
          ML4H 2024

- **What's New**: 이 논문에서는 meningeal lymphatic vessels (MLVs)의 세그멘테이션을 위한 새로운 rater-aware training 스킴과 ensembling 전략을 제안하고, nnU-Net 모델의 성능을 향상시키는 방법을 탐구합니다. MLV는 노화 및 뇌 질환과 관련이 있으며, 이번 연구에서는 새로운 자동 세그멘테이션 방법인 MLV$^2$-Net을 소개합니다.

- **Technical Details**: MLV$^2$-Net은 nnU-Net 기반으로, 3D FLAIR MRI를 입력으로 받아들이며, rater 인코딩을 추가적으로 적용합니다. 또한, 다양한 rater들이 제공한 세그멘테이션 스타일을 학습하고, 예측을 집계하기 위해 가중 다수 결정을 사용하는 방식으로 이루어집니다. 이 과정에서 모델의 예측 볼륨에 대한 오류 경계를 파생하는 기술적 기여도 포함됩니다.

- **Performance Highlights**: MLV$^2$-Net은 인간의 참조 표준에 대해 0.806의 Dice similarity coefficient를 달성하여 MLV의 세그멘테이션 결과가 인간 간 신뢰성과 일치합니다. 또한, MLV의 부피와 관련된 노화 관련 연관성을 잘 재현할 수 있습니다.



### ACROSS: A Deformation-Based Cross-Modal Representation for Robotic Tactile Perception (https://arxiv.org/abs/2411.08533)
Comments:
          Paper Submitted to ICRA2025. arXiv admin note: text overlap with arXiv:2410.14310

- **What's New**: 로봇 기술에서 촉각( tactile ) 센서와 데이터 호환성을 높이는 ACROSS 프레임워크를 소개합니다. 이는 기존 데이터셋을 새로운 센서에 맞춰 가치 있게 변환하는 방법을 제공합니다.

- **Technical Details**: ACROSS는 기존의 촉각 센서 신호를 3D 변형 메쉬( deformation meshes )로 변환하고, 다른 센서의 메쉬로 전환한 후 최종적으로 해당 출력 공간으로 변환하는 과정을 포함합니다. 이 과정에서 BioTac의 낮은 차원의 촉각 신호를 DIGIT의 고차원 촉각 이미지로 변환합니다. 또한, 155K개의 고유한 3D 메쉬 변형 쌍( mesh deformation pairs )을 포함하는 공개 데이터셋을 제공합니다.

- **Performance Highlights**: ACROSS는 데이터 전환의 효율성을 높이고 기존 데이터셋의 재사용을 촉진하여 비효율적인 데이터 수집 과정을 피합니다. 이를 통해 다양한 센서와 설정의 연구자들 간 데이터 교환이 가능해집니다.



### Gendered Words and Grant Rates: A Textual Analysis of Disparate Outcomes in the Patent System (https://arxiv.org/abs/2411.08526)
- **What's New**: 이 연구는 특허법에서 성별 불균형을 연구하며, 특허 신청서의 텍스트 내용을 분석합니다. 기존 연구는 주로 메타데이터에 초점을 맞췄으나, 본 연구는 머신러닝(machine learning)과 자연어 처리(natural language processing) 기술을 활용하여 текст (text)에서 숨겨진 정보를 추출합니다.

- **Technical Details**: 연구에서는 텍스트의 특성을 기반으로 발명자의 성별을 예측하는 데 머신러닝과 자연어 처리 기법을 사용합니다. 발명자의 이름을 몰라도 성별을 상당한 정확도로 식별할 수 있음을 발견하였습니다. 이 연구는 특허 심사에서 익명화(anonymized) 방식이 성별에 따른 결과를 완전히 해결하지 못할 것임을 시사합니다.

- **Performance Highlights**: 특허 문서 내에서 성별에 따른 텍스트 선택의 차이 및 발명자가 선택하는 분야에서의 성별적 차이를 식별하였습니다. 이러한 발견은 텍스트 선택, 성별, 특허 확보 성공 간의 복잡한 상호작용을 강조하며, 성별 평등과 효율성을 달성하려는 현재의 제안들이 과연 효과적인지에 대한 중요한 질문을 제기합니다.



### SAD-TIME: a Spatiotemporal-fused network for depression detection with Automated multi-scale Depth-wise and TIME-interval-related common feature extractor (https://arxiv.org/abs/2411.08521)
Comments:
          21pages, 7 figures

- **What's New**: 이번 연구에서는 우울증 진단을 위한 새로운 자동화된 딥 러닝 기반 방법인 SAD-TIME을 제안합니다. SAD-TIME은 EEG 신호의 고유한 시공간(spatiotemporal) 정보를 융합하여 우울증을 분류하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SAD-TIME은 자동화된 노드 공통 특징 추출기(CFE), 공간 부문(SpS), 수정된 시간 부문(TeS), 도메인 적대적 학습기(DAL)를 포함합니다. CFE는 다중 규모 깊이별 1D-컨볼루션 신경망 및 시간-간격 임베딩 생성을 이용하여 각 채널의 고유한 정보를 유지합니다. SpS는 기능적 연결성을 EEG 전극의 공간 위치와 결합하여 융합하며, 다중 헤드 주의(graph attention) 네트워크를 이용하여 특성을 융합합니다. TeS는 LSTM(long short-term memory)과 그래프 변환기 네트워크를 기반으로 하여 서로 다른 시간 창의 정보를 융합합니다.

- **Performance Highlights**: 제안된 SAD-TIME 방법은 10겹 교차 검증에서 두 개의 데이터셋에 대해 각각 92.00% 및 94.00%의 우울증 분류 정확도를 달성하였습니다. 이는 EEG 신호의 고유한 시공간 정보 융합을 통한 강력한 우울증 감지 모델을 나타냅니다.



### An Information Theoretic Approach to Operationalize Right to Data Protection (https://arxiv.org/abs/2411.08506)
Comments:
          First two authors contributed equally to this work

- **What's New**: 본 연구는 데이터 보호 법규(GDPR) 준수의 문제를 해결하기 위해 자연어 데이터셋에 무시할 수 있는 잡음을 주입하는 RegText라는 새로운 프레임워크를 소개합니다.

- **Technical Details**: RegText는 자연어 데이터셋에 무시할 수 있는 가짜 상관관계를 주입하여 언어 모델이 이를 학습할 수 없도록 만듭니다. 이는 기존의 이미지 기반 접근 방식과는 달리 의미론적 내용에는 영향을 미치지 않습니다.

- **Performance Highlights**: RegText를 사용했을 때 GPT-4o 및 Llama와 같은 최신 모델이 생성된 데이터에서 학습하지 못하게 하여 테스트 정확도가 하락했습니다. 이로 인해 제로-샷(Zero-shot) 성능보다 낮은 결과를 나타냈습니다.



### Towards Objective and Unbiased Decision Assessments with LLM-Enhanced Hierarchical Attention Networks (https://arxiv.org/abs/2411.08504)
- **What's New**: 이 연구는 대학 입시 과정에서 인간 전문가의 결정에서 인지 편향(cognitive bias)을 식별하고, 이를 극복하기 위한 AI 증강 워크플로우(BGM-HAN)를 제안합니다. 기존의 결정 과정에서 드러나는 불일치와 편향을 분석하여 AI의 도움으로 보다 공정하고 일관된 결정을 원하는 방향으로 나아가고자 했습니다.

- **Technical Details**: 제안된 모델인 BGM-HAN은 'Byte-Pair Encoding'과 'Multi-Head Attention', 'Gated Residual Connection'을 이용하여 다층적인 세미 구조화된 데이터를 효과적으로 표현합니다. 또한 이 모델은 'Shortlist-Analyze-Recommend' (SAR) 에이전틱 워크플로우를 사용하여 기존의 인간 결정 과정을 모방하면서 일관성을 향상시키고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델과 워크플로우는 기존 인간 평가를 기준으로 F1-score와 정확도에서 9.6% 이상의 개선을 보였습니다. 이러한 결과는 계층적 학습(hierarchical learning)이 AI의 자동화된 결정-making 과정에서 내재된 공정성과 일관성을 제공하는 잠재력을 보여줍니다.



### Learning Model Agnostic Explanations via Constraint Programming (https://arxiv.org/abs/2411.08478)
- **What's New**: 이번 연구에서는 블랙박스 모델에 대한 모델 비의존적인 설명을 찾기 위해 제약 최적화 문제(Constraint Optimization Problem)로 접근하여 최소 오류(minimum error)와 바운드된 크기(bounded size)를 갖는 설명을 찾는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 가설 클래스(hypothesis class)를 정의하고, 각 설명을 if-then 규칙으로 나타내며, 이를 통해 PAC-style 보장을 제공합니다. 이러한 접근은 선형 제약조건(linear constraints)과 채널링 제약조건(channeling constraints)을 조합하여 제약 최적화 문제로 모델링됩니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험한 결과, 제안된 접근 방식이 기존의 Anchors 방법보다 통계적으로 우수한 정밀도(precision)를 보여주며, SOLVER에 대해 합리적인 실행 시간을 유지합니다.



### Trap-MID: Trapdoor-based Defense against Model Inversion Attacks (https://arxiv.org/abs/2411.08460)
Comments:
          Accepted by Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이번 논문에서는 Trapdoor 기반 모델 역전 방어(Trap-MID)를 제안하여 모델 역전 공격(Model Inversion, MI 공격)에 대한 새로운 방어 전략을 소개합니다. 이 방어 방법은 입력에 트리거를 주입하였을 때 특정 레이블을 예측하도록 설계된 트랩도어(trapdoor)를 통합하여 MI 공격을 오도하는 방식으로 작동합니다.

- **Technical Details**: Trap-MID는 모델의 동작에 트랩도어를 삽입하여 MI 공격자가 개인 정보를 추출하기보다는 트랩도어 트리거를 추출하도록 유도합니다. 이 연구는 트랩도어의 효과성과 자연스러움이 MI 공격을 오도하는데 미치는 영향을 이론적으로 논의하며, 다양한 MI 공격에 대한 실험에서 Trap-MID가 기존 방법보다 우수한 방어 성능을 발휘함을 보여줍니다.

- **Performance Highlights**: Trap-MID는 추가 데이터나 대규모 계산 오버헤드 없이 다양한 MI 공격에 대해 최신 방어 성능을 나타내며, 실험을 통해 그 효율성과 효과성을 입증했습니다.



### Learning Dynamic Cognitive Map with Autonomous Navigation (https://arxiv.org/abs/2411.08447)
Comments:
          under submission at Frontiers Computer Neuroscience

- **What's New**: 동물의 내비게이션 전략에서 영감을 받아, 본 논문에서는 생물학적 원칙에 뿌리를 둔 공간 내비게이션 및 맵핑을 위한 새로운 계산 모델을 소개합니다. 이 모델은 동적으로 확장되는 인지 맵(cognitive map)을 포함하여 새로운 환경 변화를 빠르게 학습할 수 있는 오류 수정 기능을 갖추고 있습니다.

- **Technical Details**: 모델은 Active Inference(AIF) 프레임워크를 사용하여 생성 모델의 플라스틱성을 높이고, 예측된 자세를 기반으로 인지 맵을 동적으로 확장합니다. Bayesian inference를 활용하여 미방문 지점에 대한 선험적 신념을 형성하며, 새로운 관측이 이루어질 때 내부 모델을 업데이트하여 복잡한 환경에서 효과적으로 내비게이션을 수행합니다.

- **Performance Highlights**: 모델은 Clone-Structured Cognitive Graph 모델(CSCG)과 비교하여 단일 에피소드 내에서 환경 구조를 신속하게 학습하며, 이전의 신념과 상반되는 새로운 증거가 주어졌을 때 이를 반영하여 맵을 업데이트하는 능력을 보여줍니다. 모든 목표 지향적인 내비게이션 및 탐험 task에서 강력한 적응력을 갖추고 있으며, 실제 내비게이션 작동을 위한 생물학적 기전을 통합하여 실용적으로 적용될 수 있음을 강조합니다.



### 3D Multi-Object Tracking with Semi-Supervised GRU-Kalman Filter (https://arxiv.org/abs/2411.08433)
- **What's New**: 본 연구에서는 GRU 기반의 Kalman 필터를 도입한 새로운 3D Multi-Object Tracking(MOT) 방법을 제안합니다. 이를 통해 전통적인 방법에서의 노이즈 불일치와 운동 과정 선형화로 인한 정확도 손실을 피할 수 있습니다. 또한, 반감독 학습 기법을 설계하여 사용 가능한 학습 데이터의 양과 레이블의 강건성을 크게 확장합니다.

- **Technical Details**: 본 시스템은 세 가지 주요 모듈로 구성됩니다: 전처리 모듈, 운동 모듈(GRU 기반 Kalman 필터 사용), 연관 모듈 및 궤적 관리 모듈입니다. GRU는 Recursive Bayesian Filtering의 루프를 시뮬레이션하도록 설계되었으며, 시스템의 노이즈 분포, 상태 전이 행렬 및 관측 행렬을 자동으로 학습합니다. 이 과정은 모델 기반 상태 추정법의 간소화를 가능하게 합니다.

- **Performance Highlights**: nuScenes 및 Argoverse2 데이터셋을 통한 평가 실험에서, 본 시스템은 전통적인 TBD 방법에 비해 우수한 성능을 보이며, 각 객체 카테고리에 대한 수동 모델 설계의 필요성을 제거하였습니다.



### One STEP at a time: Language Agents are Stepwise Planners (https://arxiv.org/abs/2411.08432)
- **What's New**: STEP라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 언어 에이전트의 계획 능력을 향상시키기 위해 과거 경험에서 학습하는 것을 목표로 합니다.

- **Technical Details**: STEP는 Planner, Executor, Evaluator, Memory의 4가지 상호 연결된 구성 요소로 작동합니다. Planner는 작업을 하위 작업으로 분해하고 관련 정보를 Memory에서 검색합니다. Executor는 동작 후보를 생성하고, Evaluator는 이 동작들이 이전 경험에서 학습한 규칙들과 일치하는지 확인합니다. Memory는 경험을 저장하여 미래의 결정을 알리는 역할을 합니다.

- **Performance Highlights**: ScienceWorld 벤치마크에서 STEP는 67.4의 전체 점수를 기록하며, 18개 작업 중 12개를 성공적으로 완료했다는 결과를 보여주었습니다. 이는 STEP의 언어 에이전트의 계획 능력을 향상시키기 위한 프레임워크로서의 가능성을 부각시킵니다.



### A Heterogeneous Graph Neural Network Fusing Functional and Structural Connectivity for MCI Diagnosis (https://arxiv.org/abs/2411.08424)
- **What's New**: 이 연구는 기능적 연결성과 구조적 연결성을 통합하는 이질적 그래프 신경망(HGNN)을 제안하여 다양한 두 가지 모드 이미지를 활용할 수 있는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 우리의 접근 방식은 혈액 산소 수준 의존성(rs-fMRI)과 백색질 구조(DTI) 정보를 사용하여 각각 동질 메타 경로(homo-meta-path) 및 이질 메타 경로(hetero-meta-path)를 구성합니다. 이 방법은 동질적 정보와 이질적 정보를 효과적으로 혼합하고, 이질적 그래프 풀링 전략을 통해 다양한 기능 간의 혼란을 방지합니다.

- **Performance Highlights**: ADNI-3 데이터셋에서 경미한 인지 장애(MCI) 진단을 위해 평가된 결과, 우리의 방법은 평균 분류 정확도 93.3%를 달성하여 다른 알고리즘보다 우수한 성능을 보였습니다.



### Material Property Prediction with Element Attribute Knowledge Graphs and Multimodal Representation Learning (https://arxiv.org/abs/2411.08414)
- **What's New**: 본 연구에서는 기존 방법의 한계를 극복하기 위해 원소 속성 지식 그래프를 구축하고, 이를 ESNet이라는 다중 모드 융합 프레임워크에 통합하여 결정성 물질의 성능 예측을 수행합니다.

- **Technical Details**: ESNet는 원소 속성과 결정 구조 기능을 결합하여 조인트 다중 모드 표현을 생성합니다. 원소 속성 지식 그래프에 기반한 임베딩 모델을 사용하여 원소의 화학적 및 물리적 특성을 인코딩하고, ComFormer 접근법을 참조하여 결정 구조 기능을 추출합니다.

- **Performance Highlights**: Materials Project 벤치마크 데이터셋에서 실시된 실험 결과, 밴드갭 예측 작업에서 선도적인 성능을 보였으며, 형성 에너지 예측 작업에서도 기존 벤치마크와 동등한 결과를 달성했습니다.



### BAMAX: Backtrack Assisted Multi-Agent Exploration using Reinforcement Learning (https://arxiv.org/abs/2411.08400)
- **What's New**: 본 논문은 백트랙을 지원하는 다중 에이전트 탐사를 위한 방법인 BAMAX를 소개합니다. BAMAX는 다수의 로봇이 협력하여 미지의 환경을 탐험할 수 있도록 돕는 새로운 접근법으로, 전통적인 방법에 비해 더 빠른 환경 탐사와 적은 백트랙을 달성할 수 있습니다.

- **Technical Details**: BAMAX는 DQN(Deep Q-Network) 알고리즘을 활용하여 최적의 행동-가치 함수(Q-function)를 근사화하고, 각 로봇은 환경과의 상호작용을 통해 최적의 행동을 학습합니다. 탐사 시 발생하는 장애물 문제를 해결하기 위해 로봇은 이전에 알려진 열린 위치로 돌아갈 수 있는 능력을 부여받습니다. 이 방법은 10x10에서 60x60 크기의 육각형 그리드에서 실험되어 성능을 평가했습니다.

- **Performance Highlights**: BAMAX는 전통적인 탐사 방법에 비해 처리 속도가 빨랐으며, 더 적은 백트랙을 통해 더 넓은 지역을 빠르게 커버했습니다. 또한, 다양한 크기의 육각형 그리드에서도 효과적으로 스케일링할 수 있는 가능성을 보였습니다.



### Physics Informed Distillation for Diffusion Models (https://arxiv.org/abs/2411.08378)
- **What's New**: 본 논문에서는 Physics Informed Distillation (PID)라는 새로운 지식 증류 기술을 소개합니다. 이 기술은 학생 모델이 교사 모델인 확산 모델(teacher diffusion model)의 솔루션을 표현하도록 설계되어, Physics Informed Neural Networks (PINNs)의 원리를 활용합니다.

- **Technical Details**: PID는 확산 모델을 ODE 시스템으로 관찰하여, 학생 모델이 ODE의 경로를 예측하도록 훈련합니다. 이를 통해 단일 단계의 이미지 생성을 가능하게 하며, CIFAR 10과 ImageNet 64x64 데이터셋을 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: PID는 최신 증류 방법들과 동등한 성능을 보이며, 메서드 고유의 하이퍼파라미터에 대한 조정 없이도 안정적인 경향성을 유지합니다. 따라서 PID는 확산 모델에 대한 사용의 용이성을 높이는데 기여합니다.



### Developing an Effective Training Dataset to Enhance the Performance of AI-based Speaker Separation Systems (https://arxiv.org/abs/2411.08375)
Comments:
          in Arabic language

- **What's New**: 이 논문은 현실적인 녹음 조건에서의 스피커 분리 문제에 대한 새로운 접근을 제안합니다. 특히, 우리의 방법론은 혼합 신호와 각 스피커에 대한 해당 진실값을 포함한 현실적인 훈련 세트를 구축하는 것입니다.

- **Technical Details**: 스피커 분리의 도전 과제를 다루며, 기존의 신경 네트워크 모델이 종종 합성 데이터셋으로 훈련되어 현실 세계의 복잡성을 충분히 반영하지 못한다는 점을 강조합니다. 우리의 새로운 방법론은 깊은 학습 모델(deep learning model)을 사용하여 현실적인 혼합 신호(realistic mixing)에서의 성능 향상을 평가합니다.

- **Performance Highlights**: 실제 혼합 신호에서 스피커 분리 정확도가 1.65 dB 향상되었습니다. 이는 현실적인 훈련 세트가 실제 시나리오에서 스피커 분리 모델의 성능을 향상시킬 수 있는 잠재력을 시사합니다.



### Surprisingly Popular Voting for Concentric Rank-Order Models (https://arxiv.org/abs/2411.08367)
- **What's New**: 이 논문에서는 소셜 정보 공유에서 'Surprisingly Popular (SP)' 알고리즘이 전문가가 소수일 때에도 사실 정보를 회복할 수 있는 방법을 제안합니다. SP 알고리즘은 개별 의견 외에도 다수의 답변에 대한 예측을 요구하여 실제 빈도수가 평균 예측 빈도보다 큰 것을 선택하는 방식으로 작동합니다.

- **Technical Details**: 저자들은 Mallows 및 Plackett-Luce 모델의 동심원 혼합(concentric mixtures)을 제안하여 G(≥2) 그룹을 고려한 새로운 순위 모델을 확장하였습니다. 이 모델들은 SP-voting의 샘플 복잡성을 분석하기 위한 기반으로 사용됩니다.

- **Performance Highlights**: 실험 결과, SP-voting은 Copeland 투표 규칙과 비교하여 적은 데이터셋에서도 더 나은 성능을 보여주며, G=3인 경우 전문가 그룹 외에 중간 수준 전문가 그룹을 도입하여 데이터셋에 대한 설명력이 향상되었습니다.



### A Chinese Multi-label Affective Computing Dataset Based on Social Media Network Users (https://arxiv.org/abs/2411.08347)
- **What's New**: 이 연구는 중국어 감정 데이터세트를 수집하고, 개인의 성격 특성과 감정을 통합한 다중 라벨 감정 데이터세트를 구축했습니다. 기존 데이터세트는 감정과 성격 특성을 별도로 주석을 달아 왔으며, 미세 감정(micro-emotions)과 감정 강도의 세밀한 라벨링이 부족했습니다.

- **Technical Details**: 연구진은 중소형 SNS 플랫폼인 위보(Weibo)에서 50,000명 이상의 개인 중 11,338명의 유효 사용자를 선별하여 이들의 MBTI 성격 라벨과 함께 566,900개의 게시글을 수집했습니다. EQN 방법론을 사용하여 같은 사용자의 MBTI 성격 특성을 6가지 감정(emotions)과 미세 감정(micro-emotions)과 통합하여 주석이 달린 감정 데이터세트를 작성했습니다.

- **Performance Highlights**: 다양한 NLP(classification) 모델을 통한 검증 결과, 이 데이터세트는 복잡한 인간 감정의 기계 인식을 진전시킬 수 있는 강력한 유용성을 입증했습니다. 이 데이터세트는 심리학, 교육, 마케팅, 금융 및 정치 연구에 자료 지원을 제공하는 데 목적이 있습니다.



### Generative AI for Data Augmentation in Wireless Networks: Analysis, Applications, and Case Study (https://arxiv.org/abs/2411.08341)
- **What's New**: 이번 연구는 Generative Artificial Intelligence (GenAI)를 활용한 무선 데이터 증강의 잠재력과 효과를 체계적으로 탐구합니다. GenAI는 기존의 전통적인 데이터 증강 기술로는 해결할 수 없는 무선 데이터 특성을 고려하여 고품질의 다양한 데이터 생성을 통해 모델 성능을 향상시킬 수 있는 혁신적 방법입니다.

- **Technical Details**: 이 논문에서는 GenAI 모델과 이의 데이터 증강에 대한 응용을 검토하고, 물리적, 네트워크, 애플리케이션 계층에서의 GenAI 기반 데이터 증강 아키텍처를 제안합니다. 특히, Wi-Fi 제스처 인식을 위한 일반적인 Generative Diffusion Model (GDM) 기반 데이터 증강 프레임워크를 개발하고, 잔차 신경망 모델을 사용하여 증강된 데이터의 효과를 평가합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 프레임워크가 Wi-Fi 제스처 인식에서 효과적임을 입증하였으며, 데이터 증강 기법이 DL 모델의 성능을 어떻게 개선하는지를 중심으로 논의되었습니다.



### DEEGITS: Deep Learning based Framework for Measuring Heterogenous Traffic State in Challenging Traffic Scenarios (https://arxiv.org/abs/2411.08335)
Comments:
          Submitted for presentation at the 103 rd Annual Meeting of Transportation Research Board and publication in Transportation Research Record: Journal of Transportation Research Board

- **What's New**: 이번 논문에서는 DEEGITS(Deep Learning Based Heterogeneous Traffic State Measurement)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 최신 convolutional neural network (CNN) 기술을 활용하여 차량과 보행자를 정확하고 신속하게 감지하고, 혼잡 및 가려짐과 같은 어려운 상황에서도 교통 상태를 측정합니다.

- **Technical Details**: DEEGITS는 데이터 융합(data fusion)을 통해 차량과 보행자를 동시에 감지할 수 있는 훈련 데이터셋을 향상시키며, 이미지 전처리(image preprocessing) 및 증강(augmentation)을 통해 데이터셋의 품질과 양을 개선합니다. YOLOv8로 사전 훈련된 모델에 전이 학습(transfer learning)을 적용하여 다양한 차량을 식별하는 모델의 능력을 증가시키고, Grid Search 알고리즘을 통해 최적의 하이퍼파라미터를 구합니다. 여기서 Stochastic Gradient Descent (SGD) 옵티마이저가 다른 옵티마이저보다 우수한 성능을 보입니다.

- **Performance Highlights**: 광범위한 실험과 평가 결과, 검증 세트에서 0.794 mAP@0.5, 테스트 세트에서 0.786 mAP@0.5의 높은 정확도를 달성하며, 유사한 데이터셋에서 이전 벤치마크를 초월한 성과를 보여 줍니다. 또한, DeepSORT 다중 객체 추적(multi-object tracking) 알고리즘을 통합하여 감지된 차량과 보행자를 추적합니다. 다양한 교통 조성 및 혼잡 수준을 가진 두 위치에서 실험을 진행하였으며, 두 경우 모두 통계적으로 유의미하지 않은 오류를 보였고, 이질적 교통 흐름과 속도 측정에서 각각 0.99~0.88 및 0.91~0.97의 상관관계를 나타냈습니다.



### Enhancing Multimodal Query Representation via Visual Dialogues for End-to-End Knowledge Retrieva (https://arxiv.org/abs/2411.08334)
- **What's New**: 본 논문에서는 기존의 분리된 모델에 의존하는 다중 모달(multi-modal) 검색 시스템의 한계를 극복하기 위해, Ret-XKnow라는 엔드 투 엔드(end-to-end) 검색 시스템을 제안합니다. 이 시스템은 텍스트 검색기에 다중 모달 쿼리를 이해하는 능력을 부여하며, 비주얼 정보와 텍스트 쿼리 간의 동적 상호작용을 통해 성능을 향상시킵니다.

- **Technical Details**: Ret-XKnow는 부분 합성곱(partial convolution) 메커니즘을 사용하여 주어진 텍스트 쿼리와 관련된 비주얼 정보에 집중합니다. 이를 통해 비주얼 임베딩(visual embeddings)을 압축하고, 텍스트 쿼리 표현과의 관련성 점수를 적응형 마스크로 활용하여 시각적 정보의 중요성을 강조합니다. 또한, Visual Dialogue-to-Retrieval (ViD2R) 데이터셋을 도입하여, 시각적 대화 데이터셋으로부터 자동으로 생성된 데이터셋을 통해 다중 모달 상호작용을 효과적으로 학습합니다.

- **Performance Highlights**: Ret-XKnow는 사전 훈련(pre-training) 데이터셋으로 ViD2R를 사용하여 네 가지 다중 모달 데이터셋에서 제로샷(zero-shot) 검색 성능의 유의미한 향상을 보여주며, 추가적인 모델 조정 없이도 높은 차세대 검색 성능을 입증합니다. 이는 Ret-XKnow 모델이 기존의 다양한 기준선(baseline) 방법들과 비교하여 우수한 결과를 도출했음을 나타냅니다.



### Are LLMs Prescient? A Continuous Evaluation using Daily News as the Orac (https://arxiv.org/abs/2411.08324)
- **What's New**: 새로운 평가 벤치마크인 Daily Oracle을 제안합니다. 이 벤치마크는 매일 뉴스에서 생성된 질문-답변(QA) 쌍을 사용하여 LLM의 시간적 일반화(temporal generalization) 및 예측 능력을 평가합니다.

- **Technical Details**: Daily Oracle은 True/False(TF) 및 Multiple Choice(MC) 형태의 질문으로 구성되어 있으며, 다양한 카테고리(예: 비즈니스, 정치, 예술)에서 LLM이 미래 사건을 예측하도록 도전합니다. 연구 결과, LLM은 2020년 1월부터 2024년 9월까지 TF 질문에서 평균 20.14%, MC 질문에서 23.26%의 성능 저하를 경험했습니다.

- **Performance Highlights**: 이 연구는 LLM의 예측 정확도가 시간이 지남에 따라 지속적으로 감소한다는 것을 증명합니다. RAG 방식을 활용한 모델이 예측 성능이 향상될 수 있지만, 여전히 성능 감소 패턴은 지속됩니다. 이는 항상 업데이트되어야 하는 모델의 필요성을 강조합니다.



### R3HF: Reward Redistribution for Enhancing Reinforcement Learning from Human Feedback (https://arxiv.org/abs/2411.08302)
- **What's New**: 본 논문은 Reinforcement Learning from Human Feedback (RLHF)에서 발생하는 기존의 보상 구조를 개선하여 각 토큰에 대해 더 세밀한 보상을 할당하는 R3HF 방법을 제안합니다.

- **Technical Details**: R3HF 방법은 보상 모델을 회귀(regression) 문제로 간주하여 각 토큰의 기여도를 분석한 후, 정확한 보상을 분배합니다. 이 과정은 Sequence-Markov Decision Process (SDP) 프레임워크를 통해 구현되며, 각 토큰에 대해서도 그 이전 시간 단계와의 비교를 통해 보상 크기를 유도하는 방법입니다.

- **Performance Highlights**: 다양한 데이터셋과 작업을 통해 실시한 실험 결과, R3HF는 기존 RLHF 기술에 비해 학습 효율성과 성능에서 개선된 결과를 보여 주었습니다. 제안된 방법이 기존의 RLHF 방법에 원활하게 통합될 수 있음을 입증하였습니다.



### TowerDebias: A Novel Debiasing Method based on the Tower Property (https://arxiv.org/abs/2411.08297)
Comments:
          To be submitted to a journal soon

- **What's New**: 이번 논문에서는 black-box 모델의 예측에서 민감한 속성의 영향을 줄이기 위한 새로운 접근법인 towerDebias (tDB)를 제안합니다. 이 방법은 공정성과 유틸리티 간의 균형을 고려하여 후처리 단계에서 예측 공정성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: tDB는 확률 이론에서의 타워 속성을 활용하여 민감한 변수가 포함된 예측을 수정합니다. 이 방법은 원래 모델의 내부 구조에 대한 사전 지식이 필요하지 않으며 다양한 애플리케이션으로 확장 가능합니다. 공정성이 어떻게 향상되는지를 엄밀한 개선 정리를 통해 증명합니다.

- **Performance Highlights**: tDB는 회귀 및 분류 작업 모두에서 효과성을 입증하였으며, 이를 통해 공정성-유틸리티(tradeoff) 간의 관계를 탐구합니다. 이 방법을 사용하여 민감한 속성이 제거된 예측을 통해 더 높은 공정성을 달성할 수 있음을 보여줍니다.



### Hashing for Protein Structure Similarity Search (https://arxiv.org/abs/2411.08286)
- **What's New**: 이번 논문에서는 PSSS (Protein Structure Similarity Search)를 위한 새로운 방법인 POSH (Protein Structure Hashing)를 제안합니다. 기존의 정렬 기반 방법들의 시간 및 메모리 비용을 획기적으로 줄이고, 더 높은 정확도를 달성할 수 있는 방법입니다.

- **Technical Details**: POSH는 각 단백질 구조에 대해 이진 벡터 (binary vector) 표현을 학습하여 PSSS의 시간 및 메모리 비용을 크게 줄입니다. 구조 인코더 (structure encoder)와 표현력이 우수한 핸드 크래프트 기능 (hand-crafted features)을 사용하여 단백질 내의 노드와 엣지 상호작용을 효과적으로 모델링합니다. POSH는 그래프 구성을 통해 각 단백질을 그래프 형태로 나타내며, KNN (k-nearest neighbors)을 통해 엣지를 구성합니다.

- **Performance Highlights**: POSH는 실제 데이터셋에서 실험한 결과, 다른 방법들에 비해 6배 이상의 메모리 절약과 4배 이상의 속도 향상을 달성하며 최첨단의 정확도를 기록했습니다.



### Knowledge Bases in Support of Large Language Models for Processing Web News (https://arxiv.org/abs/2411.08278)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)을 활용하여 웹 뉴스 처리를 위한 지식 기반 구축을 지원하는 일반적인 프레임워크를 소개합니다. 이 프레임워크는 뉴스 정보 추출기(NewsIE)를 사용하여 뉴스 항목에서 관계형 튜플을 추출하고 이를 그래프 합성곱 네트워크(GCN)를 통해 LLM의 암묵적 지식 사실과 결합하여 분류하는 방식입니다.

- **Technical Details**: 프레임워크는 두 가지 가벼운 구성 요소로 이루어져 있습니다: 1) NewsIE: 뉴스 항목의 구조적 정보(관계형 튜플)를 추출합니다; 2) BERTGraph: NewsIE에서 추출한 관계형 튜플과 LLM에서 얻은 암묵적 지식 사실을 그래프 합성곱하여 처리합니다. BERT(양방향 인코더 표현)를 사용하여 GCN에 입력될 수 있도록 텍스트-그래프 어댑터를 설계하였습니다.

- **Performance Highlights**: BERTGraph는 N24News, SNOPES, politifact의 세 가지 공개 웹 뉴스 데이터셋에서 뉴스 카테고리 분류 성능을 평가했으며, 모든 성능 메트릭에서 기존 BERT보다 뛰어난 결과를 보였습니다. 특히 politifact에서 정확도(precision) 이외의 모든 측정 항목에서 우수한 성과를 기록했습니다.



### GPTree: Towards Explainable Decision-Making via LLM-powered Decision Trees (https://arxiv.org/abs/2411.08257)
- **What's New**: GPTree는 의사결정 트리의 설명 가능성(explainability)과 대형 언어 모델(LLMs)의 고급 추론 능력을 결합한 새로운 프레임워크로, 특성 공학(feature engineering)과 프롬프트 체이닝(prompt chaining)의 필요성을 없애고, 작업 특정 프롬프트와 트리 기반 구조를 활용하여 샘플을 동적으로 분할합니다.

- **Technical Details**: GPTree는 최신 기본 모델(OpenAI, 2024)의 추론 및 생성 능력을 의사결정 트리의 설명 가능성과 견고성과 결합하여 지능적이고 적응적인 의사결정을 제공합니다. 이 프로세스는 작업 맥락화(Task Contextualization), 통찰 생성(Insight Generation), 질문 후보 생성(Question Candidate Generation), 결정 분할 최적화(Decision Split Optimization), 전문가 정제(Expert Refinement) 등의 주요 단계를 포함합니다.

- **Performance Highlights**: GPTree는 스타트업 초기 단계에서 '유니콘' 스타트업을 식별하는 데 있어 7.8%의 정밀도(precision rate)를 기록하며, 이는 gpt-4o와 몇 번의 샷 학습(few-shot learning)과 최고의 인간 결정자(3.1%에서 5.6% 사이)의 성과를 초월합니다.



### VALTEST: Automated Validation of Language Model Generated Test Cases (https://arxiv.org/abs/2411.08254)
- **What's New**: 이 논문은 LLM (Large Language Models)에서 생성된 테스트 케이스의 유효성을 자동으로 검증할 수 있는 새로운 프레임워크인 VALTEST를 소개합니다. VALTEST는 토큰 확률(token probabilities)을 활용하여 테스트 케이스의 유효성을 판단합니다.

- **Technical Details**: VALTEST는 HumanEval, MBPP, LeetCode의 세 가지 데이터셋에서 생성된 아홉 개의 테스트 스위트를 사용하여 평가되었습니다. 모델은 LLM에서 생성된 테스트 케이스의 통계적 특징을 추출하고, 이를 기반으로 테스트 케이스의 유효성을 예측하기 위해 머신러닝 모델을 훈련시켰습니다. VALTEST는 테스트 케이스의 유효성을 최대 24%까지 향상시킬 수 있었습니다.

- **Performance Highlights**: VALTEST는 LLM이 생성한 테스트 케이스에서 토큰 확률을 통한 유효성 예측으로 인한 유효성 비율을 6.2%에서 24%까지 증가시키며, 유효한 테스트와 유효하지 않은 테스트를 구분하는 데 효과적인 지표가 됨을 보여주었습니다.



### Retrieval Augmented Time Series Forecasting (https://arxiv.org/abs/2411.08249)
- **What's New**: 이번 연구는 Retrieval-augmented generation (RAG)의 개념을 이용하여 시간 시계열 예측(TSF) 모델인 Retrieval Augmented Forecasting (RAF)을 제안합니다. RAG는 최신 정보를 반영하는 데 중요하며, RAF는 시간 시계열 데이터의 동적 특성을 활용하여 예측 성능을 향상시킵니다.

- **Technical Details**: RAF는 관련된 시간 시계열 예시를 효율적으로 검색하고 이를 예측에 통합하는 전략을 개발합니다. RAF는 모델의 가중치를 조정하지 않고 예측을 수행하는 Naive RAF와 추가로 모델을 조정하는 Advanced RAF 두 가지 변형을 통해 구현됩니다.

- **Performance Highlights**: RAF는 다양한 시간 시계열 도메인에서 예측 정확도를 상당히 향상시켰으며, 특히 새로운 도메인에서 데이터 부족 문제를 해결하는 데 효과적이었습니다. 연구 결과, Advanced RAF는 기존의 모든 기법을 초월하는 성능을 보여주었으며, 이는 더 큰 모델에서 특히 두드러지게 나타났습니다.



### Deceiving Question-Answering Models: A Hybrid Word-Level Adversarial Approach (https://arxiv.org/abs/2411.08248)
- **What's New**: 본 연구는 QA 모델을 대상으로 하는 새로운 적대적 공격 전략인 QA-Attack을 제안합니다. 이 방식은 특정 단어를 표적으로 삼아 QA 모델을 속이는 워드 수준의 적대적 기법입니다.

- **Technical Details**: QA-Attack은 Hybrid Ranking Fusion (HRF) 알고리즘을 사용하여 Attention-based Ranking (ABR) 및 Removal-based Ranking (RBR) 두 가지 방법을 통합합니다. ABR은 질문 처리 중 주의 가중치를 분석하여 중요 단어를 식별하고, RBR은 특정 단어 제거 시 모델의 출력 변화를 관찰하여 단어의 중요성을 평가합니다.

- **Performance Highlights**: QA-Attack은 다양한 질문 유형에 걸쳐 효과적으로 작동하며, 여러 벤치마크 데이터셋에서 기존의 QA 모델을 성공적으로 속였습니다. 연구 결과는 성공률, 의미 변화, BLEU 점수, 유창성 및 문법 오류율 면에서 기존 적대적 기법들을 초월하는 성과를 보여주었습니다.



### A Social Outcomes and Priorities centered (SOP) Framework for AI policy (https://arxiv.org/abs/2411.08241)
- **What's New**: 이 논문은 AI의 급속한 발전이 사회에 미치는 영향을 강조하며, 기술 중심의 접근 방식에서 사회 중심의 정책으로 전환할 필요성을 주장합니다. 이를 통해 AI 정책을 위한 포괄적이고 일관된 프레임워크인 SOP(Social Outcomes and Priorities) 프레임워크를 제안합니다.

- **Technical Details**: SOP 프레임워크는 미국 중심으로 제시되지만, 글로벌 차원에서도 적용할 수 있는 원칙을 제공합니다. AI 시스템의 진화와 그에 따른 위험을 다루기 위해 실질적인 정책과 평가방법론의 필요성을 강조합니다. 최근의 AI 발전이 가져온 위기들을 평가하고, 정책 개발이 어떻게 이루어져야 하는지에 대한 방향성을 제시합니다.

- **Performance Highlights**: 정책의 효율성과 추진력을 높이기 위해 AI 개발과 사회적 요구를 조화롭게 맞출 필요가 있습니다. 현재의 AI 관련 정책들은 구체적인 성과를 제시하지 못하고 있으며, 또 다른 AI 시스템들의 도입과 안전성 부족이 리스크를 증가시키고 있습니다. 따라서 기술 중심의 접근에서 벗어나 사회의 필요와 목표를 우선시하는 정책으로 전환해야 한다고 주장합니다.



### DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection (https://arxiv.org/abs/2411.08227)
- **What's New**: 이 논문은 다중 모달(멀티모달) OOD (Out-of-Distribution) 탐지에서 intra-class variability (클래스 내 변동성)를 고려하지 않는 기존 접근 방식의 한계를 지적합니다. 이를 해결하기 위해 Dynamic Prototype Updating (DPU) 프레임워크를 제안하여 클래스 중심 표현을 동적으로 업데이트하여 견고성과 일반화를 향상시킵니다.

- **Technical Details**: DPU는 유사 샘플의 분산을 측정하여 클래스 별 중심 표현을 동적으로 조정합니다. 이는 instance-level invariant training (인스턴스 수준 불변 훈련)을 활용하여 intra-class cohesion (클래스 내 응집력)과 inter-class separation (클래스 간 분리)을 최적화합니다. 초기 Cohesive-Separate Contrastive Training 절차를 통해 클래스 내부의 변동성을 줄이고, 이후 동적 프로토타입 근사 메커니즘을 통해 prototyping representation을 정교화하여 outlier (외부 작용물)가 클러스터 진화에 미치는 부정적 영향을 완화합니다.

- **Performance Highlights**: DPU는 기존의 9가지 OOD 알고리즘을 기반으로 한 포괄적인 실험에서 Near-OOD 탐지에서는 모든 메트릭에서 약 10% 향상된 성능을, Far-OOD 탐지에서는 최대 80% 개선된 성능을 보여주며 새로운 최첨단 성능을 달성했습니다.



### PERFT: Parameter-Efficient Routed Fine-Tuning for Mixture-of-Expert Mod (https://arxiv.org/abs/2411.08212)
Comments:
          Code available via this https URL

- **What's New**: 본 논문은 Mixture-of-Experts (MoE) 모델을 효과적으로 미세 조정하기 위한 새로운 접근 방식인 Parameter-Efficient Routed Fine-Tuning (PERFT)을 제안한다. 기존의 PEFT(Parameter-Efficient Fine-Tuning) 기술을 MoE 구조에 통합하여 MoE 모델을 보다 유연하고 효율적으로 조정할 수 있는 프레임워크를 소개한다.

- **Technical Details**: PERFT는 두 가지 주요 디자인 차원인 기능적 전략과 구성 전략을 포함하여 다양한 PEFT 모듈을 MoE 메커니즘에 통합한다. 기능적 전략은 PEFT 모듈 내부의 아키텍처, PEFT 모듈의 다중성 및 이들 간의 라우팅 메커니즘을 정의하며, 구성 전략은 원래 MoE 아키텍처와 PEFT 모듈의 상호 작용 방식을 설명한다.

- **Performance Highlights**: OLMoE-1B-7B 및 Mixtral-8×7B 모델을 대상으로 한 실험에서 PERFT는 MoE 모델의 효율적인 조정과 경쟁력 있는 성능을 달성하였다. 특히 PERFT-R은 MoE-agnostic 기준 방식보다 평균 성능이 최대 17.2%까지 향상되었음을 보여준다. 이는 PERFT가 MoE 모듈의 미세 조정 시 매개변수 효율성, 희소성 및 라우팅 간의 무역을 탐색할 수 있는 시스템적인 방법을 제공함을 의미한다.



### What Representational Similarity Measures Imply about Decodable Information (https://arxiv.org/abs/2411.08197)
- **What's New**: 이 논문은 신경 반응의 기하학적 표현과 정보를 선형적으로 디코드하는 능력 간의 밀접한 연관성을 보여줍니다. 구체적으로, CKA(Centered Kernel Alignment) 및 CCA(Canonical Correlation Analysis)와 같은 여러 유사도 측정 방법들이 디코딩 관점에서 설명될 수 있음을 제안합니다.

- **Technical Details**: 논문에서는 M개의 자극 조건에서 신경 집단의 반응을 나타내는 행렬 X와 Y를 사용하여 타겟 벡터 z를 선형 함수로 예측(디코딩)하는 문제를 다룹니다. 이 모델에서 w는 시냅스 가중치 벡터로 해석되며, 이론적으로 디코딩 거리와 Procrustes 거리 간의 경계 관계를 도출합니다.

- **Performance Highlights**: Procrustes 거리가 낮으면 최적의 디코더 평균 거리도 작다는 것을 보여주며, 이는 기존 문헌에서 관찰된 사항을 형식화합니다. 또한 Procrustes 유사성이 높은 경우 CKA 유사성이 높은 값으로 나타나는 등 실제 설정에서 두 방법 간의 관계를 명확히 했습니다.



### An Explainable Machine Learning Approach for Age and Gender Estimation in Living Individuals Using Dental Biometrics (https://arxiv.org/abs/2411.08195)
- **What's New**: 이 연구는 살아있는 개인에서의 나이(age)와 성별(gender) 추정에 대한 새로운 예측 시스템을 개발하고 있으며, 치아 데이터를 활용하여 개선된 정확성을 보여줍니다.

- **Technical Details**: 연구에서는 Cat Boost Classifier (Catboost), Gradient Boosting Machine (GBM), Ada Boost Classifier (AdaBoost), Random Forest (RF), eXtreme Gradient Boosting (XGB), Light Gradient Boosting Machine (LGB), Extra Trees Classifier (ETC)와 같은 다양한 머신러닝(ML) 모델을 활용하여, 862명의 생존 개인의 치아 데이터 분석을 수행했습니다. 또한, SHAP 기법을 활용하여 해석 가능한 AI 모델을 개발하였습니다.

- **Performance Highlights**: RF와 XGB 모델은 나이와 성별 추정에 있어 가장 높은 F1 스코어를 기록하였으며, 특히 XGB 모델은 나이 추정에서 73.26%, RF 모델은 성별 추정에서 77.53%의 F1 스코어를 달성하였습니다.



### TractoEmbed: Modular Multi-level Embedding framework for white matter tract segmentation (https://arxiv.org/abs/2411.08187)
Comments:
          Accepted at 27th International Conference on Pattern Recognition (ICPR), 2024 15 pages, 2 figures

- **What's New**: 이 논문은 백질(White Matter) 경로 세분화의 효율성을 높이기 위해 TractoEmbed라는 모듈 기반의 다단계 임베딩 프레임워크를 제안합니다. 이 프레임워크는 급격한 구조적 차이를 극복하고, 다양한 연령대 데이터셋에서 백질 경로 세분화의 성능을 향상시키면서도, 미래 연구에 추가적인 임베딩을 통합할 수 있는 유연성을 제공합니다.

- **Technical Details**: TractoEmbed는 각기 다른 레벨에서 지역화된 표현을 인코딩하는 학습 과제를 통해 구성된 모듈식 구조를 가지고 있습니다. 이 방법은 개별 스트림라인(streamline), 클러스터(cluster), 패치(patch) 등으로 최대한의 공간 정보를 포착하는 새로운 계층적 스트림라인 데이터 표현을 도입합니다. 또한, 이 프레임워크는 주변 스트림라인의 최소화를 통해 실질적인 임상 환경에서 견고성을 강화하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, TractoEmbed는 다양한 데이터셋에서 기존의 최첨단(SOTA) 방법들을 초월하는 성능을 보여주었으며, 특히 다양한 연령대의 데이터에서도 일반화된 성능을 발휘했습니다. 이는 백질 경로 세분화 분야의 연구와 수술 계획에 중요한 기여를 할 것입니다.



### SCORE: Syntactic Code Representations for Static Script Malware Detection (https://arxiv.org/abs/2411.08182)
- **What's New**: 이 논문에서는 서버 측 스크립트로 인한 악성 코드 탐지를 위한 새로운 기능 추출 및 딥 러닝(DL) 기반 접근 방식을 제안합니다. 기존의 방법들과 비교하여, 스크립트의 복잡성을 효과적으로 다룰 수 있는 새로운 모델들이 개발되었습니다.

- **Technical Details**: 논문에서 제안하는 두 가지 주요 기능 추출 기법은 구문 강조(syntactic code highlighting, SCH)와 추상 구문 트리(abstract syntax tree, AST)입니다. SCH는 정규 표현식(regex)을 사용하여 코드의 구문 요소를 파싱하고, AST는 코드의 구문 구조를 계층적으로 표현합니다. 또한, 순차 모델(SM)과 그래프 기반 모델(GRL) 등을 포함한 여러 딥 러닝 모델을 개발하여 이들 특성을 활용하여 스크립트 악성 코드를 탐지합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 서명 기반 안티바이러스 솔루션보다 최대 81% 높은 진양성률(true positive rate, TPR)을 달성하면서 0.17%의 낮은 위양성율(false positive rate, FPR)을 유지한 것으로 나타났습니다. 또한, 다양한 신경망 기반 탐지기들보다 우수한 성능을 보였습니다.



### Comprehensive and Comparative Analysis between Transfer Learning and Custom Built VGG and CNN-SVM Models for Wildfire Detection (https://arxiv.org/abs/2411.08171)
Comments:
          In Proc. of the 2024 IEEE International Conference On Intelligent Computing in Data Sciences

- **What's New**: 이번 연구는 산불 감지 분야에서 전이 학습(Transfer Learning)의 효율성과 효용성을 분석하고, 사용자 맞춤형 모델과 사전 훈련된(pretrained) 모델 간의 성능을 비교합니다.

- **Technical Details**: 세 가지 목적에 맞춘 모델(VGG-7, VGG-10, CNN-Support Vector Machine(CNN-SVM))과 세 가지 사전 훈련된 모델(VGG-16, VGG-19, Residual Neural Network(ResNet) ResNet101)을 비교하였으며, 다양한 조명 조건, 시간대 및 지형 등 복잡성을 반영하는 산불 데이터셋을 사용하여 훈련 및 평가하였습니다.

- **Performance Highlights**: 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수(F1 Score)와 같은 성능 지표를 사용한 결과, 전이 학습이 사용자 맞춤형 모델보다 효과적임을 입증하였고, 이 연구는 AI 및 ML 분야의 향후 방향성을 제시합니다.



### Large Language Models Can Self-Improve in Long-context Reasoning (https://arxiv.org/abs/2411.08147)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 긴 맥락 추론 능력을 자가 개선할 수 있는 가능성을 탐구하며, 이를 위한 새로운 방법론인 SeaLong을 제안합니다. SeaLong은 여러 출력 샘플을 생성하여 Minimum Bayes Risk를 기반으로 평가한 후, 이를 통해 감독 학습(supervised fine-tuning)이나 선호 최적화(preference optimization)를 적용하는 접근 방식입니다.

- **Technical Details**: SeaLong은 먼저 LLM으로부터 여러 추론 경로를 샘플링하고, 각 경로를 Minimum Bayes Risk (MBR)로 점수화하여, 높은 점수를 받은 출력을 사용해 감독 학습을 진행하거나, 높은 점수와 낮은 점수의 출력을 모두 활용해 최적화를 시도합니다. 이 방식은 일관성이 높은 출력을 우선시하며, 잘못된 출력을 줄이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, SeaLong을 적용한 Llama-3.1-8B-Instruct는 50.8에서 55.0으로 4.2포인트 향상되었습니다. 또한 Qwen-2.5-14B-Instruct는 32B 변형모델보다 더 우수한 성능을 보였으며, 이는 기존의 인적 주석이나 전문가 모델에 의존하지 않고도 이루어진 결과입니다.



### On the Role of Speech Data in Reducing Toxicity Detection Bias (https://arxiv.org/abs/2411.08135)
- **What's New**: 이번 연구에서는 텍스트 기반의 독성 감지 시스템이 특정 인구 집단에 대해 불균형한 오탐지를 나타내는 것과 더불어, 음성 기반의 독성 감지 시스템에서 이러한 편견이 얼마나 완화되는지를 조사했습니다.

- **Technical Details**: 연구에서는 다국어 MuTox 데이터셋을 활용하여 고품질의 그룹 주석(annotations)을 생성하고, 텍스트 기반 독성 분류기와 음성 기반 독성 분류기를 체계적으로 비교했습니다. 그 결과 음성 데이터 사용이 불명확하거나 논쟁을 일으키는 샘플에서 그룹 언급에 대한 편향을 줄이는 데 도움이 됨을 발견했습니다.

- **Performance Highlights**: 음성 기반 시스템이 그룹 편향을 줄이는 데 더 효과적이며, 독성 감지기의 개선이 전사 파이프라인(transcription pipelines)보다 더 유용하다는 것이 밝혀졌습니다. 연구에서는 주석을 공개하고 향후 독성 데이터셋 구축에 대한 권장 사항을 제시했습니다.



### MatPilot: an LLM-enabled AI Materials Scientist under the Framework of Human-Machine Collaboration (https://arxiv.org/abs/2411.08063)
- **What's New**: 이 논문은 새로운 인공지능 재료 과학자인 MatPilot를 제안하고 개발했으며, 이는 재료 과학 연구에서 새로운 기회를 제공합니다.

- **Technical Details**: MatPilot는 다중 에이전트 시스템(multi-agent system)을 통해 인간 과학자 팀의 연구 능력을 강화하는 자연어 인터랙티브(human-machine collaboration) 기능을 갖추고 있습니다. 이 시스템은 복잡한 지식 저장(complex knowledge storage)과 고차원 정보 처리(high-dimensional information processing) 능력을 활용하여 과학적 가설을 생성하고 실험 계획을 수립할 수 있습니다.

- **Performance Highlights**: MatPilot는 효율적인 검증(efficient validation), 지속적인 학습(continuous learning), 반복 최적화(iterative optimization) 능력을 통해 자동화 실험 플랫폼을 통해 실험을 수행할 수 있는 잠재력을 가지고 있습니다.



### Online Collision Risk Estimation via Monocular Depth-Aware Object Detectors and Fuzzy Inferenc (https://arxiv.org/abs/2411.08060)
Comments:
          7 pages (IEEE double column format), 5 figures, 3 tables, submitted to ICRA 2025

- **What's New**: 이 논문은 자율주행차(AV)의 충돌 위험 수준을 모니터링 할 수 있는 프레임워크를 제안합니다. 이 프레임워크는 단일 모노큘러 카메라 이미지만을 사용하여 객체 탐지기의 성능에 기반하여 위험을 추론합니다.

- **Technical Details**: 프레임워크는 주어진 두 세트의 예측값을 활용하며, 하나는 깊이 맵에서 안전 관련 2.5D 객체를 추출하고, 다른 하나는 AV의 3D 객체 탐지기로부터 얻습니다. 논문에서는 Intersection-over-Union (IoU)와 깊이 불일치를 측정하여 두 예측 세트 간의 불일치가 3D 객체 탐지기의 안전 관련 오류와 강력히 상관관계가 있다는 실험적 검증을 제공합니다. 또한 퍼지 추론 시스템 (Fuzzy Inference System, FIS)을 구축하여 이러한 불일치를 충돌 위험 지표에 매핑합니다.

- **Performance Highlights**: 이 프레임워크는 대규모 nuScenes 데이터셋을 사용하여 검증하며, AV를 보호하는 데 유용한 위험 지표를 추론할 수 있음을 보여줍니다. 또한, 제안된 방법은 객체 탐지 기능의 결과로부터 충돌 위험 추정값을 도출하는 혁신적인 시도로, 해석 가능성과 적응성을 제공합니다.



### GREI Data Repository AI Taxonomy (https://arxiv.org/abs/2411.08054)
- **What's New**: 이 논문은 NIH의 지원을 받아 개발된 Generalist Repository Ecosystem Initiative (GREI)에 대한 정보를 제공합니다. 이 이니셔티브는 데이터 리포지토리 역할에 맞춘 AI 분류 체계를 제안하여 리포지토리 관리 전반에 걸쳐 AI 통합을 안내합니다.

- **Technical Details**: GREI는 데이터 리포지토리의 다양한 역할을 여러 단계로 나누어, 데이터의 수집(acquisition), 검증(validation), 조직(organization), 향상(enhancement), 분석(analysis), 공유(sharing), 사용자 지원(user support) 등의 단계로 구성된 체계를 제시합니다. 이 체계는 리포지토리 워크플로우에 AI를 효과적으로 적용하는데 도움을 줍니다.

- **Performance Highlights**: AI를 리포지토리 관리에 통합함으로써 작업 효율성을 높이고, 데이터 사용을 최적화할 수 있는 기회를 제공합니다.



### GraphAide: Advanced Graph-Assisted Query and Reasoning System (https://arxiv.org/abs/2411.08041)
- **What's New**: GraphAide라는 고급 쿼리 및 추론 시스템을 소개하며, 다양한 출처의 지식 그래프(Knowledge Graph, KG)를 구축하고, 생성된 KG에 대해 쿼리를 실행하며 추론을 수행할 수 있게 합니다. 이 시스템은 도메인 특정 데이터와 연결하여 사용자 활동에 맞춘 디지털 어시스턴트를 신속하게 개발하는 데 초점을 맞추고 있습니다.

- **Technical Details**: GraphAide는 다수의 출처에서 수집된 데이터를 바탕으로 KG를 구축하며, LLM(대형 언어 모델)을 활용하여 도메인 특정 디지털 어시스턴트를 개발합니다. RAG(검색 보강 생성)와 시맨틱 웹의 디자인 패턴을 통합하여 제공하는 이 시스템은, 데이터 추출, 명명된 개체 인식, 데이터 모델링 및 쿼리 인터페이스 설계 등 여러 연구 과제를 해결합니다.

- **Performance Highlights**: GraphAide는 지식 그래프(KG)와 LLM을 통해 도메인 특정 디지털 어시스턴트의 효율적이고 간소화된 개발을 강조합니다. 이 시스템은 LLM 생성 응답의 정확도와 일관성을 높이고, 사용자는 자연어 질문을 통해 효과적인 인터페이스와 상호작용할 수 있습니다.



### LAuReL: Learned Augmented Residual Layer (https://arxiv.org/abs/2411.07501)
Comments:
          Accepted at the 2nd Efficient Systems for Foundation Models Workshop at the International Conference on Machine Learning (ICML) 2024

- **What's New**: 논문에서는 기존의 잔여 연결(residual connection)을 일반화한 새로운 구조인 \emph{Learned Augmented Residual Layer} (LAuReL)를 소개하고 있다. LAuReL은 기존 구조를 대체하면서도 모델의 품질과 파라미터 수에서 개선된 성능을 보인다.

- **Technical Details**: LAuReL은 잔여 연결의 기본 구조를 재구성하여 학습 가능한 스칼라 파라미터와 선형 함수를 포함한다. 이 구조는 비선형성에 노출되지 않은 정보가 흐르는 ‘잔여 스트림(residual stream)’ 개념을 활용하여 비선형 구성요소의 학습을 최적화한다. 세 가지 특정 LAuReL 버전이 실험적으로 연구되었으며, 이를 통해 모델의 크기와 속도를 최적화하였다.

- **Performance Highlights**: ResNet-50 모델을 활용한 실험에서, LAuReL을 적용했을 때 추가 레이어를 추가하지 않고도 60%의 성능 증가를 달성하였다. 오히려 0.003%의 파라미터만 증가시키면서 성능을 유지하거나 개선할 수 있었다.




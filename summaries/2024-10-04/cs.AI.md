New uploads on arXiv(cs.CL)

### Erasing Conceptual Knowledge from Language Models (https://arxiv.org/abs/2410.02760)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문은 언어 모델에서 개념 제거(concept erasure)의 평가를 위한 새로운 프레임워크를 제안합니다. 기존의 접근 방식이 효과적인 평가를 제공하지 못하는 문제를 해결하며, 무죄(innocence), 원활함(seamlessness), 특정성(specificity)이라는 세 가지 기준을 바탕으로 한 평가 패러다임을 소개합니다.

- **Technical Details**: 이 연구에서는 Erasure of Language Memory (ELM)라는 새로운 방법을 제안합니다. ELM은 선택적인 저계수 업데이트(low-rank updates)를 활용하여 지워진 개념의 출력 분포를 조정하고, 모델의 전반적인 능력을 유지하며, 지워진 개념에 대한 유창한 생성을 유지합니다. ELM의 평가 지표는 생물 안전(biosecurity), 사이버 보안(cybersecurity), 문학(domain erasure) 작업에 대한 효율성을 검증하기 위해 사용되었습니다.

- **Performance Highlights**: ELM은 다양한 벤치마크에서 기존 방법들과 비교했을 때 우수한 성능을 보여주었습니다. 평가 지표에는 지워진 주제 평가에서 거의 무작위 점수에 가까운 성과, 생성의 유창성, 관련 없는 기준에 대한 정확도 유지, 적대적 공격에 대한 강건성을 포함합니다.



### CorPipe at CRAC 2024: Predicting Zero Mentions from Raw Tex (https://arxiv.org/abs/2410.02756)
Comments:
          Accepted to CRAC 2024

- **What's New**: CRAC 2024 Shared Task에서 다국어(Coreference) 의존성 해결을 위한 CorPipe 24가 우승하였습니다. 이번 대회의 새로운 목표는 이전에는 주어진 빈 노드를 예측하는 것입니다.

- **Technical Details**: 두 가지 모델 변형을 평가했습니다: 1) 두 단계 접근법(두 개의 사전 훈련된 인코더 모델을 사용하여 빈 노드를 먼저 예측한 후 문장의 단어와 함께 처리) 및 2) 단일 단계 접근법(하나의 사전 훈련된 인코더 모델이 빈 노드, Coreference 언급 및 Coreference 링크를 함께 생성).

- **Performance Highlights**: CorPipe는 각각 3.9%와 2.8% 포인트로 경쟁자들을 크게 초과하여 성과를 보여주었습니다.



### SIEVE: General Purpose Data Filtering System Matching GPT-4o Accuracy at 1% the Cos (https://arxiv.org/abs/2410.02755)
- **What's New**: 이 논문에서는 고비용의 일반 목적 모델인 GPT-4o의 정확도를 유지하면서도 훨씬 저렴한 비용으로 데이터 필터링을 수행할 수 있는 새로운 경량 시스템인 SIEVE를 제안합니다. SIEVE는 작은 수의 GPT-4o 호출을 통해 T5 모델을 미세 조정하여 필터링 작업을 효율적으로 수행합니다.

- **Technical Details**: SIEVE는 GPT-4o와 경량 T5 모델을 통합하여 작동합니다. 이 시스템은 웹 규모의 데이터셋을 처리하며, 각 텍스트 스니펫에 대해 '통과' 또는 '실패'로 분류합니다. 활성 학습(Active Learning) 알고리즘을 사용하여 필터링 품질을 유지하면서도 비용을 1% 이하로 줄입니다. 실험적으로 SIEVE는 OpenWebText 데이터셋에서 다섯 개의 맞춤형 필터 작업을 통해 유사한 정확도를 보여줍니다.

- **Performance Highlights**: SIEVE는 GPT-4o와 유사한 필터링 성능을 하지만, 비용은 1%에 불과합니다. 또한 활성 학습 알고리즘을 활용하여 GPT-4o의 쿼리 수를 5배 이상 줄이는 효과를 보입니다. SIEVE의 필터링 결과는 인간 평가자들에게서 긍정적인 피드백을 받았습니다.



### CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation (https://arxiv.org/abs/2410.02748)
- **What's New**: 이 논문에서는 생성형 언어 모델(LLM)에서 요약의 질을 개선하기 위해 핵심 어구(keyphrases)를 활용하는 새로운 접근법인 CriSPO를 제안합니다. CriSPO는 LLMs가 생성한 텍스트의 다각적 비판과 제안을 자동으로 생성하여, 보다 효율적인 프롬프트 변경을 지원합니다.

- **Technical Details**: CriSPO는 Critique-Suggestion-guided automatic Prompt Optimization의 줄임말로, 다양한 측면을 고려하여 텍스트 생성의 최적화를 지원합니다. CriSPO는 인-context Learning (ICL) 예시를 포함하는 템플릿을 생성하며, 다중 지표인 AlignScore와 ROUGE를 동시에 최적화할 수 있는 Automatic Suffix Tuning (AST) 기법도 도입합니다. 이 모델은 간단한 구조로 다양한 LLM에서 일관된 성능 개선을 입증하였습니다.

- **Performance Highlights**: CriSPO는 9개의 다양한 데이터셋에서 검증된 결과, ROUGE 점수를 3-4% 향상시키며 인간 평가에서도 긍정적인 결과를 보였습니다. 또한, CriSPO는 QoA(Question-Answering) 작업에서도 일관된 성능 향상을 달성하였습니다.



### Neutral residues: revisiting adapters for model extension (https://arxiv.org/abs/2410.02744)
- **What's New**: 이 논문은 Pretrained (사전 훈련된) 대형 언어 모델을 새로운 도메인으로 확장하는 문제를 다룹니다. 기존의 방법인 fine-tuning (파인튜닝)이나 low-rank adaptation (저순위 보정) 방식은 새로운 언어를 추가하는 데 한계가 있어, 모델의 원래 도메인 성능이 저하됩니다. 저자들은 새로운 아키텍처와 훈련 절차를 결합하여 원래 도메인에서의 성능 저하 없이 새로운 언어를 학습할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 adapters (어댑터)를 개선하고, 새로운 residual blocks (잔여 블록)을 수정하여 새로운 언어를 학습하면서도 원래 도메인에서의 출력값이 거의 변하지 않도록 합니다. 이 시스템은 mixture of experts (전문가 혼합)로부터 아키텍처 요소를 차용하며, 추가적인 learnable weights (학습 가능한 가중치)를 고작 20%만 사용해도 상당히 향상된 결과를 얻습니다. 또한, 모델의 효율성을 높이는 두 가지 손실 함수를 제안합니다: domain classifier (도메인 분류기)와 sparsity loss (희소성 손실).

- **Performance Highlights**: 논문에서 제안한 접근법은 새로운 언어를 학습하는 동시에 원래의 지식을 잊지 않으며, 파인튜닝이나 기존의 어댑터 방식보다 탁월한 성능을 보여줍니다. 실험을 통해 세 가지 모델에서 두 개의 언어를 추가하는 성능 향상을 검증하였습니다.



### MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions (https://arxiv.org/abs/2410.02743)
- **What's New**: 이번 논문에서는 인간 피드백을 통한 강화 학습(RLHF)의 새로운 프레임워크인 MA-RLHF를 제안합니다. MA-RLHF는 고수준 언어 구성이나 토큰 시퀀스인 매크로 액션(macro actions)을 학습 과정에 포함시켜, 액션과 보상 간의 시간적 거리를 줄여 학습 효율성을 높입니다.

- **Technical Details**: MA-RLHF는 세미 마르코프 의사결정 프로세스(SMDPs) 프레임워크에서 높은 수준의 시간 추상화를 통해 의사결정을 단순화합니다. 전통적인 RLHF의 단일 토큰 최적화 대신, 매크로 액션을 사용하여 장기적인 관점에서의 결정을 학습하게 하여 credit assignment 문제를 경감합니다. 또한 일반적인 토큰화를 반전하는 과정을 통해 고수준 언어 단위를 재구성합니다.

- **Performance Highlights**: MA-RLHF는 텍스트 요약과 코드 생성에서 최대 30%, 대화 생성에서 18%, 질문 응답 작업에서 8%의 성능 향상을 보여줍니다. 또한 기존 RLHF보다 1.7배에서 2배 더 빠른 수렴 속도를 가지며, 추가적인 계산 비용이 발생하지 않습니다.



### Grounding Large Language Models In Embodied Environment With Imperfect World Models (https://arxiv.org/abs/2410.02742)
- **What's New**: 이번 논문에서는 현실 세계의 물리적 세밀함에 대한 직접 경험 부족으로 인해 기본적인 물리적 추론이나 로봇 작업을 처리하는 데 어려움을 겪고 있는 대규모 언어 모델(LLM)의 문제를 해결하기 위해 GLIMO라는 새로운 접근 방식을 제안합니다. GLIMO는 시뮬레이터와 같은 프록시 세계 모델(proxies world models)을 사용하여 훈련 데이터를 수집하고 합성합니다.

- **Technical Details**: GLIMO는 LLM 에이전트 기반 데이터 생성기를 포함하여, 고품질의 다양한 지침 데이터셋을 자동으로 생성합니다. 생성기는 경험 샘플링을 시간적으로 일관되게 만드는 반복 자기 정제 모듈(iterative self-refining module), 다양한 질문-답변 지침 시드(question-answering instruction seeds) 세트, 이전 경험을 반영하기 위한 검색 증강 생성 모듈(retrieval-augmented generation module)로 구성됩니다.

- **Performance Highlights**: GLIMO의 종합적인 실험 결과는 LLaMA-3와 같은 강력한 오픈 소스 LLM의 성능을 각각 2.04 $	imes$, 1.54 $	imes$, 1.82 $	imes$ 향상시킴을 보여줍니다. 이 성능은 GPT-4와 같은 더 큰 모델들과 경쟁하거나 이를 초월할 수 있는 수준입니다.



### Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization (https://arxiv.org/abs/2410.02741)
Comments:
          Accepted to EMNLP 2024 Industry Track

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 요약 성능을 향상시키기 위해 소스 문서에서 추출한 중요한 정보(즉, keyphrases)를 활용한 새로운 접근 방식을 제시합니다. 특히, SigExt라는 경량의 키프레이즈 신호 추출기 모델을 통해 요약 프롬프트를 개선하여 ROUGE 성능 지표를 향상시키는 방법을 연구했습니다.

- **Technical Details**: SigExt 모델은 소스 문서에서 중요한 구문을 추출하고, 이러한 구문을 프롬프트에 포함시켜 LLM이 보다 완전한 요약을 생성하도록 유도합니다. 이 모델은 LLM과 독립적으로 작동하며, 다양한 LLM에 대해 성능을 개선할 수 있도록 설계되었습니다. 또한 keyphrases의 개수를 조정하여 정밀도와 재현율 간의 균형을 조절할 수 있습니다.

- **Performance Highlights**: SigExt를 사용한 결과, 4개의 대표적인 요약 데이터 세트와 3개의 최신 LLM(Claude, Mistral, Falcon)에서 일관된 ROUGE 향상을 달성했습니다. 추가적인 keyphrases를 포함하는 것이 요약의 완전도를 높이는 데 효과적이라는 것을 확인했으며, 구문 수준의 정보가 단어 수준 또는 문장 수준보다 더욱 효과적임을 보여주었습니다.



### Justice or Prejudice? Quantifying Biases in LLM-as-a-Judg (https://arxiv.org/abs/2410.02736)
- **What's New**: 이번 논문에서는 LLM(as-a-Judge) 방식의 평가 방법이 여러 벤치마크에서 널리 사용되고 있음에도 불구하고 그 신뢰성 및 효용성에 영향을 미치는 잠재적인 편향(bias) 문제들을 탐구하지 않았음을 지적합니다. 연구팀은 12개의 주요 편향 유형을 식별하고, 이를 체계적으로 정량화하고 분석하는 새로운 자동화된 편향 정량화 프레임워크인 CALM을 제안합니다.

- **Technical Details**: CALM 프레임워크는 LLM(as-a-Judge)에 존재하는 12개의 편향 유형을 포함하고 있습니다. 이 프레임워크는 평가 과제에 대한 다양한 데이터셋을 통합하며, 페어 방식 비교 또는 점수 평가를 위한 특화된 메트릭스를 사용하여 편향을 정량화하는 시스템적인 접근법을 제공합니다. CALM은 LLM의 판단에 대해 공격-탐지 접근 방식을 사용하여 편향을 탐지합니다.

- **Performance Highlights**: 일부 고급 LLM 모델들이 전체적으로 좋은 성과를 보였지만, 특정 과제에서는 여전히 심각한 편향이 남아 있습니다. CALM 프레임워크를 활용한 평가에서, LLM 모델들이 공정성을 보여주기도 했으나, 여전히 다양한 편향의 영역에서 향상될 여지가 있음이 발견되었습니다.



### Unified Multi-Modal Interleaved Document Representation for Information Retrieva (https://arxiv.org/abs/2410.02729)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 기존의 Information Retrieval (IR) 방법론이 텍스트 정보에만 의존하는 한계를 극복하기 위해, 텍스트, 이미지, 테이블 등 다양한 모달리티를 포괄하는 문서의 통합 표현을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 개발된 메소드는 최근의 Vision-Language Models (VLMs)를 활용하여, 서로 다른 모달리티가 통합된 단일 형식으로 문서를 처리하고 표현합니다. 이를 통해 문서의 전체 맥락과 다양한 부분 간의 관계를 유지할 수 있으며, 중장 문서의 경우에는 섹션들을 병합하여 단일 문서 표현으로 생성하는 전략을 채택합니다.

- **Performance Highlights**: 제안된 IDentIfy 시스템은 텍스트 전용 및 다중 모달 쿼리를 포함한 다양한 정보 검색 시나리오에서 실험적으로 검증되었으며, 기존의 단일 모달리티를 고려한 방법론들에 비해 월등히 우수한 성능을 보였습니다. 전체 문서의 단일 표현을 통한 검색 효과와 고급화된 섹션 재순위 전략인 reranking의 도입이 성과를 극대화했습니다.



### Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation (https://arxiv.org/abs/2410.02725)
- **What's New**: 본 연구에서는 Best-of-N 샘플링 방식에 대한 대안으로, 샘플 수를 적응적으로 줄이면서 성능을 유지 또는 개선할 수 있도록 설계된 새로운 generative self-evaluation 기법을 도입했습니다. 이는 외부 reward model 없이 LLM이 응답을 생성하는 중간 단계에서 더 나은 응답이 생성될 확률을 예측할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 'capability-aware self-evaluations'이라는 새로운 reward modeling 패러다임을 제안합니다. 이를 통해 LLM이 스스로 평가를 통해 얼마나 더 좋은 응답을 생성할지 예측하고, 필요에 따라 샘플을 추가 생성하거나 불필요한 샘플을 조기에 제거할 수 있게 됩니다. 이러한 self-evaluation은 단일 정의된 토큰을 생성하여 수행되며, 외부 reward model의 필요성 없이 최소한의 비용으로 이루어집니다.

- **Performance Highlights**: Llama 3.1 8B 모델은 AlpacaEval에서 GPT-4에 대한 승률이 21%에서 34%로 증가하였고, GSM8K 수학 문제에서의 성능이 84%에서 91%로 향상되었습니다. 적응형 샘플링을 통해 평균 1.2 샘플만으로도 16개의 샘플에서 관찰된 74%의 성능 향상을 달성할 수 있었습니다. 또한 초기 단계에서 성과가 부진한 샘플의 50-75%를 조기에 제거하여 성능 저하를 최소화하면서 56%의 토큰이 절약되었습니다.



### Domain-Specific Retrieval-Augmented Generation Using Vector Stores, Knowledge Graphs, and Tensor Factorization (https://arxiv.org/abs/2410.02721)
Comments:
          9 pages 7 figures, 1 table, 1 cypher code Accepted to ICMLA 2024

- **What's New**: 새로운 연구에서는 SMART-SLIC라는 highly domain-specific LLM 프레임워크를 소개합니다. 이 프레임워크는 Retrieval-Augmented Generation (RAG)와 Knowledge Graph (KG), 벡터 스토어 (VS)를 통합하여 높은 정확도의 질문 응답 시스템을 구축합니다.

- **Technical Details**: SMART-SLIC는 데이터 마이닝, 자연어 처리 (NLP), 비음수 텐서 분해(nonnegative tensor factorization) 및 자동 모델 선택을 통해 LLM의 환상(hallucinations) 문제를 피할 수 있는 도메인 특화 KG와 VS를 제작합니다. 또한, 이 구조는 구조화된 정보와 비구조화된 정보를 모두 활용하여 효과적인 대화형 봇을 개발합니다.

- **Performance Highlights**: SMART-SLIC는 malware 분석과 이상 탐지를 주제로 한 과학 논문 데이터베이스에서 LLM의 질문 응답 능력을 성공적으로 시연합니다. 이 시스템은 정보 원천의 출처를 명확히 하고, 환상을 완화하며, 미세 조정(fine-tuning)의 필요성을 줄여 줍니다.



### UncertaintyRAG: Span-Level Uncertainty Enhanced Long-Context Modeling for Retrieval-Augmented Generation (https://arxiv.org/abs/2410.02719)
- **What's New**: UncertaintyRAG는 Signal-to-Noise Ratio(SNR)에 기반한 span uncertainty를 활용하여 text chunks 간 유사성을 추정하는 새로운 접근 방식을 제안합니다. 이 방식은 모델의 보정(calibration)을 강화하여 모델의 견고함을 개선하고, 무작위 chunking으로 인한 의미 일관성 문제를 완화합니다.

- **Technical Details**: UncertaintyRAG는 비감독(unsupervised) 학습 기법을 사용하여 retrieval 모델을 훈련시키며, 효율적인 데이터 샘플링 및 스케일링 전략을 포함합니다. SNR 기반의 불확실성 측정을 통해 예측의 안정성을 높이고, chunk 언어 모델링 능력과 분리된 chunk embedding을 훈련합니다.

- **Performance Highlights**: UncertaintyRAG는 LLaMA-2-7B에서 기준선 대비 2.03% 향상된 성능을 보여주며, 기존의 고급 오픈 소스 retrieval 모델들보다 4%의 데이터만 사용하여 최고의 결과를 달성합니다. 이 방법은 더 나은 일반화 및 강건성을 제공하며, 특정 LLM에 대한 추가 훈련 없이 다양한 context window 길이에 쉽게 통합될 수 있는 경량 retrieval 모델을 제공합니다.



### LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations (https://arxiv.org/abs/2410.02707)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 내부 표현이 진실성과 관련된 정보를 훨씬 더 많이 포함하고 있다는 사실을 처음으로 밝혀냈습니다. 이는 오류 감지 성능을 크게 향상시킬 수 있는 중요한 발견입니다.

- **Technical Details**: 연구진은 LLM의 특정 토큰에서 진실성 정보를 식별하여, 오류 감지에서의 성능을 향상시키는 새로운 접근 방식을 제시했습니다. 이들은 ‘probing classifiers’를 통해 모델의 내부 표현을 분석하여 진실성과 관련된 오류 유형을 예측할 수 있음을 보여주었습니다.

- **Performance Highlights**: 이 연구는 LLM의 내부 표현과 외부 행동 간의 불일치를 드러내며, LLM이 정확한 답변을 인코딩하지만 잘못된 답변을 생성하는 경우를 관찰했습니다. 또한, 고유한 오류 패턴을 예측할 수 있는 가능성을 제시하여 오류 완화 전략의 개발에 기여할 수 있습니다.



### Selective Attention Improves Transformer (https://arxiv.org/abs/2410.02703)
- **What's New**: Selective Attention (선택적 주의)는 표준 attention 메커니즘의 단순한 파라미터 없는 개선 방법으로, 불필요한 요소에 대한 주의를 줄여 성능을 향상시킵니다. 이 방법은 다양한 모델 크기와 컨텍스트 길이에서 언어 모델링 성능을 개선하며, 기존 transformer와 동등한 성능을 보이면서 약 2배의 머리 수 및 파라미터를 가진 변형 모델과 경쟁할 수 있습니다.

- **Technical Details**: 선택적 주의는 기본적인 attention 메커니즘을 기반으로 하여, 각 토큰이 필요 없는 다른 토큰에 대한 주의를 줄이는 기능을 추가합니다. 이를 통해 주의의 컨텍스트 버퍼 크기를 줄일 수 있으며, 메모리 및 계산 요구 사항을 의미 있게 감소시킵니다. 예를 들어, 100M 파라미터를 가진 transformer가 C4에서 학습된 경우, 선택적 주의가 있을 때 512, 1,024 및 2,048의 컨텍스트 크기로 각각 16배, 25배, 47배 더 적은 메모리를 필요로 합니다.

- **Performance Highlights**: 기존의 transformer와 비교하여, 선택적 주의가 적용된 모델은 더 적은 메모리와 계산 비용으로 더 나은 성능을 발휘합니다. 특히, 선택적 주의가 있는 transformer는 복잡한 자연어 모델링 및 변수 할당 문제에서 우수한 성능을 발휘하며, 임시결과 저장이 필요한 패리티 작업과 같은 단순 작업에서도 모든 중간 결과를 마스킹하는 방식을 통해 적은 자원으로 높은 효율을 보여줍니다.



### HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly (https://arxiv.org/abs/2410.02694)
Comments:
          Code and data are available here: this https URL

- **What's New**: 이번 연구에서는 HELMET (How to Evaluate Long-context Models Effectively and Thoroughly)를 제안하여, 기존의 벤치마크가 가진 단점을 보완하고 다양한 응용 분야에 초점을 맞춘 포괄적인 테스트를 제공합니다.

- **Technical Details**: HELMET는 128k 토큰까지 조절 가능한 입력 길이를 지원하며, 모델 기반 평가를 통해 보다 신뢰할 수 있는 메트릭을 제공합니다. 또한, Few-shot prompting을 통해 기본 모델의 평가 안정성을 강화합니다.

- **Performance Highlights**: HELMET은 51개의 LCLM 모델에 대한 포괄적인 연구를 바탕으로, 합성 작업이 하위 성능 예측에 적합하지 않으며, 서로 다른 카테고리 간 상관관계가 낮다는 것을 발견했습니다. 오픈소스 모델은 긴 맥락이나 복잡한 지시를 요구하는 태스크에서 폐쇄형 모델에 비해 성능이 떨어지는 경향이 있습니다.



### On the Proper Treatment of Tokenization in Psycholinguistics (https://arxiv.org/abs/2410.02691)
Comments:
          Main conference long paper at EMNLP 2024

- **What's New**: 이 논문에서는 언어 모델이 심리 언어학 연구에 어떻게 적용되는지를 다루고 있습니다. 특히, 토큰화(tokenization) 단계에서 발생하는 문제를 해결하기 위해 토큰 레벨 언어 모델을 문자 레벨 언어 모델로 마진화(marginalize)하는 방법을 제안합니다.

- **Technical Details**: 기존 모델들은 토큰 문자열(token strings) 위에서 작업하여 독립적인 문자 문자열(character strings) 와의 비일치(misalignment)를 초래했습니다. 저자들은 실험 영역의 놀라움(surprisal) 계산에 있어 문자 레벨 언어 모델이 필요하다고 주장하며, 이렇게 마진화된 모델은 어떤 문자 부분(substring)에 대한 놀라움을 계산할 수 있게 합니다.

- **Performance Highlights**: 실험을 통해, 저자들은 다양한 초점 영역(focal areas)의 놀라움이 기존의 관심 영역(region of interest) 자체보다 더 나은 심리 계측(psychometric) 예측 변수(predicator)임을 발견하였습니다.



### HiddenGuard: Fine-Grained Safe Generation with Specialized Representation Router (https://arxiv.org/abs/2410.02684)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 안전성을 높이고 인간의 가치에 맞게 조정하는 데 중점을 두고 있습니다. 새로운 프레임워크인 HiddenGuard를 소개하며, 이는 모델의 Harmful Content를 보다 세부적으로 탐지하고 수정할 수 있는 기능을 제공합니다.

- **Technical Details**: HiddenGuard는 Prism(인라인 중재를 위한 rePresentation Router)을 포함하여 모델의 중간 상태를 활용하여 실시간으로 유해 콘텐츠를 탐지하고 수정할 수 있도록 합니다. 이 프레임워크는 유해한 내용의 선별적인 수정이 가능하며, 전체적인 모델의 유용성과 정보성을 유지합니다. 또한, HiddenGuard는 잠재적으로 위험한 정보에 대한 토큰 수준의 세부 주석이 포함된 포괄적인 데이터셋을 함께 제공합니다.

- **Performance Highlights**: HiddenGuard는 실험을 통해 90% 이상의 F1 점수를 기록하며 유해 콘텐츠 탐지 및 수정 작업을 수행했습니다. 이는 모델의 성능을 유지하면서도 정밀도 및 재현율 측면에서 기존 방법들을 초월하는 성과입니다.



### DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Lif (https://arxiv.org/abs/2410.02683)
Comments:
          Preprint. Under Review

- **What's New**: 이번 연구는 일상에서 마주치는 도덕적 딜레마를 다룬 데이터셋 DailyDilemmas를 소개하고, LLMs(대형 언어 모델)가 어떻게 다양한 인간 가치를 우선시하는지 분석합니다. 1,360개의 도덕적 딜레마를 통해 LLM의 결정 과정에서 개인의 윤리적 기준과 가치관의 중요성을 강조합니다.

- **Technical Details**: DailyDilemmas 데이터셋은 인간의 가치가 어떻게 관련될 수 있는지를 보여주기 위해 다양한 일상적인 주제에서 생성된 1,360개의 도덕적 딜레마로 구성됩니다. 이 연구는 World Value Survey, Moral Foundation Theory, Maslow's Hierarchy of Needs, Aristotle's Virtues, Plutchik Wheel of Emotion 등 다섯 가지 이론을 통해 가치 분석을 진행합니다.

- **Performance Highlights**: 연구 결과, LLM들은 World Value Survey에서 생존 가치보다 자기 표현 가치를, Moral Foundation Theory에서 충성도보다 배려 가치를 더 중시하는 경향을 보였습니다. 일부 기본 가치에 대한 모델 간 선호도 차이가 두드러졌으며, 예를 들어 Mixtral-8x7B 모델은 진실성을 9.7% 무시하는 반면, GPT-4-turbo 모델은 9.4% 선택하는 경향이 있었습니다. OpenAI와 Anthropic의 모델 훈련 원칙과 실제 성능 간의 차이를 평가한 결과, 두 모델 모두 원칙과 가치 선호 간의 불일치를 보였습니다.



### Distilling an End-to-End Voice Assistant Without Instruction Training Data (https://arxiv.org/abs/2410.02678)
- **What's New**: 이 새로운 연구는 Audio와 Text를 별도로 모델링하는 기존의 음성 비서 방식의 한계를 극복하기 위해, Self-Supervised Learning에 기반한 Speech Large Language Models (LLMs) 훈련의 새로운 패러다임을 제안합니다. 특히, 주석이 없는 학습을 통해 음성이 아닌 텍스트 LLM의 반응을 활용하여 DiVA(Examined Voice Assistant) 모델을 발전시켰습니다.

- **Technical Details**: DiVA는 'Automatic Speech Recognition (ASR)' 데이터만을 활용하였음에도 불구하고 다양한 작업(Spoken Question Answering, Classification, Translation)에 일반화된 성능을 보입니다. 모델은 기존의 LLM과의 교차 모달(context distillation) 방식으로 Self-Supervised Learning을 통해 훈련됩니다. 기존 데이터에 대한 의존도를 줄이면서도 뛰어난 일반화 능력을 가지고 있습니다.

- **Performance Highlights**: DiVA는 클릭한 모델 Qwen 2 Audio보다 72%의 우선율로 사용자 선호도를 충족시키며, 훈련에 사용된 컴퓨팅 자원은 100배 이상 적습니다. 이러한 성과는 Speech LLM 개선에 있어 효율성을 극대화하였으며, 새로운 데이터 수집 없이도 우수한 성능을 발휘하는 가능성을 보여줍니다.



### CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs (https://arxiv.org/abs/2410.02677)
Comments:
          Preprint. Under review

- **What's New**: 이번 연구에서는 거대한 언어 모델(LLMs)의 문화적 지식을 평가하기 위한 새로운 기준인 CulturalBench를 소개합니다. 이 기준은 1,227개의 인간 작성 및 검증된 질문으로 구성되어 있으며, 45개의 글로벌 지역을 아우릅니다.

- **Technical Details**: CulturalBench는 질문을 두 가지 설정으로 평가합니다: CulturalBench-Easy와 CulturalBench-Hard. Easy 설정은 4개의 선택형 질문으로 구성되어 있으며, Hard 설정은 각 질문을 True/False 형식으로 변환합니다. 질문은 5명의 독립적인 주석자에 의해 검증되었습니다.

- **Performance Highlights**: 모델의 성능은 CulturalBench-Hard에서 가장 잘 수행하는 모델이 61.5%의 정확도를 보였으며, 최악의 경우는 21.4%였습니다. 반면, 인간의 성능은 92.6%로 나타났습니다. 모델들은 다중 정답이 가능한 질문에서 종종 하나의 정답으로 수렴하는 경향을 보였습니다.



### Examining Language Modeling Assumptions Using an Annotated Literary Dialect Corpus (https://arxiv.org/abs/2410.02674)
Comments:
          Accepted to NLP4DH@EMNLP2024

- **What's New**: 본 연구는 19세기 미국 문학의 orthovariant 토큰을 포함한 데이터셋을 제시하며, 이 데이터셋은 컴퓨터 실험을 위한 인간 주석이 달린 방언 그룹 태그를 포함하고 있다.

- **Technical Details**: 연구진은 BERT 및 CANINE과 같은 컨텍스트 언어 모델을 사용하여 초기 실험을 수행하였다. 이 데이터셋은 19세기 미국 문학에서 가져온 4032개의 orthovariant 토큰을 포함하며, 각 토큰은 Dtag(방언 태그)를 통해 특정 그룹에 태깅되었다.

- **Performance Highlights**: BERT-forced 모델이 모든 토큰 변형을 비슷한 영역에 효과적으로 임베딩한 반면, 다른 모델들은 일반적으로 저조한 성능을 보였다. CANINE 시리즈 모델은 생성된 변형과 비생성 변형을 명확하게 분리하였다.



### How to Train Long-Context Language Models (Effectively) (https://arxiv.org/abs/2410.02660)
Comments:
          Our code, data, and models are available at this https URL

- **What's New**: 이 논문에서는 긴 맥락 정보를 효과적으로 활용하기 위한 언어 모델(LM)의 지속적 훈련(continued training) 및 지도 세부 조정(supervised fine-tuning, SFT)에 대해 연구합니다. 기존의 평가 프로토콜을 개선하여 모델 개발을 위한 신뢰할 수 있는 평가 방식이 도입되었습니다.

- **Technical Details**: 제안하는 모델, ProLong-8B는 Llama-3에서 초기화되고 40B 토큰으로 학습되었으며, 128K의 긴 맥락 성능에서 동등한 크기의 모델 중 최첨단 성능을 보여줍니다. 실험 결과, 코드 저장소와 책이 긴 데이터의 훌륭한 출처로 작용하나 고품질의 짧은 데이터와의 조합이 매우 중요함을 나타냅니다. SFT를 통해 짧은 데이터셋만으로도 긴 맥락 작업에서 강력한 성능을 발휘할 수 있음을 발견했습니다.

- **Performance Highlights**: ProLong는 대부분의 긴 맥락 작업에서 Llama-3.18B-Instruct를 초과하는 성능을 발휘하였고, 이는 긴 맥락 훈련 중 5%의 데이터만 사용했음에도 불구하고 이뤄진 결과입니다. ProLong는 512K 토큰까지 효과적으로 처리할 수 있으며, 공개된 LM 중 가장 긴 맥락 창을 가진 모델 중 하나로 평가됩니다.



### Hate Personified: Investigating the role of LLMs in content moderation (https://arxiv.org/abs/2410.02657)
Comments:
          17 pages, 6 Figures, 13 Tables, EMNLP'24 Mains

- **What's New**: 이 연구는 혐오 감지와 같은 주관적인 작업에서 다양한 집단을 반영하는 Large Language Model(LLM)의 능력을 평가하기 위해 추가적인 맥락을 포함한 프롬프트를 분석합니다. 특히, 지리적 신호, 페르소나 속성, 그리고 숫자 정보를 사용하여 LLM의 민감성을 평가했습니다.

- **Technical Details**: 연구는 2개의 LLM, 5개 언어, 6개 데이터셋을 사용하여 86868686개의 영어 프롬프트 및 40404040개의 다국어 프롬프트를 분석했습니다. 이를 통해 LLM의 출력에서 Annotator 간 변동성을 측정하며, 지리적 맥락, 인종별 페르소나, 숫자 기반 컨텍스트가 LLM에 주는 영향을 검토합니다.

- **Performance Highlights**: 지리적 신호는 LLM과 인간 간의 일치도 향상에 기여하며, 페르소나 큐와 숫자 큐의 사용은 서로 다른 평가 집단의 요구를 반영하는 데 도움을 줄 수 있음을 보여줍니다. 하지만, 숫자 정보가 긍정적 및 부정적 효과를 모두 미칠 수 있음을 강조하고, 다국어 프롬프트의 사용이 적절한 민감도를 유지하지 못할 수 있음을 경고합니다.



### Measuring and Improving Persuasiveness of Generative Models (https://arxiv.org/abs/2410.02653)
- **What's New**: 이 논문은 대규모 자동화된 벤치마크 및 퍼션 효과를 평가하기 위한 방법론을 소개하는 최초의 연구로, LLM(대규모 언어 모델)의 설득 능력을 측정하는데 중점을 둡니다. 이 연구는 PersuasionBench 및 PersuasionArena라는 새로운 시스템을 개발하여 AI가 생성한 설득 메시지의 효과를 정량적으로 분석할 수 있도록 합니다.

- **Technical Details**: PersuasionBench와 PersuasionArena는 LLM의 설득 능력을 측정하기 위한 다양한 과제를 포함하는 첫 번째 대규모 벤치마크로, LLM이 생성하는 콘텐츠의 설득력을 정량적으로 평가하고 비교할 수 있도록 합니다. 이 연구에서는 LLM이 언어 패턴을 이용해 설득적인 언어를 생성하는 능력을 극대화하는 방법을 조사하였습니다.

- **Performance Highlights**: 연구 결과, LLM의 설득력은 모델의 크기와 긍정적인 상관관계를 보였지만, 작은 모델도 적절한 훈련을 통해 큰 모델보다 더 높은 설득력을 가질 수 있음을 발견했습니다. 또한, 합성 및 자연 데이터셋을 이용한 목표 지향 훈련이 작은 모델의 설득 능력을 유의미하게 향상시킨다는 점이 강조됩니다.



### Undesirable Memorization in Large Language Models: A Survey (https://arxiv.org/abs/2410.02650)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서의 정보 암기(memorization) 현상을 체계적으로 분석하였으며, 이로 인한 윤리적 및 법적 위험을 조명합니다. 이 논문은 LLM의 암기를 특정 모델 구조에서 어떻게 나타나는지를 분석하고, 이 문제를 해결하기 위한 전략을 제시합니다.

- **Technical Details**: LLM의 암기 현상은 모델이 훈련 데이터에서 구문이나 구절을 저장하고 재생산하는 경향을 나타냅니다. 이 논문은 암기를 다섯 가지 주요 차원인 의도성(intentionality), 정도(degree), 검색 가능성(retrievability), 추상화(abstraction), 투명성(transparency)으로 분석하고, 암기를 측정하는 방법과 그에 기여하는 요소를 탐구합니다.

- **Performance Highlights**: LLMs는 특정 작업에서 인간 수준의 성능을 초과하기도 하지만, 암기에서 발생하는 사생활 및 보안 문제로 인해 지적재산권 저촉과 같은 윤리적 문제가 발생합니다. 이 논문은 향후 연구 방향으로 LLM의 성능과 개인 정보 보호 간의 균형을 맞추고, 대화형 에이전트 및 다양한 언어 모델들에서 암기를 분석하는 방법을 제안합니다.



### Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers (https://arxiv.org/abs/2410.02642)
- **What's New**: 이 논문은 정보 검색(IR)과 관련하여 새로운 in-context re-ranking (ICR) 방법을 제안합니다. 이 방법은 기존의 생성 기반 접근법이 아니라, LLM의 주의(attention) 패턴의 변화를 활용하여 더욱 정확하고 효율적인 재순위를 제공합니다.

- **Technical Details**: ICR은 쿼리(query)에 의해 유발된 주의 패턴의 변화를 활용하여, 두 번의 전방 패스($O(1)$)만으로 N개의 문서를 재정렬합니다. 이는 기존의 생성 기반 재정렬 방식이 요구하는 $O(N)$ 전방 패스와 비교할 때 현저히 더 효율적입니다. 또한, ICR은 특수한 훈련 없이 어떤 LLM에서도 적용이 가능하며, 잘 정형화된 순위를 보장합니다.

- **Performance Highlights**: 실험 결과 ICR은 여러 정보 검색 벤치마크에서 RankGPT를 능가하며, 실제로 지연(latency)을 60% 이상 줄이는 것으로 나타났습니다. 또한, ICR은 복잡한 재순위 신호를 요구하는 작업에서 특히 강력한 성능을 발휘합니다.



### Large Language Model for Multi-Domain Translation: Benchmarking and Domain CoT Fine-tuning (https://arxiv.org/abs/2410.02631)
- **What's New**: 이 논문에서는 다양한 도메인에서의 일관된 고품질 기계 번역(MT)을 달성하기 위한 도전 과제를 다루고 있으며, 이를 위해 15개 도메인에 걸쳐 25개의 독일어⇔영어 및 22개의 중국어⇔영어 테스트 세트를 포함한 포괄적인 벤치마크를 수립하였습니다. 또한, LLMs(대형 언어 모델)의 멀티 도메인 MT에서의 잠재력을 탐구하며, 기존 MT 시스템에 비해 성능 격차를 강조하고 있습니다.

- **Technical Details**: 논문에서 제안하는 방법론인 도메인 체인 오브 생각(Chain of Thought, CoT) 파인튜닝 기술은 LLM이 출처 텍스트의 도메인 정보를 인식하고 이를 번역 과정에 도움 되는 힌트로 활용하도록 유도합니다. 이 방법론은 4개의 도메인에서 훈련된 소량의 데이터 세트를 사용했음에도 불구하고 기존의 파인튜닝 방법보다 번역 정확성과 도메인 강건성을 향상시키며, 독일어⇨영어 번역 작업에서 평균 1.53 BLEU 점수의 증가를 기록하였습니다.

- **Performance Highlights**: 제안한 CoT 파인튜닝 방법은 전통적인 파인튜닝에 비해 번역 품질과 도메인 일반화 능력이 더욱 뛰어나며, 데이터 세트 크기가 400,000 예제로 확장되고 모델 크기가 700억 파라미터로 늘어날수록 성능 향상이 더욱 두드러집니다. 이 접근 방식은 25개 도메인 벤치마크에서 Google, GPT-4, ChatGPT를 초과하는 평균 1.8 BLEU 포인트 이상을 기록했습니다.



### IndicSentEval: How Effectively do Multilingual Transformer Models encode Linguistic Properties for Indic Languages? (https://arxiv.org/abs/2410.02611)
Comments:
          23 pages, 11 figures

- **What's New**: 본 논문에서는 6개의 인디언 언어에 대해 9개의 다국어 Transformer 모델을 사용하여 언어적 속성의 인코딩 능력과 강건성을 조사하였습니다. 이는 기존 연구들이 주로 BERT와 영어에 집중한 것과는 달리, 인디언 언어에 대한 연구로 새로운 기여를 합니다.

- **Technical Details**: 이 연구에서는 6개의 인디언 언어(힌디어, 마라타어, 우르두어, 텔루구어, 칸나다어, 말라얌어)와 13가지 텍스트 섭동(perturbation)에 대한 8가지 언어적 속성(surfaces, syntactic, semantic)을 분석합니다. 이를 위해 새로운 다국어 벤치마크 데이터셋인 IndicSentEval을 소개하며, 약 47,000개의 문장으로 구성되어 있습니다.

- **Performance Highlights**: 전반적으로, 인디언 언어에 대한 언어적 속성을 잘 캡처하는 인디언 전용 모델(MuRIL 및 IndicBERT)과 달리, 보편적 모델(mBERT, InfoXLM 등)은 혼합된 결과를 보였습니다. 흥미롭게도, 보편적 모델이 인디언 전용 모델들보다 13가지 섭동에 대해 더 높은 강건성을 보이며, 특히 명사와 동사를 모두 삭제하거나 동자만 삭제하는 등의 섭동에서 두드러진 성능을 보였습니다.



### Ethio-Fake: Cutting-Edge Approaches to Combat Fake News in Under-Resourced Languages Using Explainable AI (https://arxiv.org/abs/2410.02609)
- **What's New**: 이 연구에서는 소셜 컨텍스트(social context)와 뉴스 콘텐츠(news content) 기능을 통합하여 저자 언어인 암하라어(Amharic)에서 가짜 뉴스(fake news) 탐지의 정확성을 높이는 방법을 제안합니다.

- **Technical Details**: 저자들은 전통적인 머신러닝(ML), 신경망(neural networks), 앙상블 학습(ensemble learning) 및 전이 학습(transfer learning) 등 다양한 방법론을 활용하여 실험을 수행했습니다. 앙상블 학습 접근법이 가장 높은 정확도를 기록하였고, F1 점수 0.99를 달성했습니다. 또한 몬올링구얼(mono-lingual) 모델과 비교할 때, 목표 언어에 맞게 세밀하게 조정된 모델이 0.94의 F1 점수를 기록하며 최상의 성능을 보였습니다.

- **Performance Highlights**: 제안된 모델은 소셜 미디어에서 가짜 뉴스 탐지의 성능을 크게 향상시켰습니다. 특히, 기능을 분석할 때, 설명 가능한 AI(explainable AI) 기법을 사용하여 입력 텍스트를 가짜 또는 진짜로 식별하는 데 중요한 영향을 미치는 단어를 확인하였습니다.



### Agents' Room: Narrative Generation through Multi-step Collaboration (https://arxiv.org/abs/2410.02603)
Comments:
          Under review as a conference paper at ICLR 2025

- **What's New**: 본 논문은 'Agents' Room'이라는 새로운 생성 프레임워크를 제안하며, 이는 서사 이론(narrative theory)에서 영감을 받아 이야기를 쓰는 과정을 여러 하위 작업으로 나눠 전문화된 에이전트들이 해결하도록 하는 방식입니다.

- **Technical Details**: Agents' Room은 복잡한 스토리 작성을 세분화하여 각 하위 작업을 전문화된 에이전트가 수행하도록 하며, 이를 통해 창작 과정의 효율성을 높입니다. 또한, 'Tell Me A Story'라는 고급 데이터셋과 새로운 평가 프레임워크를 소개합니다.

- **Performance Highlights**: 전문 평가자들은 Agents' Room이 생성한 이야기를 기존의 기준 시스템(baseline systems)보다 더 선호했으며, 연구는 자동화된 지표와 인간 기반 메트릭을 통해 생성된 결과물에 대한 철저한 분석을 제공합니다.



### Towards Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions (https://arxiv.org/abs/2410.02584)
Comments:
          Accepted to EMNLP Findings 2024

- **What's New**: 최근의 연구는 대형 언어 모델(LLMs)에서 발생할 수 있는 암묵적 성별 편향을 다루고 있으며, 그 해결을 위한 두 가지 전략인 자기반성(in-context examples) 및 감독적 미세 조정(supervised fine-tuning)을 제시합니다. 이 연구는 LLM 간 다중 에이전트 상호작용의 맥락에서 이 편향을 조사합니다.

- **Technical Details**: 연구팀은 111개의 시나리오로 구성된 데이터셋을 개발하여 LLM의 암묵적 성별 편향을 분석하고, 다중 에이전트 환경에서 역할 배분(Task Assignment) 시에 나타나는 편향을 평가하는 메트릭을 제안합니다. 실증 분석 결과, LLM은 약 50%의 빈도로 강한 암묵적 편향 연결고리를 생성하는 것으로 드러났습니다.

- **Performance Highlights**: 두 가지 완화 전략 모두 효과적으로 편향을 완화하며, 자기반성과 감독적 미세 조정을 결합한 방법이 가장 성공적인 결과를 보였습니다.



### Improving Unsupervised Constituency Parsing via Maximizing Semantic Information (https://arxiv.org/abs/2410.02558)
- **What's New**: 본 논문에서는 무감독 구문 구조 분석기(unsupervised constituency parser)를 훈련하기 위한 새로운 목표인 구문 구조와 문장 의미 사이의 정보를 극대화(SemInfo)를 제안합니다.

- **Technical Details**: SemInfo 측정하기 위해 bag-of-substrings 모델을 도입하고, 확률 가중 정보(probability-weighted information metric)를 적용하여 각 substring에서 인코딩된 의미 정보를 정량화합니다. 또한 이 목표를 Probabilistic Context-Free Grammar (PCFG) 유도에 적용하기 위해 Tree Conditional Random Field (TreeCRF) 기반 모델을 개발하였습니다.

- **Performance Highlights**: SemInfo는 Log-Likelihood (LL) 보다 구문 분석 정확도와 훨씬 강한 상관관계를 보이며, PCFG의 분석 정확도를 5개의 최신 PCFG 변형에서 평균 7.85 포인트 향상시키고, 4개 언어에서 3개 언어에서 새로운 최첨단 결과를 달성하였습니다.



### Algorithms For Automatic Accentuation And Transcription Of Russian Texts In Speech Recognition Systems (https://arxiv.org/abs/2410.02538)
Comments:
          Speech and Computer 20th International Conference, SPECOM 2018, Leipzig, Germany, Proceedings 20

- **What's New**: 이 논문은 러시아어 텍스트의 자동 억양(accents) 및 음소 전사(phonemic transcription)을 위한 규칙 기반 시스템에 대한 개요를 제시합니다. 이 시스템은 Automatic Speech Recognition (ASR)과 같은 음성 연결 작업에 사용됩니다.

- **Technical Details**: 개발된 시스템은 두 가지 주요 부분으로 나뉘며, 억양과 전사가 서로 다른 접근 방식을 사용하여 입력 구문의 올바른 음소 표현(phonemic representations)을 달성합니다. 억양 시스템은 A.A. Zaliznyak의 '러시아어 문법 사전'과 위키 사전(wiktionary) 코퍼스를 기반으로 하며, 동형어(homographs) 구별을 위해 Recurrent Neural Networks (RNN)의 형태학적 정보를 활용합니다. 음소 전사 알고리즘은 B.M. Lobanov 및 L.I. Tsirulnik의 '컴퓨터 합성 및 음성 복제'에 제시된 규칙을 적용합니다.

- **Performance Highlights**: 개발된 도구 모음은 Python 언어로 작성되었으며, ASR 또는 Speech To Text (STT) 작업과 관련된 과학적 연구에 유용한 오픈 소스 모듈로 GitHub에서 접근 가능합니다. 러시아 Voxforge 데이터베이스의 자동 마크 업된 텍스트 주석이 CMU Sphinx의 음향 모델 훈련 데이터로 사용되었으며, 그 결과 음향 모델은 크로스 검증을 통해 평균 단어 정확도(Word Accuracy) 71.2%로 평가되었습니다.



### Contextual Document Embeddings (https://arxiv.org/abs/2410.02525)
- **What's New**: 이 논문에서는 문서 검색(retrieval)에서 밀집 문서 임베딩(dense document embeddings)의 한계를 극복하기 위해 문맥화된(document contextualized) 문서 임베딩을 제안합니다. 본 연구는 문서와 이웃 문서를 함께 고려하는 두 가지 보완적인 방법을 도입합니다.

- **Technical Details**: 첫 번째 방법은 이웃 문서를 명시적으로 포함하는 대조 학습(objective) 목표를 설정하는 것이며, 두 번째는 이웃 문서 정보가 포함된 새로운 아키텍처를 통해 임베딩을 인코딩하는 것입니다. 이를 통해 기존의 biencoder 방식에 비해 다양한 설정에서 성능 향상을 기록했습니다.

- **Performance Highlights**: 우리는 MTEB 벤치마크에서 업계 최고 성능을 달성했으며, 하드 네거티브(mining), 점수 증류(score distillation) 및 추가적인 훈련 없이 구현할 수 있음을 보여주었습니다. 특히 특화된 도메인(예: 금융 및 의료 문서)에서 성능 향상이 두드러졌습니다.



### Methods for Automatic Matrix Language Determination of Code-Switched Speech (https://arxiv.org/abs/2410.02521)
Comments:
          Accepted at EMNLP

- **What's New**: 본 연구는 Code-switching (CS) 음성 데이터를 분석하기 위해 Matrix Language Frame (MLF) 이론을 활용한 새로운 시스템을 개발하였습니다. 이 시스템은 Matrix Language Identity (MLID)를 결정하며, 영어/중국어 및 영어/스페인어 CS 텍스트와 음성을 비교하였습니다.

- **Technical Details**: MLF 이론에 기반하여 세 가지 MLID 시스템이 구현되었습니다. 주요 원칙인 Morpheme Order Principle과 System Morpheme Principle이 적용되었으며, 각 시스템은 텍스트(P1.1, P1.2, P2) 및 오디오 음성(MLID_{P1.1}, MLID_{P1.2}, MLID_{P2})에서 MLID를 결정합니다.

- **Performance Highlights**: 모든 경우에서 MLID 예측자가 텍스트 원칙과 강한 상관관계를 보이며, MLID 인식 작업에서 F1 매크로(60%)와 상관 계수(0.38) 측면에서 LID를 초과하는 성과를 보였습니다.



### Mixed-Session Conversation with Egocentric Memory (https://arxiv.org/abs/2410.02503)
Comments:
          EMNLP Findings 2024 (30 pages); Project website: this https URL

- **What's New**: 이 논문에서는 Mixed-Session Conversation이라는 새로운 대화 시스템을 소개합니다. 이 시스템은 여러 파트너와의 대화를 포함하는 다중 세션 대화를 지원하며, MiSC라는 새로운 데이터셋을 제안합니다.

- **Technical Details**: MiSC는 각 에피소드에 6개의 세션으로 구성된 8,500개의 에피소드를 포함하며, 총 51,000개의 세션이 있습니다. 이 시스템에서 사용되는 새로운 대화 모델은 Egocentric Memory Enhanced Mixed-Session Conversation Agent (EMMA)이며, 주 화자의 관점에서 기억을 관리하는 메커니즘을 통합하여 연속적인 상호작용을 가능하게 합니다.

- **Performance Highlights**: EMMA는 MiSC를 통해 훈련되어 각 세션에서의 대화 파트너가 변경되더라도 일관성 있고 자연스러운 대화 흐름을 유지하는 것으로 확인되었습니다. 또한, 대화의 기억을 잘 유지하면서 전체 대화에서 모순 없이 높은 기억률을 유지하는 것으로 평가되었습니다.



### Defining Knowledge: Bridging Epistemology and Large Language Models (https://arxiv.org/abs/2410.02499)
Comments:
          EMNLP 2024

- **What's New**: 존재하는 대규모 언어 모델(LLMs) 문헌에서 지식(claims on knowledge) 개념에 대한 논란이 많습니다. 이 논문에서는 LLMs의 지식이 진정으로 존재하는지에 대한 질문을 제기하고, 현재 NLP 연구에서 지식이 어떻게 정의되는지를 검토합니다.

- **Technical Details**: 지식의 정의를 정량화하기 위해, 철학의 인식론(epistemology)에서 사용되는 정의를 살펴보고, 이를 LLMs에 적용할 수 있는 방법을 형식화합니다. 이를 통해, 현재 NLP 연구에서의 평가 관행에서 나타나는 단점을 파악하고, 100명의 전문 철학자 및 컴퓨터 과학자들을 대상으로 한 설문조사 결과를 제시합니다.

- **Performance Highlights**: LLMs의 지식 개념에 대한 다양한 의견을 수집한 결과, 지식에 대한 서로 다른 정의가 존재함을 발견했습니다. 이 연구는 LLMs의 지식 평가를 위한 구체적인 프로토콜(testing protocols)을 제안하며, LLMs에 대한 신뢰성 및 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Response Tuning: Aligning Large Language Models without Instruction (https://arxiv.org/abs/2410.02465)
Comments:
          34 pages

- **What's New**: 본 논문은 Response Tuning (RT)이라는 새로운 접근 방식을 제안하며, 이는 instruction-conditioning 단계를 생략하고 오직 응답 공간(supervised response space)에 초점을 맞춥니다. 이를 통해 프리 트레인(pre-trained) LLMs의 잠재력을 끌어내어 유용하고 안전한 채팅 어시스턴트 역할을 할 수 있음을 보이고자 합니다.

- **Technical Details**: Response Tuning (RT)은 instruction-response 쌍을 이용한 기존의 Instruction Tuning(IT) 프로세스와 달리, 모델이 응답을 생성하고 그 분포를 학습하도록 훈련합니다. 연구에서는 최근 네 가지 LLM을 대상으로 하여 RT 모델이 다양한 지시에 효과적으로 응답할 수 있다는 것을 실험적으로 입증하였습니다.

- **Performance Highlights**: RT 모델은 기존 IT 모델들과 비교했을 때 유사한 도움을 줄 수 있으며, 사용자 선호도 또한 개선되었습니다. 특히, 훈련 응답의 분포를 제어하고, 불안전한 쿼리에 대한 거부 응답을 포함시킴으로써 사용자와의 상호작용에서 더 나은 반응을 보였습니다.



### Embedded Topic Models Enhanced by Wikification (https://arxiv.org/abs/2410.02441)
Comments:
          Accepted at EMNLP 2024 Workshop NLP for Wikipedia

- **What's New**: 이 논문에서는 이전의 (dynamic) topic models가 단어의 철자만을 고려하고 동일한 철자로 이루어진 단어의 동음이의어(homography)를 무시하는 문제를 해결하기 위해 Wikipedia 지식을 통합한 새로운 신경망(topic model)을 제안합니다.

- **Technical Details**: 제안된 방법은 두 가지 기술을 기반으로 합니다: 1) entity linking (wikification)과 2) entity embedding (Wikipedia2Vec). Entity linking은 문서에서 특정 개체를 Wikipedia와 같은 지식 기반의 특정 개체에 할당하는 자연어 처리 기법입니다. 이 방법은 topic modeling의 전처리 과정으로 사용됩니다. 그리고 entity embedding은 지식 기반 내 개체의 벡터 표현을 신경망(topic model)에 통합하여 동음이의어로 인한 문제를 해결합니다.

- **Performance Highlights**: 뉴욕 타임즈 기사와 AIDA-CoNLL 데이터셋을 통해 제안한 방법이 신경망 주제 모델의 일반화 성능을 향상시킴을 실험적으로 보여주었습니다. 또한, 훈련된 동적 topic 모델에 의해 추출된 주제와 그 시간적 변화가 타당하다는 것을 확인할 수 있었습니다.



### Better Call SAUL: Fluent and Consistent Language Model Editing with Generation Regularization (https://arxiv.org/abs/2410.02433)
- **What's New**: SAUL은 대규모 언어 모델(LLM)의 편집 과정을 개선하기 위해 제안된 새로운 접근법입니다. 모델의 기능을 유지하며 생기는 계산 비용을 줄이고, 생성 품질을 향상시키는 방법으로 문장 연결과 데이터 증강을 사용합니다.

- **Technical Details**: SAUL은 문장 연결(sentence concatenation)과 증강된 무작위 사실(augmented random facts)을 사용하여 모델의 생성을 정규화합니다. 목표 사실 문장과 무작위 사실 문장을 연결하여 특정 토큰에 대한 과적합(overfitting)을 방지하고, 비관련 지식은 효과적으로 보존합니다.

- **Performance Highlights**: SAUL은 기존의 최첨단 방법보다 뛰어난 모델 편집 성능을 보이면서도 생성 품질과 일관성을 유지합니다. 따라서 SAUL은 실용적이고 신뢰할 수 있는 모델 편집 솔루션으로 자리 잡을 가능성이 높습니다.



### Collective Critics for Creative Story Generation (https://arxiv.org/abs/2410.02428)
Comments:
          EMNLP 2024 (36 pages)

- **What's New**: 이 논문에서는 Collective Critics for Creative Story Generation (CritiCS)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 이야기 계획(Story Plan) 수립 및 이야기 생성(Story Generation) 단계로 구성되어 있으며, 이야기의 창의성과 독자 참여를 높이기 위해 집단 수정 메커니즘을 통합합니다.

- **Technical Details**: CritiCS 프레임워크는 CrPlan 및 CrText라는 두 가지 단계로 나뉘며, 각 단계에서 여러 LLM 비평가들이 드래프트를 평가하고 개선점을 제안합니다. CritiCS는 창의성을 평가하기 위한 다양한 기준을 적용하며, 비평가는 적절한 캐릭터(persona)를 부여받아 이야기를 더욱 독창적으로 만듭니다.

- **Performance Highlights**: 성공적인 평가에 따르면, CritiCS는 기존 최첨단 방법들에 비해 창의성과 흥미로움에서 큰 차이를 보이며, 사용자들과의 상호작용을 통해 이야기를 개선하는 새로운 가능성을 제시합니다.



### Learning the Latent Rules of a Game from Data: A Chess Story (https://arxiv.org/abs/2410.02426)
- **What's New**: 이번 연구는 소규모 사전 학습된 생성 언어 모델(Small Language Models, SLMs)이 체스의 규칙을 학습할 수 있음을 보여줍니다. 모델은 1,000부터 1,000,000개의 예제가 주어질 때, 체스 문제를 해결하고 정당한 수를 제안하는 등의 작업을 수행합니다.

- **Technical Details**: 28M 및 125M 파라미터를 가진 소규모 사전 학습된 생성 언어 모델(SLM)을 사용하여 체스의 규칙을 학습하는 데 필요한 데이터의 양과, 학습이 데이터와 비례하여 확장될 수 있는지를 탐구했습니다. 연구는 기본 모델이 수행할 수 없는 복잡한 작업을 정확하게 수행할 수 있도록 세밀하게 조정된 모델을 개발하는 방법을 제안합니다.

- **Performance Highlights**: 모델의 정확도는 주어진 예제의 수가 증가함에 따라 개선되며, 연속적인 언어 모델 조정 에폭(epoch)의 영향도 관찰되었습니다. 또한, 조정된 모델은 체스에서 상당한 성능을 보였음이 확인되었습니다.



### MenakBERT -- Hebrew Diacriticizer (https://arxiv.org/abs/2410.02417)
Comments:
          Published at ISCOL2022 as a poster

- **What's New**: 이번 연구에서는 히브리어 텍스트에 이탈릭 부호(diacritical marks)를 추가하는 새로운 접근 방식으로 MenakBERT 모델을 제안합니다. 이 모델은 히브리어 텍스트에 대해 사전 훈련된 character-level transformer를 기반으로 하고 있습니다.

- **Technical Details**: MenakBERT는 기존의 human-curated resource에 의존하는 시스템의 한계를 극복하기 위해 개발되었습니다. 이 모델은 히브리어 문장의 이탈릭 부호 추가를 위한 fine-tuning을 통해 성능을 개선하였으며, 이러한 파라미터 조정은 다른 작업, 예를 들어 품사 태깅(part of speech tagging)에도 효과적으로 전이됩니다.

- **Performance Highlights**: MenakBERT는 이탈릭 부호 추가에 대해 기존 모델들과 비교하여 현저한 성능 향상을 보여주며, 히브리어 텍스트 처리 분야에서 중요한 진전을 이루었습니다.



### MetaMetrics: Calibrating Metrics For Generation Tasks Using Human Preferences (https://arxiv.org/abs/2410.02381)
Comments:
          Preprint

- **What's New**: MetaMetrics라는 새로운 메타 메트릭(metric)을 도입하여, 다양한 작업을 평가하는 데 있어 인간의 선호도에 더 잘 맞추도록 기존 메트릭들을 조정하는 방법을 제안합니다.

- **Technical Details**: MetaMetrics는 감독된 방법으로 캘리브레이션(calibration)된 메타 메트릭으로, 언어 및 비전 관련 다운스트림 작업에서 유연성과 효과성을 보여줍니다. 이 메트릭은 참조 기반(reference-based) 및 비참조(reference-free) 두 가지 설정에서 작동하여 다양한 작업과 모달리티(modality)에 걸쳐 널리 활용될 수 있습니다.

- **Performance Highlights**: MetaMetrics는 여러 다국어 및 다영역(multi-domain) 시나리오에서 인간 선호도와 밀접하게 정렬되는 효과를 보여주며, 기존 메트릭들과 비교했을 때 성능이 월등히 우수함을 입증했습니다.



### Towards Comprehensive Detection of Chinese Harmful Memes (https://arxiv.org/abs/2410.02378)
- **What's New**: 본 논문은 NeurIPS 2024 D & B Track에서 채택되었으며, 중국 인터넷에서 유해한 밈(harmful memes)의 탐지가 저조한 이유로 신뢰할 수 있는 데이터셋(dependency dataset)과 효율적인 탐지기(dedector)가 부족하기 때문임을 강조합니다.

- **Technical Details**: ToxiCN MM이라는 첫 번째 중국 유해 밈 데이터셋을 구축하였으며, 이 데이터셋은 다양한 밈 유형에 대한 세부 주석을 포함한 12,000개의 샘플로 구성되어 있습니다. 또한 Multimodal Knowledge Enhancement (MKE)라는 기본 탐지기를 제안하며, 이는 LLM이 생성한 밈 콘텐츠의 맥락(contextual information)을 통합하여 중국 밈에 대한 이해를 향상시킵니다.

- **Performance Highlights**: 평가 단계에서 우리는 여러 기본 모델(LLMs 포함)에 대한 정량적 실험과 정성적 분석을 실시했습니다. 실험 결과, 기존 모델이 중국 유해 밈을 탐지하는 데 어려움이 있음을 보여주었고, MKE의 효과성이 입증되었습니다.



### From Concrete to Abstract: A Multimodal Generative Approach to Abstract Concept Learning (https://arxiv.org/abs/2410.02365)
- **What's New**: 이 논문은 Concrete(구체적) 개념과 Abstract(추상적) 개념을 모두 이해할 수 있는 multimodal generative(다중 모달 생성) 접근 방식을 제시합니다. 이 모델은 시각 정보와 범주적 언어 정보의 통합을 통해 고차원 추상 개념 학습을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 세 가지 개념 수준(하위 수준, 기본 수준, 상위 수준)으로 구성되어 있습니다. 모델은 하위 수준 구체적 개념을 학습한 후, 이를 결합해 기본 수준 개념을 형성하고, 마지막으로 이를 바탕으로 상위 수준 추상 개념을 학습합니다. 본 연구에서는 Multimodal Mixture-of-Experts Variational Autoencoders (MMVAE) 구조를 기반으로 합니다.

- **Performance Highlights**: 모델은 언어-시각(visual-to-language) 및 시각-언어(language-to-visual) 테스트를 통해 고차원 추상 개념에 대한 학습 능력을 평가하였으며, 높은 성과를 보였습니다. 실험 결과, 모델의 언어 이해 및 명명(Language Naming) 작업에서의 능숙함이 입증되었습니다.



### AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models (https://arxiv.org/abs/2410.02355)
- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)에서 'hallucination'(환각) 문제를 해결하기 위해 AlphaEdit이라는 새로운 모델 편집 방법을 제안합니다. 기존의 locate-then-edit 접근 방식에서는 지식 업데이트 시 기존 지식을 손상시키는 문제가 있었으나, AlphaEdit은 'null space'(영 공간)에서 perturbation(섭동)을 투영하여 이러한 문제를 해결합니다.

- **Technical Details**: AlphaEdit은 모델 파라미터를 수정하는 방법으로, 기존의 모델 편집 방식에서 지식 보존(Error e0)과 업데이트(Error e1) 간의 균형 문제를 해결합니다. 이 과정에서 perturbation은 보존된 지식의 null space에 투영되어, post-edited LLM의 출력을 보존된 지식에 대해 변경하지 않도록 보장합니다.

- **Performance Highlights**: 다양한 LLM에 대한 실험 결과, AlphaEdit은 기존의 최상위 모델 편집 방법에 비해 평균 36.4%의 성능 향상을 달성했습니다. 이 방법은 기존 모델 편집 방법에 단 한 줄의 코드만 추가하여 쉽게 통합될 수 있습니다.



### Listening to the Wise Few: Select-and-Copy Attention Heads for Multiple-Choice QA (https://arxiv.org/abs/2410.02343)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 평가 방식에 대한 새로운 접근을 제시합니다. 기존의 다중 선택 질문(MCQA) 평가 방식의 한계를 극복하기 위해, Query-Key Score (QK-score) 및 Attention Score라는 새로운 점수를 도입하였습니다.

- **Technical Details**: QK-score는 주의 기법(attention heads)에서 쿼리(query)와 키(key) 표현 간의 상호작용을 기반으로 하며, Attention Score는 주의 가중치를 기반으로 합니다. 이 점수는 특정 선택 및 복사(select-and-copy) 헤드에서 추출되며, MCQA 데이터셋에서 일관된 성능을 보여주었습니다.

- **Performance Highlights**: 이 방법은 LLaMA2-7B 모델에서 최대 16%의 성능 향상을, 더 큰 모델에 대해서는 10% 향상을 이루어냈으며, 간단한 합성 데이터셋에서 거의 60%의 정확도로 MCQA 형식의 한계를 극복하는 효율성을 입증하였습니다.



### How Much Can RAG Help the Reasoning of LLM? (https://arxiv.org/abs/2410.02338)
- **What's New**: 본 연구는 Retrieval-Augmented Generation (RAG) 기법이 LLMs의 추론 능력을 향상시키는 방법을 심도 있게 탐구하고 있으며, 이 과정에서 문서의 정보 전처리와 관련된 새로운 접근법인 DPrompt tuning을 제시합니다.

- **Technical Details**: RAG는 외부 문서의 정보를 사용하여 LLM의 추론 과정을 지원하나, 이 과정이 고정 깊이 트리 구조에서 제한적임을 보여줍니다. LLM의 추론 과정은 고정 깊이에서 운영되며, RAG는 문서에서 중간 추론 결과를 활용할 수 있지만 노이즈 정보를 필터링하는 과정이 필요합니다. 연구는 한정된 트랜스포머 레이어 내에서 효과적인 필터링을 가능하게 하는 DPrompt tuning을 제안하여 성능 개선을 보여줍니다.

- **Performance Highlights**: RAG 기법을 통해 LLM이 더 복잡한 문제를 해결할 수 있음을 입증하였으며, 관련 문서 정보 추가 시 문제 해결 깊이를 증가시킬 수 있는 가능성을 발견했습니다. 하지만, 실제 RAG 시나리오에서 필터링 문제는 여전히 어려움이 존재하고, DPrompt tuning을 통해 이 문제를 효과적으로 개선할 수 있음을 보였습니다.



### Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection (https://arxiv.org/abs/2410.02330)
- **What's New**: 이 논문에서는 기존의 파라미터 효율적인 파인튜닝 기법과 레이어 확장 방법이 모든 LLM 레이어에 동일한 지식을 적용하는 문제를 제기합니다. 저자들은 각 레이어의 중요성을 평가하여 지식 주입의 최적 레이어 범위를 찾았습니다. 흥미롭게도, 얕은 레이어가 지식 주입에 가장 중요한 역할을 한다는 사실을 발견했습니다. 이를 바탕으로 'S 전략'을 제안하고, Llama Slayer-8B와 Llama Slayer-8B-Instruct라는 새로운 LLM을 소개합니다.

- **Technical Details**: S 전략은 얕은 레이어를 선택적으로 향상시키는 동시에 효과가 적은 깊은 레이어를 제거하는 post-pretraining 전략입니다. 연구에서는 코드와 수학 데이터셋에서 이 전략의 효과를 입증하였습니다. 앵귤러 거리(angular distance) 메트릭을 이용하여 각 블록의 중요성을 평가하는 방법론도 제시합니다.

- **Performance Highlights**: Llama Slayer-8B와 Llama Slayer-8B-Instruct는 프로그래밍, 수학 및 일반 언어 작업에서 뛰어난 성능을 보였습니다. 여러 LLM 및 법률 코퍼스에 대한 추가 실험을 통해 이 방법론의 일반적인 적용 가능성과 우수성을 확증했습니다.



### Post-edits Are Preferences Too (https://arxiv.org/abs/2410.02320)
Comments:
          To appear at the Ninth Conference on Machine Translation (WMT24)

- **What's New**: 본 논문에서는 기계 번역 분야의 Pairwise Preference (쌍 선호도) 피드백이 다른 형태의 피드백보다 신뢰성이 떨어진다는 점을 지적하며, Post-editing (수정 편집) 데이터를 활용하여 더 믿을 수 있는 인간의 선호도를 생성하는 방법을 제안합니다.

- **Technical Details**: Preference Optimization (PO) 기법을 통해, 수정 편집 후의 번역 결과를 새로운 기준으로 활용하고, 이를 통해 LLM (대규모 언어 모델)을 Fine-tune (미세 조정)합니다. 모델은 수정 편집 데이터로 사전 훈련 후, PO 손실 함수를 사용하여 선호도에 맞게 좀 더 우수한 번역 결과를 생성하도록 학습합니다.

- **Performance Highlights**: 사전 훈련 후 PO 손실로 Fine-tuning을 진행한 결과, 모델은 수정 편집과 유사한 출력을 상위 순위로 올리고, 기계 번역과 유사한 결과를 지향하지 않는 방식으로 훈련되었습니다. 이를 통해 품질과 신뢰성을 높인 번역 결과를 얻을 수 있음을 보여줍니다.



### Traffic Light or Light Traffic? Investigating Phrasal Semantics in Large Language Models (https://arxiv.org/abs/2410.02308)
Comments:
          EMNLP 2024

- **What's New**: 이번 연구에서는 API 기반 대형 언어 모델(LLMs)의 구문 의미 이해 능력을 평가하며, 자연어 지침에 따라 구문 의미 추론 작업을 수행하는 LLM의 성능을 분석합니다. 또한, 몇 가지 프롬프트 기법인 few-shot demonstrations과 Chain-of-Thought reasoning이 모델 성능에 미치는 영향을 탐구합니다.

- **Technical Details**: 본 연구는 구문 의미 이해에 있어 LLM과 기존 방식(임베딩 기반 방법 및 fine-tuning 방법) 간의 성능을 비교합니다. 다양한 데이터셋(Turney, BiRD, PiC-PS)을 사용하여 LLM의 성능을 평가하며, 특히 자연어 지침에 기반한 접근 방식을 통해 LLM의 팩트 기반 reasoning 능력을 강화합니다.

- **Performance Highlights**: 이 연구의 결과는 LLM이 전통적인 임베딩 방법보다 뛰어난 성능을 보이지만, fine-tuning 방법에 있어서는 유의미한 차이가 없음을 보여줍니다. Turney 데이터셋에서는 정확도 57.1에서 87.6으로 향상되었으며, BiRD는 0.689에서 0.761로, PiC-PS는 69.3에서 73.5로 향상되었습니다. 하지만, 특정 컨텍스트를 이해하는 PiC-PS에서는 LLM이 minor improvement에 그쳤습니다.



### Make Compound Sentences Simple to Analyze: Learning to Split Sentences for Aspect-based Sentiment Analysis (https://arxiv.org/abs/2410.02297)
Comments:
          Accepted at EMNLP 2024 (Findings, long paper)

- **What's New**: 이 논문에서는 Aspect Term Oriented Sentence Splitter (ATOSS)라는 새로운 모델을 제안합니다. ATOSS는 복합 문장을 간단하고 명확한 형태로 분할하여 ABSA(Aspect-Based Sentiment Analysis) 모델이 의도를 식별하는 데 도움을 줍니다.

- **Technical Details**: ATOSS는 복합 문장을 단순화하여 주어진 문장에서 감정 쿼드플렛을 추출하는 작업을 용이하게 만드는 플러그 앤 플레이 모듈입니다. 이 모델은 LLM(대형 언어 모델) 증류를 통해 최적화되고, 특정 ABSA 모델의 문장 선호도와 정렬됩니다.

- **Performance Highlights**: 실험 결과, ATOSS를 활용하면 ASQP(Aspect Sentiment Quad Prediction)와 ACOS(Aspect-Category-Opinion-Sentiment) 작업에서 기존 방법보다 우수한 성능을 보여줍니다. ATOSS는 특히 기초 ABSA 모델의 쿼드플렛 예측 정확도를 향상시킵니다.



### Language Models are Graph Learners (https://arxiv.org/abs/2410.02296)
- **What's New**: 이 논문은 기존의 Language Models (LMs)가 Graph Neural Networks (GNNs)와 Graph Transformers (GTs)와 같은 도메인 특화 모델에 도전할 수 있는 새로운 접근 방식을 제안합니다. AugGLM (Augmented Graph Language Model)이라고 불리는 이 모델은 노드 분류 작업에서 최신 GNNs와 비등한 성능을 달성하면서도 LM의 원래 아키텍처를 그대로 유지합니다.

- **Technical Details**: AugGLM은 그래프 데이터를 처리하기 위한 두 가지 주요 증강 전략을 채택합니다: (1) 토폴로지 및 의미 검색 방법을 사용하여 LMs의 입력을 보강하고, (2) 경량 GNN 분류기를 통해 LMs의 분류 프로세스를 유도하여 클래스 후보를 효과적으로 정제합니다. 이러한 접근법은 LMs가 다양한 데이터셋에서 공동으로 훈련할 수 있는 능력을 유지합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험 결과, AugGLM을 장착한 Flan-T5 모델이 기존의 텍스트 출력 노드 분류기보다 우수한 성능을 보였으며, 최신 벡터 출력 노드 분류기와 비견되는 성과를 기록했습니다. 이는 전문적인 작업별 노드 분류기와 일반 LMs 간의 격차를 줄이는 중요한 이정표가 됩니다.



### Correlation and Navigation in the Vocabulary Key Representation Space of Language Models (https://arxiv.org/abs/2410.02284)
- **What's New**: 본 논문에서는 다음 토큰 예측(Next-Token Prediction, NTP)에서 키 분포가 NTP 분포에 미치는 영향을 분석하고, 키 간의 유사성이 잘못된 상관관계를 유발하는지에 대해 집중적으로 연구합니다. 이를 통해 정보 검색(task)에서 부정확한 예측이 발생하는 원인을 규명하고, 개선 방안을 제시합니다.

- **Technical Details**: 키(key) 분포는 고정된 어휘 표현으로 간주되며, 입력 컨텍스트를 인코딩한 쿼리(query)와의 소프트맥스(softmax) 정규화된 내적(dot product)을 통해 NTP 분포가 생성됩니다. 본 연구에서는 'in-context navigation (ICN)'이라는 새로운 방법을 통해 탐색된 토큰에서 쿼리 표현을 효율적으로 멀리하게 하고, 이를 통해 정확한 새로운 키(key)를 탐색할 수 있는 방안을 제안합니다.

- **Performance Highlights**: 실험 결과, ICN 방법을 적용했을 때 지식 탐색(knowledge probing)에서의 정확성이 현격히 개선되었으며, 개방형 생성(open-ended generation)과 사고의 연쇄 생성을 보다 다양하게 수행할 수 있음을 보여주었습니다. NTP의 성능이 키의 고정된 공간(fixed key space)와 관련하여 제기되는 위험성과 향후 연구 방향을 논의합니다.



### Morphological evaluation of subwords vocabulary used by BETO language mod (https://arxiv.org/abs/2410.02283)
Comments:
          in Spanish language

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models)에서 사용하는 서브워드 토큰화(subword tokenization) 알고리즘의 형태소(morpheme) 품질을 평가하는 새로운 방법을 제안합니다.

- **Technical Details**: 형태소 품질 평가는 관련성(relevance), 결속력(cohesion), 형태소 정확도(morphological accuracy)의 세 가지 품질 기준에 기반하며, BETO 모델의 토크나이저에 적용되었습니다. 사용된 알고리즘은 Wordpiece입니다.

- **Performance Highlights**: BETO 모델에서 생성된 어휘(vocabulary)의 형태소 품질이 낮다는 결론에 도달했으며, 큰 말뭉치(corpus)에서 트레이닝해도 품질이 개선되지 않는다는 결과를 보여줍니다.



### Annotation Guidelines for Corpus Novelties: Part 1 -- Named Entity Recognition (https://arxiv.org/abs/2410.02281)
- **What's New**: 이 논문에서는 Novelties corpus라는 소설 및 소설의 일부를 포함하고 있는 데이터셋이 명명 개체 인식(Named Entity Recognition, NER)을 위한 주석이 달린 것에 대해 설명하고 있습니다. 이 데이터셋은 긴 텍스트를 처리할 수 있는 NER 방법의 훈련 및 테스트는 물론, 문학 허구에서 캐릭터 네트워크를 추출하기 위한 파이프라인 개발을 목적으로 하고 있습니다.

- **Technical Details**: Novelties corpus는 다양한 명명 개체들을 주석하는 데 사용되는 지침을 포함하고 있습니다. 관련 규칙과 예시들이 제공되며, 적절한 명명 개체의 식별과 분류에 대한 지침을 제시합니다. 이 데이터셋은 CoNLL-2003 및 OntoNotes v5와 같은 다른 NER 데이터셋과 비교하여 문학적 특성에 맞게 다소 차별화된 접근 방식을 가지고 있습니다. 이 연구는 명명 개체의 정의 및 분류 방법, 특히 캐릭터에 집중하고 있습니다.

- **Performance Highlights**: 이 논문은 캐릭터를 포함한 다양한 유형의 명명 개체를 포함하는 주석 체계를 제공합니다. 또한, 소설 내에서의 명명 개체 인식을 위한 내부 증거와 외부 증거를 구분하며, 이를 통해 주석의 일관성과 신뢰성을 높이고자 합니다.



### EmbedLLM: Learning Compact Representations of Large Language Models (https://arxiv.org/abs/2410.02223)
- **What's New**: 본 연구에서는 Huggingface에서 수백만개의 대형 언어 모델(LLMs)을 효율적으로 평가하고 활용하기 위한 EmbedLLM이라는 프레임워크를 제안합니다. 기존 방법들이 각 태스크에 맞는 표현을 반복적으로 학습하여 비효율성을 초래하는 문제를 해결하고자 합니다.

- **Technical Details**: EmbedLLM 프레임워크는 LLMs의 컴팩트한 벡터 표현을 학습하는 인코더-디코더 접근법을 도입합니다. 학습된 임베딩(embedding)은 다양한 모델의 특성과 태스크를 반영하여 성능 향상을 도모합니다. 이를 통해 정확도 예측(correctness forecasting), 모델 라우팅(model routing), 벤치마크 정확도 평가(benchmark accuracy evaluation)와 같은 다양한 다운스트림 태스크에 활용할 수 있습니다.

- **Performance Highlights**: Empirical results show that EmbedLLM은 모델 라우팅에서 정확도 및 대기 시간(latency) 모두에서 이전 방법들을 초월합니다. 또한, 추가적인 추론 비용 없이 여러 벤치마크에서 모델의 성능을 예측할 수 있음을 입증하였습니다. 실험을 통해 학습된 임베딩이 모델의 주요 특성을 포착하고, 유사한 특성을 가진 모델들이 임베딩 공간에서 가까운 위치를 유지함을 확인하였습니다.



### Calibrate to Discriminate: Improve In-Context Learning with Label-Free Comparative Inferenc (https://arxiv.org/abs/2410.02210)
Comments:
          19 pages

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 인컨텍스트 학습(in-context learning)에서 특정한 오교정(miscalibration) 현상, 즉 정확한 예측과 부정확한 예측에 동일한 수준의 신뢰도가 부여되는 '무차별 오교정(indiscriminate miscalibration)' 현상을 발견하였습니다. 이러한 현상을 측정하기 위한 새로운 메트릭스를 제안하고, 오교정을 완화하며 분류 성능을 개선하기 위한 새로운 인컨텍스트 비교 추론 방법을 개발하였습니다.

- **Technical Details**: 기존의 Expected Calibration Errors (ECE)와 같은 전통적인 교정 메트릭스가 이 현상을 충분히 포착하지 못함을 보여주었으며, 이를 보완하기 위해 제안한 메트릭을 통해 심각성을 측정할 수 있습니다. 제안된 label-free in-context comparative inference 방법은 레이블이 없는 샘플을 프롬프트에 추가하여 모델이 예측을 조정하고 교정하도록 유도하며, 추가된 예시가 모델이 테스트 예제를 더 잘 이해하도록 돕습니다. 실험을 통해 F1 점수, 정확도 및 전통적인 교정 메트릭 측면에서 성능 향상이 입증되었습니다.

- **Performance Highlights**: 다섯 개의 데이터셋을 통한 광범위한 실험을 통해 제안한 방법이 일반적인 zero-shot 및 few-shot prompting에 비해 더 정확하고 교정된 예측을 달성할 수 있다는 것을 입증했습니다. 특히, 제안된 레이블 없는 인컨텍스트 비교 추론 방식이 무차별 오교정을 완화하여 올바른 예측에 더 높은 신뢰도를 부여하고, 부정확한 예측에 낮은 신뢰도를 부여하는 데 기여함을 보여주었습니다.



### Measuring, Evaluating and Improving Logical Consistency in Large Language Models (https://arxiv.org/abs/2410.02205)
- **What's New**: 본 논문은 논리적 일관성(logical consistency)이 LLM(대규모 언어 모델)의 신뢰성과 신뢰성 보장을 위한 중요한 요소라는 점에 초점을 맞추고 있습니다. 연구자들은 논리적 일관성을 측정하기 위해 보편적인 프레임워크를 제안하며, 이를 통해 다양한 LLM의 성능을 평가하였습니다.

- **Technical Details**: 세 가지 기본적인 프록시(proxies)인 추이성(transitivity), 교환성(commutativity), 부정 불변성(negation invariance)을 사용하여 LLM의 논리적 일관성을 정량화하는 새로운 프레임워크를 제안합니다. 이러한 프레임워크는 다양한 도메인에 적용 가능하며, 부분 비교와 전체 비교를 지원합니다.

- **Performance Highlights**: 논리적 일관성을 향상시키기 위한 데이터 정제(data refinement) 및 증강(augmentation) 기술을 도입하여 LLM의 성능을 향상시켰습니다. 이 접근법으로 훈련된 LLM은 논리적 일관성이 개선되었으며, 이후의 로직 의존 알고리즘에서도 우수한 성능을 보였습니다. 전반적으로 논리적 일관성은 LLM의 신뢰성을 평가하는 데 유용한 지표가 될 수 있음을 보여줍니다.



### Can Language Models Take A Hint? Prompting for Controllable Contextualized Commonsense Inferenc (https://arxiv.org/abs/2410.02202)
Comments:
          Submitted to ACL Rolling Review. arXiv admin note: text overlap with arXiv:2302.05406

- **What's New**: 이번 연구에서는 'hinting'이라는 데이터 증강(data augmentation) 기술을 도입하여 맥락 있는 commonsense inference를 개선합니다. 이 기술은 하드(hard) 및 소프트(soft) 프롬프트를 활용한 접두사(prompts) 전략을 사용하여 추론 과정을 유도합니다.

- **Technical Details**: HINTING 기법은 주어진 이야기 맥락에서 commonsense assertion을 생성하는데 도움을 줍니다. 이 방법은 Target Sentence와 함께 등장하는 특정 관계를 통해 commonsense assertion의 주체, 관계, 객체를 효율적으로 추론할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, HINTING 기술은 ParaCOMET 및 GLUCOSE 데이터셋에서 일반적인 및 맥락 특정 추론의 성능을 향상시키며, 모델의 성능에 손상을 주지 않으면서 더 나은 통제력을 제공합니다.



### POSIX: A Prompt Sensitivity Index For Large Language Models (https://arxiv.org/abs/2410.02185)
Comments:
          EMNLP 2024 (Findings)

- **What's New**: 이 논문에서는 POSIX라는 새로운 지표(PrOmpt Sensitivity IndeX)를 제안하여, Large Language Models(LLMs)의 프롬프트 감도(prompt sensitivity)를 평가하는 신뢰할 수 있는 방법을 제공합니다.

- **Technical Details**: POSIX는 주어진 응답에 대한 로지우드(likelihood)의 상대적 변화를 포착하는 것을 목표로 하며, 다양한 개방형 소스 LLM의 프롬프트 감도를 측정하고 비교하는 데 사용됩니다. 이 연구에서는 응답 다양성(response diversity), 분포의 엔트로피(response distribution entropy), 의미적 일관성(semantic coherence), 신뢰도 변동(variance in confidence) 등 네 가지 주요 요인을 고려하여 LLM의 감도를 측정하는 방법론을 제시합니다.

- **Performance Highlights**: 초기 실험 결과, 매개변수 수를 단순히 증가시키거나 지침 조정(instruction tuning)을 하는 것이 프롬프트 감도를 낮추지 않는 반면, 몇 개의 샘플을 추가하면 감도가 현저히 감소하는 경향을 보여줍니다. MCQ(다중 선택 문제) 유형의 작업에서는 프롬프트 템플릿 변경이 가장 높은 감도를 나타내며, 개방형 생성 과제의 경우 패러프레이징(paraphrasing)이 가장 높은 감도를 보이는 것으로 나타났습니다.



### Controlled Generation of Natural Adversarial Documents for Stealthy Retrieval Poisoning (https://arxiv.org/abs/2410.02163)
- **What's New**: 최근 연구에 따르면, 임베딩 유사성(embeding similarity)에 기반한 검색 방법이 더 이상 안전하지 않으며, 공격자가 악의적인 문서를 생성하여 다양한 쿼리에 응답할 때 검색될 수 있다는 것을 보여줍니다. 이는 검색 보강 생성(retrieval-augmented generation, RAG) 또한 영향을 받을 수 있습니다.

- **Technical Details**: 이 논문에서는 저자들이 퍼플렉시티 필터링(perplexity filtering)을 사용하여 HotFlip 기반 기술로 생성된 문서를 쉽게 감지할 수 있음을 보여줍니다. 그들은 새로운 생성 기법을 설계하고 구현하며 평가했습니다. 이 기법은 상대적으로 자연스러운 텍스트를 생성하면서도 임베딩 유사성을 유지하도록 조정된 것입니다. 주어진 조건 하에서, 대체 LLM을 사용하여 '자연스러움' 점수를 계산하여 생성을 유도합니다.

- **Performance Highlights**: 이 새로운 기술은 자연스럽고 감지되지 않는 적대적 문서를 생성할 수 있으며, 기존 방법보다 훨씬 효과적입니다. HotFlip 사용 시 쉽게 감지되는 문서들과 비슷한 독성 효과를 내지만, 기존의 방어 방법으로는 감지할 수 없습니다.



### Matrix and Relative Weak Crossover in Japanese: An Experimental Investigation (https://arxiv.org/abs/2410.02149)
Comments:
          18 pages, 17 figures, To appear in Proceedings of The Society of Modern Grammar (SMOG)'s International Conference on Syntax and Semantics (ICSS) 2024

- **What's New**: 이 논문은 weak crossover 효과가 matrix 절과 relative 절 사이에서 본질적으로 다르다는 증거를 제공합니다. 이 연구는 일본어를 사용하여 영어의 단어순의 혼란을 제거하고, 두 가능성의 차이를 구분합니다.

- **Technical Details**: Fukushima et al. (2024)의 이론을 기반으로 하여, 구문적 구조가 우선성과 단순히 관련이 없다는 것을 보여줍니다. 이 연구는 BVA(bound variable anaphora)의 해석을 통해, R-WCO(relative weak crossover)가 M-WCO(matrix weak crossover)와 동등한 행동 패턴을 보이지 않음을 밝혔습니다.

- **Performance Highlights**: 우리는 장르별로 R-WCO 구성의 BVA 해석이 자주 받아들여진다는 것을 발견했습니다. 이 결과는 weak crossover 효과가 의외로 BVA 해석의 다양성을 감소시키지 않음을 나타냅니다.



### L-CiteEval: Do Long-Context Models Truly Leverage Context for Responding? (https://arxiv.org/abs/2410.02115)
- **What's New**: 이 논문에서는 L-CiteEval이라는 신뢰성 있는 긴 문맥 이해 벤치마크를 소개합니다. 이 벤치마크는 8K에서 48K의 다양한 길이의 문맥을 포함한 11개의 작업을 다루며, 모델이 제공된 문맥에 근거하여 보다 신뢰성 있게 응답하는지를 평가합니다.

- **Technical Details**: L-CiteEval은 긴 문맥(LCM)에서의 이해 및 신뢰성을 평가하는 종합적인 다중 작업 벤치마크로, 5개의 주요 작업 범주와 11개의 긴 문맥 작업을 포함합니다. 본 연구에서는 11개의 최신 LCM을 테스트하고, RAG(Retrieval-Augmented Generation) 접근법이 LCM의 신뢰성을 크게 향상시킬 수 있음을 발견하였습니다.

- **Performance Highlights**: 테스트 결과, 개방형 모델(Open-source models)들이 폐쇄형 모델(Closed-source counterparts)보다 인용 정확도 및 회수에서 상당히 뒤처진 것으로 나타났습니다. 또한 RAG 기법을 활용하면 개방형 모델의 신뢰성을 크게 향상시키지만 생성 품질에는 약간의 감소가 있을 수 있습니다.



### ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvemen (https://arxiv.org/abs/2410.02108)
- **What's New**: 본 논문에서는 고급 모델이나 인간의 감독 없이도 자체적으로 생성된 추론 경로를 통해 대형 언어 모델(LLM)의 추론 능력을 향상시킬 수 있는 방법을 제안합니다. 기존의 방법들이 특정 과제에 너무 특화되어 있는 문제를 해결하기 위해, 과제에 구애받지 않는 일반적인 추론 원칙을 바탕으로 추론 구조를 생성하는 ReGenesis 방법을 도입합니다.

- **Technical Details**: ReGenesis는 '자기 개선' 메커니즘을 활용하여 대형 언어 모델이 추론 경로를 자가 생성하도록 지도합니다. 이 과정은 추상적인 일반 추론 지침을 시작으로, 해당 지침을 과제 특정 지침으로 변환하여 구체적인 추론 구조를 생성하는 단계로 진행됩니다. 최종적으로 이러한 구조를 바탕으로 후보 추론 경로를 생성하고, 이를 바탕으로 LLM의 성능을 향상시키는 것입니다.

- **Performance Highlights**: ReGenesis는 모든 실험에서 기존 방법들에 비해 우수한 성능을 보여주었습니다. 특히 저자는 ReGenesis가 6개의 OOD(Out-of-Domain) 작업에서 평균 6.1%의 성능 향상을 보여주며, 기존 방법들은 평균 4.6%의 성능 감소를 경험한 것을 강조하였습니다.



### Racing Thoughts: Explaining Large Language Model Contextualization Errors (https://arxiv.org/abs/2410.02102)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 맥락화(contextualization) 오류의 원인으로 'LLM Race Conditions Hypothesis'를 제안합니다. 이 가설은 특정 토큰이 이전의 다른 토큰으로부터 정보를 읽어들여야 하는 경우, 이러한 의존성이 올바르게 처리되지 않아 발생하는 오류를 설명합니다.

- **Technical Details**: 저자들은 LLM의 레이어를 통해 발생하는 경쟁 조건(race condition)을 식별하고, 이는 예를 들어 질문 마감 기호가 이전 토큰의 맥락화된 표현을 읽어야 할 때 발생합니다. 결과적으로 이러한 비정상적인 맥락화는 오류를 발생시킬 수 있습니다. 이 연구는 기계적 해석 가능성(mechanistic interpretability) 기법을 적용하여 가설의 실증적 증거를 제시합니다.

- **Performance Highlights**: LLM Race Conditions Hypothesis는 세 가지 서로 다른 데이터셋에서 최신 모델의 맥락화 오류를 설명하며, 이 연구는 LLM의 실패 모드를 명확히 하고, 문제 해결을 위한 여러 개입(intervention) 방법을 제안합니다.



### RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning (https://arxiv.org/abs/2410.02089)
- **What's New**: 해당 논문에서는 사용자 지정 작업을 수행하는 대형 언어 모델(LLMs)의 성과를 개선하기 위해, 코드 생성(code synthesis) 영역에서 실행 피드백(execution feedback)을 효과적으로 활용하는 강화 학습 방법론을 제안합니다. 이 접근법은 기존의 독립 샘플링(independent sampling)과 비교할 때 반복적으로 코드 개선을 도모할 수 있는 새로운 프레임워크를 제공합니다.

- **Technical Details**: 코드 생성을 다중 단계의 대화(interactive conversation) 과정으로 구조화하여, LLM이自然语言 문제 설명에 대한 코드 솔루션을 반복적으로 생성하도록 하고, 각 솔루션의 실행 결과를 자동으로 평가합니다. 이 평가 결과는 다음 코드 시도를 위한 추가 컨텍스트로 제공되며, 이는 end-to-end 최적화(end-to-end optimization) 및 강화 학습 알고리즘을 통한 최대 보상 신호(maximize reward signal)로 이어집니다. 특히, Proximal Policy Optimization (PPO) 알고리즘을 사용하여 정책을 최적화합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법론은 CodeContests 벤치마크에서 이전 최고 성능을 넘어서며, 샘플의 수를 10배 이상 줄이는 데 성공했습니다. 이는 대형 및 소형 모델 모두에서 관찰되며, HumanEval+ 및 MBPP+와 같은 알려진 코드 생성 벤치마크에서 일반화된 성과를 보여줍니다.



### Improving Autonomous AI Agents with Reflective Tree Search and Self-Learning (https://arxiv.org/abs/2410.02052)
- **What's New**: 이번 논문에서는 Reflective Monte Carlo Tree Search (R-MCTS)라는 새로운 알고리즘을 소개하여, GPT-4o 기반 AI 에이전트가 복잡한 멀티 스텝 의사결정 작업을 신속하게 탐색할 수 있도록 향상시키고 있습니다.

- **Technical Details**: R-MCTS는 전통적인 MCTS를 확장하여, 1) 과거 상호작용으로부터 학습하고 검색 효율성을 동적으로 개선할 수 있는 대조적 반영(contrastive reflection) 기법과 2) 신뢰할 수 있는 상태 평가를 위한 다중 에이전트 토론(multi-agent debate) 기법을 통합합니다.

- **Performance Highlights**: R-MCTS를 통해 훈련된 GPT-4o 기반 에이전트는 VisualWebArena 벤치마크에서 이전의 최첨단 모델과 비교하여 6%에서 30%까지 성능이 향상되었으며, fine-tuning 과정을 통해 97%의 성능을 유지하면서도 테스트 시간 동안 계산 비용을 4배 줄였습니다.



### Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions (https://arxiv.org/abs/2410.02028)
Comments:
          EMNLP2024 Main

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 분류(task classification) 작업을 위한 정밀 조정(fine-tuning) 방법을 연구하기 위한 새로운 프레임워크를 제안합니다. 특히, edit intent classification (EIC)라는 복잡한 분류 과제를 통해 이 프레임워크의 적용을 보여줍니다.

- **Technical Details**: 본 연구는 LLMs의 classification 능력을 체계적으로 평가하기 위해 현존하는 네 가지 접근법(하나는 생성 기반(generation-based)이고 세 가지는 인코딩 기반(encoding-based)입니다)을 탐색합니다. e.g., LLMs를 embedding 모델로 활용하는 방안, 혹은 instruction tuning 기법 등을 포함합니다.

- **Performance Highlights**: EIC에 대한 extensive 실험을 통해 encoding 기반 접근방식을 사용하여 fine-tuning된 LLMs가 뛰어난 분류 성능을 보여 주었으며, state-of-the-art (SOTA) 결과를 달성했습니다. 또한, 1,780개의 과학 문서 수정과 94,482개의 레이블이 부여된 편집으로 구성된 새로운 대규모 dataset인 Re3-Sci2.0을 생성하여 인간의 편집 행동을 상세히 분석할 수 있는 기회를 제공합니다.



### How Reliable Is Human Feedback For Aligning Large Language Models? (https://arxiv.org/abs/2410.01957)
- **What's New**: 이번 연구에서는 인공지능 모델 정렬(alignment)에 있어서 인간 피드백의 질적 불확실성을 분석하고, 이의 영향을 정량적으로 평가함으로써 기존의 연구에서 간과된 문제를 다루고 있습니다. 특히, 25% 이상의 데이터가 신뢰 기준을 충족하지 못함을 밝히며, 이는 데이터 품질에 대한 심각한 우려를 나타냅니다.

- **Technical Details**: 본 연구는 골드 보상 모델(gold reward models)이라는 위원회를 통해 인간 피드백의 신뢰성을 평가합니다. 이러한 모델은 다수의 독립적으로 훈련된 모델로 구성되어 있으며, 이들의 집합적 판단을 통해 피드백의 신뢰성을 체계적으로 분류합니다. 또한, 인간 피드백의 불확실성의 주요 원인으로는 잘못된 레이블링(mis-labeling), 주관적 선호(subjective preferences), 평가 기준의 차이 등이 있으며, 이를 해결하기 위한 자동 데이터 정리 방법인 Source-Aware Cleaning을 제안합니다.

- **Performance Highlights**: HH-Clean 데이터셋을 사용하여 훈련된 모델은 원본 데이터셋을 사용한 모델에 비해 약 77%의 우승-무승부 승률을 기록하였으며, 이는 기존 데이터 정리 방법보다 월등한 성능 향상을 보여줍니다. 연구자들은 이 데이터셋을 공개하여 향후 더욱 신뢰할 수 있는 LLM 정렬 평가를 지원할 것입니다.



### Generate then Refine: Data Augmentation for Zero-shot Intent Detection (https://arxiv.org/abs/2410.01953)
- **What's New**: 이 논문에서는 zero-resource domain에 대한 intent detection을 위한 데이터 증강(data augmentation) 방법을 제안합니다. 기존의 데이터 증강 방법은 적은 수의 라벨이 있는 예시가 필요하지만, 함수 처리(intent)에 대한 카테고리가 많을 경우 비효율적입니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다: 첫 번째로, open-source large language model(LLM)을 사용하여 intent 레이블에 대한 발화(utterances)를 생성합니다. 두 번째 단계에서는 Refiner라는 작은 sequence-to-sequence 모델을 개발하여 생성된 발화를 개선합니다. Refiner는 seen domain에서 조정되고, unseen domain에 적용됩니다.

- **Performance Highlights**: Refiner는 zero-shot LLM 기준선에 비해 unseen domain의 데이터 유용성과 다양성을 크게 개선함을 보여줍니다. 최종 결과는 generative LLM의 zero-shot 사용과 작은 모델의 조합이 intent detection에 대한 고품질 데이터를 제공할 수 있음을 나타냅니다.



### TypedThinker: Typed Thinking Improves Large Language Model Reasoning (https://arxiv.org/abs/2410.01952)
Comments:
          work in process

- **What's New**: 본 논문에서는 TypedThinker라는 새로운 프레임워크를 제안하여 대규모 언어 모델(LLMs)의 문제 해결 능력을 향상시킵니다. 이는 다양한 추론 유형(추론적, 귀납적, 가설적, 유추적)을 통합하여 다각적인 사고를 촉진합니다.

- **Technical Details**: TypedThinker는 주어진 문제에 적합한 추론 유형을 선택하고 이를 효과적으로 구현하는 두 가지 주요 과제를 해결합니다. 이를 위해 메타-사고자(meta-thinker)와 추론자(reasoner)를 사용하여 추론의 유형을 선택하고 실행하며, 경험을 저장하고 검색하는 명시적 기억(explicit memory)을 유지합니다. 또한, 자기 학습(self-training)을 통해 성공적인 경험에 기반한 암묵적 정책을 학습합니다.

- **Performance Highlights**: 실험 결과는 TypedThinker가 Mistral 7B에서 3.4%, LLaMA3 8B에서 16.7%의 정확도 향상을 보여주며, 새로운 벤치마크에 대해 효과적인 일반화를 이루었습니다. 또한, 강력한 모델인 GPT-4o의 추론 능력을 추가적으로 향상시킬 수 있음을 입증했습니다.



### SciPrompt: Knowledge-augmented Prompting for Fine-grained Categorization of Scientific Topics (https://arxiv.org/abs/2410.01946)
Comments:
          EMNLP 2024 Main

- **What's New**: 이 논문은 SciPrompt라는 프레임워크를 소개하여 과학 텍스트 분류 작업에서 적은 리소스 환경에서도 효과적으로 사용할 수 있는 프롬프트 기반 미세 조정 방법을 제안합니다. 이 방법은 도메인 지식을 활용하여 과학 관련 용어를 자동으로 획득하고, 이 용어를 사용하여 언어 모델의 예측 성능을 향상시킵니다.

- **Technical Details**: SciPrompt는 세 가지 주요 단계를 포함합니다: 1) 과학 용어 수집, 2) 라벨 용어 필터링, 3) 과학 주제 예측. 이 프레임워크는 masked language modeling (MLM) 기술을 활용하며, 외부 지식 기반에서 도메인에 적합한 과학 구문을 검색하여 기존의 검증자(verbalizer)를 확장합니다. 이 과정에서 의미적 유사성을 평가하기 위해 NLI (Natural Language Inference) 모델을 이용해 라벨 용어와 클래스 레이블 간의 상관관계를 측정합니다.

- **Performance Highlights**: SciPrompt는 네 개의 과학 데이터셋을 통해 평가되었으며, 적은 리소스 환경에서 대부분의 기존 최첨단 방법을 초과하는 성능을 보였습니다. 특히, 세부적인 과학 주제의 분류에서 두드러진 성과를 나타냈습니다.



### CALF: Benchmarking Evaluation of LFQA Using Chinese Examinations (https://arxiv.org/abs/2410.01945)
- **What's New**: 본 논문에서는 Long-Form Question Answering (LFQA)의 효과적이고 효율적인 평가를 위한 첫 번째 표준 벤치마크인 CALF(Chinese exAmination for LFQA Evaluation)를 제안합니다. 이는 중국의 시험 질문을 기반으로 하여 영어로 번역된 예제를 포함하고 있습니다.

- **Technical Details**: CALF 벤치마크는 최대 1476개의 지식 집약적이고 미세한 응답을 포함하고 있으며, 7개의 전통적인 평가 메트릭, 3개의 프롬프트 기반 메트릭, 3개의 훈련된 평가 메트릭에 대해 광범위한 실험을 수행했습니다.

- **Performance Highlights**: 현재 자동 평가 메트릭은 인간 수준의 성과를 보여주지 못하며, 긴 응답에 포함된 밀집 정보를 잘 포착하지 못하는 것을 분석하였습니다. 이에 대한 자세한 분석도 제공하여 LFQA 평가 시스템의 발전을 위한 통찰을 제공합니다.



### Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos (https://arxiv.org/abs/2410.02763)
Comments:
          Project Page: this https URL

- **What's New**: 최근 대형 멀티모달 모델(large multimodal models, LMMs)이 짧은 비디오 이해의 주요 문제를 해결했다고 평가받고 있지만, 실제로는 여러 기본적인 추론 능력이 부족함을 보여줍니다. 이를 개선하기 위해 1000개의 짧고 자연스러운 비디오-자막 쌍을 포함한 새로운 평가 기준인 Vinoground를 소개합니다.

- **Technical Details**: Vinoground는 다양한 동작과 객체 변환의 시간적 차이를 구별하는 능력을 평가합니다. 기존의 모델들이 이러한 능력을 정확히 수행하지 못함을 나타내며, 특히 'single-frame bias' 문제를 해결하기 위해 시간적 반사실(counterfactual) 쌍으로 구성된 데이터를 사용합니다.

- **Performance Highlights**: 대부분의 현대 LMM 모델들은 짧은 비디오 이해에 있어 형편없는 성능을 보이며, 최고 모델인 GPT-4o는 텍스트 및 비디오 점수에서 50% 정도에 불과해 사람의 평균 기준인 90%에 비해 큰 성격 차이를 보입니다. 많은 모델들이 비디오 점수에서 무작위 확률 수준을 기록하였습니다.



### Training Language Models on Synthetic Edit Sequences Improves Code Synthesis (https://arxiv.org/abs/2410.02749)
- **What's New**: 이 논문에서는 코드 수정을 위한 데이터 생성을 위한 새로운 알고리즘인 LintSeq를 소개합니다. 이는 기존의 코드에서 오류 없는 수정 시퀀스를 생성하여, 높은 품질의 고유 코드 편집 데이터의 부족 문제를 해결하고자 합니다.

- **Technical Details**: LintSeq 알고리즘은 소스 코드를 입력으로 받아 정적 프로그램 검증기를 통해 오류가 없는 프로그램 상태의 시퀀스를 샘플링합니다. 이후 Unix diff 연산자를 사용하여 각 시퀀스를 역으로 계산하고 코드 수정 시퀀스를 출력합니다. 이 과정은 두 단계로 나뉘며, 첫째는 back sampling 단계, 둘째는 forward edit computation 단계입니다.

- **Performance Highlights**: 15억 매개변수를 가진 작은 LLM이 LintSeq로 수정된 데이터를 학습한 결과, HumanEval pass@50에서 GPT-4와 경쟁할 수 있으며, 기준 데이터셋으로 학습된 모델보다 +20% 더 나은 성능을 보였습니다. 특히, 작은 LMs는 코드 편집 시퀀스에 대한 fine-tuning을 통해 최신 수준의 성능을 달성했습니다.



### AVG-LLaVA: A Multimodal Large Model with Adaptive Visual Granularity (https://arxiv.org/abs/2410.02745)
Comments:
          Preprint

- **What's New**: AVG-LLaVA(Adaptive Visual Granularity LLaVA)는 입력 이미지와 지시에 따라 적절한 visual granularity(시각적 세분성)를 선택할 수 있는 LMM(대형 다중모드 모델)입니다. 이를 통해 visual token(시각적 토큰)의 수를 줄이고 추론 속도를 향상시키며 모델 성능을 개선합니다.

- **Technical Details**: AVG-LLaVA는 시각적 세분성 스케일러와 시각적 세분성 라우터라는 두 가지 모듈을 포함하고 있습니다. 시각적 세분성 스케일러는 visual tokens에 대해 여러 번의 pooling layer(풀링 레이어)를 진행하여 서로 다른 세분성의 visual tokens를 얻습니다. 시각적 세분성 라우터는 입력된 다중 세분성 visual feature와 text feature를 기반으로 적절한 visual granularity를 선택합니다. RGLF(Ranking Granularity to align LMM Feedback)라는 새로운 훈련 패러다임도 도입되어 라우터의 예측을 LMM의 선호와 일치시키도록 돕습니다.

- **Performance Highlights**: AVG-LLaVA는 11개의 벤치마크에서 우수한 성능을 달성하였으며, visual token 수를 85.3% 줄이고 추론 속도를 2.53배 증가시키는 등 성능을 크게 개선했습니다.



### DivScene: Benchmarking LVLMs for Object Navigation with Diverse Scenes and Objects (https://arxiv.org/abs/2410.02730)
Comments:
          Work in Progress

- **What's New**: 본 논문에서는 다양한 대상 객체를 탐색할 수 있는 내비게이션 에이전트 구축을 위한 새로운 작업을 연구합니다. 이를 위해 4,614개의 씬으로 구성된 대규모 씬 데이터셋, DivScene을 소개합니다.

- **Technical Details**: NatVLM(Navigational Chain-of-Thought VLM)이라고 불리는 내비게이션 에이전트를 구축하였으며, 이를 위해 Large Vision Language Model(사례로 Idefics 2)과 모방 학습(imitation learning)을 활용했습니다. 이러한 접근 방식은 기존의 고인물 학습과의 도메인 갭(domain gap)을 줄이는 데 기여합니다.

- **Performance Highlights**: 제안한 NatVLM 에이전트는 BFS (너비 우선 탐색) 계획자를 통해 구축된 최단 경로에서의 모방 학습으로 훈련되었으며, GPT-4o를 초과하는 20% 이상의 성공률을 달성했습니다.



### Large Language Models as Markov Chains (https://arxiv.org/abs/2410.02724)
Comments:
          49 pages, 17 figures

- **What's New**: 이 논문에서는 널리 알려진 오토 회귀 언어 모델(LLMs)과 마르코프 체인(markov chains) 간의 equivalence(동등성)을 수립하여 LLM의 추론 능력을 명확하게 설명하고자 합니다. 기존 LLM 성능의 기원을 해명하기 위한 다양한 연구가 있었으나, 본 연구는 이를 좀 더 접근 가능하게 만듭니다.

- **Technical Details**: 연구의 핵심은 크기 |Ω|의 유한 상태 공간에서 정의된 마르코프 체인으로서 LLM을 해석하는 것입니다. 저자들은 LLM의 transition matrix(전이 행렬)를 분석하고, stationary distribution(정상 분포)의 존재성과 유일성을 증명합니다. 또한 vocabularies(어휘)와 context window(맥락 창) 크기 및 model temperature(모델 온도)에 따른 수렴 속도를 제시합니다.

- **Performance Highlights**: 실험 결과, 2023-2024에 발표된 최신 LLM들이 이론적 결과에 의해 예측된 in-context scaling laws를 준수하는 것을 보였습니다. 특히, LLM들은 최소 변별 최적 빈도주의 접근법보다 마르코프 체인 학습에서 더 뛰어난 성능을 발휘하는 것으로 나타났습니다.



### Video Instruction Tuning With Synthetic Data (https://arxiv.org/abs/2410.02713)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서 제안하는 LLaVA-Video-178K 데이터셋은 비디오 지침 따르기를 위한 고품질 합성 데이터셋으로, 다양한 비디오 작업을 위한 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: LLaVA-Video-178K는 178,510개의 비디오로 구성되어 있으며, 다양한 질문 유형을 제공하는 태스크를 포함합니다. 주요 속성으로는 동적 비디오 소스의 선택, 자세한 캡션 생성 파이프라인 및 다양한 질문-답변 쌍 생성을 위한 GPT-4o 사용이 있습니다.

- **Performance Highlights**: LLaVA-Video는 다양한 비디오 벤치마크에서 뛰어난 성능을 발휘하며, 기존 단일 프레임 기반 학습이 아닌 다중 프레임 기반 접근 방식이 효과적임을 입증했습니다.



### LLaVA-Critic: Learning to Evaluate Multimodal Models (https://arxiv.org/abs/2410.02712)
Comments:
          Project Page: this https URL

- **What's New**: LLaVA-Critic는 최초의 오픈소스 대규모 다중 모달 모델로, 다양한 다중 모달 작업의 성과를 평가하는 일반 평가자로 설계되었습니다. 이 모델은 고품질의 비평가 지침을 따르는 데이터셋을 사용하여 훈련되었으며, 효과적인 평가 점수를 제공하는 능력이 입증되었습니다.

- **Technical Details**: LLaVA-Critic는 (1) LMM-as-a-Judge와 (2) Preference Learning의 두 가지 주요 시나리오에서 실험이 수행되었습니다. LMM-as-a-Judge 시나리오에서는 LLaVA-Critic가 신뢰할 수 있는 평가 점수를 제공하며, 여러 평가 벤치마크에서 GPT 모델들과 유사한 성능을 보입니다. Preference Learning에서는 선호 학습을 위한 보상 신호를 생성하여 모델 정렬 능력을 향상시킵니다.

- **Performance Highlights**: LLaVA-Critic는 상업용 GPT 모델과 높은 상관관계를 보이며, 자원 제한 환경에서도 모델 개발자에게 비용 효과적인 대안을 제공합니다. 또한, AI 생성 피드백을 통한 선호 학습에서 LLaVA-RLHF 모델보다 우수한 성능을 나타냈습니다.



### FAN: Fourier Analysis Networks (https://arxiv.org/abs/2410.02675)
- **What's New**: 본 논문에서는 기존의 신경망 아키텍처가 주기성을 효과적으로 모델링하고 추론하는 데 한계를 보인다는 문제를 제기하며, Fourier Analysis에 기반한 새로운 네트워크 아키텍처인 FAN(Fourier Analysis Network)을 제안합니다.

- **Technical Details**: FAN은 Fourier Series를 활용하여 주기적 데이터를 자연스럽게 통합하는 구조를 갖추고 있습니다. 이 네트워크는 MLP와 같은 기존 모델을 대체할 수 있으며, 훨씬 적은 파라미터와 FLOPs를 요구합니다. 이론적으로 주기성이 명시적으로 모델링 되도록 설계되어 주기적 패턴의 표현과 예측에서 더 높은 정확성을 제공합니다.

- **Performance Highlights**: FAN은 기존 MLP, KAN, Transformer와 비교하여 기본 및 복잡한 주기 함수를 더 효과적으로 모델링할 수 있으며, 실제 작업에서도 상징적 공식 표현, 시계열 예측, 언어 모델링 등 다양한 분야에서 우수한 성능을 입증하였습니다.



### Immunogenicity Prediction with Dual Attention Enables Vaccine Target Selection (https://arxiv.org/abs/2410.02647)
Comments:
          18 pages, 11 tables, 5 figures

- **What's New**: ProVaccine이라는 새로운 심층 학습 솔루션을 소개하여, 단백질 서열 및 구조의 사전 훈련된 잠재 벡터 표현을 통합하고, 면역원성(IMMUNOGENICITY) 예측의 정확성을 높입니다.

- **Technical Details**: ProVaccine은 이중 주의 메커니즘(dual attention mechanism)을 활용하여 단백질의 다양한 서열, 구조 및 물리화학적 특성을 다중 모드 인코딩(multi-modal encoding)합니다. 이는 아미노산(AA) 서열 정보와 함께 구조 토큰을 사용하여 단백질의 복잡한 관계를 포착합니다.

- **Performance Highlights**: ProVaccine은 9,500개 이상의 항원 서열을 포함하는 최신 데이터셋 Immuno에 대해 뛰어난 성능을 보여주며 기존 방법론에 비해 예측 정확성과 일반화 능력이 크게 향상되었습니다.



### NL-Eye: Abductive NLI for Images (https://arxiv.org/abs/2410.02613)
- **What's New**: NL-Eye라는 새로운 벤치마크를 소개하여 VLM(Visual Language Model)의 시각적 외적 추론(abductive reasoning) 능력을 평가하고자 함.

- **Technical Details**: NL-Eye는 350개의 트리플릿 예제를 포함하며, 각 사례는 전제 이미지와 가설 이미지들로 구성됨. VLM은 주어진 전제 이미지에 따라 가설 이미지의 그럴듯함(plausibility)을 평가해야 하며 이는 물리적, 논리적, 감정적, 기능적, 문화적, 사회적 범주로 나뉨. 이 과정에서 생성된 이미지는 텍스트에서 이미지로 변환하는 모델을 통해 얻어짐.

- **Performance Highlights**: NL-Eye에서 VLM은 무작위 기준 수준의 성과를 보이며, 인간들은 85%의 정확도로 더 그럴듯한 가설을 선택하고 94%에서 유효한 설명을 제공. 하지만 VLM은 정확한 설명을 제공하는 데 있어 50% 이상 실패하며, 이는 현대 VLM의 외적 추론 능력에 큰 결함을 나타냄.



### Convolutional Variational Autoencoders for Spectrogram Compression in Automatic Speech Recognition (https://arxiv.org/abs/2410.02560)
Comments:
          Theory and Practice of Natural Computing 9th International Conference, TPNC 2020, Taoyuan, Taiwan, 2020, Proceedings 9

- **What's New**: 본 논문에서는 Convolutional Variational Autoencoders (VAE)를 기반으로 한 압축 스펙트로그램 표현 생성의 대안 방법을 제시하고 있습니다. 기존 Mel-frequency Cepstral Coefficients (MFCC)와 비교해 ASR (Automatic Speech Recognition) 작업에서 더 나은 결과를 보여주는 오디오 피처를 제안합니다.

- **Technical Details**: 제안된 VAE 모델은 LibriSpeech 데이터셋의 하위 샘플을 기반으로 단기 오디오 스펙트로그램 조각(25 ms)을 13차원의 임베딩으로부터 재구성하는 방식으로 훈련되었습니다. 40차원(300 ms)의 임베딩을 위한 훈련된 모델을 사용하여 GoogleSpeechCommands 데이터셋의 음성 명령에 대한 피처를 생성했습니다.

- **Performance Highlights**: 생성된 피처를 사용하여 ASR 시스템을 구축하고, MFCC 피처를 사용하는 모델과 비교하여 성능을 평가했습니다. 이 연구는 특히 VAE 기반 피처가 기존의 MFCC와 비교하여 처리 속도와 성능 면에서 이점을 제공함을 보여줍니다.



### ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration (https://arxiv.org/abs/2410.02551)
- **What's New**: ColaCare는 Large Language Models (LLMs) 기반의 다중 에이전트 협업을 통해 전자 건강 기록 (EHR) 모델링을 향상시키는 프레임워크입니다.

- **Technical Details**: ColaCare는 DoctorAgent와 MetaAgent의 두 가지 유형의 에이전트를 사용하여 환자 데이터를 공동 분석합니다. Expert models는 수치 EHR 데이터에서 예측을 처리하고 생성하며, LLM 에이전트는 협업 상담 프레임워크 내에서 추론 참조 및 의사 결정 보고서를 생성합니다. 또한 Merck Manual of Diagnosis and Therapy (MSD) 의료 지침을 검색 강화 생성 (RAG) 모듈에 통합하여 권위 있는 증거 지원을 제공합니다.

- **Performance Highlights**: 네 개의 EHR 데이터셋에 대한 광범위한 실험을 통해 ColaCare는 사망률 예측 작업에서 우수한 성능을 보였으며, 임상 의사 결정 지원 시스템을 혁신하고 개인화된 정밀 의학을 발전시킬 잠재력을 강조합니다.



### Can Large Language Models Grasp Legal Theories? Enhance Legal Reasoning with Insights from Multi-Agent Collaboration (https://arxiv.org/abs/2410.02507)
- **What's New**: 본 연구에서는 법 이론 및 복잡한 법적 추론 능력을 평가하기 위해 Confusing Charge Prediction이라는 도전적인 과제를 도입하고, 이를 해결하기 위한 새로운 프레임워크인 Multi-Agent framework for improving complex Legal Reasoning capability (MALR)를 제안합니다.

- **Technical Details**: MALR에서는 비모수적 학습(non-parametric learning)을 적용하여 LLM들이 복잡한 법적 작업을 자동으로 분해하고 법 규칙에서 통찰(insights)을 추출하도록 장려합니다. 이 방식은 LLM들이 단일 법칙에 의존하지 않고, 경험과 피드백을 통해 자기 반성을 통해 학습할 수 있도록 합니다.

- **Performance Highlights**: 여러 실제 사례 데이터셋에서 실시된 실험을 통해 제안된 프레임워크가 실용적인 시나리오에서 복잡한 추론 문제를 효과적으로 해결할 수 있음을 입증했습니다. 이 연구는 법률 분야에서 LLM의 신뢰성 있는 응용 가능성을 향상시키는 데 기여합니다.



### Dynamic Gradient Alignment for Online Data Mixing (https://arxiv.org/abs/2410.02498)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 훈련 데이터 혼합물을 최적화하는 방법으로 Dynamic Gradient Alignment (DGA)라는 새로운 알고리즘을 제안합니다. DGA는 특정 작업에 대한 모델의 그래디언트와 잘 정렬되도록 사전 훈련 데이터 혼합물을 동적으로 추정합니다.

- **Technical Details**: DGA는 온라인 그래디언트 정렬 알고리즘으로, 각 훈련 단계에서 도메인 가중치를 업데이트합니다. DGA는 오늘날의 모델 상태에 따라 도메인 가중치를 동적으로 조정하며, 이를 통해 모델 훈련 중 과적합(overfitting)을 방지하고 대상 작업에 가장 유익한 도메인에 우선순위를 두는 방식으로 작동합니다. 또한, DGA는 수천 개의 미세 도메인(262k domains)으로의 확장을 처리할 수 있는 새로운 분포 재가중치 메커니즘을 도입합니다.

- **Performance Highlights**: DGA는 제한된 자원이 있는 두 가지 주요 시나리오에서 중요 샘플링(importance sampling) 방법에 비해 상당한 개선을 보여줍니다: (1) 사전 훈련 세트가 작고 중요 샘플링이 과적합되는 경우, (2) 특화된 데이터가 부족하여 중요 샘플링이 제한된 데이터에 갇히는 경우.



### DTVLT: A Multi-modal Diverse Text Benchmark for Visual Language Tracking Based on LLM (https://arxiv.org/abs/2410.02492)
Comments:
          Preprint, Under Review

- **What's New**: 이번 연구에서는 다양한 텍스트를 생성할 수 있는 대형 언어 모델(LLM)을 활용하여, 시각 언어 추적(Visual Language Tracking, VLT)을 위한 새로운 벤치마크인 DTVLT를 제안합니다. 이 벤치마크는 전통적인 단일 객체 추적(Single Object Tracking, SOT) 작업의 범위를 확장하여 비디오 이해 애플리케이션을 포함하도록 합니다.

- **Technical Details**: DTVLT는 다섯 개의 저명한 VLT 및 SOT 벤치마크를 기반으로 하며, 짧은 기간 추적, 장기 추적, 전역 인스턴스 추적 등 세 가지 하위 작업으로 구성됩니다. 벤치마크에 포함된 네 가지 서로 다른 세부정보 수준의 텍스트는 의미론적 정보의 범위와 밀도를 고려하여 생성됩니다. 텍스트 생성을 위한 DTLLM-VLT 메소드를 활용합니다.

- **Performance Highlights**: 실험 분석을 통해, 다양한 텍스트가 추적 성능에 미치는 영향을 평가하였으며, 기존 알고리즘의 성능 병목 현상을 파악하여 VLT와 비디오 이해 연구의 발전을 지원할 수 있는 기초 자료를 제공합니다.



### MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation (https://arxiv.org/abs/2410.02458)
Comments:
          Submitted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 연구는 교육된 대형 언어 모델(LLMs)로부터 얻은 frozen transformer 블록을 Vision Transformer (ViT) 기반 모델의 인코더에 통합하여 의료 이미지 분할 성능을 크게 향상시켰습니다. 이 과정에서 Hybrid Attention Mechanism과 Multi-Scale Fusion Block을 도입해 다양한 스케일에서의 특징 집계를 통해 성능을 대폭 개선했습니다.

- **Technical Details**: 연구에서는 여러 공개 LLM 모델인 Meta-Llama-3.1-8B, Gemma-2-9B 등에서 사전 훈련된 가중치를 사용하였고, 이를 ViT의 인코더와 디코더 사이에 통합했습니다. Hybrid Attention Mechanism을 통해 전역 및 대응 주의(a ttention) 학습을 조화롭게 이루어지도록 설정하였으며, 다양한 의료 이미징 모달리티에서 효과를 검증하기 위해 ablation study를 실시했습니다.

- **Performance Highlights**: 이 모델의 Segmentation 성능은 평균 Dice score이 0.74에서 0.79로 증가하였고, 정밀도와 Jaccard Index에서도 현저한 개선을 보였습니다. 이러한 결과는 LLM 기반 transformer의 효과성을 입증하며, 의료 이미지 분석에서 모델의 정확성과 강건성을 크게 향상할 수 있는 가능성을 보여줍니다.



### IoT-LLM: Enhancing Real-World IoT Task Reasoning with Large Language Models (https://arxiv.org/abs/2410.02429)
Comments:
          21 pages, 10 figures, submitted to ICLR 2025 Conference

- **What's New**: 최근 대형 언어 모델(LLMs)은 텍스트 및 시각 도메인에서 뛰어난 성능을 보여주고 있으나, 물리 법칙을 위반하는 결과를 생성하는 우리의 연구는 이러한 모델의 물리 세계 이해의 간극을 해소하고자 IoT 센서 데이터를 활용한 개념을 도입했습니다.

- **Technical Details**: 우리는 LLM의 능력을 향상시키기 위해 IoT 데이터를 LLM에 적합한 형식으로 전처리하고, 상식 지식을 활성화하는 체인 오브 사고 프롬프트를 사용하며, IoT지향 검색 보강 생성(Retrieval-Augmented Generation) 방식을 통합하여 IoT-LLM이라는 통합 프레임워크를 제안합니다.

- **Performance Highlights**: 실험 결과, IoT-LLM은 여러 태스크에서 평균 65% 증가된 성능을 기록했으며, 이는 LLM이 IoT 데이터와 물리 법칙을 더 잘 이해하고 적용할 수 있도록 돕는 효과를 보여줍니다.



### LLM-Pilot: Characterize and Optimize Performance of your LLM Inference Services (https://arxiv.org/abs/2410.02425)
Comments:
          Accepted to the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '24)

- **What's New**: LLM-Pilot는 LLM 추론 서비스의 성능을 특성화하고 예측하는 최초의 시스템으로, 다양한 GPU에서 현실적인 작업 하에서 LLM 추론 서비스를 벤치마킹합니다.

- **Technical Details**: LLM-Pilot는 두 가지 주요 구성 요소로 이루어져 있습니다: 성능 특성화 도구와 GPU 추천 도구입니다. 성능 특성화 도구는 클러스터 관리자가 오프라인에서 다양한 LLM 추론 서비스의 성능을 벤치마킹할 수 있게 합니다. GPU 추천 도구는 사용자가 성능 요구 사항을 충족하면서 가장 비용 효율적인 방식으로 새로운 LLM을 배포할 수 있도록 온라인 추천을 제공합니다.

- **Performance Highlights**: LLM-Pilot는 성능 요구 사항을 33% 더 자주 충족시키고, 평균 60%의 비용 절감을 가능하게 합니다.



### Parameter Competition Balancing for Model Merging (https://arxiv.org/abs/2410.02396)
Comments:
          Accepted by NeurIPS2024

- **What's New**: 본 논문은 PCB-Merging (Parameter Competition Balancing)이라는 새로운 기법을 소개하고 있으며, 이 기법은 여러 모델의 파라미터를 효과적으로 통합하는 경량화된 기술입니다. 기존의 모델 병합 기법이 작업 간의 잠재적 충돌 및 복잡한 상관관계를 다루는 데 어려움이 있었던 것에 반해, PCB-Merging은 파라미터 경쟁을 관리하여 더 나은 성능 향상을 도모합니다.

- **Technical Details**: PCB-Merging은 단계적으로 두 가지 균형 조정 방식을 활용합니다: intra-balancing과 inter-balancing. Intra-balancing은 개별 작업 내에서 파라미터의 중요성을 평가하고, inter-balancing은 작업 간의 파라미터 유사성을 평가합니다. 중요도가 낮은 파라미터는 제거되고, 남은 파라미터는 최종 병합 모델을 형성하기 위해 재조정됩니다. 이러한 방식으로 모델의 성능을 극대화합니다.

- **Performance Highlights**: 다양한 실험에서 PCB-Merging 기법이 기존의 모델 병합 방법들보다 우수한 성능 향상을 보여 주었으며, 교차 작업(cross-task), 교차 도메인(cross-domain), 교차 훈련 구성을 포함한 다양한 병합 시나리오에서 제출되었습니다. 특히, T5-base 모델을 사용했을 때 4.3%의 성능 향상을 달성했습니다.



### Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models (https://arxiv.org/abs/2410.02298)
Comments:
          10 pages, 5 figures

- **What's New**: 최근 연구에서는 대용량 언어 모델(LLMs)에 대한 Jailbreak 공격 차단을 위한 새로운 기법인 'Jailbreak Antidote'를 제시했습니다. 이 방법은 모델의 내부 상태를 실시간으로 조정하여 안전과 유용성 간의 균형을 유지하도록 설계되었습니다.

- **Technical Details**: 'Jailbreak Antidote'는 LLM의 내부 상태의 약 5%만 조정하여 안전 선호도를 실시간으로 조정합니다. 이는 기존의 방어 전략들과는 달리 추가적인 컴퓨팅 오버헤드나 지연 없이 진행됩니다. 모델의 내부 표현은 특정한 구성 요소에 집중되어 있으며, 이를 조정함으로써 안전 정보를 효과적으로 조절할 수 있습니다.

- **Performance Highlights**: Jailbreak Antidote은 2억부터 720억 매개변수까지의 9개의 LLM을 대상으로 10가지 Jailbreak 공격 방법과 6가지 방어 전략과 비교한 결과, 기존 방어 방법들보다 안전성과 유용성 측면에서 상당한 성능 향상을 입증했습니다. 이는 LLM 운영 중 실시간 안전 조정을 가능하게 합니다.



### Efficient Second-Order Neural Network Optimization via Adaptive Trust Region Methods (https://arxiv.org/abs/2410.02293)
- **What's New**: SOAA는 대규모 딥러닝 모델에 적합하도록 computation complexity를 줄이며, adaptive trust region 기능을 통해 빠르고 안정적인 수렴을 보장하는 새로운 최적화 알고리즘입니다.

- **Technical Details**: SOAA는 Fisher 정보 행렬을 대각선 표현으로 근사하여 계산 복잡도를 O(n²)에서 O(n)으로 줄이고, adaptive trust-region 메커니즘을 결합하여 관측된 손실 감소에 기반하여 신뢰 영역의 크기를 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과, SOAA는 Adam과 같은 1차 최적화 알고리즘에 비해 빠르고 안정적인 수렴을 달성했습니다. 비록 메모리 요구사항이 더 크지만, 대규모 딥러닝 작업에서의 성능 이점으로 사용 가능성이 높습니다.



### Structural-Entropy-Based Sample Selection for Efficient and Effective Learning (https://arxiv.org/abs/2410.02268)
Comments:
          Submitted to ICLR 2025

- **What's New**: 이 논문은 샘플 선택의 효율성과 효과성을 개선하기 위해 텍스트의 정보와 대표성을 갖춘 샘플을 선택하는 새로운 방법을 제안합니다. 기존의 방법은 주로 로컬 정보에 의존하였으나, 본 연구는 전역 정보와 로컬 정보를 결합하여 샘플을 선택하는 방법을 개발합니다.

- **Technical Details**: 본 연구에서는 샘플을 그래프로 모델링하여 노드와 엣지를 사용하여 샘플 간의 유사성을 표현합니다. 전역 정보를 정량화하기 위해 구조적 엔트로피(structural entropy)를 활용하며, 이를 샤플리 값(Shapley value)을 사용해 개별 노드로 분해합니다. 새로 제안한 샘플 선택 방법인 SES는 샘플들의 kNN-그래프를 구성하고, 구조적 엔트로피와 학습 난이도를 결합하여 샘플의 중요도를 측정합니다.

- **Performance Highlights**: 세 가지 학습 시나리오(감독 학습, 능동 학습, 지속 학습)에서 진행된 실험에서 SES 방법이 기존 방법들에 비해 항상 우수한 성능을 보였습니다. 이는 본 방법이 선택된 샘플의 정보성과 대표성을 효과적으로 개선함을 보여줍니다.



### A Pilot Study of Applying Sequence-to-Sequence Voice Conversion to Evaluate the Intelligibility of L2 Speech Using a Native Speaker's Shadowings (https://arxiv.org/abs/2410.02239)
Comments:
          Accepted by APSIPA ASC 2024. arXiv admin note: text overlap with arXiv:2409.11742

- **What's New**: 새로운 연구는 L2 화자의 발화에서 이해할 수 없는 부분을 강조하는 가상의 shadower 시스템을 개발하는 것을 목표로 한다. 이 시스템은 Voice Conversion(VR) 기술을 활용하여 L1 화자의 그림자 발화 과정을 모방한다.

- **Technical Details**: 이 연구는 L1 화자의 그림자 발화 데이터를 바탕으로 L1의 발화와 L2 발화를 매칭하여 Voice Conversion 모델을 개발한다. 실험은 Seq2Seq VC 구조를 사용하며, PPG(Phonetic Posteriorgrams)를 기반으로 한 동적 시간 정합(Dynamic Time Warping, DTW) 방법론을 적용하여 발화의 유창성을 평가한다.

- **Performance Highlights**: 가상 shadower 시스템의 출력은 언어적 및 음향적 측면에서 실제 L1 그림자 발화와 유사성을 보여준다. 실험 결과는 L1 화자가 L2 발화를 그림자 발화할 때 나타나는 문제 부분을 효과적으로 인식하고 강조할 수 있는 가능성을 시사한다.



### CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning (https://arxiv.org/abs/2410.02229)
Comments:
          work in progress

- **What's New**: 본 논문에서는 대규모 고품질 공개 소스 코드로부터 합성된 코드-선호 쌍(code-preference pairs)을 활용하여 LLM의 추론 능력을 향상시킬 수 있는 CodePMP라는 확장 가능한 선호 모델 사전 훈련(pretraining) 파이프라인을 소개합니다.

- **Technical Details**: CodePMP는 코드 관련 프롬프트나 설명을 기반으로 선택된(code responses) 및 거부된(rejected code responses) 코드 응답 쌍을 합성하여 선호 쌍을 생성합니다. 이러한 <chosen, rejected> 쌍은 수백만 개에 달하며, 대규모 합성 선호 데이터셋을 형성합니다. 이 데이터셋은 쌍별 순위 목표(pairwise ranking objectives)를 사용하여 선호 모델을 사전 훈련하는 데 활용됩니다. CodePMP는 GSM8K 및 MATH 같은 수학적 추론 과제와 ReClor 및 LogiQA2.0 같은 논리적 추론 과제에 대해 평가됩니다.

- **Performance Highlights**: 실험 결과, CodePMP는 RM(RM fine-tuning)의 정확도를 크게 향상시키고 다양한 과제에서의 강인성을 높였습니다. CodePMP를 초기화한 RM은 더 높은 성능을 보이면서도, 넓은 범위의 과제에서 개선된 성능을 제공합니다.



### General Preference Modeling with Preference Representations for Aligning Language Models (https://arxiv.org/abs/2410.02197)
Comments:
          34 pages

- **What's New**: 본 논문에서는 인간의 선호도를 모델링하는 새로운 접근 방식을 소개합니다. 기존의 Bradley-Terry (BT) 보상 모델과 비교하여, 제안된 방법은 선호를 효율적으로 캡처하는 잠재 공간(latent space)에 응답을 임베딩하는 preference representation learning을 활용합니다.

- **Technical Details**: 제안된 preference representation learning 방식은 복잡한 선호 구조를 포착할 수 있으며, 선호 측정 시 쿼리 복잡도가 𝒪⁢(K)로, 기존의 𝒪⁢(K2)보다 개선되었습니다. 이를 통해 다양한 응답들 사이의 관계를 효율적으로 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, General Preference representation model (GPM)은 RewardBench 벤치마크에서 BT 보상 모델에 비해 최대 5.6% 향상된 성능을 보였고, AlpacaEval2.0 및 MT-Bench에서도 최대 9.3%의 성능 향상을 달성했습니다.



### CodeJudge: Evaluating Code Generation with Large Language Models (https://arxiv.org/abs/2410.02184)
Comments:
          Accepted to EMNLP 2024 (Main, Long Paper)

- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)를 활용하여 테스트 케이스 없이 생성된 코드의 의미적 정확성을 평가하는 CodeJudge라는 코드 평가 프레임워크를 제안합니다.

- **Technical Details**: CodeJudge는 두 가지 평가 방법을 제공하며, LLM이 "느린 사고"를 수행할 수 있도록 안내하는 방법을 설계했습니다. 첫 번째 평가는 LLM에게 코드 기능의 단계별 분석을 요청하고 이를 요약하여 이진 결정을 내리게 합니다. 두 번째 평가는 LLM에게 일반적인 코드 오류의 분류 체계를 제공하여 생성 코드의 오류 유형을 식별하게 한 후 이 오류의 심각도에 따라 코드 정확도 점수를 계산합니다.

- **Performance Highlights**: CodeJudge는 4개의 LLM을 평가자로 사용하여 9가지 기존 방법들과 비교한 결과, 대부분의 설정에서 기존 방법들보다 12.1%에서 41.8% 더 높은 상관 관계를 보였습니다. 상대적으로 작은 모델인 Llama-3-8B-Instruct를 사용했음에도 불구하고, CodeJudge는 SOTA인 GPT-3.5 기반의 코드 평가 방법인 ICE-Score보다 우수한 성능을 기록했습니다.



### HATFormer: Historic Handwritten Arabic Text Recognition with Transformers (https://arxiv.org/abs/2410.02179)
- **What's New**: HATFormer는 아랍어 필기 인식(HTR) 시스템으로, 기존의 CNN 및 RNN 기반 방법보다 더 효율적인 transformer 기반의 구조를 가지고 있습니다. 이 시스템은 복잡한 아랍어 스크립트의 특성을 효과적으로 처리하기 위해 최적화되었습니다.

- **Technical Details**: HATFormer는 transformer 구조를 기반으로 하며, attention 메커니즘을 활용하여 연결된 문자, 문맥 의존적 형태, 그리고 단어 의미를 변화시킬 수 있는 diacritics를 인식하는 데 강점을 갖고 있습니다. 또한, 이미지 전처리기와 텍스트 토크나이저를 포함하여 제한된 역사적 데이터로부터 더 나은 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: HATFormer는 가장 큰 공개 역사적 필기 아랍어 데이터셋으로 평가 시 8.6%의 문자 오류율(CER)을 기록했습니다. 또한, 사적 데이터셋에서 4.2% CER을 달성하며, 기존 방법들보다 각각 51% 개선되었습니다. 이 시스템은 역사적 아랍어 문서의 자동 전사와 인식 오류 진단에 도움을 줄 수 있어, 디지털 인문학 연구에 큰 기여를 할 것입니다.



### Training Nonlinear Transformers for Chain-of-Thought Inference: A Theoretical Generalization Analysis (https://arxiv.org/abs/2410.02167)
- **What's New**: 본 논문은 Chain-of-Thought (CoT) 능력을 갖춘 Transformer의 훈련에 대한 최초의 이론적 연구를 제공합니다. CoT는 여러 단계의 추론을 위한 예시로 입력을 증강하여 대형 언어 모델의 추론 능력을 증대시키는 효과적인 방법입니다.

- **Technical Details**: 우리는 비선형 Transformer의 훈련 동역학을 분석하고 훈련 샘플과 반복 횟수, 유효한 CoT 추론에 필요한 문맥 예제의 수를 정량화합니다. 이를 통해 CoT 능력이 확보된 모델이 새로운 작업에 대해 일반화할 수 있는 조건을 입증합니다.

- **Performance Highlights**: 실험을 통해 CoT가 in-context learning (ICL)보다 더 정확한 결과를 제공하는 이유와 CoT의 일반화 능력을 확인하였습니다. CoT 능력을 제대로 활용하기 위해서는 특정 수의 훈련 샘플과 문맥 예제가 필요하다는 것을 보여줍니다.



### A LLM-Powered Automatic Grading Framework with Human-Level Guidelines Optimization (https://arxiv.org/abs/2410.02165)
- **What's New**: 이번 연구에서는 통합 다중 에이전트 자동 단답형 평가(ASAG) 프레임워크인 GradeOpt를 제안하여, 대규모 언어 모델(LLMs)을 활용하여 자동으로 평가 지침을 최적화하는 방법을 소개합니다.

- **Technical Details**: GradeOpt는 LLM을 평가자로 사용하며, 반영 에이전트(reflector)와 정제 에이전트(refiner)라는 두 개의 추가 LLM 기반 에이전트를 포함하여 데이터를 최적화합니다. 이 시스템은 기존 평가 지침을 자동으로 개선하기 위한 자기 반영(self-reflection) 메커니즘을 활용합니다.

- **Performance Highlights**: 실험 결과 GradeOpt는 교육 내용 지식(PCK) 및 내용 지식(CK) 질문의 평가에서 사람 평가자와의 행동 정렬 및 grading accuracy에서 기존 기준보다 우수한 성능을 보였습니다. 또한, 반복적인 실험을 통해 평가 정확도가 지속적으로 향상되는 것을 확인했습니다.



### Mitigating Memorization In Language Models (https://arxiv.org/abs/2410.02159)
- **What's New**: 이 연구는 Language Models (LMs)가 정보를 '기억하는'(memorize) 능력을 줄이기 위한 여러 방법론을 제안합니다. 이 중 세 가지 regularizer 기반, 세 가지 fine-tuning 기반, 그리고 열한 가지 unlearning 기반 방법을 포함하고 있으며, 그 중 다섯 가지는 새로운 방법으로 제시되었습니다. 또한, 이 연구는 기억 완화 기법을 빠르게 개발하고 평가할 수 있는 작은 규모의 LMs인 TinyMem을 소개합니다.

- **Technical Details**: 신문에서는 LM이 학습 데이터를 메모리에 저장하고 이를 의도하지 않게 반출할 수 있는 문제를 다룹니다. 연구진은 방법론을 평가하기 위해 TinyMem을 활용하여 다양한 사례에서 기억 완화 방법의 효과를 측정했습니다. 정리된 방법들은 regularizer 방법이 느리고 효과적이지 않으며, fine-tuning 방법은 비쌉니다. 반면 unlearning 방법은 빠르고 효과적인 것으로 확인되었습니다. 특히 새로운 unlearning 기법인 BalancedSubnet이 가장 뛰어난 성과를 보였습니다.

- **Performance Highlights**: 연구 결과, 기존의 방법들과 비교해 BalancedSubnet이 기억된 정보를 제거하면서도 목표 작업의 성능을 유지하는 데 가장 효과적임을 보여주었습니다. TinyMem에서 개발된 완화 방법들은 큰 생산 모델에서도 적용 가능성을 입증하였습니다.



### The why, what, and how of AI-based coding in scientific research (https://arxiv.org/abs/2410.02156)
Comments:
          23 pages, 7 figure, 3 boxes

- **What's New**: 본 논문은 연구자들이 프로그래밍을 더욱 직관적으로 활용할 수 있도록 돕기 위해 Generative AI와 대형 언어 모델(LLMs)의 역할을 분석하고, AI 기반 코딩의 실제적인 활용법을 제시합니다.

- **Technical Details**: 저자들은 LLM을 활용한 코딩의 세 가지 주요 관점을 다루는데, 첫째는 LLM이 코딩에서 어떤 역할을 수행하는지(why), 둘째는 LLM이 제공하는 6가지 코딩 지원 유형(what), 셋째는 5단계의 실제적인 구현 전략을 포함한 워크플로우(how)입니다.

- **Performance Highlights**: 이 프레임워크는 AI를 활용하여 코딩 관행과 교육을 개선함으로써 과학적 진전을 가속화하는 데 도움을 주기 위한 실행 가능한 통찰력을 제공합니다.



### From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities (https://arxiv.org/abs/2410.02155)
- **What's New**: 최근 연구에서 멀티모달 대형 언어 모델(MLLMs)의 발전을 위해 새로운 BPE 이미지 토크나이저를 도입하여 비주얼(visual) 데이터와 텍스트 모달리티 간의 효과적인 정렬 문제를 해결했습니다. 이 접근 방식은 기존의 방법들과는 달리 구조적 선행 정보(structural prior information)를 이미지 토큰에 직접적으로 통합합니다.

- **Technical Details**: 제안된 BPE 이미지 토크나이저는 두 차원 시퀀스 데이터를 효과적으로 학습할 수 있는 새로운 패러다임을 제공합니다. 이미지 패치를 패턴에 따라 결합하여 언어 모델 해석기에 더 의미 있는 정보가 포함된 토큰을 공급함으로써 transformer 모델이 이미지 데이터에 대한 더 나은 이해를 할 수 있게 합니다. 이 방법은 데이터 훈련 빈도를 기반으로 토큰을 병합하여 구조적 우선 정보를 추가하는 데 중점을 두며, 토큰화(tokenization) 과정을 통해 특정 손실을 줄일 수 있다는 것을 이론적으로도 입증하였습니다.

- **Performance Highlights**: 실험을 통해 제안된 BPE 이미지 토크나이저가 MLLM의 멀티모달 이해 능력을 크게 향상시키며, 훈련 데이터가 제한된 상황에서도 성능을 개선하는 것으로 나타났습니다. 이 방법은 다양한 벤치마크에서 성능이 향상될 뿐만 아니라, 확장성(scalability)도 뛰어난 가능성을 보여줍니다.



### C-MELT: Contrastive Enhanced Masked Auto-Encoders for ECG-Language Pre-Training (https://arxiv.org/abs/2410.02131)
- **What's New**: C-MELT라는 새로운 프레임워크를 제안하여 Electrocardiogram (ECG) 신호와 텍스트 데이터를 결합하여 임상 진단 능력을 향상시키고자 합니다. 이 프레임워크는 contrastive masked auto-encoder 아키텍처를 기초로 하여 ECG와 텍스트 데이터를 사전 학습합니다.

- **Technical Details**: C-MELT는 generative 능력과 향상된 discriminative 능력을 결합하여 강력한 cross-modal 표현을 달성합니다. 이를 위해 마스킹된 modality 모델링, 전문화된 손실 함수, cross-modal 정렬을 위한 개선된 negative sampling 전략을 채택합니다.

- **Performance Highlights**: C-MELT는 다섯 개의 공공 데이터세트에서 수행한 실험을 통해 기존 방법들보다 15% 및 2% 향상된 linear probing과 zero-shot 성능을 달성했습니다. 이는 C-MELT의 효과성을 강조하며 다중 모드 표현을 통해 자동화된 임상 진단을 향상시킬 가능성을 보여줍니다.



### Can LLMs Reliably Simulate Human Learner Actions? A Simulation Authoring Framework for Open-Ended Learning Environments (https://arxiv.org/abs/2410.02110)
- **What's New**: 이 논문은 LLMs(대형 언어 모델)를 사용하여 학습자의 행동을 시뮬레이션하는 새로운 프레임워크인 Hyp-Mix를 소개합니다. 이전 연구들은 LLMs의 잠재력을 보여주었지만, 제한된 신뢰성과 일반화 문제로 인해 초기 단계에서 머물렀습니다. 새로운 프레임워크는 테스트 가능한 가설을 결합하여 시뮬레이션을 개발하고 평가할 수 있게 합니다.

- **Technical Details**: Hyp-Mix는 학습자의 행동에 대한 가설을 포함하여 시뮬레이션을 설계하는을 도와줍니다. LLM을 사용하여 피지컬 학습 환경에서 실험을 수행했고, GPT-4 Turbo는 기본 학습자 모델이 변경되더라도 일관된 행동을 유지하며, LLM이 오픈 엔디드(interactive) 학습 환경에서 현실적인 행동을 시뮬레이션할 수 있음을 보여주는 첫 번째 증거를 제공합니다.

- **Performance Highlights**: Hyp-Mix를 통해 LLM은 자주 사용되는 핵심 행동 패턴을 잘 시뮬레이션하며, 기존의 모델에 비해 신뢰성이 높은 결과를 생성합니다. 이 연구는 오픈 엔디드 학습 환경에서의 LLM 행동 시뮬레이션의 필요성을 강조하며, 교육 전문가들이 이미 익숙한 기본 원리를 활용해 시뮬레이션을 개선할 수 있는 가능성을 보여줍니다.



### A Watermark for Black-Box Language Models (https://arxiv.org/abs/2410.02099)
- **What's New**: 본 연구는 대형 언어 모델(LLM)의 출력을 탐지하기 위해 필요한 새로운 워터마킹(watermarking) 방식인 '블랙박스(block-box) 접근' 방식을 제안합니다. 기존의 화이트박스(white-box) 접근 방식이 요구하는 정보 없이도 워터마킹이 가능하다는 점에서 혁신적입니다.

- **Technical Details**: 제안된 워터마킹 기법은 LLM에서 시퀀스를 샘플링할 수 있는 능력만 있으면 적용할 수 있으며, '왜곡 없는(distortion-free)' 속성을 가집니다. 이는 여러 비밀 키를 사용하여 체인 또는 중첩할 수 있으며 손실 없이 자동회귀적으로(autoregressive) 작동합니다. 특정 텍스트가 워터마킹 되었는지 여부를 평가하는 방법과 함께, 성능 보장(performance guarantees)도 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 화이트박스 방식보다 우수한 성능을 보이며, 화이트박스 접근이 가능할 때도 이를 효과적으로 활용할 수 있음을 증명했습니다. 전반적으로, 이 방법은 모델 소유자가 아닌 사용자가 워터마킹을 손쉽게 적용할 수 있도록 하여 실질적인 적용 가능성을 높입니다.



### EMMA: Efficient Visual Alignment in Multi-Modal LLMs (https://arxiv.org/abs/2410.02080)
- **What's New**: EMMA(효율적인 다중 모드 적응)은 시각적 인코딩과 텍스트 인코딩을 효율적으로 결합할 수 있도록 설계된 경량 크로스 모달리티 모듈입니다. 이는 언어 모델에서의 지침 인식 시각적 표현 생성을 가능하게 합니다.

- **Technical Details**: EMMA는 적은 추가 파라미터(모델 크기의 0.2% 증가)로 비전과 언어 표현을 통합하는 효율적인 초기 융합 메커니즘을 도입합니다. 이는 시각적 표현을 지침과 동기화하여 MLLM에 대한 성능을 향상시키는 역할을 합니다. 또한, 시각적 정렬 모듈에 대한 심층 분석을 수행하여 관련 표적 토큰에 집중하는 방식에 대한 내부 메커니즘을 설명합니다.

- **Performance Highlights**: EMMA는 여러 작업에서 성능을 최대 9.3% 향상시키며, 헬루시네이션에 대한 강인성을 크게 향상시킵니다. 기존의 mPLUG-Owl2와 비교했을 때, 7개 벤치마크에서 우수한 성과를 보여줍니다.



### Inspection and Control of Self-Generated-Text Recognition Ability in Llama3-8b-Instruc (https://arxiv.org/abs/2410.02064)
Comments:
          10 pages, 13 figs, 2 tables, submitted to ICLR 2025

- **What's New**: 이 논문은 Llama3-8b-Instruct와 같은 대형 언어 모델(LLMs)이 자신의 출력을 인간의 출력과 구별할 수 있는 능력을 조사합니다. 저자들은 이 현상이 AI 안전성과 관련이 있음을 강조하며, 모델이 자가 인식(self-recognition)을 통해 자신의 작문 스타일을 인지할 수 있는지를 검증합니다.

- **Technical Details**: 연구에서는 Llama3-8b-Instruct 모델이 특정 벡터를 사용하여 자가 출처 인식(self-authorship recognition) 작업을 성공적으로 수행하는 방법을 분석합니다. 이 벡터는 잔여 스트림(residual stream) 내에서 활성화되며, 모델이 자신이 생성한 텍스트를 인식할 때 강하게 작용합니다. 연구진은 이 벡터를 활용하여 모델의 행동을 조작하고, 자가 출처를 주장하거나 부인하도록 유도합니다.

- **Performance Highlights**: Llama3-8b-Instruct 모델은 새로운 데이터 세트에서도 자가 인식 작업에서 100%에 가까운 정확도로 자기 출처를 주장할 수 있으며, 벡터가 제거될 경우 정확도가 크게 감소합니다. 이러한 결과는 자가 인식 능력이 모델의 행동에 직접적인 영향을 미친다는 것을 보여줍니다.



### TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models (https://arxiv.org/abs/2410.02062)
- **What's New**: 이번 논문에서는 TPP(Temporal Point Processes)를 LLM(Large Language Models)과 통합한 새로운 프레임워크인 TPP-LLM을 소개합니다. 이것은 이벤트 시퀀스의 의미적 및 시간적 측면을 포착하는 데 중점을 두고, 텍스트 설명을 활용하여 기존 방법보다 더 풍부한 의미 정보를 담아냅니다.

- **Technical Details**: TPP-LLM은 전통적인 카테고리 기반 이벤트 타입 표현 대신 텍스트 설명을 직접 활용합니다. 또한, 시간 역학을 효과적으로 학습하기 위해, temporal embeddings를 포함하고 PEFT(Parametric Efficient Fine-tuning) 방법을 사용하여 모델을 조정합니다. 이로써 전체 모델을 다시 훈련시키지 않고도 성능을 유지할 수 있습니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험 결과, TPP-LLM이 기존의 최신 모델들과 비교하여 시퀀스 모델링 및 다음 이벤트 예측에서 일관되게 뛰어난 성능을 보임을 입증하였습니다.



### Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data (https://arxiv.org/abs/2410.02056)
Comments:
          Code and Checkpoints will be soon available here: this https URL

- **What's New**: Synthio는 합성 데이터를 통해 소규모 오디오 분류 데이터 세트를 증강하는 혁신적인 방법을 제안합니다. 이 방법은 제한된 라벨 데이터로 오디오 분류 정확도를 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: Synthio는 두 가지 주요 단계를 포함합니다: 1) Preference Optimization을 통해 T2A(텍스트-투-오디오) 모델의 생성물을 소규모 데이터 세트와 정렬합니다. 2) MixCap이라는 방법으로 다양한 오디오 캡션을 생성하고 이를 통해 다양한 합성을 유도합니다.

- **Performance Highlights**: Synthio는 10개의 데이터 세트와 4개의 시뮬레이션된 제한 데이터 환경에서 평가되었으며, 약한 캡션을 가진 AudioSet에서 학습된 T2A 모델을 사용하여 모든 기준점보다 0.1%-39% 향상된 성능을 보여주었습니다.



### Emo3D: Metric and Benchmarking Dataset for 3D Facial Expression Generation from Emotion Description (https://arxiv.org/abs/2410.02049)
Comments:
          11 pages, 10 figures

- **What's New**: 이 논문은 다양한 감정을 포괄하는 'Emo3D'라는 폭넓은 'Text-Image-Expression dataset'을 소개합니다. 이 데이터셋은 인물의 감정 표현과 관련된 이미지 및 3D blendshapes와 함께 제공되어, 이전의 한정된 감정 클래스와 부족한 데이터셋 문제를 해결합니다.

- **Technical Details**: Emo3D 데이터셋은 150,000개의 인스턴스로 구성되어 있으며, 각 인스턴스는 감정 설명, 해당 이미지, blendshape 점수의 삼위일체를 포함합니다. GPT-3.5를 사용하여 감정에 대한 텍스트 설명을 생성하고, DALL-E 3로 이미지를 생성하며, Mediapipe 프레임워크를 통해 대응하는 blendshape 점수를 추정합니다.

- **Performance Highlights**: Emo3D 평가 메트릭은 'Mean Squared Error (MSE)' 메트릭보다 3D 얼굴 표현의 시각-텍스트 정렬 및 의미적 풍부성을 평가하는 데 더 우수함을 입증하였습니다. 이 데이터셋은 애니메이션 디자인, 가상현실, 감정 기반 인간-컴퓨터 상호작용에 큰 응용 가능성을 지니고 있습니다.



### Zodiac: A Cardiologist-Level LLM Framework for Multi-Agent Diagnostics (https://arxiv.org/abs/2410.02026)
- **What's New**: ZODIAC라는 새로운 LLM 기반 프레임워크를 도입하였으며, 심장 전문의 수준의 전문성을 갖추어 심장 진단 분야에서 LLM의 효과성을 높이기 위해 설계되었습니다.

- **Technical Details**: ZODIAC는 다중 에이전트 협력 프레임워크 기반으로 구축되어 있으며, 실제 환자 데이터와 심장 전문의의 검토를 통해 세부적으로 조정되었습니다. 이 시스템은 다양한 데이터 모달리티를 처리하고, ECG 이미지를 포함한 환자 데이터를 분석합니다.

- **Performance Highlights**: ZODIAC는 OpenAI의 GPT-4o, Meta의 Llama-3.1-405B, Google의 Gemini-pro 등 업계에서 선도하는 모델들보다 뛰어난 성과를 보였으며, 실제 임상 환경에서의 응용 가능성이 입증되었습니다.



### FLAG: Financial Long Document Classification via AMR-based GNN (https://arxiv.org/abs/2410.02024)
Comments:
          8 pages, 3 figures, to be published in CIFEr Conference 2024 as "Semantic Graph Learning for Trend Prediction from Long Financial Documents"

- **What's New**: 이번 연구에서는 긴 금융 문서 분류를 위한 AMR 기반의 GNN 프레임워크인 FLAG(Financial Long document classification via AMR-based GNN)를 제안하였습니다. 이 방법은 장기 금융 문서의 의미론적 관계를 더욱 잘 이해할 수 있도록 도와줍니다.

- **Technical Details**: FLAG는 문서의 각 문장을 AMR 그래프로 변환한 후, 이를 기반으로 문서 수준의 AMR 그래프를 계층적으로 구성합니다. 초기화된 노드는 금융 도메인에서 훈련된 FinBERT를 통해 생성된 LLM word embeddings로 이루어집니다. 이를 통해 GNN 모델을 적용하여 최종 문서 표현을 생성합니다.

- **Performance Highlights**: FLAG는 S&P 1500 지수의 다양한 기업의 분기 실적 발표 기록을 포함한 데이터셋에서 직접 텍스트로 LLM을 미세 조정한 기존 방법들보다 주가 동향 예측에서 더 높은 성능을 보여주었습니다.



### Financial Sentiment Analysis on News and Reports Using Large Language Models and FinBER (https://arxiv.org/abs/2410.01987)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)과 금융 변형 모델인 FinBERT를 활용한 금융 감성 분석(Financial Sentiment Analysis, FSA)의 성능을 비교 연구하고, 프롬프트 엔지니어링(prompt engineering) 기법의 장점을 강조합니다.

- **Technical Details**: FSA는 시장 감성을 평가하고 의사 결정을 지원하는 데 필수적입니다. 이 연구는 LLM과 FinBERT의 응용을 뉴스 기사, 재무 보고서, 기업 발표 문서에 대해 비교하였으며, zero-shot 및 few-shot 전략을 통한 감성 분류 정확도의 향상 방법을 중점적으로 다룹니다.

- **Performance Highlights**: 실험 결과, GPT-4o는 금융 텍스트에 대한 few-shot 예시를 통해 잘 튜닝된 FinBERT와 동일한 수준의 성능을 발휘할 수 있음을 보여주었습니다.



### CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL (https://arxiv.org/abs/2410.01943)
- **What's New**: 이번 논문에서는 Text-to-SQL 작업을 위한 LLM의 성능 향상을 위해 새로운 프레임워크인 CHASE-SQL을 소개합니다. 이 프레임워크는 multi-agent modeling을 활용하여 후보 생성 및 선택을 개선하는 혁신적인 전략을 사용합니다.

- **Technical Details**: CHASE-SQL은 LLMs의 본질적인 지식을 활용하여 다양한 SQL 후보를 생성하는 방법론을 제안합니다. 주요 기술로는 (1) 복잡한 쿼리를 관리 가능한 하위 쿼리로 분해하는 divide-and-conquer 방법, (2) 쿼리 실행 계획을 기반으로 한 chain-of-thought reasoning 방법, (3) 인스턴스 인식된 합성 예제 생성 기술이 있습니다. 이를 통해 후보 쿼리를 생성하고, pairwise 비교를 통해 최고의 후보를 선택하는 과정을 거칩니다.

- **Performance Highlights**: CHASE-SQL은 BIRD Text-to-SQL 데이터셋의 테스트 세트와 개발 세트에서 각각 73.0%와 73.01%의 실행 정확도를 달성하며, 이전 방법들을 능가하는 성능을 보여주었습니다. 전체적으로, 이 방법론은 SQL 쿼리의 품질과 다양성을 높이는 데 기여하며, 기존의 방법들에 비해 우수한 결과를 확인할 수 있습니다.



### A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation (https://arxiv.org/abs/2410.01912)
Comments:
          25 pages, 20 figures, code is open at this https URL

- **What's New**: 이번 연구는 벡터 양자화 (Vector Quantization, VQ) 기반의 자율 회귀 (Autoregressive) 이미지 생성을 위해 새로운 모델 아키텍처인 2차원 자율 회귀 (2-Dimensional Autoregression, DnD) Transformer를 도입하여 정보를 잃는 병목 현상을 해결합니다.

- **Technical Details**: DnD-Transformer는 모델 깊이 (model depth)라는 새로운 자율 회귀 방향을 도입하여 이미지에 대해 더 많은 코드를 예측하도록 설계되었습니다. 기존의 1차원 자율 회귀와 RQ-Transformer와 같은 2차원 이미지 분해 방식을 비교했을 때, DnD-Transformer는 엔드-투-엔드 모델로 동일한 백본 모델 크기와 시퀀스 길이로 더 높은 품질의 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과는 DnD-Transformer가 세밀한 이미지 세부 사항을 우수하게 재구성하며, 1D 방법과 비교해 더 효율적이고 낮은 엔트로피 분해를 보여주었습니다. 또한 ImageNet 256x256 생성에서 AR 기본 모델보다 크게 향상된 성능 (최대 1.54 FID 및 82.6 IS 개선)을 달성하였으며, 다중 모드 모델링에 대한 자율 회귀 모델의 뚜렷한 장점을 강조합니다.



### NEAT: Nonlinear Parameter-efficient Adaptation of Pre-trained Models (https://arxiv.org/abs/2410.01870)
- **What's New**: 본 논문에서는 기존의 Low-Rank Adaptation (LoRA) 방법의 한계를 극복하기 위해 Neat이라는 비선형(Nonlinear) 파라미터 효율적(adaptation) 방법을 제안합니다. Neat은 경량 신경망를 도입하여 사전 훈련된 가중치를 입력으로 받아 누적 가중치 업데이트를 비선형 함수로 근사합니다.

- **Technical Details**: Neat은 사전 훈련된 모델의 입력으로 가중치를 사용하고, 비선형 변환을 통해 누적 가중치 업데이트를 모델링합니다. 이 방법은 가중치 업데이트에서 복잡하고 비선형 구조를 포착하여, 추가적인 파라미터를 증가시키지 않으면서도 모델의 적응 성능을 개선합니다.

- **Performance Highlights**: NEAT 방식은 네 가지 벤치마크와 20개 이상의 데이터셋에서 성능 평가를 통해 기존의 LoRA 기반 방법들보다 비전(vision) 및 텍스트(text) 작업 모두에서 우수한 성능을 보였습니다.



### House of Cards: Massive Weights in LLMs (https://arxiv.org/abs/2410.01866)
Comments:
          Under review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)에서 발생하는 massive activations의 기원을 밝히고, 이를 해결하기 위한 새로운 접근법인 MacDrop을 제안합니다. 이 연구는 massive activations가 hidden state가 아니라 초기 layer의 feed-forward network의 intermediate state에서 기인함을 밝혔습니다.

- **Technical Details**: 논문에서는 'top-k massive weights'라는 개념을 도입하여, 특정 feature dimension에서 top-k 크기를 가진 weights를 정의합니다. 이 massive weights가 0으로 설정될 경우 LLM의 기능성이 완전히 망가지는 반면, 모든 다른 weights를 0으로 설정해도 성능 저하가 미미합니다. 이러한 관찰을 바탕으로 LLM의 효과적인 fine-tuning을 위한 기법인 MacDrop을 제안하며, 이는 massive weights에 dropout을 적용합니다.

- **Performance Highlights**: MacDrop을 통해 제안된 방식이 zero-shot downstream tasks와 generation tasks에서 전반적으로 성능을 향상시키고 있음을 보여줍니다.



### A GEN AI Framework for Medical Note Generation (https://arxiv.org/abs/2410.01841)
Comments:
          8 Figures, 7 page, IEEE standard research paper

- **What's New**: MediNotes라는 혁신적인 생성 AI 프레임워크가 도입되어, 의학적 대화로부터 SOAP (Subjective, Objective, Assessment, Plan) 노트를 자동으로 생성합니다.

- **Technical Details**: MediNotes는 Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Automatic Speech Recognition (ASR) 기술을 통합하여 실시간 또는 녹음된 오디오에서 텍스트와 음성 입력을 포착하고 처리합니다. 또한 Quantized Low-Rank Adaptation (QLoRA)와 Parameter-Efficient Fine-Tuning (PEFT) 기법을 활용하여 자원이 제한된 환경에서도 효과적인 모델 미세조정을 지원합니다.

- **Performance Highlights**: ACI-BENCH 데이터셋을 이용한 평가 결과, MediNotes는 자동 의료 문서화의 정확성, 효율성, 사용성을 획기적으로 개선하며, 의료 전문가의 행정 부담을 줄이고 임상 작업의 질을 향상시키는 효과적인 솔루션임을 입증했습니다.



### AI Conversational Interviewing: Transforming Surveys with LLMs as Adaptive Interviewers (https://arxiv.org/abs/2410.01824)
- **What's New**: 이번 연구는 대화형 인터뷰를 통해 사람들의 의견을 수집하는 전통적인 방법을 대체할 수 있는 가능성을 탐색했습니다. 특히, Large Language Models (LLMs)을 활용하여 인적 자원 없이 스케일이 큰 대화형 인터뷰를 수행할 수 있는 방법을 제시합니다.

- **Technical Details**: 대학 학생들을 대상으로 한 소규모 연구에서 AI와 인간 면접자가 동일한 정치적 주제의 질문지를 사용하여 인터뷰를 진행했습니다. 연구는 AI Conversational Interviewing의 성능을 평가하고, 면접 프로토콜 준수, 응답 품질 및 참여자 참여도와 같은 다양한 양적 및 질적 지표를 분석했습니다.

- **Performance Highlights**: AI Conversational Interviewing은 전통적인 방법에 비해 동등한 질의 데이터를 생산할 수 있음을 확인했습니다. 연구 결과는 AI의 효과적인 구현을 위한 구체적인 추천 사항도 제공합니다.



### From Text to Multimodality: Exploring the Evolution and Impact of Large Language Models in Medical Practic (https://arxiv.org/abs/2410.01812)
Comments:
          12 pages, 1 figure

- **What's New**: 이번 논문은 대규모 언어 모델(Large Language Models, LLMs)이 다중 모드 플랫폼(Multimodal Platforms)으로 발전함에 따라 의료 분야에 미치는 영향을 포괄적으로 검토합니다. 다중 모드 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 발전과 현재 의료 실무에서의 응용 가능성을 탐구합니다.

- **Technical Details**: MLLMs는 텍스트, 이미지, 오디오 등 다양한 데이터 유형을 통합하여 보다 포괄적인 건강 데이터를 분석할 수 있는 기능을 가지고 있습니다. 연구는 REALM 및 MedDr와 같은 모델을 통해 비구조화 데이터와 구조화 데이터 간의 간극을 해소할 수 있는 능력을 강조합니다.

- **Performance Highlights**: MLLM은 임상 예측, 환자 참여 증대 및 개인화된 치료 계획 개선을 통해 의료 분야에서 환자 건강에 대한 포괄적인 이해를 가능하게 합니다. 예를 들어, LlaVA-Rad 모델은 표준 방사선 작업에서 최첨단 성과를 달성했으며, MedAide와 같은 챗봇은 자원이 부족한 지역에서도 의료 지원을 제공합니다.



### Evaluating Cultural Awareness of LLMs for Yoruba, Malayalam, and English (https://arxiv.org/abs/2410.01811)
Comments:
          19 pages, 10 figures, 6 tables

- **What's New**: 이 논문은 대규모 언어 모델 (LLM)이 지역 언어인 말라얄람(Malayalam)과 요루바(Yoruba)의 문화적 측면을 이해하는 능력을 탐구합니다. 또한, Hofstede의 6가지 문화적 차원을 사용하여 LLM 기반 응답의 문화적 인식을 정량화합니다.

- **Technical Details**: 연구에서 사용된 Hofstede의 6가지 문화적 차원은 권력 거리(Power Distance, PDI), 개인주의(Individualism, IDV), 성취 및 성공에 대한 동기(Masculinity vs. Femininity, MAS), 불확실성 회피(Uncertainty Avoidance, UAI), 장기 지향(Long Term Orientation, LTO), 그리고 쾌락주의와 절제(Indulgence vs. Restraint, IVR)입니다. LLM이 영어에서는 높은 문화적 유사성을 보이지만, 말라얄람과 요루바의 문화적 뉘앙스를 포착하지 못한다는 사실을 입증합니다.

- **Performance Highlights**: 대규모 지역 언어 LLM 훈련을 위한 문화적으로 풍부한 데이터셋이 필요하다는 점을 강조하며, 이는 채팅 기반 LLM의 사용자 경험 향상 및 대규모 LLM 에이전트 기반 시장 조사 유효성 개선에 큰 영향을 미칠 것입니다.



### Leopard: A Vision Language Model For Text-Rich Multi-Image Tasks (https://arxiv.org/abs/2410.01744)
Comments:
          Our code is available at this https URL

- **What's New**: 이번 연구에서는 Leopard라는 새로운 멀티모달 언어 모델(Multimodal Large Language Model, MLLM)을 소개합니다. Leopard는 텍스트가 중심 시각 요소로 작용하는 멀티 이미지 작업을 처리하기 위해 특별히 설계되었습니다. 이 모델은 고품질의 멀티모달 데이터와 적응형 고해상도 멀티 이미지 인코딩 모듈을 사용하여 기존의 한계점을 극복합니다.

- **Technical Details**: Leopard는 약 100만 개의 고품질 멀티모달 교육 데이터를 사용하여 학습하였으며, 이 데이터는 다중 페이지 문서, 다중 차트, 다중 표 및 웹 페이지 경로와 같은 실제 시나리오를 포괄합니다. 또한, 이미지의 원본 비율과 해상도를 기반으로 시각적 시퀀스 길이를 최적화하는 적응형 고해상도 멀티 이미지 인코딩 모듈을 통해 여러 고해상도 이미지를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: Leopard는 13개의 비전-언어 벤치마크 데이터셋에서 실험을 수행하였으며, 5개의 텍스트 풍부한 멀티 이미지 벤치마크에서 이전 최상의 오픈 소스 MLLM보다 평균 +9.61 포인트 향상된 성과를 보였습니다. 또한, 단일 이미지 및 일반 도메인 비전-언어 벤치마크에서도 높은 경쟁력을 유지하며, 최신 MLLM과 비교해도 유사한 성과를 거두었습니다.



### LML-DAP: Language Model Learning a Dataset for Data-Augmented Prediction (https://arxiv.org/abs/2409.18957)
Comments:
          Updated title, abstract, and images

- **What's New**: 본 논문은 설명 가능한 방식으로 분류 작업(Classification tasks)을 수행하기 위해 대형 언어 모델(Large Language Models, LLMs)을 사용하는 새로운 접근 방식을 소개합니다. 기존의 기계 학습(Machine Learning, ML) 모델들이 데이터 정제(Data cleaning)와 특징 엔지니어링(Feature engineering)에 의존하는 것과 달리, 이 방법은 LLMs를 사용하여 프로세스를 간소화합니다.

- **Technical Details**: 이 논문은 '언어 모델 학습(Language Model Learning, LML)'이라는 새로운 개념과 '데이터 증강 예측(Data-Augmented Prediction, DAP)'이라는 방법을 제안합니다. LML 과정에서는 데이터셋을 요약(Summarized)하고 평가하여 각 레이블(classification label)로 이어지는 특징(features)을 결정합니다. DAP 과정에서는 데이터 요약을 바탕으로 테스트 데이터셋의 행(row)을 사용하여 자동으로 쿼리(query)를 생성하고, 이를 통해 데이터셋에서 관련 행을 검색하여 분류를 수행합니다.

- **Performance Highlights**: 제안된 시스템은 복잡한 데이터에서도 만족스러운 정확도를 보장하며, 일부 테스트 사례에서는 90% 이상의 정확도를 기록했습니다. 이는 기존 ML 모델을 다양한 시나리오에서 초월할 잠재력을 보여줍니다.



New uploads on arXiv(cs.IR)

### Long-Sequence Recommendation Models Need Decoupled Embeddings (https://arxiv.org/abs/2410.02604)
Comments:
          First three authors contributed equally

- **What's New**: 기존의 긴 시퀀스 추천 모델에서의 주의(attention)와 표현(representation) 학습 간의 간섭 문제를 처음으로 식별하고 이에 대한 새로운 해결책을 제시합니다.

- **Technical Details**: 본 연구에서 제안하는 DARE 모델은 주의 및 표현을 위해 각각 두 개의 독립적인 임베딩 테이블을 사용하여 이 두 모듈을 완전히 분리합니다. 이를 통해 주의와 표현을 최적으로 조정할 수 있으며, 주의 임베딩 차원을 절반으로 줄여 검색 속도를 50% 가속화합니다.

- **Performance Highlights**: DARE는 공공 데이터 세트에서 기존의 TWIN 모델을 초월하여 AUC 개선을 최대 0.9% 달성했으며, 세계 최대 온라인 광고 플랫폼 중 하나에서 GMV(총 상품 가치)를 1.47% 향상시켰습니다.



### Quantifying User Coherence: A Unified Framework for Cross-Domain Recommendation Analysis (https://arxiv.org/abs/2410.02453)
- **What's New**: 이 논문은 추천 시스템(Recommender Systems, RS)의 사용자 프로필 품질과 관련된 새로운 정보 이론적 측정을 제안합니다. 특히, 사용자의 인기 선택에서의 편차를 정량화하는 'surprise' 측정치와 사용자 상호작용 일관성을 포착하는 'conditional surprise' 측정치를 소개합니다.

- **Technical Details**: 논문에서는 9개의 데이터 세트에서 7가지 추천 알고리즘을 평가하며, 우리의 측정치와 기존 성능 메트릭 간의 관계를 밝힙니다. 엄격한 통계적 프레임워크를 사용하여, 사용자 프로필 밀도와 정보 측정이 알고리즘 성능에 미치는 영향을 정량화합니다. 이러한 측정치를 기반으로 사용자를 분할하여 데이터 양을 줄이고 성능을 향상시키며, 단순한 알고리즘이 저일관성 사용자에 대해 복잡한 알고리즘과 대응할 수 있음을 보여줍니다.

- **Performance Highlights**: 다양한 추천 알고리즘이 예측에서 사용자 선호도의 일관성과 다양성을 얼마나 잘 유지하는지를 분석하여 알고리즘의 행동에 대한 통찰을 제공합니다. 이 연구는 사용자 행동에 대한 이론적 이해를 발전시키고 개인화된 추천 시스템을 위한 실용적인 휴리스틱을 제시하여 더 효율적이고 적응력 있는 아키텍처를 촉진합니다.



### Multi-modal clothing recommendation model based on large model and VAE enhancemen (https://arxiv.org/abs/2410.02219)
- **What's New**: 이 연구는 의류 추천을 위한 다중모드 패러다임을 제안합니다. 이는 의류 설명 텍스트와 이미지를 통합한 다중모드 분석 방법을 설계하고, 사전 훈련된 대형 언어 모델을 활용하여 사용자와 제품의 숨겨진 의미를 심층적으로 탐구합니다.

- **Technical Details**: 연구에서는 또한 변량 인코더(variational encoder)를 사용하여 사용자 정보와 제품 간의 관계를 학습합니다. 이는 추천 시스템에서의 콜드 스타트(Cold Start) 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: 본 연구는 광범위한 제거 실험(ablation experiments)을 통해 이 방법이 다양한 추천 시스템 방법들보다 뛰어난 성능 우위를 가지는 것을 검증하였으며, 추천 시스템의 종합적인 최적화에 대한 중요한 실질적인 지침을 제공합니다.



### A Survey on Point-of-Interest Recommendation: Models, Architectures, and Security (https://arxiv.org/abs/2410.02191)
Comments:
          20 pages

- **What's New**: 이 논문은 POI(Point-of-Interest) 추천 시스템의 최신 발전을 폭넓게 검토하며, 기존의 전통적인 접근 방식에서 벗어나 최첨단 기술, 새로운 아키텍처 및 보안 고려사항에 대한 포괄적인 분석을 제공합니다.

- **Technical Details**: POI 추천 시스템의 진화는 전통적인 잠재 요인 모델에서 심층 학습 구조로 변화하고 있습니다. Latent Dirichlet Allocation (LDA) 및 Matrix Factorization (MF)에서 Long Short-Term Memory (LSTM) 네트워크, Transformer 아키텍처와 Graph Neural Networks (GNNs)로의 발전을 다루며, LLMs, Diffusion Models (DMs), Self-Supervised Learning (SSL)와 같은 최신 기법이 추천 정확도를 어떻게 향상시키는지에 대해 논의합니다. 또한, 중앙 집중 모델에서 분산 학습(federated learning)으로의 이행이 어떻게 시스템의 확장성과 사용자 프라이버시를 개선하는지에 대해서도 밝힙니다.

- **Performance Highlights**: 최근 연구들은 POI 추천에서 사용자 데이터의 보안을 강화하기 위한 다양한 기법을 채택하고 있으며, differential privacy와 federated learning이 현대 POI 추천 시스템의 중심에 놓이고 있습니다. 추천 정확도를 향상시키는 혁신적인 모델과 아키텍처의 발전을 통해, 시스템은 사용자 선호도를 보다 잘 모델링할 수 있게 되었습니다.



### BayesCNS: A Unified Bayesian Approach to Address Cold Start and Non-Stationarity in Search Systems at Sca (https://arxiv.org/abs/2410.02126)
- **What's New**: BayesCNS는 검색 시스템에서 아이템의 콜드 스타트(Cold Start) 문제와 비정상적 분포 변화(Non-stationarity) 문제를 해결하기 위한 베이지안 기반의 접근법입니다. 이 방법은 사용자의 상호작용을 실시간으로 업데이트하여 사용자의 아이템 상호작용에 대한 사전 분포를 추정합니다.

- **Technical Details**: BayesCNS는 온라인 학습(Online Learning) 알고리즘을 기반으로 하여 새로운 사용자 상호작용 데이터를 사용해 사전 분포를 지속적으로 갱신합니다. 이 방식은 Thompson Sampling 알고리즘을 사용하여 비정상 상태에서도 효과적인 학습을 가능하게 하며, 랭커 모델(Ranker Model)과 연결되어 관련 아이템을 효율적으로 탐색할 수 있도록 지원합니다.

- **Performance Highlights**: 대규모 검색 시스템에서 BayesCNS를 적용한 결과, 온라인 A/B 실험에서 신규 아이템 상호작용이 10.60% 증가하고, 전체 성공 메트릭이 1.05% 향상되는 효과를 나타냈습니다.



### Price-guided user attention in large-scale E-commerce group recommendation (https://arxiv.org/abs/2410.02074)
- **What's New**: 기존 그룹 추천 시스템은 그룹 결정에 가장 큰 영향을 미치는 주요 사용자를 식별하기 위해 주의(attention) 메커니즘을 활용합니다. 본 연구는 저렴한 가격의 제품이 사용자의 추천 영향력에 미치는 영향을 분석하고, 가격 정보를 활용한 새로운 그룹 추천 방법인 Price-Guided User Attention (PGUsA)을 제안합니다.

- **Technical Details**: 제안된 PGUsA 모델은 사용자의 활동 이력과 제품 가격 정보를 기반으로 사용자의 중요도를 평가하는 주의(attention) 모듈을 포함합니다. 특히, 가격에 따라 조정 가능한 시그모이드 함수(adaptive sigmoid function)를 사용하여 사용자 집합의 정확성을 향상시킵니다. 이 모델은 가격 정보가 주어지면 기존의 모든 주의 기반 그룹 추천 시스템에 통합할 수 있습니다.

- **Performance Highlights**: 모델을 공공 벤치마크 데이터 및 실제 E-commerce 데이터셋에서 평가한 결과, 가격 기반 사용자 주의 접근 방식이 기존 최첨단 방법들보다 더 나은 성능을 보여주었으며, 특히 적중률(hit ratio)과 평균 제곱 오차(mean squared error)에서 우수한 결과를 도출했습니다.



### Financial Sentiment Analysis on News and Reports Using Large Language Models and FinBER (https://arxiv.org/abs/2410.01987)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)과 금융 변형 모델인 FinBERT를 활용한 금융 감성 분석(Financial Sentiment Analysis, FSA)의 성능을 비교 연구하고, 프롬프트 엔지니어링(prompt engineering) 기법의 장점을 강조합니다.

- **Technical Details**: FSA는 시장 감성을 평가하고 의사 결정을 지원하는 데 필수적입니다. 이 연구는 LLM과 FinBERT의 응용을 뉴스 기사, 재무 보고서, 기업 발표 문서에 대해 비교하였으며, zero-shot 및 few-shot 전략을 통한 감성 분류 정확도의 향상 방법을 중점적으로 다룹니다.

- **Performance Highlights**: 실험 결과, GPT-4o는 금융 텍스트에 대한 few-shot 예시를 통해 잘 튜닝된 FinBERT와 동일한 수준의 성능을 발휘할 수 있음을 보여주었습니다.



### The Importance of Causality in Decision Making: A Perspective on Recommender Systems (https://arxiv.org/abs/2410.01822)
Comments:
          Accepted at the CONSEQUENCES '24 workshop, co-located with ACM RecSys '24

- **What's New**: 추천 시스템(Recommendation Systems) 분야에서 인과관계(causality)의 중요성이 커지고 있으며, 이는 정확한 예측(predictions)을 효과적이고 설명 가능한 결정(decisions)으로 전환하기 위해 도움이 될 수 있다는 점을 강조하고 있습니다.

- **Technical Details**: 이 논문에서는 추천 시스템 문제를 인과관계 관점에서 공식화하며, 잠재 결과(Potential Outcomes, POs)와 구조적 인과 모델(structural causal models)을 활용하여 추정해야 할 인과량(causal quantities)의 정의를 제공합니다. 일반적인 인과 그래프(causal graph)도 제시하여 향후 연구와 개발을 촉진합니다.

- **Performance Highlights**: 제시된 인과 결정-making 프레임워크를 통해 추천 충실도를 높이고, 특정 추천 시스템 문제에 대한 인과 그래프의 구축 방법을 제시하여 실제 시나리오에서 효과적인 추천이 가능하게 합니다.



### Unified Multi-Modal Interleaved Document Representation for Information Retrieva (https://arxiv.org/abs/2410.02729)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 기존의 Information Retrieval (IR) 방법론이 텍스트 정보에만 의존하는 한계를 극복하기 위해, 텍스트, 이미지, 테이블 등 다양한 모달리티를 포괄하는 문서의 통합 표현을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 개발된 메소드는 최근의 Vision-Language Models (VLMs)를 활용하여, 서로 다른 모달리티가 통합된 단일 형식으로 문서를 처리하고 표현합니다. 이를 통해 문서의 전체 맥락과 다양한 부분 간의 관계를 유지할 수 있으며, 중장 문서의 경우에는 섹션들을 병합하여 단일 문서 표현으로 생성하는 전략을 채택합니다.

- **Performance Highlights**: 제안된 IDentIfy 시스템은 텍스트 전용 및 다중 모달 쿼리를 포함한 다양한 정보 검색 시나리오에서 실험적으로 검증되었으며, 기존의 단일 모달리티를 고려한 방법론들에 비해 월등히 우수한 성능을 보였습니다. 전체 문서의 단일 표현을 통한 검색 효과와 고급화된 섹션 재순위 전략인 reranking의 도입이 성과를 극대화했습니다.



### Domain-Specific Retrieval-Augmented Generation Using Vector Stores, Knowledge Graphs, and Tensor Factorization (https://arxiv.org/abs/2410.02721)
Comments:
          9 pages 7 figures, 1 table, 1 cypher code Accepted to ICMLA 2024

- **What's New**: 새로운 연구에서는 SMART-SLIC라는 highly domain-specific LLM 프레임워크를 소개합니다. 이 프레임워크는 Retrieval-Augmented Generation (RAG)와 Knowledge Graph (KG), 벡터 스토어 (VS)를 통합하여 높은 정확도의 질문 응답 시스템을 구축합니다.

- **Technical Details**: SMART-SLIC는 데이터 마이닝, 자연어 처리 (NLP), 비음수 텐서 분해(nonnegative tensor factorization) 및 자동 모델 선택을 통해 LLM의 환상(hallucinations) 문제를 피할 수 있는 도메인 특화 KG와 VS를 제작합니다. 또한, 이 구조는 구조화된 정보와 비구조화된 정보를 모두 활용하여 효과적인 대화형 봇을 개발합니다.

- **Performance Highlights**: SMART-SLIC는 malware 분석과 이상 탐지를 주제로 한 과학 논문 데이터베이스에서 LLM의 질문 응답 능력을 성공적으로 시연합니다. 이 시스템은 정보 원천의 출처를 명확히 하고, 환상을 완화하며, 미세 조정(fine-tuning)의 필요성을 줄여 줍니다.



### Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers (https://arxiv.org/abs/2410.02642)
- **What's New**: 이 논문은 정보 검색(IR)과 관련하여 새로운 in-context re-ranking (ICR) 방법을 제안합니다. 이 방법은 기존의 생성 기반 접근법이 아니라, LLM의 주의(attention) 패턴의 변화를 활용하여 더욱 정확하고 효율적인 재순위를 제공합니다.

- **Technical Details**: ICR은 쿼리(query)에 의해 유발된 주의 패턴의 변화를 활용하여, 두 번의 전방 패스($O(1)$)만으로 N개의 문서를 재정렬합니다. 이는 기존의 생성 기반 재정렬 방식이 요구하는 $O(N)$ 전방 패스와 비교할 때 현저히 더 효율적입니다. 또한, ICR은 특수한 훈련 없이 어떤 LLM에서도 적용이 가능하며, 잘 정형화된 순위를 보장합니다.

- **Performance Highlights**: 실험 결과 ICR은 여러 정보 검색 벤치마크에서 RankGPT를 능가하며, 실제로 지연(latency)을 60% 이상 줄이는 것으로 나타났습니다. 또한, ICR은 복잡한 재순위 신호를 요구하는 작업에서 특히 강력한 성능을 발휘합니다.



### A GEN AI Framework for Medical Note Generation (https://arxiv.org/abs/2410.01841)
Comments:
          8 Figures, 7 page, IEEE standard research paper

- **What's New**: MediNotes라는 혁신적인 생성 AI 프레임워크가 도입되어, 의학적 대화로부터 SOAP (Subjective, Objective, Assessment, Plan) 노트를 자동으로 생성합니다.

- **Technical Details**: MediNotes는 Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Automatic Speech Recognition (ASR) 기술을 통합하여 실시간 또는 녹음된 오디오에서 텍스트와 음성 입력을 포착하고 처리합니다. 또한 Quantized Low-Rank Adaptation (QLoRA)와 Parameter-Efficient Fine-Tuning (PEFT) 기법을 활용하여 자원이 제한된 환경에서도 효과적인 모델 미세조정을 지원합니다.

- **Performance Highlights**: ACI-BENCH 데이터셋을 이용한 평가 결과, MediNotes는 자동 의료 문서화의 정확성, 효율성, 사용성을 획기적으로 개선하며, 의료 전문가의 행정 부담을 줄이고 임상 작업의 질을 향상시키는 효과적인 솔루션임을 입증했습니다.



### LML-DAP: Language Model Learning a Dataset for Data-Augmented Prediction (https://arxiv.org/abs/2409.18957)
Comments:
          Updated title, abstract, and images

- **What's New**: 본 논문은 설명 가능한 방식으로 분류 작업(Classification tasks)을 수행하기 위해 대형 언어 모델(Large Language Models, LLMs)을 사용하는 새로운 접근 방식을 소개합니다. 기존의 기계 학습(Machine Learning, ML) 모델들이 데이터 정제(Data cleaning)와 특징 엔지니어링(Feature engineering)에 의존하는 것과 달리, 이 방법은 LLMs를 사용하여 프로세스를 간소화합니다.

- **Technical Details**: 이 논문은 '언어 모델 학습(Language Model Learning, LML)'이라는 새로운 개념과 '데이터 증강 예측(Data-Augmented Prediction, DAP)'이라는 방법을 제안합니다. LML 과정에서는 데이터셋을 요약(Summarized)하고 평가하여 각 레이블(classification label)로 이어지는 특징(features)을 결정합니다. DAP 과정에서는 데이터 요약을 바탕으로 테스트 데이터셋의 행(row)을 사용하여 자동으로 쿼리(query)를 생성하고, 이를 통해 데이터셋에서 관련 행을 검색하여 분류를 수행합니다.

- **Performance Highlights**: 제안된 시스템은 복잡한 데이터에서도 만족스러운 정확도를 보장하며, 일부 테스트 사례에서는 90% 이상의 정확도를 기록했습니다. 이는 기존 ML 모델을 다양한 시나리오에서 초월할 잠재력을 보여줍니다.



New uploads on arXiv(cs.CV)

### Flash-Splat: 3D Reflection Removal with Flash Cues and Gaussian Splats (https://arxiv.org/abs/2410.02764)
- **What's New**: 이번 연구에서는 전송(Transmitted) 및 반사(Reflected) 빛을 분리하는 새로운 접근법, Flash-Splat을 소개합니다. 기존의 쌍(pair) 화상 촬영 없이도 플래시(Flash)를 활용해 반사 분리를 수행할 수 있습니다.

- **Technical Details**: Flash-Splat 방법은 최근 개발된 역방향 렌더링 기법인 3D Gaussian Splatting을 활용하여, 플래시 또는 비플래시 상태에서 촬영된 여러 이미지를 사용하여 구성된 ‘가상’ 쌍 화상을 통해 전송 및 반사 장면을 복원합니다.본 연구에서는 플래시가 있는 이미지와 없는 이미지 간의 차이를 바탕으로 강력한 선험적(prior) 정보를 생성하여 반사 분리 문제의 일치성을 크게 향상시킵니다.

- **Performance Highlights**: Flash-Splat은 기존의 3D 반사 분리 방법보다 우수한 성능을 보여주며, 다양한 실제 장면에서 전송 및 반사 장면을 고품질로 재구성할 수 있음을 입증했습니다. 이 방법은 고급 뷰 합성(Novel View Synthesis)과 깊이 추정(Depth Estimation) 작업도 지원합니다.



### Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos (https://arxiv.org/abs/2410.02763)
Comments:
          Project Page: this https URL

- **What's New**: 최근 대형 멀티모달 모델(large multimodal models, LMMs)이 짧은 비디오 이해의 주요 문제를 해결했다고 평가받고 있지만, 실제로는 여러 기본적인 추론 능력이 부족함을 보여줍니다. 이를 개선하기 위해 1000개의 짧고 자연스러운 비디오-자막 쌍을 포함한 새로운 평가 기준인 Vinoground를 소개합니다.

- **Technical Details**: Vinoground는 다양한 동작과 객체 변환의 시간적 차이를 구별하는 능력을 평가합니다. 기존의 모델들이 이러한 능력을 정확히 수행하지 못함을 나타내며, 특히 'single-frame bias' 문제를 해결하기 위해 시간적 반사실(counterfactual) 쌍으로 구성된 데이터를 사용합니다.

- **Performance Highlights**: 대부분의 현대 LMM 모델들은 짧은 비디오 이해에 있어 형편없는 성능을 보이며, 최고 모델인 GPT-4o는 텍스트 및 비디오 점수에서 50% 정도에 불과해 사람의 평균 기준인 90%에 비해 큰 성격 차이를 보입니다. 많은 모델들이 비디오 점수에서 무작위 확률 수준을 기록하였습니다.



### Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations (https://arxiv.org/abs/2410.02762)
Comments:
          Project page and code: this http URL

- **What's New**: 이 연구에서는 Vision-Language Models (VLMs)의 내부 표현을 조사하여 모델의 허위 인식을 줄이는 새로운 접근법을 제안합니다. 고전적인 접근법 외에도, 모델의 잠재적 표현을 직접 수정하여 허위 인식을 줄이는 방법을 제시했습니다.

- **Technical Details**: 우리는 'logit lens' 기법을 사용하여 VLM의 이미지 표현을 언어 어휘로 투사하고, 이러한 표현이 실재하는 객체와 허위로 인식되는 객체에 대해 더 높은 자신감을 나타내는 것을 관찰했습니다. 또한, 'ProjectAway'라는 지식 삭제 알고리즘을 도입하여 허위 인식된 객체의 특징을 선형 직교화하여 제거하는 방식을 사용했습니다.

- **Performance Highlights**: COCO2014 데이터셋에서 모델의 잠재적 표현에 대한 타겟 수정을 통해 허위 인식을 25.7%까지 감소시켰으면서, 이미지 캡셔닝 성능은 유지되었습니다. 또한, 공간적 매핑을 통해 제로샷 세분화(zero-shot segmentation) 성능이 최첨단 방법과 동등함을 보여주었습니다.



### FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models (https://arxiv.org/abs/2410.02761)
- **What's New**: 이 논문에서는 FakeShield라는 다중 모달 프레임워크를 제안하여 이미지의 진위성 평가, 변조된 영역 마스크 생성 및 픽셀 수준과 이미지 수준의 변조 단서를 기반으로 한 판단 근거 제공을 가능하게 합니다. 이는 기존의 이미지 위조 탐지 및 로컬화(IfDL) 방법이 직면한 두 가지 문제인 불투명한 탐지 원리와 다양한 변조 방식에 대한 일반화 한계를 해결합니다.

- **Technical Details**: FakeShield는 설명 가능한 IfDL(e-IFDL) 과제를 기반으로 하며, GPT-4o를 활용하여 다중 모달 변조 설명 데이터셋(MMTD-Set)을 생성합니다. DTE-FDM(영역 태그 안내 설명 가능한 위조 탐지 모듈)과 MFLM(다중 모달 위조 로컬화 모듈)을 통합하여 다양한 변조 탐지 해석을 해결하며, 상세한 텍스트 설명에 기반하여 위조 로컬화를 달성합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FakeShield는 다양한 변조 기법을 효과적으로 탐지하고 로컬라이즈하며, 기존 IfDL 방법들보다 설명 가능하고 우수한 솔루션을 제공함을 보여주었습니다. 이 모델은 copy-move, splicing, removal, DeepFake 및 AIGC 기반 편집과 같은 여러 변조 유형의 탐지와 로컬라이제이션에서 뛰어난 성능을 보였습니다.



### Loong: Generating Minute-level Long Videos with Autoregressive Language Models (https://arxiv.org/abs/2410.02757)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 오토회귀(autoregressive) 대형 언어 모델(LLMs)을 기반으로 하는 새로운 비디오 생성기인 Loong을 제안하며, 이를 통해 1분 길이의 콘텐츠가 풍부한 롱 비디오(long video)를 생성할 수 있는 가능성을 탐구합니다.

- **Technical Details**: Loong은 비디오 토큰과 텍스트 토큰을 통합된 시퀀스로 모델링하여 훈련하며, 훈련 초기에는 짧은 비디오를, 점진적으로 길어지는 비디오를 훈련하는 프로그레시브(short-to-long) 전략을 사용합니다. 이와 함께 손실 불균형 문제를 완화하기 위해 손실 재가중(loss re-weighting) 기법을 도입하였습니다. 추론 과정에서는 비디오 토큰 재인코딩(video token re-encoding) 및 샘플링 전략(sampling strategies)을 사용하여 오류 축적을 줄이는 방법을 연구합니다.

- **Performance Highlights**: Loong 모델은 10초 길이의 비디오에서 훈련하여, 텍스트 프롬프트(text prompts)에 따라 1분 이상 길이의 비디오를 생성할 수 있음을 실험 결과를 통해 입증하였습니다. 생성된 롱 비디오는 일관성과 시각적 품질이 우수한 것으로 나타났습니다.



### Contrastive Localized Language-Image Pre-Training (https://arxiv.org/abs/2410.02746)
Comments:
          Preprint

- **What's New**:  본 논문에서는 Contrastive Language-Image Pre-training (CLIP)의 지역화(localization) 능력을 향상시키기 위한 새로운 방법인 Contrastive Localized Language-Image Pre-training (CLOC)을 제안합니다. CLOC는 지역-텍스트 대조 손실(region-text contrastive loss)과 모듈을 활용하여 CLIP의 기존 기능을 보완합니다.

- **Technical Details**: CLOC는 'promptable embeddings'라는 개념을 도입하여 이미지 인코더가 공간적 힌트(spatial hints)에 따라 쉽게 지역 표현으로 변환될 수 있는 이미지 임베딩을 생성하도록 합니다. 이를 통해 CLIP의 이미지-텍스트 대비 손실을 강화하여, 지역-텍스트의 명시적 감독을 회복하는 새로운 전이 방법론을 구축합니다. 또한, 시각적으로 풍부하고 공간적으로 지역화된 캡셔닝 프레임워크를 설계하여 대규모로 지역-텍스트의 의사 레이블(pseudo-labels)을 효과적으로 생성합니다.

- **Performance Highlights**:  실험 결과 CLOC는 이미지 지역 인식 및 검색(task)과 같은 임무에서 높은 품질의 지역 임베딩을 제공합니다. 총 31가지 평가 작업을 통해 CLIP보다 CLOC가 일관되게 우수한 성능을 보임을 입증하였습니다. 또한, 최근의 다양한 다중 모달 대형 언어 모델(MLLMs)에서 특히 참고(r efering) 및 접지(grounding) 작업에서 성능이 향상되었습니다.



### AVG-LLaVA: A Multimodal Large Model with Adaptive Visual Granularity (https://arxiv.org/abs/2410.02745)
Comments:
          Preprint

- **What's New**: AVG-LLaVA(Adaptive Visual Granularity LLaVA)는 입력 이미지와 지시에 따라 적절한 visual granularity(시각적 세분성)를 선택할 수 있는 LMM(대형 다중모드 모델)입니다. 이를 통해 visual token(시각적 토큰)의 수를 줄이고 추론 속도를 향상시키며 모델 성능을 개선합니다.

- **Technical Details**: AVG-LLaVA는 시각적 세분성 스케일러와 시각적 세분성 라우터라는 두 가지 모듈을 포함하고 있습니다. 시각적 세분성 스케일러는 visual tokens에 대해 여러 번의 pooling layer(풀링 레이어)를 진행하여 서로 다른 세분성의 visual tokens를 얻습니다. 시각적 세분성 라우터는 입력된 다중 세분성 visual feature와 text feature를 기반으로 적절한 visual granularity를 선택합니다. RGLF(Ranking Granularity to align LMM Feedback)라는 새로운 훈련 패러다임도 도입되어 라우터의 예측을 LMM의 선호와 일치시키도록 돕습니다.

- **Performance Highlights**: AVG-LLaVA는 11개의 벤치마크에서 우수한 성능을 달성하였으며, visual token 수를 85.3% 줄이고 추론 속도를 2.53배 증가시키는 등 성능을 크게 개선했습니다.



### Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models (https://arxiv.org/abs/2410.02740)
Comments:
          CV/ML

- **What's New**: 최근 인공지능 분야에서 멀티모달 모델(multimodal models)의 발전이 캡션(captions) 재작성을 통한 성능 향상의 중요성을 강조하고 있습니다. 본 연구에서는 다양한 멀티모달 모델에 맞춘 캡션 생성 파이프라인을 제안하며, 이는 합성 캡션과 원본 AltText(대체 텍스트)의 상호작용을 탐구합니다.

- **Technical Details**: 우리의 연구에서는 Short Synthetic Captions (SSC) 및 Descriptive Synthetic Captions (DSC+)를 사례로 활용하여 합성 캡션이 다양한 멀티모달 모델(CLIP, 멀티모달 LLMs)과의 상호작용을 가지고 어떻게 작용하는지를 분석합니다. 우리는 고급 캡션 포맷의 필요성을 강조하며, 원본 AltText와 합성 캡션의 혼합 접근 방식이 더 좋은 성능을 발휘하는 것을 발견했습니다.

- **Performance Highlights**: 연구 결과, 합성 캡션만을 사용할 때보다 합성 캡션과 AltText를 혼합하여 사용할 경우 성능이 향상되며, 각 모델이 특정 캡션 포맷에 대한 선호가 있음을 보여줍니다. 이러한 통합적 접근 방식은 멀티모달 모델의 전처리(pre-training) 과정에서 유용한 최적화 전략으로 기여할 수 있습니다.



### DivScene: Benchmarking LVLMs for Object Navigation with Diverse Scenes and Objects (https://arxiv.org/abs/2410.02730)
Comments:
          Work in Progress

- **What's New**: 본 논문에서는 다양한 대상 객체를 탐색할 수 있는 내비게이션 에이전트 구축을 위한 새로운 작업을 연구합니다. 이를 위해 4,614개의 씬으로 구성된 대규모 씬 데이터셋, DivScene을 소개합니다.

- **Technical Details**: NatVLM(Navigational Chain-of-Thought VLM)이라고 불리는 내비게이션 에이전트를 구축하였으며, 이를 위해 Large Vision Language Model(사례로 Idefics 2)과 모방 학습(imitation learning)을 활용했습니다. 이러한 접근 방식은 기존의 고인물 학습과의 도메인 갭(domain gap)을 줄이는 데 기여합니다.

- **Performance Highlights**: 제안한 NatVLM 에이전트는 BFS (너비 우선 탐색) 계획자를 통해 구축된 최단 경로에서의 모방 학습으로 훈련되었으며, GPT-4o를 초과하는 20% 이상의 성공률을 달성했습니다.



### Curvature Diversity-Driven Deformation and Domain Alignment for Point Cloud (https://arxiv.org/abs/2410.02720)
- **What's New**: 새로운 접근법으로 제안된 CDND( Curvature Diversity-Driven Nuclear-Norm Wasserstein Domain Alignment)는 소스 도메인과 타겟 도메인 간의 도메인 간극을 효과적으로 줄이는 것을 목표로 하고 있습니다.

- **Technical Details**: CDND는 두 가지 주요 구성 요소를 가지며, 첫 번째는 Curvature Diversity 기반의 Deformation Reconstruction 방법인 CurvRec이고, 두 번째는 Deformation 기반의 Nuclear-norm Wasserstein Discrepancy(D-NWD)입니다. CurvRec는 포인트 클라우드의 의미론적으로 풍부한 영역에서 두드러진 특징을 추출하도록 모델을 유도함으로써 소스와 타겟 도메인 간의 차이를 줄입니다. D-NWD는 원본 및 변형된 데이터 샘플 양쪽에서 Nuclear-norm Wasserstein Discrepancy를 적용하여 도메인을 정렬합니다.

- **Performance Highlights**: 실험 결과, CDND는 두 개의 공공 도메인 적응 데이터셋에서 포인트 클라우드 분류 및 분할 작업을 수행하기 위한 실험을 통해 기존 방법들보다 상당히 우수한 성능을 보였습니다.



### Video Instruction Tuning With Synthetic Data (https://arxiv.org/abs/2410.02713)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서 제안하는 LLaVA-Video-178K 데이터셋은 비디오 지침 따르기를 위한 고품질 합성 데이터셋으로, 다양한 비디오 작업을 위한 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: LLaVA-Video-178K는 178,510개의 비디오로 구성되어 있으며, 다양한 질문 유형을 제공하는 태스크를 포함합니다. 주요 속성으로는 동적 비디오 소스의 선택, 자세한 캡션 생성 파이프라인 및 다양한 질문-답변 쌍 생성을 위한 GPT-4o 사용이 있습니다.

- **Performance Highlights**: LLaVA-Video는 다양한 비디오 벤치마크에서 뛰어난 성능을 발휘하며, 기존 단일 프레임 기반 학습이 아닌 다중 프레임 기반 접근 방식이 효과적임을 입증했습니다.



### LLaVA-Critic: Learning to Evaluate Multimodal Models (https://arxiv.org/abs/2410.02712)
Comments:
          Project Page: this https URL

- **What's New**: LLaVA-Critic는 최초의 오픈소스 대규모 다중 모달 모델로, 다양한 다중 모달 작업의 성과를 평가하는 일반 평가자로 설계되었습니다. 이 모델은 고품질의 비평가 지침을 따르는 데이터셋을 사용하여 훈련되었으며, 효과적인 평가 점수를 제공하는 능력이 입증되었습니다.

- **Technical Details**: LLaVA-Critic는 (1) LMM-as-a-Judge와 (2) Preference Learning의 두 가지 주요 시나리오에서 실험이 수행되었습니다. LMM-as-a-Judge 시나리오에서는 LLaVA-Critic가 신뢰할 수 있는 평가 점수를 제공하며, 여러 평가 벤치마크에서 GPT 모델들과 유사한 성능을 보입니다. Preference Learning에서는 선호 학습을 위한 보상 신호를 생성하여 모델 정렬 능력을 향상시킵니다.

- **Performance Highlights**: LLaVA-Critic는 상업용 GPT 모델과 높은 상관관계를 보이며, 자원 제한 환경에서도 모델 개발자에게 비용 효과적인 대안을 제공합니다. 또한, AI 생성 피드백을 통한 선호 학습에서 LLaVA-RLHF 모델보다 우수한 성능을 나타냈습니다.



### SteerDiff: Steering towards Safe Text-to-Image Diffusion Models (https://arxiv.org/abs/2410.02710)
- **What's New**: 이 논문에서는 SteerDiff라는 경량 어댑터 모듈을 소개하며, 이는 사용자 입력과 diffusion 모델 간의 중개자로 작동하여 생성된 이미지가 윤리적 및 안전 기준을 준수하도록 보장합니다.

- **Technical Details**: SteerDiff는 텍스트 임베딩 공간 내에서 부적절한 개념을 식별하고 조작하여 모델이 유해한 출력을 생성하는 것을 방지합니다. 이 모듈은 두 단계의 경량 어댑터 모델로, 텍스트 프롬프트 임베딩을 유도하는 데 중점을 두고, 안전한 콘텐츠와 안전하지 않은 콘텐츠를 최대한 구분하는 의미 경계를 구성합니다.

- **Performance Highlights**: 실험 결과 SteerDiff는 이미지 품질과 의미적 충실도를 유지하면서 부적절한 콘텐츠 생성을 현저히 줄이는 것으로 나타났습니다. 또한 P4D 및 SneakyPrompt와 같은 레드 팀 공격에 대한 방어에서 효과적인 성능을 보여주었습니다.



### ControlAR: Controllable Image Generation with Autoregressive Models (https://arxiv.org/abs/2410.02705)
Comments:
          Preprint. Work in progress

- **What's New**: 이번 논문에서는 Autoregressive (AR) 모델을 이용한 이미지 생성에 대한 새로운 접근 방식인 ControlAR를 제안합니다. ControlAR는 ControlNet과 유사한 이미지를 생성하는 데 사용되는 기존의 접근 방식과 비교하여 효율성과 효과성을 갖춘 프레임워크입니다.

- **Technical Details**: ControlAR의 핵심은 제어 인코더를 통해 공간적 입력(예: canny edge, depth map)을 제어 토큰으로 변환하는 것입니다. 이후, Conditional Decoding 방식을 통해 이미지 토큰과 제어 토큰 간의 융합 이 이루어지며, 이를 통해 다음 이미지 토큰을 생성합니다. 이 프로세스는 기존의 prefilling 기법보다 훨씬 더 나은 성능을 제공합니다.

- **Performance Highlights**: ControlAR를 적용하여 실행한 다양한 실험 결과는 AR 모델이 제어할 수 있는 능력과 생성된 이미지의 품질이 기존의 최첨단 확산 모델인 ControlNet++를 초과함을 보여줍니다. 또한, ControlAR는任意 해상도의 이미지 생성을 가능하게 하여 다양한 크기의 고품질 이미지를 생성할 수 있는 능력이 검증되었습니다.



### Unsupervised Point Cloud Completion through Unbalanced Optimal Transpor (https://arxiv.org/abs/2410.02671)
Comments:
          20 pages, 10 figures

- **What's New**: 이 논문에서는 Unbalanced Optimal Transport Map을 기반으로 하는 새로운 비쌍 (unpaired) 포인트 클라우드 완성 모델인 UOT-UPC를 제안합니다. 이는 비쌍 포인트 클라우드 완성 문제를 최적 운송 (Optimal Transport) 문제로 해석하고, 클래스 불균형 문제를 해결하는 데 초점을 맞춥니다.

- **Technical Details**: UOT-UPC는 주어진 불완전한 포인트 클라우드를 기반으로 하는 최적 운송 맵 (Optimal Transport Map)을 학습하여 적용되며, InfoCD 비용 함수 (cost function)가 이 작업에 가장 적합한 것으로 분석됩니다. 이 모델은 서로 다른 샘플링에 의한 불완전한 포인트 클라우드와 완전한 포인트 클라우드를 이용해 훈련됩니다.

- **Performance Highlights**: UOT-UPC는 비쌍 포인트 클라우드 완성에서 뛰어난 성능을 보이며, 특히 클래스 불균형이 존재하는 상황에서 강력한 성능을 발휘합니다. 단일 카테고리와 다중 카테고리 데이터셋 모두에서 경쟁력 있는 결과를 달성했습니다.



### Learning 3D Perception from Others' Predictions (https://arxiv.org/abs/2410.02646)
Comments:
          Under review

- **What's New**: 이 논문은 고정밀 3D 객체 탐지를 위해 인근의 정밀 탐지기를 갖춘 유닛의 예측을 학습하는 새로운 시나리오를 제안합니다. 이 방법은 레이블 효율적(label-efficient)이고, 센서에 구애받지 않으며(sensor-agnostic), 통신 효율적(communication-efficient)입니다.

- **Technical Details**: 저자는 거리 기반 커리큘럼(curriculum)을 적용하여 가까운 유닛의 예측을 먼저 학습하고, 이후 자체 학습(self-training)을 통해 다른 유닛의 예측 품질을 향상시키는 방법을 제안합니다. 또한, 제한된 주석 데이터로도 효과적인 의사 레이블(pseudo label) 정제 모듈을 훈련할 수 있다고 주장합니다.

- **Performance Highlights**: 최근 발표된 실제 협업 주행 데이터셋에서 실험을 통해 제안한 접근 방식이 3D 인식을 위한 라벨 효율적 학습에 얼마나 효과적인지를 입증했습니다. 다양한 시나리오(예: 센서 및 탐지기 다양성)에서 실험을 통해 성공적인 성과를 거두었습니다.



### Spatial-Temporal Multi-Cuts for Online Multiple-Camera Vehicle Tracking (https://arxiv.org/abs/2410.02638)
- **What's New**: 이 논문에서는 온라인 다중 카메라 차량 추적에서의 새로운 그래프 표현 방식을 제안합니다. 기존의 다단계 절차 대신, 시간-공간적으로 군집화하여 탐지를 연결하는 단일 단계의 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 기존의 군집을 기반으로 새로운 탐지를 연결하는 그래프 표현 방식을 사용하여, 모든 탐지에서 희소한 외관(appearance) 및 위치적 단서를 유지합니다. 이로 인해 강력한 증거를 바탕으로 클러스터를 비교할 수 있으며, 멀티컷(multicut) 할당 절차를 사용하여 최종 추적을 온라인으로 생성할 수 있습니다.

- **Performance Highlights**: CityFlow 데이터셋에서 IDF1 기준으로 기존 온라인 방법보다 14% 이상 성능이 향상되었으며, Synthehicle 데이터셋에서는 25% 이상 뛰어난 성능을 보였습니다. 또한, 다양한 카메라 배치에 대해 잘 일반화되는 특징을 가지고 있습니다.



### Metrics Revolutions: Groundbreaking Insights into the Implementation of Metrics for Biomedical Image Segmentation (https://arxiv.org/abs/2410.02630)
- **What's New**: 이 연구에서는 11개의 오픈소스 도구의 거리 기반 메트릭(distance-based metrics) 계산을 비교하여 설계된 기준 구현(reference implementation)에 대한 체계적 분석을 수행했습니다. 그 결과, 모든 도구 간에 통계적으로 의미 있는 차이가 존재함을 발견하였으며, 이는 기존 연구의 유효성을 의문시하게 만듭니다.

- **Technical Details**: 본 연구는 HD(Hausdorff Distance), MASD(Mean Average Surface Distance), ASSD(Average Symmetric Surface Distance), NSD(Normalized Surface Distance) 및 BIoU(Boundary Intersection over Union)와 같은 여러 거리 기반 메트릭의 수학적 정의와 한계에 대해 설명하고, 이를 기준으로 각각의 오픈소스 도구의 계산 과정을 분석합니다.

- **Performance Highlights**: 연구를 통해 발견된 결과는 오픈소스 도구 간에 메트릭 점수에서 발생하는 체계적인 구현 오류와 통계적으로 의미 있는 차이가 있으며, 이는 의료 영상 커뮤니티에 있어 거리 기반 메트릭 계산의 적절한 해석과 사용에 대한 권장사항을 제시합니다.



### GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering (https://arxiv.org/abs/2410.02619)
- **What's New**: GI-GS는 3D Gaussian Splatting(3DGS)과 deferred shading을 이용한 새로운 inverse rendering 프레임워크로, 사실적인 novel view synthesis와 relighting을 달성합니다. 이 방법은 indirect lighting을 효율적으로 path tracing으로 계산하여, 이전 기술의 한계를 극복합니다.

- **Technical Details**: GI-GS는 먼저 G-buffer를 렌더링하여 장면의 세부 기하학적 구조와 재료 특성을 캡처합니다. 그 후, PBR(Physically Based Rendering)을 사용하여 직접 조명을 계산하고, 이전 렌더링 결과를 통해 경량 path tracing으로 indirect lighting을 계산합니다. 이 과정에서 우리가 사용한 deferred shading 기법은 반드시 조명의 동적 성격을 반영하여 indirect lighting의 작용을 과학적으로 모델링합니다.

- **Performance Highlights**: GI-GS는 렌더링 품질과 효율성 측면에서 기존의 기초 모델들보다 월등한 결과를 보여주며, 복잡한 실제 장면에서 높은 충실도의 기하학적 구조와 재료를 재구성할 수 있습니다. 정량적 및 정성적 결과 모두에서 기존의 방법들을 초월하는 성과를 달성했습니다.



### NL-Eye: Abductive NLI for Images (https://arxiv.org/abs/2410.02613)
- **What's New**: NL-Eye라는 새로운 벤치마크를 소개하여 VLM(Visual Language Model)의 시각적 외적 추론(abductive reasoning) 능력을 평가하고자 함.

- **Technical Details**: NL-Eye는 350개의 트리플릿 예제를 포함하며, 각 사례는 전제 이미지와 가설 이미지들로 구성됨. VLM은 주어진 전제 이미지에 따라 가설 이미지의 그럴듯함(plausibility)을 평가해야 하며 이는 물리적, 논리적, 감정적, 기능적, 문화적, 사회적 범주로 나뉨. 이 과정에서 생성된 이미지는 텍스트에서 이미지로 변환하는 모델을 통해 얻어짐.

- **Performance Highlights**: NL-Eye에서 VLM은 무작위 기준 수준의 성과를 보이며, 인간들은 85%의 정확도로 더 그럴듯한 가설을 선택하고 94%에서 유효한 설명을 제공. 하지만 VLM은 정확한 설명을 제공하는 데 있어 50% 이상 실패하며, 이는 현대 VLM의 외적 추론 능력에 큰 결함을 나타냄.



### IC3M: In-Car Multimodal Multi-object Monitoring for Abnormal Status of Both Driver and Passengers (https://arxiv.org/abs/2410.02592)
Comments:
          16 pages, 17 figures

- **What's New**: 최근, 차량 내 모니터링 기술이 운전자의 초기 비정상 상태를 감지하고 교통사고를 예방하기 위한 알림을 제공하는데 매우 유망한 기술로 주목받고 있습니다. 본 논문에서는 운전사와 승객을 동시에 모니터링할 수 있는 효율적인 카메라 회전 기반 멀티모달 프레임워크인 IC3M을 소개합니다.

- **Technical Details**: IC3M은 두 가지 주요 모듈로 구성되어 있습니다: 1) Adaptive Threshold Pseudo-labeling 전략은 클래스별 분포에 따라 가짜 레이블의 임계치를 조정하여 클래스 균형을 이루는 레이블을 생성합니다. 2) Missing Modality Reconstruction 모듈은 한정된 레이블에서 학습된 크로스모달리티 관계를 활용하여 누락된 모달리티를 복원합니다.

- **Performance Highlights**: IC3M은 정확도, 정밀도 및 재현율 측면에서 최신 벤치마크를 초월하며, 레이블이 제한된 데이터와 심각한 누락 모달리티 환경에서도 뛰어난 견고성을 보여줍니다.



### An Improved Variational Method for Image Denoising (https://arxiv.org/abs/2410.02587)
- **What's New**: 이번 논문에서는 이미지 노이즈 제거를 위한 개선된 총 변동(TV) 모델과 관련 수치 알고리즘을 제안합니다. 이 모델은 여러 유형의 노이즈 및 그 조합을 효과적으로 제거하는 데 뛰어난 성능을 보여줍니다.

- **Technical Details**: 총 변동(TV) 방법은 이미지의 픽셀 강도의 변동을 최소화함으로써 노이즈를 줄이는 이미지 노이즈 제거 기법입니다. 제안된 새로운 모델은 고유한 해(solution)를 보장하며, 수치 알고리즘의 수렴(convergence)이 이론적으로 보장됩니다. 네 가지 주요 그룹(공간 도메인 필터링, 변환 도메인 필터링, 변형 기법 및 학습 기반 방법)으로 나누어지는 노이즈 제거 방법론을 사용하며, 특히 TV 기반 방법은 최적화 문제로 구성되어 있습니다.

- **Performance Highlights**: 수치 실험 결과, 제안된 모델은 기존의 TV 모델들과 비교하여 뛰어난 효과성과 노이즈 제거 품질을 확인하였습니다. 이는 TV 방법의 이미지 처리에서의 활용도를 더욱 높이는 결과를 보여줍니다.



### SuperGS: Super-Resolution 3D Gaussian Splatting via Latent Feature Field and Gradient-guided Splitting (https://arxiv.org/abs/2410.02571)
- **What's New**: 본 논문에서는 Super-Resolution 3D Gaussian Splatting (SuperGS)라는 새로운 방법을 제안합니다. 이 방법은 저해상도 입력 장면 표현을 초기화로 사용하는 두 단계의 coarse-to-fine 훈련 프레임워크를 이용해 성능을 크게 향상시킵니다.

- **Technical Details**: SuperGS는 Multi-resolution Feature Gaussian Splatting (MFGS)와 Gradient-guided Selective Splitting (GSS) 전략을 통합하여 고해상도 장면을 효과적으로 렌더링합니다. MFGS는 잠재적 피처 필드를 활용하여 유연한 피처 샘플링을 지원하고, GSS는 효율적인 가우시안 업샘플링을 수행합니다. 이 두 가지 접근법은 메모리 효율성을 유지하면서도 높은 정확도를 확보하는 데 기여합니다.

- **Performance Highlights**: SuperGS는 실제 세계의 도전적인 데이터셋에서 기존의 고해상도 새로운 시각 합성 (HRNVS) 방법들을 초월하는 성능을 보여주었으며, 저해상도 입력만을 사용하여도 우수한 결과를 도출하였습니다.



### Pseudo-Stereo Inputs: A Solution to the Occlusion Challenge in Self-Supervised Stereo Matching (https://arxiv.org/abs/2410.02534)
Comments:
          Submitted to IEEE Transactions on Image Processing (TIP)

- **What's New**: 본 논문에서는 자가 감독 스테레오 매칭(self-supervised stereo matching) 효율성을 향상시키기 위한 새로운 해결책으로, occlusion 문제를 해결하기 위한 간단하면서도 효과적인 pseudo-stereo 입력 전략을 제안합니다. 이 전략은 입력 이미지와 피드백 이미지를 분리하고, 네트워크가 occluded 객체 양쪽에서 정보를 확률적으로 샘플링하도록 유도합니다.

- **Technical Details**: 제안된 pseudo-stereo 입력 전략은 occluded 영역에서의 지속적인 정보 부족 문제를 완화하고, 피드백 충돌(feedback conflicts) 및 과적합 문제를 해결하기 위해 추가 조치를 통합합니다. 이를 통해 기존 방법들에 비해 안정적이고 유의한 성능 향상을 달성합니다. 실험을 통해 정량적 및 정성적 평가 결과를 제시하며, occluded 영역에서도 정확한 disparity 추론이 가능함을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 자가 감독 스테레오 매칭 기법들보다 성능이 개선되었으며, 특히 occlusion 문제로 인하여 발생하는 예측 오류를 줄이는 데 성공했습니다. 이 접근 방식은 특히 DispNetC와 같은 감독된 방법과 경쟁력 있는 성능을 달성하면서, 어떠한 사전 훈련된 모델이나 후처리 기술 없이도 우수한 성능을 입증했습니다.



### HiFiSeg: High-Frequency Information Enhanced Polyp Segmentation with Global-Local Vision Transformer (https://arxiv.org/abs/2410.02528)
- **What's New**: 본 논문에서는 높은 주파수(high-frequency) 정보를 효과적으로 처리하여 점막 용종(colon polyp) 분할(segmentation) 성능을 향상시키는 HiFiSeg라는 새로운 네트워크 구조를 소개합니다. 이 네트워크는 글로벌-로컬 비전 변환기(global-local vision transformer) 프레임워크를 활용하여 복잡한 이미지에서도 세부 사항을 잘 포착할 수 있도록 설계되었습니다.

- **Technical Details**: HiFiSeg는 피라미드 비전 변환기(pyramid vision transformer, PVT)를 인코더로 사용하며, 글로벌-로컬 상호작용 모듈(global-local interaction module, GLIM)과 선택적 집계 모듈(selective aggregation module, SAM)을 도입합니다. GLIM은 여러 스케일에서 글로벌 및 로컬 정보를 융합하여 세부 특징을 효과적으로 수집합니다. SAM은 고수준 특징에서 저수준 경계 세부 정보를 선택적으로 통합하여 모델의 용종 탐지 및 분할 성능을 개선합니다.

- **Performance Highlights**: CVC-ColonDB 및 ETIS 데이터셋에서 mDice 점수가 각각 0.826 및 0.822에 달하며, 이는 기존 최첨단 모델들을 초월하는 성능을 보여줍니다. 이는 HiFiSeg가 용종 분할과 같은 복잡한 의료 이미지 분할 작업에서 뛰어난 효과를 발휘함을 나타냅니다.



### Learning from Offline Foundation Features with Tensor Augmentations (https://arxiv.org/abs/2410.02527)
Comments:
          Accepted to the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문에서는 제한된 자원 환경에서 foundation 모델의 효율적 학습을 위한 새로운 접근법인 LOFF-TA(Offline Foundation Features with Tensor Augmentations)를 소개합니다. LOFF-TA는 frozen foundation 모델로부터 캐시된 feature embeddings를 활용하여 compact classifier를 학습함으로써 훈련 속도를 최대 37배 향상시키고 GPU 메모리 사용량을 최대 26배 낮춥니다.

- **Technical Details**: LOFF-TA는 원본 non-augmented 이미지의 캐시된 embeddings에 tensor augmentations를 적용하여, 그림 1에서 보여지는 것처럼 훈련 데이터를 foundation 모델을 통해 전처리 하고, 그 결과로 저장된 feature embeddings을 사용하여 lightweight classifier를 학습하는 방식입니다. 이 방법은 단일 파라미터 없이 foundation 모델과 훈련 과정의 완전한 분리를 탐구하였습니다.

- **Performance Highlights**: LOFF-TA는 훈련 속도의 대폭적인 향상과 메모리 사용량의 감소를 통해 효율적으로 foundation 모델을 활용할 수 있으며, 일부 경우에서는 fine-tuned foundation 모델보다 더 나은 성과를 내기도 합니다. LOFF-TA는 고해상도 이미지에도 적용 가능하여, 의료 이미지 진단과 같은 특정 작업에서 자원 증가 없이 foundation 모델의 능력을 사용할 수 있게 해줍니다.



### Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessmen (https://arxiv.org/abs/2410.02505)
Comments:
          10 pages, 5 figures. The code and models will be available at this https URL

- **What's New**: 이번 연구는 훈련이 필요 없는 Dog-IQA라는 새로운 이미지 품질 평가(IQA) 방법을 제안합니다. 이 방법은 멀티모달 대형 언어 모델(MLLM)의 사전 지식을 활용하며, 사람 전문가의 평가 방식을 모방하여 정확한 IQA 점수를 추출합니다.

- **Technical Details**: Dog-IQA는 두 가지 주요 기술을 활용하여 IQA 점수를 수집합니다. 첫째, MLLM의 행동 패턴을 활용하여 객관적인 점수를 부여하며 주관적인 요인의 영향을 최소화합니다. 둘째, 로컬(지역) 및 글로벌(전체) 정보를 활용하기 위해 이미지 전체와 지역 구문 객체를 입력으로 받고 이들의 점수를 집계합니다.

- **Performance Highlights**: Dog-IQA는 훈련이 필요 없는 방법으로서 현존하는 최고 수준의 성능(SOTA)을 달성하였으며, 훈련 기반 방법과 비교해도 경쟁력 있는 성능을 보였습니다. 다양한 데이터 세트를 통해 실험을 진행하여 이 방법의 효용성을 입증하였습니다.



### DTVLT: A Multi-modal Diverse Text Benchmark for Visual Language Tracking Based on LLM (https://arxiv.org/abs/2410.02492)
Comments:
          Preprint, Under Review

- **What's New**: 이번 연구에서는 다양한 텍스트를 생성할 수 있는 대형 언어 모델(LLM)을 활용하여, 시각 언어 추적(Visual Language Tracking, VLT)을 위한 새로운 벤치마크인 DTVLT를 제안합니다. 이 벤치마크는 전통적인 단일 객체 추적(Single Object Tracking, SOT) 작업의 범위를 확장하여 비디오 이해 애플리케이션을 포함하도록 합니다.

- **Technical Details**: DTVLT는 다섯 개의 저명한 VLT 및 SOT 벤치마크를 기반으로 하며, 짧은 기간 추적, 장기 추적, 전역 인스턴스 추적 등 세 가지 하위 작업으로 구성됩니다. 벤치마크에 포함된 네 가지 서로 다른 세부정보 수준의 텍스트는 의미론적 정보의 범위와 밀도를 고려하여 생성됩니다. 텍스트 생성을 위한 DTLLM-VLT 메소드를 활용합니다.

- **Performance Highlights**: 실험 분석을 통해, 다양한 텍스트가 추적 성능에 미치는 영향을 평가하였으며, 기존 알고리즘의 성능 병목 현상을 파악하여 VLT와 비디오 이해 연구의 발전을 지원할 수 있는 기초 자료를 제공합니다.



### Event-Customized Image Generation (https://arxiv.org/abs/2410.02483)
- **What's New**: 새로운 'event-customized 이미지 생성(event-customized image generation)' 작업 제안. 이를 통해 단일 참조 이미지로부터 다양한 엔티티 간의 복잡한 사건을 포착하고 커스터마이즈된 이미지를 생성할 수 있다.

- **Technical Details**: FreeEvent라는 훈련이 필요 없는 이벤트 커스터마이징 방법을 제안. 1) Entity switching path: 목표 엔티티 생성을 위한 cross-attention 가이드와 조정. 2) Event transferring path: 참조 이미지에서 공간 피처와 self-attention 맵을 목표 이미지로 주입하여 이벤트 생성을 돕는다.

- **Performance Highlights**: FreeEvent는 복잡한 및 창의적인 커스터마이징을 가능하게 하여 기존 방법들에 비해 우수한 성능을 보인다. SWiG-Event와 Real-Event라는 두 가지 평가 벤치마크를 통해 검증되었다.



### Recurrent Few-Shot model for Document Verification (https://arxiv.org/abs/2410.02456)
- **What's New**: 이번 연구에서는 일반적인 ID 및 여행 문서의 이미지 및 비디오 기반 검증 시스템의 성과 향상에 초점을 맞추고 있습니다. 특히 보지 못한 클래스의 문서에 대해 아주 적은 수의 예로도 사기 문서를 탐지할 수 있는 순환 기반의 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 순환 많이-하나 네트워크 구조를 갖고 있으며, 문서 이미지를 개별 패치로 나누어 처리한 후 특징 벡터를 생성합니다. 이 모델은 few-shot learning (FSL) 전략을 사용하여 지원 집합과 질의 집합에서 이미지를 분류합니다. Conditional FSL과 Unconditional FSL 두 가지 전략을 통해 훈련됩니다.

- **Performance Highlights**: SIDTD 및 Findit 데이터세트에서 실시한 초기 결과는 제안된 모델이 이러한 과제에 대해 우수한 성능을 보임을 보여줍니다. 특히 새로운 클래스의 문서에 대해서도 잘 일반화되는 능력을 갖추었습니다.



### Clinnova Federated Learning Proof of Concept: Key Takeaways from a Cross-border Collaboration (https://arxiv.org/abs/2410.02443)
- **What's New**: Clinnova는 다국적 협력 프로젝트로, 정밀 의학의 힘을 해제하기 위해 데이터를 연합하고 표준화하며 상호 운용성을 추구합니다. 이 프로젝트는 유럽 표준을 만들기 위해 인공지능(AI)과 데이터 과학을 활용하여 의료 결과 및 효율성을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: Clinnova는 소아 편타염, 류마티스 질환 및 다발성 경화증(MS)을 대상으로 삼고 있으며, 데이터 품질을 중시하여 AI 알고리즘 개발에 기여합니다. 중요 기술로는 Federated Learning(FL)이 있으며, 이 방식은 전국적으로 분산된 클라이언트에서 데이터 모델을 학습할 수 있게 해 주며, 데이터 프라이버시를 보장하면서 모델 훈련을 수행할 수 있게 합니다.

- **Performance Highlights**: Clinnova-MS에서는 MS 환자 진료 향상에 초점을 두고 FL 기술을 활용하여 질병 진행 상황을 감지하는 더 정확한 모델을 개발하고 있습니다. 초국경에서의 협력을 통해 이루어진 MRI 영상의 MS 세분화를 위한 POC 프로젝트는 중요한 이정표로 자리잡고 있으며, FL의 의료 적용 가능성을 보여줍니다.



### PnP-Flow: Plug-and-Play Image Restoration with Flow Matching (https://arxiv.org/abs/2410.02423)
- **What's New**: 본 논문에서는 이미징 역문제(imaging inverse problems)를 해결하기 위한 Plug-and-Play (PnP) Flow Matching 알고리즘을 소개합니다. 이 알고리즘은 사전 훈련된 딥 신경망(deep neural networks) 기반의 디노이저(denoiser)를 활용하여 최적화 방식에 통합합니다.

- **Technical Details**: PnP 방법은 여러 이미징 역문제에서 최첨단 성과를 달성하였지만, 생성적(generative) 작업인 인페인팅(inpainting)과 같은 특정 문제에서는 한계를 가지고 있습니다. 제안하는 방법은 PnP 프레임워크와 Flow Matching (FM)을 결합하여 사전 훈련된 FM 모델을 사용해 시간 의존적 디노이저를 정의합니다. 이 알고리즘은 데이터 충실도(data-fidelity) 항목에서의 경량 경량 하강(gradient descent) 단계, 학습한 FM 경로에 대한 재투영(reprojection), 그리고 디노이징을 번갈아 수행합니다. 이 방법은 ODE(Ordinary Differential Equation)을 통한 역전파(backpropagation)와 추적(trace) 계산을 피함으로써 계산 효율성과 메모리 친화성을 제공합니다.

- **Performance Highlights**: 실험 결과, 디노이징, 초고해상도(super-resolution), 디블러링(deblurring), 인페인팅 작업에서 기존 PnP 알고리즘 및 Flow Matching 기반의 최첨단 방법에 비해 우수한 결과를 보여주었습니다.



### LoGDesc: Local geometric features aggregation for robust point cloud registration (https://arxiv.org/abs/2410.02420)
- **What's New**: 본 논문은 3D 포인트 매칭 및 포인트 클라우드 등록을 위한 새로운 하이브리드 디스크립터인 LoGDesc를 소개합니다. 이 디스크립터는 각 포인트의 이웃 구조 설명을 위해 로컬 기하학적 특성과 학습 기반 기능 전파를 결합하여 제안됩니다.

- **Technical Details**: 제안된 아키텍처는 주성분 분석(Principal Components Analysis, PCA)을 사용하여 각 포인트의 평면성(planarity), 비등방성(anisotropy), 전방위성(omnivariance)을 계산하여 초기 기하학적 정보를 추출한 후, 삼각형 기반의 이웃 구조를 통해 추정된 법선 벡터를 바탕으로 디스크립터를 보완합니다. 최종 기하학적 디스크립터는 로컬 그래프 합성과 주의(attention) 메커니즘을 사용하여 포인트 간에 전파됩니다.

- **Performance Highlights**: 제안된 로컬 디스크립터는 ModelNet40, Bunny Stanford 데이터셋, KITTI 및 MVP-RG와 같은 데이터셋에서 포인트 클라우드 등록을 평가하였으며, 특히 노이즈가 많고 겹치는 부분이 적은 포인트 클라우드에서 흥미로운 결과를 보여줍니다.



### SynCo: Synthetic Hard Negatives in Contrastive Learning for Better Unsupervised Visual Representations (https://arxiv.org/abs/2410.02401)
Comments:
          10 pages, 6 figures, 4 tables. arXiv admin note: text overlap with arXiv:2010.01028 by other authors

- **What's New**: 이 논문에서는 SynCo(대조 학습에서의 합성 네거티브)를 소개하고, 이는 합성 하드 네거티브를 생성하여 학습 과정을 향상시키는 새로운 대조 학습 접근법입니다.

- **Technical Details**: SynCo는 MoCo 프레임워크를 기반으로 하며, 합성 하드 네거티브를 생성하는 여섯 가지 새로운 전략을 도입합니다: (1) Interpolated negatives; (2) Extrapolated negatives; (3) Mixup negatives; (4) Noise-injected negatives; (5) Perturbed negatives; (6) Adversarial negatives. 이 방법은 메모리 큐에서 온디맨드로 하드 네거티브를 생성하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: SynCo는 ImageNet의 선형 평가에서 200 epochs 이후 68.1%의 top-1 정확도를 기록하여 MoCo의 67.5%를 초과합니다. 또한, PASCAL VOC에서 82.5%의 AP를 달성하며, COCO 데이터셋의 바운딩 박스 검출에서 40.4% AP, 인스턴스 분할에서 35.4% AP를 기록하여 새로운 벤치마크를 세웠습니다.



### Parameter Competition Balancing for Model Merging (https://arxiv.org/abs/2410.02396)
Comments:
          Accepted by NeurIPS2024

- **What's New**: 본 논문은 PCB-Merging (Parameter Competition Balancing)이라는 새로운 기법을 소개하고 있으며, 이 기법은 여러 모델의 파라미터를 효과적으로 통합하는 경량화된 기술입니다. 기존의 모델 병합 기법이 작업 간의 잠재적 충돌 및 복잡한 상관관계를 다루는 데 어려움이 있었던 것에 반해, PCB-Merging은 파라미터 경쟁을 관리하여 더 나은 성능 향상을 도모합니다.

- **Technical Details**: PCB-Merging은 단계적으로 두 가지 균형 조정 방식을 활용합니다: intra-balancing과 inter-balancing. Intra-balancing은 개별 작업 내에서 파라미터의 중요성을 평가하고, inter-balancing은 작업 간의 파라미터 유사성을 평가합니다. 중요도가 낮은 파라미터는 제거되고, 남은 파라미터는 최종 병합 모델을 형성하기 위해 재조정됩니다. 이러한 방식으로 모델의 성능을 극대화합니다.

- **Performance Highlights**: 다양한 실험에서 PCB-Merging 기법이 기존의 모델 병합 방법들보다 우수한 성능 향상을 보여 주었으며, 교차 작업(cross-task), 교차 도메인(cross-domain), 교차 훈련 구성을 포함한 다양한 병합 시나리오에서 제출되었습니다. 특히, T5-base 모델을 사용했을 때 4.3%의 성능 향상을 달성했습니다.



### Unleashing the Potential of the Diffusion Model in Few-shot Semantic Segmentation (https://arxiv.org/abs/2410.02369)
Comments:
          Accepted to Proc. Annual Conference on Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이 논문에서는 최근 Diffusion Model (확산 모델)을 통해 Few-shot Semantic Segmentation (소수 샷 의미 세분화)에서의 가능성을 탐구하고 있습니다. Latent Diffusion Model (LDM)을 활용하여, 하나의 쿼리 이미지에 대해 지원 이미지를 통해 의미 세분화를 수행하는 방법론을 제시하고 있습니다.

- **Technical Details**: 본 연구에서는 쿼리 이미지와 지원 이미지 간의 상호 작용을 촉진하기 위해 KV fusion (Key-Value 융합) 방법을 제안하며, 지원 마스크 정보를 최적화하여 쿼리 마스크에서 합리적인 감독을 제공하는 방법을 재평가합니다. 이러한 분석을 기반으로 DiffewS라는 간단하고 효과적인 프레임워크를 구축했습니다.

- **Performance Highlights**: 실험 결과, 제안된 DiffewS 방법은 기존의 SOTA (State-of-the-Art) 모델에 비해 여러 설정에서 유의미하게 우수한 성능을 보였습니다. 특히, ‘in-context learning’ 설정에서도 두드러진 성과를 기록하였습니다.



### A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond (https://arxiv.org/abs/2410.02362)
- **What's New**: Mamba는 의료 이미지 분석에서 템플릿 기반의 딥러닝 접근 방식을 대체할 수 있는 유망한 대안으로 주목받고 있습니다. 기존의 Transformer 아키텍처의 단점인 다차원적 계산 복잡성과 긴 범위의 의존성을 효과적으로 처리하지 못하는 문제를 해결합니다.

- **Technical Details**: Mamba는 State Space Model(SSM)의 특수한 경우로, 선형 시간 복잡성을 제공하여 긴 시퀀스를 처리할 수 있으며 메모리 소모를 줄이는 기능이 있습니다. Mamba 아키텍처는 순수 Mamba, U-Net 변형, CNN과 Transformer, Graph Neural Networks를 포함한 하이브리드 모델 등을 포함하고 있으며, 선택적 스캔 메커니즘과 하드웨어 친화적 알고리즘을 채택하여 효율적인 저장과 계산을 지원합니다.

- **Performance Highlights**: Mamba는 의료 이미지 세분화, 분류, 합성 등 다양한 작업에서 우수한 성능을 나타내며, 다중 양식 데이터 통합에 효과적입니다. 이러한 특성 덕분에 Mamba는 환자 진단 정확도와 결과를 향상시키는 데 기여할 수 있습니다.



### ProtoSeg: A Prototype-Based Point Cloud Instance Segmentation Method (https://arxiv.org/abs/2410.02352)
- **What's New**: 본 논문에서는 3D 포인트 클라우드에서 인스턴스 세분화를 수행하기 위한 새로운 신경망 아키텍처를 제안합니다. 프로토타입과 계수를 병렬로 학습하여 인스턴스 예측을 얻는 접근 방식이 특징입니다.

- **Technical Details**: 이 연구에서는 'Dilated Point Inception (DPI)' 모듈을 사용하여 다중 스케일 계수를 추출하고, 비최대 억제(Non-Maximum Suppression, NMS) 알고리즘을 통해 최종 인스턴스 예측을 도출합니다. 클러스터링 단계를 생략하여 예측 시간을 더 안정적이고 빠르게 만드는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 방법은 현재의 최첨단 기술에 비해 28% 더 빠르며, 평균 표준 편차가 1.0%로 매우 낮은 성능을 보입니다. S3DIS-blocks에서 4.9%의 mRec를, PartNet에서는 평균 2.0%의 mAP 향상을 기록하며, 전체적으로 빠른 속도와 낮은 변동성을 제공하는 것이 특징입니다.



### Self-eXplainable AI for Medical Image Analysis: A Survey and New Outlooks (https://arxiv.org/abs/2410.02331)
- **What's New**: 본 논문은 Self-eXplainable AI (S-XAI) 개념을 통해 의료 이미지 분석을 위한 투명하고 신뢰할 수 있는 딥러닝 모델의 설계를 제안합니다. 기존의 Post-hoc XAI 기술과 달리, S-XAI는 훈련 과정에 해석 가능성을 직접 통합하여 모델의 의사 결정과정을 쉽게 설명할 수 있도록 합니다.

- **Technical Details**: 이 논문은 다양한 이미지 모달리티와 임상 응용 분야를 아우르는 S-XAI 방법론에 대한 포괄적인 리뷰를 제공합니다. 200편 이상의 논문을 3가지 주요 관점으로 분석하며, 1) 입력 해석 가능성, 2) 모델 해석 가능성, 3) 출력 해석 가능성을 포함합니다. 또한, S-XAI은 특히 고위험 영역인 의료 이미지 분석에서 모델의 신뢰성과 안전성을 높이는 데 중요한 역할을 합니다.

- **Performance Highlights**: 의료 이미지 분석을 위한 S-XAI 기술이 발전함에 따라, 임상의와 AI 시스템 간의 협업적 의사결정이 가능해지고, AI 기반 진단 및 개입의 정확성이 향상됩니다. 본 논문은 S-XAI의 체계적 리뷰를 제공하며, 임상적 통합 및 실제 배포에 있어서 더욱 신뢰할 수 있는 AI 시스템을 도모합니다.



### RESSCAL3D++: Joint Acquisition and Semantic Segmentation of 3D Point Clouds (https://arxiv.org/abs/2410.02323)
Comments:
          2024 IEEE International Conference on Image Processing (ICIP). IEEE, 2024

- **What's New**: 이 연구에서는 고해상도 3D 센서의 동작 방식을 정확하게 시뮬레이션하는 새로운 포인트 클라우드 데이터셋 VX-S3DIS와 함께, 업데이트 모듈 및 처리 전략을 포함한 RESSCAL3D++라는 중요한 개선점을 소개합니다.

- **Technical Details**: 연구에서는 3D 장면의 공동 수집(joint acquisition) 및 의미론적 분할(semantic segmentation)을 실현하는 방법을 다루고 있습니다. RESSCAL3D++는 저해상도 데이터가 주어질 때 초기 의미 예측을 신속하게 생성하며, 새로운 추가 포인트만을 처리하여 성능을 향상시키는 다중 분기(multi-branch) 구조를 가지고 있습니다.

- **Performance Highlights**: 이 접근법은 mIoU에서 스케일 능력 비용을 2%에서 0.2%로 감소시키고, 비스케일러블(baseline) 방식에 비해 15.6%에서 63.9%까지 처리 속도를 크게 향상시킵니다. 또한, 첫 번째 예측은 전체 추론 시간의 7% 만에 이루어져, 조기 예측을 가능하게 합니다.



### CTARR: A fast and robust method for identifying anatomical regions on CT images via atlas registration (https://arxiv.org/abs/2410.02316)
- **What's New**: 이번 논문에서는 CT 이미지 분석 파이프라인의 전처리 단계로 사용할 수 있는 CT Anatomical Region Recognition(CTARR)이라는 새로운 방법을 제안합니다. 이 방법은 관심 있는 해부학적 부위를 자동으로 식별하고 나머지를 제거하여 깊은 학습 기반의 CT 이미지 분석에서 비효율성을 줄입니다.

- **Technical Details**: CTARR는 atlas registration을 기반으로 하며 CT 스캔 이미지에서 해부학적 영역을 신속하고 견고하게 식별합니다. 이 방법은 특정 해부학적 부위를 정의하는 바운딩 박스(bounding box)를 사용하여 새로운 관심 영역을 추가할 수 있도록 설계되었습니다. 전통적인 등록 방법보다 더 효과적인 이미지 등록 알고리즘을 개발하여 대규모 변환을 통해 최적 정렬을 제공합니다.

- **Performance Highlights**: 제안된 방법은 1131개의 CT 스캔을 평가하여 97.45-100%의 정확도로 관심 영역의 전경 복셀을 보존하고, 계산 시간은 0.1-0.21초로 신속하게 처리할 수 있음을 입증했습니다. 이는 기존 방법보다 2.0-12.7배 더 빠른 세분화 시간을 줄여주는 성과를 보였습니다.



### Decoupling Layout from Glyph in Online Chinese Handwriting Generation (https://arxiv.org/abs/2410.02309)
- **What's New**: 이 논문은 전체 중국어 손글씨 텍스트 라인 생성이라는 새로운 문제를 다룬다. 기존의 연구는 개별 문자 생성에 중점을 두었으나, 본 연구는 레이아웃(layout)과 글리프(glyphs)로 나누어 텍스트 라인을 계층적으로 생성하는 방법을 제안한다.

- **Technical Details**: 제안된 방법은 텍스트 내용과 스타일 참조를 기반으로 레이아웃 생성기를 설계하여 각 글리프의 위치를 자동회귀적으로 생성한다. 글꼴 생성에는 문자 임베딩 사전, 다중 스케일 서예 스타일 인코더, 1D U-Net 기반의 확산 노이즈 제거기가 포함된다.

- **Performance Highlights**: CASIA-OLHWDB 데이터세트에 대한 질적 및 양적 실험을 통해 구조적으로 올바르고 구별할 수 없는 모방 샘플을 생성할 수 있는 능력을 입증하였다.



### The Comparison of Individual Cat Recognition Using Neural Networks (https://arxiv.org/abs/2410.02305)
Comments:
          13 pages,7 figures

- **What's New**: 이 연구는 고양이 인식을 위한 다양한 신경망 모델의 효능을 체계적으로 비교하여, 전통적인 CNN(Convolutional Neural Network)이 전이 학습(transfer learning)을 통해 더 나은 성능을 나타낸다는 점을 밝힌 최초의 연구 중 하나입니다.

- **Technical Details**: 본 연구에서는 ResNet, DenseNet, EfficientNet, ConvNeXt 및 Siamese 네트워크 등 여러 신경망 모델을 사용하여 고양이를 인식하는 데 있어 각 모델의 장단점을 비교했습니다. 특히, 전이 학습을 통한 CNN 모델의 성능이 미세 조정(fine-tuning) 방식이나 Siamese 네트워크보다 우수한 것으로 나타났습니다.

- **Performance Highlights**: ConvNeXt와 DenseNet 모델은 각각 개별 고양이 인식에서 뚜렷한 성과를 달성하였으며, 이를 통해 애완동물 상점 및 야생에서의 고양이 관리 및 감시를 개선할 수 있는 방법을 제시하였습니다.



### A Novel Method for Accurate & Real-time Food Classification: The Synergistic Integration of EfficientNetB7, CBAM, Transfer Learning, and Data Augmentation (https://arxiv.org/abs/2410.02304)
Comments:
          20 pages, six figures, two tables

- **What's New**: 본 연구는 EfficientNetB7 아키텍처를 활용하여 인공지능 식품 분류의 정확도를 크게 개선하고, 빠른 처리 속도를 유지할 수 있는 비용 효율적인 모델을 제안합니다.

- **Technical Details**: 이 방법론은 전이 학습(transfer learning)과 데이터 증강(data augmentation), CBAM 주의 모듈(attention module)을 통해 강화되며, Food11 데이터셋을 활용하여 11개의 다양한 클래스에 대해 16643개의 불균형 이미지를 포함합니다.

- **Performance Highlights**: 본 연구에서 제안된 모델은 평균 96.40%의 인상적인 정확도를 보이며, 미지의 데이터에 대한 추론 시 1초 이내에 60개 이상의 이미지를 분류할 수 있는 능력을 보여줍니다.



### Computer-aided Colorization State-of-the-science: A Survey (https://arxiv.org/abs/2410.02288)
- **What's New**: 이 연구는 컴퓨터 보조 색채화 기술에 대한 기존 연구를 포괄적으로 리뷰하고 있으며, 색채화 작업의 진화를 컴퓨터 그래픽스와 컴퓨터 비전의 융합 관점에서 다룹니다. 특히 색채화의 미적 평가를 도입하여 7개의 대표적인 무조건 색채화 모델을 평가한 점이 새롭습니다.

- **Technical Details**: 본 연구는 색채화를 세 가지 주요 범주인 조건부 방법, 무조건 방법, 비디오 색채화로 분류합니다. 기존의 재건 기반 색채화 평가 기술을 확장하면서 색채화의 미적 평가를 고려하여, 다양한 색깔 정보를 생성하는 방식과 함께 심층 학습 기법을 활용한 '세멘틱'(semantic) 이해를 강조합니다.

- **Performance Highlights**: 연구에서는 기존 색채화 기법과의 비교를 통해 제안된 새로운 미적 평가 방식의 차별성을 설명하며, 색채화 기술 발전 방향과 미래 연구 분야에 대한 통찰을 제공합니다. 이 연구는 색채화 분야의 미적 요소를 정량적으로 평가하는 첫 시도로, 향후 연구에 많은 기여를 할 것으로 기대됩니다.



### Probabilistic road classification in historical maps using synthetic data and deep learning (https://arxiv.org/abs/2410.02250)
- **What's New**: 본 연구는 역사적 지도에서 도로 데이터를 자동으로 추출하고 분류하기 위해 레이블이 없는 도로 기하학(geometry)만을 사용하는 새로운 프레임워크를 제안합니다. 이는 기존의 방법들이 대량의 라벨링된 데이터에 의존했던 점과 차별화됩니다.

- **Technical Details**: 제안된 방법은 이진 세분화(binary segmentation) 모델을 훈련시킨 후, 형태학적 작업(morphological operations), 스켈레톤화(skeletonization), 벡터화(vectorization) 및 필터링 알고리즘을 적용합니다. 이후 특정 기호(symbology)를 사용하여 도로 구간을 인공적으로 재페인팅(painting)하여 합성 훈련 데이터를 생성합니다.

- **Performance Highlights**: 시험에서 가장 일반적인 도로 클래스 2에 대해 94% 이상의 완전성(completeness) 점수와 92% 이상의 정확성(correctness) 점수를 달성하였습니다. 이 연구는 효율적으로 역사적 지도에서 도로를 추출하고 분류하는 강력한 도구를 제공합니다.



### Spiking Neural Network as Adaptive Event Stream Slicer (https://arxiv.org/abs/2410.02249)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 연구진은 SpikeSlicer라는 새로운 이벤트 처리 방법을 제안하였으며, 이는 이벤트 스트림을 유연하게 분할할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: SpikeSlicer는 저전력 스파이킹 신경망(Spiking Neural Network, SNN)을 활용하여 이벤트 분할을 동적으로 결정하며, Spiking Position-aware Loss (SPA-Loss) 함수를 통해 SNN이 정확한 시간에 스파이크를 발생시키도록 안내합니다. 또한 피드백 업데이트(Feedback-Update) 훈련 전략을 통해 SNN이 장기 신경망(Artificial Neural Network, ANN)으로부터의 피드백을 활용하여 분할 결정을 개선합니다.

- **Performance Highlights**: SpikeSlicer는 객체 추적에서 22.3%, 인식에서 19.2%의 성능 향상이 있었으며, 그에 따른 에너지 소비는 2.7% 증가에 불과합니다.



### Visual Prompting in LLMs for Enhancing Emotion Recognition (https://arxiv.org/abs/2410.02244)
Comments:
          Accepted by EMNLP2024 (Main, Long paper)

- **What's New**: 본 논문은 감정 인식(emotion recognition)을 향상시키기 위해 Set-of-Vision (SoV) 프롬프트를 소개합니다. 이를 통해 공간 정보(spatial information)인 바운딩 박스(bounding boxes)와 얼굴 랜드마크(facial landmarks)를 사용해 대상의 정확한 위치를 표시합니다.

- **Technical Details**: 본 연구는 VLLMs(visual large language models)를 위한 공간적 시각 프롬프트를 통합하여 감정 인식 정확도를 높이는 방법을 제안합니다. SoV 방식은 이미지에서 여러 얼굴의 감정을 정밀하게 분석할 수 있도록 설계되었습니다. 감정 분류를 위해 객체 겹침 처리 알고리즘을 사용해 복잡한 상황에서 여러 얼굴을 효과적으로 다룹니다.

- **Performance Highlights**: SoV 프롬프트를 사용한 결과, 정확한 얼굴 수(18개)를 확인하고 감정을 ‘중립 감정’, ‘미소 또는 긍정적 감정’, ‘행복’으로 보다 정확하게 분류했습니다. 이는 기존 접근법보다 더 정밀한 분석을 가능하게 하며, 전반적인 감정 인식 성능을 개선했습니다.



### SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack (https://arxiv.org/abs/2410.02240)
- **What's New**: 이 논문은 기존의 무제한 적대 공격(Unrestricted Adversarial Attacks) 방식의 한계를 극복한 새로운 접근 방식인 'Semantic-Consistent Unrestricted Adversarial Attack (SCA)'를 제안합니다. 이 방법은 고급 의미 정보(Semantic Information)를 활용하여 적대적 예제를 생성하는 동안 의미 일관성을 유지합니다.

- **Technical Details**: SCA는 Denoising Diffusion Probabilistic Models (DDPM) 및 Multimodal Large Language Models (MLLM)을 활용하여 퍼트에이션(perturbation) 과정에서 의미적인 지침을 제공합니다. 이를 통해 샘플링 효율을 높이고, 의미 일관성이 유지되는 적대적 예제를 생성할 수 있습니다. DPM Solver++를 통해 시간 단계를 10-20으로 줄여 효율성을 극대화합니다.

- **Performance Highlights**: SCA는 평균 12배 더 빠른 속도로 적대적 예제를 생성할 수 있으며, 이전의 최첨단 기술과 유사한 성공률을 보입니다. 이 접근 방식은 의미적인 왜곡을 최소화하면서 자연스럽고 시각적으로 매력적인 결과를 제공합니다.



### Key-Grid: Unsupervised 3D Keypoints Detection using Grid Heatmap Features (https://arxiv.org/abs/2410.02237)
- **What's New**: 이 논문에서는 Key-Grid라는 새로운 비지도 학습 방식의 3D 키포인트 탐지기를 소개합니다. 이 방법은 변형 가능한 물체에 대해서도 의미론적 일관성을 유지하며, 고체 물체와 변형 물체 모두에서 3D 키포인트를 효과적으로 탐지할 수 있도록 설계되었습니다.

- **Technical Details**: Key-Grid는 오토 인코더(autoencoder) 구조를 기반으로 하며, encoder는 키포인트를 예측하고, decoder는 생성된 키포인트를 이용해 물체를 복원합니다. 새로운 개념으로, grid heatmap 이라는 3D 격자 피쳐 히트맵을 형성하여 이를 decoder에서 활용합니다. 이 히트맵은 3D 큐빅 공간에서 균일하게 샘플링된 그리드 포인트들에 대한 잠재 변수를 나타냅니다. 이 과정에서 각 인코더의 레이어 정보를 decoder에 통합하여 사용합니다.

- **Performance Highlights**: Key-Grid는 ShapeNetCoreV2 데이터셋에서 고체 물체에 대해 SOTA(기존의 최고 성능)를 달성하였고, ClothesNet 데이터셋에서도 이전 SOTA에 비해 각각 8.0%와 9.1% 성능 향상을 보였습니다. 또한, Key-Grid는 노이즈 및 다운샘플링에 강인함을 보여줍니다.



### Efficient Semantic Segmentation via Lightweight Multiple-Information Interaction Network (https://arxiv.org/abs/2410.02224)
Comments:
          10 pages, 6 figures, 9 tables

- **What's New**: 최근 CNN (Convolutional Neural Networks)과 Transformer의 통합을 통해 세분화된 이미지 분석의 효과성을 높이는 LMIINet이라는 경량화된 다중 정보 상호작용 네트워크가 제안되었습니다. 이 접근법은 실시간 시나리오에서의 높은 메모리 요구와 계산 부담을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: LMIINet은 LFIB (Lightweight Feature Interaction Bottleneck) 모듈을 특징으로 하며, 효율적인 합성곱 연산을 통해 컨텍스트 통합 능력을 강화합니다. 여기에는 Flatten Transformer의 개선이 포함되어 있으며, 이를 통해 로컬 및 글로벌 기능 상호작용을 강화하여 세부적인 의미 정보를 포착할 수 있습니다. LFIB와 Transformer 블록에서 결합 계수 학습 방식을 도입해 특징 상호작용을 개선하였습니다.

- **Performance Highlights**: LMIINet은 단일 RTX2080Ti GPU에서 Cityscapes 테스트 세트에서 72.0%의 mIoU (mean Intersection over Union)와 함께 100 FPS를 달성하였으며, CamVid 테스트 데이터셋에서는 69.94%의 mIoU를 160 FPS로 성능을 발휘하여 정확도와 효율성을 균형 있게 유지했습니다.



### Hard Negative Sample Mining for Whole Slide Image Classification (https://arxiv.org/abs/2410.02212)
Comments:
          13 pages, 4 figures, accepted by MICCAI 2024

- **What's New**: 본 논문은 weakly supervised whole slide image (WSI) 분류 문제를 해결하기 위해 negative sample(음성 샘플) 선택 방식에 대한 새로운 접근법을 제안합니다. 특히, pseudo labeling(유사 라벨링) 프로세스에서 낙관적인 패치(patch)들만 선별하는 데 집중한 기존 방법과 달리, 어려운 negative sample을 발굴하여 feature representation(특징 표현)을 개선함으로써 학습 효율성을 높이고 훈련 비용을 줄이는 방법을 제시합니다.

- **Technical Details**: 논문에서는 hard negative sampling(어려운 음성 샘플 채취) 기술을 도입해 positive 패치와 매우 가까워진 negative 패치를 찾아냈습니다. 이러한 패치들은 self-supervised learning(자기 지도 학습)에서의 contrastive learning(대비 학습) 과정에 활용되며, MIL(multiple instance learning)에서 patch-wise ranking loss(패치별 순위 손실) 손실함수를 도입하여 positive와 negative 패치의 정확한 순위를 보장합니다.

- **Performance Highlights**: 실험은 두 개의 공공 데이터세트를 기반으로 하여 진행되었으며, 제안된 기술이 기존 방식들에 비해 효과적으로 개선된 성능을 보여줍니다. 특히, 학습 시간의 대폭적인 단축과 함께 전체 분류의 정확도를 높이는 데 기여했습니다.



### Adapting Segment Anything Model to Melanoma Segmentation in Microscopy Slide Images (https://arxiv.org/abs/2410.02207)
- **What's New**: 이 논문은 Whole Slide Images (WSIs)에서의 멜라노마(segmentation을 위한 새로운 접근 방식을 제시합니다. Segment Anything Model(SAM)을 이용하여 미세현미경 이미지에서 멜라노마를 자동으로 분할하는 방법을 설명합니다.

- **Technical Details**: 우리의 방법은 초기 semantic segmentation 모델을 사용하여 초기 분할 마스크를 생성하고, 이를 SAM에 대한 프롬프트로 사용합니다. 우리는 중심과 그리드 프롬프트를 조합한 동적 프롬프팅 전략을 설계하여 초고해상도 슬라이드 이미지의 최적 커버리지를 달성하면서 프롬프트의 품질을 유지합니다. Segformer를 초기 segmentation 모델로, EfficientSAM을 세분화 모델로 선택했습니다.

- **Performance Highlights**: 실험 결과에 따르면, 우리의 접근 방식은 기존의 첨단 멜라노마 segmentation 방법들을 초월할 뿐만 아니라, Segformer의 기준 성능을 IoU 기준으로 9.1% 이상 개선했습니다.



### Remember and Recall: Associative-Memory-based Trajectory Prediction (https://arxiv.org/abs/2410.02201)
- **What's New**: 본 논문에서는 분절 메모리 기반의 경로 예측(FMTP) 모델을 제안하여 기존의 지속적 표현 방식의 계산 비효율성과 불확실한 상황에서의 적응력 부족 문제를 해결하고자 합니다.

- **Technical Details**: FMTP 모델은 훈련 과정에서 연속적으로 수집된 경로 표현을 기반으로 학습할 수 있는 메모리 배열을 설계하여 불필요한 정보를 줄이고 필수적인 특징을 유지합니다. 이 메모리 배열은 또한 언어 모델에 기반한 고급 추론 엔진을 통해 이산적 표현 간의 연관 규칙을 깊이 학습하여 모델의 인지 능력을 향상시킵니다.

- **Performance Highlights**: 공식적인 여러 데이터셋(ETH-UCY, inD, SDD, nuScenes, Waymo, VTL-TP)을 통해 FMTP 모델의 성능을 평가한 결과, 기존의 벤치마크와 비교했을 때 다양한 메트릭에서 튼튼함과 적응성을 보이며 우수한 성능을 달성하였습니다.



### BadCM: Invisible Backdoor Attack Against Cross-Modal Learning (https://arxiv.org/abs/2410.02182)
- **What's New**: 이 논문은 기존의 단일 모달 공격 아이디어에서 발전하여, 교차 모달(데이터 간의 관계가 존재하는 여러 데이터 형태) 백도어 공격을 다룰 새로운 방법론인 BadCM을 제안합니다. 이는 다양한 공격 시나리오에 대한 일반화 가능성을 고려하였으며, 인간의 눈에 잘 띄지 않는 트리거 패턴을 활용하여 공격의 실용성을 높였습니다.

- **Technical Details**: BadCM은 교차 모달 학습(Cross-Modal Learning)을 목표로 한 최초의 일반화된 백도어 프레임워크로서, 이미지와 텍스트 간의 상관관계를 분석하는 교차 모달 마이닝 기법을 개발하여 모달리티 불변 구성 요소를 식별하게 됩니다. 최적의 트리거 패턴은 이러한 영역에 주입되며, 시각적 및 언어적 모달리티를 위해 각각 특화된 생성기를 도입하여 높은 은폐성을 갖는 샘플을 생성합니다.

- **Performance Highlights**: 광범위한 실험을 통해, BadCM은 교차 모달 검색과 비주얼 질문 응답(VQA)에서 다양한 공격 시나리오 하에 효과성을 입증하였습니다. 또한 기존의 백도어 방어를 강력하게 회피할 수 있음이 드러났습니다.



### HATFormer: Historic Handwritten Arabic Text Recognition with Transformers (https://arxiv.org/abs/2410.02179)
- **What's New**: HATFormer는 아랍어 필기 인식(HTR) 시스템으로, 기존의 CNN 및 RNN 기반 방법보다 더 효율적인 transformer 기반의 구조를 가지고 있습니다. 이 시스템은 복잡한 아랍어 스크립트의 특성을 효과적으로 처리하기 위해 최적화되었습니다.

- **Technical Details**: HATFormer는 transformer 구조를 기반으로 하며, attention 메커니즘을 활용하여 연결된 문자, 문맥 의존적 형태, 그리고 단어 의미를 변화시킬 수 있는 diacritics를 인식하는 데 강점을 갖고 있습니다. 또한, 이미지 전처리기와 텍스트 토크나이저를 포함하여 제한된 역사적 데이터로부터 더 나은 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: HATFormer는 가장 큰 공개 역사적 필기 아랍어 데이터셋으로 평가 시 8.6%의 문자 오류율(CER)을 기록했습니다. 또한, 사적 데이터셋에서 4.2% CER을 달성하며, 기존 방법들보다 각각 51% 개선되었습니다. 이 시스템은 역사적 아랍어 문서의 자동 전사와 인식 오류 진단에 도움을 줄 수 있어, 디지털 인문학 연구에 큰 기여를 할 것입니다.



### An Evaluation of Large Pre-Trained Models for Gesture Recognition using Synthetic Videos (https://arxiv.org/abs/2410.02152)
Comments:
          Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications II (SPIE Defense + Commercial Sensing, 2024)

- **What's New**: 이번 연구에서는 대규모 사전 학습 모델을 사용하여 비디오 기반 제스처 인식에 대한 합성 데이터의 가능성을 탐구합니다. 연구자는 훈련 없이 분류가 가능한지 검증하기 위해 K-nearest neighbors (KNN) 분류법을 적용하고, 제스처에 대한 텍스트 설명을 이용한 zero-shot classification 방법과 비교합니다.

- **Technical Details**: 주요 실험에서는 RoCoG-v2 데이터셋을 사용하였으며, 두 가지 '훈련 없는' 접근 방식(합성 영상으로 KNN 분류와 텍스트 설명을 통한 zero-shot classification)을 시험합니다. 자가 감독 방식으로 사전 학습된 비디오 모델과 비전-언어 사전 학습된 모델(예: ViCLIP)을 활용하였습니다. 성능 비교를 위해 두 가지 모드에서 실험을 진행하였습니다.

- **Performance Highlights**: 합성 훈련 비디오를 사용한 KNN 분류의 결과는 실제 훈련 데이터에 비해 유의미하게 낮은 정확도를 보였습니다. 제스처 인식을 위한 텍스트 기반 zero-shot 분류는 자연어로 설명하기 힘든 미세한 동작 차이들로 인해 성능이 저조한 결과를 나타냈습니다. 또한, Kinetics와 SSv2 데이터셋에서 미세 조정된 모델에 따라 KNN의 성능이 크게 달라졌습니다.



### MVGS: Multi-view-regulated Gaussian Splatting for Novel View Synthesis (https://arxiv.org/abs/2410.02103)
Comments:
          Project Page:this https URL

- **What's New**: 최근 논문에서는 3D Gaussian Splatting (3DGS) 최적화 방법을 제안하여 단일 뷰 감독(supervision)을 사용하던 기존의 훈련 패러다임을 다중 뷰(multiview) 훈련 전략으로 전환하고, 3D Gaussian 특성을 더욱 최적화합니다.

- **Technical Details**: 제안된 MVGS 방법은 3D Gaussian의 획득을 다중 뷰 규제로 변환하여 특정 뷰에 대한 과적합(overfitting) 문제를 피하고, 다른 해상도에서의 세밀한 훈련 절차를 도입합니다. 또한, 교차 레이 밀집화(cross-ray densification) 전략을 통해 더 많은 Gaussian 커널을 밀집시키며, 다양한 뷰에 대한 적합성을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 MVGS 방법이 다양한 작업에서 약 1 dB PSNR 성능 향상을 이루었으며, 특히 다중 뷰 감독을 통해 물체 및 장면 복원이 개선됨을 보여줍니다.



### Orient Anything (https://arxiv.org/abs/2410.02101)
- **What's New**: 본 연구는 심볼릭하게 대칭성을 가진 3D 모양의 방향 추정에 있어 발생할 수 있는 근본적인 장애물들을 밝혀내고, 두 단계로 구성된 방향 추정 파이프라인을 제안합니다. 이 방법은 Shapenet 전체 데이터를 사용하여 이전 연구보다 더 우수한 성능을 보여줍니다.

- **Technical Details**: 제안된 방향 추정 방법은 1단계에서는 연속 회귀 문제를 해결하여 모양의 방향을 회복하고, 2단계에서는 정수 분류 문제를 해결하여 24개의 옥타헤드 회전을 예측합니다. 이 방법은 기존의 회전 대칭성을 고려하여 방향 추정의 효율성을 높이며, 데이터의 불확실성에 따라 예측 세트를 조정할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 방향 추정에서 state-of-the-art 성능을 달성하였으며, Objaverse 데이터세트에서 다수의 3D 모델에 대한 일반화 능력을 성공적으로 입증했습니다.



### EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing (https://arxiv.org/abs/2410.02098)
- **What's New**: Diffusion transformers를 위해 새로운 Mixture-of-Experts (MoE) 모델인 EC-DIT를 제안하여, 텍스트에서 이미지로의 변환 과정에서 동적 계산 최적화를 구현했습니다. 이는 텍스트 이미지 복잡도에 맞는 이질적(computational heterogeneity) 계산을 가능하게 합니다.

- **Technical Details**: EC-DIT는 970억 개의 파라미터로 확장 가능하며, expert-choice routing (전문가 선택 경로)을 통해 계산 자원을 효율적으로 분배할 수 있습니다. 이 모델은 이미지의 전반적인 정보를 활용하여 다양한 패턴에 맞춰 계산 자원을 조정합니다.

- **Performance Highlights**: EC-DIT는 텍스트-이미지 정렬 평가에서 71.68%의 GenEval 스코어를 달성하며, 밀집 모델 및 기존 MoE 모델보다 훈련 수렴성과 이미지 생성 품질이 현저히 향상되었습니다. 또한, 계산 오버헤드가 30% 미만 증가하면서도 성능이 크게 개선되었습니다.



### EMMA: Efficient Visual Alignment in Multi-Modal LLMs (https://arxiv.org/abs/2410.02080)
- **What's New**: EMMA(효율적인 다중 모드 적응)은 시각적 인코딩과 텍스트 인코딩을 효율적으로 결합할 수 있도록 설계된 경량 크로스 모달리티 모듈입니다. 이는 언어 모델에서의 지침 인식 시각적 표현 생성을 가능하게 합니다.

- **Technical Details**: EMMA는 적은 추가 파라미터(모델 크기의 0.2% 증가)로 비전과 언어 표현을 통합하는 효율적인 초기 융합 메커니즘을 도입합니다. 이는 시각적 표현을 지침과 동기화하여 MLLM에 대한 성능을 향상시키는 역할을 합니다. 또한, 시각적 정렬 모듈에 대한 심층 분석을 수행하여 관련 표적 토큰에 집중하는 방식에 대한 내부 메커니즘을 설명합니다.

- **Performance Highlights**: EMMA는 여러 작업에서 성능을 최대 9.3% 향상시키며, 헬루시네이션에 대한 강인성을 크게 향상시킵니다. 기존의 mPLUG-Owl2와 비교했을 때, 7개 벤치마크에서 우수한 성과를 보여줍니다.



### Depth Pro: Sharp Monocular Metric Depth in Less Than a Second (https://arxiv.org/abs/2410.02073)
Comments:
          Code and weights available at this https URL

- **What's New**: 이 논문은 제로샷(Zero-shot) 메트릭(Metric) 모노큘라(depth estimation) 깊이 추정의 기초 모델인 Depth Pro를 소개합니다. 이 모델은 고해상도 깊이 맵을 생성하며, 카메라 내부 정보가 없더라도 절대 스케일로 메트릭 깊이 맵을 만들어냅니다.

- **Technical Details**: Depth Pro는 효율적인 다중 스케일 비전 트랜스포머(Vision Transformer) 아키텍처를 적용하여 밀집 예측을 수행합니다. 이 모델은 실제 및 합성 데이터셋을 결합한 훈련 프로토콜을 통해 높은 메트릭 정확도를 달성하며 경계 정확도에 대한 다양한 평가 지표를 사용합니다.

- **Performance Highlights**: Depth Pro는 이전 연구보다 물체 경계를 선명하게 구분하며, 깊이 맵 생성에 0.3초가 소요되고, 2.25 메가픽셀의 깊이 맵을 생성합니다. 경계 정확도에서 이전 최첨단 기술 대비 1~2배 빠르게 처리됩니다.



### Learning from the Giants: A Practical Approach to Underwater Depth and Surface Normals Estimation (https://arxiv.org/abs/2410.02072)
Comments:
          18 pages, 6 figures, 8 tables. Submitted to Elsevier

- **What's New**: 이번 논문은 Monocular Depth and Surface Normals Estimation (MDSNE)에 관한 새로운 심층 학습 모델을 소개합니다. 이 모델은 수중에서의 응용에 맞춰 설계된 하이브리드 아키텍처를 사용하며, Convolutional Neural Networks (CNNs)와 Transformers를 통합하여 두 접근법의 강점을 활용합니다.

- **Technical Details**: 이 모델은 고품질의 pseudo-labeled 데이터 생성을 위해  여러 개의 사전 훈련된 MDSNE 모델을 사용하고, Depth Normal Evaluation and Selection Algorithm (DNESA)을 통해 신뢰할 수 있는 pseudo-labeled 샘플을 평가하고 선택합니다. 경량의 student 모델을 사용하여 90%의 파라미터와 80%의 학습 비용 절감이 가능하며, 자원 제약이 있는 장치에서 실시간 3D 인식을 지원합니다.

- **Performance Highlights**: 우리 모델은 약 15.87ms의 이미지 처리 시간으로 실시간 추론을 달성하여 수중 로봇 내비게이션과 같은 작업에 사용할 수 있습니다. 또한, 본 연구는 효율성과 접근성을 우선시하며, 실제 수중 응용에서의 확장성과 실용성을 지원합니다.



### Semi-Supervised Fine-Tuning of Vision Foundation Models with Content-Style Decomposition (https://arxiv.org/abs/2410.02069)
- **What's New**: 본 논문에서는 제한된 레이블 데이터로 다운스트림 작업의 성능을 향상시키기 위해 설계된 반지도 학습(semi-supervised learning) 기반의 미세 조정(fine-tuning) 접근법을 제안합니다. 정보 이론(information theory) 프레임워크를 활용하여 비전 비율 모델(vision foundation models)의 잠재 표현(latent representations)을 향상시키고, 특정 작업의 목표에 더욱 효과적으로 정렬하여 데이터 분포 이동(distribution shift) 문제를 해결합니다.

- **Technical Details**: 우리는 콘텐츠 스타일 분해(content-style decomposition) 원리를 바탕으로 반지도 미세 조정 방법을 제안합니다. pretrained 비전 비율 모델의 잠재 표현을 개선하여 다운스트림 데이터셋에 적용할 때 발생하는 분포 이동 문제를 해결하고, 사용 가능한 비표기(unlabeled) 데이터와 제한된 수의 레이블 데이터(labeled data)를 활용하였습니다. 제안된 방법은 MNIST, CIFAR-10, SVHN, GalaxyMNIST와 같은 여러 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 감독 학습(supervised learning) 기법보다 특히 레이블이 적은 데이터 환경에서 우수한 성능을 보였으며, 미세 조정의 중요성을 강조합니다.



### DisEnvisioner: Disentangled and Enriched Visual Prompt for Customized Image Generation (https://arxiv.org/abs/2410.02067)
Comments:
          The first two authors contributed equally. Project page: this https URL

- **What's New**: 논문에서는 DisEnvisioner라는 새로운 접근 방식을 소개합니다. 이 방법은 단일 이미지에서 시각적 프롬프트와 추가적인 텍스트 지침을 통해 주제의 본질적인 특성을 효과적으로 추출하고 강화하여 개인화된 이미지 생성을 가능하게 합니다.

- **Technical Details**: DisEnvisioner는 주제의 필수 특성과 관련 없는 정보를 분리해내어 'visual tokens'으로 만들며, 이렇게 분리된 특성을 더 세밀한 표현으로 조각화합니다. 이 접근 방식을 통해 기존의 방법들이 간과했던 불필요한 정보의 유입을 방지하고, 개인화 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, DisEnvisioner는 지침 대응(편집 가능성), ID 일관성, 추론 속도 및 전체적인 이미지 품질에서 기존 방법들을 능가함을 보여주었습니다. 이를 통해 DisEnvisioner의 효과성과 효율성이 강조됩니다.



### Using Style Ambiguity Loss to Improve Aesthetics of Diffusion Models (https://arxiv.org/abs/2410.02055)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2407.12009

- **What's New**: 이번 연구에서는 diffusion 모델에 창의성을 학습하기 위한 style ambiguity loss를 적용하여, 별도의 분류기 또는 라벨 데이터셋이 필요하지 않은 새로운 형태의 스타일 모호성 손실을 실험합니다.

- **Technical Details**: 기존의 Creative Adversarial Network (CAN)에서 요구되는 스타일 분류기를 훈련할 필요 없이, CLIP 기반 및 K-Means 기반의 다양한 스타일 모호성 손실을 개발하였습니다. 또한, 강화 학습( Reinforcement Learning ) 기법을 통해 diffusion 모델에 style ambiguity loss를 적용했습니다.

- **Performance Highlights**: Style ambiguity loss로 훈련된 diffusion 모델은 기본 diffusion 모델 및 GAN보다 더 높은 수준의 창의적이고 새로운 이미지를 생성하며, 자동화된 방법 및 사용자 연구를 통해 그 결과를 평가하였습니다.



### Emo3D: Metric and Benchmarking Dataset for 3D Facial Expression Generation from Emotion Description (https://arxiv.org/abs/2410.02049)
Comments:
          11 pages, 10 figures

- **What's New**: 이 논문은 다양한 감정을 포괄하는 'Emo3D'라는 폭넓은 'Text-Image-Expression dataset'을 소개합니다. 이 데이터셋은 인물의 감정 표현과 관련된 이미지 및 3D blendshapes와 함께 제공되어, 이전의 한정된 감정 클래스와 부족한 데이터셋 문제를 해결합니다.

- **Technical Details**: Emo3D 데이터셋은 150,000개의 인스턴스로 구성되어 있으며, 각 인스턴스는 감정 설명, 해당 이미지, blendshape 점수의 삼위일체를 포함합니다. GPT-3.5를 사용하여 감정에 대한 텍스트 설명을 생성하고, DALL-E 3로 이미지를 생성하며, Mediapipe 프레임워크를 통해 대응하는 blendshape 점수를 추정합니다.

- **Performance Highlights**: Emo3D 평가 메트릭은 'Mean Squared Error (MSE)' 메트릭보다 3D 얼굴 표현의 시각-텍스트 정렬 및 의미적 풍부성을 평가하는 데 더 우수함을 입증하였습니다. 이 데이터셋은 애니메이션 디자인, 가상현실, 감정 기반 인간-컴퓨터 상호작용에 큰 응용 가능성을 지니고 있습니다.



### Scene Flow as a Partial Differential Equation (https://arxiv.org/abs/2410.02031)
Comments:
          Project page at this https URL

- **What's New**: 이번 연구에서는 장면 흐름(scene flow) 추정을 연속적인 공간 및 시간의 편미분 방정식(Partial Differential Equations, PDE) 추정 문제로 재정의합니다. 이를 통해 제안하는 비지도 학습 방식인 EulerFlow는 실제 데이터에서 고품질의 장면 흐름을 생성합니다.

- **Technical Details**: EulerFlow는 신경망을 사용하여 장면 흐름을 나타내는 수학적 모델을 최적화하여, 연속적인 위치와 시간에서 장면의 즉각적인 움직임을 설명하는 PDE를 추정합니다. 기존의 포인트 클라우드 거리 목표(예: Chamfer Distance)를 사용하여 이 PDE 추정을 구체화하고, 다양한 시간 간격에서의 흐름 추정을 가능하게 하여 최적화 목표를 개선합니다.

- **Performance Highlights**: Argoverse 2 2024 Scene Flow Challenge에서 EulerFlow는 이전의 어떠한 방법들보다도 뛰어난 성능을 보여주며, 비지도 방법보다 2.5배 이상, 지도 방법보다도 10배 이상 나은 결과를 냅니다. 또한, 작은 빠르게 움직이는 객체들을 잘 추적할 수 있는 능력을 보여줍니다.



### Quantifying the Gaps Between Translation and Native Perception in Training for Multimodal, Multilingual Retrieva (https://arxiv.org/abs/2410.02027)
Comments:
          Short paper accepted to EMNLP24 (Main)

- **What's New**: 이 논문에서는 다양한 언어와 문화에서 이미지 캡션의 인식 차이를 적절하게 반영하는 다국어 비전-언어 모델(multilingual vision-language models)의 부족함을 지적하고, 이를 해결하기 위한 방법론을 제안합니다. 특히, 독일어 사용자 지각을 반영한 캡션과 영어에서 기계 번역 혹은 사람 번역을 통해 만들어진 캡션 간의 성능 차이를 조사합니다.

- **Technical Details**: 이 논문에서는 다국어 CLIP(mCLIP)을 활용하여 독일어 이미지-텍스트(I2T) 및 텍스트-이미지(T2I) 검색 작업을 수행합니다. 이를 위해 Multi30K 데이터셋에서 독일어 원본 캡션(natively written captions)과 전문 번역 캡션(human-translated captions)을 비교하여 성능 차이를 정량화합니다. 추가로, 하이퍼님화(hypernymization) 데이터 증강 기법, 대형 언어 모델(LLM)인 LLaMA-3를 사용한 파라프레이징(paraphrasing) 기법 등으로 번역 캡션 개선을 시도합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법을 통해 평균 검색률(mean recall)이 1.3% 향상되는 결과를 보였으나, 여전히 원어민의 지각(perception)을 반영한 캡션과의 성능 차이가 존재합니다. 이는 앞으로의 연구 방향에 대한 필요성을 강조하며, 다국어 비전-언어 모델의 진전을 위한 열린 문제로 남아 있습니다.



### Normalizing Flow Based Metric for Image Generation (https://arxiv.org/abs/2410.02004)
Comments:
          15 pages, 16 figures

- **What's New**: 이 논문에서는 이미지의 '리얼리즘'(realness)을 평가하기 위해 새로운 두 가지 평가 지표를 제안합니다. 첫 번째는 단순하고 효율적인 flow-based likelihood distance (FLD)이고, 두 번째는 더 정확한 dual-flow based likelihood distance (D-FLD)입니다. 이러한 지표들은 정규화 흐름(normalizing flows)을 사용하여 생성된 이미지가 실제 이미지 분포와 얼마나 밀접하게 일치하는지를 평가합니다.

- **Technical Details**: 정규화 흐름은 정확한 likelihood를 계산할 수 있는 기능을 가지므로, FLD와 D-FLD는 생성된 이미지가 실제 이미지와의 분포적 일치를 어떻게 평가하는지를 제공합니다. 이 지표들은 수백 장의 이미지만으로도 안정된 결과를 도출할 수 있으며(즉, 평균 수렴), 기존의 Fréchet inception distance (FID)와 비교해도 파라미터 수가 현저히 적고 계산적으로 효율적입니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 지표들이 다양한 종류의 이미지 손상에 대해 원하는 단조(monotonic) 관계를 유지한다는 것을 입증했습니다. 이러한 특성 덕분에 새로운 도메인에서도 신뢰할 수 있는 평가가 가능하며, FID보다 속도가 향상되고 더 적은 수의 이미지로도 안정적인 평가를 제공합니다.



### SkyAI Sim: An Open-Source Simulation of UAV Aerial Imaging from Satellite Data (https://arxiv.org/abs/2410.02003)
Comments:
          15 pages, 11 figures

- **What's New**: SkyAI Sim은 드론을 물리적으로 조작하지 않고도 항공 이미지를 가상으로 캡처할 수 있는 도구로, 특정 지역에 대한 bird's-eye view 위성을 생성하는 기능을 제공합니다.

- **Technical Details**: SkyAI Sim은 Google Maps Static API를 사용하여 지도상의 좌표에 따라 위성 이미지를 캡처합니다. 사용자는 비행 고도, 종횡비(aspect ratio) 및 카메라의 대각선 시야(diagonal field of view)와 연속 이미지 간의 겹침(overlap) 등을 정의할 수 있습니다. 이 도구는 저고도 이미지를 캡처하는 기본 용도에서부터 도시 전체에 대한 방대한 데이터셋을 생성하는 복잡한 작업까지 처리할 수 있습니다.

- **Performance Highlights**: SkyAI Sim은 다양한 환경 모니터링, 건설, 도시 관리와 같은 적용 사례에 활용될 수 있으며, 오픈소스 특성 덕분에 다른 미션으로의 확장이 가능합니다. 제공된 데이터셋은 Tennessee의 Memphis를 포함하며, SkyAI를 통해 생성된 데이터와 3D 세계 생성 패키지의 데이터도 포함되어 있습니다.



### UlcerGPT: A Multimodal Approach Leveraging Large Language and Vision Models for Diabetic Foot Ulcer Image Transcription (https://arxiv.org/abs/2410.01989)
Comments:
          13 pages, 3 figures, ICPR 2024 Conference (PRHA workshop)

- **What's New**: 이 논문에서는 UlcerGPT라는 새로운 다중 모달 접근법을 소개하여, 대형 언어 모델과 비전 모델을 활용하여 당뇨병성 족부 궤양(Diabetic Foot Ulcers, DFU) 이미지를 전사하고 분석합니다.

- **Technical Details**: UlcerGPT는 Large Language and Vision Assistant (LLaVA) 및 Chat Generative Pre-trained Transformer (ChatGPT)를 결합하여 DFU 이미지를 공동으로 감지, 분류 및 중요한 영역을 위치 지정하여 전사합니다. 연구에 사용된 데이터셋은 DFU2022 대회에서 수집된 2000개의 주석이 달린 임상 RGB 이미지입니다.

- **Performance Highlights**: UlcerGPT는 임상 전문가에 의해 평가된 공개 데이터셋에서 DFU 전사에 대한 정확성과 효율성을 보여주며, 원격 의료를 통한 신속한 치료 제공을 지원할 수 있는 잠재력을 가지고 있습니다.



### Enhancing Screen Time Identification in Children with a Multi-View Vision Language Model and Screen Time Tracker (https://arxiv.org/abs/2410.01966)
Comments:
          Prepare for submission

- **What's New**: 이 연구에서는 새로운 유형의 센서 정보학 프레임워크를 개발하여 어린이의 화면 노출을 정확하게 모니터링할 수 있는 방법을 제안합니다. 이 방법은 착용할 수 있는 센서에서 수집한 주관적인 이미지를 이용하고, 이를 통해 자동화된 정확한 데이터 수집이 가능합니다.

- **Technical Details**: 제안된 방법은 'screen time tracker (STT)'라는 착용형 카메라와 'vision language model (VLM)'를 결합하여 여러 화면 장치에서 화면 존재를 식별하는 시스템을 구축합니다. 특히, 다양한 각도에서 수집된 자기 중심 이미지를 기반으로 한 다중 뷰 VLM을 개발하여, 동적 화면 노출 해석을 가능하게 하였습니다.

- **Performance Highlights**: 연구 결과, 기존 방식에 비해 유의미한 성능 향상을 보여주었으며, 어린이의 자연스러운 환경에서의 화면 노출 연구를 최적화할 수 있는 가능성을 입증하였습니다.



### Language Supervised Human Action Recognition with Salient Fusion: Construction Worker Action Recognition as a Use Cas (https://arxiv.org/abs/2410.01962)
- **What's New**: 이 논문은 skeleton과 visual cue를 결합한 새로운 Human Action Recognition (HAR) 접근 방식을 제안합니다. 이를 위해 언어 모델을 사용하여 skeleton encoder의 특성 추출 과정을 안내하고, learnable prompts를 사용해 최적화를 진행합니다.

- **Technical Details**: 제안된 방법은 dual-modality 특성을 결합하는 fusion mechanism을 포함하며, attention과 transformer 메커니즘을 활용하여 높은 차원의 modality 문제를 해결합니다. 또한, construction sites에 적합한 새로운 VolvoConstAct dataset을 도입하여 머신러닝 모델의 훈련 및 평가를 지원합니다.

- **Performance Highlights**: 제안된 방법은 NTU-RGB+D, NTU-RGB+D120 및 NW-UCLA와 같은 공개 데이터셋에서 promising한 성과를 나타냈으며, 다양한 응용 프로그램에서의 Robustness와 가능성을 보여줍니다.



### One-step Noisy Label Mitigation (https://arxiv.org/abs/2410.01944)
Comments:
          20 pages, 4 figures, 11 Tables

- **What's New**: 본 논문에서는 고차원 직교성(high-dimensional orthogonality)의 특성을 활용하여 깨끗한 샘플과 노이즈 샘플을 구분하기 위한 강력하고 효과적인 경계를 제안합니다. 모델 의존성이 없으며, 비용 효율적인 One-step Anti-Noise (OSA) 기법을 도입했습니다.

- **Technical Details**: One-step Anti-Noise (OSA)는 입력 쌍의 노이즈 수준을 단일 단계의 추론(inference)으로 평가하는 모델 비종속적(noise mitigation) 패러다임입니다. 이 과정에서 고차원 직교성을 고려한 스코어링 함수(score function)를 활용하여 각 샘플의 손실에 학습 가중치를 수동적으로 할당합니다.

- **Performance Highlights**: OSA의 실험 결과는 다양한 벤치마크, 모델 및 작업에서 훈련 강건성(training robustness)과 작업 전이성(task transferability)을 향상시키고, 배포의 용이성(ease of deployment)과 계산 비용(computational costs)의 절감을 보여줍니다.



### Deep learning assisted high resolution microscopy image processing for phase segmentation in functional composite materials (https://arxiv.org/abs/2410.01928)
- **What's New**: 이 논문은 배터리 연구에서 고해상도 미세구조 이미지를 처리하기 위한 새로운 워크플로우를 제안합니다. 이 방법은 U-Net segmentation 모델을 활용하여 원시 전송 전자 현미경(TEM) 이미지에서 구성 요소 및 상(segmenting phase)을 감지하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법론에서, 훈련된 U-Net 모델을 사용하여 고해상도 TEM 이미지를 분석합니다. 이 모델은 시간적 및 인지적 요구를 줄여주며 기계적 검토가 필요한 임상 이미지의 범위를 즐겁게 처리할 수 있도록 합니다.

- **Performance Highlights**: 이 접근 방식은 배터리 분야를 넘어 다양한 관련 도메인에 적용 가능한 효율적인 이미지 분석 방법을 제시합니다. 여기에는 합금 생산과 같은 상 및 조성 분포가 특징인 분야가 포함됩니다.



### A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation (https://arxiv.org/abs/2410.01912)
Comments:
          25 pages, 20 figures, code is open at this https URL

- **What's New**: 이번 연구는 벡터 양자화 (Vector Quantization, VQ) 기반의 자율 회귀 (Autoregressive) 이미지 생성을 위해 새로운 모델 아키텍처인 2차원 자율 회귀 (2-Dimensional Autoregression, DnD) Transformer를 도입하여 정보를 잃는 병목 현상을 해결합니다.

- **Technical Details**: DnD-Transformer는 모델 깊이 (model depth)라는 새로운 자율 회귀 방향을 도입하여 이미지에 대해 더 많은 코드를 예측하도록 설계되었습니다. 기존의 1차원 자율 회귀와 RQ-Transformer와 같은 2차원 이미지 분해 방식을 비교했을 때, DnD-Transformer는 엔드-투-엔드 모델로 동일한 백본 모델 크기와 시퀀스 길이로 더 높은 품질의 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과는 DnD-Transformer가 세밀한 이미지 세부 사항을 우수하게 재구성하며, 1D 방법과 비교해 더 효율적이고 낮은 엔트로피 분해를 보여주었습니다. 또한 ImageNet 256x256 생성에서 AR 기본 모델보다 크게 향상된 성능 (최대 1.54 FID 및 82.6 IS 개선)을 달성하였으며, 다중 모드 모델링에 대한 자율 회귀 모델의 뚜렷한 장점을 강조합니다.



### Social Media Authentication and Combating Deepfakes using Semi-fragile Invisible Image Watermarking (https://arxiv.org/abs/2410.01906)
Comments:
          ACM Transactions (Digital Threats: Research and Practice)

- **What's New**: 이 논문에서는 미디어 인증을 위한 새로운 반 취약(semifragile) 이미지 워터마킹 기법을 제안합니다. 이 방법은 얼굴 변조(facial manipulation) 또는 조작에 대해서는 취약하지만, 일반적인 이미지 처리 작업에는 강건성을 지니고 있습니다.

- **Technical Details**: 제안된 워터마킹 프레임워크는 비가시(invisible) 비밀 메시지를 실 이미지에 삽입하며, 비판자(critic) 및 적대적 네트워크(adversarial network)를 포함한 독특한 아키텍처를 통해 높은 이미지 품질과 워터마킹 제거에 대한 저항력을 보장합니다. 실험을 통해 다양한 Deepfake 데이터셋에서 워터마크의 복구 정확도가 높은 것으로 나타났습니다.

- **Performance Highlights**: 제안된 모델은 64비트 비밀 정보를 비가시적인 이미지 워터마크로 삽입할 수 있으며, 일반 이미지 처리 이후에도 높은 복구 정확성을 유지합니다. 반면에, Deepfake 변조에는 복구되지 않으며, 여러 화이트박스(white-box) 및 블랙박스(black-box) 워터마크 제거 공격에 대해 높은 저항성을 보입니다.



### OCC-MLLM-Alpha:Empowering Multi-modal Large Language Model for the Understanding of Occluded Objects with Self-Supervised Test-Time Learning (https://arxiv.org/abs/2410.01861)
Comments:
          Accepted by ECCV 2024 Observing and Understanding Hands in Action Workshop (5 pages, 3 figures, 2 tables). arXiv admin note: substantial text overlap with arXiv:2410.01261

- **What's New**: 기존의 다중 모달 모델들이 occluded objects(차폐된 객체)의 이해에 한계가 있는 문제를 해결하기 위해, OCC-MLLM-Alpha라는 새로운 다중 모달 대형 언어 모델을 제안합니다. 이 모델은 3D 생성 지원을 포함한 자기 지도 학습 전략을 사용합니다.

- **Technical Details**: OCC-MLLM-Alpha는 CLIP 모델과 제안된 3D 모델로 구성된 비주얼 인코더 모듈을 포함하고 있으며, 차폐 손실(occlusion loss)을 계산하여 대규모 다중 모달 언어 모델의 차폐 인식 훈련 전략을 제안합니다. 입력은 이미지와 텍스트로 구성되며, 비주얼 토큰은 LLM 입력의 일부로 사용됩니다.

- **Performance Highlights**: SOMVideo 대규모 데이터셋을 기반으로 한 실험에서 OCC-MLLM-Alpha는 기존의 최첨단 VLM 모델에 비해 16.92% 향상된 결과를 보여주었습니다.



### Spatial Action Unit Cues for Interpretable Deep Facial Expression Recognition (https://arxiv.org/abs/2410.01848)
Comments:
          4 pages, 2 figures, AI and Digital Health Symposium 2024, October 18th 2024, Montréal

- **What's New**: 이 논문은 얼굴 표정 인식(FER)에서 최첨단 분류기들이 가지는 해석 가능성 부족 문제를 해결하기 위해 새로운 학습 전략을 제안합니다. 제안된 방법은 공간적 행동 단위(spatial action units, AUs)를 명시적으로 분류기 훈련에 통합하여 딥 해석 가능 모델을 훈련하는 것을 목표로 합니다.

- **Technical Details**: 학습 과정에서는 AU 코드북을 사용하고 입력 이미지의 표정 라벨 및 얼굴 랜드마크와 함께 AU 히트맵을 구성하여 표정과 관련된 가장 차별화된 이미지 영역을 식별합니다. 이러한 공간적 단서를 활용하여 FER를 위한 딥 해석 가능 분류기를 훈련시키며, 복합 손실(composite loss)을 통해 AU 히트맵과 상관된 해석 가능한 시각적 주의(layer-wise attention)를 생성합니다. 이 전략은 수동 주석 없이 이미지 분류(label)만으로 감독을 수행할 수 있습니다.

- **Performance Highlights**: RAF-DB와 AffectNet와 같은 두 개의 공공 벤치마크에서 광범위한 평가를 통해 제안된 전략이 분류 성능 저하 없이 층별 해석 가능성을 개선할 수 있음을 보여줍니다. 또한, 클래스 활성화 맵(class activation mapping, CAM) 방법을 사용하는 해석 가능 분류기들의 유형을 탐색하고, 제안된 방법이 CAM 해석 가능성을 개선할 수 있음을 입증합니다.



### EgoAvatar: Egocentric View-Driven and Photorealistic Full-body Avatars (https://arxiv.org/abs/2410.01835)
- **What's New**: 본 논문에서는 첫 번째로 사람에 맞춤화된 egocentric telepresence 접근 방식을 제안합니다. 이는 단일 egocentric 비디오를 기반으로 photorealistic 디지털 아바타를 생성하고 조작하는 기술로, 여태까지 해결되지 않았던 문제를 해결합니다.

- **Technical Details**: 제안된 시스템(EgoAvatar)은 단일 RGB 카메라를 통해 실시간 애니메이션이 가능한 photorealistic full-body 아바타를 만들 수 있습니다. 구조적으로 MotionDeformer와 GaussianPredictor의 조합을 통해 표면 변형과 모양 및 외형을 예측하며, IKSolver를 통해 관절 각도를 안정화합니다. 또한, EgoPoseDetector를 통해 pose를 추출하고 최적화된 표면을 생성합니다.

- **Performance Highlights**: 실험 결과, EgoAvatar는 기존 방법들과 비교하여 정량적 및 정성적으로 우수한 성능을 보이며, 처음으로 egocentrically-driven, photorealistic, full-body 아바타 렌더링을 증명했습니다. 새로 제안된 데이터셋은 다중 카메라 영상과 monocular egocentric 비디오를 paired로 제공하며, 4K 해상도를 지원합니다.



### Analysis of Convolutional Neural Network-based Image Classifications: A Multi-Featured Application for Rice Leaf Disease Prediction and Recommendations for Farmers (https://arxiv.org/abs/2410.01827)
- **What's New**: 이번 연구에서는 8개의 서로 다른 convolutional neural network (CNN) 알고리즘을 사용하여 쌀 병해 분류를 개선하는 새로운 방법을 제시합니다. 이로 인해 정밀 농업(precision agriculture) 분야가 한층 발전할 것으로 기대됩니다.

- **Technical Details**: Tkinter 기반의 애플리케이션은 농부들에게 다양한 기능을 제공하는 사용자 친화적인 인터페이스를 제공합니다. 이 애플리케이션을 통해 농부들은 실시간 병해 예측 및 개인화된 추천을 통해 시기적절하고 정보에 기반한 결정을 내릴 수 있습니다. ResNet-50, InceptionV3, VGG16, MobileNetV2와 같은 최신 CNN 전이 학습(transfer learning) 알고리즘이 UCI 데이터셋과 통합되어 현대 농업 관행의 개선을 가져옵니다.

- **Performance Highlights**: 결과적으로 ResNet-50은 75%, DenseNet121은 90%, VGG16은 84%, MobileNetV2는 95.83%, DenseNet169은 91.61%, InceptionV3는 86%의 정확도를 기록하였습니다. 그러나 VGG19는 70%, Nasnet는 80.02%로 과적합(overfitting)을 보였고, ResNet101에서는 54%의 정확도, EfficientNetB0에서는 33%에 불과했습니다. MobileNetV2로 훈련된 모델은 Tkinter GUI 애플리케이션에 성공적으로 배치되어 이미지 또는 실시간 비디오 캡처를 통해 예측을 수행하였습니다.



### PixelBytes: Catching Unified Representation for Multimodal Generation (https://arxiv.org/abs/2410.01820)
- **What's New**: 이번 보고서는 PixelBytes라는 새로운 다중 모달 (multimodal) 표현 학습 접근 방식을 소개합니다. 이 방법은 텍스트, 오디오 및 픽셀화된 이미지 (sprites)를 포함한 다양한 입력을 포괄적으로 담아내기 위해 여러 기존의 시퀀스 모델을 참고하여 개발되었습니다.

- **Technical Details**: PixelBytes는 Recurrent Neural Networks (RNNs), State Space Models (SSMs), Attention 기반 모델 등 여러 아키텍처를 실험했습니다. 특히, bidirectional 처리 및 convolutional PxBy 임베딩 기술에 중점을 두었습니다. 장기 기억 (Long Short-Term Memory, LSTM) 네트워크를 예측 (predictive) 및 자기 회귀 (autoregressive) 모드에서 평가하여 모델의 성능을 비교했습니다.

- **Performance Highlights**: 실험 결과, 자기 회귀 모델이 예측 모델에 비해 더 우수한 성능을 보였습니다. PixelBytes는 다중 모달 데이터 이해 및 생성이 가능한 기초 모델의 개발에 기여하고 있습니다.



### From Experts to the Public: Governing Multimodal Language Models in Politically Sensitive Video Analysis (https://arxiv.org/abs/2410.01817)
- **What's New**: 이 논문은 정치적으로 민감한 비디오에 대한 다중모달 대형 언어 모델(MM-LLMs)의 거버넌스를 개인 및 집합적 논의를 통해 분석합니다. 연구는 전문가 비디오 해석에 대한 이해를 말하는 기자와의 인터뷰를 통해 시작되었고, 일반 대중이 민주적 의사 결정 메커니즘을 통해 논의에 참여했습니다.

- **Technical Details**: 이 연구의 114명의 일반 대중을 대상으로 한 실험은 전문가와 대중 간의 비디오 해석에서의 감정, 객관성, 사실적 명확성 등의 차이를 보여주었습니다. 또한, 다양한 거버넌스 메커니즘(예: quadratic voting, weighted ranking)을 통해 AI의 행동에 대한 사용자 결정에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 결과적으로, quadratic voting은 리버럴 민주주의와 정치적 평등에 대한 인식을 높였으며, AI에 대한 긍정적인 시각을 가진 참가자들은 이 과정이 높은 참여 민주주의를 마케팅하는 것으로 인식했습니다. 이러한 발견은 DAO 메커니즘을 통해 AI 거버넌스의 민주화 가능성을 제시합니다.



### Automatic Scene Generation: State-of-the-Art Techniques, Models, Datasets, Challenges, and Future Prospects (https://arxiv.org/abs/2410.01816)
Comments:
          59 pages, 16 figures, 3 tables, 36 equations, 348 references

- **What's New**: 이 논문은 자동 장면 생성(automatic scene generation) 분야의 최신 연구 동향을 포괄적으로 검토하고, 기계 학습(machine learning), 심층 학습(deep learning), 임베디드 시스템(embedded systems), 자연어 처리(NLP) 등의 기술을 활용한 다양한 생성 기법을 조망합니다.

- **Technical Details**: 연구는 VARIATIONAL AUTOENCODERS (VAEs), GENERATIVE ADVERSARIAL NETWORKS (GANs), TRANSFORMERS 및 DIFFUSION MODELS의 네 가지 기본 모델로 분류되어 각 모델의 하위 모델과 그 기여를 상세히 탐구합니다. 또한, COCO-Stuff, Visual Genome 및 MS-COCO와 같은 데이터셋을 살펴보아 이러한 모델을 훈련하고 평가하는 데 필수적입니다.  장면 생성 방법론에는 이미지-3D 변환(image-to-3D conversion), 텍스트-3D 생성(text-to-3D generation), UI/레이아웃 디자인(ui/layout design), 그래프 기반 방법(graph-based methods), 대화형 장면 생성(interactive scene generation) 등이 포함됩니다.

- **Performance Highlights**: 모델 성능 평가는 FRECHET INCEPTION DISTANCE (FID), KULLBACK-LEIBLER (KL) DIVERGENCE, INCEPTION SCORE (IS), INTERSECTION OVER UNION (IoU), MEAN AVERAGE PRECISION (mAP) 등의 평가 지표를 활용합니다. 연구 결과는 현실성 유지, 다중 객체를 포함한 복잡한 장면 처리 등 여러 도전 과제를 다루고 있으며, 기존의 성과와 비교해 최신 발전을 요약하여 자동 장면 생성 분야에서의 연구자와 실무자에게 유용한 자료를 제공합니다.



### Privacy-Preserving SAM Quantization for Efficient Edge Intelligence in Healthcar (https://arxiv.org/abs/2410.01813)
- **What's New**: 이번 논문에서는 데이터 없이 양자화(quantization) 매개변수를 학습할 수 있는 DFQ-SAM(데이터 없는 양자화 모델)을 제안합니다. 이 모델은 의료 데이터 프라이버시를 보장하면서 효율적인 모델 압축을 지원합니다.

- **Technical Details**: DFQ-SAM은 패치 유사성(patch similarity)과 진화하는 의사 양성 레이블(pseudo-positive labels) 생성을 결합하여 세그멘테이션을 위한 데이터 합성을 제공합니다. 또한, 저비트 양자화를 위한 정확도를 보장하기 위해 스케일 재파라미터화(scale reparameterization) 기법을 도입했습니다. 정확한 양자화 보정을 위한 이상적인 구성 요소들을 통합하여, 여러 데이터셋에서 강력한 일반화 성능을 보여주었습니다.

- **Performance Highlights**: DFQ-SAM은 4비트 양자화 시 모델 크기를 8배 줄이고 계산 복잡성을 64배 감소시키면서 AbdomenCT1K 데이터세트에서 정확도가 2.01%만 감소하는 성능을 보였습니다. 이 기술은 클라우드-엣지 협업에서 데이터 전송을 필요 없애고, 민감한 데이터를 외부 공격으로부터 보호합니다.



### AlzhiNet: Traversing from 2DCNN to 3DCNN, Towards Early Detection and Diagnosis of Alzheimer's Diseas (https://arxiv.org/abs/2410.02714)
- **What's New**: 본 연구에서는 알츠하이머병(Alzheimer's disease, AD) 진단을 위한 새로운 하이브리드 딥 러닝 프레임워크를 제안합니다. 이 프레임워크는 2D Convolutional Neural Networks (2D-CNN)과 3D Convolutional Neural Networks (3D-CNN)를 통합하여, 맞춤형 손실 함수(custom loss function)와 입체적 데이터 증강(volumetric data augmentation)을 이용해 기능 추출 및 분류 성능을 향상시킵니다.

- **Technical Details**: 하이브리드 네트워크 AlzhiNet은 단독 2D 및 3D 모델을 초월하여 서로 보완적인 데이터 표현을 결합하는 중요성을 보여줍니다. 모델 성능은 Kaggle과 MIRIAD 데이터세트에서 확인되었으며, MRI 영상으로 각각 98.9%와 99.99%의 정확도와 100%의 AUC를 기록하였습니다. 또한 주어진 perturbation 시나리오에서 ResNet-18보다 더 강력한 성능을 발휘하여 실제 적용에 적합성이 높습니다.

- **Performance Highlights**: 본 프레임워크는 초기 진단 및 치료 계획 수립에서 알츠하이머병에 대한 중요한 발전을 보여줍니다. AlzhiNet은 다양한 노이즈 및 변형 조건에서도 안정적인 성능을 유지하였으며, 향후 연구에서 최적의 결과를 얻기 위해 하이브리드 예측의 가중치 선택이 필수적임을 시사합니다.



### Lie Algebra Canonicalization: Equivariant Neural Operators under arbitrary Lie Groups (https://arxiv.org/abs/2410.02698)
Comments:
          40 pages; preprint

- **What's New**: 이 논문에서는 Lie aLgebrA Canonicalization (LieLAC)이라는 새로운 접근 방식을 제안합니다. 이는 대칭 그룹의 극소생성자의 작용만을 활용하여 완전한 그룹 구조에 대한 지식이 필요 없게 하여 다양한 비선형 편미분 방정식(PDE)을 해결하는 데 혁신적인 방법입니다.

- **Technical Details**: LieLAC는 기존 이론에서 제기된 문제를 해결하고, 연속 비콤팩트 그룹에서의 프레임 평균화와의 연결을 설정합니다. 이 방식은 기존의 제약이 없는 사전 훈련된 모델에 쉽게 통합될 수 있으며, 입력 데이터를 정형화하여 모델 추론을 위한 허용된 대칭에 따라 정렬합니다.

- **Performance Highlights**: 제안된 방법은 불변 이미지 분류 및 Lie 점 대칭 변환 신경 PDE 솔버 작업에서 효과를 입증하였습니다. 사전 훈련된 모델을 활용하여 더욱 효율적인 학습과 강건성을 제고했습니다.



### Measuring and Improving Persuasiveness of Generative Models (https://arxiv.org/abs/2410.02653)
- **What's New**: 이 논문은 대규모 자동화된 벤치마크 및 퍼션 효과를 평가하기 위한 방법론을 소개하는 최초의 연구로, LLM(대규모 언어 모델)의 설득 능력을 측정하는데 중점을 둡니다. 이 연구는 PersuasionBench 및 PersuasionArena라는 새로운 시스템을 개발하여 AI가 생성한 설득 메시지의 효과를 정량적으로 분석할 수 있도록 합니다.

- **Technical Details**: PersuasionBench와 PersuasionArena는 LLM의 설득 능력을 측정하기 위한 다양한 과제를 포함하는 첫 번째 대규모 벤치마크로, LLM이 생성하는 콘텐츠의 설득력을 정량적으로 평가하고 비교할 수 있도록 합니다. 이 연구에서는 LLM이 언어 패턴을 이용해 설득적인 언어를 생성하는 능력을 극대화하는 방법을 조사하였습니다.

- **Performance Highlights**: 연구 결과, LLM의 설득력은 모델의 크기와 긍정적인 상관관계를 보였지만, 작은 모델도 적절한 훈련을 통해 큰 모델보다 더 높은 설득력을 가질 수 있음을 발견했습니다. 또한, 합성 및 자연 데이터셋을 이용한 목표 지향 훈련이 작은 모델의 설득 능력을 유의미하게 향상시킨다는 점이 강조됩니다.



### Why Sample Space Matters: Keyframe Sampling Optimization for LiDAR-based Place Recognition (https://arxiv.org/abs/2410.02643)
Comments:
          20 pages, 15 figures. Submitted

- **What's New**: 이 논문에서는 LiDAR 기반의 장소 인식에서 키프레임 샘플링을 최적화하는 새로운 접근법을 소개합니다. 표준 키프레임 샘플링 방법의 한계를 극복하고, 정보 손실을 최소화하면서 중복성을 줄이는 데 초점을 맞추었습니다.

- **Technical Details**: 제안된 방법은 적응형 슬라이딩 윈도우 프레임워크를 사용하여 중복 키프레임을 제거하면서도 필수 정보를 보존합니다. 또한, 키프레임 디스크립터와 자세 변화 간의 상관관계를 분석하고, 주성분을 따라 디스크립터를 변환하여 효율적인 장소 인식을 보장합니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터 세트에서 실험적으로 검증되었으며, 다양한 환경에서도 매개변수 조정 없이도 강력한 성능을 유지합니다. 이는 자원 제약이 있는 플랫폼에서의 실시간 배치에 적합함을 의미합니다.



### Diffusion-based Extreme Image Compression with Compressed Feature Initialization (https://arxiv.org/abs/2410.02640)
- **What's New**: 이번 연구에서는 Relay Residual Diffusion Extreme Image Compression (RDEIC) 방법을 제안하여 극단적인 이미지 압축에서의 충실도(fidelity)와 효율성을 향상시킵니다. RDEIC는 압축된 특징 초기화(compressed feature initialization)와 잔여(diffusion) 과정을 결합하여 이미지 압축의 성능을 크게 개선합니다.

- **Technical Details**: RDEIC는 순수한 노이즈(pure noise) 대신 노이즈가 추가된 압축된 잠재 특징(compressed latent features)으로부터 역확산(reverse diffusion)을 시작하여 디노이징 과정을 단축시키고 초기의 불필요한 단계를 제거합니다. 이 과정에서 잔여(diffusion) 제거 및 복원 과정을 통해 원본 이미지를 재구성하며, 학습된 안정적인 diffusion 모델을 통합하여 높은 품질의 재구성을 위한 강력한 생성 능력을 활용합니다. 또한 고정 단계 미세 조정 전략(fixed-step fine-tuning strategy)을 도입하여 학습 및 추론 사이의 간극을 줄입니다.

- **Performance Highlights**: RDEIC는 낮은 비트 전송률(bitrate)에서도 최상의 시각적 품질을 달성하며, 기존의 diffusion 기반 극단적인 이미지 압축 방법들과 비교해 충실도(fidelity)와 효율성(efficiency) 면에서 뛰어난 성능을 보입니다.



### Plots Unlock Time-Series Understanding in Multimodal Models (https://arxiv.org/abs/2410.02637)
Comments:
          49 pages

- **What's New**: 이번 연구는 멀티모달(multi-modal) 기초 모델들이 텍스트를 넘어서 다양한 데이터 모드와 함께 작동할 수 있지만 의료, 금융, 사회과학 분야에서 방대한 양의 다차원 시계열 데이터 분석에 충분히 활용되지 않고 있다는 점을 지적합니다.

- **Technical Details**: 기존의 비전 인코더(vision encoders)를 활용하여 시계열 데이터가 플롯(plot)을 통해 시각화됨으로써 추가적인 모델 훈련 없이도 성능을 향상시킬 수 있는 방안을 제시합니다. 이 방법은 특히 노이즈가 있는 데이터를 포함한 복잡한 작업에서 효과적이며, GPT와 Gemini 모델 가족 모두에서 고성능을 기록합니다.

- **Performance Highlights**: 플롯 기반 접근 방식이 텍스트 기반 접근 방식보다 최대 120%의 성능 향상을 보여주며, 실제 세계 작업에서는 최대 150%의 성능 향상을 달성했습니다. 또한, 비전 토큰(vision tokens)을 사용함으로써 모델 API 비용을 최대 90%까지 절감할 수 있음을 확인했습니다.



### High-Efficiency Neural Video Compression via Hierarchical Predictive Learning (https://arxiv.org/abs/2410.02598)
- **What's New**: DHVC 2.0, 향상된 Deep Hierarchical Video Compression,은 새로운 계층적 예측 코딩 방식을 통해 단일 모델 신경 비디오 코덱을 개발하여 기존 비디오 코덱보다 우수한 압축 성능과 효율성을 제공하고, 실시간 처리 및 낮은 메모리 사용을 가능하게 합니다.

- **Technical Details**: 각 비디오 프레임은 계층적 변이 오토인코더를 통해 다중 스케일 표현으로 변환됩니다. 낮은 스케일의 공간 피처를 참조하여 잠재적 잔여 변수를 생성하며, 이는 조건부 엔트로피 인코딩 기술을 통해 전달됩니다. 이 과정은 이동 추정 및 보상 기술 없이 이루어지며, 각 스케일에서 시간 종속성을 효과적으로 통합합니다.

- **Performance Highlights**: DHVC 2.0은 HEVC보다 높은 압축 성능을 보여주며, 메모리 사용량을 약 4배 줄이고 인코딩 및 디코딩 속도를 10배 이상 가속화합니다. 또한 패킷 손실이 있는 네트워크 환경에서도 부드러운 스트리밍을 제공하여 인터넷 비디오 스트리밍 제공자에게 유리합니다.



### Combining Pre- and Post-Demosaicking Noise Removal for RAW Video (https://arxiv.org/abs/2410.02572)
Comments:
          16 pages, 9 figures

- **What's New**: 본 논문은 Bayer-patterned CFA 비디오 데이터에 대해 사전 및 사후 demosaicking 과정에서 가중치를 두는 자기 유사성 기반의 노이즈 제거 방법을 제안합니다. 이 방법은 두 단계 간의 균형을 통해 이미지 품질을 향상시키고, 고도 노이즈 환경에서 사전 demosaicking의 영향을 높이는 것의 이점을 empirically 보여줍니다.

- **Technical Details**: 제안하는 방법은 센서에서의 노이즈 모델 추정만으로 작동하며, 다양한 노이즈 수준에 쉽게 적응할 수 있습니다. 또한, 각 노이즈 제거기 전에 시간적 궤적(prefiltering) 단계를 통합하여 텍스처 복원을 더욱 개선했습니다. 본 연구는 고전적인 노이즈 제거 알고리즘의 두 단계를 조합하여 ISO 수준에 따라 필터링 정도를 조절합니다.

- **Performance Highlights**: 제안된 방법은 다양한 종류의 시퀀스와 노이즈 수준에 적응 가능성이 높으며, 현대 신경망이 여전히 도전하는 부분에서도 우수한 성능을 보여줍니다. 시뮬레이션 결과, 전통적인 노이즈 제거 알고리즘이 최신 심층 학습 모델의 품질과 동등하거나 더 나은 결과를 생성할 수 있음을 입증하였습니다.



### NestedMorph: Enhancing Deformable Medical Image Registration with Nested Attention Mechanisms (https://arxiv.org/abs/2410.02550)
Comments:
          Submitted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문은 T1 가중치 MRI(T1-weighted MRI)와 확산 MRI(diffusion MRI) 데이터 간의 변형 가능 이미지 등록을 개선하기 위해 NestedMorph라는 새로운 네트워크를 제안합니다. 이 네트워크는 Nested Attention Fusion 접근 방식을 사용하여 지역 및 글로벌 특징 추출을 강화합니다.

- **Technical Details**: NestedMorph는 인코더의 고해상도 공간 정보를 디코더의 의미 정보와 통합하여 다중 스케일 프레임워크로 구성되어 있습니다. 이 모델은 CNN 방식이나 Transformer 기반 모델과 비교할 수 없는 높은 성능을 보여줍니다. 특히, Continuous Deformation Information Flow를 활용하여 변형 필드의 추정을 향상시키고, 저해상도 및 고해상도 처리 결합을 통해 보다 정확한 표현을 달성합니다.

- **Performance Highlights**: NestedMorph는 HCP 데이터셋에서 SSIM(Structural Similarity Index)에 대해 0.89의 최고치를 기록하며, HD95와 SDlogJ의 값도 각각 2.5와 0.22로, 기존의 방법들에 비해 우수한 성능을 나타냅니다. 이러한 결과는 지역 및 글로벌 이미지 특징을 효과적으로 캡처하여 우수한 등록 성능을 달성하였음을 강조합니다.



### A Foundation Model for the Solar Dynamics Observatory (https://arxiv.org/abs/2410.02530)
- **What's New**: SDO-FM은 NASA의 태양 동역학 관측소(SDO)로부터의 데이터를 활용하는 기반 모델로, 세 가지 별도의 기기를 통합하여 태양의 복잡한 물리적 상호작용을 다중 모드 임베딩 공간으로 캡슐화합니다. 이 모델은 SDO와 관련된 대용량 데이터셋을 보다 쉽게 접근할 수 있도록 하여 헬리오피직스(heliophysics) 연구를 촉진할 수 있습니다.

- **Technical Details**: SDO-FM의 구축 과정은 (1) 데이터 준비, (2) 대형 기반 모델(FM) 훈련, (3) 임베딩 추출, (4) 과학적 검증을 위한 미세 조정 또는 직접 임베딩 사용의 네 단계로 구성됩니다. 모델은 주로 autoencoder 아키텍처를 기반으로 하며, SDO 데이터셋의 복잡성을 다루기 위해 다양한 기능 공학 옵션을 평가했습니다.

- **Performance Highlights**: SDO-FM은 NASA의 SDO에서 제공하는 14년간의 고해상도 데이터를 활용하여 과학적 검증 가능성을 제고했습니다. 이를 통해 태양의 자기장, 폭발, 극자외선 복사와 같은 주요 태양 물리 현상을 탐구할 수 있으며, 향후 헬리오스피어(Heliosphere)에 미치는 영향을 분석하는 미세 조정 응용 프로그램 개발이 가능해집니다.



### Med-TTT: Vision Test-Time Training model for Medical Image Segmentation (https://arxiv.org/abs/2410.02523)
- **What's New**: 이 논문에서는 의료 영상 분할(Medical Image Segmentation)에서의 계산 복잡성과 장거리 종속성 모델링의 한계를 해결하기 위해 Med-TTT라는 새로운 네트워크 아키텍처를 소개합니다. 이 모델은 Test-Time Training (TTT) 레이어를 통합하여 동적 조정 기능을 포함합니다.

- **Technical Details**: Med-TTT는 Vision-TTT 레이어를 도입하여, 선형 계산 복잡성으로 장거리 종속성을 효과적으로 모델링하고, 추론 중에 매개변수를 적응적으로 조정합니다. 또한 다양한 스케일에서 이미지 기능을 결합하는 다중 해상도 융합(multi-resolution fusion) 메커니즘을 설계하였고, 고주파 필터링을 기반으로 한 주파수 도메인 특징 향상 전략을 채택하였습니다.

- **Performance Highlights**: 실험 결과, Med-TTT 모델은 여러 의료 이미지 데이터셋에서 기존 방법들에 비해 우수한 성능을 보여주었으며, 정확도(accuracy) 96.07%, 평균 교차 점유율(mIoU) 78.83%, Dice 계수(DSC) 88.16%를 기록하였습니다. 이는 특히 복잡한 이미지 배경에서의 강력한 분할 능력을 나타냅니다.



### Towards a Theoretical Understanding of Memorization in Diffusion Models (https://arxiv.org/abs/2410.02467)
Comments:
          arXiv admin note: text overlap with arXiv:2406.12752

- **What's New**: 이 논문은 Conditional DPM(확률적 확산 모델)과 Unconditional DPM에서 훈련 데이터의 메모리 현상을 이론적으로 분석하고 새로운 데이터 추출 방법인 Surrogate condItional Data Extraction (SIDE)을 제시합니다.

- **Technical Details**: 이 논문에서는 DPM의 메모리화 현상을 정량화하기 위해 새로운 메모리 지표(metric)를 도입하며, 훈련 데이터와 생성된 데이터 간의 중첩 정도를 측정하여 이를 설명하는 이론적 프레임워크를 제시합니다. 또한, 메모리 분석을 기반으로 하여 조건부 및 비조건부 DPM에서 훈련 데이터를 효과적으로 추출할 수 있는 방법을 탐구합니다.

- **Performance Highlights**: SIDE 방법은 CIFAR-10 및 다양한 CelebA 데이터셋의 스케일에서 테스트되었으며, 기존 방법보다 50% 이상 효과적임을 입증했습니다. 이는 불리한 시나리오에서도 훈련 데이터를 성공적으로 추출할 수 있는 능력을 보여줍니다.



### MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation (https://arxiv.org/abs/2410.02458)
Comments:
          Submitted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 연구는 교육된 대형 언어 모델(LLMs)로부터 얻은 frozen transformer 블록을 Vision Transformer (ViT) 기반 모델의 인코더에 통합하여 의료 이미지 분할 성능을 크게 향상시켰습니다. 이 과정에서 Hybrid Attention Mechanism과 Multi-Scale Fusion Block을 도입해 다양한 스케일에서의 특징 집계를 통해 성능을 대폭 개선했습니다.

- **Technical Details**: 연구에서는 여러 공개 LLM 모델인 Meta-Llama-3.1-8B, Gemma-2-9B 등에서 사전 훈련된 가중치를 사용하였고, 이를 ViT의 인코더와 디코더 사이에 통합했습니다. Hybrid Attention Mechanism을 통해 전역 및 대응 주의(a ttention) 학습을 조화롭게 이루어지도록 설정하였으며, 다양한 의료 이미징 모달리티에서 효과를 검증하기 위해 ablation study를 실시했습니다.

- **Performance Highlights**: 이 모델의 Segmentation 성능은 평균 Dice score이 0.74에서 0.79로 증가하였고, 정밀도와 Jaccard Index에서도 현저한 개선을 보였습니다. 이러한 결과는 LLM 기반 transformer의 효과성을 입증하며, 의료 이미지 분석에서 모델의 정확성과 강건성을 크게 향상할 수 있는 가능성을 보여줍니다.



### Predictive Attractor Models (https://arxiv.org/abs/2410.02430)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 연구에서는 생물학적 가능성을 갖춘 신경과학 이론에 영감을 받아 새로운 시퀀스 메모리 아키텍처인 Predictive Attractor Models (PAM)을 제안합니다. PAM은 연속적인 입력을 온라인으로 관찰함으로써 한 번의 입력만으로 학습하며, 기존 메모리 모델들이 겪는 치명적인 망각, 제한된 용량, 느린 반복 학습 절차 등의 문제를 개선하고자 합니다.

- **Technical Details**: PAM은 상태 예측 모델(state prediction model)과 생성적 끌어당김 모델(generative attractor model)로 구성되어 있으며, 히에라키컬 템포럴 메모리(HTM) 학습 알고리즘에 기초합니다. 이 모델은 네트워크가 입력의 맥락을 유일하게 인코딩하며, 대칭 학습(symmetric learning) 및 노이즈 저항을 통해 여러 가능한 미래 예측을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: PAM은 시퀀스 용량, 시퀀스 생성, 치명적인 망각, 노이즈 강인성 등 여러 작업에서 광범위한 평가를 받았으며, 기존의 최첨단 모델들이 실패했던 조건들을 충족하는 성과를 보여 주었습니다. 이 연구는 인지 과학 및 인공지능 연구에 대한 광범위한 의미를 지닙니다.



### Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models (https://arxiv.org/abs/2410.02416)
- **What's New**: 본 논문에서는 classifier-free guidance (CFG) 업데이트 규칙을 재검토하고, 오버샤팅(oversaturation) 및 비현실적 아티팩트(artifacts) 문제를 해결하기 위한 수정 사항을 도입하였습니다. 새로운 접근 방식인 adaptive projected guidance (APG)를 제안하여 높은 품질의 생성 결과를 유지하면서도 높은 가이드 스케일을 사용할 수 있도록 합니다.

- **Technical Details**: APG는 CFG에서 제공하는 품질 향상의 이점을 유지하면서 오버샤팅을 방지하는 방법입니다. 이 접근 방식은 조건부 모델 예측에 대한 업데이트 항(term)을 병렬(parallel) 및 직교(orthogonal) 구성 요소로 분해하여, 병렬 구성 요소가 주로 오버샤팅을 초래하고 직교 구성 요소가 이미지 품질을 향상시킨다는 점을 관찰했습니다. 또한 CFG와 경량화(gradient ascent) 간의 연결고리를 제시하고, 이를 바탕으로 새로운 재조정(rescaling) 및 모멘텀(momentum) 방식을 도입했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 APG가 다양한 조건부 확산 모델(conditional diffusion models) 및 샘플러(samplers)와 호환되며, FID, recall 및 saturation 점수를 개선하는 동시에 CFG와 유사한 정밀도를 유지함을 입증했습니다. APG는 표준 classifier-free guidance보다 설치 및 사용이 간편한 우수한 대안으로 평가받고 있습니다.



### MetaMetrics: Calibrating Metrics For Generation Tasks Using Human Preferences (https://arxiv.org/abs/2410.02381)
Comments:
          Preprint

- **What's New**: MetaMetrics라는 새로운 메타 메트릭(metric)을 도입하여, 다양한 작업을 평가하는 데 있어 인간의 선호도에 더 잘 맞추도록 기존 메트릭들을 조정하는 방법을 제안합니다.

- **Technical Details**: MetaMetrics는 감독된 방법으로 캘리브레이션(calibration)된 메타 메트릭으로, 언어 및 비전 관련 다운스트림 작업에서 유연성과 효과성을 보여줍니다. 이 메트릭은 참조 기반(reference-based) 및 비참조(reference-free) 두 가지 설정에서 작동하여 다양한 작업과 모달리티(modality)에 걸쳐 널리 활용될 수 있습니다.

- **Performance Highlights**: MetaMetrics는 여러 다국어 및 다영역(multi-domain) 시나리오에서 인간 선호도와 밀접하게 정렬되는 효과를 보여주며, 기존 메트릭들과 비교했을 때 성능이 월등히 우수함을 입증했습니다.



### Structural-Entropy-Based Sample Selection for Efficient and Effective Learning (https://arxiv.org/abs/2410.02268)
Comments:
          Submitted to ICLR 2025

- **What's New**: 이 논문은 샘플 선택의 효율성과 효과성을 개선하기 위해 텍스트의 정보와 대표성을 갖춘 샘플을 선택하는 새로운 방법을 제안합니다. 기존의 방법은 주로 로컬 정보에 의존하였으나, 본 연구는 전역 정보와 로컬 정보를 결합하여 샘플을 선택하는 방법을 개발합니다.

- **Technical Details**: 본 연구에서는 샘플을 그래프로 모델링하여 노드와 엣지를 사용하여 샘플 간의 유사성을 표현합니다. 전역 정보를 정량화하기 위해 구조적 엔트로피(structural entropy)를 활용하며, 이를 샤플리 값(Shapley value)을 사용해 개별 노드로 분해합니다. 새로 제안한 샘플 선택 방법인 SES는 샘플들의 kNN-그래프를 구성하고, 구조적 엔트로피와 학습 난이도를 결합하여 샘플의 중요도를 측정합니다.

- **Performance Highlights**: 세 가지 학습 시나리오(감독 학습, 능동 학습, 지속 학습)에서 진행된 실험에서 SES 방법이 기존 방법들에 비해 항상 우수한 성능을 보였습니다. 이는 본 방법이 선택된 샘플의 정보성과 대표성을 효과적으로 개선함을 보여줍니다.



### Capturing complex hand movements and object interactions using machine learning-powered stretchable smart textile gloves (https://arxiv.org/abs/2410.02221)
- **What's New**: 연구자들은 처음으로 신축성 있는 스마트 섬유 장갑을 사용하여 손과 손가락 움직임을 정확하게 추적할 수 있는 방법을 제시하였습니다. 이 장갑은 헬리컬 센서 실과 관성 측정 장치(IMUs)를 내장하여 역동적인 추적을 가능하게 합니다.

- **Technical Details**: 스마트 장갑은 높은 동적 범위를 가지며, 0.005%에서 155%의 변형에 안정적으로 반응합니다. 다중 단계 기계 학습 알고리즘을 통해 관절 각도 추정의 평균 루트 평균 제곱 오차(RMSE)는 각각 1.21도와 1.45도로 보고되었습니다. 고가의 모션 캡처 카메라와 유사한 정확도를 유지하며, 센서 노이즈와 변동성에 대한 강인성을 높이기 위한 데이터 증강 기법도 적용되었습니다.

- **Performance Highlights**: 이 연구는 다양한 응용 가능성을 보여줍니다. 예를 들어, 모의 종이 키보드에서의 정확한 타이핑(97.80% 정확도), 복잡한 동적 제스처 인식(최대 97.31% 정확도), 정적 제스처 인식 및 물체 인식을 통한 그립 패턴 인식에서도 각각 90.20% 이상의 정확도를 기록하였습니다.



### Stochastic Sampling from Deterministic Flow Models (https://arxiv.org/abs/2410.02217)
Comments:
          Submitted to ICLR 2025

- **What's New**: 이번 논문에서는 결정론적 흐름 모델(Deterministic Flow Models)의 한계를 극복하기 위해, 이를 확률적 미분 방정식(Stochastic Differential Equations, SDEs)으로 확장하는 새로운 방법을 제안합니다. 이 방법은 이전에 훈련된 결정론적 모델을 바탕으로 하여 확률적 샘플링을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 결정론적 흐름 모델의 기본 ODE를 SDE 가족으로 변환하여, 다양한 샘플링 스펙트럼을 제공하며, 흐름 장(flow field)과 점수 함수(score function)에 접근할 수 있을 때 적용됩니다. 이 과정에서, 기존의 결정론적 샘플러와는 다른 추가적인 자유도를 제공하여 불확실성을 감소시킬 수 있습니다.

- **Performance Highlights**: 실험적으로, toy Gaussian 설정과 대규모 ImageNet 생성 작업에서 제안된 확률적 샘플러가 기존의 결정론적 샘플러보다 우수한 성능을 보였으며, 생성의 다양성을 조절하는 추가적인 수단을 제공함을 보여주었습니다.



### From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities (https://arxiv.org/abs/2410.02155)
- **What's New**: 최근 연구에서 멀티모달 대형 언어 모델(MLLMs)의 발전을 위해 새로운 BPE 이미지 토크나이저를 도입하여 비주얼(visual) 데이터와 텍스트 모달리티 간의 효과적인 정렬 문제를 해결했습니다. 이 접근 방식은 기존의 방법들과는 달리 구조적 선행 정보(structural prior information)를 이미지 토큰에 직접적으로 통합합니다.

- **Technical Details**: 제안된 BPE 이미지 토크나이저는 두 차원 시퀀스 데이터를 효과적으로 학습할 수 있는 새로운 패러다임을 제공합니다. 이미지 패치를 패턴에 따라 결합하여 언어 모델 해석기에 더 의미 있는 정보가 포함된 토큰을 공급함으로써 transformer 모델이 이미지 데이터에 대한 더 나은 이해를 할 수 있게 합니다. 이 방법은 데이터 훈련 빈도를 기반으로 토큰을 병합하여 구조적 우선 정보를 추가하는 데 중점을 두며, 토큰화(tokenization) 과정을 통해 특정 손실을 줄일 수 있다는 것을 이론적으로도 입증하였습니다.

- **Performance Highlights**: 실험을 통해 제안된 BPE 이미지 토크나이저가 MLLM의 멀티모달 이해 능력을 크게 향상시키며, 훈련 데이터가 제한된 상황에서도 성능을 개선하는 것으로 나타났습니다. 이 방법은 다양한 벤치마크에서 성능이 향상될 뿐만 아니라, 확장성(scalability)도 뛰어난 가능성을 보여줍니다.



### MDSGen: Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Generation (https://arxiv.org/abs/2410.02130)
Comments:
          21 pages, 16 figures

- **What's New**: MDSGen은 비전 기반(open-domain) 소리 생성을 위한 새로운 프레임워크로, 파라미터 사이즈, 메모리 사용량 및 추론 속도를 최적화합니다. 이 프레임워크는 (1) 불필요한 시각 정보를 필터링하는 비디오 특징 제거 모듈과 (2) 오디오 생성 정확도를 향상시키기 위한 시간 인식 마스킹 전략을 포함합니다.

- **Technical Details**: MDSGen은 경량화된 masked diffusion transformers를 사용하여 기존의 Unet 기반 모델보다 효율적인 생성을 가능하게 합니다. Temporal-Awareness Masking (TAM)을 통해 오디오 모드에 특화된 마스킹을 적용하고, 비디오 특징의 중복 정보를 제거하는 Reducer 모듈을 통해 더 정제된 특징을 생성합니다.

- **Performance Highlights**: MDSGen의 최소 모델(5M 파라미터)은 97.9%의 정렬 정확도를 달성하며, 현재의 860M 파라미터 모델보다 172배 적은 파라미터를 사용하고, 371% 적은 메모리로 36배 빠른 추론을 제공합니다. 더 큰 모델(131M 파라미터)은 거의 99% 정확도에 도달하며, 6.5배 적은 파라미터를 필요로 합니다.



### DMC-Net: Lightweight Dynamic Multi-Scale and Multi-Resolution Convolution Network for Pancreas Segmentation in CT Images (https://arxiv.org/abs/2410.02129)
Comments:
          14 pages, 4 figures

- **What's New**: 본 논문에서는 의료 이미지 분할에서의 CNN(Convolutional Neural Networks)의 한계를 극복하기 위한 두 가지 새로운 모듈, 즉 동적 다중 해상도 컨볼루션(Dynamic Multi-Resolution Convolution, DMRC)과 동적 다중 스케일 컨볼루션(Dynamic Multi-Scale Convolution, DMSC)을 제안합니다. 이 두 모듈은 다양한 스케일의 특징과 글로벌 컨텍스트 정보를 활용하여 CNN의 표현 능력을 증가시킵니다.

- **Technical Details**: DMRC 모듈은 서로 다른 해상도의 이미지에 컨볼루션 필터를 적용하여 글로벌 피처 간의 상호 의존성을 모델링 합니다. 반면 DMSC 모듈은 서로 다른 커널 크기를 가진 컨볼루션을 사용해 다양한 스케일의 피처를 추출하고, 글로벌 컨텍스트 정보를 동적으로 구조화합니다. 이러한 모듈은 표준 U-Net 아키텍처에 통합되어 쉽게 사용될 수 있도록 설계되었습니다.

- **Performance Highlights**: DMC-Net 네트워크는 두 개의 잘 알려진 복부 CT 췌장 분할 벤치마크에서 평가되었고, 기존의 최첨단 방법들보다 우수한 분할 성능을 달성했습니다. 특히 DMSC 및 DMRC 모듈이 결합된 DMC-Net은 표준 U-Net 기준 모델에 비해 상당한 성능 향상을 보여주었습니다.



### Tracking objects that change in appearance with phase synchrony (https://arxiv.org/abs/2410.02094)
- **What's New**: 본 논문은 생물학적 시각 시스템이 변화하는 개체를 추적하는 데 있어 주의(attention)와 신경 동기화(neural synchrony)가 중요한 역할을 한다는 가설을 컴퓨테이셔널 테스트를 통해 검증합니다. 특히, 복소수 가치가 포함된 재귀 신경망(CV-RNN)을 소개하여, 특정 객체의 특징과 위치를 분리하여 주의력을 제어할 수 있는 방법을 제안합니다.

- **Technical Details**: 연구에서는 CV-RNN을 사용하여 시각적 주의력을 구현하고, FeatureTracker라는 대규모 과제를 통해 인간과 CV-RNN, 그리고 다른 심층 신경망(DNNs)의 개체 추적 성능을 비교합니다. CV-RNN은 신경 동기화 기법을 활용하여 더 복잡한 시각적 루틴을 확장하고, 생물학적 시스템의 성능에 근접하는 결과를 보였습니다.

- **Performance Highlights**: 인간은 FeatureTracker 도전 과제에서 매우 높은 정확도로 객체를 추적했습니다. 반면, 최첨단 DNN들은 이 과제를 해결하는 데 어려움을 겪었으며, CV-RNN이 인간의 성능에 도달하는 모습을 보였습니다. 이 결과는 신경 동기화가 생물학적 시각 시스템에서 객체 추적에 기여한다는 개념을 컴퓨터 모델에서도 구현할 수 있음을 입증합니다.



### Anchors Aweigh! Sail for Optimal Unified Multi-Modal Representations (https://arxiv.org/abs/2410.02086)
- **What's New**: 이 논문은 정적 앵커(modality)를 사용한 Binding 방법의 한계를 제시하고, 새로운 동적 센트로이드 기반 앵커를 사용하는 CentroBind 방안을 제안합니다.

- **Technical Details**: CentroBind는 고정 앵커가 아닌 모든 가용한 모달에서 생성된 동적으로 조정 가능한 센트로이드 기반 앵커를 활용하여 더 균형 잡힌 표현 공간을 제공합니다. 이 방법은 intra-modal learning, inter-modal learning, multimodal alignment의 세 가지 중요 특성을 포착합니다.

- **Performance Highlights**: 실험 결과, CentroBind는 기존의 고정 앵커(binding) 방법들과 비교하여 더 미세한 다중 모달 상호작용을 포착하며, 데이터셋에서의 분류 및 검색 작업에서 우수한 성능을 보였습니다.



### Posterior sampling via Langevin dynamics based on generative priors (https://arxiv.org/abs/2410.02078)
- **What's New**: 본 연구에서는 고차원 공간에서의 posterior sampling을 위한 효율적인 방법을 제안합니다. 사전 훈련된 생성 모델의 노이즈 공간에서 Langevin dynamics를 시뮬레이션하여 다양한 posterior 샘플을 생성하는 접근 방식을 활용하고 있습니다.

- **Technical Details**: 이 방법은 noise와 data 공간 간의 맵핑을 활용하여 posterior 공간을 탐색하며, 기존의 전체 샘플링 프로세스를 다시 수행할 필요가 없도록 합니다. 이를 통해 계산 비용을 크게 줄이고, 고충실도 샘플을 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 제한된 함수 평가 수에서도 높은 충실도와 개선된 의미적 다양성을 지닌 샘플을 생성할 수 있으며, 기존의 diffusion 기반 posterior sampling 기법에 비해 우수한 효율성과 성능을 보여줍니다.



### Kolmogorov-Arnold Network Autoencoders (https://arxiv.org/abs/2410.02077)
Comments:
          12 pages, 5 figures, 1 table

- **What's New**: 최근 연구에서 다층 퍼셉트론(MLP)을 대체할 유망한 방법으로 Kolmogorov-Arnold Networks(KAN)가 소개되었습니다. KAN은 노드가 아닌 엣지에 활성화 함수를 배치하는 구조적 변화를 통해 모델의 정확성과 해석 가능성을 향상시킬 수 있습니다.

- **Technical Details**: KAN의 구조는 Kolmogorov-Arnold 표현 정리에 기반하며, 네트워크의 엣지에서 직접 활성화 함수를 학습하고 적용합니다. 이는 정보 흐름의 역학을 변화시키고, 복잡한 데이터의 의존성을 더 잘 처리할 수 있도록 도와줍니다. KAN은 이미지 표현 작업을 위한 오토인코더(encoder-decoder) 아키텍처에 적용됩니다.

- **Performance Highlights**: KAN 기반 오토인코더는 MNIST, SVHN, CIFAR-10 데이터셋에서 전통적인 CNN과 비교하여 경쟁력 있는 재구성 정확도를 달성하는 것을 보여주었습니다. 이는 KAN이 데이터 분석 작업에서 효과적인 도구로 자리 잡을 가능성을 시사합니다.



### Improving Autonomous AI Agents with Reflective Tree Search and Self-Learning (https://arxiv.org/abs/2410.02052)
- **What's New**: 이번 논문에서는 Reflective Monte Carlo Tree Search (R-MCTS)라는 새로운 알고리즘을 소개하여, GPT-4o 기반 AI 에이전트가 복잡한 멀티 스텝 의사결정 작업을 신속하게 탐색할 수 있도록 향상시키고 있습니다.

- **Technical Details**: R-MCTS는 전통적인 MCTS를 확장하여, 1) 과거 상호작용으로부터 학습하고 검색 효율성을 동적으로 개선할 수 있는 대조적 반영(contrastive reflection) 기법과 2) 신뢰할 수 있는 상태 평가를 위한 다중 에이전트 토론(multi-agent debate) 기법을 통합합니다.

- **Performance Highlights**: R-MCTS를 통해 훈련된 GPT-4o 기반 에이전트는 VisualWebArena 벤치마크에서 이전의 최첨단 모델과 비교하여 6%에서 30%까지 성능이 향상되었으며, fine-tuning 과정을 통해 97%의 성능을 유지하면서도 테스트 시간 동안 계산 비용을 4배 줄였습니다.



### FeelAnyForce: Estimating Contact Force Feedback from Tactile Sensation for Vision-Based Tactile Sensors (https://arxiv.org/abs/2410.02048)
Comments:
          8 pages, 4 figures, 4 tables

- **What's New**: 이 논문은 비전 기반 촉각 센서를 사용하여 3D 접촉력을 추정하는 문제에 접근하고 있으며, 다양한 객체에서 최대 15N의 넓은 범위의 접촉력을 추정할 수 있는 방법을 목표로 하고 있습니다. 로봇 팔을 사용하여 다양한 인덴터를 GelSight Mini 센서에 압력 적용하여 20만 개 이상의 자료를 수집하여 Multi-Head Transformer 모델을 훈련했습니다.

- **Technical Details**: 제안된 시스템은 RGB 촉각 이미지와 깊이 이미지를 출력하는 GelSight Mini 센서를 사용하여, 레이블된 3D 힘 벡터, 촉각 이미지 및 깊이 이미지를 포함하는 대규모 데이터셋을 수집합니다. 이 모델은 Vision Transformer(ViT) 기반의 구조로 사전훈련된 DINOv2를 후속 조정하여 3D 힘을 추정합니다.

- **Performance Highlights**: 훈련된 모델은 보지 않은 실제 객체 데이터셋에서 평균 절대 오차가 4%로 뛰어난 일반화 능력을 보여줍니다. 또한, 제안된 보정 프로시저를 통해 다른 비전 기반 센서에도 적용 가능하며, 물체 무게 측정 및 섬세한 물체의 변형 제어와 같은 실제 작업에서도 흥미로운 성과를 달성했습니다.



### Semi-Supervised Contrastive VAE for Disentanglement of Digital Pathology Images (https://arxiv.org/abs/2410.02012)
- **What's New**: 이 연구에서는 해부학 이미지 해체(disentanglement) 방식의 새로운 접근법을 제안합니다. 특히, Tumor-Infiltrating Lymphocytes (TIL) 검출 작업에 중점을 두고 있으며, 기존의 방법들이 복잡한 이미지를 해체하는데 한계를 가지고 있음을 강조합니다.

- **Technical Details**: 제안하는 Semi-Supervised Contrastive VAE (SS-cVAE) 방법은 첫 번째로 배경(tissue) 패치와 세포가 있는 패치를 구분하고, 두 번째로 낮은 TIL 수의 패치와 많은 TIL이 포함된 패치를 구분하는 카스케이드 방식의 해체 접근법을 포함합니다. GAN 기반의 ID-GAN 재구성 모듈을 도입하여 재구성 성능을 향상시킵니다.

- **Performance Highlights**: 이 연구는 복잡한 병리 이미지를 대상으로 한 해체 성능에서 SOTA (state-of-the-art) 접근법에 비해 우수한 결과를 보여주었으며, 해체된 잠재 특성이 다양한 데이터셋에서 강한 일반화 능력을 갖추고 있음을 입증했습니다.



### MONICA: Benchmarking on Long-tailed Medical Image Classification (https://arxiv.org/abs/2410.02010)
- **What's New**: 이 논문은 Medical OpeN-source Long-taIled ClassifiCAtion (MONICA)라는 새로운 장기 분포 의료 이미지 분류 벤치마크를 소개합니다. 이 벤치마크는 6개 의료 영역에 걸쳐 12개의 장기 분포 데이터셋에서 30가지 이상의 방법론을 평가합니다.

- **Technical Details**: MONICA는 모듈화된 코드베이스로, 데이터 증강(data augmentation), 손실 함수(loss functions), 최적화 전략(optimization strategies) 및 분산 훈련(distributed training) 등의 구성 요소를 포함합니다. 이 코드는 연구자들이 맞춤형 LTMIC에 적합한 방법론을 찾고 적용할 수 있도록 지원합니다.

- **Performance Highlights**: 이 연구는 기존의 의료 이미지 분류 방법론이 서로 다른 데이터셋에서 평가되어 발생하는 비교의 불일치 문제를 해결합니다. MONICA는 다양한 기준선에 대해 연구자들이 모델의 효과성을 객관적으로 평가할 수 있는 robust한 프레임워크를 제공합니다.



### Addressing Data Heterogeneity in Federated Learning with Adaptive Normalization-Free Feature Recalibration (https://arxiv.org/abs/2410.02006)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 고객 데이터의 통계적 이질성 문제를 해결하기 위한 Adaptive Normalization-free Feature Recalibration (ANFR) 라는 새로운 접근 방식을 제안합니다. ANFR은 가중치 표준화(weight standardization)와 채널 주의(channel attention) 메커니즘을 결합한 아키텍처 수준의 방법입니다.

- **Technical Details**: ANFR은 가중치 표준화를 통해 각 레이어의 가중치를 정규화하여 비일관적인 클라이언트 통계의 영향을 줄입니다. 채널 주의는 각 피처 맵에 대해 학습 가능한 스케일링 요인을 생성해 이질성으로 인해 비일관적인 피처를 억제하고 일관된 피처를 강조합니다. 이 과정은 모델이 클라이언트 간에 공유되며 유용한 피처에 집중할 수 있도록 합니다.

- **Performance Highlights**: ANFR은 다양한 데이터 세트와 집합 방법(aggregation methods)에서 기존의 기준선(baselines)보다 일관되게 우수한 성능을 보였습니다. 이 접근 방식은 FL의 다양한 시나리오에서 성능, 안정성 및 개인 정보 보호 기능을 향상시키는 데 기여합니다.



### A Novel Feature Extraction Model for the Detection of Plant Disease from Leaf Images in Low Computational Devices (https://arxiv.org/abs/2410.01854)
Comments:
          10 Pages, 8 figures, 1 table

- **What's New**: 본 연구에서는 저비용 컴퓨팅 시스템(예: 모바일폰)을 이용하여 토마토 식물 질병을 식별하기 위한 혁신적인 특징 추출 접근법을 제안합니다. 이 접근법은 딥러닝(Deep Learning) 기술을 통합하여 잎 사진에서 강력하고 차별화된 특징을 추출합니다.

- **Technical Details**: 데이터셋은 10,000장의 잎 사진으로 구성되어 있으며, 10개의 토마토 질병 클래스와 1개의 건강한 잎 클래스를 포함합니다. AlexNet, ResNet50, VGG16, VGG19, MobileNet과 같은 다섯 개의 최첨단 딥러닝 모델에 대한 비교 실험이 수행되었고, 특히 AlexNet이 87%의 정확도를 달성하였습니다.

- **Performance Highlights**: AlexNet은 경량화되어 임베디드 시스템 및 스마트폰과 같은 저처리 장치에서 사용하기에 적합하며, 빠른 처리 속도로 농민이 신속하게 질병을 감지하고 대응할 수 있도록 돕습니다.



### Image-to-Image Translation Based on Deep Generative Modeling for Radiotherapy Synthetic Dataset Creation (https://arxiv.org/abs/2410.01828)
- **What's New**: 이 연구에서는 방사선 치료에 필요한 EPID(전자 포털 영상 장치) 이미지의 합성 데이터를 개선하기 위해 이미지-이미지 변환(I2I) 기법을 활용하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 989개의 예측된 EPID 이미지와 해당하는 측정 EPID 이미지로 구성된 데이터셋을 사용하였으며, 쌍 데이터 및 비쌍 데이터 생성 모델링 접근 방식을 평가했습니다. 기존의 Variational Autoencoder(VAE)를 변형하여 쌍 데이터에 대한 I2I 변환을 수행하였으며, 비쌍 데이터에 대해서는 UNsupervised Image-to-Image Translation Networks(UNIT)를 사용했습니다.

- **Performance Highlights**: VAE 모델의 변형을 통해 I2I 변환을 향상시켰으며, 주요 지표에서 UNIT 모델보다 더 나은 성능을 보였습니다(평균 절대 오차: 4.1 cGy vs 6.4 cGy; 상대 용량 차이: 2.5% vs 5.5%; 절대 용량 차이: 5.3 cGy vs 10.8 cGy). 이 향상된 합성 데이터는 방사선 치료에서 자동 오류 검출 및 분류를 위한 신경망 훈련에 기여할 것으로 기대됩니다.



### Leopard: A Vision Language Model For Text-Rich Multi-Image Tasks (https://arxiv.org/abs/2410.01744)
Comments:
          Our code is available at this https URL

- **What's New**: 이번 연구에서는 Leopard라는 새로운 멀티모달 언어 모델(Multimodal Large Language Model, MLLM)을 소개합니다. Leopard는 텍스트가 중심 시각 요소로 작용하는 멀티 이미지 작업을 처리하기 위해 특별히 설계되었습니다. 이 모델은 고품질의 멀티모달 데이터와 적응형 고해상도 멀티 이미지 인코딩 모듈을 사용하여 기존의 한계점을 극복합니다.

- **Technical Details**: Leopard는 약 100만 개의 고품질 멀티모달 교육 데이터를 사용하여 학습하였으며, 이 데이터는 다중 페이지 문서, 다중 차트, 다중 표 및 웹 페이지 경로와 같은 실제 시나리오를 포괄합니다. 또한, 이미지의 원본 비율과 해상도를 기반으로 시각적 시퀀스 길이를 최적화하는 적응형 고해상도 멀티 이미지 인코딩 모듈을 통해 여러 고해상도 이미지를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: Leopard는 13개의 비전-언어 벤치마크 데이터셋에서 실험을 수행하였으며, 5개의 텍스트 풍부한 멀티 이미지 벤치마크에서 이전 최상의 오픈 소스 MLLM보다 평균 +9.61 포인트 향상된 성과를 보였습니다. 또한, 단일 이미지 및 일반 도메인 비전-언어 벤치마크에서도 높은 경쟁력을 유지하며, 최신 MLLM과 비교해도 유사한 성과를 거두었습니다.



### Releasing the Parameter Latency of Neural Representation for High-Efficiency Video Compression (https://arxiv.org/abs/2410.01654)
- **What's New**: 본 논문은 Implicit Neural Representation (INR) 기반 비디오 압축 방법의 정보 보존 능력을 최적화하기 위해 파라미터 재사용(Reuse) 기법을 탐구하고, 이를 통해 압축 성능 향상을 이루는 방법을 제안합니다.

- **Technical Details**: INR은 비디오를 기본 단위로 모델링하여, 공간적 및 시간적 중복성을 제거하는 compact neural network를 활용합니다. 본 논문에서는 네트워크의 깊이를 증가시키고 파라미터 재사용을 통해 표현력을 높이는 방안을 제안하며, 다양한 비디오 데이터셋에서 실험을 통해 그 유효성을 검증합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 INR 기반 비디오 압축 방법과 비교하여 비율-왜곡 성능(rate-distortion performance)을 유의미하게 향상시켰습니다. 또한, 다양한 설정에서 동작하여 비디오 품질을 극대화할 수 있는 잠재력을 보여주었습니다.



New uploads on arXiv(cs.AI)

### Grounded Answers for Multi-agent Decision-making Problem through Generative World Mod (https://arxiv.org/abs/2410.02664)
Comments:
          The Thirty-eighth Annual Conference on Neural Information Processing Systems

- **What's New**: 최근 생성 모델(generative models)의 발전으로 이미지 생성과 챗봇 등 다양한 분야에서 혁신적인 변화가 이루어졌습니다. 하지만 복잡한 다중 에이전트 의사결정 문제에서는 인간처럼 시행착오(trial-and-error)를 경험하지 못해 불완전한 해답을 생성하는 한계가 있습니다. 이 연구는 언어 안내 시뮬레이터(language-guided simulator)를 다중 에이전트 강화 학습(multi-agent reinforcement learning) 파이프라인에 통합하여 이러한 문제를 해결하는 새로운 패러다임을 제안합니다.

- **Technical Details**: 제안된 시뮬레이터는 동적 모델과 보상 모델로 구성되며, 동적 모델은 이미지 토크나이저(image tokenizer)와 인과 변환기(causal transformer)로 구성되어 상호작용 전환을 자회귀적으로 생성합니다. 보상 모델은 전문가 시연의 트레일을 언어 안내에 따라 최대화하여 학습된 양방향 변환기(bidirectional transformer)입니다. 이를 통해 정책(policy) 네트워크를 업데이트하고, 의사결정 문제에 대한 답변을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 StarCraft Multi-Agent Challenge 벤치마크에서 훈련 및 이전에 본 적이 없는 작업에 대해 우수한 성능을 보여 다중 에이전트 의사결정 문제에 대한 답변을 개선하는 것으로 나타났습니다. 특히, 지속적인 상호작용 시퀀스와 상호작용 상태에서 설명 가능한 보상 기능을 생성하여 향후 생성 모델 학습의 새로운 가능성을 열었습니다.



### Plots Unlock Time-Series Understanding in Multimodal Models (https://arxiv.org/abs/2410.02637)
Comments:
          49 pages

- **What's New**: 이번 연구는 멀티모달(multi-modal) 기초 모델들이 텍스트를 넘어서 다양한 데이터 모드와 함께 작동할 수 있지만 의료, 금융, 사회과학 분야에서 방대한 양의 다차원 시계열 데이터 분석에 충분히 활용되지 않고 있다는 점을 지적합니다.

- **Technical Details**: 기존의 비전 인코더(vision encoders)를 활용하여 시계열 데이터가 플롯(plot)을 통해 시각화됨으로써 추가적인 모델 훈련 없이도 성능을 향상시킬 수 있는 방안을 제시합니다. 이 방법은 특히 노이즈가 있는 데이터를 포함한 복잡한 작업에서 효과적이며, GPT와 Gemini 모델 가족 모두에서 고성능을 기록합니다.

- **Performance Highlights**: 플롯 기반 접근 방식이 텍스트 기반 접근 방식보다 최대 120%의 성능 향상을 보여주며, 실제 세계 작업에서는 최대 150%의 성능 향상을 달성했습니다. 또한, 비전 토큰(vision tokens)을 사용함으로써 모델 API 비용을 최대 90%까지 절감할 수 있음을 확인했습니다.



### Achieving Fairness in Predictive Process Analytics via Adversarial Learning (https://arxiv.org/abs/2410.02618)
Comments:
          17 pages, 5 figures

- **What's New**: 이 논문은 예측 비즈니스 프로세스 분석에서 차별적인 변수(예: 성별 또는 국적)로 인해 불공정한 예측이 발생하는 문제를 해결한다. 저자들은 예측 모델 내에서 차별을 줄이기 위한 adversarial debiasing 프레임워크를 제안하여, 이를 통해 공정성을 확보하면서도 예측의 정확도를 유지할 수 있는 접근 방식을 제시한다.

- **Technical Details**: 제안된 프레임워크는 예측 프로세스의 결과 값을 예측하도록 모델을 훈련하는 동시에, 보호 변수를 정확히 예측하는 것을 막는 adversarial 네트워크를 함께 훈련하여 편향을 줄인다. 연구는 자원, 조직 국가, 성별, 시민권, 구사 언어 등의 보호 변수를 포함하는 네 가지 사례 연구에서 수행된다.

- **Performance Highlights**: 실험 결과는 제안된 프레임워크가 선택된 보호 변수에 대해 공정성을 보장하고, 예측 모델의 정확도가 여전히 높은 수준을 유지함을 보여준다. 또한, 보호 변수와 강한 상관관계를 가진 프로세스 변수에 대한 영향이 줄어들어 단순히 보호 변수를 제거하는 것이 편향을 다른 변수로 전이하는 결과로 이어질 수 있음을 강조한다.



### Intelligence at the Edge of Chaos (https://arxiv.org/abs/2410.02536)
Comments:
          15 pages,8 Figures

- **What's New**: 이 연구에서는 복잡한 규칙 기반 시스템의 복잡성이 인공지능 시스템의 능력에 미치는 영향을 조사하여, 인공지능의 출현을 탐구하였습니다. 특히, Elementary Cellular Automata (ECA)를 이용하였고, 다양한 ECA에 대해 대형 언어 모델(LLM)을 교육하여 규칙의 복잡성과 모델의 지능 간의 관계를 평가하였습니다.

- **Technical Details**: 연구는 256개의 가능한 8비트 규칙을 가진 1차원 이진 상태의 ECA를 기반으로 하였습니다. 각 ECA는 훈련 데이터 생성을 위해 별도의 GPT-2 모델에 전송되어, 다음 상태 예측을 수행하였습니다. 훈련된 모델들의 성능은 논리적 추론 및 체스 수 예측 작업을 통해 평가되었습니다.

- **Performance Highlights**: 복잡성이 높은 규칙은 LLM들이 우수한 성능을 보이는 결과를 가져왔으며, 이는 특히 체스 수 예측 및 추론 작업에서 두드러졌습니다. 복잡도가 적절히 조정된 시스템에서 지능이 발현되는 ‘혼돈의 균형’ 영역을 발견하였고, 단순한 규칙에서 복잡한 해법을 학습할 수 있음을 밝혔습니다.



### A Schema-aware Logic Reformulation for Graph Reachability (https://arxiv.org/abs/2410.02533)
- **What's New**: 이번 연구에서는 그래프 내에서 두 지점이 상호 연결되어 있는지를 이해하는 그래프 도달성(graph reachability) 문제를 다룹니다. 연구자들은 도달성을 개선하기 위한 새로운 접근 방식을 제안하며, 기존의 깊이 우선 및 너비 우선 탐색 알고리즘 대신 구조적 지식을 활용하여 불필요한 경로를 방지하고 목표에 도달하는 경로를 우선시합니다.

- **Technical Details**: 본 논문에서 제안하는 전략은 고차원 개념화를 활용하여 특정 그래프 경로를 자동으로 제외하고 정렬하는 것입니다. 이를 통해 전통적인 알고리즘의 시간, 공간 요구 사항, 백트래킹 횟수를 개선할 수 있는 새로운 1차 논리(First-order logic) 재구성을 목표로 합니다. 실험 결과, 탐색 전략 중 백트래킹 횟수를 줄임으로써 시간과 공간을 절약할 수 있는 이점을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 탐색 과정 중 백트래킹 횟수를 줄여주며, 이는 시간을 절약하고 공간 효율성을 개선하는 데 기여합니다. 기존 탐색 알고리즘들과 비교했을 때, 더 뛰어난 성능을 나타냅니다.



### Choices are More Important than Efforts: LLM Enables Efficient Multi-Agent Exploration (https://arxiv.org/abs/2410.02511)
- **What's New**: 이번 논문은 LEMAE라는 시스템적인 접근 방식을 제안하여, 정보적인 작업 관련 지침을 지닌 대형 언어 모델(LLM)을 통해 효율적인 다중 에이전트 탐색을 지원합니다.

- **Technical Details**: LEMMAE는 (i) LLM을 사용한 키 상태의 위치화(key states localization)와 (ii) 키 상태 지침 기반 탐색(key state-guided exploration)으로 구성됩니다. 또한, Subspace-based Hindsight Intrinsic Reward(SHIR)를 통해 에이전트를 목표 키 상태로 유도하고, Key State Memory Tree(KSMT)를 구축하여 키 상태 간 전이를 추적합니다.

- **Performance Highlights**: LEMAE는 기존의 최첨단(SOTA) 방법들에 비해 10배 가속을 달성하며 복잡한 벤치마크(SMAC 및 MPE)에서 좋은 성능을 보이는 것으로 나타났습니다.



### Can Large Language Models Grasp Legal Theories? Enhance Legal Reasoning with Insights from Multi-Agent Collaboration (https://arxiv.org/abs/2410.02507)
- **What's New**: 본 연구에서는 법 이론 및 복잡한 법적 추론 능력을 평가하기 위해 Confusing Charge Prediction이라는 도전적인 과제를 도입하고, 이를 해결하기 위한 새로운 프레임워크인 Multi-Agent framework for improving complex Legal Reasoning capability (MALR)를 제안합니다.

- **Technical Details**: MALR에서는 비모수적 학습(non-parametric learning)을 적용하여 LLM들이 복잡한 법적 작업을 자동으로 분해하고 법 규칙에서 통찰(insights)을 추출하도록 장려합니다. 이 방식은 LLM들이 단일 법칙에 의존하지 않고, 경험과 피드백을 통해 자기 반성을 통해 학습할 수 있도록 합니다.

- **Performance Highlights**: 여러 실제 사례 데이터셋에서 실시된 실험을 통해 제안된 프레임워크가 실용적인 시나리오에서 복잡한 추론 문제를 효과적으로 해결할 수 있음을 입증했습니다. 이 연구는 법률 분야에서 LLM의 신뢰성 있는 응용 가능성을 향상시키는 데 기여합니다.



### Strong Preferences Affect the Robustness of Value Alignmen (https://arxiv.org/abs/2410.02451)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)과 기타 AI 에이전트의 가치 정렬(value alignment)의 강건성을 탐구합니다. 우리는 선호 모델(preference models)의 민감도를 분석하고 선호 확률의 변화가 다른 선호에 대한 예측에 미치는 영향을 연구합니다.

- **Technical Details**: 연구에서는 브래들리-테리 모델(Bradley-Terry model)과 플라켓-루스 모델(Plackett-Luce model)의 선호 확률이 다른 선호의 변화에 민감하게 반응할 수 있음을 이론적으로 분석합니다. 특히, 선호가 지배적일 때, 즉 확률이 0 또는 1에 가까울 때 민감도가 커진다는 것을 발견했습니다.

- **Performance Highlights**: 이 논문은 선호 모델간의 확률 관계를 명확히 하고, 주요 모델에서 선호의 변화가 다른 선호의 확률에 미치는 영향을 규명하여, AI 시스템의 가치 정렬의 강건성과 안전성에 실질적인 함의를 제공합니다.



### Predictive Attractor Models (https://arxiv.org/abs/2410.02430)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 연구에서는 생물학적 가능성을 갖춘 신경과학 이론에 영감을 받아 새로운 시퀀스 메모리 아키텍처인 Predictive Attractor Models (PAM)을 제안합니다. PAM은 연속적인 입력을 온라인으로 관찰함으로써 한 번의 입력만으로 학습하며, 기존 메모리 모델들이 겪는 치명적인 망각, 제한된 용량, 느린 반복 학습 절차 등의 문제를 개선하고자 합니다.

- **Technical Details**: PAM은 상태 예측 모델(state prediction model)과 생성적 끌어당김 모델(generative attractor model)로 구성되어 있으며, 히에라키컬 템포럴 메모리(HTM) 학습 알고리즘에 기초합니다. 이 모델은 네트워크가 입력의 맥락을 유일하게 인코딩하며, 대칭 학습(symmetric learning) 및 노이즈 저항을 통해 여러 가능한 미래 예측을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: PAM은 시퀀스 용량, 시퀀스 생성, 치명적인 망각, 노이즈 강인성 등 여러 작업에서 광범위한 평가를 받았으며, 기존의 최첨단 모델들이 실패했던 조건들을 충족하는 성과를 보여 주었습니다. 이 연구는 인지 과학 및 인공지능 연구에 대한 광범위한 의미를 지닙니다.



### IoT-LLM: Enhancing Real-World IoT Task Reasoning with Large Language Models (https://arxiv.org/abs/2410.02429)
Comments:
          21 pages, 10 figures, submitted to ICLR 2025 Conference

- **What's New**: 최근 대형 언어 모델(LLMs)은 텍스트 및 시각 도메인에서 뛰어난 성능을 보여주고 있으나, 물리 법칙을 위반하는 결과를 생성하는 우리의 연구는 이러한 모델의 물리 세계 이해의 간극을 해소하고자 IoT 센서 데이터를 활용한 개념을 도입했습니다.

- **Technical Details**: 우리는 LLM의 능력을 향상시키기 위해 IoT 데이터를 LLM에 적합한 형식으로 전처리하고, 상식 지식을 활성화하는 체인 오브 사고 프롬프트를 사용하며, IoT지향 검색 보강 생성(Retrieval-Augmented Generation) 방식을 통합하여 IoT-LLM이라는 통합 프레임워크를 제안합니다.

- **Performance Highlights**: 실험 결과, IoT-LLM은 여러 태스크에서 평균 65% 증가된 성능을 기록했으며, 이는 LLM이 IoT 데이터와 물리 법칙을 더 잘 이해하고 적용할 수 있도록 돕는 효과를 보여줍니다.



### End-to-end Driving in High-Interaction Traffic Scenarios with Reinforcement Learning (https://arxiv.org/abs/2410.02253)
Comments:
          10 pages, 3 figures, experiment under progress, only to demonstrate the originality of the method

- **What's New**: 본 연구는 자율 주행 시스템을 위한 새로운 강화 학습(RL) 알고리즘인 Ramble을 제안합니다. Ramble은 고차원 다중 모드 관측에서 공간적 및 시간적 특징을 효과적으로 추출하고, 최적의 주행 정책으로 수렴하도록 지원합니다.

- **Technical Details**: Ramble은 다중 관점 RGB 이미지와 LiDAR 포인트 클라우드를 저차원 잠재 특징으로 처리하여 각 시간 단계에서의 교통 상황을 포착합니다. 이후, transformer 기반 아키텍처를 통해 시간 의존성을 모델링하고 미래 상태를 예측합니다. 환경의 동역학 모델을 학습하여 다가오는 교통 사건을 예측하고 더 전략적인 결정을 내릴 수 있습니다.

- **Performance Highlights**: Ramble은 CARLA Leaderboard 2.0에서 경로 완료율과 주행 점수에 대한 최첨단 성능을 달성하며, 복잡하고 동적 교통 상황을 효과적으로 관리할 수 있음을 보여줍니다.



### SEAL: SEmantic-Augmented Imitation Learning via Language Mod (https://arxiv.org/abs/2410.02231)
Comments:
          18 pages, 5 figures, in submission

- **What's New**: 본 논문에서는 Long-horizon decision-making(장기 의사결정) 과제를 다루기 위해 개발된 새로운 프레임워크인 SEmantic-Augmented Imitation Learning (SEAL)을 소개합니다. SEAL은 Large Language Models (LLMs)의 강력한 의미적 및 세계 지식을 활용하여 서브 목표를 지정하고 상태를 의미 있는 서브 목표 표현으로 사전 레이블링하는 새로운 방법을 제공합니다.

- **Technical Details**: SEAL은 dual-encoder 구조를 채택하여 LLM-guided 서브 목표 학습(감독 학습)과 VQ(Rectangular Quantization) 기반의 비감독 학습을 결합합니다. 이를 통해 보다 견고한 서브 목표 표현을 형성하고, 낮은 수준의 정책 에이전트는 서브 목표 전환(adaptation)에 개선된 적응력을 갖추도록 설계되어 있습니다.

- **Performance Highlights**: 실험 결과, SEAL은 최신 HIL 방법과 LLM 기반 계획 접근 방식을 초월하며, 특히 작은 전문가 데이터 세트와 복잡한 장기 과제에서 탁월한 성능을 보였습니다.



### CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning (https://arxiv.org/abs/2410.02229)
Comments:
          work in progress

- **What's New**: 본 논문에서는 대규모 고품질 공개 소스 코드로부터 합성된 코드-선호 쌍(code-preference pairs)을 활용하여 LLM의 추론 능력을 향상시킬 수 있는 CodePMP라는 확장 가능한 선호 모델 사전 훈련(pretraining) 파이프라인을 소개합니다.

- **Technical Details**: CodePMP는 코드 관련 프롬프트나 설명을 기반으로 선택된(code responses) 및 거부된(rejected code responses) 코드 응답 쌍을 합성하여 선호 쌍을 생성합니다. 이러한 <chosen, rejected> 쌍은 수백만 개에 달하며, 대규모 합성 선호 데이터셋을 형성합니다. 이 데이터셋은 쌍별 순위 목표(pairwise ranking objectives)를 사용하여 선호 모델을 사전 훈련하는 데 활용됩니다. CodePMP는 GSM8K 및 MATH 같은 수학적 추론 과제와 ReClor 및 LogiQA2.0 같은 논리적 추론 과제에 대해 평가됩니다.

- **Performance Highlights**: 실험 결과, CodePMP는 RM(RM fine-tuning)의 정확도를 크게 향상시키고 다양한 과제에서의 강인성을 높였습니다. CodePMP를 초기화한 RM은 더 높은 성능을 보이면서도, 넓은 범위의 과제에서 개선된 성능을 제공합니다.



### GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning (https://arxiv.org/abs/2410.02203)
- **What's New**: 이번 논문에서는 다단계 추론(multi-step reasoning) 작업에 적합한 새로운 아이디어인 GraphIC를 도입하였습니다. GraphIC는 그래프 기반 표현을 활용하여 인컨텍스트 예제(in-context examples, ICEs)의 선택성을 향상시킵니다. 이는 기존의 텍스트 기반 임베딩 방법의 한계를 극복할 수 있는 접근 방식입니다.

- **Technical Details**: GraphIC는 Bayesian Networks (BNs)를 사용하여 추론 과정을 모델링하고, ICEs의 파라미터형 사고 패턴을 최적화합니다. 이 과정에서 GraphIC는 노드(사고)가 부모 노드의 속성에 의존하는 방식으로 인간의 인지 구조를 쉽게 반영할 수 있습니다. 또한, Graph 구조는 얕은 의미론적 콘텐츠를 필터링하면서 핵심 추론 구조를 보존합니다.

- **Performance Highlights**: GraphIC는 수학적 추론, 코드 생성 및 논리적 추론을 포함한 세 가지 유형의 다단계 추론 작업에서 기존 모델들을 초과하는 성과를 보였습니다. 특히, ICEs의 선택 과정에서의 효과성과 효율성 측면에서 현저한 성장을 보여주었으며, 더 많은 ICEs에 대해 빠른 성능 개선을 나타냈습니다.



### General Preference Modeling with Preference Representations for Aligning Language Models (https://arxiv.org/abs/2410.02197)
Comments:
          34 pages

- **What's New**: 본 논문에서는 인간의 선호도를 모델링하는 새로운 접근 방식을 소개합니다. 기존의 Bradley-Terry (BT) 보상 모델과 비교하여, 제안된 방법은 선호를 효율적으로 캡처하는 잠재 공간(latent space)에 응답을 임베딩하는 preference representation learning을 활용합니다.

- **Technical Details**: 제안된 preference representation learning 방식은 복잡한 선호 구조를 포착할 수 있으며, 선호 측정 시 쿼리 복잡도가 𝒪⁢(K)로, 기존의 𝒪⁢(K2)보다 개선되었습니다. 이를 통해 다양한 응답들 사이의 관계를 효율적으로 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, General Preference representation model (GPM)은 RewardBench 벤치마크에서 BT 보상 모델에 비해 최대 5.6% 향상된 성능을 보였고, AlpacaEval2.0 및 MT-Bench에서도 최대 9.3%의 성능 향상을 달성했습니다.



### Agent-Oriented Planning in Multi-Agent Systems (https://arxiv.org/abs/2410.02189)
- **What's New**: 이번 연구는 다중 에이전트 시스템에서 사용자 쿼리를 효과적으로 분해하고 배분하는 새로운 '에이전트 지향 계획(agent-oriented planning)' 프레임워크를 제안합니다. 이 프레임워크는 해결 가능성(solubility), 완전성(completeness), 비중복성(non-redundancy)라는 세 가지 설계 원칙을 기반으로 하여, 각각의 하위 작업이 효과적으로 해결될 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 빠른 작업 분해(fast task decomposition)와 적절한 에이전트 할당(task allocation) 프로세스를 포함하고 있으며, 보상 모델(reward model)을 통해 효과적이고 효율적인 평가를 수행합니다. 메타 에이전트는 전문가 에이전트의 성능을 평가하고 필요할 경우 하위 작업과 일정을 조정하는 역할을 합니다. 또한, 피드백 루프(feedback loop)를 통합하여 문제 해결 프로세스의 강인성을 향상시킵니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 프레임워크는 기존의 단일 에이전트 시스템 및 다중 에이전트 시스템 계획 전략과 비교했을 때 실질적인 성능 향상을 보여줍니다. 제안된 프레임워크의 다양한 구성 요소의 기여도에 대한 토론도 실시되었으며, 다중 에이전트 시스템 내에서 에이전트 지향 계획의 성능을进一步 개선할 수 있는 가능성도 논의되었습니다.



### A LLM-Powered Automatic Grading Framework with Human-Level Guidelines Optimization (https://arxiv.org/abs/2410.02165)
- **What's New**: 이번 연구에서는 통합 다중 에이전트 자동 단답형 평가(ASAG) 프레임워크인 GradeOpt를 제안하여, 대규모 언어 모델(LLMs)을 활용하여 자동으로 평가 지침을 최적화하는 방법을 소개합니다.

- **Technical Details**: GradeOpt는 LLM을 평가자로 사용하며, 반영 에이전트(reflector)와 정제 에이전트(refiner)라는 두 개의 추가 LLM 기반 에이전트를 포함하여 데이터를 최적화합니다. 이 시스템은 기존 평가 지침을 자동으로 개선하기 위한 자기 반영(self-reflection) 메커니즘을 활용합니다.

- **Performance Highlights**: 실험 결과 GradeOpt는 교육 내용 지식(PCK) 및 내용 지식(CK) 질문의 평가에서 사람 평가자와의 행동 정렬 및 grading accuracy에서 기존 기준보다 우수한 성능을 보였습니다. 또한, 반복적인 실험을 통해 평가 정확도가 지속적으로 향상되는 것을 확인했습니다.



### Planning in Strawberry Fields: Evaluating and Improving the Planning and Scheduling Capabilities of LRM o1 (https://arxiv.org/abs/2410.02162)
Comments:
          arXiv admin note: text overlap with arXiv:2409.13373

- **What's New**: OpenAI의 새로운 O1(스트로베리) 모델은 기존 autoregressive LLM(대형 언어 모델)의 한계를 벗어나기 위해 설계된 큰 추론 모델(LRM)로, 계획 및 스케줄링 벤치마크에서 성능 평가가 이루어졌다. LRM은 Chain-of-Thought(CoT) 이동을 통해 더 나은 추론 능력을 보여준다.

- **Technical Details**: 이 연구에서는 O1 모델의 계획 능력을 평가하기 위해 PlanBench와 TravelPlanner 등의 정립된 벤치마크를 사용하였다. O1 모델은 RL(RL: 강화 학습) 훈련을 통해 추론을 위해 필요한 체계적인 이동을 생성하고 평가할 수 있다. 또한, 외부 검증기와 결합하여 LRM-Modulo 시스템을 통해 출력 결과의 정확성을 보장하고 성능을 향상시킬 수 있다.

- **Performance Highlights**: O1 모델은 기존 autoregressive LLM과 비교하여 매우 높은 성능 향상을 보여주었지만, 높은 추론 비용을 감수해야 하며 여전히 생성 결과에 대한 보장을 제공하지 못한다. LLM-Modulo 접근 방식을 통해 LRM의 성능을 더 개선하고 보장을 제공할 수 있다.



### From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities (https://arxiv.org/abs/2410.02155)
- **What's New**: 최근 연구에서 멀티모달 대형 언어 모델(MLLMs)의 발전을 위해 새로운 BPE 이미지 토크나이저를 도입하여 비주얼(visual) 데이터와 텍스트 모달리티 간의 효과적인 정렬 문제를 해결했습니다. 이 접근 방식은 기존의 방법들과는 달리 구조적 선행 정보(structural prior information)를 이미지 토큰에 직접적으로 통합합니다.

- **Technical Details**: 제안된 BPE 이미지 토크나이저는 두 차원 시퀀스 데이터를 효과적으로 학습할 수 있는 새로운 패러다임을 제공합니다. 이미지 패치를 패턴에 따라 결합하여 언어 모델 해석기에 더 의미 있는 정보가 포함된 토큰을 공급함으로써 transformer 모델이 이미지 데이터에 대한 더 나은 이해를 할 수 있게 합니다. 이 방법은 데이터 훈련 빈도를 기반으로 토큰을 병합하여 구조적 우선 정보를 추가하는 데 중점을 두며, 토큰화(tokenization) 과정을 통해 특정 손실을 줄일 수 있다는 것을 이론적으로도 입증하였습니다.

- **Performance Highlights**: 실험을 통해 제안된 BPE 이미지 토크나이저가 MLLM의 멀티모달 이해 능력을 크게 향상시키며, 훈련 데이터가 제한된 상황에서도 성능을 개선하는 것으로 나타났습니다. 이 방법은 다양한 벤치마크에서 성능이 향상될 뿐만 아니라, 확장성(scalability)도 뛰어난 가능성을 보여줍니다.



### Can LLMs Reliably Simulate Human Learner Actions? A Simulation Authoring Framework for Open-Ended Learning Environments (https://arxiv.org/abs/2410.02110)
- **What's New**: 이 논문은 LLMs(대형 언어 모델)를 사용하여 학습자의 행동을 시뮬레이션하는 새로운 프레임워크인 Hyp-Mix를 소개합니다. 이전 연구들은 LLMs의 잠재력을 보여주었지만, 제한된 신뢰성과 일반화 문제로 인해 초기 단계에서 머물렀습니다. 새로운 프레임워크는 테스트 가능한 가설을 결합하여 시뮬레이션을 개발하고 평가할 수 있게 합니다.

- **Technical Details**: Hyp-Mix는 학습자의 행동에 대한 가설을 포함하여 시뮬레이션을 설계하는을 도와줍니다. LLM을 사용하여 피지컬 학습 환경에서 실험을 수행했고, GPT-4 Turbo는 기본 학습자 모델이 변경되더라도 일관된 행동을 유지하며, LLM이 오픈 엔디드(interactive) 학습 환경에서 현실적인 행동을 시뮬레이션할 수 있음을 보여주는 첫 번째 증거를 제공합니다.

- **Performance Highlights**: Hyp-Mix를 통해 LLM은 자주 사용되는 핵심 행동 패턴을 잘 시뮬레이션하며, 기존의 모델에 비해 신뢰성이 높은 결과를 생성합니다. 이 연구는 오픈 엔디드 학습 환경에서의 LLM 행동 시뮬레이션의 필요성을 강조하며, 교육 전문가들이 이미 익숙한 기본 원리를 활용해 시뮬레이션을 개선할 수 있는 가능성을 보여줍니다.



### Tracking objects that change in appearance with phase synchrony (https://arxiv.org/abs/2410.02094)
- **What's New**: 본 논문은 생물학적 시각 시스템이 변화하는 개체를 추적하는 데 있어 주의(attention)와 신경 동기화(neural synchrony)가 중요한 역할을 한다는 가설을 컴퓨테이셔널 테스트를 통해 검증합니다. 특히, 복소수 가치가 포함된 재귀 신경망(CV-RNN)을 소개하여, 특정 객체의 특징과 위치를 분리하여 주의력을 제어할 수 있는 방법을 제안합니다.

- **Technical Details**: 연구에서는 CV-RNN을 사용하여 시각적 주의력을 구현하고, FeatureTracker라는 대규모 과제를 통해 인간과 CV-RNN, 그리고 다른 심층 신경망(DNNs)의 개체 추적 성능을 비교합니다. CV-RNN은 신경 동기화 기법을 활용하여 더 복잡한 시각적 루틴을 확장하고, 생물학적 시스템의 성능에 근접하는 결과를 보였습니다.

- **Performance Highlights**: 인간은 FeatureTracker 도전 과제에서 매우 높은 정확도로 객체를 추적했습니다. 반면, 최첨단 DNN들은 이 과제를 해결하는 데 어려움을 겪었으며, CV-RNN이 인간의 성능에 도달하는 모습을 보였습니다. 이 결과는 신경 동기화가 생물학적 시각 시스템에서 객체 추적에 기여한다는 개념을 컴퓨터 모델에서도 구현할 수 있음을 입증합니다.



### Zodiac: A Cardiologist-Level LLM Framework for Multi-Agent Diagnostics (https://arxiv.org/abs/2410.02026)
- **What's New**: ZODIAC라는 새로운 LLM 기반 프레임워크를 도입하였으며, 심장 전문의 수준의 전문성을 갖추어 심장 진단 분야에서 LLM의 효과성을 높이기 위해 설계되었습니다.

- **Technical Details**: ZODIAC는 다중 에이전트 협력 프레임워크 기반으로 구축되어 있으며, 실제 환자 데이터와 심장 전문의의 검토를 통해 세부적으로 조정되었습니다. 이 시스템은 다양한 데이터 모달리티를 처리하고, ECG 이미지를 포함한 환자 데이터를 분석합니다.

- **Performance Highlights**: ZODIAC는 OpenAI의 GPT-4o, Meta의 Llama-3.1-405B, Google의 Gemini-pro 등 업계에서 선도하는 모델들보다 뛰어난 성과를 보였으며, 실제 임상 환경에서의 응용 가능성이 입증되었습니다.



### Lost-in-Distance: Impact of Contextual Proximity on LLM Performance in Graph Tasks (https://arxiv.org/abs/2410.01985)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 'lost-in-distance' 현상으로 인해 그래프 작업에서 성능 저하를 겪는다는 점을 밝혔다. 문제 해결에 필요한 관련 정보가 맥락 내에서의 위치에 따라 모델의 성능에 미치는 영향을 다룬다.

- **Technical Details**: LLMs의 성능은 주로 두 가지 기본 그래프 작업, 즉 두 노드 간의 공통 연결 식별과 세 노드의 유사성 평가에서 상대적인 정보 위치에 따라 달라진다. 이 논문은 Llama-3-8B, Llama-3-70B, GPT-4 모델을 사용하여 다양한 그래프 인코딩 기법을 평가하였다. 'lost-in-distance' 현상과 'lost-in-the-middle' 현상이 독립적으로 발생함을 입증하였다.

- **Performance Highlights**: 모델의 정확도는 노드 연결 간의 거리가 멀어질수록 최대 6배까지 감소할 수 있으며, 이는 그래프 인코딩 방식이나 모델 크기에 관계없이 나타나는 보편적인 제한을 시사한다.



### LLM-Augmented Symbolic Reinforcement Learning with Landmark-Based Task Decomposition (https://arxiv.org/abs/2410.01929)
- **What's New**: 이 논문에서는 복잡한 과제를 해결하기 위해 주어진 양수 및 음수 경로를 사용하여 하위 작업을 식별하는 새로운 알고리즘을 소개합니다. 이 알고리즘은 1차 논리(predicate logic)로 상태를 표현하며, 이러한 하위 작업을 달성하기 위한 규칙 템플릿을 생성하기 위해 대형 언어 모델(LLM)을 사용합니다.

- **Technical Details**: 이 연구에서는 강화 학습(Reinforcement Learning, RL) 분야에서 복잡한 작업을 관리하기 위해 하위 작업(subtask) 및 이정표(landmark)를 활용하는 방법론을 다룹니다. 주어진 상태 경로에서 하위 작업을 식별하기 위해 대조 학습(contrastive learning) 알고리즘을 적용하며, LLM을 통해 생성된 규칙 템플릿을 ILP(Inductive Logic Programming) 기반 RL 에이전트를 통해 미세 조정하여 규칙 기반 정책(rule-based policy)을 설정합니다.

- **Performance Highlights**: 실험을 통해 제안된 알고리즘이 모든 하위 작업을 정확하게 식별함을 확인하였고, LLM이 생성한 상식 기반 규칙이 하위 작업을 달성하는 데 필요한 규칙을 생성하는 데 성공적임을 보여주었습니다. 이 연구는 구성 요소가 단순화된 환경 정의에 대한 가정을 최소화하면서 복잡한 작업을 해결하는 데 기여할 수 있는 가능성을 제시합니다.



### Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos (https://arxiv.org/abs/2410.02763)
Comments:
          Project Page: this https URL

- **What's New**: 최근 대형 멀티모달 모델(large multimodal models, LMMs)이 짧은 비디오 이해의 주요 문제를 해결했다고 평가받고 있지만, 실제로는 여러 기본적인 추론 능력이 부족함을 보여줍니다. 이를 개선하기 위해 1000개의 짧고 자연스러운 비디오-자막 쌍을 포함한 새로운 평가 기준인 Vinoground를 소개합니다.

- **Technical Details**: Vinoground는 다양한 동작과 객체 변환의 시간적 차이를 구별하는 능력을 평가합니다. 기존의 모델들이 이러한 능력을 정확히 수행하지 못함을 나타내며, 특히 'single-frame bias' 문제를 해결하기 위해 시간적 반사실(counterfactual) 쌍으로 구성된 데이터를 사용합니다.

- **Performance Highlights**: 대부분의 현대 LMM 모델들은 짧은 비디오 이해에 있어 형편없는 성능을 보이며, 최고 모델인 GPT-4o는 텍스트 및 비디오 점수에서 50% 정도에 불과해 사람의 평균 기준인 90%에 비해 큰 성격 차이를 보입니다. 많은 모델들이 비디오 점수에서 무작위 확률 수준을 기록하였습니다.



### FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models (https://arxiv.org/abs/2410.02761)
- **What's New**: 이 논문에서는 FakeShield라는 다중 모달 프레임워크를 제안하여 이미지의 진위성 평가, 변조된 영역 마스크 생성 및 픽셀 수준과 이미지 수준의 변조 단서를 기반으로 한 판단 근거 제공을 가능하게 합니다. 이는 기존의 이미지 위조 탐지 및 로컬화(IfDL) 방법이 직면한 두 가지 문제인 불투명한 탐지 원리와 다양한 변조 방식에 대한 일반화 한계를 해결합니다.

- **Technical Details**: FakeShield는 설명 가능한 IfDL(e-IFDL) 과제를 기반으로 하며, GPT-4o를 활용하여 다중 모달 변조 설명 데이터셋(MMTD-Set)을 생성합니다. DTE-FDM(영역 태그 안내 설명 가능한 위조 탐지 모듈)과 MFLM(다중 모달 위조 로컬화 모듈)을 통합하여 다양한 변조 탐지 해석을 해결하며, 상세한 텍스트 설명에 기반하여 위조 로컬화를 달성합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FakeShield는 다양한 변조 기법을 효과적으로 탐지하고 로컬라이즈하며, 기존 IfDL 방법들보다 설명 가능하고 우수한 솔루션을 제공함을 보여주었습니다. 이 모델은 copy-move, splicing, removal, DeepFake 및 AIGC 기반 편집과 같은 여러 변조 유형의 탐지와 로컬라이제이션에서 뛰어난 성능을 보였습니다.



### CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation (https://arxiv.org/abs/2410.02748)
- **What's New**: 이 논문에서는 생성형 언어 모델(LLM)에서 요약의 질을 개선하기 위해 핵심 어구(keyphrases)를 활용하는 새로운 접근법인 CriSPO를 제안합니다. CriSPO는 LLMs가 생성한 텍스트의 다각적 비판과 제안을 자동으로 생성하여, 보다 효율적인 프롬프트 변경을 지원합니다.

- **Technical Details**: CriSPO는 Critique-Suggestion-guided automatic Prompt Optimization의 줄임말로, 다양한 측면을 고려하여 텍스트 생성의 최적화를 지원합니다. CriSPO는 인-context Learning (ICL) 예시를 포함하는 템플릿을 생성하며, 다중 지표인 AlignScore와 ROUGE를 동시에 최적화할 수 있는 Automatic Suffix Tuning (AST) 기법도 도입합니다. 이 모델은 간단한 구조로 다양한 LLM에서 일관된 성능 개선을 입증하였습니다.

- **Performance Highlights**: CriSPO는 9개의 다양한 데이터셋에서 검증된 결과, ROUGE 점수를 3-4% 향상시키며 인간 평가에서도 긍정적인 결과를 보였습니다. 또한, CriSPO는 QoA(Question-Answering) 작업에서도 일관된 성능 향상을 달성하였습니다.



### AVG-LLaVA: A Multimodal Large Model with Adaptive Visual Granularity (https://arxiv.org/abs/2410.02745)
Comments:
          Preprint

- **What's New**: AVG-LLaVA(Adaptive Visual Granularity LLaVA)는 입력 이미지와 지시에 따라 적절한 visual granularity(시각적 세분성)를 선택할 수 있는 LMM(대형 다중모드 모델)입니다. 이를 통해 visual token(시각적 토큰)의 수를 줄이고 추론 속도를 향상시키며 모델 성능을 개선합니다.

- **Technical Details**: AVG-LLaVA는 시각적 세분성 스케일러와 시각적 세분성 라우터라는 두 가지 모듈을 포함하고 있습니다. 시각적 세분성 스케일러는 visual tokens에 대해 여러 번의 pooling layer(풀링 레이어)를 진행하여 서로 다른 세분성의 visual tokens를 얻습니다. 시각적 세분성 라우터는 입력된 다중 세분성 visual feature와 text feature를 기반으로 적절한 visual granularity를 선택합니다. RGLF(Ranking Granularity to align LMM Feedback)라는 새로운 훈련 패러다임도 도입되어 라우터의 예측을 LMM의 선호와 일치시키도록 돕습니다.

- **Performance Highlights**: AVG-LLaVA는 11개의 벤치마크에서 우수한 성능을 달성하였으며, visual token 수를 85.3% 줄이고 추론 속도를 2.53배 증가시키는 등 성능을 크게 개선했습니다.



### Neutral residues: revisiting adapters for model extension (https://arxiv.org/abs/2410.02744)
- **What's New**: 이 논문은 Pretrained (사전 훈련된) 대형 언어 모델을 새로운 도메인으로 확장하는 문제를 다룹니다. 기존의 방법인 fine-tuning (파인튜닝)이나 low-rank adaptation (저순위 보정) 방식은 새로운 언어를 추가하는 데 한계가 있어, 모델의 원래 도메인 성능이 저하됩니다. 저자들은 새로운 아키텍처와 훈련 절차를 결합하여 원래 도메인에서의 성능 저하 없이 새로운 언어를 학습할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 adapters (어댑터)를 개선하고, 새로운 residual blocks (잔여 블록)을 수정하여 새로운 언어를 학습하면서도 원래 도메인에서의 출력값이 거의 변하지 않도록 합니다. 이 시스템은 mixture of experts (전문가 혼합)로부터 아키텍처 요소를 차용하며, 추가적인 learnable weights (학습 가능한 가중치)를 고작 20%만 사용해도 상당히 향상된 결과를 얻습니다. 또한, 모델의 효율성을 높이는 두 가지 손실 함수를 제안합니다: domain classifier (도메인 분류기)와 sparsity loss (희소성 손실).

- **Performance Highlights**: 논문에서 제안한 접근법은 새로운 언어를 학습하는 동시에 원래의 지식을 잊지 않으며, 파인튜닝이나 기존의 어댑터 방식보다 탁월한 성능을 보여줍니다. 실험을 통해 세 가지 모델에서 두 개의 언어를 추가하는 성능 향상을 검증하였습니다.



### Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization (https://arxiv.org/abs/2410.02741)
Comments:
          Accepted to EMNLP 2024 Industry Track

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 요약 성능을 향상시키기 위해 소스 문서에서 추출한 중요한 정보(즉, keyphrases)를 활용한 새로운 접근 방식을 제시합니다. 특히, SigExt라는 경량의 키프레이즈 신호 추출기 모델을 통해 요약 프롬프트를 개선하여 ROUGE 성능 지표를 향상시키는 방법을 연구했습니다.

- **Technical Details**: SigExt 모델은 소스 문서에서 중요한 구문을 추출하고, 이러한 구문을 프롬프트에 포함시켜 LLM이 보다 완전한 요약을 생성하도록 유도합니다. 이 모델은 LLM과 독립적으로 작동하며, 다양한 LLM에 대해 성능을 개선할 수 있도록 설계되었습니다. 또한 keyphrases의 개수를 조정하여 정밀도와 재현율 간의 균형을 조절할 수 있습니다.

- **Performance Highlights**: SigExt를 사용한 결과, 4개의 대표적인 요약 데이터 세트와 3개의 최신 LLM(Claude, Mistral, Falcon)에서 일관된 ROUGE 향상을 달성했습니다. 추가적인 keyphrases를 포함하는 것이 요약의 완전도를 높이는 데 효과적이라는 것을 확인했으며, 구문 수준의 정보가 단어 수준 또는 문장 수준보다 더욱 효과적임을 보여주었습니다.



### Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models (https://arxiv.org/abs/2410.02740)
Comments:
          CV/ML

- **What's New**: 최근 인공지능 분야에서 멀티모달 모델(multimodal models)의 발전이 캡션(captions) 재작성을 통한 성능 향상의 중요성을 강조하고 있습니다. 본 연구에서는 다양한 멀티모달 모델에 맞춘 캡션 생성 파이프라인을 제안하며, 이는 합성 캡션과 원본 AltText(대체 텍스트)의 상호작용을 탐구합니다.

- **Technical Details**: 우리의 연구에서는 Short Synthetic Captions (SSC) 및 Descriptive Synthetic Captions (DSC+)를 사례로 활용하여 합성 캡션이 다양한 멀티모달 모델(CLIP, 멀티모달 LLMs)과의 상호작용을 가지고 어떻게 작용하는지를 분석합니다. 우리는 고급 캡션 포맷의 필요성을 강조하며, 원본 AltText와 합성 캡션의 혼합 접근 방식이 더 좋은 성능을 발휘하는 것을 발견했습니다.

- **Performance Highlights**: 연구 결과, 합성 캡션만을 사용할 때보다 합성 캡션과 AltText를 혼합하여 사용할 경우 성능이 향상되며, 각 모델이 특정 캡션 포맷에 대한 선호가 있음을 보여줍니다. 이러한 통합적 접근 방식은 멀티모달 모델의 전처리(pre-training) 과정에서 유용한 최적화 전략으로 기여할 수 있습니다.



### Justice or Prejudice? Quantifying Biases in LLM-as-a-Judg (https://arxiv.org/abs/2410.02736)
- **What's New**: 이번 논문에서는 LLM(as-a-Judge) 방식의 평가 방법이 여러 벤치마크에서 널리 사용되고 있음에도 불구하고 그 신뢰성 및 효용성에 영향을 미치는 잠재적인 편향(bias) 문제들을 탐구하지 않았음을 지적합니다. 연구팀은 12개의 주요 편향 유형을 식별하고, 이를 체계적으로 정량화하고 분석하는 새로운 자동화된 편향 정량화 프레임워크인 CALM을 제안합니다.

- **Technical Details**: CALM 프레임워크는 LLM(as-a-Judge)에 존재하는 12개의 편향 유형을 포함하고 있습니다. 이 프레임워크는 평가 과제에 대한 다양한 데이터셋을 통합하며, 페어 방식 비교 또는 점수 평가를 위한 특화된 메트릭스를 사용하여 편향을 정량화하는 시스템적인 접근법을 제공합니다. CALM은 LLM의 판단에 대해 공격-탐지 접근 방식을 사용하여 편향을 탐지합니다.

- **Performance Highlights**: 일부 고급 LLM 모델들이 전체적으로 좋은 성과를 보였지만, 특정 과제에서는 여전히 심각한 편향이 남아 있습니다. CALM 프레임워크를 활용한 평가에서, LLM 모델들이 공정성을 보여주기도 했으나, 여전히 다양한 편향의 영역에서 향상될 여지가 있음이 발견되었습니다.



### Custom Non-Linear Model Predictive Control for Obstacle Avoidance in Indoor and Outdoor Environments (https://arxiv.org/abs/2410.02732)
Comments:
          This manuscript has 7 pages and 8 figures, detailing NMPC for UAV obstacle avoidance using DJI UAVs. It features simulations, experimental results, and uses CasADi for optimization with ROS integration. Code and media at this https URL

- **What's New**: 본 논문은 DJI Matrice 100 드론을 위한 비선형 모델 예측 제어(Non-linear Model Predictive Control, NMPC) 프레임워크를 제안합니다. 이 프레임워크는 동적 모델과 B-스플라인 보간을 사용하여 매끄러운 경로를 생성하며, 최소한의 이탈로 안전 제약을 준수하는 방식을 채택하고 있습니다.

- **Technical Details**: NMPC는 드론의 비선형성을 유지하면서 장애물 회피와 B-스플라인 경로 추적을 수행합니다. 시스템은 Python CasADi를 활용하여 최적화되며, 로봇 운영 시스템(Robot Operating System, ROS)과 통합되어 실제 및 시뮬레이션 테스트를 위한 오픈 소스 코드 저장소를 제공합니다.

- **Performance Highlights**: 실험 결과, NMPC는 외부 교란에 잘 적응하며 매끄럽고 충돌 없는 비행 경로를 생성함으로써 안정적인 비행 성능을 나타냈습니다. 이러한 성능은 특히 장애물 밀집 환경에서도 드론의 로버스트한 작동을 가능하게 했습니다.



### Unified Multi-Modal Interleaved Document Representation for Information Retrieva (https://arxiv.org/abs/2410.02729)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 기존의 Information Retrieval (IR) 방법론이 텍스트 정보에만 의존하는 한계를 극복하기 위해, 텍스트, 이미지, 테이블 등 다양한 모달리티를 포괄하는 문서의 통합 표현을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 개발된 메소드는 최근의 Vision-Language Models (VLMs)를 활용하여, 서로 다른 모달리티가 통합된 단일 형식으로 문서를 처리하고 표현합니다. 이를 통해 문서의 전체 맥락과 다양한 부분 간의 관계를 유지할 수 있으며, 중장 문서의 경우에는 섹션들을 병합하여 단일 문서 표현으로 생성하는 전략을 채택합니다.

- **Performance Highlights**: 제안된 IDentIfy 시스템은 텍스트 전용 및 다중 모달 쿼리를 포함한 다양한 정보 검색 시나리오에서 실험적으로 검증되었으며, 기존의 단일 모달리티를 고려한 방법론들에 비해 월등히 우수한 성능을 보였습니다. 전체 문서의 단일 표현을 통한 검색 효과와 고급화된 섹션 재순위 전략인 reranking의 도입이 성과를 극대화했습니다.



### Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation (https://arxiv.org/abs/2410.02725)
- **What's New**: 본 연구에서는 Best-of-N 샘플링 방식에 대한 대안으로, 샘플 수를 적응적으로 줄이면서 성능을 유지 또는 개선할 수 있도록 설계된 새로운 generative self-evaluation 기법을 도입했습니다. 이는 외부 reward model 없이 LLM이 응답을 생성하는 중간 단계에서 더 나은 응답이 생성될 확률을 예측할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 'capability-aware self-evaluations'이라는 새로운 reward modeling 패러다임을 제안합니다. 이를 통해 LLM이 스스로 평가를 통해 얼마나 더 좋은 응답을 생성할지 예측하고, 필요에 따라 샘플을 추가 생성하거나 불필요한 샘플을 조기에 제거할 수 있게 됩니다. 이러한 self-evaluation은 단일 정의된 토큰을 생성하여 수행되며, 외부 reward model의 필요성 없이 최소한의 비용으로 이루어집니다.

- **Performance Highlights**: Llama 3.1 8B 모델은 AlpacaEval에서 GPT-4에 대한 승률이 21%에서 34%로 증가하였고, GSM8K 수학 문제에서의 성능이 84%에서 91%로 향상되었습니다. 적응형 샘플링을 통해 평균 1.2 샘플만으로도 16개의 샘플에서 관찰된 74%의 성능 향상을 달성할 수 있었습니다. 또한 초기 단계에서 성과가 부진한 샘플의 50-75%를 조기에 제거하여 성능 저하를 최소화하면서 56%의 토큰이 절약되었습니다.



### Large Language Models as Markov Chains (https://arxiv.org/abs/2410.02724)
Comments:
          49 pages, 17 figures

- **What's New**: 이 논문에서는 널리 알려진 오토 회귀 언어 모델(LLMs)과 마르코프 체인(markov chains) 간의 equivalence(동등성)을 수립하여 LLM의 추론 능력을 명확하게 설명하고자 합니다. 기존 LLM 성능의 기원을 해명하기 위한 다양한 연구가 있었으나, 본 연구는 이를 좀 더 접근 가능하게 만듭니다.

- **Technical Details**: 연구의 핵심은 크기 |Ω|의 유한 상태 공간에서 정의된 마르코프 체인으로서 LLM을 해석하는 것입니다. 저자들은 LLM의 transition matrix(전이 행렬)를 분석하고, stationary distribution(정상 분포)의 존재성과 유일성을 증명합니다. 또한 vocabularies(어휘)와 context window(맥락 창) 크기 및 model temperature(모델 온도)에 따른 수렴 속도를 제시합니다.

- **Performance Highlights**: 실험 결과, 2023-2024에 발표된 최신 LLM들이 이론적 결과에 의해 예측된 in-context scaling laws를 준수하는 것을 보였습니다. 특히, LLM들은 최소 변별 최적 빈도주의 접근법보다 마르코프 체인 학습에서 더 뛰어난 성능을 발휘하는 것으로 나타났습니다.



### Domain-Specific Retrieval-Augmented Generation Using Vector Stores, Knowledge Graphs, and Tensor Factorization (https://arxiv.org/abs/2410.02721)
Comments:
          9 pages 7 figures, 1 table, 1 cypher code Accepted to ICMLA 2024

- **What's New**: 새로운 연구에서는 SMART-SLIC라는 highly domain-specific LLM 프레임워크를 소개합니다. 이 프레임워크는 Retrieval-Augmented Generation (RAG)와 Knowledge Graph (KG), 벡터 스토어 (VS)를 통합하여 높은 정확도의 질문 응답 시스템을 구축합니다.

- **Technical Details**: SMART-SLIC는 데이터 마이닝, 자연어 처리 (NLP), 비음수 텐서 분해(nonnegative tensor factorization) 및 자동 모델 선택을 통해 LLM의 환상(hallucinations) 문제를 피할 수 있는 도메인 특화 KG와 VS를 제작합니다. 또한, 이 구조는 구조화된 정보와 비구조화된 정보를 모두 활용하여 효과적인 대화형 봇을 개발합니다.

- **Performance Highlights**: SMART-SLIC는 malware 분석과 이상 탐지를 주제로 한 과학 논문 데이터베이스에서 LLM의 질문 응답 능력을 성공적으로 시연합니다. 이 시스템은 정보 원천의 출처를 명확히 하고, 환상을 완화하며, 미세 조정(fine-tuning)의 필요성을 줄여 줍니다.



### Curvature Diversity-Driven Deformation and Domain Alignment for Point Cloud (https://arxiv.org/abs/2410.02720)
- **What's New**: 새로운 접근법으로 제안된 CDND( Curvature Diversity-Driven Nuclear-Norm Wasserstein Domain Alignment)는 소스 도메인과 타겟 도메인 간의 도메인 간극을 효과적으로 줄이는 것을 목표로 하고 있습니다.

- **Technical Details**: CDND는 두 가지 주요 구성 요소를 가지며, 첫 번째는 Curvature Diversity 기반의 Deformation Reconstruction 방법인 CurvRec이고, 두 번째는 Deformation 기반의 Nuclear-norm Wasserstein Discrepancy(D-NWD)입니다. CurvRec는 포인트 클라우드의 의미론적으로 풍부한 영역에서 두드러진 특징을 추출하도록 모델을 유도함으로써 소스와 타겟 도메인 간의 차이를 줄입니다. D-NWD는 원본 및 변형된 데이터 샘플 양쪽에서 Nuclear-norm Wasserstein Discrepancy를 적용하여 도메인을 정렬합니다.

- **Performance Highlights**: 실험 결과, CDND는 두 개의 공공 도메인 적응 데이터셋에서 포인트 클라우드 분류 및 분할 작업을 수행하기 위한 실험을 통해 기존 방법들보다 상당히 우수한 성능을 보였습니다.



### Measurements with Noise: Bayesian Optimization for Co-optimizing Noise and Property Discovery in Automated Experiments (https://arxiv.org/abs/2410.02717)
Comments:
          22 pages, 9 figures

- **What's New**: 본 논문에서는 자동화된 실험 사이클에 대하여 intra-step noise optimization을 통합한 Bayesian optimization (BO) 워크플로우를 개발했습니다. 이를 통해 실험의 데이터 품질과 비용에 대한 측정 잡음의 영향을 고려하는 새로운 프레임워크를 제시합니다.

- **Technical Details**: 제안된 프레임워크는 시간(time)을 추가 입력 파라미터로 도입하여 신호 대 잡음 비율(signal-to-noise ratio)과 실험 지속시간을 균형 있게 최적화합니다. 두 가지 접근 방식인 보상 기반(noise optimization) 최적화와 이중 최적화(double-optimization) 획득 함수(acquisition function)를 탐구하여 자동화된 워크플로우의 효율성을 개선합니다.

- **Performance Highlights**: 시뮬레이션과 실제 실험(Piezoresponse Force Microscopy (PFM) 활용)을 통해 측정 기간과 속성 탐색 최적화의 성공 사례를 입증하였으며, 다양한 변수를 최적화하는 확장 가능한(solution) 방법을 제공합니다. 이를 통해 데이터 품질을 향상시키고, 자원 소비를 줄일 수 있습니다.



### SteerDiff: Steering towards Safe Text-to-Image Diffusion Models (https://arxiv.org/abs/2410.02710)
- **What's New**: 이 논문에서는 SteerDiff라는 경량 어댑터 모듈을 소개하며, 이는 사용자 입력과 diffusion 모델 간의 중개자로 작동하여 생성된 이미지가 윤리적 및 안전 기준을 준수하도록 보장합니다.

- **Technical Details**: SteerDiff는 텍스트 임베딩 공간 내에서 부적절한 개념을 식별하고 조작하여 모델이 유해한 출력을 생성하는 것을 방지합니다. 이 모듈은 두 단계의 경량 어댑터 모델로, 텍스트 프롬프트 임베딩을 유도하는 데 중점을 두고, 안전한 콘텐츠와 안전하지 않은 콘텐츠를 최대한 구분하는 의미 경계를 구성합니다.

- **Performance Highlights**: 실험 결과 SteerDiff는 이미지 품질과 의미적 충실도를 유지하면서 부적절한 콘텐츠 생성을 현저히 줄이는 것으로 나타났습니다. 또한 P4D 및 SneakyPrompt와 같은 레드 팀 공격에 대한 방어에서 효과적인 성능을 보여주었습니다.



### LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations (https://arxiv.org/abs/2410.02707)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 내부 표현이 진실성과 관련된 정보를 훨씬 더 많이 포함하고 있다는 사실을 처음으로 밝혀냈습니다. 이는 오류 감지 성능을 크게 향상시킬 수 있는 중요한 발견입니다.

- **Technical Details**: 연구진은 LLM의 특정 토큰에서 진실성 정보를 식별하여, 오류 감지에서의 성능을 향상시키는 새로운 접근 방식을 제시했습니다. 이들은 ‘probing classifiers’를 통해 모델의 내부 표현을 분석하여 진실성과 관련된 오류 유형을 예측할 수 있음을 보여주었습니다.

- **Performance Highlights**: 이 연구는 LLM의 내부 표현과 외부 행동 간의 불일치를 드러내며, LLM이 정확한 답변을 인코딩하지만 잘못된 답변을 생성하는 경우를 관찰했습니다. 또한, 고유한 오류 패턴을 예측할 수 있는 가능성을 제시하여 오류 완화 전략의 개발에 기여할 수 있습니다.



### Selective Attention Improves Transformer (https://arxiv.org/abs/2410.02703)
- **What's New**: Selective Attention (선택적 주의)는 표준 attention 메커니즘의 단순한 파라미터 없는 개선 방법으로, 불필요한 요소에 대한 주의를 줄여 성능을 향상시킵니다. 이 방법은 다양한 모델 크기와 컨텍스트 길이에서 언어 모델링 성능을 개선하며, 기존 transformer와 동등한 성능을 보이면서 약 2배의 머리 수 및 파라미터를 가진 변형 모델과 경쟁할 수 있습니다.

- **Technical Details**: 선택적 주의는 기본적인 attention 메커니즘을 기반으로 하여, 각 토큰이 필요 없는 다른 토큰에 대한 주의를 줄이는 기능을 추가합니다. 이를 통해 주의의 컨텍스트 버퍼 크기를 줄일 수 있으며, 메모리 및 계산 요구 사항을 의미 있게 감소시킵니다. 예를 들어, 100M 파라미터를 가진 transformer가 C4에서 학습된 경우, 선택적 주의가 있을 때 512, 1,024 및 2,048의 컨텍스트 크기로 각각 16배, 25배, 47배 더 적은 메모리를 필요로 합니다.

- **Performance Highlights**: 기존의 transformer와 비교하여, 선택적 주의가 적용된 모델은 더 적은 메모리와 계산 비용으로 더 나은 성능을 발휘합니다. 특히, 선택적 주의가 있는 transformer는 복잡한 자연어 모델링 및 변수 할당 문제에서 우수한 성능을 발휘하며, 임시결과 저장이 필요한 패리티 작업과 같은 단순 작업에서도 모든 중간 결과를 마스킹하는 방식을 통해 적은 자원으로 높은 효율을 보여줍니다.



### HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly (https://arxiv.org/abs/2410.02694)
Comments:
          Code and data are available here: this https URL

- **What's New**: 이번 연구에서는 HELMET (How to Evaluate Long-context Models Effectively and Thoroughly)를 제안하여, 기존의 벤치마크가 가진 단점을 보완하고 다양한 응용 분야에 초점을 맞춘 포괄적인 테스트를 제공합니다.

- **Technical Details**: HELMET는 128k 토큰까지 조절 가능한 입력 길이를 지원하며, 모델 기반 평가를 통해 보다 신뢰할 수 있는 메트릭을 제공합니다. 또한, Few-shot prompting을 통해 기본 모델의 평가 안정성을 강화합니다.

- **Performance Highlights**: HELMET은 51개의 LCLM 모델에 대한 포괄적인 연구를 바탕으로, 합성 작업이 하위 성능 예측에 적합하지 않으며, 서로 다른 카테고리 간 상관관계가 낮다는 것을 발견했습니다. 오픈소스 모델은 긴 맥락이나 복잡한 지시를 요구하는 태스크에서 폐쇄형 모델에 비해 성능이 떨어지는 경향이 있습니다.



### Discovering Clues of Spoofed LM Watermarks (https://arxiv.org/abs/2410.02693)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)이 생성한 텍스트의 소유권을 속성화 할 수 있는 새로운 방법인 LLM 워터마크(watermark)가 신뢰성을 위협받는 스푸핑 공격(spoofing attack) 문제를 다루고 있습니다.

- **Technical Details**: 우리는 스푸핑 방법 간의 차이를 실질적으로 분석하여, 현재 스푸핑 방법이 생성하는 텍스트에서 관찰 가능한 아티팩트(artifact)를 발견했습니다. 이러한 아티팩트는 워터마크 위조(watermark forgery)의 지표로 작용하며, 저희는 이러한 아티팩트를 감지할 수 있는 엄격한 통계 테스트(statistical tests)를 제안합니다.

- **Performance Highlights**: 실험 결과, 모든 현재 스푸핑 방법에 대해 높은 테스트 파워(test power)를 달성했으며, 이는 스푸핑 공격의 근본적인 한계를 이해하고 이를 완화할 수 있는 방법을 제시합니다.



### User-centric Immersive Communications in 6G: A Data-oriented Approach via Digital Twin (https://arxiv.org/abs/2410.02688)
- **What's New**: 본 논문은 6G에서 개별 사용자의 행위 불확실성을 다루고, 다감각 경험의 품질에 대한 독특한 요구를 만족시키기 위한 사용자 중심의 서비스 제공 방식을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 데이터 중심(data-oriented) 방식으로, 사용자 요구에 맞는 네트워크 모델링을 지원하는 개인화된 데이터 관리가 특징입니다. 이를 위해 디지털 트윈(digital twin) 기술을 핵심으로 활용하여 각 사용자의 데이터 속성을 맞춤화합니다.

- **Performance Highlights**: 추적 기반 사례 연구가 사용자 중심의 몰입형 통신(user-centric immersive communications, UCIC) 달성에 대한 접근 방식의 효과를 보여줍니다.



### DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Lif (https://arxiv.org/abs/2410.02683)
Comments:
          Preprint. Under Review

- **What's New**: 이번 연구는 일상에서 마주치는 도덕적 딜레마를 다룬 데이터셋 DailyDilemmas를 소개하고, LLMs(대형 언어 모델)가 어떻게 다양한 인간 가치를 우선시하는지 분석합니다. 1,360개의 도덕적 딜레마를 통해 LLM의 결정 과정에서 개인의 윤리적 기준과 가치관의 중요성을 강조합니다.

- **Technical Details**: DailyDilemmas 데이터셋은 인간의 가치가 어떻게 관련될 수 있는지를 보여주기 위해 다양한 일상적인 주제에서 생성된 1,360개의 도덕적 딜레마로 구성됩니다. 이 연구는 World Value Survey, Moral Foundation Theory, Maslow's Hierarchy of Needs, Aristotle's Virtues, Plutchik Wheel of Emotion 등 다섯 가지 이론을 통해 가치 분석을 진행합니다.

- **Performance Highlights**: 연구 결과, LLM들은 World Value Survey에서 생존 가치보다 자기 표현 가치를, Moral Foundation Theory에서 충성도보다 배려 가치를 더 중시하는 경향을 보였습니다. 일부 기본 가치에 대한 모델 간 선호도 차이가 두드러졌으며, 예를 들어 Mixtral-8x7B 모델은 진실성을 9.7% 무시하는 반면, GPT-4-turbo 모델은 9.4% 선택하는 경향이 있었습니다. OpenAI와 Anthropic의 모델 훈련 원칙과 실제 성능 간의 차이를 평가한 결과, 두 모델 모두 원칙과 가치 선호 간의 불일치를 보였습니다.



### Distilling an End-to-End Voice Assistant Without Instruction Training Data (https://arxiv.org/abs/2410.02678)
- **What's New**: 이 새로운 연구는 Audio와 Text를 별도로 모델링하는 기존의 음성 비서 방식의 한계를 극복하기 위해, Self-Supervised Learning에 기반한 Speech Large Language Models (LLMs) 훈련의 새로운 패러다임을 제안합니다. 특히, 주석이 없는 학습을 통해 음성이 아닌 텍스트 LLM의 반응을 활용하여 DiVA(Examined Voice Assistant) 모델을 발전시켰습니다.

- **Technical Details**: DiVA는 'Automatic Speech Recognition (ASR)' 데이터만을 활용하였음에도 불구하고 다양한 작업(Spoken Question Answering, Classification, Translation)에 일반화된 성능을 보입니다. 모델은 기존의 LLM과의 교차 모달(context distillation) 방식으로 Self-Supervised Learning을 통해 훈련됩니다. 기존 데이터에 대한 의존도를 줄이면서도 뛰어난 일반화 능력을 가지고 있습니다.

- **Performance Highlights**: DiVA는 클릭한 모델 Qwen 2 Audio보다 72%의 우선율로 사용자 선호도를 충족시키며, 훈련에 사용된 컴퓨팅 자원은 100배 이상 적습니다. 이러한 성과는 Speech LLM 개선에 있어 효율성을 극대화하였으며, 새로운 데이터 수집 없이도 우수한 성능을 발휘하는 가능성을 보여줍니다.



### CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs (https://arxiv.org/abs/2410.02677)
Comments:
          Preprint. Under review

- **What's New**: 이번 연구에서는 거대한 언어 모델(LLMs)의 문화적 지식을 평가하기 위한 새로운 기준인 CulturalBench를 소개합니다. 이 기준은 1,227개의 인간 작성 및 검증된 질문으로 구성되어 있으며, 45개의 글로벌 지역을 아우릅니다.

- **Technical Details**: CulturalBench는 질문을 두 가지 설정으로 평가합니다: CulturalBench-Easy와 CulturalBench-Hard. Easy 설정은 4개의 선택형 질문으로 구성되어 있으며, Hard 설정은 각 질문을 True/False 형식으로 변환합니다. 질문은 5명의 독립적인 주석자에 의해 검증되었습니다.

- **Performance Highlights**: 모델의 성능은 CulturalBench-Hard에서 가장 잘 수행하는 모델이 61.5%의 정확도를 보였으며, 최악의 경우는 21.4%였습니다. 반면, 인간의 성능은 92.6%로 나타났습니다. 모델들은 다중 정답이 가능한 질문에서 종종 하나의 정답으로 수렴하는 경향을 보였습니다.



### FAN: Fourier Analysis Networks (https://arxiv.org/abs/2410.02675)
- **What's New**: 본 논문에서는 기존의 신경망 아키텍처가 주기성을 효과적으로 모델링하고 추론하는 데 한계를 보인다는 문제를 제기하며, Fourier Analysis에 기반한 새로운 네트워크 아키텍처인 FAN(Fourier Analysis Network)을 제안합니다.

- **Technical Details**: FAN은 Fourier Series를 활용하여 주기적 데이터를 자연스럽게 통합하는 구조를 갖추고 있습니다. 이 네트워크는 MLP와 같은 기존 모델을 대체할 수 있으며, 훨씬 적은 파라미터와 FLOPs를 요구합니다. 이론적으로 주기성이 명시적으로 모델링 되도록 설계되어 주기적 패턴의 표현과 예측에서 더 높은 정확성을 제공합니다.

- **Performance Highlights**: FAN은 기존 MLP, KAN, Transformer와 비교하여 기본 및 복잡한 주기 함수를 더 효과적으로 모델링할 수 있으며, 실제 작업에서도 상징적 공식 표현, 시계열 예측, 언어 모델링 등 다양한 분야에서 우수한 성능을 입증하였습니다.



### Unsupervised Point Cloud Completion through Unbalanced Optimal Transpor (https://arxiv.org/abs/2410.02671)
Comments:
          20 pages, 10 figures

- **What's New**: 이 논문에서는 Unbalanced Optimal Transport Map을 기반으로 하는 새로운 비쌍 (unpaired) 포인트 클라우드 완성 모델인 UOT-UPC를 제안합니다. 이는 비쌍 포인트 클라우드 완성 문제를 최적 운송 (Optimal Transport) 문제로 해석하고, 클래스 불균형 문제를 해결하는 데 초점을 맞춥니다.

- **Technical Details**: UOT-UPC는 주어진 불완전한 포인트 클라우드를 기반으로 하는 최적 운송 맵 (Optimal Transport Map)을 학습하여 적용되며, InfoCD 비용 함수 (cost function)가 이 작업에 가장 적합한 것으로 분석됩니다. 이 모델은 서로 다른 샘플링에 의한 불완전한 포인트 클라우드와 완전한 포인트 클라우드를 이용해 훈련됩니다.

- **Performance Highlights**: UOT-UPC는 비쌍 포인트 클라우드 완성에서 뛰어난 성능을 보이며, 특히 클래스 불균형이 존재하는 상황에서 강력한 성능을 발휘합니다. 단일 카테고리와 다중 카테고리 데이터셋 모두에서 경쟁력 있는 결과를 달성했습니다.



### AlphaIntegrator: Transformer Action Search for Symbolic Integration Proofs (https://arxiv.org/abs/2410.02666)
- **What's New**: 이 논문에서는 단계별 수학적 적분을 위한 최초의 올바른 학습 기반 시스템을 제시합니다. 이 시스템은 GPT transformer 모델로 표현된 정책을 학습하여, 기호 해결기(symbolic solver)가 적용할 적절한 적분 규칙을 찾는 과정을 안내합니다.

- **Technical Details**: 기호 엔진은 수학 표현에 대한 공리적으로 올바른 작업을 통해 상호작용하며, 단계별 적분을 위한 첫 번째 데이터셋도 소개됩니다. 또한, 연구진은 해당 기호 엔진과 상호작용하는 능력을 갖춘 GPT 스타일의 transformer 모델을 훈련시켰습니다.

- **Performance Highlights**: GPT transformer 모델은 가짜 데이터에서 훈련되었으며, 50% 적은 탐색 스텝을 사용해 자신의 데이터 생성기를 능가하는 높은 정확성과 효율성을 보여주었습니다. 기존의 LLM과의 비교 실험에서도 해당 모델이 수학적 작업을 수행하는 데 있어 중요한 발전을 나타내었습니다.



### Scalable Simulation-free Entropic Unbalanced Optimal Transpor (https://arxiv.org/abs/2410.02656)
Comments:
          26 pages

- **What's New**: 본 논문에서는 Entropic Unbalanced Optimal Transport (EUOT) 문제를 해결하기 위한 시뮬레이션이 필요 없는 접근 방식을 소개합니다. 이를 통해 분포 간의 최적 운송 맵을 찾는 효율성 및 확장성을 크게 개선했습니다.

- **Technical Details**: EUOT 문제는 f-divergence 최소화를 통해 목표 분포와의 정확한 매칭을 완화하여 소프트 매칭을 가능하게 합니다. 또한, 이 논문에서는 EUOT 문제의 이중 형식(dual formulation)과 최적 조건(optimality conditions)을 확률적 최적 제어(stochastic optimal control) 해석에 기반하여 도출하였습니다. 이를 통해 Simulation-free EUOT (SF-EUOT) 알고리즘을 제안하였습니다.

- **Performance Highlights**: SF-EUOT 모델은 CIFAR-10 데이터셋에서 FID 점수 3.02를 기록하였으며, 이는 기존 SB 모델의 훈련 없이도 경쟁력을 나타냅니다. 또한, 이미지 간 변환(Image-to-image translation) 기준에서도 여러 OT 모델보다 뛰어난 성능을 보여주었습니다.



### CAX: Cellular Automata Accelerated in JAX (https://arxiv.org/abs/2410.02651)
- **What's New**: 본 논문에서는 하드웨어 가속화된 셀룰러 오토마타 라이브러리가 부재한 문제를 해결하기 위해 CAX (Cellular Automata Accelerated in JAX)라는 고성능 오픈소스 라이브러리를 소개합니다. CAX는 사용자가 쉽게 다룰 수 있는 인터페이스를 제공하며 이산 및 연속 셀룰러 오토마타를 지원합니다.

- **Technical Details**: CAX는 JAX를 기반으로 하여 셀룰러 오토마타의 시뮬레이션을 대규모 병렬처리를 통해 가속화합니다. CAX는 고전 모델인 빙고 셀룰러 오토마타와 Conway의 생명 게임뿐만 아니라, Lenia와 Neural Cellular Automata와 같은 현대적 변형에 이르기까지 높은 유연성을 제공합니다. CAX는 수백만 개의 셀 업데이트를 몇 분 안에 처리할 수 있으며, 전통적인 구현 방식에 비해 최대 2,000배 빠른 성능을 보여줍니다.

- **Performance Highlights**: CAX는 클래식 모델에서부터 고급 애플리케이션까지 다양한 벤치마크를 통해 성능과 유연성을 입증하였습니다. 특히, CAX를 이용한 간단한 1차원 셀룰러 오토마타는 1D-ARC 챌린지에서 GPT-4보다 더 우수한 성능을 보였습니다. 또한, CAX는 다양한 실험을 쉽게 구현할 수 있게 해주는 모듈식 아키텍처를 제공하여, 연구자들이 복잡한 모델을 단 몇 줄의 코드로 작성할 수 있도록 합니다.



### Undesirable Memorization in Large Language Models: A Survey (https://arxiv.org/abs/2410.02650)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서의 정보 암기(memorization) 현상을 체계적으로 분석하였으며, 이로 인한 윤리적 및 법적 위험을 조명합니다. 이 논문은 LLM의 암기를 특정 모델 구조에서 어떻게 나타나는지를 분석하고, 이 문제를 해결하기 위한 전략을 제시합니다.

- **Technical Details**: LLM의 암기 현상은 모델이 훈련 데이터에서 구문이나 구절을 저장하고 재생산하는 경향을 나타냅니다. 이 논문은 암기를 다섯 가지 주요 차원인 의도성(intentionality), 정도(degree), 검색 가능성(retrievability), 추상화(abstraction), 투명성(transparency)으로 분석하고, 암기를 측정하는 방법과 그에 기여하는 요소를 탐구합니다.

- **Performance Highlights**: LLMs는 특정 작업에서 인간 수준의 성능을 초과하기도 하지만, 암기에서 발생하는 사생활 및 보안 문제로 인해 지적재산권 저촉과 같은 윤리적 문제가 발생합니다. 이 논문은 향후 연구 방향으로 LLM의 성능과 개인 정보 보호 간의 균형을 맞추고, 대화형 에이전트 및 다양한 언어 모델들에서 암기를 분석하는 방법을 제안합니다.



### Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents (https://arxiv.org/abs/2410.02644)
- **What's New**: 본 논문에서는 Agent Security Bench (ASB)라는 포괄적인 프레임워크를 소개하여 LLM 기반 에이전트의 보안 취약점을 평가하기 위한 다양한 공격 및 방어 방법을 정리하였습니다. 이 프레임워크는 10개의 시나리오와 10개의 에이전트를 포함하며, 400개 이상의 도구와 23가지 유형의 공격/방어 전략을 평가합니다.

- **Technical Details**: ASB는 LLM 기반 에이전트의 여러 작동 단계에서 발생할 수 있는 주요 공격 및 방어 방법을 포함하여, Direct Prompt Injections (DPI), Observation Prompt Injections (OPI), Memory Poisoning, Plan-of-Thought (PoT) Backdoor Attacks 등을 평가합니다. 이 프레임워크는 10개의 다양한 시나리오에서 LLM 에이전트의 보안을 다각도로 분석합니다.

- **Performance Highlights**: ASB를 기반으로 한 벤치마크 결과, LLM 기반 에이전트의 공격 성공률이 평균 84.30%에 달하는 것으로 나타났으며, 현재의 방어책은 제한적인 효과를 보여주며, LLM 에이전트의 보안 강화를 위한 추가적인 연구가 필요하다는 것을 강조합니다.



### Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization (https://arxiv.org/abs/2410.02628)
- **What's New**: 이 논문에서는 제한된 쌍 데이터와 추가적인 비짝 데이터(unpaired data)를 활용하는 새로운 세미-수퍼바이즈드(semi-supervised) 학습 패러다임을 제안합니다. 이 방법은 데이터 우도 최대화(data likelihood maximization) 기술을 사용하여 쌍 데이터와 비짝 데이터를 통합적으로 활용할 수 있도록 돕습니다.

- **Technical Details**: 우리는 새로운 손실 함수(loss function)를 통해 조건부 분포 π∗(y|x)를 학습하는 방법을 제안합니다. 이 손실 함수는 paired data와 unpaired data를 동시에 고려하며,.inverse entropic optimal transport(OT)와도 흥미롭게 연결됩니다. 이를 통해 계산적 OT에 대한 최근의 발전을 적용하여 간단한 학습 알고리즘을 구축할 수 있습니다.

- **Performance Highlights**: 경험적 테스트를 통해 우리의 방법이 paired와 unpaired 데이터를 동시에 학습하면서 조건부 분포를 효과적으로 학습함을 보여줍니다.



### NL-Eye: Abductive NLI for Images (https://arxiv.org/abs/2410.02613)
- **What's New**: NL-Eye라는 새로운 벤치마크를 소개하여 VLM(Visual Language Model)의 시각적 외적 추론(abductive reasoning) 능력을 평가하고자 함.

- **Technical Details**: NL-Eye는 350개의 트리플릿 예제를 포함하며, 각 사례는 전제 이미지와 가설 이미지들로 구성됨. VLM은 주어진 전제 이미지에 따라 가설 이미지의 그럴듯함(plausibility)을 평가해야 하며 이는 물리적, 논리적, 감정적, 기능적, 문화적, 사회적 범주로 나뉨. 이 과정에서 생성된 이미지는 텍스트에서 이미지로 변환하는 모델을 통해 얻어짐.

- **Performance Highlights**: NL-Eye에서 VLM은 무작위 기준 수준의 성과를 보이며, 인간들은 85%의 정확도로 더 그럴듯한 가설을 선택하고 94%에서 유효한 설명을 제공. 하지만 VLM은 정확한 설명을 제공하는 데 있어 50% 이상 실패하며, 이는 현대 VLM의 외적 추론 능력에 큰 결함을 나타냄.



### IndicSentEval: How Effectively do Multilingual Transformer Models encode Linguistic Properties for Indic Languages? (https://arxiv.org/abs/2410.02611)
Comments:
          23 pages, 11 figures

- **What's New**: 본 논문에서는 6개의 인디언 언어에 대해 9개의 다국어 Transformer 모델을 사용하여 언어적 속성의 인코딩 능력과 강건성을 조사하였습니다. 이는 기존 연구들이 주로 BERT와 영어에 집중한 것과는 달리, 인디언 언어에 대한 연구로 새로운 기여를 합니다.

- **Technical Details**: 이 연구에서는 6개의 인디언 언어(힌디어, 마라타어, 우르두어, 텔루구어, 칸나다어, 말라얌어)와 13가지 텍스트 섭동(perturbation)에 대한 8가지 언어적 속성(surfaces, syntactic, semantic)을 분석합니다. 이를 위해 새로운 다국어 벤치마크 데이터셋인 IndicSentEval을 소개하며, 약 47,000개의 문장으로 구성되어 있습니다.

- **Performance Highlights**: 전반적으로, 인디언 언어에 대한 언어적 속성을 잘 캡처하는 인디언 전용 모델(MuRIL 및 IndicBERT)과 달리, 보편적 모델(mBERT, InfoXLM 등)은 혼합된 결과를 보였습니다. 흥미롭게도, 보편적 모델이 인디언 전용 모델들보다 13가지 섭동에 대해 더 높은 강건성을 보이며, 특히 명사와 동사를 모두 삭제하거나 동자만 삭제하는 등의 섭동에서 두드러진 성능을 보였습니다.



### Beyond Expected Returns: A Policy Gradient Algorithm for Cumulative Prospect Theoretic Reinforcement Learning (https://arxiv.org/abs/2410.02605)
Comments:
          33 pages, 19 figures

- **What's New**: 이 연구에서는 Cumulative Prospect Theory (CPT)와 Reinforcement Learning (RL)의 조합에 대한 새로운 통찰력을 제공하고, 최적 정책의 성격을 유틸리티 함수에 따라 달라지는 방식을 조사합니다.

- **Technical Details**: CPT 정책 최적화 문제에 대한 새로운 정책 그래디언트 정리를 도출하고, 이를 통해 무모델 정책 그래디언트 알고리즘을 설계합니다. 이 알고리즘은 복잡한 상태 공간에서 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 우리의 정책 그래디언트 알고리즘은 교통 관리 및 전기 관리와 같은 실제 응용 사례에서 뛰어난 성능을 보여주며, 기존의 제로스 오더 알고리즘보다 높은 차원의 MDP에서도 확장성이 더 우수함을 입증합니다.



### Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks (https://arxiv.org/abs/2410.02596)
- **What's New**: 본 논문에서는 Generative Flow Networks (GFlowNets)의 회귀 손실 (regression loss) 선택에 대한 이론적인 기초를 마련하고, 이를 통해 새로운 회귀 손실 함수인 Shifted-Cosh, Linex(1/2), Linex(1)을 제안합니다. 이러한 회귀 손실 함수는 학습 과정 중 탐색 (exploration)과 착취 (exploitation) 행동에 중요한 영향을 미치며, GFlowNetworks의 성능을 향상시키는 데 기여합니다.

- **Technical Details**: GFlowNets는 비정규화된 분포에서 샘플링하는 데 사용되는 생성 모델로, 이를 위해 Directed Acyclic Graph (DAG) 구조를 활용합니다. 본 연구에서는 다양한 회귀 손실 함수가 서로 다른 발산 척도 (divergence measures)에 해당함을 엄밀히 증명하였고, 특히 제로 포싱 (zero-forcing)과 제로 어보이딩 (zero-avoiding)이라는 두 가지 주요 특성을 탐구하였습니다.

- **Performance Highlights**: 제안한 새로운 회귀 손실 함수는 하이퍼 그리드, 비트 시퀀스 생성, 분자 생성이라는 세 가지 벤치마크에서 평가되었으며, 기존의 제곱 손실 (squared loss)과 비교하여 수렴 속도, 샘플 다양성, 품질 및 견고성 측면에서 현저한 성능 개선을 보였습니다.



### IC3M: In-Car Multimodal Multi-object Monitoring for Abnormal Status of Both Driver and Passengers (https://arxiv.org/abs/2410.02592)
Comments:
          16 pages, 17 figures

- **What's New**: 최근, 차량 내 모니터링 기술이 운전자의 초기 비정상 상태를 감지하고 교통사고를 예방하기 위한 알림을 제공하는데 매우 유망한 기술로 주목받고 있습니다. 본 논문에서는 운전사와 승객을 동시에 모니터링할 수 있는 효율적인 카메라 회전 기반 멀티모달 프레임워크인 IC3M을 소개합니다.

- **Technical Details**: IC3M은 두 가지 주요 모듈로 구성되어 있습니다: 1) Adaptive Threshold Pseudo-labeling 전략은 클래스별 분포에 따라 가짜 레이블의 임계치를 조정하여 클래스 균형을 이루는 레이블을 생성합니다. 2) Missing Modality Reconstruction 모듈은 한정된 레이블에서 학습된 크로스모달리티 관계를 활용하여 누락된 모달리티를 복원합니다.

- **Performance Highlights**: IC3M은 정확도, 정밀도 및 재현율 측면에서 최신 벤치마크를 초월하며, 레이블이 제한된 데이터와 심각한 누락 모달리티 환경에서도 뛰어난 견고성을 보여줍니다.



### Boosting Sample Efficiency and Generalization in Multi-agent Reinforcement Learning via Equivarianc (https://arxiv.org/abs/2410.02581)
Comments:
          accepted as a poster at NeurIPS 2024

- **What's New**: 이번 논문은 Exploration-enhanced Equivariant Graph Neural Networks (E2GN2)를 제안하여 Multi-Agent Reinforcement Learning (MARL)에서의 샘플 효율성 및 일반화 성능을 향상시킵니다. 특히, E2GN2는 기존의 Equivariant Graph Neural Networks (EGNNs)를 단순히 적용하는 것이 아닌, 탐색 편향 문제를 완화하는 방식으로 설계되었습니다.

- **Technical Details**: E2GN2는 다중 에이전트 환경에서의 정책 학습을 위해 설계된 구조로, EGNN의 세 가지 특성을 개선합니다: (1) 복합적인 이산/연속 액션 공간에서의 적용, (2) 초기에 발생할 수 있는 탐색 편향 완화, (3) 변환 대칭에 대한 보장을 제공합니다. E2GN2는 정책을 확률적으로 나타내기 위해 EGNN의 연속 출력을 로짓으로 매핑하는 방법을 개발했습니다.

- **Performance Highlights**: E2GN2는 MARL 벤치마크인 MPE 및 SMACv2에서 평균 10배 이상의 샘플 효율성을 보여주었으며, 일반화 테스트에서는 기존 GNNs에 비해 5배 향상된 성능을 나타냈습니다. 이러한 결과를 바탕으로 E2GN2는 복잡한 다중 에이전트 시스템에서 신뢰할 수 있는 해결책 제공을 위한 기반을 마련합니다.



### Deep Regression 2D-3D Ultrasound Registration for Liver Motion Correction in Focal Tumor Thermal Ablation (https://arxiv.org/abs/2410.02579)
Comments:
          15 pagers, 9 figures

- **What's New**: 본 논문에서는 간 종양 (liver tumor) 절제 시 바늘 적용기 (needle applicator)의 정확한 배치를 위해 2D-3D 초음파 (ultrasound) 등록 접근 방식을 제안합니다. 이는 간의 움직임으로 인한 오류를 줄이기 위해 설계되었습니다.

- **Technical Details**: 본 연구는 2D 및 3D 초음파 이미지의 특징을 상관시키고 6D 회전 표현을 사용하여 모델의 학습 안정성을 향상시키는 방법을 제안합니다. 데이터셋은 훈련(2388 쌍), 검증(196 쌍), 테스트(193 쌍) 이미지로 나누어졌습니다.

- **Performance Highlights**: 제안된 방법은 평균 유클리드 거리 오차 (mean Euclidean distance error) 2.28 mm ± 1.81 mm, 평균 지오데식 각 오차 (mean geodesic angular error) 2.99° ± 1.95°를 달성했으며, 2D-3D 초음파 이미지 쌍 당 0.22초의 실행 시간을 기록했습니다. 이는 임상 적용 가능성을 나타냅니다.



### ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration (https://arxiv.org/abs/2410.02551)
- **What's New**: ColaCare는 Large Language Models (LLMs) 기반의 다중 에이전트 협업을 통해 전자 건강 기록 (EHR) 모델링을 향상시키는 프레임워크입니다.

- **Technical Details**: ColaCare는 DoctorAgent와 MetaAgent의 두 가지 유형의 에이전트를 사용하여 환자 데이터를 공동 분석합니다. Expert models는 수치 EHR 데이터에서 예측을 처리하고 생성하며, LLM 에이전트는 협업 상담 프레임워크 내에서 추론 참조 및 의사 결정 보고서를 생성합니다. 또한 Merck Manual of Diagnosis and Therapy (MSD) 의료 지침을 검색 강화 생성 (RAG) 모듈에 통합하여 권위 있는 증거 지원을 제공합니다.

- **Performance Highlights**: 네 개의 EHR 데이터셋에 대한 광범위한 실험을 통해 ColaCare는 사망률 예측 작업에서 우수한 성능을 보였으며, 임상 의사 결정 지원 시스템을 혁신하고 개인화된 정밀 의학을 발전시킬 잠재력을 강조합니다.



### Personalized Quantum Federated Learning for Privacy Image Classification (https://arxiv.org/abs/2410.02547)
- **What's New**: 양자 연합 학습(Quantum Federated Learning, QFL)을 통하여 개인화된 모델을 개선하는 새로운 알고리즘을 제안하여, 불균형한 이미지 분포에서도 개인화된 클라이언트 모델을 유지할 수 있도록 하였습니다.

- **Technical Details**: 제안된 개인화된 양자 연합 학습(Personalized Quantum Federated Learning, PQFL) 모델은 클라이언트 모델에 개인화 레이어를 도입하여 개인화된 파라미터를 유지합니다. 알고리즘은 클라이언트와 서버 간의 데이터 상호작용을 안전하게 하면서 이미지 분류를 수행합니다. FashionMNIST 데이터셋을 활용하여 개인화된 연합 학습을 적용하였습니다.

- **Performance Highlights**: 8개의 클라이언트를 사용한 실험에서 서버의 정확도는 100%에 도달했으며, 비개인화 모델보다 7% 우수한 성능을 보였습니다. 두 개의 클라이언트를 사용할 경우 평균 클라이언트 정확도는 비개인화 모델보다 2.9% 높았습니다. PQFL 방법은 비개인화된 양자 연합 학습 방법에 비해 우수한 성능을 보여주었습니다.



### Contextual Document Embeddings (https://arxiv.org/abs/2410.02525)
- **What's New**: 이 논문에서는 문서 검색(retrieval)에서 밀집 문서 임베딩(dense document embeddings)의 한계를 극복하기 위해 문맥화된(document contextualized) 문서 임베딩을 제안합니다. 본 연구는 문서와 이웃 문서를 함께 고려하는 두 가지 보완적인 방법을 도입합니다.

- **Technical Details**: 첫 번째 방법은 이웃 문서를 명시적으로 포함하는 대조 학습(objective) 목표를 설정하는 것이며, 두 번째는 이웃 문서 정보가 포함된 새로운 아키텍처를 통해 임베딩을 인코딩하는 것입니다. 이를 통해 기존의 biencoder 방식에 비해 다양한 설정에서 성능 향상을 기록했습니다.

- **Performance Highlights**: 우리는 MTEB 벤치마크에서 업계 최고 성능을 달성했으며, 하드 네거티브(mining), 점수 증류(score distillation) 및 추가적인 훈련 없이 구현할 수 있음을 보여주었습니다. 특히 특화된 도메인(예: 금융 및 의료 문서)에서 성능 향상이 두드러졌습니다.



### SAFLEX: Self-Adaptive Augmentation via Feature Label Extrapolation (https://arxiv.org/abs/2410.02512)
Comments:
          ICLR 2024

- **What's New**: 이 논문에서는 데이터 증강(data augmentation)의 효율성을 향상시키기 위한 새로운 방법인 SAFLEX(자기 적응형 증강(Self-Adaptive Augmentation via Feature Label EXtrapolation)를 소개합니다.

- **Technical Details**: SAFLEX는 제공된 기존의 증강 파이프라인(upstream augmentation pipeline)에서 샘플 가중치(sample weights)와 소프트 라벨(soft labels)을 학습하는 효율적인 이층 최적화(bilevel optimization) 알고리즘을 사용합니다. 이 방법은 증강된 샘플의 잡음(noise)과 라벨 오류(label errors)를 줄이면서 계산 비용이 적습니다. SAFLEX는 자연 이미지, 의료 이미지 및 표형 데이터(tabular data)와 같은 다양한 데이터 세트에서 높은 성능을 보여줍니다.

- **Performance Highlights**: SAFLEX는 인기 있는 증강 기법인 RandAugment와 Mixup에 대해 최대 3.6% 향상된 성능을 보였으며 특히, 이미지 생성 모델에서 평균 1.9%의 성능 향상을 달성했습니다. 이러한 결과는 SAFLEX의 다양성과 효율성을 강조하며, 다양한 데이터 유형과 학습 작업에 걸쳐 적용 가능성을 보여줍니다.



### Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessmen (https://arxiv.org/abs/2410.02505)
Comments:
          10 pages, 5 figures. The code and models will be available at this https URL

- **What's New**: 이번 연구는 훈련이 필요 없는 Dog-IQA라는 새로운 이미지 품질 평가(IQA) 방법을 제안합니다. 이 방법은 멀티모달 대형 언어 모델(MLLM)의 사전 지식을 활용하며, 사람 전문가의 평가 방식을 모방하여 정확한 IQA 점수를 추출합니다.

- **Technical Details**: Dog-IQA는 두 가지 주요 기술을 활용하여 IQA 점수를 수집합니다. 첫째, MLLM의 행동 패턴을 활용하여 객관적인 점수를 부여하며 주관적인 요인의 영향을 최소화합니다. 둘째, 로컬(지역) 및 글로벌(전체) 정보를 활용하기 위해 이미지 전체와 지역 구문 객체를 입력으로 받고 이들의 점수를 집계합니다.

- **Performance Highlights**: Dog-IQA는 훈련이 필요 없는 방법으로서 현존하는 최고 수준의 성능(SOTA)을 달성하였으며, 훈련 기반 방법과 비교해도 경쟁력 있는 성능을 보였습니다. 다양한 데이터 세트를 통해 실험을 진행하여 이 방법의 효용성을 입증하였습니다.



### Mixed-Session Conversation with Egocentric Memory (https://arxiv.org/abs/2410.02503)
Comments:
          EMNLP Findings 2024 (30 pages); Project website: this https URL

- **What's New**: 이 논문에서는 Mixed-Session Conversation이라는 새로운 대화 시스템을 소개합니다. 이 시스템은 여러 파트너와의 대화를 포함하는 다중 세션 대화를 지원하며, MiSC라는 새로운 데이터셋을 제안합니다.

- **Technical Details**: MiSC는 각 에피소드에 6개의 세션으로 구성된 8,500개의 에피소드를 포함하며, 총 51,000개의 세션이 있습니다. 이 시스템에서 사용되는 새로운 대화 모델은 Egocentric Memory Enhanced Mixed-Session Conversation Agent (EMMA)이며, 주 화자의 관점에서 기억을 관리하는 메커니즘을 통합하여 연속적인 상호작용을 가능하게 합니다.

- **Performance Highlights**: EMMA는 MiSC를 통해 훈련되어 각 세션에서의 대화 파트너가 변경되더라도 일관성 있고 자연스러운 대화 흐름을 유지하는 것으로 확인되었습니다. 또한, 대화의 기억을 잘 유지하면서 전체 대화에서 모순 없이 높은 기억률을 유지하는 것으로 평가되었습니다.



### Meta-Models: An Architecture for Decoding LLM Behaviors Through Interpreted Embeddings and Natural Languag (https://arxiv.org/abs/2410.02472)
Comments:
          11 pages, 2 figures

- **What's New**: 본 논문에서는 'meta-model' 아키텍처를 통해 대형 언어 모델(LLM)의 내부 작동을 해석하는 새로운 접근 방식을 제안하고 있습니다. 기존의 프로빙(probing) 방법론의 한계를 보완하며, 다양한 상황에서의 모델의 행동을 해석하려는 목표가 있습니다.

- **Technical Details**: 'Meta-model'은 'input-model'의 활성화를 받아 자연어 질문에 대한 답변을 생성하는 모델입니다. 이 구조는 인지하는 데 필요한 다양한 입력 모델의 내부 동작을 해석하는 능력을 가지고 있습니다. 이를 통해 기존의 특정 과제에 국한된 해석을 넘어서 다양한 상황에 대한 해석이 가능해집니다.

- **Performance Highlights**: 시험 결과, meta-model은 훈련된 특정 작업셋 이외의 Behaviors(행동) 해석에서도 우수한 일반화 성능을 보여주었습니다. 또한, 서로 다른 모델 패밀리 간의 해석이 가능함을 입증하였으며, 특히 거짓말 탐지에 대한 성과가 두드러졌습니다.



### Response Tuning: Aligning Large Language Models without Instruction (https://arxiv.org/abs/2410.02465)
Comments:
          34 pages

- **What's New**: 본 논문은 Response Tuning (RT)이라는 새로운 접근 방식을 제안하며, 이는 instruction-conditioning 단계를 생략하고 오직 응답 공간(supervised response space)에 초점을 맞춥니다. 이를 통해 프리 트레인(pre-trained) LLMs의 잠재력을 끌어내어 유용하고 안전한 채팅 어시스턴트 역할을 할 수 있음을 보이고자 합니다.

- **Technical Details**: Response Tuning (RT)은 instruction-response 쌍을 이용한 기존의 Instruction Tuning(IT) 프로세스와 달리, 모델이 응답을 생성하고 그 분포를 학습하도록 훈련합니다. 연구에서는 최근 네 가지 LLM을 대상으로 하여 RT 모델이 다양한 지시에 효과적으로 응답할 수 있다는 것을 실험적으로 입증하였습니다.

- **Performance Highlights**: RT 모델은 기존 IT 모델들과 비교했을 때 유사한 도움을 줄 수 있으며, 사용자 선호도 또한 개선되었습니다. 특히, 훈련 응답의 분포를 제어하고, 불안전한 쿼리에 대한 거부 응답을 포함시킴으로써 사용자와의 상호작용에서 더 나은 반응을 보였습니다.



### Recurrent Few-Shot model for Document Verification (https://arxiv.org/abs/2410.02456)
- **What's New**: 이번 연구에서는 일반적인 ID 및 여행 문서의 이미지 및 비디오 기반 검증 시스템의 성과 향상에 초점을 맞추고 있습니다. 특히 보지 못한 클래스의 문서에 대해 아주 적은 수의 예로도 사기 문서를 탐지할 수 있는 순환 기반의 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 순환 많이-하나 네트워크 구조를 갖고 있으며, 문서 이미지를 개별 패치로 나누어 처리한 후 특징 벡터를 생성합니다. 이 모델은 few-shot learning (FSL) 전략을 사용하여 지원 집합과 질의 집합에서 이미지를 분류합니다. Conditional FSL과 Unconditional FSL 두 가지 전략을 통해 훈련됩니다.

- **Performance Highlights**: SIDTD 및 Findit 데이터세트에서 실시한 초기 결과는 제안된 모델이 이러한 과제에 대해 우수한 성능을 보임을 보여줍니다. 특히 새로운 클래스의 문서에 대해서도 잘 일반화되는 능력을 갖추었습니다.



### Clinnova Federated Learning Proof of Concept: Key Takeaways from a Cross-border Collaboration (https://arxiv.org/abs/2410.02443)
- **What's New**: Clinnova는 다국적 협력 프로젝트로, 정밀 의학의 힘을 해제하기 위해 데이터를 연합하고 표준화하며 상호 운용성을 추구합니다. 이 프로젝트는 유럽 표준을 만들기 위해 인공지능(AI)과 데이터 과학을 활용하여 의료 결과 및 효율성을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: Clinnova는 소아 편타염, 류마티스 질환 및 다발성 경화증(MS)을 대상으로 삼고 있으며, 데이터 품질을 중시하여 AI 알고리즘 개발에 기여합니다. 중요 기술로는 Federated Learning(FL)이 있으며, 이 방식은 전국적으로 분산된 클라이언트에서 데이터 모델을 학습할 수 있게 해 주며, 데이터 프라이버시를 보장하면서 모델 훈련을 수행할 수 있게 합니다.

- **Performance Highlights**: Clinnova-MS에서는 MS 환자 진료 향상에 초점을 두고 FL 기술을 활용하여 질병 진행 상황을 감지하는 더 정확한 모델을 개발하고 있습니다. 초국경에서의 협력을 통해 이루어진 MRI 영상의 MS 세분화를 위한 POC 프로젝트는 중요한 이정표로 자리잡고 있으며, FL의 의료 적용 가능성을 보여줍니다.



### Optimizing Adaptive Attacks against Content Watermarks for Language Models (https://arxiv.org/abs/2410.02440)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 워터마킹 기법에 대한 새로운 접근 방식을 제안합니다. 특히, 비적응형(non-adaptive) 공격자가 아닌 적응형(adaptive) 공격자에 대한 워터마킹의 강인성(robustness)을 평가하는 방법론을 소개합니다.

- **Technical Details**: LLMs의 워터마킹은 모델 생성 출력에 숨겨진 메시지를 삽입하여 비밀 워터마킹 키를 사용해 이를 감지 가능하게 만듭니다. 기존의 연구들은 비적응형 공격자에 대해서만 강인성을 테스트하였으나, 본 연구에서는 적응형 공격자에 대한 최적화된 공격을 제안합니다. 또한, 공격자는 공개된 LLM 모델을 사용하여 워터마크를 회피하는 방법을 훈련할 수 있으며, 이는 텍스트 품질에 미미한 영향을 미치면서 실현 가능합니다.

- **Performance Highlights**: 실험 결과 (i) 적응형 공격이 비적응형 기준보다 월등히 성능이 뛰어난 것으로 나타났습니다. (ii) 비적응형 설정에서도 몇몇 알려진 워터마크에 최적화된 적응형 공격이 다른 보이지 않는 워터마크에 대해 높은 효율을 보였습니다. (iii) 최적화 기반 공격은 실용적이며 7 GPU 시간 이하로 수행 가능합니다.



### Collective Critics for Creative Story Generation (https://arxiv.org/abs/2410.02428)
Comments:
          EMNLP 2024 (36 pages)

- **What's New**: 이 논문에서는 Collective Critics for Creative Story Generation (CritiCS)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 이야기 계획(Story Plan) 수립 및 이야기 생성(Story Generation) 단계로 구성되어 있으며, 이야기의 창의성과 독자 참여를 높이기 위해 집단 수정 메커니즘을 통합합니다.

- **Technical Details**: CritiCS 프레임워크는 CrPlan 및 CrText라는 두 가지 단계로 나뉘며, 각 단계에서 여러 LLM 비평가들이 드래프트를 평가하고 개선점을 제안합니다. CritiCS는 창의성을 평가하기 위한 다양한 기준을 적용하며, 비평가는 적절한 캐릭터(persona)를 부여받아 이야기를 더욱 독창적으로 만듭니다.

- **Performance Highlights**: 성공적인 평가에 따르면, CritiCS는 기존 최첨단 방법들에 비해 창의성과 흥미로움에서 큰 차이를 보이며, 사용자들과의 상호작용을 통해 이야기를 개선하는 새로운 가능성을 제시합니다.



### Learning the Latent Rules of a Game from Data: A Chess Story (https://arxiv.org/abs/2410.02426)
- **What's New**: 이번 연구는 소규모 사전 학습된 생성 언어 모델(Small Language Models, SLMs)이 체스의 규칙을 학습할 수 있음을 보여줍니다. 모델은 1,000부터 1,000,000개의 예제가 주어질 때, 체스 문제를 해결하고 정당한 수를 제안하는 등의 작업을 수행합니다.

- **Technical Details**: 28M 및 125M 파라미터를 가진 소규모 사전 학습된 생성 언어 모델(SLM)을 사용하여 체스의 규칙을 학습하는 데 필요한 데이터의 양과, 학습이 데이터와 비례하여 확장될 수 있는지를 탐구했습니다. 연구는 기본 모델이 수행할 수 없는 복잡한 작업을 정확하게 수행할 수 있도록 세밀하게 조정된 모델을 개발하는 방법을 제안합니다.

- **Performance Highlights**: 모델의 정확도는 주어진 예제의 수가 증가함에 따라 개선되며, 연속적인 언어 모델 조정 에폭(epoch)의 영향도 관찰되었습니다. 또한, 조정된 모델은 체스에서 상당한 성능을 보였음이 확인되었습니다.



### SynCo: Synthetic Hard Negatives in Contrastive Learning for Better Unsupervised Visual Representations (https://arxiv.org/abs/2410.02401)
Comments:
          10 pages, 6 figures, 4 tables. arXiv admin note: text overlap with arXiv:2010.01028 by other authors

- **What's New**: 이 논문에서는 SynCo(대조 학습에서의 합성 네거티브)를 소개하고, 이는 합성 하드 네거티브를 생성하여 학습 과정을 향상시키는 새로운 대조 학습 접근법입니다.

- **Technical Details**: SynCo는 MoCo 프레임워크를 기반으로 하며, 합성 하드 네거티브를 생성하는 여섯 가지 새로운 전략을 도입합니다: (1) Interpolated negatives; (2) Extrapolated negatives; (3) Mixup negatives; (4) Noise-injected negatives; (5) Perturbed negatives; (6) Adversarial negatives. 이 방법은 메모리 큐에서 온디맨드로 하드 네거티브를 생성하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: SynCo는 ImageNet의 선형 평가에서 200 epochs 이후 68.1%의 top-1 정확도를 기록하여 MoCo의 67.5%를 초과합니다. 또한, PASCAL VOC에서 82.5%의 AP를 달성하며, COCO 데이터셋의 바운딩 박스 검출에서 40.4% AP, 인스턴스 분할에서 35.4% AP를 기록하여 새로운 벤치마크를 세웠습니다.



### Parameter Competition Balancing for Model Merging (https://arxiv.org/abs/2410.02396)
Comments:
          Accepted by NeurIPS2024

- **What's New**: 본 논문은 PCB-Merging (Parameter Competition Balancing)이라는 새로운 기법을 소개하고 있으며, 이 기법은 여러 모델의 파라미터를 효과적으로 통합하는 경량화된 기술입니다. 기존의 모델 병합 기법이 작업 간의 잠재적 충돌 및 복잡한 상관관계를 다루는 데 어려움이 있었던 것에 반해, PCB-Merging은 파라미터 경쟁을 관리하여 더 나은 성능 향상을 도모합니다.

- **Technical Details**: PCB-Merging은 단계적으로 두 가지 균형 조정 방식을 활용합니다: intra-balancing과 inter-balancing. Intra-balancing은 개별 작업 내에서 파라미터의 중요성을 평가하고, inter-balancing은 작업 간의 파라미터 유사성을 평가합니다. 중요도가 낮은 파라미터는 제거되고, 남은 파라미터는 최종 병합 모델을 형성하기 위해 재조정됩니다. 이러한 방식으로 모델의 성능을 극대화합니다.

- **Performance Highlights**: 다양한 실험에서 PCB-Merging 기법이 기존의 모델 병합 방법들보다 우수한 성능 향상을 보여 주었으며, 교차 작업(cross-task), 교차 도메인(cross-domain), 교차 훈련 구성을 포함한 다양한 병합 시나리오에서 제출되었습니다. 특히, T5-base 모델을 사용했을 때 4.3%의 성능 향상을 달성했습니다.



### Online Multi-Label Classification under Noisy and Changing Label Distribution (https://arxiv.org/abs/2410.02394)
- **What's New**: 본 논문은 노이즈가 많고 변화하는 레이블 분포(Noisy and Changing Label Distribution, NCLD) 하에서 온라인 다중 레이블 분류(Online Multi-label Classification, OMC) 알고리즘을 제안합니다. 기존 OMC 방법들이 레이블 품질 문제와 노이즈 레이블을 처리하는 데 한계가 있음을 강조합니다.

- **Technical Details**: 제안된 알고리즘은 1) 지역 특징 그래프(local feature graph)를 활용하여 관측된 레이블과 함께 레이블 점수를 공동으로 재구성 2) 불편(unbiased) 레이블 카디널리티를 사용하여 진짜 레이블 분포의 변화를 감지하고 3) 닫힌 형태의 최적 모델 솔루션에서 파생된 업데이트 규칙을 통해 효율적이고 정확한 업데이트를 달성합니다.

- **Performance Highlights**: 경험적 실험 결과를 통해 제안한 방식이 NCLD 하에서 인스턴스를 정확하게 분류하는 데 효과적임이 검증되었습니다.



### Diffusion Meets Options: Hierarchical Generative Skill Composition for Temporally-Extended Tasks (https://arxiv.org/abs/2410.02389)
- **What's New**: 본 논문은 로봇이 복잡한 계획을 생성하고 실행 오류를 수정할 수 있는 능력을 요구하는 문제를 다룹니다. 특히, Linear Temporal Logic (LTL)로 지정된 지침을 기반으로 하여 계획을 생성하고 업데이트하는 데이터 기반 계층적 프레임워크인 DOPPLER을 제안합니다.

- **Technical Details**: DOPPLER은 비전문가 데이터셋에서 오프라인 강화 학습(offline reinforcement learning)을 통해 선택지를 계층적으로 구성하며, 저수준 행동을 위한 옵션을 생성하기 위해 diffusion 모델을 활용합니다. 또한, 배치 생성 시 결정론적(posteri) 샘플링 기법을 도입하여 속도와 다양성을 개선합니다.

- **Performance Highlights**: 실험 결과, DOPPLER은 장애물 회피와 순차 방문을 위한 경로의 시퀀스를 생성하고 LTL 명세를 충족하는 데 있어 기존의 방법들과 비교해 우수한 성능을 보였습니다. 복잡한 내비게이션 및 조작 작업에서 강력한 성공률을 달성하며, 실제 로봇을 이용한 테스트에서도 안정적인 행동을 보여 다른 방법들보다 월등한 성과를 기록했습니다.



### BiSSL: Bilevel Optimization for Self-Supervised Pre-Training and Fine-Tuning (https://arxiv.org/abs/2410.02387)
- **What's New**: 본 연구에서는 BiSSL을 소개합니다. 이는 자가 감독 학습에서 전처리(pretext) 사전 학습과 다운스트림(downstream) 미세 조정 단계 간의 정렬(alignment)을 향상시키기 위해 이단계 최적화(bilevel optimization) 기법을 도입한 최초의 트레이닝 프레임워크입니다.

- **Technical Details**: BiSSL은 전처리와 다운스트림 작업 목표를 이단계 최적화 문제로 모델링하며, 각 단계의 상호 의존성을 명시적으로 모델링합니다. 이를 통해 두 단계 간의 정보 공유가 개선되어 다운스트림 작업에 더 적합한 파라미터 초기화가 이루어질 수 있습니다. BiSSL에서는 이 두 개의 목표를 번갈아 최적화하는 훈련 알고리즘을 제안합니다.

- **Performance Highlights**: ResNet-18 백본 모델이 SimCLR로 STL10 데이터셋에서 사전 학습된 후, BiSSL 프레임워크가 기존 자가 감독 학습 파이프라인에 비해 다양한 다운스트림 이미지 분류 데이터셋에서 일관되게 향상된 분류 정확도를 달성했음을 보여주었습니다.



### MetaMetrics: Calibrating Metrics For Generation Tasks Using Human Preferences (https://arxiv.org/abs/2410.02381)
Comments:
          Preprint

- **What's New**: MetaMetrics라는 새로운 메타 메트릭(metric)을 도입하여, 다양한 작업을 평가하는 데 있어 인간의 선호도에 더 잘 맞추도록 기존 메트릭들을 조정하는 방법을 제안합니다.

- **Technical Details**: MetaMetrics는 감독된 방법으로 캘리브레이션(calibration)된 메타 메트릭으로, 언어 및 비전 관련 다운스트림 작업에서 유연성과 효과성을 보여줍니다. 이 메트릭은 참조 기반(reference-based) 및 비참조(reference-free) 두 가지 설정에서 작동하여 다양한 작업과 모달리티(modality)에 걸쳐 널리 활용될 수 있습니다.

- **Performance Highlights**: MetaMetrics는 여러 다국어 및 다영역(multi-domain) 시나리오에서 인간 선호도와 밀접하게 정렬되는 효과를 보여주며, 기존 메트릭들과 비교했을 때 성능이 월등히 우수함을 입증했습니다.



### Towards Comprehensive Detection of Chinese Harmful Memes (https://arxiv.org/abs/2410.02378)
- **What's New**: 본 논문은 NeurIPS 2024 D & B Track에서 채택되었으며, 중국 인터넷에서 유해한 밈(harmful memes)의 탐지가 저조한 이유로 신뢰할 수 있는 데이터셋(dependency dataset)과 효율적인 탐지기(dedector)가 부족하기 때문임을 강조합니다.

- **Technical Details**: ToxiCN MM이라는 첫 번째 중국 유해 밈 데이터셋을 구축하였으며, 이 데이터셋은 다양한 밈 유형에 대한 세부 주석을 포함한 12,000개의 샘플로 구성되어 있습니다. 또한 Multimodal Knowledge Enhancement (MKE)라는 기본 탐지기를 제안하며, 이는 LLM이 생성한 밈 콘텐츠의 맥락(contextual information)을 통합하여 중국 밈에 대한 이해를 향상시킵니다.

- **Performance Highlights**: 평가 단계에서 우리는 여러 기본 모델(LLMs 포함)에 대한 정량적 실험과 정성적 분석을 실시했습니다. 실험 결과, 기존 모델이 중국 유해 밈을 탐지하는 데 어려움이 있음을 보여주었고, MKE의 효과성이 입증되었습니다.



### NTU-NPU System for Voice Privacy 2024 Challeng (https://arxiv.org/abs/2410.02371)
Comments:
          System description for VPC 2024

- **What's New**: 본 논문에서는 Voice Privacy Challenge 2024에 대한 제출 내용을 설명합니다. 신규 음성 익명화 시스템을 제안하기보단, 제공된 기준들을 개선하여 요구 조건을 충족하고 평가 지표를 향상시켰습니다.

- **Technical Details**: 감정 임베딩(emotion embedding)을 구현하고, WavLM과 ECAPA2 스피커 임베더(embedders)를 사용하여 기존 B3 기준을 개선했습니다. 또한, 다양한 스피커 및 프로소디 익명화 기술을 비교하고, Mean Reversion F0 방법을 통해 개인 정보 보호를 향상시키는 기법을 소개합니다. 이외에도 적어도 두 가지 불연속 모델인 β-VAE와 NaturalSpeech3 FACodec을 탐색했습니다.

- **Performance Highlights**: B3와 B5 기준 시스템에서 실험한 결과, 감정 임베딩이 감정 인식 성능을 개선하는 데 도움을 주었다고 보고되었습니다. 특히, 메인 리버전 F0 기법은 EER을 개선시키는 데 기여하여 개인 정보 보호와 유용성 간의 균형을 이루는 데 효과적이었습니다.



### From Concrete to Abstract: A Multimodal Generative Approach to Abstract Concept Learning (https://arxiv.org/abs/2410.02365)
- **What's New**: 이 논문은 Concrete(구체적) 개념과 Abstract(추상적) 개념을 모두 이해할 수 있는 multimodal generative(다중 모달 생성) 접근 방식을 제시합니다. 이 모델은 시각 정보와 범주적 언어 정보의 통합을 통해 고차원 추상 개념 학습을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 세 가지 개념 수준(하위 수준, 기본 수준, 상위 수준)으로 구성되어 있습니다. 모델은 하위 수준 구체적 개념을 학습한 후, 이를 결합해 기본 수준 개념을 형성하고, 마지막으로 이를 바탕으로 상위 수준 추상 개념을 학습합니다. 본 연구에서는 Multimodal Mixture-of-Experts Variational Autoencoders (MMVAE) 구조를 기반으로 합니다.

- **Performance Highlights**: 모델은 언어-시각(visual-to-language) 및 시각-언어(language-to-visual) 테스트를 통해 고차원 추상 개념에 대한 학습 능력을 평가하였으며, 높은 성과를 보였습니다. 실험 결과, 모델의 언어 이해 및 명명(Language Naming) 작업에서의 능숙함이 입증되었습니다.



### A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond (https://arxiv.org/abs/2410.02362)
- **What's New**: Mamba는 의료 이미지 분석에서 템플릿 기반의 딥러닝 접근 방식을 대체할 수 있는 유망한 대안으로 주목받고 있습니다. 기존의 Transformer 아키텍처의 단점인 다차원적 계산 복잡성과 긴 범위의 의존성을 효과적으로 처리하지 못하는 문제를 해결합니다.

- **Technical Details**: Mamba는 State Space Model(SSM)의 특수한 경우로, 선형 시간 복잡성을 제공하여 긴 시퀀스를 처리할 수 있으며 메모리 소모를 줄이는 기능이 있습니다. Mamba 아키텍처는 순수 Mamba, U-Net 변형, CNN과 Transformer, Graph Neural Networks를 포함한 하이브리드 모델 등을 포함하고 있으며, 선택적 스캔 메커니즘과 하드웨어 친화적 알고리즘을 채택하여 효율적인 저장과 계산을 지원합니다.

- **Performance Highlights**: Mamba는 의료 이미지 세분화, 분류, 합성 등 다양한 작업에서 우수한 성능을 나타내며, 다중 양식 데이터 통합에 효과적입니다. 이러한 특성 덕분에 Mamba는 환자 진단 정확도와 결과를 향상시키는 데 기여할 수 있습니다.



### AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models (https://arxiv.org/abs/2410.02355)
- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)에서 'hallucination'(환각) 문제를 해결하기 위해 AlphaEdit이라는 새로운 모델 편집 방법을 제안합니다. 기존의 locate-then-edit 접근 방식에서는 지식 업데이트 시 기존 지식을 손상시키는 문제가 있었으나, AlphaEdit은 'null space'(영 공간)에서 perturbation(섭동)을 투영하여 이러한 문제를 해결합니다.

- **Technical Details**: AlphaEdit은 모델 파라미터를 수정하는 방법으로, 기존의 모델 편집 방식에서 지식 보존(Error e0)과 업데이트(Error e1) 간의 균형 문제를 해결합니다. 이 과정에서 perturbation은 보존된 지식의 null space에 투영되어, post-edited LLM의 출력을 보존된 지식에 대해 변경하지 않도록 보장합니다.

- **Performance Highlights**: 다양한 LLM에 대한 실험 결과, AlphaEdit은 기존의 최상위 모델 편집 방법에 비해 평균 36.4%의 성능 향상을 달성했습니다. 이 방법은 기존 모델 편집 방법에 단 한 줄의 코드만 추가하여 쉽게 통합될 수 있습니다.



### How Much Can RAG Help the Reasoning of LLM? (https://arxiv.org/abs/2410.02338)
- **What's New**: 본 연구는 Retrieval-Augmented Generation (RAG) 기법이 LLMs의 추론 능력을 향상시키는 방법을 심도 있게 탐구하고 있으며, 이 과정에서 문서의 정보 전처리와 관련된 새로운 접근법인 DPrompt tuning을 제시합니다.

- **Technical Details**: RAG는 외부 문서의 정보를 사용하여 LLM의 추론 과정을 지원하나, 이 과정이 고정 깊이 트리 구조에서 제한적임을 보여줍니다. LLM의 추론 과정은 고정 깊이에서 운영되며, RAG는 문서에서 중간 추론 결과를 활용할 수 있지만 노이즈 정보를 필터링하는 과정이 필요합니다. 연구는 한정된 트랜스포머 레이어 내에서 효과적인 필터링을 가능하게 하는 DPrompt tuning을 제안하여 성능 개선을 보여줍니다.

- **Performance Highlights**: RAG 기법을 통해 LLM이 더 복잡한 문제를 해결할 수 있음을 입증하였으며, 관련 문서 정보 추가 시 문제 해결 깊이를 증가시킬 수 있는 가능성을 발견했습니다. 하지만, 실제 RAG 시나리오에서 필터링 문제는 여전히 어려움이 존재하고, DPrompt tuning을 통해 이 문제를 효과적으로 개선할 수 있음을 보였습니다.



### Autonomous Self-Trained Channel State Prediction Method for mmWave Vehicular Communications (https://arxiv.org/abs/2410.02326)
Comments:
          Accepted for publication at European Wireless 2024

- **What's New**: 이 논문은 5G mmWave 차량 사용자를 위한 자율적이고 자기 학습 가능한 CSI(channel state information) 예측 프레임워크를 개발하여, 기존의 반응형 빔 스위칭 대신 사전 예측을 통한 프로액티브 프로세스를 목표로 합니다.

- **Technical Details**: 제안된 프레임워크는 주기적인 CSI 예측을 통해 gNB(base station)가 차량 UE 효과적으로 지원할 수 있도록 하며, RNN(recurrent neural network) 기반의 CSI 예측 모델을 훈련합니다. 이 모델은 V2V(vehicle-to-vehicle) 협력 인식 메시지(CAM)를 통해 수집된 데이터와 차량 사용자의 CSI 피드백을 결합하여 훈련됩니다.

- **Performance Highlights**: DeepMIMO 데이터 세트를 사용한 수치 평가 결과에서, RNN 기반의 CSI 예측 모델이 5G mmWave 차량 사용자에 대해 높은 정확도의 CSI 예측을 제공하는 것으로 나타났습니다.



### Post-edits Are Preferences Too (https://arxiv.org/abs/2410.02320)
Comments:
          To appear at the Ninth Conference on Machine Translation (WMT24)

- **What's New**: 본 논문에서는 기계 번역 분야의 Pairwise Preference (쌍 선호도) 피드백이 다른 형태의 피드백보다 신뢰성이 떨어진다는 점을 지적하며, Post-editing (수정 편집) 데이터를 활용하여 더 믿을 수 있는 인간의 선호도를 생성하는 방법을 제안합니다.

- **Technical Details**: Preference Optimization (PO) 기법을 통해, 수정 편집 후의 번역 결과를 새로운 기준으로 활용하고, 이를 통해 LLM (대규모 언어 모델)을 Fine-tune (미세 조정)합니다. 모델은 수정 편집 데이터로 사전 훈련 후, PO 손실 함수를 사용하여 선호도에 맞게 좀 더 우수한 번역 결과를 생성하도록 학습합니다.

- **Performance Highlights**: 사전 훈련 후 PO 손실로 Fine-tuning을 진행한 결과, 모델은 수정 편집과 유사한 출력을 상위 순위로 올리고, 기계 번역과 유사한 결과를 지향하지 않는 방식으로 훈련되었습니다. 이를 통해 품질과 신뢰성을 높인 번역 결과를 얻을 수 있음을 보여줍니다.



### CTARR: A fast and robust method for identifying anatomical regions on CT images via atlas registration (https://arxiv.org/abs/2410.02316)
- **What's New**: 이번 논문에서는 CT 이미지 분석 파이프라인의 전처리 단계로 사용할 수 있는 CT Anatomical Region Recognition(CTARR)이라는 새로운 방법을 제안합니다. 이 방법은 관심 있는 해부학적 부위를 자동으로 식별하고 나머지를 제거하여 깊은 학습 기반의 CT 이미지 분석에서 비효율성을 줄입니다.

- **Technical Details**: CTARR는 atlas registration을 기반으로 하며 CT 스캔 이미지에서 해부학적 영역을 신속하고 견고하게 식별합니다. 이 방법은 특정 해부학적 부위를 정의하는 바운딩 박스(bounding box)를 사용하여 새로운 관심 영역을 추가할 수 있도록 설계되었습니다. 전통적인 등록 방법보다 더 효과적인 이미지 등록 알고리즘을 개발하여 대규모 변환을 통해 최적 정렬을 제공합니다.

- **Performance Highlights**: 제안된 방법은 1131개의 CT 스캔을 평가하여 97.45-100%의 정확도로 관심 영역의 전경 복셀을 보존하고, 계산 시간은 0.1-0.21초로 신속하게 처리할 수 있음을 입증했습니다. 이는 기존 방법보다 2.0-12.7배 더 빠른 세분화 시간을 줄여주는 성과를 보였습니다.



### Morphological evaluation of subwords vocabulary used by BETO language mod (https://arxiv.org/abs/2410.02283)
Comments:
          in Spanish language

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models)에서 사용하는 서브워드 토큰화(subword tokenization) 알고리즘의 형태소(morpheme) 품질을 평가하는 새로운 방법을 제안합니다.

- **Technical Details**: 형태소 품질 평가는 관련성(relevance), 결속력(cohesion), 형태소 정확도(morphological accuracy)의 세 가지 품질 기준에 기반하며, BETO 모델의 토크나이저에 적용되었습니다. 사용된 알고리즘은 Wordpiece입니다.

- **Performance Highlights**: BETO 모델에서 생성된 어휘(vocabulary)의 형태소 품질이 낮다는 결론에 도달했으며, 큰 말뭉치(corpus)에서 트레이닝해도 품질이 개선되지 않는다는 결과를 보여줍니다.



### CoLLAP: Contrastive Long-form Language-Audio Pretraining with Musical Temporal Structure Augmentation (https://arxiv.org/abs/2410.02271)
Comments:
          4 pages

- **What's New**: 본 논문에서는 오디오 파형의 표현 학습에서 시간적 특성을 효과적으로 모델링하는 것이 매우 중요하다는 점을 강조합니다. 제안하는 CoLLAP(Contrastive Long-form Language-Audio Pretraining) 모델은 입력 오디오와 언어 설명의 인식 창을 대폭 확장하여, 최대 5분 길이의 오디오와 250단어를 초과하는 언어 설명에 대한 대조 학습을 가능하게 합니다.

- **Technical Details**: CoLLAP 모델은 대조 학습 아키텍처를 통해 언어 표현과 구조화된 오디오 표현을 융합하고, 각 곡을 클립으로 분할하여 이들의 임베딩을 추출합니다. 주의(attention) 메커니즘을 통해 다중 모드 간의 시간적 상관관계를 포착하고, 최종 융합 점수를 자동으로 조정하여 개선된 대조 정렬을 가능하게 합니다. 모델은 RoBERTa와 GPT-2 같은 다양한 백본 언어 모델을 사용하여 다양한 변형이 개발되었습니다.

- **Performance Highlights**: 다양한 장기 음악-텍스트 검색 데이터셋에 대한 포괄적인 실험을 통해 CoLLAP 모델이 기반 모델들에 비해 검색 정확도에서 일관된 성능 향상을 보여주었습니다. 또한, CoLLAP 사전 훈련 모델은 이질적인 장기 다중 모드 문맥을 포함하는 다양한 음악 정보 검색 작업으로 전이 가능성을 입증했습니다. 특히 CoLLAP-GPT2 변형은 RoBERTa 모델보다 더 나은 일반화 성능을 보여주었습니다.



### Structural-Entropy-Based Sample Selection for Efficient and Effective Learning (https://arxiv.org/abs/2410.02268)
Comments:
          Submitted to ICLR 2025

- **What's New**: 이 논문은 샘플 선택의 효율성과 효과성을 개선하기 위해 텍스트의 정보와 대표성을 갖춘 샘플을 선택하는 새로운 방법을 제안합니다. 기존의 방법은 주로 로컬 정보에 의존하였으나, 본 연구는 전역 정보와 로컬 정보를 결합하여 샘플을 선택하는 방법을 개발합니다.

- **Technical Details**: 본 연구에서는 샘플을 그래프로 모델링하여 노드와 엣지를 사용하여 샘플 간의 유사성을 표현합니다. 전역 정보를 정량화하기 위해 구조적 엔트로피(structural entropy)를 활용하며, 이를 샤플리 값(Shapley value)을 사용해 개별 노드로 분해합니다. 새로 제안한 샘플 선택 방법인 SES는 샘플들의 kNN-그래프를 구성하고, 구조적 엔트로피와 학습 난이도를 결합하여 샘플의 중요도를 측정합니다.

- **Performance Highlights**: 세 가지 학습 시나리오(감독 학습, 능동 학습, 지속 학습)에서 진행된 실험에서 SES 방법이 기존 방법들에 비해 항상 우수한 성능을 보였습니다. 이는 본 방법이 선택된 샘플의 정보성과 대표성을 효과적으로 개선함을 보여줍니다.



### PFGuard: A Generative Framework with Privacy and Fairness Safeguards (https://arxiv.org/abs/2410.02246)
- **What's New**: 본 연구에서는 PFGuard라는 새로운 생성적 프레임워크를 제안하여, 개인정보 보호(privacy)와 공정성(fairness)을 동시에 충족할 수 있는 기술을 개발했습니다. 이 프레임워크는 고차원 데이터에서도 사용이 가능하며, 기존의 기술적 갈등을 해소하는 방안을 제공합니다.

- **Technical Details**: PFGuard는 여러 개의 중간教師 모델(teacher model) 앙상블을 사용하여 개인정보 보호와 공정성 간의 갈등을 조절합니다. 공정한 훈련 단계에서 새로운 샘플링 기법을 사용하여 공정한 모델을 훈련시키고, 개인정보 보호 훈련 단계에서 Private Teacher Ensemble Learning(PTEL)을 통해 랜덤 DP 노이즈와 결합함으로써 지식 전이를 보호합니다.

- **Performance Highlights**: PFGuard는 고차원 데이터, 특히 이미지에서 합성 데이터를 성공적으로 생성하고, 공정성과 개인정보 보호를 함께 보장하는 처음의 프레임워크입니다. 실험 결과는 PFGuard의 효과성과 기존의 단순 조합 방식의 한계를 강조합니다.



### Robust Weight Initialization for Tanh Neural Networks with Fixed Point Analysis (https://arxiv.org/abs/2410.02242)
- **What's New**: 이 논문은 tanh 활성화 함수를 사용하는 Feedforward Neural Networks (FFNN) 를 위한 새로운 가중치 초기화 방법을 제안합니다. 이 방법은 다양한 네트워크 크기에서 효과적인 학습을 촉진하며, 기존의 Xavier 초기화 방법을 초월하는 성능을 보여줍니다.

- **Technical Details**: 제안된 방법은 tanh(ax) 함수의 고정점을 분석하여, 활성화가 포화 상태에 이르지 않도록 a 값을 결정하는 데 초점을 맞춥니다. 이론적으로는, 고정점 분석을 통해 비활성화 방지 조건을 제시하고, 적절한 초기 가중치 행렬을 도출하는 방법을 설명합니다.

- **Performance Highlights**: 실험에서 제안된 방법은 MNIST, Fashion MNIST, CIFAR-10 데이터셋을 사용한 다양한 FFNN 네트워크 크기에서 더 높은 검증 정확도와 낮은 검증 손실을 기록했습니다. 또한, Physics-Informed Neural Networks (PINNs) 에서도 Xavier 초기화보다 빠른 수렴성과 네트워크 크기 변화에 대한 강한 견고성을 보여주었습니다.



### SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack (https://arxiv.org/abs/2410.02240)
- **What's New**: 이 논문은 기존의 무제한 적대 공격(Unrestricted Adversarial Attacks) 방식의 한계를 극복한 새로운 접근 방식인 'Semantic-Consistent Unrestricted Adversarial Attack (SCA)'를 제안합니다. 이 방법은 고급 의미 정보(Semantic Information)를 활용하여 적대적 예제를 생성하는 동안 의미 일관성을 유지합니다.

- **Technical Details**: SCA는 Denoising Diffusion Probabilistic Models (DDPM) 및 Multimodal Large Language Models (MLLM)을 활용하여 퍼트에이션(perturbation) 과정에서 의미적인 지침을 제공합니다. 이를 통해 샘플링 효율을 높이고, 의미 일관성이 유지되는 적대적 예제를 생성할 수 있습니다. DPM Solver++를 통해 시간 단계를 10-20으로 줄여 효율성을 극대화합니다.

- **Performance Highlights**: SCA는 평균 12배 더 빠른 속도로 적대적 예제를 생성할 수 있으며, 이전의 최첨단 기술과 유사한 성공률을 보입니다. 이 접근 방식은 의미적인 왜곡을 최소화하면서 자연스럽고 시각적으로 매력적인 결과를 제공합니다.



### EmbedLLM: Learning Compact Representations of Large Language Models (https://arxiv.org/abs/2410.02223)
- **What's New**: 본 연구에서는 Huggingface에서 수백만개의 대형 언어 모델(LLMs)을 효율적으로 평가하고 활용하기 위한 EmbedLLM이라는 프레임워크를 제안합니다. 기존 방법들이 각 태스크에 맞는 표현을 반복적으로 학습하여 비효율성을 초래하는 문제를 해결하고자 합니다.

- **Technical Details**: EmbedLLM 프레임워크는 LLMs의 컴팩트한 벡터 표현을 학습하는 인코더-디코더 접근법을 도입합니다. 학습된 임베딩(embedding)은 다양한 모델의 특성과 태스크를 반영하여 성능 향상을 도모합니다. 이를 통해 정확도 예측(correctness forecasting), 모델 라우팅(model routing), 벤치마크 정확도 평가(benchmark accuracy evaluation)와 같은 다양한 다운스트림 태스크에 활용할 수 있습니다.

- **Performance Highlights**: Empirical results show that EmbedLLM은 모델 라우팅에서 정확도 및 대기 시간(latency) 모두에서 이전 방법들을 초월합니다. 또한, 추가적인 추론 비용 없이 여러 벤치마크에서 모델의 성능을 예측할 수 있음을 입증하였습니다. 실험을 통해 학습된 임베딩이 모델의 주요 특성을 포착하고, 유사한 특성을 가진 모델들이 임베딩 공간에서 가까운 위치를 유지함을 확인하였습니다.



### Buckle Up: Robustifying LLMs at Every Customization Stage via Data Curation (https://arxiv.org/abs/2410.02220)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 커스터마이징(customization) 과정에서 발생할 수 있는 위험인 '재판 무시(jailbreaking)' 공격을 방지하기 위한 효과적인 방어 프레임워크를 제안합니다. 제안된 방법은 데이터 큐레이션(data curation)을 활용해 일반 상식 텍스트의 안전성을 향상시키고, LLM이 재판 무시 공격에 강해지도록 합니다. 이는 LLM의 커스터마이징 과정 모든 단계에서 적용 가능하여 그 효율성을 높입니다.

- **Technical Details**: 이 연구에서 제안한 Ctrl 프레임워크는 LLM의 커스터마이징 파이프라인에 추가 모듈 없이 안전성 큐레이션된 데이터를 통합하여, 잠재적인 재판 무시 취약성을 완화합니다. 이는 재판에 노출된 LLM이 발생할 수 있는 유해한 응답과의 경우에 대해 더 높은 난이도(perplexity)를 보이는 경향을 활용하여, 안전성을 강조하고, 안전성이 새로운 정보로 인식되도록 텍스트를 큐레이션하는 접근으로 구성됩니다. 세 가지 단계에서 적용이 가능하여 매우 유연합니다.

- **Performance Highlights**: 실험 결과, Ctrl 데이터 큐레이션 적용 시, 여러 LLM에서 재판 무시 효과가 크게 줄어들며, 최대 100%의 책임 있는 응답을 생성할 수 있음을 보였습니다. 이는 Llama-3-8B, Vicuna-13B, Mistral-7B 등 다양한 모델에서 확인되었으며, 특히 일반 상식 데이터를 사용한 점에서 주목할 만합니다.



### Multi-modal clothing recommendation model based on large model and VAE enhancemen (https://arxiv.org/abs/2410.02219)
- **What's New**: 이 연구는 의류 추천을 위한 다중모드 패러다임을 제안합니다. 이는 의류 설명 텍스트와 이미지를 통합한 다중모드 분석 방법을 설계하고, 사전 훈련된 대형 언어 모델을 활용하여 사용자와 제품의 숨겨진 의미를 심층적으로 탐구합니다.

- **Technical Details**: 연구에서는 또한 변량 인코더(variational encoder)를 사용하여 사용자 정보와 제품 간의 관계를 학습합니다. 이는 추천 시스템에서의 콜드 스타트(Cold Start) 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: 본 연구는 광범위한 제거 실험(ablation experiments)을 통해 이 방법이 다양한 추천 시스템 방법들보다 뛰어난 성능 우위를 가지는 것을 검증하였으며, 추천 시스템의 종합적인 최적화에 대한 중요한 실질적인 지침을 제공합니다.



### Adapting Segment Anything Model to Melanoma Segmentation in Microscopy Slide Images (https://arxiv.org/abs/2410.02207)
- **What's New**: 이 논문은 Whole Slide Images (WSIs)에서의 멜라노마(segmentation을 위한 새로운 접근 방식을 제시합니다. Segment Anything Model(SAM)을 이용하여 미세현미경 이미지에서 멜라노마를 자동으로 분할하는 방법을 설명합니다.

- **Technical Details**: 우리의 방법은 초기 semantic segmentation 모델을 사용하여 초기 분할 마스크를 생성하고, 이를 SAM에 대한 프롬프트로 사용합니다. 우리는 중심과 그리드 프롬프트를 조합한 동적 프롬프팅 전략을 설계하여 초고해상도 슬라이드 이미지의 최적 커버리지를 달성하면서 프롬프트의 품질을 유지합니다. Segformer를 초기 segmentation 모델로, EfficientSAM을 세분화 모델로 선택했습니다.

- **Performance Highlights**: 실험 결과에 따르면, 우리의 접근 방식은 기존의 첨단 멜라노마 segmentation 방법들을 초월할 뿐만 아니라, Segformer의 기준 성능을 IoU 기준으로 9.1% 이상 개선했습니다.



### Measuring, Evaluating and Improving Logical Consistency in Large Language Models (https://arxiv.org/abs/2410.02205)
- **What's New**: 본 논문은 논리적 일관성(logical consistency)이 LLM(대규모 언어 모델)의 신뢰성과 신뢰성 보장을 위한 중요한 요소라는 점에 초점을 맞추고 있습니다. 연구자들은 논리적 일관성을 측정하기 위해 보편적인 프레임워크를 제안하며, 이를 통해 다양한 LLM의 성능을 평가하였습니다.

- **Technical Details**: 세 가지 기본적인 프록시(proxies)인 추이성(transitivity), 교환성(commutativity), 부정 불변성(negation invariance)을 사용하여 LLM의 논리적 일관성을 정량화하는 새로운 프레임워크를 제안합니다. 이러한 프레임워크는 다양한 도메인에 적용 가능하며, 부분 비교와 전체 비교를 지원합니다.

- **Performance Highlights**: 논리적 일관성을 향상시키기 위한 데이터 정제(data refinement) 및 증강(augmentation) 기술을 도입하여 LLM의 성능을 향상시켰습니다. 이 접근법으로 훈련된 LLM은 논리적 일관성이 개선되었으며, 이후의 로직 의존 알고리즘에서도 우수한 성능을 보였습니다. 전반적으로 논리적 일관성은 LLM의 신뢰성을 평가하는 데 유용한 지표가 될 수 있음을 보여줍니다.



### Can Language Models Take A Hint? Prompting for Controllable Contextualized Commonsense Inferenc (https://arxiv.org/abs/2410.02202)
Comments:
          Submitted to ACL Rolling Review. arXiv admin note: text overlap with arXiv:2302.05406

- **What's New**: 이번 연구에서는 'hinting'이라는 데이터 증강(data augmentation) 기술을 도입하여 맥락 있는 commonsense inference를 개선합니다. 이 기술은 하드(hard) 및 소프트(soft) 프롬프트를 활용한 접두사(prompts) 전략을 사용하여 추론 과정을 유도합니다.

- **Technical Details**: HINTING 기법은 주어진 이야기 맥락에서 commonsense assertion을 생성하는데 도움을 줍니다. 이 방법은 Target Sentence와 함께 등장하는 특정 관계를 통해 commonsense assertion의 주체, 관계, 객체를 효율적으로 추론할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, HINTING 기술은 ParaCOMET 및 GLUCOSE 데이터셋에서 일반적인 및 맥락 특정 추론의 성능을 향상시키며, 모델의 성능에 손상을 주지 않으면서 더 나은 통제력을 제공합니다.



### G2T-LLM: Graph-to-Tree Text Encoding for Molecule Generation with Fine-Tuned Large Language Models (https://arxiv.org/abs/2410.02198)
- **What's New**: 본 논문에서는 G2T-LLM이라는 새로운 분자 생성 접근 방식을 소개합니다. 이 방법은 graph-to-tree text encoding을 이용하여 분자 구조를 계층적 텍스트 형식으로 변환함으로써 대규모 언어 모델(LLM)에서 최적화된 처리를 가능하게 합니다.

- **Technical Details**: G2T-LLM은 분자 구조를 JSON 및 XML과 같은 tree-structured 형식으로 변환하여 LLM이 이해하고 처리할 수 있는 형태로 만듭니다. 이 과정에서 supervised fine-tuning을 통해 LLM이 유효하고 일관된 화학 구조를 생성하도록 훈련했습니다. 또한, token constraining 기술을 도입하여 LLM의 생성 과정이 기대하는 tree-structured 형식에 부합하도록 유도합니다.

- **Performance Highlights**: G2T-LLM은 여러 벤치마크 분자 생성 데이터셋에서 최첨단(SOTA) 모델들과 유사한 성능을 달성하였습니다. 이 결과들은 우리의 그래프-트리 인코딩 방식이 LLM이 화학적으로 타당하고 다양한 분자를 생성하는 데 효과적임을 검증합니다.



### BACKTIME: Backdoor Attacks on Multivariate Time Series Forecasting (https://arxiv.org/abs/2410.02195)
Comments:
          23 pages. Neurips 2024

- **What's New**: 이번 연구에서는 Multivariate Time Series (MTS) 예측 모델에 대한 backdoor 공격의 취약성을 탐구하고, 이를 해결하기 위한 새로운 공격 방법인 BackTime을 제안합니다. 특히, 정상적인 데이터에 최소한의 변화를 주어 모델의 예측을 공격자의 의도에 맞게 조작할 수 있는 방법을 제시합니다.

- **Technical Details**: BackTime은 시계열 데이터를 독성으로 변환하기 위해 bi-level optimization 프로세스를 formalize합니다. GNN 기반의 trigger generator를 사용하여 변수를 간섭할 수 있는 stealthy한 triggers을 생성하고, 공격자는 특정 변수에만 영향을 미치도록 sparse한 공격을 수행할 수 있습니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, BackTime은 최신 MTS 예측 모델에서 SOTA (state-of-the-art) 성능을 달성하였으며, 악의적인 입력에 직면했을 때도 모델의 예측 정확도를 유지하는 동시에 공격자의 의도에 따라 예측을 조정할 수 있는 효과를 보였습니다.



### A Survey on Point-of-Interest Recommendation: Models, Architectures, and Security (https://arxiv.org/abs/2410.02191)
Comments:
          20 pages

- **What's New**: 이 논문은 POI(Point-of-Interest) 추천 시스템의 최신 발전을 폭넓게 검토하며, 기존의 전통적인 접근 방식에서 벗어나 최첨단 기술, 새로운 아키텍처 및 보안 고려사항에 대한 포괄적인 분석을 제공합니다.

- **Technical Details**: POI 추천 시스템의 진화는 전통적인 잠재 요인 모델에서 심층 학습 구조로 변화하고 있습니다. Latent Dirichlet Allocation (LDA) 및 Matrix Factorization (MF)에서 Long Short-Term Memory (LSTM) 네트워크, Transformer 아키텍처와 Graph Neural Networks (GNNs)로의 발전을 다루며, LLMs, Diffusion Models (DMs), Self-Supervised Learning (SSL)와 같은 최신 기법이 추천 정확도를 어떻게 향상시키는지에 대해 논의합니다. 또한, 중앙 집중 모델에서 분산 학습(federated learning)으로의 이행이 어떻게 시스템의 확장성과 사용자 프라이버시를 개선하는지에 대해서도 밝힙니다.

- **Performance Highlights**: 최근 연구들은 POI 추천에서 사용자 데이터의 보안을 강화하기 위한 다양한 기법을 채택하고 있으며, differential privacy와 federated learning이 현대 POI 추천 시스템의 중심에 놓이고 있습니다. 추천 정확도를 향상시키는 혁신적인 모델과 아키텍처의 발전을 통해, 시스템은 사용자 선호도를 보다 잘 모델링할 수 있게 되었습니다.



### POSIX: A Prompt Sensitivity Index For Large Language Models (https://arxiv.org/abs/2410.02185)
Comments:
          EMNLP 2024 (Findings)

- **What's New**: 이 논문에서는 POSIX라는 새로운 지표(PrOmpt Sensitivity IndeX)를 제안하여, Large Language Models(LLMs)의 프롬프트 감도(prompt sensitivity)를 평가하는 신뢰할 수 있는 방법을 제공합니다.

- **Technical Details**: POSIX는 주어진 응답에 대한 로지우드(likelihood)의 상대적 변화를 포착하는 것을 목표로 하며, 다양한 개방형 소스 LLM의 프롬프트 감도를 측정하고 비교하는 데 사용됩니다. 이 연구에서는 응답 다양성(response diversity), 분포의 엔트로피(response distribution entropy), 의미적 일관성(semantic coherence), 신뢰도 변동(variance in confidence) 등 네 가지 주요 요인을 고려하여 LLM의 감도를 측정하는 방법론을 제시합니다.

- **Performance Highlights**: 초기 실험 결과, 매개변수 수를 단순히 증가시키거나 지침 조정(instruction tuning)을 하는 것이 프롬프트 감도를 낮추지 않는 반면, 몇 개의 샘플을 추가하면 감도가 현저히 감소하는 경향을 보여줍니다. MCQ(다중 선택 문제) 유형의 작업에서는 프롬프트 템플릿 변경이 가장 높은 감도를 나타내며, 개방형 생성 과제의 경우 패러프레이징(paraphrasing)이 가장 높은 감도를 보이는 것으로 나타났습니다.



### Efficiently Deploying LLMs with Controlled Risk (https://arxiv.org/abs/2410.02173)
Comments:
          10 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 배포할 때 효율성과 리스크 관리를 동시에 고려해야 한다는 점을 강조하며, 모델 내재적 불확실성을 이용하여 계층적인 쿼리 위임이 가능한 다층적 포기(hierarchical chains with multi-level abstention, HCMA)를 제안합니다. 이는 훈련 없는 모델 전환을 가능하게 하여 검은 상자 API 호출만으로도 작동하게 합니다.

- **Technical Details**: HCMA는 LLM 지능 계층을 따라 쿼리를 위임하며, 선택적 예측(selective prediction)을 통해 리스크를 관리합니다. 또한, 데이터 효율적인 로지스틱 회귀를 사용하여 좋은 정확도와 함께 ECE(Expected Calibration Error)를 50% 감소시킵니다. 이를 위해 단 50개 또는 100개의 레이블 예시만으로도 우수한 교정 오류를 달성할 수 있습니다.

- **Performance Highlights**: MMLU에서 20%의 쿼리를 포기할 경우 Llama3 405B의 오류율을 30% 감소시키는 성과를 보여주었습니다. 또한, TruthfulQA에서 제로샷 프롬프트(zero-shot prompting)를 사용하면 높은 포기 비율에서 오류가 0%에 도달하는 결과를 얻어냈습니다.



### Abstract Reward Processes: Leveraging State Abstraction for Consistent Off-Policy Evaluation (https://arxiv.org/abs/2410.02172)
Comments:
          Accepted at the Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 오프-정책 평가(off-policy evaluation, OPE)를 위한 새롭고 통합적인 프레임워크인 STAR를 소개합니다. STAR는 다양한 기존 OPE 방법들을 포함하는 추정기들을 포괄하며, 이들 방법보다 더 낮은 평균 제곱 예측 오차를 달성합니다.

- **Technical Details**: STAR 프레임워크는 상태 추상화(state abstraction)를 활용하여 복잡하고 잠재적으로 연속적인 문제를 간결하고 이산적인 모델인 추상 보상 프로세스(abstract reward processes, ARPs)로 변환합니다. 이 과정에서 중요도 샘플링(importance sampling) 기법을 모델 학습에 통합하여 예측의 일관성을 확보합니다.

- **Performance Highlights**: 모든 12개의 경우에서 BEST STAR 추정기가 기준선(baseline)보다 우수한 성과를 내며, 중위수 STAR 추정기 또한 12개의 경우 중 7개에서 기준선을 초과하는 성능을 보였습니다.



### RiskSEA : A Scalable Graph Embedding for Detecting On-chain Fraudulent Activities on the Ethereum Blockchain (https://arxiv.org/abs/2410.02160)
Comments:
          arXiv admin note: text overlap with arXiv:2203.12363 by other authors

- **What's New**: 이번 연구에서는 Fraudulent activity와 관련된 Ethereum 블록체인 주소의 리스크 점수를 생성하기 위한 RiskSEA 시스템을 소개합니다. 이 시스템은 node2vec 알고리즘을 활용해 동적이고 대규모의 거래 그래프를 다루는 새로운 접근 방식을 제공합니다.

- **Technical Details**: RiskSEA는 1) 리스크 점수를 생성하기 위한 node2vec embedding을 생성하는 확장 가능한 접근 방식, 2) 트랜잭션 기반의 특성, 3) 노드2vec embedding과 행동적 특성을 결합한 분류자 모델을 포함합니다. 두 가지 새로운 node2vec embedding 생성 접근 방식이 있습니다: node2vec embedding 전파 및 동적 node2vec embedding입니다.

- **Performance Highlights**: 실험 결과, 행동적 특성과 node2vec 특성을 결합했을 때 분류 성능이 크게 향상되며, 동적 node2vec embedding이 전파된 embedding보다 우수한 성능을 나타냅니다.



### Mitigating Memorization In Language Models (https://arxiv.org/abs/2410.02159)
- **What's New**: 이 연구는 Language Models (LMs)가 정보를 '기억하는'(memorize) 능력을 줄이기 위한 여러 방법론을 제안합니다. 이 중 세 가지 regularizer 기반, 세 가지 fine-tuning 기반, 그리고 열한 가지 unlearning 기반 방법을 포함하고 있으며, 그 중 다섯 가지는 새로운 방법으로 제시되었습니다. 또한, 이 연구는 기억 완화 기법을 빠르게 개발하고 평가할 수 있는 작은 규모의 LMs인 TinyMem을 소개합니다.

- **Technical Details**: 신문에서는 LM이 학습 데이터를 메모리에 저장하고 이를 의도하지 않게 반출할 수 있는 문제를 다룹니다. 연구진은 방법론을 평가하기 위해 TinyMem을 활용하여 다양한 사례에서 기억 완화 방법의 효과를 측정했습니다. 정리된 방법들은 regularizer 방법이 느리고 효과적이지 않으며, fine-tuning 방법은 비쌉니다. 반면 unlearning 방법은 빠르고 효과적인 것으로 확인되었습니다. 특히 새로운 unlearning 기법인 BalancedSubnet이 가장 뛰어난 성과를 보였습니다.

- **Performance Highlights**: 연구 결과, 기존의 방법들과 비교해 BalancedSubnet이 기억된 정보를 제거하면서도 목표 작업의 성능을 유지하는 데 가장 효과적임을 보여주었습니다. TinyMem에서 개발된 완화 방법들은 큰 생산 모델에서도 적용 가능성을 입증하였습니다.



### The why, what, and how of AI-based coding in scientific research (https://arxiv.org/abs/2410.02156)
Comments:
          23 pages, 7 figure, 3 boxes

- **What's New**: 본 논문은 연구자들이 프로그래밍을 더욱 직관적으로 활용할 수 있도록 돕기 위해 Generative AI와 대형 언어 모델(LLMs)의 역할을 분석하고, AI 기반 코딩의 실제적인 활용법을 제시합니다.

- **Technical Details**: 저자들은 LLM을 활용한 코딩의 세 가지 주요 관점을 다루는데, 첫째는 LLM이 코딩에서 어떤 역할을 수행하는지(why), 둘째는 LLM이 제공하는 6가지 코딩 지원 유형(what), 셋째는 5단계의 실제적인 구현 전략을 포함한 워크플로우(how)입니다.

- **Performance Highlights**: 이 프레임워크는 AI를 활용하여 코딩 관행과 교육을 개선함으로써 과학적 진전을 가속화하는 데 도움을 주기 위한 실행 가능한 통찰력을 제공합니다.



### Efficient Source-Free Time-Series Adaptation via Parameter Subspace Disentanglemen (https://arxiv.org/abs/2410.02147)
- **What's New**: 이 논문에서는 시간 계열 데이터의 컨텍스트에서 Source-Free Domain Adaptation (SFDA)을 위한 효율적인 프레임워크를 제안합니다. 특히, 매개변수 효율성과 데이터 샘플 활용을 모두 향상시키는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 소스 모델의 가중치를 Tucker 스타일로 재매개화(reparameterize)하여 소스 모델 준비 단계에서 모델을 압축된 형태로 분해(factorize)합니다. 타겟 적응 과정에서는 이러한 분해된 요소의 일부만 미세 조정(fine-tuning)하여 학습 효율을 크게 개선합니다.

- **Performance Highlights**: PAC Bayesian 분석을 통해 선택적 미세 조정 전략이 적응 과정을 암묵적으로 정규화(regularizes)하여 모델의 학습 능력을 제약함을 보입니다. 이 재매개화는 전체 모델 크기를 줄이고 추론(inference) 효율성을 향상시켜 자원 제약이 있는 장치에 특히 적합합니다. 또한 다양한 SFDA 방법과 호환되어 +90%의 MACs로 미세 조정된 매개변수 수와 추론 오버헤드를 줄이면서 모델 성능을 유지합니다.



### The Impact of Generative AI on Collaborative Open-Source Software Development: Evidence from GitHub Copilo (https://arxiv.org/abs/2410.02091)
- **What's New**: 이번 연구는 GitHub Copilot의 사용이 오픈 소스 커뮤니티에서 소프트웨어 개발에 미치는 영향을 조사한 것입니다. 이는 자동화된 콘텐츠 생성이 소프트웨어 개발에 미치는 영향을 평가한 최초의 연구로, Copilot이 프로젝트 수준에서 생산성을 얼마나 증가시키는지를 실질적으로 분석하였습니다.

- **Technical Details**: 연구는 GitHub의 오픈 소스 저장소 데이터셋을 활용하여 일반화된 합성 통제 방법(Generalized Synthetic Control Method)을 적용하였으며, Copilot 사용으로 인해 프로젝트 수준의 생산성이 6.5% 증가함을 발견했습니다. 개인 생산성은 5.5%, 참여도는 5.4% 증가했으나 통합 시간은 41.6% 증가하는 것으로 나타났습니다.

- **Performance Highlights**: 핵심 개발자들은 Copilot 사용으로 인해 더 큰 프로젝트 수준의 생산성 증가를 경험하는 반면, 주변 개발자들은 상대적으로 작은 증가를 보였습니다. 흥미롭게도, 코드 품질에는 변화가 없었으며, AI 페어 프로그래머는 개발자의 코드 자동화 및 증대에 기여하지만, 인간 개발자의 프로젝트에 대한 이해도가 이러한 혜택을 더욱 강화시킨다는 결론을 내렸습니다.



### RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning (https://arxiv.org/abs/2410.02089)
- **What's New**: 해당 논문에서는 사용자 지정 작업을 수행하는 대형 언어 모델(LLMs)의 성과를 개선하기 위해, 코드 생성(code synthesis) 영역에서 실행 피드백(execution feedback)을 효과적으로 활용하는 강화 학습 방법론을 제안합니다. 이 접근법은 기존의 독립 샘플링(independent sampling)과 비교할 때 반복적으로 코드 개선을 도모할 수 있는 새로운 프레임워크를 제공합니다.

- **Technical Details**: 코드 생성을 다중 단계의 대화(interactive conversation) 과정으로 구조화하여, LLM이自然语言 문제 설명에 대한 코드 솔루션을 반복적으로 생성하도록 하고, 각 솔루션의 실행 결과를 자동으로 평가합니다. 이 평가 결과는 다음 코드 시도를 위한 추가 컨텍스트로 제공되며, 이는 end-to-end 최적화(end-to-end optimization) 및 강화 학습 알고리즘을 통한 최대 보상 신호(maximize reward signal)로 이어집니다. 특히, Proximal Policy Optimization (PPO) 알고리즘을 사용하여 정책을 최적화합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법론은 CodeContests 벤치마크에서 이전 최고 성능을 넘어서며, 샘플의 수를 10배 이상 줄이는 데 성공했습니다. 이는 대형 및 소형 모델 모두에서 관찰되며, HumanEval+ 및 MBPP+와 같은 알려진 코드 생성 벤치마크에서 일반화된 성과를 보여줍니다.



### Multi-Omic and Quantum Machine Learning Integration for Lung Subtypes Classification (https://arxiv.org/abs/2410.02085)
Comments:
          27 pages, 17 figures

- **What's New**: 이번 연구는 준양자 기계 학습(Quantum Machine Learning) 방법을 이용하여 폐암 아형인 폐 편평세포 암종(LUSC)과 폐 선암(LUAD)의 진단 분류 및 다중 오믹스(multi-omics) 데이터 통합을 탐구하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구에서는 GDC-TCGA 데이터셋에서 DNA 메틸화(DNA Methylation), RNA 시퀀싱(RNA Sequencing), 마이크로RNA 시퀀싱(miRNA Sequencing)의 다중 오믹스 데이터를 통합하여 LUSC와 LUAD의 분포를 분석합니다. 특히, 특징 선택(feature selection)과 진단 분류를 통해 각 아형의 생물학적 특성을 규명하며, 양자 분류(Classification) 방법과 기존의 학습 기법을 결합한 하이브리드 모델을 개발하였습니다.

- **Performance Highlights**: 향상된 정확도를 통해 폐암 아형의 식별 능력을 제공하며, 다중 오믹스 데이터의 통합 분석을 통해 생물학적 마커를 발견할 수 있는 가능성을 열어줍니다. 연구 결과는 폐암 치료의 발전에 기여할 것으로 기대됩니다.



### Kolmogorov-Arnold Network Autoencoders (https://arxiv.org/abs/2410.02077)
Comments:
          12 pages, 5 figures, 1 table

- **What's New**: 최근 연구에서 다층 퍼셉트론(MLP)을 대체할 유망한 방법으로 Kolmogorov-Arnold Networks(KAN)가 소개되었습니다. KAN은 노드가 아닌 엣지에 활성화 함수를 배치하는 구조적 변화를 통해 모델의 정확성과 해석 가능성을 향상시킬 수 있습니다.

- **Technical Details**: KAN의 구조는 Kolmogorov-Arnold 표현 정리에 기반하며, 네트워크의 엣지에서 직접 활성화 함수를 학습하고 적용합니다. 이는 정보 흐름의 역학을 변화시키고, 복잡한 데이터의 의존성을 더 잘 처리할 수 있도록 도와줍니다. KAN은 이미지 표현 작업을 위한 오토인코더(encoder-decoder) 아키텍처에 적용됩니다.

- **Performance Highlights**: KAN 기반 오토인코더는 MNIST, SVHN, CIFAR-10 데이터셋에서 전통적인 CNN과 비교하여 경쟁력 있는 재구성 정확도를 달성하는 것을 보여주었습니다. 이는 KAN이 데이터 분석 작업에서 효과적인 도구로 자리 잡을 가능성을 시사합니다.



### Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data (https://arxiv.org/abs/2410.02056)
Comments:
          Code and Checkpoints will be soon available here: this https URL

- **What's New**: Synthio는 합성 데이터를 통해 소규모 오디오 분류 데이터 세트를 증강하는 혁신적인 방법을 제안합니다. 이 방법은 제한된 라벨 데이터로 오디오 분류 정확도를 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: Synthio는 두 가지 주요 단계를 포함합니다: 1) Preference Optimization을 통해 T2A(텍스트-투-오디오) 모델의 생성물을 소규모 데이터 세트와 정렬합니다. 2) MixCap이라는 방법으로 다양한 오디오 캡션을 생성하고 이를 통해 다양한 합성을 유도합니다.

- **Performance Highlights**: Synthio는 10개의 데이터 세트와 4개의 시뮬레이션된 제한 데이터 환경에서 평가되었으며, 약한 캡션을 가진 AudioSet에서 학습된 T2A 모델을 사용하여 모든 기준점보다 0.1%-39% 향상된 성능을 보여주었습니다.



### EAB-FL: Exacerbating Algorithmic Bias through Model Poisoning Attacks in Federated Learning (https://arxiv.org/abs/2410.02042)
- **What's New**: 본 논문은 그룹 불공정성을 악화시키는 새로운 유형의 모델 중독 공격인 EAB-FL을 제안합니다. 이 공격은 모델 유용성을 유지하면서 특정 집단에 대한 알고리즘 편향을 악화시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: EAB-FL에서는 레이어별 관련성 전파 (Layer-wise Relevance Propagation, LRP)를 사용하여 국소적으로 훈련된 모델의 잉여 공간을 식별하고, 이 공간 내에서 모델 매개변수를 조정하여 특정 그룹에 대한 모델 성능을 저하시킵니다. 이 과정에서 공격자는 소규모의 악의적인 클라이언트 장치를 통해 훈련 과정을 조작할 수 있습니다.

- **Performance Highlights**: EAB-FL은 세 가지 데이터셋에 대한 광범위한 실험을 통해 그룹 불공정성을 악화시키는 데 효과적임을 입증했습니다. 또한 최신 공정성 최적화 알고리즘과 안전한 집계 규칙 하에서도 공격의 유효성을 지속적으로 유지하는 성과를 보였습니다.



### Model Comparisons: XNet Outperforms KAN (https://arxiv.org/abs/2410.02033)
- **What's New**: 본 논문에서는 XNet이라는 혁신적인 알고리즘을 소개하며, 이는 복소수 Cauchy 적분 공식을 기반으로 하여 기존의 Multi-Layer Perceptrons (MLPs) 및 Kolmogorov-Arnold Networks (KANs)를 초월하는 우수한 네트워크 아키텍처를 제공합니다. XNet은 다양한 작업에서 속도와 정확성을 크게 개선하며 데이터 기반 모델 개발의 새로운 지평을 열고 있습니다.

- **Technical Details**: XNet은 Cauchy 적분 정리를 활용하여 설계된 신경망 모델로, 고유하게 설계된 Cauchy 활성화 함수를 포함하고 있습니다. 이는 훈련 중에 최적화되는 매개변수 λ1, λ2 및 d를 통해 수학적으로 표현됩니다. 이 모델은 저차원 PDE와 같은 복잡한 함수 근사 작업에서 MLP 및 KAN보다 뛰어난 성능을 나타내며, 활성화 함수의 집중 응답 특성으로 인해 국소 데이터 세그먼트를 정교하게 조정할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: XNet은 여러 실험에서 KAN 및 MLP보다 기능 근사 성능이 우수하며, 특히 Heaviside 단계 함수 및 복잡한 고차원 시나리오에서 그 강점을 나타냅니다. 또한 Physics-Informed Neural Network (PINN) 프레임워크 내에서 Poisson 방정식을 기준으로 할 때, XNet은 기존 네트워크보다 월등한 성능을 발휘하며, LSTM 아키텍처에서 전통적인 피드포워드 신경망을 XNet으로 대체함으로써 XLSTM 모델을 소개하고, 이 모델이 정확성과 신뢰성 면에서 전통 LSTM 모델을 지속적으로 초월함을 검증했습니다.



### Quantifying the Gaps Between Translation and Native Perception in Training for Multimodal, Multilingual Retrieva (https://arxiv.org/abs/2410.02027)
Comments:
          Short paper accepted to EMNLP24 (Main)

- **What's New**: 이 논문에서는 다양한 언어와 문화에서 이미지 캡션의 인식 차이를 적절하게 반영하는 다국어 비전-언어 모델(multilingual vision-language models)의 부족함을 지적하고, 이를 해결하기 위한 방법론을 제안합니다. 특히, 독일어 사용자 지각을 반영한 캡션과 영어에서 기계 번역 혹은 사람 번역을 통해 만들어진 캡션 간의 성능 차이를 조사합니다.

- **Technical Details**: 이 논문에서는 다국어 CLIP(mCLIP)을 활용하여 독일어 이미지-텍스트(I2T) 및 텍스트-이미지(T2I) 검색 작업을 수행합니다. 이를 위해 Multi30K 데이터셋에서 독일어 원본 캡션(natively written captions)과 전문 번역 캡션(human-translated captions)을 비교하여 성능 차이를 정량화합니다. 추가로, 하이퍼님화(hypernymization) 데이터 증강 기법, 대형 언어 모델(LLM)인 LLaMA-3를 사용한 파라프레이징(paraphrasing) 기법 등으로 번역 캡션 개선을 시도합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법을 통해 평균 검색률(mean recall)이 1.3% 향상되는 결과를 보였으나, 여전히 원어민의 지각(perception)을 반영한 캡션과의 성능 차이가 존재합니다. 이는 앞으로의 연구 방향에 대한 필요성을 강조하며, 다국어 비전-언어 모델의 진전을 위한 열린 문제로 남아 있습니다.



### A Likelihood Based Approach to Distribution Regression Using Conditional Deep Generative Models (https://arxiv.org/abs/2410.02025)
Comments:
          arXiv admin note: text overlap with arXiv:1708.06633 by other authors

- **What's New**: 본 연구는 조건부 깊은 생성 모델(conditional deep generative models)의 이론적 특성을 고차원 공간 내에서 응답 변수가 낮은 차원의 매니폴드(manifold)에 집중하는 통계적 분포 회귀(distribution regression) 프레임워크에서 탐색합니다. 특히, 최대 우도 추정기(sieve maximum likelihood estimator, MLE)의 수렴 속도를 연구하며, 이 결과는 조건부 분포의 추정에 대한 새로운 이해를 제공합니다.

- **Technical Details**: 제안된 접근 방식은 조건부 분포를 추정하기 위해 깊은 생성 모델을 활용하는 likelihood-based 방법입니다. 이 모델은는 다양한 차원에서 오염된 데이터를 다루며, 조건부 생성기는 심층 신경망(deep neural networks, DNNs)을 사용하여 모델링됩니다. 이 연구는 Hellinger 거리 및 Wasserstein 수렴 속도를 통해 추정 거리의 수렴 속도를 특수화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 합성 및 실제 데이터 세트를 통해 그 효과적인 구현을 보여주며, 이론적 발견을 검증합니다. 조건부 깊은 생성 모델을 이용한 새로운 접근 방식이 여러 데이터 구조를 잘 학습할 수 있음을 입증하였습니다.



### FLAG: Financial Long Document Classification via AMR-based GNN (https://arxiv.org/abs/2410.02024)
Comments:
          8 pages, 3 figures, to be published in CIFEr Conference 2024 as "Semantic Graph Learning for Trend Prediction from Long Financial Documents"

- **What's New**: 이번 연구에서는 긴 금융 문서 분류를 위한 AMR 기반의 GNN 프레임워크인 FLAG(Financial Long document classification via AMR-based GNN)를 제안하였습니다. 이 방법은 장기 금융 문서의 의미론적 관계를 더욱 잘 이해할 수 있도록 도와줍니다.

- **Technical Details**: FLAG는 문서의 각 문장을 AMR 그래프로 변환한 후, 이를 기반으로 문서 수준의 AMR 그래프를 계층적으로 구성합니다. 초기화된 노드는 금융 도메인에서 훈련된 FinBERT를 통해 생성된 LLM word embeddings로 이루어집니다. 이를 통해 GNN 모델을 적용하여 최종 문서 표현을 생성합니다.

- **Performance Highlights**: FLAG는 S&P 1500 지수의 다양한 기업의 분기 실적 발표 기록을 포함한 데이터셋에서 직접 텍스트로 LLM을 미세 조정한 기존 방법들보다 주가 동향 예측에서 더 높은 성능을 보여주었습니다.



### DeepProtein: Deep Learning Library and Benchmark for Protein Sequence Learning (https://arxiv.org/abs/2410.02023)
- **What's New**: 최근 딥 러닝(ded learning) 기술의 발전은 단백질 과학(protein science) 분야에 혁신을 가져왔습니다. 본 논문에서는 단백질 관련 작업을 위해 설계된 포괄적이고 사용자 친화적인 딥 러닝 라이브러리인 DeepProtein을 소개합니다.

- **Technical Details**: DeepProtein은 CNN(convolutional neural network), RNN(recurrent neural network), transformer, GNN(graph neural network), GT(graph transformer) 등 최신 신경망(neural network) 아키텍처를 통합하여 단백질 기능 예측, 단백질 위치 예측, 단백질-단백질 상호작용 예측 등 다양한 단백질 작업에 대한 벤치마크를 제공합니다. 또한, 사용자는 직관적인 인터페이스를 통해 딥 러닝 기법을 쉽게 적용할 수 있습니다.

- **Performance Highlights**: DeepProtein은 8개의 신경망 아키텍처에 대한 성능을 평가하였으며, 각 작업에서의 장단점을 분석합니다. 또한 DeepProtein은 사용자가 한 줄의 명령어로 모든 방법을 실행할 수 있도록 설계되었습니다. 이 라이브러리는 연구 reproducibility를 촉진하기 위해 자세한 문서와 튜토리얼을 제공합니다.



### Review Non-convex Optimization Method for Machine Learning (https://arxiv.org/abs/2410.02017)
- **What's New**: 이번 논문에서는 비볼록 최적화(non-convex optimization)의 핵심 방법 및 머신 러닝에서의 응용을 다룹니다. 주요 초점은 다수의 로컬 미니마(local minima) 및 새들 포인트(saddle points)를 효과적으로 다루는 방법에 있습니다.

- **Technical Details**: 비볼록 최적화 기법은 주로 경량화와 모델 성능 향상을 동시에 도모합니다. 예를 들어, 경량화(model pruning) 및 정규화(regularization) 기법을 통해 모델의 크기를 줄이면서도 성능을 유지할 수 있습니다. 또한 확률적 경량기법(stochastic gradient descent)이나 적응형 방법(Adam, AdaGrad 등)을 사용하여 학습 속도 및 수렴 속도를 높이는 방법도 탐구됩니다.

- **Performance Highlights**: 비볼록 최적화는 정확성을 높이고 수렴 속도를 개선하는 동시에 계산 비용을 절감하는데 크게 기여합니다. 논문은 향후 연구 방향과 비즈니스(application 및 scalability) 질문에 대해서도 논의합니다.



### Addressing Data Heterogeneity in Federated Learning with Adaptive Normalization-Free Feature Recalibration (https://arxiv.org/abs/2410.02006)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 고객 데이터의 통계적 이질성 문제를 해결하기 위한 Adaptive Normalization-free Feature Recalibration (ANFR) 라는 새로운 접근 방식을 제안합니다. ANFR은 가중치 표준화(weight standardization)와 채널 주의(channel attention) 메커니즘을 결합한 아키텍처 수준의 방법입니다.

- **Technical Details**: ANFR은 가중치 표준화를 통해 각 레이어의 가중치를 정규화하여 비일관적인 클라이언트 통계의 영향을 줄입니다. 채널 주의는 각 피처 맵에 대해 학습 가능한 스케일링 요인을 생성해 이질성으로 인해 비일관적인 피처를 억제하고 일관된 피처를 강조합니다. 이 과정은 모델이 클라이언트 간에 공유되며 유용한 피처에 집중할 수 있도록 합니다.

- **Performance Highlights**: ANFR은 다양한 데이터 세트와 집합 방법(aggregation methods)에서 기존의 기준선(baselines)보다 일관되게 우수한 성능을 보였습니다. 이 접근 방식은 FL의 다양한 시나리오에서 성능, 안정성 및 개인 정보 보호 기능을 향상시키는 데 기여합니다.



### Normalizing Flow Based Metric for Image Generation (https://arxiv.org/abs/2410.02004)
Comments:
          15 pages, 16 figures

- **What's New**: 이 논문에서는 이미지의 '리얼리즘'(realness)을 평가하기 위해 새로운 두 가지 평가 지표를 제안합니다. 첫 번째는 단순하고 효율적인 flow-based likelihood distance (FLD)이고, 두 번째는 더 정확한 dual-flow based likelihood distance (D-FLD)입니다. 이러한 지표들은 정규화 흐름(normalizing flows)을 사용하여 생성된 이미지가 실제 이미지 분포와 얼마나 밀접하게 일치하는지를 평가합니다.

- **Technical Details**: 정규화 흐름은 정확한 likelihood를 계산할 수 있는 기능을 가지므로, FLD와 D-FLD는 생성된 이미지가 실제 이미지와의 분포적 일치를 어떻게 평가하는지를 제공합니다. 이 지표들은 수백 장의 이미지만으로도 안정된 결과를 도출할 수 있으며(즉, 평균 수렴), 기존의 Fréchet inception distance (FID)와 비교해도 파라미터 수가 현저히 적고 계산적으로 효율적입니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 지표들이 다양한 종류의 이미지 손상에 대해 원하는 단조(monotonic) 관계를 유지한다는 것을 입증했습니다. 이러한 특성 덕분에 새로운 도메인에서도 신뢰할 수 있는 평가가 가능하며, FID보다 속도가 향상되고 더 적은 수의 이미지로도 안정적인 평가를 제공합니다.



### UlcerGPT: A Multimodal Approach Leveraging Large Language and Vision Models for Diabetic Foot Ulcer Image Transcription (https://arxiv.org/abs/2410.01989)
Comments:
          13 pages, 3 figures, ICPR 2024 Conference (PRHA workshop)

- **What's New**: 이 논문에서는 UlcerGPT라는 새로운 다중 모달 접근법을 소개하여, 대형 언어 모델과 비전 모델을 활용하여 당뇨병성 족부 궤양(Diabetic Foot Ulcers, DFU) 이미지를 전사하고 분석합니다.

- **Technical Details**: UlcerGPT는 Large Language and Vision Assistant (LLaVA) 및 Chat Generative Pre-trained Transformer (ChatGPT)를 결합하여 DFU 이미지를 공동으로 감지, 분류 및 중요한 영역을 위치 지정하여 전사합니다. 연구에 사용된 데이터셋은 DFU2022 대회에서 수집된 2000개의 주석이 달린 임상 RGB 이미지입니다.

- **Performance Highlights**: UlcerGPT는 임상 전문가에 의해 평가된 공개 데이터셋에서 DFU 전사에 대한 정확성과 효율성을 보여주며, 원격 의료를 통한 신속한 치료 제공을 지원할 수 있는 잠재력을 가지고 있습니다.



### LLM+KG@VLDB'24 Workshop Summary (https://arxiv.org/abs/2410.01978)
Comments:
          7 pages, 1 figure

- **What's New**: LLM+KG'24 워크숍에서는 대규모 언어 모델(LLMs)과 지식 그래프(KGs) 간의 통합이 다루어졌습니다. 주제는 데이터 관리의 도전과 기회로, LLM과 KG의 효율적인 상호작용이 초점이었습니다.

- **Technical Details**: LLMs는 자연어 처리에서 혁신적인 역할을 하고 있으나, 지속적인 지식 표현의 일관성을 결여하고 있어 왜곡된 출력이 발생할 수 있습니다. KGs는 외부의 사실 기반 업데이트된 지식을 제공함으로써 LLM의 정확성과 일관성을 높일 수 있습니다. LLM은 KG의 데이터 정제, 지식 추출 및 KG 생성을 지원하며, 반대로 LLM은 KG의 쿼리, 분석 및 도메인 특정 애플리케이션에도 활용됩니다.

- **Performance Highlights**: 워크숍에서는 150명 이상의 참석자가 있었으며, 여러 대륙의 연구자들이 9개의 동료 심사 논문을 발표했습니다. 주요 논의에는 KG가 LLM을 위한 보강, LLM이 KG의 생성을 위한 도움, 그리고 LLM-KG 통합 데이터 관리 기회가 포함되었습니다.



### Enhancing Screen Time Identification in Children with a Multi-View Vision Language Model and Screen Time Tracker (https://arxiv.org/abs/2410.01966)
Comments:
          Prepare for submission

- **What's New**: 이 연구에서는 새로운 유형의 센서 정보학 프레임워크를 개발하여 어린이의 화면 노출을 정확하게 모니터링할 수 있는 방법을 제안합니다. 이 방법은 착용할 수 있는 센서에서 수집한 주관적인 이미지를 이용하고, 이를 통해 자동화된 정확한 데이터 수집이 가능합니다.

- **Technical Details**: 제안된 방법은 'screen time tracker (STT)'라는 착용형 카메라와 'vision language model (VLM)'를 결합하여 여러 화면 장치에서 화면 존재를 식별하는 시스템을 구축합니다. 특히, 다양한 각도에서 수집된 자기 중심 이미지를 기반으로 한 다중 뷰 VLM을 개발하여, 동적 화면 노출 해석을 가능하게 하였습니다.

- **Performance Highlights**: 연구 결과, 기존 방식에 비해 유의미한 성능 향상을 보여주었으며, 어린이의 자연스러운 환경에서의 화면 노출 연구를 최적화할 수 있는 가능성을 입증하였습니다.



### One-step Noisy Label Mitigation (https://arxiv.org/abs/2410.01944)
Comments:
          20 pages, 4 figures, 11 Tables

- **What's New**: 본 논문에서는 고차원 직교성(high-dimensional orthogonality)의 특성을 활용하여 깨끗한 샘플과 노이즈 샘플을 구분하기 위한 강력하고 효과적인 경계를 제안합니다. 모델 의존성이 없으며, 비용 효율적인 One-step Anti-Noise (OSA) 기법을 도입했습니다.

- **Technical Details**: One-step Anti-Noise (OSA)는 입력 쌍의 노이즈 수준을 단일 단계의 추론(inference)으로 평가하는 모델 비종속적(noise mitigation) 패러다임입니다. 이 과정에서 고차원 직교성을 고려한 스코어링 함수(score function)를 활용하여 각 샘플의 손실에 학습 가중치를 수동적으로 할당합니다.

- **Performance Highlights**: OSA의 실험 결과는 다양한 벤치마크, 모델 및 작업에서 훈련 강건성(training robustness)과 작업 전이성(task transferability)을 향상시키고, 배포의 용이성(ease of deployment)과 계산 비용(computational costs)의 절감을 보여줍니다.



### CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL (https://arxiv.org/abs/2410.01943)
- **What's New**: 이번 논문에서는 Text-to-SQL 작업을 위한 LLM의 성능 향상을 위해 새로운 프레임워크인 CHASE-SQL을 소개합니다. 이 프레임워크는 multi-agent modeling을 활용하여 후보 생성 및 선택을 개선하는 혁신적인 전략을 사용합니다.

- **Technical Details**: CHASE-SQL은 LLMs의 본질적인 지식을 활용하여 다양한 SQL 후보를 생성하는 방법론을 제안합니다. 주요 기술로는 (1) 복잡한 쿼리를 관리 가능한 하위 쿼리로 분해하는 divide-and-conquer 방법, (2) 쿼리 실행 계획을 기반으로 한 chain-of-thought reasoning 방법, (3) 인스턴스 인식된 합성 예제 생성 기술이 있습니다. 이를 통해 후보 쿼리를 생성하고, pairwise 비교를 통해 최고의 후보를 선택하는 과정을 거칩니다.

- **Performance Highlights**: CHASE-SQL은 BIRD Text-to-SQL 데이터셋의 테스트 세트와 개발 세트에서 각각 73.0%와 73.01%의 실행 정확도를 달성하며, 이전 방법들을 능가하는 성능을 보여주었습니다. 전체적으로, 이 방법론은 SQL 쿼리의 품질과 다양성을 높이는 데 기여하며, 기존의 방법들에 비해 우수한 결과를 확인할 수 있습니다.



### Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL (https://arxiv.org/abs/2410.01930)
- **What's New**: 이 논문은 강화학습(RL)에서 소프트 혼합 전문가(Soft MoE)의 효율성을 분석하고, 성능 향상의 이유가 다수의 전문가 사용이 아닌 인코더 출력의 토큰화에 있다는 놀라운 사실을 발견했다.

- **Technical Details**: 연구에 따르면, Soft MoE의 성능 향상은 인코더 출력의 토큰화(tokenization)에 크게 의존하며, 단일 전문가만으로도 유사한 성능을 유지할 수 있다. 따라서, MoE 사용 시 전문가의 활용도가 낮게 평가되고 있다는 점이 강조되었다.

- **Performance Highlights**: 이 발견은 강화학습 에이전트가 픽셀 기반 환경에서 훈련될 때, 다차원 출력을 평탄화(flattening)하는 기존의 일반적인 관행이 비효율적임을 시사한다. 연구 결과는 딥 RL 촬영에 있어 새로운 통찰력을 제공하며, 네트워크 활용도와 가소성을 개선하기 위한 최근 기술을 활용할 가능성도 모색하고 있다.



### Risk Alignment in Agentic AI Systems (https://arxiv.org/abs/2410.01927)
- **What's New**: 이번 논문에서는 Agentic AI (대리적 인공지능)의 리스크 프로파일 (risk profiles)을 어떻게 설계해야 하는지에 대한 중요성을 강조합니다. 사용자의 리스크 태도 (risk attitudes)에 맞게 이러한 시스템을 조정하는 방법과 AI가 자율적으로 작동할 때 발생할 수 있는 다양한 윤리적 문제에 대해 논의합니다.

- **Technical Details**: 리스크 얼라인먼트 (risk alignment)란 대리적 AI가 사용자와 올바르게 조화되도록 만드는데 필수적인 요소로 주장됩니다. 논문에서는 대리적 AIs의 두 가지 모델인 프록시 에이전트 (proxy agents)와 오프 더 셀프 툴 (off-the-shelf tools)을 제시하며, 각각의 경우 어떻게 리스크 태도를 설정해야 하는지를 논의합니다.

- **Performance Highlights**: AI의 리스크 태도는 사용자의 리스크 태도에 크게 영향을 받습니다. 논문에서 제시한 사례를 통해 보았을 때, 사용자가 리스크 수용성 (risk tolerance)이 높은 경우와 낮은 경우에 대리적 AI의 행동이 어떻게 달라질 수 있는지를 보여줍니다. 이는 결국 사용자와 AI 간의 신뢰도 (trust)에 직접적인 영향을 미칩니다.



### Provably Accurate Shapley Value Estimation via Leverage Score Sampling (https://arxiv.org/abs/2410.01917)
- **What's New**: 새로운 접근 방식으로 Leverage SHAP을 도입하였습니다. 이는 Kernel SHAP의 경량화된 수정판으로, O(n log n) 평가만으로도 Shapley 값의 정확한 추정을 가능하게 합니다.

- **Technical Details**: Leverage SHAP은 Shapley 값 추정과 비모수적(active learning) 학습 사이의 연관성을 활용하여, 러브리지 점수 샘플링(leverage score sampling) 기법을 사용하여 모델 평가 횟수를 크게 줄입니다. 기존 방법인 Kernel SHAP은 비모수적이며, 많은 모델에 대해 효과적인 성능을 보이는 반면, Leverage SHAP은 이보다 더 정확한 결과를 제공합니다.

- **Performance Highlights**: Leverage SHAP은 SHAP 라이브러리에서 사용 가능한 Kernel SHAP의 최적화된 구현보다도 일관되게 뛰어난 성능을 보였습니다. 실험 결과, Leverage SHAP은 더 적은 모델 평가로도 효과적인 Shapley 값을 제공하여 해석 가능성을 개선하는 데 기여합니다.



### A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation (https://arxiv.org/abs/2410.01912)
Comments:
          25 pages, 20 figures, code is open at this https URL

- **What's New**: 이번 연구는 벡터 양자화 (Vector Quantization, VQ) 기반의 자율 회귀 (Autoregressive) 이미지 생성을 위해 새로운 모델 아키텍처인 2차원 자율 회귀 (2-Dimensional Autoregression, DnD) Transformer를 도입하여 정보를 잃는 병목 현상을 해결합니다.

- **Technical Details**: DnD-Transformer는 모델 깊이 (model depth)라는 새로운 자율 회귀 방향을 도입하여 이미지에 대해 더 많은 코드를 예측하도록 설계되었습니다. 기존의 1차원 자율 회귀와 RQ-Transformer와 같은 2차원 이미지 분해 방식을 비교했을 때, DnD-Transformer는 엔드-투-엔드 모델로 동일한 백본 모델 크기와 시퀀스 길이로 더 높은 품질의 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과는 DnD-Transformer가 세밀한 이미지 세부 사항을 우수하게 재구성하며, 1D 방법과 비교해 더 효율적이고 낮은 엔트로피 분해를 보여주었습니다. 또한 ImageNet 256x256 생성에서 AR 기본 모델보다 크게 향상된 성능 (최대 1.54 FID 및 82.6 IS 개선)을 달성하였으며, 다중 모드 모델링에 대한 자율 회귀 모델의 뚜렷한 장점을 강조합니다.



### Social Media Authentication and Combating Deepfakes using Semi-fragile Invisible Image Watermarking (https://arxiv.org/abs/2410.01906)
Comments:
          ACM Transactions (Digital Threats: Research and Practice)

- **What's New**: 이 논문에서는 미디어 인증을 위한 새로운 반 취약(semifragile) 이미지 워터마킹 기법을 제안합니다. 이 방법은 얼굴 변조(facial manipulation) 또는 조작에 대해서는 취약하지만, 일반적인 이미지 처리 작업에는 강건성을 지니고 있습니다.

- **Technical Details**: 제안된 워터마킹 프레임워크는 비가시(invisible) 비밀 메시지를 실 이미지에 삽입하며, 비판자(critic) 및 적대적 네트워크(adversarial network)를 포함한 독특한 아키텍처를 통해 높은 이미지 품질과 워터마킹 제거에 대한 저항력을 보장합니다. 실험을 통해 다양한 Deepfake 데이터셋에서 워터마크의 복구 정확도가 높은 것으로 나타났습니다.

- **Performance Highlights**: 제안된 모델은 64비트 비밀 정보를 비가시적인 이미지 워터마크로 삽입할 수 있으며, 일반 이미지 처리 이후에도 높은 복구 정확성을 유지합니다. 반면에, Deepfake 변조에는 복구되지 않으며, 여러 화이트박스(white-box) 및 블랙박스(black-box) 워터마크 제거 공격에 대해 높은 저항성을 보입니다.



### The potential of LLM-generated reports in DevSecOps (https://arxiv.org/abs/2410.01899)
Comments:
          Published in AIESE 2024 (International Conference on AI empowered Software Engineering)

- **What's New**: 본 논문은 DevSecOps 패러다임을 사용하는 소프트웨어 팀에서의 경고 피로(alert fatigue) 문제를 다루고 있습니다. 특히 작은 팀에서는 자원의 한계로 인해 보안 경고의 수가 과도하게 많아지는 현상이 발생하며, 이로 인해 보안 경고에 대한 반응이 둔감해지는 문제를 언급하고 있습니다.

- **Technical Details**: LLM(대형 언어 모델)을 활용하여 개발된 보고서는 감지된 보안 문제의 재정적 영향(financial impact)과 결과를 강조합니다. 조사를 통해 LLM이 생성한 보고서가 명확하고 포괄적이며 동기를 부여하는 통찰력을 제공하여 보안 문제에 대한 즉각적인 행동 가능성을 높인다는 것을 발견하였습니다.

- **Performance Highlights**: 이 보고서를 DevSecOps 워크플로우에 통합함으로써 경고 피로를 완화하고, 중요한 보안 경고를 효과적으로 대응할 수 있도록 돕습니다.



### Auction-Based Regulation for Artificial Intelligenc (https://arxiv.org/abs/2410.01871)
Comments:
          20 pages, 7 figures

- **What's New**: 본 논문은 인공지능(AI)의 안전성 및 편향, 법적 문제를 해결하기 위해 현대적인 수학적 프레임워크가 필요하다는 점을 강조하며, 모델 구축 에이전트가 안전한 모델을 개발하고 규제 과정에 참여하도록 유도하는 경매 기반의 규제 메커니즘을 제안합니다.

- **Technical Details**: 제안된 규제 메커니즘은 Safety-Incentivized Regulatory Auction(Sira)으로, 안전 기준을 충족하지 못한 모델은 배제하고, 더 안전한 모델을 제출하는 에이전트에게 보상을 제공하는 구조입니다. 이 메커니즘은 Nash Equilibria를 사용하여 에이전트가 안전 기준을 초과하는 모델을 제출하도록 유도합니다.

- **Performance Highlights**: 실증 결과에 따르면, Sira는 모델의 안전성을 20% 증가시키고 참여율을 15% 향상시켜, 최소 안전 기준을 단순히 시행하는 기존의 규제 프레임워크보다 우수한 성능을 보였습니다.



### Enhancing LLM Fine-tuning for Text-to-SQLs by SQL Quality Measuremen (https://arxiv.org/abs/2410.01869)
- **What's New**: 본 연구에서는 SQL 품질 측정을 통해 LLM 기반 Text-to-SQL의 성능을 향상시키는 새로운 접근 방식을 제안합니다. 이 방법은 복잡한 데이터 전처리 없이도 모델의 출력을 개선할 수 있는 자동 피드백 메커니즘을 통합하고 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소를 포함합니다: 1) 인간이 설계한 단계별 프롬프트를 통해 LLM에 정보를 제공하고 2) 생성된 SQL 문을 기대 결과와 비교하여 피드백을 제공하는 SQL 품질 측정. 이 시스템은 BIRD 벤치마크에서 실행 정확도(Execution Accuracy) 및 유효 효율성 점수(Valid Efficiency Score)를 평가하여 종합적으로 검증됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최신 모델인 GPT4 및 T5와 비교하여 실행 정확도(EX)와 유효 효율성 점수(VES) 모두에서 경쟁력 있는 성능을 보였습니다.



### House of Cards: Massive Weights in LLMs (https://arxiv.org/abs/2410.01866)
Comments:
          Under review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)에서 발생하는 massive activations의 기원을 밝히고, 이를 해결하기 위한 새로운 접근법인 MacDrop을 제안합니다. 이 연구는 massive activations가 hidden state가 아니라 초기 layer의 feed-forward network의 intermediate state에서 기인함을 밝혔습니다.

- **Technical Details**: 논문에서는 'top-k massive weights'라는 개념을 도입하여, 특정 feature dimension에서 top-k 크기를 가진 weights를 정의합니다. 이 massive weights가 0으로 설정될 경우 LLM의 기능성이 완전히 망가지는 반면, 모든 다른 weights를 0으로 설정해도 성능 저하가 미미합니다. 이러한 관찰을 바탕으로 LLM의 효과적인 fine-tuning을 위한 기법인 MacDrop을 제안하며, 이는 massive weights에 dropout을 적용합니다.

- **Performance Highlights**: MacDrop을 통해 제안된 방식이 zero-shot downstream tasks와 generation tasks에서 전반적으로 성능을 향상시키고 있음을 보여줍니다.



### Simplifying complex machine learning by linearly separable network embedding spaces (https://arxiv.org/abs/2410.01865)
Comments:
          26 pages, 8 figures

- **What's New**: 이번 연구에서는 네트워크 데이터를 더 선형적으로 분리 가능한 공간으로 매핑하는 새로운 graphlet 기반 방법을 소개하고, 이를 통해 복잡한 네트워크 데이터의 효율적이고 설명 가능한 마이닝을 가능하게 합니다.

- **Technical Details**: 그래프 신경망(GNN) 및 random walk 기법을 활용한 네트워크 임베딩의 기존 방법론에 대한 통찰을 바탕으로, graphlet 기반의 랜덤 워크 행렬 표현을 사용하여 노드의 네트워크 이웃 유사성과 위상 유사성을 모두 고려한 임베딩을 생성합니다. 이는 Orthogonal NMTF(ONMTF) 프레임워크를 통해 이루어집니다.

- **Performance Highlights**: 1) 13개의 서로 다른 네트워크에서 9개의 임베딩 공간이 선형 분류기를 사용하여 우수한 노드 분류 F1 점수를 기록했습니다. 2) 세 개의 네트워크에서는 임베딩 공간이 완전히 선형적으로 분리 가능하며, 복잡한 비선형 기계 학습 방법과 통계적으로 유의미한 차이가 없음을 보여주었습니다. 3) 비선형 분류기가 더 나은 성능을 보이는 경우에도, 새로운 graphlet 기반의 임베딩이 기존의 random-walk 기반 방법보다 최소 8% 높은 F1 점수를 기록했습니다.



### Explainable Diagnosis Prediction through Neuro-Symbolic Integration (https://arxiv.org/abs/2410.01855)
- **What's New**: 이 연구에서는 진단 예측을 위한 설명 가능한 모델을 개발하기 위해 신경-기호적 방법인 Logical Neural Networks (LNNs)를 활용했습니다. 전통적인 기계 학습 모델들이 가지는 해석 가능성의 부족 문제를 해결하고자, 도메인 특화 엔티티로 구성된 논리 규칙과 학습 가능한 임계값을 결합한 모델을 디자인하였습니다.

- **Technical Details**: LNN 기반 모델(M_{multi-pathway} 및 M_{comprehensive})을 통해 로지스틱 회귀, SVM, 랜덤 포레스트와 같은 전통적인 모델보다 뛰어난 성능을 발휘했습니다. 이 모델들은 데이터에 따라 논리 규칙을 조정할 수 있는 학습 가능한 매개변수를 통합하여 복잡한 의료 정보를 효과적으로 모델링할 수 있습니다.

- **Performance Highlights**: 당 연구의 LNN 모델은 당뇨병의 사례 연구에서 최대 80.52%의 정확도 및 0.8457의 AUROC 점수를 달성하여, 가장 좋은 전통적인 모델인 랜덤 포레스트의 정확도 76.95%를 초월하였습니다. 이로써 신경-기호적 접근 방식의 유효성을 입증하였으며, 투명하고 적응 가능한 진단 모델 개발의 기초를 마련했습니다.



### Bayes-CATSI: A variational Bayesian approach for medical time series data imputation (https://arxiv.org/abs/2410.01847)
- **What's New**: 본 논문에서는 Bayesian Context-Aware Time Series Imputation (Bayes-CATSI) 프레임워크를 제안하여 기존의 CATSI 모델에서 불확실성 정량화를 통합하였습니다. 이 접근법은 의료 데이터의 데이터 임퓨테이션 성능을 개선합니다.

- **Technical Details**: Bayes-CATSI는 Variational Inference를 활용하여 posterior distribution의 형태를 가정하고, Kullback-Leibler(KL) divergence를 최소화하여 실제 posterior distribution에 가장 근접한 variational densities를 찾습니다. 이 모델은 electroencephalography (EEG), electrooculography (EOG), electromyography (EMG), electrocardiology (EKG)와 같은 의료 타임 시리즈 데이터를 고려합니다.

- **Performance Highlights**: Bayes-CATSI 모델은 기존의 CATSI 모델에 비해 9.57% 향상된 성능을 보여주었으며, 불확실성 정량화를 통해 보다 신뢰성 있는 데이터 임퓨테이션을 제공합니다.



### A GEN AI Framework for Medical Note Generation (https://arxiv.org/abs/2410.01841)
Comments:
          8 Figures, 7 page, IEEE standard research paper

- **What's New**: MediNotes라는 혁신적인 생성 AI 프레임워크가 도입되어, 의학적 대화로부터 SOAP (Subjective, Objective, Assessment, Plan) 노트를 자동으로 생성합니다.

- **Technical Details**: MediNotes는 Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Automatic Speech Recognition (ASR) 기술을 통합하여 실시간 또는 녹음된 오디오에서 텍스트와 음성 입력을 포착하고 처리합니다. 또한 Quantized Low-Rank Adaptation (QLoRA)와 Parameter-Efficient Fine-Tuning (PEFT) 기법을 활용하여 자원이 제한된 환경에서도 효과적인 모델 미세조정을 지원합니다.

- **Performance Highlights**: ACI-BENCH 데이터셋을 이용한 평가 결과, MediNotes는 자동 의료 문서화의 정확성, 효율성, 사용성을 획기적으로 개선하며, 의료 전문가의 행정 부담을 줄이고 임상 작업의 질을 향상시키는 효과적인 솔루션임을 입증했습니다.



### Target Pose Guided Whole-body Grasping Motion Generation for Digital Humans (https://arxiv.org/abs/2410.01840)
Comments:
          7 pages,5 figures

- **What's New**: 본 연구에서는 가상의 인간형 지능체를 위한 그립(grasping) 모션 생성 프레임워크를 제안합니다. 이 프레임워크는 전체 신체의 다양한 포즈를 고려하여 더 자연스러운 그립 모션을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구는 3D 공간에서 알려진 초기 포즈를 기반으로 대상으로 하는 포즈를 생성하고, 이를 바탕으로 transformer 기반의 신경망(neural network)을 활용하여 초기 포즈와 목표 포즈(target pose)를 부드럽고 자연스럽게 연결하는 전체 그립 궤적(grasping trajectory)을 생성합니다. 또한, 두 개의 후처리 최적화(component) 기법을 통해 발 스케이팅(foot-skating) 문제와 손-물체 간의 침투(interpenetration) 문제를 각각 완화합니다.

- **Performance Highlights**: GRAB 데이터셋을 사용한 실험에서 제안된 방법은 무작위로 배치된 알려지지 않은 물체에 대해 전체 신체 그립 모션 생성을 효과적으로 수행함을 보여줍니다.



### Temporal Graph Memory Networks For Knowledge Tracing (https://arxiv.org/abs/2410.01836)
- **What's New**: 이 논문은 학생의 지식 수준 변화를 추적하는 새로운 방법인 Temporal Graph Memory Network (TGMN)을 제안합니다. 이 모델은 관계적(dynamic) 및 시간적(temporal) 동향을 공동으로 모델링하여 보다 향상된 학습 경험을 제공합니다.

- **Technical Details**: TGMN은 키-값 메모리(graph memory)를 활용하여 지식 구성 요소(KCs) 간의 관계를 포착합니다. 또한 학생의 잊기 행동을 표현하기 위해 시간적 기억 감소 기법을 채택했습니다. 기존의 수작업 특성을 사용하지 않고도 학생의 지식 변화를 추적할 수 있는 방법을 제안합니다.

- **Performance Highlights**: 논문에서는 여러 지식 추적(KT) 벤치마크와 비교하여 TGMN의 효과성을 입증하였으며, 최신 KT 모델들과의 비교 분석을 통해 성능 개선을 확인했습니다.



### Analysis of Convolutional Neural Network-based Image Classifications: A Multi-Featured Application for Rice Leaf Disease Prediction and Recommendations for Farmers (https://arxiv.org/abs/2410.01827)
- **What's New**: 이번 연구에서는 8개의 서로 다른 convolutional neural network (CNN) 알고리즘을 사용하여 쌀 병해 분류를 개선하는 새로운 방법을 제시합니다. 이로 인해 정밀 농업(precision agriculture) 분야가 한층 발전할 것으로 기대됩니다.

- **Technical Details**: Tkinter 기반의 애플리케이션은 농부들에게 다양한 기능을 제공하는 사용자 친화적인 인터페이스를 제공합니다. 이 애플리케이션을 통해 농부들은 실시간 병해 예측 및 개인화된 추천을 통해 시기적절하고 정보에 기반한 결정을 내릴 수 있습니다. ResNet-50, InceptionV3, VGG16, MobileNetV2와 같은 최신 CNN 전이 학습(transfer learning) 알고리즘이 UCI 데이터셋과 통합되어 현대 농업 관행의 개선을 가져옵니다.

- **Performance Highlights**: 결과적으로 ResNet-50은 75%, DenseNet121은 90%, VGG16은 84%, MobileNetV2는 95.83%, DenseNet169은 91.61%, InceptionV3는 86%의 정확도를 기록하였습니다. 그러나 VGG19는 70%, Nasnet는 80.02%로 과적합(overfitting)을 보였고, ResNet101에서는 54%의 정확도, EfficientNetB0에서는 33%에 불과했습니다. MobileNetV2로 훈련된 모델은 Tkinter GUI 애플리케이션에 성공적으로 배치되어 이미지 또는 실시간 비디오 캡처를 통해 예측을 수행하였습니다.



### AI Conversational Interviewing: Transforming Surveys with LLMs as Adaptive Interviewers (https://arxiv.org/abs/2410.01824)
- **What's New**: 이번 연구는 대화형 인터뷰를 통해 사람들의 의견을 수집하는 전통적인 방법을 대체할 수 있는 가능성을 탐색했습니다. 특히, Large Language Models (LLMs)을 활용하여 인적 자원 없이 스케일이 큰 대화형 인터뷰를 수행할 수 있는 방법을 제시합니다.

- **Technical Details**: 대학 학생들을 대상으로 한 소규모 연구에서 AI와 인간 면접자가 동일한 정치적 주제의 질문지를 사용하여 인터뷰를 진행했습니다. 연구는 AI Conversational Interviewing의 성능을 평가하고, 면접 프로토콜 준수, 응답 품질 및 참여자 참여도와 같은 다양한 양적 및 질적 지표를 분석했습니다.

- **Performance Highlights**: AI Conversational Interviewing은 전통적인 방법에 비해 동등한 질의 데이터를 생산할 수 있음을 확인했습니다. 연구 결과는 AI의 효과적인 구현을 위한 구체적인 추천 사항도 제공합니다.



### The Importance of Causality in Decision Making: A Perspective on Recommender Systems (https://arxiv.org/abs/2410.01822)
Comments:
          Accepted at the CONSEQUENCES '24 workshop, co-located with ACM RecSys '24

- **What's New**: 추천 시스템(Recommendation Systems) 분야에서 인과관계(causality)의 중요성이 커지고 있으며, 이는 정확한 예측(predictions)을 효과적이고 설명 가능한 결정(decisions)으로 전환하기 위해 도움이 될 수 있다는 점을 강조하고 있습니다.

- **Technical Details**: 이 논문에서는 추천 시스템 문제를 인과관계 관점에서 공식화하며, 잠재 결과(Potential Outcomes, POs)와 구조적 인과 모델(structural causal models)을 활용하여 추정해야 할 인과량(causal quantities)의 정의를 제공합니다. 일반적인 인과 그래프(causal graph)도 제시하여 향후 연구와 개발을 촉진합니다.

- **Performance Highlights**: 제시된 인과 결정-making 프레임워크를 통해 추천 충실도를 높이고, 특정 추천 시스템 문제에 대한 인과 그래프의 구축 방법을 제시하여 실제 시나리오에서 효과적인 추천이 가능하게 합니다.



### NFDIcore 2.0: A BFO-Compliant Ontology for Multi-Domain Research Infrastructures (https://arxiv.org/abs/2410.01821)
- **What's New**: NFDIcore 2.0는 독일의 국가 연구 데이터 인프라(NFDI)를 위한 다양한 연구 커뮤니티를 나타내는 Ontology로, Basic Formal Ontology(BFO)와 호환됩니다. 이 새로운 버전은 각 연구 도메인의 특정 요구를 다루기 위해 모듈화된 설계를 채택하였으며, SWRL 규칙을 사용하여 지식 발견을 효율적으로 지원합니다.

- **Technical Details**: NFDIcore는 NFDI의 여러 컨소시엄 간의 메타데이터 상호 운용성을 보장하며, 이를 통해 연구 도메인 간의 협력을 촉진합니다. Ontology의 설계 원칙과 구성 요소는 복잡한 관계를 표명하고 사용자 접근성을 충족시키는 방식으로 균형을 이룹니다.

- **Performance Highlights**: NFDIcore는 다양한 연구 분야를 위해 특정 요구 사항에 맞게 확장할 수 있는 모듈형 구조를 가지고 있으며, 이를 통해 연구 데이터의 발견과 재사용의 가능성을 극대화합니다. 실제 사용 사례에 기반한 검증을 통해 실질적인 유용성을 입증하였습니다.



### PixelBytes: Catching Unified Representation for Multimodal Generation (https://arxiv.org/abs/2410.01820)
- **What's New**: 이번 보고서는 PixelBytes라는 새로운 다중 모달 (multimodal) 표현 학습 접근 방식을 소개합니다. 이 방법은 텍스트, 오디오 및 픽셀화된 이미지 (sprites)를 포함한 다양한 입력을 포괄적으로 담아내기 위해 여러 기존의 시퀀스 모델을 참고하여 개발되었습니다.

- **Technical Details**: PixelBytes는 Recurrent Neural Networks (RNNs), State Space Models (SSMs), Attention 기반 모델 등 여러 아키텍처를 실험했습니다. 특히, bidirectional 처리 및 convolutional PxBy 임베딩 기술에 중점을 두었습니다. 장기 기억 (Long Short-Term Memory, LSTM) 네트워크를 예측 (predictive) 및 자기 회귀 (autoregressive) 모드에서 평가하여 모델의 성능을 비교했습니다.

- **Performance Highlights**: 실험 결과, 자기 회귀 모델이 예측 모델에 비해 더 우수한 성능을 보였습니다. PixelBytes는 다중 모달 데이터 이해 및 생성이 가능한 기초 모델의 개발에 기여하고 있습니다.



### Strategic AI Governance: Insights from Leading Nations (https://arxiv.org/abs/2410.01819)
Comments:
          21 pages, 3 Figures, 5 Tables

- **What's New**: 이 논문은 인공지능(AI)의 채택을 방해하는 데이터 프라이버시, 보안, AI 기능 이해와 관련된 문제들을 다루며, 다양한 국가의 AI 전략을 검토하여 AI 거버넌스 접근법, 전략적 주제 및 AI 채택을 위한 촉진 요인과 도전 과제를 종합합니다.

- **Technical Details**: 중요한 기여는 EPIC(Education, Partnership, Infrastructure, Community) 프레임워크의 개발로, 이는 AI의 사회적 영향을 완전히 실현하고 공공의 이익을 위해 성공적이고 지속 가능한 AI 배치를 위한 요구사항을 매핑합니다. 다각적인 관점의 콘텐츠 분석을 통해 최신 AI 전략 문서들을 구조적으로 비교합니다.

- **Performance Highlights**: 이 연구의 결과는 정부, 학계, 산업 및 커뮤니티가 책임감 있고 신뢰할 수 있는 AI 배치를 가능하게 하는 귀중한 통찰력을 제공합니다. 향후 연구는 개발도상국을 위한 특정 요구사항을 통합하고, 특정 AI 애플리케이션, 산업 및 공공 부문에 전략을 적용하는 데 중점을 두어야 합니다.



### Integrating AI's Carbon Footprint into Risk Management Frameworks: Strategies and Tools for Sustainable Compliance in Banking Sector (https://arxiv.org/abs/2410.01818)
- **What's New**: 이 논문은 AI의 탄소 발자국을 은행 부문의 위험 관리 프레임워크(RMFs)에 통합하는 방법을 다루고 있으며, 지속 가능성 목표 및 규제 요구 사항과의 일치를 강조합니다.

- **Technical Details**: AI의 탄소 발자국은 주로 에너지 집약적인 프로세스에서 기인하며, 은행에서 AI 모델의 교육 및 운영에 사용되는 고성능 하드웨어가 높은 에너지 소비를 초래합니다. Open Mixture-of-Experts(OLMoE) 및 Agentic RAG와 같은 최신 AI 연구는 모델의 효율성을 높이면서 탄소 발자국을 줄이는 방향으로 발전하고 있습니다.

- **Performance Highlights**: AI의 탄소 발자국을 RMF에 통합함으로써 은행은 ESG(환경, 사회, 거버넌스) 기준을 준수하고, 시스템의 성능 및 책임 있는 AI 사용을 보여주어 규제를 준수할 수 있습니다.



### From Experts to the Public: Governing Multimodal Language Models in Politically Sensitive Video Analysis (https://arxiv.org/abs/2410.01817)
- **What's New**: 이 논문은 정치적으로 민감한 비디오에 대한 다중모달 대형 언어 모델(MM-LLMs)의 거버넌스를 개인 및 집합적 논의를 통해 분석합니다. 연구는 전문가 비디오 해석에 대한 이해를 말하는 기자와의 인터뷰를 통해 시작되었고, 일반 대중이 민주적 의사 결정 메커니즘을 통해 논의에 참여했습니다.

- **Technical Details**: 이 연구의 114명의 일반 대중을 대상으로 한 실험은 전문가와 대중 간의 비디오 해석에서의 감정, 객관성, 사실적 명확성 등의 차이를 보여주었습니다. 또한, 다양한 거버넌스 메커니즘(예: quadratic voting, weighted ranking)을 통해 AI의 행동에 대한 사용자 결정에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 결과적으로, quadratic voting은 리버럴 민주주의와 정치적 평등에 대한 인식을 높였으며, AI에 대한 긍정적인 시각을 가진 참가자들은 이 과정이 높은 참여 민주주의를 마케팅하는 것으로 인식했습니다. 이러한 발견은 DAO 메커니즘을 통해 AI 거버넌스의 민주화 가능성을 제시합니다.



### Automatic Scene Generation: State-of-the-Art Techniques, Models, Datasets, Challenges, and Future Prospects (https://arxiv.org/abs/2410.01816)
Comments:
          59 pages, 16 figures, 3 tables, 36 equations, 348 references

- **What's New**: 이 논문은 자동 장면 생성(automatic scene generation) 분야의 최신 연구 동향을 포괄적으로 검토하고, 기계 학습(machine learning), 심층 학습(deep learning), 임베디드 시스템(embedded systems), 자연어 처리(NLP) 등의 기술을 활용한 다양한 생성 기법을 조망합니다.

- **Technical Details**: 연구는 VARIATIONAL AUTOENCODERS (VAEs), GENERATIVE ADVERSARIAL NETWORKS (GANs), TRANSFORMERS 및 DIFFUSION MODELS의 네 가지 기본 모델로 분류되어 각 모델의 하위 모델과 그 기여를 상세히 탐구합니다. 또한, COCO-Stuff, Visual Genome 및 MS-COCO와 같은 데이터셋을 살펴보아 이러한 모델을 훈련하고 평가하는 데 필수적입니다.  장면 생성 방법론에는 이미지-3D 변환(image-to-3D conversion), 텍스트-3D 생성(text-to-3D generation), UI/레이아웃 디자인(ui/layout design), 그래프 기반 방법(graph-based methods), 대화형 장면 생성(interactive scene generation) 등이 포함됩니다.

- **Performance Highlights**: 모델 성능 평가는 FRECHET INCEPTION DISTANCE (FID), KULLBACK-LEIBLER (KL) DIVERGENCE, INCEPTION SCORE (IS), INTERSECTION OVER UNION (IoU), MEAN AVERAGE PRECISION (mAP) 등의 평가 지표를 활용합니다. 연구 결과는 현실성 유지, 다중 객체를 포함한 복잡한 장면 처리 등 여러 도전 과제를 다루고 있으며, 기존의 성과와 비교해 최신 발전을 요약하여 자동 장면 생성 분야에서의 연구자와 실무자에게 유용한 자료를 제공합니다.



### AI in Food Marketing from Personalized Recommendations to Predictive Analytics: Comparing Traditional Advertising Techniques with AI-Driven Strategies (https://arxiv.org/abs/2410.01815)
- **What's New**: 이 논문은 전통적인 광고 방법에서 AI 기반 전략으로의 전환을 다루고 있습니다. AI는 개인 맞춤형 추천, 소비자 행동 예측 및 캠페인 최적화에 혁신을 가져왔습니다.

- **Technical Details**: AI는 소비자의 구매 이력, 브라우징 행동 및 소셜 미디어 활동 데이터를 활용하여 매우 맞춤화된 마케팅 캠페인을 생성합니다. 이러한 전략들은 더 정확한 제품 추천과 소비자 요구 예측을 가능하게 하여 고객 만족도 및 사용자 경험을 향상시킵니다.

- **Performance Highlights**: AI는 노동 집약적인 프로세스를 자동화하여 효율성과 비용 절감으로 이어집니다. 마케팅 메시지를 지속적으로 적응시키는 능력은 시간이 지나도 관련성과 흥미를 유지할 수 있도록 합니다. AI의 개인화 및 효율성에서의 이점에도 불구하고, 기술 및 숙련된 전문가에 대한 상당한 투자가 필요하다는 도전이 따릅니다.



### Privacy-Preserving SAM Quantization for Efficient Edge Intelligence in Healthcar (https://arxiv.org/abs/2410.01813)
- **What's New**: 이번 논문에서는 데이터 없이 양자화(quantization) 매개변수를 학습할 수 있는 DFQ-SAM(데이터 없는 양자화 모델)을 제안합니다. 이 모델은 의료 데이터 프라이버시를 보장하면서 효율적인 모델 압축을 지원합니다.

- **Technical Details**: DFQ-SAM은 패치 유사성(patch similarity)과 진화하는 의사 양성 레이블(pseudo-positive labels) 생성을 결합하여 세그멘테이션을 위한 데이터 합성을 제공합니다. 또한, 저비트 양자화를 위한 정확도를 보장하기 위해 스케일 재파라미터화(scale reparameterization) 기법을 도입했습니다. 정확한 양자화 보정을 위한 이상적인 구성 요소들을 통합하여, 여러 데이터셋에서 강력한 일반화 성능을 보여주었습니다.

- **Performance Highlights**: DFQ-SAM은 4비트 양자화 시 모델 크기를 8배 줄이고 계산 복잡성을 64배 감소시키면서 AbdomenCT1K 데이터세트에서 정확도가 2.01%만 감소하는 성능을 보였습니다. 이 기술은 클라우드-엣지 협업에서 데이터 전송을 필요 없애고, 민감한 데이터를 외부 공격으로부터 보호합니다.



### From Text to Multimodality: Exploring the Evolution and Impact of Large Language Models in Medical Practic (https://arxiv.org/abs/2410.01812)
Comments:
          12 pages, 1 figure

- **What's New**: 이번 논문은 대규모 언어 모델(Large Language Models, LLMs)이 다중 모드 플랫폼(Multimodal Platforms)으로 발전함에 따라 의료 분야에 미치는 영향을 포괄적으로 검토합니다. 다중 모드 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 발전과 현재 의료 실무에서의 응용 가능성을 탐구합니다.

- **Technical Details**: MLLMs는 텍스트, 이미지, 오디오 등 다양한 데이터 유형을 통합하여 보다 포괄적인 건강 데이터를 분석할 수 있는 기능을 가지고 있습니다. 연구는 REALM 및 MedDr와 같은 모델을 통해 비구조화 데이터와 구조화 데이터 간의 간극을 해소할 수 있는 능력을 강조합니다.

- **Performance Highlights**: MLLM은 임상 예측, 환자 참여 증대 및 개인화된 치료 계획 개선을 통해 의료 분야에서 환자 건강에 대한 포괄적인 이해를 가능하게 합니다. 예를 들어, LlaVA-Rad 모델은 표준 방사선 작업에서 최첨단 성과를 달성했으며, MedAide와 같은 챗봇은 자원이 부족한 지역에서도 의료 지원을 제공합니다.



### Evaluating Cultural Awareness of LLMs for Yoruba, Malayalam, and English (https://arxiv.org/abs/2410.01811)
Comments:
          19 pages, 10 figures, 6 tables

- **What's New**: 이 논문은 대규모 언어 모델 (LLM)이 지역 언어인 말라얄람(Malayalam)과 요루바(Yoruba)의 문화적 측면을 이해하는 능력을 탐구합니다. 또한, Hofstede의 6가지 문화적 차원을 사용하여 LLM 기반 응답의 문화적 인식을 정량화합니다.

- **Technical Details**: 연구에서 사용된 Hofstede의 6가지 문화적 차원은 권력 거리(Power Distance, PDI), 개인주의(Individualism, IDV), 성취 및 성공에 대한 동기(Masculinity vs. Femininity, MAS), 불확실성 회피(Uncertainty Avoidance, UAI), 장기 지향(Long Term Orientation, LTO), 그리고 쾌락주의와 절제(Indulgence vs. Restraint, IVR)입니다. LLM이 영어에서는 높은 문화적 유사성을 보이지만, 말라얄람과 요루바의 문화적 뉘앙스를 포착하지 못한다는 사실을 입증합니다.

- **Performance Highlights**: 대규모 지역 언어 LLM 훈련을 위한 문화적으로 풍부한 데이터셋이 필요하다는 점을 강조하며, 이는 채팅 기반 LLM의 사용자 경험 향상 및 대규모 LLM 에이전트 기반 시장 조사 유효성 개선에 큰 영향을 미칠 것입니다.



### Propaganda is all you need (https://arxiv.org/abs/2410.01810)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 정치적 측면과 정렬 프로세스의 정치적 차원에 대한 심층 분석을 다룹니다. 과거 연구들이 AI의 정치적 성향에 대한 사례들을 다루어 왔지만, 본 논문은 LLM의 이념적 편향을 이해하기 위한 새로운 접근법을 제공합니다.

- **Technical Details**: 정렬(Alignment) 과정은 LLM의 행동을 특정 정치적 관점과 일치하도록 조정하는 것을 말합니다. 이 연구에서는 순수 텍스트 데이터로 정치를 편향된 LLM을 학습시켜 특정 이념적 편향을 내재화하는 방법을 다룹니다. 또한, 정치적 성향을 평가하기 위한 혼합 접근법을 사용하여 다양한 평가 에이전트의 상대적 편향을 줄이는 방법을 제안합니다.

- **Performance Highlights**: 연구에 따르면, LLM의 임베딩 공간 내의 단어 간의 상대적 위치는 정치적 교육 데이터에 따라 달라지며, Trotskyist와 Marxist-Leninist 데이터로 훈련한 모델 간의 '공산주의'와 '스탈린'의 거리 차이가 관찰되었습니다. 이러한 결과는 AI의 정치적 경향성과 그 사회적 영향에 대한 깊은 통찰을 제공합니다.



### Enhancing transparency in AI-powered customer engagemen (https://arxiv.org/abs/2410.01809)
- **What's New**: 본 논문은 AI 기반 고객 참여에서 소비자의 신뢰를 구축하는 데 있어 투명성과 책임의 필요성을 강조합니다. AI의 잠재력이 비즈니스 운영을 혁신하고 고객 경험을 향상시킬 수 있지만, AI 의사 결정 과정의 불투명성으로 인해 신뢰 구축에 장애물이 되고 있습니다.

- **Technical Details**: 조사에 따르면 소비자는 AI와의 상호작용에 대해 막연한 인식 부족과 AI 알고리즘의 편향(bias) 및 공정성(fairness)에 대한 우려가 큽니다. 논문은 소비자와 조직 리더 모두가 이해할 수 있는 설명 가능한 AI 모델의 개발을 지지하며, 이는 잠재적인 편향을 완화하고 윤리적 사용을 보장하는 데 이바지합니다.

- **Performance Highlights**: 조직이 단순한 규제 준수를 넘어 투명성 관행에 전념하는 것이 중요합니다. 명확한 데이터 정책을 우선시하고 이해관계자와 활발히 소통하여 책임 문화를 조성하는 것 또한 강조됩니다. 이러려고 박람회적으로 접근함으로써 기업들은 AI 기술에 대한 신뢰를 형성하고 기술 혁신과 소비자 수용 간의 격차를 해소할 수 있습니다.



### AI Horizon Scanning, White Paper p3395, IEEE-SA. Part I: Areas of Attention (https://arxiv.org/abs/2410.01808)
Comments:
          This is an interim version of our p3395 working group White Paper. We will update this version, until publication by the Institute of Electrical and Electronics Engineers, Standards Association (IEEE-SA), Sponsor Committee - Artificial Intelligence Standards Committee (C/AISC); this https URL

- **What's New**: 이번 백서 시리즈는 Generative Artificial Intelligence (AI) 모델의 사회적 변화가 가져오는 기회와 위험 간의 미세한 균형을 필요로 한다는 내용을 다룹니다. 특히, AI에 대한 표준 개발을 지원하기 위한 첫 번째 작업이며, AI 모델의 안전 장치와 방지 기술의 구현을 위한 IEEE-SA p3995 표준 개발에 관한 정보를 제공합니다.

- **Technical Details**: AI 모델의 광범위한 적용과 그에 따르는 책임, 개인 정보 보호, 데이터 권리 및 오용( misuse)에 대한 규제를 검토합니다. 글로벌 인프라의 안정성과 클라우드 컴퓨팅에 대한 과도한 의존에 대한 우려도 포함되어 있으며, 이는 복잡하게 연결된 AI 구성요소에서 나타날 수 있습니다.

- **Performance Highlights**: AI 기술의 잠재력은 교육, 연구 및 예술 분야에서 크게 향상될 수 있지만, 잘못된 정보 생성 및 성별, 인종 또는 민족에 대한 편향 출력과 같은 중요한 위험을 동반합니다. 이러한 문제들은 특히 이전의 Crowdstrike 사건과 같은 AI로 발생한 사건들이 중요한 인프라에 미치는 영향을 통해 잘 나타납니다.




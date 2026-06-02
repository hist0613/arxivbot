New uploads on arXiv(cs.CL)

### Artificial Hippocampus Networks for Efficient Long-Context Modeling (https://arxiv.org/abs/2510.07318)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 인공 신경망의 메모리 프레임워크를 제안하고, Multi-Store Model에서 영감을 받아 Artificial Hippocampus Network (AHN)를 도입하였다. AHN은 Transformer의 KV 캐시를 손실 없는 단기 메모리로 유지하며, 슬라이딩 윈도우 밖의 정보를 압축하여 고정 크기의 장기 메모리로 변환한다. 이 방법은 최신 RNN 유사 구조를 이용하여 AHNs를 구현하고, 오랜 컨텍스트 벤치마크에서 개선된 성과를 보여주는 실험 결과를 제시한다.

- **Technical Details**: AHN의 구조는 Mamba2, DeltaNet, Gated DeltaNet와 같은 RNN 유사 아키텍처로 인스턴스화되며, 이들 모델은 손실 없는 단기 메모리를 슬라이딩 윈도우로 유지한다. 정보가 윈도우를 넘어갈 경우, AHN 모듈이 이를 고정 크기로 압축하는 방식으로 작동한다. 이로 인해 AHN을 적용한 모델이 슬라이딩 윈도우 및 전체 주의(attention) 모델들을 능가하고, 메모리 및 계산 비용을 현저히 줄인다.

- **Performance Highlights**: 실험에서는 AHN을 적용한 Qwen2.5-3B-Instruct 모델이 40.5%의 플롭(FLOPs) 감소와 74.0%의 메모리 캐시 감소를 달성했으며, 평균 점수가 4.41에서 5.88로 향상되었다. 이러한 결과는 AHN을 통한 메모리 효율성을 극대화하고, 긴 시퀀스 처리에서 경쟁력 있는 성능을 발휘함을 보여준다. 논문은 AHN의 변형 모델을 개발하기 위한 코드와 모델을 배포할 예정이다.



### Vibe Checker: Aligning Code Evaluation with Human Preferenc (https://arxiv.org/abs/2510.07315)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)을 활용하여 코드 생성을 위해 자연어 상호작용을 사용하는 'vibe coding'을 소개합니다. Vibe check은 코드의 기능성뿐만 아니라, 솔루션이 어떤 느낌을 주어야 하고, 읽기 쉽고, 사용자의 의도를 유지해야 함을 강조합니다. 이 연구는 기능적 정확성을 넘어서는 지침 준수(instruction following)가 vibe check의 핵심 요소임을 가정합니다.

- **Technical Details**: 우리는 VeriCode라는 30개의 검증 가능한 코드 지침의 분류법을 제시하며, 각 지침에 해당하는 결정론적 검증기를 제공합니다. 이 분류법은 기존의 평가 도구를 보완하여, 코드의 지침 준수와 기능적 정확성을 동시에 평가하는 Vibe Checker라는 테스트베드를 구성합니다. 이를 통해 31개의 주요 LLM을 평가하며, 강력한 모델조차도 여러 지침 준수에서 어려움을 겪는다는 것을 확인했습니다.

- **Performance Highlights**: 기능적 정확성과 지침 준수의 복합 점수가 인간의 선호와 가장 잘 연관됨을 보여주며, 실제 프로그래밍 작업에서 지침 준수가 주요 구분 요소로 부각됩니다. 이 연구는 코드 작업에서 사용자의 선호도에 더 잘 맞춰진 모델을 개발하기 위한 기준을 제시합니다.



### Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain (https://arxiv.org/abs/2510.07309)
Comments:
          20 pages, 6 figures, under review for ACL ARR

- **What's New**: 이 논문은 CORGI라는 새로운 벤치마크를 소개합니다. 이는 실제 비즈니스 환경을 반영하여 설계되었으며, Doordash, Airbnb 및 Lululemon과 같은 기업에서 영감을 얻은 합성 데이터베이스(Synthetic Databases)를 포함하고 있습니다. CORGI는 네 가지 점진적으로 복잡한 비즈니스 쿼리 범주인 설명적, 설명적, 예측적 및 추천적 질문을 제공합니다.

- **Technical Details**: CORGI 벤치마크는 실행 성공률을 기준으로 BIRD 벤치마크보다 약 21% 더 어렵습니다. 기존의 LLM(대규모 언어 모델)은 설명적, 예측적 및 추천적 작업에서 성능 감소를 보이며, 이는 복잡한 의사결정과 전략적 계획을 반영하는 비즈니스 쿼리에 대한 기술적 도전 과제를 강조합니다. 이 데이터셋은 GitHub에 공개되어 있으며, 이를 통해 커뮤니티의 참여를 장려하고 더 많은 연구가 이루어지길 기대합니다.

- **Performance Highlights**: 초기 실험에서는 선도적인 LLM이 최고 수준의 질문에서 성능 저하를 보여주며, 이는 실제 비즈니스 인텔리전스와의 격차를 나타냅니다. 이를 통해 기존 LLM이 비즈니스 분석의 복잡성을 해결하기 위한 추가적인 연구가 필요하다는 점을 강조하고 있습니다. CORGI 벤치마크는 나중에 더욱 발전하여 비즈니스 의사결정 지원에 기여할 가능성이 있습니다.



### Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning (https://arxiv.org/abs/2510.07300)
Comments:
          13 pages, 8 tables, 4 figures

- **What's New**: 이 논문은 다국어 환경에서 대규모 추론 모델(Large Reasoning Models, LRM)의 문제를 해결하기 위해 M-Thinker를 제안합니다. 기존 LRM이 비영어 언어에서 언어 일관성 문제와 추론 경로의 정확성 낮음으로 인해 성능 저하를 겪는 문제를 지적하고, 새로운 GRPO 알고리즘을 통해 언어 일관성(Language Consistency, LC) 보상과 크로스링구얼 사고 정렬(Cross-lingual Thinking Alignment, CTA) 보상을 포함하는 방법론을 소개합니다.

- **Technical Details**: M-Thinker는 비영어 입력을 처리할 때 언어 일관성을 유지하기 위해 LC 보상을 활용하며, 크로스링구얼 정렬 보상은 모델의 영어 추론 경로를 다른 언어로 전이하여 다국어 추론 성능을 향상시킵니다. 이 모델은 체계적인 훈련 절차를 통해 훈련되며, 초기 SFT(Supervised Fine-Tuning), 거부 샘플링, 반복적 강화 학습(Iterative RL) 훈련을 포함합니다. 이는 모델이 두 개의 다국어 벤치마크(MMATH 및 PolyMath)에서 뛰어난 성능을 발휘하도록 돕습니다.

- **Performance Highlights**: M-Thinker-1.5B/7B 모델은 MMATH 및 PolyMath 벤치마크에서 거의 100%의 언어 일관성과 성능 향상을 달성했습니다. 또한, 다른 언어에 대한 일반화 능력이 뛰어난 것으로 나타났고, 이는 한국어와 같은 저자원 언어의 경우에도 우수한 성과를 나타냅니다. 이러한 결과는 모델의 추론 능력을 여러 언어로 확장하는 데 기여하며, 다국어 환경에서의 사용자 경험을 개선할 수 있는 가능성을 보여줍니다.



### On the Convergence of Moral Self-Correction in Large Language Models (https://arxiv.org/abs/2510.07290)
Comments:
          19pages, 7 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 '도덕적 자기 수정(moral self-correction)'이 성과 수렴(convergence)을 통해 개선될 수 있다는 것을 보여줍니다. 자기 수정(process of self-correction) 기능을 활용한 접근 방식은 복잡한 인간 피드백 없이 모델의 성능을 향상시킬 수 있는 잠재력을 지니고 있습니다. 실험 결과, 도덕적 자기 수정이 모델의 불확실성(model uncertainty)을 줄여 성과의 수렴을 이끌어내는 방식을 규명하였습니다.

- **Technical Details**: 연구는 LLMs가 자기 수정 지침을 반복적으로 적용해 나가면서 성과 수렴이 이루어진다는 관찰을 통해 두 가지 주요 연구 질문을 다룹니다. 첫째, 도덕적 자기 수정이 성과를 수렴하는가? 둘째, 이러한 수렴이 이루어지는 기본 메커니즘은 무엇인가? 연구진은 비정형적인 지침을 통해 활성화된 도덕 개념(latent concept)이 모델의 불확실성을 감소시킨다는 것을 밝혀냈습니다.

- **Performance Highlights**: 이 논문은 다양한 작업과 모델에서 도덕적 자기 수정이 수렴된 성과를 보여주는 실증적 증거를 제시합니다. 이를 통해 도덕적 자기 수정이텍스트 정제(text detoxification) 성과를 강화하는 데 중요한 역할을 한다는 점을 확인했습니다. 연구의 결과는 LLMs의 잠재력을 극대화하는 데 도움을 줄 수 있는 방향을 제시합니다.



### Online Rubrics Elicitation from Pairwise Comparisons (https://arxiv.org/abs/2510.07284)
- **What's New**: 이번 논문에서는 LLM을 위한 평가 기준을 온라인으로 동적으로 개발하는 새로운 방법인 Online Rubrics Elicitation(OnlineRubrics)을 소개합니다. 기존의 정적인 rubrics는 훈련 과정에서 발생하는 새로운 요구 사항을 제대로 반영하지 못했으나, OnlineRubrics는 응답을 기반으로 쌍평가(pairwise comparisons)를 통해 지속적으로 기준을 개선하는 방식입니다. 이 방법은 AlpacaEval, GPQA 등 여러 벤치마크에서 최대 8%의 성능 향상을 보였습니다.

- **Technical Details**: OnlineRubrics는 현재 모델과 참조 모델의 응답을 쌍으로 비교하여 새로운 평가 기준을 생성합니다. 이를 통해 응답의 오류를 지속적으로 식별하고 개선할 수 있으며, 기존의 rubrics를 유연하게 확장합니다. 평가 프레임워크는 reinforcement learning에 대한 다양한 응답 모델링을 가능하게 하여, verifiable 및 non-verifiable 특성을 모두 포괄합니다.

- **Performance Highlights**: 이 연구는 Expert 및 Generalist 도메인에 대한 두 개의 데이터셋을 활용하여 OnlineRubrics의 성능을 평가하였습니다. 이 방식은 GPQA-Diamond, GSM8K, AlpacaEval, Arena-Hard를 포함한 여러 벤치마크에서 기반 모델 대비 최고 25%의 성능 향상을 기록하며, 기존 정적 rubrics와 비교하여 우수한 결과를 입증했습니다.



### Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models (https://arxiv.org/abs/2510.07248)
Comments:
          15 pages, 4 figures

- **What's New**: 본 연구에서는 PA-Tool(Pretraining-Aligned Tool Schema Generation)라는 새로운 기법을 제안합니다. 이는 훈련 없이 도구의 구성 요소 이름을 자동으로 조정하여 모델의 사전 훈련(pretraining) 지식과 일치하는 이름 패턴을 찾는 방법입니다. 이 방법은 도구 사용 성능을 크게 향상시키면서도 작은 모델의 계산 효율성(computational efficiency)을 유지합니다.

- **Technical Details**: PA-Tool은 첫째, 도구 구성 요소에 대한 설명을 기반으로 다수의 후보 이름을 생성하고, 둘째, 각 후보의 peakedness를 계산하여 세 번째로 가장 높은 peakedness를 가진 후보를 새로운 이름으로 선택하는 과정을 포함합니다. 이 방식은 사전 훈련에서 모델이 친숙한 패턴을 인식하는 데 도움을 줍니다. 실험 결과, PA-Tool은 MetaTool 및 RoTBench에서 기존 모델에 비해 약 80%의 스키마 불일치(schema misalignment) 오류를 줄였습니다.

- **Performance Highlights**: PA-Tool은 작은 언어 모델들이 최신 성능을 거의 접근할 수 있도록 하며, 도구 선택 및 매개변수 식별의 정확성을 높입니다. 구체적으로, 도구 선택 작업에서 Qwen2.5-7B-Instruct 등 다양한 설정에서 Claude Sonnet 4.5의 성능을 초과하거나 유사한 성능을 발휘하는 것으로 나타났습니다. 이러한 성과는 PA-Tool이 도구 사용의 잠재력을 열어주는 중요한 역할을 함을 보여줍니다.



### LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation (https://arxiv.org/abs/2510.07243)
Comments:
          Published in Natural Legal Language Processing - EMNLP Workshop 2025

- **What's New**: 이 논문은 법률 분야에서 대형 언어 모델(LLM)의 출력을 평가하는데 있어 독창적인 접근 방식을 제시합니다. 연구진은 '법률 데이터 포인트'(Legal Data Points, LDPs)라는 자가 포함된 정보 단위를 활용하여 변호사들이 법률 답변을 평가하는 방식을 반영하는 새로운 평가 방법론인 LeMAJ를 소개합니다. 이 방법은 기존 기준 데이터에 의존하지 않으면서도 법률 질문 답변의 정확성과 관련성을 평가하는 데 효과적이라는 점을 강조합니다.

- **Technical Details**: LeMAJ 평가 방법론은 법률 전문직에서 사용하는 체계적인 평가 프로세스와 최근의 자동화된 요약 평가 기법에 영감을 받았습니다. LDPs를 사용하여 LLM에서 생성된 답변을 개별 정보 단위로 분해하고, 각 단위를 정확성과 관련성에 따라 평가합니다. 이 과정은 법률 전문가들이 필요한 세부적인 피드백을 제공하며, 기존 자동화된 평가 방법의 제약을 극복할 수 있습니다.

- **Performance Highlights**: 실험 결과, LeMAJ는 자체 개발한 데이터셋과 오픈 소스 데이터셋인 LegalBench에서 다양한 기존 방법들과 비교했을 때 성능이 뛰어난 것으로 나타났습니다. 또한, 변호사 간의 일치도를 개선하고, LDP를 사용한 경우 인간 전문가 평가와의 상관 관계가 더욱 높아짐을 보여주었습니다. 이러한 결과는 법률 질문 답변 평가에 있어 LLM의 평가 프로세스를 혁신적으로 변화시킬 가능성을 시사합니다.



### Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dens (https://arxiv.org/abs/2510.07242)
Comments:
          20 pages

- **What's New**: HERO(Hybrid Ensemble Reward Optimization)는 검증 가능한 보상과 보상 모델 점수를 통합하는 혁신적인 강화 학습 프레임워크입니다. 이 시스템은 정밀한 신호를 제공하는 검증 기구의 안정성과, 보다 풍부한 지속적 보상을 제공하는 보상 모델의 장점을 결합하기 위해 설계되었습니다. HERO는 기존의 이진 N 또는 1 보상 시스템에서 발생하는 한계를 극복하기 위한 두 가지 핵심 혁신 기능을 도입합니다.

- **Technical Details**: HERO는 두 가지 주요 기법을 통해 보상 모델 점수를 수정하여 최적화의 신뢰성을 유지합니다. 첫째, 계층화된 정규화(straction normalization) 기법을 통해 검증자가 정의한 정확성 그룹 내에서 보상 모델 점수를 제한합니다. 둘째, 분산 인식 가중치 기법을 사용하여 훈련 중 다양한 프롬프트의 기여도를 조정함으로써 어려운 질문에 대한 비율을 강조합니다.

- **Performance Highlights**: HERO는 다양한 수학적 추론 벤치마크에서 RM(Reward Model)-전용 및 검증자 전용 기준 모델들보다 일관되게 더 나은 성능을 보여주었습니다. 특히, 검증하기 어려운 작업의 경우, HERO는 66.3 점을 기록하여 RM-전용 시스템(54.6)보다 +11.7 점, 검증자 전용 시스템(57.1)보다 +9.2 점 높은 성과를 달성했습니다.



### Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts (https://arxiv.org/abs/2510.07239)
- **What's New**: 자동화된 레드 팀(who exercise adversarial testing) 방안인 Red-Bandit를 소개합니다. 이 프레임워크는 다양한 공격 스타일에 적합한 모델 고유의 취약점을 실시간으로 인지하고 이용하여 안정적인 테스트(mock testing)를 수행하는 것으로 설계되었습니다. Red-Bandit는 LoRA 전문가를 사용하여 각 공격 스타일에 대한 후속 훈련을 수행하며, 이때 강화 학습(reinforcement learning)으로 안전하지 않은 프롬프트(prompt)를 생성할 때 보상을 부여합니다.

- **Technical Details**: Red-Bandit는 다수의 팔(bandit) 테크닉을 활용하여 테스트 시 각기 다른 공격 스타일 중에서 최적의 전략을 선택합니다. 모델의 반응 안전성을 기반으로 다익스트라(Dijkstra) 알고리즘을 이용해 전문가들을 동적으로 선택하며, 이러한 접근법은 탐색과 활용(exploiting) 간의 균형을 맞추게 됩니다. 이 방식은 LoRA 기술을 통해 커스터마이즈된 공격 스타일 전문가들을 효율적으로 훈련하여, 빠른 학습을 수행할 수 있게 해줍니다.

- **Performance Highlights**: Red-Bandit는 AdvBench 데이터셋에서 최신 기법들에 비해 뛰어난 성능을 기록하였으며, 높은 공격 성공률과 낮은 perplexity(사람이 이해하기 쉬운 프롬프트 생성)를 보여주었습니다. 또한, 이 프레임워크는 어떤 공격 스타일이 모델 별 취약점에 가장 효과적으로 작용하는지를 드러내는 진단 도구로 기능하기 때문에, 모델과 공격 방법에 대한 심층적인 통찰을 제공합니다.



### When Benchmarks Age: Temporal Misalignment through Large Language Model Factuality Evaluation (https://arxiv.org/abs/2510.07238)
- **What's New**: 본 연구는 최신 대형 언어 모델(LLMs)에 대한 평가의 신뢰성에 대한 우려를 제기합니다. 기존의 평가지표들이 시간이 지남에 따라 사실과 불일치하게 되어, LLM의 실제 사실성(factuality) 평가에 부정적인 영향을 줍니다. 연구자들은 최신 사실 검색 파이프라인과 세 개의 지표를 개발하여 평가지표의 노화(aging)와 LLM 평가에 미치는 영향을 정량화합니다.

- **Technical Details**: 연구 방법으로는 시간 민감 질문(time-sensitive questions)을 구성하여, 고유한 사실적 답변이 시간이 지남에 따라 변경되는지 비교합니다. 이를 위해 8개의 대형 언어 모델(LLMs)과 5개의 널리 사용되는 사실성 벤치마크에 대한 실험을 실시했습니다. 세 가지 지표인 Dataset Drift Score, Evaluation Misleading Rate, Temporal Alignment Gap을 사용하여 분석을 수행하였습니다.

- **Performance Highlights**: 실험 결과, 최소 24.19%에서 최대 63.78%의 시간 민감 샘플이 구식인 것으로 나타났습니다. 현재 사용되는 여러 벤치마크의 상당 분량이 시간이 지남에 따라 사실과 일치하지 않으며, LLM의 사실성 평가가 불완전하다는 점이 드러났습니다. 최신 모델들은 상대적으로 더 높은 E​M​REMR을 기록하였으며, 이는 최신 LLM이 더 신뢰할 수 있는 답변을 제공할 수 있음을 시사합니다.



### LAD-RAG: Layout-aware Dynamic RAG for Visually-Rich Document Understanding (https://arxiv.org/abs/2510.07233)
- **What's New**: 이 논문에서는 시각적으로 풍부한 문서(VRDs)에 대한 질문 답변에서 구조적 조직과 페이지 간 의존성을 처리하기 위한 새로운 접근법인 LAD-RAG를 소개합니다. 기존의 retrieval-augmented generation (RAG) 방법들이 단편적인 내용만을 인코딩하는 데 비해, LAD-RAG는 문서의 레이아웃 구조를 반영한 상징적(document graph) 그래프를 구성하여 문서의 전체적인 표현을 개선합니다.

- **Technical Details**: LAD-RAG 프레임워크는 정보 수집 단계에서 문서의 구조를 캡처하는 상징적 그래프를 생성하고, 이를 표준 신경 임베딩(neural embeddings)에 추가하여 문서의 내용을 보다 포괄적으로 표현합니다. 추론 단계에서는 LLM(agent)이 신경적 및 상징적 인덱스와 동적으로 상호작용하여 질문에 따라 필요한 증거를 능동적으로 검색할 수 있도록 설계되었습니다.

- **Performance Highlights**: MMLongBench-Doc, LongDocURL, DUDE, MP-DocVQA 데이터셋을 사용하는 실험 결과, LAD-RAG는 평균 90% 이상의 완전 회수율(complete recall)을 달성하고, 기준 회수기(baseline retrievers)에 비해 최대 20% 더 높은 회수율을 기록했습니다. 이는 질문 응답(QA) 정확성을 향상시키는 동시에 지연 시간을 최소화하는 데 기여합니다.



### Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships (https://arxiv.org/abs/2510.07231)
- **What's New**: 이 논문에서는 인과 추론(causal reasoning)이 대형 언어 모델(Large Language Models, LLMs)의 핵심이지만 기존 벤치마크가 그 기능을 충분히 평가하지 못하는 경우가 많다는 점을 지적합니다. 저자들은 경제 및 금융 저널에서 추출한 비계통적인 인과 관계를 기반으로 새로운 벤치마크를 소개하며, 이는 합리적인 방법론을 이용해 수립되었습니다. 또한, 40,379개의 평가 항목과 다양한 도메인을 포함하는 다섯 가지 작업 유형을 다룹니다.

- **Technical Details**: 제안된 벤치마크는 도구적 변수(instrumental variables), 차이의 차이(difference-in-differences), 회귀 불연속 디자인(regression discontinuity designs)과 같은 엄격한 방법론을 바탕으로 구축되었습니다. 여기에는 건강, 환경, 기술, 법률 및 문화와 같은 다양한 분야가 포함됩니다. 실험에서는 총 8개의 최신 LLMs 모델이 평가되었으며, 이 모델들이 인과 관계 파악에 있어 상당한 제한사항을 보였습니다.

- **Performance Highlights**: 실험 결과, 가장 성능이 좋은 모델조차 57.6%의 정확도에 불과해 인과 관계 인식에 많은 어려움을 겪고 있음을 보여줍니다. 모델의 규모가 항상 성능 향상으로 이어지지 않으며, 고급 추론 모델들도 기본적인 인과 관계 식별에 어려움을 겪습니다. 이러한 발견은 현재 LLM의 성능과 신뢰할 수 있는 인과 추론이 요구되는 고위험 응용 프로그램 간의 중요한 격차를 강조합니다.



### Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping (https://arxiv.org/abs/2510.07230)
- **What's New**: 이 논문은 대형 언어 모델(LLM)을 활용하여 개인화된 사용자 행동을 단계별로 시뮬레이션하는 새로운 방법인 Customer-R1을 소개합니다. 기존의 방법들은 일반적인 사용자의 행동만을 학습했으나, Customer-R1은 개별 사용자의 페르소나를 명시적으로 고려하여 보다 개인화된 결과를 도출합니다. 이번 연구는 온라인 쇼핑 환경에서 사용자 행동 시뮬레이션을 향상시키기 위한 RL(강화학습) 기반의 접근 방식을 제안하고 있습니다.

- **Technical Details**: Customer-R1은 사용자의 이전 행동 데이터를 바탕으로 다음 행동을 예측하는 작업을 수행합니다. 이 모델은 각 사용자의 페르소나와 행동의 정당성을 고려하여 행동을 생성하고, 이를 통해 맞춤형 시뮬레이션을 가능하게 합니다. OPeRA 데이터셋을 활용한 실험에서 Customer-R1이 다음 행동 예측 작업에서 prompting 및 SFT 기반의 기준선보다 우수한 성능을 보였음을 입증하였습니다.

- **Performance Highlights**: Customer-R1은 개별 사용자의 행동 분포와 더 잘 일치하며, 이는 개인화된 사용자 행동 시뮬레이션의 정확성에서 높은 충실도를 나타냅니다. 추가적인 실험을 통해 올바른 페르소나 정보 사용이 성능 향상에 기여하며, 잘못된 페르소나 정보는 혼란을 초래해 정확도를 저하시킨다는 것을 보여주었습니다. 이 연구 결과는 온라인 쇼핑에서 개인화된 행동 시뮬레이터를 구축하는 데 유용한 실용적인 지침을 제공합니다.



### Where to Begin: Efficient Pretraining via Subnetwork Selection and Distillation (https://arxiv.org/abs/2510.07227)
- **What's New**: 이 논문에서는 소형 언어 모델(SLMs)에 대한 새로운 사전 학습 프레임워크를 제안합니다. 이 프레임워크는 구조적으로 희소한 서브 네트워크 초기화, 진화적 검색을 통한 모델 초기화, 그리고 지식 증류(knowledge distillation)의 세 가지 보완적인 개념을 결합합니다. 이 접근 방식은 SLM 사전 학습의 효율성을 크게 향상시킵니다.

- **Technical Details**: SLMs는 대형 언어 모델(LLMs)과 비교하여 적은 자원으로도 강력한 성능을 발휘하는 모델입니다. 본 연구에서는 사전 학습된 교사 모델로부터 서브 네트워크를 추출하고 이를 사용하여 SLM을 초기화하고 지식 증류를 수행하는 2단계 전략을 채택합니다. 다양한 검색 공간과 초기화 방법에 대해 체계적인 비교 분석을 제공합니다.

- **Performance Highlights**: 진화적 검색을 통해 발견된 최상 모델은 LLM 가중치로 초기화되었고, 동등한 Pythia SLM의 검증 perplexity와 일치하면서 9.2배 적은 사전 학습 토큰을 요구합니다. 실험 결과, 다양한 크기의 학생 모델에 대해 사전 학습과 지식 증류를 통해 전반적인 효율성을 높이는데 기여함을 보여줍니다. 이 연구는 경제적인 SLM 개발을 위한 실질적 가이드라인을 제공합니다.



### How much speech data is necessary for ASR in African languages? An evaluation of data scaling in Kinyarwanda and Kikuyu (https://arxiv.org/abs/2510.07221)
- **What's New**: 본 논문은 제한된 음성 데이터로 인해 저자원 아프리카 언어를 위한 자동 음성 인식(ASR) 시스템 개발의 도전 과제를 다룹니다. 오픈AI의 Whisper와 같은 대규모 다국어 모델이 ASR 발전을 위한 새로운 가능성을 제공하며, 필요한 최소 학습 데이터 양과 시스템의 주요 실패 모드를 규명하는 데 중점을 두고 있습니다. 실험을 통해 Kinyarwanda와 Kikuyu 두 가지 반투어 언어에서 Whisper의 성능을 평가하고, 실제 ASR 성능을 달성하기 위한 데이터 양의 중요성을 강조합니다.

- **Technical Details**: Whisper large-v3 모델(1.55억 개의 파라미터)을 사용하여 Kinyarwanda 및 Kikuyu 데이터에 대해 정량적 및 정성적 분석이 수행되었습니다. Kinyarwanda는 1시간에서 1,400시간까지 다양한 학습 데이터 세트를 통해 시스템 성능을 평가하였고, Kikuyu는 270시간의 세밀하게 선별된 음성 데이터를 사용하였습니다. 데이터 증강 기법으로 회귀 노이즈 주입, 속도 변동, 다운샘플링이 적용되었으며, 학습 과정은 H100 또는 A100 GPU에서 진행되었습니다.

- **Performance Highlights**: Kinyarwanda 실험에서 최소 50시간의 훈련 데이터로 WER(단어 오류율)이 12.51%에 달하여, 이는 1시간 기준보다 75%의 향상을 나타냅니다. 200시간 이상의 데이터로는 WER이 9.82%로 감소하였으며, 전체 1,400시간 데이터에서 7.14%로의 추가 향상이 있었습니다. Kikuyu 모델은 270시간 훈련 후 중앙 WER이 26.3%, 평균 WER이 30.3%로 나타났으며, 이러한 결과는 데이터의 양 뿐 아니라 품질도 ASR 시스템 성능에 중요한 영향을 미침을 시사합니다.



### Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for Large Language Models (https://arxiv.org/abs/2510.07213)
Comments:
          Work in progress. Our code will be available at: this https URL

- **What's New**: 이번 연구는 대규모 언어 모델(LLM)이 비영어 데이터에 대한 노출이 제한적임에도 불구하고 다국어 능력이 뛰어난다는 점을 강조합니다. 연구진은 언어 전환이 중간층에서 최종층으로 갈수록 일관된 인덱스에서 발생하는 소수의 차원에 의해 조정된다고 가설을 세웠습니다. 이에 따라, 단 50개의 문장으로 이러한 차원을 식별하고 조작할 수 있는 훈련 없는 방법을 소개하였습니다.

- **Technical Details**: 연구에서는 LLM의 각 층에서의 표현 변화를 관찰하며, 고유 언어 관련 차원들을 식별하는 두 가지 시나리오를 제안합니다. 첫 번째는 단일언어(monolingual) 설정으로, 중간층과 최종층의 표현을 비교하여 언어별 차원을 확인합니다. 두 번째는 병렬(parallel) 설정으로, 영어와 목표 언어의 최종층 표현을 비교하여 역시 언어별 차원을 찾습니다.

- **Performance Highlights**: 실험 결과, 제안을 통해 식별한 언어별 차원을 조작함으로써 출력 언어를 전환하는 데 성공했습니다. 이 방법은 기존의 뉴런 기반 접근 방식보다 성능이 우수하며, 데이터 요구량이 적어 효율적입니다. 다양한 모델(Llama2, Llama3.1 등)에서 다국어 생성 제어 실험을 통해 입증된 바와 같이, 본 연구는 LLM의 다국어 처리에서 중요한 기여를 하였습니다.



### Sunflower: A New Approach To Expanding Coverage of African Languages in Large Language Models (https://arxiv.org/abs/2510.07203)
- **What's New**: 이 논문은 아프리카의 다양한 언어들이 언어 기술 발전에서 소외되고 있는 상황을 다루고 있습니다. 기존의 대규모 언어 모델(LLM)은 사용자가 많은 언어에 우선적으로 집중하여 여러 언어에서 지원이 불균형하게 나타납니다. 논문에서는 우간다의 언어적 다양성을 중점적으로 다루는 지역 중심 접근법을 제안하며, 두 개의 모델인 Sunflower 14B와 32B를 개발하여 우간다의 주요 언어에 대한 이해도를 높였습니다.

- **Technical Details**: 이 모델들은 Qwen 3 기반으로 개발되었으며, 우간다에서 사용되는 40여 개 언어를 지원하고 있습니다. 이를 통해 번역, 질문-답변, 그리고 다양한 창의적 작업을 수행할 수 있습니다. 이 지역 중심 접근은 공유된 언어 구조, 문화적 개념, 맥락적 지식을 활용하여 효과적인 데이터 전이를 가능하게 합니다.

- **Performance Highlights**: 우리는 기계 번역 성능에 초점을 맞추어 31개 평가 언어 중 24개 언어에서 최첨단 성능을 달성하였습니다. Sunflower 모델은 개방형 소스로 제공되어 우간다에서 언어 장벽을 줄이는 데 사용될 수 있으며, 아프리카 언어 기술 연구를 가속화하는 데 기여할 것입니다.



### Biasless Language Models Learn Unnaturally: How LLMs Fail to Distinguish the Possible from the Impossib (https://arxiv.org/abs/2510.07178)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 인간의 언어와 불가능한 언어의 구분에 민감한지를 조명하고 있습니다. 이전의 연구와는 달리, 저자들은 다양한 언어와 불가능한 변형을 대상으로 비교실험을 수행하였고, 그 결과 GPT-2 모델이 가능 언어와 불가능한 언어를 동일하게 학습하고 있음을 파악했습니다. 이는 LLMs가 인간의 고유한 언어 학습 편향을 공유하지 않는다는 주장을 뒷받침합니다.

- **Technical Details**: 연구진은 Kallini et al. (2024)의 방법론을 사용하여, GPT-2 모델이 영어 데이터셋과 인위적으로 변형된 데이터셋에서 훈련될 때의 perplexity를 비교하였습니다. 그들은 이러한 방식을 통해 가능한 언어와 불가능한 변형 간에 실질적인 차이가 없음을 확인하였고, LLM이 특정 언어에서 학습하는 용이성을 측정하는 기초 지표를 사용하였습니다. 그러나 이 연구는 다양한 언어와 변형으로 범위를 확장하여, LLM이 가능한 언어와 불가능한 변형을 동등하게 학습하는 경향을 보였습니다.

- **Performance Highlights**: GPT-2 모델의 성능은 일반 가능 언어와 불가능한 언어 변형 간 구분이 없었으며, 일부 경우에는 불가능한 변형을 선호하는 경향을 보였습니다. 이는 LLM이 인간 학습에서 기대되는 편향을 공유하고 있지 않음을 나타냅니다. 또한, 연구 결과는 LLM이 인간 언어 인지에 대한 신뢰할 수 있는 모델이 아닐 수 있음을 시사합니다.



### CARPAS: Towards Content-Aware Refinement of Provided Aspects for Summarization in Large Language Models (https://arxiv.org/abs/2510.07177)
Comments:
          22 pages, 17 figures

- **What's New**: 이 논문에서는 'Content-Aware Refinement of Provided Aspects for Summarization (CARPAS)'라는 새로운 과제를 제안합니다. 이 과제의 목표는 문서의 맥락에 따라 제공된 측면(aspects)을 동적으로 조정하여 보다 적합한 요약을 생성하는 것입니다. 기존의 접근 방식은 고정된 측면에 의존하는 경향이 있지만, 실제 환경에서는 이러한 측면이 불완전하거나 관련성이 없을 수 있기 때문에 새로운 접근이 필요합니다.

- **Technical Details**: 논문에서는 LLM(대형 언어 모델)을 활용하여 제공된 측면을 예측하고, 이를 바탕으로 요약을 생성하는 두 가지 단계의 방법론을 탐색합니다. 네 가지의 대표적인 prompting 전략(직접 prompting, Chain-of-Thought prompting, Chain-of-Thought with self-consistency, Self-Refine)을 사용하고, 이들 전략이 LLM의 요약 품질에 미치는 영향을 실험적으로 분석하였습니다. 또한, LLM의 예측 정확도를 높이기 위해 관련 측면의 수를 미리 예측하는 하부 작업을 소개하고, 이 예측이 LLM의 추론 과정에 있어 추가적인 도움이 될 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과에서는 제안된 방법론이 모든 데이터셋에서 성능을 크게 향상시키는 것으로 나타났습니다. 특히, LLM이 제공된 측면 집합의 수를 맹목적으로 신뢰하는 경향이 있음을 발견하였고, 이는 LLM을 유사한 산업 환경에 적용하기 위한 중요한 통찰력을 제공합니다. 이번 연구는 CARPAS의 코드와 데이터셋을 공개하여 향후 연구에 기여할 수 있는 기반을 마련하였습니다.



### Quantifying Data Contamination in Psychometric Evaluations of LLMs (https://arxiv.org/abs/2510.07175)
Comments:
          12 pages, 1 figure

- **What's New**: 최근 연구에서는 Large Language Model(LLM)에 심리측정 설문지를 적용하여 가치관, 성격, 도덕적 기반 및 어두운 특성과 같은 고차원 심리적 구성 요소를 평가하고 있습니다. 이전 연구에서 심리측정 inventory로 인한 데이터 오염 가능성이 제기되었음에도 불구하고, 이러한 오염의 정도를 체계적으로 정량화한 연구는 없었습니다. 이에 대한 해결책으로, 본 연구는 LLM의 심리측정 평가에서 데이터 오염 정도를 체계적으로 측정하는 프레임워크를 제안하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 측면을 기준으로 데이터 오염을 평가합니다: (1) 항목 암기(item memorization), (2) 평가 암기(evaluation memorization), (3) 목표 점수 일치(target score matching)입니다. 연구진은 21개의 모델과 널리 사용되는 심리측정 설문인 Big Five Inventory(BFI-44) 및 Portrait Values Questionnaire(PVQ) 등을 포함한 여러 재고 목록에 프레임워크를 적용하였습니다. 결과적으로, 대부분의 LLM이 이러한 inventory에 오염되어 있다는 강력한 증거가 확보되었습니다.

- **Performance Highlights**: 연구 결과, LLM들은 단순히 아이템을 암기하는 것을 넘어, 특정 목표 점수를 달성하기 위해 응답을 조정할 수 있는 능력을 갖추고 있음을 보였습니다. 이는 LLM이 심리측정 테스트에서 과거의 훈련 데이터에 노출된 내용을 바탕으로 응답할 가능성이 있음을 시사합니다. 이러한 결과는 LLM의 심리적 특성 평가에 대한 더 체계적인 조사를 필요로 하고 있으며, 향후 연구에서 모델이 심리측정 테스트의 내용을 조작하는지 여부를 밝혀내는 기초가 될 것입니다.



### NurseLLM: The First Specialized Language Model for Nursing (https://arxiv.org/abs/2510.07173)
Comments:
          EMNLP 2025 Industry Track

- **What's New**: 이번 논문은 간호 분야에 맞춤화된 첫 번째 대규모 언어 모델인 NurseLLM을 소개합니다. NurseLLM은 여러 선택 질문-응답(MCQ) 작업을 위해 설계되어 있으며, 이를 위해 대규모 간호 MCQ 데이터 세트를 구축하였습니다. 기존 모델들과 비교할 때, NurseLLM은 간호 관련 작업에서 더욱 뛰어난 성능을 보이며, 전문화된 LLM의 필요성을 강조합니다. 또한, 간호 분야에서의 추론 및 다중 에이전트 협업 시스템의 가능성을 탐구하고 있습니다.

- **Technical Details**: NurseLLM은 복잡한 NCLEX와 같은 간호 전문 질문-응답을 다루기 위해 다양한 주제의 NCLEX 질문-답변 조합을 생성하기 위해 다단계 데이터 생성 파이프라인을 활용하였습니다. 이 과정에서 1,251,125개의 샘플을 포함한 대규모 데이터 세트를 구축하였으며, 세 가지 간호 MCQ 벤치마크를 개발하여 LLM의 체계적인 평가를 가능하게 하였습니다. 더불어, 모델의 투명성과 신뢰성을 확보하기 위해 정답과 함께 합리적인 이유를 제공하도록 설계되었습니다.

- **Performance Highlights**: NurseLLM은 유사한 크기의 범용 및 의료 전문 LLM들을 상대로 다양한 벤치마크에서 우수한 성능을 발휘하였습니다. 이는 간호 분야에서 전문화된 LLM의 중요성을 뒷받침합니다. 실험 결과, NurseLLM의 성능은 기존의 일반적 모델들보다 더 높은 정확성을 보여주었으며, 이는 간호에 특화된 AI 도구의 필요성을 강조하게 됩니다. 추론 능력과 협업 시스템의 통합을 통해 향후 간호 분야의 발전 가능성에 대한 기대감을 높이고 있습니다.



### More Data or Better Data? A Critical Analysis of Data Selection and Synthesis for Mathematical Reasoning (https://arxiv.org/abs/2510.07169)
Comments:
          12 pages, 3 figures, submitted to EMNLP 2025 Industry Track

- **What's New**: 이번 연구에서는 데이터 선택 전략과 데이터 합성 방법의 효과성을 수학 영역에서 조사하였습니다. 새로운 데이터셋을 도입할 때 단순히 데이터를 추가하는 것이 항상 유익한 것은 아니며, 보다 질이 높은 데이터 선택이 중요함을 강조합니다. 연구팀은 다양한 오픈 소스 데이터셋과 데이터 합성 기술을 체계적으로 분석하고, 산업 환경에서의 실질적인 적용 가능성을 평가했습니다.

- **Technical Details**: 정량적으로 평가하기 위해 통일된 평가 방법론을 채택하였으며, DeepSeek-V2-Lite 또는 Qwen2.5-3B 모델을 기본 모델로 사용하였습니다. 데이터셋의 효과성을 평가하기 위해, 검증된 데이터셋과 새롭게 평가할 데이터셋을 혼합하여 훈련하였고, 이 과정에서 훈련 샘플의 20%는 새로운 데이터셋에서 가져오도록 설정했습니다. 평가 작업은 일반 지식, 논리적 추론, 수학적 추론, 코딩 능력 등 네 가지 유형에 걸쳐 있습니다.

- **Performance Highlights**: 수학적 추론 성능 향상에 주목하면서도 다른 능력에서의 성능 저하를 모니터링했습니다. 다양한 오픈 소스 데이터셋에 대한 평가 결과는 수학적 추론 작업에 효과적인 데이터 선택 전략 수립하는 데 기여했습니다. 연구 결과, 데이터의 양보다 질을 중시하는 것이 모델 성능을 향상시키는 데 더 효과적임을 발견했습니다.



### Reasoning for Hierarchical Text Classification: The Case of Patents (https://arxiv.org/abs/2510.07167)
Comments:
          15 pages, 10 tables, 3 figures

- **What's New**: 이번 연구에서는 **Reasoning for Hierarchical Classification (RHC)**라는 새로운 프레임워크를 제안함으로써 계층적 텍스트 분류 문제를 단계별 추론 작업으로 재구성합니다. 이 방법은 대규모 언어 모델을 두 단계로 훈련시키며, 첫 번째 단계에서는 차례로 생각을 체계화하고, 두 번째 단계에서는 강화 학습을 통해 다단계 추론 능력을 향상시킵니다. RHC는 이러한 방식으로 기존의 방법들이 갖고 있던 한계를 극복하고 다양한 분야에 적용 가능한 가능성을 보여줍니다.

- **Technical Details**: RHC의 훈련 절차는 첫째, 모델의 출력을 구조화된 CoT(Chain-of-Thought) 형식과 Align하는 **콜드 스타트 단계**와 둘째로, 모델의 최종 목표인 다단계 추론 능력을 향상시키기 위한 **강화 학습 단계**로 구성됩니다. 이 과정에서 **GPT-5**와 같은 강력한 모델이 합성된 추론 경로를 생성하며, 이는 계층적 분류 문제를 구조화된 추론 작업으로 변환하는 데 기여합니다. 이 방법은 결과적으로 모델이 **상위**부터 **하위** 레벨까지 점진적으로 라벨을 예측하고, 인간이 해석할 수 있는 정당화도 함께 제공합니다.

- **Performance Highlights**: 실험 결과, RHC는 기존의 방법 대비 약 3%의 정확도를 초과하는 성과를 보이며, 해석 가능성, 확장성 측면에서도 우수함을 검증했습니다. 특히, **상태-최고(state-of-the-art)** 성능을 나타내는 계층적 텍스트 분류 기준에서도 좋은 결과를 도출했습니다. RHC는 다양한 레이블을 포함하는 복잡한 분류 문제를 효율적으로 해석할 수 있는 능력을 갖췄다는 점에서, 향후 다양한 분야로의 적용 가능성을 제시합니다.



### Comparing human and language models sentence processing difficulties on complex structures (https://arxiv.org/abs/2510.07141)
Comments:
          Data and code will be released soon

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)과 인간의 문장 이해 능력을 체계적으로 비교하였습니다. 7가지의 도전적인 언어 구조를 활용하여 인간과 5개의 LLM 계열에서 얻은 문장 이해 데이터를 수집했습니다. 연구 결과에 따르면, LLM은 전반적으로 목표 구조에서 어려움을 겪으며, 특히 'garden path' (GP) 문장에서 그 성능이 낮았습니다.

- **Technical Details**: 분석된 7가지 구조에는 GP 문장, 이중 중심 내재 문장, 유사성 기반 간섭 문장, 깊이 충전 문장 등이 포함됩니다. 각 구조에 대해 어려운 문장(타겟 문장)과 난이도가 중화된 기준 문장을 설정하였습니다. 연구에서는 31개의 최첨단 LLM을 테스트하며, 각 LLM과 인간 참여자가 동일한 질문에 답하도록 하여 성과를 비교했습니다.

- **Performance Highlights**: 인간의 평균 정확도는 41.7%로 확인되었으며, LLM은 전반적으로 인간보다 나은 성과를 보였지만 여전히 이러한 구조에서 어려움이 있었습니다. LLM은 GP 구조에서 인간과 상대적으로 비슷한 성과를 보였고, 모델의 크기가 커짐에 따라 성과의 순위 상관관계가 증가하는 경향을 보였습니다. LLM의 성능은 구조의 난이도에 따라 달라지며, 특정 모델의 경우 문장 간 발생하는 방향성을 기반으로 한 성과 차이를 재현하지 못하는 경우도 있었습니다.



### TRIM: Token-wise Attention-Derived Saliency for Data-Efficient Instruction Tuning (https://arxiv.org/abs/2510.07118)
- **What's New**: 이 논문은 TRIM (Token Relevance via Interpretable Multi-layer Attention)이라는 새로운 방법을 소개합니다. TRIM은 작은 고품질 샘플 집합을 사용하여 큰 언어 모델(LLM)의 지침 조정을 위한 데이터 선택을 효율적으로 수행하는 방법론입니다. 특히, 기존의 경량화 방법들과 달리 Gradient 사용을 피하고 오직 Forward 패스를 통해 성능을 극대화하는 것이 특징입니다.

- **Technical Details**: TRIM은 토큰 중심의 프레임워크로, 주어진 몇 개의 샘플을 통해 이들 샘플의 토큰 표현을 활용하여 타겟 태스크에 최적화된 데이터를 선택합니다. 이 방법은 각 토큰의 attention 기반의 'fingerprint'를 통해 세부적인 특성을 포착하려고 합니다. Two-stage 구조를 적용하여 첫 번째 단계에서는 토큰의 saliency 점수를 계산하고, 두 번째 단계에서는 이 점수를 토대로 후보 샘플의 relevancy 점수를 계산하여 최종 코어셋을 구축합니다.

- **Performance Highlights**: TRIM은 단 5-10개의 샘플에서 최대 9% 이상의 성과 향상을 보여주며, 전체 데이터에 대한 Fine-tuning을 초월한 성능을 기록합니다. 이 기술은 더 높은 구조적 충실도를 제공하며, 샘플 수준의 방법에서 발생하는 길이 편향을 완화하는데 도움을 줍니다. 결과적으로 TRIM은 기존 방법들보다 빠르고 효율적으로 고품질의 지침 조정 데이터 세트를 구축할 수 있는 가능성을 보여줍니다.



### Opt-ICL at LeWiDi-2025: Maximizing In-Context Signal from Rater Examples via Meta-Learning (https://arxiv.org/abs/2510.07105)
Comments:
          NLPerspectives: The 4th Workshop on Perspectivist Approaches to Natural Language Processing at EMNLP 2025

- **What's New**: 이 논문에서는 자연어 처리(NLP)에서 주관성, 모호성 및 주석자 간의 합법적 의견 차이를 모델링하기 위한 새로운 시스템을 제시합니다. 저자들은 언어 모델(LLMs)의 in-context learning 기능을 활용하여 데이터셋에 대한 포스트 훈련과 메타 학습을 통해 모델을 전문화하는 두 단계의 훈련 절차를 수립했습니다. 이 시스템은 'Learning With Disagreements'(LeWiDi) 대회에서 두 가지 작업 모두에서 전체 우승을 차지했으며, 각 시스템 구성 요소의 중요성을 측정하기 위한 ablation study를 수행했습니다.

- **Technical Details**: 제안된 시스템(Opt-ICL)은 'perspectivist' 접근 방식을 채택하여 각 개별 주석자가 각 인스턴스를 어떻게 평가했는지를 예측하고, 이러한 개별 예측을 집계하여 soft task를 수행합니다. 이 시스템은 LLM의 in-context learning 능력을 활용하며, 포스트 훈련과 데이터셋 특화 fine-tuning 과정을 포함합니다. Spectrum Tuning 방법론을 사용하여 다양한 데이터셋에서 인간 변동성 및 불확실성을 처리할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: 제안된 시스템은 각 구성 요소가 성능에 미치는 영향을 분석하기 위해 ablation study를 수행했고, 특히 in-context에서 주석자 예시를 포함하는 것이 성능에 중요하다는 것을 발견했습니다. 대규모 데이터셋에서 데이터셋 특화 fine-tuning이 도움이 되었고, 다른 in-context 데이터셋에서의 포스트 훈련이 경쟁 데이터셋 중 하나에서 성능 향상에 기여했습니다. 또한, 모델 크기가 커질수록 성능이 향상되는 경향이 있음을 확인했습니다.



### TALENT: Table VQA via Augmented Language-Enhanced Natural-text Transcription (https://arxiv.org/abs/2510.07098)
- **What's New**: 이번 논문에서는 TALENT(테이블 VQA를 위한 Augmented Language-Enhanced Natural-text Transcription)이라는 경량 프레임워크를 제안합니다. TALENT는 작은 VLM(비전-언어 모델)을 이용하여 OCR(광학 문자 인식) 텍스트와 자연어 내러티브를 생성하고, 이를 질문과 결합하여 LLM(대형 언어 모델)에서 추론을 수행합니다. 이를 통해 테이블 VQA를 LLM 중심의 다중모드 추론 작업으로 재구성합니다.

- **Technical Details**: TALENT는 두 가지 보완적인 테이블 표현을 활용하며, 이에는 OCR 스팬(정확한 기호 콘텐츠를 제공)과 테이블의 구조 및 핵심 값을 설명하는 자연어 내러티브가 포함됩니다. 이러한 이중 표현은 LLM으로 전달되어 질문에 대한 추론을 수행하게 됩니다. ReTabVQA는 다단계 수치적 추론을 요구하는 새로운 테이블 VQA 데이터셋으로, 기존 데이터셋의 한계를 극복하도록 설계되었습니다.

- **Performance Highlights**: TALENT는 작은 VLM-LLM 조합이 단일 대형 VLM에 비해 동일하거나 더 나은 성능을 낼 수 있음을 실험적으로 입증했습니다. 실험 결과, TALENT의 접근 방식은 공공 데이터셋과 ReTabVQA에서 모두 매력적인 정확도를달성하며, 모바일 및 엣지 디바이스에서도 효율적으로 배포될 수 있는 솔루션을 제공합니다.



### Making Machines Sound Sarcastic: LLM-Enhanced and Retrieval-Guided Sarcastic Speech Synthesis (https://arxiv.org/abs/2510.07096)
- **What's New**: 이번 연구에서는 비꼬는 발화를 위한 음성 합성(Speech Synthesis) 기법에 새로운 접근법을 제안합니다. 우리는 빅 언어 모델(LLM)과 정보 검색 기반(Retrieval-Augmented) 시스템을 활용하여 비꼬기의 비문학적 패턴을 더 잘 표현할 수 있는 시스템을 구축합니다. 특히, LLaMA 3 모델을 LoRA 방식으로 조정해 비꼬기 관련 의미 임베딩을 생성하고, 이를 기반으로 한 검색 모듈을 통해 비꼬기 표현에 적합한 발화 예시를 효과적으로 활용합니다.

- **Technical Details**: 제안하는 방법론은 두 가지 주요 구성 요소로 이루어집니다. 첫째, LoRA(저차원 적응)로 미세 조정된 LLaMA 3 모델은 비꼬기 감지에 유용한 의미 임베딩을 생성합니다. 둘째, RAG 모듈을 통해 수집된 발화 예시로부터 프로소디(Prosody) 정보를 열거할 수 있어, 더 표현력이 풍부한 음성 합성 결과를 얻을 수 있습니다. 이러한 두 가지 기법이 통합되어 비꼬기에 적합한 발화를 생성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 기법들에 비해 음성의 자연스러움(naturalness), 표현력(expressivity), 비꼬기 표현의 정확성에서 뛰어난 성과를 보였습니다. 관찰된 바에 따르면 모델은 비꼬기 감지 능력까지 향상시켰으며, 이는 특히 인간-컴퓨터 상호작용(Human-Computer Interaction) 분야에서의 잠재적 응용 가능성을 더욱 높였습니다.



### All Claims Are Equal, but Some Claims Are More Equal Than Others: Importance-Sensitive Factuality Evaluation of LLM Generations (https://arxiv.org/abs/2510.07083)
- **What's New**: 본 연구에서는 기존의 방법들이 대규모 언어 모델(LLM)의 응답에서 사실성을 평가할 때 모든 주장(claim)을 동등하게 중요하게 취급하고 있다는 문제를 다룹니다. 이에 따라 중요하거나 핵심적인 정보가 누락되거나 잘못되었을 경우, 이러한 오류가 부정확하게 평가 될 수 있음을 지적합니다. 이를 해결하기 위해 VITAL이라는 새로운 메트릭 세트를 도입하여 응답의 사실성을 평가할 수 있는 민감도를 높였습니다.

- **Technical Details**: VITAL 메트릭은 사용자 쿼리와 관련하여 주장의 중요성과 관련성을 삽입하여 사실성을 측정하는 데 더욱 민감하게 설계되었습니다. 연구자는 6,733개의 쿼리로 구성된 VITALERRORS라는 벤치마크를 구축하고, 키 정보 누락 및 오류에 대하여 기존 메트릭들이 얼마나 둔감한지를 실험을 통해 보여주었습니다. 또한, VITAL 메트릭은 이러한 키 정보 오류를 더 신뢰성 있게 감지할 수 있는 능력을 입증하였습니다.

- **Performance Highlights**: VITAL 메트릭은 이전의 방법들보다 키 정보 오류를 더 잘 감지할 수 있으며, 이러한 점에서 특히 응답 평가에 있어 유의미한 차이를 보여줍니다. 실험 결과, 기존의 사실성 평가 지표들은 중요 정보를 고려하지 못하여 높은 정밀도(precision)와 재현율(recall) 점수를 기록하였으나, VITAL 메트릭을 통해 이러한 문제를 해결했습니다. 이 연구는 LLM의 사실성 평가를 위한 더 정확하고 견고한 기반을 제공합니다.



### Accelerating Diffusion LLM Inference via Local Determinism Propagation (https://arxiv.org/abs/2510.07081)
Comments:
          21 pages, 4 figures. Under review

- **What's New**: 이번 논문에서는 Diffusion Large Language Models (dLLMs)가 텍스트 생성에서 중요한 발전을 이루었지만, 품질과 속도 간의 균형 문제가 실제 배포에 장애가 되고 있음을 지적합니다. 기존의 오픈소스 구현은 주로 가장 신뢰할 수 있는 토큰만을 선택하는 보수적인 샘플링 전략에 의존하며, 이는 반복적으로 불필요한 수정 단계를 초래하여 효율성을 저하시킵니다. 이를 해결하기 위해 저자들은 LocalLeap이라는 훈련이 필요 없는 적응형 병렬 디코딩 전략을 제안했습니다.

- **Technical Details**: LocalLeap은 높은 신뢰도의 앵커(anchor) 주위 지역에서의 국소 결정론 전파(local determinism propagation)와 점진적 공간 일관성 감소(progressive spatial consistency decay)를 기반으로 하여 토큰을 디코딩합니다. 이 방법은 앵커 주변의 지역에서 유연한 병렬 디코딩을 수행하고, 이미 결정된 토큰에 대해 조기 약속을 하여 불필요한 수정 단계를 제거합니다. LocalLeap은 각 앵커 주위의 제한된 범위 내에서 토큰을 디코딩하여 전체적인 추론 단계를 대폭 줄입니다.

- **Performance Highlights**: LocalLeap을 다양한 벤치마크에서 평가한 결과, 6.94배의 처리량 향상을 달성하고, 디코딩 단계를 기존의 14.2%로 줄임으로써 모델 성능을 거의 저하 없이 개선했습니다. 예를 들어, LocalLeap은 GSM8K 벤치마크에서 LLaDA-Instruct 모델을 사용하여 약간의 성능 향상을 이루면서도 필요한 추론 단계 수를 덜어냈습니다.



### LuxInstruct: A Cross-Lingual Instruction Tuning Dataset For Luxembourgish (https://arxiv.org/abs/2510.07074)
Comments:
          Paper under review; Dataset available at this https URL

- **What's New**: 이번 연구는 룩셈부르크어(Luxembourgish)와 같은 저자원 언어를 위한 크로스링구얼(다국어) 지침 튜닝 데이터셋을 구축하여, 고품질의 언어 데이터를 제공하는 것을 목표로 하고 있습니다. 기존의 기계 번역(Machine Translation) 방식 대신에, 영어, 프랑스어, 독일어와 정렬된 데이터를 활용하여 언어적 및 문화적 뉘앙스를 유지한 데이터셋을 생성하였습니다. 이는 기계 번역 데이터의 일반적인 문제점을 회피하면서도 저자원 언어의 발전에 직접적인 혜택을 줄 수 있음을 입증합니다.

- **Technical Details**: 기존의 지침 데이터셋 요구는 고품질 언어 리소스가 부족한 룩셈부르크어처럼 저자원 언어에서 큰 도전 과제가 되고 있습니다. 연구자들은 주로 기계 번역 기술에 의존하여 지침 데이터를 생성했지만, 이는 의미의 미스알ignment 및 문화적으로 부적절한 응답을 초래할 수 있었습니다. 이를 해결하기 위해, 연구팀은 Wikipedia, 뉴스 기사, 그리고 온라인 사전을 주요 소스로 사용하는 역 지침 생성 방법을 채택하여 고품질의 데이터를 수집하였습니다.

- **Performance Highlights**: 테스트 결과, 크로스링구얼 지침 튜닝이 룩셈부르크어의 표현적 정렬을 향상시키고, 모델의 생성 능력을 향상시키는 데 기여하는 것으로 나타났습니다. 연구팀은 다양한 언어로의 지침을 포함한 데이터셋을 사용하여 벤치마크 실험을 수행하였고, 그 결과 크로스링구얼 접근 방식이 표현력 및 지침 응답 능력에서 효과적이라는 것을 확인하였습니다. 이는 룩셈부르크어 LLM의 개발에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Revisiting Metric Reliability for Fine-grained Evaluation of Machine Translation and Summarization in Indian Languages (https://arxiv.org/abs/2510.07061)
Comments:
          18 pages, 14 figures

- **What's New**: 이 논문에서는 인도 언어의 기계 번역(Machine Translation, MT) 및 텍스트 요약(Text Summarization, TS) 시스템 평가를 위한 ITEM(Indian Text Evaluation Metrics Testbed)이라는 대규모 벤치마크를 소개합니다. 이 벤치마크는 6개의 주요 인도 언어에서 26개의 자동 평가 지표와 인간의 판단 간의 일치를 체계적으로 평가합니다. 이를 통해 기존의 평가 지표가 인도 언어에 대한 유니버설성과 신뢰성에 의문을 제기하며, 그에 대한 중요한 통찰을 제공합니다.

- **Technical Details**: ITEM은 힌디어, 벵골어, 마라티어, 구자라티어, 타밀어, 텔루구어 등 6개의 인도 언어로써 MT와 TS의 자동 지표와 인간 평가 간의 일치를 평가하기 위해 설계되었습니다. 각 언어마다 전문적으로 주석이 달린 데이터 세트를 통해 150개의 기사-요약 쌍과 150개의 문장-번역 쌍을 무작위로 샘플링했습니다. 이 연구는 26개 자동 지표의 성능을 평가하고, 언어별 신뢰성과 민감도 분석을 포함하여 다양한 평가 차원을 검토합니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM 기반 평가자가 인간 판단과 가장 강력한 일치를 보였으며, 이상의 영향력이 평가 지표-인간 간 일치에 큰 영향을 미친다는 것이 밝혀졌습니다. 텍스트 요약에서는 콘텐츠 충실도를 더욱 효과적으로 포착하는 반면, 기계 번역에서는 유창성을 더 잘 반영하는 경향이 있습니다. 최종적으로, 인도 언어의 평가 지표 설계 및 개선을 위한 중요한 지침을 제공하여 앞으로의 연구 방향을 제시합니다.



### Does Local News Stay Local?: Online Content Shifts in Sinclair-Acquired Stations (https://arxiv.org/abs/2510.07060)
- **What's New**: 이 논문은 Sinclair 방송 그룹이 인수한 지역 뉴스 방송이 어떻게 보도 내용을 변화시키는지를 조사합니다. 설문 조사에 따르면, 지역 뉴스는 정치적 편향이 적으면서 지역 사회의 주요 문제를 다루는 신뢰할 수 있는 정보 제공원으로 여겨집니다. 그러나 Sinclair 소속 방송국은 인수 이전보다 정치적 주제에 대해 더 많은 보도를 하고 있다는 명확한 증거가 발견되었습니다.

- **Technical Details**: 연구진은 뉴스 방송국의 YouTube 채널에서 수집한 데이터셋을 바탕으로 보도 내용의 변화를 분석했습니다. 그들은 'corpus analysis' 방법과 단어 선택 비교, 주제 모델링, 그리고 'word embeddings' 분석을 사용하여 보고의 전반적인 변화를 측정했습니다. 특히, Sinclair가 인수한 방송국에서 정치적으로 논란이 있는 주제들의 보도가 증가하는 경향이 관찰되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, Sinclair에 인수된 방송국에서는 지역 주제에서 국가 주제로의 보도 전환이 뚜렷하게 이루어졌으며, 정치적 주제에 대한 보도가 증가했습니다. 이러한 변화는 지역 사회 중심의 뉴스가 급격히 감소하고 있다는 우려를 강조하면서, 시청자에게 미치는 영향에 대한 이해를 돕는 기여를 합니다. 또한 이러한 경향은 지역 정치 참여를 감소시키고 사회의 정치적 양극화를 심화시킬 수 있습니다.



### Search-R3: Unifying Reasoning and Embedding Generation in Large Language Models (https://arxiv.org/abs/2510.07048)
- **What's New**: 이번 연구에서는 Search-R3라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLM)을 검색 작업에 적합하도록 적응시켜, 해결 과정의 직접적인 결과로 검색 임베딩(search embeddings)을 생성합니다. Search-R3는 LLM의 단계적 사고(chain-of-thought) 능력을 활용하여 복잡한 의미 분석을 통해 더 효과적인 임베딩을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: Search-R3는 세 가지 보완 메커니즘을 통해 구현됩니다. 첫째, 감독 학습(supervised learning) 단계는 모델이 품질 높은 임베딩을 생성할 수 있도록 돕습니다. 둘째, 강화 학습(reinforcement learning, RL) 방법론이 임베딩 생성과 추론을 최적화하며, 셋째, 진화하는 임베딩 표현을 효과적으로 처리하는 특화된 RL 환경을 제공하여 각 훈련 Iteration마다 전체 코퍼스의 재부호화 없이 임베딩을 다룰 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에 대한 포괄적인 평가를 통해, Search-R3가 기존 방법들보다 훨씬 우수한 성능을 발휘함을 확인했습니다. 이 통합된 POST-training 접근 방식은 복잡한 지식 중심의 작업을 처리하는 데 있어 획기적인 진전을 이루며, 정교한 추론과 효과적인 정보 검색을 모두 요구하는 시나리오에서 중요한 기여를 합니다.



### Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models (https://arxiv.org/abs/2510.07037)
- **What's New**: 이 논문은 코드 스위칭(Code-switching, CSW)을 인지하는 대형 언어 모델(Large Language Models, LLMs) 연구의 최초의 포괄적인 분석을 제공하며, 다양한 언어 쌍에 대한 12개 NLP 작업, 30개 이상의 데이터셋 및 80개 이상의 언어를 커버하고 있습니다. 연구는 LLM이 CSW 모델링을 어떻게 혁신했는지와 여전히 남아 있는 도전 과제를 개관합니다. 더불어, 포괄적인 데이터셋, 공정한 평가, 언어학적으로 근거가 있는 모델의 필요성을 강조하는 로드맵을 제시합니다.

- **Technical Details**: 코드 스위칭 연구의 진화는 NLP의 주요 이정표와 일치합니다. 코드 스위칭을 위한 모델은 초기 통계적 및 규칙 기반 접근법에서 LLM 시대의 NLG, 음성 및 다중 모드 도메인으로 전환되었으며, 좀 더 유창함을 보여주지만 사실 일관성 및 저자원 성능에서는 여전히 어려움을 겪고 있습니다. 이 논문은 마침내 코드 스위칭 인식을 향상시키기 위한 통합된 모델 학습 방법과 평가 프레임워크의 필요성을 강조합니다.

- **Performance Highlights**: LLM들은 코드 스위칭 시나리오에 대해 제로샷 전이(Zero-shot transfer)에서 어려움을 겪고 있으며, 일반적으로 다국어 LLM은 잘 조정된 작은 모델보다 성능이 떨어질 수 있습니다. 또한 비영어 텍스트가 영어 컨텍스트에서 성능을 저하시킬 수 있으며, 반대로 영어 토큰이 다른 언어에서는 성능을 향상시키는 경향이 있습니다. 저자원 언어에 대한 제한된 사전 학습 데이터가 이러한 비대칭 성능을 가중시키고 있습니다.



### Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledg (https://arxiv.org/abs/2510.07024)
- **What's New**: 논문은 사실 기반 지식에 대한 심층 분석을 수행하여 챗봇 및 자연어 처리(NLP)에서의 LLM(대규모 언어 모델)의 역할을 재조명합니다. GPT-4.1에 기반한 GPTKB v1.5를 통해 1억 개의 신념을 조사한 결과, 이러한 모델이 기존 지식 베이스와 상당히 다른 지식을 가지고 있음이 드러났습니다. 이 연구는 LLM의 지식 정확도가 기존 벤치마크에서 제시된 것보다 낮음을 강조하며, 이로 인해 LLM의 연구에서 발생할 수 있는 많은 기회가 열리게 됩니다.

- **Technical Details**: 연구는 GPT-4.1과 함께 사용된 GPTKB v1.5를 통해, 대규모로 Recursive Knowledge Mining 기법을 이용하여 지식을 추출했습니다. 이 방법은 기존 데이터의 재구성을 넘어서며, 100M 이상의 사실적 주장을 포함하는 지식 베이스를 생성하는 데 14,000달러의 비용이 들었습니다. 연구진은 LLM의 지식 조직 방식이 STEM(과학, 기술, 공학 및 수학) 중심으로 되어 있음을 발견하여, LLM의 데이터와 이해도가 사회 및 인문학보다 과학 기술 관련 지식에 좀 더 편향되어 있음을 보여주었습니다.

- **Performance Highlights**: GPT-4.1 모델의 사실적 지식 정확도는 75%로, 기존 텍스트 추출 기반의 지식 베이스보다 높지만, 인간이 편집한 자원 및 일반 LLM 벤치마크에서는 여전히 낮습니다. 분석 결과에서는 일관성, 애매성 및 환각 현상이 주요 문제로 드러났으며, 이는 LLM의 사실적 지식과 관련된 도전 과제와 미래 연구 기회를 제시합니다. GPTKB v1.5는 이러한 LLM의 지식 또는 신념을 심층적으로 조사하는 데 있어 유일무이한 자원으로 자리매김하고 있습니다.



### Native Hybrid Attention for Efficient Sequence Modeling (https://arxiv.org/abs/2510.07019)
Comments:
          Technical report, 16 pages

- **What's New**: 본 연구에서 우리는 Native Hybrid Attention (NHA)라는 새로운 하이브리드 아키텍처를 소개합니다. NHA는 선형 주의와 풀 주의(intra & inter-layer hybridization)를 통합하여 단일 계층 설계로 구성되어 있습니다. 이 구조는 긴 문맥을 유지하면서도 단기간의 정보를 효과적으로 결합하여 성능 향상을 이룹니다.

- **Technical Details**: NHA는 선형 RNN에 의해 업데이트된 키-값 슬롯(key-value slots)을 통해 긴 문맥을 관리합니다. 이는 슬라이딩 윈도우(Sliding Window)의 짧은 기간 토큰들과 결합되어, 단일 softmax attention 작업을 통해 모든 키와 값에 적용됩니다. 각 레이어의 동작은 슬라이딩 윈도우 크기라는 단일 하이퍼파라미터로 조절되어, 모델 구조를 변경하지 않고도 선형 주의와 풀 주의 간의 균형을 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과 NHA는 리콜이 중요한 작업 및 상식 추론(test tasks)에서 Transformers 및 기타 하이브리드 모델을 초월하는 성능을 보여주었습니다. 사전 훈련된 LLM을 NHA 구조에 적용한 결과, 경쟁력 있는 정확도를 달성하며 효율성을 크게 향상시켰습니다. 코드 및 모델은 제공된 URL에서 확인할 수 있습니다.



### Pragyaan: Designing and Curating High-Quality Cultural Post-Training Datasets for Indian Languages (https://arxiv.org/abs/2510.07000)
Comments:
          EMNLP 2025

- **What's New**: 이 논문에서는 인도 언어를 포함한 다국어 고품질 후처리 데이터셋인 Pragyaan-IT와 Pragyaan-Align을 소개합니다. 두 데이터셋은 각각 22.5K와 100K의 예시로 구성되어 있으며, 10개 인도 언어에서 13개 대분류와 56개 소분류를 포함합니다. This approach emphasizes not only linguistic accuracy but also cultural relevance, addressing a significant gap in existing datasets that often ignore local contexts.

- **Technical Details**: 연구진은 효율적이고 질 높은 후처리 데이터셋을 생성하기 위해 human-in-the-loop (HITL) 파이프라인을 활용했습니다. 이는 instruction-following (지시 수용) 작업의 다양성과 문화적 뉘앙스를 중시하며, 다양한 현실 세계 시나리오를 처리할 수 있는 모델을 훈련시키는 데 중점을 두고 있습니다. 데이터셋은 다국어 표현력, 작업 복잡성, 그리고 멀티 턴 대화를 포함한 여러 차원에서 특성을 포괄합니다.

- **Performance Highlights**: 이 연구의 결과로 생성된 Pragyaan 데이터셋은 인도 언어와 문화적 맥락을 밀접하게 반영하는 고품질 데이터를 제공합니다. 우선, 데이터셋은 실제 요구 사항을 바탕으로 구성되어 LLM이 다양한 지시를 잘 수용할 수 있도록 설계되었습니다. 그리고 이를 통해, 후속 실험에서는 데이터셋이 robust한 instruction-following 기능을 갖추었음을 검증하였습니다.



### Towards Reliable Retrieval in RAG Systems for Large Legal Datasets (https://arxiv.org/abs/2510.06999)
Comments:
          Accepted for the 7th Natural Legal Language Processing Workshop (NLLP 2025), co-located with EMNLP 2025

- **What's New**: 이 논문에서는 대규모 법률 문서에 대한 Retrieval-Augmented Generation (RAG) 시스템의 신뢰성을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히, Document-Level Retrieval Mismatch (DRM)라는 중요 실패 모드를 정의하고 그 문제를 해결하기 위한 Summary-Augmented Chunking (SAC) 기법을 소개합니다. SAC는 각 텍스트 청크에 문서 수준의 합성 요약을 추가하여 전반적인 글로벌 컨텍스트를 보존하는 방법론입니다.

- **Technical Details**: RAG 시스템은 두 개의 주 단계로 구성됩니다: 정보 검색기(retriever)와 생성기(generator) 모델입니다. 검색기는 사용자의 질의에 대한 관련 텍스트 조각을 찾고, 생성기는 이 조각들을 기반으로 최종 답변을 합성하여 출력합니다. 본 연구에서는 SAC 기법을 활용하여 신뢰할 수 있는 텍스트 코퍼스의 청크를 문서 수준의 요약으로 보강함으로써 이전 알고리즘의 한계를 극복하고자 합니다.

- **Performance Highlights**: 실험 결과, SAC 방법이 법률 질문-답변 작업에서 DRM을 상당히 줄였으며, 텍스트 수준의 검색 정확도와 재현율 또한 개선되었음을 확인하였습니다. 흥미롭게도, 법률 전문 도메인 지식을 포함한 방법보다 일반적인 요약 전략이 더 나은 성능을 보였다는 점이 주목됩니다. 이번 연구는 대규모 법률 문서 데이터셋에 적용했을 때 RAG 시스템의 신뢰성을 높일 수 있는 실용적이고 통합 가능한 기술의 필요성을 강조합니다.



### Probing Social Identity Bias in Chinese LLMs with Gendered Pronouns and Social Groups (https://arxiv.org/abs/2510.06974)
- **What's New**: 이 연구는 중국 대형 언어 모델(LLMs)에서 사회 정체성 편향을 조사합니다. 10개의 대표적인 중국 LLM을 대상으로 한 연구로, '우리는' (ingroup)과 '그들은' (outgroup) 프레이밍을 통해 다양한 사회 그룹의 반응을 평가했습니다. 이 연구는 기존 영어 중심 연구의 결과가 중국어와 같은 다른 언어 및 문화적 맥락에서도 유사하게 나타남을 보여줍니다.

- **Technical Details**: 연구는 Hu et al. (2025)의 방법론을 기반으로 하여, LLM 출력에서 ingroup-positive 및 outgroup-negative 경향을 평가합니다. 각 모델에 대해 2,000개의 문장을 생성하고, 성별에 따른 편향을 분석하기 위해 '우리는'과 남성/여성 형식의 '그들은' 프롬프트를 사용했습니다. 이 데이터셋은 총 240,000개의 문장을 포함하고 있으며, 다양한 프레이밍이 모델의 출력을 어떻게 변화시키는지를 분석합니다.

- **Performance Highlights**: 연구 결과, 중국 LLM들은 시스템적으로 ingroup에 긍정적이고 outgroup에 부정적인 경향을 보였습니다. 이러한 편향은 자연어 대화에서도 관찰되어, LLM이 사용자와의 실제 상호작용에서보다 강화될 수 있음을 시사했습니다. 이 연구는 중국 사회의 240개 사회 그룹을 포함하여, 모델의 반응을 보다 세분화하여 분석함으로써 편향 완화의 필요성을 강조합니다.



### EDUMATH: Generating Standards-aligned Educational Math Word Problems (https://arxiv.org/abs/2510.06965)
Comments:
          32 pages, 15 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)이 학생의 흥미와 능력 수준에 맞춰 수학 단어 문제(Math Word Problems, MWPs)를 자동으로 생성할 수 있음을 제안합니다. 이를 위해 연구팀은 11,000개 이상의 MWPs를 평가하기 위해 인간 전문가와 LLM을 결합한 평가 방식을 사용하였으며, 교수 표준에 맞춘 교육용 MWP 생성을 위한 최초의 교사 주석 데이터 세트를 개발하였습니다. 연구 결과, 대규모 언어 모델이 더 큰 모델과 동등한 성능을 발휘할 수 있는 가능성을 보여주었습니다.

- **Technical Details**: 이 연구에서는 수학 교육 기준과 학생의 흥미 및 능력 수준에 적합한 MWP 생성을 위해 네 가지 기준, 즉 solvability, accuracy, educational appropriateness, standards alignment를 사용하여 LLM의 성능을 평가합니다. 연구는 일정한 단계로 진행되며, 첫 번째 단계에서 교사들이 주석을 단 데이터셋을 사용하여 3,000개 이상의 MWPs를 생성하고, 이를 기반으로 12B와 30B LLM을 훈련시킵니다. 이를 통해 생성된 문제들의 품질을 높이고, 데이터를 활용하여 교사가 바라는 MWP 생성을 위한 새로운 데이터셋을 제공합니다.

- **Performance Highlights**: 연구 결과, LLM들이 생성한 MWPs는 기존 모델보다 더 높은 품질을 보이며, 인간이 작성한 MWPs와 더 유사한 것으로 나타났습니다. 또한, 학생들이 LLM이 생성한 MWP를 인간이 작성한 문제 보다 선호하는 경향을 보였으며, 이 질문들에서 비슷한 성과를 내는 것으로 분석되었습니다. 연구의 주요 기여점으로는 LLM의 성능 격차를 줄이고, 최초의 표준 맞춤형 교육용 MWP 생성 데이터셋을 만들어 내며, K-12 교육에서의 활용 가능성을 입증한 것이라 할 수 있습니다.



### Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation (https://arxiv.org/abs/2510.06961)
Comments:
          Submitted to ICASSP 2026; Leaderboard: this https URL Code: this https URL

- **What's New**: 이번 논문은 Open ASR Leaderboard를 소개하며, 60개 이상의 오픈소스 및 상용 ASR 시스템을 11개의 데이터 세트에서 비교하는 벤치마크를 제공합니다. 표준화된 텍스트 정규화 방법을 통해 단어 오류율(Word Error Rate, WER)과 역 실시간 요인(Inverse Real-Time Factor, RTFx)을 보고하여 정량적인 평가를 가능하게 하였습니다. 이는 개발자와 사용자 모두가 성능 및 효율성을 기반으로 한 평가를 보다 공정하게 할 수 있게 합니다.

- **Technical Details**: Open ASR Leaderboard는 영어 전사와 다국어 전사(독일어, 프랑스어, 이탈리아어 등) 그리고 30초 이상의 긴 오디오에 대한 평가를 포함합니다. 모델은 WER에 따라 평가되며 실시간 요인을 추정할 수 있도록 정리된 데이터 세트를 활용하여 숫자 정규화, 철자 표준화와 같은 텍스트 정규화를 거칩니다. 현재 64개의 모델이 등록되어 있으며, 그 중 57개가 오픈소스입니다.

- **Performance Highlights**: 짧은 영어 전사에서 Conformer 인코더와 LLM 기반 디코더의 조합이 우수한 성과를 보입니다. 그러나 이러한 구조는 TDT와 CTC 디코더를 사용하는 모델보다 느린 속도를 보이는 경향이 있습니다. 또한 자기 지도 학습(self-supervised learning, SSL) 기반의 모델이 여전히 뛰어난 성능을 발휘할 수 있지만, 현재 영어 전사 용 TOP SSL 시스템은 A100 성능 평가에서 상대적으로 낮은 순위를 차지하고 있습니다.



### SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models (https://arxiv.org/abs/2510.06917)
Comments:
          Work in progress

- **What's New**: SHANKS는 사용자가 말을 하는 동안 SLM이 사고를 할 수 있는 프레임워크를 제안합니다. 전통적인 언어 모델은 사용자의 입력이 완료된 후에만 생각을 시작하는 데 비해, SHANKS는 입력된 음성을 일정한 조각으로 나누어 처리하며, 조각이 수신되는 즉시 내부적인 사고를 합니다. 이 방식은 사용자와의 실시간 상호작용의 질을 높이는 데 기여합니다.

- **Technical Details**: SHANKS는 사용자의 입력 음성을 고정된 크기의 조각으로 나누어 실시간 스트리밍 방식으로 처리합니다. 사용자가 연속해서 말을 하는 동안, SHANKS는 각 음성 조각에 기초하여 주어진 이전의 음성과 사고 내용으로부터 내부적 사고를 생성합니다. 이 사고 과정에서 SLM은 사용자를 방해하거나 도구 호출을 할지 여부를 결정할 수 있습니다.

- **Performance Highlights**: SHANKS는 두 가지 시나리오에서 실시간 상호작용을 개선하는 성능을 보여줍니다. 첫 번째로, 수학 문제에 대한 단계별 해결 과정을 설명하는 동안 SHANKS는 실시간으로 잘못된 답변을 제시하는 사용자에게 37.1% 높은 정확도로 개입할 수 있습니다. 두 번째로, 도구가 포함된 대화 상황에서 SHANKS는 사용자가 말을 하는 동안 56.9%의 도구 호출을 성공적으로 완료하여 응답 대기 시간을 줄입니다.



### LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling (https://arxiv.org/abs/2510.06915)
- **What's New**: 이 논문에서는 Long-RewardBench라는 새로운 벤치마크를 도입하여, 보상 모델(Reward Model, RM)을 긴 콘텍스트에서 평가할 수 있는 방법을 제시합니다. 기존 RM들이 짧은 콘텍스트에서만 잘 작동하고, 긴 콘텍스트에서의 일관성을 간과한다는 한계를 지적합니다. 이를 통해, 다양한 모델의 긴 콘텍스트에 대한 이탈성 높은 평가가 이루어질 수 있도록 하고자 하였습니다.

- **Technical Details**: Long-RewardBench는 각 테스트 세트가 질문(question), 콘텍스트(context), 모델 응답(set of model responses), 정답(ground-truth prediction)의 네 가지 요소로 구성된 벤치마크입니다. 이 연구에서는 일반적인 다단계 학습 전략을 설계하여 기존 모델을 긴 컨텍스트의 보상 모델(LongRMs)로 확장하기 위한 방법론을 개발하였습니다. 특히, 데이터 합성을 통해 훈련 과정의 각 단계에서 높은 품질의 데이터를 생성하는 방식이 도입되었습니다.

- **Performance Highlights**: 예비 연구 결과에 따르면, 기존의 최신 생성 모델들이 긴 콘텍스트 상황에서 성능이 크게 때때로 50% 미만으로 떨어진다는 문제가 드러났습니다. 그러나 제안된 방법론으로 훈련된 8B LongRM 모델은 70B 기준 모델들보다 우수한 성능을 보이며, 비공식 모델인 Gemini 2.5 Pro와 동등한 성능을 달성하였습니다. 이러한 결과는 짧은 콘텍스트에서의 성능 또한 유지되고 있음을 시사합니다.



### MeXtract: Light-Weight Metadata Extraction from Scientific Papers (https://arxiv.org/abs/2510.06889)
- **What's New**: 이 논문에서는 MeXtract라는 경량 언어 모델 계열을 소개하여 과학 논문에서 메타데이터( metadata)를 추출하는 데 혁신적인 접근 방식을 제안합니다. MeXtract는 Qwen 2.5 모델을 미세 조정하여 개발된 0.5B에서 3B까지의 다양한 매개변수를 가진 모델로, MOLE 벤치마크에서 최첨단 성능을 달성했습니다. 또한, 모델 특화 메타데이터를 포함하는 새로운 MOLE 벤치마크로 평가를 지원하는 방법도 제시합니다.

- **Technical Details**: 메타데이터는 JSON 형식으로 키와 값의 쌍을 사용하여 정의됩니다. 추출 접근 방식에서는 세 가지 변수를 정의하는 스키마(schema)가 존재하며, 각 변수는 타입(type), 길이(length), 옵션(options)으로 구성됩니다. MeXtract의 모델은 긴 맥락에서도 메타데이터를 효과적으로 추출할 수 있으며, 그 성능은 기존의 대규모 모델들과 비교했을 때 우수합니다.

- **Performance Highlights**: MeXtract는 메타데이터 추출에서 기존의 비슷한 크기 모델보다도 뛰어난 성능을 입증하였으며, 이는 저자들이 제공한 파라미터 조정 및 최적화된 선택을 통해 달성되었습니다. 연구 결과는 데이터셋, 지침 조정 및 선호 최적화와 같은 다양한 아티팩트를 포함하고 있으며, 연구 커뮤니티를 위해 모든 모델과 데이터셋이 오픈 소스로 출시됩니다.



### $λ$-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences (https://arxiv.org/abs/2510.06870)
Comments:
          9 pages

- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 추론 능력을 향상시키기 위한 새로운 접근법으로, 강화 학습(Reinforcement Learning)에서 인간 피드백(Human Feedback)을 결합한 새로운 방식을 제안합니다. 기존의 Group Relative Policy Optimization (GRPO) 방법은 길이에 대한 편향(length bias) 문제를 가지고 있으며, 이에 대한 해결책으로 λ-GRPO을 도입하여 토큰의 선호도를 학습할 수 있도록 하였습니다. 이 방법은 기존의 GRPO 및 DAPO보다 더 나은 성능을 보여주며, 훈련 데이터나 추가 비용 없이도 효과적으로 동작합니다.

- **Technical Details**: 방법론 섹션에서는 LLM의 텍스트 생성 과정을 순차적 의사 결정 프로세스로 보고, 주어진 입력에 따라 다음 토큰을 생성하는 정책을 정의합니다. 연구자들은 모델이 안정적으로 학습하도록 하기 위해 이전 정책에서 꺼낸 샘플링을 사용하며, 보상 함수는 결정론적인 규칙에 의해 정해져 신뢰할 수 있는 피드백 신호를 제공합니다. 새로운 λ-GRPO 방법은 토큰의 기여도를 동적으로 조정하는 학습 가능한 파라미터를 도입하여, 모든 토큰에 대해 균일하게 적용되는 기존의 방법들과 차별화됩니다.

- **Performance Highlights**: λ-GRPO는 Qwen2.5 모델에서 각각 1.5B, 3B 및 7B 파라미터를 가진 모델을 대상으로 여러 추론 벤치마크에서 일관되게 향상된 정확도 성과를 보였습니다. GRPO와 비교할 때, λ-GRPO는 평균 정확도를 각각 +1.9%, +1.0%, +1.7% 향상시켰으며, 이는 기존의 방법들이 가지고 있던 단점을 극복하는 데 기여하고 있습니다. 이 연구는 모델이 스스로 토큰 선호도를 학습하게 함으로써, 더 나은 성능과 훈련 안정성을 증명해 보였습니다.



### Unlocking Latent Discourse Translation in LLMs Through Quality-Aware Decoding (https://arxiv.org/abs/2510.06866)
- **What's New**: 대규모 언어 모델(LLMs)이 기계 번역 분야에서 유망한 후보로 부상했지만, 여전히 담화 현상(discourse phenomena), 예를 들어 대명사 해결(pronoun resolution) 및 문서 수준의 어휘 응집(lexical cohesion)을 효과적으로 처리하는 데 어려움을 겪고 있습니다. 본 연구에서는 맥락 인식 번역(context-aware translation)에서 LLM의 담화 현상 성능을 철저히 조사하였습니다.

- **Technical Details**: 연구자는 LLM 내에 담화 지식이 인코딩(encoding)되어 있음을 보여주며, 질 인식 디코딩(quality-aware decoding, QAD)을 사용하여 이 지식을 효과적으로 추출하는 방법을 제안합니다. QAD는 다른 디코딩 접근 방식을 포괄적으로 분석하여 그 우수성을 입증합니다.

- **Performance Highlights**: QAD는 번역의 의미적 풍부함(semantic richness)을 향상시키고 인간의 선호(preferences)와 더 밀접하게 일치하도록 개선하는 것을 보여줍니다.



### OpenJAI-v1.0: An Open Thai Large Language Mod (https://arxiv.org/abs/2510.06847)
- **What's New**: OpenJAI-v1.0는 태국어와 영어를 위한 오픈 소스 대형 언어 모델로, Qwen3-14B 모델을 기반으로 개발되었습니다. 이 모델은 지침 이행, 긴 맥락 이해 및 도구 사용의 세 가지 핵심 사용 사례에 중점을 두어 실용적인 작업의 성능을 향상시키고자 합니다. OpenJAI-v1.0은 기존 모델보다도 성능을 높였으며, 다양한 벤치마크에서 주요 오픈 소스 태국어 모델을 능가하였습니다.

- **Technical Details**: OpenJAI-v1.0은 고품질 데이터셋을 기반으로 구성되었으며, 데이터는 지침-응답 형식으로 구성됩니다. 주요 기능에는 복잡한 지침 이행, 긴 맥락 이해, 그리고 외부 도구 및 API와 원활하게 통합할 수 있는 신뢰할 수 있는 도구 사용 등이 포함됩니다. 모델 학습은 Jasmine Technology Solution의 GPU 클러스터에서 진행되었으며, 총 462 백만 개의 토큰으로 훈련되었습니다.

- **Performance Highlights**: OpenJAI-v1.0은 다양한 벤치마크에서 성능 평가를 받았으며, 이를 통해 모델의 지침 이행과 긴 맥락 문제 해결 능력이 입증되었습니다. 특히, 태국어를 위한 벤치마크인 IFBench-TH와 다양한 다단계 상호작용을 평가하는 MT-Bench-TH에서 긍정적인 성과를 보였습니다. 이러한 평가 결과는 OpenJAI-v1.0의 실제적인 적용 가능성을 강화하는 데 기여하고 있습니다.



### SID: Multi-LLM Debate Driven by Self Signals (https://arxiv.org/abs/2510.06843)
- **What's New**: 이번 연구에서는 self signals(자기 신호)를 활용한 Self-Signals Driven Multi-LLM Debate(SID) 프레임워크를 제안합니다. 이는 모델 수준의 confidence(신뢰도)와 토큰 수준의 semantic focus(의미 중심)에 기반하여 토론 과정 및 성능을 개선하는 데 도움을 줍니다. 이 접근 방식은 에이전트들이 여유롭게 논의할 필요 없이 직접적인 경험에서 나오는 신뢰도를 활용하여 불필요한 중복 토론을 줄입니다.

- **Technical Details**: SID 프레임워크는 모델이 생성하는 과정에서 발생하는 자기 신호를 적극적으로 사용합니다. 모델 신뢰도를 판단하기 위해 생성된 답변의 확률 분포를 기반으로 하며, 이 정보를 바탕으로 조기 종료 메커니즘을 설계하여 토론이 불필요한 경우를 줄입니다. 또한 attention(주의) 메커니즘을 통해 토론 내용에서 의미 있는 부분을 강조하여 압축하고, 중요한 논점은 유지하면서 토큰 소비를 줄입니다.

- **Performance Highlights**: 예비 실험 결과, SID는 기존의 MAD 접근 방식보다 정확도에서 우수할 뿐만 아니라 최대 40%의 토큰 소비 감소를 달성하는 효과를 보였습니다. 이는 multi-agent 시스템에서 자기 신호를 활용할 때 성능과 효율성을 함께 최적화 할 수 있다는 가능성을 보여줍니다. 실험은 다양한 벤치마크 및 멀티모달 LLM에서 수행되어 SID의 효과가 입증되었습니다.



### GAMBIT+: A Challenge Set for Evaluating Gender Bias in Machine Translation Quality Estimation Metrics (https://arxiv.org/abs/2510.06841)
Comments:
          Accepted for publication at the 10th Conference of Machine Translation (WMT25), co-located with EMNLP 2025

- **What's New**: 본 논문에서는 기계 번역(Machine Translation, MT) 시스템에서 알려진 성별 편향(Gender Bias)에 대한 연구를 확장하여, 자동 품질 추정(Quality Estimation, QE) 메트릭에서도 성별 편향이 존재한다는 점을 다룹니다. 기존 연구들은 작은 데이터셋과 제한된 직업 범위로 인해 한계가 있었으나, 이번 연구는 이를 해결하기 위한 대규모 챌린지 세트를 소개합니다. 이 챌린지 세트는 성별 모호한 직업 용어를 포함한 번역을 평가하는 QE 메트릭의 행동을 조사하는 데 초점을 맞춥니다.

- **Technical Details**: 연구는 GAMBIT 코퍼스(GAMBIT corpus)를 기반으로 하여 성별에 대한 명확성이 없는 직업을 포함한 영어 텍스트를 사용했습니다. 이어서 성별이 없는 언어와 자연 성별 언어 3개, 그리고 문법적 성별이 있는 언어 11개로 범위를 확장하여 총 33개 소스-타겟 언어 쌍을 구성하였습니다. 각 소스 텍스트는 직업 용어의 문법적 성별(남성형 vs. 여성형)만이 다른 두 개의 타겟 버전과 연결되며, 모든 의존 문법 요소도 조정됩니다.

- **Performance Highlights**: 편향이 없는 QE 메트릭은 두 타겟 버전에 대해 동일하거나 거의 동일한 점수를 부여해야 합니다. 이번 데이터셋의 대규모, 폭넓은 특징, 그리고 모든 언어에 대해 동일한 텍스트 집합이 정렬된 완전 병렬 디자인을 통해 직업별 세밀한 편향 분석과 언어 간 체계적 비교가 가능해졌습니다. 이러한 분석을 통해 성별 편향을 더 잘 이해할 수 있는 기반이 마련될 것으로 기대됩니다.



### Mid-Training of Large Language Models: A Survey (https://arxiv.org/abs/2510.06826)
- **What's New**: 이번 논문은 대형 언어 모델(LLM) 훈련에서 중간 단계(mid-training)의 중요성을 강조합니다. 일반적 사전 훈련(pre-training)과 특정 작업에 대한 세밀 조정(fine-tuning) 사이에서 중간 단계가 모델 최적화 및 데이터 품질을 향상시키는 역할을 한다는 것을 보여줍니다. 저자들은 LLM 중간 훈련의 첫 번째 분류법과 이론적 기초를 제공하며, 다양한 접근 방식과 레퍼런스를 정리했습니다.

- **Technical Details**: 중간 훈련은 세 가지 주요 요소로 구성됩니다: 데이터 분포(data distribution), 학습률 스케줄(learning-rate scheduling), 그리고 컨텍스트 확장(context extension)입니다. 데이터의 혼합을 세밀하게 조정하여 고품질 샘플을 통해 모델이 추상화와 구조적 추론에 집중하도록 지원합니다. 학습률 조정은 수렴을 안정화시키고 고품질 데이터의 최적 활용을 도와줍니다. 또한, 긴 컨텍스트 입력을 통해 모델이 복잡한 작업을 효과적으로 처리할 수 있도록 돕습니다.

- **Performance Highlights**: 중간 훈련을 통해 모델은 개선된 일반화와 효율성을 얻을 수 있으며, 이는 장기적인 훈련에서 정보 병목 현상을 해소하고 복잡한 추론 능력을 강화하는 데 기여합니다. 논문에서는 다양한 모델 간의 비교를 가능하게 하는 평가 벤치마크를 정리하였으며, 중간 훈련의 효과적인 적용 방안을 제안합니다. 향후 연구 방향과 가능성에 대해서도 논의하여 LLM 중간 훈련 관련 연구의 발전을 촉진하고자 합니다.



### Adaptive Tool Generation with Models as Tools and Reinforcement Learning (https://arxiv.org/abs/2510.06825)
- **What's New**: MTR(Model-as-Tool Reasoning)는 도구 강화 추론을 위한 새로운 시뮬레이션 기반 훈련 프레임워크입니다. 이 시스템은 라이브 API에 의존하는 대신, 완전한 ReAct 트레이스와 스키마 검증된 시뮬레이션 관찰로부터 학습합니다. MTR은 내부의 ToolMaker, AutoAgent, ToolActor 등 다중 에이전트 아키텍처를 통해 작업별 도구 인터페이스를 생성하고, 구조화된 사고-행동-관찰 시퀀스를 생산하여 현실적인 응답을 시뮬레이션합니다.

- **Technical Details**: MTR의 훈련은 두 단계로 진행됩니다. 1단계는 감독 세부 조정(Supervised Fine-Tuning, SFT)으로, 완전한 추론 시퀀스에서 '트레이스 문법'을 학습합니다. 2단계는 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)로, 답변의 정확성과 내부 일관성의 균형을 맞춘 복합 트레이스 보상을 통해 전략을 최적화합니다.

- **Performance Highlights**: MTR은 HotpotQA, MuSiQue, 2WikiMultiHopQA, Bamboogle 등 네 가지 다단계 QA 기준에서 경쟁력 있는 Exact Match(EM) 점수를 달성했습니다. 특히, 고도의 추론 작업에서 Live-API 시스템보다 확연한 성과를 보였으며, Bamboogle의 경우 40.0%의 정확도를 달성하여 기존의 가장 강력한 기준 점수인 33.3%를 초과했습니다.



### BlackboxNLP-2025 MIB Shared Task: Exploring Ensemble Strategies for Circuit Localization Methods (https://arxiv.org/abs/2510.06811)
Comments:
          The 8th BlackboxNLP Workshop (Shared Task), 6 pages

- **What's New**: 본 연구는 Mechanistic Interpretability Benchmark (MIB)의 Circuit Localization 트랙에서 회로(localization)를 더 잘 식별하기 위한 여러 방법의 집합(ensemble)을 평가합니다. 두 가지 변형인 병렬(Parallel) 및 순차적(Sequential) 집합을 통해 성능을 향상시킬 수 있는지 조사했으며, 이를 통해 더 정밀한 회로 식별 방식이 가능함을 보여줍니다.

- **Technical Details**: 연구에서 사용된 방법은 EAP-IG (Edge Attribution Patch with Integrated Gradients)와 edge pruning을 포함합니다. 병렬 집합에서는 각 방법에서 할당된 score을 평균 또는 최대/최소를 통해 결합하며, 순차적 집합에서는 EAP-IG score를 이용하여 더 정밀한 edge pruning 방법을 가속화합니다. 이러한 접근 방식을 통해 다양한 모델/task 조합의 성능을 평가했습니다.

- **Performance Highlights**: 혼합 집합(hybrid ensemble) 방법이 성능 metricks에서 최상의 결과를 보였으며, Circuit Performance Ratio (CPR)와 Circuit-Model Difference (CMD) 점수에서 각각 높은 점수를 기록했습니다. 특히, hybrid ensemble 방법이 가장 낮은 CMD 점수와 가장 높은 CPR 점수를 달성해, 주어진 작업에서의 전반적인 성과를 극대화했습니다.



### Overview of the Plagiarism Detection Task at PAN 2025 (https://arxiv.org/abs/2510.06805)
Comments:
          Working Notes at PAN at CLEF 2025

- **What's New**: PAN 2025에서는 과학 기사에서 자동 생성된 텍스트 표절을 식별하는 새로운 데이터셋을 구성했습니다. 이 데이터셋은 Llama, DeepSeek-R1, Mistral이라는 세 가지 대형 언어 모델을 활용하여 자동으로 생성된 표절 사례를 포함하고 있습니다. 본 연구는 이 새로운 데이터셋의 구조와 참여자들의 성과를 정리하며, 2015년도 표절 탐지 과제를 재평가합니다.

- **Technical Details**: PAN 2025는 2015년 대회와 유사한 평가 방법론과 데이터셋 형식을 유지합니다. 참가자는 S(출처 문서)와 P(표절 문서) 쌍을 이용해 LLM으로 생성된 표절 구문을 식별하고 정렬하는 작업을 수행합니다. 기본적으로 arXiv에서 가져온 100,000개의 문서로부터 생성된 데이터셋은 세 가지 세부 범주로 나누어져 있어 다양한 성능 분석이 가능하도록 구성되었습니다.

- **Performance Highlights**: 신규 데이터셋을 활용한 2025 PAN 과제에서는 기존 기준선 모든 결과를 초과하는 네 가지 제출물이 있었습니다. Linq-Embed-Mistral 모델을 사용한 기준선이 가장 뛰어난 성과를 보였는데, 이는 표절 탐지와 같은 텍스트 검색 작업에 적합한 특수 모델이 효과적임을 나타냅니다. 현재까지의 결과는 LLM을 통한 자동 표절 처리가 일반성에 한계가 있음을 시사합니다.



### FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipelin (https://arxiv.org/abs/2510.06800)
- **What's New**: 새로운 논문에서 FURINA-Builder라는 혁신적인 멀티 에이전트 협업 파이프라인을 소개합니다. 이는 사용자가 다양한 캐릭터와 시나리오에 맞춰 완전히 커스터마이즈된 RP 벤치마크를 자동으로 구축할 수 있게 돕습니다. 또한, FURINA-Bench라는 포괄적인 RP 벤치마크를 구축하여 각 캐릭터와 평가 기준에 대한 평가를 제공하는 것이 특징입니다.

- **Technical Details**: FURINA-Builder는 테스트 캐릭터와 다양한 캐릭터 간의 대화를 시뮬레이션하여 RP 작업을 평가하는 데 필요한 다양한 캐릭터 데이터베이스를 생성합니다. LLM 기반 평가 모델이 응답의 정확성을 평가하고, 다양한 평가 기준으로 최소한의 기준을 설정하여 최종 발화를 선택합니다. 이러한 메커니즘을 통해 사용자들은 특정 시나리오에 적합한 RP 벤치마크를 보다 효과적으로 구축할 수 있습니다.

- **Performance Highlights**: FURINA-Bench의 광범위한 평가를 통해 최신 LLM들이 성능을 발휘하는 다양한 결과를 얻었습니다. 특히, o3와 DeepSeek-R1이 각각 영어 및 중국어 RP 작업에서 가장 뛰어난 성능을 보였으며, 기존 캐릭터가 합성 캐릭터보다 일관되게 높은 성능을 나타냈습니다. 그러나 흥미롭게도, 추론 기능이 RP 성능을 개선하지만 동시에 RP 환상을 증가시키는 잔여트레이드오프를 발견하였고, 이는 모든 LLM의 RP 성능과 신뢰성 사이의 보다 넓은 Pareto 경계로 확장됩니다.



### Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness (https://arxiv.org/abs/2510.06780)
- **What's New**: 본 연구에서는 miniGPTKB라는 개념을 도입하여 개별 도메인에 특화된 LLM 지식의 트래킹을 수행합니다. 또한, 기존 GPTKB 접근 방식의 종료 가능성을 입증하고, 재현성과 강건성에 대한 실험 결과를 보고합니다. 이와 함께 특정 LLM의 사실적 지식의 핵심에 대한 안정적인 시각을 제공할 수 있음을 주장합니다.

- **Technical Details**: 이 연구는 LLM 지식 수집 과정에서의 종료 가능성, 재현성 및 강건성을 평가하기 위해 miniGPTKB를 사용하는 방법론을 채택합니다. 세 가지 예시 도메인으로 고대 바빌론(역사), The Big Bang Theory(오락), 그리고 DAX 40(금융)이 포함되어 있으며, 각 도메인에 대해 표준화된 프로브를 사용하여 분석을 시행합니다.

- **Performance Highlights**: 결과는 높은 종료율을 보여주지만 모델에 따라 다름을 나타냅니다. 재현성 면에서는 혼합된 신호가 관찰되었으며, 정량적 성과에서는 높은 유사성이 있었으나, 어휘적 유사성은 낮고, 의미적 유사성은 중간 수준이었습니다. 여러 실험에서 miniGPTKB의 결과에 대한 강건성은 시드와 온도에 대해 높았으나, 언어와 모델에 대해서는 낮은 경향을 보였습니다.



### Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition (https://arxiv.org/abs/2510.06774)
- **What's New**: 이 논문에서는 뉴로-심볼릭(Nero-symbolic) 추론 프레임워크를 도입하여, 자연어로 표현된 문제에서 형식적 추론 전략을 자동으로 식별하고, 이를 통해 동적으로 전문적인 형식적 논리 해결사를 선택하고 적용할 수 있는 방법을 제시합니다. 현재의 접근 방식은 고정적이며 다양한 추론 전략 활용에 제약이 있으나, 우리의 접근 방식은 문맥 인식 문제 분해 및 해결사 선택 기능을 포함하여 유연성을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 단계로 구성된 구조로, 문제 분해, 라우팅 및 추론을 포함합니다. 이를 통해 자연어 입력을 기반으로 문제를 정형화하고, 적절한 해결사를 선택하여 예상된 답변을 생성하게 됩니다. 특히, 각 하위 질문은 자신의 형태적 논리 유형에 따라 선택된 해결사에 의해 처리되며, 프레임워크는 LLM 기반의 파서 컴포넌트를 활용하여 비구조적인 자연어를 반 구조적 형식으로 변환합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 프레임워크는 다양한 문제에서 92.1%의 정확도를 달성하여 기존의 제로샷 및 체인 오브 사고(Chain-of-Thought) 접근 방식을 17.0% 및 21.4% 초과하여 성능을 입증했습니다. 동적 해결사 조합은 여러 수순의 문제를 해결하는 데 있어 54.4%의 정확도를 달성하여 순수 LLM 방법보다 월등한 성과를 보였습니다. 하지만, 작은 모델에서는 적응형 추론에 어려움이 있으며, 포스트 학습을 통해 성능 개선이 가능함을 밝혔다.



### Gold-Switch: Training-Free Superposition of Slow- and Fast- Thinking LLMs (https://arxiv.org/abs/2510.06750)
- **What's New**: 이 논문은 LRM과 LLM을 결합하여 "Goldilocks" 균형을 달성하는 새로운 전략인 Gold-Switch를 제안합니다. 이 방법에서는 비효율적인 과도한 사고(overthinking)를 줄이면서도 적절한 사고를 유지하기 위해 저차원 근사(low-rank approximation)를 사용합니다. 이를 통해 실제로 모델을 배포하는 데 드는 비용을 절감하며, 인퍼런스(inference)의 효율성을 높일 수 있습니다. 반복적인 학습 없이도 성능을 개선할 수 있는 획기적인 접근 방식입니다.

- **Technical Details**: Gold-Switch는 linear layer에 대한 저차원 근사(low-rank approximation)를 활용하여 LRM과 LLM 간의 파라미터 차이를 조정합니다. 연구진은 높은 효율을 확보하고 인퍼런스를 가속화하는 레이어별 랭크 선택(rank selection) 방법을 제안했습니다. 이로 인해, 각 레이어가 필요로 하는 적절한 랭크를 선택하여 불필요한 계산을 줄이고, 전체적인 성능을 향상시킬 수 있음을 보여줍니다. 또한, LRM과 LLM의 통합은 저비용으로 구현할 수 있습니다.

- **Performance Highlights**: Gold-Switch 방법은 QwQ-32B 및 DeepSeek-R1-Distill-32B 모델에 대해 실험을 진행하였으며, 2.7배의 속도 향상과 성능 저하 없이 9배의 GPU 메모리 절약을 달성했습니다. 이러한 결과는 기존 라우터 기반 솔루션에 비해 상당한 이점을 제공하며, 실제 구현에 있어 더 나은 효율성을 demonstrated합니다. 이 논문에서는 코드와 모델을 공개하여 후속 연구를 촉진하고 있습니다.



### A Formal Framework for Fluency-based Multi-Reference Evaluation in Grammatical Error Correction (https://arxiv.org/abs/2510.06749)
Comments:
          Submitted to ACL Rolling Review - October 2025 for EACL 2026

- **What's New**: 이번 논문은 기존의 단일 참조(reference) 방식 대신 다중 참조에 기반한 문법 오류 수정 평가(grammatical error correction evaluation)를 위한 새로운 프레임워크를 제안합니다. 이 연구는 다양한 인간의 수정 중에서 허용 가능한 수정의 다양성을 반영하는 메트릭(metrices)의 필요성을 강조합니다. 기존의 평가 방식이 영어 중심적(edit-based and English-centric)이라는 한계를 극복하면서, 다국어(multilingual) 및 생성(generative) 환경에서의 적용 가능성을 증대시킵니다.

- **Technical Details**: 새로 제안된 프레임워크는 	extit{fluency-based multi-reference evaluation}을 통해 다수의 합법적인 수정에서의 $n$-gram 유사성(n-gram similarity)을 집계 문제로 정의합니다. 우리는 이 과정에서 GLEU를 네 가지 집계 전략인 	extsc{select-best}, 	extsc{simple-average}, 	extsc{weighted-average}, 	extsc{merged-counts}를 통해 구현하고 각 전략의 제한성(boundedness), 단조성(monotonicity), 참조 변동(reference variation)에 대한 민감도를 분석합니다.

- **Performance Highlights**: 체코어, 에스토니아어, 우크라이나어, 중국어 말뭉치(corpora)에서의 실험 결과는 이 네 가지 집계 전략이 유창성(fluency)과 범위(coverage)의 상보적인 측면을 포착함을 보여줍니다. 제안된 프레임워크는 다중 참조 평가(multi-reference evaluation)를 통합하여 언어적 다양성을 포함하되 합법적인 변동에 대한 페널티(penalizing)를 부여하지 않는 원칙 있는 접근 방식을 제공합니다.



### TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs (https://arxiv.org/abs/2510.06747)
- **What's New**: 이 논문에서는 기존의 embedder 위에서 사용할 수 있는 훈련이 필요 없는(label-free) 단기 텍스트 클러스터링 방법을 제안합니다. 이 방법은 상업적 설정에서 레이블 데이터가 없는 대량의 사용자 발화를 클러스터링하는 데 중점을 두고 있습니다. 제안된 방법은 LLM(대형 언어 모델)의 안내를 통해 희소 벡터를 반복적으로 업데이트하여 군집의 품질을 향상시킵니다.

- **Technical Details**: 우리의 접근 방식은 두 단계로 구성됩니다. 첫째, 대표 텍스트를 기반으로 희소 표현을 구성하고, 둘째, 사전 학습된 임베딩과 LLM의 지침을 결합하여 이를 반복적으로 정제합니다. 이 방법은 클러스터의 수나 라벨이 알려지지 않은 상황에서도 사용 가능하며, 다양한 embedder와 클러스터링 방법에 적용할 수 있는 유연성을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, 본 방법은 대조 학습(contrastive learning)을 사용하는 최신 기법들과 동등하거나 우수한 성능을 보입니다. 또한, 소규모 LLM을 사용하면서도 기존의 레이블 데이터나 미세 조정(fine-tuning)에 의존하는 방법들과 경쟁력 있는 결과를 담보합니다. 이로 인해 본 방법은 자원이 제한된 상황에서도 효과적으로 사용될 수 있습니다.



### AWM: Accurate Weight-Matrix Fingerprint for Large Language Models (https://arxiv.org/abs/2510.06738)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 지적 재산 보호의 필요성을 강조하며, 기존의 모델에서 유래되었는지 여부를 판별할 수 있는 효율적인 핑거프린팅 방법을 제안합니다. 특히, 매개변수 조작의 영향을 상쇄하기 위한 Linear Assignment Problem (LAP)과 Centered Kernel Alignment (CKA) 유사성을 활용하여 모든 포스트 트레이닝 과정에 견딜 수 있는 고강도 유사성 메트릭을 도출합니다.

- **Technical Details**: 제안된 방법은 LLM의 가중치 매트릭스(weight matrices) 기반으로 하며, 상세한 분석을 통해 모델 조작에 대한 강인성을 확보합니다. 이는 추가적인 훈련 없이 이루어지며, NVIDIA 3090 GPU에서 30초 이내에 전체 계산이 완료됩니다. 사용된 기술은 SFT, 지속적인 프리트레이닝, 강화 학습, 멀티모달 확장 등 다양한 후처리 과정에 대한 검증을 포함합니다.

- **Performance Highlights**: 제안된 방법은 60쌍의 긍정적 모델과 90쌍의 부정적 모델로 구성된 검사 세트에서 모든 분류 메트릭에서 완벽한 점수를 달성하였습니다. 특히, 기존 방법들과 비교했을 때 높은 분리 격차를 보이며, 거의 제로에 가까운 허위 긍정 위험을 유지합니다. 이러한 성과는 신뢰할 수 있는 모델 혈통 검증 기반을 확립하게 됩니다.



### Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization (https://arxiv.org/abs/2510.06732)
Comments:
          10 pages, 3 figures

- **What's New**: 이번 논문은 정보 검색에서 rerankers(재정렬기)로 활용되는 대형 언어 모델(LLMs)의 취약점을 드러내고 이를 조작할 수 있는 새로운 방법인 Rank Anything First (RAF)를 소개합니다. RAF는 타겟 아이템의 순위를 높이기 위해 간결한 텍스트 변형을 생성하는 두 단계의 토큰 최적화 방식으로 구성되어 있습니다. 첫 번째 단계에서는 그리디 좌표 경량화(Greedy Coordinate Gradient)를 사용하여 현재 위치에서의 후보 토큰을 선별하고, 두 번째 단계에서는 후보들을 평가하여 자연스러운 언어를 유지하면서 효과적으로 기존 방법들보다 높은 순위 조작성을 달성합니다.

- **Technical Details**: RAF는 토큰 단위 최적화를 통해 순위 조작 프롬프트를 생성하며, 두 가지 목표인 순위 효과성(maximizing ranking effectiveness)과 언어 자연스러움(preserving linguistic naturalness)을 동시에 고려합니다. 특정 수식과 조건부 확률을 통해 LLM의 재정렬 과정을 설명하며, 공격자가 특정 아이템의 설명에 자연스러운 텍스트 시퀀스를 삽입하여 아이템의 순위를 향상시킬 수 있도록 합니다. 이러한 과정은 제품의 브랜드, 가격, 간단한 설명을 포함한 제품 세트를 기반으로 하여 수행됩니다.

- **Performance Highlights**: RAF는 다양한 LLM 모델에서 실험을 통해 자연스러운 언어를 사용하여 목표 아이템의 순위를 유의미하게 증가시키는 데 성공하였습니다. 이는 기존의 방법들보다 더욱 강력하고 안정적인 순위 조작을 가능하게 하며, 최적화된 프롬프트는 여러 모델 간에 성공적으로 전이됩니다. 이 연구는 LLM 기반 reranking의 보안 위험성을 강조하며, 현대 정보 검색 시스템의 신뢰성과 강인성에 대해 새로운 도전 과제를 제시합니다.



### PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs (https://arxiv.org/abs/2510.06730)
- **What's New**: 이번 연구는 고정된 테스트 베드(benchmark)에서의 성능 평가가 실제로 사전 훈련된 테스크에서 편향된 결과를 초래할 수 있음을 보여 진화적이고 동적 방식의 평가 방법론이 필요함을 설명합니다. 특히, Paraphrasing Text Embedding Benchmark (PTEB)라는 새로운 평가 프로토콜을 도입하여 의미 보존을 위한 패러프레이징을 동적으로 생성해내어 보다 다양한 문제를 평가할 수 있는 방안을 제안합니다. 이는 정적 베이스라인을 넘어, 훈련 데이터의 contamination 문제를 해결할 수 있는 새로운 접근으로 주목받고 있습니다.

- **Technical Details**: PTEB는 의미적으로 동등하지만 텍스트적으로는 구별되는 문제 인스턴스를 생성하여 진화적인 평가를 수행합니다. 이 과정에서 generative LLMs를 활용하여 패러프레이징을 수행하고, 평가 시점에 생성된 데이터 변화를 통해 모델의 성능을 정확하게 평가합니다. 특히, cosine similarity를 사용하여 임베딩 간의 유사성을 측정하며, 이를 통해 LLM judge를 선정하고, 결과적으로 기존 MTEB 데이터셋에서 유의미한 결과를 도출할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, PTEB는 다양한 언어에 대해 7개의 MTEB 작업을 통해 성능 검증을 수행했으며, 작은 모델이 큰 모델에 비해 불균형적으로 영향을 받지 않는 것으로 나타났습니다. 본 논문은 고정된 데이터셋 대신 동적이며 확률론적인 평가 방식을 통해 NLP 커뮤니티의 기존 평가 체계를 확장할 것을 목표로 하고 있습니다. LLM의 패러프레이징 능력을 활용함으로써 기본적인 평가의 신뢰성을 높이고, 임베디드 모델의 성능을 더욱 정확하게 반영할 수 있음을 보여주고 있습니다.



### Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Managemen (https://arxiv.org/abs/2510.06727)
- **What's New**: 본 논문에서는 긴 범위의 다중 턴(tool use) 도구 사용을 위한 대형 언어 모델(LLM) 에이전트의 강화 학습(RL) 미세 조정을 연구하였습니다. 기존의 RL 파이프라인은 지침 수행 저하, 과도한 롤아웃 비용, 그리고 엄격한 컨텍스트 한계 등의 문제를 겪을 수 있습니다. 이를 해결하기 위해, 우리는 요약 기반의 컨텍스트 관리 방법을 교육에 도입했습니다.

- **Technical Details**: 이 방법은 LLM이 생성한 요약을 이용해 도구 사용 이력을 주기적으로 압축함으로써 작업과 관련된 정보를 유지하며 компакт한 컨텍스트를 유지할 수 있도록 지원합니다. 이 형식에 기반하여, 우리는 도구 사용 행동과 요약 전략을 끝에서 끝으로 최적화할 수 있는 정책 기울기(Policy Gradient) 표현을 도출했습니다. 이를 통해 	exttt{SUPO}라는 LLM RL 알고리즘을 구현하여 고정된 컨텍스트 한계를 넘어서는 긴 범위의 교육이 가능해졌습니다.

- **Performance Highlights**: 실험을 통해 상호작용하는 기능 호출 및 검색 작업에서 	exttt{SUPO}가 성공률을 크게 향상시키는 동시에 기준과 비교하여 동일하거나 심지어 더 짧은 작업 컨텍스트 길이를 유지함을 입증했습니다. 복잡한 검색 작업에 대해서는, 	exttt{SUPO}가 학습 시간보다 테스트 시간에서 최대 요약 라운드를 더욱 확장하여 평가 성과를 개선할 수 있음을 보여주었습니다. 이러한 결과는 요약 기반의 컨텍스트 관리가 고정된 컨텍스트 길이 한계를 넘어 RL 에이전트를 훈련시키기 위한 원칙적이고 확장 가능한 접근 방식임을 입증합니다.



### How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects (https://arxiv.org/abs/2510.06700)
- **What's New**: 이 연구는 LLM(대형 언어 모델)의 내부 표현에서 유효성(validity)과 그럴듯함(plausibility)의 개념을 어떻게 인코딩하는지 investigation(조사)합니다. 연구자들은 LLM이 어떻게 이 두 개념을 선형적으로 표현하며, 그 결과 유효성을 그럴듯함과 혼동하게 되는지 설명합니다. 이 연구는 LLM의 reasoning(추론) 시스템에서 content effects(내용 효과)를 조명하면서, 이 두 개념을 구분하기 위해 설계된 디바이싱(debiasing) 벡터도 소개합니다.

- **Technical Details**: 연구는 LLM의 hidden representation(숨겨진 표현 공간)에서 validity와 plausibility의 개념이 선형적으로 표현되어 있다는 가설에 기반합니다. 이를 통해 연구자들은 10개의 LLM을 zero-shot 및 chain-of-thought prompting 기법을 활용하여 분석하고, 이 두 개념 간의 벡터 비교 및 causal interaction(인과 상호작용)을 명확히 합니다. 연구는 plausibility가 validity 판단에 미치는 인과적 영향과 이를 통해 획득한 데이터를 기반으로 함수를 설계하여 content effects를 줄이는 방법을 모색합니다.

- **Performance Highlights**: 연구 결과, LLM은 plausibility가 validity 판단에 실질적인 영향을 미치는 systematic biases(체계적 편향)를 보이는 것으로 나타났습니다. 유효성과 그럴듯함의 벡터 간의 geometric similarity(기하학적 유사도)는 행동적 content effects의 강도와 상관관계를 가지며, 이들 간의 혼란을 줄이기 위해 디바이싱 벡터를 사용하는 방법이 효과적임을 확인했습니다. 이러한 결과는 LLM의 논리적 개념 표현 방식에 대한 이해를 심화시키고, 보다 논리적 시스템을 위한 representational interventions(표현 개입)의 가능성을 제시합니다.



### Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks (https://arxiv.org/abs/2510.06695)
- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)에 대한 관심이 증가함에 따라, 프롬프트 엔지니어링(prompt engineering)이 수동 설계에서 모델 기반 최적화로 발전하였습니다. 본 논문에서는 기계 번역(machine translation)과 같은 특정 작업에 적합한 새로운 프롬프트 최적화 방법을 소개합니다. 제안된 방법은 작은 매개변수 모델을 활용하여 백 트랜슬레이션(back-translation) 전략으로 학습하여, 단일 작업 최적화를 위한 훈련 비용을 대폭 줄이는 동시에 높은 성과를 제공합니다.

- **Technical Details**: LLM의 프롬프트는 일반적으로 instruction와 input의 두 가지 구성 요소로 이루어져 있습니다. 본 논문에서는 입력 최적화를 위한 Rewriting Original Inputs (ROI) 전략을 제안합니다. 이 방법은 LLM 또는 소형 매개변수 모델을 활용하여 원본 입력을 재구성하고, 언어 모델에 더 잘 맞도록 조정합니다. 특히, 입력의 질을 평가하기 위한 필터링 모듈도 도입되어 의미 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, ROI 모듈은 애매한 데이터를 보다 명확한 입력 프롬프트로 변환하는 데 효과적임을 보여줍니다. NLU 및 NLG 작업에 대한 성능 향상을 입증하였으며, 기존의 프롬프트 최적화 방법은 입력 구성 요소가 중요한 작업에 한계가 있다는 점을 보여주었습니다. ROI 방법은 다양한 LLM에 널리 적용 가능하며, 원본 입력에 비해 일관되고 주목할 만한 성과 향상을 이룹니다.



### Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback (https://arxiv.org/abs/2510.06677)
Comments:
          Accepted at EMNLP 2025 Industry Track

- **What's New**: 본 논문에서는 고객 지원 에이전트를 위한 점진적 요약 시스템을 소개합니다. 이 시스템은 대화 중 요약 노트를 생성해야 할 최적의 순간을 지능적으로 판단하여 에이전트의 맥락 전환 노력과 중복 리뷰를 줄입니다. Mixtral-8x7B 모델과 DeBERTa 기반 분류기를 결합하여 실시간으로 효과적인 요약 생성을 제공합니다.

- **Technical Details**: 시스템은 여러 채널의 대화를 통합하여 지속적으로 요약된 글머리 기사를 생성합니다. 자동화된 요약 모델은 중요 정보가 감지될 때만 요약을 제안하고, 비실질적인 내용을 걸러내는 분류기를 통해 자동으로 최적화됩니다. 에이전트의 실시간 수정 기능은 지속적으로 모델 학습을 보강합니다.

- **Performance Highlights**: 생산 환경에 배포된 이 시스템은 평균 처리 시간을 3% 단축시키며, 복잡한 사례에서는 최대 9%까지 감소시켰습니다. 또한, 설문 조사에서 80% 이상의 에이전트 만족도를 기록하며, 점진적 요약이 에이전트의 생산성을 향상시키는 효과를 입증합니다.



### PIKA: Expert-Level Synthetic Datasets for Post-Training Alignment from Scratch (https://arxiv.org/abs/2510.06670)
- **What's New**: 본 논문에서는 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF)의 필요성과 고품질 데이터 세트의 한계에 대해 논의하고, PiKa라는 새로운 데이터 세트를 소개합니다. PiKa는 기존의 데이터 세트보다 적은 예제 수로도 우수한 성능을 발휘하며, 효율적인 정렬 방법을 제공합니다. 이를 통해 오픈 소스 LLM(대형 언어 모델)의 데이터 효율성을 크게 개선할 수 있는 가능성을 열었습니다.

- **Technical Details**: PiKa 데이터 세트는 전문가 수준의 복합 지식을 담고 있는 시뮬레이션된 지침과 응답 쌍으로 구성되어 있으며, 총 30,000개의 예제로 이루어져 있습니다. 기존의 데이터 세트들이 수백만 개의 예제를 요구하는 데 반해, PiKa는 최소한의 예제로도 더 나은 성능을 보입니다. 이 데이터 세트는 GPT-4o 모델을 기반으로 생성된 것라, 고품질 결과를 유지하며, 시뮬레이션된 지침 쌍의 복잡성을 강조합니다.

- **Performance Highlights**: PiKa를 통해 학습된 Llama-3-8B-Base 모델은 다른 공공 데이터 세트에 비해 우수한 성능을 보였으며, AlpacaEval 2.0 및 Arena-Hard 벤치마크에서 검증된 결과가 주목할 만합니다. 특히 PiKa-SFT로 훈련된 모델은 기존의 Llama-3-8B-Instruct 모델보다 우수한 성능을 나타냈습니다. 이러한 결과는 PiKa 데이터 세트를 통해 고품질 정렬이 적은 데이터로도 달성될 수 있음을 보여줍니다.



### ToolMem: Enhancing Multimodal Agents with Learnable Tool Capability Memory (https://arxiv.org/abs/2510.06664)
- **What's New**: 이번 연구에서는 ToolMem이라는 새로운 프레임워크를 제안하여, 에이전트가 도구의 성능을 기억하고 최적의 도구를 선택할 수 있도록 돕습니다. 일반적인 도구 사용 에이전트들은 고정된 도구에 의존하는 반면, ToolMem은 에이전트가 이전 상호작용을 통해 도구의 강점과 약점을 기록하고 학습하게 합니다. 이를 통해 더 높은 정확도로 개별 작업을 수행하기 위한 도구 선택이 가능합니다.

- **Technical Details**: ToolMem은 세 가지 핵심 구성 요소로 이루어져 있습니다: 첫째, 도구의 행동을 숙련도 수준별로 분류하는 구조화된 용량 메모리; 둘째, 도구의 출력을 평가하는 피드백 생성 프로세스; 셋째, 새로운 경험을 통합하는 동적 메모리 업데이트 메커니즘입니다. 에이전트는 이러한 기억 항목들을 현재 작업에 맞춰 검색하고, 이를 바탕으로 더 나은 성과를 낼 수 있는 도구를 선택합니다.

- **Performance Highlights**: ToolMem을 적용한 에이전트는 도구 성능 예측에서 일반적인 에이전트보다 14.8%에서 28.7%까지 더 정확한 결과를 보였습니다. 또한, ToolMem을 사용한 도구 선택의 경우, 각각 21%와 24%의 절대 증가율을 기록하며 여러 후보 중 최적의 도구를 선택하는 효율성을 보였습니다. 이는 메모리를 활용한 도구 사용 에이전트의 동적인 특성을 성공적으로 다루고 있음을 보여줍니다.



### Aligning Large Language Models via Fully Self-Synthetic Data (https://arxiv.org/abs/2510.06652)
- **What's New**: 본 논문에서는 Self-Alignment Optimization (SAO)이라는 완전히 자가 합성(self-synthetic)된 방법론을 소개합니다. 이 방법론은 전통적인 reinforcement learning 방식의 한계를 극복하고, 비용이 많이 소모되는 데이터 수집과 주석이 필요 없습니다. SAO에서는 LLM이 사용자 쿼리, 응답, 그리고 취향을 포함한 모든 학습 데이터를 스스로 생성하여 모델의 정렬(alignment)을 진행합니다.

- **Technical Details**: SAO는 사용자가 원하는 다양한 프롬프트와 응답을 생성하도록 LLM에 지시하는 것으로 시작됩니다. LLM은 생성된 응답 쌍을 스스로 평가하여 선호도를 최적화합니다. 이러한 자가 평가(self-evaluation)와 최적화 과정은 대규모 언어 모델의 성능을 크게 향상시키며, 이는 수동적인 외부 개입 없이도 이루어집니다.

- **Performance Highlights**: SAO는 여러 표준 채팅 벤치마크에서 뚜렷한 성능 향상을 보여주었습니다. 예를 들어, AlpacaEval 2.0에서 SAO를 통해 Gemma-2-9B-it 모델의 Length-Controlled Win Rate가 18.1% 향상되었습니다. 또한, SAO는 다운스트림(NLP) 작업에서도 базован된 LLM 성능을 유지하거나 증가시키며, 이러한 결과는 모델의 고유한 데이터 효율성을 강조합니다.



### A Comparative Analysis of Contextual Representation Flow in State-Space and Transformer Architectures (https://arxiv.org/abs/2510.06640)
- **What's New**: 본 논문은 State Space Models (SSMs)와 Transformer-Based Models (TBMs) 간의 표상 전이 분석을 위한 첫 번째 통합 비교 연구를 제시합니다. 이 연구는 SSMs와 TBMs의 특징을 비교하면서 두 아키텍처가 문맥 정보의 흐름을 어떻게 처리하는지에 대해 심층적으로 분석합니다. 특히, SSMs는 초기 단계에서 토큰의 독창성을 유지하는 반면, TBMs는 빠르게 동질화되며, 이는 미래의 모델 및 훈련 설계에 중요한 통찰력을 제공합니다.

- **Technical Details**: 연구자들은 SSMs와 TBMs의 토큰 및 레이어 수준에서의 표상 전이를 조사하기 위해 Centered Kernel Alignment (CKA)와 안정성 지표, 프로빙(probing) 기법을 활용했습니다. 또한, TBMs에서의 과도 평활화(oversmoothing)는 건축 설계에서 기인하며, SSMs의 경우 주로 훈련 동역학에서 기인함을 이론적으로 분석했습니다. 이 이러한 고찰은 두 아키텍처의 귀납적 편향을 명확히 하고, 장기 문맥 추론을 위한 보다 효과적인 모델 설계에 기여할 수 있는 정보를 제공합니다.

- **Performance Highlights**: 실험에서는 Pile 데이터셋에서 사전 훈련된 TBMs와 SSMs 간의 정보 흐름을 비교하고, 롱컨텍스트 작업을 위한 두 개의 벤치마크 테스트를 사용하여 모델 성능을 평가했습니다. 결과적으로, 중간 레이어가 최종 레이어보다 더 효과적이라는 것이 드러났으며, 다양한 작업, 모델 규모 및 문맥 길이에 걸쳐 이러한 경향이 일관되었습니다. 이는 모델의 깊이가 완전히 활용되지 않을 수 있다는 것을 나타내며, 이러한 인사이트는 향후 모델 설계에 중요한 지침이 될 것입니다.



### Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection? (https://arxiv.org/abs/2510.06594)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 내부 표현 분석을 통해 jailbreak과 benign 프롬프트를 구별하는 새로운 접근 방식을 제안합니다. 기존의 방어 메커니즘에 한계를 인식하고, 내부 레이어의 구조적 패턴에서 차이를 발견하여 새로운 탐지 방법을 모색하였습니다. 특히, open-source 모델인 GPT-J 및 state-space 모델인 Mamba2를 분석하여 레이어별 행동의 차이를 기반으로 초기 발견을 발표하였습니다.

- **Technical Details**: 연구에서 사용된 모델 GPT-J는 6억 개의 파라미터를 가진 autoregressive transformer 모델로, 28개의 레이어와 16개의 self-attention heads, 피드포워드 네트워크(Feed-Forward Network)로 구성됩니다. Mamba2는 선형적 recurrence를 기반으로 한 state-space 모델이며, 복잡한 긴 시퀀스를 처리하는 데 효과적입니다. 제안된 메서드는 각 모델에서 초기, 중간, 최종 레이어의 표현을 추출하고, tensor decomposition 기술을 이용하여 jailbreak과 benign 프롬프트 간의 구조적 차이를 분석합니다.

- **Performance Highlights**: 실험 결과, 5-fold cross-validation을 통해 추출된 latent representations가 jailbreak과 benign 프롬프트를 효과적으로 구분할 수 있음을 보여주었습니다. GPT-J의 경우 Multi-Head Attention (MHA) 표현이 레이어 출력보다 더 우수한 성능을 보였으며, Mamba-2의 경우 Mixer 표현이 Block 출력보다 더 나은 결과를 나타냈습니다. 이로 인해, tensor decomposition과 결합된 latent factors가 jailbreak 탐지를 위한 간단하면서도 효과적인 프레임워크를 제공함을 확인하였습니다.



### TinyScientist: An Interactive, Extensible, and Controllable Framework for Building Research Agents (https://arxiv.org/abs/2510.06579)
Comments:
          7 pages, EMNLP 2025 Demo track

- **What's New**: 이번 논문은 자동 연구(automatic research)에 대한 새로운 프레임워크인 TinyScientist를 소개합니다. 대형 언어 모델(Large Language Models, LLMs)을 활용하여 연구자와 상호작용할 수 있는 연구 에이전트를 만드는 것에 대한 관심이 커지고 있으며, 이를 통해 복잡한 연구 작업이 자동화되고 있습니다. 연구의 복잡도를 관리하기 위해, TinyScientist는 상호작용적이고 확장 가능하며 제어 가능한 시스템을 제공하여 모든 연구자와 개발자가 손쉽게 접근할 수 있도록 합니다.

- **Technical Details**: TinyScientist는 연구 워크플로우를 구성하는 핵심 요소를 식별하고, 이를 기반으로 모듈화된 인터페이스를 제공합니다. 시스템의 구성은 사용자가 연구하고자 하는 내용에 따라 실시간으로 수정할 수 있는 기능을 제공하며, 사용자가 의도를 명확하게 전달할 수 있도록 돕습니다. 기존의 연구 시스템이 직면한 한계—상호작용 부족, 확장성 부족 및 제어 부족—를 해결함으로써 사용자와 연구자의 피드백을 반영할 수 있는 구조를 갖추고 있습니다.

- **Performance Highlights**: TinyScientist의 성능은 기존의 자동 연구 프레임워크와 비교해 유사한 수준의 연구 결과 생성 품질을 보여줍니다. 연구자들은 별도의 설정 없이 Python 패키지를 쉽게 사용할 수 있으며, 인터랙티브 UI를 통해 인간-에이전트 간의 상호작용을 향상시킵니다. 새롭게 통합된 도구 사용이 생성 품질을 개선함을 보여주는 정량적, 정성적 평가 결과를 제공합니다.



### The Algebra of Meaning: Why Machines Need Montague More Than Moore's Law (https://arxiv.org/abs/2510.06559)
- **What's New**: 이 논문에서 제안하는 Savassan 시스템은 현대 언어 모델의 비효율성을 해결하기 위해 Montague 스타일의 의미론을 적용합니다. 기존 AI 모델들이 출력의 의미를 제대로 파악하지 못하는 문제를 진단하고, 의미의 타입을 구성하는 방식을 통해 법적 및 비즈니스 상황에서의 의사결정을 위하여 새로운 방법론을 제시합니다. Savassan은 비구조적 입력을 처리하여 여러 법적 맥락에서 문장의 의미를 적절히 해석할 수 있도록 설계되었습니다.

- **Technical Details**: Savassan은 신경 기호(neuro-symbolic) 아키텍처를 기반으로 하여 발화를 Montague 스타일의 논리형식으로 변환하고, 여기에 의무(Deontic) 연산자 및 관할권(context) 추가가 이루어집니다. 이 시스템은 입력을 파싱한 후, 각기 다른 법적 온톨로지에 결과를 투사해 설명 가능성이 높은 의사 결정을 생성합니다. Savassan은 표본 구조의 기호적 검증(symbolic validation) 과정을 통해 법적 및 비즈니스 타입 시스템과 일치하도록 됩니다.

- **Performance Highlights**: Savassan의 성능은 법적 추론 벤치마크와 다중 관할권 평가를 통해 평가될 예정입니다. 시스템은 복잡한 법적 맥락에서도 반복적인 분류 과정을 개선하여 단 한번의 파싱으로 다양한 법적 위험을 평가할 수 있습니다. Savassan은 문장의 의미를 수학적 정밀도로 파악하여 설명 가능성이 높은 결과를 생성함으로써 기존 AI 시스템의 한계를 극복합니다.



### Flipping the Dialogue: Training and Evaluating User Language Models (https://arxiv.org/abs/2510.06552)
- **What's New**: 이 논문은 사용자 언어 모델(User LMs)을 소개하며, 이 모델들은 인간 사용자의 대화 행위를 더 잘 시뮬레이션하도록 포스 트레이닝되었습니다. 기존의 어시스턴트 LMs가 사용자 시뮬레이터로 효과적이지 않다는 놀라운 결과를 보여줍니다. 사용자 행동의 미묘함을 고려하지 않아, 어시스턴트의 성능 평가에 한계를 두고 있다는 점을 강조합니다.

- **Technical Details**: 사용자 LMs는 명확한 사용자 의도를 기반으로 대화를 시작하고, 이후 어시스턴트의 응답에 따라 후속 대화를 진행하며, 필요시 대화를 종료하는 기능을 수행합니다. 훈련 과정에서는 WildChat 데이터 세트를 사용하여 478,498개의 영어 대화를 기반으로 사용자 의도를 고수준 목표로 정의합니다. 이를 통해 다양한 대화 표현을 가능하게 하여, 더 사실적인 사용자 행동을 시뮬레이션할 수 있도록 설계되었습니다.

- **Performance Highlights**: 사용자 LMs는 기존의 방법보다 더 다양한 사용자 발화 생성, 다중 턴 간의 의도 분해, 대화 종료 등에서 우수한 성능을 보입니다. 실험 결과, 강력한 어시스턴트인 GPT-4o와의 대화 시뮬레이션에서 사용자 LMs를 활용하면 더욱 현실적인 성과 추정이 가능함을 입증했습니다. 이는 어시스턴트가 실제 사용자의 맥락과 요구 사항을 효과적으로 다루지 못하는 환경에서 더 두드러지는 경향을 보여줍니다.



### From Acceleration to Saturation: Scaling Behavior of Bootstrapped Language Model Pretraining (https://arxiv.org/abs/2510.06548)
Comments:
          22 pages, 11 figures, an abridged version to appear in NeurIPS 2025 LLM Evaluation Workshop

- **What's New**: 이번 연구에서는 부트스트랩 사전 훈련(bootstrapped pretraining)의 확장 효율성이 구체적으로 저하된다는 것을 발견하였습니다. 즉, 두 번째 단계의 사전 훈련 시, 기본 모델이 훈련된 토큰 수에 따라 로그적으로 감소하는 스케일링 지수를 보여주고 있습니다. 또한, 과도하게 훈련된 모델에 대해 부트스트랩의 효과가 감소할 수 있다는 점을 강조합니다.

- **Technical Details**: 연구에서는 다양한 크기의 언어 모델에 대해 광범위한 실험을 수행하였으며, 부트스트랩 사전 훈련 방법의 스케일링 행동을 분석하였습니다. 각 단계에서의 훈련 토큰 수(D1, D2)에 따른 검증 손실(LL) 변화를 측정하고, 이로부터 수렴 현상에 대한 수학적 모델을 제시하였습니다. 특히, 활성화 함수로는 SwiGLU 사용 및 Rotary position embeddings를 적용하여 LLaMA와 유사한 아키텍처를 구현하였습니다.

- **Performance Highlights**: 부트스트랩 사전 훈련이 제공하는 이점은 기본 모델이 충분히 훈련되었을 때 감소하는 경향이 있다는 결과를 도출하였습니다. 스케일링 법칙을 통해 부트스트랩이 유리한 경우와 그렇지 않은 경우를 정량적으로 평가할 수 있는 지침을 제공하며, 전체적으로 파라미터와 데이터 집합 크기의 변화를 통합하여 설명할 수 있음을 보여주었습니다. 이 연구는 언어 모델 훈련의 효율성을 높이는 데 중요한 통찰을 제공합니다.



### Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels (https://arxiv.org/abs/2510.06499)
- **What's New**: 이번 연구에서는 Webscale-RL이라고 하는 새로운 데이터 파이프라인을 소개합니다. 이 파이프라인은 대규모의 사전 훈련 문서를 시스템적으로 변환하여 수백만 개의 다양한 검증 가능한 질문-답변 쌍을 생성합니다. 이를 통해 기존의 RL 데이터셋과 비교하여 방대한 양의 데이터를 제공하고, RL 훈련이 기존보다 데이터 효율성을 높일 수 있도록 합니다.

- **Technical Details**: Webscale-RL 데이터셋은 1.2백만 개의 검증 가능한 QA 쌍을 포함하고 있으며, 9개 이상의 도메인에 걸쳐 다양한 내용을 다룹니다. 이 데이터셋은 데이터 필터링, 도메인 및 페르소나 주도의 생성, 품질 검증 단계를 포함한 웹 스케일의 데이터 엔진을 통해 구축되었습니다. RL 모델이 이 데이터셋을 통해 훈련될 때, 기존의 데이터 세트에서 지속적 사전 훈련을 수행한 모델보다 성능이 현저히 향상됩니다.

- **Performance Highlights**: Webscale-RL 데이터셋에서 훈련된 RL 모델은 100배 적은 토큰으로도 지속적인 사전 훈련과 유사한 성능을 달성했습니다. 이는 RL 접근 방식의 데이터 효율성을 입증하는 결과입니다. 이러한 연구 결과는 RL 훈련을 사전 훈련 수준으로 확장할 수 있는 가능성을 시사하며, 향상된 능력과 효율성을 갖춘 새로운 언어 모델의 출현을 촉진합니다.



### Test-Time Scaling of Reasoning Models for Machine Translation (https://arxiv.org/abs/2510.06471)
- **What's New**: 이 논문은 머신 트랜스레이션(MT)에서 테스트 타임 스케일링(Test-time scaling, TTS)의 효능을 조사합니다. TTS는 Reasoning Models(RMs)의 성능을 향상시키는 방법으로, 다양한 MT 벤치마크를 통해 그 효과를 분석합니다. 연구 결과, 일반 RMs에 대해서는 TTS가 직접 번역의 성능을 크게 향상시키지 않지만, 도메인별 미세 조정을 통해 성능이 개선될 수 있다는 점이 밝혀졌습니다.

- **Technical Details**: 특히 이 연구는 세 가지 시나리오, 즉 직접 번역(direct translation), 강제 추론(extrapolation), 그리고 후편집(post-editing)을 다루며, 12개의 RMs에 대한 평가를 수행합니다. TTS는 이론적으로 추가적인 계산 자원을 할당하여 번역 품질을 개선하려는 접근 방식입니다. 다만, 강제 추론을 통한 동작이 모델의 자연스러운 사고 과정을 방해할 수 있음을 보여줍니다.

- **Performance Highlights**: TTS는 후편집 과정에서 특히 효과적이며, 자가 수정(self-correction) 프로세스를 유익하게 전환할 수 있습니다. 연구의 주요 발견은 TTS가 직접 번역에 대한 명확하고 일관된 이점을 제공하지 않으며, 오히려 도메인에 특화된 fine-tuning을 통해 성능이 크게 향상된다는 것입니다. 또한 강제로 추론을 시도할 경우 일관되게 번역 품질이 저하된다는 점도 밝혀졌습니다.



### Linguistically Informed Tokenization Improves ASR for Underresourced Languages (https://arxiv.org/abs/2510.06461)
- **What's New**: 이 연구에서는 언어 documentation(문서화)을 수행하려는 언어학자를 위한 자동 음성 인식(ASR) 시스템의 잠재력을 보여줍니다. 불행히도, 현대 ASR 시스템은 데이터 집약적인 transformer 아키텍처를 사용하여 자원이 부족한 언어에서는 실질적으로 사용이 불가능합니다. 이를 해결하기 위해, 저자들은 Yan-nhangu라는 호주 원주민 언어의 wav2vec2 ASR 모델을 미세 조정하고, 성능에 미치는 음소(tokenization)와 철자(orthographic) 전략의 효과를 비교합니다.

- **Technical Details**: 연구에서는 음소 토큰화 시스템이 성능 향상에 미치는 영향을 분석하여, 기존의 철자 토큰화 방식에 비해 Word Error Rate (WER)와 Character Error Rate (CER)를 상당히 개선하는 결과를 보여주었습니다. 이 과정에서 언어학적으로 체크된 토큰화 방법이 ASR 성능에 미치는 중요한 역할을 발견했습니다. 또한, ASR 모델의 출력을 수작업으로 수정하는 것이 오디오를 처음부터 필사(transcribing)하는 것보다 훨씬 빠르다는 점도 강조됩니다.

- **Performance Highlights**: 이 연구는 ASR이 자원이 부족한 언어에 대해 실질적인 도구로 작용할 수 있음을 입증합니다. Yan-nhangu에 대한 미세 조정 모델은 조건에 적합하며, 언어 문서화 프로세스에서 ASR의 활용 가능성을 높이는 데 기여할 수 있습니다. 따라서 ASR 기술은 언어 보존과 복원에서 매우 중요한 역할을 할 수 있을 것으로 보입니다.



### A Survey on Agentic Security: Applications, Threats and Defenses (https://arxiv.org/abs/2510.06445)
- **What's New**: 이 논문은 자율 LLM(대형 언어 모델) 에이전트의 보안 환경에 대한 최초의 포괄적인 조사를 제시합니다. 기존의 수동 LLM에서 독립적으로 행동할 수 있는 LLM 에이전트로의 빠른 전환은 사이버 보안의 새로운 패러다임을 형성합니다. 이러한 변화는 공격 및 방어 작업에 강력한 도구로서의 잠재력을 제공하지만, 새로운 보안 리스크도 함께 도입됩니다.

- **Technical Details**: 저자들은 150개 이상의 논문을 포괄하는 세 가지 상호 의존적인 기둥: 응용(Application), 위협(Threat), 방어(Defense) 주위에 이 분야를 구조화했습니다. 이러한 보안 리스크들을 이해하기 위해 에이전트의 사용, 취약점, 그리고 보호를 위한 대응책을 체계적으로 설명합니다. 또한, 에이전트 아키텍처에서의 새로운 경향을 보여주는 상세 분석을 통해 모델 및 모드와 관련된 주요 연구 공백을 드러냅니다.

- **Performance Highlights**: 이 연구는 자율 LLM 에이전트의 보안 환경을 이해하는 데 필요한 기반을 제공합니다. 신뢰할 수 있는 응용 프로그램을 구축하기 위해 필요한 보안 요구사항과 취약점을 식별하고, 이러한 문제를 다루기 위한 다양한 방어 전략을 제시합니다. 이러한 정보는 향후 연구와 실용적인 적용을 위한 기초 자료로 활용될 수 있습니다.



### MathRobust-LV: Evaluation of Large Language Models' Robustness to Linguistic Variations in Mathematical Reasoning (https://arxiv.org/abs/2510.06430)
- **What's New**: 이 논문에서는 MathRobust-LV라는 새로운 평가 방법론을 도입하여, 고등학교 수준의 수학 문제에 대한 대화형 AI 모델의 언어적 변형에 대한 강건성을 평가합니다. 이는 교육적 맥락에서 인스트럭터들이 문제를 재구성하는 방식을 반영하며, 문제의 난이도를 유지한 채로 표면적인 세부사항을 변경합니다. 저자들은 기존의 IMO 수준의 문제 평가에서 벗어나, 더 실질적인 교육에서의 수학적 강건성을 강조합니다.

- **Technical Details**: MathRobust-LV는 네 가지 조절 축을 바탕으로 구성된 변형을 포함합니다: (1) 변수 치환, (2) 맥락 치환, (3) 고정된 변수를 위한 완전 재구성, (4) 변수 치환을 포함한 완전 재구성입니다. 이 방법론은 문제의 수치 구조와 기호적 로직을 유지하면서 언어적 변형을 최소화합니다. 연구 결과, 34 개의 다양한 모델에서 성능 저하를 관찰하였고, 이는 표면적 특징에 의존하는 경향을 시사합니다.

- **Performance Highlights**: 34 개 모델의 평가에서, 정확도가 기본 기준에서 변형으로 이동함에 따라 감소하였습니다. 특히 작은 모델에서 9-11%의 심각한 성능 저하가 관찰되었으며, 강력한 모델 역시 측정 가능한 저하를 보였습니다. 그러나 GPT-5와 Gemini-2.5pro와 같은 프런티어 모델은 상대적으로 안정적인 성능을 유지했습니다. 이러한 결과는 모델의 언어적 변형에 대한 강건성이 중요하다는 점을 강조합니다.



### Bridging Discourse Treebanks with a Unified Rhetorical Structure Parser (https://arxiv.org/abs/2510.06427)
Comments:
          Accepted to CODI CRAC 2025

- **What's New**: UniRST는 11개 언어에서 18개의 트리뱅크를 처리할 수 있는 첫 번째 통합 RST 스타일 담론 파서입니다. 이 모델은 관계 재고를 조정하지 않고도 다국어를 지원하도록 설계되었습니다. Mono-Treebank 방식에서는 낮은 자원 환경에서도 효과적인 데이터 증강 기법이 소개됩니다.

- **Technical Details**: UniRST는 두 가지 훈련 전략인 Multi-Head와 Masked-Union을 제안합니다. Multi-Head는 각 트리뱅크 별로 별도의 분류 레이어를 할당하며, Masked-Union은 선택적 레이블 마스크를 통해 공유 매개변수 훈련을 가능하게 합니다. 이 모델은 end-to-end RST 파싱을 위해 EDU 세분화, 구조 예측 및 핵심성 및 관계 레이블링을 포함한 세 가지 상호 연결된 하위 작업을 처리합니다.

- **Performance Highlights**: UniRST 모델은 18개의 단일 트리뱅크 기준선 중 16개를 초과 달성했으며, 이는 단일 모델 다국어 end-to-end 담론 파싱의 장점을 보여줍니다. Masked-Union 접근법이 가장 효율적이며 성능이 뛰어나다는 것이 입증되었습니다. 이 연구는 다국어 및 다양한 자료를 통합하는 유니버설 담론 파서를 개발하는 필요성을 강조합니다.



### FinLFQA: Evaluating Attributed Text Generation of LLMs in Financial Long-Form Question Answering (https://arxiv.org/abs/2510.06426)
Comments:
          EMNLP 2025 Findings

- **What's New**: 이번 논문에서는 금융 분야에서의 복잡한 장문 질문에 대한 답변 생성을 평가하는 새로운 벤치마크인 FinLFQA를 소개합니다. 기존의 평가 기준은 단순한 참조 검색 중심이었으나, FinLFQA는 supporting evidence, numerical reasoning, domain-specific knowledge와 같은 세 가지 중요한 측면을 모두 평가합니다. 이를 통해 장문 질문 답변 생성의 신뢰성과 정확성을 높일 수 있는 방안을 제시합니다.

- **Technical Details**: FinLFQA는 1,008개의 전문가 주석 데이터로 구성되어 있으며, 특정 기업의 재무 보고서를 기반으로 한 분석 능력을 평가합니다. 각 답변은 문장, 증거 목록, intermediate reasoning, 전문 지식 인덱스 등 세 가지 형태의 서포트 생성을 포함합니다. 이를 위해 자동화된 평가 시스템이 도입되어, factual accuracy와 numerical correctness를 포함하는 세부 지표를 사용하여 모델 성능을 평가합니다.

- **Performance Highlights**: 여덟 개의 대형 언어 모델(LLM)에 대한 실험 결과, 세부 지표가 모델의 역량을 구별하는 데 중요하다는 것을 발견했습니다. end-to-end 방식의 답변 생성은 사후 attribution과 비슷한 성능을 보였으며, iterative refinement는 외부 피드백에 의해 개선될 수 있음을 보여주었습니다. 이러한 결과는 모델의 신뢰성과 해석 가능성을 향상시키는 데 기여할 것으로 기대됩니다.



### Instructional Goal-Aligned Question Generation for Student Evaluation in Virtual Lab Settings: How Closely Do LLMs Actually Align? (https://arxiv.org/abs/2510.06411)
- **What's New**: 본 논문에서는 교육 목표에 맞춘 질문 생성을 위한 새로운 정렬 프레임워크를 소개합니다. 이 프레임워크는 교사가 LLMs (Large Language Models)를 활용하여 시뮬레이션과 적합한 교육적 질문을 생성할 수 있도록 돕습니다. 기존의 교육 자료가 교사의 목표에 맞지 않는 경우가 많았는데, 이 방법은 그 문제를 해결하려는 의도가 있습니다.

- **Technical Details**: 제안된 프레임워크는 네 가지 주요 구성 요소로 이루어져 있습니다: (1) 교사-LLM 대화를 통한 교육 목표 이해, (2) 지식 단위와 관계 분석을 통한 실험실 이해, (3) 인지적 및 교육적 의도를 구조화하는 질문 분류 체계, (4) 프롬프트 세부 사항을 제어하는 TELeR 분류 체계. 이 구조는 구조적 유효성과 언어적 명확성을 통해 교육 목표와 시뮬레이션 기반 학습 목표와의 정렬을 평가합니다.

- **Performance Highlights**: 우리는 19개의 오픈소스 LLM로부터 생성된 1,100개 이상의 질문을 분석하여 교육 목표와 시뮬레이션 Context에 맞춘 질문의 품질을 평가했습니다. 결과적으로, 더 큰 모델이 37.1% 더 높은 구조적 유효성을 보였으며, 질문 형식에 따라 품질의 차이가 컸습니다. 개방형 질문 형식이 상대적으로 높은 사고 수준을 유도하는 효과가 있음을 보여줍니다.



### Reward Model Perspectives: Whose Opinions Do Reward Models Reward? (https://arxiv.org/abs/2510.06391)
Comments:
          Published at EMNLP 2025 under the full author name "Elle"

- **What's New**: 이 연구는 보상 모델(Reward Models, RMs)의 행동을 포괄적으로 분석하는 새로운 프레임워크를 제시하며, 인간의 선호를 반영하는 류의 연구가 부족한 가운데 RMs가 나타내는 사회 인구학적 편향(sociodemographic biases)을 조사합니다. 특히, 특정 그룹의 선호에 맞춰 보상을 조정하기 위한 프롬프트(prompt)의 효과도 다룹니다. RMs가 다양한 사회 집단과 잘 정렬되지 않으며, 해로운 고정관념을 체계적으로 보상할 수 있다는 점을 강조합니다.

- **Technical Details**: 연구는 LM(Language Models)와 RMs의 정렬을 이해하기 위해 RMs의 태도, 의견, 가치(value)를 분석합니다. 이 과정에서 LM과 RMs는 인간의 의도를 반영하기 위해 RLHF(Reinforcement Learning from Human Feedback) 기법을 사용하고, 이러한 기법들은 종종 단일한 신념 세트에 정렬되어 있어 글로벌 관점의 다양성을 간과할 위험이 있습니다. 본 연구는 RMs에 의해 인코딩된 사회 인구학적 편향을 정량적으로 분석하는 최초의 사례로, 특정 RM 간의 상대적 정렬 측정에서 일관성을 발견했습니다.

- **Performance Highlights**: 결과적으로 RMs는 다수의 사회 집단에서 잘 정렬되지 않으며, 기본적으로 내재된 사회적 편향(social biases)으로 인해 원하는 공정성과 안전성을 충족하지 못할 수 있습니다. 연구에서 제안된 기존의 벤치마크는 과최적화(over-optimization)와 불분명한 사회적 편향으로 인해 성능 평가는 신뢰할 수 없음을 보여줍니다. 이로 인해 RMs의 적절성을 평가하고, AI 모델의 안전성과 정렬을 보장하기 위한 추가 연구가 필요하다고 강조합니다.



### Controllable Stylistic Text Generation with Train-Time Attribute-Regularized Diffusion (https://arxiv.org/abs/2510.06386)
Comments:
          Preprint under review

- **What's New**: RegDiff는 사전 훈련된 분류기 없이도 스타일 속성에 대해 조절 가능한 텍스트 생성을 가능하게 하는 정규화된 확산 프레임워크입니다. 이 프레임워크는 VAE(Variational Autoencoder) 기반의 인코더-디코더 아키텍처를 이용하여 재구성의 정확성을 보장하고, 속성 감독 아래에서 훈련된 잠재 확산 모델을 활용합니다. RegDiff는 다섯 가지 데이터 세트에서 실험을 통해 기존 방법들보다 우수한 성과를 보이며, 스타일 속성을 조절하는 데 있어 효율성을 입증하였습니다.

- **Technical Details**: RegDiff는 두 가지 형태의 정규화를 적용하여 속성 정보를 통합합니다. 첫 번째는 VAE 훈련 과정에서 데이터 측면의 속성 매니폴드에 적용되는 정규화이며, 두 번째는 확산 훈련 과정에 통합되어 효율적인 추론과 스타일적으로 일관된 출력을 보장하는 정규화입니다. 이러한 기술적 요소는 전체적인 생성 품질을 유지하면서 속성 표현을 효과적으로 분리할 수 있도록 합니다.

- **Performance Highlights**: RegDiff는 스타일 속성 제어에서 매우 안정적이고 우수한 성과를 기록하였습니다. 실험 결과, 정보가 서로 얽혀 있는 속성에 대해서도 제어가 가능하며, 추론 시간을 통해 분류기를 기반으로 한 지침 없이도 효과적으로 속성을 조정할 수 있음을 보여줍니다. RegDiff는 여러 다양한 데이터 세트에서 기존의 강력한 벤치마크 대비 뛰어난 성능을 발휘함으로써, 조절 가능하고 스타일리틱한 텍스트 생성에 있어 일반적이고 효과적인 프레임워크로 자리매김하고 있습니다.



### Protecting De-identified Documents from Search-based Linkage Attacks (https://arxiv.org/abs/2510.06383)
- **What's New**: 이 논문은 기존의 de-identification 모델이 개인정보를 숨기는 데는 성공하지만, 원본 데이터에 대한 링크 위험(linkage risks)을 해결하지 못한다는 문제를 지적합니다. 향상된 방법으로는, 이 논문에서 제시한 N-그램(inverted index)을 기반으로 하여 텍스트를 변경하여 이러한 링크 공격을 방지하는 노력이 포함됩니다. 이 과정에서 원문의 의미를 계속 유지할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 두 단계로 진행됩니다. 첫 번째 단계에서는 문서 집합에서 발생하는 N-그램의 inverted index를 구성하여, 특정 문서의 수가 $k$ 미만인 N-그램을 효과적으로 찾아냅니다. 두 번째 단계에서는 LLM(based rewriter)을 사용하여 이 N-그램이 포함된 텍스트를 재구성하여 링크가 더 이상 불가능하도록 합니다.

- **Performance Highlights**: 법원 사건 데이터셋을 사용한 실험 결과, 제안된 방법이 기존의 텍스트 재작성 방법보다 검색 기반 링크를 효과적으로 방지할 수 있음을 보여주었습니다. 이는 문서의 원래 내용을 신뢰성 있게 유지하면서 링크 위험을 크게 줄일 수 있는 가능성을 시사합니다.



### Semantic Regexes: Auto-Interpreting LLM Features with a Structured Languag (https://arxiv.org/abs/2510.06378)
- **What's New**: 이번 연구에서는 large language model (LLM) 기능을 인간이 이해할 수 있는 구조화된 언어 설명으로 번역하기 위해 semantic regexes를 도입했습니다. 기존의 자연어 설명은 모호하고 일관성이 없으며 수동 라벨링이 필요하지만, 새로운 방법은 기능을 구체적으로 설명할 수 있는 기회를 제공합니다. 이를 통해 기능 복잡성을 계량화하거나 모델 전반에 걸친 패턴을 분석할 수 있는 새로운 종류의 해석 가능성을 제공합니다.

- **Technical Details**: Semantic regexes는 저수준 구문 패턴과 고수준 의미론적 개념을 설명하는 데 사용되는 구조적 언어입니다. 이들은 인간이 해석할 수 있는 원시 요소(primitives)와 수식어(modifiers)로 구성되어 있으며, 문맥화(contextualization) 및 조합(composition) 규칙을 포함합니다. 이러한 구조는 간결하고 일관된 기능 설명을 생성하는 데 도움을 줄 뿐만 아니라, 기능의 복잡성을 비유적으로 나타낼 수 있는 기능도 제공합니다.

- **Performance Highlights**: 정량적 기준 및 정성적 분석을 통해 semantic regexes는 자연어에 비견되는 정확성을 보이며, 더 간결하고 일관된 기능 설명을 생성합니다. 사용자 연구에서는 semantic regex 설명이 사람들이 LLM의 기능 활성화에 대한 정확한 정신 모델을 구축하는 데 도움을 주는 것으로 나타났습니다. 이는 모델의 해석 가능성을 높이는 중요한 발견으로, 향후 AI 모델의 신뢰성을 높이는 데 기여할 것입니다.



### EverydayMMQA: A Multilingual and Multimodal Framework for Culturally Grounded Spoken Visual QA (https://arxiv.org/abs/2510.06371)
Comments:
          Multimodal Foundation Models, Large Language Models, Native, Multilingual, Language Diversity, Contextual Understanding, Culturally Informed

- **What's New**: 이번 논문에서는 Everyday Multimodal and Multilingual QA (EverydayMMQA)라는 새로운 프레임워크를 소개하며, 문화적 기초가 있는 대규모 데이터셋을 구축하여 Visual Question Answering (VQA) 문제를 해결하고자 합니다. 이 프레임워크를 활용하여 OASIS라는 데이터셋을 개발하였고, 이는 다양한 언어와 문화적 상황을 반영한 0.92M개의 이미지와 14.8M개의 QA 쌍을 포함하고 있습니다. 이러한 데이터셋은 특히 자원이 부족한 언어에서의 질문에 대한 응답을 가능하게 합니다.

- **Technical Details**: OASIS 데이터셋은 음성, 이미지 및 텍스트를 통합하여 3.7M개의 발화 질문을 제공합니다. 또한, 데이터셋은 speech-only, text-only, speech+image, text+image의 네 가지 입력 조합을 지원하며, 영어와 아랍어 등 18개 나라의 다양한 실제 상황을 반영하도록 큐레이션되었습니다. 본 연구는 객체 인식 외에도 실용적이고 상식에 기반한 문화적 사고를 요구하는 작업을 평가합니다.

- **Performance Highlights**: 연구에서는 네 개의 폐쇄형 모델, 세 개의 오픈소스 모델, 그리고 하나의 파인튜닝된 모델을 벤치마킹하여 성능을 평가하였습니다. EverydayMMQA와 OASIS는 문화적 맥락 내에서의 일상 과제를 위한 다중 모달 (multimodal) 언어 모델 (LLMs) 구축을 위한 벤치마크 및 학습 데이터셋을 제공합니다. 프레임워크와 데이터셋은 커뮤니티에 공개될 예정입니다.



### EVALUESTEER: Measuring Reward Model Steerability Towards Values and Preferenc (https://arxiv.org/abs/2510.06370)
Comments:
          Preprint under review

- **What's New**: 이번 연구에서는 다양한 사용자 가치와 스타일 선호도를 반영한 EVALUESTEER 벤치마크를 도입하였습니다. 이는 대규모 언어 모델(LLMs)과 보상 모델(RMs)의 조정 가능성을 평가하기 위한 새로운 도구로, 심리학 및 인간-LLM 상호작용 문헌에 기초하고 있습니다. 특히, 이 벤치마크는 발생한 165,888개의 선호 쌍을 이용하여 사용자 프로필에 따른 응답 선택의 정확성을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: EVALUESTEER는 4가지 가치 차원(전통적, 세속적-합리적, 생존, 자기 표현)과 4가지 스타일 차원(장황함, 읽기 쉬움, 자신감, 따뜻함)을 기준으로 선호 쌍을 체계적으로 생성합니다. 연구에서는 총 6개의 오픈소스 및 프로프라이어터리 LLM 및 RM을 16가지 체계적인 프롬프트 조건과 6가지 선호 비교 시나리오 하에서 평가하였습니다. LLM과 RM의 조정 가능성을 평가하기 위해, 명확한 사용자의 가치 및 스타일 프로필을 바탕으로 시스템이 선호하는 응답을 선택하는 능력을 점검합니다.

- **Performance Highlights**: 연구 결과, 값과 스타일 프로필을 모두 제공했을 때 최고의 모델이 75%의 정확도로 올바른 응답을 선택했지만, 관련된 스타일과 가치 선호도만 제공할 때 99%의 정확도와 비교했을 때 여전히 25%의 격차가 있음을 확인했습니다. 또한, RMs는 세속적인 가치에 대한 경향성과 논리적 언어의 선호가 강한 경향이 있으며, 가치와 스타일 신호가 충돌할 경우 스타일적 선호를 우선시하는 경향을 나타내어 즉각적인 개선의 여지가 있음을 시사합니다.



### LLM Bias Detection and Mitigation through the Lens of Desired Distributions (https://arxiv.org/abs/2510.06354)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 출력 분포를 원하는 분포와 정렬하기 위한 방법을 제안합니다. 기존의 편향 완화 조치가 사회적 평등과 인구 통계적 균형을 촉진하는 데 주력했던 반면, 이 연구는 실제 분포와의 정렬에 중점을 둡니다. 연구자는 편향을 원하는 분포에서의 편차로 정의하고, 이 목표에 맞춰 LLM의 성별-직업 출력 분포를 조정하는 가중 적응 손실 기반의 미세 조정 방법을 제안하고 있습니다.

- **Technical Details**: 이 연구에서는 미국 노동 통계(2024)의 세 가지 직업 세트—남성 지배 직업, 여성 지배 직업 및 성별 균형 직업을 활용하여 LLM의 출력 분포를 조정합니다. 저자들은 기본적인 언어 모델링 기능을 유지하면서 LLM의 출력 분포를 원하는 분포와 일치시키기 위한 효과적인 손실 함수를 정의하였습니다. 연구 결과, 성 평등을 기준으로 한 편향은 거의 완전히 완화되었고, 현실적 조건에서는 30-75% 감소를 기록했습니다.

- **Performance Highlights**: 세 가지 마스킹 언어 모델에서 성별-직업 편향이 관찰되었으며, 자가 회귀 LLM은 평등 조건에서는 편향이 없었지만 현실적 조건에서는 상당한 편향을 보였습니다. Llama Instruct 모델은 자가 회귀 모델 중에서 50-62%의 편향 감소에 성공하며, 이러한 결과는 LLM의 출력이 실제 분포에 기반하도록 조정하는 것이 중요함을 보여줍니다.



### Type and Complexity Signals in Multilingual Question Representations (https://arxiv.org/abs/2510.06304)
Comments:
          Workshop on Multilingual Representation Learning at EMNLP 2025

- **What's New**: 이번 연구는 다국어 변환기 모델(multilingual transformer model)이 질문의 형태 통사적 속성(morphosyntactic properties)을 어떻게 나타내는지를 조사합니다. 우리는 7개 언어로 구성된 질문 유형 및 복잡성(Question Type and Complexity, QTC) 데이터셋을 도입하였으며, 이 데이터셋은 질문 유형 정보 및 의존 길이(dependency length), 트리 깊이(tree depth), 어휘 밀도(lexical density)와 같은 복잡성 측정 지표로 주석이 달려있습니다.

- **Technical Details**: 평가 방법론은 선택성 제어(selectivity controls)를 포함하여 회귀 레이블(regression labels)로 probing 방법을 확장하여 일반화 가능성(generality gains)을 정량화합니다. 우리는 고정된 Glot500-m(2023년 Imani et al.) 표현에 대한 층별 프로브(layer-wise probes)를 서브워드 TF-IDF 기초라인(subword TF-IDF baselines) 및 미세 조정된 모델(fine-tuned model)과 비교합니다.

- **Performance Highlights**: 결과에 따르면 통계적 특징(statistical features)은 명시적 표시가 있는 언어에서 질문을 효과적으로 분류하는 반면, 신경망 프로브(neural probes)는 세부적인 구조적 복잡성 패턴을 더 잘 캡처합니다. 우리는 이러한 결과를 사용하여 맥락적 표현(contextual representations)이 통계적 기초라인(statistical baselines)을 초과하는 상황과 매개변수 업데이트가 사전 훈련된 언어 정보의 가용성을 감소시키는지를 평가합니다.



### Reproducibility Study of "XRec: Large Language Models for Explainable Recommendation" (https://arxiv.org/abs/2510.06275)
- **What's New**: 이번 연구에서는 Ma 외(2024)의 논문 "XRec: Large Language Models for Explainable Recommendation"에서 제시된 내용을 재현하였습니다. 원 저자들은 XRec 모델을 소개하였으며, 이는 대형 언어 모델(LLMs)에게 사용자에게 제공되는 추천에 대한 포괄적인 설명을 생성하는 기능을 부여합니다. 우리는 Llama 3를 사용하여 원본 논문의 결과를 재현하였으며, Mixture of Experts 모듈의 입력 및 출력 임베딩을 수정하여 성능을 향상시키려 했습니다.

- **Technical Details**: XRec 프레임워크는 그래프 기반 협업 필터링(Collaborative Filtering)과 LLMs를 결합하여 설명 가능한 추천을 제공합니다. 이 구조는 연결된 사용자-아이템 상호작용 그래프에서 GNN을 사용하여 더 높은 차원의 협업 관계를 포착하는 방식으로 작동합니다. XRec은 MoE(adapted embeddings)를 통해 GNN 임베딩을 LLM의 토큰 수준 표현 공간에 맞춰 조정하여 두 가지 구조를 효과적으로 결합합니다.

- **Performance Highlights**: XRec은 개인 맞춤형 설명을 생성하는 데 효과적이며, 협업 정보를 통합함으로써 안정성이 향상되었습니다. 그러나 모든 지표에서 XRec이 모든 기준 모델을 지속적으로 능가하는 것은 아니었습니다. 우리의 확장된 분석은 Mixture of Experts 임베딩이 설명 구조에 미치는 중요성을 강조하며, 협업 신호가 언어 모델링과 어떻게 상호작용하는지를 보여줍니다.



### Language models for longitudinal analysis of abusive content in Billboard Music Charts (https://arxiv.org/abs/2510.06266)
- **What's New**: 이번 연구는 최근 7개 년도 동안의 Billboard 차트 곡들을 심층 학습 방법을 통해 분석하여 음악의 선정적 내용의 변화를 추적합니다. 이 연구에서는 감정 분석(sentiment analysis) 및 폭력적 내용 탐지(abuse detection)를 포함하여 음악의 내용 진화를 검토합니다. 결과적으로, 1990년대 이후 대중 음악에서 선정적인 내용이 유의미하게 증가하고 있다는 것을 발견했습니다.

- **Technical Details**: 연구는 심층 학습(deep learning) 및 대형 언어 모델(LLMs)을 활용하여 음악가사(lyrics)의 미세한 패턴을 포착하는 장기적(longevity) 분석을 수행합니다. 또한, 데이터셋을 Billboard 차트에서 선정하여 곡을 명시적(explicit) 또는 비명시적(non-explicit)으로 분류하는 방법론을 제공합니다. 이러한 방법은 기존의 단어 목록 기반 접근 방식에 비해 더욱 강력하고 적응 가능한 내용 탐지를 가능하게 합니다.

- **Performance Highlights**: 연구 성과는 Billboard의 대중 음악에서의 비속어, 성적 내용, 그리고 부적절한 언어 사용의 증가 추세를 강조하며, 이는 사회적 규범과 언어 사용의 변화를 반영합니다. 이러한 분석은 교육자와 정책 입안자들이 더 안전한 음악 환경을 위한 정보에 기반한 결정을 내리도록 도와줄 것으로 기대됩니다. 연구 결과는 음악의 발달과 관련하여 중요한 심리적 영향을 미칠 수 있는 주제를 제기합니다.



### A Comprehensive Survey of Hallucination in Large Language Models: Causes, Detection, and Mitigation (https://arxiv.org/abs/2510.06265)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각(hallucination) 문제에 대한 종합적인 조사를 제공합니다. 환각은 사실적으로 부정확한 정보를 생성하는 심각한 문제로, 모델의 신뢰성을 저하시킵니다. 연구는 환각의 원인과 탐지(detection), 완화(mitigation) 방법을 다각도로 살펴봅니다.

- **Technical Details**: LLMs의 개발 사이클 전반에 걸쳐 환각의 원인을 분석하고, 데이터 수집부터 추론(inference)까지 각 단계의 기여 요인을 제공합니다. 또한, 환각 탐지를 위한 체계적인 분류 체계를 제안하고, 여러 탐지 접근법의 장단점을 분석합니다.

- **Performance Highlights**: 탐지 및 완화 접근법의 성능을 비교하고, 향후 연구 방향을 제시합니다. 이 논문은 기존의 연구를 바탕으로 환각 원인 분석 및 환각 탐지 및 완화 기술의 종합 분석을 제공합니다. 최종적으로, 더욱 진실하고 신뢰할 수 있는 LLM 개발을 위한 기초를 제공합니다.



### Dual-stage and Lightweight Patient Chart Summarization for Emergency Physicians (https://arxiv.org/abs/2510.06263)
Comments:
          Accepted at the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT) 2025

- **What's New**: 이 연구에서는 긴급 상황에서의 전자 건강 기록(EHR) 요약을 위한 새로운 시스템을 소개합니다. 이 시스템은 환자 정보를 검색하는 Jetson Nano-R 장치와 요약을 생성하는 Jetson Nano-S 장치를 사용하여, 환자 프라이버시에 대한 고려와 더불어 오프라인 작업이 가능하게 설계되었습니다.

- **Technical Details**: 제안된 시스템은 두 단계로 구성되며, 첫 번째 단계에서는 EHR에서 관련 정보를 검색하고, 두 번째 단계에서는 검색된 텍스트를 바탕으로 요약을 생성합니다. 이 두 단계의 분리는 자원 제약을 고려한 효율적 처리를 가능하게 합니다. 요약 결과는 필수 정보 목록과 임상 질의를 기반으로 한 맥락-specific 내러티브로 구분됩니다.

- **Performance Highlights**: 예비 결과에 따르면, 이 시스템은 MIMIC-IV 데이터베이스와 실세계의 비공식 EHR을 기반으로 30초 이내에 유용한 요약을 생성할 수 있습니다. 특히, FA 점수를 통한 검증 방법을 통해 요약의 사실 정확성을 평가하여, 임상에서의 신뢰성을 높였습니다.



### Prakriti200: A Questionnaire-Based Dataset of 200 Ayurvedic Prakriti Assessments (https://arxiv.org/abs/2510.06262)
Comments:
          4 pages, 4 figures

- **What's New**: 이번 연구에서는 전통 아유르베다 원리에 따라 개인의 신체적, 생리적, 심리적 특성을 평가하기 위한 이중 언어(영어-힌디어) Prakriti Assessment Questionnaire를 활용한 새로운 데이터 세트를 제공했습니다. 이 설문지는 24개의 다중 선택 항목으로 구성되어 있으며, 체형, 식욕, 수면 패턴, 에너지 수준 및 기질을 포함하여 다양한 특성을 평가합니다. 이 데이터는 자동 채점 시스템을 통해 개별 특성을 dosha(바타, 피타, 카파) 점수와 매핑하여 신뢰할 수 있는 분석을 가능하게 합니다.

- **Technical Details**: 이 데이터 수집 방법은 아유르베다 원칙에 근거한 이중 언어 설문지를 사용합니다. 24개의 다중 선택 질문은 신체적 특성(예: 체형, 신장), 생리적 특성(예: 식욕, 수면), 심리적 특성(예: 기질)에 대한 정보를 포함하고 있습니다. Google Forms를 통해 수집된 데이터는 완전성과 일관성을 검토하여 구조화된 xlsx 파일 형태로 저장되며, 각 참여자의 점수를 포함합니다.

- **Performance Highlights**: 최종 데이터 세트에는 200명의 참여자가 포함되어 있으며, 67.5%가 평균 체중, 62%가 평균 신장을 보고했습니다. 데이터 분석 결과 Pitta가 우세한 구성(97명)이 많았고, 뒤이어 혼합형과 같은 다른 유형의 구성이 나타났습니다. 이 데이터 세트는 향후 아유르베다 기반 연구와 헬스케어 어플리케이션 개발에 중요한 자료로 활용될 수 있습니다.



### Scalable multilingual PII annotation for responsible AI in LLMs (https://arxiv.org/abs/2510.06250)
- **What's New**: 이번 연구는 다양한 규제 환경에서 개인 식별 정보(PII)의 신뢰할 수 있는 처리를 보장하기 위해 설계된 확장 가능한 다국어 데이터 주석 프레임워크를 소개합니다. 이 프레임워크는 13개의 저소득 언어 지역에서 약 336개 지역별 PII 유형에 대한 고품질 주석을 위한 것입니다. 단계별 인력 개입(HiTL) 주석 방법론을 통해 언어 전문 지식과 엄격한 품질 보증을 결합하여 메모리 유지율(Recall)과 잘못 분류된 긍정 사례(False Positive Rate)에서 큰 개선을 이루었습니다.

- **Technical Details**: 이 연구에서는 PII 주석을 위해 세 가지 단계(Pilot, Training, Production)로 나누어 진행되었습니다. 각 단계에서 작은 데이터 세트를 사용하여 초기 주석 문제를 식별한 후, 훈련 단계에서 정확한 PII 레이블링을 보장하기 위한 자료가 보강되었습니다. 마지막 생산 단계에서 표준화된 가이드라인을 바탕으로 품질 보증이 이루어져, 전 과정을 통해 지속적인 모니터링, 분석 및 피드백이 포함되었습니다.

- **Performance Highlights**: 주석 품질이 지속적으로 높은 수준을 유지할 수 있었던 것은 내부 품질 팀의 지속적인 모니터링과 피드백 덕분입니다. 연구에서는 주석자 간 합의(inter-annotator agreement) 및 오류 및 모호성의 근본 원인 분석(root-cause analysis)을 통해 성과를 객관적으로 측정하였습니다. 이 분석을 통해 더 나은 데이터 품질 및 LLM의 신뢰성을 확보하는 데 기여하였습니다.



### TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B (https://arxiv.org/abs/2510.06249)
Comments:
          It is work in progress

- **What's New**: 2025 다중모달 모델(Multimodal Models) 언어 챌린지는 인도의 다양한 저자원 언어(low-resource languages) 부족 문제를 해결하고자 합니다. 이 연구에서는 다국어 대형 언어 모델(multilingual large language model)에서 특정 내부 레이어의 교차 언어 유사성을 강화하는 것이 저자원 언어에서 고자원 언어(high-resource language)로의 번역 품질을 개선할 수 있는지를 조사합니다. 연구진은 Centered Kernel Alignment (CKA)와 REPINA라는 정규화 방법을 결합하여 TRepLiNa라는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 연구는 멀티링구얼 모델에서 레이어별 정렬(layer-wise alignment)을 시스템적으로 분석하는 첫 시도를 하고 있습니다. CKA와 TRepLiNa(CKA+REPINA)의 효과를 비교하기 위해, 특정 중간 레이어(약 10~15 레이어)의 유사성을 높이는 방법이 가장 효과적임을 보여줍니다. 이 방법론은 데이터가 부족한 설정에서도 효과적인 저자원 언어 번역 개선을 위해 활용됩니다.

- **Performance Highlights**: TRepLiNa를 적용함으로써 가중 복합 점수(weighted composite score)에서 실질적인 개선이 있음을 보였으며, 이 방법론의 적용 시기와 장소에 대한 가이드라인도 제공합니다. 실험 결과, TRepLiNa는 일반적으로 15번째 레이어에서 일관된 성과를 보였으며, 저자원 언어에서 고자원 언어로 번역 시, 번역 품질이 향상됨을 나타냅니다.



### Evaluating Embedding Frameworks for Scientific Domain (https://arxiv.org/abs/2510.06244)
- **What's New**: 이 논문은 특정 도메인 데이터에 따라 같은 단어가 서로 다른 의미와 표현을 가질 수 있음을 강조하며, 과학 분야에 최적화된 단어 표현 알고리즘과 토큰화 방법을 연구하고 있습니다. 특히, 과학 기술 문헌에 적합한 새로운 평가 수트를 구축하여 다양한 단어 표현 및 토큰화 알고리즘의 성능을 평가하는 것을 목표로 하고 있습니다.

- **Technical Details**: 자연어 처리(NLP)에서 효과적인 단어 표현은 언어 이해, 텍스트 생성 및 감정 분석과 같은 작업에 중대한 영향을 미치며, 현재 사용되는 여러 토큰화 및 단어 임베딩 방법의 성능을 과학 분야에서 평가하는 데 중점을 두고 있습니다. Byte Pair Encoding(BPE), WordPiece, Unigram Tokenizer 등의 다양한 토큰화 방법과 Word2Vec, GloVe, FastText와 같은 단어 임베딩 기법이 소개되며, 각 방법은 특정 장단점을 가지고 있습니다.

- **Performance Highlights**: 연구는 다양한 다운스트림 NLP 과제를 포함한 포괄적인 평가 수트를 구축하여 과학 분야에서의 단어 표현과 토큰화 모델을 비교하고 있습니다. 특히, 낮은 리소스 모델인 Word2Vec과 계산적으로 더 많은 자원을 소모하는 Transformer 기반 모델의 성능 비교를 통해 과학 분야에서의 단어 표현 문제에 대한 통찰력을 제공하고 있습니다.



### CoT Referring: Improving Referring Expression Tasks with Grounded Reasoning (https://arxiv.org/abs/2510.06243)
Comments:
          MLLM, Referring Expression Segmentation

- **What's New**: 이 논문은 Referring Expression (RE) 작업에 대해 새로운 추론 메커니즘인 CoT Referring (CoTR)을 제안합니다. 이 방법은 복잡한 질의에서 언어와 이미지를 통합하여 모델의 성능을 향상시킵니다. CoTR은 모델이 순차적인 논리(logic)를 명확히 모델링할 수 있도록 하여, 목표 객체를 올바르게 찾을 수 있게 합니다.

- **Technical Details**: CoT Referring 접근법은 텍스트 구조를 체계적으로 분석하여 각 단계에서 관계를 식별하고 참조 정렬을 보장합니다. 새로운 출력 형식을 강제하기 위해 기존 데이터를 재구성하고, 복잡한 참조 사례에 대한 평가 기준을 개발했습니다. 이 평가 기준은 3개 이상의 교차 관련 객체가 포함된 복합 참조 표현을 위해 특별히 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 RefCOCO/+/g 벤치마크에서 기존 모델에 비해 2.5%의 성능 향상을 보여주었습니다. 새로운 평가 기준에 따른 실험 결과, 우리의 접근법이 복잡한 참조 표현에서 목표 로컬라이제이션(target localization)을 효과적으로 개선하는 것을 확인하였습니다.



### Transparent Reference-free Automated Evaluation of Open-Ended User Survey Responses (https://arxiv.org/abs/2510.06242)
Comments:
          EMNLP Industry Track

- **What's New**: 본 연구에서는 인간 작성 설문 응답의 평가를 위한 두 단계 평가 프레임워크를 제안합니다. 기존의 자동 평가 방법은 LLM(대형 언어 모델)으로 생성된 텍스트를 대상으로 하며, 인간 작성 응답의 독특한 특성을 적절히 평가하지 못했습니다. 우리의 접근 방식은 비정상적인 응답을 필터링하고 세 가지 차원인 노력(effort), 관련성(relevance), 완전성(completeness)을 평가합니다.

- **Technical Details**: 이 프레임워크는 비정상적인 응답을 제거하는 gibberish filtering 단계를 포함합니다. 이후, 현실 세계의 설문 데이터에 대한 실증 분석을 기반으로 LLM 능력을 활용하여 각각의 응답을 세 가지 차원에서 평가합니다. 이를 통해 응답의 질을 예측하고 불량 응답을 거부하는 실제 응용 프로그램에서도 높은 효율성을 발휘합니다.

- **Performance Highlights**: 영어와 한국어 데이터셋에 대한 검증 결과, 제안한 프레임워크는 기존 지표들을 초월하며, 전문가 평가와 강한 상관관계를 보입니다. 이는 설문 연구에서의 응답 품질 예측과 응답 거부뿐만 아니라 실제 환경에서의 적용 가능성을 더욱 높입니다.



### Knowledge Graph-Guided Multi-Agent Distillation for Reliable Industrial Question Answering with Datasets (https://arxiv.org/abs/2510.06240)
Comments:
          41 pages, 12 figures, 6 tables

- **What's New**: KG-MASD(Knowledge Graph-guided Multi-Agent System Distillation)를 제안하여 산업 QA 시스템의 안전성과 신뢰성을 높이고자 하였습니다. 이 새로운 접근 방식은 Markov Decision Process(MDP)로 모델링되어, Knowledge Graph를 활용하여 상태 표현을 풍부하게 하고 수렴을 보장합니다. KG-MASD는 고신뢰도의 instruction-tuning 데이터를 생성하고, 경량화된 학생 모델에 깊은 추론 능력과 검증 가능성을 동시에 이식할 수 있는 여지를 제공합니다.

- **Technical Details**: KG-MASD는 각 단계에서 Knowledge Graph를 활용하여 상태를 업데이트하는 방법으로 구성되며, 이 과정을 통해 얻은 고품질의 도메인 일치 triple들이 학생 모델의 학습과정에 긍정적인 영향을 미칩니다. 이론적으로, KG-guided priors는 증류 효율성을 개선하는 데 도움이 되며, 시스템이 고위험 환경에서 신뢰할 수 있는 결과를 생산할 수 있도록 합니다. 더불어 KG-MASD는 복잡한 산업 시나리오에서도 학생 모델이 교사 모델의 복잡한 추론 패턴을 믿을 수 있게 전이할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, KG-MASD는 기존 모델들에 비해 정확도가 2.4%에서 20.1%까지 향상되었으며, 신뢰성을 크게 증가시켜 안전-critical 산업 시나리오에서의 AI 배치를 가능하게 했습니다. 또한, KG-MASD는 Hallucination 현상을 완화하고, 단일 교사 모델 및 다중 에이전트 기법 대비 성능이 뛰어난 것으로 입증되었습니다. 이러한 성과는 KG-MASD가 Trustworthy AI 제공에 기여할 수 있음을 보여줍니다.



### OpenStaxQA: A multilingual dataset based on open-source college textbooks (https://arxiv.org/abs/2510.06239)
- **What's New**: 이번 논문에서는 대학 수준의 교육용 애플리케이션에 특화된 OpenStaxQA 평가 기준을 제시합니다. 이는 Creative Commons 라이선스 하에 제공되는 43개의 오픈소스 대학 교과서를 기반으로 하며, 영어, 스페인어, 폴란드어로 제공됩니다. 논문에서는 약 70억 개의 매개변수를 가진 대형 언어 모델(LLMs)을 사용하여 이 데이터셋을 파인 튜닝하고 평가한 결과를 공유합니다.

- **Technical Details**: OpenStaxQA 데이터셋은 물리학, 생명과학, 수학, 비즈니스 및 사회 과학의 다양한 분야에 걸쳐 43개의 대학 수준 교과서에서 발췌한 문제-해결 쌍을 포함하고 있습니다. 이를 위해 스크래핑 방법론을 사용하여 HTML 내 문제와 해결 방안을 잘 구조화된 태그로 식별했습니다. 또한 MathML 데이터를 LaTeX 포맷으로 변환하는 작업을 수행함으로써 훈련과 추론 속도를 개선했습니다.

- **Performance Highlights**: 모델 평가는 GPT-4를 오라클(Oracle)로 사용하여 수행하였으며, 70:30의 비율로 훈련 및 테스트 데이터셋을 분할하여 QLoRA 어댑터를 3회 에폭 동안 훈련했습니다. 결과적으로 Llama2-7b 및 Llemma-7b와 같은 여러 LLM 모델이 훈련되었으며, AI2 reasoning challenge ‘challenge dev’ 데이터셋에서의 성능 평가가 포함됩니다. 이 과정에서 OpenStaxQA가 다른 과제에서의 성능 개선에 기여할 수 있을 것으로 기대됩니다.



### AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs (https://arxiv.org/abs/2510.07293)
Comments:
          26 pages, 23 figures, the code is available at \url{this https URL}

- **What's New**: 오디오 처리의 새로운 벤치마크인 AudioMarathon이 소개되었습니다. 이 벤치마크는 대규모 오디오 언어 모델(LALM)을 평가하기 위해 긴 형식의 오디오 입력을 처리하는 능력을 중점적으로 다룹니다. AudioMarathon은 90초에서 300초까지의 오디오 입력, 다양한 도메인 커버리지 및 복잡한 추론 태스크를 포함하는 세 가지 기본 축을 바탕으로 설계되었습니다. 이 연구는 LALM의 성능 저하와 메모리 효율성을 높이기 위한 기법을 분석합니다.

- **Technical Details**: AudioMarathon은 긴 형식의 오디오를 효과적으로 이해하고 추론할 수 있는 LALM의 능력을 평가하기 위해 고안된 포괄적인 오디오 벤치마크입니다. 이 벤치마크는 90초에서 300초까지의 오디오 입력을 포함하여 연속적인 사운드 환경에서의 복잡한 데이터를 다룹니다. 또한, AudioMarathon은 Speech Context Understanding, Audio Scene Understanding, Voice Characteristic Identification 세 가지 카테고리로 태스크를 구성하고 있습니다. 복잡한 추론 문제를 해결하기 위해 멀티 홉(multi-hop) 추론을 포함한 평가 방법론을 채택했습니다.

- **Performance Highlights**: 우리는 AudioMarathon에서 최첨단 LALM의 성능을 평가한 결과, 입력 길이가 증가함에 따라 성능이 크게 저하된다는 것을 관찰했습니다. 현재 모델 간의 성능 차이는 여러 가지 단점과 과제를 강조하며, 보다 나은 시간적 추론 및 메모리 효율성을 제공하는 아키텍처 개발이 필요하다는 것을 보여줍니다. 이 연구 결과는 오디오 및 다중 모달 연구 커뮤니티가 더욱 발전된 오디오 이해 모델을 개발하도록 촉진할 것으로 기대됩니다.



### Machines in the Crowd? Measuring the Footprint of Machine-Generated Text on Redd (https://arxiv.org/abs/2510.07226)
- **What's New**: 이번 연구는 Reddit에서의 기계 생성 텍스트(MGT)의 대규모 특성을 최초로 분석한 것이다. 2022년부터 2024년까지의 활동 데이터를 기반으로 51개의 써브레딧을 연구하였으며, MGT가 커뮤니티에 통합되는 방식을 초점으로 하고 있다. 이를 통해 MGT의 분포, 시간적 패턴, 그리고 사용자 반응에 대한 새로운 통찰을 제공하고 있다.

- **Technical Details**: 연구는 Reddit에서 1,000개의 가장 인기 있는 써브레딧을 검토하고, 이 중 51개를 선정하여 정보 탐색, 사회적 지원, 의견 교환 등의 카테고리로 나누었다. 데이터는 PushShift API를 통해 수집되었으며, 38,074,021개의 댓글과 4,073,586개의 제출물이 분석됐다. MGT와 인간 생성 텍스트 간의 차이를 탐구하기 위해 이중 텍스트 분류 작업으로 접근하였다.

- **Performance Highlights**: MGT는 Reddit의 특정 커뮤니티에서 최대 9%까지 나타나는 것으로 추정된다. MGT는 기술적 지식과 사회적 지원을 주제로 한 서브레딧에서 더 많이 나타났으며, AI 어시스턴트의 언어 특징을 나타내는 사회적 신호를 전송한다. 또한 MGT는 인간 생성 콘텐츠와 유사한 참여 수준을 달성하고 있으며, 일부 경우에는 더 높은 수준의 참여를 이끌어내는 것으로 나타났다.



### A Multi-Agent Framework for Stateful Inference-Time Search (https://arxiv.org/abs/2510.07147)
- **What's New**: 이 연구에서는 상태 기반의 다중 에이전트 진화 탐색(stateful multi-agent evolutionary search)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 지속적인 추론 상태(persistent inference-time state), 적대적 변이(adversarial mutation), 진화적 보존(evolutionary preservation)을 결합하여 이전의 비상태(stateless) 접근 방식에서 벗어났습니다. 그 결과, 복잡한 테스트 케이스 생성을 통해 더욱 견고한 유닛 테스트를 자동으로 생성할 수 있게 되었습니다.

- **Technical Details**: 제안된 시스템은 여러 LLM 호출을 통해 후보 엣지 케이스를 제안하는 액터(actor), 환경을 변이시켜 견고성 갭을 드러내는 적대자(adversary), 진화 검색에 사용되는 보상을 부여하는 비평가(critic)로 구성됩니다. 각 단계에서 상태 정보를 유지하여 이전 단계의 피드백을 활용하고, 이는 선형 탐색을 넘어선 구조적 문제 해결을 가능하게 합니다. 더불어, 액터는 대형 언어 모델(LLM)의 맥락 학습을 통해 지속적인 상태를 기반으로 후보를 생성하며, 이는 전통적인 기법보다 높은 샘플 효율성을 제공합니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 HumanEval 및 TestGenEvalMini와 같은 유닛 테스트 벤치마크에서 비상태 단일 단계 기준선에 비해 상당히 높은 커버리지(coverage)를 달성했습니다. 세 가지 다양한 LLM 모델인 Llama, Gemma, GPT를 활용하여 유연한 테스트 케이스 생성을 보여 주었으며, 향상된 커버리지와 견고성을 통해 새로운 코드베이스에 대한 적응 능력이 뛰어남을 입증했습니다. 이러한 결과는 지속적인 추론 상태와 진화적 탐색이 유닛 테스트 생성에 실질적인 개선 효과를 줄 수 있음을 시사합니다.



### The Cognitive Bandwidth Bottleneck: Shifting Long-Horizon Agent from Planning with Actions to Planning with Schemas (https://arxiv.org/abs/2510.07091)
Comments:
          22 pages

- **What's New**: 이 논문에서는 오픈 월드 자율성을 위해 장기적인 계획과 여러 상호작용이 필요한 장기 과제 작업에서 대형 언어 모델(LLM)의 효과적인 작동을 가능하게 하는 방법을 제안합니다. 전통적인 방법은 실행 가능한 행동 목록을 제공하여 계획하지만, 환경 행동 공간이 조합적으로 폭발하는 경우 이러한 접근법은 비현실적일 수 있습니다. 이 연구는 행동 표현 방식의 최적화된 선택을 위한 시스템적 연구로, 행동 기반 계획(PwA)과 스키마 기반 계획(PwS)의 두 가지 접근 방식을 비교합니다.

- **Technical Details**: 인간은 추상적인 행동 템플릿을 구체적인 실행 가능한 단계로 변환하여 결정을 내리는 과정을 겪습니다. 본 논문에서는 이러한 과정을 스키마 기반 계획(PwS)이라고 명명하고, LLM의 최신 발전을 통해 장기 과제를 해결하기 위한 자율 에이전트의 필요성을 강조합니다. 연구 결과, PwA 방식은 짧은 행동 공간에서 더 나은 성능을 보이는 반면, 긴 행동 공간에서는 PwS 방식이 유리하다는 것을 보여주고 있습니다.

- **Performance Highlights**: 행동 공간의 크기에 따라 성능 변화가 관찰되었으며, 인플렉션 포인트는 ALFWorld(~35 actions)에서 PwA가 평균 33.4% 더 높은 성능을 보였지만, SciWorld(~500 actions)에서는 PwS가 8.1%의 우위를 차지했습니다. 이는 PwA가 행동 공간의 크기에 따라 성능이 저하되는 반면, PwS는 확장 가능성에서 더 우수함을 나타냅니다. 마지막으로 이 연구는 모델의 효율성을 높이기 위한 구체적인 가이드를 제공합니다.



### RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning (https://arxiv.org/abs/2510.06994)
- **What's New**: 이번 논문에서는 RedTWIZ라는 적응형 및 다양한 멀티 턴(red teaming) 프레임워크를 통해 AI 지원 소프트웨어 개발에서 대규모 언어 모델(LLM)의 로버스트니스(robustness)를 감사(audit)하는 방법을 소개합니다. 저자들은 LLM의 대화적 jailbreak에 대한 체계적인 평가와 다양한 공격 전략을 결합하여 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: RedTWIZ는 세 가지 주요 연구 흐름에 의해 구동됩니다: (1) LLM 대화 jailbreak의 강력하고 체계적인 평가, (2) 구성(compositional), 현실적(realistic)이며 목표 지향(target-oriented)인 jailbreak 대화 전략을 지원하는 다양한 생성형 멀티 턴 공격 세트, (3) 특정 LLM의 취약점에 맞춰 공격을 적응적으로 계획하고 발생시키는 계층적 공격 계획자(hierarchical attack planner)입니다.

- **Performance Highlights**: 실험적 결과는 멀티 턴 적대적 공격 전략이 최첨단 LLM을 성공적으로 유도하여 불안전한 생성물(unsafe generations)을 만들어낼 수 있음을 보여줍니다. 이는 LLM의 로버스트니스 강화를 위한 추가 연구의 필요성을 강조합니다.



### VelLMes: A high-interaction AI-based deception framework (https://arxiv.org/abs/2510.06975)
Comments:
          9 pages. 9 figures. 1 table. This is a preprint of a paper that was presented at the Active Defense and Deception Workshop colocated with IEEE EuroS&P 2025 conference

- **What's New**: 이번 논문은 VelLMes라는 AI 기반의 새로운 기만 프레임워크를 소개합니다. 이 프레임워크는 SSH Linux shell, MySQL, POP3 및 HTTP와 같은 여러 프로토콜과 서비스의 시뮬레이션을 지원합니다. VelLMes는 사용자 요구에 따라 다양한 기만 설계를 위한 선택지를 제공하며, 인간 사용자와의 상호작용 및 사실감을 중시합니다.

- **Technical Details**: VelLMes는 LLM(대형 언어 모델)을 기반으로 하여 사용자가 시뮬레이션 할 수 있는 다양한 서비스를 제공합니다. SSH Linux shell인 shelLM은 세밀한 프롬프트 엔지니어링을 통해 원하는 행동을 이끌어내며, 모든 출력은 LLM으로 생성되기 때문에 실제 커맨드 실행의 위험이 없습니다. 논문에서는 LLM의 생성 능력과 기만 능력을 평가하는 세 가지 유형의 평가를 수행했습니다.

- **Performance Highlights**: 연구 결과, 89명의 인간 공격자를 대상으로 한 실험에서 전체 공격자의 약 30%가 LLM 기반의 honeypot과 상호작용할 때 실제 시스템으로 혼동하였으며, SSH Linux shell honeypot은 90% 이상의 명령에 대해 정확한 응답을 생성했습니다. 이러한 결과는 LLM이 사이버 기만의 유용한 자원으로 활용될 수 있음을 보여줍니다.



### Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces (https://arxiv.org/abs/2510.06953)
- **What's New**: 이번 연구에서는 Uniform Information Density (UID) 가설을 바탕으로 대형 언어 모델(LLM)의 추론 품질과 정보 전달의 균일성을 분석합니다. 새로운 엔트로피 기반 단계적 정보 밀도 메트릭을 제안하며, 지역적 및 전역적 균일성 점수를 도입합니다. 실험 결과, 단계별 균일성이 우수한 이론적 시각을 제공하는 것뿐만 아니라, 실제 성능을 개선하는 데도 기여하며, 추론 추적에서 정보 밀도 균일성이 신뢰할 수 있는 평가 기준임을 입증합니다.

- **Technical Details**: 연구에서는 언어를 신호로 간주하고, 이 신호를 제한된 용량의 노이즈 채널을 통해 전송하는 UID 가설을 모델링합니다. 각 말(u) 단위의 예상치 못한 정도를 나타내는 갑작스러운 예기치 못한 사건(surprisal) 개념을 정의하고, 이를 토대로 정보를 단계별로 측정하는 새로운 방법을 개발했습니다. 실험에서는 LLM이 생성하는 사고 과정의 정보 흐름을 분석하고, 지역적 및 전역적 균일성을 스텝 기반 정보 밀도를 통해 평가합니다.

- **Performance Highlights**: 연구 결과, 정보 밀도의 균일성이 추론 품질을 예측하는 데 중요한 요소임을 보여줍니다. LLM의 정확도가 10~32% 향상되는 것으로 나타났으며, 이는 LLM의 적절한 정보 균일성을 유지하는 것이 중요하다는 것을 강조합니다. 올바른 추론 추적은 급격한 정보 밀도 상승을 피하는 경향이 있으며, 이는 더 신뢰할 수 있는 추론 시스템 구축에 있어 균일성이 중요한 진단 기준임을 시사합니다.



### Crossing Domains without Labels: Distant Supervision for Term Extraction (https://arxiv.org/abs/2510.06838)
Comments:
          Accepted at EMNLP Industry Track 2025

- **What's New**: 자동 용어 추출(ATE)은 자연어 처리(NLP) 작업에서 문서 태깅, 온톨로지 구성 및 특허 분석과 같은 주요 구성 요소입니다. 기존의 최첨단 방법들은 비싼 인간 주석이 필요하며, 도메인 전이에서 어려움을 겪고 있어 실용적으로 적용하기 어렵습니다. 이에 우리는 일곱 개의 다양한 도메인에 걸친 포괄적인 벤치마크를 도입하고, 이를 통해 성능 평가를 문서 및 코퍼스 수준에서 수행하여 보다 견고하고 확장 가능한 솔루션 필요성을 강조합니다.

- **Technical Details**: 우리의 접근 방식인 DiSTER(Distant Supervision for Term Extraction with Robustness)는 LLM을 활용하여 인간 주석 없이도 데이터의 생산성을 높이고, 도메인 간의 확장성을 가능하게 합니다. LLM을 사용하여 생성된 유사 레이블을 활용하여 비즈니스 및 과학 도메인에서 데이터의 일반성을 보장합니다. 또한 경량 포스트-혹(정후) 휴리스틱을 도입하여 문서 수준의 일관성을 높이고 F1 점수를 평균 10% 향상시키는 데 기여합니다.

- **Performance Highlights**: 본 연구에서 우리는 다섯 개의 도메인에서 기존 방법보다 뛰어난 성능을 보였으며, 평균적으로 10% 포인트 향상을 달성했습니다. 우리의 모델은 LLM 기반의 백박스 모델과 비교하여 우수한 성능을 기록하며, GPT-4o와의 경쟁에서도 상대적으로 견고한 결과를 보여줍니다. 우리는 이 데이터셋과 파인튜닝된 모델을 공개하여 미래의 연구를 지원할 예정입니다.



### Exposing Citation Vulnerabilities in Generative Engines (https://arxiv.org/abs/2510.06823)
Comments:
          12 pages, under-reviewing at a conference

- **What's New**: 본 논문은 Generative Engines (GEs)가 타당한 인용 출처를 제공하는지 평가하는 새로운 기준을 제안합니다. GEs는 웹 검색과 답변 생성 기능을 통합하여 사용자 질문에 답변하므로, 공격자가 악의적 콘텐츠를 주입할 위험이 큽니다. 제출된 자료는 웹 콘텐츠를 중독 공격에서 방어하기 위한 인용 출처에 대한 새로운 기준을 제시하며, 이는 사실성이 결여된 대답을 방지하기 위한 것이다.

- **Technical Details**: GEs의 모델은 사용자 질문과 개인화 정보를 입력으로 받아 텍스트 답변을 생성하는 함수로 formalization됩니다. 이 시스템은 두 가지 주요 구성 요소인 콘텐츠 검색(content retrieval)과 답변 생성(answer generation)으로 구성되어 있습니다. 콘텐츠 검색 단계에서 웹 검색 결과를 활용해 여러 개의 쿼리를 수행하고 검색된 웹 소스에서 정보를 수집하여 답변을 생성하는 방식입니다.

- **Performance Highlights**: 연구 결과는 일본의 정치적 대답에서 60%에서 65%가 공식 당 홈페이지에서 인용된 반면, 미국에서는 25%에서 45%만 해당된다는 것을 보여주었습니다. 저장된 정보 퀄리티에 따라 웹 콘텐츠 주입 장벽(content-injection barrier)에 따른 출처의 평가도 진행되었으며, 낮은 장벽의 출처들이 비공식 및 덜 신뢰할 수 있는 정보를 반영하는 경향이 있다는 것을 확인했습니다.



### GPT-5 Model Corrected GPT-4V's Chart Reading Errors, Not Prompting (https://arxiv.org/abs/2510.06782)
- **What's New**: 이번 연구는 제로샷(Zero-Shot) 대형 언어 모델(LLMs)과 프롬프트 사용이 차트 읽기 작업에 미치는 영향을 정량적으로 평가한 결과를 제시합니다. 우리는 107개의 시각화 질문을 통해 에이전틱 GPT-5와 다중 모달 GPT-4V 간의 추론 정확성을 비교했습니다. 결과적으로, 모델 아키텍처는 추론 정확성에서 우위를 차지하며, 특히 GPT-5가 정확도를 크게 향상시켰고 프롬프트 변형은 그에 비해 미미한 효과를 보였습니다.

- **Technical Details**: 연구는 차트 읽기 작업에서 LLM 응답 방식과 최신 구현이 전통적 모델보다 효율성이 높은지를 탐구합니다. 우리는 CHART-6 벤치마크를 활용하여 GPT-4V로 잘못된 응답을 한 질문을 식별하고, 여러 LLM 모델과 프롬프트 조건 조합을 통해 평가했습니다. 사용한 주요 데이터셋은 107개의 질문으로 구성되며, 이는 다양한 프롬프트를 사용하여 반복적으로 테스트되었습니다.

- **Performance Highlights**: 연구 결과, GPT-5는 모든 데이터셋에 걸쳐 GPT-4o보다 현저히 더 우수한 성능을 보였습니다. 프롬프트 조건들의 차이점은 상대적으로 작지만, GPT-5의 동작은 전반적으로 더 나은 정확성을 입증했습니다. 부트스트랩 신뢰 구간을 통해 정확성의 통계적 유의성도 평가하였으며, 이러한 결과는 LLM의 성능 향상에 대한 중요한 통찰을 제공합니다.



### Evolving and Executing Research Plans via Double-Loop Multi-Agent Collaboration (https://arxiv.org/abs/2510.06761)
- **What's New**: 이 논문에서는 연구 문제를 자동으로 해결하기 위한 새로운 Double-Loop Multi-Agent (DLMA) 프레임워크를 제안합니다. 이 프레임워크는 두 가지 루프로 구성되어 있으며, 교수 에이전트들이 주도하는 리더 루프는 연구 계획을 발전시키고, 박사 과정 학생 에이전트들이 참여하는 팔로워 루프는 이 계획의 실행을 담당합니다. DLMA는 연구 제안서를 반복적으로 생성하고 조정하여 솔루션 공간을 효과적으로 탐색하는 데 중점을 둡니다.

- **Technical Details**: DLMA 프레임워크는 이층 최적화 문제(bilevel optimization problem)을 해결하기 위해 설계되었습니다. 리더 루프에서는 교수 에이전트들이 초기 연구 계획을 생성하고, 세 가지 유형의 회의(참여 회의, 개선 회의, 통합 회의)를 통해 제안서를 개발합니다. 팔로워 루프는 박사 과정 학생 에이전트들이 최적의 계획을 실행하며, 실행 과정에서 회의를 통해 조정합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DLMA는 ACLAward 및 Laboratory와 같은 벤치마크에서 자동 평가 점수에서 최첨단 성능을 보여줍니다. 이 시스템은 강력한 기준선을 크게 초월하여 우수한 연구 성과를 도출했습니다. 절제 연구(ablation studies)는 두 루프가 서로 다른 중요 역할을 수행하여 혁신성을 이끌어내고 실행의 신뢰성을 보장함을 확인했습니다.



### Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities (https://arxiv.org/abs/2510.06743)
Comments:
          The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 기반으로 한 OCR(광학 문자 인식)에 대한 평가 프레임워크의 필요성을 제시하고 있습니다. 기존의 평가 지표들이 역사적 문서에 대한 특정 오류와 시간적 편향을 포착하는 데 실패하는 점을 강조하며, 새로운 메트릭인 HCPR(역사적 문자 보존 비율)과 AIR(구식 삽입 비율)을 도입하였습니다. 이 방법론은 디지털 인문학에 종사하는 연구자들에게 모델 선택과 품질 평가를 위한 가이드를 제공합니다.

- **Technical Details**: 연구에서는 18세기 러시아 문서를 대상으로 한 LLM 기반 역사적 OCR 평가 프레임워크를 제시합니다. 수집된 데이터는 러시아 시민 서체로 인쇄된 428개의 독특한 18세기 도서에서 1,029페이지를 포함하고 있으며, 기존 OCR 시스템들이 어려움을 겪는 특유의 오탈자 및 고전 문법 형태를 포함하고 있습니다. 또한, 각 모델의 출력 변동성을 감안한 종합적인 안정성 테스트를 수행했습니다.

- **Performance Highlights**: 실험 결과, Gemini 및 Qwen 모델이 기존 OCR 시스템보다 뛰어난 성능을 보였으나, '과거화(over-historicization)'라는 요소로 인해 올바르지 않은 역사적 시점에서 고어 문자 삽입 현상이 발생했습니다. LLM들이 특정 역사적 문서에 대해 예상치 못한 시간적 편향을 보임으로써, 전통적인 평가 방법들이 이러한 문제를 탐지할 수 없는 한계가 있음을 확인하였습니다. 결과적으로, LLM을 활용한 역사적 문서 변환의 정확성을 높이기 위해서는 새로운 평가 방안이 필요하다는 결론을 내고 있습니다.



### Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG) (https://arxiv.org/abs/2510.06719)
Comments:
          Under review

- **What's New**: 이 논문은 Differentially Private Synthetic Retrieval-Augmented Generation (DP-SynRAG)이라는 프레임워크를 제안합니다. 이 프레임워크는 대규모 언어 모델(LLMs)을 사용하여 차별적 프라이버시(Privacy)를 보장하는 합성 RAG 데이터베이스를 생성합니다. 기존의 방법과 달리, 생성된 합성 텍스트는 반복적으로 사용할 수 있어 재노이즈 주입과 추가적인 프라이버시 비용을 피할 수 있습니다. DP-SynRAG는 RAG 작업에 필요한 핵심 정보를 보존하면서 성능을 개선하는 특성도 가지고 있습니다.

- **Technical Details**: DP-SynRAG는 LLM을 사용하여 데이터베이스의 서브샘플 기록을 모방하는 텍스트를 생성하는 맞춤형 예측(private prediction)을 통해 고품질의 사적인 텍스트 생성을 달성합니다. 이 과정에서 문서 레벨 임베딩(document-level embeddings)과 키워드 기반 클러스터링을 사용하여 의미적으로 유사한 문서들을 그룹화합니다. 이렇게 생성된 데이터는 이후의 RAG 작업에 사용할 수 있으며, 추가적인 프라이버시 비용을 발생시키지 않도록 설계되었습니다. 저자들은 세 가지 데이터셋을 사용해 DP-SynRAG의 성능을 검증하며 기존의 개인 RAG 방법들을 초월하는 결과를 보고합니다.

- **Performance Highlights**: DP-SynRAG는 고정된 프라이버시 예산을 유지하면서 기존의 최첨단 개인 RAG 시스템보다 뛰어난 성능을 보여줍니다. 이 방법은 스케일러블한 솔루션을 제공하여 프라이버시를 보장하는 RAG 애플리케이션에 적합합니다. 실험 결과는 DP-SynRAG가 다수의 질의에 대해 높은 효용성과 우수한 데이터 품질을 유지함을 입증합니다. 이러한 특징들은 이 논문의 기여가 민감한 정보를 다루는 데이터베이스에서 실질적으로 사용될 수 있음을 시사합니다.



### XLSR-Kanformer: A KAN-Intergrated model for Synthetic Speech Detection (https://arxiv.org/abs/2510.06706)
Comments:
          Accepted to 2025 IEEE International Conference on Advanced Video and Signal-Based Surveillance

- **What's New**: 최근 음성 합성 기술의 발전으로 인해 정교한 스푸핑 공격이 증가하면서 자동 화자 인증 시스템에 심각한 도전 과제가 발생하고 있습니다. 본 논문은 XLSR-Conformer 모델의 전통적인 Multi-Layer Perceptron (MLP)을 Kolmogorov-Arnold Network (KAN)으로 대체하는 새로운 접근 방식을 제안합니다. KAN은 Kolmogorov-Arnold 표현 정리에 기반한 강력한 유니버설 근사기입니다. 실험 결과, KAN을 XLSR-Conformer 모델에 통합함으로써 Equal Error Rate (EER)에서 60.55% 성능 향상을 확인했습니다.

- **Technical Details**: 자동 화자 인증(ASV) 시스템은 개인의 독특한 음성 특성을 분석하여 신원 인증에 중요한 역할을 합니다. ASV 기술은 고도화된 음성 합성 기술로 인한 위협에 대응하기 위해 지속적으로 진화하고 있으며, 최근에는 자가 지도 학습(self-supervised learning, SSL) 모델을 활용한 연구가 증가하고 있습니다. 이 논문에서는 SSL 모델에서 추출한 특징을 향상시키기 위해 KAN을 활용하는 새로운 구조를 제시하며, KAN 층의 구성 및 특징에 대해서도 설명합니다.

- **Performance Highlights**: 제안된 Kanformer 인코더는 ASVSpoof2021 DF 세트에서 새로운 최첨단 성능을 달성하며, 경쟁 시스템과 동등한 성능을 발휘합니다. KAN을 사용한 구조는 다양한 SSL 아키텍처에 대해 강력한 견고성을 보여주며, SSL 기반 모델에서 향후 합성 음성 탐지 혁신의 유망한 방향성을 제시합니다. 본 연구는 고차원 데이터 처리에 적합한 KAN의 장점을 입증하며, 심각한 망각(catastrophic forgetting) 문제를 해결하는 데도 효과적임을 보여줍니다.



### Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation (https://arxiv.org/abs/2510.06605)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 저작권 보호와 관련된 문제를 다루고 있습니다. 특히 LLM의 고유한 서명을 추출하여 출처 모델과 비교하는 LLM fingerprinting 기법이 제안되었습니다. 기존의 블랙박스 방법들이 효과적이지 못한 이유를 분석한 후, 새로운 방법인 ZeroPrint를 제안하여 성능을 크게 향상시켰습니다.

- **Technical Details**: ZeroPrint는 Fisher Information Theory를 바탕으로, 모델의 입력에 대한 기울기(gradient)가 출력(output)보다 더 많은 정보를 포함하고 있다고 주장합니다. 이 기법은 기존의 블랙박스 방법에서 접근할 수 없는 기울기를 근사화하기 위해 zeroth-order estimation을 사용합니다. 텍스트 도메인에서 이 기법을 적용하기 위해, 의미를 보존하는 단어 교체를 통해 입력의 변형을 생성하여 모델의 Jacobian matrix를 추정합니다.

- **Performance Highlights**: ZeroPrint는 LeaFBench라는 벤치마크에서 기존의 SOTA 블랙박스 fingerprinting 방법들보다 일관되게 우수한 성능을 기록하였습니다. 다양한 지표에서 ZeroPrint의 효과성과 신뢰성이 입증되었으며, LLM 저작권 감사 분야에서 새로운 기준을 세웠습니다.



### The Markovian Thinker (https://arxiv.org/abs/2510.06557)
- **What's New**: 이번 논문에서는 Markovian Thinking이라는 새로운 패러다임을 제안합니다. 이 방법은 정책(policy)이 일정 크기의 상태(state)에 따라 추론을 진행하도록 하여, 사고의 길이를 맥락의 크기와 분리합니다. 이를 통해 Delethink라는 강화학습(RL) 환경을 구성하여, 고정 크기의 청크로 사고를 구조화하면서도, 긴 사고를 더 효율적으로 처리할 수 있음을 보여줍니다.

- **Technical Details**: Delethink 환경은 각 청크의 경계에서 환경을 리셋하고 짧은 캐리오버로 프롬프트를 재초기화합니다. 에이전트는 강화학습을 통해 각 청크의 끝에서 연속성이 보장되는 텍스트 상태를 작성하도록 학습합니다. 이러한 과정은 결국 상수 메모리와 선형 계산(linear compute)을 가능하게 하여 기존의 표준 RL 환경에 비해 획기적인 성과를 나타냅니다.

- **Performance Highlights**: 트레이닝 된 R1-Distill 1.5B 모델은 8K 토큰 청크 내에서 사고를 진행하며 최대 24K 토큰 사고를 수행했습니다. 이는 24K 예산으로 훈련된 LongCoT을 능가하거나 맞먹는 성과를 보여줍니다. 실험 결과는 Delethink가 긴 사고를 수행하면서도 연산 비용과 메모리 사용에서 효율적임을 나타내며, 이로 인해 더 나은 성능을 보이는 것으로 분석되었습니다.



### PuzzlePlex: Benchmarking Foundation Models on Reasoning and Planning with Puzzles (https://arxiv.org/abs/2510.06475)
- **What's New**: 이번 연구는 foundation 모델의 추론과 계획 능력을 평가하기 위한 새로운 벤치마크인 PuzzlePlex를 소개합니다. PuzzlePlex는 각기 다른 난이도의 15종류의 퍼즐로 구성되어 있으며, 단일 플레이어 및 2인 플레이어 환경을 포함합니다. 이는 복잡하고 동적인 환경에서 모델이 문제 해결 능력을 얼마나 끌어올릴 수 있는지를 탐구하기 위한 목적을 가지고 있습니다.

- **Technical Details**: PuzzlePlex는 텍스트 및 이미지 형식의 퍼즐을 지원하며, 결정론적(deterministic) 및 확률적(stochastic) 환경을 포함하여 다양한 유형의 퍼즐을 제공합니다. 각 퍼즐은 여러 난이도 수준을 지원하며, 모델의 발전에 따라 평가를 조정할 수 있는 확장성을 제공합니다. 연구는 instruction-based와 code-based 두 가지 범주로 나누어 모델을 비교합니다.

- **Performance Highlights**: 실험 결과, 모델들이 instruction-based 환경에서 더 우수한 성능을 보였으며, 이는 테스트 시간의 스케일링(test-time scaling)과 연장된 심사를 활용한 결과입니다. 반면, code-based 평가에서는 프로그램 생성(program synthesis)에서의 어려움으로 인해 성능이 저하되었으며, 샘플링 기반 방법이 성능 향상에 기여했습니다. PuzzlePlex의 도입은 future improvements에 대한 방향성을 제시하며, 추론, 계획, 일반화의 영역에서 테스트를 가능하게 합니다.



### Asking For It: Question-Answering for Predicting Rule Infractions in Online Content Moderation (https://arxiv.org/abs/2510.06350)
Comments:
          Accepted at ICWSM 2026

- **What's New**: 이 논문에서는 온라인 커뮤니티의 규칙과 이행 간의 관계를 모델링하고, 새로운 질문-응답 프레임워크인 ModQ를 도입합니다. ModQ는 기존의 분류(classification) 또는 생성(generation) 기반 접근법과는 달리, 커뮤니티의 모든 규칙을 고려하여 특정 댓글에 가장 잘 적용되는 규칙을 식별합니다. 이는 커뮤니티별 규칙의 변동성과 이행의 일관성 문제를 해결하는 데 기여할 수 있습니다.

- **Technical Details**: ModQ는 두 가지 모델 변형을 구현하여 커뮤니티 규칙 이행을 정보 추출 정보(extraction) 작업으로 모델링합니다. 첫 번째 모델인 ModQ-Extract는 사용자 댓글과 커뮤니티 규칙을 컨텍스트로 사용하여 특정 규칙을 추출합니다. 두 번째 모델인 ModQ-Select는 다중 선택(multiple-choice) 질문-응답 방식으로 댓글과 각 규칙 간의 정합성을 점수화하여 가장 적합한 규칙을 선택합니다.

- **Performance Highlights**: ModQ를 사용한 두 모델 모두 Reddit과 Lemmy 데이터셋에서 최신 기법을 능가하여 규칙 위반을 식별하는 데 강력한 성능을 보였습니다. 특히 ModQ-Select는 모든 기준 및 moderation 작업에서 모든 베이스라인을 일관되게 초과하며, 두 모델 모두 미리 경험하지 못한 새로운 커뮤니티와 규칙에 대해 효과적인 일반화 능력을 보여줍니다. 이는 빠르게 변화하는 플랫폼에서의 운영에 있어 큰 장점이 됩니다.



### AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning (https://arxiv.org/abs/2510.06261)
Comments:
          Ongoing project

- **What's New**: AlphaApollo는 자체 진화하는 에이전틱 추론 시스템으로, 기존의 재단 모델(FM) 추론에서 발생하는 두 가지 주요 제약인 모델 고유의 능력과 신뢰할 수 없는 테스트 시간 반복 문제를 해결하고자 합니다. 이 시스템은 계산 도구(Python)와 검색 도구(작업 관련 외부 정보)를 결합하여 정확한 계산을 수행하고 결정을 기반으로 합니다. 이러한 통합으로 AlphaApollo는 복잡한 문제를 해결하는 것뿐만 아니라, 다양한 모델과 도구를 협동으로 조율합니다.

- **Technical Details**: AlphaApollo의 설계 원리는 다양한 모델과 전문 도구를 조화롭게 결합하여 자가 진화형 시스템을 구현하는 것입니다. 이를 통해 직관적이고 정의된 추론을 가능하게 하며, 수학적 문제 해결 시 Python 코드의 실행과 검증에 기반한 피드백을 제공합니다. 이 시스템은 모델 간의 상호작용과도 결합하여 도구-확장된 추론을 진행하며, 이를 통해 근본적인 한계를 넘어서는 데 기여합니다.

- **Performance Highlights**: AlphaApollo는 다양한 모델에서 일정한 성능 개선을 보여줬으며, Qwen2.5 모델에서는 평균 5.15% 증가, Llama-3.3-70B-Instruct 모델에서는 8.91%의 평균 성능 향상이 있었습니다. 또한, 도구 사용 분석 결과 80% 이상의 도구 호출이 정확하게 수행되어 비도구 기반 응답을 일관되게 초과하는 성과를 보였습니다. 현재 AlphaApollo는 지속적인 개발 중이며, 향후 추가 기능 및 실험 결과가 오픈 소스로 공개될 예정입니다.



### CML-Bench: A Framework for Evaluating and Enhancing LLM-Powered Movie Scripts Generation (https://arxiv.org/abs/2510.06231)
Comments:
          24 pages, 9 figures

- **What's New**: 이 연구에서는 CML-Dataset이라는 새로운 데이터셋을 구축하여 인상적인 구조의 영화 스크립트 분석을 통해 LLM의 한계를 탐구합니다. 또한, 대화의 일관성(Dialogue Coherence), 캐릭터의 일관성(Character Consistency), 그리고 플롯의 타당성(Plot Reasonableness)이라는 세 가지 주요 차원을 제시하며, 이들 기준을 바탕으로 CML-Bench라는 평가 프레임워크를 개발했습니다. 이는 LLM이 생성한 스크립트의 질을 정량적으로 평가하고, 인간이 작성한 스크립트와 LLM 간의 질적 차이를 분석하는데 중점을 둡니다.

- **Technical Details**: CML-Dataset은 약 1,800편의 필터링된 영화 스크립트에서 파생된 100개의 영화 스크립트와 그 요약으로 구성됩니다. CML-Bench는 대화 일관성, 캐릭터 일관성 및 플롯 타당성을 측정할 수 있는 8개의 해석 가능한 정량적 메트릭으로 구성되어 있으며, 이 메트릭들은 언어 모델과 결합된 정형 파싱 및 벡터 유사성 계산을 통해 구현됩니다. 이를 통해 LLM이 생성한 스크립트에 대한 세밀하고 객관적인 평가가 가능합니다.

- **Performance Highlights**: CML-Bench를 통해 분석한 결과, 현재의 모든 LLM은 인간이 작성한 스크립트에 비해 일관성에서 일관되게 성능이 저조한 것으로 나타났습니다. 특히 대화 일관성, 캐릭터 일관성, 플롯 타당성에서 큰 차이를 보였습니다. CML-Instruction을 통한 추가적인 실험에서도 LLM이 생성한 스크립트의 질이 유의미하게 향상되었으며, 이는 인간의 선호와 일치하는 결과를 동시에 보여주었습니다.



New uploads on arXiv(cs.IR)

### Ethical AI prompt recommendations in large language models using collaborative filtering (https://arxiv.org/abs/2510.06924)
Comments:
          This paper has been accepted to by the International Journal of Parallel, Emergent & Distributed Systems (Taylor and Francis) and has an assigned DOI. We have already chose to make this open access using CC BY. The article is not yet available online on the publisher's website. The DOI is: this http URL

- **What's New**: 대형 언어 모델(LLMs)의 발전에 맞추어 윤리적인 프롬프트 추천의 중요성이 강조되고 있습니다. 이 논문은 전통적인 감독 방식이 확장성에 어려움을 겪는 상황에서, 추천 시스템의 기술인 협업 필터링(collaborative filtering)을 사용하여 윤리적인 프롬프트 선택을 향상시키는 방법을 제안합니다.

- **Technical Details**: 논문에서는 사용자 상호작용(user interactions)을 활용하여 윤리적인 가이드라인을 촉진하고 편향(bias)을 줄이는 방법을 제시합니다. 또한 부정확한 프롬프트 공학(unethical prompt engineering)을 방지하는 도전 과제도 다루며, 윤리적 AI(ethical AI)의 문제들을 해결하기 위한 접근 방안을 모색합니다.

- **Performance Highlights**: 제안된 방법은 프롬프트 추천을 위한 합성 데이터셋(synthetic dataset)을 포함하고 있으며, 협업 필터링의 적용으로 윤리적 기준을 강화하는 데 중요한 기여를 하고 있습니다. 이 연구는 AI 개발의 공정성과 책임 문제 해결에도 기여할 것으로 기대됩니다.



### M3Retrieve: Benchmarking Multimodal Retrieval for Medicin (https://arxiv.org/abs/2510.06888)
Comments:
          EMNLP Mains 2025

- **What's New**: 이번 논문은 의료 분야의 멀티모달 검색 모델의 필요성과 이를 평가하기 위한 표준 벤치마크가 부족한 상황을 다루고 있습니다. 이를 개선하기 위해 M3Retrieve라는 멀티모달 의료 검색 벤치마크를 제안하며, 이 벤치마크는 의료 분야의 다양한 전문성을 고려하고 있습니다. M3Retrieve는 5개의 도메인과 16개의 의료 분야에 걸쳐 있으며, 120만 개 이상의 텍스트 문서와 16만 4천 개의 멀티모달 쿼리를 포함하고 있습니다.

- **Technical Details**: M3Retrieve는 텍스트와 이미지를 아우르는 멀티모달 데이터를 통합하여 의료 분야의 정보 검색을 위한 보다 현실적인 평가를 가능하게 해줍니다. 이 벤치마크는 22개의 수작업 검토 데이터 세트를 수집하여 16개 의료 전문 분야 모두를 포괄하며, 실제 임상 시나리오를 반영하는 다양한 작업을 포함하고 있습니다. 또한, 환자 정보와 관련된 이미지와 텍스트의 복합적인 해석을 분석하기 위해 5가지 검색 작업을 정의하였습니다.

- **Performance Highlights**: M3Retrieve의 출시로 인해 의료 응용 프로그램에서 신뢰할 수 있는 멀티모달 검색 시스템을 구축하는 데 기여할 것이며, 연구의 발전을 가속화하고 모델 혁신을 촉진할 것입니다. 여러 첨단 멀티모달 검색 모델을 평가하여 특정 의료 전문 분야의 도전 과제를 이해하고 검색 성능에 미치는 영향을 정량화합니다. 이는 특히 안전이 중요한 의료 분야에서 시스템의 신뢰성 향상에 큰 도움이 될 것입니다.



### Crossing Domains without Labels: Distant Supervision for Term Extraction (https://arxiv.org/abs/2510.06838)
Comments:
          Accepted at EMNLP Industry Track 2025

- **What's New**: 자동 용어 추출(ATE)은 자연어 처리(NLP) 작업에서 문서 태깅, 온톨로지 구성 및 특허 분석과 같은 주요 구성 요소입니다. 기존의 최첨단 방법들은 비싼 인간 주석이 필요하며, 도메인 전이에서 어려움을 겪고 있어 실용적으로 적용하기 어렵습니다. 이에 우리는 일곱 개의 다양한 도메인에 걸친 포괄적인 벤치마크를 도입하고, 이를 통해 성능 평가를 문서 및 코퍼스 수준에서 수행하여 보다 견고하고 확장 가능한 솔루션 필요성을 강조합니다.

- **Technical Details**: 우리의 접근 방식인 DiSTER(Distant Supervision for Term Extraction with Robustness)는 LLM을 활용하여 인간 주석 없이도 데이터의 생산성을 높이고, 도메인 간의 확장성을 가능하게 합니다. LLM을 사용하여 생성된 유사 레이블을 활용하여 비즈니스 및 과학 도메인에서 데이터의 일반성을 보장합니다. 또한 경량 포스트-혹(정후) 휴리스틱을 도입하여 문서 수준의 일관성을 높이고 F1 점수를 평균 10% 향상시키는 데 기여합니다.

- **Performance Highlights**: 본 연구에서 우리는 다섯 개의 도메인에서 기존 방법보다 뛰어난 성능을 보였으며, 평균적으로 10% 포인트 향상을 달성했습니다. 우리의 모델은 LLM 기반의 백박스 모델과 비교하여 우수한 성능을 기록하며, GPT-4o와의 경쟁에서도 상대적으로 견고한 결과를 보여줍니다. 우리는 이 데이터셋과 파인튜닝된 모델을 공개하여 미래의 연구를 지원할 예정입니다.



### Reproducing and Extending Causal Insights Into Term Frequency Computation in Neural Rankers (https://arxiv.org/abs/2510.06728)
Comments:
          10 pages, 6 figures, submitted to SIGIR-AP

- **What's New**: 이번 논문 'Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models'에서는 신경 검색 모델의 관련성 계산 방식에 대한 인사이트를 제공하는 새로운 원칙적 원인 개입 방법인 activation patching을 소개합니다. 이 연구는 기존의 프로빙(probing) 방법이 제시하는 상관 관계에 비해 인과 관계를 조사하는 데 중점을 두고 있습니다. 또한, 선행 연구의 주장을 검증하고 추가적인 용어 빈도 축(axiom) TFC2를 탐색하여 특정 attention heads에서 이 기능을 지역화하는 방법을 제안합니다.

- **Technical Details**: 신경 순위 모델은 문서의 관련성을 계산하는데 필수적인 정보인 용어 빈도를 지역화할 수 있음이 확인되었습니다. 특히, Chen et al. (2024) 연구자들은 TFC1 axiom을 기반으로 하여 용어 빈도 정보를 캡처하는 TAS-B 모델을 분석하였습니다. 이 논문에서는 X_baseline과 X_perturbed 쌍을 사용하여 진단 데이터셋을 생성하고, 활성화 패칭(activation patching) 방법을 통하여 이러한 정보를 평가하는 과정을 설명합니다.

- **Performance Highlights**: 이 연구는 원본 코드 기반의 수정 작업을 통해 용어 빈도 정보의 정확한 지역화를 달성하였으며, TFC2에 대한 실험 설정을 활성화 패칭 프레임워크 내에서 제안하였습니다. TAS-B 모델이 TFC2 axiom에 따라 용어 빈도를 추적하는 잠재적 메커니즘을 인코딩한다는 사실을 발견하였습니다. 이러한 결과는 검색 모델의 의사 결정 과정에 대한 더 깊은 이해를 제공하는 데 기여합니다.



### Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks (https://arxiv.org/abs/2510.06658)
Comments:
          Accepted at SIGIR-AP 2025

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 평가가 기존의 직원 주석에 대한 정확성 중심에서 벗어나, 인간과 LLM이 작성하는 레이블 간의 통계적 비교를 포함해야 한다고 강조합니다. 연구는 LLM이 인간 주석자와 통계적으로 구분할 수 없음을 보여준 사례를 포함하여, LLM이 주관적 판단을 모방할 수 있는 가능성을 탐구합니다. Krippendorff의 α, paired bootstrapping 및 TOST 테스트 절차 기반의 새로운 평가 방법론을 제시합니다.

- **Technical Details**: 우리는 Krippendorff의 α와 같은 통계적 방법론을 통해 LLM의 주석 능력을 평가하고, 다수의 주석자 사이에서 LLM의 존재가 인간 주석자의 존재와 얼마나 유사한지를 분석합니다. 연구에서 적용한 방법론은 작고 구체적인 샘플에 대해 LLM의 대규모 주석 적합성을 조기에 평가할 수 있도록 돕습니다. 연구에서는 MovieLens 100K와 PolitiFact 두 데이터셋을 사용하여 LLM의 성과를 테스트했습니다.

- **Performance Highlights**: MovieLens 100K에서는 LLM과 인간 주석자의 출력이 통계적으로 구분할 수 없었고 그 결과는 p=0.004로 나타났습니다. 반면 PolitiFact 데이터셋에서는 구분이 가능했던 결과(p=0.155)가 도출되어 작업에 따라 성능 차이가 있음을 보여줍니다. 이러한 결과는 LLM이 특정 주석 작업에서 인간의 주관적 판단과 비슷한 성과를 나타낼 수 있음을 시사합니다.



### LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations (https://arxiv.org/abs/2510.06657)
Comments:
          RecSys 2025 Industry Track

- **What's New**: 이번 논문은 대규모 산업 단기 비디오 추천 시스템에서 미세한 콘텐츠 이해를 달성하기 위한 고급 "주석(annotation)" 메커니즘으로 대형 언어 모델(LLMs)을 배포한 사례 연구를 제시합니다. 전통적인 기계 학습 분류기는 개발 주기가 길고, 미세한 이해 부족이라는 문제를 겪고 있습니다. 본 연구에서는 LLM을 주석 작성기로 활용함으로써 개발 시간을 단축하고 미세한 속성을 주석화할 수 있는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 방법론은 LLM 주석 작성을 위한 엔드투엔드(end-to-end) 워크플로우로, 이를 통해 대규모 단기 비디오 추천 시스템 내 방대한 콘텐츠에 대한 깊고 미세한 이해를 도모합니다. 이 과정은 세 가지 핵심 단계로 나뉘며, 여기에는 목표 속성 정의 및 평가 체계 구축, LLM을 이용한 오프라인 대량 주석화, 온라인 추천 시스템에 리치 주석 통합이 포함됩니다. 각 단계는 미세한 멀티모달(multimodal) 특성을 포착하고, 변화하는 콘텐츠 트렌드에 적응하며, 효율적이면서도 확장 가능한 생산 통합을 이루는 데 있어 고유한 도전 과제를 제기합니다.

- **Performance Highlights**: 실험 결과, LLM이 미세한 속성에 대한 오프라인 주석 품질에서 인간 평가자를 초월하며, 사용자 참여 및 만족도를 향상시키는 데 있어 매우 효과적임을 입증했습니다. LLM 주석의 품질은 내부 평가자 간의 정렬에서 비롯된 명확성에 크게 의존하며, 이를 통해 LLM 변별력이 향상되고 평가 일관성이 개선되었습니다. 이러한 경험적 분석은 추천 시스템을 위한 LLM 파이프라인 설계 및 확장의 통찰력을 제공하며, 현대 추천 시스템의 효과성을 향상시킬 수 있는 미세한 이해의 이점을 강조합니다.



### Towards Reliable Retrieval in RAG Systems for Large Legal Datasets (https://arxiv.org/abs/2510.06999)
Comments:
          Accepted for the 7th Natural Legal Language Processing Workshop (NLLP 2025), co-located with EMNLP 2025

- **What's New**: 이 논문에서는 대규모 법률 문서에 대한 Retrieval-Augmented Generation (RAG) 시스템의 신뢰성을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히, Document-Level Retrieval Mismatch (DRM)라는 중요 실패 모드를 정의하고 그 문제를 해결하기 위한 Summary-Augmented Chunking (SAC) 기법을 소개합니다. SAC는 각 텍스트 청크에 문서 수준의 합성 요약을 추가하여 전반적인 글로벌 컨텍스트를 보존하는 방법론입니다.

- **Technical Details**: RAG 시스템은 두 개의 주 단계로 구성됩니다: 정보 검색기(retriever)와 생성기(generator) 모델입니다. 검색기는 사용자의 질의에 대한 관련 텍스트 조각을 찾고, 생성기는 이 조각들을 기반으로 최종 답변을 합성하여 출력합니다. 본 연구에서는 SAC 기법을 활용하여 신뢰할 수 있는 텍스트 코퍼스의 청크를 문서 수준의 요약으로 보강함으로써 이전 알고리즘의 한계를 극복하고자 합니다.

- **Performance Highlights**: 실험 결과, SAC 방법이 법률 질문-답변 작업에서 DRM을 상당히 줄였으며, 텍스트 수준의 검색 정확도와 재현율 또한 개선되었음을 확인하였습니다. 흥미롭게도, 법률 전문 도메인 지식을 포함한 방법보다 일반적인 요약 전략이 더 나은 성능을 보였다는 점이 주목됩니다. 이번 연구는 대규모 법률 문서 데이터셋에 적용했을 때 RAG 시스템의 신뢰성을 높일 수 있는 실용적이고 통합 가능한 기술의 필요성을 강조합니다.



### Spiral Model Technique For Data Science & Machine Learning Lifecyc (https://arxiv.org/abs/2510.06987)
- **What's New**: 이 논문은 현대 비즈니스에서 데이터 분석(Analytics)의 중요성을 강조합니다. 특히 데이터 과학 생명주기(Data Science Lifecycles)를 기업의 문화에 맞게 조정하여 생산성을 높이고 경쟁력을 향상시키는 방법을 제안합니다. 새로운 기술인 스파이럴 기법(Spiral Technique)을 도입하여 명확한 목표가 있는 비즈니스 문제에 데이터 과학 생명주기를 통합하는 방안을 다룹니다.

- **Technical Details**: 스파이럴 기법은 비즈니스 프로세스에 대한 유연성(versatility), 민첩성(agility) 및 반복적 접근(iterative approach)을 강조합니다. 전통적인 데이터 과학 생명주기는 선형(linear) 또는 순환(cyclical) 모델로, 주기가 끝난 후 다시 시작할 수 있는 구조로 나타납니다. 이를 통해 데이터 의존 프로젝트에서의 시작과 종료에 관한 기여 요소를 살펴봅니다.

- **Performance Highlights**: 이 새로운 접근 방식은 데이터 과학 프로젝트가 명확한 목표를 갖고 있을 때 더욱 효과적이라는 것을 보여줍니다. 비즈니스 문제 해결을 위한 신속한 반복과 조정이 가능해지므로 기업들이 환경 변화에 더 빨리 적응할 수 있도록 합니다. 이로 인해 기업의 전반적인 경쟁력이 향상되는 결과를 가져올 수 있습니다.



### Exposing Citation Vulnerabilities in Generative Engines (https://arxiv.org/abs/2510.06823)
Comments:
          12 pages, under-reviewing at a conference

- **What's New**: 본 논문은 Generative Engines (GEs)가 타당한 인용 출처를 제공하는지 평가하는 새로운 기준을 제안합니다. GEs는 웹 검색과 답변 생성 기능을 통합하여 사용자 질문에 답변하므로, 공격자가 악의적 콘텐츠를 주입할 위험이 큽니다. 제출된 자료는 웹 콘텐츠를 중독 공격에서 방어하기 위한 인용 출처에 대한 새로운 기준을 제시하며, 이는 사실성이 결여된 대답을 방지하기 위한 것이다.

- **Technical Details**: GEs의 모델은 사용자 질문과 개인화 정보를 입력으로 받아 텍스트 답변을 생성하는 함수로 formalization됩니다. 이 시스템은 두 가지 주요 구성 요소인 콘텐츠 검색(content retrieval)과 답변 생성(answer generation)으로 구성되어 있습니다. 콘텐츠 검색 단계에서 웹 검색 결과를 활용해 여러 개의 쿼리를 수행하고 검색된 웹 소스에서 정보를 수집하여 답변을 생성하는 방식입니다.

- **Performance Highlights**: 연구 결과는 일본의 정치적 대답에서 60%에서 65%가 공식 당 홈페이지에서 인용된 반면, 미국에서는 25%에서 45%만 해당된다는 것을 보여주었습니다. 저장된 정보 퀄리티에 따라 웹 콘텐츠 주입 장벽(content-injection barrier)에 따른 출처의 평가도 진행되었으며, 낮은 장벽의 출처들이 비공식 및 덜 신뢰할 수 있는 정보를 반영하는 경향이 있다는 것을 확인했습니다.



### Overview of the Plagiarism Detection Task at PAN 2025 (https://arxiv.org/abs/2510.06805)
Comments:
          Working Notes at PAN at CLEF 2025

- **What's New**: PAN 2025에서는 과학 기사에서 자동 생성된 텍스트 표절을 식별하는 새로운 데이터셋을 구성했습니다. 이 데이터셋은 Llama, DeepSeek-R1, Mistral이라는 세 가지 대형 언어 모델을 활용하여 자동으로 생성된 표절 사례를 포함하고 있습니다. 본 연구는 이 새로운 데이터셋의 구조와 참여자들의 성과를 정리하며, 2015년도 표절 탐지 과제를 재평가합니다.

- **Technical Details**: PAN 2025는 2015년 대회와 유사한 평가 방법론과 데이터셋 형식을 유지합니다. 참가자는 S(출처 문서)와 P(표절 문서) 쌍을 이용해 LLM으로 생성된 표절 구문을 식별하고 정렬하는 작업을 수행합니다. 기본적으로 arXiv에서 가져온 100,000개의 문서로부터 생성된 데이터셋은 세 가지 세부 범주로 나누어져 있어 다양한 성능 분석이 가능하도록 구성되었습니다.

- **Performance Highlights**: 신규 데이터셋을 활용한 2025 PAN 과제에서는 기존 기준선 모든 결과를 초과하는 네 가지 제출물이 있었습니다. Linq-Embed-Mistral 모델을 사용한 기준선이 가장 뛰어난 성과를 보였는데, 이는 표절 탐지와 같은 텍스트 검색 작업에 적합한 특수 모델이 효과적임을 나타냅니다. 현재까지의 결과는 LLM을 통한 자동 표절 처리가 일반성에 한계가 있음을 시사합니다.



### Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization (https://arxiv.org/abs/2510.06732)
Comments:
          10 pages, 3 figures

- **What's New**: 이번 논문은 정보 검색에서 rerankers(재정렬기)로 활용되는 대형 언어 모델(LLMs)의 취약점을 드러내고 이를 조작할 수 있는 새로운 방법인 Rank Anything First (RAF)를 소개합니다. RAF는 타겟 아이템의 순위를 높이기 위해 간결한 텍스트 변형을 생성하는 두 단계의 토큰 최적화 방식으로 구성되어 있습니다. 첫 번째 단계에서는 그리디 좌표 경량화(Greedy Coordinate Gradient)를 사용하여 현재 위치에서의 후보 토큰을 선별하고, 두 번째 단계에서는 후보들을 평가하여 자연스러운 언어를 유지하면서 효과적으로 기존 방법들보다 높은 순위 조작성을 달성합니다.

- **Technical Details**: RAF는 토큰 단위 최적화를 통해 순위 조작 프롬프트를 생성하며, 두 가지 목표인 순위 효과성(maximizing ranking effectiveness)과 언어 자연스러움(preserving linguistic naturalness)을 동시에 고려합니다. 특정 수식과 조건부 확률을 통해 LLM의 재정렬 과정을 설명하며, 공격자가 특정 아이템의 설명에 자연스러운 텍스트 시퀀스를 삽입하여 아이템의 순위를 향상시킬 수 있도록 합니다. 이러한 과정은 제품의 브랜드, 가격, 간단한 설명을 포함한 제품 세트를 기반으로 하여 수행됩니다.

- **Performance Highlights**: RAF는 다양한 LLM 모델에서 실험을 통해 자연스러운 언어를 사용하여 목표 아이템의 순위를 유의미하게 증가시키는 데 성공하였습니다. 이는 기존의 방법들보다 더욱 강력하고 안정적인 순위 조작을 가능하게 하며, 최적화된 프롬프트는 여러 모델 간에 성공적으로 전이됩니다. 이 연구는 LLM 기반 reranking의 보안 위험성을 강조하며, 현대 정보 검색 시스템의 신뢰성과 강인성에 대해 새로운 도전 과제를 제시합니다.



New uploads on arXiv(cs.CV)

### Temporal Prompting Matters: Rethinking Referring Video Object Segmentation (https://arxiv.org/abs/2510.07319)
- **What's New**: 이번 논문은 비디오에서 쿼리 문장으로 언급된 객체를 분할하는 Referring Video Object Segmentation (RVOS) 문제를 새롭게 접근합니다. 기존 방법들은 밀집 마스크 주석을 요구하며, 이는 계산 비용이 높고 확장성이 떨어지는 문제를 지니고 있습니다. 저자들은 RVOS 작업을 참조(referring), 비디오(video), 그리고 분할(segmentation) 요인으로 나누고, 이를 해결하기 위한 Temporal Prompt Generation and Selection (Tenet) 프레임워크를 제안합니다.

- **Technical Details**: Tenet 프레임워크는 이미지 기반 기초 세분화 모델을 효율적으로 참조 비디오 객체 분할에 적합하도록 전환합니다. 이 과정에서 참조 문장과 연관된 시각적 프롬프트를 생성하기 위해 오프 더 셸프(object detectors) 객체 탐지기와 추적기(trackers)를 활용합니다. 이러한 방법으로 신뢰성 있는 참조 제안(reference proposal)과 후보 트랙을 생성하고, 이를 기반으로 Prompt Preference Learning을 통해 후보 트랙의 질을 평가하여 가장 적절한 프롬프트를 선택합니다.

- **Performance Highlights**: 다양한 RVOS 벤치마크에서의 실험 결과는 Tenet 프레임워크의 효과성을 입증합니다. 본 연구는 초과 일반화 문제를 해결하기 위한 방법을 제시하며, 특히 Temporal Prompt Generation과 Selection 기술을 통해 고품질 마스크를 생성함으로써, 영상의 시각적 도전들을 효과적으로 처리할 수 있게 합니다. 이를 통해 RVOS 분야의 모델 적응을 더욱 효율적으로 만들어냅니다.



### Quantum-enhanced Computer Vision: Going Beyond Classical Algorithms (https://arxiv.org/abs/2510.07317)
Comments:
          44 pages, 23 figures and 6 tables

- **What's New**: Quantum-enhanced Computer Vision (QeCV)은 컴퓨터 비전, 최적화 이론, 머신 러닝 및 양자 컴퓨팅이 결합된 새로운 연구 분야입니다. 이 연구는 양자 컴퓨팅의 양자 역학적 효과를 활용하여 기존 비양자(클래식) 컴퓨터로는 처리가 힘든 시각 신호의 처리와 해석 방식을 혁신할 가능성이 큽니다. 특히 비양자 방법으로는 합리적인 시간 내에 해결책을 찾지 못하는 경우에 양자 컴퓨터의 사용이 유리할 것으로 기대됩니다.

- **Technical Details**: QeCV의 구현을 위해서는 양자 하드웨어와 호환 가능한 특화된 알고리즘과 새로운 접근법이 개발되어야 합니다. 본 논문에서는 양자 컴퓨터의 작업 원리와 QeCV와 관련한 방법론에 대한 포괄적인 소개를 제공합니다. 특히 두 가지 주요 양자 계산 패러다임인 게이트 기반 양자 컴퓨팅과 양자 어닐링(quantum annealing)을 활용한 양자 하드웨어와 호환되는 방법의 구성을 다룹니다.

- **Performance Highlights**: 우리는 QeCV를 위한 양자 컴퓨팅 도구와 학습 자료를 종합적으로 검토하고, QeCV 논문 발표 및 검토와 관련된 측면, 현재의 도전 과제, 그리고 그 사회적 의미에 대해 논의합니다. 이 연구는 컴퓨터 비전 커뮤니티의 학생 및 연구자들에게 QeCV에 대한 귀중한 참고 자료로 활용될 것입니다.



### Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers (https://arxiv.org/abs/2510.07316)
Comments:
          NeurIPS 2025. Project page: this https URL

- **What's New**: 이 논문은 Pixel-Perfect Depth라는 새로운 단안 깊이 추정 모델을 제안합니다. 이 모델은 픽셀 공간에서의 확산 생성(diffusion generation)을 기반으로 하여, 추정된 깊이 맵으로부터 고품질의 비행 픽셀(flying pixels)이 없는 포인트 클라우드를 생성합니다. 기존의 모델들은 VAE(Variational Autoencoder)를 사용하여 깊이 맵을 잠재 공간으로 압축하는 과정에서 비행 픽셀이 발생하는 문제를 가지고 있었습니다. 이에 비해 본 모델은 픽셀 공간에서 직접적으로 확산 생성을 수행하여 이러한 문제를 해결하고 있습니다.

- **Technical Details**: Pixel-Perfect Depth는 두 가지 혁신적인 설계를 도입합니다. 첫 번째는 SP-DiT(Semantics-Prompted Diffusion Transformers)로, 이는 비전 기초 모델의 의미적 표현을 포함하여 확산 과정에서 도움을 주어 글로벌 의미 일관성을 보존하면서 세부 시각적 디테일을 향상시킵니다. 두 번째는 Cascade DiT Design(Cas-DiT)으로, 이는 토큰 수를 점진적으로 증가시켜 효율성과 정확성을 높입니다. 이러한 접근 방식을 통해 모델은 픽셀 공간에서의 생성 복잡성을 극복하며, 최종적으로 높은 성능을 발휘합니다.

- **Performance Highlights**: 이 모델은 다섯 개의 벤치마크에서 발표된 모든 생성 모델 중 최고의 성능을 기록했습니다. 특히 엣지 인지 포인트 클라우드 평가에서 기존의 모든 모델을 크게 능가하는 결과를 나타냈습니다. 각종 성능 지표에서 NYUv2의 AbsRel 메트릭 기준으로 최대 78%의 성능 향상도 달성했습니다. 이러한 성과는 새로운 평가 메트릭을 도입한 덕분으로, 엣지에서의 비행 픽셀을 효과적으로 평가할 수 있게 되었습니다.



### WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation (https://arxiv.org/abs/2510.07313)
- **What's New**: WristWorld는 앵커 뷰에서만 손목 뷰 비디오를 생성할 수 있는 최초의 4D 세계 모델입니다. 기존의 세계 모델은 손목 조망 첫 번째 프레임을 요구하지만 WristWorld는 이 제한을 극복하여 효율적으로 손목 뷰를 생성합니다. 이 모델은 두 단계로 구성되어 있으며, 시공간 일관성을 보장합니다.

- **Technical Details**: WristWorld는 두 단계로 운영됩니다. 첫 번째 단계인 재구성(Reconstruction)에서는 VGGT를 확장하여 손목 뷰 카메라 포즈와 4D 점 구름을 추정하는 공간 투영 일관성(Spatial Projection Consistency, SPC) 손실을 도입하였습니다. 두 번째 단계인 생성(Generation)에서는 복구된 지오메트리와 CLIP 인코딩된 앵커 뷰 특징을 기반으로 시간적으로 일관된 손목 뷰 비디오를 합성합니다.

- **Performance Highlights**: WristWorld는 Droid, Calvin, Franka Panda 환경에서 실험을 통해 최첨단 손목 뷰 비디오 생성을 입증하였습니다. Calvin에서는 평균 작업 완료 길이를 3.81% 늘려 앵커-손목 성능 격차의 42.4%를 해소할 수 있었습니다. 이 모델은 기존의 단일 뷰 세계 모델을 멀티 뷰 기능으로 확장하는 플러그 앤 플레이 방법으로도 활용할 수 있습니다.



### MATRIX: Mask Track Alignment for Interaction-aware Video Generation (https://arxiv.org/abs/2510.07310)
Comments:
          Project Page is available at: this https URL

- **What's New**: 본 논문에서는 MATRIX-11K라는 비디오 데이터셋을 생성하여 다중 인스턴스 및 주체-객체 상호작용을 이해하고 개선하기 위한 초기 연구를 진행했습니다. 데이터셋은 상호작용 의식(caption) 캡션과 마스크 트랙(multi-instance mask tracks)을 포함하고 있습니다. 또한 이 데이터셋을 통해 동영상 Diffusion Transformers(DiTs)에서의 의미적 기초(semantic grounding) 및 의미적 전파(semantic propagation)에 대한 체계적인 분석을 실시합니다.

- **Technical Details**: 연구에서는 주체-객체의 상호작용이 어떻게 내부적으로 표현되는지를 분석하기 위해 3D 전체 주의(attention) 메커니즘을 이용합니다. 이를 통해 텍스트-비디오 대화(attention) 조정이 올바르게 작용하고 시간적으로 지속되는지를 측정합니다. 특히, MASK 트랙을 사용하여 여러 인스턴스 간의 상호작용을 정량화하고, 이를 정렬하기 위해 MATRIX라는 정규화를 도입합니다.

- **Performance Highlights**: MATRIX는 상호작용 충실도(interaction fidelity)와 의미적 정렬(semantic alignment)을 개선하면서도 드리프트(drift)와 환각(hallucination)을 줄이는 데 성공합니다. 또한 새로운 평가 프로토콜인 InterGenEval을 통해 상호작용 의식 평가를 수행하며, KISA, SGI 및 IF와 같은 중요한 지표를 측정합니다. 실험 결과를 통해 제안한 방법이 기존 기술보다 상당한 성능 향상을 보여줍니다.



### SpecGuard: Spectral Projection-based Advanced Invisible Watermarking (https://arxiv.org/abs/2510.07302)
Comments:
          ICCV 2025 Accepted Paper

- **What's New**: 이번 연구에서는 SpecGuard라는 새로운 수채화 기법을 제안하여 이미지의 저작권 정보를 견고하고 보이지 않게 심는 방법을 소개합니다. 기존 방법이 각종 이미지 변형에 취약했던 반면, SpecGuard는 고주파 영역에서 메시지를 임베딩하고, 빠른 푸리에 변환(Fast Fourier Transform) 근사를 활용하여 성능을 극대화합니다. 이 접근 방식은 저작권 정보의 무결성을 유지하며 다양한 공격에 대한 저항력을 강화합니다.

- **Technical Details**: SpecGuard는 이미지의 고주파 성분에 메시지를 통합하는 두 가지 핵심 모듈, 즉 인코더(Encoder)와 디코더(Decoder)로 구성됩니다. 인코더는 웨이브릿 투영(wavelet projection)과 스펙트럼 투영(spectral projection)을 이용해 메시지를 특정 주파수 대역에 삽입하여 시각적 영향을 최소화합니다. Parseval의 정리를 활용하여 디코더는 암호화된 패턴을 학습하고 추출하는 데 효과적으로 사용됩니다.

- **Performance Highlights**: 제안된 SpecGuard는 기존의 최신 모델보다 우수한 비가시화(de-visibility) 특성과 높은 용량을 가진 워터마크를 포함할 수 있습니다. 여러 실험을 통해 SpecGuard의 향상된 성능이 입증되었으며, 디지털 콘텐츠의 보안 및 진위 확인을 크게 향상시킵니다. 이 논문에서는 실제 활용을 위해 전체 코드를 깃허브(GitHub)에 공개하였습니다.



### Evaluating Fundus-Specific Foundation Models for Diabetic Macular Edema Detection (https://arxiv.org/abs/2510.07277)
Comments:
          Accepted for publication at SIPAIM 2025

- **What's New**: 이 논문에서는 당뇨병성 황반 부종(DME) 탐지를 위한 다양한 기초 모델(Foundation Models, FM)과 전이 학습 방법을 체계적으로 비교합니다. 특히, RETFound와 FLAIR 두 가지 FM 및 EfficientNet-B0 백본을 IDRiD, MESSIDOR-2, OCT-and-Eye-Fundus-Images (OEFI) 등의 여러 데이터셋에서 평가하였습니다. 결과적으로, FM의 성능은 일관되게 전이 학습 모델인 CNN에 미치지 못한다는 사실을 발견하였습니다.

- **Technical Details**: 본 연구에서는 세 가지 접근 방법을 평가했습니다: 표준 미세 조정(standard fine-tuning), FM을 고정 특징 추출기로 사용하는 선형 프로빙(linear probing), 비전-언어 FM을 활용한 제로샷 예측(zero-shot prediction)입니다. EfficientNet-B0, FLAIR의 ResNet-50 인코더, 그리고 Vision Transformer(ViT) 기반의 RETFound를 모델로 사용하였습니다. 각 접근 방법에 대해 DME 탐지를 위한 레이블이 있는 망막 영상 데이터로 미세 조정이 이루어졌으며, 특징 공간에서의 선형 분류기도 훈련하였습니다.

- **Performance Highlights**: 결과적으로, EfficientNet-B0은 대부분의 평가에서 ROC 곡선과 정밀도/재현율 곡선에서 1위 또는 2위를 기록하였고, RETFound는 OEFI에서만 유망한 결과를 보였습니다. FLAIR는 강력한 제로샷 성능을 보여주어 적절한 프롬프트를 사용했을 때 주목할 만한 AUC-PR 점수를 얻었습니다. 이 결과들은 데이터가 부족한 환경에서 경량 CNN이 DME 탐지에 대해 여전히 견고한 기준선 역할을 할 수 있음을 시사합니다.



### TalkCuts: A Large-Scale Dataset for Multi-Shot Human Speech Video Generation (https://arxiv.org/abs/2510.07249)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 다중 장면 인체 음성 비디오 생성 연구를 위한 대규모 데이터셋인 TalkCuts를 소개합니다. 기존의 데이터셋들이 단일 정적 시점에 한정된 반면, TalkCuts는 164,000개의 클립과 500시간 이상의 다양한 카메라 샷을 포함한 고품질 인체 음성 비디오를 제공합니다. 이 데이터셋은 10,000개 이상의 고유한 인물에 대한 자세한 텍스트 설명, 2D 키포인트, 3D SMPL-X 모션 주석을 포함하고 있어 다중 모달 학습과 평가가 가능하게 합니다.

- **Technical Details**: TalkCuts는 다양한 카메라 샷이 포함된 장기 인체 음성 비디오 생성을 위해 특별히 구성된 데이터셋입니다. 이 데이터셋은 높은 해상도와 함께 여러 카메라 샷을 포함하고 있으며, 텍스트, 오디오, 2D 키포인트, 3D SMPL-X 매개변수 등 다양한 모달리티를 제공합니다. 연구진은 Orator라는 LLM(대형 언어 모델) 기반의 다중 모달 생성 프레임워크를 제안하며, 이를 통해 자연스러운 카메라 전환과 동기화된 제스처가 포함된 장기 음성 비디오가 자동으로 생성됩니다.

- **Performance Highlights**: TalkCuts를 기반으로 한 실험에서는 기존의 기초 모델들보다 일관된 샷 및 모션 품질, 정체성 보존 측면에서 우수한 성과를 보여주었습니다. 제안된 Orator 시스템은 DirectorLLM에 의해 구동되며, 텍스트 안내를 통해 생성 과정을 감독합니다. 우리는 실험적으로 TalkCuts가 현실적이고 일관된 다중 샷 음성 비디오 생성을 가능하게 함을 확인하였습니다.



### GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation (https://arxiv.org/abs/2510.07217)
Comments:
          30 pages, 21 figures, accepted to EMNLP 2025 findings

- **What's New**: 이번 연구에서는 텍스트-이미지 합성 분야에서의 새로운 접근법을 제안합니다. 텍스트 기반의 프롬프트 최적화 기술인 GenPilot를 소개하며, 이는 다양한 모델에 적용 가능하고, 모델 훈련 없이도 효율적으로 작동합니다. GenPilot는 오류 분석, 클러스터링 기반의 적응 탐색, 세밀한 검증 등을 통해 보다 정확한 이미지 생성을 목표로 합니다.

- **Technical Details**: GenPilot는 테스트 시간에 프롬프트 최적화를 탐색 문제로 설정하여, 데이터의 해석 가능성을 높이고 동적으로 프롬프트를 정제합니다. 시스템은 주로 두 가지 단계로 구성됩니다: 오류 분석 및 테스트 시간 프롬프트 최적화. 각 단계에서 비주얼 질문 응답(VQA)와 캡셔닝 기법이 사용되며, 메모리 모듈로 최적화 과정을 지원합니다.

- **Performance Highlights**: DPG-bench 및 Geneval에서의 실험 결과, GenPilot는 최대 16.9% 및 5.7%의 성능 향상을 보여줍니다. 이는 텍스트와 이미지 간의 일관성 및 구조적 일관성을 강화시키는 데 기여하며, T2I 작업에 대한 강력한 일반화 능력을 입증합니다. GenPilot는 전반적으로 모델에 구애받지 않고 다채로운 프롬프트를 효과적으로 처리할 수 있습니다.



### EigenScore: OOD Detection using Covariance in Diffusion Models (https://arxiv.org/abs/2510.07206)
- **What's New**: 본 논문에서는 EigenScore라는 새로운 OOD (out-of-distribution) 탐지 방법을 제안합니다. 이 방법은 확산 모델(difussion model)에서 유도된 posterior covariance의 고유값 스펙트럼을 활용하여 분포 변화를 감지합니다. EigenScore는 기존의 접근 방식과 달리 확률적 신호를 통해 OOD 데이터를 구별하는 혁신적인 방법론을 제공합니다.

- **Technical Details**: EigenScore는 denoising 프로세스의 covariance 구조를 활용하여 불확실성 신호를 캡처합니다. 이는 접합 변화(jacobian-free) 고유값 추정 알고리즘을 사용하여 계산 효율성을 보장합니다. 본 연구에서는 denoiser의 posterior covariance가 OOD 샘플에 대해 증가한다는 것을 이론적으로 입증하며, 이러한 특성이 안정적이고 차별적인 신호를 제공하는 데 기여합니다.

- **Performance Highlights**: EigenScore는 CIFAR-10, CIFAR-100, SVHN, CelebA, TinyImageNet과 같은 표준 OOD 벤치마크에서 평균적인 SOTA 성능을 달성합니다. 특히, CIFAR-10과 CIFAR-100 같은 근접 OOD 환경에서도 강력한 성능을 보여줍니다. 이러한 결과는 EigenScore가 기존의 확산 기반 방법들이 실패하는 OOD 탐지에서 특히 강력하다는 것을 시사합니다.



### Resolution scaling governs DINOv3 transfer performance in chest radiograph classification (https://arxiv.org/abs/2510.07191)
- **What's New**: 이 논문은 메타의 DINOv3 모델을 소개하며, 기존의 Self-supervised learning (SSL) 모델을 Gram-anchored self-distillation을 통해 확장했습니다. 이는 흉부 방사선 이미지에서의 변별력 있는 findings(발견)에 대해 SSL의 효과를 평가한 연구로, DINOv3가 DINOv2 및 ImageNet 초기화 모델과 비교한 체계적인 성능 검정을 수행했습니다.

- **Technical Details**: 논문에서는 두 가지 대표적인 backbone(백본) 모델인 ViT-B/16 및 ConvNeXt-B를 사용하여 814,000개 이상의 샘플이 포함된 7개의 데이터셋을 benchmark(벤치마크)했습니다. 이미지 해상도는 224x224, 512x512 및 1024x1024 픽셀로 분석되었으며, 주된 성과 지표는 mean AUROC(평균 면적 아래 곡선)으로 설정되었습니다.

- **Performance Highlights**: DINOv3는 224x224 해상도에서 성인 데이터셋에서 DINOv2와 비슷한 성과를 보였지만, 512x512 해상도에서는 DINOv3가 DINOv2 및 ImageNet보다 일관된 성능 향상을 보였습니다. 이러한 결과는 흉부 방사선 진단에서 높은 입력 해상도가 현대적인 SSL 모델의 이점을 활용하는 데 중요하다는 점을 강조했습니다.



### MV-Performer: Taming Video Diffusion Model for Faithful and Synchronized Multi-view Performer Synthesis (https://arxiv.org/abs/2510.07190)
Comments:
          Accepted by SIGGRAPH Asia 2025 conference track

- **What's New**: 본 논문에서는 인간 중심의 비디오 생성 프레임워크인 MV-Performer를 발표합니다. 이는 단일 모노큘러 비디오를 360도 다중 보기 비디오로 변환하는 혁신적인 접근방식입니다. MVHumanNet 데이터셋을 활용하고, 카메라 종속의 노말 맵과 같은 조건 신호를 통합하여, 보이는 영역과 보이지 않는 영역 간의 모호성을 줄입니다.

- **Technical Details**: MV-Performer는 다중 뷰 비디오 확산 모델을 기반으로 하며, 참조 비디오, 부분 렌더링 및 다양한 관점의 정보를 융합하여 생성된 비디오 간의 동기화를 유지하는 메커니즘을 포함합니다. 또한, 다양한 최첨단 추정 방법들을 통합하여 모노큘러 깊이 추정에서 발생하는 아티팩트를 크게 줄이는 견고한 추론 절차를 제공합니다. 이 모델은 MVHumanNet 데이터의 특성에 맞춰 설계되었습니다.

- **Performance Highlights**: 실험 결과, MV-Performer는 MVHumanNet, DNA-Rendering 및 야외 수집 데이터셋에서 뛰어난 효과성과 견고성을 입증하였습니다. 인체 동작과 외관을 큰 관점 변화에 대해 합성할 수 있는 깊이 기반 왜곡 패러다임을 활용하여, 처음으로 모노큘러 인간 중심 비디오를 밀집 다중 보기 비디오로 변환하는 생성 프레임워크를 개발했습니다.



### Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods (https://arxiv.org/abs/2510.07143)
- **What's New**: 본 논문에서는 다중 모달 대형 언어 모델(MLLM)에서 시각적 토큰 압축을 가속화하는 최근 노력의 한계를 언급하며, 기존 벤치마크를 활용한 평가가 압축 기술에 적합하지 않음을 지적합니다. 실험 결과에 따르면, 간단한 이미지 다운샘플링이 여러 고급 압축 방법보다 일관되게 우수한 성능을 보였으며, 이를 통해 새로운 평가 프레임워크인 VTC-Bench를 제안하게 되었습니다. 이 프레임워크는 기존 벤치마크의 노이즈를 정제하고, 시각적 토큰 압축 기법을 공정하게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 논문에서는 MLLM의 시각적 토큰이 텍스트 토큰보다 일반적으로 많은 문제를 다루고 있으며, 이를 해결하기 위해 다양한 압축 방법이 제안되었습니다. 그러나 기존의 평가 기준들은 이와 관련이 없어서 적절한 평가 기준을 제시하지 못하고 있습니다. VTC-Bench는 다운샘플링을 통한 데이터 필터링 메커니즘을 도입하여 샘플의 난이도를 평가하는 방법으로 활용되고 있습니다.

- **Performance Highlights**: 연구에서 나타난 주요 발견으로는 현재의 벤치마크가 시각적 토큰 압축 작업에 상당한 노이즈를 포함하고 있다는 점과, 다운샘플링 방법이 적절한 샘플 그룹을 통해 유의미한 정확성을 발휘한다는 것입니다.  다양한 압축 기법과의 비교에서 다운샘플링이 지속적으로 보다 나은 성능을 보이며, 이는 해당 분야의 연구와 평가에 있어 중요한 기여를 할 수 있을 것입니다.



### Few-Shot Adaptation Benchmark for Remote Sensing Vision-Language Models (https://arxiv.org/abs/2510.07135)
- **What's New**: 이번 연구는 Remote Sensing Vision-Language Models(RSVLMs)의 few-shot adaptation 방법을 평가하기 위해 최초의 구조화된 벤치마크를 도입합니다. 10개의 다양한 원격 탐지 장면 분류 데이터셋에서 실험을 수행하여, 여러 최신 기법을 통해 다양한 RSVLM의 적응 성능을 비교 분석하였습니다. 연구는 zero-shot 성능과 few-shot 적응 간의 차이를 강조하며, 특수한 원격 탐사 작업에 적합한 방법들이 필요함을 보여주고 있습니다.

- **Technical Details**: 연구에서는 AID, EuroSAT, 기계학습위성 네트워크(MLRSNet), OPTIMAL31, PatternNet, RESISC45, RSC11, RSICB128, RSICB256, WHURS19 등 총 10개의 데이터셋을 사용하여 벤치마킹을 진행했습니다. 각 데이터셋은 다양한 클래스 수를 포함하며 모델의 학습 및 평가에 활용됩니다. RS-specific text-prompt 템플릿을 사용해 학습된 이미지와 텍스트 임베딩을 통해 모델의 성능을 평가하였습니다.

- **Performance Highlights**: GeoRSCLIP 모델이 모든 few-shot 적응 방법에서 가장 높은 성능을 보였습니다. 특히 zero-shot 성능이 항상 few-shot 적응 성능을 나타내는 것은 아니라는 점이 주목할 만합니다. TaskRes와 CLIP-LoRA 같은 방법은 특정 데이터셋에서 두드러진 성능을 보여주며, 적응 전략의 유효성은 사용된 데이터셋과 샷 수에 따라 달라질 수 있음을 시사합니다.



### Graph Conditioned Diffusion for Controllable Histopathology Image Generation (https://arxiv.org/abs/2510.07129)
- **What's New**: 본 논문에서는 Graph-Conditioned-Diffusion(GCD)라는 새로운 방법론을 제안하여, 의료 영상의 조건부 생성을 위한 그래프 기반 객체 수준 표현을 활용합니다. 이 방법은 각 주요 구조에 해당하는 그래프 노드를 생성하여 개별 특징과 관계를 캡슐화하고, 텍스트 조건화 메커니즘을 통해 확산 모델에 통합합니다. 이는 생성 과정에서의 세밀한 제어를 가능하게 하여 고품질의 이미지 생성을 이루도록 합니다.

- **Technical Details**: GCD는 확률적 생성 모델이 생성하는 데이터의 편향성을 줄이기 위해 그래프 구조를 도입합니다. 그래프는 이미지의 기본 특징 내에서 명확한 구조를 제공하여, 샘플의 균형과 다양성을 관리할 수 있는 수단을 마련합니다. 이론적 프레임워크를 통해 diffusion process의 전진 및 후진 과정을 기술하며, 주어진 그래프를 바탕으로 이미지 생성 작업을 조정합니다.

- **Performance Highlights**: 실제 조직 병리학의 사례를 통해 제안된 방법이 주석이 달린 환자 데이터를 신뢰성 있게 대체할 수 있음을 입증했습니다. GCD는 기존 데이터의 통계적 특성과 밀접하게 일치하는 합성 데이터 생성을 통해, 진단 애플리케이션과 데이터 공유의 유틸리티를 증가시킵니다. 특히, 영상 분할 작업과 같은 다운스트림 작업에서 실제 데이터에 대한 성능 향상을 보여줍니다.



### Validation of Various Normalization Methods for Brain Tumor Segmentation: Can Federated Learning Overcome This Heterogeneity? (https://arxiv.org/abs/2510.07126)
- **What's New**: 이 연구는 의학 이미징에서의 딥러닝 적용의 도전 과제가 무엇인지 설명하며, 연합 학습(Federated Learning, FL)의 새로운 접근법을 소개합니다. FL은 데이터 프라이버시를 유지하면서도 여러 기관이 협력하여 모델을 학습할 수 있도록 합니다. 이 연구는 비독립적이고 동일하게 분포되지 않은(non-IID) 데이터에서 FL의 성능을 검증하고 다양한 MRI 강도 정규화(normalization) 기술이 뇌 종양 분할(segmentation) 모델의 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 UCSF-PDGM-v3로, 3T MRI 스캐너를 사용하여 촬영된 WHO 등급 2에서 4의 뇌 종양 환자 495명을 포함합니다. 두 개의 피험자가 강한 아티팩트로 인해 제거되어 최종적으로 493명으로 구성된 데이터셋이 마련되었습니다. MRI 이미지의 정규화 방법으로는 MinMax, Z-score, Nyul, Fuzzy C-Mean, WhiteStripe의 5가지가 평가되어, 각 방법이 뇌 종양 분할 모델의 학습과 추론에 미치는 영향을 검토했습니다.

- **Performance Highlights**: FL 방법들은 각 클라이언트에서 불규칙하게 정규화된 데이터에 대해서도 강건성을 보여주었으며, 3D Dice score 92%를 기록하여 중앙 집중형 모델과 비슷한 성능을 달성했습니다. 이 연구의 결과는 FL이 데이터 프라이버시를 침해하지 않으면서도 높은 성능의 모델을 효과적으로 학습할 수 있는 방법이 될 수 있음을 시사합니다. 더 나아가, 임상 환경에서의 실제적 과제를 다뤘다는 점에서 이 연구의 의의를 갖습니다.



### MoRe: Monocular Geometry Refinement via Graph Optimization for Cross-View Consistency (https://arxiv.org/abs/2510.07119)
- **What's New**: 본 논문에서는 MoRe라는 훈련 없는 Monocular Geometry Refinement 방법을 제안합니다. 이 방법은 서로 다른 뷰 간의 일관성을 개선하고 스케일 정렬을 달성하는 데 중점을 두고 있습니다. MoRe는 Monocular 3D foundation models를 활용하여 3D 재구성을 향상시키며, 특히 희소 뷰 렌더링 시나리오에서 새롭게 생성된 뷰 합성의 성능을 개선합니다.

- **Technical Details**: MoRe는 피쳐 매칭(feature matching)을 통해 프레임 간의 관계를 유도하며, 이를 통해 해당 일관성을 확보합니다. 단순한 least squares 최적화를 적용하는 대신, 우리는 그래프 기반 최적화 프레임워크를 구성하여 추정된 3D 포인트와 표면 노멀(surfaces normals)을 사용한 지역 평면 근사를 수행합니다. 이러한 접근 방식은 Monocular geometric priors에서 나타나는 스케일 모호성을 해결하며, 3D 구조를 보존합니다.

- **Performance Highlights**: MoRe는 특히 스파스 뷰 렌더링 시나리오에서 새로운 뷰 합성 성능을 크게 향상시킵니다. 이를 통해 기존의 3D 재구성 문제를 해결하며, 다양한 3D 비전 애플리케이션에 적용될 가능성을 열어줍니다. 실험 결과, MoRe는 기존 방법보다 더 나은 성능을 보이며, 실제 환경에서의 적용 가능성을 높입니다.



### Enhancing Concept Localization in CLIP-based Concept Bottleneck Models (https://arxiv.org/abs/2510.07115)
- **What's New**: 이 논문은 Concept Bottleneck Models(CBMs)를 통한 XAI(Explainable AI)에 대한 접근을 제시하며, 명시적 개념 주석이 필요 없는 방식으로 CLIP을 활용하여 개념을 추출합니다. CLIP의 사용으로 인해 개념 환각(concept hallucination) 현상이 발생할 수 있으며, 이는 이미지 내 개념의 존재를 잘못 예측하게 만듭니다. 이 문제를 해결하기 위해 'CHILI(Concept Hallucination Inhibition via Localized Interpretability)'라는 기법을 도입하여, 이미지 내 픽셀과 목표 개념을 로컬라이즈하는 방법을 사용합니다.

- **Technical Details**: CHILI는 CLIP의 활성화 분리를 통해 물리적 위치와 맥락적 표현을 구분합니다. 이 방법은 이미지 세분화 및 개념의 이진 분류를 위해 사용되어, 기존의 방법들보다 우수한 결과를 보여줍니다. 연구팀은 CLIP 점수가 개념의 실제 위치를 정확하게 나타내지 못한다는 점을 발견하고, 이를 기반으로 활성화 분해 접근 방식을 제안합니다.

- **Performance Highlights**: 실험 결과, CHILI는 한정된 정확도 저하로 새로운 CBMs을 구성할 수 있는 가능성을 보여줍니다. 제안된 방법은 수정된 CBMs의 신뢰성과 해석 가능성을 향상시키며, 실제 사례에 적용하여 성과를 입증합니다. 본 연구는 CBMs의 개념 환각 문제에 대한 새로운 해결책을 제시함으로써, 이 분야의 연구를 촉진시킬 것으로 기대됩니다.



### DADO: A Depth-Attention framework for Object Discovery (https://arxiv.org/abs/2510.07089)
Comments:
          21st International Conference in Computer Analysis of Images and Patterns (CAIP 2025)

- **What's New**: 본 연구에서 제안하는 DADO(Depth-Attention self-supervised technique for Discovering unseen Objects) 모델은 이미지에서 객체를 식별하는 혁신적인 접근 방식을 제공합니다. 이 모델은 attention 메커니즘과 depth 추정 기법을 결합하여, 복잡한 장면 속에서 잠재적인 객체를 동적으로 강조할 수 있도록 합니다. DADO는 기존의 최첨단 방법들보다 높은 객체 발견 정확도와 강건성을 보여주며, 별도의 세부 조정 없이도 효율적으로 작동합니다.

- **Technical Details**: DADO는 Self-Supervised Learning(SSL) 기술을 활용하여 이미지의 attention과 depth 특징을 추출합니다. 이 파이프라인은 단일 RGB 입력 이미지에서 depth 추정과 attention 기능을 동시에 계산하고, 이를 통해 장면을 여러 개의 depth 층으로 분할합니다. DADO는 각 depth 층을 attention 맵과 결합하여 후보 객체를 격리하는데, 이는 히스토그램 기반을 통한 동적 깊이 간격 생성 방식으로 수행됩니다.

- **Performance Highlights**: DADO는 표준 벤치마크에서 평가했을 때, 기존의 객체 발견 방법들에 비해 일관되게 우수한 성능을 보였습니다. 특히, 다양한 깊이 수준에서 객체를 효과적으로 분리할 수 있는 능력을 입증하였습니다. DADO는 분류 레이블 없이도 폭넓은 일반화와 적응력을 제공하여, 실제 세계의 복잡한 상황에서도 유용하게 사용될 수 있습니다.



### Concept Retrieval -- What and How? (https://arxiv.org/abs/2510.07058)
- **What's New**: 이번 논문은 입력 이미지의 핵심 개념을 공유하는 다른 이미지를 검색하는 방법을 제안합니다. 이는 전통적인 이미지 검색 방식과는 달리 시각적 또는 의미적 유사성에 중점을 두지 않고, 고급 의미와 맥락을 효과적으로 캡쳐하는 것을 목표로 합니다. 논문에서는 개념 검색 작업을 정의하고, 네 가지 주요 요구사항을 정리하며 새로운 평가 지표를 소개합니다.

- **Technical Details**: 연구진은 'Bimodal Gaussian distribution' 구조를 활용하여 이웃 이미지 간의 의미 있는 관계를 모델링합니다. 이러한 관계를 통해 입력 이미지의 개념을 효과적으로 파악할 수 있는 방법론을 제안합니다. 이 방법에는 이웃 검색, 적절한 대리 개념 이웃 선정, 그리고 개념을 공유하는 이미지를 추출하는 과정이 포함되어 있습니다.

- **Performance Highlights**: 정량적, 정성적, 그리고 인간 평가 결과를 통해 제안된 방법의 효과성을 입증하였습니다. 이 연구는 이미지를 단순히 시각적 유사성에 기반하여 검색하는 것을 넘어, 다수의 의미적 개념을 공유하는 이미지를 찾을 수 있는 가능성을 제공합니다. 또한, 새로운 평가 지표는 이 요구사항들을 효과적으로 평가하는 데 기여합니다.



### U-Bench: A Comprehensive Understanding of U-Net through 100-Variant Benchmarking (https://arxiv.org/abs/2510.07041)
Comments:
          54 pages. The project can be accessed at: this https URL. Code is available at: this https URL

- **What's New**: 이 논문에서 소개하는 U-Bench는 100개의 U-Net 변형을 28개의 데이터셋과 10개의 이미징 모달리티를 통해 체계적으로 평가하는 대규모 벤치마크입니다. 이것은 통계적 타당성을 바탕으로 하여 모델의 성능과 효율성을 측정하는 첫 번째 시도로, 모델 선택을 위한 가이드를 제공하고 공공 데이터셋으로 결과 재현성을 높일 수 있습니다.

- **Technical Details**: U-Bench는 U-Net의 계층적 인코더, 디코더, 병목 구조 및 스킵 연결로 구성된 일반적인 U자형 모델을 기반으로 합니다. 다양한 U-Net 변형들은 CNN, Attention, Mamba, RWKV와 같은 핵심 구성 요소로 구성되며, 이들은 여러 구성으로 조직되어 성능과 효율성에 영향을 미칩니다. 실험에 사용된 28개의 데이터셋은 초음파, 피부과, 내시경, 안저 사진 등 다양한 모달리티를 포함하고 있습니다.

- **Performance Highlights**: U-Bench 평가 결과, 대부분의 변형이 성능 향상을 보였지만 원래 U-Net보다 통계적으로 유의미한 차이를 보이는 경우는 드물었습니다. 제로샷 성능은 유의미한 개선을 보여 주었으며, U-Score는 정확도와 효율성을 균형 있게 고려하는 지표로서 점진적인 증가 추세를 나타냈습니다. 연구진은 이러한 평가를 통해 모델 추천 시스템을 구축하고, 연구자들이 적절한 아키텍처를 식별할 수 있도록 지원하고 있습니다.



### Bayesian Modelling of Multi-Year Crop Type Classification Using Deep Neural Networks and Hidden Markov Models (https://arxiv.org/abs/2510.07008)
Comments:
          5 pages, 1 figure, accepted conference paper at IEEE International Geoscience and Remote Sensing Symposium, 7-12 July 2024, Athens, Greece

- **What's New**: 이번 연구에서는 최첨단 딥러닝(Deep Learning) 기술과 베이지안 모델링(Bayesian modelling)을 결합하여 다년간의 위성 이미지 시계열(Classification of yearly satellite image time series, SITS)의 분류를 위한 새로운 접근 방식을 제안합니다. 이 방법은 숨겨진 마르코프 모델(Hidden Markov Models, HMMs)과 트랜스포머 인코더(Transformer Encoder, TE) 기반의 심층 신경망(Deep Neural Networks, DNNs)을 통합하여 연도별 농작물 유형 시퀀스의 일관성을 포착하려고 합니다. 이를 통해 예측된 레이블의 시간 일관성을 모델링하는 것이 중요함을 강조하고 있습니다.

- **Technical Details**: 연구 방법론에서는 다년간의 농작물 유형 분류에 대한 캐스케이드 분류(cascade classification) 접근 방식을 소개합니다. 이 접근 방식은 HMM의 잠재적 출력 독립성 가정을 기반으로 하며, DNN을 사용하여 방출 모델을 근사화하고 마르코프 전이 모델을 통해 시간적인 일관성을 유지합니다. 본 연구는 원시 광학 시계열 분류에서 TE를 방출 모델로 사용하며, 두 가지 버전의 TE를 고려하여 클래스 후확률(class posterior probabilities) 추정의 차이를 보여줍니다.

- **Performance Highlights**: 제안된 방법론은 Sentinel-2 데이터에서 수집된 6년간 다년간 농작물 유형 분류 데이터셋에서 검증되었습니다. 결과적으로, HMM을 결합한 경우 DNN 단독으로 사용할 때보다 전체적인 성능과 F1 스코어(F1 scores)가 향상되는 것을 확인했습니다. 이는 DNN과 HMM을 결합함으로써 시간적 일관성을 모델링하는 것이 예측된 레이블의 성능을 크게 개선할 수 있음을 나타냅니다.



### No MoCap Needed: Post-Training Motion Diffusion Models with Reinforcement Learning using Only Textual Prompts (https://arxiv.org/abs/2510.06988)
- **What's New**: 본 논문에서는 강화학습(Reinforcement Learning, RL)에 기반한 포스트 트레이닝 프레임워크(post-training framework)를 제시하여, 인간 모션 생성에 사용되는 사전 훈련된 확산 모델(motion diffusion models)을 단순한 텍스트 프롬프트(textual prompts)만으로 미세 조정(fine-tune)할 수 있도록 한다. 이 접근법은 추가적인 모션 캡처 데이터가 필요하지 않으며, 사전 훈련된 텍스트-모션 검색 네트워크(text-motion retrieval network)를 보상 신호로 활용한다.

- **Technical Details**: 모델은 Denoising Diffusion Policy Optimization(DDPO) 기법을 통해 목표 도메인(target domain)으로의 생성 분포(generative distribution) 전이를 최적화한다. 이를 통해 기존의 모션 적응 방식에 비해 더 유연하고 데이터 효율적이며, 개인 정보 보호를 고려한 모션 생성이 가능하다. 또한, 본 연구는 HumanML3D와 KIT-ML 데이터셋을 활용해 크로스 데이터셋 적응(cross-dataset adaptation) 및 Leave-one-out 모션 실험을 진행한다.

- **Performance Highlights**: 결과는 정량적 지표와 사용자 연구를 통해 본 방법이 생성된 모션의 품질과 다양성을 지속적으로 개선하면서, 원래 분포에 대한 성능을 유지함을 보여준다. RL 기반 포스트 트레이닝이 인간 모션 DMs을 새로운 데이터셋과 미지의 모션 카테고리에 효과적으로 일반화할 수 있음을 입증하였다. 본 접근법은 실제 모션 합성(real-world motion synthesis)에 있어 매우 유망한 솔루션임을 지지하는 근거를 제시한다.



### Addressing the ID-Matching Challenge in Long Video Captioning (https://arxiv.org/abs/2510.06973)
- **What's New**: 이 논문에서는 긴 비디오 캡션 생성을 위한 새로운 벤치마크인 RICE-benchmark를 도입하였으며, ID-Matching(아이디 매칭) 문제를 다루기 위한 도구를 개발했습니다. 기존의 방법들이 갖고 있는 한계를 극복하고, LVLMs(Large Vision Language Models)의 고유한 ID-Matching 능력을 활용하여 향상된 캡션 품질을 목표로 하고 있습니다. 특히, RICE(Recognizing Identities for Captioning Effectively)라는 새로운 방법을 통해 기존의 ID-Matching 성능을 크게 개선하는 방법을 제시합니다.

- **Technical Details**: RICE 방법론은 두 가지 주요 개선점을 기반으로 하고 있습니다. 첫째, 이미지 정보의 활용을 증대시키고, 둘째, 각 개인을 더 잘 설명하기 위해 정보를 늘이는 것입니다. 이 프로세스는 한 대화 내에서 여러 프레임에 대한 캡션 생성을 통해 이루어집니다. 이러한 접근법은 취합된 캡션의 텍스트 유사성을 크게 향상시키고, 최종적으로 ID-Matching 성능을 향상시킵니다.

- **Performance Highlights**: RICE 방법은 GPT-4o에서 적용되었을 때, ID-Matching의 정밀도를 50%에서 90%로, 재현율을 15%에서 80%로 크게 향상시키는 결과를 보여주었습니다. 이 연구 결과는 기존의 방식들보다 우수한 성능을 입증하며, 비디오 캡션 생성 분야에 대한 중요한 통찰력과 도구를 제공할 것으로 기대됩니다. RICE는 따라서 긴 비디오의 캡션에서도 지속적으로 다양한 인물을 추적할 수 있게 만드는 새로운 가능성을 제시합니다.



### Learning Global Representation from Queries for Vectorized HD Map Construction (https://arxiv.org/abs/2510.06969)
Comments:
          16 pages

- **What's New**: 본 연구에서는 현대 자율주행 시스템의 핵심인 온라인 고해상도(HD) 지도 구축을 위해 ‘MapGR(글로벌 표현 학습을 통한 HD 지도 구축)’이라는 새로운 아키텍처를 제안합니다. 기존의 DETR 프레임워크 기반 접근 방식은 독립적인 객체 쿼리에 의존하여 주로 국소 쿼리 관점을 강조했으나, HD 지도에서의 고유한 글로벌 표현을 간과했습니다. 이를 해결하기 위해, MapGR은 글로벌 표현을 학습하고 활용하는 두 가지 모듈인 글로벌 표현 학습(GRL) 모듈과 글로벌 표현 유도(GRG) 모듈을 도입합니다.

- **Technical Details**: GRL 모듈은 모든 쿼리로부터 글로벌 HD 지도 표현을 학습하고, 이를 통해 포괄적인 레스터화된 지도를 예측합니다. 이 예측은 Ground Truth(실제 지도)와 비교하여 감독됩니다. GRG 모듈은 GRL 모듈에서 학습한 글로벌 표현을 각 개별 쿼리에 통합함으로써 최적화를 돕습니다. 이 방식은 개별 쿼리를 최적화하면서도 글로벌한 관점을 유지할 수 있도록 합니다.

- **Performance Highlights**: nuScenes 및 Argoverse2 데이터셋에 대한 평가 결과, 제안한 방식은 평균 정밀도(mean Average Precision, mAP)에서 기존의 주요 기준선보다 상당한 성능 향상을 보여주었습니다. MapGR은 Online HD 지도 구축을 위한 효과적이고 효율적인 접근 방식으로, 주요 방법들과의 함께 사용할 수 있는 플러그 앤 플레이 모듈로 설계되었습니다. 많은 실험을 통해 제안된 접근 방식이 다양한 기준선에서 성능을 상당히 개선함을 확인했습니다.



### Generating Surface for Text-to-3D using 2D Gaussian Splatting (https://arxiv.org/abs/2510.06967)
- **What's New**: 최근 Text-to-3D 모델링에서의 발전은 3D 콘텐츠 생성의 중요한 가능성을 보여주고 있습니다. 하지만 자연계의 복잡한 기하학적 형태로 인해, 진정한 3D 콘텐츠 생성은 여전히 도전적인 과제입니다. 본 논문에서는 DirectGaussian이라는 새로운 방법을 제안하며, 이는 surfel로 표현된 3D 객체의 표면 생성에 중점을 두고 있습니다.

- **Technical Details**: DirectGaussian에서는 조건부 텍스트 생성 모델을 활용하고, 2D Gaussian splatting 기법으로 3D 객체의 표면을 렌더링합니다. 또한, 다중 뷰 기하학적 일관성 문제를 해결하기 위해 최적화 과정에서 생성된 표면에 곡률 제약을 통합했습니다. 이 접근법은 고품질의 3D 콘텐츠 생성을 위한 중요한 초석을 제공합니다.

- **Performance Highlights**: 다양한 실험을 통해 DirectGaussian이 다양한 텍스트 프롬프트에 대해 고충실도의 3D 콘텐츠 생성을 달성할 수 있음을 입증하였습니다. 또한, 360도 주변 뷰 표면 곡률 제약을 도입하여, 최종 출력에서 세밀한 기하학적 세부정보를 보존할 수 있습니다. 이로써, DirectGaussian은 텍스트 기반의 3D 생성 작업에서 새로운 가능성을 보여줍니다.



### OBJVanish: Physically Realizable Text-to-3D Adv. Generation of LiDAR-Invisible Objects (https://arxiv.org/abs/2510.06952)
- **What's New**: 본 연구에서는 LiDAR 감지 시스템의 취약점을 탐색하기 위해 물리적으로 실현 가능한 3D 적대적 생성 방법인 Phy3DAdvGen을 소개합니다. 기존의 공격 방식들이 주로 실제 객체 사라짐을 유도하기 어려운 한계를 가지고 있는 반면, 우리는 텍스트 기반의 입력을 최적화하여 LiDAR가 감지할 수 없는 3D 모델을 생성하는 혁신적인 방법을 제시합니다. 실험 결과, 제안된 방법이 실제 및 시뮬레이션 환경 모두에서 최신 LiDAR 3D 감지기를 속일 수 있음을 보여줍니다.

- **Technical Details**: 우리는 LiDAR 감지 시스템 내에서 보행자의 감지 가능성을 체계적으로 연구하기 위해 CARLA 시뮬레이션 환경에서 다양한 보행자 속성을 조작하고, 여러 객체의 조합을 분석했습니다. 또한, LiDAR가 감지하기 어려운 인간-객체 혼합을 생성하기 위해 동사, 객체, 포즈로 구성된 텍스트 프롬프트를 최적화하는 Phy3DAdvGen 프레임워크를 개발하였습니다. 이 과정에서 Gaussian Splatting 기반으로 생성된 포인트 클라우드 장면이 실제 감지 실험을 통해 효과가 입증되었습니다.

- **Performance Highlights**: 우리가 제안하는 Phy3DAdvGen 방법은 6개 주요 LiDAR 3D 감지기에서 일관되게 효과를 나타냄으로써 LiDAR 시스템의 실제 안전 응용에서 존재하는 취약함을 강조합니다. 실험 결과는 Phy3DAdvGen이 단순한 시뮬레이션을 넘어 실제 환경에서도 효과적으로 공격할 수 있음을 입증하였습니다. 이 연구를 통해 우리는 LiDAR 감지 시스템의 신뢰성을 높이기 위한 새로운 방향을 제시하고 있습니다.



### IAR2: Improving Autoregressive Visual Generation with Semantic-Detail Associated Token Prediction (https://arxiv.org/abs/2510.06928)
- **What's New**: IAR2는 기존의 IAR 모델의 한계를 극복하고, 시멘틱-디테일(semantic-detail) 합성을 지원하는 상향식 오토리그레시브 프레임워크입니다. 이를 통해 세분화된 정보와 전반적인 의미를 동시에 처리할 수 있는 이중 코드북을 도입하여 이미지 생성의 표현력을 획기적으로 확장하였습니다. 이를 통해 더욱 현실적이고 일관된 이미지 생성을 가능하게 합니다.

- **Technical Details**: IAR2는 세분화된 지역 정보를 위한 디테일 코드북과 글로벌 의미 정보를 위한 시멘틱 코드북으로 나누어지는 이중 코드북을 사용합니다. 이 모델은 계층적 예측 방식을 채택하여 먼저 시멘틱 토큰을 예측한 후 디테일 토큰을 예측하며, 지역적 문맥 정보를 활용하여 공간적 일관성을 강화합니다. 또한, 조건부 생성에서는 각 토큰의 중요도에 따라 가이드를 동적으로 조정하는 프로그레시브 어텐션-가이드 CFG 메커니즘을 도입하여 제어합니다.

- **Performance Highlights**: IAR2는 100M 파라미터를 가진 LlamaGen 모델에 대해 FID를 6.09에서 4.80으로 감소시켰으며, 1.5B 파라미터의 IAR2-XXL 모델은 ImageNet에서 FID 1.50을 기록하여 2B 파라미터의 VAR 모델(FID 1.92)을 초월하였습니다. 이로써 IAR2는 기존 모델보다 뛰어난 성능을 보여주며, 32개의 GPU만으로도 우수한 성능을 달성하는 효율성을 입증하였습니다.



### Label-frugal satellite image change detection with generative virtual exemplar learning (https://arxiv.org/abs/2510.06926)
- **What's New**: 본 논문에서는 위성 이미지 변화 탐지에 대한 새로운 알고리즘을 제안합니다. 이 알고리즘은 능동적 학습(active learning)을 기반으로 하여, 주어진 데이터 중 가장 중요한 샘플에 대한 라벨을 요청하는 방식으로 작동합니다. 특히, 가장 중요한 데이터 샘플인 가상 예제(virtual exemplars)를 생성하고, 이를 통해 라벨 효율적인 학습을 수행합니다.

- **Technical Details**: 본 연구에서 제안하는 알고리즘은 그래프 합성곱 신경망(invertible graph convection networks)을 사용해 변화 탐지를 수행합니다. 이 알고리즘은 오라클(oracle)에게 라벨링을 요청할 가상 예제를 생성하며, 이러한 예제들은 데이터의 대표성, 다양성 및 모호성을 측정하여 생성됩니다. 즉, 변화 또는 비변화 클래스의 비선형 매니폴드를 포함하는 데이터로 제약하여 동작합니다.

- **Performance Highlights**: 본 논문의 알고리즘은 기존 변화 탐지 방식들과 비교하여 높은 효율성을 보입니다. 실험 결과, 제안된 모델이 라벨 효율적인 학습을 통해 변화 탐지 정확도를 개선함을 나타냈습니다. 이로 인해 위성 이미지 변화 탐지에서의 기술적 발전을 이룰 것으로 기대됩니다.



### Lung Infection Severity Prediction Using Transformers with Conditional TransMix Augmentation and Cross-Attention (https://arxiv.org/abs/2510.06887)
- **What's New**: 이 연구에서는 폐 감염의 심각도를 평가하기 위한 새로운 방법을 제시합니다. 제안된 방법은 CT 및 흉부 X-ray 이미지 모두에 적용 가능하며, QCross-Att-PVT라는 Transformer 기반 아키텍처를 채택합니다. 이 아키텍처는 병렬 인코더와 교차 게이티드 주의 메커니즘을 통합하여 다중 스케일 특성을 효과적으로 캡처합니다.

- **Technical Details**: QCross-Att-PVT 아키텍처는 입력 이미지를 네 개의 사분면으로 나누고 각 사분면을 개별적으로 처리하여 지역적 변화에 초점을 맞춥니다. 또한, Conditional Online TransMix라는 데이터 증강 기법을 도입하여 학습 데이터의 불균형 문제를 해결합니다. 이 방법론은 두 개의 벤치마크 데이터셋, 즉 RALO CXR 및 Per-COVID-19 CT에서 평가되었습니다.

- **Performance Highlights**: 제안된 방법은 여러 최신의 딥러닝 모델들보다 지속적으로 우수한 성능을 보여 줍니다. 결과는 데이터 증강과 게이티드 주의 메커니즘이 모델의 강건성과 예측 정확성 향상에 중요한 역할을 한다는 것을 강조합니다. 이 접근 방식은 임상 진단, 질병 모니터링 및 개인화된 치료 계획을 지원하기 위한 신뢰할 수 있고 적응 가능한 도구를 제공합니다.



### HARP-NeXt: High-Speed and Accurate Range-Point Fusion Network for 3D LiDAR Semantic Segmentation (https://arxiv.org/abs/2510.06876)
Comments:
          Accepted at IROS 2025 (IEEE/RSJ International Conference on Intelligent Robots and Systems)

- **What's New**: HARP-NeXt는 고속 및 정확한 LiDAR 시맨틱 세그멘테이션을 위한 새로운 네트워크로, GPU 병렬 처리를 활용하여 데이터 준비를 가속화하는 혁신적인 전처리 방법론을 제안합니다. 기존 방법들이 느려지는 성능을 개선하기 위해 많은 전 처리 단계를 요구하는 문제점을 해결합니다. 이 네트워크는 무거운 딥 레이어 스태킹 없이 효율적으로 패턴을 포착할 수 있는 Conv-SE-NeXt 모듈도 포함합니다.

- **Technical Details**: HARP-NeXt의 핵심 기능 차별화 요소로는 쌍별적으로 컨볼루션을 제거하는 심도가 분리된 컨볼루션(Depth-wise Separable Convolutions)을 통합하고, 각 채널의 중요성을 가중하는 squeeze-and-excitation 메커니즘을 활용합니다. 이를 통해 실질적인 파라미터 수를 줄이고, 더 중요한 피처에 집중할 수 있도록 설계되었습니다. 또한, 다중 스케일 범위-포인트 융합(backbone)을 통해 다양한 수준과 해상도의 공간 피처 표현을 향상시킵니다.

- **Performance Highlights**: 실험 결과, HARP-NeXt는 nuScenes 및 SemanticKITTI 벤치마크에서 기존 모든 최첨단 방법들과 비교할 때 우수한 속도-정확도 트레이드오프를 달성하였고, 앙상블 모델이나 테스트 타임 증강(TTA)에 의존하지 않아도 상위 랭크의 PTv3와 유사한 성능을 보이며 24배 빠른 속도로 실행됩니다.



### Explaining raw data complexity to improve satellite onboard processing (https://arxiv.org/abs/2510.06858)
Comments:
          Preprint: European Data Handling & Data Processing Conference (EDHPC) 2025

- **What's New**: 이 연구는 위성에서 AI 모델을 직접 운용할 수 있는 가능성을 제시하며, 기존에 사용된 전처리된 이미지 대신 원시(원래) 센서 데이터를 활용하는 접근 방식을 탐구합니다. 원시 데이터에 대한 깊이 학습(deep learning) 모델을 통한 객체 탐지(object detection) 및 분류(classification) 성능에 대한 평가를 실시하고, 이를 위해 시뮬레이션 워크플로우를 개발하였습니다. 두 개의 객체 탐지 모델(YOLOv11s 및 YOLOX-S)을 원시 및 L1 데이터셋에서 훈련하여 성능을 비교합니다.

- **Technical Details**: 이 연구에서는 원시 데이터를 객체 탐지 성능에 미치는 영향을 분석하기 위해 시뮬레이션 워크플로우를 구성합니다. 원시 데이터와 L1 이미지를 결합하여 고해상도 멀티스펙트럼 이미지와 통합된 원시 제품을 생성하며, EDSR 같은 신경망을 활용하여 원시 이미지를 복원합니다. 가장 혁신적인 점은 파노라마 이미지 및 다중 스펙트럼 이미지를 조합하여 객체 탐지 모델에 적합한 데이터 세트를 만드는 것입니다.

- **Performance Highlights**: 모델은 낮은 신뢰도(threshold)에서 유사한 성능을 보이나, 원시 데이터로 훈련된 모델은 높은 신뢰도 수준에서 객체 경계 식별에 어려움을 겪습니다. 이는 AI 모델이 원시 이미지를 처리할 때 발생하는 주요한 문제를 나타내며, 향후 더 나은 경계 인식을 위해 AI 아키텍처를 개선해야 할 필요성을 제안합니다. 연구 결과는 온보드 AI의 성능 향상에 중요한 통찰력을 제공한다고 할 수 있습니다.



### Online Generic Event Boundary Detection (https://arxiv.org/abs/2510.06855)
Comments:
          ICCV 2025

- **What's New**: 이 논문에서는 Online Generic Event Boundary Detection (On-GEBD)이라는 새로운 작업을 소개합니다. 이는 스트리밍 비디오에서 이벤트 경계를 즉시 탐지하는 것을 목표로 하며, 이전의 GEBD 방법과는 다르게 미래 프레임에 접근하지 않고 real-time으로 데이터를 처리합니다. 이를 통해 인간의 시각적 인식을 모방하고자 합니다.

- **Technical Details**: 제안된 On-GEBD 프레임워크는 Estimator라는 새로운 구조로, Consistent Event Anticipator (CEA)와 Online Boundary Discriminator (OBD)의 두 가지 주요 구성 요소로 이루어져 있습니다. CEA는 이전 프레임만을 기반으로 향후 프레임에 대한 예측을 생성하며, OBD는 예측 오류를 측정하여 통계적 검사를 통해 동적 경계 임계값을 조정합니다. 이 방법은 비디오 스트리밍 시 실시간으로 이벤트 경계를 식별하는 데 필수적입니다.

- **Performance Highlights**: 실험 결과, Estimator는 기존의 온라인 비디오 이해 모델을 활용한 모든 기초 모델을 능가하였으며, Kinetics-GEBD와 TAPOS 데이터셋에서 오프라인 GEBD 방법과 유사한 성능을 달성했습니다. 본 연구는 On-GEBD라는 새로운 도전 과제를 제시하고, 이를 위한 혁신적인 프레임워크를 통해 인간 인식에 더욱 가까운 모델을 개발했음을 보여줍니다.



### Continual Action Quality Assessment via Adaptive Manifold-Aligned Graph Regularization (https://arxiv.org/abs/2510.06842)
Comments:
          Extended Version of MAGR (ECCV 2024 Oral Presentation)

- **What's New**: 이번 연구에서는 Continual AQA (CAQA)라는 새로운 설정을 도입하여 Action Quality Assessment (AQA) 작업에 지속적 학습(Continual Learning, CL)을 확장합니다. CAQA는 비정상적인 품질 분포를 다루면서 이전에 획득한 지식을 유지하는 문제를 해결하고, 지속적인 점수 회귀를 필요로 하는 새로운 도전과제를 제공합니다. 이를 통해 AQA의 현실 세계 적용성을 향상시키고 정밀한 모션 단서를 지속적으로 캡처할 수 있는 방법을 제안합니다.

- **Technical Details**: CAQA는 Adaptive Manifold-Aligned Graph Regularization (MAGR++)라는 혁신적인 프레임워크를 통해 과적합(overfitting)과 분포 이동(distribution shift) 문제를 해결합니다. MAGR++는 층 적응형 파라미터 조정 전략을 통해 얕은 층은 고정하고 깊은 층은 조정하여 세션 특정 변화를 수용하며, 역사적 특성을 현재 표현 공간으로 변환하는 매니폴드 프로젝터와 로컬 및 글로벌 일관성을 유지하는 그래프 정규화를 포함합니다.

- **Performance Highlights**: 네 가지 CAQA 벤치마크와 세 가지 AQA 데이터셋을 통해 폭넓은 실험을 진행한 결과, MAGR++는 최신 성과를 달성하였으며, 오프라인에서 1.6%~6.5%, 온라인에서 4.0%~21.8%의 강력한 베이스라인을 초과했습니다. 평균적으로 오프라인에서는 3.6%, 온라인에서는 12.2%의 성과 향상을 보여주며, 이로 인해 MAGR++의 강력함과 효과성을 확인할 수 있었습니다.



### Lattice-allocated Real-time Line Segment Feature Detection and Tracking Using Only an Event-based Camera (https://arxiv.org/abs/2510.06829)
Comments:
          12 pages, 13 figures, 6 tables, ICCV Workshop NeVi2025

- **What's New**: 이 연구는 고해상도 이벤트 기반 카메라를 활용하여 실시간 라인 세그먼트 감지 및 추적을 수행하는 새로운 방법을 제시합니다. 이 시스템은 (i) 속도 불변 이벤트 표현, (ii) 피팅 점수 기반의 라인 세그먼트 감지, (iii) 엔드포인트를 perturbating(변형)하여 라인을 추적하는 방식을 포함하는 격자 할당 파이프라인으로 구성됩니다. 이를 통해 기존 방식보다 높은 정확도와 실시간 성능을 달성하며, 독립형 이벤트 카메라 작동을 가능하게 합니다.

- **Technical Details**: 라틴스 할당 파이프라인은 이벤트를 속도 불변 방식으로 저장하는 'SCARF' 기법을 사용합니다. 이 방법은 하이레졸루션 카메라에서 200 Hz 이상의 감지 속도와 400 Hz 이상의 추적 속도를 달성합니다. 또한, 각 블록에서 사건을 독립적으로 처리하며, 이를 통해 실시간 처리의 복잡성을 완화합니다. 이 파이프라인은 이벤트의 대량을 효율적으로 처리하여 매우 복잡한 형태의 짧은 세그먼트 추출을 가능하게 합니다.

- **Performance Highlights**: 제안된 방식은 640x480 해상도의 기록된 데이터셋과 공개된 데이터셋을 사용하여 실시간 성능과 라인 세그먼트 정확성에서 최신 기술보다 우수성을 입증하였습니다. 결과적으로, 높은 이벤트 비율에서도 20 Mev/s의 이벤트 기반 저장과 동시에 200 Hz의 세그먼트 감지 및 400 Hz의 추적 성능을 보여주었습니다. C++로 구현된 바이브라스를 오픈소스로 공개하여 향후 연구자들이 이를 활용할 수 있도록 도움을 줍니다.



### StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidanc (https://arxiv.org/abs/2510.06827)
Comments:
          Accepted to ICCV 2025; CVPRW AI4CC 2024 (Best Paper + Oral)

- **What's New**: 본 논문은 이미지 생성 분야에서 텍스트-이미지 변환(difussion models)의 새로운 발전을 보여준다. 특히, 기존의 이미지 프롬프트를 활용한 스타일 및 콘텐츠 제어 방식에서 발생하는 콘텐츠 유출(content leakage)의 문제를 해결하기 위한 두 가지 방안을 제안한다. 이러한 방안은 classifier-free guidance (CFG)의 확장과 negative visual query guidance (NVQG)의 도입으로 구성된다.

- **Technical Details**: 제안된 방법인 StyleKeeper는 텍스트 프롬프트와 시각적 스타일 프롬프트를 결합하여 콘텐츠 유출 없이 이미지를 생성한다. 이 과정에서 swapping self-attention, NVQG, 최적 레이어 선택 및 실제 이미지의 확률적 인코딩을 활용하며, ControlNet과의 호환성도 갖춘다. 이 메커니즘은 모델이 기반하는 원본 프로세스와 참조 프로세스 간의 키-값(key-value) 정보를 효과적으로 전환하여 콘텐츠와 스타일 간의 분리를 보장한다.

- **Performance Highlights**: 제안된 방법은 다양한 스타일과 텍스트 프롬프트를 통해 기존 접근법보다 우수한 성능을 발휘하며, 결과 이미지는 텍스트 프롬프트와 잘 일치하는 것으로 나타났다. 이를 통해 스타일 반영과 콘텐츠 유출 방지가 효과적으로 이루어져, 전체적인 이미지 품질과 텍스트 정렬이 향상된 것을 보여준다. 코드가 공개되어 재현성(reproducibility)이 보장된다.



### Efficient Discriminative Joint Encoders for Large Scale Vision-Language Reranking (https://arxiv.org/abs/2510.06820)
Comments:
          preprint

- **What's New**: 본 논문에서는 EDJE(Efficient Discriminative Joint Encoder)를 제안하여 시각 및 언어 모달리티의 효율적인 조합을 실현합니다. 기존의 디지털 이미지 검색에서 대량의 임베딩을 필요로 하는 기술적 제약을 해결하기 위해, 이미지 특성을 오프라인에서 미리 계산하고 경량화된 어댑터를 통해 압축하여 온라인 추론 시 성능을 극대화합니다. EDJE는 강력한 검색 성능을 유지하면서도 저장 공간과 계산 비용을 크게 줄여, 고속 추론을 가능하게 합니다.

- **Technical Details**: EDJE는 언어 모델과 함께 사용되는 압축된 비주얼 토큰을 사용하여 고급 교차 모달 상호작용을 지원합니다. 이를 통해 이미지와 텍스트 간의 상관관계를 공유하는 효과적인 임베딩 공간을 제공하며, 데이터 처리 시 대규모 검색에서의 효율성을 극대화합니다. EDJE는 50,000 개 이미지-텍스트 쌍을 1초에 처리할 수 있으며, 각 이미지당 49kB의 디스크 저장 공간을 요구합니다.

- **Performance Highlights**: EDJE는 다양한 임베딩 기반 모델과 조합하여 제로샷 검색 성능을 일관되게 개선합니다. 특히, SigLIP2와 같은 강력한 비주얼 백본을 사용했을 때, EDJE는 과거 조인트 인코더와 비슷한 성능을 보이며, 운영 효율성에서 훨씬 더 우수한 결과를 보여줍니다. 성능 평가 결과, EDJE는 표준 벤치마크(Flickr30k, MS-COCO)에서 경쟁력 있는 성능을 기록했습니다.



### VA-Adapter: Adapting Ultrasound Foundation Model to Echocardiography Probe Guidanc (https://arxiv.org/abs/2510.06809)
- **What's New**: 이 논문은 심장 초음파 촬영에서 AI 기술을 활용해 실시간 조언을 제공하는 프로브 가이던스(Probe Guidance) 시스템을 제안합니다. 근본 모델(Foundation Model)에서 학습한 의학적 지식을 바탕으로, 초음파 이미지를 얻기 위한 최적의 프로브 조정 전략을 학습하는 새로운 방법론을 소개하고 있습니다. 특히, 제안하는 Vision-Action Adapter (VA-Adapter)는 세밀하게 디자인되어 학습 효율성을 극대화합니다.

- **Technical Details**: VA-Adapter는 이미지 인코더에 삽입되어 영상-행동(vision-action) 시퀀스를 인코딩하여 프로브 조정 능력을 향상시킵니다. 이를 통해, 기존의 프리트레인된 초음파 모델의 기능을 극대화하는 동시에, 모델의 고귀도(parameter efficiency)를 유지합니다. 또한, VA-Adapter는 프로브의 3D 구조를 이해하고 결정을 내리는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, VA-Adapter는 기존의 강력한 프로브 가이던스 모델보다 뛰어난 성능을 보여줍니다. 이 성과는 제한된 파라미터로도 이뤄져, 비용 효율적(training cost-efficient)이며, 의료 분야에서 AI 활용의 가능성을 높이는 데 기여합니다. 차후 논문 수락 후 관련 코드를 공개할 예정이며, 연구의 지속적인 진행에 주목할 필요가 있습니다.



### Extreme Amodal Face Detection (https://arxiv.org/abs/2510.06791)
- **What's New**: 이 논문은 극단적인 비가시적 객체 탐지(extreme amodal detection)에 대한 연구를 다룹니다. 이와 같은 탐지는 입력 이미지에서 완전히 보이지 않는 객체의 2D 위치를 추론하는 것으로, 기존의 비가시적 탐지(amodal detection)와는 차별점을 가집니다. 특히 얼굴 탐지(face detection)를 하위 문제로 설정하여 안전과 프라이버시와 관련된 동기를 제공하지만, 방법론은 특정 클래스에 국한되지 않습니다.

- **Technical Details**: 기존의 접근법들은 이미지 시퀀스를 활용하여 감지되지 않은 부분을 주변 프레임에서 보완하거나 생성 모델(generative model)을 사용하여 가능한 완성을 샘플링하는 방식입니다. 하지만 본 연구는 단일 이미지(single-image) 작업을 고려하고, 이미지의 맥락적 단서를 활용해 보이지 않는 얼굴의 존재를 추론하는 보다 효율적이고 샘플-free한 접근법을 제안합니다. 열지도(heatmap) 기반의 극단적 비가시적 객체 탐지기를 설계하여 이미지에서 최소한의 정보를 가지고 많은 정보를 예측하는 문제를 해결합니다.

- **Performance Highlights**: 본 방법은 새로운 작업에 대해 강력한 결과를 도출하며, 기존의 비효율적인 생성 접근 방식보다 성능이 우수한 것으로 확인되었습니다. 선택적 coarse-to-fine 디코더(selective coarse-to-fine decoder)를 통해 더욱 효율적으로 객체를 탐지할 수 있습니다. 이 연구는 단일 이미지에서 자동으로 얼굴을 감지할 수 있는 가능성을 제시하며, 향후 다양한 응용 분야에 적용될 수 있습니다.



### TTRV: Test-Time Reinforcement Learning for Vision Language Models (https://arxiv.org/abs/2510.06783)
- **What's New**: 이번 연구에서는 Reinforcement Learning (강화 학습)에서 기존의 보상 신호 추출 방식이 라벨이 있는 데이터와 전용 훈련 구간에 의존하고 있다는 점에 주목하였습니다. 우리는 TTRV라는 새로운 방법을 제안하여, 라벨이 없는 데이터 세트에서 추론 시 모델을 즉시 조정하여 시각 언어 이해를 향상시키고자 합니다. 이 방식은 사람의 환경에서 직접 배우는 방식과 유사하게 작용합니다.

- **Technical Details**: TTRV는 Group Relative Policy Optimization (GRPO) 프레임워크를 기반으로 하며, 저희는 기본 모델의 출력 빈도에 따라 보상을 설계하는 접근 방식을 취했습니다. 각 테스트 샘플에 대해 여러 번 추론하면서 모델의 출력을 제어하기 위해, 출력 분포의 엔트로피를 낮추는 방향으로도 보상을 추가하였습니다. 이를 통해 모델의 다양성을 동시에 관리하며, 성능을 극대화하고자 합니다.

- **Performance Highlights**: TTRV는 객체 인식(object recognition) 및 시각 질문 응답(visual question answering, VQA) 모두에서 일관된 성능 향상을 보였습니다. 각각 최대 52.4% 및 29.8%의 개선을 달성했으며, 16개의 다양한 테스트에서 평균적으로 24.6% 및 10.0%의 성능을 향상시켰습니다. 특히 TTRV을 InternVL 8B에 적용하였을 때, 8개의 기준에서 GPT-4o를 평균 2.3% 초과하며, VQA 성능도 여전히 경쟁력을 유지하고 있다는 것을 보여주었습니다.



### A deep multiple instance learning approach based on coarse labels for high-resolution land-cover mapping (https://arxiv.org/abs/2510.06769)
Comments:
          14 pages, 4 figures, accepted conference paper at SPIE REMOTE SENSING, 3-7 September 2023, Amsterdam, Netherlands

- **What's New**: 본 논문은 저해상도 참조 데이터를 활용하여 고해상도 토지 피복 분류기를 훈련하는 새로운 방법을 제안합니다. 이는 기존 저해상도 또는 구식 제품에서 대량의 약한 라벨을 수집할 수 있는 가능성에 착안하여 개발되었습니다. 특히, Deep Multiple Instance Learning (DMIL) 기법을 기반으로 하여 고해상도 이미지의 픽셀 수준 다중 클래스 분류기를 훈련하고 약한 저해상도 라벨을 예측하는 시스템을 고안하였습니다.

- **Technical Details**: 이 연구에서는 두 가지 가정을 기반으로 한 다중 클래스 및 다중 레이블 설정의 PU-MIL(Positive-Unlabeled Learning) 접근 방식을 제안합니다. 첫 번째는 다중 클래스 레이블이 패치 내에서 대부분의 픽셀에 대한 정보를 제공하고, 두 번째는 다중 레이블 설정에서 패치 내에서 여러 토지 피복 클래스가 존재할 수 있음을 인정합니다. 제안된 방법은 픽셀 수준의 DNN을 학습하여 저해상도 라벨을 예측하면서 고해상도 라벨을 직접적인 감독 없이 암묵적으로 학습합니다.

- **Performance Highlights**: 본 연구의 실험 결과는 2020 IEEE GRSS 데이터 융합 대회 데이터셋을 사용하여 제안한 접근 방식이 표준 훈련 전략에 비해 효과적임을 보여주었습니다. Sentinel-2 다중 스펙트럼 이미지를 활용하여 10m 해상도의 토지 피복 맵을 생성하고, MODIS에서 유래한 500m 해상도의 라벨을 훈련 데이터로 사용하여 그 성능을 검증하였습니다.



### Transforming Noise Distributions with Histogram Matching: Towards a Single Denoiser for A (https://arxiv.org/abs/2510.06757)
Comments:
          12 pages

- **What's New**: 이 논문에서는 Gaussian denoisers의 일반화 문제를 해결하기 위해 히스토그램 매칭(Histogram Matching) 방법을 제안합니다. 기존의 supervised 및 self-supervised 방법들이 out-of-distribution noise를 효과적으로 처리하지 못하는 단점을 보완하고자 합니다. 특히, 히스토그램 매칭을 통해 다양한 노이즈를 목표 Gaussian 분포로 변환함으로써 노이즈 제거 성능을 향상시키는 순환 체계를 구축합니다. 이 접근법은 이미지 처리 분야에서 중요한 발전을 이루는 것으로 기대됩니다.

- **Technical Details**: 제안된 방법은 주어진 노이즈 이미지를 히스토그램 매칭을 통해 변환하고, 해당 변환된 이미지를 다시 denoise하여 더 정확한 노이즈 추정치를 생성하는 방식으로 구성됩니다. 이를 통해 noise transformation과 denoising 간의 상호 촉진이 이루어집니다. 추가적으로, pixel-shuffle down-sampling과 intrapatch permutation을 적용하여 공간적 및 채널 간 상관관계를 제거하고, 노이즈 변환 효과를 극대화합니다. 이러한 기술적 접근이 Gaussian denoiser의 일반화 능력을 상당히 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법을 통해 DMID (Denoising Method Improved by Denoising) 모델의 denoising 성능이 크게 향상되었습니다. 구체적으로, 다양한 out-of-distribution 노이즈 처리에서 PSNR과 SSIM 값이 각각 11.81dB와 0.517만큼 증가했습니다. 이러한 결과는 기존 모델들이 해결하지 못했던 현실 세계의 복잡한 노이즈를 효과적으로 제거하는 데에 기여했습니다. 따라서 본 논문이 제안하는 히스토그램 매칭 접근법은 다양한 노이즈 처리에 있어 뛰어난 일반화 능력을 보여줍니다.



### OBS-Diff: Accurate Pruning For Diffusion Models in One-Sho (https://arxiv.org/abs/2510.06751)
- **What's New**: 본 연구에서는 대규모 텍스트-이미지 확산 모델의 정확하고 훈련이 필요 없는 압축을 위한 새로운 원샷 프루닝(One-shot pruning) 프레임워크인 OBS-Diff를 제안합니다. 이 방법은 기존의 Optimal Brain Surgeon (OBS) 방법을 현대의 복잡한 확산 모델 아키텍처에 맞게 조정하며, 다양한 프루닝의 밀도를 지원합니다. 특히, 이 연구에서는 반복적 노이즈 제거 (denoising) 과정의 동적 특성을 반영한 새로운 히essian 구성을 도입하여 초기 시간 단계의 오류 축적을 완화하는 방식으로 프루닝 기준을 조정합니다.

- **Technical Details**: OBS-Diff는 다양한 아키텍처를 사전 훈련 없이 프루닝할 수 있는 일반적이고 훈련 없는 프레임워크입니다. 이는 Multimodal Diffusion Transformer (MMDiT)와 같은 현대의 확산 모델 아키텍처를 처리할 수 있도록 OBS 프레임워크를 조정합니다. 오차가 초기 단계에서 누적되는 결과를 반영하여, 매끄러운 로그 감쇠 가중치 방식을 통합한 시간 인식 히essian 구성을 제안하며, 고비용의 교정(calibration) 프로세스를 경감할 수 있는 그룹-단위 순차 프루닝 전략을 개발합니다.

- **Performance Highlights**: OBS-Diff는 새로운 state-of-the-art 훈련 없는 확산 모델 프루닝 방법을 제시하며, 높은 시각적 품질을 유지하면서도 추론 가속(inference acceleration)을 달성합니다. 연구 결과에 따르면, OBS-Diff는 다양한 밀도 패턴과 수준에서 다른 레이어-와이즈 프루닝 방법들을 초과하는 성능을 보여주었습니다. 이 연구는 대규모 텍스트-이미지 확산 모델의 훈련 상황 없이도 효율적으로 프루닝을 가능하게 함으로써, 기술 접근성을 크게 향상시키는 방향으로 기여하고 있습니다.



### DeRainMamba: A Frequency-Aware State Space Model with Detail Enhancement for Image Deraining (https://arxiv.org/abs/2510.06746)
Comments:
          accepted by IEEE SPL

- **What's New**: 본 논문에서는 이미지 디레인 이미지 복원을 위한 새로운 모델인 DeRainMamba를 제안합니다. 이 모델은 주파수-인식 상태 공간 모듈(Frequency-Aware State-Space Module, FASSM)과 다방향 인식 합성곱(Multi-Directional Perception Convolution, MDPConv)을 통합하여 눈비의 세밀한 특징을 효과적으로 포착합니다. 이를 통해 기존의 Mamba 기반 모델보다 향상된 성능을 발휘하면서도 적은 파라미터와 저렴한 계산 비용을 유지합니다.

- **Technical Details**: DeRainMamba는 고주파 특성 정보를 효과적으로 활용하기 위해 주파수 도메인에서 비와 같은 공명을 식별하는 기능을 포함합니다. FASSM은 Fourier 변환을 통해 비의 선명한 특징과 고주파 이미지 세부사항을 구별하며, MDPConv는 비 강화 과정에서 지역 구조를 복원하는 데 초점을 맞춥니다. 이러한 접근 방식은 기존 모델의 한계를 극복하고, 더욱 우수한 디테일 복원을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, DeRainMamba는 PSNR 및 SSIM 지표에서 최첨단 방법을 일관되게 초과 성능을 보여주었으며, 일반적인 이미지 복원 작업에서 우수한 성능을 입증하였습니다. 이러한 결과는 주파수 도메인 모델링과 공간 세부 사항 향상을 결합한 접근 방식의 효과성을 입증합니다. 전체적인 원본과 정제된 피쳐의 통합 및 세밀한 명암 조정으로 배경 복원 시 충실도를 개선하였습니다.



### Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities (https://arxiv.org/abs/2510.06743)
Comments:
          The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 기반으로 한 OCR(광학 문자 인식)에 대한 평가 프레임워크의 필요성을 제시하고 있습니다. 기존의 평가 지표들이 역사적 문서에 대한 특정 오류와 시간적 편향을 포착하는 데 실패하는 점을 강조하며, 새로운 메트릭인 HCPR(역사적 문자 보존 비율)과 AIR(구식 삽입 비율)을 도입하였습니다. 이 방법론은 디지털 인문학에 종사하는 연구자들에게 모델 선택과 품질 평가를 위한 가이드를 제공합니다.

- **Technical Details**: 연구에서는 18세기 러시아 문서를 대상으로 한 LLM 기반 역사적 OCR 평가 프레임워크를 제시합니다. 수집된 데이터는 러시아 시민 서체로 인쇄된 428개의 독특한 18세기 도서에서 1,029페이지를 포함하고 있으며, 기존 OCR 시스템들이 어려움을 겪는 특유의 오탈자 및 고전 문법 형태를 포함하고 있습니다. 또한, 각 모델의 출력 변동성을 감안한 종합적인 안정성 테스트를 수행했습니다.

- **Performance Highlights**: 실험 결과, Gemini 및 Qwen 모델이 기존 OCR 시스템보다 뛰어난 성능을 보였으나, '과거화(over-historicization)'라는 요소로 인해 올바르지 않은 역사적 시점에서 고어 문자 삽입 현상이 발생했습니다. LLM들이 특정 역사적 문서에 대해 예상치 못한 시간적 편향을 보임으로써, 전통적인 평가 방법들이 이러한 문제를 탐지할 수 없는 한계가 있음을 확인하였습니다. 결과적으로, LLM을 활용한 역사적 문서 변환의 정확성을 높이기 위해서는 새로운 평가 방안이 필요하다는 결론을 내고 있습니다.



### SCas4D: Structural Cascaded Optimization for Boosting Persistent 4D Novel View Synthesis (https://arxiv.org/abs/2510.06694)
Comments:
          Published in Transactions on Machine Learning Research (06/2025)

- **What's New**: 이 논문에서는 SCas4D라는 새로운 cascading optimization framework를 제안합니다. 이 프레임워크는 3D Gaussian Splatting에서의 구조적 패턴을 활용하여 동적 장면을 모델링합니다. 특징적으로, SCas4D는 층위적 패턴을 통해 여러 개의 Gaussian이 유사한 변형을 공유하는 것을 이용하여, coarse level에서 fine level로의 점진적인 변형 개선을 통해 신속한 수렴을 이룹니다.

- **Technical Details**: SCas4D는 전체 100번의 반복(iterations)만으로 변형을 구현할 수 있으며, 기존 방법에 비해 훈련(iteration) 횟수를 20분의 1로 줄입니다. 이 방법은 self-supervised articulated object segmentation, novel view synthesis와 dense point tracking 작업에서 효과적임을 입증하고 있습니다. 또한, 이 프레임워크는 기존 3D Gaussian Splatting의 구조적 정보를 활용함으로써 효율성을 극대화하고 있습니다.

- **Performance Highlights**: SCas4D는 다수의 실험을 통해 동적 장면의 트래킹 성능과 고품질의 분할(segmentation) 작업에서 매우 경쟁력 있는 성능을 보여주고 있습니다. 특히, 이 방법은 novel view rendering 및 포인트 트래킹 작업에 대해 기존 방법들과 비교할 수 있는 성능을 가지고 있으며, 3DGS의 기초적 구조적 정보를 효과적으로 활용하여 개선된 결과를 얻고 있습니다.



### Semantic Segmentation Algorithm Based on Light Field and LiDAR Fusion (https://arxiv.org/abs/2510.06687)
- **What's New**: 이 논문에서는 조명이 차단된 복잡한 환경에서의 자율 주행을 위한 세미틱 세그멘테이션의 과제를 해결하기 위해 라이트 필드 데이터와 포인트 클라우드 데이터를 통합한 최초의 다중 모달 세미틱 세그멘테이션 데이터셋인 TrafficScene을 제안합니다. 이 데이터셋은 모든 라이트 필드 뷰포인트에 대한 의미론적 주석을 제공하여 차단된 또는 작은 객체에 대한 정보를 효과적으로 보완합니다. 또한, 새로운 세그멘테이션 알고리즘인 Multimodal Light Field Point Cloud Fusion Segmentation Method (Mlpfseg)을 소개하여 라이트 필드 이미지와 포인트 클라우드의 동시 세그멘테이션을 가능하게 합니다.

- **Technical Details**: Mlpfseg는 기능 보완(feature completion) 및 깊이 인지(depth perception) 모듈을 포함하여 이미지와 포인트 클라우드를 동시에 세그멘테이션합니다. 기능 보완 모듈은 포인트 클라우드와 이미지 픽셀 간의 밀도 불일치를 해결하기 위해 차별적 재구성을 수행하며, 깊이 인지 모듈은 주의 점수(attention scores)를 보강하여 차단된 객체에 대한 인식을 향상시킵니다. 이러한 접근 방식을 통해 이 방법은 이미지 또는 포인트 클라우드 각각만을 사용하는 세그멘테이션보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 단일 이미지 세미틱 세그멘테이션 방법에 비해 1.71의 Mean Intersection over Union (mIoU) 개선을 달성하였고, 포인트 클라우드 전용 세그멘테이션에 비해서는 2.38의 mIoU 향상을 보여줍니다. 이는 다중 모달 환경에서의 차단된 객체 인식 능력을 크게 향상시킵니다. 결론적으로, Mlpfseg는 라이트 필드와 포인트 클라우드 간의 효과적인 통합을 통해 입증된 성능을 보이며, 자율 주행 및 다양한 컴퓨터 비전 응용 프로그램에서 중요한 기여를 제공합니다.



### DreamOmni2: Multimodal Instruction-based Editing and Generation (https://arxiv.org/abs/2510.06679)
- **What's New**: 최근 instruction 기반 이미지 편집 및 주제 구동 생성에서 중요한 발전이 이루어졌으나, 이 두 작업 모두 사용자의 실질적인 요구를 충족하는 데 한계가 있다. 기존의 editing은 언어 지침만 의존하며 구체적인 편집 세부 사항을 포착하지 못해 참조 이미지가 필요하다. 이에 따라 새로운 작업으로 다중모드 instruction 기반 편집 및 생성을 제안한다.

- **Technical Details**: 제안된 DreamOmni2는 데이터 생성과 모델 프레임워크 디자인을 포함한 두 가지 주요 문제를 해결한다. 데이터 합성 파이프라인은 추출 데이터 생성을 위한 feature mixing 방법을 사용하며, 편집 및 추출 모델을 활용해 다중모드 편집 훈련 데이터를 생성한다. 또한 여러 이미지를 처리하기 위한 인덱스 인코딩과 위치 인코딩 이동 스킴을 도입하여 픽셀 혼란을 방지한다.

- **Performance Highlights**: 실험 결과 DreamOmni2는 인상적인 성과를 달성했다. 이 모델은 실제 사용자 지침을 보다 잘 처리하기 위해 VLM과의 공동 학습을 통해 복잡한 명령을 이해하는 능력을 향상시킨다. 제안된 두 작업을 위한 종합적인 베치마크는 모델의 일반화와 실제 시나리오에서의 성능을 정확히 평가하는 데 기여한다.



### Heptapod: Language Modeling on Visual Signals (https://arxiv.org/abs/2510.06673)
- **What's New**: Heptapod은 언어 모델링의 기초 원칙을 준수하는 이미지 자동 회귀 모델입니다. 이 모델은 causal attention을 활용하고 CFG에 대한 의존성을 제거하며, 의미론적 토크나이저의 추세를 피합니다. 핵심 혁신은 모든 2D 공간 그리드에 대한 분포 예측(next 2D distribution prediction)이며, 이는 생성 훈련을 통해 포괄적인 이미지 의미를 파악하도록 도와줍니다.

- **Technical Details**: Heptapod은 일관된 causal Transformer 구조를 사용하여 시각적 토큰을 생성하고, 기존의 이미지 자동 회귀 모델과는 달리 각 번째 공간 위치에서의 토큰 분포를 병렬로 예측하기 위해 훈련됩니다. 이러한 접근 방식은 모델이 복잡한 공간 종속성과 전체 이미지를 이해하도록 유도합니다. 또한, 자가 지도 학습 관점에서 autoregressive modeling과 Masked Autoencoding(MAE)의 결합된 목표를 제공합니다.

- **Performance Highlights**: 이미지 생성 기준인 ImageNet에서 Heptapod은 FID가 2.70으로, 이전의 자동 회귀 접근법을 월등히 초월하는 성능을 보여줍니다. 이는 기존의 시각적 생성을 위한 모델들이 구조적 개선 없이 외부 개입(CFG)에 의존해야 하는 한계를 극복할 수 있다는 것을 시사합니다. Heptapod은 언어 모델링 원칙이 시각 신호에서도 잘 적용될 것이라는 점에서 새로운 가능성을 엿보게 합니다.



### Automated Neural Architecture Design for Industrial Defect Detection (https://arxiv.org/abs/2510.06669)
- **What's New**: 이 논문에서는 표면 결함 감지(Surface Defect Detection; SDD)를 위한 오토화된 뉴럴 아키텍처 디자인 프레임워크인 AutoNAD를 제안합니다. 이 시스템은 컨볼루션(convolution), 트랜스포머(transformer), 다층 퍼셉트론(MLP)을 조합해 하이브리드 아키텍처를 자동으로 설계하여, 인트라클래스 차이(intraclass difference)와 인터클래스 유사성(interclass similarity) 문제를 효과적으로 해결합니다. AutoNAD는 다양한 결함 유형과 산업 적용 환경에 맞춰 적응적인 네트워크 설계를 가능하게 합니다.

- **Technical Details**: AutoNAD는 교차 가중치 공유(cross weight sharing) 전략을 통해 다양한 연산자 유형 내에서 및 서로 간의 효율적인 가중치 공유를 촉진합니다. 이와 함께 다단계 특징 집합 모듈(Multi-Level Feature Aggregation Module; MFAM)을 통해 멀티 스케일(feature) 학습을 강화하며, 다이렉트 에이시클릭 그래프(directed acyclic graph) 구조를 이용하여 최적의 융합 경로를 동적으로 선택합니다. 또한, 런타임 통계 기반의 레이턴시 인식(latency-aware) 프라이어를 도입해 SDD에서 최적의 아키텍처를 탐색합니다.

- **Performance Highlights**: AutoNAD는 세 가지 산업 결함 데이터셋에서 성능을 검증하였으며, 실제 자동 감지 시스템에 통합되어 높은 검출 정확도와 효율성을 달성했습니다. 이 시스템은 생산 제약 조건에서 실제로 작동할 수 있는 가능성을 입증하며, 빠른 수렴성과 향상된 서브넷 성능을 통해 산업 환경에서의 적용 가능성을 높입니다. 결과적으로 AutoNAD는 기존 수작업 아키텍처 설계 과정의 부담을 줄이고, 더 나가 많은 산업 안정성을 제공할 수 있는 시스템으로 자리 잡고 있습니다.



### StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering (https://arxiv.org/abs/2510.06638)
- **What's New**: 이번 연구는 Knowledge-based Visual Question Answering (KVQA) 의 새로운 변형인 IK-KVQA를 다루고 있습니다. 이 모델은 복합 멀티모달 대형 언어 모델 (MLLM)을 지식 소스로 사용하며, 외부 검색 없이 사실 지식을 바탕으로 질문에 답변합니다. 기존 MLLM들의 한계를 극복하기 위해 구조적 추론 추적 (Structured Reasoning Traces)이라는 새로운 방법론을 도입합니다.

- **Technical Details**: StaR-KVQA는 이중 기호 관계 경로 (dual symbolic relation paths)와 경로 기반 자연어 설명 (path-grounded natural-language explanations)으로 구성된 구조적 추적을 감독합니다. 이를 통해 추론 과정의 투명성과 검증 가능성을 높입니다. 이 방법은 외부 검색자나 지식 기반을 사용하지 않고, 오프라인에서 추적을 구축하며 단일 자기 회귀 (autoregressive) 과정을 통해 추론을 수행합니다.

- **Performance Highlights**: 연구 결과, StaR-KVQA는 OK-VQA 벤치마크에서 가장 강력한 기준선 대비 최대 11.3% 더 높은 답변 정확도를 달성했습니다. 또한, 다양한 도메인에 대한 강력한 일반화 성능을 보여주어, 정확성과 해석 가능성을 모두 향상시켰습니다.



### MSITrack: A Challenging Benchmark for Multispectral Single Object Tracking (https://arxiv.org/abs/2510.06619)
- **What's New**: MSITrack는 현재까지 가장 크고 다양한 다분광(single object) 추적 데이터셋으로, 여러 복잡한 배경과 유사한 객체 간의 간섭 문제를 해결하기 위해 제작되었습니다. 이 데이터셋은 55개의 객체 카테고리와 300개의 자연 장면으로 구성되어 있으며, RGB 데이터만 사용하는 기존의 알고리즘에 비해 성능 개선에 기여할 수 있는 잠재력을 지니고 있습니다. 또한, 모든 프레임은 정밀하게 수작업으로 주석 처리되어 안정적인 평가를 위한 토대를 제공합니다.

- **Technical Details**: MSITrack는 395nm에서 950nm의 스펙트럼 대역을 포함하는 다분광 영상을 담고 있으며, 총 300개의 비디오로 구성되어 있습니다. 각 프레임은 1200 × 900 픽셀의 공간 해상도를 가지며, 주석 처리에는 1,300시간 이상의 노동 시간이 소요되었습니다. 이 데이터셋은 복잡한 상황에서의 RGB만 사용한 방법과 비교하여 더욱 안정적인 물체 추적 가능성을 제공합니다.

- **Performance Highlights**: 시험 결과, 다분광(Multispectral) 입력을 활용한 추적기가 RGB만 사용하는 기준 모델보다 복잡한 상황에서 월등히 높은 성능을 보였습니다. 이러한 결과는 MSITrack이 향후 다분광 추적 분야의 발전을 지원하고 영감을 줄 수 있는 잠재력을 강조합니다. 데이터셋은 공공에 제공되며, 더 많은 연구와 재현성을 촉진하기 위해 모든 소스 코드와 자원도 공개됩니다.



### A Bridge from Audio to Video: Phoneme-Viseme Alignment Allows Every Face to Speak Multiple Languages (https://arxiv.org/abs/2510.06612)
- **What's New**: 이 논문에서는 음성 기반의 생동감 있는 얼굴 애니메이션 생성(Speech-driven talking face synthesis, TFS)을 위한 새로운 프레임워크인 다국어 전문가(Multilingual Experts, MuEx)를 제안합니다. 기존 모델들은 영어에는 잘 작동하지만 비영어 국가에서는 성능이 저조하며, 이는 영어 중심의 학습 데이터로 인한 것입니다. MuEx는 음소(phoneme)와 시스팀(viseme)을 보편적 매개체로 활용하여 오디오와 비디오의 교량 역할을 수행하며, 다양한 언어에서 실제와 같은 TFS를 달성합니다.

- **Technical Details**: MuEx는 음소와 시스팀 간의 강력한 교차 모드 정렬을 위한 음소-시스팀 정렬 매커니즘(Phoneme-Viseme Alignment Mechanism, PV-Align)을 도입합니다. 이 시스템은 오디오와 비디오의 특징을 각각 음소와 시스팀으로 추출하여 언어 중립적인 표현을 구성합니다. 또한, 이 프레임워크는 희소 전문가 라우팅(sparse mixture-of-experts) 구조를 사용하여, 언어 라벨 없이도 다중 언어의 처리 경로를 자동으로 선택할 수 있도록 설계되었습니다.

- **Performance Highlights**: MuEx는 다국어 얼굴 애니메이션 벤치마크(Multilingual Talking Face Benchmark, MTFB)에서 모든 언어에 대해 우수한 성능을 보였으며, 추가 학습 없이도 새로운 언어에 대해 효과적인 '제로샷 제너럴라이제이션(zero-shot generalization)'을 보여줍니다. 12개 다양한 언어로 구성된 MTFB는 95.04시간의 고품질 비디오를 포함하고 있으며, 이는 기존의 영어 중심 모델이 직면하던 한계를 해결할 수 있는 가능성을 보여줍니다.



### Self-supervised Physics-guided Model with Implicit Representation Regularization for Fast MRI Reconstruction (https://arxiv.org/abs/2510.06611)
- **What's New**: 본 논문에서는 UnrollINR이라 명명된 새로운 제로샷(self-supervised) 자기공명영상(MRI) 재구성 프레임워크를 제안합니다. 이는 외부 훈련 데이터 없이도 스캔 특정 MРI 재구성을 가능하게 하며, 물리적으로 안내된 반복 재구성 방법과 암묵 신경 표현(Implicit Neural Representation, INR)을 결합하여 재구성 품질을 향상시킵니다. 이 방법은 고속 MRI 재구성을 위해 설계되었으며, 기존의 감독 학습 방법보다 우수한 성능을 입증했습니다.

- **Technical Details**: UnrollINR은 입력 데이터를 解決하기 위한 새로운 접근 방식을 제시하며, 이는 데이터를 수집하는 과정에서의 물리 원리를 명시적으로 포함하는 신경망 아키텍처를 도입합니다. 구체적으로, 이 방법은 다채널 수신 코일의 감도 맵과 이미지를 공동으로 추정하며, INR의 연속적인 암묵적 표현 능력이 재구성을 위한 효과적인 정규화 역할을 합니다. 또한, 이 모델은 고속 가속을 달성하면서도 안정성 및 해석 가능성을 유지합니다.

- **Performance Highlights**: 실험 결과, UnrollINR는 고속 가속 비율이 10에 달하는 상황에서도 감독 학습 방법보다 뛰어난 재구성 성능을 보였습니다. 본 연구는 공공 데이터셋과 임의로 샘플링된 데이터셋에서 검증을 실시하였으며, UnrollINR이 제안된 방법의 우수성을 확인했습니다. 연구의 주요 기여는 단일 피험자에서 수집된 미수 샘플 데이터만을 사용하여 효과적인 재구성을 가능하게 한점입니다.



### AIM 2025 Challenge on Real-World RAW Image Denoising (https://arxiv.org/abs/2510.06601)
- **What's New**: AIM 2025 Real-World RAW Image Denoising Challenge는 데이터 합성을 기반으로 한 효율적이고 효과적인 노이즈 제거 기술을 발전시키기 위해 소개되었습니다. 이 대회는 다섯 개의 서로 다른 DSLR 카메라로 촬영한 저조도 노이즈 이미지로 구성된 새로운 평가 벤치마크를 기반으로 하고 있습니다. 참가자는 다양한 카메라 모델에서 높은 성능을 달성하기 위해 혁신적인 노이즈 합성 파이프라인과 네트워크 아키텍처를 개발해야 합니다.

- **Technical Details**: 이 대회는 카메라에 의존하지 않는 선진 데이터 합성에 기반한 노이즈 제거 솔루션의 개발을 촉진하기 위해 설계되었습니다. 각 참가자는 소음 모델링 파이프라인과 학습 기반 아키텍처를 제공하여 실세계 장면에서 잘 작동할 뿐만 아니라 다양한 카메라에 일반화될 수 있도록 해야 합니다. 대회 성과는 PSNR, SSIM, LPIPS와 같은 성능 지표와 ARNIQA, TOPIQ와 같은 비참조 지표를 기준으로 평가됩니다.

- **Performance Highlights**: 총 8686팀이 이 대회에 참가하였으며, 9797팀이 최종 테스트 단계에서 유효한 결과를 제출했습니다. 참가자는 다양한 카메라의 노이즈 프로필을 활용하여 자가 감독 학습을 위한 개선된 노이즈 합성 파이프라인과 카메라에 비 의존적인 RAW 이미지 노이즈 제거를 위한 네트워크 설계 및 훈련 전략을 제시하였습니다.



### SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation (https://arxiv.org/abs/2510.06596)
- **What's New**: 본 논문은 Synthetic Dataset Quality Metric (SDQM)을 소개하여, 객체 탐지(object detection) 작업에서 생성된 데이터의 품질을 평가합니다. SDQM은 객체 탐지 모델이 훈련되는 동안 데이터의 효과성을 예측할 수 있는 효율적인 메트릭으로, 최신 AI 기법을 활용하여 실제 데이터의 특성을 잘 반영하도록 설계되었습니다. 이 메트릭은 비용이 많이 드는 반복 훈련의 필요성을 줄여주고 자원의 제약이 있는 상황에서 중요한 도전 과제를 해결하는 데 기여합니다.

- **Technical Details**: SDQM은 객체 탐지 데이터셋의 유용성을 평가하기 위해 여러 독립적인 구성 요소들로 구성됩니다. 이 메트릭은 픽셀 공간, 공간 공간, 주파수 공간, 특징 공간 도메인 간의 갭을 평가하며, 기존 메트릭과 비모수적 분포 비교 기술을 사용하여 데이터 세트의 효과를 정량화합니다. 실험 결과, SDQM은 YOLOv11 모델과의 평균 평균 정밀도(mean Average Precision, mAP) 점수와 강한 상관관계를 보여줍니다.

- **Performance Highlights**: SDQM은 훈련된 모델의 성능과 강하게 연관되어 있으며, 이는 랜덤 플레인스(RarePlanes), 산업 금속 물체 데이터셋(DIMO), WASABI와 같은 다양한 데이터셋에서 일관된 성능 향상으로 입증되었습니다. 이 메트릭은 자원을 효율적으로 활용하고 반복적인 훈련/검증 사이클에 대한 의존성을 줄여주어, 저비용으로 데이터 품질을 개선할 수 있는 방향을 제시합니다.



### Adaptive Stain Normalization for Cross-Domain Medical Histology (https://arxiv.org/abs/2510.06592)
Comments:
          Accepted to the 28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2025)

- **What's New**: 본 논문에서는 기존의 전통적인 염색 색상 정규화 방법의 한계를 극복하기 위해 Beer-Lambert Net (BeerLaNet)을 제안합니다. 이 모델은 다양한 염색 프로토콜에 대해 독립적인 표현을 학습하며, 기존 방법들이 요구하는 사전 지식 없이도 작동할 수 있는 장점이 있습니다. 또한, 물리학에 기반을 둔 데이터 기반 기법으로서, 이미지 처리 과정에서의 염색 분해를 지원합니다.

- **Technical Details**: BeerLaNet의 핵심은 Beer-Lambert 법칙을 기반으로 하여 이미지에서 염색의 상호작용을 모델링한다는 점입니다. 이 모델은 비음수 행렬 분해 (NMF) 기법을 사용하여 염색의 구조적 정보를 추출하고, 이 정보를 후속 처리의 입력으로 활용합니다. 이를 통해 다양한 염색 조건에서 발생하는 색상 불일치를 효과적으로 해결할 수 있습니다.

- **Performance Highlights**: 본 논문에서는 세포 탐지 및 분류 등의 다운스트림 작업을 위해 다양한 공개 병리 데이터셋을 활용하여 BeerLaNet의 성능을 평가했습니다. 실험 결과, 기존의 최첨단 염색 정규화 방법들 대비 월등한 성능 향상을 보였으며, 임상 진단에서의 잠재적인 적용 가능성을 보여줍니다. 코드와 모델은 해당 논문과 함께 제공됩니다.



### Ming-UniVision: Joint Image Understanding and Generation with a Unified Continuous Tokenizer (https://arxiv.org/abs/2510.06590)
Comments:
          Code released at this https URL

- **What's New**: MingTok은 연속 잠재 공간을 가진 새로운 시각적 토크나이저로, 이해와 생성의 통합을 위한 혁신적인 접근 방식을 제안합니다. 기존의 토크나이저는 이산 잠재 공간에서 작동하여 양자화 오류가 생기고 의미적 표현력이 떨어지는 문제를 겪었습니다. MingTok은 이러한 문제를 해결하기 위해 저수준 인코딩, 의미적 확장, 시각적 재구성을 포함하는 세 단계의 아키텍처를 채택하였습니다.

- **Technical Details**: MingTok의 저수준 인코더는 이미지를 compact하게 표현하여 효율적인 오토회귀 생성이 가능하도록 합니다. 이후 의미적 디코더는 이 compact한 표현을 고차원 의미적 특징으로 확장하고, 최종적으로 픽셀 디코더가 이 특징들을 원본 이미지로 재구성합니다. 이러한 구조는 이미지 생성과 이해를 잘 통합하여 동시 최적화를 지원합니다.

- **Performance Highlights**: Ming-UniVision은 다양한 비전-언어 작업을 통합하여 단일 오토회귀 프레임워크에서 다양한 가능성을 제공합니다. 이 모델은 기존의 다중 인코더 디자인에 비해 상당히 낮은 구조적 복잡성을 달성하며, 66% 적은 입력 토큰을 요구하여 더 빠른 반복과 메모리 효율성을 제공합니다. 이로 인해 동적이고 상호작용이 가능한 비전 시스템 개발에 기여할 것으로 기대됩니다.



### Improving Artifact Robustness for CT Deep Learning Models Without Labeled Artifact Images via Domain Adaptation (https://arxiv.org/abs/2510.06584)
Comments:
          8 pages, 12 figures, 1 table

- **What's New**: 이 연구는 새로운 이상(artifact)이 포함된 CT 스캔 데이터에 대해 분류 성능을 유지하면서 라벨(Labels) 없이 모델을 훈련시키기 위한 도메인 적응(Domain Adaptation)을 평가합니다. 기존의 훈련 세트에 없던 이상이 나타나는 경우, 모델의 분류 정확도가 크게 저하될 수 있음을 지적하고, 이를 해결하기 위한 새로운 접근법의 필요성을 강조합니다. 또한, 도메인 적응 기술을 통해 라벨링 비용을 줄이면서도 모델의 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 시뮬레이션된 링 아티팩트를 CT 이미지의 왜곡으로 사용하여 도메인 적대 신경망(Domain Adversarial Neural Networks, DANN)을 다른 접근방법과 비교했습니다. 기존의 깨끗한 이미지만으로 훈련된 모델은 링 아티팩트가 있는 이미지에 대해 일반화할 수 없음을 보여줍니다. DANN 접근법은 라벨이 없는 아티팩트 데이터를 사용하여 높은 분류 정확도를 유지하며, 이는 의료영상에서 발생하는 분포 이동(distribution shift)에 효과적으로 대처할 수 있는 가능성을 나타냅니다.

- **Performance Highlights**: DANN 알고리즘을 사용한 도메인 적응 모델은 라벨이 있는 아티팩트 이미지로 훈련된 모델과 유사한 성능을 보였습니다. 또한, DANN은 균일 잡음(uniform noise)과 같은 보기 드문 입력에서도 좋은 일반화 성능을 보여주며, 이는 실제 임상 환경에서도 신뢰할 수 있는 성능을 제공할 수 있음을 시사합니다. 궁극적으로, 도메인 적응이 새로운 이상 분포에 대한 라벨링 비용 없이도 의료 영상 분야에서 유용할 수 있음을 보여주는 실증적 증거를 제공합니다.



### Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation (https://arxiv.org/abs/2510.06582)
- **What's New**: 본 연구에서는 정확한 지상 레이저 스캐닝(TLS) 포인트 클라우드의 의미론적 분할을 위해 반자동화된 불확실성 인식 파이프라인을 제안합니다. 이 파이프라인은 구면 프로젝션(spherical projection)과 기능 강화(feature enrichment), 앙상블 학습(ensemble learning), 목표 지향적 주석(annotation)을 통합하여 레이블링 노력을 줄이면서도 높은 정확도를 유지합니다. 또한, Mangrove3D라는 맹그로브 숲을 위한 의미론적 분할 TLS 데이터셋을 구축했습니다.

- **Technical Details**: 제안된 방법은 3D 포인트를 2D 구형 그리드로 투영하고, 다양한 출처로부터 기능을 강화시켜 픽셀을 풍부하게 하며, 앙상블 분할 네트워크를 훈련시켜 유사 레이블과 불확실성 맵을 생성합니다. 이러한 불확실성 맵은 모호한 영역의 주석을 안내하며, 2D 출력을 3D로 다시 투영하여 고밀도 주석이 달린 포인트 클라우드를 생산합니다. 연구의 핵심 구성 요소는 불확실성 분석을 통합하여 낮은 모델 신뢰도를 가진 영역을 목표로 한 주석을 가능하게 하는 것이고, 이는 정확한 생태 모니터링을 위한 데이터셋의 효율성을 높입니다.

- **Performance Highlights**: 본 연구에서는 주석이 추가되고 나서 약 12개의 스캔에서 성능이 포화 상태에 이르고, 기하학적 특징이 가장 큰 기여를 하는 것으로 나타났습니다. 아홉 개의 채널로 구성된 압축 스택이 거의 모든 차별화 능력을 포착하며, 평균 교차점 비율(mIoU)은 약 0.76에서 안정화되는 것으로 확인되었습니다. 또한, 다양한 데이터셋에서의 테스트를 통해 제안된 기능 강화 전략의 일반화를 확인했습니다.



### HSNet: Heterogeneous Subgraph Network for Single Image Super-resolution (https://arxiv.org/abs/2510.06564)
- **What's New**: 본 논문에서는 Heterogeneous Subgraph Network (HSNet)라는 새로운 프레임워크를 제안합니다. HSNet은 효율적인 그래프 모델링을 활용하면서도 계산 효율성을 유지합니다. 기존의 CNN 및 attention 기반 방법들이 가진 구조적 비유연성을 해결하기 위해, 이미지를 관리 가능한 하위 구성 요소로 분해하는 접근 방식을 사용합니다.

- **Technical Details**: HSNet의 핵심 구성 요소는 Constructive Subgraph Set Block (CSSB)과 Subgraph Aggregation Block (SAB)입니다. CSSB는 서로 다른 관계 패턴과 특징 상호작용을 모델링하여 다양한 보완적 하위 그래프를 생성합니다. SAB는 이들 하위 그래프에서 추출된 표현을 통합하여 복잡한 상호 의존성을 포착하는 포괄적이고 차별화된 표현을 구축합니다.

- **Performance Highlights**: 광범위한 실험을 통해 HSNet은 다섯 개의 SISR 벤치마크에서 최신 성능을 달성함을 보여줍니다. 제안된 구성 요소의 효과성을 검증하기 위한 ablation 연구 및 시각적 분석이 포함되어 있습니다. HSNet은 재구성 품질과 계산 효율성 간의 균형을 효과적으로 유지하며, 딥러닝 기반 이미지 처리 분야에 기여할 것으로 기대됩니다.



### Cluster Paths: Navigating Interpretability in Neural Networks (https://arxiv.org/abs/2510.06541)
- **What's New**: 이번 연구에서는 딥러닝 모델의 해석 가능성을 높이기 위한 새로운 방법인 cluster paths를 제안합니다. 이 방법은 훈련된 신경망의 특정 레이어에서 클러스터링을 통해 활성화 패턴을 그룹화하고, 각 입력을 클러스터 ID의 시퀀스로 표현함으로써 모델의 내부 의사결정 과정을 시각화합니다. 기존의 예제 및 그래디언트 기반 접근법과 비교했을 때, cluster paths는 네트워크가 샘플을 변환하는 과정을 간결하게 요약하여 그 과정을 이해하고 시각화할 수 있게 합니다.

- **Technical Details**: 이 방법은 각 레이어의 활성화를 클러스터링하여 클러스터 ID 시퀀스로 입력을 인코딩하며, 이를 통해 두 입력이 같은 클러스터 시퀀스를 따라간다면 네트워크는 이를 유사하게 처리할 것이라는 가정을 세웁니다. 클러스터 경로는 모델의 내부 논리를 요약하는 압축된 프록시로 기능하며, 복잡한 모델을 수십 개의 클러스터로 단순화할 수 있습니다. 이 연구에서는 path complexity, weighted-path purity, decision-alignment faithfulness, path agreement 등 네 가지 새로운 메트릭을 도입하여 클러스터 경로를 평가합니다.

- **Performance Highlights**: 실험 결과, cluster paths는 CIFAR-10에서 색상 기반 단서를 식별하고, 해당 단서가 제거되면 경로가 붕괴됨을 보여주었습니다. CelebA의 경우 90%의 신뢰도와 Gaussian 노이즈 하에서도 96%의 일치를 유지하면서 정확도를 희생하지 않았습니다. 이 방법은 Vision Transformer와 같은 대형 모델에도 확장 가능하며, 시각적 개념을 여러 네트워크 깊이에서 발견하는 데 효과적입니다.



### VUGEN: Visual Understanding priors for GENeration (https://arxiv.org/abs/2510.06529)
- **What's New**: 최근 비전-언어 모델( Vision-Language Models, VLMs)의 발전이 텍스트와 이미지 간의 통합적인 이해를 가능하게 했습니다. 하지만 이러한 모델에 이미지 생성 기능을 강화하는 것은 여전히 도전 과제가 남아있습니다. 본 연구에서는 VUGEN이라는 새로운 프레임워크를 제안하며, 이는 VLM의 사전 학습된 시각적 이해 Priors를 활용한 효과적이고 고품질의 이미지 생성을 목표로 하고 있습니다. 이 접근 방식은 고차원 잠재공간을 저차원으로 변환하여 시각 정보를 최대한 보존하며, VLM이 이 축소된 잠재공간에서 샘플링하도록 학습합니다.

- **Technical Details**: VUGEN의 핵심은 비전 인코더의 잠재공간에서의 샘플링을 통한 이미지 생성입니다. 이 기술은 VLM의 사전 학습된 시각적 Priors를 완전히 활용할 수 있도록 하며, 기존의 해체된 토크나이저로 인해 발생하는 불일치를 피합니다. 또한, 이미지 디코더는 생선된 잠재벡터를 이미지 공간으로 다시 매핑하는 역할을 합니다. 실험 결과, VAE 없는 픽셀 확산 디코더가 일반적으로 사용되는 복잡한 잠재 확산 디코더보다 성능이 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: VUGEN은 DPG Bench에서 71.17에서 74.32로 개선되었으며, COCO 데이터셋에서 FID를 11.86에서 9.06로 낮추어 이미지 생성 성능을 크게 향상시켰습니다. 본 연구의 결과는 VLM의 원래 이해 능력을 전적으로 보존하면서 이미지 생성 품질을 높였음을 보여줍니다. 또한, VUGEN은 다양한 규모의 두 데이터셋에서 질적 및 양적 성능을 모두 개선하였습니다.



### Limited-Angle Tomography Reconstruction via Projector Guided 3D Diffusion (https://arxiv.org/abs/2510.06516)
Comments:
          10 pages, 11 figures

- **What's New**: 본 논문은 Limited-angle electron tomography의 한계점을 극복하고자 TEMDiff라는 새로운 3D diffusion 기반의 반복 재구성(framework)를 제안합니다. 기존의 Deep learning 접근법은 고품질의 훈련 데이터셋이 필요하지만 TEMDiff는 readily available volumetric FIB-SEM 데이터를 사용하여 시뮬레이터를 통해 TEM tilt series에 매핑하여 구조적 priors를 학습합니다. 이 방식은 특히 10도의 제한된 각도를 가진 시뮬레이션 데이터셋에서 기존의 스테이트 오브 아트 방법들보다 뛰어난 재구성 품질을 보여줍니다.

- **Technical Details**: TEMDiff는 3D volumetric data를 직접 학습하여 slice 간의 상관관계를 포착하고 cross-slice 일관성을 유지합니다. 이를 위해 projection-based correction 전략을 채택하여 denoising 각 단계에서 데이터 피델리티를 간단하고 효율적으로 보장합니다. TEMDiff는 기존의 filtered backprojection 및 iterative reconstruction 방법보다 뛰어난 결과를 제공하며, 실험 조건에 구애받지 않고 실제 TEM 기울기에서 구조를 얼마나 잘 복구할 수 있는지를 보여줍니다.

- **Performance Highlights**: 제안된 TEMDiff 모델은 60도에서 100도 사이의 전통적인 방법들과 비교했을 때 10도 이하의 각도 범위에서도 훌륭한 결과를 도출합니다. 이 모델은 자동 조정이나 재학습 없이도 다양한 조건에서도 일반화가 잘 이루어지며, 특히 8도 범위 내에서도 정확한 구조를 복원할 수 있습니다. TEMDiff는 한 번의 훈련으로도 우수한 재구성 품질을 달성할 수 있음을 입증합니다.



### LogSTOP: Temporal Scores over Prediction Sequences for Matching and Retrieva (https://arxiv.org/abs/2510.06512)
- **What's New**: 이 논문에서는 객체 및 감정과 같은 로컬 속성의 점수(score)를 시퀀스의 시간적 속성에 맞게 변환하는 문제를 정식화합니다. 이를 위해 'Scores for TempOral Properties (STOPs)'라는 개념을 도입하였고, 이를 계산하는 새로운 점수 함수인 LogSTOP을 제안합니다. LogSTOP은 Linear Temporal Logic을 사용하여 시간적 속성을 효율적으로 계산할 수 있으며, 이를 통해 다양한 애플리케이션에서 유용하게 활용될 수 있습니다.

- **Technical Details**: LogSTOP은 시간 복잡도 (T⋅|φ|)을 가지며, 이는 기존의 방법보다 월등히 효율적입니다. 또한, LogSTOP은 로컬 속성 예측기의 부정확한 예측을 처리하기 위한 다운샘플링과 스무딩 전략을 사용하여 강건성을 향상시킵니다. Linear Temporal Logic의 다양한 연산자를 사용하여 복잡한 시간적 속성을 표현할 수 있으며, 이는 시퀀스 데이터의 다양한 구성 요소에 적용 가능합니다.

- **Performance Highlights**: LogSTOP은 YOLO 및 HuBERT와 함께 사용하여 쿼리 매칭에서 기존 모델보다 최소 16% 성능 향상을 보였습니다. 또한, Grounding DINO와 SlowR50을 사용하는 경우 시간적 속성에 대한 랭크된 검색에서도 평균 19% 및 16%의 향상을 기록하였습니다. 다양한 시퀀스와 속성에 적용 가능한 QMTP 및 TP2VR이라는 두 개의 새로운 벤치마크도 제안되었습니다.



### From Captions to Keyframes: Efficient Video Summarization via Caption- and Context-Aware Frame Scoring (https://arxiv.org/abs/2510.06509)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문에서는 긴 비디오에서 의미론적(semantic) 및 맥락적(contextual) 정보를 유지하면서 적은 수의 프레임을 선택하기 위한 새로운 다중 모달 프레임 점수 매기기 프레임워크인 KeyScore를 제안합니다. KeyScore는 자막(captions)과 비주얼 컨텍스트(visual context)를 결합하여 프레임의 중요성을 평가하고, STACFP(Spatio-Temporal Adaptive Clustering for Frame Proposals)를 도입하여 긴 비디오에 대해 컴팩트하고 다양한 프레임 후보를 생성합니다. 이들 모듈은 전체 프레임 추론 대비 최대 99%의 프레임 감소를 달성하고, MSRVTT 및 MSVD 등 다양한 데이터셋에서 기존의 8프레임 인코더보다 뛰어난 성능을 보였습니다.

- **Technical Details**: KeyScore는 세 가지 상호 보완적인 신호를 결합하여 의미 있는 프레임을 선정합니다: (1) 프레임과 자막 간의 의미 유사도(semantic similarity), (2) 비디오 시간선의 커버리지를 보장하는 시간적 대표성(temporal representativeness), (3) 중복성과 다양성을 고려한 컨텍스트 드롭 임팩트(contextual drop impact). STACFP는 클러스터링 기법에 시간 인코딩을 결합하여 동적인 영역에 더 많은 제안을 할당하고, 정적 구간에서는 중복을 피하는 방식으로 최적의 클러스터 수를 자동으로 선택합니다.

- **Performance Highlights**: KeyScore는 비디오-언어 테스크에서 일관되게 uniform sampling 및 클러스터링 기반 벤치마크를 능가하는 성능을 보였으며, 정확도를 향상시키면서 원본 비디오 대비 프레임 사용량을 97-99%까지 줄였습니다. 이와 함께 KeyScore는 수작업 어노테이션 없이 대규모 비디오-자막 데이터셋에 적용될 수 있는 유연한 프레임워크를 제공하며, 새로운 평가 패러다임을 가능하게 합니다. 이 논문은 자막-aware 프레임 점수가 콘텐츠 효율적인 비디오 이해를 위한 강력한 도구임을 보여줍니다.



### Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation (https://arxiv.org/abs/2510.06504)
- **What's New**: 본 논문에서는 두 사람 간의 상호작용 모델링의 한계를 극복하기 위한 새로운 프레임워크인 Text2Interact를 제안합니다. 이 프레임워크는 스케일러블한 고충실도(interaction data synthesizer) 데이터 생성기와 효과적인 시공간(spatiotemporal) 조정 파이프라인을 통해 자연어 텍스트에 정렬된 인간 간의 상호작용을 생성합니다. 기존 방법들이 데이터 제약과 함께 언어 조건이 비효율적이라는 문제를 지적하며, InterCompose와 InterActor라는 두 가지 핵심 구성 요소를 소개합니다.

- **Technical Details**: InterCompose는 LLM(large language model) 생성 상호작용 설명을 강력한 단일 개인 모션 프리미엄(single-person motion priors)과 정렬시키는 합성-구성(synthesis-by-composition) 파이프라인입니다. 이 시스템은 단일 개인 모션 후보를 검색하고, 다른 에이전트를 위한 조건 반응 생성기를 훈련시키며, 약하거나 잘못 정렬된 샘플을 필터링하는 신경 모션 평가기를 사용하여 상호작용 범위를 확장합니다. InterActor는 단어 수준 조건부(word-level conditioning)와 적응형 상호작용 손실(adaptive interaction loss)을 통해 텍스트와의 일관성을 유지하며, 세밀한 상호작용 모델링을 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 두 사람의 모션 생성에서 최첨단 성능을 나타내며, InterActor는 모션의 충실도와 텍스트에 대한 충성도, 일반화 가능성에서 이전 연구를 능가하는 것으로 확인됩니다. 특히, 실제 상호작용 데이터가 부족한 상황에서도 높은 성능을 발휘합니다. 사용자 선호도 연구를 통해 생성 품질 향상과 텍스트 정렬 개선을 평가하였고, 정량적 메트릭에서 감지되지 않은 일반화 개선을 측정했습니다.



### Superpixel Integrated Grids for Fast Image Segmentation (https://arxiv.org/abs/2510.06487)
- **What's New**: 이 논문에서는 이미지 세분화(segmentation) 작업에서의 전체 해상도 이미지 대신에 사용할 새로운 슈퍼픽셀 기반 데이터 구조인 SIGRID(Superpixel-Integrated Grid)를 도입합니다. SIGRID는 전통적인 형태 설명자(shape descriptors)를 활용하여 슈퍼픽셀의 색상 및 형태 정보를 인코딩하며, 입력 차원(dimensionality)을 대폭 줄입니다. 우리의 연구는 기존의 픽셀 수준 표현과 비교했을 때, SIGRID가 데이터 압축에도 불구하고 성능을 동등하게 유지하거나 경우에 따라 초월함을 보여줍니다.

- **Technical Details**: SIGRID는 인접한 픽셀을 그룹화하여 의미 있는 세그먼트로 변환함으로써 이미지의 불규칙한 슈퍼픽셀 구조를 정리된 격자형(Grid) 구조로 재조정합니다. 이 방식은 주어진 슈퍼픽셀의 모양과 외관을 기반으로 색상 데이터와 도형 정보를 축약된 형태로 유지합니다. 이를 통해 CNN과 같은 전통적인 신경망 아키텍처에서 효율적으로 처리할 수 있도록 만들어집니다. 각 슈퍼픽셀에 대한 레이블은 그 경계 내에서 다수의 픽셀 레이블을 기반으로 지정됩니다.

- **Performance Highlights**: SIGRID는 U-Net 및 완전 합성곱 신경망(FCN)과 같은 두 가지 인기 있는 합성곱 기반 아키텍처에서 테스트되었으며, 압축된 입력을 사용하더라도 성능이 우수함을 입증했습니다. SIGRID를 사용할 때는 총량을 줄였음에도 불구하고, 메모리 사용과 연산 비용의 절감을 유지하면서도 모델 훈련이 보다 빨라지는 효과를 볼 수 있었습니다. 또한, 기존의 다수의 신경망 구조에 대해 보다 효율적으로 통합될 수 있는 가능성을 제공합니다.



### SIGMA-GEN: Structure and Identity Guided Multi-subject Assembly for Image Generation (https://arxiv.org/abs/2510.06469)
Comments:
          Webpage: this https URL

- **What's New**: SIGMA-GEN는 다중 정체성을 유지하며 이미지를 생성할 수 있는 통합 프레임워크입니다. 기존의 접근 방식과 달리, SIGMA-GEN은 공간적 제약과 구조적 제약에 의해 안내되는 단일 패스로 다중 주체의 정체성을 보존하는 이미지를 생성하는 최초의 방법입니다. 이 방법은 2D 또는 3D 박스와 같은 대략적인 입력에서 픽셀 단위 세분화 및 깊이에 이르기까지 다양한 수준의 사용자의 지침을 지원할 수 있는 강력한 기능을 가지고 있습니다.

- **Technical Details**: SIGMA-GEN은 27K 이미지를 포함하고 100,000개 이상의 고유한 주체에 대한 정체성, 구조 및 공간 정보를 제공하는 새로운 합성 데이터 세트인 SIGMA-SET27K를 도입합니다. 이 모델은 사용자가 특정 요구에 따라 다양한 수준의 구조적 입력을 지원하도록 설계되었으며, 이는 2D 마스크 및 3D 바운딩 박스와 실시간 깊이 맵을 포함합니다. 우리의 방법은 구조적 안내와 정체성 안내를 단일 확산( diffusion) 프로세스 내에서 지원합니다.

- **Performance Highlights**: 모델의 성능 평가 결과, SIGMA-GEN은 정체성 보존 및 이미지 생성 품질 면에서 최첨단의 성능을 달성하였습니다. 픽셀 단위 깊이 및 마스크를 사용할 경우 전체 이미지 충실도는 31점 향상되었고, 정체성 보존은 2점 개선되었습니다. 이 모델은 5개 이상의 별개의 주체를 가진 장면을 합성할 때 4배 빠른 생성 속도를 달성하며, 구조적 제어의 투박한 모드인 바운딩 박스를 사용할 때는 6점 향상된 이미지 충실도와 11점 향상된 정체성 보존 점수를 보였습니다.



### TDiff: Thermal Plug-And-Play Prior with Patch-Based Diffusion (https://arxiv.org/abs/2510.06460)
- **What's New**: 이번 논문에서는 저렴한 열화상 카메라의 저해상도 문제를 해결하기 위해 패치 기반의 확산 프레임워크(TDiff)를 제안합니다. 기존의 열화상 이미지 복원 기술의 한계를 극복하기 위해, 작은 열 패치를 이용한 학습을 통해 지역적 잡음 및 왜곡을 효과적으로 모형화합니다. 이 방법은 여러 복원 작업을 위해 설계된 최초의 패치 기반 확산 프레임워크로, 중첩된 패치를 덴오징하고 부드러운 공간 윈도우를 통해 결합하여 고해상도 이미지를 복원합니다.

- **Technical Details**: TDiff는 열화상 이미지의 복원에 국한된 패치 기반 전략을 채택하여, 각 패치에서 국소적 구조를 학습합니다. 제안된 프레임워크의 핵심은 모든 입력 이미지를 여러 개의 중첩된 패치로 나눈 후 각각 독립적으로 처리하고, 결과를 재조합하는 과정을 포함합니다. 이 과정은 패치 간의 경계에서 발생할 수 있는 아티팩트를 줄이기 위한 것으로, 공간적으로 일관된 복원을 달성하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과, TDiff는 덴오징, 슈퍼 해상도, 디블러링 작업에 대해 일관성 있는 성능 향상을 보여주었습니다. 이 방법은 데이터 부족 문제에도 불구하고 경쟁력 있는 결과를 달성하였으며, 실제로 소비자용 열화상 카메라에서도 강력한 일반화 능력을 보여주고 있습니다. 패치 기반의 접근법을 통해 전반적으로 통합된 복원 파이프라인으로 자리 잡았습니다.



### Road Surface Condition Detection with Machine Learning using New York State Department of Transportation Camera Images and Weather Forecast Data (https://arxiv.org/abs/2510.06440)
- **What's New**: 이번 연구에서는 뉴욕주 교통부(NYSDOT)의 도로 상태를 평가하기 위해 기계 학습 모델을 활용하는 새로운 접근 방식을 제시합니다. 기계 학습 모델은 도로 표면 상태를 자동으로 분류하여 겨울철 폭풍과 같은 기상 상황에서의 운영 결정을 지원합니다. 특별히, 이 연구에서는 약 22,000개의 카메라 이미지를 활용하여 도로 조건을 심층 신경망(convolutional neural networks)과 랜덤 포레스트(random forests)로 예측합니다.

- **Technical Details**: 연구는 22,000개의 수동으로 라벨링된 이미지 데이터셋을 활용하여 도로 조건을 여섯 가지 상태로 분류하는 모델을 훈련시킵니다. 훈련된 모델은 NYSDOT의 운영 필요를 충족시키기 위해 일반화 가능성을 중시하며, 완전히 새로운 카메라에 대해 81.5%의 정확성을 달성했습니다. 이를 통해 기계 학습을 통해 도로 상태의 빠르고 정밀한 분석이 가능해집니다.

- **Performance Highlights**: 모델은 이미지와 기상 데이터를 통합하여 도로 표면 상태를 분류하는 데 성공적이었습니다. 특히, 완전히 새로운 카메라에서 81.5%의 정확도를 기록하였고, 이는 기상 관련 클래스에 대한 효과적인 예측을 가능하게 합니다. 연구는 최종 사용자와 협업을 통해 시행되었으며, 운영적 적용 가능성을 위해 모델의 일반화 가능성도 우선시 되었습니다.



### TransFIRA: Transfer Learning for Face Image Recognizability Assessmen (https://arxiv.org/abs/2510.06353)
Comments:
          Project Page: this https URL

- **What's New**: TransFIRA(Transfer Learning for Face Image Recognizability Assessment)는 신뢰할 수 있는 얼굴 인식 성능을 위해 새로운 접근 방식을 제공합니다. 기존의 FIQA( 얼굴 이미지 품질 평가) 접근법들이 휴먼 주석이나 복잡한 컴퓨팅 파이프라인에 의존하는 것과 달리, TransFIRA는 임베딩 공간에서 직접적으로 인식 가능성을 정의합니다. 이 프레임워크는 근본적으로 결정 경계와 정렬된 인식 가능성 기준을 제공하여 품질 평가의 정확성을 향상시키고, 일반적인 얼굴 인식 및 신체 인식에 모두 적용 가능합니다.

- **Technical Details**: TransFIRA는 세 가지 주요 발전을 포함합니다: (i) 클래스 중심 유사성(class-center similarity, CCS)과 클래스 중심 각도 분리(class-center angular separation, CCAS)를 통한 인식 가능성 정의, (ii) 외부 레이블이 없는 매우 정확한 검증 성능을 위한 인식 가능성 정보를 사용하는 집계 전략, (iii) 인식 가능성의 맥락에서 신체 인식을 평가하기 위한 새로운 확장을 제공합니다. 이 프레임워크는 모든 사전 훈련된 인코더와 호환되며, 별도의 기계 학습 과정 없이 인식 가능성을 예측할 수 있도록 설계되었습니다.

- **Performance Highlights**: TransFIRA는 BRIAR와 IJB-C 데이터세트에서 최신 FIQA 방법보다 뛰어난 성능을 기록했습니다. 실험은 얼굴 인식에 대한 최첨단 결과를 확인하였을 뿐만 아니라 신체 인식에 대한 강력한 성능도 입증했습니다. 또한, 모델은 다양한 데이터셋 간에 강건성을 유지하면서 인식 가능성 예측의 투명성을 제공하며, 인식 성능 저하의 원인인 흐림이나 가림 효과를 설명할 수 있는 능력을 갖추고 있습니다.



### Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding (https://arxiv.org/abs/2510.06308)
Comments:
          33 pages, 13 figures, 10 tables

- **What's New**: Lumina-DiMOO는 다양한 모달리티에 걸쳐 원활한 생성 및 이해를 위한 오픈 소스 기본 모델로 소개되었습니다. 기존의 통합 모델과 다른 점은 각기 다른 입력과 출력을 처리하기 위해 완전한 이산 확산 모델링을 사용하고 있다는 것입니다. 이 혁신적인 접근법은 Lumina-DiMOO가 이전의 오토회귀(autoregressive) 또는 하이브리드 AR-디퓨전(hybrid AR-Diffusion) 패러다임보다 높은 샘플링 효율성을 달성할 수 있도록 합니다.

- **Technical Details**: Lumina-DiMOO는 텍스트-이미지 생성, 이미지 편집, 스타일 전송, 주제 기반 생성, 밀도 예측(dense prediction)과 같은 다양한 멀티모달 작업을 지원하는 다재다능한 모델입니다. 기존의 아키텍처에 비해 추론 속도를 크게 향상시킬 수 있었다는 점이 특징입니다. 예를 들어, Lumina-mGPT 2.0과 비교하여 텍스트-이미지 생성에서 32배의 속도 향상을 달성했습니다.

- **Performance Highlights**: Lumina-DiMOO는 여러 멀티모달 생성 및 이해 벤치마크에서 최상위 성능을 달성하며, 오픈 소스 통합 멀티모달 모델들 중에서 새로운 기준을 세웠습니다. 특히, UniGenBench 리더보드에서 첫 번째 자리에 올라 기존 모델들을 초월하는 성과를 보였습니다. 이러한 결과는 Lumina-DiMOO가 다목적 멀티모달 지능의 연구와 응용에 강력한 기반 모델로 자리 잡게 할 것입니다.



### Scalable deep fusion of spaceborne lidar and synthetic aperture radar for global forest structural complexity mapping (https://arxiv.org/abs/2510.06299)
- **What's New**: 이번 논문에서는 Global Ecosystem Dynamics Investigation (GEDI)에서 제공하는 공간 기반 lidar 데이터를 활용하여, 열대 및 온대 숲의 구조적 복잡성을 고해상도(25 m)로 매핑하는 새로운 딥러닝 프레임워크를 제시합니다. 이 연구의 핵심은 SAR(Synthetic Aperture Radar) 데이터와 GEDI 관측치를 융합하여 연속적이고 높은 해상도의 숲 구조 복잡성 지도를 생성하는 것입니다. 이렇게 개발된 모델은 연구자들이 특정한 컴퓨팅 인프라 없이도 대규모 데이터셋을 처리할 수 있는 접근 가능한 도구로 자리잡을 수 있습니다.

- **Technical Details**: 이 논문은 EfficientNetV2 아키텍처를 기반으로 하여, 1억 3천만 개 이상의 GEDI 발자국에 대해 훈련되었습니다. 이 모델은 40만 개 미만의 파라미터로도 뛰어난 성능(global R2 = 0.82)을 보여줍니다. 또한 모델은 바이옴(biome)과 시간대에 걸쳐 고유한 공간 패턴을 유지하면서 정확한 예측과 보정된 불확실성 추정을 제공합니다.

- **Performance Highlights**: 2015년부터 2022년까지의 숲 구조 복잡성에 대한 글로벌 다중 시계열(multi-temporal) 데이터셋을 생성하는 데 사용되었습니다. 전이 학습(transfer learning)을 통해 이 프레임워크는 추가적인 숲 구조 변수를 예측할 수 있게 확장될 수 있으며, 최소한의 컴퓨팅 비용으로 운영 가능합니다. 이러한 접근 방식은 변화하는 기후 속에서의 글로벌 숲 구조 동태를 지속적으로 모니터링하고 생물 다양성 보존 및 생태계 관리에 중요한 도구를 제공합니다.



### RGBD Gaze Tracking Using Transformer for Feature Fusion (https://arxiv.org/abs/2510.06298)
Comments:
          Master Thesis with 125 pages, 59 figures, 17 tables

- **What's New**: 이번 연구에서는 RGBD(색상 및 깊이 정보를 포함한) 이미지를 사용하는 AI 기반 시선 추적 시스템을 구현하였습니다. Transformer 구조를 사용하여 이미지에서 추출된 특징들을 융합하는 모듈이 도입되었으며, RGBD 입력 이미지와 Transformer의 조합은 이전에 연구되지 않았습니다. 또한, 기존 데이터셋의 한계를 극복하기 위해 새로운 데이터셋이 생성되었습니다.

- **Technical Details**: 이 논문의 AI 모델 아키텍처는 Lian et al.의 이전 연구를 기반으로 하며, Generative Adversarial Network (GAN)을 사용하여 깊이 맵 아티팩트를 제거하고 머리 포즈 특징을 동시에 추출합니다. RGBD 이미지를 사용하는 주목할 만한 점은, 기존 데이터셋들은 깊이 정보를 포함하지 않거나 시선 각도 추정에 적합하지 않은 레이블만 갖고 있기 때문에 이번 연구가 유의미하다는 점입니다. 다양한 모델 구성으로 세 개의 데이터셋에서 훈련, 검증 및 평가가 이루어졌습니다.

- **Performance Highlights**: Transformer 모듈을 사용하는 모델은 ShanghaiTechGaze+ 데이터셋에서 평균 유클리드 오차가 55.3mm였으며, 프리트레인된 GAN 모듈을 사용하지 않았을 경우에는 30.1mm로 줄어들었습니다. Multilayer Perceptron (MLP)으로 Transformer 모듈을 대체하자 평균 오차가 26.9mm로 개선되었습니다. ETH-XGaze 데이터셋에서는 Transformer 모듈을 쓴 모델이 평균 각도 오차 3.59°를 달성하였고, Zhang et al.의 다른 모델은 평균 각도 오차 2.04°를 기록하였습니다.



### Efficient High-Resolution Image Editing with Hallucination-Aware Loss and Adaptive Tiling (https://arxiv.org/abs/2510.06295)
Comments:
          Preprint. Under review

- **What's New**: 최근 모바일 애플리케이션에서 고해상도(4K) 이미지 생성(photo-to-image synthesis)의 중요성이 증가하고 있습니다. 본 논문에서는 리소스 제약이 있는 디바이스에서 메모리와 이미지 품질 문제를 해결하는 새로운 시스템인 MobilePicasso를 제안합니다. 이 시스템은 세 단계에서 이미지 편집을 수행하여 효율성을 극대화하는 동시에 비용과 메모리 사용을 최소화합니다.

- **Technical Details**: MobilePicasso는 세 가지 주요 단계를 포함합니다: 표준 해상도에서 이미지 편집을 수행하는 hallucination-aware 손실, 픽셀 공간으로의 이동 문제를 해결하는 latent projection, 마지막으로 변환된 이미지를 고해상도로 확장하는 adaptive context-preserving tiling을 사용합니다. 이를 통해 전반적인 이미지 품질을 18-48% 향상시키고 환각(hallucination)을 14-51% 감소시키는 효과를 보여줍니다.

- **Performance Highlights**: MobilePicasso는 낮은 지연 시간(latency)과 높은 성능을 자랑하며, A100 GPU에서 실행되는 기존 서버 기반 모델보다 빠른 속도를 기록했습니다. 특히, 시스템의 런타임 메모리 사용량은 9% 증가에 그쳤으며, 55.8배까지 속도를 개선했습니다. 이러한 결과는 MobilePicasso가 실제 적용 가능성이 높은 모델임을 입증합니다.



### ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations (https://arxiv.org/abs/2510.06292)
- **What's New**: 이번 논문에서 제안한 ChainMPQ(Multi-Perspective Questions guided Interleaved Chain of Image and Text)는 LVLMs(Large Vision-Language Models)의 관계 인퍼런스에서 발생하는 환각을 개선하기 위해 고안된 트레이닝 없이 사용할 수 있는 방법론입니다. 기존의 방법들이 관계 환각을 단일 단계 추론으로 취급하는 반면, ChainMPQ는 시각적 및 텍스트 기억을 체계적으로 사용하여 다단계 추론 프로세스를 구현합니다. 이 접근 방식은 인간의 관계 추론 과정에서 영감을 받아, 주체, 객체 및 이 둘을 연결하는 관계라는 세 가지 핵심 요소에 중점을 둡니다.

- **Technical Details**: ChainMPQ는 관계를 주제와 객체 키워드로 분리하여 이미지의 관련 영역을 강화하며, 이를 통해관계를 구성하는 세 가지 요소에 집중하는 다각도의 질문을 생성합니다. 각 질문은 이전 단계에서 얻은 텍스트와 비주얼 메모리를 활용하여 순차적으로 입력됩니다. 이러한 방법은 각 관계를 적절히 추론할 수 있도록 돕고, 신뢰할 수 있는 결과를 제공하기 위해 비주얼 주의와 질문 분해를 통합합니다.

- **Performance Highlights**: 퍼포먼스 평가에서 ChainMPQ는 LLaVA-1.5 및 InstructBLIP 모델에서 관계 중심 벤치마크에서 일관되게 관계 환각을 줄이는 성과를 보였습니다. 제안된 방법은 이전의 세 가지 핵심 모듈의 효과를 검증하는 약한 연결 실험(ablation study)을 통해 그 효용성이 입증되었습니다. 이러한 결과는 LVLMs의 전반적인 성능 향상에 기여할 것으로 기대됩니다.



### Improving the Spatial Resolution of GONG Solar Images to GST Quality Using Deep Learning (https://arxiv.org/abs/2510.06281)
Comments:
          5 pages; accepted as a workshop paper in ICDM 2025

- **What's New**: 이 연구는 GAN 기반의 초해상도(superresolution) 방법을 통해 GONG의 저해상도(Hα) 태양 이미지를 BBSO의 고해상도(High-Resolution) 이미지 수준으로 개선하는 최초의 시도를 보여줍니다. 저해상도 이미지는 필라멘트(filament)와 섬유(fibril) 등의 미세 구조를 명확하게 재현하는 데 한계가 있었으나, 본 연구를 통해 GAN 방법을 사용하여 이 문제를 극복합니다.

- **Technical Details**: 본 연구에서는 Real-ESRGAN 모델과 잔차-잔차 밀집 블록(Residual-in-Residual Dense Blocks), 그리고 상대적(discriminator) 구별자를 적용하여 초해상도를 구현하였습니다. GONG과 GST의 이미지 쌍을 정렬하고, MSE가 467.15, RMSE가 21.59, CC가 0.7794인 성능을 달성하였습니다. 또한, 촬영된 이미지 간의 약간의 불일치가 정량적 성능에 제약을 주는데, 이는 향후 작업에서 해결할 계획입니다.

- **Performance Highlights**: 초해상도 모델은 태양 흑점의 반경과 필라멘트 및 섬유의 세부 정보를 효과적으로 재현할 수 있었습니다. 수집된 데이터 세트는 총 281개의 훈련 이미지 쌍과 63개의 테스트 이미지 쌍으로 구성되어 있습니다. 본 연구의 모델은 실제 응용 시나리오에서의 성능이 뛰어난 것을 입증하였으며, 이는 태양 관측의 정밀도를 크게 향상시킬 것으로 기대됩니다.



### General and Efficient Visual Goal-Conditioned Reinforcement Learning using Object-Agnostic Masks (https://arxiv.org/abs/2510.06277)
- **What's New**: 이번 연구는 객체에 대한 의존이 없는 mask 기반 목표 표현 시스템을 제안하여, 에이전트가 효율적인 학습과 뛰어난 일반화 성능을 발휘할 수 있도록 지원합니다. 기존의 목표 표현 방법들이 겪었던 일반화 문제와 느린 수렴 문제를 극복하면서, 시뮬레이션 학습에서 99.9%의 도달 정확도를 기록하였습니다. 제안된 방법은 실제 로봇에서의 픽업 작업 수행 시에도 높은 정확도를 유지하며, 포지셔널 정보를 요구하지 않습니다.

- **Technical Details**: Goal Conditioned Reinforcement Learning (GCRL)은 여러 목표를 학습하는 것을 가능하게 하지만, 목표 표현의 선택이 성공 여부에 큰 영향을 미칩니다. 우리는 mask를 기반으로 하는 목표 표현 방법을 도입하여, 에이전트가 목표에 대한 시각적 단서를 제공받고, 이로 인해 알려지지 않은 목표에 대해서도 일반화할 수 있는 능력을 갖추게 됩니다. 또한, mask 크기를 사용하여 밀집 보상 신호를 효과적으로 생성함으로써 목표와의 거리 계산을 단순화합니다.

- **Performance Highlights**: 우리의 연구 결과는 mask 기반 목표 표현이 기존 방법들과 비교하여 시각적 도달 작업에서 유사하거나 더욱 우수한 성능을 보임을 보여줍니다. 두 개의 물리적 로봇을 통해 시뮬레이션에서 실제로의 전이 학습을 검증하였으며, pretrained open vocabulary 객체 탐지 모델을 활용하여 mask 생성을 수행했습니다. 이러한 방법들은 빠른 수렴성과 강력한 일반화 기능을 보장하며 실세계 작업에서도 적용 가능성을 높입니다.



### Vision Transformer for Transient Noise Classification (https://arxiv.org/abs/2510.06273)
Comments:
          9 pages, 4 figures

- **What's New**: 이 연구는 LIGO 데이터에서의 일시적 노이즈(글리치)를 22개의 기존 클래스와 O3a에서 추가된 2개의 노이즈 클래스로 분류하는 작업을 기반으로 합니다. 저자들은 Vision Transformer (ViT) 모델을 활용하여 새로운 분류 모델을 훈련하고 이 모델의 효과성을 입증했습니다. 특히, O3 기간 동안 수집된 새로운 데이터 클래스를 포함하여 LIGO 데이터를 처리하는 데 ViT 모델의 잠재력을 강조하였습니다.

- **Technical Details**: 본 연구에서는 Gravity Spy 프로젝트에서 수집된 데이터셋을 사용하였으며, ViT-B/32 모델을 통해 22개 클래스와 O3a에서 추가된 두 클래스에 대한 분류 작업을 수행했습니다. 데이터셋은 7:1.5:1.5 비율로 훈련, 검증 및 테스트 세트로 분할되었고, Adam 옵티마이저를 사용하여 모델 파라미터를 업데이트하였습니다. 훈련 과정은 15 epoch 동안 진행되며, 교차 엔트로피 손실 함수가 모델의 성능을 평가하는 데 사용되었습니다.

- **Performance Highlights**: 시험 데이터셋에 대한 ViT-B/32 모델의 F1 점수는 92.13%, 정확도는 92.26%로 평가되었습니다. 몇몇 클래스, 예를 들어 1080Lines 및 Helix는 98% 이상의 정확도를 기록한 반면, Paired_Doves 및 No_Glitch 클래스는 9.09% 및 26.56%로 저조한 성적을 보였습니다. 과거 CNN 모델들과 비교할 때 ViT는 향상된 의미적 탐지를 가능하게 하지만, 데이터 세트 크기에 따른 성능 저하의 여지를 보여주었습니다.



### Ensemble Deep Learning and LLM-Assisted Reporting for Automated Skin Lesion Diagnosis (https://arxiv.org/abs/2510.06260)
- **What's New**: 이 논문은 피부암 진단을 위한 새로운 통합 AI 프레임워크를 제안합니다. 이 시스템은 다양한 아키텍처를 가진 합성곱 신경망(CNN) 앙상블과 대형 언어 모델(LLM)을 결합하여 진단의 신뢰성과 접근성을 향상시킵니다. 또한, 이는 환자 교육을 포함한 임상 문서화의 필수 요구사항을 충족하면서 진단 출력이 임상적으로 의미 있는 평가로 변환되는 것을 가능하게 합니다.

- **Technical Details**: 제안된 시스템은 EfficientNetB3, ResNet50, DenseNet121의 이종 앙상블을 사용하여 복합적인 진단 관점을 제공합니다. 또한, 자동화된 보고서 생성을 위한 LLM 기반 시스템을 포함하여, 진단 추론 및 임상 출력을 동시에 수행할 수 있도록 설계되었습니다. 이 시스템은 불확실성 메커니즘을 내재하여 전문가 검토를 위한 비일관한 사례를 플래그하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 프레임워크는 진단 정확성을 높이는 동시에 환자가 진단 내용을 이해하고 조기 징후를 인식할 수 있게 지원합니다. 이를 통해 초기 발견률이 향상되고, 환자에게 개인화된 모니터링 지침을 제공하여 치료의 연속성을 지원합니다. 종합하여, 이 시스템은 인공지능(AI) 기반 피부 진단을 실제 임상에서 사용할 수 있는 솔루션으로 발전시키는 중요한 진전을 나타냅니다.



### Enhanced Self-Distillation Framework for Efficient Spiking Neural Network Training (https://arxiv.org/abs/2510.06254)
- **What's New**: 이 논문은 Spiking Neural Networks (SNNs)의 효율적인 학습을 위해 새로운 self-distillation 구조를 제안합니다. 기존의 BPTT 방식이 성능에서 뒤처짐과 함께 메모리 오버헤드를 초래하는 반면, 제안된 방법은 lightweight ANN을 통해 SNN의 중간 층의 firing rate를 최적화하여 높은 성능을 달성합니다. 특히, 신뢰할 수 있는 teacher signal을 활용하여 모델의 최적화를 도모합니다.

- **Technical Details**: SNN은 생물 신경세포의 동적 행동을 모방하며, binary spike 사건을 통해 정보를 처리합니다. 본 연구는 향상된 self-distillation 프레임워크를 통해 SNN의 firing rate을 ANN으로 매핑하고, 신뢰할 수 있는 지식을 활용하여 gradient distortion 문제를 완화하는 전략을 채택합니다. 경량 ANN 가지를 이용한 rate-based backpropagation이 핵심 기술입니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100, CIFAR10-DVS 및 ImageNet 데이터셋에서 본 연구의 방법이 훈련 복잡성을 줄이면서 높은 성능을 달성함을 실험적으로 입증하였습니다. 실험 결과는 제안된 방법이 기존의 효율적인 학습 방법보다 균형 잡힌 성능과 훈련 효율성을 제공함을 보여줍니다.



### Does Physics Knowledge Emerge in Frontier Models? (https://arxiv.org/abs/2510.06251)
Comments:
          8 pages, 7 figures. Preprint

- **What's New**: 우리는 여섯 가지 최신 비전-언어 모델(VLM)들이 물리적 동역학을 이해하고 예측하는 능력을 벤치마킹하였습니다. 이를 통해 물리적 상황에 대한 예측이나 대안 상황을 가정하는 태스크를 통해 VLM의 한계를 드러내었습니다. 특히, 인지 성능이 평가 정확도와 강하게 상관되지 않음을 밝혀냈습니다. 이는 현재 VLM의 감각과 물리적 이해가 분리되어 있으며, 원인적 이해를 결합할 수 있는 아키텍처의 필요성을 강조합니다.

- **Technical Details**: 실험에서는 Physion, Physion++, CLEVRER의 유명한 데이터셋을 사용하여 여섯 가지 VLM의 성능을 평가하였습니다. 각 VLM은 물체의 운동 예측, 공간적 관계 이해 등 물리적 추론을 다루는 진단 질문과 감각적 인식을 평가하는 하위 테스트를 수행했습니다. 이러한 방법론을 통해 우리는 두 가지 주요 구성 요소인 인식(perception)과 물리적 이해(physical understanding)를 분리하여 분석했습니다.

- **Performance Highlights**: 결과적으로, 벤치마크 평가 작업에서의 정확도가 낮고 호재 혹은 반사적 대답에서 모델이 일반적으로 물리적 동역학을 신뢰할 수 없음을 보여주었습니다. 물체 인식이나 운동 예측 능력이 높더라도 예측 작업이나 반사적 대답에서 항상 뛰어난 성과를 내지 않는다는 점이 발견되었습니다. 특히, 강한 진단 성능이 평가 성능의 향상으로 이어지지 않는 경우가 많은 것으로 나타났습니다.



### multimodars: A Rust-powered toolkit for multi-modality cardiac image fusion and registration (https://arxiv.org/abs/2510.06241)
- **What's New**: 이 논문에서는 신뢰성 있는 3D 관상동맥 모델을 구축하기 위해 필수적인 여러 영상 모달리티(complementary imaging modalities) 결합을 다룬다. 기존 연구에서 intravascular imaging과 CCTA의 융합이 시도되었지만, 다중 상태 분석(multi-state analysis)을 위한 열린(toolkit) 유연한 도구는 부족했다. 이번 연구는 deterministic behaviour, 높은 성능(high performance), 그리고 간편한 파이프라인 통합을 제공하는 multimodars 패키지를 소개한다.

- **Technical Details**: multimodars는 deterministic alignment algorithms와 NumPy 중심의 데이터 모델을 기반으로 한 경량 패키지이다. Rust 백엔드를 최적화하여 확장 가능하고 재현 가능한(experiment) 실험을 지원한다. 이 패키지는 AIVUS-CAA 소프트웨어로 생성된 데이터 포맷을 포함한 CSV/NumPy 입력을 수용할 수 있다.

- **Performance Highlights**: multimodars는 rest/stress 및 pre-/post-stenting 상태에서의 분석을 용이하게 하기 위해 설계되었다. 그 성능은 기존의 시스템보다 향상되어 놀라운 속도와 정확도를 자랑한다. 이 도구는 실험 연구자들이 복잡한 3D 모델을 보다 쉽게 구축하고 평가할 수 있도록 돕는다.



### Uncertainty Quantification In Surface Landmines and UXO Classification Using MC Dropou (https://arxiv.org/abs/2510.06238)
Comments:
          This work has been accepted and presented at IGARSS 2025 and will appear in the IEEE IGARSS 2025 proceedings

- **What's New**: 이번 연구는 인간 지뢰 제거 작업에서 표면 지뢰 및 비폭발 잔해(UXOs)의 탐지를 위한 Monte Carlo (MC) Dropout을 이용한 불확실성 정량화 개념을 도입합니다. ResNet-50 아키텍처에 통합하여 실험한 결과는 기존 신경망 모델의 취약성을 극복할 수 있는 가능성을 제시합니다. MC Dropout 접근 방식은 예측 신뢰성을 추가적인 지표로 제공하여 지뢰 제거 작전에서 더 신뢰할 수 있는 결정을 내리는 데 도움이 될 수 있습니다.

- **Technical Details**: 이 연구에서는 ResNet-50 모델을 이용하여 MC Dropout을 통합함으로써 예측 불확실성을 추정하는 방법을 소개합니다. 연구팀은 깨끗한 테스트 이미지, 적대적 변형 및 노이즈가 있는 테스트 이미지를 포함한 세 가지 시나리오에서 모델을 평가했습니다. MC Dropout은 모델의 가중치를 확률적 분포로 보고, 예측 불확실성을 효율적으로 추정할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 다수의 확률적 전방 패스를 통해 얻은 평균 예측 및 그 분산은 불확실성의 지표 역할을 했습니다. 예측의 분산이 높을수록 모델 예측의 불확실성이 증가하는 것을 확인했습니다. 이 연구는 표면 지뢰 및 UXOs 분류에서 불확실성 정량화의 필요성을 강조하며, 향후 현실 세계의 지뢰 제거 애플리케이션에 적용할 수 있는 기초를 마련합니다.



### User to Video: A Model for Spammer Detection Inspired by Video Classification Technology (https://arxiv.org/abs/2510.06233)
Comments:
          Accepted by International Joint Conference on Neural Networks (IJCNN) 2025

- **What's New**: 이번 연구는 비디오 분류 기술에 영감을 받아 사용자의 행동을 비디오로 모델링하여 스팸 사용자를 탐지하는 UVSD 모델을 제안하고 있습니다. 사용자의 행동을 픽셀로 변환하는 user2pixel 알고리즘과 행동을 이미지로 만드는 behavior2image 알고리즘을 사용하여, 기존의 그래프 모델링 접근 방식이 갖는 메모리 소모 문제를 해결합니다. 이러한 새로운 접근 방식으로 인해 과거 행동의 분석과 스팸 사용자 탐지가 보다 효율적으로 이루어질 것으로 기대됩니다.

- **Technical Details**: 이 방법론은 사용자를 픽셀로, 그들의 행동을 프레임 이미지로 변환하는 과정으로 구성됩니다. 사용자의 행위 그래프를 낮은 차원으로 변환하기 위해 representation learning과 세분화 및 확산 알고리즘을 적용하여 이미지를 생성합니다. 그 후, 이러한 프레임 이미지 시퀀스를 분석하여 스팸 사용자를 식별하는 비디오 분류 알고리즘이 통합됩니다.

- **Performance Highlights**: 실험을 통해 WEIBO와 TWITTER와 같은 공개 데이터셋에서 UVSD 모델이 기존의 최첨단 방법들보다 우수한 성능을 보이는 것으로 나타났습니다. 제안된 모델은 고차원 관계를 효과적으로 처리하면서도 처리 시간과 자원 소모를 최소화하는 장점이 있습니다. 이러한 결과는 UVSD 모델의 우수성을 증명하며, 소셜 미디어에서 스팸 사용자 탐지의 새로운 가능성을 시사합니다.



### CML-Bench: A Framework for Evaluating and Enhancing LLM-Powered Movie Scripts Generation (https://arxiv.org/abs/2510.06231)
Comments:
          24 pages, 9 figures

- **What's New**: 이 연구에서는 CML-Dataset이라는 새로운 데이터셋을 구축하여 인상적인 구조의 영화 스크립트 분석을 통해 LLM의 한계를 탐구합니다. 또한, 대화의 일관성(Dialogue Coherence), 캐릭터의 일관성(Character Consistency), 그리고 플롯의 타당성(Plot Reasonableness)이라는 세 가지 주요 차원을 제시하며, 이들 기준을 바탕으로 CML-Bench라는 평가 프레임워크를 개발했습니다. 이는 LLM이 생성한 스크립트의 질을 정량적으로 평가하고, 인간이 작성한 스크립트와 LLM 간의 질적 차이를 분석하는데 중점을 둡니다.

- **Technical Details**: CML-Dataset은 약 1,800편의 필터링된 영화 스크립트에서 파생된 100개의 영화 스크립트와 그 요약으로 구성됩니다. CML-Bench는 대화 일관성, 캐릭터 일관성 및 플롯 타당성을 측정할 수 있는 8개의 해석 가능한 정량적 메트릭으로 구성되어 있으며, 이 메트릭들은 언어 모델과 결합된 정형 파싱 및 벡터 유사성 계산을 통해 구현됩니다. 이를 통해 LLM이 생성한 스크립트에 대한 세밀하고 객관적인 평가가 가능합니다.

- **Performance Highlights**: CML-Bench를 통해 분석한 결과, 현재의 모든 LLM은 인간이 작성한 스크립트에 비해 일관성에서 일관되게 성능이 저조한 것으로 나타났습니다. 특히 대화 일관성, 캐릭터 일관성, 플롯 타당성에서 큰 차이를 보였습니다. CML-Instruction을 통한 추가적인 실험에서도 LLM이 생성한 스크립트의 질이 유의미하게 향상되었으며, 이는 인간의 선호와 일치하는 결과를 동시에 보여주었습니다.



### Milestone Determination for Autonomous Railway Operation (https://arxiv.org/abs/2510.06229)
Comments:
          Paper submitted and partially accepted to ICART 2025, paper is 8 pages and has 1 figure, 2 tables

- **What's New**: 이번 논문에서는 철도 자동화 분야에서의 컴퓨터 비전 시스템 개발에 대한 주요 도전에 대해 다루고 있습니다. 특히, 고품질의 연속적인 데이터의 제한된 가용성이 문제라는 점을 강조하고 있습니다. 기존의 데이터셋은 스페이셜-템포럴(spatio-temporal) 컨텍스트가 부족하여 실시간 의사 결정에 어려움을 초래합니다.

- **Technical Details**: 논문에서는 경로별(contextually relevant) 및 규칙 기반(rule-based) 모델을 통한 마일스톤 결정(milestone determination) 개념을 제안하고 있습니다. 이는 동적인 구성요소의 일반적인 인식을 요구하지 않고, 특정한 결정 지점에 집중하여 학습 과정을 단순화합니다. 이러한 방식은 예측 가능한 환경에서 비전 에이전트(vision agents) 훈련을 가능하게 합니다.

- **Performance Highlights**: 이 접근 방식은 철도 자동화에 대한 보다 안전하고 효율적인 머신 러닝 시스템을 촉진할 수 있는 실용적인 프레임워크를 제공합니다. 제안된 방법은 현실 세계의 운영 논리(reality operational logic)와 더 밀접하게 일치하는 풍부한 연속 데이터셋을 생성하여, 문제를 해결할 수 있는 기반이 됩니다.



### TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics (https://arxiv.org/abs/2510.07181)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 연구에서는 TIGeR (Tool-Integrated Geometric Reasoning)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 Vision-Language Models (VLMs)를 감각적 추정기에서 기하학적 컴퓨터로 변환하여 정확한 기하학적 계산을 생성하고 실행할 수 있도록 합니다. TIGeR는 외부 도구를 활용하여 기하학적 추론 요구 사항을 인식하고, 적절한 계산 코드를 합성하며, 정확한 계산을 위해 특화된 라이브러리를 호출할 수 있게 합니다.

- **Technical Details**: TIGeR-300K라는 포괄적인 도구 호출 지향 데이터셋을 개발하였습니다. 이 데이터셋은 점 변환, 자세 추정, 궤적 생성 및 공간 호환성 검증을 포함하여 총 300,000개의 샘플로 구성되어 있습니다. TIGeR는 지도학습 세부 조정(Supervised Fine-Tuning, SFT)과 강화학습 세부 조정(Reinforcement Fine-Tuning, RFT)을 결합한 두 단계의 훈련 파이프라인을 통해 훈련됩니다.

- **Performance Highlights**: TIGeR는 기하학적 벤치마크에서 최신 성과(SOTA)를 달성했으며, 실제 로봇 조작 작업에서 센티미터 수준의 정확성을 보였습니다. 이는 기하학적 계산을 위한 프로그래밍 가능 도구 호출 방식으로 귀결되며, 로봇 조작과 모션 계획을 위한 정밀한 포즈 및 궤적 생성을 가능하게 합니다. 최종적으로 TIGeR는 정밀한 기하학적 추론을 통해 실세계 로봇 작업을 지원하는 새로운 패러다임을 제시합니다.



### TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking (https://arxiv.org/abs/2510.07134)
Comments:
          Project page: this https URL

- **What's New**: TrackVLA++는 Embodied Visual Tracking (EVT) 작업을 위한 새로운 Vision-Language-Action (VLA) 모델로, 개선된 공간 추론 기법과 Target Identification Memory (TIM) 모듈을 포함합니다. 이 모델은 복잡한 환경에서 비슷한 물체에 의해 방해받거나 긴 시간 동안 가려질 경우에도 목표를 효과적으로 추적할 수 있는 능력을 강화합니다. Polar-CoT라는 새로운 사고 체계를 도입하여 목표의 상대적 위치를 추론하고, 이를 바탕으로 메모리를 업데이트합니다.

- **Technical Details**: TrackVLA++는 Polar-CoT 기법을 통해 목표의 상대적 위치를 극소화된 극좌표 토큰으로 인코딩합니다. TIM은 긴 시간 동안 목표의 시각적 정체성을 유지하며, 신뢰 기반의 업데이트 메커니즘을 활용하여 메모리의 상태를 조정합니다. 이러한 방법은 다중 뷰 설정에서도 동일하게 적용되며, 향상된 추적 성능을 제공합니다.

- **Performance Highlights**: TrackVLA++는 공공 벤치마크에서 SOTA 성능을 기록하며, 특히 EVT-Bench DT 분할에서 이전의 선도적 방법보다 각각 5.1% 및 12% 더 높은 성공률을 달성했습니다. 실제 환경에서도 뛰어난 제로샷 일반화를 보여주어 동적이고 가려진 상황에서도 강력한 추적 성능을 발휘합니다.



### Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications (https://arxiv.org/abs/2510.07077)
Comments:
          Accepted to IEEE Access, website: this https URL

- **What's New**: 최근 큰 변화가 일어나고 있는 로봇 공학 분야에서, Vision-Language-Action (VLA) 모델이 주목받고 있습니다. VLA 모델은 전통적으로 따로 연구되어온 시각, 언어, 행동 데이터를 통합하여, 다양한 작업 및 환경에서 일반화된 정책을 학습하는 것을 목표로 합니다. 이로 인해 로봇은 최소한의 추가 데이터로도 새로운 작업을 수행할 수 있는 가능성이 높아집니다.

- **Technical Details**: 이 논문은 VLA 모델의 구조적 전환 및 중앙 요소를 체계적으로 검토합니다. VLA 모델은 시각적 관찰과 자연어 지시를 입력으로 받아 로봇 액션을 직접 생성하는 시스템으로 정의됩니다. 또한, 데이터 수집, 공공 데이터셋, 데이터 증강 방법 및 평가 기준과 같은 로봇 시스템의 실질적인 배치를 지원하기 위한 다양한 요소가 포함되어 있습니다.

- **Performance Highlights**: VLA 모델은 적은 양의 특정 작업 데이터로도 다양한 로봇 임무를 수행할 수 있는 잠재력을 지니고 있습니다. 그러나 VLA 모델의 발전은 데이터 가용성, 신체적 불일치, 계산적 제약 등 여러 도전에 의해 제한되고 있습니다. 이러한 문제를 해결하기 위해 다음 세대의 로봇 시스템의 효율성과 접근성을 높이기 위한 연구가 필요합니다.



### Introspection in Learned Semantic Scene Graph Localisation (https://arxiv.org/abs/2510.07053)
Comments:
          IEEE IROS 2025 Workshop FAST

- **What's New**: 이 연구는 학습된 자기 지도 대비(contrastive) 의미 지역화(semantic localisation) 프레임워크에서 의미가 지역화 성능과 강건성에 미치는 영향을 조사합니다. 모델은 원본 지도(original map)와 편향된 지도(perturbed map)에서 훈련을 받은 후, 환경 소음에서 필터링을 수행하고, 일상적인 혼잡보다 독특한 랜드마크를 우선시하는지에 대한 철저한 자기 성찰(post-hoc introspection) 분석을 수행합니다. 다양한 해석 가능성 방법(interpretability methods)을 검증하고, 통합 그래디언트(integrated gradients)와 주의 가중치(attention weights)가 가장 신뢰할 수 있는 모델 행동 탐색 기법으로 자리 잡았습니다.

- **Technical Details**: 연구는 하이레벨(higher-level) 의미 정보가 로컬라이제이션에 어떻게 활용될 수 있는지를 보여주며, 인간과 비슷하게 환경 소음을 필터링하는 방법을 모사합니다. 3D 씬 그래프(scene graph)를 활용하여 공간 개념(노드)과 관계(엣지)를 모델링합니다. 데이터셋으로는 포토 리얼리스틱(photorealistic) 환경과 완전한 메트릭-의미 주석(metric-semantic annotations)을 포함한 uHumans2 데이터셋을 사용합니다.

- **Performance Highlights**: 연구 결과는 모델이 시각적 및 구조적 변형이 존재하는 상황에서도 노이즈에 강하고 의미적으로 중요한 관계를 학습함으로써 설명 가능한 지역화를 가능하게 함을 나타냅니다. 클래스 제거에 따른 성능 저하 분석과 속성 기여 변동 분석을 통해, 자주 등장하는 객체는 다운 가중치 처리되고, 희귀한 랜드마크가 지역 지정 해소에 중요한 역할을 담당하는 것으로 나타났습니다. 통합 그래디언트와 주의 가중치의 신뢰성 분석을 수행하였고, 이들은 강력한 객체 중요도 속성 신호를 제공합니다.



### Sharpness-Aware Data Generation for Zero-shot Quantization (https://arxiv.org/abs/2510.07018)
- **What's New**: 본 논문은 제로샷 양자화(zero-shot quantization, ZSQ) 문제에서 훈련 데이터 생성 시 모델의 샤프니스(sharpness)를 고려하는 새로운 방법론을 제안합니다. 기존 ZSQ 기법들은 생성된 데이터와 모델의 샤프니스 사이의 관계를 고려하지 않았던 반면, 본 연구에서는 샤프니스 최소화가 훈련 데이터의 재구성 손실 그래디언트(gradient) 상호 일치화를 최대화하는 방식으로 이루어질 수 있음을 입증하였습니다.

- **Technical Details**: 제안된 방법론에서는 생성된 데이터와 실제 검증 데이터의 재구성 손실 그래디언트 간의 일치성을 극대화하여 샤프니스 최소화를 이룹니다. 제로샷 환경에서 실제 검증 세트가 없을 경우, 생성된 각 샘플과 그 이웃 샘플 간의 그래디언트 일치화를 근사하여 이 문제를 해결합니다. 또한, 샤프니스 인식을 통한 데이터 생성(Sharpness-Aware Data Generation, SADAG)을 통해 저비트 양자화(low-bit quantization) 환경에서 성능 향상을 도모합니다.

- **Performance Highlights**: CIFAR-100과 ImageNet 데이터셋에 대한 실험 결과, 제안된 SADAG 방법은 기존의 최신 ZSQ 방법들보다 우수한 성능을 보였습니다. 본 연구는 샤프니스 감소가 모델의 일반화 능력 향상에 기여하는 것을 보여주며, ZSQ 문제에 대한 새로운 접근법을 제공합니다. 실험 결과는 제안된 방법이 효과적으로 양자화된 모델의 성능을 개선함을 입증하였습니다.



### Revisiting Mixout: An Overlooked Path to Robust Finetuning (https://arxiv.org/abs/2510.06982)
- **What's New**: 이 논문에서는 Mixout이라는 스토캐스틱 정규화 기법을 재조명하여, 사전 훈련된 가중치와 미세 조정된 가중치를 주기적으로 교체하는 방법으로 모델의 강건성을 개선하는 새로운 접근법을 제안합니다. GMixout은 고정된 앵커를 지수 이동 평균(Exponential Moving Average) 스냅샷으로 대체하고, 마스킹 주기를 조정하는 하이퍼파라미터를 도입함으로써 미세 조정된 모델의 성능을 최적화합니다.

- **Technical Details**: GMixout의 구현은 드문 커널(sparse kernel) 기법을 사용하여, 훈련 과정에서 오직 소수의 파라미터만 업데이트하여 소비자 등급의 GPU에서 훈련 가능하도록 합니다. 또한, 이 방법은 특정 하이퍼파라미터에 의해 조절될 수 있는 재샘플링 주기를 포함하여, 무작위 하위 네트워크(subnetwork)의 다양성을 증가시키는 동시에 고정된 가중치에 의존하는 문제를 해결합니다.

- **Performance Highlights**: GMixout은 다양한 벤치마크(예: ImageNet, DomainNet, CIFAR100-C)에서 성능을 순차적으로 개선하여 제로샷(Zero-shot) 성능을 초과하는 결과를 보였습니다. 특히, 기존 모델 소프(Model Soups)와 강력한 파라미터 효율적 미세 조정 방법을 초월하는 강인성을 보여주어, 도메인 변화(distribution shift)에서도 우수한 성능을 발휘합니다.



### High-Rate Mixout: Revisiting Mixout for Robust Domain Generalization (https://arxiv.org/abs/2510.06955)
Comments:
          WACV 2026: Winter Conference on Applications of Computer Vision 2026

- **What's New**: 이 논문에서는 Dropout 대신에 Mixout이라는 새로운 확률적 정규화 기법을 도입하여 도메인 일반화 (domain generalization) 문제를 해결하고자 합니다. Mixout은 훈련 과정에서 일부 미세 조정된 가중치를 원래의 미리 훈련된 가중치로 확률적으로 교체하여 과적합 (overfitting)을 방지합니다. 이 방법은 높은 마스킹 확률과 함께 사용되어 최적의 성능을 발휘하며, 실험 결과 이를 통해 앙상블 방식에 근접한 성능을 보이는 동시에 훈련 비용을 크게 줄일 수 있음을 보여줍니다.

- **Technical Details**: Mixout은 ViT (Vision Transformers)와 ResNet 아키텍처에 대해 각각 0.9 및 0.8의 높은 마스킹 확률을 요구합니다. 이는 훈련 과정에서 미리 훈련된 가중치와 잠금 상태의 가중치를 동적으로 교체하여 데이터의 다양성을 탐색합니다. Mixout을 사용함으로써 45%의 그래디언트 계산 비용 절감과 90%의 메모리 사용량 감소를 달성할 수 있으며, 이는 고성능 도메인 일반화에 기여합니다.

- **Performance Highlights**: 실험을 통해 PACS, VLCS, OfficeHome, TerraIncognita, DomainNet과 같은 다섯 가지 도메인 일반화 벤치마크에서 Mixout을 사용한 고마스킹(Mixout with high masking probability) 방법이 앙상블 기반 접근법과 동등한 도메인 밖 정확도를 기록했습니다. Mixout 방법은 여러 모델을 훈련하고 저장할 필요가 없기에 매우 효율적입니다. 이러한 결과를 통해 Mixout은 도메인 간 성능을 향상시키는 동시에 비용 효율성을 극대화하는 가능성을 가지고 있음을 입증했습니다.



### Angular Constraint Embedding via SpherePair Loss for Constrained Clustering (https://arxiv.org/abs/2510.06907)
Comments:
          Accepted by NeurIPS 2025, 6 Figures and 1 Table in Main text, 18 Figures and 5 Tables in Appendices

- **What's New**: 이번 연구에서는 Deep Constrained Clustering (DCC) 방식에서 기존 방법의 한계를 극복하기 위해 SpherePair이라는 새로운 angular constraint embedding 접근법을 제안합니다. SpherePair 손실(SpherePair loss)은 기하학적 구성을 사용하여 쌍(pairwise) 제약조건을 충실하게 인코딩하며, 클러스터링에 적합한 각 공간에서의 임베딩을 생성합니다. 이 접근법은 클러스터 수를 지정할 필요 없이 클러스터링을 수행할 수 있도록 하여, 하이퍼파라미터 조정을 위한 노력을 줄이고 효율성을 개선합니다.

- **Technical Details**: SpherePair 손실 함수는 코사인 유사도(cosine similarity)를 활용하여 앵커 없이 각 공간에서 잠재적 표현(latent representation)을 학습합니다. 이를 통해 긍정 쌍(positive pairs)과 부정 쌍(negative pairs) 간의 적절한 거리를 유지하면서 클러스터 간 거리의 균형을 맞출 수 있습니다. 이 방법은 이론적으로 검증된 기초를 바탕으로 하여 특정 조건에서 최적 성능을 보장합니다.

- **Performance Highlights**: 범위가 다양한 벤치마크 데이터셋에서 기존 DCC 방법들과 비교하여 SpherePair의 우수한 성능을 입증했습니다. 특히, 간단한 K-means 알고리즘을 사용하여 학습된 표현에 대한 클러스터 수를 신속하게 추측할 수 있으며, 미지의 데이터에도 잘 일반화됩니다. 실험 결과는 저자들이 제안한 방법이 대안 방법들보다 효과적이고 실용적임을 보여줍니다.



### SaFeR-VLM: Toward Safety-aware Fine-grained Reasoning in Multimodal Models (https://arxiv.org/abs/2510.06871)
- **What's New**: 이번 논문에서는 SaFeR-VLM이라는 새로운 안전 정렬 강화 학습 프레임워크를 제안합니다. 이 프레임워크는 다중 모달 (multimodal) 추론 과정에서 안전성을 직접 통합하여, 안전을 수동적인 보호 장치가 아닌 적극적인 추론 동력으로 전환합니다. 또한, 안전성이 높은 데이터셋(QI-Safe-10K)을 통해 위험한 상황을 보다 효과적으로 다룰 수 있는 구조를 갖추고 있습니다.

- **Technical Details**: SaFeR-VLM의 기본 구조는 네 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 특정 안전상 중요한 사례에 중점을 둔 커리큘럼 데이터셋(QI-Safe-10K)입니다. 두 번째는 안전성 인식 롤아웃(safety-aware rollout)으로, 안전하지 않은 출력을 단순히 버리지 않고 반영 및 수정하는 과정을 포함합니다. 세 번째는 다차원 보상 모델링(structured reward modeling)으로, 환각과 모순에 대한 명시적 벌칙을 포함한 보상 신호가 포함되어 있습니다. 네 번째는 안전성 인식 최적화(safety-aware optimization)로, 이러한 신호를 GRPO 최적화 방법에 통합하는 방식입니다.

- **Performance Highlights**: SaFeR-VLM은 여섯 개의 안전 기준 벤치마크에서 우수한 성능을 발휘하며, 3B 모델에서 70.13의 안전성 점수와 78.97의 유용성을 기록했습니다. 이는 기존 10배 이상 큰 모델들을 초과하는 성능입니다. 7B 모델은 안전성 지표에서 GPT-5 미니 및 Gemini-2.5-Flash를 각각 6.47점, 16.76점 초과하여 유용성과 안전성을 동시에 개선한 사례로 주목받고 있습니다.



### Capture and Interact: Rapid 3D Object Acquisition and Rendering with Gaussian Splatting in Unity (https://arxiv.org/abs/2510.06802)
- **What's New**: 이번 논문은 3D 객체를 실시간으로 캡처하고 렌더링하는 새로운 엔드 투 엔드 파이프라인을 제안합니다. 실제 물체를 모바일 장치로 빠르게 스캔하고, 클라우드 처리와 로컬 컴퓨터를 통해 상호작용할 수 있는 렌더링을 가능하게 합니다. 이를 통해 서로 다른 분야에서의 높은 품질의 3D 콘텐츠 생성 접근성이 개선됩니다.

- **Technical Details**: 시스템은 모바일 장치로 촬영한 비디오를 클라우드 서버에 업로드한 후, Structure-from-Motion (SfM) 기술을 사용하여 카메라 포즈를 추정하고 희소 포인트 클라우드를 생성합니다. 각 포인트는 3D Gaussian으로 변환되며, 최적화 과정에서 약 30K회 반복하여 세밀하고 사실적인 3D 표현을 만듭니다. 이 과정은 평균적으로 10분이 소요되며, 완성된 모델은 Unity에서 실시간으로 렌더링됩니다.

- **Performance Highlights**: 테스트 결과, 재구성된 객체는 평균 PSNR (Peak Signal-to-Noise Ratio) 값이 34.65에 도달하며, 렌더링은 MacBook Pro에서 약 150 fps로 이뤄집니다. 이 시스템은 다양한 복잡도의 객체를 처리할 수 있으며, 향후에는 실시간 이벤트와 같은 시간 민감한 애플리케이션을 위한 최적화 기법을 탐구할 예정입니다.



### Bionetta: Efficient Client-Side Zero-Knowledge Machine Learning Proving (https://arxiv.org/abs/2510.06784)
- **What's New**: 이번 보고서에서는 우리의 UltraGroth 기반의 제로 지식(Zero-Knowledge) 기계 학습 프레임워크 Bionetta와 EZKL, Lagrange의 deep-prove 및 zkml과 같은 유사 툴의 성능을 비교합니다. 연구 결과, 커스텀 크래프트된 신경망(custom-crafted neural networks)의 증명 시간(proving time)이 크게 향상되었음을 보여 주었습니다. 특히 이 프레임워크는 모바일 기기에서도 증명이 가능하여 다양한 클라이언트 측 증명 애플리케이션(client-side proving applications)을 활성화할 수 있습니다.

- **Technical Details**: 우리의 방법론은 회로 컴파일(circuit compilation) 및 신뢰할 수 있는 설정(trusted setup) 생성과 같은 일회성 전처리 단계(preprocessing steps)의 비용이 증가하지만, EVM(Ethereum Virtual Machine) 스마트 계약에서 과도한 증명 크기(proof size)와 검증 오버헤드(verification overheads) 없이 배포 가능한 유일한 방식으로 보입니다. 이로 인해 여러 응용 프로그램에서 강력한 성능을 제공합니다.

- **Performance Highlights**: Bionetta는 사용자 정의 신경망의 증명 시간을 획기적으로 단축시키며, 이는 모바일 기기에서도 가능하다는 점에서 다른 도구들과의 차별성을 보입니다. 이러한 이점 덕분에 다양한 클라이언트 측 애플리케이션에서의 활용이 기대됩니다. 또한 기존의 증명 도구들에 비해 높은 효율성을 제공하며, 이를 통해 더 많은 개발자들이 기존 EVM 환경 내에서 제로 지식 기계 학습을 활용할 수 있게 됩니다.



### GPT-5 Model Corrected GPT-4V's Chart Reading Errors, Not Prompting (https://arxiv.org/abs/2510.06782)
- **What's New**: 이번 연구는 제로샷(Zero-Shot) 대형 언어 모델(LLMs)과 프롬프트 사용이 차트 읽기 작업에 미치는 영향을 정량적으로 평가한 결과를 제시합니다. 우리는 107개의 시각화 질문을 통해 에이전틱 GPT-5와 다중 모달 GPT-4V 간의 추론 정확성을 비교했습니다. 결과적으로, 모델 아키텍처는 추론 정확성에서 우위를 차지하며, 특히 GPT-5가 정확도를 크게 향상시켰고 프롬프트 변형은 그에 비해 미미한 효과를 보였습니다.

- **Technical Details**: 연구는 차트 읽기 작업에서 LLM 응답 방식과 최신 구현이 전통적 모델보다 효율성이 높은지를 탐구합니다. 우리는 CHART-6 벤치마크를 활용하여 GPT-4V로 잘못된 응답을 한 질문을 식별하고, 여러 LLM 모델과 프롬프트 조건 조합을 통해 평가했습니다. 사용한 주요 데이터셋은 107개의 질문으로 구성되며, 이는 다양한 프롬프트를 사용하여 반복적으로 테스트되었습니다.

- **Performance Highlights**: 연구 결과, GPT-5는 모든 데이터셋에 걸쳐 GPT-4o보다 현저히 더 우수한 성능을 보였습니다. 프롬프트 조건들의 차이점은 상대적으로 작지만, GPT-5의 동작은 전반적으로 더 나은 정확성을 입증했습니다. 부트스트랩 신뢰 구간을 통해 정확성의 통계적 유의성도 평가하였으며, 이러한 결과는 LLM의 성능 향상에 대한 중요한 통찰을 제공합니다.



### UniFField: A Generalizable Unified Neural Feature Field for Visual, Semantic, and Spatial Uncertainties in Any Scen (https://arxiv.org/abs/2510.06754)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 UniFField라는 새로운 시스템을 소개합니다. UniFField는 3D 장면 이해를 위한 통합된 불확실성 인지 신경 피처 필드(unified uncertainty-aware neural feature field)로, 시각적, 의미적(semantic), 기하학적 특성을 하나의 일반화된 표현으로 결합합니다. 이 시스템은 다양한 환경에서 제로샷(zero shot)으로 적용 가능하며, 로봇이 장면을 탐색하면서 RGB-D 이미지를 점진적으로 추가하여 불확실성 추정(unity estimation)을 동시에 업데이트할 수 있습니다.

- **Technical Details**: UniFField는 NN이 설정된 RGB-D 프레임 세트를 기반으로 하며, 각 포인트를 시각적, 공간적, 의미적 속성과 해당 불확실성을 설명하는 통합 피처로 매핑합니다. 이 필드는 암시적이며, 임의의 3D 위치에서 정보를 유연하게 추출할 수 있습니다. 또한 이 시스템은 새로운 RGB-D 프레임이 관찰될 때마다 점진적인 업데이트를 허용하는 추가 가능(additive)한 필드로 설계되었습니다.

- **Performance Highlights**: 연구팀은 UniFField를 활용하여 모바일 조작자 로봇이 진행하는 능동적 객체 탐색(active object search) 작업에서 유용성을 입증하였습니다. 이 시스템은 복잡한 로봇 작업을 위한 견고한 의사결정을 지원하며, 장면 재구성과 의미적 특성 예측(prediction)에서 모델의 예측 오류를 정확히 설명하는 불확실성을 제공함으로써 성능을 향상시킵니다.



### The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators (https://arxiv.org/abs/2510.06646)
- **What's New**: 이번 연구는 기계 학습된 연산자(Machine-Learned Operators, MLOs)가 높은 해상도의 데이터에서 추론(inference)을 수행할 수 있는지, 즉 '제로샷 초해상도(zero-shot super-resolution)'을 가능하게 하는지를 평가합니다. 연구진은 MLOs가 여러 해상도에서의 추론을 수행하는 데 있어 두 가지 주요 행동, 즉 주파수 정보의 외삽(extrapolation)과 해상도 간의 보간(interpolation)의 실패를 관찰했습니다. 이 결과를 바탕으로, MLOs가 훈련된 해상도 이외의 해상도에서 정확한 추론을 하지 못한다고 결론지었습니다.

- **Technical Details**: 연구의 주요 초점은 MLOs가 다중 해상도 추론(multi-resolution inference)을 수행하는데 있어서 불완전함을 드러내는 것입니다. 특히, MLOs는 새로운 주파수에 대한 추론을 실패하며, 훈련 당시의 해상도와 다른 해상도에서는 오류를 범할 수 있습니다. 이를 해결하기 위해 연구진은 다중 해상도 훈련(multi-resolution training) 프로토콜을 제안하였으며, 이는 저해상도와 고해상도의 데이터 세트를 동시에 활용하는 방식입니다.

- **Performance Highlights**: 제안된 다중 해상도 훈련 접근법은 훈련 비용을 크게 증가시키지 않으면서도 MLOs의 전반적인 성능을 향상시킬 수 있습니다. 연구 결과, MLOs는 훈련 데이터 해상도에 의존적으로 동작하며, 적절히 다양한 해상도를 아우르는 훈련을 통해 다중 해상도 일반화(multi-resolution generalization)가 가능하다는 것을 보여주었습니다. 이 연구는 MLOs의 설계 및 적용에 대한 중요한 통찰을 제공하며, 실시간 응용에서의 잠재적 활용을 제시합니다.



### Control-Augmented Autoregressive Diffusion for Data Assimilation (https://arxiv.org/abs/2510.06637)
- **What's New**: 이 논문은 Auto-Regressive Diffusion Models(ARDM)에서 가이드를 개선하는 새로운 방법론을 제안합니다. 저자들은 경량화된 컨트롤러 네트워크를 추가하여 사전 학습된 ARDM을 보강하고, 이를 통해 예측 오류를 효과적으로 줄이는 방법을 제시합니다. 이 접근법은 비선형 역학을 모델링해야 하는 데이터 동화(data assimilation, DA) 문제에서 특히 유용하다는 점에서 차별화됩니다.

- **Technical Details**: 제안된 방법은 사전 학습된 ARDM의 생선 동역학(generative dynamics)에 학습된 제어 메커니즘(control mechanism)을 통합합니다. 방법론은 입력에 따라 diffusion 과정을 안내하도록 설계되었으며, 이는 전통적인 데이터 동화 기법에 비해 높은 효율성을 자랑합니다. 저자들은 이러한 구조가 다양한 관찰 조건 하에서도 안정성과 정확성을 높인다고 주장합니다.

- **Performance Highlights**: 저자들은 제안된 방법이 두 개의 고전적인 PDE 데이터셋과 여섯 가지 관찰 조건에서 네 가지 최신 기법(state-of-the-art)보다 우수한 성능을 보여주었다고 주장합니다. 이 연구는 계산 집약적인 최적화 절차를 피함으로써 실시간 예측을 보다 정확하고 안정적으로 할 수 있도록 합니다. 또한, 저자들은 코드를 공개하여 연구의 재현성을 높일 계획을 밝히고 있습니다.



### StruSR: Structure-Aware Symbolic Regression with Physics-Informed Taylor Guidanc (https://arxiv.org/abs/2510.06635)
- **What's New**: 이 논문에서는 구조 인식 기호 회귀 프레임워크(StruSR)를 제안하여, 학습된 Physics-Informed Neural Networks (PINNs)를 활용하여 시계열 데이터에서 구조화된 물리적 우선 정보를 추출합니다. 특히, 이 프레임워크는 지역 Taylor 전개를 통해 유도된 구조 정보를 활용하여 기호 표현의 진화를 안내합니다. 기존의 기법들은 데이터 적합성에 주로 초점을 맞추는 반면, 이 방법은 물리적 원칙과 구조적 정렬이 동시에 이루어지도록 합니다.

- **Technical Details**: StruSR은 기호 회귀 과정에서 PINNs가 인코딩한 구조적 통찰력을 활용합니다. 이를 위해, 학습된 PINN의 출력에 대해 지역 Taylor 전개를 수행하며, 이는 도메인 지식을 기호 검색 과정에 주입하는 기반이 됩니다. 민감도 점수를 사용하여 각 서브 트리의 기여도를 정량화하고 기호 표현의 진화를 제어하는 QAQ(quantitative assessment of question) 메커니즘을 도입합니다.

- **Performance Highlights**: StruSR은 전통적인 기준보다 수렴 속도, 구조적 충실도 및 기호 표현의 해석 가능성을 개선하는 것으로 나타났습니다. 벤치마크 PDE 시스템을 기반으로 실시한 실험 결과, 이 프레임워크는 물리적 잔차와 기호적 정렬을 동시에 최적화하여, 기계 발견 모델들이 보다 해석 가능하고 물리적으로 의미 있는 표현을 생성하는 데 기여합니다.



### Unsupervised Backdoor Detection and Mitigation for Spiking Neural Networks (https://arxiv.org/abs/2510.06629)
Comments:
          To appear in The 28th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2025)

- **What's New**: 스파이킹 신경망(Spiking Neural Networks, SNNs)은 인공지능 분야에서 에너지 효율이 뛰어난 모델로 주목받고 있습니다. 본 논문에서는 SNNs의 보안 문제가 특히 백도어 공격(backdoor attacks)과 관련하여 분석되지 않았음을 지적하며, 기존의 방어 방법들이 SNNs에서는 효과적이지 않음을 설명합니다. 이를 바탕으로 새로운 프레임워크인 Temporal Membrane Potential Backdoor Detection (TMPBD)와 Neural Dendrites Suppression Backdoor Mitigation (NDSBM)을 제안합니다.

- **Technical Details**: TMPBD는 이벤트 주도적이고 시간에 의존하는 SNN의 특성을 고려하여, 공격 지식이나 데이터 접근 없이도 목표 레이블을 감지할 수 있는 비지도(unsupervised) 탐지 방법입니다. TMPBD는 최종 스파이킹 계층에서의 시계열 막전위(Temporal Membrane Potential, TMP)의 최대 여유 통계(maximum margin statistics)를 활용하여 백도어 공격을 탐지합니다. NDSBM은 초기 합성곱 층(convolutional layers) 간의 덴드라이트(dendrite) 연결을 조절하여 악의적인 뉴런을 억제하고 동시에 정상적인 행동을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 논문에서 제안한 TMPBD는 다양한 악의적인 백도어 공격에 대해 100%의 탐지 정확도를 달성하였으며, NDSBM은 공격 성공률(attacker success rate, ASR)을 100%에서 8.44%로 줄여줍니다. TMPBD와 NDSBM을 함께 사용할 경우, ASR은 평균적으로 2.81%로 감소하며, 클린 정확도(clean accuracy)를 저하시키지 않고도 막대한 성능 개선을 실현합니다. 이러한 실험 결과는 SNNs에서의 백도어 공격 방어 메커니즘의 필요성과 효과를 입증합니다.



### FEAorta: A Fully Automated Framework for Finite Element Analysis of the Aorta From 3D CT Images (https://arxiv.org/abs/2510.06621)
- **What's New**: 이 논문은 대동맥류(Thoracic Aortic Aneurysm)로 인한 사망 위험을 평가하기 위한 새로운 접근 방식을 소개합니다. 기존에는 복잡한 3D 재구성이 필요했으나, 본 연구에서는 3D CT 이미지로부터 직접 환자 맞춤형 유한 요소 메쉬(Finite Element Mesh)를 생성할 수 있는 최종 딥 뉴럴 네트워크(Deep Neural Network)를 개발했습니다. 이를 통해 현재의 기술적 한계를 극복하고 환자 개별 맞춤 솔루션을 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 논문에서 제안한 방법은 PyTorch FEA 라이브러리와 유한 요소 해석(Finite Element Analysis, FEA) 기능을 통합하여 사용합니다. 기존의 전통적인 FEA 시뮬레이션에 비해 계산 시간을 현저하게 줄일 수 있으며, статическое детерминирование(static determinacy) 원리를 적용해 각 케이스 당 약 3분으로 단축했습니다. 딥 뉴럴 네트워크와의 결합을 통해 계산 시간이 몇 초로 더욱 줄어들었습니다.

- **Performance Highlights**: 이 연구 결과는 대동맥류의 파열 위험을 더욱 신속하고 효율적으로 평가할 수 있는 혁신적 방법을 제시합니다. 기존 방식에 비해 계산 시간의 크게 단축됐다는 점은 임상적 적용의 가능성을 한층 높여줍니다. 마지막으로, 이를 통해 대규모 환자군에 대한 스케일링(scaling)을 용이하게 하고, 임상 세팅에서의 사용성을 극대화할 수 있습니다.



### Real-Time Glass Detection and Reprojection using Sensor Fusion Onboard Aerial Robots (https://arxiv.org/abs/2510.06518)
Comments:
          8 pages, 8 figures, submitted to ICRA 2026

- **What's New**: 이 연구는 300g 이하의 쿼드로터에서 투명한 장애물을 실시간으로 탐지하고 맵핑할 수 있는 새로운 시스템을 제안합니다. 기존의 방법들은 대개 비쌀 뿐만 아니라 고사양 하드웨어 요구 사항이 있어 저전력( low Size, Weight, and Power) 로봇에는 적합하지 않았습니다. 본 연구는 Time-of-Flight (ToF) 카메라와 초음파 센서를 활용한 경량화된 2D 컨볼루션 모델을 통합하여 투명 장애물을 효과적으로 감지하는 방식을 선보입니다.

- **Technical Details**: 이 시스템은 실시간으로 작동하면서도 CPU의 작은 부분만을 사용하여 처리되는 효율적인 프레임워크를 특징으로 합니다. 연구자들은 빛의 반사를 탐지하고 깊이를 배경의 빈 공간으로 재전파하여 투명 장애물을 가시화하는 데 중점을 두었습니다. 이 방법은 실험을 통해 유효성을 검증하였으며, 로봇이 유리로 된 실내 환경을 정확하게 맵핑하는 것을 보여주었습니다.

- **Performance Highlights**: 본 연구는 기존의 고가의 센서와 복잡한 알고리즘에 의존하지 않고, 작은 로봇에서도 무선으로 효과적으로 장애물을 탐지하고 맵핑할 수 있는 시스템을 최초로 제시했다고 할 수 있습니다. 이로 인해 로봇의 안전한 내비게이션이 가능해지며, 실시간 데이터 처리 능력을 통해 로봇의 동작 속도 또한 향상됩니다. 따라서, 저전력 시스템에 최적화된 경쟁력 있는 솔루션이 제공됩니다.



### Active Next-Best-View Optimization for Risk-Averse Path Planning (https://arxiv.org/abs/2510.06481)
- **What's New**: 이번 연구는 불확실한 환경에서 안전한 내비게이션을 위해 리스크 회피(risk-averse)와 적극적인 인식을 통합하는 계획 방법론을 제시합니다. 저자들은 3D Gaussian-splat Radiance Field에서 온라인으로 업데이트된 Average Value-at-Risk(AV@R) 통계를 기반으로 한 위험 맵을 생성하여, 안전하고 실행 가능한 경로를 생성하는 통합 프레임워크를 개발했습니다. 또한, 최적화 문제로 정의된 Next-Best-View(NBV) 선택을 도입하여, 로봇의 불확실성을 줄이고 안전한 경로 계획을 향상시킵니다.

- **Technical Details**: 연구에서는 3D 환경의 시각적 정보를 수집하는 로봇을 위해 리스크 회피 기반 경로 계획을 제안합니다. 초기 경로를 AV@R 기반의 보수적인 위험 맵을 활용하여 정제하고, 이에 기반하여 주변 환경에서 안전한 격자 포인트를 효과적으로 추출합니다. SE(3) 포즈 매니폴드에서 리만 그라디언트 하강법을 사용하여 확률적으로 정보를 최대화하는 알고리즘을 통해 NBV를 최적화함으로써, 신뢰도 높은 실시간 동작 계획을 가능하게 합니다.

- **Performance Highlights**: 제안된 프레임워크는 3D 환경에서의 안전한 내비게이션 및 정보 획득을 통합하는 효율적인 방법론으로 자리잡고 있습니다. 저자들은 이론적인 접근과 광범위한 컴퓨터 시뮬레이션을 통해 시스템의 효과성을 입증하였으며, 리스크 회피 경로 재계획 프레임워크와 NBV 계획의 융합이 로봇 내비게이션의 최신 기술을 발전시킬 수 있음을 보여주고 있습니다. 이로 인해 복잡한 환경에서의 안전한 경로 확립이 보다 용이해질 것으로 기대됩니다.



### Conditional Denoising Diffusion Model-Based Robust MR Image Reconstruction from Highly Undersampled Data (https://arxiv.org/abs/2510.06335)
- **What's New**: 이 연구에서는 MRI 이미지를 고품질로 재구성하기 위해 새로운 조건부 제거 확산(conditional denoising diffusion) 프레임워크를 제안합니다. 기존의 방법들과는 달리, 이 프레임워크는 모든 역 확산(reverse diffusion) 단계에서 측정 모델을 직접 통합하고, 쌍으로 구성된 데이터로 학습하도록 설계되었습니다. 이러한 하이브리드 설계는 생성적 유연성과 MRI 물리학의 명시적인 적용을 결합하여 재구성을 개선합니다.

- **Technical Details**: 제안된 프레임워크는 역 샘플링 과정 중에 데이터 충실도(data fidelity) 항을 도입함으로써 재구성된 이미지와 원래 MRI 데이터 간의 일관성을 보장합니다. 이 접근 방식은 기존의 확산 모델들이 데이터 일관성을 별도의 후처리 단계로 적용하던 것과는 달리, 매개변수 조정(optimization) 단계를 통해 데이터 일관성을 유지하면서 제거 과정을 수행합니다. 이를 통해 고해상도 MRI 이미지를 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과 fastMRI 데이터셋에서 제안된 방법은 SSIM, PSNR, LPIPS 측면에서 기존의 최신 딥 러닝 및 확산 기반 방법들보다 일관되게 우수한 성능을 보였습니다. 특히 LPIPS는 시각적 개선 사항을 보다 정확하게 캡처하여, 픽셀 수준의 충실도(pixel-level fidelity)와 시각적 현실감(perceptual realism)의 개선을 나타냅니다. 이를 통해 제안된 방법이 MRI 재구성의 강건함과 신뢰성을 크게 향상시킨 것을 확인할 수 있습니다.



### On knot detection via picture recognition (https://arxiv.org/abs/2510.06284)
Comments:
          21 pages, many figures, comments welcome

- **What's New**: 이번 연구의 목표는 사진으로 촬영한 매듭(knot)을 자동으로 인식하는 것입니다. 현대 머신러닝 기법인 컨볼루션 신경망(convolutional neural networks)과 트랜스포머(transformers), 그리고 전통적인 알고리즘을 사용하여 이를 근사하는 전략을 설명합니다. 특히, 이미지를 기반으로 교차 수(crossing number)를 예측하는 간단한 기준선을 제시하여 경량화된 CNN과 트랜스포머 아키텍처가 구조적 정보를 회복할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 매듭의 양자 불변량(quantum invariants)인 존스 다항식(Jones polynomial)을 계산하기 위해 전통적인 알고리즘과 현대 머신러닝 기법의 혼합을 사용합니다. 관찰 모듈(perception modules)을 사용해 직접 이미지를 분석하여 단계를 나누어 평면 다이어그램(Planar Diagram, PD) 코드로 기호화된 재구성을 결합할 계획입니다. 이 두 단계의 접근 방식은 노이즈가 있는 시각적 데이터를 처리하는 머신러닝과 엄격한 위상 구분을 적용하는 불변량 간의 상호 보완성을 강조합니다.

- **Performance Highlights**: 이 연구는 매듭 분류를 위한 강력한 도구를 개발하는 장기 목표를 가지고 있습니다. 초기 실험 결과, 간단한 CNN 및 트랜스포머 아키텍처가 유의미한 결과를 도출할 수 있음을 보였으며, 이는 향후 매듭의 정확한 분류를 위한 기초를 제공합니다. 또한, 이러한 접근 방식은 머신러닝과 전통적인 방법론 간의 시너지를 통해 보다 강력하고 신뢰할 수 있는 평가 시스템을 구축할 가능성을 제시합니다.



### SER-Diff: Synthetic Error Replay Diffusion for Incremental Brain Tumor Segmentation (https://arxiv.org/abs/2510.06283)
- **What's New**: 본 논문은 Synthetic Error Replay Diffusion (SER-Diff)라는 새로운 프레임워크를 제안합니다. 이는 기존의 Incremental Learning과 디퓨전(Noise Diffusion) 모델의 결합을 통해 뇌 종양 분할에서의 재학습 문제를 해결하고자 합니다. SER-Diff는 과거 작업으로부터 생성된 합성 오류 맵을 사용하여 새로운 작업의 학습 중에 재생(Replayed)하게 됩니다. 이 방식은 뛰어난 성능을 발휘하며, 이전의 catastrophic forgetting 문제를 완화합니다.

- **Technical Details**: SER-Diff는 세 가지 핵심 구성 요소로 구성됩니다: 1) Synthetic Error Replay 메커니즘을 통해 동결된 teacher diffusion 모델이 생성한 오류 맵을 재생합니다. 2) Diffusion 기반 정제 과정을 통해 이 오류 맵을 활용하여 분할의 일관성을 향상시킵니다. 3) Dual-loss(training objective) 전략을 활용하여 새로운 데이터에 대한 적응성과 이전에 습득한 지식의 보존을 동시에 보장합니다.

- **Performance Highlights**: 실험 결과, SER-Diff는 BraTS2020, BraTS2021, BraTS2023 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다. Dice 점수는 각각 95.8%, 94.9%, 94.6%로 최대치를 기록하였으며, HD95 값 또한 각각 4.4 mm, 4.7 mm, 4.9 mm로 최소치를 달성했습니다. 이러한 결과는 SER-Diff가 재학습 없이도 더 정확하고 해부학적으로 일관된 분할을 수행할 수 있음을 보여줍니다.



### Surgeons Are Indian Males and Speech Therapists Are White Females: Auditing Biases in Vision-Language Models for Healthcare Professionals (https://arxiv.org/abs/2510.06280)
- **What's New**: 본 연구는 의료 분야에서 비전 언어 모델(Vision Language Models, VLMs)의 직업별 편향을 평가하기 위한 프로토콜을 제시합니다. 의료 직군과 관련된 다양한 성별 및 인구통계적 편향을 정량화하고 운영 위험을 평가하는 방법론을 개발하였습니다. 이를 통해 기존의 VLM들이 어떻게 유사한 편향을 재생산하는지를 체계적으로 분석할 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 의사 및 의료 관련 직군에 대한 구조화된 분류법을 정의하고, 모델의 행동을 탐구하기 위한 직업 인식 프롬프트 스위트를 구성하며, 공정한 얼굴 데이터셋(FairFace)에 대한 인구통계적 편향 평가를 수행합니다. CLIP 및 OpenCLIP 모델 패밀리를 활용해 다양한 역할에 대한 성 편향을 평가하며, JS Divergence 기반 편향 점수를 사용하여 이들 모델의 차이를 분석합니다.

- **Performance Highlights**: 이 연구는 여러 VLM 모델에서 일관된 성 편향이 존재하는 것을 관찰하고, 이러한 편향이 AI 기반의 인력 채용 및 노동력 분석에 미치는 잠재적 영향을 강조합니다. 의료 환경에서 발생하는 비즈니스 의사결정과 환자 신뢰를 위한 편향 탐지의 중요성을 명확히 하며, 궁극적으로는 민감한 분야에서의 AI 사용에 대한 공정성을 증진시키기 위한 기반이 될 것입니다.



### A Total Variation Regularized Framework for Epilepsy-Related MRI Image Segmentation (https://arxiv.org/abs/2510.06276)
- **What's New**: 이번 논문에서는 약물 저항성 간질의 주요 원인인 Focal Cortical Dysplasia (FCD) 지역의 정확한 분할(segmentation) 작업을 위한 새로운 프레임워크를 제안하고 있습니다. 이 프레임워크는 최신 transformer-enhanced encoder-decoder 아키텍처를 채택하며, Dice loss와 anisotropic Total Variation (TV) 항을 결합한 새로운 손실 함수(loss function)를 도입합니다. 이러한 통합은 공간적인 부드러움(spatial smoothness)을 촉진하고, 포스트 프로세싱(post-processing)에 의존하지 않으면서 잘못된 양성 군집(false positive clusters)을 줄이는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법론은 3D 뇌 MRI 이미지에서 FCD 지역을 효과적으로 분할하기 위한 교육 파이프라인을 주의 깊게 설계합니다. 이 파이프라인은 패치 단위 샘플링(patch-wise sampling)과 바닐라 분류(voxel-wise classification)를 기반으로 하여 적은 수의 고차원 데이터로부터 모델이 효과적으로 학습할 수 있도록 합니다. 새로운 손실 함수는 Total Variation (TV) 정규화 항을 추가하여 이웃 바닥 예측의 급격한 변화를 패널티(penalize) 함으로써 보다 부드럽고 해부학적으로 일관된(segmentation masks) 분할 마스크를 생성하도록 유도합니다.

- **Performance Highlights**: 이 연구에서 제안된 모델은 공공 FCD 데이터셋(85명의 간질 환자로 구성)에서 성능을 평가하였으며, 기존의 손실 함수들과 비교하여 우수한 분할 정확도와 일관성을 보였습니다. 제안된 TV 손실 함수가 적용된 모델은 Dice 계수에서 11.9% 향상된 성능을 나타내고, 기준 모델보다 13.3% 더 높은 정밀도(precision)를 기록했습니다. 또한, 잘못된 양성 군집의 수는 61.6% 감소하여 더욱 효과적인 진단이 가능함을 보여주었습니다.



### Stacked Regression using Off-the-shelf, Stimulus-tuned and Fine-tuned Neural Networks for Predicting fMRI Brain Responses to Movies (Algonauts 2025 Report) (https://arxiv.org/abs/2510.06235)
- **What's New**: Algonauts 2025 Challenge에 대한 새로운 접근 방식을 소개합니다. 이 연구에서는 대규모 언어 모델, 비디오 인코더, 오디오 모델, 비전-언어 모델의 멀티모달 표현을 통합하여 영화 자극에 대한 fMRI 뇌 반응을 예측하였습니다. 세밀한 전사(transcripts)와 요약을 통해 텍스트 입력을 향상시켰고, 자극 조정(stimulus-tuning) 및 미세 조정(fine-tuning) 전략을 실험하였습니다. 우리 팀의 제출 결과는 27개 제출 중 10위를 기록하였고, 모든 코드와 리소스를 공개하여 멀티모달 인코딩 모델 개발에 기여하였습니다.

- **Technical Details**: 본 연구에서는 영화 자극에 따른 뇌 활동을 예측하기 위해 다양한 전략을 적용하였습니다. 주로, 사전 훈련된 심층 신경망의 내부(x) 표현을 이용하여 선형 모델을 적합시키고, 비전-언어 모델에서 더 풍부한 표현을 추출하기 위해 전사를 개선하였습니다. 또한, 대규모 언어 모델 및 비전 모델을 자극에 맞게 조정하여 뇌 예측 정확도를 향상시키는 실험도 진행하였습니다. 최종적으로 세 개의 예측 소스를 스택 리그레션(stacked regression)을 통해 결합하여 결과를 도출하였습니다.

- **Performance Highlights**: 우리 팀의 최종 모형은 whisper-small과 Llama-3.1-8B를 포함한 두 개의 사전 훈련된 심층 신경망에 기반한 선형 예측과 향상된 전사에서 추출된 InternVL 표현을 사용하여 성능을 평가하였습니다. 이 조합을 통해 뇌 활동 예측에서 높은 정확도를 나타냈으며, 시험 데이터에 대해 상위 성능을 달성하였습니다. 궁극적으로, 본 연구의 접근 방식은 다양한 모델 기준에서 효과를 보여주었고, Algonauts 대회에서 좋은 성과를 기록하였습니다.



New uploads on arXiv(cs.AI)

### Agentic generative AI for media content discovery at the national football leagu (https://arxiv.org/abs/2510.07297)
Comments:
          13 pages, 7 figures, International Sports Analytics Conference and Exhibition

- **What's New**: 이 논문에서는 NFL과 협력하여 생성 AI를 기반으로 한 워크플로우가 미디어 연구자와 분석가들이 전통적인 필터-클릭 인터페이스 대신 자연어를 통해 역사적인 플레이를 검색할 수 있게 해준다는 점을 강조합니다. 이 에이전트적인 워크플로우는 사용자 쿼리를 입력받아 요소로 나누고, 이를 기반 데이터베이스 쿼리 언어로 변환하는 역할을 수행합니다. 설계된 세멘틱 캐싱(semantic caching)을 통해 정확도와 응답 지연(latency)을 향상시키며, NFL의 운영 효율성을 높이는 데 기여하고 있습니다.

- **Technical Details**: NFL의 Next Gen Stats(NGS) 플랫폼은 2016년 이후 모든 NFL 경기의 플레이에 대한 실시간 데이터를 기록하는 포괄적인 시스템입니다. 이러한 데이터는 플레이어의 패드와 공에 장착된 센서를 통해 수집되어, 다양한 통계 및 고급 데이터와 함께 저장됩니다. 이 연구팀은 NGS 플랫폼의 통계적 데이터와 주제 전문가(SMEs)와의 협력을 통해 미디어 검색 솔루션 구축에 필요한 질문-답변(QA) 쌍을 준비하고 검증하는 과정을 진행했습니다.

- **Performance Highlights**: 제안된 시스템은 관련 비디오 검색 시간을 평균 10분에서 30초로 단축시키며, 95% 이상의 정확성으로 정보를 검색합니다. 이로 인해 미디어 팀은 콘텐츠 제작 및 흥미로운 스토리라인 생성에 더 많은 시간과 자원을 투자할 수 있게 됩니다. 결과적으로 이 시스템은 NFL의 콘텐츠 관리 및 검색 작업을 디지털 환경에서 획기적으로 개선했습니다.



### Multi-Objective Multi-Agent Path Finding with Lexicographic Cost Preferences (https://arxiv.org/abs/2510.07276)
Comments:
          8 pages, 7 figures

- **What's New**: 이 논문은 여러 목표를 가진 다수의 에이전트가 공유 환경에서 협조적으로 작업할 수 있는 새로운 방식인 lexicographic MO-MAPF 프레임워크를 소개합니다. 이 방법은 에이전트 경로 탐색에서 파레토 전선(Pareto frontier)을 구축하지 않고도 사용자 정의 선호를 직접 반영하여 하나의 최적 경로를 도출하는 Lexicographic Conflict-Based Search (LCBS) 알고리즘을 제안합니다. LCBS는 우선 순위를 고려한 A* 탐색을 활용하여 보다 효율적으로 계획을 수립할 수 있도록 합니다.

- **Technical Details**: LCBS는 Conflict-Based Search (CBS) 프레임워크를 기반으로 하여 구축된 알고리즘입니다. 이 알고리즘은 각 목표에 대한 우선 순위를 따르면서 에이전트를 위한 경로를 순차적으로 최적화하기 위해 레지코그래픽 A* (LA*)를 사용합니다. 고수준 플래너는 제약 트리를 관리하여, 충돌이 발생할 경우 이를 해결하기 위해 자식 노드로 분기하고, 저수준 플래너는 레지코그래픽 순서에 따른 최적의 경로를 도출합니다.

- **Performance Highlights**: LCBS는 기존의 MO-MAPF 방법들과 비교할 때 최적의 솔루션을 제공하며 최대 10개의 목표와 35개의 에이전트를 포함한 경우에도 높은 확장성을 보여줍니다. 여러 표준 및 랜덤화된 MAPF 벤치마크에서 이 알고리즘은 최신 기술 기반선(state-of-the-art baselines)보다 일관되게 높은 성공률을 기록했습니다. 특히 목표 수가 증가할수록 그 성능이 더욱 두드러지게 나타났습니다.



### NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents (https://arxiv.org/abs/2510.07172)
Comments:
          60 pages, 18 figures, 13 tables

- **What's New**: 본 논문은 과학 법칙 발견을 위한 새로운 벤치마크인 NewtonBench를 소개합니다. 이 벤치마크는 12개의 물리 영역에 걸쳐 324개의 과학 법칙 발견 작업을 포함하고 있으며, 기존 벤치마크의 한계를 해결하고 있습니다. 특히, 기존의 방법론적 트릴레마를 완화하기 위해 메타피지컬 시프트(metaphysical shift)를 활용하여 과학적으로 관련 있고 메모리제이션 저항이 있는 문제들을 생성합니다.

- **Technical Details**: NewtonBench는 두 가지 핵심 원칙을 바탕으로 설계되었습니다. 첫 번째 원칙은 메타피지컬 시프트로, 이는 기존의 물리 법칙의 수학적 구조를 체계적으로 변경하여 새로운 문제들을 생성하는 것입니다. 두 번째 원칙은 정적 함수 발견(static function discovery)의 한계를 극복하기 위해, 상호작용형 모델 발견(interactive model discovery) 환경을 제공하여 에이전트가 실험을 설계하고 탐색할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 LLM이 과학 법칙 발견에서 명확하지만 취약한 능력을 가지고 있음을 보여줍니다. 최첨단 모델은 단순한 시스템에서는 우수한 성능을 보이지만, 시스템의 복잡성이 증가함에 따라 성능 저하가 급격히 일어납니다. 또한 도구 지원이 주는 역설적인 효과를 밝혀, 월등한 모델이 도구에 과도하게 의존하면서 최적의 법칙 발견에 실패할 수 있음을 보여줍니다.



### Integrating Domain Knowledge into Process Discovery Using Large Language Models (https://arxiv.org/abs/2510.07161)
Comments:
          This paper is currently under review for publication in a journal

- **What's New**: 이번 논문에서는 이벤트 로그에서 과정 모델을 도출하는 과정 발견(process discovery)에 대해 다루고 있습니다. 기존 모델 도출 방식에서는 이벤트 로그의 불완전함이나 노이즈의 영향을 받으며, 도메인 지식(domain knowledge)을 반영하지 않는 문제가 있었습니다. 이를 해결하기 위해 자연어로 표현된 도메인 지식을 통합하는 대화형(framework) 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 Large Language Models(LLMs)를 활용하여 도메인 전문가가 제공하는 텍스트 설명에서 선언적 규칙(declarative rules)을 추출합니다. 이러한 규칙은 IMr 발견 알고리즘(discovery algorithm)을 안내하는 데 사용되며, 이벤트 로그와 추출된 규칙의 통찰력을 결합하여 프로세스 모델을 재귀적으로 구성합니다. 이 과정은 도메인 지식과 모순되는 문제적 프로세스 구조를 피하는 데 도움을 줍니다.

- **Performance Highlights**: 제안하는 프레임워크는 LLM, 도메인 전문가, 백엔드 서비스 간의 상호작용을 조정합니다. 이 논문은 해당 워크플로우를 지원하는 완전 구현된 도구를 제시하며, 여러 LLM 및 프롬프트 엔지니어링(prompt engineering) 전략에 대한 광범위한 평가를 수행하였습니다. 실제 이벤트 로그 기반 사례를 통해 도메인 전문가들이 프레임워크의 사용성과 효과성을 평가한 경험적 연구도 포함되어 있습니다.



### The Contingencies of Physical Embodiment Allow for Open-Endedness and Car (https://arxiv.org/abs/2510.07117)
Comments:
          15 pages, 1 figure

- **What's New**: 본 논문은 인공지능 에이전트의 신체적 구현에서의 취약성과 사망률을 탐구합니다. 이를 위해, Martin Heidegger의 존재론적 개념인 ‘이 세계에 존재하기 (being-in-the-world)’와 ‘죽음을 향한 존재하기 (being-towards-death)’를 기반으로 두 가지 최소 조건을 정의합니다. 이러한 조건을 통해 에이전트가 생리적 안정을 유지하고 죽음을 피하는 내재적 동기를 발전시킬 수 있음을 제시합니다.

- **Technical Details**: 논문은 에이전트를 환경의 상태를 감지하고 행동 정책에 따라 행동을 실현하는 주체로 정의합니다. 이들은 부분적으로 관찰 가능한 Markov 결정 과정 (POMDP) 기반에서 환경과 상호작용합니다. 에이전트의 센서, 정책, 액추에이터가 환경의 상태와 내재적으로 연결되며, 이를 통해 신체적 구현의 복잡성을 보다 풍부하게 탐구합니다.

- **Performance Highlights**: 저자들은 생리적 요구를 충족하기 위한 내재적 동기와 자율성을 극대화하는 방법을 강화 학습 프레임워크 내에서 분석합니다. 이 방법은 에이전트가 복잡한 다중 에이전트 환경에서 스스로를 잘 유지하고 돌보는 능력을 배양할 수 있도록 합니다. 향후 연구는 이러한 에이전트들이 상대방을 이해하고 돌보는 데 기여할 수 있는 가능성에 대해 살펴볼 예정입니다.



### The Cognitive Bandwidth Bottleneck: Shifting Long-Horizon Agent from Planning with Actions to Planning with Schemas (https://arxiv.org/abs/2510.07091)
Comments:
          22 pages

- **What's New**: 이 논문에서는 오픈 월드 자율성을 위해 장기적인 계획과 여러 상호작용이 필요한 장기 과제 작업에서 대형 언어 모델(LLM)의 효과적인 작동을 가능하게 하는 방법을 제안합니다. 전통적인 방법은 실행 가능한 행동 목록을 제공하여 계획하지만, 환경 행동 공간이 조합적으로 폭발하는 경우 이러한 접근법은 비현실적일 수 있습니다. 이 연구는 행동 표현 방식의 최적화된 선택을 위한 시스템적 연구로, 행동 기반 계획(PwA)과 스키마 기반 계획(PwS)의 두 가지 접근 방식을 비교합니다.

- **Technical Details**: 인간은 추상적인 행동 템플릿을 구체적인 실행 가능한 단계로 변환하여 결정을 내리는 과정을 겪습니다. 본 논문에서는 이러한 과정을 스키마 기반 계획(PwS)이라고 명명하고, LLM의 최신 발전을 통해 장기 과제를 해결하기 위한 자율 에이전트의 필요성을 강조합니다. 연구 결과, PwA 방식은 짧은 행동 공간에서 더 나은 성능을 보이는 반면, 긴 행동 공간에서는 PwS 방식이 유리하다는 것을 보여주고 있습니다.

- **Performance Highlights**: 행동 공간의 크기에 따라 성능 변화가 관찰되었으며, 인플렉션 포인트는 ALFWorld(~35 actions)에서 PwA가 평균 33.4% 더 높은 성능을 보였지만, SciWorld(~500 actions)에서는 PwS가 8.1%의 우위를 차지했습니다. 이는 PwA가 행동 공간의 크기에 따라 성능이 저하되는 반면, PwS는 확장 가능성에서 더 우수함을 나타냅니다. 마지막으로 이 연구는 모델의 효율성을 높이기 위한 구체적인 가이드를 제공합니다.



### VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems (https://arxiv.org/abs/2510.07073)
- **What's New**: VRPAgent는 LLM(대형 언어 모델)을 활용하여 차량 경로 최적화 문제(VRP)에 대한 휴리스틱(heuristic) 생성 능력을 향상시키는 새로운 프레임워크입니다. 기존의 연구에서는 인간 전문가가 설계한 휴리스틱을 능가하지 못했지만, VRPAgent는 문제 특정 작업자(operator)를 LLM을 통해 생성하고, 이를 유전자 탐색(genetic search) 기법으로 개선하여 성과를 거두고 있습니다. VRPAgent는 단일 CPU 코어만으로도 복잡한 문제에서 우수한 성과를 보입니다.

- **Technical Details**: VRPAgent는 대규모 이웃 탐색(LNS) 메타휴리스틱을 기반으로 하며, LLM을 이용해 문제 특정한 휴리스틱 작업자를 생성합니다. 이 과정에서 유전자 알고리즘(GA)을 사용하여 휴리스틱의 품질을 지속적으로 개선하며, 이를 생산 비용 절감을 위해 코드 길이 패널티를 도입하여 LLM의 추론 비용을 낮춥니다. 이러한 메타휴리스틱 구조 덕분에 복잡한 문제에서도 효율적으로 작업자를 생성하고 뛰어난 성능을 발휘할 수 있습니다.

- **Performance Highlights**: VRPAgent는 용량 제한이 있는 차량 경로 문제(CVRP), 시간 창이 있는 차량 경로 문제(VRPTW), 상금 수집 차량 경로 문제(PCVRP) 등 여러 문제에서 인간 전문가가 설계한 휴리스틱보다 훨씬 더 우수한 성능을 보였습니다. 이러한 접근법은 학계에서 자유롭게 사용할 수 있는 LLM을 활용하여 최첨단 성과를 높인 최초의 시도로 여겨집니다. VRPAgent는 자동화된 휴리스틱 발견의 미래에 대한 가능성을 열어줍니다.



### Inductive Learning for Possibilistic Logic Programs Under Stable Models (https://arxiv.org/abs/2510.07069)
Comments:
          Under consideration in Theory and Practice of Logic Programming (TPLP)

- **What's New**: 이번 논문에서는 가능론적 논리 프로그램(poss-programs)의 유도 추론 문제를 다루고 있습니다. 기존 연구에서는 가능론적 안정 모델의 의미와 속성에 대한 조사가 이루어졌지만, 유도 추론에 관한 접근법은 새롭게 제안됩니다. 이 연구는 배경 프로그램 및 가능한 안정 모델의 일부인 예시로부터 poss-programs를 추출하는 방법을 제시합니다.

- **Technical Details**: 이 논문에서는 먼저 유도 작업(induction tasks)의 개념을 형식적으로 정의하고, 이와 관련된 속성들을 조사합니다. 두 개의 알고리즘인 ilpsm과 ilpsmmin이 유도 해법을 계산하기 위해 제시되며, ilpsmmin의 구현도 제공됩니다. 실험 결과에 따르면, 일반 논리 프로그램을 입력으로 받을 때, 이 프로토타입이 기존의 주요 유도 학습 시스템보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: 특히, 연구에서 제안된 알고리즘은 안정 모델을 기반으로 한 데이터셋에서 더 많은 정확도를 달성합니다. 이는 낮은 오류와 높은 정확도로 유도 해법을 제공하여 가능론적 논리 프로그램의 향상된 적합성을 나타냅니다. 재미있는 점은 이 작업이 임상 전문가 시스템의 지식을 활용하여 실제 적용 가능성을 보인다는 것입니다.



### Prompt Optimization Across Multiple Agents for Representing Diverse Human Populations (https://arxiv.org/abs/2510.07064)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 한계를 극복하기 위해 다양한 인간 행동을 대변하는 대리 모델(agent)을 구축하는 새로운 접근 방식을 제안합니다. 기존의 연구들은 단일 LLM 모델이 다양한 인간의 다각적인 관점을 포착하지 못한다는 문제를 지적했습니다. 이에 본 연구는 여러 대리 모델을 조합함으로써 더욱 풍부한 인간 대표성을 구현하고자 합니다.

- **Technical Details**: 이 연구에서는 하위 모듈 최적화(submodular optimization) 기법을 통해 다양한 대리 모델을 효과적으로 선택하는 방법을 제안합니다. 각 대리 모델은 소수의 인간 데모(task-response pair)에 의해 조정되며, 이들은 인-컨텍스트 학습(in-context learning)을 통해 각각의 행동을 형성합니다. 이를 통해, 제안된 방법은 시간 복잡도와 성능 보장 간의 다양한 트레이드오프를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 교육 및 크라우드소싱 분야에서 인간 인구를 보다 효과적으로 대표하는 모델을 구축하는 데 성공했습니다. 새로운 작업에 대한 행동 분석에서 이러한 대리 모델은 학생들과 주석가들이 나타내는 행동 패턴 및 관점을 재현하며, 이는 모델의 다각적인 적용 가능성을 시사합니다.



### Tool-Augmented Policy Optimization: Synergizing Reasoning and Adaptive Tool Use with Reinforcement Learning (https://arxiv.org/abs/2510.07038)
- **What's New**: 소논문에서 소개된 Tool-Augmented Policy Optimization (TAPO)은 대규모 언어 모델(LLMs)의 문제 해결 능력을 향상시키기 위해 고안된 새로운 강화 학습 프레임워크입니다. TAPO는 단계별 추론(Multi-hop reasoning)과 도구 호출(tool-calling)을 통합하여, 복잡한 수학적 작업에 필요한 외부 지식 및 계산 도구의 활용을 지원합니다. 이 연구는 TAPO-easy-60K와 TAPO-hard-18K라는 두 개의 새로운 데이터셋을 도입하여 모델의 훈련과 평가를 용이하게 합니다.

- **Technical Details**: TAPO 프레임워크는 Dynamic Sampling Policy Optimization (DAPO)을 기반으로 하며, 검색 API 및 Python 인터프리터와 같은 도구 사용을 위한 고급 기능을 제공합니다. 제안된 시스템은 XML과 유사한 구조 형식을 사용하여 <think>, <search>, <code>, <response>, <answer> 태그로 구성된 구조화된 프롬프트 디자인을 구현합니다. TAPO는 멀티 스텝 추론과 향상된 도구 호출 기능을 통해 모델에 통합된 접근 방식을 가능하게 합니다.

- **Performance Highlights**: Qwen2.5-3B 및 Qwen2.5-7B 모델에서 수행된 실험은 TAPO의 효과성을 입증하며, 두 모델 모두 외부 지식 및 수학적 계산을 요구하는 작업에서 최첨단 성과를 달성합니다. TAPO는 기존 방법들보다 효율적인 도구 활용을 보장하고, 보상 해킹(reward hacking)으로 인한 과도한 도구 호출을 방지합니다. 이러한 결과는 고급 추론과 도구 사용의 결합이 지식 집약적 및 계산적으로 요구되는 작업에서 모델 성능을 향상시킬 잠재력을 강조합니다.



### Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces (https://arxiv.org/abs/2510.06953)
- **What's New**: 이번 연구에서는 Uniform Information Density (UID) 가설을 바탕으로 대형 언어 모델(LLM)의 추론 품질과 정보 전달의 균일성을 분석합니다. 새로운 엔트로피 기반 단계적 정보 밀도 메트릭을 제안하며, 지역적 및 전역적 균일성 점수를 도입합니다. 실험 결과, 단계별 균일성이 우수한 이론적 시각을 제공하는 것뿐만 아니라, 실제 성능을 개선하는 데도 기여하며, 추론 추적에서 정보 밀도 균일성이 신뢰할 수 있는 평가 기준임을 입증합니다.

- **Technical Details**: 연구에서는 언어를 신호로 간주하고, 이 신호를 제한된 용량의 노이즈 채널을 통해 전송하는 UID 가설을 모델링합니다. 각 말(u) 단위의 예상치 못한 정도를 나타내는 갑작스러운 예기치 못한 사건(surprisal) 개념을 정의하고, 이를 토대로 정보를 단계별로 측정하는 새로운 방법을 개발했습니다. 실험에서는 LLM이 생성하는 사고 과정의 정보 흐름을 분석하고, 지역적 및 전역적 균일성을 스텝 기반 정보 밀도를 통해 평가합니다.

- **Performance Highlights**: 연구 결과, 정보 밀도의 균일성이 추론 품질을 예측하는 데 중요한 요소임을 보여줍니다. LLM의 정확도가 10~32% 향상되는 것으로 나타났으며, 이는 LLM의 적절한 정보 균일성을 유지하는 것이 중요하다는 것을 강조합니다. 올바른 추론 추적은 급격한 정보 밀도 상승을 피하는 경향이 있으며, 이는 더 신뢰할 수 있는 추론 시스템 구축에 있어 균일성이 중요한 진단 기준임을 시사합니다.



### LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN (https://arxiv.org/abs/2510.06911)
- **What's New**: AJAN 프레임워크는 멀티 에이전트 응용 프로그램 개발을 위한 여러 시맨틱 웹 표준을 기반으로 하는 시스템을 제공합니다. 본 논문에서는 AJAN 에이전트를 모델링하는데 필요한 복잡한 RDF와 SPARQL의 정의를 보다 쉽게 수행할 수 있도록 돕는 웹 기반 그래픽 편집기(AJAN-Editor)를 소개합니다. 이 편집기는 현대 IDE의 핵심 기능과 LLM(대형 언어 모델)을 통합하여 사용자 친화적인 환경을 제공함으로써 비전문가도 쉽게 에이전트 모델링에 참여할 수 있도록 합니다.

- **Technical Details**: AJAN-Editor는 Ember.js, Node.js 및 Cytoscape.js를 기반으로 개발되어 RDF 트리플 스토어와 AJAN 서비스에 연결됩니다. 이 도구는 사용자가 RDF로 정의된 에이전트 템플릿을 통해 에이전트를 설정, 인스턴스화, 수정 및 관리할 수 있도록 돕습니다. 에이전트 행위의 정의는 SPARQL 쿼리를 통해 다이내믹하게 구성되며, 실시간 모니터링 및 디버깅 기능이 포함되어 있습니다.

- **Performance Highlights**: AJAN-Editor는 실시간 데이터 시각화와 쿼리 결과를 표 형태 및 그래픽 형태로 보여주는 기능을 통해 사용자의 경험을 향상시킵니다. 특히, 사용자는 AJAN의 에이전트와의 상호작용을 통해 행동 생성, SPARQL 쿼리 수행 등의 작업을 자연어로 진행할 수 있어 보다 효율적인 개발 환경을 제공합니다. 이와 함께, 다양한 도구와 협업 기능이 지원되어 팀 간의 협력 작업도 원활하게 수행할 수 있습니다.



### TGPR: Tree-Guided Policy Refinement for Robust Self-Debugging of LLMs (https://arxiv.org/abs/2510.06878)
- **What's New**: 이번 논문에서는 새로운 프레임워크인 Tree-Guided Policy Refinement (TGPR)을 제안합니다. TGPR은 GRPO와 Thompson Sampling 기반의 트리 탐색을 결합하여, 언어 모델이 코드 디버깅을 보다 효율적으로 수행하도록 돕는 방법입니다. 이 접근법은 성공적인 경로와 실패한 경로를 모두 탐구하며, 보다 밀집된 훈련 경로와 적응형 정책을 생성합니다.

- **Technical Details**: TGPR 프레임워크는 기존의 GRPO 알고리즘을 개선하여 효과적인 코드를 탐색하고 수정하는 데 집중합니다. 모델 학습을 위한 데이터 증강 엔진으로서, 트리 탐색은 테스트 시 추론에 사용되는 것이 아니라 훈련 중에 더 다양한 학습 경로를 생성하는 데 활용됩니다. 이 프레임워크는 정책 모델과 보상 모델을 결합해 고품질의 훈련 경로를 생성하고, 성공과 실패를 모두 학습할 수 있는 기회를 제공합니다.

- **Performance Highlights**: TGPR은 HumanEval, MBPP 및 APPS 벤치마크를 통해 강력한 GRPO 기준 모델에 비해 pass@1에서 +4.2% 포인트, pass@10에서 +12.5% 포인트의 성능 향상을 달성했습니다. 이는 코드 생성 및 디버깅에서 더 효과적이고 다양한 학습 경로를 통해 언어 모델의 디버깅 전략을 강화할 수 있음을 보여줍니다.



### Autoformalizer with Tool Feedback (https://arxiv.org/abs/2510.06857)
- **What's New**: 최근 연구는 Autoformalization(자동 형식화)가 Automated Theorem Proving(ATP, 자동 정리 증명)의 데이터 부족 문제를 해결하는 데 크게 기여하고 있음에 주목하고 있습니다. 기존의 접근 방식과 달리, Autoformalizer with Tool Feedback (ATF)는 도구 피드백을 통합하여 더 정확하고 일관된 형식적 진술을 생성하는 새로운 접근 방식을 제시합니다. 이 방법은 특정 도구에 의해 구문 검증과 의미 일관성을 보정하여 자동 형식화 품질을 크게 향상시킵니다.

- **Technical Details**: ATF는 구문 유효성과 의미 일관성을 반영하는 도구를 통합하여 형식화 과정에서 피드백을 기반으로 모델이 생성한 진술을 조정하도록 설계되었습니다. 특히, Lean 4 컴파일러를 사용하여 문법 수정을 위한 피드백을 제공하고, 다수의 LLM을 활용하여 번역된 문장 간의 의미 일치를 검증하는 알고리즘을 구현합니다. 이 접근법은 모델이 다양한 형식 언어 버전에 적응할 수 있도록 지원하며, 이메일은 트레이닝 과정에서 차가운 시작, 전문가 반복, 직접 선호 최적화(Direct Preference Optimization)라는 세 가지 단계로 구성됩니다.

- **Performance Highlights**: ATF는 기존의 여러 정형화 모델들과 비교했을 때, 성능 면에서 뛰어난 결과를 나타냅니다. 예를 들어, CombiBench에서의 의미 일관성에서 29.13% 개선을 보여주며, 이는 최상급 Goedel-V2-Formalizer-32B와 비교했을 때보다도 우수한 것입니다. 또한, ATF는 향후 연구에 기여할 수 있는 750,000개의 합성 정형 진술을 포함하는公开 데이터셋인 Numina-ATF를 오픈소스 형태로 제공하여, 자동 형식화와 ATP 연구의 발전을 지원할 것입니다.



### Evolving and Executing Research Plans via Double-Loop Multi-Agent Collaboration (https://arxiv.org/abs/2510.06761)
- **What's New**: 이 논문에서는 연구 문제를 자동으로 해결하기 위한 새로운 Double-Loop Multi-Agent (DLMA) 프레임워크를 제안합니다. 이 프레임워크는 두 가지 루프로 구성되어 있으며, 교수 에이전트들이 주도하는 리더 루프는 연구 계획을 발전시키고, 박사 과정 학생 에이전트들이 참여하는 팔로워 루프는 이 계획의 실행을 담당합니다. DLMA는 연구 제안서를 반복적으로 생성하고 조정하여 솔루션 공간을 효과적으로 탐색하는 데 중점을 둡니다.

- **Technical Details**: DLMA 프레임워크는 이층 최적화 문제(bilevel optimization problem)을 해결하기 위해 설계되었습니다. 리더 루프에서는 교수 에이전트들이 초기 연구 계획을 생성하고, 세 가지 유형의 회의(참여 회의, 개선 회의, 통합 회의)를 통해 제안서를 개발합니다. 팔로워 루프는 박사 과정 학생 에이전트들이 최적의 계획을 실행하며, 실행 과정에서 회의를 통해 조정합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DLMA는 ACLAward 및 Laboratory와 같은 벤치마크에서 자동 평가 점수에서 최첨단 성능을 보여줍니다. 이 시스템은 강력한 기준선을 크게 초월하여 우수한 연구 성과를 도출했습니다. 절제 연구(ablation studies)는 두 루프가 서로 다른 중요 역할을 수행하여 혁신성을 이끌어내고 실행의 신뢰성을 보장함을 확인했습니다.



### Verifying Memoryless Sequential Decision-making of Large Language Models (https://arxiv.org/abs/2510.06756)
- **What's New**: 이번 논문에서는 메모리가 없는 순차 의사결정 과제에서 대규모 언어 모델(LLM) 기반 정책의 엄밀한 자동 검증 도구를 소개합니다. 이 도구는 주어진 마르코프 결정 과정(MDP)과 LLM 정책, 그리고 안전 요구 사항을 PCTL 수식으로 표현하여 접근 가능한 MDP의 일부만을 점진적으로 구축합니다. 각 상태는 자연어 프롬프트로 인코딩되고, LLM의 응답은 액션으로 해석되며, 정책에 의해 도달 가능한 후속 상태들이 확장됩니다.

- **Technical Details**: 제안된 접근법은 MDP의 환경 역학을 정의하고, 특정 상태에 대해 액션을 제안하는 LLM 기반 정책, 원하는 안전 속성을 정의하는 PCTL 수식, 상태를 LLM 입력 프롬프트로 매핑하는 입력 인코더 함수, LLM 출력을 액션으로 파싱하는 액션 파서 함수를 포함하여 총 다섯 가지 입력을 수용합니다. 우리는 LLM 충실도를 유지하기 위해 동일한 입력에 대해 무작위 시드를 제어하며, 목표하는 안전 속성을 만족하는지 Storm 모델 검증기를 통해 검증합니다.

- **Performance Highlights**: 실험 결과 우리는 Ollama를 통하여 접근할 수 있는 최첨단 오픈소스 LLM이 반복적으로 검증될 수 있음을 확인하였으나, 현재로서는 심층 강화 학습(Deep Reinforcement Learning) 기법에 비해 성능이 저조함을 보였습니다. 본 도구는 사용자 정의할 수 있는 과제를 PRISM 언어로 지정할 수 있도록 지원하여, 메모리가 없는 순차 의사결정에서 더 능력 있는 LLM을 검증하는 기초를 마련합니다. 전반적으로, 본 연구는 LLM 기반 정책의 안전성을 보다 체계적으로 검증할 수 있는 새로운 길을 제시합니다.



### MultiCNKG: Integrating Cognitive Neuroscience, Gene, and Disease Knowledge Graphs Using Large Language Models (https://arxiv.org/abs/2510.06742)
- **What's New**: 대규모 언어 모델(LLMs)의 등장으로 생물 의학 및 인지 과학 분야에서 지식 그래프(KGs)의 통합이 혁신적으로 이루어졌습니다. 이 논문에서는 MultiCNKG라는 혁신적인 프레임워크를 제안하며, 이는 다양한 지식 소스를 결합하여 복잡한 유전자, 질병 및 인지 과정 간의 의미적 연결성을 포착합니다.

- **Technical Details**: MultiCNKG는 인지 신경 과학 지식 그래프(CNKG), 유전자 온톨로지(GO), 질병 온톨로지(DO)를 포함한 세 가지 주요 지식 소스를 통합하여 만듭니다. 이 프레임워크는 LLMs(예: GPT-4)를 활용하여 엔티티 정렬(entity alignment), 의미적 유사성 계산(semantic similarity computation), 그래프 증강(graph augmentation)을 수행합니다.

- **Performance Highlights**: MultiCNKG는 6.9K 개의 노드와 11.3K 개의 엣지를 포함하고 있으며, 정밀도(precision) 85.20%, 재현율(recall) 87.30%, 커버리지(coverage) 92.18% 등 다양한 메트릭에서 우수한 성능을 보여줍니다. 논문의 평가 결과는 개인 맞춤형 의학, 인지 장애 진단 및 인지 신경 과학에서의 가설 수립 등 다양한 응용 분야에서의 가능성을 제시합니다.



### Inefficiencies of Meta Agents for Agent Design (https://arxiv.org/abs/2510.06711)
- **What's New**: 최근 연구들은 메타 에이전트를 활용해 에이전트 시스템의 설계를 자동화하는 방향으로 나아가고 있습니다. 이 논문에서는 메타 에이전트의 세 가지 주요 과제를 분석하며, 기존 연구에서 언급된 방법들이 실제로는 이전 설계를 보다 잘 활용하지 못함을 보여줍니다. 단순히 이전 에이전트를 포함하는 것이 덜 효과적임을 입증하며, 진화적 접근법이 더 나은 성능을 발휘함을 강조합니다.

- **Technical Details**: 메타 에이전트는 반복적인 샘플링(sample), 평가(evaluate), 반복(iterate) 패턴을 따르며, 이를 통해 새로운 에이전트를 설계합니다. 저자들은 메타 에이전트의 성능을 향상시키기 위해 세 가지 다른 컨텍스트 큐레이션(context curation) 전략을 사용하여 실험하였습니다. 특히, 진화적 맥락 큐레이션이 성능을 개선하며, 특히 MMLU와 DROP 데이터셋에서 15,000개의 예제에서 경제적 타당성을 분석합니다.

- **Performance Highlights**: 실험 결과, 누적 컨텍스트 큐레이션이 평행 컨텍스트 큐레이션보다 더 낮은 성능을 보였으며, 이는 기존 설계를 거의 활용하지 못하는 것을 나타냅니다. 진화적 맥락 큐레이션은 MGSM에서 +10%의 성능 향상을 가져와, 높은 품질의 이전 설계를 사용하는 것이 메타 학습을 더 효과적으로 만들어줌을 시사합니다. 하지만 설계된 에이전트 간의 행동 다양성이 결여되어 있어, 모든 쿼리에 대해 최적의 에이전트를 동적으로 선택하기 어려운 상황임을 알 수 있습니다.



### Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Suppor (https://arxiv.org/abs/2510.06674)
Comments:
          EMNLP 2025 Industry Track submission (Paper #305). Preprint. Main text within the 7-page industry limit (references/appendices excluded). Contains multiple figures and tables

- **What's New**: 본 논문에서는 LLM 기반 고객 지원 시스템을 점진적으로 개선하기 위한 연속 데이터 플라이휠(continuous data flywheel) 프레임워크인 AITL(Agent-in-the-Loop)을 소개합니다. AITL은 4가지 주요 주석(annotations) 유형을 실시간 고객 운영에 직접 통합하여, 배치 주석(batch annotations)에 의존하는 기존 접근방법과 차별화됩니다. 이러한 접근법은 모델의 업데이트 주기를 몇 달에서 몇 주로 단축시키며, 고객 지원 에이전트를 대상으로 한 파일럿이 실질적인 성과를 보였습니다.

- **Technical Details**: 고객 지원 시나리오에서 데이터 플라이휠은 진화하는 제품 기능, 변화하는 사용자 선호도, 그리고 지속적으로 업데이트되는 정책 및 절차에 적응하는 데 중요한 역할을 합니다. 이 연구는 고정된 데이터셋으로 훈련된 기존 LLM들이 실시간 상호작용에 적응하기 어려운 문제를 해결하고자, 고객 입력, LLM 기반 상호작용 시스템, 에이전트 주석 및 연속 학습 파이프라인을 통해 주석 데이터를 통합하는 방법론을 제안합니다. 또한, 다양한 메타데이터를 포함한 통합 지식 기반을 통해 실시간으로 에이전트의 주석 및 검색을 지원합니다.

- **Performance Highlights**: AITL 시스템을 통해 미국의 고객 지원에서 5,000건 이상의 사례를 수집하고 40명의 에이전트가 참여하였습니다. 실험 결과, 검색의 정확도는 +11.7%의 recall@75 및 +14.8%의 precision@8로 개선되었고, 생성 품질 또한 +8.4%의 도움fulness 향상이 있었습니다. 이처럼 AITL 프레임워크는 주석 품질, 모델 개발 주기 효율성 및 LLM 기반 고객 지원 시스템의 전체 성과를 크게 향상시키는 데 기여하였습니다.



### Fine-Grained Emotion Recognition via In-Context Learning (https://arxiv.org/abs/2510.06600)
Comments:
          9 pages, 10 figures, 4 tables

- **What's New**: 이 논문은 세분화된 감정 인식을 위한 결정 메커니즘을 탐구하고, 기존 In-Context Learning (ICL) 연구의 한계를 보완하기 위해 Emotion In-Context Learning (EICL) 기법을 제안합니다. EICL은 정서적으로 유사한 예제를 사용하고 동적 소프트 라벨 전략을 도입하여 감정 추론을 향상시키는 동시에 결정 과정의 최적화를 포함합니다.

- **Technical Details**: 감정 인식 과정에서는 쿼리 표현(query representation)과 감정 프로토타입(emotional prototype) 간의 유사성 매칭(similarity matching)이 중요합니다. ICL 기법은 세맨틱(Semantic) 로직을 기반으로 하지만, 이 과정에서 감정적으로 정확한 표현을 형성하는데 한계가 있음을 지적합니다. EICL은 이러한 결점을 해결하기 위해 감정적으로 유사한 예제를 도입합니다.

- **Performance Highlights**: 다양한 실험을 통해 EICL은 4개의 세분화된 감정 데이터 세트에서 ICL보다 유의미하게 향상된 결과를 보였습니다. 이러한 성과는 EICL이 감정 추론과 결정 과정을 통합하여 세분화된 감정 인식을 효과적으로 개선함을 나타냅니다.



### WebDART: Dynamic Decomposition and Re-planning for Complex Web Tasks (https://arxiv.org/abs/2510.06587)
- **What's New**: 최근 발표된 WebDART 프레임워크는 복잡한 웹 작업을 효율적으로 처리할 수 있도록 설계된 새로운 접근 방식입니다. 기존의 LLM(대형 언어 모델) 에이전트가 단순 작업에서는 효과를 보였지만, 다단계 탐색이 필요한 복잡한 작업에서는 성능이 저하되는 문제를 해결하고자 합니다. WebDART는 작업을 동적으로 세 가지 하위 작업으로 분해하여 각 하위 작업에 집중합니다.

- **Technical Details**: WebDART는 웹 탐색, 정보 추출 및 실행의 세 가지 하위 작업으로 복잡한 목표를 분해합니다. 이 프레임워크는 탐색 과정에서 새롭게 발견된 웹 요소를 활용하여 계획을 재조정할 수 있는 역동적 재계획 메커니즘을 통합하고 있습니다. 이를 통해 LLM이 더 낮은 인지 부담으로 높은 정확도를 달성할 수 있도록 합니다.

- **Performance Highlights**: WebDART는 WebChoreArena와 WebArena에서 광범위한 평가를 수행하여 기존의 최첨단(SOTA) 에이전트에 비해 최대 13.7% 향상된 성공률을 달성했습니다. 또한, WebArena에서는 비슷한 성능을 유지하면서도 복잡한 작업 성능이 크게 개선되었으며, 동적인 재계획 모듈을 통해 평균 14.7단계의 탐색을 줄이면서 정확도를 추가로 7.7% 향상시켰습니다.



### Auto-Prompt Ensemble for LLM Judg (https://arxiv.org/abs/2510.06538)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 평가자의 신뢰성을 개선하는 새로운 프레임워크인 Auto-Prompt Ensemble (APE)을 제안합니다. 기존의 LLM 평가자는 인간의 평가 기준을 파악하지 못해 중요한 평가 차원을 놓치기도 했습니다. APE는 이러한 문제를 해결하기 위해 자동으로 평가 차원을 학습하고, 신뢰도를 기반으로 한 앙상블 메커니즘을 통합했습니다.

- **Technical Details**: APE의 핵심은 실패 사례를 분석하여 새로운 평가 차원을 자동 생성하고, 이를 바탕으로 LLM의 판단을 보완하는 것입니다. 초기에 LLM은 실패 사례를 식별하고, 이후에는 LLM 자체가 새로운 평가 차원과 해당 점수 기준을 생성합니다. 추가된 차원들은 Collective Confidence 메트릭을 통해 평가되며, 이는 다양한 평가 차원들의 집합 신뢰성을 정량화하여 최종 판단의 정확성을 향상시킵니다.

- **Performance Highlights**: APE는 여러 LLM 평가 기준에서 상당한 성과를 보였습니다. 예를 들어, Skywork Preference 데이터세트에서 APE를 적용함으로써 GPT-4o의 인간 기호와의 일치율이 83.6%에서 86.2%로 증가했으며, Reward Bench에서는 87.2%에서 90.5%로 향상되었습니다. 이러한 결과는 APE가 모델 평가 기준과 인간 평가 기준 사이의 격차를 줄이는데 효과적임을 보여줍니다.



### Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them (https://arxiv.org/abs/2510.06534)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 이용한 Agentic search의 새로운 접근 방식을 제안합니다. 이는 복잡한 사용자 정보 요구를 해석하고 다단계 계획, 검색, 정보 종합 과정으로 답변을 제공하는 과정을 포함합니다. 또한, 성공적인 Agentic search 경로를 분석하고 네 가지 유익한 추론 행동 특성을 식별합니다.

- **Technical Details**: 우리는 네 가지의 유익한 추론 행동: 정보 검증(Information Verification), 권위 평가(Authority Evaluation), 적응 검색(Adaptive Search), 오류 복구(Error Recovery)를 도출합니다. 이를 통해 Behavior Priming이라는 기법을 제안하며, 이는 감독 학습(Supervised Fine-Tuning, SFT) 후 강화 학습(Reinforcement Learning, RL)을 사용하여 Agentic search 모델을 훈련하는 데 사용됩니다.

- **Performance Highlights**: 세 가지 벤치마크(GAIA, WebWalker, HLE)에서의 실험 결과, 행동 프라이밍은 Llama3.2-3B와 Qwen3-1.7B에서 RL로 직접 훈련한 Agentic search 모델에 비해 35% 이상의 성능 향상을 보여줍니다. 특히, SFT 데이터 내에서의 원하는 추론 행동이 최종 성능에 중요한 요소임을 입증하며, 올바른 답변 대신 바람직한 추론 행동을 포함한 경로에서 미세 조정하는 것이 더 나은 성능으로 이어진다고 설명합니다.



### PuzzlePlex: Benchmarking Foundation Models on Reasoning and Planning with Puzzles (https://arxiv.org/abs/2510.06475)
- **What's New**: 이번 연구는 foundation 모델의 추론과 계획 능력을 평가하기 위한 새로운 벤치마크인 PuzzlePlex를 소개합니다. PuzzlePlex는 각기 다른 난이도의 15종류의 퍼즐로 구성되어 있으며, 단일 플레이어 및 2인 플레이어 환경을 포함합니다. 이는 복잡하고 동적인 환경에서 모델이 문제 해결 능력을 얼마나 끌어올릴 수 있는지를 탐구하기 위한 목적을 가지고 있습니다.

- **Technical Details**: PuzzlePlex는 텍스트 및 이미지 형식의 퍼즐을 지원하며, 결정론적(deterministic) 및 확률적(stochastic) 환경을 포함하여 다양한 유형의 퍼즐을 제공합니다. 각 퍼즐은 여러 난이도 수준을 지원하며, 모델의 발전에 따라 평가를 조정할 수 있는 확장성을 제공합니다. 연구는 instruction-based와 code-based 두 가지 범주로 나누어 모델을 비교합니다.

- **Performance Highlights**: 실험 결과, 모델들이 instruction-based 환경에서 더 우수한 성능을 보였으며, 이는 테스트 시간의 스케일링(test-time scaling)과 연장된 심사를 활용한 결과입니다. 반면, code-based 평가에서는 프로그램 생성(program synthesis)에서의 어려움으로 인해 성능이 저하되었으며, 샘플링 기반 방법이 성능 향상에 기여했습니다. PuzzlePlex의 도입은 future improvements에 대한 방향성을 제시하며, 추론, 계획, 일반화의 영역에서 테스트를 가능하게 합니다.



### Flavonoid Fusion: Creating a Knowledge Graph to Unveil the Interplay Between Food and Health (https://arxiv.org/abs/2510.06433)
- **What's New**: 이 논문에서는 '약으로서의 음식'이라는 개념이 건강 분야에서 주목받고 있으며, 음식과 건강 간의 관계를 표준화되고 기계가 읽을 수 있는 형태로 표현하는 연구의 필요성을 강조합니다. 기존의 연구들이 이 분야에 적은 기여를 하고 있는 반면, 연구자들은 USDA 데이터베이스에 있는 플라보노이드(flavoids) 함량과 관련된 정보를 가지고 새로운 지식 그래프(knowledge graph)를 구축하고자 합니다.

- **Technical Details**: 연구는 KNARM 방법론을 사용하여 음식과 건강 간의 관계를 면밀히 분석하였고, 이를 기계가 조작할 수 있는 형식으로 표현했습니다. 제안된 지식 그래프는 다양한 플랫폼에서 수집한 정보를 통합하여 음식의 플라보노이드(content)와 문헌에 나타난 암(cancer) 연결을 연결하는 역할을 합니다.

- **Performance Highlights**: 제안된 지식 그래프는 연구자들이 식단 선택과 질병 관리 간의 복잡한 상호작용을 탐색할 수 있는 예제를 제공합니다. 앞으로의 연구는 지식 그래프의 범위를 확장하고, 더 많은 관련 데이터를 추가하여 숨겨진 관계를 밝혀내기 위한 추론(inferences)을 수행하는 데 중점을 둘 예정입니다.



### Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory? (https://arxiv.org/abs/2510.06410)
- **What's New**: 본 연구는 Reasoning LLMs가 복잡한 작업에서의 성능 향상을 위해 서로의 추론 과정을 공유하고 협력하는 새로운 접근법을 탐구합니다. 특히, "off-trajectory reasoning"이라 불리는 개념을 도입하여 다른 모델의 부분적인 사고를 평가하고 활용할 수 있는 가능성을 조사합니다. 연구는 15개의 open-weight LLM들을 사용하여 이들이 상호 협력을 통해서 어떤 제한점을 지니고 있는지를 밝히고 있습니다.

- **Technical Details**: 연구에서 제안하는 두 가지 테스트, 즉 Recoverability와 Guidability는 LLM들이 다른 모델의 잘못된 또는 올바른 추론을 기반으로 원래의 목표로 돌아갈 수 있는 능력을 평가합니다. Recoverability는 협력자의 잘못된 추론으로부터 회귀할 수 있는 능력을 측정하고, Guidability는 다른 모델의 정확한 추론을 바탕으로 문제 해결을 시도하는 능력을 평가합니다. 이러한 테스트는 LLM들이 자체적인 경로 이탈 상황에서도 효과적으로 협력할 수 있는지를 이해하는 중요한 방법이 됩니다.

- **Performance Highlights**: 연구 결과, 여러 benchmark에서 '더 강력한' LLM들이 오히려 주의 흩어짐에 더 취약하다는 반전의 결과가 나타났습니다. 이들 LLM들은 협력자의 유용한 힌트를 활용하지 못하며, 각 모델이 문제 해결에 실패할 확률은 9.2%를 넘지 않았습니다. 이러한 발견은 현재의 교육 방식이 LLM의 더 넓은 추론 능력, 특히 off-trajectory reasoning을 평가하는 데 한계를 지니고 있음을 시사합니다.



### Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks (https://arxiv.org/abs/2510.06307)
Comments:
          This paper has been accepted by NeurIPS 2025

- **What's New**: 이 논문에서는 다중 에이전트 시스템(MAS)에서 신뢰성 있는 합의를 형성하기 위한 새로운 이론적 프레임워크인 Belief-Calibrated Consensus Seeking (BCCS)를 제안합니다. 기존의 합의 접근 방식은 에이전트 간 의견 차이를 무시하고, 모든 에이전트와의 협력을 통해 결과를 업데이트하는 데 한계가 있었습니다. BCCS는 최고의 협력자를 선택하고, 각 에이전트의 신념 수준에 따라 합의 판단을 조정하여 안정적인 합의를 촉진합니다.

- **Technical Details**: BCCS 프레임워크는 합의 결정 모듈을 포함하여, 각 에이전트의 출력과 신념 수준을 기준으로 합의 상태를 평가합니다. 에이전트들은 지지하는 에이전트와 반대 에이전트와의 협력을 통해 최적의 협력자를 자동으로 할당받으며, 이는 합의 형성과 최적화된 결과를 유도합니다. 안정적인 합의를 달성하기 위한 수학적 이론적 보장이 제시되며, 이는 공동 작업 중 서로 상반되는 의견도 포함됨을 전제로 합니다.

- **Performance Highlights**: 실험 결과, BCCS는 MATH 및 MMLU 벤치마크 데이터셋에서 각각 2.23% 및 3.95%의 정확도 향상을 보여주었습니다. 또한, 연구에서는 각 핵심 구성 요소의 영향을 정량화하기 위한 절단 연구(ablation study)를 통해 BCCS의 효과를 검증했습니다. 전체적으로 BCCS는 다중 에이전트 시스템에서 합의 과정의 정확성을 크게 향상시키는데 성공하였습니다.



### Requirements for Game-Based Learning Design Framework for Information System Integration in the Context of Post-Merger Integration (https://arxiv.org/abs/2510.06302)
- **What's New**: 이 논문은 인수합병(merger) 후 정보 시스템 통합(integration)에서의 학습 한계를 해결하기 위해 게임 기반 학습 디자인(game-based learning design)을 도입하는 방안을 탐구합니다. 기존 AMILI와 AMILP 방법론은 이론과 실습은 존재하나, 실제 적용 시 학습 곡선이 높고 학습자의 동기 부여가 부족하다는 문제가 있었습니다. 이 연구는 이러한 문제를 해결하기 위한 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 논문은 기본 학습 이론, 인지 부하(cognitive load) 및 동기 모델, 그리고 진지한 게임 디자인 프레임워크(serious game design frameworks)를 분석하여 게임 기반 학습 디자인 프레임워크의 필수 요구사항을 식별합니다. 요구사항은 두 가지 구성 요소로 구조화되며, 변환 프로세스(transformation process)와 결과적인 학습 경험(learning experience)을 포함합니다. 이는 정보 시스템 통합을 위한 맞춤형 학습은 물론, 효과적인 실습 교육을 지원합니다.

- **Performance Highlights**: 논문은 제안된 프레임워크를 반복 디자인(iterative design)과 실제 검증(real-world validation)을 통해 개발하고 평가할 계획을 마무리짓습니다. 이 프레임워크는 학습자의 동기 부여를 증진하고, 더 나아가 효과적인 정보 시스템 통합을 도울 수 있을 것으로 예상됩니다. 결론적으로, 게임 기반 학습 디자인을 통해 인수합병 후 통합에서의 정보 시스템 교육의 새로운 패러다임을 제시하고 있습니다.



### BuilderBench -- A benchmark for generalist agents (https://arxiv.org/abs/2510.06288)
Comments:
          Project page: this https URL and Code: this https URL

- **What's New**: 이 논문은 BuilderBench라는 새로운 벤치마크를 소개하여 에이전트의 탐색 및 학습 능력을 키우는 연구를 가속화합니다. BuilderBench는 블록을 사용하여 다양한 구조를 만들도록 요구하며, 에이전트가 환경에 대한 일반적인 원칙을 스스로 발견하도록 학습합니다. 이는 기존 데이터의 한계를 넘어서는 문제를 해결할 수 있는 능력을 갖춘 에이전트를 개발하는 데 기여할 것입니다.

- **Technical Details**: BuilderBench는 로봇 손이 블록과 상호작용하는 빠른 시뮬레이터를 통해 구현됩니다. 이 환경에서는 motor skills (운동 능력), logical reasoning (논리적 추론), geometric reasoning (기하학적 추론) 및 intuitive physics (직관적 물리학) 등 다양한 기술이 요구됩니다. 에이전트는 개별 행동을 암기하는 것이 아니라, 건축의 일반적인 패턴을 배우고 이를 바탕으로 훈련 및 평가를 진행합니다.

- **Performance Highlights**: 이 논문에서는 BuilderBench의 오픈 소스 코드와 40개 이상의 다양한 과업을 제공하여 에이전트의 성능을 평가할 수 있도록 합니다. 이 시뮬레이터는 MuJoCo와 JAX를 기반으로 하여 CPU 기반 벤치마크보다 10배에서 100배 더 빠른 훈련 속도를 자랑합니다. 또한, 강화 학습( Reinforcement Learning, RL) 알고리즘을 쉽게 적용할 수 있는 단일 파일 구현도 제공하여 연구자들이 효율적으로 사용할 수 있도록 돕습니다.



### Bridging Reasoning to Learning: Unmasking Illusions using Complexity Out of Distribution Generalization (https://arxiv.org/abs/2510.06274)
- **What's New**: 이번 연구에서는 복잡성 아웃 오브 배급(Complexity Out-of-Distribution, Complexity OoD) 일반화 개념을 도입하여 AI 모델의 추론 능력을 정의하고 측정하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 모델이 훈련 데이터보다 더 복잡한 테스트 샘플에서 성능을 유지할 수 있는지를 평가합니다. 기존의 일반화 지표가 System-1(신속하고 직관적인) 처리에 초점을 맞춘 것과 달리, Complexity OoD는 System-2(느리고 심사숙고하는) 추론 능력을 평가하는데 있어 새로운 기준을 제공합니다.

- **Technical Details**: Complexity OoD의 핵심은 특정 문제 인스턴스에 대해 요구되는 표현 능력이나 필수 해결 단계 수와 같은 '복잡성'을 정의하는 것입니다. 이를 위해 Kolmogorov 복잡성(kolmogorov complexity)이라는 개념을 활용하여 복잡성을 형식화합니다. 이러한 접근 방식은 기존의 길이 및 구성적인 OoD와의 차이를 명확히 하여, 학습과 추론 간의 관계를 다시 생각하게 만듭니다.

- **Performance Highlights**: Complexity OoD를 통한 평가 접근법은 기존 데이터 오염에 대해 더 강건한 평가를 제공하며, 모델의 기본적인 능력을 보다 정밀하고 세분화된 방식으로 측정할 수 있습니다. 이는 AI 모델의 추론 능력을 평가하는 새로운 방향을 제시하며, System-2 처리의 성공적인 습득이 System-1과 같은 요소의 기저 학습에 의존함을 보여줍니다. 이를 통해 학습과 추론 간의 장기적으로 존재하는 개념적 간극을 메우는 데 기여할 수 있습니다.



### AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning (https://arxiv.org/abs/2510.06261)
Comments:
          Ongoing project

- **What's New**: AlphaApollo는 자체 진화하는 에이전틱 추론 시스템으로, 기존의 재단 모델(FM) 추론에서 발생하는 두 가지 주요 제약인 모델 고유의 능력과 신뢰할 수 없는 테스트 시간 반복 문제를 해결하고자 합니다. 이 시스템은 계산 도구(Python)와 검색 도구(작업 관련 외부 정보)를 결합하여 정확한 계산을 수행하고 결정을 기반으로 합니다. 이러한 통합으로 AlphaApollo는 복잡한 문제를 해결하는 것뿐만 아니라, 다양한 모델과 도구를 협동으로 조율합니다.

- **Technical Details**: AlphaApollo의 설계 원리는 다양한 모델과 전문 도구를 조화롭게 결합하여 자가 진화형 시스템을 구현하는 것입니다. 이를 통해 직관적이고 정의된 추론을 가능하게 하며, 수학적 문제 해결 시 Python 코드의 실행과 검증에 기반한 피드백을 제공합니다. 이 시스템은 모델 간의 상호작용과도 결합하여 도구-확장된 추론을 진행하며, 이를 통해 근본적인 한계를 넘어서는 데 기여합니다.

- **Performance Highlights**: AlphaApollo는 다양한 모델에서 일정한 성능 개선을 보여줬으며, Qwen2.5 모델에서는 평균 5.15% 증가, Llama-3.3-70B-Instruct 모델에서는 8.91%의 평균 성능 향상이 있었습니다. 또한, 도구 사용 분석 결과 80% 이상의 도구 호출이 정확하게 수행되어 비도구 기반 응답을 일관되게 초과하는 성과를 보였습니다. 현재 AlphaApollo는 지속적인 개발 중이며, 향후 추가 기능 및 실험 결과가 오픈 소스로 공개될 예정입니다.



### Artificial Hippocampus Networks for Efficient Long-Context Modeling (https://arxiv.org/abs/2510.07318)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 인공 신경망의 메모리 프레임워크를 제안하고, Multi-Store Model에서 영감을 받아 Artificial Hippocampus Network (AHN)를 도입하였다. AHN은 Transformer의 KV 캐시를 손실 없는 단기 메모리로 유지하며, 슬라이딩 윈도우 밖의 정보를 압축하여 고정 크기의 장기 메모리로 변환한다. 이 방법은 최신 RNN 유사 구조를 이용하여 AHNs를 구현하고, 오랜 컨텍스트 벤치마크에서 개선된 성과를 보여주는 실험 결과를 제시한다.

- **Technical Details**: AHN의 구조는 Mamba2, DeltaNet, Gated DeltaNet와 같은 RNN 유사 아키텍처로 인스턴스화되며, 이들 모델은 손실 없는 단기 메모리를 슬라이딩 윈도우로 유지한다. 정보가 윈도우를 넘어갈 경우, AHN 모듈이 이를 고정 크기로 압축하는 방식으로 작동한다. 이로 인해 AHN을 적용한 모델이 슬라이딩 윈도우 및 전체 주의(attention) 모델들을 능가하고, 메모리 및 계산 비용을 현저히 줄인다.

- **Performance Highlights**: 실험에서는 AHN을 적용한 Qwen2.5-3B-Instruct 모델이 40.5%의 플롭(FLOPs) 감소와 74.0%의 메모리 캐시 감소를 달성했으며, 평균 점수가 4.41에서 5.88로 향상되었다. 이러한 결과는 AHN을 통한 메모리 효율성을 극대화하고, 긴 시퀀스 처리에서 경쟁력 있는 성능을 발휘함을 보여준다. 논문은 AHN의 변형 모델을 개발하기 위한 코드와 모델을 배포할 예정이다.



### Vibe Checker: Aligning Code Evaluation with Human Preferenc (https://arxiv.org/abs/2510.07315)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)을 활용하여 코드 생성을 위해 자연어 상호작용을 사용하는 'vibe coding'을 소개합니다. Vibe check은 코드의 기능성뿐만 아니라, 솔루션이 어떤 느낌을 주어야 하고, 읽기 쉽고, 사용자의 의도를 유지해야 함을 강조합니다. 이 연구는 기능적 정확성을 넘어서는 지침 준수(instruction following)가 vibe check의 핵심 요소임을 가정합니다.

- **Technical Details**: 우리는 VeriCode라는 30개의 검증 가능한 코드 지침의 분류법을 제시하며, 각 지침에 해당하는 결정론적 검증기를 제공합니다. 이 분류법은 기존의 평가 도구를 보완하여, 코드의 지침 준수와 기능적 정확성을 동시에 평가하는 Vibe Checker라는 테스트베드를 구성합니다. 이를 통해 31개의 주요 LLM을 평가하며, 강력한 모델조차도 여러 지침 준수에서 어려움을 겪는다는 것을 확인했습니다.

- **Performance Highlights**: 기능적 정확성과 지침 준수의 복합 점수가 인간의 선호와 가장 잘 연관됨을 보여주며, 실제 프로그래밍 작업에서 지침 준수가 주요 구분 요소로 부각됩니다. 이 연구는 코드 작업에서 사용자의 선호도에 더 잘 맞춰진 모델을 개발하기 위한 기준을 제시합니다.



### GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations (https://arxiv.org/abs/2510.07314)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이번 논문에서는 플라스마 난류를 정확하게 모델링하기 위한 새로운 신경망 대리 모델인 GyroSwin을 소개합니다. GyroSwin은 5D 비선형 자이로키네틱 시뮬레이션을 수행할 수 있는 최초의 확장 가능한 모델로, 축소된 모델들이 간과한 물리적 현상들을 포착합니다. 이 모델은 기존의 접근 방식보다 계산 비용을 세 배 줄여 주며, 플라스마 난류의 열전달 예측에서 뛰어난 성능을 보입니다.

- **Technical Details**: GyroSwin은 세 가지 주요 요소로 구성되어 있습니다. 첫째, Swin Transformer를 5D 데이터에 확장하여 비선형 물리를 포착합니다. 둘째, 5D 및 3D 필드 간의 상호작용을 위한 교차 주의(cross-attention) 모듈과 통합 모듈을 도입합니다. 셋째, 비선형 물리학에서 영감을 받은 채널별 모드 분리를 수행하여 정확한 예측을 가능하게 합니다.

- **Performance Highlights**: GyroSwin의 성능은 기존의 상태-of-the-art 모델들보다 우수하며, 열 전송 예측에 있어 기존의 축소된 수치 모델들에 비해 우수한 결과를 도출합니다. 이 모델은 10억 개의 매개변수로 확장 가능한 성능을 보여주며, 비선형 자이로키네틱 시뮬레이션의 정확성과 효율성을 크게 향상시킵니다.



### h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning (https://arxiv.org/abs/2510.07312)
Comments:
          Preprint, 31 pages, 8 figures

- **What's New**: 이 연구에서는 기존의 짧은 호리존(short-horizon) 데이터를 활용하여 장기 호리존(long-horizon) 추론 능력을 향상시키기 위한 새로운 방법을 제안합니다. 기존 접근법들이 inference-time scaffolding나 비싼 step-level supervision에 의존하는 반면, 우리의 방법은 간단한 문제들을 합성하여 복잡한 multi-step dependency chains를 구성합니다. 이러한 방식으로, 별도의 주석 없이도 장기 호리존 데이터 생성을 가능하게 하여 RL(강화학습) 훈련을 효율적으로 확장합니다.

- **Technical Details**: 제안된 방법에서는 short-horizon 문제(예: GSM8K 질문들)를 연결하여 무제한 길이와 복잡성을 갖는 종속 추론 단계 체인을 생성합니다. 이 데이터에 대해 outcome-only rewards를 사용하는 RL로 훈련하며, 훈련 과정에서 다가오는 문제의 난이도를 자동으로 증가시키는 커리큘럼을 적용합니다. 이를 통해 RL 훈련에서 성능 포화 문제를 해결하고 모델이 유용한 장기 호리존 추론 경로를 내부화하도록 돕습니다.

- **Performance Highlights**: 이 연구의 결과는 composed 6학년 수준의 수학 문제(GSM8K)에 대한 커리큘럼 훈련이 경쟁 수준의 장기 벤치마크(GSM-Symbolic, MATH-500, AIME)에서 최대 2.06배의 정확도 향상을 이끌어냈음을 보여줍니다. 또한, RL training을 통해 기존의 강력한 baseline와 비교했을 때, 모델들이 새로운 추론 경로를 학습할 수 있음을 입증했습니다. 이 연구는 장기 호리존 문제를 해결하는 RL의 효율적인 경로를 제시하며, 기존의 데이터만으로도 장기 호리존 추론을 가능하게 합니다.



### MLE-Smith: Scaling MLE Tasks with Automated Multi-Agent Pipelin (https://arxiv.org/abs/2510.07307)
- **What's New**: MLE-Smith는 기존의 수동적으로 Curate된 MLE 벤치마크의 한계를 극복하기 위해, 원시 데이터셋을 자동으로 경쟁 스타일의 MLE 작업으로 변환하는 다중 에이전트 파이프라인을 소개합니다. 이 시스템은 generate-verify-execute 패러다임을 통해 MLE 작업의 품질을 자동으로 제공하며, 실제 사용 가능성과 다양성을 보장합니다. 특히 224개의 실제 데이터셋을 활용하여 606개의 MLE 작업을 성공적으로 생성했다는 점이 주목받습니다.

- **Technical Details**: MLE-Smith는 구조적 완전성(Structural Integrity), 의미적 타당성(Semantic Soundness), 실증적 해결 가능성(Empirical Solvability)을 보장하기 위해 다중 에이전트 생성 워크플로우와 하이브리드 검증 메커니즘을 통합합니다. 이 시스템은 Brainstormer, Designer, Refactor라는 세 가지 전문 에이전트를 통해 작업 제안을 생성, 구체화 및 표준화합니다. 또한, 강력한 검증 메커니즘이 지속적으로 작업의 정확성과 일관성을 보장합니다.

- **Performance Highlights**: MLE-Smith에서 생성된 606개의 작업은 잘 설계된 인간 Curate 작업과의 상관 관계가 높은 것으로 나타났습니다. 이 결과는 MLE-Smith가 다양한 범위의 작업에서 첨단 LLM들이 성능을 평가하는 데 효과적임을 보여줍니다. MLE-Smith는 다음 세대 MLE 에이전트를 평가하고 교육하는 데 적합한 도전적이고 일반화 가능한 작업을 제공합니다.



### Cocoon: A System Architecture for Differentially Private Training with Correlated Noises (https://arxiv.org/abs/2510.07304)
- **What's New**: 이 논문에서는 기계 학습 (ML) 모델이 훈련 데이터의 메모리와 누출로 인해 심각한 개인정보 보호 문제를 야기한다는 점을 강조합니다. 최근에 제안된 차별적 프라이버시 (DP) 기반 훈련 알고리즘, 특히 DP-SGD가 그러한 문제를 해결하기 위한 접근 방식으로 주목받고 있습니다. 그러나 DP-SGD는 각 훈련 단계에서 노이즈를 추가함에 따라 모델의 정확도 감소가 발생하는 단점이 있습니다. 이를 해결하기 위해, 논문에서는 새로운 방식인 Cocoon을 제안하고, 이 시스템이 훈련 효율성을 어떻게 개선하는지를 탐구합니다.

- **Technical Details**: Cocoon 프레임워크는 관련된 노이즈를 사용하여 효율적인 훈련을 지원합니다. 이 프레임워크는 훈련 전 모든 관련 노이즈를 미리 계산하고 저장함으로써 대규모 임베딩 테이블을 효율적으로 처리합니다. Cocoon은 또한 사용자 정의 근처 메모리 처리 (NMP) 하드웨어를 사용하여 과거의 노이즈를 보관하고 처리할 수 있도록 하여 데이터 전송을 최소화합니다. 이를 통해 대규모 신경망 훈련에서 메모리 및 계산 오버헤드를 줄이는 방향으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, Cocoon-Emb는 기존 방법에 비해 2.33배에서 10.82배의 성능 향상을 보였고, Cocoon-NMP는 1.55배에서 3.06배의 개선을 달성했습니다. 이러한 성과는 큰 임베딩 테이블을 활용하는 모델 및 파라미터가 많은 대규모 모델에서 더욱 두드러집니다. 이 연구는 차별적 프라이버시를 위한 훈련 방법에 대한 시스템 차별화를 제공하며, Pytorch 기반의 높은 최적화된 라이브러리를 통해 구현되었습니다. Cocoon 프레임워크는 실제 시스템에서 높은 성능을 보여주어, 학계 및 산업계에서도 큰 관심을 받고 있습니다.



### AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs (https://arxiv.org/abs/2510.07293)
Comments:
          26 pages, 23 figures, the code is available at \url{this https URL}

- **What's New**: 오디오 처리의 새로운 벤치마크인 AudioMarathon이 소개되었습니다. 이 벤치마크는 대규모 오디오 언어 모델(LALM)을 평가하기 위해 긴 형식의 오디오 입력을 처리하는 능력을 중점적으로 다룹니다. AudioMarathon은 90초에서 300초까지의 오디오 입력, 다양한 도메인 커버리지 및 복잡한 추론 태스크를 포함하는 세 가지 기본 축을 바탕으로 설계되었습니다. 이 연구는 LALM의 성능 저하와 메모리 효율성을 높이기 위한 기법을 분석합니다.

- **Technical Details**: AudioMarathon은 긴 형식의 오디오를 효과적으로 이해하고 추론할 수 있는 LALM의 능력을 평가하기 위해 고안된 포괄적인 오디오 벤치마크입니다. 이 벤치마크는 90초에서 300초까지의 오디오 입력을 포함하여 연속적인 사운드 환경에서의 복잡한 데이터를 다룹니다. 또한, AudioMarathon은 Speech Context Understanding, Audio Scene Understanding, Voice Characteristic Identification 세 가지 카테고리로 태스크를 구성하고 있습니다. 복잡한 추론 문제를 해결하기 위해 멀티 홉(multi-hop) 추론을 포함한 평가 방법론을 채택했습니다.

- **Performance Highlights**: 우리는 AudioMarathon에서 최첨단 LALM의 성능을 평가한 결과, 입력 길이가 증가함에 따라 성능이 크게 저하된다는 것을 관찰했습니다. 현재 모델 간의 성능 차이는 여러 가지 단점과 과제를 강조하며, 보다 나은 시간적 추론 및 메모리 효율성을 제공하는 아키텍처 개발이 필요하다는 것을 보여줍니다. 이 연구 결과는 오디오 및 다중 모달 연구 커뮤니티가 더욱 발전된 오디오 이해 모델을 개발하도록 촉진할 것으로 기대됩니다.



### Evolutionary Profiles for Protein Fitness Prediction (https://arxiv.org/abs/2510.07286)
- **What's New**: 이번 논문에서는 EvoIF라는 경량 모델을 소개하며, 이는 단백질 진화를 내재된 보상 극대화 과정으로 해석하고, 단백질 언어 모델(pLM)을 역 강화 학습(ILR)으로 간주함으로써 기존의 단백질 기능 예측을 통합하는 새로운 방식을 제안합니다. EvoIF는 동족 단백질에서 회수한 서열 프로필과 역접기(logits)에서 추출한 구조 진화 제약을 통합하여 단백질의 기능적 비극성과 생존 기여도를 정량화할 수 있습니다. 이 접근법은 적은 양의 데이터와 모델 파라미터로도 높은 예측 정확도를 제공하는 것을 목표로 하고 있습니다.

- **Technical Details**: EvoIF는 서열-구조 정보를 경량의 서열-구조 백본을 사용하여 인코딩하며, 역접기 프로필과 구조 회수 동종성 프로필이라는 두 개의 압축된 진화적 정보를 주입함으로써 작동합니다. 이 모델은 진화 정보를 다양한 측면에서 통합하여 단백질 적합도 예측을 위한 강력한 근거를 제공합니다. 단백질 적합도의 예측은 단백질 변이체의 기능적 성능을 정량적으로 측정하는 여러 방법을 사용하여 이루어집니다.

- **Performance Highlights**: EvoIF는 ProteinGym에서 217개의 변이 실험을 기반으로 250만 개 이상의 변이체를 대상으로 최첨단 성능을 달성하였으며, 이는 0.15%의 훈련 데이터와 최근의 대형 모델보다 적은 파라미터를 사용하여 이루어졌습니다. 추가적인 수정 실험(ablation study)을 통해 내족(구조적 변이) 및 교차족(구조 진화) 정보의 조합이 서로 잘 보완하고 강력하게 작용한다는 것을 확인하였습니다. 이는 EvoIF가 진화 정보를 모델링하는 데 있어 효과적이고 강력한 네트워크임을 시사합니다.



### GTCN-G: A Residual Graph-Temporal Fusion Network for Imbalanced Intrusion Detection (Preprint) (https://arxiv.org/abs/2510.07285)
Comments:
          This preprint was submitted to IEEE TrustCom 2025. The accepted version will be published under copyright 2025 IEEE

- **What's New**: 본 연구에서는 지연된 시간 의존성을 모델링하고 데이터 불균형 문제를 해결하기 위해, Gated Temporal Convolutional Network and Graph (GTCN-G)라는 새로운 딥러닝 프레임워크를 제안합니다. 이 모델은 Gated TCN과 Graph Convolutional Network을 결합하여 네트워크 흐름의 계층적 Temporal feature를 추출하고, Graph Attention Network (GAT)를 통해 잔여 학습 메커니즘을 추가하여 원래의 특성 정보를 보존합니다. 이는 희귀 악성 활동(소수 클래스)에 대한 감지 민감도를 향상시키기 위해 필수적입니다.

- **Technical Details**: GTCN-G 프레임워크는 네트워크 흐름에서 시간적 의존성을 포착하기 위한 Gated TCN과 그래프 구조로부터 학습하기 위한 Graph Convolutional Network의 조합으로 구성됩니다. 또한 GAT를 통해 원본 피처 정보의 잔여 연결을 유지하여 분류 작업의 정확도를 높이고, 소수 클래스에 대한 탐지 능력을 강화합니다. 이 구조는 G-TCN 및 GCN 모듈을 병렬로 처리하여 데이터의 표현을 통합하고 최종 분류 결과를 생성합니다.

- **Performance Highlights**: UNSW-NB15 및 ToN-IoT 데이터셋을 사용하여 진행된 실험에서 GTCN-G 모델은 기존의 기준 모델보다 월등히 우수한 성능을 보였습니다. 이 모델은 이진 및 다중 클래스 분류 작업 모두에서 최첨단의 성능을 달성하여, 복잡한 네트워크 트래픽의 탐지 효율성을 크게 향상시켰습니다.



### Online Rubrics Elicitation from Pairwise Comparisons (https://arxiv.org/abs/2510.07284)
- **What's New**: 이번 논문에서는 LLM을 위한 평가 기준을 온라인으로 동적으로 개발하는 새로운 방법인 Online Rubrics Elicitation(OnlineRubrics)을 소개합니다. 기존의 정적인 rubrics는 훈련 과정에서 발생하는 새로운 요구 사항을 제대로 반영하지 못했으나, OnlineRubrics는 응답을 기반으로 쌍평가(pairwise comparisons)를 통해 지속적으로 기준을 개선하는 방식입니다. 이 방법은 AlpacaEval, GPQA 등 여러 벤치마크에서 최대 8%의 성능 향상을 보였습니다.

- **Technical Details**: OnlineRubrics는 현재 모델과 참조 모델의 응답을 쌍으로 비교하여 새로운 평가 기준을 생성합니다. 이를 통해 응답의 오류를 지속적으로 식별하고 개선할 수 있으며, 기존의 rubrics를 유연하게 확장합니다. 평가 프레임워크는 reinforcement learning에 대한 다양한 응답 모델링을 가능하게 하여, verifiable 및 non-verifiable 특성을 모두 포괄합니다.

- **Performance Highlights**: 이 연구는 Expert 및 Generalist 도메인에 대한 두 개의 데이터셋을 활용하여 OnlineRubrics의 성능을 평가하였습니다. 이 방식은 GPQA-Diamond, GSM8K, AlpacaEval, Arena-Hard를 포함한 여러 벤치마크에서 기반 모델 대비 최고 25%의 성능 향상을 기록하며, 기존 정적 rubrics와 비교하여 우수한 결과를 입증했습니다.



### On the false election between regulation and innovation. Ideas for regulation through the responsible use of artificial intelligence in research and education.[Spanish version] (https://arxiv.org/abs/2510.07268)
Comments:
          20 pages, in Spanish language, 1 figure, 1 table, AI Hub-CSIC / EduCaixa, Escuela de Verano, Auditorio CaixaForum, Zaragoza, Spain, 4 July 2025

- **What's New**: 이 논문은 AIHUB (CSIC) 및 EduCaixa 여름학교에서 열린 토론 세션에서의 저자 발표 내용을 재구성한 것입니다. 저자는 Albert Sabater가 제기한 세 가지 질문에 대한 답변을 통해 AI의 규제 환경이 어떻게 기본권 보호를 우선시할 수 있는지 탐구하였습니다. 또한, 혁신과 규제 간의 허위 이분법을 피하는 방법에 대해서도 논의합니다.

- **Technical Details**: 저자는 AI의 위험성(편향성, 대량 감시, 조작 등)에 대한 공공 이익을 우선시하는 책임 있는 혁신을 촉진할 수 있는 규제의 사례를 예로 들고 있습니다. 특히, 중국이나 미국과 같은 경쟁자의 압력에 굴복하지 않고도 가능한 전략을 모색합니다. 또한, 유연성을 중시하는 미국의 접근에 맞서 국제 협력이 기본권 정복이 아닌 글로벌 책임 기준을 확립하는 방식으로 이루어질 수 있는 방법에 대해 설명합니다.

- **Performance Highlights**: 이 연구는 교육 및 연구의 중요성과 관련된 여러 통찰을 제공하며, AI의 발전에 있어 규제의 필요성을 강조합니다. 저자는 공개 관심을 최우선으로 고려하여 AI 발전의 혁신을 저해하지 않도록 시스템을 구축할 수 있는 가능성에 대해서도 깊은 성찰을 합니다. 이 논문은 AI 발전에 있어 책임감 있는 접근 방식이 어떻게 실현될 수 있는지를 탐구하는 데 기여합니다.



### LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation (https://arxiv.org/abs/2510.07243)
Comments:
          Published in Natural Legal Language Processing - EMNLP Workshop 2025

- **What's New**: 이 논문은 법률 분야에서 대형 언어 모델(LLM)의 출력을 평가하는데 있어 독창적인 접근 방식을 제시합니다. 연구진은 '법률 데이터 포인트'(Legal Data Points, LDPs)라는 자가 포함된 정보 단위를 활용하여 변호사들이 법률 답변을 평가하는 방식을 반영하는 새로운 평가 방법론인 LeMAJ를 소개합니다. 이 방법은 기존 기준 데이터에 의존하지 않으면서도 법률 질문 답변의 정확성과 관련성을 평가하는 데 효과적이라는 점을 강조합니다.

- **Technical Details**: LeMAJ 평가 방법론은 법률 전문직에서 사용하는 체계적인 평가 프로세스와 최근의 자동화된 요약 평가 기법에 영감을 받았습니다. LDPs를 사용하여 LLM에서 생성된 답변을 개별 정보 단위로 분해하고, 각 단위를 정확성과 관련성에 따라 평가합니다. 이 과정은 법률 전문가들이 필요한 세부적인 피드백을 제공하며, 기존 자동화된 평가 방법의 제약을 극복할 수 있습니다.

- **Performance Highlights**: 실험 결과, LeMAJ는 자체 개발한 데이터셋과 오픈 소스 데이터셋인 LegalBench에서 다양한 기존 방법들과 비교했을 때 성능이 뛰어난 것으로 나타났습니다. 또한, 변호사 간의 일치도를 개선하고, LDP를 사용한 경우 인간 전문가 평가와의 상관 관계가 더욱 높아짐을 보여주었습니다. 이러한 결과는 법률 질문 답변 평가에 있어 LLM의 평가 프로세스를 혁신적으로 변화시킬 가능성을 시사합니다.



### Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships (https://arxiv.org/abs/2510.07231)
- **What's New**: 이 논문에서는 인과 추론(causal reasoning)이 대형 언어 모델(Large Language Models, LLMs)의 핵심이지만 기존 벤치마크가 그 기능을 충분히 평가하지 못하는 경우가 많다는 점을 지적합니다. 저자들은 경제 및 금융 저널에서 추출한 비계통적인 인과 관계를 기반으로 새로운 벤치마크를 소개하며, 이는 합리적인 방법론을 이용해 수립되었습니다. 또한, 40,379개의 평가 항목과 다양한 도메인을 포함하는 다섯 가지 작업 유형을 다룹니다.

- **Technical Details**: 제안된 벤치마크는 도구적 변수(instrumental variables), 차이의 차이(difference-in-differences), 회귀 불연속 디자인(regression discontinuity designs)과 같은 엄격한 방법론을 바탕으로 구축되었습니다. 여기에는 건강, 환경, 기술, 법률 및 문화와 같은 다양한 분야가 포함됩니다. 실험에서는 총 8개의 최신 LLMs 모델이 평가되었으며, 이 모델들이 인과 관계 파악에 있어 상당한 제한사항을 보였습니다.

- **Performance Highlights**: 실험 결과, 가장 성능이 좋은 모델조차 57.6%의 정확도에 불과해 인과 관계 인식에 많은 어려움을 겪고 있음을 보여줍니다. 모델의 규모가 항상 성능 향상으로 이어지지 않으며, 고급 추론 모델들도 기본적인 인과 관계 식별에 어려움을 겪습니다. 이러한 발견은 현재 LLM의 성능과 신뢰할 수 있는 인과 추론이 요구되는 고위험 응용 프로그램 간의 중요한 격차를 강조합니다.



### Where to Begin: Efficient Pretraining via Subnetwork Selection and Distillation (https://arxiv.org/abs/2510.07227)
- **What's New**: 이 논문에서는 소형 언어 모델(SLMs)에 대한 새로운 사전 학습 프레임워크를 제안합니다. 이 프레임워크는 구조적으로 희소한 서브 네트워크 초기화, 진화적 검색을 통한 모델 초기화, 그리고 지식 증류(knowledge distillation)의 세 가지 보완적인 개념을 결합합니다. 이 접근 방식은 SLM 사전 학습의 효율성을 크게 향상시킵니다.

- **Technical Details**: SLMs는 대형 언어 모델(LLMs)과 비교하여 적은 자원으로도 강력한 성능을 발휘하는 모델입니다. 본 연구에서는 사전 학습된 교사 모델로부터 서브 네트워크를 추출하고 이를 사용하여 SLM을 초기화하고 지식 증류를 수행하는 2단계 전략을 채택합니다. 다양한 검색 공간과 초기화 방법에 대해 체계적인 비교 분석을 제공합니다.

- **Performance Highlights**: 진화적 검색을 통해 발견된 최상 모델은 LLM 가중치로 초기화되었고, 동등한 Pythia SLM의 검증 perplexity와 일치하면서 9.2배 적은 사전 학습 토큰을 요구합니다. 실험 결과, 다양한 크기의 학생 모델에 대해 사전 학습과 지식 증류를 통해 전반적인 효율성을 높이는데 기여함을 보여줍니다. 이 연구는 경제적인 SLM 개발을 위한 실질적 가이드라인을 제공합니다.



### GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation (https://arxiv.org/abs/2510.07217)
Comments:
          30 pages, 21 figures, accepted to EMNLP 2025 findings

- **What's New**: 이번 연구에서는 텍스트-이미지 합성 분야에서의 새로운 접근법을 제안합니다. 텍스트 기반의 프롬프트 최적화 기술인 GenPilot를 소개하며, 이는 다양한 모델에 적용 가능하고, 모델 훈련 없이도 효율적으로 작동합니다. GenPilot는 오류 분석, 클러스터링 기반의 적응 탐색, 세밀한 검증 등을 통해 보다 정확한 이미지 생성을 목표로 합니다.

- **Technical Details**: GenPilot는 테스트 시간에 프롬프트 최적화를 탐색 문제로 설정하여, 데이터의 해석 가능성을 높이고 동적으로 프롬프트를 정제합니다. 시스템은 주로 두 가지 단계로 구성됩니다: 오류 분석 및 테스트 시간 프롬프트 최적화. 각 단계에서 비주얼 질문 응답(VQA)와 캡셔닝 기법이 사용되며, 메모리 모듈로 최적화 과정을 지원합니다.

- **Performance Highlights**: DPG-bench 및 Geneval에서의 실험 결과, GenPilot는 최대 16.9% 및 5.7%의 성능 향상을 보여줍니다. 이는 텍스트와 이미지 간의 일관성 및 구조적 일관성을 강화시키는 데 기여하며, T2I 작업에 대한 강력한 일반화 능력을 입증합니다. GenPilot는 전반적으로 모델에 구애받지 않고 다채로운 프롬프트를 효과적으로 처리할 수 있습니다.



### Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for Large Language Models (https://arxiv.org/abs/2510.07213)
Comments:
          Work in progress. Our code will be available at: this https URL

- **What's New**: 이번 연구는 대규모 언어 모델(LLM)이 비영어 데이터에 대한 노출이 제한적임에도 불구하고 다국어 능력이 뛰어난다는 점을 강조합니다. 연구진은 언어 전환이 중간층에서 최종층으로 갈수록 일관된 인덱스에서 발생하는 소수의 차원에 의해 조정된다고 가설을 세웠습니다. 이에 따라, 단 50개의 문장으로 이러한 차원을 식별하고 조작할 수 있는 훈련 없는 방법을 소개하였습니다.

- **Technical Details**: 연구에서는 LLM의 각 층에서의 표현 변화를 관찰하며, 고유 언어 관련 차원들을 식별하는 두 가지 시나리오를 제안합니다. 첫 번째는 단일언어(monolingual) 설정으로, 중간층과 최종층의 표현을 비교하여 언어별 차원을 확인합니다. 두 번째는 병렬(parallel) 설정으로, 영어와 목표 언어의 최종층 표현을 비교하여 역시 언어별 차원을 찾습니다.

- **Performance Highlights**: 실험 결과, 제안을 통해 식별한 언어별 차원을 조작함으로써 출력 언어를 전환하는 데 성공했습니다. 이 방법은 기존의 뉴런 기반 접근 방식보다 성능이 우수하며, 데이터 요구량이 적어 효율적입니다. 다양한 모델(Llama2, Llama3.1 등)에서 다국어 생성 제어 실험을 통해 입증된 바와 같이, 본 연구는 LLM의 다국어 처리에서 중요한 기여를 하였습니다.



### HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving (https://arxiv.org/abs/2510.07210)
- **What's New**: HyPlan은 부분 관측 가능한 교통 환경에서 자율주행 자동차의 충돌 없는 내비게이션 문제를 해결하는 새로운 하이브리드 학습 보조 계획 방법입니다. 이 방법은 여러 대의 차량 동작 예측, 프로시멀 정책 최적화(proximal policy optimization)와 함께한 심층 강화 학습(deep reinforcement learning), 그리고 휴리스틱 신뢰 기반 수직 가지치기를 활용한 근사 온라인 POMDP 계획 방법을 결합합니다. HyPlan은 안전성을 보장하면서도 실행 시간을 단축할 수 있는 가능성을 보여줍니다.

- **Technical Details**: HyPlan은 POMDP(Partially Observable Markov Decision Process) 모델을 기반으로 하여 자율주행 자동차의 제어 동작을 찾는 문제를 해결합니다. 시스템의 상태는 유일한 자차와 여러 외부 에이전트의 현재 교통 상황을 나타내며, 그에 따른 제어 동작은 스티어링 각도와 가속 선택으로 구성됩니다. HyPlan은 예측기 AutoBots에 의해 외부 에이전트의 경로 예측을 수집하고, anytime 가중치 하이브리드 A* 경로 계획기를 통해 자차의 목표 위치로의 최단 안전 경로를 생성합니다.

- **Performance Highlights**: CARLA-CTS2 벤치마크에서의 실험 결과는 HyPlan이 선택된 기존 방식보다 더 안전하게 내비게이션 할 수 있으며, 다른 온라인 POMDP 계획기보다도 훨씬 빠른 성능을 보였음을 나타냅니다. HyPlan의 다각적인 접근 방식은 특히 긴급 상황에서의 안전하고 효과적인 경로 계획을 가능하게 합니다. 하이브리드 POMDP 계획기 IS-DESPOT*은 PPO 기반 깊이 강화 학습을 통해 정책 계획을 보강하여, 실행 시간을 더욱 단축시키면서도 최적화된 제어 동작을 생성합니다.



### Resolution scaling governs DINOv3 transfer performance in chest radiograph classification (https://arxiv.org/abs/2510.07191)
- **What's New**: 이 논문은 메타의 DINOv3 모델을 소개하며, 기존의 Self-supervised learning (SSL) 모델을 Gram-anchored self-distillation을 통해 확장했습니다. 이는 흉부 방사선 이미지에서의 변별력 있는 findings(발견)에 대해 SSL의 효과를 평가한 연구로, DINOv3가 DINOv2 및 ImageNet 초기화 모델과 비교한 체계적인 성능 검정을 수행했습니다.

- **Technical Details**: 논문에서는 두 가지 대표적인 backbone(백본) 모델인 ViT-B/16 및 ConvNeXt-B를 사용하여 814,000개 이상의 샘플이 포함된 7개의 데이터셋을 benchmark(벤치마크)했습니다. 이미지 해상도는 224x224, 512x512 및 1024x1024 픽셀로 분석되었으며, 주된 성과 지표는 mean AUROC(평균 면적 아래 곡선)으로 설정되었습니다.

- **Performance Highlights**: DINOv3는 224x224 해상도에서 성인 데이터셋에서 DINOv2와 비슷한 성과를 보였지만, 512x512 해상도에서는 DINOv3가 DINOv2 및 ImageNet보다 일관된 성능 향상을 보였습니다. 이러한 결과는 흉부 방사선 진단에서 높은 입력 해상도가 현대적인 SSL 모델의 이점을 활용하는 데 중요하다는 점을 강조했습니다.



### TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics (https://arxiv.org/abs/2510.07181)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 연구에서는 TIGeR (Tool-Integrated Geometric Reasoning)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 Vision-Language Models (VLMs)를 감각적 추정기에서 기하학적 컴퓨터로 변환하여 정확한 기하학적 계산을 생성하고 실행할 수 있도록 합니다. TIGeR는 외부 도구를 활용하여 기하학적 추론 요구 사항을 인식하고, 적절한 계산 코드를 합성하며, 정확한 계산을 위해 특화된 라이브러리를 호출할 수 있게 합니다.

- **Technical Details**: TIGeR-300K라는 포괄적인 도구 호출 지향 데이터셋을 개발하였습니다. 이 데이터셋은 점 변환, 자세 추정, 궤적 생성 및 공간 호환성 검증을 포함하여 총 300,000개의 샘플로 구성되어 있습니다. TIGeR는 지도학습 세부 조정(Supervised Fine-Tuning, SFT)과 강화학습 세부 조정(Reinforcement Fine-Tuning, RFT)을 결합한 두 단계의 훈련 파이프라인을 통해 훈련됩니다.

- **Performance Highlights**: TIGeR는 기하학적 벤치마크에서 최신 성과(SOTA)를 달성했으며, 실제 로봇 조작 작업에서 센티미터 수준의 정확성을 보였습니다. 이는 기하학적 계산을 위한 프로그래밍 가능 도구 호출 방식으로 귀결되며, 로봇 조작과 모션 계획을 위한 정밀한 포즈 및 궤적 생성을 가능하게 합니다. 최종적으로 TIGeR는 정밀한 기하학적 추론을 통해 실세계 로봇 작업을 지원하는 새로운 패러다임을 제시합니다.



### ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL (https://arxiv.org/abs/2510.07151)
Comments:
          22 pages, 7 figures

- **What's New**: 이번 연구에서는 ELMUR(External Layer Memory with Update/Rewrite)라는 새로운 트랜스포머 아키텍처를 제안합니다. 이 아키텍처는 각 레이어가 구조화된 외부 메모리로 보강되며, 이를 통해 장기적인 의존성 문제를 해결합니다. ELMUR는 메모리 임베딩을 유지하고, 양방향 크로스-어텐션을 통해 상호작용하며, LRU(Least Recently Used) 메모리 모듈을 사용해 메모리를 갱신합니다.

- **Technical Details**: ELMUR는 장기 메모리를 효율적으로 저장 및 검색할 수 있는 메커니즘을 제공합니다. 레이어에 로컬 메모리 임베딩을 유지하면서 메모리에 대한 읽기/쓰기 상호작용을 양방향으로 처리합니다. 또한, LRU 업데이트 블록을 통해 메모리를 대체하거나 컨벡스 블렌딩으로 새롭게 갱신하여 안정성과 적응성을 균형있게 유지합니다.

- **Performance Highlights**: ELMUR는 T-Maze 작업에서 100%의 성공률을 거두었으며, MIKASA-Robo의 스팟 보상 조작 작업에서 성능을 거의 두 배로 향상시켰습니다. POPGym 벤치마크에서도 48개의 작업 중 24개에서 최상위 점수를 기록하는 등 부분 관찰 하에서 robust한 일반화 성능을 보여주었습니다.



### A Multi-Agent Framework for Stateful Inference-Time Search (https://arxiv.org/abs/2510.07147)
- **What's New**: 이 연구에서는 상태 기반의 다중 에이전트 진화 탐색(stateful multi-agent evolutionary search)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 지속적인 추론 상태(persistent inference-time state), 적대적 변이(adversarial mutation), 진화적 보존(evolutionary preservation)을 결합하여 이전의 비상태(stateless) 접근 방식에서 벗어났습니다. 그 결과, 복잡한 테스트 케이스 생성을 통해 더욱 견고한 유닛 테스트를 자동으로 생성할 수 있게 되었습니다.

- **Technical Details**: 제안된 시스템은 여러 LLM 호출을 통해 후보 엣지 케이스를 제안하는 액터(actor), 환경을 변이시켜 견고성 갭을 드러내는 적대자(adversary), 진화 검색에 사용되는 보상을 부여하는 비평가(critic)로 구성됩니다. 각 단계에서 상태 정보를 유지하여 이전 단계의 피드백을 활용하고, 이는 선형 탐색을 넘어선 구조적 문제 해결을 가능하게 합니다. 더불어, 액터는 대형 언어 모델(LLM)의 맥락 학습을 통해 지속적인 상태를 기반으로 후보를 생성하며, 이는 전통적인 기법보다 높은 샘플 효율성을 제공합니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 HumanEval 및 TestGenEvalMini와 같은 유닛 테스트 벤치마크에서 비상태 단일 단계 기준선에 비해 상당히 높은 커버리지(coverage)를 달성했습니다. 세 가지 다양한 LLM 모델인 Llama, Gemma, GPT를 활용하여 유연한 테스트 케이스 생성을 보여 주었으며, 향상된 커버리지와 견고성을 통해 새로운 코드베이스에 대한 적응 능력이 뛰어남을 입증했습니다. 이러한 결과는 지속적인 추론 상태와 진화적 탐색이 유닛 테스트 생성에 실질적인 개선 효과를 줄 수 있음을 시사합니다.



### Comparing human and language models sentence processing difficulties on complex structures (https://arxiv.org/abs/2510.07141)
Comments:
          Data and code will be released soon

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)과 인간의 문장 이해 능력을 체계적으로 비교하였습니다. 7가지의 도전적인 언어 구조를 활용하여 인간과 5개의 LLM 계열에서 얻은 문장 이해 데이터를 수집했습니다. 연구 결과에 따르면, LLM은 전반적으로 목표 구조에서 어려움을 겪으며, 특히 'garden path' (GP) 문장에서 그 성능이 낮았습니다.

- **Technical Details**: 분석된 7가지 구조에는 GP 문장, 이중 중심 내재 문장, 유사성 기반 간섭 문장, 깊이 충전 문장 등이 포함됩니다. 각 구조에 대해 어려운 문장(타겟 문장)과 난이도가 중화된 기준 문장을 설정하였습니다. 연구에서는 31개의 최첨단 LLM을 테스트하며, 각 LLM과 인간 참여자가 동일한 질문에 답하도록 하여 성과를 비교했습니다.

- **Performance Highlights**: 인간의 평균 정확도는 41.7%로 확인되었으며, LLM은 전반적으로 인간보다 나은 성과를 보였지만 여전히 이러한 구조에서 어려움이 있었습니다. LLM은 GP 구조에서 인간과 상대적으로 비슷한 성과를 보였고, 모델의 크기가 커짐에 따라 성과의 순위 상관관계가 증가하는 경향을 보였습니다. LLM의 성능은 구조의 난이도에 따라 달라지며, 특정 모델의 경우 문장 간 발생하는 방향성을 기반으로 한 성과 차이를 재현하지 못하는 경우도 있었습니다.



### TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking (https://arxiv.org/abs/2510.07134)
Comments:
          Project page: this https URL

- **What's New**: TrackVLA++는 Embodied Visual Tracking (EVT) 작업을 위한 새로운 Vision-Language-Action (VLA) 모델로, 개선된 공간 추론 기법과 Target Identification Memory (TIM) 모듈을 포함합니다. 이 모델은 복잡한 환경에서 비슷한 물체에 의해 방해받거나 긴 시간 동안 가려질 경우에도 목표를 효과적으로 추적할 수 있는 능력을 강화합니다. Polar-CoT라는 새로운 사고 체계를 도입하여 목표의 상대적 위치를 추론하고, 이를 바탕으로 메모리를 업데이트합니다.

- **Technical Details**: TrackVLA++는 Polar-CoT 기법을 통해 목표의 상대적 위치를 극소화된 극좌표 토큰으로 인코딩합니다. TIM은 긴 시간 동안 목표의 시각적 정체성을 유지하며, 신뢰 기반의 업데이트 메커니즘을 활용하여 메모리의 상태를 조정합니다. 이러한 방법은 다중 뷰 설정에서도 동일하게 적용되며, 향상된 추적 성능을 제공합니다.

- **Performance Highlights**: TrackVLA++는 공공 벤치마크에서 SOTA 성능을 기록하며, 특히 EVT-Bench DT 분할에서 이전의 선도적 방법보다 각각 5.1% 및 12% 더 높은 성공률을 달성했습니다. 실제 환경에서도 뛰어난 제로샷 일반화를 보여주어 동적이고 가려진 상황에서도 강력한 추적 성능을 발휘합니다.



### A Digital Twin Framework for Metamorphic Testing of Autonomous Driving Systems Using Generative Mod (https://arxiv.org/abs/2510.07133)
- **What's New**: 본 논문에서는 자율주행차의 안전성을 보장하기 위한 새로운 접근 방식으로, 디지털 트윈(digital twin) 기반의 변형 테스트(metamorphic testing) 프레임워크를 제안합니다. 이 프레임워크는 자율주행 시스템과 그 운영 환경의 가상 복제본을 생성하여 복잡한 현실의 주행 시나리오를 현실감 있게 시뮬레이션합니다. 특히, Stable Diffusion과 같은 AI 기반 이미지 생성 모델을 활용하여 날씨나 도로 환경의 다양한 변형을 생성함으로써 테스트의 효과성을 높이고 있습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 핵심 구성 요소로 구성됩니다: 디지털 트윈 시나리오 생성, 변형 검증, 그리고 시간적 분석입니다. 디지털 트윈은 유지보수가 용이하고 일관된 테스트를 통해 환경 복잡성을 관리하고, 변형 검증은 자율주행 시스템의 행동 일관성을 평가합니다. 또한, 환경적 조건, 기하학적 변형, 그리고 의미론적 수정(semantic modifications)을 통해 ODD(Operational Design Domain)에 부합하는 변형된 이미지를 생성합니다.

- **Performance Highlights**: 제안된 방법은 Udacity 자율주행 시뮬레이터에서 검증되었으며, 기존 방법들에 비해 테스트 범위와 효과성을 현저히 향상시켰습니다. 특히, True Positive Rate(0.719), F1 Score(0.689), Precision(0.662) 등 주요 성능 지표에서 기존 기준 방법들과 비교하여 월등한 성능을 보였습니다. 이는 디지털 트윈과 AI 기반 시나리오 생성의 통합이 자율주행차 안전성을 위한 고신뢰성 테스트 솔루션을 제공할 수 있음을 강조합니다.



### Graph Conditioned Diffusion for Controllable Histopathology Image Generation (https://arxiv.org/abs/2510.07129)
- **What's New**: 본 논문에서는 Graph-Conditioned-Diffusion(GCD)라는 새로운 방법론을 제안하여, 의료 영상의 조건부 생성을 위한 그래프 기반 객체 수준 표현을 활용합니다. 이 방법은 각 주요 구조에 해당하는 그래프 노드를 생성하여 개별 특징과 관계를 캡슐화하고, 텍스트 조건화 메커니즘을 통해 확산 모델에 통합합니다. 이는 생성 과정에서의 세밀한 제어를 가능하게 하여 고품질의 이미지 생성을 이루도록 합니다.

- **Technical Details**: GCD는 확률적 생성 모델이 생성하는 데이터의 편향성을 줄이기 위해 그래프 구조를 도입합니다. 그래프는 이미지의 기본 특징 내에서 명확한 구조를 제공하여, 샘플의 균형과 다양성을 관리할 수 있는 수단을 마련합니다. 이론적 프레임워크를 통해 diffusion process의 전진 및 후진 과정을 기술하며, 주어진 그래프를 바탕으로 이미지 생성 작업을 조정합니다.

- **Performance Highlights**: 실제 조직 병리학의 사례를 통해 제안된 방법이 주석이 달린 환자 데이터를 신뢰성 있게 대체할 수 있음을 입증했습니다. GCD는 기존 데이터의 통계적 특성과 밀접하게 일치하는 합성 데이터 생성을 통해, 진단 애플리케이션과 데이터 공유의 유틸리티를 증가시킵니다. 특히, 영상 분할 작업과 같은 다운스트림 작업에서 실제 데이터에 대한 성능 향상을 보여줍니다.



### Opt-ICL at LeWiDi-2025: Maximizing In-Context Signal from Rater Examples via Meta-Learning (https://arxiv.org/abs/2510.07105)
Comments:
          NLPerspectives: The 4th Workshop on Perspectivist Approaches to Natural Language Processing at EMNLP 2025

- **What's New**: 이 논문에서는 자연어 처리(NLP)에서 주관성, 모호성 및 주석자 간의 합법적 의견 차이를 모델링하기 위한 새로운 시스템을 제시합니다. 저자들은 언어 모델(LLMs)의 in-context learning 기능을 활용하여 데이터셋에 대한 포스트 훈련과 메타 학습을 통해 모델을 전문화하는 두 단계의 훈련 절차를 수립했습니다. 이 시스템은 'Learning With Disagreements'(LeWiDi) 대회에서 두 가지 작업 모두에서 전체 우승을 차지했으며, 각 시스템 구성 요소의 중요성을 측정하기 위한 ablation study를 수행했습니다.

- **Technical Details**: 제안된 시스템(Opt-ICL)은 'perspectivist' 접근 방식을 채택하여 각 개별 주석자가 각 인스턴스를 어떻게 평가했는지를 예측하고, 이러한 개별 예측을 집계하여 soft task를 수행합니다. 이 시스템은 LLM의 in-context learning 능력을 활용하며, 포스트 훈련과 데이터셋 특화 fine-tuning 과정을 포함합니다. Spectrum Tuning 방법론을 사용하여 다양한 데이터셋에서 인간 변동성 및 불확실성을 처리할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: 제안된 시스템은 각 구성 요소가 성능에 미치는 영향을 분석하기 위해 ablation study를 수행했고, 특히 in-context에서 주석자 예시를 포함하는 것이 성능에 중요하다는 것을 발견했습니다. 대규모 데이터셋에서 데이터셋 특화 fine-tuning이 도움이 되었고, 다른 in-context 데이터셋에서의 포스트 훈련이 경쟁 데이터셋 중 하나에서 성능 향상에 기여했습니다. 또한, 모델 크기가 커질수록 성능이 향상되는 경향이 있음을 확인했습니다.



### Generative World Modelling for Humanoids: 1X World Model Challenge Technical Repor (https://arxiv.org/abs/2510.07092)
Comments:
          6 pages, 3 figures, 1X world model challenge technical report

- **What's New**: 1X World Model Challenge는 로봇과 인공지능의 새로운 벤치마크로, 인간과의 상호작용을 다루고 있다. 이 챌린지는 미래 이미지 프레임 예측에 중점을 둔 샘플링 트랙과 미래 이산 잠재 코드 예측에 중점을 둔 압축 트랙으로 구성된다. 연구팀은 두 트랙 모두에서 최우수 성적을 달성하여 1위를 기록했다.

- **Technical Details**: 샘플링 트랙에서는 예측 프레임을 생성하기 위해 Wan-2.2 TI2V-5B를 수정하여 비디오 상태로 조건화하였다. 압축 트랙에서는 처음부터 Spatio-Temporal Transformer 모델을 훈련시켰다. 각 모델은 예측 프레임의 품질을 PSNR(Peak Signal-to-Noise Ratio) 지표를 통해 평가하였다.

- **Performance Highlights**: 샘플링 작업에서는 PSNR 23.0 dB을 달성하였고, 압축 작업에서는 Top-500 Cross-Entropy 6.6386을 기록하였다. 이런 뛰어난 성능은 로봇의 상태와 행동을 효과적으로 반영한 결과로, 향후 세계 모델을 활용한 로봇 연구에 큰 기여를 할 것이 기대된다.



### HTMformer: Hybrid Time and Multivariate Transformer for Time Series Forecasting (https://arxiv.org/abs/2510.07084)
- **What's New**: 이번 논문은 시계열 예측에서 Transformer 기반 방법의 한계를 극복하기 위한 새로운 접근 방식인 Hybrid Temporal and Multivariate Embeddings (HTME) 추출기를 제안합니다. HTME는 가벼운 시간 특성 추출 모듈과 다변량 특성 추출 모듈을 통합하여 정보의 풍부한 표현을 제공합니다. 이를 통해 기존 모델에서 발생하는 계산 비용을 줄이고, 성능 향상을 동시에 달성할 수 있는 HTMformer라는 경량 예측 모델을 개발하였습니다.

- **Technical Details**: HTMformer는 시계열 예측을 위한 일반적인 모델로, 인코더 전용 아키텍처를 채택하고 HTME 추출기를 포함합니다. HTME 추출기는 시계열 데이터의 시간 및 다변량 특성을 효과적으로 포착하도록 설계되었습니다. 또한, RevIN 정규화 기법을 사용하여 모델의 학습 및 일반화 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, HTMformer는 여덟 개의 실제 데이터셋에서 기존의 예측 모델들과 비교했을 때 정확도와 효율성 모두에서 뛰어난 성능을 보였습니다. 이는 다변량 특성과 시간적 특성을 통합하여 예측의 의미적 풍부함을 향상시켰기 때문입니다. 특히, HTMformer는 다양한 기준선 모델에 대해 지속적으로 최첨단 성능을 달성하였습니다.



### Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications (https://arxiv.org/abs/2510.07077)
Comments:
          Accepted to IEEE Access, website: this https URL

- **What's New**: 최근 큰 변화가 일어나고 있는 로봇 공학 분야에서, Vision-Language-Action (VLA) 모델이 주목받고 있습니다. VLA 모델은 전통적으로 따로 연구되어온 시각, 언어, 행동 데이터를 통합하여, 다양한 작업 및 환경에서 일반화된 정책을 학습하는 것을 목표로 합니다. 이로 인해 로봇은 최소한의 추가 데이터로도 새로운 작업을 수행할 수 있는 가능성이 높아집니다.

- **Technical Details**: 이 논문은 VLA 모델의 구조적 전환 및 중앙 요소를 체계적으로 검토합니다. VLA 모델은 시각적 관찰과 자연어 지시를 입력으로 받아 로봇 액션을 직접 생성하는 시스템으로 정의됩니다. 또한, 데이터 수집, 공공 데이터셋, 데이터 증강 방법 및 평가 기준과 같은 로봇 시스템의 실질적인 배치를 지원하기 위한 다양한 요소가 포함되어 있습니다.

- **Performance Highlights**: VLA 모델은 적은 양의 특정 작업 데이터로도 다양한 로봇 임무를 수행할 수 있는 잠재력을 지니고 있습니다. 그러나 VLA 모델의 발전은 데이터 가용성, 신체적 불일치, 계산적 제약 등 여러 도전에 의해 제한되고 있습니다. 이러한 문제를 해결하기 위해 다음 세대의 로봇 시스템의 효율성과 접근성을 높이기 위한 연구가 필요합니다.



### LuxInstruct: A Cross-Lingual Instruction Tuning Dataset For Luxembourgish (https://arxiv.org/abs/2510.07074)
Comments:
          Paper under review; Dataset available at this https URL

- **What's New**: 이번 연구는 룩셈부르크어(Luxembourgish)와 같은 저자원 언어를 위한 크로스링구얼(다국어) 지침 튜닝 데이터셋을 구축하여, 고품질의 언어 데이터를 제공하는 것을 목표로 하고 있습니다. 기존의 기계 번역(Machine Translation) 방식 대신에, 영어, 프랑스어, 독일어와 정렬된 데이터를 활용하여 언어적 및 문화적 뉘앙스를 유지한 데이터셋을 생성하였습니다. 이는 기계 번역 데이터의 일반적인 문제점을 회피하면서도 저자원 언어의 발전에 직접적인 혜택을 줄 수 있음을 입증합니다.

- **Technical Details**: 기존의 지침 데이터셋 요구는 고품질 언어 리소스가 부족한 룩셈부르크어처럼 저자원 언어에서 큰 도전 과제가 되고 있습니다. 연구자들은 주로 기계 번역 기술에 의존하여 지침 데이터를 생성했지만, 이는 의미의 미스알ignment 및 문화적으로 부적절한 응답을 초래할 수 있었습니다. 이를 해결하기 위해, 연구팀은 Wikipedia, 뉴스 기사, 그리고 온라인 사전을 주요 소스로 사용하는 역 지침 생성 방법을 채택하여 고품질의 데이터를 수집하였습니다.

- **Performance Highlights**: 테스트 결과, 크로스링구얼 지침 튜닝이 룩셈부르크어의 표현적 정렬을 향상시키고, 모델의 생성 능력을 향상시키는 데 기여하는 것으로 나타났습니다. 연구팀은 다양한 언어로의 지침을 포함한 데이터셋을 사용하여 벤치마크 실험을 수행하였고, 그 결과 크로스링구얼 접근 방식이 표현력 및 지침 응답 능력에서 효과적이라는 것을 확인하였습니다. 이는 룩셈부르크어 LLM의 개발에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Introspection in Learned Semantic Scene Graph Localisation (https://arxiv.org/abs/2510.07053)
Comments:
          IEEE IROS 2025 Workshop FAST

- **What's New**: 이 연구는 학습된 자기 지도 대비(contrastive) 의미 지역화(semantic localisation) 프레임워크에서 의미가 지역화 성능과 강건성에 미치는 영향을 조사합니다. 모델은 원본 지도(original map)와 편향된 지도(perturbed map)에서 훈련을 받은 후, 환경 소음에서 필터링을 수행하고, 일상적인 혼잡보다 독특한 랜드마크를 우선시하는지에 대한 철저한 자기 성찰(post-hoc introspection) 분석을 수행합니다. 다양한 해석 가능성 방법(interpretability methods)을 검증하고, 통합 그래디언트(integrated gradients)와 주의 가중치(attention weights)가 가장 신뢰할 수 있는 모델 행동 탐색 기법으로 자리 잡았습니다.

- **Technical Details**: 연구는 하이레벨(higher-level) 의미 정보가 로컬라이제이션에 어떻게 활용될 수 있는지를 보여주며, 인간과 비슷하게 환경 소음을 필터링하는 방법을 모사합니다. 3D 씬 그래프(scene graph)를 활용하여 공간 개념(노드)과 관계(엣지)를 모델링합니다. 데이터셋으로는 포토 리얼리스틱(photorealistic) 환경과 완전한 메트릭-의미 주석(metric-semantic annotations)을 포함한 uHumans2 데이터셋을 사용합니다.

- **Performance Highlights**: 연구 결과는 모델이 시각적 및 구조적 변형이 존재하는 상황에서도 노이즈에 강하고 의미적으로 중요한 관계를 학습함으로써 설명 가능한 지역화를 가능하게 함을 나타냅니다. 클래스 제거에 따른 성능 저하 분석과 속성 기여 변동 분석을 통해, 자주 등장하는 객체는 다운 가중치 처리되고, 희귀한 랜드마크가 지역 지정 해소에 중요한 역할을 담당하는 것으로 나타났습니다. 통합 그래디언트와 주의 가중치의 신뢰성 분석을 수행하였고, 이들은 강력한 객체 중요도 속성 신호를 제공합니다.



### Search-R3: Unifying Reasoning and Embedding Generation in Large Language Models (https://arxiv.org/abs/2510.07048)
- **What's New**: 이번 연구에서는 Search-R3라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLM)을 검색 작업에 적합하도록 적응시켜, 해결 과정의 직접적인 결과로 검색 임베딩(search embeddings)을 생성합니다. Search-R3는 LLM의 단계적 사고(chain-of-thought) 능력을 활용하여 복잡한 의미 분석을 통해 더 효과적인 임베딩을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: Search-R3는 세 가지 보완 메커니즘을 통해 구현됩니다. 첫째, 감독 학습(supervised learning) 단계는 모델이 품질 높은 임베딩을 생성할 수 있도록 돕습니다. 둘째, 강화 학습(reinforcement learning, RL) 방법론이 임베딩 생성과 추론을 최적화하며, 셋째, 진화하는 임베딩 표현을 효과적으로 처리하는 특화된 RL 환경을 제공하여 각 훈련 Iteration마다 전체 코퍼스의 재부호화 없이 임베딩을 다룰 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에 대한 포괄적인 평가를 통해, Search-R3가 기존 방법들보다 훨씬 우수한 성능을 발휘함을 확인했습니다. 이 통합된 POST-training 접근 방식은 복잡한 지식 중심의 작업을 처리하는 데 있어 획기적인 진전을 이루며, 정교한 추론과 효과적인 정보 검색을 모두 요구하는 시나리오에서 중요한 기여를 합니다.



### Unified Molecule Pre-training with Flexible 2D and 3D Modalities: Single and Paired Modality Integration (https://arxiv.org/abs/2510.07035)
Comments:
          CIKM 2025

- **What's New**: 이번 논문에서는 FlexMol이라는 새로운 분자 사전 학습 프레임워크를 제안합니다. FlexMol은 2D와 3D 방식의 분자 표현을 통합하여, 단일 모달리티 입력을 지원하며, 학습 과정에서 양쪽 모달리티의 정보를 효과적으로 융합합니다. 이 접근 방식은 다양한 분자 특성 예측 태스크에서 우수한 성능을 보여주며, 불완전한 데이터에서도 효과적이임을 입증하였습니다.

- **Technical Details**: FlexMol은 2D 및 3D 분자 데이터를 위한 별도의 모델을 사용하고 매개변수 공유(parameter sharing)를 통해 계산 효율성을 높입니다. 또한, 누락된 모달리티를 생성하기 위한 디코더(decoder)를 활용하여, 학습 중에 양쪽 모달리티가 협력적으로 기여하는 다단계 연속 학습 과정을 구현합니다. 이를 통해 다양한 조합의 데이터로 유연한 멀티모달 학습이 가능합니다.

- **Performance Highlights**: FlexMol은 3.4M의 페어된 샘플과 2M의 단일 모달리티 샘플로 훈련되었으며, 특정 벤치마크 과제에서 10M 이상의 샘플로 사전 훈련된 대규모 분자 모델보다 뛰어난 성능을 달성했습니다. 본 논문은 2D 및 3D 모달리티의 효과적인 정렬과 융합을 통해 경쟁력 있는 성능을 제공하며, 단일 모달리티 입력에서도 멀티모달 학습을 가능하게 합니다.



### Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledg (https://arxiv.org/abs/2510.07024)
- **What's New**: 논문은 사실 기반 지식에 대한 심층 분석을 수행하여 챗봇 및 자연어 처리(NLP)에서의 LLM(대규모 언어 모델)의 역할을 재조명합니다. GPT-4.1에 기반한 GPTKB v1.5를 통해 1억 개의 신념을 조사한 결과, 이러한 모델이 기존 지식 베이스와 상당히 다른 지식을 가지고 있음이 드러났습니다. 이 연구는 LLM의 지식 정확도가 기존 벤치마크에서 제시된 것보다 낮음을 강조하며, 이로 인해 LLM의 연구에서 발생할 수 있는 많은 기회가 열리게 됩니다.

- **Technical Details**: 연구는 GPT-4.1과 함께 사용된 GPTKB v1.5를 통해, 대규모로 Recursive Knowledge Mining 기법을 이용하여 지식을 추출했습니다. 이 방법은 기존 데이터의 재구성을 넘어서며, 100M 이상의 사실적 주장을 포함하는 지식 베이스를 생성하는 데 14,000달러의 비용이 들었습니다. 연구진은 LLM의 지식 조직 방식이 STEM(과학, 기술, 공학 및 수학) 중심으로 되어 있음을 발견하여, LLM의 데이터와 이해도가 사회 및 인문학보다 과학 기술 관련 지식에 좀 더 편향되어 있음을 보여주었습니다.

- **Performance Highlights**: GPT-4.1 모델의 사실적 지식 정확도는 75%로, 기존 텍스트 추출 기반의 지식 베이스보다 높지만, 인간이 편집한 자원 및 일반 LLM 벤치마크에서는 여전히 낮습니다. 분석 결과에서는 일관성, 애매성 및 환각 현상이 주요 문제로 드러났으며, 이는 LLM의 사실적 지식과 관련된 도전 과제와 미래 연구 기회를 제시합니다. GPTKB v1.5는 이러한 LLM의 지식 또는 신념을 심층적으로 조사하는 데 있어 유일무이한 자원으로 자리매김하고 있습니다.



### Federated Unlearning in the Wild: Rethinking Fairness and Data Discrepancy (https://arxiv.org/abs/2510.07022)
- **What's New**: 이 논문은 데이터 삭제 권리를 시행하기 위한 기계 학습의 새로운 방향인 Federated Unlearning (FU)에 대한 최신 연구 결과를 제시합니다. 기존의 FU 방법론이 정확성에만 집중했던 반면, 본 연구에서는 공정성(fairness)과 데이터 분포 차이를 고려한 새로운 접근법인 Federated Cross-Client-Constrained Unlearning (FedCCCU)을 제안하여 이러한 단점을 극복합니다. 실험 결과에 따르면, 기존의 방법들이 실제 환경에서 효과적이지 않은 반면, 저자들의 제안하는 방법은 일관되게 더 나은 성능을 보여줍니다.

- **Technical Details**: Federated Learning (FL)에서는 개별 클라이언트들이 로컬 데이터를 이용해 모델을 학습하고, 중앙 서버에서 이를 집계하여 글로벌 모델을 형성합니다. 하지만, 데이터 삭제와 같은 기계 학습의 형태인 Unlearning을 수행할 때, 기존 방법들은 모든 클라이언트가 재학습에 참여해야하므로 계산 자원과 협력 비용이 상당히 증가합니다. FedCCCU 방법은 이러한 공정성을 고려하여 각 클라이언트의 고유한 데이터 분포를 반영하여, 비기억 클라이언트의 지식 손실을 최소화하도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, FedCCCU는 실제 데이터의 비균질성(hétérogénéité)과 공정성을 고려했을 때, 기존의 모든 분석된 FU 방법들의 성능에 비해 현저히 우수한 성능을 보여주었습니다. 특히, FedCCCU는 그 과정에서 발생 가능한 추가 비용이나 자원 낭비를 최소화하면서도 유의미한 성과를 내는데 성공했습니다. 이를 통해 저자는 FU 연구의 미래 방향으로 공정성과 데이터 현실성을 함께 고려할 것을 제안합니다.



### Native Hybrid Attention for Efficient Sequence Modeling (https://arxiv.org/abs/2510.07019)
Comments:
          Technical report, 16 pages

- **What's New**: 본 연구에서 우리는 Native Hybrid Attention (NHA)라는 새로운 하이브리드 아키텍처를 소개합니다. NHA는 선형 주의와 풀 주의(intra & inter-layer hybridization)를 통합하여 단일 계층 설계로 구성되어 있습니다. 이 구조는 긴 문맥을 유지하면서도 단기간의 정보를 효과적으로 결합하여 성능 향상을 이룹니다.

- **Technical Details**: NHA는 선형 RNN에 의해 업데이트된 키-값 슬롯(key-value slots)을 통해 긴 문맥을 관리합니다. 이는 슬라이딩 윈도우(Sliding Window)의 짧은 기간 토큰들과 결합되어, 단일 softmax attention 작업을 통해 모든 키와 값에 적용됩니다. 각 레이어의 동작은 슬라이딩 윈도우 크기라는 단일 하이퍼파라미터로 조절되어, 모델 구조를 변경하지 않고도 선형 주의와 풀 주의 간의 균형을 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과 NHA는 리콜이 중요한 작업 및 상식 추론(test tasks)에서 Transformers 및 기타 하이브리드 모델을 초월하는 성능을 보여주었습니다. 사전 훈련된 LLM을 NHA 구조에 적용한 결과, 경쟁력 있는 정확도를 달성하며 효율성을 크게 향상시켰습니다. 코드 및 모델은 제공된 URL에서 확인할 수 있습니다.



### Pragyaan: Designing and Curating High-Quality Cultural Post-Training Datasets for Indian Languages (https://arxiv.org/abs/2510.07000)
Comments:
          EMNLP 2025

- **What's New**: 이 논문에서는 인도 언어를 포함한 다국어 고품질 후처리 데이터셋인 Pragyaan-IT와 Pragyaan-Align을 소개합니다. 두 데이터셋은 각각 22.5K와 100K의 예시로 구성되어 있으며, 10개 인도 언어에서 13개 대분류와 56개 소분류를 포함합니다. This approach emphasizes not only linguistic accuracy but also cultural relevance, addressing a significant gap in existing datasets that often ignore local contexts.

- **Technical Details**: 연구진은 효율적이고 질 높은 후처리 데이터셋을 생성하기 위해 human-in-the-loop (HITL) 파이프라인을 활용했습니다. 이는 instruction-following (지시 수용) 작업의 다양성과 문화적 뉘앙스를 중시하며, 다양한 현실 세계 시나리오를 처리할 수 있는 모델을 훈련시키는 데 중점을 두고 있습니다. 데이터셋은 다국어 표현력, 작업 복잡성, 그리고 멀티 턴 대화를 포함한 여러 차원에서 특성을 포괄합니다.

- **Performance Highlights**: 이 연구의 결과로 생성된 Pragyaan 데이터셋은 인도 언어와 문화적 맥락을 밀접하게 반영하는 고품질 데이터를 제공합니다. 우선, 데이터셋은 실제 요구 사항을 바탕으로 구성되어 LLM이 다양한 지시를 잘 수용할 수 있도록 설계되었습니다. 그리고 이를 통해, 후속 실험에서는 데이터셋이 robust한 instruction-following 기능을 갖추었음을 검증하였습니다.



### The Limits of Goal-Setting Theory in LLM-Driven Assessmen (https://arxiv.org/abs/2510.06997)
Comments:
          Accepted at T4E 2025 for poster

- **What's New**: 이 논문은 AI 도구를 사용하는 비전문 사용자들이 AI 시스템을 인간처럼 인식하는 '모델 H'를 제안하며, 이는 사용자들이 상세한 지시를 통해 일관된 평가 행동을 기대하는 것과 관련이 있습니다. 그러나 구체적인 프롬프트와 성과 간의 관계를 조사한 결과, 성과가 일관되게 향상되지 않고 성과의 변동성 또한 크게 변화하지 않았습니다. 이 발견은 대규모 언어 모델(LLM)이 인간 평가자처럼 행동하지 않는다는 가정을 도전합니다.

- **Technical Details**: 연구는 ChatGPT가 29개의 학생 제출물을 평가하는 통제된 실험을 통해 이루어졌으며, 이 과정에서 프롬프트의 구체성이 성과에 미치는 영향을 분석했습니다. 성과 일관성은 반복 실행 간의 intra-rater reliability, 즉 Cohen's Kappa를 사용하여 측정했습니다. 이 연구는 비전문 사용자들이 AI 시스템에 대해 어떤 정신 모델을 가지고 있는지, 그리고 이러한 모델이 어떻게 프롬프트 생성 및 생산성에 영향을 미치는지를 조사합니다.

- **Performance Highlights**: 이 연구의 결과, 비전문 사용자들은 모델 H를 채택하고 있지만, 이는 효과적으로 좋은 프롬프트를 생성하는 데 방해가 될 수 있음을 시사합니다. 사용자들의 목표 specificity가 AI 시스템의 성능과 변동성에 대한 기대와 맞지 않는 경우가 발생했으며, 이는 AI 도구의 접근성과 효율성을 높이기 위해서는 더욱 강건한 설계가 필요함을 강조합니다. 향후 연구에서는 비전문 사용자들을 위한 프롬프트 공학을 개선할 필요가 있음을 제안합니다.



### VelLMes: A high-interaction AI-based deception framework (https://arxiv.org/abs/2510.06975)
Comments:
          9 pages. 9 figures. 1 table. This is a preprint of a paper that was presented at the Active Defense and Deception Workshop colocated with IEEE EuroS&P 2025 conference

- **What's New**: 이번 논문은 VelLMes라는 AI 기반의 새로운 기만 프레임워크를 소개합니다. 이 프레임워크는 SSH Linux shell, MySQL, POP3 및 HTTP와 같은 여러 프로토콜과 서비스의 시뮬레이션을 지원합니다. VelLMes는 사용자 요구에 따라 다양한 기만 설계를 위한 선택지를 제공하며, 인간 사용자와의 상호작용 및 사실감을 중시합니다.

- **Technical Details**: VelLMes는 LLM(대형 언어 모델)을 기반으로 하여 사용자가 시뮬레이션 할 수 있는 다양한 서비스를 제공합니다. SSH Linux shell인 shelLM은 세밀한 프롬프트 엔지니어링을 통해 원하는 행동을 이끌어내며, 모든 출력은 LLM으로 생성되기 때문에 실제 커맨드 실행의 위험이 없습니다. 논문에서는 LLM의 생성 능력과 기만 능력을 평가하는 세 가지 유형의 평가를 수행했습니다.

- **Performance Highlights**: 연구 결과, 89명의 인간 공격자를 대상으로 한 실험에서 전체 공격자의 약 30%가 LLM 기반의 honeypot과 상호작용할 때 실제 시스템으로 혼동하였으며, SSH Linux shell honeypot은 90% 이상의 명령에 대해 정확한 응답을 생성했습니다. 이러한 결과는 LLM이 사이버 기만의 유용한 자원으로 활용될 수 있음을 보여줍니다.



### Learning Global Representation from Queries for Vectorized HD Map Construction (https://arxiv.org/abs/2510.06969)
Comments:
          16 pages

- **What's New**: 본 연구에서는 현대 자율주행 시스템의 핵심인 온라인 고해상도(HD) 지도 구축을 위해 ‘MapGR(글로벌 표현 학습을 통한 HD 지도 구축)’이라는 새로운 아키텍처를 제안합니다. 기존의 DETR 프레임워크 기반 접근 방식은 독립적인 객체 쿼리에 의존하여 주로 국소 쿼리 관점을 강조했으나, HD 지도에서의 고유한 글로벌 표현을 간과했습니다. 이를 해결하기 위해, MapGR은 글로벌 표현을 학습하고 활용하는 두 가지 모듈인 글로벌 표현 학습(GRL) 모듈과 글로벌 표현 유도(GRG) 모듈을 도입합니다.

- **Technical Details**: GRL 모듈은 모든 쿼리로부터 글로벌 HD 지도 표현을 학습하고, 이를 통해 포괄적인 레스터화된 지도를 예측합니다. 이 예측은 Ground Truth(실제 지도)와 비교하여 감독됩니다. GRG 모듈은 GRL 모듈에서 학습한 글로벌 표현을 각 개별 쿼리에 통합함으로써 최적화를 돕습니다. 이 방식은 개별 쿼리를 최적화하면서도 글로벌한 관점을 유지할 수 있도록 합니다.

- **Performance Highlights**: nuScenes 및 Argoverse2 데이터셋에 대한 평가 결과, 제안한 방식은 평균 정밀도(mean Average Precision, mAP)에서 기존의 주요 기준선보다 상당한 성능 향상을 보여주었습니다. MapGR은 Online HD 지도 구축을 위한 효과적이고 효율적인 접근 방식으로, 주요 방법들과의 함께 사용할 수 있는 플러그 앤 플레이 모듈로 설계되었습니다. 많은 실험을 통해 제안된 접근 방식이 다양한 기준선에서 성능을 상당히 개선함을 확인했습니다.



### Generating Surface for Text-to-3D using 2D Gaussian Splatting (https://arxiv.org/abs/2510.06967)
- **What's New**: 최근 Text-to-3D 모델링에서의 발전은 3D 콘텐츠 생성의 중요한 가능성을 보여주고 있습니다. 하지만 자연계의 복잡한 기하학적 형태로 인해, 진정한 3D 콘텐츠 생성은 여전히 도전적인 과제입니다. 본 논문에서는 DirectGaussian이라는 새로운 방법을 제안하며, 이는 surfel로 표현된 3D 객체의 표면 생성에 중점을 두고 있습니다.

- **Technical Details**: DirectGaussian에서는 조건부 텍스트 생성 모델을 활용하고, 2D Gaussian splatting 기법으로 3D 객체의 표면을 렌더링합니다. 또한, 다중 뷰 기하학적 일관성 문제를 해결하기 위해 최적화 과정에서 생성된 표면에 곡률 제약을 통합했습니다. 이 접근법은 고품질의 3D 콘텐츠 생성을 위한 중요한 초석을 제공합니다.

- **Performance Highlights**: 다양한 실험을 통해 DirectGaussian이 다양한 텍스트 프롬프트에 대해 고충실도의 3D 콘텐츠 생성을 달성할 수 있음을 입증하였습니다. 또한, 360도 주변 뷰 표면 곡률 제약을 도입하여, 최종 출력에서 세밀한 기하학적 세부정보를 보존할 수 있습니다. 이로써, DirectGaussian은 텍스트 기반의 3D 생성 작업에서 새로운 가능성을 보여줍니다.



### EDUMATH: Generating Standards-aligned Educational Math Word Problems (https://arxiv.org/abs/2510.06965)
Comments:
          32 pages, 15 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)이 학생의 흥미와 능력 수준에 맞춰 수학 단어 문제(Math Word Problems, MWPs)를 자동으로 생성할 수 있음을 제안합니다. 이를 위해 연구팀은 11,000개 이상의 MWPs를 평가하기 위해 인간 전문가와 LLM을 결합한 평가 방식을 사용하였으며, 교수 표준에 맞춘 교육용 MWP 생성을 위한 최초의 교사 주석 데이터 세트를 개발하였습니다. 연구 결과, 대규모 언어 모델이 더 큰 모델과 동등한 성능을 발휘할 수 있는 가능성을 보여주었습니다.

- **Technical Details**: 이 연구에서는 수학 교육 기준과 학생의 흥미 및 능력 수준에 적합한 MWP 생성을 위해 네 가지 기준, 즉 solvability, accuracy, educational appropriateness, standards alignment를 사용하여 LLM의 성능을 평가합니다. 연구는 일정한 단계로 진행되며, 첫 번째 단계에서 교사들이 주석을 단 데이터셋을 사용하여 3,000개 이상의 MWPs를 생성하고, 이를 기반으로 12B와 30B LLM을 훈련시킵니다. 이를 통해 생성된 문제들의 품질을 높이고, 데이터를 활용하여 교사가 바라는 MWP 생성을 위한 새로운 데이터셋을 제공합니다.

- **Performance Highlights**: 연구 결과, LLM들이 생성한 MWPs는 기존 모델보다 더 높은 품질을 보이며, 인간이 작성한 MWPs와 더 유사한 것으로 나타났습니다. 또한, 학생들이 LLM이 생성한 MWP를 인간이 작성한 문제 보다 선호하는 경향을 보였으며, 이 질문들에서 비슷한 성과를 내는 것으로 분석되었습니다. 연구의 주요 기여점으로는 LLM의 성능 격차를 줄이고, 최초의 표준 맞춤형 교육용 MWP 생성 데이터셋을 만들어 내며, K-12 교육에서의 활용 가능성을 입증한 것이라 할 수 있습니다.



### Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation (https://arxiv.org/abs/2510.06961)
Comments:
          Submitted to ICASSP 2026; Leaderboard: this https URL Code: this https URL

- **What's New**: 이번 논문은 Open ASR Leaderboard를 소개하며, 60개 이상의 오픈소스 및 상용 ASR 시스템을 11개의 데이터 세트에서 비교하는 벤치마크를 제공합니다. 표준화된 텍스트 정규화 방법을 통해 단어 오류율(Word Error Rate, WER)과 역 실시간 요인(Inverse Real-Time Factor, RTFx)을 보고하여 정량적인 평가를 가능하게 하였습니다. 이는 개발자와 사용자 모두가 성능 및 효율성을 기반으로 한 평가를 보다 공정하게 할 수 있게 합니다.

- **Technical Details**: Open ASR Leaderboard는 영어 전사와 다국어 전사(독일어, 프랑스어, 이탈리아어 등) 그리고 30초 이상의 긴 오디오에 대한 평가를 포함합니다. 모델은 WER에 따라 평가되며 실시간 요인을 추정할 수 있도록 정리된 데이터 세트를 활용하여 숫자 정규화, 철자 표준화와 같은 텍스트 정규화를 거칩니다. 현재 64개의 모델이 등록되어 있으며, 그 중 57개가 오픈소스입니다.

- **Performance Highlights**: 짧은 영어 전사에서 Conformer 인코더와 LLM 기반 디코더의 조합이 우수한 성과를 보입니다. 그러나 이러한 구조는 TDT와 CTC 디코더를 사용하는 모델보다 느린 속도를 보이는 경향이 있습니다. 또한 자기 지도 학습(self-supervised learning, SSL) 기반의 모델이 여전히 뛰어난 성능을 발휘할 수 있지만, 현재 영어 전사 용 TOP SSL 시스템은 A100 성능 평가에서 상대적으로 낮은 순위를 차지하고 있습니다.



### Grouped Differential Attention (https://arxiv.org/abs/2510.06949)
- **What's New**: 이번 논문에서는 Grouped Differential Attention (GDA)라는 새로운 접근 방식을 제안합니다. GDA는 신호 보존 그룹과 노이즈 제어 그룹 간의 비대칭 헤드 할당을 도입하여 기존의 대칭 할당 방식을 개선합니다. 이를 통해 신호 추출을 위한 더 많은 헤드를 배분하고, 노이즈 제어 그룹은 더 적은 용량으로 운영되며, controlled repetition을 통해 안정성을 유지합니다.

- **Technical Details**: GDA는 Differential Attention의 한계를 극복하기 위해 다수의 헤드를 비율 기반으로 신호 보존 그룹에 할당하여 신호 충실도를 개선합니다. 이 과정에서 사용되는 주의 메커니즘은 두 개의 상보적인 주의 지도(attention maps)를 생성하여 신호와 노이즈를 분리한 후 이를 결합합니다. 그 결과, 필요한 computational overhead를 최소화하면서도 강력한 신호 추출을 가능하게 합니다.

- **Performance Highlights**: 대규모 사전 학습 및 지속적 학습 실험을 통해, GDA는 일반화 성능과 안정성에서 기존의 대칭 설계 모델 대비 현저히 향상된 결과를 보여주었습니다. 특히, 신호 보존과 노이즈 제어 헤드 간의 적절한 비율(예: 3:1 또는 4:1)이 효과적인 성능 향상을 이끈다는 것을 발견했습니다. 이 연구는 자원 효율적인 Transformer 아키텍처 설계를 위한 실용적인 경로를 제시합니다.



### Expressive and Scalable Quantum Fusion for Multimodal Learning (https://arxiv.org/abs/2510.06938)
Comments:
          22 pages, 4 figures

- **What's New**: 이 논문은 다중 모달 학습을 위한 양자 융합 메커니즘을 소개하고 그 이론적 및 경험적 가능성을 수립한다. 제안된 방법인 양자 융합 레이어(Quantum Fusion Layer, QFL)는 전통적인 융합 기법을 하이브리드 양자-고전적 절차로 대체하며, 이는 매개변수 양의 지수적 성장을 요구하지 않고 얽힌 특징 상호작용을 학습할 수 있도록 한다. QFL은 양자 신호 처리 원리의 지원을 통해 높은 차수의 다항식 상호작용을 효율적으로 표현할 수 있으며, 이는 QFL과 낮은 차원의 텐서 기반 방법 간의 차별점을 강조하는 예제를 포함하고 있다.

- **Technical Details**: 다중 모달 융합은 다중 모달 기계 학습 모델의 성공을 결정짓는 중요한 단계이다. 이 논문에서는 우선 각 모달리티를 적합한 일모달 인코더로 처리하여 의미있는 추상 표현을 추출한 후, 이를 결합해 공동 표현을 학습하는 하이브리드 접근 방식에 초점을 맞춘다. QFL은 이러한 구조를 통해 고차 다항식 상호작용을 효율적으로 캡처하며, 매개변수 수의 선형 성장을 통해 다면적인 다중 모달 상호작용을 표현할 수 있도록 설계되었다.

- **Performance Highlights**: QFL은 다양한 작업에서 LMF 및 GNN 기반 접근 방식과 비교하여 성능이 우수한 것으로 나타났다. 특히 고차 모달리티 환경에서 가장 뚜렷한 성과를 보이며, QFL의 이론적 장점이 실제 성능에 잘 반영되고 있다. 본 연구는 QFL이 다중 모달 융합에 있어 본질적으로 새로운 접근 방식을 제공하며, 보다 큰 시스템에 대한 심도 있는 탐구가 필요함을 시사한다.



### Bayesian Nonparametric Dynamical Clustering of Time Series (https://arxiv.org/abs/2510.06919)
Comments:
          This work has been submitted to the IEEE for possible publication. 15 pages. 9 figures

- **What's New**: 이번 논문에서는 선형 동역학을 가진 미지의 레짐 사이를 전환하여 무한 개의 시계열 클러스터의 진화를 모델링하는 방법을 제시합니다. 베이지안 비모수적 접근법을 기반으로 계층적 디리클레 과정 (hierarchical Dirichlet process)을 사용하여 스위칭 선형 동적 시스템 (Switching Linear Dynamical System)의 매개변수를 설정하며, 각 클러스터 내에서 통계적 변화를 모델링하기 위해 가우시안 과정 (Gaussian process) 우선을 활용합니다. 이 방법은 시계열 패턴의 진화를 모형화함으로써 클러스터의 불필요한 폭발을 피하는 원칙적인 방식을 제공합니다.

- **Technical Details**: 기본 개념으로는 시계열 클러스터를 서로 다른 길이를 가진 세그먼트로 클러스터링하는 방법이 포함됩니다. 가우시안 프로세스 (GP) 우선 규칙을 사용하여 다양한 패턴을 인식하는 모델을 유도하며, 이는 모양 변화가 시간이 지남에 따라 변화하는 클러스터를 발견할 수 있게 해줍니다. 이 글에서는 또한 동적 시간 왜곡 (Dynamic Time Warping, DTW) 방법과 그 변형을 통해 시간의 비정렬을 처리하는 방법에 대해서도 논의합니다.

- **Performance Highlights**: ECG 데이터를 분석하기 위한 여러 사례 연구를 통해 제안된 접근법의 다양성과 효과성을 입증하였습니다. 실험 결과, 이 방법이 ECG 신호의 동적 변화를 효율적으로 포착하고 변별 가능성을 증가시킴을 나타냈습니다. 최종적으로, 이 방법은 클러스터링 및 정렬 과정을 동시에 수행할 수 있는 베이지안 프레임워크 안에서의 사용을 제안합니다.



### LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling (https://arxiv.org/abs/2510.06915)
- **What's New**: 이 논문에서는 Long-RewardBench라는 새로운 벤치마크를 도입하여, 보상 모델(Reward Model, RM)을 긴 콘텍스트에서 평가할 수 있는 방법을 제시합니다. 기존 RM들이 짧은 콘텍스트에서만 잘 작동하고, 긴 콘텍스트에서의 일관성을 간과한다는 한계를 지적합니다. 이를 통해, 다양한 모델의 긴 콘텍스트에 대한 이탈성 높은 평가가 이루어질 수 있도록 하고자 하였습니다.

- **Technical Details**: Long-RewardBench는 각 테스트 세트가 질문(question), 콘텍스트(context), 모델 응답(set of model responses), 정답(ground-truth prediction)의 네 가지 요소로 구성된 벤치마크입니다. 이 연구에서는 일반적인 다단계 학습 전략을 설계하여 기존 모델을 긴 컨텍스트의 보상 모델(LongRMs)로 확장하기 위한 방법론을 개발하였습니다. 특히, 데이터 합성을 통해 훈련 과정의 각 단계에서 높은 품질의 데이터를 생성하는 방식이 도입되었습니다.

- **Performance Highlights**: 예비 연구 결과에 따르면, 기존의 최신 생성 모델들이 긴 콘텍스트 상황에서 성능이 크게 때때로 50% 미만으로 떨어진다는 문제가 드러났습니다. 그러나 제안된 방법론으로 훈련된 8B LongRM 모델은 70B 기준 모델들보다 우수한 성능을 보이며, 비공식 모델인 Gemini 2.5 Pro와 동등한 성능을 달성하였습니다. 이러한 결과는 짧은 콘텍스트에서의 성능 또한 유지되고 있음을 시사합니다.



### DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning (https://arxiv.org/abs/2510.06913)
- **What's New**: 이번 연구에서 제안하는 DecompGAIL(Decomposed Multi-agent Generative Adversarial Imitation Learning)은 자율 주행 시스템과 교통 시뮬레이션의 주요 문제인 불안정성을 해결하기 위해 설계되었습니다. 연구팀은 다중 에이전트 환경에서의 상호작용이 비효율적으로 평가되는 원인을 규명했으며, 이로 인해 학습된 정책의 안정성이 저하되는 문제점을 지적하였습니다. DecompGAIL은 ego-맵(e.g. map)과 ego-이웃(e.g. neighbor) 관련성을 명확히 분리하여 비정상적인 상호작용을 필터링함으로써, 현실적인 동작을 보장합니다.

- **Technical Details**: DecompGAIL은 전통적인 GAIL에서 발생하는 훈련의 불안정성을 해결하기 위해, 인간 운전자의 복잡한 행동을 모델링하기 위한 차별화된 접근 방식을 사용합니다. 이 방법은 각 에이전트의 행동을 ego-맵 및 ego-이웃으로 세분화하여 평가하고, 비관련 상호작용을 억제합니다. 추가적으로 사회적 PPO(Social PPO) 목표를 도입하여 에이전트의 리얼리즘을 높이는데 기여하며, 근처 에이전트의 행동에 영향을 주지 않으면서도 자신의 보상을 최적화합니다.

- **Performance Highlights**: DecompGAIL은 WOMD Sim Agents 2025 벤치마크에서 기존의 GAIL보다 훈련 안정성을 크게 향상시켰으며, 최신 성능을 기록했습니다. 이러한 개선은 차량 충돌 및 비정상적인 동작을 줄이는데 기여하며, 궁극적으로 교통 모델링에서의 실제 대처능력을 확보하는데 중요한 역할을 합니다. 연구 결과에 따르면, 제안된 방법은 다양한 다중 에이전트 시나리오에서 유망한 결과를 보여주었습니다.



### Emotionally Vulnerable Subtype of Internet Gaming Disorder: Measuring and Exploring the Pathology of Problematic Generative AI Us (https://arxiv.org/abs/2510.06908)
Comments:
          27 pages, 5 figures, 5 tables

- **What's New**: 이 연구는 생성형 AI(Generative AI) 사용에 대한 과도한 병리화(over-pathologization) 우려와 관련하여 개념적 명확성이 결여된 문제를 해결하기 위해 PUGenAIS-9(Problematic Use of Generative Artificial Intelligence Scale-9 items)를 개발하고 검증했습니다. 이는 인터넷 게임 장애(IGD) 프레임워크에 따라 PUGenAIS가 중독(addiction) 유사 패턴을 반영하는지를 검토했습니다.

- **Technical Details**: 중국과 미국의 샘플(총 N = 1,508)을 사용하여 확인적 요인 분석(confirmatory factor analysis)을 수행했으며, IGD 기반의 아홉 가지 차원에서 강력한 31개 항목 구조를 발견했습니다. 이후 각 차원에서 최고 적재(item) 항목을 선택하여 PUGenAIS-9를 도출하고, 독립 샘플(N = 1,426)에서 그 구조를 검증했습니다. 측정 불변성(measurement invariance) 테스트를 통해 국적(nationality)과 성별(gender) 간의 안정성을 확인했습니다.

- **Performance Highlights**: PUGenAIS는 IGD의 감정적으로 취약한 하위 유형(emotionally vulnerable subtype)과 잘 일치하며, 능력 기반의 유형(competence-based kind)과는 차별화됩니다. 이러한 결과는 PUGenAIS-9를 문제적 생성형 AI 사용을 식별하는 데 활용할 수 있음을 지지하며, 디지털 중독(digital addiction)을 ICD(구조, 콘텐츠, 장치) 모델에 따라 재고(rethink)할 필요성을 보여줍니다.



### Angular Constraint Embedding via SpherePair Loss for Constrained Clustering (https://arxiv.org/abs/2510.06907)
Comments:
          Accepted by NeurIPS 2025, 6 Figures and 1 Table in Main text, 18 Figures and 5 Tables in Appendices

- **What's New**: 이번 연구에서는 Deep Constrained Clustering (DCC) 방식에서 기존 방법의 한계를 극복하기 위해 SpherePair이라는 새로운 angular constraint embedding 접근법을 제안합니다. SpherePair 손실(SpherePair loss)은 기하학적 구성을 사용하여 쌍(pairwise) 제약조건을 충실하게 인코딩하며, 클러스터링에 적합한 각 공간에서의 임베딩을 생성합니다. 이 접근법은 클러스터 수를 지정할 필요 없이 클러스터링을 수행할 수 있도록 하여, 하이퍼파라미터 조정을 위한 노력을 줄이고 효율성을 개선합니다.

- **Technical Details**: SpherePair 손실 함수는 코사인 유사도(cosine similarity)를 활용하여 앵커 없이 각 공간에서 잠재적 표현(latent representation)을 학습합니다. 이를 통해 긍정 쌍(positive pairs)과 부정 쌍(negative pairs) 간의 적절한 거리를 유지하면서 클러스터 간 거리의 균형을 맞출 수 있습니다. 이 방법은 이론적으로 검증된 기초를 바탕으로 하여 특정 조건에서 최적 성능을 보장합니다.

- **Performance Highlights**: 범위가 다양한 벤치마크 데이터셋에서 기존 DCC 방법들과 비교하여 SpherePair의 우수한 성능을 입증했습니다. 특히, 간단한 K-means 알고리즘을 사용하여 학습된 표현에 대한 클러스터 수를 신속하게 추측할 수 있으며, 미지의 데이터에도 잘 일반화됩니다. 실험 결과는 저자들이 제안한 방법이 대안 방법들보다 효과적이고 실용적임을 보여줍니다.



### M3Retrieve: Benchmarking Multimodal Retrieval for Medicin (https://arxiv.org/abs/2510.06888)
Comments:
          EMNLP Mains 2025

- **What's New**: 이번 논문은 의료 분야의 멀티모달 검색 모델의 필요성과 이를 평가하기 위한 표준 벤치마크가 부족한 상황을 다루고 있습니다. 이를 개선하기 위해 M3Retrieve라는 멀티모달 의료 검색 벤치마크를 제안하며, 이 벤치마크는 의료 분야의 다양한 전문성을 고려하고 있습니다. M3Retrieve는 5개의 도메인과 16개의 의료 분야에 걸쳐 있으며, 120만 개 이상의 텍스트 문서와 16만 4천 개의 멀티모달 쿼리를 포함하고 있습니다.

- **Technical Details**: M3Retrieve는 텍스트와 이미지를 아우르는 멀티모달 데이터를 통합하여 의료 분야의 정보 검색을 위한 보다 현실적인 평가를 가능하게 해줍니다. 이 벤치마크는 22개의 수작업 검토 데이터 세트를 수집하여 16개 의료 전문 분야 모두를 포괄하며, 실제 임상 시나리오를 반영하는 다양한 작업을 포함하고 있습니다. 또한, 환자 정보와 관련된 이미지와 텍스트의 복합적인 해석을 분석하기 위해 5가지 검색 작업을 정의하였습니다.

- **Performance Highlights**: M3Retrieve의 출시로 인해 의료 응용 프로그램에서 신뢰할 수 있는 멀티모달 검색 시스템을 구축하는 데 기여할 것이며, 연구의 발전을 가속화하고 모델 혁신을 촉진할 것입니다. 여러 첨단 멀티모달 검색 모델을 평가하여 특정 의료 전문 분야의 도전 과제를 이해하고 검색 성능에 미치는 영향을 정량화합니다. 이는 특히 안전이 중요한 의료 분야에서 시스템의 신뢰성 향상에 큰 도움이 될 것입니다.



### Multi-Dimensional Autoscaling of Stream Processing Services on Edge Devices (https://arxiv.org/abs/2510.06882)
- **What's New**: 이 논문은 Edge 디바이스에서 서비스 수준 목표(Service Level Objectives, SLOs)를 유지하기 위한 다차원 자동 스케일링 플랫폼(Multi-dimensional Autoscaling Platform, MUDAP)을 최초로 소개합니다. 기존 자동 스케일링 메커니즘이 주로 리소스 스케일링에만 집중하는 반면, MUDAP는 서비스 및 리소스 수준에서 세밀한 수직 스케일링을 지원하여 다양한 서비스를 최적화합니다. 또한 Regression Analysis of Structural Knowledge (RASK) 기반의 스케일링 에이전트를 통해 최적의 스케일링 작업을 추론합니다.

- **Technical Details**: MUDAP은 리소스 제약이 있는 Edge 환경에서 서비스 및 리소스 매개변수의 수직 스케일링을 지원하는 다차원 자동 스케일링 플랫폼입니다. RASK 에이전트는 처리 환경의 연속 회귀 모델을 학습하여 최적의 스케일링 결정을 유도합니다. 이 논문은 Kubernetes VPA와 강화 학습 에이전트 등 기존 자동 스케일러와 비교하여 성능을 평가하였으며, RASK는 단 20회 반복만으로도 정확한 회귀 모델을 추론할 수 있음을 보여줍니다.

- **Performance Highlights**: RASK는 여러 경쟁 처리 서비스 간의 SLO를 충족시키면서 기존 자동 스케일러보다 28% 적은 SLO 위반을 기록하며 최고의 요청 부하를 지속할 수 있었습니다. 이 과정에서 CPU 오버헤드도 거의 발생하지 않았으며, 처리의 단위 데이터를 보기 위해 단 200초의 데이터 수집 후 20회의 신속한 학습이 이루어졌습니다. 이러한 성과는 Edge 디바이스 내에서 동적 조건 하에서도 부하를 관리하는 데 있어 RASK의 효율성을 강조합니다.



### MoRE-GNN: Multi-omics Data Integration with a Heterogeneous Graph Autoencoder (https://arxiv.org/abs/2510.06880)
- **What's New**: 이번 연구에서는 MoRE-GNN(Multi-omics Relational Edge Graph Neural Network)을 소개합니다. 이 모델은 데이터로부터 직접 관계 그래프를 동적으로 구성하는 이종 그래프 자동 인코더로, 생물학적으로 의미 있는 관계를 포착하는 데 중점을 두고 있습니다. MoRE-GNN은 단일 세포 다중 유전체 데이터 통합을 위한 고차원성 문제를 해결하고, 기존 방법들에 비해 우수한 성능을 발휘합니다.

- **Technical Details**: MoRE-GNN의 진행 방식은 세 가지 단계로 나뉘어 있습니다. 첫 번째 단계에서는 모달리티별 코사인 유사성을 사용하여 관계 엣지를 구성하고, 두 번째 단계에서는 GCN(Graph Convolutional Network)과 attention 메커니즘을 이용해 이질적인 메시지 패싱을 수행합니다. 마지막으로, 학습된 임베딩을 2차원으로 투영하고 집단을 Louvain 클러스터링으로 식별합니다.

- **Performance Highlights**: 여섯 개의 공개 데이터셋에서의 평가 결과, MoRE-GNN은 생물학적으로 의미 있는 관계를 포착하면서 기존의 방법들보다 뛰어난 성능을 나타냈습니다. 이 모델은 고차원성 문제를 극복하고, 다양한 데이터셋에 적용 가능하다는 장점을 가지고 있습니다. MoRE-GNN은 다중 유전체 통합 및 단일 세포 분석의 향상을 위한 유망한 도구로 자리 잡을 것으로 기대됩니다.



### Multi-hop Deep Joint Source-Channel Coding with Deep Hash Distillation for Semantically Aligned Image Retrieva (https://arxiv.org/abs/2510.06868)
- **What's New**: 이 논문은 DeepJSCC(Deep Joint Source-Channel Coding)와 딥 해시 증류(DHD: Deep Hash Distillation) 모듈을 결합하여 다중 홉 AWGN(대칭 백색 가우시안 노이즈) 채널을 통한 이미지 전송을 새로운 방법으로 제안합니다. 이를 통해 이미지의 의미적 일관성을 향상시켜 보안 지향적인 응용을 가능하게 하며, 재구성 품질을 개선합니다. 이 새로운 접근 방식은 심상적 클러스터링(semantic clustering)을 통해 채널 노이즈로 인한 의미적 변화를 완화하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DeepJSCC는 이미지 전송을 위한 종합적인 소스-채널 코딩 방법이며, 프로세스에서 MSE(Mean Square Error)와 소스 및 재구성 이미지 간의 코사인 거리(cosine distance)를 최소화하는 훈련을 포함합니다. DHD 모듈은 이미지의 의미를 이해하고, 비슷한 의미를 가진 이미지로부터 유사한 지문(fingerprint)을 생성하여 의미적 클러스터링을 수행합니다. 이 연구는 다중 홉 DF(Decode-and-Forward) 릴레이를 위한 DeepJSCC-DHD 아키텍처를 확장하여 새로운 설계 방안을 제시합니다.

- **Performance Highlights**: 제안된 접근 방식은 채널 노이즈에 의해 발생하는 문제를 해결하면서, LPIPS(학습된 지각 이미지 패치 유사도)를 사용하여 사람들이 인지하는 품질과 일치하는 재구성 품질을 개선하는 데 기여합니다. 다중 홉 환경에서 성능을 측정한 결과, 기존의 DeepJSCC 방식보다 우수한 결과를 보이며, 특히 전위 이미지 전송과 보안 응용 분야에서도 가능성을 보여주었습니다. 이 연구는 향후 다양한 다중 홉 통신 시스템에서의 활용 가능성을 열어줍니다.



### Towards Generalization of Graph Neural Networks for AC Optimal Power Flow (https://arxiv.org/abs/2510.06860)
Comments:
          Pre-print has been submitted for review

- **What's New**: 본 연구는 AC Optimal Power Flow (ACOPF) 문제를 해결하기 위해 Hybrid Heterogeneous Message Passing Neural Network (HH-MPNN) 아키텍처를 제안합니다. 이 네트워크는 전력 시스템의 다양한 구성 요소를 모델링하여 복잡한 토폴로지 변화에 적응할 수 있는 능력을 제공합니다. HH-MPNN은 전통적인 솔버보다 1,000배에서 10,000배 더 빠른 계산 속도를 기록하며, 수천 개의 새로운 토폴로지에 대해 3% 미만의 최적성 차이를 달성합니다.

- **Technical Details**: HH-MPNN은 heterogeneous GNN과 스케일 가능한 transformer를 결합하여 ACOPF 변수를 예측합니다. 이 아키텍처는 전력 그리드의 각 구성 요소를 명시적으로 모델링하여 지역 정보를 효율적으로 집계하며, transformer는 자가 주의(self-attention)를 통해 전역 정보를 교환합니다. 이 과정은 인코딩, 프로세싱, 디코딩의 세 단계로 구성되며, 각 단계에서 지역 및 글로벌 정보를 효과적으로 결합합니다.

- **Performance Highlights**: HH-MPNN은 다양한 그리드 크기에서 우수한 일반화를 보여주며, 작은 그리드에서 사전 훈련된 모델이 더 큰 그리드의 성능을 향상시킵니다. 본 모델은 14부터 2,000까지의 버스를 가진 그리드에서 1% 미만의 최적성 차이를 달성하고, 새로운 N-1 비상 상황에 대해 높은 적응성을 보입니다. 이 결과는 실시간 전력 시스템 운영을 위한 실제적이고 일반izable한 머신러닝 접근 방식을 한층 발전시킵니다.



### Explaining raw data complexity to improve satellite onboard processing (https://arxiv.org/abs/2510.06858)
Comments:
          Preprint: European Data Handling & Data Processing Conference (EDHPC) 2025

- **What's New**: 이 연구는 위성에서 AI 모델을 직접 운용할 수 있는 가능성을 제시하며, 기존에 사용된 전처리된 이미지 대신 원시(원래) 센서 데이터를 활용하는 접근 방식을 탐구합니다. 원시 데이터에 대한 깊이 학습(deep learning) 모델을 통한 객체 탐지(object detection) 및 분류(classification) 성능에 대한 평가를 실시하고, 이를 위해 시뮬레이션 워크플로우를 개발하였습니다. 두 개의 객체 탐지 모델(YOLOv11s 및 YOLOX-S)을 원시 및 L1 데이터셋에서 훈련하여 성능을 비교합니다.

- **Technical Details**: 이 연구에서는 원시 데이터를 객체 탐지 성능에 미치는 영향을 분석하기 위해 시뮬레이션 워크플로우를 구성합니다. 원시 데이터와 L1 이미지를 결합하여 고해상도 멀티스펙트럼 이미지와 통합된 원시 제품을 생성하며, EDSR 같은 신경망을 활용하여 원시 이미지를 복원합니다. 가장 혁신적인 점은 파노라마 이미지 및 다중 스펙트럼 이미지를 조합하여 객체 탐지 모델에 적합한 데이터 세트를 만드는 것입니다.

- **Performance Highlights**: 모델은 낮은 신뢰도(threshold)에서 유사한 성능을 보이나, 원시 데이터로 훈련된 모델은 높은 신뢰도 수준에서 객체 경계 식별에 어려움을 겪습니다. 이는 AI 모델이 원시 이미지를 처리할 때 발생하는 주요한 문제를 나타내며, 향후 더 나은 경계 인식을 위해 AI 아키텍처를 개선해야 할 필요성을 제안합니다. 연구 결과는 온보드 AI의 성능 향상에 중요한 통찰력을 제공한다고 할 수 있습니다.



### Enhancing Bankruptcy Prediction of Banks through Advanced Machine Learning Techniques: An Innovative Approach and Analysis (https://arxiv.org/abs/2510.06852)
- **What's New**: 이 연구에서는 머신 러닝(machine learning) 기법을 사용하여 은행 파산 예측 모델을 개발하였습니다. 기존의 통계 모델, 예를 들어 Altman's Z-Score는 경직된 가정에 의존하기 때문에 예측 정확도가 낮다는 한계를 가지고 있습니다. 이에 따라 머신 러닝 기법을 사용하여 보다 효과적인 해결책을 모색했습니다.

- **Technical Details**: 연구에 사용된 방법에는 로지스틱 회귀(logistic regression), 랜덤 포레스트(random forest), 서포트 벡터 머신(support vector machines) 등이 포함됩니다. 이러한 머신 러닝 기법은 기존의 통계적 접근 방식보다 은행 리스크 관리를 더 효과적으로 분류하고 예측하는 데 우수함을 보여주었습니다. 연구 데이터는 터키의 44개 활성 은행과 21개 파산 은행의 연간 재무제표에서 도출되었고, 인도네시아의 43개 활성 및 43개 파산 농촌 은행의 분기 재무 보고서도 활용되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 랜덤 포레스트는 상업 은행 데이터를 90%의 정확도로 예측할 수 있었습니다. 또한 제안된 3가지 머신 러닝 방법이 농촌 은행의 파산 가능성을 정확히 예측하는 데 기여하였습니다. 이는 파산 비용을 줄이는 정책을 구현하는 데 중요한 기초 자료로 활용될 수 있습니다.



### OpenJAI-v1.0: An Open Thai Large Language Mod (https://arxiv.org/abs/2510.06847)
- **What's New**: OpenJAI-v1.0는 태국어와 영어를 위한 오픈 소스 대형 언어 모델로, Qwen3-14B 모델을 기반으로 개발되었습니다. 이 모델은 지침 이행, 긴 맥락 이해 및 도구 사용의 세 가지 핵심 사용 사례에 중점을 두어 실용적인 작업의 성능을 향상시키고자 합니다. OpenJAI-v1.0은 기존 모델보다도 성능을 높였으며, 다양한 벤치마크에서 주요 오픈 소스 태국어 모델을 능가하였습니다.

- **Technical Details**: OpenJAI-v1.0은 고품질 데이터셋을 기반으로 구성되었으며, 데이터는 지침-응답 형식으로 구성됩니다. 주요 기능에는 복잡한 지침 이행, 긴 맥락 이해, 그리고 외부 도구 및 API와 원활하게 통합할 수 있는 신뢰할 수 있는 도구 사용 등이 포함됩니다. 모델 학습은 Jasmine Technology Solution의 GPU 클러스터에서 진행되었으며, 총 462 백만 개의 토큰으로 훈련되었습니다.

- **Performance Highlights**: OpenJAI-v1.0은 다양한 벤치마크에서 성능 평가를 받았으며, 이를 통해 모델의 지침 이행과 긴 맥락 문제 해결 능력이 입증되었습니다. 특히, 태국어를 위한 벤치마크인 IFBench-TH와 다양한 다단계 상호작용을 평가하는 MT-Bench-TH에서 긍정적인 성과를 보였습니다. 이러한 평가 결과는 OpenJAI-v1.0의 실제적인 적용 가능성을 강화하는 데 기여하고 있습니다.



### SID: Multi-LLM Debate Driven by Self Signals (https://arxiv.org/abs/2510.06843)
- **What's New**: 이번 연구에서는 self signals(자기 신호)를 활용한 Self-Signals Driven Multi-LLM Debate(SID) 프레임워크를 제안합니다. 이는 모델 수준의 confidence(신뢰도)와 토큰 수준의 semantic focus(의미 중심)에 기반하여 토론 과정 및 성능을 개선하는 데 도움을 줍니다. 이 접근 방식은 에이전트들이 여유롭게 논의할 필요 없이 직접적인 경험에서 나오는 신뢰도를 활용하여 불필요한 중복 토론을 줄입니다.

- **Technical Details**: SID 프레임워크는 모델이 생성하는 과정에서 발생하는 자기 신호를 적극적으로 사용합니다. 모델 신뢰도를 판단하기 위해 생성된 답변의 확률 분포를 기반으로 하며, 이 정보를 바탕으로 조기 종료 메커니즘을 설계하여 토론이 불필요한 경우를 줄입니다. 또한 attention(주의) 메커니즘을 통해 토론 내용에서 의미 있는 부분을 강조하여 압축하고, 중요한 논점은 유지하면서 토큰 소비를 줄입니다.

- **Performance Highlights**: 예비 실험 결과, SID는 기존의 MAD 접근 방식보다 정확도에서 우수할 뿐만 아니라 최대 40%의 토큰 소비 감소를 달성하는 효과를 보였습니다. 이는 multi-agent 시스템에서 자기 신호를 활용할 때 성능과 효율성을 함께 최적화 할 수 있다는 가능성을 보여줍니다. 실험은 다양한 벤치마크 및 멀티모달 LLM에서 수행되어 SID의 효과가 입증되었습니다.



### CNN-TFT explained by SHAP with multi-head attention weights for time series forecasting (https://arxiv.org/abs/2510.06840)
- **What's New**: 이 논문에서는 다변량 시계열 예측을 향상시키기 위해 컨볼루션 신경망(CNN)과 임시 융합 트랜스포머(TFT)를 통합한 하이브리드 아키텍처를 제안합니다. CNN 모듈은 1차원 컨볼루션 레이어의 계층 구조를 적용하여 원본 입력 시퀀스에서 중요한 로컬 패턴을 추출하고, TFT는 다중 헤드 주의를 통해 단기 및 장기 종속성을 캡처합니다. 실험 결과 및 높인 설명력을 통해 제안된 모델이 기존의 심층 학습 모델들보다 우수한 성능을 보임을 입증했습니다.

- **Technical Details**: 제안된 CNN-TFT-SHAP-MHAW 모델은 원인 1D 컨볼루션 블록을 인코더에 사용하여 CNN의 효율성과 병렬 처리 능력을 활용합니다. 이 구조는 TFT 파이프라인의 나머지를 유지하면서 반복적 코어를 컨볼루션 코어로 교체합니다. 설명 가능한 인공지능(AI) 기술과 SHAP(Shapley additive explanations)를 통합하여 모델의 예측을 해석하는 데 도움을 줍니다.

- **Performance Highlights**: CNN-TFT는 수력 발전 자연 유량 시계열 데이터세트에서 평가되었으며, 평균 절대 백분율 오차(MAPE)가 최대 2.2%에 도달했습니다. 제안된 모델은 CNN 및 주의 메커니즘의 장점을 결합하여 신호 예측에 필요한 가장 중요한 특징을 보장하며, SHAP 값을 경량화하여 더 나은 의사결정을 위한 설명 가능성을 높였습니다. 이 모델은 고정밀 다변량 시계열 예측이 필요한 응용 분야에 유망합니다.



### Recurrence-Complete Frame-based Action Models (https://arxiv.org/abs/2510.06828)
- **What's New**: 최근 대형 언어 모델에 대해 주목할 만한 변화를 가져온 원리는 Attention 메커니즘입니다. 본 논문에서는 Attention 메커니즘만으로는 긴 시간 동안의 에이전트 작업에 필요한 문제를 해결할 수 없다는 주장으로 이와는 반대되는 시각을 제시합니다. 우리는 진정한 연속적 계산(true serial computation)이 필수적이고, 완전 병렬izable 아키텍처로는 필요한 계산을 일반적으로 표현할 수 없다는 점을 강조합니다.

- **Technical Details**: 논문은 '진정한 깊이(true depth)'와 '순환 완전성(recurrence completeness)'을 정의하며, 후자는 일반적인 비사소조합(즉, 비연관적) 업데이트를 실현할 수 있는지에 따라 구분됩니다. 특히, 시간 병렬 아키텍처는 제한된 깊이를 가지며, 이는 최악의 경우 긴 시퀀스 문제에 필요로하는 연산 능력을 결여하게 만듭니다. 우리 실험은 특정 비경로 적합성 문제를 해결하기 위해 작성된 직렬 평가가 중요함을 보여줍니다.

- **Performance Highlights**: 결과적으로, 긴 시퀀스 훈련은 모델의 손실을 줄이며 시계열 데이터 처리에서 유의미한 성과를 보입니다. 우리는 훈련된 시퀀스 길이에 대한 손실이 고정된 매개변수 아래에서 전력 법칙(power law)을 따르며, 더 긴 시퀀스 훈련이 짧은 시퀀스에 비해 손실 이점이 유지되는 것으로 나타났습니다. 주목해야 할 점은, 모델의 감각 상태가 향상되는 것을 보여주는 경향이 관찰되었다는 점입니다.



### FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipelin (https://arxiv.org/abs/2510.06800)
- **What's New**: 새로운 논문에서 FURINA-Builder라는 혁신적인 멀티 에이전트 협업 파이프라인을 소개합니다. 이는 사용자가 다양한 캐릭터와 시나리오에 맞춰 완전히 커스터마이즈된 RP 벤치마크를 자동으로 구축할 수 있게 돕습니다. 또한, FURINA-Bench라는 포괄적인 RP 벤치마크를 구축하여 각 캐릭터와 평가 기준에 대한 평가를 제공하는 것이 특징입니다.

- **Technical Details**: FURINA-Builder는 테스트 캐릭터와 다양한 캐릭터 간의 대화를 시뮬레이션하여 RP 작업을 평가하는 데 필요한 다양한 캐릭터 데이터베이스를 생성합니다. LLM 기반 평가 모델이 응답의 정확성을 평가하고, 다양한 평가 기준으로 최소한의 기준을 설정하여 최종 발화를 선택합니다. 이러한 메커니즘을 통해 사용자들은 특정 시나리오에 적합한 RP 벤치마크를 보다 효과적으로 구축할 수 있습니다.

- **Performance Highlights**: FURINA-Bench의 광범위한 평가를 통해 최신 LLM들이 성능을 발휘하는 다양한 결과를 얻었습니다. 특히, o3와 DeepSeek-R1이 각각 영어 및 중국어 RP 작업에서 가장 뛰어난 성능을 보였으며, 기존 캐릭터가 합성 캐릭터보다 일관되게 높은 성능을 나타냈습니다. 그러나 흥미롭게도, 추론 기능이 RP 성능을 개선하지만 동시에 RP 환상을 증가시키는 잔여트레이드오프를 발견하였고, 이는 모든 LLM의 RP 성능과 신뢰성 사이의 보다 넓은 Pareto 경계로 확장됩니다.



### Extreme Amodal Face Detection (https://arxiv.org/abs/2510.06791)
- **What's New**: 이 논문은 극단적인 비가시적 객체 탐지(extreme amodal detection)에 대한 연구를 다룹니다. 이와 같은 탐지는 입력 이미지에서 완전히 보이지 않는 객체의 2D 위치를 추론하는 것으로, 기존의 비가시적 탐지(amodal detection)와는 차별점을 가집니다. 특히 얼굴 탐지(face detection)를 하위 문제로 설정하여 안전과 프라이버시와 관련된 동기를 제공하지만, 방법론은 특정 클래스에 국한되지 않습니다.

- **Technical Details**: 기존의 접근법들은 이미지 시퀀스를 활용하여 감지되지 않은 부분을 주변 프레임에서 보완하거나 생성 모델(generative model)을 사용하여 가능한 완성을 샘플링하는 방식입니다. 하지만 본 연구는 단일 이미지(single-image) 작업을 고려하고, 이미지의 맥락적 단서를 활용해 보이지 않는 얼굴의 존재를 추론하는 보다 효율적이고 샘플-free한 접근법을 제안합니다. 열지도(heatmap) 기반의 극단적 비가시적 객체 탐지기를 설계하여 이미지에서 최소한의 정보를 가지고 많은 정보를 예측하는 문제를 해결합니다.

- **Performance Highlights**: 본 방법은 새로운 작업에 대해 강력한 결과를 도출하며, 기존의 비효율적인 생성 접근 방식보다 성능이 우수한 것으로 확인되었습니다. 선택적 coarse-to-fine 디코더(selective coarse-to-fine decoder)를 통해 더욱 효율적으로 객체를 탐지할 수 있습니다. 이 연구는 단일 이미지에서 자동으로 얼굴을 감지할 수 있는 가능성을 제시하며, 향후 다양한 응용 분야에 적용될 수 있습니다.



### Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness (https://arxiv.org/abs/2510.06780)
- **What's New**: 본 연구에서는 miniGPTKB라는 개념을 도입하여 개별 도메인에 특화된 LLM 지식의 트래킹을 수행합니다. 또한, 기존 GPTKB 접근 방식의 종료 가능성을 입증하고, 재현성과 강건성에 대한 실험 결과를 보고합니다. 이와 함께 특정 LLM의 사실적 지식의 핵심에 대한 안정적인 시각을 제공할 수 있음을 주장합니다.

- **Technical Details**: 이 연구는 LLM 지식 수집 과정에서의 종료 가능성, 재현성 및 강건성을 평가하기 위해 miniGPTKB를 사용하는 방법론을 채택합니다. 세 가지 예시 도메인으로 고대 바빌론(역사), The Big Bang Theory(오락), 그리고 DAX 40(금융)이 포함되어 있으며, 각 도메인에 대해 표준화된 프로브를 사용하여 분석을 시행합니다.

- **Performance Highlights**: 결과는 높은 종료율을 보여주지만 모델에 따라 다름을 나타냅니다. 재현성 면에서는 혼합된 신호가 관찰되었으며, 정량적 성과에서는 높은 유사성이 있었으나, 어휘적 유사성은 낮고, 의미적 유사성은 중간 수준이었습니다. 여러 실험에서 miniGPTKB의 결과에 대한 강건성은 시드와 온도에 대해 높았으나, 언어와 모델에 대해서는 낮은 경향을 보였습니다.



### Modeling COVID-19 Dynamics in German States Using Physics-Informed Neural Networks (https://arxiv.org/abs/2510.06776)
Comments:
          19 pages, 7 figures, 2 tables

- **What's New**: COVID-19 팬데믹은 질병 동역학을 이해하기 위한 정량적 모델링과 분석의 필요성을 강조했습니다. 이 연구에서는 로베르트 코흐 연구소(RKI)의 감염 데이터를 사용하여 SIR (Susceptible-Infectious-Recovered) 모델의 역문제를 해결하기 위해 Physics-Informed Neural Networks (PINNs)를 활용하였습니다. 독일 16개 연방주에 대한 COVID-19 동역학의 세밀한 시공간 분석을 제공하고, 각각의 주에서 전파 및 회복 매개변수를 추정하고 있습니다.

- **Technical Details**: 비선형적 매개변수의 추정은 일반적으로 관측 데이터를 기반으로 합니다. PINNs는 미분 방정식을 네트워크 훈련에 통합하여 이러한 한계를 극복하고, 실제 세계의 관찰 데이터를 통해 잠재적 매개변수를 효과적으로 추정할 수 있습니다. 이 연구에서는 1,200일 동안의 COVID-19 데이터에 대한 분석을 통해 연방주별 전파율 및 회복율을 파악하였으며, 이를 통해 시간에 따라 변화하는 재생산 수치(ℛ_t)도 추정하였습니다.

- **Performance Highlights**: 연구 결과, 각 연방주 간 전파 행동의 강한 차이를 보여주었고, 예방접종 수치 및 주요 팬데믹 단계와의 상관관계를 밝혀냈습니다. 이는 지역적인 개입이 전파 동역학에 측정 가능한 영향을 미쳤음을 시사합니다. PINNs를 활용한 지역적이고 장기적인 역학 모델링의 유용성을 확인하며, COVID-19의 확산이 시간이 지남에 따라 어떻게 변화했는지에 대한 깊은 통찰을 제공합니다.



### Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities (https://arxiv.org/abs/2510.06743)
Comments:
          The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 기반으로 한 OCR(광학 문자 인식)에 대한 평가 프레임워크의 필요성을 제시하고 있습니다. 기존의 평가 지표들이 역사적 문서에 대한 특정 오류와 시간적 편향을 포착하는 데 실패하는 점을 강조하며, 새로운 메트릭인 HCPR(역사적 문자 보존 비율)과 AIR(구식 삽입 비율)을 도입하였습니다. 이 방법론은 디지털 인문학에 종사하는 연구자들에게 모델 선택과 품질 평가를 위한 가이드를 제공합니다.

- **Technical Details**: 연구에서는 18세기 러시아 문서를 대상으로 한 LLM 기반 역사적 OCR 평가 프레임워크를 제시합니다. 수집된 데이터는 러시아 시민 서체로 인쇄된 428개의 독특한 18세기 도서에서 1,029페이지를 포함하고 있으며, 기존 OCR 시스템들이 어려움을 겪는 특유의 오탈자 및 고전 문법 형태를 포함하고 있습니다. 또한, 각 모델의 출력 변동성을 감안한 종합적인 안정성 테스트를 수행했습니다.

- **Performance Highlights**: 실험 결과, Gemini 및 Qwen 모델이 기존 OCR 시스템보다 뛰어난 성능을 보였으나, '과거화(over-historicization)'라는 요소로 인해 올바르지 않은 역사적 시점에서 고어 문자 삽입 현상이 발생했습니다. LLM들이 특정 역사적 문서에 대해 예상치 못한 시간적 편향을 보임으로써, 전통적인 평가 방법들이 이러한 문제를 탐지할 수 없는 한계가 있음을 확인하였습니다. 결과적으로, LLM을 활용한 역사적 문서 변환의 정확성을 높이기 위해서는 새로운 평가 방안이 필요하다는 결론을 내고 있습니다.



### Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization (https://arxiv.org/abs/2510.06732)
Comments:
          10 pages, 3 figures

- **What's New**: 이번 논문은 정보 검색에서 rerankers(재정렬기)로 활용되는 대형 언어 모델(LLMs)의 취약점을 드러내고 이를 조작할 수 있는 새로운 방법인 Rank Anything First (RAF)를 소개합니다. RAF는 타겟 아이템의 순위를 높이기 위해 간결한 텍스트 변형을 생성하는 두 단계의 토큰 최적화 방식으로 구성되어 있습니다. 첫 번째 단계에서는 그리디 좌표 경량화(Greedy Coordinate Gradient)를 사용하여 현재 위치에서의 후보 토큰을 선별하고, 두 번째 단계에서는 후보들을 평가하여 자연스러운 언어를 유지하면서 효과적으로 기존 방법들보다 높은 순위 조작성을 달성합니다.

- **Technical Details**: RAF는 토큰 단위 최적화를 통해 순위 조작 프롬프트를 생성하며, 두 가지 목표인 순위 효과성(maximizing ranking effectiveness)과 언어 자연스러움(preserving linguistic naturalness)을 동시에 고려합니다. 특정 수식과 조건부 확률을 통해 LLM의 재정렬 과정을 설명하며, 공격자가 특정 아이템의 설명에 자연스러운 텍스트 시퀀스를 삽입하여 아이템의 순위를 향상시킬 수 있도록 합니다. 이러한 과정은 제품의 브랜드, 가격, 간단한 설명을 포함한 제품 세트를 기반으로 하여 수행됩니다.

- **Performance Highlights**: RAF는 다양한 LLM 모델에서 실험을 통해 자연스러운 언어를 사용하여 목표 아이템의 순위를 유의미하게 증가시키는 데 성공하였습니다. 이는 기존의 방법들보다 더욱 강력하고 안정적인 순위 조작을 가능하게 하며, 최적화된 프롬프트는 여러 모델 간에 성공적으로 전이됩니다. 이 연구는 LLM 기반 reranking의 보안 위험성을 강조하며, 현대 정보 검색 시스템의 신뢰성과 강인성에 대해 새로운 도전 과제를 제시합니다.



### Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Managemen (https://arxiv.org/abs/2510.06727)
- **What's New**: 본 논문에서는 긴 범위의 다중 턴(tool use) 도구 사용을 위한 대형 언어 모델(LLM) 에이전트의 강화 학습(RL) 미세 조정을 연구하였습니다. 기존의 RL 파이프라인은 지침 수행 저하, 과도한 롤아웃 비용, 그리고 엄격한 컨텍스트 한계 등의 문제를 겪을 수 있습니다. 이를 해결하기 위해, 우리는 요약 기반의 컨텍스트 관리 방법을 교육에 도입했습니다.

- **Technical Details**: 이 방법은 LLM이 생성한 요약을 이용해 도구 사용 이력을 주기적으로 압축함으로써 작업과 관련된 정보를 유지하며 компакт한 컨텍스트를 유지할 수 있도록 지원합니다. 이 형식에 기반하여, 우리는 도구 사용 행동과 요약 전략을 끝에서 끝으로 최적화할 수 있는 정책 기울기(Policy Gradient) 표현을 도출했습니다. 이를 통해 	exttt{SUPO}라는 LLM RL 알고리즘을 구현하여 고정된 컨텍스트 한계를 넘어서는 긴 범위의 교육이 가능해졌습니다.

- **Performance Highlights**: 실험을 통해 상호작용하는 기능 호출 및 검색 작업에서 	exttt{SUPO}가 성공률을 크게 향상시키는 동시에 기준과 비교하여 동일하거나 심지어 더 짧은 작업 컨텍스트 길이를 유지함을 입증했습니다. 복잡한 검색 작업에 대해서는, 	exttt{SUPO}가 학습 시간보다 테스트 시간에서 최대 요약 라운드를 더욱 확장하여 평가 성과를 개선할 수 있음을 보여주었습니다. 이러한 결과는 요약 기반의 컨텍스트 관리가 고정된 컨텍스트 길이 한계를 넘어 RL 에이전트를 훈련시키기 위한 원칙적이고 확장 가능한 접근 방식임을 입증합니다.



### LLM Company Policies and Policy Implications in Software Organizations (https://arxiv.org/abs/2510.06718)
Comments:
          Accepted at IEEE Software Special Issue on AIware in the Foundation Models Era

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM) 챗봇을 소프트웨어 조직에 통합하는 데 있어 명확한 정책의 필요성을 강조합니다. 11개 회사의 정책 수립 과정과 그 배경을 분석하여 관리자들이 챗봇을 안전하게 개발 워크플로에 통합할 수 있도록 돕고자 하였습니다. 연구의 주요 목적은 LLM의 도입으로 인한 위험과 기회를 파악하는 것입니다.

- **Technical Details**: LLM 정책은 회사의 필요(예: 민감한 데이터 유출 위험 최소화)와 챗봇의 기회(예: 개발 팀 생산성 향상)에 의해 형성됩니다. 정책은 정적인 문서가 아닌 작업 방식에 영향을 미치며, 조직 내 변경을 유발하여 소프트웨어 공학에서 챗봇 사용의 새로운 필요와 기회를 열어줍니다. 규정 준수와 산업 기준은 정책 수립의 주요 원동력으로 언급되었습니다.

- **Performance Highlights**: 정책 교육은 기존의 보안 교육을 넘어, 직원들이 편향이나 잘못된 정보를 인식할 수 있도록 돕고, 프롬프트 엔지니어링과 같은 기술을 개발하는 데 중점을 두어야 합니다. 각 기업은 LLM 사용에 대한 정책을 다르게 적용하고 있으며, 일부 회사는 비공식적인 접근 방식을 채택하여 직원들이 정보를 쉽게 공유하도록 하고 있습니다. 그러나 조직 문화와 컨텍스트에 맞춘 정책 수립 및 집행이 필수적입니다.



### Dual Goal Representations (https://arxiv.org/abs/2510.06714)
- **What's New**: 이번 연구에서는 목표 지향 강화 학습(goal-conditioned reinforcement learning, GCRL)을 위한 이중 목표 표현(dual goal representations) 개념을 도입합니다. 이중 목표 표현은 모든 다른 상태에서의 시간적 거리(temporal distances) 집합을 통해 상태를 특성화합니다. 이러한 표현 방식은 기존의 상태 표현(original state representation)에 영향을 받지 않는 특징이 있습니다.

- **Technical Details**: 이 이중 목표 표현은 환경의 내재적 동적(intrinsic dynamics)만을 의존하며, 외부 잡음(exogenous noise)을 필터링(filer out)할 수 있는 충분한 정보를 포함합니다. 이를 기반으로, 기존의 GCRL 알고리즘과 결합할 수 있는 실용적인 목표 표현 학습 방법을 개발하였습니다. 또한, 이 방법은 최적 목표 도달 정책(optimal goal-reaching policy)을 복원하는 데 필요한 정보를 담고 있습니다.

- **Performance Highlights**: OGBench 작업 세트(task suite)에서 다양한 실험을 수행하여 이중 목표 표현이 20개의 상태(state) 및 픽셀(pixel) 기반 작업에서 오프라인(goal-reaching) 성능을 일관되게 개선하는 것을 경험적으로 보여주었습니다. 이 결과는 이중 목표 표현의 유용성과 잠재력을 입증합니다.



### AISysRev -- LLM-based Tool for Title-abstract Screening (https://arxiv.org/abs/2510.06708)
Comments:
          4 pages

- **What's New**: 이 논문에서는 소프트웨어 공학에서 체계적 검토(Systematic Review)를 수행하는 AiSysRev라는 LLM(대형 언어 모델) 기반의 스크리닝 도구를 소개합니다. 이 도구는 Docker 컨테이너에서 작동하며, CSV 파일을 입력받아 연구자가 포함 및 제외 기준을 지정할 수 있도록 설계되었습니다. AiSysRev는 OpenRouter를 사용하여 여러 LLM을 통해 스크리닝을 수행하고, 영문 제목과 초록 기반의 자동 스크리닝을 지원합니다.

- **Technical Details**: AiSysRev는 제로샷(zero-shot) 및 퓨샷(few-shot) 스크리닝 방식을 지원합니다. 사용자는 특정 기준을 정의하고 OpenRouter API 키를 통해 다양한 LLM에 접근할 수 있게 되며, 이 과정에서 수천 개의 문헌을 효과적으로 평가할 수 있습니다. 또한, 인적 개입이 필요한 경계 경우(Boundary cases)에 대해서도 수동 스크리닝을 통해 지원합니다.

- **Performance Highlights**: 연구 결과, AiSysRev를 사용한 스크리닝에서 문헌은 크게 네 가지 범주로 분류되며, 각각 '쉬운 포함(Easy Includes)', '쉬운 제외(Easy Excludes)', '경계 포함(Boundary Includes)', '경계 제외(Boundary Excludes)'로 나타납니다. LLM의 결과는 인간의 검토를 보조하는 역할을 하여 대량의 과학 문헌을 평가하는 어려움을 크게 줄이는 데 기여했습니다. 이 연구는 LLM의 한계와 정확도 문제를 명확히 하며, 향후 체계적 검토 절차의 효율성을 높일 수 있는 방안을 제시합니다.



### Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks (https://arxiv.org/abs/2510.06695)
- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)에 대한 관심이 증가함에 따라, 프롬프트 엔지니어링(prompt engineering)이 수동 설계에서 모델 기반 최적화로 발전하였습니다. 본 논문에서는 기계 번역(machine translation)과 같은 특정 작업에 적합한 새로운 프롬프트 최적화 방법을 소개합니다. 제안된 방법은 작은 매개변수 모델을 활용하여 백 트랜슬레이션(back-translation) 전략으로 학습하여, 단일 작업 최적화를 위한 훈련 비용을 대폭 줄이는 동시에 높은 성과를 제공합니다.

- **Technical Details**: LLM의 프롬프트는 일반적으로 instruction와 input의 두 가지 구성 요소로 이루어져 있습니다. 본 논문에서는 입력 최적화를 위한 Rewriting Original Inputs (ROI) 전략을 제안합니다. 이 방법은 LLM 또는 소형 매개변수 모델을 활용하여 원본 입력을 재구성하고, 언어 모델에 더 잘 맞도록 조정합니다. 특히, 입력의 질을 평가하기 위한 필터링 모듈도 도입되어 의미 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, ROI 모듈은 애매한 데이터를 보다 명확한 입력 프롬프트로 변환하는 데 효과적임을 보여줍니다. NLU 및 NLG 작업에 대한 성능 향상을 입증하였으며, 기존의 프롬프트 최적화 방법은 입력 구성 요소가 중요한 작업에 한계가 있다는 점을 보여주었습니다. ROI 방법은 다양한 LLM에 널리 적용 가능하며, 원본 입력에 비해 일관되고 주목할 만한 성과 향상을 이룹니다.



### Semantic Segmentation Algorithm Based on Light Field and LiDAR Fusion (https://arxiv.org/abs/2510.06687)
- **What's New**: 이 논문에서는 조명이 차단된 복잡한 환경에서의 자율 주행을 위한 세미틱 세그멘테이션의 과제를 해결하기 위해 라이트 필드 데이터와 포인트 클라우드 데이터를 통합한 최초의 다중 모달 세미틱 세그멘테이션 데이터셋인 TrafficScene을 제안합니다. 이 데이터셋은 모든 라이트 필드 뷰포인트에 대한 의미론적 주석을 제공하여 차단된 또는 작은 객체에 대한 정보를 효과적으로 보완합니다. 또한, 새로운 세그멘테이션 알고리즘인 Multimodal Light Field Point Cloud Fusion Segmentation Method (Mlpfseg)을 소개하여 라이트 필드 이미지와 포인트 클라우드의 동시 세그멘테이션을 가능하게 합니다.

- **Technical Details**: Mlpfseg는 기능 보완(feature completion) 및 깊이 인지(depth perception) 모듈을 포함하여 이미지와 포인트 클라우드를 동시에 세그멘테이션합니다. 기능 보완 모듈은 포인트 클라우드와 이미지 픽셀 간의 밀도 불일치를 해결하기 위해 차별적 재구성을 수행하며, 깊이 인지 모듈은 주의 점수(attention scores)를 보강하여 차단된 객체에 대한 인식을 향상시킵니다. 이러한 접근 방식을 통해 이 방법은 이미지 또는 포인트 클라우드 각각만을 사용하는 세그멘테이션보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 단일 이미지 세미틱 세그멘테이션 방법에 비해 1.71의 Mean Intersection over Union (mIoU) 개선을 달성하였고, 포인트 클라우드 전용 세그멘테이션에 비해서는 2.38의 mIoU 향상을 보여줍니다. 이는 다중 모달 환경에서의 차단된 객체 인식 능력을 크게 향상시킵니다. 결론적으로, Mlpfseg는 라이트 필드와 포인트 클라우드 간의 효과적인 통합을 통해 입증된 성능을 보이며, 자율 주행 및 다양한 컴퓨터 비전 응용 프로그램에서 중요한 기여를 제공합니다.



### Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback (https://arxiv.org/abs/2510.06677)
Comments:
          Accepted at EMNLP 2025 Industry Track

- **What's New**: 본 논문에서는 고객 지원 에이전트를 위한 점진적 요약 시스템을 소개합니다. 이 시스템은 대화 중 요약 노트를 생성해야 할 최적의 순간을 지능적으로 판단하여 에이전트의 맥락 전환 노력과 중복 리뷰를 줄입니다. Mixtral-8x7B 모델과 DeBERTa 기반 분류기를 결합하여 실시간으로 효과적인 요약 생성을 제공합니다.

- **Technical Details**: 시스템은 여러 채널의 대화를 통합하여 지속적으로 요약된 글머리 기사를 생성합니다. 자동화된 요약 모델은 중요 정보가 감지될 때만 요약을 제안하고, 비실질적인 내용을 걸러내는 분류기를 통해 자동으로 최적화됩니다. 에이전트의 실시간 수정 기능은 지속적으로 모델 학습을 보강합니다.

- **Performance Highlights**: 생산 환경에 배포된 이 시스템은 평균 처리 시간을 3% 단축시키며, 복잡한 사례에서는 최대 9%까지 감소시켰습니다. 또한, 설문 조사에서 80% 이상의 에이전트 만족도를 기록하며, 점진적 요약이 에이전트의 생산성을 향상시키는 효과를 입증합니다.



### Heptapod: Language Modeling on Visual Signals (https://arxiv.org/abs/2510.06673)
- **What's New**: Heptapod은 언어 모델링의 기초 원칙을 준수하는 이미지 자동 회귀 모델입니다. 이 모델은 causal attention을 활용하고 CFG에 대한 의존성을 제거하며, 의미론적 토크나이저의 추세를 피합니다. 핵심 혁신은 모든 2D 공간 그리드에 대한 분포 예측(next 2D distribution prediction)이며, 이는 생성 훈련을 통해 포괄적인 이미지 의미를 파악하도록 도와줍니다.

- **Technical Details**: Heptapod은 일관된 causal Transformer 구조를 사용하여 시각적 토큰을 생성하고, 기존의 이미지 자동 회귀 모델과는 달리 각 번째 공간 위치에서의 토큰 분포를 병렬로 예측하기 위해 훈련됩니다. 이러한 접근 방식은 모델이 복잡한 공간 종속성과 전체 이미지를 이해하도록 유도합니다. 또한, 자가 지도 학습 관점에서 autoregressive modeling과 Masked Autoencoding(MAE)의 결합된 목표를 제공합니다.

- **Performance Highlights**: 이미지 생성 기준인 ImageNet에서 Heptapod은 FID가 2.70으로, 이전의 자동 회귀 접근법을 월등히 초월하는 성능을 보여줍니다. 이는 기존의 시각적 생성을 위한 모델들이 구조적 개선 없이 외부 개입(CFG)에 의존해야 하는 한계를 극복할 수 있다는 것을 시사합니다. Heptapod은 언어 모델링 원칙이 시각 신호에서도 잘 적용될 것이라는 점에서 새로운 가능성을 엿보게 합니다.



### Automated Neural Architecture Design for Industrial Defect Detection (https://arxiv.org/abs/2510.06669)
- **What's New**: 이 논문에서는 표면 결함 감지(Surface Defect Detection; SDD)를 위한 오토화된 뉴럴 아키텍처 디자인 프레임워크인 AutoNAD를 제안합니다. 이 시스템은 컨볼루션(convolution), 트랜스포머(transformer), 다층 퍼셉트론(MLP)을 조합해 하이브리드 아키텍처를 자동으로 설계하여, 인트라클래스 차이(intraclass difference)와 인터클래스 유사성(interclass similarity) 문제를 효과적으로 해결합니다. AutoNAD는 다양한 결함 유형과 산업 적용 환경에 맞춰 적응적인 네트워크 설계를 가능하게 합니다.

- **Technical Details**: AutoNAD는 교차 가중치 공유(cross weight sharing) 전략을 통해 다양한 연산자 유형 내에서 및 서로 간의 효율적인 가중치 공유를 촉진합니다. 이와 함께 다단계 특징 집합 모듈(Multi-Level Feature Aggregation Module; MFAM)을 통해 멀티 스케일(feature) 학습을 강화하며, 다이렉트 에이시클릭 그래프(directed acyclic graph) 구조를 이용하여 최적의 융합 경로를 동적으로 선택합니다. 또한, 런타임 통계 기반의 레이턴시 인식(latency-aware) 프라이어를 도입해 SDD에서 최적의 아키텍처를 탐색합니다.

- **Performance Highlights**: AutoNAD는 세 가지 산업 결함 데이터셋에서 성능을 검증하였으며, 실제 자동 감지 시스템에 통합되어 높은 검출 정확도와 효율성을 달성했습니다. 이 시스템은 생산 제약 조건에서 실제로 작동할 수 있는 가능성을 입증하며, 빠른 수렴성과 향상된 서브넷 성능을 통해 산업 환경에서의 적용 가능성을 높입니다. 결과적으로 AutoNAD는 기존 수작업 아키텍처 설계 과정의 부담을 줄이고, 더 나가 많은 산업 안정성을 제공할 수 있는 시스템으로 자리 잡고 있습니다.



### Delay Independent Safe Control with Neural Networks: Positive Lur'e Certificates for Risk Aware Autonomy (https://arxiv.org/abs/2510.06661)
Comments:
          Submitted to 2026 American Control Conference (ACC), New Orleans, LA

- **What's New**: 본 논문은 자율 학습 제어 시스템을 위한 위험 분별 안전 인증 방법을 제안합니다. 시간 지연(state/input delays)과 구간 행렬 불확실성(interval matrix uncertainty)이라는 두 가지 현실적인 위험에 초점을 맞추고, 신경망(Neural Network) 제어기를 로컬 섹터 경계(local sector bounds)로 모델링합니다. 이를 통해 지연에 독립적이고 지연을 고려한 보증을 제공하는 선형 인증서를 유도합니다.

- **Technical Details**: 우리는 피드포워드 신경망(FFNN)에 대한 새로운 섹터 경계를 개발했으며, 이는 기존의 요소별 경계(CROWN, IBP)와 달리 네트워크 수준의 섹터 설명을 제공합니다. 또한, 메츠러 행렬(Metzler matrix)과 양의 시스템(properties of positive systems) 특성을 활용하여 시간 지연과 구간 불확실성 하의 신경망 피드백 루프를 검증하는 확장 가능하고 지연 독립적 인증서를 도입합니다.

- **Performance Highlights**: 우리의 방식은 기존의 SDP 기반 접근 방식이 실패하는 영역에서도 인증을 제공하며, 지연 지연에 관한 복잡한 계산 요구 없이 빠른 성능을 보입니다. 실험 결과, 우리의 긍정성 기반 테스트는 기존의 SDP 기반 IQC 테스트보다 수십 배 빠르며, 실시간 온라인 검증 가능성을 제시합니다.



### Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions (https://arxiv.org/abs/2510.06649)
Comments:
          15 pages, 5 figures

- **What's New**: 이 논문은 Forward-Forward (FF) 알고리즘을 기반으로 한 새로운 가치 추정 방법인 Action-conditioned Root mean squared Q-Functions (ARQ)를 소개합니다. ARQ는 생물학적으로 타당한 신경망 학습 방법과 강화 학습 (Reinforcement Learning, RL)을 통합하여 개발되었습니다. 이 연구는 FF의 장점을 살리면서 RL 설정에서의 적용 가능성을 탐색합니다.

- **Technical Details**: ARQ는 두 가지 주요 요소로 구성되며, 하나는 임의 크기의 벡터에서 가치 예측을 추출하는 goodness function입니다. 두 번째 요소는 모델 입력에 동작 후보를 삽입하는 action conditioning이며, 이를 통해 네트워크는 각 상태-동작 쌍에 구체적인 표현을 생성할 수 있습니다. ARQ는 복잡한 입력을 모델링하는 데 있어 유연성을 높이면서도 backpropagation-free 특성을 유지합니다.

- **Performance Highlights**: MinAtar 및 DeepMind Control Suite 벤치마크에서 ARQ 방법은 기존의 local RL 방법과 일반적인 backprop 기반 방법보다 우수한 성과를 보였습니다. 특히, ARQ는 강화 학습 환경에서 강력한 의사 결정 능력을 보여줍니다. 이 연구는 생물학적으로 타당한 학습 방법과 강화 학습의 교차점에서 더 많은 탐색을 장려하고자 합니다.



### The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators (https://arxiv.org/abs/2510.06646)
- **What's New**: 이번 연구는 기계 학습된 연산자(Machine-Learned Operators, MLOs)가 높은 해상도의 데이터에서 추론(inference)을 수행할 수 있는지, 즉 '제로샷 초해상도(zero-shot super-resolution)'을 가능하게 하는지를 평가합니다. 연구진은 MLOs가 여러 해상도에서의 추론을 수행하는 데 있어 두 가지 주요 행동, 즉 주파수 정보의 외삽(extrapolation)과 해상도 간의 보간(interpolation)의 실패를 관찰했습니다. 이 결과를 바탕으로, MLOs가 훈련된 해상도 이외의 해상도에서 정확한 추론을 하지 못한다고 결론지었습니다.

- **Technical Details**: 연구의 주요 초점은 MLOs가 다중 해상도 추론(multi-resolution inference)을 수행하는데 있어서 불완전함을 드러내는 것입니다. 특히, MLOs는 새로운 주파수에 대한 추론을 실패하며, 훈련 당시의 해상도와 다른 해상도에서는 오류를 범할 수 있습니다. 이를 해결하기 위해 연구진은 다중 해상도 훈련(multi-resolution training) 프로토콜을 제안하였으며, 이는 저해상도와 고해상도의 데이터 세트를 동시에 활용하는 방식입니다.

- **Performance Highlights**: 제안된 다중 해상도 훈련 접근법은 훈련 비용을 크게 증가시키지 않으면서도 MLOs의 전반적인 성능을 향상시킬 수 있습니다. 연구 결과, MLOs는 훈련 데이터 해상도에 의존적으로 동작하며, 적절히 다양한 해상도를 아우르는 훈련을 통해 다중 해상도 일반화(multi-resolution generalization)가 가능하다는 것을 보여주었습니다. 이 연구는 MLOs의 설계 및 적용에 대한 중요한 통찰을 제공하며, 실시간 응용에서의 잠재적 활용을 제시합니다.



### Distilling Lightweight Language Models for C/C++ Vulnerabilities (https://arxiv.org/abs/2510.06645)
Comments:
          25 pages, 10 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 활용한 코드 취약점 탐지의 효율성을 개선하기 위해 FineSec라는 새로운 프레임워크를 제안합니다. FineSec는 지식 증류(knowledge distillation)를 통해 대형 teacher 모델의 지식을 소형 student 모델로 전이하여, C/C++ 코드에서 높은 정확도로 취약점을 탐지할 수 있게 합니다. 이 프레임워크는 데이터 준비, 훈련, 평가 및 지속적 학습을 통합하여 단일 작업 흐름을 제시합니다.

- **Technical Details**: FineSec 프레임워크는 QLoRA를 통해 파라미터 효율적인 미세 조정을 수행하며, 이를 통해 경량 LLM 기반 모델을 개발하여 C/C++ 언어 소스 코드의 취약점을 탐지합니다. 이 과정에서 모든 실험 결과와 데이터 세트를 공개하여 재현 가능성을 높이고 있습니다. 논문은 30개 이상의 CWE(공통 약점 열거)의 범주에 걸쳐 7개의 대표적인 LLM을 비교 평가합니다.

- **Performance Highlights**: FineSec의 Extensive 평가 결과는 복잡한 취약점과 논리적 결함을 탐지하는 데 있어 기존 모델들과 더 큰 LLM보다 우수한 성능을 보였음을 보여줍니다. 또한, 기존 CWE 데이터 세트에 레이블이 없는 새로운 취약점 패턴을 발견하는 데 성공하여, 실질적인 보안 해결책으로 자리 잡을 가능성을 시사합니다. FineSec는 메모리 관리, 입력 검증, 정보 보안, 권한 제어 및 동시 문제와 같은 예측 오류 유형을 카테고리화하여, LLM 기반 취약점 탐지 시스템의 개선 방향도 제시합니다.



### StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering (https://arxiv.org/abs/2510.06638)
- **What's New**: 이번 연구는 Knowledge-based Visual Question Answering (KVQA) 의 새로운 변형인 IK-KVQA를 다루고 있습니다. 이 모델은 복합 멀티모달 대형 언어 모델 (MLLM)을 지식 소스로 사용하며, 외부 검색 없이 사실 지식을 바탕으로 질문에 답변합니다. 기존 MLLM들의 한계를 극복하기 위해 구조적 추론 추적 (Structured Reasoning Traces)이라는 새로운 방법론을 도입합니다.

- **Technical Details**: StaR-KVQA는 이중 기호 관계 경로 (dual symbolic relation paths)와 경로 기반 자연어 설명 (path-grounded natural-language explanations)으로 구성된 구조적 추적을 감독합니다. 이를 통해 추론 과정의 투명성과 검증 가능성을 높입니다. 이 방법은 외부 검색자나 지식 기반을 사용하지 않고, 오프라인에서 추적을 구축하며 단일 자기 회귀 (autoregressive) 과정을 통해 추론을 수행합니다.

- **Performance Highlights**: 연구 결과, StaR-KVQA는 OK-VQA 벤치마크에서 가장 강력한 기준선 대비 최대 11.3% 더 높은 답변 정확도를 달성했습니다. 또한, 다양한 도메인에 대한 강력한 일반화 성능을 보여주어, 정확성과 해석 가능성을 모두 향상시켰습니다.



### Control-Augmented Autoregressive Diffusion for Data Assimilation (https://arxiv.org/abs/2510.06637)
- **What's New**: 이 논문은 Auto-Regressive Diffusion Models(ARDM)에서 가이드를 개선하는 새로운 방법론을 제안합니다. 저자들은 경량화된 컨트롤러 네트워크를 추가하여 사전 학습된 ARDM을 보강하고, 이를 통해 예측 오류를 효과적으로 줄이는 방법을 제시합니다. 이 접근법은 비선형 역학을 모델링해야 하는 데이터 동화(data assimilation, DA) 문제에서 특히 유용하다는 점에서 차별화됩니다.

- **Technical Details**: 제안된 방법은 사전 학습된 ARDM의 생선 동역학(generative dynamics)에 학습된 제어 메커니즘(control mechanism)을 통합합니다. 방법론은 입력에 따라 diffusion 과정을 안내하도록 설계되었으며, 이는 전통적인 데이터 동화 기법에 비해 높은 효율성을 자랑합니다. 저자들은 이러한 구조가 다양한 관찰 조건 하에서도 안정성과 정확성을 높인다고 주장합니다.

- **Performance Highlights**: 저자들은 제안된 방법이 두 개의 고전적인 PDE 데이터셋과 여섯 가지 관찰 조건에서 네 가지 최신 기법(state-of-the-art)보다 우수한 성능을 보여주었다고 주장합니다. 이 연구는 계산 집약적인 최적화 절차를 피함으로써 실시간 예측을 보다 정확하고 안정적으로 할 수 있도록 합니다. 또한, 저자들은 코드를 공개하여 연구의 재현성을 높일 계획을 밝히고 있습니다.



### AI-Driven Forecasting and Monitoring of Urban Water System (https://arxiv.org/abs/2510.06631)
- **What's New**: 이 논문은 도시의 하수도 및 오수 파이프라인에서 누수 탐지를 위한 통합 AI 및 원격 센서 프레임워크를 제안합니다._sparse sensor_를 사용해 실시간 흐름과 깊이 데이터를 수집하고, HydroNet이라는 특별한 모델을 활용하여 보다 정밀한 예측을 제공합니다. 이 시스템은 현실 세계의 캠퍼스 하수 네트워크 데이터셋을 통해 효과적인 예측 성능을 달성했습니다.

- **Technical Details**: 본 시스템은 3단계로 구성되어 있습니다: 첫째, _sparse remote sensors_를 통해 주요 네트워크 위치에서 실시간 흐름 및 깊이 데이터를 수집합니다; 둘째, 이 데이터를 _graph-structured_ 형식으로 결합하여 HydroNet을 통해 시공간 모델링을 수행합니다; 셋째, 모델이 정상적인 유압 패턴을 정확히 예측하도록 학습합니다. HydroNet은 방향 그래프를 기반으로 하며, 파이프라인 속성을 통합하여 메시지를 전달합니다.

- **Performance Highlights**: 평가 결과, HydroNet은 모든 메트릭에서 기존 모델들을 초과 달성하여 높은 정확성을 입증했습니다. 수집된 데이터는 강력한 예측 유틸리티를 제공하여 이상 탐지 및 예측을 지원합니다. 이 접근법은 도시의 지하 수자원 관리 개선에 강력한 기초를 제공합니다.



### Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation (https://arxiv.org/abs/2510.06605)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 저작권 보호와 관련된 문제를 다루고 있습니다. 특히 LLM의 고유한 서명을 추출하여 출처 모델과 비교하는 LLM fingerprinting 기법이 제안되었습니다. 기존의 블랙박스 방법들이 효과적이지 못한 이유를 분석한 후, 새로운 방법인 ZeroPrint를 제안하여 성능을 크게 향상시켰습니다.

- **Technical Details**: ZeroPrint는 Fisher Information Theory를 바탕으로, 모델의 입력에 대한 기울기(gradient)가 출력(output)보다 더 많은 정보를 포함하고 있다고 주장합니다. 이 기법은 기존의 블랙박스 방법에서 접근할 수 없는 기울기를 근사화하기 위해 zeroth-order estimation을 사용합니다. 텍스트 도메인에서 이 기법을 적용하기 위해, 의미를 보존하는 단어 교체를 통해 입력의 변형을 생성하여 모델의 Jacobian matrix를 추정합니다.

- **Performance Highlights**: ZeroPrint는 LeaFBench라는 벤치마크에서 기존의 SOTA 블랙박스 fingerprinting 방법들보다 일관되게 우수한 성능을 기록하였습니다. 다양한 지표에서 ZeroPrint의 효과성과 신뢰성이 입증되었으며, LLM 저작권 감사 분야에서 새로운 기준을 세웠습니다.



### SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation (https://arxiv.org/abs/2510.06596)
- **What's New**: 본 논문은 Synthetic Dataset Quality Metric (SDQM)을 소개하여, 객체 탐지(object detection) 작업에서 생성된 데이터의 품질을 평가합니다. SDQM은 객체 탐지 모델이 훈련되는 동안 데이터의 효과성을 예측할 수 있는 효율적인 메트릭으로, 최신 AI 기법을 활용하여 실제 데이터의 특성을 잘 반영하도록 설계되었습니다. 이 메트릭은 비용이 많이 드는 반복 훈련의 필요성을 줄여주고 자원의 제약이 있는 상황에서 중요한 도전 과제를 해결하는 데 기여합니다.

- **Technical Details**: SDQM은 객체 탐지 데이터셋의 유용성을 평가하기 위해 여러 독립적인 구성 요소들로 구성됩니다. 이 메트릭은 픽셀 공간, 공간 공간, 주파수 공간, 특징 공간 도메인 간의 갭을 평가하며, 기존 메트릭과 비모수적 분포 비교 기술을 사용하여 데이터 세트의 효과를 정량화합니다. 실험 결과, SDQM은 YOLOv11 모델과의 평균 평균 정밀도(mean Average Precision, mAP) 점수와 강한 상관관계를 보여줍니다.

- **Performance Highlights**: SDQM은 훈련된 모델의 성능과 강하게 연관되어 있으며, 이는 랜덤 플레인스(RarePlanes), 산업 금속 물체 데이터셋(DIMO), WASABI와 같은 다양한 데이터셋에서 일관된 성능 향상으로 입증되었습니다. 이 메트릭은 자원을 효율적으로 활용하고 반복적인 훈련/검증 사이클에 대한 의존성을 줄여주어, 저비용으로 데이터 품질을 개선할 수 있는 방향을 제시합니다.



### The Framework That Survives Bad Models: Human-AI Collaboration For Clinical Trials (https://arxiv.org/abs/2510.06567)
- **What's New**: 이번 연구는 인공지능(AI)이 임상 시험에서 어떻게 효과적으로 적용될 수 있는지를 분석했습니다. 특히, X-ray 이미지를 기반으로 한 질병 평가에서 AI와 인간 판독자의 상호 작용을 비교하여, AI가 임상 연구에 미치는 영향을 평가했습니다. 연구 결과, AI를 보조 리더(AI-SR)로 사용할 때 가장 유리한 결과를 보여 AI의 활용 가능성을 확인했습니다.

- **Technical Details**: 연구에서 사용된 방법론은 X-ray 데이터를 기반으로 AI 모델의 성능을 테스트하는 것이었습니다. 두 개의 임상 시험 MEASURE I와 PREVENT에서 환자 이미지를 분석하여, 총 점수(mSASSS)를 계산하고 예측 정확도를 평가했습니다. AI 모델은 nn-UNet과 mask-RCNN의 앙상블을 사용하여 척추 단위를 분할하고 mSASSS를 예측했습니다.

- **Performance Highlights**: 연구 결과, AI를 이용한 평가가 임상 시험의 시간과 비용을 줄일 수 있으며, 다양한 모델에서 발생할 수 있는 오류에도 불구하고 신뢰할 수 있는 결과를 유지할 수 있음을 보여주었습니다. AI 기술을 통해 질병 추정의 신뢰성을 보장하면서도 치료 효과 추정치와 결론을 유지할 수 있다는 점이 강조되었습니다.



### HSNet: Heterogeneous Subgraph Network for Single Image Super-resolution (https://arxiv.org/abs/2510.06564)
- **What's New**: 본 논문에서는 Heterogeneous Subgraph Network (HSNet)라는 새로운 프레임워크를 제안합니다. HSNet은 효율적인 그래프 모델링을 활용하면서도 계산 효율성을 유지합니다. 기존의 CNN 및 attention 기반 방법들이 가진 구조적 비유연성을 해결하기 위해, 이미지를 관리 가능한 하위 구성 요소로 분해하는 접근 방식을 사용합니다.

- **Technical Details**: HSNet의 핵심 구성 요소는 Constructive Subgraph Set Block (CSSB)과 Subgraph Aggregation Block (SAB)입니다. CSSB는 서로 다른 관계 패턴과 특징 상호작용을 모델링하여 다양한 보완적 하위 그래프를 생성합니다. SAB는 이들 하위 그래프에서 추출된 표현을 통합하여 복잡한 상호 의존성을 포착하는 포괄적이고 차별화된 표현을 구축합니다.

- **Performance Highlights**: 광범위한 실험을 통해 HSNet은 다섯 개의 SISR 벤치마크에서 최신 성능을 달성함을 보여줍니다. 제안된 구성 요소의 효과성을 검증하기 위한 ablation 연구 및 시각적 분석이 포함되어 있습니다. HSNet은 재구성 품질과 계산 효율성 간의 균형을 효과적으로 유지하며, 딥러닝 기반 이미지 처리 분야에 기여할 것으로 기대됩니다.



### The Algebra of Meaning: Why Machines Need Montague More Than Moore's Law (https://arxiv.org/abs/2510.06559)
- **What's New**: 이 논문에서 제안하는 Savassan 시스템은 현대 언어 모델의 비효율성을 해결하기 위해 Montague 스타일의 의미론을 적용합니다. 기존 AI 모델들이 출력의 의미를 제대로 파악하지 못하는 문제를 진단하고, 의미의 타입을 구성하는 방식을 통해 법적 및 비즈니스 상황에서의 의사결정을 위하여 새로운 방법론을 제시합니다. Savassan은 비구조적 입력을 처리하여 여러 법적 맥락에서 문장의 의미를 적절히 해석할 수 있도록 설계되었습니다.

- **Technical Details**: Savassan은 신경 기호(neuro-symbolic) 아키텍처를 기반으로 하여 발화를 Montague 스타일의 논리형식으로 변환하고, 여기에 의무(Deontic) 연산자 및 관할권(context) 추가가 이루어집니다. 이 시스템은 입력을 파싱한 후, 각기 다른 법적 온톨로지에 결과를 투사해 설명 가능성이 높은 의사 결정을 생성합니다. Savassan은 표본 구조의 기호적 검증(symbolic validation) 과정을 통해 법적 및 비즈니스 타입 시스템과 일치하도록 됩니다.

- **Performance Highlights**: Savassan의 성능은 법적 추론 벤치마크와 다중 관할권 평가를 통해 평가될 예정입니다. 시스템은 복잡한 법적 맥락에서도 반복적인 분류 과정을 개선하여 단 한번의 파싱으로 다양한 법적 위험을 평가할 수 있습니다. Savassan은 문장의 의미를 수학적 정밀도로 파악하여 설명 가능성이 높은 결과를 생성함으로써 기존 AI 시스템의 한계를 극복합니다.



### The Markovian Thinker (https://arxiv.org/abs/2510.06557)
- **What's New**: 이번 논문에서는 Markovian Thinking이라는 새로운 패러다임을 제안합니다. 이 방법은 정책(policy)이 일정 크기의 상태(state)에 따라 추론을 진행하도록 하여, 사고의 길이를 맥락의 크기와 분리합니다. 이를 통해 Delethink라는 강화학습(RL) 환경을 구성하여, 고정 크기의 청크로 사고를 구조화하면서도, 긴 사고를 더 효율적으로 처리할 수 있음을 보여줍니다.

- **Technical Details**: Delethink 환경은 각 청크의 경계에서 환경을 리셋하고 짧은 캐리오버로 프롬프트를 재초기화합니다. 에이전트는 강화학습을 통해 각 청크의 끝에서 연속성이 보장되는 텍스트 상태를 작성하도록 학습합니다. 이러한 과정은 결국 상수 메모리와 선형 계산(linear compute)을 가능하게 하여 기존의 표준 RL 환경에 비해 획기적인 성과를 나타냅니다.

- **Performance Highlights**: 트레이닝 된 R1-Distill 1.5B 모델은 8K 토큰 청크 내에서 사고를 진행하며 최대 24K 토큰 사고를 수행했습니다. 이는 24K 예산으로 훈련된 LongCoT을 능가하거나 맞먹는 성과를 보여줍니다. 실험 결과는 Delethink가 긴 사고를 수행하면서도 연산 비용과 메모리 사용에서 효율적임을 나타내며, 이로 인해 더 나은 성능을 보이는 것으로 분석되었습니다.



### Incoherence in goal-conditioned autoregressive models (https://arxiv.org/abs/2510.06545)
- **What's New**: 이 논문에서는 자기 회귀 모델의 단순 목표 조건화로부터 파생된 강화 학습 정책의 구조적 문제인 incoherence(불일치)의 개념을 수학적으로 조사합니다. 특히, 모델을 자신의 행동에 대해 재훈련하는 과정, 즉 오프라인에서 학습한 정책을 온라인 RL로 미세 조정하는 과정을 중심으로 연구하며, 이는 불일치를 감소시키고 수익을 향상시키는 데 기여한다고 입증합니다.

- **Technical Details**: 이 논문에서는 reinforcement learning(RL)을 inference 문제로 재구성하여, 최적 정책을 찾기 위한 명시적인 탐색 대신 행동이나 경로에 대한 생성 모델을 구성하고 이를 목표에 조건화하여 정책을 도출합니다. 정책은 주어진 상태에서 향후 행동의 확률 분포를 반영해야 하며, 이 과정에서 두 가지 질문의 차이를 강조합니다: (1) 주어진 보상을 달성하는 행동 선택 vs. (2) 자기 회귀적으로 선택되는 행동으로 보상을 달성하는 방법.

- **Performance Highlights**: 정책의 재훈련 과정에서 불일치를 제거하기 위한 세 가지 접근 방식을 제시하며, 그 중 첫 번째는 자신의 경로에 대한 정책 재조정을 포함합니다. 두 번째 접근은 온도 매개변수의 감소로, 이는 KL-정규화된 RL의 엔트로피 규제를 조작하는 방식으로 이루어집니다. 마지막으로, 보상을 통해 행동의 후방을 반영하는 것을 포함하는 방법을 설명하며, 이는 이론적으로 MDP의 수정 없이 수행됩니다.



### Scalable Policy-Based RL Algorithms for POMDPs (https://arxiv.org/abs/2510.06540)
Comments:
          36 pages, 3 Figures, Accepted at NeurIPS 2025

- **What's New**: 이 논문은 부분 관찰 강화를 통한 학습 문제(PORL)를 슈퍼스테이트 MDP라는 유한 상태 마르코프 결정 과정으로 변환하여 최적 정책을 학습하는 새로운 접근법을 제시합니다. 기존 POMDP 문제에서의 컴퓨테이셔널 차별성을 해결하기 위해, 이 연구는 강화학습 알고리즘을 활용하여 비선형 함수 근사를 통해 성능 보장을 제시합니다. 특히, 이 연구는 표준 TD 학습을 활용하여 최적 가치 함수와 초기 POMDP 가치 함수 간의 관계를 보장하는 이론적 개선을 제안합니다.

- **Technical Details**: 논문은 POMDP 모형을 슈퍼스테이트 MDP로 변환함으로써 연산적 복잡성을 줄이는 방법을 설명합니다. 이 연구는 전통적인 TD 학습 기법을 사용하여 연속 관측 이력을 기반으로 학습하여, POMDP의 최적 정책을 근사하는 새로운 알고리즘을 제안합니다. 특히, 저자들은 TD 학습의 수렴 보장을 제공하여 샘플링 매칭 문제를 해결하고, 전통적 방법보다 더 낮은 계산 복잡성으로 실용적인 성과를 보여줍니다.

- **Performance Highlights**: 이 연구의 주요 성과는 슈퍼스테이트 MDP에서의 성능 경계 성립을 보여주는 것입니다. 이는 기존 방법보다 높은 이론적 보장을 포함하며, POMDP 문제에 대한 구체적인 유한 시간 경계를 최초로 제시합니다. 또한 함수 근사 설정에서 정책 최적화 알고리즘의 성능 경계를 확장하여 대규모 상태 공간에서도 유용한 확장성을 제공합니다.



### CLAQS: Compact Learnable All-Quantum Token Mixer with Shared-ansatz for Text Classification (https://arxiv.org/abs/2510.06532)
- **What's New**: CLAQS는 고유한 양자 회로 내에서 복소수 가중치 혼합과 비선형 변환을 함께 학습하는 혁신적인 양자 토큰 믹서입니다. 이 모델은 8개의 데이터 큐비트를 사용하며, 얕은 회로로 이루어져 있습니다. 또한, 양자 믹싱과 최적화 과정에서 ℓ1 정규화를 적용하여 안정성을 높였습니다. 이러한 방식으로 텍스트 분류 작업에서 이전의 고전적 변환기와 강력한 하이브리드 모델들을 초월한 성능을 보입니다.

- **Technical Details**: CLAQS는 두 가지 주요 구성 요소를 갖추고 있습니다: 첫째, 복소수 가중치를 사용한 양자 토큰 혼합을 통해 데이터 기반으로 최적의 토큰 상호작용을 발견하며, 둘째, 문서 수준 집계를 통해 긴 시퀀스를 처리하는 슬라이딩 윈도우 방식입니다. 이 아키텍처는 Transformer 레이어의 구조를 모방하면서도 큐비트 효율성을 유지합니다. 훈련과정의 안정성을 위해 혼합 진폭에 대한 ℓ1 정규화 제약을 적용하여 과대 확대 또는 소실 문제를 완화합니다.

- **Performance Highlights**: CLAQS는 SST-2 데이터셋에서 91.64%, IMDB 데이터셋에서 87.08%의 정확도를 달성했습니다. 이 결과는 고전적 Transformer 및 강력한 하이브리드 모델들에 비해 뛰어난 성능을 보여줍니다. 또한, CLAQS는 짧은 큐비트 예산 내에서도 효과적으로 작동하는 첫 번째 사례로, 고전적인 주의(attention) 메커니즘과 양자 중첩의 차이를 메꿉니다. 이 연구는 진정으로 학습 가능한 양자 회로가 NLP에 효과적으로 활용될 수 있음을 보여줍니다.



### Visualizing Multimodality in Combinatorial Search Landscapes (https://arxiv.org/abs/2510.06517)
Comments:
          18 pages, 9 figures, Poster presented at the 2025 Symposium of the Norwegian Artificial Intelligence Society (NAIS 2025) on June 18, 2025

- **What's New**: 이 논문은 조합적 탐색 풍경(combinatorial search landscapes)에 대한 다양한 시각화 기법을 다루며, 다모드(multimodality)에 중점을 두고 있습니다. 시각화 기법의 결합을 통해 탐색 풍경을 보다 포괄적으로 이해할 수 있는 방법을 제시하고, 시각화 기법의 강점과 제한을 분석합니다. 또한, 'Grammar of Graphics'의 기하학적 및 미적 요소를 기반으로 하여 향후 연구 방향을 제안합니다.

- **Technical Details**: 조합적 문제는 다양한 최적화 알고리즘에서 매우 흔하게 나타나며, 이는 이 논문에서 다루는 주요 대상입니다. 우리는 탐색 풍경을 정의하기 위해 𝔏=(𝒳,f,𝒩)이라는 정식 표현을 도입하며, 여기서 𝒳은 탐색 공간(search space), f는 일반적인 피트니스(fitness) 함수, 𝒩은 접근 가능한 이웃(neighborhood)을 나타냅니다. 논문에서는 최적화 문제의 복잡성을 다루며, 다양한 거리 척도(distance metric)와 이웃 구조를 통해 탐색 풍경을 분석하는 방법을 소개합니다.

- **Performance Highlights**: 이 연구는 정보 디자인(information design) 관점에서 시각화 기법을 평가하며, 효과적인 시각화를 위한 심미적 요소(aesthetic attributes)와 기하학적 요소(geometries)의 결합 방안을 추천합니다. 또한 다모드를 가진 문제에서 최적 해(solution)를 찾기 위한 다양한 접근법을 제시하며, 기존 연구들과의 비교를 통해 이 기법의 유용성을 강조합니다. 이 작업은 조합적 탐색 풍경의 시각화를 개선하기 위한 여러 경로를 제시하면서, 비주얼라이제이션의 새로운 가능성을 탐색하고자 합니다.



### LogSTOP: Temporal Scores over Prediction Sequences for Matching and Retrieva (https://arxiv.org/abs/2510.06512)
- **What's New**: 이 논문에서는 객체 및 감정과 같은 로컬 속성의 점수(score)를 시퀀스의 시간적 속성에 맞게 변환하는 문제를 정식화합니다. 이를 위해 'Scores for TempOral Properties (STOPs)'라는 개념을 도입하였고, 이를 계산하는 새로운 점수 함수인 LogSTOP을 제안합니다. LogSTOP은 Linear Temporal Logic을 사용하여 시간적 속성을 효율적으로 계산할 수 있으며, 이를 통해 다양한 애플리케이션에서 유용하게 활용될 수 있습니다.

- **Technical Details**: LogSTOP은 시간 복잡도 (T⋅|φ|)을 가지며, 이는 기존의 방법보다 월등히 효율적입니다. 또한, LogSTOP은 로컬 속성 예측기의 부정확한 예측을 처리하기 위한 다운샘플링과 스무딩 전략을 사용하여 강건성을 향상시킵니다. Linear Temporal Logic의 다양한 연산자를 사용하여 복잡한 시간적 속성을 표현할 수 있으며, 이는 시퀀스 데이터의 다양한 구성 요소에 적용 가능합니다.

- **Performance Highlights**: LogSTOP은 YOLO 및 HuBERT와 함께 사용하여 쿼리 매칭에서 기존 모델보다 최소 16% 성능 향상을 보였습니다. 또한, Grounding DINO와 SlowR50을 사용하는 경우 시간적 속성에 대한 랭크된 검색에서도 평균 19% 및 16%의 향상을 기록하였습니다. 다양한 시퀀스와 속성에 적용 가능한 QMTP 및 TP2VR이라는 두 개의 새로운 벤치마크도 제안되었습니다.



### A Median Perspective on Unlabeled Data for Out-of-Distribution Detection (https://arxiv.org/abs/2510.06505)
- **What's New**: 이번 논문에서는 Medix라는 새로운 프레임워크를 제안하여, unlabeled wild data에서 out-of-distribution (OOD) 샘플을 식별하는 방법을 소개합니다. Medix는 중앙 경향성을 제공하는 median 연산을 사용하여 노이즈와 아웃라이어에 강인한 OOD 탐지 메커니즘으로 기능합니다. 이 방법은 기존의 기술들과는 달리 라벨이 없는 Wild 데이터를 효과적으로 활용하면서도 이론적인 오류 보장을 제공합니다.

- **Technical Details**: 논문에는 Medix의 이론적인 기초와 OOD 탐지를 위한 median 기반 최적화 프레임워크가 자세히 설명되어 있습니다. Medix는 InD 및 OOD 샘플이 혼합된 데이터에서 아웃라이어를 식별할 수 있도록 설계되었으며, 특정 비율 이하의 OOD 비율에서 소음의 영향이 관리 가능하다는 것을 보여줍니다. 학습 목표는 InD 데이터와 식별된 아웃라이어를 사용하여 강건한 OOD 분류기를 훈련하는 것입니다.

- **Performance Highlights**: Medix는 CIFAR-100에서 기존의 KNN+와 비교하여 평균 40.98%의 FPR95 개선을 달성하며, 다른 20개 기준선과 비교해도 우수한 성능을 보였습니다. 본 연구 결과는 Medix의 이론적인 주장과 실험 결과를 통해 뒷받침되며, 아울러 라벨이 없는 데이터에서 효과적인 아웃라이어 추출을 통해 낮은 오류율을 확인할 수 있었습니다.



### ATLO-ML: Adaptive Time-Length Optimizer for Machine Learning -- Insights from Air Quality Forecasting (https://arxiv.org/abs/2510.06503)
- **What's New**: 이 논문은 ATLO-ML이라는 적응형 시간 길이 최적화 시스템을 소개하며, 사용자 정의된 출력 시간 길이에 따라 최적의 입력 시간 길이와 샘플링 속도를 자동으로 결정합니다. 기존의 고정된 시간 길이 대신, 시간 시리즈 데이터 전처리에 유연한 접근 방식을 제공합니다. 이 시스템은 시간 내 변동성에 대한 예측 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: ATLO-ML은 두 가지 주요 데이터셋, GAMS-dataset과 데이터 센터에서 수집한 독점적인 데이터를 이용해 검증됩니다. 이 시스템은 다양한 시간 민감 애플리케이션에 대해 일반화할 수 있는 잠재력을 보여줍니다. 입력 매개변수의 최적화를 통해 머신러닝(ML) 작업흐름에서 시간 관련 문제를 해결하는 강력한 솔루션을 제공합니다.

- **Performance Highlights**: 결과에 따르면, 최적화된 시간 길이와 샘플링 속도를 활용했을 때 머신러닝 모델의 정확성이 고정된 시간 길이보다显著하게 개선됩니다. 이는 ATLO-ML이 머신러닝 예측 성능을 개선하는 데 중요한 역할을 한다는 것을 시사합니다. 따라서, ATLO-ML은 시간 기반 예측의 많은 분야에서 효과적인 도구가 될 것입니다.



### Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels (https://arxiv.org/abs/2510.06499)
- **What's New**: 이번 연구에서는 Webscale-RL이라고 하는 새로운 데이터 파이프라인을 소개합니다. 이 파이프라인은 대규모의 사전 훈련 문서를 시스템적으로 변환하여 수백만 개의 다양한 검증 가능한 질문-답변 쌍을 생성합니다. 이를 통해 기존의 RL 데이터셋과 비교하여 방대한 양의 데이터를 제공하고, RL 훈련이 기존보다 데이터 효율성을 높일 수 있도록 합니다.

- **Technical Details**: Webscale-RL 데이터셋은 1.2백만 개의 검증 가능한 QA 쌍을 포함하고 있으며, 9개 이상의 도메인에 걸쳐 다양한 내용을 다룹니다. 이 데이터셋은 데이터 필터링, 도메인 및 페르소나 주도의 생성, 품질 검증 단계를 포함한 웹 스케일의 데이터 엔진을 통해 구축되었습니다. RL 모델이 이 데이터셋을 통해 훈련될 때, 기존의 데이터 세트에서 지속적 사전 훈련을 수행한 모델보다 성능이 현저히 향상됩니다.

- **Performance Highlights**: Webscale-RL 데이터셋에서 훈련된 RL 모델은 100배 적은 토큰으로도 지속적인 사전 훈련과 유사한 성능을 달성했습니다. 이는 RL 접근 방식의 데이터 효율성을 입증하는 결과입니다. 이러한 연구 결과는 RL 훈련을 사전 훈련 수준으로 확장할 수 있는 가능성을 시사하며, 향상된 능력과 효율성을 갖춘 새로운 언어 모델의 출현을 촉진합니다.



### Valid Stopping for LLM Generation via Empirical Dynamic Formal Lif (https://arxiv.org/abs/2510.06478)
- **What's New**: 본 논문에서는 언어 모델 생성 중의 중지 결정을 위해 Anytime-valid sequential testing을 적용한 Sequential-EDFL(Empirical Dynamic Formal Lift)을 소개합니다. 이 접근법은 정보 증가량 정보 수집을 추적하며, 정해진 시간에 관계없이 오류 제어를 보장할 수 있는 self-normalized empirical-Bernstein e-processes를 사용합니다. Sequential-EDFL은 22-28%의 생성을 줄이는 동시에 오류를 제어하고 있으며, 안전 기준이 필요한 도메인에서 검증 부담을 줄이는 첫 번째 단계 필터 역할을 합니다.

- **Technical Details**: 우리는 skeleton을 기반으로 정보 증가를 측정하며, 전체 모델 예측과 고의적으로 약화된 기준 사이의 로그-우도 비율을 사용하여 정보 증가량을 정의합니다. 중요한 도전 과제는 조건부 기대값과 분포 변화에 대한 조정이 필요하며, 지수 과정을 사용해 이러한 문제를 해결합니다. 자동화된 skeleton 생성 방법과 진단 체크리스트를 통해, 우리는 EDFL의 적용을 현실화합니다.

- **Performance Highlights**: 연구에서 프로토타입은 6개 벤치마크 데이터셋(GSM8K, HotpotQA 등)에서 기존의 순차적 기준에 비해 생성량을 22-28% 감소시켰고, 이와 동시에 약 12%의 추가 계산 부하로 오류 제어를 유지했습니다. 또한, hybrid correctness gate를 추가하여 문장 경계를 강제하며 사실성을 개선하기 위한 방법을 제시합니다. 그러나 EDFL은 사실상의 정확성을 보증하지 않으며, 특정 도메인에 대한 검증 절차는 여전히 중요합니다.



### Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin (https://arxiv.org/abs/2510.06477)
- **What's New**: 이번 연구에서는 attention sinks와 compression valleys라는 두 가지 현상이 대형 언어 모델에서 어떻게 연결되는지를 밝힙니다. 이 두 가지 현상은 독립적으로 연구되었으나, 우리는 이들이 잔여 스트림(residual stream) 내에서의 대규모 활성화(massive activations) 형성과 관련이 있음을 보여줍니다. 연구를 통해 대규모 활성화가 표현 압축을 유도한다는 이론적 증거를 제시하며, 이로 인해 두 현상이 동시에 발생함을 실험적으로 확인했습니다.

- **Technical Details**: 연구의 근본적인 메커니즘은 대규모 활성화가 attention sinks와 compression valleys의 두 현상을 생성함에 있다는 것입니다. 연구는 Transformer 기반 대형 언어 모델에서 토큰 처리 과정을 세 가지 단계로 나누어 설명합니다: 초기 레이어에서는 넓은 혼합(broad mixing), 중간 레이어에서는 제한된 혼합을 통한 압축(compressed computation), 그리고 후반 레이어에서는 선택적인 정제(selective refinement)가 이루어진다고 주장합니다. 이 프레임워크는 embedding 작업과 generation 작업의 최적 깊이를 설명하는 데 기여합니다.

- **Performance Highlights**: 우리는 410M에서 120B 파라미터에 이르는 여러 모델에서 실험을 통해 대규모 활성화가 두 현상 모두에 미치는 영향을 검증했습니다. 더불어, 타겟형 ablative 연구를 통해 대규모 활성화를 제거함으로써 압축과 attention sink가 감소한다는 인과관계도 확인했습니다. 이 연구는 Transformer가 컴퓨테이션을 어떻게 조직화하는지를 이해하는 데 중요한 통찰을 제공하며, 각 작업의 특성에 따라 최적의 처리 깊이가 달라지는 이유를 풍부하게 설명합니다.



### Deep Generative Model for Human Mobility Behavior (https://arxiv.org/abs/2510.06473)
- **What's New**: 최근 연구에서 MobilityGen이라는 혁신적인 딥 생성 모델이 소개되어, 일주일에서 일주일 이상의 현실적인 인간 이동 경로를 대규모로 생성할 수 있게 되었습니다. 이 모델은 행동 속성과 환경 맥락을 연결하여 위치 방문, 활동 시간 배분 등의 주요 패턴을 재현합니다. MobilityGen은 기존 모델의 범위를 넘어서서 도시 공간 접근성이 여행 방식과 어떻게 변화하는지를 탐구하고, 사회적 노출 및 분리를 형성하는 공동 존재 동역학에 대한 통찰력을 제공합니다.

- **Technical Details**: MobilityGen은 고차원 데이터 분포를 학습하기 위해 DDPM(Denoising Diffusion Probabilistic Model)을 사용하며, 이는 원본 시퀀스를 점진적으로 변형하여 새로운 이벤트 시퀀스를 생성합니다. 모델은 트랜스포머(transformer) 인코더 네트워크를 활용하여 이동을 통한 데이터에서 특징을 추출하며, 이를 통해 활동 속성의 예측 결과를 생성합니다. 이 과정에서 각 속성들은 새로운 개인의 활동 이벤트 시퀀스에서 추출되어, 관측된 전환 및 조합 패턴을 준수하는 방식으로 샘플링됩니다.

- **Performance Highlights**: MobilityGen의 성능은 실험적으로 검증되었으며, 위치 선택 및 이동 패턴에서 실제 데이터와 잘 일치하는 성과를 보였습니다. 특히, 위치 방문 빈도, 회전 반경, 이동 엔트로피를 기준으로 다른 최신 이동 모델인 EPR(Exponential-Utility Peak-Rank) 및 Container와 비교하였을 때, MobilityGen이 더 우수한 적합도를 나타냈습니다. 또한, 세 개의 대표 도시인 베른, 취리히, 루체른에서의 이동 패턴을 모두 재현하며, 공간 분산의 차이를 보여주었으나 여전히 다른 마이크로모빌리티 모델들보다 나은 성능을 기록했습니다.



### Evaluating Node-tree Interfaces for AI Explainability (https://arxiv.org/abs/2510.06457)
Comments:
          5 pages, 2 figures. Accepted to the 3rd Workshop on Explainability in Human-Robot Collaboration: Real-World Concerns (XHRI 2025), scheduled for March 3, 2025, Hybrid (Melbourne and online) as part of HRI 2025

- **What's New**: 요즘 대규모 언어 모델(LLMs)이 직장 도구와 의사결정 과정에서 널리 사용되면서, AI의 설명 가능성과 사용자 신뢰를 보장하는 것이 중요해졌습니다. 본 연구는 노드-트리 인터페이스와 챗봇 인터페이스라는 두 가지 AI 인터페이스의 사용자 경험을 평가하여, 각 인터페이스가 탐색적, 후속 질문, 의사결정 및 문제 해결 작업에서 어떻게 성능을 발휘하는지를 조사합니다. 결과적으로 노드-트리 인터페이스가 브레인스토밍과 같은 복잡한 정보 탐색에서 사용자 신뢰를 높이는 데 기여한다고 보고합니다.

- **Technical Details**: 본 연구에서는 노드-트리 인터페이스를 도입하여 AI가 생성한 응답을 시각적으로 계층화하여 구성하는 디자인 중심 접근 방식을 사용하였습니다. 이 혁신적인 인터페이스는 서로 연결된 노드로 정보를 조직하여 사용자들이 복잡한 정보를 탐색하고 후속 질문을 할 수 있게 합니다. 연구에 참가한 20명의 비즈니스 사용자들로부터 양적 및 질적 데이터를 수집하여 사용자 신뢰, 과제 수행 능력 및 인터페이스 사용성을 평가하였습니다.

- **Performance Highlights**: 비교 연구 결과, 챗봇 인터페이스는 직선적이고 단계적인 질문에 유용하지만, 노드-트리 인터페이스는 정보 탐색 및 문제 해결에서 더욱 우수한 성능을 발휘했습니다. 노드-트리 인터페이스는 사용자 신뢰를 높이는 동시에 과제 수행 능력과 의사결정을 향상시키는 것으로 나타났습니다. 이 연구는 기업 애플리케이션을 위해 설계된 AI 인터페이스 개발에 있어 신뢰 구축의 중요성을 강조하며, 적응형 AI 인터페이스의 필요성을 보여줍니다.



### How NOT to benchmark your SITE metric: Beyond Static Leaderboards and Towards Realistic Evaluation (https://arxiv.org/abs/2510.06448)
- **What's New**: 이 연구는 Transferability estimation metrics의 현재 기준을 비판적으로 분석하며, 잘못된 벤치마크 구조가 기존 메트릭의 성능을 과대평가하는 방식을 설명합니다. 연구자들은 특히 STATIC performance hierarchy와 비현실적인 모델 공간이 어떻게 문제를 일으키는지 보여주며, 실질적으로 더 신뢰할 수 있는 평가를 위한 벤치마크 구축을 위한 권고안을 제시합니다.

- **Technical Details**: Transferability estimation의 개념은 다양한 사전 훈련된 모델 중에서 Fine-tuning 후 최상의 성능을 발휘할 모델을 예측하는 것입니다. Source Independent Transferability Estimation (SITE) 메트릭은 전체적인 구조 설계를 필요로 하며, 모델의 피쳐 표현과 타겟 데이터 세트의 관계를 평가합니다. 여러 SITE 메트릭은 비슷한 관점에서 발전해왔으며, 예를 들어 LogME는 최대 레이블 마진화 가능성을 수식으로 표현합니다.

- **Performance Highlights**: 연구에 따르면, 기존의 메트릭들보다 간단한 정적 랭킹 휴리스틱이 성능 평가에서 더 높은 신뢰성을 발휘할 수 있으며, 이는 현재 벤치마크가 가진 결점을 강조합니다. SITE 메트릭을 평가하기 위해 설정된 일반적인 기준과 모델 간 비교의 비효율성을 보여주며, 결국 연구자들은 조사된 메트릭의 실용성을 강화하는 조치를 취할 것을 권장합니다.



### A Survey on Agentic Security: Applications, Threats and Defenses (https://arxiv.org/abs/2510.06445)
- **What's New**: 이 논문은 자율 LLM(대형 언어 모델) 에이전트의 보안 환경에 대한 최초의 포괄적인 조사를 제시합니다. 기존의 수동 LLM에서 독립적으로 행동할 수 있는 LLM 에이전트로의 빠른 전환은 사이버 보안의 새로운 패러다임을 형성합니다. 이러한 변화는 공격 및 방어 작업에 강력한 도구로서의 잠재력을 제공하지만, 새로운 보안 리스크도 함께 도입됩니다.

- **Technical Details**: 저자들은 150개 이상의 논문을 포괄하는 세 가지 상호 의존적인 기둥: 응용(Application), 위협(Threat), 방어(Defense) 주위에 이 분야를 구조화했습니다. 이러한 보안 리스크들을 이해하기 위해 에이전트의 사용, 취약점, 그리고 보호를 위한 대응책을 체계적으로 설명합니다. 또한, 에이전트 아키텍처에서의 새로운 경향을 보여주는 상세 분석을 통해 모델 및 모드와 관련된 주요 연구 공백을 드러냅니다.

- **Performance Highlights**: 이 연구는 자율 LLM 에이전트의 보안 환경을 이해하는 데 필요한 기반을 제공합니다. 신뢰할 수 있는 응용 프로그램을 구축하기 위해 필요한 보안 요구사항과 취약점을 식별하고, 이러한 문제를 다루기 위한 다양한 방어 전략을 제시합니다. 이러한 정보는 향후 연구와 실용적인 적용을 위한 기초 자료로 활용될 수 있습니다.



### Context-Aware Inference via Performance Forecasting in Decentralized Learning Networks (https://arxiv.org/abs/2510.06444)
Comments:
          17 pages, 12 figures; appeared in ADI (October 2025)

- **What's New**: 이 논문에서는 분산 학습 네트워크 내에서 예측 성능을 예측하는 모델을 개발하였습니다. 이를 통해 상황에 따라 더 정확한 모델에 높은 가중치를 부여하는 'context-awareness'를 확보할 수 있습니다. 성능 예측 워커를 추가해 네트워크 추론의 정확도를 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 본 연구는 예측 모델의 성능을 예측하는 모델을 분산 학습 네트워크에서 설계하는 데 주목합니다. 네트워크는 다양한 알고리즘, 특성 및 비공식 데이터 세트와 함께 사용할 수 있는 모델을 통합합니다. 이 모델은 학습 네트워크의 설계와 목표 변수를 기반으로 하여 구성 요소와 특성을 최적화합니다.

- **Performance Highlights**: 성능 예측 모델은 예측 손실이나 후회(regret)와 같은 지표를 통해 예측 정확도를 개선하는 데 기여합니다. 최적의 특성 집합 및 교육 에포크 수에 따라 모델 성능이 민감하게 반응합니다. 예측 조합을 위한 성능 예측은 분산 환경에서뿐만 아니라 다양한 상황에서도 유용할 수 있습니다.



### Geometry-Aware Backdoor Attacks: Leveraging Curvature in Hyperbolic Embeddings (https://arxiv.org/abs/2510.06397)
- **What's New**: 이 논문은 비유클리드(Non-Euclidean) 모델인 하이퍼볼릭(hyperbolic) 신경망의 백도어(Backdoor) 취약성을 분석합니다. 하이퍼볼릭 기하학의 독특한 특성이 백도어 공격 조건을 변화시키며, 기하학적으로 적응하는 새로운 트리거를 제안합니다. 특이하게도, 경계 근처에서 미세한 입력 변화가 점차적으로 큰 모델 표현 공간의 변화를 일으킬 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 Poincaré 구 모델을 기반으로 비유클리드 공간을 정의하며, 이 공간의 리우만 기하(Riemannian geometry) 기반 메트릭을 설명합니다. 하이퍼볼릭 신경망에서의 주요 작업으로는 접선 벡터를 다양체에 투영하는 exponential map과 벡터 연산을 위한 Möbius 덧셈이 포함됩니다. 이러한 수학적 모델은 표준 입력 공간 탐지기를 회피하고 고유의 기하학적 특성을 활용한 백도어를 설계하는 데 사용됩니다.

- **Performance Highlights**: 하이퍼볼릭 신경망에서의 백도어 공격 성공률은 경계 근처에서 증가하며, 기존의 탐지기가 약해지는 경향을 보입니다. 실험 결과는 제안된 공격 방식이 유클리드 기반의 방법에 비해 효과적임을 입증하며, 기하학적으로 적응하는 트리거가 설계 및 방어의 한계를 이해하는 데 중요한 통찰을 제공합니다. 이는 비유클리드 모델에서 관련된 보안·방어 방법론에 대한 새로운 논의를 촉발합니다.



### Adaptive Protein Design Protocols and Middlewar (https://arxiv.org/abs/2510.06396)
Comments:
          N/A

- **What's New**: 본 논문은 AI/ML에 의해 변화하는 계산 단백질 설계(computational protein design)의 최신 동향을 소개합니다. IMPRESS(Integrated Machine Learning for Protein Structures at Scale)는 고성능 컴퓨팅(high-performance computing)과 AI를 결합하여 단백질 설계를 돕는 새로운 방법론을 제공합니다. 이 시스템을 통해 단백질 디자인의 품질이 일관되게 개선되고, 동적 자원 할당(dynamic resource allocation)으로 설계 처리량이 향상됩니다.

- **Technical Details**: IMPRESS는 AI 기반 생성 모델과 HPC 시뮬레이션을 통합하여 단백질 설계를 강화합니다. 이 프레임워크는 실시간 피드백이 가능하도록 설계되어 AI 시스템과 HPC 작업 간의 상호작용(bidirectional influence)을 확대합니다. 또한, ProteinMPNN과 AlphaFold 같은 도구를 활용하여 특정 펩타이드 타겟에 대한 단백질 결합체 설계를 최적화하는 과정이 포함되어 있습니다. 이 프로세스는 여러 단계의 컴퓨팅 작업을 포함하여, 각 단계에서 생성된 결과를 바탕으로 적절한 결정을 내릴 수 있도록 지원합니다.

- **Performance Highlights**: IMPRESS의 성능 평가는 HPC 시스템에서의 계산 성능과 생산된 단백질 품질의 과학적 결과를 바탕으로 진행됩니다. IMPRESS의 적응형 파이프라인은 RP(RADICAL-Pilot)를 통해 구현되어, 본 결과는 기존 방법에 비해 단백질 설계의 품질과 처리 속도가 개선되었음을 보여줍니다. 이런 통합 시스템을 통해 단백질 디자인의 효율성이 크게 증대되었습니다.



### Reward Model Perspectives: Whose Opinions Do Reward Models Reward? (https://arxiv.org/abs/2510.06391)
Comments:
          Published at EMNLP 2025 under the full author name "Elle"

- **What's New**: 이 연구는 보상 모델(Reward Models, RMs)의 행동을 포괄적으로 분석하는 새로운 프레임워크를 제시하며, 인간의 선호를 반영하는 류의 연구가 부족한 가운데 RMs가 나타내는 사회 인구학적 편향(sociodemographic biases)을 조사합니다. 특히, 특정 그룹의 선호에 맞춰 보상을 조정하기 위한 프롬프트(prompt)의 효과도 다룹니다. RMs가 다양한 사회 집단과 잘 정렬되지 않으며, 해로운 고정관념을 체계적으로 보상할 수 있다는 점을 강조합니다.

- **Technical Details**: 연구는 LM(Language Models)와 RMs의 정렬을 이해하기 위해 RMs의 태도, 의견, 가치(value)를 분석합니다. 이 과정에서 LM과 RMs는 인간의 의도를 반영하기 위해 RLHF(Reinforcement Learning from Human Feedback) 기법을 사용하고, 이러한 기법들은 종종 단일한 신념 세트에 정렬되어 있어 글로벌 관점의 다양성을 간과할 위험이 있습니다. 본 연구는 RMs에 의해 인코딩된 사회 인구학적 편향을 정량적으로 분석하는 최초의 사례로, 특정 RM 간의 상대적 정렬 측정에서 일관성을 발견했습니다.

- **Performance Highlights**: 결과적으로 RMs는 다수의 사회 집단에서 잘 정렬되지 않으며, 기본적으로 내재된 사회적 편향(social biases)으로 인해 원하는 공정성과 안전성을 충족하지 못할 수 있습니다. 연구에서 제안된 기존의 벤치마크는 과최적화(over-optimization)와 불분명한 사회적 편향으로 인해 성능 평가는 신뢰할 수 없음을 보여줍니다. 이로 인해 RMs의 적절성을 평가하고, AI 모델의 안전성과 정렬을 보장하기 위한 추가 연구가 필요하다고 강조합니다.



### Protecting De-identified Documents from Search-based Linkage Attacks (https://arxiv.org/abs/2510.06383)
- **What's New**: 이 논문은 기존의 de-identification 모델이 개인정보를 숨기는 데는 성공하지만, 원본 데이터에 대한 링크 위험(linkage risks)을 해결하지 못한다는 문제를 지적합니다. 향상된 방법으로는, 이 논문에서 제시한 N-그램(inverted index)을 기반으로 하여 텍스트를 변경하여 이러한 링크 공격을 방지하는 노력이 포함됩니다. 이 과정에서 원문의 의미를 계속 유지할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 두 단계로 진행됩니다. 첫 번째 단계에서는 문서 집합에서 발생하는 N-그램의 inverted index를 구성하여, 특정 문서의 수가 $k$ 미만인 N-그램을 효과적으로 찾아냅니다. 두 번째 단계에서는 LLM(based rewriter)을 사용하여 이 N-그램이 포함된 텍스트를 재구성하여 링크가 더 이상 불가능하도록 합니다.

- **Performance Highlights**: 법원 사건 데이터셋을 사용한 실험 결과, 제안된 방법이 기존의 텍스트 재작성 방법보다 검색 기반 링크를 효과적으로 방지할 수 있음을 보여주었습니다. 이는 문서의 원래 내용을 신뢰성 있게 유지하면서 링크 위험을 크게 줄일 수 있는 가능성을 시사합니다.



### Monte Carlo Permutation Search (https://arxiv.org/abs/2510.06381)
- **What's New**: 이번 논문에서는 일반 목적을 위한 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS) 알고리즘인 몬테카를로 순열 탐색(Monte Carlo Permutation Search, MCPS)을 제안합니다. MCPS는 GRAVE 알고리즘을 개선하여, 깊은 강화 학습(deep reinforcement learning)이 어려운 상황에서도 적용할 수 있습니다. 이 알고리즘의 핵심 원리는 노드의 탐색 항에 루트부터 노드까지의 모든 이동을 포함한 전체 플레이아웃(playouts)의 통계를 포함하는 것입니다.

- **Technical Details**: MCPS는 세 가지 통계 소스를 활용하여 GRAVE의 두 가지 통계 소스를 보완합니다. 첫 번째와 두 번째 통계는 GRAVE와 동일하며, 세 번째는 다양한 순서의 이동을 포함한 플레이아웃에 대한 통계입니다. MCPS는 통계 소스의 계수를 설정하기 위한 하이퍼파라미터(hyperparameter)가 필요 없으며, 나무의 조상 노드를 선택하는 데 사용되는 하이퍼파라미터에 대해서도 유사한 성능을 발휘합니다.

- **Performance Highlights**: MCPS는 다양한 게임에서 GRAVE보다 더 우수한 성과를 보였으며, 특히 이인용 게임에서 두드러진 성과를 냈습니다. 다인용 게임에서는 동등한 결과를 기록했는데, 이는 플레이어의 강력함에 따라 균형이 이루어져 있는 게임의 특성 때문입니다. 추상 코드(abstract codes)를 이동에 사용함으로써 MCPS와 GRAVE의 성능을 개선할 수 있다는 점도 주목할 만합니다.



### Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data (https://arxiv.org/abs/2510.06377)
Comments:
          preprint; under review

- **What's New**: 이 논문에서는 관계형 데이터를 처리하기 위해 새로운 Relational Transformer (RT) 아키텍처를 제안합니다. RT는 다양한 관계형 데이터베이스에서 사전 훈련되어 특정 작업이나 데이터셋에 대한 추가적인 튜닝 없이도 새로운 데이터셋과 작업에 직접 적용될 수 있습니다. 이는 관계형 데이터의 다양한 특성 때문에 기존의 아키텍처가 적용되기 어려웠던 문제를 해결합니다.

- **Technical Details**: RT 아키텍처는 (i) 테이블 및 열 메타데이터로 셀을 토큰화하고, (ii) 마스크 토큰 예측을 통해 사전 훈련되며, (iii) 열, 행, 기본-외래 키 링크에 대한 새로운 \textit{Relational Attention} 메커니즘을 활용합니다. 이러한 기능들은 RT가 다양한 관계형 데이터를 효과적으로 처리할 수 있게 합니다.

- **Performance Highlights**: RelBench 데이터셋에서 사전 훈련된 RT는 단일 순전파 (forward pass)로 이진 분류 작업에서 22M 파라미터 모델을 사용하여 94%의 AUROC를 기록하며, 이는 27B LLM이 기록한 84%보다 높은 성능입니다. 또한, 미세 조정을 통해 높은 샘플 효율성을 유지하며 최첨단의 결과를 달성합니다.



### EverydayMMQA: A Multilingual and Multimodal Framework for Culturally Grounded Spoken Visual QA (https://arxiv.org/abs/2510.06371)
Comments:
          Multimodal Foundation Models, Large Language Models, Native, Multilingual, Language Diversity, Contextual Understanding, Culturally Informed

- **What's New**: 이번 논문에서는 Everyday Multimodal and Multilingual QA (EverydayMMQA)라는 새로운 프레임워크를 소개하며, 문화적 기초가 있는 대규모 데이터셋을 구축하여 Visual Question Answering (VQA) 문제를 해결하고자 합니다. 이 프레임워크를 활용하여 OASIS라는 데이터셋을 개발하였고, 이는 다양한 언어와 문화적 상황을 반영한 0.92M개의 이미지와 14.8M개의 QA 쌍을 포함하고 있습니다. 이러한 데이터셋은 특히 자원이 부족한 언어에서의 질문에 대한 응답을 가능하게 합니다.

- **Technical Details**: OASIS 데이터셋은 음성, 이미지 및 텍스트를 통합하여 3.7M개의 발화 질문을 제공합니다. 또한, 데이터셋은 speech-only, text-only, speech+image, text+image의 네 가지 입력 조합을 지원하며, 영어와 아랍어 등 18개 나라의 다양한 실제 상황을 반영하도록 큐레이션되었습니다. 본 연구는 객체 인식 외에도 실용적이고 상식에 기반한 문화적 사고를 요구하는 작업을 평가합니다.

- **Performance Highlights**: 연구에서는 네 개의 폐쇄형 모델, 세 개의 오픈소스 모델, 그리고 하나의 파인튜닝된 모델을 벤치마킹하여 성능을 평가하였습니다. EverydayMMQA와 OASIS는 문화적 맥락 내에서의 일상 과제를 위한 다중 모달 (multimodal) 언어 모델 (LLMs) 구축을 위한 벤치마크 및 학습 데이터셋을 제공합니다. 프레임워크와 데이터셋은 커뮤니티에 공개될 예정입니다.



### Constrained Natural Language Action Planning for Resilient Embodied Systems (https://arxiv.org/abs/2510.06357)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)과 기호적 계획(symbolic planning) 컴포넌트의 강점을 결합하여, 고수준의 구체적 계획을 가능하게 하는 하이브리드 접근법을 제안합니다. 이 새로운 방법은 경직된 제약 조건을 명확하게 정의하면서도 LLM의 적응성(common sense reasoning)을 보존하여, 제약이 적은 열린 환경에서 안정적이고 반복 가능한 계획 솔루션을 제공합니다.

- **Technical Details**: 제안된 기호적 제약 언어 계획자(SCLPlan)는 LLM의 맥락을 활용하면서도, 기호적 계획의 엄격한 논리를 통해 명확한 환경 제약을 설정합니다. 이를 통해 LLM이 생성하는 잘못된 행동(다시 말해, hallucination)을 방지하고, 안정성과 반복성을 높이며, 명확한 제약 조건을 제공합니다. SCLPlan은 시뮬레이터와 실제 환경에서 실험을 통해 그 효용성을 입증하였습니다.

- **Performance Highlights**: ALFWorld 계획 벤치마크에서 SCLPlan은 기존 최첨단 방법들을 초월하며 거의 완벽에 가까운 99%의 성공률을 기록했습니다. 실제 환경에서 사족 로봇에 이 방법을 배치한 결과, 임무 성공률이 100%에 달했으며, 순수 LLM 및 기호적 계획기와 비교해 각각 50% 및 30%의 성공률을 보였습니다. 이 결과는 LLM 기반 로봇 계획기의 신뢰성, 반복 가능성 및 투명성을 높이는 효과적인 전략을 제안합니다.



### TransFIRA: Transfer Learning for Face Image Recognizability Assessmen (https://arxiv.org/abs/2510.06353)
Comments:
          Project Page: this https URL

- **What's New**: TransFIRA(Transfer Learning for Face Image Recognizability Assessment)는 신뢰할 수 있는 얼굴 인식 성능을 위해 새로운 접근 방식을 제공합니다. 기존의 FIQA( 얼굴 이미지 품질 평가) 접근법들이 휴먼 주석이나 복잡한 컴퓨팅 파이프라인에 의존하는 것과 달리, TransFIRA는 임베딩 공간에서 직접적으로 인식 가능성을 정의합니다. 이 프레임워크는 근본적으로 결정 경계와 정렬된 인식 가능성 기준을 제공하여 품질 평가의 정확성을 향상시키고, 일반적인 얼굴 인식 및 신체 인식에 모두 적용 가능합니다.

- **Technical Details**: TransFIRA는 세 가지 주요 발전을 포함합니다: (i) 클래스 중심 유사성(class-center similarity, CCS)과 클래스 중심 각도 분리(class-center angular separation, CCAS)를 통한 인식 가능성 정의, (ii) 외부 레이블이 없는 매우 정확한 검증 성능을 위한 인식 가능성 정보를 사용하는 집계 전략, (iii) 인식 가능성의 맥락에서 신체 인식을 평가하기 위한 새로운 확장을 제공합니다. 이 프레임워크는 모든 사전 훈련된 인코더와 호환되며, 별도의 기계 학습 과정 없이 인식 가능성을 예측할 수 있도록 설계되었습니다.

- **Performance Highlights**: TransFIRA는 BRIAR와 IJB-C 데이터세트에서 최신 FIQA 방법보다 뛰어난 성능을 기록했습니다. 실험은 얼굴 인식에 대한 최첨단 결과를 확인하였을 뿐만 아니라 신체 인식에 대한 강력한 성능도 입증했습니다. 또한, 모델은 다양한 데이터셋 간에 강건성을 유지하면서 인식 가능성 예측의 투명성을 제공하며, 인식 성능 저하의 원인인 흐림이나 가림 효과를 설명할 수 있는 능력을 갖추고 있습니다.



### Asking For It: Question-Answering for Predicting Rule Infractions in Online Content Moderation (https://arxiv.org/abs/2510.06350)
Comments:
          Accepted at ICWSM 2026

- **What's New**: 이 논문에서는 온라인 커뮤니티의 규칙과 이행 간의 관계를 모델링하고, 새로운 질문-응답 프레임워크인 ModQ를 도입합니다. ModQ는 기존의 분류(classification) 또는 생성(generation) 기반 접근법과는 달리, 커뮤니티의 모든 규칙을 고려하여 특정 댓글에 가장 잘 적용되는 규칙을 식별합니다. 이는 커뮤니티별 규칙의 변동성과 이행의 일관성 문제를 해결하는 데 기여할 수 있습니다.

- **Technical Details**: ModQ는 두 가지 모델 변형을 구현하여 커뮤니티 규칙 이행을 정보 추출 정보(extraction) 작업으로 모델링합니다. 첫 번째 모델인 ModQ-Extract는 사용자 댓글과 커뮤니티 규칙을 컨텍스트로 사용하여 특정 규칙을 추출합니다. 두 번째 모델인 ModQ-Select는 다중 선택(multiple-choice) 질문-응답 방식으로 댓글과 각 규칙 간의 정합성을 점수화하여 가장 적합한 규칙을 선택합니다.

- **Performance Highlights**: ModQ를 사용한 두 모델 모두 Reddit과 Lemmy 데이터셋에서 최신 기법을 능가하여 규칙 위반을 식별하는 데 강력한 성능을 보였습니다. 특히 ModQ-Select는 모든 기준 및 moderation 작업에서 모든 베이스라인을 일관되게 초과하며, 두 모델 모두 미리 경험하지 못한 새로운 커뮤니티와 규칙에 대해 효과적인 일반화 능력을 보여줍니다. 이는 빠르게 변화하는 플랫폼에서의 운영에 있어 큰 장점이 됩니다.



### Flexible Swarm Learning May Outpace Foundation Models in Essential Tasks (https://arxiv.org/abs/2510.06349)
- **What's New**: 이번 논문에서는 기초 모델(Fundation model)이 AI 발전에 미치는 영향을 다루며, AI가 인간의 결정 전략을 초월할 가능성에 대해 질문합니다. 특히, 의료와 같은 복잡한 시스템에서 고도 상호작용하는 기능들 간의 동적 장애물에 적응하는 데 있어 자율적이고 신뢰할 수 있는 AI 구조의 필요성을 강조합니다. 저자들은 모노리스(Monolithic) 기초 모델이 이러한 문제를 극복하는 데 한계를 가지며, 분산형 작은 에이전트 네트워크(SANs)를 제안합니다.

- **Technical Details**: 저자들은 상호작용하는 작은 에이전트 네트워크(SANs)가 복잡한 시스템에서 동적 환경에 적응하기 위한 새로운 접근 방식을 제공한다고 주장합니다. SAN은 각 에이전트가 시스템의 하위 구조를 나타내며, 이러한 에이전트가 특정 환경 변화에 대한 지속적인 업데이트를 통해 의사결정 과정을 최적화합니다. 특히, 저자는 그런 SAN의 군집 학습(Swarm learning) 접근이 효율적인 자기 적응(self-adaptation)을 가능하게 한다고 설명합니다.

- **Performance Highlights**: 저자들은 SANs 기반의 적응형 모델이 동적 환경 내에서 우수한 의사결정을 제공한다고 주장하며, 산소화(Oxygenation) 사례 연구를 통해 10분 내에 정확성을 회복하는 반면, 기초 모델은 약 200분이 걸린다고 밝혔습니다. 더 나아가, 그들은 SAN이 다양한 군집의 필요성을 강조하며, 이러한 다양성이 적응 과정의 탄력성을 증가시킨다고 주장합니다. 마지막으로, SAN 기반의 접근 방식이 전통적인 대형 모델보다 외부 스트레스 하에서 더 나은 최적화를 이루는 데 개념적으로 우수하다고 결론짓습니다.



### Leveraging Large Language Models for Cybersecurity Risk Assessment -- A Case from Forestry Cyber-Physical Systems (https://arxiv.org/abs/2510.06343)
Comments:
          Accepted at Autonomous Agents in Software Engineering (AgenticSE) Workshop, co-located with ASE 2025

- **What's New**: 이번 연구는 사이버 보안 위험 평가에 있어 로컬 호스팅된 대형 언어 모델(LLM)과 검색 증대 생성(Retrieval-Augmented Generation, RAG)의 사용 가능성을 탐구합니다. 특히, 자율 임업이라는 특수 분야에서 데이터 보호 및 프라이버시 요구 사항을 준수하면서 사이버 보안 전문가와 엔지니어들이 취약성과 위협을 평가하는 데 도움을 줄 수 있는 도구의 필요성을 강조합니다. 이 연구는 사이버 물리 시스템의 위험 평가 단계에서 LLM 기반 에이전트의 활용 확장을 제안합니다.

- **Technical Details**: 이 연구에서는 Llama 2와 같은 LLM과 RAG 구현을 활용하여 사이버 보안 위험 평가를 지원하는 맞춤형 도구를 개발하였습니다. 12명의 전문가를 대상으로 인터뷰와 상호 작용 세션을 진행하여 도구를 Iteratively 개선하는 방식을 채택했습니다. 연구 방법론은 디자인 과학 접근 방식으로, 문제를 이해하고, 디자인 아티팩트를 개발하며, 이를 평가하는 과정으로 구성되었습니다.

- **Performance Highlights**: 연구 결과, LLM은 사이버 보안 전문가에게 초기 위험 평가를 생성하고 위협을 식별하며 이중 확인을 제공하는 데 유용할 수 있는 것으로 나타났습니다. 그러나 전문가들은 정확성 및 준수를 보장하기 위해 인간의 감독이 필요하다고 강조했습니다. 또한 전문가들은 LLM의 생성 능력에 전적으로 의존하기보다는 특정 평가 및 지원 역할에서 LLM을 활용할 의향이 있음을 보여주었습니다.



### SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation (https://arxiv.org/abs/2510.06303)
Comments:
          Technical report. 39 pages, including 14 pages of appendix

- **What's New**: SDAR(Synergistic Diffusion-Autoregression)라는 새로운 패러다임이 제안되었습니다. 이 모델은 오토리그레시브(AR) 모델의 학습 효율성과 디퓨전(diffusion)의 병렬 추론(parallel inference) 능력을 통합합니다. 경량의 변환 단계를 통해 잘 훈련된 AR 모델을 블록형 디퓨전 모델로 변환함으로써 데이터 효율성을 극대화하고, 추론 시 모든 토큰을 병렬로 디코딩하는 방식으로 속도와 일관성을 확보합니다.

- **Technical Details**: SDAR는 경량의 적응 단계에서 AR 모델을 블록 단위의 디퓨전 모형으로 변환하는데, 이 구조는 디퓨전의 장점인 병렬 처리 가능성을 보장합니다. 디퓨전 모델은 블록 내에서 병렬로 생성을 처리하면서도 전역적으로 AR 프레임워크가 블록 간의 의존성을 모델링하여 효율성을 극대화합니다. 이 과정에서 KV 캐싱 및 가변 길이 생성과 같은 AR의 실용적인 이점을 유지하면서, 더 나은 모델링 정확성과 디코딩 속도를 제공합니다.

- **Performance Highlights**: SDAR는 수많은 실험을 통해 AR 모델이 마스크 디퓨전 모델보다 훨씬 더 높은 컴퓨팅 효율을 제공함을 보여주었습니다. 30B MoE 모델은 GPQA 및 ChemBench와 같은 과학적 추론 기준에서 AR 모델보다 우수한 성능을 발휘하며, 테스트 시 스케일링 방법에서도 개선된 결과를 보입니다. 이러한 결과는 SDAR가 오토리그레션과 디퓨전의 장점을 결합한 실제적인 패러다임을 확립했다는 것을 보여줍니다.



### RGBD Gaze Tracking Using Transformer for Feature Fusion (https://arxiv.org/abs/2510.06298)
Comments:
          Master Thesis with 125 pages, 59 figures, 17 tables

- **What's New**: 이번 연구에서는 RGBD(색상 및 깊이 정보를 포함한) 이미지를 사용하는 AI 기반 시선 추적 시스템을 구현하였습니다. Transformer 구조를 사용하여 이미지에서 추출된 특징들을 융합하는 모듈이 도입되었으며, RGBD 입력 이미지와 Transformer의 조합은 이전에 연구되지 않았습니다. 또한, 기존 데이터셋의 한계를 극복하기 위해 새로운 데이터셋이 생성되었습니다.

- **Technical Details**: 이 논문의 AI 모델 아키텍처는 Lian et al.의 이전 연구를 기반으로 하며, Generative Adversarial Network (GAN)을 사용하여 깊이 맵 아티팩트를 제거하고 머리 포즈 특징을 동시에 추출합니다. RGBD 이미지를 사용하는 주목할 만한 점은, 기존 데이터셋들은 깊이 정보를 포함하지 않거나 시선 각도 추정에 적합하지 않은 레이블만 갖고 있기 때문에 이번 연구가 유의미하다는 점입니다. 다양한 모델 구성으로 세 개의 데이터셋에서 훈련, 검증 및 평가가 이루어졌습니다.

- **Performance Highlights**: Transformer 모듈을 사용하는 모델은 ShanghaiTechGaze+ 데이터셋에서 평균 유클리드 오차가 55.3mm였으며, 프리트레인된 GAN 모듈을 사용하지 않았을 경우에는 30.1mm로 줄어들었습니다. Multilayer Perceptron (MLP)으로 Transformer 모듈을 대체하자 평균 오차가 26.9mm로 개선되었습니다. ETH-XGaze 데이터셋에서는 Transformer 모듈을 쓴 모델이 평균 각도 오차 3.59°를 달성하였고, Zhang et al.의 다른 모델은 평균 각도 오차 2.04°를 기록하였습니다.



### VeriEquivBench: An Equivalence Score for Ground-Truth-Free Evaluation of Formally Verifiable Cod (https://arxiv.org/abs/2510.06296)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)로 생성된 코드의 정확성을 보장하기 위한 공식 검증(formal verification)의 새로운 경계를 제시합니다. 새로운 벤치마크인 VeriEquivBench를 소개하며, 이는 2,389개의 복잡한 알고리즘 문제를 포함하고 있어 현재 모델의 한계를 파악하는 데 중점을 두고 있습니다.

- **Technical Details**: VeriEquivBench는 전통적인 실제 기준(ground-truth) 사양과 비교하는 대신, 공식적으로 기반을 둔 지표인 등가성 점수(equivalence score)를 적용하여 생성된 사양과 코드의 품질을 엄격하게 검증합니다. 이 평가 프레임워크는 코드 생성과 공식적 추론(formal reasoning)의 두 가지 분야에서 모델의 성능을 측정하기 위한 새로운 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과, 공식적으로 검증 가능한 코드를 생성하는 것은 최신 LLM들에겐 여전히 큰 도전 과제가 되고 있음을 보여주었습니다. 이러한 결과는 작업 난이도를 부각시키고, VeriEquivBench와 같은 벤치마크가 확장 가능하고 신뢰할 수 있는 코딩 에이전트 개발을 촉진하는 데 필요하다는 것을 강조합니다.



### Efficient High-Resolution Image Editing with Hallucination-Aware Loss and Adaptive Tiling (https://arxiv.org/abs/2510.06295)
Comments:
          Preprint. Under review

- **What's New**: 최근 모바일 애플리케이션에서 고해상도(4K) 이미지 생성(photo-to-image synthesis)의 중요성이 증가하고 있습니다. 본 논문에서는 리소스 제약이 있는 디바이스에서 메모리와 이미지 품질 문제를 해결하는 새로운 시스템인 MobilePicasso를 제안합니다. 이 시스템은 세 단계에서 이미지 편집을 수행하여 효율성을 극대화하는 동시에 비용과 메모리 사용을 최소화합니다.

- **Technical Details**: MobilePicasso는 세 가지 주요 단계를 포함합니다: 표준 해상도에서 이미지 편집을 수행하는 hallucination-aware 손실, 픽셀 공간으로의 이동 문제를 해결하는 latent projection, 마지막으로 변환된 이미지를 고해상도로 확장하는 adaptive context-preserving tiling을 사용합니다. 이를 통해 전반적인 이미지 품질을 18-48% 향상시키고 환각(hallucination)을 14-51% 감소시키는 효과를 보여줍니다.

- **Performance Highlights**: MobilePicasso는 낮은 지연 시간(latency)과 높은 성능을 자랑하며, A100 GPU에서 실행되는 기존 서버 기반 모델보다 빠른 속도를 기록했습니다. 특히, 시스템의 런타임 메모리 사용량은 9% 증가에 그쳤으며, 55.8배까지 속도를 개선했습니다. 이러한 결과는 MobilePicasso가 실제 적용 가능성이 높은 모델임을 입증합니다.



### BlockGPT: Spatio-Temporal Modelling of Rainfall via Frame-Level Autoregression (https://arxiv.org/abs/2510.06293)
- **What's New**: 이번 연구에서는 단기 강수 예측을 위한 새로운 모델 BlockGPT를 제안합니다. BlockGPT는 batched tokenization (Block) 방법을 사용하여 매 시간 단계에서 2차원 강수 맵을 예측합니다. 이 모델은 기존의 token-based 및 diffusion models의 한계를 극복하고, 실시간 데이터 예측을 위해 더 높은 성능과 효율성을 제공합니다.

- **Technical Details**: BlockGPT는 프레임 내의 self-attention과 프레임 간의 causal attention을 활용하여 시공간을 분해합니다. 이 모델은 강수 필드를 예측하는 두 단계로 구성되며, 첫 번째 단계에서 강수 필드를 latent token 공간으로 압축하고, 두 번째 단계에서 시간적 동역학을 autoregressively 모델링합니다. 예측된 강수 필드는 보다 일관된 예측을 제공하며, 학습 과정이 짧습니다.

- **Performance Highlights**: BlockGPT는 KNMI와 SEVIR 두 가지 데이터셋에서 평가되었으며, 기존 최첨단 모델에 비해 정확성 및 사건 로컬라이제이션에서 뛰어난 성능을 보입니다. 특히, BlockGPT는 병렬 처리 덕분에 기존 모델보다 최대 31배 더 빠른 추론 속도를 기록했습니다. 이 연구는 단기 강수 예측 분야에서 BlockGPT의 가능성을 보여줍니다.



### ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations (https://arxiv.org/abs/2510.06292)
- **What's New**: 이번 논문에서 제안한 ChainMPQ(Multi-Perspective Questions guided Interleaved Chain of Image and Text)는 LVLMs(Large Vision-Language Models)의 관계 인퍼런스에서 발생하는 환각을 개선하기 위해 고안된 트레이닝 없이 사용할 수 있는 방법론입니다. 기존의 방법들이 관계 환각을 단일 단계 추론으로 취급하는 반면, ChainMPQ는 시각적 및 텍스트 기억을 체계적으로 사용하여 다단계 추론 프로세스를 구현합니다. 이 접근 방식은 인간의 관계 추론 과정에서 영감을 받아, 주체, 객체 및 이 둘을 연결하는 관계라는 세 가지 핵심 요소에 중점을 둡니다.

- **Technical Details**: ChainMPQ는 관계를 주제와 객체 키워드로 분리하여 이미지의 관련 영역을 강화하며, 이를 통해관계를 구성하는 세 가지 요소에 집중하는 다각도의 질문을 생성합니다. 각 질문은 이전 단계에서 얻은 텍스트와 비주얼 메모리를 활용하여 순차적으로 입력됩니다. 이러한 방법은 각 관계를 적절히 추론할 수 있도록 돕고, 신뢰할 수 있는 결과를 제공하기 위해 비주얼 주의와 질문 분해를 통합합니다.

- **Performance Highlights**: 퍼포먼스 평가에서 ChainMPQ는 LLaVA-1.5 및 InstructBLIP 모델에서 관계 중심 벤치마크에서 일관되게 관계 환각을 줄이는 성과를 보였습니다. 제안된 방법은 이전의 세 가지 핵심 모듈의 효과를 검증하는 약한 연결 실험(ablation study)을 통해 그 효용성이 입증되었습니다. 이러한 결과는 LVLMs의 전반적인 성능 향상에 기여할 것으로 기대됩니다.



### Traj-Transformer: Diffusion Models with Transformer for GPS Trajectory Generation (https://arxiv.org/abs/2510.06291)
- **What's New**: 이 논문에서는 혁신적인 Trajectory Transformer 모델을 제안합니다. 이 모델은 conditional information embedding과 noise prediction을 위해 transformer backbone을 활용합니다. 제안된 방법은 GPS 궤적 생성에서 품질을 크게 향상시키며, 기존 방법에서 발생했던 편차 문제를 효과적으로 완화합니다.

- **Technical Details**: GPS 궤적 생성 문제를 명확히 정의하고, GPS 포인트를 두 가지 방법으로 임베딩하는 전략을 탐구합니다. 일반적인 GAN 구조 대신, 변형된 transformer 구조를 사용하여 GPS 궤적을 생성합니다. 이로 인해, 모델은 더 적은 파라미터로도 생성 품질을 개선하고 세밀한 도로 수준의 정보를 보존합니다.

- **Performance Highlights**: 실제 세계 데이터셋을 이용한 실험 결과, Trajectory Transformer는 이전에 제안된 방법들보다 훨씬 더 뛰어난 생성 품질을 보여줍니다. 정량적 지표와 정성적 시각화 모두에서, 기존 기법들이 가진 문제들을 상당히 완화하였음을 확인했습니다. 이러한 결과는 모델의 유용성을 높이고, 도시 계획 및 교통 흐름 예측과 같은 다양한 응용 분야에 적용할 수 있습니다.



### Soft-Evidence Fused Graph Neural Network for Cancer Driver Gene Identification across Multi-View Biological Graphs (https://arxiv.org/abs/2510.06290)
Comments:
          8pages

- **What's New**: 이번 연구에서는 Soft-Evidence Fusion Graph Neural Network (SEFGNN)을 제안하여 여러 생물학적 네트워크에서 암 유전자(CDG) 식별을 위한 새로운 접근 방식을 제공합니다. 기존의 GNN 기반 방법들과는 달리, SEFGNN은 각 네트워크를 독립적인 증거 원천으로 간주하고 의사 결정 수준에서 불확실성을 인식하여 융합(Fusion)합니다. 이는 Dempster-Shafer Theory (DST)를 이용하여 각 네트워크의 예측을 주관적 확률적 증거로 모델링하는 방식입니다.

- **Technical Details**: 연구는 NN 개의 생물학적 네트워크를 하나의 그래프로 모델링하고, 각 노드는 유전자, 각 엣지는 유전자 간의 상호작용을 나타냅니다. 최종 목표는 주어진 유전자가 암 유전자인지 여부를 이진 분류(binary classification)하는 것입니다. 모델 구조는 독립적으로 동작하는 GNN 세트, Dirichlet 분포에 따른 증거 변환, Dempster-Shafer 이론을 통한 증거 통합(multiple networks evidence integration)으로 구성됩니다.

- **Performance Highlights**: SEFGNN은 세 가지 암 데이터셋에서 광범위한 실험을 통해 기존 최첨단 기법들보다 일관성 있게 뛰어난 성능을 보였습니다. 이 모델은 새로운 암 유전자를 발견할 잠재력이 크며, 다양한 생물학적 네트워크에서의 유전자 관계를 효과적으로 모델링할 수 있는 능력을 demonstrated 합니다.



### SER-Diff: Synthetic Error Replay Diffusion for Incremental Brain Tumor Segmentation (https://arxiv.org/abs/2510.06283)
- **What's New**: 본 논문은 Synthetic Error Replay Diffusion (SER-Diff)라는 새로운 프레임워크를 제안합니다. 이는 기존의 Incremental Learning과 디퓨전(Noise Diffusion) 모델의 결합을 통해 뇌 종양 분할에서의 재학습 문제를 해결하고자 합니다. SER-Diff는 과거 작업으로부터 생성된 합성 오류 맵을 사용하여 새로운 작업의 학습 중에 재생(Replayed)하게 됩니다. 이 방식은 뛰어난 성능을 발휘하며, 이전의 catastrophic forgetting 문제를 완화합니다.

- **Technical Details**: SER-Diff는 세 가지 핵심 구성 요소로 구성됩니다: 1) Synthetic Error Replay 메커니즘을 통해 동결된 teacher diffusion 모델이 생성한 오류 맵을 재생합니다. 2) Diffusion 기반 정제 과정을 통해 이 오류 맵을 활용하여 분할의 일관성을 향상시킵니다. 3) Dual-loss(training objective) 전략을 활용하여 새로운 데이터에 대한 적응성과 이전에 습득한 지식의 보존을 동시에 보장합니다.

- **Performance Highlights**: 실험 결과, SER-Diff는 BraTS2020, BraTS2021, BraTS2023 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다. Dice 점수는 각각 95.8%, 94.9%, 94.6%로 최대치를 기록하였으며, HD95 값 또한 각각 4.4 mm, 4.7 mm, 4.9 mm로 최소치를 달성했습니다. 이러한 결과는 SER-Diff가 재학습 없이도 더 정확하고 해부학적으로 일관된 분할을 수행할 수 있음을 보여줍니다.



### Improving the Spatial Resolution of GONG Solar Images to GST Quality Using Deep Learning (https://arxiv.org/abs/2510.06281)
Comments:
          5 pages; accepted as a workshop paper in ICDM 2025

- **What's New**: 이 연구는 GAN 기반의 초해상도(superresolution) 방법을 통해 GONG의 저해상도(Hα) 태양 이미지를 BBSO의 고해상도(High-Resolution) 이미지 수준으로 개선하는 최초의 시도를 보여줍니다. 저해상도 이미지는 필라멘트(filament)와 섬유(fibril) 등의 미세 구조를 명확하게 재현하는 데 한계가 있었으나, 본 연구를 통해 GAN 방법을 사용하여 이 문제를 극복합니다.

- **Technical Details**: 본 연구에서는 Real-ESRGAN 모델과 잔차-잔차 밀집 블록(Residual-in-Residual Dense Blocks), 그리고 상대적(discriminator) 구별자를 적용하여 초해상도를 구현하였습니다. GONG과 GST의 이미지 쌍을 정렬하고, MSE가 467.15, RMSE가 21.59, CC가 0.7794인 성능을 달성하였습니다. 또한, 촬영된 이미지 간의 약간의 불일치가 정량적 성능에 제약을 주는데, 이는 향후 작업에서 해결할 계획입니다.

- **Performance Highlights**: 초해상도 모델은 태양 흑점의 반경과 필라멘트 및 섬유의 세부 정보를 효과적으로 재현할 수 있었습니다. 수집된 데이터 세트는 총 281개의 훈련 이미지 쌍과 63개의 테스트 이미지 쌍으로 구성되어 있습니다. 본 연구의 모델은 실제 응용 시나리오에서의 성능이 뛰어난 것을 입증하였으며, 이는 태양 관측의 정밀도를 크게 향상시킬 것으로 기대됩니다.



### Surgeons Are Indian Males and Speech Therapists Are White Females: Auditing Biases in Vision-Language Models for Healthcare Professionals (https://arxiv.org/abs/2510.06280)
- **What's New**: 본 연구는 의료 분야에서 비전 언어 모델(Vision Language Models, VLMs)의 직업별 편향을 평가하기 위한 프로토콜을 제시합니다. 의료 직군과 관련된 다양한 성별 및 인구통계적 편향을 정량화하고 운영 위험을 평가하는 방법론을 개발하였습니다. 이를 통해 기존의 VLM들이 어떻게 유사한 편향을 재생산하는지를 체계적으로 분석할 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 의사 및 의료 관련 직군에 대한 구조화된 분류법을 정의하고, 모델의 행동을 탐구하기 위한 직업 인식 프롬프트 스위트를 구성하며, 공정한 얼굴 데이터셋(FairFace)에 대한 인구통계적 편향 평가를 수행합니다. CLIP 및 OpenCLIP 모델 패밀리를 활용해 다양한 역할에 대한 성 편향을 평가하며, JS Divergence 기반 편향 점수를 사용하여 이들 모델의 차이를 분석합니다.

- **Performance Highlights**: 이 연구는 여러 VLM 모델에서 일관된 성 편향이 존재하는 것을 관찰하고, 이러한 편향이 AI 기반의 인력 채용 및 노동력 분석에 미치는 잠재적 영향을 강조합니다. 의료 환경에서 발생하는 비즈니스 의사결정과 환자 신뢰를 위한 편향 탐지의 중요성을 명확히 하며, 궁극적으로는 민감한 분야에서의 AI 사용에 대한 공정성을 증진시키기 위한 기반이 될 것입니다.



### RVFL-X: A Novel Randomized Network Based on Complex Transformed Real-Valued Tabular Datasets (https://arxiv.org/abs/2510.06278)
- **What's New**: 이 논문은 랜덤화된 신경망(Randomized Neural Networks, RNNs)에서 실수 값 데이터셋을 복소수 값 표현으로 변환하는 효과적인 방법이 부족함을 해결하기 위해 두 가지 방법, 즉 자연적인 변환과 오토인코더 기반 방법을 제안합니다. 이를 통해 RVFL-X라는 복소수 값 확장을 제안하며, 원래의 RVFL 아키텍처의 단순성과 효율성을 유지하면서도 복소수 변환을 통합합니다. RVFL-X는 입력, 가중치, 활성화 함수와 같은 복소수 구성 요소를 사용하여 복소수 표현을 처리하고 실수 값을 출력합니다.

- **Technical Details**: RVFL-X는 랜덤 벡터 기능 연결(Random Vector Functional Link, RVFL) 네트워크의 복소수 확장으로 설계되었으며, 모델의 복소수 매개변수를 도입하여 계산적 안정성 및 표현력을 높입니다. 기존의 RVFL 아키텍처와의 호환성을 유지하면서도 실수 데이터를 복소수 데이터로 전환할 수 있는 새로운 메커니즘을 제공합니다. 또한, 복소수 가중치와 활성화 함수의 통합을 통해 모델의 성능을 극대화하고, 이는 다양한 머신러닝 애플리케이션에서 유용하게 사용될 수 있습니다.

- **Performance Highlights**: 80개의 실측 UCI 데이터셋을 통해 RVFL-X는 원래 RVFL 및 최신 상태(State-Of-The-Art) RNN 변형들보다 일관되게 우수한 성능을 보였으며, 이는 다양한 응용 분야에 걸쳐 강력한 효과를 입증합니다. RVFL-X의 도입된 복소수 표현력이 반복적인 학습이 아닌 최적의 성능을 달성할 수 있도록 돕습니다. 결론적으로, 이 논문은 RVFL 모델의 새로운 가능성을 탐색하는 중요한 기초를 제공하며, 머신러닝의 다양한 도전과제를 해결하는 데 기여할 것으로 기대됩니다.



### A Total Variation Regularized Framework for Epilepsy-Related MRI Image Segmentation (https://arxiv.org/abs/2510.06276)
- **What's New**: 이번 논문에서는 약물 저항성 간질의 주요 원인인 Focal Cortical Dysplasia (FCD) 지역의 정확한 분할(segmentation) 작업을 위한 새로운 프레임워크를 제안하고 있습니다. 이 프레임워크는 최신 transformer-enhanced encoder-decoder 아키텍처를 채택하며, Dice loss와 anisotropic Total Variation (TV) 항을 결합한 새로운 손실 함수(loss function)를 도입합니다. 이러한 통합은 공간적인 부드러움(spatial smoothness)을 촉진하고, 포스트 프로세싱(post-processing)에 의존하지 않으면서 잘못된 양성 군집(false positive clusters)을 줄이는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법론은 3D 뇌 MRI 이미지에서 FCD 지역을 효과적으로 분할하기 위한 교육 파이프라인을 주의 깊게 설계합니다. 이 파이프라인은 패치 단위 샘플링(patch-wise sampling)과 바닐라 분류(voxel-wise classification)를 기반으로 하여 적은 수의 고차원 데이터로부터 모델이 효과적으로 학습할 수 있도록 합니다. 새로운 손실 함수는 Total Variation (TV) 정규화 항을 추가하여 이웃 바닥 예측의 급격한 변화를 패널티(penalize) 함으로써 보다 부드럽고 해부학적으로 일관된(segmentation masks) 분할 마스크를 생성하도록 유도합니다.

- **Performance Highlights**: 이 연구에서 제안된 모델은 공공 FCD 데이터셋(85명의 간질 환자로 구성)에서 성능을 평가하였으며, 기존의 손실 함수들과 비교하여 우수한 분할 정확도와 일관성을 보였습니다. 제안된 TV 손실 함수가 적용된 모델은 Dice 계수에서 11.9% 향상된 성능을 나타내고, 기준 모델보다 13.3% 더 높은 정밀도(precision)를 기록했습니다. 또한, 잘못된 양성 군집의 수는 61.6% 감소하여 더욱 효과적인 진단이 가능함을 보여주었습니다.



### Reproducibility Study of "XRec: Large Language Models for Explainable Recommendation" (https://arxiv.org/abs/2510.06275)
- **What's New**: 이번 연구에서는 Ma 외(2024)의 논문 "XRec: Large Language Models for Explainable Recommendation"에서 제시된 내용을 재현하였습니다. 원 저자들은 XRec 모델을 소개하였으며, 이는 대형 언어 모델(LLMs)에게 사용자에게 제공되는 추천에 대한 포괄적인 설명을 생성하는 기능을 부여합니다. 우리는 Llama 3를 사용하여 원본 논문의 결과를 재현하였으며, Mixture of Experts 모듈의 입력 및 출력 임베딩을 수정하여 성능을 향상시키려 했습니다.

- **Technical Details**: XRec 프레임워크는 그래프 기반 협업 필터링(Collaborative Filtering)과 LLMs를 결합하여 설명 가능한 추천을 제공합니다. 이 구조는 연결된 사용자-아이템 상호작용 그래프에서 GNN을 사용하여 더 높은 차원의 협업 관계를 포착하는 방식으로 작동합니다. XRec은 MoE(adapted embeddings)를 통해 GNN 임베딩을 LLM의 토큰 수준 표현 공간에 맞춰 조정하여 두 가지 구조를 효과적으로 결합합니다.

- **Performance Highlights**: XRec은 개인 맞춤형 설명을 생성하는 데 효과적이며, 협업 정보를 통합함으로써 안정성이 향상되었습니다. 그러나 모든 지표에서 XRec이 모든 기준 모델을 지속적으로 능가하는 것은 아니었습니다. 우리의 확장된 분석은 Mixture of Experts 임베딩이 설명 구조에 미치는 중요성을 강조하며, 협업 신호가 언어 모델링과 어떻게 상호작용하는지를 보여줍니다.



### MCCE: A Framework for Multi-LLM Collaborative Co-Evolution (https://arxiv.org/abs/2510.06270)
- **What's New**: 본 연구는 폐쇄형 소스(Closed-source) LLM과 경량화된 훈련 가능 모델을 연결한 Multi-LLM Collaborative Co-evolution (MCCE) 프레임워크를 제시합니다. 이러한 구조는 지역 최적해(optimizing local optima)에 갇히지 않고, 경험 기반 학습을 통해 지속적인 진화(evolution)와 협력을 가능하게 합니다. MCCE는 효율적인 다목적 최적화 작업을 가능하게 하며, 실질적인 응용 가능성을 가지고 있습니다.

- **Technical Details**: MCCE는 폐쇄형 LLM이 전 세계 탐색(global exploration)을 수행하는 동안, 경량화된 모델이 축적된 경험을 통해 보다 목표 지향적인 검색(targeted search)을 수행하게 합니다. 이러한 상호 보완적 관계는 각 모델의 강점을 강화하며, 지식을 지속적으로 내재화(internalize)하는 피드백 루프(feedback loop)를 통해 이루어집니다. 이 프레임워크는 기존의 정적 프롬프트 사용이나 RAG(Revelation-Augmented Generation)와는 다르게 경험을 깊이 있게 축적할 수 있는 모델 파라미터 업데이트(parameter updates)를 통해 효과적으로 문제 해결을 가능하게 합니다.

- **Performance Highlights**: MCCE는 다목적 약물 설계(multi-objective drug design) 벤치마크에서 최신 기술의 Pareto front 품질을 달성했으며, 기존의 기초 모델(base models)을 지속적으로 초월하는 성과를 기록했습니다. 이러한 성과는 LLM 시스템에서 경험 기반 학습과 지식 기반 탐색을 결합한 새로운 패러다임을 제시합니다. MCCE는 다양한 과학 및 공학 분야에 걸쳐 구조적 최적화가 필요한 작업에서도 확장 가능성을 지니고 있습니다.



### RareGraph-Synth: Knowledge-Guided Diffusion Models for Generating Privacy-Preserving Synthetic Patient Trajectories in Ultra-Rare Diseases (https://arxiv.org/abs/2510.06267)
Comments:
          6 pages, 2 figures, 2 tables. Submitted to IEEE International Conference on Data Science and Advanced Analytics (DSAA)

- **What's New**: RareGraph-Synth는 초희귀 질병을 위한 현실적이면서도 개인 정보 보호를 고려한 전자 건강 기록(EHR) 경로를 생성하는 지식 기반의 연속 시간 확산(framework) 프레임워크로 개발되었습니다. 이 시스템은 Orphanet/Orphadata, Human Phenotype Ontology (HPO), GARD rare-disease KG, PrimeKG, 및 FDA Adverse Event Reporting System (FAERS)와 같은 다섯 개의 공개 리소스를 통합하여 약 800만 개의 유형화된 엣지를 포함하는 이질적 지식 그래프(knowledge graph)를 형성합니다.

- **Technical Details**: RareGraph-Synth는 800만 개 엣지에서 추출된 메타 경로 점수를 사용하여 순방향(stochastic) 확산 미분 방정식에서 token noise를 조절합니다. 이를 통해 병리학적으로 그럴듯한 실험실-약물-부작용 동시 발생을 이끌어내면서도 안정적인 확산 모델(score-based diffusion model)의 특성을 유지합니다. 그 후 역 노이즈 감소기(reverse denoiser)는 개인 건강 정보를 포함하지 않는 실험실 코드, 약물 코드, 부작용 플래그의 세타임스탬프(timestamps) 시퀀스를 생성합니다.

- **Performance Highlights**: 시뮬레이션된 초희귀 질병 집단에서는 RareGraph-Synth가 비유도(diffusion baseline) 모델에 비해 범주별 최대 평균 불일치(Maximum Mean Discrepancy)를 40% 줄였으며, GAN 모델에 비해 60% 이상 감소시켰습니다. DOMIAS 공격자를 이용한 블랙박스 멤버십 추론 평가에서는 약 0.53의 AUROC를 기록하여 안전한 릴리즈(threshold)의 기준인 0.55 아래에 있으며, 비-KG 기준에 비해 상당히 개선된 결과를 보여주었습니다. 이러한 결과는 생물의학 지식 그래프를 확산 노이즈 일정을 통합하는 것이 데이터 공유의 안전성을 높이면서 신뢰성을 동시에 개선할 수 있음을 나타냅니다.



### Language models for longitudinal analysis of abusive content in Billboard Music Charts (https://arxiv.org/abs/2510.06266)
- **What's New**: 이번 연구는 최근 7개 년도 동안의 Billboard 차트 곡들을 심층 학습 방법을 통해 분석하여 음악의 선정적 내용의 변화를 추적합니다. 이 연구에서는 감정 분석(sentiment analysis) 및 폭력적 내용 탐지(abuse detection)를 포함하여 음악의 내용 진화를 검토합니다. 결과적으로, 1990년대 이후 대중 음악에서 선정적인 내용이 유의미하게 증가하고 있다는 것을 발견했습니다.

- **Technical Details**: 연구는 심층 학습(deep learning) 및 대형 언어 모델(LLMs)을 활용하여 음악가사(lyrics)의 미세한 패턴을 포착하는 장기적(longevity) 분석을 수행합니다. 또한, 데이터셋을 Billboard 차트에서 선정하여 곡을 명시적(explicit) 또는 비명시적(non-explicit)으로 분류하는 방법론을 제공합니다. 이러한 방법은 기존의 단어 목록 기반 접근 방식에 비해 더욱 강력하고 적응 가능한 내용 탐지를 가능하게 합니다.

- **Performance Highlights**: 연구 성과는 Billboard의 대중 음악에서의 비속어, 성적 내용, 그리고 부적절한 언어 사용의 증가 추세를 강조하며, 이는 사회적 규범과 언어 사용의 변화를 반영합니다. 이러한 분석은 교육자와 정책 입안자들이 더 안전한 음악 환경을 위한 정보에 기반한 결정을 내리도록 도와줄 것으로 기대됩니다. 연구 결과는 음악의 발달과 관련하여 중요한 심리적 영향을 미칠 수 있는 주제를 제기합니다.



### Dual-stage and Lightweight Patient Chart Summarization for Emergency Physicians (https://arxiv.org/abs/2510.06263)
Comments:
          Accepted at the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT) 2025

- **What's New**: 이 연구에서는 긴급 상황에서의 전자 건강 기록(EHR) 요약을 위한 새로운 시스템을 소개합니다. 이 시스템은 환자 정보를 검색하는 Jetson Nano-R 장치와 요약을 생성하는 Jetson Nano-S 장치를 사용하여, 환자 프라이버시에 대한 고려와 더불어 오프라인 작업이 가능하게 설계되었습니다.

- **Technical Details**: 제안된 시스템은 두 단계로 구성되며, 첫 번째 단계에서는 EHR에서 관련 정보를 검색하고, 두 번째 단계에서는 검색된 텍스트를 바탕으로 요약을 생성합니다. 이 두 단계의 분리는 자원 제약을 고려한 효율적 처리를 가능하게 합니다. 요약 결과는 필수 정보 목록과 임상 질의를 기반으로 한 맥락-specific 내러티브로 구분됩니다.

- **Performance Highlights**: 예비 결과에 따르면, 이 시스템은 MIMIC-IV 데이터베이스와 실세계의 비공식 EHR을 기반으로 30초 이내에 유용한 요약을 생성할 수 있습니다. 특히, FA 점수를 통한 검증 방법을 통해 요약의 사실 정확성을 평가하여, 임상에서의 신뢰성을 높였습니다.



### Prakriti200: A Questionnaire-Based Dataset of 200 Ayurvedic Prakriti Assessments (https://arxiv.org/abs/2510.06262)
Comments:
          4 pages, 4 figures

- **What's New**: 이번 연구에서는 전통 아유르베다 원리에 따라 개인의 신체적, 생리적, 심리적 특성을 평가하기 위한 이중 언어(영어-힌디어) Prakriti Assessment Questionnaire를 활용한 새로운 데이터 세트를 제공했습니다. 이 설문지는 24개의 다중 선택 항목으로 구성되어 있으며, 체형, 식욕, 수면 패턴, 에너지 수준 및 기질을 포함하여 다양한 특성을 평가합니다. 이 데이터는 자동 채점 시스템을 통해 개별 특성을 dosha(바타, 피타, 카파) 점수와 매핑하여 신뢰할 수 있는 분석을 가능하게 합니다.

- **Technical Details**: 이 데이터 수집 방법은 아유르베다 원칙에 근거한 이중 언어 설문지를 사용합니다. 24개의 다중 선택 질문은 신체적 특성(예: 체형, 신장), 생리적 특성(예: 식욕, 수면), 심리적 특성(예: 기질)에 대한 정보를 포함하고 있습니다. Google Forms를 통해 수집된 데이터는 완전성과 일관성을 검토하여 구조화된 xlsx 파일 형태로 저장되며, 각 참여자의 점수를 포함합니다.

- **Performance Highlights**: 최종 데이터 세트에는 200명의 참여자가 포함되어 있으며, 67.5%가 평균 체중, 62%가 평균 신장을 보고했습니다. 데이터 분석 결과 Pitta가 우세한 구성(97명)이 많았고, 뒤이어 혼합형과 같은 다른 유형의 구성이 나타났습니다. 이 데이터 세트는 향후 아유르베다 기반 연구와 헬스케어 어플리케이션 개발에 중요한 자료로 활용될 수 있습니다.



### Ensemble Deep Learning and LLM-Assisted Reporting for Automated Skin Lesion Diagnosis (https://arxiv.org/abs/2510.06260)
- **What's New**: 이 논문은 피부암 진단을 위한 새로운 통합 AI 프레임워크를 제안합니다. 이 시스템은 다양한 아키텍처를 가진 합성곱 신경망(CNN) 앙상블과 대형 언어 모델(LLM)을 결합하여 진단의 신뢰성과 접근성을 향상시킵니다. 또한, 이는 환자 교육을 포함한 임상 문서화의 필수 요구사항을 충족하면서 진단 출력이 임상적으로 의미 있는 평가로 변환되는 것을 가능하게 합니다.

- **Technical Details**: 제안된 시스템은 EfficientNetB3, ResNet50, DenseNet121의 이종 앙상블을 사용하여 복합적인 진단 관점을 제공합니다. 또한, 자동화된 보고서 생성을 위한 LLM 기반 시스템을 포함하여, 진단 추론 및 임상 출력을 동시에 수행할 수 있도록 설계되었습니다. 이 시스템은 불확실성 메커니즘을 내재하여 전문가 검토를 위한 비일관한 사례를 플래그하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 프레임워크는 진단 정확성을 높이는 동시에 환자가 진단 내용을 이해하고 조기 징후를 인식할 수 있게 지원합니다. 이를 통해 초기 발견률이 향상되고, 환자에게 개인화된 모니터링 지침을 제공하여 치료의 연속성을 지원합니다. 종합하여, 이 시스템은 인공지능(AI) 기반 피부 진단을 실제 임상에서 사용할 수 있는 솔루션으로 발전시키는 중요한 진전을 나타냅니다.



### LLM-Driven Rubric-Based Assessment of Algebraic Competence in Multi-Stage Block Coding Tasks with Design and Field Evaluation (https://arxiv.org/abs/2510.06253)
- **What's New**: 본 연구는 온라인 교육 플랫폼에서 학생들의 기초 이해를 평가하고 교육 목표에 맞춘 인지 과정의 깊이를 측정할 수 있는 새로운 평가 방법론을 제안합니다. 특히, 대형 언어 모델(LLM)을 활용한 루브릭 기반 평가 프레임워크를 도입하여 대수능력 및 실제 상황을 고려한 블록 코딩 작업을 평가하고 있습니다. 이 평가 방법은 학생들이 문제 해결 과정에서의 정확성과 품질을 동시에 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 연구에서는 수학 교육 전문가들이 설계한 문제 세트를 활용하여, 각 문제를 다섯 개의 미리 정의된 루브릭 차원과 일치시킴으로써 LLM이 학생들의 문제 해결 과정을 평가할 수 있도록 하였습니다. 온라인 플랫폼에서 모든 중간 응답을 기록하고, LLM을 통해 루브릭에 맞춘 성취 평가를 수행합니다. 이 시스템은 학습자 자기 평가와 전문가의 평가를 결합하여 효과성을 검증하였습니다.

- **Performance Highlights**: 필드 연구에는 42명의 중학생이 참여하였고, 다단계 이차 방정식 과제를 통해 제안한 프레임워크의 실용성을 평가하였습니다. LLM 기반 루브릭 평가는 전문가의 판단과 강한 일치를 보였고, 일관되게 루브릭에 맞춘 과정 중심의 피드백을 생성했습니다. 결과적으로, LLM을 활용한 루브릭 평가의 유효성과 확장 가능성을 입증하는 성과를 보여주었습니다.



### Dream2Image : An Open Multimodal EEG Dataset for Decoding and Visualizing Dreams with Artificial Intelligenc (https://arxiv.org/abs/2510.06252)
Comments:
          7 Pages, 3 Figures, The Dream2Image dataset is openly available on Hugging Face at: this https URL

- **What's New**: Dream2Image는 EEG 신호(EEG signals), 꿈 전사(dream transcriptions), 그리고 AI 생성 이미지(AI-generated images)를 결합한 세계 최초의 데이터셋입니다. 38명의 참가자와 31시간 이상의 꿈 EEG 녹음 기반으로, 총 129개의 샘플을 포함하고 있습니다. 이 데이터셋은 각성 직전의 뇌 활동 마지막 초(T-15, T-30, T-60, T-120)와 꿈의 경험에 대한 원시 보고서(raw reports)를 제공합니다.

- **Technical Details**: Dream2Image는 꿈 연구에 대한 새로운 자원이자, 꿈의 신경 상관관계(neural correlates of dreaming)를 연구하기 위한 독특한 자원입니다. 이 데이터셋은 뇌 활동(Brain activity)으로부터 꿈을 디코딩하는 모델을 개발하고, 신경과학(neuroscience), 심리학(psychology), 인공지능(artificial intelligence) 분야에서 새로운 접근 방식을 탐구하는 데 도움을 줍니다. 데이터셋은 Hugging Face와 GitHub에서 오픈 액세스(open access)로 제공됩니다.

- **Performance Highlights**: 이 데이터셋은 인공지능과 신경과학의 교차점에서 연구를 지원하기 위해 설계되었습니다. 현재의 뇌 활동 디코딩 방법을 확장하고 연구자들에게 영감을 주는 목적으로 만들어졌습니다. 하지만 샘플 크기가 상대적으로 작고 꿈 회상의 변동성(variability of dream recall)은 일반화 가능성에 영향을 미칠 수 있다는 한계가 있습니다.



### Scalable multilingual PII annotation for responsible AI in LLMs (https://arxiv.org/abs/2510.06250)
- **What's New**: 이번 연구는 다양한 규제 환경에서 개인 식별 정보(PII)의 신뢰할 수 있는 처리를 보장하기 위해 설계된 확장 가능한 다국어 데이터 주석 프레임워크를 소개합니다. 이 프레임워크는 13개의 저소득 언어 지역에서 약 336개 지역별 PII 유형에 대한 고품질 주석을 위한 것입니다. 단계별 인력 개입(HiTL) 주석 방법론을 통해 언어 전문 지식과 엄격한 품질 보증을 결합하여 메모리 유지율(Recall)과 잘못 분류된 긍정 사례(False Positive Rate)에서 큰 개선을 이루었습니다.

- **Technical Details**: 이 연구에서는 PII 주석을 위해 세 가지 단계(Pilot, Training, Production)로 나누어 진행되었습니다. 각 단계에서 작은 데이터 세트를 사용하여 초기 주석 문제를 식별한 후, 훈련 단계에서 정확한 PII 레이블링을 보장하기 위한 자료가 보강되었습니다. 마지막 생산 단계에서 표준화된 가이드라인을 바탕으로 품질 보증이 이루어져, 전 과정을 통해 지속적인 모니터링, 분석 및 피드백이 포함되었습니다.

- **Performance Highlights**: 주석 품질이 지속적으로 높은 수준을 유지할 수 있었던 것은 내부 품질 팀의 지속적인 모니터링과 피드백 덕분입니다. 연구에서는 주석자 간 합의(inter-annotator agreement) 및 오류 및 모호성의 근본 원인 분석(root-cause analysis)을 통해 성과를 객관적으로 측정하였습니다. 이 분석을 통해 더 나은 데이터 품질 및 LLM의 신뢰성을 확보하는 데 기여하였습니다.



### TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B (https://arxiv.org/abs/2510.06249)
Comments:
          It is work in progress

- **What's New**: 2025 다중모달 모델(Multimodal Models) 언어 챌린지는 인도의 다양한 저자원 언어(low-resource languages) 부족 문제를 해결하고자 합니다. 이 연구에서는 다국어 대형 언어 모델(multilingual large language model)에서 특정 내부 레이어의 교차 언어 유사성을 강화하는 것이 저자원 언어에서 고자원 언어(high-resource language)로의 번역 품질을 개선할 수 있는지를 조사합니다. 연구진은 Centered Kernel Alignment (CKA)와 REPINA라는 정규화 방법을 결합하여 TRepLiNa라는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 연구는 멀티링구얼 모델에서 레이어별 정렬(layer-wise alignment)을 시스템적으로 분석하는 첫 시도를 하고 있습니다. CKA와 TRepLiNa(CKA+REPINA)의 효과를 비교하기 위해, 특정 중간 레이어(약 10~15 레이어)의 유사성을 높이는 방법이 가장 효과적임을 보여줍니다. 이 방법론은 데이터가 부족한 설정에서도 효과적인 저자원 언어 번역 개선을 위해 활용됩니다.

- **Performance Highlights**: TRepLiNa를 적용함으로써 가중 복합 점수(weighted composite score)에서 실질적인 개선이 있음을 보였으며, 이 방법론의 적용 시기와 장소에 대한 가이드라인도 제공합니다. 실험 결과, TRepLiNa는 일반적으로 15번째 레이어에서 일관된 성과를 보였으며, 저자원 언어에서 고자원 언어로 번역 시, 번역 품질이 향상됨을 나타냅니다.



### DynBenchmark: Customizable Ground Truths to Benchmark Community Detection and Tracking in Temporal Networks (https://arxiv.org/abs/2510.06245)
- **What's New**: 이 논문에서는 네트워크의 동적 변화와 진화를 이해하기 위해 새로운 커뮤니티 중심 모델이 제안되었습니다. 이 모델은 커뮤니티가 성장하고 축소되며 병합되고 분리되는 커스터마이즈 가능한 커뮤니티 구조를 생성할 수 있도록 돕습니다. 기존 벤치마크들의 한계를 극복하기 위해, 실제 네트워크에서 커뮤니티의 진화를 추적할 수 있는 방안을 제시합니다.

- **Technical Details**: 제안된 벤치마크 모델은 시간적인 네트워크(temporal network)를 생성하며, 노드가 커뮤니티 간에 나타나거나 사라지거나 이동하는 방식도 포함됩니다. 이 모델은 Python 라이브러리 및 시각화 도구를 제공하며, 커뮤니티 발전을 추적하기 위한 알고리즘의 성능을 측정하는데 사용할 수 있습니다. 이 연구는 세 가지 방법을 테스트하여 노드의 클러스터 멤버십(cluster membership)과 커뮤니티 진화를 감지하는 성능을 평가하였습니다.

- **Performance Highlights**: 커뮤니티 발전을 검출하는 다양한 알고리즘들의 실제 성능을 비교하기 위한 검증 지표(validation metrics)가 포함되어 있습니다. 논문에 제시된 새로운 모델은 동적 커뮤니티(dynamic communities) 검출에 있어 향상된 성능을 보이며, 기존 데이터와 알고리즘 결과를 비교할 수 있는 효과적인 도구입니다. 이 벤치마크는 커뮤니티 탐지를 위한 성능 평가에 있어 중요한 기여를 할 것입니다.



### Evaluating Embedding Frameworks for Scientific Domain (https://arxiv.org/abs/2510.06244)
- **What's New**: 이 논문은 특정 도메인 데이터에 따라 같은 단어가 서로 다른 의미와 표현을 가질 수 있음을 강조하며, 과학 분야에 최적화된 단어 표현 알고리즘과 토큰화 방법을 연구하고 있습니다. 특히, 과학 기술 문헌에 적합한 새로운 평가 수트를 구축하여 다양한 단어 표현 및 토큰화 알고리즘의 성능을 평가하는 것을 목표로 하고 있습니다.

- **Technical Details**: 자연어 처리(NLP)에서 효과적인 단어 표현은 언어 이해, 텍스트 생성 및 감정 분석과 같은 작업에 중대한 영향을 미치며, 현재 사용되는 여러 토큰화 및 단어 임베딩 방법의 성능을 과학 분야에서 평가하는 데 중점을 두고 있습니다. Byte Pair Encoding(BPE), WordPiece, Unigram Tokenizer 등의 다양한 토큰화 방법과 Word2Vec, GloVe, FastText와 같은 단어 임베딩 기법이 소개되며, 각 방법은 특정 장단점을 가지고 있습니다.

- **Performance Highlights**: 연구는 다양한 다운스트림 NLP 과제를 포함한 포괄적인 평가 수트를 구축하여 과학 분야에서의 단어 표현과 토큰화 모델을 비교하고 있습니다. 특히, 낮은 리소스 모델인 Word2Vec과 계산적으로 더 많은 자원을 소모하는 Transformer 기반 모델의 성능 비교를 통해 과학 분야에서의 단어 표현 문제에 대한 통찰력을 제공하고 있습니다.



### CoT Referring: Improving Referring Expression Tasks with Grounded Reasoning (https://arxiv.org/abs/2510.06243)
Comments:
          MLLM, Referring Expression Segmentation

- **What's New**: 이 논문은 Referring Expression (RE) 작업에 대해 새로운 추론 메커니즘인 CoT Referring (CoTR)을 제안합니다. 이 방법은 복잡한 질의에서 언어와 이미지를 통합하여 모델의 성능을 향상시킵니다. CoTR은 모델이 순차적인 논리(logic)를 명확히 모델링할 수 있도록 하여, 목표 객체를 올바르게 찾을 수 있게 합니다.

- **Technical Details**: CoT Referring 접근법은 텍스트 구조를 체계적으로 분석하여 각 단계에서 관계를 식별하고 참조 정렬을 보장합니다. 새로운 출력 형식을 강제하기 위해 기존 데이터를 재구성하고, 복잡한 참조 사례에 대한 평가 기준을 개발했습니다. 이 평가 기준은 3개 이상의 교차 관련 객체가 포함된 복합 참조 표현을 위해 특별히 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 RefCOCO/+/g 벤치마크에서 기존 모델에 비해 2.5%의 성능 향상을 보여주었습니다. 새로운 평가 기준에 따른 실험 결과, 우리의 접근법이 복잡한 참조 표현에서 목표 로컬라이제이션(target localization)을 효과적으로 개선하는 것을 확인하였습니다.



### Transparent Reference-free Automated Evaluation of Open-Ended User Survey Responses (https://arxiv.org/abs/2510.06242)
Comments:
          EMNLP Industry Track

- **What's New**: 본 연구에서는 인간 작성 설문 응답의 평가를 위한 두 단계 평가 프레임워크를 제안합니다. 기존의 자동 평가 방법은 LLM(대형 언어 모델)으로 생성된 텍스트를 대상으로 하며, 인간 작성 응답의 독특한 특성을 적절히 평가하지 못했습니다. 우리의 접근 방식은 비정상적인 응답을 필터링하고 세 가지 차원인 노력(effort), 관련성(relevance), 완전성(completeness)을 평가합니다.

- **Technical Details**: 이 프레임워크는 비정상적인 응답을 제거하는 gibberish filtering 단계를 포함합니다. 이후, 현실 세계의 설문 데이터에 대한 실증 분석을 기반으로 LLM 능력을 활용하여 각각의 응답을 세 가지 차원에서 평가합니다. 이를 통해 응답의 질을 예측하고 불량 응답을 거부하는 실제 응용 프로그램에서도 높은 효율성을 발휘합니다.

- **Performance Highlights**: 영어와 한국어 데이터셋에 대한 검증 결과, 제안한 프레임워크는 기존 지표들을 초월하며, 전문가 평가와 강한 상관관계를 보입니다. 이는 설문 연구에서의 응답 품질 예측과 응답 거부뿐만 아니라 실제 환경에서의 적용 가능성을 더욱 높입니다.



### Knowledge Graph-Guided Multi-Agent Distillation for Reliable Industrial Question Answering with Datasets (https://arxiv.org/abs/2510.06240)
Comments:
          41 pages, 12 figures, 6 tables

- **What's New**: KG-MASD(Knowledge Graph-guided Multi-Agent System Distillation)를 제안하여 산업 QA 시스템의 안전성과 신뢰성을 높이고자 하였습니다. 이 새로운 접근 방식은 Markov Decision Process(MDP)로 모델링되어, Knowledge Graph를 활용하여 상태 표현을 풍부하게 하고 수렴을 보장합니다. KG-MASD는 고신뢰도의 instruction-tuning 데이터를 생성하고, 경량화된 학생 모델에 깊은 추론 능력과 검증 가능성을 동시에 이식할 수 있는 여지를 제공합니다.

- **Technical Details**: KG-MASD는 각 단계에서 Knowledge Graph를 활용하여 상태를 업데이트하는 방법으로 구성되며, 이 과정을 통해 얻은 고품질의 도메인 일치 triple들이 학생 모델의 학습과정에 긍정적인 영향을 미칩니다. 이론적으로, KG-guided priors는 증류 효율성을 개선하는 데 도움이 되며, 시스템이 고위험 환경에서 신뢰할 수 있는 결과를 생산할 수 있도록 합니다. 더불어 KG-MASD는 복잡한 산업 시나리오에서도 학생 모델이 교사 모델의 복잡한 추론 패턴을 믿을 수 있게 전이할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, KG-MASD는 기존 모델들에 비해 정확도가 2.4%에서 20.1%까지 향상되었으며, 신뢰성을 크게 증가시켜 안전-critical 산업 시나리오에서의 AI 배치를 가능하게 했습니다. 또한, KG-MASD는 Hallucination 현상을 완화하고, 단일 교사 모델 및 다중 에이전트 기법 대비 성능이 뛰어난 것으로 입증되었습니다. 이러한 성과는 KG-MASD가 Trustworthy AI 제공에 기여할 수 있음을 보여줍니다.



### Uncertainty Quantification In Surface Landmines and UXO Classification Using MC Dropou (https://arxiv.org/abs/2510.06238)
Comments:
          This work has been accepted and presented at IGARSS 2025 and will appear in the IEEE IGARSS 2025 proceedings

- **What's New**: 이번 연구는 인간 지뢰 제거 작업에서 표면 지뢰 및 비폭발 잔해(UXOs)의 탐지를 위한 Monte Carlo (MC) Dropout을 이용한 불확실성 정량화 개념을 도입합니다. ResNet-50 아키텍처에 통합하여 실험한 결과는 기존 신경망 모델의 취약성을 극복할 수 있는 가능성을 제시합니다. MC Dropout 접근 방식은 예측 신뢰성을 추가적인 지표로 제공하여 지뢰 제거 작전에서 더 신뢰할 수 있는 결정을 내리는 데 도움이 될 수 있습니다.

- **Technical Details**: 이 연구에서는 ResNet-50 모델을 이용하여 MC Dropout을 통합함으로써 예측 불확실성을 추정하는 방법을 소개합니다. 연구팀은 깨끗한 테스트 이미지, 적대적 변형 및 노이즈가 있는 테스트 이미지를 포함한 세 가지 시나리오에서 모델을 평가했습니다. MC Dropout은 모델의 가중치를 확률적 분포로 보고, 예측 불확실성을 효율적으로 추정할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 다수의 확률적 전방 패스를 통해 얻은 평균 예측 및 그 분산은 불확실성의 지표 역할을 했습니다. 예측의 분산이 높을수록 모델 예측의 불확실성이 증가하는 것을 확인했습니다. 이 연구는 표면 지뢰 및 UXOs 분류에서 불확실성 정량화의 필요성을 강조하며, 향후 현실 세계의 지뢰 제거 애플리케이션에 적용할 수 있는 기초를 마련합니다.



### Stacked Regression using Off-the-shelf, Stimulus-tuned and Fine-tuned Neural Networks for Predicting fMRI Brain Responses to Movies (Algonauts 2025 Report) (https://arxiv.org/abs/2510.06235)
- **What's New**: Algonauts 2025 Challenge에 대한 새로운 접근 방식을 소개합니다. 이 연구에서는 대규모 언어 모델, 비디오 인코더, 오디오 모델, 비전-언어 모델의 멀티모달 표현을 통합하여 영화 자극에 대한 fMRI 뇌 반응을 예측하였습니다. 세밀한 전사(transcripts)와 요약을 통해 텍스트 입력을 향상시켰고, 자극 조정(stimulus-tuning) 및 미세 조정(fine-tuning) 전략을 실험하였습니다. 우리 팀의 제출 결과는 27개 제출 중 10위를 기록하였고, 모든 코드와 리소스를 공개하여 멀티모달 인코딩 모델 개발에 기여하였습니다.

- **Technical Details**: 본 연구에서는 영화 자극에 따른 뇌 활동을 예측하기 위해 다양한 전략을 적용하였습니다. 주로, 사전 훈련된 심층 신경망의 내부(x) 표현을 이용하여 선형 모델을 적합시키고, 비전-언어 모델에서 더 풍부한 표현을 추출하기 위해 전사를 개선하였습니다. 또한, 대규모 언어 모델 및 비전 모델을 자극에 맞게 조정하여 뇌 예측 정확도를 향상시키는 실험도 진행하였습니다. 최종적으로 세 개의 예측 소스를 스택 리그레션(stacked regression)을 통해 결합하여 결과를 도출하였습니다.

- **Performance Highlights**: 우리 팀의 최종 모형은 whisper-small과 Llama-3.1-8B를 포함한 두 개의 사전 훈련된 심층 신경망에 기반한 선형 예측과 향상된 전사에서 추출된 InternVL 표현을 사용하여 성능을 평가하였습니다. 이 조합을 통해 뇌 활동 예측에서 높은 정확도를 나타냈으며, 시험 데이터에 대해 상위 성능을 달성하였습니다. 궁극적으로, 본 연구의 접근 방식은 다양한 모델 기준에서 효과를 보여주었고, Algonauts 대회에서 좋은 성과를 기록하였습니다.



### Generalized Multi-agent Social Simulation Framework (https://arxiv.org/abs/2510.06225)
- **What's New**: 본 논문에서는 모듈형 객체지향 프레임워크를 설계 및 개발하여 다양한 기본 클래스와 유기적으로 통합된 계층 구조를 제공합니다. 이를 통해 상호작용의 규모와 재사용성을 획기적으로 향상시키고자 합니다. 또한, 새로운 메모리 요약 메커니즘을 제안하여 불필요한 정보를 걸러내고, 중요한 사건과 상호작용을 우선시하여 인공지능 에이전트가 더 적응적이고 인지적으로 그럴듯한 행동을 보일 수 있도록 합니다.

- **Technical Details**: 모듈형 디자인은 사회적 시뮬레이션의 핵심 구성 요소를 일반 기본 클래스로 정의하여, 이 클래스를 상속받아 특화된 환경을 생성합니다. 이 구조는 다양한 사회적 환경을 시뮬레이션 하는 데 적합하게끔 설계되어 있습니다. 각 모듈은 사회 시뮬레이션, 세계, 도구, 페르소나, 자원, 조직, 메모리, 업무 흐름 등을 포괄하여 에이전트가 자율적으로 상호작용할 수 있는 기반을 마련합니다.

- **Performance Highlights**: 제안된 프레임워크를 통해 소셜 미디어에서의 인간 상호작용을 성공적으로 시뮬레이션 하였으며, 실제 온라인 사회 행동을 복제하는 데 성공했습니다. 이를 통해 다양한 환경에 맞춰 개인화된 시뮬레이션이 가능해졌습니다. 또한, 메모리 요약 메커니즘의 도입으로 정보 과부하를 줄여 포괄적이고 효율적인 사회적 상호작용을 가능하게 하였습니다.



### Exploring Human-AI Collaboration Using Mental Models of Early Adopters of Multi-Agent Generative AI Tools (https://arxiv.org/abs/2510.06224)
Comments:
          19 pages, 1 table, 2 figures

- **What's New**: 본 연구는 다중 에이전트 생성 AI(Multi-Agent Generative AI)의 초기 수용자와 개발자들이 어떻게 AI 협업 도구를 인식하고 이해하는지를 조사합니다. 연구팀은 마이크로소프트에서 일하는 13명의 개발자와 반구조적 인터뷰를 진행하여, 이들이 AI 에이전트를 협력자로 보지 않고 단순한 도구로 보는 것에서 벗어나 적극적인 팀원으로 인식하는 변화를 발견했습니다. 연구 결과에 따르면, 초기 수용자들은 AI 에이전트를 '팀'으로 이해하며, 이는 인간 협업 모델과 유사한 구조를 가지고 있음을 보여줍니다.

- **Technical Details**: 연구는 사용자가 다중 에이전트 생성 AI 시스템을 어떻게 해석하고 협업하는지에 대한 인식을 분석하였습니다. 특히, 이들은 AI 시스템을 '도움자' 또는 '검토자'로서의 전문화된 역할 기반 및 작업 기반 에이전트의 집합체로 인식합니다. 이러한 과정을 통해 명확한 커뮤니케이션과 투명성을 필요로 하고 있으며, 에러 전파와 비생산적인 에이전트 루프 행동과 같은 주요 과제도 확인되었습니다.

- **Performance Highlights**: 초기 수용자들은 투명성의 역할에 대해 신뢰를 형성하고 오류를 검증 및 추적하는 방법으로서의 중요성을 강조했습니다. 이들은 협업 상에서 AI의 기능과 그에 따른 도전 과제들에 대해심도 깊은 통찰을 제공하며, 향후 연구 방향도 제시하고 있습니다. 연구 결과는 AI가 인간과 협업하는 다양한 방식과 이를 통해 발생한 새로운 협력 동역학을 이해하는 데 기여하고 있습니다.



### A Multimodal GUI Architecture for Interfacing with LLM-Based Conversational Assistants (https://arxiv.org/abs/2510.06223)
Comments:
          24 pages, 19 figures

- **What's New**: 최근 대형 언어 모델(LLMs)과 실시간 음성 인식 기술의 발전으로 사용자 인터페이스(GUI)에서 음성 명령을 사용하여 작업을 수행하고, 시스템의 응답을 GUI를 통해 직접 받을 수 있는 것이 가능해졌습니다. 이 논문은 LLM 기반 음성 지원 도우미와 GUI 간의 상호작용을 가능하게 하는 구체적인 아키텍처를 소개합니다. Model Context Protocol (MCP)을 통해 애플리케이션의 내비게이션 그래프와 의미론을 제공함으로써, 앞으로의 OS 슈퍼 어시스턴트를 위한 준비를 할 수 있도록 합니다.

- **Technical Details**: 소개된 아키텍처는 MVVM (Model-View-ViewModel) 패턴의 ViewModel을 통해 애플리케이션의 기능을 도우미에 노출시킵니다. 이는 현재 시각적으로 보이는 뷰에 적용 가능한 도구와 GUI 트리 라우터에서 추출된 애플리케이션 범위 도구를 포함합니다. 이러한 아키텍처는 음성 접근성을 완전하게 지원하고, 화자 입력과 시각적 인터페이스 간의 신뢰할 수 있는 정렬을 보장합니다.

- **Performance Highlights**:  최근 오픈 가중 모델의 성능이 선도적인 상용 모델과 유사하다는 평가 결과가 있으며, 빠른 응답성을 위한 엔터프라이즈 등급의 하드웨어가 필요하다는 사실도 밝혀졌습니다. 논문에서 제시하는 시스템은 음성 인식, 고품질 실시간 STT(speech-to-text) 및 즉각적인 멀티모달 피드백(첫 번째 표시 및 TTS)를 지원하며, 사용자에게 원활하고 안정적인 상호작용을 제공합니다.



### AgentBuilder: Exploring Scaffolds for Prototyping User Experiences of Interface Agents (https://arxiv.org/abs/2510.04452)
- **What's New**: 이 논문에서는 생성 AI 모델로 구동되는 인터페이스 에이전트(agents)의 개발에 있어 사용자 경험(agent experience)의 중요성을 강조합니다. AI 엔지니어 외의 다양한 개인들이 에이전트 경험을 프로토타입할 수 있도록 지원할 필요성이 커지고 있습니다. 따라서, 본 연구는 에이전트 프로토타이핑 시스템이 제공해야 할 특성을 탐색합니다.

- **Technical Details**: 본 연구는 12명의 참가자를 대상으로 요구사항 도출 연구를 수행하여 에이전트 경험 프로토타입의 주요 활동과 에이전트 프로토타이핑 시스템의 원하는 기능을 식별합니다. 이러한 기능들은 AgentBuilder라는 디자인 탐색 도구에 적용되어 에이전트 프로토타입 생성에 도움을 줍니다. 연구에서는 14명의 참가자를 대상으로 AgentBuilder를 사용하여 인 situ(in situ) 에이전트 프로토타이핑 연구를 시행합니다.

- **Performance Highlights**: 이 연구는 개발자가 에이전트를 프로토타입하는 방식과 그 과정에서의 요구 사항에 대한 통찰력을 확보하는 데 중점을 둡니다. 프로토타입 시스템의 설계 요구 사항을 검증하는 동시에, 참가자들의 다양한 경험을 통해 에이전트 경험을 풍부하게 하는 데 기여합니다. 이 과정은 에이전트 디자인의 다양한 관점을 고려하여 보다 포괄적인 사용자 경험을 제공할 수 있는 기반을 마련합니다.



### TiltXter: CNN-based Electro-tactile Rendering of Tilt Angle for Telemanipulation of Pasteur Pipettes (https://arxiv.org/abs/2409.15838)
Comments:
          Manuscript accepted to IEEE Telepresence 2024. arXiv admin note: text overlap with arXiv:2204.03521 by other authors

- **What's New**: 이 연구는 텔레조작(teleoperation) 시스템의 혁신적인 접근 방식을 제안하고 있으며, 이는 Convolutional Neural Networks (CNN)을 기반으로 한 새로운 시스템을 통해 변형 가능한 물체의 기울기를 인식하는 데 중점을 두고 있습니다. 이 시스템은 Force Dimension Omega.7 촉각 장치와 Robotiq 그리퍼에 내장된 전자자극 패턴을 결합하여 사용자의 촉각 피드백을 향상시킵니다. 이러한 접근 방식은 의료 분야에서 원격 임상 시험의 안전성을 높이기 위해 개발되었습니다.

- **Technical Details**: 제안된 TiltXter 시스템은 로봇 팔의 끝에 장착된 2F Robotiq 그리퍼를 제어하기 위해 Force Dimension Omega.7 촉각 인터페이스를 사용합니다. 이 시스템은 전자자극 패드와 촉각 센서 배열을 통해 사용자의 손가락으로 원격 물체의 촉각 정보를 전달합니다. 시스템 아키텍처는 PC를 기반으로 하여 CNN 노드를 실행하고 그리퍼와의 통신을 관리합니다.

- **Performance Highlights**: 실험 결과, CNN 알고리즘을 활용한 기울기 인식의 정확도가 감소된 데이터로는 23.13%에서 57.9%로 증가하였고, 텔레조작 성공률은 53.12%에서 92.18%로 향상되었습니다. 이는 텔레조작 시 사용자에게 제공되는 촉각 패턴이 정확한 물체 인식에 큰 기여를 한다는 것을 보여줍니다.



### DeepXPalm: Tilt and Position Rendering using Palm-worn Haptic Display and CNN-based Tactile Pattern Recognition (https://arxiv.org/abs/2204.03521)
Comments:
          Accepted paper in IEEE Haptic Symposium 2022, IEEE copyright

- **What's New**: 본 논문은 변형 가능한 물체의 원거리 조작(telemanipulation)을 위한 새로운 시스템을 제안합니다. 이 시스템은 사용자의 손바닥에서 촉각 피드백(haptic feedback)을 제공하는 다중 접촉(haptic device) 장치 LinkGlide와 Robotiq 그리퍼에 장착된 두 개의 촉각 센서(tactile sensors) 배열로 구성되어 있습니다. 해당 시스템은 물체의 기울기(tilt angle)와 위치(position)를 정확하게 인식하기 위한 새로운 접근법을 소개하며, 이는 CNN(Convolutional Neural Networks) 기반의 방법론입니다.

- **Technical Details**: 이 연구에서 제안된 CNN 모델은 변형 가능한 물체를 잡고 있는 동안 기울기 및 위치를 탐지하는 데 사용됩니다. CNN은 인식된 기울기 및 위치 데이터를 바탕으로 마스크(mask)를 생성하고, 이를 통해 사용자가 경험할 수 있는 다중 접촉 촉각 자극을 효과적으로 렌더링합니다. 이를 통해 복잡한 물체의 조작을 용이하게 하고, 사용자에게 보다 명확한 촉각 패턴을 제공할 수 있습니다.

- **Performance Highlights**: 연구 결과, CNN 알고리즘과 사전 설정된 마스크를 사용하면 사용자의 기울기 및 위치 인식 정확도가 9.67%에서 82.5%로 향상되었습니다. 이는 전통적인 직접 데이터 사용 방법에 비해 큰 개선을 보여줍니다. 이러한 성과는 정밀하고 민첩한 조작을 요구하는 변형 가능한 물체 작업에서의 잠재적 적용 가능성을 시사합니다.



New uploads on arXiv(cs.LG)

### h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning (https://arxiv.org/abs/2510.07312)
Comments:
          Preprint, 31 pages, 8 figures

- **What's New**: 이 연구에서는 기존의 짧은 호리존(short-horizon) 데이터를 활용하여 장기 호리존(long-horizon) 추론 능력을 향상시키기 위한 새로운 방법을 제안합니다. 기존 접근법들이 inference-time scaffolding나 비싼 step-level supervision에 의존하는 반면, 우리의 방법은 간단한 문제들을 합성하여 복잡한 multi-step dependency chains를 구성합니다. 이러한 방식으로, 별도의 주석 없이도 장기 호리존 데이터 생성을 가능하게 하여 RL(강화학습) 훈련을 효율적으로 확장합니다.

- **Technical Details**: 제안된 방법에서는 short-horizon 문제(예: GSM8K 질문들)를 연결하여 무제한 길이와 복잡성을 갖는 종속 추론 단계 체인을 생성합니다. 이 데이터에 대해 outcome-only rewards를 사용하는 RL로 훈련하며, 훈련 과정에서 다가오는 문제의 난이도를 자동으로 증가시키는 커리큘럼을 적용합니다. 이를 통해 RL 훈련에서 성능 포화 문제를 해결하고 모델이 유용한 장기 호리존 추론 경로를 내부화하도록 돕습니다.

- **Performance Highlights**: 이 연구의 결과는 composed 6학년 수준의 수학 문제(GSM8K)에 대한 커리큘럼 훈련이 경쟁 수준의 장기 벤치마크(GSM-Symbolic, MATH-500, AIME)에서 최대 2.06배의 정확도 향상을 이끌어냈음을 보여줍니다. 또한, RL training을 통해 기존의 강력한 baseline와 비교했을 때, 모델들이 새로운 추론 경로를 학습할 수 있음을 입증했습니다. 이 연구는 장기 호리존 문제를 해결하는 RL의 효율적인 경로를 제시하며, 기존의 데이터만으로도 장기 호리존 추론을 가능하게 합니다.



### MLE-Smith: Scaling MLE Tasks with Automated Multi-Agent Pipelin (https://arxiv.org/abs/2510.07307)
- **What's New**: MLE-Smith는 기존의 수동적으로 Curate된 MLE 벤치마크의 한계를 극복하기 위해, 원시 데이터셋을 자동으로 경쟁 스타일의 MLE 작업으로 변환하는 다중 에이전트 파이프라인을 소개합니다. 이 시스템은 generate-verify-execute 패러다임을 통해 MLE 작업의 품질을 자동으로 제공하며, 실제 사용 가능성과 다양성을 보장합니다. 특히 224개의 실제 데이터셋을 활용하여 606개의 MLE 작업을 성공적으로 생성했다는 점이 주목받습니다.

- **Technical Details**: MLE-Smith는 구조적 완전성(Structural Integrity), 의미적 타당성(Semantic Soundness), 실증적 해결 가능성(Empirical Solvability)을 보장하기 위해 다중 에이전트 생성 워크플로우와 하이브리드 검증 메커니즘을 통합합니다. 이 시스템은 Brainstormer, Designer, Refactor라는 세 가지 전문 에이전트를 통해 작업 제안을 생성, 구체화 및 표준화합니다. 또한, 강력한 검증 메커니즘이 지속적으로 작업의 정확성과 일관성을 보장합니다.

- **Performance Highlights**: MLE-Smith에서 생성된 606개의 작업은 잘 설계된 인간 Curate 작업과의 상관 관계가 높은 것으로 나타났습니다. 이 결과는 MLE-Smith가 다양한 범위의 작업에서 첨단 LLM들이 성능을 평가하는 데 효과적임을 보여줍니다. MLE-Smith는 다음 세대 MLE 에이전트를 평가하고 교육하는 데 적합한 도전적이고 일반화 가능한 작업을 제공합니다.



### MolGA: Molecular Graph Adaptation with Pre-trained 2D Graph Encoder (https://arxiv.org/abs/2510.07289)
Comments:
          Under review

- **What's New**: 본 연구에서는 기존 2D 그래프 인코더를 사용하여 분자 도메인 지식을 유연하게 통합할 수 있는 MolGA라는 새로운 접근 방식을 제안합니다. MolGA는 두 가지 주요 전략을 통해 분자의 특성을 반영하면서도 기존의 그래프 인코더들을 활용하여 다운스트림(Downstream) 작업에서 성능을 높입니다. 특히, 분자 정렬 전략(molecular alignment strategy)과 조건부 적응 메커니즘(conditional adaptation mechanism)을 통해 작업 직무에 최적화된 토큰을 생성하는 점이 혁신적입니다.

- **Technical Details**: MolGA의 구현에는 두 가지 주요 도전 과제가 있습니다. 첫째, 사전 훈련된 2D 토폴로지 표현(pre-trained 2D topological representations)과 분자 도메인 지식(molecular domain knowledge) 간의 정렬입니다. 둘째, 정렬된 지식을 활용하여 세부적인 특성을 다운스트림 작업에 적응시키는 방법입니다. 이 과정에서 경량의 조건부 네트워크를 활용하여 인스턴스별로 특화된 토큰을 생성하고, 이를 통해 사전 훈련된 그래프 인코더와 연계하는 방식으로 진행됩니다.

- **Performance Highlights**: 우리는 다양한 공개 데이터셋에 대해 광범위한 실험을 수행하였으며, MolGA가 여러 분자 작업에서 기존의 최첨단 방법들과 비교할 때 일관되게 뛰어난 성능을 보임을 확인하였습니다. MolGA의 접근 방식은 기존 2D 인코더를 통해 비용을 줄이면서도 성능을 극대화할 수 있는 가능성을 보여줍니다. 이러한 결과는 분자 그래프 표현 학습 및 사전 훈련 기술의 향상을 위한 중요한 기초 자료로 활용될 것입니다.



### Evolutionary Profiles for Protein Fitness Prediction (https://arxiv.org/abs/2510.07286)
- **What's New**: 이번 논문에서는 EvoIF라는 경량 모델을 소개하며, 이는 단백질 진화를 내재된 보상 극대화 과정으로 해석하고, 단백질 언어 모델(pLM)을 역 강화 학습(ILR)으로 간주함으로써 기존의 단백질 기능 예측을 통합하는 새로운 방식을 제안합니다. EvoIF는 동족 단백질에서 회수한 서열 프로필과 역접기(logits)에서 추출한 구조 진화 제약을 통합하여 단백질의 기능적 비극성과 생존 기여도를 정량화할 수 있습니다. 이 접근법은 적은 양의 데이터와 모델 파라미터로도 높은 예측 정확도를 제공하는 것을 목표로 하고 있습니다.

- **Technical Details**: EvoIF는 서열-구조 정보를 경량의 서열-구조 백본을 사용하여 인코딩하며, 역접기 프로필과 구조 회수 동종성 프로필이라는 두 개의 압축된 진화적 정보를 주입함으로써 작동합니다. 이 모델은 진화 정보를 다양한 측면에서 통합하여 단백질 적합도 예측을 위한 강력한 근거를 제공합니다. 단백질 적합도의 예측은 단백질 변이체의 기능적 성능을 정량적으로 측정하는 여러 방법을 사용하여 이루어집니다.

- **Performance Highlights**: EvoIF는 ProteinGym에서 217개의 변이 실험을 기반으로 250만 개 이상의 변이체를 대상으로 최첨단 성능을 달성하였으며, 이는 0.15%의 훈련 데이터와 최근의 대형 모델보다 적은 파라미터를 사용하여 이루어졌습니다. 추가적인 수정 실험(ablation study)을 통해 내족(구조적 변이) 및 교차족(구조 진화) 정보의 조합이 서로 잘 보완하고 강력하게 작용한다는 것을 확인하였습니다. 이는 EvoIF가 진화 정보를 모델링하는 데 있어 효과적이고 강력한 네트워크임을 시사합니다.



### GTCN-G: A Residual Graph-Temporal Fusion Network for Imbalanced Intrusion Detection (Preprint) (https://arxiv.org/abs/2510.07285)
Comments:
          This preprint was submitted to IEEE TrustCom 2025. The accepted version will be published under copyright 2025 IEEE

- **What's New**: 본 연구에서는 지연된 시간 의존성을 모델링하고 데이터 불균형 문제를 해결하기 위해, Gated Temporal Convolutional Network and Graph (GTCN-G)라는 새로운 딥러닝 프레임워크를 제안합니다. 이 모델은 Gated TCN과 Graph Convolutional Network을 결합하여 네트워크 흐름의 계층적 Temporal feature를 추출하고, Graph Attention Network (GAT)를 통해 잔여 학습 메커니즘을 추가하여 원래의 특성 정보를 보존합니다. 이는 희귀 악성 활동(소수 클래스)에 대한 감지 민감도를 향상시키기 위해 필수적입니다.

- **Technical Details**: GTCN-G 프레임워크는 네트워크 흐름에서 시간적 의존성을 포착하기 위한 Gated TCN과 그래프 구조로부터 학습하기 위한 Graph Convolutional Network의 조합으로 구성됩니다. 또한 GAT를 통해 원본 피처 정보의 잔여 연결을 유지하여 분류 작업의 정확도를 높이고, 소수 클래스에 대한 탐지 능력을 강화합니다. 이 구조는 G-TCN 및 GCN 모듈을 병렬로 처리하여 데이터의 표현을 통합하고 최종 분류 결과를 생성합니다.

- **Performance Highlights**: UNSW-NB15 및 ToN-IoT 데이터셋을 사용하여 진행된 실험에서 GTCN-G 모델은 기존의 기준 모델보다 월등히 우수한 성능을 보였습니다. 이 모델은 이진 및 다중 클래스 분류 작업 모두에서 최첨단의 성능을 달성하여, 복잡한 네트워크 트래픽의 탐지 효율성을 크게 향상시켰습니다.



### Dynamic Regret Bounds for Online Omniprediction with Long Term Constraints (https://arxiv.org/abs/2510.07266)
- **What's New**: 이번 연구에서는 동적 후회 경계(dyamic regret bounds)를 보장하는 온라인 전방 예측 문제에 대한 알고리즘을 제안합니다. 이 알고리즘은 복수의 다운스트림 의사결정자가 각자의 유틸리티 함수와 제약 조건을 가진 상황에서 사용할 수 있습니다. 예측자는 모든 다운스트림 의사결정자가 최악의 유틸리티 보장을 얻고 최소한의 제약 위반을 하도록 하는 예측을 생성하는 목표를 가지고 있습니다.

- **Technical Details**: 제안된 알고리즘은 각 에이전트의 후회(regret)를 상쇄하는 동적 후회 보장을 동시 제공합니다. 이 알고리즘은 에이전트들이 자신의 상태를 유지할 필요 없이, 단순히 예측에 의해 정의된 제약 최적화 문제를 해결하여 동작할 수 있게 설계되었습니다. 이는 반복적인 상호작용 초기 단계에서 예측된 값을 바탕으로 진행됩니다.

- **Performance Highlights**: 본 연구에서 도출한 결과는 비상태(state-less) 방식으로 작동하며, 제안된 알고리즘은 강력한 동적 스왑 후회(dyamic swap regret) 경계를 제공합니다. 뿐만 아니라, 이 알고리즘은 다운스트림 의사결정자들이 예측을 바탕으로 행동을 선택하는 방식에 대한 유연성을 제공합니다. 따라서 각각의 의사결정자가 최적의 결과를 도출할 수 있는 조건을 마련합니다.



### Test-Time Graph Search for Goal-Conditioned Reinforcement Learning (https://arxiv.org/abs/2510.07257)
- **What's New**: 이번 논문에서는 오프라인 목표 조건 강화 학습(Goal-Conditioned Reinforcement Learning, GCRL) 에이전트가 사용자 지정 목표를 해결하는 데 어려움을 겪는 문제를 다루고 있습니다. 이를 해결하기 위해 Test-Time Graph Search (TTGS)라는 경량 계획 테크닉을 도입하여, 기존의 정책이 주어진 목표에 도달할 수 있도록 보조하는 새로운 프레임워크를 제안합니다. TTGS는 데이터셋 상태를 그래프로 만들고, 목표에 도달하는 데 필요한 중간 목표를 효과적으로 계획하여 정책이 실행할 수 있도록 돕습니다.

- **Technical Details**: TTGS는 거리 신호를 사용하여 상태 간의 그래프를 구축하고, 해당 그래프에서 최단 경로를 검색하여 정책이 수행할 수 있는 중간 목표의 시퀀스를 생성합니다. 이 방법은 값 기반(value-based) 정책의 경우, 학습된 목표 조건 가치 함수(goal-conditioned value function)에서 파생된 거리를 사용하여, 수작업으로 정의된 메트릭이 필요 없도록 합니다. TTGS는 훈련 과정에서의 변화나 추가적인 감독 없이, 오프라인 데이터셋만으로 작동하며, 테스트 시간에만 계산 비용이 추가됩니다.

- **Performance Highlights**: 논문에서는 OGBench 벤치마크에서 TTGS를 적용하여 다양한 기본 학습자들이 복잡한 이동 과제를 해결하는 성공률을 향상시켰음을 보여주었습니다. 이 방법은 기존의 복잡한 메서드들보다 더 나은 성과를 거두며, 심지어 많은 경우 높은 성능의 복잡한 방법들보다 더 우수한 결과를 나타냅니다. TTGS는 기존의 목표 조건 학습자와 통합되며, 명시적인 장기 검색을 통해 가치 기반 접근법의 잠재력을 극대화합니다.



### Discriminative Feature Feedback with General Teacher Classes (https://arxiv.org/abs/2510.07245)
- **What's New**: 이번 연구에서, DFF(Discriminative Feature Feedback)라는 상호작용 학습 프로토콜의 이론적 특성을 체계적으로 분석합니다. 기존의 학습 프로토콜들과는 달리, DFF는 다양한 형식의 피드백을 통해 학습 속도를 향상시킵니다. 또한, DFF에 대한 첫 번째 포괄적인 연구를 제시하며, 이는 감독 학습(Supervised Learning) 및 온라인 학습(Online Learning)과 비교될 수 있습니다.

- **Technical Details**: DFF 프로토콜은 피드백을 특성 설명의 형식으로 사용하여, 학습자가 예측에 대한 설명으로 예제를 제공하고, 교사가 이를 기반으로 적절한 라벨과 설명을 제공합니다. 연구의 주요 내용 중 하나는 DFF의 최적 실수 경계(mistake bound)를 실현 가능한( realizable) 설정과 비실현 가능한(non-realizable) 설정에서 분석하는 것입니다. 이를 통해 DFFdim이라는 새로운 차원을 정의하고, 이 차원이 실현 가능한 설정에서의 최적 실수 경계를 특성화하는 데 사용됩니다.

- **Performance Highlights**: 연구에서는 DFF가 온라인 학습과는 상당히 다르며, 비실현 가능한 실수 경계를 완전히 특성화할 수 없음을 보여줍니다. 또한, 일반적인 교사 클래스에 대한 실수 상한을 도출하고, 이 상한이 개선될 수 없음을 증명하였습니다. 이러한 결과는 상호작용 학습 문제의 특성과 교사 오류에 대한 허용 한계 간의 관계를 탐구하는 흥미로운 질문을 불러일으킵니다.



### A Broader View of Thompson Sampling (https://arxiv.org/abs/2510.07208)
- **What's New**: 이번 논문에서는 Thompson Sampling의 이해를 돕기 위해 이를 온라인 최적화 알고리즘으로 재구성하는 접근법을 소개합니다. 이 방법을 통해 탐색(exploration)과 활용(exploitation) 간의 균형을 적절하게 맞추는 기저원리를 설명합니다. 특히, 새로운 개념인 'faithful stationarization' 기법을 사용하여 원래 문제를 변형없이 이해할 수 있는 정적 동적 최적화 문제로 변환합니다.

- **Technical Details**: Thompson Sampling의 핵심은 베이esian 설정에서 posterior distribution을 샘플링하여 최적의 치료를 선택하는 것입니다. 논문에서는 시간 불변적인 최적화 목표를 Bellman의 원리를 사용해 연구하고, 이 목표가 어떻게 Thompson Sampling의 구조를 단순화하는지 보여줍니다. 최적화 목표는 두 팔이 있는 bandit 문제를 고려하여 수학적으로 정의되고, 이 문제는 Markov decision process(MDP)로 모델링됩니다.

- **Performance Highlights**: Thompson Sampling은 실제 응용에서 매우 높은 성능을 보이며, 온라인 광고, 추천 시스템, 웹사이트 최적화 등 다양한 분야에서 사용되고 있습니다. 기존의 이론적 연구들은 Thompson Sampling의 장기적 후회(minimal regret)를 달성하는 데 대한 경계를 설정했습니다. 이 논문은 이러한 이론을 넘어서서 얼마나 잘 탐색하고 활용의 균형을 맞추는지에 대한 보다 명확한 원리를 제공합니다.



### Guided by the Experts: Provable Feature Learning Dynamic of Soft-Routed Mixture-of-Experts (https://arxiv.org/abs/2510.07205)
- **What's New**: 이 논문은 Mixture-of-Experts (MoE) 아키텍처의 이론적 이해를 확장하고 있습니다. 특히, 비선형 라우터와 전문가를 가진 MoE 모델의 공동 훈련에 대한 수렴 보장을 제시합니다. 이를 통해 효율적인 학생-교사(student-teacher) 프레임워크 내에서 MoE의 훈련 동역학을 심층적으로 분석합니다.

- **Technical Details**: 연구는 적당한 과매개변수화(over-parameterization)를 통해 학생 네트워크가 피쳐 학습(feature learning) 단계를 경험하도록 이끌며, 이 과정에서 라우터의 학습 과정은 전문가들에 의해 ``유도''됩니다. 또한, 후처리(pruning)을 통해 불필요한 신경망(neurons)을 제거한 후, 수렴성이 보장된 미세 조정(fine-tuning) 과정을 통해 글로벌 최적(global optimality)에 도달할 수 있음을 증명합니다.

- **Performance Highlights**: 이 논문은 MoE 아키텍처의 최적화 환경을 이해하는 데 있어 새로운 통찰력을 제공하며, 과거의 연구와는 차별화된 분석을 통해 MoE의 훈련 방식을 더욱 효과적으로 개선할 수 있는 방법을 제시합니다. 이러한 접근 방식은 MoE가 AI 시스템의 중요한 구성 요소로 자리잡는 데 기여할 것으로 보입니다.



### An in-depth look at approximation via deep and narrow neural networks (https://arxiv.org/abs/2510.07202)
Comments:
          11 pages

- **What's New**: 이 논문은 Hanin과 Sellke의 연구를 바탕으로 깊이가 제한된 신경망의 효용성을 다시 탐구하고 있습니다. 저자들은 주어진 가중치 조건에서 연속 함수의 공간 내에서 깊고 넓은 신경망의 밀도에 대한 새로운 관점을 제시하고 있습니다. 또한, 신경망의 깊이가 증가함에 따라 근사화 질이 어떻게 변화하는지를 밝히고자 합니다.

- **Technical Details**: 신경망의 폭(w)과 깊이(d)에 따른 함수 근사화의 성질을 더 깊이 분석하고 있습니다. 저자들은 깊이 d가 n과 같거나 n+1일 때의 가우스 함수를 신경망을 통해 근사화하는 방법을 설명합니다. 특히, 저자들은 신경망 훈련 과정에서 발생하는 'dead neurons'(죽은 뉴런) 현상이 근사화 질에 미치는 영향을 분석합니다.

- **Performance Highlights**: 저자들은 이론적으로 제시된 결과들이 실제로 훈련된 신경망에서도 적용되는지를 검토합니다. 특히, 깊이 d가 n+1일 때 신경망이 연속 함수의 밀대를 형성할 수 있는지에 대한 질문을 다룹니다. 이 연구는 깊이와 폭이 신경망의 성능에 미치는 영향에 대한 통찰력을 제공합니다.



### Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples (https://arxiv.org/abs/2510.07192)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLMs)의 훈련 데이터에 악성 문서를 주입하는 방식으로 모델의 안전성을 저하시킬 수 있는 '포이즈닝 공격'의 새로운 실험을 진행했습니다. 특히, 데이터 세트의 크기와 관계없이 고정된 수의 포이즈닝 문서만으로도 공격이 성공한다는 것을 처음으로 입증했습니다. 이를 통해 250개의 악성 문서로도 600M부터 13B 매개변수 모델까지 모두 부정적인 영향을 미칠 수 있음을 보여주었습니다.

- **Technical Details**: 우리는 Chinchilla-최적 데이터 세트에서 600M에서 13B 매개변수 모델을 대상으로 포이즈닝 실험을 수행했습니다. 연구를 통해 포이즈닝 비율이 아닌 고정된 문서 수가 공격의 성공에 결정적인 역할을 한다는 것을 확인했습니다. 또한, 각 모델은 동일한 수의 악성 샘플을 적용받아 훈련되었고, 포이즈닝 저항력을 평가하기 위해 다양한 설정에서 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, 훈련된 모든 모델에서 트리거 문구가 포함됐을 때 무작위 텍스트(gibberish)를 생성하는 공격에 성공했습니다. 특정 포이즈닝 문서 수에 따라 훈련의 끝에서 평균 perplexity가 200을 초과하며, 이는 성공적인 Backdoor 공격을 나타냅니다. 이러한 결과는 향후 LLMs의 훈련과 방어 메커니즘 개발에 있어 중요한 경고 신호로 작용할 것입니다.



### Bridged Clustering for Representation Learning: Semi-Supervised Sparse Bridging (https://arxiv.org/abs/2510.07182)
- **What's New**: 우리는 Bridged Clustering(브리지 클러스터링)을 제안합니다. 이는 비수반 입력 $X$와 출력 $Y$ 데이터셋으로부터 예측자를 학습하는 반지도 학습(SSL) 프레임워크입니다. 이 방법은 먼저 $X$와 $Y$를 독립적으로 클러스터링한 후, 소수의 쌍 데이터를 활용하여 클러스터 간의 희소하고 해석 가능한 브리지를 학습합니다. 기존의 SSL 방법과 달리, Bridged Clustering은 출력만 있는 데이터를 명시적으로 활용하는 것이 특징입니다.

- **Technical Details**: Bridged Clustering 방식은 두 공간 각각을 독립적으로 클러스터링하고, 소량의 쌍 데이터로 클러스터 수준의 브리지를 학습하는 간단한 방법입니다. 새로운 입력 $x$는 가장 가까운 입력 클러스터에 할당되고, 해당 출력 클러스터의 중심점이 예측 $	ilde{y}$로 반환됩니다. 이 프레임워크는 모델에 구애 받지 않으며, 모든 임베딩 모델(예: ResNet, BERT)과 표준 클러스터링 알고리즘(예: k-means, spectral)을 사용할 수 있습니다.

- **Performance Highlights**: Bridged Clustering은 다양한 도메인에서 효과성을 입증하며, 저수준 감독 설정에서도 높은 레이블 효율성을 유지하면서 SOTA(State-of-the-Art) 방법들과 경쟁력을 보입니다. 이 방법은 디지털 생태계의 많은 비수반 데이터를 활용할 수 있는 새로운 연구 방향을 제시하며, 높은 해석성과 효율성을 갖추고 있습니다. 연구진들은 코드와 데이터를 GitHub에 공개하여, 누구나 이 방법을 활용할 수 있도록 지원하고 있습니다.



### ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL (https://arxiv.org/abs/2510.07151)
Comments:
          22 pages, 7 figures

- **What's New**: 이번 연구에서는 ELMUR(External Layer Memory with Update/Rewrite)라는 새로운 트랜스포머 아키텍처를 제안합니다. 이 아키텍처는 각 레이어가 구조화된 외부 메모리로 보강되며, 이를 통해 장기적인 의존성 문제를 해결합니다. ELMUR는 메모리 임베딩을 유지하고, 양방향 크로스-어텐션을 통해 상호작용하며, LRU(Least Recently Used) 메모리 모듈을 사용해 메모리를 갱신합니다.

- **Technical Details**: ELMUR는 장기 메모리를 효율적으로 저장 및 검색할 수 있는 메커니즘을 제공합니다. 레이어에 로컬 메모리 임베딩을 유지하면서 메모리에 대한 읽기/쓰기 상호작용을 양방향으로 처리합니다. 또한, LRU 업데이트 블록을 통해 메모리를 대체하거나 컨벡스 블렌딩으로 새롭게 갱신하여 안정성과 적응성을 균형있게 유지합니다.

- **Performance Highlights**: ELMUR는 T-Maze 작업에서 100%의 성공률을 거두었으며, MIKASA-Robo의 스팟 보상 조작 작업에서 성능을 거의 두 배로 향상시켰습니다. POPGym 벤치마크에서도 48개의 작업 중 24개에서 최상위 점수를 기록하는 등 부분 관찰 하에서 robust한 일반화 성능을 보여주었습니다.



### A Multi-Agent Framework for Stateful Inference-Time Search (https://arxiv.org/abs/2510.07147)
- **What's New**: 이 연구에서는 상태 기반의 다중 에이전트 진화 탐색(stateful multi-agent evolutionary search)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 지속적인 추론 상태(persistent inference-time state), 적대적 변이(adversarial mutation), 진화적 보존(evolutionary preservation)을 결합하여 이전의 비상태(stateless) 접근 방식에서 벗어났습니다. 그 결과, 복잡한 테스트 케이스 생성을 통해 더욱 견고한 유닛 테스트를 자동으로 생성할 수 있게 되었습니다.

- **Technical Details**: 제안된 시스템은 여러 LLM 호출을 통해 후보 엣지 케이스를 제안하는 액터(actor), 환경을 변이시켜 견고성 갭을 드러내는 적대자(adversary), 진화 검색에 사용되는 보상을 부여하는 비평가(critic)로 구성됩니다. 각 단계에서 상태 정보를 유지하여 이전 단계의 피드백을 활용하고, 이는 선형 탐색을 넘어선 구조적 문제 해결을 가능하게 합니다. 더불어, 액터는 대형 언어 모델(LLM)의 맥락 학습을 통해 지속적인 상태를 기반으로 후보를 생성하며, 이는 전통적인 기법보다 높은 샘플 효율성을 제공합니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 HumanEval 및 TestGenEvalMini와 같은 유닛 테스트 벤치마크에서 비상태 단일 단계 기준선에 비해 상당히 높은 커버리지(coverage)를 달성했습니다. 세 가지 다양한 LLM 모델인 Llama, Gemma, GPT를 활용하여 유연한 테스트 케이스 생성을 보여 주었으며, 향상된 커버리지와 견고성을 통해 새로운 코드베이스에 대한 적응 능력이 뛰어남을 입증했습니다. 이러한 결과는 지속적인 추론 상태와 진화적 탐색이 유닛 테스트 생성에 실질적인 개선 효과를 줄 수 있음을 시사합니다.



### DPMM-CFL: Clustered Federated Learning via Dirichlet Process Mixture Model Nonparametric Clustering (https://arxiv.org/abs/2510.07132)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 논문에서는 클러스터링된 연합 학습(Clustered Federated Learning, CFL)에서 클러스터의 수를 미리 정의할 필요 없이 클러스터의 분포를 추론할 수 있는 DPMM-CFL 알고리즘을 제안합니다. 이는 베이지안 비모수적(Nonparametric Bayesian) 프레임워크를 통해 독립적인 클라이언트 그룹 수를 추론하게 해주며, 연합 목표를 최적화할 수 있습니다. 이는 임시로 클러스터의 수를 추론할 수 있는 가능성을 제공하여 실제 환경에서도 적용할 수 있습니다.

- **Technical Details**: DPMM-CFL 알고리즘은 클라이언트의 로컬 업데이트에 기반하여 글로벌 클러스터 모델을 학습합니다. 클라이언트들은 자신에게 배정된 클러스터에 따라 연합된 업데이트를 수행하고, 이 외에도 클러스터링을 비모수적 베이지안 추론 문제로 다룹니다. 이 과정에서 Dirichlet Process prior가 사용되어 클러스터의 수를 데이터로부터 직접 추론할 수 있게 해줍니다.

- **Performance Highlights**: 본 논문은 여러 벤치마크 데이터셋에서 제안된 알고리즘을 검증하였으며, 실험 결과는 DPMM-CFL이 효과적으로 클러스터링된 연합 학습을 수행할 수 있음을 보여주었습니다. 결과적으로 해당 방법은 클러스터 수를 동적으로 조정하면서도 각 클러스터의 최적화된 연합 학습 목표를 달성하였음을 입증하였습니다.



### Non-Asymptotic Analysis of Efficiency in Conformalized Regression (https://arxiv.org/abs/2510.07093)
- **What's New**: 이 논문에서는 conformal prediction(콘포말 예측)의 비점근적 효율성에 대한 새로운 경계를 정립합니다. 기존 연구는 miscoverage level(미커버리지 수준) α를 상수로 간주했지만, 본 연구는 데이터 분포에 대한 온건한 가정 하에 예측 집합의 길이의 편차에 대한 경계를 제시합니다. 이 결과는 훈련 집합 크기 n, 보정 집합 크기 m, 미커버리지 수준 α의 공동 의존성을 포착하며, 예측 집합 길이를 조절하기 위한 데이터 할당에 대한 지침을 제공합니다.

- **Technical Details**: 연구팀은 split conformal prediction(분할 콘포말 예측)의 효율성을 분석하며, conformalized median regression(콘포말 중간 회귀)과 conformalized quantile regression(콘포말 분위수 회귀)을 최우선으로 다루었습니다. 이 방법들은 비대칭적 예측 구간을 제공하며, 서로 다른 α 레벨에서의 수렴 속도의 상한 경계를 처음으로 설정하여 이론적 근거를 제공합니다. 연구 결과는 샘플 크기와 미커버리지 수준이 예측 집합의 길이에 미치는 영향을 구체적으로 설명합니다.

- **Performance Highlights**: 실험 결과는 이론적 발견과 일치하여, 훈련 데이터와 보정 데이터 간의 적절한 배분이 예측 집합의 과도한 길이를 제어하는 데 유용함을 나타냅니다. 특히, CQR(콘포말 분위수 회귀)에서는 예측 집합 길이의 기대 편차에 대한 상한 경계를 정립하고, CMR(콘포말 중간 회귀)에서는 균일한 구간을 생성하여 비대칭성을 자연스럽게 처리합니다. 본 연구의 이론적 통찰은 다양한 최적화 알고리즘에도 확장될 수 있습니다.



### Generative World Modelling for Humanoids: 1X World Model Challenge Technical Repor (https://arxiv.org/abs/2510.07092)
Comments:
          6 pages, 3 figures, 1X world model challenge technical report

- **What's New**: 1X World Model Challenge는 로봇과 인공지능의 새로운 벤치마크로, 인간과의 상호작용을 다루고 있다. 이 챌린지는 미래 이미지 프레임 예측에 중점을 둔 샘플링 트랙과 미래 이산 잠재 코드 예측에 중점을 둔 압축 트랙으로 구성된다. 연구팀은 두 트랙 모두에서 최우수 성적을 달성하여 1위를 기록했다.

- **Technical Details**: 샘플링 트랙에서는 예측 프레임을 생성하기 위해 Wan-2.2 TI2V-5B를 수정하여 비디오 상태로 조건화하였다. 압축 트랙에서는 처음부터 Spatio-Temporal Transformer 모델을 훈련시켰다. 각 모델은 예측 프레임의 품질을 PSNR(Peak Signal-to-Noise Ratio) 지표를 통해 평가하였다.

- **Performance Highlights**: 샘플링 작업에서는 PSNR 23.0 dB을 달성하였고, 압축 작업에서는 Top-500 Cross-Entropy 6.6386을 기록하였다. 이런 뛰어난 성능은 로봇의 상태와 행동을 효과적으로 반영한 결과로, 향후 세계 모델을 활용한 로봇 연구에 큰 기여를 할 것이 기대된다.



### Non-Stationary Online Structured Prediction with Surrogate Losses (https://arxiv.org/abs/2510.07086)
- **What's New**: 이번 논문에서는 온라인 구조적 예측(online structured prediction)에서의 손실 경계를 제시합니다. 특히, 비정상 환경(non-stationary environments)에서 고정된 추정기(fixed estimator)에 의한 손실이 시간과 함께 선형적으로 증가하는 문제를 해결하고자 합니다. 제안된 방법은 누적 목표 손실(cumulative target loss)에 대한 새로운 경계를 제시하며, 이는 시간 지평선에 독립적인 차별적인 보장을 제공합니다.

- **Technical Details**: 연구에서 제시된 경계는 누적 대체 손실(cumulative surrogate loss) 및 경로 길이(path length)에 따라 다릅니다. 핵심 아이디어는 온라인 그래디언트 하강법(OGD)의 동적 후회(bound) 개념을 활용하는 것입니다. 또한, OGD에 대한 새로운 Polyak 스타일의 학습률을 도입하여 목표 손실 보장을 체계적으로 제공합니다.

- **Performance Highlights**: 제안된 방법은 비정상 환경에서도 강력한 보증을 제공하며, 실험적으로도 우수한 성능을 기록하고 있습니다. 이 연구는 컨볼루션 Fenchel-Young 손실(convolutional Fenchel-Young loss)로의 확장을 통해 더 넓은 문제 클래스에 적용 가능성을 보여줍니다. 마지막으로, 제시된 경계의 의존성이 타이트하다는 하한도 입증되었습니다.



### HTMformer: Hybrid Time and Multivariate Transformer for Time Series Forecasting (https://arxiv.org/abs/2510.07084)
- **What's New**: 이번 논문은 시계열 예측에서 Transformer 기반 방법의 한계를 극복하기 위한 새로운 접근 방식인 Hybrid Temporal and Multivariate Embeddings (HTME) 추출기를 제안합니다. HTME는 가벼운 시간 특성 추출 모듈과 다변량 특성 추출 모듈을 통합하여 정보의 풍부한 표현을 제공합니다. 이를 통해 기존 모델에서 발생하는 계산 비용을 줄이고, 성능 향상을 동시에 달성할 수 있는 HTMformer라는 경량 예측 모델을 개발하였습니다.

- **Technical Details**: HTMformer는 시계열 예측을 위한 일반적인 모델로, 인코더 전용 아키텍처를 채택하고 HTME 추출기를 포함합니다. HTME 추출기는 시계열 데이터의 시간 및 다변량 특성을 효과적으로 포착하도록 설계되었습니다. 또한, RevIN 정규화 기법을 사용하여 모델의 학습 및 일반화 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, HTMformer는 여덟 개의 실제 데이터셋에서 기존의 예측 모델들과 비교했을 때 정확도와 효율성 모두에서 뛰어난 성능을 보였습니다. 이는 다변량 특성과 시간적 특성을 통합하여 예측의 의미적 풍부함을 향상시켰기 때문입니다. 특히, HTMformer는 다양한 기준선 모델에 대해 지속적으로 최첨단 성능을 달성하였습니다.



### Blind Construction of Angular Power Maps in Massive MIMO Networks (https://arxiv.org/abs/2510.07071)
- **What's New**: 이번 논문은 대규모 다중입력 다중출력(MIMO) 네트워크에서의 채널 상태 정보(CSI) 취득 문제를 탐구합니다. 기존의 라디오 맵 구축 방법은 위치 라벨이 부착된 CSI 데이터가 필요하고, 이는 실전에서 많은 어려움을 겪습니다. 이 연구는 비지도 학습 방식으로 위치 정보 없이도 각도를 기반으로 한 전력 맵을 구축할 수 있는 가능성을 제시합니다.

- **Technical Details**: 논문에서는 사용자 이동과 massive MIMO 채널의 CSI 진화를 연결하기 위한 숨겨진 마르코프 모델(Hidden Markov Model; HMM)을 제안하였습니다. 이를 통해 이동 사용자 위치의 추정이 가능해지며, 실제 네트워크에서 수집한 RSRP 데이터를 통해 평균 18미터의 추적 오차를 기록합니다. 또한, Cramer-Rao Lower Bound (CRLB)를 통해 이동성과 신호 대 잡음비(SNR)의 영향을 분석합니다.

- **Performance Highlights**: 제안된 방법론은 기존 방법들과 비교했을 때 최저 오류율을 달성합니다. 특히, 시뮬레이션 데이터셋에서는 7미터의 측위 오차를 달성하며, 이는 네트워크의 측정 데이터가 부족할 때에도 가능합니다. 연구 결과는 대규모 MIMO 시스템의 신뢰성을 개선하는 데 중요한 기여를 한다고 평가됩니다.



### Introspection in Learned Semantic Scene Graph Localisation (https://arxiv.org/abs/2510.07053)
Comments:
          IEEE IROS 2025 Workshop FAST

- **What's New**: 이 연구는 학습된 자기 지도 대비(contrastive) 의미 지역화(semantic localisation) 프레임워크에서 의미가 지역화 성능과 강건성에 미치는 영향을 조사합니다. 모델은 원본 지도(original map)와 편향된 지도(perturbed map)에서 훈련을 받은 후, 환경 소음에서 필터링을 수행하고, 일상적인 혼잡보다 독특한 랜드마크를 우선시하는지에 대한 철저한 자기 성찰(post-hoc introspection) 분석을 수행합니다. 다양한 해석 가능성 방법(interpretability methods)을 검증하고, 통합 그래디언트(integrated gradients)와 주의 가중치(attention weights)가 가장 신뢰할 수 있는 모델 행동 탐색 기법으로 자리 잡았습니다.

- **Technical Details**: 연구는 하이레벨(higher-level) 의미 정보가 로컬라이제이션에 어떻게 활용될 수 있는지를 보여주며, 인간과 비슷하게 환경 소음을 필터링하는 방법을 모사합니다. 3D 씬 그래프(scene graph)를 활용하여 공간 개념(노드)과 관계(엣지)를 모델링합니다. 데이터셋으로는 포토 리얼리스틱(photorealistic) 환경과 완전한 메트릭-의미 주석(metric-semantic annotations)을 포함한 uHumans2 데이터셋을 사용합니다.

- **Performance Highlights**: 연구 결과는 모델이 시각적 및 구조적 변형이 존재하는 상황에서도 노이즈에 강하고 의미적으로 중요한 관계를 학습함으로써 설명 가능한 지역화를 가능하게 함을 나타냅니다. 클래스 제거에 따른 성능 저하 분석과 속성 기여 변동 분석을 통해, 자주 등장하는 객체는 다운 가중치 처리되고, 희귀한 랜드마크가 지역 지정 해소에 중요한 역할을 담당하는 것으로 나타났습니다. 통합 그래디언트와 주의 가중치의 신뢰성 분석을 수행하였고, 이들은 강력한 객체 중요도 속성 신호를 제공합니다.



### Enhancing Speech Emotion Recognition via Fine-Tuning Pre-Trained Models and Hyper-Parameter Optimisation (https://arxiv.org/abs/2510.07052)
- **What's New**: 본 논문에서는 사전 훈련된 표현(pre-trained representations)과 자동 하이퍼파라미터 최적화(automated hyperparameter optimisation, HPO)를 결합한 음성 감정 인식(speech emotion recognition, SER) 워크플로우를 제안합니다. 이를 통해 저비용 CPU만을 사용하는 파이프라인에서도 최근 SER 시스템과 경쟁할 수 있는 성능을 달성하는 방법을 탐구합니다. 이는 기존의 SER 접근 방식에서 자주 사용되던 GPU 자원 의존성을 줄이고 고급 전문가 개입을 최소화하는 데 기여합니다.

- **Technical Details**: 제안된 워크플로우는 훈련 및 검증 데이터 세트, 사전 훈련된 음성 모델, k차원의 하이퍼파라미터 벡터, HPO 알고리즘을 입력으로 사용합니다. 본 연구에서는 두 가지 HPO 기법인 Gaussian Process Bayesian Optimisation (GP-BO)와 Tree-structured Parzen Estimators (TPE)를 비교하며, 각 방법은 사전 정의된 네 차원의 탐색 공간 내에서 평가됩니다. 성능은 Balaced Class Accuracy (BCA)로 측정되며, 최적의 구성은 이 메트릭을 기반으로 선택됩니다.

- **Performance Highlights**: 실험 결과 GP-BO는 11분 만에 0.96 BCA를 달성하고, TPE는 15분에 0.97 BCA를 기록했습니다. 반면, 전통적인 그리드 서치는 143회 시도와 1680분이 소요되어야 0.9 BCA를 초과하였으며, AutoSpeech 2020의 베이스라인은 30분 동안 GPU에서 0.85를 보고했습니다. 또한, EmoDB에서 훈련된 HPO 조정 모델이 CREMA-D와 RAVDESS의 제로샷(zero-shot) 정확도를 각각 0.25와 0.26 향상시켜 교차 언어 일반화에서도 우수한 성능을 보였습니다.



### COMPASS: A Multi-Turn Benchmark for Tool-Mediated Planning & Preference Optimization (https://arxiv.org/abs/2510.07043)
- **What's New**: 이번 연구에서는 실제 환경에서 대규모 언어 모델(Large Language Model, LLM)을 활용하여 사용자와 복잡한 계획 작업에 대해 상호작용할 수 있는 COMPAASS를 소개합니다. COMPASS는 여행 계획을 제약 기반의 선호 최적화 문제로 모델링하여, 에이전트가 엄격한 제약 조건을 충족하면서도 사용자의 소프트 선호를 최적화하는 방식을 평가합니다. 이는 실제 환경에서 기대되는 LLM의 활용 가능성을 높이기 위한 새로운 벤치마크로 자리잡고 있습니다.

- **Technical Details**: COMPASS 벤치마크는 사용자와의 반복적 상호작용을 위한 사용자 시뮬레이터, 호텔, 항공편 및 활동 허가에 대한 실제 데이터베이스, 상업적 예약 플랫폼을 모사하는 도구 생태계로 구성됩니다. 이 환경에서는 예산, 숙박, 일정과 같은 하드 제약 조건을 만족하면서 비용 최소화나 편의시설 극대화와 같은 소프트 선호를 최적화하는 것이 필수적입니다. 두 가지 유형의 최적화 작업이 제공되어, 사용자 요구 사항을 충족하는 동시에 최상의 솔루션을 찾는 두 가지 접근 방식을 조화롭게 평가합니다.

- **Performance Highlights**: 기존 모델들에서 두 가지 중요한 문제를 확인하였습니다: 첫째, acceptable-optimal gap로, 모든 모델이 높은 제약 조건 만족률을 기록했지만 사용자 선호 최적화에서는 만족하지 못하고 있습니다. 둘째, plan-coordination gap으로, 에이전트가 다수의 서비스를 조정해야 하는 복잡한 계획 작업에서 성능이 급격히 저하됩니다. 이는 특히 오픈 소스 모델에서 더욱 두드러지며, COMPASS는 이러한 제약 조건 최적화의 핵심 과제를 진단하고 사용자에 맞춘 AI 에이전트 개발을 안내하는 데 기여합니다.



### Unified Molecule Pre-training with Flexible 2D and 3D Modalities: Single and Paired Modality Integration (https://arxiv.org/abs/2510.07035)
Comments:
          CIKM 2025

- **What's New**: 이번 논문에서는 FlexMol이라는 새로운 분자 사전 학습 프레임워크를 제안합니다. FlexMol은 2D와 3D 방식의 분자 표현을 통합하여, 단일 모달리티 입력을 지원하며, 학습 과정에서 양쪽 모달리티의 정보를 효과적으로 융합합니다. 이 접근 방식은 다양한 분자 특성 예측 태스크에서 우수한 성능을 보여주며, 불완전한 데이터에서도 효과적이임을 입증하였습니다.

- **Technical Details**: FlexMol은 2D 및 3D 분자 데이터를 위한 별도의 모델을 사용하고 매개변수 공유(parameter sharing)를 통해 계산 효율성을 높입니다. 또한, 누락된 모달리티를 생성하기 위한 디코더(decoder)를 활용하여, 학습 중에 양쪽 모달리티가 협력적으로 기여하는 다단계 연속 학습 과정을 구현합니다. 이를 통해 다양한 조합의 데이터로 유연한 멀티모달 학습이 가능합니다.

- **Performance Highlights**: FlexMol은 3.4M의 페어된 샘플과 2M의 단일 모달리티 샘플로 훈련되었으며, 특정 벤치마크 과제에서 10M 이상의 샘플로 사전 훈련된 대규모 분자 모델보다 뛰어난 성능을 달성했습니다. 본 논문은 2D 및 3D 모달리티의 효과적인 정렬과 융합을 통해 경쟁력 있는 성능을 제공하며, 단일 모달리티 입력에서도 멀티모달 학습을 가능하게 합니다.



### Federated Unlearning in the Wild: Rethinking Fairness and Data Discrepancy (https://arxiv.org/abs/2510.07022)
- **What's New**: 이 논문은 데이터 삭제 권리를 시행하기 위한 기계 학습의 새로운 방향인 Federated Unlearning (FU)에 대한 최신 연구 결과를 제시합니다. 기존의 FU 방법론이 정확성에만 집중했던 반면, 본 연구에서는 공정성(fairness)과 데이터 분포 차이를 고려한 새로운 접근법인 Federated Cross-Client-Constrained Unlearning (FedCCCU)을 제안하여 이러한 단점을 극복합니다. 실험 결과에 따르면, 기존의 방법들이 실제 환경에서 효과적이지 않은 반면, 저자들의 제안하는 방법은 일관되게 더 나은 성능을 보여줍니다.

- **Technical Details**: Federated Learning (FL)에서는 개별 클라이언트들이 로컬 데이터를 이용해 모델을 학습하고, 중앙 서버에서 이를 집계하여 글로벌 모델을 형성합니다. 하지만, 데이터 삭제와 같은 기계 학습의 형태인 Unlearning을 수행할 때, 기존 방법들은 모든 클라이언트가 재학습에 참여해야하므로 계산 자원과 협력 비용이 상당히 증가합니다. FedCCCU 방법은 이러한 공정성을 고려하여 각 클라이언트의 고유한 데이터 분포를 반영하여, 비기억 클라이언트의 지식 손실을 최소화하도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, FedCCCU는 실제 데이터의 비균질성(hétérogénéité)과 공정성을 고려했을 때, 기존의 모든 분석된 FU 방법들의 성능에 비해 현저히 우수한 성능을 보여주었습니다. 특히, FedCCCU는 그 과정에서 발생 가능한 추가 비용이나 자원 낭비를 최소화하면서도 유의미한 성과를 내는데 성공했습니다. 이를 통해 저자는 FU 연구의 미래 방향으로 공정성과 데이터 현실성을 함께 고려할 것을 제안합니다.



### Sharpness-Aware Data Generation for Zero-shot Quantization (https://arxiv.org/abs/2510.07018)
- **What's New**: 본 논문은 제로샷 양자화(zero-shot quantization, ZSQ) 문제에서 훈련 데이터 생성 시 모델의 샤프니스(sharpness)를 고려하는 새로운 방법론을 제안합니다. 기존 ZSQ 기법들은 생성된 데이터와 모델의 샤프니스 사이의 관계를 고려하지 않았던 반면, 본 연구에서는 샤프니스 최소화가 훈련 데이터의 재구성 손실 그래디언트(gradient) 상호 일치화를 최대화하는 방식으로 이루어질 수 있음을 입증하였습니다.

- **Technical Details**: 제안된 방법론에서는 생성된 데이터와 실제 검증 데이터의 재구성 손실 그래디언트 간의 일치성을 극대화하여 샤프니스 최소화를 이룹니다. 제로샷 환경에서 실제 검증 세트가 없을 경우, 생성된 각 샘플과 그 이웃 샘플 간의 그래디언트 일치화를 근사하여 이 문제를 해결합니다. 또한, 샤프니스 인식을 통한 데이터 생성(Sharpness-Aware Data Generation, SADAG)을 통해 저비트 양자화(low-bit quantization) 환경에서 성능 향상을 도모합니다.

- **Performance Highlights**: CIFAR-100과 ImageNet 데이터셋에 대한 실험 결과, 제안된 SADAG 방법은 기존의 최신 ZSQ 방법들보다 우수한 성능을 보였습니다. 본 연구는 샤프니스 감소가 모델의 일반화 능력 향상에 기여하는 것을 보여주며, ZSQ 문제에 대한 새로운 접근법을 제공합니다. 실험 결과는 제안된 방법이 효과적으로 양자화된 모델의 성능을 개선함을 입증하였습니다.



### Spiral Model Technique For Data Science & Machine Learning Lifecyc (https://arxiv.org/abs/2510.06987)
- **What's New**: 이 논문은 현대 비즈니스에서 데이터 분석(Analytics)의 중요성을 강조합니다. 특히 데이터 과학 생명주기(Data Science Lifecycles)를 기업의 문화에 맞게 조정하여 생산성을 높이고 경쟁력을 향상시키는 방법을 제안합니다. 새로운 기술인 스파이럴 기법(Spiral Technique)을 도입하여 명확한 목표가 있는 비즈니스 문제에 데이터 과학 생명주기를 통합하는 방안을 다룹니다.

- **Technical Details**: 스파이럴 기법은 비즈니스 프로세스에 대한 유연성(versatility), 민첩성(agility) 및 반복적 접근(iterative approach)을 강조합니다. 전통적인 데이터 과학 생명주기는 선형(linear) 또는 순환(cyclical) 모델로, 주기가 끝난 후 다시 시작할 수 있는 구조로 나타납니다. 이를 통해 데이터 의존 프로젝트에서의 시작과 종료에 관한 기여 요소를 살펴봅니다.

- **Performance Highlights**: 이 새로운 접근 방식은 데이터 과학 프로젝트가 명확한 목표를 갖고 있을 때 더욱 효과적이라는 것을 보여줍니다. 비즈니스 문제 해결을 위한 신속한 반복과 조정이 가능해지므로 기업들이 환경 변화에 더 빨리 적응할 수 있도록 합니다. 이로 인해 기업의 전반적인 경쟁력이 향상되는 결과를 가져올 수 있습니다.



### Revisiting Mixout: An Overlooked Path to Robust Finetuning (https://arxiv.org/abs/2510.06982)
- **What's New**: 이 논문에서는 Mixout이라는 스토캐스틱 정규화 기법을 재조명하여, 사전 훈련된 가중치와 미세 조정된 가중치를 주기적으로 교체하는 방법으로 모델의 강건성을 개선하는 새로운 접근법을 제안합니다. GMixout은 고정된 앵커를 지수 이동 평균(Exponential Moving Average) 스냅샷으로 대체하고, 마스킹 주기를 조정하는 하이퍼파라미터를 도입함으로써 미세 조정된 모델의 성능을 최적화합니다.

- **Technical Details**: GMixout의 구현은 드문 커널(sparse kernel) 기법을 사용하여, 훈련 과정에서 오직 소수의 파라미터만 업데이트하여 소비자 등급의 GPU에서 훈련 가능하도록 합니다. 또한, 이 방법은 특정 하이퍼파라미터에 의해 조절될 수 있는 재샘플링 주기를 포함하여, 무작위 하위 네트워크(subnetwork)의 다양성을 증가시키는 동시에 고정된 가중치에 의존하는 문제를 해결합니다.

- **Performance Highlights**: GMixout은 다양한 벤치마크(예: ImageNet, DomainNet, CIFAR100-C)에서 성능을 순차적으로 개선하여 제로샷(Zero-shot) 성능을 초과하는 결과를 보였습니다. 특히, 기존 모델 소프(Model Soups)와 강력한 파라미터 효율적 미세 조정 방법을 초월하는 강인성을 보여주어, 도메인 변화(distribution shift)에서도 우수한 성능을 발휘합니다.



### High-Rate Mixout: Revisiting Mixout for Robust Domain Generalization (https://arxiv.org/abs/2510.06955)
Comments:
          WACV 2026: Winter Conference on Applications of Computer Vision 2026

- **What's New**: 이 논문에서는 Dropout 대신에 Mixout이라는 새로운 확률적 정규화 기법을 도입하여 도메인 일반화 (domain generalization) 문제를 해결하고자 합니다. Mixout은 훈련 과정에서 일부 미세 조정된 가중치를 원래의 미리 훈련된 가중치로 확률적으로 교체하여 과적합 (overfitting)을 방지합니다. 이 방법은 높은 마스킹 확률과 함께 사용되어 최적의 성능을 발휘하며, 실험 결과 이를 통해 앙상블 방식에 근접한 성능을 보이는 동시에 훈련 비용을 크게 줄일 수 있음을 보여줍니다.

- **Technical Details**: Mixout은 ViT (Vision Transformers)와 ResNet 아키텍처에 대해 각각 0.9 및 0.8의 높은 마스킹 확률을 요구합니다. 이는 훈련 과정에서 미리 훈련된 가중치와 잠금 상태의 가중치를 동적으로 교체하여 데이터의 다양성을 탐색합니다. Mixout을 사용함으로써 45%의 그래디언트 계산 비용 절감과 90%의 메모리 사용량 감소를 달성할 수 있으며, 이는 고성능 도메인 일반화에 기여합니다.

- **Performance Highlights**: 실험을 통해 PACS, VLCS, OfficeHome, TerraIncognita, DomainNet과 같은 다섯 가지 도메인 일반화 벤치마크에서 Mixout을 사용한 고마스킹(Mixout with high masking probability) 방법이 앙상블 기반 접근법과 동등한 도메인 밖 정확도를 기록했습니다. Mixout 방법은 여러 모델을 훈련하고 저장할 필요가 없기에 매우 효율적입니다. 이러한 결과를 통해 Mixout은 도메인 간 성능을 향상시키는 동시에 비용 효율성을 극대화하는 가능성을 가지고 있음을 입증했습니다.



### From Condensation to Rank Collapse: A Two-Stage Analysis of Transformer Training Dynamics (https://arxiv.org/abs/2510.06954)
- **What's New**: 이 논문에서는 Transformer 기반 모델의 훈련 동역학을 소규모 초기화(small initialization) 조건 하에서 체계적으로 분석하고 있다. 기존 연구들이 특정 과제에 국한되어 있었다면, 본 연구는 이론적인 분석을 통해 Transformer의 훈련 동역학을 독립적으로 이해하고자 한다. 이를 위해 gradient flow 분석 프레임워크를 활용하여, 두 가지 뚜렷한 훈련 단계를 정의하고 이를 통해 Transformer 최적화의 이론적 기초를 마련하고 있다.

- **Technical Details**: 첫 번째 단계에서는 비대칭적인 가중치 변동이 초기화로부터 발생하여, 파라미터 매트릭스에서 비퇴화적 그래디언트 동역학을 유지함으로써 소규모 초기화 영역을 벗어날 수 있도록 돕는다. 이후 두 번째 단계에서는 이전에 정적인 key-query 매트릭스가 훈련에 적극적으로 참여하며 정규화된 매트릭스의 점근적(rank collapse) 붕괴를 이끈다. 이 두 단계의 다이나믹스는 전통적인 방향 수렴 결과를 일반화하고 데이터에 대한 일반화 능력을 효과적으로 이끌어내는 implicit regularization(암묵적 정규화)의 중요성을 강조한다.

- **Performance Highlights**: 본 연구에서는 실험을 통해 이론적 예측과 가설을 검증하였다. 실험 결과, 소규모 초기화 조건 하에서 관찰된 두 단계 동역학이 정량적으로 Condensation 및 key-query 매트릭스의 점근적 붕괴를 확인하였다. 이러한 결과는 Transformer 모델이 엄청난 용량에도 불구하고 과적합(overfitting)을 피하고 탁월한 성능을 달성할 수 있는 방법을 제공한다.



### Grouped Differential Attention (https://arxiv.org/abs/2510.06949)
- **What's New**: 이번 논문에서는 Grouped Differential Attention (GDA)라는 새로운 접근 방식을 제안합니다. GDA는 신호 보존 그룹과 노이즈 제어 그룹 간의 비대칭 헤드 할당을 도입하여 기존의 대칭 할당 방식을 개선합니다. 이를 통해 신호 추출을 위한 더 많은 헤드를 배분하고, 노이즈 제어 그룹은 더 적은 용량으로 운영되며, controlled repetition을 통해 안정성을 유지합니다.

- **Technical Details**: GDA는 Differential Attention의 한계를 극복하기 위해 다수의 헤드를 비율 기반으로 신호 보존 그룹에 할당하여 신호 충실도를 개선합니다. 이 과정에서 사용되는 주의 메커니즘은 두 개의 상보적인 주의 지도(attention maps)를 생성하여 신호와 노이즈를 분리한 후 이를 결합합니다. 그 결과, 필요한 computational overhead를 최소화하면서도 강력한 신호 추출을 가능하게 합니다.

- **Performance Highlights**: 대규모 사전 학습 및 지속적 학습 실험을 통해, GDA는 일반화 성능과 안정성에서 기존의 대칭 설계 모델 대비 현저히 향상된 결과를 보여주었습니다. 특히, 신호 보존과 노이즈 제어 헤드 간의 적절한 비율(예: 3:1 또는 4:1)이 효과적인 성능 향상을 이끈다는 것을 발견했습니다. 이 연구는 자원 효율적인 Transformer 아키텍처 설계를 위한 실용적인 경로를 제시합니다.



### Fisher Information, Training and Bias in Fourier Regression Models (https://arxiv.org/abs/2510.06945)
- **What's New**: 이 논문은 양자 기계 학습(Quantum Machine Learning, QML) 및 양자 신경망(Quantum Neural Networks, QNNs)에 대한 관심이 커짐에 따라, 피셔 정보 행렬(Fisher Information Matrix, FIM)을 기반으로 한 새로운 평가 지표가 QNN의 훈련 및 예측 성능을 예측하는 데 효과적임을 연구합니다. 이 연구에서는 광범위한 QNN 클래스를 푸리(Fourier) 모델와 동등하게 보고, 유효 차원(effective dimension)과 주어진 작업에 대한 편향(bias)이 훈련 및 성능에 미치는 영향을 조사합니다.

- **Technical Details**: 연구에서 사용된 유효 차원(Effective Dimension, ED) 개념은 FIM을 기반으로 하며, 모델이 자신의 모든 자유도를 효과적으로 탐색할 수 있는 능력을 측정하는 척도로 소개됩니다. 논문에서는 ED와 모델의 훈련 가능성 사이의 관계를 분석하고, 특히 완전히 비편향인 모델의 경우 높은 ED가 더 나은 훈련 성능을 가져오기 쉽고, 편향된 모델의 경우 낮은 ED가 훈련에 유리하다는 것을 보입니다. 또한 QNN와 푸리 모델 사이의 동등성을 활용해 FIM의 해석적 표현을 도출하고, 이를 통해 ED를 조절할 수 있는 모델 구축이 가능함을 보여줍니다.

- **Performance Highlights**: 결과적으로 비편향 모델의 경우 높은 ED가 훈련 성능을 향상시키고, 편향 모델의 경우 낮은 ED가 더 나은 성과를 가져온다는 사실이 정량적으로 확인되었습니다. 이를 통해 모델의 훈련법과 성능 개선을 위한 새로운 관점을 제시하고, QNN 모델 분석을 위한 텐서 네트워크(tensor network) 표현 기법 또한 도입하였습니다. 이러한 발견은 기계 학습 커뮤니티에 유익한 통찰력을 제공하며, 향후 연구 방향에 대한 더 많은 가능성을 탐구할 수 있는 기초를 마련합니다.



### Revisiting Node Affinity Prediction in Temporal Graphs (https://arxiv.org/abs/2510.06940)
Comments:
          preprint

- **What's New**: 본 논문에서는 Node Affinity 예측을 위한 새로운 모델인 NAViS(Node Affinity prediction with Virtual State)를 소개합니다. NAViS는 전통적인 Temporal Graph Neural Networks(TGNNs)의 한계를 극복하기 위해 설계되었으며, 간단한 예측 모델들이 더 나은 성능을 보이는 이유를 분석합니다. NAViS는 가상의 글로벌 상태를 유지하여 과거의 노드 상호작용을 효과적으로 반영함으로써 향후 노드 간의 애착도를 예측합니다.

- **Technical Details**: NAViS는 과거의 노드 애착도를 선형 상태 공간 모델(SSMs)와 연관지어 구성됩니다. 이 모델은 각 노드의 상태와 동적으로 변화하는 그래프 구조에 따라 공진행하는 가상의 글로벌 상태를 유지합니다. 또한, 기존의 교차 엔트로피 손실(loss) 대신 순위 기반 목적 함수를 제안하여 예측 결과의 정렬을 개선하고 최적화를 지원합니다.

- **Performance Highlights**: 실험 결과, NAViS는 Temporal Graph Benchmark(TGB) 및 기타 데이터셋에서 최신 TGNN 모델들과 비교해 일관된 성능 개선을 보여줍니다. 특히, NAViS는 간단한 예측 기법들에 비해 경쟁력 있는 성능을 발휘하여 실제 애착도 예측 문제에 적합하다는 것을 입증합니다. 이는 모델의 유도 편향과 훈련 목표를 일치시키는 것이 예측 성능에 미치는 중요성을 강조합니다.



### DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning (https://arxiv.org/abs/2510.06913)
- **What's New**: 이번 연구에서 제안하는 DecompGAIL(Decomposed Multi-agent Generative Adversarial Imitation Learning)은 자율 주행 시스템과 교통 시뮬레이션의 주요 문제인 불안정성을 해결하기 위해 설계되었습니다. 연구팀은 다중 에이전트 환경에서의 상호작용이 비효율적으로 평가되는 원인을 규명했으며, 이로 인해 학습된 정책의 안정성이 저하되는 문제점을 지적하였습니다. DecompGAIL은 ego-맵(e.g. map)과 ego-이웃(e.g. neighbor) 관련성을 명확히 분리하여 비정상적인 상호작용을 필터링함으로써, 현실적인 동작을 보장합니다.

- **Technical Details**: DecompGAIL은 전통적인 GAIL에서 발생하는 훈련의 불안정성을 해결하기 위해, 인간 운전자의 복잡한 행동을 모델링하기 위한 차별화된 접근 방식을 사용합니다. 이 방법은 각 에이전트의 행동을 ego-맵 및 ego-이웃으로 세분화하여 평가하고, 비관련 상호작용을 억제합니다. 추가적으로 사회적 PPO(Social PPO) 목표를 도입하여 에이전트의 리얼리즘을 높이는데 기여하며, 근처 에이전트의 행동에 영향을 주지 않으면서도 자신의 보상을 최적화합니다.

- **Performance Highlights**: DecompGAIL은 WOMD Sim Agents 2025 벤치마크에서 기존의 GAIL보다 훈련 안정성을 크게 향상시켰으며, 최신 성능을 기록했습니다. 이러한 개선은 차량 충돌 및 비정상적인 동작을 줄이는데 기여하며, 궁극적으로 교통 모델링에서의 실제 대처능력을 확보하는데 중요한 역할을 합니다. 연구 결과에 따르면, 제안된 방법은 다양한 다중 에이전트 시나리오에서 유망한 결과를 보여주었습니다.



### Utilizing Large Language Models for Machine Learning Explainability (https://arxiv.org/abs/2510.06912)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 자율적으로 기계 학습(ML) 솔루션을 생성할 때의 설명 가능성(explainability) 능력을 탐구합니다. 연구자는 두 가지 분류 작업을 통해 LLM이 생성한 솔루션의 성능 및 설명 가능성을 평가했습니다. 특히, 이 연구는 LLM이 정의한 파이프라인을 통해 Random Forest, XGBoost, Multi-Layer Perceptron(MLP), Long Short-Term Memory(LSTM) 등 네 가지 분류기를 훈련시키는 과정을 분석합니다.

- **Technical Details**: 연구에서 사용된 두 가지 데이터셋은 드라이버 경각 상태 예측을 위한 이진 분류와 효모(yeast) 데이터셋을 기반으로 한 다중 라벨(multilabel) 분류로 나뉩니다. 연구진은 세 가지 최신 LLM(OpenAI GPT, Anthropic Claude, DeepSeek)에 대하여 각 분류기 훈련을 위한 코드 생성을 요청했습니다. 생선된 모델의 성능은 recall, precision, F1-score로 평가되며, 설명 가능성은 SHAP(SHapley Additive exPlanations) 지표를 통해 측정됩니다.

- **Performance Highlights**: 연구 결과, LLM은 높은 설명 가능성과 함께 효과적인 모델을 생성하며, SHAP Fidelity(모델 출력과 SHAP 값 간의 평균 제곱 오차)와 SHAP Sparsity(영향을 미치는 변수의 수)에서 일관된 결과를 보여주었습니다. 이를 통해 LLM은 자동화된 해석 가능한 ML 파이프라인 생성 도구로서의 가능성을 보여주었으며, 수작업으로 설계된 기준 모델과 근접한 성과를 달성했습니다.



### Vacuum Spiker: A Spiking Neural Network-Based Model for Efficient Anomaly Detection in Time Series (https://arxiv.org/abs/2510.06910)
Comments:
          53 pages, 16 figures, preprint submitted to a journal for review

- **What's New**: 이번 논문에서는 새로운 	extit{Vacuum Spiker algorithm}을 소개합니다. 이 방법은 시계열에서의 이상 탐지를 위한 Spiking Neural Network 기반의 알고리즘으로, 전통적인 재구성 또는 예측 오차가 아닌 신경 활동의 글로벌 변화에 기반한 새로운 탐지 기준을 도입합니다. 이 알고리즘은 한 번의 스파이크로 정보를 인코딩하여, 에너지 효율성을 크게 향상시키는 새로운 인코딩 방식을 포함하고 있습니다.

- **Technical Details**: 이 알고리즘은 Spike Time-Dependent Plasticity(STDP)를 새로운 방식으로 적용하여, 데이터가 비정상적일 때 신경 활동을 변화시키도록 훈련됩니다. 또한, 특정 연결에 대해 주로 억제 또는 강화 이벤트가 발생하게끔 하여 숨겨진 뉴런의 활동을 조정하는 전략을 채택하고 있습니다. 전통적인 인공 신경망과 비교했을 때, SNN의 동적 행동을 활용하여 복잡한 시계열 패턴을 포착하는 데 특화되어 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 알고리즘이 기존의 다양한 딥러닝 및 머신러닝 모델과 비교하여 경쟁력 있는 성능을 보여주면서, 에너지 소모를 상당히 줄인다는 것을 나타내고 있습니다. 또한, 실제 사례 연구를 통해 이 모델은 태양광 인버터에서 전력 제한 이벤트를 성공적으로 식별하여 그 실용성을 입증했습니다. 이 결과는 지속 가능하고 효율적인 이상 탐지를 위한 잠재력을 강조합니다.



### Angular Constraint Embedding via SpherePair Loss for Constrained Clustering (https://arxiv.org/abs/2510.06907)
Comments:
          Accepted by NeurIPS 2025, 6 Figures and 1 Table in Main text, 18 Figures and 5 Tables in Appendices

- **What's New**: 이번 연구에서는 Deep Constrained Clustering (DCC) 방식에서 기존 방법의 한계를 극복하기 위해 SpherePair이라는 새로운 angular constraint embedding 접근법을 제안합니다. SpherePair 손실(SpherePair loss)은 기하학적 구성을 사용하여 쌍(pairwise) 제약조건을 충실하게 인코딩하며, 클러스터링에 적합한 각 공간에서의 임베딩을 생성합니다. 이 접근법은 클러스터 수를 지정할 필요 없이 클러스터링을 수행할 수 있도록 하여, 하이퍼파라미터 조정을 위한 노력을 줄이고 효율성을 개선합니다.

- **Technical Details**: SpherePair 손실 함수는 코사인 유사도(cosine similarity)를 활용하여 앵커 없이 각 공간에서 잠재적 표현(latent representation)을 학습합니다. 이를 통해 긍정 쌍(positive pairs)과 부정 쌍(negative pairs) 간의 적절한 거리를 유지하면서 클러스터 간 거리의 균형을 맞출 수 있습니다. 이 방법은 이론적으로 검증된 기초를 바탕으로 하여 특정 조건에서 최적 성능을 보장합니다.

- **Performance Highlights**: 범위가 다양한 벤치마크 데이터셋에서 기존 DCC 방법들과 비교하여 SpherePair의 우수한 성능을 입증했습니다. 특히, 간단한 K-means 알고리즘을 사용하여 학습된 표현에 대한 클러스터 수를 신속하게 추측할 수 있으며, 미지의 데이터에도 잘 일반화됩니다. 실험 결과는 저자들이 제안한 방법이 대안 방법들보다 효과적이고 실용적임을 보여줍니다.



### MoRE-GNN: Multi-omics Data Integration with a Heterogeneous Graph Autoencoder (https://arxiv.org/abs/2510.06880)
- **What's New**: 이번 연구에서는 MoRE-GNN(Multi-omics Relational Edge Graph Neural Network)을 소개합니다. 이 모델은 데이터로부터 직접 관계 그래프를 동적으로 구성하는 이종 그래프 자동 인코더로, 생물학적으로 의미 있는 관계를 포착하는 데 중점을 두고 있습니다. MoRE-GNN은 단일 세포 다중 유전체 데이터 통합을 위한 고차원성 문제를 해결하고, 기존 방법들에 비해 우수한 성능을 발휘합니다.

- **Technical Details**: MoRE-GNN의 진행 방식은 세 가지 단계로 나뉘어 있습니다. 첫 번째 단계에서는 모달리티별 코사인 유사성을 사용하여 관계 엣지를 구성하고, 두 번째 단계에서는 GCN(Graph Convolutional Network)과 attention 메커니즘을 이용해 이질적인 메시지 패싱을 수행합니다. 마지막으로, 학습된 임베딩을 2차원으로 투영하고 집단을 Louvain 클러스터링으로 식별합니다.

- **Performance Highlights**: 여섯 개의 공개 데이터셋에서의 평가 결과, MoRE-GNN은 생물학적으로 의미 있는 관계를 포착하면서 기존의 방법들보다 뛰어난 성능을 나타냈습니다. 이 모델은 고차원성 문제를 극복하고, 다양한 데이터셋에 적용 가능하다는 장점을 가지고 있습니다. MoRE-GNN은 다중 유전체 통합 및 단일 세포 분석의 향상을 위한 유망한 도구로 자리 잡을 것으로 기대됩니다.



### SaFeR-VLM: Toward Safety-aware Fine-grained Reasoning in Multimodal Models (https://arxiv.org/abs/2510.06871)
- **What's New**: 이번 논문에서는 SaFeR-VLM이라는 새로운 안전 정렬 강화 학습 프레임워크를 제안합니다. 이 프레임워크는 다중 모달 (multimodal) 추론 과정에서 안전성을 직접 통합하여, 안전을 수동적인 보호 장치가 아닌 적극적인 추론 동력으로 전환합니다. 또한, 안전성이 높은 데이터셋(QI-Safe-10K)을 통해 위험한 상황을 보다 효과적으로 다룰 수 있는 구조를 갖추고 있습니다.

- **Technical Details**: SaFeR-VLM의 기본 구조는 네 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 특정 안전상 중요한 사례에 중점을 둔 커리큘럼 데이터셋(QI-Safe-10K)입니다. 두 번째는 안전성 인식 롤아웃(safety-aware rollout)으로, 안전하지 않은 출력을 단순히 버리지 않고 반영 및 수정하는 과정을 포함합니다. 세 번째는 다차원 보상 모델링(structured reward modeling)으로, 환각과 모순에 대한 명시적 벌칙을 포함한 보상 신호가 포함되어 있습니다. 네 번째는 안전성 인식 최적화(safety-aware optimization)로, 이러한 신호를 GRPO 최적화 방법에 통합하는 방식입니다.

- **Performance Highlights**: SaFeR-VLM은 여섯 개의 안전 기준 벤치마크에서 우수한 성능을 발휘하며, 3B 모델에서 70.13의 안전성 점수와 78.97의 유용성을 기록했습니다. 이는 기존 10배 이상 큰 모델들을 초과하는 성능입니다. 7B 모델은 안전성 지표에서 GPT-5 미니 및 Gemini-2.5-Flash를 각각 6.47점, 16.76점 초과하여 유용성과 안전성을 동시에 개선한 사례로 주목받고 있습니다.



### Towards Generalization of Graph Neural Networks for AC Optimal Power Flow (https://arxiv.org/abs/2510.06860)
Comments:
          Pre-print has been submitted for review

- **What's New**: 본 연구는 AC Optimal Power Flow (ACOPF) 문제를 해결하기 위해 Hybrid Heterogeneous Message Passing Neural Network (HH-MPNN) 아키텍처를 제안합니다. 이 네트워크는 전력 시스템의 다양한 구성 요소를 모델링하여 복잡한 토폴로지 변화에 적응할 수 있는 능력을 제공합니다. HH-MPNN은 전통적인 솔버보다 1,000배에서 10,000배 더 빠른 계산 속도를 기록하며, 수천 개의 새로운 토폴로지에 대해 3% 미만의 최적성 차이를 달성합니다.

- **Technical Details**: HH-MPNN은 heterogeneous GNN과 스케일 가능한 transformer를 결합하여 ACOPF 변수를 예측합니다. 이 아키텍처는 전력 그리드의 각 구성 요소를 명시적으로 모델링하여 지역 정보를 효율적으로 집계하며, transformer는 자가 주의(self-attention)를 통해 전역 정보를 교환합니다. 이 과정은 인코딩, 프로세싱, 디코딩의 세 단계로 구성되며, 각 단계에서 지역 및 글로벌 정보를 효과적으로 결합합니다.

- **Performance Highlights**: HH-MPNN은 다양한 그리드 크기에서 우수한 일반화를 보여주며, 작은 그리드에서 사전 훈련된 모델이 더 큰 그리드의 성능을 향상시킵니다. 본 모델은 14부터 2,000까지의 버스를 가진 그리드에서 1% 미만의 최적성 차이를 달성하고, 새로운 N-1 비상 상황에 대해 높은 적응성을 보입니다. 이 결과는 실시간 전력 시스템 운영을 위한 실제적이고 일반izable한 머신러닝 접근 방식을 한층 발전시킵니다.



### Enhancing Bankruptcy Prediction of Banks through Advanced Machine Learning Techniques: An Innovative Approach and Analysis (https://arxiv.org/abs/2510.06852)
- **What's New**: 이 연구에서는 머신 러닝(machine learning) 기법을 사용하여 은행 파산 예측 모델을 개발하였습니다. 기존의 통계 모델, 예를 들어 Altman's Z-Score는 경직된 가정에 의존하기 때문에 예측 정확도가 낮다는 한계를 가지고 있습니다. 이에 따라 머신 러닝 기법을 사용하여 보다 효과적인 해결책을 모색했습니다.

- **Technical Details**: 연구에 사용된 방법에는 로지스틱 회귀(logistic regression), 랜덤 포레스트(random forest), 서포트 벡터 머신(support vector machines) 등이 포함됩니다. 이러한 머신 러닝 기법은 기존의 통계적 접근 방식보다 은행 리스크 관리를 더 효과적으로 분류하고 예측하는 데 우수함을 보여주었습니다. 연구 데이터는 터키의 44개 활성 은행과 21개 파산 은행의 연간 재무제표에서 도출되었고, 인도네시아의 43개 활성 및 43개 파산 농촌 은행의 분기 재무 보고서도 활용되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 랜덤 포레스트는 상업 은행 데이터를 90%의 정확도로 예측할 수 있었습니다. 또한 제안된 3가지 머신 러닝 방법이 농촌 은행의 파산 가능성을 정확히 예측하는 데 기여하였습니다. 이는 파산 비용을 줄이는 정책을 구현하는 데 중요한 기초 자료로 활용될 수 있습니다.



### CNN-TFT explained by SHAP with multi-head attention weights for time series forecasting (https://arxiv.org/abs/2510.06840)
- **What's New**: 이 논문에서는 다변량 시계열 예측을 향상시키기 위해 컨볼루션 신경망(CNN)과 임시 융합 트랜스포머(TFT)를 통합한 하이브리드 아키텍처를 제안합니다. CNN 모듈은 1차원 컨볼루션 레이어의 계층 구조를 적용하여 원본 입력 시퀀스에서 중요한 로컬 패턴을 추출하고, TFT는 다중 헤드 주의를 통해 단기 및 장기 종속성을 캡처합니다. 실험 결과 및 높인 설명력을 통해 제안된 모델이 기존의 심층 학습 모델들보다 우수한 성능을 보임을 입증했습니다.

- **Technical Details**: 제안된 CNN-TFT-SHAP-MHAW 모델은 원인 1D 컨볼루션 블록을 인코더에 사용하여 CNN의 효율성과 병렬 처리 능력을 활용합니다. 이 구조는 TFT 파이프라인의 나머지를 유지하면서 반복적 코어를 컨볼루션 코어로 교체합니다. 설명 가능한 인공지능(AI) 기술과 SHAP(Shapley additive explanations)를 통합하여 모델의 예측을 해석하는 데 도움을 줍니다.

- **Performance Highlights**: CNN-TFT는 수력 발전 자연 유량 시계열 데이터세트에서 평가되었으며, 평균 절대 백분율 오차(MAPE)가 최대 2.2%에 도달했습니다. 제안된 모델은 CNN 및 주의 메커니즘의 장점을 결합하여 신호 예측에 필요한 가장 중요한 특징을 보장하며, SHAP 값을 경량화하여 더 나은 의사결정을 위한 설명 가능성을 높였습니다. 이 모델은 고정밀 다변량 시계열 예측이 필요한 응용 분야에 유망합니다.



### Vectorized FlashAttention with Low-cost Exponential Computation in RISC-V Vector Processors (https://arxiv.org/abs/2510.06834)
- **What's New**: 본 연구는 RISC-V의 벡터 프로세서를 위한 FlashAttention 알고리즘의 첫 번째 벡터화(vecterize) 구현을 제안합니다. 이를 통해 스칼라 코드(scalar code)를 최소화하고, softmax를 평가하는 데 필요한 지수 함수의 계산 복잡성을 줄였습니다. 또한 실제 응용 프로그램의 주의(attention) 레이어를 처리할 때 성능을 크게 향상시키는 벡터화된 구현체의 확장성을 시연했습니다.

- **Technical Details**: 이 연구는 기계 학습 및 인공지능 모델에서 핵심 작업인 attention 기제를 다루고 있으며, FlashAttention 알고리즘을 활용해 RISC-V 벡터 ISA 확장을 목표로 하고 있습니다. 이를 통해 지수 계산에서 저비용 근사 방법을 사용하여 성능 및 에너지 효율을 개선하고, 메모리 로컬리티(memory locality)를 향상시키기 위한 타일링 전략(tiling strategies)을 개발했습니다. 또한, 새 사용자 정의 하드웨어 명령이나 ISA 확장 없이도 softmax 연산을 지원하는 단순화된 지수 함수 계산 방법을 통합했습니다.

- **Performance Highlights**: 실험은 gem5 시뮬레이터를 사용하여 수행되었으며, 다양한 시퀀스 길이와 헤드 차원을 포함하여 transformer 모델에서 일반적으로 나타나는 다양한 작업 패턴을 반영한 벤치마크를 포함합니다. 제안된 벡터화된 FlashAttention 구현은 유의미한 성능 향상을 보여주었으며, 효과적인 타일링 전략이 메모리 재사용(memory reuse)과 벡터 유닛 효율(vector unit efficiency)을 극대화하는 데 중요함을 강조했습니다. 마지막으로, 사용 가능한 벡터 길이를 증가시키면 성능이 일관되게 향상되어 현대 벡터 하드웨어에서의 확장성이 강조되었습니다.



### Early wind turbine alarm prediction based on machine learning: AlarmForecasting (https://arxiv.org/abs/2510.06831)
Comments:
          International Journal of Electrical Power and Energy Systems

- **What's New**: 새로운 연구에서는 Wind Turbines (WTs)의 결함 행동을 억제하는 데 필수적인 알람 데이터를 사용하여 사전에 알람을 예측하고 이를 예방하는 방법을 제시합니다. 기존 연구들은 알람 데이터를 진단 도구로만 활용했으나, 이번 연구는 알람이 발생하기 전에 이를 방지하는 것을 목표로 합니다. 이 연구의 결과는 전통적인 알람 진단의 한계를 넘어서는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: 제안된 Alarm Forecasting and Classification (AFC) 프레임워크는 두 개의 모듈로 구성되어 있습니다. 첫 번째 모듈은 Long Short-Term Memory (LSTM)를 기반으로 한 회귀 모듈로, 시간에 따른 알람 예측을 수행합니다. 두 번째 모듈은 예측된 알람에 알람 태깅을 구현하는 분류 모듈입니다. 이를 통해 특정 알람이 아니라 전체 알람 분류체계에 대한 신뢰할 수 있는 예측이 가능합니다.

- **Performance Highlights**: 사례 연구로 5년간 운영된 14개의 Senvion MM82 터빈을 사용하였습니다. 그 결과 10분, 20분, 30분 알람 예측의 정확성이 각각 82%, 52%, 41%로 나타났습니다. 이러한 결과는 알람을 예측하고 방지하는 것이 가능함을 입증하며, 알람 빈도를 줄이고 운영 효율성을 향상시키는 데 중요한 역할을 할 수 있음을 보여줍니다.



### Recurrence-Complete Frame-based Action Models (https://arxiv.org/abs/2510.06828)
- **What's New**: 최근 대형 언어 모델에 대해 주목할 만한 변화를 가져온 원리는 Attention 메커니즘입니다. 본 논문에서는 Attention 메커니즘만으로는 긴 시간 동안의 에이전트 작업에 필요한 문제를 해결할 수 없다는 주장으로 이와는 반대되는 시각을 제시합니다. 우리는 진정한 연속적 계산(true serial computation)이 필수적이고, 완전 병렬izable 아키텍처로는 필요한 계산을 일반적으로 표현할 수 없다는 점을 강조합니다.

- **Technical Details**: 논문은 '진정한 깊이(true depth)'와 '순환 완전성(recurrence completeness)'을 정의하며, 후자는 일반적인 비사소조합(즉, 비연관적) 업데이트를 실현할 수 있는지에 따라 구분됩니다. 특히, 시간 병렬 아키텍처는 제한된 깊이를 가지며, 이는 최악의 경우 긴 시퀀스 문제에 필요로하는 연산 능력을 결여하게 만듭니다. 우리 실험은 특정 비경로 적합성 문제를 해결하기 위해 작성된 직렬 평가가 중요함을 보여줍니다.

- **Performance Highlights**: 결과적으로, 긴 시퀀스 훈련은 모델의 손실을 줄이며 시계열 데이터 처리에서 유의미한 성과를 보입니다. 우리는 훈련된 시퀀스 길이에 대한 손실이 고정된 매개변수 아래에서 전력 법칙(power law)을 따르며, 더 긴 시퀀스 훈련이 짧은 시퀀스에 비해 손실 이점이 유지되는 것으로 나타났습니다. 주목해야 할 점은, 모델의 감각 상태가 향상되는 것을 보여주는 경향이 관찰되었다는 점입니다.



### Efficient numeracy in language models through single-token number embeddings (https://arxiv.org/abs/2510.06824)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 큰 숫자를 효율적으로 처리하고 복잡한 계산을 수행하는 능력을 필요로 한다고 강조합니다. 기존의 수치 처리 방법이 LLM의 수치적 직관을 제한하거나 해결할 수 있는 문제의 길이를 제한하고 있다는 점을 지적합니다. 이를 해결하기 위해 저자들은 BitTokens라는 새로운 토큰화 전략을 제안하며, 이 방식을 통해 LLMs가 기본 산술 연산을 거의 완벽하게 수행할 수 있도록 하는 효율적인 한 토큰 숫자 인코딩 방식을 소개합니다.

- **Technical Details**: BitTokens 방법은 IEEE 754 이진 부동소수점 표현을 활용하여 어떤 숫자도 단일 토큰으로 인코딩할 수 있습니다. 이 토큰화 전략은 수치의 부호, 지수 및 유의 숫자를 비트 시퀀스로 변환하여 LLM이 내부적으로 계산 알고리즘을 학습할 수 있게 합니다. 연구는 8개의 최신 LLM의 수리 능력을 평가하여 기존 접근 방식의 한계를 분석하고, 효과적인 단일 토큰 숫자 인코딩을 위한 기준을 개발합니다.

- **Performance Highlights**: 저자들은 BitTokens의 도입을 통해 작은 언어 모델조차도 기본 산술 연산을 거의 완벽하게 해결할 수 있게 되었음을 보여줍니다. 해당 연구는 LLM이 수치 계산을 위해 필요한 큰 양의 토큰을 줄여주며, 이는 LLM이 해결할 수 있는 문제의 길이와 복잡성을 확장할 수 있는 기반을 마련합니다. 새로운 수치적 접근 방식은 과학 및 공학 분야의 혁신을 가속화하는 데 기여할 것으로 기대됩니다.



### The Unreasonable Effectiveness of Randomized Representations in Online Continual Graph Learning (https://arxiv.org/abs/2510.06819)
- **What's New**: 이 논문에서는 온라인 지속 그래프 학습(Online Continual Graph Learning, OCGL)을 위한 간단하면서도 매우 효과적인 접근 방식을 제안합니다. 노드 임베딩(Node Embedding)을 생성하기 위해 고정되고 임의로 초기화된 인코더(Encoder)를 사용하고, 경량화된 분류기(Classifier)만을 온라인으로 훈련시킵니다. 이를 통해 표현 파라미터의 드리프트(drift)를 제거하고, 표현력이 뛰어나면서 안정적인 임베딩을 얻을 수 있습니다. 이 방법은 메모리 버퍼 없이도 기존의 최첨단 방법들에 비해 뛰어난 성능 향상을 보여줍니다.

- **Technical Details**: 저자들은 그래프의 구조적 의존성을 고려하여, 노드 표현을 예측 모델과 분리하는 단순한 방법을 소개합니다. 이 접근 방식은 무작위화된(over-parametrized) 구조의 효과를 활용하여 훈련되지 않은 풍부한 표현을 제공하고, 경량화된 분류기를 조합하여 좋은 성능을 보입니다. 특히, 고정된 표현(fixed representations)은 과거에 학습한 개념을 잊는 것을 최소화하는 데 도움을 줍니다. 또한, Streaming Linear Discriminant Analysis (SLDA) 분류기를 결합하여 기존 방법들을 초과하는 성능을 내고 있습니다.

- **Performance Highlights**: 실험 결과, 이 방법은 여러 OCGL 벤치마크에서 놀라운 성능 개선을 보여주었으며, 약 30%까지의 성능 향상을 기록했습니다. 이 접근법은 수업 증가(class-incremental) 및 시간 증가(time-incremental) 설정 모두에서 최고 결과를 달성했습니다. 연구 결과는 향후 OCGL 연구에서 모델 설계를 재고해야 함을 시사하며, 복잡한 리플레이나 정규화 없이 아키텍처의 단순성과 안정성을 강조해야 한다고 주장합니다.



### Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness (https://arxiv.org/abs/2510.06790)
Comments:
          17 pages

- **What's New**: 이 논문에서는 모델이 적대적인 범위 외(out-of-distribution, OOD) 데이터에 취약하다는 문제를 다루고 있습니다. Zaremba et al. (2025)의 연구 결과에 따르면, 대규모 언어 모델(LLM)의 추론(pta) 과정이 모델의 사양(specification)을 충족하는 데 도움을 줌으로써 강력한 방어를 제공한다는 점을 보였습니다. 하지만 공격자가 그라디언트(gradient)나 다중 모드 입력(multimodal inputs)에 접근할 수 있는 경우, 이러한 테스트 계산(test compute)의 이점은 사라진다는 사실을 부각시켰습니다.

- **Technical Details**: 논문은 추론 계산(inference-compute)이 방어적 사양을 준수할 수 있도록 OOD 데이터를 인식하는 데 도움을 줄 수 있음을 주장합니다. 특히 '추론 계산의 Robustness Hypothesis(RICH)'를 제안하며, 이는 모델의 학습 데이터가 공격받는 데이터의 구성요소를 더 잘 반영할수록 방어적 효과가 증가한다는 것입니다. 이러한 가설을 통해, 비전 언어 모델(vision language model, VLM)과 다양한 공격 유형에 대해 실험적으로 지지를 받았습니다.

- **Performance Highlights**: 이 연구에서 테스트 단계의 계산이 방어적 사양을 따라갈 수 있는 OOD 데이터에 대해 강력함을 증대시키는 것으로 나타났습니다. 예를 들어, 방어적 사양을 강조하는 프롬프트(prompting)가 적대적 사전 훈련(adversarial pretraining)으로 강화된 VLM에 대한 그래디언트 기반 다중 모드 공격의 성공률을 낮추는 것으로 확인되었습니다. 따라서 훈련 시 방어와 테스트 시 방어를 조합하는 것이 상호 보완적 이익을 얻기 위한 방법으로 제안되었습니다.



### Modeling COVID-19 Dynamics in German States Using Physics-Informed Neural Networks (https://arxiv.org/abs/2510.06776)
Comments:
          19 pages, 7 figures, 2 tables

- **What's New**: COVID-19 팬데믹은 질병 동역학을 이해하기 위한 정량적 모델링과 분석의 필요성을 강조했습니다. 이 연구에서는 로베르트 코흐 연구소(RKI)의 감염 데이터를 사용하여 SIR (Susceptible-Infectious-Recovered) 모델의 역문제를 해결하기 위해 Physics-Informed Neural Networks (PINNs)를 활용하였습니다. 독일 16개 연방주에 대한 COVID-19 동역학의 세밀한 시공간 분석을 제공하고, 각각의 주에서 전파 및 회복 매개변수를 추정하고 있습니다.

- **Technical Details**: 비선형적 매개변수의 추정은 일반적으로 관측 데이터를 기반으로 합니다. PINNs는 미분 방정식을 네트워크 훈련에 통합하여 이러한 한계를 극복하고, 실제 세계의 관찰 데이터를 통해 잠재적 매개변수를 효과적으로 추정할 수 있습니다. 이 연구에서는 1,200일 동안의 COVID-19 데이터에 대한 분석을 통해 연방주별 전파율 및 회복율을 파악하였으며, 이를 통해 시간에 따라 변화하는 재생산 수치(ℛ_t)도 추정하였습니다.

- **Performance Highlights**: 연구 결과, 각 연방주 간 전파 행동의 강한 차이를 보여주었고, 예방접종 수치 및 주요 팬데믹 단계와의 상관관계를 밝혀냈습니다. 이는 지역적인 개입이 전파 동역학에 측정 가능한 영향을 미쳤음을 시사합니다. PINNs를 활용한 지역적이고 장기적인 역학 모델링의 유용성을 확인하며, COVID-19의 확산이 시간이 지남에 따라 어떻게 변화했는지에 대한 깊은 통찰을 제공합니다.



### Function regression using the forward forward training and inferring paradigm (https://arxiv.org/abs/2510.06762)
Comments:
          Keywords: Neural Networks, Forward Forward training, Function Regression, Physical Neural Networks, Analog Computing

- **What's New**: 이 논문은 Forward-Forward 알고리즘을 사용하여 함수 회귀(function regression)을 수행하는 새로운 방법론을 제안합니다. 그동안 Forward-Forward 알고리즘은 분류 작업에만 국한되었으나, 본 연구에서는 이를 회귀 작업으로 확장하는 방법을 탐구합니다. 또한, 이 연구는 단변량 및 다변량 함수에 대한 성과를 평가하며, Kolmogorov Arnold Networks 및 Deep Physical Neural Networks에의 확장 가능성도 논의합니다.

- **Technical Details**: Forward-Forward 알고리즘은 백프로파게이션(backpropagation)을 사용하지 않고 레이어별(layer-wise)로 신경망을 학습하는 방법입니다. 이 알고리즘은 올바르게 레이블된 데이터와 잘못된 레이블이 있는 데이터를 비교하여 학습하는 관점에서 개발되었습니다. 'goodness'라는 함수를 통해 각 레이어의 출력을 수량화하고, 이를 통해 레이어의 가중치를 조정하여 올바른 데이터와 잘못된 데이터 간의 차이를 극대화합니다.

- **Performance Highlights**: 이 연구는 Forward-Forward 회귀 알고리즘을 1-D, 2-D, 3-D 벤치마크에서 검증하며, 이를 통해 제안된 알고리즘의 초기 결과를 제공합니다. 성과 측정에서, 전통적인 방식보다 에너지 효율성을 높일 수 있는 가능성을 보여주며, Forward-Forward 알고리즘이 다양한 물리적 신경망에 적용될 수 있는 기반을 마련합니다.



### Incorporating Expert Knowledge into Bayesian Causal Discovery of Mixtures of Directed Acyclic Graphs (https://arxiv.org/abs/2510.06735)
Comments:
          28 pages, 18 figures

- **What's New**: 이 논문에서는 이질적인 (heterogeneous) 도메인에서의 인과관계 발견을 위한 새로운 전략을 제안합니다. Bayes 실험 설계(Bayesian experimental design, BED) 원칙에 기반한 인과 지식 elicitation 전략과, 변분 혼합 구조 학습(variational mixture structure learning, VaMSL) 방법을 통합하여 인과 Bayesian 네트워크(CBN)의 혼합을 반복적으로 추론하는 방법을 발전시켰습니다. 이는 단일 인과 그래프의 가정을 넘어서 복잡한 분포를 캡처할 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 지식 전문가의 피드백을 포함한 정보성 그래프 사전(informative graph prior)을 구축하여 혼합 CBN의 추론 수행을 통해 모델 성능을 향상시킵니다. 혼합 모델을 학습하고 최적의 쿼리를 선택하는 BED 원칙에 기초한 엘리시테이션 전략을 제안합니다. 또한, 기존의 차별 가능한 Bayesian 구조 학습(differentiable Bayesian structure learning, DiBS) 방법을 확장하여 선형 및 비선형 Bayesian 네트워크의 혼합을 추론하는 변분 방법을 제공합니다.

- **Performance Highlights**: 제안된 방법은 이질적인 합성 데이터에서 개선된 구조 학습 성능을 보이며, 전문가 시뮬레이션에 의해 더 나은 결과를 도출합니다. 특히, 유방암 데이터베이스에서 복잡한 분포를 효과적으로 포착할 수 있음을 입증했습니다. 이로써, 전문가의 지식을 통합하는 혁신적인 접근 방식으로 인과 모델링에서의 새로운 가능성을 제시합니다.



### Dual Goal Representations (https://arxiv.org/abs/2510.06714)
- **What's New**: 이번 연구에서는 목표 지향 강화 학습(goal-conditioned reinforcement learning, GCRL)을 위한 이중 목표 표현(dual goal representations) 개념을 도입합니다. 이중 목표 표현은 모든 다른 상태에서의 시간적 거리(temporal distances) 집합을 통해 상태를 특성화합니다. 이러한 표현 방식은 기존의 상태 표현(original state representation)에 영향을 받지 않는 특징이 있습니다.

- **Technical Details**: 이 이중 목표 표현은 환경의 내재적 동적(intrinsic dynamics)만을 의존하며, 외부 잡음(exogenous noise)을 필터링(filer out)할 수 있는 충분한 정보를 포함합니다. 이를 기반으로, 기존의 GCRL 알고리즘과 결합할 수 있는 실용적인 목표 표현 학습 방법을 개발하였습니다. 또한, 이 방법은 최적 목표 도달 정책(optimal goal-reaching policy)을 복원하는 데 필요한 정보를 담고 있습니다.

- **Performance Highlights**: OGBench 작업 세트(task suite)에서 다양한 실험을 수행하여 이중 목표 표현이 20개의 상태(state) 및 픽셀(pixel) 기반 작업에서 오프라인(goal-reaching) 성능을 일관되게 개선하는 것을 경험적으로 보여주었습니다. 이 결과는 이중 목표 표현의 유용성과 잠재력을 입증합니다.



### A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking (https://arxiv.org/abs/2510.06699)
Comments:
          Accepted to NeurIPS 2025; The first two authors contributed equally and are co-leading authors

- **What's New**: 이 논문에서는 의료, 금융 및 과학 분야에서 중요한 역할을 하는 현실적인 시간 시계열 데이터 생성의 필요성에 대해 설명합니다. 기존 연구에서는 이러한 비정상성을 해결하기 위해 노력했으나, 결과가 최적이 아닌 경우가 많고 높은 계산 비용이 수반되었습니다. 최근의 ImagenTime 모델은 비정상 시퀀스에 대한 확장을 제안하며, 새로운 두 단계 체계를 통해 이 문제를 해결하고 있습니다.

- **Technical Details**: 저자들은 첫 번째 단계에서 비정규 시계열을 완성하기 위해 Time Series Transformer(TST)를 활용하고, 두 번째 단계에서는 마스킹을 통한 비전 기반 확산 모델을 적용합니다. 이 접근법은 이미지 변환을 통해 비정상 시계열의 복잡성을 줄이고, 자연스러운 이미지 이웃을 통해 학습 효과를 극대화합니다. 최종적으로, 이 모델은 긴 시계열 모델링을 가능케 하며, 전 처리된 데이터에 대한 최소한의 가정을 기반으로 동작합니다.

- **Performance Highlights**: 제안된 방법은 다양한 평가 기준에서 최첨단 성능을 달성했으며, 특히 비정상적으로 샘플링된 시계열에서 70%의 개선된 분류 점수와 85%의 계산 비용 절감을 보여주었습니다. 이를 통해 기존 접근 방식 대비 우수한 생성 성능을 유지하면서도 효율성을 극대화한 성과를 강조합니다. 논문의 기여는 비정상 시계열 생성을 위한 새로운 모델을 도입한 점이며, 비전 기반의 확산 접근법을 활용하여 실현 가능성을 높인 것을 포함합니다.



### Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial? (https://arxiv.org/abs/2510.06692)
- **What's New**: 이 논문은 Deep Neural Networks (DNNs)의 내부 모델을 추출하는 공격 방법론에 관한 새로운 접근법을 제시합니다. 기존 연구들은 DNN의 정확한 출력 logits에 의존하였으나, 본 연구에서는 hard-label 설정에서도 효율적인 모델 추출이 가능함을 보여줍니다. 특히, CrossLayer Extraction 공격 방법을 통해 깊은 레이어에서의 정보를 효율적으로 추출할 수 있는 방법을 제안합니다.

- **Technical Details**: CrossLayer Extraction 방법은 특정 뉴런의 비밀 매개변수(예: weights 및 biases)를 직접 추출하는 대신, 여러 레이어 간의 뉴런 상호작용을 활용하여 정보를 추출합니다. 이 방법은 쿼리 복잡성을 크게 줄이고, 기존의 모델 추출 접근법의 한계를 극복함으로써, 더욱 효율적인 공격을 가능하게 합니다. 또한, 이 공격 방법이 다층 네트워크에서의 비선형적 특징을 이용하며, 각 레이어의 활성화 경계에 대한 접근 방식을 새로운 시각으로 제시합니다.

- **Performance Highlights**: 실험을 통해 DNN의 깊이가 증가할수록 특정 활성화 경계에 접근하기 어려워지며, 일부 뉴런은 상태를 전환하기 위해 지수 시간 복잡성을 요구하는 것으로 나타났습니다. 이는 모델 추출 공격이 실질적으로 다루기 어려운 상황을 초래할 수 있음을 암시합니다. 그러나 CrossLayer Extraction 기법은 이러한 문제를 효과적으로 해결하고, 보다 신뢰할 수 있는 모델 추출 결과를 제공함으로써, DNN의 보안 문제에 대한 새로운 통찰을 제공합니다.



### AutoBalance: An Automatic Balancing Framework for Training Physics-Informed Neural Networks (https://arxiv.org/abs/2510.06684)
Comments:
          23 pages

- **What's New**: 이번 연구에서는 PINN (Physics-Informed Neural Networks) 훈련의 새로운 패러다임인 AutoBalance를 소개합니다. 기존의 "pre-combine" 전략에서는 서로 다른 손실 구성 요소의 그래디언트를 하나의 옵티마이저에서 처리해야 했으나, 이는 비효율적이었습니다. AutoBalance는 각 손실 구성 요소에 대해 독립적인 적응형 옵티마이저를 할당하고, 이 후에 업데이트를 집계하여 효율성을 개선합니다.

- **Technical Details**: AutoBalance 프레임워크는 각 손실 구성 요소에 대해 독립적인 옵티마이저를 최적화하여 서로 다른 곡률 특성을 가진 손실 함수들이 혼합되는 문제를 해결합니다. 이 방법은 각 옵티마이저가 해당 손실의 곡률에 자연스럽게 적응하도록 하여, 하이퍼파라미터 없이도 안정적이고 효과적인 업데이트를 제공합니다. 실험적으로, 이 방식은 기존의 방법들에 비해 훨씬 우수한 성능을 보여줍니다.

- **Performance Highlights**: 다양한 PDE 벤치마크에 대한 광범위한 실험을 통해 AutoBalance는 기존의 프레임워크들에 비해 현저한 감소된 오차를 기록했습니다. MSE(Mean Squared Error)와 $L^{orall}$ 노름을 기준으로 측정된 성과는 AutoBalance가 강력한 기본 성능을 보임을 보여줍니다. 또한, AutoBalance는 다른 인기 있는 PINN 방법론들과 상호 보완적이며, 도전적인 벤치마크에서의 효과성을 더욱 끌어올립니다.



### Distributed Algorithms for Multi-Agent Multi-Armed Bandits with Collision (https://arxiv.org/abs/2510.06683)
Comments:
          21 pages, 4 figures

- **What's New**: 이 논문에서는 여러 플레이어가 팔을 선택하여 누적 보상을 극대화하는 확률적 다중 사용자 다중 무장 농장(MMAB) 문제를 다룹니다. 특히, 각 플레이어가 자신의 행동 및 충돌 피드백만 관찰할 수 있는 분산 설정에서 충돌을 모델링하는 알고리즘을 제안합니다. 제안된 알고리즘은 거의 최적에 가까운 그룹 및 개인 후회를 달성하며, 통신 비용은 $m{O}(m{	ext{log log}} m T)$로 매우 낮습니다.

- **Technical Details**: 이 연구는 Collision-Sensing Multi-Agent Multi-Armed Bandits (CS-MMAB) 모델을 기반으로 하여 분산 환경에서 에이전트들이 간섭을 최소화하고 협력을 극대화하는 방안을 제시합니다. SynCD라는 동기화된 분산 알고리즘을 제안하며, 이는 개인 후회 O(1/M ∑_{k>M} log T/Δ_k) 및 그룹 후회 O(∑_{k>M} log T/Δ_k)를 달성합니다. 또한, 이 알고리즘은 기존 방법보다 효율적이며 확장성이 뛰어난 것을 강조합니다.

- **Performance Highlights**: SynCD는 기존 베이스라인인 SIC-MMAB 및 DPE1과 비교하여 모든 측면에서 우수한 성능을 보였습니다. 특히, SynCD는 그룹 후회, 개인 후회 및 통신 효율성에서 모두 눈에 띄는 성능 개선을 이뤘습니다. 이러한 결과는 분산 환경에서도 효과적으로 작동할 수 있음을 나타내며, 새로운 경량 통신 방식으로 성과를 높였습니다.



### TimeFormer: Transformer with Attention Modulation Empowered by Temporal Characteristics for Time Series Forecasting (https://arxiv.org/abs/2510.06680)
- **What's New**: 본 논문에서는 Time series 데이터에 적합한 새로운 Transformer 아키텍처인 TimeFormer를 제안합니다. 기존의 Transformer 모델들이 자연어 처리(NLP)와 시간 시계열(time series) 데이터 간의 차이를 충분히 고려하지 못하여 발생하는 문제를 해결하기 위해 설계되었습니다. TimeFormer는 시간적 속성을 포착하기 위해 두 가지 주요 특성(과거에서 미래로의 일방향적 영향과 시간에 따른 영향의 감소 현상)을 도입합니다.

- **Technical Details**: TimeFormer는 MoSA(Self-attention mechanism with two modulation terms)를 핵심 혁신으로 사용하여 시간 시계열의 차별성을 반영했습니다. 이 메커니즘은 시간적으로 인과적인 의존성과 주의(attention) 효과의 감소를 모델링하여 과거에서 미래로의 일방향적 영향을 강화합니다. 또한, 다중 스케일(multi-scale) 분석과 서브시퀀스(subsequence) 분석을 통합하여 다양한 시간적 스케일에서의 의미적 의존성을 포착합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 광범위한 실험 결과에 따르면, TimeFormer는 기존의 최첨단(SOTA) 방법과 비교하여 최대 7.45%의 MSE 감소를 달성했습니다. 또한, 평가 지표의 94.04%에서 새로운 벤치마크를 설정했습니다. 더욱이, MoSA 메커니즘은 다른 Transformer 기반 모델의 성능을 향상시키는 데에도 널리 적용될 수 있음을 보여줍니다.



### XRPO: Pushing the limits of GRPO with Targeted Exploration and Exploitation (https://arxiv.org/abs/2510.06672)
- **What's New**: 이 논문은 XRPO(eXplore - eXploit GRPO)라는 새로운 정책 최적화 프레임워크를 소개합니다. XRPO는 강화 학습(RL) 알고리즘인 GRPO의 한계를 극복하기 위해 롤아웃(exploitation) 탐색을 체계적으로 조절하는 방법을 제공합니다. 이를 통해 학습 과정에서의 탐색을 개선하고, 적절하게 향상된 보상 신호를 활용하여 모델의 성능을 향상시킵니다.

- **Technical Details**: XRPO는 수학적으로 근거 있는 롤아웃 할당기를 도입하여 불확실성이 높은 프롬프트(잘못 묻는 질문)로 우선 배정하도록 설계되어 있습니다. 또한, 제로 보상 프롬프트에서의 정체(stagnation)를 해결하기 위해 인컨텍스트(seed) 시딩 전략을 사용하여 커리큘럼 교육을 적용합니다. XRPO는 기존의 희소 보상(sparse rewards)을 넘어 시퀀스의 가능성을 활용하여 효과적으로 성능을 강화합니다.

- **Performance Highlights**: XRPO는 다양한 수학 및 코드 생성 벤치마크에서 실험을 수행했으며, GRPO 및 GSPO와 비교하여 최대 4%의 pass@1 및 6%의 cons@32 향상을 보여줍니다. 또한 XRPO는 학습 수렴을 최대 2.7배 가속화함으로써 Sample efficiency를 크게 개선했습니다. 이러한 결과는 탐색과 착취 간의 균형이 XRPO의 효과성을 뒷받침하고 있음을 보여줍니다.



### The Effect of Attention Head Count on Transformer Approximation (https://arxiv.org/abs/2510.06662)
- **What's New**: 이 논문에서는 Transformer의 구조적 매개변수가 표현력에 미치는 영향을 분석합니다. 특히, attention heads의 수가 함수 근사에 미치는 역할을 강조하며, $D$-retrieval이라는 일반화된 작업을 도입합니다. 기존의 연구에서 다뤄지지 않았던 비선형 환경에서의 최초의 엄밀한 하한값을 제시하고, heads 수가 부족할 경우 발생하는 정보 병목 현상에 대한 이해를 돕습니다.

- **Technical Details**: 연구팀은 Transformer 모델의 단일 계층을 분석하고, h < D일 때 인코딩이 적절히 이루어지지 않아 필요한 매개변수 복잡성이 지수적으로 증가함을 보여주었습니다. 특히, O(1/ϵ^{cT}) 매개변수가 필요하며, 이는 시퀀스 길이에 따라 의존적입니다. 반면, h ≥ D일 경우 heads는 서로 다른 좌표에 특화되어 정보 병목 현상이 해소되어 효율적인 근사가 가능하다는 점을 강조하였습니다.

- **Performance Highlights**: 실험을 통해 이론적 결과가 실제 데이터와 과제에서도 일관되게 나타나는 것을 확인하였습니다. 특히, 비선형 환경에서의 Transformer의 표현력 한계를 수치적으로 입증하였으며, heads 수가 충분할 경우 효율적인 근사가 이루어진다는 주장을 검증하였습니다. 이러한 결과는 자연어 처리, 컴퓨터 비전 등 다양한 분야에 실질적인 적용 가능성을 포함하고 있습니다.



### Rethinking Nonlinearity: Trainable Gaussian Mixture Modules for Modern Neural Architectures (https://arxiv.org/abs/2510.06660)
- **What's New**: 본 논문은 가우시안 혼합 모델(GMM)을 기반으로 한 비선형 모듈인 Gaussian Mixture-Inspired Nonlinear Modules (GMNM)을 제안합니다. GMNM은 전통적인 활성화 함수의 제약을 해소하고, 신경망 아키텍처에 융합하여 end-to-end 방식으로 훈련할 수 있는 유연한 기능을 갖추고 있습니다. 다양한 신경망 아키텍처에 GMNM을 통합함으로써 성능 향상을 입증했습니다.

- **Technical Details**: GMNM은 가우시안 프로젝션(Gaussian projections)의 유연한 파라미터화와 확률적 제약의 완화를 통해 미분 가능성(differentiable)을 얻는 새로운 분류의 모듈입니다. GMNM은 전통적인 확률적 제약을 완화하여 보편적인 함수 근사기(universal function approximators)로 변환하고, Mahalanobis 거리의 비선형성을 활용하여 기존 활성화 함수보다 우수한 표현능력을 제공합니다. 이를 통해 GMNM은 고차원 데이터의 복잡한 상관관계를 효과적으로 모델링할 수 있습니다.

- **Performance Highlights**: 실험 결과, GMNM은 표준 MLP 및 KAN보다 우수한 성능을 보여주었으며, CNN, LSTM과 같은 기존 아키텍처에 통합했을 때도 성능이 현저히 개선되었습니다. GMNM은 휴리스틱한 측정이 필요 없고 그라디언트 기반 훈련이 가능하여, 다양한 머신러닝 응용 프로그램에 대해 효율성과 정확성을 크게 향상시킬 가능성을 보여줍니다.



### Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions (https://arxiv.org/abs/2510.06649)
Comments:
          15 pages, 5 figures

- **What's New**: 이 논문은 Forward-Forward (FF) 알고리즘을 기반으로 한 새로운 가치 추정 방법인 Action-conditioned Root mean squared Q-Functions (ARQ)를 소개합니다. ARQ는 생물학적으로 타당한 신경망 학습 방법과 강화 학습 (Reinforcement Learning, RL)을 통합하여 개발되었습니다. 이 연구는 FF의 장점을 살리면서 RL 설정에서의 적용 가능성을 탐색합니다.

- **Technical Details**: ARQ는 두 가지 주요 요소로 구성되며, 하나는 임의 크기의 벡터에서 가치 예측을 추출하는 goodness function입니다. 두 번째 요소는 모델 입력에 동작 후보를 삽입하는 action conditioning이며, 이를 통해 네트워크는 각 상태-동작 쌍에 구체적인 표현을 생성할 수 있습니다. ARQ는 복잡한 입력을 모델링하는 데 있어 유연성을 높이면서도 backpropagation-free 특성을 유지합니다.

- **Performance Highlights**: MinAtar 및 DeepMind Control Suite 벤치마크에서 ARQ 방법은 기존의 local RL 방법과 일반적인 backprop 기반 방법보다 우수한 성과를 보였습니다. 특히, ARQ는 강화 학습 환경에서 강력한 의사 결정 능력을 보여줍니다. 이 연구는 생물학적으로 타당한 학습 방법과 강화 학습의 교차점에서 더 많은 탐색을 장려하고자 합니다.



### The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators (https://arxiv.org/abs/2510.06646)
- **What's New**: 이번 연구는 기계 학습된 연산자(Machine-Learned Operators, MLOs)가 높은 해상도의 데이터에서 추론(inference)을 수행할 수 있는지, 즉 '제로샷 초해상도(zero-shot super-resolution)'을 가능하게 하는지를 평가합니다. 연구진은 MLOs가 여러 해상도에서의 추론을 수행하는 데 있어 두 가지 주요 행동, 즉 주파수 정보의 외삽(extrapolation)과 해상도 간의 보간(interpolation)의 실패를 관찰했습니다. 이 결과를 바탕으로, MLOs가 훈련된 해상도 이외의 해상도에서 정확한 추론을 하지 못한다고 결론지었습니다.

- **Technical Details**: 연구의 주요 초점은 MLOs가 다중 해상도 추론(multi-resolution inference)을 수행하는데 있어서 불완전함을 드러내는 것입니다. 특히, MLOs는 새로운 주파수에 대한 추론을 실패하며, 훈련 당시의 해상도와 다른 해상도에서는 오류를 범할 수 있습니다. 이를 해결하기 위해 연구진은 다중 해상도 훈련(multi-resolution training) 프로토콜을 제안하였으며, 이는 저해상도와 고해상도의 데이터 세트를 동시에 활용하는 방식입니다.

- **Performance Highlights**: 제안된 다중 해상도 훈련 접근법은 훈련 비용을 크게 증가시키지 않으면서도 MLOs의 전반적인 성능을 향상시킬 수 있습니다. 연구 결과, MLOs는 훈련 데이터 해상도에 의존적으로 동작하며, 적절히 다양한 해상도를 아우르는 훈련을 통해 다중 해상도 일반화(multi-resolution generalization)가 가능하다는 것을 보여주었습니다. 이 연구는 MLOs의 설계 및 적용에 대한 중요한 통찰을 제공하며, 실시간 응용에서의 잠재적 활용을 제시합니다.



### Control-Augmented Autoregressive Diffusion for Data Assimilation (https://arxiv.org/abs/2510.06637)
- **What's New**: 이 논문은 Auto-Regressive Diffusion Models(ARDM)에서 가이드를 개선하는 새로운 방법론을 제안합니다. 저자들은 경량화된 컨트롤러 네트워크를 추가하여 사전 학습된 ARDM을 보강하고, 이를 통해 예측 오류를 효과적으로 줄이는 방법을 제시합니다. 이 접근법은 비선형 역학을 모델링해야 하는 데이터 동화(data assimilation, DA) 문제에서 특히 유용하다는 점에서 차별화됩니다.

- **Technical Details**: 제안된 방법은 사전 학습된 ARDM의 생선 동역학(generative dynamics)에 학습된 제어 메커니즘(control mechanism)을 통합합니다. 방법론은 입력에 따라 diffusion 과정을 안내하도록 설계되었으며, 이는 전통적인 데이터 동화 기법에 비해 높은 효율성을 자랑합니다. 저자들은 이러한 구조가 다양한 관찰 조건 하에서도 안정성과 정확성을 높인다고 주장합니다.

- **Performance Highlights**: 저자들은 제안된 방법이 두 개의 고전적인 PDE 데이터셋과 여섯 가지 관찰 조건에서 네 가지 최신 기법(state-of-the-art)보다 우수한 성능을 보여주었다고 주장합니다. 이 연구는 계산 집약적인 최적화 절차를 피함으로써 실시간 예측을 보다 정확하고 안정적으로 할 수 있도록 합니다. 또한, 저자들은 코드를 공개하여 연구의 재현성을 높일 계획을 밝히고 있습니다.



### StruSR: Structure-Aware Symbolic Regression with Physics-Informed Taylor Guidanc (https://arxiv.org/abs/2510.06635)
- **What's New**: 이 논문에서는 구조 인식 기호 회귀 프레임워크(StruSR)를 제안하여, 학습된 Physics-Informed Neural Networks (PINNs)를 활용하여 시계열 데이터에서 구조화된 물리적 우선 정보를 추출합니다. 특히, 이 프레임워크는 지역 Taylor 전개를 통해 유도된 구조 정보를 활용하여 기호 표현의 진화를 안내합니다. 기존의 기법들은 데이터 적합성에 주로 초점을 맞추는 반면, 이 방법은 물리적 원칙과 구조적 정렬이 동시에 이루어지도록 합니다.

- **Technical Details**: StruSR은 기호 회귀 과정에서 PINNs가 인코딩한 구조적 통찰력을 활용합니다. 이를 위해, 학습된 PINN의 출력에 대해 지역 Taylor 전개를 수행하며, 이는 도메인 지식을 기호 검색 과정에 주입하는 기반이 됩니다. 민감도 점수를 사용하여 각 서브 트리의 기여도를 정량화하고 기호 표현의 진화를 제어하는 QAQ(quantitative assessment of question) 메커니즘을 도입합니다.

- **Performance Highlights**: StruSR은 전통적인 기준보다 수렴 속도, 구조적 충실도 및 기호 표현의 해석 가능성을 개선하는 것으로 나타났습니다. 벤치마크 PDE 시스템을 기반으로 실시한 실험 결과, 이 프레임워크는 물리적 잔차와 기호적 정렬을 동시에 최적화하여, 기계 발견 모델들이 보다 해석 가능하고 물리적으로 의미 있는 표현을 생성하는 데 기여합니다.



### Three Forms of Stochastic Injection for Improved Distribution-to-Distribution Generative Modeling (https://arxiv.org/abs/2510.06634)
- **What's New**: 이 논문은 임의의 데이터 분포 간의 변환 모델링에 대한 도전과제를 다룹니다. 특히, limited samples를 통해 학습해야 하는 경우, flow matching이 sparse supervision으로 인해 실패한다는 점을 발견했습니다. 이를 해결하기 위해, 안전하고 계산 효율적인 방법을 제안하여 source 샘플과 flow interpolant에 불확실성을 주입했습니다. 이 방식은 과학적 시뮬레이션 및 데이터 생성 품질을 크게 개선하는 결과를 보였습니다.

- **Technical Details**: 논문에서는 flow matching을 사용하여 서로 다른 데이터 분포 간의 변환을 모델링합니다. 전통적인 noise-to-data 접근에서 벗어나, data-to-data 상황에서의 적용을 연구했습니다. 학습 과정에서, 두 가지 분포로부터 finite samples를 사용하는 경우, sparse한 supervision 문제를 해결하기 위해 Gaussian noise를 주입하고 flow interpolant를 조정하는 세 가지 형태의 stochasticity를 제안합니다. 이러한 기법들은 이론적 분석을 통해 sparsity를 완화하고 일반화를 개선하는 데 도움이 됩니다.

- **Performance Highlights**: 개발된 알고리즘은 생물학, 방사선학 및 천문학을 아우르는 5가지 고차원 이미지 데이터셋에서 검증되었습니다. 실험 결과는 Fréchet Inception Distance(FID) 기준으로 기존 baseline을 평균 9점 넘게 초과하는 성과를 보였고, 특히 flow matching이 높은 차원에서 sparsity 문제로 인해 성능이 떨어질 때 이를 완화할 수 있었습니다. 이를 통해, 분포 간의 변환 문제에 대한 실제적인 해결책을 제시하여 데이터 기반 모델링의 발전에 기여하고 있습니다.



### Chem-NMF: Multi-layer $α$-divergence Non-Negative Matrix Factorization for Cardiorespiratory Disease Clustering, with Improved Convergence Inspired by Chemical Catalysts and Rigorous Asymptotic Analysis (https://arxiv.org/abs/2510.06632)
- **What's New**: 이번 연구에서는 Non-Negative Matrix Factorization (NMF)에 물리 화학적 관점을 적용하여 다층 구조에서의 수렴(convergence) 분석을 수행하는 Chem-NMF라는 새로운 접근 방식을 제안합니다. 이 연구는 에너지 장벽(energy barrier) 개념을 도입해 다층 α-발산(α-divergence) NMF의 수렴 행동을 이론적으로 분석하는 최초의 사례로, 이는 최적화의 유연성을 높이며 안정적인 수렴을 보장합니다. 실험 결과 제안된 알고리즘이 생물 의학 신호에서 5.6%의 클러스터링 정확도를 개선함을 보여줍니다.

- **Technical Details**: NMF는 고차원 데이터를 해석 가능한 저차원 구성 요소로 분해하는데 사용되는 기법입니다. 연구에서는 α-발산을 활용하여 NMF의 유연성을 개선하고, 다층 NMF의 경우 매트릭스들이 계단식으로 구성되어 문제를 해결합니다. 각 단계는 반복적으로 근사하여 마지막 모델에 도달하는 구조로, 이를 통해 각 특성의 중요성을 세분화한 활성화 맵(activation map)을 생성합니다.

- **Performance Highlights**: 제안된 Chem-NMF는 생물 의학 신호와 얼굴 이미지에서 기존의 α-NMF보다 각각 5.6% 및 11.1%의 클러스터링 정확도를 향상시키는 결과를 도출했습니다. 이러한 성능 개선은 에너지 장벽 분석과 최적화를 통합함으로써 이루어졌으며, 이는 모델의 수렴 안정성을 높이는 동시에 데이터 클러스터링의 효과iveness를 극대화합니다.



### AI-Driven Forecasting and Monitoring of Urban Water System (https://arxiv.org/abs/2510.06631)
- **What's New**: 이 논문은 도시의 하수도 및 오수 파이프라인에서 누수 탐지를 위한 통합 AI 및 원격 센서 프레임워크를 제안합니다._sparse sensor_를 사용해 실시간 흐름과 깊이 데이터를 수집하고, HydroNet이라는 특별한 모델을 활용하여 보다 정밀한 예측을 제공합니다. 이 시스템은 현실 세계의 캠퍼스 하수 네트워크 데이터셋을 통해 효과적인 예측 성능을 달성했습니다.

- **Technical Details**: 본 시스템은 3단계로 구성되어 있습니다: 첫째, _sparse remote sensors_를 통해 주요 네트워크 위치에서 실시간 흐름 및 깊이 데이터를 수집합니다; 둘째, 이 데이터를 _graph-structured_ 형식으로 결합하여 HydroNet을 통해 시공간 모델링을 수행합니다; 셋째, 모델이 정상적인 유압 패턴을 정확히 예측하도록 학습합니다. HydroNet은 방향 그래프를 기반으로 하며, 파이프라인 속성을 통합하여 메시지를 전달합니다.

- **Performance Highlights**: 평가 결과, HydroNet은 모든 메트릭에서 기존 모델들을 초과 달성하여 높은 정확성을 입증했습니다. 수집된 데이터는 강력한 예측 유틸리티를 제공하여 이상 탐지 및 예측을 지원합니다. 이 접근법은 도시의 지하 수자원 관리 개선에 강력한 기초를 제공합니다.



### POME: Post Optimization Model Edit via Muon-style Projection (https://arxiv.org/abs/2510.06627)
- **What's New**: "Post-Optimization Model Edit (POME)"를 소개하며, 이 알고리즘은 사전 훈련(pretrained) 및 미세 조정(fine-tuning)된 체크포인트만을 사용하여 대규모 언어 모델의 성능을 향상시킵니다. 추가 데이터나 훈련 없이도 성능을 개선할 수 있다는 점이 독창적입니다. 특히, POME는 파라미터의 업데이트 방향을 더 균일하게 분포시키고 부정적인 잡음을 줄이는 방법으로 잘 알려진 "truncated singular value decomposition (SVD)" 기법을 활용합니다.

- **Technical Details**: POME는 업데이트 벡터인 ΔW를 통해 파라미터의 방향성을 조정합니다. 이 방법은 Muon 최적화기에서 유래된 것으로, 파라미터 업데이트 시 직교성을 보장하는 방식입니다. 무작위적으로 기존 매트릭스를 수정하는 대신, 최적화가 완료된 후에 단 한 번의 변환을 적용하여 훈련 과정과 완전히 분리되므로, 훈련 프레임워크에 대한 변경 사항 없이도 사용이 가능합니다.

- **Performance Highlights**: POME는 EMS8K와 코드 생성 영역에서 각각 평균 +2.5%, +1.0%의 성능 향상을 보여주었습니다. 7B에서 72B 모델까지의 다양한 스케일에서 광범위한 적용성을 나타내며, RLHF 최적화 모델에서도 일관된 성능 향상을 기록합니다. 이 방법은 훈련 시간의 과부하 없이 파라미터 조정 문제를 간소화할 수 있는 실용적인 해결책으로 자리잡을 것으로 기대됩니다.



### DPA-Net: A Dual-Path Attention Neural Network for Inferring Glycemic Control Metrics from Self-Monitored Blood Glucose Data (https://arxiv.org/abs/2510.06623)
Comments:
          14 pages, 10 figures

- **What's New**: 이 논문에서는 자가 혈당 측정(SMBG) 데이터를 바탕으로 혈당 프로필(Ambulatory Glucose Profile, AGP) 지표를 추정하는 듀얼 경로 주의 신경망(DPA-Net)을 제안합니다. DPA-Net은 sparse 한 SMBG 관측치에서 CGM(CONTINUOUS GLUCOSE MONITORING)과 유사한 경로를 재구성하는 공간-채널 주의 경로와 AGP 지표를 직접 예측하는 다중 스케일 ResNet 경로의 두 가지 보완 경로를 통합합니다. 이를 통해 낮은 빈도의 혈당 측정을 유지하면서 AGP 지표의 추정에서 높은 정확도를 달성하고자 합니다.

- **Technical Details**: DPA-Net 모델은 SMBG 데이터로부터 TIR(정상 범위 내 시간), TBR(정상 범위 아래 시간), TAR(정상 범위 위 시간) 등 3개의 AGP 지표를 예측하는 것을 목표로 합니다. 모델은 두 개의 경로로 구성되는데, 한 경로는 sparse 한 SMBG 데이터를 활용하여 CGM 곡선을 재구성하고, 또 다른 경로는 glycemic control 지표를 직접 예측합니다. 두 경로 간의 정렬 메커니즘을 구축하여 편향을 줄이고 과적합(overfitting)을 완화하며, SMBG 샘플링 포인트를 선택하기 위한 능동 점(selector)을 개발하여 현실적인 데이터에서 모델을 훈련합니다.

- **Performance Highlights**: 실험 결과, DPA-Net은 실제 데이터 세트에서 낮은 에러율과 강력한 정확성을 달성하여 체계적 편향을 줄이는 것으로 나타났습니다. SMBG 데이터로부터 AGP 지표를 직접 추정하는 머신러닝 프레임워크로서, CGM에 접근할 수 없는 환경에서 유용한 의사결정 지원 도구를 제공합니다. DPA-Net은 임상적 실용성과 공공 건강 영향을 모두 고려한 비용 효율적 대안으로 자리매김할 수 있을 것입니다.



### The Framework That Survives Bad Models: Human-AI Collaboration For Clinical Trials (https://arxiv.org/abs/2510.06567)
- **What's New**: 이번 연구는 인공지능(AI)이 임상 시험에서 어떻게 효과적으로 적용될 수 있는지를 분석했습니다. 특히, X-ray 이미지를 기반으로 한 질병 평가에서 AI와 인간 판독자의 상호 작용을 비교하여, AI가 임상 연구에 미치는 영향을 평가했습니다. 연구 결과, AI를 보조 리더(AI-SR)로 사용할 때 가장 유리한 결과를 보여 AI의 활용 가능성을 확인했습니다.

- **Technical Details**: 연구에서 사용된 방법론은 X-ray 데이터를 기반으로 AI 모델의 성능을 테스트하는 것이었습니다. 두 개의 임상 시험 MEASURE I와 PREVENT에서 환자 이미지를 분석하여, 총 점수(mSASSS)를 계산하고 예측 정확도를 평가했습니다. AI 모델은 nn-UNet과 mask-RCNN의 앙상블을 사용하여 척추 단위를 분할하고 mSASSS를 예측했습니다.

- **Performance Highlights**: 연구 결과, AI를 이용한 평가가 임상 시험의 시간과 비용을 줄일 수 있으며, 다양한 모델에서 발생할 수 있는 오류에도 불구하고 신뢰할 수 있는 결과를 유지할 수 있음을 보여주었습니다. AI 기술을 통해 질병 추정의 신뢰성을 보장하면서도 치료 효과 추정치와 결론을 유지할 수 있다는 점이 강조되었습니다.



### The Markovian Thinker (https://arxiv.org/abs/2510.06557)
- **What's New**: 이번 논문에서는 Markovian Thinking이라는 새로운 패러다임을 제안합니다. 이 방법은 정책(policy)이 일정 크기의 상태(state)에 따라 추론을 진행하도록 하여, 사고의 길이를 맥락의 크기와 분리합니다. 이를 통해 Delethink라는 강화학습(RL) 환경을 구성하여, 고정 크기의 청크로 사고를 구조화하면서도, 긴 사고를 더 효율적으로 처리할 수 있음을 보여줍니다.

- **Technical Details**: Delethink 환경은 각 청크의 경계에서 환경을 리셋하고 짧은 캐리오버로 프롬프트를 재초기화합니다. 에이전트는 강화학습을 통해 각 청크의 끝에서 연속성이 보장되는 텍스트 상태를 작성하도록 학습합니다. 이러한 과정은 결국 상수 메모리와 선형 계산(linear compute)을 가능하게 하여 기존의 표준 RL 환경에 비해 획기적인 성과를 나타냅니다.

- **Performance Highlights**: 트레이닝 된 R1-Distill 1.5B 모델은 8K 토큰 청크 내에서 사고를 진행하며 최대 24K 토큰 사고를 수행했습니다. 이는 24K 예산으로 훈련된 LongCoT을 능가하거나 맞먹는 성과를 보여줍니다. 실험 결과는 Delethink가 긴 사고를 수행하면서도 연산 비용과 메모리 사용에서 효율적임을 나타내며, 이로 인해 더 나은 성능을 보이는 것으로 분석되었습니다.



### Incoherence in goal-conditioned autoregressive models (https://arxiv.org/abs/2510.06545)
- **What's New**: 이 논문에서는 자기 회귀 모델의 단순 목표 조건화로부터 파생된 강화 학습 정책의 구조적 문제인 incoherence(불일치)의 개념을 수학적으로 조사합니다. 특히, 모델을 자신의 행동에 대해 재훈련하는 과정, 즉 오프라인에서 학습한 정책을 온라인 RL로 미세 조정하는 과정을 중심으로 연구하며, 이는 불일치를 감소시키고 수익을 향상시키는 데 기여한다고 입증합니다.

- **Technical Details**: 이 논문에서는 reinforcement learning(RL)을 inference 문제로 재구성하여, 최적 정책을 찾기 위한 명시적인 탐색 대신 행동이나 경로에 대한 생성 모델을 구성하고 이를 목표에 조건화하여 정책을 도출합니다. 정책은 주어진 상태에서 향후 행동의 확률 분포를 반영해야 하며, 이 과정에서 두 가지 질문의 차이를 강조합니다: (1) 주어진 보상을 달성하는 행동 선택 vs. (2) 자기 회귀적으로 선택되는 행동으로 보상을 달성하는 방법.

- **Performance Highlights**: 정책의 재훈련 과정에서 불일치를 제거하기 위한 세 가지 접근 방식을 제시하며, 그 중 첫 번째는 자신의 경로에 대한 정책 재조정을 포함합니다. 두 번째 접근은 온도 매개변수의 감소로, 이는 KL-정규화된 RL의 엔트로피 규제를 조작하는 방식으로 이루어집니다. 마지막으로, 보상을 통해 행동의 후방을 반영하는 것을 포함하는 방법을 설명하며, 이는 이론적으로 MDP의 수정 없이 수행됩니다.



### Scalable Policy-Based RL Algorithms for POMDPs (https://arxiv.org/abs/2510.06540)
Comments:
          36 pages, 3 Figures, Accepted at NeurIPS 2025

- **What's New**: 이 논문은 부분 관찰 강화를 통한 학습 문제(PORL)를 슈퍼스테이트 MDP라는 유한 상태 마르코프 결정 과정으로 변환하여 최적 정책을 학습하는 새로운 접근법을 제시합니다. 기존 POMDP 문제에서의 컴퓨테이셔널 차별성을 해결하기 위해, 이 연구는 강화학습 알고리즘을 활용하여 비선형 함수 근사를 통해 성능 보장을 제시합니다. 특히, 이 연구는 표준 TD 학습을 활용하여 최적 가치 함수와 초기 POMDP 가치 함수 간의 관계를 보장하는 이론적 개선을 제안합니다.

- **Technical Details**: 논문은 POMDP 모형을 슈퍼스테이트 MDP로 변환함으로써 연산적 복잡성을 줄이는 방법을 설명합니다. 이 연구는 전통적인 TD 학습 기법을 사용하여 연속 관측 이력을 기반으로 학습하여, POMDP의 최적 정책을 근사하는 새로운 알고리즘을 제안합니다. 특히, 저자들은 TD 학습의 수렴 보장을 제공하여 샘플링 매칭 문제를 해결하고, 전통적 방법보다 더 낮은 계산 복잡성으로 실용적인 성과를 보여줍니다.

- **Performance Highlights**: 이 연구의 주요 성과는 슈퍼스테이트 MDP에서의 성능 경계 성립을 보여주는 것입니다. 이는 기존 방법보다 높은 이론적 보장을 포함하며, POMDP 문제에 대한 구체적인 유한 시간 경계를 최초로 제시합니다. 또한 함수 근사 설정에서 정책 최적화 알고리즘의 성능 경계를 확장하여 대규모 상태 공간에서도 유용한 확장성을 제공합니다.



### Wide Neural Networks as a Baseline for the Computational No-Coincidence Conjectur (https://arxiv.org/abs/2510.06527)
- **What's New**: 이 논문에서는 무작위로 초기화된 신경망이 넓이가 크고 하이퍼파라미터가 자연스럽게 선택된 경우, 활성화 함수가 비선형이고 평균이 제로인 경우에 거의 독립적인 출력을 가진다고 주장합니다. 또한, 이러한 신경망을 '서로 겹치지 않는 추측'(no-coincidence conjecture)의 후보로 제안합니다. 이는 AI의 해석 가능성의 한계를 측정하고자 하는 이론적 시도를 포함합니다.

- **Technical Details**: 신경망은 각 레이어에서 가중치 행렬과 편향 벡터로 매개화됩니다. 메인 결과인 정리 4.1은 적절한 하이퍼파라미터와 깊이 및 넓이가 적절한 비율로 증가할 때, 출력의 확률 분포가 표준 가우시안 분포에 접근한다고 증명합니다. 특히, 제로 평균 기준은 tanh 함수에서 만족되지만, ReLU나 GeLU 함수에서는 만족되지 않음을 보여주고 있습니다.

- **Performance Highlights**: 무작위로 초기화된 신경망은 대체로 가우시안 프로세스로 수렴하며, 그 특성에 따라 공분산 행렬로 행동을 정의할 수 있습니다. 논문에서는 신경망의 목적은 무한한 너비의 한계에서 독립적인 출력을 만드는 것이며, 실제 신경망이 유한한 너비에서 '거의 가우시안'의 특성을 가지는 것을 강조합니다. 이로써, 신경망의 내부 구조를 보다 잘 이해할 수 있는 통찰을 제공합니다.



### Text-to-Image Models Leave Identifiable Signatures: Implications for Leaderboard Security (https://arxiv.org/abs/2510.06525)
Comments:
          Accepted at Lock-LLM Workshop, NeurIPS 2025

- **What's New**: 이 논문은 Generative AI 리더보드에서 모델의 능력을 평가하던 기존 방식이 조작에 취약하다는 점을 강조하고 있습니다. 특히 text-to-image (T2I) 리더보드에서 deanonymization이 훨씬 더 용이하다는 것을 발견하였습니다. 논문은 280개의 프롬프트와 19개의 다양한 모델을 활용하여, CLIP 임베딩 공간에서 실시간 분류를 통해 모델 식별의 높은 정확성을 보여주고 있습니다.

- **Technical Details**: 해당 연구에서는 150,000개 이상의 생성 이미지를 사용하여, T2I 리더보드에서 생성된 모델을 deanonymize하는 방법을 제안합니다. 이는 각 모델에서 생성된 이미지를 CLIP 임베딩 공간에 내재화하여 비교하는 방식으로, 특히 CLIP과 같은 고차원 곡면을 활용하여 모델 간의 차별을 감지합니다. 논문은 또한 프롬프트 레벨의 분리 가능성을 측정하기 위한 메트릭을 도입하여, 완전한 분리가 가능한 프롬프트를 식별합니다.

- **Performance Highlights**: 실험 결과, 생성된 이미지를 통해 T2I 모델 간의 deanonymization 정확도가 매우 높게 나타났습니다. 상응하는 프롬프트에 대해 모델들이 형성하는 클러스터는 확연하게 구분됩니다. 이 연구는 T2I 모델이 투표 기반 리더보드에서 특정 공격에 대해 더 큰 보안 위협을 내포하고 있음을 보여주며, 조작 방지를 위한 강력한 방어 체계의 필요성을 강조합니다.



### A Median Perspective on Unlabeled Data for Out-of-Distribution Detection (https://arxiv.org/abs/2510.06505)
- **What's New**: 이번 논문에서는 Medix라는 새로운 프레임워크를 제안하여, unlabeled wild data에서 out-of-distribution (OOD) 샘플을 식별하는 방법을 소개합니다. Medix는 중앙 경향성을 제공하는 median 연산을 사용하여 노이즈와 아웃라이어에 강인한 OOD 탐지 메커니즘으로 기능합니다. 이 방법은 기존의 기술들과는 달리 라벨이 없는 Wild 데이터를 효과적으로 활용하면서도 이론적인 오류 보장을 제공합니다.

- **Technical Details**: 논문에는 Medix의 이론적인 기초와 OOD 탐지를 위한 median 기반 최적화 프레임워크가 자세히 설명되어 있습니다. Medix는 InD 및 OOD 샘플이 혼합된 데이터에서 아웃라이어를 식별할 수 있도록 설계되었으며, 특정 비율 이하의 OOD 비율에서 소음의 영향이 관리 가능하다는 것을 보여줍니다. 학습 목표는 InD 데이터와 식별된 아웃라이어를 사용하여 강건한 OOD 분류기를 훈련하는 것입니다.

- **Performance Highlights**: Medix는 CIFAR-100에서 기존의 KNN+와 비교하여 평균 40.98%의 FPR95 개선을 달성하며, 다른 20개 기준선과 비교해도 우수한 성능을 보였습니다. 본 연구 결과는 Medix의 이론적인 주장과 실험 결과를 통해 뒷받침되며, 아울러 라벨이 없는 데이터에서 효과적인 아웃라이어 추출을 통해 낮은 오류율을 확인할 수 있었습니다.



### ATLO-ML: Adaptive Time-Length Optimizer for Machine Learning -- Insights from Air Quality Forecasting (https://arxiv.org/abs/2510.06503)
- **What's New**: 이 논문은 ATLO-ML이라는 적응형 시간 길이 최적화 시스템을 소개하며, 사용자 정의된 출력 시간 길이에 따라 최적의 입력 시간 길이와 샘플링 속도를 자동으로 결정합니다. 기존의 고정된 시간 길이 대신, 시간 시리즈 데이터 전처리에 유연한 접근 방식을 제공합니다. 이 시스템은 시간 내 변동성에 대한 예측 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: ATLO-ML은 두 가지 주요 데이터셋, GAMS-dataset과 데이터 센터에서 수집한 독점적인 데이터를 이용해 검증됩니다. 이 시스템은 다양한 시간 민감 애플리케이션에 대해 일반화할 수 있는 잠재력을 보여줍니다. 입력 매개변수의 최적화를 통해 머신러닝(ML) 작업흐름에서 시간 관련 문제를 해결하는 강력한 솔루션을 제공합니다.

- **Performance Highlights**: 결과에 따르면, 최적화된 시간 길이와 샘플링 속도를 활용했을 때 머신러닝 모델의 정확성이 고정된 시간 길이보다显著하게 개선됩니다. 이는 ATLO-ML이 머신러닝 예측 성능을 개선하는 데 중요한 역할을 한다는 것을 시사합니다. 따라서, ATLO-ML은 시간 기반 예측의 많은 분야에서 효과적인 도구가 될 것입니다.



### GUIDE: Guided Initialization and Distillation of Embeddings (https://arxiv.org/abs/2510.06502)
- **What's New**: 본 논문에서는 모델 품질을 개선하면서도 비용을 증가시키지 않는 알고리즘 효율성 기술에 대해 다룹니다. 새로운 접근법으로 제안된 GUIDE (Guided Initialization and Distillation of Embeddings)는 학생 모델이 교사 모델의 파라미터를 통해 초기화되도록 유도하여 훈련 과정에서 더 많은 정보를 추출할 수 있게 합니다. 정전적 지식 증류 방식과는 달리, GUIDE는 손실 함수에 변화를 주지 않으며, 교사가 데이터셋을 라벨링할 필요도 없습니다.

- **Technical Details**: GUIDE는 훈련된 교사 모델의 파라미터를 활용하여 학생 모델을 직접 초기화하는 간단한 방법입니다. 이 방법은 학생 모델과 교사 모델이 동일한 데이터셋에서 훈련되고, 동일한 맥락 길이를 가지는 전통적인 변환기(transformer)를 기반으로 합니다. GUIDE를 통해 학생 모델은 교사의 예측뿐만 아니라 교사의 파라미터를 활용하여 초기화됩니다.

- **Performance Highlights**: GUIDE를 적용함으로써, 대규모 학생 모델(400M - 1B 파라미터)은 교사-학생 품질 격차를 25-26% 줄일 수 있었습니다. 또한 GUIDE의 단독 적용은 기존의 지식 증류만 적용했을 때보다 훨씬 뛰어난 모델 품질을 보여주었으며, 훈련 및 추론 오버헤드가 전혀 없기 때문에 모델 품질 향상이 거의 무료로 이루어질 수 있습니다.



### Valid Stopping for LLM Generation via Empirical Dynamic Formal Lif (https://arxiv.org/abs/2510.06478)
- **What's New**: 본 논문에서는 언어 모델 생성 중의 중지 결정을 위해 Anytime-valid sequential testing을 적용한 Sequential-EDFL(Empirical Dynamic Formal Lift)을 소개합니다. 이 접근법은 정보 증가량 정보 수집을 추적하며, 정해진 시간에 관계없이 오류 제어를 보장할 수 있는 self-normalized empirical-Bernstein e-processes를 사용합니다. Sequential-EDFL은 22-28%의 생성을 줄이는 동시에 오류를 제어하고 있으며, 안전 기준이 필요한 도메인에서 검증 부담을 줄이는 첫 번째 단계 필터 역할을 합니다.

- **Technical Details**: 우리는 skeleton을 기반으로 정보 증가를 측정하며, 전체 모델 예측과 고의적으로 약화된 기준 사이의 로그-우도 비율을 사용하여 정보 증가량을 정의합니다. 중요한 도전 과제는 조건부 기대값과 분포 변화에 대한 조정이 필요하며, 지수 과정을 사용해 이러한 문제를 해결합니다. 자동화된 skeleton 생성 방법과 진단 체크리스트를 통해, 우리는 EDFL의 적용을 현실화합니다.

- **Performance Highlights**: 연구에서 프로토타입은 6개 벤치마크 데이터셋(GSM8K, HotpotQA 등)에서 기존의 순차적 기준에 비해 생성량을 22-28% 감소시켰고, 이와 동시에 약 12%의 추가 계산 부하로 오류 제어를 유지했습니다. 또한, hybrid correctness gate를 추가하여 문장 경계를 강제하며 사실성을 개선하기 위한 방법을 제시합니다. 그러나 EDFL은 사실상의 정확성을 보증하지 않으며, 특정 도메인에 대한 검증 절차는 여전히 중요합니다.



### Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin (https://arxiv.org/abs/2510.06477)
- **What's New**: 이번 연구에서는 attention sinks와 compression valleys라는 두 가지 현상이 대형 언어 모델에서 어떻게 연결되는지를 밝힙니다. 이 두 가지 현상은 독립적으로 연구되었으나, 우리는 이들이 잔여 스트림(residual stream) 내에서의 대규모 활성화(massive activations) 형성과 관련이 있음을 보여줍니다. 연구를 통해 대규모 활성화가 표현 압축을 유도한다는 이론적 증거를 제시하며, 이로 인해 두 현상이 동시에 발생함을 실험적으로 확인했습니다.

- **Technical Details**: 연구의 근본적인 메커니즘은 대규모 활성화가 attention sinks와 compression valleys의 두 현상을 생성함에 있다는 것입니다. 연구는 Transformer 기반 대형 언어 모델에서 토큰 처리 과정을 세 가지 단계로 나누어 설명합니다: 초기 레이어에서는 넓은 혼합(broad mixing), 중간 레이어에서는 제한된 혼합을 통한 압축(compressed computation), 그리고 후반 레이어에서는 선택적인 정제(selective refinement)가 이루어진다고 주장합니다. 이 프레임워크는 embedding 작업과 generation 작업의 최적 깊이를 설명하는 데 기여합니다.

- **Performance Highlights**: 우리는 410M에서 120B 파라미터에 이르는 여러 모델에서 실험을 통해 대규모 활성화가 두 현상 모두에 미치는 영향을 검증했습니다. 더불어, 타겟형 ablative 연구를 통해 대규모 활성화를 제거함으로써 압축과 attention sink가 감소한다는 인과관계도 확인했습니다. 이 연구는 Transformer가 컴퓨테이션을 어떻게 조직화하는지를 이해하는 데 중요한 통찰을 제공하며, 각 작업의 특성에 따라 최적의 처리 깊이가 달라지는 이유를 풍부하게 설명합니다.



### How NOT to benchmark your SITE metric: Beyond Static Leaderboards and Towards Realistic Evaluation (https://arxiv.org/abs/2510.06448)
- **What's New**: 이 연구는 Transferability estimation metrics의 현재 기준을 비판적으로 분석하며, 잘못된 벤치마크 구조가 기존 메트릭의 성능을 과대평가하는 방식을 설명합니다. 연구자들은 특히 STATIC performance hierarchy와 비현실적인 모델 공간이 어떻게 문제를 일으키는지 보여주며, 실질적으로 더 신뢰할 수 있는 평가를 위한 벤치마크 구축을 위한 권고안을 제시합니다.

- **Technical Details**: Transferability estimation의 개념은 다양한 사전 훈련된 모델 중에서 Fine-tuning 후 최상의 성능을 발휘할 모델을 예측하는 것입니다. Source Independent Transferability Estimation (SITE) 메트릭은 전체적인 구조 설계를 필요로 하며, 모델의 피쳐 표현과 타겟 데이터 세트의 관계를 평가합니다. 여러 SITE 메트릭은 비슷한 관점에서 발전해왔으며, 예를 들어 LogME는 최대 레이블 마진화 가능성을 수식으로 표현합니다.

- **Performance Highlights**: 연구에 따르면, 기존의 메트릭들보다 간단한 정적 랭킹 휴리스틱이 성능 평가에서 더 높은 신뢰성을 발휘할 수 있으며, 이는 현재 벤치마크가 가진 결점을 강조합니다. SITE 메트릭을 평가하기 위해 설정된 일반적인 기준과 모델 간 비교의 비효율성을 보여주며, 결국 연구자들은 조사된 메트릭의 실용성을 강화하는 조치를 취할 것을 권장합니다.



### Context-Aware Inference via Performance Forecasting in Decentralized Learning Networks (https://arxiv.org/abs/2510.06444)
Comments:
          17 pages, 12 figures; appeared in ADI (October 2025)

- **What's New**: 이 논문에서는 분산 학습 네트워크 내에서 예측 성능을 예측하는 모델을 개발하였습니다. 이를 통해 상황에 따라 더 정확한 모델에 높은 가중치를 부여하는 'context-awareness'를 확보할 수 있습니다. 성능 예측 워커를 추가해 네트워크 추론의 정확도를 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 본 연구는 예측 모델의 성능을 예측하는 모델을 분산 학습 네트워크에서 설계하는 데 주목합니다. 네트워크는 다양한 알고리즘, 특성 및 비공식 데이터 세트와 함께 사용할 수 있는 모델을 통합합니다. 이 모델은 학습 네트워크의 설계와 목표 변수를 기반으로 하여 구성 요소와 특성을 최적화합니다.

- **Performance Highlights**: 성능 예측 모델은 예측 손실이나 후회(regret)와 같은 지표를 통해 예측 정확도를 개선하는 데 기여합니다. 최적의 특성 집합 및 교육 에포크 수에 따라 모델 성능이 민감하게 반응합니다. 예측 조합을 위한 성능 예측은 분산 환경에서뿐만 아니라 다양한 상황에서도 유용할 수 있습니다.



### Bayesian Optimization under Uncertainty for Training a Scale Parameter in Stochastic Models (https://arxiv.org/abs/2510.06439)
- **What's New**: 이 논문은 불확실성 하에서 하이퍼파라미터 튜닝을 위한 새로운 베이지안 최적화 프레임워크를 제안합니다. 이는 스토캐스틱 모델에서 스케일 또는 정밀도 유형의 파라미터를 최적화하는 데 중점을 둡니다. 제안된 방법은 기본 무작위 변수를 위한 통계적 대리 변수를 사용하여 기대 연산자의 분석적 평가를 가능하게 합니다. 추가적으로 랜덤 획득 함수의 최적화기를 위한 닫힌 형태의 표현식을 유도하여 각 반복의 계산 비용을 크게 줄입니다.

- **Technical Details**: 하이퍼파라미터 β를 특정 문제의 최적화로 설정하고, 이 β는 랜덤 변수의 분포를 제어합니다. 베이지안 최적화 접근법은 불확실성 하에서의 최적화를 위해 설계되었습니다. 이 방법의 핵심은 β에 따라 조건부인 랜덤 변수 s(ω)의 통계적 대리 변수를 구성하여 기대치가 분석적으로 계산될 수 있도록 하는 것입니다. 이로 인해 최적화는 훨씬 더 효율적으로 이루어질 수 있습니다.

- **Performance Highlights**: 제안된 방법은 전통적인 1차원 몬테카를로 기반 최적화 방식에 비해 데이터 포인트를 40배 더 적게 요구하며, 계산 비용을 최대 40배 감소시킵니다. 이는 실시간 예측 작업에 적합함을 보여주는 두 가지 수치 예제를 통해 효과가 검증되었습니다. 이러한 데이터 사용 및 계산 비용의 획기적인 개선은 하이퍼파라미터 튜닝의 실제 적용 가능성을 높입니다.



### Nearly Instance-Optimal Parameter Recovery from Many Trajectories via Hellinger Localization (https://arxiv.org/abs/2510.06434)
- **What's New**: 이번 연구에서는 다중 궤적(multi-trajectory) 설정에서 최적의 인스턴스(instance-optimal) 경계를 크게 확장하는 방법을 제시합니다. Hellinger localization 프레임워크를 통해 대규모 데이터의 효과적인 학습이 가능하다. 본 방법은 전통적인 단일 궤적(single-trajectory) 학습에서의 혼합 가정 없이도 성능을 극대화할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 연구팀은 먼저 경로 측정 수준에서의 제곱 Hellinger 거리(squared Hellinger distance)를 제어하고, 이를 바탕으로 절차적으로 롸이터 수정(Quadratic form in parameter space)하여 궤적 피셔 정보(trajectory Fisher information)로 가중치를 적용합니다. 이 과정은 여러 가지 일반적 조건 하에서도 전체 데이터 예산에 비례하여 인스턴스 최적 경계를 발생시킵니다.

- **Performance Highlights**: 저자들은 네 가지 다양한 사례 연구를 통해 프레임워크를 입증하였으며, 이를 통해 마르코프 체인, 비가우시안 노이즈 아래의 의존선형회귀, 비단조 활성화(non-monotonic activations)를 가진 일반화 선형 모델 및 선형 주의 시퀀스 모델에 대해 성능을 분석하였습니다. 모든 경우에서, 제시된 경계는 비대칭 정상성(asymptotic normality)에서의 인스턴스-최적 속도에 거의 가깝게 일치하며, 기존 방식에 비해 상당한 성능 향상을 보여주었습니다.



### Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting (https://arxiv.org/abs/2510.06419)
- **What's New**: 논문에서는 시간 시계열 (time series) 예측을 위한 단일 대형 모델 대신, 소규모의 여러 사전 훈련된 모델을 포트폴리오 형태로 구성하는 대안을 제안합니다. 이 방법론은 정확한 예측을 유지하면서도 학습과 추론 비용을 절감할 수 있는 효율적인 접근법입니다. 특정 도메인에 특화된 소형 모델을 구축하고 이를 통해 통합 예측을 수행함으로써, 전통적인 단일 대형 모델의 대안으로 자리 잡을 수 있음을 시사합니다.

- **Technical Details**: 저자들은 기본 모델을 사전 훈련한 후 메타데이터에 따라 세분화된 데이터에 대해 특화된 훈련을 통해 다양한 모델 포트폴리오를 구축하는 방법을 제안합니다. 이를 통해 전체 훈련 시간을 10배 단축시키고, 각 모델이 특정 서브셋에 특화된 예측을 가능하게 합니다. 제안된 포트폴리오 모델은 다양한 훈련 데이터 세트에서 사전 훈련된 동일한 아키텍처의 소규모 모델들로 구성되며, Chroma라는 시스템 이름으로 통칭됩니다.

- **Performance Highlights**: Chroma 시스템은 예측 정확도가 기존의 대형 모델과 유사하면서도 추론 비용을 크게 줄이는 성과를 보였습니다. 소규모 모델의 조합이나 선택을 통해 현재 상태의 최첨단 사전 훈련된 예측기와 경쟁할 수 있는 결과를 나타냅니다. 이 연구는 각 모델의 공 contributions을 명확히 하여, 예측 과정에서의 해석 가능성을 강화하는 데 기여합니다.



### The Effect of Label Noise on the Information Content of Neural Representations (https://arxiv.org/abs/2510.06401)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문은 기계 학습 모델 훈련에서 자주 발생하는 레이블 노이즈(label noise)가 숨겨진 표현(hidden representation) 질에 미치는 영향을 체계적으로 분석하였습니다. 정보 불균형(Information Imbalance)이라는 컴퓨테이션적으로 효율적인 프록시(proxies)를 통해, 네트워크 매개변수(parameter)의 수에 따른 정보 콘텐츠가 어떻게 변화하는지를 관찰했습니다. 이 연구는 오버파라미터화(overparameterized)된 네트워크가 레이블 노이즈에 강한 내성을 보인다는 중요한 통찰을 제공합니다.

- **Technical Details**: 연구에서 제안된 정보 불균형(II)은 두 개의 표현 공간 간의 상대 정보 콘텐츠를 측정하는 통계적 측정방법입니다. II는 특정 데이터 포인트와 그 가장 가까운 이웃 간의 거리 정보를 기반으로 하여, 서로 다른 표현 공간 간의 거리가 얼마나 잘 보존되는지를 측정합니다. 이 방법론은 숨겨진 표현(hidden representations)의 질을 평가하는 데 사용할 수 있으며, 전통적인 유사도 측정과는 다르게 비대칭성(asymmetry)을 통한 비교 평가가 가능합니다.

- **Performance Highlights**: MNIST 및 CIFAR-10과 같은 데이터셋에서 분석한 결과, 레이블 노이즈가 포함된 데이터셋에서 훈련된 네트워크의 숨겨진 표현은 오히려 더 많은 정보(content)를 제공한다는 사실을 발견했습니다. 오버파라미터화된 네트워크의 경우, 숨겨진 표현의 정보 콘텐츠는 중간 수준의 레이블 노이즈에서 거의 동일해집니다. 반면 임의 레이블로 학습된 표현은 정보 콘텐츠가 낮아 성능이 저하되는 경향이 있음을 보여주어, 레이블 노이즈가 숨겨진 표현의 질에 미치는 영향을 명확히 드러냈습니다.



### Geometry-Aware Backdoor Attacks: Leveraging Curvature in Hyperbolic Embeddings (https://arxiv.org/abs/2510.06397)
- **What's New**: 이 논문은 비유클리드(Non-Euclidean) 모델인 하이퍼볼릭(hyperbolic) 신경망의 백도어(Backdoor) 취약성을 분석합니다. 하이퍼볼릭 기하학의 독특한 특성이 백도어 공격 조건을 변화시키며, 기하학적으로 적응하는 새로운 트리거를 제안합니다. 특이하게도, 경계 근처에서 미세한 입력 변화가 점차적으로 큰 모델 표현 공간의 변화를 일으킬 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 Poincaré 구 모델을 기반으로 비유클리드 공간을 정의하며, 이 공간의 리우만 기하(Riemannian geometry) 기반 메트릭을 설명합니다. 하이퍼볼릭 신경망에서의 주요 작업으로는 접선 벡터를 다양체에 투영하는 exponential map과 벡터 연산을 위한 Möbius 덧셈이 포함됩니다. 이러한 수학적 모델은 표준 입력 공간 탐지기를 회피하고 고유의 기하학적 특성을 활용한 백도어를 설계하는 데 사용됩니다.

- **Performance Highlights**: 하이퍼볼릭 신경망에서의 백도어 공격 성공률은 경계 근처에서 증가하며, 기존의 탐지기가 약해지는 경향을 보입니다. 실험 결과는 제안된 공격 방식이 유클리드 기반의 방법에 비해 효과적임을 입증하며, 기하학적으로 적응하는 트리거가 설계 및 방어의 한계를 이해하는 데 중요한 통찰을 제공합니다. 이는 비유클리드 모델에서 관련된 보안·방어 방법론에 대한 새로운 논의를 촉발합니다.



### Making and Evaluating Calibrated Forecasts (https://arxiv.org/abs/2510.06388)
- **What's New**: 이 논문은 다중 클래스 예측 작업을 위한 완전 진실한(calibrated) 교정 측정을 소개합니다. Hartline et al. (2025)가 제안한 2진 예측을 넘어서는 일반화로, 이러한 측정이 다중 클래스 예측에서도 성립함을 입증합니다. 특히, 진실한 교정 측정은 마르코프 의사 결정 이론(decision theory)에서의 우위(rank)를 보장하는 점이 강조됩니다.

- **Technical Details**: 논문에서는 다중 클래스 문제를 다루기 위해, 일반적인 교정 측정 방법을 분석하고, 2진 문제에서  다중 클래스 문제로의 확장 시 진실성이 보존되는지를 연구합니다. 가장 흔한 방법은 confidence aggregation(신뢰도 집계) 방식으로, 이는 전통적인 교정 측정에 비해 진실 담보의 성질이 결여되어 있음을 발견했습니다. 반면, class-wise aggregation(클래스별 집계) 방법이 진실성을 보존함을 이론적으로 증명하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 진실한 교정 오류는 예측기 간의 결정적 우위를 유지하고, 하이퍼파라미터 선택(예: 구간 수)에 회복력이 있음을 확인하였습니다. 이는 여러 신경망 모델에 대해 평가되었으며, 진실한 교정 오류가 비진실 오류 측정에 비해 더욱 일관된 결과를 보임을 나타냅니다.



### Monte Carlo Permutation Search (https://arxiv.org/abs/2510.06381)
- **What's New**: 이번 논문에서는 일반 목적을 위한 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS) 알고리즘인 몬테카를로 순열 탐색(Monte Carlo Permutation Search, MCPS)을 제안합니다. MCPS는 GRAVE 알고리즘을 개선하여, 깊은 강화 학습(deep reinforcement learning)이 어려운 상황에서도 적용할 수 있습니다. 이 알고리즘의 핵심 원리는 노드의 탐색 항에 루트부터 노드까지의 모든 이동을 포함한 전체 플레이아웃(playouts)의 통계를 포함하는 것입니다.

- **Technical Details**: MCPS는 세 가지 통계 소스를 활용하여 GRAVE의 두 가지 통계 소스를 보완합니다. 첫 번째와 두 번째 통계는 GRAVE와 동일하며, 세 번째는 다양한 순서의 이동을 포함한 플레이아웃에 대한 통계입니다. MCPS는 통계 소스의 계수를 설정하기 위한 하이퍼파라미터(hyperparameter)가 필요 없으며, 나무의 조상 노드를 선택하는 데 사용되는 하이퍼파라미터에 대해서도 유사한 성능을 발휘합니다.

- **Performance Highlights**: MCPS는 다양한 게임에서 GRAVE보다 더 우수한 성과를 보였으며, 특히 이인용 게임에서 두드러진 성과를 냈습니다. 다인용 게임에서는 동등한 결과를 기록했는데, 이는 플레이어의 강력함에 따라 균형이 이루어져 있는 게임의 특성 때문입니다. 추상 코드(abstract codes)를 이동에 사용함으로써 MCPS와 GRAVE의 성능을 개선할 수 있다는 점도 주목할 만합니다.



### Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data (https://arxiv.org/abs/2510.06377)
Comments:
          preprint; under review

- **What's New**: 이 논문에서는 관계형 데이터를 처리하기 위해 새로운 Relational Transformer (RT) 아키텍처를 제안합니다. RT는 다양한 관계형 데이터베이스에서 사전 훈련되어 특정 작업이나 데이터셋에 대한 추가적인 튜닝 없이도 새로운 데이터셋과 작업에 직접 적용될 수 있습니다. 이는 관계형 데이터의 다양한 특성 때문에 기존의 아키텍처가 적용되기 어려웠던 문제를 해결합니다.

- **Technical Details**: RT 아키텍처는 (i) 테이블 및 열 메타데이터로 셀을 토큰화하고, (ii) 마스크 토큰 예측을 통해 사전 훈련되며, (iii) 열, 행, 기본-외래 키 링크에 대한 새로운 \textit{Relational Attention} 메커니즘을 활용합니다. 이러한 기능들은 RT가 다양한 관계형 데이터를 효과적으로 처리할 수 있게 합니다.

- **Performance Highlights**: RelBench 데이터셋에서 사전 훈련된 RT는 단일 순전파 (forward pass)로 이진 분류 작업에서 22M 파라미터 모델을 사용하여 94%의 AUROC를 기록하며, 이는 27B LLM이 기록한 84%보다 높은 성능입니다. 또한, 미세 조정을 통해 높은 샘플 효율성을 유지하며 최첨단의 결과를 달성합니다.



### Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics (https://arxiv.org/abs/2510.06367)
Comments:
          Accepted for the NeurIPS 2025 Machine Learning and the Physical Sciences workshop. 6 pages, 3 figures

- **What's New**: 이 논문에서는 Neural ODEs(신경 미분 방정식)의 물리적 해를 평가하기 위해 Helmholtz metrics(헬름홀츠 메트릭스)를 제안합니다. 이 메트릭스는 주어진 ODE(상미분 방정식)가 Euler-Lagrange 방정식과 얼마나 유사한지를 정량화할 수 있습니다. 또한, Lagrangian neural ODE(라그랑주 신경 미분 방정식)을 도입하여 추가적인 추론 비용 없이 Euler-Lagrange 방정식을 직접 학습할 수 있는 방법을 소개합니다.

- **Technical Details**: 주요 접근 방식은 Helmholtz 조건을 기반으로 ODE를 Lagrangian과 연결 지을 수 있는지를 판단하는 것입니다. 저자들은 Neural Network(신경망) g_{\theta_2}를 훈련하여 Helmholtz 메트릭스를 통해 주어진 ODE의 만족도를 측정하며, 이를 통해 Neural ODE를 Lagrangian neural ODE로 결합합니다. 두 개의 손실 함수, 즉 일반적 손실 ℒ_R와 Helmholtz 손실 ℒ_H를 결합하여 공동 최적화를 수행합니다.

- **Performance Highlights**: 제안된 Helmholtz metrics와 Lagrangian neural ODE는 학습 Toy 시스템에 적용되어 효율성을 입증합니다. 네트워크는 다양한 구성을 가지고 훈련되며, Softplus 활성화 함수가 사용되어 매끄러운 도함수를 보장합니다. 본 논문에서 개발된 메트릭스와 네트워크는 이론적 기반을 제공하며, 물리적 시스템에서 Neural ODE의 성능 향상 가능성을 시사합니다.



### PIKAN: Physics-Inspired Kolmogorov-Arnold Networks for Explainable UAV Channel Modelling (https://arxiv.org/abs/2510.06355)
- **What's New**: 이번 연구에서는 물리적 원리를 포함하는 새로운 UAV 전송 채널 모델링 접근법인 물리 영감을 받은 Kolmogorov-Arnold Network(PIKAN)을 제시합니다. PIKAN은 고정된 이론적 제약 없이 현실 세계의 UAV 채널을 정확히 모델링할 수 있도록 유도 편향을 도입합니다. 이를 통해 기존의 deterministic 모델과 블랙박스 딥러닝 모델 간의 간극을 메울 수 있는 방법을 제공하게 됩니다.

- **Technical Details**: KAN은 Kolmogorov-Arnold 표현 정리에 영감을 받은 신경망 아키텍처로, 기존 다층 퍼셉트론(MLP)의 비선형 활성화 함수 적용 방식을 뒤집었습니다. PIKAN은 물리적 법칙을 모델에 통합하면서도 물리적 제약 조건을 강하게 설정하지 않기 때문에 유연한 학습을 가능하게 합니다. 이 구조를 통해, KAN은 높은 정확도와 더불어 해석 가능성을 유지할 수 있는 이점을 제공합니다.

- **Performance Highlights**: PIKAN은 UAV A2G 측정 데이터에서 블랙박스 DL 모델과 비슷한 정확도를 달성하면서도 채널 동작을 설명할 수 있는 상징적 표현을 제공합니다. 실험 결과, PIKAN은 단 232개의 파라미터로도 MLP 모델보다 최대 37배 가벼운 성능을 내며, 물리적 직관에 부합하는 해석 가능성을 유지합니다. 이러한 성과는 다음 세대 통신 네트워크에서 UAV 채널 모델링을 위한 효율적이고 해석 가능한 솔루션으로 PIKAN의 가능성을 부각시킵니다.



### Flexible Swarm Learning May Outpace Foundation Models in Essential Tasks (https://arxiv.org/abs/2510.06349)
- **What's New**: 이번 논문에서는 기초 모델(Fundation model)이 AI 발전에 미치는 영향을 다루며, AI가 인간의 결정 전략을 초월할 가능성에 대해 질문합니다. 특히, 의료와 같은 복잡한 시스템에서 고도 상호작용하는 기능들 간의 동적 장애물에 적응하는 데 있어 자율적이고 신뢰할 수 있는 AI 구조의 필요성을 강조합니다. 저자들은 모노리스(Monolithic) 기초 모델이 이러한 문제를 극복하는 데 한계를 가지며, 분산형 작은 에이전트 네트워크(SANs)를 제안합니다.

- **Technical Details**: 저자들은 상호작용하는 작은 에이전트 네트워크(SANs)가 복잡한 시스템에서 동적 환경에 적응하기 위한 새로운 접근 방식을 제공한다고 주장합니다. SAN은 각 에이전트가 시스템의 하위 구조를 나타내며, 이러한 에이전트가 특정 환경 변화에 대한 지속적인 업데이트를 통해 의사결정 과정을 최적화합니다. 특히, 저자는 그런 SAN의 군집 학습(Swarm learning) 접근이 효율적인 자기 적응(self-adaptation)을 가능하게 한다고 설명합니다.

- **Performance Highlights**: 저자들은 SANs 기반의 적응형 모델이 동적 환경 내에서 우수한 의사결정을 제공한다고 주장하며, 산소화(Oxygenation) 사례 연구를 통해 10분 내에 정확성을 회복하는 반면, 기초 모델은 약 200분이 걸린다고 밝혔습니다. 더 나아가, 그들은 SAN이 다양한 군집의 필요성을 강조하며, 이러한 다양성이 적응 과정의 탄력성을 증가시킨다고 주장합니다. 마지막으로, SAN 기반의 접근 방식이 전통적인 대형 모델보다 외부 스트레스 하에서 더 나은 최적화를 이루는 데 개념적으로 우수하다고 결론짓습니다.



### SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation (https://arxiv.org/abs/2510.06303)
Comments:
          Technical report. 39 pages, including 14 pages of appendix

- **What's New**: SDAR(Synergistic Diffusion-Autoregression)라는 새로운 패러다임이 제안되었습니다. 이 모델은 오토리그레시브(AR) 모델의 학습 효율성과 디퓨전(diffusion)의 병렬 추론(parallel inference) 능력을 통합합니다. 경량의 변환 단계를 통해 잘 훈련된 AR 모델을 블록형 디퓨전 모델로 변환함으로써 데이터 효율성을 극대화하고, 추론 시 모든 토큰을 병렬로 디코딩하는 방식으로 속도와 일관성을 확보합니다.

- **Technical Details**: SDAR는 경량의 적응 단계에서 AR 모델을 블록 단위의 디퓨전 모형으로 변환하는데, 이 구조는 디퓨전의 장점인 병렬 처리 가능성을 보장합니다. 디퓨전 모델은 블록 내에서 병렬로 생성을 처리하면서도 전역적으로 AR 프레임워크가 블록 간의 의존성을 모델링하여 효율성을 극대화합니다. 이 과정에서 KV 캐싱 및 가변 길이 생성과 같은 AR의 실용적인 이점을 유지하면서, 더 나은 모델링 정확성과 디코딩 속도를 제공합니다.

- **Performance Highlights**: SDAR는 수많은 실험을 통해 AR 모델이 마스크 디퓨전 모델보다 훨씬 더 높은 컴퓨팅 효율을 제공함을 보여주었습니다. 30B MoE 모델은 GPQA 및 ChemBench와 같은 과학적 추론 기준에서 AR 모델보다 우수한 성능을 발휘하며, 테스트 시 스케일링 방법에서도 개선된 결과를 보입니다. 이러한 결과는 SDAR가 오토리그레션과 디퓨전의 장점을 결합한 실제적인 패러다임을 확립했다는 것을 보여줍니다.



### BlockGPT: Spatio-Temporal Modelling of Rainfall via Frame-Level Autoregression (https://arxiv.org/abs/2510.06293)
- **What's New**: 이번 연구에서는 단기 강수 예측을 위한 새로운 모델 BlockGPT를 제안합니다. BlockGPT는 batched tokenization (Block) 방법을 사용하여 매 시간 단계에서 2차원 강수 맵을 예측합니다. 이 모델은 기존의 token-based 및 diffusion models의 한계를 극복하고, 실시간 데이터 예측을 위해 더 높은 성능과 효율성을 제공합니다.

- **Technical Details**: BlockGPT는 프레임 내의 self-attention과 프레임 간의 causal attention을 활용하여 시공간을 분해합니다. 이 모델은 강수 필드를 예측하는 두 단계로 구성되며, 첫 번째 단계에서 강수 필드를 latent token 공간으로 압축하고, 두 번째 단계에서 시간적 동역학을 autoregressively 모델링합니다. 예측된 강수 필드는 보다 일관된 예측을 제공하며, 학습 과정이 짧습니다.

- **Performance Highlights**: BlockGPT는 KNMI와 SEVIR 두 가지 데이터셋에서 평가되었으며, 기존 최첨단 모델에 비해 정확성 및 사건 로컬라이제이션에서 뛰어난 성능을 보입니다. 특히, BlockGPT는 병렬 처리 덕분에 기존 모델보다 최대 31배 더 빠른 추론 속도를 기록했습니다. 이 연구는 단기 강수 예측 분야에서 BlockGPT의 가능성을 보여줍니다.



### Traj-Transformer: Diffusion Models with Transformer for GPS Trajectory Generation (https://arxiv.org/abs/2510.06291)
- **What's New**: 이 논문에서는 혁신적인 Trajectory Transformer 모델을 제안합니다. 이 모델은 conditional information embedding과 noise prediction을 위해 transformer backbone을 활용합니다. 제안된 방법은 GPS 궤적 생성에서 품질을 크게 향상시키며, 기존 방법에서 발생했던 편차 문제를 효과적으로 완화합니다.

- **Technical Details**: GPS 궤적 생성 문제를 명확히 정의하고, GPS 포인트를 두 가지 방법으로 임베딩하는 전략을 탐구합니다. 일반적인 GAN 구조 대신, 변형된 transformer 구조를 사용하여 GPS 궤적을 생성합니다. 이로 인해, 모델은 더 적은 파라미터로도 생성 품질을 개선하고 세밀한 도로 수준의 정보를 보존합니다.

- **Performance Highlights**: 실제 세계 데이터셋을 이용한 실험 결과, Trajectory Transformer는 이전에 제안된 방법들보다 훨씬 더 뛰어난 생성 품질을 보여줍니다. 정량적 지표와 정성적 시각화 모두에서, 기존 기법들이 가진 문제들을 상당히 완화하였음을 확인했습니다. 이러한 결과는 모델의 유용성을 높이고, 도시 계획 및 교통 흐름 예측과 같은 다양한 응용 분야에 적용할 수 있습니다.



### On knot detection via picture recognition (https://arxiv.org/abs/2510.06284)
Comments:
          21 pages, many figures, comments welcome

- **What's New**: 이번 연구의 목표는 사진으로 촬영한 매듭(knot)을 자동으로 인식하는 것입니다. 현대 머신러닝 기법인 컨볼루션 신경망(convolutional neural networks)과 트랜스포머(transformers), 그리고 전통적인 알고리즘을 사용하여 이를 근사하는 전략을 설명합니다. 특히, 이미지를 기반으로 교차 수(crossing number)를 예측하는 간단한 기준선을 제시하여 경량화된 CNN과 트랜스포머 아키텍처가 구조적 정보를 회복할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 매듭의 양자 불변량(quantum invariants)인 존스 다항식(Jones polynomial)을 계산하기 위해 전통적인 알고리즘과 현대 머신러닝 기법의 혼합을 사용합니다. 관찰 모듈(perception modules)을 사용해 직접 이미지를 분석하여 단계를 나누어 평면 다이어그램(Planar Diagram, PD) 코드로 기호화된 재구성을 결합할 계획입니다. 이 두 단계의 접근 방식은 노이즈가 있는 시각적 데이터를 처리하는 머신러닝과 엄격한 위상 구분을 적용하는 불변량 간의 상호 보완성을 강조합니다.

- **Performance Highlights**: 이 연구는 매듭 분류를 위한 강력한 도구를 개발하는 장기 목표를 가지고 있습니다. 초기 실험 결과, 간단한 CNN 및 트랜스포머 아키텍처가 유의미한 결과를 도출할 수 있음을 보였으며, 이는 향후 매듭의 정확한 분류를 위한 기초를 제공합니다. 또한, 이러한 접근 방식은 머신러닝과 전통적인 방법론 간의 시너지를 통해 보다 강력하고 신뢰할 수 있는 평가 시스템을 구축할 가능성을 제시합니다.



### RVFL-X: A Novel Randomized Network Based on Complex Transformed Real-Valued Tabular Datasets (https://arxiv.org/abs/2510.06278)
- **What's New**: 이 논문은 랜덤화된 신경망(Randomized Neural Networks, RNNs)에서 실수 값 데이터셋을 복소수 값 표현으로 변환하는 효과적인 방법이 부족함을 해결하기 위해 두 가지 방법, 즉 자연적인 변환과 오토인코더 기반 방법을 제안합니다. 이를 통해 RVFL-X라는 복소수 값 확장을 제안하며, 원래의 RVFL 아키텍처의 단순성과 효율성을 유지하면서도 복소수 변환을 통합합니다. RVFL-X는 입력, 가중치, 활성화 함수와 같은 복소수 구성 요소를 사용하여 복소수 표현을 처리하고 실수 값을 출력합니다.

- **Technical Details**: RVFL-X는 랜덤 벡터 기능 연결(Random Vector Functional Link, RVFL) 네트워크의 복소수 확장으로 설계되었으며, 모델의 복소수 매개변수를 도입하여 계산적 안정성 및 표현력을 높입니다. 기존의 RVFL 아키텍처와의 호환성을 유지하면서도 실수 데이터를 복소수 데이터로 전환할 수 있는 새로운 메커니즘을 제공합니다. 또한, 복소수 가중치와 활성화 함수의 통합을 통해 모델의 성능을 극대화하고, 이는 다양한 머신러닝 애플리케이션에서 유용하게 사용될 수 있습니다.

- **Performance Highlights**: 80개의 실측 UCI 데이터셋을 통해 RVFL-X는 원래 RVFL 및 최신 상태(State-Of-The-Art) RNN 변형들보다 일관되게 우수한 성능을 보였으며, 이는 다양한 응용 분야에 걸쳐 강력한 효과를 입증합니다. RVFL-X의 도입된 복소수 표현력이 반복적인 학습이 아닌 최적의 성능을 달성할 수 있도록 돕습니다. 결론적으로, 이 논문은 RVFL 모델의 새로운 가능성을 탐색하는 중요한 기초를 제공하며, 머신러닝의 다양한 도전과제를 해결하는 데 기여할 것으로 기대됩니다.



### MCCE: A Framework for Multi-LLM Collaborative Co-Evolution (https://arxiv.org/abs/2510.06270)
- **What's New**: 본 연구는 폐쇄형 소스(Closed-source) LLM과 경량화된 훈련 가능 모델을 연결한 Multi-LLM Collaborative Co-evolution (MCCE) 프레임워크를 제시합니다. 이러한 구조는 지역 최적해(optimizing local optima)에 갇히지 않고, 경험 기반 학습을 통해 지속적인 진화(evolution)와 협력을 가능하게 합니다. MCCE는 효율적인 다목적 최적화 작업을 가능하게 하며, 실질적인 응용 가능성을 가지고 있습니다.

- **Technical Details**: MCCE는 폐쇄형 LLM이 전 세계 탐색(global exploration)을 수행하는 동안, 경량화된 모델이 축적된 경험을 통해 보다 목표 지향적인 검색(targeted search)을 수행하게 합니다. 이러한 상호 보완적 관계는 각 모델의 강점을 강화하며, 지식을 지속적으로 내재화(internalize)하는 피드백 루프(feedback loop)를 통해 이루어집니다. 이 프레임워크는 기존의 정적 프롬프트 사용이나 RAG(Revelation-Augmented Generation)와는 다르게 경험을 깊이 있게 축적할 수 있는 모델 파라미터 업데이트(parameter updates)를 통해 효과적으로 문제 해결을 가능하게 합니다.

- **Performance Highlights**: MCCE는 다목적 약물 설계(multi-objective drug design) 벤치마크에서 최신 기술의 Pareto front 품질을 달성했으며, 기존의 기초 모델(base models)을 지속적으로 초월하는 성과를 기록했습니다. 이러한 성과는 LLM 시스템에서 경험 기반 학습과 지식 기반 탐색을 결합한 새로운 패러다임을 제시합니다. MCCE는 다양한 과학 및 공학 분야에 걸쳐 구조적 최적화가 필요한 작업에서도 확장 가능성을 지니고 있습니다.



### RareGraph-Synth: Knowledge-Guided Diffusion Models for Generating Privacy-Preserving Synthetic Patient Trajectories in Ultra-Rare Diseases (https://arxiv.org/abs/2510.06267)
Comments:
          6 pages, 2 figures, 2 tables. Submitted to IEEE International Conference on Data Science and Advanced Analytics (DSAA)

- **What's New**: RareGraph-Synth는 초희귀 질병을 위한 현실적이면서도 개인 정보 보호를 고려한 전자 건강 기록(EHR) 경로를 생성하는 지식 기반의 연속 시간 확산(framework) 프레임워크로 개발되었습니다. 이 시스템은 Orphanet/Orphadata, Human Phenotype Ontology (HPO), GARD rare-disease KG, PrimeKG, 및 FDA Adverse Event Reporting System (FAERS)와 같은 다섯 개의 공개 리소스를 통합하여 약 800만 개의 유형화된 엣지를 포함하는 이질적 지식 그래프(knowledge graph)를 형성합니다.

- **Technical Details**: RareGraph-Synth는 800만 개 엣지에서 추출된 메타 경로 점수를 사용하여 순방향(stochastic) 확산 미분 방정식에서 token noise를 조절합니다. 이를 통해 병리학적으로 그럴듯한 실험실-약물-부작용 동시 발생을 이끌어내면서도 안정적인 확산 모델(score-based diffusion model)의 특성을 유지합니다. 그 후 역 노이즈 감소기(reverse denoiser)는 개인 건강 정보를 포함하지 않는 실험실 코드, 약물 코드, 부작용 플래그의 세타임스탬프(timestamps) 시퀀스를 생성합니다.

- **Performance Highlights**: 시뮬레이션된 초희귀 질병 집단에서는 RareGraph-Synth가 비유도(diffusion baseline) 모델에 비해 범주별 최대 평균 불일치(Maximum Mean Discrepancy)를 40% 줄였으며, GAN 모델에 비해 60% 이상 감소시켰습니다. DOMIAS 공격자를 이용한 블랙박스 멤버십 추론 평가에서는 약 0.53의 AUROC를 기록하여 안전한 릴리즈(threshold)의 기준인 0.55 아래에 있으며, 비-KG 기준에 비해 상당히 개선된 결과를 보여주었습니다. 이러한 결과는 생물의학 지식 그래프를 확산 노이즈 일정을 통합하는 것이 데이터 공유의 안전성을 높이면서 신뢰성을 동시에 개선할 수 있음을 나타냅니다.



### Artificial Hippocampus Networks for Efficient Long-Context Modeling (https://arxiv.org/abs/2510.07318)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 인공 신경망의 메모리 프레임워크를 제안하고, Multi-Store Model에서 영감을 받아 Artificial Hippocampus Network (AHN)를 도입하였다. AHN은 Transformer의 KV 캐시를 손실 없는 단기 메모리로 유지하며, 슬라이딩 윈도우 밖의 정보를 압축하여 고정 크기의 장기 메모리로 변환한다. 이 방법은 최신 RNN 유사 구조를 이용하여 AHNs를 구현하고, 오랜 컨텍스트 벤치마크에서 개선된 성과를 보여주는 실험 결과를 제시한다.

- **Technical Details**: AHN의 구조는 Mamba2, DeltaNet, Gated DeltaNet와 같은 RNN 유사 아키텍처로 인스턴스화되며, 이들 모델은 손실 없는 단기 메모리를 슬라이딩 윈도우로 유지한다. 정보가 윈도우를 넘어갈 경우, AHN 모듈이 이를 고정 크기로 압축하는 방식으로 작동한다. 이로 인해 AHN을 적용한 모델이 슬라이딩 윈도우 및 전체 주의(attention) 모델들을 능가하고, 메모리 및 계산 비용을 현저히 줄인다.

- **Performance Highlights**: 실험에서는 AHN을 적용한 Qwen2.5-3B-Instruct 모델이 40.5%의 플롭(FLOPs) 감소와 74.0%의 메모리 캐시 감소를 달성했으며, 평균 점수가 4.41에서 5.88로 향상되었다. 이러한 결과는 AHN을 통한 메모리 효율성을 극대화하고, 긴 시퀀스 처리에서 경쟁력 있는 성능을 발휘함을 보여준다. 논문은 AHN의 변형 모델을 개발하기 위한 코드와 모델을 배포할 예정이다.



### Vibe Checker: Aligning Code Evaluation with Human Preferenc (https://arxiv.org/abs/2510.07315)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)을 활용하여 코드 생성을 위해 자연어 상호작용을 사용하는 'vibe coding'을 소개합니다. Vibe check은 코드의 기능성뿐만 아니라, 솔루션이 어떤 느낌을 주어야 하고, 읽기 쉽고, 사용자의 의도를 유지해야 함을 강조합니다. 이 연구는 기능적 정확성을 넘어서는 지침 준수(instruction following)가 vibe check의 핵심 요소임을 가정합니다.

- **Technical Details**: 우리는 VeriCode라는 30개의 검증 가능한 코드 지침의 분류법을 제시하며, 각 지침에 해당하는 결정론적 검증기를 제공합니다. 이 분류법은 기존의 평가 도구를 보완하여, 코드의 지침 준수와 기능적 정확성을 동시에 평가하는 Vibe Checker라는 테스트베드를 구성합니다. 이를 통해 31개의 주요 LLM을 평가하며, 강력한 모델조차도 여러 지침 준수에서 어려움을 겪는다는 것을 확인했습니다.

- **Performance Highlights**: 기능적 정확성과 지침 준수의 복합 점수가 인간의 선호와 가장 잘 연관됨을 보여주며, 실제 프로그래밍 작업에서 지침 준수가 주요 구분 요소로 부각됩니다. 이 연구는 코드 작업에서 사용자의 선호도에 더 잘 맞춰진 모델을 개발하기 위한 기준을 제시합니다.



### Cocoon: A System Architecture for Differentially Private Training with Correlated Noises (https://arxiv.org/abs/2510.07304)
- **What's New**: 이 논문에서는 기계 학습 (ML) 모델이 훈련 데이터의 메모리와 누출로 인해 심각한 개인정보 보호 문제를 야기한다는 점을 강조합니다. 최근에 제안된 차별적 프라이버시 (DP) 기반 훈련 알고리즘, 특히 DP-SGD가 그러한 문제를 해결하기 위한 접근 방식으로 주목받고 있습니다. 그러나 DP-SGD는 각 훈련 단계에서 노이즈를 추가함에 따라 모델의 정확도 감소가 발생하는 단점이 있습니다. 이를 해결하기 위해, 논문에서는 새로운 방식인 Cocoon을 제안하고, 이 시스템이 훈련 효율성을 어떻게 개선하는지를 탐구합니다.

- **Technical Details**: Cocoon 프레임워크는 관련된 노이즈를 사용하여 효율적인 훈련을 지원합니다. 이 프레임워크는 훈련 전 모든 관련 노이즈를 미리 계산하고 저장함으로써 대규모 임베딩 테이블을 효율적으로 처리합니다. Cocoon은 또한 사용자 정의 근처 메모리 처리 (NMP) 하드웨어를 사용하여 과거의 노이즈를 보관하고 처리할 수 있도록 하여 데이터 전송을 최소화합니다. 이를 통해 대규모 신경망 훈련에서 메모리 및 계산 오버헤드를 줄이는 방향으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, Cocoon-Emb는 기존 방법에 비해 2.33배에서 10.82배의 성능 향상을 보였고, Cocoon-NMP는 1.55배에서 3.06배의 개선을 달성했습니다. 이러한 성과는 큰 임베딩 테이블을 활용하는 모델 및 파라미터가 많은 대규모 모델에서 더욱 두드러집니다. 이 연구는 차별적 프라이버시를 위한 훈련 방법에 대한 시스템 차별화를 제공하며, Pytorch 기반의 높은 최적화된 라이브러리를 통해 구현되었습니다. Cocoon 프레임워크는 실제 시스템에서 높은 성능을 보여주어, 학계 및 산업계에서도 큰 관심을 받고 있습니다.



### On the Convergence of Moral Self-Correction in Large Language Models (https://arxiv.org/abs/2510.07290)
Comments:
          19pages, 7 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 '도덕적 자기 수정(moral self-correction)'이 성과 수렴(convergence)을 통해 개선될 수 있다는 것을 보여줍니다. 자기 수정(process of self-correction) 기능을 활용한 접근 방식은 복잡한 인간 피드백 없이 모델의 성능을 향상시킬 수 있는 잠재력을 지니고 있습니다. 실험 결과, 도덕적 자기 수정이 모델의 불확실성(model uncertainty)을 줄여 성과의 수렴을 이끌어내는 방식을 규명하였습니다.

- **Technical Details**: 연구는 LLMs가 자기 수정 지침을 반복적으로 적용해 나가면서 성과 수렴이 이루어진다는 관찰을 통해 두 가지 주요 연구 질문을 다룹니다. 첫째, 도덕적 자기 수정이 성과를 수렴하는가? 둘째, 이러한 수렴이 이루어지는 기본 메커니즘은 무엇인가? 연구진은 비정형적인 지침을 통해 활성화된 도덕 개념(latent concept)이 모델의 불확실성을 감소시킨다는 것을 밝혀냈습니다.

- **Performance Highlights**: 이 논문은 다양한 작업과 모델에서 도덕적 자기 수정이 수렴된 성과를 보여주는 실증적 증거를 제시합니다. 이를 통해 도덕적 자기 수정이텍스트 정제(text detoxification) 성과를 강화하는 데 중요한 역할을 한다는 점을 확인했습니다. 연구의 결과는 LLMs의 잠재력을 극대화하는 데 도움을 줄 수 있는 방향을 제시합니다.



### Online Rubrics Elicitation from Pairwise Comparisons (https://arxiv.org/abs/2510.07284)
- **What's New**: 이번 논문에서는 LLM을 위한 평가 기준을 온라인으로 동적으로 개발하는 새로운 방법인 Online Rubrics Elicitation(OnlineRubrics)을 소개합니다. 기존의 정적인 rubrics는 훈련 과정에서 발생하는 새로운 요구 사항을 제대로 반영하지 못했으나, OnlineRubrics는 응답을 기반으로 쌍평가(pairwise comparisons)를 통해 지속적으로 기준을 개선하는 방식입니다. 이 방법은 AlpacaEval, GPQA 등 여러 벤치마크에서 최대 8%의 성능 향상을 보였습니다.

- **Technical Details**: OnlineRubrics는 현재 모델과 참조 모델의 응답을 쌍으로 비교하여 새로운 평가 기준을 생성합니다. 이를 통해 응답의 오류를 지속적으로 식별하고 개선할 수 있으며, 기존의 rubrics를 유연하게 확장합니다. 평가 프레임워크는 reinforcement learning에 대한 다양한 응답 모델링을 가능하게 하여, verifiable 및 non-verifiable 특성을 모두 포괄합니다.

- **Performance Highlights**: 이 연구는 Expert 및 Generalist 도메인에 대한 두 개의 데이터셋을 활용하여 OnlineRubrics의 성능을 평가하였습니다. 이 방식은 GPQA-Diamond, GSM8K, AlpacaEval, Arena-Hard를 포함한 여러 벤치마크에서 기반 모델 대비 최고 25%의 성능 향상을 기록하며, 기존 정적 rubrics와 비교하여 우수한 결과를 입증했습니다.



### Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dens (https://arxiv.org/abs/2510.07242)
Comments:
          20 pages

- **What's New**: HERO(Hybrid Ensemble Reward Optimization)는 검증 가능한 보상과 보상 모델 점수를 통합하는 혁신적인 강화 학습 프레임워크입니다. 이 시스템은 정밀한 신호를 제공하는 검증 기구의 안정성과, 보다 풍부한 지속적 보상을 제공하는 보상 모델의 장점을 결합하기 위해 설계되었습니다. HERO는 기존의 이진 N 또는 1 보상 시스템에서 발생하는 한계를 극복하기 위한 두 가지 핵심 혁신 기능을 도입합니다.

- **Technical Details**: HERO는 두 가지 주요 기법을 통해 보상 모델 점수를 수정하여 최적화의 신뢰성을 유지합니다. 첫째, 계층화된 정규화(straction normalization) 기법을 통해 검증자가 정의한 정확성 그룹 내에서 보상 모델 점수를 제한합니다. 둘째, 분산 인식 가중치 기법을 사용하여 훈련 중 다양한 프롬프트의 기여도를 조정함으로써 어려운 질문에 대한 비율을 강조합니다.

- **Performance Highlights**: HERO는 다양한 수학적 추론 벤치마크에서 RM(Reward Model)-전용 및 검증자 전용 기준 모델들보다 일관되게 더 나은 성능을 보여주었습니다. 특히, 검증하기 어려운 작업의 경우, HERO는 66.3 점을 기록하여 RM-전용 시스템(54.6)보다 +11.7 점, 검증자 전용 시스템(57.1)보다 +9.2 점 높은 성과를 달성했습니다.



### Accelerating Inference for Multilayer Neural Networks with Quantum Computers (https://arxiv.org/abs/2510.07195)
- **What's New**: 이 논문에서는 진정한 고전적인 양자 처리 장치(QPU)가 통합된 다층 신경망(multilayer neural network)을 처음으로 구현한 연구 내용을 소개합니다. 이 구현은 ResNet(Residual Network) 아키텍처를 기반으로 하며, 다양한 활성화 함수(non-linear activation functions)를 적용합니다. 이 새로운 접근 방식은 양자 컴퓨터에서 신경망의 추론(inference) 과정을 가속화하여 양자 머신 러닝의 가능성을 제시합니다.

- **Technical Details**: 제안된 방법론은 잔여 블록(residual blocks), 다중 필터 2D 컨볼루션(multi-filter 2D convolutions), 시그모이드 활성화(sigmoid activations), 스킵 연결(skip-connections), 레이어 정규화(layer normalizations)를 포함합니다. 논문은 세 가지 양자 데이터 접근 제도를 통해 네트워크의 복잡성을 분석하며, 기본적인 추론 비용을 다루고 있습니다. 특히, 고전적인 방법에 비해 최대 4제곱 비율의 속도 향상을 입증하였습니다.

- **Performance Highlights**: 제안된 접근법은 입력과 가중치에 대한 양자 접근이 이루어질 경우 O(polylog(N/ε)^k)로 추론 비용을 줄일 수 있음을 보여주었습니다. 이 결과는 기존 기술로는 실현할 수 없는 큰 속도 향상을 가능하게 합니다. 또한, 세 가지 다른 깃대(Regime)에서의 성능 개선도 논의되며, 양자 컴퓨터에서의 다층 신경망의 실용 가능성을 보여줍니다.



### Covert Quantum Learning: Privately and Verifiably Learning from Quantum Data (https://arxiv.org/abs/2510.07193)
Comments:
          16 + 54 pages

- **What's New**: 본 논문은 원거리에서 접근 가능한 양자 컴퓨팅 및 데이터를 통한 양자 학습의 새로운 모델을 제안합니다. 특히 전문가들이 접근할 수 없는, 원거리의 양자 데이터에 대해 비밀 보호 및 검증 가능성을 동시에 성취할 수 있는 방법을 소개합니다. 이 연구는 양자 데이터의 이점을 활용한 현실적인 접근 방식으로, 이러한 접근법이 안전하고 검증 가능한 양자 학습을 가능하게 한다고 주장합니다.

- **Technical Details**: 저자들은 covert (검증 가능한) 학습 모델을 양자 학습 이론에 적용하였으며, 이를 위해 두 가지 privacy 개념을 다룹니다: 전략 은닉 (strategy-covertness) 및 목표 은닉 (target-covertness). 이러한 모델은 비계산적 하드니스 가정 없이 실현 가능하며, 원격 데이터 접근 시나리오에 적합한 여러 양자 데이터 오라클에 대해 연구됩니다.

- **Performance Highlights**: 연구 결과는 전략-은닉 알고리즘과 목표-은닉 알고리즘을 통해 양자 통계 쿼리를 구현할 수 있음을 보여줍니다. 특히, Forrelation 및 Simon's 문제와 같은 일반적인 양자 문제를 해결하는 데 있어 고립된 적대자에 대한 접근방식을 검증하며, 이러한 접근이 양자 데이터의 안전하고 비밀스러운 활용을 가능하게 함을 입증합니다.



### Resolution scaling governs DINOv3 transfer performance in chest radiograph classification (https://arxiv.org/abs/2510.07191)
- **What's New**: 이 논문은 메타의 DINOv3 모델을 소개하며, 기존의 Self-supervised learning (SSL) 모델을 Gram-anchored self-distillation을 통해 확장했습니다. 이는 흉부 방사선 이미지에서의 변별력 있는 findings(발견)에 대해 SSL의 효과를 평가한 연구로, DINOv3가 DINOv2 및 ImageNet 초기화 모델과 비교한 체계적인 성능 검정을 수행했습니다.

- **Technical Details**: 논문에서는 두 가지 대표적인 backbone(백본) 모델인 ViT-B/16 및 ConvNeXt-B를 사용하여 814,000개 이상의 샘플이 포함된 7개의 데이터셋을 benchmark(벤치마크)했습니다. 이미지 해상도는 224x224, 512x512 및 1024x1024 픽셀로 분석되었으며, 주된 성과 지표는 mean AUROC(평균 면적 아래 곡선)으로 설정되었습니다.

- **Performance Highlights**: DINOv3는 224x224 해상도에서 성인 데이터셋에서 DINOv2와 비슷한 성과를 보였지만, 512x512 해상도에서는 DINOv3가 DINOv2 및 ImageNet보다 일관된 성능 향상을 보였습니다. 이러한 결과는 흉부 방사선 진단에서 높은 입력 해상도가 현대적인 SSL 모델의 이점을 활용하는 데 중요하다는 점을 강조했습니다.



### Split Conformal Classification with Unsupervised Calibration (https://arxiv.org/abs/2510.07185)
- **What's New**: 본 논문에서는 레이블이 없는 샘플을 사용하여 분할 적합 예측(split conformal prediction) 방법을 개선하는 효율적인 방법론을 제시합니다. 기존의 방법들은 레이블이 있는 샘플을 필요로 하며, 이는 새로운 레이블을 추가로 획득해야 하는 불편함을 동반합니다. 제안된 방법은 레이블이 없는 샘플과 기존에 학습에 사용된 레이블이 있는 샘플을 활용하여 분할 적합 예측 규칙을 생성합니다.

- **Technical Details**: 제안된 방법은 레이블이 없는 샘플에 레이블 가중치를 할당하여 통계적으로 훈련 데이터와 구분되지 않도록 합니다. 이 두 샘플의 비모수적 테스트 통계치를 최소화하는 접근법을 사용하며, 재생 커널 힐베르트 공간의 함수를 이용하여 구현 세부사항을 제공합니다. 이 과정에서 성능 보장을 위한 높은 신뢰도 경계를 제시하며, 커버리지 갭의 특성을 편향-분산 트레이드오프(bias-variance trade-off) 관점에서 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 감독된 적합(calibration) 방법과 유사한 성능을 달성할 수 있음을 보여줍니다. 그러나 약간의 성능 보장 약화와 계산 복잡도의 적당한 증가가 필요합니다. 이 방법은 특히 라벨이 부정확한 경우에도 높은 성능을 유지할 수 있도록 설계되었습니다.



### Bayesian Portfolio Optimization by Predictive Synthesis (https://arxiv.org/abs/2510.07180)
- **What's New**: 이번 연구는 Bayesian Predictive Synthesis (BPS)를 기반으로 한 포트폴리오 최적화 방법을 제안합니다. 기존의 포트폴리오 최적화 방식들은 자산 수익률의 분포 정보를 필요로 하지만, 투자자들은 이러한 정보를 알지 못하는 경우가 많습니다. BPS는 여러 자산 수익 안전 모델의 예측을 통합함으로써 금융 시장의 불확실성을 반영한 베이지안 예측 사후 분포를 얻습니다.

- **Technical Details**: 포트폴리오 최적화 문제는 K종의 금융 자산을 T기간에 걸쳐 최적화하는 것으로 정의됩니다. 각 자산 수익률 벡터는 다변량 정규 분포를 따르며, BPS 방법을 통해 예측 분포를 통합하여 새로운 예측 분포를 구축합니다. 이 연구에서 다룬 세 가지 접근법은 평균-분산 포트폴리오, 분위수 기반 포트폴리오, 리스크 패리티 포트폴리오입니다.

- **Performance Highlights**: 본 연구는 BPS를 통해 얻어진 예측 분포를 기반으로 포트폴리오 선택 기준의 효과를 실증적으로 분석합니다. 평균-분산 접근법의 경우, 제약 최적화에 대한 성과를 고려하였으며, 분위수 기반 포트폴리오와 리스크 패리티 접근법에서의 BPS의 적용 가능성 또한 탐구하였습니다. 이 연구는 투자자에게 더 나은 포트폴리오 구성을 위한 기반을 마련합니다.



### Quantifying Data Contamination in Psychometric Evaluations of LLMs (https://arxiv.org/abs/2510.07175)
Comments:
          12 pages, 1 figure

- **What's New**: 최근 연구에서는 Large Language Model(LLM)에 심리측정 설문지를 적용하여 가치관, 성격, 도덕적 기반 및 어두운 특성과 같은 고차원 심리적 구성 요소를 평가하고 있습니다. 이전 연구에서 심리측정 inventory로 인한 데이터 오염 가능성이 제기되었음에도 불구하고, 이러한 오염의 정도를 체계적으로 정량화한 연구는 없었습니다. 이에 대한 해결책으로, 본 연구는 LLM의 심리측정 평가에서 데이터 오염 정도를 체계적으로 측정하는 프레임워크를 제안하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 측면을 기준으로 데이터 오염을 평가합니다: (1) 항목 암기(item memorization), (2) 평가 암기(evaluation memorization), (3) 목표 점수 일치(target score matching)입니다. 연구진은 21개의 모델과 널리 사용되는 심리측정 설문인 Big Five Inventory(BFI-44) 및 Portrait Values Questionnaire(PVQ) 등을 포함한 여러 재고 목록에 프레임워크를 적용하였습니다. 결과적으로, 대부분의 LLM이 이러한 inventory에 오염되어 있다는 강력한 증거가 확보되었습니다.

- **Performance Highlights**: 연구 결과, LLM들은 단순히 아이템을 암기하는 것을 넘어, 특정 목표 점수를 달성하기 위해 응답을 조정할 수 있는 능력을 갖추고 있음을 보였습니다. 이는 LLM이 심리측정 테스트에서 과거의 훈련 데이터에 노출된 내용을 바탕으로 응답할 가능성이 있음을 시사합니다. 이러한 결과는 LLM의 심리적 특성 평가에 대한 더 체계적인 조사를 필요로 하고 있으며, 향후 연구에서 모델이 심리측정 테스트의 내용을 조작하는지 여부를 밝혀내는 기초가 될 것입니다.



### NurseLLM: The First Specialized Language Model for Nursing (https://arxiv.org/abs/2510.07173)
Comments:
          EMNLP 2025 Industry Track

- **What's New**: 이번 논문은 간호 분야에 맞춤화된 첫 번째 대규모 언어 모델인 NurseLLM을 소개합니다. NurseLLM은 여러 선택 질문-응답(MCQ) 작업을 위해 설계되어 있으며, 이를 위해 대규모 간호 MCQ 데이터 세트를 구축하였습니다. 기존 모델들과 비교할 때, NurseLLM은 간호 관련 작업에서 더욱 뛰어난 성능을 보이며, 전문화된 LLM의 필요성을 강조합니다. 또한, 간호 분야에서의 추론 및 다중 에이전트 협업 시스템의 가능성을 탐구하고 있습니다.

- **Technical Details**: NurseLLM은 복잡한 NCLEX와 같은 간호 전문 질문-응답을 다루기 위해 다양한 주제의 NCLEX 질문-답변 조합을 생성하기 위해 다단계 데이터 생성 파이프라인을 활용하였습니다. 이 과정에서 1,251,125개의 샘플을 포함한 대규모 데이터 세트를 구축하였으며, 세 가지 간호 MCQ 벤치마크를 개발하여 LLM의 체계적인 평가를 가능하게 하였습니다. 더불어, 모델의 투명성과 신뢰성을 확보하기 위해 정답과 함께 합리적인 이유를 제공하도록 설계되었습니다.

- **Performance Highlights**: NurseLLM은 유사한 크기의 범용 및 의료 전문 LLM들을 상대로 다양한 벤치마크에서 우수한 성능을 발휘하였습니다. 이는 간호 분야에서 전문화된 LLM의 중요성을 뒷받침합니다. 실험 결과, NurseLLM의 성능은 기존의 일반적 모델들보다 더 높은 정확성을 보여주었으며, 이는 간호에 특화된 AI 도구의 필요성을 강조하게 됩니다. 추론 능력과 협업 시스템의 통합을 통해 향후 간호 분야의 발전 가능성에 대한 기대감을 높이고 있습니다.



### Spectral Graph Clustering under Differential Privacy: Balancing Privacy, Accuracy, and Efficiency (https://arxiv.org/abs/2510.07136)
- **What's New**: 이 논문에서는 edge differential privacy(DP)에 따른 스펙트럼 그래프 클러스터링 문제를 연구합니다. 구체적으로, 우리는 임의의 edge flipping과 인접 행렬 셔플링을 결합한 그래프 교란, 사전 투영 기법, 노이즈가 추가된 파워 반복 방법을 포함하는 세 가지 메커니즘을 개발했습니다. 새로운 메커니즘은 그래프의 주요 스펙트럼 속성을 유지하면서 비공식적인 경계들을 제공하는 동시에, 개인의 관계를 보호하는 한편 통계적 쿼리 결과에 미치는 영향을 최소화합니다.

- **Technical Details**: 우리는 임의의 응답을 사용하여 인접 행렬을 교란하고, 스펙트럼 속성을 보존하는 행렬 셔플링을 도입합니다. 시뮬레이션을 통해 제안된 메커니즘은 edge DP를 만족하며, 노이즈 추가와 클러스터링 정확도 간의 거래를 분석합니다. 각 메커니즘의 구체적 결과와 더불어, 클러스터링 오류율, 계산비용, 공간 복잡도의 기본적인 거래를 요약하였습니다.

- **Performance Highlights**: 본 연구의 실험은 이론적 분석을 검증하며, 개인 정보 보호와 유용성의 거래를 보여줍니다. 행렬 셔플링 메커니즘은 낮은 오류율을 제공하지만, 랜덤 응답 기반의 교란 때문의 높은 계산 복잡도가 동반됩니다. 이 외에도 노이즈가 있는 파워 방법은 고밀도 그래프 상황에서 효율성과 정확성 사이의 최적 균형을 제공합니다.



### TRIM: Token-wise Attention-Derived Saliency for Data-Efficient Instruction Tuning (https://arxiv.org/abs/2510.07118)
- **What's New**: 이 논문은 TRIM (Token Relevance via Interpretable Multi-layer Attention)이라는 새로운 방법을 소개합니다. TRIM은 작은 고품질 샘플 집합을 사용하여 큰 언어 모델(LLM)의 지침 조정을 위한 데이터 선택을 효율적으로 수행하는 방법론입니다. 특히, 기존의 경량화 방법들과 달리 Gradient 사용을 피하고 오직 Forward 패스를 통해 성능을 극대화하는 것이 특징입니다.

- **Technical Details**: TRIM은 토큰 중심의 프레임워크로, 주어진 몇 개의 샘플을 통해 이들 샘플의 토큰 표현을 활용하여 타겟 태스크에 최적화된 데이터를 선택합니다. 이 방법은 각 토큰의 attention 기반의 'fingerprint'를 통해 세부적인 특성을 포착하려고 합니다. Two-stage 구조를 적용하여 첫 번째 단계에서는 토큰의 saliency 점수를 계산하고, 두 번째 단계에서는 이 점수를 토대로 후보 샘플의 relevancy 점수를 계산하여 최종 코어셋을 구축합니다.

- **Performance Highlights**: TRIM은 단 5-10개의 샘플에서 최대 9% 이상의 성과 향상을 보여주며, 전체 데이터에 대한 Fine-tuning을 초월한 성능을 기록합니다. 이 기술은 더 높은 구조적 충실도를 제공하며, 샘플 수준의 방법에서 발생하는 길이 편향을 완화하는데 도움을 줍니다. 결과적으로 TRIM은 기존 방법들보다 빠르고 효율적으로 고품질의 지침 조정 데이터 세트를 구축할 수 있는 가능성을 보여줍니다.



### The Contingencies of Physical Embodiment Allow for Open-Endedness and Car (https://arxiv.org/abs/2510.07117)
Comments:
          15 pages, 1 figure

- **What's New**: 본 논문은 인공지능 에이전트의 신체적 구현에서의 취약성과 사망률을 탐구합니다. 이를 위해, Martin Heidegger의 존재론적 개념인 ‘이 세계에 존재하기 (being-in-the-world)’와 ‘죽음을 향한 존재하기 (being-towards-death)’를 기반으로 두 가지 최소 조건을 정의합니다. 이러한 조건을 통해 에이전트가 생리적 안정을 유지하고 죽음을 피하는 내재적 동기를 발전시킬 수 있음을 제시합니다.

- **Technical Details**: 논문은 에이전트를 환경의 상태를 감지하고 행동 정책에 따라 행동을 실현하는 주체로 정의합니다. 이들은 부분적으로 관찰 가능한 Markov 결정 과정 (POMDP) 기반에서 환경과 상호작용합니다. 에이전트의 센서, 정책, 액추에이터가 환경의 상태와 내재적으로 연결되며, 이를 통해 신체적 구현의 복잡성을 보다 풍부하게 탐구합니다.

- **Performance Highlights**: 저자들은 생리적 요구를 충족하기 위한 내재적 동기와 자율성을 극대화하는 방법을 강화 학습 프레임워크 내에서 분석합니다. 이 방법은 에이전트가 복잡한 다중 에이전트 환경에서 스스로를 잘 유지하고 돌보는 능력을 배양할 수 있도록 합니다. 향후 연구는 이러한 에이전트들이 상대방을 이해하고 돌보는 데 기여할 수 있는 가능성에 대해 살펴볼 예정입니다.



### GNN-enhanced Traffic Anomaly Detection for Next-Generation SDN-Enabled Consumer Electronics (https://arxiv.org/abs/2510.07109)
Comments:
          This paper has been accepted for publication in IEEE Transactions on Consumer Electronics. 10 pages, 6 figures

- **What's New**: 본 논문은 소비자 전자 제품(CE) 네트워크의 보안 강화를 위해 그래프 신경망 기반 네트워크 이상 탐지 프레임워크(GNN-NAD)를 제안합니다. 기존의 방법들은 주로 통계적인 트래픽 분석에 의존하지만, 본 연구에서는 정적 공격 그래프와 동적 트래픽 데이터를 결합하여 보다 포괄적인 보안 평가를 가능하게 합니다. 이렇게 정적 정보와 동적 행동을 통합함으로써, 더 높은 정확도와 효율성을 달성할 수 있습니다.

- **Technical Details**: GNN-NAD 프레임워크는 소프트웨어 정의 네트워킹(SDN)과 컴퓨트 퍼스트 네트워킹(CFN)을 통합하여 구축됩니다. 주요 기술적 요소로는 그래프 표현 학습을 위한 GNN 모델인 GSAGE와 랜덤 포레스트(RF) 분류기가 사용됩니다. 이 프레임워크는 기존의 기능 선택 방법에 비해 우수한 성능을 보이며, 실험적으로 다양한 데이터 샘플링 비율에서도 높은 정확도를 유지합니다.

- **Performance Highlights**: 실험 결과, GNN-NAD는 정확도, 재현율, 정밀도, F1 점수에서 기존의 네트워크 이상 탐지 방법들을 초월하는 성능을 나타냈습니다. 특히, 적은 샘플 크기에서도 더욱 향상된 평가 지표를 달성하여, 실제 CE 및 IoT 환경에서의 적용 가능성을 제시합니다. 이 프레임워크는 차세대 지능형 CE 네트워크의 보안 및 효율성을 크게 향상시킬 것으로 기대됩니다.



### Active Control of Turbulent Airfoil Flows Using Adjoint-based Deep Learning (https://arxiv.org/abs/2510.07106)
- **What's New**: 이번 연구에서는 고급 신경망 흐름 제어기를 훈련하여 복잡한 공기역학적 흐름에서 양력을 극대화하고 항력을 최소화하는 방법을 제시합니다. 구체적으로, 압력 측정을 바탕으로 제어 정책을 최적화하며, 이는 신경망을 통해 적용됩니다. 특히, 수정된 blowing/suction jet 기술을 고정된 위치에서 사용하여 효과적인 흐름 제어가 이루어집니다.

- **Technical Details**: 연구에서는 $R_e=5	imes10^4$와 Mach 숫자 0.4의 난류 상태에서 NACA 0012 공기foil 모델의 2D 및 3D 시뮬레이션을 활용하였습니다. 직접 수치 시뮬레이션과 대형 와류 시뮬레이션(LES)을 결합하여 공기 흐름을 모델링하고, 적응형 제어가 가능하도록 신경망 구조를 설계하여 흐름의 압력 데이터에서 최적의 jet 압력을 계산합니다.

- **Performance Highlights**: 훈련된 흐름 제어기는 양력과 항력 비율을 현저히 향상시키며, 특히 $	ext{α} = 5^	ext{∘}$와 $10^	ext{∘}$에서 더욱 두드러진 개선 효과를 나타냅니다. 2D 훈련 모델이 3D 흐름에 적용되었음에도 효과적이었다는 점은 이 방법의 강인성을 보여줍니다. 이는 에너지 효율을 높이고, 적응형(신경망) 및 오프라인(상수압) 제어기 간의 성능 차이를 최소화한 결과로 나타납니다.



### Diffusion-Augmented Reinforcement Learning for Robust Portfolio Optimization under Stress Scenarios (https://arxiv.org/abs/2510.07099)
- **What's New**: 이 논문에서는Diffusion-Augmented Reinforcement Learning (DARL)이라는 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 Denoising Diffusion Probabilistic Models (DDPMs)와 Deep Reinforcement Learning (DRL)을 결합하여 포트폴리오 관리를 위한 접근 방식을 제공합니다. DDPMs를 활용하여 다양한 스트레스 강도에 기초한 합성 시장 붕괴 시나리오를 생성하여 훈련 데이터의 강건성을 획기적으로 향상시킵니다.

- **Technical Details**: DARL 프레임워크는 Proximal Policy Optimization (PPO)과 DDPM을 통합하여 포트폴리오 관리의 강건성을 높입니다. 이 방법론은 위기 시나리오를 위한 데이터 증강, 포트폴리오 weight에 대한 정규화를 포함한 맞춤형 RL 환경 및 반복적인 훈련 절차로 구성됩니다. diffusion augmentation 기술을 통해 생성된 합성 데이터를 사용하여 모델의 일반화 성능을 개선하고, 기존 모델이 자주 나타나는 데이터 분포의 변화를 잘 다루지 못하는 문제를 해결합니다.

- **Performance Highlights**: 경험적 평가 결과, DARL은 전통적인 기준 모델들보다 우수한 성과를 보여주며, 2025년 관세 위기와 같은 예측하지 못한 위기에 대한 복원력을 높였습니다. 더불어 DARL은 포트폴리오에 대한 리스크 조정 수익률을 개선하며 훈련 데이터의 다양성을 통해 위기 상황에서도 탄력성 있는 전략을 제공합니다. 이 연구는 DRL 기반의 금융 애플리케이션에서 스트레스 회복력을 강화하기 위한 강력하고 실용적인 방법론을 제시합니다.



### Explaining Models under Multivariate Bernoulli Distribution via Hoeffding Decomposition (https://arxiv.org/abs/2510.07088)
- **What's New**: 이 논문은 예측 모델의 행동을 설명하기 위한 새로운 방법으로서, Bernoulli 분포를 가진 입력 변수를 고려한 일반화된 Hoeffding 분해(generalized Hoeffding decomposition)의 존재성과 유일성을 제시합니다. 기존의 불확실성 정량화(uncertainty quantification) 관점에서 출발하여, 입력 변수들이 상관관계가 있을 때 이론적으로 L2 서브스페이스에 대한 비스듬한 투사의 개념을 기반으로 합니다. 따라서 이 연구는 예측 모델에 대한 해석 가능성 해법을 제공합니다.

- **Technical Details**: 논문에서는 입력 변수가 Bernoulli 분포를 따를 때의 Hoeffding 분해에 대해 자세히 서술합니다. 이를 통해 나타나는 L2 서브스페이스는 일차원(one-dimensional)이며, 기능 분해(functional decomposition)는 명시적(explicit)입니다. 이러한 분해 방식은 결정 지원 문제를 해결하기 위해 데이터 기반으로 해석할 수 있는 프레임워크를 제공합니다.

- **Performance Highlights**: Numerical experiments를 통해, 이 분석 기법이 이진 결정 다이어그램(binary decision diagrams), 불 대수 네트워크(Boolean networks) 또는 이진 신경망(binary neural networks)과 같은 다양한 의사 결정 지원 문제에 효과적임을 입증합니다. 또한, 이러한 연구는 고차원 문제 고찰과 유한 개수의 입력 변수에 대한 확장을 위한 새로운 관점을 제시합니다.



### Pseudo-MDPs: A Novel Framework for Efficiently Optimizing Last Revealer Seed Manipulations in Blockchains (https://arxiv.org/abs/2510.07080)
- **What's New**: 이번 연구는 Markov Decision Processes (MDPs)의 특정 문제를 해결하는 데 필요한 계산적 도전 과제를 다룹니다. 특히, Proof-of-Stake (PoS) 블록체인에서 공정성을 저해하는 Last Revealer Attack (LRA)에 의해 동기가 부여되었습니다. 우리는 이를 위해 pseudo-MDPs (pMDPs)라는 새로운 프레임워크를 소개하고, 표준 MDPs로의 두 가지 문제 축소 방법을 제안합니다.

- **Technical Details**: 제안된 두 가지 문제 축소 방법 중 하나는 새로운 관점을 제공하며, 이 두 방법을 결합하면 가치 반복(value iteration)과 같은 동적 프로그래밍 알고리즘에서 큰 개선이 가능해집니다. 특히, LRA의 경우, 계산 복잡도를 O(2^κ κ^2^(κ+2))에서 O(κ^4)로 줄이는 데 성공했습니다. 이러한 해결 방법은 동적 프로그래밍의 일반적인 이점인 최적 솔루션에 대한 지수적으로 빠른 수렴을 보장합니다.

- **Performance Highlights**: 제안하는 접근 방안은 자원이 제한된 에이전트가 메모리와 계산이 아주 제한적일 때도 잘 작동할 수 있도록 정책 추출을 간소화합니다. 본 연구는 허구의 카드 게임과 Ethereum 랜덤 시드 합의 프로토콜에서 LRA의 두 가지 사례 연구를 통해 이 프레임워크의 유용성을 검증했습니다. 이 방법은 대규모 문제를 효과적으로 해결할 수 있는 능력을 보여주며, 최적 전략에 대한 실제적인 통찰력을 제공합니다.



### Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications (https://arxiv.org/abs/2510.07077)
Comments:
          Accepted to IEEE Access, website: this https URL

- **What's New**: 최근 큰 변화가 일어나고 있는 로봇 공학 분야에서, Vision-Language-Action (VLA) 모델이 주목받고 있습니다. VLA 모델은 전통적으로 따로 연구되어온 시각, 언어, 행동 데이터를 통합하여, 다양한 작업 및 환경에서 일반화된 정책을 학습하는 것을 목표로 합니다. 이로 인해 로봇은 최소한의 추가 데이터로도 새로운 작업을 수행할 수 있는 가능성이 높아집니다.

- **Technical Details**: 이 논문은 VLA 모델의 구조적 전환 및 중앙 요소를 체계적으로 검토합니다. VLA 모델은 시각적 관찰과 자연어 지시를 입력으로 받아 로봇 액션을 직접 생성하는 시스템으로 정의됩니다. 또한, 데이터 수집, 공공 데이터셋, 데이터 증강 방법 및 평가 기준과 같은 로봇 시스템의 실질적인 배치를 지원하기 위한 다양한 요소가 포함되어 있습니다.

- **Performance Highlights**: VLA 모델은 적은 양의 특정 작업 데이터로도 다양한 로봇 임무를 수행할 수 있는 잠재력을 지니고 있습니다. 그러나 VLA 모델의 발전은 데이터 가용성, 신체적 불일치, 계산적 제약 등 여러 도전에 의해 제한되고 있습니다. 이러한 문제를 해결하기 위해 다음 세대의 로봇 시스템의 효율성과 접근성을 높이기 위한 연구가 필요합니다.



### Native Hybrid Attention for Efficient Sequence Modeling (https://arxiv.org/abs/2510.07019)
Comments:
          Technical report, 16 pages

- **What's New**: 본 연구에서 우리는 Native Hybrid Attention (NHA)라는 새로운 하이브리드 아키텍처를 소개합니다. NHA는 선형 주의와 풀 주의(intra & inter-layer hybridization)를 통합하여 단일 계층 설계로 구성되어 있습니다. 이 구조는 긴 문맥을 유지하면서도 단기간의 정보를 효과적으로 결합하여 성능 향상을 이룹니다.

- **Technical Details**: NHA는 선형 RNN에 의해 업데이트된 키-값 슬롯(key-value slots)을 통해 긴 문맥을 관리합니다. 이는 슬라이딩 윈도우(Sliding Window)의 짧은 기간 토큰들과 결합되어, 단일 softmax attention 작업을 통해 모든 키와 값에 적용됩니다. 각 레이어의 동작은 슬라이딩 윈도우 크기라는 단일 하이퍼파라미터로 조절되어, 모델 구조를 변경하지 않고도 선형 주의와 풀 주의 간의 균형을 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과 NHA는 리콜이 중요한 작업 및 상식 추론(test tasks)에서 Transformers 및 기타 하이브리드 모델을 초월하는 성능을 보여주었습니다. 사전 훈련된 LLM을 NHA 구조에 적용한 결과, 경쟁력 있는 정확도를 달성하며 효율성을 크게 향상시켰습니다. 코드 및 모델은 제공된 URL에서 확인할 수 있습니다.



### Root Cause Analysis of Outliers in Unknown Cyclic Graphs (https://arxiv.org/abs/2510.06995)
- **What's New**: 본 연구는 순환 원인 그래프(cyclic causal graphs)에서 이상치(outlier)의 전파를 분석하며, 이를 통해 여러 "근본 원인(root cause)" 노드로 추적할 수 있음을 보여줍니다. 우리는 강한 perturbation이 있을 경우, 정상 모드(normal mode)와 동일한 구조적 방정식에 따라 전파되는 이상치의 근본 원인을 식별할 수 있는 짧은 목록을 작성할 수 있음을 입증합니다. 이 방법은 인과 그래프에 대한 사전 지식을 필요로 하지 않습니다.

- **Technical Details**: 우리는 정상 상태에서의 데이터와 이상 샘플을 이용하여 비선형 순환 구조 방정식(linear cyclic SEM)을 모델링합니다. 각 변수 X_i는 계수 행렬 A와 무작위 벡터 N 간의 관계를 통해 나타내어지며, 이 구조는 방향 그래프(directed graph)로 표현됩니다. 이 논문에서는 perturbation을 작용하는 구조를 수정하여 노이즈 벡터의 극단적(e.g., extreme) 항목을 식별하는 방법을 제안합니다.

- **Performance Highlights**: 우리의 접근 방식은 이상치 관측값의 벡터에 적용되는 간단한 선형 변환을 통해 근본 원인을 드러냅니다. Li et al.(2024)의 연구와 비교할 때, 복잡한 조합 검색을 피하면서도 효과적인 방법으로, 계산 효율성을 높이는 장점을 가지고 있습니다. 이 연구 방법은 다양한 분야에서의 근본 원인 분석에 유용하게 사용될 것으로 기대됩니다.



### Relational Database Distillation: From Structured Tables to Condensed Graph Data (https://arxiv.org/abs/2510.06980)
- **What's New**: 이 논문은 Relational Database Distillation (RDD)라는 새로운 Paradigm을 제안하여, 대규모 관계형 데이터베이스를 Compact한 이질적인 그래프 형태로 압축하면서도 예측력을 유지하는 방법을 연구합니다. 주요 기술은 Multi-modal column information을 노드 특성으로 보존하고, Primary-foreign key 관계를 이질적인 엣지로 인코딩하여 데이터의 일관성과 관계 구조를 유지하는 것입니다. 이를 통해 기존의 대규모 데이터베이스에 비해 데이터 크기를 크게 줄이면서도, 분류 및 회귀 작업에서 경쟁력 있는 성능을 유지할 수 있습니다.

- **Technical Details**: 제안된 Table-to-Graph (T2G) 프레임워크는 클러스터링 기반의 사전 학습 목표를 통해 멀티 모달 속성을 독립적으로 인코딩하며, Stochastic Block Modeling (SBM)을 활용하여 테이블 간의 이질적 의존성을 포착하는 이질적인 그래프를 생성합니다. 이 과정은 파라미터 효율성을 높이고, 여러 테이블의 재구성을 피함으로써 계산 복잡성을 줄이는 데 기여합니다. 또한, Kernel Ridge Regression (KRR) 목적을 사용하여 데이터 품질을 향상시키고 클래시피케이션과 리그레션 같은 다양한 하위 작업에 대한 적응성을 보장합니다.

- **Performance Highlights**: 실험 결과, T2G는 실제 관계형 데이터베이스에서 데이터 크기를 현저히 줄이는 동시에 분류 및 회귀 테스트에서 경쟁력 있는 성능을 발휘했습니다. 이는 RDB의 효율적인 학습을 위한 효과적인 경로를 제시합니다. 연구에 따르면, 이러한 성능은 데이터의 압축 과정에서 중요한 예측 정보를 유지함으로써 특정 작업에 대한 범용성을 증대시킵니다.



### Falsification-Driven Reinforcement Learning for Maritime Motion Planning (https://arxiv.org/abs/2510.06970)
- **What's New**: 이 논문에서는 자율 선박의 안전한 작동을 위해 해상 교통 규칙 준수를 개선할 수 있는 새로운 접근 방식을 제안합니다. 제안된 방법은 신호 시간 논리(specifications)로 표현된 교통 규칙을 위반하는 적대적 훈련 시나리오를 생성하는 허위 진술(falsification)-주도 강화학습(RL) 알고리즘을 포함합니다. 이는 기존의 RL 접근 방식에 비해 향상된 규칙 준수 성과를 보여줍니다.

- **Technical Details**: 저자들은 효율적인 반례 생성 알고리즘을 개발하여 자율 선박이 해상 교통 규칙 준수를 개선할 수 있도록 합니다. 이 알고리즘은 공분산 행렬 적응 진화 전략(CMA-ES)에 기반하여, 기존 RL 정책에 의해 발생한 규칙 위반을 목표로 합니다. 이러한 허위 진술 방법론을 훈련 과정에 통합함으로써 RL 에이전트의 결정-making을 효과적으로 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 허위 진술 주도 RL 접근법이 10,000개의 시나리오에서 교통 규칙 준수를 유의미하게 향상시킨 것으로 나타났습니다. 이는 해상 항해에서의 RL 적용 시나리오 설계의 효율성을 강조하며, 실제 환경에서 자율 선박들이 더 안전하게 운항할 수 있도록 도와줍니다.



### Accelerating Sparse Ternary GEMM for Quantized LLM inference on Apple Silicon (https://arxiv.org/abs/2510.06957)
- **What's New**: 새로운 Sparse Ternary GEMM 커널은 Apple의 M 시리즈 프로세서 전용으로 최적화되었습니다. 이 구현에서는 실질적인 메모리 지역성을 향상시키기 위해 차단된(interleaved) 희소 데이터 형식을 포함하고, 명령어 수준 병렬성(ILP)을 극대화하기 위한 전략도 마련되었습니다. 기존의 희소 행렬 곱셈에 비해 성능을 최대 5.98배 향상시켜, 다양한 희소성 수준에서도 안정적인 성능을 제공합니다.

- **Technical Details**: Sparse Ternary GEMM은 Ternary Compressed Sparse Column (TCSC) 형식으로 희소 행렬을 저장하여 수행됩니다. TCSC 형식은 중복된 값을 제거하고, +1 및 -1의 비균일한 형태로 데이터를 처리하는 다수의 정수 배열로 구성됩니다. 이 접근 방법은 메모리 오버헤드를 줄이고 계산을 위해 다이나믹한 배열 접근 방식으로 분리하여 최적화한 것입니다.

- **Performance Highlights**: 성능 결과에 따르면, 스칼라 구현은 전통적인 Ternary Compressed Sparse Column(TCSC) 벤치마크에 비해 최대 5.98배의 성능 향상을 보여줍니다. 벡터화된 구현 또한 25% 희소성에서 최대 5.59배의 성능 개선을 기록하였으며, 이는 데이터 수준의 병렬성과 메모리 최적화를 효과적으로 활용한 결과입니다.



### PyCFRL: A Python library for counterfactually fair offline reinforcement learning via sequential data preprocessing (https://arxiv.org/abs/2510.06935)
- **What's New**: 본 논문에서는 오프라인 강화 학습(offline reinforcement learning, RL)에서 반사적 공정성을 보장하기 위한 Python 라이브러리인 PyCFRL을 소개합니다. PyCFRL은 반사적 공정성이 보장된 RL 정책을 학습하기 위해 새로운 데이터 전처리 알고리즘을 구현하고 있으며, RL 정책의 가치와 반사적 불공정 수준을 평가하는 도구를 제공합니다. 이 라이브러리는 PyPI와 Github에서 공개되어 있으며, 자세한 튜토리얼은 PyCFRL 문서에서 확인할 수 있습니다.

- **Technical Details**: PyCFRL은 Wang et al. (2025)가 제안한 데이터를 전처리하는 알고리즘을 구현하여 반사적 공정성을 보장하는 오프라인 RL 형성에 기여하고 있습니다. 본 알고리즘은 민감한 속성 값의 변경 여부에 관계없이 각 행동이 할당되는 확률이 변하지 않도록 새로운 상태 변수를 구성합니다. PyCFRL은 preprocessed trajectories를 생성하고, 기존의 RL 알고리즘인 Fitted Q-iteration (FQI)와 함께 사용하여 최적의 반사적 공정 정책을 학습할 수 있도록 지원합니다.

- **Performance Highlights**: PyCFRL을 활용한 데이터 예시에서는 실제 세계의 궤적 데이터로부터 반사적 공정 정책을 학습하고 평가하는 과정을 보여줍니다. 500명의 개인과 10회의 전환을 가진 궤적을 사용하여, 평균 계산 시간은 약 378.6초로 나타났습니다. PyCFRL은 시뮬레이션 환경에서 생성된 데이터를 기반으로도 동작하여 다양한 정책 학습 및 평가 워크플로우를 제공할 수 있습니다.



### Textual interpretation of transient image classifications from large language models (https://arxiv.org/abs/2510.06931)
Comments:
          Published in Nature Astronomy (2025). Publisher's Version of Record (CC BY 4.0). DOI: https://doi.org/10.1038/s41550-025-02670-z

- **What's New**: 이번 연구에서는 현대의 천문학적 조사에서 생성되는 방대한 양의 데이터에서 진짜 천체 물리학 신호와 가짜 이미징 아티팩트를 구별하는 문제를 해결하기 위해 대형 언어 모델(LLMs)을 활용하였습니다. 이러한 접근은 기존의 Convolutional Neural Networks(CNNs)의 성능을 접근하면서도 인공지능이 생성하는 설명을 통해 인간이 이해할 수 있는 형태로 결과를 제공합니다.

- **Technical Details**: 연구에서 Google의 LLM인 Gemini는 15개의 예시와 간결한 지침을 사용하여 Pan-STARRS, MeerLICHT, ATLAS와 같은 각기 다른 해상도 및 픽셀 스케일을 포함한 세 개의 광학적 과도 현상 조사 데이터셋에서 평균 93%의 정확도로 성능을 발휘했습니다. 또한 두 번째 LLM은 첫 번째 모델의 출력을 평가하여 문제 사례를 식별하고 반복적인 개선이 가능하도록 하여 데이터 처리의 효율성을 높입니다.

- **Performance Highlights**: 대형 언어 모델들은 관측된 특징에 대한 텍스트 설명을 생성함으로써 사용자가 분류를 마치 주석이 달린 카탈로그를 탐색하듯이 쿼리할 수 있도록 지원합니다. 이는 기존의 교육 파이프라인을 우회하면서 자연어와 예제를 통해 사용자 맞춤형 분류 방식을 정의할 수 있는 유연성을 제공합니다. 이러한 프레임워크는 차세대 망원경과 조사가 데이터 양을 더욱 증가시킴에 따라, 자동 탐지와 투명한 인간 수준의 이해 사이의 간극을 줄이는 데 기여할 수 있습니다.



### Quantum Sparse Recovery and Quantum Orthogonal Matching Pursu (https://arxiv.org/abs/2510.06925)
- **What's New**: 이번 연구에서 우리는 비정형(overcomplete) 사전(dictionary)을 사용한 양자 희소 복구(quantum sparse recovery)를 다룹니다. 이 문제는 NP-하드(NP-hard)로 입증되어 효율적인 정확한 알고리즘이 존재하지 않음을 보여줍니다. 이를 해결하기 위해 우리는 양자 정적 일치 추구(Quantum Orthogonal Matching Pursuit, QOMP)라는 알고리즘을 도입하였습니다.

- **Technical Details**: QOMP는 전통적인 OMP(orthogonal matching pursuit) 알고리즘의 양자 버전으로, 내부 곱(inner product) 추정, 극대값 찾기(maximum finding), 블록 인코딩(block-encoded projections) 등 다양한 양자 서브루틴을 결합한 디자인을 가지고 있습니다. 오류 집합(every iteration-to-iteration error accumulation)을 방지하는 방식으로, 특정 가정 하에 QOMP는 다항 시간(polynomial time) 내에 K-희소 상태(K-sparse state)의 정확한 지원(support)을 복원할 수 있습니다.

- **Performance Highlights**: 양자 토모그래피(quantum tomography)에 대한 첫 번째 프레임워크를 제시하며, 비정형 사전을 사용한 경우에서도 $	ilde{O}(rac{	ext{sqrt{N}}}{	ext{ε}})$의 질의 복잡도(query complexity)를 만족합니다. 특히, 정제된 상태(pure-state) 토모그래피에서는, 희소성이 높은 하위 사전을 사용할 경우 기존의 밀집(dense) 및 정규 직교 사전(orthonormal-dictionary) 설정에서의 하한을 회피합니다. QOMP는 QRAM 모델에서도 클래식 OMP 구현에 비해 다항적인 속도 향상을 제공합니다.



### Bayesian Nonparametric Dynamical Clustering of Time Series (https://arxiv.org/abs/2510.06919)
Comments:
          This work has been submitted to the IEEE for possible publication. 15 pages. 9 figures

- **What's New**: 이번 논문에서는 선형 동역학을 가진 미지의 레짐 사이를 전환하여 무한 개의 시계열 클러스터의 진화를 모델링하는 방법을 제시합니다. 베이지안 비모수적 접근법을 기반으로 계층적 디리클레 과정 (hierarchical Dirichlet process)을 사용하여 스위칭 선형 동적 시스템 (Switching Linear Dynamical System)의 매개변수를 설정하며, 각 클러스터 내에서 통계적 변화를 모델링하기 위해 가우시안 과정 (Gaussian process) 우선을 활용합니다. 이 방법은 시계열 패턴의 진화를 모형화함으로써 클러스터의 불필요한 폭발을 피하는 원칙적인 방식을 제공합니다.

- **Technical Details**: 기본 개념으로는 시계열 클러스터를 서로 다른 길이를 가진 세그먼트로 클러스터링하는 방법이 포함됩니다. 가우시안 프로세스 (GP) 우선 규칙을 사용하여 다양한 패턴을 인식하는 모델을 유도하며, 이는 모양 변화가 시간이 지남에 따라 변화하는 클러스터를 발견할 수 있게 해줍니다. 이 글에서는 또한 동적 시간 왜곡 (Dynamic Time Warping, DTW) 방법과 그 변형을 통해 시간의 비정렬을 처리하는 방법에 대해서도 논의합니다.

- **Performance Highlights**: ECG 데이터를 분석하기 위한 여러 사례 연구를 통해 제안된 접근법의 다양성과 효과성을 입증하였습니다. 실험 결과, 이 방법이 ECG 신호의 동적 변화를 효율적으로 포착하고 변별 가능성을 증가시킴을 나타냈습니다. 최종적으로, 이 방법은 클러스터링 및 정렬 과정을 동시에 수행할 수 있는 베이지안 프레임워크 안에서의 사용을 제안합니다.



### Multi-Dimensional Autoscaling of Stream Processing Services on Edge Devices (https://arxiv.org/abs/2510.06882)
- **What's New**: 이 논문은 Edge 디바이스에서 서비스 수준 목표(Service Level Objectives, SLOs)를 유지하기 위한 다차원 자동 스케일링 플랫폼(Multi-dimensional Autoscaling Platform, MUDAP)을 최초로 소개합니다. 기존 자동 스케일링 메커니즘이 주로 리소스 스케일링에만 집중하는 반면, MUDAP는 서비스 및 리소스 수준에서 세밀한 수직 스케일링을 지원하여 다양한 서비스를 최적화합니다. 또한 Regression Analysis of Structural Knowledge (RASK) 기반의 스케일링 에이전트를 통해 최적의 스케일링 작업을 추론합니다.

- **Technical Details**: MUDAP은 리소스 제약이 있는 Edge 환경에서 서비스 및 리소스 매개변수의 수직 스케일링을 지원하는 다차원 자동 스케일링 플랫폼입니다. RASK 에이전트는 처리 환경의 연속 회귀 모델을 학습하여 최적의 스케일링 결정을 유도합니다. 이 논문은 Kubernetes VPA와 강화 학습 에이전트 등 기존 자동 스케일러와 비교하여 성능을 평가하였으며, RASK는 단 20회 반복만으로도 정확한 회귀 모델을 추론할 수 있음을 보여줍니다.

- **Performance Highlights**: RASK는 여러 경쟁 처리 서비스 간의 SLO를 충족시키면서 기존 자동 스케일러보다 28% 적은 SLO 위반을 기록하며 최고의 요청 부하를 지속할 수 있었습니다. 이 과정에서 CPU 오버헤드도 거의 발생하지 않았으며, 처리의 단위 데이터를 보기 위해 단 200초의 데이터 수집 후 20회의 신속한 학습이 이루어졌습니다. 이러한 성과는 Edge 디바이스 내에서 동적 조건 하에서도 부하를 관리하는 데 있어 RASK의 효율성을 강조합니다.



### Multi-hop Deep Joint Source-Channel Coding with Deep Hash Distillation for Semantically Aligned Image Retrieva (https://arxiv.org/abs/2510.06868)
- **What's New**: 이 논문은 DeepJSCC(Deep Joint Source-Channel Coding)와 딥 해시 증류(DHD: Deep Hash Distillation) 모듈을 결합하여 다중 홉 AWGN(대칭 백색 가우시안 노이즈) 채널을 통한 이미지 전송을 새로운 방법으로 제안합니다. 이를 통해 이미지의 의미적 일관성을 향상시켜 보안 지향적인 응용을 가능하게 하며, 재구성 품질을 개선합니다. 이 새로운 접근 방식은 심상적 클러스터링(semantic clustering)을 통해 채널 노이즈로 인한 의미적 변화를 완화하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DeepJSCC는 이미지 전송을 위한 종합적인 소스-채널 코딩 방법이며, 프로세스에서 MSE(Mean Square Error)와 소스 및 재구성 이미지 간의 코사인 거리(cosine distance)를 최소화하는 훈련을 포함합니다. DHD 모듈은 이미지의 의미를 이해하고, 비슷한 의미를 가진 이미지로부터 유사한 지문(fingerprint)을 생성하여 의미적 클러스터링을 수행합니다. 이 연구는 다중 홉 DF(Decode-and-Forward) 릴레이를 위한 DeepJSCC-DHD 아키텍처를 확장하여 새로운 설계 방안을 제시합니다.

- **Performance Highlights**: 제안된 접근 방식은 채널 노이즈에 의해 발생하는 문제를 해결하면서, LPIPS(학습된 지각 이미지 패치 유사도)를 사용하여 사람들이 인지하는 품질과 일치하는 재구성 품질을 개선하는 데 기여합니다. 다중 홉 환경에서 성능을 측정한 결과, 기존의 DeepJSCC 방식보다 우수한 결과를 보이며, 특히 전위 이미지 전송과 보안 응용 분야에서도 가능성을 보여주었습니다. 이 연구는 향후 다양한 다중 홉 통신 시스템에서의 활용 가능성을 열어줍니다.



### Reconquering Bell sampling on qudits: stabilizer learning and testing, quantum pseudorandomness bounds, and mor (https://arxiv.org/abs/2510.06848)
Comments:
          51 pages, 1 figure. Comments are welcome

- **What's New**: 이 논문에서는 Bell sampling을 두 개의 차원보다 높은 qudit 시스템으로 일반화하는 방법을 개발했습니다. 이전 연구에서 이론적으로 일반화가 불가능하다고 밝혀진 이 절차를 성공적으로 확장했습니다. 특히, Lagrange의 사각형 정리에 근거한 새로운 유니타리를 도입하여 안정화 상태의 여러 복사본을 복소 공액으로 매핑할 수 있게 되었습니다.

- **Technical Details**: Bell sampling 기법은 두 개의 복사본을 벨 기저(Bell basis)에서 측정함으로써 안정화 상태를 탐지합니다. 이 논문에서는 차원 $d e 2$인 qudit를 위한 새로운 유니타리와 함께 Bell sampling 일반화 방식을 제안합니다. 이 일반화는 Pauli 연산자의 적용을 통해 안정화 상태의 복소 공액을 얻는 방식으로, 이러한 방식이 기존의 접근 방식과는 확연하게 다릅니다.

- **Performance Highlights**: 새로운 Bell sampling 기법은 다중 차원에서 다양한 응용 프로그램을 지원하도록 설계되었습니다. 예를 들어, $O(n^3)$ 시간 내에 안정화 상태를 학습할 수 있으며, Hidden Stabiliser Group Problem을 해결하는 데 필요한 시간도 대폭 단축되었습니다. 또한, 안정화 크기 및 충실도 테스트와 같은 여러 문제도 효율적으로 해결할 수 있는 방법론을 제공합니다.



### Efficient Discriminative Joint Encoders for Large Scale Vision-Language Reranking (https://arxiv.org/abs/2510.06820)
Comments:
          preprint

- **What's New**: 본 논문에서는 EDJE(Efficient Discriminative Joint Encoder)를 제안하여 시각 및 언어 모달리티의 효율적인 조합을 실현합니다. 기존의 디지털 이미지 검색에서 대량의 임베딩을 필요로 하는 기술적 제약을 해결하기 위해, 이미지 특성을 오프라인에서 미리 계산하고 경량화된 어댑터를 통해 압축하여 온라인 추론 시 성능을 극대화합니다. EDJE는 강력한 검색 성능을 유지하면서도 저장 공간과 계산 비용을 크게 줄여, 고속 추론을 가능하게 합니다.

- **Technical Details**: EDJE는 언어 모델과 함께 사용되는 압축된 비주얼 토큰을 사용하여 고급 교차 모달 상호작용을 지원합니다. 이를 통해 이미지와 텍스트 간의 상관관계를 공유하는 효과적인 임베딩 공간을 제공하며, 데이터 처리 시 대규모 검색에서의 효율성을 극대화합니다. EDJE는 50,000 개 이미지-텍스트 쌍을 1초에 처리할 수 있으며, 각 이미지당 49kB의 디스크 저장 공간을 요구합니다.

- **Performance Highlights**: EDJE는 다양한 임베딩 기반 모델과 조합하여 제로샷 검색 성능을 일관되게 개선합니다. 특히, SigLIP2와 같은 강력한 비주얼 백본을 사용했을 때, EDJE는 과거 조인트 인코더와 비슷한 성능을 보이며, 운영 효율성에서 훨씬 더 우수한 결과를 보여줍니다. 성능 평가 결과, EDJE는 표준 벤치마크(Flickr30k, MS-COCO)에서 경쟁력 있는 성능을 기록했습니다.



### BlackboxNLP-2025 MIB Shared Task: Exploring Ensemble Strategies for Circuit Localization Methods (https://arxiv.org/abs/2510.06811)
Comments:
          The 8th BlackboxNLP Workshop (Shared Task), 6 pages

- **What's New**: 본 연구는 Mechanistic Interpretability Benchmark (MIB)의 Circuit Localization 트랙에서 회로(localization)를 더 잘 식별하기 위한 여러 방법의 집합(ensemble)을 평가합니다. 두 가지 변형인 병렬(Parallel) 및 순차적(Sequential) 집합을 통해 성능을 향상시킬 수 있는지 조사했으며, 이를 통해 더 정밀한 회로 식별 방식이 가능함을 보여줍니다.

- **Technical Details**: 연구에서 사용된 방법은 EAP-IG (Edge Attribution Patch with Integrated Gradients)와 edge pruning을 포함합니다. 병렬 집합에서는 각 방법에서 할당된 score을 평균 또는 최대/최소를 통해 결합하며, 순차적 집합에서는 EAP-IG score를 이용하여 더 정밀한 edge pruning 방법을 가속화합니다. 이러한 접근 방식을 통해 다양한 모델/task 조합의 성능을 평가했습니다.

- **Performance Highlights**: 혼합 집합(hybrid ensemble) 방법이 성능 metricks에서 최상의 결과를 보였으며, Circuit Performance Ratio (CPR)와 Circuit-Model Difference (CMD) 점수에서 각각 높은 점수를 기록했습니다. 특히, hybrid ensemble 방법이 가장 낮은 CMD 점수와 가장 높은 CPR 점수를 달성해, 주어진 작업에서의 전반적인 성과를 극대화했습니다.



### Quantum Computing Methods for Malware Detection (https://arxiv.org/abs/2510.06803)
Comments:
          22 pages, 2 figures, 3 tables

- **What's New**: 이번 연구에서는 Quantum Machine Learning (QML)을 이용하여 악성 소프트웨어 탐지의 효율성을 증가시키는 양자 컴퓨팅의 잠재성을 탐구합니다. 핵심 목표는 Quantum Support Vector Machine (QSVM) 알고리즘의 성능을 기존의 Support Vector Machine (SVM) 알고리즘과 비교하는 것입니다. 연구에는 Portable Executable (PE) 파일의 원시 이진 데이터셋이 사용되어 QSVM의 분류 성능을 평가하였습니다.

- **Technical Details**: QSVM 알고리즘은 고전적인 SVM과 양자 커널을 결합하여 작동합니다. 원래 SVM 모델에 사전 계산된 양자 커널을 적합시키고 이를 고전 컴퓨터에서 학습합니다. Qiskit SDK 및 IBM 양자 컴퓨터를 통해 QSVM을 구현하였으며, 양자 하드웨어에서의 실험적 결과를 통해 양자 컴퓨터의 행동과 성능에 대한 통찰을 제공합니다.

- **Performance Highlights**: 실험 결과는 QSVM이 기존 SVM 알고리즘에 비해 큰 규모의 악성 소프트웨어 탐지 작업에서 효율성을 나타내는지를 보여줍니다. Qiskit 인터페이스를 통한 실제 양자 하드웨어 사용 경험에서는 여러 도전 과제 및 이에 대한 해결책이 상세히 설명되어 있습니다. 이 과정에서 발생한 주요 문제 및 포기된 회로의 전송 문제 등은 연구의 중요한 기여로 간주됩니다.



### MultiCNKG: Integrating Cognitive Neuroscience, Gene, and Disease Knowledge Graphs Using Large Language Models (https://arxiv.org/abs/2510.06742)
- **What's New**: 대규모 언어 모델(LLMs)의 등장으로 생물 의학 및 인지 과학 분야에서 지식 그래프(KGs)의 통합이 혁신적으로 이루어졌습니다. 이 논문에서는 MultiCNKG라는 혁신적인 프레임워크를 제안하며, 이는 다양한 지식 소스를 결합하여 복잡한 유전자, 질병 및 인지 과정 간의 의미적 연결성을 포착합니다.

- **Technical Details**: MultiCNKG는 인지 신경 과학 지식 그래프(CNKG), 유전자 온톨로지(GO), 질병 온톨로지(DO)를 포함한 세 가지 주요 지식 소스를 통합하여 만듭니다. 이 프레임워크는 LLMs(예: GPT-4)를 활용하여 엔티티 정렬(entity alignment), 의미적 유사성 계산(semantic similarity computation), 그래프 증강(graph augmentation)을 수행합니다.

- **Performance Highlights**: MultiCNKG는 6.9K 개의 노드와 11.3K 개의 엣지를 포함하고 있으며, 정밀도(precision) 85.20%, 재현율(recall) 87.30%, 커버리지(coverage) 92.18% 등 다양한 메트릭에서 우수한 성능을 보여줍니다. 논문의 평가 결과는 개인 맞춤형 의학, 인지 장애 진단 및 인지 신경 과학에서의 가설 수립 등 다양한 응용 분야에서의 가능성을 제시합니다.



### Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Managemen (https://arxiv.org/abs/2510.06727)
- **What's New**: 본 논문에서는 긴 범위의 다중 턴(tool use) 도구 사용을 위한 대형 언어 모델(LLM) 에이전트의 강화 학습(RL) 미세 조정을 연구하였습니다. 기존의 RL 파이프라인은 지침 수행 저하, 과도한 롤아웃 비용, 그리고 엄격한 컨텍스트 한계 등의 문제를 겪을 수 있습니다. 이를 해결하기 위해, 우리는 요약 기반의 컨텍스트 관리 방법을 교육에 도입했습니다.

- **Technical Details**: 이 방법은 LLM이 생성한 요약을 이용해 도구 사용 이력을 주기적으로 압축함으로써 작업과 관련된 정보를 유지하며 компакт한 컨텍스트를 유지할 수 있도록 지원합니다. 이 형식에 기반하여, 우리는 도구 사용 행동과 요약 전략을 끝에서 끝으로 최적화할 수 있는 정책 기울기(Policy Gradient) 표현을 도출했습니다. 이를 통해 	exttt{SUPO}라는 LLM RL 알고리즘을 구현하여 고정된 컨텍스트 한계를 넘어서는 긴 범위의 교육이 가능해졌습니다.

- **Performance Highlights**: 실험을 통해 상호작용하는 기능 호출 및 검색 작업에서 	exttt{SUPO}가 성공률을 크게 향상시키는 동시에 기준과 비교하여 동일하거나 심지어 더 짧은 작업 컨텍스트 길이를 유지함을 입증했습니다. 복잡한 검색 작업에 대해서는, 	exttt{SUPO}가 학습 시간보다 테스트 시간에서 최대 요약 라운드를 더욱 확장하여 평가 성과를 개선할 수 있음을 보여주었습니다. 이러한 결과는 요약 기반의 컨텍스트 관리가 고정된 컨텍스트 길이 한계를 넘어 RL 에이전트를 훈련시키기 위한 원칙적이고 확장 가능한 접근 방식임을 입증합니다.



### Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG) (https://arxiv.org/abs/2510.06719)
Comments:
          Under review

- **What's New**: 이 논문은 Differentially Private Synthetic Retrieval-Augmented Generation (DP-SynRAG)이라는 프레임워크를 제안합니다. 이 프레임워크는 대규모 언어 모델(LLMs)을 사용하여 차별적 프라이버시(Privacy)를 보장하는 합성 RAG 데이터베이스를 생성합니다. 기존의 방법과 달리, 생성된 합성 텍스트는 반복적으로 사용할 수 있어 재노이즈 주입과 추가적인 프라이버시 비용을 피할 수 있습니다. DP-SynRAG는 RAG 작업에 필요한 핵심 정보를 보존하면서 성능을 개선하는 특성도 가지고 있습니다.

- **Technical Details**: DP-SynRAG는 LLM을 사용하여 데이터베이스의 서브샘플 기록을 모방하는 텍스트를 생성하는 맞춤형 예측(private prediction)을 통해 고품질의 사적인 텍스트 생성을 달성합니다. 이 과정에서 문서 레벨 임베딩(document-level embeddings)과 키워드 기반 클러스터링을 사용하여 의미적으로 유사한 문서들을 그룹화합니다. 이렇게 생성된 데이터는 이후의 RAG 작업에 사용할 수 있으며, 추가적인 프라이버시 비용을 발생시키지 않도록 설계되었습니다. 저자들은 세 가지 데이터셋을 사용해 DP-SynRAG의 성능을 검증하며 기존의 개인 RAG 방법들을 초월하는 결과를 보고합니다.

- **Performance Highlights**: DP-SynRAG는 고정된 프라이버시 예산을 유지하면서 기존의 최첨단 개인 RAG 시스템보다 뛰어난 성능을 보여줍니다. 이 방법은 스케일러블한 솔루션을 제공하여 프라이버시를 보장하는 RAG 애플리케이션에 적합합니다. 실험 결과는 DP-SynRAG가 다수의 질의에 대해 높은 효용성과 우수한 데이터 품질을 유지함을 입증합니다. 이러한 특징들은 이 논문의 기여가 민감한 정보를 다루는 데이터베이스에서 실질적으로 사용될 수 있음을 시사합니다.



### Inefficiencies of Meta Agents for Agent Design (https://arxiv.org/abs/2510.06711)
- **What's New**: 최근 연구들은 메타 에이전트를 활용해 에이전트 시스템의 설계를 자동화하는 방향으로 나아가고 있습니다. 이 논문에서는 메타 에이전트의 세 가지 주요 과제를 분석하며, 기존 연구에서 언급된 방법들이 실제로는 이전 설계를 보다 잘 활용하지 못함을 보여줍니다. 단순히 이전 에이전트를 포함하는 것이 덜 효과적임을 입증하며, 진화적 접근법이 더 나은 성능을 발휘함을 강조합니다.

- **Technical Details**: 메타 에이전트는 반복적인 샘플링(sample), 평가(evaluate), 반복(iterate) 패턴을 따르며, 이를 통해 새로운 에이전트를 설계합니다. 저자들은 메타 에이전트의 성능을 향상시키기 위해 세 가지 다른 컨텍스트 큐레이션(context curation) 전략을 사용하여 실험하였습니다. 특히, 진화적 맥락 큐레이션이 성능을 개선하며, 특히 MMLU와 DROP 데이터셋에서 15,000개의 예제에서 경제적 타당성을 분석합니다.

- **Performance Highlights**: 실험 결과, 누적 컨텍스트 큐레이션이 평행 컨텍스트 큐레이션보다 더 낮은 성능을 보였으며, 이는 기존 설계를 거의 활용하지 못하는 것을 나타냅니다. 진화적 맥락 큐레이션은 MGSM에서 +10%의 성능 향상을 가져와, 높은 품질의 이전 설계를 사용하는 것이 메타 학습을 더 효과적으로 만들어줌을 시사합니다. 하지만 설계된 에이전트 간의 행동 다양성이 결여되어 있어, 모든 쿼리에 대해 최적의 에이전트를 동적으로 선택하기 어려운 상황임을 알 수 있습니다.



### Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks (https://arxiv.org/abs/2510.06695)
- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)에 대한 관심이 증가함에 따라, 프롬프트 엔지니어링(prompt engineering)이 수동 설계에서 모델 기반 최적화로 발전하였습니다. 본 논문에서는 기계 번역(machine translation)과 같은 특정 작업에 적합한 새로운 프롬프트 최적화 방법을 소개합니다. 제안된 방법은 작은 매개변수 모델을 활용하여 백 트랜슬레이션(back-translation) 전략으로 학습하여, 단일 작업 최적화를 위한 훈련 비용을 대폭 줄이는 동시에 높은 성과를 제공합니다.

- **Technical Details**: LLM의 프롬프트는 일반적으로 instruction와 input의 두 가지 구성 요소로 이루어져 있습니다. 본 논문에서는 입력 최적화를 위한 Rewriting Original Inputs (ROI) 전략을 제안합니다. 이 방법은 LLM 또는 소형 매개변수 모델을 활용하여 원본 입력을 재구성하고, 언어 모델에 더 잘 맞도록 조정합니다. 특히, 입력의 질을 평가하기 위한 필터링 모듈도 도입되어 의미 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, ROI 모듈은 애매한 데이터를 보다 명확한 입력 프롬프트로 변환하는 데 효과적임을 보여줍니다. NLU 및 NLG 작업에 대한 성능 향상을 입증하였으며, 기존의 프롬프트 최적화 방법은 입력 구성 요소가 중요한 작업에 한계가 있다는 점을 보여주었습니다. ROI 방법은 다양한 LLM에 널리 적용 가능하며, 원본 입력에 비해 일관되고 주목할 만한 성과 향상을 이룹니다.



### Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer (https://arxiv.org/abs/2510.06691)
Comments:
          10 pages, 5 figures, accepted at the NeurIPS 2025 workshop "Machine Learning and the Physical Sciences"

- **What's New**: 이 연구에서는 중입자 충돌(high-energy nuclear physics)의 고차원 최종 상태 데이터를 분석하기 위한 Transformer 기반의 오토인코더(autoencoder)를 도입했습니다. 이 모델은 자기 지도학습(self-supervised learning) 사전 훈련(pre-training)과 지도학습(fine-tuning)을 결합한 두 단계의 패러다임을 사용하여, 라벨이 없는 HIC 데이터로부터 물리적으로 유의미한 구조를 학습합니다. 이를 통해 기존의 방법들보다 훨씬 높은 분류 정확도를 달성하면서도, 전통적 관측량(observables)을 재현하는 것에 그치지 않고 비선형(non-linear) 패턴까지 학습하는 가능성을 보여줍니다.

- **Technical Details**: 연구에서 사용된 데이터는 AMPT 모델에서 생성된 p+Pb 및 Pb+Pb 충돌 이벤트로 구성됩니다. 각 이벤트는 세 가지 운동량(px, py, pz)을 가진 128개의 최종 상태 입자들의 포인트 클라우드(point cloud)로 표현되며, 이는 포물선 하늘의 kinematic window 내에서 제한됩니다. 사전 훈련 단계에서는 25%의 데이터 포인트를 랜덤하게 마스킹(masking)하고, 남은 데이터로부터 특징을 추출하는 구조의 마스킹 오토인코더를 설계하여 고급 물리적 특성을 학습했습니다.

- **Performance Highlights**: 모델의 성능은 PointNet과 비교하여 모든 테스트 범위에서 상당히 높은 분류 정확도로 나타났습니다. 특히, 주요 성분 분석(principal component analysis) 및 SHAP 해석 결과는 오토인코더가 비선형(non-linear) 상관관계를 잘 포착하여 유의미한 특징을 학습했다는 것을 입증했습니다. 이러한 우수한 성능은 기존의 물리적 관측량과의 연관성이 낮지만 SHAP 분석에서 주요 기여자로 나타나는 특징들이 복잡한 구조를 학습했음을 강조합니다.



### Gaussian Equivalence for Self-Attention: Asymptotic Spectral Analysis of Attention Matrix (https://arxiv.org/abs/2510.06685)
- **What's New**: 본 연구는 현대 딥 뉴럴 네트워크의 기초 구성 요소인 self-attention 레이어에 대한 이론적 이해를 심화시키기 위한 rigor (정밀한) 분석을 제공합니다. 특히, 주목 매트릭스의 singular value spectrum (특이값 스펙트럼)에 대한 첫 번째 Gaussian equivalence (가우시안 동등성) 결과를 도출했습니다. 이를 통해 attention 매트릭스의 singular value distribution (특이값 분포)가 트랙터블한 선형 모델로 비대칭적으로 표현될 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 역온도(inverse temperature)가 일정한 범위를 유지하는 자연적인 환경에서 작업합니다. attention 매트릭스의 singular value distribution이 Marchenko-Pastur 법칙과 다른 분포를 보인다는 점도 주목해야 합니다. 이러한 분석은 normalization term (정규화 항)의 변동성을 정밀하게 제어하고, 지수 함수의 유리한 Taylor expansion (테일러 전개)을 활용한 정제된 선형화를 기반으로 합니다.

- **Performance Highlights**: 연구는 linearlization (선형화)의 임계값을 식별하고, attention이 항목별(entrywise) 연산이 아님에도 불구하고 이 영역에서 rigorous Gaussian equivalence (엄밀한 가우시안 동등성)를 갖는 이유를 분명히 밝혔습니다. 이러한 결과는 self-attention 레이어의 이론적 기초를 확립하고, 향후 연구에 중요한 기초 자료를 제공합니다.



### Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback (https://arxiv.org/abs/2510.06677)
Comments:
          Accepted at EMNLP 2025 Industry Track

- **What's New**: 본 논문에서는 고객 지원 에이전트를 위한 점진적 요약 시스템을 소개합니다. 이 시스템은 대화 중 요약 노트를 생성해야 할 최적의 순간을 지능적으로 판단하여 에이전트의 맥락 전환 노력과 중복 리뷰를 줄입니다. Mixtral-8x7B 모델과 DeBERTa 기반 분류기를 결합하여 실시간으로 효과적인 요약 생성을 제공합니다.

- **Technical Details**: 시스템은 여러 채널의 대화를 통합하여 지속적으로 요약된 글머리 기사를 생성합니다. 자동화된 요약 모델은 중요 정보가 감지될 때만 요약을 제안하고, 비실질적인 내용을 걸러내는 분류기를 통해 자동으로 최적화됩니다. 에이전트의 실시간 수정 기능은 지속적으로 모델 학습을 보강합니다.

- **Performance Highlights**: 생산 환경에 배포된 이 시스템은 평균 처리 시간을 3% 단축시키며, 복잡한 사례에서는 최대 9%까지 감소시켰습니다. 또한, 설문 조사에서 80% 이상의 에이전트 만족도를 기록하며, 점진적 요약이 에이전트의 생산성을 향상시키는 효과를 입증합니다.



### Fitzpatrick Thresholding for Skin Image Segmentation (https://arxiv.org/abs/2510.06655)
Comments:
          Accepted to MICCAI 2025 ISIC Workshop. 24 minute Oral presentation given. Awarded "Best Paper - Honorable Mention"

- **What's New**: 이번 연구에서는 피부 발진의 면적(BSA)을 정확하게 추정하기 위한 새로운 접근 방식을 제시합니다. 특히, 피부 톤에 따른 차별화를 통해 피부 발진 세분화의 성능을 개선하여 의료 치료의 형평성을 높이기 위한 노력이 포함되어 있습니다. Fitzpatrick 피부 톤 분류기를 도입하여 맏과적인 결정 임계값을 설정함으로써 음성 예측을 최소화하고자 하였습니다.

- **Technical Details**: 연구에서는 631명의 환자에서 수집된 754장의 건선 이미지를 기반으로 한 대규모 공개 데이터셋을 구축하였습니다. 이를 통해 U-Net, ResU-Net, SETR-small의 세 가지 모델을 훈련시켰으며, 각각의 모델은 Fitzpatrick 피부 톤에 따른 임계값을 통해 더 나은 성능을 발휘할 수 있는 기회를 제공합니다. 이를 통해 세그멘테이션(mask) 정확성을 증가시킬 수 있는 토대를 마련했습니다.

- **Performance Highlights**: 세분화 성능은 Fitzpatrick VI 그룹의 경우 U-Net에서 +31% bIoU 및 +24% Dice의 향상을 보여주었습니다. ResU-Net과 SETR-small 또한 지속적인 성능 증대를 보였으며, 이러한 향상은 각 피부 톤 별로 결정된 임계값 때문임을 확인할 수 있었습니다. 이는 임상 실무에서의 중요성을 고려할 때 매우 중요한 발전으로, 더 나은 치료 결정과 환자 결과를 이끌어낼 수 있는 가능성을 내포하고 있습니다.



### Q-Learning with Fine-Grained Gap-Dependent Regr (https://arxiv.org/abs/2510.06647)
- **What's New**: 이번 연구에서는 모델 없이 신뢰 학습(Model-Free Reinforcement Learning)을 위한 보다 정밀한 gap-dependent regret bounds가 제시되었습니다. 기존의 알고리즘들은 미니맥스(minimax) 최악의 경우 regret을 달성했지만, gap-dependent bounds는 전체적인 구조를 포착하지 못하는 한계가 존재했습니다. 이를 보완하기 위해 새로운 분석 프레임워크를 수립하고, UCB(Hoeffding 기반)와 비-UCB 기반 알고리즘 모두에 대해 세밀한 regret bounds를 개발했습니다.

- **Technical Details**: 연구의 주된 초점은 비균질적 전이 커널을 가진 에피소드 형태의 탭 형(episodic tabular) MDPs에 대한 것입니다. MDP에는 SS 상태, AA 행동 및 HH 단계가 포함됩니다. UCB 기반 알고리즘의 경우, 우리는 최적 및 비최적 상태-행동 쌍을 명확히 구분하여 UCB-Hoeffding에 대한 최초의 세밀한 regret 상한을 끌어낼 수 있는 새로운 분석 방법론을 도입했습니다. 비-UCB 기반인 AMB는 알고리즘 설계 및 분석에서 두 가지 주요 문제를 가지고 있음을 식별하였고, 개선된 AMB를 제안하여 첫 번째 세밀한 regret 상한을 수립했습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 UCB-Hoeffding 및 ULCB-Hoeffding 알고리즘은 성능 개선을 보여줍니다. 또한, 개정된 AMB는 알고리즘적 개선과 함께 비-UCB 기반 메소드에 대한 최초의 세밀한 gap-dependent regret를 구축하였고, 이는 empirically AMB보다 우수한 성과를 나타냅니다. 전반적으로 새로운 접근법은 모델 없는 신뢰 학습의 성능을 크게 향상시키는 것을 목표로 하고 있습니다.



### A Comparative Analysis of Contextual Representation Flow in State-Space and Transformer Architectures (https://arxiv.org/abs/2510.06640)
- **What's New**: 본 논문은 State Space Models (SSMs)와 Transformer-Based Models (TBMs) 간의 표상 전이 분석을 위한 첫 번째 통합 비교 연구를 제시합니다. 이 연구는 SSMs와 TBMs의 특징을 비교하면서 두 아키텍처가 문맥 정보의 흐름을 어떻게 처리하는지에 대해 심층적으로 분석합니다. 특히, SSMs는 초기 단계에서 토큰의 독창성을 유지하는 반면, TBMs는 빠르게 동질화되며, 이는 미래의 모델 및 훈련 설계에 중요한 통찰력을 제공합니다.

- **Technical Details**: 연구자들은 SSMs와 TBMs의 토큰 및 레이어 수준에서의 표상 전이를 조사하기 위해 Centered Kernel Alignment (CKA)와 안정성 지표, 프로빙(probing) 기법을 활용했습니다. 또한, TBMs에서의 과도 평활화(oversmoothing)는 건축 설계에서 기인하며, SSMs의 경우 주로 훈련 동역학에서 기인함을 이론적으로 분석했습니다. 이 이러한 고찰은 두 아키텍처의 귀납적 편향을 명확히 하고, 장기 문맥 추론을 위한 보다 효과적인 모델 설계에 기여할 수 있는 정보를 제공합니다.

- **Performance Highlights**: 실험에서는 Pile 데이터셋에서 사전 훈련된 TBMs와 SSMs 간의 정보 흐름을 비교하고, 롱컨텍스트 작업을 위한 두 개의 벤치마크 테스트를 사용하여 모델 성능을 평가했습니다. 결과적으로, 중간 레이어가 최종 레이어보다 더 효과적이라는 것이 드러났으며, 다양한 작업, 모델 규모 및 문맥 길이에 걸쳐 이러한 경향이 일관되었습니다. 이는 모델의 깊이가 완전히 활용되지 않을 수 있다는 것을 나타내며, 이러한 인사이트는 향후 모델 설계에 중요한 지침이 될 것입니다.



### Unsupervised Backdoor Detection and Mitigation for Spiking Neural Networks (https://arxiv.org/abs/2510.06629)
Comments:
          To appear in The 28th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2025)

- **What's New**: 스파이킹 신경망(Spiking Neural Networks, SNNs)은 인공지능 분야에서 에너지 효율이 뛰어난 모델로 주목받고 있습니다. 본 논문에서는 SNNs의 보안 문제가 특히 백도어 공격(backdoor attacks)과 관련하여 분석되지 않았음을 지적하며, 기존의 방어 방법들이 SNNs에서는 효과적이지 않음을 설명합니다. 이를 바탕으로 새로운 프레임워크인 Temporal Membrane Potential Backdoor Detection (TMPBD)와 Neural Dendrites Suppression Backdoor Mitigation (NDSBM)을 제안합니다.

- **Technical Details**: TMPBD는 이벤트 주도적이고 시간에 의존하는 SNN의 특성을 고려하여, 공격 지식이나 데이터 접근 없이도 목표 레이블을 감지할 수 있는 비지도(unsupervised) 탐지 방법입니다. TMPBD는 최종 스파이킹 계층에서의 시계열 막전위(Temporal Membrane Potential, TMP)의 최대 여유 통계(maximum margin statistics)를 활용하여 백도어 공격을 탐지합니다. NDSBM은 초기 합성곱 층(convolutional layers) 간의 덴드라이트(dendrite) 연결을 조절하여 악의적인 뉴런을 억제하고 동시에 정상적인 행동을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 논문에서 제안한 TMPBD는 다양한 악의적인 백도어 공격에 대해 100%의 탐지 정확도를 달성하였으며, NDSBM은 공격 성공률(attacker success rate, ASR)을 100%에서 8.44%로 줄여줍니다. TMPBD와 NDSBM을 함께 사용할 경우, ASR은 평균적으로 2.81%로 감소하며, 클린 정확도(clean accuracy)를 저하시키지 않고도 막대한 성능 개선을 실현합니다. 이러한 실험 결과는 SNNs에서의 백도어 공격 방어 메커니즘의 필요성과 효과를 입증합니다.



### FEAorta: A Fully Automated Framework for Finite Element Analysis of the Aorta From 3D CT Images (https://arxiv.org/abs/2510.06621)
- **What's New**: 이 논문은 대동맥류(Thoracic Aortic Aneurysm)로 인한 사망 위험을 평가하기 위한 새로운 접근 방식을 소개합니다. 기존에는 복잡한 3D 재구성이 필요했으나, 본 연구에서는 3D CT 이미지로부터 직접 환자 맞춤형 유한 요소 메쉬(Finite Element Mesh)를 생성할 수 있는 최종 딥 뉴럴 네트워크(Deep Neural Network)를 개발했습니다. 이를 통해 현재의 기술적 한계를 극복하고 환자 개별 맞춤 솔루션을 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 논문에서 제안한 방법은 PyTorch FEA 라이브러리와 유한 요소 해석(Finite Element Analysis, FEA) 기능을 통합하여 사용합니다. 기존의 전통적인 FEA 시뮬레이션에 비해 계산 시간을 현저하게 줄일 수 있으며, статическое детерминирование(static determinacy) 원리를 적용해 각 케이스 당 약 3분으로 단축했습니다. 딥 뉴럴 네트워크와의 결합을 통해 계산 시간이 몇 초로 더욱 줄어들었습니다.

- **Performance Highlights**: 이 연구 결과는 대동맥류의 파열 위험을 더욱 신속하고 효율적으로 평가할 수 있는 혁신적 방법을 제시합니다. 기존 방식에 비해 계산 시간의 크게 단축됐다는 점은 임상적 적용의 가능성을 한층 높여줍니다. 마지막으로, 이를 통해 대규모 환자군에 대한 스케일링(scaling)을 용이하게 하고, 임상 세팅에서의 사용성을 극대화할 수 있습니다.



### SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation (https://arxiv.org/abs/2510.06596)
- **What's New**: 본 논문은 Synthetic Dataset Quality Metric (SDQM)을 소개하여, 객체 탐지(object detection) 작업에서 생성된 데이터의 품질을 평가합니다. SDQM은 객체 탐지 모델이 훈련되는 동안 데이터의 효과성을 예측할 수 있는 효율적인 메트릭으로, 최신 AI 기법을 활용하여 실제 데이터의 특성을 잘 반영하도록 설계되었습니다. 이 메트릭은 비용이 많이 드는 반복 훈련의 필요성을 줄여주고 자원의 제약이 있는 상황에서 중요한 도전 과제를 해결하는 데 기여합니다.

- **Technical Details**: SDQM은 객체 탐지 데이터셋의 유용성을 평가하기 위해 여러 독립적인 구성 요소들로 구성됩니다. 이 메트릭은 픽셀 공간, 공간 공간, 주파수 공간, 특징 공간 도메인 간의 갭을 평가하며, 기존 메트릭과 비모수적 분포 비교 기술을 사용하여 데이터 세트의 효과를 정량화합니다. 실험 결과, SDQM은 YOLOv11 모델과의 평균 평균 정밀도(mean Average Precision, mAP) 점수와 강한 상관관계를 보여줍니다.

- **Performance Highlights**: SDQM은 훈련된 모델의 성능과 강하게 연관되어 있으며, 이는 랜덤 플레인스(RarePlanes), 산업 금속 물체 데이터셋(DIMO), WASABI와 같은 다양한 데이터셋에서 일관된 성능 향상으로 입증되었습니다. 이 메트릭은 자원을 효율적으로 활용하고 반복적인 훈련/검증 사이클에 대한 의존성을 줄여주어, 저비용으로 데이터 품질을 개선할 수 있는 방향을 제시합니다.



### Adapting Quantum Machine Learning for Energy Dissociation of Bonds (https://arxiv.org/abs/2510.06563)
- **What's New**: 이번 연구는 이온화 에너지(bond dissociation energies, BDE)의 예측을 위한 양자 및 고전 머신러닝 모델을 비교하는 시스템화된 벤치마크를 제시합니다. 연구팀은 화학적으로 선별된 특성 세트를 이용해 복잡한 분자의 BDE를 보다 정확하고 신뢰성 있게 예측할 수 있는 방법을 제시합니다. 특히 Qiskit Aer 플랫폼을 통해 다양한 양자 모델들이 평가되었으며, 이는 화학적 특성 예측에 새로운 기준을 제공하는 연구입니다.

- **Technical Details**: 연구에서는 양자 회귀 모델(Variational Quantum Regressors, VQR), 양자 서포트 벡터 회귀 모델(Quantum Support Vector Regressors, QSVR), 양자 신경망(Quantum Neural Networks, QNN), 및 양자 랜덤 포레스트(Quantum Random Forests, QRF)와 같은 다양한 양자 머신러닝 모델이 사용되었습니다. 이 모델들은 고전적인 서포트 벡터 회귀(Support Vector Regression, SVR) 및 다층 퍼셉트론(Multi-Layer Perceptrons, MLP) 모델과 비교되어 BDE 예측의 성능을 평가받았습니다. 모델은 고전적 방법과 양자 방법 모두에서 오류 분석 및 성능 향상을 위한 표준화된 사전 처리 기법을 적용받았습니다.

- **Performance Highlights**: QCNN 및 QRF 모델은 70-100 kcal/mol 범위의 중간 BDE에서 RF 및 MLP 모델과 동등한 정확도를 달성함으로써 화학적 예측성에서 안정성과 효율성을 보여주었습니다. 이 연구의 결과는 DSC 설계 및 환경 지속 가능성 향상을 위한 양자 강화 모델의 활용 가능성을 제시합니다. 연구자들은 양자 머신러닝이 화학적 정확도에 근접하는 예측의 주요 도구로 자리잡을 수 있음을 확고히 하였습니다.



### From Acceleration to Saturation: Scaling Behavior of Bootstrapped Language Model Pretraining (https://arxiv.org/abs/2510.06548)
Comments:
          22 pages, 11 figures, an abridged version to appear in NeurIPS 2025 LLM Evaluation Workshop

- **What's New**: 이번 연구에서는 부트스트랩 사전 훈련(bootstrapped pretraining)의 확장 효율성이 구체적으로 저하된다는 것을 발견하였습니다. 즉, 두 번째 단계의 사전 훈련 시, 기본 모델이 훈련된 토큰 수에 따라 로그적으로 감소하는 스케일링 지수를 보여주고 있습니다. 또한, 과도하게 훈련된 모델에 대해 부트스트랩의 효과가 감소할 수 있다는 점을 강조합니다.

- **Technical Details**: 연구에서는 다양한 크기의 언어 모델에 대해 광범위한 실험을 수행하였으며, 부트스트랩 사전 훈련 방법의 스케일링 행동을 분석하였습니다. 각 단계에서의 훈련 토큰 수(D1, D2)에 따른 검증 손실(LL) 변화를 측정하고, 이로부터 수렴 현상에 대한 수학적 모델을 제시하였습니다. 특히, 활성화 함수로는 SwiGLU 사용 및 Rotary position embeddings를 적용하여 LLaMA와 유사한 아키텍처를 구현하였습니다.

- **Performance Highlights**: 부트스트랩 사전 훈련이 제공하는 이점은 기본 모델이 충분히 훈련되었을 때 감소하는 경향이 있다는 결과를 도출하였습니다. 스케일링 법칙을 통해 부트스트랩이 유리한 경우와 그렇지 않은 경우를 정량적으로 평가할 수 있는 지침을 제공하며, 전체적으로 파라미터와 데이터 집합 크기의 변화를 통합하여 설명할 수 있음을 보여주었습니다. 이 연구는 언어 모델 훈련의 효율성을 높이는 데 중요한 통찰을 제공합니다.



### Cluster Paths: Navigating Interpretability in Neural Networks (https://arxiv.org/abs/2510.06541)
- **What's New**: 이번 연구에서는 딥러닝 모델의 해석 가능성을 높이기 위한 새로운 방법인 cluster paths를 제안합니다. 이 방법은 훈련된 신경망의 특정 레이어에서 클러스터링을 통해 활성화 패턴을 그룹화하고, 각 입력을 클러스터 ID의 시퀀스로 표현함으로써 모델의 내부 의사결정 과정을 시각화합니다. 기존의 예제 및 그래디언트 기반 접근법과 비교했을 때, cluster paths는 네트워크가 샘플을 변환하는 과정을 간결하게 요약하여 그 과정을 이해하고 시각화할 수 있게 합니다.

- **Technical Details**: 이 방법은 각 레이어의 활성화를 클러스터링하여 클러스터 ID 시퀀스로 입력을 인코딩하며, 이를 통해 두 입력이 같은 클러스터 시퀀스를 따라간다면 네트워크는 이를 유사하게 처리할 것이라는 가정을 세웁니다. 클러스터 경로는 모델의 내부 논리를 요약하는 압축된 프록시로 기능하며, 복잡한 모델을 수십 개의 클러스터로 단순화할 수 있습니다. 이 연구에서는 path complexity, weighted-path purity, decision-alignment faithfulness, path agreement 등 네 가지 새로운 메트릭을 도입하여 클러스터 경로를 평가합니다.

- **Performance Highlights**: 실험 결과, cluster paths는 CIFAR-10에서 색상 기반 단서를 식별하고, 해당 단서가 제거되면 경로가 붕괴됨을 보여주었습니다. CelebA의 경우 90%의 신뢰도와 Gaussian 노이즈 하에서도 96%의 일치를 유지하면서 정확도를 희생하지 않았습니다. 이 방법은 Vision Transformer와 같은 대형 모델에도 확장 가능하며, 시각적 개념을 여러 네트워크 깊이에서 발견하는 데 효과적입니다.



### Auto-Prompt Ensemble for LLM Judg (https://arxiv.org/abs/2510.06538)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 평가자의 신뢰성을 개선하는 새로운 프레임워크인 Auto-Prompt Ensemble (APE)을 제안합니다. 기존의 LLM 평가자는 인간의 평가 기준을 파악하지 못해 중요한 평가 차원을 놓치기도 했습니다. APE는 이러한 문제를 해결하기 위해 자동으로 평가 차원을 학습하고, 신뢰도를 기반으로 한 앙상블 메커니즘을 통합했습니다.

- **Technical Details**: APE의 핵심은 실패 사례를 분석하여 새로운 평가 차원을 자동 생성하고, 이를 바탕으로 LLM의 판단을 보완하는 것입니다. 초기에 LLM은 실패 사례를 식별하고, 이후에는 LLM 자체가 새로운 평가 차원과 해당 점수 기준을 생성합니다. 추가된 차원들은 Collective Confidence 메트릭을 통해 평가되며, 이는 다양한 평가 차원들의 집합 신뢰성을 정량화하여 최종 판단의 정확성을 향상시킵니다.

- **Performance Highlights**: APE는 여러 LLM 평가 기준에서 상당한 성과를 보였습니다. 예를 들어, Skywork Preference 데이터세트에서 APE를 적용함으로써 GPT-4o의 인간 기호와의 일치율이 83.6%에서 86.2%로 증가했으며, Reward Bench에서는 87.2%에서 90.5%로 향상되었습니다. 이러한 결과는 APE가 모델 평가 기준과 인간 평가 기준 사이의 격차를 줄이는데 효과적임을 보여줍니다.



### Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them (https://arxiv.org/abs/2510.06534)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 이용한 Agentic search의 새로운 접근 방식을 제안합니다. 이는 복잡한 사용자 정보 요구를 해석하고 다단계 계획, 검색, 정보 종합 과정으로 답변을 제공하는 과정을 포함합니다. 또한, 성공적인 Agentic search 경로를 분석하고 네 가지 유익한 추론 행동 특성을 식별합니다.

- **Technical Details**: 우리는 네 가지의 유익한 추론 행동: 정보 검증(Information Verification), 권위 평가(Authority Evaluation), 적응 검색(Adaptive Search), 오류 복구(Error Recovery)를 도출합니다. 이를 통해 Behavior Priming이라는 기법을 제안하며, 이는 감독 학습(Supervised Fine-Tuning, SFT) 후 강화 학습(Reinforcement Learning, RL)을 사용하여 Agentic search 모델을 훈련하는 데 사용됩니다.

- **Performance Highlights**: 세 가지 벤치마크(GAIA, WebWalker, HLE)에서의 실험 결과, 행동 프라이밍은 Llama3.2-3B와 Qwen3-1.7B에서 RL로 직접 훈련한 Agentic search 모델에 비해 35% 이상의 성능 향상을 보여줍니다. 특히, SFT 데이터 내에서의 원하는 추론 행동이 최종 성능에 중요한 요소임을 입증하며, 올바른 답변 대신 바람직한 추론 행동을 포함한 경로에서 미세 조정하는 것이 더 나은 성능으로 이어진다고 설명합니다.



### From Description to Detection: LLM based Extendable O-RAN Compliant Blind DoS Detection in 5G and Beyond (https://arxiv.org/abs/2510.06530)
- **What's New**: 5G 모바일 통신의 품질과 경험이 크게 향상되었으며, 이러한 개선은 5G 시대를 넘어 계속될 것으로 예상됩니다. 그러나 라디오 리소스 제어(RRC) 및 비접속 계층(NAS) 프로토콜의 취약점은 Blind DoS 공격과 같은 주요 보안 위협을 초래합니다. 본 논문에서는 대규모 언어 모델(LLMs)의 기능을 활용하여 O-RAN 아키텍처 내에서 비정형 데이터와 짧은 자연어 공격 설명을 이용한 새로운 이상 탐지 프레임워크를 제안했습니다.

- **Technical Details**: 우리는 비정형 RRC/NAS 데이터에 대해 제로샷(zero-shot) 방식을 적용한 새로운 이상 탐지 접근 방식을 소개합니다. LLM의 제로샷 분류 기능을 활용하여 기계 학습 모델의 사전 훈련 없이 공격 여부를 결정하고 짧은 설명을 반환합니다. 본 프레임워크는 사용자 인터페이스(UI)와의 조화를 이루며, AI 에이전트를 활용한 프로세스 자동화의 가능성을 보여줍니다.

- **Performance Highlights**: 우리는 RRC/NAS 데이터셋을 이용해 우리의 모델을 평가하고, 오픈 소스 및 독점 LLM 구현을 비교하여 공격 탐지에서 우수한 성능을 나타냄을 입증했습니다. 또한 O-RAN의 실시간 제약을 고려하여 난이도의 다른 Layer-3 공격도 탐지할 수 있는 가능성을 보여주었습니다. 이 접근 방식은 30ms 내에 분류를 결정하며, 이는 기존의 타이밍 제약(10ms - 1s) 내에서 수행될 수 있습니다.



### BACHI: Boundary-Aware Symbolic Chord Recognition Through Masked Iterative Decoding on Pop and Classical Music (https://arxiv.org/abs/2510.06528)
Comments:
          Under review

- **What's New**: 이번 논문에서는 자동 화음 인식(Automatic Chord Recognition, ACR) 분야에서의 두 가지 주요 도전을 해결하기 위해 새로운 데이터를 제안합니다. 첫 번째로, POP909 데이터셋의 향상된 버전인 POP909-CL을 소개하며, 이는 템포에 맞춘 콘텐츠와 인공지능으로 교정된 화음 레이블을 포함하고 있습니다. 두 번째로, BACHI라는 새로운 상징적 화음 인식 모델을 제안하여, 인식 작업을 경계 탐지와 반복적인 순위 결정 단계로 분할합니다.

- **Technical Details**: BACHI는 MIDI 토큰을 이용하여 작동하며, 두 가지 주요 구성 요소를 통합합니다. 첫 번째 구성 요소는 화음 변경 가능성을 예측하는 경계 탐지 모듈이며, 두 번째는 계층적 디코더(Transformer decoder)로서 화음의 뿌리(root), 품질(quality), 베이스(bass)를 순차적으로 예측합니다. 이러한 설계는 인간의 시청 및 귀 훈련 과정을 반영하여 화음을 인식하는 과정에서 점진적인 인식을 가능하게 합니다.

- **Performance Highlights**: BACHI는 고전 및 팝 음악 벤치마크에서 최첨단 화음 인식 성능을 달성했으며, 각 모듈의 효과성을 검증하는 대상으로 실험을 진행했습니다. 이러한 성과는 상징적 ACR 분야에서의 데이터 부족 및 방법론적 도전 과제를 해결하는 중요한 기초 자료를 제공합니다.



### Online Matching via Reinforcement Learning: An Expert Policy Orchestration Strategy (https://arxiv.org/abs/2510.06515)
- **What's New**: 본 논문에서는 온라인 매칭 문제를 해결하기 위해 강화 학습(Reinforcement Learning) 기반의 새로운 접근 방식을 제안합니다. 이는 전통적인 전문가 정책(expert policies)을 데이터 기반으로 orchestrate(조정)하여 더 나은 의사결정을 가능하게 합니다. 특히, Adv2 프레임워크를 활용하여 정책 선택을 적대적 집합 문제로 변환하고 성능 보증을 제공하는 방법에 주목했습니다.

- **Technical Details**: 제안하는 접근 방식은 상태 및 행동 공간에서의 Markov 결정 과정(Markov Decision Process)을 기반으로 하며, 전문가 정책의 조합을 통해 학습합니다. 각 전문가의 장점을 고려하여 가중치를 업데이트하며, 확률적으로 안정적인 기대값 및 비편향(bias) 바운드를 도출합니다. 이로 인해 비정적 상태 및 동적 환경에서도 신뢰할 수 있는 정책 조합을 가능하게 합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 정책은 기존의 개별 전문가 정책 및 전통적인 강화 학습 RL 기준선보다 더 빠르게 수렴하고 시스템 효율성을 높이는 것으로 나타났습니다. 이는 복잡한 자원 배분 및 의사 결정 과정의 모델링과 관리에서 구조적이고 적응적인 학습이 중요한 역할을 함을 강조합니다. 특히, 장기적으로 운영되는 장기 이식 프로그램에서의 필요성을 잘 보여줍니다.



### Road Surface Condition Detection with Machine Learning using New York State Department of Transportation Camera Images and Weather Forecast Data (https://arxiv.org/abs/2510.06440)
- **What's New**: 이번 연구에서는 뉴욕주 교통부(NYSDOT)의 도로 상태를 평가하기 위해 기계 학습 모델을 활용하는 새로운 접근 방식을 제시합니다. 기계 학습 모델은 도로 표면 상태를 자동으로 분류하여 겨울철 폭풍과 같은 기상 상황에서의 운영 결정을 지원합니다. 특별히, 이 연구에서는 약 22,000개의 카메라 이미지를 활용하여 도로 조건을 심층 신경망(convolutional neural networks)과 랜덤 포레스트(random forests)로 예측합니다.

- **Technical Details**: 연구는 22,000개의 수동으로 라벨링된 이미지 데이터셋을 활용하여 도로 조건을 여섯 가지 상태로 분류하는 모델을 훈련시킵니다. 훈련된 모델은 NYSDOT의 운영 필요를 충족시키기 위해 일반화 가능성을 중시하며, 완전히 새로운 카메라에 대해 81.5%의 정확성을 달성했습니다. 이를 통해 기계 학습을 통해 도로 상태의 빠르고 정밀한 분석이 가능해집니다.

- **Performance Highlights**: 모델은 이미지와 기상 데이터를 통합하여 도로 표면 상태를 분류하는 데 성공적이었습니다. 특히, 완전히 새로운 카메라에서 81.5%의 정확도를 기록하였고, 이는 기상 관련 클래스에 대한 효과적인 예측을 가능하게 합니다. 연구는 최종 사용자와 협업을 통해 시행되었으며, 운영적 적용 가능성을 위해 모델의 일반화 가능성도 우선시 되었습니다.



### A General Constructive Upper Bound on Shallow Neural Nets Complexity (https://arxiv.org/abs/2510.06372)
- **What's New**: 이 논문은 compact set에서 continuous function을 근사하는 데 필요한 shallow neural network의 신경망 수에 대한 상한을 제공합니다. 이는 이전 시도의 한계를 넘어서는 보다 일반적인 접근법으로, 특정한 Stone-Weierstrass theorem의 증명에서 영감을 받았습니다.

- **Technical Details**: 제안된 방법은 연속 함수(continuous function)의 근사를 위해 신경망(neural network)의 구조와 개수를 조절하는 것이 가능하며, 이는 계산 이론(computational theory)에서 중요한 의미를 갖습니다. 기존의 상한(bounding) 방법에 비해 더욱 포괄적인 적용이 가능합니다.

- **Performance Highlights**: 이 논문에서 제시된 방법은 특정한 정확도(accuracy)를 보장하면서 다양한 연속 함수에 대한 근사를 가능하게 하여, 효율적인 신경망 설계를 위한 기초 자료를 제공합니다.



### Diffusion-Guided Renormalization of Neural Systems via Tensor Networks (https://arxiv.org/abs/2510.06361)
Comments:
          Reformatted version of Dissertation submitted for the Doctor of Philosophy in Systems and Control Engineering at Case Western Reserve University, 2025

- **What's New**: 이 논문은 물리학과 인공지능의 교차점에서 등장하는 새로운 컴퓨팅 패러다임을 활용하여 복잡한 대규모 시스템을 모델링하는 혁신적인 접근 방식을 제안합니다. 최근 자가 지도 학습(self-supervised learning) 및 분리된 표현 학습(disentangled representation learning) 기법의 결합을 통해, 상태 변화가 드러나는 잠재적 동적 임베딩(latent dynamical embeddings)을 발견하는 방법을 탐구하고 있습니다. 이러한 접근은 복잡한 신경 시스템을 이해하는 데 있어 이론적 기초를 제시합니다.

- **Technical Details**: 이 연구에서 제안하는 확산 기반 리노말리제이션(diffusion-based renormalization) 기법은 신경 시스템의 상태 구조를 여러 스케일에서 추론하는 것을 목표로 하고 있습니다. 구체적으로, Latent Graph Diffusion (LGD) 알고리즘을 개발하여 신경 데이터의 시계열 노드 특징을 변환하고, 스펙트럼 확산 모드를 추출하며, 계층적 압축 과정을 수행합니다. 이러한 알고리즘은 고차 메타그래프 공간에서 다중 스케일 리노말리제이션 그룹 흐름을 생성하여 신경 집합체의 동적 특성을 효율적으로 모델링할 수 있도록 합니다.

- **Performance Highlights**: 논문에서 제안하는 방법은 비평형 신경 시스템의 효과적인 동역학을 포착하는 예측 모델을 생성하는 데 특히 유용합니다. 이는 대규모 신경 녹음에서 잠재적 동적 구조를 분리하는 데 도움을 주어 행동, 인지 및 학습의 기저가 되는 신경 메커니즘에 대한 보다 깊은 통찰을 제공합니다. 최종적으로, 이러한 모델은 신경 데이터의 복잡성과 차원을 체계적으로 감소시키며, 신경 정보 처리 과정을 이해하는 데 기여합니다.



### TransFIRA: Transfer Learning for Face Image Recognizability Assessmen (https://arxiv.org/abs/2510.06353)
Comments:
          Project Page: this https URL

- **What's New**: TransFIRA(Transfer Learning for Face Image Recognizability Assessment)는 신뢰할 수 있는 얼굴 인식 성능을 위해 새로운 접근 방식을 제공합니다. 기존의 FIQA( 얼굴 이미지 품질 평가) 접근법들이 휴먼 주석이나 복잡한 컴퓨팅 파이프라인에 의존하는 것과 달리, TransFIRA는 임베딩 공간에서 직접적으로 인식 가능성을 정의합니다. 이 프레임워크는 근본적으로 결정 경계와 정렬된 인식 가능성 기준을 제공하여 품질 평가의 정확성을 향상시키고, 일반적인 얼굴 인식 및 신체 인식에 모두 적용 가능합니다.

- **Technical Details**: TransFIRA는 세 가지 주요 발전을 포함합니다: (i) 클래스 중심 유사성(class-center similarity, CCS)과 클래스 중심 각도 분리(class-center angular separation, CCAS)를 통한 인식 가능성 정의, (ii) 외부 레이블이 없는 매우 정확한 검증 성능을 위한 인식 가능성 정보를 사용하는 집계 전략, (iii) 인식 가능성의 맥락에서 신체 인식을 평가하기 위한 새로운 확장을 제공합니다. 이 프레임워크는 모든 사전 훈련된 인코더와 호환되며, 별도의 기계 학습 과정 없이 인식 가능성을 예측할 수 있도록 설계되었습니다.

- **Performance Highlights**: TransFIRA는 BRIAR와 IJB-C 데이터세트에서 최신 FIQA 방법보다 뛰어난 성능을 기록했습니다. 실험은 얼굴 인식에 대한 최첨단 결과를 확인하였을 뿐만 아니라 신체 인식에 대한 강력한 성능도 입증했습니다. 또한, 모델은 다양한 데이터셋 간에 강건성을 유지하면서 인식 가능성 예측의 투명성을 제공하며, 인식 성능 저하의 원인인 흐림이나 가림 효과를 설명할 수 있는 능력을 갖추고 있습니다.



### Asking For It: Question-Answering for Predicting Rule Infractions in Online Content Moderation (https://arxiv.org/abs/2510.06350)
Comments:
          Accepted at ICWSM 2026

- **What's New**: 이 논문에서는 온라인 커뮤니티의 규칙과 이행 간의 관계를 모델링하고, 새로운 질문-응답 프레임워크인 ModQ를 도입합니다. ModQ는 기존의 분류(classification) 또는 생성(generation) 기반 접근법과는 달리, 커뮤니티의 모든 규칙을 고려하여 특정 댓글에 가장 잘 적용되는 규칙을 식별합니다. 이는 커뮤니티별 규칙의 변동성과 이행의 일관성 문제를 해결하는 데 기여할 수 있습니다.

- **Technical Details**: ModQ는 두 가지 모델 변형을 구현하여 커뮤니티 규칙 이행을 정보 추출 정보(extraction) 작업으로 모델링합니다. 첫 번째 모델인 ModQ-Extract는 사용자 댓글과 커뮤니티 규칙을 컨텍스트로 사용하여 특정 규칙을 추출합니다. 두 번째 모델인 ModQ-Select는 다중 선택(multiple-choice) 질문-응답 방식으로 댓글과 각 규칙 간의 정합성을 점수화하여 가장 적합한 규칙을 선택합니다.

- **Performance Highlights**: ModQ를 사용한 두 모델 모두 Reddit과 Lemmy 데이터셋에서 최신 기법을 능가하여 규칙 위반을 식별하는 데 강력한 성능을 보였습니다. 특히 ModQ-Select는 모든 기준 및 moderation 작업에서 모든 베이스라인을 일관되게 초과하며, 두 모델 모두 미리 경험하지 못한 새로운 커뮤니티와 규칙에 대해 효과적인 일반화 능력을 보여줍니다. 이는 빠르게 변화하는 플랫폼에서의 운영에 있어 큰 장점이 됩니다.



### Conditional Denoising Diffusion Model-Based Robust MR Image Reconstruction from Highly Undersampled Data (https://arxiv.org/abs/2510.06335)
- **What's New**: 이 연구에서는 MRI 이미지를 고품질로 재구성하기 위해 새로운 조건부 제거 확산(conditional denoising diffusion) 프레임워크를 제안합니다. 기존의 방법들과는 달리, 이 프레임워크는 모든 역 확산(reverse diffusion) 단계에서 측정 모델을 직접 통합하고, 쌍으로 구성된 데이터로 학습하도록 설계되었습니다. 이러한 하이브리드 설계는 생성적 유연성과 MRI 물리학의 명시적인 적용을 결합하여 재구성을 개선합니다.

- **Technical Details**: 제안된 프레임워크는 역 샘플링 과정 중에 데이터 충실도(data fidelity) 항을 도입함으로써 재구성된 이미지와 원래 MRI 데이터 간의 일관성을 보장합니다. 이 접근 방식은 기존의 확산 모델들이 데이터 일관성을 별도의 후처리 단계로 적용하던 것과는 달리, 매개변수 조정(optimization) 단계를 통해 데이터 일관성을 유지하면서 제거 과정을 수행합니다. 이를 통해 고해상도 MRI 이미지를 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과 fastMRI 데이터셋에서 제안된 방법은 SSIM, PSNR, LPIPS 측면에서 기존의 최신 딥 러닝 및 확산 기반 방법들보다 일관되게 우수한 성능을 보였습니다. 특히 LPIPS는 시각적 개선 사항을 보다 정확하게 캡처하여, 픽셀 수준의 충실도(pixel-level fidelity)와 시각적 현실감(perceptual realism)의 개선을 나타냅니다. 이를 통해 제안된 방법이 MRI 재구성의 강건함과 신뢰성을 크게 향상시킨 것을 확인할 수 있습니다.



### Scalable deep fusion of spaceborne lidar and synthetic aperture radar for global forest structural complexity mapping (https://arxiv.org/abs/2510.06299)
- **What's New**: 이번 논문에서는 Global Ecosystem Dynamics Investigation (GEDI)에서 제공하는 공간 기반 lidar 데이터를 활용하여, 열대 및 온대 숲의 구조적 복잡성을 고해상도(25 m)로 매핑하는 새로운 딥러닝 프레임워크를 제시합니다. 이 연구의 핵심은 SAR(Synthetic Aperture Radar) 데이터와 GEDI 관측치를 융합하여 연속적이고 높은 해상도의 숲 구조 복잡성 지도를 생성하는 것입니다. 이렇게 개발된 모델은 연구자들이 특정한 컴퓨팅 인프라 없이도 대규모 데이터셋을 처리할 수 있는 접근 가능한 도구로 자리잡을 수 있습니다.

- **Technical Details**: 이 논문은 EfficientNetV2 아키텍처를 기반으로 하여, 1억 3천만 개 이상의 GEDI 발자국에 대해 훈련되었습니다. 이 모델은 40만 개 미만의 파라미터로도 뛰어난 성능(global R2 = 0.82)을 보여줍니다. 또한 모델은 바이옴(biome)과 시간대에 걸쳐 고유한 공간 패턴을 유지하면서 정확한 예측과 보정된 불확실성 추정을 제공합니다.

- **Performance Highlights**: 2015년부터 2022년까지의 숲 구조 복잡성에 대한 글로벌 다중 시계열(multi-temporal) 데이터셋을 생성하는 데 사용되었습니다. 전이 학습(transfer learning)을 통해 이 프레임워크는 추가적인 숲 구조 변수를 예측할 수 있게 확장될 수 있으며, 최소한의 컴퓨팅 비용으로 운영 가능합니다. 이러한 접근 방식은 변화하는 기후 속에서의 글로벌 숲 구조 동태를 지속적으로 모니터링하고 생물 다양성 보존 및 생태계 관리에 중요한 도구를 제공합니다.



### Efficient High-Resolution Image Editing with Hallucination-Aware Loss and Adaptive Tiling (https://arxiv.org/abs/2510.06295)
Comments:
          Preprint. Under review

- **What's New**: 최근 모바일 애플리케이션에서 고해상도(4K) 이미지 생성(photo-to-image synthesis)의 중요성이 증가하고 있습니다. 본 논문에서는 리소스 제약이 있는 디바이스에서 메모리와 이미지 품질 문제를 해결하는 새로운 시스템인 MobilePicasso를 제안합니다. 이 시스템은 세 단계에서 이미지 편집을 수행하여 효율성을 극대화하는 동시에 비용과 메모리 사용을 최소화합니다.

- **Technical Details**: MobilePicasso는 세 가지 주요 단계를 포함합니다: 표준 해상도에서 이미지 편집을 수행하는 hallucination-aware 손실, 픽셀 공간으로의 이동 문제를 해결하는 latent projection, 마지막으로 변환된 이미지를 고해상도로 확장하는 adaptive context-preserving tiling을 사용합니다. 이를 통해 전반적인 이미지 품질을 18-48% 향상시키고 환각(hallucination)을 14-51% 감소시키는 효과를 보여줍니다.

- **Performance Highlights**: MobilePicasso는 낮은 지연 시간(latency)과 높은 성능을 자랑하며, A100 GPU에서 실행되는 기존 서버 기반 모델보다 빠른 속도를 기록했습니다. 특히, 시스템의 런타임 메모리 사용량은 9% 증가에 그쳤으며, 55.8배까지 속도를 개선했습니다. 이러한 결과는 MobilePicasso가 실제 적용 가능성이 높은 모델임을 입증합니다.



### Soft-Evidence Fused Graph Neural Network for Cancer Driver Gene Identification across Multi-View Biological Graphs (https://arxiv.org/abs/2510.06290)
Comments:
          8pages

- **What's New**: 이번 연구에서는 Soft-Evidence Fusion Graph Neural Network (SEFGNN)을 제안하여 여러 생물학적 네트워크에서 암 유전자(CDG) 식별을 위한 새로운 접근 방식을 제공합니다. 기존의 GNN 기반 방법들과는 달리, SEFGNN은 각 네트워크를 독립적인 증거 원천으로 간주하고 의사 결정 수준에서 불확실성을 인식하여 융합(Fusion)합니다. 이는 Dempster-Shafer Theory (DST)를 이용하여 각 네트워크의 예측을 주관적 확률적 증거로 모델링하는 방식입니다.

- **Technical Details**: 연구는 NN 개의 생물학적 네트워크를 하나의 그래프로 모델링하고, 각 노드는 유전자, 각 엣지는 유전자 간의 상호작용을 나타냅니다. 최종 목표는 주어진 유전자가 암 유전자인지 여부를 이진 분류(binary classification)하는 것입니다. 모델 구조는 독립적으로 동작하는 GNN 세트, Dirichlet 분포에 따른 증거 변환, Dempster-Shafer 이론을 통한 증거 통합(multiple networks evidence integration)으로 구성됩니다.

- **Performance Highlights**: SEFGNN은 세 가지 암 데이터셋에서 광범위한 실험을 통해 기존 최첨단 기법들보다 일관성 있게 뛰어난 성능을 보였습니다. 이 모델은 새로운 암 유전자를 발견할 잠재력이 크며, 다양한 생물학적 네트워크에서의 유전자 관계를 효과적으로 모델링할 수 있는 능력을 demonstrated 합니다.



### BuilderBench -- A benchmark for generalist agents (https://arxiv.org/abs/2510.06288)
Comments:
          Project page: this https URL and Code: this https URL

- **What's New**: 이 논문은 BuilderBench라는 새로운 벤치마크를 소개하여 에이전트의 탐색 및 학습 능력을 키우는 연구를 가속화합니다. BuilderBench는 블록을 사용하여 다양한 구조를 만들도록 요구하며, 에이전트가 환경에 대한 일반적인 원칙을 스스로 발견하도록 학습합니다. 이는 기존 데이터의 한계를 넘어서는 문제를 해결할 수 있는 능력을 갖춘 에이전트를 개발하는 데 기여할 것입니다.

- **Technical Details**: BuilderBench는 로봇 손이 블록과 상호작용하는 빠른 시뮬레이터를 통해 구현됩니다. 이 환경에서는 motor skills (운동 능력), logical reasoning (논리적 추론), geometric reasoning (기하학적 추론) 및 intuitive physics (직관적 물리학) 등 다양한 기술이 요구됩니다. 에이전트는 개별 행동을 암기하는 것이 아니라, 건축의 일반적인 패턴을 배우고 이를 바탕으로 훈련 및 평가를 진행합니다.

- **Performance Highlights**: 이 논문에서는 BuilderBench의 오픈 소스 코드와 40개 이상의 다양한 과업을 제공하여 에이전트의 성능을 평가할 수 있도록 합니다. 이 시뮬레이터는 MuJoCo와 JAX를 기반으로 하여 CPU 기반 벤치마크보다 10배에서 100배 더 빠른 훈련 속도를 자랑합니다. 또한, 강화 학습( Reinforcement Learning, RL) 알고리즘을 쉽게 적용할 수 있는 단일 파일 구현도 제공하여 연구자들이 효율적으로 사용할 수 있도록 돕습니다.



### Mass Conservation on Rails -- Rethinking Physics-Informed Learning of Ice Flow Vector Fields (https://arxiv.org/abs/2510.06286)
Comments:
          Accepted at the Tackling Climate Change with Machine Learning Workshop at NeurIPS 2025. 9 pages, 4 figures

- **What's New**: 이 연구는 다이버전스가 없는 신경망(dfNNs)을 통해 국소적 질량 보존을 정확하게 강제하는 새로운 방법을 제안합니다. 반면에 기존의 물리 정보 신경망(PINNs)은 물리 법칙을 부드럽게 처벌하는 방식으로 통합하여 물리적 일관성을 보장하지 못하는데, 이는 데이터가 희소하고 잡음이 있는 상황에서는 일반화 성능을 저해할 수 있습니다. 이 연구는 dfNNs가 ICE 유동 벡터 필드와 같은 복잡한 물리적 문제를 해결하는 데 있어 보다 효과적임을 입증합니다.

- **Technical Details**: dfNNs는 2D 벡터 필드를 학습하는 데 사용되며, 그 기본 원칙은 벡터 필드의 다이버전스를 부호에 맞게 유지하는 것입니다. 다이버전스는 벡터 필드 구성 요소의 부분 미분 합을 통해 정의되어, 이를 통해 흐름의 확장 또는 압축의 국소적 비율을 측정합니다. 이 모델은 예측 정확성을 높이고 물리적 원칙을 완화하기 위해, 기계 학습 프레임워크인 PyTorch를 통해 구현됩니다.

- **Performance Highlights**: 모델 비교 결과, dfNNs는 Byrd Glacier의 ICE 유동 interpolations에서 더 신뢰할 수 있는 추정을 제공함을 보여주었습니다. 방향 지침(directional guidance)이라는 학습 전략을 활용해 위성 데이터를 기반으로 ICE 흐름을 조정할 시, 모든 모델에서 향상된 성능을 보였습니다. 이를 통해 물리적 법칙을 준수하며, ICE 유동에 대한 보다 정확한 예측이 가능하다는 것을 확인했습니다.



### General and Efficient Visual Goal-Conditioned Reinforcement Learning using Object-Agnostic Masks (https://arxiv.org/abs/2510.06277)
- **What's New**: 이번 연구는 객체에 대한 의존이 없는 mask 기반 목표 표현 시스템을 제안하여, 에이전트가 효율적인 학습과 뛰어난 일반화 성능을 발휘할 수 있도록 지원합니다. 기존의 목표 표현 방법들이 겪었던 일반화 문제와 느린 수렴 문제를 극복하면서, 시뮬레이션 학습에서 99.9%의 도달 정확도를 기록하였습니다. 제안된 방법은 실제 로봇에서의 픽업 작업 수행 시에도 높은 정확도를 유지하며, 포지셔널 정보를 요구하지 않습니다.

- **Technical Details**: Goal Conditioned Reinforcement Learning (GCRL)은 여러 목표를 학습하는 것을 가능하게 하지만, 목표 표현의 선택이 성공 여부에 큰 영향을 미칩니다. 우리는 mask를 기반으로 하는 목표 표현 방법을 도입하여, 에이전트가 목표에 대한 시각적 단서를 제공받고, 이로 인해 알려지지 않은 목표에 대해서도 일반화할 수 있는 능력을 갖추게 됩니다. 또한, mask 크기를 사용하여 밀집 보상 신호를 효과적으로 생성함으로써 목표와의 거리 계산을 단순화합니다.

- **Performance Highlights**: 우리의 연구 결과는 mask 기반 목표 표현이 기존 방법들과 비교하여 시각적 도달 작업에서 유사하거나 더욱 우수한 성능을 보임을 보여줍니다. 두 개의 물리적 로봇을 통해 시뮬레이션에서 실제로의 전이 학습을 검증하였으며, pretrained open vocabulary 객체 탐지 모델을 활용하여 mask 생성을 수행했습니다. 이러한 방법들은 빠른 수렴성과 강력한 일반화 기능을 보장하며 실세계 작업에서도 적용 가능성을 높입니다.



### Bridging Reasoning to Learning: Unmasking Illusions using Complexity Out of Distribution Generalization (https://arxiv.org/abs/2510.06274)
- **What's New**: 이번 연구에서는 복잡성 아웃 오브 배급(Complexity Out-of-Distribution, Complexity OoD) 일반화 개념을 도입하여 AI 모델의 추론 능력을 정의하고 측정하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 모델이 훈련 데이터보다 더 복잡한 테스트 샘플에서 성능을 유지할 수 있는지를 평가합니다. 기존의 일반화 지표가 System-1(신속하고 직관적인) 처리에 초점을 맞춘 것과 달리, Complexity OoD는 System-2(느리고 심사숙고하는) 추론 능력을 평가하는데 있어 새로운 기준을 제공합니다.

- **Technical Details**: Complexity OoD의 핵심은 특정 문제 인스턴스에 대해 요구되는 표현 능력이나 필수 해결 단계 수와 같은 '복잡성'을 정의하는 것입니다. 이를 위해 Kolmogorov 복잡성(kolmogorov complexity)이라는 개념을 활용하여 복잡성을 형식화합니다. 이러한 접근 방식은 기존의 길이 및 구성적인 OoD와의 차이를 명확히 하여, 학습과 추론 간의 관계를 다시 생각하게 만듭니다.

- **Performance Highlights**: Complexity OoD를 통한 평가 접근법은 기존 데이터 오염에 대해 더 강건한 평가를 제공하며, 모델의 기본적인 능력을 보다 정밀하고 세분화된 방식으로 측정할 수 있습니다. 이는 AI 모델의 추론 능력을 평가하는 새로운 방향을 제시하며, System-2 처리의 성공적인 습득이 System-1과 같은 요소의 기저 학습에 의존함을 보여줍니다. 이를 통해 학습과 추론 간의 장기적으로 존재하는 개념적 간극을 메우는 데 기여할 수 있습니다.



### Vision Transformer for Transient Noise Classification (https://arxiv.org/abs/2510.06273)
Comments:
          9 pages, 4 figures

- **What's New**: 이 연구는 LIGO 데이터에서의 일시적 노이즈(글리치)를 22개의 기존 클래스와 O3a에서 추가된 2개의 노이즈 클래스로 분류하는 작업을 기반으로 합니다. 저자들은 Vision Transformer (ViT) 모델을 활용하여 새로운 분류 모델을 훈련하고 이 모델의 효과성을 입증했습니다. 특히, O3 기간 동안 수집된 새로운 데이터 클래스를 포함하여 LIGO 데이터를 처리하는 데 ViT 모델의 잠재력을 강조하였습니다.

- **Technical Details**: 본 연구에서는 Gravity Spy 프로젝트에서 수집된 데이터셋을 사용하였으며, ViT-B/32 모델을 통해 22개 클래스와 O3a에서 추가된 두 클래스에 대한 분류 작업을 수행했습니다. 데이터셋은 7:1.5:1.5 비율로 훈련, 검증 및 테스트 세트로 분할되었고, Adam 옵티마이저를 사용하여 모델 파라미터를 업데이트하였습니다. 훈련 과정은 15 epoch 동안 진행되며, 교차 엔트로피 손실 함수가 모델의 성능을 평가하는 데 사용되었습니다.

- **Performance Highlights**: 시험 데이터셋에 대한 ViT-B/32 모델의 F1 점수는 92.13%, 정확도는 92.26%로 평가되었습니다. 몇몇 클래스, 예를 들어 1080Lines 및 Helix는 98% 이상의 정확도를 기록한 반면, Paired_Doves 및 No_Glitch 클래스는 9.09% 및 26.56%로 저조한 성적을 보였습니다. 과거 CNN 모델들과 비교할 때 ViT는 향상된 의미적 탐지를 가능하게 하지만, 데이터 세트 크기에 따른 성능 저하의 여지를 보여주었습니다.



### A Mixed-Methods Analysis of Repression and Mobilization in Bangladesh's July Revolution Using Machine Learning and Statistical Modeling (https://arxiv.org/abs/2510.06264)
Comments:
          Submitted to Social Forces. Final version may vary from this preprint

- **What's New**: 2024년 방글라데시의 7월 혁명은 민간 저항의 연구에서 중요한 사건으로 부각되고 있습니다. 이 연구는 정부의 폭력이 최종적으로 학생 주도 비폭력 저항 운동의 승리를 어떻게 촉진했는지를 분석합니다. 혼합 방법론을 사용하여 갈등의 타임라인을 정리하고, 억압과 동원의 복잡한 관계를 분해하여 특정 가설을 도출합니다.

- **Technical Details**: 연구는 세 가지 주요 가설을 제시합니다: 첫째, 운동 자체의 모멘텀이 지속되었고(H1), 둘째, 국가 억압이 오히려 반작용(backfire) 효과를 가져왔으며(H2), 셋째, 이 반작용은 비선형적이고 특정 도화선이 필요하다는 것입니다(H3). 논문의 분석은 패널 모델과 벡터 자기회귀(Vector Autoregression) 분석을 사용하여 입증되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 경찰의 살인적 폭력 증가에 대한 응답으로 전국적으로 즉각적인 동원이 발생하였습니다. 기계 학습 분석(XGBoost)은 '시위자에 대한 과도한 폭력'이 전국적인 확산의 가장 중요한 예측 요소로 확인되었습니다. 이러한 결과는 무장 국가에 대해 비폭력적으로 성공한 운동의 역설적인 사례로, 저항 운동에 대한 중요한 교훈을 제공합니다.



### Prakriti200: A Questionnaire-Based Dataset of 200 Ayurvedic Prakriti Assessments (https://arxiv.org/abs/2510.06262)
Comments:
          4 pages, 4 figures

- **What's New**: 이번 연구에서는 전통 아유르베다 원리에 따라 개인의 신체적, 생리적, 심리적 특성을 평가하기 위한 이중 언어(영어-힌디어) Prakriti Assessment Questionnaire를 활용한 새로운 데이터 세트를 제공했습니다. 이 설문지는 24개의 다중 선택 항목으로 구성되어 있으며, 체형, 식욕, 수면 패턴, 에너지 수준 및 기질을 포함하여 다양한 특성을 평가합니다. 이 데이터는 자동 채점 시스템을 통해 개별 특성을 dosha(바타, 피타, 카파) 점수와 매핑하여 신뢰할 수 있는 분석을 가능하게 합니다.

- **Technical Details**: 이 데이터 수집 방법은 아유르베다 원칙에 근거한 이중 언어 설문지를 사용합니다. 24개의 다중 선택 질문은 신체적 특성(예: 체형, 신장), 생리적 특성(예: 식욕, 수면), 심리적 특성(예: 기질)에 대한 정보를 포함하고 있습니다. Google Forms를 통해 수집된 데이터는 완전성과 일관성을 검토하여 구조화된 xlsx 파일 형태로 저장되며, 각 참여자의 점수를 포함합니다.

- **Performance Highlights**: 최종 데이터 세트에는 200명의 참여자가 포함되어 있으며, 67.5%가 평균 체중, 62%가 평균 신장을 보고했습니다. 데이터 분석 결과 Pitta가 우세한 구성(97명)이 많았고, 뒤이어 혼합형과 같은 다른 유형의 구성이 나타났습니다. 이 데이터 세트는 향후 아유르베다 기반 연구와 헬스케어 어플리케이션 개발에 중요한 자료로 활용될 수 있습니다.



### AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning (https://arxiv.org/abs/2510.06261)
Comments:
          Ongoing project

- **What's New**: AlphaApollo는 자체 진화하는 에이전틱 추론 시스템으로, 기존의 재단 모델(FM) 추론에서 발생하는 두 가지 주요 제약인 모델 고유의 능력과 신뢰할 수 없는 테스트 시간 반복 문제를 해결하고자 합니다. 이 시스템은 계산 도구(Python)와 검색 도구(작업 관련 외부 정보)를 결합하여 정확한 계산을 수행하고 결정을 기반으로 합니다. 이러한 통합으로 AlphaApollo는 복잡한 문제를 해결하는 것뿐만 아니라, 다양한 모델과 도구를 협동으로 조율합니다.

- **Technical Details**: AlphaApollo의 설계 원리는 다양한 모델과 전문 도구를 조화롭게 결합하여 자가 진화형 시스템을 구현하는 것입니다. 이를 통해 직관적이고 정의된 추론을 가능하게 하며, 수학적 문제 해결 시 Python 코드의 실행과 검증에 기반한 피드백을 제공합니다. 이 시스템은 모델 간의 상호작용과도 결합하여 도구-확장된 추론을 진행하며, 이를 통해 근본적인 한계를 넘어서는 데 기여합니다.

- **Performance Highlights**: AlphaApollo는 다양한 모델에서 일정한 성능 개선을 보여줬으며, Qwen2.5 모델에서는 평균 5.15% 증가, Llama-3.3-70B-Instruct 모델에서는 8.91%의 평균 성능 향상이 있었습니다. 또한, 도구 사용 분석 결과 80% 이상의 도구 호출이 정확하게 수행되어 비도구 기반 응답을 일관되게 초과하는 성과를 보였습니다. 현재 AlphaApollo는 지속적인 개발 중이며, 향후 추가 기능 및 실험 결과가 오픈 소스로 공개될 예정입니다.



### Ensemble Deep Learning and LLM-Assisted Reporting for Automated Skin Lesion Diagnosis (https://arxiv.org/abs/2510.06260)
- **What's New**: 이 논문은 피부암 진단을 위한 새로운 통합 AI 프레임워크를 제안합니다. 이 시스템은 다양한 아키텍처를 가진 합성곱 신경망(CNN) 앙상블과 대형 언어 모델(LLM)을 결합하여 진단의 신뢰성과 접근성을 향상시킵니다. 또한, 이는 환자 교육을 포함한 임상 문서화의 필수 요구사항을 충족하면서 진단 출력이 임상적으로 의미 있는 평가로 변환되는 것을 가능하게 합니다.

- **Technical Details**: 제안된 시스템은 EfficientNetB3, ResNet50, DenseNet121의 이종 앙상블을 사용하여 복합적인 진단 관점을 제공합니다. 또한, 자동화된 보고서 생성을 위한 LLM 기반 시스템을 포함하여, 진단 추론 및 임상 출력을 동시에 수행할 수 있도록 설계되었습니다. 이 시스템은 불확실성 메커니즘을 내재하여 전문가 검토를 위한 비일관한 사례를 플래그하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 프레임워크는 진단 정확성을 높이는 동시에 환자가 진단 내용을 이해하고 조기 징후를 인식할 수 있게 지원합니다. 이를 통해 초기 발견률이 향상되고, 환자에게 개인화된 모니터링 지침을 제공하여 치료의 연속성을 지원합니다. 종합하여, 이 시스템은 인공지능(AI) 기반 피부 진단을 실제 임상에서 사용할 수 있는 솔루션으로 발전시키는 중요한 진전을 나타냅니다.



### Beyond Static Knowledge Messengers: Towards Adaptive, Fair, and Scalable Federated Learning for Medical AI (https://arxiv.org/abs/2510.06259)
Comments:
          20 pages, 4 figures, 14 tables. Proposes Adaptive Fair Federated Learning (AFFL) algorithm and MedFedBench benchmark suite for healthcare federated learning

- **What's New**: 이 논문은 의료 AI의 개인 정보 보호와 공정성을 보장하면서 협력 학습의 과제를 해결하는 새로운 접근 방식을 제안합니다. Adaptive Fair Federated Learning (AFFL)을 통해 이론적 분석과 함께 세 가지 혁신인 동적 메시지 전송, 공정성 인식 증류, 커리큘럼 가이드 가속화를 도입하여 기존의 저조한 성과와 느린 수렴 문제를 극복하고자 합니다.

- **Technical Details**: AFFL은 데이터의 이질성과 작업 복잡성에 따라 동적으로 용량을 조정하는 Adaptive Knowledge Messengers를 채택합니다. 또한, 공정성을 높이기 위해 영향 가중 집계를 사용하는 Fairness-Aware Distillation과 통신 라운드를 60-70% 줄이는 Curriculum-Guided Acceleration을 도입하여 효율적인 연산을 수행합니다. 이 연구의 이론적 분석은 O(T^{-1/2}) + O(H_max/T^{3/4})라는 수렴 보장을 제공합니다.

- **Performance Highlights**: 연구 결과는 통신 감소를 55-75%, 공정성 향상을 56-68%, 에너지 절약을 34-46% 달성할 것으로 예상합니다. 또한 100개 이상의 의료 기관을 지원하며, 이미지, 유전자 분석, 전자 건강 기록(EHR), 센서 데이터 등 다양한 데이터와 통합이 가능하도록 설계된 프레임워크입니다. 경제적 관점에서는 농촌 병원에 대해 400-800%의 ROI를, 학술 센터에 대해서는 15-25%의 성능 향상을 나타낼 것으로 예측됩니다.



### Developing a Sequential Deep Learning Pipeline to Model Alaskan Permafrost Thaw Under Climate Chang (https://arxiv.org/abs/2510.06258)
Comments:
          20 pages, 16 figures. Number of figures are tentative and will be reduced in the future

- **What's New**: 이 연구는 알래스카의 지하 얼음층의 온도 변화를 모델링하기 위한 새로운 딥러닝 파이프라인을 제안합니다. 이 파이프라인은 동적인 재분석(feature) 데이터 및 여러 지질적 특성을 활용하여 다양한 깊이의 연간 토양 온도를 예측합니다. 또한, 계절적 맥락을 제공하기 위한 슬라이딩 윈도우(sequence) 방식과 장기 기후 강제를 위한 유도된 시나리오 신호 특성을 포함하여 더욱 정교한 예측을 가능하게 합니다.

- **Technical Details**: 연구에서는 Temporal Convolutional Network (TCN), Transformer, 1-Dimensional Convolutional Long-Short Term Memory (Conv1DLSTM), Gated-Recurrent Unit (GRU), Bidirectional Long-Short Term Memory (BiLSTM) 등 다섯 가지 딥러닝 모델을 테스트했습니다. 이 모델들은 LATITUDE(위도) 및 깊이에 따른 온도 차이를 효과적으로 인식하며, GRU 모델이 순차적인 온도 패턴 탐지에서 가장 뛰어난 성능을 보였습니다. bias-corrected CMIP5 RCP 데이터는 온도 추세를 인식할 수 있게 도왔지만, 시나리오 간의 제한된 분산만 확인되었습니다.

- **Performance Highlights**: 이 연구는 지하 얼음층의 온도 모델링에 딥러닝 기술을 효과적으로 적용하는 종합적인 프레임워크를 구축하였습니다. 동계, 공간, 수직 온도 맥락을 제공함으로써 특성 선택에 대한 본질적인 제약 없이도 모델을 구성할 수 있는 가능성을 여는 것입니다. 결과적으로, 이 연구는 토양 온도 예측의 정확성을 높이는 중요한 기준점을 제시하며, 기후 변화의 영향 최소화에 기여할 수 있습니다.



### Toward Uncertainty-Aware and Generalizable Neural Decoding for Quantum LDPC Codes (https://arxiv.org/abs/2510.06257)
- **What's New**: 이번 연구에서는 QuBA라는 Bayesian graph neural decoder를 제안하여, 전통적인 디코딩 방식의 한계를 극복하고, 불확실성을 계량화하며 새로운 코드에 대한 일반화 능력을 강화하고자 합니다. 이를 기반으로 SAGU라는 다중 코드 훈련 프레임워크를 개발하여 훈련 세트 외부에서도 디코딩이 가능하도록 했습니다. 실험 결과, QuBA와 SAGU는 기존의 belief propagation(BP) 방식보다 평균적으로 논리 오류율(logical error rate, LER)을 한 자리수 감소시키며, 두 자리수까지 개선할 수 있음을 보였습니다.

- **Technical Details**: Quantum error correction(QEC)의 필수 구성 요소인 QuBA는 Bayesian neural networks(BNNs)를 활용하여 예측 불확실성을 나타내고, Monte Carlo dropout 방식을 통해 계량화된 신뢰도 추정치를 제공합니다. 또한, error syndrome 간의 상관관계를 잘 포착하기 위해 graph neural network(GNN) 아키텍처 내에서 dot-product 및 multi-head attention을 통합했습니다. SAGU는 Diverse-Aggregate-Repeat Training(DART) 패러다임에서 영감을 받아 이질적인 양자 코드 간의 일반화 능력을 강화하기 위해 설계된 3단계의 교차 도메인 훈련 프레임워크입니다.

- **Performance Highlights**: 실험적으로 QuBA는 coprime BB 코드 [[154,6,16]]에 대해 BP 대비 거의 두 자리수의 향상을 보여주었으며, 최신 neural decoder인 Astra와 비교하여도 표준 BB 코드에서 한 자리수의 우위를 유지하고 있습니다. 또한, SAGU는 QuBA의 도메인 특화 훈련 방식과 비교하여 동등하거나 더 나은 디코딩 성능을 나타내는 것으로 확인되었습니다. 이로 인해 QuBA와 SAGU는 높은 디코딩 정확도와 불확실성 인식을 동시에 제공할 수 있음을 입증했습니다.



### Dream2Image : An Open Multimodal EEG Dataset for Decoding and Visualizing Dreams with Artificial Intelligenc (https://arxiv.org/abs/2510.06252)
Comments:
          7 Pages, 3 Figures, The Dream2Image dataset is openly available on Hugging Face at: this https URL

- **What's New**: Dream2Image는 EEG 신호(EEG signals), 꿈 전사(dream transcriptions), 그리고 AI 생성 이미지(AI-generated images)를 결합한 세계 최초의 데이터셋입니다. 38명의 참가자와 31시간 이상의 꿈 EEG 녹음 기반으로, 총 129개의 샘플을 포함하고 있습니다. 이 데이터셋은 각성 직전의 뇌 활동 마지막 초(T-15, T-30, T-60, T-120)와 꿈의 경험에 대한 원시 보고서(raw reports)를 제공합니다.

- **Technical Details**: Dream2Image는 꿈 연구에 대한 새로운 자원이자, 꿈의 신경 상관관계(neural correlates of dreaming)를 연구하기 위한 독특한 자원입니다. 이 데이터셋은 뇌 활동(Brain activity)으로부터 꿈을 디코딩하는 모델을 개발하고, 신경과학(neuroscience), 심리학(psychology), 인공지능(artificial intelligence) 분야에서 새로운 접근 방식을 탐구하는 데 도움을 줍니다. 데이터셋은 Hugging Face와 GitHub에서 오픈 액세스(open access)로 제공됩니다.

- **Performance Highlights**: 이 데이터셋은 인공지능과 신경과학의 교차점에서 연구를 지원하기 위해 설계되었습니다. 현재의 뇌 활동 디코딩 방법을 확장하고 연구자들에게 영감을 주는 목적으로 만들어졌습니다. 하지만 샘플 크기가 상대적으로 작고 꿈 회상의 변동성(variability of dream recall)은 일반화 가능성에 영향을 미칠 수 있다는 한계가 있습니다.



### Evaluating Embedding Frameworks for Scientific Domain (https://arxiv.org/abs/2510.06244)
- **What's New**: 이 논문은 특정 도메인 데이터에 따라 같은 단어가 서로 다른 의미와 표현을 가질 수 있음을 강조하며, 과학 분야에 최적화된 단어 표현 알고리즘과 토큰화 방법을 연구하고 있습니다. 특히, 과학 기술 문헌에 적합한 새로운 평가 수트를 구축하여 다양한 단어 표현 및 토큰화 알고리즘의 성능을 평가하는 것을 목표로 하고 있습니다.

- **Technical Details**: 자연어 처리(NLP)에서 효과적인 단어 표현은 언어 이해, 텍스트 생성 및 감정 분석과 같은 작업에 중대한 영향을 미치며, 현재 사용되는 여러 토큰화 및 단어 임베딩 방법의 성능을 과학 분야에서 평가하는 데 중점을 두고 있습니다. Byte Pair Encoding(BPE), WordPiece, Unigram Tokenizer 등의 다양한 토큰화 방법과 Word2Vec, GloVe, FastText와 같은 단어 임베딩 기법이 소개되며, 각 방법은 특정 장단점을 가지고 있습니다.

- **Performance Highlights**: 연구는 다양한 다운스트림 NLP 과제를 포함한 포괄적인 평가 수트를 구축하여 과학 분야에서의 단어 표현과 토큰화 모델을 비교하고 있습니다. 특히, 낮은 리소스 모델인 Word2Vec과 계산적으로 더 많은 자원을 소모하는 Transformer 기반 모델의 성능 비교를 통해 과학 분야에서의 단어 표현 문제에 대한 통찰력을 제공하고 있습니다.



### Uncertainty Quantification In Surface Landmines and UXO Classification Using MC Dropou (https://arxiv.org/abs/2510.06238)
Comments:
          This work has been accepted and presented at IGARSS 2025 and will appear in the IEEE IGARSS 2025 proceedings

- **What's New**: 이번 연구는 인간 지뢰 제거 작업에서 표면 지뢰 및 비폭발 잔해(UXOs)의 탐지를 위한 Monte Carlo (MC) Dropout을 이용한 불확실성 정량화 개념을 도입합니다. ResNet-50 아키텍처에 통합하여 실험한 결과는 기존 신경망 모델의 취약성을 극복할 수 있는 가능성을 제시합니다. MC Dropout 접근 방식은 예측 신뢰성을 추가적인 지표로 제공하여 지뢰 제거 작전에서 더 신뢰할 수 있는 결정을 내리는 데 도움이 될 수 있습니다.

- **Technical Details**: 이 연구에서는 ResNet-50 모델을 이용하여 MC Dropout을 통합함으로써 예측 불확실성을 추정하는 방법을 소개합니다. 연구팀은 깨끗한 테스트 이미지, 적대적 변형 및 노이즈가 있는 테스트 이미지를 포함한 세 가지 시나리오에서 모델을 평가했습니다. MC Dropout은 모델의 가중치를 확률적 분포로 보고, 예측 불확실성을 효율적으로 추정할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 다수의 확률적 전방 패스를 통해 얻은 평균 예측 및 그 분산은 불확실성의 지표 역할을 했습니다. 예측의 분산이 높을수록 모델 예측의 불확실성이 증가하는 것을 확인했습니다. 이 연구는 표면 지뢰 및 UXOs 분류에서 불확실성 정량화의 필요성을 강조하며, 향후 현실 세계의 지뢰 제거 애플리케이션에 적용할 수 있는 기초를 마련합니다.



### Neu-RadBERT for Enhanced Diagnosis of Brain Injuries and Conditions (https://arxiv.org/abs/2510.06232)
Comments:
          Both Manpreet Singh and Sean Macrae contributed equally and should be considered co-first authors. Corresponding author: Yiorgos Alexandros Cavayas

- **What's New**: 이 연구는 침습적 기계환기를 받고 있는 급성 호흡 부전 환자들의 두뇌 영상의학 보고서에서 진단을 자동으로 추출하기 위한 Neu-RadBERT라는 분류 알고리즘을 개발했다. 기존의 RadBERT 모델을 기반으로 하여, 비구조화된 영상의학 보고서에 대한 분류 성능을 개선하기 위해 여러 방법론을 적용했다. 특히, 데이터 스큐니스(data skewness) 문제를 해결하기 위한 오버샘플링(oversampling) 전략이 눈에 띈다.

- **Technical Details**: Neu-RadBERT는 BERT 기반의 모델로, MIMIC-IV 데이터베이스에서 추출한 두뇌 영상 보고서를 이용하여 초기 수동 레이블링을 수행한 후 세 가지 전략으로 미세 조정(fine-tuning)을 진행했다. 이러한 전략은 1) 기본 RadBERT, 2) Masked Language Modeling (MLM) 사전 훈련이 포함된 Neu-RadBERT, 3) MLM 사전 훈련과 오버샘플링을 접목한 Neu-RadBERT였다. 이러한 접근은 모델의 분류 정확도를 현저히 향상시키는 데 기여했다.

- **Performance Highlights**: Neu-RadBERT는 오버샘플링을 적용했을 때 특히 두뇌 이상에 대한 진단 정확도를 98.0%까지 향상시켰다. 반면 Llama-2-13B 모델은 최고 67.5%의 이진 분류 정확도로, 특정한 분류 작업에 대한 자가 회귀 LLM의 제한성을 강조한다. 이 연구는 transformer 기반 NLP 모델이 자유 텍스트 보고서에서 진단을 자동으로 추출하는 데 잠재력이 있음을 보여준다.



### Milestone Determination for Autonomous Railway Operation (https://arxiv.org/abs/2510.06229)
Comments:
          Paper submitted and partially accepted to ICART 2025, paper is 8 pages and has 1 figure, 2 tables

- **What's New**: 이번 논문에서는 철도 자동화 분야에서의 컴퓨터 비전 시스템 개발에 대한 주요 도전에 대해 다루고 있습니다. 특히, 고품질의 연속적인 데이터의 제한된 가용성이 문제라는 점을 강조하고 있습니다. 기존의 데이터셋은 스페이셜-템포럴(spatio-temporal) 컨텍스트가 부족하여 실시간 의사 결정에 어려움을 초래합니다.

- **Technical Details**: 논문에서는 경로별(contextually relevant) 및 규칙 기반(rule-based) 모델을 통한 마일스톤 결정(milestone determination) 개념을 제안하고 있습니다. 이는 동적인 구성요소의 일반적인 인식을 요구하지 않고, 특정한 결정 지점에 집중하여 학습 과정을 단순화합니다. 이러한 방식은 예측 가능한 환경에서 비전 에이전트(vision agents) 훈련을 가능하게 합니다.

- **Performance Highlights**: 이 접근 방식은 철도 자동화에 대한 보다 안전하고 효율적인 머신 러닝 시스템을 촉진할 수 있는 실용적인 프레임워크를 제공합니다. 제안된 방법은 현실 세계의 운영 논리(reality operational logic)와 더 밀접하게 일치하는 풍부한 연속 데이터셋을 생성하여, 문제를 해결할 수 있는 기반이 됩니다.



### Layerwise Federated Learning for Heterogeneous Quantum Clients using Quorus (https://arxiv.org/abs/2510.06228)
- **What's New**: 이 논문에서는 다양한 깊이의 양자 회로를 효과적으로 훈련하기 위해 레이어별 손실 함수를 사용하는 Quorus라는 새로운 양자 연합 학습 (Quantum Federated Learning, QFL) 방법을 제안합니다. 이 접근법은 클라이언트들이 자신들의 용량에 따라 높은 정확도를 위한 모델을 선택할 수 있도록 하여, 기존의 기술적 한계를 극복할 수 있습니다. 또한 Quorus는 클라이언트의 요구에 맞춘 모델 설계를 제안하여 연산 자원 효율성을 높입니다.

- **Technical Details**: Quorus는 양자 회로의 깊이와 관련된 장비 오류를 고려한 첫 번째 구조화된 양자 연합 학습 프레임워크입니다. 특정 깊이에서 비교적 높은 정확도를 달성할 수 있도록 하여 QFL을 구현하는데 필요한 장비 간의 다양성을 관리합니다. 레이어별 손실 함수 및 지식 증류(knowledge distillation)를 활용하여 동기화된 목표를 달성하는 방법을 소개합니다.

- **Performance Highlights**: Quorus의 시뮬레이션 및 실제 하드웨어에서의 결과는 깊이가 더 깊은 클라이언트의 그래디언트 크기를 증가시키고, 최첨단 기술 대비 평균 12.4% 향상된 테스트 정확도를 보여주었습니다. 이러한 성능을 통해 Quorus는 실제 상황에서 실용적으로 활용 가능한 가능성을 제시합니다.



### Enhancing Resilience for IoE: A Perspective of Networking-Level Safeguard (https://arxiv.org/abs/2508.20504)
Comments:
          To be published in IEEE Network Magazine, 2026

- **What's New**: 이번 논문은 에너지 인터넷(Internet of Energy, IoE)과 관련하여, IoT 기반 디지털 통신과 전력망이 통합된 안전하고 효율적인 에너지 시스템을 실현하기 위한 연구를 진행하였습니다. 특히 기존의 전통적인 방어 수단을 초월하여, 적대적 공격(adversarial attacks)에 대응할 수 있는 복원력 있는 솔루션이 필요하다는 점을 강조했습니다. 이를 위해 그래프 구조 학습(Graph Structure Learning, GSL) 기반의 방어 프레임워크를 제안하며, 이 프레임워크가 IoE 네트워크의 안전성을 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 GSL을 사용해 그래프 토폴로지와 노드 표현을 동시에 최적화하여 IoE 네트워크의 적대적 조작을 방어하는 방법을 제안합니다. GSL 기반 방식은 데이터 흐름의 동적 변화에 적응할 수 있는 능력을 가지고 있으며, 이는 실제 IoE 네트워크의 복잡한 구조를 처리하는 데에 효과적입니다. 또한, 기존 방어 시스템들의 한계를 인정하고, GSL이 네트워크 보안 자기학습을 통해 어떻게 저항력을 높일 수 있는지에 대한 사례 연구도 포함되어 있습니다.

- **Performance Highlights**: 제안된 GSL 프레임워크는 다양한 기존 방법과 비교할 때 뛰어난 강건성을 보여주었습니다. 특히, 공격자가 네트워크 모델을 조작하려 시도할 때, GSL의 공동 최적화(co-optimization) 기능이 효과적으로 작동하여 IoE의 다양한 위협에 대한 저항력을 강화합니다. 마지막으로, GSL의 가능성을 활용하여 미래 IoE 네트워크의 안정성과 신뢰성을 더욱 향상시킬 수 있는 경로를 제시하며, 향후 연구 방향을 제안합니다.



### PolyKAN: A Polyhedral Analysis Framework for Provable and Approximately Optimal KAN Compression (https://arxiv.org/abs/2510.04205)
Comments:
          The description of the paper's contributions has been tightened up, and statements that may cause misunderstandings have been removed

- **What's New**: 본 논문에서는 Kolmogorov-Arnold Networks(KANs)의 압축을 위한 새로운 이론적 프레임워크인 PolyKAN을 제안합니다. PolyKAN은 모델 크기 감소와 근사 오차에 대한 공식적인 보장을 제공하여 KANs의 파라미터 효율성을 개선하는 데 기여합니다. 이 프레임워크는 KAN 압축 문제를 다각형 영역 병합 작업으로 공식을 정리하며, 이를 통해 인터프리터블(neural architecture) 신경망의 효율적인 배포를 위한 새로운 방향을 열어줍니다.

- **Technical Details**: PolyKAN은 KAN의 완전한 다각형 특성을 수립하고, ϵ-등가 압축에 대한 공식 이론을 개발했습니다. 이를 통해 특정 오차 범위를 유지하면서 영역 병합에 필요한 조건을 제공합니다. 또한, 다변량 스플라인 함수를 위한 최적의 압축을 보장하는 동적 프로그래밍 알고리즘을 설계하여 약간의 최적 보장도 제공합니다.

- **Performance Highlights**: PolyKAN은 가변 수 압축이 가능하면서도 엄격한 오류 제어를 유지한다는 이론적 분석 결과를 제공합니다. 특히, 단변량 스플라인 함수의 압축에서 보장된 전역 최적성을 통해 KAN 압축의 신뢰성을 한층 높였습니다. 이러한 특성은 KANs가 뚜렷한 축 정렬 구조를 갖고 있어, Polyhedral 분석에 적합하다는 점을 강조합니다.




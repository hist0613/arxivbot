### De-DSI: Decentralised Differentiable Search Index (https://arxiv.org/abs/2404.12237)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)과 진정한 탈중앙화를 결합한 새로운 프레임워크인 De-DSI를 소개합니다. De-DSI는 질의-문서 식별자 쌍만을 사용하여 효율적으로 새로운 사용자 질의와 문서 식별자를 연결하는 방식으로 작동합니다. 이는 중앙화된 방식과 비교하여 검색 성공률이 비슷하며 계산 복잡성을 네트워크 전체에 분산시킬 수 있는 이점을 제공합니다.

- **Technical Details**: De-DSI는 다양한 Differentiable Search Index(DSI) 모델을 앙상블하여 데이터셋을 작은 샤드로 분할하고 각각의 모델을 개별적으로 훈련시키는 방식을 도입했습니다. 이 구조는 정확도를 유지하면서 확장성을 촉진하고 여러 모델의 결과를 집계하는 데 도움을 줍니다. 또한, 최상위 문서 식별자(docids)를 찾기 위해 빔 탐색(beam search)을 사용하고, 점수 정규화를 위해 소프트맥스 함수(softmax function)를 적용합니다.

- **Performance Highlights**: 분산된 실행은 중앙 집중식 방법과 비교할 때 검색 성공률이 비슷하다는 것을 보여줍니다. 이 시스템은 멀티미디어 항목을 매그넷 링크(magnet links)를 통해 검색할 수 있게 함으로써 플랫폼이나 중개인 없이도 작동할 수 있습니다. 구현의 단순성, 확장성, 실행 가능성 및 배포 가능성이 이 아키텍처의 주요 설계 원칙입니다.



### How Do Recommendation Models Amplify Popularity Bias? An Analysis from  the Spectral Perspectiv (https://arxiv.org/abs/2404.12008)
Comments: 23 pages, 7 figures

- **What's New**: 이 연구에서는 추천 시스템(Recommendation Systems, RS)에서의 인기도 편향(popularity bias) 증폭 문제를 심층적으로 분석하였습니다. 특히, 아이템의 인기도가 추천 모델의 주요 단일 벡터(principal singular vector)에 기억되며, 차원 축소(dimension collapse) 현상이 이 편향을 더욱 강화시키는 것을 발견하였습니다. 이를 근거로, 주요 단일 값(principal singular value)의 크기에 패널티를 부과하여 편향을 완화하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 스펙트럴 노름 규제기(spectral norm regularizer)를 도입하여 주요 단일값의 크기를 직접 제한함으로써 인기도 편향을 완화합니다. 이 과정에서 단일 벡터의 내재된 성질을 활용하고 행렬 변환 기법을 적용하여 계산 부담을 크게 줄였습니다. 이론적으로는 추천 모델이 인기 아이템의 비율과 단일값 사이의 관계를 탐구하였으며, 이를 통해 인기도 편향의 근본 원인을 설명하는 데 기여하였습니다.

- **Performance Highlights**: 실험은 7개의 실제 데이터셋과 세 가지 테스트 시나리오에서 실시되었으며, 제안된 방법이 인기도 편향을 효과적으로 완화하는 것을 입증하였습니다. 특히, 아이템의 인기도와 주요 단일 벡터 사이의 코사인 유사도(cosine similarity)가 0.98 이상임을 확인하였으며, 이는 모델 예측에 인기도가 과도하게 영향을 미친다는 점을 시사합니다.



### Knowledge-Aware Multi-Intent Contrastive Learning for Multi-Behavior  Recommendation (https://arxiv.org/abs/2404.11993)
- **What's New**: 다양한 사용자 행동을 고려한 추천 시스템(KAMCL- Knowledge-Aware Multi-Intent Contrastive Learning)이 소개되었습니다. 기존의 다중 행동 추천 시스템이 행동 사이의 관련성을 묵시적으로 모델링하는 데 초점을 맞추었다면, KAMCL은 지식 그래프(knowledge graph)를 활용하여 사용자 의도를 구체적으로 조명하고 이를 기반으로 더 정확한 추천을 제공합니다.

- **Technical Details**: KAMCL은 두 가지 대조 학습(contrastive learning) 방법을 통합하여 데이터 희소성 문제를 완화하고 사용자 표현(user representations)을 강화합니다. 이 모델은 지식 그래프에서의 관계를 사용하여 각 행동에 대한 사용자의 의도를 구성하고, 이를 통해 다중 행동 간의 연결을 보다 효과적으로 탐색합니다.

- **Performance Highlights**: 세 개의 실제 데이터셋에서 실시한 광범위한 실험을 통해 KAMCL 모델이 우수함을 입증하였습니다. 이 모델은 다중 행동 추천에서의 데이터 희소성 문제를 줄이면서 사용자 의도를 정확하게 반영하여 추천의 정확도를 높이는 데 기여합니다.



### SIGformer: Sign-aware Graph Transformer for Recommendation (https://arxiv.org/abs/2404.11982)
Comments: Accepted by SIGIR2024

- **What's New**: SIGformer는 긍정적인 피드백과 부정적인 피드백을 모두 통합하여 서명된 그래프(sign graph)를 형성하는 새로운 접근 방식을 사용합니다. 이 방법은 사용자의 선호도를 더 정확하게 파악할 수 있도록 도와주며, 기존의 GNNs (Graph Neural Networks)나 MLPs (Multi-Layer Perceptrons) 방식이 가지고 있던 한계를 극복하기 위해 트랜스포머(Transformer) 구조를 적용했습니다.

- **Technical Details**: SIGformer는 서명 인식 그래프 기반 추천을 위해 두 가지 혁신적인 위치 인코딩(Positional Encodings)을 도입합니다. 첫 번째, Sign-aware Spectral Encoding (SSE)은 서명된 그래프의 스펙트럼 특성을 포착합니다. 두 번째, Sign-aware Path Encoding (SPE)은 사용자와 아이템 간의 경로 패턴을 분석하여 그래프 내의 협업 관계를 더 잘 이해할 수 있도록 합니다. 이 두 인코딩은 SIGformer가 전체 서명된 그래프를 활용하여 정보를 추출하고, 사용자-아이템 쌍의 유사성을 정확하게 추정하는데 기여합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에서의 광범위한 실험을 통해 SIGformer는 최신(state-of-the-art) 방법들을 뛰어넘는 성능을 보여줍니다. 이 방식은 부정적 피드백을 포함하여 그래프 기반 추천 시스템에서 중요한 협업 정보를 효과적으로 활용하는 것이 가능함을 증명하며, SIGformer가 제공하는 특별한 위치 인코딩이 큰 역할을 하는 것으로 확인됩니다.



### Generating Diverse Criteria On-the-Fly to Improve Point-wise LLM Rankers (https://arxiv.org/abs/2404.11960)
- **What's New**: LLM (Large Language Model, 대규모 언어 모델) 랭커들은 뛰어난 랭킹 성과를 보여 왔으나, 표준화된 비교 지침이 없고 복잡한 내용을 효과적으로 처리하지 못하는 문제들이 있습니다. 이를 해결하기 위해, 다양한 관점에서 기준을 적용하여 랭킹 점수를 생성하는 새로운 접근 방식의 랭커를 제안합니다. 이 접근법은 각 관점에서 독립적이면서도 상호 보완적인 평가를 제공하도록 합니다.

- **Technical Details**: 이 연구는 BEIR 벤치마크의 여덟 데이터셋을 분석하여 포인트와이즈 LLM 랭커의 성능을 향상시키기 위한 다관점 기준 앙상블 방식을 도입했습니다. 제안된 접근 방식은 각기 다른 관점에서 평가 기준을 설정하고, 이 기준들을 통합하여 더 정확하게 문서를 평가하고 순위를 매깁니다.

- **Performance Highlights**: 이 다관점 기준 앙상블 접근법을 적용한 결과, BEIR 벤치마크의 여러 데이터셋에서 포인트와이즈 LLM 랭커의 성능이 현저하게 향상되었습니다. 이는 제안된 방식이 복잡한 패싱지를 다룰 때 더 포괄적인 고려와 표준화된 비교를 가능하게 하여 전반적인 랭킹 결과를 개선했습니다.



### Automated Similarity Metric Generation for Recommendation (https://arxiv.org/abs/2404.11818)
- **What's New**: 최신 연구에서는 추천 시스템(Recommender Systems, RSs)에 대한 새로운 접근 방식을 제안합니다. AutoSMG라고 명명된 자동 유사성 메트릭 생성(Automated Similarity Metric Generation) 방법은 다양한 도메인 및 데이터셋에 맞춤화된 유사성 메트릭을 생성할 수 있습니다. 이 방법은 기본 임베딩 연산자(embedding operators)에서 샘플링하여 유사성 메트릭 공간을 구축하고, 계산 그래프(computational graph)를 사용하여 메트릭을 표현합니다. 진화 알고리즘(evolutionary algorithm)을 사용하여 최적의 메트릭을 반복적으로 검색하여 추천 모델의 성능을 향상시킵니다.

- **Technical Details**: AutoSMG는 먼저 기본 임베딩 연산자에서 샘플링하여 계산 그래프로 메트릭을 구성합니다. 이후, 진화 알고리즘을 통해 새로운 후보 메트릭을 생성하고 최적의 메트릭을 찾습니다. 모델 훈련 시간을 줄이기 위해 초기 중단 전략(early stopping strategy)과 대리 모델(surrogate model)을 사용하여 후보 메트릭의 성능을 예측합니다. 또한, 이 방법은 모델 비의존적(model-agnostic)으로, 다양한 추천 모델 아키텍처에 플러그인으로 쉽게 통합할 수 있습니다.

- **Performance Highlights**: AutoSMG는 세 가지 공개 추천 데이터셋에서 전통적인 수작업 메트릭과 다른 탐색 전략을 사용하여 생성된 메트릭보다 뛰어난 성능을 보였습니다. 이는 AutoSMG가 다양한 도메인에 걸쳐 최적화된 유사성 패턴을 효과적으로 학습하고 적용할 수 있음을 보여줍니다.



### Consolidating Ranking and Relevance Predictions of Large Language Models  through Post-Processing (https://arxiv.org/abs/2404.11791)
- **What's New**: 이 연구는 대규모 언어 모델 (LLMs)을 활용하여 검색 작업에 대한 관련성 라벨을 생성의 새로운 방법을 제시합니다. 특히, 'pairwise ranking prompting' (PRP) 접근 방식을 기반으로 하여 문서 간의 직접적인 비교를 통해 효과적으로 순위를 매기는 방법을 도입하고 있습니다. 여기서는 PRP와 기존 LLM 생성 라벨의 강점을 결합하는 새로운 후처리(post-processing) 방법을 제안하고 있습니다.

- **Technical Details**: LLM에서 생성된 관련성 라벨과 페어와이즈 선호도를 모두 사용하는 새로운 후처리 방법을 제안합니다. 이 방법은 LLM이 생성한 라벨을 조정하여 페어와이즈 선호도를 충족시키는 동시에 원래 값과 가능한 한 가깝게 유지합니다. 후처리 메커니즘은 제약 회귀(constrained regression)를 사용하여 PRP 순위와 LLM 관련성 생성을 통합합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 라벨 정확도와 순위 성능 사이의 균형을 효과적으로 맞출 수 있음을 보여줍니다. 이는 LLM의 순위와 라벨링 능력을 결합함으로써 새로운 최첨단 (state-of-the-art, SOTA) 성능에 접근할 수 있음을 시사합니다.



### Behavior Alignment: A New Perspective of Evaluating LLM-based  Conversational Recommendation Systems (https://arxiv.org/abs/2404.11773)
Comments: Accepted by the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024)

- **What's New**: 이 연구는 대화형 추천 시스템(Conversational Recommender Systems, CRS)에서 대규모 언어 모델(Large Language Models, LLMs)의 적용을 고찰하고, LLM 기반 CRS의 행동 일치도를 측정하는 새로운 평가 지표 'Behavior Alignment'를 제안했습니다. 이 지표는 LLM 기반의 추천 전략이 인간 추천자의 전략과 얼마나 일치하는지 명시적으로 평가할 수 있습니다. 추천 시스템에서 더 인간다운 대화를 구현하고자 하는 새로운 시도입니다.

- **Technical Details**: Behavior Alignment 메트릭은 인간 추천자와 LLM 기반 CRS 사이의 추천 전략의 일치도를 평가합니다. 연구팀은 두 가지 방법을 제안하는데, 하나는 명시적인 방법으로 인간의 주석(annotations)을 필요로 하며, 다른 하나는 분류 기반(classification-based) 방법으로 주석 없이 행동 일치도를 간접적으로 측정할 수 있습니다. 이 두 접근 방식은 CRS 데이터셋에서의 효율성을 입증하였습니다.

- **Performance Highlights**: 실험 결과, Behavior Alignment 지표는 사용자의 선호도와 높은 일치성을 보이며, 기존 메트릭보다 다양한 시스템의 성능을 더 잘 구분할 수 있음을 확인했습니다. 함께 제안된 분류 기반 방법도 강력한 일관성을 보이며, 행동 일치도를 효과적으로 측정하는 것으로 나타났습니다. 이 방법은 기존의 평가 방식보다 인간의 추천 전략과 더 잘 일치하고 비용 및 시간이 많이 드는 인간 주석 없이도 선택적으로 사용될 수 있는 장점이 있습니다.



### A Learning-to-Rank Formulation of Clustering-Based Approximate Nearest  Neighbor Search (https://arxiv.org/abs/2404.11731)
- **What's New**: 이 연구는 약간의 근접 이웃(ANN) 검색에 집중하여 클러스터링 기반 방법을 탐구합니다. 특히, 데이터를 파티션으로 나누고 각 파티션을 대표 벡터(centroid)로 표현하는 기존 방식에 머신 러닝, 특히 순위 매기기 문제(learning-to-rank)를 적용하여 ANN 검색의 정확성을 향상시키는 방법을 개발했습니다. 이러한 접근 방식은 주어진 쿼리에 대해 최상의 클러스터를 예측하고 그 클러스터에서 내부 검색을 수행하여 정확도를 개선합니다.

- **Technical Details**: 이 작업에서는 데이터를 L 개의 클러스터로 나누고 각 클러스터를 centroid로 표현합니다. 검색 시, 알고리즘은 쿼리와 centroid 사이의 내적을 계산하여 가장 가까운 클러스터를 식별하고 해당 클러스터에 대해서만 ANN 검색을 수행합니다. 또한, 머신 러닝 접근법인 Learning-to-Rank를 사용하여 클러스터링 기반의 MIPS(Maximum Inner Product Search)에 대한 라우팅 함수를 학습합니다. 이는 간단한 선형 함수를 학습하여 각 클러스터에 대한 순위 점수를 생성하고, 이를 통해 최초의 라우팅 메커니즘에 플러그인할 수 있습니다.

- **Performance Highlights**: 실험적 결과는 여러 텍스트 데이터셋에 대한 임베딩을 사용하여 라우팅 함수 학습이 ANN 검색의 정확도를 일관되게 향상시킬 수 있음을 보여주었습니다. 특히 이 접근 방식은 클러스터링 알고리즘의 선택에 관계없이 일관된 이득을 제공합니다. 이 결과는 ANN 검색과 Learning-to-Rank의 융합이 가지는 잠재적인 가능성을 일깨워주며 이 분야에 대한 추가 연구를 독려합니다.



### Accounting for AI and Users Shaping One Another: The Role of  Mathematical Models (https://arxiv.org/abs/2404.12366)
- **What's New**: 이 논문은 인공지능(AI) 시스템과 사용자 간의 상호 작용을 수학적으로 모델링하는 새로운 접근법인 '형식적 상호작용 모델(formal interaction models)'을 제안합니다. 이 모델들은 AI가 사회에 미치는 영향을 예측하고, 통제할 수 있는 방법을 제공함으로써, AI 설계와 평가의 방향을 새롭게 제시합니다.

- **Technical Details**: 형식적 상호작용 모델은 AI 시스템과 사용자 간의 동적 상호작용을 명확하게 정의하는 수학적 모델입니다. 이 연구에서는 구체적으로 컨텐츠 추천 시스템(content recommender systems)을 사례로 사용하여, 이러한 모델이 어떻게 사용자의 선호도와 행동에 영향을 미칠 수 있는지를 검토합니다. 이 모델들은 비선형 동역학(nonlinear dynamics), 게임 이론(game theory), 밴디트 문제(bandits), 그리고 행동 심리학(behavioral psychology)과 같은 다양한 접근법을 사용하여 설계될 수 있습니다.

- **Performance Highlights**: 형식적 상호작용 모델을 사용함으로써, AI 시스템 설계자와 평가자는 사용자와의 상호작용을 더 정확하게 예측하고, 이를 통해 사회에 미치는 불의도적인 영향을 줄일 수 있습니다. 이 모델들은 상호작용을 구체화(specifying interactions), 실제 상호작용 모니터링(monitoring interactions), 사회적 영향 예측(anticipating societal impacts) 및 개입을 통한 영향 조정(controlling societal impacts) 등 여러 사용 사례에서 유익하게 활용될 수 있습니다.



### iRAG: An Incremental Retrieval Augmented Generation System for Videos (https://arxiv.org/abs/2404.12309)
- **What's New**: iRAG는 기존의 검색 보강 생성(Retrieval Augmented Generation, RAG) 시스템을 확장하여 대규모 멀티모달 데이터를 신속하게 인덱싱하고 상호 작용적 쿼리에 대응할 수 있는 새로운 점진적 워크플로(incremental workflow)를 제안합니다. 이를 통해 비디오, 이미지, 텍스트 등을 포함한 멀티모달 데이터에 대한 실시간 상호 작용적 질의 응답이 가능해집니다. 기존 RAG 방식에서 모든 비디오 데이터를 텍스트로 전환한 후 쿼리를 처리하는 방식과 달리, iRAG는 필요할 때 선택적으로 데이터를 추출하여 처리 시간을 대폭 줄이고 응답 품질을 유지합니다.

- **Technical Details**: iRAG 시스템은 쿼리 플래너(Query Planner), 인덱서(Indexer), 추출기(Extractor) 세 가지 주요 컴포넌트로 구성됩니다. 쿼리 플래너는 사용자의 쿼리에 따라 분석이 필요한 비디오 섹션을 식별하고 상세 분석에 사용될 AI 모델을 선택합니다. 인덱서는 AI 기반의 재순위 결정 방식(re-ranking method)을 이용하여 처리해야 할 비디오의 부분을 효율적으로 줄입니다. 추출기는 쿼리와 관련된 상세 정보를 추출하고 인덱스를 업데이트하여 쿼리에 적합한 컨텍스트를 검색합니다.

- **Performance Highlights**: 실제 장시간 비디오 데이터에 대한 실험 결과, iRAG는 기존 비디오를 텍스트로 전환하는 과정에서 23배에서 25배 빠른 처리 속도를 보였으며, 대화형 쿼리에 대한 응답 품질은 전통적인 RAG 시스템과 비교할 때 비슷한 수준을 유지했습니다.



### Estimating the Hessian Matrix of Ranking Objectives for Stochastic  Learning to Rank with Gradient Boosted Trees (https://arxiv.org/abs/2404.12190)
Comments: SIGIR2024 conference Short paper track

- **What's New**: 이 연구는 확률적 순위 결정 (Stochastic Learning to Rank, LTR)의 분야에서 Gradient Boosted Decision Trees (GBDTs)를 사용하는 최초의 방법론을 소개하고 있습니다. 이는 기존의 확률적 LTR 연구가 주로 뉴럴 네트워크 (Neural Networks, NNs)에 초점을 맞추고 있었던 것과 대조적입니다. 특히, 이 방법론은 GBDTs에 필요한 두 번째 순서 도함수(Second-order derivatives), 즉 헤시안 행렬 (Hessian matrix)의 추정 방법을 개발하였습니다.

- **Technical Details**: 연구팀은 확률적 LTR에서의 주요 단점인 그라디언트 추정 문제를 해결하기 위해 헤시안 행렬의 효율적인 계산을 가능하게 하는 새로운 추정기를 개발했습니다. 이는 기존의 PL-Rank 프레임워크 내에 통합되어, 첫 번째 및 두 번째 순서 도함수를 동시에 계산할 수 있도록 하였습니다. 이 방법은 확률적 LTR 전략에 GBDTs를 접목시킴으로써, 다양성 증진, 문서의 공정한 노출 증가 및 탐색과 활용의 균형을 잡는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 우리의 헤시안 추정 방법을 통해 GBDTs는 NNs에 비해 여러 LTR 벤치마크에서 상당한 우위를 보였습니다. GBDTs를 사용했을 때에는 NNs와 달리 안정적인 수렴을 보여, 조기 종료(Early stopping)가 필요하지 않았습니다. 이는 결정적 LTR (Deterministic LTR)과의 중요한 격차를 해소하며 GBDTs의 성능과 안정성을 크게 향상시키는 데 기여했습니다.



### Shotit: compute-efficient image-to-video search engine for the cloud (https://arxiv.org/abs/2404.12169)
Comments: Submitted to ACM ICMR 2024

- **What's New**: 새로운 이미지-동영상 검색 엔진인 Shotit은 클라우드 기반 기술을 활용하여 효율적인 방식으로 동영상 검색 서비스를 제공합니다. Shotit은 대규모 데이터셋을 효율적으로 관리하며, vector database를 사용하여 빠르고 정확한 검색 결과를 제공합니다.

- **Technical Details**: Shotit는 대용량 동영상 데이터에 대한 이미지-동영상 검색 서비스를 제공합니다. 이 시스템은 이미지의 Color Layout을 통해 해시 데이터를 생성하고, 이를 vector database에 삽입하여 검색합니다. 사용된 주요 기술은 Milvus(vector database), LireSolr, Apache Solr 및 Faiss(ANN 검색)입니다.

- **Performance Highlights**: Shotit은 Blender Open Movie 데이터셋과 5000만 규모의 TV 장르 데이터셋에서 실험을 진행, 뛰어난 성능을 보여주었습니다. Intel Xeon Gold 6271C을 사용하여 효과적인 결과를 도출해냈으며, 이미지-비디오 검색에서 높은 정확도와 효율성을 입증하였습니다.



### A Fast Maximum Clique Algorithm Based on Network Decomposition for Large  Sparse Networks (https://arxiv.org/abs/2404.11862)
Comments: 12 pages, 2 figures, 1 table

- **What's New**: 새롭게 제안된 알고리즘은 Large Sparse Networks에서 최대 클리크(Maximum Clique) 문제를 정확하게 해결하기 위해 효율적인 그래프 분해 기반을 사용합니다. 특히 Complete-Upper-Bound-Induced Subgraph (CUBIS)라는 새로운 개념을 도입하여 그래프의 분해 과정에서 최대 클리크를 형성할 수 있는 구조가 유지되도록 했습니다.

- **Technical Details**: 이 알고리즘은 먼저 주변의 말단 노드들을 사전에 제거한 후(core number와 현재의 최대 클리크 크기에 의해 안내되는) 하나 또는 두 개의 소규모 CUBIS를 구성합니다. 각 CUBIS에서는 Bron-Kerbosch 탐색 알고리즘을 사용하여 최대 클리크를 찾습니다. 이 방법은 그래프의 규모에 크게 의존하지 않기 때문에 대규모 네트워크에서도 근사적으로 선형 시간(Linear Runtime)에 실행될 수 있습니다.

- **Performance Highlights**: CUBIS 개념을 도입한 결과, 알고리즘의 성능은 원래 네트워크의 규모에 대체로 독립적이며, 이는 최대 2천만까지의 네트워크 스케일에서 실험적으로 확인되었습니다. 이는 알고리즘의 확장성과 빠른 실행 속도를 보여줍니다. 따라서 이 알고리즘은 대규모 희소 그래프(Sparse Graphs)에서의 최대 클리크 문제를 효과적으로 해결할 수 있는 새로운 프레임워크를 제공합니다.




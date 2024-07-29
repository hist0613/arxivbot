### Optimizing E-commerce Search: Toward a Generalizable and Rank-Consistent Pre-Ranking Mod (https://arxiv.org/abs/2405.05606)
- **What's New**: 새로 개발된 GRACE(Generalizable and RAnk-ConsistEnt Pre-Ranking Model) 모델은 이커머스 플랫폼의 검색 시스템에서 사전 순위 결정 단계의 성능 향상에 중점을 두고 있습니다. 이 모델은 상품이 최종 순위 모델에서 상위 k개 결과 내에 속하는지 예측하는 다중 이진 분류 작업을 도입함으로써 순위 일관성을 달성합니다.

- **Technical Details**: GRACE 모델은 1) 순위 일관성을 위해, 상품이 최종 순위 결과의 상위 'top-k' 내에 속할 것인지 예측하는 다수의 이진 분류 작업과 이것을 위한 새로운 학습 목표를 정상적인 point-wise ranking 모델에 추가하는 방식으로 구현됩니다. 2) 일반화는 전체 상품에 대한 대조학습(contrastive learning)을 통해, 특정 상위 랭킹 상품 임베딩의 한 부분에 대한 사전 훈련(pre-training)으로 수행됩니다.

- **Performance Highlights**: GRACE 모델은 오프라인 지표와 온라인 A/B 테스트에서 두드러진 개선을 보였습니다. AUC(면적 아래의 곡선)은 0.75% 증가하였고, CVR(전환 비율, Conversion Rate)은 1.28% 증가하였습니다.



### Review-based Recommender Systems: A Survey of Approaches, Challenges and Future Perspectives (https://arxiv.org/abs/2405.05562)
Comments:
          The first two authors contributed equally

- **What's New**: 이 논문에서는 최근 몇 년간 리뷰 기반 추천 시스템(Review-Based Recommender Systems)의 발전에 대해 종합적으로 검토하여 리뷰가 추천 시스템에서 중요한 역할을 하며, 평가와의 통합을 위해 리뷰에서 특징을 추출하는 데 연관된 도전들을 강조합니다. 이러한 분야에서의 최신 방법들을 분류하고 요약하며, 그 특성, 효과 및 한계를 분석합니다.

- **Technical Details**: 이 논문은 리뷰를 통해 수집할 수 있는 사용자의 세밀한 선호와 제품 특성을 분석하는 것의 중요성에 초점을 맞추고 있습니다. 전통적인 추천 시스템은 사용자의 명시적 평가나 암시적 상호작용(Implicit Interactions: 예를 들어 '좋아요', 클릭, 공유, 저장 등)을 통해 학습합니다. 하지만, 텍스트 리뷰는 사용자의 정교한 선호도와 아이템 특징을 제공하며, 이를 추천 결과의 성능과 해석 가능성(Interpretability) 향상에 필수적입니다. 또한, 리뷰 기반 시스템 분류 및 최신 방법에 대한 요약이 제공되며, 특징 추출과 평가 통합의 도전 사례들도 다룹니다.

- **Performance Highlights**: 이 논문은 리뷰 기반 추천 시스템의 효과적인 성능과 한계점을 분석하며, 특히 리뷰 데이터에서 유의미한 정보를 추출하여 사용자 맞춤형 추천의 해석 가능성과 정확도를 개선할 수 있는 방법을 제안합니다. 리뷰에서 텍스트 데이터를 활용하는 다양한 최신 기술들이 언급되어 있으며, 특히 멀티 모달 데이터(Multi-modal Data), 다중 기준 평가 정보(Multi-criteria Rating Information) 통합의 잠재적 연구 방향이 제시됩니다.



### Redefining Information Retrieval of Structured Database via Large Language Models (https://arxiv.org/abs/2405.05508)
- **What's New**: 이 논문은 언어 모델(Language Models, LMs)이 외부 지식 베이스를 통해 문의와 관련된 비파라메트릭(non-parametric) 지식을 활용할 때 검색 보강(retrieval augmentation)의 중요성과 함께, 새로운 검색 보강 프레임워크인 ChatLR을 소개합니다. 이 프레임워크는 대규모 언어 모델(Large Language Models, LLMs)의 강력한 의미 이해 능력을 활용하여 정확하고 간결한 정보 검색을 달성합니다.

- **Technical Details**: ChatLR은 LLMs를 사용하여 질의와 관련된 정보를 효율적으로 추출할 수 있는 검색자(retriever)로 활용합니다. 이 프레임워크는 금융 분야에 특화된 검색 및 질의응답 시스템(question answering system)을 구축하기 위해 ‘Text2API’ 및 ‘API-ID 인식(API-ID recognition)’ 작업에 대해 LLM을 파인튜닝(fine-tuning)합니다.

- **Performance Highlights**: ChatLR은 사용자 질의에 대응하는 데 효과적임을 입증하며, 전체 정보 검색 정확도는 98.8%를 초과하는 높은 성능을 보여줍니다.



### Information Extraction from Historical Well Records Using A Large Language Mod (https://arxiv.org/abs/2405.05438)
- **What's New**: 이 연구에서는 버려진 석유 및 가스 우물(이른바 고아 우물)에 대한 정보를 신속하고 효율적으로 추출하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 사용하는 새로운 컴퓨터 기반 접근 방식을 제안합니다. 특히, 이 논문은 Llama 2 모델을 기반으로 한 정보 추출 워크플로우를 개발하고, 이를 160개 우물 문서 데이터셋에 테스트하여 그 효율성을 검증합니다.

- **Technical Details**: 연구팀은 고아 우물 문서에서 위치와 깊이 정보를 추출하기 위해 OCR(Optical Character Recognition) 기술과 LLM을 결합한 정보 추출 워크플로우를 개발했습니다. Llama 2는 7억 개에서 700억 개의 파라미터를 가진 여러 가지 사전 훈련된 모델로 구성되어 있으며, 이 모델들은 공개적으로 이용 가능합니다. 정보 추출을 위한 프롬프트(prompt)는 지시(instruction), 맥락(context), 입력 텍스트(input text)의 세 가지 요소로 구성되며, 체인-오브-소트(Chain-of-Thought) 전략을 포함하여 최적화되었습니다.

- **Performance Highlights**: 개발된 워크플로우는 PDF 기반의 깨끗한 보고서에서 위치와 깊이 정보를 추출할 때 100%의 정확도를 달성했습니다. 그러나 구조화되지 않은 이미지 기반의 문서에서는 정확도가 70%로 감소합니다. 이 기술은 수동으로 데이터를 디지털화하는 데 드는 노동력을 줄이고 자동화를 증가시키는 중요한 이점을 제공합니다.



### Optimal Baseline Corrections for Off-Policy Contextual Bandits (https://arxiv.org/abs/2405.05736)
- **What's New**: 이 연구는 일반적인 평가와 추천 시스템(Contextual Bandits, Reinforcement Learning)에서의 문제를 개선하는 새로운 방법을 제시합니다. 특히, off-policy learning에서의 variance reduction을 위한 최적의 기법을 제안하며, 이를 통해 추천 시스템과 관련된 결정 문제를 효과적으로 해결할 수 있도록 지원합니다.

- **Technical Details**: 저자들은 Control Variates와 같은 방법들을 활용하여, IPS (Inverse Propensity Scoring) estimator의 아이디어를 발전시키고, 동시에 variance를 줄이는 방법을 탐구합니다. 이들은 Multiplicative (Self-Normalisation) 및 Additive (Baseline Corrections, Doubly Robust Estimators) 방법들의 동등함을 이용하여 새로운 프레임워크를 구축했으며, 이를 통해 variance-optimal unbiased estimator를 도출해냈습니다. 또한, 이 연구에서는 gradient variance를 줄이기 위한 새로운 baseline correction 방법을 제안하고, 실제 구현을 용이하게 하는 closed-form solution을 제시합니다.

- **Performance Highlights**: 이론적 발견을 실증적으로 증명하기 위해 사용된 off-policy simulation 환경은 실제 추천 시나리오를 모방하며, stochastic rewards, large action spaces, 그리고 controlled randomisation을 포함합니다. 실험 결과는 제안된 baseline correction이 학습 도중 더 빠른 수렴과 낮은 gradient variance를 가능하게 함을 보여줍니다. 또한, 최적의 baseline correction은 폭넓게 사용되는 doubly-robust 및 SNIPS estimators보다 낮은 policy value estimation 오류를 달성함을 입증합니다.



### LayerPlexRank: Exploring Node Centrality and Layer Influence through Algebraic Connectivity in Multiplex Networks (https://arxiv.org/abs/2405.05576)
- **What's New**: 새로운 알고리즘인 LayerPlexRank가 복합 네트워크에서 노드의 중심성(node centrality)과 레이어의 영향력(layer influence)을 동시에 평가합니다. 이 방법은 다중 레이어 네트워크(multiplex networks)에서의 대수적 연결성(algebraic connectivity) 지표를 사용하여 노드의 중요도를 측정합니다.

- **Technical Details**: LayerPlexRank는 랜덤 워크(random walk)를 이용하여 레이어 간의 구조적 변화를 효과적으로 평가하고, 전체 그래프의 연결성을 고려합니다. 이 알고리즘은 다양한 실제 데이터셋에서 이론적 분석과 경험적 검증을 거쳐 개발되었습니다.

- **Performance Highlights**: LayerPlexRank는 기존의 중심성 척도(centrality measures)와 비교하여 그 효율성과 정확성에서 두드러진 성능을 보여줍니다. 이러한 결과는 다양한 실제 네트워크 데이터셋에서 얻은 것으로, 알고리즘의 유용성을 입증합니다.




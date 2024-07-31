### A Preference-oriented Diversity Model Based on Mutual-information in Re-ranking for E-commerce Search (https://arxiv.org/abs/2405.15521)
- **What's New**: 현재의 재정렬(re-ranking) 알고리즘들이 검색 결과의 정확성을 높이는 데 집중하는 반면, 다각적인 사용자 요구를 충족시키기 위해 다양성도 고려한 새로운 모델이 제안되었습니다. 본 논문에서는 정확성과 다양성 두 가지 요소를 모두 고려한 Preference-oriented Diversity Model Based on Mutual-information (PODM-MI)을 소개합니다. 이 모델은 Multi-dimensional Gaussian distribution을 사용하여 사용자의 다양한 선호도를 불확실성과 함께 포착하고, 변분 추론(maximum variational inference lower bound)을 통해 후보 항목과 사용자 선호도 간의 상호 정보를 극대화하여 상관성을 향상시킵니다.

- **Technical Details**: PODM-MI 모델은 두 가지 주요 구성 요소인 preference-oriented network (PON)와 self-adaptive model (SAM)으로 구성됩니다. PON은 사용자의 다양성 선호도와 후보 항목 내의 다양성을 Multi-dimensional Gaussian distribution을 통해 모델링합니다. 이 방법은 기존의 embedding 기법보다 높은 차원에서 다양성의 불확실성을 더 잘 포착할 수 있는 장점을 지닙니다. 이후 SAM 모듈을 통해 사용자의 다양성 선호도와 후보 항목 간의 상호 정보를 극대화하여 항목의 순위를 동적으로 조정합니다. 상호 정보 극대화를 위해 변분 포스터리어 추정기(variational posterior estimator)를 도입하여 상호 정보 목표의 낮은 경계(lower bound)를 도출합니다.

- **Performance Highlights**: 실제 전자상거래 시스템에서 실험한 결과, PODM-MI 모델은 기존 최첨단 모델에 비해 상당한 개선을 이루었습니다. 특히, JD의 홈 검색 플랫폼에 성공적으로 배포된 후, 명백한 경제적 이익을 창출했습니다.



### From Data Complexity to User Simplicity: A Framework for Linked Open Data Reconciliation and Serendipitous Discovery (https://arxiv.org/abs/2405.15520)
Comments:
          6 pages, 3 figures, part of "XIII Convegno Annuale AIUCD 2024 Proceedings"

- **Whats New**: 이 논문은 Linked Open Data 소스를 정렬하고, 사용자가 쉽게 정보를 발견할 수 있는 웹 포털 소프트웨어 솔루션을 소개합니다. Polifonia Web 포털을 동기 부여 시나리오 및 사례 연구로 사용하여 데이터 재정렬 및 음악 유산 분야에서 사용자 친화적인 인터페이스 제공과 같은 연구 문제를 다룹니다.

- **Technical Details**: 이 솔루션은 혼란스러운 데이터를 정리(data reconciliation)하고, 관대한 인터페이스(generous interfaces)로 사용자 경험을 향상시키는 것을 목표로 합니다. 특히, Linked Open Data 소스를 통합하여 다양한 데이터를 쉽게 접근 가능하도록 하는 기술이 포함됩니다.

- **Performance Highlights**: Polifonia Web 포털은 사용자 친화적인 인터페이스 덕분에 정보 발견 과정에서 매우 효과적인 것으로 나타났습니다. 음악 유산 도메인에서 데이터를 일관되게 제공함으로써 사용자 탐색 경험을 크게 향상시켰습니다.



### Hybrid Context Retrieval Augmented Generation Pipeline: LLM-Augmented Knowledge Graphs and Vector Database for Accreditation Reporting Assistanc (https://arxiv.org/abs/2405.15436)
Comments:
          17 pages, 9 figures

- **What's New**: 본 연구는 고등 교육 기관의 품질 보증 과정인 인증에 있어서 중요한 요소인 Association to Advance Collegiate Schools of Business (AACSB) 인증을 위한 새로운 하이브리드 문맥 검색 증강 생성 파이프라인 (context retrieval augmented generation pipeline)을 제안합니다. 이 파이프라인은 인증을 위해 필수적인 문서 정렬과 보고 과정을 지원합니다.

- **Technical Details**: 본 프로젝트에서는 벡터 데이터베이스(vector database)와 지식 그래프(knowledge graph)를 활용하여 기관의 데이터와 AACSB 기준 데이터를 지식 저장소로 구축하였습니다. 지식 그래프는 수동 구성 프로세스(manual construction process)와 LLM Augmented Knowledge Graph 접근 방식을 사용하여 개발되었습니다. 이렇게 생성된 파이프라인의 출력을 통해 기관의 이해 관계자들은 인증 보고서를 작성할 수 있습니다.

- **Performance Highlights**: 파이프라인의 성능은 RAGAs 프레임워크를 사용하여 평가되었으며, 답변의 적절성(answer relevancy) 및 답변 정확성(answer correctness) 지표에서 최적의 성능을 보였습니다.



### DFGNN: Dual-frequency Graph Neural Network for Sign-aware Feedback (https://arxiv.org/abs/2405.15280)
Comments:
          Accepted by KDD 2024 Research Track

- **What's New**: 이번 연구는 그래프 기반 추천 시스템에서 긍정적 피드백뿐만 아니라 부정적 피드백(예: 싫어요, 낮은 평점)까지 모델링하는 새로운 접근법을 제안합니다. 제안된 모델은 'Dual-frequency Graph Neural Network for Sign-aware Recommendation (DFGNN)'으로, 이는 사용자의 긍정적 및 부정적 피드백을 주파수 필터의 관점에서 다룹니다.

- **Technical Details**: DFGNN은 주파수 필터링을 통해 긍정적 피드백(저주파 신호)과 부정적 피드백(고주파 신호)을 캡처합니다. 이를 위해 이중 주파수 그래프 필터(DGF)를 설계했고, 저주파 필터는 긍정적 피드백을, 고주파 필터는 부정적 피드백을 모델링하는 데 사용됩니다. 또한, 서명된 그래프 정규화(Signed Graph Regularization, SGR) 손실을 적용하여 임베딩의 탈제네레이션(representation degeneration) 문제를 해결합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 광범위한 실험 결과, 제안된 DFGNN 모델이 경쟁 모델에 비해 크게 향상된 성능을 보였습니다. 이는 긍정적 및 부정적 피드백을 모두 효과적으로 모델링함으로써 달성된 결과입니다.



### Shopping Queries Image Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product Search (https://arxiv.org/abs/2405.15190)
- **What's New**: 최근 정보 검색(Information Retrieval) 및 머신 러닝(Machine Learning) 분야의 진보는 특히 온라인 쇼핑 세계에서 사용자 경험을 향상시키기 위해 검색 엔진의 성능 향상에 중점을 두고 있습니다. 본 논문은 Amazon Shopping Queries Dataset에 이미지 정보를 추가한 Shopping Queries Image Dataset(SQID)을 소개합니다. SQID는 190,000개 제품과 연관된 이미지 정보를 포함하고 있어, 텍스트와 시각 정보를 동시에 활용하는 멀티모달 학습(multimodal learning) 기법 연구를 촉진합니다.

- **Technical Details**: 본 논문에서는 SQID 데이터셋을 사용해 사전 학습된 모델(pretrained models)을 활용한 실험 결과를 제시합니다. 특히 KDD Cup 2022의 Task 1에 해당하는 쿼리-제품 랭킹(query-product ranking)에서 성능을 평가합니다. 다양한 방법론을 통해 텍스트와 이미지 데이터를 결합하여 성능을 향상시키는 방법도 다룹니다. 랭킹 품질은 정규화된 이익 누적 이득(Normalized Discounted Cumulative Gain, NDCG)을 사용해 측정됩니다.

- **Performance Highlights**: 기본 ESCI_baseline(BERT 모델 사용) 방법론에 비해, 멀티모달 데이터를 활용했을 때 검색 및 랭킹 성능이 개선되었습니다. 구체적으로, SQID 데이터셋을 사용한 결과, 전체적인 NDCG 점수가 향상됨을 확인했습니다. 이를 통해 텍스트와 이미지 데이터를 결합하는 접근 방식이 제품 검색 정확도를 향상시키는 데 있어 효과적임을 보여줍니다.



### Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning (https://arxiv.org/abs/2405.15114)
- **What's New**: 새로운 Recommender System (RS) 프레임워크인 ToolRec가 소개되었습니다. 이는 Large Language Models (LLMs)을 사용자가 의사 결정하는 과정을 시뮬레이션 하여 추천 과정을 안내하고 외부 도구들을 활용해 사용자의 미세한 선호도와 일치하는 추천 목록을 생성합니다.

- **Technical Details**: ToolRec는 기본적으로 세 가지 주요 구성 요소로 이루어집니다: (i) 사용자 의사 결정 시뮬레이션 모듈: LLMs을 사용하여 사용자의 행동 이력을 바탕으로 현재 상황에서 사용자의 선호도를 평가합니다. (ii) 속성 지향 도구: 순위 도구와 검색 도구의 두 가지 세트를 개발하여 각기 다른 속성을 기반으로 아이템을 평가합니다. (iii) 메모리 전략: 중간 결과를 저장하고 이를 참고하여 최종 추천 목록을 개선하는 데 도움을 줍니다.

- **Performance Highlights**: 세 개의 실제 데이터셋에서 실험 결과, ToolRec는 특히 세계 지식이 풍부한 분야에서 높은 효율성을 입증했습니다. LLM 도구 학습을 통해 사용자의 의사 결정 과정을 시뮬레이션하면서 더 정확한 추천을 생성할 수 있었습니다.



### Self-distilled Dynamic Fusion Network for Language-based Fashion Retrieva (https://arxiv.org/abs/2405.15451)
Comments:
          ICASSP 2024

- **What's New**: 패션 이미지 검색 분야에서 참조 이미지와 이를 설명하는 텍스트를 이용하여 정확한 패션 아이템을 찾는 것은 흥미로운 도전입니다. 기존의 방법들은 이미지와 텍스트를 정적으로 융합하는 기법에 의존하고 있으며, 이러한 접근법들은 유연성이 부족하다는 문제를 가지고 있습니다. 이에 대해, 우리는 '자가 증류 다이내믹 융합 네트워크(Self-distilled Dynamic Fusion Network, SDFN)'를 제안합니다. 이 네트워크는 경로 일관성과 모달리티별 정보를 동시에 고려하여 다중-세밀도(multi-granularity) 특징을 동적으로 구성합니다.

- **Technical Details**: 우리의 제안에는 두 가지 새로운 모듈이 포함됩니다: 1) '모달리티 특정 라우터(Modality Specific Routers)'를 사용한 다이내믹 융합 네트워크. 이 다이내믹 네트워크는 각각의 참조 이미지와 수정 텍스트의 고유한 의미와 분포를 고려하여 경로를 유연하게 결정할 수 있도록 합니다. 2) '자가 경로 증류 손실(Self Path Distillation Loss)'. 이 모듈은 쿼리의 안정적 경로 결정을 통해 특징 추출과 라우팅 최적화를 돕습니다. 우리는 이전 경로 정보를 사용하여 점진적으로 경로 결정을 개선합니다.

- **Performance Highlights**: 우리는 세 가지 널리 사용되는 CIR(Composed Image Retrieval) 벤치마크에서 철저한 평가를 수행했으며, 실험 결과에 따르면 우리의 접근법이 기존 방법들보다 우수함을 보여줍니다.



### Multi-Modal Recommendation Unlearning (https://arxiv.org/abs/2405.15328)
- **What's New**: 이 논문은 Multi-Modal 추천 시스템(MMRS)에서 처음으로 사용자의 개인 정보 유출 방지를 위한 새로운 프레임워크인 MMRecUN을 제안합니다. 기존의 추천 시스템에서 발생하는 문제점들을 해결하기 위해 새로운 타입의 개인 정보 보호 메커니즘을 도입했습니다. 이 연구는 특히 Multi-Modal 정보가 사용자 선호도에 미치는 영향을 고려하여, 사용자 데이터 및 아이템 관련 데이터를 안전하게 '잊고' 성능 저하를 최소화하는 방법을 제공합니다.

- **Technical Details**: MMRecUN은 훈련된 추천 모델과 특정 데이터를 잊어야 하는 'forget data'를 기반으로 Reverse Bayesian Personalized Ranking (BPR) objective를 사용하여 모델이 해당 데이터를 잊도록 강제합니다. 이 프레임워크는 역방향 및 순방향 BPR 손실 메커니즘을 동시에 사용하여 잊어야 하는 데이터의 영향력을 줄이면서 유지해야 할 데이터의 중요성을 강화합니다. 이러한 과정은 MMRS의 그래프 구조와 특징 임베딩(embedding)을 결합하여 작동합니다. 기존 방법들과의 차별점은 MMRecUN이 상호작용(interactions)을 세밀하게 조정할 수 있다는 것입니다.

- **Performance Highlights**: MMRecUN은 여러 벤치마크 데이터셋에서 다양한 요청에 대해 기존 방법보다 높은 성능을 보였습니다. 리콜 성능에서 최대 49.85%의 개선을 달성하였으며, 'Gold' 모델보다 1.3배 더 빠릅니다. 또한 목표 요소 제거의 성능을 유지하면서 성능을 저하시키지 않고 오버헤드 비용을 최소화하는 상용구를 제공합니다.




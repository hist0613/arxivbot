### Mixed Supervised Graph Contrastive Learning for Recommendation (https://arxiv.org/abs/2404.15954)
- **What's New**: 이 논문에서는 추천 시스템(Recommender Systems, RecSys)의 효율성과 정확성을 높이기 위해 'Mixed Supervised Graph Contrastive Learning for Recommendation (MixSGCL)'이라는 새로운 접근 방식을 제안합니다. MixSGCL은 추천 작업과 대조적 손실(contrastive losses) 학습을 감독된 대조적 학습 손실(supervised contrastive learning loss)로 통합하여 한 가지 최적화 방향으로 조정합니다.

- **Technical Details**: MixSGCL은 사용자-아이템 이분 그래프(bipartite graph)에서의 자기 감독 없는 증대(unsupervised augmentation) 대신 노드별(node-wise) 및 엣지별(edge-wise) 혼합(mixup)을 사용하여 직접적인 감독된 협업 필터링 신호(collaborative filtering signals)를 추가로 채굴합니다. 이는 데이터 희소성 문제를 해결하고, 전체 최적화 과정의 일관성을 개선하는 데 도움을 줍니다.

- **Performance Highlights**: 세 개의 실제 세계 데이터셋에서 실시한 광범위한 실험을 통해 MixSGCL이 최신 방법론(state-of-the-art methods)을 능가하는 것으로 나타났습니다. MixSGCL은 정확도(accuracy)와 효율성(efficiency) 모두에서 최고 성능을 달성하며, MixSGCL의 감독된 그래프 대조 학습(supervised graph contrastive learning)에 대한 설계의 유효성을 입증합니다.



### Telco-RAG: Navigating the Challenges of Retrieval-Augmented Language  Models for Telecommunications (https://arxiv.org/abs/2404.15939)
Comments: 6 pages, 5 Figure, 4 Tables, submitted to IEEE Globecom 2024 (see this https URL)

- **What's New**: 이 논문에서는 통신 분야의 복잡한 문서와 빠르게 진화하는 분야의 특성을 고려하여 개발된 Telco-RAG (텔코-RAG) 시스템을 소개합니다. Telco-RAG는 텔레커뮤니케이션(RAG) 표준, 특히 3rd Generation Partnership Project (3GPP) 문서를 전문으로 다루는 RAG(Retrieval-Augmented Generation) 프레임워크입니다.

- **Technical Details**: Telco-RAG는 복잡한 채팅봇 개발과 표준 문서 처리에 초점을 맞춘 특수화된 RAG 파이프라인을 구현합니다. 이 시스템은 사용자 쿼리를 향상시키고, 신경망(Neural Network, NN) 라우터를 사용하여 문서 코퍼스에서 관련 문서를 선택적으로 식별합니다. GPT-3.5와 같은 최신 언어 모델을 이용하여 검색된 컨텍스트에 기반한 응답을 생성합니다. 추가로, 낮은 RAM 사용을 위한 기법과 특정 쿼리에 기반한 3GPP 시리즈를 정확히 예측하는 NN 라우터 아키텍처도 개발했습니다.

- **Performance Highlights**: Telco-RAG의 구현은 기술적 문서 처리의 정확도와 효율성을 크게 향상시키고, 파라미터 튜닝 최적화를 통한 성능 개선을 보여줍니다. 이러한 RAG 시스템은 통신 전문가들이 국제 표준에 더 빠르고 정확하게 접근하고 준수하는 데 도움을 주어, 개발 사이클을 가속화하고 규제 준수를 개선합니다.



### Retrieval and Distill: A Temporal Data Shift Free Framework for Online  Recommendation System (https://arxiv.org/abs/2404.15678)
- **Newsletter**: [{"What's New": "최신 연구는 시간이 변화하는 데이터에 대한 클릭률 (Click-Through Rate, CTR) 예측 모델의 효과적인 적용을 위해 '검색 및 증류 패러다임' (Retrieval and Distill paradigm, RAD)을 제안합니다. 이 방법은 변화하는 데이터를 활용하여 CTR 모델의 성능을 향상시키는 새로운 접근 방식입니다."}, {'Technical Details': 'RAD는 검색 프레임워크 (Retrieval Framework)와 증류 프레임워크 (Distill Framework)의 두 주요 구성 요소로 구성됩니다. 검색 프레임워크는 유사한 데이터를 검색하여 원래의 CTR 모델에 통합하는 역할을 하며, 증류 프레임워크는 이러한 데이터에서 지식을 추출하여 경량화된 학습 모델을 생성합니다. 이 과정은 빠른 인퍼런스와 효율적인 온라인 배포를 가능하게 합니다.'}, {'Performance Highlights': 'RAD는 다양한 실제 데이터셋에서 광범위한 실험을 통해 기존 모델보다 우수한 성능을 보여줌으로써, 시간의 변동성 데이터를 효과적으로 활용하여 CTR 모델의 성능을 개선할 수 있음을 입증하였습니다. 또한, 이 방법은 모델의 시간 및 공간 복잡성을 최적화하여 온라인 추천 시스템에서의 실용적 사용을 크게 향상시켰습니다.'}]



### Hi-Gen: Generative Retrieval For Large-Scale Personalized E-commerce  Search (https://arxiv.org/abs/2404.15675)
- **What's New**: 생성적 검색(GR: Generative Retrieval) 기법을 이용하여 온라인 E-commerce 검색 시스템을 강화하는 방법은 최근 몇 년간 약속된 결과를 보여주고 있습니다. 본 논문에서는 Hi-Gen이라는 효율적인 계층적 인코딩-디코딩 생성 검색 방법을 제안하여 대규모 개인화된 E-commerce 검색 시스템에서의 문제를 해결하려고 합니다.

- **Technical Details**: Hi-Gen은 먼저 표현 학습 모델과 메트릭 학습(metric learning)을 설계하여 항목의 의미적 관련성과 효율성 정보를 모두 포착할 수 있는 차별화된 특성 표현을 학습합니다. 그 후, 카테고리 유도 계층적 분류 체계(category-guided hierarchical clustering)를 사용하여 문서 식별자(docIDs) 생성을 용이하게 합니다. 최종적으로, 위치 인식 손실(position-aware loss)을 설계하여 동일 위치에서 다양한 토큰 간의 의미적 및 효율성 차이를 채굴하고 위치의 중요성을 구별합니다.

- **Performance Highlights**: Hi-Gen은 실제 온라인 E-commerce 플랫폼에서 진행된 A/B 실험 지표의 개선을 통해 실용성을 입증하였습니다. 또한, Hi-Gen-I2I 및 Hi-Gen-Cluster라는 두 가지 변형을 제안하여 실시간 대규모 회상(online real-time large-scale recall)을 지원합니다. 이 방법들은 광범위한 공공 및 산업 데이터셋에서 최첨단 성능을 달성하였습니다.



### An Annotated Glossary for Data Commons, Data Meshes, and Other Data  Platforms (https://arxiv.org/abs/2404.15475)
Comments: 6 pages

- **What's New**: 이 논문은 데이터 공유와 연구 가속화를 지원하는 클라우드 기반 데이터 커먼즈(data commons), 데이터 메시(data meshes), 데이터 허브(data hubs) 등의 다양한 데이터 플랫폼에 사용되는 용어에 대한 주석이 달린 용어집을 제공합니다. 특히 데이터 메시와 데이터 패브릭(data fabric)과 같은 신개념 아키텍처의 정의를 제공하며, 이 용어들은 조직 내 데이터 관리를 개선하기 위해 등장했습니다.

- **Technical Details**: 클라우드 기반 데이터 커먼즈는 이제 십년이 넘게 사용되어 왔으며, 이 용어집은 데이터 커먼즈와 관련된 용어에 대한 광범위한 커뮤니티 합의를 반영합니다. 데이터 메시와 데이터 패브릭은 각각 데이터의 분산 관리 및 도메인별 데이터 소유권, 조직 전체의 일관된 데이터 통합 접근을 강조하는 관리 아키텍처입니다. 데이터 허브는 데이터 메시 내에서 데이터 검색 및 발견을 지원하는 데이터 플랫폼입니다.

- **Performance Highlights**: 이 용어집은 데이터 관리 및 공유 플랫폼에서 사용되는 중요 용어들에 대한 정확하고 상세한 정의를 제공함으로써, 연구자들이 다양한 데이터 플랫폼을 이해하고 효과적으로 활용할 수 있도록 돕습니다. 특히, 데이터 메시와 데이터 패브릭과 같은 신개념에 대해 설명함으로써 최신 데이터 아키텍처 트렌드에 대한 이해를 제고합니다.



### Introducing EEG Analyses to Help Personal Music Preference Prediction (https://arxiv.org/abs/2404.15753)
Comments: Accepted by CHCI 2022

- **What's New**: 이 연구에서는 음악 추천 시스템에 개인의 EEG (Electroencephalography) 신호를 도입하여 사용자의 음악 취향과 기분을 예측하는 새로운 방법을 제안하였습니다. 기존의 추천 시스템이 사용자의 클릭이나 체류 시간과 같은 암시적 피드백에 의존했다면, 이 연구는 EEG를 통해 사용자의 명시적이고 실시간 피드백을 제공함으로써 더 정확한 사용자 경험을 반영할 수 있습니다.

- **Technical Details**: 연구팀은 편안하고 휴대가 가능한 EEG 블루투스 헤드셋을 사용하여 사용자가 음악을 듣는 동안의 뇌 신호를 수집하였습니다. 이러한 EEG 신호는 노이즈 제거와 데이터 전처리를 거쳐, 음악 취향과 기분과의 상관관계를 분석하는 데 사용되었습니다. 분석 결과, EEG 신호와 사용자의 음악 취향 및 기분 사이에 유의미한 연관성이 있음을 확인하였고, 이를 통해 개인화된 음악 추천이 가능해졌습니다.

- **Performance Highlights**: EEG 기반의 취향 예측 실험에서, 기존 방법 대비 높은 성능을 보여 사용자의 명시적인 피드백을 활용하는 것의 잠재력을 입증하였습니다. EEG 신호를 이용한 평가 예측과 선호 분류에서 모두 상당한 개선을 보여주었으며, 이는 EEG가 사용자의 진정한 감정과 선호도를 반영할 수 있는 강력한 도구임을 시사합니다.




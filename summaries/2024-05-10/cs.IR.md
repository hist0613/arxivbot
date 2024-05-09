### myAURA: Personalized health library for epilepsy management via knowledge graph sparsification and visualization (https://arxiv.org/abs/2405.05229)
- **What's New**: 새롭게 개발된 myAURA 어플리케이션이 간질(Epilepsy) 환자, 보호자, 연구자들에게 치료와 자가 관리에 대한 의사결정을 돕기 위해 제작되었습니다. 이 어플리케이션은 생의학 데이터베이스(biomedical databases), 소셜 미디어(social media), 전자 건강 기록(electronic health records)과 같이 다양한 데이터 자원을 연동하는 것이 특징입니다.

- **Technical Details**: myAURA는 복수의 데이터 레이어를 링크하는 다층 지식 그래프(multi-layer knowledge graph)를 생성하기 위해, 인간 중심의 생의학 사전을 기반으로 하는 일반화 가능한 오픈 소스 방법론(open-source methodology)을 개발하였습니다. 특히, 네트워크의 중요한 연결을 추출하는 새로운 네트워크 희소화 방법론(network sparsification methodology)을 사용하여, 소셜 미디어에서 간질과 관련된 중요한 토론을 하는 디지털 집단을 추출했습니다.

- **Performance Highlights**: myAURA는 다중 데이터 소스를 통합하여 간질 정보에 더욱 접근하기 쉽게 만드는 다층 지식 그래프의 활용을 가능하게 합니다. 이는 약물 상호 작용(drug-drug interaction) 현상을 연구할 때 이 방법론의 효과를 입증하는 예로 사용되었습니다. 또한, 사용자 인터페이스(user interface)는 이해 관계자들로부터의 의견을 반영하여 설계되었으며, 초점 그룹과 그 외의 이해 관계자 입력을 바탕으로 파일럿 테스팅(pilot-testing)을 진행하였습니다.



### Graded Relevance Scoring of Written Essays with Dense Retrieva (https://arxiv.org/abs/2405.05200)
Comments:
          Accepted at SIGIR 2024

- **What's New**: 자동화된 에세이 채점(Automated Essay Scoring, AES)은 에세이의 채점 과정을 자동화하여 학생들의 글쓰기 능력 향상에 큰 도움을 줍니다. 기존의 연구들이 전체적인 에세이 채점에 집중했다면, 이번 연구는 특정 품질 특성에 대한 채점, 특히 에세이의 주제와 관련성(relevance)을 평가하는 새로운 접근 방식을 제시합니다. 이는 글이 주제에 얼마나 부합하는지를 측정하는 방법입니다.

- **Technical Details**: 본 연구에서는 밀집 검색 인코더(dense retrieval encoders)를 사용하여 에세이의 관련성을 평가합니다. 에세이는 다양한 관련성 수준에서 밀집된 표현(dense representations)을 이용하여 임베딩 공간(embeddings space) 내에서 클러스터(cluster)를 형성하고, 이 클러스터의 중심(centroids)을 활용하여 1-Nearest-Neighbor 분류를 통해 미보기 에세이의 관련성 수준을 결정합니다. 또한, 상호 학습(contrastive learning)으로 사전 훈련된 효과적인 비지도 밀집 인코더(Unsupervised Dense Encoder)인 Contriever를 활용하였습니다.

- **Performance Highlights**: 제안된 방법은 특정 작업 시나리오(task-specific scenario)에서 새로운 최고 성능(state-of-the-art)을 설정했으며, 다른 작업을 테스트하는 교차 작업 시나리오(cross-task scenario)에서도 최고 성능 모델과 비슷한 수준의 성능을 보였습니다. 특히 실용적인 적은 데이터 시나리오(few-shot scenario)에서는 레이블링 비용(labeling cost)을 크게 줄이면서도 효과성의 10%만을 희생하는 결과를 보여, 비용 효율적인 방법임을 입증했습니다.



### Dual-domain Collaborative Denoising for Social Recommendation (https://arxiv.org/abs/2405.04942)
Comments:
          14 pages, 9 figures

- **What's New**: 새로운 소셜 추천 모델인 'Dual-domain Collaborative Denoising for Social Recommendation (DCDSR)'이 제안되었습니다. 이 모델은 소셜 네트워크와 사용자-아이템 상호작용 데이터의 노이즈 문제를 해결하기 위해 두 가지 주요 모듈을 사용합니다. 첫 번째는 구조 수준 협력적 노이즈 제거 모듈이고, 두 번째는 임베딩 공간 협력적 노이즈 제거 모듈입니다.

- **Technical Details**: DCDSR의 구조 수준 협력적 노이즈 제거 모듈은 사용자-아이템 상호작용 도메인의 정보를 사용하여 소셜 네트워크 노이즈를 제거하는 데 도움을 줍니다. 이후, 노이즈가 제거된 소셜 네트워크는 상호작용 데이터 노이즈 제거를 지도합니다. 임베딩 공간 협력적 노이즈 제거 모듈은 상호 도메인(embedding cross-domain)의 노이즈 확산 문제를 대응하기 위한 대조 학습(contrastive learning)을 통해 실행됩니다. 또한, 노이즈 제거 능력을 강화하기 위해 'Anchor-InfoNCE'라는 새로운 대조 학습 전략이 도입되었습니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 DCDSR 모델을 평가한 결과, 상당한 노이즈 제거 효과가 있음이 확인되었으며, 현재 최고의 소셜 추천 방법보다 우수한 성능을 보였습니다. 따라서 DCDSR은 데이터 희소성 문제(data sparsity issue)를 완화하고 추천 시스템의 성능을 향상시키는 데 기여할 수 있습니다.



### Enabling Roll-up and Drill-down Operations in News Exploration with Knowledge Graphs for Due Diligence and Risk Managemen (https://arxiv.org/abs/2405.04929)
Comments:
          The paper was accepted by ICDE 2024

- **What's New**: NCExplorer, OLAP(Operation Lagrange Analytic Processing)-스타일 오퍼레이션을 이용해 뉴스 탐색 경험을 향상시키는 새로운 프레임워크를 소개합니다. 이 시스템은 사용자가 관련 키워드 목록을 생성하고, 연관된 뉴스 기사를 검색함으로써 보다 폭넓고 세밀한 탐색을 가능하게 합니다. 외부 지식 그래프(Knowledge Graphs, KGs)와의 통합을 통해 사실 기반 및 온톨로지 기반 구조를 활용하여 뉴스 콘텐츠의 내재된 구조와 뉘앙스를 더욱 효과적으로 파악할 수 있습니다.

- **Technical Details**: NCExplorer는 지식 그래프(Knowledge Graph)를 활용, 실제 뉴스 데이터셋에서 주제 도메인에 걸쳐 상태-아트(state-of-the-art) 뉴스 검색 방법론들보다 우수한 성능을 보이는 것으로 확인되었습니다. 뉴스 탐색을 위해 세맨틱(semantic) 롤업(roll-up) 및 드릴다운(drill-down) 오퍼레이션을 지원하며, 이는 온톨로지 및 팩트 네트워크를 활용합니다. 뉴스 기사와 관련 개념 간의 상관성을 평가하고, 롤업 및 드릴다운 작업을 용이하게 하는 효율적인 순위 매기기 체계를 개발했습니다.

- **Performance Highlights**: Amazon Mechanical Turk의 마스터 자격을 가진 평가자들에 의한 방대한 경험적 연구를 통해 NCExplorer의 우월성이 입증되었습니다. GPT 모델을 통합한 모든 비교 베이스라인과 비교해도 결과는 일관되게 나타났습니다. 연구진은 NCExplorer 구현체, 데이터셋, 평가 결과 및 전체 보고서를 공개하였으며, 이 데이터셋은 200k개의 뉴스 기사와 290만개의 엔티티 및 370만개의 개념 주석을 포함하고 있습니다.



### Full Stage Learning to Rank: A Unified Framework for Multi-Stage Systems (https://arxiv.org/abs/2405.04844)
Comments:
          Accepted by WWW 2024

- **What's New**: 기존 정보 검색(IR) 시스템 설계의 기반인 확률 순위 결정 원리(Probability Ranking Principle, PRP)에 대해 새로운 관점을 제시하였습니다. 이 연구에서는 PRP가 현대의 다단계 IR 시스템 전반에 걸쳐 무분별하게 적용되는 것이 적절하지 않다고 지적하고, 각 단계의 선택 편향(selection bias)과 사용자의 근본적인 관심을 모두 강조하는 일반화된 확률 순위 결정 원리(Generalized Probability Ranking Principle, GPRP)를 제안합니다.

- **Technical Details**: 제안된 일반화된 확률 순위 결정 원리는 다단계 시스템을 고려하여 알고리즘 프레임워크인 전체 단계 학습 정렬(Full Stage Learning to Rank)로 구현되었습니다. 이는 단계별 선택 편향을 추정하고, 하위 모듈의 선택 편향에 가장 잘 맞는 순위 모델을 학습하여 시스템 출력에서 최종 순위 목록으로 상위 순위 결과를 제공하는 것을 목표로 합니다.

- **Performance Highlights**: 개발된 전체 단계 학습 정렬 솔루션은 시뮬레이션 및 선도적인 짧은 비디오 추천 플랫폼에서의 온라인 A/B 테스트를 통해 광범위하게 평가되었습니다. 이 알고리즘은 검색(retrieval) 및 순위 매기기(ranking) 단계 모두에서 효과적임이 입증되었으며, 플랫폼에 지속적이고 중요한 성능 향상을 가져왔습니다.



### Federated Adaptation for Foundation Model-based Recommendations (https://arxiv.org/abs/2405.04840)
Comments:
          Accepted as a regular paper of IJCAI'24

- **What's New**: 이 논문에서는 기존 추천 시스템을 개선하기 위해 파운데이션 모델(foundation models)을 적용하는 새로운 패러다임을 소개합니다. 이 연구는 개인정보 보호를 유지하면서 파운데이션 모델 기반 추천 시스템을 향상시키기 위해 새로운 연방 적응 메커니즘(federated adaptation mechanism)을 제안합니다.

- **Technical Details**: 각 클라이언트는 개인 데이터를 사용하여 경량화된 개인화 어댑터(personalized adapter)를 학습합니다. 이 어댑터는 사전 훈련된 파운데이션 모델(pre-trained foundation models)과 협력하여 효율적으로 추천 서비스를 제공합니다. 이러한 데이터 지역화(data localization) 기반의 개인정보 보호는 연방 학습(federated learning) 프레임워크를 통해 실현됩니다.

- **Performance Highlights**: 네 가지 벤치마크 데이터셋(benchmark datasets)에서의 실험 결과는 본 방법의 우수한 성능을 입증합니다. 구현 코드가 제공되어 재현성을 높입니다.



### SVD-AE: Simple Autoencoders for Collaborative Filtering (https://arxiv.org/abs/2405.04746)
Comments:
          Accepted by IJCAI 2024

- **What's New**: 최근 연구에서는 계산량을 줄이기 위해 거의 훈련이 필요 없는 경량화된 방법이 제안되었습니다. 본 연구에서는 기존의 협업 필터링(Collaborative Filtering, CF) 기법의 정확성, 효율성, 그리고 강인성 간의 균형을 개선할 목적으로, SVD-AE(Singular Vector Decomposition-Autoencoder)라는 새로운 선형 자동인코더 방식을 설계했습니다. 이 방법은 반복적인 훈련 과정 없이 한 번에 계산 가능한 닫힌 형식(closed-form) 솔루션을 제공합니다.

- **Technical Details**: SVD-AE는 특이 벡터 분해(Singular Vector Decomposition, SVD)를 기반으로 한 닫힌 형태의 해를 가지며, 이는 CF를 위해 특별히 설계되었습니다. 본 논문에서는 기존의 CF 방법과 비교하여 SVD-AE의 강인성을 평가하고, 특히 잡음이 많은 평가 행렬 상에서의 성능을 탐구하였습니다. 이를 통해 트렁케이트 SVD(truncated SVD)를 기반으로 한 단순한 설계 선택이 추천의 효율성과 잡음에 대한 강인성을 강화할 수 있음을 보여줍니다.

- **Performance Highlights**: SVD-AE는 반복적 훈련 과정이 필요 없이 한 번의 계산으로 CF에 대한 해결책을 제공하기 때문에 높은 효율성을 보장합니다. 또한, 잡음이 많은 데이터에 대해서도 강한 강인성을 보여줌으로써, 기존의 CF 방법들과 비교하여 우수한 성능을 나타내었습니다. 이 연구 결과는 본 논문에서 제공된 코드와 함께 확인할 수 있습니다.



### LLMs Can Patch Up Missing Relevance Judgments in Evaluation (https://arxiv.org/abs/2405.04727)
Comments:
          5 pages, 4 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 정보 검색(IR) 벤치마크에서 판단되지 않은 문서(홀)를 자동으로 라벨링하는 새로운 프레임워크를 제안합니다. 이 방법은 TREC DL 트랙에서 실시한 다양한 시나리오의 시뮬레이션 실험을 통해 검증되었습니다.

- **Technical Details**: 연구팀은 TREC 가이드라인을 따라 세부적인 라벨을 할당하는 데 필요한 자세한 지침과 예제를 제공하여 LLM을 교육했습니다. 이러한 접근 방식은 이전의 이분법적 라벨링 연구와 달리 더 세밀한 관련성 판단을 가능하게 합니다. Vicuña-7B와 GPT-3.5 Turbo와 같은 다양한 LLMs는 골드 기준 관련성 판단과 강한 상관관계를 보여 주었습니다.

- **Performance Highlights**: 극단적인 시나리오(판단의 10%만 유지)에서 연구팀의 방법은 Vicuña-7B로 평균 0.87, GPT-3.5 Turbo로 평균 0.92의 Kendall tau 상관계수를 달성하였습니다. 이는 LLM을 사용한 자동 관련성 판단이 매우 높은 정확성을 제공할 수 있음을 시사합니다.



### Enhancing Knowledge Retrieval with Topic Modeling for Knowledge-Grounded Dialogu (https://arxiv.org/abs/2405.04713)
Comments:
          LREC-COLING 2024

- **What's New**: 이 연구에서는 지식 기반 대화 시스템에 적용되는 지식 검색의 정확도와 반응 생성을 개선하기 위해 지식 베이스에 토픽 모델링을 적용하는 새로운 접근 방식을 제시합니다. 또한, 개선된 검색 성능을 활용하여 대화 응답 생성에 ChatGPT 대형 언어 모델(Large Language Model)을 실험적으로 사용하였습니다.

- **Technical Details**: 본 논문에서는 지식 베이스 인코더를 개선하기 위해 토픽 모델링을 이용하여 지식 베이스를 클러스터로 분류하고, 각 클러스터마다 별도의 인코더를 훈련합니다. 그 후 입력 쿼리의 토픽 분포를 유사성 점수에 통합하여 상위 K개의 문서를 찾습니다. 이를 통해 RAG (Retrieval-Augmented Generation) 모델의 검색 모듈을 개선하였습니다. 또한, DPR (Dense Passage Retrieval) 방식을 토픽 기반으로 변형하여 각 토픽 클러스터에서 가장 관련성 높은 지식 패시지를 검색하고, 이를 RAG 모델의 생성 모듈에 적용하여 응답을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방식은 두 개의 데이터셋에서 지식 검색 및 반응 생성 성능을 향상시켰습니다. 특히, 관련 지식이 제공될 때 ChatGPT는 지식 기반 대화에서 보다 우수한 반응 생성기로서의 가능성을 보였습니다.



### Impact of Tone-Aware Explanations in Recommender Systems (https://arxiv.org/abs/2405.05061)
- **What's New**: 추천 시스템에서 설명 표현 방식의 효과에 중점을 둔 연구가 부족함에도 불구하고, 이 연구는 톤(예: 공식적, 유머러스)이 이러한 시스템 내 설명에 미치는 영향을 조명합니다. 온라인 사용자 연구를 통해 영화, 호텔, 가정용품 등 다양한 도메인에서 설명의 톤이 인식된 효과에 미치는 영향을 조사하였습니다. 이는 톤이 사용자 경험을 향상시킬 수 있는 중요한 요소임을 시사합니다.

- **Technical Details**: 연구팀은 대규모 언어 모델(LLM)을 사용하여 다양한 톤의 가상 아이템 설명을 생성한 데이터셋을 개발했습니다. 설명의 톤이 사용자의 인식에 미치는 영향을 조사하기 위해, 사용자들에게 서로 다른 톤의 설명을 제시하고 8가지 설명 효과(투명성 (transparency), 살필 수 있는 정도 (scrutability), 신뢰성 (trust), 효능 (effectiveness), 설득력 (persuasiveness), 효율성 (efficiency), 만족도 (satisfaction), 매력 (appeal))에 대한 페어 비교 평가를 요청하였습니다.

- **Performance Highlights**: 호텔 도메인에서의 톤의 영향은 거의 모든 메트릭에서 유의미한 차이가 관찰되었으며, 영화 및 제품 도메인에서는 제한적이었습니다. 사용자 특성(예: 나이, Big Five 성격 특성)에 따라 톤의 영향이 다르게 나타나, 사용자 프로필을 기반으로 톤을 개인화하면 추천 시스템에 대한 사용자의 인식을 변경할 수 있습니다.



### Hypergraph-enhanced Dual Semi-supervised Graph Classification (https://arxiv.org/abs/2405.04773)
Comments:
          Accepted by Proceedings of the 41st International Conference on Machine Learning (ICML 2024)

- **What's New**: 본 논문에서는 제한된 레이블이 있는 그래프와 다수의 레이블이 없는 그래프가 있는 상황에서 그래프 분류를 정확하게 예측하는 반지도 학습(semi-supervised learning) 그래프 분류를 연구합니다. GNN(Graph Neural Networks)의 한계를 극복하기 위해 고안된 하이퍼그래프 강화 이중 구조(dual framework)인 HEAL을 제안하여 노드 간 고차 의존성을 효과적으로 모델링하고 주석이 없는 그래프를 활용하여 반지도 학습 방식을 개선합니다.

- **Technical Details**: HEAL 프레임워크는 하이퍼그래프(hypergraph)와 라인 그래프(line graph)를 동시에 활용하여 그래프 의미론을 효과적으로 파악합니다. 하이퍼그래프 구조 학습을 통해 노드 간 복잡한 의존성을 적응적으로 학습하고, 이를 바탕으로 라인 그래프를 사용하여 하이퍼에지 간 상호작용을 포착합니다. 또한, 두 분기 간의 지식 전달을 촉진하기 위해 관계 일관성 학습(relational consistency learning)을 개발하였습니다.

- **Performance Highlights**: 실제 세계의 그래프 데이터셋에 대한 광범위한 실험을 통해 HEAL이 기존의 최신(state-of-the-art) 방법들에 비해 우수한 성능을 보임을 확인하였습니다. 이는 HEAL이 고차 의존성 모델링 및 라벨이 없는 그래프의 이점을 잘 활용하고 있음을 의미합니다.



### Cryptanalysis of the SIMON Cypher Using Neo4j (https://arxiv.org/abs/2405.04735)
Comments:
          10 pages, 10 figures, 2 algorithms, accepted by the 4th International Conference on Electrical, Computer and Energy Technologies (ICECET) to be presented in July 2024

- **What's New**: 이 논문에서는 SIMON 경량 암호화 알고리즘(Lightweight Encryption Algorithm, LEA)의 차분 보안 분석을 향상시키기 위해 지식 그래프(Knowledge Graph)를 활용하는 새로운 방법을 제시하고 있습니다. 지식 그래프를 사용하여 PDDT(Partial Difference Distribution Table) 내의 차분 간의 복잡한 관계를 식별하고 최적의 경로를 찾는 방법론이 소개됩니다.

- **Technical Details**: 이 연구는 SIMON 암호화 알고리즘에서 차분 분포 테이블(Difference Distribution Table, DDT)의 부분적 활용, 즉 PDDT의 구조를 활용하여 차분 분석(Differential Cryptanalysis, DC)의 효율성을 높이는 데 초점을 맞추고 있습니다. 특히, Neo4j를 사용하여 구현된 지식 그래프를 통해 데이터 내 패턴과 관계를 파악하고, 이를 바탕으로 암호화 분석에서 보다 효율적인 경로를 식별합니다.

- **Performance Highlights**: 지식 그래프를 사용함으로써, 휴리스틱 방법(Heuristic Methods)에 의존하는 기존의 차분 분석 방법론 대비 결과의 재현성과 정확성을 향상시킬 수 있습니다. 또한, 지식 그래프 기반 접근 방식은 데이터 내 숨겨진 패턴을 효과적으로 식별하고, 이를 통해 SIMON 암호의 보안 분석에 있어 더욱 심층적인 인사이트를 제공할 수 있습니다.



### Robust Implementation of Retrieval-Augmented Generation on Edge-based Computing-in-Memory Architectures (https://arxiv.org/abs/2405.04700)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG)을 가속화하기 위한 새로운 Computing-in-Memory (CiM) 아키텍처 프레임워크를 제안합니다. 이는 RAG의 성능과 확장성을 향상시키기 위해 널리 사용되는 대형 언어 모델(Large Language Models, LLMs)에 적용됩니다. 이는 첫 번째로 CiM을 사용하여 RAG를 가속화하는 연구로 알려져 있습니다.

- **Technical Details**: 제안된 프레임워크인 Robust CiM-backed RAG (RoCR)는 새로운 대조학습(contrastive learning)-기반 훈련 방법과 노이즈 인식 훈련을 활용합니다. 이를 통해 메모리 내에서 직접 계산을 수행함으로써 데이터 전송 비용을 절감하고 효율적으로 프로필 데이터를 검색할 수 있습니다. 또한, 이 논문은 유연한 노이즈 인식 대조학습 방법을 도입하여 모델이 압축 및 양자화로 인해 발생할 수 있는 잡음에 강인한 문장 임베딩을 생성하도록 합니다.

- **Performance Highlights**: RoCR은 기존 RAG 모델 대비 프로필 데이터 검색 속도를 크게 향상시키며, 사용자 데이터의 지속적인 증가에 따라 모델의 확장성을 유지합니다. 또한, 이 프레임워크는 대용량 데이터 처리를 요구하는 에지 장치(Edge Devices)에 부담을 주지 않고 대규모 언어 모델을 효과적으로 운용할 수 있도록 지원합니다.



### Multi-Margin Loss: Proposal and Application in Recommender Systems (https://arxiv.org/abs/2405.04614)
- **What's New**: 이 논문에서는 간단하면서도 효과적인 손실 함수인 Multi-Margin Loss (MML)을 제안합니다. MML은 여러 개의 Margin과 다양한 가중치를 부여하여 하드 네거티브(hard-negative) 뿐만 아니라 트리비얼하지 않은(none-trivial) 부정적 샘플도 효율적으로 활용할 수 있습니다. 본 연구는 제한된 리소스에서도 복잡한 메소드보다 우수한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: MML은 각기 다른 'Hardness'의 네거티브 샘플에 대해 다양한 마진(margin)과 가중치를 적용합니다. 이로써 모델은 가장 어려운 네거티브 샘플만이 아니라 세미-하드(semi-hard) 및 세미-이지(semi-easy) 샘플을 고려하여 더 일반화된(generalize) 성능을 발휘할 수 있습니다. 본 연구에서는 유명한 추천 시스템 데이터셋에서 MML을 사용하여 전통적인 대조적 손실(contrastive loss) 함수와 비교 평가를 진행하였습니다.

- **Performance Highlights**: MML은 작은 샘플 크기(예: 10 또는 100 개의 네거티브 샘플)에서 기본 대조적 손실 함수 대비 최대 20%까지 성능 향상을 보여주었습니다. 또한, 800개의 네거티브 샘플을 사용할 경우, 기존의 최신 대조적 손실 방식과 동등한 수준의 성능을 유지하면서, 네거티브 샘플 사용의 효율성을 극대화하였습니다.




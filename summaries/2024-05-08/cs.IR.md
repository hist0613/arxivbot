### Dataset and Models for Item Recommendation Using Multi-Modal User Interactions (https://arxiv.org/abs/2405.04246)
- **What's New**: 이번 연구에서는 멀티모달 사용자 상호작용(multi-modal user interactions), 즉 클릭이나 음성과 같은 다양한 방식으로 시스템과 상호작용하는 사용자들의 데이터를 활용하는 새로운 추천 시스템을 개발하였습니다. 이는 기존의 아이템 중심의 멀티모달 데이터만을 다루던 이전 연구들과 차별화된 점입니다.

- **Technical Details**: 연구팀은 실제 보험 상품 데이터를 기반으로 한 멀티모달 사용자 상호작용 데이터셋을 생성하고 공개하였으며, 이를 바탕으로 여러 추천 접근법을 실험하고 평가하였습니다. 주요 기술적 기여로는 누락된 모달리티(missing modalities)를 처리하기 위해 공통 피처 공간(common feature space)으로 사용자 상호작용을 매핑(mapping)하는 방법을 제안함으로써, 다양한 채널을 통한 상호작용을 통합할 수 있는 접근법을 제시했습니다.

- **Performance Highlights**: 실험 결과, 제안된 멀티모달 추천 시스템은 기존 방법들에 비해 각기 다른 모달리티 간의 중요한 상호작용을 포착하고, 더 자주 발생하는 모달리티에서 얻은 정보를 활용하여 덜 빈번한 모달리티의 추천 성능을 향상시키는 데 성공하였습니다. 이는 멀티모달 사용자 상호작용을 효과적으로 활용할 수 있다는 중요한 시사점을 제공합니다.



### Masked Graph Transformer for Large-Scale Recommendation (https://arxiv.org/abs/2405.04028)
- **What's New**: 새롭게 제안된 MGFormer (Masked Graph Transformer)는 그래프 구조화된 데이터를 학습하는 데 있어, 모든 노드 간의 상호 작용을 선형 복잡도로 파악할 수 있는 능력을 갖추고 있습니다. 이는 기존의 Graph Transformers가 대규모 그래프에서 주로 겪는 이차 공간 및 시간 복잡도 문제를 효과적으로 해결합니다.

- **Technical Details**: MGFormer의 핵심 구성 요소는 다음과 같습니다. 1) 구조적 인코딩(Structural encoding)을 사용하여 이분 그래프의 토폴로지로부터 각 노드의 위치를 파악합니다. 2) 커널화된 주의력 층(Kernelized attention layer), 자체 주의력(self-attention)을 선형 닷 제품으로 근사화하여 선형 복잡도를 실현합니다. 3) 학습 가능한 사인파도 형식의 정도 마스크를 이용한 재가중치 주의 메커니즘으로, 더 중요한 토큰에 더 많은 가중치를 할당합니다.

- **Performance Highlights**: 실험 결과, MGFormer는 다양한 벤치마크에서 경쟁력 있는 성능을 보였으며, GNNs와 비교할 때 비슷한 복잡도를 유지하면서 상당한 성능 향상을 달성하였습니다. 이는 특히 대규모 추천 시스템에서의 활용 가능성을 시사합니다.



### Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application (https://arxiv.org/abs/2405.03988)
Comments:
          11 pages, 6 figures

- **What's New**: 이 연구에서는 현대 추천 시스템의 한계를 극복하고자 Large Language Models(LLMs)을 사용하여 제품의 텍스트 설명에서 추출한 의미론적 정보를 활용하는 새로운 방법을 제안합니다. 'LEARN'(Llm-driven knowlEdge Adaptive RecommeNdation) 프레임워크는 협력적 지식(collaborative knowledge)과 개방형 세계 지식(open-world knowledge)을 통합하여, 특히 콜드 스타트(cold-start) 상황과 장기간 유저(long-tail user) 추천에서 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: LEARN 프레임워크는 LLM을 항목 인코더(item encoders)로 사용하고 LLM 매개 변수를 고정하여 개방형 세계 지식을 보존하고 카타스트로픽 포게팅(catastrophic forgetting)을 방지합니다. 이를 위해 별도의 트윈 타워 구조(twin-tower structure)를 설계하여 개방형 세계와 협력적 도메인 간의 격차를 줄입니다. 이 구조는 실용적인 산업 응용을 위해 추천 작업(recommendation task)에 의해 감독됩니다.

- **Performance Highlights**: 대규모 산업 데이터셋(industrial dataset)을 이용한 오프라인 실험과 A/B 테스트에 대한 온라인 실험을 통해 LEARN 방법의 효과를 입증했습니다. 특히, 콜드 스타트 및 장기 사용자 추천 성능에서 두드러진 향상을 보였습니다.



### Contextualization with SPLADE for High Recall Retrieva (https://arxiv.org/abs/2405.03972)
Comments:
          5 pages, 1 figure, accepted at SIGIR 2024 as short paper

- **What's New**: 이 연구는 높은 회수 검색(High Recall Retrieval, HRR)의 최적화를 위해, 문맥화된 희소 벡터로 문서를 변환하는 SPLADE 모델을 사용하여 변형하였습니다. 이는 선형 모델의 효율성과 사전 훈련된 언어 모델(Pretrained Language Models, PLMs)의 문맥화 능력을 조합한 새로운 접근 방식입니다. 특히, 이 연구는 한 단계 검토 워크플로(One-phase review workflow)에서 목표 회수율(Target recall) 80%를 달성하면서 리뷰 비용을 10% 및 18% 줄였다는 점에서 주목할 만합니다.

- **Technical Details**: 기존의 선형 모델과 달리, 이 연구는 문맥화된 특징을 추출하기 위해 PLMs를 사용하는데, 이는 SPLADE 모델을 응용하여 문서를 문맥화 희소 벡터(Contextualized Sparse Vectors)로 인코딩하고, 이를 선형 모델에서 사용합니다. 여러 HRR 평가 컬렉션에서의 실험을 통해 이 접근 방식의 효과를 검증하였습니다.

- **Performance Highlights**: 제안된 모델은 기존 선형 모델 대비 비용을 크게 절감하면서 회수율 목표를 빠르고 효율적으로 달성합니다. 특히, 한 단계 검토 워크플로에서는 평균 리뷰 비용을 10% 및 18% 감소시키는 성과를 보였습니다.



### The Fault in Our Recommendations: On the Perils of Optimizing the Measurab (https://arxiv.org/abs/2405.03948)
- **What's New**: 이 논문은 추천 시스템(recommendation systems)이 사용자의 실제 만족도(utility)를 어떻게 개선할 수 있는지에 대해 탐구합니다. 특히, 인기 있는 콘텐츠(popular content)와 니치 콘텐츠(niche content)를 혼합한 초기 추천이 사용자 만족도와 참여도(engagement)를 동시에 최적화할 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 반복적 사용자 소비 모델(repeated user consumption model)을 도입하여, 사용자가 외부 옵션과 추천된 아이템 세트에서 최선의 선택을 반복적으로 고르는 상황을 모델링합니다. 이 모델은 사용자의 다양성을 고려하며, 대부분이 인기 콘텐츠를 선호하지만 소수는 니치 콘텐츠를 선호하는 특성을 포함합니다. 플랫폼은 처음에는 개인의 선호를 모르지만 시간이 지남에 따라 사용자의 선택을 관찰함으로써 배울 수 있습니다.

- **Performance Highlights**: 이 연구는 사용자의 기대 만족도(expected utility)와 참여도 사이에 뚜렷한 불일치가 있음을 밝혀냈습니다. 니치 콘텐츠를 추천하지 않는 참여 최적화 정책(engagement maximizing policy)은 상대적으로 낮은 사용자 만족도를 초래합니다. 반면, PEAR 알고리즘은 초기에 인기 있는 콘텐츠 및 니치 콘텐츠의 혼합을 추천함으로써 훨씬 높은 사용자 만족도와 높은 참여도를 동시에 달성할 수 있습니다. 이는 플랫폼이 전방을 바라보면서(forward-looking) 점점 더 사용자 만족도를 극대화하는 방향으로 진화할 수 있음을 시사합니다.



### BBK: a simpler, faster algorithm for enumerating maximal bicliques in large sparse bipartite graphs (https://arxiv.org/abs/2405.04428)
Comments:
          21 pages, 4 figures, 3 tables

- **What's New**: 이 연구에서는 이분 그래프(bipartite graph)에서 최대 완전 이분 그래프(maximal bicliques)를 전수 조사하는 새로운 알고리즘이 소개되었습니다. 이 알고리즘은 이분 그래프의 특성을 고려하여 조정된 Bron-Kerbosch 알고리즘을 바탕으로 만들어진 BBK (Bipartite Bron-Kerbosch)로, 기존 알고리즘보다 빠른 속도를 자랑합니다. 이 알고리즘이 실세계 대규모 데이터셋에서의 효과적이고 효율적인 수행을 가능하게 합니다.

- **Technical Details**: BBK 알고리즘은 이분 그래프 내의 최대 완전 이분 그래프를 열거(enumerate)하는 방법을 제공하며, C++로 구현되어 공개적으로 접근 가능합니다. 이 알고리즘은 입력(input) 특성과 출력(output) 특성에 따라 복잡성(complexity) 공식을 설정하여 이론적 분석이 이루어졌습니다. 또한, BBK 알고리즘은 그래프의 정점(vertices) 처리 순서와 열거 시작점을 선택하는 두 유형의 정점 중 하나에 따라 계산 시간에 영향을 미칠 수 있습니다.

- **Performance Highlights**: 대규모 실세계 데이터셋에서의 실험을 통해 BBK 알고리즘이 기존의 최신 상태 기술(state-of-the-art) 알고리즘보다 실제 실행 시간이 짧다는 것을 검증하였습니다. 이 알고리즘은 이분 그래프 내에서 데이터 마이닝(data mining), 특히 자주 발생하는 아이템 세트(mining frequent itemsets) 및 대규모 데이터베이스에서의 연관 규칙 찾기(applications in finding association rules in large databases)에 유용하게 적용될 수 있습니다.



### Adaptive Retrieval and Scalable Indexing for k-NN Search with Cross-Encoders (https://arxiv.org/abs/2405.03651)
Comments:
          ICLR 2024

- **What's New**: 이 논문에서는 쿼리-항목(query-item) 쌍을 공동으로 인코딩하여 유사성을 계산하는 교차 인코더(Cross-encoder, CE) 모델이 이중 인코더(dual-encoders, DE)를 기반으로 하는 임베딩 방식보다 쿼리-항목 관련성(query-item relevance) 예측에서 더 나은 성능을 보이는 것에 주목합니다. 기존의 방법들은 DE 또는 CUR 행렬 분해를 이용하여 벡터 임베딩 공간을 형성해 교차 인코더 유사성을 근사화 시키지만, CUR 기반 방법은 CE 호출을 과도하게 요구하여 실제 배포에는 실용적이지 않습니다. 이러한 단점을 극복하기 위해, 저자들은 희소 행렬 분해를 기반으로 한 새로운 방법을 제안하여 라텐트 쿼리 및 항목 임베딩(latent query and item embeddings)을 효율적으로 계산하고, 교차 인코더 점수를 근사화하여 k-NN 검색을 수행합니다.

- **Technical Details**: 저자들은 훈련 쿼리 집합에 대한 쿼리-항목 CE 점수를 포함하는 희소 행렬을 분해하여 항목 임베딩을 오프라인(offline)에서 계산합니다. 이 방법은 CUR 기반 방법에 비해 CE 호출의 일부만을 요구하면서 고품질의 근사치를 생성합니다. 또한, 이 방법은 이중 인코더를 이용하여 임베딩 공간을 초기화하면서, 이중 인코더의 계산 및 자원 집약적인 파인 튜닝(finetuning)을 피할 수 있게 합니다. 테스트 시, 항목 임베딩은 고정되며, 검색은 a) 추출된 항목의 CE 점수를 근사화하는 오류를 최소화하여 테스트 쿼리 임베딩을 추정하고, b) 업데이트된 테스트 쿼리 임베딩을 사용하여 더 많은 항목을 검색하는 두 단계로 번갈아가며 이뤄집니다.

- **Performance Highlights**: 이 k-NN 검색 방법은 DE 기반 접근법보다 최대 5% (k=1에서) 및 54% (k=100에서) 향상된 리콜(recall)을 달성합니다. 또한, CUR 기반 방법에 대해 최대 100배, DE 증류 방법에 대해 5배의 속도 향상을 이루면서, 기존 방법들과 비교해 리콜 또한 매치하거나 향상시킵니다.



### Doing Personal LAPS: LLM-Augmented Dialogue Construction for Personalized Multi-Session Conversational Search (https://arxiv.org/abs/2405.03480)
Comments:
          Accepted at SIGIR 2024 (Full Paper)

- **What's New**: 새로운 대화형 에이전트는 사용자 개인에 맞춘 정보를 제공할 수 있는 능력을 보여주지만, 다양한 세션과 실제 사용자 선호를 반영하는 대규모 대화 데이터셋의 부족이 커다란 도전 과제입니다. 이전 방법들은 오즈의 마법사(Wizard-of-Oz) 설정에서 전문가들에 의존하는데, 이는 특히 개인화된 작업에서 확장하기 어렵습니다. 이를 해결하기 위해 LAPS(Large language models with a single human worker) 방법론을 도입하여 대규모, 인간이 작성한, 다중 세션 및 다중 도메인 대화를 수집합니다. 이 방법은 생성 과정을 가속화하고 품질을 향상시키는 것으로 입증되었습니다.

- **Technical Details**: LAPS는 대규모 언어 모델(LLMs)을 사용하여 단일 인간 작업자를 지도하여 개인화된 대화를 생성합니다. 이 데이터셋은 사전 기호 추출 및 개인화된 반응 생성 훈련에 적합합니다. LAPS가 생성한 대화는 완전히 인공적인 방법들과 대비되어 자연스럽고 다양한 전문가가 만든 것만큼 효과적입니다.

- **Performance Highlights**: LAPS를 사용하여 추출된 기호(preferences)를 사용한 응답이 단순한 대화 기록보다 사용자의 실제 선호와 더 일치하는 결과를 보여, 추출된 기호를 사용하는 것의 가치를 강조합니다. LAPS는 기존 방법보다 더 효율적이고 효과적으로 현실적인 개인화 대화 데이터를 생성하는 새로운 방법을 제시합니다.



### Improving (Re-)Usability of Musical Datasets: An Overview of the DOREMUS Projec (https://arxiv.org/abs/2405.03382)
- **What's New**: DOREMUS는 세 개의 프랑스 기관의 데이터를 연결하고 탐색하기 위한 새로운 도구를 구축하여 음악의 더 나은 설명을 목표로 합니다. 이 연구는 FRBRoo에 기반한 데이터 모델을 개요하고, 링크드 데이터(linked data) 기술을 사용한 변환 및 연결 과정을 설명하며, 웹 사용자의 요구에 맞춰 데이터를 사용할 수 있는 프로토타입을 제시합니다.

- **Technical Details**: DOREMUS 프로젝트는 FRBRoo를 기반으로 한 데이터 모델을 채택하여 음악작품에 대한 다양한 정보의 효과적 분류와 조직을 지원합니다. 또한, 연결된 데이터 기술과 시맨틱 웹(semantic web) 프린시플을 활용하여 데이터를 연결하고 변환하는 과정이 포함되어 있습니다.

- **Performance Highlights**: 이 프로토타입은 사용자가 음악 데이터를 보다 쉽게 탐색하고 접근할 수 있도록 설계되었으며, 효과적 데이터 통합을 통해 사용자 경험을 개선하는 데 중점을 두고 있습니다.



### TF4CTR: Twin Focus Framework for CTR Prediction via Adaptive Sample Differentiation (https://arxiv.org/abs/2405.03167)
- **What's New**: 이 논문에서는 클릭률 (CTR, Click-through Rate) 예측의 정확성을 향상시키기 위해 새로운 CTR 예측 프레임워크인 Twin Focus Framework for CTR (TF4CTR)을 소개합니다. TF4CTR은 샘플 선택 임베딩 모듈 (SSEM, Sample Selection Embedding Module), 동적 융합 모듈 (DFM, Dynamic Fusion Module), 그리고 트윈 포커스 (TF) 손실을 통합하여, 다양한 복잡성의 샘플을 효과적으로 처리하고, 더 정확한 예측을 제공합니다.

- **Technical Details**: SSEM은 모델의 하단에서 샘플을 구별하고 각 샘플에 가장 적합한 인코더를 할당하는 기능을 수행합니다. TF Loss는 간단하고 복잡한 인코더 모두에 맞춤형 감독 신호를 제공하여, 샘플의 편향 없는 학습을 돕습니다. DFM은 인코더에서 캡처된 피처 상호 작용 정보를 동적으로 융합하여, 입력 데이터의 다양성에 대응합니다.

- **Performance Highlights**: 실험은 다섯 가지 실세계 데이터셋에서 수행되었으며, TF4CTR 프레임워크가 기존의 다양한 베이스라인 모델들과 비교하여 우수한 성능을 보였습니다. 이 프레임워크는 모델에 구속되지 않고 적용가능하며, 인코더의 정보 캡처 능력과 최종 예측 정확도를 개선하였습니다.



### Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models (https://arxiv.org/abs/2405.02503)
Comments:
          10 pages, 10 figures, accepted at SIGIR 2024 as perspective paper

- **What's New**: 인공지능(AI) 뉴스레터에서 전해드리는 새로운 연구로, 신경망을 이용한 랭킹 모델의 작동 원리와 의사 결정 과정에 대한 깊은 이해를 도모하기 위한 방법론이 소개되었습니다. 이 연구에서는 인과적 개입(causal interventions)을 이용하여 랭크 결정 메커니즘을 '역설계(reverse engineering)'하는 새로운 접근 방식을 제시하고 있습니다. 특히, 이 방법은 검색 엔진(search engines) 및 추천 시스템(recommendation systems) 내에서 문서의 관련성을 계산할 때 고려되는 복잡한 요소들을 구체적으로 분석할 수 있게 해줍니다.

- **Technical Details**: 연구팀은 신경망 랭킹 모델(neural ranking models)의 내부 결정 과정을 이해하기 위해 TFC1(Term Frequency Component 1) 공리를 만족하는 구성 요소를 분리할 수 있는 '메커니즘 해석 가능성(mechanistic interpretability)' 방법을 사용했습니다. 이를 위해 활성화 패치(activation patching)라는 새로운 기술을 도입하여 모델이 용어 빈도(term frequencies)를 추적할 수 있는지를 실험적으로 조사하였습니다. 이 방법은 특히 어텐션 헤드(attention heads)를 통해 모델의 행동을 국지화하고, 이러한 헤드가 어떻게 상호 작용하여 문서의 관련성을 평가하는지를 파악하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, TAS-B라는 사전 훈련된 DistilBERT 기반 인코더에서 특정 어텐션 헤드가 용어 빈도 식별자로 작동함을 확인하였습니다. 이러한 발견은 신경망 모델이 TFC1 공리를 얼마나 잘 준수하고 있는지를 판단할 수 있는 근거를 제공합니다. 또한, 이 연구는 정보 검색(IR, Information Retrieval)의 설명 가능한 AI(Explainable AI) 연구에 새로운 방향을 제시하며, 모델의 안전한 배치를 보장하는 데 도움이 될 수 있습니다.



### Characterizing the Dilemma of Performance and Index Size in Billion-Scale Vector Search and Breaking It with Second-Tier Memory (https://arxiv.org/abs/2405.03267)
- **What's New**: 이 논문은 SSD 기반의 그래프 및 클러스터 인덱스에 대한 성능과 인덱스 크기 사이의 트레이드 오프를 최초로 분석합니다. 특히, SSD가 아닌 2차 메모리(second-tier memory)를 사용하여 벡터 인덱스(vector indexes)의 성능을 향상시키는 새로운 접근 방식을 제안합니다. 이 접근 방식은 RDMA나 CXL 같은 고속 인터커넥트를 통해 연결된 원격 DRAM/NVM(memory)을 활용합니다.

- **Technical Details**: 논문은 기존의 SSD 기반 인덱싱 방식에서 발생하는 작은 무작위 읽기 요구사항과 SSD의 대용량 읽기 요구 사이의 불일치를 지적하며, 이를 해결하기 위해 2차 메모리를 이용합니다. 이 메모리는 더 작은 액세스 단위를 지원하여 벡터 인덱스의 워크로드 패턴과 일치합니다. 제안된 시스템은 그래프 인덱스(graph index)와 클러스터 인덱스(cluster index)를 2차 메모리에 최적화하여 구현하고, 이를 통해 인덱스 크기를 크게 줄이면서도 최적의 성능을 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: SSD 대비하여 인덱스 크기를 최대 44%까지 줄이면서도, 그래프 인덱스는 최적의 성능을, 클러스터 인덱스는 40%의 인덱스 크기 증폭으로 향상된 성능을 달성합니다. 2차 메모리는 또한 작은 무작위 읽기에 강하면서도 메모리 대역폭 활용에 있어서도 SSD보다 효율적입니다.



### iSEARLE: Improving Textual Inversion for Zero-Shot Composed Image Retrieva (https://arxiv.org/abs/2405.02951)
Comments:
          Extended version of the ICCV2023 paper arXiv:2303.15247

- **What's New**: 이번 연구에서는 레이블링된 학습 데이터셋이 필요 없는 제로샷 합성 이미지 검색(ZS-CIR)을 새로운 연구 과제로 소개합니다. 이는 비용이 많이 드는 수동 레이블링 과정을 없애고 CIR 기법의 적용 범위를 확대합니다. 또한, 공통 객체 중심의 사례로 구성된, 복수의 정답 이미지와 의미론적 분류를 포함하는 새로운 CIRCO 데이터셋을 제공합니다.

- **Technical Details**: 제안된 iSEARLE 방법은 CLIP (Contrastive Language–Image Pre-training) 모델을 기반으로, 참조 이미지를 CLIP 토큰 임베딩 공간의 pseudo-word token(의사 단어 토큰)으로 매핑하고, 이를 상대적 캡션과 결합하는 방식을 사용합니다. 이 과정을 'textual inversion'(텍스트 역변환)이라고 하며, 최적화 기반 텍스트 역변환(Optimization-based Textual Inversion, OTI)과 지식 증류(distillation)를 포함하는 두 단계의 사전 훈련을 거칩니다.

- **Performance Highlights**: iSEARLE은 FashionIQ, CIRR 및 새롭게 제안된 CIRCO 데이터셋에서 최고 수준의 성능을 보여주었습니다. 추가로, 도메인 변환과 객체 구성 설정에 대한 실험에서도 우수한 일반화 능력을 입증했습니다.



### MedPromptExtract (Medical Data Extraction Tool): Anonymization and Hi-fidelity Automated data extraction using NLP and prompt engineering (https://arxiv.org/abs/2405.02664)
- **What's New**: MedPromptExtract는 의료 기록의 디지털화를 실현하기 위한 새로운 도구로, 의료 기록을 구조화된 데이터로 자동 변환합니다. MedPromptExtract는 반자동 학습(semi-supervised learning), 대규모 언어 모델(large language models, LLM), 자연어 처리(natural language processing, NLP), 프롬프트 엔지니어링(prompt engineering)을 사용하여 비구조화된 의료 기록에서 관련 의료 정보를 추출합니다. 이를 통해 더 나은 치료 계획과 연구에 필요한 데이터를 효율적으로 활용할 수 있습니다.

- **Technical Details**: MedPromptExtract는 EIGEN(Expert-Informed Joint Learning aGgrEgatioN) 모델을 사용하여 익명화를 진행하고, Document Text Recognition (DocTR) 기술을 이용해 PDF 파일을 JSON 파일로 변환합니다. 또한, LayoutLM이라는 사전 훈련된 딥 뉴럴 네트워크 모델을 이용하여 문서 이미지에서 정보를 추출합니다. 이 시스템은 학습 데이터에 대한 노이즈가 있는 레이블을 생성하는 라벨링 함수(Labeling Functions, LFs)를 사용하여 어노테이션 비용을 줄이는 한편, 추출된 정보를 구조화된 데이터프레임(DataFrame)으로 조직합니다.

- **Performance Highlights**: 비록 초기 데이터 세트에서 4개의 요약이 제외되었지만, 최종적으로 852개의 요약에서 데이터가 추출되었으며, 48개는 인간 검증을 위해 사용되었습니다. 아노테이션 검증을 위한 인간 어노테이터 간의 일치도(Kappa coefficient)는 매우 높았으며, 이는 모델의 높은 믿음성을 나타냅니다. 또한, MedPromptExtract는 병원의 전자 건강 기록(EHR) 시스템과 통합되어 더 많은 하류 응용 프로그램(downstream applications)에 쉽게 적용될 예정입니다.



### Accelerating Medical Knowledge Discovery through Automated Knowledge Graph Generation and Enrichmen (https://arxiv.org/abs/2405.02321)
Comments:
          18 pages, 5 figures

- **What's New**: 이 연구에서는 의료 지식 그래프(Medical Knowledge Graph, M-KGA) 자동화라는 새로운 접근 방식을 제안합니다. M-KGA는 사용자가 제공하는 의료 개념을 BioPortal 온톨로지를 사용하여 시맨틱하게 풍부하게 하고, 사전 훈련된 임베딩을 통합함으로써 지식 그래프의 완성도를 향상시킵니다.

- **Technical Details**: M-KGA는 두 가지 방법론을 소개합니다: 클러스터 기반 접근 방식과 노드 기반 접근 방식입니다. 이러한 방법론들은 지식 그래프 내에서 숨겨진 연결을 찾아내는 데 사용됩니다. BioPortal 온톨로지를 통해 의료 개념을 풍부하게 하고, 사전 훈련된 임베딩(pre-trained embeddings)을 통합하여 지식 그래프의 정보를 확장합니다.

- **Performance Highlights**: 이 프레임워크는 전자 건강 기록(Electronic Health Records, EHRs)에서 자주 발생하는 100개의 의료 개념을 사용하여 엄격한 테스트를 거쳤으며, 기존 지식 그래프 자동화 기술의 한계를 해결할 가능성을 보여주는 유망한 결과를 제시했습니다.



### FairEvalLLM. A Comprehensive Framework for Benchmarking Fairness in Large Language Model Recommender Systems (https://arxiv.org/abs/2405.02219)
- **What's New**: 이 논문은 큰 언어 모델(Large Language Models, LLM)을 이용한 추천 시스템인 RecLLMs의 공정성을 평가하기 위한 프레임워크를 제시합니다. 이 연구는 사용자 속성에 대한 민감도, 내재된 공정성(intrinsic fairness), 그리고 기본 이익을 바탕으로 한 공정성 논의를 포함하여 다양한 공정성 차원을 아우르는 통일된 접근법의 필요성을 해결합니다. 또한, 본 프레임워크는 가상 평가(counterfactual evaluations)를 도입하고 다양한 사용자 그룹 고려사항을 통합함으로써 RecLLMs에 대한 공정성 평가 논의를 강화합니다.

- **Technical Details**: 논문의 주요 기여도는 LLM 기반 추천에서 공정성 평가를 위한 견고한 프레임워크 개발과 인구 통계 데이터, 역사적 사용자 선호도 및 최근 상호작용에서 '정보적 사용자 프로필'(informative user profiles)을 생성하기 위한 구조화된 방법을 제시하는 것입니다. 특히 시간에 따라 변화하는 시나리오에서 이러한 프로필이 개인화 개선에 필수적임을 주장합니다. 두 데이터셋, LastFM-1K와 ML-1M에 대한 실용적 응용을 통해 이 프레임워크의 유용성을 입증하고, 각 데이터셋에서 80명의 사용자를 대상으로 50가지 시나리오에서 다양한 프롬프트 생성 및 인컨텍스트 학습(in-context learning)을 테스트하여 평가했습니다. 이로 인해 총 4000개의 추천이 생성되었습니다.

- **Performance Highlights**: 연구 결과에 따르면 민감한 속성을 다루는 시나리오에서의 불공정 문제는 유의미하지 않았으나, 민감하지 않은 내재된 공정성(intrinsic fairness) 측면에서는 인구통계학적 그룹 간 상당한 불공정이 남아 있음을 발견했습니다. 이는 RecLLMs에서 공정성을 향상시키기 위해 추가적인 조정과 연구가 필요함을 시사합니다.



### How to Diversify any Personalized Recommender? A User-centric Pre-processing approach (https://arxiv.org/abs/2405.02156)
- **What's New**: 이 논문에서는 추천 성능을 유지하면서 Top-N 추천의 다양성을 개선하는 새로운 접근 방식을 소개합니다. 사용자 중심의 사전 처리(pre-processing) 전략을 사용하여 사용자가 다양한 콘텐츠 카테고리와 주제에 노출될 수 있도록 하며, 이를 통해 사용자 프로필에서 상호작용의 일정 비율을 선택적으로 추가하거나 제거함으로써 개인화를 실현합니다. 이러한 사전 처리 기술은 유연성을 제공하며, 어떤 추천 시스템 아키텍처(recommender architecture)에도 쉽게 통합될 수 있습니다.

- **Technical Details**: 이 연구에서는 사용자 프로필을 변경하여 추천 시스템의 출력을 다양화하는 사전 처리 접근 방법을 제시합니다. 사용자 프로필에 상호 작용을 추가하는 것과 추가 및 제거를 모두 포함하는 두 가지 변형을 제시하며, 1%에서 10%까지 다양한 조정 수준을 탐구합니다. Neural network-based 추천 시스템 알고리즘을 포함한 다양한 알고리즘과 함께 테스트되었습니다.

- **Performance Highlights**: MIND(뉴스 추천)와 GoodBook(책 추천) 데이터 세트에서 접근 방식의 효과를 평가했습니다. 사전 처리된 데이터를 사용한 훈련을 통해 추천 시스템은 원본 데이터로 훈련된 시스템과 비교할 때 동등하거나 개선된 성능을 보였습니다. 다양성 면에서, 특정 알고리즘에 대해 일관되게 개선된 캘리브레이션 메트릭스(calibration metrics), 혼합 결과를 보인 커버리지(coverage) 및 지니 지수(Gini index)를 보고했습니다. 또한, 사전 처리된 데이터는 공정한 nDCG(fair-nDCG) 점수가 더 높아, 더 나은 노출 공정성과 소수 카테고리의 더 나은 표현을 나타냈습니다.



### Ah, that's the great puzzle: On the Quest of a Holistic Understanding of the Harms of Recommender Systems on Children (https://arxiv.org/abs/2405.02050)
Comments:
          7 pages, 2 figures, DCDW 2024

- **What's New**: 이 위치 논문에서는 온라인 플랫폼에서 추천 시스템(Recommender Systems, RS)이 어린이에게 제공하는 콘텐츠의 특성과 그 영향에 대해 다룹니다. 어린이의 발달 단계와 복지에 잠재적으로 위험할 수 있는 콘텐츠에 대한 문제를 제기하며, 연구자, 실무자, 정책결정자들이 RS의 어린이에 대한 영향을 보다 종합적으로 검토하도록 촉구합니다.



### Diversity of What? On the Different Conceptualizations of Diversity in Recommender Systems (https://arxiv.org/abs/2405.02026)
- **What's New**: 이 연구는 네덜란드의 세 가지 공공 서비스 미디어 기관에서 추천 시스템(Recommender Systems)의 다양성(Diversity)을 어떻게 개념화하는지에 대해 조사합니다. 반구조화된 인터뷰를 통해 실무자들이 추천 시스템에서 다양성을 어떻게 이해하고 적용하는지에 대한 이해를 돕습니다.

- **Technical Details**: 연구는 다양성을 목표로 하는 추천 시스템 설계에 있어 그 개념화가 크게 다양하다는 것을 보여줍니다. 인터뷰를 통해 각 기관의 실무자들이 시스템에서 추구하는 다양성의 목표, 관련된 측면, 그리고 추천을 다양화하는 방법에 대해 설명합니다.

- **Performance Highlights**: 연구 결과는 추천 시스템의 다양성에 대한 표준화된 개념화는 이루어지기 어렵다는 점을 지적하며, 특정 시스템에서 다양성이 무엇을 의미하는지에 대한 효과적인 소통을 강조합니다. 이는 특정 도메인의 뉘앙스와 요구사항을 표현할 수 있는 다양성의 구현을 가능하게 합니다.



### RankSHAP: a Gold Standard Feature Attribution Method for the Ranking Task (https://arxiv.org/abs/2405.01848)
- **What's New**: 이 연구는 순위 매기기 작업을 위한 새로운 특징 귀속 방법인 Rank-SHAP을 도입하고 있습니다. Rank-SHAP은 기존의 Shapley 값에 기반한 알고리즘으로, 순위 매기기 작업에 적합하도록 조정된 새로운 공리들, 즉 Rank-Efficiency, Rank-Missingness, Rank-Symmetry, Rank-Monotonicity를 만족합니다. 또한 이 방법은 계산 효율성과 정확성을 평가하여 기존 시스템과 비교해 우수한 성능을 보여주었습니다.

- **Technical Details**: Rank-SHAP 알고리즘은 Shapley 값을 확장하여 NDCG(Network Discounted Cumulative Gain) 메트릭을 특징 함수로 사용하며, 다항 시간 내에 근사적인 값을 계산할 수 있는 알고리즘을 제안합니다. 이러한 접근은 게임 이론적 방법을 사용하며, 순위 매기기 태스크(feature attribution for ranking tasks)에 대한 특징 귀속 방법을 충족시키는 공리적 접근을 제공합니다.

- **Performance Highlights**: Rank-SHAP은 BM25, BERT, T5 랭킹 모델에서 MS Marco 데이터셋을 사용한 다양한 시나리오에서 평가되었습니다. Rank-SHAP은 최고의 경쟁 시스템보다 30.78% 높은 Fidelity 성능과 23.68% 높은 wFidelity 성능을 달성했습니다. 사용자 연구를 통해 Rank-SHAP은 최종 사용자가 문서를 재정렬하고 쿼리를 정확하게 예측하는 데 도움이 되는 것으로 나타났습니다.



### Are We Really Achieving Better Beyond-Accuracy Performance in Next Basket Recommendation? (https://arxiv.org/abs/2405.01143)
Comments:
          To appear at SIGIR'24

- **What's New**: 이 논문은 '다음 장바구니 추천' (Next Basket Recommendation, NBR)의 새로운 접근법을 제안합니다. 특히, 추천 시스템에서 반복 항목과 탐색 항목을 분리하여 처리하는 두 단계 반복-탐색(Two-step Repetition-Exploration, TREx) 프레임워크를 도입하였습니다. 이는 기존의 NBR 연구에서 대부분 정확도(accuracy) 최적화에 중점을 두었던 것에서 벗어나, 항목의 공정성(item fairness)과 다양성(diversity) 같은 정확도를 넘어서는 지표(beyond-accuracy metrics)를 최적화하는 데 초점을 맞춥니다.

- **Technical Details**: TREx 프레임워크는 반복 항목과 탐색 항목을 별도로 예측하는 모델을 사용합니다. 이 구성은 반복 항목을 예측하기 위해 확률 기반 방법을 채택하고, 탐색 항목에 대해서는 두 가지 전략을 설계하여 다양한 beyond-accuracy 지표를 목표로 합니다. 이 연구는 또한 정확도와 beyond-accuracy 지표 사이의 관계를 이해하고 조정할 수 있는 유연성을 TREx에 제공합니다.

- **Performance Highlights**: 실험 결과 TREx 프레임워크는 기존 방법들과 비교할 때 상당히 높은 정확도를 유지하면서도 beyond-accuracy 지표, 특히 항목의 공정성과 다양성을 향상시킬 수 있음을 보여주었습니다. 이는 TREx가 정확도와 beyond-accuracy 지표 간의 균형을 잘 이루면서도 각각의 지표를 효과적으로 최적화할 수 있는 가능성을 시사합니다.



### Generative Relevance Feedback and Convergence of Adaptive Re-Ranking: University of Glasgow Terrier Team at TREC DL 2023 (https://arxiv.org/abs/2405.01122)
Comments:
          5 pages, 5 figures, TREC Deep Learning 2023 Notebook

- **What's New**: 이 논문은 TREC 2023 딥 러닝 트랙에 대한 참여를 설명하고 있으며, 특히 제로샷(zero-shot) 및 의사 관련성 피드백(pseudo-relevance feedback) 환경에서 큰 언어 모델(large language model)을 사용한 생성적 관련성 피드백(generative relevance feedback)을 적용한 결과를 소개합니다. BM25와 SPLADE 두 가지 희소 검색 기법(sparse retrieval approaches)을 사용하여 초기 단계의 검색 결과를 개선하고, 이후 단계에서는 BM25 코퍼스 그래프(corpus graph) 상에서 monoELECTRA cross-encoder를 이용한 적응형 재순위 결정(adaptive re-ranking)을 통해 성능을 향상시키고자 하였습니다.

- **Technical Details**: 이 연구에서는 PyTerrier 정보 검색 도구(PyTerrier Information Retrieval toolkit)를 사용하여 언어 모델을 통한 쿼리 개선과 관련성 피드백을 구현하였습니다. 생성적 쿼리 재구성(Gen-QR)과 생성적 의사 관련성 피드백(Gen-PRF)을 통해 검색 쿼리의 용어를 확장하며, 이는 검색 랭킹과 관련된 언어적 불일치(lexical mismatch) 문제를 해결하는 데 도움을 줍니다. 또한 코퍼스 그래프를 통한 적응형 재순위 결정 방법을 적용, 복잡한 초기 단계 파이프라인 없이도 충분히 큰 코퍼스 그래프를 사용할 때 동일한 성능으로 수렴할 수 있는 가능성을 탐구하였습니다.

- **Performance Highlights**: 실험 결과, 생성적 쿼리 개선을 통해 일부 성능 개선이 확인되었으며, 특히 생성적 의사 관련성 피드백과 적응형 재순위 결정을 적용한 'uogtr_b_grf_e_gb' 실행이 P@10과 nDCG@10에서 가장 강력한 성능을 보였습니다. 이는 생성적 접근 방식이 특히 일부 쿼리 타입에 유리하며, 적응형 재순위 결정은 더 높은 부가 가치를 제공할 수 있음을 시사합니다.



### "In-Context Learning" or: How I learned to stop worrying and love "Applied Information Retrieval" (https://arxiv.org/abs/2405.01116)
Comments:
          9 Pages, 3 Figures, Accepted as Perspective paper to SIGIR 2024

- **What's New**: 이 연구에서는 인-컨텍스트 학습(ICL; In-context learning)과 정보 검색(IR)을 결합한 새로운 접근 방식을 제시합니다. ICL은 더 적은 데이터로 모델을 '세밀하게 조정'하는 대신, 소수의 레이블이 붙은 예시를 명령어 프롬프트에 추가하여 디코더의 생성 과정을 제어하고자 하는 방식입니다. 본 논문은 IR 문제에 대해 수십 년 동안 연구된 솔루션을 ICL의 효율성을 개선하기 위해 적용할 수 있다고 주장합니다.

- **Technical Details**: 이 논문은 특히 네트워크 정보 검색과의 연계를 통해 ICL의 예측 성능을 향상시킬 다양한 방법을 제안합니다. 첫째, 테스트 인스턴스(test instance)와 예제(few-shot examples) 사이의 유사성을 학습가능한 메트릭 공간으로 만들어 유용한 예제를 우선적으로 랭크하는 생각입니다. 둘째, 예제의 다양성을 활용함으로써 ICL의 효율성을 높이고자 합니다. 마지막으로, 예제의 품질을 예측하는 것에 따라 예제의 숫자를 조정하는 방식을 적용하는 것을 제안합니다.

- **Performance Highlights**: 이 접근 방식은 정보 검색(IR; Information Retrieval)의 잘 알려진 문제, 예를 들어 문서 검색, 쿼리 성능 예측(QPP; Query Performance Prediction)과 같은 문제에 연구된 방법론을 인-컨텍스트 학습에 적용하기 위한 가능성을 탐색합니다. 다양한 예제를 활용하고 적절한 숫자의 예제를 선택하는 것이 상당히 정확한 예측을 가능하게 할 수 있다는 예시를 제공합니다.



### Fair Recommendations with Limited Sensitive Attributes: A Distributionally Robust Optimization Approach (https://arxiv.org/abs/2405.01063)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문에서는 추천 시스템(Recommender Systems)에서 공정성(fairness)을 확보하는 새로운 방법을 제안합니다. 민감한 속성(sensitive attributes) 정보가 부족한 상황에서 공정성을 보장하기 위해, 누락된 민감한 속성을 재구성하는 대신 재구성 오류를 고려한 분포 견고 공정 최적화(Distributionally Robust Fair Optimization, DRFO) 방법을 소개합니다.

- **Technical Details**: 제안된 DRFO 방법은 민감한 속성의 가능한 모든 확률 분포를 고려하여 가장 나쁜 경우의 불공정성을 최소화합니다. 이는 재구성된 민감한 속성 대신 재구성 오류의 영향을 고려하기 위함입니다. DRFO는 실제로 민감한 속성 정보가 제한적인 경우에도 추천 시스템의 공정성을 효과적으로 보장할 수 있는 방법론입니다.

- **Performance Highlights**: 이론적 및 실증적 증거는 DRFO가 민감한 속성 정보가 부족할 때 추천 시스템에서 공정성을 효과적으로 보장할 수 있음을 보여줍니다. 또한, DRFO는 실제 세계의 민감한 속성 재구성 문제의 복잡성과 법적 규제로 인해 불가피한 재구성 오류에 강건함을 강조합니다.



### Efficient and Responsible Adaptation of Large Language Models for Robust Top-k Recommendations (https://arxiv.org/abs/2405.00824)
- **What's New**: 어떤 새로운 점: 전통적인 추천 시스템(Recommender Systems, RSs)은 주로 모든 훈련 샘플에 대해 균등하게 성능 지표를 향상시키는 것을 목표로 삼습니다. 그러나 이러한 접근은 다양한 사용자의 특성을 다루는 데 한계가 있습니다. 최근의 연구에서는 큰 언어 모델(Large Language Models, LLMs)을 사용하여 어려운 샘플에 대응하는 새로운 방법을 보여주었지만, 수백만 사용자의 긴 사용자 질의는 LLM의 성능을 저하시키고 비용 및 처리 시간을 증가시킬 수 있습니다. 이러한 문제를 해결하기 위해, 우리는 LLM과 전통적인 RSs의 능력을 활용하는 하이브리드 작업 할당 프레임워크를 제안합니다. 이 프레임워크는 부분 집단에 대한 강인성을 향상시키기 위해 이행 단계 접근 방식을 채택합니다.

- **Technical Details**: 기술적인 세부사항: 우리의 전략은 먼저 RSs에 의해 하위 최적의 순위 성능을 받는 약하거나 비활성 사용자를 식별하는 것으로 시작합니다. 다음으로, 이러한 사용자에 대해 인-컨텍스트 학습 방식(in-context learning approach)을 사용합니다. 여기서 각 사용자 상호 작용 기록은 독립적인 랭킹 작업으로 간주되어 LLM에 제공됩니다. 우리는 협업 필터링(collaborative filtering)과 학습-순위 모델(learning-to-rank recommendation models)과 같은 다양한 추천 알고리즘을 포함하여 우리의 하이브리드 프레임워크를 테스트하였고, 두 가지 LLMs(오픈 소스 및 비공개 소스)을 사용했습니다.

- **Performance Highlights**: 성능 하이라이트: 세 개의 실제 데이터셋에서의 테스트 결과에 따르면, 우리의 하이브리드 프레임워크는 약한 사용자의 수를 상당히 줄이고 부분 집단에 대한 RSs의 강인성을 약 12% 향상시켰으며, 전체 성능도 향상시켰습니다. 이러한 성과는 불균형적인 비용 증가 없이 이루어졌습니다.



### A First Look at Selection Bias in Preference Elicitation for Recommendation (https://arxiv.org/abs/2405.00554)
Comments:
          Accepted at the CONSEQUENCES'23 workshop at RecSys '23

- **What's New**: 이 연구에서는 추천 시스템에서 선호도 수집(Preference Elicitation, PE) 단계의 선택 편향(Selection Bias)에 초점을 맞추고 있습니다. 이전의 연구들은 선택 편향이 명시적, 암묵적 피드백에 미치는 영향을 분석했지만, PE 단계에서의 선택 편향이 추천 결과에 미치는 영향은 아직 연구되지 않았습니다. 이 연구는 PE 내의 선택 편향을 시뮬레이션 및 실험을 통해 분석하고, 편향 완화 방법(Debiasing Methods)을 제시하여 추천 성능을 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구진은 주제 기반 선호도 수집 형태(Topic-based Preference Elicitation)를 제안하고, 이를 통해 사용자의 선호를 수집합니다. 시뮬레이션을 통해 생성된 데이터셋을 활용하여 선택 편향의 영향을 분석하고 있으며, 기존 데이터셋에서 PE 단계를 시뮬레이션하는 방법을 소개합니다. 또한, 선택 편향을 교정하기 위해 일반적으로 사용되는 Debiasing Methods를 적용하는 방안을 논의하고 있습니다.

- **Performance Highlights**: 실험 결과는 선택 편향을 초기에 무시하면 추천 항목에서 과대표현(Overrepresentation)의 문제가 심화될 수 있음을 보여줍니다. 반면, Debiasing Methods의 적용은 이러한 효과를 완화시키고 추천 성능을 크게 개선할 수 있다는 점을 시사합니다. 이 연구는 PE에 대한 Debiasing 방법을 개발하고 향후 연구를 위한 기초 데이터와 동기를 제공하고자 합니다.



### KVP10k : A Comprehensive Dataset for Key-Value Pair Extraction in Business Documents (https://arxiv.org/abs/2405.00505)
Comments:
          accepted ICDAR2024

- **What's New**: 이 논문은 'KVP10k'라는 새로운 데이터셋과 벤치마크를 소개하며, Key-Value Pair 추출에 중점을 둔 것이 특징입니다. 기존의 벤치마크와는 달리, 사전에 정의된 키에 의존하지 않고 다양한 템플릿과 복잡한 레이아웃에서의 정보 추출에 초점을 맞춰 기존 데이터셋의 한계를 넘어서고자 합니다.

- **Technical Details**: KVP10k 데이터셋은 10,707개의 풍부하게 주석이 달린 이미지를 포함하고 있으며, Key-Value Pair (KVP) 추출을 위한 가장 큰 데이터셋입니다. 이 데이터셋은 KIE (Key Information Extraction) 요소와 복합적으로 통합된 새로운 구조를 제시하여, 문서로부터 정보를 추출하는 모델의 성능을 평가할 수 있는 새로운 벤치마크로서의 역할을 합니다.

- **Performance Highlights**: 초기 기준 성능 결과를 공유하여 KVP 추출 방법을 개선하기 위한 연구의 기반을 마련하고 있습니다. 데이터셋은 광범위한 키 배열과 정밀한 주석을 포함하며, 이를 통해 훈련 및 평가의 실질적인 기반을 제공합니다. 이는 문서 처리 기술의 발전을 지원하고 엄격한 연구를 위한 플랫폼을 제공합니다.



### Exploiting Positional Bias for Query-Agnostic Generative Content in Search (https://arxiv.org/abs/2405.00469)
Comments:
          8 pages, 4 main figures, 7 appendix pages, 2 appendix figures

- **What's New**: 최근 몇 년 동안, 신경 순위 모델(neural ranking models, NRMs)이 텍스트 검색에서 그들의 어휘적(lexical) 대응물을 현저하게 초과 성능을 보여주었습니다. 이번 연구에서는 검색 모델에서 변환자 어텐션 메커니즘(transformer attention mechanism)이 위치적 편향(positional bias)을 통해 악용 가능한 결함을 유발할 수 있다고 제시하며, 이러한 결함이 단일 쿼리나 주제를 넘어 일반화될 수 있는 공격을 이끌 수 있는지를 탐구합니다.

- **Technical Details**: 연구진은 이미지 주입의 위치를 제어함으로써 주제의 지식 없이도 비관련 텍스트의 부정적인 영향을 줄일 수 있음을 보여줍니다. 이는 그라디언트 기반의 공격(gradient-based attacks)과는 달리, 쿼리 불특정(query-agnostic) 방식으로 편향을 입증합니다. 실험은 대상 문서의 주제 컨텍스트와 연계하여, 대규모 언어 모델(LLMs)에 의해 자동으로 생성된 주제 관련 프로모션 텍스트를 사용하여 수행되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 비관련 텍스트의 맥락화(contextualisation)는 부정적인 영향을 추가로 감소시키면서 기존의 내용 필터링 메커니즘을 우회할 가능성이 높다고 합니다. 반면, 어휘 모델(lexical models)은 이러한 내용 주입 공격에 더욱 강한 내성을 보여주었습니다. 또한, 연구진은 NRMs의 약점을 보상하기 위한 간단하지만 효과적인 방법을 조사하여, 변환자 편향(transformer bias)에 관한 가설을 검증합니다.



### Distance Sampling-based Paraphraser Leveraging ChatGPT for Text Data Manipulation (https://arxiv.org/abs/2405.00367)
Comments:
          Accepted at SIGIR 2024 short paper track

- **What's New**: 오디오-텍스트 쌍 데이터셋에서 겪는 데이터 불균형 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 이 연구는 오디오와 텍스트 모달리티(modalities) 간의 상관관계를 설정하는 오디오-언어 검색 분야에서 큰 관심을 끌고 있습니다. 기존의 오디오-텍스트 데이터셋의 한계를 극복하기 위해, ChatGPT를 활용한 거리 샘플링 기반의 재구성기(paphraser) 방식을 도입했습니다.

- **Technical Details**: 이 방법은 Jaccard 유사도(similarity)로 정의된 텍스트 클러스터를 사용하여 ChatGPT의 퓨샷 프롬프팅(few-shot prompting)을 수행함으로써, 동일한 맥락을 가진 문장 집합에 대해 두 문장 간의 조작 정도를 계산하는 거리 함수(distance function)를 사용합니다. 그 결과, 조작된 텍스트 데이터의 분포를 제어할 수 있습니다.

- **Performance Highlights**: 제안된 접근 방식은 오디오-텍스트 검색 작업에서 뛰어난 성능을 보여줍니다. 전통적인 텍스트 증강 기술(text augmentation techniques)보다 우수한 성능을 나타내며, 다양성을 조절할 수 있는 새로운 방법론으로 데이터셋의 품질을 향상시킬 수 있습니다.



### Distillation Matters: Empowering Sequential Recommenders to Match the Performance of Large Language Mod (https://arxiv.org/abs/2405.00338)
Comments:
          10 pages, 2 figures

- **What's New**: 이 논문은 Large Language Models (LLMs)에서 전통적인 순차적 모델로 지식을 증류하기 위한 새로운 전략인 DLLM2Rec을 제안합니다. 이 연구는 LLM 기반 추천 모델에서 작고 효율적인 모델로 지식 전달을 용이하게 하여 높은 추론 지연(Latency) 문제를 해결합니다.

- **Technical Details**: DLLM2Rec 전략은 중요성 인식 랭킹 증류(Importance-aware ranking distillation)와 협력적 임베딩 증류(Collaborative embedding distillation)의 두 부분을 포함합니다. 첫 번째는 교사 모델의 신뢰도와 학생-교사 일관성에 따라 인스턴스를 가중하는 방법이며, 두 번째는 교사 임베딩에서의 지식과 데이터에서 채굴된 협력적 신호를 통합하는 기법입니다.

- **Performance Highlights**: DLLM2Rec을 사용하여 전통적인 순차적 모델들의 성능을 평균 47.97% 향상시켰으며, 일부 경우에는 LLM 기반 추천 모델들을 능가하기도 했습니다. 이는 LLM 기반 모델의 높은 성능을 갖춘 추천 시스템의 경제적이고 실용적인 배포 가능성을 보여줍니다.



### Characterizing Information Seeking Processes with Multiple Physiological Signals (https://arxiv.org/abs/2405.00322)
- **What's New**: 이 연구는 정보 검색 과정에서 사용자 행동을 생리 신호를 통해 수량화하여 분석하는 최초의 연구입니다. 이 연구는 인지 부하(Cognitive Load), 정서적 각성(Affective Arousal), 감정 평가(Affective Valence) 등을 측정하여 정보 필요(IN), 쿼리 구성(QF), 쿼리 제출(QS), 관련성 판단(RJ) 등의 검색 단계에서 사용자의 행동을 특성화합니다.

- **Technical Details**: 생리 신호, 즉 전기 피부 활동(Electrodermal Activity, EDA), 광용적맥파(Photoplethysmogram, PPG), 뇌전도(Electroencephalogram, EEG), 동공 반응(Pupillary Responses) 등을 기록하여 분석하였습니다. 이러한 데이터들은 검색 과정의 각 단계에서 인지적 및 정서적 반응을 측정하는 데 사용되었습니다. 연구는 26명의 참가자를 대상으로 실험실 환경에서 실시되었습니다.

- **Performance Highlights**: 결과에 따르면, 정보 필요(IN) 단계에서는 인지 부하가 가장 높게 나타났으며, 쿼리 구성(QF) 단계에서는 주의력이 특히 요구되는 것으로 나타났습니다. 쿼리 제출(QS) 단계는 쿼리 구성(QF) 단계보다 인지 부하가 더 높았으며, 관련성 판단(RJ) 단계에서는 정서적 반응이 더욱 두드러지게 나타나, 지식 격차 해소에 따른 흥미와 참여가 증가함을 보여주었습니다.



### Stochastic Sampling for Contrastive Views and Hard Negative Samples in Graph-based Collaborative Filtering (https://arxiv.org/abs/2405.00287)
- **What's New**: 이 논문에서는 데이터 희소성과 부정적 샘플링 문제를 해결하기 위해 SCONE(Stochastic sampling for COntrastive views and hard NEgative samples)이라는 새로운 방법을 제안합니다. SCONE은 그래프 기반 협업 필터링(Collaborative Filtering, CF)과 결합하여 개인화 추천 시스템에서 최고의 성능을 달성합니다.

- **Technical Details**: SCONE은 점수 기반 생성 모델(score-based generative models)을 사용하는 통합 확률적 샘플링(stochastic sampling) 프레임워크를 통해 동적으로 증강된 뷰(dynamic augmented views)와 다양한 어려운 부정적 샘플(hard negative samples)을 생성합니다. 이 방법은 사용자의 희소성(user sparsity)과 아이템의 인기도(item popularity) 문제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 6개의 벤치마크 데이터셋을 사용한 평가에서 SCONE은 추천 정확도와 강건성을 크게 향상시키며, 기존 CF 모델들을 상회하는 우수성을 입증했습니다. 이로써, 정보가 풍부한 환경에서의 개인화 추천 시스템의 성능을 혁신적으로 향상시켰습니다.



### Global News Synchrony and Diversity During the Start of the COVID-19 Pandemic (https://arxiv.org/abs/2405.00280)
- **What's New**: 이 연구는 국가 간 뉴스 보도의 동기성(synchrony)과 다양성(diversity)의 요인들을 규명하고 국제 관계에 미치는 영향을 분석하였습니다. 특히 다양한 국가에서 발행된 6000만 개의 뉴스 기사를 분석하여, 2020년 상반기 동안 발생한 주요 글로벌 뉴스 이벤트를 식별하고 이들 이벤트에 대한 보도의 국가별 동기성과 다양성을 측정하였습니다. 이는 대규모 데이터셋을 활용하여 다국어 뉴스 유사성을 추론하는 트랜스포머 모델(transformer model), 뉴스 기사의 유사성 네트워크를 기반으로 글로벌 이벤트를 식별하는 시스템, 그리고 국가별 뉴스 다양성 및 동기성을 측정하는 방법론을 개발하는 데 초점을 맞추었습니다.

- **Technical Details**: 연구팀은 124개국, 10개 언어로 작성된 뉴스 기사 6000만 건을 분석하여 4357개의 뉴스 이벤트를 식별했습니다. 이 데이터는 트랜스포머 모델을 사용하여 다국어 뉴스의 유사성을 추정하고, 이를 기반으로 글로벌 이벤트를 식별하며, 뉴스 기사의 유사성 네트워크를 클러스터링하는 방법을 포함합니다. 또한, 이는 각 국가별로 글로벌 이벤트의 뉴스 보도 분포를 기반으로 그 국가에서의 뉴스 다양성 및 다른 국가들과의 동기성을 측정하는 새로운 지표를 개발하는 것을 포함합니다.

- **Performance Highlights**: 이 방법론은 대규모 뉴스 데이터셋에서의 확장성과 효율성을 입증하였습니다. 연구 결과, 인터넷 침투도가 높은 국가, 공식 언어의 수가 많은 국가, 종교적 다양성이 큰 국가, 경제적 불평등이 큰 국가, 인구가 많은 국가에서 뉴스 이벤트의 보도가 다양하게 나타났습니다. 또한, 높은 양자 무역량, NATO나 BRICS 같은 국제적 정치 경제 그룹에 속한 국가들, 공식 언어나 높은 GDP, 높은 민주주의 지수를 공유하는 국가들 간에는 뉴스 보도의 동기성이 더 높게 나타났습니다.




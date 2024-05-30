### Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevanc (https://arxiv.org/abs/2405.18276)
Comments:
          Accepted to SIGIR 2024 as full paper

- **What's New**: 이 연구는 추천 시스템(Recommender Systems, RS)에서 관련성(Relevance)과 공정성(Fairness)을 동시에 측정하는 지표의 신뢰성에 대한 최초의 실증적 연구를 수행했습니다. 연구진은 네 가지 실제 데이터셋과 네 가지 추천 시스템을 이용해 조합 평가 지표들의 일관성과 민감도를 분석했습니다.

- **Technical Details**: 조사한 지표들은 관련성과 분리된 공정성(Fairness-only)과 관련성을 고려한 공정성(Joint measures)으로 구분되었습니다. 관련성과 공정성의 조합 평가 지표는 각 측정값이 얼마나 서로 일치하는지, 관련성/공정성 전용 지표와 얼마나 일치하는지, 순위 변화와 공정성/관련성이 증가함에 따라 얼마나 민감하게 반응하는지 조사했습니다.

- **Performance Highlights**: 연구 결과, 대부분의 조합 평가 지표들은 서로 약하게 상관되며 때로는 모순된다는 점을 발견했습니다. 또한 이 지표들은 순위 변화에 덜 민감하여 전통적인 RS 지표보다 세밀도가 낮았습니다. 조합 평가 지표들은 점수가 낮은 범위에 압축되는 경향이 있어 표현력이 제한적이라는 점도 확인했습니다. 이러한 제한점을 고려하여 연구진은 해당 지표들을 사용할 때 주의가 필요하다는 가이드라인을 제시했습니다.



### A Vlogger-augmented Graph Neural Network Model for Micro-video Recommendation (https://arxiv.org/abs/2405.18260)
- **What's New**: 기존의 마이크로 비디오 추천 모델은 사용자와 마이크로 비디오 간의 상호 작용과/또는 마이크로 비디오의 멀티 모달 정보를 활용하여 다음에 사용자가 시청할 마이크로 비디오를 예측합니다. 하지만 이 모델들은 마이크로 비디오의 제작자인 브이로거(vlogger) 관련 정보를 무시하고 있습니다. 이에 연구팀은 브이로거의 효과를 고려한 브이로거 증강 그래프 신경망 모델인 VA-GNN를 제안합니다. 이 모델은 사용자, 마이크로 비디오, 브이로거를 노드로 하는 삼중 그래프를 구성하여, 두 가지 시점에서 사용자 선호도를 포착합니다.

- **Technical Details**: VA-GNN은 사용자, 마이크로 비디오, 브이로거 간의 복잡한 의미론적 관계를 활용합니다. 노드 임베딩(node embeddings)을 두 가지 시점에서 학습하고, 메타 경로(meta-path)를 설정하여 원래 이종 그래프의 개별 시점 간의 연결을 구축합니다. 또한, 두 가지 시점의 임베딩 일관성을 유지하기 위해 교차 시점 대조 학습(cross-view contrastive learning)을 수행합니다. 이 모델은 사용자-비디오 상호작용, 사용자-브이로거 상호작용, 브이로거-비디오 배포 관계를 완전히 활용하도록 설계되었습니다.

- **Performance Highlights**: VA-GNN은 두 가지 실제 데이터셋에 대한 광범위한 실험을 통해 성능을 검증했습니다. 실험 결과, VA-GNN은 여러 기존 GNN 기반 추천 모델보다 우수한 성과를 보였습니다. 특히, Recall (재현율)과 NDCG와 같은 평가 지표에서 더 나은 결과를 얻었습니다.



### Unified Low-rank Compression Framework for Click-through Rate Prediction (https://arxiv.org/abs/2405.18146)
Comments:
          Accepted by KDD2024 Applied Data Science (ADS) Track

- **What's New**: 새로운 논문에서는 딥 CTR(Click-Through Rate) 예측 모델에 대한 저랭크 압축(low-rank approximation) 기법을 제안하고 있습니다. 이 기법은 기존 모델의 성능을 유지하면서 메모리 사용량을 크게 줄이고, 추론 속도를 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 고전적인 SVD(matrix decomposition) 방법을 사용하며, 모델의 가중치를 압축하는 대신 출력 피처(output features)를 로컬로 압축하는 접근법을 취합니다. 이 프레임워크는 여러 CTR 예측 모델의 embedding tables와 MLP 층에 적용될 수 있습니다.

- **Performance Highlights**: 제안된 저랭크 압축 프레임워크는 두 개의 학문적 데이터셋과 실제 산업 벤치마크를 포함한 광범위한 실험을 통해 검증되었습니다. 이 실험에서 최대 3-5배의 모델 크기 감소를 이루었으며, 압축된 모델이 오리지널 모델보다 더 빠른 추론 속도를 제공하고 높은 AUC를 달성했습니다.



### A Survey of Latent Factor Models in Recommender Systems (https://arxiv.org/abs/2405.18068)
- **What's New**: 최근 발표된 논문은 추천 시스템(recommender systems)에서 잠재 요인 모델(latent factor models)에 대한 체계적인 리뷰를 제공합니다. 잠재 요인 모델은 사용자-아이템 상호작용의 숨겨진 패턴을 파악하는 데 효과적이며, 이를 통해 개인화된 콘텐츠를 제공하는 데 중요한 역할을 합니다. 본 설문조사는 학습 데이터, 모델 아키텍처, 학습 전략 및 최적화 기법의 관점에서 문헌을 정리합니다.

- **Technical Details**: 잠재 요인 모델은 매트릭스 분해 기법(matrix factorization techniques)을 활용하여 사용자와 아이템 간의 관계를 수학적으로 표현합니다. 주목할 만한 요소들은 다음과 같습니다: 학습 데이터는 명시적 피드백(explicit feedback)과 암시적 피드백(implicit feedback), 신뢰 데이터(trust data) 등을 포함하며, 모델 아키텍처는 확률적 모델(probabilistic models), 비선형 모델(nonlinear models), 신경망 모델(neural models) 등을 포함합니다. 학습 전략에는 온라인 학습(online learning), 전이 학습(transfer learning), 능동 학습(active learning) 등이 있으며, 최적화 기법으로는 확률적 경사 하강법(stochastic gradient descent) 등이 사용됩니다.

- **Performance Highlights**: 잠재 요인 모델은 데이터의 희소성(sparsity) 및 확장성(scalability) 문제를 효과적으로 해결할 수 있습니다. 특히, 암시적 피드백과 신뢰 데이터를 활용하면 추천의 정확도가 향상됩니다. 또한, 전이 학습과 능동 학습 같은 학습 전략을 활용하면 모델의 적응성(adaptability) 및 효율성을 크게 높일 수 있습니다.



### ReChorus2.0: A Modular and Task-Flexible Recommendation Library (https://arxiv.org/abs/2405.18058)
Comments:
          10 pages, 3 figures. Under review

- **What's New**: ReChorus2.0가 발표되었습니다. 기존 ReChorus를 기반으로 다양한 입력 형식, 모델, 훈련 및 평가 전략을 확장하여 다양한 추천 작업을 지원합니다. 특히, 다시 정렬(reranking) 및 CTR 예측 작업을 포함한 복잡한 작업과 여러 컨텍스트를 고려합니다.

- **Technical Details**: ReChorus2.0는 모듈 형식으로 설계되어, 사용자가 필요로 하는 실험 전략을 쉽게 구현할 수 있습니다. 주요 기능은 다음과 같습니다.: (1) 다시 정렬(reranking) 및 CTR 예측과 같은 복잡한 작업의 실현, (2) 상황별 컨텍스트(context-aware) 정보와 함께 사용자의 구체적인 설정을 지원, (3) 훈련 및 평가 시 다양한 후보군 집합을 유연하게 구성할 수 있는 기능을 포함합니다. 다양한 입력 형식을 수용하여 상황별 컨텍스트, 사용자 및 아이템 프로필, 클릭 라벨 등을 포함할 수 있습니다.

- **Performance Highlights**: ReChorus2.0는 기존 추천 시스템 연구 라이브러리의 한계를 극복하고 보다 유연하고 실용적인 연구 환경을 제공합니다. 동일한 모델을 사용하여 여러 작업을 수행하고, 사용자 맞춤형 입력을 지원하여 연구자들이 이론적, 방법론적 연구에 집중할 수 있도록 돕습니다. ReChorus2.0의 구현 및 자세한 튜토리얼은 제공된 웹사이트에서 확인할 수 있습니다.



### Rethinking Recommender Systems: Cluster-based Algorithm Selection (https://arxiv.org/abs/2405.18011)
Comments:
          16 pages, 8 figures, 2 tables

- **What's New**: 클러스터 기반 (Cluster-based) 알고리즘 선택은 사용자를 클러스터링한 후에 추천 알고리즘을 선택함으로써 추천 알고리즘의 성능을 향상시키는 효과적인 기술임을 보여줍니다. 이 연구는 8개의 데이터셋, 4개의 클러스터링 방법, 8개의 추천 알고리즘을 다루며, 각 클러스터에 대해 최적의 추천 알고리즘을 선택합니다.

- **Technical Details**: 이 연구는 클러스터링을 통해 사용자 그룹을 형성한 후, 각 클러스터에 최적화된 추천 알고리즘을 적용하여 성능을 최적화합니다. K-평균 (K-Means) 클러스터링 및 HDBSCAN 등 여러 클러스터링 접근법이 사용되었습니다. 또한, SVD와 같은 추천 알고리즘들을 사용하여 클러스터별로 성능을 평가합니다. 연구는 LensKit-Auto라는 자동화된 추천 시스템 툴킷을 사용하여 하이퍼파라미터 최적화 및 알고리즘 훈련을 실시합니다.

- **Performance Highlights**: 클러스터 기반 알고리즘 선택은 8개의 데이터셋 중 5개에서 성능을 19.28%에서 최대 360.38%까지 향상시켰습니다. 평균적으로 nDCG@10이 66.47% 향상되었으며, 이는 클러스터링 없이 단일 알고리즘을 사용할때보다 높은 성능을 보여줍니다.



### Attention-based sequential recommendation system using multimodal data (https://arxiv.org/abs/2405.17959)
Comments:
          18 pages, 4 figures, preprinted

- **What's New**: 새로운 연구는 전자상거래(e-commerce) 분야에서 사용자의 동적인 선호도를 모델링하는 연속 추천 시스템(sequential recommendation system)을 개선하는 데 주력하고 있습니다. 본 연구는 최초로 이미지, 텍스트, 카테고리와 같은 멀티모달(multi-modal) 데이터를 직접 사용하여 사용자에게 제품을 추천하는 방법을 제안합니다. 기존 시스템은 제한된 구조적 정보만 사용하였으나, 본 연구는 이러한 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 Pre-trained VGG와 BERT를 사용하여 이미지와 텍스트 특징을 추출하고, 카테고리를 멀티레이블(Multi-labeled) 형식으로 변환합니다. 그런 다음 독립적인 주의력 연산(attention operations)을 수행하며, 멀티모달 표현을 통합하여 최종 추천을 수행합니다. 결합된 주의력 정보를 조정하기 위해 주의력 융합 함수(attention fusion function)를 사용합니다. 학습 과정에서 각 모달리티를 더 잘 대표하기 위해 멀티태스크 학습 손실(multitask learning loss)이 적용되었습니다.

- **Performance Highlights**: Amazon 데이터셋을 사용한 실험 결과에서 제안된 방법이 기존의 연속 추천 시스템을 능가하는 성능을 보여 주었습니다. 멀티모달 데이터를 효과적으로 사용함으로써 추천 시스템의 성능을 크게 향상시켰습니다. 다양한 아이템 속성과 사용자의 구매 기록을 활용해 더욱 맞춤화된 추천을 가능하게 합니다.



### The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds (https://arxiv.org/abs/2405.17813)
Comments:
          15 pages, 2 figures

- **What's New**: 본 논문은 AI 응용 프로그램에서 중요한 역할을 하는 벡터 검색 시스템의 효율성을 탐구합니다. 특히, Hierarchical Navigable Small Worlds (HNSW) 알고리즘의 실제 상황에서의 동작을 집중적으로 분석합니다. 현재의 벤치마크들은 단순한 데이터셋에 중점을 두고 있어, 복잡한 실제 사례를 반영하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: 연구는 다양한 벡터 데이터셋 (벡터 영역의 내재 차원성과 데이터 삽입 순서의 관계)에서 HNSW의 동작을 평가합니다. 다양한 데이터셋에는 합성 벡터, 인기 있는 검색 벤치마크와 CLIP 모델로부터 생성된 전자상거래 이미지 데이터가 포함됩니다. HNSW의 기본 파라미터를 고정하여 실제 사례에 더 가까운 평가를 수행합니다.

- **Performance Highlights**: Approximate HNSW 검색의 리콜(Recall)은 벡터 공간의 내재 차원성과 데이터 삽입 순서에 크게 영향을 받는다는 것을 발견했습니다. 데이터 삽입 순서를 제어하면 리콜이 최대 12% 포인트까지 변동될 수 있으며, 이는 기존의 KNN 검색과 비교했을 때 주목할 만한 차이를 보입니다. 또한, 벤치마크 데이터셋의 순위는 최대 3위까지 변동될 수 있습니다.



### Learn to be Fair without Labels: a Distribution-based Learning Framework for Fair Ranking (https://arxiv.org/abs/2405.17798)
Comments:
          ICTIR'23

- **What's New**: 이 논문은 기존 학습 기반 공정성 랭킹(fair ranking) 알고리즘의 주요 한계를 해결하기 위해 라벨이 필요 없는 분포기반 공정 학습 프레임워크(Distribution-based Fair Learning, DLF)를 제안합니다. 특히, 이 프레임워크는 공정성 라벨을 목표 노출 분포(target fairness exposure distributions)로 대체하여 공정성을 보장합니다.

- **Technical Details**: DLF는 공정성 라벨이 없는 상태에서 학습할 수 있는 새로운 프레임워크입니다. 기존의 공정성-aware loss 함수를 사용하되, 라벨이 아닌 목표 노출 분포를 활용합니다. 이를 통해 DLF는 라벨이 없는 상황에서도 최소 손실을 보장할 수 있습니다. 또한 TREC fair ranking track 데이터셋을 사용하여 실험하였으며, 여기서 자연어 처리(NLP) 기법을 통해 컨텍스트(features)를 최대한 활용합니다. 결과적으로 DLF는 공정성 모델과 관련성 모델을 따로 학습하고, 이를 가중합(weighted sum) 함수로 병합하여 공정성과 관련성의 균형을 효과적으로 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 DLF 프레임워크는 공정성과 관련성의 균형을 기존의 최첨단 공정 랭킹 프레임워크보다 더 잘 관리하며, 높은 성능을 보였습니다. 특히, 공정성을 유지하면서도 랭킹의 관련성을 효과적으로 관리할 수 있음을 입증했습니다.



### Dataset Regeneration for Sequential Recommendation (https://arxiv.org/abs/2405.17795)
- **What's New**: 최신 연구에서는 순차적 추천 시스템(sequential recommender, SR)에서 **데이터 중심(data-centric)** 패러다임을 제안하여, 기존의 모델 중심(model-centric) 접근 방식이 간과하는 데이터 품질 문제를 해결하려고 합니다. 새로운 데이터 재생성 프레임워크인 DR4SR을 도입해 다양한 모델 아키텍처에서 데이터셋을 효과적으로 재생성하며, DR4SR+는 타겟 모델에 맞춰 데이터셋을 개인화합니다.

- **Technical Details**: DR4SR 프레임워크는 모델 비종속적 데이터 재생성 방법으로 뛰어난 크로스 아키텍처 일반화 성능을 지닙니다. 또한 DR4SR+ 프레임워크는 타겟 모델에 맞게 데이터셋을 조정해 성능을 극대화합니다. 이 접근법은 데이터셋에서 아이템 전이 패턴(item transition patterns)을 명시적으로 학습하여, 모델 학습 과정을 두 단계로 분리합니다.

- **Performance Highlights**: 다양한 모델 중심 방법들과 통합하여 4개의 널리 사용되는 데이터셋에서 실험을 진행한 결과, 데이터 중심 패러다임이 기존 방법들에 비해 상당한 성능 개선을 보였습니다. 또한, DR4SR의 크로스 아키텍처 일반화 능력과 데이터 중심 및 모델 중심 패러다임의 상호 보완성을 입증하였습니다.



### RREH: Reconstruction Relations Embedded Hashing for Semi-Paired Cross-Modal Retrieva (https://arxiv.org/abs/2405.17777)
Comments:
          Accepted by the 20th International Conference on Intelligent Computing (ICIC 2024)

- **What's New**: 이번 연구에서는 반쌍 대응(correspondence)이 주어진 교차 모달 검색(cross-modal retrieval) 과제에 적합한 혁신적인 비지도 해싱(unsupervised hashing) 기법을 소개했습니다. 이를 'Reconstruction Relations Embedded Hashing (RREH)'이라고 명명했습니다. 주요 특징은 쌍을 이루지 않은 데이터와 앵커(anchors) 간의 고차 관계(high-order relationships)를 이용하여 잠재 공간(latent subspace)에서 데이터를 효과적으로 재구성하고, 쌍을 이루는 데이터에서 앵커를 선택하여 해시 학습의 효율성을 높입니다.

- **Technical Details**: RREH는 여러 모달 데이터가 공통된 잠재 공간을 공유한다고 가정합니다. 쌍을 이루는 데이터는 모달리티 간의 일관된 정보를 찾기 위해 공유 표현을 사용하며, 쌍을 이루지 않은 데이터는 고차 관계를 앵커와 연관시켜 잠재 공간에 포함시킵니다. 이 과정에서 효율적인 선형 재구성(linear reconstruction)을 사용하여 시간 복잡성을 낮추고 데이터를 변별력 있게 만듭니다. 아울러 비트 코드 생성(binary code generation)을 위한 고차 재구성 관계(high-order reconstruction relationships)를 보존하기 위해 잘 설계된 목적 함수와 이산 최적화 알고리즘(discrete optimization algorithm)을 포함합니다.

- **Performance Highlights**: 두 개의 널리 사용되는 데이터셋에 대한 평가 결과, RREH는 정확도와 계산 효율성 측면에서 여러 기존 방법을 능가하는 성능을 보였습니다. 이 모델은 특히 대규모 데이터셋에 대해 시간 복잡성을 최소화하고, 해시 함수와 해시 코드를 동시에 학습하여 더 변별력 있는 비트 코드를 생성할 수 있습니다.



### Advancing Cultural Inclusivity: Optimizing Embedding Spaces for Balanced Music Recommendations (https://arxiv.org/abs/2405.17607)
- **What's New**: 이 논문에서는 음악 추천 시스템의 인기 편향(popularity bias)을 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 행렬 분해(matrix factorization) 방식과 달리, prototype-기반의 행렬 분해 방식은 해석 가능성이 높아 사용자와 항목의 문화적 미묘함을 포착할 수 있습니다. 이를 통해 소수 문화 그룹에 속한 아티스트들이 덜 추천되는 문제를 해결하고자 합니다.

- **Technical Details**: 본 연구는 prototype-based matrix factorization을 활용하여 음악 추천 시스템의 문화적 공정성을 높이기 위한 두 가지 기술적 개선을 제안합니다. 첫째, 사용되지 않는 prototype을 필터링하여 사용자와 항목 대표성을 개선하고, 둘째, 임베딩 공간에서 prototype들을 균등하게 분포시키기 위한 정규화 기술을 도입합니다. 이 접근방식은 인기와 문화적 편향을 모두 줄이는 데 기여하며, 시스템의 해석 가능성을 높여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 인기 편향을 줄이고 문화적 공정성을 향상시키는 데 있어 유의미한 성능 개선을 보여주었습니다. 더불어 두 가지 실제 데이터셋에서 경쟁력 있는 성능을 입증했습니다.



### Augmenting Textual Generation via Topology Aware Retrieva (https://arxiv.org/abs/2405.17602)
- **What's New**: 이 논문은 텍스트 생성 중 발생할 수 있는 비정확하거나 가상의 내용을 해결하기 위해 Retrieval-augmented Generation(RAG)을 제안합니다. 저자들은 텍스트 간의 관계를 활용하여 추가적인 정보를 외부 데이터베이스에서 검색하는 방식을 탐구했습니다. 특히 인접 노드 기반(proximity-based)과 역할 기반(role-based) 두 종류의 토폴로지 관계를 사용하여 RAG를 개선하는 프레임워크를 제시했습니다.

- **Technical Details**: Topo-RAG(Topology-aware Retrieval-augmented Generation) 프레임워크는 두 개의 주요 모듈로 구성됩니다. 첫 번째는 토폴로지 관계에 기반하여 텍스트를 선택하는 검색 모듈입니다. 두 번째는 선택된 텍스트를 프롬프트(prompt)에 통합하여 대형 언어 모델(LLM)이 텍스트 생성을 수행하도록 유도하는 집계 모듈입니다. 저자들은 텍스트 특성화된 네트워크(text-attributed networks)를 구축하고, 여러 데이터셋에서 프레임워크의 효용성을 검증했습니다.

- **Performance Highlights**: 다양한 도메인에 걸쳐 수행된 실험 결과, Topo-RAG 프레임워크는 기존의 RAG 방식에 비해 성능이 우수함이 입증되었습니다. 이 프레임워크는 논문 초록 작성과 같이 학술적 텍스트 생성 작업에서 특히 유의미한 성과를 보였습니다. 또한 노드 분류(node classification)와 링크 예측(link prediction)을 활용하여 생성된 텍스트의 품질을 평가하는 데 성공했습니다.



### Ranking with Ties based on Noisy Performance Data (https://arxiv.org/abs/2405.18259)
Comments:
          26 pages, 23 figures

- **What's New**: 이 논문에서는 잡음이 섞인 성능 측정을 기반으로 객체 집합을 순위 매기는 문제를 다룹니다. 측정이 여러 번 반복되면서 각 객체에 대한 다양한 측정 값들이 나옵니다. 만약 두 객체의 측정 범위가 겹치지 않으면 하나를 더 우수하다고 간주하지만, 겹치면 비교할 수 없다고 간주합니다. 그러나 이러한 비교 불가능성 관계는 일반적으로 이행적이지 않아서 모든 요구 사항을 동시에 만족할 수 없습니다. 이에 따라 부분 순위(Partial Rankings)라는 새로운 순위 매기기 방법을 정의하고 분석합니다.

- **Technical Details**: 논문은 Performance가 잡음에 취약한 고성능 컴퓨팅(HPC) 및 비즈니스 프로세스 관리(BPM)와 같은 다양한 도메인에서 순위 매기기 문제를 탐구합니다. 예를 들어, HPC에서는 일반적으로 알고리즘의 실행 시간을 비교합니다. 이 실행 시간은 환경적 요인에 따라 변동할 수 있습니다. 논문에서는 Julia 언어로 구현된 일반화된 최소 제곱(GLS) 문제의 10가지 알고리즘 변형을 논의하며 각 알고리즘 변형의 실행 시간을 10회 측정한 데이터를 box plot으로 시각화하여 분석합니다. 여기서 Inter Quartile Interval (IQI)를 사용하여 순위 매기기 방법을 설명합니다.

- **Performance Highlights**: 논문은 잡음이 섞인 측정 데이터로부터 다양한 순위를 도출할 수 있는 세 가지 방법론을 개발하고 분석합니다. 이를 통해 각각의 객체 간 성능 차이를 부분 순위를 통해 조사할 수 있음을 보여줍니다. 다양한 방법론을 통해 동일 순위 처리(ties)를 고려한 성능 클래스 분류 및 우수한 성능을 가진 객체 도출이 가능함을 입증하였습니다.



### Video Enriched Retrieval Augmented Generation Using Aligned Video Captions (https://arxiv.org/abs/2405.17706)
Comments:
          SIGIR 2024 Workshop on Multimodal Representation and Retrieval (MRR 2024)

- **What's New**: 이 연구에서는 '정렬된 시각적 캡션(aligned visual captions)'을 사용하여 동영상 내 정보를 검색 증강 생성(RAG: Retrieval Augmented Generation) 기반 챗봇 시스템에 통합하는 방법을 제안합니다. 이러한 캡션은 텍스트 형태로 되어 있어 고급 언어 모델(LLM: Large Language Model) 프롬프트에 쉽게 통합할 수 있으며, 멀티미디어 내용을 적게 사용하여 멀티모달 LLM 컨텍스트 창을 효율적으로 이용할 수 있습니다.

- **Technical Details**: 정렬된 시각적 캡션(transcripts)은 기계적으로 생성된 시각적 캡션과 관련 자막 또는 자동 음성 인식(ASR: Automatic Speech Recognition) 전사본을 포함한 동영상의 시간적으로 동기화된 장면 설명입니다. 본 연구를 위해 Panda-70M 데이터세트에서 약 2,000개의 유튜브 동영상을 샘플링하여 29,259개의 동영상과 1.5M개의 영상 클립 및 캡션을 포함하는 데이터세트를 제작하였습니다. BERTScore(Zhang* et al., 2020)를 사용해 다양한 LLM이 생성한 동영상 요약이 GPT-4 Turbo의 요약과 얼마나 유사한지 평가했습니다.

- **Performance Highlights**: GPT-4 Turbo를 기반으로 한 요약과의 유사성을 평가한 결과, 다양한 구성에 따라 높은 BERTScore를 얻었습니다. 이는 텍스트 기반의 정렬된 시각적 캡션을 사용하여 LLM이 동영상 프레임 전체를 전송해야 하는 것보다 훨씬 적은 처리 대역폭으로 유사한 수준의 출력을 생성할 수 있음을 보여줍니다. 또한, 1000개의 일반 지식 질문을 사용하여 검색 증강 생성 환경에서 텍스트 임베딩의 효용성을 확인했습니다. 실험 결과, 텍스트 임베딩이 상대적으로 낮은 K에서 유의미한 정보를 찾는 데 유효함을 확인했습니다.

- **Video Enriched Chat Bot**: RAG 기반 AI 챗봇 애플리케이션의 주요 구성 요소로 정렬된 시각적 캡션을 활용하여 사용자의 질문에 관련된 답변과 해당 비디오 클립 소스를 반환하는 구조를 설명합니다. 사용자의 질의와 도구 설명을 기반으로 적절한 검색 도구가 선택되며, 선택된 엔진 도구는 검색어를 벡터화하여 벡터 데이터베이스를 검색하고, 검색된 결과를 특정 형식으로 요약하여 사용자에게 제공합니다.



### Efficient Search in Graph Edit Distance: Metric Search Trees vs. Brute Force Verification (https://arxiv.org/abs/2405.17434)
- **What's New**: 이 보고서는 그래프 유사성 검색(graph similarity search)에서 그래프 편집 거리(Graph Edit Distance, GED) 계산의 효율성을 평가합니다. 특히, 계단식 메트릭 트리(Cascading Metric Trees, CMT)와 브루트 포스 검증(brute-force verification) 방법을 비교합니다.

- **Technical Details**: 그래프 데이터는 PubChem 데이터셋을 기반으로 하며, GED 기반의 그래프 유사성 검색(GSS)에 대한 계산 복잡도(computational complexity)가 여전히 큰 도전 과제임을 강조합니다. CMT가 이론적으로는 유리할 것으로 기대되었지만, 실험 결과에서는 CMT가 일관되게 브루트 포스 방법보다 속도가 빠르지 않음을 발견했습니다.

- **Performance Highlights**: 성능 측면에서 본 연구는 CMT가 항상 브루트 포스 검증보다 우수하지 않음을 보여줍니다. 이는 GED 기반의 GSS의 효율적인 계산이 여전히 해결해야 할 과제임을 시사합니다.




### Scenario-Adaptive Fine-Grained Personalization Network: Tailoring User  Behavior Representation to the Scenario Contex (https://arxiv.org/abs/2404.09709)
Comments: Accepted by SIGIR 2024, 10 pages, 5 figures, 5 tables

- **What's New**: 새로운 랭킹 프레임워크인 Scenario-Adaptive Fine-Grained Personalization Network (SFPNet)가 제안되었습니다. 이 네트워크는 더 세분화된 방식으로 사용자의 관심을 다양한 시나리오에 적응적으로 모델링할 수 있도록 설계되었습니다. 이는 기존의 일괄적인 사용자 행동 시퀀스 가중치 조정 방식을 개선하여 각 스케나리오별로 사용자 행동의 표현을 보다 세밀하게 조정합니다.

- **Technical Details**: SFPNet은 Scenario-Tailoring Block이 연속적으로 쌓인 구조로 구성됩니다. 각 블록은 Scenario-Adaptive Module (SAM)과 Residual-Tailoring Module (RTM)의 두 주요 모듈로 구성되어 있습니다. SAM은 시나리오 정보를 통합하여 기본 특성을 조정하고, RTM은 이러한 맥락 정보를 사용하여 각 행동의 표현을 맞춤화합니다. 이러한 고도의 맞춤화는 시나리오별 사용자의 행동 표현에 정교한 맥락을 추가하여, 다양한 시나리오에서 사용자의 관심 이동을 더 정확하게 추적할 수 있습니다.

- **Performance Highlights**: 오프라인 데이터세트와 온라인 A/B 테스팅을 통한 평가 결과, SFPNet은 기존의 최신 다중 시나리오 접근 방식들보다 우수한 성능을 보여주었습니다. 이는 SFPNet이 시나리오별 맞춤형 정보를 각 사용자 행위에 고도로 통합함으로써, 다양한 시나리오의 특성과 사용자의 입장 차이를 더욱 잘 파악할 수 있기 때문입니다.



### Recall-Augmented Ranking: Enhancing Click-Through Rate Prediction  Accuracy with Cross-Stage Data (https://arxiv.org/abs/2404.09578)
Comments: 4 pages, accepted by WWW 2024 Short Track

- **What's New**: 이 논문에서는 온라인 플랫폼에서 중요한 역할을 하는 클릭률(CTR) 예측을 위한 새로운 아키텍처인 Recall-Augmented Ranking (RAR)을 제안합니다. 기존 클릭률 예측 모델은 사용자 행동 기록의 동질성과 부족함에 의존하는 경향이 있지만, RAR은 사용자 프로파일링 및 아이템 리콜을 데이터 소스로 활용하여 사용자 표현(user representations)을 풍부하게 합니다.

- **Technical Details**: RAR은 두 가지 주요 컴포넌트, Cross-Stage User & Item Selection Module과 Co-Interaction Module을 포함합니다. 이 모듈들은 유사한 사용자(look-alike users)와 리콜 아이템으로부터 정보를 효과적으로 수집하여 사용자의 표현을 강화합니다. 특히, Co-Interaction 모듈은 set-to-set 모델링을 도입하여 클릭률 예측 작업에 처음으로 적용되었습니다. 이 구조는 기존 CTR 모델과 호환되어 plug-and-play 방식으로 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: RAR은 다양한 기존 CTR 모델들과의 호환성 및 유효성을 검증하기 위해 광범위한 실험을 수행하였으며, SOTA(State of the Art) 방법들과 비교하여 우수한 성능을 보여주었습니다. 특히, 사용자 상호 연관성과 아이템 간의 관계를 고려하여 더 정확한 사용자 흥미의 폭넓은 모델링이 가능합니다.



### UniSAR: Modeling User Transition Behaviors between Search and  Recommendation (https://arxiv.org/abs/2404.09520)
Comments: Accepted by SIGIR 2024

- **What's New**: 이 논문에서는 검색(search) 및 추천(recommendation) 서비스 간의 사용자 행동 전환을 세밀하게 모델링하는 새로운 프레임워크인 UniSAR을 제안합니다. 기존 연구들이 별개로 처리하거나 세부 전환을 간과한 반면, UniSAR은 검색과 추천 사이의 다양한 행동 전환 유형을 효과적으로 모델링하여 통합 검색 및 추천 서비스를 제공합니다.

- **Technical Details**: UniSAR은 검색 및 추천 사이의 사용자 전환 행동을 세 단계 - 추출(extraction), 조정(alignment), 그리고 융합(fusion)을 통해 모델링합니다. 이 과정은 변형자(transformer)를 사용한 마스크 메커니즘, 상반되는 학습(contrastive learning)을 통한 세밀한 사용자 전환 조정, 그리고 다른 전환들을 융합하는 교차 주의(cross-attention) 메커니즘을 포함합니다. 또한, 이러한 표현은 다운스트림 검색 및 추천 모델로 입력되어 사용자에게 통합 서비스를 제공합니다.

- **Performance Highlights**: UniSAR은 두 개의 공개 데이터셋에서 실험을 통해 검증되었으며, 전통적인 단일 시나리오 모델뿐만 아니라 기존의 결합된 검색 및 추천 모델보다도 우수한 성능을 보여줌으로써 최신 기술(state-of-the-art, SOTA) 성능을 달성했습니다. UniSAR은 검색과 추천 작업을 공동으로 학습함으로써 각각의 지식을 활용하여 서로를 강화할 수 있습니다.



### Exploring the Nexus Between Retrievability and Query Generation  Strategies (https://arxiv.org/abs/2404.09473)
Comments: Accepted at ECIR 2024

- **What's New**: 본 연구에서는 문서 검색 점수에 대한 편향을 정량화하기 위해 실제 사용자 및 쿼리를 대표로 하는 질의 생성 방법의 검증을 시도하였다. 이전 연구에서는 쿼리 로그가 없을 때 자주 발생하는 콜로케이션을 사용하는 경향이 있었지만, 이 방법이 실험 결과의 재현성을 해칠 수 있다는 점을 발견하였다. 본 논문은 아티피셜 쿼리(artificial queries)와 쿼리 로그를 통해 생성된 검색 점수 간의 상관관계를 평가하여 새로운 질의 생성 기법을 제안한다.



### Competitive Retrieval: Going Beyond the Single Query (https://arxiv.org/abs/2404.09253)
- **What's New**: 이 연구는 발행인들이 검색 엔진에 의해 자신들의 문서가 높은 순위에 오를 수 있도록 여러 쿼리(query)에 대한 문서 순위를 향상시키려는 전략을 최초로 이론적 및 실증적(empirical)으로 분석합니다. 이전 연구는 단일 쿼리(single-query) 설정에 중점을 뒀으나 이 연구에서는 다중 쿼리(multiple queries)로의 확장을 탐구하고 그로 인한 동태적인 상황을 제시합니다.

- **Technical Details**: 게임 이론적(game theoretic) 분석을 사용하여 다중 쿼리를 목표로 문서를 수정할 때 균형(equilibrium)이 반드시 존재하지 않음을 증명합니다. 이는 문서 수정 경쟁이 끝나지 않는 가능성을 암시합니다. 또한, 특징 기반 순위 결정(feature-based ranking)과 신경 순위 결정자(neural ranker)를 사용하여 다중 쿼리에 대한 문서 순위를 향상시키는 것이 얼마나 어려운지를 비교 연구했습니다. 신경 순위 결정자는 다양한 쿼리에 대해 더 다양한 순위를 생성했으며, 이로 인해 발행인들이 문서 순위를 높이기가 더 어려웠습니다.

- **Performance Highlights**: 다중 쿼리 설정에서는 발행인들이 이전에 높은 순위를 차지한 문서의 내용을 모방하는 경향이 여전히 존재했으나, 생성적 AI 도구(generative AI tools)를 사용하지 못할 때 더 두드러졌습니다. 발행인들이 다수의 쿼리에 걸쳐 문서를 개선하기 위해 경쟁하는 새로운 형태의 순위 경쟁(ranking competitions)을 조직했습니다. 또한, 다른 쿼리로부터 유도된 정보를 활용하여 어느 문서가 다음 순위에서 가장 높은 순위가 될지 예측하는 방법을 개발하여 효과를 입증했습니다.



### Approximate Cluster-Based Sparse Document Retrieval with Segmented  Maximum Term Weights (https://arxiv.org/abs/2404.08896)
- **What's New**: 이 논문은 클러스터(cluster) 기반 검색을 재검토하며, 학습된 희소 표현을 사용하여 인덱스를 여러 그룹으로 분할하고 온라인 추론 시 클러스터와 문서 수준에서 부분적으로 인덱스를 건너뛰는 방식을 제안합니다. 이는 'Segmented Maximum Term Weights'를 제어하는 두 가지 매개변수를 통해 검색의 정밀성과 효율성을 향상시킬 수 있습니다.

- **Technical Details**: 이 논문은 클러스터 수준 최대 가중치 세분화(segmentation)를 제안하여, 클러스터를 오프라인에서 부분들로 나누고 세분화된 용어 가중치 정보를 수집합니다. 이를 통해 클러스터 기반 검색이 순위 점수 경계 추정을 더욱 강화하고 저점수 문서만 포함하는 클러스터를 정확히 건너뛸 수 있도록 합니다. 또한, 임계값 추정(threshold estimation)의 근사 정확성을 분석하고, 두 매개변수를 통한 임계값 제어 방안을 제안하여 검색의 안전성을 확보합니다. SPLADE와 같은 모델 아래 희소 벡터의 대응물로 밀집 토큰 임베딩을 사용하여 문서를 군집화하고 이상의 방식을 지원합니다.

- **Performance Highlights**: MS MARCO와 BEIR 데이터셋을 사용한 폭넓은 평가를 통해 제안된 방식의 효과를 검증하였습니다. 이 방법을 사용하여 SPLADE, uniCOIL, LexMAE 같은 세 가지 학습된 희소 검색 모델의 성능을 향상시킬 수 있음을 보여줍니다. 또한, 'anytime early termination'과 'static index pruning'과 같은 두 가지 효율 최적화 기술과 함께 사용될 때의 이점을 연구합니다.



### Countering Mainstream Bias via End-to-End Adaptive Local Learning (https://arxiv.org/abs/2404.08887)
Comments: ECIR 2024

- **What's New**: 이 논문에서는 협력적 필터링(Collaborative Filtering, CF) 기반 추천 시스템에서 발생하는 '메인스트림 편향(Mainstream Bias)' 문제를 해결하기 위해 새로운 end-to-end Adaptive Local Learning (TALL) 프레임워크를 제안합니다. 메인스트림 편향은 메인스트림 사용자를 선호하며 니치 사용자에 대한 추천의 품질이 낮아지는 현상을 말합니다. 연구진은 이 문제의 두 가지 주요 원인을 식별하였으며, 이에 대응하기 위한 새로운 접근 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 메인스트림 편향의 두 가지 주요 원인인 '불일치 모델링(Discrepancy Modeling)' 문제와 '비동기 학습(Unsynchronized Learning)' 문제를 해결하고자 합니다. 첫째, TALL은 손실 기반의 전문가 혼합 모듈(Mixture-of-Experts)을 사용하여 사용자 맞춤형 로컬 모델을 제공합니다. 둘째, 적응적 가중치 모듈을 통해 다양한 사용자의 학습 속도를 동기화합니다. 이러한 기술적 특성은 다양한 니치 사용자에게도 높은 품질의 추천을 가능하게 하여 메인스트림 사용자와의 성능 격차를 줄입니다.

- **Performance Highlights**: TALL 프레임워크는 메인스트림 및 니치 사용자 모두에게 높은 품질의 추천을 제공하여, 주요 경쟁 모델 대비 니치 사용자의 유틸리티를 6.1% 향상시키는 성능을 보여주었습니다. 이는 Rawlsian Max-Min 공정성 원칙에 기반하여 메인스트림 편향을 효과적으로 줄이는 결과를 나타냅니다. 이 프레임워크의 효과적인 디바이스(debiasing) 능력이 입증되었으며, 연구진은 해당 코드와 데이터를 공개하였습니다.



### Misinformation Resilient Search Rankings with Webgraph-based  Interventions (https://arxiv.org/abs/2404.08869)
- **What's New**: 이 연구는 뉴스 도메인에 대한 신뢰도를 평가하고 가짜 뉴스 도메인의 검색 엔진 조작을 방지하기 위한 새로운 개입 방법을 소개합니다. 이 방법들은 공정성(fairness), 일반성(generality), 대상 지정성(targeted), 확장성(scalability)의 원칙에 기반을 두고 있으며, 신뢰할 수 없는 뉴스 도메인을 페널티하면서도 신뢰할 수 있는 도메인에 대한 트래픽은 유지하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 SEO 도구에서 추출한 소규모 웹그래프 데이터에 대한 실험을 통해 가짜 뉴스 도메인을 대상으로 한 두 가지 개입 클래스의 효과를 설계하고 평가했습니다. 이어서, 93.9M 개의 도메인과 1.6B 개의 엣지를 포함하는 대규모 웹그래프에서 이 개입 방법을 일반화하고 검증했습니다. 이들은 도메인의 PageRank 변화를 계산하여 개입의 효과를 검증하고, 안티-트러스트랭크(Anti-TrustRank) 점수를 포함시킨 Personalized PageRank 알고리즘 변형과의 비교를 통해 결과를 개선했습니다.

- **Performance Highlights**: 소규모 웹그래프 실험에서, 가장 효과적인 개입은 신뢰할 수 없는 도메인의 예상 트래픽과 랭킹을 각각 35% 및 27% 감소시켰습니다. 반면, 신뢰할 수 있는 도메인은 각각 11% 및 8%의 트래픽 및 랭킹 감소를 보였습니다. 대규모 웹그래프 실험에서도 유사한 유리한 상충 관계(Trade-off)가 관찰되었으나 효과는 감소되었으며, 신뢰할 수 없는 도메인의 PageRank 중심성(Centrality)이 10% 감소하는 반면 신뢰할 수 있는 도메인은 2% 감소했습니다.



### Improving Technical "How-to" Query Accuracy with Automated Search  Results Verification and Reranking (https://arxiv.org/abs/2404.08860)
Comments: 12 pages, 2 columns, 3 figures

- **What's New**: 이 논문은 기술 지원 검색 결과의 정확도와 관련성을 향상시키는 새로운 접근 방식을 소개합니다. Android 환경에서 검색 결과에 포함된 단계별 지침을 자동으로 해석하고 실행할 수 있는 AI 에이전트를 개발하였고, 에이전트의 결과를 기반으로 검색 결과를 재정렬하는 메커니즘을 통합했습니다.

- **Technical Details**: 이 연구는 'How-to' 쿼리(How-to queries)에 특화된 세 단계 솔루션을 제안합니다. 첫 번째 단계는 각 검색 결과 페이지에서 단계별 지침을 추출하는 지침 추출 모델(Instruction Extraction Model)을 포함합니다. 두 번째 단계에서는, 제안된 AI 에이전트가 Android 기기에서 지침을 실행하거나 시뮬레이션하여 그 품질을 검증합니다. 마지막 단계에서는 실행 정보를 기반으로 검색 결과를 재정렬합니다. 이는 검색 엔진이 사용자의 질문에 대해 더 효과적이고 신뢰할 수 있는 해결책을 제공할 수 있게 합니다.

- **Performance Highlights**: 초기 실험 결과, 이 방법은 기존의 선도적인 검색 엔진(Google)을 능가하는 성능 향상을 보여주었습니다. 특히, 검증 및 재정렬 과정을 거친 검색 결과가 사용자가 실제 문제를 해결하는데 더 효과적이라는 것이 입증되었습니다. 이는 검색 엔진에서 'How-to' 질의에 대한 기술 지원 검색 결과의 순위를 결정하는 방식에 패러다임 변화를 제안합니다.



### LazyDP: Co-Designing Algorithm-Software for Scalable Training of  Differentially Private Recommendation Models (https://arxiv.org/abs/2404.08847)
- **What's New**: 이 연구는 차등 개인정보 보호(Differential Privacy, DP)를 사용하여 추천 시스템(Recommender Systems, RecSys)을 훈련하는 과정에서의 컴퓨터 성능상의 병목 현상을 분석하고, 이를 해결하기 위해 LazyDP라는 새로운 알고리즘-소프트웨어 공동 설계를 제안합니다. LazyDP는 기존의 DP-SGD 방법에 비해 평균 119배의 훈련 처리량 향상을 제공합니다.

- **Technical Details**: LazyDP는 각 훈련 반복에서 잡음을 추가하여 모든 임베딩 테이블에 가우시안 잡음(Gaussian noise)을 추가하는데, 이는 전통적인 SGD의 희소 그래디언트 업데이트를 밀집된 '잡음이 있는' 그래디언트 업데이트로 변환합니다. 이로 인해 발생하는 계산 및 메모리 대역폭 제한 문제를 해결하기 위해, LazyDP는 잡음 업데이트를 지연시키고, 그 과정에서 누적된 잡음을 효율적으로 샘플링하는 방법을 사용합니다.

- **Performance Highlights**: LazyDP를 사용함으로써, 임베딩 테이블 전체에 대한 잡음 업데이트의 메모리 대역폭 요구 사항이 크게 감소하고, 잡음 샘플링의 계산 오버헤드가 크게 줄어들어, 전체적인 훈련 속도가 상당히 향상됩니다. 이는 기존의 DP-SGD 시스템에 비해 119배 빠른 훈련 처리량을 달성한다는 결과를 포함합니다.



### Measuring the Predictability of Recommender Systems using Structural  Complexity Metrics (https://arxiv.org/abs/2404.08829)
Comments: Accepted at WWW-24 Workshop: DCAI Data-centric Artificial Intelligence

- **What's New**: 이 연구는 추천시스템(Recommender systems, RS)의 예측 가능성을 측정하기 위한 데이터 기반 메트릭을 소개합니다. 협업 필터링(Collaborative Filtering, CF) 데이터셋의 구조적 복잡성을 측정하기 위해 특별한 전략을 채택하고, 이는 RS의 성능 이해와 예측의 정확도에 대한 통찰을 제공합니다.

- **Technical Details**: 이 연구에서는 특이값 분해(Singular Value Decomposition, SVD)와 행렬 분해(Matrix Factorization, MF)를 활용한 두 가지 전략을 제시하여 RS의 구조적 복잡성을 측정합니다. 데이터를 변형시키고 변형된 버전의 예측을 평가함으로써 SVD 단일 벡터가 나타내는 구조적 일관성을 탐구합니다. 높은 구조적 복잡성을 지닌 데이터의 무작위 변형은 구조를 변경하지 않는다는 가정 하에 연구가 진행됩니다.

- **Performance Highlights**: 실제 데이터셋에서 최고 성능을 보이는 예측 알고리즘의 정확도와 높은 상관관계를 보인 메트릭을 통해, 이 메트릭들이 RS의 예측 가능성을 효과적으로 측정할 수 있음을 입증하였습니다. 이 연구에서 제안하는 메트릭들은 실제 및 인공 데이터셋을 사용한 테스트에서 그 유효성이 검증되었습니다.



### The Elephant in the Room: Rethinking the Usage of Pre-trained Language  Model in Sequential Recommendation (https://arxiv.org/abs/2404.08796)
Comments: 10 pages

- **What's New**: 이 연구는 PLM(Pre-trained Language Models)을 연속 추천(Sequential Recommendation, SR)에 효과적이고 경제적으로 활용하기 위한 새로운 방법을 제시합니다. 연구진은 기존의 PLM 기반 SR 모델에서 PLM의 능력이 충분히 활용되지 않고 있음을 발견하고, 더 가벼운 방식으로 PLM을 활용하여 SR의 성능을 향상시킬 수 있는 방법을 탐구했습니다.

- **Technical Details**: 연구는 행동 기반 튜닝된 PLM(behavior-tuned PLM)을 사용하여 기존 ID 기반 SR 모델의 아이템 초기화에 활용하는 것이 가장 경제적인 PLM 기반 SR 프레임워크라는 것을 발견했습니다. 이 방법은 추가적인 추론 비용을 발생시키지 않으면서 눈에 띄는 성능 향상을 이루어 냈습니다. 또한, PLM의 중요한 시퀀스 모델링 능력과 추론 능력을 SR에 효과적으로 적용할 수 있는 간단하고 보편적인 프레임워크를 개발하였습니다.

- **Performance Highlights**: 다양한 데이터셋에서의 실험 결과, 새로운 프레임워크는 기존의 SR 모델들과 SOTA(State-of-the-Art) PLM 기반 SR 모델들을 뛰어넘는 성능을 보여주었습니다. 이는 추가적인 추론 비용 없이도 이루어졌습니다. 이는 PLM의 사용을 최적화하여 SR의 정확성과 효율성을 크게 향상시킬 수 있음을 입증합니다.



### Musical Listening Qualia: A Multivariate Approach (https://arxiv.org/abs/2404.08694)
- **What's New**: 새로운 음악 자극을 듣고 평가한 프랑스와 미국 참가자들의 연구에서, 언어적 수단(형용사)과 정량적 음악 차원을 사용하여 응답이 달라질 수 있음을 발견했습니다. 이 연구는 실험적 통제를 완화하면서 통계적 엄밀성을 유지하는 연구 방법론에 대한 사례 연구로 사용됩니다.

- **Technical Details**: 참가자들은 새로운 음악 자극에 대해 형용사와 정량적 음악 차원(quantitative musical dimensions)을 사용하여 평가했습니다. 데이터 분석은 대응분석(CA), 계층적 군집 분석(HCA), 다중 요인 분석(MFA), 그리고 부분 최소 제곱 상관분석(PLSC)을 통해 이루어졌습니다.

- **Performance Highlights**: 분석 결과, 프랑스와 미국 참가자들은 형용사를 사용하여 음악 자극을 묘사할 때 차이를 보였지만, 정량적 차원을 사용할 때는 그러한 차이가 나타나지 않았습니다. 이는 언어적 도구와 통계적 접근 방식이 음악 평가에서 중요한 역할을 함을 시사합니다.



### Enhancing Adaptive Video Streaming through Fuzzy Logic-Based Content  Recommendation Systems: A Comprehensive Review and Future Directions (https://arxiv.org/abs/2404.08691)
Comments: 7 pages

- **What's New**: 이 리뷰 논문은 적응형 비디오 스트리밍(adaptive video streaming)을 위한 콘텐츠 추천 시스템에 맞추어진 퍼지 로직(fuzzy logic)의 통합을 탐구합니다. 퍼지 로직은 불확실성과 부정확성을 처리하는 데 강점이 있고, 사용자 선호도와 맥락 요인의 동적 특성을 모델링하고 수용하는 데 유망한 프레임워크를 제공합니다.

- **Technical Details**: 이 논문은 적응형 비디오 스트리밍의 진화를 깊이 있게 논의하고, 전통적인 콘텐츠 추천 알고리즘들을 리뷰하며, 이러한 시스템의 적응성을 향상시키기 위한 해결책으로 퍼지 로직을 소개합니다. 사용자의 선호와 맥락적 정보를 동적으로 적응하는 기능을 강화하기 위해 퍼지 로직을 통합하는 방법이 자세히 설명되어 있습니다.

- **Performance Highlights**: 퍼지 로직의 통합은 사용자 만족도와 시스템 성능을 향상시키는 효과를 가지며, 다양한 사례 연구와 응용을 통해 이러한 효과가 강조되고 있습니다. 이 리뷰는 퍼지 로직 통합과 관련된 도전과제를 다루며, 이 접근법을 더욱 발전시키기 위한 미래 연구 방향을 제안합니다.



### A Survey of Reasoning for Substitution Relationships: Definitions,  Methods, and Directions (https://arxiv.org/abs/2404.08687)
- **What's New**: 이 연구는 제품 간 대체 관계를 이해하고 예측하기 위해 머신 러닝 (Machine Learning), 자연어 처리 (Natural Language Processing, NLP), 및 기타 기술을 포괄적으로 분석합니다. 다양한 도메인에서 대체품을 정의하고 대체 관계를 나타내며 학습하는 방법론을 비교함으로써, 제품 대체 추천 시스템의 개인화와 정확도를 향상시킬 수 있는 방법론적 기반을 제공합니다.

- **Technical Details**: 제품 대체 관계는 '대체 가능성(Substitutability)'이라는 속성을 가진 제품을 참조하는 것입니다. 예를 들어, 스마트폰, 식품 또는 제약 제품에서 다른 제품으로 대체할 때 그 제품들이 가진 기능성이나 특성이 비슷해야 합니다. 데이터 분석은 공동 브라우징 행위 (Co-browsing behavior), 가격 탄력성 (Price Elasticity), 사용자 리뷰 등 다양한 방법을 포함하여 대체품을 식별합니다.

- **Performance Highlights**: 대체 추천 시스템은 개별 소비자의 필요와 목표에 맞게 최상의 제품 선택을 가능하게 하며, 대체품과 보완품 간의 관계를 이해하는 것이 추천 시스템의 정확도와 개인화를 향상시키는 데 중요합니다. 연구는 또한 온라인 제품 리뷰가 소비자의 구매 결정에 끼치는 영향을 탐구하여, 긍정적인 리뷰가 대체 제품에 대한 선호도를 증가시킬 수 있음을 시사합니다.



### Access to Library Information Resources by University Students during  COVID-19 Pandemic in Africa: A Systematic Literature Review (https://arxiv.org/abs/2404.08682)
- **What's New**: 이 연구는 COVID-19 대유행 기간에 대학생의 도서관 정보 자원 접근을 조사했습니다. 특히, 대학 도서관이 정보 자원의 원활한 제공을 위해 어떤 조치를 취했는지, 어떤 기술 도구를 사용하여 정보 자원에 접근을 도왔는지를 파악하고, 학생들이 정보 자원에 접근하는 데 직면한 도전들을 조사했습니다. 주로 PRISMA (PRISMA) 가이드라인을 사용한 체계적 문헌 검토 방식을 사용하여 관련 연구를 조사했습니다.

- **Technical Details**: 이 연구에서는 Scopus, Emerald, Research4life, Google Scholar와 같은 학술 데이터베이스에서 관련 문헌을 검색하기 위해 키워드 검색 전략이 사용되었습니다. 연구 결과, 대유행 기간 동안 아프리카의 많은 대학 도서관이 캠퍼스 밖에서 전자 자원 접근을 확대하고, 가상 참조 서비스(Virtual Reference Services), 유통 및 대여 서비스를 포함해 다양한 접근 방식을 채택했습니다. 또한, 사회 미디어(Social Media), 도서관 웹사이트, 이메일, 비디오 컨퍼런싱 같은 다양한 디지털 기술 도구를 사용했습니다.

- **Performance Highlights**: 연구 결과, 인터넷 서비스와 ICT 기기에 대한 접근성 제한, 불충분한 전자 도서관 컬렉션, 그리고 디지털 및 정보 리터러시 부족이 팬데믹 동안 이용자들이 직면한 주요 도전 과제였습니다. 이 연구는 디지털 시대에 필수적인 자원인 ICT 기반 시설에의 투자와 전자 자원 컬렉션의 확장을 권장합니다.



### Information Retrieval with Entity Linking (https://arxiv.org/abs/2404.08678)
- **What's New**: 이 연구에서는 희소 검색기(sparse retrievers)의 성능을 향상시키기 위해 연결된 엔티티들(linked entities)로 쿼리(queries)와 문서(documents)를 확장하는 새로운 접근 방식을 제안합니다. 특히, 엔티티 이름(entity names)을 명시적(explicit) 형식과 해시된(hashed) 형태로 확장하는 두 가지 형식을 사용합니다. 제로샷(zero-shot) 밀도 기반(dense) 엔티티 링킹 시스템(entity linking system)을 사용하여 엔티티 인식(entity recognition)과 중의성 해소(entity disambiguation)를 수행합니다. 이 연구는 희소 검색기와 밀도 검색기(dense retrievers) 간의 효율성 격차를 좁힐 수 있는 가능성을 제시합니다.

- **Technical Details**: 엔티티 확장(entity expansion)은 MS MARCO 데이터셋을 사용하여 실험되었으며, MonoT5와 DuoT5에 의해 재순위된 qrels을 포함하여 세 가지 유형의 관련성 판단(relevance judgments)을 사용했습니다. 초기 단계 검색(early stage retrieval)에 초점을 맞추고, recall@1000을 사용하여 결과를 평가했습니다.

- **Performance Highlights**: 엔티티 확장을 통해 희소 검색기(BM25)의 리콜(recall)이 향상되었지만, 밀도 검색기의 총체적인 성능에는 큰 변화가 없었습니다. 이는 엔티티 링크(entity linking)와 정보 검색(information retrieval) 사이의 연결 가능성을 탐구하고 문제의 복잡성과 도전을 다루는 초기 연구 단계에서 중요한 발견입니다.



### Navigating the Evaluation Funnel to Optimize Iteration Speed for  Recommender Systems (https://arxiv.org/abs/2404.08671)
- **What's New**: 이 논문은 추천 시스템(evaluation funnel) 평가를 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 성공적인 결과를 정의하는 방법을 분해하고, 빠른 실패 인식을 가능하게 하여, 비효율적인 반복을 신속하게 걸러내고 개선할 수 있도록 돕습니다. 또한, 오프라인(offline) 평가와 온라인(online) 평가 방법을 포함하여 다양한 평가 방법들의 장단점을 비교하고, 이들이 어떻게 서로 보완적으로 작용하는지 설명합니다.



### Combining PatternRank with Huffman Coding: A Novel Compression Algorithm (https://arxiv.org/abs/2404.08669)
- **What's New**: 본 논문에서는 Android 백업 데이터를 효과적으로 압축하기 위해 특별히 설계된 새로운 압축 전략인 PatternRank 알고리즘을 소개합니다. PatternRank는 패턴 인식(pattern recognition) 및 순위 매기기(ranking)과 함께 허프만 코딩(Huffman coding)을 결합하여 빈번하게 발생하는 긴 패턴을 더 짧은 코드로 대체함으로써 데이터를 효율적으로 압축합니다.

- **Technical Details**: PatternRank 알고리즘은 두 가지 버전으로 구현되었습니다. 첫 번째 버전은 동적 패턴 추출(dynamic pattern extraction) 및 순위 매기기에 초점을 맞추고, 두 번째 버전은 Android 백업에서 흔히 발견되는 패턴, 특히 XML 파일 내의 패턴에 최적화된 사전 정의된 사전(pre-defined dictionary)을 통합합니다. 이러한 맞춤형 접근 방식은 PatternRank가 전통적인 압축 방법인 GZIP에 비해 압축 비율(compression ratio) 및 속도(compression speed) 측면에서 우수한 성능을 발휘하도록 보장합니다.

- **Performance Highlights**: 압축 성능에 대한 비교 연구는 GZIP, PatternRank v1, PatternRank v2, 그리고 PatternRank-Huffman 조합 방식을 포함하여 수행되었습니다. 이 연구는 PatternRank가 Android 백업 패키지의 데이터 요구를 관리하는 데 있어 우수한 효율성과 잠재력을 가지고 있음을 강조합니다. 특히 XML 데이터를 많이 포함하는 백업에서 더 뛰어난 성능을 나타낼 것으로 예상됩니다.



### A Comprehensive Survey on AI-based Methods for Patents (https://arxiv.org/abs/2404.08668)
- **What's New**: 이 논문은 인공 지능(Artificial Intelligence, AI) 및 기계 학습(machine learning)의 최신 발전을 특허 분석(patent analysis) 및 혁신 분야에 적용하는 방안을 조사합니다. 이전의 조사와 달리, 텍스트 데이터(text data)와 이미지 데이터(image data) 모두에 작동하는 방법을 포함하며, 특허 수명 주기(patent life cycle)의 작업과 AI 방법의 특성을 기반으로 한 새로운 분류 체계(taxonomy)를 소개했습니다.

- **Technical Details**: 이 조사는 2017년부터 2023년까지 26개의 출판 장소에서 발표된 40개 이상의 논문을 포함하여 특허 분류(patent classification), 검색(retrieval), 품질 분석(quality analysis), 및 생성(generation)에 대한 AI 기반 방법을 요약합니다. 특히, 다중 레이블 분류(multi-label classification), 표현 학습(representation learning), 그리고 생성 AI(generative AI) 도구가 특허 작성과 기술 언어 설명에 효과적으로 사용될 수 있다는 점을 강조합니다.

- **Performance Highlights**: AI 기반 도구는 특허 분류에서 국제 특허 분류(IPC) 및 협력 특허 분류(Cooperative Patent Classification, CPC)의 계층적 체계에 대한 다중 레이블 분류를 돕고, 검색 과정에서 새로운 특허 신청의 참신성을 평가하며 침해를 피할 수 있습니다. 이러한 도구들은 또한 품질 분석에서 특허의 가치를 평가하는 데 중요한 역할을 하며, 특허 생성에 있어서는 정확하고 기술적인 언어 설명을 생성하여 인간 자원의 최적화와 정밀성을 높입니다.



### FewUser: Few-Shot Social User Geolocation via Contrastive Learning (https://arxiv.org/abs/2404.08662)
Comments: 17 pages, 3 figures, 8 tables, submitted to ECML-PKDD 2024 for review

- **What's New**: 새로운 FewUser 프레임워크는 Few-shot 소셜 사용자 지리 위치 추정을 위해 제안되었습니다. 이 프레임워크는 사용자와 위치 간의 대조 학습(contrastive learning) 전략을 통합하여 훈련 데이터가 제한적이거나 없는 상황에서도 지리적 위치 성능을 향상시킵니다. 또한, 언어 모델과 지리적 프롬프팅 모듈을 사용하여 위치 정보의 인코딩을 강화합니다.

- **Technical Details**: FewUser는 사전 훈련된 언어 모델(PRE-TRAINED LANGUAGE MODEL, PLM)과 사용자 인코더(user encoder)를 사용하여 다양한 소셜 미디어 입력을 처리하고 융합하는 사용자 표현 모듈(user representation module)을 특징으로 합니다. 지리적 프롬프팅 모듈은 하드 프롬프트(hard prompts), 소프트 프롬프트(soft prompts), 반-소프트 프롬프트(semi-soft prompts)를 포함하여 PLM의 지식과 지리 데이터 간의 차이를 줄입니다. 대조적 학습은 대조 손실(contrastive loss)과 매칭 손실(matching loss)을 통해 구현되며, 하드 네거티브 마이닝 전략(hard negative mining strategy)으로 학습 과정을 개선합니다.

- **Performance Highlights**: FewUser는 제로-샷(zero-shot) 및 다양한 퓨-샷(few-shot) 설정에서 최신 방법들을 크게 능가하여 TwiU와 FliU 데이터셋에서 각각 26.95% 및 41.62%의 절대적 개선을 달성하였습니다. 이러한 성능은 풍부한 메타데이터를 포함하는 TwiU와 FliU 데이터셋에서의 광범위한 실험을 통해 입증되었습니다.



### How Does Message Passing Improve Collaborative Filtering? (https://arxiv.org/abs/2404.08660)
- **What's New**: 이 연구는 협업 필터링(Collaborative Filtering, CF)에서 메시지 전달이 어떻게 도움이 되는지를 체계적으로 조사합니다. 그 결과, 메시지 전달이 이웃으로부터 추가적인 표현(representation)을 통해 전달되는 것이 모델의 역전파(back-propagation) 동안 이웃의 표현에 대한 추가적인 그래디언트 업데이트보다 성능 향상에 더 중요함을 밝혔습니다. 또한, 저차수(low-degree) 노드에 더 큰 도움이 되는 것으로 나타났습니다.

- **Technical Details**: 연구팀은 Test-time Aggregation for Collaborative Filtering, 즉 TAG-CF를 제시합니다. 이는 훈련 시간 동안 메시지 전달을 요구하지 않고, 추론 시점에 단 한 번의 메시지 전달을 실행하여 다양한 CF 감독 신호(supervision signals)로 추론된 표현을 향상시키는 테스트 시간 증강(test-time augmentation) 프레임워크입니다. TAG-CF는 계산 비용이 매우 적으면서도 추천 성능을 일관되게 개선할 수 있는 강력한 도구임이 입증되었습니다.

- **Performance Highlights**: TAG-CF는 그래프를 사용하지 않는 CF 방법들을 최대 39.2%까지 성능을 향상시켰으며, 특히 새로운 사용자(cold users)에 대한 추천에서 높은 향상을 보였습니다. 또한, 전체 사용자(all users)에 대해서도 31.7%의 성능 향상을 달성했습니다. 이는 기존의 그래프 강화 CF 방법들과 비교할 때 매우 낮은 훈련 시간(1% 미만)으로 비슷하거나 더 나은 성능을 제공합니다.



### Artificial Intelligence enhanced Security Problems in Real-Time Scenario  using Blowfish Algorithm (https://arxiv.org/abs/2404.09286)
- **What's New**: 이 연구에서는 클라우드 인프라(Cloud Infrastructure)에서의 보안 문제와 유출을 전반적으로 다루고 있습니다. 특히, '하이브리드 암호화(Hybrid Encryption)' 기법을 사용하여 보안을 강화하는 새로운 접근 방식을 제안하고 있습니다. 클라우드 컴퓨팅(Cloud Computing)의 폭발적인 성장을 감안할 때, 이러한 보안 연구는 매우 중요한 시기에 이루어졌습니다.

- **Technical Details**: 이 연구에서는 클라우드 보안에 관한 다양한 모델을 검토하고 있으며, 이에는 기밀성(Confidentiality), 진정성(Authenticity), 접근성(Accessibility), 데이터 무결성(Data Integrity), 및 복구(Recovery)가 포함됩니다. 하이브리드 암호화 기법은 일반적으로 두 가지 이상의 암호화 방식을 혼합하여 사용함으로써 보안을 강화합니다.

- **Performance Highlights**: 하이브리드 암호화가 클라우드 서비스의 보안 강화에 기여할 수 있는 주요 방법 중 하나로 강조되고 있으며, 이는 특히 개인 정보의 보호와 데이터 무결성을 유지하는 데 중요한 역할을 합니다. 연구 결과는 클라우드 기반 시스템에 대한 보안 위협을 효과적으로 줄이는 데 하이브리드 암호화 방식의 유용성을 시사합니다.




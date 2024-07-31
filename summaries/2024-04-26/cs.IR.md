### MMGRec: Multimodal Generative Recommendation with Transformer Mod (https://arxiv.org/abs/2404.16555)
- **What's New**: 새로운 MMGRec 모델이 제안되었습니다. 이 모델은 멀티모달 추천에서 생성적 패러다임(generative paradigm)을 도입하여, 사용자의 과거 상호 작용 기반으로 선호 아이템의 Rec-ID를 예측합니다. 이는 효율적이고 정확한 추천을 가능하게 하는 기술적 진보를 대표합니다.

- **Technical Details**: MMGRec은 계층적 양자화 방법 Graph RQ-VAE를 사용하여 아이템의 멀티모달 및 CF 정보에서 Rec-ID를 할당합니다. 각 아이템의 고유한 식별자로서의 Rec-ID는 의미 있는 토큰의 튜플로 구성됩니다. 이후 Transformer 기반 추천자를 훈련시켜 사용자의 과거 상호 작용 시퀀스를 기반으로 사용자가 선호하는 아이템의 Rec-ID를 생성합니다. 또한, Transformer는 비순차적 상호 작용 시퀀스를 처리하기 위해 관계 인식 자기 주의 메커니즘(relation-aware self-attention mechanism)을 사용하여 절대 위치 인코딩을 대체하는 요소 쌍 관계를 탐색합니다.

- **Performance Highlights**: MMGRec는 세 개의 공개 데이터셋에서 광범위한 실험을 통해 평가되었으며, 최신 방법과 비교하여 뛰어난 성능을 보여주었습니다. 추천 효율성과 정확도에서 탁월한 결과를 보여주었으며, 이는 MMGRec의 합리성과 효과성을 입증합니다.



### RE-RecSys: An End-to-End system for recommending properties in  Real-Estate domain (https://arxiv.org/abs/2404.16553)
- **What's New**: 부동산 추천 시스템인 RE-RecSys는 새로이 제안된 모델로 실제 산업 환경에서 시행되었습니다. 이 시스템은 사용자를 4가지 카테고리로 분류하여 각기 다른 추천 기법을 적용합니다: i) 이력 데이터가 없는 콜드 스타트 (cold-start) 사용자, ii) 단기간 유저, iii) 장기간 유저, iv) 단기-장기 유저.

- **Technical Details**: RE-RecSys는 콜드 스타트 사용자를 위해 규칙 기반 엔진, 단기 사용자를 위해서는 내용 기반 필터링(content-based filtering), 장기 및 단기-장기 사용자를 위해서는 내용과 협업 필터링(collaborative filtering) 방식의 조합을 사용하며, 이는 실제로 운용 가능합니다. 특히, 클릭률(conversion rates)을 기반으로 다양한 유저 상호 작용에 대한 중요도를 조정하는 새로운 가중치 구성 방식을 도입했습니다.

- **Performance Highlights**: RE-RecSys는 인도의 주요 부동산 플랫폼에서 수집된 실제 부동산 및 클릭스트림 데이터를 기반으로 효율성을 검증했습니다. 이 시스템은 평균 40ms 미만의 지연 시간으로 초당 1000 요청을 처리할 수 있는 성능을 보여주며, 이는 실제 적용이 가능함을 시사합니다.



### OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search (https://arxiv.org/abs/2404.16260)
Comments: 8 pages, 5 figures, to be published as an oral paper in TheWebConf Industry Track 2024

- **What's New**: 본 논문에서는 Pinterest 검색을 위한 이해도를 극대화하기 위해 'OmniSearchSage'라는 새로운 시스템을 소개합니다. 이 시스템은 통합된 쿼리 임베딩(query embedding)을 학습하고, 이를 핀(pin) 및 제품(product) 임베딩과 결합하여 Pinterest의 생산 검색 시스템에서 $>8\%$의 관련성, $>7\%$의 참여도, 그리고 $>5\%$의 광고 클릭률(ads CTR) 향상을 이뤄낸 주요 성과를 달성했습니다.

- **Technical Details**: OmniSearchSage는 다양한 텍스트(이미지 캡션, 과거 참여도, 사용자 큐레이션 보드에서 파생된)를 활용하여 엔티티 표현을 풍부하게 합니다. 이 시스템은 멀티태스크 학습(multi-task learning) 환경에서 단일 검색 쿼리 임베딩을 생성하여 기존의 핀 및 제품 임베딩과 호환되도록 설계되었습니다. 속성(feature)의 가치를 평가하기 위한 제거 연구(ablation studies)를 실시하여 통합 모델의 효과를 단독 모델과 비교 분석하였습니다.

- **Performance Highlights**: 이러한 업그레이드를 통해 OmniSearchSage는 Pinterest 검색 스택 전체에서, 검색 결과를 검색하여 랭킹을 매기는 단계에 이르기까지 이르는 과정에서 초당 300,000개의 요청을 처리할 수 있는 스케일을 실현하였고, 저지연(low latency) 서비스를 제공합니다. 다양한 기능을 통합한 결과, 이 시스템은 기존의 독립 실행형(standalone) 시스템들에 비해 뛰어난 성능을 보여주었습니다.



### Advancing Recommender Systems by mitigating Shilling attacks (https://arxiv.org/abs/2404.16177)
Comments: Published in IEEE, Proceedings of 2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT)

- **What's New**: 이 논문은 '쉴링 공격(shilling attacks)'으로 알려진 추천 시스템에 대한 공격을 감지하는 새로운 알고리즘을 제안합니다. 이 공격들은 시스템의 추천을 왜곡하기 위해 어떤 아이템을 부각시키거나 무너뜨리는 목적으로 수행됩니다. 저자들은 이런 프로파일의 효과를 분석하고, 그들을 정확하게 탐지할 수 있는 방법을 연구합니다.

- **Technical Details**: 이 연구에서는 협업 필터링(collaborative filtering)을 사용하는 추천 시스템에 영향을 미치는 쉴링 프로파일을 정확히 탐지할 수 있는 알고리즘을 개발하였습니다. 협업 필터링은 사용자 선호도에 따라 콘텐츠를 분류하는데 효과적이지만, 이 방법은 시스템을 쉴링 공격에 취약하게 만듭니다. 제안된 알고리즘은 이러한 취약점을 해결할 수 있는 가능성을 탐색합니다.

- **Performance Highlights**: 이 알고리즘은 쉴링 프로파일을 탐지하는데 있어 높은 정확도를 보여줍니다. 이로 인해 추천 시스템의 신뢰도와 보안이 강화될 수 있으며, 사용자 경험을 개선할 수 있는 기여를 합니다.



### Retrieval and Distill: A Temporal Data Shift-Free Paradigm for Online  Recommendation System (https://arxiv.org/abs/2404.15678)
- **What's New**: 새로운 연구에서는 추천 시스템에서 시간적 데이터 변화(temporal data shift) 문제를 해결하기 위해 '인출 및 증류 패러다임(Retrieval and Distill, RAD)'을 제안하였습니다. 이 방식은 기존의 데이터 분포와 온라인 데이터의 분포가 일치하지 않는 문제에 집중하고, 변화하는 데이터를 이용하여 모델을 훈련할 수 있는 새로운 접근법을 제시합니다.

- **Technical Details**: RAD는 두 가지 주요 구성요소인 '인출(추출) 프레임워크(Retrieval Framework)' 및 '증류(distillation) 프레임워크'로 구성됩니다. 인출 프레임워크는 유사한 데이터를 사용하여 원본 모델의 예측을 돕고, 증류 프레임워크는 인출 네트워크에서 정보를 추출하여 매개변수화된 모듈에 정보를 증류합니다. 이를 통해 모델의 시간 및 공간 복잡성을 최적화하며, 온라인 추천 시스템에 효과적으로 배포할 수 있습니다.

- **Performance Highlights**: RAD는 여러 실제 데이터셋을 사용한 실험을 통해, 변화하는 데이터를 효과적으로 활용하여 원본 모델의 성능을 크게 향상시킬 수 있음을 입증했습니다. 이는 RAD가 추천 시스템 분야에서 새로운 방향을 제시하며, CTR(click-through rate) 모델의 성능을 향상시키기 위해 훈련 데이터의 양을 늘리는 새로운 경로를 열어주고 있습니다.




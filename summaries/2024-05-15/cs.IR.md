### Treatment Effect Estimation for User Interest Exploration on Recommender Systems (https://arxiv.org/abs/2405.08582)
Comments:
          Accepted to SIGIR 2024

- **What's New**: 새로운 'Uplift 모델 기반 추천 시스템(UpliftRec)' 프레임워크는 사용자의 감춰진 관심사를 발굴하고 추천의 정확성을 높이는 것을 목표로 합니다. 이 시스템은 사용자 피드백에서 항목 범주 비율에 따른 클릭률(CTR: Click-Through Rate)을 추정하고, 이를 바탕으로 최적의 치료법을 산출하는 방식으로 작동합니다.

- **Technical Details**: 'UpliftRec'은 관측된 사용자 피드백을 사용하여 다양한 카테고리 노출 비율 하에서의 클릭률(CTR)을 추정합니다. 그리고 이를 통해 사용자의 숨겨진 관심사를 고 CTR 보상과 연계하여 발굴합니다. 이 프레임워크는 역척도 가중치(inverse propensity weighting)를 사용하여 오인된 편향(confounder bias)을 완화하고, 동적 계획법(dynamic programming)을 적용하여 전체 CTR을 최대화하는 최적의 치료법을 계산합니다.

- **Performance Highlights**: UpliftRec는 다양한 백엔드(backend) 모델에 구현되었으며, 세 개의 데이터셋에서 광범위한 실험을 수행하였습니다. 실험 결과는 UpliftRec가 사용자의 숨겨진 관심사를 효과적으로 발견하면서 우수한 추천 정확도를 달성함을 입증합니다.



### How to Surprisingly Consider Recommendations? A Knowledge-Graph-based Approach Relying on Complex Network Metrics (https://arxiv.org/abs/2405.08465)
- **What's New**: 이 연구에서는 예상치 못한 항목을 추천함으로써 사용자 경험을 향상시킬 수 있는 Knowledge Graph (KG) 기반 추천 시스템을 제안합니다. 사용자 정의된 놀라움 정도를 도입하고 네트워크 수준 메트릭스가 추천의 놀라움 정도에 어떻게 영향을 미치는지 검토합니다.

- **Technical Details**: 이 KG 기반 시스템은 사용자 상호 작용을 아이템 카탈로그에 인코딩하여 추천을 최적화합니다. 사용자 프로파일을 대규모 카탈로그 KG의 하위 그래프로 취급하고, 아이템과 그것의 KG-정보화된 이웃이 사용자의 하위 그래프에 미치는 영향을 평가하여 추천 목록을 재순위합니다.

- **Performance Highlights**: LastFM의 청취 기록과 합성된 Netflix 시청 프로필 데이터셋을 사용하여 실험적으로 평가한 결과, 복잡한 네트워크 메트릭스에 기반한 아이템 재순위는 추천 목록의 예기치 않은 구성을 초래함을 보여줍니다.



### From Text to Context: An Entailment Approach for News Stakeholder Classification (https://arxiv.org/abs/2405.08751)
Comments:
          Accepted in SIGIR 2024

- **What's New**: 이 연구는 뉴스 기사 내에서 주요 이해관계자들의 역할을 자동으로 탐지하고 분류하는 새로운 방법을 제시합니다. 이전에는 주로 사회적 미디어 데이터를 통해 정치적 연관성이나 주요 개체의 추출에 초점을 맞추었던 것에서 나아가, 본 연구는 뉴스 콘텐츠 내에서 이해관계자의 유형을 분류하는 데 있어 효과적인 접근 방법을 소개하고 있습니다.

- **Technical Details**: 제안된 방법은 이해관계자 분류 문제를 자연어 추론(natural language inference) 작업으로 변환하여 처리합니다. 이는 뉴스 기사에서 얻은 맥락 정보와 외부 지식을 활용하여 이해관계자 유형 탐지의 정확성을 향상시키는 방식을 포함합니다. 또한, 연구팀은 zero-shot settings에서의 모델의 효과도 입증했으며, 이는 다양한 뉴스 맥락으로의 적용 가능성을 확장시킵니다.

- **Performance Highlights**: 제안된 모델은 zero-shot 설정에서도 효과적인 결과를 보여주어 새로운 뉴스 환경에 즉시 적용이 가능하다는 점에서 주목할 만합니다. 이를 통해 다양한 뉴스 기사에서 다양한 이해관계자들의 역할을 식별하고 분류하는 데 중요한 기술적 진전을 이루었습니다.




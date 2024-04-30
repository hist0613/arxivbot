### Towards Group-aware Search Success (https://arxiv.org/abs/2404.17313)
- **What's New**: 이 연구는 전통적인 검색 성공 측정 방법들이 다양한 인구 통계적 그룹의 정보 필요성을 간과한다는 문제를 지적하며, 모든 인구 통계적 그룹이 검색 결과로부터 만족을 얻을 수 있도록 보장하는 새로운 지표인 Group-aware Search Success (GA-SS)를 도입했습니다. 또한, 이 연구는 Group-aware Most Popular Completion (gMPC) 랭킹 모델을 제안하여 사용자의 의도에 있는 인구 통계적 차이를 고려합니다.

- **Technical Details**: GA-SS는 정적 및 확률적 랭킹 정책(Static and Stochastic Ranking Policies)과 사용자 브라우징 모델(User Browsing Models)을 통합하는 포괄적인 수학적 프레임워크(Mathematical Framework)로 계산됩니다. gMPC 랭킹 모델은 인구 통계적 분산을 계정하여 사용자의 다양한 필요성에 더욱 밀접하게 맞춥니다.

- **Performance Highlights**: 이 접근방식은 쿼리 자동 완성(Query Auto-Completion)과 영화 추천(Movie Recommendations)을 포함한 두 개의 실제 데이터셋에 대해 실증적으로 검증되었습니다. 확률성과 다양한 검색 성공 메트릭스(Search Success Metrics) 간의 복잡한 상호작용을 강조하는 결과를 통해, 검색 성공을 측정하는 보다 포괄적인 접근을 주장합니다. 이는 검색의 서비스 품질(Quality of Service)에 대한 미래의 조사를 영감을 주기도 합니다.



### ExcluIR: Exclusionary Neural Information Retrieva (https://arxiv.org/abs/2404.17288)
- **What's New**: 본 연구는 정보 검색 분야에서 사용자가 원하지 않는 내용을 질의하는 '배제적 검색(Exclusionary Retrieval)'에 대해 처음으로 조사하였습니다. 이를 위해 'ExcluIR'이라는 배제적 검색을 위한 자원 세트를 개발하였는데, 이는 평가 벤치마크(Evaluation Benchmark)와 훈련 세트(Training Set)로 구성되어 있습니다.

- **Technical Details**: ExcluIR 세트는 3,452개의 고품질 배제적 질의로 구성된 평가 벤치마크와 70,293개의 배제적 질의와 각각 긍정적 문서 및 부정적 문서가 쌍을 이루는 훈련 세트를 포함하고 있습니다. 연구에서는 다양한 아키텍처를 가진 기존 검색 모델들이 배제적 질의를 효과적으로 이해하는 데 어려움을 겪는 것을 확인했습니다.

- **Performance Highlights**: 훈련 데이터를 통합하면 검색 모델의 배제적 검색 성능이 향상되지만, 여전히 인간의 성능에 비해 격차가 존재합니다. 하지만, 생성적 검색 모델(Generative Retrieval Models)은 배제적 질의를 처리하는 데 자연스러운 이점이 있습니다. 이 연구는 추후 배제적 검색 연구를 위해 벤치마크 및 평가 스크립트를 공유합니다.



### TruthSR: Trustworthy Sequential Recommender Systems via User-generated  Multimodal Conten (https://arxiv.org/abs/2404.17238)
- **What's New**: 본 논문에서는 사용자의 역사적 데이터에서 사용자 선호도와 행동 패턴을 탐색하는 순차적 추천 시스템에 관하여 연구하였습니다. 이 연구는 특히 사용자가 생성한 다양한 멀티모달(multi-modal) 콘텐츠(예: 리뷰, 이미지 등)를 활용하여 순차적 추천을 개선하고자 합니다. 그러나 이러한 콘텐츠는 종종 불가피한 잡음(noise)을 포함하고 있으며, 이를 해결하기 위해 신뢰할 수 있는 순차적 추천 방법을 제안했습니다.

- **Technical Details**: 연구팀은 사용자 생성 멀티모달 콘텐츠의 일관성(consistency)과 보완성(complementarity)을 명시적으로 포착하여 잡음 간섭을 완화하는 방법을 개발했습니다. 또한, 사용자의 멀티모달 순차적 선호도(multi-modal sequential preferences) 모델링을 성공적으로 수행했습니다. 추가적으로, 주관적 사용자 관점과 객관적 아이템 관점을 통합하는 신뢰할 수 있는 의사 결정 메커니즘(trustworthy decision mechanism)을 설계하여 예측 결과의 불확실성을 동적으로 평가했습니다.

- **Performance Highlights**: 네 가지 널리 사용되는 데이터셋에서 실험적 평가를 통해 본 모델이 최신 기술(state-of-the-art) 방법들에 비해 우수한 성능을 보였음을 확인하였습니다. 이와 관련된 코드는 공개되어 있으며, 해당 URL에서 확인할 수 있습니다.



### Rank-Preference Consistency as the Appropriate Metric for Recommender  Systems (https://arxiv.org/abs/2404.17097)
- **What's New**: 이 논문에서는 기존의 추천 시스템(RS: Recommender System)의 성능을 평가하는 일반적인 단위 불변(unitary-invariant) 척도가 실제 사용자 평가와 예측 평가 사이의 차이를 측정하는 방식이 추천 시스템의 핵심적인 특성을 평가하지 못한다고 주장합니다. 추천 시스템이 실제로 필요로 하는 것은 사용자 선호도를 정확하게 예측하는 능력이며, 정확한 사용자 평가를 예측하는 최적화 문제는 이를 위한 간접적이고 부적합한 근사값에 불과하다는 것입니다.

- **Technical Details**: 저자들은 RMSE(Root Mean Square Error) 및 MAE(Mean Absolute Error)와 같은 스칼라 척도가 실제로는 추천 시스템이 사용자 선호도를 얼마나 정확하게 평가할 수 있는지를 측정하는 대리 지표일 뿐이라고 주장합니다. 대신, 사용자가 표현한 제품 선호도와 일치하지 않는 예측 쌍의 수를 단순히 계산하는 '순위-선호도 일관성(rank-preference consistency)'이 추천 시스템 성능을 평가하는 데 더 근본적으로 적합한 척도라고 제안합니다.

- **Performance Highlights**: 테스트 결과에 따르면, RMSE와 같이 임의의 척도를 최적화하기 위해 맞춤화된 방법은 일반적으로 사용자 선호도를 정확하게 예측하는데 효과적이지 않다는 것이 입증되었습니다. 이에 따라, 전통적인 추천 시스템 성능 평가 방법이 임의적이며 오도하는 결과를 초래할 수 있다는 결론을 내립니다.



### RE-RFME: Real-Estate RFME Model for customer segmentation (https://arxiv.org/abs/2404.17177)
- **What's New**: 본 연구에서는 온라인 플랫폼의 마케팅 비용을 최적화하고 고객을 효과적으로 세분화하기 위해 RE-RFME 파이프라인을 제안합니다. 이 새로운 접근 방식은 4가지 그룹(고가치, 유망, 주의 필요, 활성화 필요)으로 고객을 분류하며, 각 그룹에 맞는 마케팅 전략을 설계할 수 있게 합니다.

- **Technical Details**: 제안된 RFME(Recency, Frequency, Monetary, and Engagement) 모델은 고객의 행동 특성을 추적하여 다양한 카테고리로 분류합니다. 이 모델은 최근성(Recency), 빈도(Frequency), 금전적 가치(Monetary), 그리고 참여도(Engagement)를 기반으로 고객 데이터를 분석합니다. 마지막으로, K-means 클러스터링(K-means clustering) 알고리즘을 훈련시켜 사용자를 네 가지 범주 중 하나에 클러스터링합니다.

- **Performance Highlights**: 이 방법론은 실제 웹사이트 및 모바일 애플리케이션 사용자 데이터셋(this http URL)에 적용하여 그 효과를 검증하였습니다. 각 세분화된 그룹에 대해 맞춤형 마케팅 전략을 수립함으로써 더 높은 ROI(Return on Investment)을 달성할 수 있습니다.




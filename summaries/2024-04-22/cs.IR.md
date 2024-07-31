### FineRec:Exploring Fine-grained Sequential Recommendation (https://arxiv.org/abs/2404.12975)
Comments: This work has been accepted by SIGIR24' as a full paper

- **What's New**: FineRec은 사용자 리뷰에서 속성-의견 쌍(attribute-opinion pairs)을 활용하여 순차적 추천(Sequential Recommendation, SR)을 세밀하게 처리하는 새로운 프레임워크를 제안합니다. 이는 사용자의 선호도와 아이템 특성을 더 세밀하게 파악할 수 있는 새로운 시각을 제공합니다.

- **Technical Details**: FineRec은 대규모 언어 모델(Large Language Model, LLM)을 사용하여 리뷰에서 속성-의견 쌍을 추출합니다. 각 속성에 대해 고유한 속성별 사용자-의견-아이템 그래프(attribute-specific user-opinion-item graph)를 생성하고, 다양성을 고려한 컨볼루션(diversity-aware convolution) 작업을 통해 사용자와 아이템의 표현을 학습합니다. 마지막으로, 상호작용 주도 융합 메커니즘(interaction-driven fusion mechanism)을 통해 최종 추천을 생성합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해 FineRec이 기존의 최신 방법들보다 우수한 성능을 보임을 입증하였습니다. 이러한 세밀한 접근 방식은 추천 작업을 처리하는데 효과적임이 추가 분석을 통해 확인되었습니다.



### Disentangling ID and Modality Effects for Session-based Recommendation (https://arxiv.org/abs/2404.12969)
Comments: This work has been accepted by SIGIR24' as a full paper

- **What's New**: 새로운 프레임워크 DIMO가 제안되었습니다. 이는 세션 기반 추천에서 ID와 모달리티(modality)의 효과를 분리하여 추천의 정확성과 설명 가능성을 향상시키기 위해 설계되었습니다. 기존의 방법들이 ID와 모달리티의 효과를 혼합하여 통합하는 문제점을 해결하고자, 이 두 요소를 명확히 분리하여 모델링하는 접근법을 제시합니다.

- **Technical Details**: DIMO는 아이템 수준에서 공존 패턴(co-occurrence patterns)을 명시적으로 통합하는 공존 표현 스키마(co-occurrence representation schema)를 도입하고, 서로 다른 모달리티를 통일된 의미 있는 공간으로 정렬하여 표현합니다. 세션 수준에서는, 감독 없는 신호 없이 ID와 모달리티의 효과를 분리하는 다중 시점 자기감독 해체(multi-view self-supervised disentanglement)를 제시합니다. 이는 프록시 메커니즘(proxy mechanism)과 반사실 추론(counterfactual inference)을 포함합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에서의 광범위한 실험을 통해 DIMO의 우수성이 입증되었습니다. 이는 기존 방법들보다 나은 성능을 보였으며, 추천의 설명 가능성을 높이는 데 효과적인 것으로 나타났습니다.




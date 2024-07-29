### Debiased Collaborative Filtering with Kernel-Based Causal Balancing (https://arxiv.org/abs/2404.19596)
Comments: ICLR 24 Spotlight

- **What's New**: 이 논문은 관찰 데이터셋(observation datasets)에서 다양한 편향을 제거하면서 편향이 없는 예측 모델을 학습하는 데 목적을 둔 비편향 협업 필터링(debiased collaborative filtering)에 대해 다룹니다. 특히, 기존 방법의 한계를 극복하기 위해 선형 밸런싱(linear balancing) 요구사항을 기반으로 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 선호도 점수(propensity score)를 사용하여 관찰 샘플 분포를 목표 분포로 재조정합니다. 기존의 방법에서는 인과적 균형(causal balancing) 제약을 자주 무시하거나 비합리적인 근사를 사용했는데, 이 논문은 재생 커널 Hilbert 공간(reproducing kernel Hilbert space)에서 밸런싱 함수를 근사하는 새로운 방법을 제시합니다. 또한, 커널 함수(kernel functions)의 보편적 속성(universal property)과 대표자 정리(representer theorem)를 바탕으로 인과적 균형 제약을 보다 잘 충족시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 제안한 알고리즘은 커널 기능을 적응적으로 조정하며, 이론적으로는 일반화 오차 한계(generalization error bound)를 분석합니다. 광범위한 실험을 통해 이 방법의 효과를 입증하였고, 연구 방향을 촉진하기 위해 프로젝트를 공개했습니다.



### Interest Clock: Time Perception in Real-Time Streaming Recommendation  System (https://arxiv.org/abs/2404.19357)
Comments: Accepted by SIGIR 2024

- **What's New**: 본 논문에서는 스트리밍 추천 시스템(streaming recommendation systems)을 위한 새로운 시간 모델링 방법을 제안합니다. 'Interest Clock'이라는 방법은 사용자의 시간에 따른 동적인 선호를 인식하고, 이를 개인화된 시간 정보로 효과적으로 변환하여 추천 성능을 개선합니다. 이 방법은 사용자의 24시간 동안의 관심사를 기록하고, 시간에 따른 변화를 가우시안 분포(Gaussian distribution)를 사용하여 평활화(smoothing)하고 통합(aggregate)하여 최종적인 관심사 임베딩(interest clock embedding)을 생성합니다.

- **Technical Details**: Interest Clock은 사용자의 시간별 관심사를 시계 형태로 인코딩하고, 이러한 시간별 특성을 가우시안 분포로 평활화하여 병합하는 과정을 포함합니다. 이를 통해 추천 시스템은 현재 시간의 context에 맞는 사용자 선호를 보다 정확하게 파악할 수 있습니다. 이 방법은 기존의 시간 인코딩 방법(time encoding)과 달리, 실시간 스트리밍 트레이닝 프레임워크(real-time streaming training framework)에서도 안정적으로 작동하며, 기존 방법의 주기적 패턴과 불안정성 문제를 해결합니다.

- **Performance Highlights**: Online A/B 테스트를 통해 Interest Clock은 사용자 활성 일수에서 0.509% 및 앱 사용 시간에서 0.758%의 개선을 이끌어 냈습니다. 추가적인 오프라인 실험들 또한 이러한 개선을 확인시켜 주었습니다. 특히, Interest Clock이 도입된 더우인 음악 앱(Douyin Music App)에 성공적으로 적용됨으로써, 이 방법의 효과성과 범용성이 입증되었습니다.



### SpherE: Expressive and Interpretable Knowledge Graph Embedding for Set  Retrieva (https://arxiv.org/abs/2404.19130)
Comments: Accepted by SIGIR 2024, Camera Ready Version

- **What's New**: 이 연구는 '지식 그래프 세트 검색'라는 새로운 문제를 소개하고 있습니다. 이 문제는 불완전한 지식 그래프에서 트리플 쿼리의 모든 엔티티를 정확하게 찾는 것을 목표로 합니다. 다양한 다대다 관계를 표현할 수 있는 새로운 지식 그래프 임베딩 모델 SpherE를 제안합니다. SpherE는 각 엔티티를 벡터가 아닌 구체(sphere)로 임베딩하며, 각 관계를 회전으로 임베딩합니다.

- **Technical Details**: SpherE는 회전 기반 임베딩 방법을 기반으로 하지만, 각 엔티티를 벡터가 아닌 구체로 임베딩함으로써 일대다, 다대일, 다대다 관계를 보다 표현적으로 모델링할 수 있습니다. SpherE-kD (k≥2)는 회전의 차원에 따라 여러 방법을 포함하고 있으며, 각 엔티티의 '보편성'을 모델링하여 SpherE의 해석 가능성을 향상시킵니다.

- **Performance Highlights**: SpherE는 다양한 관계 패턴을 표현할 수 있음을 실험적으로 입증하고(Table 1), '세트 검색' 문제를 효과적으로 해결하면서 여전히 누락된 사실을 추론할 수 있는 좋은 예측 능력을 가지고 있음을 보여줍니다.



### Large Language Models as Conversational Movie Recommenders: A User Study (https://arxiv.org/abs/2404.19093)
- **What's New**: 이 연구는 대규모 언어 모델(LLM: Large Language Models)을 사용하여 온라인 필드 실험에서 사용자 관점으로 개인화된 영화 추천의 효과를 조사합니다. 이 연구는 유저들의 역사적인 소비 패턴과 대조적인 프롬프트, 그리고 유저 본인 시나리오 평가를 결합한 방법을 사용했습니다. 160명의 활성 사용자들과의 대화 및 설문조사 데이터를 분석한 결과, LLM은 추천의 설명 가능성(explainability)은 높으나 개인화(personalization), 다양성(diversity), 그리고 사용자 신뢰도(user trust) 면에서는 부족함을 보였습니다.

- **Technical Details**: LLM을 기반으로 한 추천 시스템(RecSys)에서는 다양한 개인화된 프롬프트 기술이 사용자가 인식하는 추천 품질에 큰 영향을 미치지 않는 것으로 나타났습니다. 그러나 유저가 시청한 영화의 수가 추천의 질에 더 중요한 역할을 하는 것으로 드러났습니다. LLM은 덜 알려지거나 틈새(niche) 영화에 대한 추천에서 더 큰 능력을 보였습니다. 유저와의 대화 패턴을 분석한 결과, 높은 품질의 추천을 얻기 위해서는 개인적인 맥락과 예시를 제공하는 것이 중요하다는 점이 확인되었습니다.

- **Performance Highlights**: 이 연구는 LLM 기반 추천 시스템이 설명 가능한 추천을 제공하며 인터랙티브한 사용자 경험을 생성할 수 있음을 확인했지만, 개인화된, 새로운, 다양한, 뜻밖의, 신뢰할 수 있는 영화 추천을 생성하는데는 어려움이 있음을 보여줍니다. 유저는 LLM이 제공하는 덜 인기 있는 영화나 틈새 영화 추천을 더 잘 수용하는 것으로 나타났습니다.



### Information literacy development and assessment at school level: a  systematic review of the literatur (https://arxiv.org/abs/2404.19020)
- **What's New**: 정보 리터러시(Information Literacy, IL)는 21세기에 필수적인 역량과 기본 기술을 포함하는 것으로, 현대 사회가 정보를 중심으로 운영되는 점을 고려할 때 중요성이 점차 증가하고 있습니다. 이 연구에서는 학교(K-12) 수준에서의 IL 개발 및 평가의 진화와 현재 상태를 이해하기 위해 PRISMA(statement) 지침을 기반으로 시스템적 문헌 검토를 수행했습니다.

- **Technical Details**: 연구자들은 초기에 1,234개의 논문을 조사한 후, 포함 기준을 통과한 53개의 논문을 사용하여 IL의 정의, 기술(skills), 표준(standards), 평가 도구(assessment tools)에 초점을 맞춘 여섯 가지 연구 질문에 답했습니다. 이 문헌 검토를 통해 여러 해 동안 IL의 진화와 이를 뒷받침하는 정의 및 표준의 공식화 과정이 드러났습니다.

- **Performance Highlights**: 이 리뷰는 IL 분야가 시간이 지남에 따라 어떻게 발전해 왔는지, 그리고 어떻게 표준화되고 정의되었는지를 보여줍니다. 또한, 이 분야를 더욱 발전시키기 위해 해결해야 할 주요한 격차들을 드러냅니다.



### Be Aware of the Neighborhood Effect: Modeling Selection Bias under  Interferenc (https://arxiv.org/abs/2404.19620)
Comments: ICLR 24

- **What's New**: 이 논문은 추천 시스템(Recommender System, RS)의 선택 편향(Selection Bias)과 이웃 효과(Neighborhood Effect) 문제를 인과 추론(Causal Inference)의 관점에서 새롭게 다루고 있습니다. 기존 연구들이 선택 편향에 초점을 맞추어 왔지만, 사용자-아이템 쌍(User-Item Pair)에 할당된 처리(Treatments)가 다른 쌍에 미칠 수 있는 영향, 즉 이웃 효과를 고려하지 않았습니다. 이 논문은 이웃 효과를 공식화하고, 이를 포함하여 선택 편향을 다룰 수 있는 새로운 이상 손실(Ideal Loss) 및 추정기(Estimators)를 제안합니다.

- **Technical Details**: 연구팀은 처리 표현(Treatment Representation)을 도입하여 이웃 효과를 포착하고, 확장된 잠재 결과(Potential Outcomes)와 처리 표현을 바탕으로 새로운 이상 손실을 제안합니다. 이를 평가하기 위해 근린 역방향 성향 점수(Neighborhood Inverse Propensity Score, N-IPS)와 근린 이중 견고(Neighborhood Doubly Robust, N-DR) 추정기를 개발하였습니다. 이론적 분석을 통해 제안된 N-IPS와 N-DR 추정기가 선택 편향과 이웃 효과 모두가 존재할 때 편향 없는 학습(Unbiased Learning)을 달성할 수 있음을 보여주었습니다.

- **Performance Highlights**: 다양한 반합성(Semi-Synthetic) 및 실제 데이터셋을 사용한 실험을 통해, 제안한 방법이 선택 편향 문제를 개선하는 데 효과적임을 입증하였습니다. 특히, 이전 방법들이 제대로 고려하지 못한 이웃 효과를 함께 고려함으로써 보다 정확한 사용자 선호도(User Preference) 예측이 가능하다는 점이 밝혀졌습니다.



### Automated Construction of Theme-specific Knowledge Graphs (https://arxiv.org/abs/2404.19146)
- **What's New**: 이 연구에서는 테마특화 지식 그래프(Theme-specific Knowledge Graph, ThemeKG)라는 새로운 개념을 제안하여, 주제별 맞춤형 코퍼스를 기반으로 세밀한 정보를 포함하는 지식 그래프를 구축합니다. 또한 TKGCon이라는 비감독 학습 프레임워크를 설계하여 테마에 관련된 문서에서 엔티티와 관계를 자동으로 추출하고 지식 그래프를 구성합니다.

- **Technical Details**: TKGCon 프레임워크는 테마 특화 코퍼스로부터 원시 데이터를 입력받아 세부적인 엔티티(Entity Ontology)와 관계(Relation Ontology) 온톨로지를 구축하는 데에 위키(Wikipedia)와 대규모 언어 모델(Large Language Models, LLMs)을 사용합니다. 프레임워크는 추출된 엔티티 쌍(Entity Pairs)에 대한 후보 관계를 식별하고, 문맥 정보(Contextual Information)와 결합하여 관계를 추가적으로 확정합니다. 이를 통해 테마와 일관성있는 지식 그래프를 생성합니다.

- **Performance Highlights**: 실험 결과, TKGCon은 지식 그래프 구축 기반 선을 넘는(entity recognition and relation extraction) 성능을 보여주었습니다. 이는 직접적인 GPT-4 프롬프트 호출에 비해 높은 정확도와 테마 일관성을 달성했으며, 이를 통해 세부적이고 최신의 정보를 제공하는 지식 그래프를 구축할 수 있습니다.



### Catalyzing Social Interactions in Mixed Reality using ML Recommendation  Systems (https://arxiv.org/abs/2404.19095)
- **What's New**: 새로운 혼합 현실(Mixed Reality, 이하 MR)-기반의 소셜 추천 모델이 개발되었습니다. 이 모델은 사용자의 시선 인식(gaze recognition), 근접성(proximity), 소음 수준(noise level), 혼잡도(congestion level), 그리고 대화 강도(conversational intensity)와 같은 MR 시스템을 통해 수집된 특징을 활용하여 사회적 상호작용을 증진합니다. 모델은 적절한 시기에 알림을 제공하기 위해 'right-time' 특성까지 포함하여 확장되었습니다.

- **Technical Details**: 이 연구에서는 사용자 특성(user features), MR 특성, 그리고 'right-time' 특성의 새로운 교집합을 통해 다양한 모델의 성능을 측정합니다. 네 가지 유형의 모델이 각각의 특성 조합에 따라 학습되었으며, 기존의 사용자 특성에 기반한 베이스라인 모델과 비교했습니다. 데이터 수집의 제한과 비용 문제로 MR 특성, 'right-time' 특성 및 그 조합을 사용한 모델들에서 성능 저하가 관찰되었습니다.

- **Performance Highlights**: 이러한 도전에도 불구하고, 모든 모델의 정확도를 14% 포인트 이상 향상시키는 최적화가 도입되었습니다. 가장 높은 성능을 보인 모더는 24% 향상된 정확도를 달성하였습니다.



### Exploring Weighted Property Approaches for RDF Graph Similarity Measur (https://arxiv.org/abs/2404.19052)
- **What's New**: 이 논문은 RDF (Resource Description Framework) 그래프 유사성 측정에서 속성의 상대적 중요성을 고려하는 새로운 가중치 속성 접근 방법을 제안한다. 기존의 방법들이 모든 속성을 동일하게 취급하는 문제를 해결하기 위해, 이 접근법은 다양한 상황에서 각 속성의 중요도를 반영하여 더욱 민감하고 맥락을 고려한 유사성 측정을 가능하게 한다.

- **Technical Details**: 이 연구에서는 RDF 그래프의 각 속성에 가중치를 부여하여 그 중요도를 표현하는 방법을 탐구한다. RDF 그래프는 주로 각 node (노드) 및 edge (엣지)에 연결된 다양한 속성을 포함하며, 이러한 속성의 가중치를 계산에 포함시킴으로써 더 정밀하고 상황에 맞는 유사성 평가가 수행된다. 그래프 데이터셋은 차량 도메인에 초점을 맞춰 실험적 연구를 진행하였으며, 이를 통해 속성 가중치 접근법의 효과를 검증하였다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법들에 비해 유사성 측정의 정확성에서 높은 성능을 보였다. 특히, 차량 도메인의 RDF 그래프에서의 적용 사례를 통해, 속성의 상대적 중요성을 고려하였을 때 인지된 유사성이 실제 데이터에 더 잘 반영됨을 확인할 수 있었다.



### Align-Free Multi-Plane Phase Retrieva (https://arxiv.org/abs/2404.18946)
- **What's New**: 이 연구는 다중 평면 위상 검색법에 새로운 접근법인 ACC (Adaptive Cascade Calibrated) 전략을 도입하여 실험적 정렬 문제를 극복합니다. 이 방법은 특징 점을 감지하고 인접 평면의 변환 행렬을 실시간으로 계산하여 측정 값을 디지털로 조정함으로써 정렬이 필요 없는 다중 평면 위상 검색을 가능하게 합니다.

- **Technical Details**: ACC 방법은 물체 공간에서 특징 점을 식별하고 해당 점들에 대한 affine 변환 행렬을 계산하여 측정을 정렬합니다. 이 방식은 복잡하고 비용이 많이 드는 광학 하드웨어를 필요로 하지 않으며, 기존 방법들이 직면한 실험적 정렬의 어려움을 해결합니다. ACC는 autofocus 기능을 사용해 각 측정 세트의 위치를 정확하게 결정하고, refocused 이미지를 얻어 교정 과정에 사용합니다. 이 모든 계산은 수동 개입 없이 자동으로 수행될 수 있습니다.

- **Performance Highlights**: 시뮬레이션과 실제 광학 실험을 통해 ACC 방법의 효과성이 검증되었습니다. 이 기술은 고전적인 인라인 홀로그래피(inline holography) 설정을 사용하며, 기존의 방법들과 비교하여 단순성과 비용 효율성을 유지하면서도 높은 품질의 재구성 결과를 제공합니다. ACC는 물체와 측정 공간 사이의 회절 효과를 줄이는 데 효과적이며, 실험적 오류를 보정하고 고해상도 재구성을 달성하는 데 유용합니다.




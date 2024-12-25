New uploads on arXiv(cs.IR)

### MixRec: Heterogeneous Graph Collaborative Filtering (https://arxiv.org/abs/2412.13825)
Comments:
          This paper is accepted by WSDM'2025

- **What's New**: 이 논문에서는 MixRec라는 이종 그래프 협업 필터링 모델을 제안하여 사용자와 아이템 간의 다양한 행동 패턴을 분리해내고, 각 행동 뒤에 숨겨진 의도 요인을 밝혀냅니다. 기존의 추천 시스템 모델들이 다루지 못했던 행동 이질성과 의도 다양성을 통합하여, 사용자 행동 유형 간의 관계를 효과적으로 활용합니다. 또한, 새로운 대조 학습 패러다임을 도입하여 자기 지도 데이터 증강(self-supervised data augmentation)의 장점을 탐색하고 모델의 데이터 희소성에 대한 저항력과 표현력을 향상시킵니다.

- **Technical Details**: MixRec 모델은 관계 인식(latent intents)된 표현을 생성하기 위해 매개변수화된 이종 하이퍼그래프 구조를 사용합니다. 이를 통해 다양한 의도 임베딩(embeddings)을 집계하고 사용자 행동의 다중 의존성을 효과적으로 모델링할 수 있게 됩니다. 또한, 노드와 그래프 수준에서 행동 이질성을 모델링하기 위해 관계별 대조 학습(paradigm)을 도입하며, 다양한 사용자-아이템 상호작용 패턴을 다루는 데 있어 적응적인 접근 방식을 제공합니다.

- **Performance Highlights**: MixRec 모델은 세 개의 공개 데이터셋을 이용한 실험을 통해 기존의 다양한 최신 모델(state-of-the-art baseline)에 비해 월등한 성능을 보임을 입증했습니다. 각 실험 결과는 모델의 강력한 성능 개선을 확인시켜 주며, 높은 강건성, 효율성 및 해석 가능성을 강조합니다. 결과적으로, MixRec는 추천 시스템의 표현력을 혁신적으로 향상시킬 수 있는 잠재력을 지니고 있습니다.



New uploads on arXiv(cs.AI)

### MixRec: Heterogeneous Graph Collaborative Filtering (https://arxiv.org/abs/2412.13825)
Comments:
          This paper is accepted by WSDM'2025

- **What's New**: 이 논문에서는 MixRec라는 이종 그래프 협업 필터링 모델을 제안하여 사용자와 아이템 간의 다양한 행동 패턴을 분리해내고, 각 행동 뒤에 숨겨진 의도 요인을 밝혀냅니다. 기존의 추천 시스템 모델들이 다루지 못했던 행동 이질성과 의도 다양성을 통합하여, 사용자 행동 유형 간의 관계를 효과적으로 활용합니다. 또한, 새로운 대조 학습 패러다임을 도입하여 자기 지도 데이터 증강(self-supervised data augmentation)의 장점을 탐색하고 모델의 데이터 희소성에 대한 저항력과 표현력을 향상시킵니다.

- **Technical Details**: MixRec 모델은 관계 인식(latent intents)된 표현을 생성하기 위해 매개변수화된 이종 하이퍼그래프 구조를 사용합니다. 이를 통해 다양한 의도 임베딩(embeddings)을 집계하고 사용자 행동의 다중 의존성을 효과적으로 모델링할 수 있게 됩니다. 또한, 노드와 그래프 수준에서 행동 이질성을 모델링하기 위해 관계별 대조 학습(paradigm)을 도입하며, 다양한 사용자-아이템 상호작용 패턴을 다루는 데 있어 적응적인 접근 방식을 제공합니다.

- **Performance Highlights**: MixRec 모델은 세 개의 공개 데이터셋을 이용한 실험을 통해 기존의 다양한 최신 모델(state-of-the-art baseline)에 비해 월등한 성능을 보임을 입증했습니다. 각 실험 결과는 모델의 강력한 성능 개선을 확인시켜 주며, 높은 강건성, 효율성 및 해석 가능성을 강조합니다. 결과적으로, MixRec는 추천 시스템의 표현력을 혁신적으로 향상시킬 수 있는 잠재력을 지니고 있습니다.



### SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training (https://arxiv.org/abs/2412.13148)
Comments:
          In v2 we have revised the related work, added more comprehensive citations, and clarified our key contributions

- **What's New**: 이번 연구에서는 stateless 방식으로 Stochastic Gradient Descent (SGD)를 전처리하여, 대형 언어 모델(LLM) 훈련에서 Adam 최적화기와 동일한 성능을 달성하면서 메모리 비용을 크게 줄일 수 있는 방법을 제시하고 있습니다. 이를 통해 SWAN(SGD with Whitening And Normalization)이라는 새로운 확률적 최적화기를 발전시켰습니다. SWAN은 어떤 내부 최적화 상태도 저장할 필요가 없으며, 메모리 효율성에서 SGD와 동일한 발자국을 가집니다.

- **Technical Details**: SWAN은 두 가지 잘 알려진 연산자, 즉 𝙶𝚛𝚊𝚍𝙽𝚘𝚛𝚖(GradNorm)과 𝙶𝚛𝚊𝚝𝚎𝚗𝚒𝚗𝚐(GradWhitening)를 조합하여 원시 경량들을 전처리합니다. 𝙶𝚛𝚊𝚝𝚎𝚞𝚝𝚊𝚌𝙳는 기울기 행렬에 대해 행 단위 표준화를 적용하며, 𝙶𝚛𝚊𝚝𝚎𝚝𝚊𝚕𝚈는 정규화된 기울기 행렬을 직교화하는 과정을 수행합니다. 이 두 연산자는 각각 기울기 분포의 불확실성을 안정화하고 손실 표면의 지역 기하학적 구조를 중화하는 역할을 하여, SWAN이 내부 상태 변수 저장 없이 현재 기울기 통계에만 의존할 수 있게 합니다.

- **Performance Highlights**: SWAN은 메모리 사용에서 SGD와 동일한 수준으로 유지되며, Adam과 비교해 총 메모리를 약 50% 줄입니다. LLaMA 모델의 사전 훈련에서 SWAN은 적어도 동일하거나 더 나은 성능을 보이며, 특히 350M 및 1.3B 파라미터 규모에서 동일한 평가 어려움을 달성하는 데 필요한 토큰 수를 절반으로 줄이며 2배의 속도를 기록합니다. 이러한 결과는 SWAN이 대형 언어 모델 훈련에서 매우 유용한 최적화 기법임을 나타냅니다.




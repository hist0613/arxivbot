### UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models (https://arxiv.org/abs/2405.10311)
Comments:
          11 pages, 7 figures

- **What's New**: 최근 다중 모달 대형 언어 모델(Multi-Modal Large Language Models, MM-LLMs)은 이미지 캡션 생성, 시각적 질문 응답과 같은 MM 이해 및 텍스트 기반 이미지 생성 및 편집과 같은 MM 생성 기능에서 많은 복잡한 사용 사례들을 가능하게 했습니다. 이러한 유형의 모델의 출력을 향상시키기 위해, UniRAG라는 모델에 구애받지 않는 새로운 기술이 도입되었습니다. UniRAG는 관련 정보를 검색하여 프롬프트에 제공함으로써, 추론 시 몇 가지 예시를 포함합니다. 특히, MSCOCO 데이터셋에서 일반적인 엔티티를 대상으로 한 평가에서는 GPT4와 Gemini-Pro와 같은 독점 모델뿐만 아니라 Llava, LaVIT, Emu2 같은 소규모 오픈 소스 모델에서도 출력 품질이 크게 향상되었습니다.

- **Technical Details**: UniRAG 기술은 두 단계의 워크플로우를 포함합니다. 첫 번째는 검색 단계로, 주어진 쿼리의 최상위 k개의 관련 후보를 MM 데이터베이스에서 검색합니다. 이 단계에서는 쿼리 및 후보 모달리티가 텍스트 또는 이미지의 조합일 수 있습니다. 두 번째 단계는 생성 단계로, MM-LLM이 필요한 출력을 생성하는 과정입니다. 예를 들어 이미지 캡션 생성의 쿼리는 이미지이고 검색된 후보는 캡션이며, 텍스트-이미지 생성에서는 쿼리가 캡션이고 검색된 후보가 이미지입니다. 이 기술은 UniIR MM 검색 모델(예: CLIP Score Fusion 및 BLIP Feature Fusion 모델)을 사용하여 쿼리와 데이터베이스 후보의 통합 특성 벡터를 생성합니다. 그런 다음 MM-LLMs(GPT4, Llava 등)를 통해 제로샷과 몇-샷(라그) 프롬프트 기법을 사용하여 출력을 생성합니다.

- **Performance Highlights**: UniRAG 기술을 사용한 결과, 이미지 캡션 생성 및 이미지 생성 작업에서 베이스라인 성능을 크게 초과했습니다. MSCOCO 데이터셋을 이용한 평가 결과, 이미지 캡션 생성에서는 SPICE 메트릭에서 평균 10 유닛, 이미지 생성에서는 FID 메트릭에서 평균 30 유닛 향상이 있었습니다.



### Beyond Static Calibration: The Impact of User Preference Dynamics on Calibrated Recommendation (https://arxiv.org/abs/2405.10232)
Comments:
          8 pages, 4 figures, accepted as LBR paper at UMAP '24 -- ACM Conference on User Modeling, Adaptation and Personalization 2024

- **What's New**: 이 논문에서는 추천 시스템에서 사용자 선호도의 변동성을 고려한 캘리브레이션(calibration) 접근법을 제안합니다. 기존의 캘리브레이션 방법은 사용자 선호가 고정되어 있다고 가정하지만, 이는 실제로는 구식이거나 현재의 선호도를 왜곡할 수 있습니다. 본 연구에서는 사용자와의 최신 상호작용을 기준으로 캘리브레이션을 분석하여 보다 정확한 추천을 도출하고자 합니다.

- **Technical Details**: 연구팀은 다양한 데이터셋을 사용해 사용자의 상호작용 이력을 최근부터 오래된 순으로 다양한 크기로 나누어 분석하였습니다. 이렇게 추출된 데이터의 서브샘플을 통해 추천 모델을 훈련하고 캘리브레이션을 측정했습니다. 이를 통해 사용자의 현재 선호도 분포와의 관련성을 최대화하는 가장 적합한 사용자의 프로필 세그먼트를 식별할 수 있었습니다.

- **Performance Highlights**: 본 실험에서는 두 가지 다른 도메인의 데이터셋을 사용하여 동적으로 변화하는 사용자 선호도에 따른 캘리브레이션의 정확성을 입증했습니다. 이 방법을 통해 기존의 정적인 사용자 프로필을 기반으로 한 캘리브레이션 측정이 어떻게 오도될 수 있는지 보여줍니다. 새로운 접근법은 사용자 선호도의 변화에 따라 보다 정확한 추천을 가능하게 합니다.



### HMAR: Hierarchical Masked Attention for Multi-Behaviour Recommendation (https://arxiv.org/abs/2405.09638)
- **What's New**: 추천 시스템에서 다중 행동 사용자 상호작용을 다루는 것이 중요해지고 있습니다. 이를 해결하기 위해 Hierarchical Masked Attention for multi-behavior recommendation (HMAR)라는 새로운 모델을 소개합니다. HMAR은 동일한 행동 항목에 마스킹된 자기주의를 적용한 뒤, 모든 행동에 대해 자기주의를 적용합니다. 또한, HMAR은 다중 과제 설정을 통해 아이템 행동과 순위 점수를 동시에 학습합니다.

- **Technical Details**: HMAR 모델은 여러 행동 유형을 포함하는 사용자의 역사적 상호작용 시퀀스를 캡처하기 위해 마스킹된 자기주의 기법과 역사적 행동 지표(Indicators)를 활용합니다. 모델 아키텍처는 항목 및 행동 인코딩, 계층적 마스킹된 주의 메커니즘, 그리고 다중 과제 학습으로 구성되어 있습니다. 각 항목은 완전 연결 계층을 통해 초기 임베딩으로 변환되며, 이후 역사적 행동 지표와 병합되어 사용자 행동의 순서와 빈도를 학습합니다.

- **Performance Highlights**: 네 개의 실제 데이터셋에서 수행된 광범위한 실험 결과, HMAR 모델이 최첨단 다중 행동 추천 모델들을 능가하는 성과를 보였습니다.



### Co-Matching: Towards Human-Machine Collaborative Legal Case Matching (https://arxiv.org/abs/2405.10248)
Comments:
          Draft V1: 23 pages, 7 figures

- **What's New**: 이번 논문에서는 법률 사례 매칭의 효율성을 높이기 위해 새로운 협력 매칭 프레임워크인 Co-Matching을 제안했습니다. Co-Matching은 법률 실무자와 기계를 모두 매칭 과정에 참여시켜 암묵지(tacit knowledge)를 통합하는 방식을 채택합니다. 특히 ProtoEM이라는 방법을 도입하여 인간의 결정 불확실성을 추정하고, 이를 확률론적으로 결합합니다.

- **Technical Details**: Co-Matching은 법률 실무자와 기계가 각자 주요 문장을 결정한 뒤, 이를 확률론적으로 결합하는 과정을 포함합니다. ProtoEM은 프로토타입 클러스터링과 EM 알고리즘 (Expectation Maximization)을 이용하여 인간의 결정 불확실성을 추정하는 방법입니다. 이 방법은 법률 실무자의 역사를 기반으로 의사 결정 프로토타입의 불확실성을 추정합니다.

- **Performance Highlights**: 실험 결과, Co-Matching은 기존 법률 사례 매칭 방법보다 평균적으로 5.51% (인간 기준) 및 8.71% (기계 기준) 성능이 뛰어났습니다. 이는 인간과 기계의 협력이 개별적으로 작동할 때보다 더 나은 결과를 낳음을 보여줍니다. 더불어 다양한 수준의 암묵지를 가진 법률 실무자들과의 협력에서도 높은 효율을 보였습니다.



### iDRAMA-Scored-2024: A Dataset of the Scored Social Media Platform from 2020 to 2023 (https://arxiv.org/abs/2405.10233)
- **What's New**: 온라인 커뮤니티들이 플랫폼 정책 위반으로 인해 금지될 경우, 이들은 대안 플랫폼으로 이동하는 경향이 있습니다. 이러한 이동은 새로운 플랫폼에서 독성(toxicity)의 증가와 예기치 못한 결과를 초래할 수 있습니다. 최근 연구자들은 이주 커뮤니티들이 오프라인 이벤트, 음모론 운동, 증오 연설(hate speech) 전파 및 괴롭힘(harassment)으로 이어지는 것을 적발했습니다. 이를 해결하기 위해 연구자들은 스코어드(Scored)라는 대안 레딧(Reddit) 플랫폼에서 4년간 약 5700만 개의 포스트를 수집하여 데이터셋을 공개했습니다. 이 데이터셋은 기존 플랫폼에서 금지된 커뮤니티들의 이동과 이들이 새로운 플랫폼에서 형성하는 내용을 연구하는 데 도움이 될 것입니다.

- **Technical Details**: 연구자들은 Scored 플랫폼에 있는 모든 포스트의 문장 임베딩(sentence embeddings)을 생성하여 데이터셋에 추가했습니다. 이 문장 임베딩은 최첨단 모델(state-of-the-art model)을 사용하여 만들어졌습니다. 이 데이터셋은 FAIR 원칙에 따라 공개되었으며, 이는 연구자들이 데이터를 쉽게 접근하고 활용할 수 있도록 하기 위한 목적입니다.

- **Performance Highlights**: Scored 플랫폼에서 수집된 약 5700만 개의 포스트는 4년에 걸쳐 수집되었습니다. 이 중 최소 58개의 커뮤니티는 Reddit에서 이주한 것으로 식별되었으며, 플랫폼 생성 이후 950개 이상의 새로운 커뮤니티가 생성되었습니다. 이 데이터셋을 통해 연구자들은 플랫폼 이동 패턴과 온라인 공간에서의 급진화(radicalization)에 대한 이해를 높일 수 있을 것으로 기대됩니다.



### $\Delta\text{-}{\rm OPE}$: Off-Policy Estimation with Pairs of Policies (https://arxiv.org/abs/2405.10024)
- **What's New**: 이 논문에서는 $	ext{Δ}-	ext{OPE}$라는 새로운 파생 오프-정책 추정 (off-policy estimation) 방법론을 소개합니다. 이는 기존의 오프-정책 추정 방식에서 흔히 발생하는 높은 분산 문제를 해결하는 데 중점을 두고 있습니다. 정책들 사이에 긍정적인 공분산이 존재하는 경우, 정책 가치를 낮은 분산으로 추정할 수 있다는 점을 활용한 것입니다.

- **Technical Details**: 기존의 오프-정책 추정 방식인 Inverse Propensity Scoring (IPS)와 그 확장 모델을 기반으로 $	ext{Δ} - 	ext{OPE}$ 방식을 제안했습니다. 이 방법은 데이터 수집에 사용된 확률적 로깅 정책과 목표 정책 간의 정책 값 차이를 추정하도록 설계되었습니다. 또한, 최고의 분산-최적화된 추가 제어 변수를 도입해 효율성을 더욱 향상시켰습니다.

- **Performance Highlights**: 시뮬레이션, 오프라인, 온라인 실험에서 새로운 $	ext{Δ}-	ext{OPE}$ 방식은 평가와 학습 작업 모두에서 유의미한 성능 향상을 보여주었습니다. 특히 오프-정책 평가(OPE) 시나리오에서, 생산 및 목표 정책 간의 차이를 추정하는 이 방법은 통계적 검정력 향상과 더불어 신뢰 구간을 좁히는 데 기여하여 실질적인 결정을 더 효과적으로 내릴 수 있게 합니다.




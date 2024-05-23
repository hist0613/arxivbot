### A Unified Search and Recommendation Framework Based on Multi-Scenario Learning for Ranking in E-commerc (https://arxiv.org/abs/2405.10835)
Comments:
          Accepted by SIGIR 2024

- **What's New**: 전자상거래에서 중요한 검색(Search) 및 추천(Recommendation) 시나리오를 통합하여 성능을 향상시키기 위한 'Unified Search and Recommendation (USR)' 프레임워크를 제안합니다. 이는 전통적인 다중 시나리오 모델의 한계를 극복하고, S&R 시나리오의 차이를 효과적으로 포착하는 방법론을 제시합니다.

- **Technical Details**: USR 프레임워크는 S&R 듀얼 뷰(views) 사용자 관심도 추출 레이어(Interest Extractor Layer, IE)와 S&R 듀얼 뷰 특징 생성 레이어(Feature Generator Layer, FG)로 구성됩니다. 이들은 사용자의 관심도를 각 뷰에서 추출하고 시나리오와 무관한 특징들을 생성합니다. 또한, 'Global Label Space Multi-Task Layer (GLMT)'를 도입하여 글로벌 라벨 공간을 활용하는 다중 작업 학습을 통해 주 작업과 보조 작업을 조건부 확률로 공동 모델링합니다.

- **Performance Highlights**: 실제 산업 데이터셋을 통한 광범위한 실험으로 USR이 다양한 다중 시나리오 모델에 적용될 수 있으며, 성능을 크게 향상시킬 수 있음을 확인했습니다. 또한, 온라인 A/B 테스트에서도 많은 지표에서 상당한 성능 향상을 나타냈습니다. 현재 USR은 7Fresh 앱에 성공적으로 배포되었습니다.



### Know in AdVance: Linear-Complexity Forecasting of Ad Campaign Performance with Evolving User Interes (https://arxiv.org/abs/2405.10681)
Comments:
          12 pages, 4 figures, accepted at ACM SIGKDD 2024

- **What's New**: 최신 논문인 AdVance는 실시간 비딩(RTB) 광고주의 캠페인 비용 및 수익을 사전 예측하는 새로운 타임-어웨어(time-aware) 프레임워크를 제안합니다. 이 모델은 지역 경매 수준 모델링과 글로벌 캠페인 수준 모델링을 통합하여 사용자 피로와 선호도의 변화를 분리하고, 이를 통해 더 정확한 예측을 가능하게 합니다.

- **Technical Details**: AdVance는 클릭된 아이템의 시간-위치(sequence of clicked items) 시퀀스와 모든 표시된 아이템의 간결한 벡터(fatigue vector)를 사용하여 사용자 선호도와 피로도를 분리합니다. 크로스-어텐션(cross-attention) 메커니즘을 통해 사용자 선호도의 변화를 포착하며 각 후보 광고에 대한 피로도 벡터를 조건으로 합니다. 트랜스포머 인코더(Transformer Encoder)를 사용하여 모든 경매를 임베딩(embedding)으로 압축하고, 조건부 상태 공간 모델(SSM)을 통해 롱 렌지(long-range) 의존성을 이해하면서도 글로벌 선형 복잡도를 유지합니다.

- **Performance Highlights**: 광범위한 평가와 ablation 연구에서 AdVance의 우수성이 입증되었으며, Tencent 광고 플랫폼에 배포된 이후 사용자당 평균 수익(ARPU)에서 놀라운 4.5% 상승을 기록했습니다. 이는 Art-of-the-State 방법론 대비 뛰어난 성능을 의미합니다.



### CELA: Cost-Efficient Language Model Alignment for CTR Prediction (https://arxiv.org/abs/2405.10596)
Comments:
          10 pages, 5 figures

- **What's New**: CTR 예측에서 텍스트 기반의 특징 및 PLM(Pre-trained Language Model)을 활용하여 ID 기반 모델의 한계를 극복하려는 시도가 이번 연구의 핵심입니다. 새로운 모델인 CELA(Cost-Efficient Language Model Alignment)가 이런 문제들을 해결하는 혁신적인 솔루션으로 제안되었습니다. CELA는 텍스트 특징과 언어 모델을 통합하면서도, ID 기반 모델의 협업 필터링 기능을 유지하는 것이 가능합니다. 이 접근 방식은 모델-독립적(model-agnostic)으로 기존 네트워크 아키텍처의 최소한의 변경만으로도 적용될 수 있습니다.

- **Technical Details**: CELA는 세 단계의 접근방식을 사용합니다. 첫 번째 단계는 도메인 적응형 사전 학습(domain-adaptive pre-training)으로, PLM이 특정 데이터셋에 맞추어 텍스트 특징을 추가로 학습합니다. 두 번째 단계는 추천 지향형 모달 정렬(recommendation-oriented modal alignment)로, 항목 텍스트 표현을 ID 기반 모델의 항목 특징 임베딩과 정렬시킵니다. 마지막 단계는 다중 모달 특징 융합(multi-modal feature fusion)으로, 정렬된 텍스트 표현을 ID 기반 모델과 통합하여 잠재적인 클릭 패턴을 파악합니다. 이를 통해, PLM의 출력을 추천 작업에 맞추어 정돈하고 훈련 시간을 줄이며, 효율적인 추론을 가능케 합니다.

- **Performance Highlights**: CELA는 공개 및 산업 데이터셋에서의 광범위한 오프라인 실험에서 기존 최신 기법들과 비교해 뛰어난 성능을 보였습니다. 또한, 실제 산업 앱 추천 시스템에서 수행된 온라인 A/B 테스트에서도 실질적인 효과와 타당성을 입증했습니다. 이를 통해 CELA의 실제 응용 가능성이 강하게 뒷받침되었습니다.



### In-context Contrastive Learning for Event Causality Identification (https://arxiv.org/abs/2405.10512)
- **What's New**: 이번 논문에서는 사건 인과성 식별(Event Causality Identification, ECI) 작업에서 긍정 및 부정 시범 예시를 효과적으로 활용하기 위해 대조 학습(Contrastive Learning)을 적용한 새 모델, ICCL(In-Context Contrastive Learning)을 제안합니다. 이 접근법은 Prompt Learning 기반 접근 방법의 복잡한 Prompt 설계와 파생 작업에 대한 의존성을 줄이는 것을 목표로 합니다.

- **Technical Details**: ICCL 모델은 세 가지 모듈로 구성되어 있습니다: Prompt Learning 모듈, In-Context Contrastive 모듈, Causality Prediction 모듈. Prompt Learning 모듈은 입력 이벤트 쌍과 회수된 데모 샘플을 Prompt 템플릿으로 재구성합니다. In-Context Contrastive 모듈은 긍정 시범 예시와의 일치성을 최대화하고 부정 시범 예시와의 일치성을 최소화하여 이벤트 멘션 표현을 최적화합니다. 마지막으로, Causality Prediction 모듈은 답변 단어를 예측하여 인과 관계를 식별합니다.

- **Performance Highlights**: ICCL은 EventStoryLine 및 Causal-TimeBank와 같은 널리 사용되는 코퍼스에서 평가되었으며, 최첨단 알고리즘보다 유의미한 성능 향상을 보여주었습니다.



### Positional encoding is not the same as context: A study on positional encoding for Sequential recommendation (https://arxiv.org/abs/2405.10436)
Comments:
          19 pages, 3 figures, 12 tables

- **What's New**: 이 논문은 Sequential Recommendation Systems(SRS)에서 positional encodings(포지셔널 인코딩)이 상대적 정보(temporal footprint)만으로는 유추할 수 없는 추가적인 상대적 정보를 아이템 간에 제공한다는 점을 분석하고 있습니다. 새롭게 제안된 인코딩 방식들이 SRS 모델의 메트릭과 안정성을 향상시킬 수 있음을 발견했습니다. 이를 통해 정확한 포지셔널 인코딩을 찾는 것이 최신 성능(state-of-the-art)을 달성하는데 중요한 역할을 한다는 것을 입증했습니다.

- **Technical Details**: 연구는 대표적인 포지셔널 인코딩 기법들을 분석하고 Amazon 데이터셋을 사용하여 이들의 메트릭과 안정성에 미치는 영향을 평가했습니다. 또한, 절대적 포지셔널 인코딩(Absolute Positional Encoding, APE)과 상대적 포지셔널 인코딩(Relative Positional Encoding, RPE)을 포함한 다양한 인코딩 방법들을 제안했습니다. APE는 Transformer 블록의 첫번째 부분에서 전체 시퀀스의 절대적 위치 정보를 제공하고, RPE는 Transformer의 헤드 내부에서 타겟 아이템 주변의 상대적 위치 정보를 제공합니다.

- **Performance Highlights**: 논문에서는 새롭게 제안된 인코딩 기법들이 기존의 인코딩 방법들보다 더 나은 성능을 보이며, 특히 데이터셋의 희소성과 인코딩의 관계에 대한 연구를 통해 어떤 인코딩이 가장 효과적인지에 대한 로드맵을 제시했습니다. 특정 인코딩을 사용함으로써 SRS 모델의 학습 안정성을 크게 개선할 수 있음을 확인했습니다.



### Pointwise Metrics for Clustering Evaluation (https://arxiv.org/abs/2405.10421)
- **What's New**: 이번 논문은 두 개의 클러스터링(Clustering) 간의 유사성을 평가하기 위한 포인트 단위 클러스터링 메트릭(Pointwise Clustering Metrics)을 정의합니다. 이 메트릭들은 각 항목의 상대적 중요성을 반영할 수 있으며, 클러스터 균질성과 완전성 등 중요한 측면을 특성화합니다. 또한, 개별 항목, 클러스터, 임의의 항목 슬라이스, 전체 클러스터링 등의 평가를 가능하게 하여 클러스터링 결과에 대한 심도 있는 인사이트를 제공합니다.

- **Technical Details**: 포인트 단위 메트릭은 표준 집합 이론(set-theoretic) 개념을 바탕으로 정의되며, 각 항목에 대한 가중치(weight) 부여가 가능합니다. 이상적인 클러스터링과 실제 클러스터링을 비교하여 품질을 평가할 수 있도록 설계되었습니다. 이 논문은 수학적으로 안정적인 기반을 통해 다양한 클러스터링 평가 기법을 지원합니다. 구체적으로는, 해당 메트릭이 클러스터링 실수를 깊이 있게 분석하거나 항목 슬라이스에 대한 영향을 탐구하는 데 도움을 줍니다.

- **Performance Highlights**: 이 포인트 단위 메트릭은 클러스터링 평가의 기초로서 높은 신뢰성을 제공합니다. 클러스터링의 품질에 대한 통계적 추정값(statistical estimates)을 얻는 데 이상적인 수단이 되며, 대규모 클러스터링을 평가하기에 적합합니다.



### Neural Optimization with Adaptive Heuristics for Intelligent Marketing System (https://arxiv.org/abs/2405.10490)
Comments:
          KDD 2024

- **What's New**: 이번 논문에서는 디지털 세계에서 점점 더 중요해지고 있는 컴퓨팅 마케팅(Computational Marketing)의 어려움을 해결하기 위해 NOAH(Neural Optimization with Adaptive Heuristics)라는 새로운 마케팅 AI 시스템 프레임워크를 제안합니다. NOAH는 2B(비즈니스 대상) 및 2C(소비자 대상) 제품, 그리고 소유 채널 및 유료 채널을 모두 고려한 최초의 일반 마케팅 최적화 프레임워크입니다.

- **Technical Details**: NOAH 프레임워크는 예측 모듈, 최적화 모듈, 그리고 적응적 휴리스틱(adaptive heuristics) 모듈 등 주요 구성 요소를 포함합니다. 특정 예로는 입찰 최적화와 콘텐츠 최적화를 다룹니다. 특히 NOAH는 링크드인(LinkedIn)의 이메일 마케팅 시스템에 성공적으로 적용되었으며 기존의 순위 시스템보다 상당한 성과를 보였습니다. 기술적 세부사항에는 (i) 지연된 피드백을 수명 가치로 해결, (ii) 난수화된 대규모 선형 프로그래밍, (iii) 오디언스 확장을 통한 검색 향상, (iv) 타깃 테스트에서 신호 감소 줄이기, (v) 통계 테스트에서 제로-인플레이션(zero-inflated) Heavy-tail 메트릭 처리 방법 등이 포함됩니다.

- **Performance Highlights**: NOAH 프레임워크는 링크드인 이메일 마케팅 시스템에 적용되어 기존의 순위 시스템을 뛰어넘는 성과를 거두었습니다. 구체적으로는 광고 입찰 최적화, 랜덤화된 대규모 선형 프로그래밍, 그리고 오디언스 확대를 통해 검색 기능을 향상시키는 등 다양한 측면에서 개선된 성과를 보였습니다.




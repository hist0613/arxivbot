### Corpus Poisoning via Approximate Greedy Gradient Descen (https://arxiv.org/abs/2406.05087)
- **What's New**: 새로운 연구는 정보 검색 시스템에서의 코퍼스(poisoning attack)를 효과적으로 실행할 수 있는 새로운 공격 방법인 'Approximate Greedy Gradient Descent (AGGD)'을 제안합니다. 이 연구는 기존 HotFlip 방법의 한계를 극복하고, 더 구조적인 검색을 통해 더 높은 질의 토큰 수준의 변형을 선택할 수 있음을 보입니다.

- **Technical Details**: AGGD는 랜덤하게 토큰을 샘플링하는 대신 모든 토큰 위치에서 최상위 토큰을 선택하여 점진적 경사 하강법(Greedy Gradient Descent)을 사용합니다. 이는 AGGD의 검색 궤적을 결정적(deterministic)으로 만들어 더 구조적인 최선-우선 검색(best-first search)을 가능하게 합니다. 실험 결과, AGGD는 NQ와 MS MARCO 데이터셋에서 기존 HotFlip보다 각각 17.6% 및 13.37% 높은 공격 성공률을 기록했습니다.

- **Performance Highlights**: AGGD는 여러 데이터셋과 검색 모델에서 높은 공격 성공률을 달성했습니다. 특히 ANCE 검색 모델을 공격할 때, NQ 데이터셋 코퍼스의 단 0.00003%, MS MARCO 데이터셋의 0.00001%에 해당하는 하나의 적대적 패세지를 주입함으로써, NQ 데이터셋에서 44.35%, MS MARCO 데이터셋에서 26.16%의 공격 성공률을 보여주었습니다. 또한 AGGD는 다른 도메인의 새로운 질의에 대해서도 82.28%의 공격 성공률을 기록했습니다.



### CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search (https://arxiv.org/abs/2406.05013)
- **What's New**: 이 논문에서는 오픈소스 대형 언어 모델(LLMs)을 효과적으로 활용하여 대화형 검색에서 모호한 쿼리를 개선하는 방법을 연구합니다. 새로운 'CHIQ' 방법을 도입하여, 대화 기록에서 모호성을 해결한 후 쿼리를 재작성하는 두 단계 방식을 제안합니다. 이는 주로 폐쇄형 LLMs를 사용하는 기존 연구들과는 대조적입니다. 5개의 주요 벤치마크에서 CHIQ가 대부분의 설정에서 최첨단 성능을 보임을 입증했습니다.

- **Technical Details**: CHIQ는 대화 기록의 모호성을 해결하기 위해 NLP 과제 해결 능력을 갖춘 LLM을 사용합니다. 본 연구에서는 LLaMA-2-7B와 같은 오픈소스 LLM을 사용하여, 컨텍스트를 확장하거나 코리퍼런스(coreference) 관계를 해결하고 대화 기록을 개선해 쿼리의 적절성을 높입니다. 이처럼 개선된 대화 기록을 기존 프레임워크에 통합하는 다양한 방법을 조사하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, CHIQ는 밀집(dense) 및 희소(sparse) 검색 설정에서 대부분의 벤치마크에서 최첨단 성능을 달성하였습니다. 폐쇄형 LLM과 비교했을 때, 개선된 대화 기록을 사용할 때 성능 격차가 상당히 좁아졌습니다. 이는 오픈소스 LLM이 상업용 모델과 경쟁할 수 있는 가능성을 보여줍니다.



### QAGCF: Graph Collaborative Filtering for Q&A Recommendation (https://arxiv.org/abs/2406.04828)
- **What's New**: Q&A 플랫폼의 새로운 추천 모델, QAGCF(Graph Collaborative Filtering)가 제안되었습니다. 이 모델은 기존 추천 시스템의 한계를 극복하고 질문과 답변 쌍의 협업 및 의미적 정보를 효과적으로 분리하여 사용자의 클릭 행동을 더 정확하게 예측합니다.

- **Technical Details**: QAGCF는 그래프 신경망(neural network) 모델을 기반으로하여 협업 뷰와 의미 뷰를 분리하여 각각의 협업 및 의미 정보를 분리합니다. 협업 뷰에서는 사용자가 클릭한 질문과 답변을 개별적으로 모델링하며, meaning view에서는 질문과 답변 사이, 그리고 질문-답변 쌍들 간의 의미적 연결을 캡처합니다. 이 두 뷰는 글로벌 그래프로 결합되어 전체적인 협업 및 의미 정보를 통합합니다. 글로벌 그래프에서 고차 동조성(high heterophily) 문제를 해결하기 위해 다항식 기반 그래프 필터(polynomial-based graph filters)를 사용하며, 강건한 임베딩(robust embedding)을 얻기 위해 대조 학습(contrastive learning)도 활용합니다.

- **Performance Highlights**: 산업 및 공개 데이터셋에 대한 광범위한 실험 결과 QAGCF가 지속적으로 기존 방법들을 능가하고 최첨단 성과를 달성함을 입증하였습니다.



### Scaling Automatic Extraction of Pseudocod (https://arxiv.org/abs/2406.04635)
- **What's New**: 본 연구에서는 약 32만 개의 가짜 코드(pseudocode) 예제를 포함한 대규모 컬렉션이 제공되었습니다. 이 컬렉션은 arXiv 논문에서 추출된 것으로, 이는 알고리즘 이해를 높이고 자동 코드 생성 및 Optical Character Recognition (OCR) 등의 작업에 유용할 수 있습니다. arXiv 논문 220만 편을 스캔하였으며, 그 중 1,000편은 수작업으로 점검 및 레이블링되었습니다.

- **Technical Details**: 가짜 코드 추출을 위해 arXiv 논문의 LaTex 파일 및 PDF 파일을 분석하는 메커니즘을 개발했습니다. LaTex 파일에서는 명령어를 통해 상대적으로 쉽게 가짜 코드를 추출할 수 있으나, PDF 파일에서는 텍스트와 그림의 경계를 감지하고 이를 추출하는 것이 복잡한 작업입니다. 이를 위해 머신 러닝 기반의 도구가 사용되었습니다.

- **Performance Highlights**: 통계 분석 결과, arXiv 논문에서 가짜 코드 사용이 지수적 증가를 보이고 있음을 밝혔습니다. 또한, 가짜 코드의 클러스터링과 주제별 분석을 통해 다양한 가짜 코드 구조를 조사했습니다.



### Better Late Than Never: Formulating and Benchmarking Recommendation Editing (https://arxiv.org/abs/2406.04553)
- **What's New**: 이번 논문에서는 'recommendation editing(추천 편집)'이라는 새로운 과제를 제안합니다. 이는 기존의 추천 시스템이 제공하는 부적절한 추천을 수정하는 방법으로, 기존 모델을 재학습(retraining)하거나 원본 학습 데이터를 접근하지 않고도 부적절한 아이템을 제거하는 데 중점을 둡니다.

- **Technical Details**: 추천 편집 문제는 세 가지 주요 목표를 정의합니다: (1) 엄격한 수정(Strict Rectification)은 중대한 문제를 유발하는 부적절한 추천 아이템을 제거하는 것입니다. (2) 협력적 수정(Collaborative Rectification)은 관찰되지 않은 유사한 부적절한 추천 아이템도 제거하는 것입니다. (3) 집중적 수정(Concentrated Rectification)은 적절한 추천이 대부분 유지되도록 하는 것입니다. 이를 위해, 새로운 'Editing Bayesian Personalized Ranking Loss'를 기반으로 하는 간단하지만 효과적인 기준점을 제안합니다.

- **Performance Highlights**: 제안된 방법의 효과를 입증하기 위해 다양한 관련 분야의 방법들을 통합한 포괄적인 벤치마크를 설립했습니다. 이를 통해, 제안된 추천 편집 방법이 부적절한 추천 문제를 완화하는 데 실질적으로 효과적임을 보여주었습니다.



### Innovations in Cover Song Detection: A Lyrics-Based Approach (https://arxiv.org/abs/2406.04384)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 커버 곡(cover song)을 자동으로 식별하는 새로운 방법을 제안합니다. 특히, 기존의 오디오 분석에만 의존하는 접근법과 달리, 이 방법은 곡의 가사를 활용합니다. 이를 위해 새로운 데이터셋을 구축했으며, 이 데이터셋에는 5078개의 커버 곡과 2828개의 원곡이 포함되어 있습니다. 모든 곡에는 주석이 달린 가사가 첨부되어 있습니다.

- **Technical Details**: 제안된 방법은 Levenshtein 거리와 단어 오류율(WER)을 사용하여 원곡과 커버 곡 사이의 가사 유사성을 평가합니다. 이를 위해 Levenshtein 거리 및 단어 오류율을 계산하는 기존 구현을 활용합니다. 또한, 텍스트 전처리 및 임베딩 생성을 위해 사전 학습된 XLM-RoBERTa 모델을 사용합니다. 이 임베딩 벡터를 기반으로 커버 곡과 원곡 사이의 유사성을 계산하여 가장 유사한 곡을 예측합니다. 모델 훈련에는 삼중 항 손실(triplet loss) 방법을 사용하여 유사한 샘플 간의 거리를 최소화하고 비유사한 샘플 간의 거리를 최대화합니다.

- **Performance Highlights**: 제안된 방법은 기존의 여러 기준 방법들보다 더 나은 성능을 보여주었습니다. 평가 메트릭으로는 평균 정밀도(mAP), 평균 순위(MR), 정밀도@1(P@1)을 사용하였으며, 이는 커버 곡 탐지 분야에서 널리 사용되는 메트릭입니다. 이러한 결과는 가사를 활용한 커버 곡 식별 방법의 우수성을 입증합니다.



### Dynamic Online Recommendation for Two-Sided Market with Bayesian Incentive Compatibility (https://arxiv.org/abs/2406.04374)
- **What's New**: 이 논문은 인터넷 경제에서 중요한 역할을 하는 추천 시스템의 설계 과정에서 직면하는 두 가지 주요 문제, 즉 (1) 새로운 제품 탐색과 이미 알려진 사용자 선호도 활용 간의 탐색-활용 균형 문제와 (2) 사용자들의 자발적 행동과 이질적 선호도를 고려한 동적인 인센티브 호환성 문제를 공식화했습니다. 이를 해결하기 위해 동적 베이지안 인센티브 호환 추천 프로토콜(DBICRP)을 제안하고, RCB라는 두 단계 알고리즘을 개발했습니다.

- **Technical Details**: RCB 알고리즘은 첫 번째 단계에서 동적 인센티브 호환성을 유지하면서 충분한 샘플 크기를 결정하기 위해 제품을 탐색하고, 두 번째 단계에서는 반비례 샘플링을 사용하여 적은 후회를 보장합니다. 이 알고리즘은 가우시안 사전(gaussian prior) 가정 하에서 베이지안 인센티브 호환성(Bayesian Incentive Compatibility, BIC)을 만족함을 이론적으로 증명했습니다.

- **Performance Highlights**: RCB 알고리즘은 후회(regret)가 $O(\sqrt{KdT})$임을 이론적으로 증명했으며, 시뮬레이션 및 실제 사례(예: 맞춤형 와파린 개인별 투여량)에서 강력한 인센티브 효과와 적은 후회, 높은 강건성을 입증했습니다.



### Error Bounds of Supervised Classification from Information-Theoretic Perspectiv (https://arxiv.org/abs/2406.04567)
- **What's New**: 이번 연구는 정보이론적 관점에서 심층 신경망(DNN)을 사용한 지도 분류의 이론적 기초를 탐구하며, 오버파라미터화된 신경망의 일반화 능력, 비볼록 최적화 문제에서의 효율적 성능, 플랫 최소값(flat minima)의 메커니즘을 설명하는 새로운 개념을 도입했습니다. 본 논문에서는 fitting error(맞춤 오류)와 model risk(모델 위험)을 소개하여 기존의 generalization error(일반화 오류)와 함께 기대 위험의 상한을 형성합니다.

- **Technical Details**: 일반화 오류가 데이터 분포의 스무스성과 샘플 크기에 의해 영향을 받는 복잡성에 의해 제한됨을 증명했습니다. 우리는 NTK(Neural Tangent Kernel) 및 모델의 파라미터 수와 fitting error의 상관관계를 도출합니다. KL 발산(Kullback-Leibler divergence)을 사용하여 기존 손실 함수의 의존성을 제거하고, 삼각부등식(triangle inequality)을 활용하여 기대 위험의 상한을 설정했습니다.

- **Performance Highlights**: 실증 검증은 도출된 이론적 상한과 실제 기대 위험 사이에 유의미한 양의 상관관계가 있음을 보여, 이론적 발견의 실용성을 확인했습니다. 작은 최대의 eNTK(equal-input Neural Tangent Kernel, λ_max(H(fθ(x))))은 기대 위험을 최소화하는 데 유리한 것으로 증명되었습니다.



### GNNAnatomy: Systematic Generation and Evaluation of Multi-Level Explanations for Graph Neural Networks (https://arxiv.org/abs/2406.04548)
- **What's New**: 새로운 연구는 다양한 하위 구조를 체계적으로 탐색하고 결과를 평가하는 데 어려움을 겪는 기존 방법론의 한계를 극복하기 위해 GNNAnatomy라는 시각적 분석 시스템을 소개합니다. 이는 그래프 수준의 분류 작업에서 GNN의 동작을 설명하기 위해 그래프렛(graphlets)을 사용하며, 가설적 사실(factual) 및 반사실적(counterfactual) 설명을 통해 GNN의 행동을 분석합니다.

- **Technical Details**: GNNAnatomy 시스템은 모델 및 데이터셋에 독립적으로 작동하며, 그래프렛(graphlets)을 사용해 GNN의 예측과 그래프렛 빈도의 상관 관계를 분석하여 설명을 생성합니다. 구체적으로, (1) 그래프렛 빈도와 분류 신뢰도 간의 상관 관계와 (2) 원래 그래프에서 해당 하위 구조를 제거한 후 분류 신뢰도의 변화를 평가하는 두 가지 측정을 도입합니다. 실제로 그래프렛 빈도를 계산하는 것은 NP-hard 문제이므로, GNNAnatomy는 샘플링 방법을 사용하여 3, 4, 5개의 노드를 가진 그래프렛의 빈도를 계산합니다.

- **Performance Highlights**: 실제 데이터셋과 합성 데이터셋을 활용한 사례 연구에서 GNNAnatomy는 효과적인 설명을 제공하는 것으로 입증되었습니다. 또한, 최신 GNN 설명자(state-of-the-art GNN explainer)와 비교하여 그 설계의 유용성과 다용성을 보여주었습니다.



### Negative Feedback for Music Personalization (https://arxiv.org/abs/2406.04488)
Comments:
          6 pages, 4 figures, accepted to ACM UMAP 2024

- **What's New**: 이번 연구에서는 인터넷 라디오의 Next-Song 추천 시스템에서 실제 부정 피드백을 활용하여 학습 속도와 정확성을 크게 개선할 수 있음을 입증했습니다. 또한, 사용자 피드백 시퀀스에 스킵(건너뛰기) 데이터를 추가함으로써 사용자 커버리지와 정확성을 모두 개선하는 방법을 제안했습니다.

- **Technical Details**: 본 연구에서는 SASRec와 BERT4Rec와 같은 기존의 Transformer 아키텍처를 바탕으로 한 추천 시스템을 참고했습니다. 또한, 부정 표본으로 랜덤 샘플을 사용하는 대신 실제 사용자로부터 수집된 명시적 부정 피드백(예: 'thumb-down')을 사용하여 모델을 학습시켰습니다. 이를 통해, 학습 시간을 약 60% 절감하고 테스트 정확도를 약 6% 개선할 수 있음을 확인했습니다.

- **Performance Highlights**: 실험 결과, 명시적 부정 피드백을 포함한 모델이 더 적은 학습 시간으로 더 높은 정확도를 보였으며, 특히 부정 표본으로 사용된 데이터의 양이 적절했을 때 최상의 성능을 발휘했습니다. 또한, 스킵 데이터를 추가로 입력하여 개인화된 추천의 범위를 확대하고 정확도도 약간 향상시켰습니다.




### Recommender Algorithm for Supporting Self-Management of CVD Risk Factors in an Adult Population at Hom (https://arxiv.org/abs/2405.11967)
Comments:
          26 pages, 5 figures

- **What's New**: 이번 논문에서는 심혈관 질환(CVD; Cardiovascular Diseases) 예방의 효과성을 향상시키기 위해 지식 기반 추천 알고리즘을 제안했습니다. 이 알고리즘은 성인이 가정에서 CVD 위험 요인을 자가 관리할 수 있도록 지원합니다. 제안된 알고리즘은 다차원 추천 모델(Multidimensional recommendation model)과 새로운 사용자 프로필 모델(User Profile Model)을 기반으로 합니다. 사용자 프로필 모델에는 현재 건강 상태뿐만 아니라 예측 평가도 포함됩니다.

- **Technical Details**: 제안된 알고리즘의 주요 특징은 규칙 기반 논리(rule-based logic)와 대형 언어 모델(Large Language Model)의 능력을 결합하여 다차원 추천의 설명 구성요소(Explanatory component)를 인간과 같은 텍스트로 생성하는 것입니다. 이 알고리즘은 공식 지침에 따라 유저(CVD 건강 상태)를 평가하며 다차원 추천 모델을 통해 예측 결과를 추론합니다.

- **Performance Highlights**: 제안된 알고리즘의 검증 및 평가 결과, 사용자들이 가정에서 CVD 위험 요인을 자가 관리하는 데 유용한 것으로 나타났습니다. 유사한 지식 기반 추천 알고리즘과 비교했을 때, 제안된 알고리즘은 더 많은 CVD 위험 요인을 평가하고 생성된 추천의 정보 및 의미 용량이 더 큽니다.



### CaseGNN++: Graph Contrastive Learning for Legal Case Retrieval with Graph Augmentation (https://arxiv.org/abs/2405.11791)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2312.11229

- **What's New**: 이번 논문에서는 CaseGNN++라는 새로운 모델을 소개합니다. CaseGNN++는 법률 사례 검색(LCR)에서 중요한 역할을 하는 구조적 정보를 더욱 효과적으로 활용하여 기존 CaseGNN 모델을 개선합니다. 이는 특히 법률 데이터셋의 라벨링 부족 문제를 해결하고자 추가 학습 신호를 제공하는 그래프 대비 학습 목표(graph contrastive learning objective)를 도입했습니다.

- **Technical Details**: CaseGNN++는 두 가지 주요 기술적 요소를 포함합니다. 첫째, Edge-updated Graph Attention Layer(EUGAT)를 도입하여 TACG(Text-Attributed Case Graph) 내의 노드와 엣지 정보를 업데이트합니다. 둘째, 그래프 증가(graph augmentation)를 통해 추가 학습 신호를 제공하는 그래프 대비 학습 목표를 개발했습니다. 이러한 접근법은 법적 요소 간의 관계를 포괄적으로 모델링하여 사례 표현(case representation)을 생성합니다.

- **Performance Highlights**: COLIEE 2022와 COLIEE 2023 두 가지 벤치마크 데이터셋에서 수행한 포괄적인 실험 결과, CaseGNN++는 기존 CaseGNN 모델뿐만 아니라 최신 LCR 방법들에 비해 탁월한 성능을 달성했습니다. 이 논문의 결과는 CaseGNN++가 LCR 작업에서 가장 우수한 성능을 발휘한다는 것을 입증합니다.



### Modeling User Fatigue for Sequential Recommendation (https://arxiv.org/abs/2405.11764)
Comments:
          SIGIR 2024

- **What's New**: 오늘날의 추천 시스템은 사용자의 관심사를 반영하여 정보를 필터링하지만, 사용자가 짧은 기간 동안 반복적으로 노출된 유사한 추천에 피로감을 느낄 수 있다는 문제가 있습니다. 이를 해결하기 위해, 본 논문에서는 추천 연속성 (sequential recommendations)에서 사용자 피로도 (user fatigue)를 모델링하는 FRec를 제안합니다. FRec는 관심 기반 유사성 매트릭스와 피로 게이트 순환 유닛 (fatigue-gated recurrent unit)을 사용하여 장기 및 단기 관심 학습을 향상시키고, 명시적 피로 신호를 추출하기 위한 새로운 시퀀스 증강 방법을 소개합니다.

- **Technical Details**: 첫 번째 도전 과제는 사용자 피로도를 지원하는 기능을 식별하는 것입니다. 이를 위해 다중 관심 프레임워크를 기반으로, 타겟 아이템과 역사적 아이템을 연결하고, 피로도 모델링을 지원하는 관심 기반 유사성 매트릭스 (interest-aware similarity matrix)를 구성합니다. 두 번째 과제는 피로도가 사용자 관심에 미치는 영향을 이해하는 것입니다. 이를 위해 피로 보강 다중 관심 융합 모델 (fatigue-enhanced multi-interest fusion)을 제안하여 장기 관심을 캡처합니다. 단기 관심 학습을 위해서는 피로 게이트 순환 유닛 (fatigue-gated recurrent unit)을 개발하여 피로 표현을 중요 입력으로 사용합니다. 마지막으로, 명시적 피로 신호를 얻기 위해 새로운 시퀀스 증강 방법을 적용하여 대조 학습 (contrastive learning)을 수행합니다.

- **Performance Highlights**: 본 논문에서는 두 개의 공공 데이터 세트와 하나의 대규모 산업 데이터 세트를 사용하여 실험을 진행하였으며, 그 결과 FRec는 최고 수준의 모델들과 비교하여 AUC (Area Under Curve)에서 0.026, GAUC (Group Area Under Curve)에서 0.019만큼 성능을 개선하였습니다. 또한, 대규모 온라인 실험에서도 FRec는 사용자 피로도를 줄이고 사용자 경험을 향상시키는 데 효과적임을 입증했습니다.



### Sociotechnical Implications of Generative Artificial Intelligence for Information Access (https://arxiv.org/abs/2405.11612)
- **What's New**: 이번 논문에서는 Generative AI(생성적 인공지능) 기술이 정보 접근에 미치는 시스템적 결과와 위험성을 종합적으로 분석하고 평가 및 완화에 대한 권고 사항을 제공합니다. 이는 지식 생산, 공중 보건 교육 및 민주 사회의 정보 시민양성에 중요한 역할을 합니다.

- **Technical Details**: Generative AI, 특히 Large Language Models(LLMs),는 새로운 정보 접근 방식을 가능하게 하며 기존 정보 검색 시스템의 효과성을 향상시킬 수 있습니다. 그러나 이러한 기술의 장기적인 사회적 영향에 대한 이해는 아직 초기 단계에 있습니다. 이 논문은 CMR(결과-메커니즘-위험) 프레임워크를 적용하여 시스템적 렌즈를 통해 Generative AI의 메커니즘과 결과, 그리고 관련된 위험을 구조화합니다.

- **Performance Highlights**: 이 논문은 Generative AI가 저비용으로 대규모 저질 콘텐츠를 생성할 수 있어 정보 생태계의 오염을 초래할 수 있음을 강조합니다. 이러한 콘텐츠는 믿을 만한 정보와 구분하기 어렵게 만들며 정보 신뢰성을 저해할 수 있습니다. 또한, 대화형 검색 인터페이스에서 LLM이 사용될 때 발생하는 '게임 오브 텔레폰' 현상으로 인한 정보 왜곡 및 오류 위험도 논의됩니다.



### Knowledge Graph Pruning for Recommendation (https://arxiv.org/abs/2405.11531)
- **What's New**: 최근 몇 년간 지식 그래프 기반 추천 시스템(KGRS)의 발전은 사용자, 아이템 및 엔티티의 표현을 구조적 지식을 통해 풍부하게 하여 성능을 크게 향상시켰습니다. 하지만 높은 컴퓨팅 비용이 더 정교한 모델을 탐구하는 것을 제한하고 있습니다. 이에 대응하기 위해, 새로운 접근 방식인 KGTrimmer가 제안되었습니다. 이는 추천 시스템에 맞춘 지식 그래프 가지치기(pruning) 방법으로, 성능 저하를 최소화하면서 불필요한 노드를 제거합니다.

- **Technical Details**: KGTrimmer는 이중 관점(dual-view perspective)에서 중요도 평가기를 설계합니다. 집합적 관점에서는 다수의 사용자로부터 주목받은 노드를 중요한 노드로 간주하는 '집단 지성' 아이디어를 채택하고, 전체 관점에서는 노드의 고유한 속성이나 인기도를 기반으로 글로벌 마스크(global mask)를 학습하여 가치 없는 노드를 식별합니다. 이후, 필터링 메커니즘을 갖춘 엔드 투 엔드 중요도 인식 그래프 신경망을 구축하여 사용자-아이템 상호작용을 활용하고, 구체적인 지식을 추출하여 추천 시스템의 효율성과 정확성을 향상시킵니다.

- **Performance Highlights**: KGTrimmer는 세 개의 공개 데이터셋에서 실행된 광범위한 실험을 통해 그 효과와 일반화 능력을 입증했습니다. 이는 기본적인 사실을 유지하면서도 추천 시스템을 더 잘 지원하는 경량, 안정적, 그리고 강력한 지식 그래프를 생성합니다.



### Double Correction Framework for Denoising Recommendation (https://arxiv.org/abs/2405.11272)
Comments:
          Accepted by KDD 2024

- **What's New**: 기존에 사용되던 노이즈 샘플(noisy samples) 제거 방식을 개선한 '이중 보정 프레임워크(Dual Correction Framework, DCF)'를 제안합니다. DCF는 더 정확한 샘플 제거 및 데이터 희소성을 줄이기 위한 두 가지 보정 요소를 포함하고 있습니다. 이를 통해 추천 시스템(recommender systems)에서 노이즈가 많은 샘플을 안정적으로 제거하고, 대신 '하드 샘플(hard samples)'을 재사용하여 성능을 높이는 것을 목표로 합니다.

- **Technical Details**: DCF는 '샘플 제거 보정(sample dropping correction)'과 '점진적 라벨 보정(progressive label correction)' 두 가지 보정 요소로 구성되어 있습니다. 샘플 제거 보정은 시간 경과에 따른 샘플의 손실 값(loss value)을 사용하여 노이즈 여부를 결정하는 방식으로, 손실의 평균을 계산하고 댐핑 함수(damping function)를 적용하여 이상치(outliers)의 영향을 줄입니다. 또한, '응집 부등식(concentration inequalities)'을 사용하여 높은 손실 변동성을 보이는 하드 샘플을 식별합니다. 점진적 라벨 보정은 처음에는 소수의 샘플을 재라벨링(relabel)하고, 과정을 거듭하면서 재라벨링 비율을 점차 증가시킵니다.

- **Performance Highlights**: 세 가지 데이터셋(dataset)과 네 가지 백본(backbones)에서의 광범위한 실험 결과, 제안된 DCF 프레임워크가 기존 방법들보다 더 나은 성능과 일반성을 보였습니다. 특히 노이즈 샘플 제거와 하드 샘플 재사용을 통해 추천 시스템의 정확도를 향상시켰습니다.



### The MovieLens Beliefs Dataset: Collecting Pre-Choice Data for Online Recommender Systems (https://arxiv.org/abs/2405.11053)
- **What's New**: 이번 연구는 추천 시스템이 소비자 선택에 미치는 영향을 고려하는 방법을 다룹니다. 연구자는 사용자가 경험하지 않은 항목에 대한 믿음 데이터를 수집하는 방법을 소개하며, 이를 통해 더 효과적인 추천 시스템을 개발하는 데 기여하고자 합니다. 이 방법은 MovieLens 플랫폼에서 구현되어, 사용자 평가, 믿음, 그리고 추천 사항을 포함하는 풍부한 데이터셋을 제공합니다. 이 데이터셋은 추천이 사용자 선택에 미치는 영향을 측정하고, 사용자 믿음 데이터를 활용한 알고리즘을 프로토타입화하는 데 유용합니다.

- **Technical Details**: 연구는 사용자 선택 행동을 이해하고 추천 시스템의 작동 메커니즘을 설명하기 위해 사용자 믿음 데이터가 유용하다는 점에서 출발합니다. 연구자는 효율적으로 믿음 데이터를 수집하기 위한 절차를 제공하고, 이를 MovieLens 플랫폼에 적용하여 오픈소스 데이터셋을 생성했습니다. 사용자 결정 모델을 바탕으로 어떤 데이터를 수집할지, 어떻게 다양한 데이터를 생성할지 가이드하는 방식으로 절차를 설계했습니다. 각 사용자는 서비스 내에서 주어진 시점에 다양한 상품을 평가하며, 추천 사항은 사용자 믿음에 직접적인 영향을 미칩니다. 이를 통해 사용자의 최종 선택을 예측할 수 있습니다.

- **Performance Highlights**: 제안된 데이터셋은 기존의 MovieLens 평가 데이터셋을 보완하며, 사용자의 선호도를 보다 명확하게 이해할 수 있도록 합니다. 사용자가 어떤 이유로 특정 항목을 소비하지 않는지를 구분할 수 있기 때문에 추천 평가 기준을 설계하고 테스트하는 데 도움이 됩니다. 특히, 사용자가 항목의 품질에 대해 불확실성을 가지고 있는지, 아니면 특정 유형의 항목을 좋아하지 않는지를 구분할 수 있습니다. 이를 통해 보다 사용자 중심적인 추천 시스템 설계가 가능해집니다.



### Optimistic Query Routing in Clustering-based Approximate Maximum Inner Product Search (https://arxiv.org/abs/2405.12207)
- **What's New**: 이 연구는 클러스터링 기반의 최대 내부곱 검색(MIPS)에서 라우팅 프로토콜의 중요성을 탐구하고 새로운 프레임워크를 제시합니다. 기존 라우팅 프로토콜을 분석하고 '불확실한 상황에서의 낙관주의(optimism in the face of uncertainty)' 원칙을 적용하여 개선된 알고리즘을 제안합니다.

- **Technical Details**: 기존 라우팅 프로토콜의 한계를 발견하고, 데이터셋의 내부곱 분포의 모멘트(moment)를 사용하여 내부곱을 낙관적으로 추정하는 방법을 제안합니다. 구체적으로, 알고리즘은 각 섀드(shard)의 첫 두 모멘트만을 사용하여 최신 라우터(SCANN)와 동일한 정확도를 달성하며, 프로빙(probing)하는 데이터 포인트 수를 최대 50% 줄였습니다. 또한, 제안된 알고리즘은 공간 효율적입니다. 두 번째 모멘트를 독립적인 사이즈로 스케치(Sketch)하여 실시간으로 추가적인 벡터를 적재합니다.

- **Performance Highlights**: 제안된 알고리즘은 벤치마크 MIPS 데이터셋에서 최대 50% 적은 데이터 포인트를 프로빙하면서, 기존 최상위 라우터와 유사한 정확도를 유지합니다. 또한 알고리즘은 공간 효율성 측면에서도 우수하여, 실습에서 각 섀드 당 O(1)개의 추가 벡터를 저장할 정도로 소형화되었습니다.



### Adaptive Extraction Network for Multivariate Long Sequence Time-Series Forecasting (https://arxiv.org/abs/2405.12038)
- **What's New**: 최근 다변량 장기 시계열 예측(multivariate long sequence time-series forecasting, MLSTF) 모델에서 CNN 아키텍처 모델들이 현저한 발전을 이루었지만, 전역 시계열 패턴과 변수 간 상관관계를 추출하는 능력이 부족했습니다. 이를 해결하기 위해, 다양한 해상도의 시계열 특성을 캡처할 수 있는 다중 해상도 컨볼루션 및 변형 가능 컨볼루션(deformable convolution) 연산을 도입했습니다. 이를 바탕으로, 우리는 ATVCNet이라는 적응형 시간-변수 컨볼루션 네트워크를 제안하여 다변량 시계열의 국지적/전역 시간 종속성 및 변수 간 종속성을 효과적으로 모델링합니다.

- **Technical Details**: ATVCNet의 주요 모듈은 두 가지입니다. 첫째, 시간 특성 추출 모듈은 다양한 해상도에서 국지적 문맥 정보와 패턴을 추출하기 위해 서로 다른 확장 요소(dilation factors)를 가진 CNN 커널을 사용합니다. 둘째, 변수 간 특성 적응형 추출 모듈은 이미지 처리 분야에서 자주 사용되는 변형 가능 컨볼루션(deformable convolution)을 도입하여 변수 간 교차 상관 관계를 강화합니다. 이를 통해 네트워크는 국지적 및 전역 시계열 패턴을 더욱 효과적으로 캡처할 수 있습니다.

- **Performance Highlights**: 8개의 실제 데이터셋에서 ATVCNet을 평가한 결과, ATVCNet은 기존 최첨단(SoTA) MLSTF 모델들보다 약 63.4%의 성능 향상을 달성했습니다.



### Towards Graph Contrastive Learning: A Survey and Beyond (https://arxiv.org/abs/2405.11868)
- **What's New**: 최근 몇 년간, 그래프 기반의 딥러닝이 다양한 도메인에서 큰 성과를 이루었지만, 주석이 달린 그래프 데이터(annotated graph data)에 대한 의존이 여전히 큰 병목 현상으로 남아있다. 이를 해결하기 위해, 셀프-슈퍼바이즈드 러닝(self-supervised learning, SSL)이 그래프에서 주목받고 있으며, 그 중에서도 그래프 대조 학습(Graph Contrastive Learning, GCL)에 대한 조사가 부족했다. 이번 서베이는 이러한 격차를 메우고자 GCL에 대한 포괄적인 개요를 제공한다.

- **Technical Details**: 이 논문에서는 GCL의 기본 원칙을 다루고, 데이터 증강(data augmentation) 전략, 대조 모드(contrastive modes), 대조 최적화 목표(contrastive optimization objectives)를 포함한 다양한 측면을 설명한다. 또한, 약하게 지도된 학습(weakly supervised learning), 전이 학습(transfer learning) 등 다른 데이터 효율적 그래프 학습으로의 확장도 탐구한다. 실질적인 응용 분야로는 약물 발견(drug discovery), 유전체 분석(genomics analysis), 추천 시스템(recommender systems) 등이 포함된다.

- **Performance Highlights**: GCL의 주요 성과를 통해 높은 성능을 입증하였으며, 이는 다양한 도메인에서의 효과적인 적용 가능성을 보여준다. 또한, 현재의 도전 과제 및 미래 방향에 대한 논의도 포함하고 있어, 이 분야의 연구자들에게 유익한 방향성을 제공한다.



### Increasing the LLM Accuracy for Question Answering: Ontologies to the Rescue! (https://arxiv.org/abs/2405.11706)
Comments:
          16 pages

- **What's New**: 대규모 언어 모델(LLMs)을 사용한 질문 응답(QA) 시스템에서 기존 SQL 데이터베이스보다 지식 그래프/의미 표현을 사용한 시스템이 더 높은 정확도를 보였습니다. 본 연구에서는 온톨로지 기반 쿼리 검사(OBQC)와 LLM 수정을 통해 SPARQL 쿼리의 오류를 감지하고 수정하는 새로운 접근 방식을 도입하여 정확도를 72%까지 향상시켰습니다.

- **Technical Details**: 온톨로지 기반 쿼리 검사(OBQC)는 지식 그래프의 의미론을 활용하여 SPARQL 쿼리가 온톨로지와 일치하는지 확인합니다. LLM 수정은 오류 설명을 사용하여 SPARQL 쿼리를 수정합니다. 이 방법은 데이터를 통해 채팅하는 벤치마크에서 평가되어 8%의 '모름' 결과를 포함한 경우에도 전체 오차율이 20%로 감소했습니다.

- **Performance Highlights**: OBQC와 LLM 수정을 통한 접근 방식은 이전의 54%에서 72%로 정확도를 크게 향상시켰습니다. 비록 복잡한 쿼리에서도 오차율이 상당히 감소했으며, 특히 낮은 복잡도의 스키마에서 오차율은 10.46%로 사용자들이 수용할 수 있는 수준에 도달했습니다. 이는 지식 그래프와 온톨로지 및 의미론이 LLM 기반 질문 응답 시스템의 더 높은 정확성을 달성하는 데 중요한 요소임을 보여줍니다.



### On the Convergence of No-Regret Dynamics in Information Retrieval Games with Proportional Ranking Functions (https://arxiv.org/abs/2405.11517)
- **What's New**: 이 논문에서는 프로포셔널 랭킹 함수(proportional ranking function)의 concave 활성화 함수(activation function)가 있는 경우, no-regret 학습 동학이 수렴한다는 것을 증명합니다. 이는 콘텐츠 게시자와 사용자 간의 복지(empirical welfare trade-offs)를 연구하여 다양한 활성화 함수 선택에 따른 영향을 분석하고 있습니다.

- **Technical Details**: 논문은 콘텐츠 제공자의 전략적 행동을 온라인 학습 프레임워크(online learning framework) 내에서 모델링합니다. 중심 개념인 '후회(regret)'는 학습 에이전트의 성능을 평가하는 기준으로 사용됩니다. 랭킹 함수가 concave 활성화 함수를 가지는 경우, no-regret 학습 동학이 수렴하는 것을 증명했으며, 이 게임은 socially-concave 게임과 동등하다고 설명합니다. 콘텐츠 제공자의 노출률을 결정하는 메커니즘으로 사용되는 랭킹 함수를 중심으로 분석하였고, SEO 및 관련 검색 엔진 최적화(Search Engine Optimization) 사례를 논의하고 있습니다.

- **Performance Highlights**: 논문은 최신 no-regret 동학 알고리즘을 사용해 다양한 활성화 함수 선택에 따른 게시자와 사용자의 복지 수준 간의 트레이드오프를 실험적으로 연구했습니다. 여러 게임 매개변수(예: 페널티 계수, 게시자 수)에 따른 복지와 수렴 속도를 분석한 결과를 제시하였습니다.



### DEMO: A Statistical Perspective for Efficient Image-Text Matching (https://arxiv.org/abs/2405.11496)
- **What's New**: 분포 기반의 이미지-텍스트 매칭을 위해 새로운 해싱 접근법인 Distribution-based Structure Mining with Consistency Learning(DEMO)를 도입했습니다. DEMO는 각 이미지의 잠재적 의미 분포를 다중 보강 뷰를 통해 특징 짓고, 비모수적 distribution divergence를 사용하여 강력하고 정밀한 유사성 구조를 보장합니다. 또한 협업 일관성 학습을 통해 서로 다른 방향에서의 검색 분포 일관성을 자기 지도 방식으로 강조합니다.

- **Technical Details**: DEMO는 각 샘플의 잠재된 의미 분포를 다중 랜덤 보강을 이용해 탐구합니다. 구체적으로, 데이터 보강이 일반적으로 의미를 유지한다고 가정하고, 이미지를 내재적 의미 분포에서 추출된 샘플로 간주합니다. 비모수적 metric(에너지 거리)를 사용해 분포 다이버전스를 측정하고, 강력하고 정확한 의미 구조를 재구성합니다. 해싱 네트워크 최적화는 이 의미 구조를 Hamming 공간에서 보존하여 달성됩니다.

- **Performance Highlights**: 세 가지 벤치마크 이미지-텍스트 매칭 데이터셋에 대한 광범위한 실험을 통해 DEMO가 최첨단 방법들을 능가하는 성능을 보임을 입증했습니다. DEMO는 다양한 경쟁 방법들보다 우수한 성능을 보여줍니다.




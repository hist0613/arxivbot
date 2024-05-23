### Panmodal Information Interaction (https://arxiv.org/abs/2405.12923)
- **What's New**: 최근 생성형 인공지능(GenAI: Generative Artificial Intelligence)의 등장은 정보 상호작용 방식을 혁신하고 있습니다. 수십 년 동안 구글과 빙 같은 검색 엔진이 일반 대중이 관련 정보를 찾는 주요 수단이 되어왔습니다. 이들은 동일한 표준 형식(이른바 '10개의 파란 링크')으로 검색 결과를 제공해왔습니다. 그러나 최근 AI 기반 에이전트와의 자연어 채팅 기능과 실시간으로 상위 순위 결과에 기반한 답변을 자동으로 합성하는 GenAI의 능력이 사람들의 정보 상호작용과 소비 방식을 대규모로 변화시키고 있습니다.

- **Technical Details**: 이 논문은 기존의 검색(Traditional Search)과 AI 기반 채팅(AI-powered Chat)이라는 두 가지 정보 상호작용 방식이 현재 검색 엔진에 공존하고 있음을 다루고 있습니다. 두 방식은 느슨하게 결합되거나(예: 별도 옵션 또는 탭) 긴밀하게 결합되어(예: 기존 검색 결과 페이지에 직접 통합된 채팅 답변 형태) 제공됩니다. 이러한 두 가지 또는 그 이상의 방식의 공존은 검색 경험을 재구상하고 다양한 방식의 강점을 활용하여 이를 지원하는 시스템과 전략을 개발할 기회를 만들고 있음을 강조합니다. 우리는 이를 'Panmodal Experiences'라고 부릅니다. 이는 하나의 방식만 사용할 수 있는 Monomodal Experiences와 달리, 사용자에게 다중 방식(Multimodal)을 제공하고, 방식 간 전환을 직접 지원(Crossmodal)하며, 방식들을 무결하게 결합해 작업을 지원(Transmodal)합니다.

- **Performance Highlights**: 저자들은 100명이 넘는 개별 사용자의 설문 조사 데이터를 바탕으로 검색 및 채팅 방식의 일반 작업 수행에 대한 통찰을 제시합니다. 이를 통해 다양한 방식의 강점을 활용한 정보 소비의 미래 비전을 제안하며, GenAI의 emergent capabilities에 대해 논의합니다.



### Retrievable Domain-Sensitive Feature Memory for Multi-Domain Recommendation (https://arxiv.org/abs/2405.12892)
- **What's New**: 이 논문에서는 온라인 광고의 다중 도메인(ad recommendation) 추천 시스템에서 도메인 간 차이가 큰 특징(feature)들을 효과적으로 모델링하는 방법을 제안합니다. 이러한 특징을 도메인 민감 특성(domain-sensitive features)이라고 정의하며, 이는 다중 도메인 모델링에 매우 중요합니다. 기존의 다중 도메인 추천 시스템은 이들 특징을 충분히 반영하지 못할 수 있음을 지적합니다.

- **Technical Details**: 제안된 방법은 도메인 민감 특성을 식별하기 위한 도메인 민감 특성 기법(domain-sensitive feature attribution method)을 사용합니다. 또한, 이러한 특징에서 도메인별 정보를 추출하기 위한 메모리 아키텍처(memory architecture)를 설계하여 모델이 도메인 차이를 인지할 수 있도록 합니다. 이러한 방법으로, 특징 분포와 모델 예측에 대한 도메인 차이를 효과적으로 반영합니다.

- **Performance Highlights**: 광범위한 온라인 및 오프라인 실험을 통해, 제안된 방법이 도메인 구별을 잘 포착하고 다중 도메인 추천 성능을 개선하는 데 있어서 우수함을 입증하였습니다.



### Robust portfolio optimization model for electronic coupon allocation (https://arxiv.org/abs/2405.12865)
Comments:
          9 pages, 17 figures, AAAI-2024 Workshop on Artificial Intelligence for Operations Research

- **What's New**: 이번 연구는 온라인 쿠폰을 최적화된 방식으로 고객들에게 할당하는 문제를 다루고 있습니다. 특히 예산 제약 하에 효과적인 쿠폰 배포를 위해 고객 세분화에 기반한 견고한 포트폴리오 최적화 모델(robust portfolio optimization model)을 적용하였습니다. 연구는 실제 데이터로 무작위로 배포된 쿠폰을 사용한 수치 실험을 통해 방법의 유효성을 검증했으며, 이는 기존의 다중선택 배낭 문제 모델(multiple-choice knapsack model) 및 전통적인 평균-분산 최적화 모델(mean-variance optimization model)보다 더 큰 판매 상승효과(uplift)를 달성했다는 것을 입증했습니다.

- **Technical Details**: 연구에서 핵심적으로 다룬 두 가지 기여는 첫째, 여섯 종류의 쿠폰을 다루어 다양한 쿠폰의 효과 차이를 정확하게 추정하는 것이 매우 어렵다는 점을 제시한 것이며, 둘째, 이러한 복잡한 환경에서도 견고한 최적화 모델이 실질적인 쿠폰 할당에서 더 큰 판매 상승효과를 달성했다는 것을 수치적인 결과로 보여준 점입니다.

- **Performance Highlights**: 실제 데이터 기반의 수치 실험 결과, 견고한 포트폴리오 최적화 모델이 기존의 다중선택 배낭 문제 모델(multiple-choice knapsack model) 및 전통적인 평균-분산 최적화 모델(mean-variance optimization model)보다 더 높은 판매 상승(uplift)을 이뤘다는 것을 확인할 수 있었습니다. 이는 실질적인 쿠폰 할당에 있어서 강력한 도구로서의 잠재력을 보여줍니다



### A Dataset and Baselines for Measuring and Predicting the Music Piece Memorability (https://arxiv.org/abs/2405.12847)
- **What's New**: 새로운 연구는 음악의 기억 가능성(memorability)을 측정하고 예측하는 방법에 초점을 맞추고 있습니다. 이를 위해 신뢰할 수 있는 기억 가능성 라벨을 가진 새로운 음악 데이터셋을 수집하고, 새로운 양방향 실험 절차를 사용하고 있습니다.

- **Technical Details**: 본 연구는 데이터 기반 딥러닝(deep learning) 방법을 통해 음악의 기억 가능성을 예측하는 최초의 시도입니다. 예측을 위해 해석 가능한 특징(interpretable features)과 오디오 멜 스펙트로그램(audio mel-spectrograms)을 입력으로 사용하는 기본 모델을 훈련시켰습니다. 일련의 실험과 절차 제거 연구(ablation studies)를 통해 제시된 방법의 성능을 분석했습니다.

- **Performance Highlights**: 예측 기술이 아직 개선의 여지가 있지만, 제한된 데이터로도 음악의 기억 가능성을 예측할 수 있음을 보여주었습니다. 높은 유쾌함(valence), 각성(arousal), 빠른 템포(tempo) 등 특정 내재적 요소가 기억에 남는 음악에 기여함을 발견했습니다. 이러한 연구는 음악 추천 시스템이나 음악 스타일 변환(music style transfer)과 같은 실생활 응용 프로그램에 큰 도움이 될 것입니다.



### GotFunding: A grant recommendation system based on scientific articles (https://arxiv.org/abs/2405.12840)
- **What's New**: 이번 연구에서는 연구자들이 적합한 연구비를 받을 수 있도록 돕는 추천 시스템인 	extsc{GotFunding}(Grant recOmmendaTion based on past FUNDING)을 소개했습니다. 이 시스템은 과거 연구비 및 논문 기록을 학습하여 NIH(National Institutes of Health)의 데이터로부터 최고 성능을 달성했습니다. 시스템은 연구비와 논문 매칭을 자동화하는 데 목표를 두고 있습니다.

- **Technical Details**: GotFunding 시스템은 학습 순위 결정(learning to rank) 문제로 접근하여 개발되었으며, 주요 특징으로는 1) 논문과 연구비 간의 연도 차이, 2) 논문에서 제공하는 정보의 양, 3) 논문과 연구비의 관련성을 반영합니다. 이러한 요소들이 예측 성능을 향상시키는 중요한 역할을 합니다. 시스템 성능은 NDCG@1 = 0.945로 매우 높은 수준입니다.

- **Performance Highlights**: GotFunding 시스템은 NIH의 연구비 및 출판 기록 데이터를 활용하여 높은 성능을 입증했으며, 특히 높은 정확도로 연구비 매칭 추천을 제공합니다. 과학자들을 위한 온라인 도구로 개발될 예정이며, 미래의 개선 방안도 논의되고 있습니다.



### Disentangled Representation with Cross Experts Covariance Loss for Multi-Domain Recommendation (https://arxiv.org/abs/2405.12706)
- **What's New**: 이 연구에서는 Multi-domain learning (MDL)에서 공통 특성을 학습하면서 각 도메인의 고유한 특성을 유지하는 어려움을 해결하기 위해 새로운 모델 'Crocodile'을 제안합니다. Crocodile은 Cross-experts Covariance Loss for Disentangled Learning의 약자로, 다중 임베딩 패러다임을 채택하여 모델 학습을 촉진하고, Covariance Loss를 사용하여 이러한 임베딩을 분리합니다. 또한, 새로운 게이팅 메커니즘(gating mechanism)을 도입하여 모델의 역량을 더욱 강화합니다.

- **Technical Details**: Crocodile 모델은 다양한 사용자 관심사를 효과적으로 포착하기 위해 다중 임베딩 패러다임을 사용합니다. 이 모델은 임베딩에서 Covariance Loss를 활용하여 임베딩을 분리(Disentangled Learning)하며, 이는 각 도메인의 고유한 특성을 유지하면서 공통 특성을 학습하는 균형을 맞추기 위해 필요합니다. 제안된 게이팅 메커니즘은 모델의 성능을 추가로 향상시킵니다.

- **Performance Highlights**: 실증적 분석을 통해 Crocodile이 제안한 방법이 두 가지 문제를 성공적으로 해결하고, 공개된 데이터셋에서 모든 최신 방식(state-of-the-art methods)을 능가함을 입증했습니다. 본 연구에서 제시하는 분석 관점과 설계 개념은 MDL 분야의 향후 연구에 길을 열 수 있을 것으로 확신합니다.



### Time Matters: Enhancing Pre-trained News Recommendation Models with Robust User Dwell Time Injection (https://arxiv.org/abs/2405.12486)
Comments:
          10 pages,5 figures

- **What's New**: 이 논문에서는 효과적인 사용자 선호도 모델링을 위해 대형 언어 모델(LLMs)를 사용한 최첨단 뉴스 추천 모델을 제안하고 있습니다. 특히 사용자 행동의 복잡성과 불확실성을 고려하여, 클릭(cick) 신호 외에 사용자 체류 시간(dwell time)을 활용하는 독창적인 전략을 소개합니다.

- **Technical Details**: 이 논문은 사용자 체류 시간을 보다 효과적으로 모델에 통합하기 위해 두 가지 새로운 전략인 Dwell time Weight (DweW)와 Dwell time Aware (DweA)를 제안합니다. DweW는 체류 시간을 통해 'Effective User Clicks'를 정밀하게 분석하고 초기 사용자 행동 데이터를 통합하여 더욱 강력한 사용자 선호도를 모델링합니다. DweA는 모델이 자동으로 체류 시간 정보를 인식하고 이에 맞게 주의(attention) 값을 조정할 수 있도록 합니다.

- **Performance Highlights**: MSN 웹사이트의 실제 뉴스 데이터를 사용한 실험 결과, 제안된 두 가지 전략은 높은 품질의 뉴스를 추천하는 데 있어서 성능을 크게 향상시켰으며, 사용자의 체류 시간 정보가 부족하거나 전혀 없는 극단적인 경우에서도 높은 추천 성능을 유지했습니다.



### Learning Partially Aligned Item Representation for Cross-Domain Sequential Recommendation (https://arxiv.org/abs/2405.12473)
- **What's New**: 크로스 도메인 순차 추천(CDSR)은 사용자들의 순차적 선호도를 여러 추천 도메인에서 발견하고 전송하는 기술입니다. 이번에 소개된 연구에서는 특히 항목 표현(item representation)의 정렬 문제에 중점을 두었습니다. 이를 위해 CA-CDSR(Cross-domain item Alignment for Cross-Domain Sequential Recommendation)라는 모델 독립적 프레임워크를 제안했습니다. 이 프레임워크는 항목 표현의 순차적 생성 및 적응적 부분 정렬을 달성합니다.

- **Technical Details**: CA-CDSR은 두 가지 주요 전략을 사용합니다: 첫째, 시퀀스 인지 기능 확장(strategy), 둘째, 적응적 스펙트럼 필터(spectrum filter)입니다. 시퀀스 인지 기능 확장은 협업 및 순차적 항목 상관관계를 모두 포착하여 전체적인 항목 표현 생성을 촉진합니다. 이어서 스펙트럼 관점에서 부분 표현 정렬 문제를 조사하는 실험 연구를 기반으로 스펙트럼 필터를 설계했습니다. 이 필터는 적응적으로 부분 정렬을 달성합니다. 이렇게 정렬된 항목 표현은 다양한 시퀀셜 인코더에 입력되어 사용자 표현을 얻게 됩니다. 전체 프레임워크는 멀티 테스크 학습 패러다임에 최적화되어 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 CA-CDSR은 최첨단 베이스라인을 큰 폭으로 초과하는 성능을 나타냈습니다. 또한 항목들을 표현 공간에서 효과적으로 정렬하여 성능을 향상시킬 수 있음을 입증했습니다.



### Learning Structure and Knowledge Aware Representation with Large Language Models for Concept Recommendation (https://arxiv.org/abs/2405.12442)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문은 학습자에게 다음에 공부할 개념을 추천하는 새로운 방식인 SKarREC(Structure and Knowledge Aware Representation learning framework for concept Recommendation)를 제안합니다. 기존 접근 방식들이 인간 지식 체계를 효과적으로 통합하지 못한 문제를 해결하고자, LLM(Large Language Models)을 사용해 사실적 지식을 활용하고 개념 간의 선후 관계를 텍스트로 표현하도록 설계되었습니다.

- **Technical Details**: SKarREC는 지식 그래프(knowledge graph)에서 얻은 개념들 사이의 선후 관계와 LLM이 생성하는 사실적 지식을 사용하여 텍스트 표현을 구축합니다. 이를 위해 그래프 기반 어댑터(graph-based adapter)를 도입하여 비균질한(anisotropic) 텍스트 임베딩을 개념 추천 작업에 맞게 조정합니다. 이 어댑터는 대조 학습(contrastive learning)을 통해 사전 학습되고, 추천 작업을 통해 미세 조정(fine-tuned)되어 텍스트에서 지식, 추천으로 이어지는 적응 파이프라인을 형성합니다.

- **Performance Highlights**: SKarREC는 기존의 어댑터보다 더 나은 성능을 발휘하여, 텍스트 인코딩을 개념 추천에 효과적으로 변환합니다. 실제 데이터를 기반으로 한 광범위한 실험에서 제안된 접근 방식의 효과성이 입증되었습니다.



### Diversifying by Intent in Recommender Systems (https://arxiv.org/abs/2405.12327)
- **What's New**: 이번 연구에서는 주로 단기 참여(short-term engagement)에 초점을 맞춘 추천 시스템이 장기적인 사용자 경험에 악영향을 끼칠 수 있다는 문제를 해결하려고 합니다. 이를 위해 여러 상호작용이나 추천 세션에서 지속될 수 있는 사용자 의도(user intents)를 활용하는 방법을 제안합니다. 사용자의 탐색 의도(exploration intent)를 고려한 전체 페이지 다변화 프레임워크(intent-based page diversification framework)를 개발하여 장기적인 사용자 경험을 최적화하려고 합니다.

- **Technical Details**: 우리의 다변화 프레임워크는 확률적(intent-based probabilistic) 접근 방식을 사용하여 초기 사용자 의도의 사전 신뢰(prior belief)를 바탕으로 각 위치에서 항목을 선택하고, 후속적으로 의도에 대한 사후 신뢰를 업데이트합니다. 이 접근 방식은 페이지에 다양한 사용자 의도를 반영하여 장기적인 사용자 경험을 최적화합니다. 특히, 사용자가 새로운 관심사와 콘텐츠를 탐험하려는 성향을 포착하여 사용자 탐색 의도를 시스템에 통합합니다.

- **Performance Highlights**: 세계 최대 콘텐츠 추천 플랫폼 중 하나에서 진행한 라이브 실험 결과, 제안된 프레임워크는 사용자 유지(user retention)와 전반적인 사용자 만족도를 증가시키는 것으로 나타났습니다. 사용자가 일관되게 다양한 콘텐츠를 발견하고 그에 맞는 참여를 유도하여 장기적인 사용자 경험을 향상시키는 데 효과가 있음을 확인했습니다.



### Address-Specific Sustainable Accommodation Choice Through Real-World Data Integration (https://arxiv.org/abs/2405.12934)
Comments:
          8 pages

- **What's New**: 소비자들이 지속 가능한 숙박 시설을 선택할 수 있도록 돕는 의사 결정 지원 시스템을 제안합니다. 'EcoGrade'라는 데이터 기반의 주소별 지표(metric)를 개발하여, 국가 승인 데이터셋을 통합하고 데이터가 부족한 경우 보간(interpolation) 방식을 사용합니다. 이 시스템은 실제 사용자에 의해 몇 달간 테스트되었으며 긍정적인 피드백을 받았습니다.

- **Technical Details**: 'EcoGrade'는 10개 도시에 걸쳐 10,000개의 영국 주소에서 지표의 정확성을 검증했으며, 보간 방식이 실제 상황과 통계적으로 유의미하게 일치함을 보여줍니다. 이 지표는 정부 승인 데이터셋을 활용하며, 주소별로 세세한 정보를 제공하여 지속 가능한 선택을 가능하게 합니다.

- **Performance Highlights**: 'EcoGrade'를 임베딩한 의사 결정 지원 시스템이 글로벌 숙박 시장에서 프록시 사용자들(real users)에 의해 몇 달간 테스트 되었고, 사용자들로부터 긍정적인 피드백을 받았습니다. 이 시스템은 건물 소유주들이 숙박 시설을 더욱 효율적으로 만들도록 장려하며, 빠르게 변화할 수 있는 임대 부문에 특히 긍정적인 영향을 미칠 것으로 기대됩니다.



### RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search (https://arxiv.org/abs/2405.12497)
Comments:
          The paper has been accepted by SIGMOD 2024

- **What's New**: 최신 연구에서 고차원 유클리드 공간 내에서 근삿값 최근접 이웃 검색(ANN: Approximate Nearest Neighbors)을 다루는 새로운 랜덤화 양자화 방법인 RaBitQ를 제안했습니다. 이 방법은 D-차원 벡터를 D-비트 문자열로 양자화합니다. 이 방식을 통해 이론적인 오류 한계를 보장하면서 실험적으로도 높은 정확도를 확보할 수 있습니다.

- **Technical Details**: RaBitQ는 SIMD 기반 연산(SIMD-based operations)이나 비트 단위 연산(bitwise operations)을 통해 거리를 추정할 수 있도록 효율적인 구현 방식을 도입했습니다. 이는 Product Quantization (PQ)와 그 변형법들에 비해 엄격한 이론적 오류 범위를 제공한다는 점에서 차별화됩니다. PQ는 기존에 많은 성공을 거두었으나, 이론적 오류 한계를 가지지 않아 일부 실제 데이터셋에서는 실패하는 경우가 관찰되었습니다.

- **Performance Highlights**: 실제 데이터셋을 대상으로 한 광범위한 실험에서 RaBitQ가 정확도와 효율성 면에서 PQ 및 그 변형법들을 넘어서는 성능을 보였습니다. 또한 RaBitQ의 실험적 성능이 이론적 분석과 잘 일치한다는 점이 확인되었습니다.



### Towards Detecting and Mitigating Cognitive Bias in Spoken Conversational Search (https://arxiv.org/abs/2405.12480)
- **What's New**: 이 논문은 음성 기반 상호작용 (Spoken Conversational Search; SCS)에서 사용자의 상호작용을 이해하기 위한 새로운 통찰을 제공합니다. 특히, 스크린 기반 검색 엔진과 달리, 음성 전용 채널에서의 사용자-시스템 상호작용은 캡처하는 것이 어렵습니다. 정보 과부하 시대에서 인지 편향 (cognitive bias)이 정보 검색과 소비에 미치는 영향을 탐구하고자 합니다. 이 논문은 정보 탐색, 심리학, 인지과학, 그리고 웨어러블 센서 등의 다양한 학문에서 얻은 통찰을 적용하여 새로운 논의를 촉진하고자 합니다.

- **Technical Details**: 이 논문은 다중 모달 도구 (multimodal instruments)와 실험 설계 및 설정을 위한 방법을 포함하는 프레임워크를 제안합니다. 상호작용을 효과적으로 및 정밀하게 캡처하는 도구가 부족한 문제를 해결하고자 하는 방안을 모색합니다. 초기 결과를 예로 들어 프레임워크의 가능성을 보여줍니다. 또한, 다중 모달 접근 방식의 채택에 따르는 도전 과제와 윤리적 고려 사항을 설명합니다.

- **Performance Highlights**: 예비 결과는 제안된 프레임워크가 SCS에서 인지 편향을 탐구하는 데 유용할 수 있음을 시사합니다. 다중 모달 도구의 사용이 사용자의 인지적 반응을 더 잘 이해하는 데 도움이 될 수 있으며, 이는 더 나은 검색 시스템 개발에 기초가 될 것입니다.




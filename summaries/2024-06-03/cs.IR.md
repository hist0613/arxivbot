### SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation (https://arxiv.org/abs/2405.20878)
Comments:
          Accepted by SIGIR'24

- **What's New**: 순차 추천(Sequential Recommendation)은 사용자의 시간적 및 순차적 상호작용 패턴을 모델링하여 정보 과부하 문제를 효과적으로 해결합니다. 최근의 방법들은 추천 시스템에서 자기 지도 학습(Self-Supervised Learning) 기술을 도입하고 있지만, 여전히 해결되지 않은 두 가지 중요한 문제가 있습니다. 본 논문에서는 이 문제를 해결하기 위해 SelfGNN(Self-Supervised Graph Neural Network)라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: SelfGNN 프레임워크는 시간 간격(Time Intervals)에 따라 단기 그래프를 인코딩하고, 그래프 신경망(GNNs)을 사용하여 단기 협력 관계를 학습합니다. 또한 간격 융합(Interval Fusion) 및 동적 행위 모델링을 통해 여러 세밀성 수준에서 장기 사용자 및 아이템 표현을 캡처합니다. 특히, 개인화된 자기 증강 학습 구조는 장기 사용자 관심사 및 개인적 안정성을 기반으로 단기 그래프의 노이즈를 완화하여 모델의 강건성을 향상시킵니다.

- **Performance Highlights**: 현실 세계의 4개의 데이터셋에서 광범위한 실험을 통해 SelfGNN이 다양한 최첨단 모델들을 능가하는 성능을 보였습니다. 모델 구현 코드는 공개되어 있어 연구자들이 참고할 수 있습니다.



### Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias (https://arxiv.org/abs/2405.20718)
Comments:
          Accepted by KDD 2024

- **What's New**: 이번 연구에서는 추천 시스템에서 흔히 발생하는 인기도 편향(popularity bias)을 해결하기 위해 '인기도-인식 정렬 및 대조 (Popularity-Aware Alignment and Contrast, PAAC)' 기법을 제안합니다. 이 기법은 인기 있는 항목의 감독 신호(supervisory signals)를 활용하여 인기가 없는 항목의 표현을 향상시키고, 대조 학습 손실(contrastive learning loss)을 재조정하여 인기 기준의 표현 분리를 완화합니다.

- **Technical Details**: PAAC 모델은 다음의 주요 모듈로 구성됩니다. 첫째, '감독 정렬 모듈(Supervised Alignment Module)'은 인기 있는 항목의 공통 감독 신호를 사용하여 인기가 없는 항목의 표현을 향상시킵니다. 둘째, '재조정 대조 모듈(Re-weighting Contrast Module)'은 인기도 수준(popularity levels)에 따른 표본의 가중치를 조절하여 표현 분리를 완화합니다. 이를 통해 모델은 사용자가 상호작용한 항목의 특성을 공통 신호로 활용하여 인기와 비인기 항목 간의 정렬을 유지합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 광범위한 실험을 통해 PAAC의 효과를 검증하였습니다. 결과적으로, PAAC는 인기도 편향을 완화하고, 기존 방법 대비 추천 정확도를 높이는 데 성공하였습니다.



### Information Maximization via Variational Autoencoders for Cross-Domain Recommendation (https://arxiv.org/abs/2405.20710)
- **What's New**: 새로운 Cross-Domain Sequential Recommendation(CDSR) 프레임워크인 Information Maximization Variational Autoencoder(IM-VAE)가 도입되었습니다. 이 연구에서는 사용자의 인터랙션(Interaction) 기록을 강화하기 위해 Pseudo-Sequence Generator를 사용하고, mutual information maximization(MIM) 이론에서 영감을 받은 세 가지 정규화 기법을 제안합니다. 이는 사용자의 도메인 간에 공유되는 관심사와 특정 도메인에 제한된 관심사 간의 의미적 차이를 캡처하고, 실제 인터랙션 시퀀스와 생성된 의사-시퀀스(pseudo-sequences) 간의 정보 격차를 해결합니다.

- **Technical Details**: IM-VAE는 Pseudo-Sequence Generator와 VAE(Variational Autoencoder), 그리고 mutual information maximization 이론에 의해 유도된 세 가지 정보 정규화 기법(정보성, 분리성, 잡음 제거)을 포함합니다. 이 모델은 의사-시퀀스를 생성하여 사용자 인터랙션 행동을 증강하고, 잡음이 있는 의사-시퀀스를 효과적으로 제거합니다. 이는 특히 Cold-start 사용자와 Long-tailed 사용자의 관심 정보를 효과적으로 탐구하고, 도메인 간 관심 정보를 분리하여 전달하는 데 중점을 둡니다.

- **Performance Highlights**: 	exttt{IM-VAE}는 두 개의 실제 Cross-domain 데이터셋에서 state-of-the-art 방법들을 능가하는 성능을 보여줍니다. Cold-start 사용자와 Long-tailed 사용자를 포함한 모든 종류의 사용자에서 뛰어난 성능을 입증하며, 이는 현실적인 추천 시나리오에서 IM-VAE의 효과를 확인시켜 줍니다.



### Causal Distillation for Alleviating Performance Heterogeneity in Recommender Systems (https://arxiv.org/abs/2405.20626)
Comments:
          TKDE 2023

- **What's New**: 이번 논문에서는 추천 성능의 사용자 간 성능 비대칭 문제가 발생하는 두 가지 원인을 밝혀내고, 이를 해결하기 위한 새로운 방법론 CausalD(Causal Multi-Teacher Distillation Framework)를 제안했습니다. 성능 비대칭 문제는 일부 사용자에게 추천 성능이 집중되는 현상으로, 이는 주로 사용자의 상호작용 분포 불균형과 모델 학습의 편향에서 기인합니다.

- **Technical Details**: 제안된 CausalD 프레임워크는 인과 이론의 '프론트-도어 조정(Front-Door Adjustment)'을 사용하여 역사적 데이터와 다음 행동간의 인과 효과를 추정합니다. 이를 위해 다양한 이질적 추천 모델들을 활용하여 매개 변수 분포를 모델링하고 다수의 선생님 모델을 하나의 학생 모델로 증류하여 인과 효과를 직접 추론합니다.

- **Performance Highlights**: CausalD 프레임워크는 편향된 학습 데이터를 사용하는 대신 서로 다른 추천 모델들의 결과를 종합하여 더 공정하고 정확한 추천을 가능하게 합니다. 이로 인해 사용자의 개인적 선호와 행동 패턴을 더 잘 반영하는 추천 성능을 기대할 수 있습니다.



### Knowledge Enhanced Multi-intent Transformer Network for Recommendation (https://arxiv.org/abs/2405.20565)
Comments:
          Accept By The Web Conf 2024 (WWW 2024) Industry Track. arXiv admin note: text overlap with arXiv:2204.08807

- **What's New**: 새로운 연구는 산업 분야의 추천 시스템에 지식 그래프(Knowledge Graph, KG)를 통합하는 접근법의 부정적인 피드백 문제를 해결하기 위해 고안되었습니다. 새로운 모델 KGTN(Knowledge Enhanced Multi-intent Transformer Network)은 사용자의 다중 의도와 지식 노이즈 문제를 효과적으로 처리합니다.

- **Technical Details**: KGTN은 두 개의 주요 모듈로 구성되어 있습니다: 1) Graph Transformer를 사용한 글로벌 의도 모델링(Global Intents Modeling with Graph Transformer)과 2) 의도 하에서의 지식 대조 노이즈 제거(Knowledge Contrastive Denoising under Intents)입니다. 첫 번째 모듈은 사용자-아이템 관계와 아이템-엔터티 상호작용에서 글로벌 신호를 통합하여 학습 가능한 사용자 의도를 포착하고, 의도 인식 사용자 및 아이템 표현을 학습합니다. 두 번째 모듈은 의도 인식 표현을 활용하여 관련 지식을 샘플링하고, 로컬-글로벌 대조 메커니즘을 사용해 노이즈 비관련 표현 학습을 강화합니다.

- **Performance Highlights**: KGTN은 세 가지 벤치마크 데이터셋에서 기존의 최첨단 모델보다 우수한 성능을 보였습니다. 또한, Alibaba의 대규모 산업 추천 플랫폼에서 수행한 온라인 A/B 테스트 결과에서도 KGTN의 실효성을 입증했습니다.



### Designing an Evaluation Framework for Large Language Models in Astronomy Research (https://arxiv.org/abs/2405.20389)
Comments:
          7 pages, 3 figures. Code available at this https URL

- **What's New**: 이번 연구에서는 천문학 연구자들이 대형 언어 모델(LLMs)을 어떻게 활용하는지 평가하기 위한 새로운 실험 설계를 제안합니다. 연구 팀은 arXiv 논문을 바탕으로 질문에 답변할 수 있는 Slack 챗봇을 개발했습니다. 이는 Retrieval-Augmented Generation (RAG) 방식을 사용해 사용자의 질문에 맞는 논문을 찾아 답변을 제공합니다. 사용자 질문과 챗봇 답변, 사용자의 피드백 및 유사도 점수를 기록하고 익명화된 데이터를 수집해 미래의 동적 평가를 가능하게 합니다.

- **Technical Details**: 이 연구에서는 천문학 논문을 바탕으로 질문에 답변하는 RAG 기반 LLM 챗봇을 구성했습니다. 기본적으로, gpt-4o를 생성 모델로 사용하고 bge-small-en-v1.5를 인코더로 사용합니다. 사용자의 질의는 bge-small-en-v1.5 인코더를 통해 인코딩 되고, 유사도를 기반으로 최상위 5개의 논문이 선택됩니다. 선택된 논문의 초록, 결론 및 메타데이터가 컨텍스트로 결합되어 LLM에게 전달되며, 이를 바탕으로 답변을 생성합니다.

- **Performance Highlights**: 현 단계에서는 데이터를 수집하지 않았지만, 이 프레임 워크를 통해 천문학 연구에 있어 LLM 도구의 실제 활용 과정을 평가할 수 있는 가능성을 제공할 예정입니다. 특히 Slack 응용 프로그램을 사용하여 사용자가 연구 질문을 제기하고, 챗봇의 답변에 피드백을 제공함으로써 보다 현실적인 사용자-모델 상호작용 데이터를 축적할 수 있게 됩니다.



### Analysis of Hopfield Model as Associative Memory (https://arxiv.org/abs/2402.04264)
Comments:
          35 pages, 23 figures, 3 codes

- **What's New**: 본 논문은 생물학적 신경 시스템에서 영감을 받아 호프필드 신경망 모델(Hopfield neural network model)을 탐구합니다. 특히 오디오 검색(Audio Retrieval)의 연상 기억(associative memory) 기능을 강조하며 호프필드 모델의 기초와 이해를 심화시키기 위해 기계 통계학(mechanical statistics)의 인사이트를 통합합니다.

- **Technical Details**: {'Neuroscience Background': '호프필드 모델은 인간 뇌의 생물학적 뉴런 간의 복잡한 연결에서 영감을 받았습니다. 생물학적 뉴런의 기본 구조(수상돌기(dendrites), 세포체(cell body, soma), 축색(axon))와 신경 충격 신호(action potential)를 설명합니다. 뉴런의 활성화 임계값(−55mV)과 전기 신호 생성 및 전송 과정을 다룹니다.', 'Artificial Neurons': '호프필드 모델의 기초를 설명하기 위해 McCulloch-Pitts 모델(MP 뉴런)을 예로 들고 있습니다. 이 모델은 입력(input), 가중치(weight), 임계값(threshold)으로 구성되며, 단일 뉴런의 AND, OR, NOT 연산을 통해 네트워크의 기능을 모의할 수 있습니다. XOR 연산을 수행하기 위해서는 다층 네트워크(MP Multilayer Network)가 필요하다는 점에서, 단일층 네트워크의 한계를 설명합니다.'}

- **Performance Highlights**: 실제 구현을 통해 네트워크는 다양한 패턴을 검색하는 연상 기억 능력을 입증합니다. 호프필드 모델을 오디오 검색에 적용하여 그 정확성과 효율성을 검증했으며, 이를 통해 생물학적 시스템의 기능성을 모방하는 신경망의 잠재력을 강조합니다.




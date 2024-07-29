### UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor (https://arxiv.org/abs/2406.06519)
Comments:
          5 pages, 3 figures

- **What's New**: 새로 등장한 UMBRELA 툴킷은 OpenAI의 GPT-4 모델을 사용해 대형 언어 모델(LLM)이 검색 시스템 평가에 필요한 관련성 판단을 자동화하는 오픈 소스 도구입니다. Microsoft Bing의 연구를 재현하고 이를 더욱 확장했습니다. 이 툴킷은 TREC 2024 RAG 트랙에서 사용될 예정입니다.

- **Technical Details**: UMBRELA는 LLM을 이용해 검색 쿼리와 결과 간의 관련성을 평가합니다. 논문에서 제시된 흐름을 재현하기 위해 'zero-shot DNA prompting' 기법을 사용했습니다. TREC 2019-2023의 Deep Learning Tracks 데이터를 사용하여 LLM의 관련성 판단 결과를 인간 평가자와 비교했습니다.

- **Performance Highlights**: UMBRELA의 LLM 기반 관련성 판단은 다단계 검색 시스템의 순위와 높은 상관관계를 보였습니다. 결과적으로 LLM은 인간 평가자와 유사한 수준의 정확성과 신뢰성을 제공함을 확인했습니다. 이러한 성과는 LLM이 더 비용 효율적이고 정확한 대안이 될 수 있음을 뒷받침합니다.



### Survey for Landing Generative AI in Social and E-commerce Recsys -- the Industry Perspectives (https://arxiv.org/abs/2406.06475)
- **What's New**: 최근 생성적 인공지능(GAI)의 등장으로 산업 추천 시스템(Recsys)이 혁신적인 변화를 겪고 있습니다. 이 논문은 사회적 및 전자상거래 플랫폼에서 GAI를 성공적으로 통합한 경험을 바탕으로, GAI와 Recsys의 통합에 대한 실질적인 통찰과 도전과제를 종합적으로 검토합니다. GAI와 Recsys 통합에 관한 실용적인 적용 사례와 해결책을 제시하며 향후 연구 방향을 제시합니다.

- **Technical Details**: 이 논문은 산업 Recsys의 복잡한 인프라, 운영 절차 및 비즈니스 제품 관점을 고려하여 GAI를 통합하는 데 필요한 실제 솔루션 프레임워크를 탐구합니다. 특히, GAI와 LLMOps(Long Language Model Operations) 기초, 맞춤형 추천을 강화를 위한 GAI 유스케이스, 그리고 Recsys 내의 Retrieval-Augmented Generation(RAG) 활용 방법 등을 포괄적으로 다룹니다. Prompt engineering, in-context learning, chain-of-thought와 같은 기법 적용 방법도 상세히 설명됩니다.

- **Performance Highlights**: 논문에서는 사용자 만족도 및 투명성과 신뢰성을 향상시키기 위해 GAI를 활용한 콘텐츠의 재목적화와 외부 지식을 통한 큐레이션이 강조됩니다. 또한, Recsys가 더욱 상호작용적이고 사용자 피드백 루프 기반으로 발전할 수 있는 방향성을 제시합니다. GAI 솔루션의 비용, 지연 시간, 전용 데이터 및 도메인 지식을 효율적으로 사용하기 위한 최적화 방향도 제시됩니다.



### Greedy SLIM: A SLIM-Based Approach For Preference Elicitation (https://arxiv.org/abs/2406.06061)
- **What's New**: 새로운 사용자의 선호도 추정을 위한 방법으로, 최신형 추천 시스템인 SLIM을 기반으로 하는 새로운 접근 방식을 제안합니다. 본 연구는 Greedy SLIM이라는 새로운 학습 기법을 활용해, 새로운 사용자에게 질문할 항목을 선정합니다. 이를 통해 특히 사용자 연구에서 뛰어난 성능을 보인다는 결론을 얻었습니다.

- **Technical Details**: 제안한 Greedy SLIM 방법은 기존 SLIM 학습 방법의 문제를 해결하기 위해 고안되었습니다. SLIM(Scalable Likelihood-based Item Model)은 협업 필터링에서 최적의 상위 N개의 추천을 위해 사용되는 기법입니다. Greedy SLIM은 항목을 하나씩 선택해 SLIM 손실을 최소화하는 방식으로 학습을 진행합니다. 이는 active learning 접근법의 일환으로, 새로운 사용자가 시스템에 입력할 항목을 최적화합니다.

- **Performance Highlights**: 오프라인 실험과 사용자 연구를 통해 Greedy SLIM의 성능을 평가했습니다. 사용자 연구에서는 특히 긍정적인 결과를 보이며, 기존의 잠재 인자 모델(LFM) 기반 방법보다 더 적합한 것으로 나타났습니다. 이는 사용자가 적은 항목만 평가해도 적정한 추천 결과를 얻을 수 있음을 시사합니다.



### Modeling User Retention through Generative Flow Networks (https://arxiv.org/abs/2406.06043)
Comments:
          KDD-ADS 2024

- **What's New**: 이번 연구는 사용자의 재방문 행동(user retention)을 최적화하는 새로운 추천 시스템 프레임워크인 GFN4Retention을 제안합니다. 기존 연구 대부분이 사용자의 즉각적인 피드백을 최대화하는 데 초점을 맞췄다면, 본 연구는 사용자의 세션 간 안정적이고 지속적인 사용을 고려하였습니다.

- **Technical Details**: GFN4Retention은 Generative Flow Networks (GFNs)를 기반으로 한 세션-wise 추천 시스템입니다. 이 프레임워크는 사용자의 세션 종료 시점의 만족도를 추정하고, 이를 기반으로 각 추천 항목에 대해 확률적 흐름(probabilistic flow)을 모델링합니다. 구체적으로, 추천 과정을 조건부 순방향 확률적 흐름으로 간주하고, 각각의 사용자 상태에 대한 흐름 추정기(flow estimator)를 활용합니다.

- **Performance Highlights**: GFN4Retention은 두 개의 공공 데이터셋과 실제 산업 플랫폼에서의 온라인 A/B 테스트를 통해 검증되었습니다. 기존의 강화 학습 기반 추천 모델들과 비교하여 우수한 성능을 보였으며, 각 구성 요소의 효과를 분석한 에이블레이션 연구에서도 높은 안정성을 나타냈습니다.



### A WT-ResNet based fault diagnosis model for the urban rail train transmission system (https://arxiv.org/abs/2406.06031)
Comments:
          12 pages,10 figures

- **What's New**: 새로운 연구는 도시 철도 시스템의 고장 진단을 위한 혁신적인 모델을 제안합니다. 이 모델은 웨이블릿 변환(Wavelet Transform)과 잔차 신경망(Residual Neural Network, ResNet)의 장점을 통합하여 진단 정확도와 강건성을 높였습니다.

- **Technical Details**: 제안된 모델은 웨이블릿 변환을 사용하여 도시 철도의 특징을 추출하고, ResNet을 통해 패턴 인식을 수행합니다. 이는 딥러닝(DL) 알고리즘 가운데 높은 성과를 보이는 ResNet을 활용한 것입니다. 또한, 기존의 CNN(Convolutional Neural Network), RNN(Recurrent Neural Network)와의 비교 및 다양한 데이터 세트에 대한 적응력을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안된 WT-ResNet 모델은 도시 철도 열차의 고장을 식별하는 데 있어 높은 효율성과 정확도를 입증했습니다. 이는 유지보수 전략을 개선하고 가동 중단 시간을 줄이는 데 기여할 수 있습니다.



### Weighted KL-Divergence for Document Ranking Model Refinemen (https://arxiv.org/abs/2406.05977)
- **What's New**: 이 논문은 트랜스포머 기반의 문서 검색 및 랭킹 모델을 위한 새로운 학습 손실 함수, 즉 대조적으로 조정된 KL 다이버전스(contrastively-weighted KL divergence, CKL)를 제안합니다. 기존의 KL 다이버전스 기반 지식 증류(loss)를 개선하여 문서 탐색 모델의 성능을 높이는 것을 목표로 합니다.

- **Technical Details**: CKL은 기존의 KL 다이버전스 대신, 긍정(positive) 문서와 부정(negative) 문서 간의 구분을 명확히 하기 위해 대조 학습(contrastive learning)과 결합된 새로운 방법론입니다. 이 방법은 문서 랭킹의 타이트한 분포 매칭(tight distribution matching)에 따른 과도한 보정(over-calibration) 문제를 해결하는 데 중점을 둡니다. MS MARCO와 BEIR 데이터셋을 사용한 평가에서 CKL의 유효성을 입증했습니다.

- **Performance Highlights**: CKL은 기존 KL 다이버전스 및 최근 BKL 접근법과 비교해 문서 검색의 성능을 효과적으로 향상시켰습니다. 실험 결과, CKL을 적용한 학생 모델은 긍정 문서와 부정 문서의 적절한 분포를 유지하면서도 랭킹의 관련성을 크게 높였습니다.



### Async Learned User Embeddings for Ads Delivery Optimization (https://arxiv.org/abs/2406.05898)
Comments:
          Accepted by workshop on Multimodal Representation and Retrieval at SIGIR 2024, Washington DC

- **What's New**: Meta 플랫폼에서 수십억 명의 사용자들을 대상으로 한 고품질 사용자 임베딩(user embeddings)을 비동기적으로 학습하는 방식을 제안합니다. 이 임베딩은 사용자 유사도 그래프(user similarity graphs)로 변환되어, 실시간 사용자 활동과 결합되어 광고 추천에 활용됩니다.

- **Technical Details**: 사용자 임베딩은 다중 모드(multimodal) 사용자 활동 신호를 기반으로 하는 시퀀스(sequence)에서 학습되며, 이는 Transformer와 유사한 구조로 처리됩니다. 다양한 사용자 활동에는 클릭한 광고, 댓글, 조회한 사진 및 동영상 등이 포함됩니다. 이러한 임베딩은 비동기적으로 업데이트되어, 수십억 명의 사용자 데이터를 처리할 수 있습니다. 또한, 사용자 유사도 그래프는 이 데이터를 기반으로 생성되며, 광고 추천에 더욱 효과적으로 활용됩니다.

- **Performance Highlights**: 제안된 모델은 오프라인 및 온라인 실험 모두에서 유의미한 성능 향상을 보였습니다. 특히, 사용자의 최신 상호작용과 피드백을 기반으로 사용자 임베딩을 지속적으로 업데이트하고 정제할 수 있는 능력을 갖추고 있습니다. 추가로, 사용자 임베딩은 대규모 모델에서 효율적인 스토리지와 컴퓨팅 절약을 위해 압축할 수 있습니다.



### Prioritizing Potential Wetland Areas via Region-to-Region Knowledge Transfer and Adaptive Propagation (https://arxiv.org/abs/2406.05578)
- **What's New**: 새로운 연구는 데이터 부족 문제를 해결하여 습지 식별 및 우선순위 설정을 돕기 위한 두 가지 전략을 제안합니다. 첫 번째 전략은 습지가 풍부한 지역에서 데이터가 희소한 지역으로 지식을 전달하는 것입니다. 두 번째 전략은 적응형 전파 메커니즘을 사용하는 공간 데이터 보강 전략입니다.

- **Technical Details**: 제안하는 접근법은 두 가지 주요 기술을 포함합니다. (1) 도메인 비디오 연습(domain disentanglement) 전략을 사용하여 풍부한 습지 지역의 지식을 데이터를 통해 전달합니다. 이 과정에서는 도메인별로 동일하게 적용 가능한 정보만 선택적으로 전이하고, 도메인 분리기로 도메인별 정보와 공유 가능한 정보를 분리합니다. (2) Graph Neural Networks (GNNs)의 적응형 전파 메커니즘을 도입하여 인접 노드들의 상호 영향을 구분하는 방식입니다. 이는 지역 내 세포 간의 유용한 정보를 차별화하여 전달합니다.

- **Performance Highlights**: 제안된 방법의 효과, 강인성 및 확장성을 입증하기 위해 엄격한 실험을 수행했습니다. 이를 통해 제안된 방법이 기존 최신 기준선을 능가하며, 모듈별 실험을 통해 각 모듈이 기존 습지 식별에 필수적임을 보여줍니다.



### I-SIRch: AI-Powered Concept Annotation Tool For Equitable Extraction And Analysis Of Safety Insights From Maternity Investigations (https://arxiv.org/abs/2406.05505)
- **What's New**: I-SIRch라는 새로운 접근 방식이 소개되었습니다. I-SIRch는 인공지능과 머신러닝 알고리즘을 활용하여 영국의 Healthcare Safety Investigation Branch(HSIB)에서 생산된 조사 보고서에서 발생한 임신 관련 사고에 대한 인간 요인(human factors)을 자동으로 식별하고 레이블을 지정합니다. 이는 생물의학적 개념에만 초점을 맞추는 기존 도구와 차별화되며, 인간 요인을 포함하여 의료 제공 시스템을 더 잘 이해하는 데 기여합니다.

- **Technical Details**: I-SIRch는 SIRch 택소노미를 사용하여 의료 조사로부터 안전 통찰을 추출합니다. 데이터 전처리 단계에서 PDF로 제공된 보고서를 텍스트로 추출하고, 특정 기준(예: 섹션, 페이지, 단락)에 따라 텍스트를 추출합니다. 수동으로 태그된 데이터를 사용해 모델을 학습시키고, 새로운 보고서를 처리하면서 인간 전문가의 계속된 주석을 통해 모델 성능을 향상시키는 'human-in-the-loop' 방식을 채택했습니다.

- **Performance Highlights**: I-SIRch는 실제 및 합성 데이터를 통해 연구되었습니다. 818개의 합성 문장과 97개의 실제 보고서에서 추출한 1960개의 문장을 테스트한 결과, 실제 보고서 문장에서 90%의 정확도로 관련 개념을 올바르게 식별했습니다. 이를 통해 다양한 인종 그룹에 따라 특정 인간 요인이 어떻게 차이 나는지 분석할 수 있었으며, 이는 사회적, 기술적 및 조직적 요인이 산모 안전과 인구 건강 결과에 미치는 복잡한 상호작용을 이해하는 데 새로운 가능성을 열어주었습니다.



### PTF-FSR: A Parameter Transmission-Free Federated Sequential Recommender System (https://arxiv.org/abs/2406.05387)
- **What's New**: 최근 사용자 데이터 프라이버시에 대한 우려가 증가함에 따라 연구자들은 사용자 데이터 프라이버시를 보호하면서 협력 학습을 구현하는 Federated Sequential Recommender Systems(FedSeqRecs)라는 접근 방식을 제안했습니다. 하지만 이는 모델 공유와 높은 통신 비용이라는 두 가지 큰 단점이 존재했죠. 이번에 제안된 연구는 그 단점을 극복하기 위해 모델 파라미터 전송 없이 협력 학습을 가능하게 하는 새로운 프레임워크, 'Parameter Transmission-Free Federated Sequential Recommendation (PTF-FSR)'을 소개합니다.

- **Technical Details**: PTF-FSR은 기존의 파라미터 교환 방식 대신 예측 결과만을 전송하여 모델 크기와 상관없이 통신 비용을 대폭 줄일 수 있습니다. 또한, 사용자 프라이버시 보호를 위해 클라이언트 측에서 사용자의 원본 항목 상호작용 시퀀스에 섭동을 추가하는 지수 기법(exponential mechanism)을 사용합니다. 이 프레임워크는 ID-based와 ID-free 파라다임을 아우르는 여러 순차적 추천 모델에서 효과적으로 적용되었습니다.

- **Performance Highlights**: 세 가지 널리 사용되는 추천 데이터셋에서 다양한 ID-based 및 ID-free 순차적 추천 모델을 테스트한 결과, PTF-FSR은 높은 성능과 일반화 능력을 보였습니다. 이는 보다 복잡하고 큰 순차적 추천 모델도 수용할 수 있는 새로운 연합 학습 구조의 가능성을 보여줍니다.



### Measuring Fairness in Large-Scale Recommendation Systems with Missing Labels (https://arxiv.org/abs/2406.05247)
- **What's New**: 이번 연구에서는 대규모 추천 시스템에서 발생하는 공정성 문제를 다룹니다. 특히, 추천된 항목에 대한 레이블(label)이 없는 상황에서의 공정성 문제를 해결하기 위한 새로운 방법을 제안합니다. 이 연구는 랜덤 트래픽(randomized traffic)을 활용하여 공정성 지표를 더 정확하게 추정하는 방법을 제시하며, 이를 증명하기 위해 TikTok의 실제 데이터를 사용한 실험 결과를 제공합니다. 또한, TikTok의 공정성 관련 데이터셋을 처음으로 공개하며, 추천 시스템의 데이터셋 수집 방법론에 대한 새로운 기준을 제시합니다.

- **Technical Details**: 이 연구는 'Ranking-based Equal Opportunity (REO)'라는 공정성 개념에 기반하여 진행됩니다. REO는 사용자-아이템(user-item) 상호작용이 완전히 관찰된 경우의 공정성 문제를 다루며, 사용자의 진정한 선호도를 알 수 없는 대규모 추천 시스템에서의 공정성 문제를 해결합니다. 연구팀은 랜덤 트래픽 데이터를 사용하여 공정성 메트릭의 오차 한계를 이론적으로 제시하며, 이를 TikTok의 실제 데이터로 검증했습니다.

- **Performance Highlights**: 실험 결과, 제안된 랜덤 트래픽 활용 방법이 기존의 간단한 방법들보다 훨씬 정확하고 효율적임을 보여줍니다. 특히, TikTok의 실제 데이터셋과 합성 데이터(synthetic data)를 통해 제안된 방법의 이론적 타당성과 실효성을 입증했습니다. 또한, 랜덤 트래픽 데이터를 사용하여 대규모 추천 시스템의 공정성 지표를 정확하게 추정하는 것이 필수적임을 확인했습니다.



### Black carbon plumes from gas flaring in North Africa identified from multi-spectral imagery with deep learning (https://arxiv.org/abs/2406.06183)
Comments:
          Published at the workshop Tackling Climate Change with Machine Learning at ICLR 2024

- **What's New**: 이 논문에서는 인공지능(deep learning) 프레임워크를 사용하여 인공위성 이미지 (satellite imagery)를 통해 북아프리카 지역에서 가스 플레어링 (gas flaring)으로 인한 블랙 카본(BC) 배출량을 직접 모니터링하는 방법을 소개합니다. 2022년 동안 모니터링된 결과, 이 지역의 BC 배출량은 약 백만 톤 탄소 동등 (tCO$_{2,	ext{eq}}$)에 이르렀습니다.

- **Technical Details**: 유럽 우주국(ESA)의 Sentinel-2 위성 데이터를 사용하여 알제리, 리비아 및 이집트에서 가스 플레어링 사이트를 분석했습니다. ConvLSTM 모델을 사용하여 위성 이미지에서 BC 플룸(plume)을 감지하고 분류했습니다. 이 모델은 두 RGB 이미지 시퀀스를 입력으로 받아 두 번째 이미지에서 BC 플룸을 세그먼트화(segmentation)하는 작업을 수행합니다. 모델 훈련은 Synthetic BC Plumes를 포함한 Sentinel-2 데이터로 이루어졌습니다. 추가적으로, LightGBM 분류기를 사용하여 거짓 양성(false positive)을 필터링했습니다.

- **Performance Highlights**: 2022년 동안 1963개의 개별 플레어(flares)를 감지했으며, 대부분은 짧은 시간 동안 활동했습니다. 가스 플레어링 사이트 중 약 10곳이 전체 BC 배출량의 25% 이상을 차지했습니다. 이는 효율적인 감지 및 완화 정책을 구현하기 위한 중요한 발걸음입니다.



### Thanking the World: Exploring Gender-Based Differences in Acknowledgment Patterns and Support Systems in Theses (https://arxiv.org/abs/2406.06006)
- **What's New**: 이 연구는 전자 논문 및 학위 논문(Electronic Theses and Dissertations, ETDs)에서 지원 시스템을 조사하여 학위 과정 중 연구자들이 어떤 형태의 지원을 받았는지 분석하였습니다. 특히 도서관 및 정보 과학(Library and Information Science) 분야를 대상으로 했습니다. RoBERTa 기반 모델을 사용하여 1252개의 ETD를 분석한 것은 이번 연구의 주요 기여입니다.

- **Technical Details**: 본 연구에서는 RoBERTa 모델을 사용하여 ETD 감사 섹션에서 다양한 지원 형태를 추출했습니다. 이를 통해 연구자들이 인식하는 주요 지원 유형은 학문적 지원(academic support), 도덕적 지원(moral support), 재정적 지원(financial support), 그리고 종교적 지원(religious support)이었음을 확인했습니다.

- **Performance Highlights**: 연구 결과, 종교적 및 재정적 지원에서는 성별 차이가 거의 없었으나, 학문적 지원과 도덕적 지원의 비율에서는 큰 성별 차이가 발견되었습니다. 특히 지도 교수들은 동성 연구자를 선호하는 경향이 있음을 보여주었습니다. 이 연구는 성별 간의 차이를 이해하고 포용적이며 지원적인 학문 환경을 조성하는 데 중요한 시사점을 제공합니다.



### Explainable AI for Mental Disorder Detection via Social Media: A survey and outlook (https://arxiv.org/abs/2406.05984)
- **What's New**: 이 논문은 데이터 과학과 인공지능(AI)을 이용한 정신 건강 관리의 최신 동향을 조사하며, 특히 온라인 소셜 미디어(OSM)를 통한 정신 질환 감지에 중점을 둡니다. 특히 Explainable AI (XAI) 모델의 중요성을 강조하며, 기존의 진단 방법과 최신 인공지능 구동 연구를 종합적으로 리뷰합니다.

- **Technical Details**: 정신 건강 진단에는 전통적으로 DSM-5와 같은 국제 표준을 기반으로 한 대면 인터뷰와 자기보고식 질문지가 사용됩니다. 최근에는 OSM 데이터를 활용한 딥러닝(Deep Learning) 기술을 통한 감지 모델도 연구되고 있는데, 이러한 모델의 설명 가능성(Explainability)을 높이는 작업이 필요합니다. 또, 이 논문은 기존 문헌에서 설명 가능성과 사회적 상호작용의 중요성을 간과하는 문제를 지적합니다.

- **Performance Highlights**: 다양한 최신 머신 러닝 및 딥러닝 모델을 검토한 결과, 크게 발전한 모델로는 DepressionNet과 EDNet 등이 있으며, 이러한 모델은 정신 질환의 조기 진단에 유망한 도구로 평가됩니다. 그러나, 블랙박스 모델의 활용은 의료 결정에서 안전성 문제를 야기할 수 있어, 설명 가능한 AI 모델로의 전환이 필요합니다.



### General Distribution Learning: A theoretical framework for Deep Learning (https://arxiv.org/abs/2406.05666)
- **What's New**: 새로운 연구 논문에서는 딥 러닝(deep learning, DL)에서 아직 해결되지 않은 여러 중요한 질문들을 다루기 위해 GD Learning이라는 새로운 이론적 학습 프레임워크를 도입합니다. 이 프레임워크는 분류(classification), 회귀(regression), 파라미터 추정(parameter estimation)을 포함한 다양한 머신 러닝 및 통계적 과제를 해결하는 데 초점을 맞추고 있습니다. 특히, GD Learning은 데이터 부족 상황에서 외부 지식을 통합하여 학습 오류를 최소화하는 것을 목표로 합니다.

- **Technical Details**: GD Learning 프레임워크는 학습 오류를 모델 및 학습 알고리즘에 의한 피팅 오류(fitting errors)와 제한된 샘플링 데이터로 인한 샘플링 오류(sampling errors)로 나눌 수 있습니다. 또한, 비정형(non-uniformity) 자코비안 행렬(Jacobian matrix)의 고유값(eigenvalues)를 이용하여 Gradient 구조 제어 알고리즘(Gradient Structure Control Algorithm)을 통해 글로벌 최적 해(global optimal solution)에 접근할 수 있음을 보여줍니다. 이러한 구조는 비콘백스(non-convex) 최적화 문제들, 예를 들어 피팅 오류 최소화에서도 사용될 수 있습니다.

- **Performance Highlights**: GD Learning은 과적합(overparameterization), 비콘백스 최적화(non-convex optimization), 편향-분산 트레이드오프(bias-variance trade-off)와 평평한 최소값(flat minima)의 메커니즘 등 딥 러닝에서의 해결되지 않은 질문들에 대해 새로운 통찰을 제공합니다. 기존의 통계적 학습 이론과는 다르게 진정한 기본 분포(underlying distribution)에 초점을 맞추면서, 성능을 향상시킬 수 있는 실질적인 방법을 제시합니다.




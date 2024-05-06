### A First Look at Selection Bias in Preference Elicitation for  Recommendation (https://arxiv.org/abs/2405.00554)
Comments: Accepted at the CONSEQUENCES'23 workshop at RecSys '23

- **What's New**: 이 연구에서는 소위 선호도 취출(preference elicitation, PE)에서의 선택 편향(selection bias)의 효과를 처음으로 살펴보았습니다. 선택 편향은 사용자와의 상호작용을 통해 수집된 데이터에서 일부 사용자 선호가 비대칭적으로 관찰되어 왜곡된 추천을 초래할 수 있습니다. 특히, 선호도 취출 단계에서의 선택 편향이 추후 아이템 추천에 미치는 부정적인 영향을 입증하고, 이를 완화하기 위한 방법을 제안합니다.

- **Technical Details**: 연구자들은 주제 기반의 선호도 취출 프로세스를 시뮬레이션하는 실험을 설정하여 선택 편향의 영향을 분석했습니다. 사용자의 주제에 대한 선호도(preference)가 수집된 후, 그 응답을 기반으로 시스템이 사용자에게 아이템 추천을 실행합니다. 이러한 접근은 기존의 추천 시스템과 달리, 사용자의 동적인 선호도 변화와 초기 의도 부족 문제를 해결하는데 유용합니다. 또한, 선택 편향을 처리하지 않을 경우 아이템 추천에서 과대표(overrepresentation) 문제가 심화될 수 있음을 발견했습니다.

- **Performance Highlights**: 시뮬레이션이 제시하는 결과에 따르면, 일찍이 선택 편향의 효과를 무시하면 나중에 아이템 추천에서 과대표 문제를 증가시킬 수 있습니다. 하지만, debiasing (편향 제거) 방법을 적용할 경우, 이러한 효과를 완화시키고 아이템 추천 성능을 상당히 향상시킬 수 있음을 확인할 수 있었습니다. 이러한 초기 결과는 향후 연구를 위한 출발점 및 동기를 제공하기 위한 것이며, 선택 편향 문제를 다루기 위한 기존 debiasing 방법들이 PE 무대에서도 적용될 수 있음을 보여줍니다.



### KVP10k : A Comprehensive Dataset for Key-Value Pair Extraction in  Business Documents (https://arxiv.org/abs/2405.00505)
Comments: accepted ICDAR2024

- **What's New**: KVP10k는 기존 데이터셋과 비교하여 가장 큰 특징은 미리 정의된 키(Key)를 사용하지 않고 문서로부터 키-값 쌍(Key-Value Pair, KVP)을 추출하는 데 초점을 맞추고 있다는 점입니다. 10,707개의 풍부하게 주석이 달린 이미지를 포함하여 광범위한 다양성과 상세한 주석이 특징입니다. 이러한 데이터셋과 벤치마크는 복잡한 비즈니스 문서로부터 정보 추출 분야의 발전을 촉진할 것입니다.

- **Technical Details**: KVP10k 데이터셋은 비정형 텍스트 데이터로부터 키-값 쌍을 효율적으로 추출해내는 기능을 향상시키기 위해 설계되었습니다. 이 데이터셋은 KIE(Key Information Extraction) 요소와 KVP 추출을 결합한 새로운 도전적인 과제를 도입하고 있으며, 17개의 구체적인 클래스로 더욱 상세하게 분류됩니다. 또한 이 데이터셋은 실제 문서로 구성되어 있어, 인공적인 데이터에 의존하는 이전 연구와 달리 실제 세계 시나리오에서의 유용성이 높습니다.

- **Performance Highlights**: KVP10k는 현재까지 출시된 데이터셋 중 가장 큰 규모를 자랑하며, 복잡한 실세계 문서를 대상으로 하는 KVP 추출 작업에 특화되어 있습니다. 이 데이터셋은 모델의 성능을 비교하고 평가할 수 있는 벤치마크 역할을 하며, 모델들이 다양한 형식과 맥락에서 사용될 수 있도록 견고하게 설계된 모델 개발에 기여할 것입니다. 첫번째 기준 결과(baseline results)는 이후 연구의 기초가 되며 KVP 추출 방법을 향상시키는 데 도움이 될 것입니다.



### Exploiting Positional Bias for Query-Agnostic Generative Content in  Search (https://arxiv.org/abs/2405.00469)
Comments: 8 pages, 4 main figures, 7 appendix pages, 2 appendix figures

- **What's New**: 최근 연구에서, 신경 랭킹 모델(Neural Ranking Models, NRMs)이 문서 검색에서 기존 어휘 기반 모델을 뛰어넘는 성능을 보여주었습니다. 이 연구는 변형자(Transformer)의 주의 메커니즘(Attention Mechanism)이 위치적 편향(Positional Bias)을 통해 검색 모델에서 이용될 수 있는 결함을 유발할 수 있다는 점을 제시하고 있으며, 이러한 결함이 하나의 쿼리나 주제를 넘어서 일반화될 수 있는 공격을 유도할 가능성을 탐구합니다.

- **Technical Details**: 연구자들은 관련 없는 텍스트(예: 홍보 내용)가 문서에 주입되어도 검색 결과에서의 위치에 부정적인 영향을 미치지 않음을 입증했습니다. 기존의 경사도 기반(Gradient-based) 공격과 달리, 이 연구는 쿼리와 무관한(Query-agnostic) 방식으로 편향을 입증하고, 주제성의 지식 없이도 비관련 내용 주입의 부정적인 영향을 감소시키기 위한 주입 위치의 제어 방법을 제시합니다. 실험은 대상 문서의 주제 맥락(Context)을 활용하여 언어 처리 모델(Language Model, LLM)에 요청하여 자동 생성된 주제적 홍보 텍스트를 사용하여 수행되었습니다.

- **Performance Highlights**: 문맥화(Contextualisation)는 비관련 텍스트의 부정적인 영향을 추가로 감소시킬 뿐만 아니라 기존의 내용 필터링 메커니즘을 우회할 가능성을 보여줍니다. 반면, 어휘 기반 모델은 이러한 내용 주입 공격에 대해 더욱 강건한 것으로 나타났습니다. 또한, 연구진은 변형자 기반 모델의 약점을 보완하는 간단하면서도 효과적인 방법을 조사하여 변형자 편향(Transformers Bias)에 대한 가설을 검증했습니다.



### Distance Sampling-based Paraphraser Leveraging ChatGPT for Text Data  Manipulation (https://arxiv.org/abs/2405.00367)
Comments: Accepted at SIGIR 2024 short paper track

- **What's New**: 최근 오디오-언어 검색 연구에 대한 관심이 증가하고 있으며, 이는 오디오 및 텍스트 모달리티 (modalities) 간의 상관관계를 설정하는 것을 목표로 합니다. 이 논문에서는 오디오-언어 검색 작업에서 데이터 불균형 문제를 해결하기 위한 새로운 접근 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 거리 샘플링 기반의 패러프레이저 (paraphraser)를 사용하여 ChatGPT를 활용합니다. 이 방법은 조작된 텍스트 데이터의 제어 가능한 분포를 생성하기 위해 거리 함수를 사용합니다. 동일한 맥락을 가진 문장 집합에 대해, 거리는 두 문장 간의 조작 정도를 계산하는 데 사용되며, Jaccard 유사도로 정의된 유사한 거리를 가진 텍스트 클러스터를 사용하여 ChatGPT의 퓨샷 프롬프팅 (few-shot prompting)이 수행됩니다. 이를 통해 텍스트 클러스터와 함께 퓨샷 프롬프팅을 적용할 때 ChatGPT는 거리를 기반으로 조작된 텍스트의 다양성을 조정할 수 있습니다.

- **Performance Highlights**: 제안된 접근 방법은 오디오-텍스트 검색에서 기존 텍스트 확장 기술보다 우수한 성능을 보이며, 검색 작업 성능이 크게 향상되었다는 것을 보여줍니다.



### Distillation Matters: Empowering Sequential Recommenders to Match the  Performance of Large Language Mod (https://arxiv.org/abs/2405.00338)
Comments: 10 pages, 2 figures

- **What's New**: 이 논문에서는 LLM(Large Language Models; 대규모 언어 모델)을 기반으로 한 추천 시스템 모델에서 실시간 추천을 가능하게 하는 새로운 지식 증류 방법인 DLLM2Rec을 제안합니다. DLLM2Rec은 LLM 기반 추천자의 높은 추론 지연 문제를 해결하고자 컨벤셔널 순차적 모델로 지식을 효과적으로 전달하는 전략을 포함합니다.

- **Technical Details**: DLLM2Rec은 다음 두 가지 주요 전략을 포함합니다. 첫 번째는 중요도 인지 순위 증류(Importance-aware ranking distillation)로, 교사 모델의 신뢰도와 학생-교사 일관성을 기반으로 인스턴스를 필터링하여 신뢰할 수 있는 학생 친화적 지식을 추출합니다. 두 번째는 협력적 임베딩 증류(Collaborative embedding distillation)로, 교사의 임베딩과 데이터에서 추출한 협력적 신호를 통합하여 지식을 전달합니다. 이 두 방법을 결합하여 LLM 기반 모델의 추천 능력을 유지하면서 실시간 추천이 가능한 경량 모델을 개발합니다.

- **Performance Highlights**: DLLM2Rec은 평균 47.97%의 성능 향상을 보여주며, 일부 경우에는 LLM 기반 추천자를 능가하는 결과를 달성했습니다. 이러한 결과는 LLM의 강력한 의미 추론(semantic reasoning) 능력과 컨벤셔널 모델의 신속한 응답 능력을 효과적으로 결합한 것입니다.



### Characterizing Information Seeking Processes with Multiple Physiological  Signals (https://arxiv.org/abs/2405.00322)
- **What's New**: 본 연구는 인지 부하(cognitive load), 정서적 흥분(affective arousal), 그리고 가치(valence)와 관련하여 정보 접근 시스템에서 사용자 행동을 생리적 신호를 통해 분석하는 새로운 접근 방식을 제시합니다. 사용자가 정보 탐색 과정에서 경험하는 감정적 반응과 행동을 이해하려는 시도로, 이는 정보 탐색 과정을 이해하는 데 중요한 실험적 데이터를 제공합니다.

- **Technical Details**: 이 연구는 정전기 활동(Electrodermal Activities), 광용적맥박계(Photoplethysmogram), 뇌전도(Electroencephalogram), 그리고 동공 반응(Pupillary Responses)을 포함한 다양한 생리적 신호를 수집하는 것을 특징으로 합니다. 연구는 정보 필요 인식(Information Need - IN), 질의 구성(Query Formulation - QF), 질의 제출(Query Submission - QS), 관련성 판단(Relevance Judgment - RJ)의 네 단계를 포함합니다. 또한, 텍스트 타이핑 또는 말하기를 통한 QS와 텍스트 또는 오디오 정보를 사용한 RJ 같은 다양한 상호작용 방식을 포함했습니다.

- **Performance Highlights**: 실험 결과는 IN 단계에서 상당히 높은 인지 부하와 함께 알림 상태(alertness)의 약간의 증가를 보였으며, QF 단계는 더 높은 주의 집중을 요구합니다. QS는 QF보다 더 높은 인지 부하를 요구하며, RJ는 QS나 IN보다 더 높은 감정적 반응을 보여 줬고, 이는 지식 격차가 해소됨에 따라 더 큰 흥미와 참여를 시사합니다. 이러한 발견은 정보 탐색 과정에서 사용자의 행동과 정서적 반응에 대한 깊은 통찰력을 제공합니다.



### Stochastic Sampling for Contrastive Views and Hard Negative Samples in  Graph-based Collaborative Filtering (https://arxiv.org/abs/2405.00287)
- **What's New**: 본 논문에서는 추천 시스템에서의 데이터 희소성과 부정적 샘플링 문제를 해결하기 위해 새로운 확률 샘플링 기법인 SCONE(Stochastic sampling for COntrastive views and hard NEgative samples)을 제안합니다. 이 방법은 동적인 확장 뷰(dynamic augmented views)와 다양한 어려운 부정적 샘플(hard negative samples)을 생성합니다.



### Global News Synchrony and Diversity During the Start of the COVID-19  Pandemic (https://arxiv.org/abs/2405.00280)
- **What's New**: 이 연구는 국제 뉴스 표현의 다양성과 동시성에 관한 심도있는 엠피리컬 분석을 제공합니다. 주요 기여는 다국적 뉴스 기사 6000만 건을 분석하여, 다양한 나라에서 어떤 글로벌 이벤트들이 보도되는지 파악하는 것입니다. 이를 위해 트랜스포머(transformer) 모델을 사용하여 다양한 언어의 뉴스 기사 사이의 유사성을 추정하고, 뉴스 이벤트를 식별하기 위한 자동화된 방법을 개발했습니다.

- **Technical Details**: 연구팀은 뉴스 기사의 유사성 네트워크를 기반으로 하는 글로벌 이벤트 식별 시스템과, 국가별 뉴스 다양성과 동시성을 측정하는 방법론을 개발하였습니다. 이 방법론은 뉴스 기사의 국제적 동시성과 국내 다양성에 영향을 미치는 요인들을 파악함으로써, 국제 릴레이션(international relations)을 이해하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 이 연구는 인터넷 사용률, 공식 언어의 수, 종교적 다양성, 경제적 불평등, 인구 크기 등이 높은 나라에서 뉴스 이벤트의 다양성이 더 크다는 것을 밝혀냈습니다. 또한, 고무적인 국제 무역량, 나토(NATO)와 브릭스(BRICS)와 같은 국제 정치경제 단체의 멤버국들 간에는 뉴스 이벤트의 보도가 더 동기화되어 있다는 결과를 도출했습니다. 이러한 결과는 국제 뉴스의 아젠다 세팅(agenda setting)과 미디어 모니터링에 통찰력을 제공합니다.



### Grounding Realizable Entities (https://arxiv.org/abs/2405.00197)
Comments: 13

- **What's New**: 생명 과학 연구에서 특성, 소질, 역할을 나타내는 온톨로지 표현이 지난 십 년 동안 미묘한 차이를 명확히 하도록 개선되었습니다. 기본 형식 온톨로지(Basic Formal Ontology, BFO)의 맥락에서 이러한 개체에 대한 널리 사용되는 특성화를 명시한 후, 이 처리법에서의 격차를 식별하고 BFO 특성화를 보완할 필요성을 동기화합니다. 보완의 방법으로, 우리는 특성과 소질 그리고 소질과 역할 사이에 존재하는 근거 관계에 대한 정의를 제안하고, 호스트-병원체 상호작용의 미묘한 측면을 나타내는 예를 들어 제안을 설명합니다.

- **Technical Details**: 이 연구는 BFO 내에서 특성, 소질, 역할을 구분하는 기존 방법론의 한계를 해결하고자 합니다. 새로운 정의는 특성(qualities), 소질(dispositions), 역할(roles) 사이의 근거 관계(grounding relations)를 클리어하게 정립함으로써 온톨로지의 정밀도를 높이는 것을 목표로 합니다. 호스트-병원체 관계는 이러한 새로운 정의가 실제 생명 과학 문제에 어떻게 적용될 수 있는지 보여주는 키 사례(case)로 사용됩니다.

- **Performance Highlights**: 제안된 온톨로지 모델은 호스트와 병원체 간의 복잡하고 섬세한 상호작용을 보다 정확하게 설명할 수 있습니다. 이는 생명 과학 연구에서 더 정확한 데이터 해석과 향상된 연구 결과로 이어질 수 있음을 시사합니다. 또한, 새로운 온톨로지 정의의 적용은 다른 생명 과학 분야의 세부적인 상호작용 분석에도 유용할 것으로 기대됩니다.



### Credentials in the Occupation Ontology (https://arxiv.org/abs/2405.00186)
Comments: 11

- **What's New**: 새로운 연구에서 미국 알라바마의 탤런트 트라이어드(Alabama Talent Triad, ATT) 프로그램과 협력하여, 교육적 자격증 및 정부 발급 라이센스 등의 자격증을 텍스트 및 의미적 수준에서 온톨로지학적으로 정의했습니다. 이 연구는 자격증과 관련된 용어들의 고급 계층 구조와 용어 간의 관계를 정립하는 것을 포함하여, 자격증 영역에 대한 체계적인 온톨로지 모델링을 처음으로 제공한다는 점에서 중요합니다.

- **Technical Details**: 연구팀은 Occupation Ontology (OccO, 직업 온톨로지), BFO(Basic Formal Ontology) 기반 온톨로지를 사용하여 자격증 및 관련 용어들을 정의했습니다. 다양한 자격증 유형과 그 인증 로직(authorization logic)이 모델링 되었으며, 자격증 관련 용어들의 상위 계층 구조와 다수의 용어들 간의 관계가 정의되었습니다.

- **Performance Highlights**: 이 연구는 자격증 데이터 및 지식 통합을 지원하기 위해 자격증 관련 중요 도메인에 대한 체계적인 온톨로지 모델링을 제공함으로써, 향후 교육 및 직업 개발 기관들이 자격증의 가치를 이해하고 활용하는 데 기여할 것으로 기대됩니다.



### Recommenadation aided Caching using Combinatorial Multi-armed Bandits (https://arxiv.org/abs/2405.00080)
- **What's New**: 이 연구에서는 사용자 선호도와 콘텐츠 인기도가 알려지지 않은 상태에서 콘텐츠 추천을 통한 캐시 최적화 문제를 다루고 있습니다. 이는 조합형 멀티 암드 밴딧(Combinatorial Multi-Armed Bandit, CMAB) 문제로 모델링되며, 콘텐츠를 캐싱하고 추천하기 위한 UCB(upper confidence bound)-기반 알고리즘을 제안하였습니다.

- **Technical Details**: 시스템은 하나의 기지국(base station, BS)을 중심으로 구성되며, 고정된 용량의 캐시를 갖추고 있습니다. N개의 콘텐츠와 U명의 사용자가 있으며, 사용자 당 최대 R개의 콘텐츠를 추천할 수 있습니다. 이러한 설정하에서 CMAB 프레임워크를 이용하여 각 슬롯에서 어떤 콘텐츠를 캐싱하고 추천할지 결정하는 문제를 해결합니다. 알고리즘은 추천에 의해 영향을 받는 사용자의 콘텐츠 요청 행동을 고려하여 캐시 히트(cache hit)를 극대화하는 방향으로 작동합니다.

- **Performance Highlights**: 제안된 알고리즘은 상태 최신 기법(state-of-the-art algorithms)과 비교하여 우수한 캐시 히트 성능을 보이며, 방대한 트래픽 요구와 사용자 만족도 향상에 기여할 수 있는 가능성을 시연하였습니다. 또한 알고리즘의 상한 regret도 제공되어 알고리즘의 효율성을 이론적으로도 뒷받침하고 있습니다.



### Automatic Creative Selection with Cross-Modal Matching (https://arxiv.org/abs/2405.00029)
- **What's New**: 이 연구에서는 앱 이미지와 검색어 간의 매칭을 개선하기 위해 사전 학습된 LXMERT 모델을 미세 조정(fine-tuning)하는 새로운 접근 방식을 제시합니다. 이는 앱 개발자들이 앱 이미지를 통해 광고하고 검색어에 입찰하는 것이 중요하기 때문에, 검색어와 높은 관련성을 가진 이미지 선택이 필요합니다.

- **Technical Details**: 기존의 방식(Transformer 모델을 사용한 검색어 및 ResNet 모델을 사용한 이미지 처리)과 비교하여 LXMERT 모델을 사용함으로써 정확성을 크게 향상시켰습니다. 이 모델은 이미지와 검색어 사이의 매칭 품질을 예측하기 위해 Image-Text 매칭 모델을 필요로 합니다.

- **Performance Highlights**: 광고주 관련 Ground Truth에 대해서는 0.96의 AUC (Area Under the Curve) 점수를 달성하여 Transformer+ResNet 기준 모델과 미세 조정된 CLIP 모델보다 각각 8%, 14% 향상되었습니다. 인간이 레이블링한 Ground Truth에 대해서는 0.95의 AUC 점수를 달성, 기준 모델과 미세 조정된 CLIP 모델을 각각 16%, 17% 초과하였습니다.




### A Systematic Investigation of Distilling Large Language Models into Cross-Encoders for Passage Re-ranking (https://arxiv.org/abs/2405.07920)
- **What's New**: 새롭게 공개된 Rank-DistiLLM 데이터셋은 큰 언어 모델에서 추출된 크로스-인코더(Cross-Encoders)의 성능을 향상시키기 위해 설계되었습니다. 이 데이터셋은 수동 라벨이 붙은 데이터에 대한 작업에서 얻은 통찰력인 하드-네거티브 샘플링(hard-negative sampling), 딥 샘플링(deep sampling), 리스트와이즈 손실 함수(listwise loss functions)를 큰 언어 모델 순위(Ranker) 증류에 전환 가능한지 조사합니다.

- **Technical Details**: Rank-DistiLLM 데이터셋은 크로스-인코더가 이미 강력한 큰 언어 모델의 효과성에 도달하면서도 효율성을 크게 향상시킬 수 있도록 훈련되게 합니다. 이를 통해 증류된 모델(distilled models)은 원래의 큰 언어 모델만큼의 효과를 내지 못하는 상황을 개선하고자 합니다.

- **Performance Highlights**: Rank-DistiLLM로 훈련된 크로스-인코더는 큰 언어 모델의 효과성에 도달할 수 있으며, 이는 직접 튜닝된 크로스-인코더보다 월등히 효율적인 작동을 가능하게 합니다. 이 데이터셋과 코드는 공개적으로 접근 가능합니다.



### Is Interpretable Machine Learning Effective at Feature Selection for Neural Learning-to-Rank? (https://arxiv.org/abs/2405.07782)
Comments:
          Published at ECIR 2024 as a long paper. 13 pages excl. reference, 20 pages incl. reference

- **What's New**: 본 연구는 신경 순위 매기기 모델(neural ranking models)에 집중하고, 검색 및 추천 시스템에서 널리 쓰이는 특징 선택(feature selection) 방법들을 조사하며, 이 분야에서의 해석 가능한 기계 학습(interpretable machine learning, ML) 기법들을 적용해 본다. 또한 저자들은 자체적인 수정을 추가하는 방법을 도입하여 순위 결정에 중요한 입력 특징들을 선별해낸다.

- **Technical Details**: 연구팀은 여섯 가지 해석 가능한 ML 방법을 검토하고, 그 중 TabNet (지역 선택 방법)과 G-L2X (저자들이 수정한 전역 선택 방법)을 사용하여 필수적이지 않은 특징들을 크게 줄이면서도 순위 매기기 성능을 유지할 수 있는지 평가한다.

- **Performance Highlights**: 실험 결과, TabNet은 10개 미만의 특징으로 최적의 순위 성능을 달성할 수 있었으며, G-L2X와 같은 전역 방법은 조금 더 많은 특징을 필요로 하지만, 효율성 개선 측면에서 더 높은 가능성을 보여준다. 이는 해석 가능한 ML과 순위 매기기 학습(Learning-to-Rank, LTR) 분야의 통합을 촉진할 가능성이 있음을 시사한다.



### Synthetic Test Collections for Retrieval Evaluation (https://arxiv.org/abs/2405.07767)
Comments:
          SIGIR 2024

- **What's New**: 이 논문은 정보 검색(IR) 시스템의 평가를 위한 세트 자료을 개발하는 데 있어서 대형 언어 모델(LLMs)의 활용 가능성을 탐구합니다. 이전 연구에서는 LLM을 이용하여 합성 쿼리나 문서를 생성하여 학습 데이터를 확장하고 순위 매기기 모델의 성능을 향상시키는 데 초점을 맞췄지만, 완전히 합성될 테스트 세트 자료를 구축하는 것은 상대적으로 미개척 분야입니다.

- **Technical Details**: 본 논문은 LLM을 사용하여 합성 쿼리는 물론 합성 판단(judgments)도 생성함으로써 완전히 합성된 테스트 컬렉션을 구축할 수 있는지를 철저히 조사합니다. 또한 합성 테스트 컬렉션의 신뢰성 구축 가능성과 LLM 기반 모델에 대한 편향 위험을 분석합니다.

- **Performance Highlights**: 실험 결과에 따르면 LLM을 이용하여 만들어진 합성 테스트 컬렉션은 정보 검색 평가에 신뢰성 있게 사용될 수 있는 것으로 나타났습니다. 이는 LLM을 채택한 검색 시스템의 평가 지표로써 사용되어질 수 있는 가능성을 뒷받침합니다.



### DynLLM: When Large Language Models Meet Dynamic Graph Recommendation (https://arxiv.org/abs/2405.07580)
Comments:
          11 pages, 5 figures

- **What's New**: 이 연구는 주로 LLM (Large Language Models)을 사용하여 동적 그래프 추천 작업을 처리하는 새로운 프레임워크 DynLLM을 제안한다. DynLLM은 사용자의 텍스트 기반 프로필을 생성하고 이를 시간 그래프 임베딩과 결합하여 동적 추천을 실현한다. 이는 LLM이 동적 데이터를 예측하는 데 어떻게 적용될 수 있는지에 대한 새로운 관점을 제공한다.

- **Technical Details**: DynLLM 프레임워크는 구매 항목 제목의 텍스트 기능에서 사용자 프로필을 생성하는 데 LLM을 활용한다. 이 프로필은 군중 세그먼트(crowd segments), 개인적인 관심사(personal interests), 선호하는 카테고리(preferred categories), 선호하는 브랜드(favored brands) 등 여러 면을 포함한다. 프로필 임베딩을 정제하기 위해, 본 연구는 'distilled attention mechanism'을 도입하여 노이즈를 최소화하고 임베딩의 표현력을 강화한다. 이 메커니즘은 다중 헤드 주의(multi-head attention)를 사용하여 시간 그래프 임베딩과 결합된 프로필 임베딩을 통합한다.

- **Performance Highlights**: DynLLM은 다양한 최신 기준 방법론들을 크게 능가하는 결과를 보여주었다. 두 개의 실제 전자상거래 데이터셋을 사용한 광범위한 실험을 통해, 시간 그래프나 LLM 기반 다른 모델들에 비해 월등한 성능 개선을 이루었다. 이는 DynLLM이 동적 추천 시나리오에 효과적으로 적용될 수 있음을 입증한다.



### MS MARCO Web Search: a Large-scale Information-rich Web Dataset with Millions of Real Click Labels (https://arxiv.org/abs/2405.07526)
Comments:
          10 pages, 6 figures, for associated dataset, see this http URL

- **What's New**: MS MARCO Web Search 데이터셋은 웹 문서 및 쿼리 분포를 현실 세계에 가깝게 모방하면서 수백만 개의 실제 클릭된 쿼리(document)-문서(query) 레이블을 포함하는 최초의 대규모 정보 풍부 웹 데이터셋입니다. 이 데이터셋은 다양한 종류의 하류 작업을 위한 풍부한 정보를 제공하고, 일반적인 종단 간(end-to-end) 신경 색인 모델(neural indexer models), 일반적인 임베딩 모델(embedding models), 대규모 언어 모델을 사용한 차세대 정보 접근 시스템 등 다양한 연구 분야를 장려합니다.

- **Technical Details**: MS MARCO Web Search는 ClueWeb22를 기반으로 하며, 약 100억 개의 고품질 웹 페이지를 포함합니다. 이 데이터셋은 문서의 시각적 표현, 원시 HTML 구조, 깨끗한 텍스트, 의미 주석, 언어 및 주제 태그 등 웹 페이지에서 풍부한 정보를 포함합니다. 또한 93개 언어의 1000만 개 고유 쿼리와 수백만 개의 관련 레이블이 지정된 쿼리-문서 쌍을 포함하고 있습니다. 이는 신경 색인 모델, 임베딩 모델 및 대량 데이터를 효율적으로 처리할 수 있는 ANN(근사 최근접 이웃 탐색) 시스템과 협력해야 할 필요성을 강조하고 있습니다.

- **Performance Highlights**: MS MARCO Web Search 데이터셋은 여러 최첨단 임베딩 모델, 검색 알고리즘 및 정보 검색 시스템을 구현하며, 이를 기존 데이터셋에서 개발된 시스템과 비교 분석합니다. 실험 결과에 따르면 임베딩 모델, 검색 알고리즘 및 정보 검색 시스템은 모두 웹 정보 검색에서 중요한 구성 요소임이 드러났으며, 단순히 한 구성 요소만 개선하는 것은 종단 간 검색 결과 품질 및 시스템 성능에 부정적인 영향을 미칠 수 있습니다. 이 벤치마크는 데이터 중심 기술, 임베딩 모델, 검색 알고리즘 및 검색 시스템에서의 혁신을 촉진하여 종단 간 성능을 극대화할 수 있도록 설계되었습니다.



### PromptLink: Leveraging Large Language Models for Cross-Source Biomedical Concept Linking (https://arxiv.org/abs/2405.07500)
- **What's New**: 이 연구에서는 생물의학 개념을 서로 다른 데이터 소스간에 연결(링크)하는 새로운 프레임워크인 PromptLink를 제안합니다. 기존의 생물의학 개념 링크 방법과 달리, PromptLink는 대규모 언어 모델(LLM: Large Language Model)을 사용하여 신뢰성 있는 예측을 생성합니다. 이 방법은 특히 비용과 문맥 길이에 대한 제약을 최소화하며, 신뢰할 수 없는 예측을 거부하는 능력(NIL prediction)을 갖추고 있습니다.

- **Technical Details**: PromptLink는 먼저 생물의학 전문 예비 훈련된 언어 모델(pre-trained language model)인 SAPBERT를 사용하여 후보 개념을 생성하고, 이 개념들이 LLM의 문맥 창에 맞도록 합니다. 그 후에 GPT-4 모델을 이용하여 두 단계의 프롬프트를 통해 개념을 링크합니다. 첫 번째 단계 프롬프트는 생물의학 지식을 이끌어내어 링크 작업에 활용하고, 두 번째 단계 프롬프트는 LLM이 자신의 예측을 다시 검토하게 함으로써 신뢰도를 높입니다.

- **Performance Highlights**: 실험 결과, PromptLink는 EHR 데이터셋과 외부 생물의학 지식 그래프(KG: Knowledge Graph)를 이용한 생물의학 개념 링크 작업에서 기존 방법들을 5% 이상 뛰어넘는 성능을 보여주었습니다. PromptLink는 교육 과정 없이도 제로샷(zero-shot) 프레임워크로 작동하며, 다양한 유형의 데이터 소스에서 개념 링크에 사용될 수 있습니다.



### Learnable Tokenizer for LLM-based Generative Recommendation (https://arxiv.org/abs/2405.07314)
- **What's New**: 이 연구에서는 일반적인 추천 시스템에서 Large Language Models(LLMs)를 사용하여 생성 추천을 구현하는 새로운 방법을 소개합니다. LETTER(LEarnable Tokenizer for generaTivE Recommendation)라는 새로운 토크나이저는 아이템을 효과적으로 언어 공간으로 변환하는 것을 목표로 하며, 기존에 사용되던 ID 식별자, 텍스트 식별자, 코드북 기반 식별자의 한계를 극복하고자 합니다.

- **Technical Details**: LETTER는 계층적 의미(hierarchical semantics)를 통합하고 협업 신호(collaborative signals) 및 코드 할당 다양성(code assignment diversity)을 고려하여 설계되었습니다. 이 모델은 잔류 양자화 VAE(Residual Quantized VAE)를 사용하여 의미를 규제하고, 대조 정렬 손실(contrastive alignment loss)과 다양성 손실(diversity loss)을 적용하여 코드 할당 편향을 완화합니다. 또한, 랭킹 지도 생성 손실(ranking-guided generation loss)을 통해 랭킹 능력을 향상시키는 두 가지 생성 추천 모델에 LETTER를 적용하였습니다.

- **Performance Highlights**: LETTER는 세 가지 데이터셋에서 광범위한 실험을 통해 아이템 토크나이제이션(item tokenization)에서 우수한 성능을 보였으며, 생성 추천 분야에서 최신 기술(state-of-the-art)을 뛰어넘는 결과를 보여주었습니다.



### Identifying Key Terms in Prompts for Relevance Evaluation with GPT Models (https://arxiv.org/abs/2405.06931)
Comments:
          19pages, 2 figures

- **What's New**: 최근의 연구에서는 큰 언어 모델(Large Language Models, LLMs)을 사용하여 질의(query)와 지문(passage)의 관련성 평가에 대한 연구가 활발히 이루어지고 있다. 특히, GPT-4와 같은 모델을 사용하는 연구가 주목받고 있으며, 이러한 모델들은 자연어 처리 작업에서 뛰어난 성능을 보여주었다. 본 논문은 GPT 모델을 이용하여 관련성 평가를 위한 프롬프트(prompt)에서 특정 용어가 성능에 미치는 영향을 조사하는 것을 목표로 한다. 이를 통해 'answer'라는 용어가 'relevant'보다 더 효과적임을 발견하였다.

- **Technical Details**: 이 연구에서는 두 가지 유형의 프롬프트를 사용하였으며, 하나는 이전 연구에서 사용된 프롬프트이고, 다른 하나는 LLM에 의해 자동 생성된 프롬프트이다. 실험은 few-shot 및 zero-shot 설정에서 수행되었으며, 프롬프트의 성능을 비교 분석하였다. 그 결과, 'answer' 용어를 사용한 프롬프트가 'relevant'보다 관련성 평가에서 더 정확하고 효율적인 것으로 나타났다.

- **Performance Highlights**: 'answer'을 포함한 프롬프트는 'relevant'을 포함한 프롬프트보다 효과적인 성능을 보여 관련성 평가에 있어 더욱 직접적이고 정확한 접근 방식이 우수함을 시사한다. 또한, few-shot 예제를 포함하는 것이 관련성의 범위를 보다 명확하게 정의하는 데 도움이 되며, 이는 평가의 정확도를 높이는 데 기여한다. 이러한 발견은 향후 LLMs를 활용한 관련성 평가 작업에서 프롬프트 디자인의 중요성을 강조한다.



### Multimodal Pretraining and Generation for Recommendation: A Tutoria (https://arxiv.org/abs/2405.06927)
Comments:
          Published in WWW 2024 Tutorial. Find the tutorial materials at this https URL

- **What's New**: 이 튜토리얼은 추천 시스템((recommender systems)) 분야에서 멀티모달((multimodal)) 프리트레이닝((pretraining)) 및 생성 기술((generation techniques))의 최신 발전을 탐구합니다. 텍스트, 이미지, 오디오, 비디오 등 다양한 모달리티에서 아이템 콘텐츠의 본질을 이해하는 데 기존 ID 중심 접근 방식의 한계를 극복하는 데 초점을 맞추고 있습니다. 이는 뉴스, 음악, 짧은 비디오 플랫폼과 같은 멀티미디어 서비스에 있어 특히 중요합니다.

- **Technical Details**: 이 튜토리얼은 두 가지 주요 연구 주제로 구성되어 있습니다. 첫 번째는 멀티모달 프리트레이닝 기술((multimodal pretraining techniques)) 이며, 이는 아이템 표현의 사용자 순차적 행동 패턴을 포착하기 위한 시퀀스 프리트레이닝((Sequence pretraining)), 텍스트 기반 프리트레이닝((Text-based pretraining)), 오디오 기반 프리트레이닝((Audio-based pretraining)), 그리고 멀티모달 프리트레이닝을 포함합니다. 두 번째 주제는 멀티모달 생성 기술((multimodal generation techniques)), 여기에는 텍스트, 이미지 생성과 맞춤형 생성이 포함됩니다. 이러한 기술들은 추천 시스템에서 맞춤형 컨텐츠 제공을 위해 중요합니다.

- **Performance Highlights**: 이 튜토리얼은 멀티모달 학습((multimodal learning))과 추천 시스템 영역 간 시너지를 탐구하고 학계와 산업계 연구자들에게 깊이 있는 통찰을 제공합니다. 사용된 멀티모달 프리트레이닝 모델들은 기존 추천 방식보다 향상된 성능을 보이며, 실제 산업 응용 예를 통해 그 유효성이 입증되었습니다.



### Event GDR: Event-Centric Generative Document Retrieva (https://arxiv.org/abs/2405.06886)
Comments:
          Accepted to WWW 2024

- **What's New**: 이 글에서는 '이벤트 중심 생성적 문서 검색 모델(Event GDR)'을 제안합니다. 이 모델은 문서의 내용 상관관계 및 식별자 구성의 명시적 의미 구조 두 가지 주요 도전 과제를 해결하기 위해 풍부한 이벤트 관계와 잘 정의된 분류 체계를 활용합니다. 특히, 복수의 에이전트를 활용하는 교환 후 반영(Exchange-then-Reflection) 방식을 통해 이벤트 지식을 추출하고, 문서 표현 및 식별자 구축에 이를 응용합니다.

- **Technical Details**: 이 모델은 정보 검색(Information Retrieval, IR)에서 생성적 문서 검색(Generative Document Retrieval, GDR) 기법을 사용합니다. 주요 기술적 방법으로는, 다중 에이전트(multi-agents)를 이용한 교환 후 반영 방식을 통해 이벤트 지식을 추출하고, 이벤트 및 관계를 활용하여 문서를 모델링함으로써 문서의 포괄성 및 내용 상관관계를 보장합니다. 또한, 이벤트를 잘 정의된 이벤트 분류 체계에 매핑하여 명시적 의미 구조의 식별자를 구성합니다.

- **Performance Highlights**: 이벤트 중심 생성적 문서 검색 모델(Event GDR)은 기존 베이스라인에 비해 두 데이터셋(Natural Questions 및 DuReader)에서 유의미한 성능 개선을 보였습니다. 이는 이벤트 관계와 태그 시스템이 모델 성능에 긍정적인 영향을 미치는 것으로 평가됩니다. 특히, 이 모델은 식별자 생성 및 문서 표현의 정확성을 향상시켜 효과적인 문서 검색 가능성을 보여줍니다.



### Almanac Copilot: Towards Autonomous Electronic Health Record Navigation (https://arxiv.org/abs/2405.07896)
- **What's New**: 이 연구에서는 임상 의사의 전자 의료 기록(EMR) 작업 부담을 완화하기 위해 'Almanac Copilot'이라는 자율 에이전트를 소개합니다. Almanac Copilot은 정보 검색 및 주문 배치와 같은 EMR 특정 작업을 돕는 능력을 가지고 있습니다.

- **Technical Details**: Almanac Copilot은 EHR-QA라는 실제 환자 데이터를 기반으로 한 300개의 일반 EHR 쿼리에 대한 합성 평가 데이터셋에서 74%의 성공적인 작업 완료율을 달성하였습니다. 이는 총 221개의 작업에서 평균 점수 2.45/3 (95% 신뢰 구간: 2.34-2.56)을 의미합니다.

- **Performance Highlights**: Almanac Copilot의 성과는 임상 의사들이 EMR 시스템에 의해 부과된 인지 부담을 줄이는 데 큰 잠재력을 보여주며, 불필요한 문서 작업과 경고 피로(alert fatigue)를 줄이는데 도움을 줄 수 있습니다.



### A Decentralized and Self-Adaptive Approach for Monitoring Volatile Edge Environments (https://arxiv.org/abs/2405.07806)
Comments:
          Submitted to ACM Transactions on Autonomous and Adaptive Systems

- **What's New**: 새롭게 제안된 'DEMon'은 분산되고 자가 적응 가능한 에지 컴퓨팅(edge computing) 모니터링 시스템입니다. 기존의 중앙집중식 모니터링 시스템이 가지는 높은 지연 시간, 단일 실패 지점, 그리고 변화가 많은 에지 환경에서 신속하고 신뢰성 있는 데이터 제공의 어려움을 해결하기 위해 고안되었습니다.

- **Technical Details**: DEMon은 'stochastic gossip communication protocol'을 핵심으로 사용하며, 정보 전파, 통신, 및 검색을 위한 효율적인 프로토콜을 개발했습니다. 이를 통해 단일 실패 지점을 방지하고 빠르며 신뢰할 수 있는 데이터 접근을 보장합니다. 또한, 모니터링 품질(Quality of Service, QoS)과 자원 소비(resource consumption) 사이의 균형을 맞추기 위해 모니터링 파라미터의 자가 적응 관리(self-adaptive management)가 가능한 분산 제어(decentralized control)를 사용합니다.

- **Performance Highlights**: 라이트웨이트하고 포터블한 컨테이너 기반(container-based) 시스템으로 구현된 DEMon은 실험을 통해 평가되었고, 사용 사례 연구(use case study)를 통해 그 실용성이 입증되었습니다. 결과적으로 DEMon은 에지 모니터링(edge monitoring)의 도전과제를 해결하며, 모니터링 정보를 효과적으로 전파하고 검색하는 것을 보여주었습니다.



### Decoding Geometric Properties in Non-Random Data from First Information-Theoretic Principles (https://arxiv.org/abs/2405.07803)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2303.16045. substantial text overlap with arXiv:2303.16045

- **What's New**: 새로운 단변수 신호 역합성(Univariate signal deconvolution) 방법이 정보 이론(Information Theory), 측정 이론(Measure Theory), 그리고 이론적 컴퓨터 과학(Theoretical Computer Science)의 원칙을 바탕으로 소개되었습니다. 이 방법은 특히 알려지지 않은 발생원으로부터 메시지를 해독하는 데 적용 가능한 제로 지식 원방향 통신 채널(Zero-knowledge one-way communication channels)의 코딩 이론(Coding Theory)에서 광범위한 응용이 있습니다. 더 나아가 이 도구는 모든 임의의 확률 측정에서 완전히 독립적으로 범용 모델을 구축할 수 있는 능력을 갖춘 인공 일반 지능(Artificial General Intelligence) 접근법의 원칙에서 파생됩니다.

- **Technical Details**: 이론적으로 볼 때, 이 방법은 인코딩-디코딩 방식, 계산 모델, 프로그래밍 언어, 형식 이론, 알고리즘 복잡성(approximation to algorithmic complexity)의 접근법, 그리고 임의로 선택된(계산 가능한) 사건의 확률 측정(Semi-computable method)에 대해 알지 못하는 상태(agnotic)를 유지합니다. 본 연구는 모든 임의의 확률 분포에 독립적인 일반적 목적의 모델을 구축하는 범용(Universal) 방법을 사용하고 있습니다.

- **Performance Highlights**: 이 방법은 신호 처리(Signal Processing), 인과 역합성(Causal deconvolution), 위상 및 기하학적 특성 인코딩(Topological and geometric properties encoding), 암호학(Cryptography), 그리고 생물 및 기술적 서명 탐지(Bio- and technosignature detection) 등 다양한 응용 분야에서 최적화(Optimized)되고 범용적(Universal)인 해독 방식으로 의미가 있습니다.



### SoccerNet-Echoes: A Soccer Game Audio Commentary Datas (https://arxiv.org/abs/2405.07354)
- **What's New**: 본 논문에서는 축구 경기 중 방송된 오디오 코멘터리의 자동 생성된 텍스트 변환을 제공하는 SoccerNet-Echoes 데이터셋을 소개합니다. 이는 Automatic Speech Recognition (ASR) 기술을 스포츠 분석에 적용한 것으로, 축구 경기의 사건들에 대한 깊은 통찰을 제공하고 자동 하이라이트 생성과 같은 다양한 하위 애플리케이션의 가능성을 열어줍니다.

- **Technical Details**: 이 데이터셋은 기존의 SoccerNet 데이터셋을 확장한 것으로, Whisper 모델을 사용하여 오디오 코멘터리를 텍스트로 변환하고 Google Translate로 번역하여 풍부한 텍스트 정보로 비디오 내용을 향상시킵니다. ASR을 통한 텍스트 데이터의 통합과 데이터셋 큐레이션에 관련된 방법을 상세히 설명합니다.

- **Performance Highlights**: SoccerNet-Echoes는 시각적, 청각적 내용과 함께 텍스트 데이터를 포함함으로써 축구 게임의 역동성을 포착하는 전문 알고리즘 개발을 위한 포괄적인 자원으로서 기능합니다. 또한, enriched dataset (풍부한 데이터셋)은 스포츠 분석 분야의 연구 및 개발 범위를 확대하며 향상된 액션 스포팅, 자동 캡션 생성, 게임 요약과 같은 다양한 애플리케이션을 지원합니다.



### Instruction-Guided Bullet Point Summarization of Long Financial Earnings Call Transcripts (https://arxiv.org/abs/2405.06669)
Comments:
          Accepted in SIGIR 2024

- **What's New**: FLAN-FinBPS 모델은 재무 문서인 수익 호출 전사(Earnings Call Transcripts, ECTs)를 요약하기 위해 최근에 개발된 방법입니다. 이 모델은 자동화된 점수 및 요약 방법을 통해 기존 방법들보다 더 나은 성능을 나타냈습니다. 자동 요약 기술은 주로 짧은 뉴스 기사나 구조화된 문서를 대상으로 발전했지만, 이 연구에서는 긴 ECT를 요약하는 새로운 분야에 초점을 맞춥니다.

- **Technical Details**: FLAN-FinBPS는 비지도 학습(Unsupervised) 질문 기반 추출 모듈과 매개 변수 효율적인 지시 튜닝(Directed Tuning) 지시형(Abstracting Module) 추상 모듈을 사용하여 ECTSum 데이터 세트에서 불릿 포인트 요약을 생성합니다. 이 모델은 ECT의 중요한 사실들을 효과적으로 포착할 수 있는 요약을 생성하며, 단계별 접근 방식을 통해 요약의 정확성을 높입니다.

- **Performance Highlights**: 이 모델은 평균 ROUGE 점수에서 14.88% 향상을 보여주었고, BERTScore에서도 16.36% 상승하였습니다. 또한, 정밀도 면에서 2.51% 향상되었으며, 요약의 사실 일관성에서 2.70% 개선되었습니다. 이러한 결과는 FLAN-FinBPS가 기존의 최고 성능 모델들을 효과적으로 능가함을 보여 줍니다.



### Enhancing Language Models for Financial Relation Extraction with Named Entities and Part-of-Speech (https://arxiv.org/abs/2405.06665)
Comments:
          Accepted to ICLR 2024 Tiny Paper Track

- **What's New**: 이 연구에서는 금융 텍스트(Named Entity Recognition, NER)의 명명된 엔터티 인식 및 품사 태깅(Part Of Speech, POS)을 통합하여 기존의 잘 훈련된 언어 모델의 성능을 향상시키는 새로운 접근 방식을 제안합니다. 이를 통해 금융 관련 텍스트에서 엔터티 간의 관계를 보다 정확하게 추출하는 것을 목표로 합니다.

- **Technical Details**: 본 논문은 RoBERTa 언어 모델을 기반으로 하여, 금융 문서의 NER 및 POS 정보를 추가로 통합하는 다양한 전략을 제안합니다. 금융 텍스트에서 NER로 식별된 엔티티 토큰을 이용하여 원본 텍스트를 대체하고, 해당 POS 토큰과 연결(concatenate)하는 방식으로 모델을 구성하였습니다. 이러한 구조는 금융 관계 추출(Financial Relation Extraction, FinRE) 작업에 특히 유용합니다.

- **Performance Highlights**: REFinD 데이터셋을 사용한 실험 결과, 제안된 모델은 기존 베이스라인 모델들과 비교하여 상당한 성능 향상을 보였습니다. Micro-F1과 Macro-F1 점수에서 각각 0.1144 및 0.2134의 절대적인 개선을 달성하였으며, 이는 NER 및 POS 정보의 통합이 금융 관계 추출 작업에 유의미한 영향을 미친다는 것을 확인시켜 줍니다.




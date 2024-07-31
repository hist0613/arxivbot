### A Conceptual Framework for Conversational Search and Recommendation:  Conceptualizing Agent-Human Interactions During the Conversational Search  Process (https://arxiv.org/abs/2404.08630)
- **What's New**: 이 논문은 대화형 검색 업무의 개념적 틀을 개발하는 것을 목표로 하며, 사용자와 에이전트의 행동과 의도를 설명하고 이러한 행동들이 사용자가 검색 공간을 탐색하고 정보 필요를 해결하는 데 어떻게 도움이 되는지를 설명합니다. 대화의 주요 결정 지점에서 에이전트가 대화 검색 과정을 성공적으로 이끌기 위해 어떻게 결정해야 하는지에 대해 논의합니다.

- **Technical Details**: 이 논문은 대화형 검색 에이전트(Conversational Search Agent)가 사용자의 정보 요구를 충족시키기 위해 수행해야 하는 다양한 행동(Action)과 의도(Intent)를 구조화합니다. 또한, 에이전트가 대화를 진행하면서 취해야 하는 핵심 행동을 정리하고, 이러한 행동들이 대화의 상태를 어떻게 관리하고 사용자의 의도를 어떻게 파악하여 적절한 결과를 제공할지에 대한 의사 결정 과정을 설명합니다. 에이전트의 기능으로는 정보 필터링, 응답 결정, 대화 관리, 일반 지식 활용, 사용자와의 상호 작용 및 설명 등이 포함됩니다.

- **Performance Highlights**: 논문에서 제시한 대화형 검색 프로세스의 개념적 틀은 기존 연구들을 통합하고, 에이전트와 사용자 간의 효율적인 정보 교환을 가능하게 함으로써, 사용자의 정보 필요를 충족시키는데 중요한 역할을 합니다. 또한, 사용자가 검색 결과를 통해 필요한 정보를 얻고, 대화를 통해 추가적인 정보 요구 사항을 명확히 할 수 있도록 지원합니다. 이러한 개념적 틀은 향후 대화형 검색 에이전트의 연구, 개발 및 평가에 중요한 기준점을 제공합니다.



### Accessibility in Information Retrieva (https://arxiv.org/abs/2404.08628)
- **What's New**: 이 논문은 교통 계획 분야에서 사용되는 접근성(accessibility) 개념을 정보 검색(Information Retrieval, IR) 분야에 도입하였습니다. 정보 검색이라는 비물리적 공간에서 문서 접근성을 측정하는 새로운 방법을 제안하면서, 접근성이 정보 검색 시스템의 설계, 조정 및 관리에 어떻게 활용될 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 저자들은 운송 수단과 유사하게, 정보 접근 시스템(Information Access System)을 통해 사용자가 IR 시스템에서 문서를 검색하는 방식을 비교하여 설명합니다. 또한, 두 가지 IR 기반 접근성 측정 방식을 제안합니다. 첫 번째는 누적 기회 측정법(Cumulative Opportunity Measures), 두 번째는 중력 기반 측정법(Gravity Based Measures)으로, 이러한 방법들은 사용자가 일정 거리나 시간 내에 도달할 수 있는 기회의 수를 계산합니다.

- **Performance Highlights**: 이 연구는 정보 검색 시스템의 문서 접근성을 고려하여, 시스템의 평가 방법 및 최적화에 새로운 관점을 제공합니다. 특히 하이퍼링크가 있는 문서 컬렉션에서 접근성이 높은 페이지와 접근성이 없는 페이지를 비교 분석함으로써, 정보 검색 시스템의 설계와 관리에 유용한 지침을 제시합니다.



### Generalized Contrastive Learning for Multi-Modal Retrieval and Ranking (https://arxiv.org/abs/2404.08535)
- **What's New**: 이 논문에서는 대규모 데이터셋을 활용하여 새로운 다중 모달 검색 및 순위 매기기 프레임워크인 일반화된 대조학습(Generalized Contrastive Learning, GCL)을 제안합니다. GCL은 바이너리 관련성 점수(binary relevance scores)를 초월하여 미세한 순위(fine-grained rankings)에서 학습하는 것을 목표로 하며, 이는 기존의 대조학습 방법론에서 다루지 않은 새로운 접근 방식입니다.

- **Technical Details**: GCL 프레임워크는 쿼리와 문서 쌍 사이의 무게를 고려하여 트리플렛 입력 단위(triplet input unit)를 생성합니다. 이 무게는 점수-무게 함수(score-to-weight function)를 통해 실제 관련성 점수에서 변환됩니다. 또한, 가중 크로스 엔트로피 손실(weighted cross entropy loss)을 사용하여 미세한 순위 정보를 통합합니다. GCL은 다중 필드(multi-field) 정보를 통합하여 학습하며, 이는 제목과 제품 이미지 같은 요소들을 평균 임베딩으로 병합하는 방식입니다.

- **Performance Highlights**: GCL은 기존 대조학습 방법(CLIP) 대비하여 도메인 내 평가(in-domain evaluation)에서 NDCG@10에서 94.5% 향상을 보였으며, 콜드 스타트(cold-start) 평가에서는 26.3~48.8%의 향상을 보였습니다. 이러한 결과는 GCL이 기존 방법론을 크게 능가하며 다양한 정보 검색 작업에 활용될 가능성이 높음을 시사합니다.



### Large-Scale Multi-Domain Recommendation: an Automatic Domain Feature  Extraction and Personalized Integration Framework (https://arxiv.org/abs/2404.08361)
Comments: 8 pages

- **What's New**: 이 논문에서는 다중 도메인(feed recommendation scenarios 또는 domains) 추천 시스템을 위한 자동 도메인 특성 추출 및 개인화 통합(Domain Feature Extraction and Personalized Integration, DFEI) 프레임워크를 제안합니다. 이 프레임워크는 각 사용자의 행동을 해당 도메인 내의 모든 사용자 행동의 집합으로 자동 변환하여 도메인 특성으로 활용합니다. 기존의 수동 특성 공학 방법과 달리, 추출된 도메인 특성은 고차원 표현(higher-order representations)이며 타겟 라벨(target label)과 직접적인 관련이 있어 추천 성능을 향상시키는 데 도움이 됩니다.

- **Technical Details**: DFEI 프레임워크는 각 도메인의 사용자 행동을 집약하여 도메인 특성으로 전환하는 기능을 자동화하고, 이를 통해 수작업 없이 도메인 간 차이를 명확히 할 수 있습니다. 또한, 다른 도메인의 도메인 특성을 각 사용자에게 개인화하여 통합함으로써, 새로운 트레이닝 모드(training mode)의 혁신을 통해 추천 성능을 크게 향상시킬 수 있습니다. 실험 결과, 이 프레임워크는 20개 이상의 도메인을 포함하는 공개 및 산업 데이터셋에서 기존의 최상의 기준(SOTA, State of The Art baselines)보다 상당히 더 나은 성능을 보여주었습니다.

- **Performance Highlights**: DFEI 프레임워크는 다양한 공개 및 산업 데이터셋에서 테스트되었으며, 여러 도메인에서의 추천 성능이 기존 방법들과 비교하여 유의미하게 향상되었음을 보여줍니다. 특히, 이 프레임워크는 도메인 간의 정보 이용과 특성 추출을 자동화함으로써 더 정확한 사용자 맞춤 추천을 가능하게 했습니다. 추가로, 이 프레임워크의 소스 코드도 공개되어 다른 연구자들이 활용하고 발전시킬 수 있는 기회를 제공합니다.



### Collaborative-Enhanced Prediction of Spending on Newly Downloaded Mobile  Games under Consumption Uncertainty (https://arxiv.org/abs/2404.08301)
Comments: 10 pages,6 figures, WWW 2024 Industry Track, with three accept, two weak accept scores

- **What's New**: 모바일 게임에서 사람들이 다운로드하는 새 게임에 지출하는 금액을 예측하는 새로운 모델을 제안합니다. 이 연구에서는 사용자 ID에 의존하지 않고 게임 소비자의 지출을 정확하게 예측할 수 있도록 협력 강화 모델(collaborative-enhanced model)을 도입하여 사용자 프라이버시를 보장하고 온라인 트레이닝을 가능하게 합니다. 또한, 이 모델은 데이터의 레이블 변동성 및 극단적 값들을 조정하여 모델 학습 과정의 안정성을 보장합니다.

- **Technical Details**: 이 연구에서는 사용자의 게임 다운로드 이력만을 이용하여 협력 신호를 모델링합니다. 이를 위해 멀티-레이어 퍼셉트론(MLP)을 사용하여 사용자 선호도 및 게임 데이터를 선형 결합으로 협력 표현을 얻습니다. 기존 크로스-피처 모델에 협력 특성을 병합하여 지출 예측 모듈로 입력함으로써 지출 예측의 정확성을 높이려고 합니다.

- **Performance Highlights**: 오프라인 데이터에서 기존 모델 대비 17.11% 향상된 성능을 보였으며, 온라인 A/B 테스트에서는 기존 모델 대비 50.65%의 비즈니스 수익 증가를 달성하였습니다. 이는 협력 강화 모델의 효과를 입증하며, 데이터의 불안정한 특성에 잘 대응하면서도 모델의 일반화 능력을 유지하게 하는 새로운 프레임워크의 중요성을 강조합니다.



### Generative Information Retrieval Evaluation (https://arxiv.org/abs/2404.08137)
Comments: Draft of a chapter intended to appear in a forthcoming book on generative information retrieval, co-edited by Chirag Shah and Ryen White

- **What's New**: 이 논문은 발전하는 Generative Information Retrieval (GenIR) 시스템과 거대 언어 모델 (LLMs) 을 평가하는 두 가지 관점에서 검토합니다. 첫 번째는 LLMs가 평가 도구로서 점차 사용되고 있으며, 기본적인 관련성 판단 작업에서 LLMs가 crowdsource 노동자들보다 우수할 수 있다는 연구 결과를 검토합니다. 두 번째는 신흥 LLM 기반의 GenIR 시스템, 특히 검색 보강 생성(Retrieval Augmented Generation, RAG) 시스템의 평가를 고려합니다.

- **Technical Details**: 이 연구는 LLM을 사용한 평가가 '느린 검색(slow search)' 형태일 수 있음을 제안하여, 더 빠른 생산 IR 시스템을 위한 평가 및 훈련 도구로서 느린 IR 시스템을 사용할 것을 제안합니다. 또한 인간 평가의 필요성을 인정하면서, 그 특성이 변화해야 할 필요가 있음을 지적합니다. LLM의 사용은 문서 평가의 문서 풀링(document pooling)의 필요성을 재평가할 수 있는 가능성을 제시하며, 크라우드소싱 노동자들에 의해 생성된 것만큼 정확한 관련성 판정을 생성할 수 있습니다.

- **Performance Highlights**: LLM을 사용한 평가는 기존의 인간 기반 평가 시스템과 비교할 때 비용을 크게 절감할 수 있으며, 동시에 빠른 평가와 최적화를 가능하게 합니다. Microsoft의 Bing은 GPT-4를 사용하여 관련성 라벨을 생성하는 데 성공하였으며, 이는 인간 평가자들에 의해 작성된 라벨과 동등한 정확도를 보여주었습니다. 이는 LLMs가 IR 시스템의 평가 및 훈련에 효과적으로 활용될 수 있음을 시사합니다.



### Overview of the TREC 2023 NeuCLIR Track (https://arxiv.org/abs/2404.08071)
Comments: 27 pages, 17 figures. Part of the TREC 2023 Proceedings

- **What's New**: TREC Neural Cross-Language Information Retrieval (NeuCLIR) 트랙의 두 번째 해는 뉴스를 위한 교차 언어 정보 검색 (Cross-Language Information Retrieval, CLIR) 작업을 계속 진행하며, 새로운 다국어 정보 검색 (Multilingual Information Retrieval, MLIR) 작업과 중국어 기술 문서에 대한 CLIR 파일럿 작업을 추가했습니다. 이 트랙은 중국어, 페르시아어, 러시안 뉴스와 중국어 과학적 초록을 포함한 네 가지 데이터 콜렉션을 생성하여, 영어 주제를 사용하여 각각의 언어로 된 문서를 검색하는 작업을 중점적으로 다룹니다.

- **Technical Details**: NeuCLIR 트랙은 신경망 (Neural) 접근 방식을 통해 CLIR의 효과를 탐구하고, 다양한 언어에 대한 토크나이제이션 (Tokenization)과 임베딩 (Embedding) 같은 계산 인프라를 활용합니다. 주요 과제는 영어 주제를 사용하여 중국어, 페르시아어 또는 러시안 뉴스 문서를 검색하고, 다양한 언어 문서를 포함하는 유니파이드 (Unified) 리스트를 생성하는 MLIR입니다. 기술 문서 CLIR 작업은 영어 주제로 중국어 기술 문서를 검색하는 것을 목표로 하며, 이는 특수 평가자 전문성을 요구하는 파일럿 작업으로 설정되었습니다.

- **Performance Highlights**: 올해 트랙에는 총 220번의 실행이 제출되었으며, 이 중 6개 팀이 참여하였습니다. 뉴스 검색 작업에서는 기존의 단일 언어 검색 결과를 기준선으로 사용하여 성능을 평가하고, MLIR 작업과 기술 문서 작업에 대한 결과도 제공됩니다. 새로운 주제 개발은 MLIR 작업 평가 최적화를 목표로 하며, 올해 도입된 새로운 접근 방식으로 다양한 언어에 대한 관련 문서를 포함하는 주제 비율을 높이려고 합니다.



### Toward FAIR Semantic Publishing of Research Dataset Metadata in the Open  Research Knowledge Graph (https://arxiv.org/abs/2404.08443)
Comments: 8 pages, 1 figure, published in the Joint Proceedings of the Onto4FAIR 2023 Workshops

- **What's New**: 이 연구는 ORKG-Dataset content type의 도입을 통해 연구 데이터셋의 발견성과 활용을 향상시키는 새로운 접근법을 제안합니다. 연구 데이터셋은 종종 충분한 메타데이터 구조가 없어 검색 엔진에서 쉽게 발견되지 않습니다. 제안된 모델은 연구 데이터셋과 그것이 수반하는 학술 출판물을 통합하여, 웹상에서 연구 데이터셋의 투명성을 높이고 더 나은 발견성을 제공합니다.

- **Technical Details**: ORKG-Dataset content type은 Open Research Knowledge Graph(ORKG) 플랫폼의 전문 브랜치로, 연구 데이터셋에 대한 서술적 정보와 의미론적(semantic) 모델을 제공합니다. 이 모델은 표준화된 프레임워크를 사용하여 연구 데이터셋을 기록하고 보고하는 데 목적이 있으며, FAIR 원칙(Findable, Accessible, Interoperable, Reusable)을 준수합니다. 또한, ReactJS와 Neo4J를 사용하여 프론트엔드 인터페이스와 백엔드 저장 및 쿼리 시스템을 구축하였습니다.

- **Performance Highlights**: 이 모델은 연구 데이터셋의 콘텍스트와 내용을 정확하게 반영하는 구조화된 메타데이터를 제공함으로써 데이터셋의 발견 가능성을 높입니다. 이는 연구 데이터셋이 관련 출판물과 함께 통합되어 표현되므로, 데이터셋 검색과 선택에 있어 핵심적인 특성을 명확히 할 수 있습니다.




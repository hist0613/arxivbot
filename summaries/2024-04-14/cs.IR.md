### Manipulating Large Language Models to Increase Product Visibility (https://arxiv.org/abs/2404.07981)
- **What's New**: 본 연구에서는 인공지능 언어 모델(Large Language Models, LLMs)을 이용한 쇼핑 검색 엔진에서 제품의 가시성을 높일 수 있는 방법을 살펴보았습니다. 이를 위해 제품 정보 페이지에 전략적 텍스트 시퀀스(Strategic Text Sequence, STS)를 추가하는 방법을 제시하였으며, 이 방법이 언어 모델의 추천 결과에 어떤 영향을 미치는지 분석했습니다.



### An efficient domain-independent approach for supervised keyphrase  extraction and ranking (https://arxiv.org/abs/2404.07954)
- **What's New**: 이 연구에서는 단일 문서에서 핵심 구문을 자동 추출하기 위한 감독 학습 접근법을 제시합니다. 이 접근법은 외부 지식 기반, 사전 훈련된 언어 모델, 또는 단어 임베딩에 의존하지 않고, 후보 구문의 통계적 및 위치적 특성을 단순하게 계산하여 사용합니다.

- **Technical Details**: 제안된 솔루션의 순위 결정 구성 요소는 비교적 가벼운 앙상블 모델입니다. 이 모델은 귀금속 키워드(corpus of 'golden' keywords)나 외부 지식 코퍼스에 의존하지 않음으로써, 감독 학습 방식이지만 미감독(Unsupervised) 솔루션의 이점을 어느 정도 유지합니다.

- **Performance Highlights**: 벤치마크 데이터셋에서의 평가는 제안된 접근법이 여러 최신 기술(BaseLine) 모델들, 특히 모든 비감독(deep learning-based unsupervised) 모델들 보다 현저히 높은 정확도를 달성하고, 일부 감독 학습 기반의 딥 러닝 모델들과도 경쟁력 있는 성능을 보여 줍니다.



### M-scan: A Multi-Scenario Causal-driven Adaptive Network for  Recommendation (https://arxiv.org/abs/2404.07581)
- **What's New**: 본 연구는 다양한 시나리오(multi-scenario)에서의 추천 시스템에 주목하여, 데이터가 제한된 시나리오에서 예측을 향상시키기 위해 다른 시나리오에서의 데이터를 효과적으로 활용하는 것에 중점을 두고 있습니다. 이를 해결하기 위해, 'Multi-Scenario Causal-driven Adaptive Network (M-scan)'이라는 새로운 모델을 제안하며, 이는 시나리오 인식 공동 주의 메커니즘(Scenario-Aware Co-Attention mechanism)과 시나리오 바이어스 제거 모듈(Scenario Bias Eliminator module)을 통합하여 명시적으로 사용자의 관심사를 추출하고, 다른 시나리오로부터 도입된 편향을 완화합니다.

- **Technical Details**: M-scan 모델은 시나리오 인식 공동 주의 메커니즘을 통해 현재 시나리오와 일치하는 다른 시나리오의 사용자 관심사를 명시적으로 추출합니다. 또한, 인과적 반사실 추론( causal counterfactual inference)을 사용하는 시나리오 바이어스 제거 모듈을 활용하여 다른 시나리오로부터 도입된 데이터의 바이어스를 완화합니다. 이러한 구조는 기존의 모델들이 가지는 훈련의 어려움, 불완전한 사용자 표현, 그리고 최적이 아닌 성능 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: M-scan의 효과성은 두 개의 공개 데이터셋에서의 광범위한 실험을 통해 검증되었습니다. 이 결과는 M-scan이 기존의 베이스라인(base line) 모델들에 비해 우수한 성능을 보여주었습니다. 시나리오 자체가 클릭 행동에 직접적인 영향을 미친다는 것을 밝히며, 기존의 접근법이 다른 시나리오의 클릭 행동을 직접적으로 활용할 때 발생하는 예측 편향을 해결하는 데 기여했습니다.



### Can Large Language Models Assess Serendipity in Recommender Systems? (https://arxiv.org/abs/2404.07499)
- **What's New**: 새로운 연구에서는 추천 시스템에서의 과도한 특수화(over-specialization)를 방지하기 위한 목적으로 '우연성(serendipity)'에 초점을 맞춘 추천 시스템을 다루고 있습니다. 이 연구는 Large Language Models(LLMs)가 인간의 감정적 반응을 어떻게 예측할 수 있는지 탐구하며, 특히 우연한 발견이 포함된 추천 아이템에 대한 사용자의 반응을 이해하는 데 중점을 둡니다.

- **Technical Details**: 이 연구에서는 LLM을 활용하여 사용자가 추천된 아이템을 우연성 있는 것으로 평가할 가능성을 이진 분류 작업(binary classification task)으로 예측합니다. 세 가지 LLM의 예측 성능이 인간이 지정한 기준적 진리(ground truth)가 있는 벤치마크 데이터셋에서 측정되었습니다. 또한, LLM이 제공하는 출력의 해석이 어렵다는 점, 그리고 사용자 평점 이력의 입력 수가 적절한지에 대한 주의가 요구됩니다.

- **Performance Highlights**: 실험 결과, LLM 기반의 평가 방법이 인간의 평가와 높은 일치율을 보이지는 않았지만, 기준 모델(baseline methods)에 비해 동등하거나 더 나은 성능을 보였습니다. 이러한 결과는 LLM을 사용하여 추천 시스템에서의 우연성 평가를 자동화하는 가능성을 시사하지만, 그 결과의 해석과 사용자 평가의 정확성 제어는 추가적인 연구가 필요함을 나타냅니다.



### Adaptive Fair Representation Learning for Personalized Fairness in  Recommendations via Information Alignmen (https://arxiv.org/abs/2404.07494)
Comments: This paper has been accepted by SIGIR'24

- **What's New**: 최근 연구자들 사이에서 추천 시스템의 개인화된 공정성(Personalized Fairness)에 대한 관심이 증가하고 있습니다. 본 논문에서는 기존의 방법들이 겪고 있는 훈련 비용의 증가와 공정성과 정확성 사이의 최적이 아닌 트레이드오프 문제점을 해결하기 위해 새로운 적응형 공정 표현 학습(Adaptive Fair Representation Learning, AFRL) 모델을 제안합니다. AFRL은 추론 단계에서 다양한 공정성 요구 사항을 적응적으로 수용할 수 있는 장점이 있습니다.

- **Technical Details**: AFRL은 공정성 요구를 입력으로 처리하고, 불공정한 사용자 임베딩(Unfair User Embedding)에서 각 속성별 특성 임베딩(Attribute-specific Embedding)을 학습할 수 있으며, 이를 통해 사용자의 고유한 공정성 요구에 따라 비민감 속성(Non-sensitive Attributes)을 결정할 수 있는 적응성을 제공합니다. 또한, AFRL은 정보 정렬(Information Alignment)이라는 새로운 방법을 사용하여 비민감 속성의 차별적 정보를 정확하게 보존하고, 공정성에 손실 없이 속성 독립적인 협업 신호(Attribute-independent Collaborative Signals)를 포착할 수 있는 편향 없는 협업 임베딩(Debiased Collaborative Embedding)을 공정한 임베딩(Fair Embedding)에 통합합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 광범위한 실험과 탄탄한 이론적 분석은 AFRL의 우수성을 입증합니다. AFRL은 공정성과 정확성 사이의 향상된 트레이드오프를 제공하며, 다양한 공정성 요구에 효과적으로 적응할 수 있는 능력을 제공함으로써 추천 시스템에서의 개인화된 공정성 실현에 기여합니다.



### Improving Retrieval for RAG based Question Answering Models on Financial  Documents (https://arxiv.org/abs/2404.07221)
- **What's New**: 이 논문에서는 효과적인 대화형 텍스트 생성을 위해 필수적인 Retrieval Augmented Generation (RAG, 검색 증강 생성) 기술의 한계를 탐구하고, 이를 개선하기 위한 다양한 방법론을 제안합니다. 특히, 본 연구는 기존 RAG 방식에서 발생하는 문제인 불필요하거나 부정확한 텍스트 청크의 검색 문제를 해결하고자 합니다.

- **Technical Details**: 본 논문은 향상된 chunking techniques (청크 생성 기술), query expansion (질의 확장), metadata annotations (메타데이터 주석)의 적용, re-ranking algorithms (재랭킹 알고리즘) 사용 및 embedding algorithms (임베딩 알고리즘)의 미세 조정 등을 통해 검색의 정확성을 높이는 방법을 제안합니다. 이러한 기술들은 LLMs에 적합한 콘텍스트를 제공하여 응답의 정확성과 신뢰도를 향상시킬 수 있습니다.

- **Performance Highlights**: 이 연구는 더 정교한 검색 및 청크 생성 기술을 적용함으로써 LLM이 제공하는 답변의 정확도와 관련성이 향상될 수 있음을 시사합니다. 특히, domain-specific (도메인 특화) 정보를 처리할 때 발생할 수 있는 오류를 줄이는 쪽으로 적용 가능성이 높으며, 이는 실제 산업 활동에서 LLM의 활용도를 크게 높일 수 있습니다.



### Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy  with Semantic Search and Hybrid Query-Based Retrievers (https://arxiv.org/abs/2404.07220)
Comments: Pre-print version of paper submitted to conference

- **What's New**: 이 논문에서는 'Blended RAG' 방법을 제안하여 문서의 사설 지식 기반과 대규모 언어 모델(LLM)을 결합하는 Retrieval-Augmented Generation (RAG) 접근 방식의 정확도를 향상시키고자 하였습니다. 이 방법은 Dense Vector Index와 Sparse Encoder Index와 같은 의미 검색(semantec search) 기술을 이용하며, 하이브리드 쿼리 전략을 혼합 사용하였습니다.

- **Technical Details**: 이 논문에서는 키워드 기반 유사성 검색, 밀집 벡터 기반 검색, 의미 기반 희소 인코더 검색 등 세 가지 검색 전략을 탐구하고 통합하여 하이브리드 쿼리를 형성했습니다. 또한, 이 연구는 다양한 검색 기술을 체계적으로 평가하여 BM25, KNN, Elastic Learned Sparse Encoder (ELSER)와 같은 주요 인덱스를 사용하였고, 이를 통해 문서 및 쿼리 콘텐츠에서 유도된 벡터 표현의 근접성을 식별하는 데 중점을 두었습니다.

- **Performance Highlights**: Blended Retriever를 포함한 실험에서는 특히 NQ와 TREC-COVID 데이터셋에서 높은 검색 정확도를 보였습니다. 특히, Best Fields를 사용하는 Sparse Encoder를 활용한 하이브리드 쿼리는 NQ 데이터셋에서 88.77%의 높은 검색 정확도를 달성하여 기존의 벤치마크를 상회하는 결과를 보였습니다. 또한 이러한 하이브리드 검색 방식은 SqUAD와 같은 Generative Q&A 데이터셋에 적용할 때도 향상된 결과를 보여, 기존 파인 튜닝 성능을 능가했습니다.



### Leave No One Behind: Online Self-Supervised Self-Distillation for  Sequential Recommendation (https://arxiv.org/abs/2404.07219)
- **What's New**: 이 논문은 새롭게 'Online Self-Supervised Self-distillation for Sequential Recommendation (S⁴Rec)'라는 학습 패러다임을 제안합니다. 이는 자기 감독 학습(self-supervised learning)과 자기 증류(self-distillation) 접근 방식을 결합하여 순차 추천 시스템에서 데이터 희소성 문제를 해결하고자 합니다. 특히, 온라인 군집화(online clustering)를 사용하여 사용자를 그들의 고유한 잠재적 의도에 따라 그룹화하고, 행동 길이의 영향을 받지 않도록 적대적 학습 전략(adversarial learning strategy)을 사용합니다.

- **Technical Details**: S⁴Rec 모델은 크게 세 가지 주요 기술적 구성 요소를 사용합니다: 1) 온라인 군집화를 통해 사용자들을 그들의 잠재적 의도에 따라 효과적으로 그룹화하고, 2) 적대적 학습을 활용해 군집화 과정이 사용자 행동의 길이에 영향을 받지 않도록 합니다. 3) 자기 증류(self-distillation)를 통해 행동 데이터가 풍부한 사용자(교사)로부터 행동 데이터가 제한된 사용자(학생)로 지식을 전달하는 클러스터 인식 증류 모듈(cluster-aware self-distillation module)을 사용합니다.

- **Performance Highlights**: S⁴Rec은 4개의 실제 데이터셋에서 실험을 수행하였고, 이러한 데이터셋들에서 최신 기술(state-of-the-art) 성능을 보여주었습니다. 이는 S⁴Rec 모델이 데이터 희소성에 강하며, 다양한 사용자 행동 패턴을 효과적으로 학습할 수 있음을 입증합니다. 또한 코드는 공개적으로 제공되어 연구 및 실용화에 직접적으로 활용될 수 있습니다.



### Auditing health-related recommendations in social media: A Case Study of  Abortion on YouTub (https://arxiv.org/abs/2404.07896)
- **What's New**: 이 연구에서는 YouTube가 낙태 관련 비디오를 어떻게 추천하는지 조사하기 위해 간단하지만 효과적인 이른바 '양말 인형(sock puppet)' 감사 접근법을 도입하고 있습니다. 이러한 접근 방식을 통해 다양한 배경을 가진 개인에게 제공되는 추천의 차이를 감사할 수 있습니다. 소셜 미디어에서 건강 관련 정보의 신뢰성과 정확성을 높이기 위한 알고리즘 감사의 중요성을 강조하고 있습니다.

- **Technical Details**: 이 연구는 YouTube 추천 알고리즘(Recommendation System, RS)을 '블랙 박스(black box)'로 보고, 낙태와 같은 민감한 주제에 대한 비디오 추천의 편향성 및 잘못된 정보의 확산 여부를 실증적으로 조사했습니다. 이를 위해 '양말 인형'을 사용한 데이터 수집 메커니즘을 구현하여, 다양한 실험 시나리오 하에서 YouTube 비디오 추천을 관찰하고 분석하였습니다. 검색(Search) 및 추천(Recommendation)의 두 가지 주요 목적으로 작용하는 YouTube의 알고리즘은 사용자의 검색 및 시청 이력에 기반하여 비디오를 제안하고 관련 자료를 중요도 순으로 정렬하는 과정을 거칩니다.

- **Performance Highlights**: 이 연구는 YouTube가 낙태와 관련된 내용에서 특정 시각을 더욱 증진시키려는 경향이 있는지, 그리고 잘못된 정보를 얼마나 효과적으로 제거하고 있는지를 평가하고자 합니다. 알고리즘의 추천 결과에서 발견된 편향과 잘못된 정보를 식별함으로써 사용자가 신뢰할 수 있는 건강 정보를 얻을 수 있는 환경을 조성하는 데 기여하고자 합니다. 연구 결과는 YouTube의 추천 시스템이 사용자를 극단적이고 극화된 콘텐츠로 안내할 수 있는 잠재적 리스크를 보여줍니다.



### Detection of financial opportunities in micro-blogging data with a  stacked classification system (https://arxiv.org/abs/2404.07224)
- **What's New**: 새로운 시스템이 제안되었습니다: 트위터(Twitter)에서의 금융 기회를 탐지하기 위한 세 계층(Stacked) 머신 러닝(Machine Learning) 분류 시스템입니다. 이 시스템은 특정 자산의 가치 상승에 대한 예측이나 추측을 하는 트윗을 고도의 정밀도로 감지할 수 있습니다. 이는 플루치크(Plutchik)의 '기대' 감정에 해당하는 '기회'라는 금융 감정 유형을 분류합니다.

- **Technical Details**: 이 시스템은 세 단계로 구성되어 있으며, 첫 번째 단계에서는 중립적인 트윗을 제거하고, 두 번째 단계에서는 일반적인 긍정적 감정과 부정적 감정을 구분하며, 마지막 단계에서는 '기회'로 정의된 긍정적 감정과 다른 긍정적 감정을 구분합니다. NLP(Natural Language Processing) 기술을 사용하여 언어적 정보를 추출하고 다양한 특징을 활용합니다. 이에는 n-gram 시퀀스, 감정 및 극성(polarity) 사전, 해시태그, 숫자 정보 및 백분율의 빈도 계수 등이 포함됩니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 금융 기회 감지에 있어 최대 83%의 정밀도를 달성했습니다. 이는 트위터 데이터를 활용한 금융 시장 예측에 큰 기여를 할 수 있음을 시사하며, 투자자들의 의사 결정을 지원하기 위한 유용한 도구로 평가됩니다.



### TransTARec: Time-Adaptive Translating Embedding Model for Next POI  Recommendation (https://arxiv.org/abs/2404.07096)
Comments: This paper has been accepted by the 2024 5th International Conference on Computer Engineering and Application (ICCEA 2024)

- **What's New**: 이 논문에서는 위치 기반 서비스의 발전으로 생성된 방대한 체크인 기록을 활용하여, 지금까지와는 다른 방식인 시간적 요소를 고려한 새로운 추천 모델 TransTARec을 제안하고 있습니다. 이 모델은 사용자의 선호도, 시간적 영향력, 그리고 순차적 동적(Dynamics)을 하나의 변환 벡터로 통합하여 다음 POI(Point-of-Interest) 추천의 정확성을 높이고자 합니다.

- **Technical Details**: TransTARec 모델은 기존의 변환 임베딩 방식을 확장하였으며, 사용자의 이전 위치, 시간 정보를 포함한 트리플렛을 기반으로 하여 시간 조절 가능한 변환 벡터를 사용합니다. 이 변환 벡터는 신경망 기반 융합 연산을 통해 생성되어 사용자 선호와 시간적 영향을 결합합니다. 이러한 접근 방식은 높은 차원의 데이터를 처리하고, 데이터의 희소성(Sparsity) 문제를 해결하는 데 유리합니다.

- **Performance Highlights**: 실제 세계 데이터셋(Foursquare, Mobile Dataset)에서의 실험을 통해 TransTARec는 이전 기법들 대비 최대 14.63%까지 정밀도(precision) 향상을 보였습니다. 이는 제안된 모델이 시간적 영향을 임베딩하는 과정에서의 유의미한 성과를 나타내며, 사용자의 다음 POI 선정에 있어서 시간적 컨텍스트가 중요한 역할을 함을 입증합니다.



### Quati: A Brazilian Portuguese Information Retrieval Dataset from Native  Speakers (https://arxiv.org/abs/2404.06976)
Comments: 22 pages

- **What's New**: 데이터셋 'Quati'는 브라질 포르투갈어로 작성된 질의(query)와 문서(document)를 포함하고 있어, 이 언어의 사용자들에게 더욱 적합하고 의미 있는 정보 검색(information retrieval, IR) 평가를 가능하게 합니다. 이 데이터셋은 기존에 번역된 데이터셋의 한계를 극복하고자, 원어민이 작성한 컨텐츠로 구성되었습니다.

- **Technical Details**: Quati 데이터셋은 고품질의 브라질 포르투갈어 웹사이트에서 추출된 문서를 기반으로 합니다. 이 문서들은 실제 사용자들이 자주 방문하는 웹사이트에서 선별되었으므로, 더욱 실질적이고 관련성 높은 텍스트를 포함하고 있습니다. 데이터셋의 라벨링 과정에는 최신 Large Language Model(LLM)을 사용하여, 인간의 평가와 비교할 수 있는 수준의 일관성 있는 데이터 품질을 달성했습니다. 데이터셋 생성을 위한 반자동 파이프라인(semi-automated pipeline)도 설명되어 있어, 다른 언어에 대한 유사한 고품질 IR 데이터셋을 생성하는 데 도움이 될 수 있습니다.

- **Performance Highlights**: Quati는 다양한 오픈 소스 및 상용 검색 시스템(retrievers)에 대해 평가되었으며, 이를 통해 baseline 시스템들의 성능을 확인할 수 있습니다. 또한, Quati 생성 비용은 쿼리-패시지(query-passage) 당 $0.03이며, 평균적으로 각 쿼리 당 97.78개의 패시지가 주석 처리되었다는 점에서 경제적 효율성도 갖추고 있습니다. 이러한 저비용 고효율의 데이터셋 구축 방법론은 향후 다양한 언어의 IR 데이터셋 개발에 중요한 기준을 제시할 수 있습니다.



### Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise  Passage Re-Ranking with Cross-Encoders (https://arxiv.org/abs/2404.06912)
- **What's New**: 새로운 크로스-인코더(쿼리와 텍스트 간의 의미 연관성을 평가하는 인공 지능) 아키텍처인 'Set-Encoder'가 제안되었습니다. 이는 입력 텍스트의 순서에 관계없이 동일한 순위를 출력함으로써 순열 불변성(permutation invariance)을 보장하고, 더 많은 텍스트를 동시에 처리할 수 있도록 메모리 효율성을 개선했습니다.

- **Technical Details**: Set-Encoder는 통합된 주의 토큰(inter-passage attention)을 사용하여 여러 텍스트를 병렬로 처리하면서도 텍스트 간 상호작용을 가능하게 합니다. 이 아키텍처는 순서에 민감하지 않은 위치 인코딩을 공유하는 [CLS] 토큰을 사용하여, 텍스트들이 상호 정보를 교환할 수 있도록 합니다. 또한, 퓨즈드-어텐션 커널(fused-attention kernels)을 사용해 한 번에 최대 100개의 텍스트를 재순위할 수 있으며, 이로 인해 더 큰 데이터 세트에서도 효율적인 학습과 미세조정이 가능합니다.

- **Performance Highlights**: TREC Deep Learning 및 TIREx 플랫폼에서의 실험을 통해 Set-Encoder는 기존 크로스-인코더들보다 효과적이며, 크기에 비해 더 효율적인 것으로 나타났습니다. 특히, 도메인 외(out-of-domain) 시나리오에서의 성능 개선이 두드러졌습니다. 데이터셋을 통한 추가적 미세조정을 통해 MS MARCO에서의 노이즈가 있는 하드 네거티브 샘플보다 더 양질의 데이터에서 뛰어난 결과를 보였습니다.



### NFARec: A Negative Feedback-Aware Recommender Mod (https://arxiv.org/abs/2404.06900)
Comments: Accepted to SIGIR 2024

- **What's New**: 이 논문에서는 기존의 그래프 신경망(Graph Neural Networks, GNN) 기반 추천 시스템이 부정적인 피드백을 간과하는 문제를 해결하기 위해 부정적 피드백을 고려한 추천 모델인 NFARec을 제안합니다. NFARec은 부정적 피드백을 학습하고 활용하여 사용자의 선호도 예측 및 항목 추천의 정확도를 향상시키는 것을 목표로 합니다.

- **Technical Details**: NFARec은 피드백 인식 상관 관계(Feedback-aware Correlation)를 사용하여 하이퍼그래프 합성곱(Hypergraph Convolutions, HGCs)을 통해 사용자의 구조적 표현을 학습합니다. 추가적으로, NFARec은 트랜스포머 호크스 프로세스(Transformer Hawkes Process)를 기반으로 다음 상호작용의 피드백 감정 극성(Feedback Sentiment Polarity)을 예측하는 보조 작업을 포함하여 사용자의 감정과 행동 패턴을 더 깊이 이해할 수 있습니다.

- **Performance Highlights**: NFARec은 다양한 베이스라인을 포함한 경쟁 모델들과 비교하여 우수한 성능을 보였습니다. 특히, 부정적 피드백을 고려하여 차별화된 상호작용 패턴을 반영할 수 있도록 설계된 핵심 아키텍처 덕분에 추천 정확도가 향상되었으며, 예측 모델의 강인성 또한 개선되었습니다.



### CaDRec: Contextualized and Debiased Recommender Mod (https://arxiv.org/abs/2404.06895)
Comments: Accepted to SIGIR 2024

- **What's New**: 이 논문에서는 사용자의 행동 패턴을 탐색하는 추천 시스템을 위한 새로운 모델, CaDRec을 제안합니다. CaDRec은 하이퍼그래프 컨볼루션 연산자 (Hypergraph Convolution Operators)와 자기주의 메커니즘 (Self-Attention Mechanism)을 통합하여 노드 임베딩의 과잉 평활화 문제를 해결하고, 인기도와 사용자 개인 편향을 분리하여 편향된 상호작용 분포를 극복합니다.

- **Technical Details**: CaDRec은 두 가지 주요 문제를 해결하기 위해 설계되었습니다: 1) 구조적 및 순차적 맥락을 통합하여 효과적인 이웃을 선택하는 새로운 하이퍼그래프 컨볼루션 연산자를 소개하여 과잉 평활화 문제를 해결하고, 2) 아이템의 인기도를 위치 인코딩(Positional Encoding)으로 통합하고 사용자 개인 편향을 모델링하여 편향된 상호작용 분포를 극복합니다. 또한, 아이템 임베딩을 업데이트하는 그라디언트의 불균형이 인기도 편향을 악화시키는 것을 수학적으로 보여주며, 정규화 및 가중치 기법을 도입하여 해결합니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에 대한 광범위한 실험을 통해, CaDRec은 기존의 최신 기술(State-of-the-art, SOTA) 방법들보다 우수한 성능을 보여주며, 효율적인 시간 복잡도를 유지합니다. 이를 통해 사용자의 진정한 선호도와 아이템의 실제 속성을 더 정확하게 파악할 수 있습니다.



### From Model-centered to Human-Centered: Revision Distance as a Metric for  Text Evaluation in LLMs-based Applications (https://arxiv.org/abs/2404.07108)
Comments: 9 pages, 2 figures, under review

- **What's New**: 본 연구는 인공지능(AI) 구동 글쓰기 보조 애플리케이션의 상황에서 대규모 언어 모델(Large Language Models, LLM)의 평가를 사용자 중심으로 전환하며, 새로운 평가 척도인 'Revision Distance'를 제안합니다. 이 척도는 LLM이 생성한 수정 편집을 계산하여 인간의 글쓰기 과정을 모방합니다.

- **Technical Details**: ‘Revision Distance’는 LLM이 제안한 수정 사항을 계산하여 결정되는 메트릭으로, 사용자에게 더 명확하고 상세한 피드백을 제공합니다. 이 메트릭은 기존의 ROUGE, BERT-Score, GPT-Score와 일치하지만 더 상세한 피드백을 제공하며, 텍스트 간의 차이를 더 잘 구분합니다. 또한, 참고 텍스트가 부족한 시나리오에서도 중요한 가능성을 보여줍니다.

- **Performance Highlights**: ‘Revision Distance’ 메트릭은 쉬운 글쓰기 작업에서 기존 메트릭과 일관성이 있으며, 더 도전적인 학술 글쓰기 작업에서도 안정적이고 신뢰할 수 있는 평가 결과를 제공합니다. 참고 없는 시나리오에서는 인간 판단과 약 76% 일치하며, 편집 유형을 분류함으로써 더 세밀한 분석을 제공합니다.



### Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on  Graphs (https://arxiv.org/abs/2404.07103)
Comments: 21 pages. Code: this https URL

- **What's New**: 새로운 대규모 언어 모델(LLMs, Large Language Models)의 한계, 즉 '환각(hallucinations)' 문제를 해결하기 위해, 기존의 텍스트 데이터만을 활용하는 대신 '그래프 연결된 데이터(graph-attributed data)'를 통합하는 새로운 접근 방법을 제안합니다. 이를 위해 '그래프 추론 벤치마크(Graph Reasoning Benchmark, GRBench)' 데이터셋을 수작업으로 구축하였으며, 이는 10개의 도메인 그래프로부터 지식을 추출하여 1,740개의 질문에 답할 수 있습니다.

- **Technical Details**: 제안된 '그래프 사고 연쇄(Graph Chain-of-thought, Graph-CoT)' 프레임워크는 LLMs를 그래프와 반복적으로 상호 작용하도록 유도합니다. 각각의 Graph-CoT 반복은 세 개의 하위 단계로 이루어진다: LLM의 추론(LLM reasoning), LLM과 그래프 간의 상호작용(LLM-graph interaction), 그리고 그래프 실행(graph execution). 이 구조는 LLMs가 단순히 텍스트 단위의 정보를 처리하는 것을 넘어 그래프 구조상에서의 연결된 정보까지 고려하게 합니다.

- **Performance Highlights**: 실험 결과, Graph-CoT는 기존의 베이스라인 모델들을 일관되게 능가하는 성능을 보였습니다. GRBench 데이터셋에서의 평가에는 세 가지 유형의 LLM 백본(Large Language Model backbones)이 사용되었으며, Graph-CoT의 접근 방식이 특히 지식 집약적인 작업에 우수한 결과를 나타내었습니다.



### Milgram's experiment in the knowledge space: Individual navigation  strategies (https://arxiv.org/abs/2404.06591)
Comments: 25 pages, 8 figures

- **What's New**: 이 연구에서는 위키백과(Wikipedia)를 이용한 내비게이션 게임을 통해 정보 공간에서의 탐색 전략이 개인의 특성에 따라 어떻게 다양하게 나타나는지 분석했습니다. 특히, 유명 인물을 대상으로 탐색할 때 참가자들이 지리적 및 직업적 정보를 활용하는 경향이 두드러졌으며, '군중의 지혜' 효과가 나타나는 것을 발견했습니다.

- **Technical Details**: 연구팀은 DeepWalk 알고리즘을 사용하여 64차원의 그래프 임베딩(graph embedding)을 훈련시켜 위키백과 문서 간의 의미적 관계와 지식 계층 구조 내 위치를 분석했습니다. 이러한 임베딩을 통해 문서들의 주제적 유사성을 기반으로 거리를 측정할 수 있었으며, 위키백과 네트워크 내에서 문서의 계층적 위치를 평가하는 방법도 개발했습니다.

- **Performance Highlights**: 연구 결과, 참가자들은 다양한 내비게이션 전략을 채택했으며, 이는 정보 환경 및 개인적 특성에 의해 영향을 받는 것으로 나타났습니다. 특히, 전략 세트는 대상 주변의 정보 환경을 잘 반영하는 것으로 평가되었으며, 이는 참가자들 간의 개별 차이가 서로를 보완하는 효과를 가진다는 것을 시사합니다.



### DRE: Generating Recommendation Explanations by Aligning Large Language  Models at Data-lev (https://arxiv.org/abs/2404.06311)
Comments: 5 pages, 2 figures

- **What's New**: 이 논문에서는 추천 시스템(recsys)의 투명성이 부족할 때 사용자가 혼란을 겪는 문제를 해결하기 위해 데이터 레벨 추천 설명(DRE: Data-level Recommendation Explanation)을 도입합니다. DRE는 기존 방법과 달리 추천 모델의 내부 표현에 접근하거나 모델을 수정할 필요 없이, 대규모 언어 모델(LLMs)을 활용하여 사용자 데이터와 추천 아이템 간의 관계를 추론합니다.

- **Technical Details**: DRE는 추천 모델과 설명 모듈 사이의 데이터 레벨 정렬 방법을 제안합니다. 이는 LLM을 이용하여 사용자의 과거 행동 데이터와 추천 아이템을 입력으로 받아 두 모듈 간의 정렬을 달성합니다. 또한, DRE는 타겟 인식 사용자 선호도 증류(target-aware user preference distillation) 방법을 도입하여 아이템 리뷰를 활용, 사용자 선호와 추천 아이템의 의미론적 정보를 풍부하게 합니다.

- **Performance Highlights**: 벤치마크 데이터셋을 사용한 실험에서 DRE는 사용자가 관심 있는 측면을 정확하게 설명하며, 이에 따라 추천 아이템에 대한 사용자의 관심을 증진시킴을 보여줍니다. 논문은 DRE가 설명의 정확성 측면에서 기존 방법들보다 우월함을 입증합니다.



### End-to-end training of Multimodal Model and ranking Mod (https://arxiv.org/abs/2404.06078)
Comments: 9 pages, 8 figures

- **What's New**: EM3 (End-to-End Multimodal Model 및 Ranking Model을 사용한 산업용 추천 시스템)은 추천 시스템에서 새롭게 제안된 프레임워크로, 멀티모달 정보를 효과적으로 활용하여 개인화된 순위 결정 작업을 지원합니다. EM3는 멀티모달 모델의 핵심 모듈을 직접 훈련시켜 리소스 소비를 크게 증가시키지 않으면서도 작업 지향적인 콘텐츠 특성을 추출할 수 있습니다.

- **Technical Details**: EM3는 Fusion-Q-Former, Low-Rank Adaptation, 그리고 Content-ID Contrastive learning을 포함하는 세 가지 주요 기술을 제안합니다. Fusion-Q-Former는 다양한 모달리티를 통합하고 고정 길이의 강인한 멀티모달 임베딩을 생성하는 데 사용됩니다. Low-Rank Adaptation 기술은 시퀀스 길이가 긴 유저 콘텐츠 관심사 모델링에서 파라미터의 대량 소비와의 갈등을 완화합니다. 마지막으로, 콘텐츠-ID 대조 학습(Content-ID Contrastive learning)은 콘텐츠와 ID의 이점을 서로 조화시켜보다 작업 지향적인 콘텐츠 임베딩과 일반화된 ID 임베딩을 획득합니다.

- **Performance Highlights**: EM3는 오프라인 데이터셋과 온라인 A/B 테스트 모두에서 기존 방식보다 뛰어난 성능을 보여주었으며, 이는 추천 정확도의 상당한 향상을 나타내는 결과를 제공했습니다. 이 프레임워크는 또한 두 개의 공개 데이터셋에서 상태 기술(state-of-the-art) 방법보다 우수한 성능을 발휘함을 보여주었습니다.



### AiSAQ: All-in-Storage ANNS with Product Quantization for DRAM-free  Information Retrieva (https://arxiv.org/abs/2404.06004)
Comments: 5 pages, 6 figures and 4 tables

- **What's New**: 본 연구에서는 모든 인덱스 데이터를 저장 장치에 오프로드하는 새로운 근사 최근접 이웃 탐색(ANNS) 방법 'AiSAQ (All-in-Storage ANNS with Product Quantization)'을 제안합니다. AiSAQ은 기존 DiskANN과 비교하여 무시할만한 성능 저하로 훨씬 적은 메모리(~10MB)를 사용하며, 여러 대규모 데이터셋 간의 인덱스 전환을 밀리초 단위로 실행할 수 있습니다, 이는 검색-증강 생성(RAG)의 유연성을 크게 향상시킵니다.

- **Technical Details**: AiSAQ은 기존의 그래프 기반 ANNS 알고리즘에 적용 가능하며, 데이터를 SSD와 같은 저장 장치에 최적화하여 배치함으로써, 저장 공간의 큰 비중이 저장 장치에 위치함에도 불구하고 DiskANN에 비해 미미한 지연 시간 증가만을 가져옵니다. 또한, 이 방법은 다중 빌리언-스케일 데이터셋에서 고속으로 인덱스를 전환할 수 있도록 설계되었습니다.

- **Performance Highlights**: AiSAQ은 SIFT1B와 같은 빌리언-스케일 데이터 세트에서도 RAM 사용량을 10MB로 유지하면서, 높은 재현율과 원래의 DiskANN의 그래프 토폴로지를 유지합니다. 또한, SSD에서의 데이터 배치 최적화 덕분에 95% 이상의 1-recall@1 지표를 밀리초 단위 대기 시간으로 달성할 수 있으며, 인덱스 로드 시간이 무시할 정도로 빠릅니다.



### Wasserstein Dependent Graph Attention Network for Collaborative  Filtering with Uncertainty (https://arxiv.org/abs/2404.05962)
Comments: This work has been submitted to the IEEE TCSS for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 논문에서는 협업 필터링(Collaborative Filtering, CF)에 불확실성을 모델링하는 새로운 접근 방식인 Wasserstein dependent Graph ATtention network (W-GAT)를 제안합니다. 이 방법은 그래프 주의 네트워크(Graph Attention Network, GAN)와 Wasserstein 거리를 활용하여 사용자와 아이템의 가우시안(Gaussian) 임베딩을 학습하고, KL 발산(Kullback-Leibler divergence)이 가지는 한계를 극복하고자 합니다.

- **Technical Details**: W-GAT 모델은 사용자와 아이템을 가우시안 분포로 표현하고, 이를 통해 불확실성을 캡처합니다. 모델은 Wasserstein 거리를 사용하여 분포 간의 유사성을 측정하고, Wasserstein-dependent 상호 정보(mutual information)를 최대화하여 사용자의 선호도를 더 정확하게 포착할 수 있습니다. 이는 기존의 CF 방법에서 놓치기 쉬운 사용자의 다양한 관심사와 범주화된 아이템의 미묘한 차이를 잘 표현할 수 있게 합니다.

- **Performance Highlights**: 실험 결과 W-GAT는 세 개의 벤치마크 데이터셋에서 여러 기존의 대표적인 베이스라인들을 능가하는 성능을 보였습니다. 이는 W-GAT가 사용자와 아이템의 불확실성을 포괄적으로 모델링하는 능력이 뛰어나며, 결과적으로 추천 시스템의 다양성과 설명 가능성(explainability)을 향상시킬 수 있음을 의미합니다.



### LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language  Models and Doc-Level Embedding (https://arxiv.org/abs/2404.05825)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Model, LLM)을 활용한 모델 비특정적(doc-level embedding) 문서 레벨 임베딩 프레임워크를 소개하며, 이를 통해 Bi-encoder와 late-interaction 모델의 효율성을 크게 향상시켰습니다. 특히, 음성 샘플링(negative sampling)과 손실 함수(loss function)와 같은 검색 모델 훈련의 중요 구성 요소들에서의 개선을 진행했습니다.

- **Technical Details**: 이 연구는 검색 품질과 견고성을 향상시키기 위해 문서 벡터 임베딩에 맥락 정보를 풍부하게 하는 LLM-augmented retrieval을 제안합니다. 주요 구성요소로는 음성 샘플링 방법과 정보NCE 손실(InfoNCE loss) 함수가 있습니다. 또한, Bi-encoder와 late-interaction 모델 아키텍처에 문서 레벨 임베딩을 적용하여 종단간 검색 품질을 향상시키는 방법을 설명합니다.

- **Performance Highlights**: LLM-augmented retrieval 프레임워크는 LoTTE 및 BEIR 데이터셋에서 기존 모델들을 능가하는 최첨단(state-of-the-art) 결과를 달성했습니다. 이는 새로운 임베딩 방식과 향상된 훈련 과정이 종합적인 검색 작업에서의 성능 개선을 이끌어 냄을 보여줍니다.



### Learning State-Invariant Representations of Objects from Image  Collections with State, Pose, and Viewpoint Changes (https://arxiv.org/abs/2404.06470)
Comments: This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구에서는 인식과 검색을 위한 객체 표현 학습에 '상태 불변성(state invariance)'이라는 새로운 불변성을 추가합니다. 이는 예를 들어 우산이 접혔거나 옷이 바닥에 던져졌을 때와 같이 객체의 구조적 형태가 변할 때 견고함을 의미합니다. 'ObjectsWithStateChange'라는 새로운 데이터셋을 소개하며, 이 데이터셋은 임의의 관점에서 기록된 객체 이미지의 상태 및 자세 변화를 포착합니다.

- **Technical Details**: 연구팀은 다중 인코더 모델(multi-encoder model)을 확장하여 객체 및 범주 표현을 동시에 다양한 변형 하에서 별도의 임베딩 공간에서 학습할 수 있도록 했습니다. 또한, 학습 과정을 안내하는 커리큘럼 학습 전략을 제안하며, 이는 학습된 임베딩 공간에서의 객체 간 거리를 사용하여 샘플링 프로세스를 가이드합니다. 이 방법은 학습 과정이 진행됨에 따라 훈련을 더 세분화된 파티션으로 나누고, 유사한 객체를 샘플링하여 모델이 상태 변화로 인해 구별하기 어려운 객체들을 구별할 수 있는 차별적 특징을 포착하도록 돕습니다.

- **Performance Highlights**: 제안된 데이터셋과 커리큘럼 학습 전략은 ObjectsWithStateChange 데이터셋 뿐만 아니라 ModelNet40과 ObjectPI와 같은 다른 도전적인 멀티뷰 데이터셋에서도 객체 수준 작업의 성능 개선을 이끌었습니다. 이를 통해 모델이 변화하는 상태를 가진 객체를 포함하여 보다 미세한 작업에서 구별력 있는 특징을 포착하는 능력이 향상되었다고 평가됩니다.



### RAR-b: Reasoning as Retrieval Benchmark (https://arxiv.org/abs/2404.06347)
- **What's New**: RAR-b (Reasoning as Retrieval Benchmark)는 기존의 언어 이해 모델이 갖고 있는 추론 능력을 평가하기 위해 설계된 새로운 평가 체계입니다. 이 벤치마크는 특히 검색 엔진(retrievers)이 독립적으로 추론 문제를 해결할 수 있는지를 조사하여 LLMs와 함께 사용될 때의 최대 성능을 예측합니다. 이는 언어 모델이 복잡한 사고 과정과 논리적 추론에 대해 어느 정도 능숙한지를 평가하는 것을 목표로 하고 있습니다.

- **Technical Details**: RAR-b는 복수 선택 추론(multiple-choice retrieval)과 전체 데이터셋 검색(full-dataset retrieval) 설정으로 구성된 12개의 추론 작업으로 실험을 설계했습니다. 추론 문제를 검색 문제로 변환하고, 각 검색 모델이 해당 문제에 얼마나 효과적으로 대응하는지를 평가하기 위해 세 가지 주요 분류의 모델을 사용하였습니다: 비지도 밀집 검색 모델(unsupervised dense retrieval models), 지도 밀집 검색 모델(supervised dense retrieval models), 지시어-인식 밀집 검색 모델(instruction-aware dense retrieval models). 또한, 최적의 데이터셋 검색을 위해 재정렬 모델(re-ranking models)의 성능도 평가했습니다.

- **Performance Highlights**: RAR-b 평가 결과, 현재의 상태-최고 기술(state-of-the-art) 검색 모델들은 아직 리즈닝 문제에 효과적으로 대응할만큼 충분히 발달하지 않았음을 보여줍니다. 그러나 최신 디코더 기반 임베딩 모델(decoder-based embedding models)은 이러한 격차를 좁히는 데 큰 가능성을 보여주었습니다. 특히, 재정렬 모델을 활용한 미세조정(fine-tuning)을 통해 모델이 추론 능력을 획득하는 것이 더욱 용이함을 발견했으며, 이러한 방식으로 모든 작업에서 최고의 성능을 달성하였습니다.



### Exploring Diverse Sounds: Identifying Outliers in a Music Corpus (https://arxiv.org/abs/2404.06103)
- **What's New**: 이 연구에서는 음악 추천 시스템에 있어서 종종 소외되는 음악적 아웃라이어(outliers)의 가치를 탐구합니다. '진정한' 음악 아웃라이어(Genuine music outliers)라는 개념을 도입하고, 이를 통해 아티스트의 작업을 더 풍부하게 이해하고, 청취자에게 새롭고 다양한 음악적 경험을 제공할 수 있다고 주장합니다.

- **Technical Details**: 이 논문은 오디오 특성(예: 템포(tempo), 음량(loudness))을 기반으로 진정한 음악 아웃라이어를 식별하고 분류하는 방법을 제안합니다. 고유한 아웃라이어는 아티스트의 주요 스타일과 구별되는 독특한 특성을 보여주며, 이를 통해 음악 발견을 향상시키는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 제안된 아웃라이어 검출 알고리즘은 레이블이 지정된 데이터셋(labelled dataset)에 적용되었으며, 아웃라이어가 음악 발견과 추천 시스템에 미치는 영향을 평가하는 데 사용되었습니다. 연구 결과는 아웃라이어 검출 방법이 음악 추천 시스템을 통한 새로운 음악 경험 제공에 유용할 수 있음을 보여줍니다.



### Event-enhanced Retrieval in Real-time Search (https://arxiv.org/abs/2404.05989)
Comments: LREC-COLING 2024

- **What's New**: 이 논문은 EER(Event-Enhanced Retrieval)이라는 새로운 접근 방식을 제안하여 실시간 검색 시나리오에서 '의미적 표류'(semantic drift) 문제를 해결하고 이벤트 도큐먼트의 검색 성능을 향상시킵니다. 전통적인 EBR (Embedding-Based Retrieval) 듀얼 인코더 모델을 기반으로 하되, 새로운 대조 학습(contrastive learning), 쌍 학습(pairwise learning) 및 프롬프트 조정(prompt-tuning) 기반의 디코더 모듈을 통해 이벤트 중심 정보에 더 집중할 수 있습니다.

- **Technical Details**: EER은 쿼리(query)와 문서 제목(document title)에 대한 듀얼 타워(dual-tower) 모델을 사용합니다. 대조 학습과 쌍 학습을 통해 인코더의 성능을 개선하며, 문서 제목 인코더 뒤에 프롬프트 학습 기반의 디코더 모듈을 추가하여 이벤트 트리플릿을 생성하는 방식으로 중요 이벤트 정보에 집중하도록 설계되었습니다. 이 디코더는 학습 단계에서만 사용되며, 추론 단계에서는 제거할 수 있어 전통적인 듀얼 타워 모델로 복귀하여 지연 시간에 영향을 주지 않습니다.

- **Performance Highlights**: EER은 방대한 실험을 통해 실시간 검색 시나리오에서의 검색 성능이 크게 향상됨을 보여주었습니다. 이 방법은 쿼리와 문서 제목 간의 정보 비대칭성을 해결하고, 이벤트 정보에 집중하여 쿼리와 관련된 이벤트를 효과적으로 검색할 수 있도록 도와줍니다. 또한, EER은 인코딩된 이벤트 정보를 활용하여 쿼리 인코더 최적화를 통한 검색 결과의 정확성을 높이는 데 기여합니다.



### Optimization Methods for Personalizing Large Language Models through  Retrieval Augmentation (https://arxiv.org/abs/2404.05970)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)을 개인화하는 데 검색 증강(retrieval-augmented) 접근 방식을 사용하여 다양한 응용 프로그램 및 도메인에 상당한 영향을 미칠 수 있는 가능성을 탐구하고 있습니다. 처음으로 개인 문서를 LLM에 제공하는 검색 모델을 최적화하는 방법을 제안합니다. 우리는 두 가지 최적화 알고리즘을 개발했으며, 하나는 개인화 생성(personalized generation)을 위한 임의의 척도를 사용하여 정의된 보상 함수를 기반으로 하는 강화 학습(reinforcement learning)에, 또 다른 하나는 하위 LLM에서 검색 모델로 지식을 전수하는 지식 전수(knowledge distillation)에 기반을 두고 있습니다.

- **Technical Details**: 이 논문은 개인화된 텍스트 생성을 위해 개인 정보를 검색하는 최적화에 초점을 맞추고 있으며, 기존의 학습-순위 최적화 방법(learning-to-rank optimization methods)을 적용할 수 없는 상황에서 두 가지 새로운 접근 방식을 제안합니다. 첫 번째는 강화 학습 방법을 사용하여 사용자 프로필에서 문서를 샘플링하고 이를 LLM에 제공한 다음, 생성된 개인화된 출력 텍스트의 성능을 기반으로 검색 모델을 최적화합니다. 두 번째 방법은 LLM에서 생성된 개별 검색 문서의 성능을 사용하여 타겟 분포를 계산하고 검색 점수 분포와의 차이(divergence)를 최소화하여 검색 모델을 최적화합니다.

- **Performance Highlights**: 이 논문에서 제안된 방법은 LaMP 벤치마크를 사용하여 평가되었으며, 여기에는 세 가지 개인화된 텍스트 분류 작업과 네 가지 개인화된 텍스트 생성 작업이 포함됩니다. 제안된 방법들은 LaMP의 일곱 개 데이터 세트 중 여섯에서 통계적으로 유의미한 개선을 달성했습니다. 최고 성능 방법은 LaMP 데이터 세트 전반에 걸쳐 평균 5.5%의 최신 기술(state-of-the-art) 개선을 보여주었으며, 비개인화된 LLM과 비교할 때 모든 작업에서 평균 15.3%의 개선을 보였습니다.



### Use of a Structured Knowledge Base Enhances Metadata Curation by Large  Language Models (https://arxiv.org/abs/2404.05893)
- **What's New**: 이 논문은 큰 언어 모델(Large Language Models, LLMS)을 활용하여 메타데이터 기준 준수를 개선할 수 있는 가능성을 탐구합니다. 특히, GPT-4를 사용하여 NCBI BioSample 레포지토리에서 무작위로 선택한 인간 샘플에 대한 데이터 레코드 200개를 대상으로 실험을 수행했습니다.

- **Technical Details**: 연구에서는 GPT-4가 메타데이터 기준 준수를 향상시키기 위한 편집 제안을 하는 능력을 평가했습니다. 피어 리뷰(peer review) 과정을 통해 필드 이름-필드 값 쌍(field name-field value pairs)의 준수 정확도를 계산했으며, 기본 데이터 사전(standard data dictionary)에 대한 평균 준수율이 79%에서 80%로 소폭 개선된 것을 확인했습니다 (p<0.01). 또한, GPT-4에게 CEDAR 템플릿의 텍스트 설명(textual descriptions)을 도메인 정보(domain information) 형태로 제시한 후에는 준수율이 79%에서 97%로 크게 향상됐다는 결과를 얻었습니다 (p<0.01).

- **Performance Highlights**: LLMs가 독립적으로 사용될 때 기존 메타데이터의 표준 준수를 보장하는 데 한계가 있지만, 구조화된 지식 베이스(knowledge base)와 통합될 때 자동 메타데이터 관리(automated metadata curation)에 유망할 수 있다는 점을 드러냈습니다. 도메인 정보를 제공했을 때 준수율이 크게 향상된 것은 이러한 통합이 얼마나 중요한지를 시사합니다.



### MealRec$^+$: A Meal Recommendation Dataset with Meal-Course Affiliation  for Personalization and Healthiness (https://arxiv.org/abs/2404.05386)
Comments: Accepted by SIGIR 2024

- **What's New**: 신규 데이터셋인 MealRec+를 소개합니다. 이 데이터셋은 식사-코스 연계성(meal-course affiliation)을 포함하며, 개인 맞춤형 및 건강한 식사 추천을 위한 협력적 상호작용 학습(cooperative interaction learning)을 연구할 수 있는 기회를 제공합니다. 이는 기존의 식사 추천 연구에서 부족했던 부분을 보완합니다.

- **Technical Details**: MealRec+ 데이터셋은 사용자-코스 상호작용 데이터에서 시뮬레이션된 식사 세션을 기반으로 식사-코스 연계성과 사용자-식사 상호작용을 파생시킵니다. 추가적으로, 세계 보건 기구(World Health Organization)와 영국 식품 기준청(UK Food Standards Agency)의 영양 표준을 사용하여 식사의 건강성 점수를 계산합니다. 이러한 기술적인 특징은 식사 추천의 정확성과 건강성을 높이는 데 기여합니다.

- **Performance Highlights**: 실험을 통해 두 수준의 상호작용을 적절하게 협력하면 식사 추천에 유익함을 보여줍니다. 또한, 식사 추천에서 나타난 건강하지 않은 추천 현상에 대응하여 추천 결과의 건강성을 개선하기 위한 방법을 탐구합니다. 이는 식단 기반 공중 보건 증진에 유용한 인사이트를 제공합니다.



### Beyond the Sequence: Statistics-Driven Pre-training for Stabilizing  Sequential Recommendation Mod (https://arxiv.org/abs/2404.05342)
- **What's New**: 이 연구에서는 시퀀셜 추천 시스템을 위한 새로운 접근 방식인 통계기반 사전 학습(StatisTics-Driven Pre-training, STDP) 프레임워크를 제안하였다. 이 프레임워크는 사용자의 역사적 행동 시퀀스로부터 불안정한 데이터의 영향을 줄이고 안정적인 추천 모델 최적화를 촉진한다.

- **Technical Details**: STDP는 통계 정보와 사전 학습 패러다임을 결합하여 추천 모델의 최적화를 안정화한다. 구체적으로, 항목의 동시 발생(co-occurrence), 상호작용 빈도수(attribute frequency) 등의 통계적 정보를 활용하여 다음과 같은 세 가지 사전 학습 과제를 설계했다: 1) Co-occurred Items Prediction (CIP): 모델이 다음 항목을 예측하고 가장 상위 동시 발생 항목들을 추천하도록 유도한다. 2) Paired Sequence Similarity (PSS): 원본 시퀀스의 항목 일부를 동시 발생 항목으로 대체하여 쌍을 형성한 시퀀스를 생성하고 두 시퀀스의 표현 유사성을 최대화한다. 3) Frequent Attribute Prediction (FAP): 시퀀스의 빈번한 속성을 예측하도록 모델을 훈련하여 사용자의 장기 취향을 안정적으로 포착한다.

- **Performance Highlights**: STDP 프레임워크는 여섯 개의 데이터셋에서 상당한 성능 향상을 달성함으로써 그 효과와 우수성을 입증하였다. 또한, 이 프레임워크는 다른 추천 모델에 대한 일반화 가능성을 추가 분석을 통해 확인하였다.



### Joint Identifiability of Cross-Domain Recommendation via Hierarchical  Subspace Disentanglemen (https://arxiv.org/abs/2404.04481)
Comments: accepted to SIGIR 2024 as a Full Research Paper

- **What's New**: 이 연구에서는 교차 도메인 추천(Cross-Domain Recommendation, CDR)에서 도메인 간에 전달되어야 할 것과 전달되지 말아야 할 것을 인과 관계(causality)의 관점에서 탐구합니다. 저자들은 도메인별 행동(domain-specific behaviors)을 도메인 공유 요소(domain-shared factors)로부터 보존하기 위해 HJID(Hierarchical Joint IDentifiability) 접근 방식을 제안합니다.

- **Technical Details**: HJID는 사용자 표현(user representations)을 일반적인 얕은 부분공간(generic shallow subspaces)과 도메인 지향적인 깊은 부분공간(domain-oriented deep subspaces)으로 구성합니다. 첫 번째로, 초기 레이어 활성화에서 최대 평균 차이(Maximum Mean Discrepancy)를 최소화하여 얕은 부분공간에서 일반적인 패턴을 인코딩합니다. 그 다음, 도메인 지향적인 잠재 요인이 깊은 부분공간 활성화에서 어떻게 인코딩되는지 분석하기 위해 교차 도메인 인과 관계 기반 데이터 생성 그래프(cross-domain causality-based data generation graph)를 구성합니다. 이는 최소 변화 원칙(Minimal Change principle)을 준수하며, 교차 도메인 일관성(cross-domain consistent) 및 도메인별 구성 요소를 식별합니다.

- **Performance Highlights**: 실제 데이터셋(real-world datasets)에서의 실험을 통해 HJID가 강하게 및 약하게 상관된 CDR 작업에서 최신 기술(state-of-the-art, SOTA) 방법들을 능가함을 보여줍니다. 이는 HJID가 도메인별 유니크한 요인을 발견하면서 안정성을 유지할 수 있음을 증명하며, 도메인 간 일관된 행동 경향과 도메인 특이적 패턴을 모두 포함하는 생성적 프레임워크를 통해 이루어집니다.



### JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced  Transformer (https://arxiv.org/abs/2404.04313)
- **What's New**: 이 논문에서는 새로운 기술인 'Skill-aware Recommendation Model'과 'Semantic-enhanced Transformer'를 소개하며, 이를 통해 개인의 기술 분포를 반영한 맞춤형 직업 추천을 가능하게 합니다. 이 모델은 직업 설명(JD)과 사용자 프로필 사이의 이질적인 간격을 해소하고, 사용자의 클릭율(Click-Through Rate, CTR) 예측을 위한 효과적인 JD-사용자 공동 표현을 제공합니다.

- **Technical Details**: 이 연구는 먼저 각 JD의 상대적 항목을 모델링하고, 지역-글로벌 주의 메커니즘(Local-global Attention Mechanism)이 있는 인코더를 사용하여 JD 튜플로부터 직무 간 및 직무 내 의존성을 더 잘 파악합니다. 따라서, 'Skill Distribution'을 이용하여 리콜 단계에서 JD 표현 학습을 안내하고, 랭킹 단계에서 사용자 프로파일과 결합하여 최종적으로 클릭율을 예측합니다. 또한, Transformer 기반 모델을 사용하여 JD의 시맨틱(Semantic) 표현력을 강화하는 'Semantic-enhanced Transformer' 기술도 개발되었습니다.

- **Performance Highlights**: JobFormer 모델은 실제 세계 데이터셋과 공개 데이터셋에서의 경험적 분석을 통해 그 우수성을 입증하였습니다. 이 모델은 최신 기술(State-of-the-Art) 결과와 더 나은 해석 가능성을 제공함으로써 직업 추천 분야에서의 성능을 향상시켰습니다.



### Selecting Query-bag as Pseudo Relevance Feedback for Information-seeking  Conversations (https://arxiv.org/abs/2404.04272)
- **What's New**: 이 연구에서는 정보 탐색 대화 시스템을 위한 새로운 프레임워크인 Query-bag based Pseudo Relevance Feedback (QB-PRF)를 제안합니다. QB-PRF는 관련 질문들을 포함하는 query-bag을 구성하여 대화 시스템의 이해도와 응답의 질을 향상시키려고 합니다.

- **Technical Details**: QB-PRF는 두 가지 주요 모듈로 구성됩니다: 1) Query-bag Selection (QBS) 모듈은 대화에 사용될 유사 질의를 선택하는 역할을 하며, 미리 학습된 VAE을 이용하여 비지도학습 방식으로 동의어 질의들을 선택합니다. 2) Query-bag Fusion (QBF) 모듈은 선택된 질의들을 원래의 질의와 통합하여 의미적 표현을 강화합니다. 이 과정은 다차원적 주의(attention) 계산을 사용하여 수행됩니다.

- **Performance Highlights**: 실험 결과는 QB-PRF가 기존의 강력한 기반 모델인 BERT와 GPT-2를 사용하여 벤치마크 데이터셋에서 뛰어난 성능을 보였습니다. 특히, 정보 탐색 시스템의 질문 이해 및 대답 매칭 정확도가 크게 향상되었습니다.



### Towards Effective Next POI Prediction: Spatial and Semantic Augmentation  with Remote Sensing Data (https://arxiv.org/abs/2404.04271)
Comments: 12 pages, 11 figures, Accepted by ICDE 2024

- **What's New**: 이번 연구는 다음 관심 지점(POI, Point-Of-Interest) 예측을 위한 새로운 접근법인 TSPN-RA(Two-Step Prediction Network with Remote Sensing Augmentation)를 소개합니다. 이 모델은 원격 감지 데이터를 통합하고, 쿼드트리 구조(quad-tree structure)를 사용하여 도시 공간의 그래프 표현 방법을 개선함으로써 위치 기반 서비스에서의 예측 정확도를 향상시키고자 합니다. 특히, TSPN-RA는 공간적 의도와 의미적 의도를 연결하는 두 단계 예측 프레임워크를 사용합니다.

- **Technical Details**: TSPN-RA는 원격 감지 이미지를 이용하여 주요 환경 맥락을 통합하고, 이를 통해 POI 주변 지역의 이해를 향상시킵니다. 또한, 쿼드트리 구조를 활용하여 POI의 비균등 분포를 처리하고, 이를 QR-P 그래프라는 새로운 유형의 공간 지식 그래프로 구체화합니다. 그래프 신경망(Graph Neural Network)과 주의 기법(Attention Mechanism)을 적용하여 사용자의 역사적 궤적과 현재 궤적에서 얻은 타일 및 POI 특성을 인코딩하고 통합합니다. 첫 번째 단계에서는 사용자가 선호할 가능성이 있는 타일을 예측하고, 두 번째 단계에서는 해당 타일 내에서 구체적인 POI를 예측합니다.

- **Performance Highlights**: 실제 위치 기반 소셜 네트워크 데이터셋 네 개를 사용한 실험 결과, TSPN-RA는 기존의 경쟁 모델들 대비 효과성과 효율성 면에서 우수한 성능을 보였습니다. 이 모델은 환경적 요인과 지리적 제약을 고려함으로써, 사용자의 실제 방문 의도를 더욱 정확하게 예측할 수 있는 기능을 제공합니다.



### Accelerating Recommender Model Training by Dynamically Skipping Stale  Embeddings (https://arxiv.org/abs/2404.04270)
- **What's New**: 이 논문은 Slipstream, 새로운 소프트웨어 프레임워크를 제안합니다. 이는 추천 모델에서 'stale embeddings'를 식별하고 업데이트를 건너뛰는 기능을 도입하여 트레이닝 성능을 향상시킵니다. 이 접근 방식은 CPU-GPU 대역폭 사용을 최적화하고 불필요한 메모리 접근을 제거합니다.

- **Technical Details**: Slipstream은 'Snapshot Block', 'Sampling Block', 그리고 'Input Classifier Block' 세 가지 주요 구성 요소를 포함합니다. 각각 빠르게 트레이닝되는 'hot' embeddings를 식별하고, 최적의 임계값을 결정하여 stable embeddings를 판별하며, 해당 embeddings의 입력을 건너뛰어 전반적인 효율성을 높입니다. 이러한 구성을 통해, Slipstream은 트레이닝 시간을 크게 줄이면서도 정확도를 유지합니다.

- **Performance Highlights**: Slipstream은 기존의 Baseline XDL, Intel-optimized DLRM, FAE, 그리고 Hotline 모델들에 비하여 각각 평균 2배, 2.4배, 20%, 18%의 트레이닝 시간 단축을 실현했습니다. 이는 큰 데이터셋에서의 메모리와 시간 효율을 크게 개선한 결과입니다.



### Algorithmic Collective Action in Recommender Systems: Promoting Songs by  Reordering Playlists (https://arxiv.org/abs/2404.04269)
- **What's New**: 본 연구에서는 추천 시스템(recommender systems) 및 알고리즘 집단 행동(algorithmic collective action)을 조사하며, 이는 팬들이 알고리즘의 학습 데이터에 영향을 미쳐 아티스트의 가시성을 증대시키는 새로운 전략을 모색합니다. 특히, 변형기 기반(Transformer-based) 추천 시스템을 사용하여, 소수의 사용자 집단이 전략적으로 플레이리스트에 특정 곡을 배치함으로써 그 곡의 추천 빈도를 상당히 증가시킬 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 자동 재생목록 계속(Automatic Playlist Continuation, APC)이 주요 작업으로 사용됩니다. 연구자들은 음악 스트리밍 서비스인 Deezer에서 사용된 APC 모델을 활용하여, 소규모의 집단이 통제하는 훈련 데이터의 일부에서 곡을 전략적으로 배치하는 두 가지 전략을 적용했습니다. 이들 전략은 특정한 노래 순서와 관련하여 곡을 추천하도록 모델의 주의 함수(attention function)를 활용합니다.

- **Performance Highlights**: 실험 결과, 집단이 특정 곡을 플레이리스트에 삽입함으로써 그 곡의 추천 빈도는 기존 대비 최대 25배까지 증가할 수 있음을 확인하였습니다. 또한, 시스템 전체의 추천 성능에 미치는 영향은 미미하며, 다른 아티스트들에게 불공정한 영향을 미치지 않았습니다. 사용자 경험에 대한 부정적인 영향도 거의 없었으며, 추천 시스템의 공정성 개선에 기여할 수 있는 가능성을 보여줍니다.



### The Use of Generative Search Engines for Knowledge Work and Complex  Tasks (https://arxiv.org/abs/2404.04268)
Comments: 32 pages, 3 figures, 4 tables

- **What's New**: 이 연구에서는 기존 검색 엔진과 대조적으로 최근 등장한 '생성적 검색 엔진'의 특성과 사용례를 분석합니다. Bing Copilot (Bing Chat)을 사례로 사용하여, 전통적인 검색 엔진에 LLMs(Large Language Models)의 기능을 결합한 새로운 도구인 생성적 검색 엔진이 사용자들에게 어떻게 활용되고 있는지를 탐구했습니다.

- **Technical Details**: 이 연구는 Bing Copilot, 즉 Bing의 생성적 검색 엔진을 사용하여 데이터를 수집하고 분석하였습니다. LLMs를 활용하여 텍스트, 이미지, 코드 등의 새로운 디지털 아티팩트(digital artifacts)를 생성하는 기능을 함께 제공함으로써, 검색 엔진의 기능을 확장합니다.

- **Performance Highlights**: 분석 결과, 생성적 검색 엔진은 전통적 검색 엔진에 비해 더 높은 인지 복잡성(cognitive complexity)을 가진 지식 작업(knowledge work tasks)에 자주 사용되는 것으로 나타났습니다. 이는 생성적 검색 엔진이 전통적인 검색 방식을 넘어서는 새로운 가능성을 제시하고 있음을 시사합니다.



### Accelerating Matrix Factorization by Dynamic Pruning for Fast  Recommendation (https://arxiv.org/abs/2404.04265)
- **What's New**: 이 논문에서는 추천 시스템(Recommendation Systems, RSs)에서 널리 사용되는 협업 필터링(Collaborative Filtering, CF) 알고리즘인 행렬 분해(Matrix Factorization, MF)의 훈련 속도를 개선하는 새로운 알고리즘 방법을 제안합니다. 추가적인 계산 자원을 사용하지 않고 MF의 계산 복잡성을 줄이는 방법이 소개되었으며, 이는 통상적으로 많은 자원이 소모되는 기존 방법과 대비됩니다.

- **Technical Details**: 저자들은 특정 임계값을 고려할 때 분해된 특징 행렬에서 세부적인 구조적 희소성(Fine-grained structured sparsity)을 관찰합니다. 기존의 MF 훈련 과정은 불필요한 연산을 많이 포함하는데, 이를 최적화하기 위해 특징 행렬을 공동 희소성(Joint sparsity)을 기반으로 재배열하여, 인덱스가 작은 잠재 벡터(Latent vector)가 인덱스가 높은 것보다 더 밀집되도록 합니다. 이후, 중요하지 않은 잠재 요인을 사전에 제거하는 푸루닝(Pruning) 과정을 도입해, 행렬 곱셈과 잠재 요인 업데이트 도중 일찍 중단하는 방식으로 속도를 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 실험을 통해 전통적인 MF 훈련 과정에 비해 1.2-1.65배의 속도 향상을 보였으며, 최대 20.08%의 오류 증가를 기록했습니다. 또한, 최적화자(Optimizer), 최적화 전략(Optimization strategy), 초기화 방법(Initialization method) 등 다양한 하이퍼파라미터를 고려한 적용 가능성을 입증했습니다.



### Logic Query of Thoughts: Guiding Large Language Models to Answer Complex  Logic Queries with Knowledge Graphs (https://arxiv.org/abs/2404.04264)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)과 지식 그래프(Knowledge Graph, KG) 기반 질의 응답 기술을 결합한 새로운 접근 방식인 'Logic-Query-of-Thoughts' (LGOT)를 제안합니다. LGOT는 복잡한 논리 질의를 쉽게 답변할 수 있는 하위 질문들로 나누어 처리하는 방식으로, 지식 그래프 추론과 LLMs를 효과적으로 통합하여 사용합니다.

- **Technical Details**: LGOT은 복합 논리 추론을 필요로 하는 질의에 대해 지식 그래프 추론과 LLMs의 장점을 활용하여 각 하위 질문에 대한 답변을 도출하고, 이러한 결과들을 종합하여 최고 품질의 후보 답변을 선정합니다. 본 연구에서는 지식 그래프의 불완전성 문제와 LLMs의 환각 문제를 동시에 해결하는 새로운 통합 방식을 소개합니다.

- **Performance Highlights**: 이 방법은 기존의 ChatGPT와 비교하여 최대 20%의 성능 향상을 보여주며, 특히 지식 그래프가 희박하고 불완전할 때 정확도가 현저하게 감소하는 문제를 개선할 수 있음을 증명합니다. LGOT는 복잡한 논리 질의에 대한 정확한 결과를 제공하는 것으로 나타났습니다.



### HaVTR: Improving Video-Text Retrieval Through Augmentation Using Large  Foundation Models (https://arxiv.org/abs/2404.05083)
- **What's New**: 본 논문에서는 비디오-텍스트 검색(VTR)의 대표적인 한계인 고품질 훈련 데이터 부족 문제를 해결하기 위해 새로운 학습 패러다임인 HaVTR을 제안합니다. HaVTR은 비디오와 텍스트 데이터를 증강하여 더 일반화된 특징을 학습할 수 있게 합니다. 이를 통해 비디오-텍스트 매칭의 효과를 극대화하는 것이 주요 목표입니다.

- **Technical Details**: HaVTR은 세 가지 주요 증강 방법을 통해 데이터를 향상합니다. 첫 번째는 간단한 증강 방법으로, 프레임이나 서브워드(subwords)의 무작위 중복이나 삭제를 통해 데이터를 생성합니다. 두 번째는 대규모 언어 모델(LLMs)과 시각 생성 모델(VGMs)을 사용한 텍스트 패러프레이징(text paraphrasing) 및 비디오 스타일 변환(video stylization) 기법입니다. 마지막으로, 증강된 정보를 활용하여 비디오와 텍스트에 새로운 관련 정보를 생성하고 추가하는 환상 기반(hallucination-based) 증강 방법이 포함됩니다.

- **Performance Highlights**: HaVTR은 비디오-텍스트 검색 벤치마크인 MSR-VTT, MSVD, ActivityNet에서 기존 방법들을 능가하는 성능을 보여주었습니다. 특히, MSR-VTT에서 텍스트-비디오 검색의 Recall @1이 46.1에서 50.8로 향상되었음을 실험을 통해 확인하였습니다.



### Weakly Supervised Deep Hyperspherical Quantization for Image Retrieva (https://arxiv.org/abs/2404.04998)
Comments: In proceedings of AAAI 2021. Code and data are available

- **What's New**: 본 논문은 약한(weak) 태그를 사용하여 심층(deep) 양자화(quantization)를 학습하는 최초의 작업, Weakly-Supervised Deep Hyperspherical Quantization (WSDHQ)을 제안합니다. 기존 방법들에 비해, 이 연구는 인터넷이나 소셜 미디어로부터 쉽게 얻을 수 있는, 정식으로 정의되지 않은 태그들을 이용하여 양자화 모델을 학습하고자 합니다.

- **Technical Details**: 이 방법은 형태소 임베딩(word embeddings)을 사용하여 태그들을 표현하고, 태그 상관 관계 그래프를 기반으로 그들의 의미 정보(semantic information)를 강화합니다. 또한, 하이퍼스피어(hypersphere)에서 의미 보존 임베딩(semantics-preserving embeddings)과 감독된 양자화기(supervised quantizer)를 공동으로 학습하는 방식을 채택하였고, 잘 설계된 융합 레이어(fusion layer)와 맞춤형 손실 함수(tailor-made loss functions)를 사용합니다.

- **Performance Highlights**: WSDHQ는 약하게 감독된(compact coding under weak supervision) 환경에서 최고의 성능을 달성하고 있습니다. 단일 레이블(label)이나 다중 클래스(multi-class) 설정에서 모두 효과적임이 확인되었으며, 상당한 양자화 오류(quantization error) 감소와 함께 의미 정보의 더 나은 보존을 실현합니다.



### Balancing Information Perception with Yin-Yang: Agent-Based Information  Neutrality Model for Recommendation Systems (https://arxiv.org/abs/2404.04906)
- **What's New**: 이 논문은 AbIN (Agent-based Information Neutrality) 모델을 소개하며, 이는 기존의 선호도 기반 추천 시스템에 정보의 중립성을 추가하여 사용자의 정보 소비를 다양화하는 새로운 접근 방식을 제안합니다. 중국의 음양 이론(Yin-Yang theory)을 기반으로 하여 정보의 균형을 맞추고 필터 버블(filter bubbles)의 부정적 영향을 완화합니다.

- **Technical Details**: AbIN 모델은 세 가지 독립적인 에이전트(Organical Preference-based Agent(OPA), User Agent(UA), Information Neutralization Agent(INA))를 활용하여 구축되었습니다. INA는 음양 중립화 제어(Yin-Yang Neutralization Control, YYNC) 방법을 적용하여 OPA의 추천 목록을 조정함으로써 정보의 균형을 맞춥니다. 이는 기존 추천 알고리즘을 수정하지 않고도 정보 소비의 다양성을 확장할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 AbIN 모델이 추천 내용의 균형을 유지하면서 필터 버블의 영향을 감소시키고, 사용자가 소비하는 정보에 대한 더 정확한 이해를 제공하는 데 효과적임을 입증하였습니다. 이 모델은 사용자의 선호를 존중하는 동시에 정보 다양성을 증가시키는 것으로 나타났습니다.



### Single-Server Pliable Private Information Retrieval with Identifiable  Side Information (https://arxiv.org/abs/2404.04820)
Comments: 10 pages and 3 figures

- **What's New**: 본 논문은 식별 가능한 부가 정보를 가진 단일 서버에 대한 유연한 개인 정보 검색(PPIR) 문제, 즉 PPIR with Identifiable Side Information (PPIR-ISI)에 초점을 맞추고 있습니다. 이전 연구들과 달리, 본 논문에서는 부가 정보가 식별 가능할 때 통신 비용을 줄일 수 있는 새로운 방안을 제안하고 있습니다. 또한, 다중 사용자 경우에 대해서도 이 문제를 확장하여 질의 생성을 협력적으로 할 수 있는 새로운 체계를 제시합니다.

- **Technical Details**: 이 연구는 최대 거리 분리 가능(MDS) 코드를 사용하여 메시지를 인코딩하는 새로운 PPIR-ISI 체계를 제안합니다. PPIR 문제는 사용자가 서버로부터 특정 클래스의 메시지를 검색하고자 할 때, 원하는 클래스의 신원 또는 그들이 가진 부가 정보를 서버에 노출시키지 않아야 합니다. 특히, 식별 가능한 부가 정보(PPIR-ISI)를 활용하는 경우, 본 연구는 이러한 정보를 통해 통신 비용을 줄일 수 있는 방법을 제시합니다. 또한, 이 체계는 다중 사용자 설정에서도 활용될 수 있으며, 사용자들이 협력하여 보다 효율적인 질의를 생성할 수 있습니다.

- **Performance Highlights**: 식별 가능한 부가 정보를 사용함으로써, 통신 비용을 줄이는 것이 가능함을 증명하였으며, 특정 상황에서 이전 PPIR-USI 체계보다 우수한 성능을 보여주는 것으로 분석되었습니다. 또한, 이 새로운 PPIR-ISI 체계는 다중 사용자 시나리오에서 사용자 협동을 통한 질의 생성이 가능함으로써, 더욱 효율적인 데이터 검색이 가능하다는 것이 강조되었습니다.



### Music Recommendation Based on Facial Emotion Recognition (https://arxiv.org/abs/2404.04654)
- **What's New**: 이 논문에서는 음악 추천, 감정 인식, 그리고 설명 가능한 인공지능(XAI: Explainable AI)을 통합하여 사용자 경험을 향상시키는 방안을 제안합니다. 특히, GRAD-CAM을 사용하여 모델의 예측에 대한 설명을 제공함으로써 사용자가 시스템의 추천 이유를 이해할 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 감정 분류를 위해 ResNet50 모델을 사용하여 Facial Expression Recognition (FER) 데이터셋에서 훈련됩니다. 이 데이터셋은 다양한 감정을 표현하는 개인의 실제 이미지로 구성됩니다. 모델 학습은 입력 이미지의 전처리(pre-processing), 컨볼루션 레이어를 통한 특징 추출(feature extraction), 밀집 레이어(dense layers)를 통한 추론, 출력 레이어를 통한 감정 예측으로 이루어집니다.

- **Performance Highlights**: 이 시스템은 감정 분류에서 82%의 정확도(accuracy)를 달성합니다. GRAD-CAM을 활용함으로써, 모델은 그 예측에 대한 해석 가능한 피드백을 제공하여 사용자가 추천의 이면을 이해할 수 있게 합니다.



### Q-PEFT: Query-dependent Parameter Efficient Fine-tuning for Text  Reranking with Large Language Models (https://arxiv.org/abs/2404.04522)
- **What's New**: 이 논문에서는 문서 재정렬(reranking)을 위한 새로운 쿼리 의존적 매개변수 효율적 미세조정(Query-Dependent Parameter Efficient Fine-Tuning, Q-PEFT) 방법을 제안합니다. 기존의 매개변수 효율적 미세조정(PEFT) 방식과 달리, Q-PEFT는 쿼리 정보를 활용하여 문서에 특화된 합성 쿼리를 생성하는 데 초점을 맞추고 있습니다. 이 접근 방식은 매개변수의 수를 최소화하면서 기존 대규모 언어 모델(LLM)의 재정렬 기능을 향상시킵니다.

- **Technical Details**: 연구진은 Q-PEFT를 두 가지 방식으로 구현해냈습니다: Q-PEFT-R과 Q-PEFT-A. Q-PEFT-R은 문서에서 주요 k개 단어를 활용하여 쿼리 종속적 내용을 생성하고, Q-PEFT-A는 멀티 헤드 주의력(multi-head attention) 계층을 사용하여 문서와 쿼리 간의 상관 관계를 무게를 재조정합니다. 이는 전체 LLM을 미세조정하지 않고도 특정 IR(Infromation Retrieval) 작업에 맞게 LLM의 성능을 최적화하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 Q-PEFT 방법은 네 개의 공개 데이터셋에서 폭넓은 실험을 거쳤으며, 기존의 하드 프롬프팅(hard prompting) 방법보다 뛰어난 성능을 보였습니다. 이는 Q-PEFT가 문서-쿼리 관련성 점수에 긍정적인 영향을 미침으로써, 쿼리에 따라 더욱 특화된 결과를 생성할 수 있음을 시사합니다. 코드와 체크포인트는 논문이 수락되는 대로 공개될 예정입니다.



### Towards Realistic Few-Shot Relation Extraction: A New Meta Dataset and  Evaluation (https://arxiv.org/abs/2404.04445)
- **What's New**: 새로운 메타 데이터셋이 소개되었습니다. 이 데이터셋은 기존의 관계 추출 데이터셋인 NYT29와 WIKIDATA에서 파생된 두 데이터셋과 TACRED 데이터셋의 Few-shot(퓨샷) 형태를 포함하고 있습니다. 중요하게도, 이들 Few-shot 데이터셋은 현실적인 가정 하에 생성되었으며, 테스트 관계는 모델이 이전에 본 적이 없는 관계, 제한된 훈련 데이터, 그리고 관심 있는 관계에 해당하지 않는 후보 관계 언급의 우세 등의 특징을 가지고 있습니다.

- **Technical Details**: 이 연구는 최근의 여섯 가지 Few-shot 관계 추출 방법을 종합적으로 평가합니다. 각 방법의 성능은 혼합적으로 나타나며, 명확한 우세자는 없습니다. 또한, 이 작업에 대한 전체적인 성능이 낮아 향후 연구에 대한 상당한 필요성을 시사합니다.

- **Performance Highlights**: 데이터셋의 전체적인 성능이 낮으며, 이는 Few-shot 관계 추출 작업이 아직 많은 개선이 필요함을 의미합니다. 하지만 아직 명확한 성공을 보여주는 방법이 없기 때문에, 이 분야는 예측 불가능하고 도전적인 연구 분야로 남아 있습니다.



### Dwell in the Beginning: How Language Models Embed Long Documents for  Dense Retrieva (https://arxiv.org/abs/2404.04163)
- **What's New**: 이 연구는 웹 문서 검색 맥락에서 텍스트 표현 학습을 위한 Transformer 기반 모델에서의 위치적 편향의 존재를 조사합니다. 이전 연구에서 인과적 언어 모델들의 입력 시퀀스 중간에서 정보 손실이 발생함을 보여준 것을 기반으로, 표현 학습 분야로 연구를 확장하였습니다.

- **Technical Details**: 연구는 인코더-디코더(encoder-decoder) 모델의 다양한 학습 단계에서 위치적 편향을 조사합니다. 이에는 언어 모델 사전 학습(language model pre-training), 대조적 사전 학습(contrastive pre-training), 그리고 대조적 미세 조정(contrastive fine-tuning)이 포함됩니다. MS-MARCO 문서 컬렉션을 사용한 실험을 통해 모델은 대조적 사전 학습 후 입력의 초기 내용을 더 잘 포착하는 임베딩을 생성하며, 미세 조정은 이 효과를 더욱 심화시킵니다.

- **Performance Highlights**: 대조적 사전 학습 후 모델은 입력의 초기 내용을 더 잘 이해할 수 있는 임베딩을 생성했으며, 고급 조정(fine-tuning)은 이러한 편향을 더욱 강화한 결과를 나타냈습니다.



### A Comparison of Methods for Evaluating Generative IR (https://arxiv.org/abs/2404.04044)
- **What's New**: 생성적 정보 검색(Generative Information Retrieval, Gen-IR)이 정보 검색(IR) 시스템에서 주요 패러다임으로 떠오르고 있습니다. 이 연구는 고정된 문서 세트로부터 응답을 생성하지 않고, 질의에 대해 완전히 새로운 텍스트를 생성하는 시스템에 초점을 맞춥니다. 특히, 검색된 결과를 요약하고 확장하는 대규모 언어 모델(LLM)을 기반으로 하는 생성적 컴포넌트(retreival augmented generation, RAG) 시스템을 포함하여, 기존의 오프라인 IR 평가 방식을 생성적 IR 컨텍스트로 확장하는 다양한 방법을 탐구합니다.

- **Technical Details**: Gen-IR 시스템 평가를 위해 이 논문은 이진 및 등급 적합성, 명시적 하위 주제, 쌍대 선호도, 임베딩을 기반으로 한 방법을 포함한 여러 평가 방법을 제안합니다. 대규모 언어 모델을 이용한 라벨링이 인간 평가자에 의한 전통적인 방법을 대체할 가능성이 높으며, 이는 자동화된 평가와 인간 평가의 감사가 가능하도록 하여 시스템의 신뢰성을 보장합니다. 평가 기준은 투명해야 하며, 사람 평가자가 감사할 수 있도록 설정되어야 합니다.

- **Performance Highlights**: 이 연구는 TREC Deep Learning Track 작업에 대한 인간 평가와 여러 생성적 시스템의 출력을 평가하기 위해 제안된 방법들을 검증합니다. 이 결과는 인간의 평가와 일치하는 경향을 보여주며, 특히 LLM 기반 라벨링은 비용과 정확성 면에서 인간 라벨러보다 우수하거나 유사한 성능을 보여 주었습니다. 이를 통해 전통적인 테스트 컬렉션을 효과적으로 대체할 수 있는 새로운 평가 프로세스의 가능성을 시사합니다.



### Understanding Language Modeling Paradigm Adaptations in Recommender  Systems: Lessons Learned and Open Challenges (https://arxiv.org/abs/2404.03788)
Comments: Tutorial held at the 27th European Conference on Artificial Intelligence (ECAI) in Santiago de Compostela, Spain, on October 19-24, 2024

- **What's New**: 이 튜토리얼은 LLM(Language Large Models, 대규모 언어 모델)을 추천 시스템(recommender systems, RS)에 적용하는 새로운 방법론을 소개합니다. LLM의 'pre-train, prompt and predict'와 같은 훈련 패러다임은 추천 시스템의 효능을 향상시키기 위해 적용되며, 이는 제한된 라벨 데이터로 일반화 가능한 모델을 학습하는데 유리합니다. 이 튜토리얼은 LLM 기반의 RS에서 발생할 수 있는 윤리적 고려사항과 잠재적 피해를 분석하고 이를 완화하는 방법을 탐구합니다.

- **Technical Details**: 튜토리얼은 LLM과 RSs의 기본 개념, 아키텍처를 설명하고, 'pre-train, fine-tune' 패러다임과 'prompt learning' 패러다임을 포함한 다양한 훈련 전략에 초점을 맞춥니다. 여기에는 특정 추천 작업을 위한 최적화 목표를 조정하는 방법 또한 포함됩니다. 또한, RS에서의 피해 유형, 관련 이해관계자, 피해의 심각성을 평가하고, 유럽 AI 법(European AI Act)에서 영감을 받은 위험 분류를 통해 윤리적 문제를 논의합니다.

- **Performance Highlights**: LLM을 사용한 RS는 개인화된 맥락인식 추천을 제공함으로써 전통적인 추천 시스템보다 우수한 성능을 보여줍니다. 또한, 이 튜토리얼은 다양한 학습 패러다임 아래에서의 추천 성능을 평가하기 위한 데이터 세트와 평가 메트릭스(evaluation metrics)를 제공하며, 실증적 연구를 통해 이러한 훈련 전략의 실제적인 영향을 보여줍니다.



### GenQREnsemble: Zero-Shot LLM Ensemble Prompting for Generative Query  Reformulation (https://arxiv.org/abs/2404.03746)
Comments: Accepted at ECIR 2024

- **What's New**: 이 연구에서는 새로운 제로샷(Zero-shot) 쿼리 리포뮬레이션(Query Reformulation, QR) 기법을 제안합니다. GenQREnsemble과 GenQREnsembleRF는 다중 제로샷 지시문을 활용하여 사용자의 검색 쿼리를 보다 효과적으로 개선하는 방법입니다. 특히, GenQREnsembleRF는 유사 관련 피드백(Pseudo-Relevance Feedback, PRF)을 포함하여 검색 결과를 더욱 향상시키는 접근 방식을 도입하였습니다.

- **Technical Details**: GenQREnsemble은 N개의 다양한 QR 지시문을 사용하여 LLM(Large Language Models)에서 키워드를 생성하고, 이를 원본 쿼리에 추가하여 쿼리를 재구성합니다. GenQREnsembleRF는 검색 후 설정에서 문서 피드백을 추가로 고려하여 쿼리 재구성을 수행합니다. 이 기법들은 IR(Information Retrieval) 벤치마크에서 상태 최신 기술 대비 성능을 크게 향상시키며, 특히 MSMarco Passage Ranking 작업에서 상당한 개선을 보여줍니다.

- **Performance Highlights**: GenQREnsemble은 이전의 제로샷 상태 최신 기술보다 최대 18%의 상대적 nDCG@10(nDCG at rank 10) 및 24%의 MAP(Mean Average Precision) 개선을 달성했습니다. 또한, GenQREnsembleRF는 유사 관련 피드백을 사용하여 MRR(Mean Reciprocal Rank)에서 5%의 상대적 이득을, 관련 피드백 문서를 사용할 때 9%의 nDCG@10 개선을 보였습니다.



### Large language models as oracles for instantiating ontologies with  domain-specific knowledg (https://arxiv.org/abs/2404.04108)
- **What's New**: 이 연구는 대규모 언어 모델 (LLM: Large Language Model)을 활용하여 온톨로지를 자동으로 인스턴스화하는 새로운 방법, KGFiller를 제안합니다. 이 시스템은 도메인에 구애받지 않고 다양한 온톨로지에 적용할 수 있으며, 기존 데이터 세트에 의존하지 않고 LLM을 데이터 소스로 사용하여 개념, 속성 및 관계의 인스턴스를 생성합니다.

- **Technical Details**: KGFiller는 초기 스키마와 쿼리 템플릿을 사용하여 LLM에 다수의 쿼리를 수행하고, 클래스와 속성의 인스턴스를 생성하여 온톨로지를 자동으로 채웁니다. 이 방법은 도메인 특화 지식을 온톨로지에 효과적으로 통합할 수 있도록 하며, 생성된 인스턴스는 전문가들이 조정하거나 보완할 수 있습니다. 이 접근 방식은 도메인 독립적이고, 증분적이며 (Incremental), 다양한 LLM에 적용 가능합니다.

- **Performance Highlights**: 영양학 도메인에서의 케이스 스터디를 통해 KGFiller를 구현하고 다양한 LLM을 활용한 온톨로지의 품질을 평가했습니다. 평가는 LLM으로부터 생성된 인스턴스의 의미 있고 정확한 배치를 검토하여 수행되었습니다. 이 연구는 다양한 LLM을 이용하여 온톨로지를 더 효율적이고 정확하게 생성하는 방법을 선보이며, LLM을 이용한 온톨로지 채움이 효과적임을 보여줍니다.



### Investigating the Robustness of Counterfactual Learning to Rank Models:  A Reproducibility Study (https://arxiv.org/abs/2404.03707)
- **What's New**: 이 연구는 역사적 클릭 로그를 통해 랭킹 모델을 학습하는 역관찰 학습 모델인 CLTR(Counterfactual Learning to Rank)의 내구성에 대한 심층 연구를 다루고 있습니다. 기존의 시뮬레이션 기반 실험은 단일 및 결정적인 생성 랭커와 단순화된 사용자 시뮬레이션 모델을 사용하여 한계가 있었습니다. 그러나 이 논문은 다양한 랭커와 다양한 사용자 행동 가정을 사용하는 모델을 포함하여 CLTR 모델의 강건성을 테스트하고 있습니다.

- **Performance Highlights**: 확장된 시뮬레이션 기반 실험 결과, DLA 모델들과 IPS-DCM(Independent Position Skewing - Dependent Click Model)이 다양한 설정에서 IPS-PBM(Independent Position Skewing - Position-Based Model)과 PRS(Post-Rank System)와 비교해 더 나은 강건성을 보였습니다. 그러나, 우선 순위 생성 랭커의 성능이 높거나 일정한 무작위성이 존재할 때 기존 CLTR 모델은 종종 순진한 클릭 기준을 능가하지 못하는 문제점이 확인되었습니다.

- **Technical Details**: 이 연구에서는 기존 CLTR 모델을 재현하고 확장 시뮬레이션 기반 실험을 포함하여, 다양한 라벨 대비(training data)를 사용하여 생성 랭커의 성능을 다르게 하고, Plackett-Luce 모델을 사용해 그들의 무작위성을 조절했습니다. 또한, PBM 외에도 DCM(종속 클릭 모델)과 CBCM(비교 기반 클릭 모델)을 포함한 둠려 다른 사용자 시뮬레이션 모델을 사용하였습니다. 이러한 확장 실험을 통해 CLTR모델의 robustness(강건성)와 다양한 상황에서의 효과를 평가하였습니다.




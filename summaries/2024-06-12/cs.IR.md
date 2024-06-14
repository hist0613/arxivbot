### Matryoshka Representation Learning for Recommendation (https://arxiv.org/abs/2406.07432)
- **What's New**: 이번 논문에서는 추천 시스템(Recommendation Systems)에서 사용자 선호도와 아이템 특징을 효과적으로 학습하기 위한 새로운 방법인 Matryoshka Representation Learning for Recommendation (MRL4Rec)을 소개합니다. 기존의 방법들이 사용자와 아이템을 단일 벡터 또는 비중복 클러스터로 표현하는 데 비해, MRL4Rec은 계층적 구조로 표현하여 정확도를 높입니다.

- **Technical Details**: MRL4Rec에서는 사용자와 아이템 벡터를 다차원적이고 중첩된(Matryoshka) 벡터 공간으로 재구성하여 사용자 선호도와 아이템 특징을 명시적으로 나타냅니다. 또한, 각 계층별로 특정 트레이닝 트리플렛을 구성하는 Matryoshka Negative Sampling 메커니즘을 제안하여 계층적 특성을 효과적으로 학습합니다. 이런 접근법은 사용자의 전체적인 선호도에서부터 구체적인 세부 선호도까지 점진적으로 표현할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, MRL4Rec은 여러 실제 데이터셋에서 최첨단(entangled 및 disentangled representational 학습) 경쟁자들을 크게 능가하는 성능을 보였습니다. 여러 추천 메트릭에서 상당한 향상을 보여주었습니다.



### Graph Reasoning for Explainable Cold Start Recommendation (https://arxiv.org/abs/2406.07420)
- **What's New**: 이번 연구에서는 Cold Start 문제를 해결하기 위한 프레임워크인 GRECS를 제안했습니다. GRECS는 Graph Reasoning(GR) 방법을 Cold Start 추천에 적용하여 사용자에 대한 정보가 제한적일 때도 사용자의 선호도를 반영한 추천을 할 수 있습니다. 이 연구는 5개의 표준 데이터셋에서 기존의 경쟁력 있는 기준 모델들을 능가하는 성능을 보여줍니다. 또한, 설명 가능한 추천을 제공하여 사용자 신뢰를 높일 수 있음을 강조합니다.

- **Technical Details**: GRECS는 사용자 대신에 명시적인 경로를 사용하여 사용자의 선호도를 파악합니다. Knowledge Graph(KG) 내의 구조를 활용하여 사용자와 아이템 사이의 경로를 탐색하며, 이로 인해 인기 아이템 편향(Popularity Bias)을 줄이고, 최소한의 정보로도 관련 있는 아이템을 추천할 수 있습니다. 이는 기존의 Entity Embedding이나 Graph Neural Networks(GNNs)와 달리, 특히 Cold Start 상황에서 더 효율적입니다. 또한, GRECS는 비상호작용 관계(non-interaction relations)를 통해 KG에 새로운 사용자를 통합하고 관련 Embeddings을 계산합니다.

- **Performance Highlights**: 실험 결과, GRECS는 5개의 공개된 전자상거래 및 MOOC 데이터셋에서 Cold Start 사용자 및 아이템에 대해 유의미한 추천을 제공하며, 기존의 최신 기준 모델들을 능가합니다. GRECS는 특히 인기 아이템 편향에 덜 민감하며, 최소한의 관계 및 상호작용만으로도 관련 있는 추천을 할 수 있습니다.



### Text Information Retrieval in Tetun: A Preliminary Study (https://arxiv.org/abs/2406.07331)
- **What's New**: 이번 연구는 티모르-레스트의 공용어 중 하나인 Tetun에 대한 정보 검색 솔루션을 개발하는 것을 목표로 합니다. 검색 기술이 없는 상태에서 Tetun으로 정보를 찾는 것은 매우 어려운 과제입니다. 이를 해결하기 위해, 본 논문은 ad-hoc 검색 실험의 초기 결과를 제시하며, Tetun 검색 솔루션 개발의 기초를 마련합니다.

- **Technical Details**: Tetun은 티모르-레스트에서 가장 많이 사용되는 언어 중 하나로, 많은 단어가 포르투갈어와 인도네시아어, 영어의 영향을 받았습니다. 본 연구는 텍스트 정보를 효과적으로 검색하기 위해 Tetun 말뭉치(corpus)를 개발하고, 이를 기반으로 검색 알고리즘의 성능을 평가합니다. 연구 방법으로는 웹 크롤링(Web Crawling)을 통해 데이터 세트를 구축하며, 텍스트 전처리와 표준화, 토큰화(tokenization) 등의 언어 특화 알고리즘을 적용합니다.

- **Performance Highlights**: 초기 실험 결과, Timor News 플랫폼을 사용한 인덱싱 및 검색 실험에서 유의미한 성과를 거두었습니다. Google 검색 엔진과 비교했을 때, Tetun의 혼합된 언어적 특성을 고려한 검색 결과가 더욱 정밀하게 반환되는 것을 확인했습니다.



### Fetch-A-Set: A Large-Scale OCR-Free Benchmark for Historical Document Retrieva (https://arxiv.org/abs/2406.07315)
Comments:
          Preprint for the manuscript accepted for publication in the DAS2024 LNCS proceedings

- **What's New**: Fetch-A-Set (FAS)은 역사적 문서 분석 시스템을 대상으로 하는 포괄적인 벤치마크로, 대규모 문서 검색의 도전 과제를 해결합니다. 이 벤치마크는 XVII 세기까지 거슬러 올라가는 방대한 문서 저장소를 포함하여 검색 시스템을 위한 교육 리소스 및 평가 기준을 제공하며, 문화유산 분야의 복잡한 추출 작업에 중점을 둡니다. FAS는 텍스트-이미지 검색 및 이미지-텍스트 주제 추출과 같은 다면적 역사적 문서 분석 작업을 다룹니다.

- **Technical Details**: FAS 벤치마크는 스페인의 공식 관보인 'Boletín Oficial del Estado'의 문서를 포함한 40만 개의 샘플로 구성되어 있습니다. 문서와 쿼리 간의 일대일 관계를 할당하기 위해 강력한 휴리스틱을 설계했으며, 문서의 특정 조각과 OCR 텍스트를 일치시키는 두 단계를 통해 정확한 정보 추출(이미지-텍스트) 및 주제 탐색(텍스트-이미지)을 보장합니다. 이를 위해 Mask-RCNN 모델 및 sentence-bert 인코딩을 사용하였고, 헝가리 알고리즘 등을 적용하여 데이터 정확도를 높였습니다.

- **Performance Highlights**: 벤치마크는 전통적인 OCR 기반 방법 및 비전 기반 접근 방식을 제시하여 각각이 언제 가장 잘 작동하는지 설명합니다. 이를 통해 연구자들이 다양한 역사적 문서 시나리오에서 더 효과적인 비교를 수행할 수 있도록 안내합니다. 또한, OCR 전사에서 시각적 통찰력을 통합하는 것이 중요하다는 점을 강조하며, FAS 벤치마크를 통해 저조한 가독성 시나리오에서의 성능 향상 가능성을 탐구합니다.



### Exploring Large Language Models for Relevance Judgments in Tetun (https://arxiv.org/abs/2406.07299)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 저자원 언어(Low-resource languages, LRLs) 환경에서 관련성 판단 작업을 자동화할 수 있는 가능성을 탐구하였습니다. 특히, 티모르-레스트에서 923,000명 이상이 사용하는 텟텀(Tetun) 언어를 대상으로 하여, 문서와 쿼리 쌍에 대한 관련성 점수를 할당하고, 이를 사람 평가자와 비교하였습니다. 이 연구의 결과는 고자원 언어 연구에서 보고된 결과와 큰 차이가 없었습니다.

- **Technical Details**: 이 연구에서는 텟텀 언어로 작성된 문서와 쿼리 쌍을 LLMs에 제공하여 자동으로 관련성 점수를 부여하였습니다. 사용한 LLMs 모델은 Meta의 LLaMA3 70B 모델 및 두 가지 유료 모델(Anthropic의 Haiku variant of Claude 3, OpenAI의 Turbo variant of GPT-3.5)입니다. 이러한 모델들의 평가 결과는 사람 평가자와의 Cohen's kappa score로 측정되었습니다.

- **Performance Highlights**: 실험 결과, LLaMA3 70B 모델은 사람의 평가와 0.2634의 평균 Cohen’s kappa 점수를 기록해 비교적 높은 일치도를 보였습니다. 유료 모델의 경우 번역 품질 평가(BLEU metric)에서 무료 모델보다 우수한 성능을 보여줬습니다.



### Progressive Query Expansion for Retrieval Over Cost-constrained Data Sources (https://arxiv.org/abs/2406.07136)
- **What's New**: ProQE는 기존의 PRF (Pseudo-Relevance Feedback)와 최신 대형 언어 모델(LLM, Large Language Models)을 결합한 새로운 프로그레시브 쿼리 확장 알고리즘입니다. 쿼리 확장을 반복적으로 수행하며, 얼마나 많은 문서를 검색하느냐에 따라 쿼리를 점진적으로 확장하는 방식입니다. 기존 PRF의 노이즈 문제를 줄이기 위해 검색된 각 결과의 중요도를 평가하는 데 LLM을 사용합니다.

- **Technical Details**: ProQE는 희소(Sparse)나 밀집(Dense) 검색 시스템과 호환되도록 설계되었으며, 다음과 같은 주요 설계 원칙을 따릅니다: 1) 초기 검색된 문서의 각 용어의 중요도를 LLM으로 판단하여 쿼리 확장에 반영, 2) 여러 차례의 반복을 통해 쿼리를 점진적으로 업데이트, 3) 최종 쿼리를 LLM의 chain-of-thought(논리적 사고 과정)으로 구성. 특히, 삽입형(plug-and-play) 방식으로 다양한 검색 시스템과 통합이 용이합니다.

- **Performance Highlights**: ProQE는 네 가지 검색 데이터세트(Natural Questions, Web Questions, TREC DL19, DL20)에서 최첨단 기준을 37% 이상 상회하는 성능을 보여줍니다. 또한 모든 비교 대상 중 가장 비용 효율적입니다. MRR과 Rank@1 측면에서 평균 37%의 개선을 달성했습니다.



### The Treatment of Ties in Rank-Biased Overlap (https://arxiv.org/abs/2406.07121)
Comments:
          10 pages, 5 figures, 4 tables, SIGIR 2024

- **What's New**: 이번 연구는 Rank-Biased Overlap(RBO)의 확장된 정의를 제안하며, 특히 동순위(Ties)를 다루는 방식을 개선하였다. 기존 RBO는 동순위를 다루는 방법이 불완전하였지만, 이번 연구를 통해 동순위가 포함된 랭킹의 접두사(prefix) 평가를 가능하게 하였다. 또한, 통계학에서 전통적으로 사용되던 방식과 일치하는 두 가지 변형된 RBO를 개발하였다: 하나는 참조 랭킹이 있는 경우, 다른 하나는 참조 랭킹이 없는 경우.

- **Technical Details**: 연구에서는 동순위를 처리할 수 있는 일반화된 RBO를 제안하였다. 이로 인해 랭킹 접두사만 가지고도 RBO 점수를 계산할 수 있게 되었으며, 통계 문헌에서 사용된 방식과 일치하는 두 가지 변형도 포함되었다. 동시에, 실험을 통해 새로운 동순위 인식 RBO 측정을 적용하고, 원래의 동순위 인식 불가 RBO와 비교하여 그 차이를 밝혔다.

- **Performance Highlights**: TREC 데이터를 활용한 실험 결과, 동순위 인식 RBO는 기존 RBO와 상당한 차이를 보였다. 특히, 동순위를 랜덤하게 나누거나 문서 ID와 같은 임의의 기준으로 나누는 기존 방식은 결과의 일관성을 해칠 수 있음을 입증하였다. 새로운 동순위 인식 RBO는 더욱 정확하고 신뢰할 수 있는 랭킹 유사성 측정을 제공함을 보여주었다.



### Guiding Catalogue Enrichment with User Queries (https://arxiv.org/abs/2406.07098)
Comments:
          ECML PKDD 2024

- **What's New**: 이 논문에서는 제품 검색 과정에서 사용자 검색 행동을 활용하여 사용자와의 관련성을 높이는 방식으로 제품 카탈로그에 대한 지식 그래프(knowledge graph, KG)의 완성을 개선하는 방법을 제안합니다. 특히, 사용자 쿼리를 분석하여 엔티티-속성 쌍을 추출하고 이를 통해 예측의 정확성과 관련성을 높이는 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법은 사용자가 제품 검색 엔진에서 쿼리한 로그 데이터를 활용하여, 특정 엔티티-속성 쌍을 기반으로 KG 완성 예측을 유도합니다. 이 방법은 주로 RDF 기반의 KG에 적용되며, 다양한 KGC (knowledge graph completion) 방법들과 보완적으로 사용될 수 있습니다. 논문에서는 DBPedia와 YAGO 4와 같은 두 개의 인기 있는 백과사전형 KGs를 사용해 실험을 진행했습니다. 주요 기술적 요소로는 SPARQL 쿼리 로그 분석과 이에 기반한 triplet 예측 지침이 포함됩니다.

- **Performance Highlights**: 자동화 및 인간 평가 결과, 사용자 쿼리 지침 방식을 사용하면 정확성 및 관련성이 크게 향상된다는 것이 입증되었습니다. 특히, 사용된 쿼리 지침 방법은 기존의 무작위 샘플링 방법에 비해 최소 두 배 이상 높은 정확성을 보였습니다. 또한, 논문에서는 1600개의 엔티티-속성 쌍 정답 데이터셋을 공개하여 실험 결과의 신뢰성을 높였습니다.



### Grapevine Disease Prediction Using Climate Variables from Multi-Sensor Remote Sensing Imagery via a Transformer Mod (https://arxiv.org/abs/2406.07094)
- **What's New**: 이 논문은 TabPFN 모델을 활용하여 멀티센서 원격 탐사 이미지에서 수집된 기후 변수를 기반으로 덩굴병(blockwise grapevine diseases)을 예측하는 새로운 프레임워크를 소개합니다. 이 모델링 접근법은 원격 탐사 데이터와 Transformer 모델을 결합하여 포도밭 병해 충 예측의 정확성과 효율성을 크게 향상시킵니다.

- **Technical Details**: TabPFN 모델은 12개의 층(Layer)으로 구성된 Transformer 기반의 모델로, 소규모 표(Tabular) 데이터 셋에서의 분류 작업을 위해 설계되었습니다. Bayesian 추론에 기반한 후방 확률 분포(PPD)를 근사화하는 데 중점을 두고 있으며, 사전 학습된 모델을 통해 테스트 샘플에 대해 빠르게 예측할 수 있습니다. 예측까지의 전 과정은 하이퍼파라미터 튜닝 없는 단일 포워드 패스(Forward Pass)로 수행됩니다.

- **Performance Highlights**: TabPFN 모델은 전통적인 gradient-boosted decision trees (예: XGBoost, CatBoost, LightGBM)와 비슷한 성능을 나타냈습니다. 또한, 복잡한 데이터를 처리하고 픽셀 단위로 질병 영향을 받을 확률을 제공함으로써 정밀한 중재가 가능하게 하여 지속 가능한 질병 관리 실천에 기여합니다. 실험에서는 호주의 76개의 포도밭에서 두 시즌 동안 627개의 블록에 걸쳐 측정된 데이터베이스를 활용하여 9가지 이상의 다양한 질병을 예측했습니다.



### TIM: Temporal Interaction Model in Notification System (https://arxiv.org/abs/2406.07067)
- **What's New**: TIM(Temporal Interaction Model)을 제안하며, 이는 사용자 행동 패턴을 모델링하여 하루 동안의 시간 슬롯마다 클릭률(CTR)을 추정하는 시스템입니다. TIM은 Kuaishou 같은 단기 비디오 애플리케이션에서 사용자 상호작용을 극대화하기 위해 개발되었습니다.

- **Technical Details**: TIM은 장기간 사용자 이력 상호작용 시퀀스를 특징으로 삼아, 알림 수신, 클릭, 시청 시간, 효과적인 조회수 등을 포함합니다. 또한 Temporal Attention Unit(TAU)을 사용하여 사용자 행동 패턴을 추출합니다. 하루를 여러 시간 슬롯으로 나누어 각 슬롯별 CTR을 추정하며, 효과적인 알림 타이밍 통제 전략을 제안합니다.

- **Performance Highlights**: TIM의 효용성은 오프라인 실험과 온라인 A/B 테스트를 통해 평가되었습니다. 결과는 TIM이 사용자 행동 예측에 있어 신뢰할 만한 도구임을 나타내며, 과도한 방해 없이 사용자 참여를 크게 향상시킵니다.



### Which Country Is This? Automatic Country Ranking of Street View Photos (https://arxiv.org/abs/2406.07227)
- **What's New**: Country Guesser는 Google 스트리트 뷰 이미지를 기반으로 해당 사진이 어느 나라에서 촬영되었는지를 추측하는 새로운 시스템입니다. 이 시스템은 컴퓨터 비전(computer vision), 기계 학습(machine learning), 텍스트 검색(text retrieval) 방법을 결합하여 작업을 수행합니다.

- **Technical Details**: Country Guesser는 다양한 개별 모듈의 증거를 활용하여 가능한 국가 목록을 생성합니다. 주요 기술로는 광학 문자 인식(OCR), 태양 위치 분석(sun position analysis), 색 히스토그램(color histogram), 객체 감지 YOLO 등이 포함됩니다. Geo-JSON 국가 경계 폴리곤 데이터를 사용하여 국가를 식별하고, EasyOCR을 사용하여 텍스트를 얻은 다음 Python langdetect 및 lingua 라이브러리를 사용하여 텍스트 언어를 분석합니다. 또한 자동 캡션 생성(ClipCap) 모델을 활용해 이미지 설명의 단어 빈도를 분석합니다.

- **Performance Highlights**: 시스템은 110개국에 대한 랜덤하게 선택된 좌표의 파노라마 이미지를 테스트 데이터셋으로 사용하여 평가되었습니다. 그 결과, 220개의 추측 중에서 올바른 국가가 평균적으로 14.7위에 올랐으며, 35건의 경우 올바른 국가가 첫 번째로 랭크되었습니다.

- **Applications and Impact**: Country Guesser는 엔터테인먼트, 교육, 조사 저널리즘, 법 집행 등 다양한 용도로 사용할 수 있습니다. 예를 들어, 퀴즈 게임으로 사용하여 지리적 단서를 식별하는 능력을 테스트하거나, 특정 사건의 위치를 확인하는 도구로 활용될 수 있습니다.



### EEG-ImageNet: An Electroencephalogram Dataset and Benchmarks with Image Visual Stimuli of Multi-Granularity Labels (https://arxiv.org/abs/2406.07151)
- **What's New**: EEG 기반 시각재구성 및 분류 데이터셋인 EEG-ImageNet을 소개합니다. 이 데이터셋은 16명의 참가자들이 ImageNet에서 선정된 4000개의 이미지를 볼 때 기록된 EEG 데이터를 포함하고 있습니다. 이는 기존 EEG 벤치마크보다 5배 더 많은 EEG-이미지 쌍을 제공합니다.

- **Technical Details**: EEG-ImageNet 데이터셋은 다중 그레인(Granularity) 레이블을 가지고 있습니다. 40개의 이미지가 애매한(granularity) 레이블로, 나머지 40개는 세밀한(fine-grained) 레이블로 구성되어 있습니다. 이 데이터셋은 EEG 신호로부터 객체 분류 및 이미지 재구성 작업을 벤치마크로 제공합니다.

- **Performance Highlights**: 다양한 모델로 실험한 결과, 객체 분류 작업에서 최고 60%의 정확도를 기록했으며, 이미지 재구성 작업에서는 64%의 2방향 식별 성능을 나타냈습니다. 이는 EEG 기반의 시각 뇌-컴퓨터 인터페이스(BCI)의 연구를 촉진하고 생물학적 시스템의 시각 인식을 이해하는 데 큰 잠재력을 보여줍니다.



### Unlocking the Potential of the Metaverse for Innovative and Immersive Digital Car (https://arxiv.org/abs/2406.07114)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 논문은 메타버스(Metaverse)가 헬스케어 분야에 가져올 혁신적인 잠재력에 대해 탐구합니다. 이 기술이 환자 관리, 의학 교육 및 연구를 어떻게 변화시킬 수 있는지 논의하고, 메타버스 데이터 분석을 통해 헬스케어 응용 프로그램을 향상시킬 수 있는 기회를 강조합니다.

- **Technical Details**: 메타버스는 지속적이고 몰입감 있는 가상 환경으로, 환자 참여, 의사소통, 정보 접근 및 건강 결과를 개선할 수 있는 능력을 가지고 있습니다. 또한, 머신 러닝 기법을 활용한 메타버스 데이터 분석을 통해 더 나은 헬스케어 통찰력을 얻을 수 있습니다.

- **Performance Highlights**: 논문은 메타버스 통합의 주요 발견과 실질적인 의미를 분석하며, 미래 연구를 위하여 주목해야 할 분야를 식별합니다. 주요 기술 회사들이 메타버스 기반 솔루션을 개발하는 역할과 함께, 이 기술을 헬스케어 분야에 효과적이고 윤리적으로 구현하기 위해 필요한 협력의 중요성을 강조합니다.



### ElasticRec: A Microservice-based Model Serving Architecture Enabling Elastic Resource Scaling for Recommendation Models (https://arxiv.org/abs/2406.06955)
- **What's New**: ElasticRec는 추천 시스템(RecSys)을 위한 새로운 모델 서빙 아키텍처로, 리소스 탄력성과 높은 메모리 효율성을 제공합니다. 기존의 모델 단위로 리소스를 할당하는 방식 대신, ElasticRec는 미세한 수준의 리소스 할당을 가능하게 하는 마이크로서비스(microservice) 소프트웨어 아키텍처를 바탕으로 설계되었습니다.

- **Technical Details**: ElasticRec는 두 가지 주요 특징을 가지고 있습니다. 첫째, 마이크로서비스 기반 추론 서버를 사용하여 리소스 탄력성을 극대화합니다. 마이크로서비스는 큰 단일 응용 프로그램을 여러 개의 독립적이고 세밀한 서비스로 나눌 수 있게 해줍니다. 둘째, 유틸리티 기반의 리소스 할당 정책을 통해 높은 메모리 효율성을 달성합니다. 모델은 밀집 DNN 레이어와 희석 임베딩 레이어로 나뉘며, 임베딩 레이어는 다시 '핫'과 '콜드' 임베딩 단위로 나뉩니다.

- **Performance Highlights**: ElasticRec는 메모리 할당 크기를 평균 3.3배 줄이고, 메모리 유틸리티를 8.1배 증가시켜 평균적으로 배포 비용을 1.6배 절감할 수 있습니다. 또한, 고유한 리소스 수요에 맞춘 리소스 할당을 통해 전체적인 QPS(Queries Per Second)를 최대화할 수 있습니다.



### Non-autoregressive Personalized Bundle Generation (https://arxiv.org/abs/2406.06925)
Comments:
          Submitted to Information Processing & Management

- **What's New**: 최근 추천 시스템 연구에서, 사용자의 선호에 맞춰 개인화된 번들을 생성하는 문제에 대한 관심이 증가하고 있습니다. 기존 연구들은 번들의 순서 불변성 특성을 고려하지 않아 순차적 모델링 방법을 채택한 반면, 본 연구에서는 비자기회귀(non-autoregressive) 메커니즘을 활용하여 번들을 생성하는 새로운 인코더-디코더 프레임워크인 BundleNAT을 제안합니다. 이를 통해 본 연구는 번들의 순서에 의존하지 않고 한 번에 목표 번들을 생성할 수 있습니다.

- **Technical Details**: 본 연구에서는 사전 훈련(pre-training) 기술과 그래프 신경망(Graph Neural Network, GNN)을 채택하여 사용자 기반 선호도 및 아이템 간 호환성 정보를 완전하게 내재화합니다. 이후 자기 주의(self-attention) 기반 인코더를 활용하여 글로벌 종속성 패턴을 추출합니다. 이를 기반으로 번들 내의 순열에 고유한 디코딩 아키텍처를 설계하여 직접적으로 원하는 번들을 생성합니다.

- **Performance Highlights**: YouShu와 Netease의 세 가지 실제 데이터셋에서 진행된 실험 결과, BundleNAT은 정밀도(Precision), 확장 정밀도(Precision+), 재현율(Recall)에서 각각 최대 35.92%, 10.97%, 23.67%의 절대적 향상을 보여 현재 최신 기법들을 크게 능가하는 성과를 보였습니다.



### Link Prediction in Bipartite Networks (https://arxiv.org/abs/2406.06658)
Comments:
          28th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES), Sep 2024, Sevilla, Spain

- **What's New**: 이번 연구에서는 이분 그래프(bipartite networks)에서 링크 예측(link prediction)을 위한 19개의 방법을 비교 실험하였습니다. 일부 방법은 기존 문헌에서 가져왔으며, 일부는 단일 네트워크(unipartite networks)를 위해 설계된 기술을 저자들이 이분 그래프로 수정한 것입니다. 추가적으로, 그래프 합성망(Convolutional Networks, GCN)을 기반으로 하는 추천 시스템을 이분 그래프의 새로운 링크 예측 솔루션으로 제안하였습니다.

- **Technical Details**: 이번 연구에서는 다양한 위상구조를 가진 3개의 실제 이분 그래프 데이터셋을 구축하여 실험을 진행하였습니다. 연구에 포함된 19개의 링크 예측 방법에는 기존 연구에서 사용된 방법과, 단일 네트워크를 위해 설계되었지만 이분 그래프로 수정한 방법이 포함되어 있습니다. GCN 기반 개인화 추천 시스템을 통해 링크 예측 성능을 평가하였으며, 또한 학습 프로세스에 의존하지 않는 순수한 휴리스틱 지표(Structural Perturbation Method, SPM) 역시 효과적인 결과를 보였습니다.

- **Performance Highlights**: 결과적으로, GCN 기반 개인화 추천 시스템은 이분 그래프에서 성공적인 링크 예측을 할 수 있음을 보였습니다. 또한 학습 과정이 필요 없는 구조 교란법(SPM)과 같은 순수한 휴리스틱 지표도 성공적으로 링크 예측을 수행했습니다.




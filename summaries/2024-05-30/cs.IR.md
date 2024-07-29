### A Multi-Source Retrieval Question Answering Framework Based on RAG (https://arxiv.org/abs/2405.19207)
Comments:
          4 pages,3 figures

- **What's New**: 본 연구에서는 전통적인 정보 검색기를 GPT-3.5로 대체하여 검색 정보를 생성하는 방법을 제안합니다. 이를 통해 기존의 검색 기반 생성(Re-augmented Generation, RAG) 패러다임이 겪는 오류 정보 문제를 해결하고자 했습니다. 또한, 웹 기반 검색 방법을 도입하여 세밀한 지식 검색을 구현하고자 합니다.

- **Technical Details**: GPT-3.5의 강력한 추론 능력을 활용하여 의미적인 분할을 수행함으로써, 검색 중 발생할 수 있는 노이즈를 줄이고 정확도를 높였습니다. 멀티 소스 검색 프레임워크(MSRAG)를 제안하여 GPT 검색과 웹 검색을 결합했습니다. 이를 통해 더욱 정교한 정보 검색이 가능합니다.

- **Performance Highlights**: 여러 지식 집약적 QA 데이터셋을 이용한 실험에서, 본 연구에서 제안된 MSRAG 프레임워크가 기존의 RAG 프레임워크보다 QA 시스템의 효율성 및 정확성을 높이는 데 더 우수한 성능을 나타냈습니다.



### Continual Collaborative Distillation for Recommender System (https://arxiv.org/abs/2405.19046)
Comments:
          KDD 2024. 9 pages + appendix (1 page). 5 figures

- **What's New**: 지속적으로 유입되는 데이터 스트림에서 교사-학생 지식 증류(KD)를 운영하는 새로운 접근법으로 '지속적 협업 증류(CCD)' 프레임워크를 제안하였습니다. 이는 대규모 교사 모델의 성능을 유지하면서도 신규 데이터를 효과적으로 반영할 수 있는 소형 학생 모델을 활용한 효율적 배포를 목표로 합니다.

- **Technical Details**: CCD 프레임워크에서는 교사와 학생이 각자의 특성에 맞춰 데이터 스트림을 따라 지속적으로 협업하며 진화합니다. 학생 모델은 소형이기 때문에 짧은 주기로 자주 업데이트될 수 있으며, 교사 모델은 더 큰 용량을 통해 장기간의 업데이트 주기를 가집니다. 주요 기술로는 새로운 데이터 학습을 촉진하는 엔티티 임베딩 초기화와 과거 예측을 재학습하여 잊혀진 지식을 보존하는 프록시 가이드 리플레이 학습(Proxy-Guided Replay Learning) 등이 포함됩니다.

- **Performance Highlights**: 실험 결과, CCD 프레임워크는 기존의 일회성 증류와 비교했을 때 효율적인 학습 및 최적의 성능을 보여주었으며, 두 가지의 실제 데이터셋을 통한 확장된 정량적, 선택적, 탐색적 실험으로 그 유효성을 검증했습니다.



### SynerGraph: An Integrated Graph Convolution Network for Multimodal Recommendation (https://arxiv.org/abs/2405.19031)
- **What's New**: 이번 연구에서는 다중모달 추천 시스템(multimodal recommendation systems)을 위한 새로운 접근법을 제시합니다. 본 방법론은 다양한 형태의 데이터를 정제하고(noise를 제거하여) 더 신뢰성 있는 추천을 생성하는 필터 개발로 시작합니다. 본 연구는 텍스트 정보가 시각적 데이터보다 항목을 깊이 이해하는 데 중요한 역할을 한다고 강조합니다. 제안된 모델은 사용자 선호도를 정확히 포착하는 자가 지도 보조 과제(self-supervised auxiliary task)를 포함하고 있어 기존 최첨단 다중모달 추천 시스템보다 우수한 성능을 보입니다.

- **Technical Details**: 제안한 프레임워크는 그래프 신경망(Graph Neural Networks, GNNs)을 활용하여 다중모달 데이터를 효율적으로 처리합니다. Amazon Reviews 데이터를 사용하여 장난감, 스포츠 용품, 사무 용품의 세 가지 카테고리에서 텍스트와 이미지 데이터를 결합하였습니다. 텍스트 데이터의 경우 Sentence Transformers를 사용하여 텍스트를 의미 벡터로 변환하였고, 이미지 데이터는 사전 생성된 이미지 특징을 그대로 사용하였습니다. 구매 이력을 기반으로 그래프 내의 엣지를 설정하였습니다.

- **Performance Highlights**: 제안된 모델은 여러 개의 모달리티를 통합함으로써 단일 모달리티에 의존하는 시스템보다 더 우수한 성능을 보였습니다. 필터(모달리티 정제자)를 포함하지 않은 모델은 성능이 감소하였으며, 다양한 데이터셋에서 top-K 희소화를 최적화하여 과적합과 과소적합 사이에서 균형을 유지했습니다. Extensive experiments demonstrate that our model achieves superior results than existing state-of-the-art multimodal recommendation systems.



### Mitigate Position Bias with Coupled Ranking Bias on CTR Prediction (https://arxiv.org/abs/2405.18971)
Comments:
          5 pages, 3 figures

- **What's New**: 이 논문에서는 기존의 추천 시스템에서 간과되었던 또 다른 바이어스인 ranking bias(순위 편향)와 position bias(위치 편향)가 결합하여 발생하는 문제를 해결하기 위한 새로운 방법을 제안합니다. 저자들은 gradient interpolation(그래디언트 인터폴레이션)이라는 새로운 위치 편향 추정 방법을 도입하여 두 가지 추정 방법을 융합하고, 적응적 방법을 사용하여 최적의 융합 가중치를 자동으로 결정하는 방법을 제안했습니다.

- **Technical Details**: 논문에서는 position bias가 사용자 행동에 미치는 영향을 설명하고, 이를 해결하기 위한 기존 방법들이 대부분 ranking bias라는 또 다른 중요한 요소를 무시하고 있음을 지적합니다. 저자들은 position gradient(위치 그래디언트)라는 새로운 개념을 도입하여 위치 변화에 따른 CTR(클릭률)의 변화를 분석하고, 모형이 최상위 위치에 놓인 아이템에 대해 CTR을 과대평가하는 'overestimation of position gradient' 문제를 해결하기 위한 gradient interpolation 방법을 제안합니다. 이 방법은 과대평가 모델과 과소평가 모델을 융합하여 적절한 추정치를 도출하게 합니다. 추가로, 적은 수의 랜덤 순위 샘플이 있을 경우 최적의 인터폴레이션 계수를 효과적으로 획득할 수 있는 적응적 솔루션도 제안됩니다.

- **Performance Highlights**: 제안된 방법은 인공 데이터셋과 산업 데이터셋 모두에서 기존의 위치 편향 추정 방법을 능가하는 성능을 보였습니다. 두 개의 데이터셋과 두 개의 온라인 추천 시나리오에 대한 실험 결과, 제안된 방법이 일관된 개선을 이루어냈음을 입증하였습니다.



### Content-Agnostic Moderation for Stance-Neutral Recommendation (https://arxiv.org/abs/2405.18941)
- **What's New**: 개인화된 추천 시스템이 사용자들을 더 극단적인 콘텐츠로 유도하여 의견 양극화를 악화시키는 문제를 해결하기 위해, 콘텐츠 중심의 검열 대신에 '콘텐츠 비관련(content-agnostic)' 검열 방안을 제안하고 그 실효성을 탐구했습니다. 콘텐츠 비관련 검열은 실제 콘텐츠에 의존하지 않아 검열의 위험이 낮습니다. 제안된 방법은 사용자-아이템 상호작용만을 활용하여 추천을 수정하는 것입니다.

- **Technical Details**: 콘텐츠 비관련 검열이 전반적으로 효과적이지 못할 수 있다는 이론적인 한계를 인식하면서도, 실질적으로는 사용자-아이템 상호작용 등의 관계적 특성을 활용하면 효과적으로 활용할 수 있음을 보였습니다. 두 가지 새로운 콘텐츠 비관련 검열 방법인 '랜덤 확산(Random Dispersal)'과 '유사성 기반 확산(Similarity-based Dispersal)'을 도입했습니다. 이 방법들은 콘텐츠 특징에 의존하지 않고 추천 목록을 수정함으로써 사용자의 의견 중립성을 향상시킵니다.

- **Performance Highlights**: 시뮬레이션 환경에서의 실험을 통해, 제안된 검열 방법들이 다양한 데이터 시나리오에서 의견 중립성을 향상시키고 높은 추천 품질을 유지할 수 있음을 증명했습니다. 콘텐츠 정보에 직접 의존하지 않고도 의견 중립성을 달성할 수 있는 가능성을 보여줌으로써, 사용자 참여를 크게 저해하지 않으면서 더 균형 잡힌 정보 제공 시스템을 개발하는데 기여합니다.



### Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery (https://arxiv.org/abs/2405.19164)
Comments:
          8 pages, 2 tables, 6 figures

- **What's New**: 이 논문에서는 eDiscovery 프로세스에서 문서의 관련성을 예측하고 추론하는 새로운 하이브리드 접근법인 DISCOG(DISCOvery Graph)를 소개합니다. 이 접근법은 이종 그래프 기반 방법과 대규모 언어 모델(LLM) 기반 방법의 강점을 결합하여 효율성과 비용 효과를 극대화합니다. 기존 접근 방식은 성능, 계산량, 해석력에서 도전 과제에 직면했으나, DISCOG는 이 문제를 해결합니다.

- **Technical Details**: DISCOG는 문서의 관련성을 예측하기 위해 그래프 표현 학습을 사용하고, 문서의 관련성을 추론하기 위해 LLM을 사용합니다. 이 방법은 주어진 요청에 대해 코퍼스를 랭킹하고 링크 예측을 통해 임베딩을 생성합니다. 또한 균형 잡힌 데이터셋과 불균형한 데이터셋 모두를 효과적으로 처리합니다.

- **Performance Highlights**: DISCOG는 기존 방식에 비해 F1-score, precision, recall 면에서 각각 평균 12%, 3%, 16% 더 우수한 성능을 보였습니다. 기업 환경에서는 문서 검토 비용을 수동 프로세스에 비해 99.9%, LLM 기반 분류 방법에 비해 95% 줄이는 데 크게 기여했습니다.



### CaLa: Complementary Association Learning for Augmenting Composed Image Retrieva (https://arxiv.org/abs/2405.19149)
Comments:
          arXiv admin note: text overlap with arXiv:2309.02169

- **What's New**: 최근 조사된 논문에서는 Composed Image Retrieval (CIR)을 기존의 단순 query-target 매칭 문제로 보지 않고, 새로운 관계를 도입하여 보다 정교한 접근을 제안합니다. 특히, 텍스트-이미지 쌍을 그래프 노드로 간주하고, 텍스트를 브릿지로 사용하여 image alignment(이미지 정렬)를 유도하는 hinge-based cross-attention 메커니즘과, 두 이미지를 활용해 complementary text(보완적인 텍스트)를 추론하는 방법을 통합한 twin attention-based compositor를 제안합니다. 이를 통해 CIRR 및 FashionIQ 벤치마크에서 우수한 성능을 입증했습니다.

- **Technical Details**: 제안된 방법론은 세 가지 주요 관계에 기반합니다. 첫째, 텍스트를 브릿지로 활용하여 참조 이미지와 대상 이미지를 정렬하는 hinge-based cross-attention 메커니즘을 도입했습니다. 둘째, 두 이미지를 통합해 보완 텍스트를 추론하는 twin attention-based composer를 사용했습니다. 이를 통해 참조 이미지와 대상 이미지를 효율적으로 결합하여 종합적인 시각 표현(Vision Representation)을 생성함으로써 초기 질의와 대상 이미지를 대조 학습(Contrastive Learning)하여 최적화를 수행했습니다. 전체 네트워크는 이러한 세 가지 절차에서 파생된 Alignment Loss로 최적화되었습니다.

- **Performance Highlights**: 새롭게 제안한 CaLa (Complementary Association Learning for Augmenting Composed Image Retrieval) 프레임워크는 CIRR 및 FashionIQ 벤치마크에서 다양한 백본(Backbone) 모델을 활용하여 뛰어난 성능을 발휘했습니다. 이는 CIR 작업에 있어 CaLa가 광범위하게 유익한 모듈임을 시사합니다. CLIP4Cir 및 다른 최신 방법들과 비교했을 때, 더 높은 정확성을 달성하여 테스트 이미지를 효과적으로 검색하는 데 기여했습니다.



### An engine not a camera: Measuring performative power of online search (https://arxiv.org/abs/2405.19073)
- **What's New**: 이 연구는 온라인 검색 제공자의 'performative power'를 측정하기 위한 실험을 설계하고 실행했다. performative power은 검색 엔진이 결과를 재정렬하여 웹 트래픽을 조종하는 능력을 의미한다. 이를 위해 브라우저 확장 프로그램을 개발하여 무작위 실험을 실행하고, 데이터 분석을 통해 검색 결과 배열이 클릭에 미치는 인과적 효과를 확인하였다.

- **Technical Details**: 브라우저 확장 프로그램 'Powermeter'를 사용하여 소규모 알고리즘 업데이트를 에뮬레이트하고, 사용자가 검색할 때마다 무작위로 다른 결과 배열을 보여주었다. 실험 동안 약 57,000개의 검색 쿼리를 80명 이상으로부터 수집하였다. 데이터를 통해 네이티브 사용자-플랫폼 상호작용 하에서의 배치의 인과 효과를 측정하였다.

- **Performance Highlights**: 실험 결과, 검색 결과의 첫 번째 요소를 한 단계 내리는 것만으로도 Google 검색에서는 클릭이 평균 42% 감소했다. 두 단계 내리면 50% 이상의 감소를 나타냈다. Bing에서도 유사한 결과가 발견되었으나, 데이터 샘플이 적어 신뢰 구간이 덜 타이트함을 보였다. 또한 광고나 쇼핑 박스 등을 추가하면 배열의 효과가 더욱 두드러지게 나타났다. 이를 통해 performative power가 디지털 시장 조사에서 어떻게 적용될 수 있는지 실질적인 예를 제공하였다.



### Leveraging Many-To-Many Relationships for Defending Against Visual-Language Adversarial Attacks (https://arxiv.org/abs/2405.18770)
Comments:
          Under review

- **What's New**: 이 논문은 이미지-텍스트 검색(image-text retrieval, ITR)에서 비전-언어(Vision-Language, VL) 모델이 적대적 공격(adversarial attacks)에 취약하다는 연구 결과를 바탕으로 처음으로 ITR에 대한 방어 전략을 탐구했습니다. 특히, ITR의 N:N(다대다) 관계를 활용하여 적대적 견고성(adversarial robustness)을 향상시키는 방법에 중점을 두었습니다.

- **Technical Details**: 기존의 적대적 훈련(adversarial training)은 학습 데이터에 있는 특정 1:1(일대일) 이미지-텍스트 쌍에 과적합(overfit)되는 경향이 있어, 다양한 증강 기법을 통해 1:N(일대다) 및 N:1(다대일) 이미지-텍스트 쌍을 생성하면 적대적 견고성을 크게 향상시킬 수 있음을 발견했습니다. 또한, 증강된 이미지-텍스트 쌍의 정렬(alignment)이 방어 전략의 효과에 중요한 역할을 하며, 부적절한 증강은 모델의 성능을 악화시킬 수 있음을 보였습니다. 이 발견을 바탕으로, 기본 증강 기술과 생성 모델 기반 증강을 사용하여 다양한 N:N 쌍을 효과적으로 생성하는 새로운 방어 전략을 제안했습니다.

- **Performance Highlights**: 두 개의 대규모 이미지-텍스트 데이터셋에서 실험한 결과, 제안한 증강 기법이 기존 방어 방법보다 우수한 성능을 나타냈습니다. 특히, 우리의 N:N-CoA 프레임워크는 다양한 이미지-텍스트 쌍을 생성하여 VL 모델의 적대적 견고성을 크게 향상시켰습니다.



### Cognitive Evolutionary Learning to Select Feature Interactions for Recommender Systems (https://arxiv.org/abs/2405.18708)
- **What's New**: 이 논문에서는 FEATURE INTERACTION(피처 상호작용) 선택 문제를 해결하기 위한 새로운 프레임워크인 Cognitive EvoLutionary Learning (CELL)을 소개합니다. 이는 자연적 생명체의 진화와 기능에서 영감을 얻어 개발된 방법으로써, 태스크 지침 하에서 적절한 OPERATIONS(연산), FEATURES(피처), INTERACTIONS(상호작용)을 선택할 수 있도록 모델을 적응적으로 진화시키려는 목적을 갖고 있습니다.

- **Technical Details**: CELL 프레임워크는 세 가지 주요 단계로 구성됩니다: 1) DNA 탐색 - 피처 쌍의 상호작용을 생성할 적합한 연산을 탐색, 2) 유전체(genome) 탐색 - 기여가 적은 피처와 상호작용을 약화시켜 잡음을 제거, 3) 모델 기능화 - 선택된 피처와 상호작용을 사용하여 비선형 상호작용을 캡처. 이러한 단계를 통해 모델의 적합성을 진단하고, 이를 그래디언트 디센트(gradient descent)로 최적화합니다.

- **Performance Highlights**: 4개의 실제 데이터셋(세 개는 광고 CTR 예측을 위한 공개 데이터셋, 하나는 금융 데이터셋)에서 실험을 수행해 CELL의 성능을 평가했습니다. 그 결과 CELL은 최신 기법들보다 뛰어난 성능을 보였습니다. 또한 가상 실험을 통해 CELL이 사전 정의된 상호작용 패턴을 지속적으로 발견할 수 있음을 확인했습니다.



### Potential Field Based Deep Metric Learning (https://arxiv.org/abs/2405.18560)
- **What's New**: 이번 연구에서는 기존의 튜플 기반 접근방식 대신 전기장을 모티브로 한 지속적인 포텐셜 필드(potential field) 표현을 사용하는 컴포지셔널 딥 메트릭 러닝 모델(PFML)을 제안합니다. PFML은 이미지 임베딩 간 상호작용을 동일/다른 클래스의 매력을 끌어들이거나 밀어내는 포텐셜 필드를 통해 표현하며, 이러한 필드가 거리에 따라 감소하는 특성을 활용합니다. 이는 라벨 노이즈와 큰 클래스 내 변이성을 가진 실제 데이터셋에서 성능을 향상시킵니다.

- **Technical Details**: PFML은 각각의 데이터 샘플을 전기적 장의 전하(charge)로 간주하여 반복적인 필드 생성을 통해 글로벌 포텐셜 필드를 형성합니다. 이 필드는 이미지를 나타내는 임베딩 간 상호작용을 모델링하며, 거리가 멀어질수록 영향을 감소시키는 특징을 가집니다. 또한, 프록시 기반 방법과 유사하게 클래스의 서브 집합을 대표하는 프록시를 사용하여 효율적으로 학습합니다.

- **Performance Highlights**: PFML은 Cars-196, CUB-200-2011, SOP 데이터셋에서 기존 최첨단 기법을 능가하는 성능을 보였습니다. 특히 라벨 노이즈가 있는 상황에서 큰 성능 향상을 보였으며, R@1 기준으로 7% 이상의 개선을 이루었습니다.




### A Short Review for Ontology Learning from Text: Stride from Shallow  Learning, Deep Learning to Large Language Models Trend (https://arxiv.org/abs/2404.14991)
- **What's New**: 이 논문은 텍스트에서 온톨로지 (Ontology)를 학습하는 방법들에 대한 새로운 접근방식을 제시합니다. 이전에는 주로 얕은 학습(Shallow Learning)과 깊은 학습(Deep Learning) 방법들이 사용되었지만, 최근에는 대규모 언어 모델(Large Language Models)을 활용하여 온톨로지 학습을 개선하는 새로운 추세가 두드러집니다.

- **Technical Details**: 이 논문은 온톨로지 학습에서 얕은 학습 기반 방법과 깊은 학습 기반 방법의 방법론과 한계를 분석하고, 대규모 언어 모델을 활용한 온톨로지 학습의 최전선 작업에 대한 포괄적 지식을 제공합니다. 또한, 대규모 언어 모델과 온톨로지 학습 과제의 통합을 더 깊이 탐구하기 위한 몇 가지 주목할만한 미래 방향을 제안합니다.

- **Performance Highlights**: 대규모 언어 모델을 이용한 접근법은 기존의 얕은 학습이나 깊은 학습 접근법에 비해 지식 추출과 표현에서 유의미한 개선을 보였습니다. 특히, 이러한 모델들은 복잡하고 다양한 데이터 소스에서 보다 정교하고 정확한 온톨로지를 구축할 수 있는 잠재력을 가지고 있습니다.



### Cross-Domain Causal Preference Learning for Out-of-Distribution  Recommendation (https://arxiv.org/abs/2404.14856)
Comments: 16 pages, 5 figures, accepted by DASFAA2024

- **What's New**: 이 연구에서는 사용자의 특성 변화로 인해 발생하는 훈련 데이터와 테스트 데이터 간의 데이터 분포 이동 문제를 해결하기 위한 새로운 추천 시스템 모델인 Cross-Domain Causal Preference Learning for Out-of-Distribution Recommendation (CDCOR)을 제안합니다. 이 모델은 도메인 간 adversarial network (Adversarial Network)를 사용하여 사용자의 도메인 공유 선호도를 발견하고 인과 구조 학습기를 사용하여 약한 데이터 상황과 분포 이탈 환경에서 OOD(Out-of-Distribution) 문제를 다룰 수 있는 인과 불변성을 포착합니다.

- **Technical Details**: CDCOR 모델은 사용자의 도메인-공유 선호를 파악하기 위해 도메인 adversarial 네트워크를 활용하며, 인과 구조 학습기를 통해 데이터의 인과적 관계를 학습하여 OOD 일반화 능력을 강화합니다. 이는 전통적인 추천 시스템이보다 깊은 데이터의 이해와 분석을 가능하게 하여 더욱 정확하고 일관된 추천을 가능하게 합니다.

- **Performance Highlights**: 실제 세계 데이터셋 2개를 사용한 실험을 통해 CDCOR 모델이 데이터 희소성과 분포 이탈 환경에서 뛰어난 성능을 보여줬습니다. 또한 이 모델은 OOD 일반화에 있어 벤치마크 모델들을 뛰어넘는 성능을 보여주며, 이는 다양한 시나리오에서의 활용 가능성을 시사합니다.



### Contrastive Quantization based Semantic Code for Generative  Recommendation (https://arxiv.org/abs/2404.14774)
- **What's New**: 이 논문에서는 추천 시스템에서 품목 관계와 의미 정보를 동시에 고려하여 품목의 코드 표현을 구성하는 새로운 방법을 제안합니다. 특히, 텍스트 설명에서 항목의 임베딩을 추출하기 위해 사전 훈련된 언어 모델을 사용하고, 항목 코드를 학습하기 위해 대조적 목표를 가진 인코더-디코더 기반 RQVAE 모델을 강화합니다.

- **Technical Details**: 첫 번째 단계에서는 Sentence-T5와 BERT와 같은 일반적인 사전 훈련된 텍스트 인코더를 사용하여 품목의 텍스트 설명을 임베딩으로 변환합니다. 그 다음 RQVAE(Residual-Quantized Variational AutoEncoder) 모델을 사용하여 이 임베딩을 코드로 변환합니다. RQVAE는 다단계 벡터 양자화기로, 다양한 단계에서 잔차를 양자화하여 코드 튜플을 생성합니다. 이 모델은 인코더, 디코더, 코드북을 동시에 훈련시켜 입력 데이터를 간단히 재구성합니다.

- **Performance Highlights**: 실험에서 이 방법은 MIND 데이터셋에서 NDCG@5를 43.76% 개선했으며, Office 데이터셋에서 Recall@10을 80.95% 개선했습니다. 이는 이전 기준선에 비해 상당한 향상을 보여주며, 품목의 관계를 고려하는 것이 추천 시스템에서의 품목 코드의 질을 높이는 데 중요함을 시사합니다.



### Manipulating Recommender Systems: A Survey of Poisoning Attacks and  Countermeasures (https://arxiv.org/abs/2404.14942)
- **What's New**: 이 연구는 기존의 공격과 그 탐지 방법에 초점을 맞춘 이전 조사와 달리, 추천 시스템에 대한 독이 공격(poisoning attacks)과 이러한 공격을 방어하는 대책에 중점을 둡니다. 이 연구는 30+가지의 다양한 공격과 40+가지의 대책을 체계적으로 분류하고 평가하여, 특정 유형의 공격에 대한 방어책의 효과를 평가합니다.

- **Technical Details**: 이 연구에서는 협업 필터링(collaborative filtering) 기반 추천 시스템에 초점을 맞추고 있으며, 메모리 기반(memory-based)과 모델 기반(model-based) CF 시스템으로 나누어 설명합니다. 여기서 AI 기반 공격과 고전적 휴리스틱 공격(classic heuristic attacks)을 분석하며, 각각의 공격 유형에 대응하기 위한 다양한 대책을 제공합니다.

- **Performance Highlights**: 제시된 보안 대책은 특정 타입의 독이 공격에 효과적임을 보여주며, 이는 추천 시스템을 보호하는 데 있어 중요한 기준점을 제시합니다. 또한, 이 연구는 공격에 대항하는 방어책의 능력을 평가하고, 추천 시스템에 대한 독이 공격의 위험성과 대응 전략의 성공 가능성을 평가하는 데 있어서 확실한 지침을 제공합니다.



### Multi-Sample Dynamic Time Warping for Few-Shot Keyword Spotting (https://arxiv.org/abs/2404.14903)
- **What's New**: 새롭게 제안된 'multi-sample dynamic time warping' 기법은 키워드 감지(keyword spotting)를 위해 각 클래스별 비용 텐서(cost-tensors)를 계산하는 방법입니다. 이 방법은 각 쿼리 샘플의 변동성(variability)을 포함하여 보다 정확한 감지 성능을 제공합니다.

- **Technical Details**: 이 연구에서는 전통적인 동적 시간 왜곡(dynamic time warping, DTW)에 기반하여 각 키워드 클래스별로 다수의 샘플을 활용하는 기법을 개선하였습니다. 특히, 클래스별로 생성된 비용 텐서는 추론 단계에서 비용 행렬(cost matrices)로 변환되어 계산 복잡성을 대폭 줄입니다. 이는 Fréchet 평균(Fréchet means)을 사용할 때와 비교해 성능은 유사하면서도 처리 시간은 크게 개선되었습니다.

- **Performance Highlights**: 실험적 평가에서, 이 방법은 적은 수의 학습샘플(few-shot learning)을 사용하는 키워드 스팟팅에서 개별 쿼리 샘플을 사용했을 때와 매우 유사한 성능을 보여주었습니다. 또한, Fréchet 평균을 사용했을 때보다는 약간 느리지만, 훨씬 빠른 처리 속도를 갖습니다.



### Anchor-aware Deep Metric Learning for Audio-visual Retrieva (https://arxiv.org/abs/2404.13789)
Comments: 9 pages, 5 figures. Accepted by ACM ICMR 2024

- **What's New**: 본 논문에서는 기존 데이터 포인트 간의 내재된 상관 관계를 확인하고 학습 데이터의 부족으로 인한 임베딩 공간 학습의 불완전성을 보완하기 위해 Anchor-aware Deep Metric Learning (AADML) 방법을 제안합니다. 이를 통해 유사한 샘플 간의 상호작용을 더욱 효과적으로 캡쳐하면서 dissimilar(비유사) 샘플의 간섭을 감소시키는 새로운 접근 방식을 소개합니다.

- **Technical Details**: AADML은 각 샘플을 '앵커'로 고려하고 의미론적으로 유사한 샘플들과의 종속성을 고려하여 correlation graph(상관 관계 그래프) 기반의 manifold 구조를 구축합니다. 이 구조 내에서 attention-driven mechanism(주의 기반 메커니즘)을 사용하여 동적 가중치를 적용함으로써 Anchor Awareness (AA) 점수를 생성하고, 이 점수들을 metric learning(메트릭 학습) 방법에서 상대적 거리를 계산하는 데이터 프록시로 사용합니다.

- **Performance Highlights**: AADML 방법은 두 가지 audio-visual benchmark datasets를 사용한 실험에서 기존의 최신 모델들을 크게 능가하는 성능을 보였습니다. 특히, large dataset VEGAS에서는 mean average precision (MAP) 기준으로 3.0% 향상되었고, small dataset AVE에서는 45.6% 향상되어 상당한 성과를 달성했습니다. 또한 다양한 메트릭 학습 방법에 AA 프록시를 통합하여 실험한 결과, 기존 손실 방법들보다 더 우수한 성능을 보여 주었습니다.




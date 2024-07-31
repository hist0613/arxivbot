### Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse  Representations (https://arxiv.org/abs/2404.18812)
- **What's New**: 이 연구는 학습된 희소 표현(Sparse Representations) 위에 빠르면서도 효과적인 근사 검색(approximate retrieval)을 가능하게 하는 새로운 역 인덱스(inverted index) 구조를 제안합니다. 이는 NeurIPS 2023의 BigANN Challenge에서 주목받은 과제로, Seismic라는 새로운 방법론을 도입하여 MS MARCO 데이터셋의 다양한 희소 임베딩에서 매우 빠른 검색 속도와 높은 재현율(recall)을 달성했습니다.

- **Technical Details**: 제안된 Seismic 방법은 역 리스트(inverted lists)를 기하학적으로 일관된 블록으로 구성하고 각 블록에 요약 벡터(summary vector)를 장착합니다. 쿼리 처리 시, 요약 벡터를 사용하여 블록을 평가해야 하는지 빠르게 판단합니다. 이 구조는 학습된 희소 임베딩의 특성에 맞춘 맞춤형 검색 알고리즘 설계 요구를 충족합니다.

- **Performance Highlights**: Seismic는 단일 스레드(single-threaded) 쿼리 처리를 통해 MS MARCO 데이터셋에서 쿼리 당 수밀리초(sub-millisecond) 이하의 지연 시간(latency)을 달성하며, 상태 기술의 역 인덱스 기반 솔루션보다 한두 자릿수 빠른 성능을 보여줍니다. 또한, NeurIPS 2023 BigANN Challenge에서 우승한 그래프 기반(graph-based) 제출물보다 큰 폭으로 성능이 우수함을 실험적으로 입증하였습니다.



### Efficiency-Effectiveness Tradeoff of Probabilistic Structured Queries  for Cross-Language Information Retrieva (https://arxiv.org/abs/2404.18797)
Comments: 11 pages, 5 figures

- **What's New**: 이번 연구는 PSQ(Probabilistic Structured Queries)의 효율적인 Python 구현을 소개하며, 크로스 언어 정보 검색(Cross-Language Information Retrieval, CLIR)에서의 성능과 효율성 사이의 최적의 트레이드오프를 달성하기 위해 다중 기준 가지치기(multi-criteria pruning)를 탐구합니다. 이전 연구들에서 완전히 탐색되지 않았던 이 다중 기준 가지치기는 PSQ의 유효성과 효율성을 모두 개선할 가능성이 있습니다.

- **Technical Details**: PSQ는 통계적으로 정렬된 코퍼스에서 유도된 번역 확률을 사용하는 CLIR 방법입니다. PSQ의 효과는 번역 확률이 어떻게 가지치기(pruning)되는지에 크게 의존하며, 모든 번역 확률을 사용하면 역 인덱스(inverted index)를 사용한 효율적인 쿼리 서비스가 불가능해집니다. 이번 연구에서는 모던 CLIR 테스트 컬렉션을 사용해 PSQ의 다양한 가지치기 전략(Probability Mass Function(PMF), Cumulative Distribution Function(CDF), Top-k threshold)의 조합을 실험적으로 탐구하고 있습니다.

- **Performance Highlights**: 이 구현은 대규모 CLIR 평가 컬렉션에서 테스트되었으며, 더 많고 깨끗한 병렬 텍스트를 사용하여 인덱싱 시간의 PSQ-HMM 검색 모델을 재구현했습니다. 추가적인 분석을 통해 가지치기 기술의 조합이 Pareto 최적성을 달성하는 데 얼마나 도움이 되는지를 평가했습니다. 결과적으로, 새로운 구현은 효율성과 유효성 사이의 트레이드오프를 효과적으로 관리하며 균형 잡힌 성능을 제공함을 보여줍니다.



### Going Beyond Popularity and Positivity Bias: Correcting for  Multifactorial Bias in Recommender Systems (https://arxiv.org/abs/2404.18640)
Comments: SIGIR 2024

- **What's New**: 이 연구에서는 추천 시스템(Recommender Systems, RSs)에서 사용자 상호작용 데이터의 편향성을 다루며, 특히 인기도 편향(popularity bias)과 긍정성 편향(positivity bias)의 다요소적(multifactorial) 형태를 고려합니다. 기존의 방법들이 단일 요소 편향만을 고려한 반면, 이 연구는 아이템과 평가 값 모두에 의해 영향 받는 선택 편향을 탐구하고 이를 해결하기 위한 기법을 제안합니다.

- **Technical Details**: 연구팀은 더 많은 데이터가 필요한 다요소적 편향의 정확한 추정 문제를 해결하기 위하여 스무딩(smoothing) 기법과 교대 경사 하강법(alternating gradient descent) 기술을 도입합니다. 이 방법들은 편향 수정의 분산을 줄이고 최적화의 견고성을 향상시키는데 기여합니다.

- **Performance Highlights**: 제안된 기술을 사용함으로써, 실험 결과 다요소적 편향 보정은 실제 데이터셋과 합성 데이터셋에서 단일 요소 방법보다 더 효과적이고 견고한 것으로 나타났습니다. 이는 복합적인 선택 편향을 해결하는 데 있어서의 상당한 진전을 의미합니다.



### ir_explain: a Python Library of Explainable IR Methods (https://arxiv.org/abs/2404.18546)
- **What's New**: 최근 신경 순위 매기기 모델(Neural Ranking Models)의 발전은 전통적인 통계 검색 모델(statistical retrieval models)보다 많은 향상을 이루었습니다. 그러나 이러한 큰 신경 구조와 복잡한 언어 모델(large neural architectures and complex language models)의 사용은 정보 검색(IR, Information Retrieval) 방법의 투명성을 감소시켰습니다. 그 결과, 설명 가능성과 해석 가능성(Explainability and Interpretability)이 IR 연구에서 중요한 주제로 부각되었습니다.

- **Technical Details**: 이 논문은 	extit{irexplain}, 정보 검색 설명 가능성(Explainable IR, ExIR)을 위한 다양한 기법을 구현한 오픈 소스 파이썬 라이브러리에 대해 소개합니다. 	extit{irexplain}은 점별(pointwise), 쌍별(pairwise), 목록별(listwise) 설명과 같은 세 가지 표준 후행 설명(post-hoc explanations) 범주를 지원합니다. 라이브러리는 표준 테스트 컬렉션에서 최신 ExIR 기준을 재현하기 쉽게 설계되었고 IR 모델 및 방법을 설명하는 새로운 접근 방식을 탐색하는 것을 용이하게 합니다.

- **Performance Highlights**: 	extit{irexplain}은 Pyserini 및 	extit{irdatasets}과 같은 널리 사용되는 툴킷과 잘 통합되어있으므로, 이 라이브러리의 도입을 촉진합니다. 이를 통해 사용자는 ExIR을 적용하여 IR 모델의 결과를 보다 투명하게 해석할 수 있게 도움을 줍니다.



### OAEI Machine Learning Dataset for Online Model Generation (https://arxiv.org/abs/2404.18542)
Comments: accepted as ESWC 2024 Poster

- **What's New**: 이 논문에서는 온톨로지(ontology)와 지식 그래프(knowledge graph) 매칭 시스템을 평가하기 위해 OAEI(Ontology Alignment Evaluation Initiative)에 의해 매년 실시되는 점검이 소개되며, 대규모 언어 모델을 포함한 기계 학습(machine learning)-기반 접근 방식이 점점 더 많이 사용되고 있음을 다룹니다. 이 논문은 모든 OAEI 트랙에 대한 트레이닝(training), 검증(validation), 테스트(test) 세트를 포함하는 데이터셋을 소개함으로써, 기계 학습 기반 시스템의 공정한 비교를 가능하게 하는 온라인 모델 학습(online model learning)을 가능하게 합니다.

- **Technical Details**: 기존 시스템은 개발자가 결정하는 트레이닝과 검증 데이터셋을 사용하며, 종종 참조 정렬(reference alignments)의 일부를 사용합니다. 이러한 샘플링은 OAEI 규칙에 어긋나며 공정한 비교를 불가능하게 합니다. 그러나 제안된 데이터셋은 온라인에서 모델 학습을 지원하여, 시스템이 사람의 개입 없이 주어진 입력 정렬에 적응할 수 있도록 합니다.

- **Performance Highlights**: 이 데이터셋의 유용성은 인기 있는 시스템의 신뢰도 임계값(confidence thresholds)을 미세 조정(fine-tuning)함으로써 입증되었습니다. 이는 ML(기계 학습)-기반 시스템의 성능을 향상시킬 수 있으며, 더 공정하고 정확한 시스템 평가를 가능하게 합니다.



### M3oE: Multi-Domain Multi-Task Mixture-of Experts Recommendation  Framework (https://arxiv.org/abs/2404.18465)
- **What's New**: 이 연구에서는 다양한 도메인(domain)과 과제(task) 간의 복잡한 의존성을 해결하기 위해 M3oE라는 새로운 적응형 다중 도메인, 다양한 과제를 수행할 수 있는 추천 시스템을 소개하고 있습니다. M3oE는 여러 도메인의 정보를 통합하고, 도메인과 과제 간 지식을 매핑(mapping)하며, 다중 목표를 최적화하는 첫 번째 자기적응(self-adaptive) 시스템입니다.

- **Technical Details**: M3oE 프레임워크는 세 가지 전문가 모듈(mixture-of-experts modules)을 활용하여 일반적인 사용자 선호도, 도메인 관점의 사용자 선호도, 과제 관점의 사용자 선호도를 각각 학습합니다. 이를 통해 여러 도메인과 과제 간 복잡한 의존성을 분리하여 처리할 수 있습니다. 또한, 다양한 도메인과 과제를 걸쳐 특징 추출(feature extraction)과 융합(fusion)을 정밀하게 제어할 수 있는 이중 레벨 융합 메커니즘을 설계했습니다. 자동 기계 학습(AutoML) 기술을 적용하여 프레임워크의 유연성을 더욱 강화하였으며, 이는 동적 구조 최적화(dynamic structure optimization)를 가능하게 합니다.

- **Performance Highlights**: M3oE는 두 개의 벤치마크 데이터셋을 사용하여 테스트를 수행하였고, 다양한 기준 모델(baselines)과 비교하여 우수한 성능을 보였습니다. 이 연구는 추천 시스템의 적응성과 성능을 획기적으로 개선하였으며, 구현 코드도 공개되어 재현성을 보장합니다.



### PromptReps: Prompting Large Language Models to Generate Dense and Sparse  Representations for Zero-Shot Document Retrieva (https://arxiv.org/abs/2404.18424)
- **What's New**: 이 논문에서 제안하는 PromptReps는 훈련이 필요 없고 전체 말뭉치(corpus)에서 문서를 검색할 수 있는 장점이 있습니다. 이 방법은 대규모 언어 모델(LLMs)에 프롬프트를 사용하여 효과적인 문서 검색을 위한 쿼리와 문서 표현을 생성하도록 유도합니다.

- **Technical Details**: PromptReps는 LLM을 프롬프트하여 주어진 텍스트를 하나의 단어로 표현하도록 하며, 다음 토큰의 예측과 관련된 마지막 토큰의 숨겨진 상태(hidden states)와 로짓(logits)을 사용합니다. 이는 밀집 텍스트 임베딩(dense text embedding)과 LLM이 제공하는 희소 단어 봉투(sparse bag-of-words) 표현을 모두 활용하는 하이브리드 문서 검색 시스템을 구성합니다.

- **Performance Highlights**: BEIR zero-shot 문서 검색 데이터셋에 대한 실험 평가에서, 이 간단한 프롬프트 기반 LLM 검색 방법은 대규모 비감독 데이터를 사용하여 훈련된 최신 LLM 임베딩 방법보다 유사하거나 더 높은 검색 효능(retrieval effectiveness)을 달성할 수 있음을 보여줍니다, 특히 더 큰 LLM을 사용할 때 더욱 효과적입니다.



### User Welfare Optimization in Recommender Systems with Competing Content  Creators (https://arxiv.org/abs/2404.18319)
- **What's New**: 이 연구에서는 콘텐츠 제작자들이 온라인 콘텐츠 추천 플랫폼에서 생성하는 수익에 의존하고 경쟁함에 따라, 플랫픔이 사용자 선호도(preferences) 분포 정보를 활용하여 제작자에게 정확한 신호를 보내도록 하는 새로운 알고리즘적 해결책을 제안합니다. 이는 콘텐츠 제작자들이 보다 광범위한 사용자 집단에게 관련 콘텐츠를 제공하도록 장려하며, 궁극적으로 플랫폼 상에서의 사용자 복리(user welfare)를 최적화하는데 목표를 둡니다.

- **Technical Details**: 제안된 시스템은 경쟁 게임 설정(competitive game setting) 하에서 시스템 측의 사용자 복리 최적화를 수행합니다. 플랫폼은 각 사용자의 콘텐츠 만족도에 기반하여 동적으로 가중치(weights)를 계산하는 알고리즘을 도입하고, 이 가중치를 활용하여 추천 정책(recommendation policy)이나 추천 후 보상(post-recommendation rewards)을 조정하는 메커니즘을 설계합니다. 이러한 메커니즘은 제작자의 콘텐츠 생산 전략에 영향을 줍니다.

- **Performance Highlights**: 연구팀은 다양한 데이터셋을 사용한 오프라인 실험(offline experiments) 및 주요 단편 비디오 추천 플랫폼에서 실시한 3주 간의 온라인 실험(online experiment)을 통해 제안된 방법의 효과를 검증하였습니다. 실험 결과는 플랫폼의 개입 없이는 제작자의 전략이 최적이 아닌 상태로 수렴하는 부정적인 예시(negative example)를 확인시켜주며, 제안된 개입 메커니즘이 콘텐츠 제작 전략에 긍정적인 변화를 유도하는 것을 입증했습니다.



### Retrieval-Oriented Knowledge for Click-Through Rate Prediction (https://arxiv.org/abs/2404.18304)
- **What's New**: 이 논문에서는 개인 맞춤 추천에서 중요한 역할을 하는 클릭률(CTR, Click-through rate) 예측을 위해 새로운 접근 방식을 제안합니다. 기존에는 샘플 수준 검색 기반 모델(RIM 등)이 뛰어난 성능을 보였지만, 추론 단계에서의 비효율성으로 인해 산업적 적용이 제한적이었습니다. 이를 해결하기 위해, 검색 지향 지식(Retrieval-Oriented Knowledge, ROK) 프레임워크를 제안합니다.

- **Technical Details**: ROK 프레임워크는 지식 베이스를 중심으로 구성됩니다. 이 지식 베이스는 검색 지향 임베딩 층(retrieval-oriented embedding layer)과 지식 인코더(knowledge encoder)로 이루어져 있으며, 검색 및 집계된 표현을 보존하고 모방합니다. 지식 베이스 최적화를 위해 지식 증류(knowledge distillation) 및 대조 학습(contrastive learning) 방법이 사용됩니다. 학습된 검색 강화 표현(retrieval-enhanced representations)은 임의의 CTR 모델과 인스턴스 및 특징 수준(instance-wise and feature-wise)으로 통합될 수 있습니다.

- **Performance Highlights**: 세 개의 대규모 데이터셋에서 실시한 광범위한 실험을 통해 ROK는 검색 기반 CTR 모델과 경쟁적인 성능을 보이면서도, 뛰어난 추론 효율성과 모델 호환성을 제공합니다.



### Contrastive Learning Method for Sequential Recommendation based on  Multi-Intention Disentanglemen (https://arxiv.org/abs/2404.18214)
- **What's New**: 이 연구에서 소개된 MIDCL (Multi-Intention Disentanglement Contrastive Learning)은 사용자의 다양하고 동적인 의도를 분리하여 순차적 추천 시스템에서의 정확성을 향상시키는 새로운 접근 방법입니다. MIDCL은 사용자의 다중 의도를 파악하고 이를 효과적으로 분리할 수 있는, 대조학습(Contrastive Learning)과 변분 자동 인코더(Variational Auto-Encoder, VAE) 기반 방법을 결합했습니다.

- **Technical Details**: MIDCL은 변분 자동 인코더를 사용하여 사용자의 다중 의도를 해석하고 분리합니다. 또한, 두 가지 유형의 대조학습 패러다임을 제안하여 가장 관련성 높은 사용자의 상호작용 의도를 찾고, 긍정적 샘플 쌍의 상호정보(mutual information)를 최대화합니다. 이를 통해 관련 없는 의도의 영향을 감소시키면서도 각 사용자에 대한 가장 관련성 높은 의도를 추출할 수 있게 합니다.

- **Performance Highlights**: 실험 결과들은 MIDCL이 기존의 대부분의 기준 모델들보다 뚜렷한 우위를 보이며, 의도 기반 예측 및 추천에 대한 연구에서 더 해석 가능한 사례를 제시한다는 것을 보여줍니다.



### Behavior-Contextualized Item Preference Modeling for Multi-Behavior  Recommendation (https://arxiv.org/abs/2404.18166)
Comments: This paper has been accepted by SIGIR 2024

- **What's New**: 이 논문은 추천 시스템에서 데이터 희소성(data sparsity)과 같은 문제를 해결하기 위해 다중 행동 방식(multi-behavior methods)의 효과를 입증합니다. 특히, Behavior-Contextualized Item Preference Modeling (BCIPM)이라는 새로운 접근 방식을 소개하여, 다양한 부수적 행동(auxiliary behaviors)에서 발생할 수 있는 노이즈(noise)를 효과적으로 줄이면서 사용자의 특정 아이템 선호도를 학습합니다.

- **Technical Details**: BCIPM 방식은 각 행동(behavior)내에서 사용자의 아이템 선호도를 파악하고, 그 선호도를 최종 추천 대상(target behavior)에만 적용하여 추천을 정교화합니다. 부수적 행동은 네트워크 매개변수(network parameters)의 학습에만 사용되며, 이는 타겟 행동의 정확도를 보존하면서 학습 과정을 개선합니다. 초기 임베딩(embeddings)을 사전 훈련(pre-training)하는 전략도 채택하여, 타겟 행동과 관련된 데이터가 부족한 상황에서 아이템 별 선호도를 더욱 풍부하게 합니다.

- **Performance Highlights**: 네 가지 실제 데이터 셋(real-world datasets)에서 수행한 실험은 BCIPM이 여러 선진 모델(leading state-of-the-art models)보다 우수한 성능을 보였음을 입증합니다. 이로써 BCIPM의 효율성과 강건함(robustness)이 입증되었습니다.



### Towards Robust Recommendation: A Review and an Adversarial Robustness  Evaluation Library (https://arxiv.org/abs/2404.17844)
- **What's New**: 이 논문은 추천 시스템의 견고함(robustness)에 대한 종합적인 조사를 제공합니다. 연구자들은 적대적 견고함(adversarial robustness)과 비적대적 견고함(non-adversarial robustness)으로 추천 시스템의 견고함을 분류했습니다. 또한, ShillingREC이라는 새로운 적대적 견고성 평가 라이브러리(adversarial robustness evaluation library)를 제안하고 기본 공격 모델(attack models)과 추천 모델(recommendation models)을 평가하였습니다.

- **Technical Details**: 이 연구는 추천 시스템을 대상으로 한 적대적 공격(adversarial attacks)과 방어(defenses)의 기본 원리와 고전적 방법을 소개합니다. 비적대적 견고함은 훈련 데이터의 자연 잡음(natural noise), 데이터 희소성(data sparsity), 데이터 불균형(data imbalance) 관점에서 분석됩니다. 연구자들은 견고성 평가에 사용되는 일반적인 데이터 세트(datasets)와 평가 지표(evaluation metrics)를 요약하며, 현재 도전과제와 미래 연구 방향에 대해 논의합니다.

- **Performance Highlights**: ShillingREC 프로젝트는 적대적 견고성 평가를 위해 공개되었으며, 공격 모델과 추천 모델의 공정하고 효율적인 평가를 가능하게 합니다. 이를 통해 연구자들은 추천 시스템이 적대적 공격과 비적대적 요인에 얼마나 잘 대응하는지 평가할 수 있습니다. 또한, 적대적 방해에 대한 방어능력을 정량화하며, 이러한 시스템의 다양한 측면에서의 견고함을 평가합니다.



### A Taxation Perspective for Fair Re-ranking (https://arxiv.org/abs/2404.17826)
Comments: Accepted in SIGIR 2024

- **What's New**: 이 연구는 공정하게 재순위를 매기는 'Tax-rank'라는 새로운 방법을 소개합니다. 이 방법은 경제학에서 통찰력을 얻어 공정한 재순위 작업을 조세 과정(taxation process)으로 개념화합니다. 연구진은 Tax-rank가 개별 항목들에 대한 세금을 부과함으로써 순위를 변경한다는 개념을 제시합니다. 이 방법은 이전 방법들이 달성하지 못한 '연속성(continuity)'과 '정확도 손실 제어(controllability over accuracy loss)'의 요구사항을 효과적으로 충족시킵니다.

- **Technical Details**: Tax-rank는 항목 간의 유용성 차이에 기반하여 세금을 부과합니다. 이 목표를 최적화하기 위해 최적 전송(optimal transport)에서 Sinkhorn 알고리즘을 활용합니다. 이론적으로, Tax-rank는 연속성과 정확도 손실 제어능력을 모두 갖추고 있음을 보여줍니다. 이는 세금률의 작은 변화가 정확도와 공정성에 큰 변화를 일으키지 않도록 하며, 특정 세금률 하에서의 정확도 손실을 정밀하게 추정할 수 있게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, Tax-rank는 추천 및 광고 작업에서 모든 최신 기술(state-of-the-art) 기준을 능가하는 효과와 효율성을 보여줍니다. 이는 Tax-rank가 공정한 재순위 문제를 해결하기 위한 우수한 세금 정책을 제공한다는 것을 입증합니다.



### Conformal Ranked Retrieva (https://arxiv.org/abs/2404.17769)
Comments: 14 pages, 6 figures, 1 table; 7 supplementary pages, 12 supplementary figures, 2 supplementary tables

- **What's New**: 이 연구에서는 순위 검색(Ranked Retrieval) 문제의 불확실성을 정량화하고 관리하기 위해 새로운 방법인 컨포멀 위험 제어(Conformal Risk Control) 프레임워크를 사용하여 접근하는 방법을 제안합니다. 순위 검색은 문서 또는 다른 데이터 항목을 사용자의 검색 쿼리와 관련성에 따라 순위를 매기는 기술로, 웹 검색, 추천 시스템, 질문-응답 플랫폼 등 다양한 정보 시스템에 널리 적용되고 있습니다.

- **Technical Details**: 이 연구는 검색 단계(L1)와 순위 결정 단계(L2)의 두 단계로 구성된 전형적인 순위 검색 문제에 초점을 맞춥니다. 각 단계에서 발생할 수 있는 위험을 컨포멀 위험으로 정의하고 이를 제어하기 위한 알고리즘을 개발했습니다. 컨포멀 예측(Conformal Prediction)은 머신러닝 분야에서 주목받고 있는 기술로, 예측 집합의 크기를 조정하여 특정 커버리지 요구사항을 충족시키는 방식으로 추론 결과의 불확실성을 측정합니다.

- **Performance Highlights**: 이 방법은 MSLR-WEB, Yahoo LTRC, MS MARCO 등 세 가지 대규모 공개 데이터셋에서 순위 검색 작업을 수행하며 그 효과를 입증했습니다. 개발된 알고리즘은 다단계 위험 제어 문제에 적용 가능하며, 검증된 리스크 경계 내에서 단계별 위험을 제어할 수 있습니다.



### Un análisis bibliométrico de la producción científica acerca del  agrupamiento de trayectorias GPS (https://arxiv.org/abs/2404.17761)
Comments: 16 pages, in Spanish, 8 figures

- **What's New**: 이 논문은 GPS 궤적 클러스터링(GPS trajectory clustering)과 관련된 과학적 생산을 분석하는 것을 목표로 합니다. 이 분야는 지속적으로 발전하고 있으며, 특히 배경 연구(transitional) 알고리즘의 발전, 개선되거나 독창적인 방법들이 과학의 '신규성(novelty)'으로 간주됩니다. 이 연구는 Scopus의 주요 컬렉션에서 총 559개의 논문을 분석하여, 직접적인 관련성이 없는 논문을 제외시키고 연구 주제의 최신 동향을 파악합니다.

- **Technical Details**: GPS 궤적 클러스터링에 대한 과학적 연구는 많은 관심을 받고 있으며, 새로운 기술과 알고리즘의 개발이 중심을 이룹니다. 이 논문에서는 Scopus 데이터베이스에서 수집한 총 559개의 관련 논문을 대상으로 'bibliometrics'(서지학적) 분석을 통해 이 분야의 연구 동향을 파악합니다. 이 연구는 클러스터링 알고리즘의 역동성과 창의성을 보여주는 도구로 bibliometrics을 활용함으로써, 연구 분야의 완전성과 정확성을 높이고자 합니다.

- **Performance Highlights**: 이 연구는 GPS 궤적 클러스터링 분야의 그동안의 연구가 얼마나 활성화되어 있고 그 방향성을 분석하여, 연구자들에게 해당 분야에서 가장 흔히 연구되고 있는 주제와 기술 트렌드를 제공합니다. 이러한 분석을 통해, 연구자들은 더 넓은 관점에서 현재의 연구 동향을 파악하고 미래의 연구 방향을 설정하는 데 도움을 받을 수 있습니다.



### Retrieval-Augmented Generation with Knowledge Graphs for Customer  Service Question Answering (https://arxiv.org/abs/2404.17723)
- **What's New**: 새로운 고객 서비스 질문-응답 방법에서는 전통적인 RAG(Retrieval-Augmented Generation) 방식을 발전시켜, 역사적 이슈들로부터 구축한 지식 그래프(Knowledge Graph, KG)를 이용합니다. 이 접근 방식은 고객 서비스의 구조 정보를 보존하고 텍스트 세분화 문제의 영향을 완화시키면서 검색 정확도와 응답 품질을 향상시킵니다.

- **Technical Details**: 이 방법은 과거 이슈 트래킹 티켓에서 지식 그래프를 구성하여, 각 이슈 내의 구조와 이슈 간의 관계를 유지합니다. 사용자 질의를 분석할 때는 관련된 지식 그래프의 하위 그래프를 검색하여 답변을 생성합니다. 이 프로세스는 인공지능 질문-응답 시스템의 성능을 크게 향상시킨 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 LinkedIn 고객 서비스 팀에서 구현되었으며 약 6개월 동안 사용되었습니다. 해당 방법은 벤치마크 데이터셋에서 Mean Reciprocal Rank(MRR)이 77.6% 향상되었고, BLEU 점수도 0.32만큼 개선되었습니다. 또한, 이 방법은 문제 해결 시간을 중간값 기준으로 28.6% 감소시켰습니다.



### SetCSE: Set Operations using Contrastive Learning of Sentence Embeddings (https://arxiv.org/abs/2404.17606)
- **What's New**: 새롭게 소개된 SetCSE 프레임워크는 집합(set) 이론을 활용하여 복잡한 의미(semantics)를 표현하고 정보 검색(information retrieval)을 수행합니다. 이 프레임워크는 문장 임베딩(sentence embedding) 모델이 주어진 의미를 더 잘 이해할 수 있도록 인터-세트 대조 학습(inter-set contrastive learning) 목표를 도입합니다. 또한, SetCSE 교차(intersection), 차이(difference), 그리고 연산 시리즈(operation series)와 같은 연산 모음을 사용하여 복잡한 문장 검색 작업을 위한 문장 임베딩을 활용합니다.

- **Technical Details**: SetCSE는 문장을 집합으로 표현하고, 복합적인 의미를 이해하고 쿼리하는데 필요한 구조화된 정보 쿼리 기능을 향상시키는 집합-이론적 연산을 통합합니다. 특히, 이 프레임워크는 문장의 임베딩을 통해 의미를 표현하고 복잡한 쿼리를 수행할 수 있는 기능을 제공하여, 기존의 단어 임베딩에 집중되었던 이전 작업들과 차별화됩니다.

- **Performance Highlights**: SetCSE는 인간 언어 표현의 관례를 따르며, 문장 임베딩 모델의 차별화 능력을 크게 향상시킵니다. 복잡하고 정교한 프롬프트(prompts)를 다루는 다양한 정보 검색 작업에서 기존의 쿼리 방법으로는 달성할 수 없었던 성과를 가능하게 합니다.



### Revealing and Utilizing In-group Favoritism for Graph-based  Collaborative Filtering (https://arxiv.org/abs/2404.17598)
Comments: 7 pages, 6 figures

- **What's New**: 새롭게 도입된 Co-Clustering Wrapper (CCW)는 사용자들의 선호와 구매 패턴을 추출하기 위한 개인화된 아이템 추천 시스템입니다. 이 연구에서는 실제 세계의 사용자들이 클러스터를 형성하고 각 클러스터 내에서 공통적인 취향을 가진다고 가정하고, 사용자와 아이템의 공동 클러스터를 계산하여 각 클러스터 내의 선호도를 추출하는 것이 특징입니다.

- **Technical Details**: CCW는 공동 클러스터링 알고리즘을 사용하여 사용자와 아이템의 공동 클러스터를 계산하고, 각 클러스터마다 CF (Collaborative Filtering) 서브네트워크를 추가하여 그룹 내 선호도를 추출합니다. 이를 통해 통합되고 풍부한 사용자 정보를 얻을 수 있습니다. 두 가지 측면을 고려한 실제 데이터셋 실험을 통해 그룹별 선호도에 따라 나뉘는 그룹 수를 찾고 성능 향상의 정도를 측정했습니다.

- **Performance Highlights**: CCW는 실제 세계 데이터셋을 사용하여 그룹 선호도에 따라 나뉘는 그룹 수를 찾고, 성능 향상을 측정하는 두 가지 측면에서 실험을 수행함으로써 추천 시스템의 효과를 입증했습니다.



### KamerRaad: Enhancing Information Retrieval in Belgian National Politics  through Hierarchical Summarization and Conversational Interfaces (https://arxiv.org/abs/2404.17597)
Comments: 4 pages, 2 figures, submitted to 2024 ECML-PKDD demo track

- **What's New**: KamerRaad는 벨기에 정치 정보와 상호 작용하는 시민을 돕기 위해 대규모 언어 모델을 활용하는 AI 도구입니다. 이 도구는 의회 절차에서 중요한 내용을 추출하여 간결하게 요약하고, 생성적 AI(generative AI)를 기반으로 사용자가 지속적으로 이해를 쌓을 수 있는 상호 작용이 가능합니다.

- **Technical Details**: KamerRaad의 프론트엔드(front-end)는 Streamlit을 사용하여 사용자가 쉽게 상호 작용할 수 있도록 구축되어 있고, 백엔드(back-end)는 텍스트 임베딩(text embedding)과 생성을 위한 오픈 소스 모델을 사용하여 정확하고 관련성 높은 응답을 보장합니다.

- **Performance Highlights**: KamerRaad는 사용자 피드백을 수집하여 소스 검색의 관련성과 요약의 질을 향상시키는 것을 목표로 하며, 소스 기반 대화(source-driven dialogue)에 중점을 두어 사용자 경험을 풍부하게 합니다.



### Low-Rank Online Dynamic Assortment with Dual Contextual Information (https://arxiv.org/abs/2404.17592)
- **What's New**: 이 논문에서는 전자 상거래가 확장됨에 따라 막대한 카탈로그에서 실시간으로 개인화된 추천을 제공하는 것이 소매 플랫폼에 있어 중요한 도전이 되고 있습니다. 이를 위해 사용자(user) 및 아이템(item) 특성을 모두 고려하여 동적(dynamic)으로 최적의 상품 집합(assortment)을 결정하는 ‘이중 맥락(dual contexts)’에 대한 동적 상품 집합 문제를 다루고 있습니다.

- **Technical Details**: 저자들은 고차원(high-dimensional) 상황에서 차원의 제곱적 증가로 인해 계산과 추정이 복잡해지는 문제를 해결하기 위해 새로운 저랭크(low-rank) 동적 상품 집합 모델을 소개합니다. 이렇게 함으로써 문제를 관리 가능한 규모로 변환할 수 있었습니다. 또한, 내재된 부공간(intrinsic subspaces)을 추정하고 탐색-활용 상충(exploration-exploitation trade-off)을 다루기 위해 상위 신뢰 경계(upper confidence bound, UCB) 접근 방식을 사용하는 효율적인 알고리즘을 제안합니다.

- **Performance Highlights**: 이론적으로, 제안된 모델과 알고리즘은 비대칭 성측면에서 기존 문헌보다 상당히 개선된 $	ilde{O}((d_1+d_2)r\	ext{	ext{sqrt}}{T})$의 후회 경계(regret bound)를 설립합니다. 여기서 $d_1, d_2$는 각각 사용자 및 아이템 특성의 차원, $r$은 매개변수 행렬의 랭크, $T$는 시간 지평을 나타냅니다. 이러한 결과는 Expedia 호텔 추천 데이터셋에 대한 적용을 포함한 광범위한 시뮬레이션을 통해 확인되며, 제안된 방법의 장점을 추가로 입증합니다.



### Large Language Models for Next Point-of-Interest Recommendation (https://arxiv.org/abs/2404.17591)
- **What's New**: 이 논문은 위치 기반 소셜 네트워크(Location-Based Social Network, LBSN) 데이터의 맥락 정보를 효과적으로 활용하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 사전 훈련된 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 기존의 수치적 한계를 극복하고, LBSN 데이터를 원래 형식으로 보존함으로써 맥락 정보의 손실을 방지합니다.

- **Technical Details**: 제안된 프레임워크는 LBSN 데이터의 다양한 맥락 정보를 원본 형식으로 유지하고, 상식 지식을 포함시켜 맥락 정보의 본질적인 의미를 이해할 수 있게 설계되었습니다. 이를 통해, 사용자의 이전 방문 기록을 바탕으로 다음 방문할 관심 지점(Point of Interest, POI)을 예측하는 과제에 접근합니다.

- **Performance Highlights**: 실제 세계의 세 가지 LBSN 데이터셋에서 프레임워크를 테스트한 결과, 모든 데이터셋에서 기존의 최신 모델들을 뛰어넘는 성능을 보였습니다. 특히, 흔히 발생하는 '콜드 스타트' 문제와 '짧은 궤적' 문제를 완화하는 데에 효과적임이 입증되었습니다.



### Leveraging Intra-modal and Inter-modal Interaction for Multi-Modal  Entity Alignmen (https://arxiv.org/abs/2404.17590)
- **What's New**: MIMEA(Multi-Grained Interaction framework for Multi-Modal Entity Alignment)는 다양한 멀티모달 지식 그래프(MMKGs) 내에서 동등한 엔티티 쌍을 확인하는 멀티모달 엔티티 정렬(Multi-modal entity alignment, MMEA)을 위한 새로운 프레임워크입니다. 이 프레임워크는 다양한 모달리티 간의 상호작용을 다각적으로 접근하여 처리합니다.

- **Technical Details**: MIMEA는 네 가지 주요 모듈로 구성되었습니다: 1) 다중 모달 지식 임베딩(Multi-modal Knowledge Embedding) 모듈은 개별 인코더를 사용하여 모달리티별 표현을 추출합니다; 2) 확률 유도 모달 퓨전(Probability-guided Modal Fusion) 모듈은 단일 모달 표현들을 결합하여 합동 모달 임베딩을 형성하고, 여러 모달리티 간의 상호작용을 촉진합니다; 3) 최적 운송 모달 정렬(Optimal Transport Modal Alignment) 모듈은 최적 운송 메커니즘을 도입하여 단일 모달과 합동 모달 임베딩 사이의 상호작용을 강화합니다; 4) 모달적응형 대조 학습(Modal-adaptive Contrastive Learning) 모듈은 각 모달리티에 대해 동등한 엔티티의 임베딩을 비동등한 것들과 구분합니다.

- **Performance Highlights**: 실제 데이터셋 두 개에서 수행된 광범위한 실험을 통해, MIMEA는 기존 최고성능(SoTA, state-of-the-art) 모델에 비해 우수한 성능을 보였습니다. 특히 멀티모달 지식 그래프에서의 엔티티 정렬 문제를 효과적으로 처리하는 능력이 확인되었습니다.



### An Off-Policy Reinforcement Learning Algorithm Customized for Multi-Task  Fusion in Large-Scale Recommender Systems (https://arxiv.org/abs/2404.17589)
- **What's New**: 이 연구에서는 대규모 추천 시스템(Recommender Systems, RSs)에서 멀티태스크 퓨전(Multi-Task Fusion, MTF)을 최적화하기 위해 새로운 오프-플리시 강화학습(Off-Policy Reinforcement Learning, RL) 알고리즘을 제안합니다. 기존의 알고리즘들이 외부 분포 문제(Out-of-Distribution, OOD)로 제한적이었던 것에 비해, 새로운 도전을 통해 이 문제를 해결하고자 합니다.

- **Technical Details**: 본 논문에서 제안한 RL-MTF 알고리즘은 오프-플리시 강화학습 모델을 자체 개발한 온라인 탐색 정책(online exploration policy)과 통합하여 기존의 제약을 완화시킵니다. 이는 모델 성능을 크게 향상시켰습니다. 또한, 효율적인 탐색 정책을 설계하여 낮은 가치의 탐색 공간을 제거하고 잠재적으로 높은 가치를 가진 상태-행동 쌍(state-action pairs)에 집중하도록 합니다. 점진적 학습 모드(progressive training mode)를 도입하여 탐색 정책의 도움으로 모델의 성능을 추가로 향상시킬 수 있습니다.

- **Performance Highlights**: 오프라인 및 온라인 실험을 통해, 제안한 RL-MTF 모델이 다른 모델에 비해 우수한 성능을 보였고, 텐센트 뉴스의 짧은 비디오 채널에서 약 1년 동안 전면적으로 배포되었습니다. 또한, 이 솔루션은 텐센트의 다른 대규모 RSs에서도 사용되고 있습니다.




### Comparing Personalized Relevance Algorithms for Directed Graphs (https://arxiv.org/abs/2405.02261)
Comments: 4 pages, 1 figure. To appear at 2024 IEEE 40th International Conference on Data Engineering (ICDE)

- **What's New**: 새롭게 소개하는 웹 플랫폼은 방향 그래프(directed graph)에서 주어진 쿼리 노드(query node)와 관련된 가장 중요한 노드들을 식별할 수 있게 해줍니다. PageRank와 Personalized PageRank와 같은 잘 알려진 알고리즘 외에도, Cyclerank라는 새로운 알고리즘을 소개하여 기존 알고리즘의 한계를 극복하고 순환 경로(cyclic paths)를 활용해 개인화된 관련성 점수(personalized relevance scores)를 계산합니다.

- **Technical Details**: 이 데모는 알고리즘 비교(algorithm comparison) 및 데이터셋 비교(dataset comparison) 두 가지 사용 사례를 지원합니다. 사용자는 Wikipedia, Twitter, Amazon에서 제공하는 50개의 사전 로드된 데이터셋(pre-loaded datasets)과 7가지 알고리즘을 사용할 수 있으며, 새로운 데이터셋과 알고리즘을 쉽게 추가할 수 있습니다.

- **Performance Highlights**: 이 플랫폼은 또한 여러 데이터셋 간의 탐색 및 비교를 가능하게 하여 데이터 내의 숨겨진 관계를 밝혀내는 데 효과적입니다. Cyclerank을 포함한 알고리즘은 방향 그래프에서 관련성 점수를 효율적으로 계산함으로써 그래프 분석 알고리즘(graph analysis algorithms)의 보유 목록에 가치 있는 추가를 제공합니다.



### FairEvalLLM. A Comprehensive Framework for Benchmarking Fairness in  Large Language Model Recommender Systems (https://arxiv.org/abs/2405.02219)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, RecLLMs)을 활용한 추천 시스템에서 공정성 평가를 위한 프레임워크를 제시합니다. 이는 다양한 공정성 차원을 아우르는 통합적 접근법을 제공하며, 사용자 특성에 대한 민감성, 내재적 공정성(intrinsic fairness), 그리고 근본적 이익에 근거한 공정성 논의를 포함합니다. 추가적으로, 우리의 프레임워크는 역사적 팩트 평가(counterfactual evaluations)와 다양한 사용자 그룹 고려를 통합하여 RecLLMs의 공정성 평가에 대한 논의를 향상시킵니다.

- **Technical Details**: 이 프레임워크의 주요 기고는 LLM 기반 추천에서 공정성 평가를 위한 견고한 프레임워크의 개발과 인구통계 데이터, 사용자의 역사적 선호도 및 최근 상호작용에서 구성된 정보제공 사용자 프로필(informative user profiles)을 생성하기 위한 체계적 방법을 제안하는 것입니다. 이는 특히 시간에 따라 변화하는 시나리오에서의 개인화 향상에 필수적입니다. LastFM-1K 및 ML-1M 데이터세트에 대한 실제 적용을 통해 우리 프레임워크의 유용성을 입증하였습니다. 각 데이터셋에서 80명의 사용자를 대상으로 50가지 시나리오에 걸친 다양한 프롬프트 구성과 컨텍스트 학습(in-context learning) 실험을 수행했습니다.

- **Performance Highlights**: 이 연구는 4000개 이상의 추천을 생성하였고(80 * 50 = 4000), 민감 속성을 다루는 시나리오에서 중대한 불공정 문제는 없는 것으로 나타났습니다. 그러나 인구통계 그룹 간에 직접적인 민감성을 포함하지 않는 내재적 공정성 측면에서는 여전히 상당한 불공정이 존재합니다.



### How to Diversify any Personalized Recommender? A User-centric  Pre-processing approach (https://arxiv.org/abs/2405.02156)
- **What's New**: 이 연구에서는 추천 시스템의 다양성(diversity)을 향상시키는 새로운 전처리(pre-processing) 접근 방식을 소개합니다. 사용자 프로필의 상호작용을 선택적으로 추가 또는 제거함으로써, 사용자 맞춤형으로 다양한 콘텐츠 범주와 주제에 대한 노출을 늘리는 전략을 사용합니다. 이 방법은 어떠한 추천 시스템 아키텍처와도 원활하게 통합될 수 있는 유연성을 제공합니다.

- **Technical Details**: 이 연구에서는 뉴스와 도서 추천을 위한 두 개의 공개 데이터셋(MIND, GoodBook)에서 다양한 추천 시스템 알고리즘을 테스트하였습니다. 사용자 프로필(user profiles)을 조정하여 추천의 다양성을 높이기 위해 1%에서 10% 사이에서 상호작용을 추가하거나 제거하는 두 가지 변형을 탐구합니다. 또한, 표준 협업 필터링(collaborative filtering)에서 신경망 기반(neural network-based) 추천자까지 다양한 알고리즘과 조합하여 실험하였습니다.

- **Performance Highlights**: 전처리된 데이터를 사용한 훈련은 원래 데이터에서 훈련된 시스템과 비교하여 비슷하거나 개선된 성능을 달성했습니다. 특정 알고리즘에 대해서는 교정(calibration) 메트릭이 일관되게 향상되었으며, 다양성 지표로는 커버리지(coverage)와 지니 지수(Gini index)가 다양한 전처리 수준에서 혼합된 결과를 보였습니다. 또한 전처리 데이터는 공정한 nDCG(fair-nDCG) 점수가 일관되게 높아 소수 또는 틈새 카테고리의 노출 페어니스(exposure fairness)와 더 나은 대표성을 증진시킨다는 것을 나타냈습니다.



### Multi-Objective Recommendation via Multivariate Policy Learning (https://arxiv.org/abs/2405.02141)
- **What's New**: 이 논문에서는 추천 시스템(Recommender Systems)이 다양한 목표를 균형 있게 조정해야 할 필요성에 대해 설명하며, 이를 위해 스칼라화(Scalarisation) 방법을 사용해 각 목표별 보상 신호의 가중 평균을 계산합니다. 특히, 장기 사용자 유지(Long-term user retention)나 성장(Growth) 등의 북스타 보상(North Star Reward)을 최대화하도록 스칼라화 가중치를 결정하는 결정적 과제로 접근하였습니다.

- **Technical Details**: 연구팀은 다변량 연속 행동(Multivariate Continuous Action) 영역에 정책 학습 방법(policy learning methods)을 확장하여 북스타 보상에 대한 비관적 하한(Pessimistic Lower Bound)을 최대화하는 새로운 방법을 제안했습니다. 전통적인 정규 근사(Normal Approximations)가 부족한 범위를 보완하기 위해 효율적이고 효과적인 정책 의존(policy-dependent) 보정법을 제안하였습니다.

- **Performance Highlights**: 이 방식은 시뮬레이션, 오프라인 및 온라인 실험을 통해 검증되었으며, 다양한 실험에서 이 방법이 효과적이라는 실증적 관측 결과를 제공합니다. 스토캐스틱 데이터 수집 정책(Stochastic Data Collection Policies)과 민감한 보상 신호(Sensitive Reward Signals) 설계에 대한 지침도 함께 제공되었습니다.



### Ah, that's the great puzzle: On the Quest of a Holistic Understanding of  the Harms of Recommender Systems on Children (https://arxiv.org/abs/2405.02050)
Comments: 7 pages, 2 figures, DCDW 2024

- **What's New**: 이 논문에서는 온라인 플랫폼에서 어린이들이 접근하는 콘텐츠와 추천 시스템(Recommender Systems, RS)과의 관계에 대해 탐구합니다. 어린이의 발달 단계를 고려할 때 RS가 제공하는 콘텐츠가 적절한지에 대한 의문을 제기하며, 어린이에게 적합한 콘텐츠를 제공하기 위한 전략의 필요성을 강조합니다.

- **Technical Details**: 이 포지션 논문은 RS가 어린이 사용자 그룹에게 노출시킬 수 있는 콘텐츠의 부정적 영향을 중점적으로 다루며, RS의 영향을 더 포괄적으로 이해하기 위한 연구, 실무자, 정책 입안자들의 조사 필요성을 주장합니다. 이러한 조사는 어린이의 요구와 선호에 더 잘 부합하며 잠재적 해를 적극적으로 완화하는 전략을 설계 및 배치하는 데 도움이 될 것입니다.

- **Performance Highlights**: 논문은 RS가 어린이의 복지에 미치는 잠재적 위험을 실토하고, 이에 따른 해결책 제시의 필요성을 강조합니다. 이는 어린이를 위한 안전하고 적합한 온라인 상호 작용을 증진시키는 중요한 표준을 제시할 수 있습니다.



### Comparative Analysis of Retrieval Systems in the Real World (https://arxiv.org/abs/2405.02048)
- **What's New**: 이 연구 논문은 정보 검색(information retrieval)과 자연어 처리(natural language processing, NLP) 분야에서 고급 언어 모델(language models)을 검색 및 검색 시스템(search and retrieval systems)과 통합하는 것을 종합적으로 분석합니다. 최신 기술을 기반으로 다양한 방법들을 정확성(accuracy)과 효율성(efficiency) 면에서 평가하고 비교합니다. Azure Cognitive Search Retriever와 GPT-4, Pinecone의 Canopy 프레임워크, Langchain with Pinecone과 OpenAI 및 Cohere 언어 모델, LlamaIndex와 Weaviate Vector Store의 혼합 검색(hybrid search), Google의 Cloud VertexAI-Search 상의 RAG 구현, Amazon SageMaker의 RAG, 그리고 KG-FID Retrieval이라는 새로운 접근법을 포함하여 다양한 기술 조합을 탐구합니다.

- **Technical Details**: 이 연구는 RobustQA 메트릭을 사용하여 질문의 다양한 재구성(Paraphrasing)에 대해 이러한 시스템들의 성능을 평가합니다. 다양한 도메인에서 강력하고 반응이 빠른 질문-응답 시스템(question-answering systems)에 대한 수요 증가에서 분석 동기를 얻습니다. 각 방법의 장단점을 파악하고 AI 주도의 검색 및 검색 시스템의 배포 및 개발에 있어 정보에 입각한 결정을 내릴 수 있도록 지원하는 것이 보고서의 목표입니다.

- **Performance Highlights**: Azure Cognitive Search와 GPT-4, Pinecone의 Canopy, Google의 RAG, Amazon SageMaker의 RAG 등 다양한 상태의 최첨단 기술들이 정확도와 효율성 측면에서 높은 성능을 보였습니다. KG-FID Retrieval과 같은 새로운 접근법은 특히 독창성 및 구현 가능성 측면에서 주목할 만한 결과를 제시하였습니다. 이러한 시스템들은 특히 복잡한 질문 재구성에 강력한 반응성을 보여 주었습니다.



### Diversity of What? On the Different Conceptualizations of Diversity in  Recommender Systems (https://arxiv.org/abs/2405.02026)
- **What's New**: 이 연구에서는 네덜란드의 세 공공 서비스 미디어 조직의 실무자들이 추천 시스템(Recommender Systems)에서 다양성을 어떻게 개념화하는지 반구조화된 인터뷰를 통해 탐구합니다. 추천 시스템에서 다양성의 목적, 관련 측면, 그리고 추천 내용의 다양화 방법을 개관합니다.

- **Technical Details**: 이 연구는 다양성에 대한 구체적인 해석이 일관성을 가지지 않고 다양하게 표현될 수 있음을 보여줍니다. 저자들은 추천 시스템에서 다양성을 명확히 전달하고 이해하기 위한 효과적인 의사소통의 중요성을 강조하며, 다양성(diversity)이 각 특정 도메인에서 의미하는 바를 명확히 알리는 것이 중요함을 지적합니다.

- **Performance Highlights**: 이 연구는 표준화된 다양성의 개념화가 어려울 것이라는 점을 시사하며, 각기 다른 단체의 요구사항과 뉘앙스를 표현할 수 있는 다양성의 운용 방식에 집중할 것을 제안합니다.



### Robust Explainable Recommendation (https://arxiv.org/abs/2405.01855)
- **What's New**: 본 연구에서는 외부 공격에 견딜 수 있는 기능 중심의 설명 가능한 추천 시스템을 위한 일반적인 프레임워크를 제안합니다. 이 프레임워크는 모델 기반의 화이트 박스(white box) 공격을 받았을 때 전역 설명 가능성(global explainability)을 보존하는 추가적인 방어 도구로 활용될 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 내부 모델 구조와 모델 내 본질적인 유용성(intrinsic utility)에 관계없이 다양한 방법을 지원하며 구현이 간단합니다. 연구팀은 세 가지 인기 있는 전자 상거래 데이터셋에서 구조적으로 다른 두 가지 SOTA 설명 가능 알고리즘을 이용하여 실험하였습니다.

- **Performance Highlights**: 두 알고리즘 모두 정상적인 환경뿐만 아니라 노이즈가 있는 환경에서도 모든 데이터셋에 걸쳐 전역 설명 가능성의 품질과 견고성이 향상되었음을 확인하였습니다. 이는 제안된 프레임워크의 유연성과 변형 가능성을 시사합니다.



### Stability of Explainable Recommendation (https://arxiv.org/abs/2405.01849)
- **What's New**: 이 논문은 추천 시스템(Recommender System, RS)에서 설명 가능한 추천(Explainable Recommendation)의 신뢰성을 연구합니다. 추천과 함께 제공되는 설명의 신뢰성이 다양한 시나리오에서 엄격하게 검증되지 않았기 때문에, 이 연구가 새로운 의미를 가집니다. 또한, 외부 노이즈가 모델 파라미터에 추가될 때의 추천 모델의 취약성을 분석합니다.

- **Technical Details**: 논문에서는 세 가지 주요 SOTA(최신 기법, state-of-the-art) 설명 가능 추천 모델들을 사용하여, 두 가지 규모가 다른 e-commerce 기반 추천 데이터셋에서 실험을 수행했습니다. 외부에서 추가된 다양한 수준의 노이즈(Noise)에 대한 모델의 성능을 체계적으로 분석하고, 특히 적대적 노이즈(Adversarial Noise)의 영향을 집중적으로 평가했습니다.

- **Performance Highlights**: 실험 결과, 모든 설명 가능 모델들은 높아진 노이즈 수준에서 취약함을 보였으며, 특히 적대적 노이즈는 추천의 설명 능력이 크게 감소하는 데 큰 영향을 미쳤습니다. 이는 추천의 설명 능력이 노이즈 수준이 증가함에 따라 감소한다는 가설을 실증적으로 확인시켜 주었습니다.



### RankSHAP: a Gold Standard Feature Attribution Method for the Ranking  Task (https://arxiv.org/abs/2405.01848)
- **What's New**: 이 연구는 랭킹(ranking)을 위한 특성 귀속(feature attribution) 문제에 접근하기 위해 게임 이론(game-theoretic)과 공리적 방법(axiomatic approach)을 도입합니다. 이는 예전에 분류(classification)와 회귀(regression) 작업에 사용된 공리적 특성을 랭킹 문제에 적용하는 것입니다. 이 연구는 Rank-SHAP라는 새로운 알고리즘을 제안하여, 전통적인 Shapley 값에 기반을 두고 랭킹 작업에 맞게 조정된 방식을 소개합니다.

- **Technical Details**: 랭킹을 위한 특성 귀속 방법으로 Rank-SHAP이 도입되었습니다. 이 알고리즘은 Rank-Efficiency, Rank-Missingness, Rank-Symmetry, Rank-Monotonicity라는 네 가지 공리(axioms)를 충족시키며, 이는 기존 Shapley 공리의 변형입니다. Rank-SHAP 계산을 위한 다항 시간(polynomial-time) 알고리즘도 제안되었고, 다양한 시나리오에서의 계산 효율성 및 정확도가 평가되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, Rank-SHAP는 최고의 경쟁 시스템보다 30.78% 높은 fidelity 및 23.68% 높은 wFidelity 성능을 보여 주었습니다. 뿐만 아니라 사용자 연구(user study)를 통해 Rank-SHAP이 최종 사용자(end users)에게 문서를 재정렬하고 쿼리를 추정하는 데 효과적임을 확인할 수 있었습니다.



### A Model-based Multi-Agent Personalized Short-Video Recommender System (https://arxiv.org/abs/2405.01847)
- **What's New**: 이 논문은 산업용 짧은 비디오 추천 시스템을 위한 새로운 강화학습(RL: Reinforcement Learning) 기반 프레임워크를 제안합니다. 이 프레임워크는 사용자의 다양한 측면의 취향을 고려하여 시청 시간을 최대화하는 모델을 개발하고 있습니다. 또한, 샘플 선택 편향(sample selection bias) 문제를 완화하기 위해 모델 기반 학습 접근법(model-based learning approach)을 채택하고 있습니다.

- **Technical Details**: 제안한 프레임워크는 협업 다중 에이전트(collaborative multi-agent) 구성을 사용하여 사용자의 시청 시간을 최대화하는 것을 목표로 합니다. 이는 마르코프 결정 과정(Markov decision process)으로 추천 세션을 모델링하고, 강화 학습을 통해 문제를 해결합니다. 또한, 산업 추천 시스템에서 종종 발생하는 샘플 선택 편향 문제를 해결하기 위해 모델 기반 강화 학습 방법을 적용합니다.

- **Performance Highlights**: 제안된 프레임워크는 광범위한 오프라인 평가(offline evaluations)와 실시간 실험(live experiments)을 거쳐 그 효과가 입증되었습니다. 이는 대규모 실제 짧은 비디오 공유 플랫폼에 배포되어 수 억 명의 사용자에게 성공적으로 서비스되고 있습니다.




### Disentangled Cascaded Graph Convolution Networks for Multi-Behavior  Recommendation (https://arxiv.org/abs/2404.11519)
- **What's New**: 새롭게 제안된 Disentangled Cascaded Graph Convolutional Network (Disen-CGCN)은 다양한 사용자 행동(behavior)을 이용한 추천 시스템에서 데이터 희소성(data sparsity)과 차가운 시작(cold-start) 문제를 해결하기 위해 등장하였습니다. Disen-CGCN은 사용자와 아이템 각각에 대해 독립된 요소로 표현을 분리(disentangled representation)하는 기술을 사용하여, 사용자의 선호도와 아이템의 특성이 각 행동에 맞춰 효과적으로 표현될 수 있도록 합니다.

- **Technical Details**: Disen-CGCN은 LightGCN을 기반으로 개발되었으며, 행동 간 연속적인 관계를 이용하여 특징을 전달하고 분석합니다. 각 행동에 대한 사용자와 아이템의 정보를 특화하여 변환하는 메타 네트워크와 아이템의 다양한 요소에 대한 사용자의 세밀한 선호를 모델링하는 주목 기법(attention mechanism)을 포함하고 있습니다. 이러한 구조를 통해, 다양한 사용자 행동 정보를 바탕으로 특징을 파악하고 최종적으로 사용자 선호에 기반한 추천을 실행합니다.

- **Performance Highlights**: Disen-CGCN은 벤치마크 데이터셋에서 기존의 단일 행동 및 다중 행동 추천 모델들을 평균적으로 7.07% 및 9.00% 이상 개선한 성능을 보여줍니다. 이는 Disen-CGCN이 다중 행동 데이터를 효과적으로 활용하며, 더욱 정확하고 개인 맞춤형(personalized) 추천을 가능하게 함을 강조합니다.



### Deep Pattern Network for Click-Through Rate Prediction (https://arxiv.org/abs/2404.11456)
Comments: 12 pages, 10 figures, accepted by SIGIR2024

- **What's New**: ‘클릭율(Click-through rate, CTR) 예측’ 연구에서 신규로 소개된 ‘Deep Pattern Network (DPN)’은 사용자 행동 패턴을 모델링하여 CTR 예측 성능을 향상시킨다. 기존의 타겟 주의(Target Attention, TA)를 확장한 ‘타겟 패턴 주의(Target Pattern Attention, TPA)’를 통해 패턴 수준의 종속성을 모델링한다.

- **Technical Details**: DPN은 타겟별 관련 사용자 행동 패턴을 효율적으로 검색하는 ‘타겟 인식 패턴 검색 모듈(Target-aware Pattern Retrieval Module, TPRM)’을 도입하여, 불필요한 아이템의 혼입(C1), 데이터의 희소성(C2), 계산 복잡성(C3) 등의 도전 과제를 해결한다. ‘자기 지도 학습(self-supervised learning)’에 기반한 패턴 정제 모듈을 통해 패턴을 효과적으로 정제하고 의존성 학습을 촉진한다.

- **Performance Highlights**: 세 개의 공개 데이터셋에서의 실험을 통해 DPN은 기존 방법들보다 우수한 성능을 보였으며, 다양한 사용자 행동 패턴을 효과적으로 활용함으로써 CTR 예측에서의 광범위한 호환성을 입증하였다.



### Large Language Models meet Collaborative Filtering: An Efficient  All-round LLM-based Recommender System (https://arxiv.org/abs/2404.11343)
Comments: Under review

- **What's New**: 이 논문에서는 추천 시스템에서 공동 필터링(collaborative filtering) 모델과 LLM(Large Language Model)을 통합하는 새로운 접근 방식인 A-LLMRec을 제안하고 있습니다. A-LLMRec은 차가운 시나리오(cold scenario)와 따뜻한 시나리오(warm scenario) 모두에서 탁월한 성능을 발휘합니다.

- **Technical Details**: A-LLMRec은 기존의 최첨단 공동 필터링 추천 시스템(CF-RecSys)에서 사전에 훈련된 사용자/아이템의 임베딩을 LLM에 직접 활용하게 함으로써, LLM의 기능과 CF-RecSys에서 훈련된 고품질의 임베딩을 공유합니다. 특별히, 이 모델은 아이템 임베딩을 LLM의 토큰 공간과 맞추는 정렬 네트워크(alignment network)를 사용하여 CF-RecSys에서 학습된 공동 지식을 LLM에 전달합니다.

- **Performance Highlights**: A-LLMRec은 기존의 CF-RecSys나 다른 LLM 기반 추천 시스템보다 우수하며, 추천 작업(recommendation task) 뿐만 아니라 자연어 처리(natural language processing) 작업에서도 뛰어난 성능을 보여주었습니다. 예를 들어, 사용자가 선호하는 장르를 예측하는 자연어 생성 작업에서 A-LLMRec은 사용자와 아이템에 대한 이해를 바탕으로 효과적인 결과를 생성할 수 있습니다. 또한, A-LLMRec은 사전 튜닝(fine-tuning)이 필요 없으며, LLM과 CF-RecSys의 통합이 용이하여 실행 효율성이 매우 높습니다.



### Causal Deconfounding via Confounder Disentanglement for Dual-Target  Cross-Domain Recommendation (https://arxiv.org/abs/2404.11180)
- **What's New**: 이 연구에서는 이중-대상 교차-영역 추천(CDR)에서 관찰된 단일 영역 및 교차 영역 혼란 요인들을 효과적으로 분리하고 이의 긍정적 및 부정적 영향을 제어하기 위한 인과적 디컨파운딩(Deconfounding) 프레임워크인 CD2CDR을 제안합니다. CD2CDR은 새로운 인과적 디컨파운딩 모듈 및 혼란 요인 분리 모듈을 도입하여 추천의 정확도를 두 영역 모두에 걸쳐 향상시킬 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: CD2CDR 프레임워크는 먼저 각 영역에서 단일 영역 혼란 요인(SDCs)을 디컨플하고, 교차 영역 혼란 요인(CDCs)을 반 백형 회귀(Half-sibling regression)를 통해 분리하는 혼란 요인 분리 모듈을 통해 구현됩니다. 그 다음에는 인과적 디컨파운딩 모듈을 통해 배경 제어(Backdoor adjustment)를 사용하여 이러한 관찰된 혼란 요인의 부정적 영향을 줄이고 긍정적 영향을 추천의 정확도 향상에 활용함으로써 사용자의 종합적인 선호도를 더 정확하게 파악할 수 있습니다.

- **Performance Highlights**: CD2CDR은 다섯 개의 실세계 데이터 세트에 대한 광범위한 실험을 통해 최고 수행 베이스라인을 평균 6.64% (HR@10) 및 8.92% (NDCG@10) 상회하여 성능이 입증되었습니다. 이 결과는 CD2CDR의 인과적 디컨파운딩 접근 방식이 사용자의 진정한 선호도를 보다 정확하게 추론하고 추천의 정확도를 향상시키는 데 매우 효과적임을 시사합니다.



### DRepMRec: A Dual Representation Learning Framework for Multimodal  Recommendation (https://arxiv.org/abs/2404.11119)
Comments: 8 pages, 9 figures

- **newsletter**: [{"What's New": 'DRepMRec, 새로운 이중 표현 학습 프레임워크(Dual Representation learning framework)를 도입하여 멀티모달 추천 시스템에 혁신을 제공합니다. 이 방법은 행동(behavior)과 모달(modal) 정보간에 발생하는 학습 상의 문제들, 특히 상충되는 가이드라인 문제와 표현들의 불일치 문제를 해결합니다.'}, {'Technical Details': 'DRepMRec는 행동과 모달 정보를 별도로 학습하는 두 개의 독립적인 라인을 사용합니다. 행동 정보에 대해서는 LightGCN을 사용하여 사용자와 아이템의 행동 표현을 학습하고, 모달 정보에 대해서는 모달 특정 인코더를 사용합니다. 이후, 행동-모달 정렬 모듈(Behavior-Modal Alignment, BMA)을 통해 두 표현을 정렬하고 통합합니다. 추가적으로, 이 두 표현이 동안 의미적 독립성을 유지할 수 있도록 유사성 감독 신호(Similarity-Supervised Signal, SSS)를 도입합니다.'}, {'Performance Highlights': 'DRepMRec은 세 개의 공개 데이터셋에서 최고의 성능(State-of-the-Art, SOTA)을 달성하였으며, 기존의 여러 멀티모달 추천 모델에 비해 일관된 성능 향상을 보여줍니다. 또한, BMA 모듈은 다른 추천 모델과 쉽게 통합되어 성능을 향상시킬 수 있습니다. 이 연구는 행동과 모달 표현의 더 나은 통합을 가능하게 하여 추천시스템의 학습 능력을 개선하는 데 중요한 기여를 합니다.'}]



### Threat Behavior Textual Search by Attention Graph Isomorphism (https://arxiv.org/abs/2404.10944)
- **What's New**: 이 논문에서는 트랜스포머 모델의 주의(attention) 계층을 기반으로 한 새로운 맬웨어 행동 검색 기술을 제안합니다. 특히, 그래프 이즈모피즘(graph isomorphism)을 사용하여, 맬웨어의 행동 및 기원을 파악하는 것을 목표로 합니다. 연구를 위해 다양한 기관으로부터 수집된 대규모 데이터셋도 구성하였습니다.

- **Technical Details**: 주의(attention) 그래프 기반 검색 방법은 도메인 특정 의미학(domain specific semantics)을 포착하는 데 특히 적합합니다. 트랜스포머 기반의 마스크 언어 모델(Masked Language Model, MLM)을 사용하여 데이터셋에서 비지도 학습(unsupervised learning)을 수행한 결과, 주요 공격 지표(indicators of compromise, IoC)와 그 상관 관계에 집중할 수 있는 언어 모델을 관찰하였습니다. 그 후, 트랜스포머 모델의 주의 계층에서 임계치(threshold) 이상의 주의를 보이는 두 노드(node) 사이에 에지(edge)를 도입하여 주의 그래프를 구성하고, 그래프 유사성을 이용해 CTI 보고서의 유사성을 결정합니다.

- **Performance Highlights**: 제안된 방법은 문장 임베딩(sentence embedding) 기반 기술이나 키워드(keyword) 기반 텍스트 유사성 방법과 비교할 때 6-14% 더 나은 성능을 보였습니다. 실제 세계의 맬웨어 10개를 대상으로 한 사례 연구에서는, Google 검색을 사용했을 경우 3개의 맬웨어만 올바른 기원을 찾아낼 수 있었지만, 제안하는 기술을 사용한 경우 8개의 맬웨어를 그들의 실제 기원과 정확하게 연관지을 수 있었습니다.



### Course Recommender Systems Need to Consider the Job Mark (https://arxiv.org/abs/2404.10876)
Comments: accepted at SIGIR 2024 as a perspective paper. Camera Ready will come soon

- **What's New**: 이 연구는 기존의 코스 추천 시스템이 직무 시장의 실시간 동향을 반영하는 데에 초점을 맞추어, 학습자의 커리어 목표와 시장 수요에 부합하는 교육 과정을 추천하는 '직무 시장 지향적 코스 추천 시스템'을 제안합니다. 이는 기존 시스템들이 주로 학습자와 과정 간의 상호작용, 과정 내용, 학습자 선호도 등에 기반을 둔 것에서 벗어나, 직업 시장에서 요구하는 스킬(Skill) 수요를 직접적으로 고려합니다.

- **Technical Details**: 이 연구에서 개발된 시스템은 큰 언어 모델(Large Language Models, LLMs)을 사용하여 이력서, 과정 설명, 직무 설명에서 스킬을 추출하고, 강화 학습(Reinforcement Learning, RL)을 통해 학습자의 직무 시장과의 일치도를 최대화하는 과정 시퀀스를 추천합니다. 이 시스템은 비지도 학습(Unsupervised Learning), 시퀀스 추천(Sequential Recommendation), 직무 시장 및 사용자 목표와의 일치(Alignment), 그리고 설명 가능성(Explainability)을 중요 속성으로 삼아 설계되었습니다.

- **Performance Highlights**: 초기 시스템은 공개 데이터를 사용하여 그 효과를 입증했습니다. 이 시스템은 직업 기회의 수를 극대화하는 과정의 시퀀스를 추천하며, 사용자의 진로 목표와 직업 시장과의 일치도를 예측하기 위한 척도(metrics)를 설계하는 과정에서도 더 나아갈 연구 방향을 제시합니다.



### Exploring Augmentation and Cognitive Strategies for AI based Synthetic  Persona (https://arxiv.org/abs/2404.10890)
Comments: This paper was accepted for publication: Proceedings of ACM Conf on Human Factors in Computing Systems (CHI 24), Rafael Arias Gonzalez, Steve DiPaola. Exploring Augmentation and Cognitive Strategies for Synthetic Personae. ACM SigCHI, in Challenges and Opportunities of LLM-Based Synthetic Personae and Data in HCI Workshop, 2024

- **What's New**: 이 논문은 큰 언어 모델(Large Language Models, LLMs)이 HCI(Human-Computer Interaction) 연구에서의 혁신적인 잠재력, 특허는 합성 인물 생성에서의 가능성을 강조하고 있습니다. 그러나 이 모델들의 블랙박스(black-box) 성격과 환각(hallucinations) 발생 경향이 연구에 어려움을 주고 있어, 이를 극복하기 위한 새로운 접근방식을 제시합니다.

- **Technical Details**: 본 논문은 LLM을 제로샷(zero-shot) 생성기로 사용하기 보다는 데이터 증강(data augmentation) 시스템으로 활용할 것을 제안합니다. 또한, LLM 응답을 안내하기 위한 강건한 인지 및 기억 프레임워크(cognitive and memory frameworks) 개발을 제안합니다. 데이터 풍부화(data enrichment), 에피소드 기억(episodic memory), 자기 성찰(self-reflection) 기법을 통해 합성 인물의 신뢰성을 높이고 HCI 연구에 새로운 방향을 제시할 수 있습니다.

- **Performance Highlights**: 초기 탐험을 통해 데이터 풍부화, 에피소드 기억, 자기 성찰 기법이 합성 인물의 신뢰성을 향상시키는 데 유용함을 시사하며, 이러한 기법들이 HCI 연구에 새로운 기회를 제공할 것으로 기대됩니다.



### Intelligent Message Behavioral Identification System (https://arxiv.org/abs/2404.10795)
- **What's New**: 이 연구는 소셜 미디어 플랫폼에서 이미지 재게시 (Image Retweeting)를 예측하는 새로운 방법을 제시합니다. 특히, 트위터(Twitter)에서 사용자의 사진 공유 행동을 예측하고, 이미지 재게시 모델링 (Image Retweet Modeling, IRM) 네트워크를 도입하여 이미지 트윗의 재게시에 대한 사용자의 과거 행동, SMS에서의 다음 연락처, 그리고 재게시된 사람의 선호도를 고려합니다.

- **Technical Details**: IRM 네트워크는 텍스트 가이드(TIRN)가 포함된 멀티모달 신경망(multimodal neural network)을 활용하여 이미지 트위터 표현과 사용자 선호 표현을 통합하는 새로운 다면적 주의 순위 망(multi-faceted attention ranking network) 방법론을 개발합니다. 이는 이미지와 텍스트 데이터를 조합하여 사용자의 재게시 행동을 예측하는데 고려됩니다.

- **Performance Highlights**: 다양한 데이터 셋에서 수행된 여러 실험을 통해, 현재 소셜 네트워크 플랫폼에서 사용되는 방법들을 능가하는 성능을 보여주었습니다. 이는 IRM이 트위터 사용자의 이미지 공유 선호와 행동 패턴을 더 정확하게 파악할 수 있음을 시사합니다.




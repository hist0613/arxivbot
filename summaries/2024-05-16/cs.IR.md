### Diffusion-based Contrastive Learning for Sequential Recommendation (https://arxiv.org/abs/2405.09369)
- **What's New**: 이 논문에서는 시퀀스 추천(sequential recommendation)의 향상을 위해 diffusion 기반의 대조 학습(diffusion-based contrastive learning) 접근법을 제안합니다. 기존 방법들이 랜덤 확장을 통해 시퀀스를 늘리면서 의미적 일관성을 놓치는 문제를 해결하고자 했습니다.

- **Technical Details**: 사용자 시퀀스에 대하여, 먼저 몇몇 위치를 선택하고 컨텍스트(context) 정보를 활용해 guided diffusion 모델을 통해 대체 아이템을 만듭니다. 이를 반복하여 사용자에게 의미적으로 일관된 시퀀스를 생성, 대조 학습 효과를 높입니다. 이 과정에서, diffusion 모델과 추천 모델의 표현 공간을 동일하게 유지하기 위해, 모든 프레임워크를 엔드 투 엔드(end-to-end) 방식으로 훈련하고, 동일한 아이템 임베딩(item embeddings)을 공유합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에서 확장된 실험 결과, 제안된 방법이 기존 방식들보다 우월함을 입증했습니다.



### Words Blending Boxes. Obfuscating Queries in Information Retrieval using Differential Privacy (https://arxiv.org/abs/2405.09306)
Comments:
          Preprint submitted to Information Science journal

- **What's New**: 새로운 접근법인 'Word Blending Boxes (WBB)'가 제안되었습니다. 이는 사용자의 프라이버시를 보호하면서도 검색 시스템에서 민감한 쿼리를 효과적으로 감추며, 동시에 관련 문서를 검색할 수 있도록 설계되었습니다. WBB는 차등 개인정보 보호(Differential Privacy, DP)를 활용하여 이론적으로뿐만 아니라 실질적으로도 사용자의 프라이버시를 보호합니다.

- **Technical Details**: WBB 메커니즘은 사용자의 검색 쿼리를 비식별화하는 과정에서 '안전 상자' 개념을 도입합니다. 이는 원본 쿼리와 너무 유사한 단어들이 최종 비식별화 쿼리에 포함되지 않도록 합니다. 또한, DP를 이용하여 안전 상자 바깥의 단어들을 선택함으로써 개인정보 보호 요구사항을 이론적으로 충족시킵니다. WBB는 기존의 정보 검색 시스템에 효과적으로 통합될 수 있습니다.

- **Performance Highlights**: WBB 방법은 Robust '04와 Deep Learning '19 TREC 컬렉션에서 평가되었습니다. 평가 항목은 프라이버시, 검색 재현율(recall), 그리고 유틸리티(utility)였습니다. 실험 결과, WBB 메커니즘은 원본 쿼리와 비식별화 된 쿼리 간의 어휘적 및 의미적 유사성을 낮추면서도 충분히 비식별화된 쿼리를 생성할 수 있음을 입증했습니다. 또한, 높은 프라이버시를 유지하면서도 관련 문서를 효과적으로 검색할 수 있음을 확인했습니다.



### Exploring the Individuality and Collectivity of Intents behind Interactions for Graph Collaborative Filtering (https://arxiv.org/abs/2405.09042)
Comments:
          10 pages, 7 figures, accepted by SIGIR 2024

- **What's New**: 사용자의 의도(Intent)는 추천 시스템에서 매우 중요한 요소입니다. 본 논문에서는 기존 시스템의 한계를 극복하고자 Bilateral Intent-guided Graph Collaborative Filtering (BIGCF)라는 새로운 추천 프레임워크를 제안합니다. BIGCF는 사용자-아이템 상호작용을 개별적(individual) 요인과 집단적(collective) 요인으로 나누어 보다 정교하게 모델링합니다.

- **Technical Details**: BIGCF는 사용자와 아이템의 Feature Distribution을 Gaussian 기반 그래프 생성 전략으로 인코딩합니다. 이 프로세스는 양방향 Intent 유도 그래프 재구성 샘플링을 통해 추천이 이루어집니다. 또한, 그래프 대조 규제(Graph Contrastive Regularization)를 활용하여 모든 노드의 일관성을 유지하고, 자기 지도 학습(self-supervised learning) 방식으로 최적화합니다.

- **Performance Highlights**: 세 개의 실제 데이터셋을 통해 실험한 결과, BIGCF는 기존 모델 대비 뛰어난 성능을 보였습니다. 이는 사용자의 개별적인 의도와 집단적인 의도를 동시에 고려함으로써 개인화된 추천의 정확성을 높일 수 있음을 입증합니다.



### Algorithmic Fairness: A Tolerance Perspectiv (https://arxiv.org/abs/2405.09543)
Comments:
          33 pages, 4 figures

- **What's New**: 새로운 논문에서는 알고리즘 공정성 (algorithmic fairness)과 그로 인한 다양한 사회적 영향을 조사합니다. '관용 (tolerance)'이라는 개념을 통해 공정성 결과의 변동이 어느 정도 허용될 수 있는지를 정의하는 새로운 분류법 (taxonomy)을 제안합니다. 이는 알고리즘 결정의 공정성을 이해하는 구조적 접근 방식을 제공합니다.

- **Technical Details**: 이 논문은 다양한 산업에서의 알고리즘 공정성 문제를 체계적으로 검토하며, 법적 관용 (legal tolerance), 윤리적 관용 (ethical tolerance), 개인적 관용 (personal tolerance)이라는 세 가지 차원에서의 공정성 수준을 분석합니다. 법적 관용은 법적 규제에 의해 명시적으로 금지된 차별, 윤리적 관용은 윤리적 규제에 의한 제한, 개인적 관용은 개인의 인식에 따라 변화하는 공정성 수준을 의미합니다.

- **Performance Highlights**: 연구의 주요 기여는 다음과 같습니다: 관용의 렌즈를 통해 알고리즘 공정성을 보는 새로운 공식화, 다양한 사례를 체계적으로 정리하여 공정성의 복잡성을 강조, 알고리즘 공정성 문제를 해결하기 위한 유망한 연구 방향을 제안합니다. 특히 알고리즘의 편향을 줄이고 더 공정한 시스템을 만들기 위한 전략적 방향을 제시합니다.



### Content-Based Image Retrieval for Multi-Class Volumetric Radiology Images: A Benchmark Study (https://arxiv.org/abs/2405.09334)
Comments:
          23 pages, 9 Figures, 13 Tables

- **What's New**: 이 연구는 TotalSegmentator 데이터세트(TS)를 사용해 세부 다기관 주석과 함께 영역 기반 및 다기관 검색을 위한 벤치마크를 설정하였습니다. 주석이 잘못되었거나 없는 대규모 데이터셋에서 신뢰할 수 있는 이미지 검색 방법을 개발하여 의학 영상 분석 및 진단에 도움이 되는 도구를 제공하고자 합니다.

- **Technical Details**: 이번 연구에서는 이미지 검색을 위한 사전 학습된 supervised 모델과 non-medical 이미지에서 학습된 unsupervised 모델을 비교 평가하였습니다. 또한, 텍스트 매칭에서 영감을 받은 late interaction re-ranking 방법을 채택해 원본 방법과 비교 및 벤치마크를 수행했습니다. 특히, TS 데이터세트의 29가지 범주 및 104가지 세부 해부 구조를 사용해 벤치마크를 수행했습니다.

- **Performance Highlights**: 제안된 re-ranking 방법은 다양한 크기의 해부학적 영역에서 1.0의 검색 리콜을 달성했습니다. 이는 다기관 검색의 현실적 시나리오에서 높은 효율성을 증명하며, 의학 영상에서 CBIR 접근방법의 개발 및 평가를 위한 중요한 통찰력과 벤치마크를 제공합니다.



### A Click-Through Rate Prediction Method Based on Cross-Importance of Multi-Order Features (https://arxiv.org/abs/2405.08852)
- **What's New**: 본 논문에서는 다중 차수 특징 상호작용 중요성을 학습할 수 있는 새로운 모델인 FiiNet을 제안합니다. 이 모델은 선택적 커널 네트워크(SKNets)을 이용해 명시적 높은 차수의 특징 교차(Multi-Order Feature Crosses)를 생성하고, 세밀한 방식으로 특징 상호작용의 중요성을 동적으로 학습합니다.

- **Technical Details**: FiiNet 모델은 SKNet 레이어를 사용하여 다중 차수 특징 교차를 명시적으로 구성하고, 교차 정보의 중요도를 가중 연산합니다. 구성 요소는 다음과 같습니다: 스파스 입력 레이어와 임베딩 레이어, 다중 은닉 레이어, 출력 레이어입니다. 특히 SKNet 레이어는 명시적으로 다중 차수의 특징 교차를 구성하고, 콘텐츠에 따라 교차 임베딩 크기를 조정하여 교차 조합의 중요성을 도출합니다.

- **Performance Highlights**: FiiNet 모델은 두 개의 실제 데이터셋을 사용하여 많은 기존 클릭 예측 모델들과 비교 평가되었으며, SKNe 모델을 통합한 FiiNet 모델이 추천 성능 향상과 더 나은 해석 가능성을 제공함을 입증했습니다.



### Evaluating Supply Chain Resilience During Pandemic Using Agent-based Simulation (https://arxiv.org/abs/2405.08830)
- **What's New**: 최근 팬데믹으로 인해 글로벌 경제 시스템, 특히 공급망의 취약성이 부각되었습니다. 본 연구에서는 확장된 감염-회복(SIR) 전염병 모델과 공급 수요 경제 모델을 통합한 새로운 에이전트 기반 시뮬레이션 모델을 제안합니다. 이를 통해 팬데믹 상황에서 다양한 공급망 회복 전략을 평가할 수 있습니다. 특히, 이 모델을 사용하여 균형 잡힌 공급망 회복 전략이 팬데믹 시기와 비 팬데믹 시기 모두에서 극단적인 전략보다 더 우수한 성과를 보인다는 점을 발견했습니다.

- **Technical Details**: 본 연구는 전염병 모델링에서 자주 사용되는 Susceptible-Infected-Recovered (SIR) 모델을 확장하여 경제적 공급 수요 모델과 통합한 에이전트 기반 시뮬레이션 (Agent-Based Simulation, ABS)을 사용했습니다. 시뮬레이션을 통해 다양한 팬데믹 시나리오와 공급망 회복 전략을 실험적으로 탐구하여, 경제적 및 전염병 관련 변화에 민감한 공급망 회복 전략을 분석했습니다.

- **Performance Highlights**: 실험 결과, 팬데믹 초기 경제 상태와 전염병 프로파일에 따라 기업마다 최적의 공급망 회복 전략이 상이하다는 사실을 발견했습니다. 따라서, 기계 학습 모델을 사용하여 특정 기업이 채택해야 하는 공급망 회복 전략을 예측했습니다. 이를 통해 정책 입안자와 기업이 팬데믹 상황에서 공급망 회복력을 강화하는 데 필요한 통찰력을 제공했습니다.




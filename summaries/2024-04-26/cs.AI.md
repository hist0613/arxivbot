### Distilling Privileged Information for Dubins Traveling Salesman Problems  with Neighborhoods (https://arxiv.org/abs/2404.16721)
Comments: 7 pages, 4 figures, double blind under review

- **What's New**: 이 연구는 두빈스 여행 판매원 문제(Dubins Traveling Salesman Problems, DTSP) 및 DTSP의 주변 환경이 고려된 문제인 DTSPN(DTSP with Neighborhood)을 해결하기 위한 새로운 학습 접근법을 제안합니다. 비홀로노믹 비행체가 목표 지점의   주변을 통과하는 경로를 신속하게 생성할 수 있는 방법을 소개합니다. 이는 특히 트랜스포머 모델 등 기존의 복잡한 학습 모델들과 비교할 때 뛰어난 효율성을 지닙니다.

- **Technical Details**: 이 방법론에는 모델 프리(model-free) 강화학습(reinforcement learning, RL)과 감독학습(supervised learning)의 두 단계가 포함됩니다. 첫 번째 단계에서는 특권 정보(privileged information)를 사용하여 리인커닝 휴리스틱(LinKernighan Heuristic, LKH) 알고리즘에 의해 생성된 전문가 궤적에서 지식을 추출합니다. 그 다음 감독학습 단계에서는 적응 네트워크(adaptation network)를 훈련하여 특권 정보 없이 문제를 독립적으로 해결할 수 있게 합니다. 이 프로세스는 표준 접근법와는 다르게 복잡한 샘플링 점 생성 과정을 요구하지 않으며, ATSP(Asymmetric TSP)로의 변환을 통해 문제를 해결합니다.

- **Performance Highlights**: 학습 기반 접근법을 통해 DTSPN 경로를 계산하는 속도는 기존 휴리스틱 방법보다 약 50배 빠른 것으로 나타났습니다. 또한, 이 방법은 다른 모방 학습(imitation learning) 및 시연을 통한 RL과 비교하여 우수한 성능을 보였으며, 대부분은 모든 작업 지점을 감지하도록 설정되지 않은 기존 방법을 상당히 뛰어넘는 결과를 보여줍니다. 따라서 비홀로노믹 차량의 운동 계획과 관련된 실제 문제에 크게 기여할 수 있습니다.



### Learning to Beat ByteRL: Exploitability of Collectible Card Game Agents (https://arxiv.org/abs/2404.16689)
- **What's New**: 포커 게임류(poker games)에 비해 과거에 상대적으로 덜 연구되었던 수집형 카드 게임(collectible card games)에 대한 최신 연구입니다. Hearthstone과 같은 가장 인기 있는 수집형 카드 게임에서 인간의 프로 선수와 경쟁할 수 있는 AI 에이전트(agent)가 최근에 나타났습니다.

- **Technical Details**: 이 논문에서는 Legends of Code and Magic 및 Hearthstone에서 최첨단 기술인 ByteRL 에이전트(agent)의 성능을 분석합니다. 수집형 카드 게임에서는 완벽하지 않은 정보(imperfect information)를 처리해야 할 뿐만 아니라, 상태공간(state space)이 너무 방대하여 에이전트가 믿는 모든 상태를 열거하는 것조차 불가능하여 기존의 탐색 방법(search methods)을 사용할 수 없습니다. 따라서 에이전트는 다른 기술을 채택해야 합니다.

- **Performance Highlights**: ByteRL은 중국의 상위 10위 내 Hearthstone 플레이어를 이겼지만, Legends of Code and Magic 게임에서는 쉽게 이용(exploitable)될 수 있는 약점을 보였습니다.



### Neural Interaction Energy for Multi-Agent Trajectory Prediction (https://arxiv.org/abs/2404.16579)
- **What's New**: 이 연구에서는 다중 에이전트 이동 경로 예측을 위한 새로운 프레임워크인 MATE(Multi-Agent Trajectory prediction via neural interaction Energy)를 소개합니다. MATE는 신경 상호작용 에너지(neural interaction energy)를 사용하여 에이전트들의 동적 상호작용을 평가하고, 시간적 안정성을 강화하기 위해 에이전트 간 상호작용 제약(inter-agent interaction constraint)과 내부 에이전트 움직임 제약(intra-agent motion constraint)을 도입했습니다.

- **Technical Details**: MATE 프레임워크는 다중 에이전트 시스템의 시간적 안정성을 유지하는 데 중점을 둡니다. 에이전트 간 상호작용 제약은 시스템 수준에서의 에이전트 상호작용의 안정성을 강화하며, 신경 상호작용 에너지의 변화를 최소화하여 다중 에이전트 이동 패턴의 일관성을 보장합니다. 내부 에이전트 움직임 제약은 에이전트 수준에서의 안정성을 강화하며, 에이전트의 과거 운동 상태와 주변 에이전트와의 상호작용을 기반으로 추정된 운동과 예측된 운동 사이의 불일치를 측정하는 시간 운동 분산(term)을 사용합니다.

- **Performance Highlights**: MATE 모델은 PHASE, Socialnav, Charged, NBA 데이터셋을 포함한 네 가지 다양한 데이터셋에서 최신 기술보다 우수한 예측 정확도를 보여주었습니다. 또한, 본 모델은 보지 못한 시나리오에서의 일반화 능력을 향상시키는 것으로 나타났습니다.



### SIDEs: Separating Idealization from Deceptive Explanations in xAI (https://arxiv.org/abs/2404.16534)
Comments: 18 pages, 3 figures, 2 tables Forthcoming in FAccT'24

- **What's New**: 이 논문은 설명 가능한 인공지능(xAI : explainable AI) 방법론이 신뢰를 구축하는데 중요하다는 점을 강조하면서, 현재 xAI 방법론이 불일치하고, 반드시 틀리며, 조작될 수 있다는 비판에 직면하고 있음을 지적합니다. 이러한 문제들이 블랙박스 모델(black-box models)의 배포를 저하시키고 있습니다. Rudin (2019) 은 높은 위험을 수반하는 경우에는 블랙박스 모델 사용을 전면 중단해야 한다고 주장합니다. 이 논문은 천연 과학에서 흔히 볼 수 있는 이상화(idealizations)의 개념을 도입하여 xAI 연구가 이상화의 평가에 참여해야 한다고 제안합니다.

- **Technical Details**: 저자는 자연 과학과 과학 철학에서 이상화 사용을 바탕으로, 설명 가능한 AI 방법이 성공적인 이상화 또는 속이는 설명(SIDEs : successful idealizations or deceptive explanations)을 사용하는지 평가하는 새로운 프레임워크를 제시합니다. SIDEs는 xAI 방법의 한계와 도입된 왜곡이 성공적인 이상화의 일부가 될 수 있는지, 아니면 비판가들이 제안하는 것처럼 실제로 속이는 왜곡인지를 평가합니다.

- **Performance Highlights**: 이 연구는 주요 특징 중요도 방법(feature importance methods) 및 반사실적 설명(counterfactual explanations)이 이상화 실패에 노출되어 있음을 질적 분석을 통해 발견했습니다. 또한, 이상화 실패를 개선하기 위한 방안들을 제안합니다.



### Label-Free Topic-Focused Summarization Using Query Augmentation (https://arxiv.org/abs/2404.16411)
- **What's New**: 이 연구는 주제 중심 요약(topic-focused summarization)을 위한 새로운 방법론인 Augmented-Query Summarization (AQS)을 소개합니다. AQS는 방대한 레이블된 데이터셋(labelled datasets)이 필요 없이 쿼리 증강(query augmentation)과 계층적 클러스터링(hierarchical clustering)을 활용하여 주제 중심의 요약을 가능하게 합니다. 이는 특정 토픽에 대한 사전 학습 없이도 요약 작업에 머신 러닝(machine learning) 모델을 전이할 수 있게 하여, 주제 중심 요약 기술의 접근성과 활용 범위를 넓히는 데 기여합니다.

- **Technical Details**: AQS 방법론은 쿼리 증강을 통해 입력된 텍스트에 관련된 추가 정보를 제공하고, 계층적 클러스터링을 이용하여 텍스트에서 주제별로 중요한 정보를 그룹화합니다. 이 과정을 통해 모델은 주제의 본질을 파악하고 관련성 높은 내용만을 요약에 포함시키는 것이 가능해집니다. 또한, 이 방식은 대규모 데이터셋을 요구하는 기존의 방법들에 비해 훨씬 경제적이면서 효율적입니다.

- **Performance Highlights**: 실세계 테스트에서, AQS는 관련성 높고 정확한 요약을 생성할 능력을 보여주었습니다. 이는 데이터가 풍부한 환경에서 비용 효과적인 솔루션으로서의 잠재력을 입증하며, 개인화된 콘텐츠 추출을 위한 효율적이고 확장 가능한 방법을 제공합니다.



### Optimal and Bounded Suboptimal Any-Angle Multi-agent Pathfinding (https://arxiv.org/abs/2404.16379)
- **What's New**: 이 논문에서는 기존의 독특한 접근 방식을 사용하여 ‘멀티 에이전트 패스파인딩’(Multi-Agent Pathfinding, MAPF) 문제를 해결합니다. 특히, 이 논문에서 처음으로 최적의 ‘애니앵글’(Any-Angle) 멀티 에이전트 패스파인딩 알고리즘을 제시합니다. 이 알고리즘은 에이전트가 장애물에 충돌하지 않는 한 어떤 두 지점 사이에서도 움직일 수 있는 새로운 환경에서의 경로 탐색을 가능하게 합니다.

- **Technical Details**: CCBS(Continuous Conflict-based Search) 알고리즘과 최적의 애니앵글 변형인 TO-AA-SIPP(Safe Interval Path Planning) 알고리즘을 기반으로 합니다. 그러나 이들의 단순한 결합은 아주 큰 분기 요인을 가진 탐색 트리를 유발하기 때문에 잘 확장되지 않습니다. 이를 완화하기 위하여, 두 가지 기술, '분리 분할’(Disjoint Splitting)과 '다중 제약 조건’(Multi-Constraints)을 애니앵글 설정에 적용했습니다.

- **Performance Highlights**: 실험 결과, 분리 분할과 다중 제약 조건의 다양한 조합을 사용하여 CCBS 및 TO-AA-SIPP의 기본 조합보다 30% 이상 다양한 문제를 해결할 수 있음을 보여줍니다. 또한, 실행 시간과 솔루션 비용을 제어 가능한 방식으로 교환할 수 있는 경계-부최적 변형의 알고리즘을 제시합니다.



### ReZero: Boosting MCTS-based Algorithms by Just-in-Time and Speedy  Reanalyz (https://arxiv.org/abs/2404.16364)
- **What's New**: 새로운 접근 방식인 ReZero가 제안되었습니다. 이 방법은 MCTS 기반 알고리즘을 강화하며, 데이터 수집 및 재분석(reanalyze) 과정을 단순화하여 검색 비용을 크게 줄이면서도 성능을 보장합니다. 또한, 각 검색 과정을 가속화하기 위해 경로(trajectory)의 후속 정보를 재사용하는 방법이 고안되었습니다. 이러한 설계에 대한 이론적 근거는 밴딧 모델(bandit model)에 대한 분석을 통해 제공됩니다.

- **Technical Details**: ReZero는 정기적으로 목표 네트워크(target network)를 업데이트하는 DQN에서 영감을 받아 각 반복에서 미니배치를 재분석할 필요가 없다는 점을 찾아냈습니다. 이를 통해 MCTS 호출 횟수를 현저히 줄이고, 재분석 과정에서 후속 상태의 검색 정보를 활용해 현재 상태의 검색을 가속화합니다. 이 접근 방식은 하드웨어 오버헤드 증가 없이 다양한 MCTS 기반 알고리즘과 함께 사용할 수 있습니다. 이론적 분석과 실제 실험을 통해, 이 방식은 훈련 속도를 크게 개선하면서도 높은 샘플 효율성을 유지하는 것으로 나타났습니다.

- **Performance Highlights**: ReZero는 아타리(Atari) 환경과 보드 게임에서 실시한 실험을 통해 훈련 속도를 크게 향상시키는 동시에 높은 샘플 효율성을 유지하는 것을 입증했습니다. 추가로 진행된 절제 실험(ablation experiments)은 정보 재사용의 가속 효과 및 다양한 재분석 주기의 영향을 탐구하였습니다. 이러한 결과들은 ReZero가 싱글 에이전트(single-agent) 환경과 두 플레이어 간의 보드 게임에서 모두 좋은 성능을 보임을 보여줍니다.



### Knowledge Graph Completion using Structural and Textual Embeddings (https://arxiv.org/abs/2404.16206)
- **What's New**: 이 연구에서는 텍스트 및 구조적 정보를 활용하여 지식 그래프(Knowledge Graphs, KGs) 내에서 관계를 예측하는 새로운 모델을 제안하였습니다. 이 모델은 기존 노드 간의 관계를 탐색하여 KGs의 불완전성을 해결하는 데 중점을 둡니다.

- **Technical Details**: 제안된 모델은 워크 기반 임베딩(walks-based embeddings)과 언어 모델 임베딩(language model embeddings)을 통합하여 노드를 효과적으로 표현합니다. 이를 통해 KGs에서 텍스트와 구조의 정보를 모두 활용할 수 있으며, 관계 예측(relation prediction) 작업에 있어서 높은 수준의 성능을 보였습니다.

- **Performance Highlights**: 이 모델은 널리 사용되는 데이터셋에서 평가되었을 때, 관계 예측 작업에 대해 경쟁력 있는 결과를 달성하였습니다. 이는 텍스트와 구조적 정보의 통합 접근이 KG의 불완전성을 보완하는 데 유효함을 시사합니다.



### Learning World Models With Hierarchical Temporal Abstractions: A  Probabilistic Perspectiv (https://arxiv.org/abs/2404.16078)
Comments: Doctoral Dissertation Preprint, Department of Computer Science, Karlsruhe Institute Of Technology, 2024

- **What's New**: 이 논문은 인간의 지능과 유사한 2형 추론 능력을 가진 기계의 개발에 필요한 내부 세계 모델을 개발하기 위한 새로운 형식을 제안합니다. 기존의 상태 공간 모델(State Space Models, SSMs)이 가진 여러 제한점을 인식하고, 이를 극복하기 위해 '숨겨진 매개변수 SSMs(Hidden-Parameter SSMs)'와 '다중 시간 척도 SSMs(Multi-Time Scale SSMs)'라는 두 가지 새로운 확률적 형식을 제시합니다.

- **Technical Details**: 제안된 두 모델은 그래픽 모델의 구조를 활용하여 벨리프 전파(belief propagation)를 통한 확장 가능한 정확한 확률적 추론과 시간에 따른 역전파(backpropagation through time)를 통한 엔드 투 엔드 학습을 지원합니다. 이러한 접근 방식은 다중 시간 추상화 및 규모에서 비정상 동역학을 나타낼 수 있는 확장 가능하고 적응 가능한 계층적 세계 모델의 개발을 허용합니다. 추가적으로, 이 확률적 형식은 세계 상태의 불확실성 개념을 통합하여 실제 세계의 확률적 특성을 모방하고 예측에 대한 신뢰도를 정량화하는 시스템의 능력을 향상시킵니다.

- **Performance Highlights**: 다양한 실제 및 시뮬레이션된 로봇에서의 실험을 통해 제안된 형식이 장기 예측에서 현대의 트랜스포머(Transformer) 변형 모델들의 성능을 맞추거나 종종 능가하는 것으로 나타났습니다. 이러한 결과는 제안된 모델이 예측적 처리(predictive processing) 및 베이지안 뇌 가설(Bayesian brain hypothesis)과 같은 관련 신경과학 문헌과 일치함을 보여줍니다.



### Playing Board Games with the Predict Results of Beam Search Algorithm (https://arxiv.org/abs/2404.16072)
Comments: 8 pages, 4 figures

- **What's New**: 이 논문에서는 완벽한 정보를 가진 두 플레이어의 결정론적 게임을 위한 새로운 알고리즘인 PROBS(Predict Results of Beam Search)를 소개합니다. 기존의 Monte Carlo Tree Search(MCTS)에 주로 의존하는 다른 방법들과 달리, 이 방법은 더 간단한 빔 검색(beam search) 알고리즘을 사용합니다.

- **Technical Details**: PROBS 알고리즘은 게임의 평균 턴(turn) 수보다 상대적으로 작은 빔 검색 크기(beam search size)에서도 효과적으로 작동합니다. 이는 효율적인 계산과 더 빠른 결정 과정을 가능하게 합니다.

- **Performance Highlights**: 다양한 보드 게임(board games)에서 이 알고리즘의 성능을 평가한 결과, 기준이 되는 상대(baseline opponents)에 대해 일관되게 높은 승률(winning ratio)을 보여주었습니다. 이는 PROBS 알고리즘이 기존 방법들보다 우수할 수 있다는 것을 시사합니다.




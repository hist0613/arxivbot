New uploads on arXiv(cs.MA)

### Mosaic: Runtime-Efficient Multi-Agent Embodied Planning (https://arxiv.org/abs/2607.09603)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: LLM 기반 multi-agent embodied planning은 실행 지연이 커서 실제로 쓰기 어렵다는 문제가 있었다. 기존 접근은 부분 관측 환경에서 상태 추적이 부정확해지거나, 에이전트 간 조율이 비효율적이어서 중복·충돌 행동이 늘어나는 경향이 있다.

- **Core Contribution**: 논문은 실패 행동이 지연의 핵심 병목이라고 보고, 이를 만드는 두 원인(부분 관측 하의 부정확한 상태 추적, 비효율적 조율)을 함께 해결하는 런타임 효율 프레임워크 Mosaic을 제안한다. Mosaic은 에이전트 중심 의미 메모리로 상태를 가볍게이지만 정확하게 유지하고, 매 단계마다 행동을 배정해 조율을 물리적으로 제약한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 부분 관측에서도 쓸 수 있을 만큼 가벼우면서도 정확한 상태 추적을 유지하는 것과 (2) 조율 과정에서 중복/충돌을 줄이면서 물리적 제약을 만족하는 행동 배정을 하는 것이다. Mosaic은 객체를 상대 좌표로 저장하는 agent-centric semantic memory로 기하 변환을 가능하게 해 상태 추적을 경량화하고, Integer Linear Programming로 매 planning step마다 행동을 할당해 coordination 제약과 물리적 feasibility를 강제한다.

- **Empirical Impact**: AI2-THOR 및 search-and-rescue 벤치마크에서 Mosaic은 실행을 27-32% 더 빠르게 하고, LLM 호출은 30-33% 줄였으며, 단계 수는 25-31% 감소시켰다. 또한 성공률이 4-10%p 높아져, 대규모로 확장 가능한 low-latency multi-agent planning에서 메모리 효율과 constraint-guided coordination이 중요함을 실증적으로 보여준다.



### When is Routing Meaningful? Diversity and Robustness in Language Model Societies (https://arxiv.org/abs/2607.09197)
- **Prior Approaches**: 기존 멀티모델 라우팅 평가는 과제 정확도와 추론 비용에 거의만 초점을 맞춰, 라우팅이 “의미 있게” 작동하는지(전문가 분화가 실제로 일어나는지)를 놓치는 경우가 많았습니다. 특히 모든 actor가 동일하게 반응하면 라우팅은 사실상 무의미해지지만, 이 행동적 차별화 여부는 주로 측정되지 않았습니다. 또한 같은 의미를 가진 쿼리의 표면 형태 변형에 대해 서로 다른 actor로 보내면 라우팅이 불안정해지지만, 이 안정성 역시 간과되는 경향이 있습니다.

- **Core Contribution**: 이 논문은 라우팅의 성패를 성능과 무관하게 좌우하는 두 속성으로 정리하며, (1) actor들의 행동적 차별화(behaviorally differentiated)와 (2) 라우팅 정책의 안정성(stable)을 핵심 기준으로 제안합니다. 더불어 Hierarchic Social Entropy(HSE)를 language-model society에 맞게 적용하고, 쿼리 표면형 변형에 대한 라우팅 일관성을 perturbation 기반 강건성 지표로 진단합니다. 이를 통해 정확도는 높아도 전문화가 일어나지 않는 실패 모드를 체계적으로 분리해 보여줍니다.

- **Technical Challenges**: 주된 기술적 난제는 “정확도만으로는 보이지 않는” 라우팅의 무의미함을 정량화하는 방법을 만드는 것이었습니다. 저자들은 HSE로 actor 간 분화가 실제로 얼마나 유효하게 유지되는지 측정하고, 입력 perturbation(표면형 변형)에도 같은 actor로 매핑되는지로 안정성(robustness)을 평가하도록 설계했습니다. 나아가 agent 수를 줄여도 충분한 다양성이 복원되는지(coreset 휴리스틱)를 실험으로 확인하면서 HSE의 diminishing returns 현상도 함께 분석했습니다.

- **Empirical Impact**: EmbedLLM과 RouterBench에 적용한 결과, HSE는 에이전트 수 증가에 따른 효용이 빠르게 감소해 10개 미만의 curated subset만으로도 큰 풀에서 가능한 다양성을 대부분 회복하는 것으로 나타났습니다. 또한 KNN router는 specialist society에서 정확도는 개선되지만 perturbation에 대한 robustness가 무너져 의미 있는 라우팅 성립이 약해지는 패턴을 보였습니다. 반대로 prompted routing은 다양한 perturbation 유형에서도 안정성이 유지되어, 정확도와 “라우팅의 meaningfulness”가 크게 엇갈릴 수 있음을 실증적으로 확인했다는 점에서 분야에 중요한 평가 프레임을 제공합니다.



### Control Laguerre Tessellation: Semi-discrete Optimal Transport Over Control Systems (https://arxiv.org/abs/2607.09139)
- **Prior Approaches**: 반정연속 최적수송(SDOT)은 연속 소스 μ와 이산 타깃 ν 사이에서 수송 비용 c(x,y)를 최소화하는 문제로, 타깃이 점(Dirac)으로 주어질 때 라그랑주 이중변수에 기반한 Laguerre tessellation(라게르-테셀레이션) 구조가 twist condition 하에 성립하는 것이 잘 알려져 있다. 기존에는 주로 제곱유클리드 비용 등 정해진 metric 기반 ground cost에서 이 구조(예: power diagram, Apollonius diagram)를 다뤘고, twist condition을 만족하는 경우 최적 수송맵이 거의 모든 곳에서 기하적으로 결정된다는 점이 강조돼 왔다.

- **Core Contribution**: 이 논문은 SDOT의 ground cost를 “각 에이전트가 자신에게 주어진 제어 문제를 풀어 얻는 최적제어 비용”으로 유도한다는 점에 집중한다. 특히 최소 에너지 및 최소 시간 같은 제어 목적에서 만들어지는 비용이 twist condition을 만족하면, 표준 Laguerre tessellation을 제어 이론적으로 일반화한 Control Laguerre Tessellation(CLT)로 최적 수송맵이 거의 모든 곳에서 표현된다고 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 제어 문제로부터 정의된 ground cost가 twist condition을 실제로 만족하는지, 그리고 그 결과 SDOT의 최적맵이 라그랑주 셀 기반의 닫힌(기하적) 형태를 갖는지 확인하는 것이다. 저자들은 선형 제어계의 최소 에너지 비용은 적절한 변환을 통해 2차형(quadratic) 형태로 정리되어 twist condition이 성립함을 보였고, 최소 시간 비용은 일반 해 형태가 없더라도 제어 유도 비용으로 SDOT에 같은 해결 틀을 적용할 수 있음을 정식화했다.

- **Empirical Impact**: CLT 관점은 “군집의 분배(수송)”뿐 아니라 “각 에이전트의 개별 제어 최적성”까지 비용에 내재화해 SDOT 모델을 확장한다는 점에서 의미가 있다. 또한 CLT는 dual potential(가중치 벡터) ψ를 풀어 r개의 비선형 이산 모노-앰페르 방정식(Discrete Monge-Ampère)으로 귀결되며, 이는 damped Newton 또는 Oliker-Prussner coordinate descent 같은 수치기법으로 구현 가능하다고 설명한다.



### Secret Scanner Agent: Extracting Secrets and Access Context from Unstructured Documents (https://arxiv.org/abs/2607.09011)
Comments:
          Submitted to the Conference on Applied Machine Learning for Information Security (CAMLIS) 2026

- **Prior Approaches**: 기존 secret scanner는 정규식·엔트로피·키워드 휴리스틱, 때로는 machine learning을 활용해 노출된 자격증명을 찾는 데 강점이 있다. 그러나 이메일·티켓·채팅처럼 자격증명이 조각나거나 재포맷되고, 비밀번호/토큰 등과 리소스 정보가 멀리 떨어져 있으면 탐지율이 떨어지거나 잡음 경보가 늘어난다. 또한 대부분은 ‘문자열(secret)’만 알려주고 그것이 열어주는 대상(door: 계정/테넌트/엔드포인트/DB/클라우드 리소스 등)을 함께 특정하지 못해 후속 triage가 수작업에 의존한다.

- **Core Contribution**: Secret Scanner Agent(SSA)는 unstructured 문서에서 ‘secret’과 그 secret이 접근 권한을 줄 수 있는 ‘door’를 증거와 함께 함께 추출하는 에이전트형 LLM 시스템을 제안한다. SSA는 recall을 중시하는 detection agent와 false positive를 걸러내고 누락된 문맥을 복구하는 review agent를 결합해, 단일 에이전트 대비 secret–door 페어링 정확도를 높인다. 결과물은 비밀번호 문자열만이 아니라 “무엇이 새고 어디에 조치해야 하는지”를 한 번에 제공하도록 설계됐다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 조각난/불완전한 키와 문서 전반에 흩어진 단서를 복원해 올바른 후보를 찾는 것, (2) UUID·해시·플레이스홀더처럼 secret처럼 보이지만 실제로는 무의미한 값들을 배제하는 것, (3) secret과 door를 문서 근거에 기반해 정확히 연결하는 것이다. SSA는 detection 단계에서 넓게 후보를 수집한 뒤, review 단계에서 원문 근거로 지지되는지 확인해 잡음을 줄이고 door 복구를 수행하는 detect-then-critique 다중 에이전트 흐름으로 이를 해결한다. 또한 실제 사용자 데이터 노출 위험 때문에, 23종 secret 유형과 문서 형식을 포괄하는 합성 벤치마크 파이프라인을 자체 구축해 안전하게 평가했다.

- **Empirical Impact**: 평가는 합성 문서 기반 다단계 판정(프로그램적 매칭→LLM judge→인간 검토)으로 수행됐으며, 6개 모델 구성에서 단일 에이전트 변형 대비 multi-agent SSA의 precision이 향상됐다. 특히 door 추출에서 최대 16%p까지 개선됐고, 정규식 기반 스캐너와 비슷한 precision을 유지하면서 recall은 3배 이상 증가했다. 또한 13명의 보안 분석가 대상 인간 비교에서 SSA는 더 높은 정밀도로 더 많은 secret–door 페어를 복구했으며 처리 속도도 5~17배 빨라져, credential 탐지를 triage·remediation에 바로 연결하는 “실행 가능한 finding”으로서 의미가 크다.



### Offline Nash Solvers Meet Online Tree Search in Multi-Agent Games on Graphs (https://arxiv.org/abs/2607.08892)
- **Prior Approaches**: 다중 에이전트 Pursuit–Evasion game(PEG)에서 내쉬 균형 정책을 정확히 계산하려면 joint state와 joint action이 에이전트 수에 따라 지수적으로 커져 계산이 급격히 불가능해진다. 기존 오프라인 방식(예: PSRO 계열)은 실행 시 신속하지만, 학습 중 보지 못한 적대 행동에 취약할 수 있고 에이전트 수가 늘면 학습 복잡도도 커진다. 온라인 계획(MCTS/SM-MCTS)은 상황 적응과 전략 추론이 가능하지만, 동시 행동 게임에서 branching factor가 폭발해 휴리스틱 의존도가 높아진다.

- **Core Contribution**: 이 논문은 Primitive-Guided Tree Search(PGTS)라는 하이브리드 프레임워크를 제안한다. PGTS는 원게임을 (1v1, 2v1 등) 작은 primitive sub-team game으로 분해해 오프라인에서 내쉬 균형을 정확히 계산하고, 배치 중에는 매 타임스텝 온라인 트리 탐색을 수행하되 primitive 정책과 value를 가이드로 사용한다. 그 결과 full team의 동시 조정된 행동을 고려하면서도, 원래 문제의 계산 병목을 primitive의 재사용으로 완화한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) full joint action 공간에서 트리 탐색의 branching factor를 줄이면서 (2) 트리 깊이를 과도하게 늘리지 않고 미래 보상을 잘 추정하는 것이다. PGTS는 primitive sub-team game의 equilibrium policy를 joint-action rollout에 반영해 탐색이 중요한 행동 조합에 집중되도록 하고, primitive value function을 leaf-node value 추정에 활용해 더 얕은 깊이에서 종료해도 성능을 유지하도록 설계한다. 또한 primitive 게임이 부분 집합만 다루는 구조적 한계는, rollout 시에는 여전히 full joint action을 사용해 협동적 의사결정을 보완하는 방식으로 다룬다.

- **Empirical Impact**: 다양한 그래프 토폴로지(실세계 네트워크 포함)에서 PGTS는 기존 learning 기반 방법과 휴리스틱 기준선을 유의미하게 능가하며, 적대자(adversary) 변화에도 견고한 성능을 보인다. 특히 오프라인에서 얻은 정확한 균형 정보를 온라인 탐색의 효율성과 품질에 직접 연결해, 단순 분해 기반 surrogate 접근 대비 조정된 계획 능력을 확보한 점이 관찰된다. 전반적으로 내쉬 균형 계산의 실용적 확장(에이전트 수 증가)을 보여주는 하이브리드 모델 기반 접근으로 평가된다.




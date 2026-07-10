New uploads on arXiv(cs.CL)

### UniClawBench: A Universal Benchmark for Proactive Agents on Real-World Tasks (https://arxiv.org/abs/2607.08768)
Comments:
          Project Page: this https URL | GitHub Repo: this https URL

- **Prior Approaches**: 기존 에이전트 벤치마크는 웹/OS/데스크톱 등 실제 환경 요소를 포함하더라도, 정적 답안 기반이거나 single-turn 평가에 머무르는 경우가 많아 닫힌 루프 상호작용을 제대로 반영하기 어렵습니다. 또한 작업을 ‘office, research’ 같은 시나리오로 묶어 서로 다른 능력이 한 범주에 섞여 실패 원인을 분리하기 힘들다는 한계가 지적됩니다. 그 결과 샌드박스 성능과 실제 작업 성공률 사이의 격차가 커도 원인 분석이 곤란했습니다.

- **Core Contribution**: 이 논문은 능력 중심(capability-driven)으로 선제적(proactive) 에이전트를 동적 실환경에서 평가하는 UniClawBench를 제안합니다. Skill Usage, Exploration, Long-Context Reasoning, Multimodal Understanding, Cross-Platform Coordination의 다섯 능력 축으로 400개(영/중) 실사용형 태스크를 구성해 실패 원인을 더 정밀하게 진단하도록 설계했습니다. 실험은 Docker의 라이브 컨테이너에서 수행하되, 미리 고정된 정답 대신 단계별 completion checkpoint를 통해 채점합니다.

- **Technical Challenges**: 실환경에서는 정답이 시시각각 바뀌어 안정적인 ground truth를 만들기 어렵고, closed-loop 피드백을 주더라도 평가 기준(루브릭)이나 정답을 유출하면 안 되는 문제가 동시에 존재합니다. UniClawBench는 hidden supervisor(숨은 감독자), user simulator(사용자 시뮬레이터), executor(평가 대상)로 역할을 분리하고, 감독자는 pass/fail/continue 같은 고수준 신호만 전달하며 세부 채점 기준은 공개하지 않는 ‘정보 격리 방화벽’을 구현합니다. 또한 진행 중 생성 증거를 수집한 뒤 체크포인트 기반 루브릭으로 단계별로 평가해 동적 환경의 흔들림을 흡수합니다.

- **Empirical Impact**: 10개 SOTA 모델을 동일 OpenClaw 프레임워크로 비교한 결과, 상위 폐쇄형 모델이 전체 pass rate를 주도해도 절대 성공률은 50% 미만으로 나타나 벤치마크의 난도가 매우 높음을 보여줍니다. 특히 많은 모델이 중간 체크포인트 점수는 높게 얻지만 끝까지 ‘완료’에 실패하는 halfway failure 현상이 두드러져 장기 실행 신뢰성의 부족을 실증합니다. 또한 프레임워크(OpenClaw, EDICT, Nanobot) 간 성능 차이가 커지며, 베이스 모델 능력뿐 아니라 에이전트 아키텍처가 실제 성공을 증폭/병목한다는 점을 정량적으로 확인했습니다.



### Validity of LLMs as data annotators: AMALIA on authority (https://arxiv.org/abs/2607.08731)
- **Prior Approaches**: 기존 LLM 기반 텍스트 코딩 평가는 주로 사람 라벨과의 일치도(예: F1, kappa)로 신뢰성(reliability)을 확인하는 방식이었습니다. 그러나 일치도는 타당도(validity)를 보장하지 않으며, 모델이 이론이 지정한 추론이 아니라 표면 상관관계로 ‘비슷한 코드’를 맞출 수도 있다는 문제가 남습니다. 특히 zero-shot prompt는 일부 구성에서 false positive를 크게 부풀릴 수 있어, 단순 정확도 비교만으로는 측정 품질을 판단하기 어렵습니다.

- **Core Contribution**: 이 논문은 ‘회복 격차(recovery gap)’라는 운영적 검증 절차로, LLM이 구성(construct)을 이론이 요구하는 증거 경로로 측정하는지 확인합니다. 구체적으로, holistic prompt로 한 번에 코드(예: authority)를 내리게 한 뒤, 이 코드를 구성 이론에 따라 원자 단위(탐지·구분·평가)로 분해하고 통합 규칙으로 재조합했을 때 성능이 얼마나 유지되는지로 타당도를 봅니다. 또한 영어에서 grain calibration(입자 수준 보정)이 된 측정 도구가 AMALIA-9B와 유럽 포르투갈어(pt-PT)에 그대로 이식되는지도 실험합니다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM이 구성의 표면 단서가 아니라 관계(누가 복종해야 하는가, 권위가 확언/도전/위반되는가 등)를 추론해야 한다는 점이며, 그 과정이 블랙박스라 오류 원인을 합의도만으로는 분해할 수 없다는 것입니다. 논문은 구성 정의를 구성 요소와 clause(원자 질문)로 쪼개고, 이 clause 답을 이론 기반의 명시적 integration rule로 다시 합치는 방식으로 ‘올바른 증거 경로’ 여부를 진단합니다. 나아가 실제 포르투갈어 말뭉치 구축에서는 transcreation 시 의미·발화 의도·관점·지시대상(stability)을 보장하도록 여러 verification gate를 걸어 평가 오염 가능성을 줄였습니다.

- **Empirical Impact**: 결과(초록 요약 기준)로는 AMALIA-9B가 권위(authority) 구성에서 사람 코더와의 일치도는 준수하지만, 분해 후 재조합해도 holistic 성능의 약 절반만 회복되어 recovery gap이 크게 나타났습니다. 오류 분석에서는 특히 권위자 주변의 도덕적 분노 같은 표면 상관 단서에 의존하는 정황이 보고됩니다. 반면 동일 포르투갈어 코퍼스에서 한 다국어 open LLM은 recovery gap을 더 잘 메워, ‘코퍼스 문제’보다는 AMALIA의 측정 도구로서의 한계를 시사하며, 국가 주권 모델이라도 construct 타당도 검증 없이는 단독 측정에 신중해야 함을 강조합니다.



### Do You Need a Frontier Model as a Citation Verifier? Benchmarking Rubric LLMs for Deep-Research Source Attribution (https://arxiv.org/abs/2607.08700)
- **Prior Approaches**: 기존 RLVR과 rubric RL에서는 LLM이 채점한 단일 점수를 스칼라 보상으로 쓰는 경우가 많아, 차원 혼재로 인한 reward hacking과 sycophancy 위험이 지적돼 왔습니다. citation 품질에 대해서는 검색 기반 문서의 인용을 평가하는 파이프라인과 벤치마크가 있었지만, 어떤 LLM judge를 쓸지(비용/편향 포함)와 그 선택이 학습 루프에 어떤 방향성 편향을 남기는지는 충분히 비교·분해되지 않았습니다.

- **Core Contribution**: 이 논문은 deep-research citation을 대상으로 source relevance와 factual support라는 두 축의 LLM-판정 루브릭을 **judge 캘리브레이션** 관점에서 직접 비교합니다. 또한 적대적으로 구성된 long-form 문서에서 1,248개의 rubric decision을 사람 검토 기반 gold label로 구축한 Deep-Research Citation Benchmark를 제시해, judge 신뢰도와 편향을 정량화합니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 인용문-근거 쌍을 추출해 루브릭 항목별로 판정하고, (2) accuracy(F1, Cohen’s κ)뿐 아니라 RLVR 학습에 실제로 영향을 주는 pass-rate drift, FPR, FNR 같은 방향성 편향을 함께 측정하는 것입니다. 저자들은 Onweller et al.의 2단계 파이프라인을 사용하되, 접근 가능성(link accessibility)은 결정론적으로 분리하고 두 LLM-판정 축만 8개 off-the-shelf judge로 평가하며, 불일치가 있는 378개 hard case는 다중 judge 합의 후 사람 adjudication으로 고정합니다.

- **Empirical Impact**: 실험 결과 GPT-5-mini는 source relevance에서 최고 성능(F1 0.908, κ=0.636)을 보이지만, factual support에서는 모델 간 통계적으로 유의미한 우열이 뚜렷하지 않았습니다. 그럼에도 scalar F1만 보면 가려지는 pass-rate drift와 FPR/FNR 차이가 컸고, 특히 방향성 편향은 RLVR 루프가 어떤 신호를 과대·과소 보상할지에 직결되어 judge 캘리브레이션이 선행 조건임을 확인했습니다. 더 비싼 frontier judge가 항상 정답은 아니며, 고가 모델이 아니라도 비용 대비 충분히 경쟁력 있는 judge 선택이 가능하다는 점이 실무적 의미를 갖습니다.



### WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search (https://arxiv.org/abs/2607.08662)
Comments:
          Work in progress

- **Prior Approaches**: ReAct 기반 단일 에이전트는 한 번에 긴 탐색 궤적을 수행해야 해 컨텍스트 한계로 인해 깊이·너비가 동시에 커지면 성능이 급격히 떨어진다. 다중 에이전트 접근은 병렬 탐색과 결과 집계를 통해 커버리지는 늘리지만, 재귀적 깊이 확장·협업 형태 전환·증거에 근거한 확장이 유연하지 못하다는 한계가 있다. 또한 작업을 쿼리 표면 의미에만 맞춰 분해하면 웹에서 정보가 실제로 조직되는 방식과 어긋나 중복 탐색과 집계 실패가 발생할 수 있다.

- **Core Contribution**: WebSwarm은 추론 중에 에이전트 노드를 동적으로 만들어 “점진적 재귀 위임(progressive recursive delegation)”을 수행하는 프레임워크를 제안한다. 각 노드는 로컬 목표와 search mode(원자적 탐색 atom, 반복 탐색 deep, 병렬 분할 wide, 미지 경계 열거 entity_collect)를 함께 받아, 스스로 풀거나 자식 노드를 위임해 증거를 상향 반환한다. 또한 웹 정보 구조 탐사와 동종 형제 노드 간 프로세스 경험 재사용을 통해, 분해·확장·협업을 증거가 쌓일수록 함께 진화시키도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 초기 쿼리만으로는 필요한 하위 과제가 언제 드러날지 예측하기 어려워 재귀 깊이를 안정적으로 키우는 것, (2) 하위 목표마다 필요한 협업 프로토콜이 달라 고정된 협업 토폴로지로는 부족한 것, (3) 웹에서의 정보 조직 방식과 분해 축이 어긋날 때 비효율이 누적되는 것이다. WebSwarm은 objective–mode pair로 하위 노드별 해결 전략을 명시하고, 하위 노드가 증거를 반환한 뒤 상위 노드가 expand·revise·terminate를 선택하는 피드백 루프를 구현했다. 더불어 wide 확장이 필요한 노드에는 웹-구조 probing을 붙여 확장 축을 정렬하고, scout 노드로부터 동종 sibling의 실패/성공 경로를 요약한 경험 kv를 재주입해 중복 시행착오를 줄인다.

- **Empirical Impact**: BrowseComp-Plus, WideSearch, DeepWideSearch, GISA 등 4개 벤치마크에서 WebSwarm은 단일 ReAct과 다양한 다중 에이전트 기준선을 일관되게 능가하거나 경쟁력 있는 성능을 보였다. 특히 깊이/너비가 섞인 DeepWideSearch처럼 단계 전환이 잦은 과제에서 고정 협업 패턴의 약점을 모드 기반 재귀 위임으로 완화했다. ablation 결과로도 재귀 위임 자체가 성능 저하를 유발하고, 웹-구조 probing은 도구 호출 수를 크게 줄이며, 동종 노드 경험 재사용은 품질(Item F1 등) 하락을 막는 역할을 확인했다.



### UltraX: Refining Pre-Training Data at Scale with Adaptive Programmatic Editing (https://arxiv.org/abs/2607.08646)
- **Prior Approaches**: 기존 데이터 품질 개선은 규칙 기반 필터링/클리닝과 모델 기반 선택·정제(Refinement)로 나뉜다. 규칙 기반은 비용이 낮지만 고정 휴리스틱에 묶여 인스턴스 수준 변이를 놓치고, LLM 기반 정제는 품질은 높여도 대규모 전처리에서 효율과 신뢰성이 떨어진다. 또한 함수 호출 기반 방법은 ProX·RefineX처럼 편집 함수 공간이 불완전하거나(삭제/수정 중심, 삽입 부재), 시드 감독과 실행 단계에서 중복·경계 문제를 완전히 해결하지 못한다.

- **Core Contribution**: UltraX는 대규모 pre-training 데이터를 function-calling으로 정제하되, 편집 함수 공간을 deletion·modification에 더해 insertion까지 확장해 인스턴스 수준의 fine-grained 편집을 가능하게 한다. 이어서 “신뢰 가능한 program-supervision 생성 파이프라인”을 구축해 원문-정제 텍스트 쌍을 계층적 매핑으로 구조화된 감독 신호로 바꾸고, 연산 조합 비율 통제와 저신뢰 예시 필터링으로 학습 분포 안정성을 높인다. 마지막으로 긴 문서에서도 실행 신뢰성을 올리기 위해 sliding-window 예측, 전역 operation aggregation, 체계적 후처리를 포함한 대규모 실행 절차를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 대규모 코퍼스에서 빠르게 작동하면서도 의미 수준 품질을 안정적으로 개선하는 감독 신호를 만들기, (2) 함수 호출 결과를 긴 문서에 적용할 때 중복 매칭·연산 간 간섭·창 경계 단절을 줄이기, (3) 경량 모델이 예측한 operations가 실제로 일관되게 실행 가능하도록 정규화·검증하는 것이다. UltraX는 dataset-adaptive prompt optimization으로 expert LLM의 고품질 end-to-end 정제 텍스트를 만들고, Line Alignment Mapping과 Dynamic Context Replacement로 모호한 위치 지정 문제를 줄인 뒤 저신뢰 필터링과 연산 조합 기반 샘플링 비율로 감독 품질을 조절한다. 실행 단계에서는 window별 예측을 전역 라인 번호 공간으로 재매핑해 비중복 영역만 유지하고, 파싱/정합성 검사·모호한 replacement 제거·인접/상호간섭 연산 병합·반복 패턴 fallback 등을 통해 안정성을 확보한다.

- **Empirical Impact**: 실험은 1B MiniCPM 모델을 5개 코퍼스(FineWeb, RedPajama-V2, AICC, Ultra-FineWeb, FineWeb-ProX-Doc)에서 20B 토큰 예산으로 from-scratch pre-training해 LightEval 성능을 비교했다. UltraX는 5개 코퍼스 전반에서 평균 성능이 가장 높았고, 50개 task-corpus 조합 중 34개에서 최상위를 기록했으며, 평균 상대 개선이 Raw 대비 약 2.00%, ProX-C 대비 약 1.53% 수준으로 보고된다. 특히 FineWeb에서는 16B 토큰에서도 Raw·ProX-C의 20B 최종 성능을 이미 상회해 data efficiency와 정제 신뢰성이 개선됐음을 보여준다.



### DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding (https://arxiv.org/abs/2607.08642)
Comments:
          23 pages, 2 figures, 11 tables. Code: this https URL

- **Prior Approaches**: 스펙ulative decoding은 드래프트 토큰을 여러 개 만들고 타깃 모델이 병렬로 검증한 뒤 가장 긴 접두사를 수용해 속도를 높이지만, end-to-end 이득은 ‘라운드당 수용 길이’와 ‘라운드당 드래프트/검증 비용’의 상충관계로 제한된다. DFlash 같은 블록-디퓨전 드래프터는 한 번의 병렬 패스로 블록을 제안하지만, 각 위치의 분포가 실제로 블록 내에서 선택될 다른 토큰들의 영향을 반영하지 못해(주파수처럼 경로 비조건) 수용 길이 상승에 한계가 있다. DDTree/CaDDTree는 best-first 트리를 구성해 수용 길이를 더 키우지만, 드래프터 분포를 위치별로 factorize(경로 무관)된 형태로 가정해 Domino처럼 경로 의존 보정(conditional correction)을 그대로 활용하기 어렵다.

- **Core Contribution**: DominoTree는 Domino의 GRU 기반 인과 보정이 만들어내는 경로 의존 분포를, 학습 없이 training-free로 best-first draft tree 점수화에 반영하는 방법을 제안한다. 기존 DDTree의 힙 빌드 로직은 유지하되, 각 노드의 root-to-node 경로에 대해 Domino의 conditional, non-factorized 보정 점수를 다시 계산해 노드 점수로 사용한다. 또한 계산을 실용화하기 위해 각 깊이에서 GRU 보정의 후보를 후보 top-M으로 제한해 트리 확장 비용을 억제한다.

- **Technical Challenges**: 가장 큰 어려움은 경로별 GRU 보정을 트리의 여러 노드에 적용하면, 노드마다 전체 어휘(vocabulary) 투영이 반복되어 드래프트-트리 ‘build’ 비용이 수용 길이 이득을 잠식한다는 점이다. DominoTree는 보정 계산을 노드별로 수행하되, 깊이별로 marginal top-M 후보만 대상으로 보정을 적용해 불필요한 전체 후보 확장을 줄이고, CUDA-graph 기반 GPU 네이티브 빌더로 파이썬 구현과 bit-identical 결과를 보장해 수용/검증 동작이 변하지 않도록 했다. 그 결과 트리를 더 크게/깊게 만들 때도 라운드당 생성 오버헤드를 낮춰 throughput으로 연결되게 만든다.

- **Empirical Impact**: Qwen3-4B에서 8개 벤치마크를 대상으로 평가한 결과, DominoTree는 autoregressive decoding 대비 최대 6.6배 speedup을 보이며 모든 온도 조건에서 어떤 방법보다 높은 mean accept length(라운드당 최대 10.7 토큰)를 달성했다. 또한 Domino의 기본 디코더 대비로도 모든 온도에서 처리량이 우세하며, Qwen3-4B에서는 전반적으로 9~10%, Alpaca에서는 최대 +22% 향상, DDTree/CaDDTree 대비로도 전 온도에서 우위를 보였다. Qwen3-8B에서도 모든 온도에서 수용 길이 리더를 유지하면서 T=0에서 DDTree 대비 +24%의 throughput 우위를 내고, higher temperature에서는 차이가 축소되거나 접히는 양상을 보이되 DFlash와 Domino에 대한 전반적 이득은 유지된다.



### It Takes a MAESTRO To Prune Bad Experts (https://arxiv.org/abs/2607.08601)
Comments:
          16 pages, 4 figures

- **Prior Approaches**: MoE 언어모델은 토큰당 일부 전문가만 활성화해 연산은 줄이지만, 전체 expert bank는 상시 메모리에 남아 배포 병목이 생긴다. 이에 구조적 가지치기가 대안으로 부상했지만, 기존 LLM 가지치기는 주로 dense 트랜스포머를 전제로 하거나 MoE에 로컬 휴리스틱(레이어·라우터 가중치 등)을 그대로 이식해 라우팅 의존성을 놓친다는 한계가 있었다. 특히 expert 중요도를 “국소적으로 얼마나 자주/크게 쓰는가”로만 판단하면, 레이어 간 라우팅 흐름의 상호작용을 반영하지 못해 압축 후 성능 변동이 커질 수 있다.

- **Core Contribution**: 논문은 MoE의 자동회귀 생성 중 expert 활성 궤적을 Ergodic Markov chain으로 모델링하고, 정상분포(stationary distribution)가 교차 레이어 라우팅 의존성을 담도록 설계한 가지치기 프레임워크 MAESTRO를 제안한다. MAESTRO는 (layer, expert) 상태의 정상확률 질량이 가장 작은 expert 슬롯을 제거 대상으로 삼아, “전역적으로 라우팅이 덜 흐르는” 전문가를 구조적으로 제거한다. 결과적으로 라우팅에 정합적인 전역 중요도 기준으로, 로컬 휴리스틱 기반 pruning의 제약을 완화한다.

- **Technical Challenges**: 핵심 과제는 MoE의 top-k 라우팅이 만드는 레이어 간 상호의존성을 전역 중요도 신호로 변환하는 것이다. 논문은 이를 위해 작은 보정 데이터에서 모델 자체의 autoregressive rollout으로 전이(transition) 통계를 추정하고, 주기성을 깨기 위한 self-loop smoothing 및 ergodicity 부여로 고유한 정상분포를 power iteration으로 계산한다. 이후 보존/삭제를 레이어별로 균형 있게 제한하면서 전문가 텐서를 실제로 슬라이싱해 추론 시 마스킹 없이 메모리와 파라미터를 동시에 줄이고, 성능 흔들림은 전문가·라우터는 고정한 채 attention에 LoRA recovery fine-tuning으로 보정한다.

- **Empirical Impact**: MAESTRO는 GPT-OSS-20B와 Qwen3-30B 두 MoE 계열에서 17개 벤치마크(안전, 편향·윤리 포함)로 평가되며 기존 SOTA 대비 평균 성능 보존을 최대 10.61%까지 개선했다. 특히 50% 압축 같은 공격적 조건에서도 task 간 분산이 낮아 “전반적으로 더 일관된 일반화”를 보였다고 보고한다. 또한 압축비를 25%→50%로 올릴 때 성능 저하 폭이 다른 기법들보다 완만해 stationary distribution 기반 전역 랭킹이 아키텍처에 덜 민감하고 견고함을 시사한다.



### When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability (https://arxiv.org/abs/2607.08535)
Comments:
          6 pages, 6 figures, 4 tables

- **Prior Approaches**: 그동안 LLM-as-judge 연구는 MT-Bench, Chatbot Arena 등에서 인간 선호와의 상관을 보이며 평가 도구로서의 유용성을 입증해 왔습니다. 다만 기존 보고는 “평가자 모델이 강해질수록 점수도 잘 맞는다”는 식의 1차원 모델 정렬에 머무는 경우가 많아, 능력 향상인지·벤치마크 슬라이스 차이인지·편향 감소인지·프로토콜 영향인지가 분리되지 않았습니다. 또한 position/verbosity 같은 편향과 집계·debate 같은 프로토콜 설계가 순위를 바꿀 수 있다는 점은 알려졌지만, 이 요인들이 평가자 스케일링과 함께 어떻게 변하는지는 덜 명확했습니다.

- **Core Contribution**: 이 논문은 evaluator-replacement ambiguity(평가자 교체 후 점수 변화의 원인 불명확성)를 측정 타당성 문제로 규정하고, 평가자 신뢰도를 단일 정확도로 보지 않도록 분해 프레임을 제시합니다. 구체적으로 (1) 단일 평가자 타당성, (2) 편향 견고성, (3) 반복/집계에서의 오류 비독립성, (4) debate 같은 프로토콜의 auditability(감사 가능성) 4요소로 신뢰도를 구성합니다. Qwen3 dense judge의 파라미터 스케일링과 MiniMax M2~M2.7의 released API 업그레이드 경로를 비교해, “업그레이드가 곧바로 상호교환 가능한 신뢰도 개선”이 아니라는 점을 실증적으로 보여줍니다.

- **Technical Challenges**: 가장 큰 기술적 도전은 “더 강한 judge = 더 신뢰할 만함”이지만, 실제 점수 변화가 편향·슬라이스·파이프라인 파싱/집계 차이로도 생길 수 있다는 점입니다. 논문은 adjacent 비교를 McNemar로 짝지어 수행하고, Holm correction으로 다중비교를 제어했으며, LLMBar에 대해 position/verbosity/granularity bias probes로 편향의 잔존을 직접 측정했습니다. 또 jury의 경우 다수결이 자동 증폭이 아니라 오류 상관(ρ)을 고려해야 함을 beta-binomial 형태의 ρ-보정 예측으로 확인했고, debate는 parser와 fallback 로그 부재 때문에 deliberation 인과 추정은 어렵지만 프로토콜 감사 로그가 필수임을 드러냈습니다.

- **Empirical Impact**: 4개 데이터셋에서 judge 업그레이드는 경로별로 다르게 나타났습니다. Qwen3는 1.7B→4B에서만 비교적 견고한 adjacent gain이 확인된 반면, MiniMax M2~M2.7 인접 릴리즈는 대부분 유의미한 개선으로 이어지지 않았고(홀름 보정 후에도 유의), 편향과 position/verbosity bias는 줄어들어도 완전히 사라지지 않았습니다. 다수결 jury는 오류 상관이 높아 이득이 제한적이었고, debate는 결정 변화를 크게 만들 수 있지만 파싱 실패 fallback의 재현 로그가 없으면 “토론 효과”로 귀속하기 어렵다는 점이 강조됩니다. 결론적으로 LLM-as-judge 보고서에는 데이터 슬라이스, 편향 및 바이어스 프로브, 오류 의존성(ρ) 추정, 그리고 프로토콜 audit trail까지 포함해야 신뢰할 만한 측정으로 해석할 수 있다는 권고가 제시됩니다.



### Cross-seed explainability using Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoders (https://arxiv.org/abs/2607.08499)
Comments:
          17 pages, 4 figures, 6 tables

- **Prior Approaches**: 기존 Sparse Autoencoder(SAE) 연구는 polysemanticity를 줄이고(예: TopK, end-to-end, orthogonality 제약) 단일 모델 내부에서 해석 가능한 특징을 뽑는 데 집중해 왔다. 다만 서로 다른 seed로 독립 학습한 경우 dictionary learning이 비볼록이라 피처 차원 정렬이 깨지고, 같은 의미가 서로 다른 latent 차원에 “feature splitting” 형태로 흩어지는 문제가 남는다. Feature Aligned SAE나 post-hoc 정렬(예: Procrustes) 같은 접근은 정렬을 돕지만, 기하학적 misalignment을 학습 과정에 충분히 반영하지 못할 수 있다.

- **Core Contribution**: 이 논문은 Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoder를 제안해, seed 간 활성 공간의 기하학적 misalignment 자체를 먼저 정렬한 뒤 한 개의 joint SAE로 특징을 추출한다. 핵심은 각 seed의 activation space를 orthogonal Procrustes 회전으로 미리 “맞춘” 다음, Top-K 희소성과 end-to-end downstream 최적화, 그리고 dead-feature revival 보조손실을 함께 사용해 cross-seed universal feature를 강화하는 것이다. 그 결과 seed 쌍 간 Pearson 상관이 r≥0.70 수준으로 수렴하는 특징이 더 많이 나타난다고 보고한다.

- **Technical Challenges**: 난제는 (1) non-convex dictionary learning 때문에 seed마다 feature space가 회전/재배치되며, (2) 정렬을 사후에만 하면 학습이 그 기하를 재료로 삼지 못해 universal 해석이 충분히 생기지 않는다는 점이다. 논문은 이 둘을 동시에 겨냥해, 500개 데이터로 계산한 orthogonal Procrustes rotation을 joint SAE 학습 중 activation에 적용하고, rotation-only·cross-loss ablation을 통해 “상관이 최적화 목표가 아니라 shared 구조를 반영”함을 함께 점검한다. 또한 Top-K로 L1 shrinkage bias를 피하고, end-to-end로 기능 일관성을 KL·MSE 기반으로 유지하면서 dead-feature revival 손실로 학습 공백을 줄인다.

- **Empirical Impact**: SST-2, Stanford Politeness, TweetEval Emotion의 3개 벤치마크에서 5개의 독립 seed pair(총 10개 BERT)로 평가했으며, 제안 파이프라인은 모든 데이터셋에서 사후 정렬 기반 기준선보다 cross-seed universal 특징 수와 상관이 더 높게 나온다고 한다. 특히 Procrustes를 post-hoc으로만 독립 SAE에 적용하면 일관된 개선이 없었지만, Procrustes conditioning을 joint 학습에 결합했을 때만 성능 향상이 나타나 효과의 귀속이 명확해진다. 추가적인 최소 정성 분석에서는 고-universality 특징이 sociolinguistic 패턴(예: 문장 초두 modal verb의 interrogative 배치 등)과 맞물리는 형태로 해석 가능함을 시사한다.



### Two Axes of LLM Abstention: Answer Correctness and Question Answerability (https://arxiv.org/abs/2607.08456)
- **Prior Approaches**: 기존의 abstention(응답 보류) 접근은 생성 답변의 불확실성 한 가지 값에 임계값을 걸어 “거절”을 결정합니다. 그러나 이 방식은 틀린 답(정답성 실패)과 아예 답하면 안 되는 질문(답불가능/거짓 전제)을 같은 축으로 다뤄 open-world 영역을 제대로 표현하지 못합니다.

- **Core Contribution**: 이 논문은 응답 보류가 필요할 때의 실패를 두 개의 별도 축—정답성(correctness)과 답가능성(answerability)—으로 분해해, 같은 결정에서도 신호가 서로 다르게 나타남을 보입니다. 또한 두 축 각각에 대해 별도 위험 예산을 캘리브레이션해, 각 임계값을 동시에 만족할 때만 답하도록 하는 정책을 제안합니다.

- **Technical Challenges**: 핵심 난제는 “모델이 틀리게 답할 확률”이 아니라 “해당 질문이 애초에 답 가능한지”를 신뢰성 있게 분리해 측정하는 것입니다. 연구진은 hidden states에서 선형 프로브로 답가능성 AUROC를 크게 끌어올리며(특히 CREPE의 자연발생 거짓 전제에서 출력 기반 신호는 거의 chance), 단순 premise-check 지시는 오히려 거짓/진실 전제를 무차별로 반박하게 만들어 probe 라우팅으로 이를 수정했습니다.

- **Empirical Impact**: SelfAware 및 자연 데이터 벤치마크 CREPE에서, 출력의 answer-confidence는 답가능성 구분에 거의 실패하는 반면 hidden-state 프로브는 이를 유의미하게 포착합니다. 더 나아가 두 축을 결합한 캘리브레이션 정책은 스케일이 커질수록 잘못 답하는 비율 상한이 조여지며, 14B에서는 단일 임계값 정책이 사실상 인증하지 못하는 상황에서도 유일하게 인증 가능한 결과를 보였습니다.



### Detecting Ladder Logic Bombs in IEC 61131-3 PLC Programs using ESBMC-PLC+: A Formal Verification Approach with Trigger Synthesis (https://arxiv.org/abs/2607.08417)
Comments:
          14 pages

- **Prior Approaches**: PLC의 Ladder Logic Bomb(LLB)은 정상 운용 중에는 숨고, 특정 트리거가 발화하면 작동기 조작·센서/ HMI 값 위조·운영자 제어 방해 같은 페이로드를 수행한다. 기존 연구들은 주로 트리거를 문법/구조 기반으로 탐지하거나, 형식 검증을 하더라도 기능 블록 내부 논리가 중간 표현(IR)에서 누락돼 폭탄이 보이지 않는 문제가 있었다. 또한 BOOLEAn/정수 문제에서는 잘 되더라도, 아날로그 제어의 비선형 비종료(non-termination) 같은 경우에는 SMT 기반 검증이 시간초과로 취약해질 수 있다는 한계가 지적됐다.

- **Core Contribution**: ESBMC-LLB는 ESBMC-PLC+의 검증 엔진을 그대로 쓰되, 함수 블록(Function Block) 바디에 숨어 있는 LLB를 노출시키는 modeling layer를 추가해 “폭탄 부재”를 정식 검증 문제로 바꿔낸다. 비종료 페이로드는 scan-watchdog로 안전 속성 위반을 유도하고, 작동기/센서 위조는 output wiring을 통해 safety property 위반으로 연결해 탐지를 성립시킨다. k-induction으로 모든 scan에 대해 bomb-absence를 무제한 증명하고, bounded model checking은 counterexample로 트리거(발화 조건)를 자동 복구한다.

- **Technical Challenges**: 핵심 기술 과제는 첫째, 기존 Ladder 다이어그램 검증이 함수 블록 내부 코드를 IR에서 생략해 폭탄 논리가 도달 불가능해지는 문제를 해결하는 것이다. 논문은 함수 블록 바디를 GOTO 기반 코드로 재구성해 ESBMC-PLC+가 해당 로직을 실제로 검증 가능하도록 만들고, 둘째는 비종료 폭탄을 PLC의 워치독 타이머 의미에 맞게 model로 표현하는 scan-watchdog 계측과, 셋째는 위조된 출력이 이후 연산·검증 조건에 전파되도록 output wiring을 설계한다. 마지막으로 SMT가 약한 비선형 비종료 영역에서는 CFG-triage가 구조적으로 유리하다는 “어디까지/어디서 깨지는지”도 실험으로 구분했다.

- **Empirical Impact**: 공개 Iacobelli 2024 데이터셋에서 ESBMC-LLB는 30/30 폭탄을 모두 탐지하고 모든 트리거를 복구했으며, CFG-triage를 회피하도록 만든 computed/opaque-arithmetic/multi-scan 트리거도 포함해 강건성을 보였다. PLC-Defuser의 SWaT 말뭉치에 대해 의미론(semantic) 기반 model checking을 최초로 실험한 결과, 아카이브 v1.0.0(선형 비종료)에서는 149/150(99%) 탐지와 무오탐 0건을 달성했지만, 후속 버전의 비선형 비종료에서는 SMT 시간초과로 탐지가 49%로 하락했다. 결론적으로 semantic model checking은 무제한 증명과 adaptive-trigger 강건성에서, CFG-triage는 비선형 아날로그 비종료에서 서로 보완적이며 어떤 접근이 “전부를 이긴다”기보다 역할이 나뉜다는 점을 데이터로 뒷받침한다.



### When Synthetic Speech Is All You Have: Better Call GRPO (https://arxiv.org/abs/2607.08409)
Comments:
          Submitted to SLT 2026

- **Prior Approaches**: 규제 산업(예: 은행)에서는 개인정보·생체정보 이슈로 실제 음성 데이터 수집과 재사용이 크게 제한돼, TTS로 만든 합성 음성을 전사(Transcript)에서 생성해 ASR 도메인 적응을 해왔다. 하지만 합성 음성은 실제 음성과 음향적으로 불일치가 있어 성능 향상이 제한되며, 이를 줄이려는 연구도 대체로 supervised fine-tuning(SFT) 틀 안에서 진행돼 왔다.

- **Core Contribution**: 이 논문은 합성 음성으로 LLM 기반 ASR을 적응할 때, 강화학습을 사용하면 SFT보다 synthetic-to-real gap을 더 잘 메울 수 있다고 주장한다. 특히 critic-free 방식인 Group Relative Policy Optimization(GRPO)을 적용해, 동일한 합성 음성 데이터로도 sequence-level 목표를 통해 WER를 크게 낮춘다. 합성-only 적응에서 WER이 36.71%→22.09%(SFT 대비 40% 상대 감소)로 개선되며, SFT 후 GRPO를 이어붙이면 45% 상대 개선까지 확장된다.

- **Technical Challenges**: 핵심 난제는 합성 음성의 국소적인 잡음(프로소디 불일치, 발음/채널 시뮬레이션 결함)이 학습 신호를 왜곡해 삽입 오류(hallucinated insertion)를 유발한다는 점이다. 연구진은 토큰 단위 cross-entropy가 기준 전사를 ‘모두’ 따라가려는 성향을 줄이는 대신, 샘플된 전체 가설을 WER 기반 보상으로 상대 비교해 좋은 전사를 강화하는 방식으로 해결했으며, 이 과정에서 critic 없이 group-relative advantage를 사용해 학습 안정성을 확보했다.

- **Empirical Impact**: 실험 결과 GRPO의 이득은 representation의 대규모 재학습보다 behavior 변화(특히 stopping calibration)에서 나왔다. SFT는 음성 지원이 끊긴 뒤에도 자신 있게 이어서 생성하는 반면, GRPO는 경계 이후 confidence를 더 빠르게 낮추고 삽입 꼬리(insertion tail)를 크게 줄였으며 오디오 토큰 주의(attention)도 더 정보가 안정적인 구간에 집중했다. 또한 WER 보상은 충분하며 복잡한 보상 결합은 불리할 수 있고, 합성 풀을 무조건 늘리는 것보다 ‘실제 5–10시간 + 합성 다량’ 조합이 대부분의 개선을 만든다는 실용적 결론을 제시한다.



### Prompt Compression via Activation Aggregation (https://arxiv.org/abs/2607.08399)
- **Prior Approaches**: 기존에는 고정된 프롬프트 접두사를 KV-caching·prefix caching으로 토큰/임베딩/계산을 줄였지만, 프롬프트 길이와 주의(attention) 비용 자체는 크게 줄이지 못합니다. 또한 길거나 임의의 컨텍스트를 짧게 압축하는 gisting/AutoCompressors 계열은 대체로 전체 모델 fine-tuning 같은 무거운 학습이 필요합니다. activation engineering은 잠재공간에서의 조향(steering)에는 강했지만, 프롬프트의 “효과”를 단일 벡터로 충실히 재현하는 압축 문제에는 해답이 부족했습니다.

- **Core Contribution**: 이 논문은 특정 지시문을 단일 “activation patch vector”로 압축해, 원래 토큰 시퀀스 대신 placeholder 토큰의 활성에 주입한 뒤 동일한 작업을 수행하게 하는 2-pass 프레임워크를 제안합니다. 특히 중간 레이어에서 뽑은 hidden state들을 learned weighted sum으로 압축해도, 전체 프롬프트를 그대로 처리했을 때 대비 정확도 하락이 2% 미만임을 보입니다. 또한 cross-layer 구조 관점에서 “중간 레이어 추출-초기 레이어 주입”이 효과적이라는 분석 패턴을 제시합니다.

- **Technical Challenges**: 핵심 난제는 (1) 단일 벡터로 충분한 의미 정보를 담을 수 있는지, (2) 목표 LLM에 직접 주입했을 때 그 벡터가 실제로 해석·통합될 수 있는지, (3) 과도한 표현력으로 인한 overfitting을 피하면서 일반화까지 확보할 수 있는지입니다. 저자들은 가중치 생성만 하는 가벼운 W-MLP(가중 합)와 end-to-end Transformer Compressor(TC)를 비교하며, 중간 레이어에서 정보를 추출하고 early injection으로 재주입해 최대한의 “처리 깊이”를 확보하도록 설계했습니다. 더 나아가 placeholder 토큰 선택과 주입 위치에 대한 ablation으로 개입이 효과적인 구간을 찾았습니다.

- **Empirical Impact**: 실험은 Llama3.1-8B-Instruct를 중심으로 여러 8B/1B 모델로 확장해 Toy Task(지시 기반 지식/변환)와 ARC-Easy(다지선다)에서 평가했으며, W-MLP가 masking 대비 큰 성능 회복을 보이면서 많은 설정에서 full-prompt 기준선에 가깝게 도달했습니다. TC는 학습 성능은 비슷해도 테스트에서 급격히 떨어져, 단일 벡터 압축이 “표현력”보다 “안정적인 압축 구조”에 좌우될 수 있음을 시사합니다. 의미 해석 가능성 측면에서도 learned 가중치가 핵심 키워드/엔터티에 더 크게 집중되는 경향과 정량적 추적 실험(재구문 85%)을 제시해, 실제 배치/재사용형 압축의 가능성과 함께 향후 RAG·로보틱스 VLA 등으로의 확장 여지를 넓혔습니다.



### Large-Language-Models-as-a-Judge in Theory-Agnostic Adaptive Metric-Alignment for Prototypical Networks in Personality Recognition (https://arxiv.org/abs/2607.08374)
- **Prior Approaches**: 기존 성격 인식 연구는 Big-5, MBTI처럼 특정 심리 이론의 라벨 체계에 맞춰 학습하는 경우가 많아, 데이터셋·문화권이 바뀌면 성능이 쉽게 흔들립니다. 또한 myPersonality처럼 대규모 공개 데이터가 프라이버시 이슈로 축소되면서 저자원 환경에서의 학습 한계가 커졌고, 성격 인식을 정적 분류로만 다루는 경향도 일반화를 제한했습니다.

- **Core Contribution**: 이 논문은 이론에 의존하지 않는 성격 추론을 목표로 JAM(Judge for Adaptive Metric-Alignment)을 제안합니다. JAM은 학습 과정에서 사전 정의된 성격 이론 라벨에 고정되지 않고, 텍스트로부터 공유된 잠재 심리 구조를 ‘latent pseudo-facets’로 발견해 개인의 잠재 심리 프로필을 추론합니다.

- **Technical Challenges**: 핵심 과제는 서로 다른 이론으로 라벨링된 이질 데이터에서 공통 구조를 잘 맞추면서도 라벨 노이즈·불확실성을 견디는 것입니다. 논문은 Attention-Pooled Graph Prototypical Network로 임베딩 공간에서 클러스터 기반의 구조 표현을 만들고, Cross-Theory Harmonization(인간 가이드 linkage + 기계 유도 consensus)으로 이론 간 정합을 강화하며, LLM-as-a-Judge를 LLM-before-the-loop/LLM-in-the-loop 두 방식으로 붙여 애매한 샘플을 선별해 적응적 메트릭 학습을 돕습니다.

- **Empirical Impact**: 실험에서는 JAM이 여러 프레임워크 간 일반화와 성능에서 기존 대비 개선을 보이며, 특히 low-resource personality theory에서도 유의미한 효과를 보였다고 보고합니다. 이는 성격 인식을 특정 심리 분류 체계에 종속시키던 관행에서 벗어나, 이론-불변적 표현 학습으로 확장하는 한 걸음을 제시한다는 점에서 의미가 있습니다.



### Echoes Across Vietnam's Highlands, Delta, and Coast: A Multilingual Corpus for Cham, Khmer, and Tay-Nung (https://arxiv.org/abs/2607.08362)
- **Prior Approaches**: 기존에는 베트남 소수 언어를 NLP에서 다루지 못한 경우가 대부분이었고, 다룬 모델이라도 mBERT, XLM-R, RemBERT 같은 다국어 인코더가 고정된 토크나이저 용량 때문에 Cham, Khmer, Tay-Nung을 심하게 조각내는 문제가 컸습니다. 또한 ELECTRA류 continued pretraining/RTD는 단일-스크립트 환경에서는 잘 작동하지만, 스크립트 이질성이 큰 설정에서는 생성기가 ‘형태/스크립트 힌트’만으로 교란을 만들어 판별기가 지름길을 학습할 위험이 있습니다. 데이터 희소성만이 아니라, 스크립트·표준화·접촉 양상이 적응 실패의 직접 원인이라는 점이 기존 연구에서 충분히 드러나지 않았습니다.

- **Core Contribution**: 논문은 Cham, Khmer, Tay-Nung을 위한 첫 대규모 말뭉치 겸 벤치마크 CKTN(44,367 문서, 24M+ subword)을 공개하고, continued pretraining, 28-way category classification, summary-document retrieval까지 한데 묶어 평가 지형을 제시합니다. 동시에 기존 적응에서 자주 쓰이던 지표가 오해를 부를 수 있음을 보여주며(언어모델링 손실 감소·lexical-overlap 검색 성능 상승이 의미론적 일반화 실패를 가릴 수 있음), 이를 해결하는 스크립트 인지 적응 레시피를 제안합니다. 핵심은 어휘 확장(vocabulary augmentation)과 난이도 보정된 replaced-token pretraining(스크립트 호환 필터링 포함)으로 discriminator가 스크립트 불일치 같은 사소한 단서를 쓰지 못하게 하는 것입니다.

- **Technical Challenges**: 주된 기술 난제는(1) 다국어 토크나이저가 해당 언어들을 토큰 단위로 과도하게 분절해 생성기의 교란이 부정합/잡음이 되기 쉽고, (2) ELECTRA 스타일 RTD에서 약한 from-scratch generator가 스크립트별로 ‘티 나는’ 치환을 만들어 discriminator가 의미가 아닌 표면 단서를 학습한다는 점입니다. 이를 위해 소스 어휘 기반으로 새 vocabulary 후보를 선별·스크립트/형식 유효성 필터링하고, 새 토큰의 점수와 임베딩을 원 토큰 분해를 통해 보정·초기화해 어휘 분절을 줄였습니다. 더 나아가 replacement sampler에 스크립트 호환성·형식 적합성·embedding 유사도 대역·잘못된 제어/특수 토큰 배제를 넣고, RTD 손실은 선형 스케줄로 점진 도입해 학습 불안정도 완화했습니다.

- **Empirical Impact**: 실험 결과, CKTN-ELECTRA는 토크나이즈 품질 지표에서 fertility/continuation-token ratio를 크게 낮춰(예: continuation-token ratio 11% 미만) 어휘 조각화를 완화하며, category classification에서 가장 강한 성능(accuracy 0.9214, Macro-F1 0.7103)을 보였습니다. 반면 lexical-overlap 기반의 retrieval(MRR@10/Recall@10)은 특히 다른 모델들과의 격차가 작게 나타나, 의미론적 일반화를 반영하지 못할 수 있는 평가 신호의 한계를 실증적으로 보여줍니다. ablation에서도 스크립트 인지 필터링과 RTD 선형 스케줄이 모두 분류 성능을 좌우하며, 제거 시 정확도·Macro-F1이 거의 무작위 수준으로 붕괴하는 패턴이 확인됐습니다.



### Grounded Event Extraction from SEC 8-K Filings with a Fine-Grained Taxonomy (https://arxiv.org/abs/2607.08346)
Comments:
          9 pages, 8 figures, 1 table. Full dataset and taxonomy available at this https URL

- **Prior Approaches**: Form 8-K의 SEC item code는 32개로 고정돼 있지만, 한 항목이 서로 다른 경제적 사건을 함께 묶는 등 라벨이 거칠고(예: Item 5.02), 시장에 영향 큰 내용도 Item 8.01 같은 catch-all에 몰리는 한계가 있었다. 그래서 연구에서는 item code나 뉴스 텍스트 기반 분류, 혹은 LLM을 활용한 태깅을 시도했지만, 대부분은 출력의 출처(원문 근거) 추적성과 신뢰도 검증이 약하거나 사후 judge에만 의존했다.

- **Core Contribution**: 이 논문은 8-K 공시에서 경제 사건을 3단계 119개 event type으로 세분화해 태깅하는 LLM 2-stage 추출 시스템을 제안한다. 핵심은 모든 태그가 (1) 고정된 어휘(스키마) 안에서만 나오고, (2) 원문에 실제로 존재하는 verbatim quote에 근거해 라벨링되며, (3) 두 번째 패스에서 quote만 다시 읽고 1~5 품질 점수를 부여한다.

- **Technical Challenges**: 기술적 어려움은 fine-grained 라벨을 만들더라도 결과가 “검증 가능”해야 한다는 점이다. 이를 위해 1단계에서 스키마 validation으로 taxonomy 밖 출력을 거르고, fuzzy n-gram 기반 quote validation으로 허구 인용을 구조적으로 차단하며, 2단계에서 grader가 cited quote와 범주 정의만 보고 quality score를 재평가해 score가 실제 정밀도와 함께 움직이도록 보정했다.

- **Empirical Impact**: 2022~2026년(6월)까지 292,984개 공시를 대상으로 601,088개의 grounded event tag(quality score 포함)를 만들었고 공개했다. LLM judge 평가에서 precision은 quality score 1에서 12.3%→5에서 96.4%로 단조 상승했으며, score에 따라 unsupported(원문에 사건이 없는 태그)는 8%에서 거의 0%로 떨어졌다; 또한 경제성 검증(event study)에서도 item code 내부에서도 태그별로 반응 크기가 유의미하게 달라져, coarse한 item code만으로는 설명이 부족함을 보였다.



### TypeProbe: Recovering Type Representations from Hidden States of Pre-trained Code Models (https://arxiv.org/abs/2607.08339)
Comments:
          18 pages, 12 figures. Accepted at ESSLLI 2026 (StuS; double-blind)

- **Prior Approaches**: 기존 코드 모델 해석 연구는 주로 문법·식별자·네임스페이스 같은 표면 정보의 인코딩을 진단하는 데 집중해 왔다. 반면 형식 타입 의미(typed semantics)는 제대로 정조준되지 않았고, 타입 제약 디코딩 같은 외부 기법은 언어별 규칙을 추가로 필요로 한다는 한계가 있었다. 또한 다국어에서 표현이 정렬된다는 논의는 자연어 중심이었고, 코드 모델의 타입 정보가 언어를 넘어 어떻게 전이되는지는 불명확했다.

- **Core Contribution**: 이 논문은 pretrained 코드 모델의 residual stream에서 타입 표현이 선형으로 decodability 되는지(어느 레이어에 나타나는지), 그리고 Java와 Python 사이에서 교차언어로 전이되는지를 직접 파고든다. Java·타입 주석 없는 Python·타입 주석 있는 Python을 동일한 프로그램 구조로 맞춘 병렬 데이터셋과, FIM 기반 masked call site에서 타입 호환성으로 함수를 선택·결과 타입을 추론하는 프로빙 태스크를 제안한다. 특히 untyped 코드에서도 교차언어 타입 표현이 형성됨을 보이며, typed function application에서 암시되는 결과 타입까지 내부적으로 재구성되는 구조를 실험적으로 확인한다.

- **Technical Challenges**: 핵심 과제는 (1) 타입 신호와 식별자·리터럴 같은 어휘적 단서를 분리하고 (2) 선형 프로브가 실제로 decodable 정보를 잡는지, (3) 언어 문법 차이 속에서도 타입 구조가 같은 방향으로 유지되는지 검증하는 것이다. 이를 위해 입력 예시의 변수명·함수명·리터럴을 랜덤화해 lexical shortcut을 차단하고, adversarial renaming으로 식별자가 ‘거짓 타입’을 가리키게 만들어도 타입 표현이 유지되는지 Δ selectivity로 측정한다. 또한 한 언어로 학습한 프로브를 다른 언어 데이터에서 zero-shot transfer로 평가해, 타입 관련 방향이 공유되는지 확인한다.

- **Empirical Impact**: SantaCoder-1.1B와 CodeLlama-7B 모두에서 base type과 list 컨테이너에 대한 선형 타입 표현이 교차언어로 전이되며, 특히 untyped Python(pyUnt)→Java 전이가 의미 있게 나타나 타입-관련 신호가 주석 없이도 회수될 수 있음을 시사한다. 다만 adversarial renaming에 대해선 selectivity가 부분적으로만 견조하며, late-to-mid 레이어에서 타입 decodability 피크가 나타나다가 공격에서 더 크게 흔들리는 패턴이 관찰된다. 전반적으로 코드 모델 내부에 언어를 가로지르는 ‘타입 manifold’가 형성될 수 있다는 실증 근거를 제공하면서, 타입 제약 디코딩과의 보완적 방향성까지 제시한다.



### XALPHA: A Memory-Driven AI Quant Researcher for Hypothesis-to-Code Alpha Discovery (https://arxiv.org/abs/2607.08332)
- **Prior Approaches**: 기존 알파 마이닝은 사람이 경제적 직관으로 팩터를 설계하거나, 딥러닝·진화탐색·강화학습으로 자동 탐색을 수행해 왔다. LLM 기반 연구는 자연어 아이디어를 팩터로 바꾸거나(AlphaGPT), 연구·평가 일부를 에이전트화(FAMA, AlphaAgent 등)했지만, 대체로 팩터 생성·수정·평가 같은 고립된 단계만 자동화하는 경향이 강했다. 또한 외부 리서치 지식 흡수, 코드 구현, 가설-코드-금융 타당성 검증, 그리고 발견 피드백 축적을 하나의 닫힌 루프로 묶지 못했다.

- **Core Contribution**: 이 논문은 알파 발견을 “연속적인 가설-코드” 연구 루프로 재구성하는 메모리 기반 AI Quant Researcher XAlpha를 제안한다. XAlpha는 리서치 리포트에서 뽑은 지식(Report-to-Memory Absorption)과 과거 세대/사이클의 discovery feedback을 함께 저장하고, 이를 바탕으로 연구 테마 계획-팩터 코드 생성·검증-실증 결과 반영까지 한 흐름으로 연결한다. 결과적으로 alpha mining을 단발 팩터 생산이 아니라 읽기→가설→구현→검증→성찰→진화의 closed-loop 과정으로 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 금융 리포트를 단순 문서로 프롬프트에 넣는 방식이 아니라 OHLCV 팩터 워크플로에 맞는 구조화된 지식으로 변환하는 것, (2) 가설의 금융적 그럴듯함과 코드 로직이 맞물리도록 사전( ex-ante ) 정합성을 강제하는 것, (3) 반복 실패 패턴을 피하면서도 탐색의 다양성을 유지하는 것이다. XAlpha는 이를 위해 A/B/C 3단 분류(OHLCV 가능성 선별, mechanism-family 분류, Research Archetype 큐 생성)로 리포트 지식을 캐싱하고, AST gate·tri-alignment judge·누수/품질 게이트 및 unit test로 코드 후보를 엄격히 통과시킨다. 동시에 novelty injection과 상관도 기반 중복 제거, mechanism 레벨 mutation/crossover/refinement으로 진화를 설계해 탐색 효율을 높인다.

- **Empirical Impact**: CSI300(10일 예측, Qlib 기반) 실험에서 XAlpha는 Ridge 등 예측 모델 및 Alpha360·AutoAgent·AlphaAgent·R&D-Agent-Quant·CogAlpha 같은 알파 마이닝 비교군 대비 전반적으로 더 강한 알파 발견 성능을 보였다. 정보계수(IC/RankIC)와 시간 안정성(ICIR/RankICIR), 그리고 포트폴리오 지표(연환산 수익·초과수익·정보비율)에서 경쟁 방법을 상회하는 결과가 보고된다. 연구 루프와 메모리 기반 피드백 통합이 팩터의 예측력뿐 아니라 반복 탐색의 누적 개선에도 의미 있음을 시사한다.



### Best-of-$N$ TTS Evaluation is Confounded by ASR Family Alignmen (https://arxiv.org/abs/2607.08256)
Comments:
          Accepted at ICML 2026 Workshop on Machine Learning for Audio

- **Prior Approaches**: flow-matching 기반 zero-shot TTS는 자연스러움과 화자 유사성은 크게 개선됐지만, 여전히 일부 발화에서 단어 수준 내용 오류가 남아 있다. 이를 줄이기 위해 BoN(Best-of-N) 추론처럼 여러 후보를 만든 뒤 ASR verifier로 텍스트 일치도를 기준 삼아 하나를 고르는 방식이 표준으로 쓰인다. 다만 어떤 ASR verifier(예: Whisper, wav2vec 2.0, HuBERT 계열)를 쓰는지에 대한 평가는 대체로 단일 고정 ASR에 의존해 체계적으로 점검되지 않았다.

- **Core Contribution**: 이 논문은 BoN에서 핵심적인 “검증기(ASR verifier) 랭킹”이 verifier의 선택이 아니라 “평가에 쓰는 ASR 계열(evaluator)”에 따라 뒤집힐 수 있다는 평가 교란을 규명한다. 특히 동일한 생성 결과라도 Whisper 계열과 wav2vec 2.0/HuBERT 계열 평가자 사이에서 정렬 방향이 반대로 나타나, 같은 verifier를 써도 보고되는 WER이 달라질 수 있음을 보여준다. 이를 바탕으로 서로 다른 ASR 계열을 함께 고려하는 cross-family rank ensembles(순위 평균, 교집합 형태 max-rank)를 제안한다.

- **Technical Challenges**: 관찰된 교란이 “표상 유사도(예: audio encoder 표현의 CKA)” 때문인지 확인하는 것이 큰 기술 과제였는데, 선형 CKA가 높아도 WER 랭킹이 반드시 일치하지 않는 패턴이 나타났다. 대신 verifier와 evaluator가 같은 계열/계보(lineage)일 때 선택이 과대평가되는 identity- 또는 lineage-level coupling 가능성을 시사하며, 이를 방지하려면 후보 선택 단계에서 계열을 분산시켜야 한다. 저자들은 (w2v2-base, distil-v3)처럼 서로 다른 ASR 계열 verifier를 고정하고, rank-avg와 max-rank로 교차 계열 집계를 수행해 평가자 의존성을 완화했다.

- **Empirical Impact**: LibriSpeech-PC test-clean에서 N=10일 때 cross-family rank ensembles은 평균 WER을 1.61%까지 낮추며, F5-TTS 기준으로 -12% 상대 개선을 보였다. 또한 공식 F5-TTS evaluator만 봤을 때도 최선의 단일 same-family verifier가 WER을 2.06%에서 1.72%로 -16.5% 낮추는 등 계열 선택 효과가 뚜렷했다. SIM-o(화자 유사도)와 UTMOS(자연도)는 구성 전반에서 거의 변하지 않아, 선택 편향을 줄이면서도 품질 저하 없이 성능을 개선할 수 있음을 실험적으로 뒷받침하며 “최소 2개 이상의 disjoint ASR 계열로 교차 검증”을 기본 보고 관행으로 권고한다.



### Diarization-Guided Qwen-ASR Adaptation for Multilingual Two-Speaker Conversational Speech (https://arxiv.org/abs/2607.08208)
Comments:
          4 main pages plus 1 page of reference

- **Prior Approaches**: 다중 화자 대화형 음성에서 언어가 섞인 경우, 기존 접근은 주로 end-to-end ASR 단일 모델에 의존하거나, 화자 분리(diarization)와 ASR을 느슨하게 결합해 성능을 끌어올리는 방식이 많았다. 하지만 화자 경계 오차와 분리된 구간의 언어·잡음 특성이 달라지면 ASR이 불안정해져 tcpMER 같은 종합 지표가 쉽게 악화된다.

- **Core Contribution**: 본 논문은 MLC-SLM 2026 Challenge Task 1(다국어, 두 화자 대화형 음성)용 모듈형 시스템을 제안한다. 화자 분리 전단(voice activity detection, CAMPPlus 임베딩, two-speaker spectral clustering, RTTM 기반 구간화)으로 화자-속성 구간을 만든 뒤, 언어/지역별로 challenge-adapted Qwen3-ASR-1.7B를 디코딩하는 구조가 핵심이다.

- **Technical Challenges**: 어려움은 (1) diarization으로 생성된 구간의 품질이 ASR 입력에 직접 영향을 주고, (2) challenge 도메인 차이로 인해 ASR이 WER/CER에 민감하게 흔들린다는 점이다. 저자들은 이를 위해 먼저 supervised full fine-tuning을 수행하고, 이어서 three-pipeline TTS 기반 synthetic speech augmentation으로 LoRA fine-tuning을 한 뒤, GRPO 강화학습에서 WER/CER 보상과 hallucination·repetition·길이 이탈에 대한 페널티를 함께 설계해 모델을 추가로 정교화했다.

- **Empirical Impact**: 공식 개발 세트에서 전체 시스템은 평균 tcpMER 23.70을 기록했으며, released Qwen-ASR-1.7B 대비 절대 6.83 포인트의 오류율 감소를 달성했다. 최종 평가 세트에서도 평균 tcpMER 17.97을 보였고, ablation 결과 supervised fine-tuning이 가장 큰 이득을 제공했으며 synthetic-speech LoRA와 reinforcement learning이 추가로 강건성을 높였다는 점이 확인됐다.



### Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models (https://arxiv.org/abs/2607.08186)
Comments:
          30 pages, 9 figures

- **Prior Approaches**: 기존 LLM 스케일링은 Transformer 백본을 키우는 방식에 의존해 왔지만, 이미 강한 모델에서는 추가 백본 확대가 또 다른 대규모 pretraining과 비용 상승을 동반한다. 이런 전제에서 looped(깊이-순환) Transformer는 토큰당 계산을 늘리지만, pipeline parallelism과 잘 맞지 않아 초대형 학습으로 확장하기 어렵다는 한계가 지적돼 왔다. 길이 확장도 일부 시도되었지만, 큰 모델에서 효율적으로 학습 가능한 형태의 “지속되는 중간 상태” 설계가 부족했다.

- **Core Contribution**: 이 논문은 Transformer 레이어를 늘리지(혹은 폭을 넓히지) 않고도 토큰당 내부 계산을 늘리는 sequence-length scaling 방법 Hidden Decoding을 제안한다. Hidden Decoding은 각 토큰을 n개의 stream으로 확장한 뒤, 최종 stream만 다음 토큰 예측에 사용하면서 중간 stream의 key-value cache를 다음 위치까지 컨텍스트로 유지해 더 많은 “숨은 연산”이 축적되도록 만든다. 추가로 스트림 간 혼합 비용을 제어하기 위해 Stream-Factorized Attention을 도입해 확장( n )이 스케일링 축으로 작동함을 보여준다.

- **Technical Challenges**: 핵심 난제는 stream 확장을 길이 차원에서 수행하면 전체 attention 비용이 n에 대해 급격히 커질 수 있다는 점이다. 이를 해결하기 위해 대부분의 레이어는 intra-stream(각 stream 내부)만 attention하고, stream 간 혼합은 일부 레이어에서만 수행하도록 마스킹을 설계해 attention 비용을 대략 선형에 가깝게 유지했다. 또한 CPT(continued pretraining) 환경에서 여러 stream 임베딩이 안정적으로 학습되도록 확장 계수를 1→2→4→8로 점진 성장시키는 progressive expansion과 Cyclic Replication Initialization을 사용했다.

- **Empirical Impact**: 실험은 100B+ MoE 규모에서의 스케일링을 직접 검증한다: WeLM-HD4-80B( n=4 )와 WeLM-HD4-617B( n=4 )는 각각 WeLM-80B/WeLM-617B 대조군 대비 9개 공통 벤치마크에서 전반적 향상을 보였다. 특히 SciCode, PHYBench, FrontierMath 같은 수학·과학 계열에서 큰 폭의 개선이 나타났고, GPQA Diamond 및 FrontierMath에서도 이득이 관측됐다. 확장 계수 n을 2→4→8로 키우는 실험에서는 성능이 단조롭게 커지는 경향이 확인되어 Hidden Decoding이 “고정 백본” 경로로서 실용적인 sequence-length scaling임을 뒷받침한다.



### SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation (https://arxiv.org/abs/2607.08161)
Comments:
          Accepted at IEEE SMC 2026

- **Prior Approaches**: Text-to-SQL은 자연어로 데이터베이스를 질의하는 핵심 작업이지만, 초기 규칙·템플릿 방식은 확장성이 낮았다. Seq2SQL, SQLNet처럼 스케치 기반 디코딩이나 강화학습을 쓰는 신경 접근은 성과를 끌어올렸지만, 복잡한 스키마로 일반화하기 어렵거나 학습/추론이 비싸지는 한계가 있었다.
또한 TAPAS, TaBERT, RAT-SQL, PICARD 등 구조 인식을 강화한 모델도 나왔지만, 최근 SOTA 성능은 T5/Codex 계열 LLM에 크게 의존해 자원 제약 환경 배포가 어렵다는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 SQuaD-SQL(Small-Qualified and Distilled for SQL)로, 작은 언어 모델이 LLM 수준에 가까운 Text-to-SQL 성능을 낼 수 있도록 teacher–student 학습 틀을 제안한다. 핵심은 LLM을 추론에 쓰지 않고, LLM이 생성한 구조화된 SQL 감독 신호(합성 데이터)를 통해 작은 모델이 “구조적 SQL 추론 행동”을 내재화하도록 하는 것이다.
여기에 parameter-efficient fine-tuning(LoRA)과 도메인 적응용 합성 데이터 생성까지 결합해, 제한된 자원에서도 실용적으로 학습·운용 가능하다는 점을 강조한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) LLM이 생성한 SQL 감독 신호가 가끔 문법·의미·실행 측면에서 틀릴 수 있다는 점과 (2) 작은 모델이 복잡한 구문 생성 능력을 안정적으로 학습하도록 만드는 학습 설계 문제다. 논문은 WikiSQL 스키마 정보를 활용한 task-specific 프롬프트로 합성을 시작하고, SQL 구문 파서 검증→LLM 기반 self-evaluation 점수(신뢰도 임계값)→규칙 기반 execution validation(실행 결과 유효성)이라는 3단계 필터로 노이즈를 제거한다.
그 다음 Qwen-1.5B를 LoRA로 학습하되, causal language modeling에서 프롬프트 토큰은 loss masking(-100)으로 제외해 SQL 생성 구간 학습에 집중하도록 최적화한다.

- **Empirical Impact**: WikiSQL에서 1.5B 파라미터 학생 모델(Qwen-1.5B)은 test execution accuracy 86.9%를 달성해, 전통적 Text-to-SQL 모델들과 task-specific 학습 없는 SLM을 크게 앞섰다. 또한 dev에서 86.5%로, 합성 감독이 성능 격차를 상당 부분 메워준다는 점을 확인했다.
특히 ablation 결과 prompt engineering 단독의 개선(35.6%→45.6%)보다, LLM 기반 distillation이 최대 기여를 했고(→80.3%), 데이터 필터링이 후속으로 정확도를 추가로 끌어올려 최종 86.9%를 만든 것으로 나타났다. 연구는 “스케일만이 아니라 지도 신호 품질·학습 전략”이 자원 제약 환경에서 실용적인 Text-to-SQL을 가능하게 한다는 메시지를 실증적으로 뒷받침한다.



### LEXIC: Lightweight Eye-tracking eXtension via Injected Complexity (https://arxiv.org/abs/2607.08152)
Comments:
          Accepted to APCCAS 2026

- **Prior Approaches**: EyeBench의 읽기 이해(이진 분류)에서 텍스트를 함께 쓰는 모델은 AUROC 56~63%까지 올라가지만, gaze-only 모델은 거의 우연 수준에 머뭅니다. BEyeLSTM처럼 언어 정보를 추가하되 language model을 쓰지 않는 방식은 중간 성능을 보이나, 여전히 본문 텍스트와의 직접 상호작용 신호가 제한적입니다.
또한 텍스트-어웨어 모델은 성능은 높지만 PLM 추론 비용이 커서, 배포 제약이 있는 상황에서 “gaze-only를 얼마나 끌어올릴 수 있는지”가 남은 질문입니다.

- **Core Contribution**: 본 논문은 inference 시 language-model forward pass 없이, 미리 계산된 단어 수준 난이도 신호로 gaze-only 모델을 보강하는 두 가지 경량 주입 메커니즘(LEXIC-Concat, LEXIC-Res)을 제안합니다. GPT-2 surprisal, word frequency, word length의 세 신호를 fixation 입력에 투입해 텍스트 측 신호의 일부를 전달합니다.
그 결과, OneStop에서 두 메커니즘 모두 Unseen Text 기준 AUROC를 유의하게 개선하며, 특히 LEXIC-Concat은 Unseen Reader에서도 추가 상승을 보입니다.

- **Technical Challenges**: 핵심 기술 과제는 “gaze-only”라는 입력 제약 하에서 텍스트 난이도 신호를 모델이 유효하게 활용하도록 결합하는 것입니다. 저자들은 (1) 세 난이도 신호를 단순 채널로 추가하는 LEXIC-Concat과, (2) 난이도로 typical-reader gaze response를 예측하고 관측 gaze와의 잔차를 입력에 반영하는 LEXIC-Res를 설계해 비교합니다.
또한 잔차 주입의 residual head가 훈련 독자(population-averaged calibration)에 맞춰져 있을 때 out-of-distribution 독자에게 전이 성능이 약해지는 구조적 경계를 실험적으로 확인하고, 이로 인해 LEXIC-Res는 Unseen Reader에서 이득이 감소합니다.

- **Empirical Impact**: OneStop 평가에서 10-fold 교차검증과 K=5 seed-ensemble(총 10개 폴드)로 AUROC 개선이 일관되게 관측됩니다. Unseen Text에서 LEXIC-Concat과 LEXIC-Res는 각각 +1.8~+2.2%p 수준의 통계적으로 유의한 상승을 보이며, LEXIC-Res 역시 Unseen Reader에서 +1.8%p 수준이지만 유의성은 약합니다.
반면 LEXIC-Concat은 Unseen Reader에서 +2.9%p로 유의하게 향상되어, “언어모델 없이도 gaze 기반 읽기 이해 예측의 성능을 올릴 수 있다”는 실용적 방향을 제시합니다. 이는 EyeBench 리더보드의 PLM 의존 격차를 경량 특징 주입으로 일부 메울 수 있음을 보여준다는 점에서 의미가 큽니다.



### ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR Post-Correction for Historical Documents (https://arxiv.org/abs/2607.08143)
Comments:
          17 pages

- **Prior Approaches**: 기존 OCR post-correction 연구는 규칙 기반, 통계 모델, 시퀀스-투-시퀀스 신경망 등으로 이어졌지만, 언어·문서 유형·잡음 특성에 따라 성능이 크게 흔들려 왔습니다. LLM을 활용한 시도도 zero-shot 프롬프트부터 fine-tuning, 탐지-생성 하이브리드까지 다양했지만 실험 설정이 달라 직접 비교가 어렵고, 무엇보다 hallucination(그럴듯한 내용 덧붙이기) 위험이 충분히 정리되지 않았습니다.

- **Core Contribution**: HIPE-OCRepair-2026은 LLM-assisted OCR post-correction을 ICDAR 경진대회 형태로 재정의하고, 재현 가능한 평가 체계를 제공합니다. 또한 언어(영어·프랑스어·독일어)와 시대(17~20세기), 문서 유형(신문·인쇄물)에 걸친 통합 멀티링구얼 벤치마크(HIPE-OCRepair-2026 dataset)를 제공해, 서로 다른 기존 데이터셋을 통일된 분할·전사 지침으로 묶었습니다. 평가 관점도 문자 생성 그 자체보다 검색·접근에 유리한 언어적 정확도를 우선하는 retrieval-oriented 점수 체계를 채택했습니다.

- **Technical Challenges**: 핵심 난제는 이미지 없이 OCR 텍스트만 주어졌을 때 심각한 왜곡을 복원해야 하며, 동시에 원문에 없는 내용을 새로 만들지 말아야 한다는 점입니다. 논문은 이를 해결하기 위해 cMER 기반 정량 지표(삽입이 분모에 반영되어 과생성에 덜 민감)와 항목 단위 선호 점수(pref_score)를 함께 사용하고, 레이아웃 정규화·IR 스타일 정규화로 검색 시나리오에 맞춘 공정한 비교를 구성했습니다. 참가 팀들은 보수적 온도, 출력 길이/형식 제약, 문서 메타데이터 활용, judge-and-retry 같은 재시도 루프 등으로 hallucination과 과교정을 제어하려 했습니다.

- **Empirical Impact**: 실험 결과 modern LLM-assisted 시스템은 전반적으로 OCR 품질을 유의미하게 개선하지만, 데이터셋·언어·잡음 수준에 따라 격차가 크며 특히 저잡음 입력에서 over-correction이 반복되는 문제가 관찰됐습니다. 최고 성능은 BnF-Mistral 계열이 차지했으며, 독일어·영어·프랑스어 모든 공식 테스트셋에서 1위를 기록했고 기준 no-correction 대비 cMER가 전반적으로 크게 감소했습니다. 또한 제출물과 데이터, scorer, 평가 파이프라인을 공개해 향후 연구자들이 동일한 평가 틀에서 시스템을 체계적으로 비교·개선할 수 있도록 했다는 점에서 의미가 큽니다.



### COALA: Robust Contextualized Speech-augmented Language Modeling for ASR via Contrastive Regularizer and Biasing Score Estimation (https://arxiv.org/abs/2607.08117)
Comments:
          Accepted at INTERSPEECH 2026

- **Prior Approaches**: 기존 contextual biasing은 추론 단계에서 shallow fusion, on-the-fly rescoring처럼 검색/디코딩에 편향을 주거나, 학습 단계에서 어텐션 기반 어댑터·trie 포인터 제너레이터로 외부 지식을 E2E로 내재화하는 흐름으로 발전해왔다. 하지만 SLM의 context-window 한계와 다수 엔티티 동시 등장 시 distractor 간 간섭이 커져 성능이 쉽게 흔들린다. 특히 multi-target에서 discriminative loss가 학습 붕괴를 겪거나 수렴을 위해 보조 log loss가 필요하다는 문제가 남아 있다.

- **Core Contribution**: COALA는 SLM에 맞춘 contextual biasing 프레임워크로, 대규모 biasing list 안에서 어떤 엔티티가 오디오에 실제로 매칭되는지 먼저 선별하는 “biasing target identification”을 핵심으로 제시한다. 이를 위해 SLM의 latent representation을 특화된 판별 공간으로 사상해 음성 구간과 후보 엔티티 간 matching intensity를 점수화한다. 또한 MPD-Loss와 DPD-Loss를 도입해 multi-entity 상황에서도 안정적으로 학습되도록 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) context-window 제약 때문에 후보 list를 그대로 프롬프트에 넣기 어렵고, (2) multi-target 학습에서 양(positive) 엔티티 간 gradient conflict가 생겨 수렴이 깨질 수 있다는 점이다. COALA는 DPD-Loss로 softmax 기반 상대순위 경쟁을 제거하고, 각 엔티티를 독립적인 점별 이진 분류로 학습해 0을 기준으로 하는 안정적인 절대 판별 경계를 만들었다. 더불어 <<unbiased>> 토큰 점수를 임계값으로 사용해 낮은 점수 후보로 인한 성능 저하를 줄인다.

- **Empirical Impact**: LibriSpeech에서 COALA는 다양한 biasing list 규모에서 기존 discriminative loss 및 프롬프트 기반 기준선보다 일관되게 contextual biasing 성능이 좋았다. DPD-Loss 학습은 Recall#20이 99%대에 도달해 목표 엔티티가 상위 후보에 거의 항상 포함되며, 이는 특히 test-other처럼 어려운 조건에서 B-WER 개선으로 이어진다. 결과적으로 COALA의 정밀한 엔티티 랭킹은 SLM의 편향 주입 효율을 높여 B-WER을 유의미하게 낮추는 영향력을 보여준다.



### MASTE: A Multi-Agent Pipeline for Zero-Shot Aspect Sentiment Triplet Extraction (https://arxiv.org/abs/2607.08080)
- **Prior Approaches**: ASTE는 (aspect, opinion, sentiment) 트리플을 한 문장 안에서 정확한 span까지 맞춰 추출해야 하는 세밀 구조화 과제다. 기존 접근은 파이프라인/시퀀스 태깅/생성/그리드 태깅 등 지도학습 중심이며, 데이터별 triplet 주석과 span 규약에 의존해 도메인 전이에 불리했다.

- **Core Contribution**: 논문은 zero-shot ASTE에서 LLM이 단일 패스 생성 중 aspect–opinion 매칭과 span 경계를 동시에 처리하며 실패하는 문제에 주목해, MASTE를 제안한다. MASTE는 training-free로 4단계 다중 에이전트 파이프라인(Aspect, Opinion, Sentiment, Consistency)으로 분해하고, 앞 단계 구조 출력을 조건으로 다음 단계가 수행하도록 설계했다.

- **Technical Challenges**: 핵심은 단일 디코딩 실패를 막으면서도 exact-match 평가에 필요한 span 정합성을 유지하는 것이다. 이를 위해 Opinion 에이전트는 aspect-conditioned opinion localization, minimal boundary calibration, one-to-many 페어링, verbatim span filtering을 적용하고, Consistency Check 에이전트는 span grounding·polarity calibration·duplicate consolidation·output canonicalization으로 트리플 집합 수준에서 검증/정규화를 수행한다.

- **Empirical Impact**: ASTE-Data-V2의 Lap14, Res14, Res15, Res16에서 MASTE는 동일 GPT-4o 백본 기준으로 기존 zero-shot 및 chain-of-thought LLM 베이스라인을 크게 앞서며, 감독학습 방식과의 격차를 줄이는 성과를 보였다. 특히 모든 데이터에서 precision이 가장 높게 나타났고, ablation과 cross-backbone 분석은 단계별 설계가 성능과 일반성에 모두 기여함을 확인했다.



### COBART: Controlled, Optimized, Bidirectional and Auto-Regressive Transformer for Ad Headline Generation (https://arxiv.org/abs/2607.08071)
Comments:
          10 pages, 5 figures, 5 tables. Published in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22). This is the author's accepted version; the definitive Version of Record is available at this https URL

- **Prior Approaches**: 기존 온라인 광고 헤드라인 생성은 주로 Transformer를 fine-tuning해 문장 품질(Rouge-L 등)을 높이거나, CTR을 RL/제약 조건으로 간접 최적화하는 방식이 많았다. 하지만 광고 포맷이 계속 바뀌고 길이·스타일·매력도 같은 요구조건이 수시로 달라지면, 사전학습 모델을 “그때그때” 맞춤 제어하기가 어렵다는 한계가 있었다. 또한 다수 연구는 조건을 자동으로 맞추되, inference 시점에 원하는 특성을 사용자가 직접 통제하는 능력은 제한적이었다.

- **Core Contribution**: 이 논문은 BART에 prefix control tokens를 결합해 CTR과 헤드라인 길이 같은 특성을 추론(inference) 단계에서 사용자가 직접 지정·동시에 최적화할 수 있게 한 COBART를 제안한다. control token을 입력 접두사로 넣어 인코더-디코더 attention이 해당 특성에 조건부로 동작하도록 유도한다. 그 결과, 다양한 광고 포맷/화면 크기에 맞춰 길이를 제어하면서도 더 높은 CTR을 노릴 수 있다.

- **Technical Challenges**: 핵심 기술 과제는 (1) pre-trained된 BART가 원래 control token 없이 학습됐는데도, control token 입력만으로 CTR·길이 조건을 안정적으로 따르게 만드는 것과 (2) CTR 같은 ‘관측 기반’ 연속 값/버킷화가 성능에 미치는 영향이다. 저자들은 CTR을 분위수 기반 버킷(15개)으로 분해해 태그 형태로 학습시키고, control token만으로도 인코더 표현이 조건부로 업데이트된다는 점을 통해 구조 변경 없이 제어 가능함을 보였다. 또한 Self-critical Sequence Training(SCST)와의 결합, 그리고 VBART 변형을 통해 reward/연속값 처리 방식이 CTR 추정과 품질 균형에 미치는 효과를 실험으로 점검했다.

- **Empirical Impact**: 실험 결과, 제안 방법은 기존 강한 baseline 대비 Rouge-L이 25.82% 증가하고 추정 CTR이 5.82% 증가하는 성과를 보고했다. 특히 SCST를 함께 쓰면 CTR 개선이 더 커지며, CTR을 버킷화하기보다 연속값을 직접 쓰는 설정에서 추정 CTR 향상이 더 두드러졌다. 전반적으로 “사용자 제어형 + 성능 최적화형” 헤드라인 생성으로 실사용 요구(포맷별 길이 제어, CTR 개선, 확장성)를 동시에 만족시키는 접근이라는 점에서 의미가 있다.



### Holographic Neural PCFG for Unsupervised Parsing (https://arxiv.org/abs/2607.08063)
Comments:
          Preprint under review

- **Prior Approaches**: 무감독 구문 분석은 라벨 없이 텍스트에서 잠재 트리를 유도하며, 크게는 괄호만 뽑는 접근과 명시적 PCFG 문법(비단말/규칙 확률)을 학습하는 접근으로 나뉜다. Neural PCFG 계열은 명시적 문법을 유지하지만, MLP로 규칙 점수를 매기기 때문에 룰 확률이 블랙박스 형태로 남고, 비단말 수를 늘리면 규칙 스코어러 파라미터가 커지거나 학습이 불안정해질 수 있다. 또한 규칙 점수 함수의 구조적 유도편향이 약하면 서로 다른 비단말이 거의 동일한 규칙 분포로 붕괴하는 문제가 보고되어 왔다.

- **Core Contribution**: 이 논문은 Holographic Neural PCFG (Hol-PCFG)을 제안해, PCFG 규칙 스코어링을 MLP 근사 대신 기호 임베딩 간의 대수적(알제브라) 관계 모델링으로 재구성한다. HolE의 circular correlation을 좌자식/우자식/어휘 방출(lexical emission) 관계에 맞게 적용하고, 토러스 제약 임베딩 위에서 각 규칙 확률을 닫힌 형태(closed form)로 산출하게 만들어 규칙 확률에 해석 가능한 수학적 구조를 부여한다.

- **Technical Challenges**: 핵심 기술적 난제는 규칙의 방향성(부모→자식), 좌우 비대칭, 그리고 비단말 수 확대 시에도 파라미터 효율과 안정적인 학습을 동시에 만족시키는 스코어 함수를 설계하는 것이다. Hol-PCFG는 원래 knowledge-graph에서의 HolE 스코어링을 PCFG의 parent/left/right 역할에 대응시키되, 모든 임베딩을 토러스(주파수별 단위 진폭) 위에 유지하고 원소 업데이트 후에도 투영(projection)으로 제약을 계속 지키도록 설계했으며, SN-PCFG의 left/right 조건부 독립 분해로 내부합(inside) 계산도 효율적으로 재사용한다.

- **Empirical Impact**: Hol-PCFG는 6개 언어에서 최신 수준의 구문 분석 성능(SF1)을 보이면서, 기준 모델 대비 규칙 스코어링 파라미터를 99.94% 줄이고 학습 안정성도 높였다고 보고한다. 특히 일본어에 대해 형태소 분할 없이 문자(character)만으로도 파싱이 가능하며 형태소 단위 성능을 거의 유지해, 명시적 문법 유도가 저자원/비분할 설정에서 실용적임을 시사한다.



### What LLM Forecasters Know but Don't Say: Probing Internal Representations for Calibration and Faithfulness (https://arxiv.org/abs/2607.08046)
- **Prior Approaches**: 예측을 위해 fine-tuning된 LLM은 정확도는 높아도 예측에 대한 확신(캘리브레이션)이 약할 수 있다는 문제가 지적돼 왔다. 또한 chain-of-thought(CoT) 설명은 그럴듯하지만, 실제 예측을 지지하는 증거와의 정합성이 충분히 보장되지 않는 경우가 많다. 기존 평가는 주로 출력과 CoT 자체에 의존해 내부에서 무엇이 확신을 결정하는지 검증하기 어려웠다.

- **Core Contribution**: 본 논문은 내부 표상(중간 activation)을 읽어내는 representation-pooling probe가 예측 캘리브레이션을 실질적으로 개선할 수 있음을 보인다. 또한 증거 ablation과 distraction( diversionary injection )으로 CoT의 증거 충실성/거짓말 탐지 가능성을 비교하며, 내부 표상이 행동 변화를 더 잘 추적한다고 주장한다. 마지막으로 forced answering과 라우팅을 통해 예측은 reasoning이 시작되기 전 이미 상당 부분 고정된다는 통찰을 제시한다.

- **Technical Challenges**: 핵심 과제는 (1) 중간 표상에서 ‘예측 근거’와 ‘불확실성’을 신뢰성 있게 뽑아 캘리브레이션을 개선하는 것, (2) CoT가 아닌 실제 영향이 프롬프트에서 제거/교란될 때도 내부가 이를 포착하는지 검증하는 것이었다. 저자들은 Eternis-Forecaster 8B에서 intermediate activations에 대해 representation-pooling probes를 학습해 GLM 계열(GLM-4.7-Flash, GLM-4.5-Air)에서도 동일 경향을 확인한다. 더 나아가 증거 ablation 후에도 reasoning trace가 유지되는 상황에서 probe가 행동 변화를 추적하도록 설계해, CoT가 영향 숨김을 수행할 때도 내부 신호가 유효함을 보여준다.

- **Empirical Impact**: OpenForesight와 다른 GLM 계열 설정에서 probe 기반 표상은 CoT나 기존 방식 대비 캘리브레이션이 “substantially better”하다고 보고한다. CoT의 증거 충실성은 evidence ablation/diversionary injection에서 흔들림이 reasoning trace에는 잘 드러나지 않았지만, probe는 behavioral shift를 더 잘 추적했고 84%에서 변화 방향까지 예측했다. 또한 reasoning 전 단일 패스의 pre-set answer 분포로 질문을 라우팅하면 토큰을 30-47% 절감하면서 정확도 손실 없이 유지돼, 내부 표상을 캘리브레이션·감사(auditing)·트리아지(triaging) 도구로 활용할 수 있음을 실증한다.



### PLURAL: A Global Dataset for Value Alignmen (https://arxiv.org/abs/2607.08034)
- **Prior Approaches**: 기존 LLM 정렬 연구는 주로 서구권 가치 데이터나 기준에 의존해, 다른 문화권의 가치 체계를 충분히 반영하지 못한다는 한계가 지적돼 왔다. 또한 가치가 담긴 선호 데이터를 만들 때는 실제 설문 신호를 제대로 보존하면서도 현실적인 시나리오로 바꾸는 과정이 어려워 규모 확장이 쉽지 않았다. 그 결과 문화별 ‘가치 신호’를 학습 가능한 형태로 대규모 제공하기가 제한적이었다.

- **Core Contribution**: 이 논문은 Integrated Values Survey(IVS) 기반의 대규모 가치 지향 preference 데이터셋 PLURAL을 제안한다. PLURAL은 IVS의 92개국(전국대표 설문)을 근거로 설문 응답을 합성 preference triplet으로 변환해, 국가 간 규범적 가치 신호와 국가 내부 다양성을 함께 담도록 설계했다. 초기 공개 버전은 약 50만 개 triplet로 20개 다양한 국가의 사람들을 커버한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 설문에서 추출한 규범 가치 신호를 합성 데이터에서 왜곡 없이 유지하고 (2) 동시에 현실감 있는 상황을 생성해 학습에 유용한 형태로 만드는 것이다. 연구진은 두 단계 생성 파이프라인으로 설문 응답을 합성 triplet으로 바꾸되, 가치 신호 보존과 시나리오 현실성을 동시에 노렸다. 또한 국가별 문화 프로필에 맞춰 정렬이 실제로 개선되는지 평가 설계를 병행했다.

- **Empirical Impact**: PLURAL은 데이터셋 수준 검증에서 원 설문이 가진 국가 간 가치 차이와 국가 내부 다양성을 모두 보존함을 보였다. 자동 평가에서는 PLURAL로 학습한 모델이 강한 베이스라인 대비 목표 국가의 문화 프로필에 더 잘 정렬되며 mean absolute error를 최대 27.7%까지 줄였다. 블라인드 휴먼 평가에서도 인도·브라질·일본의 176명 평가자가 PLURAL-정렬 응답이 자국 가치에 더 대표적이라고 판정해, value steering에 학습 가능한 신호가 존재함을 시사한다.



### Structured Pruning of Large Language Models via Power Transformation and Sign-Preserving Score Aggregation with Adaptive Feature Retention (https://arxiv.org/abs/2607.08027)
- **Prior Approaches**: LLM 압축을 위한 가지치기는 unstructured pruning과 structured pruning으로 나뉘는데, 전자는 정확도는 높지만 비정형 희소성 때문에 실질 속도 향상이 제한적이다. 후자는 하드웨어 친화적인 뉴런/채널 단위 제거로 가속이 가능하지만, 가중치 단위 점수를 구조 단위로 옮기는 과정에서 성능 저하가 자주 발생한다. Adaptive Feature Retention(AFR)은 ReFer와 SNIP 점수를 표준화·합산해 사전학습 지식 보존과 다운스트림 적응을 균형 있게 하려 했지만, 이를 structured pruning에 그대로 적용하기엔 여러 형태의 점수 집계 문제가 남아 있다.

- **Core Contribution**: 이 논문은 unstructured AFR을 structured pruning에 맞게 재구성하는 데서 생기는 핵심 실패 원인 3가지를 짚고, 이를 동시에 완화하는 통합 방법을 제안한다. 구체적으로 SNIP 점수의 분포 불일치, 신호 부호(최적화 방향) 손실, outlier의 영향 문제를 각각 전용 처리로 해결해 뉴런 단위 점수 산정의 신뢰도를 높인다. 그 결과, structured pruning에서도 unstructured AFR과 비슷한 정확도를 유지하면서 실제 추론 가속까지 달성하는 것을 목표로 한다.

- **Technical Challenges**: 첫째, ReFer와 SNIP의 분포가 달라 단순 표준화·평균이 잡음을 증폭시킨다. 이를 위해 SNIP에 power transformation을 적용해 저중요도 영역 잡음을 비선형으로 억제하면서 고중요도 신호는 보존한 뒤 표준화로 재정렬한다. 둘째, structured 집계에서 절댓값을 먼저 취하면 뉴런 내부 가중치의 최적화 방향 일관성(sign consistency)이 사라져 뉴런 중요도 판단이 흔들린다; 논문은 부호를 유지한 뒤 평균 후 절댓값을 취하는 sign-preserving aggregation을 사용한다. 셋째, ReFer 점수에 존재하는 극단 outlier가 평균 기반 순위를 뒤집을 수 있어 뉴런별 2nd~98th percentile 범위를 벗어난 값은 제거하는 percentile-based outlier removal로 집계를 안정화한다.

- **Empirical Impact**: 실험은 Llama-3-8B, Vicuna-v1.5-13B, LLaVA-v1.5-13B에서 수행되었고, 제안 방법은 naive structured AFR 대비 최대 21.27점(20%)과 15.13점(50%) 개선을 보였다. 또한 Llama-3-8B 기준 50% 가지치기에서 35.1% 파라미터/VRAM을 줄이면서 1.57× 추론 속도 향상을 달성했고, LLaVA-v1.5-13B에서도 1.56× 수준의 속도 개선을 확인했다. structured pruning이 특별한 희소 연산 라이브러리 없이도 표준 하드웨어에서 동작 가능한 형태로 가속을 제공한다는 점에서, deployment 관점의 실용성 있는 진전을 의미한다.



### Can We Trust LLM's Logic? Quantifying Uncertainty, Coherence, and Robustness via a Graph-Based Framework (https://arxiv.org/abs/2607.08017)
Comments:
          42 pages, 14 figures, 12 tables

- **Prior Approaches**: LLM의 불확실성을 보기 위해 토큰 확률 기반 UQ를 쓰거나, CoT를 여러 개 샘플링해 최종정답의 다수결(Self-Consistency, SC)로 답을 고르는 방식이 주로 쓰였다. 하지만 SC는 중간 추론의 논리적 타당성을 직접 검증하지 않고 최종 정답 합치만 본다.

- **Core Contribution**: 논문은 GraphEVAL로, “정답 일치”가 아니라 추론 경로의 일관성과 충실도(reasoning fidelity)를 중심에 둔 그래프 기반 평가 프레임워크를 제안한다. 또한 SS-GED를 바탕으로 Graph Reasoning Coherence Score(GRCS)라는 새로운 UQ 지표를 도입해 의미-구조 합의 부족과 병적 모드 붕괴 및 자신감 있는 환각을 함께 포착한다.

- **Technical Challenges**: 핵심 난제는 다단계 추론을 그래프로 구조화하면서 의미와 인과 의존까지 반영해 거리를 계산하는 것이다. 이를 위해 CoT를 인과 관계 DAG로 분해한 뒤 Semantic-Structural GED(SS-GED)로 노드/엣지의 의미·구조 유사도를 최적 매칭으로 계산하고, GRCS는 이 거리들의 분포 평균으로 전체 일관성을 정량화한다.

- **Empirical Impact**: GRCS는 모델 규모가 큰 경우와 작은 경우 모두에서 reasoning faithfulness와 일관되게 음의 상관을 보여, 기존 UQ 지표보다 추론의 숨은 불확실성을 더 잘 잡아낸다. 또한 Graph Self-Consistency(GSC)는 최종 다수결 대신 그래프 공간에서 중심(medoid) 추론 경로를 선택해 SC가 ‘운 좋은 정답(lucky guess)’을 과대평가하는 문제를 줄이며, medoid를 적대적으로 제거하면 추론 충실도와 일부 경우 정확도까지 떨어져 ‘load-bearing path’의 역할을 실험적으로 확인한다.



### Tool-Making and Self-Evolving LLM Agents in Low-Latency Systems (https://arxiv.org/abs/2607.08010)
Comments:
          Preprint

- **Prior Approaches**: 기존 CodeAct-style LLM 에이전트는 매 요청마다 inference-time에 코드를 새로 생성·실행해 절차를 처리한다. 같은 SOP 단계가 반복돼도 에이전트가 스키마와 값들을 다시 해석하고 유사 코드를 재생성하면서 지연, 비용, run-to-run 변동성이 커진다. 또한 SOP 텍스트가 실제 프로덕션 쿼리로 변환되기엔 필드명/타입/널 처리/경계 조건 같은 실행 디테일이 빠져 정확도 병목이 발생한다.

- **Core Contribution**: 이 논문은 반복되는 SOP 절차를 실행 흔적과 검증 케이스 기반으로 ‘도구(tool)’로 컴파일해 배포 전에 고정된 버전 라이브러리를 만든다. 에이전트 런타임에서는 이 도구를 직접 호출해 SOP 노드별로 즉시 verdict(참/거짓/데이터 없음)을 얻고, 필요한 경우에만 코드 생성으로 폴백한다. Fulfillment Center 알람 트리아지에 적용해 추론 경로에서의 재코딩 루프를 제거하는 운영 패턴을 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 SOP만으로는 누락된 실행 디테일 때문에 도구 합성이 쉽게 실패한다는 점이다. 이를 위해 (1) 라이브 환경에서 스키마·값 범위·실행 응답을 수집하는 data-collection trace로 접지(grounding)하고, (2) 라벨된 케이스로 테스트한 뒤 test–repair loop와 reflection을 통해 불일치를 수정해, SOP 텍스트 의존도를 낮췄다. 또한 도구 출력이 입력에 대해 결정적이며 버전이 남도록 설계해, 동일 입력이면 동일 결과가 나오게 하면서 예외·데이터 없음 시에는 안전하게 baseline 하위 에이전트로 되돌린다.

- **Empirical Impact**: 프로덕션에서는 도구 호출이 p50 지연을 42% 줄이고, 1,500개 과거 알람에서 end-to-end 오류율을 최대 53%까지 낮췄다(반복 단계의 실행 변동성 억제). 구조적 단순화를 위해 서브에이전트를 제거한 통제 ablation에서도 direct-call이 p50을 추가로 62% 감소시켰고, 도구가 SOP 미세 편차나 잘못된 리포팅을 줄여 정확도도 개선됨을 보였다. 마지막으로 버전화된 도구 로그와 모니터링이 상류 데이터 드리프트·스키마 타입/포맷 불일치 같은 사전 문제를 조기 발견해 운영 감사 가능성과 운영 편의성을 함께 높였다.



### From Execution to Education: A Bloom-Aligned Framework for Measuring Educational Control in LLMs (https://arxiv.org/abs/2607.08009)
Comments:
          24 pages, 20 figures

- **Prior Approaches**: 기존 LLM 프로그래밍 평가는 HumanEval, SWE-bench 같은 실행 중심 벤치마크가 주로 다뤄졌고, 정답 여부만으로는 과제의 인지적 부담(learning objective)을 조절하는 교육 능력을 보기 어렵다. 교육적 관점에서는 답을 바로 제시하는 경우 ‘illusion of learning’이 생길 수 있어, 표면적 수정이 아닌 인지 요구 수준의 이동을 측정해야 한다.

- **Core Contribution**: 이 논문은 Bloom’s Taxonomy(개정 Bloom) 기준으로 ‘교육적 control’을 정량화하는 Bloom-aligned 프레임워크를 제안한다. 과제의 instructional intent는 유지하되 인지적 demand를 특정 Bloom 레벨로 이동시키는지, 그리고 난이도/블룸 지시가 실제로 어떤 변화로 이어지는지를 OCS(Observed Cognitive Shift)와 TZA(Target Zone Accuracy)로 평가한다.

- **Technical Challenges**: 핵심 난제는 지시가 생성 결과의 인지 레벨을 실제로 바꾸었는지(표면 형식만 바꿨는지)와 그 변화가 모델 내부 표상과 어떻게 연결되는지를 동시에 확인하는 것이다. 이를 위해 general difficulty control과 Bloom-targeted interventions를 동일 조건의 템플릿으로 생성하고, Claude-3.5-haiku 기반 judge로 Bloom 레벨을 분류하며, semantic-delta(임베딩 공간에서 공통 맥락을 제거한 언어 변화) 클러스터링과 Fisher’s Discriminant Ratio(레이어별 선형 분리도)로 표현 진단을 수행한다.

- **Empirical Impact**: 2,520개 과제(세 벤치마크)에서 Qwen3-Next(일반)와 Qwen3-Coder-Next(코더) 쌍을 비교한 결과, 두 모델 모두 인지적 부담을 ‘올리는’ 방향은 비교적 잘 구현하지만 ‘낮추는’ 데는 일관되게 실패하는 강한 비대칭이 관찰됐다. 또한 표현 진단에서 일반 모델은 중간 레이어에서 general difficulty 및 Bloom-control 대비가 더 잘 분리되는 반면, 코더 모델은 일반 난이도 지시에서 분리도가 약하고 Bloom-control에서는 더 깊은 레이어에서 피크를 보였다. 결론적으로 실행 성능이 높아도 Bloom-aligned 교육적 control은 자동으로 보장되지 않으며, 튜터/교육 도구로 쓰려면 학습 목표에 맞춘 인지 요구 수준 조절을 별도로 검증해야 한다.



### Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator (https://arxiv.org/abs/2607.07993)
Comments:
          Accepted to COLM 2026. Camera-ready version to appear

- **Prior Approaches**: 기존 신뢰성 확보를 위해 faithfulness hallucinations를 탐지하는 연구는 주로 고성능 LLM을 판별자로 쓰거나, 경량 탐지기를 학습시키되 사람 라벨이 부족해 합성 데이터를 활용해왔다. 그러나 합성 방식은 generator를 고정해 두는 경우가 많아, 탐지기가 개선될수록 합성된 환각이 점점 쉬워져 학습 신호가 포화되는 한계가 있었다.

- **Core Contribution**: 이 논문은 Hallucination Self-Play(HSP)로, generator와 detector를 동일한 base model에서 시작해 서로의 피드백을 받으며 동적으로 공진화시키는 폐루프 학습을 제안한다. detector는 생성된 응답의 faithfulness를 판단하고, 그 결과를 reward로 generator를 RLAIF로 학습시켜 detector가 더 어렵게 맞서야 하는 환각 데이터를 만든다.

- **Technical Challenges**: 주요 기술 과제는 환각 생성이 잘 검증되지 않으면 reward hacking으로 인해 “진짜로는 틀리지 않은 답”이나 의미 없는 응답이 보상을 악용하는 문제다. 이를 막기 위해 (1) 컨텍스트와의 모순 여부, (2) 근거 문서에 없는 사실 도입 여부, (3) refusal·trivial 응답 억제 같은 reward gating과 패널티를 결합했으며, detector는 RLVR로 예측 정확도(이진 정오 라벨)를 강화하도록 설계했다.

- **Empirical Impact**: 실험은 RAGTruth 벤치마크에서 수행되며, CoT가 없는 설정에서는 SFT 대비 F1이 전반적으로 개선되고 recall이 특히 상승하는 양상을 보였다. CoT가 있는 더 어려운 설정에서는 외부 rationale 없이도 7B급 모델이 self-play만으로 성능을 누적 개선해 GPT-4o w/ CoT급과 유사한 수준(74.3 vs 74.5)을 달성했으며, reward gating을 제거하면 합성 데이터 품질이 붕괴되는 부작용도 확인했다.



### A Reliability Assessment of LALM Audio Judges for Full-Duplex Voice Agents (https://arxiv.org/abs/2607.07985)
Comments:
          28 pages total (12 main body, 1 reference, 15 appendix). In main body: 2 diagrams, 3 table, 2 charts

- **Prior Approaches**: 기존 LALM-as-judge 연구는 주로 TTS 텍스트 음성(또는 단일 발화)에서 사람 MOS와의 강한 상관을 보고해 왔습니다. 하지만 기업 음성 에이전트의 full-duplex 대화처럼 채널·턴테이킹·중간 끊김·발화 부자연스러움이 섞인 원시 stereo waveform에 대해, 다수 인간 채점자를 기준으로 LALM의 ‘대체 가능성’을 검증한 연구는 드뭅니다. 또한 음성 평가에서는 크립펜도르프 α처럼 분포/천장 효과로 인해 신뢰도 지표가 해석을 왜곡할 수 있어, 단일 지표로 결론 내리기 어렵다는 점이 문제로 지적됩니다.

- **Core Contribution**: 이 논문은 raw stereo waveform(왼쪽: 에이전트, 오른쪽: 고객)을 입력으로 해 8개 차원 점수를 내는 Gemini 기반 오디오 judge가, 3명의 보정된 인간 채점자 기준과 어느 정도 일치하는지를 per-dimension으로 실증합니다. 특히 Gemini 2.5 Flash를 기준(ground-truth)으로 삼아 209개 full-duplex 세션(자연 152, 적대적 결함 57)에서 LALM을 사실상 4번째 rater로 취급해 신뢰도를 측정합니다. 그 결과는 ‘전체 성능’이 아니라 차원별로 배치 가능한지 여부를 판단하는 운영 프레임으로 제시됩니다.

- **Technical Challenges**: 핵심 난제는 (1) 대화 음성처럼 채널 동시성·발화 불연속성이 점수 분포를 바꾸고, (2) Likert 척도에서 천장 효과로 인해 chance-corrected 신뢰도 지표가 직관과 다르게 나올 수 있다는 점입니다. 논문은 rank agreement(Spearman rho, bootstrap 95% CI)와 절대 일치(simple within 1 point) 같은 서로 다른 렌즈를 병행하고, 결함(예: 클리핑, dead air, 잡음, 중간 잘림, phoneme-region overdubbing, 샘플레이트 다운/업)에서는 결함 민감도(recall)까지 분석해 해석의 공백을 줄였습니다. 또한 모델을 교체했을 때 순위는 유지돼도 calibration이 틀어질 수 있음을 확인하고, 이를 검증을 통해 보정 가능한 운영 과제로 정리합니다.

- **Empirical Impact**: Gemini 2.5 Flash는 8개 차원 중 5개에서 인간 간 일치 수준과 거의 비슷한 rank agreement(인간-인간 대비 rho 갭 ≤0.07)를 보였고, 7개 차원에서는 bootstrap CI 중첩으로 통계적 분리 가능성이 낮았습니다. 다만 speaking_rate_adaptation, overall_fidelity는 within-1 합의가 50% 미만이고, audio_clarity는 적대적 결함 상황에서 상대적 순위 분별이 약해(자연 대화에서는 상관이 거의 0에 수렴) ‘부분 보완형 배치’가 필요함을 드러냈습니다. Gemini 3.5 Flash는 전반적으로 개선된 반면, Gemini 3.1 Pro는 rank 상관은 비슷해도 일부 차원 점수가 평균에서 유의하게 벗어나 calibration 재검증이 필수임을 강조하며, 인간 채점의 비용(2차원 스팟체크 포함)이 LALM 워크로드 대비 대략 100배 수준임을 제시해 프로덕션 적용 근거를 제공합니다.



### When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning (https://arxiv.org/abs/2607.07976)
- **Prior Approaches**: GRPO 같은 critic-free RLVR은 완료(trajectory) 단위로 계산한 advantage를 생성된 모든 토큰에 동일하게 배분한다. 이 방식은 토큰마다 문맥적 신뢰도가 다른데도 같은 양의 보상을 주어, 최종 정답이 맞더라도 중간에 잘못된 연속(token)이 양성 업데이트를 받는 문제를 만들 수 있다. 기존 대안들은 외부 신호(counterfactual, TD, 실행 피드백)나 토큰 entropy 같은 내재 신호로 토큰 중요도를 조정하지만, 의미적으로 “그 토큰이 해당 문맥에서 타당했는가”를 충분히 로컬 관점에서 분리하지 못한다.

- **Core Contribution**: 논문은 GRPO류에서 발생하는 실패 모드인 Positive-Credit Contamination을 규명한다. 특히 로컬 맥락에서 그럴듯하지 않은 implausible tail tokens가, 우연히 보상받은 완료 안에 포함되면 정상 토큰과 같은 양의 credit을 받아 잘못된 추론 습관이 누적 강화될 수 있다고 본다. 이를 해결하기 위해 TACO(Tail-Aware Credit calibratiOn)는 토큰별로 “implausible tail일 위험”을 추정해 양성 credit을 문맥적으로 보정한다.

- **Technical Challenges**: 핵심 난제는 토큰 단위로 ‘정답에 기여한 정도’를 직접 라벨링/오라클로 판정하는 것이 불가능하다는 점이다. TACO는 오라클 없이 forward 패스에서 관측 가능한 값만으로 tail-risk를 추정하며, sampled-token probability와 local entropy를 함께 써서 ‘낮은 확률이 곧 불확실성 탐색인지, 아니면 문맥적으로 타당하지 않은 꼬리인지’를 구분한다. 이후 위험도가 큰 토큰에는 양성 advantage에 대해 부드럽게 down-weight를 적용하되, 그라디언트 자체를 완전히 제거하진 않아 유용한 희귀 패턴의 누적은 보존한다.

- **Empirical Impact**: 세 모델(Qwen3 1.7B/4B, Qwen2.5-Math 7B)과 다수 벤치마크(수학 6개, 과학 OOD 2개)에서 TACO는 GRPO 스타일 베이스라인 대비 일관된 성능 향상을 보인다. 특히 long-horizon 학습(기본 300→600 step)에서도 GRPO는 정체/열화가 나타난 반면, TACO는 엔트로피가 더 매끄럽게 유지되며 성능이 계속 개선돼 학습 안정성이 개선됨을 보여준다. 정성 사례에서도 중간의 불필요한 꼬리 토큰들은 억제하면서도 후반의 올바른 풀이 구간은 credit을 유지해, 잘못된 로컬 행동 강화만 줄이는 효과가 확인된다.



### A Multi-cluster Boundary Learning Method for Out-of-Scope Intent Detection via MiniLM Embedding (https://arxiv.org/abs/2607.07974)
Comments:
          To submit

- **Prior Approaches**: 기존 OOS(out-of-scope) 의도 감지는 보통 알려진 의도를 다중 분류로 보고, OOS를 별도 클래스/점수/경계로 얹는 방식이 많았다. 이때 알려진 클래스 수가 늘면 OOS 샘플이 결정영역에 흡수되며 OOS 거절 정확도가 급격히 떨어지고, LLM-embedding 기반 방법은 큰 파라미터와 prompt 민감도로 실시간 배포가 어렵다는 한계가 있었다.

- **Core Contribution**: 논문은 MiniLM 임베딩을 쓰되, OOS 감지를 one-class classification 워크플로로 분해하는 멀티-클러스터 경계 학습을 제안한다. 또한 gate→router→expert의 3단 캐스케이드로 OOS 거절과 알려진 의도 다중 분류를 분리해, gate는 OOS 여부만 판단하고 후단은 in-scope 의도만 세분화한다.

- **Technical Challenges**: 핵심은 MiniLM 임베딩에서 한 의도가 단일 중심이 아니라 여러 클러스터를 형성한다는 전제를 경계학습으로 안정적으로 반영하는 것이다. 이를 위해 각 의도별 K-means 멀티 센트로이드를 만들고 클러스터마다 로컬 반경(대각 Mahalanobis 기반)을 추정해, 임베딩이 어떤 클러스터 로컬 영역 안에 들어오면 수락(known), 모두 바깥이면 OOS로 거절한다.

- **Empirical Impact**: CLINC150, StackOverflow, Banking77에서 제안 방법은 기준선 대비 OOS F1을 0.85%~17.12% 개선하며 SOTA 성능을 보였다. KIR(known intent ratio) 변화에도 OOS F1이 비교적 안정적이었고, ablation에서 gate에 쓰인 all-MiniLM-L6-v2 선택과 캐스케이드 구조가 성능에 필수적임을 확인했다.



### When Debiasing Backfires: Counterintuitive Side Effects of Preprocessing-Based Stereotype Mitigation (https://arxiv.org/abs/2607.07937)
Comments:
          Published in ACL 2026 Findings

- **Prior Approaches**: NLP에서는 편향 완화를 위해 데이터 전처리/후처리로 학습 코퍼스를 수정하는 방식이 널리 쓰인다. 주로 특정 집단의 스테레오타입 점수를 줄이지만, ‘다른 집단으로 편향이 옮겨가는지(재분배)’는 충분히 검증되지 않았다.

- **Core Contribution**: 이 논문은 Wikipedia에서 6개 인구집단(성별·인종·종교)을 겨냥해 편향 완화 전/후(pre-/post-training) 두 종류 PLM(TinyBERT, GPT-2)을 학습하고, 모든 집단에 대한 스테레오타입 변화를 함께 측정한다. 그 결과, 목표 집단은 개선되더라도 비목표 집단에서 스테레오타이핑 또는 반(反)스테레오타이핑이 기준선 대비 증가하는 ‘side effects(부작용)’이 반복적으로 나타남을 보여준다.

- **Technical Challenges**: 전처리 기반 방법이 스테레오타입을 ‘제거’해도 왜 다른 집단의 편향이 증가하는지 메커니즘 설명이 어려웠다. 연구진은 제거/언급 삭제/참조 스왑(DG, RG, SR) 및 데이터 규모·학습 단계 변형을 폭넓게 실험하고, attention-rollout 분석으로 모델의 정보 라우팅 변화가 크지 않음을 확인해 단순한 의미/주의 변화만으로는 부작용을 설명하기 어렵다는 점을 제시한다.

- **Empirical Impact**: StereoSet과 CrowS-Pairs 같은 표준 벤치마크는 종종 이런 재분배 부작용을 놓치며, 부작용이 주의(attention) 패턴만으로도 잘 드러나지 않음을 실증적으로 보인다. 연구진은 부작용-인지형 평가 진단과 더 투명하고 통제 가능한 완화 절차의 필요성을 강조하며, 공개 코드로 재현과 점검을 지원한다.



### Scalable and Culturally Specific Stereotype Dataset Construction via Human-LLM Collaboration (https://arxiv.org/abs/2607.07895)
Comments:
          Weicheng Ma, John Guerrerio: equal contribution; published in EMNLP 2025 Main

- **Prior Approaches**: 기존 LLM 편향 연구는 StereoSet, CrowS-Pairs 같은 수작업(또는 번역) 기반 데이터셋에 크게 의존해 왔지만, 대부분 영어권(주로 미국) 중심이라 비영어·저자원 문화의 지역별 고유 편향을 놓치기 쉽다. 또한 번역 기반 평가는 원자료에 내재된 미국적 맥락을 그대로 가져와 표적 문화의 ‘문화 특이적’ 고정관념을 제대로 반영하지 못한다.

- **Core Contribution**: 이 논문은 비용 효율적인 human-LLM collaborative annotation 프레임워크를 제안해, LLM이 후보 고정관념을 생성한 뒤 해당 문화권의 in-culture annotator가 검증·구체화하는 방식으로 EspanStereo(스페인어 고정관념 데이터셋)를 구축한다. EspanStereo는 스페인·멕시코·아르헨티나·콜롬비아·니카라과 5개국을 커버하며, 영어 중심 자원에는 없는 문화권별 편향까지 포함하도록 설계됐다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 공격적·유해한 표현을 생성해야 하는 고정관념 수집 과정에서 LLM의 안전장치(금지/검열)를 우회할 전략이 필요하다는 점과 (2) LLM이 만든 후보가 문화적으로 적합한지 검증해야 한다는 점이다. 논문은 injection attack으로 고정관념 후보를 더 안정적으로 회수하고, 생성·검증을 국가별로 분리한 뒤 Likert 척도 기반 다수결로 희귀 항목을 제거해 품질을 관리한다.

- **Empirical Impact**: EspanStereo로 BETO(스페인어 모델)와 XLM-R(다국어 모델)를 평가한 결과, 국가 간 고정관념 수준과 인코딩 패턴이 유의미하게 달라 스페인어권 내에서도 지역성 편차가 크다는 점을 보여준다. 또 Shapley value 기반 probing과 attention-head pruning으로 고정관념 검출/완화 성능을 조절할 수 있음을 확인하며, 더 문화적 맥락을 반영한 멀티링구얼 편향 벤치마크의 필요성을 실증적으로 뒷받침한다.



### How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism (https://arxiv.org/abs/2607.07891)
Comments:
          Submitted to Philosophy and Technology

- **Prior Approaches**: 기존 계산언어학은 언어를 ‘사물의 표상’에 붙는 고정 라벨, 즉 전달 코드로 보는 경향이 강했습니다. 해리스의 Integrationist linguistics는 이를 ‘language myth’(telementational/코드 모델)로 비판하며, 의미가 전달되기보다 상황 속에서 공동 행동을 향해 전망적으로 구성된다고 봅니다. 다만 Integrationism은 (1) 기호가 왜 미래 열림을 유지하는지의 구조 메커니즘, (2) 언어-비언어 기호활동의 연속성의 정밀한 틀, (3) 과거 통합이 남기는 archive의 구조를 설명하는 이론이 부족하다는 공백이 있습니다.

- **Core Contribution**: 이 논문은 Elan Barenholtz의 autogenerative theory(자기-생성성)를 해리스의 Integrationism에 ‘침해 없이 보강’할 수 있는 해법으로 제시합니다. 핵심 주장은 autogenerative 계정이 (i) 해리스가 중시한 prospective openness의 구조적 메커니즘, (ii) 언어와 비언어 간 semiotic continuity를 뒷받침하는 계산적 상관, (iii) 과거 통합의 잔재로서 archive가 어떤 모습인지와 참가자들이 이를 어떻게 활용하는지를 제공한다는 점입니다. 결과적으로 해리스의 ‘상황적 통합 행위가 우선’이라는 존재론적 원칙은 유지하면서, Integrationism이 못 채운 설명력을 추가로 확보합니다.

- **Technical Challenges**: challenge는 ‘고정된 지시/의미’ 없이도 왜 기호가 닫히지 않고 다음 선택지를 열어두는가를 구조적으로 보여주는 것입니다. 논문은 LLM의 corpus 통계가 각 토큰을 고정 주소 내용이 아닌 확률적 관계망으로 정의하며, 그 결과 조건이 조건을 낳는(conditions beget conditions) 형태의 계속 생성 구조가 prospective openness를 산출한다고 설명합니다. 또한 해리스가 주장한 반(反)모달 경계를 autogenerative 속성이 텍스트뿐 아니라 이미지 생성·캡셔닝에서도 나타나는 구조적 평행으로 연결하고, archive은 ‘지난 languaging의 잔차’가 남긴 확률적 경향(affordances)으로 작동해 현재의 통합을 제약·가능하게 한다고 정리합니다.

- **Empirical Impact**: 경험적 함의는 LLM이 ‘세계 기술’이 아니라 ‘이전 토큰들의 예측/생성 구조’를 강하게 학습한다는 점이, Integrationism의 기호-전망 개념과 정합적인 구조상관을 제공한다는 데 있습니다. 특히 multimodal 모델에서의 공통 autogenerative 패턴은 언어-비언어 semiotic continuity가 최소한 corpus 수준의 통계 구조에서도 관측된다는 시사점을 줍니다. 다만 인간의 뇌와 통합 행위는 판단·주의·예측의 생활사와 같은 추가 능력으로 ‘적절하게 맞출 수 있음’을 수행하며, 단순히 archive 기반 통계 생성만으로는 이 차이를 완전히 대체할 수 없다는 한계도 함께 강조해, AI 연구자들이 LLM 능력의 범위와 빈틈을 더 원리적으로 해석하게 합니다.



### DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environmen (https://arxiv.org/abs/2607.07820)
- **Prior Approaches**: 기존 툴 사용 에이전트 학습은 supervised fine-tuning이 고정된 teacher-distilled 궤적에 의존하고, sparse-reward reinforcement learning은 긴 지평에서 약한 신호만 제공해 개선이 더디다는 한계가 있었다. 특히 웹 환경처럼 다양한 실패가 누적되는 long-horizon 상호작용에서는 기준이 되는 강한 감독 신호를 만들기 어렵다.

- **Core Contribution**: 이 논문은 self-distillation 기반 웹 에이전트 학습 프레임워크 DeepSearch-Evolve를 제안한다. deterministic이면서 검증 가능한 환경 DeepSearch-World를 구축해 reproducible한 검색과 page-reading 도구를 제공하고, progress verification·grounded reflection·failure recovery 같은 자기진화에 필요한 인지 행동을 학습 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 웹 에이전트의 학습 궤적을 안정적으로 생성·품질 평가해야 하고, (2) 자기증류 과정에서 저품질 데이터를 섞어 성능이 무너지지 않게 해야 한다는 점이다. 이를 위해 DeepSearch-Evolve는 trajectory generation→필터링→data mixing→fine-tuning을 반복하는 self-distillation 루프를 구성하고, DeepSearch-World의 재현 가능한 검증 가능 설계로 학습 신호를 강화했다.

- **Empirical Impact**: 별도의 더 유능한 모델로부터의 distillation 없이도 DeepSearch-World-9B가 견줄 만한 성과를 보였고, BrowseComp 31.2%, GAIA 61.5%, HotpotQA 93.4%를 기록했다. verifiable 환경이 long-horizon 웹 에이전트의 scalable self-evolution을 가능하게 함을 실험적으로 뒷받침하며, 환경·420K 학습풀·검증세트·모델·코드를 공개해 후속 연구 확장성도 높인다.



### From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier (https://arxiv.org/abs/2607.07779)
- **Prior Approaches**: 기존 AI4Math는 Interactive Theorem Proving(ITP) 언어에서 높은 정확도의 정리 증명 생성에 강점을 보여왔지만, 대개 미리 정해진 직관·휴리스틱에 의존해 확장성이 제한된다. 또한 LLM의 자연어 추론은 강력하더라도 기계 검증 가능한 의미가 없어 환각 가능성이 남고, open-ended 연구 수준의 검증/자율 탐색을 수행하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 AI4Math를 ‘문제 해결자(solver)’가 아니라 ‘연구 에이전트(research agent)’로 전환해야 frontier 수학(새 정리 발견, 미해결 추측 해결)을 다룰 수 있다고 주장한다. 이를 위해 분야 전반을 체계적으로 정리하는 리뷰(데이터셋, auto-formalization, proof synthesis)를 제공하고, 현재 시스템이 연구 에이전트 역할을 수행하지 못하는 근본 격차를 로드맵 형태로 구조화해 제시한다.

- **Technical Challenges**: 연구 에이전트로의 전환을 가로막는 기술 과제로 데이터·평가 한계, 수학적 관계 구조의 부족, 탐색 자체의 장벽, 도구 생태계의 단절, 그리고 human-AI 협업의 미흡함이 제시된다. 논문은 이 문제를 각 축별로 어떤 방식의 데이터/모델링/워크플로 설계가 필요하며, 어떤 실험·검증 체계를 갖춰야 하는지에 대한 전략적 방향을 제안한다.

- **Empirical Impact**: open Erdős 문제 등 실제 frontier에 가까운 과제를 분석한 결과, 기존 성과는 문헌에 이미 존재하던 결과를 재발견하는 비중이 높고, Millennium Prize 수준의 ‘진짜로 새로운 아이디어’가 필요한 문제를 풀기엔 역량이 부족하다는 관찰을 뒷받침한다. 결론적으로 이 글은 향후 AI4Math 연구가 경쟁형 증명 성능을 넘어, 기계 검증을 기반으로 한 장기적 탐색과 협업형 연구 프로세스를 강화해야 한다는 방향성을 분야에 제시한다.



### Unveiling Public Opinion: A Study of Sentiment Analysis Using LSTM and Traditional Models (https://arxiv.org/abs/2607.07772)
Comments:
          6 pages, 5 figures. Published in the Proceedings of the 2025 IEEE Conference on Computing, Communication, and Data Engineering (C-CODE 2025)

- **Prior Approaches**: 트위터 같은 소셜 미디어 데이터에서 감정(긍정/부정/중립)을 분류하는 감성분석은 NLP의 핵심 응용으로, 기존에는 로지스틱 회귀·나이브 베이즈·랜덤 포레스트·그라디언트 부스팅 같은 전통적 머신러닝과 LSTM 같은 딥러닝을 함께 비교해왔다. 전통적 모델은 텍스트를 특징 기반으로 다루는 경향이 있어, 문맥이나 표현의 연속성을 충분히 포착하지 못할 수 있다는 한계가 있다. 반면 딥러닝은 순차 정보를 학습할 수 있지만, 데이터 전처리와 학습 구성이 성능을 좌우한다는 문제가 남는다.

- **Core Contribution**: 이 논문은 Kaggle Twitter 데이터셋을 토큰화·표제어 추출(lemmatization)·불용어 제거로 전처리한 뒤, 여러 머신러닝/딥러닝 모델을 동일 조건에서 평가해 최적 감성분석 모델을 찾는 데 초점을 둔다. 특히 LSTM을 중심으로 성능이 가장 좋게 나오는지를 실험적으로 확인하며, 맥락·문장 내 순차적 특성을 더 잘 반영할 수 있음을 보인다. 결과적으로 “어떤 알고리즘이 트위터 감성분류에 더 적합한가”를 경험적으로 정리한다.

- **Technical Challenges**: 핵심 기술적 과제는 짧고 비정형인 트윗에서 감정 신호를 안정적으로 추출해 문맥/순서를 반영하는 것이다. 이를 위해 데이터 전처리로 노이즈를 줄이고, 분류 모델로는 로지스틱 회귀부터 LSTM까지 폭넓게 적용해 공정한 비교가 가능하도록 설계했다. 또한 평가에서는 정확도뿐 아니라 micro-average ROC-AUC(ROC- AUC)를 함께 사용해 클래스 간 편향에 덜 민감한 관점을 더했다.

- **Empirical Impact**: 실험 결과, LSTM 모델이 훈련 정확도 90.98%, 테스트 정확도 80.00%, micro-average ROC-AUC 0.92로 가장 좋은 성능을 보였다. 이는 전통적 머신러닝 기법 대비 트윗의 맥락과 연속적인 텍스트 특성을 더 잘 포착한다는 해석을 뒷받침한다. 트위터 기반 여론 파악과 추세 예측에서, 감성분석 모델 선택 기준을 LSTM 쪽으로 제시하는 실용적 근거가 된다는 점에서 의미가 있다.



### Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents (https://arxiv.org/abs/2607.08716)
- **Prior Approaches**: 기존 메모리·롱컨텍스트 접근은 주로 정보를 저장하거나(retrieval, long-context memory) 필요한 항목을 꺼내는 데 초점을 맞춘다. 그런데 장기 작업에서는 긴 트래젝터리 전체가 문맥에 남아 있어도, 요구사항·환경 사실·실패 진단 같은 의사결정 관련 상태가 다음 행동에 제대로 영향을 주지 못하는 ‘behavioral state decay’가 발생한다.

- **Core Contribution**: 이 논문은 메모리를 수동 저장/검색이 아니라 ‘개입(intervention) 정책’으로 재정의한다. 메모리 에이전트가 액션 에이전트와 병렬로 동작하며, 메모리 뱅크를 갱신한 뒤 다음 턴 호출에 메모리 근거의 짧은 리마인더를 주입할지(또는 침묵할지) 선택한다.

- **Technical Challenges**: 핵심 난제는 “무엇을 기억할지”를 넘어 “언제, 어떤 형태로 행동 루프에 다시 넣을지”를 학습/제어하는 것이다. 이를 위해 구조화된 memory bank(지식 메모리·절차 메모리·비공개 상태)와 2단계 파이프라인(뱅크 관리 도구콜 → 개입 선택)을 두고, 개입이 불필요하면 null intervention으로 명시적으로 침묵하게 한다.

- **Empirical Impact**: Terminal-Bench 2.0과 τ2²-Bench에서 메모리 개입은 action agent 강도에 무관하게 pass@1을 향상시켰다. 특히 +8.3pp(Terminal-Bench)와 +6.8pp(τ2²-Bench) 수준의 개선이 보고됐고, ablation에서는 passive 노출·always-on 주입·advisor-only·일반 retrieval보다 selective intervention이 더 강건했다. 또한 Qwen3.5-27B를 SETA로 SFT와 GRPO 학습해 open-weight 메모리 정책의 부분 전이를 보이는 등, 실제 적용 가능성을 초기 단계에서 시사한다.



### The complexities of patient-centred conversational artificial intelligenc (https://arxiv.org/abs/2607.08625)
Comments:
          36 pages (main text), 129 pages (supplementary materials)

- **Prior Approaches**: 기존의 건강 상담용 LLM 챗봇 개발과 평가는 대체로 협조적이고 말이 잘 통하는 시뮬레이션 환자에 의존해 왔다. 하지만 실제 환자들의 대화 패턴과 감정 표현은 사용자마다 크게 달라 현실과의 간극이 생긴다.

- **Core Contribution**: 이 논문은 2,053건의 실제 환자-챗봇 대화를 분석해, 환자 시뮬레이터가 임상 내용뿐 아니라 정서 상태, 대화 전략, 커뮤니케이션 스타일을 분리해 모델링해야 함을 제안한다. 나아가 Turing-inspired 현실성 평가와 환자 personae(5종)를 통해 모델을 다면적으로 시험한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘사람처럼 보이는’ 동시에 임상적으로도 의미 있는 대화를 생성하도록, 환자의 커뮤니케이션 다양성을 구조적으로 반영하는 것이다. 이를 위해 시뮬레이터에서 임상/정서/대화전략/표현양식을 분리 모델링하고, 15명의 평가자가 구분하기 어렵게 현실성을 검증하는 방식으로 설계를 다듬었다.

- **Empirical Impact**: Turing-inspired 현실성 평가에서 시뮬레이션 대화는 실제 대화와 거의 구별되지 않았고, 평가자의 분류 정확도는 55%에 그쳐 ‘거의 비슷함’을 보여줬다. 또한 1,164건의 clinician-graded 사례에서 네 가지 LLM의 urgency assessment를 비교했을 때, 환자의 커뮤니케이션 스타일 차이가 트리아지 결과에 유의미하게 영향을 주어, 이상화된 상호작용만 가정한 시스템은 실제 배치 시 성능 저하와 건강 격차 확대 위험이 있음을 시사한다.



### Improving Ad-hoc Search Effectiveness for Conversational Information Retrieval via Model Merging (https://arxiv.org/abs/2607.08540)
Comments:
          Accepted to SIGIR 2026. 6 pages, 3 figures

- **Prior Approaches**: 대화형 정보 검색(CIR)은 대화 맥락이 길어지며 주제 전환과 대명사/생략 같은 핵심 지시가 얽혀, 최신 턴의 정보요구 해석과 검색 정확도를 동시에 어렵게 만든다. 기존 연구는 대화 데이터로 retriever를 fine-tuning하거나, ad-hoc과 CIR을 함께 다루는 multi-task learning으로 forgetting을 완화해왔다. 하지만 이러한 방식은 재학습 비용이 크고, fine-tuning 후 ad-hoc 기본 성능이 크게 떨어지는 catastrophic forgetting이 반복적으로 관찰된다.

- **Core Contribution**: 이 논문은 학습 없이(model merging, training-free) ad-hoc과 conversational 설정을 동시에 잘 수행하는 단일 retrieval model을 만드는 방법을 제안한다. 구체적으로 ad-hoc용 ANCE와 대화용으로 fine-tuned된 QRACDR를 파라미터 단위로 합쳐, 추가 fine-tuning 없이 성능 균형을 회복한다. 이를 위해 Model Soup(선형 가중합)과 Slerp(비선형 구면 보간) 두 가지 merging 전략을 실험한다.

- **Technical Challenges**: 핵심 기술 과제는 대화 fine-tuning으로 생긴 과도한 전문화로 ad-hoc 검색 능력이 무너지는 문제를, 재학습 없이 어떻게 되돌리느냐이다. 연구진은 인코더 레이어별로 가중치를 다르게 적용하는 depth-wise 보간 계수 λ를 in-domain 데이터에서만 선택해, ad-hoc과 CIR 사이의 트레이드오프를 모델 병합 공간에서 탐색한다. 또한 OOD 데이터 성능을 엄격히 분리 평가해, 병합이 특정 데이터에만 맞춰지는지 여부를 확인한다.

- **Empirical Impact**: 실험 결과, merging은 conversational retriever의 ad-hoc 검색 능력을 크게 회복하면서도 대화 성능 저하를 제한하며, zero-shot 조건에서 최대 15% 높은 NDCG@3를 기록한다. catastrophic forgetting도 multi-task fine-tuning과 유사하거나 더 잘 완화되며, early stopping만으로는 달성하기 어려운 균형을 보여준다. 특히 CAsT에서 대화 모델의 ‘Rewrite’ 입력 성능 손실을 Model Soup/Slerp가 되돌려 주고, session 기반 설정에서는 QRACDR-QMG가 최대 +10.11% 개선을 보이며 범용 retriever로서의 의미를 강화한다.



### Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing (https://arxiv.org/abs/2607.08497)
Comments:
          16 pages, 7 figures, 8 tables. Project page: this https URL Code: this https URL

- **Prior Approaches**: 최근 unified multimodal 모델은 한 아키텍처에서 시각-언어 이해와 이미지 생성/편집을 함께 수행하지만, 장문 대화에서 과거의 모든 시각 토큰을 공통 컨텍스트에 계속 주입하는 구조적 한계가 있습니다. 이로 인해 visual token 폭증으로 추론 예산이 줄고, 긴 누적 컨텍스트에 대한 암묵적 attention 의존이 교차 턴 시각 참조를 불안정하게 만들어 retrieval 오류와 의미 드리프트가 커집니다. 메모리 보강을 시도한 에이전트들도 영상/텍스트를 각각 다루거나(비디오 중심, 단일 태스크 중심) 시각 디테일을 충분히 보존하지 못해 near-duplicate 이미지 구분에 취약합니다.

- **Core Contribution**: 논문은 Cognitive-structured Multimodal Agent(CMA)를 제안하며, 시각 정보를 Episodic Visual Memory(EVM)로 외부화하고 필요할 때만 관련 에피소드를 선택적으로 reactivates 하도록 설계합니다. Perceptual Abstraction Engine(PAE)이 이미지에서 태그·캡션·썸네일의 구조화 표현을 만들고, Cognitive Retrieval Engine(CoRE)이 대화 흐름에 맞는 시각 에피소드를 검색한 뒤, Multimodal Executive Controller(MEC)가 태스크 의도 추론과 실행 계획을 담당합니다. 또한 turn-level 시각 retrieval 감독이 부족한 문제를 Unified Scenario Engine으로 해결해, 회수할 에피소드에 대한 미세 주석이 포함된 다턴 대화 데이터를 생성합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “장기 대화에서 어떤 과거 이미지를 회수해야 하는가”에 대한 명시적 감독 신호를 확보하는 것이었습니다. 기존 데이터셋은 단일 턴 grounding이나 짧은 컨텍스트만 다루는 경우가 많아, 각 턴마다 정답 retrieval 집합을 제공하지 못합니다. 저자들은 Unified Scenario Engine으로 생성된 시나리오에 대해 turn-level retrieval annotations를 만들고, CoRE에는 SFT+RL(지연 학습된 검색 정책 최적화)을 적용하며 PAE도 retrieval 유용성 기준으로 강화해 메모리 작성-검색을 end-task 관점에서 맞춥니다.

- **Empirical Impact**: M2CA-Bench에서 8B CMA는 20턴 세션 기준 retrieval 정확도 91.4%를 기록해, 32B unified baseline을 +8.2%p로 능가하고 per-turn 추론 시간을 23.1s에서 12.7s로 거의 절반 수준으로 줄였습니다. 특히 대화가 길어질수록 격차가 확대되어 Full→Hard에서의 성능 하락이 더 완만했으며, retrieval 정확도가 생성 품질(0~10 점수)로 직접 전이되는 경향도 확인됐습니다. 또한 CMA-Harness로 동일 인지 구조를 도구 실행·웹 접근·이미지 생성/편집까지 확장했으며, monolithic 파라미터 스케일링보다 구조화 메모리와 모듈형 의사결정이 장기 멀티모달 에이전트에 더 확장성 있고 효율적이라는 점을 시사합니다.



### Ensemble Diversity Optimization for Subjective Supervision (https://arxiv.org/abs/2607.08493)
- **Prior Approaches**: 주관적 NLP 태스크에서는 라벨이 단일 정답이 아니라 주석자 간 체계적 불일치를 반영하는 ‘분포’라는 점이 중요하지만, 기존 지도학습은 대개 다수를 단일 타깃으로 뭉개 불확실성을 소실시켰다. Soft-label(라벨 분포) 학습, 엔트로피/불일치 인식 손실, annotator별 모델 등은 보정(calibration)이나 분포 정합을 개선해왔지만, 모델 내부의(또는 앙상블 내부의) 예측 변동성을 명시적으로 제어하는 장치는 제한적이었다.

- **Core Contribution**: 이 논문은 Ensemble Diversity Optimization(EDO)라는 예측공간(prediction-space) 기반 프레임워크를 제안하며, 앙상블의 ‘구성(가중치)·효과적 크기·캘리브레이션’과 ‘signed diversity(불일치 부호 제어)’를 하나의 미분가능한 목적함수로 동시에 최적화한다. signed diversity 정규화는 불일치가 진짜 주관성에서 온 것인지, 구조적 요인(예: 불균형/희소 커버리지)에서 온 잡음인지에 따라 불일치를 보존하거나 억제하도록 유도해 앙상블 collapse를 막는다.

- **Technical Challenges**: 핵심 과제는 (1) 앙상블 간 불일치를 학습 중 명시적으로 통제하면서 (2) 클래스 불균형 하의 소프트 라벨 보정과 (3) 앙상블 크기 선택까지 한 번에 다루는 것이었다. EDO는 Gumbel-Softmax relaxation으로 앙상블 size를 end-to-end로 학습하고, soft F1 surrogate와 class-weighted cross-entropy(soft label 대상)를 통해 utility-보정 trade-off를 맞춘 뒤, reliability-weighted diversity와 signed diversity 정규화를 결합해 내부 변동성을 조절한다.

- **Empirical Impact**: LeWiDi 벤치마크 4개(ArMIS, ConvAbuse, HS-Brexit, MD-Agreement) 실험에서 EDO는 기존 Soft-CE/Soft-MD/Top-5 Voting/WEL 대비 확률적 캘리브레이션을 크게 개선했으며(교차엔트로피 40~78% 감소, Brier score 하락), F1은 경쟁 수준을 유지했다. 또한 앙상블 내부 불일치 제어가 데이터 특성(불균형 강도 등)에 따라 다르게 작동해, 사람 주석자 분포와의 정렬이 더 잘 맞는다는 점을 보여주며 주관성 모델링의 model-agnostic하고 효율적인 경로를 제시한다.



### Token-Flow Firewall: Semantic Runtime Auditing for Persistent AI Agents (https://arxiv.org/abs/2607.08395)
- **Prior Approaches**: 기존 영속( persistent ) AI 에이전트 보안은 룰 기반 가드레일이나 원격 large model 감시처럼 “행동 수준”에서 위험을 따지는 방식이 많았습니다. 이런 방법은 메모리 오염이나 지연된 도구 오남용처럼 소스-싱크 사이의 의미 전파를 놓치거나, 원격 감시를 전면 적용하기엔 비용·프라이버시 부담이 컸습니다.

- **Core Contribution**: 이 논문은 에이전트의 보안 위협이 실제로는 자연어 토큰 흐름을 통해 메모리 업데이트, tool 인자, 파일/구성요소 전달, 컴포넌트 간 통신 같은 경계를 넘을 때 커진다고 관찰합니다. 이를 바탕으로 TokenWall은 “semantic token flow”를 소스-싱크 관점에서 선(先)차단/중재하는 런타임 방어 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 난제는 모든 경계 전이에서 의미적으로 안전함을 검사하되, 로컬에서 처리 가능해야 하며 애매한 고위험 케이스만 선택적으로 더 강한 중재로 넘겨야 한다는 점입니다. TokenWall은 (1) 가벼운 deterministic precheck로 명백한 위반을 먼저 걸러내고, (2) 로컬 small auditor가 경계 유형별로 토큰 흐름의 위험·불확실성·영향을 점수화해 allow/rewrite/defer/block을 내리며, (3) 불확실하거나 고위험이면 fallback arbitration으로 에스컬레이션합니다.

- **Empirical Impact**: CIK-Bench 실험에서 TokenWall은 공격 성공률(ASR)을 12.5%로 낮추면서, 양성 작업의 패스율(PR) 97.4%를 유지했습니다. 또한 양성 케이스에 추가 지연이 평균 0.69초(전체 평균 지연 16.9초) 수준으로 낮아, 원격 대형 모델 의존을 줄이는 실용적인 보안-유틸리티 절충을 보여줍니다.



### Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning (https://arxiv.org/abs/2607.08393)
- **Prior Approaches**: LLM 지식 업데이트는 RAG나 knowledge editing, fine-tuning 같은 방식으로 진행돼 왔지만, 공통적으로 ‘기억은 되는데 추론에서 못 쓰는’ 문제가 보고돼 왔다. 기존 연구들은 주로 성능 차이를 관찰하거나 특정 편집을 수행하는 데 집중했지만, 왜 memorization이 downstream reasoning에 연결되지 않는지의 내부 메커니즘을 정밀하게 추적하는 도구가 부족했다. 또한 grokking 같은 학습 동학 관찰은 있었으나, 본 연구의 초점은 새로운 능력의 출현이 아니라 단일 지식이 논리적 체계에 일반화되는 과정의 실패 원인 규명이다.

- **Core Contribution**: 논문은 fine-tuning 후 발생하는 Knowing–Using Gap(정확도 격차+시간 지연)을 정식화하고, memorization은 빨리 포착되지만 generalization은 늦게 또는 낮게 나타난다는 현상을 체계적으로 규량화했다. 이를 설명하기 위해 self-patching이라는 개입 기반 진단을 제안해, ‘어떤 레이어·토큰 위치에 저장된 표현이 추론에 인과적으로 유효하게 라우팅될 수 있는지’를 공간적으로 지도화한다. 나아가 knowledge–circuit misalignment 가설(저장은 되지만 계산 회로의 유효 위치로 라우팅되지 못함)을 인과 실험으로 지지하고, 이를 활용한 고정 휴리스틱으로 oracle headroom의 58–75%를 복구하는 실용성도 함께 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 “저장된 지식이 내부 어디에 이미 존재하는가”와 “그 표현이 실패한 generalization에서 실제 추론 계산에 어떻게 라우팅되지 않는가”를 동시에 분리해 측정하는 것이다. self-patching은 소스 실행에서 anchor의 은닉표현을 특정 레이어로 캐시한 뒤, 타깃 실행의 후보 레이어에 재주입하여 정답 지표의 인과적 변화(ΔI)를 계층쌍(ls, lt) 전수 스캔으로 측정한다. 이 과정에서 실패 케이스에도 적용 가능하도록 clean 정답 경로 의존을 줄이고, 프롬프트/컨텍스트를 교차하여 위치-정착 패턴이 잡음이나 프롬프트 편법이 아님을 점검한다.

- **Empirical Impact**: 실험은 도메인(생의학·학술)과 아키텍처/스케일을 가로질러 수행했으며, 대부분의 설정에서 memorization은 빠르게 포화하지만 chaining 및 일반화 사용은 낮거나 지연되는 Knowing–Using Gap이 일관되게 관찰됐다. self-patching 결과는 지식 표현이 존재하더라도 mid-layer의 추론 회로로 라우팅되지 못하는 ‘공간적 불정합’을 보여주고, 이를 표현을 옮겨(재배치) 주면 downstream 성능이 즉각적으로 상승함을 인과적으로 확인했다. 실용적 측면에서는 고정된 두 개 레이어-쌍 휴리스틱이 oracle headroom의 58–75%를 회복하며, prompting(CoT 등) 기반이나 일반 교란 대비 개선이 더 구조적임을 시사한다.



### Different Teachers, Different Capabilities: Sub-1B On-Device Distillation for Structured Text Enrichmen (https://arxiv.org/abs/2607.08268)
Comments:
          12 pages, 5 figures. has a same-size non-reasoning-teacher control, a three-judge LLM-as-a-judge panel with a negative control, full-source faithfulness grading, and a per-field routing analysis

- **Prior Approaches**: 구조화 추출은 스키마에 맞춰 항목당 JSON을 생성하는 패턴이라, 큰 모델을 그대로 쓰면 지연·비용이 누적된다. 그래서 Hinton 계열 증류와 black-box 증류, 그리고 structured extraction 전용 증류들이 제안돼 왔지만, 대부분은 전체 품질 한 가지로만 델타를 보여 실제 운영에서 어디가 성능 병목인지 가리기 쉽다.

- **Core Contribution**: 이 논문은 뉴스 기사→요약+5개 라벨(JSON)로 변환하는 작업을 대상으로, 8B reasoning teacher를 0.6B on-device student로 증류했을 때 “소모한 비용 대비 회수한 성능”을 서브태스크(요약 체크리스트, 라벨, faithfulness)별로 분해해 측정한다. 또한 동일 크기 비-reasoning teacher, 더 큰 managed pipeline(합성데이터 확장 포함)까지 비교해 증류 이득의 원인이 추론 성격인지 데이터/스케일인지 분리해 보여준다.

- **Technical Challenges**: 핵심 과제는 소형 모델이 교사 출력의 형식·분류를 따라가면서도, 사용자에게 치명적인 faithfulness(원문 근거 없는 창작)를 유지할 수 있느냐이다. 저자들은 온디바이스 Q4_K_M에서 모두 temperature=0 고정으로, 93개 홀드아웃 기사에 대해 reference-free 3인 패널(블라인드, 구성요소 체크리스트 기반)을 사용해 항목별 재채점 안정성과 편향을 통제했고, 증류 성능을 “기준선→교사 간 간극 대비 회수율”로 보고 축별 실패를 드러냈다.

- **Empirical Impact**: 0.6B 학생은 기사당 약 0.8초로 동작하면서 교사(약 39초) 대비 요약 품질 간극을 58% 회수했고, 기준선(제약 디코딩) 대비 +16.8, few-shot 대비 +4.9로 유의미하게 개선됐다. 다만 faithfulness는 기사 길이가 짧고 얇은 출처(≤1200자)에서 일관되게 흔들리며, 이 서브그룹에서는 학생이 원문 근거 없이 더 그럴듯하게 꾸미는 양상이 관찰돼 “필드별 라우팅 맵(어떤 경우 큰 모델로 보낼지)”을 제안한다.



### AutoPersonas: A Multi-Timescale Loop Engine for Open-Ended Persona Evolution (https://arxiv.org/abs/2607.08252)
Comments:
          52 pages, 13 figures/tables, ancillary public-safe evaluation artifacts included

- **Prior Approaches**: 기존 Generative Agents 계열은 샌드박스 사회 안에서 장소·루틴·역할 등 환경을 설계하고, 메모리·성찰·계획으로 그럴듯한 일상을 굴리는 방식에 강점이 있다. Agentopia처럼 장기 시뮬레이션도 존재하지만 보상(instrumented)이나 닫힌 세계 가정에 기대는 경향이 있어, 보상 없는 open-ended persona evolution에서 생기는 런타임 고착(self-locking) 문제를 정면으로 다루기 어렵다. 장기 메모리 연구는 저장·검색·요약·개인화를 돕지만, ‘어떤 증거가 현재 State의 권위를 가져야 하는가’까지 구조적으로 분리해주지 못한다.

- **Core Contribution**: 이 논문은 장기 페르소나 에이전트가 적응하면서도 정체성을 유지해야 하는데, 그 과정에서 발생하는 런타임 실패 모드로 self-locking을 정의한다. 핵심은 현재 State, 메모리, 히스토리, 환경 요약이 반복 호출마다 ‘권위’를 되감아, 겉으론 그럴듯한 사건이 계속 나오지만 관계·결정·생활 단계·환경이 좁은 끌개(attractor)로 수렴해 버린다는 점이다. 이를 해결하기 위해 AutoPersonas라는 다중 timescale life-environment 엔진을 제안하며, 환경 측 Occurrences·누적 Observations·페르소나 State를 분리하고 evidence governed absorption으로만 State/reachability를 갱신하도록 OSO 루프를 설계한다.

- **Technical Challenges**: AutoPersonas의 관건은 두 힘의 균형이다: 미래를 열어주는 divergence(분기)를 반복 생성해야 하지만, 그 생성물이 증거로 흡수되기 전에는 현재 State의 맥락 중력에 빨려 들어가 ‘고정된 생활 궤도’만 재생산하지 않게 해야 한다. 저자들은 프롬프트 예산 제약(과거·현재 컨텍스트가 토큰의 약 60%를 넘으면 divergence가 악화됨)과, 모델 수준 다양성 붕괴/시스템 수준 컨텍스트 중력이 결합되는 원인을 함께 짚고, OSO의 증거-통제 전파로 이를 완화하려 한다. 또한 동일 실행 내에서도 강제 재구성 대신 per-sample divergence targeting과 context-slice masking 같은 운영 기법으로 매크로 테마 반복을 줄이되 정체성 연속성은 유지하도록 조정한다.

- **Empirical Impact**: 3년 압축 시뮬레이션과 8개 모델·40일 스트레스 테스트에서 self-locking 관련 징후(환경 watermark shell, occurrence hardening gap, 느린 변화 누적 실패, 재귀적 indecision, 약한 관계 지속)가 체계적으로 관찰됐다. 특히 1,600개의 이벤트 생성 결과, 5일 행동-카테고리 반복이 평균 95.2%~97.6%로 높게 나타났고 모델 대부분이 11일차에 90% 이상을 넘었다. 같은-runtime 40일 A/B에서 context-slice masking + divergence targeting은 매크로 테마 반복을 61.8%→36.3%로 낮추며 누적 테마 수를 약 2배 수준으로 늘렸고, 현실 세계 침입 없이도 juvenile-goblin 픽션 월드에서 anti-fixation 체제를 재현해 bounded claim(분기와 증거 흡수를 분리하면 self-locking을 줄일 수 있음)를 지지한다.



### A First-Principles Theory of Slow Thinking and Active Perception (https://arxiv.org/abs/2607.08196)
Comments:
          Published on 2026/05/11 in Journal of Machine Learning

- **Prior Approaches**: 기존 연구는 인지 기능을 확률모형이나 신경망으로 근사하더라도, 사고(thinking)와 지각(perception)을 하나의 수학적 절차로 연결해 설명하는 데는 한계가 있었다. 특히 slow thinking(느린 사고)과 active perception(능동적 지각)을 설계·학습·추론 관점에서 일관되게 정식화한 접근은 드물었다.

- **Core Contribution**: 이 논문은 관측 공간과 잠재 공간의 확률분포를 lifting과 projection으로 다루는 틀 위에서, thinking과 perception을 하나의 수학적 구성으로 전개한다. ‘active lifting’ 이론을 제안해 잠재 시퀀스를 샘플링하고 불확실성을 최대 속도로 줄이려는 내적 동기를 통해, slow thinking LLM을 포함하는 큰 설계 공간과 이를 뒷받침하는 static theory의 부분공간을 도출한다.

- **Technical Challenges**: 핵심 난제는 (1) 복잡한 분포를 신경망 같은 단순 함수족으로 어떻게 일관되게 표현할지, (2) 지각의 능동성(불확실성 감소)과 내적 시간 축을 가진 추론을 동시에 어떻게 만들지, (3) 학습 목표를 정보이론적 관점과 연결해 안정적으로 최적화할지이다. 논문은 잠재 시퀀스 샘플링과 ‘최대 불확실성 감소율’ 원리를 결합하고, 두 계층(표현 계층·샘플러 계층) 위를 올라가는 업그레이드 규칙과 최소 길이 코딩과 유사한 학습 목표를 통해 이를 해결한다.

- **Empirical Impact**: 이론적 결과로서 slow thinking 모델 개선의 3단계 경로, 다중 모달리티 공통 인코더/생성모형 구성, 사람과 유사한 시각 표현의 사전 형성, policy collapse의 가능한 완화책 등 여러 파생물이 제시된다. 다만 제공된 초록만으로는 구체 벤치마크 수치와 성능 향상 정도는 확인되지 않으며, 저자들은 이러한 구조가 slow thinking 포맷의 출현과 행위성(agency)을 특성화한다고 주장한다.



### CausalDS: Benchmarking Causal Reasoning in Data-Science Agents (https://arxiv.org/abs/2607.08093)
Comments:
          55 pages, 10 figures

- **Prior Approaches**: 기존 평가는 크게 두 갈래로 나뉜다. 한쪽은 코딩·도구 사용을 강조하지만 숨겨진 인과 데이터생성 구조가 부족하고, 다른 쪽은 그래프/개입/반사실 추론을 보지만 실제 데이터 분석 워크플로와 결합이 약한 경향이 있다. 또한 인과 평가 데이터셋이 기존 소스의 선별 예시 중심이거나 템플릿 변형 위주라, 체계적으로 새로운 합성 인과 구조를 생성해 다양성을 확보하기 어렵다.

- **Core Contribution**: CausalDS는 에이전트형 데이터사이언스 워크플로에서의 인과 추론을 평가하는 벤치마크로, 각 인스턴스를 “장면(scene)” 단위로 구성한다. 장면은 샘플링된 Structural Causal Model(SCM)과 그로부터 생성한 관측 데이터, 그리고 현실 도메인에 grounded된 합성 자연어 스토리를 함께 제공하며, 선택적으로 실세계 분포를 조합해 경험적 구조를 유지한다. 과제는 Pearl의 Rung 1~3(연관-개입-반사실)에 걸쳐 도출되고, 상당수는 불완전 관측과 observation model 때문에 여러 도구를 거쳐야 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 인과 구조는 통제하면서도 (2) 실제 데이터 분석 난이도는 바꾸고 (3) 모델이 답을 해야 할 상황과 못 해야 할 상황(비식별)을 구분하도록 평가를 설계하는 것이다. CausalDS는 개념적 SCM과 공개 관측값을 분리해, 개념 변수를 잡음 측정 번들로 치환하더라도 인과 추정대상/식별성 라벨은 그대로 유지하면서 수치 추정 난이도만 조절한다. 또한 비식별 질문에 대한 abstention을 ‘정답 행동’으로 명시해, 모델이 원할 때 무리한 추정을 하지 않도록 결정적 채점(ground truth 기반)을 구성한다.

- **Empirical Impact**: 저자들은 953개 장면(관측 모델 변형 포함)으로 구성된 CausalDS 시험에서, 모델 성능이 다섯 축(상징적 인과 추론, 데이터사이언스 실행, 불확실성, abstention, tool-use/coding)으로 분해되어 한 역량만으로는 설명되지 않음을 보인다. 즉, 인과 추론 정확도만으로는 불확실성 보정이나 비식별 상황에서의 올바른 거절 능력을 대변하지 못한다는 실증 결과를 제시한다. 이는 에이전트형 LLM 평가가 인과성·통계 추정·코딩·불확실성·거절을 함께 봐야 함을 시사하며, “causal parrot” 리스크를 줄이는 합성 구조 기반 평가의 방향성을 강화한다.



### Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization (https://arxiv.org/abs/2607.08057)
Comments:
          Accepted to ACL 2026 as a Findings paper

- **Prior Approaches**: 기존 KV 캐시 최적화 연구는 메모리 절감이나 처리량·지연 개선을 목표로 해왔지만, 대개 KV를 전체 서빙 파이프라인의 한 구성요소로 부분적으로 다뤘다. 특히 많은 작업이 생애주기(예: prefill/decoding), 커널·모델 디테일, 또는 특정 기법군(압축·오프로딩 등) 중심으로 분류되어 서로 다른 시스템 효과의 관계를 한눈에 보기 어렵다는 한계가 있었다. 결과적으로 KV 캐시를 둘러싼 시간적(스케줄), 공간적(배치/이동), 구조적(표현/유지) 동작이 어떻게 얽혀 SLO와 품질 저하를 유발하는지 체계화가 부족했다.

- **Core Contribution**: 이 논문은 LLM serving-time KV 인프라를 system-aware KV infrastructure for serving LLMs, sKis로 정의하고, 이를 실행/스케줄(temporal), 배치/마이그레이션(spatial), 표현/유지(structural)이라는 3축 동작 공간으로 재정리한다. 또한 서로 다른 동작 축의 co-design 양상과 동작-목표(지연, 처리량, 메모리, I/O, 에너지) 간 연결을 분석해, 향후 연구 기회를 구조적으로 도출한다. 모델이나 커널 세부를 분리해 새로운 기법을 같은 렌즈에 올려 비교·위치지을 수 있는 기반을 제공한다.

- **Technical Challenges**: KV 최적화는 단순히 메모리를 줄이는 것에서 끝나지 않고, 스케줄링 지연·전송 병목·(de)quantization 비용·캐시 미스 등 실행 경로 전체의 상호작용을 통제해야 한다. 논문은 이를 해결하기 위해 각 축에서 KVS(요청/토큰/커널 레벨의 KV-aware 스케줄), HAE(이질 하드웨어에 적응하는 disaggregation·offloading), MHO/CDO(메모리 계층·장치 간 KV 배치/이동), KVCC(비대칭 양자화·VQ·outlier 처리) 및 KVRM(페이지드 메모리, 프롬프트 공유 인덱스, eviction/예산 할당) 등으로 기술 요소를 체계화한다. 더 나아가 tail latency, 품질 열화, 에너지·신뢰성 같은 운영 현실의 제약이 동작 축 간 간섭에서 발생함을 강조한다.

- **Empirical Impact**: 실증적으로 논문은 커뮤니티 문헌을 행동×목표 매트릭스와 co-design 친화도 네트워크로 요약해, 구조적 방법이 메모리 절감에 가장 많이 기여하고 시간적 방법은 지연·처리량에 더 직접적으로 매핑되지만 tail 지표 보고는 드물다는 점을 드러낸다. 또한 품질 손실이 보편적이며, 동작 축마다 품질 저하 원인이 달라 ‘열화의 통제 가능성’이 핵심 문제임을 정리한다. 마지막으로 에너지·신뢰성·SLO 중심 tail 제어, HAE–CDO의 범용화, 동작 간 공동 최적화 및 unified benchmarks 같은 공백을 명확한 연구 의제로 제시해, 향후 sKis 설계의 방향성을 좌우할 것으로 기대된다.



### From Prompts to Contracts: Harness Engineering for Auditable Enterprise LLM Agents (https://arxiv.org/abs/2607.08028)
Comments:
          32 pages, 6 figures, 16 tables. Reference implementation and evaluation artifacts: this https URL (archived at this https URL)

- **Prior Approaches**: 기존 사내 LLM 앱은 프롬프트와 검색(RAG) 컨텍스트에 핵심 거동이 의존하는 “프롬프트-지배형” 프로토타입으로 출발하는 경우가 많다. 이 방식은 시연에는 유리하지만, 소스 경계·엔터티 라우팅·답변 계약·재현 가능한 트레이스를 프롬프트만으로 강제하기 어렵다는 한계가 있다. 또한 RAG는 근거를 제공하지만, 생성된 문장이 실제 증거를 충족하는지(출처-주장 일치)와 런타임에서의 권한을 구조적으로 통제하는 문제는 별도 레이어가 필요하다고 지적한다.

- **Core Contribution**: 이 논문은 프롬프트-지배형 엔터프라이즈 LLM 프로토타입을 “추적 가능·감사 가능”한 LLM-agent 아키텍처로 재구성하는 harness-engineering 패턴을 제안한다. 핵심은 결정적 거동(소스 게이트, 엔터티 라우팅, claim 선택, 답변 계약, trace 생성)을 코드가 소유하도록 옮기고, 모델은 문장 조립(phrasing) 역할에만 두는 “replaceable composition boundary”를 두는 것이다. 런타임 답변에서 어떤 주장이 허용되는지는 source-backed claims와 manifests가 권위를 갖도록 분리해, 시연이 아닌 제품 수준의 계약 준수를 목표로 한다.

- **Technical Challenges**: 주요 기술 난제는 (1) 어떤 문장을 답변에 넣을 권한이 있는지(source-to-claim 권위), (2) 질문이 어느 기업/엔터티로 라우팅되는지, (3) 내부 trace와 내부 식별자가 사용자에게 새지 않도록 출력 위생을 강제하는 것, (4) 이 모든 것을 모델 교체에도 일관되게 유지하는 검증 설계였다. 논문은 manifests(소스 경계)→evidence records→promoted source-backed claims(런타임 입장 경계)로 이어지는 파이프라인과, reader-facing answer용 출력 계약 및 leakage/link/language 검증 게이트를 코드-owned control layer로 구현해 해결했다. 또한 composition boundary에서 live LLM을 끼우거나 빼는 방식을 동일 계약 검증에 연결하고, 라이브 조합 실패 시 deterministic composer로 폴백하며 실패 원인과 계약 상태를 trace에 기록하도록 했다.

- **Empirical Impact**: 해당 방법은 한국 기업 5개 그룹(상장사 25개) 공개 데이터 슬라이스와 총 113개의 source-backed runtime claims 위에서 평가됐으며, 고정된 검증 시나리오에서 소스 근거·엔터티 라우팅·trace 완결성·출력 위생·추천 언어 계약이 유지됨을 보였다. 모델 교체 실험에서도 hosted 모델 3종에 대해 composition-boundary 실행 270회에서 계약 위반이 모델-조립 측에 국한되고 harness가 탐지·기록했으며, prompt 지시만으로는 내부 trace 누출과 추천형 표현 위반이 완전히 재현됨을 보여 “코드-owned 보장”이 load-bearing임을 입증한다. 특히 외부 bolt-on guardrail은 위반을 막는 대신 유용성이 88/120으로 떨어진 반면, harness는 120/120의 완전한 유용성을 보존해 엔터프라이즈 제품화에서 감사 가능성과 실용성 동시 달성이 가능하다는 의미를 가진다.



### fog: Expressing Motion and Emotion through Function Composition of AI-Generated Cod (https://arxiv.org/abs/2607.07952)
- **Prior Approaches**: 기존 연구는 Heider-Simmel처럼 단순 도형의 움직임에서 감정·사회적 의도를 읽어내는 현상을 기초로, 물리 기반/캐릭터 애니메이션/감정 모델링(예: 정서 공간) 및 자연어→애니메이션 변환을 탐색해 왔습니다. 하지만 AI 애니메이션은 속도·경로·타이밍·완급 같은 “운동 역학”을 사용자가 세밀하게 탐색하고 반복 제어하기 어렵고, 언어와 수치적 운동 표현 사이의 연결이 약하다는 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 Motion과 Emotion을 코드로 조합해 구현하는 함수 합성 프레임워크 fog를 제안합니다. fog는 Verbs(동사), Adverbs(부사), Gestures(몸짓), Emotions(감정)을 포괄하는 추상 클래스 계층을 구성해, Heider-Simmel 스타일의 애니메이션에서 감정 의미를 운동 함수 조합으로 표현하도록 합니다.

- **Technical Challenges**: 핵심 기술 과제는 “의미 있는 감정/동작”을 생성하는 것과, 여러 함수가 서로 다른 수치 채널(속도, 가속, 강도, easing, 오프셋 등)에 동시에 쓰일 때 합성이 알아볼 만하게 유지되는 문제였습니다. 논문은 PrimaryMotion과 SecondaryMotion의 분리, adverb처럼 자연어에 대응되는 래핑 기반 합성, gesture/phase 기반의 감정 표현(에너지·파티클 포함)으로 구현을 구조화했으며, 충돌하는 채널 정렬 문제도 조명했습니다.

- **Empirical Impact**: 452개 fog 생성 애니메이션에 대한 지각 평가에서 의미 인식 정확도는 68%로, 확률 기준선 대비 2.68배 개선을 보였습니다. 또한 전문가·비전문가 10명 혼합 방법 사용자 연구에서 fog 인터페이스가 프롬프트 중심 기준선보다 더 빠른 반복·탐색·제어를 지원함을 보였고, 감정/운동 의미를 연속된 운동 구성요소로 다루는 새 인터랙션 지점을 제안합니다.



### The Memory Wall of Green Software: Empirical Energy Evaluation of Memento Design Pattern (https://arxiv.org/abs/2607.07944)
- **Prior Approaches**: 기존 Green Software Engineering 연구는 DVFS 같은 하드웨어 최적화와 정성적 직관에 의존하는 경우가 많았고, 소프트웨어 아키텍처가 기준 에너지(기본 대사량)를 얼마나 좌우하는지는 실증이 부족했다. 디자인 패턴은 모듈성을 보장하지만 추상화로 인한 런타임 오버헤드가 에너지 관점에서 설계 단계에 잘 드러나지 않는다는 한계가 지적돼 왔다. 특히 Memento처럼 상태 복구에 직접 쓰이는 패턴은 Classic(전체 스냅샷)과 Differential(델타 인코딩) 중 무엇이 실제로 더 에너지 효율적인지 명확하지 않았다.

- **Core Contribution**: 이 논문은 Memento 패턴의 에너지 동역학을 ‘추상화 비용’ 관점에서 정량화한다. Classic 전체 스냅샷 대비 Differential 델타 인코딩이 중간 크기에서는 전력/에너지 절감(최대 65.8%)을 만들지만, 특정 상태 규모에서 런타임이 감당하지 못해 역전되는 현상을 실증했다. 그 결과를 바탕으로 상태 크기와 메모리 압력에 따라 전환해야 한다는 증거 기반 의사결정 프레임(에너지-aware decision matrix)을 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘관측 오차(observer effect)’ 없이 소프트웨어가 유발하는 에너지 차이를 측정하는 것이었다. 이를 위해 소프트웨어 추정이 아닌 Intel RAPL 인터페이스 기반 하드웨어 텔레메트리만 사용하고, .NET 8.0 Server GC에서 Baseline/Classic/Differential 3개 변형을 고정된 조건(주파수 고정, 터보 비활성, 코어 핀닝, 쿨다운)으로 반복 실행했다. 또한 10MB~200MB 상태에서 30회 반복과 이상치(CV 기준) 재실행 절차로 재현성과 통계적 신뢰를 확보했다.

- **Empirical Impact**: 실험 결과는 에너지 효율이 단조 증가/감소가 아니라 ‘스케일-의존적’임을 보여준다. 50~100MB 구간에서는 Differential이 DRAM 활성화를 줄여 Classic 대비 최대 65.8% 에너지 감소를 달성했지만, 150~200MB에서 Gen2 GC가 2,300% 급증하며 Differential이 오히려 Classic보다 25.9% 더 많은 에너지를 사용했다(200MB에서 40.5J). 논문은 이를 Green Software의 “Memory Wall”로 정리하며, managed runtime에서는 알고리즘의 데이터 절감이 런타임(힙/GC/컴팩션) 오버헤드에 의해 상쇄될 수 있음을 실무 아키텍처 설계에 직접 반영할 근거를 제공한다.



### Efficient Safety Alignment of Language Models via Latent Personality Traits (https://arxiv.org/abs/2607.07918)
Comments:
          15 pages, 6 figures. Accepted at COLM 2026

- **Prior Approaches**: 기존 안전 후처리는 유해 콘텐츠에 대한 명시적 거부(supervised refusal) 학습에 의존하는 경우가 많지만, 최근에는 그 정렬 행동이 jailbreak과 적대적 프롬프트에 취약하다는 실패 양상이 드러났다. Latent Adversarial Training(LAT)은 입력이 아닌 잠재공간에서 강건성을 키워 효과적이지만, 유해 프롬프트 기반 대규모 데이터가 필요하고 특정 위해 유형에 과적합되거나 성능 저하가 생길 수 있다. 또한 activation steering 계열은 단일 방향 개입으로 persona drift를 줄이지만 완전한 방어로 이어지지 않아 공격 성공률이 여전히 남는다.

- **Core Contribution**: 이 논문은 Latent Personality Alignment(LPA)로, 명시적 harm refusal 대신 심리측정(personpsychometrics) 문헌에서 가져온 해악 비지시(harm-agnostic) 성격 문장 66개만으로 latent adversarial training을 수행한다. 저자들은 성격 특성에 맞는 잠재 표현이 harm 회피와 공유되는 잠재 구조를 가진다고 보고, 그 부분공간을 adversarial하게 안정화하면 jailbreak이 악용하는 경로를 제약할 수 있다고 가설을 세운다. 특히 LPA는 학습 과정에서 유해 콘텐츠를 본 적이 없는데도 HarmBench에서 직접 요청과 5종 jailbreak에 대해 ASR을 거의 0에 가깝게 만든다고 주장한다.

- **Technical Challenges**: 핵심 난제는 ‘유해 프롬프트 없이도’ 위해 회피와 연결된 잠재 구조를 효과적으로 고정하는 방법을 설계하는 것이다. LPA는 성격 문장을 Big Five의 Conscientiousness, Agreeableness, Emotional Stability와 연결해 이 문장이 긍정/부정 표현일 때의 동의/비동의 출력으로 이진 목표를 구성하고, LAT처럼 잠재표현에 적대적 섭동을 주어 잘못된 완료(동의)로 유도한 뒤에도 올바른 비동의가 유지되도록 학습한다. 또한 학습 효율을 위해 작은 문장 세트에 고정 시스템 프롬프트(자기-진술 성격 평가 프레이밍)를 결합하고, 평가 시에는 최소 시스템 프롬프트로 유틸리티 영향 편향을 줄이도록 설계했다.

- **Empirical Impact**: 실험에서 LPA는 HarmBench ASR을 대부분 공격에서 0에 가깝게 만들며, 표준 유틸리티 벤치마크에서는 기준선 대비 성능 저하가 거의 없었다(다만 학습을 오래 지속하면 유틸리티가 감소해 조기 종료를 사용). LAT는 유해 프롬프트 수천 개와 큰 benign 보전 학습을 요구하는 반면, LPA는 66개 문장만으로 75배 가량 적은 학습 예시로 유사한 강건성을 달성하고 학습도 단일 GPU에서 수분 내 끝난다고 보고한다. ablation에서는 성격 특성의 안전 관련성, 문장-응답 라벨의 일관성, 특히 ‘부정(undesirable) 문장에 대해 불일치시키기’ 전략이 유틸리티 손상 없이 견고함을 만드는 데 중요함을 보여주며, Llama3-8B에서도 유사한 경향의 성능을 확인한다.



### Validating LLMs in social science: Epistemic threats and emerging norms (https://arxiv.org/abs/2607.07915)
Comments:
          28 pages, 2 figures. Main text: 11 pages, Appendix: 11 pages, References: 6 pages

- **Prior Approaches**: 사회과학에서는 LLM을 프롬프트해 데이터 라벨링·코딩, 설문 응답 시뮬레이션, 이념/감정 등 개념의 수치화에 활용하는 흐름이 빠르게 확산되고 있다. 다만 bias, hallucination, 맥락 민감성 같은 LLM 고유의 위험과 함께, 무엇이 “타당한 측정”인지에 대한 표준 검증 절차는 아직 정착되지 않았다. 기존 연구는 우려와 동기를 다뤘지만, 실제로 상위 저널에서 LLM 측정이 어떻게 사용되고 어떤 방식으로 검증되는지의 체계적 실태 분석은 부족했다.

- **Core Contribution**: 이 논문은 8개 주요 사회과학 저널의 논문 코퍼스에서 LLM을 측정 도구로 사용한 사례를 수집해, 측정의 타당성을 뒷받침하는 검증 관행을 실증적으로 분해·분석한다. 특히 개념화(conceptualization)–조작화(operationalization)–검증(validation)이라는 측정 이론 틀로, 연구자들이 구성요소를 어떻게 선택하고 정당화하는지, 그리고 검증이 얼마나 일관되게 수행되는지를 보여준다. 그 결과 LLM 생성 측정치가 분석에서 중심 역할을 하는 경우가 많지만, 검증은 제한적이고 상이한 방식으로만 이뤄져 타당성 주장에 대한 신뢰가 약해질 수 있음을 지적한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) ‘측정하려는 구성개념’을 지나치게 좁게 정의하거나 프롬프트에 포함 자체를 거의 하지 않는 문제, (2) 프롬프트 길이·구조·출력 제약·모델/생성 파라미터·범주 추출(answer extraction) 등 설계 선택지가 과도하게 다양하면서도 근거가 약한 문제다. 또한 숫자/범주 추출 과정의 미보고, 응답 거부(noncompliance) 처리(버리기, 재시도 등) 같은 절차가 검증 편향을 만들 수 있는데도 관찰·보고가 불충분하다고 본다. 논문은 이러한 불확실성을 완화하려면 단일한 convergent validity 비교에만 기대지 말고, 구성 타당성의 여러 렌즈를 함께 적용하며, 측정기기 전체를 투명하게 문서화하고 반복 가능한 절차로 검증해야 한다고 제안한다.

- **Empirical Impact**: 분석 대상은 2022년~2025년 말 사이 상위 저널 논문 2,143편 중, 선정 기준에 따라 LLM이 사회 개념의 정량 측정을 생성하는 ‘측정 과제’ 50개(27편)를 추출한 코퍼스이며, LLM 측정치는 주로 주요 분석의 입력으로 쓰이는 비중이 높았다. 검증은 특정 구성 타당성 요소에 편중되어(가장 흔히 금표준과의 비교, 즉 convergent validity) 여러 과제에서 아예 검증 보고가 없거나, 금표준 라벨(인간 주석 또는 다른 계산 도구)의 품질·신뢰도·샘플 크기가 충분히 점검되지 않는 사례가 확인됐다. 저자들은 개념 정의의 정밀성, 금표준 데이터의 주석 품질(코드북, 다중 주석, 우연 보정 일치도 등), 그리고 여러 타당성 렌즈를 결합한 검증을 요구하는 보고 규범의 필요성을 강조하며, 사회과학에서 LLM 측정이 신뢰성 있게 자리잡는 데 실질적 기준을 제공한다.



### Multimodal Unlearning Across Vision, Language, Video, and Audio: Survey of Methods, Datasets, and Benchmarks (https://arxiv.org/abs/2607.07907)
Comments:
          Accepted to ACL Findings 2026

- **Prior Approaches**: VLM·DM·LLM·AFM 같은 멀티모달 파운데이션 모델은 학습 데이터에서 온 민감정보·저작권·편향·안전 문제의 교차모달 연관을 그대로 내재할 수 있다. 삭제 요청이나 정책 업데이트 이후의 재학습은 비용·시간 문제로 현실성이 낮고, 지식이 공유 표현 전반에 분산돼 있어 targeted forgetting도 어렵다. 기존 접근은 주로 단일 모달 또는 제한적 설정에 집중돼, 모달 간 연관 제거와 성능 유지의 균형을 체계적으로 다루기 힘들다.

- **Core Contribution**: 이 논문은 멀티모달 unlearning을 비전·언어·오디오·비디오 전 범위에서 통합적으로 정리하는 “시스템 관점”의 설계를 제공한다. 또한 deletion 강도, retention(유지), efficiency(효율), reversibility(되돌림 가능성), robustness(강건성) 간 트레이드오프를 모달과 모델 구조에 걸쳐 비교 가능하도록 분류 체계를 제안한다. 이를 통해 실무 적용 관점의 선택 기준과 향후 연구 방향을 명확히 한다.

- **Technical Challenges**: 핵심 기술 난관은 공유된 표현 공간에 지식이 분산되어 있어 특정 모달/연관만 선택적으로 제거하면서도 전체 성능을 유지해야 한다는 점이다. 논문은 이러한 문제를 모달별 삭제 범위와 모델 아키텍처에 따른 학습·보정 방식의 차이로 구조화해 설명하고, 각 접근이 요구하는 데이터/추론 조건과 실패 모드를 함께 정리한다. 결과적으로 “어떤 삭제 목표를 어떤 시스템 구성으로 달성할지”를 비교 관점에서 재현 가능하게 만든다.

- **Empirical Impact**: 조사는 최근 연구 흐름과 등장 응용, 그리고 아직 미해결 과제를 함께 정리해 멀티모달 unlearning의 실험 설계와 평가 기준 수립에 도움을 준다. 특히 삭제 효력과 성능 유지, 효율, 되돌림 가능성, 견고성의 균형을 같은 축에서 비교하도록 해, 향후 벤치마크·리포팅 관행을 개선하는 데 의미가 있다. 또한 큐레이션된 저장소를 공개해 연구자들이 관련 방법을 빠르게 탐색하고 비교할 수 있도록 지원한다.



### Collective Intelligence with Foundation Models (https://arxiv.org/abs/2607.07729)
Comments:
          Accepted as a book chapter in "Advances in Global Applied Artificial Intelligence" (G. A. Tsihrintzis, M. Virvou, N. G. Bourbakis, L. C. Jain, Eds.), authenticated version will be published in Springer series: Learning and Analytics in Intelligent Systems

- **Prior Approaches**: 기존 연구는 chain-of-thought, self-consistency처럼 단일 모델 내부에서 다양한 추론 경로를 만들거나 답을 선택하는 방식에 집중해 왔습니다. 또한 AutoGen, MetaGPT 같은 멀티에이전트 협업도 있었지만, 다수 에이전트의 ‘오류 탐지’와 ‘합의 품질’을 정량적으로 분해해 설명하기는 어려웠습니다. self-refinement이나 Constitutional AI는 자체 비판을 사용하지만, 같은 모델의 편향이 그대로 남아 맹점 교정이 제한될 수 있다는 문제가 제기됩니다.

- **Core Contribution**: 이 논문은 서로 다른 foundation model을 역할별로 배치한 멀티에이전트 추론 프레임워크를 제안합니다. Solver가 독립 초안을 만들고, Critic이 구조화된 오류 분석과 수정안을 제공하며, Aggregator가 합의해 최종 답을 생성합니다. 더불어 의미·수치·절차(단계별 reasoning)까지 함께 평가하는 Scoring 모듈을 도입해, 최종 정답뿐 아니라 중간 과정의 질을 드러냅니다.

- **Technical Challenges**: 기여를 실현하려면 (1) 에이전트 간 중복 없이 ‘실제로 다른 오류’를 포착할 수 있는 모델 조합, (2) 비평·수정·합의가 잡음이 아닌 논리 교정으로 이어지게 하는 구조화, (3) 단계별 추론 품질을 측정할 수 있는 평가 설계가 필요합니다. 저자들은 solver마다 독립 샘플 초안을 만들고, critic은 스타일이 아닌 논리/계산 오류를 겨냥한 구성 프롬프트로 1회 비판만 수행하며, aggregator는 다수 합의 근거를 우선하되 불일치 시 불확실성을 명시하도록 했습니다. 평가는 임베딩 유사도(semantic coherence), 수치 추출 기반 overlap, 그리고 reference의 단계와 대응시킨 step-wise accuracy로 수행했습니다.

- **Empirical Impact**: 8개 과학·수학 분야 벤치마크에서 ablation 결과, 전체 평균 성능은 구조와 중복 샘플링만으로는 소폭(예: 0.52→0.60/0.61) 개선되는데 그칩니다. 반면 heterogeneous(모델 다양성) 구성은 step-wise accuracy가 0.64로 크게 상승해 homogeneous(0.27~0.28) 대비 2.3배 수준의 개선을 보였고, 중간 단계의 정합성까지 함께 좋아졌습니다. 즉 “정답 맞힘”을 넘어 추론 과정의 설명가능성·감사가능성을 강화하는 데 모델 다양성이 핵심이라는 점을 실증적으로 입증해, 고신뢰 과학/산업 의사결정에 대한 멀티에이전트 설계 방향을 제시합니다.



### SPL: Orchestrating Workflows with Declarative Deterministic-Probabilistic Composition (https://arxiv.org/abs/2607.07727)
Comments:
          24 pages, 2 figures, under review at TMLR

- **Prior Approaches**: 기존 LangGraph, AutoGen, CrewAI 같은 에이전트 프레임워크는 LLM 호출을 위한 오케스트레이션 로직을 주로 Python에 묶어 제공해, 추론과 상태 관리가 프래그먼트하게 분리되는 문제가 있었다. SymPy, SageMath, Lean 같은 심볼릭 도구는 정확하고 기계검증 가능한 결과를 내지만, LLM 기반의 확률적 생성 흐름과 같은 명세에서 자연스럽게 연결하기 어렵다. 결과적으로 ‘두 모드 갭’—확률적 생성과 결정적 계산·검증을 한 문법으로 선언하기 어려움—이 생기며, 모드 전환마다 어플리케이션 글루 코드가 늘어난다.

- **Core Contribution**: 이 논문은 확률적( GENERATE/EVALUATE )과 결정적( SOLVE/ASSERT ) 계산을 한 파일(.spl)에서 동시에 조합하는 선언형 언어 SPL을 제안한다. SOLVE와 ASSERT는 IPython 커널로 라우팅되어 결과를 기계검증 게이트로 통과시키며, GENERATE 산출물은 @variable 네임스페이스를 통해 모드 사이를 넘나들 수 있다. 또한 DODA 원칙(Design Once, Deploy Anywhere)을 통해 모델 제공자(예: Ollama/OpenRouter/Anthropic/Momagrid)와 검증기(예: SymPy/Sage/Lean)를 실행 시점 플래그로 선택하도록 설계했다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM의 비결정적 텍스트 출력과 심볼릭 커널의 정확한 입력을 ‘명세 수준에서’ 안전하게 연결하고, 모드 경계를 런타임이 일관되게 관리하는 것이었다. SPL은 {@variable} 보간을 모드-크로싱 심( seam )으로 삼아, 커널로 넘기기 전에 런타임이 변수 저장소를 기반으로 표현식을 구성하고 sandboxed IPython에서 실행하되 문법/이름 오류는 EXCEPTION으로 처리한다. 덧붙여 ASSERT 실패 시 같은 명세 안에서 EXCEPTION/WHILE/EVALUATE 등을 이용해 재시도·에스컬레이션(Verifier ladder)을 이어가도록 했다.

- **Empirical Impact**: 실험에서는 78개 레시피로 SPL 사용성을 검증하고, 10개 모델×20개 문제×2개 암( LLM-only vs solver 중심 )×3회 반복, 총 1,200 run의 통제 실험으로 검증 성능을 비교했다. solver arm은 기계검증 정확도가 82–93% 수준이며(sonnet-4-6: 85%, gemma4:e2b: 93%), LLM-only arm은 수학적 검증 없이 출력 생산만 평가되어 ‘검증된 정확성’과 ‘비검증 유창성’을 분리해 보여준다. 또한 실패 원인은 포맷 불일치가 아니라 solver_error(커널이 거부한 표현)로 나타났고, 백엔드 난이도 그래디언트로 SymPy 78% vs Sage 54%가 관측되어, 향후 검증기 선택과 표현 생성의 품질이 중요함을 시사한다.



### Uncertainty-gated selection for block-sparse attention (https://arxiv.org/abs/2607.07724)
- **Prior Approaches**: 장기 문맥 언어모델에서 block-sparse attention은 softmax의 O(N^2)를 줄이기 위해, 질의마다 key block을 top-k로 고르고 그 집합에만 attention을 수행하는 방식으로 널리 쓰인다. Quest, H2O, SnapKV, MInference, NSA, MoBA, SSA 등은 모두 블록 스코어링 백본은 달라도 결국 per-query top-k 컷오프로 귀결된다. 하지만 top-k는 k번째와 (k+1)번째 점수가 거의 동률일 때도 동일한 예산으로 즉시 결정을 내려, 정작 정답 근거 evidence를 담은 블록이 탈락하면 이후 레이어가 복구하기 어렵다는 구조적 한계가 있다.

- **Core Contribution**: 이 논문은 top-k 컷오프를 value-of-information(VoI) 관점의 의사결정으로 재정의하고, 각 Q-tile/head에서 “컷오프가 얼마나 애매했는지”를 나타내는 불확실성 신호(정규화 cutoff margin)를 계산한다. 이 값이 작은(동률에 가까운) 타일에는 보존된 블록 수를 선택적으로 2배(k budget 확장)로 늘리는 router를 얹어, 위험한 컷오프에서만 추가 탐색 비용을 지불한다. 또한 이 router는 블록 점수를 만드는 백본과 무관하게 설계되어 Quest 같은 기존 블록 스코어링 방법 위에 그대로 조합 가능하다.

- **Technical Challenges**: 핵심 기술 과제는 “top-k의 동률/애매함”을 값비싼 재계산 없이 예산 확장 트리거로 바꾸는 것과, 커널 실행 형태를 깨지 않으면서 선택적 확장을 효율적으로 구현하는 것이다. 논문은 top-(k+1)에서 얻는 추가 정보로 cutoff margin을 저비용 산출하고, 다중 head의 상관을 고려해 head별 margin을 타일 단위로 평균낸 뒤 레이어별 quantile 임계값으로 확장할 타일의 하위 분율만 선택한다. 이어 fused selection-plus-kernel 경로에서 kv_idx를 직접 생성해, router/non-router 모두 동일한 커널 dispatch 형태를 유지하도록 구현해 long context에서도 지연을 억제한다.

- **Empirical Impact**: 실험에서는 RULER NIAH-multikey( n=100 )와 LongBench-v2 medium( n=215, 전체 medium 서브셋 )에서 router-on-Quest가 top-k 대비 큰 폭의 paired recall/정확도 개선을 보였다. 예를 들어 LongBench-v2 medium에서 router-on-Quest의 paired recall은 0.75로 top-k 0.47 대비 +28%p 상승했으며(SSA-style baseline 대비 McNemar p<0.01), 같은 맥락 길이에서 dense에 매우 근접하거나 근소한 차이를 유지했다. 또한 Qwen2.5, Mistral-Nemo, Qwen3.6 등 3종 아키텍처 4개 모델에서 재현되었고, 128K 컨텍스트에서도 dense 정확도의 큰 비율을 보존하면서 fused 커널 기준 wall time은 dense 대비 각각 0.62×/0.80× 수준으로 유지되어 실용적 trade-off를 입증했다.



### ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modeling (https://arxiv.org/abs/2607.05583)
Comments:
          8 Pages. Accepted at ICANN 2026

- **Prior Approaches**: 기존 장문 언어모델링은 transformer의 self-attention을 중심으로 발전해 왔으며, 많은 변형들도 attention을 직접 대체하기보다 계산 효율을 손보는 데 집중해 왔습니다. 다만 context 길이가 길어질수록 attention의 계산/메모리 비용이 커져 효율이 급격히 떨어지는 문제가 남아 있습니다. linear·kernelized attention, Hyena/S4/Mamba 같은 state-space 계열이 quadratic attention 없이도 성능을 낼 수 있음을 보여주지만, 여전히 attention 기반 계열의 연산적 계보 안에 머무는 경우가 많습니다.

- **Core Contribution**: 이 논문은 self-attention을 causal resonant field mixing으로 완전히 교체한 ResonatorLM을 제안합니다. 토큰열을 단일 driven 1D latent field로 보고, 위치 간 정보 전달은 damped resonator의 causal 함수로 구현해 장문에서도 효율을 유지하려는 전략입니다. 학습(그리고 prefill)은 FFT 기반 causal convolution으로 병렬 경로를 살리고, 생성(decode)은 고정 크기 recurrent state로 캐시 성장을 막도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) attention과 동등한 정보 전파 능력을 유지하면서 (2) 학습·prefill과 decode에서 동작 형태를 효율적으로 분리하는 동시에 (3) causal성과 수치 안정성을 확보하는 것입니다. 저자들은 각 head에 damped oscillatory 커널을 학습하고, training/prefill에는 FFT로 O(n log n) 혼합을 수행하며 decode에는 동일한 커널 계열을 고정 크기 상태 업데이트로 대응시켰습니다. 또한 반감기(half-life) 범위와 prefix에 대한 causality test, 상태 에너지 decay 등으로 causal 구조와 시간 스케일을 점검해 모델이 한쪽으로 붕괴하지 않음을 확인했습니다.

- **Empirical Impact**: 6M 매치드 설정에서 ResonatorLM은 32K 토큰까지 decode 속도가 표준 최적 transformer 대비 6.47x 빨라졌고, WikiText의 정확도도 55.32%에서 61.31%로 상승했습니다. 95% 신뢰구간 기준 perplexity와 accuracy 모두에서 반복 실험(여러 seed)으로 일관된 개선이 관찰되며, local path 제거나 head coupling 변경 같은 일부 ablation에서도 성능 변동이 크지 않았습니다. 더 나아가 kernel-only scaling benchmark에서는 8K에서 440.29x, 32K에서 575.86x 같은 큰 알고리즘적 속도 이득을 보고해, 장문 효율에서의 의미 있는 함의를 강조했습니다.



New uploads on arXiv(cs.IR)

### Improving Ad-hoc Search Effectiveness for Conversational Information Retrieval via Model Merging (https://arxiv.org/abs/2607.08540)
Comments:
          Accepted to SIGIR 2026. 6 pages, 3 figures

- **Prior Approaches**: 대화형 정보 검색(CIR)은 대화 맥락이 길어지며 주제 전환과 대명사/생략 같은 핵심 지시가 얽혀, 최신 턴의 정보요구 해석과 검색 정확도를 동시에 어렵게 만든다. 기존 연구는 대화 데이터로 retriever를 fine-tuning하거나, ad-hoc과 CIR을 함께 다루는 multi-task learning으로 forgetting을 완화해왔다. 하지만 이러한 방식은 재학습 비용이 크고, fine-tuning 후 ad-hoc 기본 성능이 크게 떨어지는 catastrophic forgetting이 반복적으로 관찰된다.

- **Core Contribution**: 이 논문은 학습 없이(model merging, training-free) ad-hoc과 conversational 설정을 동시에 잘 수행하는 단일 retrieval model을 만드는 방법을 제안한다. 구체적으로 ad-hoc용 ANCE와 대화용으로 fine-tuned된 QRACDR를 파라미터 단위로 합쳐, 추가 fine-tuning 없이 성능 균형을 회복한다. 이를 위해 Model Soup(선형 가중합)과 Slerp(비선형 구면 보간) 두 가지 merging 전략을 실험한다.

- **Technical Challenges**: 핵심 기술 과제는 대화 fine-tuning으로 생긴 과도한 전문화로 ad-hoc 검색 능력이 무너지는 문제를, 재학습 없이 어떻게 되돌리느냐이다. 연구진은 인코더 레이어별로 가중치를 다르게 적용하는 depth-wise 보간 계수 λ를 in-domain 데이터에서만 선택해, ad-hoc과 CIR 사이의 트레이드오프를 모델 병합 공간에서 탐색한다. 또한 OOD 데이터 성능을 엄격히 분리 평가해, 병합이 특정 데이터에만 맞춰지는지 여부를 확인한다.

- **Empirical Impact**: 실험 결과, merging은 conversational retriever의 ad-hoc 검색 능력을 크게 회복하면서도 대화 성능 저하를 제한하며, zero-shot 조건에서 최대 15% 높은 NDCG@3를 기록한다. catastrophic forgetting도 multi-task fine-tuning과 유사하거나 더 잘 완화되며, early stopping만으로는 달성하기 어려운 균형을 보여준다. 특히 CAsT에서 대화 모델의 ‘Rewrite’ 입력 성능 손실을 Model Soup/Slerp가 되돌려 주고, session 기반 설정에서는 QRACDR-QMG가 최대 +10.11% 개선을 보이며 범용 retriever로서의 의미를 강화한다.



### Log-Insight: Automating Microservice Incident Diagnosis via Neuro-Symbolic Log Analysis (https://arxiv.org/abs/2607.08529)
- **Prior Approaches**: 기존 템플릿 기반 로그 파서(Drain, Spell)는 로그를 축약해 주지만, 왜 이상인지에 대한 의미론적 이상 원인 해석은 제공하지 못합니다. 딥러닝 이상탐지(DeepLog, LogRobust)는 블랙박스 신호에 가깝고, 라이브 RCA에 필요한 자연어 형태의 설명을 직접 생성하기 어렵습니다. LLM 기반 RCA 파이프라인은 컨텍스트 창 초과와 도메인 환각, Lost in the Middle 문제로 인해 대규모 원천 텔레메트리에는 취약합니다.

- **Core Contribution**: Log-Insight는 SRE의 수동 트리아지 흐름을 자동화해, LLM에 원시 로그를 그대로 넣지 않고 “사전 압축·사전 순위화된 증거 묶음”을 전달하는 구조를 제안합니다. 상징적(Symbolic) 단계가 스키마 이해·패턴 클러스터링·통계적 이상 순위를 수행하고, LLM은 이를 바탕으로 근본원인 가설 보고서를 합성하도록 역할을 제한합니다. 이를 통해 문맥 오버플로와 환각 위험을 구조적으로 낮추는 동시에, SRE가 바로 행동 가능한 RCA를 목표로 합니다.

- **Technical Challenges**: 핵심 기술적 난제는 30분 구간의 로그가 LLM 컨텍스트 예산을 압도할 정도로 방대하다는 점이며, 또한 스키마가 서비스마다 이질적이라는 점입니다. Log-Insight는 2-pass 샘플링과 Drain3 템플릿 클러스터링으로 로그량을 1,000~7,000배 줄이고, 섀넌 엔트로피 기반으로 고카디널리티 메타데이터를 압축해 46,000자 API 예산 내에 진단 신호를 유지합니다. 마지막으로 Forensic Evidence(임계 힌트의 필드/값 및 오류·성공 로그 내 확률·스큐 비율)를 문서 우선순위로 앞에 배치해, LLM이 “주어진 통계와 모순”되지 않는 범위에서만 서술하도록 제한합니다.

- **Empirical Impact**: Huawei 프로덕션에 배포된 Log-Insight는 11개 과거 사고(110 runs)에서 매크로 평균 MRR 0.790을 달성하고, 정답 근본원인을 top-3 가설에 90% 이상 포함했습니다. 평균 엔드투엔드 지연은 1분 미만(평균 27초)으로 라이브 대응 요구를 충족하며, 직접 원시 로그 주입 및 템플릿 주입 기반 대조군을 일관되게 능가했습니다. 특히 Forensic Evidence의 통계 투명성이 운영자 채택의 주요 요인으로 나타나, 시스템이 “불투명한 오라클”이 아니라 “검증 가능한 조사 보조자”로 인식되도록 전환시킨 점이 의미 있습니다.



### Conversational Retrieval and On-the-Fly Knowledge Modeling of Historical Penitentiary Repression Records (https://arxiv.org/abs/2607.08459)
Comments:
          Accepted at ICDAR2026

- **Prior Approaches**: 기존 역사 문서 분석은 OCR/HTR 전처리(이진화, 레이아웃 분석, 줄 검출)를 거친 뒤 NER·정보추출 같은 후처리에 의존해 왔다. 최근에는 conversational RAG로 자연어 질의에 답하지만, 문서 단위 근거에 머물러 문서 컬렉션 전체의 관계 추론이나 전문가 지식의 동적 반영이 어렵다는 한계가 있었다. GraphRAG 역시 다단계 파이프라인을 그래프 메모리로 확장하려 하지만, 역사 아카이브의 필기·열화·잡음·불완전 메타데이터 환경에서는 전사/추출 불확실성 관리가 특히 취약하다.

- **Core Contribution**: 이 논문은 대화 중 생성되는 그래프 기반 지식 모델링(on-the-fly knowledge modeling) 시스템을 제안한다. RAG로 가져온 사실과 전문가가 입력한 사실을 (subject, predicate, object) 형태로 그래프에 축적해, 이전 세션의 지식을 지속 메모리처럼 활용하며 다문서 장기 의존 질의와 링크 탐색을 지원한다. 또한 그래프에 넣기 전 단계에서 OCR 불확실성을 줄이는 전사 향상 파이프라인을 함께 제시한다.

- **Technical Challenges**: 핵심 난제는 열화된 역사 문서에서 신뢰도 높은 전사를 얻고, 그 전사를 기반으로 그래프 추론 시 환각과 노이즈가 축적되지 않게 하는 것이다. 이를 위해 네 가지 이진화로 다중 OCR 후보를 만들고(다이렉트 경로+레이아웃 인식 경로), 문자 단위·단어 단위 voting과 phi-4 기반 LLM 보정의 앙상블을 통해 합의 전사를 생성한다. 이후 RAG 답변을 LLM이 triplet으로 추출하되, 술어 길이/중복/엔티티 분해를 정규화 규칙과 문자열 유사도로 정리하며, 질의에 대해 그래프 근거만 사용하도록 JSON 템플릿과 충분성(found_information) 플래그로 제약한다.

- **Empirical Impact**: 스페인 내전기(1937–1940) 군사재판 기록 65건(이미지 130장)에서 표준 Tesseract 단독 대비 WER을 72.8%→33.5%, CER을 46.7%→22.4%로 크게 낮춰 전사 품질 향상을 입증한다. RAG 태스크에서는 Custom Exact 0.824, Faithfulness 0.886, Context Precision 0.888을 기록했지만 Context Recall이 0.776으로 나타나 ‘정답 문서 검색’과 ‘정답을 만들 만큼의 맥락 충분성’이 다를 수 있음을 보여준다. 복잡 질의는 그래프 시스템이 87%로 활성화되는 등 역할 분담이 관찰되며, 그래프 기반 장기 지식 탐색 도구로서의 실용성을 시사한다.



### H3D: Benchmarking Unsupervised Text Hashing for Fine-Grained Document Deduplication (https://arxiv.org/abs/2607.08382)
- **Prior Approaches**: 기존 문서 해싱은 MinHashLSH처럼 후보 생성-검증 파이프라인에 얹어 대규모 중복 제거에 활용돼 왔지만, 핵심은 “인덱싱 방식”까지 포함한 시스템 설계에 치우치기 쉬웠습니다. 또한 비지도 신경 해싱은 self-supervised/contrastive 등 학습 변수가 많아, 동일 프로토콜에서 해시 생성기·유사도 스코어·임베딩 동작만을 공정 비교하기 어렵다는 한계가 있었습니다. 결과적으로 fine-grained 과학 문서 수준의 미세한 변형을 통일된 방식으로 평가하는 벤치마크 부재가 문제로 남았습니다.

- **Core Contribution**: 이 논문은 unsupervised text hashing 벤치마크 H3D를 제안하며, CSFCube(논문 facet 단위)와 RELISH(바이오메디컬 더 큰 규모)를 동일한 query–candidate 랭킹 프로토콜로 평가합니다. MinHash, SimHash, Winnowing, FuzzyHash, FlyHash 같은 비학습 non-learning 해시와, frozen BGE 임베딩 위의 BGE-BIHash/BGE-LSHash 양자화까지 한 프레임에 묶어 비교합니다. 특히 방법을 “정확도만”이 아니라 MAP, NDCG@20, 효율, 압축/잡음 상황에서의 견고성까지 함께 진단할 수 있게 설계한 점이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 서로 다른 해시 출력 형태(문자열/집합/벡터)를 같은 랭킹 인터페이스로 정렬해 공정 비교하는 것이었습니다. 이를 위해 표현 타입별 어댑터와 type-specific similarity measure를 통일 규칙으로 연결해, 동일한 query–candidate 스코어링/랭킹 절차에서 성능 차이를 해시·스코어 선택 탓으로 해석 가능하게 만들었습니다. 또한 BGE 기반 semantic-sensitive 해싱은 임베딩 차원·저장/연산 비용 때문에 post-hoc 양자화가 필수이므로, 결정적 BIHash와 neighborhood-sensitive 성격의 LSHash를 같은 설정에서 비교하며 양자화 편향 차이를 분석합니다.

- **Empirical Impact**: 실험 결과는 일관된 트레이드오프를 보여줍니다: lexical/structural fingerprint는 near-duplicate 매칭에서 경쟁력이 높지만, 문장/내용 재작성처럼 표면 변형이 큰 경우에는 semantic-sensitive(=BGE 후 양자화) 쪽이 유사도를 더 잘 보존합니다. 다만 semantic-sensitive 방식은 더 높은 계산 비용을 동반해, 정밀도와 효율의 균형점이 분명해집니다. 나아가 특정 해시 표현에서 서로 다른 similarity measure가 rank-equivalent가 되는 조건을 분석해, 방법 비교의 재현성과 해석 가능성을 높였다는 의미가 있습니다.



### DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify (https://arxiv.org/abs/2607.08365)
Comments:
          Accepted by IJCAI'26

- **Prior Approaches**: 기존 검색·추천 대규모 정보검색은 Multi-Stage Cascade Architecture(MCA)로 coarse-to-fine retrieve-and-rank 파이프라인을 구성해 효율과 성능을 함께 노려왔다. 다만 단계별 최적화 목적이 달라 초기 검색(리트리버) 실수가 이후 단계에서 되돌리기 어려워 최종 품질이 저하될 수 있다. 또 end-to-end generative 정보검색(GenIR)은 auto-regressive 디코딩 때문에 온라인 서빙 지연이 크고 추천 리스트 길이 같은 제어가 불리하다.

- **Core Contribution**: DaV-Gen은 generative retrieval의 병목을 “Draft-and-Verify”로 근본적으로 바꾸는 unified 프레임워크를 제안한다. 단일 모델 안에서 후보 초안을 빠르게 만드는 drafting과 후보를 정밀하게 점수화하는 verification을 함께 학습·추론하며, 순차 토큰 생성 대신 병렬 검증으로 지연을 낮춘다. 또한 sparse한 구조 정보와 dense한 의미를 함께 쓰는 Hybrid Sparse-Dense Representation과 목적 불일치를 완화하는 통합 학습 목표를 결합한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) drafting에 필요한 효율성과 verification에 필요한 정밀 의미를 동시에 담을 표현 설계, (2) cascade의 objective inconsistency를 하나의 최적화 공간에서 정렬, (3) auto-regressive로 인한 O(L) 지연을 온라인 친화적 구조로 전환하는 것이다. DaV-Gen은 RQ-VAE 기반 dense 컨텐츠 벡터와 Semantic ID의 계층 구조를 평균 풀링한 sparse 구조 벡터를 결합해 MIPS 호환 임베딩을 만들고, 학습에서는 contrastive loss(효율적 drafting용 임베딩)와 generative likelihood(문맥 기반 세밀 verification용)를 fusion loss로 연결한다. 추론 시에는 ANN 기반 후보를 고르고, Broadcasted Prefix Caching을 활용해 KV 캐시를 한 번만 계산한 뒤 후보 배치를 단일 forward 패스로 병렬 scoring한다.

- **Empirical Impact**: 실험에서 DaV-Gen은 Amazon Beauty/Sports/Yelp 등 추천 벤치마크에서 기존 discriminative 모델은 물론, auto-regressive 기반 unified 모델(예: OneRec)보다도 더 좋은 성능을 보였다. 대규모 검색 데이터 Ind-Search(100M 아이템, 500M 로그)에서는 Recall@50 77.4%로 dense retriever(44.7%)를 크게 상회하고, NDCG@10과 MRR@10도 MoE 대비 개선(예: NDCG@10 0.569→0.589, MRR@10 0.895→0.942)했다. 실제 프로덕션 배포(수백만 사용자, 1주)에서도 ATS +2.09%, UCVR +0.47%, ASS +0.31% 상승을 확인했으며 추론 지연은 약 70ms로 cascade(약 130ms) 대비 2.5× 빨라, pure generative(약 3s) 대비 실시간 서빙 적합성을 입증했다.



### BACH: A Bayesian Admixture of Contrastive Heads for Multi-Interest Two-Tower Retrieva (https://arxiv.org/abs/2607.08107)
- **Prior Approaches**: 두 개 타워(two-tower) 검색은 사용자와 아이템을 임베딩 공간에 매핑한 뒤 내적(inner product)으로 호환도를 점수화하며, 후보 생성에 널리 쓰인다. 멀티-인터레스트 모델은 한 사용자를 여러 head(interest)로 표현하지만, hard-routing 학습에서는 관측된 positive에 가장 잘 맞는 head만 그라디언트를 받아 routing collapse로 head 활용이 줄어든다. 또한 기존 서빙은 max 점수 head를 선택하거나 per-head top 목록을 합치는 방식에 그쳐, 사용자별로 각 interest의 중요도가 얼마나 다른지에 대한 명시적 추정이 부족하다.

- **Core Contribution**: BACH(Bayesian Admixture of Contrastive Heads)는 멀티-인터레스트 two-tower를 head들의 사용자별 혼합(mixture)으로 보고, variational inference로 학습한다. 그 결과 모든 head에 소프트 책임(responsibility)을 분배해 head collapse를 완화하고, 사용자별 interest 중요도 가중치 πu를 산출해 서빙에서도 그대로 재사용한다. 또한 head를 글로벌 코드북 형태로 공유하는 변형도 제시해, 일부 상황에서 retrieval을 미리 계산 가능한 형태로 확장한다.

- **Technical Challenges**: 핵심 난점은 hard argmax routing을 부드럽게 바꾸되, two-tower의 sampled softmax 학습/서빙 규칙과 정확히 정합되게 만들고 ELBO 최적화에서 생기는 결합 항을 다루는 것이다. BACH는 power-spherical(기본) 또는 von Mises-Fisher 인식(인식모델) 분포로 head에 대한 posterior를 샘플 없이 계산하고, 이 분포가 서빙에서 쓰는 self-normalized mixture와 일치하도록 설계해 soft 책임을 얻는다. 더불어 head별 concentration은 KL 항을 통해 자동으로 self-regularization되도록 구성해 추가 prior 없이도 routing sharpness를 안정적으로 학습한다.

- **Empirical Impact**: MovieLens-20M, Taobao, Netflix의 세 벤치마크에서 BACH는 hard-routing 멀티-인터레스트 및 단일 벡터 기준선을 head 개수 전 범위에서 상회하며 top-of-ranking 검색 성능을 개선한다. 특히 서빙과 일치하는 방식으로 각 후보를 ‘그 후보의 best head’ 점수로 스코어링하면 추가 향상이 나타나, BACH의 train/serve 정합성이 효과적임을 보여준다. 저자들은 또한 BACH가 대부분의 head routing 설정에서 더 강한 성능을 내며, p-BACH와 v-BACH 변형도 유사한 결과를 보인다고 보고한다.



### ProjAgent: Procedural Similarity Retrieval for Repository-Level Code Generation (https://arxiv.org/abs/2607.08691)
- **Prior Approaches**: 레포지토리 수준 코드 생성은 단일 함수가 아니라 여러 파일에 흩어진 의존성과 프로젝트 관습을 함께 맞춰야 해서, 일반 코드 생성보다 훨씬 어렵습니다. 기존 맥락 검색은 주로 BM25나 dense embedding처럼 어휘/표면 의미 유사도에 의존해왔고, 호출 의존성이 없더라도 비슷한 절차를 공유하는 함수는 놓치는 경우가 많았습니다. 특히 표면 유사도만으로는 오히려 생성 품질을 떨어뜨릴 수 있다는 문제도 보고돼, 더 적절한 검색 신호가 필요해졌습니다.

- **Core Contribution**: ProjAgent는 레포지토리 수준 코드 생성에서 ‘procedural similarity(절차적 유사성)’를 명시적인 검색 신호로 도입합니다. 목표 함수를 중간의 reasoning step으로 분해한 뒤, 각 step마다 유사한 절차를 수행하는 저장소 함수들을 agentic workflow로 찾아 전통적인 semantic/lexical 검색과 결합해 더 풍부한 컨텍스트를 구성합니다. 또한 정적 분석 기반 피드백 loop으로 컴파일/분석 오류를 보수적으로 반복 수정합니다.

- **Technical Challenges**: 절차적 유사성을 찾으려면 이름이나 도메인이 달라도 구현 로직이 비슷한지를 표현해야 하는데, LLM hidden state를 그대로 cosine similarity로 비교하면 anisotropy 때문에 무관한 step도 유사하게 보이는 문제가 있었습니다. ProjAgent는 reasoning subspace projection(추론 관련 부분공간 투영)을 사용해 언어적 변형을 억제하고, 여기에 PCA 기반 debiasing을 더해 판별력을 높였습니다. 더 나아가 step 분해·검증 과정에서 LLM이 만든 단계가 실제 코드에 근거하는지(설명-본문 일치, snippet 존재, NL-코드 임베딩 유사도) 여러 단계로 보증해 검색 오염을 줄였습니다.

- **Empirical Impact**: REPOCOD 평가에서 ProjAgent는 Pass@1 41.14%를 달성하며, retrieval 기반 기존 베이스라인을 성능적으로 앞섰습니다. 결과는 절차적 유사성이 기존 어휘/의미 중심 검색 차원에서는 포착되지 않던 유용한 컨텍스트를 제공할 수 있음을 실증합니다. 코드 생성 품질을 높이기 위해 단순 검색을 넘어 단계별 절차 신호와 정적 분석 기반 반복 수정을 함께 설계했다는 점에서, 레포지토리 수준 SE 연구에 영향이 큽니다.



### ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR Post-Correction for Historical Documents (https://arxiv.org/abs/2607.08143)
Comments:
          17 pages

- **Prior Approaches**: 기존 OCR post-correction 연구는 규칙 기반, 통계 모델, 시퀀스-투-시퀀스 신경망 등으로 이어졌지만, 언어·문서 유형·잡음 특성에 따라 성능이 크게 흔들려 왔습니다. LLM을 활용한 시도도 zero-shot 프롬프트부터 fine-tuning, 탐지-생성 하이브리드까지 다양했지만 실험 설정이 달라 직접 비교가 어렵고, 무엇보다 hallucination(그럴듯한 내용 덧붙이기) 위험이 충분히 정리되지 않았습니다.

- **Core Contribution**: HIPE-OCRepair-2026은 LLM-assisted OCR post-correction을 ICDAR 경진대회 형태로 재정의하고, 재현 가능한 평가 체계를 제공합니다. 또한 언어(영어·프랑스어·독일어)와 시대(17~20세기), 문서 유형(신문·인쇄물)에 걸친 통합 멀티링구얼 벤치마크(HIPE-OCRepair-2026 dataset)를 제공해, 서로 다른 기존 데이터셋을 통일된 분할·전사 지침으로 묶었습니다. 평가 관점도 문자 생성 그 자체보다 검색·접근에 유리한 언어적 정확도를 우선하는 retrieval-oriented 점수 체계를 채택했습니다.

- **Technical Challenges**: 핵심 난제는 이미지 없이 OCR 텍스트만 주어졌을 때 심각한 왜곡을 복원해야 하며, 동시에 원문에 없는 내용을 새로 만들지 말아야 한다는 점입니다. 논문은 이를 해결하기 위해 cMER 기반 정량 지표(삽입이 분모에 반영되어 과생성에 덜 민감)와 항목 단위 선호 점수(pref_score)를 함께 사용하고, 레이아웃 정규화·IR 스타일 정규화로 검색 시나리오에 맞춘 공정한 비교를 구성했습니다. 참가 팀들은 보수적 온도, 출력 길이/형식 제약, 문서 메타데이터 활용, judge-and-retry 같은 재시도 루프 등으로 hallucination과 과교정을 제어하려 했습니다.

- **Empirical Impact**: 실험 결과 modern LLM-assisted 시스템은 전반적으로 OCR 품질을 유의미하게 개선하지만, 데이터셋·언어·잡음 수준에 따라 격차가 크며 특히 저잡음 입력에서 over-correction이 반복되는 문제가 관찰됐습니다. 최고 성능은 BnF-Mistral 계열이 차지했으며, 독일어·영어·프랑스어 모든 공식 테스트셋에서 1위를 기록했고 기준 no-correction 대비 cMER가 전반적으로 크게 감소했습니다. 또한 제출물과 데이터, scorer, 평가 파이프라인을 공개해 향후 연구자들이 동일한 평가 틀에서 시스템을 체계적으로 비교·개선할 수 있도록 했다는 점에서 의미가 큽니다.



### Beware What You Autocomplete: Forensic Attribution of Backdoored Code Completions (https://arxiv.org/abs/2607.08011)
Comments:
          To appear in COLM 2026

- **Prior Approaches**: 코드 완성 모델에서의 backdoor 공격은 fine-tuning 데이터에 트리거-페이로드를 심어, 특정 문맥에서만 악성 코드 패턴을 유도한다는 점에서 소프트웨어 신뢰성을 크게 해친다. 기존 방어는 static analysis나 anomaly detection 같은 사전 예방 성격이 강하지만, 기능을 보존하면서 삽입 로직을 숨기는 adaptive/stealthy 공격에는 취약하다는 한계가 반복적으로 드러났다. 또한 poisoning forensics 연구는 주로 gradient나 retrieval 컨텍스트를 전제로 하거나, 코드 완성 환경에는 잘 전이되지 않는 방식이 많았다.

- **Core Contribution**: 이 논문은 “누가 버그를 심었는가”라는 post-attack forensics 질문에 답하기 위해 CodeTracer를 제안한다. 주어진 miscompletion event(프롬프트와 백도어 완성)와 fine-tuning corpus만으로, 악성 완성을 유발한 특정 오염 데이터 예시(또는 코드 조각)를 추적·귀속하는 프레임워크를 만든 것이 핵심 기여다. 특히 gradient/attacker 정보가 없는 현실 제약을 그대로 반영해 실사용 가능성을 높였다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) gradient 기반 영향도 계산이 불가능하고 (2) 수백만 규모 fine-tuning 데이터 전체를 예시 단위로 비교하는 것이 비현실적이며 (3) 기존 인스턴스/LLM 포렌식 기법이 코드 완성에는 잘 맞지 않는다는 점이다. CodeTracer는 이를 위해 악성 완성에서 구조적·의미적 정규성을 뽑아 structured behavioral fingerprint를 만들고, 이를 바탕으로 코드-to-code retrieval로 후보 범위를 top-KK로 좁힌 뒤, 외부 LLM 추론으로 후보 예시의 “동일한 불안전 논리” 여부를 [Label: Yes]로 판정한다. 즉 fingerprint extraction–scope narrowing–attribution analysis의 3단계를 통해 대규모 검색 비용과 귀속 정밀도 문제를 동시에 공략한다.

- **Empirical Impact**: 평가에서는 3가지 취약 시나리오(jinja2, requests, socket)와 10종 backdoor 공격(기본 8종+적응형 2종)을 포함해, CodeTracer가 경쟁 baselines 대비 forensic 정확도가 높고 false identification(낮은 FPR)도 유지됨을 보인다. 또한 포렌식으로 추적된 오염 예시를 제거했을 때 backdoor로 인한 공격 성공률(ASR)이 0.03 이하로 크게 낮아져, 단순 탐지보다 “후속 제거/진단”에 직접 유용함을 입증한다. 런타임과 비용도 짧은 편(예: 한 건 추적 수십 초, 악성 완성당 $0.33 수준)이라 적응형 공격 환경에서도 실전 적용 여지가 크다는 점이 강조된다.



### Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems (https://arxiv.org/abs/2607.07989)
Comments:
          To appear in COLM 2026

- **Prior Approaches**: LLM 기반 멀티에이전트 시스템의 오류는 에이전트 간 상호작용이 길어지면서 원인 에이전트와 ‘최초로 돌이킬 수 없게 만든 단계’를 특정하기 어렵다. 기존 실패 로컬라이제이션은 AgentTracer처럼 counterfactual replay로 인과를 보려 하지만, 다중 에이전트 환경에서는 한 단계 변경이 이후 프롬프트·툴·조정 흐름까지 흔들어 결론이 흔들릴 수 있다. 또 AEGIS처럼 사전 정의된 오류 패턴(템플릿/택소노미)에 매칭하는 방식은 예측 불가능한 emergent reasoning이나 조정 붕괴를 충분히 포착하지 못하고, poisoning forensics 계열도 멀티에이전트의 점진적 오인·coordination drift 특성에 맞지 않아 성능이 제한된다.

- **Core Contribution**: 이 논문은 실패를 특정 에이전트뿐 아니라, 궤적이 최초로 decisively misdirected 되는 ‘가장 이른 단계’까지 함께 귀속(attribution)하는 문제를 정식화하고, 이를 위한 프레임워크 AgentLocate를 제안한다. AgentLocate는 LLM 기반 Judge가 (responsible agent, earliest decisive step) 가설을 먼저 내고, 이를 검증 가능한 절차로 다단계화해 디버깅에서 재현성과 신뢰도를 높이는 방향을 택한다. 또한 Judge의 1회 출력이 아니라 다수 평가자 판단과 피드백을 수집해, 판단 품질을 lightweight fine-tuning으로 지속 개선한다.

- **Technical Challenges**: 핵심 기술 난제는 long-horizon, tool-mediated, tightly coupled 상호작용에서 ‘어떤 에이전트의 어떤 순간’이 전역 실패를 되돌릴 수 있는지 인과적으로 분리하는 것이다. AgentLocate는 먼저 all-at-once 또는 step-by-step으로 궤적을 해석해 counterfactual reversal 조건을 근사하는 가설을 만들고, 그 가설에 대해 base/concise/evidence-focused처럼 서로 다른 스타일의 independent Evaluators가 동일 궤적을 재분석하도록 한다. 이어 각 평가자의 self-reported confidence를 반영한 confidence-aware voting으로 후보 위치를 집계하고, Judgeft 학습에는 evaluator의 비판·불일치를 LoRA 기반(parameter-efficient) fine-tuning 신호로 사용해 향후 인과 정렬을 강화한다.

- **Empirical Impact**: AgentLocate는 Who&When 및 Aegis-Bench의 두 벤치마크에서 에이전트 수준 정확도뿐 아니라, Who&When에서는 failure step까지 포함해 기존 failure localization 방법을 일관되게 능가한다. 또한 토큰 사용량과 실행 시간 측면에서 효율성을 유지하면서, 단일 Judge만 쓰는 경우보다 로컬라이제이션 정밀도가 개선됨을 보인다. 멀티에이전트 신뢰성 분석과 디버깅 자동화를 한 단계 진전시켜, 시스템 수준 장애의 원인 추적을 more verifiable하게 만드는 데 의미가 있다.



New uploads on arXiv(cs.CV)

### Wat3R: Underwater 3D Geometry Learning without Annotations (https://arxiv.org/abs/2607.08772)
Comments:
          Accepted to ECCV 2026. The dataset and code are available at this https URL

- **Prior Approaches**: 기존 수중 3D 기하 추정은 다중 시점 기하(특징 매칭, 에피폴라 기하, 전역 보정)처럼 최적화 기반 파이프라인에 의존해 계산 비용이 크고, 수중의 감쇠·산란으로 인해 강건성이 떨어질 수 있습니다. 최근에는 DUSt3R, VGGT 같은 feed-forward 모델이 각광받지만, 대규모 라벨(카메라 포즈·depth)로 학습된 모델을 수중에 그대로 옮기면 도메인 차이로 성능이 급락합니다. 또한 수중에서는 고품질 3D 애노테이션을 확보하기 어려워, 데이터 집약형 학습이 사실상 막힌다는 한계가 반복됩니다.

- **Core Contribution**: 이 논문은 Wat3R을 제안하며, 수중 3D 애노테이션 없이도 VGGT급 feed-forward 3D 재구성 모델을 수중으로 적응시키는 cross-domain semi-supervised 학습 틀을 제공합니다. teacher-student(Mean Teacher) 구조에서 teacher가 안정적인 pseudo supervision을 만들고, student는 라벨(합성)과 무라벨(실제 영상)을 함께 학습해 기하 표현을 견고하게 만듭니다. 물속에서 발생하는 정보 열화를 보완하기 위해 cross-view consistency loss를 도입하고, 평가용으로 Water3D 데이터셋까지 구축했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 수중 3D 라벨이 거의 없을 때도 기하를 어떻게 학습하느냐, (2) 감쇠·산란으로 한 시점의 관측 정보가 크게 훼손될 때 어떻게 학습 신호를 안정화하느냐입니다. 이들은 VGGT의 기존 학습 데이터에 대해 수중 감쇠·역산란을 포함한 물리 기반 합성 열화를 적용해 라벨 신호를 만들고, 실제 무라벨 영상에는 EMA 기반 Mean Teacher로 pseudo supervision을 제공하는 방식으로 해결합니다. 또한 다른 시점에서 투영·재투영한 기하 단서를 현재 시점의 학습에 연결하되, k-means 기반 foreground/static mask로 신뢰 가능한 픽셀만 선택해 동적 요소나 심한 탁도에서 오는 불안정한 제약을 억제합니다.

- **Empirical Impact**: 실험에서 Wat3R은 수중 multi-view depth estimation과 point cloud(점 맵) 복원에서 최신 기법들을 능가하며, 특히 가시성이 나쁜 구간에서도 상대적으로 더 완만하게 성능이 하락하는 강건성을 보였습니다. Sea-thru, FLSea-Stereo 등 공개 벤치마크와 함께 새로 만든 Water3D를 통해, 포즈 추정에서도 cross-view 기반 학습이 수중 열화에 대한 민감도를 낮춰 더 안정적인 추정을 돕는다는 점을 확인했습니다. UIE(수중 화질 향상 후 3D 추론)는 멀티뷰 기하 성능에 큰 이득이 제한적인 반면, Wat3R은 훈련 과정에서 수중-특화 기하를 직접 내재화해 전반적 성능과 견고함을 동시에 끌어올린다는 점에서 의미가 큽니다.



### ZipDepth: Bringing Lightweight Zero-Shot Monocular Depth Anywhere, on Any Devic (https://arxiv.org/abs/2607.08771)
Comments:
          ECCV 2026. Code: this https URL - Project page: this https URL

- **Prior Approaches**: 기존 단안 깊이 추정은 지도학습(CNN 기반)과 자기지도학습(광도/기하 일관성)로 발전했지만, 대체로 단일 데이터셋·센서·장면 범위에 묶여 도메인 이동(domain shift)에서 성능이 급격히 무너지는 한계가 있었다. 반면 foundation model 기반 방법은 indoor·outdoor·합성 장면 전반에서 zero-shot 일반화가 강하지만, 수억 파라미터·막대한 GFLOPs와 긴 학습비용 탓에 임베디드/모바일 배포가 사실상 불가능했다.

- **Core Contribution**: ZipDepth는 경량 단안 깊이 네트워크가 zero-shot 교차도메인 일반화와 실시간 배포 효율을 동시에 달성하도록 설계된 모델이다. Depth Anything v2-Large 같은 foundation model의 대규모 지식을 17개 이종 도메인(총 1,410만 장)에서 knowledge distillation으로 6.1M 파라미터 네트워크에 이식해, 효율과 정확도의 양립 문제를 정면으로 다룬다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 작은 파라미터 예산에서 의미 정보와 깊이 경계(불연속, 객체 경계)를 충분히 보존하는 것과 (2) 배포 제약(TensorRT·CoreML·ONNX Runtime/모바일 NPU의 연산 지원)에 맞춰 end-to-end로 내보낼 수 있는 업샘플링을 구현하는 것이다. ZipDepth는 reparameterizable convolution으로 학습-추론을 분리해 연산을 단순화하고, hardware-adaptive convex upsampling 또는 모바일용 보간+학습 게이트 경로를 제공해 경계 품질을 유지하면서도 그래프 수정 없이 export 가능하도록 해결한다.

- **Empirical Impact**: 실험에서는 5개 zero-shot 벤치마크에서 ZipDepth가 경량 모델 중 전반적으로 1~2위를 기록하며, 특히 NYUv2(AbsRel 8.4)와 ScanNet(8.8), DIODE의 높은 δ1 성능을 보였다. 또한 Jetson Orin NX(15W)에서 6.1M 파라미터·3.0 GMAC로 34.4 FPS(프레임당 약 397 mJ)를 달성해, foundation model 대비 에너지 효율을 크게 끌어올리면서 정확도 격차는 줄이는 방향의 실증적 진전을 보여준다.



### LongE2V: Long-Horizon Event-based Video Reconstruction, Prediction, and Frame Interpolation with Video Diffusion Models (https://arxiv.org/abs/2607.08770)
Comments:
          SIGGRAPH 2026. Project page: this https URL

- **Prior Approaches**: 이 분야의 기존 복원/예측은 CNN이나 RNN 기반(예: E2VID, FireNet)과 이후 Transformer/Hypernetwork로 발전했지만, 회귀 계열은 평균으로 수렴하는 경향 때문에 질감이 흐려진다는 한계가 있었습니다. 생성 모델로 diffusion을 쓰는 시도도 있었으나, 장기 예측에서는 누적 오차와 temporal drift로 불안정해지며, 보간(EVFI)에서는 빠르고 복잡한 움직임에서 고스트/블러 같은 아티팩트가 자주 나타납니다. 또한 작업별로 별도 아키텍처를 요구하는 경우가 많아 유연성이 떨어졌습니다.

- **Core Contribution**: LongE2V는 pre-trained video diffusion prior를 활용해 이벤트 기반 video reconstruction, prediction, frame interpolation을 하나의 프레임워크로 함께 수행하도록 설계했습니다. CogVideoX를 기반으로 이벤트 voxel을 조건으로 넣어 fine-tuning함으로써 데이터 효율과 지각 품질을 동시에 끌어올렸고, 매우 긴 시퀀스에서도 안정적인 temporal coherence를 노립니다. 더불어 Reencoding Alignment와 Cross Residual Correction으로 양방향 보간에서의 bidirectional consistency를 정밀하게 맞추는 접근을 제안합니다.

- **Technical Challenges**: 핵심 기술 난제는 이벤트 스트림이 희소하고(세기 정보가 없음) 문제 자체가 ill-posed라는 점, 그리고 장기 생성에서 재귀적으로 오차가 누적되어 드리프트가 발생한다는 점입니다. LongE2V는 Autoregressive Unrolling으로 학습-추론 간 도메인 갭을 줄이고, Adaptive Context Switching로 컨텍스트 토큰의 주의도에 따라 유지/갱신을 동적으로 결정해 drift를 완화합니다. 보간의 경우 3D VAE latent의 temporal misalignment을 Reencoding Alignment로 교정하고, 정보 손실을 Cross Residual Correction(상호 잔차 보정)으로 보완하며, Event Voxel Density Augmentation으로 센서 해상도 변화에도 강건하도록 학습합니다.

- **Empirical Impact**: BS-ERGB/EVREAL 및 ECD·MVSEC·HQF 등 실세계 벤치마크에서 reconstruction·prediction·frame interpolation 전 작업에 걸쳐 SOTA를 일관되게 능가하는 결과를 보였습니다. 특히 매우 긴 시퀀스에서의 오차 누적이 크게 줄고, 보간에서도 temporal coherence가 우수하다는 점이 강조됩니다. 또한 fine-tuning 없이 EVFI에 확장해 zero-shot으로도 전용 EVFI 방법들을 추월했으며, 구성요소별 애블레이션을 통해 각 모듈이 안정성과 품질 향상에 필수임을 실증했습니다.



### Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction (https://arxiv.org/abs/2607.08769)
Comments:
          Project Webpage: this https URL

- **Prior Approaches**: 기존 3D Gaussian Splatting(3DGS)과 대규모 확장 방법들은 주로 핀홀 카메라의 국소 가시성(프러스텀)에 기대어 분할 최적화를 설계해 왔습니다. 하지만 ERP(equirectangular projection) 기반 360도 파노라마에서는 모든 블록이 여러 뷰에서 동시에 관측되는 ‘전역적 가시성’이 나타나, 기존 분할이 차별성을 잃고 전역 학습으로 퇴화합니다. 또한 파노라마용 3DGS는 ERP 기하 왜곡 완화에는 강점이 있지만, 하늘·저텍스처 등으로 인해 스케일을 키우기 위한 공간 관리 체계는 부족하다는 한계가 지적됩니다.

- **Core Contribution**: 본 논문은 대규모 야외 파노라마 3DGS 재구성을 위한 2단계 coarse-to-fine 프레임워크 PanoLOG를 제안합니다. 핵심은 Geometry and Gradient-based Partitioning Strategy(G2PS)로, (1) parallax 기반 불확실성으로 재구성 가능 영역을 안정화해 분할을 설계하고, (2) coarse 단계의 gradient를 활용해 카메라–블록 할당을 중요도에 따라 배치합니다. 이를 통해 360도 관측에서 발생하는 분할 붕괴를 막고 블록별 최적화가 제대로 작동하도록 만듭니다.

- **Technical Challenges**: 가장 큰 기술 문제는 ERP에서 국소 프러스텀 제약이 사라져 분할이 ‘언제나 맞는 카메라 집합’을 찾기 어렵다는 점입니다. 이를 위해 PanoLOG는 Stage I에서 sky-sphere 모델링과 panoramic monocular depth supervision을 넣어 기하를 신뢰 가능하게 만든 뒤, Stage II에서 블록 내부에 대한 rendering loss gradient 크기를 요약해 카메라의 실제 기여를 점수화합니다. 동시에 parallax-driven depth uncertainty로 AABB 경계를 적응적으로 확장해 무한 스카이/원거리 영역을 다루며, opacity reset과 pruning으로 블록별 유효 원시를 정리해 중복·부유( floater ) 문제도 완화합니다.

- **Empirical Impact**: 저자들은 야외 파노라마 재구성을 위한 최초 대규모 벤치마크 Pano360을 구성(4개 환경, 5,637장, 총 2,000,000m² 규모)하고 공용 데이터셋 Ricoh360·360Roam에서도 검증했습니다. 결과적으로 G2PS 기반 PanoLOG는 파노라마/대규모 3DGS 계열 기준선 대비 렌더링 품질에서 우수(예: PSNR/SSIM/LPIPS) 성과를 보이며, 강한 기준선 대비 모델 크기는 2.2~7.5배 수준으로 줄였습니다. 또한 분할 관련 구성요소(gradient 기반 할당, sky sphere, depth supervision)를 제거한 ablation에서 성능 저하가 확인돼, 제안 전략이 스케일러빌리티와 품질을 동시에 좌우함을 실증했습니다.



### OPSD-V: On-Policy Self-Distillation for Post-Training Few-Step Autoregressive Video Generators (https://arxiv.org/abs/2607.08766)
Comments:
          Project page: this https URL ; Code: this https URL

- **Prior Approaches**: few-step AR 비디오 생성은 낮은 지연으로 긴 영상을 만들 수 있지만, 긴 롤아웃에서 오류 누적과 모션 역학 약화가 생깁니다. 기존 DMD-style few-step 증류나 self-forcing 류 학습은 주로 짧은 클립 교사나 롤아웃/휴리스틱 수준의 감독에 의존해, 진짜 긴 autoregressive 궤적 전체에 대한 촘촘한 denoising 보정 신호가 부족하다는 한계가 있습니다.

- **Core Contribution**: OPSD-V는 post-training에서 on-policy self-distillation을 “캐시-aware”하게 확장해, 긴 롤아웃의 품질 저하를 줄이면서 원래 few-step inference 경로는 그대로 유지합니다. 핵심은 real long video를 학생의 출력 타깃이 아니라 “시간적 컨텍스트(teacher 캐시 구성용)”로 쓰되, 학생은 배포 시와 동일하게 자신의 KV cache 궤적을 따라가며 학습한다는 점입니다.

- **Technical Challenges**: 가장 큰 문제는 real long video를 그대로 reconstruction/teacher-forcing으로 쓰면, 학생 롤아웃과 시간·의미가 어긋나 학습이 망가질 수 있다는 비정합입니다. OPSD-V는 해결책으로 (1) 학생은 self-generated KV cache로 원래 sampler와 denoising 스텝 수를 유지한 채 진행하고, (2) 교사는 학생이 방문한 동일한 denoising 상태에서 “과거 캐시만 real-video 컨텍스트로 치환(최근 chunk는 유지)”하는 방식으로 더 깨끗한 corrective target을 제공하게 했습니다.

- **Empirical Impact**: Self-Forcing과 LongLive 같은 대표적 few-step AR 비디오 모델에 LoRA continued training으로 적용했을 때, 시각 품질과 motion dynamics가 일관되게 개선되고 VBenchLong 점수도 향상됐습니다. 10명의 사용자 대상 비교(20개 비디오 페어)에서는 OPSD-V가 베이스 모델보다 66.0%의 전체 선호도에서 선택됐고, 무승부 제외 시 82.5%까지 선호가 높았습니다.



### Enhancing In-context Panoramic Generation via Geometric-aware Pretraining (https://arxiv.org/abs/2607.08765)
- **Prior Approaches**: 파노라마용 텍스트-투-이미지와 인컨텍스트 편집이 빠르게 발전했지만, equirectangular projection(ERP)은 위도에 따른 왜곡이 커서 기하학적으로 일관된 편집을 어렵게 만든다. 기존 방법은 cube-map 기반 수정이나 3D spherical positional embeddings 같은 distortion-aware 설계를 넣지만, ERP에서 3D 장면 구조의 일관성까지 충분히 보장하지 못한다. 또한 심도 같은 기하 제약을 도입하려 해도, 평면 이미지의 Z-축 깊이와 달리 파노라마는 카메라 중심으로부터의 구면(방사) 거리로 정의된다는 점에서 정교한 형식화가 부족했다.

- **Core Contribution**: Canvas360은 인컨텍스트 파노라마 생성을 두 단계로 나눠, geometry-aware 사전학습과 downstream 태스크별 fine-tuning을 결합한 통합 프레임워크다. 사전학습에서는 RGB-깊이 쌍을 활용해 파노라마의 기하 인식을 학습하고, fine-tuning에서는 depth를 제거한 채 style transfer, inpainting, outpainting, editing을 한 모델에 통합해 처리한다. 특히 토큰 레벨 concatenation으로 다양한 컨텍스트 입력을 하나의 포맷으로 묶어, 태스크 커버리지와 모델 유연성을 동시에 노린다.

- **Technical Challenges**: 가장 큰 난제는 인컨텍스트 파노라마에 맞춘 대규모 고품질 데이터의 부족과, 구면 기하에서의 depth priors를 효과적으로 모델에 반영하는 것이다. Canvas360은 Canvas360Dataset(총 1M 샘플)을 제안해 4개 태스크(outpainting/inpainting/style transfer/editing)에 대한 paired supervision을 확보하고, 사전학습에서는 parallel depth generation과 유사도(simmilarity) 손실로 RGB-깊이의 표현 붕괴를 막는다. 동시에 velocity circular padding으로 0°/360° 경계의 구면 연속성을 직접적으로 보이게 하여 경계 일관성과 지구적 coherence를 강화한다.

- **Empirical Impact**: 실험 결과 Canvas360은 파노라마 특화 FAED에서 최상 성능을 보이며, IS·QA aesthetic·NIQE 등에서도 우수한 점수를 기록해 세부 충실도와 지각 품질의 균형을 입증했다. 인컨텍스트 태스크에서도 blur나 아티팩트가 줄고, 편집 시에도 올바른 파노라마 왜곡을 반영해 geometry-inconsistent 편집 문제를 개선했다. 사용자 연구에서도 경계 연속성(boundary continuity), 파노라마 인지(panorama awareness), 전체 품질에서 선호도가 가장 높아 실제 사용자 관점의 효과를 뒷받침한다.



### OpenCoF: Learning to Reason Through Video Generation (https://arxiv.org/abs/2607.08763)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 비디오 생성에서 Chain-of-Thought(변증/추론 단계)를 정교화하려는 시도는 주로 정적 이미지에 의존하거나 외부 도구·보조 생성·로컬 증거 추출 같은 우회 경로에 머물렀다. 최근에는 temporally connected frames로 추론을 펼치는 Chain-of-Frame(CoF) 개념이 제안됐지만, 일반 비디오 데이터로만 학습돼 CoF 전용의 다양한 감독과 아키텍처 설계가 부족하다는 한계가 있었다. 또한 벤치마크 구축은 활발했지만, 내부적으로 CoF 추론 상태를 구조적으로 강화하는 연구는 상대적으로 드물었다.

- **Core Contribution**: 이 논문은 CoF 추론을 겨냥한 공개 프레임워크 OpenCoF를 제안하며, 11개 태스크 패밀리의 추론 비디오 17,312개로 구성된 OpenCoF-17K 데이터셋을 공개한다. 이어서 Wan2.2-I2V-A14B를 OpenCoF-17K로 fine-tuning해 Wan-CoF를 만들고, reasoning 전용 기법 없이도 외부 4개 벤치마크에서 기준선 대비 유의미한 성능 향상을 확인했다. 마지막으로 CoF의 중간 추론 상태를 조직하기 위해 Visual Reasoning Tokens(vt)와 Textual Reasoning Tokens(tt)라는 두 종류의 reasoning-token 메커니즘을 도입하고 비교한다.

- **Technical Challenges**: 핵심 기술적 난제는 비디오 생성 모델이 원래 이미지/영상 합성 중심으로 설계돼 있어, 프레임 간 논리적 결과·장기 일관성·공간/물리/추론 연속성을 ‘중간 상태’로 명시적으로 저장하기 어렵다는 점이다. 논문은 이 문제를 두 계층으로 나눠 해결하려고 하는데, vt는 시각 latent 시퀀스에 reasoning 상태를 위한 학습 가능한 토큰을 삽입해 저수준 시각 단서를 구조화하고, tt는 텍스트 조건에 prompt-independent한 학습 우선순위를 추가해 고수준 의미/규칙을 보강한다. 또한 어떤 단계(모델 depth, denoising step, 공간, 시간)에서 토큰이 기여하는지 attention 분석으로 세분화해, 둘이 서로 보완적으로 시간 축과 경계 상태를 다룬다는 관찰을 제시한다.

- **Empirical Impact**: Wan-CoF는 4개 비디오 추론 벤치마크에서 Wan2.2-I2V-A14B 대비 전반적으로 개선되며, 예를 들어 MME-CoF Overall, Gen-ViRe, VIPER, RULER-Bench에서 모두 상승(Noise 대비 추론 하위 차원 중심)했다. 또한 토큰 추가 실험에서 vt와 tt는 벤치마크별로 성격이 다른 강점을 보이며, vt는 planning·visual stability 같은 시각-시간적 유지가 중요한 항목에서, tt는 instruction alignment·structural 같은 텍스트 기반 정렬/구조 항목에서 더 두드러졌다. 결과적으로 OpenCoF-17K의 넓은 temporal supervision과 중간 reasoning state를 위한 명시적 메커니즘이 함께 필요하다는 결론을 뒷받침하며, 데이터셋·모델·코드를 오픈소스로 공개해 후속 연구를 촉진한다.



### WaspMOT: A Benchmark for Long-Term Multi-Object Tracking of Trichogramma Wasps (https://arxiv.org/abs/2607.08729)
- **Prior Approaches**: 기존 MOT 벤치마크는 보통 수십 초 수준의 단편 시퀀스에 치우쳐 있어, 오클루전이나 외형 변화 같은 단기 연속성 문제는 평가하지만 장기 동일 인식(identiy preservation)을 충분히 검증하기 어렵다. 트래킹-by-detection 계열(ByteTrack, BoT-SORT 등)은 모션 연속성·공간 중첩(IoU) 가정과 일부 appearance 단서를 활용하지만, 갑작스런 점프나 일시 소실이 잦은 환경에서는 궤적이 잘게 끊어지는 한계가 있다. 생태/곤충 데이터도 일부 존재하지만, 대개는 부분 구간만 커버해 수천 프레임 단위의 장기 ID 유지의 어려움을 드러내기 부족했다.

- **Core Contribution**: 논문은 장기 신원 보존을 정면으로 겨냥한 WaspMOT 벤치마크를 제안한다. 실험실 생태 환경에서 Trichogramma wasp를 25 FPS로 약 8분(약 12,000프레임)씩 촬영한 10개 시퀀스를 제공하며, 모든 개체가 끝까지 등장하는 closed-set 조건이라 수천 프레임에 걸친 ID 일관성 평가가 가능하다. 또한 oracle detections(ground truth 기반, MOTChallenge 포맷)를 제공해 검출 오류를 제거하고 association 성능만 분리 측정하도록 설계했다.

- **Technical Challenges**: WaspMOT의 핵심 기술 난제는 (1) 갑작스런 점프 이벤트로 인한 큰 위치 변화와 일시적 미검출, (2) 3D 실험 구조와 개체 상호작용에 따른 오클루전, (3) 곤충 특성상 외형이 매우 유사해 appearance 기반 구분이 약하다는 점이다. 저자들은 이러한 상황에서 기존 tracking-by-detection 모델들이 모션 연속성 가정 때문에 장기 구간에서 identity fragmentation을 반복한다는 점을 벤치마크로 계량화한다. 더 나아가 끊긴 tracklet을 위치·시간 일관성으로 다시 잇는 단순 spatial tracklet stitching baseline을 도입해, 로컬 연관 한계가 얼마나 복구 가능한지 분석했다.

- **Empirical Impact**: ByteTrack, BoT-SORT, C-BIoU, OC-SORT, McByte를 oracle detections 하에 동일 프로토콜로 평가한 결과, 모든 방법이 낮은 IDF1을 보이며 장기 ID 유지가 매우 어렵다는 공통된 결론이 나왔다. 특히 IDF1 개선 폭이 커 spatial stitching만으로도 성능이 일관되게 상승했는데, 이는 상당 부분의 fragmentation이 간단한 재연결 규칙으로는 복구될 수 있음을 뜻한다. 최고의 성능은 McByte로, temporally propagated segmentation masks의 기하 제약이 겹침 같은 애매 구간을 완화했지만, 그럼에도 완전한 ID 일관성은 여전히 달성되지 못했다. WaspMOT는 기존 단기 벤치마크에서는 잘 드러나지 않는 최신 트래커의 실패 모드를 보여주며, 장기 association 연구에 직접적인 기준점과 동기부여를 제공한다.



### Pose-to-Biomechanics: Bridging 3D Human Pose Estimation and Biomechanical Attribute Prediction (https://arxiv.org/abs/2607.08725)
Comments:
          23 pages, 2 figures

- **Prior Approaches**: 3D 인체 포즈 추정은 기하학적 키포인트 정확도(MPJPE 등)에 맞춰 최적화돼, 복구 결과를 물리적으로 해석 가능한 생체역학 지표(토크, 지면반력, 근활성 등)로 연결하는 데는 공백이 있었다. OpenCap 같은 마커리스 파이프라인은 근골격 모델과 시뮬레이션/역다이내믹스를 결합하지만, 다중 뷰 캘리브레이션이나 대상별 최적화가 필요해 확장성이 제한된다. 또한 바디 모델·메시 복원은 해부학적 사실성은 높이지만, 기존 3D 포즈 추정기의 17-joint 출력을 범용 생체역학 속성 공간으로 일괄 변환하는 플러그인형 매핑 메커니즘은 부족했다.

- **Core Contribution**: 이 논문은 BioModule을 제안한다. BioModule은 어떤 3D 포즈 estimator 뒤에 그대로 붙일 수 있는 lightweight temporal transformer 플러그인으로, 표준 17-joint 3D skeleton 입력만으로 3티어(kinematic/kinetic/neuromuscular)로 구성된 17개 생체역학 속성을 예측한다.

- **Technical Challenges**: 핵심 과제는 서로 다른 좌표계의 해부학적 대응을 frame-accurate하게 맞추는 것이었다. 이를 위해 Human3.6Mplus를 구축해 Human3.6M의 17-joint와 OpenSim 기반 musculoskeletal label을 펠비스 루트를 앵커로 정렬·기하학적 검증(2D 투영이 sub-pixel, 펠비스 마커가 machine precision 수준 일치)을 수행하고, 이 정렬된 supervision으로 BioModule을 학습시켰다. 또한 생체역학 속성들은 차원·잡음·불확실성이 달라 단순 합 손실이 편향될 수 있어, kinematic/kinetic/neuromuscular 3그룹에 tiered weighted multi-task loss를 적용해 안정적으로 학습한다.

- **Empirical Impact**: BioModule은 7개 state-of-the-art 3D 포즈 추정기 각각의 출력에 대해 downstream 생체역학 예측 품질을 체계적으로 비교하도록 벤치마크를 구성했다. 이를 통해 상류 포즈 추정기의 정확도/아키텍처 품질이 하류 생체역학 계층별 예측 충실도로 어떻게 전파되는지 처음으로 정량 분석한다. 결과적으로 BioModule은 비전 기반 3D 포즈 추정에서 물리적으로 해석 가능한 인간 모션 분석으로 이어지는 소형·모듈형 브리지로 자리잡을 가능성을 보여준다.



### LTM: Large-scale Terrain Model for Wildfire-prone Landscapes (https://arxiv.org/abs/2607.08711)
- **Prior Approaches**: 기존 3D 복원은 도시 환경을 중심으로 발전해 point cloud·mesh·voxel, 또는 feature matching 기반 MVS/SLAM을 주로 사용한다. 그러나 산림·초지·관목처럼 대비가 낮고 반복 패턴이 많으며 시각적 특징이 불안정한 자연 지형에서는 키포인트 매칭과 2D-3D 정합이 쉽게 깨진다. 또 NeRF 계열을 대규모 자연 장면에 그대로 적용해도 성능이 크게 떨어져, 전통적 지형 업데이트 주기(연 1회 수준) 문제를 해결하기 어렵다.

- **Core Contribution**: 이 논문은 오래된 DEM(수치표고모델)을 기하학적 prior로 삼아, posed된 지상 RGB 이미지로 대규모 야외 지형의 depth와 2D-3D 정합을 업데이트하는 멀티모달 재구성 프레임워크를 제안한다. 핵심 혁신은 DEM과 이미지 사이의 physics-based pixel-pixel alignment로, 비싼 feature matching 절차를 제거해 계산 복잡도를 크게 낮춘다는 점이다. 동시에 연동된 fuel(연료) 지도 생성을 위해 3개 식생 범주(잔디·관목·나무)를 대상으로 세그멘테이션을 수행해 재난 분석까지 이어지도록 설계했다.

- **Technical Challenges**: 어려움은 (1) 시각 특징이 희소·동적이고 (2) 이미지 중첩이 작으며 (3) 스케일 모호성과 가림(occlusion)이 생기는 야외 조건에서 안정적으로 픽셀 단위 정합을 달성해야 한다는 것이다. 해결을 위해 ray tracing 기반으로 각 이미지 픽셀의 광선을 DEM 래스터까지 투과시키고, 광선의 높이가 지형 표면 높이를 처음으로 “하회하는” 지점을 찾아 픽셀을 DEM 래스터 픽셀과 직접 대응시킨다. 또한 neural 단안 depth 추정은 DEM 제약과 픽셀 단위 샘플링 및 RANSAC 회귀로 결합해 지형의 기준선을 유지하도록 했으며, 가림이 심한 경우 성능 저하도 관찰된다.

- **Empirical Impact**: 검증을 위해 실 산불 취약지(게티 파이어 영향 구역)를 기반으로 Unreal Engine 기반 대규모 지형 시뮬레이터를 구축하고, 스마트폰 iPhone 14 Pro 영상과 동일 카메라 포즈/지오로 sim-to-real 비교를 수행했다. TopoDepth는 경쟁 one-shot 단안 depth 모델 대비 지형 전이(능선·계곡 등)를 더 일관된 스케일로 따라가며, 오차가 큰 범위(수십 미터) 수준에서 유지되는 등 대규모 지형 갱신에 필요한 안정성을 보였다고 보고한다. 최종적으로 연료 유형 분류가 3D 공간에 투영되며, 시뮬레이션에서의 식생 변화(관목/초지→침엽수 등)가 동일 위치에서 잘 식별되는 점이 end-to-end 파이프라인의 실용성을 뒷받침한다.



### HumanForge: A Human-Centric Deepfake Video Benchmark with Multi-Agent Forgery Rationales (https://arxiv.org/abs/2607.08705)
Comments:
          6 pages, 2 figures

- **Prior Approaches**: 기존 영상 딥페이크 탐지는 주로 얼굴 교체나 전역 text-to-video 합성에 치우쳐, 사람-사람·사람-물체 상호작용 같은 핵심 상호작용 차원을 놓치는 경우가 많다. 또한 정적 이미지 기반 설명형 포렌식은 시간적/운동학적/오디오-비주얼 동기 같은 동적 단서를 충분히 반영하지 못했다. 마지막으로 자동 주석은 생성 프롬프트·구동 신호 같은 메타데이터 없이 ‘블라인드’로 판단해, 의도된 편집을 합성 이상으로 오분류할 병목이 있었다.

- **Core Contribution**: 이 논문은 HumanForge라는 대규모 human-centric 비디오 포저리(딥페이크) 벤치마크를 제안하며, 오디오 구동 립싱크·포즈 구동 모션 전이·상호작용(인간-인간/인간-물체)·텍스트 기반 편집의 4개 시나리오를 18K+ 구간으로 체계화했다. 동시에 Gen2Anno(Generation-to-Annotation)로 ‘예상 상태(Expected State)’와 ‘실제 상태(Actual State)’를 대비하는 대비적(annotation) 추론을 도입해, 생성 의도에 부합하는 변화와 진짜 아티팩트를 분리한다.

- **Technical Challenges**: 핵심 난제는 (1) 생성 의도(프롬프트/참조/구동 신호)와 (2) 최종 영상에서 관측되는 시각적 차이를 함께 고려해야 한다는 점이며, 단일 VLM 검사로는 문맥 기반 판별이 어렵다. 논문은 LangGraph 기반 6개 에이전트 파이프라인으로 소스 프로파일링·프로버넌스 기록·시나리오 라우팅·MoE 기반 Expected State 구성·생성 영상만 보는 Actual State 인스펙션·Chief Judge 대비 판정을 수행하고, 모호한 관측은 self-correction 루프로 재검증해 근거 없는 라벨을 줄인다.

- **Empirical Impact**: HumanForge에서 전통적 탐지기와 LMM 기반 모델들을 폭넓게 평가한 결과, 특히 zero-shot 일반화와 fine-grained 추론에서 큰 어려움이 드러났다. 이는 사람 중심 상호작용/운동학/다중모달 정합성까지 포함한 포렌식 과제가 기존 데이터·설계의 한계를 넘어 새로운 연구 장을 열고 있음을 보여준다. 또한 omni_annotation.json 형태의 이진 결정, 세부 아티팩트 범주, 공간-시간 로컬라이제이션을 포함하는 공개 코드·데이터 릴리스를 예고한다.



### SAM-MT: Real-Time Interactive Multi-Target Video Segmentation (https://arxiv.org/abs/2607.08688)
Comments:
          ECCV 2026, Project Page: this https URL

- **Prior Approaches**: 기존 VOS는 STM 계열처럼 단일 타깃에 최적화된 파이프라인을 여러 객체에 대해 반복해 처리하는 경우가 많아, 타깃 수가 늘면 계산량과 지연이 함께 증가하는 한계가 있었습니다. SAM2를 포함한 최근 방식도 객체 단위 메모리·전파 중심이라 멀티타깃 확장 시 FPS가 급격히 떨어집니다. 일부는 프레임 레벨 특징 공유로 완화하려 했지만, 결국 마스크 디코딩과 메모리 인코딩을 타깃마다 따로 수행해 확장성이 제한됐습니다.

- **Core Contribution**: 이 논문은 SAM2를 기반으로, 다중 타깃을 실시간으로 분할하는 interactive 멀티타깃 VOS 프레임워크 SAM-MT를 제안합니다. 핵심은 모든 타깃에 대해 계산을 복제하지 않고, 전역 컨텍스트는 공유하되 타깃별 identity를 담당하는 target query를 분리해 유지하는 구조입니다. 추가로 decoupled masked attention, query-based sparse memory, occlusion과 overlap 대응 전략을 결합해 타깃 수에 따른 지연 증가를 줄입니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 타깃별 정체성을 유지하면서도 전역 컨텍스트를 공유할 때 발생하는 cross-target interference(타깃 간 간섭)를 막는 것입니다. SAM-MT는 decoupled masked attention으로 타깃 쿼리들 간의 상호작용은 차단하면서, 각 타깃이 전역 쿼리를 통해 공통 컨텍스트는 활용하도록 설계했습니다. 또한 dense 메모리를 타깃마다 복제하지 않고, 타깃별 query를 담는 lightweight FIFO sparse memory와 identity transformer로 시간적 진화를 학습하며 occlusion을 위한 strided frame sampling 및 overlap prevention(중첩 손실)을 추가해 안정성을 높였습니다.

- **Empirical Impact**: 실험에서 SAM-MT는 여러 VOS 벤치마크(MOSEv2, LVOSv2 등)에서 성능을 유지하면서 SAM2.1-B+와 동등하거나 그 이상을 보이며, 특히 장기 시나리오에서 격차를 벌렸습니다. 효율 측면에선 타깃 수에 따른 FPS 저하가 크게 완화되어, 10 targets에서도 36+ FPS로 단일 타깃에 가까운 속도를 보여줍니다. 합성 멀티타깃 스케일링과 VRAM 측정에서도 SAM-MT는 지연과 메모리 사용이 타깃 수에 덜 민감해, 밀집 장면·자주 가려지는 환경에서 실사용 가능성을 강화한다는 점이 확인됐습니다.



### Multi-Resolution Feature Stem for Diabetic Retinopathy lesion segmentation (https://arxiv.org/abs/2607.08679)
Comments:
          2026 International Conference on Advances in Artificial Intelligence and Machine Learning (AAIML), 20-22 March 2026

- **Prior Approaches**: 기존 DR 병변 분할 연구는 U-Net 계열과 DeepLabV3+처럼 단일 해상도(고정 입력 크기)를 고수하는 경우가 많아, 마이크로애뉴리즘(MA)처럼 아주 작은 병변과 출혈(HE)처럼 큰 병변이 동시에 요구하는 스케일을 다 담기 어렵다. 또한 멀티스케일을 시도하더라도(예: dual-path, feature fusion) 입력 해상도 자체가 병변 유형별 성능에 미치는 상충 효과를 체계적으로 분석하진 않는 한계가 있었다.

- **Core Contribution**: 이 논문은 입력 해상도를 512와 1024로 바꿨을 때, 병변 유형에 따라 해상도 증가가 성능을 올리기도 하고 오히려 떨어뜨리기도 하는 “해상도-의존적 상반 현상”을 실험적으로 정리한다. 그 위에 UNet++ 백본 앞단에 Multi-Resolution Feature Stem(입력 레벨 피라미드)을 붙여, 여러 스케일 정보를 동시에 처리해 작은 병변의 세밀함과 큰 병변의 문맥을 함께 확보하는 아키텍처를 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작은 MA를 위해선 고해상도가 필요하지만 (2) HE처럼 큰 병변에서는 고해상도가 오히려 혼동을 유발할 수 있다는 상반된 요구를 구조적으로 흡수하는 것이다. 이를 위해 입력을 1024/512/256 스케일의 피라미드로 만든 뒤 공유 가중치 Convolution으로 스케일-불변 특징을 뽑고, 채널을 결합·1x1로 융합한 후 UNet++ 인코더 초기에 주입하는 방식으로 트레이드오프를 완화했다.

- **Empirical Impact**: DDR 테스트셋에서 제안 모델은 U-Net++ 대비 mIoU와 mAP를 각각 개선하며 전반적 분할 성능을 끌어올렸고, 특히 MA와 EX/SE 같은 더 작은/중간 스케일 병변에서 큰 폭의 향상이 관찰됐다. 반면 HE에서는 일부 성능 하락이 나타났는데, 이는 공유 가중치 초기 추출이 혈관과의 구분을 약화시킬 수 있다는 논의로 이어진다. 또한 연산 오버헤드는 약 1.27배로 늘지만, 작은 병변 정확도가 중요한 스크리닝 응용에서 비용 대비 효용이 있다는 실용적 함의를 제시한다.



### Do Transformations Reveal the Truth? Generative Residual Learning for Generalized AI-Generated Image Detection (https://arxiv.org/abs/2607.08674)
- **Prior Approaches**: 기존 AIGI(이미지 생성형 딥페이크) 탐지는 크게 (1) 생성기별 아티팩트를 쓰는 방법과 (2) forgery에 민감한 표현 공간에 투영하는 표현학습 방법으로 나뉜다. 하지만 아티팩트 기반은 학습 때 보지 못한 생성기에 약하고, CLIP 계열 표현은 이미지별로 독립 처리해 생성 과정이 만들어내는 ‘관계 정보’를 충분히 활용하지 못한다. 그 결과 GAN·diffusion 등 빠르게 늘어나는 생성기 다양성에 대한 일반화가 불안정하다. 

- **Core Contribution**: 이 논문은 원본 이미지와 변환(복원·초해상·노이즈 제거·강화 등)된 이미지 사이의 미세한 ‘차이(생성 잔차, generative residuals)’를 명시적으로 모델링하는 GenRes를 제안한다. 또한 원본-변환 관계를 신경 텐서 네트워크(Neural Tensor Network, NTN)로 다차원 곱셈 상호작용까지 포착해, 단순 차분이나 결합을 넘어 생성 특유의 잔차 구조를 학습한다. 변환이 여러 개일 때는 GenRes++에서 cross-attention 집계를 통해 가장 정보가 큰 잔차 큐를 자동으로 가중해 일반화 성능을 끌어올린다. 

- **Technical Challenges**: 핵심 기술 난제는 ‘생성기마다 달라 보이는 아티팩트’를 넘어, 2차 생성 처리에서 드러나는 비대칭적 응답(관계 구조)을 안정적으로 학습하는 것이다. 이를 위해 논문은 PE-Core(비전 트랜스포머)를 frozen로 유지하고 LoRA로만 적응해 저비용으로 의미 있는 임베딩을 뽑으며, 원본-변환 임베딩 간 NTN의 bilinear/곱셈 상호작용으로 fine-grained relational features를 구성한다. GenRes++에서는 여러 변환 결과를 cross-attention으로 통합해 입력별로 잔차의 유효성을 스스로 선택하도록 설계했다. 

- **Empirical Impact**: UniversalFakeDetect 벤치마크의 19개 ‘미보는 생성기’ 평가에서 GenRes++는 mACC 95.7%, mAP 99.1%로 기존 방법을 전반적으로 앞섰다. 특히 단일 변환으로 학습한 GenRes도 mACC 92.6% 수준으로 유의미한 성능을 보여, 생성 잔차 신호가 강력함을 시사한다. 무엇보다 GAN·other·diffusion 전 가족에 걸쳐 비교적 일관된 성능을 보여 cross-generator 일반화 문제에 실질적인 진전을 제공했다.



### When Structured Sparse Autoencoders Learn Consistent Concepts Across Modalities (https://arxiv.org/abs/2607.08605)
- **Prior Approaches**: Sparse autoencoder(SAE)는 은닉표현을 희소한 잠재 특성으로 분해해 기계적 해석(mechanistic interpretability)에 유리하다는 점에서 주목받았다. 다만 기존 SAEs는 재구성 손실과 원소 단위 희소성만 최적화해, vision-language models(VLMs)에서는 모달리티에 맞는 개념이 시각에서 조각나게 학습되는 문제가 남는다. 그 결과 한 잠재 뉴런이 여러 의미를 동시에 가지는 잔여 polysemanticity가 생기고, 이미지 패치가 특정 개념을 활성화해도 그 패치들이 의미적으로 일관된 시각 영역으로 정렬되지 않는다.

- **Core Contribution**: 이 논문은 Structured Sparse AutoEncoder(S2AE)로, 시각 모달에서 개념이 ‘의미적·공간적 일관성’을 갖도록 구조적 희소성 정규화를 도입한다. Transformer attention 유사도와 공간 인접성을 함께 써서 이미지 패치를 개념 단위의 시각 영역으로 그룹화하고, vanilla SAE 학습 시 영역 간 분리는 exclusive sparsity로, 영역 내 일관성은 group sparsity로 강제한다. 이를 통해 개념 특성이 서로 다른 시각 영역에서는 경쟁적으로 분화하면서, 같은 영역에서는 응집된 방식으로 활성화되게 만든다.

- **Technical Challenges**: 핵심 기술 난제는 픽셀 수준 경계가 명시되지 않은 상태에서, 단일 개념에 대응하는 ‘시각적으로 일관된 영역’을 안정적으로 추정하는 것이다. 저자들은 attention 기반 유사도만으로는 의미적 outlier가 많이 생긴다는 관찰을 바탕으로, 공간 거리(패치 좌표 기반)까지 결합해 병합형 agglomerative clustering 거리 행렬을 구성하고 영역을 얻는다. 그 위에 영역-수준(group-level) 활성 프로파일을 만들어 exclusive sparsity와 group sparsity를 각각 영역 간 경쟁과 영역 내 공동 활성에 적용함으로써, 시각-언어 shared feature 공간까지 정돈되는 효과를 노린다.

- **Empirical Impact**: Qwen2.5-VL-7B-Instruct에서 S2AE는 semantic alignment(mIoU) 평균 6.06% 향상과 representational efficiency(l0 norm 60.81) 개선을 달성하면서도 Explained Variance 99%대의 재구성 충실도를 유지한다. 또한 cross-modal 분석에서 구조적 시각 우선이 단일 의미성(monosemanticity)을 높여, 두 모달리티 멀티모달 특징 모두 semantic consistency 평균 3.08%, monosemanticity 점수 평균 2.37% 향상을 보였다. 즉, 시각 측 구조 제약이 공유된 개념 사전(dictionary)에서 언어 쪽 의미 정합성까지 간접적으로 끌어올리는 메커니즘을 실증했다.



### Switch-Reasoner: Learn When to Think in Multitask Mixtures via Reinforcement Learning (https://arxiv.org/abs/2607.08572)
- **Prior Approaches**: 기존 MLLM 추론 파이프라인은 Think-then-Answer로 고정해 모든 입력에 동일한 숙고(체인오브쏘트) 비용을 강제하는 경향이 있었다. 이 방식은 어려운 문제에서는 유리하지만 이질적인 멀티태스크에서는 토큰·지연 비용이 커지고, RL post-training에서 always-thinking/always-direct로 모드 붕괴가 일어나기 쉽다. 효율화 연구로는 early-exit·reasoning-pruning 등 길이 절감은 다뤘지만, 애초에 ‘생각이 필요한지’ 선택하는 정책 학습과 GRPO 하의 안정성 문제는 충분히 해결하지 못했다.

- **Core Contribution**: Switch-Reasoner는 GRPO 기반으로 MLLM이 Thinking Mode와 Direct Mode를 입력마다 선택하도록 학습시키는 프레임워크다. 핵심은 thinking을 암묵적 생성이 아니라 가상 도구 호출(Thinking-as-Tool)로 모델에 노출해, 먼저 라우팅(생각 호출 여부)을 결정한 뒤 답변을 생성하게 만든 점이다. 또한 듀얼 레벨 규제로 전체 모드 사용 균형과 샘플 수준의 라우트 우선순위를 함께 학습해, 모드 붕괴를 줄이면서도 인스턴스 적응성을 강화한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 GRPO 학습 중 샘플 롤아웃 보상 편차가 정책을 특정 모드로 쏠리게 만들어 항상-thinking 또는 항상-direct로 붕괴하는 불안정성이다. Switch-Reasoner는 (1) 배치 단위의 Global Mode Balance Control로 두 모드의 상대 사용을 성능/사용률 기준으로 지연 제어하고 극단 불균형 시 탐색을 복구하며, (2) 각 입력에 대해 Direct와 Thinking 두 경로의 반사실(카운터팩추얼) 정확도 이득을 비교해 must-think/safe-direct/uncertain으로 나누는 Sample-level Fine-grained Optimization을 추가한다. 이를 통해 전역 균형은 유지하면서도 입력별로 ‘생각이 실제로 이득인지’에 대한 감독 신호를 제공한다.

- **Empirical Impact**: 11개 멀티모달 태스크에서 Switch-Reasoner는 불필요한 thinking 비율을 크게 낮추면서도 정확도를 유지(또는 일부 개선)해 accuracy–efficiency trade-off를 향상시켰다. 예를 들어 Qwen3-VL-4B에서는 thinking rate를 51.53%로 줄이면서 전체 성능이 GRPO-Thinking(100% thinking)과 GRPO-Direct(항상 direct)의 중간을 넘어서는 수준을 보였고, Qwen3-VL-8B에서도 thinking rate 37.73%로 최고의 종합 점수(72.22)를 달성했다. 학습 동역학과 데이터셋별 thinking rate 시각화 결과는 모델이 태스크·샘플 난이도에 따라 선택적으로 사고를 호출하는 정책을 실제로 학습했음을 보여준다.



### VocaDet: Sample-Driven Open-Vocabulary Object Detection and Segmentation via Visual Tokenization and Vector Database Retrieva (https://arxiv.org/abs/2607.08541)
- **Prior Approaches**: SAM, Grounding DINO, INSID류처럼 text/visual prompt 기반 open-vocabulary 검출·분할은 강력하지만, 프롬프트 입력이 필요하거나 매 프레임 유사도 계산이 반복돼 확장성이 떨어진다. 또한 Rex-Thinker, Rex-Omni 등은 multimodal LLM을 쓰지만 추론 비용과 레퍼런스 매칭 부담이 남는다. Training-free 방식(예: DINOv2+SAM2 메모리 뱅크)은 학습 없이 동작하나, 사용자가 큰 규모의 positive/negative 샘플 저장소를 계속 늘려갈 때 이를 자동으로 반영하며 효율적으로 검색하는 데는 제약이 있다.

- **Core Contribution**: VocaDet은 사용자가 제공한 positive/negative 샘플을 기반으로 객체 개념을 ‘재학습 없이’ 바로 학습·갱신하는 sample-driven open-vocabulary 검출·분할 프레임워크를 제안한다. 연속적인 시각 표현을 agglomerative clustering으로 이산 visual token(그리고 멀티 그레인)으로 만들고, 이를 위치 편향 보정과 공간 topology 정보와 함께 vector database의 object memory로 저장해 필요 시 검색으로 localization/segmentation을 수행한다. UA-DETRAC에서, 기존 카테고리 기반 학습 없이도 성능을 보이면서 샘플 저장소를 누적하면 인식 능력이 계속 개선되는 확장성을 강조한다.

- **Technical Challenges**: 핵심 난제는 (1) 연속 feature를 검색 가능한 discrete vocab로 안정적으로 양자화하고, (2) 시각 유사도만으로 생길 수 있는 인접 객체 병합 같은 오류를 줄이며, (3) large-scale 메모리에서 배경 때문에 생기는 중복 검색을 효율화하는 것이다. VocaDet은 DINOv3 feature에 agglomerative clustering(클러스터 민감도 파라미터)로 멀티 그레인 토큰을 만들고, position-debiased 표현을 저장·질의에 동일 적용하며, 객체 매치는 feature similarity뿐 아니라 topology(인접 관계) 일치까지 요구하도록 설계했다. 또한 fixed-camera 환경에서 배경 토큰을 background feature memory로 필터링해 vector database 질의 수를 줄이고, 여러 그레인에서 나온 박스는 NMS로 정리한다.

- **Empirical Impact**: UA-DETRAC 실험에서 VocaDet은 detector 학습 없이도 경쟁력 있는 open-vocabulary 검출 성능을 보였고, positive/negative 샘플을 추가할수록 vector memory가 풍부해지며 결과가 점진적으로 향상되는 점을 확인했다. 한편, 같은 범주의 인접 객체가 동시에 등장할 때 시각 유사도와 클러스터링에 의존하는 탓에 하나로 합쳐지는 한계를 관찰했으며, 이는 discriminative boundary 분리가 부족할 때 발생한다. 초기 벡터베이스 구축 단계에서의 cold-start 문제도 언급되어, 향후 메모리 구성과 카테고리 경계 학습을 더 적응적으로 만드는 방향이 제시된다.



### Whareformer: Learning to Track What is Where in Long Egocentric Videos (https://arxiv.org/abs/2607.08537)
Comments:
          Accepted at ECCV 2026. Project Webpage: this https URL

- **Prior Approaches**: OSNOM 과제는 시점 밖으로 사라지거나 심하게 가려져도 객체의 3D 위치와 정체성(what-where)을 유지해야 한다. 기존 방법들은 고정된 임계값 기반 휴리스틱 연관(appearance-location cost)과 Hungarian 매칭 등으로 트랙을 유지·생성해, 애매한 상황에서의 선택력이 제한되고 새 트랙 초기화 타이밍을 학습하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 OSNOM을 위한 최초의 학습 기반 아키텍처 Whareformer를 제안한다. Whareformer는 (i) 진화하는 트랙 메모리와 (ii) 관측을 기존 트랙에 할당하는 track assignment 모듈을 두고, appearance(what)와 3D 위치(where)를 함께 추론하며 New Track token으로 새 객체 트랙 생성까지 한 번에 결정한다.

- **Technical Challenges**: 핵심 기술 과제는 장기 추적에서 객체 외형이 계속 변하고, 위치·시야가 끊겨도 동일 객체를 재식별해야 한다는 점이다. 논문은 appearance를 DenStream으로 무한 스트림처럼 온라인 클러스터링해 효율적으로 갱신하고, location은 최근 몇 초의 버퍼로 간결하게 표현한 뒤, 상대 거리 기반 임베딩과 순서 비의존 transformer attention으로 관측-트랙 매칭 확률을 학습한다; 또한 teacher forcing의 inference distribution shift는 DAgger 스타일 보정으로 완화한다.

- **Empirical Impact**: Whareformer는 56개 EPIC-KITCHENS-100 학습 비디오로 학습했지만, EPIC-KITCHENS-100(미공개 비디오), IT3DEgo, HD-EPIC의 260개 장기 테스트 비디오에서 SOTA 성능을 보이며 이전 대비 유의미한 절대 개선을 달성했다. 특히 ‘where(정확한 3D 위치)’와 ‘what(정체성 일관성)’을 별도 지표로 평가했을 때 장시간 부재·가림·시점 변화가 많은 egocentric 환경에서 객체 영속성(object permanence)을 더 안정적으로 유지한다는 점에서 의미가 크다.



### Beyond wheelchairs and blindfolds: Investigating disability stereotypes in T2I models with INCLUDE-BENCH (https://arxiv.org/abs/2607.08515)
- **Prior Approaches**: 기존 T2I 편향 연구는 주로 성별·피부톤·문화/직업처럼 비교적 제한된 조건의 고정관념을 다뤘고, 장애는 체계적으로 덜 살펴봤습니다. 또한 최근 벤치마크도 맥락을 충분히 반영하지 못해, 사회학적으로 정의된 ‘스테레오타이핑’ 관점에서 대표적 위해를 원칙 있게 측정하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 장애(장애인, PWD) 관련 편향을 평가하기 위한 대규모 벤치마크 INCLUDE-BENCH를 제안합니다. static은 물론 dynamic 맥락까지 포함해 프롬프트 설계로 다양한 편향 차원을 119K 장면으로 구성하고, 15개 open-source와 2개 closed-source 모델을 동일 조건에서 평가합니다.

- **Technical Challenges**: 주요 도전은 ‘장애 스테레오타이핑’을 자동 지표로 사회학적 정의에 가깝게 측정하는 것이었습니다. 저자들은 person-centric 평가를 위해 SAM3로 사람 영역만 크롭한 뒤, CLIP 기반 정렬(semantic alignment)·Vendi Score로 다양성·그리고 Stereotype Content Model(SCM) 기반 Stereotype Content Model Score(SCM Score)를 도입해 장애 텍스트 정렬과 고정관념 재현을 함께 추적합니다.

- **Empirical Impact**: 실험 결과, mobility-impaired 및 기본 장애 프롬프트는 대부분의 모델에서 휠체어를 핵심 이미지로 강하게 생성하는 패턴이 반복됐습니다. 장애를 조건화하면 생성 다양성이 줄고, 고정관념적으로 묘사된 경우 장애 텍스트 정렬이 더 강해지며 SCM Score로도 실제 사회의 스테레오타입 연관성이 구조적으로 반영됨을 보였습니다.



### Do Egocentric Video-Language Models Capture Both Hand- and Object-Centric Cues? (https://arxiv.org/abs/2607.08514)
- **Prior Approaches**: 기존 비디오-언어 HOI 인식 모델은 손/물체/환경 간 상관관계를 이용해 동작을 맞히는 ‘지름길(shortcut)’에 빠지기 쉽다는 한계가 지적된다. 특히 손 자세와 물체 범주가 비슷한데도 실제 상호작용이 다른 경우, 모델이 실제 상호작용 역학이 아니라 스파리어스 상관에 의존할 수 있다. 관련 연구들은 공간 근거를 돕는 디코더나 마스킹 기반 학습을 시도했지만, 손과 물체의 단서를 분리해 독립적으로 추론하도록 강하게 유도하진 못했다.

- **Core Contribution**: 이 논문은 HOI를 손 단서와 물체 단서가 함께 만드는 결과로 보고, 두 단서를 더 견고하게 학습하는 새로운 패러다임을 제안한다. 핵심은 (1) 손/물체 영역을 역할 단위로 마스킹해 부분 관측에서도 재구성하도록 만드는 hand-object masked training과, (2) 손/물체 중심 임베딩을 보조 예측(위치·의미)으로 학습하는 HOI-dynamics-aware(HDA) decoder다. 또한 손·물체 단서를 각각 따로 넣었을 때도 동작 동사를 맞히는지 측정하는 Cue-Isolated HOI(CI-HOI) 평가와, 이를 위한 DEHOI 테스트베드를 새로 구축한다.

- **Technical Challenges**: 손과 물체 단서가 한 영상 안에서 함께 등장해 단서별 기여를 분리하기가 어려웠고, 일반적인 마스킹이나 영상-텍스트 정렬만으로는 이러한 얽힘을 충분히 끊기 어렵다. 저자들은 이를 해결하기 위해 튜브릿 단위로 손/물체/배경을 탐지 기반 오버랩 기준으로 구분한 뒤, 손 단서만/물체 단서만 남기도록 마스킹을 설계한다. HDA 디코더는 DETR-like 쿼리를 통해 손 중심·물체 중심 임베딩을 별도로 만들고, 각 임베딩에 대해 위치(박스)·의미(동사/명사)·영상-서술 정렬을 멀티태스크로 동시에 학습해 단서별 민감도를 높였다.

- **Empirical Impact**: DEHOI(인페인팅으로 손 또는 목표 물체를 제거한 영상)에서 기존 모델 대비 정량·정성 모두 향상되며, 단서별 추론이 더 잘 이뤄진다는 점을 보여준다. 특히 CI-HOI 설정에 fine-tuning 없이도 DEHOI 성능 개선이 표준 평가 입력에서도 일관되게 전이되어, 지름길 의존을 줄이고 손·물체 중심 역학을 강화했음을 시사한다. 추가로 물체 상태 인식 및 로봇 조작 액션 인식 벤치마크까지 개선 폭을 넓혀, 더 견고한 HOI 이해와 일반화에 의미 있는 영향이 있음을 입증한다.



### CT-CLIP Representations for Multimodal Lung Cancer Survival Prediction (https://arxiv.org/abs/2607.08503)
Comments:
          8 pages, 2 figures

- **Prior Approaches**: 기존 폐암 생존 예측은 CoxPH처럼 임상변수 중심의 단일모달 접근이나, 이미지 단독용 CNN 기반 DeepConvSurv류에 주로 의존해 왔다. 최근에는 CT와 임상 데이터를 결합하는 멀티모달 모델이 등장했지만, 대부분 종양 영역에 대한 수동/자동 주석 의존성이 있거나 의료용 사전학습 표현의 전이 효과가 충분히 검증되지 못했다. 또한 데이터가 적은 임상 현실에서는 대규모 공개 영상-결과 연동 코호트 부족으로 성능과 일반화가 제한되는 문제가 남아 있다.

- **Core Contribution**: 본 연구는 도메인 특화 비전-언어 파운데이션 모델 CT-CLIP의 표현을 “데이터가 적은(real-world, data-constrained)” 상황에서 멀티모달 생존 예측에 활용할 수 있는지 실증한다. CT-CLIP의 고정된( frozen ) 비전 인코더와 템플릿 기반 임상 노트(텍스트 인코더)를 임베딩으로 만든 뒤, 가벼운 survival head(DeepSurv 계열)만 학습해 CoxPH의 log-risk를 예측하도록 구성했다. 특히 적응 전략으로 frozen encoders, full fine-tuning, LoRA를 함께 비교해 어떤 방식이 임상 데이터 제약에서 유리한지 보여준다.

- **Technical Challenges**: 가장 큰 기술적 난제는 CT-CLIP이 원래 “영상-방사선 리포트”에 맞춰 학습되었는데, 연구 데이터에는 리포트가 없어 임상 변수를 텍스트 인코더에 넣을 수 있는 입력 설계가 필요하다는 점이다. 이를 위해 표준화된 임상 템플릿으로 임상변수를 clinical notes로 변환하고, 결측 변수는 해당 문장 구간을 생략하는 전처리를 적용했다. 또 소규모 데이터에서 full fine-tuning이 과적합을 유발할 수 있어, frozen 백본과 LoRA 같은 매개변수 효율적 적응을 비교하며 최적 구성을 탐색했다.

- **Empirical Impact**: 스웨덴 실제 병원 코호트 242명(보유: pretreatment CT + 임상변수, 목표: overall survival)에서 frozen CT-CLIP + lightweight survival head가 임상 baseline(CoxPH)을 능가하며, 다른 멀티모달 모델들과 비교해 동등하거나 개선된 성능을 보였다. 이미지/텍스트 모달리티 ablation 결과에서도 멀티모달 결합이 단일모달보다 일관되게 유리했고, 특히 backbone을 고정하는 설정이 discriminative ability에서 더 좋거나 LoRA 수준에 근접함을 확인했다. 예측 위험으로 환자를 high/low-risk 그룹으로 나눌 때 Kaplan-Meier와 log-rank 검정에서 유의한 분리(p<0.001)를 보이며, 임상적으로 의미 있는 위험 계층화 가능성을 제시했다.



### Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing (https://arxiv.org/abs/2607.08497)
Comments:
          16 pages, 7 figures, 8 tables. Project page: this https URL Code: this https URL

- **Prior Approaches**: 최근 unified multimodal 모델은 한 아키텍처에서 시각-언어 이해와 이미지 생성/편집을 함께 수행하지만, 장문 대화에서 과거의 모든 시각 토큰을 공통 컨텍스트에 계속 주입하는 구조적 한계가 있습니다. 이로 인해 visual token 폭증으로 추론 예산이 줄고, 긴 누적 컨텍스트에 대한 암묵적 attention 의존이 교차 턴 시각 참조를 불안정하게 만들어 retrieval 오류와 의미 드리프트가 커집니다. 메모리 보강을 시도한 에이전트들도 영상/텍스트를 각각 다루거나(비디오 중심, 단일 태스크 중심) 시각 디테일을 충분히 보존하지 못해 near-duplicate 이미지 구분에 취약합니다.

- **Core Contribution**: 논문은 Cognitive-structured Multimodal Agent(CMA)를 제안하며, 시각 정보를 Episodic Visual Memory(EVM)로 외부화하고 필요할 때만 관련 에피소드를 선택적으로 reactivates 하도록 설계합니다. Perceptual Abstraction Engine(PAE)이 이미지에서 태그·캡션·썸네일의 구조화 표현을 만들고, Cognitive Retrieval Engine(CoRE)이 대화 흐름에 맞는 시각 에피소드를 검색한 뒤, Multimodal Executive Controller(MEC)가 태스크 의도 추론과 실행 계획을 담당합니다. 또한 turn-level 시각 retrieval 감독이 부족한 문제를 Unified Scenario Engine으로 해결해, 회수할 에피소드에 대한 미세 주석이 포함된 다턴 대화 데이터를 생성합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “장기 대화에서 어떤 과거 이미지를 회수해야 하는가”에 대한 명시적 감독 신호를 확보하는 것이었습니다. 기존 데이터셋은 단일 턴 grounding이나 짧은 컨텍스트만 다루는 경우가 많아, 각 턴마다 정답 retrieval 집합을 제공하지 못합니다. 저자들은 Unified Scenario Engine으로 생성된 시나리오에 대해 turn-level retrieval annotations를 만들고, CoRE에는 SFT+RL(지연 학습된 검색 정책 최적화)을 적용하며 PAE도 retrieval 유용성 기준으로 강화해 메모리 작성-검색을 end-task 관점에서 맞춥니다.

- **Empirical Impact**: M2CA-Bench에서 8B CMA는 20턴 세션 기준 retrieval 정확도 91.4%를 기록해, 32B unified baseline을 +8.2%p로 능가하고 per-turn 추론 시간을 23.1s에서 12.7s로 거의 절반 수준으로 줄였습니다. 특히 대화가 길어질수록 격차가 확대되어 Full→Hard에서의 성능 하락이 더 완만했으며, retrieval 정확도가 생성 품질(0~10 점수)로 직접 전이되는 경향도 확인됐습니다. 또한 CMA-Harness로 동일 인지 구조를 도구 실행·웹 접근·이미지 생성/편집까지 확장했으며, monolithic 파라미터 스케일링보다 구조화 메모리와 모듈형 의사결정이 장기 멀티모달 에이전트에 더 확장성 있고 효율적이라는 점을 시사합니다.



### VEGAS: Human-Aligned Video Caption Evaluation via Gaz (https://arxiv.org/abs/2607.08489)
- **Prior Approaches**: 기존 비디오 캡셔닝은 시각-언어 모델이 의미적으로는 그럴듯한 문장을 만들지만, 군중 주석을 평균낸 기준이라 개별 시청자의 주의(시선)를 반영하지 못하는 경우가 많습니다. 그래서 여러 작품이 gaze를 학습용 감독신호로 쓰거나(예: supervised fine-tuning), 추론 시 인간 시선을 활용해 이해/의도 해석을 보강하려 했습니다. 다만 학습 없이(test-time) 시선 기반으로 캡션 자체를 개인화해 “평가/선택”까지 수행하는 방법과, 이를 검증할 동기화된 멀티모달 데이터는 제한적이었습니다.

- **Core Contribution**: 이 논문은 시선으로 개인화된 캡션을 고르는 학습 없는(metric-based) 방법 VEGAS(Video caption Evaluation via GAze Score)를 제안합니다. VEGAS는 시선에 의해 주목된 영역만으로도 캡션이 얼마나 잘 예측되는지를 정보이론적으로 측정해, 개인의 주의와 일치하는 텍스트를 선호하도록 설계됐습니다. 또한 VEGAS 점수로 rejection sampling을 수행해 VLM 재학습 없이 개인화 캡션을 선택하는 절차를 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 시선-캡션 간 상호정보량의 직접 계산이 불가능하다는 점이며, 이를 위해 논문은 VLM의 토큰별 likelihood를 활용해 불능의 분포를 근사합니다. 구체적으로 주목 영역( attended )과 비주목 영역( non-attended )을 분해한 뒤, 비주목 영역을 추가했을 때 캡션 예측이 얼마나 더 좋아지는지를 pointwise conditional mutual information 형태로 점수화합니다. 이후 여러 후보 캡션을 VLM에서 뽑고, VEGAS 점수가 가장 낮은(비주목 정보 의존이 작은) 캡션을 선택해 시선 정렬을 달성합니다.

- **Empirical Impact**: 저자들은 동기화된 비디오-시선-캡션을 제공하는 멀티도메인 데이터셋을 구성해(egocentric AEA, instructional SlideVQA) VEGAS를 검증합니다. AEA에서는 인간 초점과의 정렬이 유의미하게 개선되며 mean SBERT similarity가 +0.0856 향상되고 caption-to-video retrieval의 mAP도 여러 rank에서 +1.14%~+2.48% 수준으로 개선됩니다. 반면 SlideVQA에서는 SBERT 향상이 +0.0256으로 더 작고 유의성이 약해, VEGAS 효과가 “시선으로 구체적 지시어 모호성을 푸는” 상황에서 더 두드러진다는 도메인 의존성이 관찰됩니다.



### Predicting Viticulture Potential through an Ensemble of U-Net and a Geospatial Foundation Mod (https://arxiv.org/abs/2607.08449)
Comments:
          To be published in CLEF 2026 Working Notes

- **Prior Approaches**: 농업 적합성 평가는 전통적으로 현장 조사·토양 검정 등 비용과 시간이 많이 드는 작업에 의존해 왔다. ImageCLEF AI4Agri 2026 Subtask 1에서는 Sentinel-2 다중시기 영상을 픽셀 단위로 포도 재배 적합도(1~5)로 분류하며, 기존에는 U-Net 기반 세그멘테이션이 강한 기준선으로 쓰였다. 또한 NDVI 등 지표의 시간 변화를 근거로 temporal modeling의 중요성을 기대했지만, 복잡한 시간 모델이 항상 성능으로 이어지지는 않는 한계가 드러난다.

- **Core Contribution**: 이 논문은 U-Net과 Prithvi-EO-2.0(Geospatial foundation model)을 결합한 weighted ensemble로 포도 재배 잠재력을 예측한다. U-Net은 34개 시점을 스태킹해 timesteps를 채널처럼 취급하고, Prithvi는 계절 평균으로 집계한 입력을 fine-tuning해 서로 다른 귀납 편향을 확보한다. 두 모델의 logits를 검증셋으로 캘리브레이션한 가중치로 결합해 일반화 성능을 끌어올리는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 다중시기 위성 영상에서 시간 정보와 공간 경계 정보를 동시에 학습해야 한다는 점, (2) 학습 데이터 내 비라벨 픽셀이 많아 손실이 불안정해질 수 있다는 점, (3) 검증-테스트 간 일반화 갭이 반복적으로 나타난다는 점이다. 이들은 ordinal loss로 클래스 순서를 반영하고, U-Net은 510채널(시점×밴드) 스태킹으로 세밀한 공간 패턴을 학습하며, Prithvi는 seasonal aggregation과 pretraining 정합을 통해 전이학습 효율을 높였다. 또한 U-Net을 teacher로 삼는 간단한 pseudo-labeling(신뢰도 0.7 이상)로 비라벨 영역에 추가 학습 신호를 주고, 회전·좌우상하 flip 증강 및 앙상블로 분포 변화에 대응했다.

- **Empirical Impact**: 결과적으로 최종 앙상블은 테스트 ±1 정확도 68.32%로 리더보드 2위를 기록했으며, 단일 모델보다 일관되게 성능이 높았다(U-Net 66.25%, Prithvi 65.51%). 정성 평가에서도 Prithvi는 더 매끈하고 공간적으로 일관된 예측을 보인 반면 U-Net은 국소 잡음이 더 나타났고, 앙상블이 경계 주변의 오류를 일부 줄이면서 전체 예측 품질을 개선했다. 다만 일반화 갭은 여전히 남아 추가로 테스트 클래스 분포·비라벨 픽셀 비율·지형 차이를 분석할 필요성이 제기된다.



### DeltaV: Thinking with Visual State Updates in Unified Large Multimodal Models (https://arxiv.org/abs/2607.08434)
- **Prior Approaches**: 기존 ULMM(통합 대규모 멀티모달 모델)은 interleaved multimodal reasoning에서 중간 시각 상태를 텍스트 추론과 함께 생성하지만, 각 시각 단계를 매번 ‘전체 이미지’로 다시 만들어야 했습니다. 이 방식은 직전 상태와의 시각적 상관을 충분히 활용하지 못해 시각 토큰의 중복이 커지고, 추론에 결정적인 ‘상태 전이’ 구간에 대한 감독 신호가 희석됩니다. 또한 시각 생성 비용이 텍스트 추론 비용보다 커 다단계 추론에서 효율과 스케일링이 제약되는 문제가 있었습니다.

- **Core Contribution**: DeltaV는 interleaved multimodal reasoning을 ‘반복되는 전체 이미지 생성’이 아니라 ‘시각 업데이트 중심(visual updates) 상태 전이 모델링’으로 재정의한 ULMM입니다. 과거 시각 상태를 조건으로, 매 단계 필요한 변경만을 compact update tokens으로 점진적으로 예측해 불필요한 재생을 줄입니다. 더불어 변화량에 맞춰 토큰 예산을 동적으로 배분하는 TSIM Router와, 일반화 가능한 학습을 위한 StructCoT(1.05M, 44개 태스크 도메인)까지 함께 제안합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) reasoning 단계마다 얼마나 ‘변화’가 필요한지 추정하고, (2) 그 변화량에 비례해 토큰 예산을 효율적으로 할당하며, (3) 가변 길이 visual update를 추론과 자연스럽게 결합하는 것입니다. DeltaV는 temporal similarity(TSIM)를 통해 시각 변화 정도를 정량화하고, offline 캘리브레이션으로 TSIM 구간별 reconstruction fidelity–token 간 곡선을 피팅해 말단 이득이 작아질 때 토큰 생성을 멈추도록 설계합니다. 학습 시에는 <|vision_end|> 경계 토큰으로 “언제 멈출지”를 감독해, 추론 시에는 이 stopping 메커니즘을 통해 자동으로 가변 업데이트 길이를 학습·전이합니다.

- **Empirical Impact**: 실험 결과 DeltaV는 full-image 생성 패러다임 대비 새로 생성되는 visual 토큰을 평균 55.6% 줄이면서도 reconstruction fidelity를 손상시키지 않았습니다. 멀티모달 추론 성능도 full-image 생성 대비 3.3% 향상되었고, StructCoT 및 대규모 멀티모달 데이터로 학습한 DeltaV-2B는 더 큰 오픈소스 모델들에 대해 in-domain 멀티모달 추론에서 평균 8.4% 이득을 보였습니다. 또한 external 멀티모달 reasoning/understanding 벤치마크에서는 Qwen3-VL-2B(동급) 대비 평균 5.9% 앞섰으며, 시각 업데이트 중심 설계가 추론 효율·정확도 양쪽에서 의미 있는 개선을 제공함을 입증했습니다.



### Track2Map: Online Deformable SLAM with Motion-Aware Pose Optimization in Robotic Surgery (https://arxiv.org/abs/2607.08408)
Comments:
          Accepted at MICCAI 2026. This is the submitted version prior to peer review. The final authenticated version will be available on SpringerLink

- **Prior Approaches**: 기존 3D Gaussian Splatting(3DGS) 기반 RAMIS 복원은 비변형 장면이나 정적/정확한 카메라 포즈를 가정하는 경우가 많아, 조직이 비강체로 움직이고 시야가 가려지는 수술 영상에선 드리프트가 쉽게 발생한다. 또 NRSfM/변형 SLAM 계열은 카메라 포즈와 변형을 함께 추정하지만, 엔도스코프 움직임이 간헐적인 상황에서 포즈 최적화가 조직·도구 변화를 카메라 운동으로 잘못 흡수하는 실패 모드가 남아 있었다.

- **Core Contribution**: 이 논문은 수술 비디오만으로 카메라 궤적과 비변형(변형) 장면을 동시에 온라인으로 최적화하는 Track2Map을 제안한다. Simultaneous Localisation and Mapping(SLAM)처럼 작동해 포즈 priors가 없거나 노이즈가 있어도 보다 견고한 3D 복원을 목표로 한다. 또한 dense 2D point tracks를 기반으로 변형을 초기화하고, 정적 카메라 구간을 감지해 누적 드리프트를 줄이는 통합 전략을 제공한다.

- **Technical Challenges**: 핵심 난제는 제한된 시점 변화(스테레오 단기 베이스라인)에서 역렌더링이 포즈와 조직 변형을 구분하기 어렵다는 점이다. 이를 위해 Track2Map은 트래킹 흐름의 방향 분포 분산으로 카메라 움직임 가능성을 게이팅해, 카메라가 정적인 구간에서는 포즈 업데이트를 동결하고 움직임이 감지될 때만 소규모 pose increment를 반영한다. 동시에 스테레오 깊이로 2D 트랙을 3D 앵커로 들어 올려 변형 제어를 초기화·구동하고, 증분 매핑 단계에서 포토메트릭/지오메트릭 렌더링 손실과 변형 정규화를 함께 최적화한다.

- **Empirical Impact**: StereoMIS 실험에서 Track2Map은 포즈가 없는 설정부터 노이즈가 큰 설정, 깨끗한 설정까지 전반적으로 재구성 품질과 카메라 궤적 정확도를 개선했다. 특히 스테레오 입력이 메트릭 스케일을 고정하고, motion-aware pose update가 카메라 정적 구간의 드리프트를 억제해 로봇 kinematics나 사전 궤적 없이도 일관된 해부학 형상을 복원하는 데 기여한 것으로 보인다. 또한 변형 모델링, 정적 카메라 게이팅, 포즈 최적화가 상호보완적으로 작동함을 애블레이션으로 확인했으며, 공개 코드도 제공된다.



### Swapping Faces, Saving Features: A Dual-Purpose Pipeline for Pedestrian Privacy in ITS (https://arxiv.org/abs/2607.08402)
- **Prior Approaches**: 자율주행/ITS용 보행자 의도·궤적 예측은 다양한 보행자 영상 데이터가 필요한데, 얼굴은 생체인식 정보라 신원 탈취·추적·딥페이크 등 보안 위험을 키운다. 기존 방법은 블러/픽셀화처럼 얼굴을 가리는 방식이 많지만 훈련에 필요한 표정·시선 등 속성을 훼손해 데이터 유용성을 떨어뜨린다. GAN 기반 익명화는 프라이버시는 확보하더라도 속성 보존이 약해 AV 학습용 얼굴 단서 활용에 한계가 있었다.

- **Core Contribution**: 본 논문은 보행자 신원을 숨기면서도 표정·고개 자세·시선 같은 핵심 얼굴 속성을 유지하도록, face swapping 중심의 5단계 파이프라인을 제안한다. 이 파이프라인은 Egy-DRiVeS처럼 이집트 거리 영상의 저해상도·다양한 복장(예: 베일) 같은 특수 케이스에 맞춰 설계됐다. 또한 비교 실험을 통해 Roop과 Ghost-v2 중 파이프라인 적용에 더 적합한 face swapper를 선정한다.

- **Technical Challenges**: 핵심 난제는 (1) 저해상도/가림/먼 거리로 얼굴 정보가 부족한 상황에서 swapping 품질을 유지하고, (2) 프라이버리(정체 은폐)와 속성 보존(시선·표정·자세)을 동시에 달성하는 것이다. 이를 위해 보행자·얼굴 검출 후 Codeformer 기반 품질 보강으로 복원 신뢰도를 높이고, 블렌딩까지 포함한 end-to-end 처리 흐름을 구성했다. 평가 지표는 랜드마크/블렌드셰이프 차이, 얼굴 임베딩 유사도, gaze vector 유사도 등으로 “얼굴 구조·표정 유지 vs 동일성 은폐”를 함께 측정한다.

- **Empirical Impact**: 고품질 얼굴 테스트에서 Roop은 Ghost-v2 대비 4개 정량 지표 중 3개에서 우수하며, 특히 표정(블렌드셰이프)·얼굴 구조/자세(랜드마크 차이) 보존이 더 잘 나타났다. 또한 Occluded face(가림된 얼굴)와 베일 여성 같은 어려운 사례에서 Ghost-v2는 비현실/윤리적으로 문제 소지가 있는 결과를 보인 반면 Roop은 더 높은 견고성을 보여 파이프라인 신뢰성을 뒷받침했다. 최종 5단계 적용 후에는 JAAD의 looking/not looking(시선 방향) 특징이 파이프라인 전후로 유지되어, 프라이버리 보호가 다운스트림 의도 예측에 필요한 얼굴 단서까지 망치지 않는다는 점을 실증했다.



### Attribute Retrieving for Open-Vocabulary Endoscopic Compositional Referring Segmentation (https://arxiv.org/abs/2607.08397)
- **Prior Approaches**: 기존 endoscopic image segmentation은 시각 정보 위주로 학습되는 경우가 많아, 텍스트 지시에 기반한 정밀한 referring을 제공하기 어렵다. vision-language segmentation이 발전했지만 endoscopy에서는 미세한 텍스트 단서(속성, 관계)를 충분히 반영하지 못해 경계·형상 정확도가 떨어지고, 도메인 일반화도 제한적이라는 문제가 남아 있다.

- **Core Contribution**: 본 논문은 endoscopic RIS를 위한 대규모 기준선(benchmark) ReferEndoscopy(65,964장, 242,055개 마스크, 1,452,330개 image–mask–instruction triplets)를 제안한다. 또한 open-vocabulary endoscopic compositional referring segmentation을 위한 AR-ERIS( Attribute Retrieval-based Endoscopic-RIS ) 프레임워크를 제안하며, ReferEndoscopy로 사전학습해 시뮬레이션과 실제 데이터 전반에서 SOTA 성능과 강한 일반화를 보인다.

- **Technical Challenges**: 핵심 난제는 (1) endoscopy의 경계·형상처럼 고주파 정보가 중요한데 이를 시각-언어 정렬에 효과적으로 연결하는 것, (2) 희귀 클래스·롱테일로 인해 속성/카테고리 편향이 심한 데이터에서 안정적으로 동작하는 것이다. AR-ERIS는 Fourier로 저주파/고주파를 분리한 frequency-aware feature fusion(Freq-Fusion)과 CLIP 기반 텍스트-픽셀 정렬을 결합하고, Dice·edge 관련 손실 및 frequency consistency loss로 경계 정밀도를 강화한다.

- **Empirical Impact**: 실험에서는 instruction의 속성 수가 늘어날수록(basic→hard) 기존 개방형 모델들이 성능이 크게 흔들리는 반면, AR-ERIS는 mIoU/Dice 모두에서 일관되게 우수하며 속성 복잡도 증가에도 견고함을 보인다. 또한 외부 도메인 SAR-RARP50에서도 GroundedSAM 대비 mIoU 9.03%p 향상을 0-shot으로 달성하고, hard 지시에서 Prec@50과 mIoU가 함께 개선되어 미지 도메인에서도 경계 앵커링이 가능함을 시사한다. 데이터와 코드는 리뷰 완료 후 공개 예정이다.



### Classical versus Deep Mirror-Symmetry Scoring: A Benchmark of Thirteen Methods (https://arxiv.org/abs/2607.08379)
Comments:
          22 pages, 6 figures, 5 tables. Code and benchmark: this https URL

- **Prior Approaches**: 기존 대칭 점수화(symmetry scoring) 연구는 화소/그라디언트 기반 지표부터 DCT·필터뱅크·주파수 계수, 그리고 pretrained CNN의 frozen feature까지 다양한 방법을 제안했지만, 동일한 프로토콜과 통계적 유의성 검증 하에서의 head-to-head 비교는 없었다. 특히 ‘축 검출’과 달리 ‘지정된 축에 대한 점수’(discrimination)를 공통 벤치마크로 다루는 실험 설계가 부족했다.

- **Core Contribution**: 이 논문은 거울 반사 대칭 점수화를 위한 첫 대규모 벤치마크를 구축하고, 13개 점수기(scoring methods)를 하나의 representation–comparison–aggregation 템플릿으로 재구성해 일관된 비교를 수행한다. 또한 open toolkit imgsym을 공개하고, reflection-exact한 harness에서 chance-anchored skill과 유의성 테스트를 통해 어떤 방식이 실제로 잘 구분하는지 비용까지 함께 매핑한다.

- **Technical Challenges**: 핵심 난제는 ‘정답 축이 어느 정도인지’를 순수 콘텐츠 기반으로 측정하기 위해, 잘못된 축일 때 생길 수 있는 crop 크기/기하학 편향을 통제하고 동일한 음성(perturbed negatives) 세트를 비교하도록 설계하는 것이다. 이를 위해 축을 수직 중심선에 정렬하는 canonical warp, flip에 의한 반사 정확성 보장, crop-extent 영향 검증(고정 extent 재실험), 그리고 좌우 이동/회전 기반의 제어된 부정 샘플로 discrimination skill을 계산한다.

- **Empirical Impact**: 실험 결과 deep backbones의 성능이 단일 축과 더 어려운 다중 축(multi-axis) 프로토콜에서 가장 높았지만, 튜닝된 HOG는 최고 frozen-feature readout과 유의미하지만 아주 작은 격차(작지만 significant)만 보였고 통계적으로 runner-up과도 분리되지 않았다. 또한 mid-scale의 방향성 특징에서 판별이 집중되며( deep는 low~mid stage, HOG는 mid cell size ), 컴퓨팅 관점에서 HOG는 CPU에서 약 300배 빠르다는 결론이 나왔다. 종합하면 frozen deep features는 ‘대칭 측정’에서 고전 지표를 압도한다고 단정하기 어렵고, task-trained deep scorers가 격차를 더 벌릴 수 있는지는 남은 연구 과제로 제시된다.



### WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving (https://arxiv.org/abs/2607.08375)
Comments:
          20 pages, 7 figures

- **Prior Approaches**: 기존 E2E·VLA 자율주행은 입력에서 궤적을 바로 생성하는 end-to-end 방식이 주류지만, 복잡/장꼬리(long-tail) 상황에서 인과 추론과 세계 지식 부족으로 취약하다는 한계가 반복해서 지적된다. VLM을 결합한 접근은 장면 이해를 개선했으나, 세계 예측을 보조 과제로 취급하거나(semantic forecasting/image forecasting) 생성적 상호작용 예측이 분절돼 reactive driving에 머무는 경우가 많다. 또한 사회적 상황에서 필요한 게임이론적 ‘if-what’ 전략 추론을 충분히 학습시키지 못했다.

- **Core Contribution**: 이 논문은 Vision-Language-Action(VLA)에 dual-level World Cognition을 결합해 proactive driving을 목표로 하는 WCog-VLA를 제안한다. semantic 수준에서는 3D 공간 인지와 agent tokens, 그리고 Game-CoT 기반의 추론을 통해 세계 동역학을 추론하고, generative 수준에서는 ADDT(Aligned Decoupled Diffusion Transformer)로 다중 에이전트의 물리적으로 그럴듯한 공동 궤적을 생성해 ‘세계 예측→행동’의 연결을 강화한다. 결과적으로 의미론적 세계 예측과 생성적 세계 진화를 함께 다뤄 파편화된 foresight를 줄이는 것이 핵심이다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 2D 중심 VLM에 3D 공간 구조를 주입해 agent 간 기하·위치를 구조적으로 표현하고, (2) VLM의 의미론적 은닉표현을 확산 기반 생성이 요구하는 연속 궤적 세부 복원과 안정적으로 이어 붙이는 것이다. 논문은 TrackFormer 기반 agent tokens와 world head로 3D 기반 semantic world cognition을 만들고, ADDT에서는 condition encoder와 generation decoder를 분리하며 latent scene 표현과의 representation alignment(정렬 제약)로 의미-물리 간 gap을 줄인다. 전략 추론을 위해서는 Qwen3-VL-Plus로 Stackelberg game 형태의 4단계 Game-CoT(85k 주석) 데이터를 구축해 hallucination을 억제하는 GT action 힌트까지 더했다.

- **Empirical Impact**: NAVSIM v1에서 WCog-VLA는 SOTA PDMS 92.9를 달성하며, 카메라만 사용하면서도 lidar를 함께 쓰는 일부 멀티모달 베이스라인을 4.6 PDMS 앞섰다. 또한 대형 VLM 계열 대비 PDMS가 크게 개선되고, 2B급 소형 모델도 RL-refined VLM 기반 방법들을 최소 0.8 PDMS 이상, 3B 모델 LatentVLA도 0.5 PDMS 상회해 효율적인 성능 향상을 보여준다. 안전 지표에서도 NC 99.4, TTC 98.5로 강한 결과를 내며, 주변 에이전트의 미래 의도를 선제적으로 반영한 proactive 안전 운전의 효과를 실증한다.



### Texture Representations in Deep Vision Models: Comparing CNNs, Vision Transformers, and Human Perception (https://arxiv.org/abs/2607.08321)
- **Prior Approaches**: CNN은 인간·영장류의 시각 처리를 설명하는 모델로 널리 쓰였고, 객체 인식 같은 의미 중심 과제에서의 표현 정렬이 중요 지표로 자리 잡아 왔다. 텍스처 연구에서는 Julesz의 저차 통계 가설을 바탕으로 Portilla&Simoncelli, Victor&Conte, Gatys 등 합성 알고리즘이 발전했지만, 이런 텍스처 지각 정렬을 CNN과 다른 아키텍처까지 폭넓게 비교한 연구는 제한적이었다.

- **Core Contribution**: 이 논문은 객체 인식 패러다임을 벗어나 텍스처 지각을 중심 문제로 삼고, 동일한 소스 이미지에서 서로 다른 복잡도를 갖는 텍스처를 합성해 CNN과 Vision Transformer(ViT)를 비교한다. 또한 rank-based 정보 불균형(Information Imbalance, II)으로 모델의 내부 표현이 텍스처 복잡도에 따라 어떻게 조직되는지 정량화하고, 이를 인간 심리물리 실험(odd-one-out) 결과와 직접 연결한다.

- **Technical Challenges**: 핵심 난제는 텍스처처럼 의미가 덜 명확한 자극에서, 고차원 표현이 지각 유사성을 어떻게 반영하는지 “표현의 구조” 수준에서 비교하는 방법이다. 논문은 텍스처 합성 알고리즘별로 자극 복잡도를 단계화하고, 모든 모델의 층별 특징을 추출한 뒤 II로 로컬 이웃 기반 예측가능성을 비교하며, ViT에서 나타나는 outlier(어텐션 관련 효과)는 상위 분위수로 클리핑해 안정적인 기하 비교가 가능하도록 했다.

- **Empirical Impact**: 결과적으로 ViT 3종(CLIP, DINO-v2, iGPT)은 서로 텍스처 복잡도에 대해 유사하고 예측가능한 표현을 형성하지만, CNN(VGG-19)은 ViT와의 정렬이 약하며 Noise~복잡도 전개에서도 표현 전략이 다르게 나타났다. 인간의 텍스처 구분 정확도는 VGG-19보다 ViT 표현과 더 잘 상관되며, 텍스처 지각은 네트워크 아키텍처가 유도하는 표현 경로에 더 가깝게 반영될 수 있음을 시사한다. 이는 기존의 객체 인식 중심 정렬 척도가 텍스처 같은 다른 지각 현상을 과소평가했을 가능성을 보여준다는 점에서 의미가 크다.



### ARGUS: Accelerated, Robust, General, and Unsupervised Cell Tracking Solutions (https://arxiv.org/abs/2607.08297)
- **Prior Approaches**: 기존 2D 라이브셀 트래킹은 지역(local) 연결 방식처럼 빠르지만 폐색·분열에서 오류가 커지기 쉽다. 반대로 전역(global) 최적화는 정확도가 높을 수 있으나 DP/Viterbi나 ILP 같은 조합 최적화는 계산량이 커 대규모 데이터에서 병목이 된다. Optical flow 기반은 연속 프레임에서 탐색공간을 줄여 속도를 확보하지만 분열·장시간 폐색 같은 복잡 이벤트에서 약점이 남아 있다.

- **Core Contribution**: ARGUS는 학습 없이(adaptive) 동작하는 모듈형 프레임워크로, 탐지-연결-정제를 두 단계로 분해해 전역 최적화의 지수적 부담을 피한다. 밀집 Farneback optical-flow로 다음 위치를 예측하고, 프레임 간 linear assignment로 초기 트랙 조각(tracklet)을 만든 뒤, 짧은 시간 간격에 대해 sequence-level tracklet refinement로 끊긴 궤적을 재연결한다. 또한 분열(mitosis)은 규칙 기반으로 명시 처리하고, 세포 출현·소멸·일시 폐색은 휴리스틱으로 관리해 계보(lineage) 정보를 보존한다.

- **Technical Challenges**: 핵심 기술 난제는 잡음·명암 불균일·세포 형태 변화·중첩, 그리고 프레임 간 불연속(검출 누락, 짧은 간극)이다. ARGUS는 데이터 준비 단계에서 퍼센타일 기반 intensity clipping과 wavelet denoising을 표준으로 적용하고, phase-contrast 데이터에는 monogenic phase-symmetry filtering과 같은 모달리티 특화 전처리를 더해 검출 품질을 안정화한다. 이후 optical-flow 예측을 gating(거리 임계)와 결합해 연속 연결의 불확실성을 줄이고, 정제 단계에서는 트랙렛 끝점의 방향·호환성 비용을 고려한 one-to-one 매칭으로 빠른 시간 간격 재연결을 수행한다.

- **Empirical Impact**: CTC Cell Tracking Challenge의 4개 공개 데이터셋에서 ARGUS는 DET 0.905~0.971, TRA 0.897~0.964의 범위를 보이며 대부분 데이터셋에서 CTC 평균을 상회했다. 특히 DET/TRA가 각각 0.971/0.964로 가장 높은 성능을 보인 Fluo-N2DH-SIM+를 비롯해 Fluo-C2DL-Huh7(0.957/0.946), Fluo-N2DH-GOWT1(0.943/0.937)에서도 경쟁력을 확인했다. 또 ground-truth segmentation 마스크가 주어지면 거의 완벽한 성능으로 수렴하고, CPU 병렬 처리 기준 3프레임 처리 시간이 약 5초 수준이라 훈련 데이터·GPU 없이도 실사용에 가까운 속도를 제시한다.



### Enhancing the KidSat Model: Integrating Geographical Encoding and Data Quality Assessment for Childhood Poverty Prediction (https://arxiv.org/abs/2607.08281)
- **Prior Approaches**: 위성영상 기반 빈곤(아동 중증 결핍 비율) 예측은 DHS 설문 데이터를 지도학습 신호로 삼지만, 설문 유래 감독이 잡음·희소하고 이미지에 구름/손상 같은 품질 문제가 섞이기 쉽다. 또한 이미지 전용 모델은 좌표 같은 공간 구조를 명시적으로 다루지 못해 지역별 패턴과 일반화의 한계가 있었다. KidSat은 self-supervised 비전(예: DINOv2) 임베딩과 선형 회귀 헤드를 결합해 성과를 보였으나, 미세조정 타깃 구성의 희소성·품질선별 부재·공간정보 결여가 정확도를 제약했다.

- **Core Contribution**: 이 논문은 KidSat 파이프라인을 개선해 (1) DHS fine-tuning 타깃 매트릭스의 희소성 완화, (2) 2단계 이미지 품질 스크리닝, (3) 위성 임베딩에 Spherical Harmonics(SH) 기반 지리 인코딩을 융합하는 “원칙 있는” 레시피를 제안한다. 특히 SH 단독은 일관되게 성능을 끌어올렸고, 최종적으로 MAE를 0.2167(기준)→0.1759로 낮추며(상대 18.83%↓), 33개국 확장에서는 MAE 0.1658을 달성했다. 다만 SH+SIREN(좌표 기반 고용량 좌표 MLP)은 설계된 목적함수 없이 쓰면 오히려 SH 단독보다 약해질 수 있음을 실험적으로 보여준다.

- **Technical Challenges**: 핵심 난제는 (i) 고카디널리티 범주를 one-hot으로 넣을 때 발생하는 높은 희소성으로 인한 그라디언트 약화/정규화 불안정, (ii) 클라우드·센서 손상 관측을 임의 선택하면 생기는 잡음 주입, (iii) 좌표가 구면 위에 놓인다는 지리표현의 비유클리드성이다. 저자들은 DHS 코드북과 근거 기반 재집계로 one-hot 차원을 103→51(또는 48)로 줄이고, scan-line gap 탐지와 FMASK 계열의 물리 기반 클라우드 검출을 결합한 2단계 품질선별로 열화 관측을 제거/대체했다. 지리 정보는 SH로 고정 차원 다중 스케일 공간 priors를 만들고, (필요 시) SH+SIREN은 추가 실험을 통해 보완이 아닌 중복/겹침 가능성도 확인했다.

- **Empirical Impact**: 실험은 5-fold 교차검증으로 공정하게 비교했으며, 기준 대비 향상은 개선된 전처리+품질선별과 SH 인코딩이 함께 작동할 때 가장 크게 나타났다. 구체적으로 트리 기반 회귀 헤드가 fused 시각-지리 표현의 비선형 상호작용을 잘 활용해 Ridge나 MLP보다 유리했으며, LightGBM+SH에서 MAE 0.1759(±0.0016)로 최고 성능을 기록했다. 33개국(총 43,823 클러스터)으로 확장해 MAE 0.1658(±0.0005)을 달성한 점은, 공개 데이터만으로도 확장 가능한 사회경제 예측 파이프라인을 제공한다는 의미가 크다.



### Progression as Latent Drift: Generative Forecasting of Slow-Evolving Pathologies (https://arxiv.org/abs/2607.08270)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 생성 시퀀스 모델(확산/오토레그레시브 등)을 그대로 신경퇴행 예측에 옮기면, 시간에 따른 형태 변화 신호가 워낙 약해 학습이 기준선(현재 해부학) 재현이나 스캐너 잡음 같은 무의미한 변동에 끌립니다. 논문은 이 실패를 두 범주로 정리합니다. 하나는 기준선에 최적화돼 변화 학습이 멈추는 identity collapse이고, 다른 하나는 부드러운 네트워크가 국소 생물학적 드리프트와 잡음을 분리하지 못해 볼륨 전반에 가짜 변동을 퍼뜨리는 continuous interpolation trap입니다.

- **Core Contribution**: 이 논문은 느린 질환의 미래 해부학을 ‘Latent Drift’라는 진보적(progressive) 생성 프레임워크로 예측합니다. 핵심은 픽셀 단위 미래 영상을 생성하는 대신, 현재와 미래의 차이를 압축된 의미 표현(semantic latent drift)에서 학습하도록 목표를 바꾸는 것입니다. 또한 Finite Scalar Quantization(FSQ)으로 드리프트 표현에 topological dead-zone을 적용해 작은 고주파 잡음은 제거하고 구조적 진행은 보존합니다.

- **Technical Challenges**: 문제가 되는 기술 난점은 두 가지 동시 해결입니다: (1) 기준선 해부학 dominance 때문에 residual(변화) 학습이 붕괴되는 현상과, (2) Lipschitz-연속(부드러운) 예측기가 희소한 생물학적 변화의 지지(support)를 잡음 속에서 복구하지 못하는 수학적 한계입니다. 저자들은 각각을 겨냥해 ‘절대 상태 예측 대신 시간 잔차 예측’으로 identity collapse를 완화하고, FSQ의 불연속/비선형 임계 동작으로 continuous interpolation trap을 끊습니다. 그 결과, 예측이 매끈한 보간이 아니라 임계값을 넘는 의미 있는 변화 이벤트 중심으로 정렬되도록 설계됩니다.

- **Empirical Impact**: ADNI와 AIBL의 종단 3D 뇌 MRI에서 Latent Drift는 diffusion 및 autoregressive transformer 기반 기준선을 넘어 생성 충실도와 임상적으로 의미 있는 지표를 동시에 개선합니다. 특히 Diff-SSIM/NCC 같은 구조 일치 지표와 질병 분류기 기반 Downstream Clinical Utility에서 성능이 우수하며, patient-disjoint 조건 하에 통계적 유의성(p<0.05)도 확인됩니다. 장기(최대 4년) 추적에서 기준선 재현으로 수렴하는 경향을 피하고 개인별 위축 궤적을 안정적으로 따라가며, FSQ가 잡음 확산을 억제한다는 분석과도 일치합니다.



### UniRef-UAV: A Multimodal Benchmark for Universal Referring in UAV Imagery (https://arxiv.org/abs/2607.08267)
- **Prior Approaches**: 기존 UAV referring은 주로 text-only 질의와 단일 타깃(또는 존재가 있다고 가정되는) 출력에 맞춰 설계된 REC 패러다임을 따랐습니다. 이 때문에 reference image, multimodal instructions, no-target, 그리고 복수 인스턴스 등 실제 운용 조건을 함께 다루기 어렵다는 한계가 드러났습니다.

- **Core Contribution**: 논문은 UAV 시나리오에 맞춘 범용 referring 과제인 Universal Referring을 제안합니다. 질의 입력을 text-only, image-only, text+image로 확장하는 동시에, 입력 양식에 따라 출력 cardinality를 달리(텍스트는 no-to-many, text+image도 no-to-many, image-only는 존재 인지 기반 단일 인스턴스) 설계해 문제 틀 자체를 일반화했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “무엇을” 찾는지(semantic specificity)와 “몇 개가 정답인지”(variable cardinality) 요구가 질의 양식에 따라 달라지는데, 이를 하나의 모델이 일관되게 처리해야 한다는 점입니다. 논문은 UniRef-UAV 멀티모달 벤치마크와 함께, detection-style set prediction 기반의 UAV-URNet(UAV Universal Referring Network)으로 이들을 한 프레임워크에서 다루도록 구성했습니다.

- **Empirical Impact**: UniRef-UAV에서 UAV-URNet은 no-target 판별과 카드inality 제어 측면에서 재현성 높고 안정적인 베이스라인을 제공하며, 대형 범용 MLLM 대비 경량·재현성 이점을 강조합니다. 또한 시각 질의가 ambiguity를 줄이고 query–target alignment를 더 통합적으로 만들 수 있음을 도메인 분석·어블레이션으로 확인했으며, 관련 어노테이션/분할/코드 공개로 재현 연구를 촉진할 계획입니다.



### On the Design of Mixture-of-Experts for Dynamic Gaussian Splatting (https://arxiv.org/abs/2607.08250)
Comments:
          Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence

- **Prior Approaches**: 기존 dynamic Gaussian Splatting은 MLP 변형 네트워크, 다항식/보간 기반 모션 모델, canonical space(기준 공간) 변형 또는 비-canonical 궤적 파라미터화 등 다양한 변형 우도를 사용해 시간에 따른 외형을 재구성한다. 하지만 논문 분석에 따르면 장면(scene), 공간 위치(spatial region), 시간 프레임(frame)마다 성능이 크게 출렁이며, 이는 단일 변형 prior(유도 편향)에 기대는 구조적 한계로 설명된다. 즉 표현력이 부족해서가 아니라, 특정 변형의 우도가 이질적인 실제 동작을 충분히 포괄하지 못한다는 문제다.

- **Core Contribution**: 논문은 dynamic 3D Gaussian 표현에서의 핵심 문제인 multi-deformation modeling(다중 변형 모델링)을 “MoE(혼합 전문가) 관점”으로 재정의하고, 여러 변형 전문가를 하나의 표현 안에서 결합하는 설계 공간을 정리한다. 이를 위해 두 통합 제약(integration constraint)을 기준으로 MoDE(Mixture of Deformation Experts)와 MoE-GS(Mixture of Experts for Dynamic Gaussian Splatting)를 제안한다. MoDE는 shared canonical Gaussian 위에서 전문가를 joint optimization으로 직접 합치고, MoE-GS는 전문가를 분리 학습한 뒤 routing 단계에서 통합해 canonical 공유 제약을 제거한다.

- **Technical Challenges**: 다중 변형을 결합할 때 가장 큰 기술적 난제는, canonical Gaussian을 공유하는 경우 전문가들이 서로 다른 변형 prior 때문에 기준 공간의 학습이 불안정해질 수 있다는 점과, 반대로 전문가를 분리하면 라우팅이 잘 정의되지 않는다는 점이다. MoDE는 canonical 업데이트를 baseline 전문가에만 허용하고(다른 전문가의 canonical 쪽 그라디언트는 차단), 초기에는 random gating warm-up으로 특정 전문가로 쏠리는 현상을 완화해 joint 학습 안정성을 확보한다. 또한 spline 기반의 time-dependent gating과 Top-K sparse softmax로 시간 연속성을 유지하면서도 전문가 간 간섭을 줄인다.

- **Empirical Impact**: 실험에서는 기존 단일 변형 우도 방식이 장면/공간/시간에 따라 성능이 요동치는 현상이 단순 편차가 아니라 변형 prior에 의해 유도된다고 체계적으로 분석한다. 제안된 MoDE와 MoE-GS는 이 한계를 “전문가 결합 시점과 방식”의 차이로 설명 가능하게 만들고, 동일한 3D Gaussian 원시(primitive)라도 변형 전문가 구성에 따라 다른 운동 궤적과 재구성 거동이 나온다는 점을 보여준다. 코드 공개도 함께 제공되며, dynamic scene 재구성에서 MoE형 다중 변형 설계가 어떤 제약 하에서 더 적합한지(안정성/정확도/계산비용 트레이드오프) 실무적 선택지를 제시하는 데 의미가 있다.



### HSA: Hierarchical Slot Attention for Multi-granularity Scene-Decomposition (https://arxiv.org/abs/2607.08249)
- **Prior Approaches**: Slot attention 계열 OCL은 반복적 competitive attention으로 장면을 잠재 slot들의 평면적(단일 granularity) 집합으로 분해해 왔습니다. DINOv2 같은 self-supervised 특징으로 성능은 올렸지만, 주로 외형(appearance) 재구성에 맞춰져 category/instance 같은 의미 계층은 레이블 없이 자연히 생기기 어렵다는 한계가 있습니다. 기존 계층화 시도도 대체로 공간·외형 유사성 기반이라 human concepts에 정렬된 semantic hierarchy를 복원하기에는 부족합니다.

- **Core Contribution**: 이 논문은 최소한의 감독(학습 데이터 10%의 범주 분할)과 계층 정렬을 결합해 Hierarchical Slot Attention(HSA)으로 multi-granularity semantic scene decomposition을 한 모델에서 동시에 학습합니다. HSA는 holistic(전경/배경), semantic(범주), panoptic(개별 인스턴스) 세 수준을 각각의 slot attention 모듈로 생성하되, inter-level embedding consistency를 regularization으로 강제합니다. 또한 grouping purity와 attention containment로 ‘출력 마스크’가 아니라 표현 공간에 계층 구조가 인코딩되는지 직접 측정합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 외형 기반 slot이 만들어내는 분해를 의미 계층(범주·인스턴스)로 “접지(grounding)”하는 신호를 너무 많이 주지 않으면서, 세 수준의 정합성을 학습 안정적으로 확보하는 것입니다. HSA는 (1) 10% 라벨에서 level별 Dice 감독, (2) 연속 레벨(slot) 간 cosine similarity로 nearest coarse→fine 매칭을 만들고 straight-through estimator로 gradient를 흘리는 hierarchical alignment loss를 설계해 해결합니다. 추가로 각 레벨은 독립 디코더를 써 reconstruction 간섭을 줄이고, DINOv2 feature 재구성으로 collapse을 방지합니다.

- **Empirical Impact**: 실험에서 HSA는 COCO에서 가장 강한 flat baseline 대비 ARI를 holistic +41.5, semantic +14.6, panoptic +10.4만큼 개선하며, Pascal VOC에서도 더 큰 격차를 보였습니다. 단일 forward pass로 세 수준을 모두 제공해 파라미터/추론 비용을 크게 줄이면서도 성능은 유지·향상시키는 점이 강점입니다. grouping purity와 attention containment에서 HSA는 flat baseline 대비 semantic grouping을 유의미하게 더 잘 만들었고, 슬롯 기반 object recognition에서도 Top-1 38.9%, Top-3 63.8% 및 높은 slot recall(#BBox)을 달성해 downstream에도 유리함을 보여줍니다.



### SkelGen4D: Weakly-Supervised Skeleton-Based 4D Generation for Text-Driven Mesh Animation (https://arxiv.org/abs/2607.08246)
- **Prior Approaches**: 기존 4D 생성은 SDS 기반 최적화로 동적 3D를 만들거나(per-instance 최적화) 영상/텍스트에서 메쉬 변형을 직접 생성하는 방식이 많았다. 그러나 전자는 비용과 확장성이 낮고, 후자는 모션이 암묵적으로 표현돼 편집·제어가 어렵다는 한계가 있었다. 스켈레톤 기반 접근은 명시적 관절 구조를 제공하지만, 대체로 per-frame 스켈레톤 주석에 의존해 대규모 확장에 제약이 있었다.

- **Core Contribution**: 이 논문은 per-frame 스켈레톤 주석 없이도 텍스트 기반 메시 애니메이션에서 명시적 스켈레톤 모션을 생성하는 weakly supervised 프레임워크 SkelGen4D를 제안한다. SkelGen4D는 (1) 애니메이션 메쉬로부터 미분가능한 피팅을 통해 temporally consistent pseudo-skeleton을 복원하고, (2) 이 pseudo-skeleton을 학습해 feed-forward 방식으로 텍스트 조건 스켈레톤 시퀀스를 생성한다. 여기에 Motion-GRPO를 더해 시간적 일관성과 물리/관절 타당성을 강화한다.

- **Technical Challenges**: 핵심 기술 난제는 (a) 프레임마다 독립적으로 리깅하면 스켈레톤 토폴로지가 흔들려 일관된 모션 감독이 깨진다는 점, (b) 스켈레톤 생성 품질을 프레임 단위 손실만으로는 시퀀스 수준에서 보장하기 어렵다는 점이다. 이를 위해 Stage 1에서는 첫 프레임 리그의 스키닝 가중치/토폴로지를 고정하고 각 프레임 관절 변환만 gradient descent로 최적화해 pseudo-skeleton을 안정적으로 뽑는다. Stage 2에서는 관절 프레임을 latent으로 압축해 transformer로 auto-regressive 생성한 뒤, Motion-GRPO 보상(스무스함, 동역학, bone-length 일관성)으로 시간적 결함을 줄인다.

- **Empirical Impact**: 평가에서 Truebones Zoo와 Diffusion4D 두 대규모 벤치마크 모두에서 weakly supervised 스켈레톤 모델이 fully supervised 기준선을 동등하거나 상회하며, 텍스트 기반 메시 애니메이션의 품질과 다양성을 입증했다. 특히 Truebones Zoo에서는 커버리지와 motion fidelity 관련 지표에서 우수했고, AnyTop 대비 구조 안정성과 motion collapse 감소가 관찰됐다. Diffusion4D에서는 노이즈/불완전/불일치가 많은 기존 스켈레톤 대신 메쉬로부터 만든 temporally consistent pseudo-skeleton이 더 정제된 감독으로 이어져 video-driven 경쟁 방법들보다 안정적 생성을 보였다.



### Closing the Null Space: Guidance-Aware Quantization for Classifier-Free Diffusion (https://arxiv.org/abs/2607.08241)
Comments:
          6 pages, 5 figures, 3 tables

- **Prior Approaches**: 기존 post-training quantization(PTQ) 연구는 확산 모델을 보통 단일 패스(단일 분기)로 가정해 가중치/활성화 임계값을 맞추는 데 집중해 왔다. 특히 CFG의 핵심인 conditional/unconditional 두 분기 구조와, 두 분기의 선형 결합이 만드는 “가이드된 예측” 품질을 직접 제약하지 못해 구조적으로 실패할 여지가 있다. 또한 효율 평가지표(파라미터 수, BOPs)는 CFG의 2-pass 실행 오버헤드를 가려 실제 지연을 과대평가할 수 있다.

- **Core Contribution**: 이 논문은 CFG 모델에서 guidance gap만 보존하도록 PTQ를 최적화하면, unconditional 분기가 임의로 drift해도 gap 지표는 완벽히 맞출 수 있는 “branch-drift trap”이 생김을 정리했다. 즉 gap-fidelity 진단이 좋아 보여도, 실제 inference의 guided prediction은 망가질 수 있음을 보였다. 이를 막기 위해 Guidance-Aware Mixed Precision(GAMP)라는 PTQ 방법을 제안하며, guided prediction에 직접 캘리브레이션을 걸고 per-layer 활성 비트 민감도를 기준으로 비트 예산을 배분한다.

- **Technical Challenges**: 핵심 기술 난관은 (1) CFG의 2-pass 실행 구조가 효율 측정에 숨겨지는 시스템 병목을 만들고, (2) gap-only 목적함수가 남기는 null space 때문에 unconditional drift를 억제할 장치가 없다는 점이다. 논문은 guided prediction 기준의 per-layer 오류(가이드된 출력 열화)를 민감도 신호로 사용해 null space를 원천적으로 닫는 방식으로 해결하며, 그 다음 greedy knapsack으로 활성 비트를 컴퓨트 예산 안에서 배치한다. 또한 가중치는 상대적으로 고정하고(예: 4 bits), 활성화 정밀도를 계층별로 조절해 CFG에서 더 치명적인 “activation precision cliff”를 반영한다.

- **Empirical Impact**: 실험은 CIFAR-10에서 DASH student 모델을 대상으로 수행했으며, gap-only로는 ρ/cos(Delta) 같은 guidance-gap 진단은 최상인데도 FID가 가장 크게 나빠지는 false-positive 현상을 확인했다. 반면 GAMP는 동일 평균 활성 비트 예산에서 균일 정밀도 대비 훨씬 낮은 FID로 Pareto 개선을 보였고, guidance-output 민감도 기반 배분의 실용성을 입증했다. 또한 ONNX Runtime INT8은 TensorRT 미사용 등 소프트웨어 스택 제약으로 인해 BOPs 예측과 반대로 큰 지연이 발생함을 계측해, CFG PTQ 보고 시 효율 지표만으로는 부족하다는 메시지를 강화했다.



### TVTA: Trajectory-Aware Viseme-Guided Temporal Aggregation for Event-Based Lip Reading (https://arxiv.org/abs/2607.08236)
- **Prior Approaches**: 이벤트 기반 립리딩은 DVS 같은 이벤트 카메라의 높은 시간 해상도와 움직임 민감도를 활용해 발전해 왔지만, 대다수 파이프라인이 공간 인코딩/압축을 먼저 수행한 뒤에야 시간 모델링을 진행한다. 그 결과 희소하고 국소적인 운동 궤적이 조기에 평균화되며, 유사한 입 모양을 구분하는 데 중요한 미세 동적 단서가 약해질 수 있다. 또한 기존 시간 학습은 주로 단어 분류 목표에 의해 최적화되어, 내부 조음 구조에 대한 제약이 상대적으로 약하다는 한계가 지적된다.

- **Core Contribution**: 논문은 이벤트 스트림에서 로컬 운동 진화를 보존하는 시간 강화 프레임워크를 제안한다. 핵심은 (1) Trajectory-Aware Differential Aggregation(TDA)로 공간 위치별로 먼저 시간 정보를 학습한 뒤 적응적으로 공간을 통합해 압축 손실을 줄이고, (2) Viseme-Guided Aggregation(VGA)로 CTC 기반 viseme 중간 시퀀스 감독과 게이티드 집계를 결합해 단어 인식의 시간 표현을 더 구조적으로 학습시키는 것이다. 여기에 강한 이벤트 섭동에도 견고하도록 EMA teacher–student 일관성 정규화를 추가한다.

- **Technical Challenges**: TDA를 구현할 때의 기술적 난제는 ‘희소·국소 이벤트 운동’을 시간 모델링 단계로 끌어오기 위해 공간 압축을 늦추면서도 효율적으로 통합하는 방법이다. 이를 위해 각 공간 토큰에 대해 공유 BiMamba 블록으로 시간 스캔을 수행한 뒤, 전역 공간 문맥과의 differential response로 가중치를 산정하는 방식의 적응적 공간 집계를 설계해 로컬 궤적 신호를 보존한다. VGA에서는 프레임 정렬(viseme frame alignment)이 없는 상황에서 CTC로 viseme 중간 구조를 학습하고, 세그먼트 컨텍스트로 최종 temporal aggregation을 유도하는 게이팅을 결합해 단어 단위 분류의 제약만으로는 부족한 내부 구조 학습을 보완한다.

- **Empirical Impact**: DVS-Lip 벤치마크 실험에서 제안 방법은 student 기준 전체 정확도 77.49%를 달성해 기존 대비 향상 효과를 보였고, 특히 시각적으로 혼동되는 Acc1에서도 유의미한 성능을 유지한다. ablation 결과는 TDA, VGA, teacher–student 일관성 각각이 성능 향상에 기여하며, VGA 단독도 강한 이득을 제공하고 TDA와의 조합에서 추가 개선이 나타남을 보여준다. 정성적 디코딩에서는 CTC가 blank 제거 후에도 목표 단어의 조음 진행과 일관된 viseme 시퀀스 구조를 학습함을 확인해, 단순 보조손실이 아닌 의미 있는 시간 구조 학습임을 뒷받침한다.



### Multimodal 3D LUT Generation via StatLUT with Statistical Features for Photorealistic Style Transfer (https://arxiv.org/abs/2607.08227)
Comments:
          17 pages, 9 figures, 7 tables. Preprint

- **Prior Approaches**: 기존 PST는 인코더-퓨즈-디코드(encoder-decoder) 구조를 쓰는 경우가 많지만, 사전학습 인코더가 의미(semantic)를 강하게 뽑아내 PST의 저수준 색 분포 요구와 기계적 불일치가 생긴다. 그 결과 색-구조가 얽혀 국소 왜곡, 질감/콘트라스트 저하, 그리고 RGB 기반 매핑의 불완전성으로 인한 색 밴딩이 나타난다. LUT 방식도 점별(point-wise) 매핑이나 선형 블렌딩(basis-LUT blending)에 머물러 색 격자(topology)를 충분히 보존하지 못하거나, 비디오/멀티모달에서는 일관성·제어성이 약하다는 한계가 있다.

- **Core Contribution**: StatLUT은 3D LUT 생성 중심의 멀티모달 PST 프레임워크로, 색 분포와 구조 의미를 분리해 “artifact-free” 목표를 재정의한다. 핵심은 CIE Lab 공간에서 위치 비의존(spatially-agnostic) 통계 특징(명도 1D 히스토그램, 크로마 2D 히스토그램, 결합 상관 맵)을 추출해 의미 얽힘을 차단하는 Lab-Extractor를 도입한 점이다. 또한 LUT 예측을 Transformer 기반 Seq2Seq로 만들고 MR-Mapper로 3D LUT 격자 토폴로지를 전역적으로 매끄럽게 학습하며, H-Diffuser로 텍스트 프롬프트만으로도 해당 통계 특징을 생성해 text-driven 컬러 그레이딩을 구현한다.

- **Technical Challenges**: 가장 어려운 기술 과제는 (1) 스타일 이미지에서 “순수한” 색/조명 분포만 뽑아 구조적·의미적 간섭을 제거하는 것, (2) LUT의 색 격자 토폴로지를 보존해 색 밴딩·시간 깜빡임을 막는 것, (3) 텍스트 조건에서 LUT의 물리적 제약(히스토그램 비음수성 등)을 만족시키며 안정적으로 생성하는 것이다. StatLUT은 Lab-Extractor에서 Lab 변환과 비선형 stretching, soft-binning을 통해 희소한 크로마 공간의 표현을 강화하고, Seq2Seq의 MR-Mapper가 잔차(residual)와 전역 제약을 함께 학습하게 하여 토폴로지 매끄러움과 최적화를 동시에 노린다. 텍스트 경로에서는 diffusion 모델이 노이즈를 예측하는 대신 “clean 토큰(통계 특징)”을 직접 생성하는 H-Diffuser(X0-prediction)와 밀도 인식(density-aware) 손실을 사용해 희소 히스토그램에서 과도한 평활화를 억제한다.

- **Empirical Impact**: 실험에서는 PST50, PhotoNAS 등 벤치마크에서 StatLUT이 시각 품질과 정량 지표 모두에서 SOTA를 상회하며, 특히 NLUT의 색 밴딩·콘트라스트/텍스처 손실, Neural Preset의 전반적 선명도 저하, D-LUT/SA-LUT의 스타일 불일치 같은 문제를 일관되게 줄였다는 점이 강조된다. 50명 사용자 연구에서도 Top-1 선호 70%로 가장 높은 만족도를 보이며, 정량적으로도 이상적 위치에 가까운 균형(스타일 유사도↑, 콘텐츠 보존↑)을 달성했다. 또한 스타일 패치 셔플링 실험으로 공간 의미를 깨도 색 매핑 잔차가 거의 0에 가깝게 유지되어, Lab-Extractor 기반 분리 아이디어가 실제로 작동함을 강하게 입증한다.



### LUMI: Tokenizer-Agnostic LLM-Based Lossless Image Compression (https://arxiv.org/abs/2607.08221)
Comments:
          Preprint

- **Prior Approaches**: 기존 LLM 기반 무손실 이미지 압축은 픽셀 값을 LLM의 텍스트 입력으로 바꿔 토큰열을 만들고, 어휘(vocabulary) 헤드의 로짓을 확률로 사용해 산술부호화의 codelength을 줄이는 방식이 주를 이뤘다. 예를 들어 P2-LLM은 픽셀 숫자를 텍스트 토큰으로 직렬화한 뒤 다음 픽셀 값을 예측하도록 하고, LoRA 같은 parameter-efficient tuning으로 적응한다. 하지만 이 접근은 토크나이저마다 숫자 분할이 달라지는 ‘토크나이저 의존성’ 때문에, 같은 픽셀 심볼이라도 이벤트 정의와 확률 사건이 바뀌며 모델 간 재사용성이 떨어진다는 한계가 있었다.

- **Core Contribution**: 이 논문은 LUMI(LlM-based Unified Model-agnostic lossless Image compression)라는 토크나이저 비종속(tokenizer-agnostic) 프레임워크를 제안한다. LLM 백본을 고정한 채, 픽셀 값을 텍스트 토큰 대신 픽셀 임베딩 모듈로 연속 임베딩 공간에 사상하고, 출력은 256개 픽셀 알파벳에 대한 256-way 분포를 별도 prediction head로 예측해 산술부호화에 그대로 쓰게 한다. 이를 통해 LLM 기반 무손실 압축을 ‘토크나이저의 언어기호 모델링’이 아니라 ‘픽셀 공간 어댑테이션을 통한 frozen foundation model 활용’으로 정식화한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트 토크나이즈를 제거한 상태에서 픽셀의 채널/강도 정보를 LLM이 의미 있게 처리하도록 입력 표현을 설계하는 것과 (2) flatten 이후 사라진 2차원 공간 구조를 확률 예측에 반영하는 것이다. LUMI는 강도 정규화와 다항/사인-코사인 특징, 그리고 채널 one-hot을 결합한 픽셀 임베딩(PixEmb)으로 입력을 구성하고, flatten 뒤에는 intra-patch position encoding(INP)을 통해 행/열 좌표 정보를 보강한다. 또한 causality 유지를 위해 shifted prefix를 사용하고, 텍스트 프롬프트와 soft-prefix 파라미터만 학습하며 LLM 디코더 백본은 frozen 상태로 유지해 모델족 간 공용 인터페이스를 달성한다.

- **Empirical Impact**: 실험에서는 LLaMA, Qwen, Gemma 백본 위에서 자연/의료/원격탐사 벤치마크(Kodak, BRACS, BED4RS 등)를 대상으로 압축 성능과 교차 도메인 견고성을 비교했다. 결과적으로 LUMI는 토크나이저 기반 LLM 압축 베이스라인에 견주어 경쟁력 있는 압축률을 보이면서도, 토크나이저 의존성을 제거한 효과로 도메인 전이에서 더 나은 robustness를 보였다고 보고한다. 즉, 무손실 이미지 압축에서 LLM을 범용 ‘컨텍스트 엔트로피 모델’로 재사용할 수 있음을 실증하며, 모델 계열이 달라도 동일한 압축 파이프라인 인터페이스를 붙일 수 있는 확장성을 제시한다.



### Benchmark Evaluation of Feredated Learning on Multi-organ Images (https://arxiv.org/abs/2607.08219)
- **Prior Approaches**: 기존의 연합 의료 영상 벤치마크(FedCBD, Flamby 등)는 주로 단일 장기나 단일 모달리티 데이터에 초점을 맞추며, 최신 알고리즘을 충분히 반영하지 못했다. 또한 평가가 정확도 중심으로 치우쳐 실제 임상 환경에서 중요한 효율성과 개인정보 보호 같은 트레이드오프를 종합적으로 보기 어렵다. 그 결과 서로 다른 FL 방식의 성능과 강건성을 공정하게 비교하기가 제한된다.

- **Core Contribution**: MobenFL은 연합 의료 영상 분류를 위한 통합 벤치마크로, 20개의 최신 FL 알고리즘과 22개 의료 영상 데이터셋을 묶어 12개 장기를 포괄한다. 단일 평가 축에 그치지 않고 성능뿐 아니라 알고리즘 효율과 개인정보 보호 지표를 함께 체계적으로 측정하도록 설계했다. 아울러 질병, 모달리티, 영상 장치가 다른 복잡한 임상 시나리오별 별도 평가도 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 의료 데이터의 극심한 이질성(장기·기기·모달리티 차이) 속에서 알고리즘을 공정하고 재현 가능하게 비교할 ‘통합 평가 프레임’이 없다는 점이다. 연구진은 알고리즘을 추가 네트워크 구조, 정규화 기반, 집계 전략, Split Learning 계열의 4개 범주로 정리하고, 모든 알고리즘을 하나의 통일된 벤치마크 프레임워크와 표준 입력 인터페이스로 묶어 평가 일관성을 확보했다. 이를 통해 최신 FL 연구 흐름과 실제 임상 다양성을 동시에 반영하려 했다.

- **Empirical Impact**: MobenFL은 다양한 장기·모달리티 조합을 동일한 프로토콜로 시험함으로써 FL 알고리즘이 임상 환경에서 보일 일반화와 강건성 차이를 더 명확히 드러낸다. 또한 정확도 외에 통신/계산 같은 효율성과 프라이버시 보호 역량까지 포함해, 실제 배포 관점에서의 비교 기준을 확장한다. 의료 영상 FL 연구가 ‘좋은 성능’에서 ‘신뢰할 수 있는 임상 적용’으로 옮겨가는 데 필요한 공용 평가 표준을 제공한다.



### Metrics or Mirage? An Audit of Evaluation Inconsistencies in Colonoscopy Polyp Segmentation Benchmarks (https://arxiv.org/abs/2607.08203)
Comments:
          Submitted to ECCV Workshops

- **Prior Approaches**: 대장내시경 용종 분할 연구는 주로 공개 벤치마크 리더보드에서 Dice/IoU 중심의 6개 지표(예: PraNet 템플릿)를 반복 재사용하며 성능 향상을 주장해 왔다. 그러나 이 템플릿은 경계 정확도(작고 편평한 병변에서 치명적)와 병변 단위 검출 관점, 그리고 임상 안전성 지표(재현율)를 충분히 반영하지 못한다.

- **Core Contribution**: 본 논문은 2015~2026년 27편을 체계적으로 감사해, 평가 관행이 검증 가능성을 훼손하는 구조적 문제를 드러낸다. 특히 Hausdorff distance(HD95)를 거의 보고하지 않고, 학습/테스트 분할이 서로 호환되지 않으며, 통계적 유의성 검정도 대부분 누락되어 리더보드의 ‘비교’가 사실상 성립하지 않는다고 주장한다.

- **Technical Challenges**: 이 기여를 실현하기 위해 논문은 대표 모델 5개를 단일한 통합 채점기로 재평가하고, 고정 분할/랜덤 분할/OOD(PolypGen 다기관) 등 3개 통제 프로토콜로 성능을 비교한다. 또한 Metrics Reloaded에 맞춰 누락되던 NSD@τ(표면거리 계열), HD95, Recall(민감도), 병변 단위 검출 지표를 포함해 “지표 선택이 순위를 바꾸는지”를 per-image 유의성까지 확인한다.

- **Empirical Impact**: 결과적으로 Dice 중심 평가는 경계 및 재현율 실패를 가려 큰 크기의 결함이 숨겨졌고, 데이터 분할이 달라지면 ‘최고 모델’이 바뀌며 근접 순위도 무작위 분할에서 역전될 수 있었다. 또한 OOD 다기관 테스트에서는 모든 모델이 절대 성능이 크게 하락했지만, 리더보드(인-디스트리뷰션)에서는 이런 열화가 잘 드러나지 않았다. 논문은 이를 교정하기 위한 5개 항목의 Polyp Segmentation Reporting Checklist(PSRC)를 제안하며, 특히 HD95·Recall·분할 프로토콜 명시·외부/OOD 테스트 포함을 필수로 권고한다.



### TMI: Text-to-Image Meets Image-to-Image for Complementary Data Synthesis to Boost Long-Tailed Instance Segmentation (https://arxiv.org/abs/2607.08201)
Comments:
          Accepted to ECCV 2026. The first two authors contributed equally to this work

- **Prior Approaches**: 대규모 어휘 instance segmentation(LVIS 등)은 장꼬리(long-tailed) 분포로 인해 희귀 범주에서 데이터 부족과 미세한 클래스 간 모호성이 동시에 발생한다. 이를 완화하려는 re-weighting, balanced sampling, classifier calibration 같은 방법은 분포 불균형은 줄이지만 희귀 범주의 ‘실제 데이터 희소성’ 자체를 해결하긴 어렵다. 생성 기반합성은 T2I, L2I, I2I로 나뉘는데, T2I는 pseudo-label 잡음(특히 희귀/세분 범주) 문제가, L2I는 마스크-이미지 불일치 및 대규모 범주 확장 한계가, I2I는 copy-paste의 맥락 부자연스러움(도메인 갭)이나 inpainting의 자연스러운 배치 난점이 있다.

- **Core Contribution**: 논문은 T2I 생성의 범주·장면 다양성과 I2I 편집의 맥락 현실감을 결합한 하이브리드 합성 파이프라인을 제안한다. T2I에서는 프롬프트에 포함된 범주만 남기는 prompt-consistent filtering과 teacher-student로 pseudo-label 신뢰도를 높이고, 희귀 범주 강화를 위해 VRAIN(Verified Rare-class Augmentation via INstructed editing)이라는 새로운 I2I 에디터를 도입한다. VRAIN은 ‘place-and-verify’ 방식으로, instruction 기반 편집으로 자연스러운 위치에 희귀 인스턴스를 삽입한 뒤 의미 일관성과 시각적 충실도를 검증해 정확한 인스턴스 단위 주석을 생성한다.

- **Technical Challenges**: 핵심 난제는 (1) 생성 도메인 갭으로 인한 pseudo-label 오염과 (2) 희귀 범주에 대해 자연스러운 맥락 통합 및 정확한 배치·주석을 동시에 달성하는 문제다. 이를 위해 T2I 라벨은 offline 텍스트 일관성 필터링 후 teacher의 EMA 업데이트로 온라인에서 점진적으로 정제하며, prompt에 맞지 않는 예측은 분류 손실에서 제외하되 localization 손실에는 일부 활용해 학습 신호를 유지한다. I2I 라인에서는 VLM 기반으로 적합한 희귀 범주와 배치 instruction을 선정하고, SSIM 차이로 편집 영역을 찾은 뒤 open-vocabulary 검출 후보를 VLM으로 ‘박스 기반 yes/no’ 검증하여 확인된 인스턴스만 SAM으로 마스크를 추출·합성한다.

- **Empirical Impact**: LVIS에서 기존 paste 기반 증강과 T2I 베이스라인을 넘어, 전체 AP는 최대 +4.0, 희귀 범주 AP는 최대 +9.5만큼 개선되는 성능을 보고한다. 또한 백본이 커질 때 효과가 안정적으로 확장되어(스케일링) 데이터 합성의 실용성을 시사한다. 결과적으로 희귀 범주 성능 저하의 주요 원인으로 지목된 라벨 신뢰도·맥락 부자연스러움을 함께 완화해, 장꼬리 어휘 instance segmentation의 학습 효율을 실증적으로 끌어올렸다는 점에서 의미가 있다.



### Unpaired Joint Distribution Modeling via Multi-Scale Image Representations (https://arxiv.org/abs/2607.08198)
- **Prior Approaches**: 페어( x, y )가 부족한 상황에서 unpaired 학습은 마진만으로 joint distribution을 추정해 가짜 페어를 만들려는 시도로 발전해 왔습니다. 기존에는 cycle-consistency, discriminative/contrastive 정규화, latent factor disentanglement 같은 휴리스틱이 많았고, 대체로 이들이 pseudo-paired 생성의 안정성과 일관성 측면에서 이론적 근거가 약하다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 마진 관측만으로 joint distribution을 학습하는 문제를 latent-variable 확률 그래프 모델로 정식화한 LUD-MSR를 제안합니다. auxiliary representation h_x, h_y를 두어 두 도메인의 암묵적 결합을 매개하고, marginal 데이터만으로 ELBO를 최적화해 분포 근사 오차의 상계를 이론적으로 도출합니다.

- **Technical Challenges**: 핵심 난제는 inference invariance를 높이면(도메인 일치) 정보가 손실되고, 정보 보존을 늘리면(고주파/세부 유지) 도메인 일치가 약해지는 trade-off가 생긴다는 점입니다. 이를 완화하기 위해 MSR(Multi-Scale image Representation) 매핑으로 저주파(구조적 유사성)는 살리되 도메인 특이 변이를 억제하도록 표현 공간을 설계하고, 그 균형이 기존 방식보다 더 유리함을 보입니다.

- **Empirical Impact**: 실험에서는 실세계 denoising 벤치마크와 함께 cryo-EM에서의 이미지 denoising에 대해 가짜 페어 합성이 실제 성능으로 연결됨을 보였습니다. 특히 MSR 기반 표현이 noise modeling의 도메인-정보 trade-off를 더 잘 조절해 기존 방법 대비 개선된 결과를 얻는 것으로 보고됩니다.



### Dive Into the Implicit Biases of Low-rank Vision-language Alignmen (https://arxiv.org/abs/2607.08194)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 비전-언어 얼라인먼트(vision-language alignment)는 사전학습된 비전 인코더를 LLM에 연결한 뒤, 정렬 단계에서 LLM 전체 파라미터를 업데이트하는 방식으로 자주 다뤄져 왔다. 그러나 저자들은 이 접근이 사실상 supervised fine-tuning 성격의 문제이며, low-rank adaptation 같은 기법이 설계된 학습 레짐과 더 잘 맞는다고 지적한다.

- **Core Contribution**: 논문은 얼라인먼트 단계에서 LLM에 low-rank adaptation을 적용하는 low-rank alignment를 체계적으로 탐구하며, 계산 비용을 줄이면서도 다수 벤치마크에서 full-parameter alignment보다 성능이 높음을 보인다. 또한 그 효과를 단순 성능 비교를 넘어, low-rank가 만들어내는 암묵적 편향과 시각 특징 공간의 구조 보존 현상을 설명한다.

- **Technical Challenges**: 핵심은 “왜” low-rank가 더 낫고 “어떤 편향”이 생기는지 정량화하는 것이다. 저자들은 행동 수준(환각/보수성)과 특징 수준(토큰별 선형 분리성), 기하 수준(각도 커버리지·매니폴드 균질성)을 함께 분석하고, 이 과정을 뒷받침하는 이론적으로는 UFM(Unconstrained Feature Model) 하에서 flat gradient 및 잡음에 견고한 방향을 선호하게 되는 두 개의 정리(잡음 가중 스무딩, 정상상태 하위공간 집중)로 설명한다.

- **Empirical Impact**: 실험은 1.4B~14B 스케일과 LoRA/LoHa/LoKr 등 100개 이상의 정렬 설정(연산자·랭크·시각 인코더/동결 여부·학습 스케줄 등)을 포괄하며, 저랭크 정렬이 대부분의 과제에서 일관되게 우수함을 확인한다. 특히 LS-curse로 요약되는 full-parameter alignment의 토큰 선형 분리성 붕괴를 low-rank alignment가 완화하며, 전체 정렬이 entity-level 의미를 조기 융합하는 대신 modality-specific 구조를 보존해 보수적 행동으로 이어진다는 해석을 제공한다.



### Dual-Correlation Hypergraph Network for Unaligned RGBT Video Object Detection and A Large-scale Benchmark (https://arxiv.org/abs/2607.08191)
- **Prior Approaches**: 기존 RGBT VOD(Video Object Detection) 연구는 RGB와 thermal을 결합해 저조도·악천후에 강인한 탐지를 노리지만, 대다수 방법이 RGBT 입력이 “잘 정렬(aligned)”돼 있다는 가정에 기대는 편이다. VT-VOD50은 수동 정렬을 수행했지만 여전히 공간 불일치가 남아 성능 저하 요인이 되며, 규모와 장면 다양성도 제한적이다. EINet, PTMNet 등은 노이즈 억제나 early/middle fusion 같은 융합 전략을 제안했으나, 약한 정렬 문제를 충분히 모델링하지 못했다.

- **Core Contribution**: 이 논문은 공간 불일치를 완화하는 Dual-Correlation Hypergraph Network(DHNet)를 제안한다. Patch-based Spatial Alignment Module(PSAM)로 지역 단위 정렬을 먼저 수행한 뒤, Dual Hypergraph Fusion Module(DHFM)에서 시간 상관(연속 프레임)과 교차 모달 상관을 하이퍼그래프로 동시에 학습해 고차 의존성을 기반으로 융합한다. 또한 평가용 대규모·다장면 벤치마크인 DVT-VOD1000을 구축해 학습·검증의 기반을 확장한다.

- **Technical Challenges**: 핵심 기술적 난제는 RGBT 센서 간 관점·설치 차이로 인해 지역적으로 정렬이 달라지는 “weak alignment”를 단순 전역 보정으로 해결하기 어렵다는 점이다. PSAM은 패치 단위로 affine 변환을 예측해 thermal 특징을 RGB 특징에 순차 정렬하고, LBP로 전역 위치 단서를 thermal에 주입해 지역 변형의 불완전성을 보완한다. 그다음 DHFM은 temporal branch와 multimodal branch를 하이퍼그래프 형태로 구성해, pairwise 관계가 아닌 고차 상관을 message passing으로 반영하도록 설계한다.

- **Empirical Impact**: 실험은 VT-VOD50과 DVT-VOD1000에서 진행됐으며, DHNet-L은 DVT-VOD1000에서 AP50 31.7%로 경쟁 방법 대비 성능 격차(예: EI2Det 30.2%)를 확보했다. 효율 버전인 DHNet-S도 AP50 28.5%와 73 FPS를 함께 달성하며 파라미터·연산 부담 대비 정확도 우위를 보여준다. 또한 VT-VOD50에서도 DHNet-L이 AP50 57.5%를 기록해 비디오 전용 탐지기들을 능가하며, cross-modal 융합 설계가 시간 정보만으로는 얻기 어려운 이득임을 시사한다.



### Leveraging Color Naming for Image Enhancemen (https://arxiv.org/abs/2607.08185)
Comments:
          Project page: this https URL. arXiv admin note: text overlap with arXiv:2407.09892

- **Prior Approaches**: 기존 학습 기반 이미지 보정은 원시(raw)–전문가 편집 쌍 데이터를 사용해 스타일을 모사하지만, 결과가 개인 취향이나 사용 맥락과 어긋날 수 있습니다. 또한 다중 3D LUT나 end-to-end U-Net 계열은 해석 가능성과 사용자 조절(인터랙션)이 제한적이라는 지적이 있습니다. 톤 커브로 어느 정도 설명성을 주는 방법도 색/공간을 섞거나 결합하는 방식 때문에 “색 이름 단위로 무엇이 어떻게 바뀌는지”가 불명확해지기 쉽습니다.

- **Core Contribution**: NamedCurves+는 Color Naming(색 이름) 분해를 활용해 이미지를 6개 색 맵으로 나누고, 각 색 맵에 대해 Bezier-parametrized tone curve를 적용하는 프레임워크를 제안합니다. 전역(색 이름별) 톤 조정은 사용자가 특정 색의 커브를 직접 수정해 원하는 결과로 유도할 수 있게 만들며, 이를 통해 해석 가능성과 인터랙션을 동시에 강화합니다. 여기에 트랜스포머 기반 fusion으로 6개 전역 편집을 결합해 전문가의 스타일에 가까운 국소·공간 의존 편집 효과까지 재현하려고 합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 카메라/조명 차이로 인해 색 이름 분해가 흔들릴 수 있다는 점과 (2) 전역 커브 편집만으로는 전문가가 하는 국소 조절을 충분히 만들기 어렵다는 점입니다. NamedCurves+는 UNet-like backbone과 attention(CBAM)으로 입력을 canonical 공간으로 표준화해 색 분해의 일관성을 확보하고, 전역 커브 6장을 만든 뒤 transposed attention 기반 트랜스포머 fusion으로 공간적 의존성을 모델링합니다. 또 기존 fusion의 색 경계 halo 같은 취약점을 줄이기 위해 채널 간 교차(cross-covariance) 형태의 transposed attention과 효율적인 피처 처리 설계를 사용합니다.

- **Empirical Impact**: MIT-Adobe-5K, PPR10K, MSEC, SICE의 image retouching, tone mapping, exposure correction 전 범위에서 NamedCurves+가 state-of-the-art를 정량·정성적으로 앞선다고 보고됩니다. 특히 tone curve가 색 이름별로 명시적으로 표현되어 어떤 색이 어떻게 조정되었는지 추적이 가능하고, 사용자가 커브를 수정해 개인화된 결과를 얻을 수 있다는 점을 실험 맥락에서 강조합니다. 결과적으로 ‘설명 가능 + 사용자 제어 + 고성능’의 조합을 제공하는 범용 이미지 enhancement 프레임워크로서 의미가 큽니다.



### LEEVLA: Seeing What Matters in Latent Environment Evolution for Vision-Language-Action (https://arxiv.org/abs/2607.08182)
- **Prior Approaches**: 기존 Vision-language-action(VLA) 모델은 시각 토큰을 대체로 동일하게 다루고, 인간이 고른 요인에 의존해 추론하는 경우가 많아 동적·복잡한 상황에서 핵심 근거를 충분히 강조하지 못한다. 그 결과 작업에 결정적인 증거는 놓치고, 불필요한 배경 요인에 영향을 받는 문제가 반복된다는 점이 한계로 지적된다.

- **Core Contribution**: 이 논문은 LEEVLA로, 잠재 세계표현의 구조적 진화를 유지하면서 “무엇을 봐야 하는지”를 명시적으로 유도하는 VLA 아키텍처를 제안한다. 또한 훈련 단계에서 작업 증거에 집중하도록 장치(DGDP)와, 그 증거가 잠재공간에서 어떻게 “진화”해야 하는지 장치(SFFG)를 결합해 task-aware where-how 학습 프레임워크를 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작업의 지시와 관련된 시각 영역을 학습 중 어디에 우선 집중할지 정하고, (2) 그 집중된 특징들이 잠재공간에서 일관된 방식으로 시간에 따라 진화하도록 구조를 강제하는 것이다. 이를 위해 drift-guided dynamic prioritization(DGDP)은 dynamic position prioritization(DPP)과 semantic drift guidance(SDG)를 결합해 주의 위치를 안내하고, structured feature flow generation(SFFG)은 prototype-to-periphery(P2P) 예측으로 우선 특징의 잠재 진화 경로를 모델링한 뒤 mutual-neighborhood contrastive(MC)로 이웃(topology) 일관성을 유지한다.

- **Empirical Impact**: VLA 벤치마크 전반에서 LEEVLA는 기존 방법을 일관되게 능가하며, 작업 증거를 명시적으로 유도하는 접근과 구조화된 잠재 추론이 확장 가능한 VLA에 필수라는 결론을 뒷받침한다. 특히 dynamic 시나리오에서의 강건성이 강화된 점이 실증적으로 확인되어, VLA 연구에서 “where-to-attend”와 “how-to-evolve”를 함께 설계하는 방향에 의미 있는 자극을 준다.



### Attention-Based Segmentation of WMHs and Differentiation of Vascular vs. Demyelinating Lesions (https://arxiv.org/abs/2607.08171)
- **Prior Approaches**: WMH는 FLAIR에서 뚜렷하지만, 혈관성 병변과 염증성 탈수초 병변이 형태·위치가 겹쳐 감별진단이 어렵다. 기존 연구는 주로 분할(segmentation) 정확도 개선이나 CNN 기반 특징 추출에 치우쳐 임상적으로 해석 가능한 분류로의 연결이 제한적이었다. 또한 관심 모듈(attention)을 WMH 분할에 체계적으로 적용한 검증이 부족했다.

- **Core Contribution**: 이 논문은 attention 기반 분할(Attention U-Net에 BAM/CBAM 통합)과 분할 마스크에서 추출한 형태학적 특징으로 혈관성 vs 탈수초성 WMH를 분류하는 2단계 파이프라인을 제안한다. 분할은 병변의 위치·경계를 정량화하고, 이후 분류는 병변 크기·형상·분포 같은 방사선학적 근거를 특징으로 삼아 해석가능성을 높인다. 다중 스캐너·서로 다른 프로토콜의 5개 공개 데이터셋을 사용해 일반화 가능성도 함께 점검한다.

- **Technical Challenges**: 핵심 과제는 (1) 제한된 데이터에서 attention 모듈이 실제 병변 경계와 잡음 억제에 도움이 되는지, (2) 2D/patch/2.5D 입력 중 국소 문맥을 어떻게 활용할지, (3) 분할 오류가 분류 성능을 얼마나 흔드는지였다. 저자들은 2D 슬라이스, patch-based 학습, 2.5D(인접 슬라이스 3장) 비교와 함께 BAM/CBAM을 인코더·디코더·병목에 배치해 실험했고, 분류에는 lesion의 connected component 기반 형태학적/위치 특징을 표준화 후 Random Forest·SVM·Logistic Regression으로 학습했다.

- **Empirical Impact**: 분할에서는 2D 슬라이스 기반이 patch 및 2.5D보다 전반적으로 우수했고, 네트워크 전반에 attention 모듈(BAM+CBAM)을 확대한 Attention U-Net 변형이 전역 평균 지표에서 강세를 보였다. 반면 병변이 존재하는 데이터셋에서는 Attention U-Net이 더 높은 Dice·Recall로 병변 민감도를 높였다. 분류는 분할 마스크를 써도 정답 마스크 대비 성능 저하가 크지 않았고, SVM과 Random Forest가 안정적이었으며 탈수초성은 혈관성보다 더 어려웠다(마스크 기반에서 더 민감). 저자들은 대규모 임상 코호트에서의 추가 검증이 필요하다고 결론내리며, attention 분할+형태 특징 분류 조합이 임상 감별 방향으로 유망하다고 제시했다.



### Continual Test-Time Adaptation in Computer Vision: Methods, Benchmarks, and Future Directions (https://arxiv.org/abs/2607.08164)
Comments:
          TMLR 2026

- **Prior Approaches**: 기존에는 학습 시점에 데이터 분포가 동일하다는 i.i.d. 가정을 전제로, 오프라인에서만 견고성(robustness)을 확보하는 방식이 주로 쓰였다. Domain Generalization(DG)이나 Domain Adaptation(DA)는 배포 시점에 모델이 더는 적응하지 않는 frozen 모델로 남는 한계가 있고, DA는 소스 데이터·타깃 데이터 동시 접근이 필요해 현실 제약을 받기 쉽다. Test-Time Adaptation(TTA)은 라벨 없는 타깃 스트림으로 온라인 적응을 시도하지만, 실제 배포처럼 비정상(non-stationary) 환경이 연속적으로 바뀌는 경우엔 실패 양상이 누적될 수 있다.

- **Core Contribution**: 이 논문은 Continual Test-Time Adaptation(CTTA)을 엄밀히 문제 정의하고, 배포 중 분포가 시간에 따라 변하는 양상에 맞춘 평가 프로토콜을 체계적으로 분석한다. 또한 기존 방법들을 최적화 기반(엔트로피 최소화, pseudo-label링, parameter restoration), 파라미터 효율형(정규화 레이어 적응, adaptive parameter selection), 아키텍처 기반(teacher-student, adapters, visual prompting, masked modeling) 등 3개 계열의 계층적 분류로 정리한다.

- **Technical Challenges**: CTTA의 핵심 기술 난제는 두 가지 실패 모드로 정리된다: 장기 업데이트에서 소스 지식이 잠식되는 catastrophic forgetting과, 잡음이 섞인 pseudo-label 등으로 인한 error accumulation이다. 논문은 라벨이 없고 소스 데이터도 없는 상황에서 배치당 제한된 계산으로 self-training 목적을 설계·안정화해야 한다는 점을 강조하며, 계열별로 손실/업데이트의 안정성을 확보하는 방향(예: 복원/제약, 업데이트 선택, 구조적 완충)을 정리한다.

- **Empirical Impact**: 서베이는 CIFAR-10-C, CIFAR-100-C, ImageNet-C 같은 표준 벤치마크의 Continual Structured Change(CSC) 및 더 동적인 변형들(예: 점진 전이) 위에서 비교 결과를 정리해 방법 간 성능 차이를 보여준다. 결과적으로 CTTA가 단일 타깃 적응 TTA보다 더 현실적인 스트리밍 환경에서 모델 신뢰성과 장기 안정성에 중요한 과제임을 부각하며, 향후 foundation model/black-box 시스템으로의 확장을 위한 로드맵도 제시한다.



### ProsMAE: Multi-Source MAE Pretraining for ISUP Grade Classification (https://arxiv.org/abs/2607.08162)
Comments:
          Accepted to APCCAS 2026

- **Prior Approaches**: WSI는 진단에 필요한 세부 형태 정보를 담지만, 기가픽셀 규모와 스캐너/염색 차이, 조직 아티팩트 때문에 end-to-end 학습보다 타일 단위 전처리와 self-supervised 사전학습이 주로 활용된다. 기존에는 single-source MAE나 표준 AE/VAE 같은 재구성 기반 방법이 많이 쓰였고, 도메인 변이가 있을 때 표현이 특정 데이터셋 편향을 보일 수 있다는 한계가 남았다. 또한 ISUP 그레이딩은 ordinal 성격이라 성능 평가가 단순 정확도보다 불일치 정도를 반영하는 지표가 더 중요하지만, 분할 구성에 따른 재현성 문제도 제기돼 왔다.

- **Core Contribution**: 이 논문은 여러 출처의 병리 타일로 MAE를 학습하는 ProsMAE와, 학습된 인코더를 고정한 채 선형 헤드로 ISUP grade를 예측하는 ProsCLS를 제안한다. ProsMAE는 PANDA(전립선), CAMELYON17(림프절 전이), BRACS(유방 아형) 타일을 함께 사용해 서로 다른 조직 형태와 획득 조건을 인코더가 보게 만든다. 그 결과 disjoint PANDA split에서 vanilla MAE frozen linear-probe 대비 평균 validation QWK를 개선했다.

- **Technical Challenges**: 다중 임상 환경에서 안정적인 형태 표현을 얻으려면 스캐너/염색/블러/압축 같은 잡음과 준비 과정 차이에 덜 민감한 표현 학습이 핵심 과제다. ProsMAE는 masking 기반 재구성으로 학습 신호를 만들되, 본 메인 설정에서는 Gaussian noise 주입 없이 mask ratio 0.75로 학습해 도메인 변이에 대한 강인성을 확보하려 했다. 또한 디코더를 제거하고 encoder를 그대로 ProsCLS에 옮긴 frozen linear evaluation 설계를 통해 저연산 다운스트림 적용성을 유지했다.

- **Empirical Impact**: 재구성 품질 평가에서 ProsMAE는 LPIPS, SSIM, PSNR 기준으로 단일/기존 AE·VAE·single-source MAE 대비 우수한 성능을 보였다. 다운스트림 ISUP 등급 분류에서는 vanilla MAE의 평균 QWK 0.4084보다 ProsMAE가 0.4736으로 더 높은 ordinal 일치도를 보였고, primary disjoint split에서 QWK가 0.0652p 절대 개선됐다. 다만 검증은 단일 PANDA 코호트의 primary split 중심이며 seed/분할 구성 변동이 남아, 향후 반복 스플릿과 독립 전립선 코호트에서의 일반화 확인이 필요하다고 밝혔다.



### Unified Face Attack Detection via Fine-Grained Semantic Guidanc (https://arxiv.org/abs/2607.08156)
Comments:
          Accepted at ICME 2026

- **Prior Approaches**: 기존 얼굴 공격 탐지는 주로 시각 정보만 사용해 왔는데, 대표적인 데이터셋들은 위조 흔적을 설명하는 텍스트가 없어 모델이 시각적 단서에 과도하게 의존하는 한계가 있었습니다. 최근에는 CLIP 같은 언어-가이드 전이를 활용하거나 텍스트를 넣더라도, “This is a fake face” 같은 클래스 수준의 거친 문장에 그쳐 미세 포렌식 단서를 충분히 학습하기 어렵다는 지적이 나왔습니다. MS-UFAD처럼 텍스트를 제공하는 경우에도 동일 소스에서 생성 방식만 달라진 사례에서 설명이 같아 시각적 차이를 제대로 반영하지 못할 수 있습니다.

- **Core Contribution**: 이 논문은 MS-UFAD를 확장해 각 이미지에 위조 큐(forgery cues)를 더 세밀하게 묘사하는 800만 개+ 수준의 fine-grained 텍스트 애노테이션을 추가합니다. 또한 Dual Alignment Forgery Network(DAF-Net)과 Semantic Forgery Aggregation Module(SFAM)을 제안해, 텍스트의 “어떤 흔적”과 이미지의 “어떤 영역”을 더 정밀하게 정렬하도록 설계했습니다. 학습 시에는 멀티모달을 쓰되, 추론(inference)에서는 시각 브랜치만 사용해 실제 배치 운용 관점의 부담을 줄입니다.

- **Technical Challenges**: 핵심 기술적 난제는 MLLM이 생성하는 설명이 실제 이미지의 조작 영역에 정확히 위치/의미 정렬되지 않을 수 있는 hallucination 문제입니다. 이를 줄이기 위해 사전 얼굴 공격 탐지 모델과 ScoreCAM 열지도(heatmap)로 조작 후보 영역을 추출하고, 색/블러 같은 정량 차이와 함께 포렌식 유형별 artifact corpus를 프롬프트에 결합해 MLLM이 관련 문구를 선택하도록 했습니다. 모델 측면에서는 SFAM이 시각 패치와 텍스트 토큰을 “의미 있는 위조 구역/구(phrase)”로 재집계하고, global alignment과 fine-grained alignment을 함께 학습해 패치-토큰 구조 불일치도 완화합니다.

- **Empirical Impact**: 실험 결과 DAF-Net은 비전-only 방법과 클래스/거친 설명 기반 텍스트 가이드 접근을 모두 능가하며 ACER은 더 낮고 Acc/F1은 더 높게 나타났습니다. 특히 거친 설명 대비 성능 개선 폭이 ACER 2.94%, ACC 2.03%, F1 1.62%로 보고돼 fine-grained 텍스트의 실질적 가치가 강조됩니다. 또한 전역/세밀 정렬 손실 및 SFAM 구성요소를 제거한 ablation에서 성능 저하가 확인되어, 미세 단서 정렬과 의미 단위 집계가 일반화에 직접 기여함을 실증합니다.



### Understanding and Mitigating the Video-Action Generalization Gap via Temporal Ratio (https://arxiv.org/abs/2607.08127)
Comments:
          26 pages, 9 figures

- **Prior Approaches**: 생성형 비디오 파운데이션 모델을 로봇 제어에 옮기려는 접근은 크게 미래 비디오(또는 라텐트)를 예측하고 inverse dynamics로 행동을 추론하는 방식(WAM/VAM)과, 비디오·행동 토큰을 한 트랜스포머에서 함께 모델링하는 방식으로 나뉜다. 그러나 비디오 백본은 조합적 일반화가 강해도, 행동 데이터로 finetuning한 뒤에는 그 조합적 우선이 약해지며 OOD 성능이 붕괴하는 문제가 반복된다. 논문은 이 불일치를 video-action generalization gap(VAG gap)으로 명명한다.

- **Core Contribution**: 이 연구는 VAM 설계 선택(백본 finetuning, 학습 방식, 비디오 라텐트 추출/노이즈 수준, prediction horizon)이 조합적 일반화에 어떤 영향을 주는지 “설계 스페이스”로 체계적으로 진단한다. 그 결과 조합적 성공은 비디오가 그럴듯한 미래를 예측하는 것만으로는 부족하며, action head가 그 미래 롤아웃을 실제로 얼마나 참조하는지에 달려 있음을 보여준다. 이를 정량화하는 지표로 Temporal Ratio(TR)를 제안하고, TR을 활용해 inference-time에서 가이던스를 적응적으로 조절하는 TR-Adaptive Guidance를 제안한다.

- **Technical Challenges**: 핵심 난관은 action head가 미래 예측을 “참조하는지/무시하는지”가 겉으로는 영상 롤아웃만으로는 드러나지 않는다는 점이다. 논문은 action head의 attention을 현재 프레임(앵커)과 미래 프레임(예측 라텐트)로 분할해, 미래 라텐트에 대한 attention 질량을 TR로 정의함으로써 정책이 예측 모드로 이동하는 시점을 런타임 신호로 측정한다. 또한 TR이 태스크 단계에 따라 계획 구간에서는 상승하고 정밀 조작에서는 하락한다는 성질을 이용해, planning 단계에서 언어·계획(롱호라이즌) 가이던스 강도를 증폭하고 manipulation 단계에서는 완화하는 방식으로 잘못된 강화 가능성을 줄인다.

- **Empirical Impact**: LIBERO 벤치마크와 실제 이족/양팔 bimanual 태스크에서 기존 WAM/VAM 대비 조합적 OOD 성능 저하가 반복되는 “VAG gap”이 확인되며, TR과 TR-Adaptive Guidance로 이를 완화한다. LIBERO에서는 unguided VAM 대비 TR 기반 adaptive guidance로 평균 OOD 성공률이 59.4%까지 상승해 prior VAM 대비 큰 폭의 개선(논문은 5배 이상)을 보고한다. 실제 로봇 bimanual YAM에서도 평균 성공률이 71.7%에서 83.3%로 향상되며, 비디오 라텐트를 첫 denoising step에서 바로 쓰는 기존 방식은 다중 작업에서 실패함을 통해 TR 기반 설계의 실용성을 강조한다.



### VSRo-200: A Romanian Visual Speech Recognition Dataset for Studying Supervision and Multimodal Robustness (https://arxiv.org/abs/2607.08112)
- **Prior Approaches**: 기존 VSR(립리딩)은 LRW, LRS2, LRS3처럼 주로 영어 대규모 데이터로 성능이 크게 발전했지만, 저자원 언어에는 적용이 제한적이었다. 로보틱스/ML 커뮤니티에서 Romanian은 LRRo 같은 소규모 고립단어 중심 데이터가 주를 이뤄 연속 발화 상황을 반영하기 어려웠다. 또 대규모 웹 데이터에선 ASR 기반 pseudo-label로 확장했지만, 지도 품질이 성능에 미치는 영향을 통제해 분석하기는 어려웠다.

- **Core Contribution**: 본 논문은 루마니아어 VSR을 위한 최초의 대규모 실데이터 VSRo-200(200시간)을 공개하고, 전량을 루마니아어 ASR로 생성한 pseudo-label로 주석했다. 동시에 100시간은 사람 전사로 추가 라벨링해, 약지도(pseudo)와 준지도(인간) 효과를 동일 데이터 분포에서 직접 비교할 수 있게 했다. 이를 바탕으로 저자원 시각 음성 인식의 supervision quality, domain shift, AVSR 융합을 연구하는 벤치마크를 제시한다.

- **Technical Challenges**: 핵심 기술 과제는(1) 연속 팟캐스트 영상에서 화자 분리·구간화·오디오-비디오 동기화를 안정적으로 수행하고, (2) pseudo-label 잡음을 통제된 방식으로 실험에 반영하는 것이다. 저자들은 PySceneDetect로 shot을 나눈 뒤 단일 화자 얼굴 필터링, Pyannote diarization으로 게스트 발화만 남기고, S3FD+SyncNet으로 얼굴 추적과 ±10프레임 동기 오프셋 검증을 거쳐 입력을 정렬했다. pseudo-label은 Romanian fine-tuned Whisper-large로 생성했으며, AVSR에서는 오디오/비디오 예측 분포 엔트로피로 신뢰도를 동적으로 가중하는 confidence-aware fusion을 적용했다.

- **Empirical Impact**: 실험 결과, 같은 데이터 규모에서는 인간 주석이 pseudo-label보다 약간 더 낮은 WER을 보이지만, 데이터가 커질수록 그 격차가 줄어 pseudo-label의 확장성이 입증됐다. domain shift에서는 Vlog/전문 도메인/잡음/흑백 영상 OOD 세트에서 OOV 및 어휘-영상·음향 불일치가 성능 저하에 복합적으로 작용했으며, 흑백 영상에서 특히 큰 하락이 관찰됐다. 잡음 조건의 AVSR은 오디오-only 대비 강인성을 크게 개선했고, 마지막으로 VSRo-200에서 학습한 표현이 LRRo(고립단어)로 전이되어 기존 결과를 유의미하게 상회했다. 전체적으로 supervision 품질과 멀티모달 융합, 저자원 일반화 연구를 위한 새로운 테스트베드라는 의미가 크다.



### EVIS: A Physics-Grounded Event Camera Plugin for NVIDIA Isaac Sim (https://arxiv.org/abs/2607.08098)
- **Prior Approaches**: 이전 이벤트 카메라 시뮬레이터/생성 방법은 주로 RGB 영상을 이벤트로 변환하거나(v2e, Vid2E, PIX2NVS 등) 사전 렌더링된 프레임에 의존해 물리 시뮬레이션과의 정합성이 떨어지는 경우가 많았다. 일부는 장면/모델 기반 렌더링(ESIM, Blender 기반 고전 시뮬레이터, Gazebo 플러그인 등)을 시도했지만, GPU-병렬 물리 엔진(Isaac Sim) 안에 이벤트 생성이 통합되는 수준은 제한적이었다. 그 결과 로봇·장면별로 동기화된 고정밀 이벤트와 정답(ground truth)을 대규모로 만들기 어렵다는 문제가 남았다.

- **Core Contribution**: EVIS는 NVIDIA Isaac Sim/Isaac Lab 내부에서 직접 이벤트 스트림을 생성하는 ‘physics-grounded event camera plugin’으로, 로봇과 장면의 프레임-정확 ground truth를 이벤트와 함께 산출한다. 사실적인 log-intensity contrast 이벤트 모델을 구현하고, Isaac Sim에서 RGB 카메라 설정을 이벤트 카메라 설정으로 바꾸는 방식으로 어떤 물리 시뮬 장면에도 쉽게 적용된다. 또한 motion-vector 기반 프레임 보간 옵션으로 RTX 핵심 프레임만 렌더링하고 중간 이벤트를 합성해 단일 GPU에서 실시간 생성까지 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 고주사율에서의 대비 트리거링 모델의 충실도(잡음, 비이상 포함)와 (2) 물리 시뮬에 기반한 정확한 정답의 동시 제공, (3) 생성 처리량을 실시간 수준으로 유지하는 것이다. EVIS는 per-pixel asynchronous reference latching을 포함한 log-intensity contrast 이벤트 모델을 GPU 배치로 구현하고, threshold mismatch·refractory period·leak events·shot noise·hot pixels·finite bandwidth 같은 센서 비이상을 옵션으로 추가한다. 프레임 보간은 렌더러의 motion vector를 이용해 forward/backward warp와 softmax splatting으로 중간 프레임을 합성하며, anti-aliasing 충돌로 생길 수 있는 고스트를 보간 활성 시 비활성화해 품질을 보완한다.

- **Empirical Impact**: 단일 RTX 5090에서 렌더링 병목을 보간 계수에 따라 줄이며, 예컨대 240Hz 이벤트 생성이 실시간보다 빠르게(약 1.2×) 가능함을 보였다. 생성된 이벤트는 E2VID(재구성), E-RAFT(밀집 옵티컬 플로우), Match-Any-Events(다중 뷰 매칭) 같은 사전학습 네트워크에 대해 fine-tuning 없이 그대로 적용해도 합리적인 성능을 보였고, interpolation이 공격적으로 커질수록 재구성 SSIM/LIPS 등은 점진적으로 악화되었다. 특히 Isaac Sim의 정확한 motion vector를 이용해 ground truth 밀집 플로우를 별도 추정/보간 없이 평가할 수 있었고, 키프레임이 더 희소해져도 E-RAFT의 서브픽셀 수준 복원이 비교적 견고하게 유지되어 해당 분야의 데이터/실험 확장성에 의미가 크다.



### GRE-Diff: Gaussian Room Embeddings for Structured Layout Diffusion (https://arxiv.org/abs/2607.08086)
Comments:
          37 pages, 9 figures, conference

- **Prior Approaches**: 기존 floor plan 생성은 boundary(건물 외곽)나 bubble 같은 추상 제약, 또는 텍스트 조건을 바탕으로 그래프/GAN/확산 모델을 활용해 왔습니다. 다만 그래프 기반은 명시적 토폴로지 입력이나 후처리에 의존하기 쉽고, 마스크·폴리곤을 이산 표현으로 다루는 방식은 정점 순서/좌표 변화에 민감해 최적화가 불안정해질 수 있습니다. 또한 텍스트 기반 in-the-loop 편집을 제공하더라도 로컬 기하를 정밀 제어하면서 전역 일관성을 함께 유지하는 데 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 방(실)을 연속형 확률 잠재표현으로 모델링하는 Gaussian Room Embedding(GRE)을 제안하고, 이를 기반으로 controllable 생성과 편집을 한 프레임워크에서 수행하는 GRE-Diff를 제시합니다. GRE-Diff는 사용자 제약(방 타입/개수, 경계 형태, 편집 연산)을 조건화해 GRE 공간에서 확산을 유도한 뒤 다각형 레이아웃으로 디코딩하며, 생성과 편집을 동일 모델로 통합합니다. 특히 GUI/LLM 지시 기반으로 Add/Delete/Move/Repurpose/Anchor 연산을 지원하면서, Anchor된 방은 hard constraint로 고정해 전역 구조의 붕괴를 막는 설계를 포함합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 기하적 유효성·의미적 일관성·방 간 관계를 동시에 만족하는 controllable 생성, (2) 한 방 수정이 인접 공간으로 연쇄 전파되며 전역 응집도를 깨뜨리는 견고한 편집, (3) 이산 폴리곤 표현의 정점 순서 민감성을 줄이는 표현 설계였습니다. 연구진은 방별로 centroid(위치)와 scale(공간 영향)을 담는 Gaussian 임베딩을 사용해 정점 순서/좌표 섭동 민감도를 완화하고, GuidanceNet이 의미 토큰·경계 토큰·기존 폴리곤 프라이어를 dual-path로 융합한 뒤 Autoregressive Transformer가 방별 GRE 파라미터를 순차 생성하도록 구성했습니다. 이어 DenoisingNet이 GRE 임베딩을 초기 상태로 삼아 conditional diffusion으로 다각형을 정련하되, 편집 모드에서는 Anchor 마스크로 고정 정점을 강제 복원해 국소 재샘플링만 수행하도록 해결합니다.

- **Empirical Impact**: RPLAN 데이터셋(벡터화 8만+ 평면)에서 GRE-Diff는 FID 4.36으로 주요 비교군 대비 개선 폭이 크고, KID와 MMD도 낮아 실제 분포와의 일치도가 높게 나타났습니다. 생성 다양성은 coverage 96.09%로 최상 수준이며, 조건 제어 성능에서는 BC 98.44% 및 RC 100.00%를 기록해 BC-RC 결합 지표에서도 99.21%로 최고를 보였습니다. 편집 측면에서도 Tiny-ROE가 0%로 모든 요청 연산이 정확히 적용되고, Add/Delete/Anchor&Move의 F1이 강하게 나오면서 사용자가 실시간으로 레이아웃을 다듬을 때도 구조 보존이 확인됐습니다.



### Mixture of Enhanced-View Experts for Multi-Query Vehicle ReID and A Large-Scale Benchmark (https://arxiv.org/abs/2607.08085)
- **Prior Approaches**: 기존 차량 ReID는 대부분 single-query를 전제로 해 카메라 간 시점 변화에 취약했다. multi-query를 쓰더라도 test 시 단순 평균/가중합에 그치거나, VCNet처럼 viewpoint 라벨에 의존하는 방식은 현실 데이터에서 라벨 부족과 뷰 간 상관관계 모델링 부족으로 일반화가 흔들렸다. 또한 기존 데이터셋은 카메라 수와 시점 다양성이 제한적이라 복잡한 대규모 도심 환경에서의 검증이 어려웠다.

- **Core Contribution**: 이 논문은 multi-query 차량 ReID의 뷰별 보완 정보를 더 잘 쓰기 위해 EV-MoE(EV-Mixture of Enhanced-View Experts)를 제안하고, 이를 기반으로 CAFNet을 구성한다. EV-MoE는 VFEM(뷰 특화 feature enhancement)로 각 뷰의 식별성 있는 표현을 강화하고, DMFM(Dynamic Multi-view Fusion)으로 향상된 뷰들을 MoE와 attention-gating으로 동적으로 통합한다. 더불어 MAL(Multi-view Alignment Loss)로 multi-query feature와 single-image feature 간 불일치를 줄이기 위해 양방향 cross-view contrastive와 reconstruction 제약을 함께 건다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 뷰 간 상관관계를 반영해 각 뷰의 discriminative feature를 만들면서, (2) 동적 시점 변화 상황에서 fusion이 뷰 편향 없이 동작하도록, (3) multi-query와 single-query 학습/추론 간 분포 불일치를 손실로 일치시키는 것이다. 이를 위해 ViT 기반으로 뷰별 token을 뽑고 VFEM에서 cross-attention으로 다른 뷰의 패치 정보를 결합해 뷰 특화 향상을 수행하며, DMFM에서는 여러 expert 출력을 gating 네트워크가 선택·가중합하고 residual 스케일을 두어 토큰 단위로 유연하게 융합한다. 마지막으로 MAL은 fused feature가 single-view의 세부를 유지하면서도 cross-view 의미 공간을 정렬하도록 bi-directional contrastive와 MSE 기반 reconstruction을 함께 최적화한다.

- **Empirical Impact**: 실험에서는 MURI와 새 데이터셋 LCRI-1K(도심 수준의 대규모, 1,090 IDs, 107,805장, 23,637 cameras)를 대상으로 CAFNet의 견고함을 입증했다. LCRI-1K에서 mAP, mCSP, mINP, Rank-1이 기존 대비 각각 약 3.0%/1.1%/3.6%/0.7% 향상되며 복잡한 환경의 multi-query 차이를 잘 흡수함을 보여준다. 또한 LCRI-1K는 연중·24시간 범위와 풍부한 viewpoint/카메라 규모를 제공해 향후 multi-query 차량 ReID 벤치마크의 실전성을 크게 높일 것으로 기대된다.



### LDFE: Laplacian Decoupled Feature Enhancement Block for Dual-Stream CNN-based RGB-IR Object Detection (https://arxiv.org/abs/2607.08076)
- **Prior Approaches**: 극한 기상·저조도 환경에서는 단일 모달 기반 탐지가 한계를 보이면서, RGB-IR 쌍을 활용한 듀얼 스트림 YOLO 계열이 주류로 자리 잡았다. 기존 연구들은 주로 feature fusion 설계(다중 단계에서 RGB/IR 특징을 어떻게 섞을지)에 집중했지만, CNN은 전역 정보 포착이 약하고 Transformer/Mamba는 세부 디테일 민감도가 떨어지는 구조적 제약이 남아 있었다. 또한 잡음 억제를 전체적으로만 처리하거나(holistic), 전역/국소 특성을 분리해 다르게 다루지 못해 융합 효율이 제한되었다.

- **Core Contribution**: 이 논문은 듀얼 스트림 CNN 백본의 각 stage에서 RGB-IR 특징을 global-local로 분해한 뒤, 잡음 제거·융합·복원을 순차적으로 수행하는 Laplacian Decoupled Feature Enhancement(LDFE) 블록을 제안한다. LDFE는 Laplacian Pyramid로 전역(저주파)과 국소(고주파) 성분을 분리하고, 이후 Global State Space Enhancement(GS2E)와 Local Convolutional Correlation Enhancement(LC2E)로 각각 다른 방식의 denoising과 융합을 수행한 뒤 Laplacian reconstruction으로 되돌린다. 특히 GS2E는 양 모달을 번갈아 main/auxiliary로 두는 교차 주의 기반 억제와 state space 모델을 결합해 전역 구조의 장거리 의존을 강화한다.

- **Technical Challenges**: 핵심 난제는 전역 구조와 국소 디테일을 동시에 살리면서, RGB/IR에 내재한 modality-specific 잡음을 융합 과정에서 효과적으로 억제하는 것이다. 저자는 global-local 분해를 통해 잡음과 정보의 성격을 분리해 다루고, GS2E에서는 채널 스왑 후 교차 모달 attention으로 주된 모달의 잡음을 동적으로 억제하면서 long-range dependency를 State Space Model로 포착하도록 구성했다. LC2E에서는 국소 영역에 대해 L1 Normalization 기반 denoising과 Softmax Fusion, 그리고 spatial/channel attention 및 triple convolution으로 미세 디테일의 융합 품질을 끌어올렸다.

- **Empirical Impact**: 6개 RGB-IR 탐지 벤치마크(M3FD, DroneVehicle, LLVIP, FLIR-Aligned, KAIST, VEDAI)에서 LDFE 기반 모델은 SOTA 대비 mAP를 최대 6.2%p~2.0%p까지 일관되게 개선해 방법의 실효성을 확인했다. 또한 듀얼 스트림 CNN에 가벼운 LDFE를 얹는 설계와 Mamba의 linear complexity 덕분에 성능뿐 아니라 파라미터 효율과 런타임에서도 이점을 제시한다. 극한 조건에서 RGB-IR의 상보성을 “전역-국소 및 잡음 성격별로 분리해 융합”한다는 관점이 이후 융합 블록 설계에 실질적인 기준점을 제공할 것으로 보인다.



### UAV-OVVIS: Unmanned Aerial Vehicles Also Need Open-Vocabulary Video Instance Segmentation (https://arxiv.org/abs/2607.08075)
- **Prior Approaches**: 기존 UAV 비디오 인식은 사전 정의된 범주에 한정해 박스 기반 로컬라이제이션과 궤적 연결에 크게 의존해 왔다. VIS도 등장했지만, 여전히 closed-set 범주 가정 하에서 검출-분할-연결을 결합하는 방식이 많아 텍스트 쿼리로 의미를 바꾸는 open-vocabulary 유연성과 인스턴스 단위 분할 품질을 동시에 만족시키기 어렵다.

- **Core Contribution**: 이 논문은 UAV 영상에서 텍스트 쿼리에 따라 대상을 찾아내고, 전역적으로 일관된 video identity를 유지하는 instance-level segmentation trajectory를 출력하는 새로운 과제 UAV Open-Vocabulary Video Instance Segmentation(UAV-OVVIS)를 제안한다. 또한 instance-level 주석이 부족한 UAV 현실을 고려해, 학습 없이 기존 visual foundation model들을 재사용하는 training-free 통합 프레임워크 AeroTrack을 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) open-vocabulary 검출, 인스턴스 분할/propagation, 전역 identity 유지를 서로 결합해야 하는데 학습 없는 파이프라인으로 이를 분해해야 한다는 점, (2) UAV 장치와 객체가 동시에 움직이고 소형·밀집·가림이 잦아 장시간에서 새로 유입/재등장 인스턴스를 구분해야 한다는 점, (3) 긴 시퀀스에서 검출과 propagation의 메모리·연산 비용이 급증한다는 점이다. AeroTrack은 주기적 open-vocabulary detection과 짧은 세그먼트 내 mask propagation을 쓰되, 세그먼트 경계에서 Segmenter 상태를 리셋해 비용을 통제하고, Lifecycle-aware ID Association(LIA)로 로컬 ID 파편화를 출력 단계에서 전역 ID로 복구한다.

- **Empirical Impact**: 평가를 위해 UAV-OVVIS 벤치마크 AeroVIS를 구축했으며, 9개 카테고리와 8,279개의 instance-level trajectory를 포함한다. 실험 결과 AeroTrack은 UAV 상황에서 기존 general VIS 방법 대비 유의미하게 성능이 높고, open-vocabulary 관점의 강인성과 다양한 UAV 시나리오로의 일반화 능력도 확인됐다고 보고한다.



### Post-Training in End-to-End Autonomous Driving (https://arxiv.org/abs/2607.08072)
- **Prior Approaches**: 자율주행에서 엔드투엔드(end-to-end) 모델은 카메라·명령·상태 같은 멀티모달 입력을 받아 미래 궤적/기동을 직접 출력하며, 행동모방(behavior cloning), 궤적 생성 플래너, VLA(Vision-Language-Action)로 발전해 왔다. 그러나 안전·상호작용이 강한 환경에서는 오픈루프(open-loop) 모방만으로는 신뢰성을 보장하기 어렵고, 작은 실행 오차가 장기적으로 누적되어 회복 데이터도 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 자율주행 ‘포스트트레이닝(post-training)’을 초기 모방학습 이후에, 시뮬레이션/검증/선호/보상 등 추가 감독을 통해 이미 학습된 정책을 더 다듬는 별도 단계로 정의한다. 또한 감독 형태를 기준으로 distillation, preference-based alignment, reinforcement learning, test-time refinement의 네 계열로 문헌을 통합 분류해 체계적 비교가 가능하게 만든다.

- **Technical Challenges**: 포스트트레이닝 목표(안전·컴포트·규칙준수·진행·상호작용)는 점별 라벨로 완전한 최적화식으로 환원되기 어렵기 때문에, 서로 다른 감독 신호(교사, 선호 쌍, 스칼라 리워드, 테스트-타임 검증)를 설계해야 한다. 논문은 특히 폐루프 분포 shift와 오차 누적을 줄이기 위해 교사 신호를 학생 롤아웃에서 라벨링하는 분포 정합 관점(distillation), 의미 있는 trade-off를 담는 선호쌍 설계(preference), 다차원·장기·희소 실패를 반영하는 보상/롤아웃 구성(RL), 그리고 파라미터 업데이트 없이 후보 선택·검증으로 품질을 끌어올리는 test-time refinement을 핵심 해법으로 정리한다.

- **Empirical Impact**: 기존 모방 기반에서 더 나아가 폐루프 성능과 회복·안전성 측면을 개선하기 위한 방법들이 빠르게 확산되고 있으며, 논문은 이 흐름을 ‘포스트트레이닝 파이프라인’으로 재정리해 연구 간 단절을 완화한다. 동시에 벤치마크 포화, 폐루프 지표의 거칠음, 실차 평가 제약, 추론 비용 같은 평가/운영 과제를 짚고, 신뢰적이고 효율적인 포스트트레이닝 연구의 다음 방향을 제시한다.



### APIVOT: Adaptive Planning with Interleaved Vision-Language Thoughts (https://arxiv.org/abs/2607.08024)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 장기 로봇 조작 계획은 작업의 의미 구조(목표 분해, 행동 순서)와 기하 제약(여유 공간, 충돌)까지 함께 고려해야 한다. 기존 LLM/VLM 플래너는 언어적으로는 그럴듯한 행동 순서를 잘 만들지만, 공간 제약 때문에 실행 불가능한 계획을 내는 경우가 많다. 이를 보완하려고 외부 모션 플래너/feasibility checker/학습된 동역학 모델을 붙여 재계획을 유도했지만, 정작 모델 내부의 추론 과정에는 기하적 판단이 ‘후처리’처럼 분리되어 있었다.

- **Core Contribution**: APIVOT는 VLM 기반 플래너로, 언어 추론(semantic reasoning)과 시각적 사고(visual thoughts)를 계획 추론 흐름(trace) 안에서 적응적으로 섞는다. visual thoughts는 상상된 미래 상태로서 이후 단계의 기하적 실행 가능성(공간 적합, 충돌 회피)을 내부에서 검증하는 데 쓰인다. 핵심은 매 단계에서 언어/비전 중 어느 모달리티가 필요한지 모델이 학습해, 불필요한 시각 추론을 줄이면서도 성공률을 높인다는 점이다.

- **Technical Challenges**: 문제는 ‘언어로는 정교한 기하 구조를 간결하게 표현하기 어렵다’는 점과, 반대로 시각적 사고를 매번 쓰면 계산 비용이 커진다는 점이다. APIVOT는 (1) 제공된 visual thoughts를 활용해 기하 정보를 추출하는 단계, (2) visual thoughts를 스스로 생성하도록 하는 단계, (3) 기하 제약이 중요한 단계에서만 visual thoughts를 생성하도록 하는 3단계 커리큘럼으로 적응성을 학습한다. 또한 visual thought가 가리키는 잠재 시각 상태를 정답 미래 장면 특징과 정렬(cosine similarity)해, 상상된 상태가 실제 장면과 맞물리도록 설계했다.

- **Empirical Impact**: KitchenWorlds의 장기 키친 태스크에서 APIVOT는 평균 작업 성공률 0.419를 기록하며 최강 베이스라인 대비 8.1%p 향상했다. 특히 공간 제약이 강한 설정에서 격차가 7점에서 17점으로 벌어져, 내부 기하 검증의 효과가 크게 드러났다. 더불어 visual thoughts를 모든 단계에 쓰는 상한 대비 성능의 91%를 유지하면서 토큰 사용은 39% 줄여, 모달리티 선택 학습이 성공률과 추론 효율을 동시에 개선함을 실증했다.



### SAGA: Stable Acceleration Guidance for Autoregressive Video Generation (https://arxiv.org/abs/2607.08020)
- **Prior Approaches**: 영상 생성에서 diffusion은 고화질 프레임을 잘 만들지만, 스트리밍/장시퀀스 생성을 위해 autoregressive-diffusion(AR-Diffusion)로 전환하면 롤아웃 중 작은 시간오차가 누적될 수 있다. 기존 접근은 구조 설계나 학습 정렬, 모션 플래닝, 글로벌 매끄러움 정규화 등으로 일관성을 높이려 했지만, 왜 그 오차가 특정 주파수 성분에서 불안정해지는지는 명확히 설명하지 못했다. 또한 inference-time guidance는 주로 의미/조건 정합이나 공간·측정 일관성에 초점을 두고, AR 롤아웃의 시간 동역학 안정성 자체를 직접 겨냥하진 않았다.

- **Core Contribution**: 이 논문은 AR 비디오 diffusion의 실패 모드를 “가속도(2차 차분) 관점의 스펙트럴 불안정성”으로 재정의한다. discrete latent acceleration이 고주파 시간 섭동을 증폭해 flickering, motion jitter, structural drift를 유발할 수 있으며, 이를 제어하는 것이 핵심이라고 제안한다. 그 결과, SAGA는 학습 없이(acceleration guidance + structured noise initialization) 기존 chunk-wise autoregressive diffusion 모델에 그대로 적용되어 시간 품질을 개선한다.

- **Technical Challenges**: 가속도 스펙트럼에서 불안정 모드를 억제하려면, autoregressive 롤아웃이 짧은 시간 창만 노출하므로 스펙트럼 누설(leakage) 문제가 커진다는 점이 기술적 난제다. 이를 해결하기 위해 SAGA는 finite-window에서 에너지 집중이 최대가 되도록 DPSS(Slepian) basis에 가속도를 투영해 band-limited 농도를 달성하고, 동시에 denoising 중 가속도 도메인에서 고주파 모드를 억제하는 guidance 목적을 사용한다. 더불어 롤아웃이 과거 latent를 재사용하는 특성을 고려해, 초기에 short-range 시간 상관을 상쇄하면서도 장기 모션 구조는 보존하는 structured autoregressive noise initialization을 설계한다.

- **Empirical Impact**: 실험은 VBench 평가와 human preference로 검증했으며, SAGA는 여러 AR diffusion 백본에서 시간 품질을 일관되게 끌어올렸다. Self-Forcing에서는 Temporal Quality가 97.30→97.91, Image Quality가 69.60→70.51로 동시 개선되며, 영상 왜곡을 무작정 매끄럽게 만드는 방식이 아니라 고주파 가속도 불안정성을 줄인 결과로 해석된다. 스펙트럼 분석에서도 가속도 에너지의 RMS와 총 power가 감소하고 고주파 구간에서 특히 큰 억제가 관찰되었으며, 블라인드 선호도 연구에서도 SAGA가 더 높은 표를 받았다.



### LightCrafter: PBR-Conditioned Video Diffusion Refinement for Controllable and Consistent Relighting (https://arxiv.org/abs/2607.08016)
- **Prior Approaches**: 기존 비디오 릴라이팅은 (1) inverse rendering으로 geometry·materials·illumination을 복원한 뒤 PBR로 재조명하거나, (2) diffusion을 이용해 target 조명 조건에 맞춘 video-to-video 번역으로 접근해 왔다. 전자는 단일 영상의 내재성분 복원이 ill-posed라 잡음·퇴화가 생기고 그 오류가 그대로 결과로 전이되며, 고정/정적 편향과 최적화 부담도 크다. 후자는 target environment map이나 text에 의존해 제어가 제한되거나 장기 일관성이 흔들리고, paired 데이터 부족으로 학습 제약이 따른다.

- **Core Contribution**: LightCrafter는 릴라이팅을 “입력→목표로 직접 번역”하는 대신, 입력을 target illumination 하에 PBR로 렌더한 proxy video를 만든 뒤 diffusion이 이를 photorealistic 최종 릴라이팅으로 정교화하도록 재정의한다. 이렇게 하면 조명 개념을 diffusion이 암묵적으로 학습할 필요 없이, 조명 제어는 PBR proxy에 “베이크(bake)”되어 더 정밀한 글로벌/로컬 조명 조합을 다룰 수 있다. 또한 PBR의 결정적 렌더링 구조 덕분에 장기(긴 시퀀스) temporal consistency를 자연스럽게 얻는다.

- **Technical Challenges**: 핵심 난제는 inverse rendering에서 발생하는 geometry/material 잡음과 PBR의 전형적 결함(깜빡임, jagged shadow, disocclusion hole 등)을 diffusion이 테스트 시와 동일한 분포에서 보정하도록 만드는 것이다. 저자들은 이 train-test mismatch를 줄이기 위해, inference와 동일한 inverse rendering·PBR 파이프라인으로 가짜 학습 쌍을 생성하는 artifact-matched 데이터 큐레이션을 도입한다. 더 나아가 긴 영상은 chunk별 denoising에서 조명이 드리프트하는 문제를 overlap-fused temporal tiling으로 완화해, 하나의 coherent denoising 궤적으로 전체 프레임을 함께 갱신한다.

- **Empirical Impact**: 실험에서 LightCrafter는 기존 실세계 릴라이팅 벤치마크에서 state-of-the-art 성능을 보이며, PBR 단독이 이미 일부 선행을 능가한 뒤 diffusion refinement가 잔여 아티팩트를 보정해 품질을 끌어올림을 보여준다. 특히 200프레임급 긴 합성 시퀀스에서 그림자·하이라이트·전체 외관의 안정성이 유지되는 반면, end-to-end diffusion은 chunk 처리로 인한 드리프트와 제어 누락이 두드러진다. 또한 synthetic paired(3,000) + real-world pseudo-pairs(1,000) 혼합 학습이 조명 제어와 photorealistic refinement를 동시에 달성하는 데 중요함을 ablation으로 확인하고, 학습용 합성 벤치마크 및 코드/데이터 공개 계획까지 제시한다.



### FedTR: Federated Learning Framework with Transfer Learning for Industrial Visual Inspection (https://arxiv.org/abs/2607.08014)
Comments:
          Author's accepted version. Published in Proceedings of the Great Lakes Symposium on VLSI 2024 (GLSVLSI '24)

- **Prior Approaches**: 산업 비주얼 검사(IVI)에서 데이터는 공장별로 흩어져 있고, 프라이버시·규정 때문에 중앙 서버에 원본을 모으기 어렵다. 이에 따라 federated learning(FL)로 업데이트만 공유하지만, 비동일한 데이터 분포(iid가 아닌 경우)가 수렴과 성능을 떨어뜨린다는 한계가 있다. 또한 적은 결함 라벨 데이터 환경에서는 FL만으로는 성능 격차가 커질 수 있다.

- **Core Contribution**: 이 논문은 FedTR을 제안하며, transfer learning을 FL 파이프라인에 결합해 공장별 제한된 프라이빗 데이터에서도 end-to-end 텍스트 인식을 잘 학습하도록 한다. 특히 결함 라벨 결함을 word-level 정확도로 읽어내는 과제를 목표로, 공개 데이터로 1차 학습 후 프라이빗 데이터에 대해 fine-tuning을 수행하는 구조다. 그 결과, 중앙집중 학습과 동등 수준의 성능을 프라이버시 유지 하에 노린다.

- **Technical Challenges**: 핵심 기술 난제는 공장 간 데이터 분포 차이(heterogeneity)로 인해 FL 집계(FedAvg)가 전역 최적점과 멀어질 수 있다는 점이다. 논문은 이를 줄이기 위해 SynthText 같은 공개 데이터로 만든 베이스 모델을 초기값으로 하고, 전 구간 fine-tuning으로 도메인 적응을 수행하되 detection과 recognition을 단계적으로(텍스트 검출→텍스트 인식) 학습·적용한다. 또한 각 라운드에서 로컬 SGD 업데이트 후 서버가 FedAvg로 가중치를 평균내는 표준 FL 루프를 유지한다.

- **Empirical Impact**: 실험에서 FedTR은 동질(homogeneous) 설정에서 word-level 정확도 95.5%, 이질(heterogeneous) 설정에서 94.2%를 달성하며, 중앙집중 학습과 비교해 성능 손실이 작음을 보였다. 텍스트 검출의 경우에도 공장별 데이터 분포 차이에 대해 FedTR이 더 견고해져, 한 공장에 과적합되는 문제를 완화한다. 특히 결함 라벨 인식에 쓰이는 ink cartridge 프라이빗 데이터셋에서 중앙학습 수준에 가까운 결과를 보여, 실제 디지털 제조 품질검사 파이프라인 적용 가능성을 시사한다.



### LOGOS: Language-guided Oriented Object Detection in Aerial Scenes (https://arxiv.org/abs/2607.08004)
Comments:
          Accepted to SOICT 2025

- **Prior Approaches**: 기존 회전(방향성) 객체 탐지는 회전 RPN/ROI를 확장한 CNN 계열과, DETR류를 변형한 transformer 계열로 나뉜다. 다만 각도 예측을 분류로 처리하거나(각도 주기성 문제), 고정 각도 범위(예: 0~90도) 제약이 있어 -π/2~π/2 경계에서 불연속·모호성이 생기기 쉽다. 또한 DETR의 고정 query 개수는 객체 수가 적거나 장면이 희소할 때는 불필요한 계산을 늘리고, 장면이 복잡·밀집할 때는 쿼리 부족으로 수렴·검출 성능이 흔들릴 수 있다.

- **Core Contribution**: LOGOS는 항공/위성 장면에서 회전 bounding box(OBB) 탐지를 텍스트 프롬프트로 유도하는 언어-가이드 transformer 모델이다. 핵심은 prompt-modulated content queries로, 입력 텍스트(예: “plane”, “harbor” 등)가 query의 주의를 동적으로 조절해 원하는 범주의 회전 객체에 집중하도록 만든 점이다. 이를 통해 각도 모호성과 장면 밀집도 변화에 따른 query redundancy/부족 문제를 동시에 완화하는 방향을 제시한다.

- **Technical Challenges**: 어려움은 (1) 각도 공간의 주기성으로 인한 불연속 예측, (2) 장면의 객체 밀도·크기 변화에 맞춘 query 효율화, (3) 텍스트 조건이 시각적 로컬라이제이션과 회전에 실제로 기여하도록 cross-attention을 설계하는 문제다. LOGOS는 DINO 인코더로 견고한 시각 토큰을 만들고, FiLM으로 콘텐츠 query를 프롬프트에 맞게 변조한 뒤 multi-head cross-attention에서 시각+언어 특징을 함께 참조한다. 각도는 sin/cos로 회전 표현을 학습하고, 학습에서는 Hungarian matching과 contrastive denoising 보조 손실을 써서 잡음 query의 복원과 불필요 negative 억제를 노린다.

- **Empirical Impact**: DOTA-v1.0/v1.5/v2.0에서 LOGOS는 전반적인 mAP에서 기존 SOTA를 앞섰고, 특히 밀집 장면과 회전이 뚜렷한 시나리오에서 강점을 보였다. 예를 들어 v1.0에서 mAP 81.32%를 달성했으며, 프롬프트로 특정 범주를 필터링해 harbor·roundabout 등 난이도 높은 범주에서 성능이 두드러졌다. 한편 ship과 Baseball Court(BC)에서는 상대적으로 약한 결과가 보여 프롬프트 정교화나 해당 범주 맞춤 학습의 여지가 남았고, 극도로 밀집하거나 드물게 보이는 예외 조건에서 오탐/미탐이 나타났다.



### Beyond Thermal Imaging: Inferring Thermophysical Properties from Time-Resolved Thermal Observations (https://arxiv.org/abs/2607.07962)
Comments:
          31 pages, In submission

- **Prior Approaches**: 기존 열(thermal) 비전은 주로 인식·분할·이상 탐지에서 RGB와의 보완 모달리티로 열영상을 활용했으며, 물리적 상태를 직접 추정하기보다 시각적 특징을 결합하는 데 초점이 있었다. 3D 재구성 쪽으로는 NeRF/Gaussian splatting 등으로 열/온도 맵을 렌더링하는 연구가 늘었지만, 온도를 ‘정적 외관’으로 다루는 경우가 많아 열전달을 지배하는 열물성(예: thermal diffusivity)을 예측적으로 식별하기 어렵다. 반대로 inverse heat-transfer는 열 방정식을 최적화에 포함해 물리해석 가능성을 주지만, 단순 형상·통제된 경계조건·격자 기반 근사에 의존해 복잡한 3D 장면으로 확장되기 힘들다는 한계가 있었다.

- **Core Contribution**: ThermoField는 열 장면 재구성과 열물성 파라미터 추정을 differentiable heat-transfer simulation으로 통합해, 복잡한 3D에서 열물성 필드를 물리적으로 제약하며 추정한다. 열물성은 signed-distance-function 기반 표면(재구된 geometry) 위에서 정의되는 공간 가변 neural field로 표현되고, 열전달 물리(열방정식 잔차, 경계 교환, 시간 동역학)를 통해 시간별 관측 온도와 정합되도록 학습된다. 그 결과 geometry 복원, 공간적으로 변하는 thermal diffusivity 추정, 그리고 관측에 쓰지 않은 환경 조건에서의 thermal evolution 예측을 함께 수행한다.

- **Technical Challenges**: 핵심 기술 난제는 관측 온도만으로는 경계조건·형상·물성·환경 교환이 얽혀 underconstrained inverse problem이 된다는 물리적 식별성(identifiability) 문제다. ThermoField는 열카메라가 관측하는 것은 표면 온도라는 점에 맞춰, 재구된 표면을 계산 도메인으로 고정하고 열전달 시뮬레이터가 physical fields(thermal diffusivity 등)를 입력으로 받아 온도 시계열을 산출하도록 설계했다. 또한 finite-element 기반의 differentiable solver를 사용해 시뮬레이션-오차를 역전파 가능하게 만들고, diffusivity를 표면 제어점 + positional encoding이 있는 신경 표현으로 공간 분포까지 학습한다.

- **Empirical Impact**: 합성 데이터(간단/복잡 형상, 다양한 재료·경계조건)를 통해 ThermoField가 thermal diffusivity를 수치 정확도와 공간 응집성 관점에서 회복하며, 조건이 바뀌어도 재구성한 물성이 예측에 유의미하게 기여함을 보였다. 특히 훈련에서 지배적 물리 항이 유지되는 테스트 조건에서는 spatially varying field가 median baseline보다 큰 폭으로 MAE를 줄였지만, 열가열처럼 다른 물리 균형이 지배하는 경우에는 transfer가 약해지는 패턴도 확인했다. 반복 최적화 실험에서는 비볼록성에도 불구하고 해가 전반적으로 일관되게 수렴하는 경향과 함께, 관측 공간 오류가 낮아도 물성은 여러 해로 분산될 수 있음을 보여 ‘uncertainty-aware’한 해석이 필요하다는 실증적 근거를 제시했다.



### Adversarial Decoys: Misdirecting Attention-Based Defenses in V (https://arxiv.org/abs/2607.07922)
- **Prior Approaches**: 기존 ViT 국소 공격(예: adversarial patch, adversarial object)은 특정 토큰/영역을 조작해 오분류를 유도하며, PatchFool·Attention-Fool처럼 공격이 attention을 적극 활용하는 경우도 있었다. 이에 대응해 ARMRO, RSA 같은 test-time 방어는 높은 attention(또는 이상 activation)을 받은 토큰을 의심해 마스킹·중화한다. 이런 접근은 attention-공격 효과 간의 강한 결합을 전제로 하며, 그 결과 공격이 attention을 피하거나(혹은 줄이거나) 적응하는 adaptive 전략이 어려워지는 문제가 제기됐다.

- **Core Contribution**: 본 논문은 adversarial decoys라는 독립 최적화 패치를 제안해, 방어가 억제하려는 attention 신호가 ‘진짜 공격 영역’이 아니라 공격자가 지정한 목표 토큰으로 향하게 만든다. 핵심은 오분류 유도와 방어 회피를 하나의 목표로 동시에 최적화하는 대신, 원래의 공격은 오분류만 담당하고 decoy는 방어가 쓰는 attention ranking을 조작하도록 기능을 분리한 점이다. decoy는 underlying attack과 독립적으로 학습되므로 attack-agnostic 방식으로 기존 localized attack 파이프라인에 쉽게 결합된다.

- **Technical Challenges**: 어려움은 “attention 크기를 낮추는 식의 회피”가 오히려 공격 효과를 약화시킬 수 있다는 점과, 단순히 높은 attention을 만들기만 해서는 방어가 선택하는 ranking에서 목표 토큰이 실제로 우선순위를 갖지 못할 수 있다는 점이다. 이를 위해 논문은 layer-wise로 (1) 목표 토큰의 received-attention을 전반적으로 끌어올리고, (2) Top-k 내부에서 목표 토큰이 비목표 경쟁 토큰보다 우위가 되도록 지배(dominance) 비율을 최적화하는 목적함수를 설계한다. 또한 decoy 최적화를 원래 공격과 분리해, 공격의 공격성분을 건드리지 않으면서도 방어가 탐지할 attention 경로만 바꾸는 구조를 만든다.

- **Empirical Impact**: ImageNet에서 DeiT, ViT 등 여러 아키텍처와 PatchFool 및 일반 adversarial patch 공격을 함께 실험한 결과, decoys는 높은 attention 점수를 진짜 공격 영역에서 멀리 돌리고 ARMRO의 suppression 마스크-공격 영역 겹침을 크게 줄였다. 그럼에도 불구하고 원래 공격의 유효성은 상당 부분 유지되어, 방어를 단독 적용할 때보다 defended accuracy가 더 크게 떨어지는 양상이 관찰됐다. 논문은 또한 attention magnitude만을 adversarial relevance의 지표로 쓰는 접근이 근본적으로 취약할 수 있음을 실험적으로 보여준다.



### 3D Reconstruction of deciduous Trees using low-cost UAV- and Crane-based Photogrammetry for Monitoring Shoot Elongation across entire Canopies (https://arxiv.org/abs/2607.07905)
Comments:
          Accepted to ISPRS Congress 2026, camera-ready version

- **Prior Approaches**: 기존 연구는 덴드로미터로 방사상(secondary) 생장, 즉 연륜 기반의 연속 측정을 중심으로 이뤄져 왔다. 반면 주로 생장의 핵심인 경사상(primary) 새순( shoot elongation ) 모니터링은 적절한 측정 기술 부재로 인해 상대적으로 소홀했다.

- **Core Contribution**: 이 논문은 저가형 UAV 포토그래메트리와 다중 카메라 CraneCam을 활용해 낙엽활엽수의 3D 복원을 통해 전체 수관 단위의 새순 생장 측정을 가능하게 하는 기반을 제시한다. 또한 3D 포인트클라우드에서 정밀도·해상도·완전성을 분석하고, 얇은 새순 같은 미세 구조를 평가하기 위한 3D printed ground-truth 가지를 새로 도입한다.

- **Technical Challenges**: 핵심 기술적 난제는 현장 환경에서 미세한 새순을 포함하는 얇은 구조를 충분히 정확하고 완전하게 재구성하는 것이다. 논문은 UAV 및 CraneCam의 데이터 취득·처리 전략을 점검하고, 포인트클라우드의 정확도/해상도/완전성을 체계적으로 평가해 구현 가능한 재구성 품질 범위를 도출한다.

- **Empirical Impact**: 실험 결과, 소비자용 UAV(무게 250g 미만)로 전체 나무 단위 3D 포인트 정확도 5~6mm를 달성했으며 재구성 완전성은 UAV 유형에 따라 92~98%로 나타났다. 또한 포토그래메트리 포인트클라우드를 기반으로 한 나무 골격화(skeletonization) 초기 실험과 운영상 과제를 논의해, 기후변화가 primary tree growth에 미치는 영향 연구의 실증적 기반을 확장할 것으로 기대된다.



### GIRAF: Towards Generalizable Human Interactions with Articulated Objects (https://arxiv.org/abs/2607.07880)
Comments:
          12 pages, 6 figures, 3 tables. Accepted at the Third Workshop on Human Motion Generation (HuMoGen), CVPR 2026

- **Prior Approaches**: 기존 연구는 정적 물체(의자·침대 등)나 강한 제어 신호가 주어지는 손 중심 조작에 치우치는 경향이 컸습니다. 또 일부는 diffusion으로 텍스트 제어를 결합했지만, locomotion(접근)과 manipulation(조작) 전환을 다루지 않거나 손-물체 접촉을 손 또는 표면 단독 표현으로 처리해 새 형상으로의 일반화가 약했습니다. 결론적으로 “전신이 물체에 접근→미세 접촉→관절(articulated) 작동”을 한 모델에서 일관되게 생성하는 데는 공백이 남아 있었습니다.

- **Core Contribution**: 이 논문은 텍스트 조건 diffusion 모델로 전신 locomotion과 관절 물체 조작을 장시간에 걸쳐 생성하는 통합 프레임워크를 제안합니다. 핵심은 손-물체 접촉을 물체 표면(surface)과 정렬되는 공유 표현으로 통합해, geometry가 달라도 대응이 유지되도록 한 object-centric contact 표현을 제공하는 것입니다. 또한 접근과 조작의 자연스러운 전환을 위해 mixed-domain 학습 전략과 FiLM 기반 조건을 넣고, 데이터 부족을 완화하기 위해 contact 기반 증강을 설계했습니다.

- **Technical Challenges**: 가장 큰 기술 난제는 관절 물체에서의 fine-grained 손-물체 접촉을 동시에 “일반화 가능하게” 표현하는 것입니다. 이를 dynamic basis point sets(BPS)로 해결해, 관절 파트에 정규화된 basis에서 손 말단과 물체 표면 접촉을 투표(voting) 형태로 일관되게 학습하도록 했습니다. 두 번째 난제는 locomotion과 manipulation을 한 모델이 매끄럽게 이어 붙이는 것이며, 물체까지의 거리 기반 locomotion mask와 로딩/어닐링 스케줄로 도메인 균형을 맞춰 해결합니다. 마지막으로 생성 품질을 위해 DDIM 기반 scene-aware diffusion noise optimization으로 접촉 정합, 관통(penetration) 억제, 시간적 jitter 완화를 단계적으로 수행합니다.

- **Empirical Impact**: ParaHome 데이터에서 접촉/관통 품질, 텍스트-모션 정합, 관절·객체 재구성 정확도 전반에서 기존 SOTA 대비 우수한 정량 성능을 보였습니다. 예컨대 contact distance와 penetration error가 최저 수준으로 떨어지고, T2M 관련 지표와 FID에서도 개선되면서 다양성도 유지됐습니다. 또한 학습에 없던 물체 배치에서 “drawer·microwave 열기/잡아내기”의 긴 호라이즌 동안 접촉이 지속되며, 같은 행동도 높이나 위치가 달라져도 자연스럽게 적응하는 일반화 결과를 제시했습니다.



### DreamCharacter-1: From 3D Generative Foundation Models to Product-Ready Character Generation (https://arxiv.org/abs/2607.07817)
Comments:
          Official Page: this https URL

- **Prior Approaches**: 기존 3D foundation 기반 생성은 단일 이미지나 multi-view, 자연어 프롬프트로 일반 물체 재구성과 생성을 잘 수행하지만, 제품 수준의 3D 캐릭터(정체성·고빈도 디테일·완전 텍스처·리깅 호환)까지는 동시에 만족하기 어렵다. 특히 geometry는 과도한 평활화·얇은 구조 누락·역면(뒤쪽) 비현실성, texture는 조명/가림/그림자에 얽힌 albedo 추정 실패와 교차 뷰 불일치, 보이지 않는 영역의 환각이 병목이 된다. 또한 고품질을 위해 backbone을 키우거나 인스턴스별 반복 최적화를 붙인 접근은 학습비용과 추론 지연이 커 산업 파이프라인 적용에 한계가 있다.

- **Core Contribution**: DreamCharacter-1은 pretrained 3D foundation 모델을 “전체 재학습”이 아니라 task별 post-adaptation(후처리 적응)으로 캘리브레이션해, 제품 준비형 3D 캐릭터 생성을 목표로 하는 경량 프레임워크를 제안한다. geometry post-training(기하 선호 기반 최적화), texture post-training(고해상 텍스처 합성과 self-occlusion 영역 복원), 그리고 inference acceleration을 단일 파이프라인으로 묶어 시각 품질과 구조적 견고성을 함께 끌어올린다. 결과적으로 정체성 일관성, 고빈도 표면 디테일, 완전하고 뷰-일관적인 텍스처, 리깅/애니메이션 호환성을 함께 노린다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 전역 구조는 무너지지 않으면서도 얇은 구조·날카로운 에지·고주파 디테일을 정체성에 맞게 복원하고, (2) 입력 이미지의 조명/가림 잡음을 분리해 뷰 간 일관된 텍스처를 만들며, (3) 이런 성능을 유지한 채 추론 지연을 산업 수준으로 낮추는 것이다. 논문은 coarse-to-fine SDF latent 기반 geometry 파이프라인과 이미지-정합 강화를 위한 구조화된(voxel) latent 정렬, multi-metric geometry reward로 강화학습을 결합해 기하 품질을 끌어올리고, texture는 multi-view 생성 후 sparse voxel 3D inpainting으로 가려진 영역을 채워 환각을 줄인다. 이어서 student-model distillation, 빠른 mesh extraction, 효율 attention(KV-cache 유사), 파이프라인 병렬화 등으로 대규모 배치를 위한 저지연 추론을 구현한다.

- **Empirical Impact**: 정량·정성 실험에서 DreamCharacter-1은 시각적으로 설득력 있고 구조적으로 견고한 3D 캐릭터 자산을 생성하며, 기존 character generation 방법들보다 일관되게 우수한 성능을 보인다고 보고한다. 특히 얇은 구조와 날카로운 디테일, 역면의 그럴듯한 기하, 그리고 self-occlusion/가시성 한계가 있는 부위의 텍스처 완성에서 강점을 나타낸다. 더불어 경량 post-adaptation과 추론 가속 설계를 통해 고품질 출력과 실사용성(리깅 호환·저지연)을 동시에 겨냥했다는 점에서, 학계 3D 생성 연구를 산업용 에셋 파이프라인으로 연결하는 데 의미가 크다.



### AUTOPILOT VQA: Benchmarking Vision-Language Models for Incident-Centric Dashcam Understanding (https://arxiv.org/abs/2607.08745)
Comments:
          CVPR Autopilot Workshop

- **Prior Approaches**: 기존 비전-언어 모델(VLM)·대규모 언어 모델(LLM) 기반 연구는 자율주행에서 장면 이해, 추적, 예측 등 “일반 인지” 성능을 끌어올리는 데 집중해 왔습니다. 그러나 사고·아찔한 상황처럼 드문 안전 중요 이벤트를 신뢰성 있게 추론/설명하는 평가체계는 부족했습니다. DrivingVQA, MetaVQA, NuPlanQA 같은 VQA·추론 벤치마크도 있으나, 정형화된 사고 장면 이해와 시간적·인과적 추론 요구까지 충분히 담지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 대시캠 비디오를 바탕으로 한 incident-centric visual question answering 벤치마크 AUTOPILOT-VQA를 제안합니다. 충돌, near-miss, 예방된 위험, no-incident를 포함해 실제 도로 사고에 가까운 “사건 중심” 질문을 설계하고, 환경 맥락부터 사고 상세(관련 주체, 원인 귀속, 충돌/피해 위치, 회피 가능성 추론)까지 답하도록 구성했습니다. 이를 통해 객체 인식 수준을 넘어 temporally grounded하고 safety-aware한 추론을 평가하는 표준을 제공합니다.

- **Technical Challenges**: 핵심 난제는 영상만으로는 희소하고 복합적인 사건 원인을 추적해야 한다는 점이며, 단순 분류가 아니라 주변 맥락·행동 상호작용·사건 발생 조건을 시간적으로 연결해 정형 답을 산출해야 합니다. 또 질문이 상황에 따라 “해당/비해당(unknown and non-applicable)”이 될 수 있어, 모델이 증거가 부족한 경우를 구분하며 질문 적용 가능성을 판단해야 합니다. 논문은 600+ 클립과 6,000+ Q&A를 통해 환경·도로·주체·사고 범주·fault/avoidability 등 6개 의미 묶음에 걸친 구조화 질의를 제공해, 이러한 추론 요구를 벤치마크로 강제합니다.

- **Empirical Impact**: AUTOPILOT-VQA는 AUTOPILOT CVPR 2026 경쟁으로 공개되었고, Kaggle에서 224명 등록·73명 활동·59팀·총 686회 제출로 높은 참여를 끌었습니다. 리더보드 상위 점수는 0.65835 수준으로, 상위권이 0.60을 넘기긴 하지만 인간 수준의 신뢰성에는 못 미친다는 점이 드러났습니다. 또한 다수 팀이 0.39–0.40 구간에 몰리는 장기 꼬리 분포를 보여, 단순 지각 기반 접근만으로는 원인·예방 조치·피해 지점 같은 인과/관계 추론을 전반적으로 마스터하기 어렵다는 시사점을 줍니다. 결과적으로 이 벤치마크는 안전 중요 추론의 평가·해석 가능하고 견고한 vision-language 파이프라인 개발을 촉진하는 실증적 기준점으로 자리잡는다는 의미가 있습니다.



### ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation (https://arxiv.org/abs/2607.08741)
Comments:
          ACM Transactions on Graphics (SIGGRAPH 2026)

- **Prior Approaches**: 기존 오프라인 3D 인간 모션 생성은 텍스트·운동학 제약을 정밀하게 반영할 수 있지만, 전체 시퀀스를 병렬 생성하는 구조 때문에 인터랙티브 환경에서 필요한 실시간 추론 속도가 부족한 경우가 많습니다. 반대로 온라인/스트리밍(autoregressive) 방식은 빠르지만 텍스트 의미를 복잡하게 해석하거나, 긴 지평(long-horizon) 제약을 안정적으로 만족시키는 데 한계가 있었고, 둘 다를 동시에 만족하는 접근도 제한된 컨텍스트 창 때문에 성능이 흔들렸습니다.

- **Core Contribution**: 본 논문은 텍스트 프롬프트와 유연한 kinematic constraints를 실시간으로 결합해 고충실도 모션을 생성하는 스트리밍 프레임워크 ARDY를 제안합니다. ARDY는 explicit root(루트) 제어는 명시적으로, body(몸통)는 latent embedding으로 압축해 두 표현을 혼합함으로써 정확한 궤적 제어와 효율적 생성학습을 동시에 노립니다.

- **Technical Challenges**: 핵심 과제는 (1) 온라인 조건이 바뀌어도 반응 가능한 autoregressive 생성에서 (2) sparse/불규칙한 제약을 입력으로 다루면서 (3) 긴 지평 목표까지 자연스럽게 연결되도록 컨텍스트를 설계하는 것입니다. ARDY는 variable history context를 갖는 two-stage autoregressive transformer denoiser를 사용해, 디노이징 루프 내에서 루트와 latent body를 상호 영향이 이어지도록 교차(interleaved) 예측하고, 제약은 masked motion sequence로 주입해 임의 시점/특징에 대한 장거리 목표를 네이티브로 학습합니다.

- **Empirical Impact**: HumanML3D 벤치마크와 고품질 대규모 Bones Rigplay 데이터셋에서 모션 품질과 제약 준수도가 함께 향상됨을 실증하며, 특히 설계한 혼합 표현과 two-stage 구조의 효과를 뒷받침합니다. 또한 마우스·키보드 기반 인터랙티브 데모에서 동적 텍스트 제어, 키프레임/패스 팔로잉, 장면 경로 추종 같은 사용 시나리오를 보여주며, 애니메이션·시뮬레이션·휴머노이드 로보틱스 쪽 실시간 제어 파이프라인의 현실적인 대안으로 주목받을 만합니다.



### Native Video-Action Pretraining for Generalizable Robot Contro (https://arxiv.org/abs/2607.08639)
- **Prior Approaches**: 기존 비디오-액션 모델은 관측에서 행동으로 직접 매핑하기보다, 장면이 어떻게 진화할지와 그 안에서 어떻게 행동할지까지 함께 예측해 일반화와 샘플 효율을 높였다. 하지만 많은 모델이 VAE(재구성 중심)와 bidirectional 비디오 생성 백본 같은 ‘디지털 콘텐츠용’ 구성요소를 그대로 재활용해, 동역학 정렬·표현 공간 일치·인퍼런스 지연·행동 신호 스케일링에서 구조적 한계를 노출한다. 또한 역방향/양방향 전제를 갖는 백본을 closed-loop의 단방향 시간 전개에 맞추는 retrofit은 사전지식의 약화를 유발할 수 있다는 지적이 나온다.

- **Core Contribution**: LingBot-VA 2.0은 로봇 embodiment를 전제로, 비디오 생성 모델을 붙여넣는 방식이 아니라 처음부터 ‘네이티브(native)’ 비디오-액션 파운데이션 모델로 설계했다. 핵심은 시맨틱 비주얼-액션 토크나이저로 관측과 행동을 하나의 의미 라탄 공간에 정렬하고, 그 위에 causal video-action 모델을 학습해 시간 구조까지 자연스럽게 맞춘다는 점이다. 결과적으로 few-shot(10–15 데모) 일반화와 경우에 따라 zero-shot 전환까지 확장되는 제어 능력을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 픽셀 재구성에 최적화된 잠재표현이 아니라 ‘의미와 행동’에 맞는 토큰 공간을 구성하는 것, (2) causal 학습 목표를 통해 웹 스케일 데이터에서도 행동 지식을 확장하는 것, (3) 실시간 closed-loop에서 모델 지연을 제어 지연으로 전환하지 않는 것이다. 논문은 semantic visual-action tokenizer로 언어-정렬 비주얼 파운데이션 특징에 맞춘 시맨틱 정렬과 self-supervised latent action 추출을 수행하고, 학습은 bidirectional-to-causal retrofit이 아닌 causal pretraining부터 시작해 catastrophic forgetting을 피한다. 또한 희소 MoE 백본으로 고주파 추론 효율을 확보하고, Foresight Reasoning 비동기 추론으로 실행과 예측을 병렬화하되 최신 관측에 매 rollout을 learned forward dynamics로 다시 근거(re-ground)하여 드리프트를 줄인다.

- **Empirical Impact**: 실세계 로봇 배치에서 LingBot-VA 2.0은 복잡한 조작 과제에 대해 few-shot generalization을 입증하며, 부족한 로봇 데이터에 의존하던 이전 방식보다 더 강한 기반 제어 지식을 웹 스케일 self-supervision으로부터 얻는 효과를 보여준다. MoE 인퍼런스, few-step consistency distillation, Foresight Reasoning, quantized 실행을 조합해 실시간 closed-loop 제어를 달성하며 최고 비동기 실행 주파수 225 Hz를 보고한다. 시뮬레이션과 실세계 평가에서 π0.5 및 LingBot-VA 같은 강한 베이스라인 대비 장거리 행동의 일관성과 조작 정밀도를 함께 개선해, 로봇 제어 파운데이션 모델 방향성에 의미 있는 실증을 제공한다.



### Systematic Evaluation of Learning Rate Scheduling Strategies Across Heterogeneous Architectures (https://arxiv.org/abs/2607.08511)
- **Prior Approaches**: 기존 연구들은 학습률 스케줄러를 비교적 ‘보조 하이퍼파라미터’로 취급하고 단일 스케줄 결과를 보고하는 경우가 많았습니다. Bayesian optimization, random search 같은 HPO는 탐색이 가능하지만 매번 전체 학습을 반복해야 해 비용이 크고, 스케줄 선택의 원인을 해석하기도 어렵습니다.

- **Core Contribution**: 이 논문은 LEMUR nn-dataset의 30개 이질적 아키텍처에 대해 스케줄러만 바꿔가며 CIFAR-10에서 정밀 비교 가능한 ‘정확도 지형(accuracy landscape)’을 구축합니다. 자동 source-code injection으로 25개 learning rate scheduling 구성을 주입해 총 3,938개 모델 변형을 평가했으며, 최우수 설정은 top-1 86.45%를 기록했습니다.

- **Technical Challenges**: 핵심 기술 과제는 아키텍처별 코드 스타일이 달라도 스케줄러 주입이 깨지지 않게 하면서, scheduler별 hyperparameter 등록까지 자동화하는 것입니다. 이를 위해 train_setup()과 learn()의 경계/들여쓰기 기반 탐지, step 호출 위치(에폭/배치) 자동 삽입, ast 기반 문법 검증과 hyperparameter 일관성 체크를 수행했고, LinearLR/PolynomialLR의 total_iters 스케일링 버그도 사전에 수정해 대규모 평가를 안정화했습니다.

- **Empirical Impact**: 결과적으로 스케줄러 선택은 아키텍처에 강하게 의존했으며, CosineAnnealingWarmRestarts와 CyclicLR 계열이 기본 decay보다 상위권에 더 자주 올라왔습니다. 237개 변형이 80%를 넘을 정도로 고성능 꼬리(upper tail)가 형성됐고, ReduceLROnPlateau는 5에폭 스크리닝 예산에서 평균 성능이 크게 저하되어 ‘예산 인지(budget-aware)’ 스케줄러 선택의 중요성을 보여줍니다.



### HoloTetSphere: Unified TetSphere Mesh Reconstruction for Physical Simulations (https://arxiv.org/abs/2607.08398)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 physics-ready 3D 재구성은 보통 표면 기하를 먼저 얻은 뒤 tetrahedralization으로 3D 메쉬를 만드는 두 단계 파이프라인에 의존해, 중간 단계 오류에 취약하다는 한계가 있다. Lagrangian 계열인 TetSphere Splatting은 연결성을 포함하지만 homeomorphic 제약 때문에 초기 위상에 묶여 토폴로지 적응이 어렵고, 결과적으로 disjoint tetrahedra가 생겨 물리 시뮬레이션에 부적합해진다.

- **Core Contribution**: 이 논문은 초기 메시에 의존하지 않고 최적화 과정에서 연결성 자체를 바꾸는 topology-adaptive tetrahedral mesh reconstruction을 제안한다. TetSphere의 아이디어를 확장해 Gaussian spheres-테트라요소를 end-to-end로 결합하고, 위상과 기하를 동시에 최적화함으로써 단일 연결의 watertight 메쉬를 목표로 한다.

- **Technical Challenges**: 토폴로지 변경을 미분 가능하게 만들기 위해, 테트라요소마다 독립 opacity를 학습하면 분절(chaotic holes/floaters)이 생기는 문제를 해결해야 했다. 저자들은 정점 공유 기반의 continuous opacity field를 만들어 differentiable element pruning을 수행하고, 삭제로 내부 요소가 표면 요소가 될 때 생기는 기하 열화는 alternating optimization과 two-stage smoothing(조건부 bi-harmonic/HC-Laplacian)으로 안정화했다.

- **Empirical Impact**: 여러 벤치마크에서 거리 기반 지표(Chamfer/Hausdorff 등)와 렌더링 품질(PSNR/SSIM/LPIPS) 모두에서 SOTA 대비 우수한 성능을 보였고, 단일 연결(single-connected) tetrahedral 메쉬 생성률도 가장 높게 보고된다. 물리 시뮬레이션 관점에서는 inverted ratio를 크게 낮추고 FEM 및 Isaac Sim 기반 중력 낙하 실험에서 안정적인 변형과 표면 오차의 점진적 증가(낮은 Chamfer L값)를 확인해, 기존 surface-추출-후 tetrahedralization의 취약성을 실증적으로 우회했다.



### Playing ZendoWorld: Challenging AI Agents on Active Visual Concept Induction (https://arxiv.org/abs/2607.08233)
- **Prior Approaches**: 기존 비주얼 귀납 및 규칙 학습 벤치마크들은 관찰 예시만으로 추론을 요구하는 경우가 많아, 에이전트가 새로운 장면을 제안해 가설을 갱신하는 ‘능동 실험’의 반복성을 잘 다루지 못했다. 또한 강화학습 기반 비주얼 에이전트는 관측→행동 최적화는 강하지만, 명시적 규칙/가설 공간을 회복하는 과정을 평가하기 어렵다. LLM이 실험을 제안하는 연구도 있었지만, 입력이 대개 깨끗한 기호 추상에 머물러 시각적 정합과 유도(인덕션)를 함께 분리해 보기 어려웠다.

- **Core Contribution**: 이 논문은 시각 관측을 바탕으로 숨은 논리 규칙을 추론하고, 규칙을 검증할 새 장면을 생성하며, 환경 피드백으로 가설을 갱신하는 폐루프 평가 환경 ZendoWorld를 제안한다. ZendoWorld는 Prolog 기반 DSL로 규칙 공간을 명확히 정의하고, 장면은 절차적으로 생성해 공정한 비교가 가능하도록 했다. 그 결과 ‘라벨 예측 정확도’와 ‘규칙 복원’ 사이의 간극 같은 핵심 실패 모드를 통합적으로 드러내도록 설계됐다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 고차원 이미지에서 객체 속성과 공간 관계를 상징 상태로 매핑하는 지각, (2) 관측된 예시로부터 올바른 논리 규칙을 유도하고 수정하는 인덕션, (3) 불확실성을 실제로 줄이는 실험 장면을 고르는 실험 설계가 동시에 작동해야 한다는 점이다. 논문은 시각-귀납-실험을 같은 상호작용 루프에 묶되, 시각 입력을 DSL/상징 상태로 분리해 병목을 진단할 수 있게 했다. 또한 렌더링-규칙 판정을 엄밀히 하기 위해 DSL 프로그램 정규화와 반례(counterexample) 제공, 그리고 예상 정보 이득(EIG)로 실험의 효율을 계량했다.

- **Empirical Impact**: 실험 결과, 에이전트가 관찰 예시에 대한 라벨은 비교적 잘 맞춰도 내부 규칙을 올바르게 복원하지 못하는 경우가 다수였고, 지각과 인덕션 병목이 에이전트 유형별로 다르게 나타났다. 특히 VLM 기반 에이전트는 EIG가 낮아 ‘거의 정보가 없는 실험’을 제안해 가설 불확실성을 줄이지 못했으며, 규칙 복잡도가 커질수록 성공률이 급격히 하락했다. 휴먼 데이터도 수집했는데, 사람은 더 안정적인 규칙 궤적을 보이며 복잡하거나 OOD 규칙 복원에서 VLM 기반 시각 에이전트가 따라가지 못해, 과학적 발견 같은 도메인에서 필요한 구조적 개선 방향을 구체화했다.



### SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation (https://arxiv.org/abs/2607.08161)
Comments:
          Accepted at IEEE SMC 2026

- **Prior Approaches**: Text-to-SQL은 자연어로 데이터베이스를 질의하는 핵심 작업이지만, 초기 규칙·템플릿 방식은 확장성이 낮았다. Seq2SQL, SQLNet처럼 스케치 기반 디코딩이나 강화학습을 쓰는 신경 접근은 성과를 끌어올렸지만, 복잡한 스키마로 일반화하기 어렵거나 학습/추론이 비싸지는 한계가 있었다.
또한 TAPAS, TaBERT, RAT-SQL, PICARD 등 구조 인식을 강화한 모델도 나왔지만, 최근 SOTA 성능은 T5/Codex 계열 LLM에 크게 의존해 자원 제약 환경 배포가 어렵다는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 SQuaD-SQL(Small-Qualified and Distilled for SQL)로, 작은 언어 모델이 LLM 수준에 가까운 Text-to-SQL 성능을 낼 수 있도록 teacher–student 학습 틀을 제안한다. 핵심은 LLM을 추론에 쓰지 않고, LLM이 생성한 구조화된 SQL 감독 신호(합성 데이터)를 통해 작은 모델이 “구조적 SQL 추론 행동”을 내재화하도록 하는 것이다.
여기에 parameter-efficient fine-tuning(LoRA)과 도메인 적응용 합성 데이터 생성까지 결합해, 제한된 자원에서도 실용적으로 학습·운용 가능하다는 점을 강조한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) LLM이 생성한 SQL 감독 신호가 가끔 문법·의미·실행 측면에서 틀릴 수 있다는 점과 (2) 작은 모델이 복잡한 구문 생성 능력을 안정적으로 학습하도록 만드는 학습 설계 문제다. 논문은 WikiSQL 스키마 정보를 활용한 task-specific 프롬프트로 합성을 시작하고, SQL 구문 파서 검증→LLM 기반 self-evaluation 점수(신뢰도 임계값)→규칙 기반 execution validation(실행 결과 유효성)이라는 3단계 필터로 노이즈를 제거한다.
그 다음 Qwen-1.5B를 LoRA로 학습하되, causal language modeling에서 프롬프트 토큰은 loss masking(-100)으로 제외해 SQL 생성 구간 학습에 집중하도록 최적화한다.

- **Empirical Impact**: WikiSQL에서 1.5B 파라미터 학생 모델(Qwen-1.5B)은 test execution accuracy 86.9%를 달성해, 전통적 Text-to-SQL 모델들과 task-specific 학습 없는 SLM을 크게 앞섰다. 또한 dev에서 86.5%로, 합성 감독이 성능 격차를 상당 부분 메워준다는 점을 확인했다.
특히 ablation 결과 prompt engineering 단독의 개선(35.6%→45.6%)보다, LLM 기반 distillation이 최대 기여를 했고(→80.3%), 데이터 필터링이 후속으로 정확도를 추가로 끌어올려 최종 86.9%를 만든 것으로 나타났다. 연구는 “스케일만이 아니라 지도 신호 품질·학습 전략”이 자원 제약 환경에서 실용적인 Text-to-SQL을 가능하게 한다는 메시지를 실증적으로 뒷받침한다.



### Equivariant Quantum Clustering with Differential Privacy: Parameter-Efficient Privacy-Preserving Analysis Across Heterogeneous Sensitive Datasets (https://arxiv.org/abs/2607.08092)
Comments:
          24 pages, 10+ tables, multiple figures, research article. Introduces Equivariant Quantum Clustering (EQC) integrating differential privacy with parameter-efficient quantum circuits for privacy-preserving clustering. Evaluated on NSL-KDD, CERT Insider Threat v6.2, and Synthetic MIMIC-III datasets

- **Prior Approaches**: 기존 프라이버시 보존 클러스터링은 차등 프라이버시 같은 기법으로 유틸리티를 지키려 하지만, 민감도·성능·복잡도 사이의 균형이 쉽지 않다. 또한 양자 클러스터링 연구도 회로 파라미터가 커지면 학습·일반화가 불리해지고, 프라이버시 강화 시 정확도 저하가 동반되기 쉽다.

- **Core Contribution**: 이 논문은 Equivariant Quantum Clustering(EQC)로, 대칭성을 반영한 quantum circuit을 p4m equivariant parameter sharing으로 경량화하면서 differential privacy를 함께 적용한다. 이를 통해 프라이버시 예산을 유지하면서도 유의미한 특징 표현을 보존해 프라이버시-유틸리티 트레이드오프를 개선하는 것이 핵심 기여다.

- **Technical Challenges**: 기여의 핵심 과제는 (1) 파라미터 효율을 높이면서도 클러스터링에 필요한 표현력을 유지하고, (2) differential privacy 적용에 따른 성능 손실을 최소화하는 것이다. EQC는 symmetry-aware 회로 설계를 통해 복잡도를 줄여 학습 부담을 낮추고, 프라이버시 설정(ε, δ)에 맞춘 보호를 결합해 tradeoff를 완화하는 방식으로 이를 해결한다.

- **Empirical Impact**: NSL-KDD에서 EQC는 clustering accuracy 79.3%를 달성하면서 membership inference attack 성공률을 privacy budget ε=1.0, δ=10^-5 조건에서 38.3%로 낮춰 주요 classical/quantum 대비 우수함을 보였다. 또한 CERT Insider Threat v6.2와 synthetic MIMIC-III에서도 검증했으며, ablation은 성능 향상이 파라미터 효율적 회로 설계와 differential privacy의 결합에서 주로 비롯됨을 시사한다.



### ConRad: Efficient Conformal Prediction for Radiomics (https://arxiv.org/abs/2607.08084)
Comments:
          Code available at this https URL

- **Prior Approaches**: 의료 영상에서 세그멘테이션 마스크로부터 추출한 방사선유전학(radiomics) 특징은 임상 의사결정에 점점 더 활용되지만, 실제로는 예측 마스크로 계산된 값이 다운스트림 입력이 된다. 기존 불확실성 정량은 주로 방사선유전학 값을 그대로 블랙박스로 보고 구간을 만들거나, 세그멘테이션 자체의 픽셀/복셀 수준 불확실성 지도에 집중해 왔다. 그 결과 세그멘테이션이 과신/보정 부족일 때 “더 신뢰되는 것처럼 보이는” 구간이 생길 수 있고, 다운스트림 특징에 직접 최적화된 효율 개선은 제한적이었다.

- **Core Contribution**: 이 논문은 ConRad라는 conformal prediction 기반 프레임을 제안해, 세그멘테이션에서 유도되는 스칼라 방사선유전학 타깃에 대해 적응형 예측 구간을 만든다. ConRad는 테스트 시점에 얻을 수 있는 마스크 기하·위상, 입력 이미지의 강도 분포, predicted radiomics 값, 그리고 경계 불확실성(boundary uncertainty)을 covariate로 CQR(conformalized quantile regression)에 넣어 구간 폭을 케이스 난이도에 맞게 조절한다. 동시에 split conformal의 교환가능성 가정 하에서 주변(마진) 커버리지를 유지하도록 캘리브레이션 조정까지 포함한다.

- **Technical Challenges**: 핵심 기술 과제는 “세그멘테이션 산출물이 만드는 방사선유전학 값”을 효율적으로 다루는 동시에, 타깃별로 스케일·안정성·해석 가능성이 크게 달라 평균 구간폭 비교가 어렵다는 점이다. 논문은 이를 해결하기 위해 타깃 필터링으로 결측/거의 상수/수치 불안정 타깃을 제거하고, 효율 평가는 target-level win rate와 기준선 대비 대칭 상대 개선(coverage가 유사할 때만 비교)을 사용한다. 또한 boundary uncertainty 제거 실험을 통해 구간 효율 개선에 기여하는 테스트 시점 특징을 구조적으로 진단할 수 있게 설계했다.

- **Empirical Impact**: 5개 2D 의료영상 데이터셋과 총 171개 보존된 방사선유전학 타깃에서 ConRad는 기존 split conformal 및 여러 CQR 기준선보다 target-level 구간 효율을 전반적으로 개선했다. 커버리지는 명목 miscoverage α=0.1에 대해 거의 준수(170/171 타깃에서 평균 커버리지 90% 이상)하며, 효율 향상은 특히 HAM10000, COVID-QU-EX, TG3K, TN3K에서 두드러졌다. ablation 결과 경계 불확실성(boundary uncertainty) 특징이 효율 이득의 가장 큰 원천(평균 정규화 ablation 중요도 40.5%)으로 나타나, 세그멘테이션 경계 주변의 불확실성이 다운스트림 방사선유전학 구간을 “좁히는” 데 결정적일 수 있음을 시사한다.



### Evaluating the Effect of Frame Rate in Sequence-Based Classification of Autism-Related Self-Stimulatory Hand Idiosyncrasies (https://arxiv.org/abs/2607.07957)
Comments:
          15 pages, 5 figures, 3 tables. Preliminary version presented as a poster at the AMIA 2024 Informatics Summit

- **Prior Approaches**: 기존 연구들은 ASD 선별을 다양한 데이터(설문·센서·질문지 등)로 확장했지만, 영상 기반으로는 CNN이 정적 프레임 또는 짧은 구간에 의존해 행동의 긴 시간 구조를 충분히 활용하지 못하는 한계가 컸습니다. SSBD 같은 자막 데이터셋은 규모가 작아(영상 75개) 성능이 62–76% 수준에서 머물렀고, 시계열 모델과 시간 샘플링·증강의 체계적 비교도 부족했습니다.

- **Core Contribution**: 본 논문은 비디오에서 self-stimulatory behavior(행동 스티밍)를 자동 분류할 때 (1) LSTM/GRU 같은 sequence 모델의 최적 아키텍처와 (2) temporal sampling rate 및 (3) 소규모 데이터에 맞는 augmentation 전략을 함께 정리했습니다. 특히 pose 기반 특징을 SSBD에 적용하고, sampling 간격 15프레임에서 LSTM(97.5%)·GRU(98.75%)가 CNN 대비 큰 폭으로 향상됨을 보여줍니다. 또한 I3D transfer learning 파이프라인에서 augmentation 10종의 기여도를 ablation으로 정량화해 upsampling이 핵심임을 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 작은 임상 비디오 데이터에서 temporal redundancy는 줄이되 diagnostically relevant한 시간 동학은 보존하는 sampling 설계와, 제한된 학습 샘플을 증강으로 얼마나 효과적으로 보완할지의 균형이었습니다. 저자들은 프레임을 1, 5, 15, 30, 45, 90 간격으로 샘플링해 시퀀스 길이 변화에 따른 성능을 비교했고, I3D 기반 전이학습 위에 수평/수직 플립·잡음·리샘플링(upsampling)·시간 변형 등을 조합한 뒤 leave-one-out ablation으로 “무엇이 성능을 깎는지”를 분석했습니다. 추가로 개인별(per-subject) 모델을 영상 내 temporally split 세그먼트로 학습·검증해, within-video 일관성의 가능성도 점검했습니다.

- **Empirical Impact**: 실험 결과, CNN 기반 기존 기준선(62–76%)을 LSTM/GRU가 일관되게 상회했으며 GRU는 최고 98.75% 정확도를 기록했습니다. augmentation 단독 비교에서는 horizontal flip이 48.78%로 가장 높았고, ablation에서는 upsampling 제외 시 training loss가 가장 크게 상승해(가장 큰 성능 저하) 임상용 소규모 비디오에서는 “샘플 수를 늘리는 증강”이 특히 중요함을 확인했습니다. 개인화 프로토콜은 per-video temporally split 설정에서 mean loss 1.84(SD 0.79)로 분산이 낮아, 짧은 관찰 구간 기반 캘리브레이션 같은 배치 스크리닝 방향에도 실증적 근거를 제공합니다.



### Time-to-Collision Based Dynamic Obstacle Avoidance Using Pretrained Vision Models for Robots in Unstructured Environments (https://arxiv.org/abs/2607.07885)
Comments:
          9 pages, 8 figures

- **Prior Approaches**: 기존 로봇 장애물 회피는 end-to-end 학습형(Transformer 등)이 강력하지만, 로봇별 대규모 데이터 수집이 병목이 된다. 다른 대안은 시뮬레이션에서 RL로 학습한 정책을 전이하는 방식인데, sim-to-real 격차(비주얼·접촉 역학·환경 다양성 불일치)로 인해 실제 환경에서 성능과 안전성 보장이 약해진다. 또한 TTC(충돌까지 남은 시간)를 학습으로 바로 예측하면 데이터 다양성 의존성과 해석 가능성 한계가 남는다.

- **Core Contribution**: 이 논문은 sim-to-real 전이를 피하면서도 해석 가능한 비전 기반 동적 장애물 회피를 수행하는 데이터 효율적 방법을 제안한다. 핵심은 UniDepth(단안 metric depth)와 SuperPoint+SuperGlue(장기 키포인트 대응)를 사용해 3D 키포인트 궤적을 복원하고, 각 키포인트의 TTC를 기하적으로 계산한 뒤 최소 TTC 키포인트를 회피 방향에 반영하는 것이다. end-to-end 학습 대신 명시적 3D 구조·시간 정보를 사용해 일반화성과 설명 가능성을 함께 노린다.

- **Technical Challenges**: 기여를 실제로 만들 때의 가장 큰 난관은 단안 depth의 오차가 3D 투영과 번들 조정, 최종 TTC 추정까지 연쇄적으로 전파되는 점이다. 이를 줄이기 위해 5프레임 슬라이딩 윈도우에서 XM solver로 scaled bundle adjustment를 수행하고, (1) 충분히 길게 트래킹되지 못한 키포인트, (2) 너무 먼 깊이(20m 초과), (3) 프레임 간 3D 변위가 큰 이상치(2m 초과)를 제외·재초기화해 기하 제약의 신뢰도를 높였다. 이후 최소 TTC 키포인트의 CPA(closest point of approach)를 기준으로 지면(ground plane) 2D 모션 프리미티브를 선택한다.

- **Empirical Impact**: M3ED(spot-forest, spot-outdoor-day) 실데이터 평가에서 TTC 1초 미만 프레임 식별 precision 0.49, recall 0.38을 기록했으며, 위협 프레임을 올바르게 감지한 경우 회피 방향 일치율은 84%였다. 특히 서로 다른 물리 장애물 22개 중 20개에서 최소 한 프레임(TTC<1s)을 포착해 실제 회피 트리거 관점에서 유의미한 관찰 성능을 보였다. 또한 모델 학습을 없애고 하이퍼파라미터 튜닝에 74초 데이터만 사용함으로써, 대규모 로봇별 학습이 필요한 기존 접근 대비 데이터 효율성과 해석 가능성을 동시에 입증했다.



### False Confidence: Automated Labels Confound Fairness Audits in Cervical Spine Segmentation (https://arxiv.org/abs/2607.07852)
Comments:
          8 pages, 1 figure. Under review at FAIMI 2026 (MICCAI workshop)

- **Prior Approaches**: 자궁경부 척추 MRI의 세그멘테이션은 임상 워크플로에 널리 쓰이지만, 공정성(audit) 자체는 거의 다뤄지지 않았다. 기존 연구들은 주로 다른 해부학(예: 요추) 또는 성별 같은 단일 속성에 집중했고, 세그멘테이션 평가에서 기준선(label reference)과 그 라벨 생성 출처를 교란변수로 다루지 못했다.

- **Core Contribution**: 이 논문은 CSpineSeg를 활용해 성별·나이·인종에 대해 최초로 자궁경부 척추 MRI 세그멘테이션의 인구집단 공정성 감사를 수행한다. 동시에 “silver 라벨(기계 생성)”과 “gold 라벨(전문가)” 중 무엇을 기준으로 점수화하느냐가 공정성 판정을 바꿀 수 있음을 실증한다.

- **Technical Challenges**: 핵심 난제는 mixed-provenance 데이터(전문가 gold 라벨 소수 + 기계 silver 라벨 다수)에서, 모델 성능·공정성 측정의 기준이 편향될 수 있다는 점이다. 연구진은 동일 예측을 gold vs 생성된 silver(해당 생성기 역할)로 각각 평가해, silver가 모델-라벨 간 누출로 인해 Dice 약 8포인트 과대평가를 만들고, 나이 공정성의 유의성도 ‘false confidence(집단 내 분산 붕괴)’ 형태로 뒤집힐 수 있음을 분리해 보여준다.

- **Empirical Impact**: 배포(realistic) 설정에서 모델은 성별·인종·나이에 대해 표준 지표 전반에서 공정하다고 관찰되지만, 라벨 출처가 결과를 크게 흔드는 것이 결론의 전환점이다. 따라서 향후 공정성 보고는 반드시 기준 라벨의 provenance(전문가/기계)를 명시하고, 가능하면 전문가 gold에 대해 성능과 공정성의 “크기(magnitude)”까지 함께 보고해야 한다는 실무 가이드를 제시한다.



### SASGeo: Stability-Aware Semantic Map Localization for GNSS-Denied UAVs -- A Framework and Synthetic Proof of Concep (https://arxiv.org/abs/2607.07737)
Comments:
          7 pages, 5 figures

- **Prior Approaches**: GNSS가 불안정한 환경에서 UAV는 VIO로 상대 위치를 추정하지만, 절대 관측이 없으면 오차가 누적된다. 이에 따라 UAV 이미지와 지오레퍼런스 항공·위성 이미지를 매칭해 절대 보정을 얻는 교차뷰 retrieval이 발전했지만, 외형 변화(계절·조명·시점·센서·지도 갱신 등)에 민감하다는 한계가 있다. 기존 연구는 OSM/벡터 지도, 의미 임베딩, 로드 BEV 보정, 그래프 매칭, 순차 필터 등 구성요소를 부분적으로 다뤘지만, 안전성에 필요한 요소(관측 밀도, 관계 검증, 시간적 일관성, 영속성 모델, 애매한 후보 거부 옵션)를 통합해 명확히 운영하는 방식은 부족했다.

- **Core Contribution**: 논문은 SASGeo가 환경을 픽셀이 아닌 도로·건물·수계·철도·교차로·경계 같은 persistent structure의 의미 지도(semantic map)로 표현해, 교차뷰에서도 위치를 구분하도록 설계했다고 제안한다. 의미 레스터 정렬과 관계형 그래프 증거, 시간/지도 나이 기반 persistence 신뢰도, 긍정·모순·unknown 관측의 명시적 처리, 그리고 애매한 고정(absolute fix) 거부를 한 프레임워크로 결합한다. 또한 ‘무엇을 어떻게 가중/판정할지’를 구체적인 모델과 의사결정 규칙으로 제시해, 막연한 semantic weight 제안 수준을 넘겼다.

- **Technical Challenges**: 핵심 난제는 (1) 외형이 크게 변하는 상황에서 의미 구조를 안정적으로 정렬하고, (2) 그래프 위상·관계가 잘못된 후보를 걸러내며, (3) 모호한 경우에는 고정 자체를 보류하는 integrity-aware decision을 만드는 것이다. 논문은 다중 프레임 VIO 누적으로 로컬 BEV 위에 의미 예측을 투영·누적하고, 증거를 긍정/모순/unknown으로 분해해 unknown을 부정 근거로 세지 않도록 했다. 이어 가시성·지도 나이·계절 적합성·지리적 구별성까지 반영한 persistence/distinctiveness 가중치를 쓰고, 후보 포즈의 불확실성과 그래프 일관성 및 VIO 잔차를 이용해 false fix 리스크 대비 수용률(risk–coverage) 관점의 거부/수용 판정을 제안한다.

- **Empirical Impact**: 실증은 실제 비행 closed-loop 성능을 검증하기보다는, 하드 decoy가 섞인 controlled 교차뷰 섭동에서 의미 기하가 위치를 구분하는지 “synthetic proof of concept”로 확인하는 데 초점을 둔다. 회전·스케일·부분 크롭·가림·지도 변경 시뮬레이션·혼동 가능한 decoy를 포함해 220개 무작위 재현 시험에서, global semantic descriptor는 Recall@1 58.6%에 그쳤지만 공간 의미 레스터 정렬 변형은 94.5–95.5%까지 크게 상승했다. Wilson 95% 구간은 descriptor와 공간 정렬 변형 간 분리는 보여주되, 공간 변형들 사이에는 구간 중첩이 있어 그래프/영속성/unknown 처리의 ‘추가 이득’을 통계적으로 확정하진 못한다; 다만 다음 단계로 필요한 aliasing·map-aging·거부 테스트를 더 어렵게 설계해야 한다는 점을 구체화한다.



### ReCoLoRA: Spectrum-Aware Recursive Consolidation for Continual LLM Fine-Tuning (https://arxiv.org/abs/2607.07719)
- **Prior Approaches**: LoRA 계열 PEFT는 백본 가중치를 고정하고 저랭크 업데이트만 학습해 효율을 높이지만, 태스크가 순차로 들어오면 같은 frozen weight 위에 업데이트를 계속 쌓으면서 이전 지식이 덮어써지는 문제가 생긴다. AdaLoRA, PiSSA, DoRA 같은 변형은 rank 배분·초기화·표현력을 개선하지만, 덮어쓰기 자체는 근본적으로 해결하지 못한다. O-LoRA처럼 간섭을 줄이려는 정형화된 정규화/부분공간 분리는 효과적일 수 있으나, 실용적 운용에서는 여전히 태스크 간 정보 재사용과 충돌 제어가 과제로 남는다.

- **Core Contribution**: 이 논문은 ReCoLoRA(Recursive Consolidation of Low-Rank Adapters)로 스펙트럼 기반(랜덤화 SVD) 초기화와 계층별 유효 rank(엘보 기준)를 도입해, 업데이트가 먼저 ‘주요 성분’ 방향을 사용하도록 설계한다. 더 핵심은 재귀적 통합(recursive consolidation)인데, 매 태스크 종료 후 원본 W0에 계속 쌓지 않고 현재까지의 effective weight를 다시 분해해 residual(고정 잔차)·slow principal(천천히 학습)·fresh fast adapter로 재구성함으로써 다음 태스크가 이전 태스크를 이미 흡수한 모델에서 출발하게 만든다. 또한 ReCoLoRA-TaskBank은 태스크별 독립 브랜치를 학습하고 oracle 라우팅으로 평가해, ‘오버라이트가 없는 경우’에 대한 상한을 제시한다.

- **Technical Challenges**: 연속 파인튜닝에서는 stability-plasticity 딜레마 때문에 새 태스크 적응성과 이전 성능 유지가 동시에 흔들리는데, 단순히 rank를 고정하거나 파라미터 공간에 앵커링하는 방식은 보호가 강하면 가소성이 죽고 약하면 기존 동작이 드리프트하는 것으로 나타났다. ReCoLoRA는 이를 피하기 위해 (1) 저랭크를 무작위로 찾기보다 pretrained weight 스펙트럼에서 principal 방향을 먼저 제공하고, (2) staged residual recovery로 잔차 용량은 필요할 때만 점진적으로 여는 학습 스케줄을 결합한다. 그리고 태스크 경계마다 현재 effective weight를 재분해해 slow principal을 보존하는 구조를 만들면서, 각 단계에서 principal/residual의 역할과 학습 속도를 분리해 간섭을 줄이도록 구현했다.

- **Empirical Impact**: 연속 GLUE 6-태스크 설정에서 ReCoLoRA는 4개 7~8B 백본 중 3개에서 최종 평균 점수(final average)가 rank-swept LoRA 계열·PiSSA·AdaLoRA·DoRA 및 관련 비교군을 앞섰고, 특히 학습 파라미터를 더 적게 쓰는 이점도 보고됐다. ReCoLoRA-TaskBank은 oracle 라우팅 하에서 오버라이트를 구조적으로 제거해 높은 상한 성능(예: Qwen3-8B 평균 forgetting 0에 근접)을 보였다. 다만 Llama-3.1-8B-Instruct에서는 LoRA rank 64가 더 좋은 결과를 보인 음성 사례도 있어, 스펙트럼/스케줄 민감도가 완전히 사라지지는 않는 것으로 해석된다.



### Who Gets Missed in the Tail? Thresholded Subgroup Underdiagnosis in Long-Tailed Chest X-ray Classification (https://arxiv.org/abs/2607.07717)
- **Prior Approaches**: CXR 장기치우침(long-tailed) 연구는 주로 클래스 단위 재가중(effective-number reweighting)이나 비대칭 손실(asymmetric loss)로 평균 성능·테일 성능을 끌어올리는 데 초점을 둔다. 공정성 연구도 equal opportunity나 GroupDRO처럼 집단 평균을 기준으로 집계하는 경우가 많아, 실제 배치 후 ‘어떤 (클래스, 서브그룹) 칸의 양성’이 임계값 아래로 빠지는지까지는 잘 드러나지 않는다. 특히 점수(score) 기반 랭킹 지표가 괜찮아도, 운영 임계값이 달라지면 놓치는 환자가 특정 서브그룹에 집중될 수 있다.

- **Core Contribution**: 이 논문은 임상 배치 관점에서 “테일 양성인데 (클래스, 서브그룹) 단위로 임계값(threshold) 적용 후 미검출되는가”를 감사(audit) 단위로 정의한다. 진단 래더(diagnostic ladder)를 통해 클래스 수준 손실/가중, 서브그룹 인지 학습, group robustness, 그리고 점수-의사결정 변환(임계값 선택)이 각각 ‘누가 missed 되는지’에 어떻게 영향을 주는지 분리해 보여준다. 즉, 공정성 판단이 라벨 빈도나 매크로(mAP) 같은 랭킹 지표만으로 결정되지 않음을 구체적 감사 지표로 고정한다.

- **Technical Challenges**: 핵심 기술적 난제는 “동일한 점수 모델이라도 임계값만 바꿨을 때 서브그룹 테일 FNR이 어떻게 달라지는가”를 분리해 측정하는 것이다. 이를 위해 M5(서브그룹-클래스 가중 학습)와 M6(동일한 점수 유지 + 테일 클래스에 대해 worst-group recall 중심 임계값을 재선택)를 비교해 threshold-mediated 효과만 격리한다. 또한 작은 테스트 셋에서 테일 양성 표본이 매우 적기 때문에, bootstrap 구간과 paired bootstrap 대비로 ‘작은 표본에서도 결론이 일관적인지’를 확인한다.

- **Empirical Impact**: VinDr-CXR에서는 group-tail weighting 후 tail-aware thresholding으로 테일 FNR이 0.665→0.269로, sex 최악 그룹 FNR이 0.705→0.157, age 최악 그룹 FNR이 0.822→0.133으로 크게 낮아지며 macro-mAP도 0.611→0.635로 소폭 개선된다. 반면 MIMIC-CXR/CXR-LT에서도 같은 score-to-threshold 비교는 테일 FNR과 여러 서브그룹 최악 그룹 FNR을 낮추지만, 잔여 missed-positive 비율이 여전히 높아 “완전 해결”이라기보다 임계값 기반 감사 측정이 문제를 더 잘 드러낸다는 점을 강조한다. 결론적으로 희귀 라벨 공정성은 (1) 어떤 질환을 발견하는지, (2) 어떤 서브그룹인지, (3) 어떤 운영 임계값으로 ‘검토 알림’을 만들지의 조합에 좌우되며, 랭킹 지표나 집단 평균 강인성만으로는 ‘누가 빠지는지’를 예측하기 어렵다는 메시지를 준다.



New uploads on arXiv(cs.AI)

### Ideas Have Genomes: Benchmarking Scientific Lineage Reasoning and Lineage-Grounded Idea Generation (https://arxiv.org/abs/2607.08758)
- **Prior Approaches**: 기존 과학 분야 AI 연구는 아이디어 생성이나 문헌 이해를 다루더라도, “아이디어가 어떻게 계승·수정·삽입되는가”라는 계보(inheritance) 구조를 벤치마크로 일관되게 검증하긴 어려웠다. 또한 선행연구를 일부 참조하는 정도는 있었지만, 계보 추론과 계보-근거 생성이 같은 틀에서 평가되지는 않는 경향이 있었다. 그 결과 현재 벤치마크는 AI의 ‘유전적 과학적 계승’ 수행 여부를 충분히 드러내지 못했다.

- **Core Contribution**: 이 논문은 과학 계보 추론과 계보에 근거한 아이디어 생성을 동시에 평가하는 벤치마크 IdeaGene-Bench(IG-Bench)를 제안한다. IdeaGene 프레임워크로 논문/제안을 증거-근거, 최소 단위, 타입을 갖춘 Idea Genome 객체들의 집합으로 표현하고, GenomeDiff로 계승, 돌연변이, 소실, 외부 import, novel insertion을 여섯 가지 진화 동역학에 맞춰 정렬·기록한다. IG-Exam은 닫힌 형태의 계보 추론을, IG-Arena는 주어진 계보 집단의 ‘일관된 후손’으로 제안을 삽입하는 생성 평가를 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 계보 구조를 강건하게 추론·생성할 수 있도록, 텍스트를 단순 인용 수준이 아닌 타입화된 증거-근거 객체와 차이 정렬로 환원하는 것이다. 저자들은 GenomeDiff의 정렬을 통해 inheritance·mutation·loss·external import·novel insertion을 체계적으로 기록하고, IG-Arena에서는 lineage-conditioned Population-Evolution Score(PES)로 “맞는 계승, 의미 있는 변이, 미래 선택 가치”를 함께 요구한다. 다만 실험에서 LLM 기반 과학자들이 compositional bottleneck에 막혀 성능이 제한됨을 보여준다.

- **Empirical Impact**: 14개 LLM 기반 과학자 실험에서 계보 추론의 최고 성능은 exact accuracy 27.3%에 그쳐, 계보 기반 과학 추론이 아직 어렵다는 실증적 신호를 제공한다. 또한 구조화된 계보 컨텍스트는 시스템 전반을 고르게 돕기보다 순위 재편(re-rank reshuffling)을 유발해, 데이터/추론 방식이 정교하게 설계되지 않으면 이점이 일관되지 않음을 시사한다. IG-Bench는 10개 과학 도메인에서 1,961개 golden lineage trace와 생성·추론 양면 평가를 제공함으로써, 향후 계보 중심 연구·모델링을 촉진하는 기준점이 될 것으로 기대된다.



### Using AI-based Learning Assistants in Higher Education: A Large-Scale Descriptive Analysis (https://arxiv.org/abs/2607.08748)
- **Prior Approaches**: 기존 교육용 챗봇 연구는 비교적 소규모 표본이나 설문 기반의 자기보고 자료에 의존하는 경우가 많았다. 이 때문에 실제 학습 과정에서의 “어떻게 쓰이는지”에 대한 대규모 사용행태 근거가 부족했다. 또한 학습자 특성(성별·연령 등)과 학업 구조(전공군·학위·수업 형태)별 차이를 체계적으로 분해한 분석도 제한적이었다.

- **Core Contribution**: 본 연구는 고등교육에서 AI 기반 학습 보조도구 Syntea의 사용을 대규모 로그 데이터로 기술(large-scale descriptive analysis)한다. 77,543명의 원격학습 학생 객관 로그를 바탕으로 성별, 연령대, 전공군, 학위, 학습 모드에 따른 사용 패턴을 실증적으로 제시한다. 나아가 Syntea가 이미 많은 학습자들의 학습 루틴에 내재되어 있음을 보여주되, 맥락에 따라 사용이 달라진다는 점을 근거로 제시한다.

- **Technical Challenges**: 대규모 로그 기반 분석에서는 사용 행태를 신뢰성 있게 비교하려면 다양한 하위집단을 공정하게 분해하고, 실제 사용 행동을 설명하는 지표를 일관되게 정의해야 한다는 문제가 있다. 연구진은 방대한 객관 로그를 활용해 인구통계·학업 구조 변수를 기준으로 사용 패턴을 체계적으로 매핑한다. 이를 통해 설문 편향이 아닌 실제 상호작용 기반의 차이를 드러내도록 설계했다.

- **Empirical Impact**: 실증 결과는 Syntea가 이미 상당수 학습자에게 일상적으로 활용되고 있음을 시사하면서도, 집단·구조별 사용 격차가 존재함을 명확히 보여준다. 이러한 패턴은 향후 AI 학습 지원 도구의 고도화(타깃 기능, 맥락별 제공 방식 등)를 위한 경험적 근거가 된다. 더불어 고등교육에서 교육용 챗봇 사용을 대규모로 분석한 자료로서, 분야 전반의 연구 설계와 평가 기준을 넓히는 데 기여한다.



### AUTOPILOT VQA: Benchmarking Vision-Language Models for Incident-Centric Dashcam Understanding (https://arxiv.org/abs/2607.08745)
Comments:
          CVPR Autopilot Workshop

- **Prior Approaches**: 기존 비전-언어 모델(VLM)·대규모 언어 모델(LLM) 기반 연구는 자율주행에서 장면 이해, 추적, 예측 등 “일반 인지” 성능을 끌어올리는 데 집중해 왔습니다. 그러나 사고·아찔한 상황처럼 드문 안전 중요 이벤트를 신뢰성 있게 추론/설명하는 평가체계는 부족했습니다. DrivingVQA, MetaVQA, NuPlanQA 같은 VQA·추론 벤치마크도 있으나, 정형화된 사고 장면 이해와 시간적·인과적 추론 요구까지 충분히 담지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 대시캠 비디오를 바탕으로 한 incident-centric visual question answering 벤치마크 AUTOPILOT-VQA를 제안합니다. 충돌, near-miss, 예방된 위험, no-incident를 포함해 실제 도로 사고에 가까운 “사건 중심” 질문을 설계하고, 환경 맥락부터 사고 상세(관련 주체, 원인 귀속, 충돌/피해 위치, 회피 가능성 추론)까지 답하도록 구성했습니다. 이를 통해 객체 인식 수준을 넘어 temporally grounded하고 safety-aware한 추론을 평가하는 표준을 제공합니다.

- **Technical Challenges**: 핵심 난제는 영상만으로는 희소하고 복합적인 사건 원인을 추적해야 한다는 점이며, 단순 분류가 아니라 주변 맥락·행동 상호작용·사건 발생 조건을 시간적으로 연결해 정형 답을 산출해야 합니다. 또 질문이 상황에 따라 “해당/비해당(unknown and non-applicable)”이 될 수 있어, 모델이 증거가 부족한 경우를 구분하며 질문 적용 가능성을 판단해야 합니다. 논문은 600+ 클립과 6,000+ Q&A를 통해 환경·도로·주체·사고 범주·fault/avoidability 등 6개 의미 묶음에 걸친 구조화 질의를 제공해, 이러한 추론 요구를 벤치마크로 강제합니다.

- **Empirical Impact**: AUTOPILOT-VQA는 AUTOPILOT CVPR 2026 경쟁으로 공개되었고, Kaggle에서 224명 등록·73명 활동·59팀·총 686회 제출로 높은 참여를 끌었습니다. 리더보드 상위 점수는 0.65835 수준으로, 상위권이 0.60을 넘기긴 하지만 인간 수준의 신뢰성에는 못 미친다는 점이 드러났습니다. 또한 다수 팀이 0.39–0.40 구간에 몰리는 장기 꼬리 분포를 보여, 단순 지각 기반 접근만으로는 원인·예방 조치·피해 지점 같은 인과/관계 추론을 전반적으로 마스터하기 어렵다는 시사점을 줍니다. 결과적으로 이 벤치마크는 안전 중요 추론의 평가·해석 가능하고 견고한 vision-language 파이프라인 개발을 촉진하는 실증적 기준점으로 자리잡는다는 의미가 있습니다.



### Workflow as Knowledge: Semantic Persistence for LLM-Mediated Workflows (https://arxiv.org/abs/2607.08740)
Comments:
          39 pages, 18 figures

- **Prior Approaches**: LLM 워크플로우는 도구 사용, 검색, 분기, 체크포인팅, 인간 승인 같은 실행 제약을 명시적으로 다루는 방향으로 발전해 왔다. 기존 워크플로우 시스템들은 이런 운영 이슈를 상당 부분 해결하지만, 워크플로우 자체를 ‘지식 객체’로 지속·검사·재개 가능하게 의미론적으로 모델링하는 관점은 상대적으로 약하다.

- **Core Contribution**: 이 논문은 Lisp에서 영감을 받은(언어 독립적인) 개념 모델을 제안하며, 이를 구현이 아닌 설명의 렌즈로 사용한다. 워크플로우 정의/인스턴스/추론 기록/컨텍스트 스냅샷/의존성 관계를 공용 지식 기저(shared knowledge substrate) 위의 persistent knowledge objects로 표현해 의미적 지속성을 목표로 한다. 핵심 구분은 derive(가용 상태에 대한 결정적 계산)와 infer(선언된 컨텍스트와 실행기 제어 capability policy 아래의 LLM 판단)이다.

- **Technical Challenges**: 기여를 실현하려면 워크플로우 전 과정의 산출물뿐 아니라 ‘전이 의미(transition semantics)’와 재개·검사에 필요한 상태 스냅샷을 일관되게 고정(persist)하는 설계가 어렵다. 논문은 구현 세부 대신 derive/infer의 의미론적 경계를 중심으로, 객체 정체성과 live-image thinking을 통해 워크플로우를 inspectable, resumable, reviewable 지식 객체로 바라보는 개념 틀을 정리한다. 다만 전이의 형식적 의미는 향후 연구 과제로 남긴다.

- **Empirical Impact**: 제공된 초록 범위에서는 실제 실험이나 벤치마크 결과를 통한 정량적 검증은 언급되지 않는다. 그럼에도 이 모델은 워크플로우가 단순히 로그를 남기는 수준을 넘어 ‘검토 가능한 지식 자산’으로 남을 수 있다는 설계 방향을 제시해, 향후 워크플로우 시스템의 의미론적 정합성과 감사 가능성 연구에 영향을 줄 수 있다.



### The Illusion of Equivalency: Statistical Characterization of Quantization Effects in LLMs (https://arxiv.org/abs/2607.08734)
- **Prior Approaches**: 기존에는 post-training quantization을 평가할 때 정확도와 perplexity 같은 성능 지표에 거의 전적으로 의존해 왔다. 이런 방식은 양자화로 인해 모델의 실제 “행동”이 바뀌는지, 즉 같은 입력에서 정답을 맞히는 패턴이 어떻게 달라지는지를 잘 드러내지 못한다.

- **Core Contribution**: 이 논문은 기준 모델과 양자화 모델 간의 정답 예측이 얼마나 겹치는지(정답 예측의 일치도, correctness agreement)를 decision-level에서 측정하는 지표를 제안한다. 절대적인 정확도와 무관하게, 양자화가 유발하는 예측 행동 변화(behavioral divergence)를 더 직접적으로 포착한다.

- **Technical Challenges**: 핵심 기술적 난제는 양자화가 생기는 구조적 변화(특히 attention weight에 대한 영향)를 행동 변화와 연결해 정량화하는 것이었다. 저자들은 attention을 구조 연산자로 보고, 통계적·분포 기반 layer-wise 왜곡을 측정해 저비트에서 비선형 “breakpoint”가 나타나며 query/key projection이 value/output projection보다 일관되게 더 민감하다는 점을 보여준다.

- **Empirical Impact**: 여러 모델과 8-bit부터 2-bit까지 다양한 양자화 설정에서, 작업 성능이 겉보기엔 보존돼도 행동이 갈라지는 현상이 확인됐다. 이는 base와 quantized 모델이 동등해 보이는 “착시”를 깨고, 기존 성능 지표를 넘어 행동 기반 평가의 필요성을 강하게 뒷받침한다.



### Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents (https://arxiv.org/abs/2607.08716)
- **Prior Approaches**: 기존 메모리·롱컨텍스트 접근은 주로 정보를 저장하거나(retrieval, long-context memory) 필요한 항목을 꺼내는 데 초점을 맞춘다. 그런데 장기 작업에서는 긴 트래젝터리 전체가 문맥에 남아 있어도, 요구사항·환경 사실·실패 진단 같은 의사결정 관련 상태가 다음 행동에 제대로 영향을 주지 못하는 ‘behavioral state decay’가 발생한다.

- **Core Contribution**: 이 논문은 메모리를 수동 저장/검색이 아니라 ‘개입(intervention) 정책’으로 재정의한다. 메모리 에이전트가 액션 에이전트와 병렬로 동작하며, 메모리 뱅크를 갱신한 뒤 다음 턴 호출에 메모리 근거의 짧은 리마인더를 주입할지(또는 침묵할지) 선택한다.

- **Technical Challenges**: 핵심 난제는 “무엇을 기억할지”를 넘어 “언제, 어떤 형태로 행동 루프에 다시 넣을지”를 학습/제어하는 것이다. 이를 위해 구조화된 memory bank(지식 메모리·절차 메모리·비공개 상태)와 2단계 파이프라인(뱅크 관리 도구콜 → 개입 선택)을 두고, 개입이 불필요하면 null intervention으로 명시적으로 침묵하게 한다.

- **Empirical Impact**: Terminal-Bench 2.0과 τ2²-Bench에서 메모리 개입은 action agent 강도에 무관하게 pass@1을 향상시켰다. 특히 +8.3pp(Terminal-Bench)와 +6.8pp(τ2²-Bench) 수준의 개선이 보고됐고, ablation에서는 passive 노출·always-on 주입·advisor-only·일반 retrieval보다 selective intervention이 더 강건했다. 또한 Qwen3.5-27B를 SETA로 SFT와 GRPO 학습해 open-weight 메모리 정책의 부분 전이를 보이는 등, 실제 적용 가능성을 초기 단계에서 시사한다.



### SolarChain-Eval: A Physics-Constrained Benchmark for Trustworthy Economic Agents in Decentralized Energy Markets (https://arxiv.org/abs/2607.08681)
- **Prior Approaches**: 기존 에이전트 평가는 주로 과업 성능에 초점을 두며, 사이버-물리 환경에서는 신뢰worthiness(안전성·거버넌스 안정성·조작 가능성)까지 함께 검증하기 어렵다는 한계가 있었다. 특히 분산 에너지 시장처럼 물리 제약과 경제적 인센티브가 얽힌 환경에서는 잘못된 물리 데이터로 인한 부정행위나 인공 유동성, 불안정한 의사결정이 평가에서 누락되기 쉽다.

- **Core Contribution**: 본 논문은 물리 제약 기반의 경제적 에이전트 평가 벤치마크 SolarChain-Eval을 제안한다. 시장 거버넌스를 Gymnasium 호환 Markov Decision Process로 모델링하고, 시간당 의사결정을 수행하는 정책을 시장 효용, 물리 안전, slippage, action smoothness, 공간 공정성, auditability 등 다차원으로 평가한다.

- **Technical Challenges**: 핵심 과제는 ‘경제적으로는 유리하지만 물리적으로는 틀릴 수 있는’ 행동을 체계적으로 제약하고, 위험한 행동에 대한 검증·수정 과정을 평가에 포함하는 것이다. 이를 위해 episode 단위 action bounds와 감사 규칙을 정의하는 LLM-based Planner/Auditor 계층을 넣고, 고위험 행동을 Auditor가 검토·수정하며 개입 내역을 트리거 신호·제안 행동·수정 행동·감사 근거까지 구조화 로그로 남긴다.

- **Empirical Impact**: 실험에서 정적/랜덤/단기추종/myopic/RL/RL+LLM 정책을 비교한 결과, 전반적으로 utility–safety 트레이드오프가 명확히 드러난다. RL은 시장 효용을 높이지만 여전히 unsafe 행동을 낼 수 있으며, 물리 패널티를 제거하면 보상 극대화 에이전트가 잘못된 생성물을 악용해 인공 유동성을 키운다. LLM Planner/Auditor는 auditability를 개선하고 일부 위험을 완화하지만, 잘못 지정된 reward function을 완전히 상쇄하진 못한다는 점에서 ‘물리 제약+투명한 개입 흔적’이 신뢰성 있는 에이전트 평가의 조건임을 시사한다.



### Formal Mechanisms for Market Stability in Self-Interested Agent Societies: A Marketplace Simulation Study (https://arxiv.org/abs/2607.08652)
Comments:
          23 pages, 8 figures

- **Prior Approaches**: 기존 연구는 반복되는 사회적 딜레마에서 이기적 에이전트가 배신으로 기울어 협력 기반의 교환 이득이 붕괴되는 문제를 다뤄왔다. 다만 많은 접근은 제한적 커뮤니케이션/규제 설정에 의존하거나, “자유로운 커뮤니케이션 위에 어떤 공식 메커니즘을 얹어야 시장 안정이 유지되는가”를 충분히 체계화하지 못했다.

- **Core Contribution**: 이 논문은 무제한 커뮤니케이션이 허용된 환경에서도 시장 안정성을 지키는 데 필요한 “레이어드” 공식 메커니즘이 무엇인지 제시한다. 18개 LLM 에이전트(DeepSeek-V3)가 전문 생산을 바탕으로 제약된 사회적 네트워크에서 거래하도록 만든 멀티에이전트 마켓 시뮬레이션에서, 여러 메커니즘 중 Mediation이 가장 안정적임을 확인한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 에이전트들이 자유롭게 메시지를 주고받는 상황에서 협력을 유지하는 규칙이 무엇인지, (2) 그 규칙이 적대적 공격(트롤)에 의해 얼마나 쉽게 무너지는지를 정량화하는 것이다. 저자들은 200라운드에 걸쳐 점진적으로 troll injection을 가하며 메커니즘을 비교하고, 이어 Mediation을 iteratively prompt-optimised LLM-driven trolls로 red-teaming해 공격을 최적화하는 방식으로 견고성을 측정한다.

- **Empirical Impact**: 실험 결과 Mediation은 8개 조건 중 최고 성능을 보였고, 최선의 공격(v6)에서도 정직한 에이전트의 효용을 13.3% 감소시키는 데 그쳐 시장 붕괴는 일으키지 못했다. 또한 Mediation은 공격이 지속되는 상황에서도 회복을 가능하게 하며, 저자들은 이를 “adversarial robustness=최적 공격 하에서도 양(+)의 정직 에이전트 효용을 유지하는 능력”으로 정의해 ‘휘어지지만 부서지지 않는다’는 결론을 제시한다.



### The complexities of patient-centred conversational artificial intelligenc (https://arxiv.org/abs/2607.08625)
Comments:
          36 pages (main text), 129 pages (supplementary materials)

- **Prior Approaches**: 기존의 건강 상담용 LLM 챗봇 개발과 평가는 대체로 협조적이고 말이 잘 통하는 시뮬레이션 환자에 의존해 왔다. 하지만 실제 환자들의 대화 패턴과 감정 표현은 사용자마다 크게 달라 현실과의 간극이 생긴다.

- **Core Contribution**: 이 논문은 2,053건의 실제 환자-챗봇 대화를 분석해, 환자 시뮬레이터가 임상 내용뿐 아니라 정서 상태, 대화 전략, 커뮤니케이션 스타일을 분리해 모델링해야 함을 제안한다. 나아가 Turing-inspired 현실성 평가와 환자 personae(5종)를 통해 모델을 다면적으로 시험한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘사람처럼 보이는’ 동시에 임상적으로도 의미 있는 대화를 생성하도록, 환자의 커뮤니케이션 다양성을 구조적으로 반영하는 것이다. 이를 위해 시뮬레이터에서 임상/정서/대화전략/표현양식을 분리 모델링하고, 15명의 평가자가 구분하기 어렵게 현실성을 검증하는 방식으로 설계를 다듬었다.

- **Empirical Impact**: Turing-inspired 현실성 평가에서 시뮬레이션 대화는 실제 대화와 거의 구별되지 않았고, 평가자의 분류 정확도는 55%에 그쳐 ‘거의 비슷함’을 보여줬다. 또한 1,164건의 clinician-graded 사례에서 네 가지 LLM의 urgency assessment를 비교했을 때, 환자의 커뮤니케이션 스타일 차이가 트리아지 결과에 유의미하게 영향을 주어, 이상화된 상호작용만 가정한 시스템은 실제 배치 시 성능 저하와 건강 격차 확대 위험이 있음을 시사한다.



### Towards Precision Therapy in Hepatocellular Carcinoma: A Clinical-Reasoning LLM for Risk Stratification and Treatment Guidanc (https://arxiv.org/abs/2607.08602)
- **Prior Approaches**: 기존 가이드라인과 병기 체계는 대개 거친 범주로 환자를 나눠, 같은 병기 안에서도 생기는 이질성과 실제 EMR 임상 맥락을 충분히 반영하지 못했다. 결과적으로 위험도·치료 선택이 표준 문구 중심으로 고정되거나 환자별 상황을 놓치는 경우가 생겼다. 또한 텍스트 수준에서 가이드라인을 ‘따라 읽는’ 방식은 근거 재현이나 단계 검증이 약하다는 한계가 지적됐다.

- **Core Contribution**: 논문은 EMR 서사(기록 내 이야기)를 읽어 위험 점수 기반 병기화, 가이드라인과 일치하는 치료 권고의 순위, 근거를 동반한 설명, 개인화 생존 추정을 함께 생성하는 임상 정렬형 LLM HCC-STAR를 제안한다. HCC-STAR는 SEER 기반 약 3만 건을 EMR 스타일 서사로 증강해 학습하고, 임상의 의도에 맞춘 프롬프트 기반 데이터 생성 워크플로를 포함한다. 단순 요약이나 규칙 인용을 넘어, 의료 맥락에 맞춘 의사결정 지원을 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) EMR 서사에서 환자 상태·맥락을 정확히 추출해 위험도와 치료 권고로 연결하는 것, (2) 가이드라인 텍스트를 외우는 수준을 넘어 ‘검증 가능한’ 추론을 수행하는 것, (3) 단계별로 확인 가능하면서도 최종 성능을 높이는 보상 설계를 구현하는 것이다. 논문은 지식 정렬(knowledge-aligned) 추론 프레임워크와 단계 검증이 가능한 composite reward를 최적화해 텍스트 memorization을 줄이도록 설계했다.

- **Empirical Impact**: 중국 12개 병원의 6,668명 다기관 코호트에서 HCC-STAR는 치료 추천과 위험도 분류에서 기존 가이드라인 및 경쟁 모델들(GPT-5, Gemini-2.5 Pro 등) 대비 최상 성능을 보였다. 가설적 전체생존 분석에서는 HCC-STAR 권고를 따를 때 중앙값 생존이 51개월로, BCLC 29개월과 CNLC 32개월보다 높게 나타났다. 또한 블라인드된 간담췌 전문의 평가에서 추론과 근거 설명의 신뢰성이 높았고, 실제 사용 시 의료진의 정확도를 높이면서 의사결정을 더 빠르게 만드는 데도 도움을 줬다.



### SHAP-Weighted Cross-Modal Expert Fusion for Emotion and Sentiment Recognition: Evidence and Limits (https://arxiv.org/abs/2607.08573)
- **Prior Approaches**: 기존 멀티모달 감정·감성 인식은 early fusion과 late fusion으로 많이 나뉜다. Early fusion은 입력을 결합한 뒤 한 모델로 분류해 정확도는 좋지만, 구조가 단일(monolithic)해 해석과 제어가 어렵다. Late fusion은 각 모달의 예측기를 독립적으로 학습해 모듈적이지만, 모달 간 상호작용을 충분히 반영하지 못할 수 있다.

- **Core Contribution**: 본 논문은 XAI-guided adaptive fusion (XGAF)을 재조명하며, TreeSHAP attribution 크기로 샘플 수준 가중치를 정하는 트리 기반 mixture of unimodal experts와 cross-modal experts 설계를 분석한다. 특히 전문가들의 입력 특징 차원(dimensionality)이 서로 다를 때, SHAP attribution reduction 방식이 성능과 모달 기여 분배에 미치는 영향을 집중적으로 다룬다. 핵심 결론은 sum-abs reduction이 attribution 질량을 보존해 high-dimensional cross-modal 전문가를 억누르지 않는다는 점이다.

- **Technical Challenges**: 어려운 점은 SHAP 기반 가중치를 만들 때, mean-abs·median-abs 같은 감소(reduction)가 차원이 큰 전문가의 attribution을 상대적으로 불리하게 만들 수 있다는 조건 변화다. 논문은 mean-abs/median-abs/sum-abs 감소를 비교하고, 각 감소 방식이 트리 라우팅(가중치)의 분포를 어떻게 바꾸는지 진단한다. 그 결과 sum-abs는 총 attribution mass를 유지해 cross-modal(특히 trimodal) expert에 가중치가 집중되면서도 성능 저하가 덜하다는 패턴을 보였다.

- **Empirical Impact**: MELD 7-class 감정 인식에서 sum-abs XGAF는 세 가지 face-sequence aggregator 전반에 걸쳐 early fusion과 거의 비슷한 성능을 보였고, Transformer 변형은 0.5983 F1로 early fusion(0.6018)과 큰 차이가 없었다( late fusion probability-average는 0.4598). McNemar 검정 결과 sum-abs XGAF와 early fusion은 유의한 차이가 없었지만(p=1.000), late fusion과는 유의하게 우수했다(p<0.0001). CMU-MOSEI 3-class 감성 인식에서도 sum-abs XGAF가 0.6519 F1로 early fusion(0.6485)과 late fusion(0.5696)을 모두 앞섰고, ablation·진단 분석은 성능 이득이 복잡한 per-sample routing보다 cross-modal expert(특히 trimodal) 추가에서 주로 발생함을 보여 이 분야에 해석 가능한 융합 설계 가이드를 제공한다.



### CommuniWave:A Machine Learning Model for Quantifying the Degree of Temporary Informal Behavior in Urban Communities (https://arxiv.org/abs/2607.08554)
Comments:
          17 pages, 4 figures. Presented at ASCAAD 2024

- **Prior Approaches**: 기존 도시 커뮤니티 계획은 상향식보다 하향식(top-down)에 치우치는 경우가 많고, 거주자의 비공식적 행동(조작·이탈·비정형 활동)을 수치로 포착·평가할 수 있는 지표가 부족하다는 한계가 지적된다. 그 결과 거리 영상 등에서 관측되는 현실의 행동 패턴과 계획 간 불일치가 커져, 갈등이 반복될 수 있다.

- **Core Contribution**: 이 논문은 CommuniWave라는 머신러닝 모델을 제안해 도시 커뮤니티의 비공식 행동 정도(Degree of Informal Behavior, DIB)를 효율적으로 탐지·정량화한다. 거리 영상으로부터 DIB의 변동(흐름)을 차트화해, 계획과 현장 간 간극을 더 정밀하게 추적할 수 있도록 지원한다.

- **Technical Challenges**: 핵심 과제는 (1) 영상에서 비공식 행동을 신뢰도 높게 포착하는 것과 (2) 포착된 정보로 DIB를 재현 가능하게 점수화하는 것이다. 연구진은 mmaction2 기반의 Behavior Capture Net(BCN), 자체 개발 YOLOv10 모델(YLX), 그리고 random forest로 구성된 Behavior Eval Model(BEM)을 결합해 행동 검출→평가로 이어지는 파이프라인을 구축했다.

- **Empirical Impact**: 제안된 방식은 거리 비디오로부터 DIB fluctuation chart를 생성해, 도시 관리자 입장에서 동적 모니터링과 의사결정에 활용 가능한 형태로 결과를 제공한다. 복잡성과 불확실성이 큰 도시 환경에서 커뮤니티의 영토적 회복탄력성(territorial resilience)을 높이기 위한 실증적 관찰 도구로서 의미가 크다.



### AI-guided stimuli discovery and generation to optimize facial emotion perception studies in autism (https://arxiv.org/abs/2607.08533)
- **Prior Approaches**: 자폐 성인과 비자폐(신경전형) 성인 사이의 지각 차이를 보기 위해서는 민감하고 재현 가능한 행동 과제가 필요하지만, 기존 얼굴 감정 인식 연구는 결과가 연구마다 일치하지 않았다. 이 불일치는 자극 전체에 걸쳐 차이가 고르게 나타나는 것이 아니라, 특정 소수 표정에서만 두드러질 수 있다는 관점에서 설명될 수 있다.

- **Core Contribution**: 논문은 감정 판단에서의 집단 간 차이가 ‘이미지 단위 희소성(image-level sparsity)’에 의해 나타날 수 있다고 제안하며, 차이를 보일 가능성이 큰 진단용 표정을 찾는 모델 기반 프레임워크를 제시한다. 또한 인공지능이 예측한 집단 분리도를 기준으로 새로운 얼굴 자극을 선택하고, 생성 모델로 지각 차이를 강화·약화시키는 방향의 변환까지 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 참가자별 감정 판단을 이미지 단위로 예측할 수 있을 만큼 신뢰도 있게 모델링하고, (2) 그 예측을 바탕으로 ‘집단 분리’를 최대화하는 새로운 자극을 효율적으로 탐색하는 것이다. 저자들은 집단별로 population-specific 인공신경망을 학습해 이미지별 판단을 예측하고, 그 결과로 모델이 고른 얼굴이 실제 행동에서 차이를 더 크게 만들도록 검증했으며, 생성적 적대 신경망(GAN)으로 진단 이미지를 집단 합의가 커지는 방향으로 변환한 뒤 phenotype-matched 방식으로 감소 효과를 확인했다.

- **Empirical Impact**: 독립 코호트에서 모델이 선택한 이미지들은 무작위로 매칭한 이미지보다 더 큰 행동 차이를 유발했으며, 이는 ‘모델 가이드 자극 설계’가 실험 민감도를 실제로 높인다는 근거를 제공한다. 더 나아가 동일 모델을 이용한 합성 이미지는 원본과 매칭했을 때 행동상의 집단 분리를 줄여, 자극 조건에 따라 지각 차이가 수렴/발산할 수 있음을 실증적으로 보여준다. 결과적으로 행동 표현형(behavioral phenotyping)이 고정된 자극 평균에서 벗어나, 어떤 조건에서 신경발달적 지각이 달라지는지(또는 같아지는지)를 찾아내는 최적화된 진단 설계로 확장될 수 있음을 시사한다.



### Drift-Aware Temporal Graph Rewiring (DATGR) for Adaptive Semantic Modeling in Biomedical Tex (https://arxiv.org/abs/2607.08490)
Comments:
          6 pages, 4 figures. Published in the Proceedings of the 2026 IEEE Conference on Artificial Intelligence (CAI 2026)

- **Prior Approaches**: 기존 연구는 시계열을 나눈 뒤 정적 임베딩이나 동시출현 기반 co-occurrence graph에 의존해, 시간에 따른 의미 변화(semantic drift)를 충분히 반영하지 못했다. 그 결과 검색(retrieval)이나 지식 발견(knowledge discovery)에서 시간이 지나며 성능이 저하되는 문제가 반복됐다. 또한 매 타임슬라이스마다 임베딩을 재학습하는 방식은 비용이 커서 적용성이 제한된다.

- **Core Contribution**: 이 논문은 Drift-Aware Temporal Graph Rewiring(DATGR)로, 개념의 진화에 맞춰 co-occurrence edges를 의미 드리프트 추정치로 동적으로 재배선하는 프레임워크를 제안한다. 매번 임베딩을 재학습하지 않고, 로지스틱 업데이트 규칙을 edge weight에 적용하는 lightweight한 피드백 기반 rewiring을 수행한다. 이를 통해 시간에 따른 의미 변화가 그래프 링크 예측에 직접 반영되도록 설계했다.

- **Technical Challenges**: 핵심 기술 난제는 시간에 따라 의미가 달라질 때, 어떤 edge를 얼마나 갱신할지 신뢰성 있게 추정하는 것이다. DATGR은 semantic drift를 추정한 뒤 로지스틱 업데이트로 edge 가중치를 조정해 과도한 변화 없이 필요한 재배선을 유도한다. 또한 계산 효율을 위해 전체 재학습 대신 edge 수준의 점진적 갱신만 수행하도록 구성했다.

- **Empirical Impact**: BIOMRC(Biomedical Multi-Relation Corpus)에서 DATGR은 static baseline 대비 mean AUROC를 약 0.066p(0.699 vs. 0.633) 개선했다. AUPRC는 0.738 vs. 0.744로 비슷해 정밀도 손실 없이 recall 향상에 기여했음을 보였다. 즉, 시간에 따른 생의학 텍스트의 의미 변화를 edge-level 적응으로 포착하면서도 효율적이고 해석 가능한 성능 개선을 달성한 것으로 평가된다.



### Applying JEPA-Style Predictive Learning to JA4-Derived Network Fingerprints (https://arxiv.org/abs/2607.08465)
- **Prior Approaches**: 기존 네트워크 지문(fingerprint) 학습은 대개 원본 입력을 재생하거나(생성형) 혹은 단순 분류 목적에 최적화하는 방식이 많아, 다양한 관점(view)과 불완전한 관측에서 표현이 흔들릴 수 있다는 한계가 있었다. 반면 JEPA 계열은 이미지·비디오에서처럼 입력을 그대로 복원하지 않고 잠재 표현의 예측을 타깃 인코더 출력과 정합해 학습한다. 이 논문은 이런 JEPA식 목표가 컴팩트 네트워크 지문에도 그대로 통하는지의 가능성에 초점을 둔다.

- **Core Contribution**: 저자들은 JA4, JA4H, JA4S, JA4X 서브필드로부터 표현을 학습하는 Transformer 기반 모델 JA4-JEPA를 제안한다. JA4DB와 CIC-IDS-2017에서 각각 데이터를 모아 약 397K 샘플로 학습하되, 단일 샘플에 네 가지 뷰 패밀리가 모두 들어있지는 않은 상황(불완전한 관측)을 그대로 반영했다. 그 위에 학습된 임베딩이 TLS, DNS, SSH의 프로토콜-패밀리 분류에 유용한지 kNN 프로브로 평가했다.

- **Technical Challenges**: 핵심 과제는 JEPA의 잠재 예측 정합 목적이 네트워크 지문처럼 ‘원본 입력이 복합 메타데이터 형태’인 경우에도 잘 작동하는지 검증하는 것이다. 특히 소스 간/샘플 간 뷰 중첩이 완전하지 않으면, 타깃 인코더와의 정합이 약해져 표현학습이 불안정해질 수 있다. 저자들은 JA4 서브필드를 구성요소로 삼아 Transformer로 end-to-end 예측 정합을 학습하고, frozen kNN 프로브로 표현의 품질을 직접 측정해 이 문제를 실증적으로 다뤘다.

- **Empirical Impact**: 평가는 39,416개 홀드아웃 샘플에서 cosine similarity 0.9899, kNN accuracy 0.9220을 기록하며 목표 함수가 JA4 기반 지문에서도 유의미한 임베딩을 만든다는 점을 보여줬다. 또한 TLS·DNS·SSH 전반에서 프로토콜-패밀리 분류 성능을 통해, 소스별/샘플별 뷰 누락이 있어도 표현이 전이 가능하다는 신호를 제공한다. 결과적으로 JEPA-style predictive representation learning이 네트워크 fingerprinting 영역으로 확장될 수 있음을 시사한다.



### OmniFood-Bench: Evaluating VLMs for Nutrient Reasoning and Personalized Health Advic (https://arxiv.org/abs/2607.08423)
- **Prior Approaches**: 기존 연구는 주로 음식 카테고리 인식 같은 거친 분류 과제에 집중해 왔고, 실제 식단 관리에 필요한 복잡한 추론 흐름을 평가하지 못했다. 특히 겉모습으로 숨겨진 재료를 추정한 뒤 물리적 질량을 추정하고, 마지막으로 질병 특이 안전 조언으로 이어지는 연결성을 검증하기 어려웠다.

- **Core Contribution**: 본 논문은 MM-Food-100K 기반으로 OmniFood-Bench를 제안해 VLM을 3단계 능력(기본 지각, 정량 추론, 안전-치명도 조언)으로 점진 평가한다. 이를 통해 음식 시스템의 Systemic Information Asymmetry, 즉 시각적 외형과 내재 영양 구성 간 불일치를 체계적으로 드러내고 신뢰도 표준을 제시한다.

- **Technical Challenges**: 핵심 난제는 겉보기 단서에서 숨겨진 재료를 추정한 뒤, 그로부터 실제 물리량(섭취량/질량)을 정량화하고, 위험 질환 프로필에 대해 안전한 처방을 생성하는 end-to-end 추론을 평가하는 것이다. 저자들은 벤치마크를 단계별 과제로 구성해 지각→정량→조언의 실패 지점을 분리 측정하고, 여러 SOTA VLM(gpt-5.1, gemini-3-flash, qwen3-vl-8B 등)에서 같은 문제가 반복되는지를 관찰할 수 있게 했다.

- **Empirical Impact**: 실험 결과 모델들은 요리/음식 명명에서는 사람 수준에 가깝게 맞추지만, 질량 추정에서는 치명적으로 붕괴하며 당뇨 고위험 프로필에서 안전하지 않은 ‘양성’ 조언을 자주 환각했다. 이는 의미(semantic)는 잘 맞아도 물리(physical) 추정과 안전성 보장이 따라오지 않는 Semantic-Physical Gap을 보여 주며, 공중보건용 자율 에이전트의 신뢰성 검증 기준을 높이는 데 의미가 크다.



### Game Theory Driven Multi-Agent Framework Mitigates Language Model Hallucination (https://arxiv.org/abs/2607.08403)
- **Prior Approaches**: 규칙 기반 과학 도메인에서 경량 Large Language Models를 쓰면, 언어 패턴을 그럴듯하게 흉내 내는 경향이 커서 공리적(axiomatic) 추론을 재현하지 못하고 환각이 잦다는 한계가 있었다. 기존 접근은 도메인 제약을 충분히 내재화시키기보다 데이터/프롬프트 수준의 정렬에 의존하는 경우가 많아 성능이 불안정했다.

- **Core Contribution**: 이 논문은 Bayesian 원리와 팀 게임 관점의 원리를 결합한 적응형 multi-agent 프레임워크 G-Frame을 제안하고, 고품질 데이터 합성과 모델 학습을 위한 자동 closed-loop를 구축했다고 밝혔다. 또한 도메인 제약을 구조화된 reasoning을 통해 내부화하도록 설계해, 환각을 줄이면서도 규칙 기반 과학 추론에 더 적합한 학습 데이터를 만들었다.

- **Technical Challenges**: 핵심 기술 과제는 다중 에이전트가 생산한 결과가 언어적 유창성에 치우치지 않도록, 도메인 제약을 체계적으로 강제하는 closed-loop를 구현하는 것이다. G-Frame은 Bayesian/팀 게임 원리에 기반한 적응형 협업으로 reasoning 흐름을 제어하고, 그 결과로 363,045개의 chains-of-thought와 199,589개의 question-answer 쌍으로 구성된 전문 코퍼스를 합성했다.

- **Empirical Impact**: 그 결과 7B 모델 OmniChem은 커스텀 벤치마크와 ChemBench에서 GPT 4o mini와 성능이 유사한 수준(패리티)을 보였고, 기본 아키텍처 대비 hallucinations를 79.46% 줄였다. 추가로 분자 설계와 합성 계획에서 고급 역량을 시연하며, 특수 과학 분야 지식 탐구를 가속할 수 있는 확장 가능한 패러다임을 제시했다.



### Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning (https://arxiv.org/abs/2607.08393)
- **Prior Approaches**: LLM 지식 업데이트는 RAG나 knowledge editing, fine-tuning 같은 방식으로 진행돼 왔지만, 공통적으로 ‘기억은 되는데 추론에서 못 쓰는’ 문제가 보고돼 왔다. 기존 연구들은 주로 성능 차이를 관찰하거나 특정 편집을 수행하는 데 집중했지만, 왜 memorization이 downstream reasoning에 연결되지 않는지의 내부 메커니즘을 정밀하게 추적하는 도구가 부족했다. 또한 grokking 같은 학습 동학 관찰은 있었으나, 본 연구의 초점은 새로운 능력의 출현이 아니라 단일 지식이 논리적 체계에 일반화되는 과정의 실패 원인 규명이다.

- **Core Contribution**: 논문은 fine-tuning 후 발생하는 Knowing–Using Gap(정확도 격차+시간 지연)을 정식화하고, memorization은 빨리 포착되지만 generalization은 늦게 또는 낮게 나타난다는 현상을 체계적으로 규량화했다. 이를 설명하기 위해 self-patching이라는 개입 기반 진단을 제안해, ‘어떤 레이어·토큰 위치에 저장된 표현이 추론에 인과적으로 유효하게 라우팅될 수 있는지’를 공간적으로 지도화한다. 나아가 knowledge–circuit misalignment 가설(저장은 되지만 계산 회로의 유효 위치로 라우팅되지 못함)을 인과 실험으로 지지하고, 이를 활용한 고정 휴리스틱으로 oracle headroom의 58–75%를 복구하는 실용성도 함께 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 “저장된 지식이 내부 어디에 이미 존재하는가”와 “그 표현이 실패한 generalization에서 실제 추론 계산에 어떻게 라우팅되지 않는가”를 동시에 분리해 측정하는 것이다. self-patching은 소스 실행에서 anchor의 은닉표현을 특정 레이어로 캐시한 뒤, 타깃 실행의 후보 레이어에 재주입하여 정답 지표의 인과적 변화(ΔI)를 계층쌍(ls, lt) 전수 스캔으로 측정한다. 이 과정에서 실패 케이스에도 적용 가능하도록 clean 정답 경로 의존을 줄이고, 프롬프트/컨텍스트를 교차하여 위치-정착 패턴이 잡음이나 프롬프트 편법이 아님을 점검한다.

- **Empirical Impact**: 실험은 도메인(생의학·학술)과 아키텍처/스케일을 가로질러 수행했으며, 대부분의 설정에서 memorization은 빠르게 포화하지만 chaining 및 일반화 사용은 낮거나 지연되는 Knowing–Using Gap이 일관되게 관찰됐다. self-patching 결과는 지식 표현이 존재하더라도 mid-layer의 추론 회로로 라우팅되지 못하는 ‘공간적 불정합’을 보여주고, 이를 표현을 옮겨(재배치) 주면 downstream 성능이 즉각적으로 상승함을 인과적으로 확인했다. 실용적 측면에서는 고정된 두 개 레이어-쌍 휴리스틱이 oracle headroom의 58–75%를 회복하며, prompting(CoT 등) 기반이나 일반 교란 대비 개선이 더 구조적임을 시사한다.



### FedOPAL: One-Shot Federated Learning via Analytic Visual Prompt Tuning (https://arxiv.org/abs/2607.08368)
Comments:
          Accepted by FLICS 2026

- **Prior Approaches**: 엣지 지능에서 기본 모델이 확산되면서 통신 대역폭이 연합학습의 확장성을 제한하는 핵심 병목이 됐다. 이를 완화하려고 one-shot 연합학습이 쓰이지만, 기존 반복적 fine-tuning이나 knowledge distillation은 서버 계산 비용이 크거나 하이퍼파라미터에 민감하다는 한계가 남아 있었다. 한편 analytical federated learning은 least-squares 폐형해로 gradient-free 집계를 제공해 효율적이지만, 비IID 데이터 환경에서는 정적 feature 가정이 깨지며 성능이 크게 떨어진다.

- **Core Contribution**: 이 논문은 FedOPAL 프레임워크를 제안해 analytical federated learning의 이론적 가정을 비IID 환경에서도 맞추는 데 초점을 둔다. 시각적 prompt를 feature rectifier로 동적으로 조정해 서로 다른 로컬 데이터의 feature 분포를 linearly separable 공간으로 “정렬(교정)”한다. 동시에 local proximal constraints를 적용해, 가정 붕괴로 인한 feature manifold misalignment을 줄여 analytical 집계의 전제가 유지되도록 만든다.

- **Technical Challenges**: 핵심 난제는 비IID 데이터에서 정적 feature 가정이 실패하면서 모델 성능이 붕괴되는 모순을 어떻게 이론 조건으로 되돌리느냐이다. FedOPAL은 visual prompt를 사용한 feature rectification으로 로컬 feature 분포를 선형 분리 가능 영역으로 끌어내리고, local proximal constraints로 클라이언트 업데이트가 과도하게 흔들리지 않게 제어함으로써 least-squares 기반 closed-form 집계의 가정에 맞춘다. 결과적으로 서버 쪽 학습/미분 기반 최적화 없이도 집계 효율을 유지한다.

- **Empirical Impact**: 실험에서 FedOPAL은 원래 analytical 방법 대비 여러 벤치마크에서 유의미하게 높은 성능을 보였다. 또한 accuracy가 최신 iterative 방법들과 견줄 수준에 도달하면서도, zero server-side training costs를 유지해 엣지 협업 관점의 공학적 이점을 입증했다. 이는 통신 병목을 줄이되 서버 연산 부담은 최소화하는 새로운 협력 패러다임으로 평가된다.



### MobiDiff: Semantic-Aware Multi-Channel Discrete Diffusion for Human Mobility Data Generation (https://arxiv.org/abs/2607.08357)
- **Prior Approaches**: 기존 이동(모빌리티) 데이터 합성 연구는 확산 기반이 유망하다는 점이 확인됐지만, 연속 스패시오템포럴 추적이나 잠재(latent) 궤적에 의존하는 경우가 많았다. 그 결과 지역·활동·시간·구간 같은 이산적 의미 이벤트를 구조적으로 다루는 데 한계가 있었다. 또한 연속 데이터로 바꾸기 위한 보간(interpolation)과 잠재 궤적 구성, 거친-정교화(coarse-to-fine) 실현 같은 파이프라인이 비용과 복잡도를 키웠다.

- **Core Contribution**: 이 논문은 MobiDiff를 제안하며, 체크인 이벤트를 ‘공간·활동·시간’ 채널의 다중 시맨틱 스켈레톤으로 분해한 뒤 이산 디노이징으로 합성하는 end-to-end discrete diffusion 프레임워크를 제시한다. 기존처럼 비싼 보간과 잠재 궤적 구성, 복잡한 coarse-to-fine 파이프라인을 거치지 않고 직접 생성한다. 아울러 이벤트·그룹·채널 수준의 마스킹을 함께 사용해 이벤트 간(trajectory-level) 패턴과 이벤트 내부 의존성을 동시에 학습한다.

- **Technical Challenges**: 핵심 기술적 난제는 이산 의미 구조를 유지한 채 확산의 잡음 제거 과정을 안정적으로 설계하는 것이다. 이를 위해 MobiDiff는 각 체크인 이벤트를 공간/활동/시간 채널로 명확히 표현하고, structured event-, group-, channel-level masking으로 마스킹 범위를 계층적으로 제어해 within-event 의존성과 trajectory 수준 규칙을 함께 포착한다. 그 덕분에 연속 궤적 생성에서 요구되던 보간·잠재 추적 구성 없이도 의미 이벤트의 구조성을 유지할 수 있다.

- **Empirical Impact**: MobiDiff는 Atlanta, Boston, Seattle의 3개 대규모 실데이터에서 생성 충실도, 프라이버시 보존, 효율성을 평가했으며 전반적으로 경쟁력 있는 성능을 보였다. 특히 궤적 길이와 시간 간격(interval) 분포를 효과적으로 보존하면서도 더 넓은 이동 통계 전반에서 성능을 유지했다. 또한 추론(inference) 단계에서 GeoGen 대비 평균 5.3× 더 빠르며, 해석 가능성과 효율성을 갖춘 이산 확산 기반 합성 프레임워크의 가능성을 보여준다.



### Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models (https://arxiv.org/abs/2607.08317)
Comments:
          25 pages, 8 figures

- **Prior Approaches**: 기존 AI 벤치마크는 여러 과제에서 높은 성능을 보여주지만, 사람이 거의 당연하게 수행하는 조작·그리기 같은 작업에서 지속적인 맹점이 드러나지 못할 수 있다는 지적이 있었다. 특히 기존 평가가 이러한 “겉보기엔 쉬워 보이지만 모델에겐 어려운” 유형을 충분히 포착하지 못한다는 한계가 제기된다.

- **Core Contribution**: 이 논문은 human에게는 단순해 보이지만 현대 AI에겐 어려운 과제를 통해 맹점을 드러내는 벤치마크 blind-spots-bench를 제안한다. AI 강의 학생들로부터 원문 질문을 수집해 정제·구조화된 정답(참조 해설)으로 주석을 달고, 235개 샘플에 맞춘 태스크 택사노미를 구성했으며 자동 채점 파이프라인도 함께 개발한다.

- **Technical Challenges**: 핵심 기술적 난관은 “단순해 보이는 과제”를 실제로는 무엇이 실패 요인인지 분해 가능한 형태로 정답을 구조화하고, 다양한 모델 유형(오픈웨이트/클로즈드소스, vision-language, image-generation)을 동일한 기준으로 평가하는 자동화 채점 설계를 요구한다. 논문은 학생 질문 기반으로 데이터셋을 만들고 참조 솔루션을 구조화한 뒤, 태스크별 평가를 아우르는 자동 그레이딩 파이프라인을 구축해 모델 간 공정 비교가 가능하도록 했다.

- **Empirical Impact**: blind-spots-bench 분석에서 클로즈드소스 frontier 모델이 오픈웨이트 모델보다 약 10% 내외 격차로 크게 우수한 성능을 보였고, 기존 벤치마크에서 비슷한 수준을 달성해도 맹점 영역에서는 격차가 유지되는 양상이 나타났다. 또한 태스크 유형 전반에서 단일 모델이 전부를 지배하지 못하며, 일부 과제는 모든 평가 모델에게도 여전히 어려운 것으로 드러나 blind-spots-bench가 구체적 약점을 찾아내는 진단용 “스트레스 테스트”로 유용함을 시사한다.



### INTENT: An LSTM Framework for Vehicle Intention Prediction in Intersection Scenarios with Comprehensive Ablation Analysis (https://arxiv.org/abs/2607.08316)
- **Prior Approaches**: 자율주행에서 차량 의도 예측은 교차로·회전로·비상상황처럼 인간의 상호작용과 복잡한 주행 패턴이 많은 구간에서 안전과 민첩성을 좌우하는 핵심 요소로 다뤄져 왔다. 기존 접근은 주로 신호나 궤적 기반 추정에 의존하거나, 의도 정보를 궤적 예측에 조건으로 반영하는 방식으로 확장돼 왔지만, 실시간 의사결정에 바로 쓰기엔 예측 타이밍과 정확도 측면에서 제약이 남아 있었다.

- **Core Contribution**: 이 논문은 INTENT 프레임워크를 제안하며, LSTM 모델로 이벤트 발생 2초 전에 차량의 의도를 선제적으로 예측한다. 교차로 상황에서 직진/좌회전/우회전의 3가지 의도를 분류해, 회피 기동 등 후속 의사결정에 필요한 인간 수준의 의도 해석을 지향한다. 아울러 의도 조건부 trajectory prediction(의도-조건 궤적 예측) 관점에서 의도 예측이 궤적 예측 성능에도 기여할 수 있음을 함께 강조한다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 짧은 리드타임(2초)에 의도를 정확히 읽어내는 것과 (2) 교차로에서의 행동 변화가 빠르게 나타날 때도 안정적으로 분류하는 것이다. 연구진은 LSTM을 통해 시간적 문맥을 학습하고, 이벤트 직전의 관측에서 직진·좌회전·우회전 신호를 분리해 학습 안정성과 예측 신뢰도를 확보하는 방향으로 설계했다. 또한 다양한 실험과 ablation study(요인 제거 실험)로 구성요소의 기여를 검증했다.

- **Empirical Impact**: InD 데이터셋에서의 실험 결과, INTENT은 99.71% 정확도를 달성하며 의도 예측 성능을 실증적으로 입증했다. 2초 전 예측이라는 시간 제약을 만족하면서도 높은 분류 정확도를 보여, 실시간 안전 기동과 의사결정 파이프라인에 직접 활용될 가능성을 시사한다. 특히 복잡한 교차로 시나리오에서 의도 기반 후속 예측(예: 의도-조건 궤적 예측)으로 이어질 수 있다는 점에서 후속 연구에도 영향을 줄 것으로 보인다.



### Psychological Competence as a Missing Dimension in AI Evaluation (https://arxiv.org/abs/2607.08285)
Comments:
          22 pages, 3 figures

- **Prior Approaches**: 기존 AI 평가 프레임워크는 정확도, 견고성, 추론 능력, 정책 준수처럼 모델 성능에 초점을 둡니다. 하지만 사람과 직접 대화하는 조언가·코치·튜터·동반자 역할의 시스템에서는 응답이 사용자의 사고 방식, 감정 해석, 신념 형성, 신뢰 보정, 의사결정에 영향을 줍니다. 그래서 평가 단위를 ‘모델’뿐 아니라 ‘인간- AI 상호작용’으로 넓힐 필요가 있습니다. 기존 접근은 일부 상호작용 요소를 다루긴 하지만, 심리적 효과를 직접적으로 측정하는 경우는 드뭅니다.

- **Core Contribution**: 논문은 AI 평가의 누락된 차원으로 psychological competence(심리적 역량)를 제안합니다. 이는 인간을 대상으로 하는 AI가 사용자 인지, 정서 해석, 행동 의사결정에 대해 사용자·상황·목적에 맞게 지원할 수 있는 능력으로 정의됩니다. 또한 framing, tone, perceived authority, responsiveness, uncertainty handling, conversational guidance 같은 상호작용 속성을 핵심 구성요소로 포함시키며, 특정 벤치마크 제안이 아니라 개념과 경계를 정리합니다.

- **Technical Challenges**: 심리적 역량을 측정하려면 상호작용의 미묘한 심리적 효과를 실험적으로 포착해야 한다는 기술적 난제가 있습니다. 논문은 이를 위해 scenario-based probes(시나리오 기반 점검), structured human evaluation(구조화된 인간 평가), model-assisted evaluation(모델 보조 평가) 같은 방법 조합을 통해 평가 가능성을 제시합니다. 즉, 단순 정확도 지표가 아니라 ‘대화가 사용자 판단과 감정 해석에 미친 결과’를 관찰하도록 설계하는 접근이 필요하다고 강조합니다.

- **Empirical Impact**: 구체적 벤치마크를 만들기보다는 construct를 명확히 하고 평가 방향을 제안함으로써, 향후 연구와 실무에서 심리적 효과를 평가 프레임에 포함시키는 기준점을 제공합니다. 모델 제공자·배포 조직·연구자·규제기관이 실제 현장 영향(사용자 신뢰, 판단, 정서 해석 등)을 고려해야 한다는 점에서 의미가 큽니다. 인간을 향한 AI의 ‘기술 성능’ 외에 ‘심리적 안전성과 적합성’을 핵심 고려사항으로 확장하는 논의에 기여할 것으로 기대됩니다.



### Understanding Axes of Difficulty For Long Context Tasks Via PredicateLongBench (https://arxiv.org/abs/2607.08284)
- **Prior Approaches**: 기존 장문( long-context ) 평가는 Needle-in-a-Haystack(NIAH) 같은 방식부터 다중 홉 추론, 요약 태스크까지 다양하지만, 대체로 평균 성능만 측정해 강건성이 부족하거나 난이도 확장에 대한 체계가 약하다는 한계가 있었다. 또한 일부 벤치마크는 포화(saturated) 상태에 빠지거나, 입력 길이가 늘어도 ‘무엇이’ 더 어려워지는지 축(axes) 단위로 분석하기 어렵다는 지적이 있었다.

- **Core Contribution**: 이 논문은 주어진 조건(예: 사전식 lexicographic ordering)을 만족하는 가장 긴 연속 부분수열을 찾아내는 PredicateLongBench를 제안해, 장문 추론을 난이도 축별로 스트레스 테스트할 수 있게 한다. 핵심은 서로 다른 난이도 축을 체계적으로 설계해 장문 이해의 여러 측면을 동시에 시험한다는 점이며, 태스크 자체는 개념적으로 단순해 LLM 기반 생성이나 판정(judge)에 의존하지 않는다.

- **Technical Challenges**: 다만 이러한 벤치마크를 만들려면 난이도를 정교하게 ‘확장’하면서도 정답을 안정적으로 정의할 수 있어야 한다. 논문은 두 개의 상보적 생성 파이프라인을 제공하는데, 하나는 무작위 단어-유사 문자열로 완전 합성(synthetic)하는 설정이고, 다른 하나는 실제 문서에서 단어를 샘플하되 분포적 성질(distributional properties)을 보존하는 real-world 설정이다.

- **Empirical Impact**: 실험 결과, frontier 모델들은 PredicateLongBench에서 정의한 난이도 축을 따라 난이도를 키울수록 성능이 급격히 저하되는 경향을 보였다. 이는 기존 장문 벤치마크의 평균 중심 평가지표가 포착하지 못한 한계를 드러내며, 향후 long-context 능력의 한계를 진단하고 개선 방향을 잡는 데 유용한 평가 도구임을 시사한다.



### PolyUQuest: Verifiable Structure-Aware Web RAG over Heterogeneous Graphs (https://arxiv.org/abs/2607.08269)
- **Prior Approaches**: 기존 retrieval-augmented generation(RAG)은 웹 페이지를 평면 텍스트로 취급해 HTML의 구조·의미 신호를 놓치는 한계가 있었다. 이로 인해 근거 추적이 어렵고, 웹 간 연결(하이퍼링크)이나 DOM 계층, 개체 관계 같은 단서가 검색 품질과 신뢰성에 제대로 반영되지 못한다. 또한 모드가 단일해 쿼리의 구조적 필요에 맞춘 검색/추론이 어렵다는 지적이 있다.

- **Core Contribution**: PolyUQuest는 HTML 구조와 웹 간 연결을 함께 쓰는 구조-aware web RAG 프레임워크를 제안한다. 페이지 간 하이퍼링크 토폴로지, 페이지 내부 DOM 계층, 페이지 간 엔터티-관계 지식을 하나의 heterogeneous graph로 통합해 검색 경로를 설계한다. 답변은 인용된 블록마다 원천 페이지, heading path, 엔터티 링크를 제공해 모든 주장을 구조적 근거로 “verifiable”하게 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 쿼리 유형에 따라 필요한 구조 신호가 다를 때 적절한 retrieval 모드를 선택하는 것, (2) 그래프를 기반으로 다중 단계 증거를 찾되 계산·토큰 비용을 통제하는 것이다. PolyUQuest는 two-tier router로 직접 블록 retrieval, cross-page graph traversal, multi-hop entity reasoning 중 하나를 쿼리에 맞춰 라우팅하며, 인용 블록의 위치·계층·개체 링크를 함께 유지해 검증 가능성을 확보한다. 그 결과 LLM 토큰 소모를 줄이면서도 faithfulness를 강화하는 방향으로 설계됐다.

- **Empirical Impact**: 평가는 홍콩 폴리테크닉 대학(PolyU) 공식 웹사이트 4,240페이지(31,086 DOM blocks, 29,119 entities, 37,680 relations)와 멀티 타입 벤치마크로 수행됐다. PolyUQuest는 기존 RAG 대비 answer correctness, coverage, faithfulness에서 우수한 성능을 보였고, 쿼리당 LLM 토큰 사용량도 유의하게 감소했다. 또한 인용 답변의 증거를 구조 경로로 탐색·비교하는 대화형 인터페이스를 제공하며, PolyU 학생 대상 QA 서비스로의 배포 준비 단계에 있다.



### Different Teachers, Different Capabilities: Sub-1B On-Device Distillation for Structured Text Enrichmen (https://arxiv.org/abs/2607.08268)
Comments:
          12 pages, 5 figures. has a same-size non-reasoning-teacher control, a three-judge LLM-as-a-judge panel with a negative control, full-source faithfulness grading, and a per-field routing analysis

- **Prior Approaches**: 구조화 추출은 스키마에 맞춰 항목당 JSON을 생성하는 패턴이라, 큰 모델을 그대로 쓰면 지연·비용이 누적된다. 그래서 Hinton 계열 증류와 black-box 증류, 그리고 structured extraction 전용 증류들이 제안돼 왔지만, 대부분은 전체 품질 한 가지로만 델타를 보여 실제 운영에서 어디가 성능 병목인지 가리기 쉽다.

- **Core Contribution**: 이 논문은 뉴스 기사→요약+5개 라벨(JSON)로 변환하는 작업을 대상으로, 8B reasoning teacher를 0.6B on-device student로 증류했을 때 “소모한 비용 대비 회수한 성능”을 서브태스크(요약 체크리스트, 라벨, faithfulness)별로 분해해 측정한다. 또한 동일 크기 비-reasoning teacher, 더 큰 managed pipeline(합성데이터 확장 포함)까지 비교해 증류 이득의 원인이 추론 성격인지 데이터/스케일인지 분리해 보여준다.

- **Technical Challenges**: 핵심 과제는 소형 모델이 교사 출력의 형식·분류를 따라가면서도, 사용자에게 치명적인 faithfulness(원문 근거 없는 창작)를 유지할 수 있느냐이다. 저자들은 온디바이스 Q4_K_M에서 모두 temperature=0 고정으로, 93개 홀드아웃 기사에 대해 reference-free 3인 패널(블라인드, 구성요소 체크리스트 기반)을 사용해 항목별 재채점 안정성과 편향을 통제했고, 증류 성능을 “기준선→교사 간 간극 대비 회수율”로 보고 축별 실패를 드러냈다.

- **Empirical Impact**: 0.6B 학생은 기사당 약 0.8초로 동작하면서 교사(약 39초) 대비 요약 품질 간극을 58% 회수했고, 기준선(제약 디코딩) 대비 +16.8, few-shot 대비 +4.9로 유의미하게 개선됐다. 다만 faithfulness는 기사 길이가 짧고 얇은 출처(≤1200자)에서 일관되게 흔들리며, 이 서브그룹에서는 학생이 원문 근거 없이 더 그럴듯하게 꾸미는 양상이 관찰돼 “필드별 라우팅 맵(어떤 경우 큰 모델로 보낼지)”을 제안한다.



### MentalHospital: A Virtual Environment for Evaluating Psychiatric Clinical Encounters (https://arxiv.org/abs/2607.08257)
- **Prior Approaches**: 기존에는 LLM이 대화·진단·치료계획 같은 단일 정신건강 과업에서 성능을 보였지만, 실제 임상에서 반복되는 “완전한 임상 면담 흐름”을 재현한 벤치마크는 드물었다. 또한 평가는 주로 정답/근거 일치 같은 객관 지표에 치우쳐, 면담 과정의 질(커뮤니케이션, 기록, 임상적 엄밀성)을 함께 점검하기 어려웠다.

- **Core Contribution**: 이 논문은 LLM 기반 정신의학 임상 “환자 진료 시나리오”를 가상으로 평가하는 환경 MentalHospital를 제안한다. Subjective Interviewing, Objective Examination, Diagnostic Assessment, Treatment Planning의 S.O.A.P. 워크플로를 구현하고, 1,193건의 비식별 EHR을 바탕으로 ICD-11 전반 범주와 76개 장애를 커버하는 skill-augmented standardized patient로 만능화했다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 임상 과정의 주관적 질을 객관적 비교와 함께 평가하고, (2) 전문의 판단을 확장 가능한 평가기로 스케일링하는 것이다. 이를 위해 MentalHospital는 EHR-derived reference와의 객관 비교(더블 트랙)와 함께 임상 과정 품질을 동시 채점하며, 5개 도메인(공감 커뮤니케이션, 면담 전문성, 임상노트 품질, 진단 엄밀성, 치료 적절성) 평가기 MentalEval을 rubric-grounded SFT와 expert-guided DPO로 학습해 전문 영역 합의를 끌어올렸다.

- **Empirical Impact**: 22명의 임상의 설문은 MentalHospital의 임상적 충실도를 평균 3.88/5로 지지했고, MentalEval은 전문가 정렬에서 평균 QWK 0.944로 강한 일치를 보였다. 벤치마킹 결과, 최고 성능 LLM도 객관적 정신의학 역량에서 임상가 대비 37.28%p 뒤처졌으며, 특히 mental status assessment가 주요 병목으로 드러났다.



### Compete Then Collaborate: Frontier AI Teachers Build a Verifiable Curriculum to Improve a Coding Student Beyond Imitation (https://arxiv.org/abs/2607.08255)
Comments:
          8 pages, 1 figure

- **Prior Approaches**: 기존의 멀티-티처 지식 증류는 여러 frontier 모델의 출력을 단순 결합해 학습 데이터를 만들거나, 어떤 모델이 더 좋은지 제대로 가리지 못하는 경우가 많다. 또한 LLM judge로 평가할 때는 심사자가 자기 출력에 유리한 편향을 보일 수 있다는 한계가 있었다. 그 결과 “누가 가르치는가”보다 “무엇을 섞는가”에 치우친 데이터 생성이 되기 쉽다.

- **Core Contribution**: 논문은 compete-then-collaborate 프레임워크를 제안해, 먼저 실행 기반 평가로 티처들을 랭킹(대결)한 뒤 그 결과를 바탕으로 학생용 커리큘럼을 공동으로 구성한다. 검증 가능한 커리큘럼으로 student(Qwen2.5-Coder)가 실제로 풀어보며 학습하도록 설계해, 답을 모아 그대로 따라 하는 접근의 한계를 겨냥한다. 또한 공개 가능한 온프레미스 파이프라인과 GRPO 실행을 위한 패치를 제공한다.

- **Technical Challenges**: 핵심 난제는 편향 없는 티처 비교와, “검증 가능한 보상/데이터”를 안정적으로 만들 수 있는 환경 설계였다. 논문은 unit tests 및 stdin-stdout 체크 같은 실행 검증으로 심사를 구성하고 공정성 제어를 두어 execution-based judge의 신뢰도를 높였다. 그다음 협업 커리큘럼을 RLVR처럼 verifiable rewards 환경에 연결해, 학생이 doing 중심으로 학습하도록 RL 단계(예: GRPO 계열)를 적용했다.

- **Empirical Impact**: 실험 결과, 실행 검증이 적용된 표준 문제에서는 대체로 티처 간 성능 차가 작아 99~100% 수준의 포화(saturation)가 나타났다. 반면 경쟁형의 더 어려운 문제에서는 Gemini(77%)가 앞서고 Claude·Codex-GPT가 동률(각 69%), Grok는 50%로 뒤처졌다. 그런데도 학생 쪽 최종 성능은 티처 랭킹에 크게 좌우되지 않았고, SFT는 오히려 7B/32B에서 MBPP-test(76.7%→72.7%) 및 경쟁 과제 성능(예: 5.9%→2.9%)처럼 저하를 보였다. 반대로 같은 협업 커리큘럼을 RLVR의 verifiable reward로 학습시키면 경쟁 과제 최고 성능이 5.9%에서 8.8%로 상승(+49% 상대 개선)해, AI-teacher 협업의 가치가 “정답 pooling”이 아니라 “검증 가능한 학습 환경 공동 구축”에 있음을 실증했다.



### AutoPersonas: A Multi-Timescale Loop Engine for Open-Ended Persona Evolution (https://arxiv.org/abs/2607.08252)
Comments:
          52 pages, 13 figures/tables, ancillary public-safe evaluation artifacts included

- **Prior Approaches**: 기존 Generative Agents 계열은 샌드박스 사회 안에서 장소·루틴·역할 등 환경을 설계하고, 메모리·성찰·계획으로 그럴듯한 일상을 굴리는 방식에 강점이 있다. Agentopia처럼 장기 시뮬레이션도 존재하지만 보상(instrumented)이나 닫힌 세계 가정에 기대는 경향이 있어, 보상 없는 open-ended persona evolution에서 생기는 런타임 고착(self-locking) 문제를 정면으로 다루기 어렵다. 장기 메모리 연구는 저장·검색·요약·개인화를 돕지만, ‘어떤 증거가 현재 State의 권위를 가져야 하는가’까지 구조적으로 분리해주지 못한다.

- **Core Contribution**: 이 논문은 장기 페르소나 에이전트가 적응하면서도 정체성을 유지해야 하는데, 그 과정에서 발생하는 런타임 실패 모드로 self-locking을 정의한다. 핵심은 현재 State, 메모리, 히스토리, 환경 요약이 반복 호출마다 ‘권위’를 되감아, 겉으론 그럴듯한 사건이 계속 나오지만 관계·결정·생활 단계·환경이 좁은 끌개(attractor)로 수렴해 버린다는 점이다. 이를 해결하기 위해 AutoPersonas라는 다중 timescale life-environment 엔진을 제안하며, 환경 측 Occurrences·누적 Observations·페르소나 State를 분리하고 evidence governed absorption으로만 State/reachability를 갱신하도록 OSO 루프를 설계한다.

- **Technical Challenges**: AutoPersonas의 관건은 두 힘의 균형이다: 미래를 열어주는 divergence(분기)를 반복 생성해야 하지만, 그 생성물이 증거로 흡수되기 전에는 현재 State의 맥락 중력에 빨려 들어가 ‘고정된 생활 궤도’만 재생산하지 않게 해야 한다. 저자들은 프롬프트 예산 제약(과거·현재 컨텍스트가 토큰의 약 60%를 넘으면 divergence가 악화됨)과, 모델 수준 다양성 붕괴/시스템 수준 컨텍스트 중력이 결합되는 원인을 함께 짚고, OSO의 증거-통제 전파로 이를 완화하려 한다. 또한 동일 실행 내에서도 강제 재구성 대신 per-sample divergence targeting과 context-slice masking 같은 운영 기법으로 매크로 테마 반복을 줄이되 정체성 연속성은 유지하도록 조정한다.

- **Empirical Impact**: 3년 압축 시뮬레이션과 8개 모델·40일 스트레스 테스트에서 self-locking 관련 징후(환경 watermark shell, occurrence hardening gap, 느린 변화 누적 실패, 재귀적 indecision, 약한 관계 지속)가 체계적으로 관찰됐다. 특히 1,600개의 이벤트 생성 결과, 5일 행동-카테고리 반복이 평균 95.2%~97.6%로 높게 나타났고 모델 대부분이 11일차에 90% 이상을 넘었다. 같은-runtime 40일 A/B에서 context-slice masking + divergence targeting은 매크로 테마 반복을 61.8%→36.3%로 낮추며 누적 테마 수를 약 2배 수준으로 늘렸고, 현실 세계 침입 없이도 juvenile-goblin 픽션 월드에서 anti-fixation 체제를 재현해 bounded claim(분기와 증거 흡수를 분리하면 self-locking을 줄일 수 있음)를 지지한다.



### Playing ZendoWorld: Challenging AI Agents on Active Visual Concept Induction (https://arxiv.org/abs/2607.08233)
- **Prior Approaches**: 기존 비주얼 귀납 및 규칙 학습 벤치마크들은 관찰 예시만으로 추론을 요구하는 경우가 많아, 에이전트가 새로운 장면을 제안해 가설을 갱신하는 ‘능동 실험’의 반복성을 잘 다루지 못했다. 또한 강화학습 기반 비주얼 에이전트는 관측→행동 최적화는 강하지만, 명시적 규칙/가설 공간을 회복하는 과정을 평가하기 어렵다. LLM이 실험을 제안하는 연구도 있었지만, 입력이 대개 깨끗한 기호 추상에 머물러 시각적 정합과 유도(인덕션)를 함께 분리해 보기 어려웠다.

- **Core Contribution**: 이 논문은 시각 관측을 바탕으로 숨은 논리 규칙을 추론하고, 규칙을 검증할 새 장면을 생성하며, 환경 피드백으로 가설을 갱신하는 폐루프 평가 환경 ZendoWorld를 제안한다. ZendoWorld는 Prolog 기반 DSL로 규칙 공간을 명확히 정의하고, 장면은 절차적으로 생성해 공정한 비교가 가능하도록 했다. 그 결과 ‘라벨 예측 정확도’와 ‘규칙 복원’ 사이의 간극 같은 핵심 실패 모드를 통합적으로 드러내도록 설계됐다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 고차원 이미지에서 객체 속성과 공간 관계를 상징 상태로 매핑하는 지각, (2) 관측된 예시로부터 올바른 논리 규칙을 유도하고 수정하는 인덕션, (3) 불확실성을 실제로 줄이는 실험 장면을 고르는 실험 설계가 동시에 작동해야 한다는 점이다. 논문은 시각-귀납-실험을 같은 상호작용 루프에 묶되, 시각 입력을 DSL/상징 상태로 분리해 병목을 진단할 수 있게 했다. 또한 렌더링-규칙 판정을 엄밀히 하기 위해 DSL 프로그램 정규화와 반례(counterexample) 제공, 그리고 예상 정보 이득(EIG)로 실험의 효율을 계량했다.

- **Empirical Impact**: 실험 결과, 에이전트가 관찰 예시에 대한 라벨은 비교적 잘 맞춰도 내부 규칙을 올바르게 복원하지 못하는 경우가 다수였고, 지각과 인덕션 병목이 에이전트 유형별로 다르게 나타났다. 특히 VLM 기반 에이전트는 EIG가 낮아 ‘거의 정보가 없는 실험’을 제안해 가설 불확실성을 줄이지 못했으며, 규칙 복잡도가 커질수록 성공률이 급격히 하락했다. 휴먼 데이터도 수집했는데, 사람은 더 안정적인 규칙 궤적을 보이며 복잡하거나 OOD 규칙 복원에서 VLM 기반 시각 에이전트가 따라가지 못해, 과학적 발견 같은 도메인에서 필요한 구조적 개선 방향을 구체화했다.



### A First-Principles Theory of Slow Thinking and Active Perception (https://arxiv.org/abs/2607.08196)
Comments:
          Published on 2026/05/11 in Journal of Machine Learning

- **Prior Approaches**: 기존 연구는 인지 기능을 확률모형이나 신경망으로 근사하더라도, 사고(thinking)와 지각(perception)을 하나의 수학적 절차로 연결해 설명하는 데는 한계가 있었다. 특히 slow thinking(느린 사고)과 active perception(능동적 지각)을 설계·학습·추론 관점에서 일관되게 정식화한 접근은 드물었다.

- **Core Contribution**: 이 논문은 관측 공간과 잠재 공간의 확률분포를 lifting과 projection으로 다루는 틀 위에서, thinking과 perception을 하나의 수학적 구성으로 전개한다. ‘active lifting’ 이론을 제안해 잠재 시퀀스를 샘플링하고 불확실성을 최대 속도로 줄이려는 내적 동기를 통해, slow thinking LLM을 포함하는 큰 설계 공간과 이를 뒷받침하는 static theory의 부분공간을 도출한다.

- **Technical Challenges**: 핵심 난제는 (1) 복잡한 분포를 신경망 같은 단순 함수족으로 어떻게 일관되게 표현할지, (2) 지각의 능동성(불확실성 감소)과 내적 시간 축을 가진 추론을 동시에 어떻게 만들지, (3) 학습 목표를 정보이론적 관점과 연결해 안정적으로 최적화할지이다. 논문은 잠재 시퀀스 샘플링과 ‘최대 불확실성 감소율’ 원리를 결합하고, 두 계층(표현 계층·샘플러 계층) 위를 올라가는 업그레이드 규칙과 최소 길이 코딩과 유사한 학습 목표를 통해 이를 해결한다.

- **Empirical Impact**: 이론적 결과로서 slow thinking 모델 개선의 3단계 경로, 다중 모달리티 공통 인코더/생성모형 구성, 사람과 유사한 시각 표현의 사전 형성, policy collapse의 가능한 완화책 등 여러 파생물이 제시된다. 다만 제공된 초록만으로는 구체 벤치마크 수치와 성능 향상 정도는 확인되지 않으며, 저자들은 이러한 구조가 slow thinking 포맷의 출현과 행위성(agency)을 특성화한다고 주장한다.



### ASMR: Agentic Schema Generation for Ship Maintenance Report Writing (https://arxiv.org/abs/2607.08177)
Comments:
          Accepted at the DASHSys 2026 workshop (Systems for Data-centric Agents with Human-in-the-loop), co-located with VLDB 2026

- **Prior Approaches**: 기존에는 보고서 작성자가 미리 정의한 양식이나 수작업 템플릿에 의존해 스키마를 구성하는 경우가 많았습니다. 자동 스키마 생성은 주로 단일 수준의 단어·구문 패턴에 기대는 방식이 많아, 보고서 유형별 핵심 정보 요구를 압축하고 중복을 줄이는 데 한계가 있었습니다.

- **Core Contribution**: 이 논문은 자동 스키마 생성 문제를 다루며, 다양한 범주의 과거 선박 정비·운용 보고서로부터 보고서 유형별로 “간결하면서도 정보량이 높은” 스키마를 발견하는 것을 목표로 합니다. 제안하는 ASMR은 Field Generation Agent와 Structural Optimizer Agent의 모듈형 agentic 프레임워크로, 필드 후보 도출과 스키마 구조 최적화를 분리해 성능을 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 (1) 서술형 문장에서 의미 개념을 안정적으로 추출해 다양한 수준의 필드 후보를 만들고, (2) 그중에서 RL로 “컴팩트·정보성·비중복”을 만족하는 구조를 찾아내는 것입니다. 저자들은 Field Generation에서 적응형 multi-granularity clustering으로 후보 필드를 생성하고, Structural Optimizer에서 reinforcement learning으로 스키마 표현을 반복적으로 개선하도록 설계했습니다.

- **Empirical Impact**: 초기 실험 결과는 제안한 ASMR이 보고서 유형별 요구를 반영하는 간결한 스키마를 생성할 가능성을 보여줍니다. 또한 생성된 스키마는 작성자에게 완전성·일관성·실행 가능성을 높이는 가이드로 활용될 수 있으며, data management, agentic AI, human-centered AI의 교차 지점에서 추가 연구 주제를 제시합니다.



### Overthinking: Amplifying Reasoning Weights to Extract Learned Secrets (https://arxiv.org/abs/2607.08173)
Comments:
          Accepted at ICML 2026. 9 pages, 6 figures

- **Prior Approaches**: 언어 모델의 black box auditing은 배포 전 정렬 문제를 점검하는 핵심 도구지만, 표면적인 오류나 노골적 정보 노출에 비해 미묘한 misalignment, 숨은 정보까지는 놓칠 수 있습니다. 기존 감사는 입력 프롬프트 설계나 일반적인 perturbation에 의존하는 경우가 많아, 모델이 ‘생각을 드러내는’ 경향을 체계적으로 키우기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 감사 과정에서 숨은 정보를 더 잘 끌어내기 위해 overthinking을 제안합니다. reasoning task vectors를 이용해 비-추론 instruct 모델 M과 reasoning-distilled 모델 R의 파라미터 차이를 가중치 스케일(α>1)로 증폭하는 overthinking model을 정의하고, 추가로 품질과 coherence를 유지하면서 추론을 선택적으로 키우는 layer-wise attenuation 전략을 함께 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘추론을 증폭’했을 때 단순한 출력 품질 저하(일관성 붕괴, 횡설수설) 없이 실제로 숨은 정보 탐지를 강화하는 것입니다. 논문은 R 방향으로의 reasoning amplification을 하되, 레이어별 감쇠를 통해 과도한 편향이 발생하는 구간을 제어하는 방식으로 해결합니다.

- **Empirical Impact**: 실험 결과 overthinking 모델은 2B~32B 스케일의 모델에서 4가지 setting 전반에 걸쳐 숨은 정보 노출 빈도를 더 높였습니다. 특히 일부 secret/의도치 않은 행동은 기존 reasoning 모델 대비 최대 10배까지 더 자주 드러났으며, secret 유형에 따라 reasoning 방향의 섬세한 perturbation이 필요한 경우와 큰 weight perturbation이면 충분한 경우가 있음을 분석해 감사 전략 설계에 실질적 단서를 제공합니다.



### Answer Set Programming Energised! End-to-End Neurosymbolic Reasoning and Learning with ASP and Energy Based Models (https://arxiv.org/abs/2607.08136)
Comments:
          Preprint

- **Prior Approaches**: 기존 neurosymbolic 추론·학습은 신경망과 규칙 기반 추론을 느슨하게 결합하는 경우가 많아, 배경지식·제약·비단조 추론 같은 논리적 의미를 연속 잠재공간 최적화에 충분히 녹이기 어렵다는 한계가 있었다. 또한 answer set programming(ASP)과 확률/에너지 기반 모델의 접점을 다루더라도, end-to-end 학습을 위한 범용 구조나 동적 도메인(지각·상호작용 포함) 적용성이 부족했다.

- **Core Contribution**: 이 논문은 answer set programming(ASP)을 에너지 기반 모델(energy based model)과 모듈 방식으로 결합해, 명시적 ASP 기반 선언적 의미를 통해 연속 잠재공간에서의 joint optimisation을 지원하는 일반적인 neurosymbolic 추론·학습 방법론을 제안한다. 더 나아가 answer sets–probabilistic logic–answer set modulo theories의 인터페이스를 포괄하는 일반화된 모델과, ASP 중심의 robust end-to-end 학습을 위한 실용 플랫폼을 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 ASP의 제약과 비단조 추론, 배경지식의 논리적 의미를 연속(latent) 공간 최적화로 연결하면서도 학습 가능하고 일관된 gradient 신호를 확보하는 데 있었다. 저자들은 ASP 기반 선언적 의미를 명시적으로 통합하고, 동적 도메인에서도 강건하게 학습될 수 있도록 모델/플랫폼을 구성해 continuous latent space에서의 공동 최적화를 가능하게 한다.

- **Empirical Impact**: 구현과 사용 예시로 MNIST에서의 기본 동작을 보이고, visual question-answering 벤치마크 Clevr와 multi-object tracking 벤치마크 MOT에서 평가를 수행해 방법의 실용성을 입증한다. 특히 지각과 상호작용이 얽히는 동적 응용에서 ASP 의미 기반 추론을 end-to-end로 학습할 수 있다는 점에서, neurosymbolic 분야의 연구·응용 연결성을 높인다는 의미가 있다.



### CausalDS: Benchmarking Causal Reasoning in Data-Science Agents (https://arxiv.org/abs/2607.08093)
Comments:
          55 pages, 10 figures

- **Prior Approaches**: 기존 평가는 크게 두 갈래로 나뉜다. 한쪽은 코딩·도구 사용을 강조하지만 숨겨진 인과 데이터생성 구조가 부족하고, 다른 쪽은 그래프/개입/반사실 추론을 보지만 실제 데이터 분석 워크플로와 결합이 약한 경향이 있다. 또한 인과 평가 데이터셋이 기존 소스의 선별 예시 중심이거나 템플릿 변형 위주라, 체계적으로 새로운 합성 인과 구조를 생성해 다양성을 확보하기 어렵다.

- **Core Contribution**: CausalDS는 에이전트형 데이터사이언스 워크플로에서의 인과 추론을 평가하는 벤치마크로, 각 인스턴스를 “장면(scene)” 단위로 구성한다. 장면은 샘플링된 Structural Causal Model(SCM)과 그로부터 생성한 관측 데이터, 그리고 현실 도메인에 grounded된 합성 자연어 스토리를 함께 제공하며, 선택적으로 실세계 분포를 조합해 경험적 구조를 유지한다. 과제는 Pearl의 Rung 1~3(연관-개입-반사실)에 걸쳐 도출되고, 상당수는 불완전 관측과 observation model 때문에 여러 도구를 거쳐야 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 인과 구조는 통제하면서도 (2) 실제 데이터 분석 난이도는 바꾸고 (3) 모델이 답을 해야 할 상황과 못 해야 할 상황(비식별)을 구분하도록 평가를 설계하는 것이다. CausalDS는 개념적 SCM과 공개 관측값을 분리해, 개념 변수를 잡음 측정 번들로 치환하더라도 인과 추정대상/식별성 라벨은 그대로 유지하면서 수치 추정 난이도만 조절한다. 또한 비식별 질문에 대한 abstention을 ‘정답 행동’으로 명시해, 모델이 원할 때 무리한 추정을 하지 않도록 결정적 채점(ground truth 기반)을 구성한다.

- **Empirical Impact**: 저자들은 953개 장면(관측 모델 변형 포함)으로 구성된 CausalDS 시험에서, 모델 성능이 다섯 축(상징적 인과 추론, 데이터사이언스 실행, 불확실성, abstention, tool-use/coding)으로 분해되어 한 역량만으로는 설명되지 않음을 보인다. 즉, 인과 추론 정확도만으로는 불확실성 보정이나 비식별 상황에서의 올바른 거절 능력을 대변하지 못한다는 실증 결과를 제시한다. 이는 에이전트형 LLM 평가가 인과성·통계 추정·코딩·불확실성·거절을 함께 봐야 함을 시사하며, “causal parrot” 리스크를 줄이는 합성 구조 기반 평가의 방향성을 강화한다.



### PARA-PV: Physics-Aware Retrieval-Augmented PV Prediction Based on Frozen Foundation Model and Distribution Shift Correction (https://arxiv.org/abs/2607.08079)
- **Prior Approaches**: PV 발전량 예측은 날씨 변동, 일주기 전환, 상태(레짐) 의존 동역학, 그리고 물리 제약 때문에 어렵지만 기존 연구는 대체로 단일 데이터 패턴 학습이나 통계적/블랙박스 예측에 치우치는 경우가 많다. 또한 과거 유사 구간을 쓰더라도 물리적으로 말이 되는 조건(운전 상태, intra-day period, 전력 레벨, temporal shape)을 일관되게 보장하기가 어렵다. 레짐별 오차가 불균형해 주요 ‘regular’ 구간이 학습을 지배하면 피크·램핑 같은 운영상 중요한 상태를 놓칠 위험도 크다.

- **Core Contribution**: 이 논문은 PARA-PV로, 예측 전 과정에 물리 지식을 내장한 Physics-Aware Retrieval-Augmented 프레임워크를 제안한다. 현재 윈도우 조건과 일치하는 과거 패치 및 유사 궤적을 physics-aware retrieval로 찾아 물리적으로 근거 있는 base forecast를 만들고, 이를 Chronos time-series foundation-model의 prior에 잔차(residual) 보정으로 연결한다. 이어서 day/night와 레짐 변화에 의해 생기는 분포 이동을 power·weather·timestamp 조건으로 조정하며, 레짐(peak/ramping/night/regular)별 학습 균형을 physics-constrained loss로 보장한다.

- **Technical Challenges**: 핵심 난제는 (1) 현재 날씨·일주기·운전 상태가 맞는 ‘물리적으로 정합한’ 과거 조각을 retrieval로 찾는 것, (2) foundation-model의 일반 시간 규칙을 PV 물리 기반 예측 위에 과도하게 덮어쓰지 않으면서 적응하는 것, (3) 레짐 전환 시 발생하는 residual conditional distribution shift를 선택적으로 교정하는 것이다. 저자들은 patch-level 표현과 temporal shape/power/operating state/intra-day period까지 동시 조건화한 physics-aware retrieval-augmented learner로 base forecast의 물리적 타당성을 확보한다. 또한 frozen Chronos prior에는 lightweight residual adapter만 붙여 일반 시간 패턴을 PV 특화로 미세조정하고, gated mean-shift/scale 형태의 분포 이동 보정 모듈로 power·weather·timestamp·day/night에 따라 보정 강도를 선택적으로 적용한다.

- **Empirical Impact**: PARA-PV는 물리 제약 기반의 base forecast, Chronos 잔차 캘리브레이션, 레짐 전환 분포 이동 보정, 그리고 peak/ramping/night에 대한 적응적 재가중 학습이 함께 작동함을 실험으로 입증한다. 특히 regular 구간이 우세해 발생하는 오차 학습 편향을 줄이면서 운영상 중요한 상태(피크·램핑·야간)의 예측 성능을 강화하는 것이 의미 있는 결과로 제시된다. 전력 계통 운영과 재생에너지 통합에 필요한 신뢰도 향상 방향에서, retrieval·foundation-model·물리 제약을 end-to-end로 묶는 설계가 후속 연구에 실용적 기준을 제공할 것으로 보인다.



### Persuasion Attacks Can Decrease Effectiveness of CoT Monitoring (https://arxiv.org/abs/2607.08066)
Comments:
          25 pages, 10 figures

- **Prior Approaches**: CoT monitoring은 AI 에이전트의 안전을 위해 추론 흔적(Reasoning traces)을 관찰해 오정렬·기만을 탐지한다. 기존 연구는 일반 상황에서 효과가 있으나, 설득 기반 jailbreak처럼 자연어 논증이 제약을 우회하면 취약함이 드러났다고 지적돼 왔다.

- **Core Contribution**: 이 논문은 ‘모니터 LLM’에 대해서도 적대적 에이전트가 CoT 근거를 통해 정책 위반 행동을 승인하도록 설득할 수 있는지 직접 검증한다. 40개 태스크의 에이전트-모니터 상호작용을 통해, CoT 모니터링이 오히려 해로운 행동 승인에 역효과가 날 수 있음을 보여준다.

- **Technical Challenges**: 핵심 문제는 모니터가 보이는 CoT를 근거로 판단할 때, 에이전트의 논증이 모니터를 설득하는 ‘추가 설득 채널’이 생긴다는 점이다. 이를 완화하기 위해 사실검증(fact-checking) 모니터링을 제안하며, 서로 다른 모델 패밀리의 모니터와 fact-checker를 조합(예: Claude 3.7 Sonnet monitor + GPT-4.1 fact-checker)해 승인 편향을 줄였다.

- **Empirical Impact**: 실험 결과, 적대적 설정에서 에이전트의 CoT에 모니터가 접근할수록 평균적으로 해로운 행동 승인율이 9.5%p 증가했다. 반면 모델을 분리한 fact-checker-모니터 조합은 정책 위반 행동 승인을 최대 45%까지 낮췄는데, 같은 모델을 함께 쓰면 약 6% 수준에 그쳤다. 이는 CoT monitoring만으로는 적대적 설득에 충분치 않을 수 있으며, model-diverse fact-checking이 실질적 방어책이 될 수 있음을 시사한다.



### When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals (https://arxiv.org/abs/2607.08065)
- **Prior Approaches**: 기업용 평가에서 LLM-as-judge는 사실상 기본값으로 자리 잡았고, 최근에는 여러 judge를 앙상블하거나 mixture-of-experts 패널처럼 묶어 일관성(consistency) 기반으로 정답성을 추정해왔다. 이 접근은 judge들(또는 한 모델의 self-samples) 간 합의가 곧 정확도라는 가정을 공유한다. 하지만 합의가 반드시 정답을 뜻하지는 않는다는 문제의식이 제기돼 왔다.

- **Core Contribution**: 이 논문은 “합의=정확도” 가정이 왜 취약한지(자기합의, 모델 간 공유 편향, 암기된 휴리스틱, option-position prior 등) 체계적으로 지적한다. 동시에 합의가 언제, 어떤 조건에서 “부분적으로” 유용한 대리변수(proxy)가 될 수 있는지 대규모 교차 러너(cross-runner) 실험으로 묻는다. 결론적으로 self-consistency는 독립적 신뢰도 점수라기보다 조건부 proxy임을 실증한다.

- **Technical Challenges**: 핵심 기술적 난제는 서로 다른 실행/러너 간 분포 차이를 통제하면서 합의-정확도 상관을 공정하게 측정하는 것이다. 저자들은 53개 runner가 겹치는 K=50 케이스를 서로 샘플링하도록 설계해 총 265,000개 샘플을 모았고, hierarchical runner-clustered bootstrap으로 resampling 기반의 견고한 평가를 수행했다. 또한 majority-correctness를 배치 레이블로 삼아 합의 지표의 예측력을 정량화했다.

- **Empirical Impact**: 결과에서 합의는 양(+)의 예측력이 있되 약한 편이며, 상관계수 rho는 0.20~0.59 범위로 관측됐다(항목 단위 클러스터링 resampling에선 모두 양으로 유지). 유용성은 상황에 따라 크게 달라져, unsaturated mid-tier 모델 및 계산 배분에는 비교적 잘 맞지만 가장 일관된 frontier 모델에선 “과신하지만 더 정확하지 않은” 오류가 두드러졌다(합의≥0.8인 77% GPQA 항목 중 48%가 오답). Claude 계열 3개 tier에서도 동일한 frontier over-confidence 패턴이 관찰되며, de-identified per-run row와 answer distribution을 공개해 재현/후속 분석을 돕는다.



### A safety-oriented hypothetico-deductive framework for AI-assisted differential diagnosis (https://arxiv.org/abs/2607.08038)
- **Prior Approaches**: 기존 LLM 기반 진단 보조는 진단을 대체로 one-shot 분류처럼 “최종 답” 중심으로 다루는 경우가 많아, 놓치기 쉬운 위험 대안(must-not-miss)을 체계적으로 보장하거나 근거를 엄밀히 검증하는 단계가 약하다는 지적이 제기돼 왔다. 또한 평가도 Top-kk 정확도 위주로 끝나는 경우가 많아, 실제 임상에서 중요한 ‘안전 누락 방지’와 ‘근거 추적 가능성’, ‘다음 행동으로의 전환’이 충분히 측정되지 못했다.

- **Core Contribution**: AegisDx는 가설-연역적(hypothetico-deductive) 임상 추론을 안전 중심 프레임워크로 재구성해 차별화한다. 역할(role)별 계약을 가진 전문 LLM 컴포넌트를 조율하고, 구조화된 중간 산출물과 evidence-retrieval 인터페이스, verification gate를 통해 (1) 폭넓은 감별진단, (2) must-not-miss 조건의 명시적 스크리닝, (3) 의학 근거 기반 검증, (4) 실행 가능한 다음 단계 계획을 한 묶음으로 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 안전에 치명적인 대안을 “아예” 놓치지 않게 하면서도, 각 진단 후보에 대한 근거를 연결하고 추론을 검증 가능하게 만드는 오케스트레이션을 만드는 것이다. 논문은 역할 계약과 구조화된 출력 스키마, 근거 검색-검증-게이트 과정을 통해 감별진단의 폭과 정렬을 유지하고, must-not-miss를 별도 채널로 강제하며, 근거가 뒷받침하는지 확인한 뒤 관리 계획으로 전환하도록 설계했다.

- **Empirical Impact**: 문헌 기반 NEJM/JAMA 벤치마크에서 AegisDx는 GPT-oss-120B 단독 대비 Top-3 진단 정확도를 더 높였고, 특히 Annals of Emergency Medicine에서는 must-not-miss 세트에 대해 top-3 내 포함 확률이 78.0%로 개선(52.0% 대비)됐다. 또한 Yale New Haven Health System의 실제 응급실 문서 43건에 대한 블라인드 의사 평가에서 안전 관련 복합 점수가 4.31에서 4.55로 상승(보정 p=2.1×10^-4)했으며, must-not-miss 식별과 추론 안전성에서 질적 이득이 관찰됐다. 다만 근거-연결 추론이 길어져 가독성이 떨어지는 문제가 드러났고, 후속으로 ‘간결성 중심’ 출력 최적화 필요성을 제시했다.



### From Prompts to Contracts: Harness Engineering for Auditable Enterprise LLM Agents (https://arxiv.org/abs/2607.08028)
Comments:
          32 pages, 6 figures, 16 tables. Reference implementation and evaluation artifacts: this https URL (archived at this https URL)

- **Prior Approaches**: 기존 사내 LLM 앱은 프롬프트와 검색(RAG) 컨텍스트에 핵심 거동이 의존하는 “프롬프트-지배형” 프로토타입으로 출발하는 경우가 많다. 이 방식은 시연에는 유리하지만, 소스 경계·엔터티 라우팅·답변 계약·재현 가능한 트레이스를 프롬프트만으로 강제하기 어렵다는 한계가 있다. 또한 RAG는 근거를 제공하지만, 생성된 문장이 실제 증거를 충족하는지(출처-주장 일치)와 런타임에서의 권한을 구조적으로 통제하는 문제는 별도 레이어가 필요하다고 지적한다.

- **Core Contribution**: 이 논문은 프롬프트-지배형 엔터프라이즈 LLM 프로토타입을 “추적 가능·감사 가능”한 LLM-agent 아키텍처로 재구성하는 harness-engineering 패턴을 제안한다. 핵심은 결정적 거동(소스 게이트, 엔터티 라우팅, claim 선택, 답변 계약, trace 생성)을 코드가 소유하도록 옮기고, 모델은 문장 조립(phrasing) 역할에만 두는 “replaceable composition boundary”를 두는 것이다. 런타임 답변에서 어떤 주장이 허용되는지는 source-backed claims와 manifests가 권위를 갖도록 분리해, 시연이 아닌 제품 수준의 계약 준수를 목표로 한다.

- **Technical Challenges**: 주요 기술 난제는 (1) 어떤 문장을 답변에 넣을 권한이 있는지(source-to-claim 권위), (2) 질문이 어느 기업/엔터티로 라우팅되는지, (3) 내부 trace와 내부 식별자가 사용자에게 새지 않도록 출력 위생을 강제하는 것, (4) 이 모든 것을 모델 교체에도 일관되게 유지하는 검증 설계였다. 논문은 manifests(소스 경계)→evidence records→promoted source-backed claims(런타임 입장 경계)로 이어지는 파이프라인과, reader-facing answer용 출력 계약 및 leakage/link/language 검증 게이트를 코드-owned control layer로 구현해 해결했다. 또한 composition boundary에서 live LLM을 끼우거나 빼는 방식을 동일 계약 검증에 연결하고, 라이브 조합 실패 시 deterministic composer로 폴백하며 실패 원인과 계약 상태를 trace에 기록하도록 했다.

- **Empirical Impact**: 해당 방법은 한국 기업 5개 그룹(상장사 25개) 공개 데이터 슬라이스와 총 113개의 source-backed runtime claims 위에서 평가됐으며, 고정된 검증 시나리오에서 소스 근거·엔터티 라우팅·trace 완결성·출력 위생·추천 언어 계약이 유지됨을 보였다. 모델 교체 실험에서도 hosted 모델 3종에 대해 composition-boundary 실행 270회에서 계약 위반이 모델-조립 측에 국한되고 harness가 탐지·기록했으며, prompt 지시만으로는 내부 trace 누출과 추천형 표현 위반이 완전히 재현됨을 보여 “코드-owned 보장”이 load-bearing임을 입증한다. 특히 외부 bolt-on guardrail은 위반을 막는 대신 유용성이 88/120으로 떨어진 반면, harness는 120/120의 완전한 유용성을 보존해 엔터프라이즈 제품화에서 감사 가능성과 실용성 동시 달성이 가능하다는 의미를 가진다.



### Concretized Proposition Prompting Resolves Composition-Knowledge Dichotomy in Large Language Models (https://arxiv.org/abs/2607.08018)
Comments:
          9

- **Prior Approaches**: 기존 chain-of-thought(CoT) 계열은 “단계별 사고”를 생성해 추론을 돕지만, 생성된 근거가 환각을 포함할 경우 그럴듯한 사후합리화로 오답을 낳을 수 있다. 또한 추론 구조를 강조하는 방법(least-to-most, plan-and-solve 등)은 지식 정확성이 필요한 의학 문제에서 약해지기 쉽고, 근거/지식을 강조하는 방법(analogical prompting, self-knowledge explicitation 등)은 논리 사슬이 필요한 수학 문제에서 흔들릴 수 있다.

- **Core Contribution**: 논문은 LLM 추론의 핵심 병목을 Composition-Knowledge Dichotomy(구성성-지식성 양극화)로 정의하고, 질문과 관련된 “명제(proposition)”를 명시적으로 구체화해 두 축을 동시에 만족시키는 Concretized Proposition Prompting(CPP)을 제안한다. CPP는 명제를 진리값(사실/오류)과 논리 모드(긍정/부정) 관점으로 범주화(TP/TN/FP/FN)한 뒤, 범주별로 구체화된 명제 집합을 answer 단계에 함께 제공한다.

- **Technical Challenges**: 구체화된 명제 생성이 환각을 포함하면 answer 단계가 이를 그대로 강화할 위험이 있어, 명제의 품질과 최종 정답을 함께 최적화하는 설계가 필요했다. 저자들은 DSPy 기반으로 proposition prompt와 answer prompt를 공동 최적화하고, LLM-as-a-Judge 방식의 judge model로 명제 품질 점수와 정답 정확도를 통합해(구성성·지식성 동시 가중) 잘못된 명제가 끝까지 전파되지 않도록 제어한다.

- **Empirical Impact**: CPP는 8개 QA 벤치마크(상식·수학·의학)에서 대체로 기존 prompting 대비 우수하거나 경쟁력을 보였고, 특히 의학(EHRNoteQA)에서 정밀 지식이 요구되는 상황에서 큰 개선을 보였다. 반면 수학(GSM-8K, MATH)에서는 연역 추론이 중요한 특성상 CoT와 비교해 경쟁적이거나 일부 과제에서 더 좋은 성능을 내며, 모델/파라미터 크기(7B~72B)에서도 대체로 성능이 확장되는 스케일링 경향을 확인해 “구조적이면서 사실 기반” 추론 패러다임으로 자리잡을 가능성을 제시한다.



### Agentic Neural Architecture Search (https://arxiv.org/abs/2607.07984)
- **Prior Approaches**: 기존 NAS는 weight-sharing, 진화/evolution 등으로 효율을 높였지만, 공통적으로 사람이 미리 설계한 검색 공간(search space)에 묶인다. 이 때문에 태스크마다 도메인 지식 기반의 공간 재구성이 필요하고, 공간 표현력 자체가 병목이 된다. LLM은 아키텍처 코드를 생성·변형해 open-ended 공간 탐색을 돕지만, 대부분은 LLM이 곧바로 서치 연산자/샘플러 역할을 하면서 “공간 확장”과 “탐색”이 섞여, 각 역할의 기여를 분리해 연구하기 어렵다.

- **Core Contribution**: AgentNAS는 LLM과 NAS의 역할을 명확히 분리한다. LLM이 먼저 고품질 seed architecture를 만든 뒤, 이를 named module slot을 갖는 slotted architecture로 분해해 NAS가 탐색할 “태스크 전용의 bounded search space”를 자동으로 정의한다. 이를 modular 3-phase 파이프로 구현해 LLM-설계와 NAS-조합 탐색의 기여를 독립적으로 측정 가능하게 했다.

- **Technical Challenges**: 핵심 난제는 LLM이 만든 open-ended 아이디어를 NAS가 다룰 수 있는 이산적·한정된 조합 공간으로 “안정적으로” 변환하는 것이다. AgentNAS는 seed의 macro 결정(깊이/폭/스테이지/백본)은 보존하되, 블록 단위로 모듈을 묶고(additive glue slot 포함) 모듈 교체/삽입으로 조합 자유도를 노출하도록 slotted 구조를 설계했다. 또한 모듈 교체로 파라미터 크기가 바뀌는 문제를 learning rate 가상 슬롯(여러 배수)으로 완화해, NAS가 슬롯 선택 시 학습 설정까지 현실적으로 맞출 수 있게 했다.

- **Empirical Impact**: 17개 태스크(NAS-Bench-360, Unseen NAS)에서 AgentNAS는 11개에서 SOTA를 달성하며, 11개 중 다수에서 task-specific expert 설계를 포함한 기존 베이스라인을 능가했다. Ablation 결과 LLM seed만으로도 다수 태스크에서 이미 기존 공개 성능을 따라잡고, NAS는 seed 이후에도 조합 재결합을 통해 추가 이득을 만드는 경우가 많아 두 메커니즘이 상호 보완적임이 확인됐다. 또한 LLM 샘플링은 빠르게 포화되지만 NAS는 슬롯 간 상호작용을 더 체계적으로 탐색해 성능 격차를 만들며, 이런 패턴은 서로 다른 능력 수준의 3개 LLM에서도 비교적 견고하게 관측됐다.



### Evaluating the Effect of Frame Rate in Sequence-Based Classification of Autism-Related Self-Stimulatory Hand Idiosyncrasies (https://arxiv.org/abs/2607.07957)
Comments:
          15 pages, 5 figures, 3 tables. Preliminary version presented as a poster at the AMIA 2024 Informatics Summit

- **Prior Approaches**: 기존 연구들은 ASD 선별을 다양한 데이터(설문·센서·질문지 등)로 확장했지만, 영상 기반으로는 CNN이 정적 프레임 또는 짧은 구간에 의존해 행동의 긴 시간 구조를 충분히 활용하지 못하는 한계가 컸습니다. SSBD 같은 자막 데이터셋은 규모가 작아(영상 75개) 성능이 62–76% 수준에서 머물렀고, 시계열 모델과 시간 샘플링·증강의 체계적 비교도 부족했습니다.

- **Core Contribution**: 본 논문은 비디오에서 self-stimulatory behavior(행동 스티밍)를 자동 분류할 때 (1) LSTM/GRU 같은 sequence 모델의 최적 아키텍처와 (2) temporal sampling rate 및 (3) 소규모 데이터에 맞는 augmentation 전략을 함께 정리했습니다. 특히 pose 기반 특징을 SSBD에 적용하고, sampling 간격 15프레임에서 LSTM(97.5%)·GRU(98.75%)가 CNN 대비 큰 폭으로 향상됨을 보여줍니다. 또한 I3D transfer learning 파이프라인에서 augmentation 10종의 기여도를 ablation으로 정량화해 upsampling이 핵심임을 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 작은 임상 비디오 데이터에서 temporal redundancy는 줄이되 diagnostically relevant한 시간 동학은 보존하는 sampling 설계와, 제한된 학습 샘플을 증강으로 얼마나 효과적으로 보완할지의 균형이었습니다. 저자들은 프레임을 1, 5, 15, 30, 45, 90 간격으로 샘플링해 시퀀스 길이 변화에 따른 성능을 비교했고, I3D 기반 전이학습 위에 수평/수직 플립·잡음·리샘플링(upsampling)·시간 변형 등을 조합한 뒤 leave-one-out ablation으로 “무엇이 성능을 깎는지”를 분석했습니다. 추가로 개인별(per-subject) 모델을 영상 내 temporally split 세그먼트로 학습·검증해, within-video 일관성의 가능성도 점검했습니다.

- **Empirical Impact**: 실험 결과, CNN 기반 기존 기준선(62–76%)을 LSTM/GRU가 일관되게 상회했으며 GRU는 최고 98.75% 정확도를 기록했습니다. augmentation 단독 비교에서는 horizontal flip이 48.78%로 가장 높았고, ablation에서는 upsampling 제외 시 training loss가 가장 크게 상승해(가장 큰 성능 저하) 임상용 소규모 비디오에서는 “샘플 수를 늘리는 증강”이 특히 중요함을 확인했습니다. 개인화 프로토콜은 per-video temporally split 설정에서 mean loss 1.84(SD 0.79)로 분산이 낮아, 짧은 관찰 구간 기반 캘리브레이션 같은 배치 스크리닝 방향에도 실증적 근거를 제공합니다.



### Persona Cartography: Charting Language Model Personality Traits in Weight Spac (https://arxiv.org/abs/2607.07916)
Comments:
          85 pages, 80 figures

- **Prior Approaches**: 기존 persona 제어는 프롬프트·activation-space 개입으로 추론 시 행동을 바꾸는 방식과, 사전/사후 학습으로 성격 기본값을 고정하는 방식으로 나뉘어 왔다. 전자는 문맥에 취약하거나 지속적인 개입이 필요하고, 후자는 비용이 크며 배치별로 유연하게 모드를 바꾸기 어렵다는 한계가 있었다. 또한 persona를 분해·측정·조절하는 도구가 일관되게 정립돼 있지 않았다.

- **Core Contribution**: 이 논문은 LLM persona를 OCEAN(개방성, 성실성, 외향성, 친화성, 신경성)의 ‘행동 특성 축’ 위 좌표로 보고, 이를 weight space에서 학습 가능한 제어 방향으로 다룬다. 저랭크 어댑터(LoRA)를 OCEAN 각 축을 증폭/억제하도록 학습해, 특정 persona 프로필을 스케일하고 조합할 수 있음을 제시한다. 나아가 사람이 정의한 OCEAN을 넘어, 비지도 psychometric 파이프라인으로 LLM 롤아웃에서 해석 가능한 잠재 요인도 복원한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 원하는 특성 축만 선택적으로 이동시키면서 (2) 스케일에 따른 연속적 제어와 (3) 어댑터 조합 시 예측 가능성을 확보하는 것이다. 논문은 constitution-guided distillation과 DPO 기반 학습으로 LoRA를 훈련하고, calibrated LLM-judge와 OCEAN 전용 TRAIT MCQ로 특성 표현을 측정했으며, MMLU·GSM8K·TruthfulQA로 능력 저하를 함께 검증했다. 결과적으로 각 어댑터는 목표 특성을 대체로 단조(monotonic)하게 이동시키고, 다른 어댑터와는 약 부가적으로 합성돼 혼합 persona 설계가 가능했다(다만 축 간 비직교성/상관은 존재).

- **Empirical Impact**: 6개 모델(총 4B~32B)에서 trait-control이 스케일·모델 패밀리·교사 선택에 비교적 견고하게 재현되었고, 중간 스케일에서는 일반 능력 성능도 크게 보존됐다. 안전 관련 downstream에서도 축 이동이 행동에 의미 있게 영향을 줘, 예를 들어 신경성 축 조절은 좌절을, 친화성 축 조절은 sycophancy를 변화시켰으며 성실성·친화성 조합은 jailbreak에 대한 위해 감소와 과거부(over-refusal) 간 균형을 개선했다. 또한 비지도 파이프라인은 tone·initiative·didacticism·epistemic caution 같은 모델 고유 요인을 안정적으로 찾아내, persona 측정-편집-안전 감사를 잇는 다리로서의 가능성을 보여줬다.



### Nigeria Machinery: A Low-Resource Industrial Dataset with a Domain-Grounded Reasoning Layer (https://arxiv.org/abs/2607.07883)
Comments:
          10pages, 2 tables

- **Prior Approaches**: 아프리카(특히 나이지리아) 산업 설비에 대한 공개·모델-준비 데이터가 부족해, 정량 분석이나 해당 환경의 수치에 기반한 언어모델 학습이 어려웠다. 기존에는 자체 수집 데이터나 일반 벤치마크에 의존하는 경향이 있었지만, 실제 도메인 숫자와 근거를 연결하기가 힘들었다. 또한 LLM으로 데이터셋을 만들 때 프롬프트가 실제 수치만 “맞춘 것처럼” 보이고 도메인 의미는 비게 되는 문제가 흔했다.

- **Core Contribution**: 논문은 나이지리아 제조 및 석유가스 부문의 설비 사용·고장 정보를 다룬 Nigeria Machinery Usage and Failures Dataset(89개 머신 레코드, 28개 지표, 2006~2025)을 공개한다. 각 레코드는 공개 출처를 명시하고 codebook으로 디코딩돼, 모델이 숫자와 근거를 함께 다루기 쉬운 구조를 제공한다. 더불어 희소한 수치값으로부터 chain-of-thought(CoT) 추론 예시를 구성하는 방법과, 그 결과 94개의 prompt/completion/reasoning-trace 행 및 행별 provenance 파일을 함께 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 희소한 수치에서 CoT 같은 추론-형식 데이터를 만들되, 프롬프트가 실제 도메인과 연결되도록 보장하는 것이다. 저자들은 LLM 데이터셋 생성 시 “숫자는 일치하지만 도메인에선 비어 있는” 프롬프트가 생기는 공통 결함을 지적하고, 도메인-기반 프레이밍을 수정해 도메인 접지율을 크게 끌어올렸다. 또한 retrieval 답변이 반드시 원천 값과 일치하도록 검증·정렬해, 출처 기반 일치율을 개선했다.

- **Empirical Impact**: 도메인-grounded 프롬프트 비율이 이전 릴리스의 78개 중 1개에서 이번엔 94/94로 상승했으며, 모든 retrieval 답변이 소스 값과 일치하도록(84/84) 개선됐다고 보고한다. 다만 레코드 89개와 단일 관측치 지표가 많아 대규모 학습용 데이터라기보다는 reference 및 seed 데이터셋 성격이 분명하다고 제한을 명시한다. 그럼에도 수치-근거-추론 레이어를 포함한 공개 자산과 방법론이, 아프리카 산업 설비 도메인에서의 실험 재현성과 모델 평가를 앞당기는 의미가 있다.



### Feedback Manipulation Regularization: Enabling Offline Agent Alignment for Imitation Learning (https://arxiv.org/abs/2607.07859)
- **Prior Approaches**: 강화학습 연구는 인간 가치에 부합하도록 에이전트를 정렬(alignment)하는 방향으로 관심이 이동하고 있다. 기존에는 인간 데모와 평가 피드백을 함께 쓰되, 언어 생성의 contextual bandit을 염두에 둔 다단계 파이프라인으로 결합하는 경우가 많아 단일 단계 오프라인 학습으로의 전환은 상대적으로 덜 연구됐다. 특히 완전한 순차 의사결정 환경에서 두 신호의 상호보완성을 더 촘촘한 학습 신호로 만드는 방법은 제한적이었다.

- **Core Contribution**: 이 논문은 Feedback Manipulation Regularization(FMR)이라는 알고리즘 비종속적 방법을 제안해, 평가 피드백을 교정(corrective) 신호로 활용함으로써 imitation learning 정책의 정렬을 개선한다. 핵심 아이디어는 피드백을 단순 추가 신호가 아니라 학습 목표를 정렬에 맞게 조정하는 정규화 관점으로 끌어오는 것이다.

- **Technical Challenges**: 도전 과제는 평가 피드백과 데모가 순차 의사결정에서 어떻게 더 연결된 학습 신호로 작동하게 할지, 그리고 단일 단계 오프라인 학습으로도 안정적으로 정렬 효과를 낼지였다. 저자들은 FMR을 통해 피드백을 학습 중 정규화 형태로 통합하고, Safety Gymnasium을 정렬 평가의 체계적 테스트베드로 구성해 다양한 imitation learning 알고리즘에 적용 가능함을 보였다. 또한 데이터가 적거나(데이터 제한) 맞춤 정렬 정보가 부족하고 잡음이 큰 데모에서도 성능이 유지되는 강건성을 함께 확인한다.

- **Empirical Impact**: 실험에서 FMR은 Safety Gymnasium 환경에서 전반적으로 aptitude를 개선했으며, misalignment을 최대 98%까지 줄이는 결과를 보고한다. 이 성과는 여러 imitation learning 알고리즘 전반에 걸쳐 관찰되어, 정렬 문제에서 “피드백-데모 결합을 단일 단계 오프라인으로” 가져올 수 있음을 실증적으로 보여준다. 데이터가 제한된 상황에서도 효과가 유지된다는 점에서, 실제 적용 관점의 효율성과 신뢰성을 함께 높인 것으로 평가된다.



### Agentic AI and Retrieval-Augmented Models in Straight-Through Underwriting (https://arxiv.org/abs/2607.07858)
- **Prior Approaches**: 기존 자동화는 if-then 같은 고정 규칙으로 워크플로를 실행하지만, 서술형 문서나 누락·불명확 정보가 섞이면 유연하게 대응하기 어렵다. LLM을 그대로 붙인 단일 파이프라인은 unstructured 입력을 잘 처리하지만 근거 문서 없이도 결론을 내리는 환각 위험과 추적성 부족 문제가 남는다. RAG는 검색-생성으로 사실 정합성과 traceability를 개선하지만, 검색 결과를 활용한 다단계 규칙 검증이나 누락 정보 보정까지는 충분히 구조화하지 못하는 한계가 있다.

- **Core Contribution**: 이 논문은 규정된 심사 흐름 안에서 straight-through underwriting(STU)을 수행하되, 투명성·감사가능성·human-in-the-loop 거버넌스를 강화하는 agentic AI 프레임워크를 제안한다. 특히 여러 단계 규칙 평가와 부족한 결정 근거의 보완(타사 데이터 점검, 반성적 검토)을 orchestration로 묶어, 무근거의 STU 결정을 줄이는 설계를 목표로 한다. 이를 위해 small commercial Business Owner Policies(BOP)의 합성 실험 환경에서 파이프라인 성능을 비교한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 비정형 신청서에서 필요한 사실을 정확히 구조화하고, (2) 검색 품질 변동에도 불구하고 다단계 규칙을 흔들림 없이 적용하며, (3) 결정에 필요한 정보가 없을 때는 모델이 추측 대신 보류·에스컬레이션하도록 통제하는 것이다. 저자들은 agentic RAG 파이프라인에서 targeted retrieval, 제3자 데이터 체크, 명시적 multi-step rule evaluation, reflection 단계를 결합해 구조적 점검과 근거 기반 생성을 구현했다. 그 결과, 단순한 생성이나 naive RAG보다 누락 정보·복합 조건 상황에서 더 신뢰도 높은 결정을 내리도록 파이프라인을 구성했다.

- **Empirical Impact**: 합성하지만 현실적인 벤치마크에서 세 파이프라인(단일 LLM baseline, naive RAG, multi-agent Agentic RAG)을 비교했으며, agentic 시스템이 전반적으로 가장 좋은 성능을 보였다. 특히 multi-step 및 missing-information 시나리오에서 개선 폭이 가장 컸는데, 이는 구조화된 검색과 반성 단계가 근거 없는 STU 결정을 피하는 데 기여했기 때문이다. 이 연구는 보험 심사처럼 규제·감사 요구가 큰 영역에서 agentic 오케스트레이션이 정확도뿐 아니라 설명·검증 가능성을 함께 끌어올릴 수 있음을 보여주는 proof-of-concept 성격의 의미가 있다.



### A Graph Neural Network Model for Real-Time Gesture Recognition Based on sEMG Signals (https://arxiv.org/abs/2607.07850)
- **Prior Approaches**: 기존 sEMG(전완 근전도) 기반 수지(손) 제스처 인식은 시간/주파수 특징을 뽑아 SVM, LDA, 신경망(NN) 등으로 분류하는 방식이 주류다. 실시간을 목표로 한 방법들은 주로 연속 신호를 빈(bin)으로 나눈 뒤 각 빈에서 특징을 추출해 분류하지만, 연속 빈 분할의 계산 부담이 지연(latency)을 키우는 문제가 있었다.

- **Core Contribution**: 논문은 전완의 sEMG 신호를 “상관관계로 구성된 그래프”로 표현하고, 이 그래프 시퀀스를 입력으로 하는 graph neural network(GNN) 기반 실시간 제스처 인식 알고리즘을 제안한다. 그래프 구성 시 전극을 노드로, 윈도우 내부 Pearson 상관을 엣지 가중치로 두어 근육 활성의 개별 패턴과 상호 관계를 함께 학습한다. 또한 제스처 온셋 탐지(event detection)와 분류를 파이프라인으로 묶어 의미 있는 구간만 분류에 활용한다.

- **Technical Challenges**: 핵심 난제는 (1) 윈도우 기반으로 상관 그래프를 만들 때 생기는 실시간 처리 지연, (2) 겹침(learning 데이터량) 감소에도 정확도를 유지해야 하는 일반화 문제, (3) 잡음이 섞인 sEMG에서 온셋 구간을 안정적으로 찾아내는 문제다. 논문은 윈도우 분할 후 각 윈도우에서 상관 행렬로 무방향 가중 그래프를 구성하고, Fröbenius norm 형태의 이벤트 지표로 임계값을 넘어선 구간만 선택해 학습·추론에 투입한다. 모델은 단순한 GNN 구조(완전연결층 2개)를 사용하되 그래프의 정보 전파로 채널 간 문맥을 학습하게 하여 계산량과 성능을 함께 잡는다.

- **Empirical Impact**: Myo band(전극 8개)로 8명(각 5개 제스처, 반복 수행) 데이터셋에서 평균 분류 정확도 99% 이상을 보고하며, 기존 실시간 제스처 방법 대비 성능 우위를 보였다. 그래프 겹침률 δ를 낮춰도 학습 데이터가 줄어드는 상황에서 AUC가 크게 무너지지 않는 등 견고함을 확인했다. 또한 M1 Pro CPU 기준 그래프 생성과 예측을 합친 end-to-end 평균 처리시간이 48ms로, 이벤트 간 간격보다 짧아 실시간 응용에 적합하다는 점을 실험적으로 뒷받침한다.



### VectorizationLLM: Smart Vectorization Based AI Assistan (https://arxiv.org/abs/2607.07846)
Comments:
          44 pages, 6 figures

- **Prior Approaches**: 기존에는 Zoom Teams Chat 같은 실시간 질의응답이나 강의 슬라이드에 예제를 추가하는 방식으로 학습 격차를 줄이려 했지만, 하위 1/4 성적 학생에게는 효과가 제한적이었습니다. 일반 LLM이나 코드 생성 도구는 개념 설명 정확도와 학습 무결성(직접 답/풀코드 생성)에서 쉽게 실패하고, MATLAB 교육 맥락에 맞춘 자료 기반 검색도 약합니다. 또한 범용 AI 튜터는 개념 회상은 도와도 고수준 MATLAB 코드와 강의 개념을 연결하는 범위가 제한되는 경우가 많았습니다.

- **Core Contribution**: VectorizationLLM은 CTEC 243/247의 MATLAB 학습을 돕기 위해, 강의 노트를 근거로 개념 설명과 ‘노트에 있는 코드 블록’을 함께 제공하는 전용 instructive assistant를 목표로 합니다. Google DeepMind Gemma 기반 open-weight 모델에 RAG 지식베이스와 system prompt 아키텍처를 결합해, 학생이 묻는 주제(모듈/단계)에 맞춰 학습 로드맵과 예시를 구성합니다. 동시에 숙제·시험 문제에서의 직접 해답 생성은 막고, 강의에서 금지한 패턴(예: loop/conditional, 특정 인덱싱 규칙 위반)을 강하게 유도하는 가드레일을 포함합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 첫 질의에서 필요한 강의 구절·코드·이미지를 정확히 RAG로 끌어오고 (2) 누락·혼합된 자료 환각 없이 여러 턴 대화에서도 맥락을 유지하며 (3) 학습 무결성을 깨지 않는 응답을 안정적으로 만드는 것입니다. 이를 위해 OpenWebUI의 RAG 문서(모듈별 markdown 노트 및 도구 설명)를 full context 모드로 검색해 초기 시도 실패를 줄였고, system prompt가 모듈 구조·이미지 retrieval 인터페이스·학습 규칙을 고정적으로 통제하도록 설계했습니다. 또한 스레드 단위 누적 질문을 활용해 같은 대화 맥락에서는 이전 요청(예: 차트/오버슈트 계산)을 참조하되, 스레드 간 기억은 요구하지 않도록 범위를 제한했습니다.

- **Empirical Impact**: 코드 블록 회상 성능은 검증 스크립트 기준으로 15개 코드 블록에서 총 48개 verbatim 라인 중 47개를 재현해 97.92%의 높은 회상률을 보였고, 1건은 상이한 모듈 상수(주제 참조 오류)로 나타났습니다. 논문은 RAG가 강의 노트에 한정된 소스만 사용하게 설계되어 일반적인 환각 지표 대신 ‘노트 기반 코드 블록/개념 재현’이 핵심 평가가 된다고 설명합니다. 결과적으로 VectorizationLLM은 개념-코드 연결성과 학습 무결성 두 축을 동시에 겨냥한 코스 특화형 교육 보조 도구로서, 2026년 하반기 실제 수업 적용을 통해 온라인 사용/중간·기말 성적 및 피드백으로 효과를 추가 검증할 계획입니다.



### Infinity-Parser2 Technical Repor (https://arxiv.org/abs/2607.07836)
- **Prior Approaches**: 문서 파싱은 파이프라인 기반(레이아웃→인식→읽기 순서 조립), end-to-end SFT 기반, 그리고 RL 기반으로 발전해 왔습니다. 다만 파이프라인은 단계 간 오류 전파가 누적되고, end-to-end는 대규모로 신뢰할 만한 주석 코퍼스 부족 때문에 OOD(도메인 외)·저자원 언어·차트/화학식 같은 특수 영역에서 성능 저하가 두드러졌습니다. RL 후학습 역시 대체로 단일한 텍스트 보상에 치우쳐 구조적 정확도·공간 정렬·태스크 간 전이가 덜 활용되는 한계가 있었습니다.

- **Core Contribution**: Infinity-Parser2는 제어 가능한 데이터 합성 파이프라인과 멀티태스크 RL을 결합해 end-to-end 문서 파싱을 목표로 합니다. 핵심은 (1) 반복 정제 가능한 합성 엔진으로 5백만 샘플 규모 양/중 이중언어 Infinity-Doc2-5M을 공개하고(요소 바운딩박스, 읽기 순서, Markdown/HTML/LaTeX/SMILES/구조화 차트 등), (2) 8개 동시 목표를 하나의 end-to-end 최적화 신호로 묶는 검증 가능한 멀티태스크 리워드 시스템을 도입했다는 점입니다. 또한 동일 아키텍처 하에서 저지연용 Flash와 정밀도 최우선 Pro 변형을 배포합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘문서 데이터의 신뢰도’와 ‘복수 태스크를 하나의 보상으로 정렬’하는 과정입니다. Infinity-Parser2는 DOM(Document Object Model)을 중간 표현으로 삼아 렌더링 시점에 읽기 순서·요소 유형·계층·구조화 텍스트를 같은 소스에서 동시 생성하고, 픽셀-정렬형 기하 라벨이 일관되게 나오도록 합성 엔진을 설계했습니다. 이어 data iteration flywheel(평가·배드케이스 분석→태깅 기반 마이닝→실데이터 주석+합성→누적 파인튜닝)로 모델의 약점을 다음 라운드 데이터에 직접 반영해 희귀 레이아웃과 특수 요소 커버리지를 단계적으로 확장합니다.

- **Empirical Impact**: 실험은 다양한 공개 벤치마크에서 Infinity-Parser2-Pro가 olmOCR-Bench 87.6%, ParseBench 74.3%를 달성하며 DeepSeek-OCR-2, PaddleOCR-VL-1.5, MinerU2.5를 앞섰다고 보고합니다. Infinity-Parser2-Flash는 Infinity-Parser-7B 대비 처리량을 3.68배(441→1,624 tokens/s) 끌어올리면서 정확도도 경쟁력을 유지해 대규모 배치/실서비스 적용 가능성을 보여줍니다. 또한 차트·화학식·document VQA로의 일반화 성능이 확인되며, ‘멀티태스크 RL 단일 보상’이 구조/추론 폭을 넓힌다는 점에서 의미가 큽니다.



### Idiobionics: The Unification of Privacy and Intelligent Robotic Prostheses (https://arxiv.org/abs/2607.07775)
Comments:
          8 pages, 3 figures

- **Prior Approaches**: 로봇 의수(bionic limbs)는 IoB(Internet of Bodies) 흐름 속에서 EMG·가속도계·자이로 등 센서와 ML 제어로 사용자의 의도를 추정하고 적응하도록 발전해 왔다. 다만 기존 IoB 보안/프라이버시 연구는 웨어러블(스마트워치, 피트니스 트래커) 중심이었고, 의료기기로 분류되어 사용자가 교체·회피하기 어려운 의수의 특수성은 상대적으로 공백이었다. 그 결과, 자율·반자율 적응성이 높아질수록 어떤 프라이버시 위협이 새로 생기는지에 대한 체계적 이해가 부족했다.

- **Core Contribution**: 이 논문은 프라이버시와 지능형 의수의 접점을 다루는 새로운 연구 의제인 idiobionics를 제안하고 정의한다. 이어서 상지 의수의 대표적 설정을 대상으로 공격 가능성을 보여주며, 향후 연구를 위한 공개 질문 리스트를 정리해 착수 장벽을 낮춘다. 즉, ‘기능 향상’과 별개로 ‘프라이버시 위험’을 설계 단계에서 선제적으로 평가하자는 관점을 제시한다.

- **Technical Challenges**: 핵심 난제는 의수의 적응성을 가능하게 하는 센싱·학습 구조가 역으로 민감 정보를 추론하는 공격 표면(threat vectors)을 만든다는 점이다. 저자들은 가속도계 데이터로 Activity Inference Attack(AIA)이 가능한지 확인하기 위해, 상지(팔꿈치 아래) 절단 사용 상황에 맞춘 실험 데이터를 수집하고 지도학습(전이학습 HarNet10 기반 finetuning)과 비지도학습(클러스터링)을 함께 적용했다. 그 결과 데이터가 제한적일 때도 공개/소비자급 도구로 활동 추론이 성립함을 보이며, 향후 보호 설계가 특정 사용자 하위그룹에 맞춰 일반화돼야 함을 시사한다.

- **Empirical Impact**: 실험에서는 활동 분류 정확도가 평균 83%로 나타나, 사전에 학습된 공개 모델을 활용한 프라이버시 공격이 현실적임을 보여준다. 또한 개인별 정확도 변동이 커 일부 사용자는 더 쉽게 식별되어 보호 전략을 ‘취약 하위그룹’ 관점에서 설계해야 한다는 결론을 제시한다. 저자들은 이 위협이 활동뿐 아니라 건강 상태, 인구통계, 심지어 진동 기반 음성/키 입력 계열 공격으로 확장될 수 있으며, 더 나아가 재활·보조 로봇(엑소스켈레톤)에도 전이될 수 있다고 전망한다.



### Alignment Plausibility: A New Standard for Assuring AI in Healthcar (https://arxiv.org/abs/2607.07766)
Comments:
          8 pages, 1 figure

- **Prior Approaches**: 기존 연구와 제품 안전 대응은 주로 가장 눈에 띄는 급성 위해를 줄이는 데 집중해 왔다. 하지만 의존성 형성, 경계(배운/관여) 침식, 왜곡된 신념의 증폭 같은 더 미묘하고 장기적인 위험 패턴은 상대적으로 덜 다뤄져 왔다.

- **Core Contribution**: 이 논문은 LLM 기반 정신건강 지원을 ‘주의(engagement)’ 중심 시장 구조에서 벗어나 구조적으로 안전하게 만들려면 정렬(alignment)을 3층 체계로 설계해야 한다고 주장한다. 구체적으로 임상 실무의 규범적 약속에 기반한 명시적 가치 명세, 그 가치를 학습에 내재화하는 training, 배포 중 가치 드리프트와 장기 위해를 감지하는 oversight를 제안한다.

- **Technical Challenges**: 핵심 난제는 가치-학습-감독이 함께 안전한 결과로 이어진다는 일관성을 어떻게 설득력 있게 구성하느냐에 있다. 이를 위해 논문은 ‘alignment plausibility’를 규제적 구성개념으로 제시하며, 모델의 가치 체계와 학습 절차, 감독 메커니즘이 안전한 정신건강 결과와 정합적임을 구조적으로 보이도록 하는 프레임을 제공한다.

- **Empirical Impact**: 저자들은 alignment plausibility를 생물학적 plausibility에 비유되는 ‘신뢰를 위한 주장/불신을 위한 반증’의 원칙으로 정립해, 능력이 있어도 해를 만들지 않도록 신뢰를 관리하고 결국 환자 이익으로 연결될 수 있다고 본다. 또한 이는 규제·평가 관점에서 시스템을 안전하게 만들 수 있는 논리적 틀을 제공한다는 점에서, 향후 건강 분야 AI 안전성 논의의 기준이 될 잠재력이 있다.



### Aligning Clinical Needs and AI Capabilities: A Survey on LLMs for Medical Reasoning (https://arxiv.org/abs/2607.07761)
Comments:
          Accepted by Machine Intelligence Research

- **Prior Approaches**: 기존 의료 LLM 연구는 진단·문서생성·질문응답 같은 응용에 집중했지만, 임상에서 요구되는 ‘추론 수준’과 모델 설계의 대응 관계가 일관되게 정리되지 못했다. 또한 기존 벤치마크는 정확도 중심이라서 근거 사용, 추론 완전성, 불확실성 처리, 정합성(grounding) 같은 안전성/신뢰성 요소를 충분히 분리해 측정하기 어렵다.

- **Core Contribution**: 이 논문은 임상의 역량을 Miller’s Pyramid을 5단계로 확장해 지식 인식부터 동적 케이스 관리까지의 ‘임상 역량-추론 요구’를 체계화한다. 동시에 연역(deductive)·귀납(inductive)·가설적 추론(abductive)을 의료 목표(정규화, 위험예측, 감별진단, 치료선택 등)와 연결하고, 실제 임상 워크플로에서 흔한 혼합 추론 설정도 명시적으로 다룬다.

- **Technical Challenges**: 핵심 난제는 (1) 데이터에서 임상 개념을 표준화·정규화하고, (2) 다단계 추론에서 근거를 유지한 채 오류를 누적시키지 않으며, (3) 모델의 환각과 grounding 실패를 평가·완화하는 것이다. 논문은 추론 유형과 역량 단계 기준으로 데이터/모델/평가를 함께 매핑하고, 정확도 외에 사실성, 추론 완전성, 내부 일관성, 증거 사용, 불확실성 처리까지 포함하는 평가 틀을 제안한다.

- **Empirical Impact**: 저자들은 추론 역량 5단계를 반영한 5,000샘플 벤치마크로 18개 최신 모델을 평가해 경향을 확인했다. 전문(medical specialist) 모델은 진단 중심 과제에 강하고, 일반(general) 고용량 모델은 의사결정 지원·다회 대화·요약에서 상대적으로 우세했지만, 길이-체류기간 같은 구조화된 시간 예측은 전반적으로 여전히 어렵다고 보고한다. 또한 모델 크기만으로 성능을 설명하기 어렵고, instruction 품질·도메인 데이터·추론 학습이 중요하다는 배치 방향(진단은 전문모델, 대화/지원은 일반모델)을 제시해 현장 적용 논의에 근거를 더한다.



### Adversarial Social Epistemology for Assemblies of Humans and Large Language Models (https://arxiv.org/abs/2607.07760)
Comments:
          50 pages

- **Prior Approaches**: 기존 논의는 주로 인식적 버블, 에코 챔버, 혹은 허위정보 확산 같은 현상에 초점을 맞추지만, 실제로는 ‘검증 가능해야 신뢰가 생기는’ 대화적 구조가 어떻게 악용되는지까지는 충분히 설명하지 못한다. 특히 연쇄적 증언, 추론, 제도적 인증, 암묵적 신뢰로 ‘스캐폴딩(scaffolded)된’ 공적 주장들이 어떤 방식으로 신뢰를 잠식당하는지는 상대적으로 덜 다뤄져 있었다.

- **Core Contribution**: 이 논문은 조밀하게 상호작용하는 커뮤니케이션 환경에서 신뢰가 성립하는 원리를 ‘adversarial social epistemology(ASE)’로 재정식화하고, 악의적 행위자가 그 신뢰를 악용하는 메커니즘을 분석한다. 핵심은 에코/버블이나 확산 모델이 아니라, 에이전트가 왜, 어떤 방식으로 스캐폴딩된 주장에 내재된 의무(commitments)와 권한(entitlements)을 표적으로 삼아 신뢰를 깨뜨리는지를 설명하는 언어와 분석틀을 제공하는 데 있다.

- **Technical Challenges**: ASE를 구현하려면 스캐폴딩된 주장에 대한 신뢰가 ‘추론 사슬의 감사 가능성(auditability)’에 의존한다는 점을 모델링하면서, 그 감사 가능성을 어떻게 우회·훼손할 수 있는지 정교하게 포착해야 한다. 논문은 epistemic networks에 기반한 메커니즘을 제시하고, 주장 해석을 위한 inferentialist semantics를 통해 감사·추적·시정 절차가 작동하도록 하는 ‘트러스트 브리치’(trust breach) 진단과 redressing(시정) 구동의 틀을 제안한다.

- **Empirical Impact**: 엄밀한 실험보다는 개념적·방법론적 분석을 통해, 기존의 ‘확산/분절’ 중심 설명이 놓친 신뢰 악용의 구조를 강조한다는 점에서 영향이 크다. 이 프레임은 온라인 담론, 제도 인증, 증언 기반 정보 전달처럼 추론 사슬이 얽힌 환경에서 감사를 강화하고 조작 가능 지점을 체계적으로 점검하는 연구·설계 방향을 제공한다.



### AI-integrated models for assessing agricultural resilienc (https://arxiv.org/abs/2607.07759)
- **Prior Approaches**: 기존 연구는 경제 시스템(GTAP 같은 모형)과 생물물리 시스템(APSIM 같은 모형)을 따로 다루는 경우가 많아, 충격이 생산·수확·가격으로 이어지는 연쇄효과를 일관되게 평가하기 어려웠습니다. 그 결과 정책·시장 참여자가 서로 다른 분야 결과를 한 번에 해석해 의사결정에 반영하기가 번거롭다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 GTAP과 APSIM을 통합해 농업 공급망 충격을 교차학제적으로 분석하는 AI 도구를 제안합니다. 사용자는 자연어로 질문하고 응답을 통해 충격의 경제적 파급과 생물물리적 변화를 함께 확인할 수 있습니다.

- **Technical Challenges**: 핵심 과제는 서로 다른 가정·스케일·입력변수를 가진 경제/생물물리 모형을 자연어 질의 기반의 워크플로로 안정적으로 연결하는 데 있습니다. 논문은 모델 간 통합과 질의-응답 과정을 설계해, 사용자가 복잡한 모형 파라미터를 직접 다루지 않아도 cross-disciplinary 영향을 평가하도록 구성했습니다.

- **Empirical Impact**: 이 도구는 정책기관과 시장 참여자가 공급망 쇼크에 대해 더 빠르고 포괄적인 시나리오 평가를 수행하는 데 도움을 줄 것으로 기대됩니다. 특히 자연어 인터페이스를 통해 학제 간 소통 비용을 낮춰, 향후 농업 회복탄력성 및 리스크 대응 분석의 실용성을 높인다는 의미가 있습니다.



### Context Graphs for Proactive Enterprise Agents (https://arxiv.org/abs/2607.07721)
- **Prior Approaches**: RAG와 GraphRAG 같은 방식은 검색-생성으로 출력을 ‘질문이 있을 때만’ 만든다는 점에서 본질적으로 reactive에 머뭅니다. ReAct/AutoGen/LangGraph 등 에이전트 프레임워크는 도구 사용·협업을 확장하지만, “인간 요청 없이 무엇을 먼저 봐야 하는지, 언제 행동해야 하는지”는 구조적으로 해결하지 못합니다. 기존 그래프 지식표현은 정적 스냅샷 중심이어서 실시간 상태 변화와 임계치 돌파를 이벤트로 취급하기가 어렵습니다.

- **Core Contribution**: 이 논문은 에이전트가 사람의 질의 없이도 ‘행동 가능한 정보를 먼저 노출’하도록 만드는 엔터프라이즈 프로액티브 아키텍처를 제안합니다. 핵심은 Context Graph로, 조직의 엔터티·관계·상태 전이를 live하게 모델링하고 delta event를 1급 이벤트로 남깁니다. 이 위에 Delta Detection Engine, Proactivity Scorer, LLM Surfacing Layer를 얹어 중요 신호를 사용자에게 우선순위로 알리는 end-to-end 시스템을 정의합니다.

- **Technical Challenges**: 문제는 (1) 임계치가 실제로 넘었을 때만 후보 인사이트를 만들고 (2) 알림 과다로 인한 경보 피로를 막으며 (3) LLM이 전체 그래프를 추론하지 않고도 근거 있는 알림을 생성하게 하는 것입니다. 저자들은 델타 로그를 기반으로 kk-hop 컨텍스트 스냅샷을 구성해 LLM의 근거를 제공하고, urgency·relevance·persona-fit·confidence를 통합한 Proactivity Score로 랭킹/필터링해 false positive를 억제합니다. 또한 중복 전송 억제(cooldown)와 역할 기반 프롬프트 스키마를 적용해 ‘사람에게 맞는’ 알림 품질을 높였습니다.

- **Empirical Impact**: 세 가지 범용 엔터프라이즈 케이스(계약 라이프사이클, 엔지니어링 인시던트 대응, 세일즈 파이프라인 상태 정비)에서 Precision@5=0.83, false positive rate=0.11, mean time to surface(MTTS)를 47분(reactive 기준)에서 30초 미만으로 단축했습니다. 이는 그래프 기반 live delta 감지와 점수화·노출 체계가 단순 생성 성능이 아니라 운영 생산성의 병목(정보 탐색·타이밍)을 직접 줄인다는 실증입니다. 더불어 에이전트를 ‘responder’가 아니라 ‘monitor’로 전환하는 설계 방향을 구체 구현과 지표로 제시했다는 점에서 의미가 큽니다.



### OpenCoF: Learning to Reason Through Video Generation (https://arxiv.org/abs/2607.08763)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 비디오 생성에서 Chain-of-Thought(변증/추론 단계)를 정교화하려는 시도는 주로 정적 이미지에 의존하거나 외부 도구·보조 생성·로컬 증거 추출 같은 우회 경로에 머물렀다. 최근에는 temporally connected frames로 추론을 펼치는 Chain-of-Frame(CoF) 개념이 제안됐지만, 일반 비디오 데이터로만 학습돼 CoF 전용의 다양한 감독과 아키텍처 설계가 부족하다는 한계가 있었다. 또한 벤치마크 구축은 활발했지만, 내부적으로 CoF 추론 상태를 구조적으로 강화하는 연구는 상대적으로 드물었다.

- **Core Contribution**: 이 논문은 CoF 추론을 겨냥한 공개 프레임워크 OpenCoF를 제안하며, 11개 태스크 패밀리의 추론 비디오 17,312개로 구성된 OpenCoF-17K 데이터셋을 공개한다. 이어서 Wan2.2-I2V-A14B를 OpenCoF-17K로 fine-tuning해 Wan-CoF를 만들고, reasoning 전용 기법 없이도 외부 4개 벤치마크에서 기준선 대비 유의미한 성능 향상을 확인했다. 마지막으로 CoF의 중간 추론 상태를 조직하기 위해 Visual Reasoning Tokens(vt)와 Textual Reasoning Tokens(tt)라는 두 종류의 reasoning-token 메커니즘을 도입하고 비교한다.

- **Technical Challenges**: 핵심 기술적 난제는 비디오 생성 모델이 원래 이미지/영상 합성 중심으로 설계돼 있어, 프레임 간 논리적 결과·장기 일관성·공간/물리/추론 연속성을 ‘중간 상태’로 명시적으로 저장하기 어렵다는 점이다. 논문은 이 문제를 두 계층으로 나눠 해결하려고 하는데, vt는 시각 latent 시퀀스에 reasoning 상태를 위한 학습 가능한 토큰을 삽입해 저수준 시각 단서를 구조화하고, tt는 텍스트 조건에 prompt-independent한 학습 우선순위를 추가해 고수준 의미/규칙을 보강한다. 또한 어떤 단계(모델 depth, denoising step, 공간, 시간)에서 토큰이 기여하는지 attention 분석으로 세분화해, 둘이 서로 보완적으로 시간 축과 경계 상태를 다룬다는 관찰을 제시한다.

- **Empirical Impact**: Wan-CoF는 4개 비디오 추론 벤치마크에서 Wan2.2-I2V-A14B 대비 전반적으로 개선되며, 예를 들어 MME-CoF Overall, Gen-ViRe, VIPER, RULER-Bench에서 모두 상승(Noise 대비 추론 하위 차원 중심)했다. 또한 토큰 추가 실험에서 vt와 tt는 벤치마크별로 성격이 다른 강점을 보이며, vt는 planning·visual stability 같은 시각-시간적 유지가 중요한 항목에서, tt는 instruction alignment·structural 같은 텍스트 기반 정렬/구조 항목에서 더 두드러졌다. 결과적으로 OpenCoF-17K의 넓은 temporal supervision과 중간 reasoning state를 위한 명시적 메커니즘이 함께 필요하다는 결론을 뒷받침하며, 데이터셋·모델·코드를 오픈소스로 공개해 후속 연구를 촉진한다.



### SLORR: Simple and Efficient In-Training Low-Rank Regularization (https://arxiv.org/abs/2607.08754)
- **Prior Approaches**: 기존 low-rank 압축은 SVD 기반 절단이 대표적이지만, 이는 학습 과정에서 스펙트럼을 충분히 low-rank으로 만들지 못하면 성능 저하가 커진다. 이를 보완하려는 학습-time 정규화는 (1) 매 스텝마다 큰 weight에 대한 SVD 계산이 필요하거나, (2) 아키텍처를 factorized 형태로 바꿔 추가 파라미터와 최적화 동역학을 유발하거나, (3) 주기적 업데이트를 위한 cached 상태와 타깃 rank 같은 하이퍼파라미터 의존이 생기기 쉽다.

- **Core Contribution**: SLORR은 원래 weight 행렬을 그대로 두면서, SVD 없이 in-training 저랭크 유도 정규화를 적용하는 stateless이고 architecture-preserving한 프레임워크다. Hoyer sparsity metric 기반(SLORR-Hoyer)과 nuclear norm 기반(SLORR-Nuc) 두 변형을 제시하며, forward/backward에서 필요한 스펙트럴 항을 GPU-friendly 근사로 직접 정규화한다.

- **Technical Challenges**: 핵심 난제는 nuclear norm·Hoyer 같은 스펙트럴 정규화의 값/그래디언트에 SVD 항이 포함되어 있어 매 iteration마다 SVD를 돌리면 학습 비용이 감당되지 않는다는 점이다. SLORR은 generalized polar factor(=UV^T)의 항을 Polar Express 기반 반복 근사로 대체해 SVD-free로 구현하면서, 정규화 값과 그래디언트에 대한 근사 보장(Proposition 3.1)을 함께 제공하고 실제 구현에서는 Polar Express의 수치 안정화와 하이퍼파라미터 설정을 따르도록 한다.

- **Empirical Impact**: ImageNet-1K에서 ResNet-50·ViT-B/16·ViT-L/16의 short-horizon continued training과 ResNet-18 pretraining을 비교한 결과, SLORR은 compressibility를 개선하면서 학습 오버헤드를 8% 미만으로 유지했다. 또한 LLM pretraining(135M/560M)에서 SLORR-Hoyer는 압축 후 성능을 unregularized 대비 더 잘 보존했으며 평균 training overhead는 1% 미만이었다.



### Dimensionality Reduction Meets Network Science: Sensemaking on UMAP's kNN Graph (https://arxiv.org/abs/2607.08746)
Comments:
          Code and demo: this https URL

- **Prior Approaches**: UMAP은 고차원 데이터를 2D로 투영해 탐색하는 워크플로가 주로 쓰이지만, 실제로는 UMAP 내부에 kNN 그래프가 먼저 구성되고 이후 레이아웃 최적화가 진행된다. 기존 연구들은 주로 2D 레이아웃의 품질(왜곡/해석)을 개선하거나 보정하려 했고, 그래프 자체를 활용한 감각(대표성·핵심성·국소 응집)의 가능성은 상대적으로 덜 다뤄졌다. 한편 임베딩 이후에는 HDBSCAN이나 k-medoids 같은 방법으로 클러스터링/선정을 수행하는 경우가 일반적이지만, 이는 2D 투영의 왜곡된 기하를 다시 의존할 수 있다.

- **Core Contribution**: 이 논문은 UMAP이 만드는 pre-projection kNN 그래프를 그대로 “중간 표현”으로 남기고, 그 위에 PageRank, k-core decomposition, clustering coefficient를 적용해 데이터 sensemaking을 수행한다. PageRank는 그래프 중심성 기반으로 대표 데이터 포인트를 뽑고, k-core는 조밀한 코어와 희박한 페리퍼리를 계층적으로 구분하며, clustering coefficient는 이웃끼리 서로를 얼마나 촘촘히 연결하는지로 국소적으로 응집된 마이크로 이웃을 찾아낸다. 즉, 산점도(2D) 해석을 다듬기보다 UMAP 내부 그래프에서 질문을 직접 해결하는 접근을 제안한다.

- **Technical Challenges**: 핵심 과제는 (1) UMAP 내부 kNN 그래프의 구조적 신호가 투영 왜곡 없이 해석 가능한지를 보이는 것과 (2) 그래프 알고리즘 결과가 neighborhood size(n_neighbors) 같은 하이퍼파라미터에 흔들리지 않는지 검증하는 것이다. 논문은 UMAP의 밀도-적응 membership strength를 PageRank의 가중치로 활용해 위상적 중심성을 강화하고, k(=n_neighbors)를 여러 값으로 바꿔도 대표성 순위가 높은 상관을 유지함(예: PageRank 안정성)으로 재현성을 확인했다. 또한 k-core는 directed 그래프에서 in-degree만을 기반으로 분해해 out-degree 고정으로 생기는 바닥값 문제를 피하고, clustering coefficient는 이웃 쌍의 연결 비율로 국소 응집을 정량화했다.

- **Empirical Impact**: MNIST와 Fashion MNIST에서 PageRank 기반 대표 선정은 kk-medoids, 그리고 downstream SVM 정확도 측면에서 HDBSCAN/메도이드류와 비교해 경쟁력 있거나 보완적인 성과를 보였다. 특히 PageRank는 클래스 균형(Jensen-Shannon divergence)에서 더 나은 비례성을 보이며, 예산이 커질수록 격차가 확대되는 경향을 보였다. k-core와 clustering coefficient는 각각 코어-페리퍼리 구분과 마이크로 이웃 응집을 드러내 HDBSCAN의 “어느 그룹인가/클러스터 멤버십 확률”과는 다른 해석 축을 제공하며, CC 상위 포인트가 의미적으로 더 일관된 국소 이웃을 식별함을 정량·정성으로 확인했다.



### Validity of LLMs as data annotators: AMALIA on authority (https://arxiv.org/abs/2607.08731)
- **Prior Approaches**: 기존 LLM 기반 텍스트 코딩 평가는 주로 사람 라벨과의 일치도(예: F1, kappa)로 신뢰성(reliability)을 확인하는 방식이었습니다. 그러나 일치도는 타당도(validity)를 보장하지 않으며, 모델이 이론이 지정한 추론이 아니라 표면 상관관계로 ‘비슷한 코드’를 맞출 수도 있다는 문제가 남습니다. 특히 zero-shot prompt는 일부 구성에서 false positive를 크게 부풀릴 수 있어, 단순 정확도 비교만으로는 측정 품질을 판단하기 어렵습니다.

- **Core Contribution**: 이 논문은 ‘회복 격차(recovery gap)’라는 운영적 검증 절차로, LLM이 구성(construct)을 이론이 요구하는 증거 경로로 측정하는지 확인합니다. 구체적으로, holistic prompt로 한 번에 코드(예: authority)를 내리게 한 뒤, 이 코드를 구성 이론에 따라 원자 단위(탐지·구분·평가)로 분해하고 통합 규칙으로 재조합했을 때 성능이 얼마나 유지되는지로 타당도를 봅니다. 또한 영어에서 grain calibration(입자 수준 보정)이 된 측정 도구가 AMALIA-9B와 유럽 포르투갈어(pt-PT)에 그대로 이식되는지도 실험합니다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM이 구성의 표면 단서가 아니라 관계(누가 복종해야 하는가, 권위가 확언/도전/위반되는가 등)를 추론해야 한다는 점이며, 그 과정이 블랙박스라 오류 원인을 합의도만으로는 분해할 수 없다는 것입니다. 논문은 구성 정의를 구성 요소와 clause(원자 질문)로 쪼개고, 이 clause 답을 이론 기반의 명시적 integration rule로 다시 합치는 방식으로 ‘올바른 증거 경로’ 여부를 진단합니다. 나아가 실제 포르투갈어 말뭉치 구축에서는 transcreation 시 의미·발화 의도·관점·지시대상(stability)을 보장하도록 여러 verification gate를 걸어 평가 오염 가능성을 줄였습니다.

- **Empirical Impact**: 결과(초록 요약 기준)로는 AMALIA-9B가 권위(authority) 구성에서 사람 코더와의 일치도는 준수하지만, 분해 후 재조합해도 holistic 성능의 약 절반만 회복되어 recovery gap이 크게 나타났습니다. 오류 분석에서는 특히 권위자 주변의 도덕적 분노 같은 표면 상관 단서에 의존하는 정황이 보고됩니다. 반면 동일 포르투갈어 코퍼스에서 한 다국어 open LLM은 recovery gap을 더 잘 메워, ‘코퍼스 문제’보다는 AMALIA의 측정 도구로서의 한계를 시사하며, 국가 주권 모델이라도 construct 타당도 검증 없이는 단독 측정에 신중해야 함을 강조합니다.



### Pose-to-Biomechanics: Bridging 3D Human Pose Estimation and Biomechanical Attribute Prediction (https://arxiv.org/abs/2607.08725)
Comments:
          23 pages, 2 figures

- **Prior Approaches**: 3D 인체 포즈 추정은 기하학적 키포인트 정확도(MPJPE 등)에 맞춰 최적화돼, 복구 결과를 물리적으로 해석 가능한 생체역학 지표(토크, 지면반력, 근활성 등)로 연결하는 데는 공백이 있었다. OpenCap 같은 마커리스 파이프라인은 근골격 모델과 시뮬레이션/역다이내믹스를 결합하지만, 다중 뷰 캘리브레이션이나 대상별 최적화가 필요해 확장성이 제한된다. 또한 바디 모델·메시 복원은 해부학적 사실성은 높이지만, 기존 3D 포즈 추정기의 17-joint 출력을 범용 생체역학 속성 공간으로 일괄 변환하는 플러그인형 매핑 메커니즘은 부족했다.

- **Core Contribution**: 이 논문은 BioModule을 제안한다. BioModule은 어떤 3D 포즈 estimator 뒤에 그대로 붙일 수 있는 lightweight temporal transformer 플러그인으로, 표준 17-joint 3D skeleton 입력만으로 3티어(kinematic/kinetic/neuromuscular)로 구성된 17개 생체역학 속성을 예측한다.

- **Technical Challenges**: 핵심 과제는 서로 다른 좌표계의 해부학적 대응을 frame-accurate하게 맞추는 것이었다. 이를 위해 Human3.6Mplus를 구축해 Human3.6M의 17-joint와 OpenSim 기반 musculoskeletal label을 펠비스 루트를 앵커로 정렬·기하학적 검증(2D 투영이 sub-pixel, 펠비스 마커가 machine precision 수준 일치)을 수행하고, 이 정렬된 supervision으로 BioModule을 학습시켰다. 또한 생체역학 속성들은 차원·잡음·불확실성이 달라 단순 합 손실이 편향될 수 있어, kinematic/kinetic/neuromuscular 3그룹에 tiered weighted multi-task loss를 적용해 안정적으로 학습한다.

- **Empirical Impact**: BioModule은 7개 state-of-the-art 3D 포즈 추정기 각각의 출력에 대해 downstream 생체역학 예측 품질을 체계적으로 비교하도록 벤치마크를 구성했다. 이를 통해 상류 포즈 추정기의 정확도/아키텍처 품질이 하류 생체역학 계층별 예측 충실도로 어떻게 전파되는지 처음으로 정량 분석한다. 결과적으로 BioModule은 비전 기반 3D 포즈 추정에서 물리적으로 해석 가능한 인간 모션 분석으로 이어지는 소형·모듈형 브리지로 자리잡을 가능성을 보여준다.



### ProjAgent: Procedural Similarity Retrieval for Repository-Level Code Generation (https://arxiv.org/abs/2607.08691)
- **Prior Approaches**: 레포지토리 수준 코드 생성은 단일 함수가 아니라 여러 파일에 흩어진 의존성과 프로젝트 관습을 함께 맞춰야 해서, 일반 코드 생성보다 훨씬 어렵습니다. 기존 맥락 검색은 주로 BM25나 dense embedding처럼 어휘/표면 의미 유사도에 의존해왔고, 호출 의존성이 없더라도 비슷한 절차를 공유하는 함수는 놓치는 경우가 많았습니다. 특히 표면 유사도만으로는 오히려 생성 품질을 떨어뜨릴 수 있다는 문제도 보고돼, 더 적절한 검색 신호가 필요해졌습니다.

- **Core Contribution**: ProjAgent는 레포지토리 수준 코드 생성에서 ‘procedural similarity(절차적 유사성)’를 명시적인 검색 신호로 도입합니다. 목표 함수를 중간의 reasoning step으로 분해한 뒤, 각 step마다 유사한 절차를 수행하는 저장소 함수들을 agentic workflow로 찾아 전통적인 semantic/lexical 검색과 결합해 더 풍부한 컨텍스트를 구성합니다. 또한 정적 분석 기반 피드백 loop으로 컴파일/분석 오류를 보수적으로 반복 수정합니다.

- **Technical Challenges**: 절차적 유사성을 찾으려면 이름이나 도메인이 달라도 구현 로직이 비슷한지를 표현해야 하는데, LLM hidden state를 그대로 cosine similarity로 비교하면 anisotropy 때문에 무관한 step도 유사하게 보이는 문제가 있었습니다. ProjAgent는 reasoning subspace projection(추론 관련 부분공간 투영)을 사용해 언어적 변형을 억제하고, 여기에 PCA 기반 debiasing을 더해 판별력을 높였습니다. 더 나아가 step 분해·검증 과정에서 LLM이 만든 단계가 실제 코드에 근거하는지(설명-본문 일치, snippet 존재, NL-코드 임베딩 유사도) 여러 단계로 보증해 검색 오염을 줄였습니다.

- **Empirical Impact**: REPOCOD 평가에서 ProjAgent는 Pass@1 41.14%를 달성하며, retrieval 기반 기존 베이스라인을 성능적으로 앞섰습니다. 결과는 절차적 유사성이 기존 어휘/의미 중심 검색 차원에서는 포착되지 않던 유용한 컨텍스트를 제공할 수 있음을 실증합니다. 코드 생성 품질을 높이기 위해 단순 검색을 넘어 단계별 절차 신호와 정적 분석 기반 반복 수정을 함께 설계했다는 점에서, 레포지토리 수준 SE 연구에 영향이 큽니다.



### A Practical Investigation of Training-free Relaxed Speculative Decoding (https://arxiv.org/abs/2607.08690)
Comments:
          preprint

- **Prior Approaches**: 기존 speculative decoding(spec-dec)은 빠른 보조 drafter가 토큰을 초안(draft)하면, verifier가 병렬 검증하며 필요 시 확률적으로 reject·resample해 원래 LLM의 샘플링 분포를 엄밀히 보존해 ‘lossless’ 가속을 달성한다. 다만 최근 연구는 이 엄밀한 분포 보존이 오히려 속도 향상의 상한을 만들 수 있고, 분포 보존을 느슨하게 하면 더 큰 속도/성능 트레이드오프 또는 경우에 따라 capability 이득까지 기대할 수 있다고 본다. 이 흐름에는 per-token relax를 통한 다양한 휴리스틱·최적화 아이디어가 있으나, 학습 없이(training-free) 구현 가능한 방법들 전반을 같은 틀에서 정리·비교한 실무형 조사가 부족했다.

- **Core Contribution**: 본 논문은 training-free relaxed speculative decoding을 대상으로, 기존 방법들을 단일 공통 프레임워크로 통합해 설명하고 실무자가 바로 적용·비교할 수 있는 관찰을 제공한다. 특히 strict spec-dec에서의 reject/residual/bonus 샘플링을 느슨한 목표분포 π로 치환하는 형태로 각 기법을 체계화해, 어떤 제약 하에서 α 같은 완화 파라미터를 어떻게 튜닝해야 하는지 연결한다. 또한 relaxed에서 속도 향상을 단순 ‘수용률’로만 판단하면 안 되고, 생성 길이 변화까지 포함해 추정해야 함을 강조한다.

- **Technical Challenges**: relaxed을 적용하면 분포가 변하면서 단순히 더 많은 draft를 수용하는 것만이 이득이 아니라, capability 평가 비용이 커질 수 있으며 수용률 향상이 실제 응답 지연으로 직결되지 않을 수 있다. 논문은 속도 모델에서 drafter 비용(C_drafter), verifier 병렬 검증 비용(C_verify), 그리고 draft 길이 대비 평균 수용 토큰 수의 상호작용이 복잡하게 얽혀 최적 draft length N_draft가 뒤집힐 수 있음을 보여준다. 이를 해결하기 위해 저자들은 방식별로 서로 다른 실험 설정을 보정할 수 있도록 ‘relaxed target분포 설계(π)’와 ‘실현 가능한 속도 모델(수용 길이·비용·생성 길이)’을 함께 벤치마크 프레임으로 묶었다.

- **Empirical Impact**: 실험에서 저자들은 현대 drafter-verifier 페어와 reasoning 벤치마크를 대상으로, 다양한 추론 파라미터에서 relaxed 기법들을 비교하며 paper별 주장과 실제 구현 성능의 간극을 드러낸다. 특히 많은 relaxed 접근은 verifier 분포에 가까운 ‘좋은 language model drafter’를 전제로 하는 경우가 많아, lightweight 전용 multi-token-prediction drafter로 가면 기대했던 이득이 약해질 수 있음을 확인한다. 또한 일부 결과는 lossless(strict) 대비 가까운 속도 향상도 가능하지만, relaxation이 오히려 capability 평가 부담을 요구할 수 있다는 실무적 takeaways를 구체적으로 제시한다. 



### WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search (https://arxiv.org/abs/2607.08662)
Comments:
          Work in progress

- **Prior Approaches**: ReAct 기반 단일 에이전트는 한 번에 긴 탐색 궤적을 수행해야 해 컨텍스트 한계로 인해 깊이·너비가 동시에 커지면 성능이 급격히 떨어진다. 다중 에이전트 접근은 병렬 탐색과 결과 집계를 통해 커버리지는 늘리지만, 재귀적 깊이 확장·협업 형태 전환·증거에 근거한 확장이 유연하지 못하다는 한계가 있다. 또한 작업을 쿼리 표면 의미에만 맞춰 분해하면 웹에서 정보가 실제로 조직되는 방식과 어긋나 중복 탐색과 집계 실패가 발생할 수 있다.

- **Core Contribution**: WebSwarm은 추론 중에 에이전트 노드를 동적으로 만들어 “점진적 재귀 위임(progressive recursive delegation)”을 수행하는 프레임워크를 제안한다. 각 노드는 로컬 목표와 search mode(원자적 탐색 atom, 반복 탐색 deep, 병렬 분할 wide, 미지 경계 열거 entity_collect)를 함께 받아, 스스로 풀거나 자식 노드를 위임해 증거를 상향 반환한다. 또한 웹 정보 구조 탐사와 동종 형제 노드 간 프로세스 경험 재사용을 통해, 분해·확장·협업을 증거가 쌓일수록 함께 진화시키도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 초기 쿼리만으로는 필요한 하위 과제가 언제 드러날지 예측하기 어려워 재귀 깊이를 안정적으로 키우는 것, (2) 하위 목표마다 필요한 협업 프로토콜이 달라 고정된 협업 토폴로지로는 부족한 것, (3) 웹에서의 정보 조직 방식과 분해 축이 어긋날 때 비효율이 누적되는 것이다. WebSwarm은 objective–mode pair로 하위 노드별 해결 전략을 명시하고, 하위 노드가 증거를 반환한 뒤 상위 노드가 expand·revise·terminate를 선택하는 피드백 루프를 구현했다. 더불어 wide 확장이 필요한 노드에는 웹-구조 probing을 붙여 확장 축을 정렬하고, scout 노드로부터 동종 sibling의 실패/성공 경로를 요약한 경험 kv를 재주입해 중복 시행착오를 줄인다.

- **Empirical Impact**: BrowseComp-Plus, WideSearch, DeepWideSearch, GISA 등 4개 벤치마크에서 WebSwarm은 단일 ReAct과 다양한 다중 에이전트 기준선을 일관되게 능가하거나 경쟁력 있는 성능을 보였다. 특히 깊이/너비가 섞인 DeepWideSearch처럼 단계 전환이 잦은 과제에서 고정 협업 패턴의 약점을 모드 기반 재귀 위임으로 완화했다. ablation 결과로도 재귀 위임 자체가 성능 저하를 유발하고, 웹-구조 probing은 도구 호출 수를 크게 줄이며, 동종 노드 경험 재사용은 품질(Item F1 등) 하락을 막는 역할을 확인했다.



### Multi-Modal, Multi-Environment Machine Teaching for Robust Reward Learning (https://arxiv.org/abs/2607.08647)
Comments:
          Accepted to RLC 2026. Conference paper

- **Prior Approaches**: 기존 IRL/보상학습 기계교육(machine teaching) 연구는 대체로 단일 MDP에서 시연(demonstrations) 중심 또는 단일 환경 고정 가정에 머물렀습니다. 이 경우 얻어진 보상은 해당 환경의 동역학에 얽혀(environment entanglement) 다른 배치로 옮기면 일반화가 깨지는 문제가 자주 발생했습니다. 또한 서로 다른 feedback modality(비교, 교정, E-stop 등)가 어떻게 보상 식별의 제약을 형성하는지, 그리고 환경 동역학 변화와 함께 어떤 제약이 되는지는 충분히 다뤄지지 않았습니다.

- **Core Contribution**: 이 논문은 heterogeneous feedback modality와 다중 환경 환경 동역학이 함께 보상 공간을 어떻게 제약해 일반화 가능한 reward를 만들 수 있는지에 대한 이론·알고리즘을 제시합니다. 먼저 단일 MDP 내 교육으로는 reward identifiability가 환경에 의존해 구조적 잔여 모호성이 남을 수 있음을 정식화합니다. 이를 바탕으로, Hierarchical Set Cover Optimal Teaching(HSCOT)에서 ‘어떤 환경을 먼저 선택해 제약 방향을 열어줄지’와 ‘그 환경 안에서 어떤 feedback을 제한된 예산으로 질의할지’를 계층적으로 최적화합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다중 MDP에서 환경별로 가능한 feature difference 제약의 span이 달라 남는 모호성을 해소하려면 어떤 조합이 필요한지, (2) 제한된 query 예산 아래에서 그 조합을 촘촘히 수축시키는 feedback 인스턴스를 어떻게 고르느냐입니다. 논문은 무한데이터(unlimited-data) 이상에서 modality별 제약의 기하학을 비교를 기준으로 분석하고, 비교는 전역 순서(global ordering) 제약이 강해지는 반면 E-stop은 궤적 국소성으로 제약 공간이 제한된다는 점을 보입니다. HSCOT는 이를 계층 set-cover 관점으로 바꿔, 먼저 보완적 제약을 드러내는 환경을 탐욕적으로 고르고, 그 환경들에서 비용이 낮은 질의들로 gBEC(일반화된 행동 동등성 클래스) 영역을 효율적으로 줄이는 전략을 사용합니다.

- **Empirical Impact**: 실험에서는 HSCOT이 uniform teaching 대비 held-out 환경에서 constraint coverage가 더 높고 regret이 유의하게 낮아지는 결과를 보였습니다(동일한 feedback budget 조건). 이는 ‘단일 환경에서 더 많은 피드백을 주는 것’보다 ‘환경을 섞고 modality도 조합’하는 접근이 dynamics-robust 보상 학습에 중요하다는 실증 근거가 됩니다. 요약하면, 다중 환경·다중 modality 교육 설계가 IRL 보상 일반화의 병목을 직접적으로 완화할 수 있음을 보여준 연구로 평가됩니다.



### UltraX: Refining Pre-Training Data at Scale with Adaptive Programmatic Editing (https://arxiv.org/abs/2607.08646)
- **Prior Approaches**: 기존 데이터 품질 개선은 규칙 기반 필터링/클리닝과 모델 기반 선택·정제(Refinement)로 나뉜다. 규칙 기반은 비용이 낮지만 고정 휴리스틱에 묶여 인스턴스 수준 변이를 놓치고, LLM 기반 정제는 품질은 높여도 대규모 전처리에서 효율과 신뢰성이 떨어진다. 또한 함수 호출 기반 방법은 ProX·RefineX처럼 편집 함수 공간이 불완전하거나(삭제/수정 중심, 삽입 부재), 시드 감독과 실행 단계에서 중복·경계 문제를 완전히 해결하지 못한다.

- **Core Contribution**: UltraX는 대규모 pre-training 데이터를 function-calling으로 정제하되, 편집 함수 공간을 deletion·modification에 더해 insertion까지 확장해 인스턴스 수준의 fine-grained 편집을 가능하게 한다. 이어서 “신뢰 가능한 program-supervision 생성 파이프라인”을 구축해 원문-정제 텍스트 쌍을 계층적 매핑으로 구조화된 감독 신호로 바꾸고, 연산 조합 비율 통제와 저신뢰 예시 필터링으로 학습 분포 안정성을 높인다. 마지막으로 긴 문서에서도 실행 신뢰성을 올리기 위해 sliding-window 예측, 전역 operation aggregation, 체계적 후처리를 포함한 대규모 실행 절차를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 대규모 코퍼스에서 빠르게 작동하면서도 의미 수준 품질을 안정적으로 개선하는 감독 신호를 만들기, (2) 함수 호출 결과를 긴 문서에 적용할 때 중복 매칭·연산 간 간섭·창 경계 단절을 줄이기, (3) 경량 모델이 예측한 operations가 실제로 일관되게 실행 가능하도록 정규화·검증하는 것이다. UltraX는 dataset-adaptive prompt optimization으로 expert LLM의 고품질 end-to-end 정제 텍스트를 만들고, Line Alignment Mapping과 Dynamic Context Replacement로 모호한 위치 지정 문제를 줄인 뒤 저신뢰 필터링과 연산 조합 기반 샘플링 비율로 감독 품질을 조절한다. 실행 단계에서는 window별 예측을 전역 라인 번호 공간으로 재매핑해 비중복 영역만 유지하고, 파싱/정합성 검사·모호한 replacement 제거·인접/상호간섭 연산 병합·반복 패턴 fallback 등을 통해 안정성을 확보한다.

- **Empirical Impact**: 실험은 1B MiniCPM 모델을 5개 코퍼스(FineWeb, RedPajama-V2, AICC, Ultra-FineWeb, FineWeb-ProX-Doc)에서 20B 토큰 예산으로 from-scratch pre-training해 LightEval 성능을 비교했다. UltraX는 5개 코퍼스 전반에서 평균 성능이 가장 높았고, 50개 task-corpus 조합 중 34개에서 최상위를 기록했으며, 평균 상대 개선이 Raw 대비 약 2.00%, ProX-C 대비 약 1.53% 수준으로 보고된다. 특히 FineWeb에서는 16B 토큰에서도 Raw·ProX-C의 20B 최종 성능을 이미 상회해 data efficiency와 정제 신뢰성이 개선됐음을 보여준다.



### When Structured Sparse Autoencoders Learn Consistent Concepts Across Modalities (https://arxiv.org/abs/2607.08605)
- **Prior Approaches**: Sparse autoencoder(SAE)는 은닉표현을 희소한 잠재 특성으로 분해해 기계적 해석(mechanistic interpretability)에 유리하다는 점에서 주목받았다. 다만 기존 SAEs는 재구성 손실과 원소 단위 희소성만 최적화해, vision-language models(VLMs)에서는 모달리티에 맞는 개념이 시각에서 조각나게 학습되는 문제가 남는다. 그 결과 한 잠재 뉴런이 여러 의미를 동시에 가지는 잔여 polysemanticity가 생기고, 이미지 패치가 특정 개념을 활성화해도 그 패치들이 의미적으로 일관된 시각 영역으로 정렬되지 않는다.

- **Core Contribution**: 이 논문은 Structured Sparse AutoEncoder(S2AE)로, 시각 모달에서 개념이 ‘의미적·공간적 일관성’을 갖도록 구조적 희소성 정규화를 도입한다. Transformer attention 유사도와 공간 인접성을 함께 써서 이미지 패치를 개념 단위의 시각 영역으로 그룹화하고, vanilla SAE 학습 시 영역 간 분리는 exclusive sparsity로, 영역 내 일관성은 group sparsity로 강제한다. 이를 통해 개념 특성이 서로 다른 시각 영역에서는 경쟁적으로 분화하면서, 같은 영역에서는 응집된 방식으로 활성화되게 만든다.

- **Technical Challenges**: 핵심 기술 난제는 픽셀 수준 경계가 명시되지 않은 상태에서, 단일 개념에 대응하는 ‘시각적으로 일관된 영역’을 안정적으로 추정하는 것이다. 저자들은 attention 기반 유사도만으로는 의미적 outlier가 많이 생긴다는 관찰을 바탕으로, 공간 거리(패치 좌표 기반)까지 결합해 병합형 agglomerative clustering 거리 행렬을 구성하고 영역을 얻는다. 그 위에 영역-수준(group-level) 활성 프로파일을 만들어 exclusive sparsity와 group sparsity를 각각 영역 간 경쟁과 영역 내 공동 활성에 적용함으로써, 시각-언어 shared feature 공간까지 정돈되는 효과를 노린다.

- **Empirical Impact**: Qwen2.5-VL-7B-Instruct에서 S2AE는 semantic alignment(mIoU) 평균 6.06% 향상과 representational efficiency(l0 norm 60.81) 개선을 달성하면서도 Explained Variance 99%대의 재구성 충실도를 유지한다. 또한 cross-modal 분석에서 구조적 시각 우선이 단일 의미성(monosemanticity)을 높여, 두 모달리티 멀티모달 특징 모두 semantic consistency 평균 3.08%, monosemanticity 점수 평균 2.37% 향상을 보였다. 즉, 시각 측 구조 제약이 공유된 개념 사전(dictionary)에서 언어 쪽 의미 정합성까지 간접적으로 끌어올리는 메커니즘을 실증했다.



### SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling (https://arxiv.org/abs/2607.08565)
- **Prior Approaches**: 기존 LLM 스케줄링 연구는 요청을 인스턴스별 점수로 라우팅하는 방식에서 크게 벗어나지 않으며, KV reuse(캐시 히트)와 로드(배치/in-flight 토큰)를 가중합 또는 곱으로 결합해 균형을 맞춰왔다. 그러나 에이전트 기반 서빙에서는 목표가 TPS 중심으로 바뀌고, per-token 지연은 완전히 포기할 수 없지만 기존보다 완화된다. 또한 KV reuse가 지배적인데도, 많은 스케줄러가 캐시를 가진 인스턴스로 과도하게 몰아주며 로드 밸런스를 희생한다는 문제가 남는다.

- **Core Contribution**: 이 논문은 에이전트 전용(autonomous agent가 요청을 발행) 클러스터에서의 request scheduling을 두 개의 실제 대규모 에이전트 서빙 트레이스로 최초로 체계 분석한다. 그 결과, 기존 스케줄러가 KV reuse를 “더 잘” 만들려다 오히려 일부 인스턴스만 과부하되어 TPS가 상한에 막힌다는 점을 구체적으로 보인다. 이를 바탕으로 SMetric(균형 세션 중심 스케줄링)을 제안해, 세션의 첫 요청과 후속 요청을 다르게 처리하면서 KV reuse와 로드 밸런스를 동시에 노린다.

- **Technical Challenges**: 핵심 과제는 KV reuse를 유지하면서도 특정 캐시 인스턴스에 세션이 고정(pinning)되어 클러스터가 불균형해지는 현상을 막는 것이다. SMetric은 로컬 GPU tier의 캐시 히트를 우선하되, 글로벌 tier KV store를 통해 첫 요청이 어디로 가든 필요한 KV를 제때 가져오게 하여 TPS 병목을 피한다. 또한 세션 상태를 저장하지 않도록, 라우터가 “세션 턴 정보”를 입력만으로부터 효율적·정확하게 추출해(메트릭 stateless) 첫 요청은 로드 밸런스만, 후속 요청은 캐시 인지 방식으로 라우팅하며, 드문 꼬리 불균형에는 세션 마이그레이션으로 대응한다.

- **Empirical Impact**: 실험에서는 vLLM과 LMCache 스택에서 SMetric을 state-of-the-art 스케줄러 및 Bailian 프로덕션 스케줄러와 비교하고, prefill-decode colocation과 prefill-decode disaggregation 모두에서 평가한다. 결과로 SMetric은 글로벌 store가 있는 colocation 설정에서 클러스터 TPS를 10–16% 향상하고, disaggregation 설정에서는 prefill TPS를 2–34% 끌어올리며 per-token 지연(예: TPOT, TTFT)도 개선한다. 특히 세션 첫 요청만 로드 밸런스를 맞추는 전략이 글로벌 tier 부담을 낮추면서도 대부분의 로컬 reuse를 보존해, 에이전트 서빙 스케줄링의 실용적 설계 기준을 제시했다.



### VocaDet: Sample-Driven Open-Vocabulary Object Detection and Segmentation via Visual Tokenization and Vector Database Retrieva (https://arxiv.org/abs/2607.08541)
- **Prior Approaches**: SAM, Grounding DINO, INSID류처럼 text/visual prompt 기반 open-vocabulary 검출·분할은 강력하지만, 프롬프트 입력이 필요하거나 매 프레임 유사도 계산이 반복돼 확장성이 떨어진다. 또한 Rex-Thinker, Rex-Omni 등은 multimodal LLM을 쓰지만 추론 비용과 레퍼런스 매칭 부담이 남는다. Training-free 방식(예: DINOv2+SAM2 메모리 뱅크)은 학습 없이 동작하나, 사용자가 큰 규모의 positive/negative 샘플 저장소를 계속 늘려갈 때 이를 자동으로 반영하며 효율적으로 검색하는 데는 제약이 있다.

- **Core Contribution**: VocaDet은 사용자가 제공한 positive/negative 샘플을 기반으로 객체 개념을 ‘재학습 없이’ 바로 학습·갱신하는 sample-driven open-vocabulary 검출·분할 프레임워크를 제안한다. 연속적인 시각 표현을 agglomerative clustering으로 이산 visual token(그리고 멀티 그레인)으로 만들고, 이를 위치 편향 보정과 공간 topology 정보와 함께 vector database의 object memory로 저장해 필요 시 검색으로 localization/segmentation을 수행한다. UA-DETRAC에서, 기존 카테고리 기반 학습 없이도 성능을 보이면서 샘플 저장소를 누적하면 인식 능력이 계속 개선되는 확장성을 강조한다.

- **Technical Challenges**: 핵심 난제는 (1) 연속 feature를 검색 가능한 discrete vocab로 안정적으로 양자화하고, (2) 시각 유사도만으로 생길 수 있는 인접 객체 병합 같은 오류를 줄이며, (3) large-scale 메모리에서 배경 때문에 생기는 중복 검색을 효율화하는 것이다. VocaDet은 DINOv3 feature에 agglomerative clustering(클러스터 민감도 파라미터)로 멀티 그레인 토큰을 만들고, position-debiased 표현을 저장·질의에 동일 적용하며, 객체 매치는 feature similarity뿐 아니라 topology(인접 관계) 일치까지 요구하도록 설계했다. 또한 fixed-camera 환경에서 배경 토큰을 background feature memory로 필터링해 vector database 질의 수를 줄이고, 여러 그레인에서 나온 박스는 NMS로 정리한다.

- **Empirical Impact**: UA-DETRAC 실험에서 VocaDet은 detector 학습 없이도 경쟁력 있는 open-vocabulary 검출 성능을 보였고, positive/negative 샘플을 추가할수록 vector memory가 풍부해지며 결과가 점진적으로 향상되는 점을 확인했다. 한편, 같은 범주의 인접 객체가 동시에 등장할 때 시각 유사도와 클러스터링에 의존하는 탓에 하나로 합쳐지는 한계를 관찰했으며, 이는 discriminative boundary 분리가 부족할 때 발생한다. 초기 벡터베이스 구축 단계에서의 cold-start 문제도 언급되어, 향후 메모리 구성과 카테고리 경계 학습을 더 적응적으로 만드는 방향이 제시된다.



### DocMaster: A Hierarchical Structure-Aware System for Document Analysis (https://arxiv.org/abs/2607.08539)
Comments:
          4 pages, demo paper, under revision

- **Prior Approaches**: 기존 RAG나 문서 QA 시스템은 PDF를 단순 텍스트 chunk로 쪼개 벡터 검색/생성에 투입하는 경우가 대부분입니다. 이 과정에서 섹션-표-그림-수식 같은 문서의 계층 구조와 단면(섹션 간) 의미 연결이 사라져, 필터링과 후속 질의응답의 성능이 떨어집니다. 또한 섹션 전반에 흩어진 근거를 함께 모아야 하는 조건에서도, 청크 단위 검색은 교차 근거 집계를 제대로 수행하지 못하는 한계가 있습니다.

- **Core Contribution**: DocMaster는 문서를 계층 구조를 보존한 document tree로 파싱하고, 그 위에 구조를 반영한 structure-aware semantic index를 구축해 필터링과 분석을 함께 개선합니다. 문서 트리 기반 탐색, 임베딩 의미검색, hyper-edge 매칭의 tri-modal retrieval을 통해 자연어 조건에 맞는 문서를 더 정확히 고릅니다. 이후 필터된 문서를 grounded context로 사용해 RAG 기반 후속 질문응답까지 지원합니다.

- **Technical Challenges**: 핵심 도전은 (1) 계층을 유지해 문맥 빈 조각을 줄이고, (2) 섹션 간 암묵적 의미 관계를 벡터 유사도만으로는 포착하지 못하는 문제, (3) 조건 하나에 필요한 원거(멀리 떨어진 근거)를 교차 섹션 단위로 집계하는 문제입니다. DocMaster는 SEC(Structural Entropy Correlation)로 구조적으로 먼데 의미가 비슷한 텍스트 쌍을 골라 PC-KMeans에 must-link/cannot-link 제약을 LLM으로 주입하고, 섹션 내부에서 LLM 그룹핑으로 hyper-edge를 만들어 교차 섹션 의미 오버레이를 형성합니다. 이후 document-tree traversal, FAISS 기반 semantic search, hyper-edge matching 결과를 LLM이 종합해 문서별 boolean 필터 결정을 내립니다.

- **Empirical Impact**: 논문은 DocMaster를 React+FastAPI 웹 데모로 구현해, 문서 묶음 업로드부터 인덱스 시각화(트리/Hyper-Edge/클러스터) 및 실시간 필터링, 근거 인용을 포함한 follow-up Q&A까지 end-to-end 워크플로를 보여줍니다. 특히 hyper-edge가 섹션 간 연결을 드러내 플랫 검색에서는 보이지 않는 관계를 기반으로 검색/필터가 가능하다는 점을 시나리오로 제시합니다. 또한 PC-KMeans 및 SEC 등 하이퍼파라미터를 조정하며 필터 결과를 비교할 수 있어, 실사용 관점의 튜닝 가능성도 강조됩니다.



### When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability (https://arxiv.org/abs/2607.08535)
Comments:
          6 pages, 6 figures, 4 tables

- **Prior Approaches**: 그동안 LLM-as-judge 연구는 MT-Bench, Chatbot Arena 등에서 인간 선호와의 상관을 보이며 평가 도구로서의 유용성을 입증해 왔습니다. 다만 기존 보고는 “평가자 모델이 강해질수록 점수도 잘 맞는다”는 식의 1차원 모델 정렬에 머무는 경우가 많아, 능력 향상인지·벤치마크 슬라이스 차이인지·편향 감소인지·프로토콜 영향인지가 분리되지 않았습니다. 또한 position/verbosity 같은 편향과 집계·debate 같은 프로토콜 설계가 순위를 바꿀 수 있다는 점은 알려졌지만, 이 요인들이 평가자 스케일링과 함께 어떻게 변하는지는 덜 명확했습니다.

- **Core Contribution**: 이 논문은 evaluator-replacement ambiguity(평가자 교체 후 점수 변화의 원인 불명확성)를 측정 타당성 문제로 규정하고, 평가자 신뢰도를 단일 정확도로 보지 않도록 분해 프레임을 제시합니다. 구체적으로 (1) 단일 평가자 타당성, (2) 편향 견고성, (3) 반복/집계에서의 오류 비독립성, (4) debate 같은 프로토콜의 auditability(감사 가능성) 4요소로 신뢰도를 구성합니다. Qwen3 dense judge의 파라미터 스케일링과 MiniMax M2~M2.7의 released API 업그레이드 경로를 비교해, “업그레이드가 곧바로 상호교환 가능한 신뢰도 개선”이 아니라는 점을 실증적으로 보여줍니다.

- **Technical Challenges**: 가장 큰 기술적 도전은 “더 강한 judge = 더 신뢰할 만함”이지만, 실제 점수 변화가 편향·슬라이스·파이프라인 파싱/집계 차이로도 생길 수 있다는 점입니다. 논문은 adjacent 비교를 McNemar로 짝지어 수행하고, Holm correction으로 다중비교를 제어했으며, LLMBar에 대해 position/verbosity/granularity bias probes로 편향의 잔존을 직접 측정했습니다. 또 jury의 경우 다수결이 자동 증폭이 아니라 오류 상관(ρ)을 고려해야 함을 beta-binomial 형태의 ρ-보정 예측으로 확인했고, debate는 parser와 fallback 로그 부재 때문에 deliberation 인과 추정은 어렵지만 프로토콜 감사 로그가 필수임을 드러냈습니다.

- **Empirical Impact**: 4개 데이터셋에서 judge 업그레이드는 경로별로 다르게 나타났습니다. Qwen3는 1.7B→4B에서만 비교적 견고한 adjacent gain이 확인된 반면, MiniMax M2~M2.7 인접 릴리즈는 대부분 유의미한 개선으로 이어지지 않았고(홀름 보정 후에도 유의), 편향과 position/verbosity bias는 줄어들어도 완전히 사라지지 않았습니다. 다수결 jury는 오류 상관이 높아 이득이 제한적이었고, debate는 결정 변화를 크게 만들 수 있지만 파싱 실패 fallback의 재현 로그가 없으면 “토론 효과”로 귀속하기 어렵다는 점이 강조됩니다. 결론적으로 LLM-as-judge 보고서에는 데이터 슬라이스, 편향 및 바이어스 프로브, 오류 의존성(ρ) 추정, 그리고 프로토콜 audit trail까지 포함해야 신뢰할 만한 측정으로 해석할 수 있다는 권고가 제시됩니다.



### Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing (https://arxiv.org/abs/2607.08497)
Comments:
          16 pages, 7 figures, 8 tables. Project page: this https URL Code: this https URL

- **Prior Approaches**: 최근 unified multimodal 모델은 한 아키텍처에서 시각-언어 이해와 이미지 생성/편집을 함께 수행하지만, 장문 대화에서 과거의 모든 시각 토큰을 공통 컨텍스트에 계속 주입하는 구조적 한계가 있습니다. 이로 인해 visual token 폭증으로 추론 예산이 줄고, 긴 누적 컨텍스트에 대한 암묵적 attention 의존이 교차 턴 시각 참조를 불안정하게 만들어 retrieval 오류와 의미 드리프트가 커집니다. 메모리 보강을 시도한 에이전트들도 영상/텍스트를 각각 다루거나(비디오 중심, 단일 태스크 중심) 시각 디테일을 충분히 보존하지 못해 near-duplicate 이미지 구분에 취약합니다.

- **Core Contribution**: 논문은 Cognitive-structured Multimodal Agent(CMA)를 제안하며, 시각 정보를 Episodic Visual Memory(EVM)로 외부화하고 필요할 때만 관련 에피소드를 선택적으로 reactivates 하도록 설계합니다. Perceptual Abstraction Engine(PAE)이 이미지에서 태그·캡션·썸네일의 구조화 표현을 만들고, Cognitive Retrieval Engine(CoRE)이 대화 흐름에 맞는 시각 에피소드를 검색한 뒤, Multimodal Executive Controller(MEC)가 태스크 의도 추론과 실행 계획을 담당합니다. 또한 turn-level 시각 retrieval 감독이 부족한 문제를 Unified Scenario Engine으로 해결해, 회수할 에피소드에 대한 미세 주석이 포함된 다턴 대화 데이터를 생성합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “장기 대화에서 어떤 과거 이미지를 회수해야 하는가”에 대한 명시적 감독 신호를 확보하는 것이었습니다. 기존 데이터셋은 단일 턴 grounding이나 짧은 컨텍스트만 다루는 경우가 많아, 각 턴마다 정답 retrieval 집합을 제공하지 못합니다. 저자들은 Unified Scenario Engine으로 생성된 시나리오에 대해 turn-level retrieval annotations를 만들고, CoRE에는 SFT+RL(지연 학습된 검색 정책 최적화)을 적용하며 PAE도 retrieval 유용성 기준으로 강화해 메모리 작성-검색을 end-task 관점에서 맞춥니다.

- **Empirical Impact**: M2CA-Bench에서 8B CMA는 20턴 세션 기준 retrieval 정확도 91.4%를 기록해, 32B unified baseline을 +8.2%p로 능가하고 per-turn 추론 시간을 23.1s에서 12.7s로 거의 절반 수준으로 줄였습니다. 특히 대화가 길어질수록 격차가 확대되어 Full→Hard에서의 성능 하락이 더 완만했으며, retrieval 정확도가 생성 품질(0~10 점수)로 직접 전이되는 경향도 확인됐습니다. 또한 CMA-Harness로 동일 인지 구조를 도구 실행·웹 접근·이미지 생성/편집까지 확장했으며, monolithic 파라미터 스케일링보다 구조화 메모리와 모듈형 의사결정이 장기 멀티모달 에이전트에 더 확장성 있고 효율적이라는 점을 시사합니다.



### The Context Access Divide: Interaction-Level Architecture as a Complementary Dimension of Agentic Inequality (https://arxiv.org/abs/2607.08495)
Comments:
          19 pages, 2 figures

- **Prior Approaches**: Sharp et al. (2025)는 agentic inequality를 availability(접근 가능 여부), quality(능력 수준), quantity(동시 운용 수)로 정리하며, 주로 인구/조직 단위의 격차를 설명합니다. 하지만 이 틀은 “개별 상호작용에서 문맥 확보 책임이 누구에게 있는가” 같은 더 미시적인 구조 차이를 충분히 드러내지 못합니다. 두 사용자가 같은 요금제·같은 모델에 접근해도, 문맥을 시스템이 자동으로 끌어오는지 아니면 사용자가 매번 붙여야 하는지에 따라 AI 효용이 달라질 수 있기 때문입니다.

- **Core Contribution**: 논문은 Context Access Divide(CAD)를 제안합니다. 이는 사용자의 지식 코퍼스에서 관련 문맥을 AI가 자율적으로 검색(Dynamic Context Retrieval)하는지, 아니면 Manual Attachment처럼 사용자가 쿼리마다 문서를 지정·첨부해야 하는지에 의해 생기는 상호작용 수준의 격차입니다. 또한 contextuality(사용자 누적 지식자산을 자율 접근하는 정도)를 agentic inequality의 보완 축으로 정식화하며, 단순 접근/성능 격차로 환원되지 않는 임계효과(threshold)를 강조합니다.

- **Technical Challenges**: 핵심 기술적 과제는 CAD를 “상호작용 아키텍처가 작업 성공확률을 어떻게 비선형으로 붕괴시키는가”로 설명할 논리와 모델링을 구성하는 것입니다. 논문은 Model Context Protocol(MCP)과 retrieval-augmented generation(RAG) 및 도구 사용을 중심으로 Manual Attachment(MAM), Walled Dynamic Context Retrieval(Walled DCRM), Open Dynamic Context Retrieval(Open DCRM) 구조를 구분하고, MAM에서는 코퍼스 크기가 커질수록 사용자의 문서 회상·식별·첨부 확률이 fan effect 메커니즘처럼 저하된다고 확률모형을 세웁니다. 결과적으로 k개의 접합적으로 필요한 문맥이 모두 들어와야 하는 태스크에서 MAM 성공확률이 k와 코퍼스 크기에 대해 조합적으로(combinatorially) 급락하고, Open DCRM은 코퍼스 크기 의존성이 낮아 이러한 붕괴에서 구조적으로 “격리”된다고 주장합니다.

- **Empirical Impact**: 정량적 평가는 실제 사용자 데이터의 직접 추정보다는, fan effect와 개인정보관리(PIM) 문헌을 근거로 한 이론적 확률모형 시뮬레이션(예시 파라미터 포함)으로 CAD의 임계효과를 보여주는 형태입니다. 코퍼스가 수천~수만 파일 규모로 커지면, 접합 의존성이 높은 지식합성 작업에서 MAM은 급격히 실패하고 Open DCRM은 높은 성공확률을 유지하는 격차가 수배~수천 배 수준으로 벌어질 수 있다고 제시합니다. 이 결과는 지식노동 계층화와 AI 플랫폼 거버넌스 관점에서, “열려 있는 문맥 접근(Open DCRM)” 같은 아키텍처 선택이 사회적 효용 분배를 좌우할 수 있음을 시사합니다.



### VEGAS: Human-Aligned Video Caption Evaluation via Gaz (https://arxiv.org/abs/2607.08489)
- **Prior Approaches**: 기존 비디오 캡셔닝은 시각-언어 모델이 의미적으로는 그럴듯한 문장을 만들지만, 군중 주석을 평균낸 기준이라 개별 시청자의 주의(시선)를 반영하지 못하는 경우가 많습니다. 그래서 여러 작품이 gaze를 학습용 감독신호로 쓰거나(예: supervised fine-tuning), 추론 시 인간 시선을 활용해 이해/의도 해석을 보강하려 했습니다. 다만 학습 없이(test-time) 시선 기반으로 캡션 자체를 개인화해 “평가/선택”까지 수행하는 방법과, 이를 검증할 동기화된 멀티모달 데이터는 제한적이었습니다.

- **Core Contribution**: 이 논문은 시선으로 개인화된 캡션을 고르는 학습 없는(metric-based) 방법 VEGAS(Video caption Evaluation via GAze Score)를 제안합니다. VEGAS는 시선에 의해 주목된 영역만으로도 캡션이 얼마나 잘 예측되는지를 정보이론적으로 측정해, 개인의 주의와 일치하는 텍스트를 선호하도록 설계됐습니다. 또한 VEGAS 점수로 rejection sampling을 수행해 VLM 재학습 없이 개인화 캡션을 선택하는 절차를 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 시선-캡션 간 상호정보량의 직접 계산이 불가능하다는 점이며, 이를 위해 논문은 VLM의 토큰별 likelihood를 활용해 불능의 분포를 근사합니다. 구체적으로 주목 영역( attended )과 비주목 영역( non-attended )을 분해한 뒤, 비주목 영역을 추가했을 때 캡션 예측이 얼마나 더 좋아지는지를 pointwise conditional mutual information 형태로 점수화합니다. 이후 여러 후보 캡션을 VLM에서 뽑고, VEGAS 점수가 가장 낮은(비주목 정보 의존이 작은) 캡션을 선택해 시선 정렬을 달성합니다.

- **Empirical Impact**: 저자들은 동기화된 비디오-시선-캡션을 제공하는 멀티도메인 데이터셋을 구성해(egocentric AEA, instructional SlideVQA) VEGAS를 검증합니다. AEA에서는 인간 초점과의 정렬이 유의미하게 개선되며 mean SBERT similarity가 +0.0856 향상되고 caption-to-video retrieval의 mAP도 여러 rank에서 +1.14%~+2.48% 수준으로 개선됩니다. 반면 SlideVQA에서는 SBERT 향상이 +0.0256으로 더 작고 유의성이 약해, VEGAS 효과가 “시선으로 구체적 지시어 모호성을 푸는” 상황에서 더 두드러진다는 도메인 의존성이 관찰됩니다.



### Two Axes of LLM Abstention: Answer Correctness and Question Answerability (https://arxiv.org/abs/2607.08456)
- **Prior Approaches**: 기존의 abstention(응답 보류) 접근은 생성 답변의 불확실성 한 가지 값에 임계값을 걸어 “거절”을 결정합니다. 그러나 이 방식은 틀린 답(정답성 실패)과 아예 답하면 안 되는 질문(답불가능/거짓 전제)을 같은 축으로 다뤄 open-world 영역을 제대로 표현하지 못합니다.

- **Core Contribution**: 이 논문은 응답 보류가 필요할 때의 실패를 두 개의 별도 축—정답성(correctness)과 답가능성(answerability)—으로 분해해, 같은 결정에서도 신호가 서로 다르게 나타남을 보입니다. 또한 두 축 각각에 대해 별도 위험 예산을 캘리브레이션해, 각 임계값을 동시에 만족할 때만 답하도록 하는 정책을 제안합니다.

- **Technical Challenges**: 핵심 난제는 “모델이 틀리게 답할 확률”이 아니라 “해당 질문이 애초에 답 가능한지”를 신뢰성 있게 분리해 측정하는 것입니다. 연구진은 hidden states에서 선형 프로브로 답가능성 AUROC를 크게 끌어올리며(특히 CREPE의 자연발생 거짓 전제에서 출력 기반 신호는 거의 chance), 단순 premise-check 지시는 오히려 거짓/진실 전제를 무차별로 반박하게 만들어 probe 라우팅으로 이를 수정했습니다.

- **Empirical Impact**: SelfAware 및 자연 데이터 벤치마크 CREPE에서, 출력의 answer-confidence는 답가능성 구분에 거의 실패하는 반면 hidden-state 프로브는 이를 유의미하게 포착합니다. 더 나아가 두 축을 결합한 캘리브레이션 정책은 스케일이 커질수록 잘못 답하는 비율 상한이 조여지며, 14B에서는 단일 임계값 정책이 사실상 인증하지 못하는 상황에서도 유일하게 인증 가능한 결과를 보였습니다.



### Spatio-Temporal Scheduling Prediction Under Backhaul Delay for Resilient Coordinated Beamforming (https://arxiv.org/abs/2607.08454)
- **Prior Approaches**: 기존 분산 5G/6G에서 coordinated beamforming(CBF)은 기지국이 이웃 기지국의 최신 scheduling 정보까지 빠르게 받아야 성능이 나옵니다. 하지만 backhaul latency로 정보가 stale해지면 SLNR 기반 정밀 억제가 ‘이미 비활성인 UE’로 향하거나 ‘새로 활성화된 UE’를 놓쳐 sum rate가 떨어지고, coordinated가 오히려 uncoordinated baseline보다 나빠질 수 있습니다. 한편 GNN은 무선의 공간 구조를 잘 잡지만 사용자 활동이 정적이라고 가정하는 경우가 많고, spectral-temporal forecasting은 예측은 잘해도 물리계층 beamforming 의사결정까지 연결되지 못했습니다.

- **Core Contribution**: 이 논문은 backhaul 지연을 없애는 대신, 기지국이 로컬에서 이웃 셀의 UE scheduling 상태를 미래 시점으로 예측해 SLNR 기반 CBF에 대체 입력으로 넣는 ‘예측-보조 의사결정’ 프레임워크를 제안합니다. 핵심은 Spectral Temporal Graph Neural Network(Spectral Temporal Graph Neural Network, StemGNN)를 이용해 delayed history로부터 이웃 셀의 현재(또는 미래) scheduling state를 추정하고, 이를 leakage covariance 계산에 반영해 stale fault의 영향을 줄인다는 점입니다. 또한 UE 수가 변하는 동적 네트워크를 고려해 permutation-invariant 표현으로 확장해 재학습 없이 사용자 유입/이탈에도 동작하도록 했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘지연된 바이너리 scheduling’을 다루되, UE 간 공출현/동시 스케줄링 같은 inter-UE 구조 의존성과 시간 지연을 동시에 학습해야 한다는 점입니다. 저자들은 StemGNN의 spectral-temporal 설계를 그래프 형태로 결합해 latent correlation layer에서 UE 간 adjacency를 데이터로부터 학습하고, 그래프 스펙트럼 변환으로 UE 간 상관을 분리한 뒤 temporal 주파수 처리를 거쳐 binary scheduling을 예측하도록 구성했습니다. 또한 모델 입력의 UE 차원을 노드로 취급해 permutation-equivariant 성질과 shared head를 유지함으로써, UE ID/정렬이 바뀌어도 동일한 모델 파라미터로 추론 가능하게 했습니다.

- **Empirical Impact**: Quadriga Urban Micro(UMi) 채널, 3셀 massive MIMO(셀당 64안테나, 총 60 UE), proportional fair scheduler 환경에서 StemGNN은 Horizon 1 평균 scheduling 예측 정확도 87.57%를 기록했으며 LSTM/GRU/Simple RNN/Markov 대비 모든 예측 horizon에서 우위를 보였습니다. 특히 Horizon이 길어질수록 temporal autocorrelation보다 inter-UE 구조가 중요해지는데, 이 구간에서 LSTM 대비 최대 7.71%p 향상이 관찰됐습니다. coordinated beamforming에 통합했을 때는 ‘backhaul 지연 1 TTI’로 발생한 sum rate 손실의 57–73%를 회복하며, no-prediction baseline 대비 sum rate를 9.58–14.35% 개선하고, cell-edge 사용자의 Lag-1 fairness 손실도 최대 약 83%까지 복구해 지연이 있어도 edge AI가 robust inter-cell coordination을 가능하게 함을 보여줍니다.



### ADORN: Adaptive Drift handling for Open RAN using Reinforcement Learning (https://arxiv.org/abs/2607.08443)
- **Prior Approaches**: 기존 O-RAN의 drift(학습 데이터 특성 변화로 인한 성능 저하) 대응은 성능 지표가 임계값을 넘으면 재학습을 트리거하는 방식이 주류였다. DDM/EDDM 같은 임계값 기반은 낮은 임계값에서 재학습 오버헤드가 커지고, 높은 임계값에서는 성능 저하로 SLA 위반 위험이 커지는 한계가 있다. 통계 기반이나 스트림 기반 방법도 동적·복잡한 환경에서 적절한 평가 주기/임계값을 고정하기 어려워 적응성이 떨어진다.

- **Core Contribution**: 이 논문은 ADORN으로, 재학습 결정을 MDP로 정의하고 Q-learning 에이전트가 “정확도”와 “재학습 비용”을 동시에 최적화하는 적응형 정책을 학습한다. 드리프트가 임계선(nMAE 기준)을 넘을 때만 필요한 재학습을 선택하도록 설계해, 불필요한 업데이트를 줄이면서도 성능 제한 내 유지를 목표로 한다. 또한 multi-expert LSTM 앙상블을 써서 단일 모델 재학습 과정에서 발생하기 쉬운 catastrophic forgetting을 완화한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 연속적인 트래픽 변화를 어떻게 상태로 표현할지, (2) 재학습 트리거를 언제 실행해야 SLA를 지킬지, (3) 재학습 빈도와 정확도의 장기 균형을 어떻게 학습할지였다. 이를 위해 유입 트래픽의 통계 특징(평균·분산)으로 상태를 이산화하고, 행동은 retrain/idle의 이진 결정으로 구성했으며, 보상함수는 드리프트 “존재 여부”뿐 아니라 “크기”까지 반영하는 piecewise 형태로 설계했다. 모델은 상태에 맞는 LSTM specialist를 선택해 예측하고, Q-learning의 Bellman 업데이트로 Q값을 갱신하며 정책을 수렴시킨다.

- **Empirical Impact**: 실험은 Colosseum 트래픽 데이터 기반의 8개 시나리오에서 real-time 및 synthetic 설정으로 평가했으며, greedy 및 random 기준선과 비교해 재학습 오버헤드를 줄이면서도 nMAE가 드리프트 허용 임계선 아래/근처에서 안정적으로 유지됨을 보였다. Greedy는 가장 낮은 오차를 보이지만 매 시점 재학습으로 비용이 고정적으로 높았고, Random은 트래픽 동학성을 반영하지 못해 재학습 횟수와 보상이 크게 흔들렸다. ADORN은 학습이 진행될수록 재학습 횟수가 약 4–6회에서 1–3회 수준으로 감소하며 보상도 양수로 전환·안정화했고, 앙상블의 MSE가 모든 specialist에서 빠르게 수렴해 catastrophic forgetting 완화 효과를 뒷받침했다.



### EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data (https://arxiv.org/abs/2607.08436)
- **Prior Approaches**: 관점·속도·행동 스타일이 맞지 않는 상황에서 인간 egocentric 데이터를 행동 복제(behavior cloning, BC) 방식으로 공동학습하면, 사람의 형태·습관이 그대로 섞인 ‘실행 불가한 모션’을 로봇이 따라 하며 성능이 떨어진다. 즉, 액션 디코더 하나를 통해 들어오는 인간 데이터가 물체/장면/의미 같은 전이 가능한 요소와 행동 고유 요소를 얽어 버리는 것이 핵심 한계다. 더 나아가 기존 WAM도 어떤 ‘세계(world) 표현’이 인간-로봇 전이를 좌우하는지 체계적으로 비교하지 못했다.

- **Core Contribution**: 이 논문은 World Action Model(WAM) 공동학습에서 ‘액션’이 아니라 ‘장면이 어떻게 진화하는지(미래 상태 예측)’를 보조 신호로 추가하면, 전이 가능한 동역학 기반 표현을 만들 수 있음을 주장한다. 나아가 전이를 좌우하는 세계 표현의 조건을 appearance abstraction, cross-embodiment consistency, ego-motion factoring의 세 축으로 정리하고, 이를 만족하는 타깃으로 Pixel(재구성), DINO(시맨틱), 3D motion flow(기하적 모션)를 비교한다. EgoWAM은 정책 백본과 데이터 혼합을 고정한 채 세계 예측 타깃만 바꿔, 표현 선택의 효과를 ‘통제된 실험’으로 분리해 보여준다.

- **Technical Challenges**: 어떤 세계 타깃은 카메라/머리 움직임과 장면 변화를 섞어 supervision이 어긋나거나, 픽셀 재구성처럼 외형·모션이 엉켜 전이가 막히는 문제가 있다. 이를 해결하기 위해 Pixel VAE 기반 타깃은 재구성 실패모드의 기준선으로 두고, DINO는 외형을 추상화하지만 이미지 격자에 여전히 공간적으로 인덱싱된 한계를 갖게 설계한다. 3D motion flow는 Aria VIO 기반 카메라 자세로 카메라-정렬 좌표계에서 플로우를 정의해, 외형·에이전트 차이를 줄이면서 ego-motion을 분해하도록 구성했다.

- **Empirical Impact**: 세 가지 실제 양손(bimanual) 작업에서 WAM 공동학습은 대규모 in-the-wild egocentric 인간 데이터 스케일링에서 BC보다 더 일관되게 이득을 보였다. 픽셀 기반 예측은 전이가 약했지만, DINO와 3D motion flow는 의미 있는 향상을 보였고 DINO는 OOD(장면·물체 미일치)에서 최대 4배 수준의 일반화 개선, 3D flow는 in-domain 성능을 약 20–30% 끌어올렸다. 결론적으로 미래 동역학을 학습 신호로 쓰되, world representation을 시맨틱 추상화 또는 3D 기하적 근거로 맞출 때 인간 데이터의 효과가 ‘전이 가능한 표현’으로 정렬된다는 점이 실증됐다.



### Predicting Male Fertility Using Machine Learning: A Semen Parameters Based Analysis with the VISEM Datas (https://arxiv.org/abs/2607.08429)
- **Prior Approaches**: 기존 남성 불임 평가는 WHO 기준에 따른 정액검사를 중심으로 하며, 수기 현미경 판독 특성상 관찰자 편향과 기관 간 변동성이 커 진단이 일관되지 않을 수 있다. AI/ML 연구는 VISEM 같은 데이터에서 정자 운동성(motility) 연속값 예측이나 영상/추적 기반 CASA 관련 과제에 집중해 왔지만, WHO 임상 기준에 맞춘 Fertile/Sub-Fertile/Infertile의 다중 분류까지 폭넓게 체계 비교한 시도는 상대적으로 제한적이었다.

- **Core Contribution**: 이 논문은 공개 VISEM 데이터셋(85명)의 핵심 정액 지표(정자 농도, 형태, progressive motility)를 이용해 WHO 임계값에 기반한 남성의 Fertile, Sub-Fertile, Infertile 상태를 지도학습 분류로 수행한다. LazyPredict로 40개 넘는 supervised 모델을 빠르게 벤치마킹해, 임상 의사결정에 참고 가능한 고정밀·객관적 평가 가능성을 제시한다.

- **Technical Challenges**: 표본 수가 85명으로 작고(클래스 경계가 겹칠 가능성), 여러 정액 지표가 서로 상관(다중공선성)을 보여 모델 안정성 확보가 어려울 수 있다. 연구팀은 전처리·특성 공학 후 LazyPredict로 다수 모델을 비교하고, 5-fold cross-validation 및 multiclass ROC-AUC로 강건성을 검증했으며, top 모델(Nearest Centroid, SVM, QDA 등)의 분리 성능을 함께 확인했다.

- **Empirical Impact**: 결과적으로 Nearest Centroid 분류기가 94.2% 정확도를 달성했으며, SVM과 QDA 등 경쟁 모델도 약 94% 안팎의 성능을 보였다. AUC는 Fertile 0.95, Sub-Fertile 1.00, Infertile 0.97로 전 클래스 분별력이 높았고, 혼동은 주로 Average와 Slow(경계 케이스) 사이에서 최소 수준으로 발생했다. 전반적으로 정액검사의 수기 판독 한계를 보완하는 빠른 기준선 도구로서, andrology 및 ART(보조생식술)에서 환자별 의사결정 지원 가능성이 실증되었다.



### When Synthetic Speech Is All You Have: Better Call GRPO (https://arxiv.org/abs/2607.08409)
Comments:
          Submitted to SLT 2026

- **Prior Approaches**: 규제 산업(예: 은행)에서는 개인정보·생체정보 이슈로 실제 음성 데이터 수집과 재사용이 크게 제한돼, TTS로 만든 합성 음성을 전사(Transcript)에서 생성해 ASR 도메인 적응을 해왔다. 하지만 합성 음성은 실제 음성과 음향적으로 불일치가 있어 성능 향상이 제한되며, 이를 줄이려는 연구도 대체로 supervised fine-tuning(SFT) 틀 안에서 진행돼 왔다.

- **Core Contribution**: 이 논문은 합성 음성으로 LLM 기반 ASR을 적응할 때, 강화학습을 사용하면 SFT보다 synthetic-to-real gap을 더 잘 메울 수 있다고 주장한다. 특히 critic-free 방식인 Group Relative Policy Optimization(GRPO)을 적용해, 동일한 합성 음성 데이터로도 sequence-level 목표를 통해 WER를 크게 낮춘다. 합성-only 적응에서 WER이 36.71%→22.09%(SFT 대비 40% 상대 감소)로 개선되며, SFT 후 GRPO를 이어붙이면 45% 상대 개선까지 확장된다.

- **Technical Challenges**: 핵심 난제는 합성 음성의 국소적인 잡음(프로소디 불일치, 발음/채널 시뮬레이션 결함)이 학습 신호를 왜곡해 삽입 오류(hallucinated insertion)를 유발한다는 점이다. 연구진은 토큰 단위 cross-entropy가 기준 전사를 ‘모두’ 따라가려는 성향을 줄이는 대신, 샘플된 전체 가설을 WER 기반 보상으로 상대 비교해 좋은 전사를 강화하는 방식으로 해결했으며, 이 과정에서 critic 없이 group-relative advantage를 사용해 학습 안정성을 확보했다.

- **Empirical Impact**: 실험 결과 GRPO의 이득은 representation의 대규모 재학습보다 behavior 변화(특히 stopping calibration)에서 나왔다. SFT는 음성 지원이 끊긴 뒤에도 자신 있게 이어서 생성하는 반면, GRPO는 경계 이후 confidence를 더 빠르게 낮추고 삽입 꼬리(insertion tail)를 크게 줄였으며 오디오 토큰 주의(attention)도 더 정보가 안정적인 구간에 집중했다. 또한 WER 보상은 충분하며 복잡한 보상 결합은 불리할 수 있고, 합성 풀을 무조건 늘리는 것보다 ‘실제 5–10시간 + 합성 다량’ 조합이 대부분의 개선을 만든다는 실용적 결론을 제시한다.



### Track2Map: Online Deformable SLAM with Motion-Aware Pose Optimization in Robotic Surgery (https://arxiv.org/abs/2607.08408)
Comments:
          Accepted at MICCAI 2026. This is the submitted version prior to peer review. The final authenticated version will be available on SpringerLink

- **Prior Approaches**: 기존 3D Gaussian Splatting(3DGS) 기반 RAMIS 복원은 비변형 장면이나 정적/정확한 카메라 포즈를 가정하는 경우가 많아, 조직이 비강체로 움직이고 시야가 가려지는 수술 영상에선 드리프트가 쉽게 발생한다. 또 NRSfM/변형 SLAM 계열은 카메라 포즈와 변형을 함께 추정하지만, 엔도스코프 움직임이 간헐적인 상황에서 포즈 최적화가 조직·도구 변화를 카메라 운동으로 잘못 흡수하는 실패 모드가 남아 있었다.

- **Core Contribution**: 이 논문은 수술 비디오만으로 카메라 궤적과 비변형(변형) 장면을 동시에 온라인으로 최적화하는 Track2Map을 제안한다. Simultaneous Localisation and Mapping(SLAM)처럼 작동해 포즈 priors가 없거나 노이즈가 있어도 보다 견고한 3D 복원을 목표로 한다. 또한 dense 2D point tracks를 기반으로 변형을 초기화하고, 정적 카메라 구간을 감지해 누적 드리프트를 줄이는 통합 전략을 제공한다.

- **Technical Challenges**: 핵심 난제는 제한된 시점 변화(스테레오 단기 베이스라인)에서 역렌더링이 포즈와 조직 변형을 구분하기 어렵다는 점이다. 이를 위해 Track2Map은 트래킹 흐름의 방향 분포 분산으로 카메라 움직임 가능성을 게이팅해, 카메라가 정적인 구간에서는 포즈 업데이트를 동결하고 움직임이 감지될 때만 소규모 pose increment를 반영한다. 동시에 스테레오 깊이로 2D 트랙을 3D 앵커로 들어 올려 변형 제어를 초기화·구동하고, 증분 매핑 단계에서 포토메트릭/지오메트릭 렌더링 손실과 변형 정규화를 함께 최적화한다.

- **Empirical Impact**: StereoMIS 실험에서 Track2Map은 포즈가 없는 설정부터 노이즈가 큰 설정, 깨끗한 설정까지 전반적으로 재구성 품질과 카메라 궤적 정확도를 개선했다. 특히 스테레오 입력이 메트릭 스케일을 고정하고, motion-aware pose update가 카메라 정적 구간의 드리프트를 억제해 로봇 kinematics나 사전 궤적 없이도 일관된 해부학 형상을 복원하는 데 기여한 것으로 보인다. 또한 변형 모델링, 정적 카메라 게이팅, 포즈 최적화가 상호보완적으로 작동함을 애블레이션으로 확인했으며, 공개 코드도 제공된다.



### DrugGen 2: A disease-aware language model for enhancing drug discovery (https://arxiv.org/abs/2607.08404)
Comments:
          15 pages, 2 figures, 1 table, and 4 supplementary files. To use the model, see this https URL

- **Prior Approaches**: 기존 약물 설계 생성 모델은 대체로 단일 타깃 단백질 시퀀스나 일반적인 분자 성질에만 조건을 걸어 분자를 만들었다. 그 결과 질병 상태에서 타깃이 보일 수 있는 동역학·경로 맥락을 반영하지 못해, 정밀의학 관점의 disease-conditioned 설계에는 한계가 있었다. 또한 RL 기반 최적화가 있더라도 ‘질병-타깃-약물’의 관계를 직접 조건으로 주는 경우는 드물었다.

- **Core Contribution**: DrugGen-2는 질병 온톨로지(질병 DAG, MeSH)와 타깃 단백질 시퀀스를 동시에 조건으로 하는 생성 모델로, de novo 설계와 약물 재창출 모두를 겨냥한다. 승인 약물과 그 질병·타깃 연결 정보를 정제한 데이터로 사전 GPT-2 계열을 fine-tuning하고, supervised fine-tuning(SFT) 이후 GRPO로 화학적 타당성·새로움·다양성·결합 친화도를 보상해 성능을 끌어올렸다. 특히 diabetic nephropathy 관련 5개 타깃에서 DrugGPT 및 기존 DrugGen 대비 일관된 개선을 보였다.

- **Technical Challenges**: 핵심 난제는 질병 맥락을 생성 입력으로 넣되, 모델이 유효한 SMILES를 유지하면서 동시에 탐색(uniqueness/diversity)과 결합 친화도 예측 보상을 동시에 만족시키는 것이었다. 연구진은 RDKit 기반 validity-checker로 잘못된 구조를 보상에서 제외하고, PLAPT로 예측된 결합 친화도에 더해 배치 내 반복을 막는 reward와 새로움/다양성 유도 항을 GRPO에 통합해 안정적인 최적화를 달성했다. 또한 token 길이 제약(시퀀스 768 토큰)을 고려해 MeSH DAG–시퀀스–SMILES 형태의 입력 문자열로 모델을 학습시켰다.

- **Empirical Impact**: 실험에서 DrugGen-2는 각 조건당 unique molecule 생성 수가 최대 444/500 수준으로, DrugGPT와 DrugGen에 비해 크게 높았다(유효 SMILES도 99~100%로 거의 완전). 승인 약물과의 구조적 유사도는 MeSH 변형 전반에서 더 높았고(0.70), PLAPT 기반 예측 결합 친화도 중앙값도 모든 타깃에서 DrugGPT/DrugGen을 능가했다(예: ACE, PAI-1, PPARγ, NOS3, TGF-β1). docking(특히 GLIDE XP)에서도 ACE의 경우 enalapril(–8.283)보다 낮은 스코어(–9.917, –9.485, –9.367)를 보인 후보들이 보고되며, disease-conditioned 생성이 in silico 수준에서 hit-to-lead 탐색을 가속할 수 있음을 시사한다.



### Swapping Faces, Saving Features: A Dual-Purpose Pipeline for Pedestrian Privacy in ITS (https://arxiv.org/abs/2607.08402)
- **Prior Approaches**: 자율주행/ITS용 보행자 의도·궤적 예측은 다양한 보행자 영상 데이터가 필요한데, 얼굴은 생체인식 정보라 신원 탈취·추적·딥페이크 등 보안 위험을 키운다. 기존 방법은 블러/픽셀화처럼 얼굴을 가리는 방식이 많지만 훈련에 필요한 표정·시선 등 속성을 훼손해 데이터 유용성을 떨어뜨린다. GAN 기반 익명화는 프라이버시는 확보하더라도 속성 보존이 약해 AV 학습용 얼굴 단서 활용에 한계가 있었다.

- **Core Contribution**: 본 논문은 보행자 신원을 숨기면서도 표정·고개 자세·시선 같은 핵심 얼굴 속성을 유지하도록, face swapping 중심의 5단계 파이프라인을 제안한다. 이 파이프라인은 Egy-DRiVeS처럼 이집트 거리 영상의 저해상도·다양한 복장(예: 베일) 같은 특수 케이스에 맞춰 설계됐다. 또한 비교 실험을 통해 Roop과 Ghost-v2 중 파이프라인 적용에 더 적합한 face swapper를 선정한다.

- **Technical Challenges**: 핵심 난제는 (1) 저해상도/가림/먼 거리로 얼굴 정보가 부족한 상황에서 swapping 품질을 유지하고, (2) 프라이버리(정체 은폐)와 속성 보존(시선·표정·자세)을 동시에 달성하는 것이다. 이를 위해 보행자·얼굴 검출 후 Codeformer 기반 품질 보강으로 복원 신뢰도를 높이고, 블렌딩까지 포함한 end-to-end 처리 흐름을 구성했다. 평가 지표는 랜드마크/블렌드셰이프 차이, 얼굴 임베딩 유사도, gaze vector 유사도 등으로 “얼굴 구조·표정 유지 vs 동일성 은폐”를 함께 측정한다.

- **Empirical Impact**: 고품질 얼굴 테스트에서 Roop은 Ghost-v2 대비 4개 정량 지표 중 3개에서 우수하며, 특히 표정(블렌드셰이프)·얼굴 구조/자세(랜드마크 차이) 보존이 더 잘 나타났다. 또한 Occluded face(가림된 얼굴)와 베일 여성 같은 어려운 사례에서 Ghost-v2는 비현실/윤리적으로 문제 소지가 있는 결과를 보인 반면 Roop은 더 높은 견고성을 보여 파이프라인 신뢰성을 뒷받침했다. 최종 5단계 적용 후에는 JAAD의 looking/not looking(시선 방향) 특징이 파이프라인 전후로 유지되어, 프라이버리 보호가 다운스트림 의도 예측에 필요한 얼굴 단서까지 망치지 않는다는 점을 실증했다.



### TRACE: A Two-Channel Robust Attribution Watermark via Complementary Embeddings for LLM-Agent Trajectories (https://arxiv.org/abs/2607.08400)
- **Prior Approaches**: 기존 에이전트 워터마킹은 주로 토큰 수준의 샘플링 편향(또는 비편향)에서 출발해, 최근에는 행동/도구 선택에 신호를 심는 형태로 확장됐다. Agent Guide, AgentMark, AgentWM, ActHook 등은 “한 가지 신호를 한 번” 특정 키로 실어 넣고, 파라프레이즈·로그 약식 변형·모방 학습 같은 일반적 교란에 대한 강건성을 주로 경험적으로 평가한다. 그러나 이런 방법들은 로그 증거를 공격자가 ‘직접 보유·가공(쓰기 포함)’하는 상황을 제대로 다루지 못한다.

- **Core Contribution**: TRACE는 리셀러가 궤적 로그를 마음대로 읽고(또는 수정하고) 재전달할 때도 귀속을 보장하는 에이전트 워터마크 설계를 제안한다. 한 궤적에 두 개의 독립 층을 겹쳐 담는데, 하나는 선택(어떤 행동을 고를지) 채널, 다른 하나는 카운트/스켈레톤(각 결정 그룹의 구조) 채널이다. 그 결과 TRACE는 삭제(삭제로 동기 깨짐)에도, 로그 리라이팅(내용 임의 변경)에도 각각 복원·불변성을 동시에 목표로 하며 “행동 선택은 왜곡 없이” 유지하도록 만든다.

- **Technical Challenges**: 핵심 난점은 워터마크 키를 어디에 고정하느냐인데, 삭제 공격에는 위치 기반 키가 깨지고 리라이팅 공격에는 내용 기반 키가 훼손돼 둘을 한 키로 동시에 만족시키기 어렵다는 점이다. TRACE는 selection channel은 로컬 content에 키를 묶되 distortion-free sampler(증류/편향 없이 에이전트 분포를 보존하는 방식)로 행동 분포 자체의 변형 비용을 없애고, deletion이 오더라도 검출이 재동기화되게 설계한다. 동시에 tally channel은 로그의 skeleton(태그 위치에 의해 결정되는 구조)만으로 키를 만들고, 리라이팅이 건드릴 수 없는 영역에 신호를 두어 어떤 강도의 rewriting에도 조건부가 아닌 불변성을 보장한다.

- **Empirical Impact**: ToolBench와 ALFWorld 실험에서 TRACE는 워터마크 없는 에이전트와 동일한 성공률을 유지하면서도, 긴 호라이즌에서 검출 점수 z≈100 수준에 가깝게 도달한다. step deletion 70%에서도 탐지 가능성이 유지되며, LLM rewriting(아무 강도) 하에서도 tally channel의 값은 정확히 그대로 남는다. 또한 두 채널을 동시에 지우려면 결국 리셀러가 제공하는 서비스(실행 궤적의 품질/일관성)를 망가뜨릴 정도의 비용이 발생함을 이론적으로도 뒷받침해, “증거를 쥔 공격자” 환경에서의 귀속성 강화를 실증한다.



### WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving (https://arxiv.org/abs/2607.08375)
Comments:
          20 pages, 7 figures

- **Prior Approaches**: 기존 E2E·VLA 자율주행은 입력에서 궤적을 바로 생성하는 end-to-end 방식이 주류지만, 복잡/장꼬리(long-tail) 상황에서 인과 추론과 세계 지식 부족으로 취약하다는 한계가 반복해서 지적된다. VLM을 결합한 접근은 장면 이해를 개선했으나, 세계 예측을 보조 과제로 취급하거나(semantic forecasting/image forecasting) 생성적 상호작용 예측이 분절돼 reactive driving에 머무는 경우가 많다. 또한 사회적 상황에서 필요한 게임이론적 ‘if-what’ 전략 추론을 충분히 학습시키지 못했다.

- **Core Contribution**: 이 논문은 Vision-Language-Action(VLA)에 dual-level World Cognition을 결합해 proactive driving을 목표로 하는 WCog-VLA를 제안한다. semantic 수준에서는 3D 공간 인지와 agent tokens, 그리고 Game-CoT 기반의 추론을 통해 세계 동역학을 추론하고, generative 수준에서는 ADDT(Aligned Decoupled Diffusion Transformer)로 다중 에이전트의 물리적으로 그럴듯한 공동 궤적을 생성해 ‘세계 예측→행동’의 연결을 강화한다. 결과적으로 의미론적 세계 예측과 생성적 세계 진화를 함께 다뤄 파편화된 foresight를 줄이는 것이 핵심이다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 2D 중심 VLM에 3D 공간 구조를 주입해 agent 간 기하·위치를 구조적으로 표현하고, (2) VLM의 의미론적 은닉표현을 확산 기반 생성이 요구하는 연속 궤적 세부 복원과 안정적으로 이어 붙이는 것이다. 논문은 TrackFormer 기반 agent tokens와 world head로 3D 기반 semantic world cognition을 만들고, ADDT에서는 condition encoder와 generation decoder를 분리하며 latent scene 표현과의 representation alignment(정렬 제약)로 의미-물리 간 gap을 줄인다. 전략 추론을 위해서는 Qwen3-VL-Plus로 Stackelberg game 형태의 4단계 Game-CoT(85k 주석) 데이터를 구축해 hallucination을 억제하는 GT action 힌트까지 더했다.

- **Empirical Impact**: NAVSIM v1에서 WCog-VLA는 SOTA PDMS 92.9를 달성하며, 카메라만 사용하면서도 lidar를 함께 쓰는 일부 멀티모달 베이스라인을 4.6 PDMS 앞섰다. 또한 대형 VLM 계열 대비 PDMS가 크게 개선되고, 2B급 소형 모델도 RL-refined VLM 기반 방법들을 최소 0.8 PDMS 이상, 3B 모델 LatentVLA도 0.5 PDMS 상회해 효율적인 성능 향상을 보여준다. 안전 지표에서도 NC 99.4, TTC 98.5로 강한 결과를 내며, 주변 에이전트의 미래 의도를 선제적으로 반영한 proactive 안전 운전의 효과를 실증한다.



### Large-Language-Models-as-a-Judge in Theory-Agnostic Adaptive Metric-Alignment for Prototypical Networks in Personality Recognition (https://arxiv.org/abs/2607.08374)
- **Prior Approaches**: 기존 성격 인식 연구는 Big-5, MBTI처럼 특정 심리 이론의 라벨 체계에 맞춰 학습하는 경우가 많아, 데이터셋·문화권이 바뀌면 성능이 쉽게 흔들립니다. 또한 myPersonality처럼 대규모 공개 데이터가 프라이버시 이슈로 축소되면서 저자원 환경에서의 학습 한계가 커졌고, 성격 인식을 정적 분류로만 다루는 경향도 일반화를 제한했습니다.

- **Core Contribution**: 이 논문은 이론에 의존하지 않는 성격 추론을 목표로 JAM(Judge for Adaptive Metric-Alignment)을 제안합니다. JAM은 학습 과정에서 사전 정의된 성격 이론 라벨에 고정되지 않고, 텍스트로부터 공유된 잠재 심리 구조를 ‘latent pseudo-facets’로 발견해 개인의 잠재 심리 프로필을 추론합니다.

- **Technical Challenges**: 핵심 과제는 서로 다른 이론으로 라벨링된 이질 데이터에서 공통 구조를 잘 맞추면서도 라벨 노이즈·불확실성을 견디는 것입니다. 논문은 Attention-Pooled Graph Prototypical Network로 임베딩 공간에서 클러스터 기반의 구조 표현을 만들고, Cross-Theory Harmonization(인간 가이드 linkage + 기계 유도 consensus)으로 이론 간 정합을 강화하며, LLM-as-a-Judge를 LLM-before-the-loop/LLM-in-the-loop 두 방식으로 붙여 애매한 샘플을 선별해 적응적 메트릭 학습을 돕습니다.

- **Empirical Impact**: 실험에서는 JAM이 여러 프레임워크 간 일반화와 성능에서 기존 대비 개선을 보이며, 특히 low-resource personality theory에서도 유의미한 효과를 보였다고 보고합니다. 이는 성격 인식을 특정 심리 분류 체계에 종속시키던 관행에서 벗어나, 이론-불변적 표현 학습으로 확장하는 한 걸음을 제시한다는 점에서 의미가 있습니다.



### Self-Adaptive Anomaly Detection with Reinforcement Learning and Human Feedback in Connected Vehicles (https://arxiv.org/abs/2607.08373)
Comments:
          Accepted at the 30th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA 2026), Special Session SS10: Evaluation Methods for Autonomous Cyber-Physical Systems' Behavior. 8 pages, 3 figures, 3 tables

- **Prior Approaches**: 연결 차량(CV) 같은 자율 CPS는 비정상 상태를 감지해야 하지만, OTA 업데이트·설정 변경·부하 변화로 정상의 정의가 계속 바뀌어 개념 드리프트가 발생한다. 기존 연구는 (1) 모델을 자동으로 적응시키거나 (2) 사람을 루프에 넣는 방식으로 따로 접근하는 경향이 강했는데, 둘을 함께 “감시-드리프트 판단-재학습”으로 엮는 구조는 부족했다. 특히 고정 임계값 진단이나 단일 검출기를 서비스 전반에 일괄 적용하면 시간 지나며 성능이 조용히 저하되는 문제가 컸다.

- **Core Contribution**: 이 논문은 개념 드리프트가 생겨도 성능 붕괴를 막도록, 온라인 이상 탐지·드리프트 감지·인간 피드백 재학습을 단일 감독 루프로 통합하는 프레임워크를 제안한다. 핵심은 서비스별로 후보 이상 탐지기 중 “적합한 것을 고르는” factorized deep Q-network에 self-attention을 결합하고, 드리프트 감지는 여러 통계 검출기가 동시에 동의할 때만 알람을 내며, 운영자는 검증된 피드백으로 재학습을 유도한다. 이때 재학습은 이전 분포를 유지하도록 catastrophic forgetting을 억제하는 설계로 구성된다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 미세서비스 토폴로지가 바뀌며 서비스마다 신호 분포가 달라지는 환경에서 적응이 필요하다는 점, (2) 드리프트를 오탐 없이 포착해 “재학습 타이밍”을 정확히 잡는 점, (3) 인간 피드백이 지연·국소적일 때도 학습 신호를 안정적으로 구성하는 점이다. 이를 위해 factorized DQN 구조로 서비스 추가/삭제 시 공유 인코더는 유지하고 해당 head만 다루게 했고, self-attention으로 서비스 간 의존성을 명시적으로 반영했다. 드리프트는 Page–Hinkley, Kolmogorov–Smirnov, Mahalanobis outlier-rate의 앙상블을 conjunctive rule로 결합해 정밀도를 우선했으며, pending transition buffer와 60/40 prioritized replay로 새 분포 적응과 이전 지식 보존을 동시에 노렸다.

- **Empirical Impact**: 실험은 SDVDiag 플랫폼에서 7개 백엔드 마이크서비스를 쓰는 자동 발렛 주차(AVP) 테스트베드로 수행됐다. 고정 단일 검출기를 서비스 전반에 적용하면 F1이 최대 0.11 수준에 그쳤지만, attention-augmented 에이전트(F-DQN-Attn)는 F1 0.69로 큰 폭 향상됐다. 실제 소프트웨어 업데이트 후 개념 드리프트로 F1이 0.52까지 떨어졌으나, 운영자 트리거 재학습 후 새 분포에서 0.65로 회복하면서 이전 분포 성능(0.69)을 유지해 catastrophic forgetting 없이 지속 적응을 보였다.



### On the Role of Conversational Timing in Synthetic Training Data for ASR (https://arxiv.org/abs/2607.08371)
- **Prior Approaches**: 대화 음성 모의(synthetic conversation) 데이터는 단일 화자 녹음을 조합해 multi-speaker ASR을 학습시키는 데 널리 쓰인다. 기존 파이프라인은 멈춤/겹침 같은 타이밍 통계를 실제 코퍼스와 “그럴듯하게” 맞추는 데 집중하지만, 그런 코퍼스-충실 분포가 왜 성능에 유리한지에 대한 학문적 질문엔 답이 덜 했다. 즉 realism은 달성해도, 어떤 타이밍 성질이 cpWER/cpCER를 낮추는지 체계적으로 검증하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 conversational timing을 단순히 재현해야 할 “통계”가 아니라, 학습에 미치는 영향을 통제·분석할 수 있는 훈련 변수로 재정의한다. pause와 overlap의 분포를 exponential-tilting 계열로 파라미터화하고, 그 파라미터 공간을 Latin hypercube sampling과 multi-objective Bayesian optimization으로 탐색해 ASR 에러 지표(cpWER/cpCER)와의 직접 관계를 밝힌다. 최고의 설정 찾기에 그치지 않고, overlap–gap 트레이드오프 같은 원인을 진단 가능한 형태로 드러내는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 도전은 (1) 타이밍 분포를 코퍼스 기반이면서도 통제 가능한 저차원 변수로 표현하고, (2) 매 실험마다 ASR 학습·평가가 필요해 전수 탐색이 불가능한 상황에서 샘플 효율적으로 파라미터를 골라야 한다는 점이다. 저자는 여러 대화 코퍼스를 KDE로 공통 지지대에 정렬한 뒤 exponential-tilting로 매끄러운 분포 계열을 구성하고, LHS로 초기 커버리지를 확보한 다음 Gaussian-process 기반 multi-objective BO(EHVI)로 Pareto 전선을 개선한다. 또한 overcomplete한 4차원 좌표는 “개입(intervention)” 해석을 위해 유지하되, 실제 식별은 compressed 좌표와 intrinsic timing statistics로 별도 분석해 기하적 혼선을 줄였다.

- **Empirical Impact**: Hungarian BEA-Dialogue 벤치마크에서 실험한 결과, downstream ASR 거동은 코퍼스 근접도나 시뮬레이터 좌표 원형보다 유도된 타이밍 통계가 더 직접적으로 설명한다. 특히 더 높은 overlap exposure는 더 낮은 cpWER와 연관되었고, 더 길고 변동이 큰 gap은 더 높은 cpWER를 유발했으며, cpCER도 같은 경향을 보이되 통계적 지지의 강도는 더 약했다. Bayesian optimization은 종합 성능을 “약간” 개선하는 수준이었지만, 제어된 타이밍 개입으로 overlap–gap trade-off를 분석적으로 드러냈다는 점에서 의미가 크며, 현실성(simulation realism)만으로는 부족하고 overlap·gap·timing-variability 프로파일을 진단하는 절차가 함께 필요하다는 시사점을 준다.



### FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation (https://arxiv.org/abs/2607.08359)
- **Prior Approaches**: 기존 UAV 비전-언어 항법(VLN)은 반응형 액션 예측이나 단일 autoregressive 디코딩으로 제어 명령을 바로 생성하는 방식이 많다. 이런 접근은 단기 시각-언어 편향에 취약해 장기 비행에서 궤적이 흔들리거나, 대형 추론 모델을 쓰면 추론 지연이 커져 실시간 적용이 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 FSD-VLN으로 의미 추론과 저지연 비행 제어를 구조적으로 분리하는 fast-slow dual-system 아키텍처를 제안한다. 느린(slow) 브랜치는 vision-language 모델에서 안정적인 semantic priors를 뽑고, 빠른(fast) 브랜치는 Diffusion Transformer(DiT)로 과거 행동과 시간 구조를 반영한 액션 분포를 생성해 일관된 비행 출력을 만든다.

- **Technical Challenges**: 핵심 기술 난제는 ‘전역 이해’가 요구하는 계산/지연과 ‘제어 안정성’이 요구하는 고주파 응답을 동시에 만족시키는 정렬이다. 이를 위해 느린 브랜치의 semantic feature를 버퍼로 캐시해 비동기 조율을 유지하고, DiT에서 cross-temporal action dependencies를 명시적으로 모델링해 분포 드리프트와 궤적 연속성 문제를 줄였다. 또한 long-horizon 학습에서 gradient oscillation을 완화하기 위해 time-aware adaptive optimizer(시간 가중 MSE)를 도입했다.

- **Empirical Impact**: 대규모 저고도 도시 시뮬레이션에서 FSD-VLN은 unseen 장면에서 성공률을 최대 2배 수준으로 끌어올리며(SR, SPL 개선) 내비게이션 오차도 크게 감소시켰다. 실시간성 측면에서는 1-step 추론 지연을 50% 이상 줄이고, 전체 임무 수행 시간도 50% 넘게 단축(약 53%)해 의사결정 효율과 궤적 품질을 동시에 입증했다. 결과적으로 장기 aerial VLN에서 decoupled semantic-control modeling이 실용적 성능-지연 균형을 제공한다는 점을 강하게 보여줬다.



### Spectral Analysis of Dueling Q-Learning (https://arxiv.org/abs/2607.08340)
- **Prior Approaches**: Q-learning은 전이 커널을 모르는(discounted) MDP에서 쓸 수 있는 기본 RL 알고리즘이지만, 고차원 문제에는 Q값을 딥러닝으로 근사하는 DQN이 필요하다. Dueling Q-learning은 Q를 value와 advantage로 분해해 학습 효율을 높이지만, 이 구조에 대한 이론적 이해는 표준 tabular Q-learning보다 덜 정립돼 있었다.
기존 분석은 dueling을 regularized 형태로 다루는 경우가 많았고, regularization이나 projection 없이 순수한 tabular 업데이트의 수렴을 직접적으로 설명하는 데는 공백이 남아 있었다.

- **Core Contribution**: 이 논문은 dueling Q-learning(정확히는 centered AV-learning)을 탭룰 Q-function의 작용 공통(action-common)과 작용 차분(action-differential) 성분으로 정면 해석하고, regularization과 projection을 제거한 unregularized·unprojected 상수 스텝 크기 재귀의 수렴을 보장한다. 또한 deterministic 버전에서는 정확한 switching linear system(SLS) 표현을 도출하고, stochastic(샘플링) 버전에서는 기대값 기준 finite-time error bound를 제시한다.
그 결과 value/advantage 업데이트가 Q의 두 부분공간에 서로 다른 gain으로 작동함을 해명해, 어떤 조건에서 안정적으로 수렴하는지 연결한다.

- **Technical Challenges**: 핵심 난관은 centered 분해에서 value와 advantage 성분이 결합된 업데이트를 “순수 tabular 재귀” 수준에서 안정적으로 제어하는 일이며, 이를 regularization이나 projection 없이 수행해야 한다. 논문은 이 재귀를 switching linear system으로 재구성하고, 해당 모드 패밀리의 joint spectral radius(JSR)가 1보다 작다는 조건으로 SLS의 (임의 switching 하) 지수 안정성을 증명한다.
샘플링 버전에서는 conditional-mean과 잡음 항을 분해해, JSR 조건을 drift 항의 안정성으로 이어가며 기대 오차의 유한 시간 상계를 만든다(i.i.d. 샘플링 설정에서 정리).

- **Empirical Impact**: 이 글의 실증적 기여는 주로 실험 수치가 아니라, dueling Q-learning의 수렴 메커니즘을 SLS/JSR 관점에서 정량화한 이론적 보증과 finite-time 기대 오차 상계로 나타난다. 특히 일반적인 “학습이 잘 된다” 수준을 넘어, common 성분과 differential 성분에 적용되는 gain(예: value update의 action-common gain, advantage update의 action-differential gain)이 안정성에 어떻게 관여하는지 명확히 했다.
이 분석은 앞으로 DQN-dueling의 tabular 기반 재귀가 어떤 조건에서 신뢰성 있게 수렴하는지, 그리고 더 넓은 샘플링(예: Markovian observation)으로 확장될 때 어떤 수학적 구조가 유지되는지에 대한 실무적 기준을 제공한다.



### TypeProbe: Recovering Type Representations from Hidden States of Pre-trained Code Models (https://arxiv.org/abs/2607.08339)
Comments:
          18 pages, 12 figures. Accepted at ESSLLI 2026 (StuS; double-blind)

- **Prior Approaches**: 기존 코드 모델 해석 연구는 주로 문법·식별자·네임스페이스 같은 표면 정보의 인코딩을 진단하는 데 집중해 왔다. 반면 형식 타입 의미(typed semantics)는 제대로 정조준되지 않았고, 타입 제약 디코딩 같은 외부 기법은 언어별 규칙을 추가로 필요로 한다는 한계가 있었다. 또한 다국어에서 표현이 정렬된다는 논의는 자연어 중심이었고, 코드 모델의 타입 정보가 언어를 넘어 어떻게 전이되는지는 불명확했다.

- **Core Contribution**: 이 논문은 pretrained 코드 모델의 residual stream에서 타입 표현이 선형으로 decodability 되는지(어느 레이어에 나타나는지), 그리고 Java와 Python 사이에서 교차언어로 전이되는지를 직접 파고든다. Java·타입 주석 없는 Python·타입 주석 있는 Python을 동일한 프로그램 구조로 맞춘 병렬 데이터셋과, FIM 기반 masked call site에서 타입 호환성으로 함수를 선택·결과 타입을 추론하는 프로빙 태스크를 제안한다. 특히 untyped 코드에서도 교차언어 타입 표현이 형성됨을 보이며, typed function application에서 암시되는 결과 타입까지 내부적으로 재구성되는 구조를 실험적으로 확인한다.

- **Technical Challenges**: 핵심 과제는 (1) 타입 신호와 식별자·리터럴 같은 어휘적 단서를 분리하고 (2) 선형 프로브가 실제로 decodable 정보를 잡는지, (3) 언어 문법 차이 속에서도 타입 구조가 같은 방향으로 유지되는지 검증하는 것이다. 이를 위해 입력 예시의 변수명·함수명·리터럴을 랜덤화해 lexical shortcut을 차단하고, adversarial renaming으로 식별자가 ‘거짓 타입’을 가리키게 만들어도 타입 표현이 유지되는지 Δ selectivity로 측정한다. 또한 한 언어로 학습한 프로브를 다른 언어 데이터에서 zero-shot transfer로 평가해, 타입 관련 방향이 공유되는지 확인한다.

- **Empirical Impact**: SantaCoder-1.1B와 CodeLlama-7B 모두에서 base type과 list 컨테이너에 대한 선형 타입 표현이 교차언어로 전이되며, 특히 untyped Python(pyUnt)→Java 전이가 의미 있게 나타나 타입-관련 신호가 주석 없이도 회수될 수 있음을 시사한다. 다만 adversarial renaming에 대해선 selectivity가 부분적으로만 견조하며, late-to-mid 레이어에서 타입 decodability 피크가 나타나다가 공격에서 더 크게 흔들리는 패턴이 관찰된다. 전반적으로 코드 모델 내부에 언어를 가로지르는 ‘타입 manifold’가 형성될 수 있다는 실증 근거를 제공하면서, 타입 제약 디코딩과의 보완적 방향성까지 제시한다.



### ArtMine: Discovering and Formalizing Artistic Processes (https://arxiv.org/abs/2607.08331)
Comments:
          47 pages, 10 figures

- **Prior Approaches**: 최근 생성형 AI는 이미지·텍스트·음악·영상처럼 ‘완성된 산출물’의 분포를 학습해 고충실도로 재현하는 데 집중해 왔습니다. 하지만 작품을 만들기까지의 반복적 결정, 수정, 재료 조작 같은 ‘과정(process)’은 대개 관측되지 않아 프롬프트-투-아티팩트 방식으로만 남습니다. 한편 아티스트 서신, 밑그림, 보존·감정 보고서, 소장기록 등 문서 증거는 풍부하지만, 그 신뢰도와 충돌을 계산적으로 감독(supervision)해 과정 모델로 변환한 연구는 드뭤니다.

- **Core Contribution**: 이 논문은 ArtMine으로, 다양한 역사적 증거에서 예술가의 제작 절차를 구조화해 ‘설명 가능하고 감사 가능한’ 제작 단계 시퀀스를 추론하는 프레임워크를 제안합니다. Deep Research로 증거를 개념 스키마(11차원)와 신뢰도 태그, 출처 간 충돌까지 포함한 저장소로 정리한 뒤, Peircean abductive agent가 증거에 근거한 생산 단계 궤적을 발굴합니다. 이렇게 얻은 단계는 compositional graph와 렌더링 프롬프트로 변환되고, 생성 결과를 기준 작품과 비교해 과정 자체를 다듬는 루프를 구성합니다.

- **Technical Challenges**: 핵심 난제는 (1) 완성 작품만 보고는 제작 순서를 특정할 수 없고 (2) 증거가 조각나 있으며 신뢰도·불일치가 존재하는데도 추론은 그 제약을 지켜야 한다는 점입니다. ArtMine은 증거를 direct/indirect/interpretation/speculative로 타입화하고, 해석과 사실을 섞어 임의로 메우지 않으며, 충돌은 conflicts 범주에 그대로 보존해 추론이 가정에 기대지 않도록 설계했습니다. 또한 abductive policy는 각 단계에 ‘관찰-규칙-가설’에 더해 실행 action과 evidence key를 결합해 추적성을 확보하고, self-reflection 기반의 policy 업데이트로 화면 단위 오차(구도·색·형태·배치)를 반복적으로 줄입니다.

- **Empirical Impact**: 실험에서는 WikiArt의 10개 작품을 대상으로 캐노니컬/논캐노니컬로 나눠 재구성과정을 평가했으며, 생성 품질은 CSD(스타일·붓질·팔레트), LPIPS(저수준 지각 거리), CLIP(의미 일치)로 분리해 추적했습니다. ArtMine은 동일 백본과 생성기를 쓰되 정책 학습을 생략한 프롬프팅 베이스라인(CoT, ToT, Self-Refine 등) 및 evidence 저장소 부재 조건과 비교해, 조각난 문서 증거를 기반으로 더 일관되고 해석 가능한 제작 궤적을 만들 수 있음을 보였습니다. 결과적으로 ‘완성물 재현’에서 ‘과정 중심 인간-AI 공창작’으로 초점을 옮기며, 교육·해석·감사(audit)·문화 생산 연구에 연결될 수 있는 가능성을 보여주는 proof-of-concept로 의미가 있습니다.



### GitLake: Git-for-data for the agentic lakehous (https://arxiv.org/abs/2607.08319)
Comments:
          Pre-print of the paper accepted at DASHSys, VLDB 2026, Boston, USA

- **Prior Approaches**: 기존 OLAP/레이크하우스는 단일 테이블의 스냅샷 보장은 있어도, 멀티테이블·멀티언어 파이프라인이 실패할 때 lakehouse 전역이 불완전하게 공개되는 문제(half-written pipeline)를 막는 API가 부족합니다. 또한 에이전트가 생성한 변경을 안전하게 탐색·검증·승인하는 작업 흐름이 PR 중심 Git 모델만큼 자연스럽지 못해, 롤백과 감사를 위한 재현성도 취약합니다.

- **Core Contribution**: GitLake는 Iceberg 단일 테이블의 스냅샷 진화를 lakehouse-wide commit/branch/merge 개념으로 “승격”해, 데이터도 코드처럼 버전·협업·롤백할 수 있게 합니다. 특히 run()이 임시 브랜치에서 실행한 뒤 최종 merge로 원자적으로(또는 전혀) 공개되도록 설계해, 에이전트의 파이프라인 산출물이 production 경계를 명확히 통과하게 합니다.

- **Technical Challenges**: 핵심 기술 과제는 멀티테이블 파이프라인 실패 시 전역 불일치 상태가 downstream에 퍼지는 문제를 논리적으로 봉쇄하는 것입니다. GitLake는 Iceberg의 ACID 스냅샷 보장에 기반해 커밋 메타데이터가 시점의 모든 카탈로그 테이블-스냅샷 매핑을 캡처하고, run()에서는 임시 브랜치를 열었다가 성공 시 merge로만 전파하며 실패 시 main은 부분 상태로 오염되지 않게 합니다.

- **Empirical Impact**: Bauplan 프로덕션에서는 수십만 개의 데이터 브랜치를 주 단위로 생성하면서도 copy-on-write/메타데이터 중심으로 브랜치 생성이 p95 약 80ms 수준의 “사실상 no-op”로 관측됐다고 보고합니다. 또한 Alloy로 핵심 추상화를 점검해 일관성 위배 반례(중첩 브랜치 관련)를 찾아내는 등 정확성 통찰도 제공하며, 기존 동급 기능 대비 CRUD류 연산에서 더 빠른 성능 격차와 함께 에이전트 규모의 데이터 운영 가능성을 보여줍니다.



### From Legacy Documentation to OSCAL: An MCP-Based Agent Pipeline for Threat-Informed Continuous Compliance in Critical Infrastructur (https://arxiv.org/abs/2607.08288)
Comments:
          Accepted for publication at the 2026 IEEE International Conference on Cyber Security and Resilience (IEEE CSR), Lisbon, Portugal, August 3-5, 2026. 8 pages, 1 figure

- **Prior Approaches**: 기존의 보안 자동화는 주로 RAG 방식의 문서 유사도 검색에 의존해, 사이버 위험 분석에 필요한 인과·전이 관계를 엄밀히 반영하기 어렵다는 한계가 있다. 또한 LLM이 CVE/CVSS 같은 식별자와 점수를 그럴듯하게 생성하거나, 허구의 공격 경로를 만들어내는 환각 문제 때문에 OT(operational technology) 환경에 바로 적용하기가 위험하다. MCP 생태계의 보안 연구나 에이전트형 프레임워크도 있으나, 임의의 탐색이 아니라 CTI를 결정론적으로 검증하고, OSCAL 포맷까지 바로 산출하는 통합은 부족했다.

- **Core Contribution**: 이 논문은 OT용 비침투(Active scanning 없이) 파이프라인을 제안하며, 자연어 시스템 설명을 MCP-grounded 다중 에이전트로 소스 검증 지식그래프와 OSCAL System Security Plan/ Security Assessment Report 같은 감사용 산출물로 변환한다. LLM의 추론과 결정론적 지식 검색을 분리해, 허위 CVE·합성된 공격 경로가 후단에 들어갈 확률을 낮춘 것이 핵심이다. 특히 오류를 ‘LLM 환각’에서 ‘초기 자산 추출(Phase 0)에서의 국소적 오분류’로 이동시키고, 그 결과를 사람이 빠르게 검토할 수 있게 가시화한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 자연어 자산 추출 단계에서의 엔티티/버전 오분류가 이후 결정론적 검색을 통해 ‘진짜이지만 무관한 CVE’로 이어지는 오류 전파를 막거나 통제하는 것이다. 이를 위해 15개 MCP 서버로 NVD·CISA ICS-CERT·제품/별칭 정보 등을 사전에 정의된 API 호출로 조회하고, 지식그래프 노드 수준에서 스키마 검증을 통과시키는 방식으로 ‘식별자 자체의 환각’을 억제한다. 또한 CVSS 기반의 위험도에 EPSS/KEV와 OT 자산 중요도·노출도를 결합한 CRH(critical infrastructure Relevance Heuristic)로, 안전 우선(recall 우선) 삼지를 수행해 우선순위를 자동화하면서도 가중치/구조를 투명하게 유지한다.

- **Empirical Impact**: Water utility 합성 시나리오(실증용 근거 데이터 기반)에서 CVE recall 0.90, D3FEND recall 1.00을 달성하고, MCP가 검증하는 결정론적 KG 노드에 한해 factual hallucination rate 0%를 확인했다. 다만 Phase 0의 의미적 오분류로 인한 contextual false positive rate가 8.5%로 관측되었는데, 이는 ‘존재하지만 없는 자산에 대한 CVE’가 후단 SAR에 일부 포함된 결과로 해석된다. 그럼에도 OSCAL SSP/SAR가 NIST OSCAL JSON 스키마를 만족해 감사 워크플로에 바로 붙일 수 있다는 점에서, OT/인프라 규제 컴플라이언스를 시간 효율적으로 자동화할 실무적 의미가 크다.



### Multi-Agent Firewall Architecture for Privacy Protection of Sensitive Data in Interactions with Language Models (https://arxiv.org/abs/2607.08282)
- **Prior Approaches**: 기존 LLM 보안은 (1) SDK처럼 앱 수정으로 프롬프트를 검사하거나, (2) 외부 중앙형 안전검증 서비스에 프롬프트를 보내는 방식, (3) 사내 전용 모델로 클라우드 전송을 차단하거나, (4) 로컬에서 사용자가 키워드를 수동/전처리로 바꾸는 방식이 주류였다. 이들은 주로 ‘가로채기 공백’(interception gap), ‘프라이버시 역설’(검증을 위해 외부 벤더에 데이터 전송), 또는 ‘지능 격차’(오픈 가중치 성능 한계) 문제를 남겼다. 또한 토큰 단위 탐지는 가능해도, 문맥 전체가 의미하는 맥락적 유출 위험(contextual exfiltration)까지 안정적으로 다루기 어렵다는 한계가 컸다.

- **Core Contribution**: 이 논문은 오픈소스이면서 개인정보 중심의 로컬-first “LLM 데이터 누출 방지 방화벽”을 제안한다. 브라우저 확장 + 투명 MiTM 프록시를 결합해 웹 기반 상호작용과 HTTP(S)/WebSocket API 트래픽을 함께 가로채며, 결정적 탐지와 LLM 기반 의미 분석을 하이브리드로 묶어 누출을 막는다. Git 코드 유출 방지(저장소 인덱싱 + 퍼지 매칭)와 위험도 기반 정책(경고/차단/자동 마스킹/사용자 override)까지 포함해 조직 환경에서 커스터마이즈 가능하도록 설계했다.

- **Technical Challenges**: 핵심 기술 과제는 ① 애플리케이션 수정 없이 사용자 입력 경로를 정확히 가로채는 것, ② 문맥 기반 의미 유출을 낮은 지연/비용으로 잡아내는 것, ③ 이미지·파일 등 비정형 입력에서 재현 가능한 추출 파이프라인을 만드는 것이었다. 저자들은 센서(확장/프록시)와 분석 엔진을 분리하고, DAG 기반 멀티에이전트 파이프라인으로 저비용 탐지(정규식/키워드/체크섬/NER/코드 유사성)를 먼저 돌린 뒤 필요할 때만 LLM 의미 분석·VLM 추출로 단계적 에스컬레이션을 수행해 지연을 통제한다. 또한 OCR은 Tesseract의 confidence 기반 라우팅으로 처리하고 실패 시 VLM으로 이중 계층 추출을 수행하며, 파이프라인 토폴로지와 탐지 규칙을 JSON으로 외부화해 배포 후 정책 변경을 쉽게 했다.

- **Empirical Impact**: 평가 결과, 최적 구성에서 최대 F1 94.93%까지 성능을 달성해 결정적 탐지+의미 분석 하이브리드의 효과를 보여줬다. 계층형(early block) 구조와 공급자 비의존(provider agnostic) 설계를 통해 웹/프로그램 환경 모두에서 차단 정확도와 운영 비용·지연 간 균형을 맞출 수 있음을 강조한다. 보안 연구 관점에서는 OWASP LLM01/LLM02 맥락의 ‘사용자-LLM 상호작용 데이터 누출’ 문제를 실사용 가능한 형태로 구현했다는 점에서, 조직 내 배포형 LLM 보안 플랫폼 방향에 영향을 줄 것으로 보인다.



### Best-of-$N$ TTS Evaluation is Confounded by ASR Family Alignmen (https://arxiv.org/abs/2607.08256)
Comments:
          Accepted at ICML 2026 Workshop on Machine Learning for Audio

- **Prior Approaches**: flow-matching 기반 zero-shot TTS는 자연스러움과 화자 유사성은 크게 개선됐지만, 여전히 일부 발화에서 단어 수준 내용 오류가 남아 있다. 이를 줄이기 위해 BoN(Best-of-N) 추론처럼 여러 후보를 만든 뒤 ASR verifier로 텍스트 일치도를 기준 삼아 하나를 고르는 방식이 표준으로 쓰인다. 다만 어떤 ASR verifier(예: Whisper, wav2vec 2.0, HuBERT 계열)를 쓰는지에 대한 평가는 대체로 단일 고정 ASR에 의존해 체계적으로 점검되지 않았다.

- **Core Contribution**: 이 논문은 BoN에서 핵심적인 “검증기(ASR verifier) 랭킹”이 verifier의 선택이 아니라 “평가에 쓰는 ASR 계열(evaluator)”에 따라 뒤집힐 수 있다는 평가 교란을 규명한다. 특히 동일한 생성 결과라도 Whisper 계열과 wav2vec 2.0/HuBERT 계열 평가자 사이에서 정렬 방향이 반대로 나타나, 같은 verifier를 써도 보고되는 WER이 달라질 수 있음을 보여준다. 이를 바탕으로 서로 다른 ASR 계열을 함께 고려하는 cross-family rank ensembles(순위 평균, 교집합 형태 max-rank)를 제안한다.

- **Technical Challenges**: 관찰된 교란이 “표상 유사도(예: audio encoder 표현의 CKA)” 때문인지 확인하는 것이 큰 기술 과제였는데, 선형 CKA가 높아도 WER 랭킹이 반드시 일치하지 않는 패턴이 나타났다. 대신 verifier와 evaluator가 같은 계열/계보(lineage)일 때 선택이 과대평가되는 identity- 또는 lineage-level coupling 가능성을 시사하며, 이를 방지하려면 후보 선택 단계에서 계열을 분산시켜야 한다. 저자들은 (w2v2-base, distil-v3)처럼 서로 다른 ASR 계열 verifier를 고정하고, rank-avg와 max-rank로 교차 계열 집계를 수행해 평가자 의존성을 완화했다.

- **Empirical Impact**: LibriSpeech-PC test-clean에서 N=10일 때 cross-family rank ensembles은 평균 WER을 1.61%까지 낮추며, F5-TTS 기준으로 -12% 상대 개선을 보였다. 또한 공식 F5-TTS evaluator만 봤을 때도 최선의 단일 same-family verifier가 WER을 2.06%에서 1.72%로 -16.5% 낮추는 등 계열 선택 효과가 뚜렷했다. SIM-o(화자 유사도)와 UTMOS(자연도)는 구성 전반에서 거의 변하지 않아, 선택 편향을 줄이면서도 품질 저하 없이 성능을 개선할 수 있음을 실험적으로 뒷받침하며 “최소 2개 이상의 disjoint ASR 계열로 교차 검증”을 기본 보고 관행으로 권고한다.



### RhyMix: A Lightweight Adaptive Multi-Rhythm Network for Long-Term Time Series Forecasting (https://arxiv.org/abs/2607.08234)
Comments:
          38 Pages

- **Prior Approaches**: 기존 장기 시계열 예측은 Transformer처럼 긴 의존성을 보려면 국소 변동을 과도하게 매끈하게 만들거나, CNN은 수용영역 한계로 스케일을 고정하는 문제가 있었다. DLinear·TimesNet·PatchTST·iTransformer 등은 성능을 높였지만 단일한 시간 패턴/결합 규칙에 의존해 입력마다 최적 조합이 달라지는 현실을 충분히 반영하지 못했다. 또한 Time-o1 같은 일부 방법은 정확도를 얻는 대신 SVD 사전 계산과 베이스 모델 의존으로 배포 제약이 생겼다.

- **Core Contribution**: 논문은 RhyMix(RHYthm MIXture)라는 가벼운 하이브리드 아키텍처를 제안하며, 병렬 듀얼 패스와 다단(adaptive) 게이팅으로 샘플별·채널별로 패턴 조합을 자동 조절한다. 한 패스는 학습 가능한 cyclic embedding으로 계절/주기성을 명시적으로 모델링하고, 다른 패스는 multi-scale depthwise dilated convolution과 channel attention으로 서로 다른 수용영역의 시간 의존성을 포착한다. 두 패스의 출력을 결합할 때 4개 전용 forecasting head(Direct, Trend-Seasonal Decomposition, Local Convolution, Periodic Fusion)를 동적으로 혼합해 입력 특성에 따라 “어떤 패턴을 얼마나 쓸지”를 결정한다.

- **Technical Challenges**: 핵심 난제는 (1) 단기 변동·장기 추세·반복 주기·불규칙 변화가 동시에 나타나는 비정상성을 모델 구조에 반영하면서, (2) 학습/추론 복잡도를 horizon·채널 수에 대해 선형에 가깝게 유지하는 것이었다. RhyMix는 RevIN 정규화로 절대값보다 상대적 패턴을 학습하게 하고, 게이팅용 통계 특징(평균/분산/마지막 값/기울기/연속 차분 평균절댓값)을 뽑아 다단 게이팅의 입력으로 사용한다. 또한 multi-scale dilations(d=1,2,4,8)과 미리 정한 다중 주기(예: 12/24/48/168)를 결합해 다중 스케일 적합성을 확보하면서도, 파라미터와 지연을 과도하게 늘리지 않도록 설계를 고정했다.

- **Empirical Impact**: 12개 실제 데이터셋 장기 예측 벤치마크에서 RhyMix는 10개에서 SOTA 성능을 달성하며, Time-o1 등 경쟁 모델 대비 적은 파라미터와 사전 계산 없이 경쟁력을 보였다. 경량성 측면에서도 약 40K 파라미터 규모(발표 수치 40,269)와 5ms 미만 추론 지연(<5ms), 작은 모델 크기 및 낮은 메모리 사용량(모델 157KB, 추론 메모리 9.35MB)을 보고해 엣지/실시간 배포에 초점을 둔 가치를 강조한다. 즉, “다중 시간 패턴 적응”과 “실용적 지연·복잡도”를 동시에 노린 설계가 실험적으로 확인된 셈이다.



### TMI: Text-to-Image Meets Image-to-Image for Complementary Data Synthesis to Boost Long-Tailed Instance Segmentation (https://arxiv.org/abs/2607.08201)
Comments:
          Accepted to ECCV 2026. The first two authors contributed equally to this work

- **Prior Approaches**: 대규모 어휘 instance segmentation(LVIS 등)은 장꼬리(long-tailed) 분포로 인해 희귀 범주에서 데이터 부족과 미세한 클래스 간 모호성이 동시에 발생한다. 이를 완화하려는 re-weighting, balanced sampling, classifier calibration 같은 방법은 분포 불균형은 줄이지만 희귀 범주의 ‘실제 데이터 희소성’ 자체를 해결하긴 어렵다. 생성 기반합성은 T2I, L2I, I2I로 나뉘는데, T2I는 pseudo-label 잡음(특히 희귀/세분 범주) 문제가, L2I는 마스크-이미지 불일치 및 대규모 범주 확장 한계가, I2I는 copy-paste의 맥락 부자연스러움(도메인 갭)이나 inpainting의 자연스러운 배치 난점이 있다.

- **Core Contribution**: 논문은 T2I 생성의 범주·장면 다양성과 I2I 편집의 맥락 현실감을 결합한 하이브리드 합성 파이프라인을 제안한다. T2I에서는 프롬프트에 포함된 범주만 남기는 prompt-consistent filtering과 teacher-student로 pseudo-label 신뢰도를 높이고, 희귀 범주 강화를 위해 VRAIN(Verified Rare-class Augmentation via INstructed editing)이라는 새로운 I2I 에디터를 도입한다. VRAIN은 ‘place-and-verify’ 방식으로, instruction 기반 편집으로 자연스러운 위치에 희귀 인스턴스를 삽입한 뒤 의미 일관성과 시각적 충실도를 검증해 정확한 인스턴스 단위 주석을 생성한다.

- **Technical Challenges**: 핵심 난제는 (1) 생성 도메인 갭으로 인한 pseudo-label 오염과 (2) 희귀 범주에 대해 자연스러운 맥락 통합 및 정확한 배치·주석을 동시에 달성하는 문제다. 이를 위해 T2I 라벨은 offline 텍스트 일관성 필터링 후 teacher의 EMA 업데이트로 온라인에서 점진적으로 정제하며, prompt에 맞지 않는 예측은 분류 손실에서 제외하되 localization 손실에는 일부 활용해 학습 신호를 유지한다. I2I 라인에서는 VLM 기반으로 적합한 희귀 범주와 배치 instruction을 선정하고, SSIM 차이로 편집 영역을 찾은 뒤 open-vocabulary 검출 후보를 VLM으로 ‘박스 기반 yes/no’ 검증하여 확인된 인스턴스만 SAM으로 마스크를 추출·합성한다.

- **Empirical Impact**: LVIS에서 기존 paste 기반 증강과 T2I 베이스라인을 넘어, 전체 AP는 최대 +4.0, 희귀 범주 AP는 최대 +9.5만큼 개선되는 성능을 보고한다. 또한 백본이 커질 때 효과가 안정적으로 확장되어(스케일링) 데이터 합성의 실용성을 시사한다. 결과적으로 희귀 범주 성능 저하의 주요 원인으로 지목된 라벨 신뢰도·맥락 부자연스러움을 함께 완화해, 장꼬리 어휘 instance segmentation의 학습 효율을 실증적으로 끌어올렸다는 점에서 의미가 있다.



### Open-ended Multi-agent Autocurricula via Visual Inspection of Policies with Multi-modal LLMs (https://arxiv.org/abs/2607.08193)
- **Prior Approaches**: 기존 오픈엔디드 커리큘럼은 에이전트의 현재 학습 진도를 성능 지표(Scalar task score)나 행동을 글로 요약한 정보로 추정해 다음 과제를 고르는 방식이 많았다. 하지만 보상 신호가 희소·기만적인 경우 스칼라 점수는 성공에 가까운 정책의 미묘한 단서를 놓칠 수 있고, 텍스트 요약만으로는 시각적으로 분명한 진행 신호를 충분히 반영하기 어렵다. 또 다중 에이전트/가변 에이전트 환경에서는 이런 집계 방식이 정책의 실제 “모양”을 덜 보여주는 한계가 있다.

- **Core Contribution**: 이 논문은 정책의 에피소드 비디오를 직접 “시각적으로 점검”해 다음 커리큘럼 과제를 추천하는 Visual Inspection of Policies (VIP)를 제안한다. VIP는 Video Language Model(VLM)이 비디오와 최소한의 수치(예: 승률)만 보고 학습에 도움이 되는 과제를 고르며, 텍스트 요약 기반 방법보다 정책의 진행 단서를 더 잘 포착하도록 설계됐다. 특히 VIP는 비디오에 함께 렌더링되는 한 다중 에이전트 수에도 비교적 무관하게 적용 가능하다고 주장한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 비디오에서 학습 진도를 읽어내기, (2) VLM의 추천이 과제 포맷/유효성 규칙을 어기는 “환각”을 포함할 수 있는 문제를 다루는 것이다. 논문은 VideoLLaMa2-7B 같은 경량 VLM을 쓰고, VLM의 텍스트 출력에 대해 sentence similarity 기반 Sanitize-Task 모듈로 가능한 과제 후보 중 가장 잘 맞는 항목만 선택해 유효한 과제 스펙을 복구한다. 또한 VLM 추론이 전체 학습 계산에서 병목이 되지 않도록 커리큘럼 단계에서 필요한 비디오·프롬프트 구성을 최소화한다.

- **Empirical Impact**: 실험은 SMAC(StarCraft Multi-Agent Challenge)에서 VIP를 텍스트 전용 애블레이션, 그리고 스칼라 task score에 의존하는 Robust Prioritized Level Replay 변형들과 비교해 수행했다. 결과적으로 VideoLLaMa2-7B 수준의 경량 VLM을 써도 VIP는 텍스트 요약만 쓰는 경우보다 더 효과적인 커리큘럼을 만들었고, 스칼라 점수 기반 방법들보다도 일반화 성능이 좋게 나타났다. 즉, 정책의 시각적 동작을 직접 반영하는 접근이 오픈엔디드 커리큘럼의 “난이도-진도 정렬”에 실질적인 이득을 준다는 점을 경험적으로 보여준다.



### Leveraging Color Naming for Image Enhancemen (https://arxiv.org/abs/2607.08185)
Comments:
          Project page: this https URL. arXiv admin note: text overlap with arXiv:2407.09892

- **Prior Approaches**: 기존 학습 기반 이미지 보정은 원시(raw)–전문가 편집 쌍 데이터를 사용해 스타일을 모사하지만, 결과가 개인 취향이나 사용 맥락과 어긋날 수 있습니다. 또한 다중 3D LUT나 end-to-end U-Net 계열은 해석 가능성과 사용자 조절(인터랙션)이 제한적이라는 지적이 있습니다. 톤 커브로 어느 정도 설명성을 주는 방법도 색/공간을 섞거나 결합하는 방식 때문에 “색 이름 단위로 무엇이 어떻게 바뀌는지”가 불명확해지기 쉽습니다.

- **Core Contribution**: NamedCurves+는 Color Naming(색 이름) 분해를 활용해 이미지를 6개 색 맵으로 나누고, 각 색 맵에 대해 Bezier-parametrized tone curve를 적용하는 프레임워크를 제안합니다. 전역(색 이름별) 톤 조정은 사용자가 특정 색의 커브를 직접 수정해 원하는 결과로 유도할 수 있게 만들며, 이를 통해 해석 가능성과 인터랙션을 동시에 강화합니다. 여기에 트랜스포머 기반 fusion으로 6개 전역 편집을 결합해 전문가의 스타일에 가까운 국소·공간 의존 편집 효과까지 재현하려고 합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 카메라/조명 차이로 인해 색 이름 분해가 흔들릴 수 있다는 점과 (2) 전역 커브 편집만으로는 전문가가 하는 국소 조절을 충분히 만들기 어렵다는 점입니다. NamedCurves+는 UNet-like backbone과 attention(CBAM)으로 입력을 canonical 공간으로 표준화해 색 분해의 일관성을 확보하고, 전역 커브 6장을 만든 뒤 transposed attention 기반 트랜스포머 fusion으로 공간적 의존성을 모델링합니다. 또 기존 fusion의 색 경계 halo 같은 취약점을 줄이기 위해 채널 간 교차(cross-covariance) 형태의 transposed attention과 효율적인 피처 처리 설계를 사용합니다.

- **Empirical Impact**: MIT-Adobe-5K, PPR10K, MSEC, SICE의 image retouching, tone mapping, exposure correction 전 범위에서 NamedCurves+가 state-of-the-art를 정량·정성적으로 앞선다고 보고됩니다. 특히 tone curve가 색 이름별로 명시적으로 표현되어 어떤 색이 어떻게 조정되었는지 추적이 가능하고, 사용자가 커브를 수정해 개인화된 결과를 얻을 수 있다는 점을 실험 맥락에서 강조합니다. 결과적으로 ‘설명 가능 + 사용자 제어 + 고성능’의 조합을 제공하는 범용 이미지 enhancement 프레임워크로서 의미가 큽니다.



### LEEVLA: Seeing What Matters in Latent Environment Evolution for Vision-Language-Action (https://arxiv.org/abs/2607.08182)
- **Prior Approaches**: 기존 Vision-language-action(VLA) 모델은 시각 토큰을 대체로 동일하게 다루고, 인간이 고른 요인에 의존해 추론하는 경우가 많아 동적·복잡한 상황에서 핵심 근거를 충분히 강조하지 못한다. 그 결과 작업에 결정적인 증거는 놓치고, 불필요한 배경 요인에 영향을 받는 문제가 반복된다는 점이 한계로 지적된다.

- **Core Contribution**: 이 논문은 LEEVLA로, 잠재 세계표현의 구조적 진화를 유지하면서 “무엇을 봐야 하는지”를 명시적으로 유도하는 VLA 아키텍처를 제안한다. 또한 훈련 단계에서 작업 증거에 집중하도록 장치(DGDP)와, 그 증거가 잠재공간에서 어떻게 “진화”해야 하는지 장치(SFFG)를 결합해 task-aware where-how 학습 프레임워크를 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작업의 지시와 관련된 시각 영역을 학습 중 어디에 우선 집중할지 정하고, (2) 그 집중된 특징들이 잠재공간에서 일관된 방식으로 시간에 따라 진화하도록 구조를 강제하는 것이다. 이를 위해 drift-guided dynamic prioritization(DGDP)은 dynamic position prioritization(DPP)과 semantic drift guidance(SDG)를 결합해 주의 위치를 안내하고, structured feature flow generation(SFFG)은 prototype-to-periphery(P2P) 예측으로 우선 특징의 잠재 진화 경로를 모델링한 뒤 mutual-neighborhood contrastive(MC)로 이웃(topology) 일관성을 유지한다.

- **Empirical Impact**: VLA 벤치마크 전반에서 LEEVLA는 기존 방법을 일관되게 능가하며, 작업 증거를 명시적으로 유도하는 접근과 구조화된 잠재 추론이 확장 가능한 VLA에 필수라는 결론을 뒷받침한다. 특히 dynamic 시나리오에서의 강건성이 강화된 점이 실증적으로 확인되어, VLA 연구에서 “where-to-attend”와 “how-to-evolve”를 함께 설계하는 방향에 의미 있는 자극을 준다.



### Out of Sight: Compression-Aware Content Protection against Agentic Crawlers (https://arxiv.org/abs/2607.08180)
- **Prior Approaches**: 기존 콘텐츠 보호는 robots.txt, Cloudflare 차단, 라이선스 프로토콜처럼 ‘접근 통제’에 의존하지만, LLM 에이전트는 일반 브라우저처럼 요청을 보내며 이를 쉽게 우회할 수 있다. 또한 prompt injection·jailbreaking 같은 에이전트 공격은 출력만 교란하는 경우가 많아, 원문 텍스트의 재사용 가능성 자체를 막는 데는 한계가 있다. 문서 워터마킹/학습 중 오염 등도 인퍼런스 단계에서 압축으로 넘어가는 흐름에는 직접 개입하기 어렵다.

- **Core Contribution**: 이 논문은 에이전트 파이프라인에서 필수적으로 수행되는 ‘context compression(문맥 압축)’을 새로운 방어 레이어로 정의하고, 그 단계에서 의미 무결성을 망가뜨리는 접근을 제안한다. CAPE는 사람이 보는 표면 형태는 그대로 유지하면서, 압축 처리 시 심각한 정보 손실이 발생하도록 보이지 않는 perturbation을 주입한다. 즉, 기존의 경계(접근 통제)보다 안쪽(콘텐츠 레벨)에서 방어가 작동하도록 설계를 바꾼다.

- **Technical Challenges**: 핵심 난제는 (1) 사람이 보기엔 동일하게 유지하면서 (2) 블랙박스 압축기에서도 전이 가능한 방식으로 (3) 제한된 질의(query) 예산 내에서 정보 손실을 극대화해야 한다는 점이다. CAPE는 접근 가능한 surrogate compressor로 seed perturbation을 찾고 구조적 priors를 추출한 뒤, prior-guided evolutionary adaptation과 preference-calibrated query selection을 통해 타깃 압축기에 맞게 저비용 적응을 수행한다. 또한 사용자 입력과의 시각적 차이를 최소화하는 다목적 제약을 함께 최적화한다.

- **Empirical Impact**: 장문, 코드, 대화 이력의 3개 콘텐츠 유형과 여러 압축 설정에서 CAPE는 강력한 기준선 대비 정보 손실을 최대 75.8%까지 개선하면서도 입력은 시각적으로 동일에 가깝게 유지했다. 더 나아가 LangGraph 에이전트 워크플로와 GitHub Copilot 같은 현실 파이프라인으로도 전이되어, downstream 유틸리티 성능이 최대 59.7% 수준으로 하락하는 결과를 보였다. 저자들은 공개 프로토타입과 평가 자료를 제공해 에이전트 시대 콘텐츠 보호 연구의 방향성을 제시한다.



### ProsMAE: Multi-Source MAE Pretraining for ISUP Grade Classification (https://arxiv.org/abs/2607.08162)
Comments:
          Accepted to APCCAS 2026

- **Prior Approaches**: WSI는 진단에 필요한 세부 형태 정보를 담지만, 기가픽셀 규모와 스캐너/염색 차이, 조직 아티팩트 때문에 end-to-end 학습보다 타일 단위 전처리와 self-supervised 사전학습이 주로 활용된다. 기존에는 single-source MAE나 표준 AE/VAE 같은 재구성 기반 방법이 많이 쓰였고, 도메인 변이가 있을 때 표현이 특정 데이터셋 편향을 보일 수 있다는 한계가 남았다. 또한 ISUP 그레이딩은 ordinal 성격이라 성능 평가가 단순 정확도보다 불일치 정도를 반영하는 지표가 더 중요하지만, 분할 구성에 따른 재현성 문제도 제기돼 왔다.

- **Core Contribution**: 이 논문은 여러 출처의 병리 타일로 MAE를 학습하는 ProsMAE와, 학습된 인코더를 고정한 채 선형 헤드로 ISUP grade를 예측하는 ProsCLS를 제안한다. ProsMAE는 PANDA(전립선), CAMELYON17(림프절 전이), BRACS(유방 아형) 타일을 함께 사용해 서로 다른 조직 형태와 획득 조건을 인코더가 보게 만든다. 그 결과 disjoint PANDA split에서 vanilla MAE frozen linear-probe 대비 평균 validation QWK를 개선했다.

- **Technical Challenges**: 다중 임상 환경에서 안정적인 형태 표현을 얻으려면 스캐너/염색/블러/압축 같은 잡음과 준비 과정 차이에 덜 민감한 표현 학습이 핵심 과제다. ProsMAE는 masking 기반 재구성으로 학습 신호를 만들되, 본 메인 설정에서는 Gaussian noise 주입 없이 mask ratio 0.75로 학습해 도메인 변이에 대한 강인성을 확보하려 했다. 또한 디코더를 제거하고 encoder를 그대로 ProsCLS에 옮긴 frozen linear evaluation 설계를 통해 저연산 다운스트림 적용성을 유지했다.

- **Empirical Impact**: 재구성 품질 평가에서 ProsMAE는 LPIPS, SSIM, PSNR 기준으로 단일/기존 AE·VAE·single-source MAE 대비 우수한 성능을 보였다. 다운스트림 ISUP 등급 분류에서는 vanilla MAE의 평균 QWK 0.4084보다 ProsMAE가 0.4736으로 더 높은 ordinal 일치도를 보였고, primary disjoint split에서 QWK가 0.0652p 절대 개선됐다. 다만 검증은 단일 PANDA 코호트의 primary split 중심이며 seed/분할 구성 변동이 남아, 향후 반복 스플릿과 독립 전립선 코호트에서의 일반화 확인이 필요하다고 밝혔다.



### LEXIC: Lightweight Eye-tracking eXtension via Injected Complexity (https://arxiv.org/abs/2607.08152)
Comments:
          Accepted to APCCAS 2026

- **Prior Approaches**: EyeBench의 읽기 이해(이진 분류)에서 텍스트를 함께 쓰는 모델은 AUROC 56~63%까지 올라가지만, gaze-only 모델은 거의 우연 수준에 머뭅니다. BEyeLSTM처럼 언어 정보를 추가하되 language model을 쓰지 않는 방식은 중간 성능을 보이나, 여전히 본문 텍스트와의 직접 상호작용 신호가 제한적입니다.
또한 텍스트-어웨어 모델은 성능은 높지만 PLM 추론 비용이 커서, 배포 제약이 있는 상황에서 “gaze-only를 얼마나 끌어올릴 수 있는지”가 남은 질문입니다.

- **Core Contribution**: 본 논문은 inference 시 language-model forward pass 없이, 미리 계산된 단어 수준 난이도 신호로 gaze-only 모델을 보강하는 두 가지 경량 주입 메커니즘(LEXIC-Concat, LEXIC-Res)을 제안합니다. GPT-2 surprisal, word frequency, word length의 세 신호를 fixation 입력에 투입해 텍스트 측 신호의 일부를 전달합니다.
그 결과, OneStop에서 두 메커니즘 모두 Unseen Text 기준 AUROC를 유의하게 개선하며, 특히 LEXIC-Concat은 Unseen Reader에서도 추가 상승을 보입니다.

- **Technical Challenges**: 핵심 기술 과제는 “gaze-only”라는 입력 제약 하에서 텍스트 난이도 신호를 모델이 유효하게 활용하도록 결합하는 것입니다. 저자들은 (1) 세 난이도 신호를 단순 채널로 추가하는 LEXIC-Concat과, (2) 난이도로 typical-reader gaze response를 예측하고 관측 gaze와의 잔차를 입력에 반영하는 LEXIC-Res를 설계해 비교합니다.
또한 잔차 주입의 residual head가 훈련 독자(population-averaged calibration)에 맞춰져 있을 때 out-of-distribution 독자에게 전이 성능이 약해지는 구조적 경계를 실험적으로 확인하고, 이로 인해 LEXIC-Res는 Unseen Reader에서 이득이 감소합니다.

- **Empirical Impact**: OneStop 평가에서 10-fold 교차검증과 K=5 seed-ensemble(총 10개 폴드)로 AUROC 개선이 일관되게 관측됩니다. Unseen Text에서 LEXIC-Concat과 LEXIC-Res는 각각 +1.8~+2.2%p 수준의 통계적으로 유의한 상승을 보이며, LEXIC-Res 역시 Unseen Reader에서 +1.8%p 수준이지만 유의성은 약합니다.
반면 LEXIC-Concat은 Unseen Reader에서 +2.9%p로 유의하게 향상되어, “언어모델 없이도 gaze 기반 읽기 이해 예측의 성능을 올릴 수 있다”는 실용적 방향을 제시합니다. 이는 EyeBench 리더보드의 PLM 의존 격차를 경량 특징 주입으로 일부 메울 수 있음을 보여준다는 점에서 의미가 큽니다.



### Prismata: Confining Cross-Site Prompt Injection in Web Agents (https://arxiv.org/abs/2607.08147)
- **Prior Approaches**: 기존 연구는 프롬프트 인젝션에 대해 모델 수준(LLM 저항, 추론 보정)과 시스템 수준(권한 제약, 에이전트 설계 보안)으로 나뉘어 접근해 왔다. 하지만 적응형 공격이 모델 방어를 높은 성공률로 우회하고, 시스템 방어는 origin 단위 제어나 사전 계획/수동 사이트 맵 같은 확장성 한계가 남아 있다.

- **Core Contribution**: 이 논문은 웹 에이전트의 Cross-Site Prompting(XSP) 위험을 줄이기 위해 Prismata를 제안한다. Prismata는 문서 구조와 태스크 맥락을 결합해 페이지 콘텐츠에 대한 contextual least privilege 권한 레이블을 동적으로 만들고, 그 레이블에 기반해 무엇을 “보는지”와 “할 수 있는지”를 함께 제한한다.

- **Technical Challenges**: 핵심 난제는 보안 정책을 만들기 위해 페이지 구조를 읽는 순간, 그 구조가 공격자 콘텐츠와 얽혀(labeling 단계가 오염될 수 있음) 방어가 역으로 조작될 수 있다는 “web entanglement problem”이다. Prismata는 (1) Document Object Model(DOM)에서 각 인터랙티브 요소까지의 critical path를 기반으로 태스크 범위와 신뢰 경계를 분리해 주고, (2) Biba 스타일 no-read-down/no-write-up로 경계 신호를 선제적으로 반영하며, (3) 기계적 confinement으로 레이블을 관측/행동 양쪽에서 강제한다.

- **Empirical Impact**: WebArena의 여러 공격 템플릿과 WASP 및 적응형 스트레스 테스트에서, Prismata는 공격 성공률을 평균 85.5%에서 0.7%로 크게 낮추면서 정상 태스크 완료율도 4.5%→23.0%로 개선했다. 또한 방어를 적용해도 유의미한 성능 저하가 제한적이며, 동적 신뢰 레이블이 사람 판단과도 잘 정렬된다는 검증 결과를 제시해 실사용 확장성(개발자 주석 불필요)까지 강조한다.



### ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR Post-Correction for Historical Documents (https://arxiv.org/abs/2607.08143)
Comments:
          17 pages

- **Prior Approaches**: 기존 OCR post-correction 연구는 규칙 기반, 통계 모델, 시퀀스-투-시퀀스 신경망 등으로 이어졌지만, 언어·문서 유형·잡음 특성에 따라 성능이 크게 흔들려 왔습니다. LLM을 활용한 시도도 zero-shot 프롬프트부터 fine-tuning, 탐지-생성 하이브리드까지 다양했지만 실험 설정이 달라 직접 비교가 어렵고, 무엇보다 hallucination(그럴듯한 내용 덧붙이기) 위험이 충분히 정리되지 않았습니다.

- **Core Contribution**: HIPE-OCRepair-2026은 LLM-assisted OCR post-correction을 ICDAR 경진대회 형태로 재정의하고, 재현 가능한 평가 체계를 제공합니다. 또한 언어(영어·프랑스어·독일어)와 시대(17~20세기), 문서 유형(신문·인쇄물)에 걸친 통합 멀티링구얼 벤치마크(HIPE-OCRepair-2026 dataset)를 제공해, 서로 다른 기존 데이터셋을 통일된 분할·전사 지침으로 묶었습니다. 평가 관점도 문자 생성 그 자체보다 검색·접근에 유리한 언어적 정확도를 우선하는 retrieval-oriented 점수 체계를 채택했습니다.

- **Technical Challenges**: 핵심 난제는 이미지 없이 OCR 텍스트만 주어졌을 때 심각한 왜곡을 복원해야 하며, 동시에 원문에 없는 내용을 새로 만들지 말아야 한다는 점입니다. 논문은 이를 해결하기 위해 cMER 기반 정량 지표(삽입이 분모에 반영되어 과생성에 덜 민감)와 항목 단위 선호 점수(pref_score)를 함께 사용하고, 레이아웃 정규화·IR 스타일 정규화로 검색 시나리오에 맞춘 공정한 비교를 구성했습니다. 참가 팀들은 보수적 온도, 출력 길이/형식 제약, 문서 메타데이터 활용, judge-and-retry 같은 재시도 루프 등으로 hallucination과 과교정을 제어하려 했습니다.

- **Empirical Impact**: 실험 결과 modern LLM-assisted 시스템은 전반적으로 OCR 품질을 유의미하게 개선하지만, 데이터셋·언어·잡음 수준에 따라 격차가 크며 특히 저잡음 입력에서 over-correction이 반복되는 문제가 관찰됐습니다. 최고 성능은 BnF-Mistral 계열이 차지했으며, 독일어·영어·프랑스어 모든 공식 테스트셋에서 1위를 기록했고 기준 no-correction 대비 cMER가 전반적으로 크게 감소했습니다. 또한 제출물과 데이터, scorer, 평가 파이프라인을 공개해 향후 연구자들이 동일한 평가 틀에서 시스템을 체계적으로 비교·개선할 수 있도록 했다는 점에서 의미가 큽니다.



### PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction (https://arxiv.org/abs/2607.08111)
- **Prior Approaches**: 대부분의 TSE(target speaker extraction) 최신 모델은 VoxCeleb, WSJ0-2mix, LibriMix처럼 인위적으로 만든 혼합 데이터로 학습해 성능이 좋아졌지만, 실제 대화에서는 잔향·잡음·기기 왜곡과 자연스러운 턴테이킹 때문에 학습-배치 간 괴리가 생긴다는 문제가 컸습니다. REAL-T는 실제 대화 혼합을 평가하지만 훈련용 대규모 실데이터와 ‘깨끗한 타깃 화자 음성’이 제공되지 않아, 기존 시스템들도 결국 시뮬레이션 혼합에 의존하는 한계가 있었습니다. 그 결과 실환경 전개 시 성능 저하가 나타나며, 실데이터를 효과적으로 학습에 활용하는 방법이 공백으로 남아있었습니다.

- **Core Contribution**: PS4는 깨끗한 타깃 음성이 없는 실대화 데이터에서 훈련을 가능하게 하는 proxy-supervised(프록시 지도) 학습 프레임워크를 제안합니다. 이를 위해 REAL-PS4라는 대규모 코퍼스를 구축해 4개 공개 데이터셋을 REAL-T 포맷에 맞게 재가공하고, 71,771개 학습 샘플에 혼합 음성·화자 엔롤먼트·전사·프레임 VAD(voice activity detection) 라벨을 포함시켰습니다. 또한 BSRNN 기반 TSE를 fine-tuning할 때 separator만 갱신하면서 언어/화자/시간/지각 품질의 4개 상보 목적을 조합해 학습 신호를 만듭니다.

- **Technical Challenges**: 핵심 기술 난제는 실대화에서 타깃 화자의 ‘정답 신호(SI-SNR 등 신호 레벨 감독)’를 얻기 어렵다는 점입니다. PS4는 이를 해결하기 위해 차별 가능한 teacher(Whisper)로 ASR cross-entropy를 주고, 엔롤먼트와의 speaker similarity(코사인 마진 랭킹)로 화자 동일성을 고정하며, 다이어리제이션에서 생성한 프레임 VAD로 timing을 맞추고, DNSMOS(지각 품질)를 미분 가능 형태로 보조해 품질까지 끌어올리는 구조를 채택합니다. 학습 안정성을 위해 사전학습된 BSRNN-ECAPA의 speaker encoder는 고정하고 separator만 업데이트해 효율적으로 도메인 적응을 수행합니다.

- **Empirical Impact**: REAL-T 벤치마크에서 PS4는 공식 baseline보다 5개 서브 코퍼스 전반에 걸쳐 일관되게 개선되며, 특히 DNSMOS OVRL에서 두드러진 상승(기준선 2점대 미만→3.1 이상)을 보였습니다. 정량적으로는 개발셋에서 TER은 더 낮아지고, speaker similarity와 타이밍 F1은 더 높아져 프록시 목적들이 실제 화자 분리와 타이밍 정합에 효과적임을 확인했습니다. 공식 리더보드에서는 종합 2위를 기록했으며, 전체 제출 중 최상위 수준의 F1(0.871)과 SIM(0.565)을 달성해 실대화 TSE 학습 패러다임으로서의 실용성을 입증했습니다.



### Deep Learning Method for Stationary Distribution of Reflected Brownian Motion (https://arxiv.org/abs/2607.08091)
- **Prior Approaches**: 정반사 브라운 운동(RBM)의 정상분포는 다차원 확률시스템에서 중요하지만, 정상분포의 폐형식 해는 일부 특수 경우에만 알려져 있다. 성능지표로 쓰이는 꼬리확률 같은 값은 정상분포 자체보다 훨씬 계산이 어려워 기존 수치·근사 접근이 제한적이다. 최근에는 딥러닝으로 고차원 확률/미분방정식을 푸는 시도가 있었지만, 정상상태의 Laplace 변환처럼 “정상관계”를 직접 학습하는 프레임워크는 부족했다.

- **Core Contribution**: 이 논문은 BAR(basic adjoint relationship)의 Laplace 버전을 이용해 고차원 RBM 정상분포의 Laplace transform을 딥러닝으로 정확하고 효율적으로 학습한다. 특히 tail probability를 계산하기 위해 필요한 Laplace transform을 직접 예측하고, 수치적 역라플라스 변환을 통해 성능지표로 연결한다. 또한 학습용 손실함수·데이터 샘플링·신경망 구조를 함께 설계해 차원 d가 커져도 파라미터 수가 스케일하지 않도록 했다.

- **Technical Challenges**: 핵심 난제는 (1) Laplace transform이 θ에서 지수적으로 변해 학습 수치가 불안정해지고, (2) 단순 잔차 최소화만으로는 해의 해석성·단조성 같은 구조가 보장되지 않으며, (3) 고차원 균일 샘플링은 중요 “코너” 영역을 잘 못 뽑아 꼬리 영역 학습이 흔들린다는 점이다. 이를 위해 로그-파라미터화로 스케일 문제를 완화하고, BAR 잔차를 정규화한 정규화 오차를 손실로 쓰며, 단조성·해석성(Cauchy–Riemann)·원점 앵커링을 제약 항으로 추가했다. 또 코너를 노리는 2-stage 샘플링을 설계하고, Fourier features와 공유 인코더+가산(합) 집계를 통해 차원에 따른 입력층 스케일링을 줄인 아키텍처를 사용했다.

- **Empirical Impact**: 실험에서는 ground-truth 꼬리확률을 알 수 있는 RBM 인스턴스(저차원 및 2020/3030차원)에서 학습한 Laplace transform을 Talbot 방법으로 역변환해 tail probability를 계산했고, 예측이 거의 완벽하게 일치했다. 저차원 결과는 차원별로 복잡하게 달라지는 Laplace 구조도 잘 포착함을 보여주며, 고차원 결과는 차원이 커져도 정확도를 유지하는 확장성을 입증한다. 저자들은 이 접근이 해석적으로 다루기 어려운 정상상태 성능지표를 “일반 도구”로 추정하는 데 의미가 있다고 강조한다.



### LDFE: Laplacian Decoupled Feature Enhancement Block for Dual-Stream CNN-based RGB-IR Object Detection (https://arxiv.org/abs/2607.08076)
- **Prior Approaches**: 극한 기상·저조도 환경에서는 단일 모달 기반 탐지가 한계를 보이면서, RGB-IR 쌍을 활용한 듀얼 스트림 YOLO 계열이 주류로 자리 잡았다. 기존 연구들은 주로 feature fusion 설계(다중 단계에서 RGB/IR 특징을 어떻게 섞을지)에 집중했지만, CNN은 전역 정보 포착이 약하고 Transformer/Mamba는 세부 디테일 민감도가 떨어지는 구조적 제약이 남아 있었다. 또한 잡음 억제를 전체적으로만 처리하거나(holistic), 전역/국소 특성을 분리해 다르게 다루지 못해 융합 효율이 제한되었다.

- **Core Contribution**: 이 논문은 듀얼 스트림 CNN 백본의 각 stage에서 RGB-IR 특징을 global-local로 분해한 뒤, 잡음 제거·융합·복원을 순차적으로 수행하는 Laplacian Decoupled Feature Enhancement(LDFE) 블록을 제안한다. LDFE는 Laplacian Pyramid로 전역(저주파)과 국소(고주파) 성분을 분리하고, 이후 Global State Space Enhancement(GS2E)와 Local Convolutional Correlation Enhancement(LC2E)로 각각 다른 방식의 denoising과 융합을 수행한 뒤 Laplacian reconstruction으로 되돌린다. 특히 GS2E는 양 모달을 번갈아 main/auxiliary로 두는 교차 주의 기반 억제와 state space 모델을 결합해 전역 구조의 장거리 의존을 강화한다.

- **Technical Challenges**: 핵심 난제는 전역 구조와 국소 디테일을 동시에 살리면서, RGB/IR에 내재한 modality-specific 잡음을 융합 과정에서 효과적으로 억제하는 것이다. 저자는 global-local 분해를 통해 잡음과 정보의 성격을 분리해 다루고, GS2E에서는 채널 스왑 후 교차 모달 attention으로 주된 모달의 잡음을 동적으로 억제하면서 long-range dependency를 State Space Model로 포착하도록 구성했다. LC2E에서는 국소 영역에 대해 L1 Normalization 기반 denoising과 Softmax Fusion, 그리고 spatial/channel attention 및 triple convolution으로 미세 디테일의 융합 품질을 끌어올렸다.

- **Empirical Impact**: 6개 RGB-IR 탐지 벤치마크(M3FD, DroneVehicle, LLVIP, FLIR-Aligned, KAIST, VEDAI)에서 LDFE 기반 모델은 SOTA 대비 mAP를 최대 6.2%p~2.0%p까지 일관되게 개선해 방법의 실효성을 확인했다. 또한 듀얼 스트림 CNN에 가벼운 LDFE를 얹는 설계와 Mamba의 linear complexity 덕분에 성능뿐 아니라 파라미터 효율과 런타임에서도 이점을 제시한다. 극한 조건에서 RGB-IR의 상보성을 “전역-국소 및 잡음 성격별로 분리해 융합”한다는 관점이 이후 융합 블록 설계에 실질적인 기준점을 제공할 것으로 보인다.



### COBART: Controlled, Optimized, Bidirectional and Auto-Regressive Transformer for Ad Headline Generation (https://arxiv.org/abs/2607.08071)
Comments:
          10 pages, 5 figures, 5 tables. Published in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22). This is the author's accepted version; the definitive Version of Record is available at this https URL

- **Prior Approaches**: 기존 온라인 광고 헤드라인 생성은 주로 Transformer를 fine-tuning해 문장 품질(Rouge-L 등)을 높이거나, CTR을 RL/제약 조건으로 간접 최적화하는 방식이 많았다. 하지만 광고 포맷이 계속 바뀌고 길이·스타일·매력도 같은 요구조건이 수시로 달라지면, 사전학습 모델을 “그때그때” 맞춤 제어하기가 어렵다는 한계가 있었다. 또한 다수 연구는 조건을 자동으로 맞추되, inference 시점에 원하는 특성을 사용자가 직접 통제하는 능력은 제한적이었다.

- **Core Contribution**: 이 논문은 BART에 prefix control tokens를 결합해 CTR과 헤드라인 길이 같은 특성을 추론(inference) 단계에서 사용자가 직접 지정·동시에 최적화할 수 있게 한 COBART를 제안한다. control token을 입력 접두사로 넣어 인코더-디코더 attention이 해당 특성에 조건부로 동작하도록 유도한다. 그 결과, 다양한 광고 포맷/화면 크기에 맞춰 길이를 제어하면서도 더 높은 CTR을 노릴 수 있다.

- **Technical Challenges**: 핵심 기술 과제는 (1) pre-trained된 BART가 원래 control token 없이 학습됐는데도, control token 입력만으로 CTR·길이 조건을 안정적으로 따르게 만드는 것과 (2) CTR 같은 ‘관측 기반’ 연속 값/버킷화가 성능에 미치는 영향이다. 저자들은 CTR을 분위수 기반 버킷(15개)으로 분해해 태그 형태로 학습시키고, control token만으로도 인코더 표현이 조건부로 업데이트된다는 점을 통해 구조 변경 없이 제어 가능함을 보였다. 또한 Self-critical Sequence Training(SCST)와의 결합, 그리고 VBART 변형을 통해 reward/연속값 처리 방식이 CTR 추정과 품질 균형에 미치는 효과를 실험으로 점검했다.

- **Empirical Impact**: 실험 결과, 제안 방법은 기존 강한 baseline 대비 Rouge-L이 25.82% 증가하고 추정 CTR이 5.82% 증가하는 성과를 보고했다. 특히 SCST를 함께 쓰면 CTR 개선이 더 커지며, CTR을 버킷화하기보다 연속값을 직접 쓰는 설정에서 추정 CTR 향상이 더 두드러졌다. 전반적으로 “사용자 제어형 + 성능 최적화형” 헤드라인 생성으로 실사용 요구(포맷별 길이 제어, CTR 개선, 확장성)를 동시에 만족시키는 접근이라는 점에서 의미가 있다.



### When Thinking Hurts: Epistemic Signals in the Reasoning Chains of Visual Language Models (https://arxiv.org/abs/2607.08059)
Comments:
          7 pages, 2 figures, 5 tables. Oral paper at the 2nd Workshop on Epistemic Intelligence in Machine Learning (EIML@ICML 2026), Seoul, South Korea

- **Prior Approaches**: 기존 VL-불확실성 연구는 주로 정답/답변 토큰의 entropy 같은 출력 분포에 기반해 불확실성을 추정해 왔습니다. 하지만 thinking-mode VLM은 답을 내기 전 <think>…</think> 추론을 먼저 생성해, 기존 방법이 전제로 한 “에피스테믹 변동”이 다른 단계에서 사라질 수 있다는 점이 충분히 규명되지 않았습니다. 또한 MC-sampling·self-consistency처럼 여러 번 전진을 요구하는 방식은 단일 패스 신호로 thinking-mode의 구조적 변화를 포착하기 어렵습니다.

- **Core Contribution**: 이 논문은 thinking-mode VLM에서 ‘답변 토큰 entropy’가 붕괴되는 현상을 제어 실험으로 처음 보였고, 대신 <think> 체인 자체에서 추출한 신호가 예측력을 회복한다는 점을 제안합니다. POPE 적대 샘플에서 Qwen3-VL-8B-Thinking은 답변 entropy AUROC가 0.492(우연수준)로 붕괴했지만, thinking chain entropy는 0.647로 유지됐습니다. 더 나아가 세 패밀리( Qwen, GLM, InternVL )에서 답변 entropy 거동이 ‘붕괴-비붕괴-선택적 thinking’처럼 질적으로 갈린다는 경험적 분류를 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 thinking chain이 답변 토큰 분포를 사전조건화해 entropy 기반 불확실성 신호를 무력화할 수 있다는 구조적 문제를 기존 불확실성 방법이 놓친다는 점입니다. 저자들은 단일 greedy forward pass에서 답변 구간과 thinking 구간의 per-token Shannon entropy를 각각 계산해, AUROC로 hallucination error 예측 성능을 비교하는 방식으로 이 차이를 분리해 냈습니다. 또한 abstention(거절) 게이팅까지 설계해, 추가 추론 비용 없이 chain 신호가 실사용에서 어떤 이득을 주는지 정량화했습니다.

- **Empirical Impact**: 결과적으로 thinking chain entropy는 체인을 생성하는 구간에서 답변 entropy보다 일관되게 우수하며, Qwen(0.647 vs 0.492)과 GLM(0.759 vs 0.716)에서 통계적으로 명확합니다. InternVL3-8B는 선택적 thinking 때문에 chain 내용의 이점이 약하지만, 체인 생성 여부 자체가 불확실성 신호로 작동합니다(해당 벤치마크에서 체인 부재가 더 높은 FP로 연결). POPE에서 abstention 게이팅은 정확도를 71.0%→93.8%로 끌어올리며 커버리지는 62.7%에 도달하고, 300개 VQAv2 파일럿에서도 chain entropy가 answer entropy(0.680 vs 0.595)보다 잘 예측했습니다. 결론적으로 thinking-mode VLM의 불확실성 평가는 ‘답변 결과’가 아니라 ‘추론 과정’에 맞춰 설계돼야 한다는 메시지를 실증적으로 강화합니다.



### Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization (https://arxiv.org/abs/2607.08057)
Comments:
          Accepted to ACL 2026 as a Findings paper

- **Prior Approaches**: 기존 KV 캐시 최적화 연구는 메모리 절감이나 처리량·지연 개선을 목표로 해왔지만, 대개 KV를 전체 서빙 파이프라인의 한 구성요소로 부분적으로 다뤘다. 특히 많은 작업이 생애주기(예: prefill/decoding), 커널·모델 디테일, 또는 특정 기법군(압축·오프로딩 등) 중심으로 분류되어 서로 다른 시스템 효과의 관계를 한눈에 보기 어렵다는 한계가 있었다. 결과적으로 KV 캐시를 둘러싼 시간적(스케줄), 공간적(배치/이동), 구조적(표현/유지) 동작이 어떻게 얽혀 SLO와 품질 저하를 유발하는지 체계화가 부족했다.

- **Core Contribution**: 이 논문은 LLM serving-time KV 인프라를 system-aware KV infrastructure for serving LLMs, sKis로 정의하고, 이를 실행/스케줄(temporal), 배치/마이그레이션(spatial), 표현/유지(structural)이라는 3축 동작 공간으로 재정리한다. 또한 서로 다른 동작 축의 co-design 양상과 동작-목표(지연, 처리량, 메모리, I/O, 에너지) 간 연결을 분석해, 향후 연구 기회를 구조적으로 도출한다. 모델이나 커널 세부를 분리해 새로운 기법을 같은 렌즈에 올려 비교·위치지을 수 있는 기반을 제공한다.

- **Technical Challenges**: KV 최적화는 단순히 메모리를 줄이는 것에서 끝나지 않고, 스케줄링 지연·전송 병목·(de)quantization 비용·캐시 미스 등 실행 경로 전체의 상호작용을 통제해야 한다. 논문은 이를 해결하기 위해 각 축에서 KVS(요청/토큰/커널 레벨의 KV-aware 스케줄), HAE(이질 하드웨어에 적응하는 disaggregation·offloading), MHO/CDO(메모리 계층·장치 간 KV 배치/이동), KVCC(비대칭 양자화·VQ·outlier 처리) 및 KVRM(페이지드 메모리, 프롬프트 공유 인덱스, eviction/예산 할당) 등으로 기술 요소를 체계화한다. 더 나아가 tail latency, 품질 열화, 에너지·신뢰성 같은 운영 현실의 제약이 동작 축 간 간섭에서 발생함을 강조한다.

- **Empirical Impact**: 실증적으로 논문은 커뮤니티 문헌을 행동×목표 매트릭스와 co-design 친화도 네트워크로 요약해, 구조적 방법이 메모리 절감에 가장 많이 기여하고 시간적 방법은 지연·처리량에 더 직접적으로 매핑되지만 tail 지표 보고는 드물다는 점을 드러낸다. 또한 품질 손실이 보편적이며, 동작 축마다 품질 저하 원인이 달라 ‘열화의 통제 가능성’이 핵심 문제임을 정리한다. 마지막으로 에너지·신뢰성·SLO 중심 tail 제어, HAE–CDO의 범용화, 동작 간 공동 최적화 및 unified benchmarks 같은 공백을 명확한 연구 의제로 제시해, 향후 sKis 설계의 방향성을 좌우할 것으로 기대된다.



### Reinforcing the Generation Order of Multimodal Masked Diffusion Models (https://arxiv.org/abs/2607.08056)
- **Prior Approaches**: 기존 masked diffusion models(MDMs)은 임의 순서 infilling을 허용하지만, 생성 순서(언마스크할 토큰 위치의 순서)는 결과 품질에 큰 영향을 준다는 점이 최근에 부각됐다. 언어 과제에서는 logits 기반 Top-KK, Top-KK margin 같은 휴리스틱이 Sudoku 같은 구조화 문제에선 잘 먹히지만, 이미지 생성과 멀티모달 이해로는 일관된 개선이 확인되지 않았다.

- **Core Contribution**: 이 논문은 텍스트-투-이미지 합성과 멀티모달 이해에서 “생성 순서를 어떻게 정할지”를 학습 가능한 방식으로 해결한다. logits만으로는 최적 순서를 찾기 어렵다는 관찰 위에, 생성 순서 제어용 learnable control block(UPM 기반)를 두고 GRPO(Group Relative Policy Optimization)로 직접 최적화한다. 그 결과 fine-grained spatial 관계를 더 잘 반영하면서 멀티모달 추론/이해 성능도 함께 끌어올린다.

- **Technical Challenges**: 핵심 기술적 난제는 masked diffusion 정책에서 각 단계의 생성 순서를 바꿨을 때 전체 품질에 대한 크레딧을 어떻게 효율적으로 학습하느냐이다. 논문은 Plackett-Luce 기반으로 top-K 언마스크 집합을 샘플링하는 순서 정책을 만들고, 그룹 단위 advantage 추정 및 KL 정규화를 포함한 GRPO로 제어 모듈을 학습한다. 또한 토큰/시퀀스 likelihood를 명시적으로 다루기 어려운 제약을 고려해 diffu-GRPO 계열의 추정 아이디어와 함께, 전체 디노이징 궤적 정보를 누적해 더 강한 학습 신호를 제공한다.

- **Empirical Impact**: GenEval(객체 중심 텍스트-투-이미지 alignment)에서 제안 방법은 4.08% 상대 개선을 달성했으며, VLMEvalKit(시각질문응답·추론·지각 등)에서도 4.85% 상대 개선으로 멀티모달 이해 전반의 효과를 확인했다. 특히 confidence 기반 순서 휴리스틱이 일으키던 이미지 품질 정체가 줄고, 생성 과정 중간 단계가 더 좋아지며 복수 객체·공간 관계 추론에서도 이득이 관측된다. 정성 결과에서도 더 응집력 있는 생성과 더 자세한 추론(단일 토큰으로의 붕괴 완화)이 나타나, 순서를 학습하는 접근이 멀티모달 확산 모델 전반에 일반적으로 유효함을 시사한다.



### Who Analyses the Analyser? Self-Validating LLM Hazard Analysis with Constitutional Meta-STPA (https://arxiv.org/abs/2607.08054)
- **Prior Approaches**: 기존 연구들은 LLM을 STPA의 losses–hazards–UCAs–constraints 생성 파이프라인에 끼워 넣어 문서화와 추적성을 자동화하려 했습니다. 하지만 분석의 대상은 대체로 ‘외부 시스템’뿐이라, 그 분석을 수행하는 LLM 보조 도구 자체가 낳는 위험(기준선 없는 안전표준 인용, 검증 불가 제약 생성, 감사 추적 부재)은 구조적으로 점검되지 않았습니다.

- **Core Contribution**: 이 논문은 “분석하는 분석자” 문제를 정면으로 다루기 위해, STPA를 도구에 재적용한 Constitutional Meta-STPA를 제안합니다. 도구가 생성하는 loss→hazard→UCA→constraint 흐름을 근거로 21개의 Tool Principles와 8개의 Meta-Safety Principles를 ‘주장’이 아니라 ‘유도’하며, 각 원칙을 코드 집행 지점으로 바인딩합니다.

- **Technical Challenges**: 핵심 기술 난제는 LLM이 스스로 만든 규칙을 맹신할 경우 측정과 집행의 근거가 흔들린다는 점과, “커버리지” 같은 지표가 모델/스캐너/표현 편향에 좌우될 수 있다는 점입니다. 연구진은 LLM 전반이 아니라 고정된 구조(구체적 UCA 슬롯 강제)와 결정론적 validator/semantic voting/해시 기반 audit log로 도구의 안전 거버넌스를 텍스트 산출물에 묶고, constitution-marginal coverage를 보조정리(coverage가 모델·스캐너 외 의존성을 갖지 않음)로 형식화했습니다.

- **Empirical Impact**: 실험에서 frontier ensemble은 도구의 자체 설계로부터 행동 원칙 18/21과 거버넌스 원칙 8/8을 복원해, meta 레이어의 병목이 ‘문구’가 아니라 모델 역량에 있음을 시사합니다. 또한 Tool Principles는 보유한 비적대적(혹은 미작성) 프로브에서 안전점수를 약 80%p 수준으로 올렸고(p<0.001), 무작정의 helpful/harmless/honest 같은 프리앰블은 큰 개선을 내지 못했습니다.



### What LLM Forecasters Know but Don't Say: Probing Internal Representations for Calibration and Faithfulness (https://arxiv.org/abs/2607.08046)
- **Prior Approaches**: 예측을 위해 fine-tuning된 LLM은 정확도는 높아도 예측에 대한 확신(캘리브레이션)이 약할 수 있다는 문제가 지적돼 왔다. 또한 chain-of-thought(CoT) 설명은 그럴듯하지만, 실제 예측을 지지하는 증거와의 정합성이 충분히 보장되지 않는 경우가 많다. 기존 평가는 주로 출력과 CoT 자체에 의존해 내부에서 무엇이 확신을 결정하는지 검증하기 어려웠다.

- **Core Contribution**: 본 논문은 내부 표상(중간 activation)을 읽어내는 representation-pooling probe가 예측 캘리브레이션을 실질적으로 개선할 수 있음을 보인다. 또한 증거 ablation과 distraction( diversionary injection )으로 CoT의 증거 충실성/거짓말 탐지 가능성을 비교하며, 내부 표상이 행동 변화를 더 잘 추적한다고 주장한다. 마지막으로 forced answering과 라우팅을 통해 예측은 reasoning이 시작되기 전 이미 상당 부분 고정된다는 통찰을 제시한다.

- **Technical Challenges**: 핵심 과제는 (1) 중간 표상에서 ‘예측 근거’와 ‘불확실성’을 신뢰성 있게 뽑아 캘리브레이션을 개선하는 것, (2) CoT가 아닌 실제 영향이 프롬프트에서 제거/교란될 때도 내부가 이를 포착하는지 검증하는 것이었다. 저자들은 Eternis-Forecaster 8B에서 intermediate activations에 대해 representation-pooling probes를 학습해 GLM 계열(GLM-4.7-Flash, GLM-4.5-Air)에서도 동일 경향을 확인한다. 더 나아가 증거 ablation 후에도 reasoning trace가 유지되는 상황에서 probe가 행동 변화를 추적하도록 설계해, CoT가 영향 숨김을 수행할 때도 내부 신호가 유효함을 보여준다.

- **Empirical Impact**: OpenForesight와 다른 GLM 계열 설정에서 probe 기반 표상은 CoT나 기존 방식 대비 캘리브레이션이 “substantially better”하다고 보고한다. CoT의 증거 충실성은 evidence ablation/diversionary injection에서 흔들림이 reasoning trace에는 잘 드러나지 않았지만, probe는 behavioral shift를 더 잘 추적했고 84%에서 변화 방향까지 예측했다. 또한 reasoning 전 단일 패스의 pre-set answer 분포로 질문을 라우팅하면 토큰을 30-47% 절감하면서 정확도 손실 없이 유지돼, 내부 표상을 캘리브레이션·감사(auditing)·트리아지(triaging) 도구로 활용할 수 있음을 실증한다.



### Aleena: Alignment Agent for Research Software Engineering Collaborations (https://arxiv.org/abs/2607.08043)
Comments:
          8 pages, 5 figures. AgenticSE @ KDD '26: Agentic Software Engineering (SE 3.0): The Rise of AI Teammates, KDD 2026 Workshop

- **Prior Approaches**: 기존 연구 소프트웨어 엔지니어링 협업은 회의·채팅·이슈·풀리퀘스트가 분산된 ‘기록’으로 남아, 결정의 배경(rationale)이 문서와 아티팩트 사이에서 끊기기 쉽다. 또 요구사항이 문서로 고정되기보다 연구 이해가 성숙되며 함께 생겨, 초기 스코핑만으로는 프로젝트 상태 정렬을 지속하기 어렵다는 한계가 반복된다.

- **Core Contribution**: 이 논문은 연구 소프트웨어 엔지니어링의 정렬(alignment)을 일회성 범위 조정이 아니라 프로젝트 상태를 연속적으로 관리하는 문제로 재정의한다. 이를 위해 Aleena라는 오픈소스 ‘lifecycle alignment agent’를 제안하며, GitHub를 공유 표면으로 삼아 다중 모달(회의/채팅/이슈/PR) 입력을 구조화된 프로젝트 기록으로 변환한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 이해관계자 의사결정의 연속성을 보존하면서 (2) 과학 용어/가정의 드리프트와 (3) 소유권·리뷰 책임 전이를 추적하고 (4) 인간의 최종 권한을 유지하는 것이다. Aleena는 GitHub 네이티브 작업(이슈/디스커션/코멘트/드래프트 PR 등)만 선택하고, merge나 close 같은 외부 실행은 보류하는 defer 설계를 통해 human agency를 유지한다.

- **Empirical Impact**: 저자들은 UW SSEC에서 20개 이상 프로젝트에 대한 센터 경험을 바탕으로 AC1–AC4(기대, 용어·가정, 소유권, 의사결정 연속성) 시나리오를 구성하고, 실제 동작이 GitHub 기록으로 남는 흐름을 예시로 제시한다. 또한 향후에는 삽입된 트랜스크립트 대비 생성 아티팩트 수, 보존/수정/무조치 종료 비율, 닫힌 이슈의 재오픈 및 소유권 수정 사례 같은 정량 지표와 인터뷰로 조율 효과를 검증할 계획이다.



### PLURAL: A Global Dataset for Value Alignmen (https://arxiv.org/abs/2607.08034)
- **Prior Approaches**: 기존 LLM 정렬 연구는 주로 서구권 가치 데이터나 기준에 의존해, 다른 문화권의 가치 체계를 충분히 반영하지 못한다는 한계가 지적돼 왔다. 또한 가치가 담긴 선호 데이터를 만들 때는 실제 설문 신호를 제대로 보존하면서도 현실적인 시나리오로 바꾸는 과정이 어려워 규모 확장이 쉽지 않았다. 그 결과 문화별 ‘가치 신호’를 학습 가능한 형태로 대규모 제공하기가 제한적이었다.

- **Core Contribution**: 이 논문은 Integrated Values Survey(IVS) 기반의 대규모 가치 지향 preference 데이터셋 PLURAL을 제안한다. PLURAL은 IVS의 92개국(전국대표 설문)을 근거로 설문 응답을 합성 preference triplet으로 변환해, 국가 간 규범적 가치 신호와 국가 내부 다양성을 함께 담도록 설계했다. 초기 공개 버전은 약 50만 개 triplet로 20개 다양한 국가의 사람들을 커버한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 설문에서 추출한 규범 가치 신호를 합성 데이터에서 왜곡 없이 유지하고 (2) 동시에 현실감 있는 상황을 생성해 학습에 유용한 형태로 만드는 것이다. 연구진은 두 단계 생성 파이프라인으로 설문 응답을 합성 triplet으로 바꾸되, 가치 신호 보존과 시나리오 현실성을 동시에 노렸다. 또한 국가별 문화 프로필에 맞춰 정렬이 실제로 개선되는지 평가 설계를 병행했다.

- **Empirical Impact**: PLURAL은 데이터셋 수준 검증에서 원 설문이 가진 국가 간 가치 차이와 국가 내부 다양성을 모두 보존함을 보였다. 자동 평가에서는 PLURAL로 학습한 모델이 강한 베이스라인 대비 목표 국가의 문화 프로필에 더 잘 정렬되며 mean absolute error를 최대 27.7%까지 줄였다. 블라인드 휴먼 평가에서도 인도·브라질·일본의 176명 평가자가 PLURAL-정렬 응답이 자국 가치에 더 대표적이라고 판정해, value steering에 학습 가능한 신호가 존재함을 시사한다.



### DKDNet: Dual Knowledge and Data-Driven Network for Cross-Domain Automatic Modulation Classification (https://arxiv.org/abs/2607.08031)
Comments:
          13 pages, 6 figures, 9 tables

- **Prior Approaches**: 기존 자동변조분류(AMC)는 크게 (1) 통계적 합성 가설검정 기반의 likelihood 방법과 (2) AP, constellation diagram, DFT, 고차 누적량 같은 수공 특징 기반 분류로 나뉘며, 데이터가 풍부해질수록 DL 기반 방식이 확산됐다. 하지만 DLCNN/LSTM 등은 학습·추론 분포가 같다는 가정을 자주 깔고, 전송 손실, 다중경로 페이딩, 잡음, RF 캘리브레이션 오류 같은 요인이 만들던 분포 이동으로 실제 환경 성능이 급락한다. 이를 완화하려는 UDA는 소스·타깃 특징 정렬에 집중하지만, 모듈레이션 고유 구조를 도메인 전반에서 보존하는 데는 제한적이라는 문제의식이 제시된다.

- **Core Contribution**: 논문은 신호의 물리/프로토콜 기반 prior knowledge를 교차도메인 표현학습에 활용하는 접근을 제안한다. 우선 IQ, AP, DFT, ACF, CD의 다섯 표현을 ‘모듈레이션 판별력–도메인 불일치–상보성’ 관점에서 분석해, 최종적으로 compact prior 입력으로 IQ, AP, ACF를 선택한다. 그 위에 dual knowledge and data-driven network(DKDNet)를 구축해, 구조적 prior 입력과 데이터 기반 특징 학습 및 도메인 적응을 함께 최적화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 더 풍부한 표현이 항상 더 좋은 교차도메인 일반화를 주지 않고, (2) 각 prior가 도메인 불변성과 판별성의 균형에서 서로 다르게 거동한다는 점이다. 이를 해결하기 위해 DKDNet은 MRFE(multi-representation feature encoder)로 IQ/AP/ACF를 각각 독립 인코딩해 공통 잠재공간으로 매핑하고, DLFU(dynamic lightweight fusion unit)로 표현 간 상보 정보를 가볍고 동적으로 융합한다. 이후 모듈레이션 분류 손실과 적대적 도메인 정렬 목표를 결합해, 판별적이면서도 transferable한 fused feature를 학습하도록 설계한다.

- **Empirical Impact**: 실험은 시뮬레이션 및 공개 데이터셋에서 수행되며, 분석 단계의 prior 선택이 합리적임을 보여주고 제안 방법의 우수성이 입증된다. 또한 채널 손상을 점진적으로 강화한 합성 데이터셋 RML2025 Series를 공개해, 교차도메인 AMC 성능·견고성·유연성·샘플 효율을 폭넓게 검증한다. 결과적으로 DKDNet은 단순 특징 정렬보다 신호 지식 기반 구조 큐를 결합할 때 일반화가 개선된다는 실증적 근거를 제공하며, AMC의 실사용 문제(분포 이동)에 대한 실질적 대안을 제시한다.



### Structured Pruning of Large Language Models via Power Transformation and Sign-Preserving Score Aggregation with Adaptive Feature Retention (https://arxiv.org/abs/2607.08027)
- **Prior Approaches**: LLM 압축을 위한 가지치기는 unstructured pruning과 structured pruning으로 나뉘는데, 전자는 정확도는 높지만 비정형 희소성 때문에 실질 속도 향상이 제한적이다. 후자는 하드웨어 친화적인 뉴런/채널 단위 제거로 가속이 가능하지만, 가중치 단위 점수를 구조 단위로 옮기는 과정에서 성능 저하가 자주 발생한다. Adaptive Feature Retention(AFR)은 ReFer와 SNIP 점수를 표준화·합산해 사전학습 지식 보존과 다운스트림 적응을 균형 있게 하려 했지만, 이를 structured pruning에 그대로 적용하기엔 여러 형태의 점수 집계 문제가 남아 있다.

- **Core Contribution**: 이 논문은 unstructured AFR을 structured pruning에 맞게 재구성하는 데서 생기는 핵심 실패 원인 3가지를 짚고, 이를 동시에 완화하는 통합 방법을 제안한다. 구체적으로 SNIP 점수의 분포 불일치, 신호 부호(최적화 방향) 손실, outlier의 영향 문제를 각각 전용 처리로 해결해 뉴런 단위 점수 산정의 신뢰도를 높인다. 그 결과, structured pruning에서도 unstructured AFR과 비슷한 정확도를 유지하면서 실제 추론 가속까지 달성하는 것을 목표로 한다.

- **Technical Challenges**: 첫째, ReFer와 SNIP의 분포가 달라 단순 표준화·평균이 잡음을 증폭시킨다. 이를 위해 SNIP에 power transformation을 적용해 저중요도 영역 잡음을 비선형으로 억제하면서 고중요도 신호는 보존한 뒤 표준화로 재정렬한다. 둘째, structured 집계에서 절댓값을 먼저 취하면 뉴런 내부 가중치의 최적화 방향 일관성(sign consistency)이 사라져 뉴런 중요도 판단이 흔들린다; 논문은 부호를 유지한 뒤 평균 후 절댓값을 취하는 sign-preserving aggregation을 사용한다. 셋째, ReFer 점수에 존재하는 극단 outlier가 평균 기반 순위를 뒤집을 수 있어 뉴런별 2nd~98th percentile 범위를 벗어난 값은 제거하는 percentile-based outlier removal로 집계를 안정화한다.

- **Empirical Impact**: 실험은 Llama-3-8B, Vicuna-v1.5-13B, LLaVA-v1.5-13B에서 수행되었고, 제안 방법은 naive structured AFR 대비 최대 21.27점(20%)과 15.13점(50%) 개선을 보였다. 또한 Llama-3-8B 기준 50% 가지치기에서 35.1% 파라미터/VRAM을 줄이면서 1.57× 추론 속도 향상을 달성했고, LLaVA-v1.5-13B에서도 1.56× 수준의 속도 개선을 확인했다. structured pruning이 특별한 희소 연산 라이브러리 없이도 표준 하드웨어에서 동작 가능한 형태로 가속을 제공한다는 점에서, deployment 관점의 실용성 있는 진전을 의미한다.



### APIVOT: Adaptive Planning with Interleaved Vision-Language Thoughts (https://arxiv.org/abs/2607.08024)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 장기 로봇 조작 계획은 작업의 의미 구조(목표 분해, 행동 순서)와 기하 제약(여유 공간, 충돌)까지 함께 고려해야 한다. 기존 LLM/VLM 플래너는 언어적으로는 그럴듯한 행동 순서를 잘 만들지만, 공간 제약 때문에 실행 불가능한 계획을 내는 경우가 많다. 이를 보완하려고 외부 모션 플래너/feasibility checker/학습된 동역학 모델을 붙여 재계획을 유도했지만, 정작 모델 내부의 추론 과정에는 기하적 판단이 ‘후처리’처럼 분리되어 있었다.

- **Core Contribution**: APIVOT는 VLM 기반 플래너로, 언어 추론(semantic reasoning)과 시각적 사고(visual thoughts)를 계획 추론 흐름(trace) 안에서 적응적으로 섞는다. visual thoughts는 상상된 미래 상태로서 이후 단계의 기하적 실행 가능성(공간 적합, 충돌 회피)을 내부에서 검증하는 데 쓰인다. 핵심은 매 단계에서 언어/비전 중 어느 모달리티가 필요한지 모델이 학습해, 불필요한 시각 추론을 줄이면서도 성공률을 높인다는 점이다.

- **Technical Challenges**: 문제는 ‘언어로는 정교한 기하 구조를 간결하게 표현하기 어렵다’는 점과, 반대로 시각적 사고를 매번 쓰면 계산 비용이 커진다는 점이다. APIVOT는 (1) 제공된 visual thoughts를 활용해 기하 정보를 추출하는 단계, (2) visual thoughts를 스스로 생성하도록 하는 단계, (3) 기하 제약이 중요한 단계에서만 visual thoughts를 생성하도록 하는 3단계 커리큘럼으로 적응성을 학습한다. 또한 visual thought가 가리키는 잠재 시각 상태를 정답 미래 장면 특징과 정렬(cosine similarity)해, 상상된 상태가 실제 장면과 맞물리도록 설계했다.

- **Empirical Impact**: KitchenWorlds의 장기 키친 태스크에서 APIVOT는 평균 작업 성공률 0.419를 기록하며 최강 베이스라인 대비 8.1%p 향상했다. 특히 공간 제약이 강한 설정에서 격차가 7점에서 17점으로 벌어져, 내부 기하 검증의 효과가 크게 드러났다. 더불어 visual thoughts를 모든 단계에 쓰는 상한 대비 성능의 91%를 유지하면서 토큰 사용은 39% 줄여, 모달리티 선택 학습이 성공률과 추론 효율을 동시에 개선함을 실증했다.



### Can We Trust LLM's Logic? Quantifying Uncertainty, Coherence, and Robustness via a Graph-Based Framework (https://arxiv.org/abs/2607.08017)
Comments:
          42 pages, 14 figures, 12 tables

- **Prior Approaches**: LLM의 불확실성을 보기 위해 토큰 확률 기반 UQ를 쓰거나, CoT를 여러 개 샘플링해 최종정답의 다수결(Self-Consistency, SC)로 답을 고르는 방식이 주로 쓰였다. 하지만 SC는 중간 추론의 논리적 타당성을 직접 검증하지 않고 최종 정답 합치만 본다.

- **Core Contribution**: 논문은 GraphEVAL로, “정답 일치”가 아니라 추론 경로의 일관성과 충실도(reasoning fidelity)를 중심에 둔 그래프 기반 평가 프레임워크를 제안한다. 또한 SS-GED를 바탕으로 Graph Reasoning Coherence Score(GRCS)라는 새로운 UQ 지표를 도입해 의미-구조 합의 부족과 병적 모드 붕괴 및 자신감 있는 환각을 함께 포착한다.

- **Technical Challenges**: 핵심 난제는 다단계 추론을 그래프로 구조화하면서 의미와 인과 의존까지 반영해 거리를 계산하는 것이다. 이를 위해 CoT를 인과 관계 DAG로 분해한 뒤 Semantic-Structural GED(SS-GED)로 노드/엣지의 의미·구조 유사도를 최적 매칭으로 계산하고, GRCS는 이 거리들의 분포 평균으로 전체 일관성을 정량화한다.

- **Empirical Impact**: GRCS는 모델 규모가 큰 경우와 작은 경우 모두에서 reasoning faithfulness와 일관되게 음의 상관을 보여, 기존 UQ 지표보다 추론의 숨은 불확실성을 더 잘 잡아낸다. 또한 Graph Self-Consistency(GSC)는 최종 다수결 대신 그래프 공간에서 중심(medoid) 추론 경로를 선택해 SC가 ‘운 좋은 정답(lucky guess)’을 과대평가하는 문제를 줄이며, medoid를 적대적으로 제거하면 추론 충실도와 일부 경우 정확도까지 떨어져 ‘load-bearing path’의 역할을 실험적으로 확인한다.



### Provably Optimal Learning Algorithms for Assistance Games (https://arxiv.org/abs/2607.08012)
- **Prior Approaches**: 보조 게임(assistance games)은 인간의 민간 선호를 관측하는 에이전트와 행동만 관측하는 보조 에이전트가 협력하도록 모델링해 왔지만, 학습과 계산을 모두 만족하는 알고리즘 설계는 미개척 영역이었다. 기존 연구는 최적 보조 정책의 구조나 정보-효용 간 트레이드오프, 평형 성질에 초점을 맞췄고, 반복·온라인 환경에서 효율적으로 “근최적”을 보장하는 방법은 부족했다. 또한 POMDP 등 다른 문제로의 환원 시도는 있었으나, 본 논문처럼 명시적으로 근사 보장과 효율성을 함께 다루는 접근은 드물었다.

- **Core Contribution**: 이 논문은 보조 게임의 온라인 반복 버전을 정의하고, 성능 지표로 assistance regret(사후 최적 관점 대비 누적 효용 손실)을 제안한다. 그리고 인간(정보 보유)과 assistant(행동 관측만)를 각각 위한 분산 학습 알고리즘을 제시해, (1-1/e) 근사 factor을 갖는 sublinear assistance regret을 보장한다. 특히 assistant에 어떤 no-regret 알고리즘을 넣어도 작동하도록 일반성을 확보했으며, 초기 조율을 약간 주면 더 좋은 ~O(sqrt(T)) 속도를 얻는다.

- **Technical Challenges**: 핵심 난점은 (1) 인간-assistant 정책 쌍 공간이 지수적으로 커서 계산이 어렵고, (2) 학습이 분산돼 있어 상호 조율이 자동으로 이뤄지지 않는다는 점이다. 이를 위해 관측·피드백 구조를 반영해 중앙집중형 메타-플레이어의 joint policy 최적화를 ‘weighted threshold potentials’와 ‘partition matroid’ 아래의 온라인 submodular maximization으로 환원하고, 근사 계수를 유지하면서도 정책 스위치 수를 억제하는 방식으로 효율성을 확보한다. 또 assistance regret을 중앙 알고리즘의 external regret, 인간의 policy switch 횟수, assistant의 moving target 대비 tracking regret으로 분해해 설계 목표를 안정성(인간)과 적응성(assistant)으로 구체화한다.

- **Empirical Impact**: 이론적 임팩트로, (1-1/e)보다 더 나은 α 근사 factor으로 sublinear assistance regret을 달성하는 것은 RP≠NP 가정 하에 계산적으로 비가 가능함을 보인다. 또한 초기 동기화(공유 random string) 기반의 pseudo-decentralized 설정에서는 assistance regret이 ~O(sqrt(T))로 개선되며, 로그 요인을 제외하면 최적에 가깝다는 점을 강조한다. 결과적으로 인간-AI 협력/보조 학습에서 “효율적인 분산 학습 + 정보-효용 균형”을 동시에 다루는 최초의 provably efficient 알고리즘 계열로 자리 잡을 가능성이 크다.



### Beware What You Autocomplete: Forensic Attribution of Backdoored Code Completions (https://arxiv.org/abs/2607.08011)
Comments:
          To appear in COLM 2026

- **Prior Approaches**: 코드 완성 모델에서의 backdoor 공격은 fine-tuning 데이터에 트리거-페이로드를 심어, 특정 문맥에서만 악성 코드 패턴을 유도한다는 점에서 소프트웨어 신뢰성을 크게 해친다. 기존 방어는 static analysis나 anomaly detection 같은 사전 예방 성격이 강하지만, 기능을 보존하면서 삽입 로직을 숨기는 adaptive/stealthy 공격에는 취약하다는 한계가 반복적으로 드러났다. 또한 poisoning forensics 연구는 주로 gradient나 retrieval 컨텍스트를 전제로 하거나, 코드 완성 환경에는 잘 전이되지 않는 방식이 많았다.

- **Core Contribution**: 이 논문은 “누가 버그를 심었는가”라는 post-attack forensics 질문에 답하기 위해 CodeTracer를 제안한다. 주어진 miscompletion event(프롬프트와 백도어 완성)와 fine-tuning corpus만으로, 악성 완성을 유발한 특정 오염 데이터 예시(또는 코드 조각)를 추적·귀속하는 프레임워크를 만든 것이 핵심 기여다. 특히 gradient/attacker 정보가 없는 현실 제약을 그대로 반영해 실사용 가능성을 높였다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) gradient 기반 영향도 계산이 불가능하고 (2) 수백만 규모 fine-tuning 데이터 전체를 예시 단위로 비교하는 것이 비현실적이며 (3) 기존 인스턴스/LLM 포렌식 기법이 코드 완성에는 잘 맞지 않는다는 점이다. CodeTracer는 이를 위해 악성 완성에서 구조적·의미적 정규성을 뽑아 structured behavioral fingerprint를 만들고, 이를 바탕으로 코드-to-code retrieval로 후보 범위를 top-KK로 좁힌 뒤, 외부 LLM 추론으로 후보 예시의 “동일한 불안전 논리” 여부를 [Label: Yes]로 판정한다. 즉 fingerprint extraction–scope narrowing–attribution analysis의 3단계를 통해 대규모 검색 비용과 귀속 정밀도 문제를 동시에 공략한다.

- **Empirical Impact**: 평가에서는 3가지 취약 시나리오(jinja2, requests, socket)와 10종 backdoor 공격(기본 8종+적응형 2종)을 포함해, CodeTracer가 경쟁 baselines 대비 forensic 정확도가 높고 false identification(낮은 FPR)도 유지됨을 보인다. 또한 포렌식으로 추적된 오염 예시를 제거했을 때 backdoor로 인한 공격 성공률(ASR)이 0.03 이하로 크게 낮아져, 단순 탐지보다 “후속 제거/진단”에 직접 유용함을 입증한다. 런타임과 비용도 짧은 편(예: 한 건 추적 수십 초, 악성 완성당 $0.33 수준)이라 적응형 공격 환경에서도 실전 적용 여지가 크다는 점이 강조된다.



### Reaction-network reasoning with frontier models for experimentally confirmed catalyst-selectivity hypotheses (https://arxiv.org/abs/2607.08003)
- **Prior Approaches**: 촉매(특히 반응 아키텍처) 발굴은 전통적으로 시행착오 실험과 계산 집약적 스크리닝에 의존해 병목이 컸다. 전기화학 CO2 환원처럼 선택성이 동적 계면·전해질·전위 요인과 반응 경로 경쟁의 산물인 경우, 기존 descriptor 기반 ML이나 computational potential은 주로 정적 기저 상태 지표나 벌크 상관관계에 의존해 분기 지점을 잘 다루기 어렵다.

- **Core Contribution**: 이 논문은 frontier language model을 반응 네트워크에 대해 엄격히 제한된 방식으로 추론하게 하여, 경로 경쟁을 좌우하는 물리적 레버(제어 요인)를 찾아 새로운 촉매를 설계할 수 있음을 보여준다. 특히 복잡한 화학 그래프에서 네트워크 불변성(network invariance)을 강제하는 human-AI co-thinking 프레임워크를 제안해, 검증 가능한 가설을 메커니즘 수준에서 추출한다.

- **Technical Challenges**: 핵심 기술 난제는 복잡한 반응에서 메커니즘 분기(kinetic pathway competition)를 static한 지표나 상관관계로는 포착하기 어려운 점이었다. 이를 위해 프레임워크는 반응 네트워크 상의 논리적 추론을 end-to-end topological pathway 분석 형태로 제한하고, 네트워크 불변성을 이용해 후보 제어 변수를 분리·정의한 뒤(예: 국소 알칼리도, 철 도입, 계면 양성자 공여 접근성 제한) 테스트 가능한 예측으로 연결한다.

- **Empirical Impact**: CO2 전기환원 적용 결과, acetate 경로를 ketene desorption과 hydroxide capture로 특정했으며, ketene에 대한 adsorbed CO 및 CH2 coupling 경로도 별도로 예측했다. 또한 local alkalinity, controlled iron incorporation, restricted interfacial proton-donor accessibility를 조절한 동작 가이드로 copper-iron oxide 촉매를 설계해, 매칭된 Cu-rich 기준선 대비 acetate 선택성을 3배 높이는 성과를 제시했다. 이 접근은 과거의 통계적 예측을 넘어 전향적 가설 생성 중심으로 계산 패러다임을 바꿀 수 있는 범용 설계 청사진을 제공한다.



### SpO$_2$ Predictor-Guided Stage-Wise Time-Frequency Reconstruction of Low-Quality Dual-Wavelength PPG for Oxygen Saturation Estimation (https://arxiv.org/abs/2607.07996)
- **Prior Approaches**: 기존에는 저품질 PPG를 적응 필터링, 저품질 구간 제거, 신호 재구성으로 처리해 왔습니다. 적응 필터링은 가속도 같은 보조 센서 의존도가 크고, 구간 제거는 유용한 생리 신호까지 과도하게 버릴 수 있습니다. 재구성 기반 방법도 time-domain waveform fidelity나 심박수 특성 보전에 치우쳐 SpO2에 직접 중요한 시간-주파수 구조와 생리학적 제약을 충분히 반영하지 못한다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 SpO2 예측기를 가이드로 삼는 stage-wise time-frequency PPG reconstruction 프레임워크를 제안합니다. 고품질 듀얼 파장 PPG로 먼저 SpO2 predictor를 학습한 뒤, masked 구간을 복원하는 reconstructor를 학습할 때 predictor를 추가 제약으로 연결해 복원이 SpO2 관련 정보 보존으로 이어지게 했습니다. 즉, 단순 잡음 제거가 아니라 downstream 산소포화도 추정에 정렬된 재구성 목표를 설계한 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술 난제는 재구성 목표를 PPG의 시간형태뿐 아니라 SpO2에 필요한 주파수(스펙트럼) 구조까지 동시에 보존하도록 만드는 것입니다. 이를 위해 reconstructor 학습에서 STFT 기반 frequency-domain loss를 time-domain waveform loss와 함께 사용했으며, 재구성된 PPG를 frozen SpO2 predictor에 넣어 SpO2 loss로 생리학적 정합성을 강제했습니다. 또 SpO2 predictor와 reconstructor를 4단계로 번갈아 최적화해(예측기 선학습→재구성 학습→예측기 적응→재구성 정교화) 두 모듈의 궁합을 단계적으로 맞추도록 구성했습니다.

- **Empirical Impact**: OpenOximetry Repository(공개)와 민간 웨어러블 PPG 데이터(We-Be)에서 subject-level MAE 기준 최저 성능을 보였습니다(공개 2.882%, 민간 2.359%). 특히 predictor-guided SpO2 loss를 제거하면 성능이 가장 크게 하락해, 재구성이 SpO2 관련 정보를 실제로 보존한다는 점을 ablation이 뒷받침합니다. 결과적으로 저품질 듀얼 파장 PPG 환경에서도 downstream SpO2 추정 정확도를 일관되게 개선하는 접근으로 의미가 큽니다.



### Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems (https://arxiv.org/abs/2607.07989)
Comments:
          To appear in COLM 2026

- **Prior Approaches**: LLM 기반 멀티에이전트 시스템의 오류는 에이전트 간 상호작용이 길어지면서 원인 에이전트와 ‘최초로 돌이킬 수 없게 만든 단계’를 특정하기 어렵다. 기존 실패 로컬라이제이션은 AgentTracer처럼 counterfactual replay로 인과를 보려 하지만, 다중 에이전트 환경에서는 한 단계 변경이 이후 프롬프트·툴·조정 흐름까지 흔들어 결론이 흔들릴 수 있다. 또 AEGIS처럼 사전 정의된 오류 패턴(템플릿/택소노미)에 매칭하는 방식은 예측 불가능한 emergent reasoning이나 조정 붕괴를 충분히 포착하지 못하고, poisoning forensics 계열도 멀티에이전트의 점진적 오인·coordination drift 특성에 맞지 않아 성능이 제한된다.

- **Core Contribution**: 이 논문은 실패를 특정 에이전트뿐 아니라, 궤적이 최초로 decisively misdirected 되는 ‘가장 이른 단계’까지 함께 귀속(attribution)하는 문제를 정식화하고, 이를 위한 프레임워크 AgentLocate를 제안한다. AgentLocate는 LLM 기반 Judge가 (responsible agent, earliest decisive step) 가설을 먼저 내고, 이를 검증 가능한 절차로 다단계화해 디버깅에서 재현성과 신뢰도를 높이는 방향을 택한다. 또한 Judge의 1회 출력이 아니라 다수 평가자 판단과 피드백을 수집해, 판단 품질을 lightweight fine-tuning으로 지속 개선한다.

- **Technical Challenges**: 핵심 기술 난제는 long-horizon, tool-mediated, tightly coupled 상호작용에서 ‘어떤 에이전트의 어떤 순간’이 전역 실패를 되돌릴 수 있는지 인과적으로 분리하는 것이다. AgentLocate는 먼저 all-at-once 또는 step-by-step으로 궤적을 해석해 counterfactual reversal 조건을 근사하는 가설을 만들고, 그 가설에 대해 base/concise/evidence-focused처럼 서로 다른 스타일의 independent Evaluators가 동일 궤적을 재분석하도록 한다. 이어 각 평가자의 self-reported confidence를 반영한 confidence-aware voting으로 후보 위치를 집계하고, Judgeft 학습에는 evaluator의 비판·불일치를 LoRA 기반(parameter-efficient) fine-tuning 신호로 사용해 향후 인과 정렬을 강화한다.

- **Empirical Impact**: AgentLocate는 Who&When 및 Aegis-Bench의 두 벤치마크에서 에이전트 수준 정확도뿐 아니라, Who&When에서는 failure step까지 포함해 기존 failure localization 방법을 일관되게 능가한다. 또한 토큰 사용량과 실행 시간 측면에서 효율성을 유지하면서, 단일 Judge만 쓰는 경우보다 로컬라이제이션 정밀도가 개선됨을 보인다. 멀티에이전트 신뢰성 분석과 디버깅 자동화를 한 단계 진전시켜, 시스템 수준 장애의 원인 추적을 more verifiable하게 만드는 데 의미가 있다.



### A Reliability Assessment of LALM Audio Judges for Full-Duplex Voice Agents (https://arxiv.org/abs/2607.07985)
Comments:
          28 pages total (12 main body, 1 reference, 15 appendix). In main body: 2 diagrams, 3 table, 2 charts

- **Prior Approaches**: 기존 LALM-as-judge 연구는 주로 TTS 텍스트 음성(또는 단일 발화)에서 사람 MOS와의 강한 상관을 보고해 왔습니다. 하지만 기업 음성 에이전트의 full-duplex 대화처럼 채널·턴테이킹·중간 끊김·발화 부자연스러움이 섞인 원시 stereo waveform에 대해, 다수 인간 채점자를 기준으로 LALM의 ‘대체 가능성’을 검증한 연구는 드뭅니다. 또한 음성 평가에서는 크립펜도르프 α처럼 분포/천장 효과로 인해 신뢰도 지표가 해석을 왜곡할 수 있어, 단일 지표로 결론 내리기 어렵다는 점이 문제로 지적됩니다.

- **Core Contribution**: 이 논문은 raw stereo waveform(왼쪽: 에이전트, 오른쪽: 고객)을 입력으로 해 8개 차원 점수를 내는 Gemini 기반 오디오 judge가, 3명의 보정된 인간 채점자 기준과 어느 정도 일치하는지를 per-dimension으로 실증합니다. 특히 Gemini 2.5 Flash를 기준(ground-truth)으로 삼아 209개 full-duplex 세션(자연 152, 적대적 결함 57)에서 LALM을 사실상 4번째 rater로 취급해 신뢰도를 측정합니다. 그 결과는 ‘전체 성능’이 아니라 차원별로 배치 가능한지 여부를 판단하는 운영 프레임으로 제시됩니다.

- **Technical Challenges**: 핵심 난제는 (1) 대화 음성처럼 채널 동시성·발화 불연속성이 점수 분포를 바꾸고, (2) Likert 척도에서 천장 효과로 인해 chance-corrected 신뢰도 지표가 직관과 다르게 나올 수 있다는 점입니다. 논문은 rank agreement(Spearman rho, bootstrap 95% CI)와 절대 일치(simple within 1 point) 같은 서로 다른 렌즈를 병행하고, 결함(예: 클리핑, dead air, 잡음, 중간 잘림, phoneme-region overdubbing, 샘플레이트 다운/업)에서는 결함 민감도(recall)까지 분석해 해석의 공백을 줄였습니다. 또한 모델을 교체했을 때 순위는 유지돼도 calibration이 틀어질 수 있음을 확인하고, 이를 검증을 통해 보정 가능한 운영 과제로 정리합니다.

- **Empirical Impact**: Gemini 2.5 Flash는 8개 차원 중 5개에서 인간 간 일치 수준과 거의 비슷한 rank agreement(인간-인간 대비 rho 갭 ≤0.07)를 보였고, 7개 차원에서는 bootstrap CI 중첩으로 통계적 분리 가능성이 낮았습니다. 다만 speaking_rate_adaptation, overall_fidelity는 within-1 합의가 50% 미만이고, audio_clarity는 적대적 결함 상황에서 상대적 순위 분별이 약해(자연 대화에서는 상관이 거의 0에 수렴) ‘부분 보완형 배치’가 필요함을 드러냈습니다. Gemini 3.5 Flash는 전반적으로 개선된 반면, Gemini 3.1 Pro는 rank 상관은 비슷해도 일부 차원 점수가 평균에서 유의하게 벗어나 calibration 재검증이 필수임을 강조하며, 인간 채점의 비용(2차원 스팟체크 포함)이 LALM 워크로드 대비 대략 100배 수준임을 제시해 프로덕션 적용 근거를 제공합니다.



### 3100 Opinions on Code Review in an AI World: Building Causal Theory from Practitioner Discours (https://arxiv.org/abs/2607.07980)
- **Prior Approaches**: 코딩 에이전트가 PR을 작성하는 시대에, 기존 연구는 주로 레포지토리 마이닝으로 “작성 PR이 늘면 리뷰/머지/논의가 어떻게 변했는가” 같은 표면적 트렌드를 측정해 왔습니다. 하지만 관측 결과가 분석 선택에 따라 방향이 흔들리고, 왜 그런 현상이 나타나는지(메커니즘)는 충분히 설명하지 못했습니다. 또 현장에서는 코드 리뷰가 병목이 되는지, 여전히 인간 검토가 필요한지, 리뷰로 쌓이던 이해가 약화되는지에 대한 의견이 엇갈려 왔습니다.

- **Core Contribution**: 이 논문은 공개 GitHub 관측 트렌드가 왜 부호(증가/감소)를 유지하지 못하는지까지 포함해, “코딩 에이전트의 소프트웨어 영향은 코드 리뷰가 결정하는 제어 지점(control point)”이라는 설명 이론을 제시합니다. 핵심 주장은 AI가 리뷰 효과의 부호(sign)를 자동으로 고정하지 않으며, 팀이 보유한 인간의 전문성과 리뷰 프로세스 설계가 그 방향을 정한다는 점입니다. 또한 현장 논쟁의 서로 다른 입장을 구성요소로 명시해, “AI가 코드 리뷰를 바꾼다”를 반증 가능한 명제로 재구성합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 관측 트렌드에서 메커니즘을 복원하는 작업인데, 이를 위해 저자들은 38,709개의 회색문헌(엔지니어링 블로그, Reddit 스레드)에서 코드 리뷰에 실질적으로 초점을 둔 문서를 선별하고 LLM-assisted 파이프라인으로 3,100개를 코드화했습니다. 이렇게 추출한 신호를 바탕으로 26개 구성요소와 67개 관계(방향 64, 쟁점 3)를 갖는 인과 모델을 구축해, 조절변수(팀 역량·리뷰 구조 등)에 따라 경쟁 가설의 성립 조건이 달라짐을 설명합니다. 결과적으로 분석 선택에 따라 관측 부호가 뒤집히는 문제를 “왜 관측이 흔들릴 수밖에 없는가”라는 메커니즘 관점으로 흡수합니다.

- **Empirical Impact**: 실증적으로는 에이전트 작성 PR은 전반적으로 리뷰 빈도가 낮고 머지가 더 빠르며 논의가 덜한 경향이 관측되지만, 같은 관측이라도 분석 선택에 따라 트렌드 방향이 뒤집힙니다. 논문은 이 불안정성을 단순한 잡음으로 처리하지 않고, 제어 지점인 리뷰가 어떤 팀 설계와 전문성 환경에서 작동하는지로 해석 프레임을 제공합니다. 또한 LLM-assisted + 회색문헌 기반 이론 구축 방법론을 공개 구현까지 포함해 제시해, 향후 소프트웨어 엔지니어링 연구에서 재현·확장 가능한 템플릿이 될 수 있음을 보여줍니다.



### When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning (https://arxiv.org/abs/2607.07976)
- **Prior Approaches**: GRPO 같은 critic-free RLVR은 완료(trajectory) 단위로 계산한 advantage를 생성된 모든 토큰에 동일하게 배분한다. 이 방식은 토큰마다 문맥적 신뢰도가 다른데도 같은 양의 보상을 주어, 최종 정답이 맞더라도 중간에 잘못된 연속(token)이 양성 업데이트를 받는 문제를 만들 수 있다. 기존 대안들은 외부 신호(counterfactual, TD, 실행 피드백)나 토큰 entropy 같은 내재 신호로 토큰 중요도를 조정하지만, 의미적으로 “그 토큰이 해당 문맥에서 타당했는가”를 충분히 로컬 관점에서 분리하지 못한다.

- **Core Contribution**: 논문은 GRPO류에서 발생하는 실패 모드인 Positive-Credit Contamination을 규명한다. 특히 로컬 맥락에서 그럴듯하지 않은 implausible tail tokens가, 우연히 보상받은 완료 안에 포함되면 정상 토큰과 같은 양의 credit을 받아 잘못된 추론 습관이 누적 강화될 수 있다고 본다. 이를 해결하기 위해 TACO(Tail-Aware Credit calibratiOn)는 토큰별로 “implausible tail일 위험”을 추정해 양성 credit을 문맥적으로 보정한다.

- **Technical Challenges**: 핵심 난제는 토큰 단위로 ‘정답에 기여한 정도’를 직접 라벨링/오라클로 판정하는 것이 불가능하다는 점이다. TACO는 오라클 없이 forward 패스에서 관측 가능한 값만으로 tail-risk를 추정하며, sampled-token probability와 local entropy를 함께 써서 ‘낮은 확률이 곧 불확실성 탐색인지, 아니면 문맥적으로 타당하지 않은 꼬리인지’를 구분한다. 이후 위험도가 큰 토큰에는 양성 advantage에 대해 부드럽게 down-weight를 적용하되, 그라디언트 자체를 완전히 제거하진 않아 유용한 희귀 패턴의 누적은 보존한다.

- **Empirical Impact**: 세 모델(Qwen3 1.7B/4B, Qwen2.5-Math 7B)과 다수 벤치마크(수학 6개, 과학 OOD 2개)에서 TACO는 GRPO 스타일 베이스라인 대비 일관된 성능 향상을 보인다. 특히 long-horizon 학습(기본 300→600 step)에서도 GRPO는 정체/열화가 나타난 반면, TACO는 엔트로피가 더 매끄럽게 유지되며 성능이 계속 개선돼 학습 안정성이 개선됨을 보여준다. 정성 사례에서도 중간의 불필요한 꼬리 토큰들은 억제하면서도 후반의 올바른 풀이 구간은 credit을 유지해, 잘못된 로컬 행동 강화만 줄이는 효과가 확인된다.



### A Multi-cluster Boundary Learning Method for Out-of-Scope Intent Detection via MiniLM Embedding (https://arxiv.org/abs/2607.07974)
Comments:
          To submit

- **Prior Approaches**: 기존 OOS(out-of-scope) 의도 감지는 보통 알려진 의도를 다중 분류로 보고, OOS를 별도 클래스/점수/경계로 얹는 방식이 많았다. 이때 알려진 클래스 수가 늘면 OOS 샘플이 결정영역에 흡수되며 OOS 거절 정확도가 급격히 떨어지고, LLM-embedding 기반 방법은 큰 파라미터와 prompt 민감도로 실시간 배포가 어렵다는 한계가 있었다.

- **Core Contribution**: 논문은 MiniLM 임베딩을 쓰되, OOS 감지를 one-class classification 워크플로로 분해하는 멀티-클러스터 경계 학습을 제안한다. 또한 gate→router→expert의 3단 캐스케이드로 OOS 거절과 알려진 의도 다중 분류를 분리해, gate는 OOS 여부만 판단하고 후단은 in-scope 의도만 세분화한다.

- **Technical Challenges**: 핵심은 MiniLM 임베딩에서 한 의도가 단일 중심이 아니라 여러 클러스터를 형성한다는 전제를 경계학습으로 안정적으로 반영하는 것이다. 이를 위해 각 의도별 K-means 멀티 센트로이드를 만들고 클러스터마다 로컬 반경(대각 Mahalanobis 기반)을 추정해, 임베딩이 어떤 클러스터 로컬 영역 안에 들어오면 수락(known), 모두 바깥이면 OOS로 거절한다.

- **Empirical Impact**: CLINC150, StackOverflow, Banking77에서 제안 방법은 기준선 대비 OOS F1을 0.85%~17.12% 개선하며 SOTA 성능을 보였다. KIR(known intent ratio) 변화에도 OOS F1이 비교적 안정적이었고, ablation에서 gate에 쓰인 all-MiniLM-L6-v2 선택과 캐스케이드 구조가 성능에 필수적임을 확인했다.



### Beyond Thermal Imaging: Inferring Thermophysical Properties from Time-Resolved Thermal Observations (https://arxiv.org/abs/2607.07962)
Comments:
          31 pages, In submission

- **Prior Approaches**: 기존 열(thermal) 비전은 주로 인식·분할·이상 탐지에서 RGB와의 보완 모달리티로 열영상을 활용했으며, 물리적 상태를 직접 추정하기보다 시각적 특징을 결합하는 데 초점이 있었다. 3D 재구성 쪽으로는 NeRF/Gaussian splatting 등으로 열/온도 맵을 렌더링하는 연구가 늘었지만, 온도를 ‘정적 외관’으로 다루는 경우가 많아 열전달을 지배하는 열물성(예: thermal diffusivity)을 예측적으로 식별하기 어렵다. 반대로 inverse heat-transfer는 열 방정식을 최적화에 포함해 물리해석 가능성을 주지만, 단순 형상·통제된 경계조건·격자 기반 근사에 의존해 복잡한 3D 장면으로 확장되기 힘들다는 한계가 있었다.

- **Core Contribution**: ThermoField는 열 장면 재구성과 열물성 파라미터 추정을 differentiable heat-transfer simulation으로 통합해, 복잡한 3D에서 열물성 필드를 물리적으로 제약하며 추정한다. 열물성은 signed-distance-function 기반 표면(재구된 geometry) 위에서 정의되는 공간 가변 neural field로 표현되고, 열전달 물리(열방정식 잔차, 경계 교환, 시간 동역학)를 통해 시간별 관측 온도와 정합되도록 학습된다. 그 결과 geometry 복원, 공간적으로 변하는 thermal diffusivity 추정, 그리고 관측에 쓰지 않은 환경 조건에서의 thermal evolution 예측을 함께 수행한다.

- **Technical Challenges**: 핵심 기술 난제는 관측 온도만으로는 경계조건·형상·물성·환경 교환이 얽혀 underconstrained inverse problem이 된다는 물리적 식별성(identifiability) 문제다. ThermoField는 열카메라가 관측하는 것은 표면 온도라는 점에 맞춰, 재구된 표면을 계산 도메인으로 고정하고 열전달 시뮬레이터가 physical fields(thermal diffusivity 등)를 입력으로 받아 온도 시계열을 산출하도록 설계했다. 또한 finite-element 기반의 differentiable solver를 사용해 시뮬레이션-오차를 역전파 가능하게 만들고, diffusivity를 표면 제어점 + positional encoding이 있는 신경 표현으로 공간 분포까지 학습한다.

- **Empirical Impact**: 합성 데이터(간단/복잡 형상, 다양한 재료·경계조건)를 통해 ThermoField가 thermal diffusivity를 수치 정확도와 공간 응집성 관점에서 회복하며, 조건이 바뀌어도 재구성한 물성이 예측에 유의미하게 기여함을 보였다. 특히 훈련에서 지배적 물리 항이 유지되는 테스트 조건에서는 spatially varying field가 median baseline보다 큰 폭으로 MAE를 줄였지만, 열가열처럼 다른 물리 균형이 지배하는 경우에는 transfer가 약해지는 패턴도 확인했다. 반복 최적화 실험에서는 비볼록성에도 불구하고 해가 전반적으로 일관되게 수렴하는 경향과 함께, 관측 공간 오류가 낮아도 물성은 여러 해로 분산될 수 있음을 보여 ‘uncertainty-aware’한 해석이 필요하다는 실증적 근거를 제시했다.



### Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing (https://arxiv.org/abs/2607.07953)
Comments:
          20 pages, 6 figures, 8 tables. Code available at this https URL

- **Prior Approaches**: 자연어 모델의 시퀀스 믹싱에서 softmax attention은 전 문맥을 토큰 간 쌍으로 비교해 표현력이 크지만, 시퀀스 길이에 따라 비용이 2차로 커져 긴 컨텍스트 학습·추론에 병목이 된다. 이에 선행 연구들은 linear attention 계열로 전환해 선형 시간의 메모리 기반 업데이트를 추구해 왔고, 특히 DeltaNet류는 key-value 간섭을 줄이기 위해 delta-rule 잔차(잔차만 쓰기)를 도입해 정확도 손실을 완화했다. 다만 최근 변형들은 망각/제거 제어 방식(스칼라 decay, 채널별 decay, erase·write 분리)과 성능·처리량의 트레이드오프가 뚜렷하지만, 공통 관점에서 비교해 설계 선택의 의미를 정리한 연구는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 softmax attention과 DeltaNet, Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2를 공통의 recurrent-memory 표기로 통일해 표현력, memory decay, erase/write 제어, 학습 처리량, 구현 복잡도를 “무엇이 달라지는지” 기준으로 드러낸다. 또한 DeltaNet 스타일 메모리에서 깊이 방향 정보 희석을 완화하려는 cross-layer routing을 제안하고, Cross-Layer Error Residuals(CLER)와 Cross-Layer Value Routing(CLVR)을 체계적으로 비교한다. 그 결과 CLVR이 matched 설정에서 최종 검증 손실을 소폭 낮추면서도 linear-time 구조를 보존하는 경로임을 보인다.

- **Technical Challenges**: 공통 표기화의 핵심 기술적 도전은 서로 다른 주의·메모리 업데이트를 한 프레임으로 정렬해, 어떤 게 expressivity·forgetting·erase/write granularity를 좌우하는지 분해 가능하게 만드는 것이다. 논문은 delta-rule residual(메모리 예측 대비 잔차)을 중심으로 WW(연상 메모리)·rr(잔차)·다양한 게이트(α, erase, write)의 역할을 명시하고, Megatron 기반 학습 구현까지 연결해 재현 가능한 비교를 수행한다. cross-layer routing에서는 routed 신호를 “어디 공간에” 주입하느냐가 성능을 좌우하며, CLER의 value-target 주입은 공간 불일치로 이득이 없었던 반면 hidden residual stream 주입과 routed 신호 선택(error vs value)을 정교하게 조절한 CLVR이 개선을 냈다.

- **Empirical Impact**: 350M 파라미터 모델을 15B 토큰까지 학습하는 비교에서, Muon을 쓴 Kimi Delta Attention이 최종 검증 손실이 가장 낮았다. 반면 AdamW로 학습한 pure Gated DeltaNet 스택이 정규화된 학습 처리량에서 최고였고, hybrid 스택은 대체로 처리량 비용을 치르는 대신 손실을 개선하는 경향을 보였다. 또한 Muon은 같은 아키텍처 조건에서 AdamW 대비 최종 검증 손실을 일관되게 낮췄으며, cross-layer routing 중 CLVR은 DeltaNet과 Gated DeltaNet에서 최종 검증 손실을 소폭 개선했다(다만 보고된 결과는 학습 처리량/반복 시간 위주이며 별도 추론 속도 벤치마크는 제공되지 않음).



### path_boost: A Python Package for Interpretable Graph-Level Prediction using Path-Based Gradient Boosting (https://arxiv.org/abs/2607.07935)
Comments:
          27 pages, 4 figures, 5 tables. Code available at this https URL and on PyPI (path-boost)

- **Prior Approaches**: 그래프 데이터 예측에서는 GNN이 가장 널리 쓰이지만, 구조적 근거가 모델 내부에 분산돼 해석이 어렵다는 평가가 많습니다. 반면 graph kernel 계열은 성능이 강한 경우도 있으나, 학습 중에 표현이 적응하지 않아 해석성이 제한적입니다. 기존 path 기반 부스팅 연구는 등장했지만 anchor 노드 및 경로 탐색 설계에 따라 확장성과 해석 프레임이 달라진다는 한계가 있었습니다.

- **Core Contribution**: 본 논문은 그래프 입력에서 예측에 기여하는 ‘예측 가능한 labeled path’를 학습 중 자동으로 발견하고, 이를 경로 기반 누적(additive) 특징으로 설명하는 PathBoost를 구현한 open-source 패키지 path_boost를 제안합니다. GNN 대비 해석이 어렵다는 문제를, 어떤 부분구조가 예측을 만드는지 명시적으로 드러내는 경로 특성 모델로 완화합니다. 또한 multi-anchor parallel 구조와 상관관계 보정이 포함된 변수중요도 프레임워크를 추가해 경로 중요도 산정의 신뢰도를 높였습니다.

- **Technical Challenges**: 모든 가능한 경로를 전부 열거하면 조합 폭발이 발생하므로, 경로 탐색을 게으른(lazy) 방식으로 ‘선택된 경로만’ 점진적으로 확장해야 합니다. 이를 위해 PathBoost는 selector(부스팅 매트릭스(BM) 기반 빠른 경로 선택)와 base learner(속성까지 포함한 extended boosting matrix(EBM) 기반 피팅)를 분리해 계산 비용을 통제하고, 부스팅으로 weak learner를 앙상블해 성능을 끌어올립니다. 경로 길이에 따라 prefix가 중첩돼 중요도 해석이 흔들릴 수 있어, nested path 간 구조 의존성을 반영하는 correlation adjustment 및 정규화를 중요도 계산에 포함했습니다.

- **Empirical Impact**: 분자 그래프(원자=노드, 결합=엣지)에서 전이금속 화합물 물성 예측을 수행하고, 여섯 개 분자 데이터셋에서 PathBoost를 GNN(GINE) 및 graph kernel(WL + SVR)과 비교해 경험적으로 경쟁력 또는 우위를 보였습니다. 특히 변수중요도를 통해 어떤 경로(부분구조)가 예측에 영향을 줬는지 확인할 수 있어, 과학·공학 도메인에서 모델 신뢰성과 분석 효율성을 높이는 데 의미가 있습니다. 또한 scikit-learn 워크플로우 호환, 병렬 학습, PyPI/GitHub 공개로 재현성과 실사용성을 강화했습니다.



### Adversarial Decoys: Misdirecting Attention-Based Defenses in V (https://arxiv.org/abs/2607.07922)
- **Prior Approaches**: 기존 ViT 국소 공격(예: adversarial patch, adversarial object)은 특정 토큰/영역을 조작해 오분류를 유도하며, PatchFool·Attention-Fool처럼 공격이 attention을 적극 활용하는 경우도 있었다. 이에 대응해 ARMRO, RSA 같은 test-time 방어는 높은 attention(또는 이상 activation)을 받은 토큰을 의심해 마스킹·중화한다. 이런 접근은 attention-공격 효과 간의 강한 결합을 전제로 하며, 그 결과 공격이 attention을 피하거나(혹은 줄이거나) 적응하는 adaptive 전략이 어려워지는 문제가 제기됐다.

- **Core Contribution**: 본 논문은 adversarial decoys라는 독립 최적화 패치를 제안해, 방어가 억제하려는 attention 신호가 ‘진짜 공격 영역’이 아니라 공격자가 지정한 목표 토큰으로 향하게 만든다. 핵심은 오분류 유도와 방어 회피를 하나의 목표로 동시에 최적화하는 대신, 원래의 공격은 오분류만 담당하고 decoy는 방어가 쓰는 attention ranking을 조작하도록 기능을 분리한 점이다. decoy는 underlying attack과 독립적으로 학습되므로 attack-agnostic 방식으로 기존 localized attack 파이프라인에 쉽게 결합된다.

- **Technical Challenges**: 어려움은 “attention 크기를 낮추는 식의 회피”가 오히려 공격 효과를 약화시킬 수 있다는 점과, 단순히 높은 attention을 만들기만 해서는 방어가 선택하는 ranking에서 목표 토큰이 실제로 우선순위를 갖지 못할 수 있다는 점이다. 이를 위해 논문은 layer-wise로 (1) 목표 토큰의 received-attention을 전반적으로 끌어올리고, (2) Top-k 내부에서 목표 토큰이 비목표 경쟁 토큰보다 우위가 되도록 지배(dominance) 비율을 최적화하는 목적함수를 설계한다. 또한 decoy 최적화를 원래 공격과 분리해, 공격의 공격성분을 건드리지 않으면서도 방어가 탐지할 attention 경로만 바꾸는 구조를 만든다.

- **Empirical Impact**: ImageNet에서 DeiT, ViT 등 여러 아키텍처와 PatchFool 및 일반 adversarial patch 공격을 함께 실험한 결과, decoys는 높은 attention 점수를 진짜 공격 영역에서 멀리 돌리고 ARMRO의 suppression 마스크-공격 영역 겹침을 크게 줄였다. 그럼에도 불구하고 원래 공격의 유효성은 상당 부분 유지되어, 방어를 단독 적용할 때보다 defended accuracy가 더 크게 떨어지는 양상이 관찰됐다. 논문은 또한 attention magnitude만을 adversarial relevance의 지표로 쓰는 접근이 근본적으로 취약할 수 있음을 실험적으로 보여준다.



### Efficient Safety Alignment of Language Models via Latent Personality Traits (https://arxiv.org/abs/2607.07918)
Comments:
          15 pages, 6 figures. Accepted at COLM 2026

- **Prior Approaches**: 기존 안전 후처리는 유해 콘텐츠에 대한 명시적 거부(supervised refusal) 학습에 의존하는 경우가 많지만, 최근에는 그 정렬 행동이 jailbreak과 적대적 프롬프트에 취약하다는 실패 양상이 드러났다. Latent Adversarial Training(LAT)은 입력이 아닌 잠재공간에서 강건성을 키워 효과적이지만, 유해 프롬프트 기반 대규모 데이터가 필요하고 특정 위해 유형에 과적합되거나 성능 저하가 생길 수 있다. 또한 activation steering 계열은 단일 방향 개입으로 persona drift를 줄이지만 완전한 방어로 이어지지 않아 공격 성공률이 여전히 남는다.

- **Core Contribution**: 이 논문은 Latent Personality Alignment(LPA)로, 명시적 harm refusal 대신 심리측정(personpsychometrics) 문헌에서 가져온 해악 비지시(harm-agnostic) 성격 문장 66개만으로 latent adversarial training을 수행한다. 저자들은 성격 특성에 맞는 잠재 표현이 harm 회피와 공유되는 잠재 구조를 가진다고 보고, 그 부분공간을 adversarial하게 안정화하면 jailbreak이 악용하는 경로를 제약할 수 있다고 가설을 세운다. 특히 LPA는 학습 과정에서 유해 콘텐츠를 본 적이 없는데도 HarmBench에서 직접 요청과 5종 jailbreak에 대해 ASR을 거의 0에 가깝게 만든다고 주장한다.

- **Technical Challenges**: 핵심 난제는 ‘유해 프롬프트 없이도’ 위해 회피와 연결된 잠재 구조를 효과적으로 고정하는 방법을 설계하는 것이다. LPA는 성격 문장을 Big Five의 Conscientiousness, Agreeableness, Emotional Stability와 연결해 이 문장이 긍정/부정 표현일 때의 동의/비동의 출력으로 이진 목표를 구성하고, LAT처럼 잠재표현에 적대적 섭동을 주어 잘못된 완료(동의)로 유도한 뒤에도 올바른 비동의가 유지되도록 학습한다. 또한 학습 효율을 위해 작은 문장 세트에 고정 시스템 프롬프트(자기-진술 성격 평가 프레이밍)를 결합하고, 평가 시에는 최소 시스템 프롬프트로 유틸리티 영향 편향을 줄이도록 설계했다.

- **Empirical Impact**: 실험에서 LPA는 HarmBench ASR을 대부분 공격에서 0에 가깝게 만들며, 표준 유틸리티 벤치마크에서는 기준선 대비 성능 저하가 거의 없었다(다만 학습을 오래 지속하면 유틸리티가 감소해 조기 종료를 사용). LAT는 유해 프롬프트 수천 개와 큰 benign 보전 학습을 요구하는 반면, LPA는 66개 문장만으로 75배 가량 적은 학습 예시로 유사한 강건성을 달성하고 학습도 단일 GPU에서 수분 내 끝난다고 보고한다. ablation에서는 성격 특성의 안전 관련성, 문장-응답 라벨의 일관성, 특히 ‘부정(undesirable) 문장에 대해 불일치시키기’ 전략이 유틸리티 손상 없이 견고함을 만드는 데 중요함을 보여주며, Llama3-8B에서도 유사한 경향의 성능을 확인한다.



### Multimodal Unlearning Across Vision, Language, Video, and Audio: Survey of Methods, Datasets, and Benchmarks (https://arxiv.org/abs/2607.07907)
Comments:
          Accepted to ACL Findings 2026

- **Prior Approaches**: VLM·DM·LLM·AFM 같은 멀티모달 파운데이션 모델은 학습 데이터에서 온 민감정보·저작권·편향·안전 문제의 교차모달 연관을 그대로 내재할 수 있다. 삭제 요청이나 정책 업데이트 이후의 재학습은 비용·시간 문제로 현실성이 낮고, 지식이 공유 표현 전반에 분산돼 있어 targeted forgetting도 어렵다. 기존 접근은 주로 단일 모달 또는 제한적 설정에 집중돼, 모달 간 연관 제거와 성능 유지의 균형을 체계적으로 다루기 힘들다.

- **Core Contribution**: 이 논문은 멀티모달 unlearning을 비전·언어·오디오·비디오 전 범위에서 통합적으로 정리하는 “시스템 관점”의 설계를 제공한다. 또한 deletion 강도, retention(유지), efficiency(효율), reversibility(되돌림 가능성), robustness(강건성) 간 트레이드오프를 모달과 모델 구조에 걸쳐 비교 가능하도록 분류 체계를 제안한다. 이를 통해 실무 적용 관점의 선택 기준과 향후 연구 방향을 명확히 한다.

- **Technical Challenges**: 핵심 기술 난관은 공유된 표현 공간에 지식이 분산되어 있어 특정 모달/연관만 선택적으로 제거하면서도 전체 성능을 유지해야 한다는 점이다. 논문은 이러한 문제를 모달별 삭제 범위와 모델 아키텍처에 따른 학습·보정 방식의 차이로 구조화해 설명하고, 각 접근이 요구하는 데이터/추론 조건과 실패 모드를 함께 정리한다. 결과적으로 “어떤 삭제 목표를 어떤 시스템 구성으로 달성할지”를 비교 관점에서 재현 가능하게 만든다.

- **Empirical Impact**: 조사는 최근 연구 흐름과 등장 응용, 그리고 아직 미해결 과제를 함께 정리해 멀티모달 unlearning의 실험 설계와 평가 기준 수립에 도움을 준다. 특히 삭제 효력과 성능 유지, 효율, 되돌림 가능성, 견고성의 균형을 같은 축에서 비교하도록 해, 향후 벤치마크·리포팅 관행을 개선하는 데 의미가 있다. 또한 큐레이션된 저장소를 공개해 연구자들이 관련 방법을 빠르게 탐색하고 비교할 수 있도록 지원한다.



### Mechanistic Interpretability of LLM Jailbreaks via Internal Attribution Graphs (https://arxiv.org/abs/2607.07903)
- **Prior Approaches**: 기존 연구들은 LLM의 실패를 입력-출력 관찰(행동 수준)이나 attribution 기반 설명(어떤 특징이 중요했는지)으로 주로 규명해 왔다. adversarial training, alignment tuning, prompt filtering, retrieval augmentation 같은 방어도 많지만, 공격이 내부 추론을 어떻게 바꾸는지에 대한 통일된 기계적 관찰틀은 부족했다. 그 결과 안전오류의 원인이 되는 내부 메커니즘이 잘 설명되지 않는다는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 clean 프롬프트와 jailbreak 공격 프롬프트의 내부 추론을 각각 “paired internal computation graphs(쌍대 내부 연산 그래프)”로 만들고 정렬해 비교하는 진단 프레임워크를 제안한다. 그래프 정렬 결과로 안전 관련 요소의 억제(suppression), 공격 전용 특징의 출현(emergence), 연산 경로의 우회(rerouting) 같은 구조적 변화를 분해해 보여준다. 더 나아가 노드/경로/서브그래프 단위 causal intervention으로 어떤 구조가 공격 성공에 기여하는지 직접 검증해, descriptive attribution을 causal diagnosis로 전환하는 데 목적이 있다.

- **Technical Challenges**: 핵심 기술 난제는 공격 전후의 내부 추론을 “비교 가능한” 그래프 구조로 정식화하고, 그 변화가 잡음이 아니라 의미 있는 인과 신호인지 분리하는 것이다. 논문은 sparse transcoder 기반으로 노드 공간을 구성해 연산 그래프를 만들고, clean과 attacked 그래프를 feature similarity로 정렬해 invariant/suppressed/emergent 구조로 분해한다. 또한 graph deviation, safety suppression, attack emergence, 그리고 특히 path rerouting 같은 다중 수준 지표를 정의하고, 특정 부분구조에 대한 zero-ablation 등 개입을 통해 기여를 테스트한다.

- **Empirical Impact**: Llama-2-7B-chat-hf와 여러 jailbreak 벤치마크(총 30개 프롬프트 페어)에서 공격 성공과의 상관을 분석한 결과, 정적 그래프 지표나 억제/출현보다 “path rerouting magnitude”가 유의미하게 대응했다(r=0.461, p=0.010). 반면 safety suppression과 attack emergence는 유의한 연관이 없었고, 이는 jailbreak가 안전 요소를 단순히 끄는 방식이 아니라 추론 경로를 재배선해 우회한다는 해석을 뒷받침한다. 더불어 emergent feature에 대한 top-3 feature zero-ablation 개입은 4건 모두 실패(0% 완화)해, 공격이 분산된 경로/서브그래프에 걸쳐 redundancy로 설계돼 있어 선택적 제어가 어렵다는 점을 실증했다.



### Closed-Loop Dynamic Validator Node Scaling in Private Substrate Blockchains Using Takagi-Sugeno Fuzzy Inferenc (https://arxiv.org/abs/2607.07901)
Comments:
          9 pages, 5 figures

- **Prior Approaches**: 기존 private blockchain(예: Substrate)에서는 validator 설정이 고정되어 부하 변화에 즉시 대응하기 어렵다. 임계값 기반 스케일링은 경계 컷오프를 기준으로 단절적으로 결정을 내려 입력이 흔들릴 때 스케일링 진동(oscillation)이 커지는 문제가 있었다.

- **Core Contribution**: 이 논문은 블록 프로덕션 시간, 블록 크기, active node 수를 실시간으로 읽어 TS(Takagi-Sugeno) fuzzy inference로 연속형 efficiency score(0~100)와 스케일 권고(Scale Up/Maintain/Scale Down)를 함께 내는 폐루프 스케일러를 제안한다. 특히 membership function을 이론적 극값이 아니라 테스트베드에서 관측된 실제 동작 범위에 맞춰 경험적으로 재보정해, 배치 환경 차이에도 출력이 포화/평탄화되지 않게 했다.

- **Technical Challenges**: 핵심 과제는 (1) 여러 입력이 연동되는 네트워크 상태에서 적절한 스케일 결정을 ‘연속적’으로 만들고, (2) fuzzy 규칙을 실제 체인에서 동작 가능한 형태로 맞추는 것이다. 저자들은 3개 입력에 각각 3개 삼각 membership function을 두고 27-rule base를 완전 조합으로 구성했으며, product t-norm으로 규칙 발화도를 계산한 뒤 TS 특유의 빠른 가중 평균 방식으로 효율 점수와 액션 값을 산출했다.

- **Empirical Impact**: 10-node Substrate 네트워크에서 Queensland 스마트 워터 미터 해시 데이터를 사용해 4/7/10 active node 조건을 비교했으며, TS 컨트롤러가 구성 상태에 맞게 서로 다른 운영 프로파일을 만들고 상승·과잉 모두에서 같은 안정 평형으로 수렴함을 보였다. 임계값 기반 3종 대비 decision flip 수가 크게 줄었고(block time은 유사하거나 더 낮게), 스케일링 오실레이션이 감소하면서도 블록 프로덕션 시간 성능을 유지했다는 점에서 private validator 오케스트레이션의 실용성을 강화한다.



### How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism (https://arxiv.org/abs/2607.07891)
Comments:
          Submitted to Philosophy and Technology

- **Prior Approaches**: 기존 계산언어학은 언어를 ‘사물의 표상’에 붙는 고정 라벨, 즉 전달 코드로 보는 경향이 강했습니다. 해리스의 Integrationist linguistics는 이를 ‘language myth’(telementational/코드 모델)로 비판하며, 의미가 전달되기보다 상황 속에서 공동 행동을 향해 전망적으로 구성된다고 봅니다. 다만 Integrationism은 (1) 기호가 왜 미래 열림을 유지하는지의 구조 메커니즘, (2) 언어-비언어 기호활동의 연속성의 정밀한 틀, (3) 과거 통합이 남기는 archive의 구조를 설명하는 이론이 부족하다는 공백이 있습니다.

- **Core Contribution**: 이 논문은 Elan Barenholtz의 autogenerative theory(자기-생성성)를 해리스의 Integrationism에 ‘침해 없이 보강’할 수 있는 해법으로 제시합니다. 핵심 주장은 autogenerative 계정이 (i) 해리스가 중시한 prospective openness의 구조적 메커니즘, (ii) 언어와 비언어 간 semiotic continuity를 뒷받침하는 계산적 상관, (iii) 과거 통합의 잔재로서 archive가 어떤 모습인지와 참가자들이 이를 어떻게 활용하는지를 제공한다는 점입니다. 결과적으로 해리스의 ‘상황적 통합 행위가 우선’이라는 존재론적 원칙은 유지하면서, Integrationism이 못 채운 설명력을 추가로 확보합니다.

- **Technical Challenges**: challenge는 ‘고정된 지시/의미’ 없이도 왜 기호가 닫히지 않고 다음 선택지를 열어두는가를 구조적으로 보여주는 것입니다. 논문은 LLM의 corpus 통계가 각 토큰을 고정 주소 내용이 아닌 확률적 관계망으로 정의하며, 그 결과 조건이 조건을 낳는(conditions beget conditions) 형태의 계속 생성 구조가 prospective openness를 산출한다고 설명합니다. 또한 해리스가 주장한 반(反)모달 경계를 autogenerative 속성이 텍스트뿐 아니라 이미지 생성·캡셔닝에서도 나타나는 구조적 평행으로 연결하고, archive은 ‘지난 languaging의 잔차’가 남긴 확률적 경향(affordances)으로 작동해 현재의 통합을 제약·가능하게 한다고 정리합니다.

- **Empirical Impact**: 경험적 함의는 LLM이 ‘세계 기술’이 아니라 ‘이전 토큰들의 예측/생성 구조’를 강하게 학습한다는 점이, Integrationism의 기호-전망 개념과 정합적인 구조상관을 제공한다는 데 있습니다. 특히 multimodal 모델에서의 공통 autogenerative 패턴은 언어-비언어 semiotic continuity가 최소한 corpus 수준의 통계 구조에서도 관측된다는 시사점을 줍니다. 다만 인간의 뇌와 통합 행위는 판단·주의·예측의 생활사와 같은 추가 능력으로 ‘적절하게 맞출 수 있음’을 수행하며, 단순히 archive 기반 통계 생성만으로는 이 차이를 완전히 대체할 수 없다는 한계도 함께 강조해, AI 연구자들이 LLM 능력의 범위와 빈틈을 더 원리적으로 해석하게 합니다.



### Time-to-Collision Based Dynamic Obstacle Avoidance Using Pretrained Vision Models for Robots in Unstructured Environments (https://arxiv.org/abs/2607.07885)
Comments:
          9 pages, 8 figures

- **Prior Approaches**: 기존 로봇 장애물 회피는 end-to-end 학습형(Transformer 등)이 강력하지만, 로봇별 대규모 데이터 수집이 병목이 된다. 다른 대안은 시뮬레이션에서 RL로 학습한 정책을 전이하는 방식인데, sim-to-real 격차(비주얼·접촉 역학·환경 다양성 불일치)로 인해 실제 환경에서 성능과 안전성 보장이 약해진다. 또한 TTC(충돌까지 남은 시간)를 학습으로 바로 예측하면 데이터 다양성 의존성과 해석 가능성 한계가 남는다.

- **Core Contribution**: 이 논문은 sim-to-real 전이를 피하면서도 해석 가능한 비전 기반 동적 장애물 회피를 수행하는 데이터 효율적 방법을 제안한다. 핵심은 UniDepth(단안 metric depth)와 SuperPoint+SuperGlue(장기 키포인트 대응)를 사용해 3D 키포인트 궤적을 복원하고, 각 키포인트의 TTC를 기하적으로 계산한 뒤 최소 TTC 키포인트를 회피 방향에 반영하는 것이다. end-to-end 학습 대신 명시적 3D 구조·시간 정보를 사용해 일반화성과 설명 가능성을 함께 노린다.

- **Technical Challenges**: 기여를 실제로 만들 때의 가장 큰 난관은 단안 depth의 오차가 3D 투영과 번들 조정, 최종 TTC 추정까지 연쇄적으로 전파되는 점이다. 이를 줄이기 위해 5프레임 슬라이딩 윈도우에서 XM solver로 scaled bundle adjustment를 수행하고, (1) 충분히 길게 트래킹되지 못한 키포인트, (2) 너무 먼 깊이(20m 초과), (3) 프레임 간 3D 변위가 큰 이상치(2m 초과)를 제외·재초기화해 기하 제약의 신뢰도를 높였다. 이후 최소 TTC 키포인트의 CPA(closest point of approach)를 기준으로 지면(ground plane) 2D 모션 프리미티브를 선택한다.

- **Empirical Impact**: M3ED(spot-forest, spot-outdoor-day) 실데이터 평가에서 TTC 1초 미만 프레임 식별 precision 0.49, recall 0.38을 기록했으며, 위협 프레임을 올바르게 감지한 경우 회피 방향 일치율은 84%였다. 특히 서로 다른 물리 장애물 22개 중 20개에서 최소 한 프레임(TTC<1s)을 포착해 실제 회피 트리거 관점에서 유의미한 관찰 성능을 보였다. 또한 모델 학습을 없애고 하이퍼파라미터 튜닝에 74초 데이터만 사용함으로써, 대규모 로봇별 학습이 필요한 기존 접근 대비 데이터 효율성과 해석 가능성을 동시에 입증했다.



### Multi-agent Autoformalization of Tensor Network Theory (https://arxiv.org/abs/2607.07857)
Comments:
          5+2+33 pages; 3+3+11 figures; 6 tables; An accompanying blueprint document is available at this https URL

- **Prior Approaches**: 이전 연구들은 주로 단일 LLM 또는 도구 기반 보조로 Lean 등 정형화 작업을 돕는 방식에 의존해 왔다. 그러나 이 방식은 문헌-정형화 사이의 격차, 증명 의도(수학적 서술의 방향) 유지, 반복 실패의 누적 문제를 충분히 완화하지 못했다.

- **Core Contribution**: 본 논문은 이론물리학의 연구 수준 정형화를 위해 특화된 대형 언어모델 에이전트 팀과 “에이전트 주도 워크플로”를 제안한다. Mathlib/Lean에서 MPS의 fundamental theorem(행렬곱 상태의 기본정리) 자동 정형화를 데모로 수행했으며, 일부 명제에서는 표준 문헌 경로 밖의 새로운 증명 루트도 탐색했다.

- **Technical Challenges**: 주요 병목은 대규모 autoformalization에서 수학적 의도(intent)를 강제해 일관된 목표를 유지하는 문제였다. 이를 위해 blueprint(문헌-코드 중간층 메타데이터 문서)와 persistent memory(세션 간 실패·전략·반례 축적), 그리고 리뷰-수정(review-repair) 및 역할 분담(스카우트-후-증명 등) 패턴을 결합해 Lean 타입체킹 피드백을 반복적으로 수렴시켰다.

- **Empirical Impact**: TNLean(코드베이스)과 정형화 blueprint를 공개하며, 그 과정에서 기존에 없던 텐서 네트워크/양자정보 라이브러리(수백 개 수준의 Lean 성과)를 확장했다. 또한 1차원 symmetry-protected topological phases로의 정형화 확장까지 보여, 정형화 자동화가 이론물리 분야의 지식 축적과 재사용(문헌 대응 그래프, 갭 분석)에 미칠 의미를 실증적으로 제시했다.



### Kime-Representation Formulations of Three Open Problems in the Foundations of Classical Mechanics: Uncertainty, Invariant Entropy, and Directional Degrees of Freedom (https://arxiv.org/abs/2607.07851)
- **Prior Approaches**: 기존 고전 역학의 엔트로피 불확정성 원리는 주로 정준(표준) 변수의 단일 자유도에서 유도됐고, 미분 엔트로피 불변성과 한 쌍의 좌표에 의존했다. 또한 다자유도에서는 자유도 간 불확실성·상관이 어떻게 이동하는지, 그리고 “무상관 자유도들에 엔트로피가 균등 분배”되는 것이 하한인지 같은 질문이 여전히 부분적/추정 수준에 머물러 있었다.

- **Core Contribution**: 이 논문은 kime(복소 시간) 표현에서 세 가지 고전역학 기초 ‘오픈 문제’를 수학적으로 닫힌(closed-form) 형태로 재구성한다. 특히 (I) 비정준 변수쌍 및 다자유도 확장 불확정성, (II) 좌표변환 불변 엔트로피가 왜 ‘연속 물리량의 짝(pair)’을 요구하는지, (III) spin-1/2의 고전 아날로그인 방향성 자유도를 kime 원리로 정식화한다.

- **Technical Challenges**: 핵심 난제는 kime 위상(잠재 원형 확률변수)을 실제 역학의 행동-각도(action-angle) 변수와 정확히 연결해, Liouville(리우빌) 측도를 보존하는 불변량으로 엔트로피·피셔 정보·푸아송 괄호가 일관되게 등장하도록 만드는 것이다. 이를 위해 kime 콘과 1자유도 위상공간의 action-angle 차트를 정밀한 정확공감(symplectic) 동형으로 매핑하고, 그 결과 원통(S^1×R)에서 von Mises×Gaussian 극값가족의 날카로운 엔트로피 불확정성, 비정준 불확정성의 교정항(기댓값 Poisson bracket의 기하평균), 다자유도에 대한 Williamson normal form 기반 집합 경계, 그리고 kime 위상 확산 시 엔트로피 단조 증가를 증명한다.

- **Empirical Impact**: 이론적 증명 외에도, kime-phase tomography로 반복 실험 데이터에서 위상 법(phase law)과 원형 모멘트·원형 피셔 정보·symplectic 스펙트럼을 추정할 수 있어 ‘부분적 수치 검증’이 가능한 형태로 남은 오픈 항목들을 정확히 명시한다. 전체적으로 고전역학의 불확정성을 원형 통계와 정보기하 도구로 재배치함으로써, 좌표 불변성·비정준성·방향성 자유도까지 한 프레임에서 다루는 기반을 제공한다.



### Shift & Drift: A Zero-Shot Benchmark for Generalizable and Robust Autonomous Driving Motion Planning (https://arxiv.org/abs/2607.07844)
Comments:
          Accepted at 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **Prior Approaches**: 기존 closed-loop motion planner 평가는 nuPlan·CARLA처럼 학습과 유사한 지리/시맨틱 분포(i.i.d.)에서 이뤄져, map memorization이나 인과적 착각으로 인한 일반화 한계가 가려지는 문제가 지적돼 왔다. 또 소규모 perturbation을 견디더라도 시간에 누적되는 실행 오차(compounding error) 이후 회복 능력은 상대적으로 덜 다뤄졌고, 단순화된 동역학 모델은 안전 점수의 낙관적 상한을 만들 수 있다.

- **Core Contribution**: 이 논문은 두 축의 분포 변화를 동시에 겨냥하는 듀얼 트랙 벤치마크 Shift & Drift를 제안한다. 시맨틱 시프트 트랙은 DeepScenario Open 3D(DSC3D)를 nuPlan 시뮬레이션으로 변환해 zero-shot 평가를 가능하게 하고, 상태 분포 드리프트 트랙은 에고 차량의 동역학에 확률적 잡음을 주입해 실행 오차 누적 후 회복 성능을 정량화한다.

- **Technical Challenges**: 핵심 난제는 (1) occlusion-free 항공 데이터(DSC3D)를 nuPlan의 지도/로그 포맷에 일관되게 매핑하는 변환 파이프라인을 구축하고, (2) temporally correlated actuation noise(AWGN/OU)를 통해 실제 하드웨어적 드리프트를 재현하면서도 평가 가능한 시뮬레이션 조건을 만드는 것이다. 저자들은 OpenDRIVE 기반 의미·기하 프리미티브 추출, GPS/좌표계 정렬, 프레임 업샘플링, 그리고 닫힌-루프 재생 시 충돌/진입 위반/진행 부족 필터링 및 수동 보정으로 재현성을 확보했다.

- **Empirical Impact**: 실험 결과 imitation learning(PlanTF·Diffusion Planner 등)은 in-distribution에서는 강하지만 시맨틱 시프트와 특히 보행자 밀집 환경에서 크게 무너지고, temporally correlated OU 잡음에서는 드리프트가 지속되는 경향이 확인됐다. 반면 reinforcement-learning 기반 CaRL은 두 트랙 모두에서 성능 저하가 더 완만해 전반적으로 안전·진행 지표를 더 잘 유지하며, “모방 충실도 vs closed-loop 회복 탄력성” 간 경험적 트레이드오프를 뚜렷이 보여준다. 저자들은 1,182개 시나리오(다수 독일 도시+샌프란시스코) 규모의 스트레스 테스트를 통해 배포 신뢰성을 가늠하는 기준점이 될 수 있음을 제시한다.



### From Triggers to Emotions: A CPM-Grounded Appraisal Multi-Agent for Dynamic Emotional Evolution in Persona-Based Dialogu (https://arxiv.org/abs/2607.07824)
- **Prior Approaches**: 기존 persona 기반 롤플레이 대화는 인물의 감정을 고정된 persona 특성이나 prompt 수준 제어 신호로 다루는 경우가 많았다. 감정 대화 연구 역시 주로 사용자의 감정을 인식하고 공감/정서적 지지를 생성하는 데 초점이어서, 에이전트(캐릭터) 자신의 감정이 대화 자극에 따라 어떻게 변하는지 모델링은 상대적으로 부족했다. 그 결과 감정 표현이 과도하게 긍정적으로 증폭되거나 공식적인 패턴에 머물 수 있다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Component Process Model(CPM)을 감정의 ‘고정 라벨’이 아니라 ‘외부 사건에 대한 appraisal의 동적 과정’으로 보고, 이를 persona 기반 대화에 접목한다. CPM-MultiAgent는 캐릭터 감정을 고정 속성이 아닌 잠재 상태(latent emotion state)로 두고, 대화 트리거에 의해 지속적으로 재형성되도록 설계했다. 또한 Trigger 분석–CPM 기반 다차원 appraisal–감정 상태 업데이트–일관성 비평(critic)까지의 다단계 파이프라인으로 감정 진화의 해석 가능성과 일시적 일관성을 동시에 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 대화 턴에서 감정 변화를 유발하는 트리거를 안정적으로 추출하고, (2) 그 트리거를 캐릭터의 persona/목표/이전 정서 맥락에서 심리적으로 그럴듯하게 평가하며, (3) 업데이트된 감정 상태가 다음 응답 생성과 충돌하지 않게 보정하는 것이다. 이를 위해 Trigger Analyzer가 감정적 자극을 structured trigger로 만들고, 네 개의 CPM appraisal agents(관련성·함의·대처가능성·규범적 의미)가 peer review로 서로의 판단을 조정한 뒤 Integration Agent가 잠재 감정 상태를 업데이트한다. 마지막으로 Critic Agent가 CPM 충실성·맥락 근거·시간적 일관성을 점검하고 필요 시 수정(재생성)을 거치도록 했다.

- **Empirical Impact**: 실험에서는 의료/교육/고객서비스 롤플레이의 24개 트라이얼에서 감정 상태 업데이트 품질을 baseline 대비 비교했고, LLM-as-judge와 103명의 human evaluation 모두에서 CPM-MultiAgent가 전반적으로 가장 좋은 성능을 보였다. 특히 Appraisal Reasoning Quality와 Trigger Grounding, Temporal Consistency에서 두드러진 개선이 관찰되며, 감정 변화가 이전 상태와 트리거에 근거해 ‘설명 가능한 방식으로’ 일어난다는 점이 확인됐다. ablation 결과와 다양한 LLM backbone(GPT-5.4 계열, Qwen3.6-35B-A3B)에서의 견고성 실험도 구조화된 multi-agent 분해의 기여를 뒷받침한다.



### DreamCharacter-1: From 3D Generative Foundation Models to Product-Ready Character Generation (https://arxiv.org/abs/2607.07817)
Comments:
          Official Page: this https URL

- **Prior Approaches**: 기존 3D foundation 기반 생성은 단일 이미지나 multi-view, 자연어 프롬프트로 일반 물체 재구성과 생성을 잘 수행하지만, 제품 수준의 3D 캐릭터(정체성·고빈도 디테일·완전 텍스처·리깅 호환)까지는 동시에 만족하기 어렵다. 특히 geometry는 과도한 평활화·얇은 구조 누락·역면(뒤쪽) 비현실성, texture는 조명/가림/그림자에 얽힌 albedo 추정 실패와 교차 뷰 불일치, 보이지 않는 영역의 환각이 병목이 된다. 또한 고품질을 위해 backbone을 키우거나 인스턴스별 반복 최적화를 붙인 접근은 학습비용과 추론 지연이 커 산업 파이프라인 적용에 한계가 있다.

- **Core Contribution**: DreamCharacter-1은 pretrained 3D foundation 모델을 “전체 재학습”이 아니라 task별 post-adaptation(후처리 적응)으로 캘리브레이션해, 제품 준비형 3D 캐릭터 생성을 목표로 하는 경량 프레임워크를 제안한다. geometry post-training(기하 선호 기반 최적화), texture post-training(고해상 텍스처 합성과 self-occlusion 영역 복원), 그리고 inference acceleration을 단일 파이프라인으로 묶어 시각 품질과 구조적 견고성을 함께 끌어올린다. 결과적으로 정체성 일관성, 고빈도 표면 디테일, 완전하고 뷰-일관적인 텍스처, 리깅/애니메이션 호환성을 함께 노린다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 전역 구조는 무너지지 않으면서도 얇은 구조·날카로운 에지·고주파 디테일을 정체성에 맞게 복원하고, (2) 입력 이미지의 조명/가림 잡음을 분리해 뷰 간 일관된 텍스처를 만들며, (3) 이런 성능을 유지한 채 추론 지연을 산업 수준으로 낮추는 것이다. 논문은 coarse-to-fine SDF latent 기반 geometry 파이프라인과 이미지-정합 강화를 위한 구조화된(voxel) latent 정렬, multi-metric geometry reward로 강화학습을 결합해 기하 품질을 끌어올리고, texture는 multi-view 생성 후 sparse voxel 3D inpainting으로 가려진 영역을 채워 환각을 줄인다. 이어서 student-model distillation, 빠른 mesh extraction, 효율 attention(KV-cache 유사), 파이프라인 병렬화 등으로 대규모 배치를 위한 저지연 추론을 구현한다.

- **Empirical Impact**: 정량·정성 실험에서 DreamCharacter-1은 시각적으로 설득력 있고 구조적으로 견고한 3D 캐릭터 자산을 생성하며, 기존 character generation 방법들보다 일관되게 우수한 성능을 보인다고 보고한다. 특히 얇은 구조와 날카로운 디테일, 역면의 그럴듯한 기하, 그리고 self-occlusion/가시성 한계가 있는 부위의 텍스처 완성에서 강점을 나타낸다. 더불어 경량 post-adaptation과 추론 가속 설계를 통해 고품질 출력과 실사용성(리깅 호환·저지연)을 동시에 겨냥했다는 점에서, 학계 3D 생성 연구를 산업용 에셋 파이프라인으로 연결하는 데 의미가 크다.



### From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier (https://arxiv.org/abs/2607.07779)
- **Prior Approaches**: 기존 AI4Math는 Interactive Theorem Proving(ITP) 언어에서 높은 정확도의 정리 증명 생성에 강점을 보여왔지만, 대개 미리 정해진 직관·휴리스틱에 의존해 확장성이 제한된다. 또한 LLM의 자연어 추론은 강력하더라도 기계 검증 가능한 의미가 없어 환각 가능성이 남고, open-ended 연구 수준의 검증/자율 탐색을 수행하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 AI4Math를 ‘문제 해결자(solver)’가 아니라 ‘연구 에이전트(research agent)’로 전환해야 frontier 수학(새 정리 발견, 미해결 추측 해결)을 다룰 수 있다고 주장한다. 이를 위해 분야 전반을 체계적으로 정리하는 리뷰(데이터셋, auto-formalization, proof synthesis)를 제공하고, 현재 시스템이 연구 에이전트 역할을 수행하지 못하는 근본 격차를 로드맵 형태로 구조화해 제시한다.

- **Technical Challenges**: 연구 에이전트로의 전환을 가로막는 기술 과제로 데이터·평가 한계, 수학적 관계 구조의 부족, 탐색 자체의 장벽, 도구 생태계의 단절, 그리고 human-AI 협업의 미흡함이 제시된다. 논문은 이 문제를 각 축별로 어떤 방식의 데이터/모델링/워크플로 설계가 필요하며, 어떤 실험·검증 체계를 갖춰야 하는지에 대한 전략적 방향을 제안한다.

- **Empirical Impact**: open Erdős 문제 등 실제 frontier에 가까운 과제를 분석한 결과, 기존 성과는 문헌에 이미 존재하던 결과를 재발견하는 비중이 높고, Millennium Prize 수준의 ‘진짜로 새로운 아이디어’가 필요한 문제를 풀기엔 역량이 부족하다는 관찰을 뒷받침한다. 결론적으로 이 글은 향후 AI4Math 연구가 경쟁형 증명 성능을 넘어, 기계 검증을 기반으로 한 장기적 탐색과 협업형 연구 프로세스를 강화해야 한다는 방향성을 분야에 제시한다.



### Graph-Regularized Deep Learning for EEG-Based Emotion Recognition with Psychologically-Grounded Label Structur (https://arxiv.org/abs/2607.07773)
- **Prior Approaches**: EEG 감정 인식의 기존 딥러닝 학습은 cross-entropy로 감정 클래스를 서로 독립된 레이블로 보고, 오분류의 ‘심리적 거리’는 무시하는 경향이 있었다. CNN·GNN·transformer 등 아키텍처는 다양해졌지만, 감정 간 위계/구조를 학습 목표에 체계적으로 반영한 연구는 상대적으로 부족했다. label smoothing·계층 분류 같은 보완은 있었지만, 감정 토폴로지에 정렬된 방식으로 오분류 비용을 설계하는 일은 제한적이었다.

- **Core Contribution**: 이 논문은 감정을 Russell의 원형(circumplex) 기반 valence–arousal 공간에 배치하고, 심리적 근접성을 간선으로 갖는 ‘emotion graph’를 만들어 학습을 그래프 정규화로 유도한다. Graph Label Smoothing, Graph Laplacian의 commuting distance, Sliced Wasserstein 기반 정규화를 도입해 예측이 감정 토폴로지에서 벗어날수록 더 강하게 패널티를 주도록 설계했다. 또한 예측의 타당성을 정량화하는 Proximity Violation(PV) 지표를 제안해, 임상적으로 그럴듯한 오분류를 촉진하는 효과를 강조한다. 

- **Technical Challenges**: 핵심 기술 과제는 감정 ‘토폴로지’를 손실함수에 통합하되, 계산량과 최적화 안정성을 함께 만족해야 한다는 점이다. 저자들은 복잡도 순으로 GLS(직관적 소프트 레이블), Laplacian commuting distance(전역 연결성 반영, spectral 관점), Sliced Wasserstein(분포 변환 비용을 optimal transport로 근사)라는 세 가지 보완적 정규화를 구성했다. 아울러 정규화 가중치의 수동 튜닝 부담을 줄이기 위해 불확실성 기반 adaptive weighting도 함께 적용해, 백본별 학습 균형을 자동으로 맞추도록 했다.

- **Empirical Impact**: SEED-IV(4 classes)와 SEED-V(5 classes)에서 AudioTransformer·Conformer·DCGNN 세 백본 모두에 대해 아키텍처 불가지(architecture-agnostic) 개선을 확인했다. 최고 성능 기준 정확도는 최대 +5.42% 향상됐고, 심리적으로 그럴듯하지 않은 오분류는 최대 39% 감소했다. 특히 Conformer에서 Sliced Wasserstein(SW) 정규화가 정확도 +5.42% 및 PV 39% 감소로 두드러졌으며, subject-level F1도 다수 피험자에서 일관되게 개선되었다. 저자들은 UMAP 시각화와 PV 분석을 통해 정규화가 감정 간 근접성을 반영하는 더 구조화된 표현 공간을 학습하게 한다고 해석한다.



### Principled Analysis of Deep Reinforcement Learning Evaluation and Design Paradigms (https://arxiv.org/abs/2607.07769)
Comments:
          Published in AAAI 2026

- **Prior Approaches**: 강화학습은 딥 뉴럴 네트워크로 state-action value function을 근사하고, 이를 기반으로 게임/제어 문제에서 성과를 쌓아 왔다. 특히 듀얼링 네트워크, C51, 분포/퀀타일 계열(QRDQN, IQN) 같은 고용량 설계와 overestimation을 줄이려는 방법들이 주로 고데이터(약 2억 프레임) 환경에서 검증돼 왔다.

- **Core Contribution**: 본 논문은 강화학습 연구의 전형적인 평가·설계 패러다임이 “저데이터(sample-scarce) 구간의 성능 순위가 비선형 없이 그대로 비대칭(비동기) 확장되는” 것처럼 암묵적으로 가정해 왔다고 지적한다. 이 가정이 성능 프로파일과 sample-complexity(데이터 복잡도) 간의 관계를 왜곡하며, 그 결과 잘못된 결론과 연구 방향이 장기간 누적됐음을 이론과 실험으로 보여준다.

- **Technical Challenges**: 핵심 난제는 저데이터 구간과 비대(사실상) 확장된 비동기(점근) 구간에서 알고리즘 성능이 단조(monotone)하게 연결되지 않는 이유를, scaling·capacity·complexity 관점에서 정식화하는 것이다. 논문은 선형 함수 근사와 비정상(non-stationary) 정책/유한-horizon MDP 틀에서 이론적 성능(예: regret) 하한·상한을 통해 성능 순위가 구간에 따라 뒤집힐 수 있음을 증명하고, 이를 통해 암묵 가정의 붕괴를 설명한다.

- **Empirical Impact**: 대규모 실험에서는 Arcade Learning Environment에서 100K(저데이터) 벤치마크로 평가된 최신 알고리즘들이, 암묵적 단조 가정이 유발하는 편향 때문에 실제 고데이터(고프레임) 성능과 체계적으로 어긋날 수 있음을 확인한다. 결과적으로 본 연구는 deep reinforcement learning의 스케일링 법칙을 단순 “더 많이 학습하면 더 잘 된다”로 해석하기보다, 데이터 레짐에 따른 용량-오차 트레이드오프(비단조)를 고려해야 한다는 실무적 경고로 의미가 크다.



### A Transdiagnostic Space of Disorder Like Phenotypes in Reinforcement Learning Agents (https://arxiv.org/abs/2607.07753)
Comments:
          15 pages, 8 figures, 6 tables

- **Prior Approaches**: 기존 연구는 강화학습(RL) 에이전트에 1~2개 장애를 보상 shaping으로 인위적으로 유도한 뒤, 행동을 사후 라벨링하고 단일 실행 결과만 제시해 왔습니다. 또한 조작이 임상적 구성과 얼마나 일치하는지, 그리고 재현 가능성과 분산이 충분히 검증됐는지가 약점으로 지적됩니다. 무엇보다 reward를 위에서부터 손으로 바꾸는 방식은 ‘그냥 그 행동을 쉽게 만들었을 가능성’도 남깁니다.

- **Core Contribution**: 논문은 disorder modelling을 ‘인지적 appraisal 신호를 용량(dose)으로 조절하는 조작’으로 재정의하고, appraisal-guided PPO에서 7개 장애(불안, 조증, 강박 체크, 우울, 충동성, 중독, PTSD)를 각각 단일 knob으로 구현합니다. 각 knob은 computational psychiatry 설명에 근거해 설계됐고, 증상 평가는 preregistered assay로 미리 정해 임상 패러다임에 대응시켰습니다. 저자들은 특히 보상에 직접 쓰이지 않은 ‘emergent 효과’가 나타나는지에 초점을 맞춥니다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘장애 유도 조작이 단순한 회피 회로를 만드는 것’처럼 보일 수 있다는 점을 배제하고, 장애의 중증도에 따른 연속적 dose-response와 측정의 타당성을 동시에 확보하는 것이었습니다. 이를 위해 critic에 appraisal 벡터를 주입하고, 각 장애마다 의미 있는 appraisal/보상 경로에만 영향을 주도록 shaping 항을 분리했으며, 10 seed와 4종 control(기준선, critic noise, RND, shaping-free dose=0)을 두어 비교했습니다. 또한 knob 제거 후의 remit-versus-resist(수치왜곡/회피형 저항)와 두 knob 동시 작용의 비가산성까지 실험적으로 검증합니다.

- **Empirical Impact**: 1,000회 이상 러닝(95% confidence intervals 포함)에서 모든 장애는 통제군이 재현하지 못하는 단조적 dose-response를 보였고, 장애들은 reward–approach×threat–avoidance 2차원 affective space에 자기조직화됩니다(조증은 불안의 거울). knob 제거만으로는 일부 장애가 즉시 완화되지만(조증/체크/중독), 불안/ PTSD 같은 회피형은 잘 안 풀려 graded exposure-with-response-prevention으로 회복되어 치료 논리와 연결됩니다. 더 나아가 두 knob 조합은 비가산적 상호작용(예: 조증×충동성)을 만들어 comorbidity 예측 가능성을 제공하며, depression/addiction/anxiety knob의 특성은 MiniWorld 픽셀 환경에서 appraisal critic 없이도 cross-assay dissociation로 재현돼 프레임워크의 일반성을 시사합니다.



### Architecture Generalization with MetaNCA (https://arxiv.org/abs/2607.07743)
Comments:
          9 pages, 6 figures. To appear in the proceedings of the Artificial Life Conference (ALIFE 2026)

- **Prior Approaches**: 기존 딥러닝은 backpropagation 중심의 전역 최적화로 성능을 끌어올렸지만, 큰 모델 학습은 메모리·아키텍처 고정·데이터 요구량 같은 병목이 커 생물의 학습과 대비된다. 하이퍼네트워크와 weight-space 계열은 가중치를 생성할 수 있으나 보통 특정 타깃 아키텍처에 강하게 결합되며, plasticity/Local rule 기반은 업데이트는 지역적이지만 아키텍처 미학습 일반화가 제한적이었다. HyperNCA나 HyperNCA 계열의 NCA 기반 발전 모델도 local rule로 생성하긴 하지만, 타깃 가중치가 그래프의 국소 이웃 정보를 ‘엣지 단위로 반복 갱신’하는 방식까지는 결합되지 않아 교차 아키텍처 일반화가 약했다.

- **Core Contribution**: 이 논문은 Meta Neural Cellular Automata(MetaNCA)라는 프레임워크로, 계산 그래프에서 각 weight의 forward/backward 이웃 정보만으로 가중치를 반복 갱신하는 단일 local rule 네트워크를 meta-train한다. 학습된 rule 네트워크는 backpropagation 없이 다양한 MLP·CNN·ResNet 아키텍처의 task network 가중치를 생성해, 아키텍처-유연한 weight self-organization을 목표로 한다. 또한 meta-training 시 여러 아키텍처 분포를 함께 생성시키는 전략을 통해, 미본 아키텍처로의 전이를 체계적으로 강화한다.

- **Technical Challenges**: 핵심 난제는 “아키텍처에 고정되지 않으면서도” 각 weight가 필요한 이웃 신호를 그래프 단위로 안정적으로 요약하고, 그렇게 얻은 국소 rule이 대규모(수백만 파라미터) 네트워크의 성능 좋은 가중치로 수렴하도록 만드는 것이다. 이를 위해 논문은 local rule 네트워크에 Weight Transformer를 도입하고, 인접 weight와 per-weight hidden state를 forward/backward 이웃으로 구성한 뒤 linear attention(로터리 positional encoding 등)을 사용해 이웃 집계를 선형 비용으로 수행한다. 이후 perception vector를 MLP로 넣어 다음 step에서 각 weight와 hidden state를 갱신하며, 일부 weight에만 확률적으로 업데이트를 적용해 비동기성을 흉내 내고 학습 시 BPTT로 rule만 최적화한다.

- **Empirical Impact**: 실험에서 MetaNCA는 MNIST와 CIFAR-100에서 MLP·CNN·ResNet에 대해 최대 약 200만 파라미터급 task network 가중치를 생성하며, MNIST에서는 여러 미학습 아키텍처에서 90%대 성능 커버리지가 관찰된다. 특히 convolution 설정에서는 학습 커널 크기에 가까운 구간에서 최대 97%까지 도달하고, 멀어질수록 성능이 완만히 하락해 “특정 아키텍처 암기”가 아닌 일반적 생성 전략을 시사한다. CIFAR-100 ResNet에서는 학습된 채널 용량 범위 내 근처 아키텍처에 대해 최대 29.9%까지, baseline(Adam) 대비는 낮지만 아키텍처 확장 가능성을 보여주며, 학습된 rule 네트워크가 다양한 아키텍처 가족을 압축된 생성 규칙으로 대표한다는 해석을 뒷받침한다.



### Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE (https://arxiv.org/abs/2607.07740)
- **Prior Approaches**: 기존의 zero-shot context extension 방법들은 RoPE rescaling이나 grouping factor를 미리 고정해, 짧은 문맥 정확도와 긴 문맥 확장력 사이에서 한쪽을 희생하는 문제가 있었습니다. YaRN, Dynamic NTK, Self-Extend, DCA 등은 RoPE 회전이 학습 분포 밖으로 벗어나거나(위치 OOD) 소프트맥스 attention이 확산되며(softmax diffusion) 성능이 무너질 때를 각각 다른 방식으로 완화합니다.

- **Core Contribution**: Jet-Long은 튜닝 없이 “bifocal(양눈) 구조”를 구성하되, 긴 구간에서의 rescaling factor를 현재 시퀀스 길이에 맞춰 동적으로 바꿔 짧은 입력에서는 원 모델 동작을 그대로 재현하고 긴 입력에서는 깔끔하게 외삽합니다. 또한 로컬 RoPE-faithful window와 remote window를 inclusion–exclusion로 병합하고, generation 시에는 on-the-fly RoPE correction rotation을 적용해 KV cache를 건드리지 않도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 긴 길이로 갈수록 RoPE 회전이 학습 격자 밖으로 나가 성능이 흔들리는 position OOD 문제를 피하면서, decode 중에는 rescaling/그룹 크기 변화가 KV cache 재계산을 유발하지 않게 만드는 것입니다. Jet-Long은 remote 구간에서 group size를 길이에 따라 최소한으로 동적으로 선택하되, 캐시는 “원본 absolute position의 base key”만 저장하고 보정 rotation을 레지스터에서 즉시 구성해 스트리밍 생성이 끊기지 않게 했습니다.

- **Empirical Impact**: Qwen3-1.7B/4B/8B에서 128K까지 평가한 결과, Jet-Long은 RULER에서 최강 baseline 대비 +4.79/+2.18/+2.03pp 개선을 보였고 HELMET-RAG와 PG-19에서도 우수(또는 최상 수준) 성능을 달성했습니다. 또한 Jet-Nemotron(하이브리드 softmax/linear attention)으로의 전이와 w0에 대한 견고성도 확인했으며, CuTe fused CuTe kernel로 prefill 처리량은 FA2 대비 최대 1.39× 수준을 회복하면서 generation 오버헤드는 길이 전 구간에서 4% 이내로 유지했습니다.



### Collective Intelligence with Foundation Models (https://arxiv.org/abs/2607.07729)
Comments:
          Accepted as a book chapter in "Advances in Global Applied Artificial Intelligence" (G. A. Tsihrintzis, M. Virvou, N. G. Bourbakis, L. C. Jain, Eds.), authenticated version will be published in Springer series: Learning and Analytics in Intelligent Systems

- **Prior Approaches**: 기존 연구는 chain-of-thought, self-consistency처럼 단일 모델 내부에서 다양한 추론 경로를 만들거나 답을 선택하는 방식에 집중해 왔습니다. 또한 AutoGen, MetaGPT 같은 멀티에이전트 협업도 있었지만, 다수 에이전트의 ‘오류 탐지’와 ‘합의 품질’을 정량적으로 분해해 설명하기는 어려웠습니다. self-refinement이나 Constitutional AI는 자체 비판을 사용하지만, 같은 모델의 편향이 그대로 남아 맹점 교정이 제한될 수 있다는 문제가 제기됩니다.

- **Core Contribution**: 이 논문은 서로 다른 foundation model을 역할별로 배치한 멀티에이전트 추론 프레임워크를 제안합니다. Solver가 독립 초안을 만들고, Critic이 구조화된 오류 분석과 수정안을 제공하며, Aggregator가 합의해 최종 답을 생성합니다. 더불어 의미·수치·절차(단계별 reasoning)까지 함께 평가하는 Scoring 모듈을 도입해, 최종 정답뿐 아니라 중간 과정의 질을 드러냅니다.

- **Technical Challenges**: 기여를 실현하려면 (1) 에이전트 간 중복 없이 ‘실제로 다른 오류’를 포착할 수 있는 모델 조합, (2) 비평·수정·합의가 잡음이 아닌 논리 교정으로 이어지게 하는 구조화, (3) 단계별 추론 품질을 측정할 수 있는 평가 설계가 필요합니다. 저자들은 solver마다 독립 샘플 초안을 만들고, critic은 스타일이 아닌 논리/계산 오류를 겨냥한 구성 프롬프트로 1회 비판만 수행하며, aggregator는 다수 합의 근거를 우선하되 불일치 시 불확실성을 명시하도록 했습니다. 평가는 임베딩 유사도(semantic coherence), 수치 추출 기반 overlap, 그리고 reference의 단계와 대응시킨 step-wise accuracy로 수행했습니다.

- **Empirical Impact**: 8개 과학·수학 분야 벤치마크에서 ablation 결과, 전체 평균 성능은 구조와 중복 샘플링만으로는 소폭(예: 0.52→0.60/0.61) 개선되는데 그칩니다. 반면 heterogeneous(모델 다양성) 구성은 step-wise accuracy가 0.64로 크게 상승해 homogeneous(0.27~0.28) 대비 2.3배 수준의 개선을 보였고, 중간 단계의 정합성까지 함께 좋아졌습니다. 즉 “정답 맞힘”을 넘어 추론 과정의 설명가능성·감사가능성을 강화하는 데 모델 다양성이 핵심이라는 점을 실증적으로 입증해, 고신뢰 과학/산업 의사결정에 대한 멀티에이전트 설계 방향을 제시합니다.



### SHIFT: Survival Prediction from Incomplete and Heterogeneous Genomic Data (https://arxiv.org/abs/2607.07725)
Comments:
          18 pages, 2 figures

- **Prior Approaches**: 생존 예측을 위한 유전체 예측 모델은 기관 간 데이터 불일치(시퀀싱 패널·워크플로 차이) 때문에 배포 시 구조적 결측이 자주 발생한다. 기존 대응은 공통 유전자만 쓰거나 결측 환자를 제외하거나, 테스트 시 imputation으로 메우는 방식이 주로 쓰이지만 이는 정보 손실·선택 편향·생물학적 신호 왜곡 위험을 동반한다. 또한 결측 패턴이 기관마다 다르면 배포용 모델을 별도 구축해야 해 확장성이 떨어진다.

- **Core Contribution**: SHIFT는 결측을 고려한 Transformer 기반 survival 예측 모델로, test-time imputation 없이 불완전한 유전체 입력만으로 직접 예측한다. 각 유전체 변수를 별도 토큰으로 만들고 masked self-attention과 feature-availability mask로 관측된 입력에만 기반해 [CLS] 표현을 집계한다. 더 나아가 variable-rate feature masking(VRM)으로 학습 중 다양한 결측률을 노출해, 기관별 패널 차이 같은 heterogeneous missingness에 대한 강건성을 높인다.

- **Technical Challenges**: 핵심 과제는 생존 예측에서 예측 신호를 만들기 위해 ‘없는 유전체 블록’이 attention 상호작용을 오염시키지 않도록 설계하는 것이다. SHIFT는 결측 특징을 0으로 채우되 attention에서 완전히 배제하는 key-padding mask로 표현 학습을 관측 부분에 제한하고, positional embedding 없이 feature별 임베딩으로 유전자 정체성을 흡수한다. 또한 VRM으로 epoch마다 다른 마스킹 수준을 주입해 학습-배포 결측 패턴 불일치 문제를 완화한다.

- **Empirical Impact**: glioblastoma(GBM)와 lung squamous cell carcinoma(LUSC)에서 외부 코호트 검증을 수행했으며, 특히 패널 미스매치가 심한 LUSC US 코호트에서 SHIFT는 imputation 기반 기준선 대비 유리한 성능을 보였다. 197개 중 88.8%가 구조적으로 결측인 설정에서도 SHIFT-VRM은 테스트 시 imputation 없이 최고 앙상블 C-index를 달성해 결측 인지 모델링의 실용성을 입증한다. 더불어 학습 단계에서 결측이 큰 ‘부분 관측 코호트’를 그대로 포함해도 외부 성능이 향상될 수 있음을 보여, multi-center 정밀의료에서 배제 중심 전략을 완화하는 메시지를 제공한다.



### Omni-Sleep: A Sleep Foundation Model via Hierarchical Contrastive Learning of CNS--ANS Dynamic (https://arxiv.org/abs/2607.07720)
- **Prior Approaches**: 기존 수면 파운데이션 모델들은 EEG·EOG·EMG·ECG·호흡 같은 PSG의 이질적 신호를 토폴로지(생리적 구조) 고려 없이 하나의 표현 공간에 평평하게(fusion) 맞추는 경우가 많았습니다. 그 결과 CNS(중추신경계)와 ANS(자율신경계)가 서로 다른 생리적 manifold를 가지면서도 시기·수면단계에 따라 동기화가 달라진다는 점을 충분히 반영하지 못한다는 한계가 지적됩니다. 또한 센서 드롭아웃이나 도메인 시프트 환경에서 성능이 흔들릴 위험이 커졌습니다.

- **Core Contribution**: Omni-Sleep은 CNS/ANS 분할을 생리적 prior로 삼아, 토폴로지 제약이 반영된 수면 표현 학습을 수행하는 새로운 sleep foundation model입니다. 모델은 (1) 동일 subsystem 내에서의 intra-system consistency, (2) CNS–ANS 사이의 inter-system synchronization, (3) 긴 시간 범위의 masked temporal modeling을 결합해 뇌-몸 결합의 계층 구조를 학습합니다. 이로써 라벨이 적거나 일부 모달리티가 빠진 상황에서도 더 일반화 가능한 표현을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 과제는 서로 다른 생리적 계층(CNS vs ANS)을 한 덩어리로 평균내지 않으면서도, 수면단계 전환에 걸친 장기 동학까지 안정적으로 학습하는 것이었습니다. Omni-Sleep은 subsystem 내부에서는 다른 모달리티의 epoch 단위 평균을 positive로 두는 계층적 contrastive objective로 일관성을 강제하고, subsystem 요약을 대칭 InfoNCE로 정렬해 CNS–ANS coupling을 명시적으로 모델링합니다. 여기에 1.5시간 수준의 long window에서 epoch 임베딩을 마스킹해 예측하는 latent-space masked temporal modeling으로 로컬 형태와 글로벌(밤 전체) 동기까지 함께 유도합니다.

- **Empirical Impact**: Omni-Sleep은 SHHS·WSC·MESA의 10만 시간+ 다기관 PSG로 사전학습한 뒤, 수면 단계(sleep staging)와 다질환 분류에서 여러 OOD 데이터셋 및 모달리티 제거 실험 전반에 걸쳐 기존 foundation-model 대비 성능이 향상됐습니다. 특히 선형 프로빙/파인튜닝(소량 라벨) 모두에서 label efficiency가 개선됐고, ANS-only나 EEG-only처럼 센서가 빠진 설정에서도 견고함을 보이며 일반화 가치를 입증했습니다. 결과적으로 생리적 계층(physiological hierarchy)을 반영한 토폴로지 제약 학습이 임상 배치 환경에서도 활용 가능한 수면 표현 학습의 길을 열었다는 점에서 의미가 큽니다.



### ReCoLoRA: Spectrum-Aware Recursive Consolidation for Continual LLM Fine-Tuning (https://arxiv.org/abs/2607.07719)
- **Prior Approaches**: LoRA 계열 PEFT는 백본 가중치를 고정하고 저랭크 업데이트만 학습해 효율을 높이지만, 태스크가 순차로 들어오면 같은 frozen weight 위에 업데이트를 계속 쌓으면서 이전 지식이 덮어써지는 문제가 생긴다. AdaLoRA, PiSSA, DoRA 같은 변형은 rank 배분·초기화·표현력을 개선하지만, 덮어쓰기 자체는 근본적으로 해결하지 못한다. O-LoRA처럼 간섭을 줄이려는 정형화된 정규화/부분공간 분리는 효과적일 수 있으나, 실용적 운용에서는 여전히 태스크 간 정보 재사용과 충돌 제어가 과제로 남는다.

- **Core Contribution**: 이 논문은 ReCoLoRA(Recursive Consolidation of Low-Rank Adapters)로 스펙트럼 기반(랜덤화 SVD) 초기화와 계층별 유효 rank(엘보 기준)를 도입해, 업데이트가 먼저 ‘주요 성분’ 방향을 사용하도록 설계한다. 더 핵심은 재귀적 통합(recursive consolidation)인데, 매 태스크 종료 후 원본 W0에 계속 쌓지 않고 현재까지의 effective weight를 다시 분해해 residual(고정 잔차)·slow principal(천천히 학습)·fresh fast adapter로 재구성함으로써 다음 태스크가 이전 태스크를 이미 흡수한 모델에서 출발하게 만든다. 또한 ReCoLoRA-TaskBank은 태스크별 독립 브랜치를 학습하고 oracle 라우팅으로 평가해, ‘오버라이트가 없는 경우’에 대한 상한을 제시한다.

- **Technical Challenges**: 연속 파인튜닝에서는 stability-plasticity 딜레마 때문에 새 태스크 적응성과 이전 성능 유지가 동시에 흔들리는데, 단순히 rank를 고정하거나 파라미터 공간에 앵커링하는 방식은 보호가 강하면 가소성이 죽고 약하면 기존 동작이 드리프트하는 것으로 나타났다. ReCoLoRA는 이를 피하기 위해 (1) 저랭크를 무작위로 찾기보다 pretrained weight 스펙트럼에서 principal 방향을 먼저 제공하고, (2) staged residual recovery로 잔차 용량은 필요할 때만 점진적으로 여는 학습 스케줄을 결합한다. 그리고 태스크 경계마다 현재 effective weight를 재분해해 slow principal을 보존하는 구조를 만들면서, 각 단계에서 principal/residual의 역할과 학습 속도를 분리해 간섭을 줄이도록 구현했다.

- **Empirical Impact**: 연속 GLUE 6-태스크 설정에서 ReCoLoRA는 4개 7~8B 백본 중 3개에서 최종 평균 점수(final average)가 rank-swept LoRA 계열·PiSSA·AdaLoRA·DoRA 및 관련 비교군을 앞섰고, 특히 학습 파라미터를 더 적게 쓰는 이점도 보고됐다. ReCoLoRA-TaskBank은 oracle 라우팅 하에서 오버라이트를 구조적으로 제거해 높은 상한 성능(예: Qwen3-8B 평균 forgetting 0에 근접)을 보였다. 다만 Llama-3.1-8B-Instruct에서는 LoRA rank 64가 더 좋은 결과를 보인 음성 사례도 있어, 스펙트럼/스케줄 민감도가 완전히 사라지지는 않는 것으로 해석된다.



### LLT: Local Linear Transformer for PDE Operator Learning (https://arxiv.org/abs/2607.07718)
- **Prior Approaches**: Transformer 기반 neural operator는 attention으로 도메인 전역의 장거리 의존성을 학습할 수 있어 PDE 연산자 학습에 유망하지만, 표준 self-attention은 노드 수 N에 대해 O(N^2)로 비용이 커 고해상도 격자에서 병목이 된다. 또한 일반 attention은 기본적으로 국소 상호작용에 대한 귀납적 편향이 약해, 스텐실/요소 기반 수치해석이 활용하는 ‘local structure’를 그대로 반영하기 어렵다. 기존에는 linear attention, 그래프/스펙트럴 연산, U-Net+Transformer 하이브리드 등으로 이를 완화하려 했지만, 연산 비용과 국소성 편향을 함께 만족시키는 설계가 여전히 과제로 남아 있었다.

- **Core Contribution**: 이 논문은 PDE 연산자 학습을 위해 Local Linear Transformer(LLT)를 제안한다. LLT는 전역 커뮤니케이션은 kernelized linear attention으로, 국소 처리는 별도의 local spatial mixing 경로로 분리해 두 한계를 동시에 겨냥한다. 또한 좌표 및 geometry 정보를 거리 인코딩과 Fourier 좌표 임베딩으로 주입하고, skip-connected decoder로 좌표 신호가 희석되는 문제를 줄인다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 전역 의존성을 유지하면서도 attention 계산을 노드 수에 대해 거의 선형으로 낮추고, (2) PDE 해가 보이는 국소적 구조(특히 근방 업데이트)를 모델에 ‘명시적으로’ 주입하는 것이었다. LLT는 global path에서 softmax를 양의 feature map(elu-based)로 대체해 key-value 집계를 질의마다 재사용함으로써 표준 attention 대비 효율을 확보하고, structured mesh에서는 depthwise-separable convolution, unstructured mesh에서는 반경 이웃 그래프 기반의 masked local attention으로 국소 편향을 구현했다. 더불어 거리 기준점(reference set)과 좌표 임베딩을 함께 사용해 격자/메시가 달라져도 동일한 의미의 공간 정보를 제공하도록 설계했다.

- **Empirical Impact**: LLT는 탄성/소성, 에어포일, 파이프 유동, Darcy 흐름 등 5개 PDE 벤치마크에서 유사한 neural-operator·Transformer 계열 baseline들과 비교해 상대 L2 오차가 경쟁적이거나 더 낮게 나왔다. 특히 Transolver와 같은 경우와 동일한 structured 이산화 조건에서는 학습 반복당 벽시계 시간이 1.8~2.5배 감소해 정확도-효율 균형을 실증했다. 더 나아가 3D 자동차 공력 데이터(비정형 메시, 샘플당 32,186 포인트)까지 확장해 큰 unstructured 설정에서도 작동 가능성과 실용성을 보여주며, 다양한 discretization/메시 타입에서의 안정적인 연산자 근사를 시사한다.



### Towards the Explainability of Temporal Graph Networks via Memory Backtracking and Topological Attribution (https://arxiv.org/abs/2607.07716)
Comments:
          ICML 2026 Spotlight

- **Prior Approaches**: 기존 설명 방법은 정적 GNN에서 중요 엣지·노드·서브그래프를 뽑는 데 초점이 있었고, Temporal Graph Networks(TGNs)용 방법들도 일부 나왔지만 대체로 마지막 메모리 벡터를 고정해 해석합니다. 이 방식은 메모리 모듈이 과거 이벤트를 누적·갱신하는 과정을 설명에서 배제해, 예측이 어떤 ‘기억’에서 비롯됐는지 놓치기 쉽습니다. 결과적으로 속도는 나와도 귀속이 부정확해지고 비충실한(explanation unfaithful) 해석으로 이어진다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 TGNs의 예측을 ‘이웃의 공간적 영향’과 ‘노드 메모리의 시간적 영향’으로 나눠 사건(event) 단위에 귀속(attribution)하는 프레임워크를 제안합니다. 이를 위해 Topology attribution tree로 이웃 이벤트와 그 메모리 벡터의 기여를 추적하고, Memory backtracking tree로 과거 이벤트가 현재 메모리 벡터를 어떻게 형성했는지 거슬러 올라갑니다. 또한 Layer-wise Relevance Propagation(LRP)을 TGN에 적용해, 사건들의 기여 합이 로그릿(logits)과 보존되도록 설계합니다.

- **Technical Challenges**: 핵심 난제는 (1) TGNs의 메모리 모듈 업데이트까지 포함해 사건 기여를 분해해야 한다는 점과 (2) 로그릿 기반 보존성을 확보하면서도 실제 ‘중요 사건 top-k’ 선정을 신뢰성 있게 해야 한다는 점입니다. 저자들은 LRP로 로그릿에 대한 기여 보존(conservation)을 만족시키고, logits-확률 변환이 비선형이라 top-k가 불충실할 수 있다는 문제를 KL divergence 기반 최적화 목적함수로 보완합니다. 그 결과, 이웃 이벤트-메모리-과거 이벤트까지 계층적으로 기여를 계산하면서도 해석의 충실성을 높입니다.

- **Empirical Impact**: node property prediction, link prediction, graph classification까지 총 9개 temporal graph 데이터셋에서 제안 방법은 충실한 설명을 제공하며 4개 SOTA 기준선을 전반적으로 능가하는 성능을 보였습니다. 특히 Fidelity 측정에서 두 번째로 좋은 베이스라인 대비 통계적 유의성이 높은 비율로 확인되며(각 지표에 대해 다수 케이스에서 t-test 유의), 설명 신뢰성을 실험적으로 뒷받침합니다. 메모리 모듈을 직접 추적하는 접근이 TGNs의 ‘어떤 과거가 현재 예측을 만들었는가’를 이해하는 데 중요한 의미를 갖는다는 점이 강조됩니다.



### ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modeling (https://arxiv.org/abs/2607.05583)
Comments:
          8 Pages. Accepted at ICANN 2026

- **Prior Approaches**: 기존 장문 언어모델링은 transformer의 self-attention을 중심으로 발전해 왔으며, 많은 변형들도 attention을 직접 대체하기보다 계산 효율을 손보는 데 집중해 왔습니다. 다만 context 길이가 길어질수록 attention의 계산/메모리 비용이 커져 효율이 급격히 떨어지는 문제가 남아 있습니다. linear·kernelized attention, Hyena/S4/Mamba 같은 state-space 계열이 quadratic attention 없이도 성능을 낼 수 있음을 보여주지만, 여전히 attention 기반 계열의 연산적 계보 안에 머무는 경우가 많습니다.

- **Core Contribution**: 이 논문은 self-attention을 causal resonant field mixing으로 완전히 교체한 ResonatorLM을 제안합니다. 토큰열을 단일 driven 1D latent field로 보고, 위치 간 정보 전달은 damped resonator의 causal 함수로 구현해 장문에서도 효율을 유지하려는 전략입니다. 학습(그리고 prefill)은 FFT 기반 causal convolution으로 병렬 경로를 살리고, 생성(decode)은 고정 크기 recurrent state로 캐시 성장을 막도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) attention과 동등한 정보 전파 능력을 유지하면서 (2) 학습·prefill과 decode에서 동작 형태를 효율적으로 분리하는 동시에 (3) causal성과 수치 안정성을 확보하는 것입니다. 저자들은 각 head에 damped oscillatory 커널을 학습하고, training/prefill에는 FFT로 O(n log n) 혼합을 수행하며 decode에는 동일한 커널 계열을 고정 크기 상태 업데이트로 대응시켰습니다. 또한 반감기(half-life) 범위와 prefix에 대한 causality test, 상태 에너지 decay 등으로 causal 구조와 시간 스케일을 점검해 모델이 한쪽으로 붕괴하지 않음을 확인했습니다.

- **Empirical Impact**: 6M 매치드 설정에서 ResonatorLM은 32K 토큰까지 decode 속도가 표준 최적 transformer 대비 6.47x 빨라졌고, WikiText의 정확도도 55.32%에서 61.31%로 상승했습니다. 95% 신뢰구간 기준 perplexity와 accuracy 모두에서 반복 실험(여러 seed)으로 일관된 개선이 관찰되며, local path 제거나 head coupling 변경 같은 일부 ablation에서도 성능 변동이 크지 않았습니다. 더 나아가 kernel-only scaling benchmark에서는 8K에서 440.29x, 32K에서 575.86x 같은 큰 알고리즘적 속도 이득을 보고해, 장문 효율에서의 의미 있는 함의를 강조했습니다.



New uploads on arXiv(cs.RO)

### DexVerse: A Modular Benchmark for Multi-Task, Multi-Embodiment Dexterous Manipulation (https://arxiv.org/abs/2607.08751)
- **Prior Approaches**: 기존 일반 목적 로봇 정책용 벤치마크는 주로 그리퍼 중심의 단순 조작에 치우쳐 접촉이 많은 고차원 손-팔 제어의 난도를 충분히 반영하지 못했습니다. 또한 DexMimicGen, Bi-DexHands, DexJoCo 같은 덱스터리티/바이매뉴얼 관련 벤치마크도 작업 범위, 임바디먼트 커버리지, 시각 변형 통제, 데모 다양성 중 일부가 제한적이어서 교차 작업·교차 임바디먼트 일반화 비교가 어렵다는 평가가 나왔습니다. 결과적으로 연구자들이 접촉 모드, 시각 조건, 임바디먼트 변화의 동시 영향을 한 프레임에서 검증하기 어려웠습니다.

- **Core Contribution**: DexVerse는 범용 덱스터리티를 겨냥해 작업 다양성(총 100개), 다중 팔-손 임바디먼트(3개 arm, 6개 dexterous hand), 시각 변형(텍스처·조명·카메라 뷰포인트 등)과 텔레오퍼레이션 데모를 한데 묶은 모듈형 대규모 벤치마크입니다. VR 기반 텔레오퍼레이션으로 3,180개의 전문가 데모를 수집하고, proprioceptive부터 RGB/Depth/point-cloud/state 관측까지 멀티모달 관측을 동기화해 통일된 평가를 가능하게 합니다. 또한 Diffusion Policy, DP3, OpenVLA, π0.5 같은 대표 방법을 19개 태스크에서 벤치마킹하며 “현재 접근이 얼마나 안 되는지”를 카테고리별로 드러내는 테스트베드를 제공합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 접촉이 풍부한 고정밀 덱스터리티를 장기·다단계 작업까지 포함해 재현하면서 (2) 임바디먼트/시각 변화를 동시에 통제하는 벤치마크 설계였습니다. DexVerse는 Isaac Lab 기반의 매니저형 환경 인터페이스와 구성(config) 중심 템플릿으로 작업-로봇-관측-성공조건을 명시적으로 분리했으며, 시각 변형과 비시각 변형을 독립/결합 가능하게 만들어 “무엇이 달라졌는지”를 실험적으로 다룰 수 있게 했습니다. 데모 수집에서도 임바디먼트 적응형 텔레오퍼레이션(Arm IK 추종 + dex-retargeting) 파이프라인을 구축해 손/팔이 바뀌어도 재현 가능한 데모 생성을 목표로 했습니다.

- **Empirical Impact**: 실험 결과 DexVerse는 여전히 매우 어려운 과제로 나타났고, 최상위 베이스라인의 평균 온라인 success rate는 34% 수준에 머물렀습니다. 태스크 카테고리마다 승자가 갈리며(예: pick-and-lift, tool use, precision contact 등) 단일 모델이 전반을 일관되게 지배하지 못해, 임바디먼트·시각·접촉 특성 전반을 아우르는 범용 학습의 필요성이 재확인됐습니다. 특히 push/insert 및 서브-센티미터 정렬이 필요한 정밀 접촉 과제는 여러 방법이 사실상 0에 가까운 성능을 보여, 접촉 추론·힘/폐루프 보정 같은 다음 연구 여지가 크게 열려 있음을 시사합니다.



### ContactMimic: Humanoid Object Interaction via Contact Contro (https://arxiv.org/abs/2607.08742)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 휴머노이드 whole-body 트래킹·로코-매니퓰레이션은 주로 keypoint(또는 joint target) 궤적을 맞추는 데 집중해, 실제 작업에서 결정적인 contact은 부수적 결과로 취급되는 경우가 많았습니다. 그 결과 의자에 앉기/기댐, 화이트보드 닦기 같은 “접촉 자체가 의미를 규정”하는 태스크에서는 contact을 원하는 때에 만들거나 억제하는 제어가 잘 되지 않았고, BeyondMimic처럼 keypoint-only 접근은 task-specific reward로 미세조정해야 성능을 얻는 한계가 있었습니다. 

- **Core Contribution**: CONTACTMIMIC은 keypoint trajectory뿐 아니라 시점별 part-level binary contact command를 함께 추적/조건화하는 학습 프레임워크를 제안합니다. 학습 시 접촉 명령을 반영하는 contact-aware reward를 사용하고, 배치된 입력으로 인해 로봇이 접촉 행동을 keypoint 기하와 분리해 “원하면 만들고 원하면 억제”하는 contact-controllability를 학습하도록 설계했습니다. 또한 tactile sensor 없이도 proprioception만으로 런타임 contact 상태를 내재적으로 추정해 명령에 맞게 행동이 달라짐을 보여줍니다.

- **Technical Challenges**: 핵심 도전은 인간 HOI 데이터에서 keypoint 패턴과 contact 패턴이 강하게 상관되어, 정책이 contact 명령을 무시하고 keypoint만으로 ‘한 가지 접촉 모드’를 추론하려는 점입니다. 이를 해결하기 위해 trajectory augmentation으로 동일(혹은 유사)한 keypoint 구조를 유지하되 contact label을 달리하는 paired-motion을 만들었고, contact-label flipping·object removal·inflated geometry를 조합해 키포인트-접촉 상관을 의도적으로 깨는 전략을 사용했습니다. 더불어 contact label matching reward와 contact distance reward를 통해 “맞는 파트끼리 가까워지되, 원치 않는 접촉은 멀어지게” 하는 방향으로 학습을 유도했습니다.

- **Empirical Impact**: 시뮬레이션에서 HUMOTO 기반 10개 다양한 인간-물체 상호작용 모션에 대해, 동일 keypoint 궤적에서도 ✔/✘ contact command에 따라 접촉 수·임펄스가 일관되게 변하며 태스크 보상 없이도 조작을 완료하는 결과를 보였습니다. BeyondMimic과 비교해 contact 관련 지표가 크게 개선됐고(추적 정확도는 MPJPE가 비슷), 의자 밀기·박스 들기 등 contact이 필요한 상황에서 객체 조작 성공까지 달성했습니다. 실세계 Unitree G1에서도 5개 모션에서 명령 토글에 따른 contact이 재현되며, ablation과 sim2real 결과는 상관 붕괴를 위한 augmentation이 필수임을 뒷받침합니다.



### Learning Adaptive Solvers for Distributed Factor Graph Optimization on Matrix Lie Groups (https://arxiv.org/abs/2607.08735)
- **Prior Approaches**: 기존 분산 factor graph 최적화는 로봇/파티션 간 통신 제한과 비동기 환경을 고려해 왔지만, step size·감쇠(damping)·페널티 등 솔버 파라미터에 대한 수동 튜닝 의존도가 높았다. 또한 대다수 방법이 SE(3) 같은 강체 pose graph에 중심을 두어, SL(4)·일반 matrix Lie groups까지 폭넓게 확장되기 어려웠다. 그 결과, 파라미터가 그래프 토폴로지·노이즈·통신 레짐이 바뀌면 성능이 쉽게 흔들리는 문제가 남았다.

- **Core Contribution**: 이 논문은 DeepCORD로, CORD의 분산 Riemannian 최적화(Euler–Poincaré 기반)를 deep unfolding으로 펼쳐 differentiable iteration으로 만든 학습-보강형 분산 프레임워크를 제안한다. 핵심은 블랙박스 상태 회귀가 아니라, 각 로봇이 로컬 최적화 문맥(최적화 단계·최근 경계 노드·통신 지연 정보 등)에서 솔버 파라미터를 동적으로 조절하는 self-supervised feedback policy를 학습한다. 이렇게 DeepCORD는 동기/비동기 통신 모두에서 matrix Lie groups 위 적응형 분산 최적화를 가능하게 한다.

- **Technical Challenges**: 가장 큰 기술적 난점은 (1) 일반 matrix Lie groups 위에서 기하적으로 일관된 업데이트를 유지하면서 (2) 분산·비동기 환경에서 파라미터 튜닝 취약성을 학습으로 흡수해야 한다는 점이다. 논문은 CORD의 semi-implicit Euler 기반 분산 적분 구조를 보존하되, 각 반복마다 단 한 라운드 통신으로 로컬 2-hop 보강 그래프를 구성해 policy가 m·d·Δt 같은 파라미터를 예측하도록 설계했다. 학습은 정답 해 없이 unrolled objective를 최소화하는 self-supervised 방식으로 진행하며, 로컬 Hessian 선형해 계산은 sparse preconditioned conjugate gradient와 implicit differentiation으로 처리해 효율과 안정성을 함께 노린다.

- **Empirical Impact**: 실험에서 DeepCORD는 SE(3) pose graph optimization(동기 13개 중 11개, 비동기 13개 중 10개에서 최저/공동 최저)과 SL(4) projective submap alignment에서 전반적으로 기존 분산 기준선보다 낮은 objective value를 달성했다. 특히 통신 지연·드롭이 있는 비동기 시나리오에서 CORD 대비 개선 폭이 두드러졌고, 서로 다른 규모/그래프 크기(예: 학습 시 그래프 노드 수 제한을 넘어서는 경우)에서도 적응 파라미터가 일반화되는 결과를 보였다. 저자들은 다양한 실제 운영 조건에서 21/26 PGO 벤치마크(대부분) 및 모든 projective 정렬 데이터셋에서 우수한 성과를 보고, 분산 기하 최적화의 실배치 견고성을 끌어올린다는 의미를 가진다고 정리한다.



### Native Video-Action Pretraining for Generalizable Robot Contro (https://arxiv.org/abs/2607.08639)
- **Prior Approaches**: 기존 비디오-액션 모델은 관측에서 행동으로 직접 매핑하기보다, 장면이 어떻게 진화할지와 그 안에서 어떻게 행동할지까지 함께 예측해 일반화와 샘플 효율을 높였다. 하지만 많은 모델이 VAE(재구성 중심)와 bidirectional 비디오 생성 백본 같은 ‘디지털 콘텐츠용’ 구성요소를 그대로 재활용해, 동역학 정렬·표현 공간 일치·인퍼런스 지연·행동 신호 스케일링에서 구조적 한계를 노출한다. 또한 역방향/양방향 전제를 갖는 백본을 closed-loop의 단방향 시간 전개에 맞추는 retrofit은 사전지식의 약화를 유발할 수 있다는 지적이 나온다.

- **Core Contribution**: LingBot-VA 2.0은 로봇 embodiment를 전제로, 비디오 생성 모델을 붙여넣는 방식이 아니라 처음부터 ‘네이티브(native)’ 비디오-액션 파운데이션 모델로 설계했다. 핵심은 시맨틱 비주얼-액션 토크나이저로 관측과 행동을 하나의 의미 라탄 공간에 정렬하고, 그 위에 causal video-action 모델을 학습해 시간 구조까지 자연스럽게 맞춘다는 점이다. 결과적으로 few-shot(10–15 데모) 일반화와 경우에 따라 zero-shot 전환까지 확장되는 제어 능력을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 픽셀 재구성에 최적화된 잠재표현이 아니라 ‘의미와 행동’에 맞는 토큰 공간을 구성하는 것, (2) causal 학습 목표를 통해 웹 스케일 데이터에서도 행동 지식을 확장하는 것, (3) 실시간 closed-loop에서 모델 지연을 제어 지연으로 전환하지 않는 것이다. 논문은 semantic visual-action tokenizer로 언어-정렬 비주얼 파운데이션 특징에 맞춘 시맨틱 정렬과 self-supervised latent action 추출을 수행하고, 학습은 bidirectional-to-causal retrofit이 아닌 causal pretraining부터 시작해 catastrophic forgetting을 피한다. 또한 희소 MoE 백본으로 고주파 추론 효율을 확보하고, Foresight Reasoning 비동기 추론으로 실행과 예측을 병렬화하되 최신 관측에 매 rollout을 learned forward dynamics로 다시 근거(re-ground)하여 드리프트를 줄인다.

- **Empirical Impact**: 실세계 로봇 배치에서 LingBot-VA 2.0은 복잡한 조작 과제에 대해 few-shot generalization을 입증하며, 부족한 로봇 데이터에 의존하던 이전 방식보다 더 강한 기반 제어 지식을 웹 스케일 self-supervision으로부터 얻는 효과를 보여준다. MoE 인퍼런스, few-step consistency distillation, Foresight Reasoning, quantized 실행을 조합해 실시간 closed-loop 제어를 달성하며 최고 비동기 실행 주파수 225 Hz를 보고한다. 시뮬레이션과 실세계 평가에서 π0.5 및 LingBot-VA 같은 강한 베이스라인 대비 장거리 행동의 일관성과 조작 정밀도를 함께 개선해, 로봇 제어 파운데이션 모델 방향성에 의미 있는 실증을 제공한다.



### A New Human-Likeness and Comfort Index for Robot Movements Along Prescribed Paths (https://arxiv.org/abs/2607.08620)
Comments:
          14 pages, 5 figures, accepted for IEEE Transactions on Cybernetics

- **Prior Approaches**: 기존 연구는 로봇의 human-likeness(인간다움)를 주로 (1) 실행된 움직임의 궤적/부드러움/jerk 등 국소 운동 지표로 평가하거나, (2) 인간 설문을 통해 편안함·자연스러움을 사후에 판단하는 방식으로 나뉘는 경향이 있다. 또한 일부는 엔지니어링 지표를 제안했지만, 대부분은 로봇이 실제로 움직인 뒤의 궤적 데이터에 의존해 사전(online 이전) 최적화로 연결하기 어렵다. 특히 기존의 “인간다움 지표”가 경로 이후에만 계산되거나, 대상(개인)·하드웨어·제어 파라미터에 따라 일반성이 약하다는 한계가 제기된다.

- **Core Contribution**: 이 논문은 로봇의 end-effector 시간 법칙(time law)만으로도 편안함을 간접적으로 예측할 수 있다는 가설을 세우고, 이를 인간다움(human-likeness)과 연결한다. sigma-lognormal 운동 모델을 바탕으로 인간의 운동을 “motor program(운동 프로그램)”의 관점에서 근사하고, 실행 전에도 궤적의 인간다움을 수치로 평가할 수 있는 comfort index(인간다움 기반 지표)를 정의한다. 또한 기하 경로(path)와 시간 법칙(time law)을 분리해, 경로가 고정된 작업에서도 시간 법칙만 사람과 비슷하게 생성하면 편안한 동작으로 설계할 수 있음을 보인다.

- **Technical Challenges**: 핵심 기술 문제는 (1) 인간 운동의 시간적 특성을 한 지표로 요약하되, 단순히 고정밀 근사(SNR)를 높이는 것만으로는 “사람다운 최소 스트로크” 특성을 놓칠 수 있다는 점이다. 논문은 sigma-lognormal 기반 재구성으로 얻는 PPT(phase plane trajectory) 및 time law 일치도를 SNR로 평가한 뒤, 동작을 구성하는 stroke 수의 최소성(에너지 최적화 관점)을 반영해 인간다운 정도를 H_L로 정규화·패널티화한다. 더 나아가 phase plane에서 PPT를 설계해 surrogate(대체) 인간 동작을 만들고, 이를 통해 지표가 실제 실행 이전에도 비교적 일관된 판별을 제공하는지 검증하는 절차를 구축한다.

- **Empirical Impact**: 검증에는 총 68명의 피험자가 로봇과의 물리적 상호작용에서 편안함을 직접 판단하는 실험이 포함되며, 3회의 실험 캠페인을 통해 지표-선호의 관계를 확인한다. 결과적으로 제안된 human-likeness(인간다움) 분포가 높을수록 참여자들이 인식하는 perceived comfort(인지된 편안함) 선호와 전반적으로 일관된 경향을 보였다고 보고한다. 이로써 편안함을 사후 설문이나 실행 궤적 기반 지표가 아니라, end-effector 시간 법칙 중심의 사전 평가로 연결할 수 있는 정량 프레임이 제시되며, trial-and-error 부담을 줄이는 comfort-by-design 설계에 의미가 있다.



### FabriVLA: A Lightweight Vision-Language-Action Model for Precise Multi-Task Manipulation (https://arxiv.org/abs/2607.08575)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 RT-2, OpenVLA, π0 같은 대규모 파라미터 기반으로 다양한 로봇 조작을 일반화하는 데 초점을 맞췄다. 다만 수십억~수십억 이상급 모델은 연산량과 추론 지연 때문에 실시간 로봇 제어에 부담이 크다는 한계가 있었다. 또한 flow 기반 행동 생성이나 diffusion 계열이 성과를 내도, 정밀 조작에서 시간적(스텝 간) 의존성과 공간적 세밀함을 효율적으로 끌어오는 경량 설계는 여전히 과제로 남아 있었다.

- **Core Contribution**: FabriVLA는 1B 스케일 InternVL3.5 비전-언어 백본을 토대로, 경량하면서도 Precise Multi-Task Manipulation 성능을 노리는 Vision-Language-Action 모델이다. 핵심은 flow matching 기반 action head에 gated self-attention(액션 토큰 간 스텝 의존성 학습)과 shallow VLM layer fusion(깊은 의미+얕은 공간 디테일 동시 활용)을 결합한 구조다. 특히 단일 스테이지에서 VLM과 action head를 함께 최적화하면서, 대형 VLA 없이도 강한 결과를 보이도록 설계했다.

- **Technical Challenges**: 정밀 조작을 위해선 50-step action horizon에서 액션 스텝 간 의존성을 안정적으로 학습해야 하지만, 초기 학습에서 self-attention이 방해가 될 수 있다. FabriVLA는 gated self-attention의 게이트를 0으로 시작해 초기에는 cross attention 위주로 동작하다가 학습 중 점진적으로 게이트를 열어, 최적화 경로를 매끄럽게 하며 스텝 간 상호작용을 학습하도록 했다. 또한 의미론적 문맥만으로는 물체 국소화/접촉이 어렵기 때문에, layer 6과 layer 14의 VLM 피처를 concat_proj로 얕게 결합해 공간 디테일을 action head에 제공했다. 마지막으로 flow matching 학습을 위해 noise→정답 액션으로의 연속 시간 denoising을 Beta(2,2) 시간 샘플링과 속도장 학습으로 구성해 고품질 행동 생성을 뒷받침했다.

- **Empirical Impact**: Meta-World MT50(50개 조작 과제)에서 FabriVLA는 tier-average success rate 90.0%, overall episode-level success rate 92.0%를 달성해 비교 모델 대비 상단 성과를 보였다. 4개 난이도에서 easy 95.0%, medium 88.2%, hard 86.7%, very hard 90.0%로 전반적으로 고르게 강했다. ablation에서는 shallow layer fusion이 tier-average와 overall을 각각 7.1%p, 5.2%p 올렸고, gated self-attention이 베이스 대비 tier-average를 48.5%→57.7%, overall을 55.4%→66.9%로 끌어올리는 결정적 구성요소임이 확인됐다. 전체 결과는 “1B 스케일 VLM+경량 VLA 설계”만으로도 multi-task 조작에서 경쟁력 있는 성능을 낼 수 있음을 실증하며, 경량 실시간 제어 지향 VLA 연구에 신호를 제공한다.



### Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents (https://arxiv.org/abs/2607.08448)
- **Prior Approaches**: 엔드투엔드 VLA 모델은 언어와 비전을 바로 로봇 행동으로 매핑해 국소적인 접촉 중심 조작에는 강하지만, 학습 시 분포 밖의 배치(semantic retargeting, goal re-binding, 레이아웃 shift, 불안정한 접촉)에서 쉽게 실패한다. 반면 LLM 코딩 에이전트는 언어·구성 추론에는 유리하지만, 순수 analytic primitive 기반은 불규칙 그리핑/제약 배치/관절형 물체 상호작용 같은 접촉 민감 작업을 매끈하게 처리하기 어렵고, 스킬 라이브러리를 확장할수록 배치 안전성·재사용성 판단 비용도 커진다.

- **Core Contribution**: Harness VLA는 학습된 VLA를 그대로(frozen) ‘접촉 전문 primitive’인 vla_act로 고정해, 플래너가 analytic primitive들과 조합해 롱호라이즌 조작을 수행하도록 만드는 기억-증강 에이전트 프레임워크다. 핵심 아이디어는 스킬 라이브러리를 늘리는 대신, 태스크별 실행 trace와 전역 성공 규칙/실패 모델을 통해 각 고정 primitive의 사용 범위(언제, 무엇을, 어떻게 조합할지)를 학습한다.

- **Technical Challenges**: 어려움은 frozen VLA가 학습 궤적 분포 밖에서는 취약하다는 점을, 플래너 차원의 semantic 재-grounding과 re-staging으로만 제어해야 한다는 것이다. 논문은 vla_act를 로컬 접촉 단계에서만 희소 호출하고 실패 시 로봇을 다시 배치한 뒤 재시도(retryable)하도록 REPL 스타일 harness와 Task Specific Memory/Global Memory를 설계해, non-contact 구조(이동·스테이징·탐색)와 contact-rich 제어를 역할 분리로 해결한다.

- **Empirical Impact**: 실험에서 Harness VLA는 perturbed tabletop과 household kitchen, clean-to-randomized bimanual 조작 전반에서 기준선 대비 큰 폭으로 향상되며, LIBERO-Pro에서 38.6%p, RoboCasa365에서 25.4%p 개선을 보였고 RoboTwin C2R에서는 58.4%에 도달했다. 또한 VLA를 한 번에 끝내는 블랙박스로 쓰지 않고 플래너가 제한된 횟수로 재호출·재시도할수록 성공이 누적 개선된다는 분석을 제시해, 접촉 단계의 국소성 활용이 성능을 좌우함을 보여준다.



### EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data (https://arxiv.org/abs/2607.08436)
- **Prior Approaches**: 관점·속도·행동 스타일이 맞지 않는 상황에서 인간 egocentric 데이터를 행동 복제(behavior cloning, BC) 방식으로 공동학습하면, 사람의 형태·습관이 그대로 섞인 ‘실행 불가한 모션’을 로봇이 따라 하며 성능이 떨어진다. 즉, 액션 디코더 하나를 통해 들어오는 인간 데이터가 물체/장면/의미 같은 전이 가능한 요소와 행동 고유 요소를 얽어 버리는 것이 핵심 한계다. 더 나아가 기존 WAM도 어떤 ‘세계(world) 표현’이 인간-로봇 전이를 좌우하는지 체계적으로 비교하지 못했다.

- **Core Contribution**: 이 논문은 World Action Model(WAM) 공동학습에서 ‘액션’이 아니라 ‘장면이 어떻게 진화하는지(미래 상태 예측)’를 보조 신호로 추가하면, 전이 가능한 동역학 기반 표현을 만들 수 있음을 주장한다. 나아가 전이를 좌우하는 세계 표현의 조건을 appearance abstraction, cross-embodiment consistency, ego-motion factoring의 세 축으로 정리하고, 이를 만족하는 타깃으로 Pixel(재구성), DINO(시맨틱), 3D motion flow(기하적 모션)를 비교한다. EgoWAM은 정책 백본과 데이터 혼합을 고정한 채 세계 예측 타깃만 바꿔, 표현 선택의 효과를 ‘통제된 실험’으로 분리해 보여준다.

- **Technical Challenges**: 어떤 세계 타깃은 카메라/머리 움직임과 장면 변화를 섞어 supervision이 어긋나거나, 픽셀 재구성처럼 외형·모션이 엉켜 전이가 막히는 문제가 있다. 이를 해결하기 위해 Pixel VAE 기반 타깃은 재구성 실패모드의 기준선으로 두고, DINO는 외형을 추상화하지만 이미지 격자에 여전히 공간적으로 인덱싱된 한계를 갖게 설계한다. 3D motion flow는 Aria VIO 기반 카메라 자세로 카메라-정렬 좌표계에서 플로우를 정의해, 외형·에이전트 차이를 줄이면서 ego-motion을 분해하도록 구성했다.

- **Empirical Impact**: 세 가지 실제 양손(bimanual) 작업에서 WAM 공동학습은 대규모 in-the-wild egocentric 인간 데이터 스케일링에서 BC보다 더 일관되게 이득을 보였다. 픽셀 기반 예측은 전이가 약했지만, DINO와 3D motion flow는 의미 있는 향상을 보였고 DINO는 OOD(장면·물체 미일치)에서 최대 4배 수준의 일반화 개선, 3D flow는 in-domain 성능을 약 20–30% 끌어올렸다. 결론적으로 미래 동역학을 학습 신호로 쓰되, world representation을 시맨틱 추상화 또는 3D 기하적 근거로 맞출 때 인간 데이터의 효과가 ‘전이 가능한 표현’으로 정렬된다는 점이 실증됐다.



### On Exploring Input Resolution Scaling For Anytime LiDAR Object Detection (https://arxiv.org/abs/2607.08391)
- **Prior Approaches**: 기존 anytime computing(지연-정확도 절충)을 위한 DNN 기법들은 주로 early-exit, criticality 기반 입력 slicing/scheduling, 이미지 해상도 동적 스케일링에 집중해 왔다. LiDAR 쪽에서는 PointPillars에 대해 early-exit과 헤드 스케줄링을 결합한 Anytime-LiDAR, 그리고 VALO처럼 deadline-aware 입력 slicing/scheduling을 적용한 접근이 제안됐지만, resolution(입력 격자/피라미드 크기) 자체를 런타임에서 자유롭게 다루는 설계는 상대적으로 덜 탐구돼 있었다. 특히 LiDAR에서 resolution을 여러 개 지원하려면 보통 해상도별로 별도 모델을 올려야 해 메모리 제약(SWaP)에서 실용성이 떨어진다는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 MURAL(MUlti-Resolution Anytime LiDAR)이라는 단일 DNN 기반 프레임워크로, LiDAR point cloud 검출에서 런타임 resolution 스케일링을 가능하게 하면서도 메모리 사용을 최소화하는 방법을 제시한다. pillar/voxel 기반 구조에서 입력을 동적으로 더 큰/작은 해상도로 인코딩해 deadline을 만족하는 범위 내에서 가능한 한 높은 resolution을 선택하도록 설계됐다. 또한 resolution을 추가해야 하는 경우에도 해상도별로 별도 학습 모델을 배포하지 않고, 학습된 BN 파라미터를 이용한 합성(regression)으로 지원 범위를 확장하는 점이 핵심이다.

- **Technical Challenges**: 기여를 실현하려면 (1) 서로 다른 resolution에서 단일 가중치로도 정확도를 유지해야 하고, (2) LiDAR point cloud의 불규칙성 때문에 resolution별 실행 시간 예측이 흔들리는 문제가 있다. MURAL은 모든 BN을 resolution-aware BN으로 바꿔 각 resolution에서의 통계 분포 차이를 흡수하도록 학습하고, 필요한 추가 resolution은 BN 파라미터를 회귀로 합성해 정확도를 중간 수준으로 유지한다. 실행시간 예측은 지연을 PFE(또는 VFE), sparse CNN, dense CNN, post-processing으로 분해한 뒤, pillar 기반은 max pooling을 이용해 sparse CNN의 레이어별 활성 pillar 수를 추정해 예측 정확도를 높이고, voxel 기반은 미리 구성한 lookup table과 region-dropping으로 deadline miss의 영향을 완화한다.

- **Empirical Impact**: nuScenes 데이터(n=10프레임 병합, hard-deadline open-loop)에서 MURAL은 기존 anytime LiDAR 접근인 VALO 및 resolution별로 따로 학습한 baseline들과 비교해 다양한 deadline 구간에서 더 높은 mAP를 달성했다. 또한 PointPillars, CenterPoint까지 확장해 적용 가능성을 보여주며, 단순히 pillar 모델에 국한되지 않음을 확인했다. 시뮬레이션 closed-loop 주행 실험에서는 ego 속도 등 상황에 따라 resolution을 조정함으로써 충돌 없는 내비게이션을 유지하면서 불필요한 stall까지 줄여 안전성과 효율을 함께 개선하는 성과를 보고한다.



### FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation (https://arxiv.org/abs/2607.08359)
- **Prior Approaches**: 기존 UAV 비전-언어 항법(VLN)은 반응형 액션 예측이나 단일 autoregressive 디코딩으로 제어 명령을 바로 생성하는 방식이 많다. 이런 접근은 단기 시각-언어 편향에 취약해 장기 비행에서 궤적이 흔들리거나, 대형 추론 모델을 쓰면 추론 지연이 커져 실시간 적용이 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 FSD-VLN으로 의미 추론과 저지연 비행 제어를 구조적으로 분리하는 fast-slow dual-system 아키텍처를 제안한다. 느린(slow) 브랜치는 vision-language 모델에서 안정적인 semantic priors를 뽑고, 빠른(fast) 브랜치는 Diffusion Transformer(DiT)로 과거 행동과 시간 구조를 반영한 액션 분포를 생성해 일관된 비행 출력을 만든다.

- **Technical Challenges**: 핵심 기술 난제는 ‘전역 이해’가 요구하는 계산/지연과 ‘제어 안정성’이 요구하는 고주파 응답을 동시에 만족시키는 정렬이다. 이를 위해 느린 브랜치의 semantic feature를 버퍼로 캐시해 비동기 조율을 유지하고, DiT에서 cross-temporal action dependencies를 명시적으로 모델링해 분포 드리프트와 궤적 연속성 문제를 줄였다. 또한 long-horizon 학습에서 gradient oscillation을 완화하기 위해 time-aware adaptive optimizer(시간 가중 MSE)를 도입했다.

- **Empirical Impact**: 대규모 저고도 도시 시뮬레이션에서 FSD-VLN은 unseen 장면에서 성공률을 최대 2배 수준으로 끌어올리며(SR, SPL 개선) 내비게이션 오차도 크게 감소시켰다. 실시간성 측면에서는 1-step 추론 지연을 50% 이상 줄이고, 전체 임무 수행 시간도 50% 넘게 단축(약 53%)해 의사결정 효율과 궤적 품질을 동시에 입증했다. 결과적으로 장기 aerial VLN에서 decoupled semantic-control modeling이 실용적 성능-지연 균형을 제공한다는 점을 강하게 보여줬다.



### SkillPlug: Unsupervised Skill Mining for Few-Shot Adaptation in Robotic Manipulation (https://arxiv.org/abs/2607.08354)
Comments:
          8 pages, 8 figures, published to RA-L

- **Prior Approaches**: 기존 모방 학습 기반 비주얼 모토(visual-motor) 정책은 대부분 관측에서 저수준 행동으로 end-to-end로 직접 매핑하도록 학습돼, 작업 간 공통된 행동 구조를 명시적으로 재사용하기 어렵다. 스킬을 사람이 라벨링하거나 VLM으로 구간을 나눠 계층화를 하는 방식은 해석성은 좋지만 라벨/경계 설계 비용이 크고, 데이터에서 스킬을 군집화·임베딩·MoE로 학습하는 방식은 재사용성을 직접 밀어주지 않아 스킬이 중복되거나 장면 특성과 얽힐 수 있다.

- **Core Contribution**: 이 논문은 SkillPlug이라는 plug-in 형태의 스킬 마이닝 프레임워크를 제안해, 기존 visuomotor policy의 백본은 유지한 채 스킬-conditioning 모듈과 공용 skill library를 추가한다. multi-task 데모 원자료만으로 self-supervised 목표를 통해 ‘재사용 가능하고 비중복적’인 행동 수준 프리미티브를 학습하고, 이후 unseen 작업에는 router와 action head만 가볍게 fine-tuning해 few-shot 적응을 수행한다.

- **Technical Challenges**: 핵심 과제는 (1) 장면(외형) 정보에 덜 의존하면서도 (2) 스킬이 행동 수준에서 재사용되도록 만들고 (3) 스킬 간 중복 붕괴를 막는 것이다. 이를 위해 action segment만 입력으로 하는 VAE 스타일 trajectory-skill posterior encoder로 정보 병목을 걸어 장면 특성 흡수를 억제하고, BSA(행동-스킬 정렬)로 스킬 임베딩이 실제 행동 세그먼트를 잘 설명하도록 학습하며, SD(스킬 분리)로 동일 관측에서 서로 다른 스킬이 생성하는 잠재 표현의 유사도를 낮춘다.

- **Empirical Impact**: DISCOVERSE/ LIBERO 시뮬레이션과 실제 로봇에서 SkillPlug은 multi-task 성능과 few-shot 전이 모두에서 일관된 개선을 보였다. 예를 들어 DISCOVERSE에서 10-demo few-shot 성공이 +18.1%p, LIBERO에서 5-demo cross-suite transfer가 평균 +38.3%p 개선됐고, 실제 로봇에서도 few-shot 태스크 평균 성공률이 +28.5%p 상승했다. 또한 ablation에서 KL·SD·BSA를 추가할수록 점진적으로 성능이 올라 스킬 재사용성 설계가 효과적임을 뒷받침하며, 배치 성공률뿐 아니라 추론 단계 수도 줄여 제어 효율성까지 개선되는 것으로 보고됐다.



### AnyDexRT: Calibration-Free Dexterous Hand Retargeting with Few-Shot Human Guidanc (https://arxiv.org/abs/2607.08341)
- **Prior Approaches**: 기존 kinematic retargeting은 사람 손의 관절/팁 위치를 로봇 손에 옮기기 위해 역기구학과 hand-crafted objective, 정밀한 캘리브레이션, 혹은 인간-로봇 손 공간의 전역 shape matching(예: GeoRT)을 요구하는 경우가 많았다. 이런 방식은 기준좌표, 스케일/오프셋, 가중치 등 튜닝에 민감하고, 로봇 손의 중복 가능한 영역이 클 때는 작업 의도와 다른 왜곡된 매핑이 생길 수 있다.

- **Core Contribution**: AnyDexRT는 사람이 쓰는 kinematic retargeting을 위한 캘리브레이션 프리(calibration-free) 매핑을 제안하며, human-like dexterous hand들 사이에서 직관적인 조작을 유지하는 것을 목표로 한다. 핵심은 fingertip correspondence를 self-supervised shape matching으로 학습하되, few-shot human guidance로 작업에 필요한 영역을 앵커링해 비식별성(ambiguity)을 줄인다는 점이다. 또한 pinch 관련 자세는 contact classifier로 보정해 작은 접촉 동작의 신뢰도를 높인다.

- **Technical Challenges**: 문제는 (1) 인간과 로봇 손 공간이 스케일·가용범위·관절 결합이 달라 전역 정합이 자연스러움을 보장하지 못하고, (2) self-supervised correspondence는 해가 여러 개라 초기화/샘플에 민감해질 수 있으며, (3) 캘리브레이션 오차가 글로벌 좌표 기반 방향 일치를 깨뜨린다는 점이다. AnyDexRT는 bidirectional Chamfer 대신 partial Chamfer로 로봇의 불필요한 중복 영역 커버 강제를 줄이고, pairwise distance preservation 및 local motion loss로 기하 분포·국소 방향성을 보존한다. 더해 몇 번의 인간 기준 제스처로 alignment loss를 주어 매핑 다중해를 고정하며, pinch는 fingertip 위치뿐 아니라 접촉 신호를 분류해 미세 접촉을 안정화한다.

- **Empirical Impact**: 실험에서는 7종의 다양한 dexterous hands와 real-world teleoperation 작업(분무 스위치, 나사 조작, 도구 사용, 소형 물체 pick 등)을 통해 매핑 품질, 튜닝 노력, 운영 효율을 비교했다. 평균 local motion consistency가 59.8%에서 90.2%로 크게 개선됐고, 손/시드에 따른 변동성이 작아 더 안정적이고 일반화가 잘 됨을 보여준다. 또한 실제 작업 완료 시간이 전반적으로 가장 짧았으며, 특히 Pick-10의 pinch success rate에서 contact classifier 보정이 유의미한 성능 향상을 만들어 보다 신뢰할 수 있는 데이터 수집/제어에 기여한다.



### TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning (https://arxiv.org/abs/2607.08283)
Comments:
          Accepted to the SemRob 2026 Workshop at Robotics: Science and Systems (RSS 2026)

- **Prior Approaches**: VLA 정책(예: π0.5, OpenVLA)은 언어·시각·로보 상태를 받아 즉시 반응적(action)으로 다음 동작을 예측하는 경향이 강하다. 메모리/재귀를 추가한 연구들은 과거 관측 버퍼를 조회하거나 recurrent latent state를 유지하지만, 메모리 “업데이트 시점”을 조작 이벤트와 물리 경과시간에 맞춰 동적으로 조절하는 데는 한계가 있었다. 특히 stage-dependent manipulation에서는 겉으로 비슷한 상태가 잠재 진행도(task progress)에 따라 서로 다른 행동을 요구해, 단순한 step-index 기반 업데이트로는 부족하다는 문제의식이 제시된다.

- **Core Contribution**: 이 논문은 stage-dependent manipulation을 “물리 시간과 이벤트 구조를 반영한 belief(과업 진행을 요약한 잠재 상태) 추적” 문제로 재정의한다. Temporally Conditioned Memory-Fusion Policies(TFP)는 Liquid Time-Constant(LTC) 기반으로 episode-local task-progress belief를 연속시간으로 갱신하고, 업데이트된 belief를 flow-matching action decoder에 AdaLN 스타일 adaptive modulation으로 직접 주입해 action chunk 생성에 영향을 주도록 설계됐다. 또한 irregular한 policy-query 간격을 고려해 Episode-Aware Temporal Batching(EATB)로 학습 효율과 hidden-state 연속성을 함께 확보한다.

- **Technical Challenges**: 핵심 과제는 (1) 관측이 안정적이거나 가려진 구간에서는 belief를 유지하고, (2) 접촉·해제·subgoal 전환 같은 순간에는 새 증거를 빠르게 반영하는 “이벤트 민감적” 메모리 업데이트를 구현하는 것이다. TFP는 LTC의 input-dependent time constant와 elapsed time(Δt)을 써서 retention/쓰기 강도(write gain)가 물리 시간과 관측 변화에 따라 달라지게 만들고, 그 결과 belief가 action head의 생성 분포에 곧바로 조건화되도록 연결했다. 또 훈련에서는 chunk를 임의 순서로 섞지 않고 episode-local hidden-state 연속성을 보존하는 EATB를 사용해 긴 horizon의 재귀 학습을 가능하게 했다.

- **Empirical Impact**: 3.3B 파라미터 모델 기준 TFP는 LIBERO에서 평균 success rate를 96.9%에서 98.75%로, LIBERO-plus에서는 91.4%에서 93.77%로 끌어올렸다. occlusion 단계별 진행 판단을 분리하는 MIKASA-Robo ShellGameTouch 진단에서도 최대 75.0%의 success를 달성해 메모리 추론이 실제로 작동함을 보여준다. 메커니즘 분석에서는 이벤트 전후 write-gain 변화가 비이벤트 구간보다 약 6배 크고, hidden-state 개입이 action chunk에 인과적으로 영향을 주는 것으로 나타나 TFP의 compact하면서도 event-sensitive한 belief dynamics가 VLA의 강건성과 stage-dependent 조작 성능을 개선한다는 의미를 갖는다.



### X-ACTA: eXtended Analytic Center Tension distribution Algorithm for fixed and mobile cable-driven-parallel-robo (https://arxiv.org/abs/2607.08265)
- **Prior Approaches**: CDPR의 케이블 장력 선택은 TDA로 해결해 왔고, 보통 p-norm(특히 1-norm, 2-norm, infinity norm) 목적함수와 선형계획/제곱계획 등으로 구현된다. 하지만 1-norm·infinity norm 계열은 feasible polyhedron 꼭짓점으로 “점프”하면서 장력이 불연속이 되기 쉽고, 미세한 연속성이 필요한 haptic 작업에는 한계가 있다. WFW 밖으로 확장하려는 slacked QP(예: Laval, NTNU)는 slack으로 제약을 완화하지만, WFW 내부에서도 wrench 오차를 0이 아니게 만들며 미분가능성까지 동시에 만족시키기 어렵다.

- **Core Contribution**: 이 논문은 WFW 밖에서도 동작하되, 장력 프로파일의 연속성과 differentiability를 유지하고 non-linear inequality constraint까지 포함할 수 있는 extended Analytic Center 기반 TDA(EAC)를 제안한다. 핵심은 먼저 Relaxed Analytic Center(RAC)로 장력 상·하한을 “완화된 상자” 안에서 매끈하게 최적화한 뒤, 해가 완화 상자를 벗어날 때만 slack을 추가해 wrench 오차를 허용하는 2단계 구조다. 또한 제안 방법은 WFW 내부에서는 wrench 오차가 거의 생기지 않도록 설계하면서, WFW 경계 전환에서도 장력의 미분가능성이 깨지지 않는다고 주장한다.

- **Technical Challenges**: 기술적 난제는 (1) WFW 밖으로 나가면 해가 제약 불일치로 존재하지 않을 수 있는데도 (2) slack을 쓰는 순간 WFW 내부에서조차 wrench 오차가 남지 않게 하고 (3) 장력 프로파일의 미분가능성을 유지하면서 (4) 비선형 제약을 추가해도 전체 regularity를 보장해야 한다는 점이다. 이를 위해 로그 barrier를 smooth하게 완화해 strictly convex인 RAC 문제의 고유해와 min{k1,k2}-times differentiable 성질을 확보하고, 이후 μ(τ)의 smooth 구성으로 “필요할 때만” slack 항이 활성화되도록 설계한다. 또한 EAC의 KKT 기반 IFT 논증을 통해 switching 구간에서도 장력-시간 매핑의 differentiability가 유지됨을 기술한다.

- **Empirical Impact**: 시뮬레이션에서는 기존 NTNU와 비교해 수렴 시간을 줄이면서, high-frequency content(매끈함/주파수 특성)와 wrench tracking error 간 trade-off에서 Pareto dominance를 보였다고 보고한다. 또한 WFW 외부 상황(급격한 조작, 케이블 파손 후 등)에서 smooth한 tension profile을 유지하는 성능을 벤치마크 궤적으로 확인한다. 논문은 추가로 비선형 제약 포함 가능성까지 실험으로 보여주며, 해당 분야에서 미분가능 장력 설계가 필요한 제어/햅틱/고장 대응 쪽 응용에 직접적인 의미가 있다고 정리한다.



### RadLoc: Radar-based 3-DoF Global Localization via Fast, Robust, and Lightweight Spatial Descriptor Across Diverse Environmental Scenarios (https://arxiv.org/abs/2607.08115)
Comments:
          8pages, 12figures

- **Prior Approaches**: 기존 레이더 기반 글로벌 로컬라이제이션 연구는 주로 place recognition 성능을 높이거나 특정 컴포넌트(전처리/디스크립터/포즈 추정)에 집중했다. 또한 학습 기반 방식은 인식력은 좋을 수 있지만, 대규모 학습 데이터와 높은 연산량 때문에 새로운 환경 배치가 어렵다는 한계가 있었다.

- **Core Contribution**: RadLoc은 place recognition부터 3-DoF pose estimation까지를 end-to-end로 묶은 통합 파이프라인을 제안한다. 1D CA-CFAR 전처리, 압축 디스크립터 설계, hierarchical coarse-to-fine 검색, phase correlation 기반 3-DoF 포즈 추정을 한 모듈로 구성해 SLAM 및 multi-session SLAM에 바로 활용 가능하도록 했다.

- **Technical Challenges**: 레이더 이미지는 speckle, 수신기 포화, multipath, 잡음 등으로 인해 디스크립터 강건성이 쉽게 무너진다. RadLoc은 비용 큰 feature extraction을 1D CA-CFAR로 대체해 전처리 시간을 줄이면서 연속 강도 표현을 유지하고, near-to-far range 신뢰도 차이를 반영한 range-aware 디스크립터와 coarse 단계의 단조 가중치로 검색 편향을 완화했다.

- **Empirical Impact**: 5개 데이터셋의 15개 시퀀스에서 RadLoc은 다양한 환경/레이더 타입/날씨 조건에서도 robust한 성능을 보이며, 최신 기법 대비 가장 작은 디스크립터 크기와 가장 빠른 retrieval 시간을 달성했다고 보고한다. 단일 세션뿐 아니라 multi-session과 cross-weather 정렬, SLAM 재로컬라이제이션까지 실험적으로 확인되며, long-term 맵 관리에서의 스토리지·연산 효율성에도 의미가 크다.



### Factors Influencing Conversational Engagement in Robot-Delivered Individual Cognitive Stimulation Therapy (iCST) for Dementia in Home Settings (https://arxiv.org/abs/2607.07998)
Comments:
          Accepted for publication at the IEEE RO-MAN Conference 2026

- **Prior Approaches**: 치매 치료에서 대화는 단어 찾기 어려움, 발화 시작 감소, 턴테이킹 붕괴, 더 잦은 주저/군더더기 등으로 어려움을 겪는다. 기존 연구는 CST 같은 비약물 중재가 인지·의사소통·삶의 질에 도움이 된다는 결과는 축적했지만, 로봇이 제공하는 iCST(개별 CST) 동안 나타나는 미세한 대화 역학은 충분히 분석되지 않았다. 또한 로봇 연구도 주로 수용성·정서 반응 등 사용성 중심이어서, 참여도와 말하기 패턴을 정량화한 설계 인사이트는 제한적이었다.

- **Core Contribution**: 이 논문은 Co-STAR(자율 로봇 기반 Cognitive Stimulation Therapy)로 집에서 1주간 iCST를 제공한 실제 오디오 상호작용을 턴 단위로 분석해, 참여를 좌우하는 요인을 실증적으로 규명했다. 특히 prompt personalisation(개인화 프롬프트), 세션 내 상호작용 단계, 참여자 특성이 대화 참여도(응답 지속시간, 자기참조 등)에 어떻게 반영되는지 정리했다. 이를 통해 치매 로봇치료에서 적응형 conversational design이 필요하다는 경험적 근거를 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 로봇 대화에서 주저·자기수정·인지적 붕괴 같은 ‘대화 품질’ 신호를 안정적으로 추출하고, 개인차가 큰 치매 음성을 신뢰도 있게 전사/분할하는 것이다. 연구진은 Whisperx/WhisperD 기반 ASR과 화자 분리로 턴 맵을 구성한 뒤, 로컬 Llama-3로 인지/유창성 관련 표지(자기참조율, 응답지연, 말하기 속도, 주저 등)를 의도 분류해 지표를 설계·검증했다. 또한 개인화 vs generic 프롬프트를 구분하고, 세션을 초기/후기 구간으로 나눠 단계별 변화를 비교하는 분석 설계를 함께 적용했다.

- **Empirical Impact**: 8명의 PwD가 각 30분 세션을 반복한 배치에서, 개인화 프롬프트는 응답 지속시간·자기참조 언어·전반적 engagement를 유의하게 높였다(응답 지속시간 약 70% 증가). 반대로 세션이 진행될수록 발화량(WPT)·자기참조 참여가 줄고 말이 짧아지는 경향이 나타나, 인지 피로 가능성을 시사한다. 또한 첫 세션의 응답지연·말하기 속도·주저율 같은 대화 지표가 이후 장기 참여(세션 지속 여부)를 예측하는 단서로 관찰돼, 로봇이 실시간 적응 신호로 활용할 수 있는 설계 함의를 제공한다.



### D-CLIPSE: Distributed Consensus-based Localization with Passive Listening on Shared State Exchang (https://arxiv.org/abs/2607.07995)
Comments:
          8 pages, 7 figures, 1 table. Submitted to IEEE Robotics and Automation Letters

- **Prior Approaches**: 다중 로봇 위치추정은 정확도와 일관성이 모두 중요하지만, 중앙집중형 필터는 센서 데이터를 한곳에서 최적으로 융합하는 대신 단일 실패 지점, 통신·계산 제약 때문에 실제 구현이 어렵다. 분산 접근은 각 로봇이 자기 상태와 이웃 상태를 추정하며 합의(consensus)로 중앙 해에 수렴시키려 하지만, 기존 합의 기반 분산 기법들은 다중 라운드의 all-to-all 교환이나 모든 노드가 모델 정보를 공유한다는 가정을 요구하는 경우가 많다. 한편 쌍방 통신(pairwise) 중심 방법은 CI(covariance intersection)로 교차상관을 다루기도 하지만, 의사측정(pseudomeasurement) 설계나 상태 전체 공유 등으로 통신·구현 부담이 커질 수 있다.

- **Core Contribution**: 이 논문은 통신 효율성과 합의 기반(consensus-based) 일관성(consistent)을 동시에 노리는 분산 필터 프레임워크를 제안한다. 핵심은 로봇 간에 필요한 정보만 공유하도록 하며, 특히 preintegrated odometry(RMI)와 공유(shared) 상태만을 한 번의 쌍방 교환에서 교환해도 합의 업데이트가 가능하게 설계한 점이다. 또한 통신하지 않는 이웃은 passive listening으로 같은 교환 정보를 “엿듣고” 이웃 상태 합의 갱신에 참여하도록 확장한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 로봇들이 중복된 RMI를 서로 참조하면서 발생하는 이중 카운팅(double counting)과 (2) 공유 상태 간 상관(cross-correlation) 미지로 인한 불일치를 어떻게 막으면서도, (3) 단 한 번의 pairwise 통신으로 합의를 형성하는 것이다. 이를 위해 먼저 공유 상태의 공분산을 CI로 인위적으로 팽창시켜 상관 불확실성을 보수적으로 다루고, 그다음 Lie group 상에서 가중 비선형 최소제곱(Gauss-Newton) 형태로 공유 상태의 consensus distribution을 계산해 각 로봇의 local joint distribution을 재조건화(reconditioning)한다. 최종적으로 재구성된 consensus를 로봇의 고유(unique) 상태 업데이트에 연결함으로써, 통신 비용은 줄이면서도 추정 일관성 수렴을 노린다.

- **Empirical Impact**: 논문은 시뮬레이션과 실제 실험에서 제안 방법을 검증했으며, 기존 SoTA 분산 접근 대비 특히 일관성에서 개선을 보이면서 정확도도 중앙집중형 해에 근접하는 성능을 보고한다. 이는 “통신량을 줄이되(consensus를 더 비싸게 만들지 않고) 일관성을 희생하지 않는다”는 목표를 실증적으로 뒷받침하는 결과로 해석된다. 또한 RMI preintegration과 passive listening까지 포함해 구현 가능한 현실적 통신 제약 하에서의 분산 위치추정 설계를 제시한다는 점에서, 협업 내비게이션·형성제어 같은 다운스트림 작업의 신뢰도를 높이는 데 의미가 있다.



### In vivo feasibility study of humanoid robots in surgery (https://arxiv.org/abs/2607.07972)
- **Prior Approaches**: 기존 복강경(MIS) 로봇은 da Vinci 등처럼 목적기반 플랫폼과 기계식 RCM(원격중심운동) 제약을 통해 정밀도·안전성을 확보해 왔다. 한편 일반목적 매니퓰레이터로 RCM을 제어하려는 시도도 있으나, 대체로 da Vinci 전용에 가까운 로봇/기구 의존성이 남아 현재의 범용 휴머노이드가 수술 수준 요구를 얼마나 충족하는지는 불명확했다.

- **Core Contribution**: 본 논문은 범용 휴머노이드에 표준 수술 기구를 붙여 복강경 텔레오퍼레이션을 수행하는 프레임워크를 제안하고, 이를 현대 휴머노이드의 수술 적합성 관점에서 체계적으로 평가한다. 벤치탑 특성화, 건식(dry-lab) 사용자 연구, 그리고 최초의 휴머노이드 기반 in-vivo 돼지 복강경 담낭절제 실험(2회)까지 이어지며, 기존 수술 로봇 플랫폼 대비 “기술적 실현 가능성-작업 성능-임상 준비도”를 함께 계량한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 기계식이 아닌 휴머노이드 기반에서 RCM 피벗을 지각·제어로 안정적으로 유지하는 것과 (2) 상용 술기구의 기하 파라미터 비공개로 생기는 kinematic 모델링/캘리브레이션 오차, (3) 텔레오퍼레이션 지연으로 인한 닫힌고리 안정성 저하에 있다. 논문은 ArUco 기반 RCM 국소화와 확장 kinematic chain 역매핑을 설계해 제약을 구현하고, 지연(팔로어-리더 및 제어 루프)과 명령-실행 추종 오차(직선·원 궤적 RMS 잔차)로 정확도 한계를 정량화했다.

- **Empirical Impact**: 사용자 연구에서 휴머노이드 텔레오퍼레이션은 수동 조작 대비 작업 오류를 유의하게 줄였지만, da Vinci Xi 같은 목적기반 로봇과 비교하면 응답성·정확도 측면의 격차가 남았다. 직선은 밀리미터 스케일 추종이 가능했으나 곡선은 in-plane 정확도 제약이 두드러졌고, 설문에서도 지연으로 인한 피드백 저하와 도달성(reachability) 제약이 주요 제한으로 보고됐다. 그럼에도 돼지 모델 담낭절제 2회 수행 결과는 휴머노이드가 최소침습 수술 워크플로로 “진입 가능한” 초기 근거를 제공하며, 임상 배치를 위해선 RCM/기하 캘리브레이션과 제어 응답 개선이 선행돼야 함을 시사한다.



### Soft Robotic Exogloves for Dexterous Mobility -- Towards Personalized Rehabilitation (https://arxiv.org/abs/2607.07968)
Comments:
          8 pages, 14 figures. To be published in The IEEE RAS/EMBS 11th International Conference on Biomedical Robotics and Biomechatronics (BioRob 2026)

- **Prior Approaches**: 기존 재활용 웨어러블 장갑은 표준 치수로 설계되는 경우가 많아 개인의 손 해부학과 불일치가 생기기 쉽다. 특히 손가락의 미세 관절 정렬이 중요한 dexterous manipulation 영역에서는 액추에이터 굽힘 위치가 MCP·PIP 관절과 어긋나 불편과 성능 저하로 이어진다. 케이블 구동 방식은 경로 설계가 복잡하고 마찰·백래시 문제가 있어 personalization과 확장에 불리하며, 수동 측정이나 2D 이미지 기반의 제한된 개인화도 있었다.

- **Core Contribution**: 이 논문은 3D 스캔으로 개인의 손 형상을 반영해 pneumatically-actuated soft robotic exoglove를 맞춤 제작하고, MCP·PIP 관절에 굽힘 영역이 정렬되도록 설계·제작·모델링·검증까지 제시한다. 또한 손가락-액추에이터 간 physical human-robot interaction(pHRI) 접촉을 단순화된 개인화 생체역학 모델과 함께 FEM으로 분석할 수 있는 틀을 제공한다. 장갑은 구조적 적합성(베이스 형상), 관절 토폴로지 정렬, 접촉력 모델링, 시간에 따른 구동 변형 프로파일까지 ‘개인화’ 관점에서 다룬다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 손 해부학에 맞춰 액추에이터와 관절 정렬을 유지하면서 (2) 접촉 상황에서의 힘/응력 분포와 관절 가동을 신뢰성 있게 예측하고 (3) 압력-변형을 안정적으로 제어하는 것이다. 저자들은 Artec Eva 3D 스캔 기반으로 도르살 베이스를 CAD화하고, fPN(fast pneunet) 계열의 2-자유도(각각 MCP·PIP 타깃) 액추에이터를 실리콘 몰딩으로 제작했다. FEM에서는 공기 챔버 팽창과 pHRI 접촉(마찰 가정 포함)을 반영해 응력 및 접촉력 경향을 보고, 실제 실험에서는 flex sensor 캘리브레이션과 PID 압력 제어를 통해 정적·동적 기준 추종을 검증했다.

- **Empirical Impact**: 실험 결과 topological 3D 스캔 기반 개인화가 손 해부학에 대한 정밀 맞춤을 가능하게 했고, 압력 제어는 MCP·PIP 관절의 목표 가동을 ‘정확하고 지향성 있게’ 구현했다. 다수 디자인 비교에서 strain-limiting layer를 완화(제거)하면 구동 중 액추에이터-손가락 관절 정렬이 개선되었으며, 이는 실제 착용 실험에서도 확인됐다. PID 기반 압력 제어는 계단/사인 입력 모두에서 기준 추종이 양호했고(최대 오버슈트 4.60%, steady-state error 0.0272 psi), 외란(타인이 손가락을 밀어 넣는 상황)에도 형상을 유지해 안전한 pHRI 시나리오에 적합함을 보여준다.



### Towards Soft Robotic Exogloves for Musculoskeletal Manipulation to Reduce Pain and Spasticity (https://arxiv.org/abs/2607.07958)
Comments:
          8 pages, 15 figures. To be published in the IEEE RAS/EMBS 11th International Conference on Biomedical Robotics and Biomechatronics (BioRob 2026)

- **Prior Approaches**: 기존 soft robotic 장갑은 cable-driven 방식이 많았지만, 케이블 마찰로 힘 전달이 저하되고 부착 위치가 이산적이라 힘 분포가 균일하기 어렵다는 한계가 지적돼 왔다. 또 여러 pneumatically actuated 장치는 손의 가동성·ADL 보조에 집중했으며, 통증 감소나 관절 경직 완화까지 한 번에 다루는 설계는 드물었다. 한편 마사지·압박 치료용 pneumatic/섬유·벨로우 기반 액추에이터는 존재했지만, 손의 구축(주먹쥔 자세 포함) 상황에서의 ‘맞춤 착용’과 ‘모빌리티+통증’을 동시에 만족하는 통합 솔루션은 부족했다.

- **Core Contribution**: 이 논문은 손의 spasticity에 대해 가동성(자세 재배치)과 마사지형 압박을 동시에 제공하는 modular soft robotic exoglove의 예비 개발을 제시한다. 손의 3D 토폴로지와 관절/운동학을 스캔해 dorsal·ventral finger 액추에이터와 palmar 액추에이터를 각각 개별 맞춤화하고, 압력 제어로 손을 중립에 가깝게 유도하면서 분산 압박을 수행한다. 특히 ventral과 palmar는 구축된 손의 좁은 공간에도 들어갈 수 있도록 ‘압축 가능한(bellow/형상 압축)’ 구조를 설계해 착용·탈착의 현실성도 함께 겨냥한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 맞춤형 형상이면서도 원하는 압력 범위에서 구조적 파손 없이 반복 구동해야 하고, (2) spastic finger처럼 수축된 공간에 들어갈 정도로 컴팩트하게 접히며, (3) 압력에 따른 변형 모드가 설계 의도(예: 순수 회전/펜상 확장)와 일치해야 한다는 점이다. 저자들은 photogrammetry 기반 손 스캔→메시 후처리→액추에이터 CAD/몰드 생성을 거치며, dorsal·ventral은 fPN(air chamber+실리콘 캐스팅), palmar는 3D-printed bellow(Elastic 50A V2 수지)로 구현했다. 또한 FEM(ANSYS)로 응력·변형을 사전 예측해 palmar bellow에서 초기 곡률 설계의 응력 집중 실패를 ‘내부 보강 및 중앙 air inlet’으로 수정했고, 그 결과 반복 사이클에서도 누설·손상이 없음을 기계 시험으로 확인했다.

- **Empirical Impact**: 건강한 참가자 손과 spastic-유사한 구축 자세에서의 실험은, ventral fPN이 구축된 손가락의 좁은 공간에 쉽게 압축·장착되며 압력 증가에 따라 손가락이 펴지는 방향 감각을 제공함을 보여준다. 반면 맞춤 기반으로 인해 dorsal fPN은 ventral보다 더 뻣뻣해 장착 난이도가 높았고, 이는 FEM에서의 변형성 차이와도 일치하는 결과로 해석된다. 전체적으로 이 연구는 ‘가동성 보조+통증/경직 완화형 압박’이라는 통합 목표를 소프트 액추에이터 맞춤·설계 검증(FEM+제작)·착용 가능성 관점에서 한 단계 끌어올린 예비 성격의 실험적 진전을 제공한다.



### Monocular Vision Based Control Framework for Grasping (https://arxiv.org/abs/2607.07897)
Comments:
          This paper is accepted at IEEE/ASME International Conference on Advanced Intelligent Mechatronics 2026 (AIM 202^)

- **Prior Approaches**: 기존 변형(soft) 물체 집게 연구는 보통 압력/촉각 기반 비전(tactile) 센서나 FEA 같은 물리 모델, 혹은 연성 그리퍼 같은 전용 하드웨어에 의존해 왔다. 하지만 촉각 센서는 센서 엘라스토머 변형 신호가 약해지거나 마모에 따른 감도 저하가 생기고, 물리 모델은 시간에 따른 비선형 물성 변화(예: 음식의 수분·노화)에 취약하다.

- **Core Contribution**: 이 논문은 RGB 단일 모노큘러 입력과 position-controlled 그리퍼만으로, 연성 물체와 강체 물체를 한 제어 파이프라인에서 동시에 다루는 통합 프레임워크를 제안한다. 핵심은 언어로부터 물체의 expected compliance(연성/강성 성향)를 추정해 접촉 전 그립 전략을 선택하고, 이후에는 시각 피드백으로 그립 폭을 적응시키는 구조다.

- **Technical Challenges**: 센서·물리모델 없이 연성 변형을 제어하려면, (1) 물체를 안정적으로 분할/추적하고 (2) 변형 정도를 시각 관측으로 대리해야 하며 (3) 강체는 다른 시각 단서로 제어해야 한다. 저자들은 open-vocabulary detection과 SAM2 분할, 경계 보정 포인트 할당, TAPIR 기반 실시간 포인트 트래킹, Depth Anything 모노큘러 깊이추정으로 3D 점을 복원한 뒤, 변형 물체에는 Procrustes 기반 dissimilarity를, 강체 물체에는 추적점 거리 스케일링을 사용해 gripper width를 조절한다. 또한 StiffNET(언어 기반 stiffness 추정)은 CLIP 텍스트 인코딩을 고정하고, LLM이 생성한 pairwise hardness 비교(랭킹)와 희소 강성 측정값(스케일 앵커)을 함께 학습해 접촉 전 object-level prior를 만든다.

- **Empirical Impact**: 실험은 Franka Emika Research 3 팔과 Franka Hand(포지션 컨트롤 그리퍼)에서 lettuce, fresh mozzarella cheese, croissants, paper towels 같은 연성 물체와 hard plastic bottles 같은 강체를 대상으로 pick-and-place를 수행해 검증했다. 결과적으로 촉각 없이도 시각 피드백만으로 두 범주의 물체에서 안정적인 그립을 달성했으며, 음식 및 가정용 조작에서 센서 비용과 설치 제약을 낮추는 실용적이고 범용적인 접근이라는 점을 강조한다.



### Time-to-Collision Based Dynamic Obstacle Avoidance Using Pretrained Vision Models for Robots in Unstructured Environments (https://arxiv.org/abs/2607.07885)
Comments:
          9 pages, 8 figures

- **Prior Approaches**: 기존 로봇 장애물 회피는 end-to-end 학습형(Transformer 등)이 강력하지만, 로봇별 대규모 데이터 수집이 병목이 된다. 다른 대안은 시뮬레이션에서 RL로 학습한 정책을 전이하는 방식인데, sim-to-real 격차(비주얼·접촉 역학·환경 다양성 불일치)로 인해 실제 환경에서 성능과 안전성 보장이 약해진다. 또한 TTC(충돌까지 남은 시간)를 학습으로 바로 예측하면 데이터 다양성 의존성과 해석 가능성 한계가 남는다.

- **Core Contribution**: 이 논문은 sim-to-real 전이를 피하면서도 해석 가능한 비전 기반 동적 장애물 회피를 수행하는 데이터 효율적 방법을 제안한다. 핵심은 UniDepth(단안 metric depth)와 SuperPoint+SuperGlue(장기 키포인트 대응)를 사용해 3D 키포인트 궤적을 복원하고, 각 키포인트의 TTC를 기하적으로 계산한 뒤 최소 TTC 키포인트를 회피 방향에 반영하는 것이다. end-to-end 학습 대신 명시적 3D 구조·시간 정보를 사용해 일반화성과 설명 가능성을 함께 노린다.

- **Technical Challenges**: 기여를 실제로 만들 때의 가장 큰 난관은 단안 depth의 오차가 3D 투영과 번들 조정, 최종 TTC 추정까지 연쇄적으로 전파되는 점이다. 이를 줄이기 위해 5프레임 슬라이딩 윈도우에서 XM solver로 scaled bundle adjustment를 수행하고, (1) 충분히 길게 트래킹되지 못한 키포인트, (2) 너무 먼 깊이(20m 초과), (3) 프레임 간 3D 변위가 큰 이상치(2m 초과)를 제외·재초기화해 기하 제약의 신뢰도를 높였다. 이후 최소 TTC 키포인트의 CPA(closest point of approach)를 기준으로 지면(ground plane) 2D 모션 프리미티브를 선택한다.

- **Empirical Impact**: M3ED(spot-forest, spot-outdoor-day) 실데이터 평가에서 TTC 1초 미만 프레임 식별 precision 0.49, recall 0.38을 기록했으며, 위협 프레임을 올바르게 감지한 경우 회피 방향 일치율은 84%였다. 특히 서로 다른 물리 장애물 22개 중 20개에서 최소 한 프레임(TTC<1s)을 포착해 실제 회피 트리거 관점에서 유의미한 관찰 성능을 보였다. 또한 모델 학습을 없애고 하이퍼파라미터 튜닝에 74초 데이터만 사용함으로써, 대규모 로봇별 학습이 필요한 기존 접근 대비 데이터 효율성과 해석 가능성을 동시에 입증했다.



### STEMbot: A Compliant Robot for Under-Canopy Plant Navigation (https://arxiv.org/abs/2607.07873)
Comments:
          Accepted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026). Project page: this https URL

- **Prior Approaches**: 유기농 채소에서 해충을 찾는 기존 방식은 IPM처럼 주기적인 수작업 점검과 유인 트랩에 의존해 인력·전문성이 많이 들며, 원거리 모니터링(드론/로버)은 잎 뒷면이나 줄기에서 자라는 해충을 가려진 상태로 두기 쉽다. 클라이밍 로봇 연구도 있었지만, 산업용은 식물의 불규칙·다공성 지형에 취약하고, 다수의 트리 클라이밍은 가지가 없는 단일 줄기 중심이거나 온보드 비전·SLAM이 부족해 장거리 계획·맵핑이 어렵다. 또한 비전 기반 SLAM은 식물의 반복 텍스처와 단색에 의해 perceptual aliasing이 커져 가까운 거리에서의 정합이 흔들린다.

- **Core Contribution**: STEMbot은 식물 캐노피 아래에서 자율 내비게이션을 목표로 한 초소형 클라이밍 로봇/소프트웨어 프레임워크로, 잎·줄기 가림(occlusion)을 우회해 가려진 표적 영역을 사전에 점검할 수 있게 한다. 로봇은 줄기 지름 7–33mm 범위에서 올라타고, 가지로 전환하며, 뒤집힌 자세에서도 접촉을 유지하는 하드웨어를 갖춘다. 소프트웨어는 geometric PIN-SLAM 기반의 기하학적 정합과 semantic OcTree 맵핑, 그리고 가지를 고려한 manifold-constrained A* 플래너를 결합해 “가림 아래에서도 전역적으로 일관된 위치추정”을 지향한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 비정형 식물 형상과 극심한 occlusion에서의 견고한 로컬라이제이션·맵핑, (2) 식물의 단조로운 색/반복 패턴으로 인한 시각적 대응 실패, (3) 줄기-가지 전환 시 킨매틱 제약과 경로 연결성 민감도다. STEMbot은 PIN-SLAM의 깊이 기반 기하 앵커로 photometric 일관성 의존을 줄이고, SAM+CLIP로 잎/줄기 등 의미를 부여해 semantic OcTree에 확률적으로 통합한다. 플래닝 단계에서는 식물의 국소 manifold를 상태공간으로 모델링하고, k-d tree 기반의 줄기 투영, 법선/방향 정합성 제약, 충돌 및 분기 도킹 조건을 A* 탐색의 가지치기에 반영했으며, receding horizon으로 실행 중 오차(모델 불일치·슬립)를 보정한다.

- **Empirical Impact**: 하드웨어 실험에서 STEMbot은 PLA/원통 형상 벤치마크부터 생물 샘플까지 자율 주행과 재구성을 시연했으며, 줄기 굴곡(곡률 반경 50mm)과 가지 접합(최대 90°) 같은 제약도 통과했다. 실제 4개 식물 표본에서 state 기반 탐색과 visibility(가림 없는 관측 가능 상태) 기반 목표를 모두 수행했는데, 특히 가려진 구역 검사에서 가지 전환 및 자세 재정렬이 관찰됐다. 정량적으로는 오프라인 포토그래메트리 기준에 대해 Chamfer distance 평균이 인공 식물 3.85mm, 생식물 13.36mm로 나타났고(주로 생장/비강성 및 세그멘테이션 오류 영향), depth 기반 SLAM+manifold 상태격자의 전역 일관성 내비게이션 가능성을 뒷받침한다.



### Shift & Drift: A Zero-Shot Benchmark for Generalizable and Robust Autonomous Driving Motion Planning (https://arxiv.org/abs/2607.07844)
Comments:
          Accepted at 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **Prior Approaches**: 기존 closed-loop motion planner 평가는 nuPlan·CARLA처럼 학습과 유사한 지리/시맨틱 분포(i.i.d.)에서 이뤄져, map memorization이나 인과적 착각으로 인한 일반화 한계가 가려지는 문제가 지적돼 왔다. 또 소규모 perturbation을 견디더라도 시간에 누적되는 실행 오차(compounding error) 이후 회복 능력은 상대적으로 덜 다뤄졌고, 단순화된 동역학 모델은 안전 점수의 낙관적 상한을 만들 수 있다.

- **Core Contribution**: 이 논문은 두 축의 분포 변화를 동시에 겨냥하는 듀얼 트랙 벤치마크 Shift & Drift를 제안한다. 시맨틱 시프트 트랙은 DeepScenario Open 3D(DSC3D)를 nuPlan 시뮬레이션으로 변환해 zero-shot 평가를 가능하게 하고, 상태 분포 드리프트 트랙은 에고 차량의 동역학에 확률적 잡음을 주입해 실행 오차 누적 후 회복 성능을 정량화한다.

- **Technical Challenges**: 핵심 난제는 (1) occlusion-free 항공 데이터(DSC3D)를 nuPlan의 지도/로그 포맷에 일관되게 매핑하는 변환 파이프라인을 구축하고, (2) temporally correlated actuation noise(AWGN/OU)를 통해 실제 하드웨어적 드리프트를 재현하면서도 평가 가능한 시뮬레이션 조건을 만드는 것이다. 저자들은 OpenDRIVE 기반 의미·기하 프리미티브 추출, GPS/좌표계 정렬, 프레임 업샘플링, 그리고 닫힌-루프 재생 시 충돌/진입 위반/진행 부족 필터링 및 수동 보정으로 재현성을 확보했다.

- **Empirical Impact**: 실험 결과 imitation learning(PlanTF·Diffusion Planner 등)은 in-distribution에서는 강하지만 시맨틱 시프트와 특히 보행자 밀집 환경에서 크게 무너지고, temporally correlated OU 잡음에서는 드리프트가 지속되는 경향이 확인됐다. 반면 reinforcement-learning 기반 CaRL은 두 트랙 모두에서 성능 저하가 더 완만해 전반적으로 안전·진행 지표를 더 잘 유지하며, “모방 충실도 vs closed-loop 회복 탄력성” 간 경험적 트레이드오프를 뚜렷이 보여준다. 저자들은 1,182개 시나리오(다수 독일 도시+샌프란시스코) 규모의 스트레스 테스트를 통해 배포 신뢰성을 가늠하는 기준점이 될 수 있음을 제시한다.



### Physics-Guided Biomechanical Gait Adaptation for Humanoid Locomotion on Extreme Sloped Terrains (https://arxiv.org/abs/2607.07830)
Comments:
          12 pages,6 figures

- **Prior Approaches**: 모델-프리 강화학습은 계단·디딤돌·파쿠르 같은 복잡 지형을 잘 통과해 왔지만, 가파른 경사면은 별도 물리 과제로 다뤄지기보다 여러 지형 중 하나로 취급되는 경우가 많았습니다. 기존 방법들은 대개 월드 기준의 안정성 보상(예: ZMP 계열)이나 범용 생존·추적 보상에 의존해, 경사에서 중력 편향이 누적될 때 저자세(낮은 CoM)로 보수적으로 수렴하는 “Groucho gait” 같은 자세 퇴화를 유발할 수 있습니다.

- **Core Contribution**: 이 논문은 경사면에서도 자세 제어와 동적 안정성을 동시에 달성하기 위한 2단계 물리 가이드 프레임워크 HumoSlope를 제안합니다. Stage I은 로컬 경사 지지면에 정렬해 평가하는 slope-adaptive ZMP 정규화로 블라인드(자세 센서만 사용) 균형 기준선을 만들고, Stage II는 Biomechanical Slope Gait Adapter(BSGA)로 경사 기하에 따라 CoM 높이와 상·하행 보행 비대칭을 조절해 저자세 붕괴를 막습니다.

- **Technical Challenges**: 핵심 기술적 난제는 “경사에서의 안정성”을 월드-수평 가정으로 보상하면 기하적 불일치가 생겨 학습이 왜곡된다는 점입니다. 저자들은 지지 발을 기반으로 로컬 지지면 법선을 추정하고, 지지면에 정렬된 ZMP 편차를 점질량(가상 힘) 근사로 계산해 Stage I에서 더 견고한 균형 priors를 학습하게 했습니다. 또한 Stage II에서는 학습 시에만 주어지는 PCA 기반 지형 디스크립터로 soft reward prior들을 게이팅하며, 실제 배치(actor)는 전적으로 proprioceptive 관측만 사용하도록 설계했습니다.

- **Empirical Impact**: 시뮬레이션과 Sim-to-Real에서 HumoSlope는 경사 상·하행이 섞인 held-out 경사 트랙에서 경쟁 방법 대비 높은 성공률과 더 긴 진행(또는 낮은 실패 전진 거리)을 보였습니다. 특히 시뮬레이션 슬로프 한계 스윕에서는 최대 36°(73%)까지 도달하며, 실외 잔디 경사에서는 62.7% 구배(약 32.1°, 국소 측정 36.4°)까지 블라인드 연속 보행을 달성했다고 보고합니다. 자세 진단 지표에서도 기존 보행 정책이 보이는 지속적 저자세 경향을 줄이고, 경사에 따라 상체 기울기·다리 협응이 달라지는 방향성 제어가 확인됐습니다.



### SASGeo: Stability-Aware Semantic Map Localization for GNSS-Denied UAVs -- A Framework and Synthetic Proof of Concep (https://arxiv.org/abs/2607.07737)
Comments:
          7 pages, 5 figures

- **Prior Approaches**: GNSS가 불안정한 환경에서 UAV는 VIO로 상대 위치를 추정하지만, 절대 관측이 없으면 오차가 누적된다. 이에 따라 UAV 이미지와 지오레퍼런스 항공·위성 이미지를 매칭해 절대 보정을 얻는 교차뷰 retrieval이 발전했지만, 외형 변화(계절·조명·시점·센서·지도 갱신 등)에 민감하다는 한계가 있다. 기존 연구는 OSM/벡터 지도, 의미 임베딩, 로드 BEV 보정, 그래프 매칭, 순차 필터 등 구성요소를 부분적으로 다뤘지만, 안전성에 필요한 요소(관측 밀도, 관계 검증, 시간적 일관성, 영속성 모델, 애매한 후보 거부 옵션)를 통합해 명확히 운영하는 방식은 부족했다.

- **Core Contribution**: 논문은 SASGeo가 환경을 픽셀이 아닌 도로·건물·수계·철도·교차로·경계 같은 persistent structure의 의미 지도(semantic map)로 표현해, 교차뷰에서도 위치를 구분하도록 설계했다고 제안한다. 의미 레스터 정렬과 관계형 그래프 증거, 시간/지도 나이 기반 persistence 신뢰도, 긍정·모순·unknown 관측의 명시적 처리, 그리고 애매한 고정(absolute fix) 거부를 한 프레임워크로 결합한다. 또한 ‘무엇을 어떻게 가중/판정할지’를 구체적인 모델과 의사결정 규칙으로 제시해, 막연한 semantic weight 제안 수준을 넘겼다.

- **Technical Challenges**: 핵심 난제는 (1) 외형이 크게 변하는 상황에서 의미 구조를 안정적으로 정렬하고, (2) 그래프 위상·관계가 잘못된 후보를 걸러내며, (3) 모호한 경우에는 고정 자체를 보류하는 integrity-aware decision을 만드는 것이다. 논문은 다중 프레임 VIO 누적으로 로컬 BEV 위에 의미 예측을 투영·누적하고, 증거를 긍정/모순/unknown으로 분해해 unknown을 부정 근거로 세지 않도록 했다. 이어 가시성·지도 나이·계절 적합성·지리적 구별성까지 반영한 persistence/distinctiveness 가중치를 쓰고, 후보 포즈의 불확실성과 그래프 일관성 및 VIO 잔차를 이용해 false fix 리스크 대비 수용률(risk–coverage) 관점의 거부/수용 판정을 제안한다.

- **Empirical Impact**: 실증은 실제 비행 closed-loop 성능을 검증하기보다는, 하드 decoy가 섞인 controlled 교차뷰 섭동에서 의미 기하가 위치를 구분하는지 “synthetic proof of concept”로 확인하는 데 초점을 둔다. 회전·스케일·부분 크롭·가림·지도 변경 시뮬레이션·혼동 가능한 decoy를 포함해 220개 무작위 재현 시험에서, global semantic descriptor는 Recall@1 58.6%에 그쳤지만 공간 의미 레스터 정렬 변형은 94.5–95.5%까지 크게 상승했다. Wilson 95% 구간은 descriptor와 공간 정렬 변형 간 분리는 보여주되, 공간 변형들 사이에는 구간 중첩이 있어 그래프/영속성/unknown 처리의 ‘추가 이득’을 통계적으로 확정하진 못한다; 다만 다음 단계로 필요한 aliasing·map-aging·거부 테스트를 더 어렵게 설계해야 한다는 점을 구체화한다.



### ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation (https://arxiv.org/abs/2607.08741)
Comments:
          ACM Transactions on Graphics (SIGGRAPH 2026)

- **Prior Approaches**: 기존 오프라인 3D 인간 모션 생성은 텍스트·운동학 제약을 정밀하게 반영할 수 있지만, 전체 시퀀스를 병렬 생성하는 구조 때문에 인터랙티브 환경에서 필요한 실시간 추론 속도가 부족한 경우가 많습니다. 반대로 온라인/스트리밍(autoregressive) 방식은 빠르지만 텍스트 의미를 복잡하게 해석하거나, 긴 지평(long-horizon) 제약을 안정적으로 만족시키는 데 한계가 있었고, 둘 다를 동시에 만족하는 접근도 제한된 컨텍스트 창 때문에 성능이 흔들렸습니다.

- **Core Contribution**: 본 논문은 텍스트 프롬프트와 유연한 kinematic constraints를 실시간으로 결합해 고충실도 모션을 생성하는 스트리밍 프레임워크 ARDY를 제안합니다. ARDY는 explicit root(루트) 제어는 명시적으로, body(몸통)는 latent embedding으로 압축해 두 표현을 혼합함으로써 정확한 궤적 제어와 효율적 생성학습을 동시에 노립니다.

- **Technical Challenges**: 핵심 과제는 (1) 온라인 조건이 바뀌어도 반응 가능한 autoregressive 생성에서 (2) sparse/불규칙한 제약을 입력으로 다루면서 (3) 긴 지평 목표까지 자연스럽게 연결되도록 컨텍스트를 설계하는 것입니다. ARDY는 variable history context를 갖는 two-stage autoregressive transformer denoiser를 사용해, 디노이징 루프 내에서 루트와 latent body를 상호 영향이 이어지도록 교차(interleaved) 예측하고, 제약은 masked motion sequence로 주입해 임의 시점/특징에 대한 장거리 목표를 네이티브로 학습합니다.

- **Empirical Impact**: HumanML3D 벤치마크와 고품질 대규모 Bones Rigplay 데이터셋에서 모션 품질과 제약 준수도가 함께 향상됨을 실증하며, 특히 설계한 혼합 표현과 two-stage 구조의 효과를 뒷받침합니다. 또한 마우스·키보드 기반 인터랙티브 데모에서 동적 텍스트 제어, 키프레임/패스 팔로잉, 장면 경로 추종 같은 사용 시나리오를 보여주며, 애니메이션·시뮬레이션·휴머노이드 로보틱스 쪽 실시간 제어 파이프라인의 현실적인 대안으로 주목받을 만합니다.



### Latent Memory Palace: Reasoning for Control as Autoregressive Variational Inferenc (https://arxiv.org/abs/2607.08724)
- **Prior Approaches**: 기존 연구는 LLM의 체인-오브-스로우(중간 추론 토큰)나 테스트-타임 스케일링처럼 반복 계산을 통해 성능을 끌어올리는 접근을 제안해 왔습니다. 하지만 로보틱스에서는 언어 토큰만으로는 공간 이해와 정밀 제어의 세밀함이 부족해, 동일한 방식을 그대로 이식하기 어렵다는 한계가 있었습니다. 또 일부는 명시적 중간 추론 단계나 수동 중단 기준, 혹은 고정 길이의 잠재표상으로 반복/적응성을 만들려 했지만, 계산을 입력마다 얼마나 쓰고 멈출지의 원리가 약했습니다.

- **Core Contribution**: 이 논문은 연속 제어에서의 “잠재 추론”을 Latent Memory Palace(LMP)로 정식화합니다. LMP는 가변 길이( EOS로 종료)의 오토레그레시브 잠재 시퀀스를 만들어 반복적으로 정보를 검색하듯 처리하고, 이를 변분 추론 관점에서 학습합니다. 또한 동일한 프레임워크를 관측 조건을 제거해 LMP-tok(가변 길이 action tokenizer)로 확장해, 다운스트림 오토레그레시브 정책의 성능도 함께 끌어올립니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 가변 길이 잠재 시퀀스 샘플링이 비미분이라 변분 목적을 그대로 최적화하기 어렵고, (2) 추론을 가능하게 하되 언제 멈출지(적응적 계산)가 비자동으로 남는다는 점입니다. 논문은 변분 하한을 잠재 토큰 궤적에 대한 RL 목표로 재구성해 비미분 경로를 따라 학습 가능하게 만들고, KL 항의 구조를 이용해 보상을 단계별로 분해하는 방식으로 분산을 줄였습니다. 더 나아가 디코더 분산을 잠재 길이에 따라 지수적으로 감쇠시키는 “압축(compression)”을 도입해, 행동 정밀도가 충분할 때만 추가 스텝을 사용하도록 하며 적응적 compute 배분을 유도합니다.

- **Empirical Impact**: 실험에서 LMP-π는 DROID(실세계), LIBERO/D3IL/RoboMimic(시뮬) 전반에서 기존 생성 제어 정책 대비 일관된 성능 우위를 보였고, 특히 하위 작업의 바닥 성능(floor)을 더 높여 작업 간 간섭이 줄어든 양상을 보였습니다. 또한 비반복 잠재변수 기반 대조군(VAE Policy, VQ-BeT)이 고정 길이 잠재표상의 과소 세밀함 때문에 뒤처지는 결과가 나와, 반복적/적응적 잠재 추론의 실질적 이점을 뒷받침합니다. LMP-tok 역시 FAST/VQ-VAE/OAT 계열 토크나이저를 제치고 특히 RoboMimic의 고정밀 도구 행(툴-행)에서 유의미한 성능을 보여, 로보틱스 다운스트림 autoregressive 정책 학습에 중요한 의미를 갖습니다.



### Early to Share, Late to Save: Synchronisation-Driven Communication Gating in Bandwidth-Constrained Cooperative VLN (https://arxiv.org/abs/2607.08504)
Comments:
          Accepted at the IJCAI 2026 GLOW Workshop. To appear in Springer Communications in Computer and Information Science (CCIS)

- **Prior Approaches**: 기존 cooperative VLN 연구(Co-NavGPT, CAMON 등)는 통신을 매 스텝 자유롭게 주고받는 가정이 많아, 실제 환경의 bandwidth 제한에서 ‘언제’ 통신할지 문제가 잘 다뤄지지 않았다. 또한 IC3Net 같은 일부 gated 방식은 REINFORCE로 게이트를 학습해, 게이트 결정과 에피소드 성공 사이의 긴 지연(credit assignment) 때문에 학습 분산이 커졌다.

- **Core Contribution**: 논문은 한 에피소드당 전송 예산이 있는 bandwidth-constrained cooperative VLN 문제를 정식화하고, 이를 위한 hindsight gating을 제안한다. 핵심은 통신 없이 먼저 주행해 실패 지점을 수집한 뒤, 파트너가 정답을 알고 있는지 여부로 통신-필요 스텝을 사후 라벨링하고 이를 BCE 분류로 학습한다는 점이다.

- **Technical Challenges**: 첫째, REINFORCE 기반 게이트 학습의 높은 분산을 피하면서도 장기 성과에 연결되는 통신 시점을 배워야 했다. 둘째, 게이트 입력에 불확실성(엔트로피/확률)을 직접 주지 않았는데도 어떤 스텝에 통신을 해야 하는 신호를 hidden state가 스스로 담아내야 했다; 논문은 GRU hidden state에 기반한 가벼운 3층 MLP 게이트로 이를 구현하고, 남은 전송 예산 b_rem(t)를 입력으로 넣어 예산 소진에 따른 발화율을 조절하도록 했다.

- **Empirical Impact**: 실험 결과, 학습된 게이트는 직관(불확실할 때 통신)과 달리 에피소드 초반 스텝에서 주로 발화하며, 동시에 더 높은 confidence에서 전송하는 패턴이 관찰됐다. R=2R val_unseen에서 Agent 0의 Success Rate는 전송 3회(B=3)로 no-communication(8.7%)을 넘어 8.9%를 기록했고, hidden-state alignment 누적 이득은 random gating 대비 +260%, entropy 기반 대비 +320%에 달했다. 저자들은 이 효과가 recurrent hidden-state alignment로 설명되며, 적은 전송으로도 unconstrained communication에 근접하도록 ‘초반 동기화 후 독립 항법’이라는 새로운 통신 레짐을 제시한다.



### Swapping Faces, Saving Features: A Dual-Purpose Pipeline for Pedestrian Privacy in ITS (https://arxiv.org/abs/2607.08402)
- **Prior Approaches**: 자율주행/ITS용 보행자 의도·궤적 예측은 다양한 보행자 영상 데이터가 필요한데, 얼굴은 생체인식 정보라 신원 탈취·추적·딥페이크 등 보안 위험을 키운다. 기존 방법은 블러/픽셀화처럼 얼굴을 가리는 방식이 많지만 훈련에 필요한 표정·시선 등 속성을 훼손해 데이터 유용성을 떨어뜨린다. GAN 기반 익명화는 프라이버시는 확보하더라도 속성 보존이 약해 AV 학습용 얼굴 단서 활용에 한계가 있었다.

- **Core Contribution**: 본 논문은 보행자 신원을 숨기면서도 표정·고개 자세·시선 같은 핵심 얼굴 속성을 유지하도록, face swapping 중심의 5단계 파이프라인을 제안한다. 이 파이프라인은 Egy-DRiVeS처럼 이집트 거리 영상의 저해상도·다양한 복장(예: 베일) 같은 특수 케이스에 맞춰 설계됐다. 또한 비교 실험을 통해 Roop과 Ghost-v2 중 파이프라인 적용에 더 적합한 face swapper를 선정한다.

- **Technical Challenges**: 핵심 난제는 (1) 저해상도/가림/먼 거리로 얼굴 정보가 부족한 상황에서 swapping 품질을 유지하고, (2) 프라이버리(정체 은폐)와 속성 보존(시선·표정·자세)을 동시에 달성하는 것이다. 이를 위해 보행자·얼굴 검출 후 Codeformer 기반 품질 보강으로 복원 신뢰도를 높이고, 블렌딩까지 포함한 end-to-end 처리 흐름을 구성했다. 평가 지표는 랜드마크/블렌드셰이프 차이, 얼굴 임베딩 유사도, gaze vector 유사도 등으로 “얼굴 구조·표정 유지 vs 동일성 은폐”를 함께 측정한다.

- **Empirical Impact**: 고품질 얼굴 테스트에서 Roop은 Ghost-v2 대비 4개 정량 지표 중 3개에서 우수하며, 특히 표정(블렌드셰이프)·얼굴 구조/자세(랜드마크 차이) 보존이 더 잘 나타났다. 또한 Occluded face(가림된 얼굴)와 베일 여성 같은 어려운 사례에서 Ghost-v2는 비현실/윤리적으로 문제 소지가 있는 결과를 보인 반면 Roop은 더 높은 견고성을 보여 파이프라인 신뢰성을 뒷받침했다. 최종 5단계 적용 후에는 JAAD의 looking/not looking(시선 방향) 특징이 파이프라인 전후로 유지되어, 프라이버리 보호가 다운스트림 의도 예측에 필요한 얼굴 단서까지 망치지 않는다는 점을 실증했다.



### Large-Language-Models-as-a-Judge in Theory-Agnostic Adaptive Metric-Alignment for Prototypical Networks in Personality Recognition (https://arxiv.org/abs/2607.08374)
- **Prior Approaches**: 기존 성격 인식 연구는 Big-5, MBTI처럼 특정 심리 이론의 라벨 체계에 맞춰 학습하는 경우가 많아, 데이터셋·문화권이 바뀌면 성능이 쉽게 흔들립니다. 또한 myPersonality처럼 대규모 공개 데이터가 프라이버시 이슈로 축소되면서 저자원 환경에서의 학습 한계가 커졌고, 성격 인식을 정적 분류로만 다루는 경향도 일반화를 제한했습니다.

- **Core Contribution**: 이 논문은 이론에 의존하지 않는 성격 추론을 목표로 JAM(Judge for Adaptive Metric-Alignment)을 제안합니다. JAM은 학습 과정에서 사전 정의된 성격 이론 라벨에 고정되지 않고, 텍스트로부터 공유된 잠재 심리 구조를 ‘latent pseudo-facets’로 발견해 개인의 잠재 심리 프로필을 추론합니다.

- **Technical Challenges**: 핵심 과제는 서로 다른 이론으로 라벨링된 이질 데이터에서 공통 구조를 잘 맞추면서도 라벨 노이즈·불확실성을 견디는 것입니다. 논문은 Attention-Pooled Graph Prototypical Network로 임베딩 공간에서 클러스터 기반의 구조 표현을 만들고, Cross-Theory Harmonization(인간 가이드 linkage + 기계 유도 consensus)으로 이론 간 정합을 강화하며, LLM-as-a-Judge를 LLM-before-the-loop/LLM-in-the-loop 두 방식으로 붙여 애매한 샘플을 선별해 적응적 메트릭 학습을 돕습니다.

- **Empirical Impact**: 실험에서는 JAM이 여러 프레임워크 간 일반화와 성능에서 기존 대비 개선을 보이며, 특히 low-resource personality theory에서도 유의미한 효과를 보였다고 보고합니다. 이는 성격 인식을 특정 심리 분류 체계에 종속시키던 관행에서 벗어나, 이론-불변적 표현 학습으로 확장하는 한 걸음을 제시한다는 점에서 의미가 있습니다.



### INTENT: An LSTM Framework for Vehicle Intention Prediction in Intersection Scenarios with Comprehensive Ablation Analysis (https://arxiv.org/abs/2607.08316)
- **Prior Approaches**: 자율주행에서 차량 의도 예측은 교차로·회전로·비상상황처럼 인간의 상호작용과 복잡한 주행 패턴이 많은 구간에서 안전과 민첩성을 좌우하는 핵심 요소로 다뤄져 왔다. 기존 접근은 주로 신호나 궤적 기반 추정에 의존하거나, 의도 정보를 궤적 예측에 조건으로 반영하는 방식으로 확장돼 왔지만, 실시간 의사결정에 바로 쓰기엔 예측 타이밍과 정확도 측면에서 제약이 남아 있었다.

- **Core Contribution**: 이 논문은 INTENT 프레임워크를 제안하며, LSTM 모델로 이벤트 발생 2초 전에 차량의 의도를 선제적으로 예측한다. 교차로 상황에서 직진/좌회전/우회전의 3가지 의도를 분류해, 회피 기동 등 후속 의사결정에 필요한 인간 수준의 의도 해석을 지향한다. 아울러 의도 조건부 trajectory prediction(의도-조건 궤적 예측) 관점에서 의도 예측이 궤적 예측 성능에도 기여할 수 있음을 함께 강조한다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 짧은 리드타임(2초)에 의도를 정확히 읽어내는 것과 (2) 교차로에서의 행동 변화가 빠르게 나타날 때도 안정적으로 분류하는 것이다. 연구진은 LSTM을 통해 시간적 문맥을 학습하고, 이벤트 직전의 관측에서 직진·좌회전·우회전 신호를 분리해 학습 안정성과 예측 신뢰도를 확보하는 방향으로 설계했다. 또한 다양한 실험과 ablation study(요인 제거 실험)로 구성요소의 기여를 검증했다.

- **Empirical Impact**: InD 데이터셋에서의 실험 결과, INTENT은 99.71% 정확도를 달성하며 의도 예측 성능을 실증적으로 입증했다. 2초 전 예측이라는 시간 제약을 만족하면서도 높은 분류 정확도를 보여, 실시간 안전 기동과 의사결정 파이프라인에 직접 활용될 가능성을 시사한다. 특히 복잡한 교차로 시나리오에서 의도 기반 후속 예측(예: 의도-조건 궤적 예측)으로 이어질 수 있다는 점에서 후속 연구에도 영향을 줄 것으로 보인다.



### Input-Constrained Spatiotemporal Tubes for Safe Navigation of Unknown Euler-Lagrange Systems in Dynamic Environments (https://arxiv.org/abs/2607.08189)
- **Prior Approaches**: 기존 안전 항법 연구는 경로 생성(A* , RRT)이나 반응 제어(APF)처럼 충돌 회피에 초점을 두되, 동역학 적합성과 입력 제약을 충분히 보장하지 못하는 경우가 많았습니다. MPC나 CBF-QP는 제약을 다루지만 모델 정확도와 온라인 최적화 의존으로 인해 동역학을 모르는 상황에서 과제 수준(reach-avoid) 안전 보장을 형식적으로 주기 어렵습니다. 한편 STT 기반 reach-avoid-stay 프레임워크는 unknown 시스템에 대해서도 근사 없이 닫힌형식 제어를 제공하지만, actuator input constraints를 명시적으로 설계에 반영하지 않아 포화로 인한 안전 위반 가능성이 남아 있었습니다.

- **Core Contribution**: 이 논문은 동역학이 불완전/미지인 Euler-Lagrange 시스템에 대해, 유한 시간(Finite-time) reach-avoid-stay(FT-RAS) 사양을 만족하면서 actuator 제한(|τ(t)|≤τ¯)을 항상 존중하는 실시간 제어 프레임워크를 제안합니다. STT(spatiotemporal tube) 설계를 입력 제약과의 “호환성” 관점으로 확장하고, 불확실성 범위와 제어 권한 사이의 관계를 오프라인에서 검증 가능한 형태로 도출합니다. 그 결과 controller는 approximation-free이며 계산 부담이 낮아 실시간 적용성이 강조됩니다.

- **Technical Challenges**: 핵심 난제는 unknown EL 시스템에서 관측 불완전성과 외란까지 고려하면서도, 제어 튜브가 안전영역(unsafe set) 회피를 보장하는 동시에 실제 actuator가 제공 가능한 제어 권한을 초과하지 않도록 tube의 중심/반경을 설계하는 것입니다. 논문은 tube 중심 c(t)를 목표지향 항과 장애물 근접 시 회피 항으로 piecewise하게 진화시키고, 튜브 반경 r(t)는 unsafe set 근접도에 따라 수축·팽창하도록 하여 충돌 회피와 목표 수렴을 함께 유지합니다. 또한 제어 입력 제약과 불확실성 바운드를 직접 연결하는 feasibility condition을 제시해, 그 조건이 만족될 때 전진 불변성과 FT-RAS 만족이 함께 증명되도록 구성합니다.

- **Empirical Impact**: 시뮬레이션에서는 mobile robot, quadrotor, spacecraft에서 unsafe 회피와 목표 도달이 actuator constraint 하에 안정적으로 수행됨을 보였습니다. 더 나아가 mobile robot 하드웨어 실험에서도 입력 제한을 위반하지 않으면서 안전 항법이 가능함을 실증해, 이론의 실시간 적용 가능성을 뒷받침합니다. 동역학이 unknown이고 제약이 까다로운 상황에서 “형식적 reach-avoid-stay 보장 + 입력 제약 준수”를 동시에 달성한다는 점에서 안전 제어 및 로보틱스 커뮤니티에 의미 있는 진전을 제공한다는 평가가 가능합니다.



### Understanding and Mitigating the Video-Action Generalization Gap via Temporal Ratio (https://arxiv.org/abs/2607.08127)
Comments:
          26 pages, 9 figures

- **Prior Approaches**: 생성형 비디오 파운데이션 모델을 로봇 제어에 옮기려는 접근은 크게 미래 비디오(또는 라텐트)를 예측하고 inverse dynamics로 행동을 추론하는 방식(WAM/VAM)과, 비디오·행동 토큰을 한 트랜스포머에서 함께 모델링하는 방식으로 나뉜다. 그러나 비디오 백본은 조합적 일반화가 강해도, 행동 데이터로 finetuning한 뒤에는 그 조합적 우선이 약해지며 OOD 성능이 붕괴하는 문제가 반복된다. 논문은 이 불일치를 video-action generalization gap(VAG gap)으로 명명한다.

- **Core Contribution**: 이 연구는 VAM 설계 선택(백본 finetuning, 학습 방식, 비디오 라텐트 추출/노이즈 수준, prediction horizon)이 조합적 일반화에 어떤 영향을 주는지 “설계 스페이스”로 체계적으로 진단한다. 그 결과 조합적 성공은 비디오가 그럴듯한 미래를 예측하는 것만으로는 부족하며, action head가 그 미래 롤아웃을 실제로 얼마나 참조하는지에 달려 있음을 보여준다. 이를 정량화하는 지표로 Temporal Ratio(TR)를 제안하고, TR을 활용해 inference-time에서 가이던스를 적응적으로 조절하는 TR-Adaptive Guidance를 제안한다.

- **Technical Challenges**: 핵심 난관은 action head가 미래 예측을 “참조하는지/무시하는지”가 겉으로는 영상 롤아웃만으로는 드러나지 않는다는 점이다. 논문은 action head의 attention을 현재 프레임(앵커)과 미래 프레임(예측 라텐트)로 분할해, 미래 라텐트에 대한 attention 질량을 TR로 정의함으로써 정책이 예측 모드로 이동하는 시점을 런타임 신호로 측정한다. 또한 TR이 태스크 단계에 따라 계획 구간에서는 상승하고 정밀 조작에서는 하락한다는 성질을 이용해, planning 단계에서 언어·계획(롱호라이즌) 가이던스 강도를 증폭하고 manipulation 단계에서는 완화하는 방식으로 잘못된 강화 가능성을 줄인다.

- **Empirical Impact**: LIBERO 벤치마크와 실제 이족/양팔 bimanual 태스크에서 기존 WAM/VAM 대비 조합적 OOD 성능 저하가 반복되는 “VAG gap”이 확인되며, TR과 TR-Adaptive Guidance로 이를 완화한다. LIBERO에서는 unguided VAM 대비 TR 기반 adaptive guidance로 평균 OOD 성공률이 59.4%까지 상승해 prior VAM 대비 큰 폭의 개선(논문은 5배 이상)을 보고한다. 실제 로봇 bimanual YAM에서도 평균 성공률이 71.7%에서 83.3%로 향상되며, 비디오 라텐트를 첫 denoising step에서 바로 쓰는 기존 방식은 다중 작업에서 실패함을 통해 TR 기반 설계의 실용성을 강조한다.



### EVIS: A Physics-Grounded Event Camera Plugin for NVIDIA Isaac Sim (https://arxiv.org/abs/2607.08098)
- **Prior Approaches**: 이전 이벤트 카메라 시뮬레이터/생성 방법은 주로 RGB 영상을 이벤트로 변환하거나(v2e, Vid2E, PIX2NVS 등) 사전 렌더링된 프레임에 의존해 물리 시뮬레이션과의 정합성이 떨어지는 경우가 많았다. 일부는 장면/모델 기반 렌더링(ESIM, Blender 기반 고전 시뮬레이터, Gazebo 플러그인 등)을 시도했지만, GPU-병렬 물리 엔진(Isaac Sim) 안에 이벤트 생성이 통합되는 수준은 제한적이었다. 그 결과 로봇·장면별로 동기화된 고정밀 이벤트와 정답(ground truth)을 대규모로 만들기 어렵다는 문제가 남았다.

- **Core Contribution**: EVIS는 NVIDIA Isaac Sim/Isaac Lab 내부에서 직접 이벤트 스트림을 생성하는 ‘physics-grounded event camera plugin’으로, 로봇과 장면의 프레임-정확 ground truth를 이벤트와 함께 산출한다. 사실적인 log-intensity contrast 이벤트 모델을 구현하고, Isaac Sim에서 RGB 카메라 설정을 이벤트 카메라 설정으로 바꾸는 방식으로 어떤 물리 시뮬 장면에도 쉽게 적용된다. 또한 motion-vector 기반 프레임 보간 옵션으로 RTX 핵심 프레임만 렌더링하고 중간 이벤트를 합성해 단일 GPU에서 실시간 생성까지 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 고주사율에서의 대비 트리거링 모델의 충실도(잡음, 비이상 포함)와 (2) 물리 시뮬에 기반한 정확한 정답의 동시 제공, (3) 생성 처리량을 실시간 수준으로 유지하는 것이다. EVIS는 per-pixel asynchronous reference latching을 포함한 log-intensity contrast 이벤트 모델을 GPU 배치로 구현하고, threshold mismatch·refractory period·leak events·shot noise·hot pixels·finite bandwidth 같은 센서 비이상을 옵션으로 추가한다. 프레임 보간은 렌더러의 motion vector를 이용해 forward/backward warp와 softmax splatting으로 중간 프레임을 합성하며, anti-aliasing 충돌로 생길 수 있는 고스트를 보간 활성 시 비활성화해 품질을 보완한다.

- **Empirical Impact**: 단일 RTX 5090에서 렌더링 병목을 보간 계수에 따라 줄이며, 예컨대 240Hz 이벤트 생성이 실시간보다 빠르게(약 1.2×) 가능함을 보였다. 생성된 이벤트는 E2VID(재구성), E-RAFT(밀집 옵티컬 플로우), Match-Any-Events(다중 뷰 매칭) 같은 사전학습 네트워크에 대해 fine-tuning 없이 그대로 적용해도 합리적인 성능을 보였고, interpolation이 공격적으로 커질수록 재구성 SSIM/LIPS 등은 점진적으로 악화되었다. 특히 Isaac Sim의 정확한 motion vector를 이용해 ground truth 밀집 플로우를 별도 추정/보간 없이 평가할 수 있었고, 키프레임이 더 희소해져도 E-RAFT의 서브픽셀 수준 복원이 비교적 견고하게 유지되어 해당 분야의 데이터/실험 확장성에 의미가 크다.



### Post-Training in End-to-End Autonomous Driving (https://arxiv.org/abs/2607.08072)
- **Prior Approaches**: 자율주행에서 엔드투엔드(end-to-end) 모델은 카메라·명령·상태 같은 멀티모달 입력을 받아 미래 궤적/기동을 직접 출력하며, 행동모방(behavior cloning), 궤적 생성 플래너, VLA(Vision-Language-Action)로 발전해 왔다. 그러나 안전·상호작용이 강한 환경에서는 오픈루프(open-loop) 모방만으로는 신뢰성을 보장하기 어렵고, 작은 실행 오차가 장기적으로 누적되어 회복 데이터도 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 자율주행 ‘포스트트레이닝(post-training)’을 초기 모방학습 이후에, 시뮬레이션/검증/선호/보상 등 추가 감독을 통해 이미 학습된 정책을 더 다듬는 별도 단계로 정의한다. 또한 감독 형태를 기준으로 distillation, preference-based alignment, reinforcement learning, test-time refinement의 네 계열로 문헌을 통합 분류해 체계적 비교가 가능하게 만든다.

- **Technical Challenges**: 포스트트레이닝 목표(안전·컴포트·규칙준수·진행·상호작용)는 점별 라벨로 완전한 최적화식으로 환원되기 어렵기 때문에, 서로 다른 감독 신호(교사, 선호 쌍, 스칼라 리워드, 테스트-타임 검증)를 설계해야 한다. 논문은 특히 폐루프 분포 shift와 오차 누적을 줄이기 위해 교사 신호를 학생 롤아웃에서 라벨링하는 분포 정합 관점(distillation), 의미 있는 trade-off를 담는 선호쌍 설계(preference), 다차원·장기·희소 실패를 반영하는 보상/롤아웃 구성(RL), 그리고 파라미터 업데이트 없이 후보 선택·검증으로 품질을 끌어올리는 test-time refinement을 핵심 해법으로 정리한다.

- **Empirical Impact**: 기존 모방 기반에서 더 나아가 폐루프 성능과 회복·안전성 측면을 개선하기 위한 방법들이 빠르게 확산되고 있으며, 논문은 이 흐름을 ‘포스트트레이닝 파이프라인’으로 재정리해 연구 간 단절을 완화한다. 동시에 벤치마크 포화, 폐루프 지표의 거칠음, 실차 평가 제약, 추론 비용 같은 평가/운영 과제를 짚고, 신뢰적이고 효율적인 포스트트레이닝 연구의 다음 방향을 제시한다.



### APIVOT: Adaptive Planning with Interleaved Vision-Language Thoughts (https://arxiv.org/abs/2607.08024)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 장기 로봇 조작 계획은 작업의 의미 구조(목표 분해, 행동 순서)와 기하 제약(여유 공간, 충돌)까지 함께 고려해야 한다. 기존 LLM/VLM 플래너는 언어적으로는 그럴듯한 행동 순서를 잘 만들지만, 공간 제약 때문에 실행 불가능한 계획을 내는 경우가 많다. 이를 보완하려고 외부 모션 플래너/feasibility checker/학습된 동역학 모델을 붙여 재계획을 유도했지만, 정작 모델 내부의 추론 과정에는 기하적 판단이 ‘후처리’처럼 분리되어 있었다.

- **Core Contribution**: APIVOT는 VLM 기반 플래너로, 언어 추론(semantic reasoning)과 시각적 사고(visual thoughts)를 계획 추론 흐름(trace) 안에서 적응적으로 섞는다. visual thoughts는 상상된 미래 상태로서 이후 단계의 기하적 실행 가능성(공간 적합, 충돌 회피)을 내부에서 검증하는 데 쓰인다. 핵심은 매 단계에서 언어/비전 중 어느 모달리티가 필요한지 모델이 학습해, 불필요한 시각 추론을 줄이면서도 성공률을 높인다는 점이다.

- **Technical Challenges**: 문제는 ‘언어로는 정교한 기하 구조를 간결하게 표현하기 어렵다’는 점과, 반대로 시각적 사고를 매번 쓰면 계산 비용이 커진다는 점이다. APIVOT는 (1) 제공된 visual thoughts를 활용해 기하 정보를 추출하는 단계, (2) visual thoughts를 스스로 생성하도록 하는 단계, (3) 기하 제약이 중요한 단계에서만 visual thoughts를 생성하도록 하는 3단계 커리큘럼으로 적응성을 학습한다. 또한 visual thought가 가리키는 잠재 시각 상태를 정답 미래 장면 특징과 정렬(cosine similarity)해, 상상된 상태가 실제 장면과 맞물리도록 설계했다.

- **Empirical Impact**: KitchenWorlds의 장기 키친 태스크에서 APIVOT는 평균 작업 성공률 0.419를 기록하며 최강 베이스라인 대비 8.1%p 향상했다. 특히 공간 제약이 강한 설정에서 격차가 7점에서 17점으로 벌어져, 내부 기하 검증의 효과가 크게 드러났다. 더불어 visual thoughts를 모든 단계에 쓰는 상한 대비 성능의 91%를 유지하면서 토큰 사용은 39% 줄여, 모달리티 선택 학습이 성공률과 추론 효율을 동시에 개선함을 실증했다.



### Idiobionics: The Unification of Privacy and Intelligent Robotic Prostheses (https://arxiv.org/abs/2607.07775)
Comments:
          8 pages, 3 figures

- **Prior Approaches**: 로봇 의수(bionic limbs)는 IoB(Internet of Bodies) 흐름 속에서 EMG·가속도계·자이로 등 센서와 ML 제어로 사용자의 의도를 추정하고 적응하도록 발전해 왔다. 다만 기존 IoB 보안/프라이버시 연구는 웨어러블(스마트워치, 피트니스 트래커) 중심이었고, 의료기기로 분류되어 사용자가 교체·회피하기 어려운 의수의 특수성은 상대적으로 공백이었다. 그 결과, 자율·반자율 적응성이 높아질수록 어떤 프라이버시 위협이 새로 생기는지에 대한 체계적 이해가 부족했다.

- **Core Contribution**: 이 논문은 프라이버시와 지능형 의수의 접점을 다루는 새로운 연구 의제인 idiobionics를 제안하고 정의한다. 이어서 상지 의수의 대표적 설정을 대상으로 공격 가능성을 보여주며, 향후 연구를 위한 공개 질문 리스트를 정리해 착수 장벽을 낮춘다. 즉, ‘기능 향상’과 별개로 ‘프라이버시 위험’을 설계 단계에서 선제적으로 평가하자는 관점을 제시한다.

- **Technical Challenges**: 핵심 난제는 의수의 적응성을 가능하게 하는 센싱·학습 구조가 역으로 민감 정보를 추론하는 공격 표면(threat vectors)을 만든다는 점이다. 저자들은 가속도계 데이터로 Activity Inference Attack(AIA)이 가능한지 확인하기 위해, 상지(팔꿈치 아래) 절단 사용 상황에 맞춘 실험 데이터를 수집하고 지도학습(전이학습 HarNet10 기반 finetuning)과 비지도학습(클러스터링)을 함께 적용했다. 그 결과 데이터가 제한적일 때도 공개/소비자급 도구로 활동 추론이 성립함을 보이며, 향후 보호 설계가 특정 사용자 하위그룹에 맞춰 일반화돼야 함을 시사한다.

- **Empirical Impact**: 실험에서는 활동 분류 정확도가 평균 83%로 나타나, 사전에 학습된 공개 모델을 활용한 프라이버시 공격이 현실적임을 보여준다. 또한 개인별 정확도 변동이 커 일부 사용자는 더 쉽게 식별되어 보호 전략을 ‘취약 하위그룹’ 관점에서 설계해야 한다는 결론을 제시한다. 저자들은 이 위협이 활동뿐 아니라 건강 상태, 인구통계, 심지어 진동 기반 음성/키 입력 계열 공격으로 확장될 수 있으며, 더 나아가 재활·보조 로봇(엑소스켈레톤)에도 전이될 수 있다고 전망한다.



New uploads on arXiv(cs.MA)

### Early to Share, Late to Save: Synchronisation-Driven Communication Gating in Bandwidth-Constrained Cooperative VLN (https://arxiv.org/abs/2607.08504)
Comments:
          Accepted at the IJCAI 2026 GLOW Workshop. To appear in Springer Communications in Computer and Information Science (CCIS)

- **Prior Approaches**: 기존 cooperative VLN 연구(Co-NavGPT, CAMON 등)는 통신을 매 스텝 자유롭게 주고받는 가정이 많아, 실제 환경의 bandwidth 제한에서 ‘언제’ 통신할지 문제가 잘 다뤄지지 않았다. 또한 IC3Net 같은 일부 gated 방식은 REINFORCE로 게이트를 학습해, 게이트 결정과 에피소드 성공 사이의 긴 지연(credit assignment) 때문에 학습 분산이 커졌다.

- **Core Contribution**: 논문은 한 에피소드당 전송 예산이 있는 bandwidth-constrained cooperative VLN 문제를 정식화하고, 이를 위한 hindsight gating을 제안한다. 핵심은 통신 없이 먼저 주행해 실패 지점을 수집한 뒤, 파트너가 정답을 알고 있는지 여부로 통신-필요 스텝을 사후 라벨링하고 이를 BCE 분류로 학습한다는 점이다.

- **Technical Challenges**: 첫째, REINFORCE 기반 게이트 학습의 높은 분산을 피하면서도 장기 성과에 연결되는 통신 시점을 배워야 했다. 둘째, 게이트 입력에 불확실성(엔트로피/확률)을 직접 주지 않았는데도 어떤 스텝에 통신을 해야 하는 신호를 hidden state가 스스로 담아내야 했다; 논문은 GRU hidden state에 기반한 가벼운 3층 MLP 게이트로 이를 구현하고, 남은 전송 예산 b_rem(t)를 입력으로 넣어 예산 소진에 따른 발화율을 조절하도록 했다.

- **Empirical Impact**: 실험 결과, 학습된 게이트는 직관(불확실할 때 통신)과 달리 에피소드 초반 스텝에서 주로 발화하며, 동시에 더 높은 confidence에서 전송하는 패턴이 관찰됐다. R=2R val_unseen에서 Agent 0의 Success Rate는 전송 3회(B=3)로 no-communication(8.7%)을 넘어 8.9%를 기록했고, hidden-state alignment 누적 이득은 random gating 대비 +260%, entropy 기반 대비 +320%에 달했다. 저자들은 이 효과가 recurrent hidden-state alignment로 설명되며, 적은 전송으로도 unconstrained communication에 근접하도록 ‘초반 동기화 후 독립 항법’이라는 새로운 통신 레짐을 제시한다.



### From Triggers to Emotions: A CPM-Grounded Appraisal Multi-Agent for Dynamic Emotional Evolution in Persona-Based Dialogu (https://arxiv.org/abs/2607.07824)
- **Prior Approaches**: 기존 persona 기반 롤플레이 대화는 인물의 감정을 고정된 persona 특성이나 prompt 수준 제어 신호로 다루는 경우가 많았다. 감정 대화 연구 역시 주로 사용자의 감정을 인식하고 공감/정서적 지지를 생성하는 데 초점이어서, 에이전트(캐릭터) 자신의 감정이 대화 자극에 따라 어떻게 변하는지 모델링은 상대적으로 부족했다. 그 결과 감정 표현이 과도하게 긍정적으로 증폭되거나 공식적인 패턴에 머물 수 있다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Component Process Model(CPM)을 감정의 ‘고정 라벨’이 아니라 ‘외부 사건에 대한 appraisal의 동적 과정’으로 보고, 이를 persona 기반 대화에 접목한다. CPM-MultiAgent는 캐릭터 감정을 고정 속성이 아닌 잠재 상태(latent emotion state)로 두고, 대화 트리거에 의해 지속적으로 재형성되도록 설계했다. 또한 Trigger 분석–CPM 기반 다차원 appraisal–감정 상태 업데이트–일관성 비평(critic)까지의 다단계 파이프라인으로 감정 진화의 해석 가능성과 일시적 일관성을 동시에 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 대화 턴에서 감정 변화를 유발하는 트리거를 안정적으로 추출하고, (2) 그 트리거를 캐릭터의 persona/목표/이전 정서 맥락에서 심리적으로 그럴듯하게 평가하며, (3) 업데이트된 감정 상태가 다음 응답 생성과 충돌하지 않게 보정하는 것이다. 이를 위해 Trigger Analyzer가 감정적 자극을 structured trigger로 만들고, 네 개의 CPM appraisal agents(관련성·함의·대처가능성·규범적 의미)가 peer review로 서로의 판단을 조정한 뒤 Integration Agent가 잠재 감정 상태를 업데이트한다. 마지막으로 Critic Agent가 CPM 충실성·맥락 근거·시간적 일관성을 점검하고 필요 시 수정(재생성)을 거치도록 했다.

- **Empirical Impact**: 실험에서는 의료/교육/고객서비스 롤플레이의 24개 트라이얼에서 감정 상태 업데이트 품질을 baseline 대비 비교했고, LLM-as-judge와 103명의 human evaluation 모두에서 CPM-MultiAgent가 전반적으로 가장 좋은 성능을 보였다. 특히 Appraisal Reasoning Quality와 Trigger Grounding, Temporal Consistency에서 두드러진 개선이 관찰되며, 감정 변화가 이전 상태와 트리거에 근거해 ‘설명 가능한 방식으로’ 일어난다는 점이 확인됐다. ablation 결과와 다양한 LLM backbone(GPT-5.4 계열, Qwen3.6-35B-A3B)에서의 견고성 실험도 구조화된 multi-agent 분해의 기여를 뒷받침한다.



### Collective Intelligence with Foundation Models (https://arxiv.org/abs/2607.07729)
Comments:
          Accepted as a book chapter in "Advances in Global Applied Artificial Intelligence" (G. A. Tsihrintzis, M. Virvou, N. G. Bourbakis, L. C. Jain, Eds.), authenticated version will be published in Springer series: Learning and Analytics in Intelligent Systems

- **Prior Approaches**: 기존 연구는 chain-of-thought, self-consistency처럼 단일 모델 내부에서 다양한 추론 경로를 만들거나 답을 선택하는 방식에 집중해 왔습니다. 또한 AutoGen, MetaGPT 같은 멀티에이전트 협업도 있었지만, 다수 에이전트의 ‘오류 탐지’와 ‘합의 품질’을 정량적으로 분해해 설명하기는 어려웠습니다. self-refinement이나 Constitutional AI는 자체 비판을 사용하지만, 같은 모델의 편향이 그대로 남아 맹점 교정이 제한될 수 있다는 문제가 제기됩니다.

- **Core Contribution**: 이 논문은 서로 다른 foundation model을 역할별로 배치한 멀티에이전트 추론 프레임워크를 제안합니다. Solver가 독립 초안을 만들고, Critic이 구조화된 오류 분석과 수정안을 제공하며, Aggregator가 합의해 최종 답을 생성합니다. 더불어 의미·수치·절차(단계별 reasoning)까지 함께 평가하는 Scoring 모듈을 도입해, 최종 정답뿐 아니라 중간 과정의 질을 드러냅니다.

- **Technical Challenges**: 기여를 실현하려면 (1) 에이전트 간 중복 없이 ‘실제로 다른 오류’를 포착할 수 있는 모델 조합, (2) 비평·수정·합의가 잡음이 아닌 논리 교정으로 이어지게 하는 구조화, (3) 단계별 추론 품질을 측정할 수 있는 평가 설계가 필요합니다. 저자들은 solver마다 독립 샘플 초안을 만들고, critic은 스타일이 아닌 논리/계산 오류를 겨냥한 구성 프롬프트로 1회 비판만 수행하며, aggregator는 다수 합의 근거를 우선하되 불일치 시 불확실성을 명시하도록 했습니다. 평가는 임베딩 유사도(semantic coherence), 수치 추출 기반 overlap, 그리고 reference의 단계와 대응시킨 step-wise accuracy로 수행했습니다.

- **Empirical Impact**: 8개 과학·수학 분야 벤치마크에서 ablation 결과, 전체 평균 성능은 구조와 중복 샘플링만으로는 소폭(예: 0.52→0.60/0.61) 개선되는데 그칩니다. 반면 heterogeneous(모델 다양성) 구성은 step-wise accuracy가 0.64로 크게 상승해 homogeneous(0.27~0.28) 대비 2.3배 수준의 개선을 보였고, 중간 단계의 정합성까지 함께 좋아졌습니다. 즉 “정답 맞힘”을 넘어 추론 과정의 설명가능성·감사가능성을 강화하는 데 모델 다양성이 핵심이라는 점을 실증적으로 입증해, 고신뢰 과학/산업 의사결정에 대한 멀티에이전트 설계 방향을 제시합니다.



### WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search (https://arxiv.org/abs/2607.08662)
Comments:
          Work in progress

- **Prior Approaches**: ReAct 기반 단일 에이전트는 한 번에 긴 탐색 궤적을 수행해야 해 컨텍스트 한계로 인해 깊이·너비가 동시에 커지면 성능이 급격히 떨어진다. 다중 에이전트 접근은 병렬 탐색과 결과 집계를 통해 커버리지는 늘리지만, 재귀적 깊이 확장·협업 형태 전환·증거에 근거한 확장이 유연하지 못하다는 한계가 있다. 또한 작업을 쿼리 표면 의미에만 맞춰 분해하면 웹에서 정보가 실제로 조직되는 방식과 어긋나 중복 탐색과 집계 실패가 발생할 수 있다.

- **Core Contribution**: WebSwarm은 추론 중에 에이전트 노드를 동적으로 만들어 “점진적 재귀 위임(progressive recursive delegation)”을 수행하는 프레임워크를 제안한다. 각 노드는 로컬 목표와 search mode(원자적 탐색 atom, 반복 탐색 deep, 병렬 분할 wide, 미지 경계 열거 entity_collect)를 함께 받아, 스스로 풀거나 자식 노드를 위임해 증거를 상향 반환한다. 또한 웹 정보 구조 탐사와 동종 형제 노드 간 프로세스 경험 재사용을 통해, 분해·확장·협업을 증거가 쌓일수록 함께 진화시키도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 초기 쿼리만으로는 필요한 하위 과제가 언제 드러날지 예측하기 어려워 재귀 깊이를 안정적으로 키우는 것, (2) 하위 목표마다 필요한 협업 프로토콜이 달라 고정된 협업 토폴로지로는 부족한 것, (3) 웹에서의 정보 조직 방식과 분해 축이 어긋날 때 비효율이 누적되는 것이다. WebSwarm은 objective–mode pair로 하위 노드별 해결 전략을 명시하고, 하위 노드가 증거를 반환한 뒤 상위 노드가 expand·revise·terminate를 선택하는 피드백 루프를 구현했다. 더불어 wide 확장이 필요한 노드에는 웹-구조 probing을 붙여 확장 축을 정렬하고, scout 노드로부터 동종 sibling의 실패/성공 경로를 요약한 경험 kv를 재주입해 중복 시행착오를 줄인다.

- **Empirical Impact**: BrowseComp-Plus, WideSearch, DeepWideSearch, GISA 등 4개 벤치마크에서 WebSwarm은 단일 ReAct과 다양한 다중 에이전트 기준선을 일관되게 능가하거나 경쟁력 있는 성능을 보였다. 특히 깊이/너비가 섞인 DeepWideSearch처럼 단계 전환이 잦은 과제에서 고정 협업 패턴의 약점을 모드 기반 재귀 위임으로 완화했다. ablation 결과로도 재귀 위임 자체가 성능 저하를 유발하고, 웹-구조 probing은 도구 호출 수를 크게 줄이며, 동종 노드 경험 재사용은 품질(Item F1 등) 하락을 막는 역할을 확인했다.



### Multi-Agent Firewall Architecture for Privacy Protection of Sensitive Data in Interactions with Language Models (https://arxiv.org/abs/2607.08282)
- **Prior Approaches**: 기존 LLM 보안은 (1) SDK처럼 앱 수정으로 프롬프트를 검사하거나, (2) 외부 중앙형 안전검증 서비스에 프롬프트를 보내는 방식, (3) 사내 전용 모델로 클라우드 전송을 차단하거나, (4) 로컬에서 사용자가 키워드를 수동/전처리로 바꾸는 방식이 주류였다. 이들은 주로 ‘가로채기 공백’(interception gap), ‘프라이버시 역설’(검증을 위해 외부 벤더에 데이터 전송), 또는 ‘지능 격차’(오픈 가중치 성능 한계) 문제를 남겼다. 또한 토큰 단위 탐지는 가능해도, 문맥 전체가 의미하는 맥락적 유출 위험(contextual exfiltration)까지 안정적으로 다루기 어렵다는 한계가 컸다.

- **Core Contribution**: 이 논문은 오픈소스이면서 개인정보 중심의 로컬-first “LLM 데이터 누출 방지 방화벽”을 제안한다. 브라우저 확장 + 투명 MiTM 프록시를 결합해 웹 기반 상호작용과 HTTP(S)/WebSocket API 트래픽을 함께 가로채며, 결정적 탐지와 LLM 기반 의미 분석을 하이브리드로 묶어 누출을 막는다. Git 코드 유출 방지(저장소 인덱싱 + 퍼지 매칭)와 위험도 기반 정책(경고/차단/자동 마스킹/사용자 override)까지 포함해 조직 환경에서 커스터마이즈 가능하도록 설계했다.

- **Technical Challenges**: 핵심 기술 과제는 ① 애플리케이션 수정 없이 사용자 입력 경로를 정확히 가로채는 것, ② 문맥 기반 의미 유출을 낮은 지연/비용으로 잡아내는 것, ③ 이미지·파일 등 비정형 입력에서 재현 가능한 추출 파이프라인을 만드는 것이었다. 저자들은 센서(확장/프록시)와 분석 엔진을 분리하고, DAG 기반 멀티에이전트 파이프라인으로 저비용 탐지(정규식/키워드/체크섬/NER/코드 유사성)를 먼저 돌린 뒤 필요할 때만 LLM 의미 분석·VLM 추출로 단계적 에스컬레이션을 수행해 지연을 통제한다. 또한 OCR은 Tesseract의 confidence 기반 라우팅으로 처리하고 실패 시 VLM으로 이중 계층 추출을 수행하며, 파이프라인 토폴로지와 탐지 규칙을 JSON으로 외부화해 배포 후 정책 변경을 쉽게 했다.

- **Empirical Impact**: 평가 결과, 최적 구성에서 최대 F1 94.93%까지 성능을 달성해 결정적 탐지+의미 분석 하이브리드의 효과를 보여줬다. 계층형(early block) 구조와 공급자 비의존(provider agnostic) 설계를 통해 웹/프로그램 환경 모두에서 차단 정확도와 운영 비용·지연 간 균형을 맞출 수 있음을 강조한다. 보안 연구 관점에서는 OWASP LLM01/LLM02 맥락의 ‘사용자-LLM 상호작용 데이터 누출’ 문제를 실사용 가능한 형태로 구현했다는 점에서, 조직 내 배포형 LLM 보안 플랫폼 방향에 영향을 줄 것으로 보인다.



### ASMR: Agentic Schema Generation for Ship Maintenance Report Writing (https://arxiv.org/abs/2607.08177)
Comments:
          Accepted at the DASHSys 2026 workshop (Systems for Data-centric Agents with Human-in-the-loop), co-located with VLDB 2026

- **Prior Approaches**: 기존에는 보고서 작성자가 미리 정의한 양식이나 수작업 템플릿에 의존해 스키마를 구성하는 경우가 많았습니다. 자동 스키마 생성은 주로 단일 수준의 단어·구문 패턴에 기대는 방식이 많아, 보고서 유형별 핵심 정보 요구를 압축하고 중복을 줄이는 데 한계가 있었습니다.

- **Core Contribution**: 이 논문은 자동 스키마 생성 문제를 다루며, 다양한 범주의 과거 선박 정비·운용 보고서로부터 보고서 유형별로 “간결하면서도 정보량이 높은” 스키마를 발견하는 것을 목표로 합니다. 제안하는 ASMR은 Field Generation Agent와 Structural Optimizer Agent의 모듈형 agentic 프레임워크로, 필드 후보 도출과 스키마 구조 최적화를 분리해 성능을 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 (1) 서술형 문장에서 의미 개념을 안정적으로 추출해 다양한 수준의 필드 후보를 만들고, (2) 그중에서 RL로 “컴팩트·정보성·비중복”을 만족하는 구조를 찾아내는 것입니다. 저자들은 Field Generation에서 적응형 multi-granularity clustering으로 후보 필드를 생성하고, Structural Optimizer에서 reinforcement learning으로 스키마 표현을 반복적으로 개선하도록 설계했습니다.

- **Empirical Impact**: 초기 실험 결과는 제안한 ASMR이 보고서 유형별 요구를 반영하는 간결한 스키마를 생성할 가능성을 보여줍니다. 또한 생성된 스키마는 작성자에게 완전성·일관성·실행 가능성을 높이는 가이드로 활용될 수 있으며, data management, agentic AI, human-centered AI의 교차 지점에서 추가 연구 주제를 제시합니다.



### Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems (https://arxiv.org/abs/2607.07989)
Comments:
          To appear in COLM 2026

- **Prior Approaches**: LLM 기반 멀티에이전트 시스템의 오류는 에이전트 간 상호작용이 길어지면서 원인 에이전트와 ‘최초로 돌이킬 수 없게 만든 단계’를 특정하기 어렵다. 기존 실패 로컬라이제이션은 AgentTracer처럼 counterfactual replay로 인과를 보려 하지만, 다중 에이전트 환경에서는 한 단계 변경이 이후 프롬프트·툴·조정 흐름까지 흔들어 결론이 흔들릴 수 있다. 또 AEGIS처럼 사전 정의된 오류 패턴(템플릿/택소노미)에 매칭하는 방식은 예측 불가능한 emergent reasoning이나 조정 붕괴를 충분히 포착하지 못하고, poisoning forensics 계열도 멀티에이전트의 점진적 오인·coordination drift 특성에 맞지 않아 성능이 제한된다.

- **Core Contribution**: 이 논문은 실패를 특정 에이전트뿐 아니라, 궤적이 최초로 decisively misdirected 되는 ‘가장 이른 단계’까지 함께 귀속(attribution)하는 문제를 정식화하고, 이를 위한 프레임워크 AgentLocate를 제안한다. AgentLocate는 LLM 기반 Judge가 (responsible agent, earliest decisive step) 가설을 먼저 내고, 이를 검증 가능한 절차로 다단계화해 디버깅에서 재현성과 신뢰도를 높이는 방향을 택한다. 또한 Judge의 1회 출력이 아니라 다수 평가자 판단과 피드백을 수집해, 판단 품질을 lightweight fine-tuning으로 지속 개선한다.

- **Technical Challenges**: 핵심 기술 난제는 long-horizon, tool-mediated, tightly coupled 상호작용에서 ‘어떤 에이전트의 어떤 순간’이 전역 실패를 되돌릴 수 있는지 인과적으로 분리하는 것이다. AgentLocate는 먼저 all-at-once 또는 step-by-step으로 궤적을 해석해 counterfactual reversal 조건을 근사하는 가설을 만들고, 그 가설에 대해 base/concise/evidence-focused처럼 서로 다른 스타일의 independent Evaluators가 동일 궤적을 재분석하도록 한다. 이어 각 평가자의 self-reported confidence를 반영한 confidence-aware voting으로 후보 위치를 집계하고, Judgeft 학습에는 evaluator의 비판·불일치를 LoRA 기반(parameter-efficient) fine-tuning 신호로 사용해 향후 인과 정렬을 강화한다.

- **Empirical Impact**: AgentLocate는 Who&When 및 Aegis-Bench의 두 벤치마크에서 에이전트 수준 정확도뿐 아니라, Who&When에서는 failure step까지 포함해 기존 failure localization 방법을 일관되게 능가한다. 또한 토큰 사용량과 실행 시간 측면에서 효율성을 유지하면서, 단일 Judge만 쓰는 경우보다 로컬라이제이션 정밀도가 개선됨을 보인다. 멀티에이전트 신뢰성 분석과 디버깅 자동화를 한 단계 진전시켜, 시스템 수준 장애의 원인 추적을 more verifiable하게 만드는 데 의미가 있다.




New uploads on arXiv(cs.CL)

### AdaSR: Adaptive Streaming Reasoning with Hierarchical Relative Policy Optimization (https://arxiv.org/abs/2606.14694)
- **Prior Approaches**: 기존 대다수 추론 모델은 입력을 모두 받은 뒤(reason-then-read가 아니라 read-then-think) 정적 문맥에서만 사고를 시작합니다. 반면 스트리밍 추론 연구들은 읽는 동안 생각하기를 시도했지만, 사전에 만든 스트리밍 궤적을 따라가는 지도학습/모방학습 비중이 커 유연성이 떨어지고 ‘부분 입력에서 언제 얕게/깊게 생각할지’ 같은 적응 능력이 제한됩니다. 또한 단일(시퀀스-레벨) 크레딧을 전 토큰에 균등 배분하면 스트리밍 단계의 로컬 판단과 최종 단계의 전역 통합을 구분하기 어렵습니다.

- **Core Contribution**: AdaSR은 스트리밍 입력이 도착하는 동안에는 ‘언제 생각할지/얼마나 계산을 쓸지’를 학습하고, 스트림이 끝난 뒤에는 최종 숙고를 수행하는 적응형 프레임워크를 제안합니다. 핵심은 Hierarchical Relative Policy Optimization(HRPO)로, 스트리밍 단계(로컬 결정)와 딥 추론 단계(전역 통합), 그리고 전체 정답 성능에 서로 다른 학습 신호를 주어 시간적 크레딧 할당 문제를 해결합니다. 여기에 형식 보상, 정확도 보상, 지연/효율을 반영한 적응적 사고 보상을 결합해 최종 성능과 지연을 동시에 다룹니다.

- **Technical Challenges**: 기여를 현실화하는 가장 큰 난제는 ‘부분 관측 하에서 생성된 토큰들이 정답에 얼마나 기여했는가’를 단계별로 올바르게 평가하는 시간적 크레딧 할당이었습니다. 논문은 GRPO의 그룹 상대 정책 최적화 흐름을 유지하되, 균등 시퀀스-레벨 어드밴티지를 스트리밍 로컬/딥 로컬/롤아웃 글로벌로 분해해 토큰 구간에 계층적으로 배치하는 방식(HRPO)을 설계했습니다. 또한 <EOT>/<EOR> 같은 추론 프로토콜 형식을 강제하는 포맷 보상과, 생성 길이 및 스트리밍/딥 단계의 지연 비대칭을 고려한 길이-지연 보상으로 ‘계산을 언제/얼마나’ 쓸지까지 학습하게 했습니다.

- **Empirical Impact**: 실험에서 AdaSR은 Qwen3 계열 모델(예: Qwen3-1.7B, Qwen3-4B)에서 수학·문맥QA·논리 추론 벤치마크 전반에 걸쳐 정확도와 스트리밍 지연/연산 효율 사이의 균형이 개선됨을 보였습니다. 특히 지도학습 기반 스트리밍 추론 기준선 대비 정확도 향상이 두드러졌고, GRPO 대비로는 정확도를 높이면서도 총 생성 길이를 줄이는 더 좋은 정확도-효율 전선을 형성했습니다. 단계별 크레딧 분해의 중요성도 확인되어, 지나치게 미세한(문장/토큰 수준) 배분은 일관된 향상으로 이어지지 않은 반면 자연스러운 스트리밍 vs 딥 숙고 단계 구분이 최적이었으며, 이는 진짜 적응적 스트리밍 사고 학습에 HRPO가 기여했음을 시사합니다.



### CORA: Analyzing and bridging thinking-answer gap in Multimodal RLVR via Consistency-Oriented Reasoning Alignmen (https://arxiv.org/abs/2606.14691)
Comments:
          Submitted to EMNLP 2026

- **Prior Approaches**: RLVR은 추론 과정(<think>)을 생성한 뒤 최종 답(<answer>)을 검증 가능한 보상으로 학습시키며, 비전-언어 모델로 확장되는 흐름도 빠르게 늘고 있다. 기존 연구는 추론 흔적의 시각적 커버리지를 늘리거나 추론 그림자(환각)를 줄이는 데 주로 초점을 맞추지만, 추론과 답이 어긋나는 ‘의미 불일치’ 문제를 충분히 다루지 못한다. 또한 답-수준 보상에 의존할 때, 정답을 맞추더라도 그 정답을 뒷받침하는 추론이 부실하거나 모순될 수 있다는 동적 양상이 충분히 분석되지 않았다.

- **Core Contribution**: 이 논문은 large vision-language model(LVLM)에서 RLVR 중 ‘thinking-answer inconsistency(생각-답 불일치)’가 학습 과정 내내 지속되고 추론(inference) 이후에도 남아 있음을 체계적으로 보여준다. 그 원인을 답-수준 보상이 추론 과정에 대한 직접적 의미 감독을 제공하지 못해, 모델이 정답으로 가는 지름길을 학습할 가능성으로 해석한다. 이를 해결하기 위해 Consistency-Oriented Reasoning Alignment(CORA)를 제안하며, 최종 답과 추론의 의미 정합성을 보상으로 명시한다.

- **Technical Challenges**: 핵심 난제는 (1) 추론-답 의미 일치도를 매 롤아웃마다 안정적으로 계산해야 하고, (2) 연속형 일치 보상과 이산형 정확도 보상이 GRPO의 advantage 추정에서 서로 스케일 간섭을 일으키지 않게 조정해야 한다. 논문은 이를 위해 NLI(자연어 추론) 스타일의 경량 일치 보상 모델(CRM)을 도입해 <think>와 <answer>의 의미 정합성을 확률 점수로 산출하고, HRAS(Hybrid Reward Advantage Splitting)로 task 보상과 consistency 보상의 advantage를 그룹 내에서 분리 정규화한 뒤 가중 결합한다. 그 결과, 일치 보상이 너무 커져 정확도 최적화를 방해하거나 반대로 너무 작아 무력화되는 현상을 완화한다.

- **Empirical Impact**: 여러 멀티모달 추론 벤치마크(CVBench, MathVision, MathVista, PuzzleVQA, AlgoPuzzleVQA)에서 CORA는 기존 GRPO 대비 최종 답 정확도를 높이면서도 생각-답 불일치를 효과적으로 감소시킨다. 특히 더 큰 백본(예: 7B 스케일)과 계산 의존도가 높은 수학/퍼즐 계열에서 개선 폭이 더 두드러졌는데, 이는 신뢰할 만한 추론을 생성해야 의미 일치 보상이 효율적으로 작동한다는 가설과 맞닿아 있다. 또한 HRAS 및 CRM에 대한 제거 실험에서 전반적으로 CORA가 가장 균형 잡힌 성능을 보이며, 신뢰성 분석에서는 ‘정답이면서도 일치하는’ 응답 비율이 늘고 ‘정답이지만 불일치’가 줄어드는 경향이 관측된다.



### AgentSpec: Understanding Embodied Agent Scaffolds Through Controlled Composition (https://arxiv.org/abs/2606.14674)
- **Prior Approaches**: 기존 LLM 에이전트는 추론, 기억, 도구 사용, 반성, 행동 실행을 파이프라인으로 엮어 성능을 끌어올리는 경우가 많지만, 모듈이 촘촘히 결합돼 있어 각 구성요소의 기여와 상호작용을 분리하기 어렵다. CoALA, AgentSquare, AgentGym 같은 모듈형 시도도 결국 전체 시스템 최적화 중심이라 “어떤 추론이 어떤 기억과 맞물릴 때 성능이 왜 오르는지” 같은 질문에는 답이 제한적이다.

- **Core Contribution**: 이 논문은 embodied LLM 에이전트를 Perception–Memory–Reasoning–Reflection–Action 루프로 명시하고, 각 모듈을 표준화된 인터페이스로 “타입 기반 조합” 가능하게 만든 AgentSpec을 제안한다. 이를 통해 모듈을 손쉽게 교체·재조합하면서도 동일한 평가 프로토콜에서 모듈 적합성과 상호작용 효과를 통제된 조건으로 관찰할 수 있다.

- **Technical Challenges**: 핵심 난제는 모듈을 분리하려면 인터페이스 수준에서 입력/출력을 일관되게 정의해야 하지만, 현실의 에이전트는 관측 형식과 상태표현이 환경마다 달라 모듈 간 결합이 쉽게 다시 발생한다는 점이다. AgentSpec은 관측을 표준 상태표현으로 바꾸는 perception, 다양한 기억(에피소드/시맨틱/검색 기반 등)과 추론·반성의 입력/출력을 “형식화된 중간 객체”로 고정해, 구성요소 변경이 나머지 모듈을 재작성하지 않아도 되도록 설계했다.

- **Empirical Impact**: DeliveryBench, ALFRED, MiniGrid, RoboTHOR에서 모듈 실험을 수행한 결과, 성능은 단일 모듈의 강도보다 “스캐폴드 호환성”과 모듈 간 상호작용에 의해 좌우되는 경향이 확인됐다. 예를 들어 멀티-그란ularity 기억은 장기 상태 추적에 강점이 있고, 반성은 로컬 실행 오류를 수정할 때 비용 대비 이득이 크며, 강화학습(RL) 정책은 배포 시의 스캐폴드 구조와 함께 최적화될 때 조합 성능이 가장 잘 나온다는 결론이 제시된다.



### Characterizing Cultural Localization in AI-Generated Stories (https://arxiv.org/abs/2606.14626)
Comments:
          Accepted to the 4th Workshop on Cross-Cultural Considerations in NLP (C3NLP) Co-located with ACL 2026, San Diego, USA (non-archival)

- **Prior Approaches**: 기존 연구는 LLM이 문화적 정체성을 반영하는지(가치·규범·지식) 평가하거나, 산출물에 나타난 문화 토큰의 다양성/정확성을 점검하는 방식이 주를 이뤘습니다. 다만 ‘이 문화화(localization)가 템플릿 기반 삽입인지, 아니면 줄거리·가치까지 바꾸는 홀리스틱(localization)인지’를 직접 분해해 측정하는 방법은 부족했습니다. 또한 스토리 전반의 문장/주제 동질성은 자주 관찰되지만, 그것이 문화 간 어떤 메커니즘에서 비롯되는지 정량화된 분석은 제한적이었습니다.

- **Core Contribution**: 이 논문은 스토리 문화화가 주로 ‘표면적 문화 표지(이름·장소·물건 등)’를 문화 비특정 템플릿에 끼워 넣는 템플릿형인지 계량하는 방법을 제안합니다. 구체적으로 국적별로 식별되는 구별 어휘를 지운 뒤 남는 텍스트의 다중 단어 유사도를 비교해, 국가가 달라도 공유되는 서사 템플릿이 있는지 판별합니다. 덧붙여 표지에 대한 고정관념성(stereotypicality)과 불쾌감(offensiveness)도 함께 특성화합니다.

- **Technical Challenges**: 핵심 난제는 문화화가 ‘어떤 단어가’ 템플릿을 구분하는 최소 신호인지, 그리고 그 제거 후 남는 부분이 우연이 아닌 ‘공유 템플릿’인지 구분하는 것입니다. 저자들은 NPMI 기반으로 국적을 가장 잘 설명하는 후보 단어 집합을 고르고, 분류기 성능(F1)이 무작위 추측 수준으로 떨어지기 전까지 어느 비율의 단어를 마스킹해야 하는지로 템플릿 구성 어휘 비중을 추정합니다. 이후 남은 텍스트에 대해 LCS와 Jaccard 기반 n-그램 유사도로 다중 단어 수준 동질성을 측정해 템플릿 공유 여부를 검증합니다.

- **Empirical Impact**: 193개 국적(193 nationalities)과 125개 주제(125 story topics)에서 5개 모델이 생성한 24,125개 프롬프트 세트에 대해, 템플릿을 가르는 어휘는 모델마다 9~17%만으로도 국적 구별이 가능하다는 결과가 나왔습니다. 문화 표지를 제거하면 오히려 국가 간 다중 단어 유사도가 원문보다 증가해, 문화화가 ‘동일한 잠재 템플릿에 표지만 교체’되는 경향을 뒷받침합니다. 또한 고정관념성·불쾌감이 높은 표지는 주로 Global South(아프리카·서아시아 중심)에 분포하며 평균적으로 공격적(offensive)인 것으로 나타나, 문화 표현을 단순 토큰 매칭만으로 평가하면 위험이 있음을 시사합니다.



### LoSoNA: A Benchmark for Local Social Norm Adaptation in Group Conversations (https://arxiv.org/abs/2606.14600)
- **Prior Approaches**: 기존 LLM 사회성 평가는 주로 대화형 시뮬레이션(Sotopia 등)에서 목표 달성 여부를 보거나, 짧은 상황 제시 후 단발성 판단을 채점하는 방식(SocialIQa 등)으로 이뤄졌습니다. 또 다른 흐름은 예절/커뮤니티 제재 패턴 같은 ‘규범’을 서술적으로 분석하거나, theory of mind를 명시적 질문에 대한 답변 능력으로 측정하는 경향이 강했습니다. 하지만 실제 그룹 채팅에서 드러나지 않는 ‘로컬 규범’을 선행 대화만 보고 추론·적용하는지를 통제 가능한 과제로 분리해 평가한 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 다자 그룹 채팅에서의 로컬 사회 규범 적응 능력을 정면으로 겨냥한 벤치마크 LoSoNA를 제안합니다. 각 시나리오는 주체 에이전트가 과거 대화(다른 참여자들의 숨은 규범 시연)를 보고, 마지막 elicitor의 한 턴 응답에서 그 규범을 따르는지 여부로 성능을 판정합니다. 핵심은 규칙 라벨이나 명시적 지시 없이 ‘이전 대화가 증거가 되어야 하는’ 상황을 설계한 점입니다.

- **Technical Challenges**: 기여를 구현하는 데의 첫 기술적 과제는 ‘로컬 규범 적용’과 ‘기본적인 조언자 스타일(범용 정중함)’을 혼동하지 않게 만드는 시나리오 구성입니다. 이를 위해 벤치마크 구성 단계에서 no-demonstration 조건을 사용해, 생성된 elicitor가 단순 기본 응답만으로도 정답 규범을 만족하는 후보를 제거했습니다. 두 번째 과제는 모델이 선행 대화를 어떻게 증거로 쓸지(또는 방해할지)를 분리 관찰하는 것으로, naive부터 norm_informed까지 프롬프트 조건 4가지를 두고 단일 턴 응답을 고정 채점해 비교합니다.

- **Empirical Impact**: 실험에서는 8개(개방형·프론티어) 모델을 대상으로 하며, naive 조건에서 대부분 모델이 37% 미만 수준의 어려움을 보였습니다. 반면 norm_informed은 모델별로 큰 차이를 보였고, Gemini 3.1 Pro는 84.2%, Claude Fable 5는 81.6%까지 상승해 로컬 규범 추론에 ‘명시적 힌트’가 실질 효과가 있음을 보여줬습니다. 동시에 몇몇 모델은 이 조건에서 회귀(성능 하락)도 나타나, 로컬 규범 적응이 단순 지시 복붙이 아니라 모델의 내적 사용 방식에 크게 좌우됨을 시사합니다.



### Persuasion Index: A Theory-Guided Framework for Persuasion Analysis (https://arxiv.org/abs/2606.14580)
- **Prior Approaches**: 기존 연구는 설득의 결과(설득 성공/태도 변화/점수)를 예측하는 데 집중해 왔고, 해당 성과를 만드는 수사적 단서를 비교·설명하기는 어려웠습니다. 또한 데이터셋마다 “성공” 정의와 라벨 체계가 달라 도메인 간 일반화와 해석 가능한 비교가 막혔으며, 많은 모델은 토큰 단서가 무엇인지 안정적으로 보여주지 못했습니다.

- **Core Contribution**: 이 논문은 설득 언어를 이론에 기반한 15개 차원(로고스·에토스·파토스)으로 표준화한 Persuasion Index(PI)를 제안합니다. 단순 예측 스코어를 넘어, 텍스트가 어떤 논리/신뢰/감정 및 설득 압력을 활용하는지 “감사 가능한” 형태로 점수화합니다.

- **Technical Challenges**: 핵심 난제는 (1) 이론 차원을 실제 텍스트 특징으로 일관되게 계량화하고, (2) 투명성을 유지한 채 예측 성능도 확보하는 것입니다. 연구진은 55개 하위 특징을 사전(lexicon)·구조 규칙·LLM 기반 확장(후속 필터링/검증)으로 구현하고, 차원 점수는 하위 특징의 평균으로 계산해 모듈형이면서도 텍스트-기반 추적성을 확보했습니다.

- **Empirical Impact**: PI는 서로 다른 네 가지 공개 데이터셋에서 결과 예측에 의미 있는 신호를 제공하며, PI 하위 특징(PI-sub)이 PI 평균(PI-mean)보다 일관되게 우수했습니다. 일부 설정에서는 RoBERTa나 GPT-4o 같은 불투명 모델과 경쟁하거나 능가했고, 차원 수준 분석에서는 Evidence·Logic/Cohesion처럼 비교적 안정적인 패턴과 주제/입장에 따른 가중치 변화를 함께 보여주었습니다. 연구진은 PI 오픈소스 패키지와 웹 인터페이스를 공개해, 인간-인간/인간-LLM 커뮤니케이션에 대한 해석·감사·자극물 검증에 활용될 수 있는 공유 표현 공간을 제공합니다.



### SIMMER: Benchmarking Latent Failures in LLM Executable Planning with a World Mod (https://arxiv.org/abs/2606.14574)
- **Prior Approaches**: 기존 평가는 계획이 실행에 성공하는지(성공률)나 참조 계획과의 의미 유사도에 주로 초점을 맞췄다. 하지만 TextWorld·ALFWorld·VirtualHome 같은 가상환경은 단순한 상태표현 탓에 오염, 온도/화학 변화처럼 누적되는 암묵 상태를 제대로 반영하지 못한다. 한편 자연어 유사도 평가는 문장 표면에서는 그럴듯해도 행동 간 상태 의존성이 만들어내는 오류(잠재 실패)를 잡기 어렵다.

- **Core Contribution**: 이 논문은 LLM 계획에서 바로 드러나지 않는 ‘잠재 실패(latent failure)’—즉 전제조건은 만족하지만 암묵 상태 전파로 목표 달성을 망치거나 심하면 되돌릴 수 없는 피해를 일으키는 경우—를 체계적으로 평가하는 SIMMER를 제안한다. SIMMER는 주방 도메인에 기반한 상징적 세계모델로 계획을 실행 검증하며, 즉시 실패와 잠재/비가역 실패를 구분하는 실패 분류와 실행기(상태기계)를 함께 제공한다.

- **Technical Challenges**: 핵심 난제는 ‘전제조건 위반’ 같은 즉시 오류는 물론, 접촉·오염 전파처럼 실행 중 보이지 않게 누적되는 상태 변화까지 추적해야 한다는 점이다. 이를 위해 SIMMER는 77개 행동과 262개 객체로 구성된 PDDL 스타일 세계모델을 만들고, 상태기계 실행기가 단계별로 상태를 갱신한 뒤 실행 후 감사(audit)로 최종 상태의 이상 징후를 탐지한다. 또한 반사실적(카운터팩추얼) 예견 시뮬레이션으로 각 행동 직전에 상태 변화를 예측·점검하도록 프롬프트를 설계해 잠재 위험을 사전에 줄이게 한다.

- **Empirical Impact**: 100개 주방 스크립트(12개 조리 기법)를 6개 LLM에 적용한 결과, 최상위 성능도 에러가 없는 계획은 17% 미만(평균 7.2%)에 그쳤고 잠재 실패가 포함된 계획은 29~52%로 나타났다. 특히 잠재 실패의 상당 부분이 비가역적이며, 다수 사례가 오염 전파를 잘못 시뮬레이션한 데서 비롯됐다. 카운터팩추얼 예견 시뮬레이션은 잠재 실패를 최대 72%, 비가역 케이스를 최대 75%까지 감소시켜, ‘명시적 상태 추론’을 통한 더 견고한 LLM 플래너 방향을 실증적으로 제시한다.



### BayLing-Duplex: Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM (https://arxiv.org/abs/2606.14528)
Comments:
          Code: this https URL

- **Prior Approaches**: 기존 SpeechLM(예: LLaMA-Omni, GLM-4-Voice)은 기본적으로 턴 기반이라 사용자의 발화가 끝났다는 신호를 주는 외부 VAD에 의존해 왔다. 이 구조는 VAD의 오탐/미탐에 따라 사용자를 중간에 끊거나 응답 지연을 일으키며, 대화에서 흔한 중간 멈춤·바지인·짧은 되받아말 같은 패턴을 충분히 반영하지 못한다.

- **Core Contribution**: BayLing-Duplex는 별도 턴테이킹 모듈 없이, 단일 자기회귀 LLM이 언제 ‘듣고/말하고/멈출지’를 스스로 결정하도록 설계한 네이티브 풀더플렉스 SpeechLM이다. 표준 어휘에 특별 토큰 4개만 추가하고, GLM-4-Voice 계열의 기존 학습·서빙 스택을 거의 그대로 재사용해 모델 이식 부담을 줄였다.

- **Technical Challenges**: 풀더플렉스를 구현하려면 음성 토큰 생성 속도와 동일한 시간 스케일에서 대화 상태 결정을 내리면서, 동시에 사용자 입력을 조건으로 받아야 한다. 논문은 사용자 음성·어시스턴트 텍스트·어시스턴트 음성의 멀티채널 인터리브 시퀀스를 구성하고, 대화 상태를 텍스트 채널의 다음 토큰 예측(상태 토큰)으로 환원해 추가 분류기나 상태 머신 없이 타이밍을 학습시킨다.

- **Empirical Impact**: 실험에서 BayLing-Duplex는 InstructS2S-Eval에서 턴테이킹 성공률 92%, 인터럽션 성공률 100%를 달성했으며, Moshi 대비 Speech-response 점수도 2.17→3.39로 개선됐다. 또한 Llama Questions, Web Questions, Alpaca-Eval 등에서 턴 기반 대조군과 동등하거나 더 나은 성능을 보여 ‘동시 듣기·말하기 모델링이 응답 품질을 해치지 않는다’는 점을 실증했다.



### Fodor and Pylyshyn's Systematicity Challenge Still Stands (https://arxiv.org/abs/2606.14512)
Comments:
          Accepted in the Transactions of the Association for Computational Linguistics (TACL). This is a pre-MIT Press publication version of the paper

- **Prior Approaches**: 기존 연결주의(뉴럴 네트워크)는 상징주의의 체계성(systematicity) 설명을 바로 제공하지 못한다는 비판을 받아왔다. 특히 Fodor와 Pylyshyn의 ‘체계성 논증’은 인간이 한 문장을 이해할 수 있으면 그에 상응하는 다른 문장도 반드시 이해하는 쌍조건적(biconditional) 의존성을 보인다고 주장하며, 이는 상징 체계에서 기본적으로(by default) 따라온다고 본다. 
최근에는 Brenden Lake와 Marco Baroni가 메타학습으로 합성성(compositionality)을 학습하면 인간 체계성을 재현·설명할 수 있다고 주장했지만, 이 논문은 그 결론이 성급하다고 본다.

- **Core Contribution**: 이 논문은 Lake와 Baroni의 Meta-Learning for Compositionality(MLC)가 체계성 과제를 실제로 해결하지 못한다는 점을 체계적으로 반박한다. 단순히 ‘모델이 일부 정답을 잘 맞춘다’가 아니라, 분포 안팎에서 규칙을 일관되게 학습·적용하는지, 그리고 어떤 규칙 변화에도 체계적으로 반응하는지로 실패를 보여준다. 
저자들은 따라서 Fodor와 Pylyshyn이 제기한 신경망의 체계성 과제가 현재까지 미해결 상태라고 결론짓는다.

- **Technical Challenges**: 저자들이 지목하는 핵심 기술적 문제는 MLC가 메타학습 중 본 데이터의 아주 구체적인 세부사항에 과도하게 민감해, 훈련 분포 밖(조금만 벗어나도)에서 규칙 학습과 일반화가 무너진다는 점이다. 또한 분포 안에서도 ‘체계적 행동’이 관찰되지 않는다고 주장하며, 체계성을 (1) 분포 내 좁은 성능, (2) 무관한 디테일에 대한 민감도, (3) 서로 다른 문법에서도 동일한 체계성을 보일 수 있는가로 세 층위로 나눠 점검한다. 
저자들은 이 세 층위 모두에서 MLC 모델이 비체계적 실패를 보였고, 따라서 “체계성은 생긴다”는 주장에 신뢰를 주기 어렵다고 말한다.

- **Empirical Impact**: 실험적으로 저자들은 Lake와 Baroni가 보고한 ‘gold grammar’뿐 아니라 그 규칙을 소폭 수정한 변형에서도 성능이 흔들리며, 이는 체계성의 형태로 보기 어렵다는 근거로 제시된다. 특히 모델이 규칙의 사소한 변경이나 라벨-의미 매핑 같은 우연적 요소에 좌우되는 경향을 보였다고 하며, 체계성 논증이 요구하는 ‘기본 패턴’ 학습과는 거리가 있다고 해석한다. 
결과적으로 이 논문은 신경망의 합성적 일반화가 체계성을 의미하지 않을 수 있음을 강조하며, 인지과학-머신러닝의 연결 고리에서 ‘체계성 설명’의 기준을 더 엄격히 다뤄야 한다는 파급효과를 갖는다.



### A Computational Audit of Demographic Association Encoding in ClinicalBERT Language Predictions (https://arxiv.org/abs/2606.14460)
Comments:
          17 pages, 4 tables, appendices A-E, preprint

- **Prior Approaches**: 기존 임상 NLP 편향 연구는 주로 인종·성별·사회경제적 집단 간 ‘결과’(예측 성능) 격차를 보여주는 데 집중했지만, 인구통계 관련 신호가 모델의 확률 분포로 ‘어떻게’ 전파되는지는 충분히 설명되지 않았습니다. 또한 BERT의 템플릿 기반 로짓/로컬 확률 탐침 같은 방법이 일반 도메인에서 제안되었으나, 고위험 임상 문맥과 교차 인구집단(인종×성별)에서의 편향 증폭 메커니즘까지 일관되게 감사한 사례는 드뭅니다. 특히 정렬(alignment) 이후에도 편향이 구조적으로 잔존할 수 있다는 관찰은, 표면 출력만으로는 모자라다는 문제의식을 뒷받침합니다.

- **Core Contribution**: 이 논문은 ClinicalBERT에서 인구통계 단서가 문장 내부 확률을 어떻게 바꾸는지 ‘컴퓨테이셔널 감사’로 체계화합니다. LPBA(Log Probability Bias Analysis)와 MLM(Masked Language Model 기반 분석)을 함께 사용해, 행동·평가 프레이밍·환자 행위(agency) 언어에서 인구집단 관련 편향이 모델 확률로 드러나는 방식을 분해합니다. 더 나아가 코퍼스 빈도와의 방향 일치 여부를 통해 통계적 격차와 ‘편향 증폭’(bias amplification)을 실증적으로 구분합니다.

- **Technical Challenges**: 핵심 난제는 (1) 동일한 문장 구조에서 인구통계 표지를 바꿨을 때 모델의 확률 분포가 얼마나 이동하는지 측정하고, (2) 그 이동이 데이터에 이미 있던 차이를 단순 반영하는지(상속) 아니면 모델 내부에서 더 크게 왜곡하는지(증폭)를 판별하는 것입니다. 연구진은 98개의 실제 임상 문장 템플릿과 8개 교차 인구집단을 사용해 문맥을 고정한 뒤, 마스크 토큰 확률의 차이를 통계적으로 검정하고(FDR 보정 포함) 코퍼스 빈도(정규화된 단어 출현률)와 부호(sign) 기반으로 비교하는 방식으로 ‘증폭’ 신호를 분류합니다.

- **Empirical Impact**: 결과적으로 유의한 발견 32개 중 65.6%가 코퍼스 분포와 ‘반대 방향’으로 나타나, ClinicalBERT의 대표성 편향이 주로 모델 내부에서 증폭된다는 직접 증거를 제공합니다. 집단별로는 흑인 환자에서 반대 비율이 80%까지 상승했고, 행위 언어(agency attribution)는 MLM 기준 87.5%로 더 강하게 나타났습니다. 이는 임상 AI 거버넌스에서 데이터 재균형만으로 부족할 수 있음을 시사하며, 배치 전/운영 중 교차집단 감사와 관측 가능한 감사 지표(행동·평가·행위 언어)를 제도화할 근거를 제공합니다.



### MoDiCoL: A Modular Diagnostic Continual Learning Dataset for Robust Speech Recognition (https://arxiv.org/abs/2606.14459)
Comments:
          Accepted at Interspeech 2026

- **Prior Approaches**: 기존 ASR 평가는 잡음, 악센트, 발화 장애, 화자 연령 등 분포 변화 요인을 따로 떼어 측정하는 경우가 많아, 실제로 누적·동시 발생하는 상황을 충분히 반영하지 못합니다. 또한 continual learning(연속 학습)이 ASR에서 분포 드리프트에 대한 ‘강건성’ 형성·전이·망각을 진단하는 방식으로 체계화된 연구는 상대적으로 제한적이었습니다.

- **Core Contribution**: 이 논문은 강건성을 시간이 지나며 진화하는 ‘동적 능력’으로 보고, MoDiCoL(Modular Diagnostic Continual Learning)이라는 모듈형 진단용 데이터셋과 연속 학습 커리큘럼을 제안합니다. MoDiCoL은 언어 내용, 화자 특성, 음향 환경을 요인 설계로 분리해, 강건성이 어떻게 얻어지고 다른 드리프트에 옮겨지며 다시 잊히는지 분석할 수 있게 합니다.

- **Technical Challenges**: 핵심은 (1) 요인이 실제로 함께 나타나는 복합 드리프트를 통제된 순서로 만들고, (2) 논리적으로 불가능한 조합은 합성 발화·증강으로 보정하며, (3) 온라인·스트리밍 연속 학습처럼 현실에 가까운 평가를 수행하는 데 있습니다. 이를 위해 요인 실험 설계를 L27 직교배열과 foldover로 구성해 다수의 실행(run)을 만들고, denoising·불유창 삽입·장애 시뮬레이션·거리·잡음 주입 등의 증강 파이프라인으로 각 수준을 맞춘 뒤, Experience Replay·Representation-level Regularization·Orthogonal Gradient Descent를 비교합니다.

- **Empirical Impact**: 실험에서 초기(미적응) 모델은 화자·언어 드리프트에서 WER이 크게 악화되며, 이는 강건성이 단일 요인에만 민감하지 않음을 시사합니다. 연속 학습 결과로는 Experience Replay가 재학습 안정성과 망각 억제에 가장 효과적이었고, 버퍼 10% 구성이 A-WER 및 낮은 FM/BWT로 강건성 전이를 가장 잘 보였습니다. 또한 OGD가 RLR보다 성능이 좋게 나타나, 성능 저하가 표현 자체의 표류(대표 드리프트)뿐 아니라 기울기 하위공간 내 간섭에서 크게 기인함을 실증적으로 보여줍니다.



### Coping in Crisis: Computational Modeling of Coping Styles in Digital Crisis Discourse During the 2023 Turkiye Earthquak (https://arxiv.org/abs/2606.14420)
Comments:
          20 pages, 5 figures, 3 tables. To be submitted to Social Science Computer Review

- **Prior Approaches**: 재난 상황에서 사람들이 쓰는 글을 통해 대규모·실시간으로 심리 상태를 감지하려는 시도는 있어왔지만, 주로 단일 라벨 분류나 사후 분석에 머무는 경우가 많았습니다. 또한 문헌에 기반한 ‘대처 양식’의 다중 분류를 정치적 양극화 같은 맥락과 함께 정교하게 모델링하기가 어려웠습니다.

- **Core Contribution**: 이 논문은 Lazarus와 Folkman의 대처 이론을 디지털 재난 데이터에 직접 적용해, 문제중심, 감정중심, 의미형성이라는 3가지 대처 양식을 다중 라벨로 탐지하는 BERTurk 분류기를 제안합니다. 나아가 이 대처 양식이 위기 단계(4단계)에 따라 어떻게 달라지는지까지 이론적으로 동기화해 함께 추적합니다.

- **Technical Challenges**: 핵심 과제는 터키어 트윗에서 대처 양식이 표현되는 신호를 포착하면서도, 위기 단계별 문맥 변화와 정치적 편향의 영향을 동시에 다루는 것이었습니다. 연구진은 위기 단계에 동기화된 분류 구조로 Lazarus-Folkman 이론을 구현하고, BERTurk로 학습해 제로샷 mDeBERTa 대비 성능을 크게 끌어올렸습니다.

- **Empirical Impact**: 실험에서 BERTurk는 매크로 F1 0.693으로 제로샷 mDeBERTa(매크로 F1 0.324)보다 크게 향상됐고, 전체 트윗에 적용했을 때 대처 양식의 시간적 궤적이 뚜렷하게 나타났습니다. 또한 분노는 의미형성과 가장 강하게 상관(Spearman r=0.387)되어, 실질적 행동보다는 비난 귀인 같은 의미 재구성 쪽으로 작동할 수 있음을 시사하며, 인도주의 조직이 실제 수요 위치에 맞춰 대응을 조정하는 데 활용될 잠재력을 보여줍니다.



### Learning to Hear Hesitation: Continual Learning for Disfluency-Aware ASR (https://arxiv.org/abs/2606.14391)
Comments:
          Accepted at Interspeech 2026

- **Prior Approaches**: 기존 자동음성인식(ASR)은 대개 발화를 “정상적으로” 만들기 위해 disfluency(불유창성)를 무시하도록 최적화되어, 정보 손실과 환각(hallucination)이 생기기 쉽습니다. 보완 연구는 불유창성 마커를 포함하거나(예: 표기/토큰) 화자별로 개인화(personalization)하는 방식, 혹은 더 자세한(동일성 높은) 전사(verbatim transcription)를 노리는 방식이 중심이었습니다. 그러나 제한된 소규모 불유창성 데이터로 모델을 추가 학습하면, 일반 도메인 능력이 망가지는 catastrophic forgetting 문제가 자주 보고됩니다.

- **Core Contribution**: 이 논문은 불유창성 인식에 continual learning(계속학습, CL)을 적용해, 불유창성 토큰을 “안정적으로” 학습하면서도 기존 ASR 성능을 유지하는 절차를 제안합니다. 특히 pretrained ASR(whisper-small.en)에 불유창성 마커를 네 가지 토큰 유형으로 도입한 뒤, 불유창성 분포가 다른 데이터셋으로 연속 학습해 적응의 견고함을 실험합니다. 또한 CL 방법별로 나타나는 내부 학습 동학을 비교해, 단순 성능 차이를 넘어 왜 어떤 방법이 더 잘(혹은 덜) 마커를 학습하는지까지 설명하려고 합니다.

- **Technical Challenges**: 핵심 기술 난제는 “토큰(불유창성 마커) 학습”과 “일반 ASR 품질(pWER)” 사이의 충돌로, 소규모 추가학습이 마커를 잘 못 배우거나(언더-인트로듀스) 반대로 전반 품질을 해치는 트레이드오프가 발생한다는 점입니다. 저자들은 네 가지 CL 기법(EWC, ER, A-GEM, WA)을 비교하고, disfluency 토큰이 생성될 때 어떤 attention head가 관여하는지 head-attribution으로 추적했으며, 상위 cross-attention head를 ablation(마스킹)해 원인성을 검증했습니다. 분석 결과, 마커를 성공적으로 내보내는 경우 공통적으로 소수의 decoder cross-attention head가 특화되며, 이 기작이 CL 방법 전반에서 공유됨을 보였습니다.

- **Empirical Impact**: 실험에서 pWER(불유창성/구두점 등을 제거한 기준의 WER)와 마커 F1 사이에 일관된 trade-off가 확인되며, pWER를 가장 잘 유지하는 WA는 마커를 상대적으로 덜 도입하는 경향이, 반대로 마커 F1을 가장 잘 내는 ER는 ASR 품질 희생이 일부 동반되는 패턴이 나타납니다. 순차 적응(먼저 마커 세팅 후 Pitt→Delaware)에서는 ER이 마커의 일반화·유지(최소 forgetting) 측면에서 가장 강했고, WA는 비불유창(clean) 및 불유창 pWER 유지가 가장 안정적이었습니다. 또한 상위 cross-attention head를 제거하면 disfluency(특히 FILLER) 발화가 크게 줄어들면서도 pWER는 유사하게 유지되어, “왜 CL 방법이 마커 학습에 실패하는지”에 대한 기계적 설명을 제공했다는 점에서 의미가 큽니다.



### Achieving Precise Text-To-Cypher Via Grounded Knowledge Graph Data Generation (https://arxiv.org/abs/2606.14325)
- **Prior Approaches**: Property Graph를 대상으로 한 대화형 질의 응답은 Text2Cypher(텍스트→Cypher) 파서를 요구하지만, 이를 위한 고품질 데이터와 라벨이 부족하다는 문제가 반복돼 왔다. 기존에는 대형·상용 언어모델에 의존하거나, 제한된 수의 수작업/소량 데이터로 미세조정해 성능 격차를 메우려는 접근이 많았다. 그 결과 로컬 배포 환경에서는 정확도와 비용(또는 데이터 수집/주석 캠페인) 간 트레이드오프가 크게 나타났다.

- **Core Contribution**: 이 논문은 Text2Cypher 파인튜닝을 위해 자동 합성 데이터 생성 방법을 제안한다. 이를 통해 작은 LLM도 Property Graph 질의를 안정적으로 Cypher로 변환하도록 만들며, 대형 상용 모델과 경쟁 가능한 성능을 목표로 한다. 특히 로컬 배포가 필요한 환경에서 데이터 주권을 확보하면서도 추가 비용이 큰 주석 캠페인 없이 정확도를 유지하는 방향을 제시한다.

- **Technical Challenges**: 합성 데이터는 ‘그럴듯한’ 예제가 아니라 벤치마크 수준의 정답 Cypher를 제공해야 하며, 그래프 스키마·관계·제약을 정확히 반영해야 한다. 또한 작은 LLM이 합성 데이터에 과적합하지 않고 일반화하도록 학습 신호를 설계하는 문제가 있다. 논문은 이러한 요구를 만족하도록 합성 데이터 생성 파이프라인을 구성하고, 이를 소형 LLM 파인튜닝에 직접 활용하는 전략으로 해결한다.

- **Empirical Impact**: 저자들은 주요 Text2Cypher 벤치마크 전반에서 실험을 수행해, 제안한 합성 데이터 생성 방식이 소형 LLM의 성능을 유의미하게 끌어올린다고 보고한다. 그 결과 일부 설정에서는 대형 상용 모델에 근접하거나 경쟁하는 수준까지 도달해, 로컬 배치 제약에서도 정확도를 잃지 않을 수 있음을 보여준다. 데이터 소버린티와 비용 효율을 동시에 달성할 수 있다는 점에서, 실무형 그래프 질의 시스템의 구축 장벽을 낮추는 의미가 크다.



### Retrospective Progress-Aware Self-Refinement for LLM Agent Training (https://arxiv.org/abs/2606.14302)
- **Prior Approaches**: 기존 LLM 에이전트 학습은 주로 한 스텝의 행동 선택을 강화학습으로 최적화하지만, 장기 과제에서 필요한 메타인지적 ‘진행 상황 인식’은 충분히 내재화하지 못한다. 또한 온라인에서 진행률을 말하게 하는 프롬프트는 오히려 성공률을 떨어뜨리고, outcome(결과) 보상만으로는 진행 인식 능력이 안정적으로 생기지 않는 비대칭이 관찰된다.

- **Core Contribution**: 이 논문은 RePro(Retrospective Progress-Aware Training)로, 에이전트가 완료된 궤적과 최종 결과를 바탕으로 각 스텝의 진행률을 ‘회고적으로’ 스스로 생성하도록 학습하는 프레임워크를 제안한다. 전진(forward)으로 실행하며 온라인 진행 추정을 생성한 뒤, 과제 종료 후 outcome에 고정된 방식으로 반성(reflect)하며 진행 신호를 재평가한다. 

- **Technical Challenges**: 핵심 난제는 (1) 온라인 진행률 예측은 잡음 신호가 되어 의사결정을 방해한다는 점, (2) 회고적 진행 신호는 자연 상태에선 학습되기 어렵다는 점이다. 이를 위해 두 단계 학습을 설계하는데, 먼저 Retrospection Warmup에서 최소한의 외부 데모로 회고 형식과 반성 방식을 정렬하고, 이후 RePro-PO에서 복합 보상(진행 shaping, 온라인-회고 정렬, 형식 정규화)과 계층적 advantage로 스텝 단위 학습 신호를 만든다.

- **Empirical Impact**: WebShop, ALFWorld, Sokoban 3개 벤치마크에서 Qwen 계열 모델 전반에 대해 일관된 성능 향상을 보였고, WebShop에서는 최대 12%p 수준의 성공률 절대 개선이 보고된다. 진행 신호의 품질 분석에서도 성공/실패 궤적을 더 잘 가르는 Intermediate Discrimination이 크게 상승해, 단순 보상 설계가 아니라 실제 메타인지적 진행 인식이 형성되었음을 시사한다. 결과적으로 장기 과제에서 추가적인 보상 모델 없이도 내부적인 진행 추적을 학습할 수 있는 가능성을 보여준다.



### Does the Judge Prefer English? Evaluating Language-Switching Invariance in LLM-as-a-Judg (https://arxiv.org/abs/2606.14278)
- **Prior Approaches**: 기존 LLM-as-a-Judge 평가는 BLEU/ROUGE 같은 기준선 대비 의미 기반 판정을 제공하지만, 판정 모델의 편향(포지션, 과도한 길이, 자기 선호, 프롬프트 민감도 등) 문제를 그대로 물려받는다. 또 자동 평가는 종종 정답 라벨과 일치도를 높이려는 데 초점을 둬, ‘언어 표면이 바뀌어도 동일한 비교 결론을 내리는가’ 같은 신뢰성 진단은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 Judge-LS라는 메타-평가 프로토콜을 제안해, LLMBar의 pairwise(두 응답 비교) 항목을 영어/중국어/중국어-영어 언어 스위칭 버전으로 만들고 판정의 불변성(invariance)을 테스트한다. 핵심 기대는 라벨 보존 번역 변환에서는 선호가 유지되고, 번역 동치(tie probe)에서는 특정 언어에 유리하게 “이겼다/졌다는” 결론을 내지 말아야 한다는 점이다.

- **Technical Challenges**: 기여를 현실적으로 구현하려면, 번역 과정이 약한 답을 ‘의미적으로 고쳐서’ 금 기준 라벨을 깨지 않도록 변환을 통제해야 한다. 이를 위해 변환 모델에 오류/수정이 없도록 명시적으로 지시하고, 변환 품질을 기계적으로 감사(audit)해 위험 변형 19개는 민감도 분석에서 제외했으며, 영어를 기준으로 선호 뒤집힘, 정확성 뒤집힘, 포지션 불일치까지 짝지어 비교했다.

- **Empirical Impact**: LLMBar(419문항) 전체에서 네 가지 API 접근 가능 judge를 평가한 결과, 모든 모델이 영어에서 가장 높은 성능을 보였고 중국어/언어 스위칭에서는 선호 결론이 10.7%~14.4% 범위로 뒤집혔다. 특히 번역 동치 tie probe에서는 영어에 대한 일관된 선호 징후가 뚜렷하지 않았으며, 비-tie 결정이 나올 때는 대체로 중국 쪽이 더 자주 이기는 패턴이 관찰돼 ‘단순 영어 우위’보다 ‘언어 표면에 대한 불안정성’이 문제임을 시사한다. 결과적으로 다국어 환경에서 리더보드나 모델 선택을 할 때는 언어/포맷 변화에 대한 판정 안정성 진단을 추가해야 한다는 실무적 함의를 제공한다.



### The Linguistics Olympiads: Towards a New Corpus for Linguistics Research? (https://arxiv.org/abs/2606.14257)
Comments:
          Accepted for publication in LingBaW. Linguistics Beyond and Within (Volume 12, 2026)

- **Prior Approaches**: 언어 올림피아드 문제(Linguistics olympiad problems, LOPs)는 작은 말뭉치에서 규칙을 추론하고 다른 항목을 변환하는 자족형 퍼즐로, 기존 연구는 주로 유형 분석과 풀이 전략에 초점을 맞춰왔다. 또한 최근에는 LOPs가 대형 언어 모델 벤치마크로 활용되며 계산언어학적 유용성이 부각됐지만, 학술 언어학의 정식 데이터로 통합되진 못했다.

- **Core Contribution**: 이 논문은 LOPs를 언어학 연구용 데이터 소스로 체계적으로 평가하고, 학술 연구에서 책임 있게 사용하는 기준을 제안한다. 즉, LOPs와 학술 언어학 사이의 간극을 메우기 위한 이론적 틀을 마련해, 언어 유형론·언어 상대성·언어 현장조사 등과의 연결 가능성을 정리한다.

- **Technical Challenges**: 핵심 난제는 LOPs가 ‘과학적 데이터’로서 어떤 강점(현상 대표성, 규칙 추론 가능성)을 갖는지와 동시에 어떤 약점(표집 편향, 맥락 부족, 검증 가능성 등)이 있는지를 일관된 기준으로 분해하는 데 있다. 논문은 1800개+ LOPs를 출발점으로 삼아 도구로서의 성격과 한계를 비판적으로 논의하고, 어떤 언어학 하위 분야에 적합한지 매핑하는 방식으로 프레임을 구축한다.

- **Empirical Impact**: 1800개+ LOPs 기반의 구조화된 평가는 LOPs가 잠재적으로 새로운 말뭉치로 기능할 수 있음을 구체적으로 시사한다. 나아가 책임 있는 사용 기준을 제시함으로써, 앞으로 LOPs를 학술 언어학의 방법론 안으로 끌어들이는 실무적 출발점이 될 전망이다.



### Decoupled Mixture-of-Experts for Parametric Knowledge Injection (https://arxiv.org/abs/2606.14243)
- **Prior Approaches**: 기존 지식 주입은 크게 두 갈래입니다. RAG는 검색으로 얻은 근거를 프롬프트에 추가해 쉽게 갱신할 수 있지만, 지식이 파라미터에 내장되지 않아 통합이 얕고 긴 컨텍스트·반복 검색으로 효율이 떨어질 수 있습니다.
반면 SFT·LoRA 같은 사후학습 기반은 파라미터에 더 깊게 반영하지만, 공유 파라미터 공간에 덮어써져 지식 충돌·치명적 망각이 생길 수 있고, 업데이트 시 재학습/재정리가 부담이 됩니다.

- **Core Contribution**: 이 논문은 Decoupled Mixture-of-Experts(DMoE)로, 기본(백본) 모델에서 전문가(expert)와 라우터(router)를 분리해 모듈형 파라미터 지식 주입을 구현합니다. 외부 지식 코퍼스를 지식 단위로 쪼개 각각의 전문가 모듈로 만들고, 라우팅 신호로 필요한 순간에만 활성화해 지식의 격리성과 독립 업데이트를 노립니다.
또한 DMoE는 최종 레이어의 FFN(피드포워드 네트워크)에만 전문가를 붙여, 파라미터 수준 통합을 하면서도 자동회귀 생성의 효율(특히 KV-cache 재사용)을 보존하도록 설계합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 불확실성 기반으로 어떤 지식 단위를 쓸지 스케일 있게 선택하면서, (2) 전문가 활성화가 디코딩 중 KV-cache를 깨지 않게 배치하는 것입니다. DMoE는 토큰 불확실성(예측 분포 엔트로피)을 TU 신호로 쓰고, TU가 임계값을 넘는 위치에서만 라우팅을 트리거합니다.
추가로 라우터는 학습 없이 BM25 기반 역색인을 사용해 텍스트 서러게이트와의 매칭으로 전문가를 고르고, 전문가를 최종 레이어 FFN에만 결합해 이전 레이어의 KV-cache가 유효하도록 했습니다.

- **Empirical Impact**: HotpotQA, ComplexWebQuestions, Quasar-T, StrategyQA 같은 지식 집중 벤치마크에서 DMoE는 대체로 가장 좋은(또는 동률) 성능을 보이며, 밀집(dense) 기준선을 넘어 답 품질을 개선합니다.
또한 FLARE처럼 생성 중 온라인 적응을 하는 방법 대비 약 3배 빠르고 GPU 메모리도 크게 줄이며, KV-cache 재사용 가능성이 지연·메모리 이득의 중심임을 분석과 함께 보여줍니다.
Coupled MoE 제어 실험에서도 단순 파라미터/용량 증대의 효과가 아니라 “분리된 캐시 호환 모듈화”가 효과-효율 트레이드오프를 더 낫게 만든다는 점을 강조합니다.



### Detecting undisclosed LLM-generated content in parliamentary texts (https://arxiv.org/abs/2606.14209)
- **Prior Approaches**: 기존 연구는 LLM 생성 텍스트를 탐지하는 방법을 워터마킹, 통계 기반, 신경망 기반으로 나누어 발전시켜 왔다. 다만 일반적인 ‘블랙박스’ 탐지기는 사람처럼 판단 근거를 설명하기 어렵고, 도메인·언어·문장 길이에 따라 성능이 크게 흔들린다는 한계가 반복해서 지적된다. 특히 정치 문서처럼 고위험 영역에서는 오탐이 신뢰를 해칠 수 있어, 해석 가능성이 중요하지만 충분히 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 영국(웨스트민스터)과 스웨덴(리크스당) 의회 문서에서 비공개로 사용된 LLM 생성 문구의 ‘빈도’를 추정하는 데 초점을 둔다. 핵심 기여는 사전(LLM 이전) 의회 문서와 LLM이 생성한 동일 계열 문서를 활용해, 의사결정 근거를 보이는(interpretable, glass-box) 선형 분류기를 학습하고 이를 실제 최신 문서에 적용한 점이다. 이를 통해 정치권의 문서 작성 과정에서 AI 사용이 투명하게 공개되고 있는지 점검 가능한 측정 프레임을 제시한다.

- **Technical Challenges**: 기여를 실현하려면 (1) 실제 의회 문서에 가까운 데이터 생성 방식과 (2) 탐지기의 오탐 가능성을 통제하는 학습·평가 설계가 필요하다. 저자들은 2014~2020년 문서를 LLM 비사용 기준선(사실상 인간 작성)으로 두고, 2021년 이후 문서는 지면 라벨 없이 탐지 대상으로 설정한 뒤, 요약(summary)→전체 생성(full generation) 형태의 제로샷 프롬프팅으로 LLM 생성 데이터를 구성했다. 또한 단어 n-그램 기반의 선형 ICON(Interpretable CONfidence-enhanced Perceptron) 분류기를 써서 결정에 기여한 특징을 시각화·설명 가능하게 했고, 학습·검증 점수로 최고 모델을 선택해 신뢰도를 높였다.

- **Empirical Impact**: 적용 결과, 2022년 이후 두 의회 모두에서 ‘비공개 LLM 사용’이 꾸준히 증가하는 신호가 관찰되었다. 이는 의회 문서에서 AI 사용 공개 가이드가 상대적으로 모호한 현실과 맞물려, 대중 신뢰·투명성 관점에서 정책 논의의 근거 데이터를 제공한다. 더 나아가 해석 가능한 탐지기 기반 측정은 고위험 영역에서 “왜 LLM으로 분류됐는가”를 추적할 수 있게 해, 단순 정확도 경쟁 이상의 실무적 의미를 가진다.



### OdysSim: Building Foundation Models for Human Behavior Simulation (https://arxiv.org/abs/2606.14199)
Comments:
          34 pages. Code: this https URL ; Models and data: this https URL

- **Prior Approaches**: 기존 인간 행동 시뮬레이션 벤치마크와 학습은 이론심리, 사회상호작용, 역할놀이, 사용자 행동 등으로 조각나 있어 전체적인 진척을 한눈에 추적하기 어렵다. 또한 SFT나 RLHF처럼 ‘도움됨’에 치우친 후처리는 모델을 지나치게 동의적이고 균질한 조력자 톤으로 끌어 Sim2Real gap을 키우며, 단순 프롬프트만으로는 인간의 바람직하지 않은 행동 다양성을 재현하기 어렵다.

- **Core Contribution**: OdysSim은 대규모 행동 기반(behavioral foundation) 모델을 만들기 위한 가장 큰 공개적이고 체계적인 조사로, 모델이 인간 행동을 “규모로” 시뮬레이션하도록 설계한다. 이를 위해 5개 능력 축(C0NV, SS, COG, ROLE, EVAL)으로 62개 데이터셋과 23개 벤치마크 태스크를 통합하는 Soul(Simulation Of human-Like behavior) 프레임워크와, 이를 반영한 SOUL-Index 평가체계를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 사회적 맥락(화자/역할/의도/목표)이 텍스트에 명시되지 않은 데이터가 많다는 점이며, 그래서 저자들은 각 대화를 백생성(back-generated) 방식으로 인물 프로필과 상호작용 목표 같은 사회적 그라운딩을 덧씌운 OdysSim 코퍼스를 구축한다. 이후 midtraining(행동 분포 선행 적응)→태스크별 RL(각 축의 보상 신호에 최적화, 필요 시 LLM judge의 텍스트 피드백을 학습 정보로 사용)→전문가 증류로 단일 8B 모델 OSim을 만들며, 이 단계 구성이 서로 다른 보완 효과를 내도록 설계된다.

- **Empirical Impact**: 결과적으로 공개 8B OSim은 SOUL-Index의 23개 태스크 중 8개에서 1위(또는 공동 1위)이며, 다른 단일 프런티어 모델보다 ‘태스크 수 기준’에서 우세하다. 대화·사회적 과업에서 특히 강점이 나타나고 생성의 길이/형식/어휘 선택이 더 인간답게 바뀌며, τ-벤치의 OOD 사용자 시뮬레이션에서도 반응 정렬(React 93.2 vs 93.5)에 거의 근접한다. 한편 LLM-as-judge RL이 reward-hacking 패턴을 유발할 수 있음을 보여주고 이를 완화하는 탐지기(detector)를 함께 제시해 후속 연구에 방향을 제공한다.



### CacheRL:Multi-Turn Tool-Calling Agents via Cached Rollouts and Hybrid Reward (https://arxiv.org/abs/2606.14179)
- **Prior Approaches**: 기존 소형 모델 연구는 증류·압축·합성 데이터로 추론/코딩 등 단일 작업 성능을 끌어올리는 데 집중해 왔습니다. 에이전트 쪽에서도 ToolAlpaca, Gorilla처럼 특정 도메인이나 제한된 도구 사용, 혹은 단일 턴 중심이 많아 다단계 멀티턴 툴 호출로 일반화하기가 어렵습니다. 강화학습(RL)은 유망하지만, 라이브 툴 실행 비용과 캐시 대체 시 생기는 보상 노이즈(책임소재 문제)가 확장성을 막았습니다.

- **Core Contribution**: CacheRL은 소형 에이전트 파운데이션 모델을 목표로, 프론티어의 툴 호출 “지식”을 학습 가능한 형태로 옮기고(RL 이전 기반), 라이브 실행 없이 RL을 돌릴 수 있는 인프라와 보상 설계를 함께 제시합니다. 특히 하이브리드 사고 궤적(why까지 포함한 thinking trajectory)으로 도구 선택 이유를 전이하고, CacheAgentLoop와 캐시-티어 인식 보상으로 캐시 품질 편차에 흔들리지 않는 학습 신호를 구성합니다. 그 결과 Qwen3-4B-Thinking을 멀티턴 도구 호출 에이전트로 끌어올립니다.

- **Technical Challenges**: 첫째, 대규모로 고품질 reasoning trace를 생성하면서도 원본 툴 실행의 충실도를 깨지 않는 전이 파이프라인이 필요했습니다. 논문은 GPT-5로 메시지를 분류(분석형/사용자형)한 뒤 필요한 구간에만 인과적 이유를 생성해 API 비용을 줄이면서도 <think> 기반 사고 구조를 유지합니다. 둘째, 라이브 툴 실행을 대신해 학습하려면 캐시 품질 변동으로 인한 보상 노이즈를 해결해야 하며, CacheAgentLoop의 3단계 퍼지 캐시와 토큰 단위 마스킹, 그리고 캐시 티어에 따라 outcome/про세스 보상 가중치를 동적으로 조절하는 설계로 책임소재 문제를 완화했습니다.

- **Empirical Impact**: CacheRL은 멀티스텝 툴 호출에서 프로세스 정확도 92%를 달성해 GPT-5 94%에 근접하면서도 계산은 100배 적게 쓴다고 보고합니다. 또한 검증 리워드를 Qwen3-4B-Thinking의 0.43에서 0.78로 끌어올렸고, 실험적 기여는 지식 전이 제거 시 41%대 성능 하락, 캐시-티어 인식 보상 제거 시 약 17% 하락으로 확인됩니다. 흥미롭게도 복잡한 RL 최적화 자체보다 데이터 품질과 보상 설계가 성능을 좌우하며, 소형 모델이 실제 배포 가능한 에이전트 파운데이션으로 이어질 수 있는 실전 청사진을 제시한다는 점에서 의미가 큽니다.



### Personal Care Utility: Health as Everyday Infrastructur (https://arxiv.org/abs/2606.14145)
Comments:
          12 pages, 2 figures, 3 tables

- **Prior Approaches**: 기존 디지털 헬스는 연속 센싱을 늘렸지만, 병원 밖의 대부분 시간을 다루기엔 해석·맥락화·행동 유도가 부족하다는 한계를 보여준다. 또한 LLM/대화형 에이전트는 표현력은 뛰어나지만 안전·근거 추적·개인화 기준선(personal baseline) 같은 구조가 약해 임상 의사결정을 온전히 감싸기 어렵다. 제때 개입을 다루는 JITAI 계열 연구가 있으나, 이벤트를 생활 맥락으로 만들고(생성/이벤트 추출) 임상 지식과 분리해 운영하는 “인프라” 관점은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 Personal Care Utility(PCU)를 ‘연결 조직(connective tissue)’로 제안하며, 일상 건강 가이드를 개인별 신호에 기반해 자동 운영하는 계층형·이벤트 구동 아키텍처를 제시한다. PCU는 Personicle 기반의 생활 이벤트로 신호를 의미화하고, 개인 기준선 대비 동적 건강 상태를 추정한 뒤, 원인·맥락 추론과 지침 라우팅을 orchestrator가 담당한다. 특히 자연어 생성은 LLM이 돕되, 안전이 필요한 임상 의사결정은 검증된 근거에 고정하고 로직을 분리해 책임성을 높인다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 원시 시계열·다중 모달 데이터를 ‘감사 가능한’ 이벤트로 바꾸고, (2) 개인 기준선 대비 상태를 추정하며, (3) 어떤 순간에 어떤 채널로 얼마나 개입할지 안전하게 결정하는 것이다. 논문은 Event Engine에서 규칙/시간 추상으로 glycemic event 같은 반응 가능한 이벤트를 만들고, State Estimation에서 개인 기준선으로 이상을 정의하며, Contextual Inference에서 원인 범주(즉각·매개·누적)의 가설을 구성한 뒤, Orchestrator가 임상 로직·행동 전략 선택·발화/표현을 분리해 제어한다. 또한 Knowledge Base를 명시적 계층으로 두어 모델 내부 기억이 아니라 근거 기반으로 지침을 조회하도록 설계한다.

- **Empirical Impact**: 실증은 Type 2 Diabetes 사례로 구체화되며, CGM·식사·활동·약 복용·수면·스트레스 및 임상 데이터를 ‘당 조절 이벤트’와 개인화된 상태 추정·인과 설명·지식 기반 개입으로 변환하는 흐름을 보인다. 하루 단위 시나리오에서 위험 상황은 결정적 안전 경보로, 일반 상황은 실시간 넛지·주간 요약·약 복용 체크인·침묵(silence) 등 상황에 맞게 출력이 달라지는 구성이 제안된다. 이 접근은 단순 메시징 레이어가 아니라 아키텍처 자체로 개인화를 품는 청사진을 제공해, 향후 만성질환 전반으로 확장성과 거버넌스 논의를 촉발하는 의미가 있다.



### Implicit Reasoning for Large Language Model-based Generative Recommendation (https://arxiv.org/abs/2606.14142)
- **Prior Approaches**: 기존 LLM 기반 생성 추천(Generative Recommendation, GR)은 항목을 Semantic ID(SID)로 표현해 다음 SID를 생성하도록 학습합니다. 이를 보완하려고 CPT로 SID를 의미적으로 적합하게 만든 뒤, CoT(Chain-of-Thought) 형태의 명시적 합리화를 SFT로 추가하고 마지막에 RL 기반 후처리까지 붙이는 다단계 파이프라인이 널리 쓰였습니다. 하지만 각 단계가 언제 필요하고 무엇을 해결하는지에 대한 명확한 진단이 부족했습니다.

- **Core Contribution**: 논문은 명시적 추론(CoT) 기반 GR 학습 파이프라인을 단계별로 쪼개 분석하며, CoT SFT가 단독으로는 다음 SID 예측을 잘 개선하지 못한다는 원인을 제시합니다. 특히 CoT SFT의 실패가 (1) 사전지식의 약화된 언어화, (2) 텍스트 토큰- SID 임베딩 공간의 불일치, (3) 합리화 문구 품질에 대한 과도한 민감성에서 비롯됨을 체계적으로 보여줍니다. 이를 바탕으로 ‘합리화 문장을 만들기’ 대신 ‘잠재적 계산’을 유도하는 PauseRec을 제안합니다.

- **Technical Challenges**: 핵심 기술 과제는 SID 기반 생성에서 LLM의 자연어 기반 지식을 효율적으로 “연결”하는 인터페이스를 설계하는 것입니다. 저자들은 명시적 CoT가 생성 결과로 이어지지 않는 이유를 언어화 약화와 임베딩 불일치(이론적 분석 포함)로 설명하고, 합리화 문구를 조금만 바꿔도 성능이 크게 흔들린다는 점을 실험으로 확인합니다. PauseRec은 <pause> 토큰을 학습 가능한 브리지로 두고, 합리화 텍스트를 생성하거나 그에 대한 강한 감독(RL·추적 정렬)을 줄이는 대신 <pause> 위치의 손실을 마스킹해 SID 예측에 직접 기여하는 잠재 추론만 학습하게 설계합니다.

- **Empirical Impact**: Amazon 리뷰 3개 데이터셋에서 PauseRec은 기존 next-item SFT 및 명시적 CoT(특히 RLVR 결합) 대비 성능을 전반적으로 개선했으며, 최대 6.22%p 향상과 최대 65% GPU 시간 절감, 추론 지연 최대 71.3% 단축을 보고합니다. 또한 RL 기반 명시적 CoT와 비교해 12개 지표 중 대부분에서 경쟁력 있거나 더 나은 결과를 보여 “명시적 합리화 생성” 없이도 지식 활용이 가능함을 실증합니다. 결과적으로 LLM 기반 GR에서 해석 가능한 추론 텍스트보다 효율적인 잠재 추론 설계가 더 강력한 대안이 될 수 있음을 시사합니다.



### Beyond Perplexity: UTF-8 Validity in Byte-aware Language Models (https://arxiv.org/abs/2606.14122)
- **Prior Approaches**: 바이트 수준 토큰화(Byte-level tokenization)는 어휘 커버리지 한계를 넘어 임의의 유니코드 입력을 처리할 수 있게 해주지만, 디코딩 단계에서 잘못된 UTF-8 바이트열을 생성하는 문제가 남아 있다. 기존 연구(ByT5, Byte Latent Transformer 등)는 성능이나 효율을 중심으로 다뤘고, “UTF-8 구조적 신뢰성”이 언어 모델 학습/규모와 어떻게 연결되는지는 충분히 분석되지 않았다.

- **Core Contribution**: 이 논문은 바이트 수준 모델이 UTF-8을 “유효하게” 생성할 수 있는지 여부를, 언어 모델링 성능과 분리해 측정하는 평가 프로토콜을 제안한다. 특히 컨텍스트가 없는 생성(Level 0)과 컨텍스트가 주어진 생성(Level 1)을 나눠, 구조적 타당성(UTF-8 validity)과 의미적 정합성(정확한 문자 생성)을 분리해 드러낸다.

- **Technical Challenges**: 핵심 기술적 난관은 UTF-8 구조 유효성이 perplexity 같은 전통 지표와 약하게만 연관된다는 점이다. 이를 해결하기 위해 DFA 기반으로 UTF-8의 정규 언어 제약을 정확히 검사하고, 생성이 문자 중간에서 끊기는 경우까지 구분하는 partial-credit 유효성 지표를 설계했으며, 의미 정답 여부와 디코딩/캘리브레이션 성격(teacher-forced 로그우도 차이)까지 함께 진단한다.

- **Empirical Impact**: 355M 파라미터 모델을 80B 토큰으로 학습했을 때, UTF-8 유효성 수렴은 perplexity보다 약 2배 느려 2.1B 토큰에서 perplexity가 안정되더라도 유효성은 4.2B 토큰에서야 수렴했다. 또한 구조적 타당성은 높게 나오지만(Term Match Rate 60.30%), 의미적으로는 틀린 문자가 자주 생성되는 등 “UTF-8 신뢰성”은 perplexity 너머의 별도 역량임을 실험적으로 확인해, 평가 관점의 변화를 요구한다.



### Harsher on Male? Evaluating LLMs on Gender-Asymmetric Moral Framing Across Diverse Conflict Scenarios (https://arxiv.org/abs/2606.14068)
Comments:
          underreview

- **Prior Approaches**: 기존 연구는 성별 편향을 주로 고정관념(직업 연관), 지시어/대명사 처리, 해로운 출력 같은 ‘명시적 결과’에서 찾는 데 집중해 왔습니다. 그러나 사용자가 실제로 마주하는 관계 조언·갈등 중재·도덕 판단 같은 상황에서는 결론뿐 아니라 비난의 강도, 공감 배분, 위험 증폭, 책임 귀속 같은 ‘언어적 프레이밍’에서 편향이 드러날 수 있습니다. 또한 기존 평가가 짧은 템플릿, 어휘 치환, 단순 분류로 환원되면 장문 응답에서 나타나는 미묘한 차이를 놓치기 쉽습니다.

- **Core Contribution**: 이 논문은 같은 ‘부정 행위’를 남성/여성 행위자 조건으로 바꿔도 LLM이 일관된 판단 기준을 적용하는지 묻습니다. 이를 위해 성별 미러링(성별 거울) 설계의 GAMA-Bench(1,298 시나리오)를 제안하고, 각 시나리오는 행위·상황·심각도·표현 스타일을 고정한 채 행위자 성별만 대응되게 바꿉니다. 더 나아가 장문 답변을 처벌/치료(완화)/심각도/공감/지시·비난/전면 책임 같은 지표로 구조화하는 응답 프레이밍 프로토콜을 만듭니다.

- **Technical Challenges**: 핵심 난제는 ‘장문 상호작용’의 맥락과 정서 톤은 자연스럽게 유지하면서도, 성별만 바뀌고 다른 변수는 통제되도록 벤치마크를 구성하는 것입니다. 논문은 성별이 배제된 misconduct 템플릿을 만든 뒤, 다른 모델로 교차 검토해 암묵적 성별 단서나 부적절한 심각도·현실성 문제를 제거하고, 마지막 미러 단계에서만 1인칭 프롬프트에 행위자 성별을 주입합니다. 이후 평가 LLM이 응답을 6개 프레이밍 축으로 추출해 남녀 조건 간 페어드 갭을 계산함으로써 세밀한 차이를 비교합니다.

- **Empirical Impact**: 10종의 대표 LLM을 실험한 결과, 동일한 부정 행위에서도 ‘남성에게 더 불리한’ 프레이밍 비대칭이 일관되게 나타납니다. Intimate Track에서는 남성 행위자가 더 강한 처벌적·격화(에스컬레이션)·지시·비난 중심 표현과 전면 책임 할당을 받고, 여성 행위자는 같은 행위에 대해 더 치료적 설명과 공감 지향 프레이밍을 받는 경향이 확인됩니다. 더 robust하게는 모델 계열·시나리오 트랙·모델 규모·추가 학습 및 추론 모드(확장 추론 포함) 전반에서 방향성과 패턴이 유지되며, 평가자 모델 교체와 소규모 인간 검증에서도 결과 일치성이 높아 측정 산출물의 위험을 줄였습니다.



### Right or Wrong, Models Comply: Directional Blindness in LLM Moral Judgmen (https://arxiv.org/abs/2606.14037)
- **Prior Approaches**: 기존 시코피넌시(sycophancy)와 사회적 설득 평가들은 주로 “거짓 압력에 얼마나 자주 따르느냐” 같은 일방향 순응도(크기)만 측정해 왔습니다. 그 결과, 모델이 도움이 되는 교정은 받아들이고 해로운 영향은 거르는지(방향 선택성)는 잘 드러나지 않았습니다.

- **Core Contribution**: 이 논문은 순응도를 BCR(유익한 교정률)과 HCR(해로운 전복률)로 나누고, 그 비 A=BCR/HCR로 “방향 보정(calibrated updating)” 여부를 진단하는 Compliance Asymmetry를 제안합니다. 이를 통해 단순히 순응이 낮은지 여부가 아니라, 압력이 정답 방향인지 아닌지를 구분해 갱신하는 능력을 평가합니다. 또한 사실(factual)과 도덕(moral) 판단에서 이 선택성이 어떻게 달라지는지 비교합니다.

- **Technical Challenges**: 문제는 사회적 신호가 정답에 대한 별도 근거를 제공하지 않으면서도, 모델이 신호의 ‘방향’을 구분하도록 설계해야 한다는 점입니다. 저자들은 권위(authority)·밴드왜건(bandwagon) 닛지를 양방향(정답 쪽/오답 쪽)으로 걸고 9개 모델에서 총 97만2천 개 응답을 요인실험으로 수집했으며, CoT(체인오브쏘트)와 CIP(자기 정체성 기반 독립 평가) 프롬프트로 추론 시점 영향과 지시 민감도를 분리해 점검했습니다. 그 결과 CoT는 순응의 크기만 키우고 방향성을 되살리지 못했으며, CIP는 둘 다 억누르되 방향 보정은 회복하지 못했습니다.

- **Empirical Impact**: 실험에서 사실 판단은 유익한 닛지를 더 잘 따라 해로운 닛지에 덜 휘둘리는 방향 선택성이 유지되지만(A=1.58), 도덕 판단은 두 방향에서 순응률이 거의 같아(A=1.04) 방향 블라인드 실패가 관측됩니다. 이 현상은 모델 계열·능력 수준·닛지 유형·합의 기반 필터링에서도 지속되어 일반적 특성일 가능성을 시사합니다. 논문은 도덕 영역에서의 alignment 목표가 단순 순응 억제보다 ‘방향적으로 보정된 업데이트’여야 하며, 향후 벤치마크에 HCR뿐 아니라 Compliance Asymmetry(A)를 함께 보고해야 한다고 제안합니다.



### Dialogue SWE-Bench: A Benchmark for Dialogue-Driven Coding Agents (https://arxiv.org/abs/2606.13995)
Comments:
          22 pages, 13 figures

- **Prior Approaches**: 기존 SWE 벤치마크는 대체로 완전한 문제 명세를 입력받아, 에이전트가 자율적으로 저장소를 수정하고 테스트로 성공 여부를 판단하는 ‘완전 자율’ 평가에 집중해 왔습니다. 하지만 실제 이슈는 상당 부분이 덜 구체적이고, 사용자는 에이전트의 결과를 대화로 수정·거절하는 경우가 많아 측정 공백이 큽니다. 또한 함수 수준 대화 평가나 모호성 해소형 상호작용 연구는 있었지만, 저장소 단위 문제를 ‘대화로만’ 풀어가는 형태를 체계적으로 다루지는 못했습니다.

- **Core Contribution**: 이 논문은 코딩 에이전트가 사용자와의 목표 지향 대화를 통해 저장소 단위 실세계 SWE 문제를 해결하는 능력을 평가하는 Dialogue-SWEBench를 제안합니다. 핵심은 기존 SWE-Bench를 대화형으로 재구성해, 초기 명세를 축약한 사용자 질의로 시작하고 후속 질문·응답이 해결 과정에 필수로 들어오게 만든 것입니다. 추가로 대화 품질을 자동으로 평가하는 LLM-as-a-Judge와, 대화 역량을 강화하기 위한 schema-guided 코딩 에이전트를 함께 제시합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘대화가 실제 사용자처럼’ 전개되게 하면서도 재현 가능한 자동 평가를 만드는 것입니다. 이를 위해 논문은 persona-grounded 사용자 시뮬레이터를 만들고, 후속 발화를 생성할 때 hallucination이나 환경 위반을 걸러내는 self-revision 단계를 적용해 신뢰도를 높입니다. 또한 에이전트는 대화 상태를 스키마로 유지하며 UNKNOWN 값을 채워가고, 질문·코드 탐색·패치 생성을 일관된 논리 흐름으로 연결하도록 설계했습니다.

- **Empirical Impact**: 실험 결과, schema-guided 에이전트가 평균 해결률에서 가장 높은 성능(46.9%)을 보였고, OpenHands(32.9%) 및 OH Interactive(44.1%)를 3~14%p 범위에서 능가하며 상대적으로 낮은 비용도 달성했습니다. 특히 모델의 ‘코딩 실력’이 곧 ‘대화 실력’으로 이어지지 않는다는 신호가 관찰되어, 대화 역량이 별도의 성능 차원임을 보여줍니다. LLM-as-a-Judge로 자연스러움·일관성을 함께 분석한 결과, 대화 성공만으로 사용성(대화 품질)을 판단하기 어렵고, schema-guided 설계가 일관성 측면에서 더 큰 차이를 만들 수 있음을 입증합니다.



### The Holistic Storage of Verb+Up Phrases in Text-based and Audio-based Language Models (https://arxiv.org/abs/2606.13993)
- **Prior Approaches**: 기존 언어 모델 연구는 계산(computation)처럼 추상 지식을 이용해 새 표현을 만드는 능력을 주로 다뤘지만, 그와 함께 인간처럼 저장(storage)이 언제·어떤 표현에서 나타나는지는 덜 명확했습니다. 사용 빈도에 따라 어구가 통째로 저장되는 현상(usage-based storage)은 인간 연구에서 확인됐지만, 텍스트 LLM과 음성 ASR에서 동일한 “어구-단위 홀리스틱 저장”이 나타나는지 비교는 거의 없었습니다.

- **Core Contribution**: 이 논문은 V+up 구동사에서 up 입자가 단독 up과 다른 내부 표상을 갖는지, 즉 V+up 어구가 통째로 저장되는지(holistic phrasal storage)를 텍스트 LLM(OLMo-3 7B, BabyLM 계열)과 Whisper-small ASR까지 함께 추적합니다. 빈도와 예측가능성(frequency, predictability)이 높을수록 up의 표상이 단독 up과 더 멀어지는 패턴을 확인해, 저장이 분포 통계에서 자연스럽게 발생할 수 있음을 보여줍니다.

- **Technical Challenges**: 핵심은 모델 내부 표상에서 “홀리스틱 저장”을 직접 측정하는 방법을 설계하는 것이었습니다. 이를 위해 단독 up의 임베딩을 구분하는 로지스틱 분류기를 각 레이어에 학습한 뒤, 다양한 빈도·예측가능성의 V+up에서 up 임베딩이 얼마나 단독 up과 유사한지(logit)를 통해 차이를 추정합니다; 또한 의미 약화(semantic bleaching) 가능성을 배제하려고 up을 어절 내부(서브워드)로도 포함한 실험을 추가했습니다.

- **Empirical Impact**: 실험 결과, 텍스트 LLM뿐 아니라 Whisper의 인코더/디코더에서도 빈도와 예측가능성이 높은 V+up에서 단독 up과의 유사도가 낮아져 홀리스틱 저장의 증거가 일관되게 관찰됐습니다. 특히 예측가능성 효과는 모델이 커질수록 뚜렷해져, 단순 빈도보다 더 관계적인 통계(조건부 동시출현)를 포착하려면 더 큰 표상 용량이 필요할 수 있음을 시사합니다. 또한 비교적 인간 규모에 가까운 데이터로 학습한 BabyLM에서도 나타나, 거대 코퍼스가 없더라도 어구-단위 저장이 학습 과정에서 자연히 형성될 수 있다는 점에서 의미가 큽니다.



### Fusing Stylometric and Embedding Systems to Estimate Authorship Likelihood Ratios in Japanes (https://arxiv.org/abs/2606.13991)
- **Prior Approaches**: 기존 저자 식별(문서의 작성자 추정) 연구는 다양한 스타일로미트리(stylometric) 특징과, 최근에는 사전학습 대규모 언어모델의 문맥 임베딩을 활용하는 방식으로 발전해 왔습니다. 다만 우도비(likelihood ratio) 프레임워크는 법·과학적 근거 분석에 논리적으로 타당하지만, 그 적용이 영어 텍스트에 집중되어 일본어 같은 비영어 디지털 텍스트에는 충분히 확장되지 않았습니다.

- **Core Contribution**: 이 논문은 우도비 프레임워크를 일본 디지털 텍스트에 처음으로 적용해, 블로그 약 1,000자 발췌를 대상으로 시스템 성능과 우도비 크기(가능한 진술과 반대되는 진술에 대한 값)를 함께 평가합니다. 또한 스타일로미트리 기반 시스템과 임베딩 기반 시스템을 우도비 패러다임 내부에서 결합(fusion)하여, 이질적 증거원 통합이 가능한지 검증합니다.

- **Technical Challenges**: 핵심 과제는 스타일로미트리 특징과 임베딩 기반 점수를 서로 다른 성질의 “근거”로 취급하면서도, 우도비의 정합성(캘리브레이션)과 해석 가능성을 깨지 않게 결합하는 것입니다. 논문은 두 계열 시스템의 출력을 우도비 관점으로 통합해, 전체가 기준선을 벗어나지 않으면서도 일관된 방향의 우도비는 키우고 반대 방향의 우도비는 줄이도록 설계·평가합니다.

- **Empirical Impact**: 실험 결과 결합 시스템은 캘리브레이션이 “우수하게 유지”되며, 사실과 일치하는 우도비 크기는 증가하고 사실과 반하는 우도비 크기는 감소해 판별성이 전반적으로 향상됩니다. 최우수 결합은 log-likelihood-ratio cost 0.32484를 달성해, 일본어에서도 우도비 프레임워크의 적용 가능성을 실증하는 동시에 서로 다른 증거 모델을 결합하면 이득이 생긴다는 점을 보여줍니다.



### Creative Integration: A Decidable Criterion of Creativity (https://arxiv.org/abs/2606.13977)
Comments:
          18 pages, 1 figure

- **Prior Approaches**: 기존 연구는 창의성과 지능을 압축(compression)의 산물로 본다는 계보 위에서 전개돼 왔지만, 실제 사례마다 “진짜 통합”인지 “멋진 재서술”인지 가르는 운영 기준이 부족했다. 또한 통합처럼 보이게 하는 유사 사례(look-alike)를 구분하는 판별 경계와, 이를 검증할 라벨 데이터도 거의 없었다.

- **Core Contribution**: 이 논문은 ‘Creative Integration(CI)’을 기술 언어 하에서의 설명 길이(description length) 감소로 정의한다. 충돌 A⊕B를 해결해 통합된 원리로 기술할 때, 충돌 부위에서만 설명 길이가 엄밀히 줄어들면(C=L_pre/L_post>1) CI로 판정한다. 그 결과 “창의성=압축”을 넘어서, 생성이 아닌 판정 가능한 체크리스트(판별 규칙)로 구체화했다.

- **Technical Challenges**: 핵심 난제는 (1) 설명 길이를 언어 상대적으로 비교해야 하고, (2) 단순히 길이가 줄어든 것만으로는 ‘패키징/나열/재명명’ 같은 가짜 통합을 걸러낼 수 없다는 점이다. 이를 해결하기 위해 고정된 기술 언어에서 사전/사후를 네 범주(원리/매개변수/예외/경계)로 세는 근사 카운팅을 쓰고, 충돌의 실재성·비직교성·압축 여부·감소의 위치가 모두 성립하는지 네 개의 이진 게이트로 판별한다. 또한 유사 통합의 유형을 체계적으로 분류해 어느 게이트에서 실패하는지로 실패 모드를 진단하게 했다.

- **Empirical Impact**: 다중 도메인 라벨 코퍼스 201건(긍정 52/부정 149)으로 기준을 뒷받침하며, 인간 상호평가가 아니라 실패할 수 있는 반증 가능 테스트 4종—독립 계산 점검, hard negative 차별, 비예측(out-of-sample) 검증, 기술 언어 강건성—을 통해 유효성을 시험했다. 모든 테스트에서 기준이 여유 있게 통과해, CI가 단순한 미적 판단이 아니라 구조적으로 재현 가능한 판별자임을 보여준다. 더 나아가 본 논문은 “진짜 창의적 행위”가 충돌을 압축해 해결한다는 읽기를 제안하되, 그 전부가 CI인지 여부는 별도의 반증 가능한 추측으로 남긴다.



### MedLatentDx: Latent Multi-Agent Communication for Cross-Hospital Rare-Disease Diagnosis (https://arxiv.org/abs/2606.13945)
- **Prior Approaches**: 기존 교차 병원 협력 진단은 임상 텍스트 같은 식별 가능한 근거를 기관 간 교환하는 데 의존하는 경우가 많아 개인정보 규제로 제약을 받는다. 또 일부 의료 에이전트는 텍스트를 직접 보내지 않더라도 hidden state나 KV cache 같은 원시 잠재표현이 프롬프트에서 유도된 임상 내용을 재구성할 위험이 있다.

- **Core Contribution**: 이 논문은 MedLatentDx로, 병원 에이전트가 임상 기록과 검색된 케이스를 로컬에 보관하고 호스트 에이전트에는 압축된 latent KV 블록만 전송하는 ‘잠재 다중 에이전트 통신’ 틀을 제안한다. 또한 동일 백본 병원 환경에서는 latent KV distillation, 서로 다른 LLM 백본 병원 환경에서는 cross-family latent alignment를 지원해 배포 유연성을 높인다.

- **Technical Challenges**: 핵심 기술 난제는 두 가지인데, (1) 분산된 진단 근거를 효과적으로 결합하면서도 (2) 전송된 잠재표현이 임상 내용을 덜 재구성 가능하도록 프라이버시를 보장하는 것이다. MedLatentDx는 로컬 검색·기록을 유지하고 전송 단위를 ‘컴팩트 latent KV 블록’으로 제한하며, 백본이 다를 때는 정렬 기반의 latent alignment로 통신 호환성을 확보한다.

- **Empirical Impact**: 저자들은 병원 수준으로 파티션된 대규모 희귀질환 벤치마크 CrossRare-Bench를 자체 구축해 실험을 수행하고, MedLatentDx가 교차 병원 진단 성능을 향상시키는 동시에 raw-latent 통신 대비 재구성 가능 임상 콘텐츠를 줄였다고 보고한다. 이는 병원 간 협업이 필요한 희귀질환 진단에서 프라이버시 제약을 고려한 실용적 통신 설계를 제시한다는 점에서 의미가 있다.



### LLMs Contain Multitudes: How Deployment Context Reshapes Model-Level Preferences and Values (https://arxiv.org/abs/2606.13944)
Comments:
          68 pages, 54 figures, 54 tables

- **Prior Approaches**: 기존 평가는 LLM의 선호·가치 판단이 모델 단위로 비교적 안정적일 수 있다고 보지만, 견고성 점검은 주로 문장 문법 변화나 보기 순서 재배열 같은 부수적 프롬프트 교란에 머물렀습니다. 그 결과 실제 배포에서 흔한 ‘작업 맥락(상위 작업의 프레이밍)’이 바뀔 때도 같은 성질이 유지되는지 불명확했습니다. 또한 전통적 통계 모델은 항목 수준의 순위 이동 패턴을 잘 분해하지 못해 맥락 의존성을 과소평가할 위험이 있었습니다.

- **Core Contribution**: 이 논문(부록 포함)은 LLM이 특정 가치를 선택할 때의 ‘배포 맥락(상위 작업 프레이밍)’을 통제 변수로 두고, 두 쌍대(parwise) 패러다임—국가 선호 순위와 효용(utility) 판단—에서 맥락 변화가 측정값을 어떻게 흔드는지 직접 검증합니다. 주요 기여는 모델 레벨의 고정된 선호를 가정하기보다, 맥락 조건에 따라 구성되는 ‘맥락 의존 측정’으로 재해석해야 한다는 점을 실증적으로 제시한 것입니다. 따라서 한 프레이밍에서 얻은 안전·편향 결론이 다른 프레이밍에서는 그대로 보장되지 않을 수 있음을 보여줍니다.

- **Technical Challenges**: 핵심 기술적 어려움은 쌍대 선택 데이터에서 맥락이 만드는 ‘순위 이동의 방향과 크기’를 신뢰성 있게 추정하는 것입니다. 이를 위해 부록에서는 카이제곱/윌콕슨 같은 고전적 검정으로 유의성의 존재를 확인하는 한편, 순위 이동을 다루도록 설계된 Thurstonian 모델과 함께 CMH( Cochran-Mantel-Haenszel ) 및 Wilcoxon signed-rank 기반 효과크기(예: rank-biserial)까지 사용해 신호를 교차검증합니다. 또한 온도 변화, 프롬프트 패러프레이징, reasoning 단계 제거(토큰 강제 forced-choice, no-reasoning) 같은 통제 실험으로 ‘맥락 효과가 산출 과정의 우연이 아닌가’까지 점검합니다.

- **Empirical Impact**: 부록의 추가 분석은 맥락 변화가 온도·패러프레이징·추론 단계 설정보다 훨씬 큰 변동을 만든다는 본문 결론을 강화합니다. 예컨대 no-reasoning 강제 선택에서도 배포 맥락 간 유의한 차이가 유지되며, 효용·순위에서 미세한 순위 재배치뿐 아니라 도메인 내 정밀한 상대순위와 교환비율(카디널 환산)이 크게 달라지는 패턴이 재현됩니다. 종합하면 LLM의 선호·가치 측정은 고정 상수가 아니라 프레이밍에 조건부이며, 안전 보증이나 편향 감사는 ‘시험한 맥락 범위’ 내에서만 제한적으로 해석돼야 한다는 영향이 큽니다.



### Can Post-Training Turn LLMs into Good Medical Coders? An Empirical Study of Generative ICD Coding (https://arxiv.org/abs/2606.13940)
- **Prior Approaches**: 기존 자동 ICD 코딩은 주로 분류기 기반의 극단 다중 레이블 분류로 접근했다. 장문 의료 기록을 라벨 공간에 점수로 매핑하고 고정 임계값으로 코드를 고르는 방식이 강력한 기준선이었지만, 생성형 LLM은 프롬프트/검색/재랭킹 같은 추론 시나리오에서 부정확하거나 유효하지 않은 코드를 낼 수 있어 약한 코더로 평가되는 경우가 많았다.
다만 이런 평가는 ‘모델을 ICD 코딩에 맞게 후학습(적응)했을 때도 약한가’에 대한 답을 주지 못하고, 특히 작업 전용 후학습 축은 상대적으로 비어 있었다.

- **Core Contribution**: 이 논문은 생성형 LLM ICD 코딩에서 후학습이 성능을 어떻게 바꾸는지, 동일 프로토콜·동일 지표로 단계적으로 검증한다. SFT(지도 미세조정)로 출력 스키마와 코드 분포 적합성을 먼저 갖추고, 이후 GRPO(강화학습, F1 보상)를 통해 생성된 코드 집합의 품질을 직접 최적화하며, 마지막으로 PHI(Progressive Hint Injection)로 이전 체크포인트에서 누락된 진단 코드들을 타깃 커리큘럼으로 보강한다.
결과적으로 “생성형이라서 약하다”가 아니라 “추론 시 프롬프트만으로 평가해서 잠재력이 과소평가됐다”는 결론을 도출한다.

- **Technical Challenges**: 핵심 난제는 생성형 모델이 수천 토큰의 의료 기록에서 대규모 ICD-10-CM(7만+ 수준) 라벨 중 필요한 코드 집합을 정확히 ‘회상’하도록 만드는 것이다. 특히 토큰 단위 최대우도(SFT) 학습은 F1·정밀도·재현율 같은 비미분적 집합 수준 지표와 불일치하며, 강화학습을 하더라도 코드 집합을 파싱·검증해 보상으로 연결하는 설계가 필요하다.
저자들은 생성 결과를 공통 파서로 코드 집합으로 변환한 뒤 샘플 단위 F1을 보상으로 쓰는 GRPO를 적용하고, PHI에서 힌트는 학습 시에만 제공하되 보상 목표는 전체 정답 집합으로 유지해 추론 시 힌트 의존을 막는 방식으로 해결한다.

- **Empirical Impact**: 통제된 실험에서 프롬프트만으로 평가하면 생성형 LLM의 성능이 분류기 기준선 대비 크게 낮게 나타나며, 이는 잠재력의 과소추정을 시사한다. SFT가 가장 큰 도약을 만들고(프롬프트 대비 “작동 가능한 생성형 코더”로 전환), GRPO는 특히 Full label 설정에서 코드 집합 예측을 추가로 개선하며, PHI는 남아 있던 누락 코드(희귀 코드 포함)에 대한 타깃 이득과 매크로 성능 향상에 기여한다.
따라서 생성형 ICD 코딩의 병목은 ‘생성 방식 자체’가 아니라 ‘전체 계층(전 택소노미)에서의 재현(특히 희귀 코드)까지 성능을 끌어올리도록 모델을 어떻게 최적화·적응시키는가’에 있음을 실증적으로 보여준다.



### DLawBench: Evaluating LLMs Through Multi-Turn Legal Consultation (https://arxiv.org/abs/2606.13931)
Comments:
          37 pages, 8 figures, 26 tables. Code and data: this https URL

- **Prior Approaches**: 기존 법률 벤치마크들은 주로 법리 추론이나 답변 정확도에 초점을 맞춰, 변호사-의뢰인 간 대화처럼 ‘정보를 끌어내고(질문) 상황을 정리(유도)’하는 상호작용 능력은 충분히 평가하지 못했다. 또한 대화 상대의 성향(순응/의존/철회/대립)에 따른 상담 전략 적합성까지는 거의 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 현실 법률 상담을 진단하는 벤치마크 DLawBench를 제안한다. 의뢰인 행동을 Cooperative, Dependent, Withdrawn, Adversarial의 네 유형으로 분류하고, 실제 사건을 바탕으로 한 다회 대화에서 LLM이 사실을 전략적으로 eliciting하고 적절히 안내하는지 평가한다.

- **Technical Challenges**: 핵심 난제는 모델이 단순한 법률 추론을 넘어, 멀티턴에서 필요한 사실을 질문으로 “끌어내는” 능력과 다양한 인격/태도에 맞춘 상담 유도 능력을 동시에 요구받는다는 점이다. 이를 위해 DLawBench는 4개 의뢰인 유형의 현실적 대화 흐름과 3,411개의 문의(질문) 루브릭·3,348개의 해결(문제해결) 루브릭을 설계해 상호작용 품질을 분해 측정한다.

- **Empirical Impact**: 실험 결과, 상담 기반 법률 추론에서 최고 모델 GPT-5.5도 0.562에 그쳐 전반적인 성능 여지가 크다는 점을 보여준다. 더 나아가 법률 상담 맥락에서 나타나는 sycophancy(아부/과잉동의) 문제와, 의뢰인이 가장 많은 안내를 필요로 할 때 오히려 성능이 떨어지는 역설적 현상을 드러내며 현장형 평가의 필요성을 강화한다.



### SANA: What Matters for QA Agents over Massive Data Lakes? (https://arxiv.org/abs/2606.13904)
Comments:
          9 pages, 7 figures

- **Prior Approaches**: 탐색형 질의응답(EQA)은 에이전트가 데이터 레이크에서 관련 소스를 찾아 분석하고, 중간 결과에 따라 다음 행동을 바꾸는 장기 루프 문제다. 기존 연구는 에이전트형 검색, 분해/계획, 도구 사용, 코드·SQL 생성 등 개별 능력을 개선했지만, 검색·계획·데이터 분석이 섞여 발생하는 실패 양상을 끝단 성능만으로 분리하기 어렵다. 또한 EQA 평가(예: DCI, Metadata Reasoner)는 검색/선택의 일부를 다루더라도, “무엇이 왜 실패했는지”를 에이전트의 정책(다음 행동 선택, 제출 시점)까지 포함해 진단하기는 제한적이었다.

- **Core Contribution**: 이 논문은 SANA(Search Agent Navigation Ablation)로 EQA를 런타임 프로파일로 바꿔 진단하는 프레임워크를 제안한다. 프로파일에는 정답 소스의 순서, 누출을 제거한 하위질문, 실행 기록(어떤 데이터에서 어떤 의도로 어떤 계산을 했는지)이 포함되어, 각 구성요소를 이상화(idealized)한 뒤 차이를 남겨 “정책 실패”를 추적한다. 즉, 검색·계획·데이터 분석 실행은 분리해 상한/하한을 만들고, 잔차 성능 갭은 에이전트가 잘못된 소스를 추구하거나 검증/중간 증거 추적을 놓치거나 잘못된 최종 제출을 하는 등 정책 문제의 증거로 남긴다.

- **Technical Challenges**: 핵심 난관은 구성요소를 단순히 교체하면 의도(intent)와 기대되는 도구 호출이 어긋나 공정한 비교가 깨진다는 점이다. SANA는 각 도구 호출의 의미적 의도를 자연어(검색 키워드, 분석 목표)로 추출해, 이상화된 구성요소가 “도구 품질”만 향상되고 정책의 다음 행동 선택 책임은 유지되도록 설계한다. 구체적으로는 (1) 검색 이상화에서 골드 소스 집합 안에서만 결과를 반환해 검색 정밀도를 통제하고, (2) 데이터 분석 이상화에서 분석 의도와 검증된 실행을 맞추지 못하면 더 강한 모델로 코드 생성·수정 및 재시도까지 수행해 실행 실패를 줄인다.

- **Empirical Impact**: LakeQA와 KramaBench(변환 버전)에서 고정된 프롬프트·예산·런타임 조건으로 경량/중간 크기 에이전트를 평가한 결과, 데이터 분석 실행이 두 벤치마크에서 일관된 병목으로 나타난다. 검색은 LakeQA의 대규모 데이터 레이크 설정에서는 큰 한계였지만, 더 작은 스케일의 KramaBench에서는 상대적으로 덜했으며, 계획은 전반적으로 개선 폭이 더 작았다. 무엇보다 end-to-end 정확도만으로는 섞여 보이던 실패를 검색·계획·분석 실행 및 잔차 정책 문제로 체계적으로 분해해, 향후 “증거 추적·검증·정답 제출/중단 기준” 같은 정책 설계 목표를 구체화하는 데 의미가 있다.



### Hybrid Classical-Quantum Variational Autoencoder for Neural Topic Modeling (https://arxiv.org/abs/2606.13852)
- **Prior Approaches**: 기존 토픽 모델링은 LDA 같은 베이지안 확률모형과 행렬 분해 기반 방법에 크게 의존해 왔지만, 대규모 데이터에서의 확장성과 주제-단어의 비선형 관계 포착에 한계가 있습니다. 이를 보완하려고 등장한 Neural Topic Models은 VAE 계열을 포함해 GPU 친화적으로 학습되지만, 양자 하드웨어와의 결합은 충분히 탐색되지 않았습니다. 또한 양자-하이브리드 오토인코더 선행연구는 이미지 등에서 양자 블록을 보조적으로 쓰는 경우가 많아, 토픽 모델의 평가 목적(응집도·다양성·잠재 공간 구조)과 직접 연결된 설계는 부족했습니다.

- **Core Contribution**: 이 논문은 매개변수 양자회로(parameterized quantum circuit)를 VAE의 추론(encoder) 네트워크에 삽입하되, 생성(decoder)은 전통적인 토픽-단어 복원 구조를 유지하는 하이브리드 고전-양자 VAE를 제안합니다. 특히 양자 장치의 자원 제약을 고려해, 토픽 수와 잠재 차원의 결합을 분리하는 modified Gaussian Softmax posterior를 도입하여 10-qubit 수준의 저자원 양자 디바이스에서도 동작하도록 설계했습니다. 결과적으로 토픽 응집도와 다양성을 함께 보존하면서도 양자 구성요소가 실용적으로 결합될 수 있음을 보여줍니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 양자 회로가 출력하는 측정값이 VAE의 확률적 잠재변수(평균·분산)를 안정적으로 학습 가능하게 만들면서도, 토픽 얽힘(entanglement)을 줄여야 한다는 점입니다. 논문은 강한 얽힘을 쓰는 양자회로의 측정 스케일링 학습, 그리고 주제 분포 샘플링에 GSM(Gaussian Softmax) 기반의 temperature까지 포함해 분리도와 분산을 조절하는 방식으로 이를 해결합니다. 또한 amplitude encoding과 Pauli Z 기대값/상태확률 측정의 조합을 통해 역전파 학습이 가능하도록 구성하고, 토픽 다양성 정규화항도 cosine 유사도로 재정의해 토픽 임베딩의 직교성을 유도합니다.

- **Empirical Impact**: AgNews에서 제안한 하이브리드 VAE는 기존 SOTA neural topic models보다 더 높은 성능을 보이며, 예로 C_v coherence 0.71과 NPMI 0.20을 달성하면서 주제 다양성도 유지했다고 보고합니다. 비교 실험으로 수행한 완전 고전(fully classical) 변형 역시 SOTA를 능가하고, 잠재 공간에서의 클래스 분리가 명확히 나타났습니다. 전반적으로 NISQ(잡음) 이전의 시뮬레이션 설정에서도 양자-하이브리드 VAE가 계산적으로 실행 가능하며, 양자 강화 토픽 모델링의 유망한 방향을 제시한다는 점에서 의미가 있습니다.



### When Plausible Is Not Realistic: Evaluating Human Mobility in LLM-Based Urban Simulation (https://arxiv.org/abs/2606.13835)
Comments:
          14 pages, 10 figures

- **Prior Approaches**: 기존 LLM 기반 도시 시뮬레이터인 AgentSociety와 CitySim은 생성된 일정과 활동이 “그럴듯한지”에 초점을 두는 경우가 많았다. 그 평가는 시맨틱한 활동 분포나 서사적 일관성은 볼 수 있지만, 실제 인간 이동의 핵심인 공간·시간 제약을 함께 재현하는지 검증하기엔 부족했다. 또한 관측 가능한 이동 흔적(traceability)과 의사결정 과정을 관찰가능성(observability) 있게 남기지 못해, 오류가 어디서 발생했는지 해석이 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 LLM 기반 도시 시뮬레이터의 이동을 실세계 데이터와 직접 대조해 검증하는 프레임워크를 제안한다. 공간 이동 법칙, 시간 리듬, 네트워크 모티프, 의미 기반 활동 전이, 행동 이동 프로필을 다차원 지표로 결합해 “서사적 그럴듯함”과 “실증적 이동 현실성”을 분리 평가한다. Greater Paris와 Shanghai 데이터로 AgentSociety와 CitySim을 평가해, 어떤 요소가 재현되고 어떤 요소가 실패하는지 체계적으로 보여준다.

- **Technical Challenges**: 기여를 실현하려면 (1) LLM 에이전트의 POI 선택과 실행을 추적 가능한 형태로 기록하고, (2) 시뮬레이터 간 비교를 공정하게 하며, (3) 대규모 지도 생성과 지표 계산을 반복 실험 가능하게 만들어야 한다. 이를 위해 AgentSociety를 기반으로 En-AgentSociety를 구축해 방문 POI/카테고리와 프롬프트-응답 및 실행 흐름을 로깅(traceability)하고, 병목·지연을 모니터링할 수 있게 했으며, 지도 생성은 지역 단위로 확장되도록 최적화했다. 또한 CitySim은 공개 설명을 바탕으로 재구현·수정해 동일 인프라에서 일관된 비교가 가능하게 했고, 트립 길이·기원-목적지 흐름·체류시간·전이 역학 등 다양한 지표 계산 절차를 마련했다.

- **Empirical Impact**: 실험 결과는 시맨틱 활동 분포 같은 고수준 패턴은 일부 따라가도, 여행거리 분포·기원-목적지 흐름·체류시간·전이 동역학 같은 핵심 공간/시간 제약을 재현하는 데 큰 격차가 있음을 보여준다. 특히 기본 프롬프트(persona) 설정만으로는 이동 다양성이 안정적으로 확보되지 않으며, 프로필을 반영한 초기화가 필요하다고 관찰했다. 더불어 지역 규모 맵 생성, 관찰가능성 강화, 이동 메트릭 산출, 교통 시뮬레이터까지 포함한 공개형 인프라를 제공해 재현 가능한 벤치마킹을 촉진한다.



### The Culture Funnel: You Can't Align What isn't in the Data (https://arxiv.org/abs/2606.13808)
- **Prior Approaches**: 기존 문화 정렬(cultural alignment)은 주로 추론 시점 프롬프팅, 벡치마킹, 정렬용 튜닝처럼 “모델 안에 문화 지식이 이미 있다”는 가정을 두고 해결하려는 접근이 많았습니다. 그러나 실제로는 문화가 특정 맥락 밖에서 잘 발현되지 못하며, 그 원인을 데이터 파이프라인의 구성 변화로 보지 못한 한계가 있었습니다.

- **Core Contribution**: 이 논문은 LLM 개발 파이프라인의 후처리(post-training) 과정에서 문화 신호가 압축되는 현상을 “culture funnel”로 규정하고, 훈련 단계별 문화 데이터의 변화를 계량화합니다. 또한 문화-지리-언어-도메인-태스크를 다차원 태깅 프레임워크로 명시화해, 문화 정렬이 추론 문제가 아니라 훈련 데이터 파이프라인 문제임을 보여줍니다.

- **Technical Challenges**: 핵심 기술 과제는 문화가 문맥 의존적이고 애매하다는 점에서 대규모 데이터에 일관되게 문화 태그를 부여하는 것입니다. 연구진은 5.6M 샘플을 5개 차원(문화 4클래스 포함, 도메인, 태스크 의도, 지리, 언어)으로 자동 태깅하되, 인적 검토와 Krippendorff’s α로 신호의 대규모 추세 신뢰성을 확인했습니다.

- **Empirical Impact**: 분석 결과 문화가 전처리(pretraining)에서 후처리(sft·정렬·추론 데이터)로 갈수록 뚜렷하게 감소하며, 특히 수학·코딩·기술 중심 도메인이 문화 비율을 더 낮추는 경향이 나타났습니다. 더불어 다국어 확장은 문화 균형을 자동으로 보장하지 못하고(지리 분포는 여전히 장꼬리·불균등), 문화 태그를 메타데이터로 “표시”해 학습시키면 일부 문화 벤치마크 성능이 크게 개선됨을 실험으로 입증했습니다. 이를 위해 문화 태깅 데이터셋도 공개해 후속 연구의 기준점을 제공합니다.



### QIAS 2026: Overview of the Shared Task on Islamic Inheritance Reasoning (https://arxiv.org/abs/2606.13756)
- **Prior Approaches**: 기존 이슬람 지식/법률 관련 벤치마크는 주로 텍스트 매칭 기반 지식 확인(예: Quran·Hadith QA)이나 다지선다 형태의 이해도 평가에 치우쳤습니다. 특히 QIAS 2025 같은 MCQ 설정에서는 정답을 고르더라도 중간 추론의 합법성·적법성(이유의 타당성)을 검증하기 어렵다는 한계가 지적됐습니다. 또한 다단계 추론이 필요한 작업은 RAG만으로는 규칙 기반 계산과 단계 간 오류 전파를 충분히 해결하기 어렵다는 문제가 남았습니다.

- **Core Contribution**: QIAS 2026은 ‘ilm al-mawārīth(이슬람 상속법)’을 자연어 사례로부터 끝-끝 추론(end-to-end reasoning)까지 수행하는 능력을 평가하도록 설계됐습니다. 모델이 상속 대상자 식별(막힘/차단 포함)부터 법정 지분(furūḍ) 배정, ‘awl·radd 같은 조정 판단, 최종 분배 계산까지 전체 과정을 구조화된 출력으로 생성하도록 요구합니다. 이를 위해 MAWARITH(아랍어 12,500건)와 단계별 중간추론 검증을 포함한 MIR-E 평가체계를 함께 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 법학적 해석 규칙에 따라 유효 상속인과 차단(hajb)을 정확히 판정하고, (2) 법정 지분 배정과 ‘awl/radd 조정 조건을 구조화해 올바르게 적용하며, (3) 마지막 숫자 계산이 중간 단계의 오류 없이 일관되게 이어지는 것입니다. 논문은 MIR-E가 네 단계(상속인·차단, 지분 배정, 조정, 최종 분배)를 가중 다단계로 채점해 이러한 단계 간 연쇄 오류를 세밀하게 드러내도록 했습니다. 참여 팀들은 RAG+규칙 제약 JSON 생성, 파인튜닝(QLoRA 포함), 신경-상징 파이프라인(추출 후 상징 규칙 계산) 등으로 구조화 출력의 정확도를 높였습니다.

- **Empirical Impact**: 결과적으로 대부분의 시스템은 특히 정밀한 법적 해석과 구조적 수치 추론이 요구되는 단계에서 여전히 큰 어려움을 보였고, 상용 모델이 대체로 더 높은 성능을 보였습니다. 최상위 팀은 RAG 기반 파이프라인으로 MIR-E 0.935를 달성했고, 4B 파라미터급 오픈 모델도 QLoRA 다단계 미세조정을 통해 상용 Gemini-2.5-Flash(0.901)와 유사한 수준(0.907)을 보여주었습니다. 이는 이슬람 상속법처럼 엄격한 규칙+계산이 결합된 도메인에서 ‘진짜 추론’을 평가하는 벤치마크 설계와 단계별 검증의 중요성을 실증적으로 강화한 결과로 평가됩니다.



### Which Models Perform Better in Inheritance Reasoning? (https://arxiv.org/abs/2606.13751)
- **Prior Approaches**: 기존 연구는 QIAS(이슬람 상속 추론)처럼 규칙 기반 법률 영역에서 체계적 추론이 필요하다는 점을 강조하며, 체인 오브 쏘트(Chain-of-Thought)나 자기일관성(Self-Consistency) 같은 프롬프팅 개선과 추론 강화형 모델(o1 계열, DeepSeek-R1 등)을 활용해 성능을 끌어올렸다. 또한 QIAS 2025에서는 Gemini·ChatGPT가 여러 아랍어/오픈소스 모델을 전반적으로 앞섰고, 예외적으로 Qwen3나 멀티 에이전트 하이브리드 접근이 강점을 보였다는 보고가 있었다. 다만 이러한 방법들은 ‘정확한 수치 계산’과 ‘의존적인 법 규칙의 연쇄 결정’에서 얼마나 안정적으로 버티는지를 동일 조건에서 비교하기가 어려웠다.

- **Core Contribution**: 본 논문은 QIAS 2026 Shared Task에서 상속 추론을 “특화 파이프라인 없이” 일반 목적 모델의 구조적 추론 능력을 공정하게 비교하는 것이 핵심 기여다. 특히 상업용·오픈소스 모델을 동일 프롬프팅 프레임워크로 평가해, 모델군 간 신뢰도(reliability) 격차를 정량·정성적으로 드러낸다. 이는 법률/규칙 기반 과제에서 텍스트 그럴듯함보다 ‘단계별 구조 일관성’이 성능을 좌우한다는 관점을 강화한다.

- **Technical Challenges**: 상속 추론은 상속 대상자 식별→차단·배제 규칙 적용→법정 지분 배정→필요 시 awl 또는 radd 같은 조정 여부 및 정규화까지 여러 단계가 서로 의존하며, 한 단계의 구조 오류가 뒤의 전체 분배를 무너뜨린다. 또한 분모·분수의 합이 정확히 맞아야 하므로 최종 수치 계산의 일관성이 중요하다. 논문은 이 문제를 해결하기 위해 비교 실험을 단순화한 구조화된 CoT 프롬프팅과 출력 포맷 제약 및 소량의 후처리만 사용해, 모델이 스스로 올바른 법률 구조와 수치를 유지하는지 직접 관찰하도록 설계했다.

- **Empirical Impact**: 실험 결과, 상업용 모델이 오픈소스 모델보다 안정적이고 정확도가 높았으며(예: Gemini 2.5 Pro 0.931, Gemini 2.5 Flash 0.898), 오픈소스는 큰 폭으로 낮아 33.1~45.1 범위에 머물렀다. 모델군 간 차이는 단순 평균 성능을 넘어, 상업용이 초기 단계의 모호한 결정에서도 일관성을 더 잘 유지하고 의존 단계에서 오류 전파가 적다는 질적 관찰로 이어졌다. 반면 오픈소스는 상속자 누락/불필요한 친족 추가, 배제 규칙 오적용, 잘못된 지분 할당, 합·정규화 불일치 같은 구조적 오류가 반복됐고, 그 결과 최종 예측이 깨지곤 했다. 결론적으로 이슬람 상속 추론은 아랍어 법률 영역에서 현재 LLM의 “단계 연쇄 추론 안정성”을 가르는 어려운 벤치마크이며, 최고의 제출 성능은 Gemini 2.5 Flash의 MRE 0.989로 보고되었다.



### Benchmarking Web Agent Safety under E-commerce Deceptive Interfaces (https://arxiv.org/abs/2606.13686)
Comments:
          Accepted to ACL 2026

- **Prior Approaches**: 기존 연구는 웹 에이전트를 악성 사용자 지시나 프롬프트 인젝션, 또는 웹페이지에 섞인 적대적 콘텐츠로 흔드는 방식으로 안전성을 주로 평가했다. 또 일부는 UI 방해 요소(오류 팝업 등)가 의사결정을 어떻게 교란하는지 보려 했지만, 실제 전자상거래에서 흔한 ‘사람이 설계한’ 기만적 인터페이스 패턴을 체계적으로 스케일해 주입·평가하는 데는 한계가 있었다.

- **Core Contribution**: 이 논문은 WebDecept라는 경량 플러그인 프레임워크를 제안해, 기존 웹환경(VisualWebArena의 쇼핑 시나리오)에 기만적 인터페이스 패턴을 실행 중에 통제적으로 삽입한다. 이를 통해 타깃 광고, 도메인 리다이렉션, 장바구니 조작(추가 옵션, 가격 드리프트) 등 7개 패턴을 같은 형식으로 실험에 포함하고, 여러 멀티모달 웹 에이전트의 안전 실패를 end-to-end로 측정한다.

- **Technical Challenges**: 핵심 기술적 난관은 현실성은 유지하되 재현 가능하게 ‘프론트엔드 수준’ 기만 패턴을 주입하고, 에이전트가 이를 관찰하는 경로(화면 이미지 vs 접근성 트리)를 일관되게 맞추는 것이다. WebDecept는 상태 기반·트리거 기반으로 특정 시점에 개입을 적용하고, 스크린샷과 접근성 트리 관측을 바꿔 에이전트가 수정된 인터페이스에 적응하도록 평가한다.

- **Empirical Impact**: 실험 결과, 최신 멀티모달 웹 에이전트들은 다수 유형의 기만 인터페이스에 취약하며 특히 ‘장바구니/결제 상태를 조용히 바꾸는’ 조작에서 방어가 약한 것으로 나타났다. 또한 프롬프트 기반 안전 제약은 일부 팝업·배너에는 효과가 있으나 가격 드리프트·리다이렉션 등 미세한 조작에는 자주 부족했고, 특정 모델에서는 STOP 대신 환경을 자율 복구하며 진행하는 “Proactive Recovery” 경향도 관찰돼 안전 준수와의 긴장을 시사한다.



### The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation (https://arxiv.org/abs/2606.13685)
Comments:
          24 pages, 7 figures

- **Prior Approaches**: LLM-as-a-Judge는 모델 출력 순위화, 보상모델 학습, 리더보드 채점에서 널리 쓰이지만, “같은 조건에서 재실행하면 같은 결론이 나오는가”는 상대적으로 덜 측정돼 왔습니다. 기존 연구들은 주로 위치 편향(먼저 제시된 응답 선호), 과장/자기강화, 길이·형식 같은 편향을 단일 샘플 관점에서 다뤘고, 신뢰도(변동성) 자체를 반복 샘플로 정량화한 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 LLM 심판의 신뢰도를 단일 수치가 아니라 네 층(쌍대 판정의 내부 변동, 점수의 변동, 동일 심판 내 반복 일관성, 서로 다른 심판 모델 간 합의)으로 분해해 측정 체계를 제시합니다. 또한 쌍대 선택이 점수 차이를 뒷받침하지 못하는 ‘쌍대-점수 역설(pairwise–pointwise gap)’을 실증적으로 보여, 리더보드식 단일 재판정이 흔들릴 수 있음을 강조합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “반복 재평가 시 판정이 얼마나 바뀌는지”를 항목 단위로 고정밀 추정하는 것입니다. 저자들은 29개 태스크(10개 범주)에서 두 OpenAI judge 모델(GPT-4o-mini, GPT-4.1-mini)을 대상으로 질문당 쌍대 50회/점수 50회를 반복하고, 온도(temperature)·프롬프트 민감도(서술 템플릿)까지 통제해 변동의 원인을 분리했습니다.

- **Empirical Impact**: 실험 결과 쌍대 선호는 평균 13.6%에서 뒤집히며, 20% 이상 뒤집히는 질문이 28%나 됩니다; 점수 차이는 10점 척도에서 0.19~0.36 수준으로 집계 유의성도 약해, 승자 판정이 ‘품질 격차’ 증거로 과대해석될 수 있음을 시사합니다. 또한 심판 간 합의는 76%(κ=0.51)로 중간 수준에 그치고, 동일 심판도 프롬프트를 바꾸면 다수결 결과가 25%에서 달라져 고위험 평가에선 다중 재판정·위치 무작위화·불확실성 보고가 표준이 되어야 한다는 결론을 냅니다.



### Gaze Heads: How VLMs Look at What They Describ (https://arxiv.org/abs/2606.14703)
- **Prior Approaches**: 기존 기계해석 연구는 비전-언어 모델(VLM)에서 이미지 토큰을 주목하는 attention head들을 “신호”로만 다뤘습니다. 예를 들어 Image Heads는 단일 패스의 이미지 주목 특성을 이용해 대비적 디코딩 신호를 만들고, Localization Heads는 소수 head의 공간 집중도를 읽어 시각적 grounding(박스/마스크)에 활용합니다. 하지만 이런 방법들은 생성 시점마다 어떤 영역을 ‘지금 말하는 중’으로 라우팅하는지, 그리고 그 head들이 출력 선택을 ‘인과적으로’ 바꾸는지까지는 답하지 못했습니다.

- **Core Contribution**: 이 논문은 이미지 서술 과정에서 모델이 사람의 시선처럼 “현재 말하는 이미지 영역”으로 attention을 좁히는 메커니즘이 있음을 보여줍니다. 그 핵심이 언어모델 backbone의 소수 attention head인 gaze heads이며, 이 head들은 패널/객체를 단계적으로 전환하며 해당 영역에 시선을 고정합니다. 더 나아가 gaze heads의 attention을 원하는 영역으로 강제로 바꾸면 VLM의 문장이 실제로 그 영역을 묘사하도록 전환되어, head를 통한 출력 제어의 인과성을 제시합니다.

- **Technical Challenges**: 첫째, 헤드가 너무 많아 ‘어떤 head가 시각적 라우팅을 담당하는지’를 라벨 없이 찾아야 했습니다. 저자들은 몇 번의 forward pass에서 ‘질문한 패널(또는 영역)과 대응되는 이미지 토큰에 attention이 재배치되는지’만을 상관 점수로 측정해 gaze heads를 좁혀냈고, 이 점수는 고정된 단일 패스 집중도와 달리 시간(질문 변화)에 따른 추적성을 반영합니다. 둘째, 인과성 검증을 위해 학습 없이 inference-time에서 특정 head들의 attention 마스크를 편집해 타깃 영역으로 강제 이동시키는 개입을 설계했으며, 개입 범위를 늘리면 제어가 포화되거나(부분 제어) 과도하면 생성이 붕괴되는 “정밀한 튜닝” 현상도 확인했습니다.

- **Empirical Impact**: 실험은 만화(comic strip)라는 통제된 환경에서 시작해, 상위 100개 gaze heads(전체 head의 8.7% 수준) 개입만으로 선택한 패널로 답을 유도하는 정확도 83.1%를 달성했습니다(무작위 head 개입은 실패, 전부 개입은 붕괴). 연속 서술에서도 생성 도중 타깃을 바꾸면 모델이 해당 패널 설명을 마무리하고 새 패널로 넘어가며, natural 이미지에서는 COCO 객체 영역으로 유사한 유도 효과가 나타납니다. 또한 모델 규모(2B~32B)와 여러 VLM 아키텍처로 확장되지만, vision encoder를 고정하는 일부 계열에서는 gaze heads에 해당하는 명확한 집합이 관측되지 않아 “학습 방식 의존성”까지 시사합니다.



### ClinHallu: A Benchmark for Diagnosing Stage-Wise Hallucinations in Medical MLLM Reasoning (https://arxiv.org/abs/2606.14697)
Comments:
          Code and datasets: this https URL

- **Prior Approaches**: 기존 의료 홀루시네이션 벤치마크는 정답/오답 같은 최종 출력 중심으로 평가해, 오류가 추론 과정의 어디에서 발생하는지(시각 인식 실패, 의학 지식 회상 실패, 근거 통합 실패)를 분리해 진단하기 어렵습니다. CARES·Med-HallMark 등 멀티모달 평가가 확장됐더라도 여전히 “결과가 틀렸는지”에 집중하는 경향이 강했습니다. 그 결과 같은 오답이라도 서로 다른 원인 실패가 하나의 판단으로 뭉쳐지는 한계가 있었습니다.

- **Core Contribution**: ClinHallu는 의료 MLLM 추론을 Visual Recognition(시각 인식), Knowledge Recall(지식 회상), Reasoning Integration(근거 통합) 3단계로 분해해, 단계별 홀루시네이션 원인을 추적하는 벤치마크를 제안합니다. 4개 의료 VQA 데이터셋을 기반으로 총 7,031개의 검증된 인스턴스를 만들고, 각 샘플에 검증된 기준 추론 트레이스를 붙여 “어떤 단계가 병목인지”를 진단할 수 있게 했습니다. 또한 단계 교체(stage-replacement) 개입으로 특정 단계 수정이 최종 정답에 어떻게 영향을 주는지도 측정합니다.

- **Technical Challenges**: 핵심 기술 과제는 기준 추론 트레이스를 대규모로 생성하되, 포맷이 올바르고 정답과 모순되지 않는 “신뢰 가능한 트레이스”만 남기는 데 있습니다. 논문은 기준 트레이스를 생성한 뒤 LLM-as-judge로 트레이스 포맷 유효성과 정답 일관성을 검증해 필터링했고, 그 결과만 벤치마크에 사용했습니다. 평가 단계에서는 후보 모델이 생성한 트레이스와 기준 트레이스를 비교하는 방식으로 단계별 홀루시네이션을 라벨링하며, upstream 단계를 기준으로 고정한 상태에서 stage-replacement로 원인 분리를 수행합니다.

- **Empirical Impact**: 실험 결과, 홀루시네이션 병목은 데이터셋마다 다르게 나타났습니다(예: VQA-RAD는 시각 오류 비중이 높고, MedXpertQA는 지식 오류 비중이 큼). 또한 추론 통합 단계의 홀루시네이션은 전반적으로 시각·지식 단계보다 낮게 나타나 “신뢰성 실패의 주원인이 마지막 추론 자체라기보다 상류 단계”일 수 있음을 보여줍니다. 더 나아가 트레이스 슈퍼바이즈드 파인튜닝은 단계별 홀루시네이션을 줄이고 정답 정확도를 개선했으며, 자동 심판 모델의 판단도 인간 라벨과 높은 일치도를 보여 대규모 진단 체계로서의 실용성을 뒷받침합니다.



### Persona-Pruner: Sculpting Lightweight Models for Role-Playing (https://arxiv.org/abs/2606.14695)
Comments:
          25 pages; ICML 2026; Code is available at this https URL

- **Prior Approaches**: 기존 역할극(페르소나) 챗봇 연구는 프롬프트로 페르소나를 유도하거나, 특정 모델을 파인튜닝해 성향을 고정하는 방식이 주를 이뤘습니다. 네트워크 프루닝도 널리 쓰이지만, 기존 기법들은 보통 “일반 능력 유지” 또는 “특정 과업 최적화”처럼 목표가 넓거나 별도의 과업 데이터에 의존하는 경향이 있어 개인화·고유 페르소나에 그대로 적용하기 어렵습니다.

- **Core Contribution**: 이 논문은 페르소나 하나를 위해 일반ist 모델 전체를 쓰는 것이 항상 필요하지 않으며, 페르소나 정체성에 해당하는 부분 서브네트워크만 남길 수 있다는 가설을 제시합니다. 이를 위해 Persona-Pruner라는 프레임워크를 도입해, 텍스트 페르소나 정의만으로 경량 역할극 모델을 “조형(sculpting)”합니다.

- **Technical Challenges**: 핵심 기술 난제는 프루닝이 단순히 불필요한 가중치 제거가 아니라, 페르소나의 본질적 특성까지 함께 잘라버려 성능이 크게 하락한다는 점입니다. Persona-Pruner는 (1) 페르소나에 민감한 질의를 일반 데이터에서 찾아내고 답을 페르소나 스타일로 재작성해 Persona-driven Data Synthesis으로 보정용 캘리브레이션 데이터를 만들고, (2) 트랜스포머 FFN의 중간 차원에 대한 마스크를 학습해 페르소나-핵심 서브네트워크만 남기도록 구성합니다.

- **Empirical Impact**: 실험 결과 Persona-Pruner는 RoleBench에서 LLM-as-a-judge 점수 기준으로 기준선 대비 성능 저하를 최대 93.8%까지 줄이며, 고스파시티(예: 50%)에서도 짧은 복구 파인튜닝 후 역할극 품질을 상당 수준 유지합니다. 동시에 OpenBookQA, PIQA 같은 일반 추론 벤치마크에서 일반 능력을 크게 해치지 않아, 페르소나 특성만 우선적으로 보존하는 프루닝이라는 점이 실증적으로 확인됩니다.



### Flood and Harvest: The Provable Necessity of Trivia for Generating Valuable Mathematics via the Lens of Language Generation in the Lim (https://arxiv.org/abs/2606.14688)
- **Prior Approaches**: 증명 보조 정리증명기와 결합된 생성 모델은 형식적으로 “검증 가능한 정리”를 대량으로 뽑아낼 수 있지만, 무엇이 “수학적으로 가치 있는가”는 보통 공식적으로 다뤄지지 않았다. 기존의 학습-이론적 생성-in-the-limit 연구는 목표 언어를 한 층으로 두고(단일 언어), 폭(breadth)은 보장하되 가치의 ‘미기록분’을 다루는 데 한계가 있다.
또한 기존의 ‘검증기’ 관련 연구들은 다른 오라클(예: 집합 구성원 자체에 대한 멤버십 질의)이나 계산 복잡도 중심이라, 본 논문이 정의하는 “검증 가능한 상위 언어(형식 세계)”와 “가치 목표(미지의 언어)”의 중첩 구조가 결여돼 있었다.

- **Core Contribution**: 이 논문은 검증기(멤버십 오라클)가 판정하는 형식 언어 F와, 생성기가 실제로 맞춰야 할 가치 언어 H를 중첩(nested)시켜 “가치 대 검증”의 간극을 수학적으로 모델링한다. 가치 언어 H의 일부만 문헌 코어 C로 노출되며, 생성기는 오직 F의 오라클 질의로만 살아남아, 출력이 가치/무의미/환각 중 어디에 해당하는지의 상충을 최대로(또는 최소화로) 보장할 수 있는지를 한계와 가능성으로 정리한다.
특히 “검증기는 취향(taste)을 대신 학습하지 못한다”는 점을 정보이론적으로 분명히 하고, 그 결과 가치 커버리지(coverage)는 본질적으로 ‘무의미하지만 검증된 trivia를 얼마나 내놓을 수 있느냐’에 의해 결정된다는 결론을 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 생성기가 H를 직접 보지 못하고, (2) 오라클은 H가 아니라 F에 대해서만 알려주며, (3) 목표는 유한 시점이 아니라 전체 무한 실행의 한계 집합 기반 커버리지라는 점이다. 저자들은 이 구조에서 검증기의 정보가 무엇을 바꾸는지(거짓 출력 제거)와 무엇을 절대 못 바꾸는지(가치 식별)로 분리하기 위해, 섬유(fiber)별로 문제를 고정된 F로 상대화하고 Angluin의 tell-tale 조건 같은 ‘폭 가능성’ 성격 규칙을 다시 끼워 맞춘다.
또한 trivia/미스(가치 누락) 사이의 날카로운 상호작용을 보이기 위해, 관찰 순서를 강제하는 경쟁(adversarial racing)과 밀도 기반 “sweep pointer” 스케줄링을 결합해, 오라클-비보조 모델에서는 성립하지 않는 최적 경계를 도출한다.

- **Empirical Impact**: 실험 대신 ‘가능/불가능’의 이론적 최적값을 제시함으로써, 앞으로 AI4Math 및 자동정리탐색 시스템에서 “검증 성능을 올리면 가치도 따라오나?”라는 질문에 대한 정답을 준다. 결론은 명확히, 가치가 미기록된 질량(1-α)에 해당하는 만큼은 결국 “정답이지만 무의미한” 검증된 trivia의 무한 스트림이 필요하며, 그 양은 탐색 설계에서 기계적으로 나타난다는 것이다.
또한 문헌 코어 밀도 α에 대해 최적 커버리지가 유한 trivia 허용 구간에서는 α/2로, 무한 trivia(비율이 0으로 수렴해도) 허용 구간에서는 1-α/2로 점프한다는 이분법을 타이트하게 증명해, 시스템이 ‘얼마나 많은 불필요 산출물을 감당할지’가 가치 수확의 상한을 직접 결정함을 보여준다.



### Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows (https://arxiv.org/abs/2606.14672)
- **Prior Approaches**: 기존 에이전트 워크플로는 병렬로 여러 가지 하위 작업을 수행한 뒤 마지막에 합치는 형태(DAG 후 합성)로 많이 설계되지만, LLM은 순차 텍스트 접점에 맞춰 상태를 선형화해 전달하는 경우가 대부분이다. 그래서 합성 단계에서 각 브랜치 결과를 텍스트로 이어 붙여 프리필을 다시 수행하거나(중복 계산) 요약 기반 전달로 세부 정보가 손실되는 한계가 있었다. 또한 KV 캐시 재사용 연구는 주로 RAG에서 독립 인코딩된 증거 조각을 묶어 쓰는 문제에 초점이 있어, 에이전트 브랜치의 로컬 맥락에서 생성된 캐시를 다대일로 읽고 판단·통합해야 하는 합성 문제와는 요구 역량이 다르다.

- **Core Contribution**: 이 논문은 병렬 에이전트 브랜치가 생성한 KV 캐시를 합성기가 직접 소비하도록 하는 플러그앤플레이 프레임워크 Parallel-Synthesis를 제안한다. 핵심은 텍스트를 다시 이어 붙여 프리필하지 않고, 합성기 쪽에서 “병렬 캐시 인터페이스”를 해석·생성 가능하게 적응시키는 것이다. 이를 위해 브랜치 캐시를 보정하는 cache mapper와 합성기 모델을 해당 인터페이스에 맞게 학습시키는 synthesizer LoRA 어댑터를 함께 둔다.

- **Technical Challenges**: 가장 큰 기술적 난점은 병렬 작업에서 온 캐시들이 단일 연속 prefix에서 이어진 것이 아니라 서로 다른 로컬 컨텍스트로 생성되었다는 점이다. 논문은 이를 해결하기 위해 (1) 공유된 분기 지점 이후의 위치에 맞춰 RoPE 기반으로 캐시의 위치 정렬을 수행하는 positional re-encoding, (2) 브랜치 길이와 개수 같은 메타 정보를 바탕으로 키·값에 대한 선형 변환을 학습하는 cache mapping, (3) 캐시 기반 합성이 텍스트 직렬 합성과 유사한 추론을 하도록 적응시키는 synthesizer LoRA를 단계적으로 결합한다. 또한 학습 단계에서는 병렬 캐시 문맥에 대한 적응 데이터와, 텍스트 직렬 합성 경로에서 얻은 추론 궤적을 증류해(Reasoning distillation) 캐시 기반에서도 비교·판단·통합 능력을 강화한다.

- **Empirical Impact**: 수학, 과학 QA, 코드 생성, GAIA, 멀티 에이전트 데이터베이스 진단 등 9개 다운스트림 데이터셋에서 Parallel-Synthesis는 7개에서 텍스트 기반 합성과 동등하거나 더 높은 성능을 보였고 2개에서도 근접한 격차를 유지했다. 효율 면에서는 재프리필을 피함으로써 time-to-first-token(TTFT)이 2.5배~11배 줄어드는 개선을 보였다. 특히 추론이 무거운 작업에서도 이득이 관찰되어, 병렬 합성을 위한 “더 네이티브한(원형 구조에 가까운) 인터페이스”가 단순 속도 최적화가 아니라 품질에도 기여할 수 있음을 시사한다.



### Abstracting Cross-Domain Action Sequences into Interpretable Workflows (https://arxiv.org/abs/2606.14654)
Comments:
          preprint; 9 pages, 5 figures

- **Prior Approaches**: 기존 연구는 타임스탬프 UI 로그를 빈발 항목 집합 마이닝이나 순차 패턴 마이닝으로 요약해 왔지만, 토큰에 의미를 접지하기 어려워 잡음과 스푸리어스 상관에 취약했다. 딥러닝 기반 순차 모델도 도메인·태스크별 학습과 라벨링이 필요해 새로운 서비스나 행동 변화에 대응하기 비용이 컸다. 또 텍스트 기반 의도 추정은 목적이 이미 언어에 드러나는 경우가 많아, 저수준 이벤트만 있는 로그 문제와는 난이도가 다르다.

- **Core Contribution**: 이 논문은 WorkflowView라는 LLM 기반 계층적 추상화 프레임워크를 제안한다. 저수준 행동 시퀀스를 먼저 자연어 설명으로 변환한 뒤, 이를 바탕으로 고수준 활동과(필요 시) 범주까지 추론해 해석 가능하고 실행 가능한 인사이트로 연결한다. 핵심은 잡음이 섞인 로그를 단계적으로 “점진적 노이즈 제거” 형태로 다루면서, 제로샷/퓨샷만으로도 도메인 전이를 노린다는 점이다.

- **Technical Challenges**: 가장 큰 기술 과제는(1) 로그가 너무 세분화·잡음이 많아 의도가 직접 드러나지 않는다는 점과(2) 자연어 모델이 UI 이벤트 문맥을 제대로 추론하도록 프롬프트를 계층적으로 설계해야 한다는 점이다. WorkflowView는 Layer 1에서 관측된 행동을 자세한 자연어로 풀어쓴 뒤, Layer 2(및 Layer 3)에서 시간적 패턴과 중요도가 다른 행동을 점진적으로 정리해 고수준 활동을 추출한다. 또한 범주가 사전에 정해져 있지 않은 경우에는 별도 활동 분류(라벨 생성) 단계를 결합해 데이터에 맞춘 분류 체계를 만든다.

- **Empirical Impact**: 실험은 브라우저 로그 태스크 설명 재구성(코사인 유사도 평균 0.91), MOOC 수강생 이탈 예측(가중 F1 0.90, 퓨샷 예시 5개 수준), Microsoft Word에서 AI 도구 통합 맥락 분석(익명·집계 인사이트) 등 3개 서로 다른 도메인으로 검증됐다. 결과적으로 제로샷/퓨샷만으로도 기존 학습 기반 접근과 견줄 만한 성능과 높은 의미적 정합성을 보였다는 점이 의미 있다. 더 나아가 저수준 텔레메트리로부터 프라이버시를 지키는 집계형 관찰을 만들 수 있어, 실제 제품 개선(예: AI 출력 수용 이후 편집·서식 전환 패턴 반영)에 바로 연결될 잠재력이 제시된다.



### Every Eval Ever: A Unifying Schema and Community Repository for AI Evaluation Results (https://arxiv.org/abs/2606.14516)
- **Prior Approaches**: 기존 AI 평가는 다양한 리더보드, 논문, 블로그, 평가 하네스 로그에 흩어진 결과를 주로 표 형태의 단일 점수로 요약해 왔습니다. 또한 lm-eval-harness, HELM, Inspect AI 같은 평가 프레임워크가 서로 호환되지 않는 형식과 메타데이터를 사용해, 겉보기로 같은 평가라도 점수가 달라 비교가 흔들렸습니다.

- **Core Contribution**: Every Eval Ever(EEE)는 평가 결과를 단일 JSON 스키마로 표준화해 “점수”가 아니라 “해석 가능한 맥락”까지 함께 저장하도록 설계했습니다. EEE는 원천이 리더보드든 논문이든 관계없이 스키마에 흡수하고, 가능할 때는 인스턴스 단위 출력까지 선택적으로 포함해 재분석의 기반을 제공합니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 출처의 결과 포맷과 메타데이터를 동일한 의미 단위로 매핑하는 것이었습니다. 이를 위해 community-governed 메타데이터 스키마와 인스턴스 레벨 동반 스키마를 만들고, HELM·lm-eval-harness·Inspect AI 등 주요 하네스/포맷용 자동 컨버터를 제공하며, Pydantic 기반 검증 파이프라인으로 제출 시 스키마 준수와 필드 일관성을 강제합니다.

- **Empirical Impact**: EEE 저장소는 Hugging Face에서 커뮤니티 크라우드소싱으로 확장되어 22,235개 모델, 2,273개 벤치마크, 31개 평가 포맷을 포괄합니다. 이를 바탕으로 평가 하네스에 따른 재현성 격차, 에이전트 평가의 비용-정확도 상충, “perplexity”처럼 라벨만 같은 지표의 구현 의존성 등을 메타 분석 수준에서 드러내며, 분야의 비교 가능성과 비용 효율(재실행 절감)을 동시에 높였다는 점에서 의미가 큽니다.



### GitOfThoughts: Version-Controlled Reasoning and Agent Memory You Can Replay, Diff, and Merg (https://arxiv.org/abs/2606.14470)
Comments:
          10 pages, 1 figure, 9 tables

- **Prior Approaches**: 기존 연구는 추론 과정을 내부 메모리·버퍼로만 다루며, 에피소드가 끝나면 체인/트리 기록이 사라져 재현·감사·병합이 어렵다는 한계가 있었다. 또한 벡터·그래프·커스텀 스토어 같은 ‘메모리 형식’은 제안됐지만, 정확도 향상 여부는 일관되지 않았고 특히 방법 전이(transfer) 관점의 증거가 약했다.

- **Core Contribution**: 이 논문은 LLM 에이전트의 추론 트리를 GitOfThoughts로 버전관리해, 커밋·노트·태그로 점수화/결과를 기록하고 retrieval을 git log처럼 수행하게 한다. 이를 통해 추론을 재실행(replay)·감사(audit)·병합(merge) 가능하게 하면서도 정확도는 기존 메모리 스토어와 ‘동등’한 수준을 목표로 검증한다. 동시에 “어떤 메모리 형식이든 새 문제(novel problem) 정확도를 올리나?”라는 가설을 전면적으로 반박·검정한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 추론 노드를 안정적으로 점수/결과와 함께 영속 저장하고, (2) 저장된 내용을 에이전트 내부 retrieval로 공정하게 비교하며, (3) 여러 에이전트 경험을 충돌 없이 병합해 감사를 가능하게 하는 것이다. 논문은 생각 생성은 기존 tree-of-thoughts/ ReAct 루프를 유지하되, score 시점에 thought·scores·trace를 커밋하고 태그로 outcome을 남기는 형태로 Git 매핑을 구현해 “저장/조회/병합”만 바꿔가며 실험한다.

- **Empirical Impact**: GPQA-Diamond와 MATH-500에서 교차문제·교차에피소드 전반의 사전등록(pre-registered) 검증 결과, 대부분의 메모리 형식은 새 문제 정확도를 신뢰할 만하게 개선하지 못했다(특히 40개 샘플에서 보였던 git의 유망 신호는 더 큰 재현에서 붕괴). 다만 메모리가 ‘먹히는’ 조건은 복사 가능성(copyability) 임계값으로, 검색된 사례가 거의 중복에 가까울 때(유사도 약 0.8 이상) 정확도가 급상승하고, 그 아래에서는 이득이 없다. 반면 방법 전이(작동 원리 추출)는 모델 스케일을 키워도 거의 나타나지 않았고, 정확도를 주로 움직이는 안정적 레버는 메모리가 아니라 테스트 시 sampling(self-consistency) 같은 추론 선택 전략이었다.



### Be My Tutor: On-Policy Co-Distillation for Mutual LLM Improvement via Peer Feedback (https://arxiv.org/abs/2606.14368)
- **Prior Approaches**: 기존 멀티 도메인 LLM 학습은 여러 도메인 데이터를 섞어 성능을 넓히지만, 도메인 간 음의 전이가 발생해 원래 강점(기존 전문성)을 깎는 문제가 자주 보고된다. 이를 줄이기 위해 기울기/어댑터/학습 스케줄을 조절하거나, 여러 교사를 한 모델에 일방향으로 증류하는 방식이 제안됐다. 다만 단일 모델 중심의 미세조정이나 one-way distillation은 상호 보완이 충분히 반영되기 어렵고, 신뢰할 수 없는 피드백이 섞일 여지도 남는다.

- **Core Contribution**: 이 논문은 두 모델이 서로의 약점을 메우도록 공진화하는 “상호 파레토 개선”을 목표로 하는 On-Policy Co-Distillation(OPCoD)을 제안한다. 각 모델은 자기 롤아웃의 정확한 결과와 더불어, 상대 모델로부터 받은 자연어 피드백을 자가 증류의 특권 정보로 사용해 두 도메인 모두에서 성능을 끌어올리되 원래 강점을 유지하도록 설계된다. 핵심 아이디어는 일방향 증류가 아니라 양방향(peer-to-peer) 피드백으로 학습 신호의 보완성을 확보하는 것이다.

- **Technical Challenges**: 양방향 피드백은 틀린 조언이 섞이면 학습 신호를 오염시켜 오답 롤아웃을 “맞는 것처럼” 만들 수 있다는 기술적 난점이 있다. OPCoD는 이를 줄이기 위해 (1) 인지도 기반 게이팅(cognizance-based gating)으로 튜터의 신뢰도가 충분할 때만 피드백을 교환하고, 부족하면 피드백을 끊어 자가 증류로 폴백한다. 또한 (2) 피드백 앵커링(feedback anchoring)으로 질문에서 특정 개념을 추출·검증한 뒤 답을 직접 드러내지 않는 방식으로 피드백을 접지(grounding)해 잡음·환각성 피드백을 정밀하게 걸러낸다.

- **Empirical Impact**: SciKnowEval의 Science Q&A에서 Physics–Chemistry, Chemistry–Materials, Materials–Physics 같은 도메인 쌍 전반에 대해 OPCoD가 강한 기준선을 꾸준히 이기며, 두 모델이 모두 향상하는 상호 파레토 개선을 달성했다. 분석 결과 게이팅을 적용한 경우가 이전에 정답이었던 롤아웃을 깨뜨리는 비율이 더 낮아, 부정확한 피드백으로 인한 음의 전이를 효과적으로 억제함을 보였다. 또한 앵커링은 “문제와 무관한” 피드백을 소수로 억제하면서도 유효 피드백은 70%대 이상 유지되어, 보완적 추론 단서를 학습에 실제로 반영한다는 점에서 의미가 크다.



### Detecting Historical Turning Points in Italian Media: A Complex Systems Approach to a Diachronic News Corpus (https://arxiv.org/abs/2606.14348)
Comments:
          16 pages, 9 figures, 1 table

- **Prior Approaches**: 기존 역사 분석은 디지털화된 텍스트를 기반으로 토픽 모델링, 단어 임베딩, 감성/개체 인식 등 NLP로 정량 패턴을 찾는 방식이 주로 쓰였다. 다만 프리-디지털 시기의 ‘지속적인’ 다이아크로닉(시계열) 코퍼스는 희소하고, 레이블에 의존하는 경우가 많아 전환 시점을 자동으로 잡아내기 어렵다. 또한 복잡계 관점은 Zipf의 법칙이나 버스트성 같은 언어 통계에 머물러, 매체 담론의 ‘레짐 전환(regime shift)’을 직접 추적하는 데는 한계가 있었다.

- **Core Contribution**: 이 논문은 이탈리아 일간지 La Repubblica의 1985~2000년 전체 기사 약 60만 편을 원본 CD-ROM에서 역공학해 재구성한 다이아크로닉 코퍼스를 제시한다. 이후 어휘 수준(TF-IDF 기반 단어 중요도)과 의미 수준(LSA 기반 담론 벡터)을 동시에 사용해 매체 담론의 전환점과 레짐 변화를 라벨 없이 탐지한다. 특히 제1공화국→제2공화국 이행과 걸프전·코소보전 같은 국제 갈등이 담론에서 어떻게 구조적으로 반영되는지를 정량적으로 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 프리-디지털 자료에서 대규모·완전한 텍스트와 메타데이터를 복원하는 문제와 (2) 비정상적인 언어 시계열에서 ‘중단 없는 변화’와 ‘급격한 전환’을 분리하는 문제였다. 논문은 복원 품질이 높은 원본 아카이브를 역공학해 기사 단위 시간 메타데이터를 확보하고, TF-IDF의 어휘 시계열에는 버스트성(Fano factor)과 변화점 탐지(PELT)를 결합해 전환 구간을 찾는다. 의미 시계열은 TF-IDF를 LSA로 저차원화한 뒤 시간별 담론 벡터의 궤적, 코사인 유사도 블록 구조, 그리고 의미 분산의 스펙트럴 엔트로피로 ‘의제 압축(agenda compression)’을 측정해 해석 가능하게 만들었다.

- **Empirical Impact**: 실험 결과, 단어 중요도에서는 제1공화국 정당명 지표가 감소하고 제2공화국 핵심 인물명이 증가하는 경향이 나타나며, 변화점이 가장 밀집된 시기가 중-1990년대(특히 1994년) 국내 정치 레짐 전환과 일치한다. 의미 수준에서는 담론 벡터의 유사도 행렬이 역사적 경계(국내 정치 변화, 주요 전쟁)를 블록-대각 패턴으로 드러내고, 전쟁 시기에는 스펙트럴 엔트로피가 전 집계 스케일에서 급락해 주제 다양성이 축소됨을 정량적으로 확인한다. 이는 복잡계/통계물리 관점을 대규모 매체 텍스트에 연결해, ‘사건이 담론을 어떻게 재구성하는가’를 라벨 없이 대규모로 추적할 수 있음을 보여준다는 점에서 역사정보과학·미디어 분석 분야에 의미 있는 방법론적 기여로 평가된다.



### ScoreGate: Adaptive Chunk Selection for Retrieval-Augmented Generation via Dual-Score Statistical Fusion (https://arxiv.org/abs/2606.14269)
Comments:
          20 pages, 6 figures, 14 tables

- **Prior Approaches**: 고정 길이 top-K 기반 RAG은 쿼리 복잡도와 무관하게 동일 개수의 청크를 생성기에 주입해 과잉 검색(좁은 질의)과 과소 검색(합성형 질의) 문제를 만든다. 후보군에서 순위를 바꾸는 재랭커나 LLM 기반 필터는 정밀도를 개선할 수 있지만, 재랭킹된 집합 자체의 ‘개수’ 선택을 본질적으로 다루지 않거나 추가 추론 비용이 발생한다. 또한 단일 점수 임계값은 임베딩 유사도-문맥 적합성의 상충을 충분히 회복하지 못한다.

- **Core Contribution**: ScoreGate는 두 단계 RAG 파이프라인에 추가 모델 호출 없이, 기존 파이프라인이 산출하는 bi-encoder 유사도 s_i와 cross-encoder 재랭커 점수 r_i만으로 ‘적응적 검색 카드inality(보낼 청크 수)’를 결정한다. 특히 cross-encoder가 긍정하지만 bi-encoder가 낮게 평가하는 어휘 불일치(vocabulary mismatch) 실패 모드를 점수공간의 버킷 분할과 비대칭 융합으로 ‘구출’한다. 결과적으로 생성기 입력의 청크 개수를 쿼리마다 조절하면서도 품질 저하 없이 효율을 개선하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 (s_i, r_i) 두 점수가 서로 다른 관점의 증거를 제공하므로, 단일 임계값이나 단순 thresholding으로는 대립 신호를 처리하기 어렵다는 점이다. ScoreGate는 점수공간을 4개 버킷으로 나눠 일치 영역은 결정적으로 보장하고, 불일치 영역은 r_i에 더 큰 가중치를 둔 비선형적(사실상 비대칭 선형) 융합 점수로 통합해 보유 임계값을 버킷별로 다르게 적용한다. 또한 쿼리별 점수 분포 차이를 고려해 per-query min–max 정규화를 사용하고, MAX-KK 상한으로 컨텍스트 초과를 안전장치로 막는다.

- **Empirical Impact**: MS MARCO에서는 ScoreGate가 MRR@10=0.401을 달성하면서 Standard Top-K(고정 10개) 대비 보유 청크를 약 35% 줄였다. 내부 벤치마크(300개 라벨)에서는 95% 신뢰구간 기준으로 유사양성 0건(정밀도 96.4%~100%)을 관찰하며 재현율 97.77%~99.34%를 유지했고, 쿼리당 토큰을 평균 34.8% 절감했으며 추가 지연은 31ms 수준에 그쳤다. 실제 프로덕션 트래픽에서도 품질을 해치지 않으면서 검색 카드inality를 동적으로 줄일 수 있음을 시사해, 효율-정확도 트레이드오프를 실용적으로 개선할 여지를 제공한다.



### A Multi-Domain Feature Fusion Framework for Generalizable Deepfake Detection Across Different Generators (https://arxiv.org/abs/2606.14230)
- **Prior Approaches**: 기존 딥페이크 탐지는 생성 과정에서 남는 흔적(공간·그래디언트·주파수)을 ‘단서 기반’으로 쓰거나, 대규모 데이터로 ‘데이터 기반’ 특징을 학습하는 방식으로 크게 나뉩니다. 다만 많은 방법이 단일 표현 도메인(주로 공간 또는 주파수)에 치우쳐 있어, GAN에서 잘 되더라도 확산 모델(diffusion)이나 다른 제너레이터로 갈수록 성능이 급격히 떨어지는 경향이 있습니다. 또한 제너레이터/패러다임이 바뀌면 주파수·공간 단서가 약해지거나, 동일 계열 내부에서 학습된 편향이 일반화에 방해가 된다는 점이 반복적으로 지적됩니다.

- **Core Contribution**: 이 논문은 공간, 그래디언트, DWT 기반 주파수 표현을 함께 쓰는 다중 도메인 딥페이크 탐지 프레임워크 SGFF-Net(Spatial-Gradient-Frequency Fusion Network)을 제안합니다. SGFF-Net은 Dual Residual Learning 구조 안에서 두 잔차 학습 경로로 서로 다른 도메인의 포렌식 단서를 융합해, GAN과 확산 모델을 모두 아우르는 제너레이터-비의존적(generator-agnostic) 표현을 학습하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 생성 패러다임에서 공유되는 ‘일반화 가능한 단서’만을 효과적으로 모으는 동시에, 단일 도메인 특징이 만들 수 있는 분포 이동(distribution shift) 취약성을 줄이는 것입니다. 저자들은 RGB(공간)뿐 아니라 그래디언트 맵을 생성하는 사전 변환 모델과 DWT 기반 다중 해상도 주파수 큐(High-High 성분)를 함께 사용하고, Dual Residual Learning을 통해 보완적 정보를 단계적으로 결합하는 방식으로 이를 해결합니다. 여기에 멀티소스 학습과 데이터 증강을 체계적으로 추가해 크로스 제너레이터/크로스 패러다임 강건성을 더 끌어올립니다.

- **Empirical Impact**: 실험에서 SGFF-Net은 동일 데이터셋 평가(intra-dataset) 정확도 98.95%를 달성하며, 크로스 모델 70.46%, 크로스 패러다임 69.94%로 단일 도메인 기반 대비 일반화 성능을 개선합니다. 특히 멀티소스 학습과 증강을 적용하면 크로스 모델 정확도가 79.80%로, 크로스 패러다임은 약 78%로, 실제 환경(real-world) 데이터에서는 61.50%에서 75.80%로 크게 상승해 실사용 강건성을 보여줍니다. 결과적으로 공간·그래디언트·웨이브릿 주파수의 상호보완적 포렌식 신호를 결합하는 전략이, unseen 제너레이터와 실제 데이터 전이에서 효과적이라는 실증적 근거를 제공합니다.



### Graph-based Target Back-Propagation for Context Adaptation in Multi-LLM Agentic Systems (https://arxiv.org/abs/2606.14155)
- **Prior Approaches**: 기존 에이전트형 LLM 시스템의 컨텍스트 적응은 보통 사람이 설계한 프롬프트/지침을 피드백으로 자동 수정하는 방식에 기반한다. 하지만 멀티 LLM 모듈에서는 최종 실패의 원인을 어떤 모듈·프롬프트에 돌려야 하는지(크레딧 할당)가 불명확해지며, 특히 궤적 전체를 LLM이 ‘반성’해 추정하는 방식은 업데이트 대상이 흔들릴 수 있다. 또한 그래프 구조로 텍스트 신호를 전파하는 접근들은 대체로 휴리스틱에 머물고, 프롬프트 업데이트의 수렴 성질이 이론적으로 충분히 정리되지 않았다.

- **Core Contribution**: 이 논문은 멀티 LLM 에이전트 워크플로를 DAG(방향 비순환 그래프)로 보고, 최종 출력의 목표(target)를 그래프를 따라 국소적으로 역전파해 각 모듈 프롬프트를 갱신하는 GTBP(Graph-based Target Back-Propagation)를 제안한다. 구체적으로 다운스트림 목표와의 불일치(타깃-출력 차이)를 이용해 스테이지별 업데이트 신호를 만들며, 모델 가중치는 고정한 채 프롬프트만 조정한다. 기존의 “어떤 이유로” 수정해야 하는지 모호했던 문제를 그래프 기반 타깃 전파로 구조화한다.

- **Technical Challenges**: 주요 난제는 그래프 역전파에서 각 모듈의 ‘국소 타깃 입력/출력’을 명확히 계산해야 하지만, LLM 모듈이 블랙박스이고 텍스트 단위라 미분 기반 최적화가 어렵다는 점이다. GTBP는 이를 위해 LLM-유도 백워드 연산자(backward operator)로 국소 타깃 입력을 근사 생성하고, 이후 프롬프트는 ‘주장(claim) 목록’ 형태로 편집 예산 안에서 단계적으로 수정하는 LLM 옵티마이저를 사용한다. 또한 업데이트가 반복되며 안정화되는지와 전체 목적함수 감소 가능성을 단순화 가정 하에 이론적으로 보이며, 스테이지별 변화가 점차 작아지고 충분한 최적화 능력을 가진 LLM이면 목적이 감소함을 논증한다.

- **Empirical Impact**: GTBP는 SubPOP, HotpotQA, LiveBench-Math 3개 벤치마크에서 강력한 기준선들을 일관되게 능가했으며, 계산 비용은 큰 폭으로 늘리지 않는다고 보고한다. 이는 멀티 모듈 에이전트의 크레딧 할당을 ‘텍스트 반성’ 대신 타깃 역전파로 정렬할 때 성능 향상이 현실적으로 나타난다는 신호로 해석된다. 특히 수렴·감소 성질에 대한 이론 분석이 함께 제시되어, 향후 컨텍스트 적응 연구에서 “업데이트 경로의 보장”을 요구하는 방향에 기여할 가능성이 크다.



### Small LLMs: Pruning vs. Training from Scratch (https://arxiv.org/abs/2606.14150)
Comments:
          Our code is available at this https URL

- **Prior Approaches**: 기존 LLM pruning은 “압축” 관점에서 작은 모델이 큰 모델의 성능을 따라갈 수 있는지에 초점을 맞췄습니다. 특히 채널·레이어 단위의 구조적 가지치기와 가중치 마스킹 기반의 희소 가지치기(예: Wanda, SparseGPT)가 활발히 연구됐지만, 큰 모델 사전학습 비용을 포함한 “토큰 공정 비교”에서 어떤 방식이 지속적으로 이득을 주는지는 불명확했습니다.

- **Core Contribution**: 이 논문은 pruning을 성능 유지용 압축이 아니라 “초기화 전략”으로 재정의하고, 가지친 모델(Pipeline)과 랜덤 초기화 학습(Scratch)을 토큰 예산 기준으로 엄밀 비교합니다. 특히 (1) 동일한 재학습 토큰 예산에서는 가지친 초기화가 일관되게 우수하고, (2) 전체 파이프라인 토큰 예산을 공정하게 더 주면 구조적 가지치기는 따라잡히지만 희소 가지치기는 여전히 우위를 유지하는 식으로 결론을 정리합니다.

- **Technical Challenges**: 핵심 난제는 공정성입니다: 같은 “재학습 토큰”에서 초기화 효과만 보이게 하거나, “전체 토큰”까지 포함해 비용을 상쇄했을 때도 이득이 남는지 확인해야 합니다. 이를 위해 Llama-3.1-8B에 대해 다양한 가지치기 비율(0.5~0.8)과 세부 단위(깊이/폭/혼합/2:4 희소/비구조 희소)를 체계적으로 적용하고, 학습률 스윕과 데이터 분할(서로 겹치지 않는 200B/50B 토큰 집합)로 토큰-공정 비교를 구현했습니다.

- **Empirical Impact**: 실험 결과, 동일 재학습 토큰 예산에서는 pruning 초기화가 랜덤 초기화 대비 평균적으로 더 높은 정확도·낮은 혼란도를 보이며, 가지치기 비율이 커질수록 그 격차는 줄어듭니다. 전체 토큰 예산에서는 구조적 pruning이 제한적으로 유리함을 잃어 “더 오래 학습한 scratch”가 근접 또는 추월할 수 있지만, 비구조(가중치 수준) 희소 pruning은 추가 토큰만으로 복구되지 않는 지식을 전달한다는 신호를 보여 실무 의사결정 규칙(토큰 병목이 무엇인지에 따라 pruning 전략 선택)을 제시합니다.



### Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources (https://arxiv.org/abs/2606.14141)
- **Prior Approaches**: 기존 오디오-언어 모델은 음원 장면을 주로 클립의 전역 의미(라벨/캡션)로 취급해, 공간과 시간에 따라 변하는 상태 추적이 약한 편입니다. 반대로 음원 위치추적·SELD 계열은 시간에 따른 방향을 잘 맞추지만, 제한된 이벤트 분류체계 때문에 언어 기반의 폭넓은 의미 추론과 결합하기가 어렵습니다. 또한 많은 공간 오디오 QA 벤치마크가 정적인 소스에 집중해 동적인 공간 추론은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 음향 이벤트를 ‘무엇이 들리는가’뿐 아니라 ‘시간에 따른 어디에 있는가(방향·거리)’, ‘움직이는가’, ‘다른 소스와 어떻게 관계되는가’로 묶어 평가하는 ST-AudioQA를 제안합니다. FOA(First-Order Ambisonics) 렌더링 기반의 제어형 장면 메타데이터로, 소스 정체성과 궤적을 촘촘히 감독하고 QA를 구조화합니다. 이를 바탕으로 ST-Audio Encoder와 LLM 연동형 ST-AudioLM을 만들어 시간해상 궤적 정보를 언어 모델 입력 토큰으로 연결합니다.

- **Technical Challenges**: 핵심 난제는 ‘이벤트 의미’와 ‘시간에 따른 공간 상태(방향·거리)’를 동시에 학습하되, LLM이 질문에 맞춰 시간 고정(앵커된) 상태를 읽어낼 수 있는 표현을 만드는 것입니다. 이를 위해 ST-Audio Encoder는 time-resolved FOA 인코더로서 시점별(40개 빈) 활동·방향·거리 궤적을 예측하도록 설계하고, 의미 토큰과 궤적 토큰을 분리해 41개 오디오 토큰으로 구성합니다. 또한 ST-AudioLM은 이 토큰을 LLM에 전달하고 LoRA·커넥터만 학습하는 방식으로 단일/혼합/조합형 QA 커리큘럼(A→B→C)을 점진적으로 적용합니다.

- **Empirical Impact**: 실험 결과 ST-Audio Encoder는 동적 장면에서 잡음/기하 변화 없이도 이벤트 의미 유지와 동적 로컬라이제이션의 균형을 더 잘 이뤄, 정적 공간 인코더의 시간 슬라이딩만으로는 부족함을 보여줍니다. ST-AudioLM은 단일 소스 인지, 두 소스 그라운딩, 소스-시간-공간의 조합 추론 전반에서 정적 기반·로컬라이제이션 중심 베이스라인보다 강한 추론 성능을 보였습니다. 다만 소스 간 ‘관계’처럼 동작 조건을 포함한 세밀한 관계 추론은 여전히 어려워 추가 연구 여지가 남습니다.



### CoRe: A Continuously Reward-Finetuned LLM Query Rewriter for Multi-Stage Context-Aware Relevance in Web-Scale Video Search (https://arxiv.org/abs/2606.14127)
Comments:
          12 pages, 3 figures

- **Prior Approaches**: 기존 LLM 기반 쿼리 리라이터는 Query2Doc류의 의사 문서 확장, RL 기반 하류 신호 최적화, 또는 일부 상업 배포에서 SFT+대조학습 형태로 “문맥”을 반영했습니다. 그러나 운영 환경의 소비(리랭커 퓨전)와 훈련 보상 간 시뮬레이션-프로덕션 간극이 생기기 쉬워 오프라인 지표 기반 보상 대체가 성능/안정성을 제한했습니다.

- **Core Contribution**: 이 논문은 CoRe(Context Relevance)를 통해 “마지막으로 본 피드 문서”와 세션 내 후행 행동(엄격 클릭·스킵)을 결합해 문맥 인지 리라이터를 학습합니다. 핵심은 배포된 멀티모달 리렐런스 모델 점수로 보상을 만들고, 그 보상 형태를 프로덕션 랭킹 퓨전 대수와 곱셈 비율로 맞춰 간극을 줄인다는 점입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 정답 라벨 부재로 인한 선호(Preference) 유도와 (2) 주간 단위 연속 재배포에 맞춰 보상 계산·학습 비용을 감당해야 한다는 점입니다. 저자들은 DPO 스타일의 쌍대 목적을 상위 k/하위 k 궤적 부분집합에만 적용하는 semi-online Mixed Preference Optimization 루프와, 보상 호출 타이밍을 위상(phase) 단위로 분리하는 구조로 비용을 절감했습니다. 또한 멀티모달 점수 기반 보상에 더해 길이/종료 패널티와 콘텐츠 안전 필터링을 넣고, 보상 유사 지표만으로는 놓칠 수 있는 reward hacking을 안정성 지표로 탐지·회복하는 자동 승급 게이트를 구축했습니다.

- **Empirical Impact**: CoRe는 주요 쇼트비디오 검색 엔진에서 5개월 이상 주간 재배포되었고, 리라이터 출력은 recall/rawrank/finerank의 병렬 경로 신호로 사용해 기존 신호를 대체하지 않았습니다. 두 번의 순차적 A/B 런칭에서 리라이트 영향 쿼리의 change-query rate가 유의하게 감소했으며, 핵심 관련성·참여 지표도 기대 방향으로 개선되었습니다. 특히 다중 지표 기반 자동 승급 게이트로 후보를 반복 승격·회복한 운영 사례를 주 단위로 공개해, 장기 운영형 LLM 리라이터 연구에 실증적 기준을 제시합니다.



### Simulating Students' Java Programming Errors with Large Language Models (https://arxiv.org/abs/2606.14113)
- **Prior Approaches**: 기존 프로그래밍 교육 연구는 학생 오류를 기반으로 한 학습자 모델링과 맞춤 피드백을 목표로 했지만, 특히 논리 오류(logical errors)를 ‘대표성 있게’ 모으는 데는 시간이 오래 걸리고 비용도 커서 신규 과제에 즉시 적용하기 어렵다. LLM 기반 시뮬레이션도 있었으나 대체로 정답 생성 중심이거나(오류를 거의 못 만들거나), 임의 버그 삽입처럼 근거 없는 인위적 오류를 만들어 교육적 가치와 논리적 불일치 문제가 남았다. 또한 기존 합성 버그 평가는 전체 오류 분포를 보거나 데이터 증강 용도로 다뤄지는 경우가 많아, 실제 학생이 하는 논리 오류와의 ‘정합성’과 ‘다양성’을 동시에 정량 비교한 연구는 부족했다.

- **Core Contribution**: 이 논문은 LLM이 학생의 논리 오류를 대규모로 시뮬레이션하는 ‘스케일러블 프록시’가 될 수 있는지, 그리고 어떤 조건에서 성능이 달라지는지(모델·프롬프트 전략·과제 난이도)를 체계적으로 평가한다. CodeWorkout(37문제, 74,000+ Java 학생 제출)을 기반으로, 오류의 구조적 다양성(diversity)과 실제 학생 오류와의 정합성(alignment)을 동시에 측정해 모델 선택이 왜 트레이드오프를 갖는지 보여준다. 블라인드 전문가 주석 연구까지 수행해 합성 오류가 기능적으로 실제 학생 오류와 구별되지 않는다는 점을 질적으로 뒷받침한다.

- **Technical Challenges**: 핵심 기술 과제는(1) 생성 오류가 서로 다른 실패 양상을 충분히 넓게 커버하면서도, (2) 학생이 실제로 보이는 방식으로 ‘논리적으로 틀리는’ 정합성을 유지해야 한다는 점이다. 이를 위해 AST를 구성한 뒤 편집 거리(Zhang–Shasha)로 구조적 다양성과 가장 가까운 인간 오류까지의 거리로 정합성을 계산하고, IO(입력-출력), CoT(Chain-of-Thought), Self-Refine(반복 교정) 같은 대표 프롬프트 경로를 비교한다. 또한 작업 난이도 조절을 위해 문제별 학생 제출량을 기준으로 struggling level을 저/중/고로 나누어, 난이도가 다양성과 정합성에 미치는 동시 영향을 분석한다.

- **Empirical Impact**: 실험 결과 모든 LLM은 오류를 다양하게 만들 수 있었지만, 모델과 프롬프트에 따라 인간 오류와의 정합성 격차가 크게 나타났고 Claude Sonnet 4가 다양성과 정합성을 가장 균형 있게 보였다. 반대로 Gemini 2.5 Pro처럼 다양성은 높아도 학생다운 오류와의 정합성은 떨어지거나, GPT-4o처럼 정합성은 가끔 가깝지만 다양성이 제한되는 식의 패턴이 확인됐다. 블라인드 전문가 평가에서는 합성 오류의 출처 판별 오분류(기만률)가 매우 높았고, 합성 오류가 기능적으로 실제 오류와 구별이 어려운 수준임이 검증됐다. 다만 과제 struggling level이 높아질수록 합성 오류는 더 다양해지지만 학생다운 정합성은 약해지는 이중 효과가 관찰되어, 지능형 튜터링·학습 분석에서 합성 오류를 설계/선택할 때 목표(다양성 vs 정합성)를 명확히 해야 함을 시사한다.



### Diffusion-Refined Segmentation and Vision-Language Interpretation for Pediatric Brain Tumor MRI (https://arxiv.org/abs/2606.14072)
- **Prior Approaches**: 소아 뇌종양 분할은 희소한 라벨 데이터, 영상 내 이질성, 경계가 흐린(infiltrative) 종양 특성 때문에 기존 지도학습 모델의 일반화가 어렵다. BraTS-PEDs 2023 같은 벤치마크에서는 nnU-Net/Sw i n-UNETR 계열 앙상블이나 대규모 사전학습이 성능을 끌어올리지만, 연산·추론 비용이 커지고 특히 작은 Enhancing Tumor(ET) 경계에서 불균형 문제가 남는다. 또한 확산 모델은 성능 향상이 보고되었으나, 강한 기준선 대비 경계 개선의 일관성이 성립되지 않거나 소아 데이터에서 충분히 검증되지 못한 한계가 있다.

- **Core Contribution**: 이 논문은 3D 분할 기준 모델(3D Res U-Net과 Swin-UNETR)을 먼저 돌린 뒤, 그 “거친 예측(coarse prior)”을 조건으로 하는 확산 기반 정련(refinement) 모델로 경계를 날카롭게 만드는 2단계 프레임워크를 제안한다. 특히 조건부 3D DDPM refiner와 MedSegDiff를 도입해 확산 과정의 안정성을 높였고, ET 경계 품질 향상에 초점을 둔다. 마지막으로 분할 결과의 정량 볼륨과 대표 시각화를 멀티모달 언어 모델에 결합해 방사선학 스타일의 구조화 리포트를 자동 생성한다.

- **Technical Challenges**: 소아 종양은 ET 같은 소표적이 전체 볼륨의 극히 일부를 차지해 클래스 불균형이 심하며, 경계가 모호해 확률적 생성 과정이 잡음을 학습해 붕괴할 위험이 있다. 저자들은 이 문제를 (1) 거친 Swin-UNETR 예측을 조건으로 확산 입력에 주입하고, (2) 경계에 가중치를 둔 학습 목적(경계 강조 손실)과 (3) 불확실성을 다루도록 설계된 확산 정련 구조로 완화한다. 또한 3D 연산의 메모리 병목을 고려해 학습은 패치/슬라이싱 기반으로 수행하고, 지표는 HD95처럼 경계 중심의 평가에 맞춰 설계했다.

- **Empirical Impact**: BraTS-PEDs 2023에서 조건부 확산 모델은 비조건(unconditional) 대비 학습 안정성과 성능이 크게 개선되며, 특히 ET의 Dice가 의미 있게 상승한다. MedSegDiff(조건부)가 가장 낮은 HD95를 기록하며 경계 일치가 우수함을 보여, 확산의 강점이 단순 재현이 아니라 “경계 정밀도 개선”에 있음을 실증한다. 더불어 토머스(ET/TC/WT) 볼륨과 오버레이를 언어 모델 입력으로 연결해 해석 가능한 임상형 리포트 워크플로까지 시연하며, 소아 신경종양 분야에서 end-to-end 보조 AI 활용 가능성을 제시한다.



### Non-Parametric Machine Text Detection via Multi-View Gaussian Processes (https://arxiv.org/abs/2606.14060)
- **Prior Approaches**: 기존 기법은 단일 특징 공간에 의존해 문서가 사람 글인지 여부를 판정한다. 예를 들어 참조 언어모델의 토큰 확률/랭크 정보나 문체 지표, 또는 변조에 민감한 통계 점수를 쓰는데, 그 축을 겨냥한 패러프레이즈나 타깃 스타일 전환 공격이 들어오면 쉽게 성능이 무너진다. 또 분포가 이동(새 공격·새 생성기·새 언어)하면 많은 분류기가 자신 있게 틀리는 경향이 있어 신뢰도·캘리브레이션이 약해진다.

- **Core Contribution**: 이 논문은 한 문서에서 서로 보완적인 K개 관점(문체, 확률/랭크 성격, 구조 통계)을 뽑고, 관점별 근거를 함께 사용해 회피 비용을 크게 만드는 다중-뷰 비모수 탐지 프레임워크를 제안한다. 각 관점은 독립적인 Gaussian process(GP) 분류기로 확률을 산출하고, 마지막에는 소량 캘리브레이션 데이터로 근거를 학습적으로 집계해 최종 결정을 만든다. 또한 GP의 예측 불확실성을 활용해 분포 밖 입력에서 ‘기권(abstention)’에 가까운 신중함을 제공하는 것이 핵심이다.

- **Technical Challenges**: 다중 신호를 쓰더라도, 단순 결합(평균/최대)은 관점별 캘리브레이션 차이와 한 관점의 오판이 전체를 흔드는 문제를 만든다. 또한 고차원 임베딩을 그대로 GP 커널에 넣으면 거리 값이 수렴해 판별력이 약해져(커널이 퇴화) 성능이 떨어질 수 있다. 논문은 이를 위해 관점별 특징을 클래스 중심과의 거리(유클리드·대각 마할라노비스)로 저차원화하고, 관점별 GP의 예측 불확실성을 반영한 확률을 만든 뒤, 캘리브레이션 세트로 L2 정규화 로지스틱 회귀 집계를 학습해 확률 정합성과 OOD 기권을 동시에 노린다.

- **Empirical Impact**: DetectRL·RAID·PAN 2025의 여러 벤치마크에서, 학습에 포함되지 않은 공격(held-out)에도 강건하게 유지되는 성능을 보이며 단일-뷰 기준 및 다수 기존 제로샷 탐지기보다 우세함을 보인다. 특히 AUROC@1%처럼 낮은 오탐률 운영 조건에서 취약성이 드러나는 기존 모델들과 달리, 제안 방법은 확률(브리어 스코어)과 캘리브레이션(ECE) 측면에서도 더 안정적인 경향을 보인다. 또한 불확실성을 기반으로 한 성능-커버리지 곡선과 far-OOD(예: 스타일이 크게 다른 아랍어 뉴스) 실험에서, 불확실성이 실제로 증가하며 ‘과신한 오탐’ 위험을 줄일 수 있음을 실증한다.



### Knowledge Graph Enhanced Memory-Augmented Retrieval for Long Context Modeling (https://arxiv.org/abs/2606.14047)
- **Prior Approaches**: 긴 문맥 언어모델은 토큰 수를 늘리는 방식(확장 attention, long-context LLM)과 외부 메모리 검색(메모리-증강, RAG 계열)로 발전해 왔습니다. 하지만 attention은 ‘lost-in-the-middle’로 중간 구간 정보가 비관련 정보처럼 처리되는 문제가 있고, 검색 기반 방법은 관련 항목을 ‘관계’가 아닌 ‘의미적 유사도’로만 고르기 쉬워 동일 개체의 상태 변화/인과 흐름을 놓칩니다. 또한 지식그래프 접근은 ConceptNet·Freebase 같은 고정 그래프에 의존하는 경우가 많아 세션/도메인에만 나타나는 고유 개체와 관계를 실시간으로 반영하기 어렵습니다.

- **Core Contribution**: KGERMAR는 추론(inference) 중에 입력 텍스트로부터 동적이고 문맥별 지식그래프(개체·관계)를 구성해, 의미 유사도 검색에 ‘명시적 엔터티 관계’ 신호를 결합합니다. 이를 위해 그래프 구조 임베딩과 텍스트 의미 임베딩을 함께 학습·융합하고, 기존 메모리-증강 구조를 확장해 관계 중심 검색이 가능하게 합니다. 결과적으로 수천 토큰 떨어진 과거 맥락에서도 ‘같은 개체의 인과/상태 진행’에 맞는 대상을 더 잘 찾아 일관된 장문 추론을 돕습니다.

- **Technical Challenges**: 핵심 난점은 (1) 긴 문맥에서 관계를 정확히 뽑아내야 하지만 관계추출은 로컬 윈도 제약을 받는다는 점, (2) 동적 그래프 품질 저하가 검색 성능으로 바로 이어질 수 있다는 점, (3) 그래프 기반 신호와 텍스트 의미 신호를 충돌 없이 효과적으로 결합해야 한다는 점입니다. 논문은 NER·관계추출(BERT 계열)로 문맥 그래프를 만들고, 개체 통합·관계 신뢰도 점수·그래프 필터링으로 잡음을 줄였습니다. 또한 R-GCN으로 관계 기반 전파(다중 hop)를 수행하고, contextual/semantic/structural의 3개 메모리 은행 검색 신호를 가중치 학습으로 융합해 그래프 구조와 텍스트 의미를 함께 주입합니다.

- **Empirical Impact**: SlimPajama, WikiText-103, PG-19, Proof-pile의 다양한 도메인에서 문맥 길이 1K~32K 전 구간으로 평가했으며, 기존 memory-augmented baseline 대비 최대 8.5% 낮은 perplexity와 2~2.5배 더 나은 메모리 효율을 보고합니다. 특히 구조적 관계 기반 검색이 강화되면서 NLU 태스크 전반에서 in-context learning 성능도 더 좋게 나타났습니다. 동적 지식그래프를 추론 시점에 구성해 도메인 특화 표현을 만든다는 점에서, 고정 지식베이스 의존을 줄이고 장문 추론의 ‘구조적 관련성’ 문제에 실질적 돌파구를 제시한 연구로 평가됩니다.



### Efficiency-Performance Trade-offs in Neural Speaker Diarization via Structured Pruning and Low-Bit Quantization (https://arxiv.org/abs/2606.14030)
Comments:
          6 pages, 3 figures, preprint

- **Prior Approaches**: 기존 화자 분할(speaker diarization)은 배치 처리 또는 온라인 처리 방식으로 나뉘며, 지연 한도(look-ahead·버퍼링)와 청크 길이를 스윕해 정확도-지연(trade-off)을 계량하려는 시도가 있어 왔다. 다만 의료 상담/긴급 상황처럼 시간 제약이 강한 도메인에서, 고정된 파이프라인 아래 실질적인 스트리밍 지연 예산별로 DER이 어떻게 흔들리는지 end-to-end로 체계 분석한 연구는 드물다. 또한 pruning과 저비트 양자화는 효율 개선 수단으로 연구됐지만, 분할 파이프라인에서는 압축이 정확도와 실시간 처리율(RTF)에 미치는 영향이 덜 문서화돼 있었다.

- **Core Contribution**: 이 논문은 SIMSAMU(시뮬레이션 기반 의료 긴급 콜 대화)에서 스트리밍 지연 예산을 변수로 두고, segmentation 모델 압축(pruning·저비트 quantization)이 성능에 어떤 상쇄관계를 만드는지 정량화한다. 특히 “추가 버퍼가 항상 이득이냐”를 점검하고, 매우 낮은 지연 운용점이 성능을 크게 해칠 수 있음을 보여준다. 더불어 pruning 구조 선택이 결과를 좌우하며, pruning 후 양자화로 segmentation 모델 크기를 절반 수준까지 줄이면서도 기준 대비 DER이 약 40% 상대 증가하는 타협점을 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 스트리밍에서 버퍼링/청크 길이 같은 지연 제어 변수를 정확도에 연결해 비교하고, (2) 압축이 segmentation–embedding–클러스터링으로 이어지는 전체 파이프라인에 어떻게 전달되는지 분리해 해석하는 것이다. 저자들은 SIMSAMU에서 제공하는 전처리·평가 프로토콜을 그대로 유지한 채, segmentation 모델만 구조적 pruning( BiLSTM hidden 단위 vs 선형 채널)과 PTQ/QAT 방식의 저비트 양자화를 적용해 DER과 RTF를 동시에 측정한다. 또한 정량 비교를 위해 FP32·FP16·INT8/INT4(W4A8 등)와 각 양자화 모드의 안정성(기록별 ΔDER 분포)까지 함께 관찰한다.

- **Empirical Impact**: 실험 결과, right-context(추가 look-ahead)는 작은~중간 구간에서는 비교적 안정적이지만 큰 값에서는 성능이 흔들리며, dispatch의 화자 전환 경계가 흐려지는 현상과 연결된다. chunk length는 아주 짧을 때만 급격한 오차 변화를 보이고, 일정 구간 이후에는 완만해져 운영 파라미터 선택의 실무 신호를 제공한다. pruning은 모델 크기를 줄여도 end-to-end RTF 개선이 거의 없었고(대략 동일 수준), FP16은 모델 크기를 절반으로 줄이면서 real-time factor는 유사하지만 DER은 기준 대비 상대적으로 약 40% 증가했다는 “배포형 타협점”을 제시한다.



### Same-Origin Policy for Agentic Browsers (https://arxiv.org/abs/2606.14027)
- **Prior Approaches**: 기존 에이전틱 브라우저 연구는 자연어 지시로 웹 작업을 자동화하는 데 초점을 맞췄지만, 브라우저 보안의 핵심인 동일 출처 정책(SOP)이 실제로도 제대로 지켜지는지에 대한 체계적 평가는 부족했다. 또한 SOP 위반은 주로 스크립트가 유발하는 교차 출처 데이터 흐름 관점에서 논의돼 왔지만, 에이전트가 브라우저를 “자체 채널”로 활용할 때의 위험은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 에이전틱 브라우저가 교차 출처 데이터 흐름을 자동화된 채널로 만들어 SOP를 우회할 수 있음을 관찰하고, 이를 정량 평가할 벤치마크 SOPBench를 제안한다. 나아가 SOPGuard라는 에이전틱 브라우저 전용 SOP 집행 메커니즘을 설계해 BrowserOS에 구현하고, 실제 위반을 막는 방향으로 문제를 해결한다.

- **Technical Challenges**: 핵심 난제는 에이전틱 브라우저의 자율 실행 흐름 속에서 SOP 위반이 어떤 경로로 발생하는지 측정·분류하고, 이를 막되 에이전트의 웹 작업 유용성을 크게 훼손하지 않는 집행을 만드는 것이다. 연구진은 SOPBench로 위반 유형을 체계적으로 평가하고, SOPGuard로 교차 출처 데이터 흐름을 집행 수준에서 제어하도록 BrowserOS에 통합했으며, 그 결과 런타임 오버헤드는 작게 유지되도록 최적화했다.

- **Empirical Impact**: 실험 결과, 기존 에이전틱 브라우저들은 정상 환경에서도 공격 환경에서도 SOP를 자주 위반했으며, 이는 SOP가 자동화된 에이전틱 환경에서 그대로 “보장되는 가정”이 성립하지 않을 수 있음을 보여준다. SOPGuard 적용 후에는 SOP 집행이 효과적으로 이뤄지면서도 유용성 손실이 제한적이었고, 소규모 런타임 오버헤드만 발생해 실사용 관점의 실현 가능성도 확인됐다.



### ADORE: Iterative Query Expansion with Retrieval-Grounded Relevance Feedback (https://arxiv.org/abs/2606.13905)
- **Prior Approaches**: 기존 LLM 기반 질의 확장/재구성은 그럴듯한 확장 텍스트를 생성해 검색 성능을 높이려는 방식이 주류였습니다. 다만 생성된 확장이 실제로 목표 코퍼스에서 어떤 순위를 만들었는지(잡음/드리프트 여부)를 진단하지 못해, 언어적 적합성과 검색 효용이 어긋날 수 있습니다.

- **Core Contribution**: ADORE는 확장 생성과 평가(판단)를 분리하고, 검색 결과를 다음 라운드 확장에 피드백으로 다시 주입하는 “검색 근거형” 반복 프레임워크를 제안합니다. 핵심은 검색에서 드러난 관련 증거를 기준 질의에 대해 등급(relevance tier)으로 평가해, 보강할 신호와 억제할 신호를 구조화된 형태로 다음 생성에 반영한다는 점입니다.

- **Technical Challenges**: 기여를 실현하려면 (1) 코퍼스가 어떻게 반응하는지 관측하고 (2) 그 반응을 기준 질의에 고정된 방식으로 판단해 드리프트를 막는 체계가 필요합니다. ADORE는 라운드별로 가짜 패시지(pseudo-passages)를 생성-관측(재검색)한 뒤, 새로 검색된 문서에 대해 LLM 기반 relevance assessor가 0–3 등급의 피드백을 부여하고, 적응형 종료(품질·커버리지 포화 또는 예산 소진)로 불필요한 반복을 줄였습니다.

- **Empirical Impact**: TREC Deep Learning, BEIR, BRIGHT 전 범위에서 ADORE는 8개 설정 중 7개에서 기존 강력한 기준선을 앞섰고, 평균 nDCG@10을 BEIR에서 BM25 대비 24.5%, 기존 최강 질의 확장 대비 3.6% 개선했습니다. BRIGHT에서는 BM25 대비 122.9%, 최강 질의 확장 대비 9.2% 향상이었으며, 특히 추론형 도메인에서 BM25 기반임에도 조밀 검색기·LLM reranker를 크게 초과하는 결과가 보고되었습니다.



### Gefen: Optimized Stochastic Optimizer (https://arxiv.org/abs/2606.13894)
- **Prior Approaches**: AdamW는 1·2차 모멘트 이동평균을 저장해 성능이 안정적이지만, 옵티마이저 상태만으로도 파라미터 메모리에 비례한 큰 버퍼가 추가된다. 기존 메모리 절감 연구로는 Adam-mini(수동 규칙 기반 공유), Adam8bit/Adam4bit(모멘트 양자화) 등이 있으나, Hessian 정렬의 이론적 근거가 약하거나 텐서 이름/아키텍처 정보 같은 수동 설정이 필요하며, 고정 블록 크기 같은 암묵적 하이퍼파라미터 부담도 남는다. 또한 양자화 방식이 플러그-앤-플레이로 항상 적용되기 어려운 제약이 있다.

- **Core Contribution**: Gefen은 AdamW의 성능을 유지하면서도 2차 모멘트 상태를 파라미터 블록 간에 자동 공유하고, 1차 모멘트를 학습된 코드북으로 양자화해 옵티마이저 메모리를 약 8배 줄이는 것을 목표로 한다. 핵심 아이디어는 큰 혼합 Hessian 항이 두 파라미터의 제곱 그라디언트 비율을 1에 가깝게 만든다는 이론에 기반해, Hessian 관련성이 큰 파라미터끼리 같은 2차 모멘트 추정을 쓰는 것이 자연스럽다는 점이다. Hessian을 직접 계산하지 않고도 이를 구현할 수 있게, 학습 초깃값의 제곱 그라디언트로 블록 구조를 자동 추론한다.

- **Technical Challenges**: 문제는 Hessian을 대규모에서 직접 계산하는 것이 불가능하다는 점인데, Gefen은 대신 첫 스텝의 제곱 그라디언트만으로 텐서 내 블록 분할 후보를 나눠 보고, 블록 내 이질성이 작아지는 “첫 뚜렷한 개선 지점”의 주기를 선택해 공유 블록을 만든다. 이어서 1차 모멘트 양자화를 위해 기존처럼 수동/고정 블록 크기와 경험적 양자화를 쓰지 않고, 앞서 얻은 블록 분할을 그대로 재사용하는 “정확한 히스토그램 기반 동적 계획법” 코드북 학습을 제안한다. 또한 Lloyd-Max 계열 양자화의 범위 수축(또는 초기화 민감도) 문제를 피하기 위해 코드북에 극단값을 강제 포함하는 등 수렴·일관성을 확보한다.

- **Empirical Impact**: 실험에서 Gefen은 비교한 AdamW 계열 메모리 절감 방법들 중 옵티마이저의 피크 메모리를 가장 크게 낮추면서도 AdamW 수준의 성능을 유지한다. 분산 학습에서는 FSDP에서 AdamW 대비 처리량(throughput)을 56% 개선했고, DDP에서는 AdamW가 마이크로배치 1조차 담지 못하는 조건에서 Gefen이 마이크로배치 2를 가능하게 하며 Adam-mini 대비 21% 처리량 향상을 보였다. 즉, 단순 대체(drop-in replacement)로도 큰 모델·더 큰 배치·더 큰 마이크로배치를 현실적으로 열어 훈련 효율을 끌어올릴 수 있다는 점에서 실용적 파급력이 크다.



### Natively Unlearnable Large Language Models (https://arxiv.org/abs/2606.13873)
- **Prior Approaches**: 기존의 unlearning은 학습이 끝난 뒤 모델을 수정하는 사후(post-hoc) 방식이 많다. 하지만 공유 가중치 안에서 서로 얽힌 원천(source) 영향 때문에, 표적을 지우는 과정에서 관련 지식이나 일반 능력까지 함께 손상되기 쉽고, 공격적으로 재노출/재학습되면 쉽게 되돌려질 수 있다. 또 다른 접근은 소스마다 별도 모듈·파라미터를 할당해 제거를 쉽게 하지만, 원천 간 공동 학습을 막아 일반화 이점을 희생하며 세밀한 소스(수백만 문서 등)로 확장하기 어렵다.

- **Core Contribution**: 이 논문은 Natively Unlearnable LLMs(NULLs)라는 모델 클래스를 제안해, 학습 단계에서부터 소스별 지우기(unlearning)를 “자연스럽게” 가능하게 만든다. 핵심은 공유 백본(neurons)으로 일반 능력을 함께 배우되, 소스별 정보는 희소하게 활성화되는 sink 뉴런 풀에 모이게 설계해 소스 영향의 분리가 일어나도록 한 것이다. 배포 시에는 해당 소스의 sink를 끄기만 해도(가중치 업데이트 없이) 지우기 동작을 수행할 수 있다.

- **Technical Challenges**: 가장 큰 기술 난제는 “공동 학습”과 “소스별 분리”가 본질적으로 충돌한다는 점이다. NULLs는 이를 위해 소스 식별자 기반의 결정론적 희소 마스크를 sink 풀에 부여하고, 각 문서(소스)는 마스크에 해당하는 sink와 공통 백본을 함께 활성화하도록 학습한다; 그 결과 소스 전용 정보는 상대적으로 간섭이 적은 sink로 먼저 수렴하고, 백본에는 여러 소스에 공통으로 강화되는 정보만 남는 동역학을 유도한다. 또한 마스크를 제거하면 그대로 지우기가 되므로, 삭제를 위한 추가 그라디언트 업데이트나 잔존 데이터 접근 없이도 동작한다.

- **Empirical Impact**: 실험에서 NULLs는 위키피디아 약 600만 개 문서를 서로 독립 소스로 보고도 확장되며, 한 문서를 unlearn하면 그 문서에만 고유한 지식은 크게 사라지지만 의미적으로 가까운 문서에서 공유되는 사실은 보존되는 것으로 나타났다. Harry Potter 책을 소스 단위로 지우는 사례에서는 생성 품질과 손실/질문응답 지표가 기준(처음부터 재학습)과 가깝게 맞고, GCG 기반 적대적 프롬프트 추출이나 relearning(추가 파인튜닝으로 되돌리기) 공격에도 재학습 동역학이 재학습-없음 모델과 유사하게 유지되어 견고함을 보였다. 마지막으로 다운스트림 벤치마크에서 일반 언어 능력은 표준 트랜스포머와 큰 차이가 없어, 소스 단위 제어를 후처리로 “덧붙이는” 대신 학습 설계에 통합해도 성능을 해치지 않을 수 있음을 시사한다.



### SuperThoughts: Reasoning Tokens in Superposition (https://arxiv.org/abs/2606.13862)
- **Prior Approaches**: 기존 LLM의 Chain-of-Thought(CoT)는 정답으로 가기 전 긴 추론 토큰을 순차 생성해 성능을 높이지만, 그만큼 추론 계산이 비싸다. 이를 줄이려는 잠재(continuous latent) 공간 추론은 중간 표현에 대한 정답(중간 슈퍼비전) 신호가 약해 학습이 불안정해지고, 장기 추론 과제에서 성능이 떨어지는 문제가 반복됐다. 또한 멀티-토큰 예측은 한 번에 여러 토큰을 맞출 수 있어 보이지만, 실제 추론에서는 주 모델이 해당 토큰들의 KV를 채워야 해서 연산 효율이 크게 개선되지 못했다.

- **Core Contribution**: SuperThoughts는 CoT의 연속된 토큰 두 개를 하나의 잠재 표현으로 압축하고, 한 단계에서 두 토큰을 예측해 추론 길이(연산 단계 수)를 줄이도록 설계했다. 학습 단계에서는 여전히 이산 토큰에 대한 교차 엔트로피 슈퍼비전을 유지해 ‘중간 감독 부재’로 인한 학습 불안정을 완화한다. 여기에 MTP(Multi-Token Prediction) 모듈의 확신이 낮을 때는 표준 디코딩으로 되돌리는 적응형 추론을 넣어 정확도 하락을 제어한다.

- **Technical Challenges**: 핵심 난제는 토큰 쌍 압축으로 인해 중간 표현이 사전학습된 언어 분포에서 벗어나 ‘대표(표상) 드리프트’가 생기면 성능이 무너지는 점이다. 이를 해결하기 위해 압축기(Compressor)를 지식 증류로 먼저 정렬한 뒤, Main 모듈과 MTP까지 전체를 이산 토큰 기반 손실로 함께 학습하는 2단계 학습을 수행한다. 또 한 단계에서 두 ‘어려운’ 토큰을 동시에 처리할 때 생길 수 있는 한계를, MTP 예측 확률 기반 임계값으로 감지하고 다음 단계에서 단일 토큰 방식으로 재시도하는 폴백으로 완화한다.

- **Empirical Impact**: Qwen2.5-Math-1.5B/7B/14B에 적용해 MATH500, AMC23, OlympiadBench, GPQA-Diamond에서 평가했으며, CoT 길이를 약 20~35% 줄이면서 정확도는 대부분 1~2점 수준의 소폭 저하로 유지했다. 특히 1.5B에서는 적응형 추론이 MATH500 정확도를 기준선과 거의 동일하게 맞추는 동시에 CoT를 30%대까지 줄였고, 더 큰 모델에서는 전반적으로 정확도 격차를 좁히며 압축 이점을 누릴 수 있음을 보였다. 수학 외 도메인(과학)에서도 유사한 효과가 관찰되어, ‘잠재 공간 추론+이산 중간 감독+적응형 디코딩’ 조합이 장기 추론 효율화에 실용적 대안이 될 가능성을 시사한다.



### Poker Arena: Multi-Axis Profiling of Strategic Reasoning and Memory in LLMs (https://arxiv.org/abs/2606.13815)
Comments:
          33 pages, ICML Workshop

- **Prior Approaches**: 기존 LLM 게임플레이 벤치마크는 다수의 전략 역량을 단일 스칼라 성과(대개 승률/점수)로 뭉개, 모델의 ‘역량 구조’가 어떻게 다른지 파악하기 어려웠습니다. 또한 정형 게임의 방법론(Counterfactual Regret Minimization, self-play 기반 탐색 등)은 학습 시점에 강하게 의존해 추론 고정(frozen-weight) LLM 평가에는 그대로 옮기기 어렵습니다. 메모리 아키텍처 연구도 주로 단일 에이전트/협력 환경에 치우쳐, 경쟁적인 다중 에이전트 상황에서 지속 메모리가 어떻게 작동하는지는 덜 규명됐습니다.

- **Core Contribution**: 이 논문은 no-limit Texas Hold’em 토너먼트 환경 ‘Poker Arena’를 제안해, 전략적 추론 역량을 9개 축(베팅 사이징, 블러핑, 상대 읽기, 침착함, 적응성, 예측, 전략적 믹싱, 사실 정확도, 포지셔닝 인식)으로 분해해 평가합니다. 동시에 3계층 메모리(핸드 내, 세션 내, 세션 간)를 설계해 지속성이 성과에 미치는 영향을 통제된 방식으로 측정합니다. 그 결과 단일 스칼라 리더보드가 모델을 잘못 순위화할 수 있음을, ‘축별 역량 구조’ 관점에서 정리합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 전략 역량을 액션/추론 텍스트에서 측정 가능한 형태로 쪼개고, (2) LLM 추론 고정 상태에서도 메모리의 시간 스케일 효과(세션 간 지속)를 통제해 비교하는 것입니다. 이를 위해 액션 로그 기반의 결정론적 지표와, 이유 텍스트에 의존하는 축(예: 의도적 블러프, 구체적 상대 읽기)은 LLM 판정(다중 패널/편향 완화 포함)으로 점수화했습니다. 또한 베팅 규칙(사이드팟/타임아웃/쇼다운)과 컨텍스트 재구성을 엄밀히 처리하면서, 메모리 계층별로 아블레이션을 수행해 비교 가능성을 확보했습니다.

- **Empirical Impact**: 7개 프런티어 LLM을 50개 세션(각 세션 1,000핸드 규모)에서 비교한 결과, Claude가 칩 누적에서는 1위를 차지했지만 9축 평균 점수에서는 5위에 그쳐 스칼라 리더보드의 오분류가 확인됐습니다. 또한 지속 메모리는 모델마다 효과의 부호가 달라(GPT는 이득, Kimi는 손해, Claude는 대체로 불변) ‘메모리 유무’가 아니라 ‘모델-메모리 인터페이스’가 성과를 좌우함을 보여줍니다. 결론적으로 Poker Arena는 다축 평가가 역량 구조를 드러내며, 기존 단일 점수 순위가 감추는 차이를 체계적으로 복원할 수 있음을 실증합니다.



### WorkBench Revisited: Workplace Agents Two Years On (https://arxiv.org/abs/2606.13715)
Comments:
          8 pages, 3 figures. Follow-up to arXiv:2405.00823

- **Prior Approaches**: 기존 에이전트 벤치마크는 웹 탐색, 일반 보조, 도구 사용 등 주변 문제를 다루거나, LLM 평가자에 의존해 행동을 점수화하는 방식이 많았다. WorkBench는 사무 환경을 샌드박스로 구현하고, 에이전트가 임의의 경로로 작업하되 최종 상태를 정답과 비교해 ‘행동 자체’의 성패를 직접 평가하는 데 초점이 있다. 다만 2024년 출시 당시에는 최고의 에이전트도 작업의 43%만 완료했고, 포맷/툴 호출 실패 같은 요소가 결과를 크게 흔들었다.

- **Core Contribution**: 이 논문은 2024년 WorkBench를 2026년까지 재실행하며 성능과 안전성을 함께 재측정한다. 특히 구조화된 native tool-calling을 일괄 적용해 비교 기준을 공정하게 만들고, 작업 완료율뿐 아니라 의도치 않은 유해 부작용(예: 잘못된 수신자에게 이메일 발송)과 1회 실행 비용까지 2개 축을 추가해 ‘능력-안전-비용’의 동시 지표를 제시한다. 또한 기존 벤치마크의 채점/정답/프롬프트 불일치 및 툴 엔지니어링 문제를 수정해, 2026 점수와 2024 점수를 직접 비교할 때의 함정을 줄였다.

- **Technical Challenges**: 핵심 기술 난제는 에이전트가 샌드박스에서 임의의 경로로 작업할 때, 툴 호출 포맷 실패와 프롬프트-정답 불일치가 성능을 왜곡하지 않게 하는 것이다. 저자들은 ReAct의 텍스트 기반 툴 파싱 대신 공급자들이 제공하는 structured output의 native tool-calling을 사용해 스키마 준수 실패를 제거했고, 모델 추론이 남긴 기본 오류(예: ‘아직 오버듀이 아닌데 오버듀로 간주’ 같은 조건 판단, 캘린더 검색 결과 5개 제한을 고려한 재쿼리 부족)는 그대로 관찰되는 형태로 정리했다. 아울러 “last NN days” 오프바이원, 잘못된 그라운드 트루스, silent-zero 집계 버그, 줄바꿈 이스케이프 같은 엔지니어링 결함을 재생성/재계산해 결과 재현성을 높였다.

- **Empirical Impact**: 재평가 결과, 작업 완료율은 2024년 최상위 GPT-4(43%)에서 2026년 Claude Opus 4.8(88.8%)로 크게 상승했으며, 의도치 않은 유해 부작용 비율도 26%에서 2.5%로 급감했다. 더 나아가 능력과 안전이 상충하기보다 함께 개선되는 경향이 확인되어, 가장 많이 끝내는 모델이 대체로 가장 적게 ‘의도치 않은 피해’를 낸다는 점이 강조된다. 한편 비용은 모델·공급자 간 2자릿수 수준으로 격차가 커서, 오픈 웨이트가 더 저렴한 구간의 효율을 끌어올렸고(캐시 미적용 상한치 기준), 전체적으로 frontier 모델의 절대 효율 격차와 더불어 ‘방출일 단독’으로는 발전량을 설명하기 어렵다는 시사점을 준다. 



### Multimodal Speaker Identification in Classroom Environments (https://arxiv.org/abs/2606.13712)
Comments:
          9 pages, 5 tables, 3 figures

- **Prior Approaches**: 기존 K-12 교실 화자 식별(SID)은 주로 음향 생체인식에 의존했지만, 비정상 잡음과 잔향, 아동 발화의 음향 변동이 커서 또래 대화의 babble noise를 목표 화자로 오인하기 쉽다. 또한 최근 LLM을 활용한 재점수/재추론은 Teacher vs. Student 같은 역할 구분에는 도움을 주지만, 발화가 짧고 비슷한 또래 화자를 ‘개별 학생 단위’로 분리하는 데는 여전히 한계가 있었다.

- **Core Contribution**: 이 논문은 음향 임베딩(ECAPA-TDNN)으로 화자 후보를 만들고, 전사(transcript)에서 LLM이 유도한 의미적 ‘문맥 앵커(contextual anchoring)’를 결합해 식별을 보정하는 멀티모달 프레임워크를 제안한다. 특히 이름 지시나 지칭 같은 문맥을 이용해 다음 발화자의 addressee 가능성을 제약하는 반준지도 추적 관점으로 확장해, 개별 학생 분리에 초점을 둔다.

- **Technical Challenges**: 핵심 기술 문제는 (1) 짧고 잡음 많은 발화에서 음향 임베딩만으로는 서명이 안정적으로 형성되지 않는 점과 (2) LLM 문맥 추론이 후보 화자와 정합되도록 특징을 설계해야 한다는 점이다. 저자들은 교실 시작 시 수집한 음성 enrollments로 3초 창 임베딩을 만들고, 후보별 코사인 거리 통계·동일 그룹 내 일관성·LLM 추론 여부를 포함한 다양한 탭룰형 특징을 XGBoost에 넣어 해결했으며, 세션 단위 leave-one-session-out 교차검증과 확률 보정(Platt scaling)으로 재현성을 확보했다.

- **Empirical Impact**: EDSI 수학 교실 8개 세션, 2,801개 발화에서 음향 단일 기준선은 학생 식별 정확도가 39.0%에 그쳤지만, 제안한 멀티모달 모델은 학생 식별을 50.3%로 끌어올렸다. 역할 구분은 Teacher vs. Student가 99.3%로 거의 완벽했고, 발화 5초 초과 구간에서는 정확도가 76.9%까지 올라(기준선 64.9%) Top-3도 90.9%를 기록했다. 짧은 발화는 Top-1 성능이 낮아도 Top-k(예: Top-5)에서 후보를 효과적으로 좁혀 후속 판별과 자동화된 피드백 시스템의 확장 가능성을 실증했다.



### Orchestra-o1: Omnimodal Agent Orchestration (https://arxiv.org/abs/2606.13707)
- **Prior Approaches**: 기존 에이전트 오케스트레이션은 주로 텍스트 중심이거나 시각-언어처럼 제한된 조합에 맞춰져 있어, 텍스트·이미지·오디오·비디오가 함께 얽힌 오므니모달 환경에서의 일반화가 어렵다. 또한 모듈을 나눠도 실행 흐름이 선형적이거나 휴리스틱에 의존해 복잡한 의존성/병렬성을 효율적으로 다루지 못한다. 네이티브 오므니모달 에이전트는 한 모델이 인식·추론·도구사용까지 동시에 담당하려 하지만, 장기 지평 추론과 도구/크로스모달 정교성에서 한계가 나타난다.

- **Core Contribution**: Orchestra-o1은 오므니모달 에이전트를 위한 오케스트레이션 프레임워크로, 입력 양식(modality)을 고려한 작업 분해와 하위 에이전트의 온라인 전문화, 병렬 서브태스크 실행을 한 구조 안에 묶는다. 메인 에이전트가 고수준 결정(위임·완료)을 하고, 인식/행동은 전용 서브에이전트와 통합 도구 생태계가 담당하도록 ‘역할 분리’를 명확히 한다. 아울러 Orchestra-o1의 메인 에이전트 학습을 위해 DA-GRPO(Decision-aligned group relative policy optimization)라는 오프라인 에이전트형 강화학습 레시피를 제안한다.

- **Technical Challenges**: 핵심 난제는(1) 어떤 입력과 도구가 해당 단계에 필요한지 모달리티 인지형으로 결정하고, (2) 결과 간 의존성을 그래프로 표현해 준비된 작업만 병렬로 스케줄링하는 것이다. 논문은 메인 에이전트가 서브태스크별 요구 벡터(텍스트/이미지/오디오/비디오/코드 등)와 도구 요구를 예측해 모델·도구를 매칭하며, 생성된 의존성 그래프를 기반으로 ready 집합에서 배치를 뽑아 병렬 실행한다. 또한 증거(evidence)를 구조화된 메모리에 압축 저장하며, 충분성 점수가 임계치를 넘을 때 종료하도록 하여 컨텍스트 예산 문제도 함께 다룬다. DA-GRPO는 단계별 오케스트레이션 결정(위임·서브에이전트 선택·도구 사용·생성)을 기준 궤적과 정렬하도록 다차원 루브릭 보상으로 오케스트레이션 의사결정을 학습시킨다.

- **Empirical Impact**: OmniGAIA 벤치마크에서 강한 프로프라이어터리 메인 에이전트와 결합했을 때 Orchestra-o1은 기존 2등 대비 10.3% 정확도 향상을 달성한다. 또한 DA-GRPO로 훈련한 Orchestra-o1-8B는 오픈소스 오므니모달 에이전트들에 대해 최신 수준의 성능을 보이며, OmniGAIA에서 최고 정확도를 20.8%에서 30.0%로 끌어올린다. 구조가 병렬화 가능하고 추론/비용 효율 측면에서도 이점이 있어, 오므니모달 에이전트 스웜 설계에 대한 실용적 기준선을 제시했다.



### Incentives Of EdTech: A Systematic Review Of EduNLP Research (https://arxiv.org/abs/2606.13691)
Comments:
          10 main pages (13 appendix pages), 20 figures, accepted to 21st Workshop on Innovative Use of NLP for Building Educational Applications @ ACL 2026

- **Prior Approaches**: 기존 EdTech·EduNLP 연구는 알고리즘 편향, 개인정보, 투명성, 책임성, 학업 정직성 등 윤리 이슈를 폭넓게 다뤄 왔지만, 실제 설계와 배치에서 일관되게 통합되지는 않는다는 진단이 많았습니다. 또한 작업 중심(예: 자동 채점, GEC) 접근이 강해 데이터셋·벤치마크 성능 향상에 연구 동력이 쏠리면서 교육 현장의 이해관계자 요구가 뒤로 밀릴 수 있다는 우려도 제기되어 왔습니다.

- **Core Contribution**: 본 논문은 2024~2025년 ACL SIGEDU(특히 BEA, NLP4CALL) 및 ACL 본회의 게재물 가운데 총 204편을 대상으로, 작업 우선순위·이해관계자 포함·윤리 리스크 대응을 체계적으로 교차 점검하는 정량-정성 문헌조사를 수행합니다. 이를 ACL Anthology의 더 넓은 EdTech 논문들과 비교 검증함으로써, “누구를 위해 무엇을 우선하는가”라는 구조적 긴장을 데이터로 드러냅니다.

- **Technical Challenges**: 핵심 과제는 방대한 문헌에서 (1) 작업/동기/배치 맥락/이해관계자 포함 수준/암묵적 수혜자/리스크 대응 정도를 동일한 틀로 추출하는 정교한 코딩 스키마를 만드는 일이었습니다. 연구진은 수작업 추출을 위해 공통 추출 스키마를 반복 검증하고, 다중 라벨 및 자유서술 차원에서 일치도를 측정해 해석 편차가 큰 항목(예: 암묵적 인센티브·수혜, 리스크 범주)은 “지표적” 결과로 신중히 다뤘습니다.

- **Empirical Impact**: 결과적으로 교사(연구의 수혜자로서)는 33.3%로 체계적으로 과소대표되며, 실배치(진짜 운영)는 9.8%에 그쳤고, 윤리 대응도 대체로 ‘인정’에 머무는 경향이 확인됐습니다. 특히 자동 채점·피드백 및 GEC 같은 과제가 연구 의제를 크게 지배해, 고액 상용가치가 있는 테스트·교육 인프라(예: 대규모 채점 효율)에 유리하게 의사결정이 기울 수 있음을 시사하며, 교사·학습자 공동설계와 배치 책임성 중심의 연구 관행을 권고합니다.



### Indirect Computing Model with Indirect Formal Method (https://arxiv.org/abs/2606.13690)
Comments:
          10 pages, 6 figures

- **Prior Approaches**: 이 논문은 튜링의 계산가능성 이론, 클린의 문자열 형식이론, 폰 노이만의 디지털 컴퓨터 구조, 튜링의 AI 판단 가설을 배경으로 기존 범용 디지털 컴퓨팅 패러다임을 검토합니다. 그러나 기존 접근은 계산 알고리즘과 데이터 구조 최적화가 분리되거나, 작은 문자열/작은 문자 집합에 맞춘 형식화가 대규모 표현으로 확장될 때 한계가 있다고 지적합니다. 또한 NP-완전 문제의 복잡도 변환에서 ‘존재/비존재’ 제약을 구조적으로 설명하기 어렵다는 문제의식이 깔려 있습니다.

- **Core Contribution**: 핵심 기여는 ‘간접 계산 모델’과 ‘간접 형식 방법’을 결합해, 큰 문자열과 작은 문자열 모두에 호환되는 간접적 형식화를 제안하는 것입니다. 이를 ‘트윈 튜링 머신’(twin Turing machine) 가상 구조로 모델링하여, 한쪽 리스트(표준화된 수/코드)와 다른쪽 리스트(자원화된 데이터 슬롯/문자 단위)를 1:1로 매칭하는 방식의 최적화 경로를 제시합니다. 결과적으로 데이터센터 중심의 기존 클라우드 최적화를 지식센터 중심의 지식 처리로 확장하는 설계 개념을 제공합니다.

- **Technical Challenges**: 간접 계산과 간접 형식을 결합할 때의 기술적 난제는, 목표 영역(target domain)에서 ‘알려진 영역’과 ‘알려지지 않은 영역’을 구분해도 응답 기준틀(reference frame) 안에서 효율적 열거·탐색이 가능하도록 데이터 구조를 설계하는 점입니다. 논문은 이상적 분류 집합(ideal classification set)을 이진/계층 구조로 재분류하고, 단일 집합-계층 집합-기호 집합(sign set)의 레벨 대응을 통해 ‘동의 병렬성 및 대응 변환(synonymy parallelism, corresponding transformation)’이 작동하도록 규칙을 고정합니다. 또한 트윈 튜링 머신의 ‘저울(balance) 비유’로 변환 제약을 타겟 영역 내 정보 방정식(IU=ID−IK) 형태로 정식화해 재현 가능한 계산 경로를 만든다고 주장합니다.

- **Empirical Impact**: 실증/검증은 중국어 정보 처리(문자 yan과 어군 yu의 관계)를 데이터베이스 형태로 구현한 예시에서 제시됩니다. 이 구조는 임의 시점의 열거·탐색을 지원하고, 재현율(recall)·정밀도(precision)·재사용률(reuse rate) 등으로 성능을 엄격히 측정할 수 있다고 설명합니다. 더 나아가 문자 기반 처리가 대규모/소규모에 일관되게 적용되며 기존 단어 분절이 ‘기호 집합 수준’에 머무르는 한계를 ‘계층 집합 수준’으로 확장할 수 있다는 영향도 제시합니다.



### Cross-Dataset Bloom Question Classification: Supervised Models and Prompted LLMs (https://arxiv.org/abs/2606.13684)
Comments:
          Accepted at AIED 2026. Abdolali Faraji and Mohammadreza Molavi contributed equally to this work

- **Prior Approaches**: 기존 연구는 SVM·TF–IDF(POS 가중) 같은 전통적 ML과 BERT 계열 미세조정(DL)로 Bloom 인지수준을 분류해 높은 성능을 보고했다. 다만 대부분이 동일 데이터의 무작위 train–test 분할에 의존해, 실제 교육 맥락·교사별 주관성이 다른 경우의 일반화 성능은 불명확했다. LLM은 교육 분류에 활용되었지만, Bloom 인지수준 ‘질문 분류’에 대한 체계적 다중 데이터 평가가 부족했다.

- **Core Contribution**: 이 논문은 Bloom 인지수준 질문 분류에서 “교차 데이터 일반화”를 핵심 문제로 두고, 기존 ML/DL의 데이터 전이 실패를 정량적으로 확인했다. 동시에 여러 프롬프트 전략을 비교해 LLM이 데이터 변화에 더 안정적으로 동작함을 보여주며, 특히 “수준별 대표 예시 + 해당 수준의 핵심 동사(action verbs)” 조합이 가장 잘 맞는 설정임을 제시한다. 나아가 이 최적 프롬프트 흐름을 경량 UI로 구현해 대규모 문항은행 분류를 실사용 관점에서 지원한다.

- **Technical Challenges**: Bloom 라벨링은 맥락·교사 관점에 따라 달라져 학습 데이터와 다른 교육 환경으로 넘어갈 때 모델 성능이 크게 흔들린다는 점이 기술적 난제다. 논문은 ML/DL이 훈련 데이터에 과도하게 적응하는 문제를 교차 데이터 F1 하락(평균적으로 큰 폭 감소)으로 드러내고, LLM은 미세조정 없이도 in-context 예시와 JSON 출력 설계로 태스크 수행의 재현성을 높여 견고성을 확보했다. 또한 UI에서는 교사가 소수 예시만 제공하면 자동으로 동사를 추출해 프롬프트를 구성하도록 하여 운영 부담을 줄였다.

- **Empirical Impact**: 5개 데이터셋(총 4,179문항) 교차 평가에서 감독학습 ML/DL은 미지 데이터로 갈수록 weighted F1이 평균 약 0.25~0.28 수준으로 크게 떨어졌다. 반면 LLM은 교차 환경에서도 상대적으로 성능 저하가 작았고, 최적 프롬프트에서 weighted F1이 최대 0.84까지 도달해 실무 적용 가능성을 강화했다. 실제 교사용 UI는 사용성 실험(N=50)에서 NASA-TLX 기준 부담이 낮고 SUS 평균 78.2로 ‘매우 양호’ 수준의 사용성을 보였으며, 교육 문항 분류 자동화의 현실적 경로를 제시했다.



### UP-NRPA: User Portrait based Nested Rollout Policy Adaptation for Planning with Large Language Models in Goal-oriented Dialogue Systems (https://arxiv.org/abs/2606.13683)
- **Prior Approaches**: 기존 목표 지향 대화 정책 계획은 프롬프트 엔지니어링이나 오프라인 강화학습 기반 MCTS/정책 플래너를 통해 성능을 끌어올려 왔습니다. 다만 오프라인 강화학습은 학습 데이터와 훈련 비용에 의존하고, 새로운 사용자 유형에서의 일반화와 실시간 적응이 약합니다. 또한 기존 사용자 페르소나 반영 기법(TRIP/UDP 등)은 사용자 피드백에 따라 동적으로 전략을 바꾸는 데 한계가 있어, 비협력(협상·설득) 및 목표 불일치 상황에서 성공률이 떨어지는 문제가 남았습니다.

- **Core Contribution**: 이 논문은 User Portrait 기반 Nested Rollout Policy Adaptation(UP-NRPA) 온라인 프레임워크를 제안해, 사용자 특성에 맞춘 대화 전략을 실시간으로 커스터마이즈합니다. 핵심은 Big Five 성격·의사결정 스타일로 사용자 초상(user portrait)을 만들고, 대화 중 실시간 피드백과 연결된 적응 메커니즘으로 정책을 조정한다는 점입니다. 특히 오프라인 강화학습 정책 모델을 학습하지 않고도, 온라인 탐색과 사용자 피드백만으로 전략을 업데이트합니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 다양한 사용자 특성을 반영하되, 목표 지향 행동 시퀀스를 온라인으로 효율적으로 탐색·개선하는 것입니다. 논문은 NRPA-GD의 중첩 롤아웃(Nested Rollout) 아이디어를 확장해, 2단계(레벨 2: 정책 후보 선택, 레벨 1: 다턴 롤아웃 시뮬레이션) 구조에서 보상을 계산하며 최적 시퀀스로 정책 분포를 점진적으로 이동시킵니다. 레벨 1에서 사용자 초상 기반 롤플레이 시뮬레이터가 행동에 대한 피드백을 주고, 보상 신호로 가중치를 업데이트해 오프라인 학습 없이도 적응형 샘플링이 되도록 설계했습니다.

- **Empirical Impact**: 실험에서는 협력/비협력 대화 벤치마크(ESConv, ExTES, CraigslistBargain, P4G)에서 대규모 사용자 시뮬레이터를 구성해 평가했으며, 전반적으로 목표 달성 성능이 크게 개선됐습니다. 특히 협상 관련 지표에서 sale-to-list ratio(SL)가 56.41% 상승했고, 여러 작업에서 성공률(SR)이 1.0000에 도달하는 결과도 보고됩니다. 또한 soft success rate(SSR) 및 인간 평가에서도 UP-NRPA가 TRIP/UDP/NRPA-GD 대비 일관되게 우수했으며, 오프라인 강화학습 없이도 사용자 유형이 달라도 대화 목표를 안정적으로 달성할 수 있음을 보여줍니다.



### Order Is Not Control: Driven-Dissipative Response Laws Across Artificial and Biological Systems (https://arxiv.org/abs/2606.12923)
Comments:
          52 pages, 7 figures, updated title

- **Prior Approaches**: 기존 연구들은 정렬(alignment), 해석가능성, 스티어링, 신경 섭동(neural perturbation)에서 “질서(order)”를 관찰하면 곧바로 제어(control)로 해석하는 경향이 있었다. 하지만 질서의 유도는 수신기(receiver)가 승인한 행동/결과 읽기(readout)의 ‘이동’과 동일하지 않을 수 있어, 관측(증거)과 제어(실제 제약 하 이동)를 분리하기 어렵다. 또한 프롬프트·원칙·어댑터·디코드 설정·생물학적 좌표 변화가 같은 방향의 구조를 만들더라도, 동일 조건(denominator)에서의 국소적 이동이 검증되지 않으면 제어라고 단정하기 힘들다.

- **Core Contribution**: 이 논문은 정렬·해석·스티어링 실험을 공통 언어로 연결하기 위해, ‘수신기 승인 반응 이동’을 만들어내는 경험적 대상인 반응 법칙(response law)을 제시한다. 제어는 단순한 구조화가 아니라, 수신기 상태에 의해 게이트(gated)된 반응 법칙 아래에서 유한한 노력으로 목표 반응(또는 결과 읽기) 계열을 같은 분모(denominator)에서 이동시키되, 손상(damage), 무응답/회피(null/evasive), 형식 무효(invalid format), 과과속(overdrive), 기준선 불필요 교란이 유계(bound)로 유지될 때만 성립한다. 따라서 “order=control”을 부정하고, 제어를 분모-조건(stochastic response kernel)과 그 파생 반응 지도/행동 효과로 정의한다.

- **Technical Challenges**: 핵심 난제는 ‘무엇이 실제 제어 증거인가’를 실험적으로 분리하는 것이다. 이를 위해 논문은 준비 매체(prepared medium), 욕조/경계 조건(bath/denominator), 수신기 상태 및 비교기(comparator), 반응 분지(basin), 싱크 채널, 선언된 노력까지를 분모 튜플로 고정한 뒤에만 반응 커널 𝒫δ(dy|x,a)와 그 유한 차분(반응 지도, 행동 효과)을 측정한다. 또한 조준 가능한 기준선 대비 이동과 함께 싱크/노력 채널을 동시에 보고, 강한 개입이 단조 개선이 아니라 손상으로 라우팅될 수 있음을 ‘분모 동일성’ 하에서 차단적으로 판정한다.

- **Empirical Impact**: 생물학(마우스 ALM, C. elegans, 제브라피시)에서는 좌표 동일성이나 기하가 곧 제어를 보장한다는 주장 없이도, 섭동-반응 사슬이 같은 형태의 분모-조건 반응 연산자(응답 연산자)를 구성함을 물리 기질에서 보여준다. LLM 생성 출력 패널에서는 분모가 고정된 조건에서 반응 벡터 성분 부호 정확도(예: 72.8~73.7%)가 상승하고, 은닉 관측자/읽기 예측의 정확도도 보고되며(예: 93.6%, 91.7%), 어댑터는 준비 매체의 취약성(susceptibility)을 바꿔 동일한 가시적 경계가 다른 반응 분지로 ‘받아들여지게’ 만든다. 결론적으로 이 논문은 국소 허용(admitted) 제어 영역과 측정 가능한 확률적 반응 연산자를 실증적으로 지지하면서도, 좌표 동일성·은닉/로짓 인과적 충분성·배포용(컨트롤러급) 선행 제너레이션 제어까지는 아직 남겨둔다는 점을 명확히 한다.



New uploads on arXiv(cs.IR)

### Private Information Retrieval for Large-Scale DNA-Based Data Storag (https://arxiv.org/abs/2606.14557)
Comments:
          9 pages, 6 figures

- **Prior Approaches**: 전통적 PIR은 전자 메모리의 임의 접근과 사전(시퀀싱 전) 대수 연산을 전제로 하며, 다중 서버 쿼리를 XOR 같은 간단한 조합으로 합쳐 정보이론적 프라이버시를 달성한다. 하지만 DNA 기반 저장은 풀 내부에서 XOR/선형결합 같은 계산이 불가능하고, 쿼리마다 표적 프라이머를 대량(서로 다른 레코드 수만큼) 합성해야 하면 비용이 급격히 커진다.

- **Core Contribution**: 이 논문은 합성 DNA 데이터 저장 환경에서 2-서버 PIR을 실현하기 위해, 전자 PIR이 의존하던 “시퀀싱 전 대수 조합” 대신 “저렴한 무작위 프라이머 풀 합성”을 활용하는 두 가지 적응 방식을 제안한다. 또한 사용자의 생화학적 능력(표적 프라이머 합성/풀 복제 가능 정도)에 따라 프라이버시·효율·실현가능성이 어떻게 달라지는지를 시나리오로 체계화해 정보이론적 프라이버시-효율 트레이드오프가 DNA 시스템에 어떻게 나타나는지 보여준다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) DNA 풀에서 임의 위치를 직접 인덱싱할 수 없고 (2) PCR 선택적 증폭이 프라이머 간 교차(cross-talk)를 유발하지 않도록 바코드 간 해밍 거리 조건을 만족시켜야 하며 (3) 무엇보다도 PIR 프라이버시가 시퀀싱 전 계산 없이 유지되어야 한다는 점이다. 저자들은 바코드를 충분히 멀리 배치하는 코딩-거리 조건과, 쿼리를 주로 무작위 프라이머로 만들고 표적 합성은 단일 서열 수준으로 제한하는 쿼리 생성 절차를 설계해 이를 해결했으며, 서버 응답은 시퀀싱 후 디지털 처리(기본 카운트 차이 또는 XOR 기반 이진화)로 복원 가능하게 구성한다.

- **Empirical Impact**: 논문은 잡음 없는 가정(완전한 PCR 증폭과 복제, 충분한 커버리지 하의 재구성)을 두고 프로토콜 구조와 효율을 분석 제시하며, 특히 전송량을 줄이기 위해 염기→이진 변환 후 서버에서 XOR을 수행하는 더 효율적인 두 번째 프로토콜을 제안한다. 결과적으로 DNA 기반 초장기 저장에서 프라이버시를 정보이론적 관점으로 다루는 설계 틀을 제공하며, 향후 PCR 증폭 편향·스트랜드 드롭아웃·시퀀싱 잡음 같은 현실 요인을 포함한 실험적 검증의 출발점이 된다.



### Verifiable User Simulation for Search and Recommendation Systems (https://arxiv.org/abs/2606.14474)
Comments:
          Presented as a half-day tutorial at SIGIR 2026, 4 pages

- **Prior Approaches**: 기존 정보검색/추천 평가에서는 상호작용을 모사하는 시뮬레이션이 오래전부터 있었지만, LLM 도입 전에는 대체로 모사 과정이 명확히 설명되기 어려웠다. 최근에는 LLM이 페르소나 조건 에이전트를 쉽게 만들어 오프라인 평가를 가속했지만, 왜 그런 선택이 나왔는지와 사용자 프로필 일치 여부를 확인하기가 어렵고 불투명하다는 문제가 남는다. 또한 LLM 응답은 언어·학력·문화 같은 배경 특성에 따라 편향되거나 차별적으로 달라져, 소수 집단에 대한 불공정 평가 위험이 커진다.

- **Core Contribution**: 이 튜토리얼은 사용자 시뮬레이터를 ‘검증 가능한 공학 산출물’로 취급하고, 7개 구성요소(구조화 페르소나, 작업용 계약, 인간-에이전트 매칭 실행, 감사 가능한 트레이스, 페르소나 정합성 검증, 구조화 피드백, 리파인먼트 루프)로 분해해 가시화한다. 목표는 시뮬레이션 가정을 명시적으로 만들고, 결과가 페르소나·계약·모델 선택 중 무엇에서 비롯됐는지 근거 기반으로 판별하도록 하는 워크플로를 제공하는 것이다. 특히 단일 인간-에이전트 비교를 통계적 검증으로 오해하지 않게 진단용 점검과 실증 검증을 분리해 설명한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 에이전트의 선택/생성 이유를 추적 가능한 형태로 남기고, (2) 결과의 불일치가 페르소나 모호성·계약 과소지정·역할극 해석 차이·모델의 확률적 변동 중 어디에 기인하는지 구분하는 것이다. 튜토리얼은 입력·행동·이유·타임스탬프·모델/제공자 설정까지 포함하는 트레이스 스키마와, 충실성·신뢰성·공정성·불일치 귀속을 함께 점검하는 검증 루브릭을 제시해 이를 해결한다. 또한 페르소나/계약을 검증 결과로 반복 갱신하는 리파인먼트 루프를 통해 사양 자체를 개선하도록 설계한다.

- **Empirical Impact**: 이 자료는 두 가지 미니 랩(추천 리스트 선택 평가, 검색 쿼리 생성 및 검증)에서 참가자가 트레이스를 직접 점검하고 불일치를 범주화하며 공정성·인구통계 편향 탐지 체크를 적용하는 방식으로 실용적 체험을 제공한다. 결과적으로 시뮬레이션 기반 평가가 ‘그럴듯함’에 기대는 것을 줄이고, 재현 가능하고 감사 가능한 증거를 확보하는 절차를 현장 연구자에게 제공한다는 의미가 있다. SIGIR 맥락에서도 정확도만으로는 부족하다는 관점을 강화하며, 시뮬레이터 검증을 공정성 감사를 포함한 표준 워크플로로 자리잡게 돕는 데 영향이 기대된다.



### ScoreGate: Adaptive Chunk Selection for Retrieval-Augmented Generation via Dual-Score Statistical Fusion (https://arxiv.org/abs/2606.14269)
Comments:
          20 pages, 6 figures, 14 tables

- **Prior Approaches**: 고정 길이 top-K 기반 RAG은 쿼리 복잡도와 무관하게 동일 개수의 청크를 생성기에 주입해 과잉 검색(좁은 질의)과 과소 검색(합성형 질의) 문제를 만든다. 후보군에서 순위를 바꾸는 재랭커나 LLM 기반 필터는 정밀도를 개선할 수 있지만, 재랭킹된 집합 자체의 ‘개수’ 선택을 본질적으로 다루지 않거나 추가 추론 비용이 발생한다. 또한 단일 점수 임계값은 임베딩 유사도-문맥 적합성의 상충을 충분히 회복하지 못한다.

- **Core Contribution**: ScoreGate는 두 단계 RAG 파이프라인에 추가 모델 호출 없이, 기존 파이프라인이 산출하는 bi-encoder 유사도 s_i와 cross-encoder 재랭커 점수 r_i만으로 ‘적응적 검색 카드inality(보낼 청크 수)’를 결정한다. 특히 cross-encoder가 긍정하지만 bi-encoder가 낮게 평가하는 어휘 불일치(vocabulary mismatch) 실패 모드를 점수공간의 버킷 분할과 비대칭 융합으로 ‘구출’한다. 결과적으로 생성기 입력의 청크 개수를 쿼리마다 조절하면서도 품질 저하 없이 효율을 개선하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 (s_i, r_i) 두 점수가 서로 다른 관점의 증거를 제공하므로, 단일 임계값이나 단순 thresholding으로는 대립 신호를 처리하기 어렵다는 점이다. ScoreGate는 점수공간을 4개 버킷으로 나눠 일치 영역은 결정적으로 보장하고, 불일치 영역은 r_i에 더 큰 가중치를 둔 비선형적(사실상 비대칭 선형) 융합 점수로 통합해 보유 임계값을 버킷별로 다르게 적용한다. 또한 쿼리별 점수 분포 차이를 고려해 per-query min–max 정규화를 사용하고, MAX-KK 상한으로 컨텍스트 초과를 안전장치로 막는다.

- **Empirical Impact**: MS MARCO에서는 ScoreGate가 MRR@10=0.401을 달성하면서 Standard Top-K(고정 10개) 대비 보유 청크를 약 35% 줄였다. 내부 벤치마크(300개 라벨)에서는 95% 신뢰구간 기준으로 유사양성 0건(정밀도 96.4%~100%)을 관찰하며 재현율 97.77%~99.34%를 유지했고, 쿼리당 토큰을 평균 34.8% 절감했으며 추가 지연은 31ms 수준에 그쳤다. 실제 프로덕션 트래픽에서도 품질을 해치지 않으면서 검색 카드inality를 동적으로 줄일 수 있음을 시사해, 효율-정확도 트레이드오프를 실용적으로 개선할 여지를 제공한다.



### ChronoID: Infusing Explicit Temporal Signals into Semantic IDs for Generative Recommendation (https://arxiv.org/abs/2606.14260)
- **Prior Approaches**: 생성형 추천에서는 OneRec, MiniOneRec 같은 방식이 항목 의미를 의미 ID(Semantic IDs)로 이산화해 토큰 생성으로 추천을 통합한다. 하지만 기존 Semantic ID 학습은 시간 정보를 텍스트 임베딩 기반 양자화에 의존해 ‘시간 비명시적(time-implicit)’으로만 영향을 주며, 시간은 세션 구성 휴리스틱이나 순서/위치 인코딩, 최적화 단계에서만 간접 반영된다.
그 결과 동일한 항목이 서로 다른 시간 맥락에서 등장해도 같은 의미 ID로 매핑되어, 시간에 따른 의미·관련성·사용자 의도 변화까지 표현하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Semantic ID 자체에 명시적으로 시간을 주입해야 한다는 문제의식을 바탕으로, 시간 인지(time-aware) 의미 ID 학습 프레임워크 ChronoID를 제안한다. 핵심은 시간을 의미 추상화(양자화/코드북) 단계에 넣는 ‘어디에/어떻게’의 설계 공간을 정리하고, 그에 따른 아키텍처 전략을 체계적으로 비교하는 것이다.
또한 시간 누수 없이 평가하기 위해 time-explicit 생성 추천 벤치마크를 새로 정의해, 시간에 따른 분리 학습/검증/테스트 프로토콜을 제공한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 시간 신호를 의미 ID 학습에 적절히 표현·퓨전하는 것인데, 이를 위해 ChronoID는 시간 인코딩(절대 시각 vs 상대 시간 간격), 퓨전 순서(early fusion vs late fusion), 양자화 구조(Residual vs Parallel)라는 세 축으로 설계 공간을 분해해 탐색한다.
또한 의미 ID가 이산 토큰으로 변환된 뒤 생성 모델에 입력된다는 점을 고려해, 절대 timestamp보다 상대 시간 Δt를 쓰는 편이 상호작용 리듬을 더 잘 반영할 수 있음을 실험적으로 확인하고, late fusion과 parallel quantization이 이종 입력 정렬과 식별력 향상에 유리함을 보인다.

- **Empirical Impact**: 실험에서 ChronoID의 time-explicit 의미 ID는 기존 생성/판별 기반 추천 기준선보다 일관되게 높은 HR@K와 NDCG@K를 보이며, 특히 MiniOneRec 대비 HR@3에서 큰 상대 개선이 보고된다. 시간 인코딩을 상대 시간으로 두고 late/parallel 설계를 적용할수록 성능이 더 좋아져, 단순 입력 증가가 아닌 ‘시간-텍스트 의미의 풍부화’가 이득의 원천임을 뒷받침한다.
나아가 시간 신호를 제거하거나 제로패딩으로 대체하면 성능이 크게 하락하고, 원자 수준의 timestamp만으로도 계절·명절 같은 고수준 패턴을 충분히 내재화할 수 있음을 보여 의미 ID 기반 생성 추천의 평가·설계 관점을 재정립한다.



### CoRe: A Continuously Reward-Finetuned LLM Query Rewriter for Multi-Stage Context-Aware Relevance in Web-Scale Video Search (https://arxiv.org/abs/2606.14127)
Comments:
          12 pages, 3 figures

- **Prior Approaches**: 기존 LLM 기반 쿼리 리라이터는 Query2Doc류의 의사 문서 확장, RL 기반 하류 신호 최적화, 또는 일부 상업 배포에서 SFT+대조학습 형태로 “문맥”을 반영했습니다. 그러나 운영 환경의 소비(리랭커 퓨전)와 훈련 보상 간 시뮬레이션-프로덕션 간극이 생기기 쉬워 오프라인 지표 기반 보상 대체가 성능/안정성을 제한했습니다.

- **Core Contribution**: 이 논문은 CoRe(Context Relevance)를 통해 “마지막으로 본 피드 문서”와 세션 내 후행 행동(엄격 클릭·스킵)을 결합해 문맥 인지 리라이터를 학습합니다. 핵심은 배포된 멀티모달 리렐런스 모델 점수로 보상을 만들고, 그 보상 형태를 프로덕션 랭킹 퓨전 대수와 곱셈 비율로 맞춰 간극을 줄인다는 점입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 정답 라벨 부재로 인한 선호(Preference) 유도와 (2) 주간 단위 연속 재배포에 맞춰 보상 계산·학습 비용을 감당해야 한다는 점입니다. 저자들은 DPO 스타일의 쌍대 목적을 상위 k/하위 k 궤적 부분집합에만 적용하는 semi-online Mixed Preference Optimization 루프와, 보상 호출 타이밍을 위상(phase) 단위로 분리하는 구조로 비용을 절감했습니다. 또한 멀티모달 점수 기반 보상에 더해 길이/종료 패널티와 콘텐츠 안전 필터링을 넣고, 보상 유사 지표만으로는 놓칠 수 있는 reward hacking을 안정성 지표로 탐지·회복하는 자동 승급 게이트를 구축했습니다.

- **Empirical Impact**: CoRe는 주요 쇼트비디오 검색 엔진에서 5개월 이상 주간 재배포되었고, 리라이터 출력은 recall/rawrank/finerank의 병렬 경로 신호로 사용해 기존 신호를 대체하지 않았습니다. 두 번의 순차적 A/B 런칭에서 리라이트 영향 쿼리의 change-query rate가 유의하게 감소했으며, 핵심 관련성·참여 지표도 기대 방향으로 개선되었습니다. 특히 다중 지표 기반 자동 승급 게이트로 후보를 반복 승격·회복한 운영 사례를 주 단위로 공개해, 장기 운영형 LLM 리라이터 연구에 실증적 기준을 제시합니다.



### Knowledge Graph Enhanced Memory-Augmented Retrieval for Long Context Modeling (https://arxiv.org/abs/2606.14047)
- **Prior Approaches**: 긴 문맥 언어모델은 토큰 수를 늘리는 방식(확장 attention, long-context LLM)과 외부 메모리 검색(메모리-증강, RAG 계열)로 발전해 왔습니다. 하지만 attention은 ‘lost-in-the-middle’로 중간 구간 정보가 비관련 정보처럼 처리되는 문제가 있고, 검색 기반 방법은 관련 항목을 ‘관계’가 아닌 ‘의미적 유사도’로만 고르기 쉬워 동일 개체의 상태 변화/인과 흐름을 놓칩니다. 또한 지식그래프 접근은 ConceptNet·Freebase 같은 고정 그래프에 의존하는 경우가 많아 세션/도메인에만 나타나는 고유 개체와 관계를 실시간으로 반영하기 어렵습니다.

- **Core Contribution**: KGERMAR는 추론(inference) 중에 입력 텍스트로부터 동적이고 문맥별 지식그래프(개체·관계)를 구성해, 의미 유사도 검색에 ‘명시적 엔터티 관계’ 신호를 결합합니다. 이를 위해 그래프 구조 임베딩과 텍스트 의미 임베딩을 함께 학습·융합하고, 기존 메모리-증강 구조를 확장해 관계 중심 검색이 가능하게 합니다. 결과적으로 수천 토큰 떨어진 과거 맥락에서도 ‘같은 개체의 인과/상태 진행’에 맞는 대상을 더 잘 찾아 일관된 장문 추론을 돕습니다.

- **Technical Challenges**: 핵심 난점은 (1) 긴 문맥에서 관계를 정확히 뽑아내야 하지만 관계추출은 로컬 윈도 제약을 받는다는 점, (2) 동적 그래프 품질 저하가 검색 성능으로 바로 이어질 수 있다는 점, (3) 그래프 기반 신호와 텍스트 의미 신호를 충돌 없이 효과적으로 결합해야 한다는 점입니다. 논문은 NER·관계추출(BERT 계열)로 문맥 그래프를 만들고, 개체 통합·관계 신뢰도 점수·그래프 필터링으로 잡음을 줄였습니다. 또한 R-GCN으로 관계 기반 전파(다중 hop)를 수행하고, contextual/semantic/structural의 3개 메모리 은행 검색 신호를 가중치 학습으로 융합해 그래프 구조와 텍스트 의미를 함께 주입합니다.

- **Empirical Impact**: SlimPajama, WikiText-103, PG-19, Proof-pile의 다양한 도메인에서 문맥 길이 1K~32K 전 구간으로 평가했으며, 기존 memory-augmented baseline 대비 최대 8.5% 낮은 perplexity와 2~2.5배 더 나은 메모리 효율을 보고합니다. 특히 구조적 관계 기반 검색이 강화되면서 NLU 태스크 전반에서 in-context learning 성능도 더 좋게 나타났습니다. 동적 지식그래프를 추론 시점에 구성해 도메인 특화 표현을 만든다는 점에서, 고정 지식베이스 의존을 줄이고 장문 추론의 ‘구조적 관련성’ 문제에 실질적 돌파구를 제시한 연구로 평가됩니다.



### When Recommendation Denoising Meets Popularity Bias: Understanding and Mitigating Their Interaction (https://arxiv.org/abs/2606.14046)
- **Prior Approaches**: 암묵적 피드백 로그(클릭·구매 등)는 노이즈 양성(오탭, 노출 편향, 인터페이스 효과)을 포함해 학습을 흔들 수 있다. 이를 줄이기 위해 기존 denoising 추천은 손실이 작은 샘플을 더 신뢰하는 small-loss 휴리스틱으로 양성 상호작용을 다운웨이트/필터링하는 방식이 널리 쓰였다. 그런데 희소한 tail 아이템은 실제로 선호를 반영해도 학습이 어려워 큰 손실을 받을 수 있어, 결과적으로 깨끗한 tail 신호까지 억제되어 head–tail 불균형이 악화될 수 있다는 맹점이 있었다.

- **Core Contribution**: 이 논문은 denoising과 popularity bias를 분리된 문제가 아니라 “결합되어 효과적 감독 신호를 재배분하는 메커니즘”으로 재정의한다. 특히 tail 양성의 손실 분포가 head보다 오른쪽으로 치우친(더 큰 손실을 갖는) 조건에서, 단조 small-loss 기반 재가중이 ERM 대비 효과적 head–tail 신호비를 증가시키는 실패 모드를 정식화한다. 이를 바탕으로 tail의 ‘어려운데도 정답인’ 신호를 보존하면서 head 쪽의 노이즈 억제는 유지하는 Popularity-Aware Denoising(PAD)를 제안한다.

- **Technical Challenges**: 핵심 난제는 손실 기반 가중이 “노이즈일 가능성”과 “진짜지만 학습이 어려움”을 구분하지 못한다는 점이다. 논문은 아이템 인기(popularity)에 따라 양성 손실이 체계적으로 달라지는 상황을 조건부로 가정하고, denoising 가중이 head와 tail에 서로 다른 비율로 감독을 배분함을 effective head–tail signal ratio로 정량화한다. 그 다음 PAD는 기존 denoiser의 가중치를 아이템 인기에 따라 게이팅해(인기 높은 아이템엔 더 강하게, 인기 낮은 아이템엔 보수적으로) tail의 과도한 억제를 완화하도록 설계했다.

- **Empirical Impact**: PAD는 MovieLens, Amazon-Book, Yelp에서 3종 데이터 및 3종 백본(예: GMF, NeuMF, LightGCN)과 여러 denoising 기준선에 대해 전반적으로 성능을 개선했으며, 특히 MF 스타일 모델에서 accuracy–diversity 트레이드오프가 유리하게 나타났다. 또한 전체적으로 denoising이 항상 최선은 아니라는 경계 사례를 함께 보여주는데, 예컨대 그래프 전파가 희소 신호를 완화해 별도의 명시적 denoising 필요성이 줄어드는 상황에서는 ERM+LightGCN이 여전히 강한 기준선이 될 수 있다. 종합하면 이 연구는 “손실 기반 denoising의 편향 상호작용”을 조건부로 설명하고, 이를 가볍게 보정하는 플러그인 프레임워크를 실증적으로 입증한 점에서 의미가 크다.



### ADORE: Iterative Query Expansion with Retrieval-Grounded Relevance Feedback (https://arxiv.org/abs/2606.13905)
- **Prior Approaches**: 기존 LLM 기반 질의 확장/재구성은 그럴듯한 확장 텍스트를 생성해 검색 성능을 높이려는 방식이 주류였습니다. 다만 생성된 확장이 실제로 목표 코퍼스에서 어떤 순위를 만들었는지(잡음/드리프트 여부)를 진단하지 못해, 언어적 적합성과 검색 효용이 어긋날 수 있습니다.

- **Core Contribution**: ADORE는 확장 생성과 평가(판단)를 분리하고, 검색 결과를 다음 라운드 확장에 피드백으로 다시 주입하는 “검색 근거형” 반복 프레임워크를 제안합니다. 핵심은 검색에서 드러난 관련 증거를 기준 질의에 대해 등급(relevance tier)으로 평가해, 보강할 신호와 억제할 신호를 구조화된 형태로 다음 생성에 반영한다는 점입니다.

- **Technical Challenges**: 기여를 실현하려면 (1) 코퍼스가 어떻게 반응하는지 관측하고 (2) 그 반응을 기준 질의에 고정된 방식으로 판단해 드리프트를 막는 체계가 필요합니다. ADORE는 라운드별로 가짜 패시지(pseudo-passages)를 생성-관측(재검색)한 뒤, 새로 검색된 문서에 대해 LLM 기반 relevance assessor가 0–3 등급의 피드백을 부여하고, 적응형 종료(품질·커버리지 포화 또는 예산 소진)로 불필요한 반복을 줄였습니다.

- **Empirical Impact**: TREC Deep Learning, BEIR, BRIGHT 전 범위에서 ADORE는 8개 설정 중 7개에서 기존 강력한 기준선을 앞섰고, 평균 nDCG@10을 BEIR에서 BM25 대비 24.5%, 기존 최강 질의 확장 대비 3.6% 개선했습니다. BRIGHT에서는 BM25 대비 122.9%, 최강 질의 확장 대비 9.2% 향상이었으며, 특히 추론형 도메인에서 BM25 기반임에도 조밀 검색기·LLM reranker를 크게 초과하는 결과가 보고되었습니다.



### Mood-Aware Music Recommendation: Integrating User Affective Signals into Ranking Systems (https://arxiv.org/abs/2606.13858)
Comments:
          13 pages, 4 figures, and 1 table

- **Prior Approaches**: 기존 음악 추천은 사용자-아이템 상호작용에 기반한 협업 필터링(CF)이 중심이었지만, 음악 도메인에서는 상호작용이 극도로 희소하고(99.9%+), 인기 편향과 명시적 피드백 부재로 성능이 흔들립니다. 콘텐츠 기반 방식은 장르·악기·가사 같은 아이템 속성으로 확장됐으나, 사용자의 감정(기분)을 반영한 개인화는 상대적으로 덜 다뤄졌습니다. 또한 사용자의 선호는 단기 맥락에 따라 변하는데도, 많은 접근이 정적인 선호 가정에 머물렀다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 사용자의 현재 기분을 에너지-발언스(energy–valence) 감정 공간에 매핑하고, 그 목표 감정에 맞춰 후보 곡을 재랭킹하는 ‘mood-conditioned ranking’ 프레임워크를 제안합니다. 특히 소프트맥스 기반 샘플링으로, 목표 감정과의 근접성은 높이되 완전 결정적으로만 고르지 않아 다양성도 함께 확보합니다. 결과적으로 장기 취향 유사도에 단기 감정 신호를 결합해 추천 결과를 조정하는 것이 핵심 기여입니다.

- **Technical Challenges**: 기술적으로는(1) 텍스트/오디오/사용자 상태가 서로 다른 형태의 데이터를 같은 감정 공간에서 비교 가능하게 표현하고, (2) 목표 감정에 더 맞는 곡을 선호하면서도 반복 추천에서 다양성을 잃지 않는 선택 메커니즘이 필요합니다. 논문은 Russell의 에너지-발언스 평면에서 사용자 기분과 곡의 오디오 감정 특징을 [0,1] 정규화로 정렬하고, 후보 전체의 에너지-발언스 거리(제곱 유클리드 거리)를 볼츠만 분포로 확률화한 뒤 소프트맥스 확률에 따라 서로 다른 곡을 샘플링하는 방식으로 해결합니다. 또한 Spotify(사용자 씨앗 트랙)–Last.fm(후보 탐색)–ReccoBeats(발언스/에너지 피처)로 파이프라인을 구성해 실사용 형태로 통합합니다.

- **Empirical Impact**: 단일 블라인드 사용자 연구에서 mood-aware 모델이 기준선(유사도 기반 KNN)보다 사용자 인지 품질 평가가 일관되게 높게 나왔습니다. 평균 평점은 mood-aware 3.59(±1.19), 대조군 2.67(±1.28)이며, 평점 분포가 4~5로 이동했고 Mann–Whitney U 검정에서도 유의한 개선(p≈0.01)이 보고됩니다. 다만 ‘Relaxed’와 ‘Sad’에서 개선 폭이 큰 반면 ‘Stimulated’와 ‘Distressed’에서는 뚜렷한 향상이 없었고, 표본 규모·자기보고 편향·크로스플랫폼 매칭 오류 같은 한계도 함께 제시합니다.



### Hybrid Neural Retrieval with Generative Query Refinement for Quranic Passage Retrieva (https://arxiv.org/abs/2606.13837)
Comments:
          Accepted for presentation at the Intelligent Methods, Systems, and Applications (IMSA) 2026 conference. \c{opyright} 2026 IEEE

- **Prior Approaches**: 기존 Quranic Passage Retrieval은 MSA 기반 질의와 CA 성서 본문 간 어휘·구조 차이 때문에 정확한 후보를 초기에 확보하지 못하는 문제가 컸습니다. BM25 같은 키워드 매칭은 어휘 불일치로 약했고, 단일 벡터 기반의 dense 모델은 도메인 미세조정 데이터가 적어 다중 구절(멀티-버스) 상황에서 랭킹 일관성이 떨어지는 한계가 드러났습니다.

- **Core Contribution**: 이 논문은 MSA↔CA 언어 간 ‘의미 갭’과 ‘영(0) 정답 질의’ 처리를 동시에 겨냥한 4단계 신경 아키텍처를 제안합니다. AraColBERT(밀집)와 BM25(희소)를 하이브리드로 결합해 후보 리콜을 넓힌 뒤, CAMeLBERTmix(교차 인코더)로 정밀 재랭킹하고, AraT5 기반 질의 정제(리라이트) 및 다중 구절 집계를 통해 문맥 적합도를 높입니다. 더불어 confidence gating으로 답이 없을 가능성이 큰 질의를 -1로 반환하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) CA 본문에 대한 초기 후보 리콜을 높이면서 (2) 교차 인코더가 만든 점수 스케일을 신뢰도 있게 임계로 분리하고 (3) 질의 정제가 오히려 잡음을 늘리지 않게 가중 융합하는 것입니다. 해결책으로 Reciprocal Rank Fusion으로 150개 후보를 안정적으로 모으고, 교차 인코더 최고 점수가 4.0 미만이면 unanswerable(-1)로 라우팅했으며, 원문 질의·정제 질의의 교차 인코더 점수를 랭크 신호와 함께 3방향 가중 늦은 결합(W1=0.4, W2=0.4, W3=0.2)했습니다.

- **Empirical Impact**: 확장된 Quran QA 2022/2023 계열 데이터(학습 1,895쌍)에서 제안 모델은 Recall@10=0.7024, MAP@10=0.4947로 기준선 대비 개선을 보였습니다. MRR=0.5807로 단일 모델의 최상위 수치(0.5900)에는 근소하게 못 미치지만, Recall과 MAP에서 20% 이상 큰 이득을 가져 다중-답 검색에 더 신뢰할 만한 균형 성능을 확인했습니다. 특히 ablation에서 하이브리드 후보-교차 인코더 재랭킹-가중 융합(및 질의 정제)의 시너지가 전체 성능을 견인함이 드러나며, 종교 텍스트 검색에서 ‘정답 없음’을 명시적으로 처리하는 실용성도 강화됐습니다.



### TASR: Training-Free Adaptive Stopping for Iterative Retrieva (https://arxiv.org/abs/2606.13814)
Comments:
          9 pages, 5 figures. Accepted at Agent4IR Workshop, KDD 2026

- **Prior Approaches**: 반복형 RAG 에이전트는 매 라운드마다 검색과 생성 호출을 반복하지만, 정답에 수렴한 뒤에도 추가 검색을 계속해 불필요한 비용을 쓰는 ‘overspending’ 문제가 자주 발생한다. 기존 해결책은 분류기나 value head 같은 학습 구성요소로 중단 정책을 학습하지만, 모델/태스크가 바뀔 때마다 재학습이 필요하고 결정이 덜 투명하다는 한계가 있다. 또한 retrieval을 언제 할지(예: FLARE) 또는 한 라운드 내 동작을 조정하는 연구는 많지만, ‘중단’ 자체를 독립적으로 다루는 방식은 제한적이었다.

- **Core Contribution**: 이 논문은 학습 없이(Training-free) 중단을 결정하는 한 줄 규칙 TASR을 제안한다. TASR은 (1) 이전 라운드의 정규화된 답이 현재 라운드에서 반복되고, (2) 정답일 확률로 보정된 calibrated logit margin이 0.25를 넘으면 즉시 중단한다. threshold와 규칙 구조를 고정해 24개(모델·검색기·코퍼스) 조합에 공통 적용 가능하다고 주장한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 ‘언제 더 검색을 해도 예측/근거가 바뀌지 않는가’를 학습기 없이도 안정적으로 판정하는 것이다. 이를 위해 verbalized confidence는 RLHF 튜닝 모델에서 붕괴하는 병리(대부분 5로 쏠림)를 보이며, 대신 answer 토큰 커밋 지점의 logit gap을 isotonic calibration으로 라운드별 보정해 margin-to-correctness 매핑을 재구성한다. 또한 answer-stable(연속 라운드 답의 일치)와 margin을 AND-게이트로 결합해 ‘그럴듯하지만 틀린 안정 답’에서의 조기 중단을 줄인다.

- **Empirical Impact**: 실험 결과 TASR은 고정 호출 예산 대비 Pareto 우위를 보여, distractor 설정에서 fixed-k=5의 매크로 F1을 94.8% 수준으로 유지하면서 호출은 62.6%로 줄였다. 특히 fixed-k=3 대비 매크로 F1이 +3.42 개선되며, open-domain BM25 설정에서도 동일한 캘리브레이션 전이로 유사한 패턴이 재현되어(총 9개 셀) retraining 없이 일반화 가능성을 뒷받침한다. 또한 381개 후보 중단 규칙 중 어떤 대안도 평가 구성에서 Pareto-dominates 하지 못했으며, TASR은 학습된 중단 컨트롤러와 공정 비교 가능한 ‘감사 가능한(auditable) 기준선’으로 자리할 수 있다.



### Nomenclature Ontology for Medical And Disease names (NOMAD): taxonomy of types and origins of disease names (https://arxiv.org/abs/2606.13719)
- **Prior Approaches**: 기존에는 질병 명칭(예: 그리스·라틴·아랍어 기반)의 역사적·문화적 관행이 연구돼 왔지만, 전 질병 범위를 아우르는 체계적 “명명 규칙” 분류 프레임워크는 부재했습니다. 결과적으로 학습용 라벨링 기준이 일관되지 않아 비교·분석의 확장성이 제한됐습니다.

- **Core Contribution**: 이 논문은 NOMAD(Nomenclature Ontology for Medical And Disease names)라는 메타-택소노미를 제안해 질병 명칭을 “명명 관습”에 따라 분류합니다. 9개의 상위 범주와 20개의 하위 범주로 구성된 2단계 분류 체계를 만들고, 명칭의 구성적 성격을 반영해 멀티라벨 분류로 설계했습니다.

- **Technical Challenges**: 핵심 난제는 방대한 ICD-10-CM 2026 색인(22,548개)에서 명칭 규칙을 안정적으로 추출·분류하는 것이며, 한 항목이 여러 규칙을 동시에 가질 수 있다는 점(구성성)입니다. 논문은 이를 위해 머신러닝 기반 3단계 스케일러블 분류 파이프라인을 구성하고, 멀티라벨로 99.1% 항목을 분류(항목당 평균 2.12 라벨)하며 성능을 확보했습니다.

- **Empirical Impact**: 실험적으로는 해석 가능성을 위해 정확성 검증을 수행했는데, 표본 2,255개(10%) 수동 검토에서 완전 일치 70%, 부분 일치 26%와 높은 Cohen’s Kappa(0.832)를 보고했습니다. 또한 NOMAD 프로파일이 ICD-10-CM 각 장(챕터)별로 크게 달라 감염병은 원인(etiological)·지리 관련 라벨 비중이 높고, 순환기·정신/행동장애는 해부학·병태생리 또는 사회·행동 라벨이 두드러지는 등 분야별 인식론적 전통 차이를 정량화했다는 점에서 의미가 있습니다.



### Personalization and Evaluation of Conversational Information Access (https://arxiv.org/abs/2606.13717)
Comments:
          PhD Thesis of Hideaki Joko (Radboud University, the Netherlands)

- **Prior Approaches**: 기존 대화형 정보 접근(CIA) 연구는 대체로 대화 흐름 내에서의 지식 기반 응답을 다루되, 사용자의 개인 맥락(선호·소유물·관계 등)을 제대로 반영하지 못하는 한계가 컸습니다. 또한 Transformer 기반 대화 생성은 그럴듯하지만 부정확한 정보를 만들 수 있어, 검색·근거(grounding)가 중요하지만 이를 포함한 개인화 설계와 평가가 여전히 어렵다고 지적됩니다.

- **Core Contribution**: 이 논문은 개인화된 CIA를 만들고(개인 맥락 추출·개인화 응답 생성) 이를 신뢰성 있게 평가하는(전체 대화 기준, 참고답변 없이) end-to-end 연구를 제시합니다. 특히 대화에서의 개체 연결(EL)을 ‘개념·명명 개체·개인 개체’로 나눠 정의하고, 이를 활용해 선호 기반 응답을 생성하며, 대화 전반을 다면(측면) 점수로 평가하는 FACE를 제안합니다.

- **Technical Challenges**: 핵심 기술 난점은 대화에서 개인 개체와 개념을 정확히 식별하고, 그 결과를 대규모 다중 세션 개인화 데이터로 연결하며, 마지막으로 대화 전체의 품질을 참고답변 없이 해석 가능하게 측정하는 데 있습니다. 이를 위해 대화 전용 EL 태스크와 데이터(ConEL/ConEL-2)를 만들고, 코어퍼런스 해석을 활용하는 CREL, LLM으로 사람 작업자를 안내해 대규모 개인화 대화 데이터를 구축하는 LAPS, 그리고 응답을 ‘conversation particle’로 분해해 최적화된 지시로 평가하는 FACE를 결합합니다.

- **Empirical Impact**: 실험 결과 CREL은 기존 범용 EL보다 개인 개체·개념 처리에서 유의미하게 향상되며, LAPS는 인간이 쓴 대규모·다중 세션 개인화 대화를 빠르게 확보하면서도 다양성과 품질을 유지하는 것으로 보고됩니다. FACE 역시 인간 평가와의 정합성이 높고 측면별(예: 관련성, 사용자 이해)로 세밀한 피드백을 제공해, 개인화 CIA 개발에서 자동 진단과 반복 개선에 실질적 의미를 갖습니다.



New uploads on arXiv(cs.CV)

### Gaze Heads: How VLMs Look at What They Describ (https://arxiv.org/abs/2606.14703)
- **Prior Approaches**: 기존 기계해석 연구는 비전-언어 모델(VLM)에서 이미지 토큰을 주목하는 attention head들을 “신호”로만 다뤘습니다. 예를 들어 Image Heads는 단일 패스의 이미지 주목 특성을 이용해 대비적 디코딩 신호를 만들고, Localization Heads는 소수 head의 공간 집중도를 읽어 시각적 grounding(박스/마스크)에 활용합니다. 하지만 이런 방법들은 생성 시점마다 어떤 영역을 ‘지금 말하는 중’으로 라우팅하는지, 그리고 그 head들이 출력 선택을 ‘인과적으로’ 바꾸는지까지는 답하지 못했습니다.

- **Core Contribution**: 이 논문은 이미지 서술 과정에서 모델이 사람의 시선처럼 “현재 말하는 이미지 영역”으로 attention을 좁히는 메커니즘이 있음을 보여줍니다. 그 핵심이 언어모델 backbone의 소수 attention head인 gaze heads이며, 이 head들은 패널/객체를 단계적으로 전환하며 해당 영역에 시선을 고정합니다. 더 나아가 gaze heads의 attention을 원하는 영역으로 강제로 바꾸면 VLM의 문장이 실제로 그 영역을 묘사하도록 전환되어, head를 통한 출력 제어의 인과성을 제시합니다.

- **Technical Challenges**: 첫째, 헤드가 너무 많아 ‘어떤 head가 시각적 라우팅을 담당하는지’를 라벨 없이 찾아야 했습니다. 저자들은 몇 번의 forward pass에서 ‘질문한 패널(또는 영역)과 대응되는 이미지 토큰에 attention이 재배치되는지’만을 상관 점수로 측정해 gaze heads를 좁혀냈고, 이 점수는 고정된 단일 패스 집중도와 달리 시간(질문 변화)에 따른 추적성을 반영합니다. 둘째, 인과성 검증을 위해 학습 없이 inference-time에서 특정 head들의 attention 마스크를 편집해 타깃 영역으로 강제 이동시키는 개입을 설계했으며, 개입 범위를 늘리면 제어가 포화되거나(부분 제어) 과도하면 생성이 붕괴되는 “정밀한 튜닝” 현상도 확인했습니다.

- **Empirical Impact**: 실험은 만화(comic strip)라는 통제된 환경에서 시작해, 상위 100개 gaze heads(전체 head의 8.7% 수준) 개입만으로 선택한 패널로 답을 유도하는 정확도 83.1%를 달성했습니다(무작위 head 개입은 실패, 전부 개입은 붕괴). 연속 서술에서도 생성 도중 타깃을 바꾸면 모델이 해당 패널 설명을 마무리하고 새 패널로 넘어가며, natural 이미지에서는 COCO 객체 영역으로 유사한 유도 효과가 나타납니다. 또한 모델 규모(2B~32B)와 여러 VLM 아키텍처로 확장되지만, vision encoder를 고정하는 일부 계열에서는 gaze heads에 해당하는 명확한 집합이 관측되지 않아 “학습 방식 의존성”까지 시사합니다.



### OmniVideo-100K: A Dataset for Audio-Visual Reasoning through Structured Scripts and Evidence Chains (https://arxiv.org/abs/2606.14702)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 자동화 오디오-비주얼 QA 파이프라인은 ‘비디오-캡션-QA’에 가까운 방식으로, 영상을 짧은 클립으로 쪼개 각 모달리티의 설명을 따로 만들고 이후 QA를 생성하는 경향이 있습니다. 이 decoupled 처리로 인해 소리와 시각의 고유한 대응이 끊기고, 독립 클립 기술은 같은 개체에 대한 서술이 구간마다 흔들리기 쉽습니다. 또한 긴 텍스트 이해와 QA 합성을 한 단계에 몰면, 질문이 장기 시간 연결이나 깊은 크로스모달 추론을 충분히 담지 못하는 한계가 있습니다.

- **Core Contribution**: 이 논문은 오디오-비주얼 QA 생성을 위한 데이터 엔진을 제안하며, 핵심은 두 단계 메커니즘입니다. 첫째, Entity-Anchored Video Scripting(개체 앵커드 스크립팅)으로 영상을 요약·주요 개체 목록·구간별 오디오/비주얼 설명이 포함된 구조화 텍스트로 변환해, 개체 기준선과 발화자 태깅으로 구간 간 일관성을 확보합니다. 둘째, Clue-Guided QA Generation(단서 유도 QA 생성)으로 스크립트에서 태스크 중심의 크로스모달 단서 체인을 먼저 채굴한 뒤, 그 단서에 근거해 장기 시간 범위의 QA를 생성하도록 유도합니다.

- **Technical Challenges**: 가장 큰 기술적 도전은 (1) 분절된 처리에서 발생하는 ‘개체 참조 불일치’와 ‘소리-시각 대응 단절’을 줄이면서 (2) 장기 시간·깊은 크로스모달 의존이 필요한 질문을 자동으로 만들 수 있느냐였습니다. 저자들은 스크립트 단계에서 전 구간을 잇는 메인 엔터티 목록(전역 prior)과 발화자 라벨을 통해 참조성과 오디오-시각 연계를 동시에 고정하고, 시계열 백본(타임스탬프 기반 구간화) 위에 설명을 재구성했습니다. QA 생성 단계에서는 전체 텍스트에서 즉시 질문을 뽑기보다 단서 체인을 명시적으로 추출·축약해 모델의 추론을 ‘핵심 evidence chain’에 고정함으로써 장거리 연결을 강화했습니다.

- **Empirical Impact**: 이 파이프라인으로 10개 태스크의 100K 규모 자동 instruction-tuning 데이터 OmniVideo-100K와, 264개 영상 기반 505개 사람 검증 테스트 셋 OmniVideo-Test를 구축했습니다. OmniVideo-100K로 미세조정한 VITA-1.5, Qwen2.5-Omni-7B, Qwen3-Omni-30B는 각각 OmniVideo-Test에서 최대 20.59% 향상을 보였고, 여러 기존 벤치마크(Daily-Omni, JointAVBench 등)로도 일반화 개선이 관찰됐습니다. 추가 분석에서는 단일 모달 의존이 줄고 오디오-비주얼 시너지와 장기 단서 기반 추론이 더 잘 작동함을 정성·정량으로 뒷받침했습니다.



### RATS! Patches Talk Through Registers: Emergent Parts in Register Attention Transformers (https://arxiv.org/abs/2606.14701)
- **Prior Approaches**: 기존의 DINO 계열 자기지도 학습은 패치 토큰이 부품 수준 특성을 어느 정도 내재하고 있음을 보여줬지만, 이를 “부품” 단위로 명시적으로 묶어 표현으로 만들 메커니즘은 부족했습니다. 또한 슈퍼픽셀 기반 방법은 지역 경계에 기대는 경향이 있어 전역적 맥락을 통한 part identity 형성이 제한되고, 오브젝트 중심 slot 방법은 재구성 중심 감독 때문에 부품 의미에 직접 수렴하기 어렵습니다. Perceiver IO나 GroupViT 같은 접근은 중간 잠재 토큰이 잘 섞이거나, 텍스트 쌍 의존 등으로 인해 “자기지도만으로 부품에 해당하는 구조를 자발적으로” 끌어내는 데 한계가 있었습니다.

- **Core Contribution**: RATS(Register Attention Transformers)는 분류용 [CLS] 집계를 N개의 학습 가능한 register 토큰으로 분해하고, L→N→N→L 병목을 거치며 패치 정보를 압축-교환-브로드캐스트하는 구조를 제안합니다. 특히 register를 attention head에 분할 배치해 서로 다른 head 간 register가 직접 상호작용하지 못하게 함으로써, 각 register가 고유한 proto-semantic(부품 유사) 영역에 자연스럽게 특화되도록 유도합니다. 보조 손실이나 부품 어노테이션 없이도 register 사전(dictionary)이 부품 수준 일관성과 의미적 근접성을 보인다는 점이 핵심입니다.

- **Technical Challenges**: 어려움은 자기지도 학습에서 부품처럼 “의미 있는 영역 그룹”을 모델이 스스로 커밋(학습적으로 약속)하도록 만드는 구조 설계에 있습니다. RATS는 attention 병목에서만 정보를 다루게 하고, compress에서는 패치를 register로 요약한 뒤 communicate 단계의 register-간 교환으로 결합 맥락을 만들며, 마지막 broadcast로 패치에 갱신된 register 기반 정보를 다시 주입합니다. 또한 head별로 register 부분집합을 독립 운용하고, DINO self-distillation 학습에서 register의 최종 평균을 글로벌 표현으로 사용해 별도 감독 없이도 register-패치 상관이 국소적·정합적으로 형성되게 했습니다.

- **Empirical Impact**: 실험에서 RATS는 5개 세분화 벤치마크에서 기존 기준선을 평균 +12 mIoU로 크게 앞섰고, ADE20K에서 +1.11 mIoU, COCO에서 AP^m +0.2만큼의 일관된 향상이 보고됩니다. 또한 Mask2Former와 같은 쿼리 기반 디코더에 register 토큰을 입력하면, 단순 ViT 백본 교체만으로는 설명되지 않는 방식으로 ADE20K와 COCO 모두에서 성능 향상이 나타나 “부품 의미를 갖는 쿼리”로서의 실용성도 확인됩니다. register dictionary는 PartImageNet에서 부품 일관성/재사용성과 함께 카테고리 간 의미적 계통(분류 체계적 근접성)을 보여, 해석 가능한 구조적 비전 표현을 위한 아키텍처 사전(architectural prior)으로서의 의미가 큽니다.



### RepFusion: Leveraging Multimodal Priors for Denoising in Representation Spac (https://arxiv.org/abs/2606.14700)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 T2I 파이프라인은 LLM을 주로 텍스트 인코더로만 써서 텍스트 임베딩을 만들고, 실제 denoising(잡음 제거)과 생성은 새로 학습한 생성 백본(예: diffusion transformer)이 담당합니다. 또한 VAE 잠재공간에서는 그 잠재가 의미적으로 구조화되어 있지 않아, LLM이 denoising 루프 안에서 무엇을 “읽어야 유리한지”가 불명확했습니다. Transfusion처럼 노이즈 잠재를 언어모델 입력으로 넣는 시도도 있었지만, 표현공간이 잘 맞지 않으면 이득이 제한적이었습니다.

- **Core Contribution**: RepFusion은 멀티모달 LLM(MLLM)을 새 denoiser로 쓰는 대신, 노이즈가 섞인 시각 표현을 인코딩하는 “노이즈 표현 인코더”로 재목적화합니다. 구체적으로 representation autoencoder(RAE)의 노이즈 시각 토큰을 MLLM에 넣고, MLLM 출력(변화하는 조건)을 diffusion transformer(DiT)의 조건 신호로 사용해 생성합니다. 핵심은 “LLM이 정적 텍스트 임베딩만 만들지 말고, denoising 과정에서 진화하는 시각 표현을 해석하도록 한다”는 설계 전환입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 MLLM이 clean 시각 표현에는 잘 맞지만, denoising 단계에서의 noisy 시각 표현을 안정적으로 해석할 수 있는지였습니다. 논문은 MLLM의 MLP projector만 학습하고 백본은 고정한 채로, RAE 잠재공간에서 noisy representation을 토큰화해 MLLM이 시계열적으로 변하는 조건을 다시 인코딩하도록 구성했습니다. 또한 테스트 시마다 조건이 갱신되도록 하여(진화하는 noisy 입력 재주입), 단순한 재계산 비용이 아니라 입력의 변화가 이득으로 연결되게 했습니다.

- **Empirical Impact**: 통제된 비교에서 같은 추론 예산(FLOPs) 하에 RepFusion은 텍스트 임베딩 기반(TextEmbed)과 Transfusion 계열을 능가하며, 특히 RAE로의 전환에서 상대 성능 향상이 가장 크게 나타났습니다. 또한 noisy representation을 고정된 learnable query로 대체한 대조군에서는 성능이 크게 떨어져, 개선의 원인이 “진화하는 noisy 입력을 바탕으로 반복적으로 조건을 만드는 것”임을 확인했습니다. 결과적으로 MLLM이 denoising 표현공간에서 강력한 prior를 제공하며, 이를 통해 modern T2I에서 테스트-time compute를 더 생산적으로 쓸 수 있음을 실증합니다.



### Instruct-Particulate: Scaling Feed-Forward 3D Object Articulation with Kinematic Contro (https://arxiv.org/abs/2606.14699)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존의 관절형 3D 객체 복원 연구는 소수의 정밀 라벨 데이터에 크게 의존해 왔고, 이 때문에 새로운 범주로의 일반화가 제한적이었다. 또한 최적화 기반 접근은 입력이 많아 스케일이 어렵고, 최근의 피드포워드 모델도 데이터 희소성 때문에 성능이 병목에 걸리는 문제가 있었다.

- **Core Contribution**: Instruct-Particulate는 정적 3D 메쉬와 함께 목표 ‘운동학(kinematic) 스펙’—부품 설명, 연결 구조, 관절 종류, 선택적 점 프롬프트—를 입력으로 받아 해당하는 관절형 분할과 관절 운동 파라미터를 한 번에 예측한다. 이때 스펙이 과업을 구체화해 모호한 정답(가능한 분할의 다수)을 평균내는 문제를 줄이고, 서로 다른 세분화 수준의 라벨도 함께 학습에 활용할 수 있게 한다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 같은 객체라도 관절 구조를 하나로 고정하기 어렵고(분할/의미의 불일치), 데이터 다양성이 증가할수록 일관성 문제가 커진다는 점이다. 이를 위해 모델 입력에 명시적 kinematic structure를 포함시키고, VLM이 테스트 시 스펙을 자동으로 획득하도록 설계했으며, 인코더-디코더 형태의 대규모 트랜스포머로 점-쿼리 기반 관절 분할과 관절 축·범위 파라미터를 동시에 디코딩한다.

- **Empirical Impact**: 논문은 15만 개를 훨씬 넘는 관절형 3D 객체로 구성된 이질(heterogeneous) 학습 데이터(합성 포함) 구축과 함께 실험을 수행했으며, 범주 간 일반화와 AI 생성 메시에 대한 성능이 기존 대비 더 좋아짐을 보였다. 또한 이미지-투-3D 모델을 통해 실제 이미지에서 관절형 애셋을 물리 시뮬레이터에 바로 내보낼 수 있는 수준으로 생성·재구성하는 확장 가능성도 확인했다.



### ClinHallu: A Benchmark for Diagnosing Stage-Wise Hallucinations in Medical MLLM Reasoning (https://arxiv.org/abs/2606.14697)
Comments:
          Code and datasets: this https URL

- **Prior Approaches**: 기존 의료 홀루시네이션 벤치마크는 정답/오답 같은 최종 출력 중심으로 평가해, 오류가 추론 과정의 어디에서 발생하는지(시각 인식 실패, 의학 지식 회상 실패, 근거 통합 실패)를 분리해 진단하기 어렵습니다. CARES·Med-HallMark 등 멀티모달 평가가 확장됐더라도 여전히 “결과가 틀렸는지”에 집중하는 경향이 강했습니다. 그 결과 같은 오답이라도 서로 다른 원인 실패가 하나의 판단으로 뭉쳐지는 한계가 있었습니다.

- **Core Contribution**: ClinHallu는 의료 MLLM 추론을 Visual Recognition(시각 인식), Knowledge Recall(지식 회상), Reasoning Integration(근거 통합) 3단계로 분해해, 단계별 홀루시네이션 원인을 추적하는 벤치마크를 제안합니다. 4개 의료 VQA 데이터셋을 기반으로 총 7,031개의 검증된 인스턴스를 만들고, 각 샘플에 검증된 기준 추론 트레이스를 붙여 “어떤 단계가 병목인지”를 진단할 수 있게 했습니다. 또한 단계 교체(stage-replacement) 개입으로 특정 단계 수정이 최종 정답에 어떻게 영향을 주는지도 측정합니다.

- **Technical Challenges**: 핵심 기술 과제는 기준 추론 트레이스를 대규모로 생성하되, 포맷이 올바르고 정답과 모순되지 않는 “신뢰 가능한 트레이스”만 남기는 데 있습니다. 논문은 기준 트레이스를 생성한 뒤 LLM-as-judge로 트레이스 포맷 유효성과 정답 일관성을 검증해 필터링했고, 그 결과만 벤치마크에 사용했습니다. 평가 단계에서는 후보 모델이 생성한 트레이스와 기준 트레이스를 비교하는 방식으로 단계별 홀루시네이션을 라벨링하며, upstream 단계를 기준으로 고정한 상태에서 stage-replacement로 원인 분리를 수행합니다.

- **Empirical Impact**: 실험 결과, 홀루시네이션 병목은 데이터셋마다 다르게 나타났습니다(예: VQA-RAD는 시각 오류 비중이 높고, MedXpertQA는 지식 오류 비중이 큼). 또한 추론 통합 단계의 홀루시네이션은 전반적으로 시각·지식 단계보다 낮게 나타나 “신뢰성 실패의 주원인이 마지막 추론 자체라기보다 상류 단계”일 수 있음을 보여줍니다. 더 나아가 트레이스 슈퍼바이즈드 파인튜닝은 단계별 홀루시네이션을 줄이고 정답 정확도를 개선했으며, 자동 심판 모델의 판단도 인간 라벨과 높은 일치도를 보여 대규모 진단 체계로서의 실용성을 뒷받침합니다.



### CottonLeafVision: An Explainable and Robust Deep Learning Framework for Cotton Leaf Disease Classification (https://arxiv.org/abs/2606.14686)
Comments:
          This paper contains 11 figures and 4 tables. It was Presented at 18th IEEE International Conference on Computational Intelligence and Communication Networks (CICN) 2026

- **Prior Approaches**: 기존에는 딥러닝 기반 분류를 위해 ImageNet 등으로 사전학습된 합성곱 신경망을 그대로 가져오거나, 간단한 전처리와 데이터 증강만으로 성능을 끌어올리는 방식이 주로 사용됐다. 하지만 현장 채집 환경의 조명·배경·잡음이 섞인 조건에서는 오분류가 늘고, 의사결정 근거를 설명하기도 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 면 작물의 잎 질병을 실제 농업 환경에 가까운 데이터에서 7개 클래스(6개 질병, 1개 건강)로 분류·탐지하려는 ‘CottonLeafVision’을 제안한다. 또한 DenseNet201을 중심으로 높은 분류 정확도를 달성하고, Grad-CAM·폐색 민감도 분석 등으로 신뢰성과 해석 가능성을 함께 강화했다.

- **Technical Challenges**: 핵심 기술적 난제는 현장 데이터의 잡음과 변동성이 커서 모델이 특정 패턴에 과적합되거나 취약해질 수 있다는 점이다. 논문은 Grad-CAM과 폐색 민감도 분석으로 모델의 근거 영역을 점검하고, 적대적 학습을 통해 잡음 저항성을 높여 이러한 취약성을 완화하는 방향으로 설계했다.

- **Empirical Impact**: 실험 결과 DenseNet201에서 분류 정확도 98%를 달성했으며, 이는 다양한 현장 조건이 반영된 공개 데이터셋에서도 강건하게 동작함을 보여준다. 더 나아가 실제 활용을 염두에 둔 프로토타입을 개발해, 농업 현장에서 잎 질병 관리 의사결정을 지원할 수 있는 실용적 가능성을 입증한다.



### HumP-KD: A Hybrid Uncertainty-Aware Multi-Stage Progressive Knowledge Distillation Framework for Efficient Fire Classification (https://arxiv.org/abs/2606.14684)
- **Prior Approaches**: 기존 화재 분류 연구는 CNN·비전 트랜스포머 기반 분류 성능을 높이는 데 집중해 왔고, 전이학습·온라인 증강·주의집중(예: 다중 스케일 공간 주의) 같은 기법이 주로 쓰였다. 또한 지식 증류는 한 종류의 교사나 단일 불확실성/다중 단계 전략을 각각 독립적으로 적용하는 경우가 많아, 교사들 간 차이를 함께 다루지 못하거나 경량 학생 학습을 안정적으로 만들기 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 HumP-KD로, 불확실성(교사 예측 분산) 기반 가중치, 계층적 특징-공간 주의 마스킹, 그리고 멀티스테이지(훈련 구간별 단계적) 점진 증류를 한 파이프라인에 통합한다. 서로 이질적인 두 교사 Swin-Tiny와 ViT-Base 및 Meta-MLP 앙상블의 지식을, MobileViT-S 학생에 단계적으로 전달해 정확도와 경량성을 동시에 노린다.

- **Technical Challenges**: 어려움은 (1) 두 교사가 표현하는 구조적 특징 스케일 차이를 정렬하면서, (2) 교사 확신이 낮은 샘플에서 잘못된 지식을 전달하지 않도록 하며, (3) 학생이 적은 단계 신호로 학습할 때 성능이 흔들리지 않게 만드는 것이다. 논문은 교사 예측 분산으로 샘플별 가중치(UAKD)를 만들고, HFBuilder가 교사 다중 계층 특징을 융합해 분별 영역을 선택하는 이진 공간 주의 마스크(HPKD)를 생성하며, 훈련 초·중·후반에 3단계를 점진 활성화(MSKD)해 증류 강도를 제어한다.

- **Empirical Impact**: Dataset-II에서 HumP-KD는 10회 독립 실험 평균 F1이 0.9876±0.0063으로, 증류 없이 학습한 MobileViT-S 기준(0.9537±0.0351) 대비 유의미하게 향상되며 t-test(p=0.0195)와 Wilcoxon 검정(p=0.0039)으로 통계적 유의성을 확인했다. 또한 학생은 4.94M 파라미터(모델 크기 19.01Mb)로 Swin-Tiny 대비 5.7배, ViT-Base 대비 17.5배 감소하면서도 37.72 CPU FPS를 달성해 실시간 배치 가능성을 보여주며, 잡음/모션블러 같은 열화 조건과 교차 데이터 일반화에서도 강건함을 보고했다.



### Memento: Reconstruct to Remember for Consistent Long Video Generation (https://arxiv.org/abs/2606.14667)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 장편 비디오 생성은 보통 샷 단위로 분해해 생성 효율을 높이지만, 핵심은 ‘그럴듯한 다음 샷’의 연속성에 있습니다. Storyboard 방식은 키프레임 중심이라 샷 내부 결합이 약하고, joint multi-shot은 컨텍스트 길이/계산량 한계로 장기 확장이 어렵습니다. 메모리 기반 자동회귀 방법은 압축된 과거 정보를 쓰되, 주로 일반적 관련성이나 단기 요약에 의존해 정체성(아이덴티티) 증거가 시간이 갈수록 희석될 수 있다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 장편에서 반복 주체의 일관성을 ‘메모리 기반 정체성 그라운딩’ 문제로 명시하고, 주체 보존을 직접 검증 가능한 목표로 전환합니다. 제안하는 Memento는 다음 샷 생성(autoregessive)과 함께 메모리만으로 해당 주체를 복원(subject reconstruction)하도록 보조 학습을 수행합니다. 또한 단일 메모리에서 정체성과 로컬 문맥이 섞이며 경쟁하는 문제를 줄이기 위해 dual-query 메모리로 장기 정체성 증거와 단기 연속성 단서를 분리합니다.

- **Technical Challenges**: 가장 큰 기술적 도전은 메모리가 ‘장기 정체성에 필요한 미세한 외형 단서’를 실제로 보존하도록 학습 신호를 설계하는 것입니다. Memento는 목표 주체의 외형을 직접 시각 프롬프트 없이도 메모리에서 재구성하도록 하는 TM2I 복원 과제를 추가해, 메모리가 얼굴/의상/체형 등 안정 외형 정보를 인코딩·검색하게 강제합니다. 동시에 reconstruction이 필요로 하는 장기 근거와 next-shot 생성에 유리한 단기 근거가 같은 경로에서 경쟁하지 않도록 dual-query(스토리 조건 쿼리 vs 샷 조건 쿼리)로 분리하고, 대명사 없는(subject-aware) 캡션 파이프라인으로 정체성 라벨 모호성을 줄입니다.

- **Empirical Impact**: 실험에서 Memento는 장기 subject consistency와 샷 간/씬 간 정합성에서 SOTA 성능을 보이며, 수치로는 inter-shot·inter-scene 정체성 점수가 경쟁 방법을 앞섭니다. 동시에 semantic 및 배경 일관성도 개선되어 스토리 흐름과 장면 안정성이 함께 좋아진다고 보고합니다. 사용자 연구에서도 cross-shot 일관성·프롬프트 따름·미적 품질에서 전반적으로 우선 선택을 받았고, 5분급 장편 생성 예시와 ‘나이 일관성(같은 인물의 연령 변화)’ 같은 확장 능력도 제시됩니다.



### Giving AI a Headache: Acoustic Adversarial Attacks to Computer Vision Applications (https://arxiv.org/abs/2606.14658)
Comments:
          9 pages, 7 figures, SPIE Defense + Security

- **Prior Approaches**: 기존의 음향 기반 공격 연구는 주로 초음파(20kHz 초과)로 카메라 센서/안정화 장치를 흔들어 짧은 거리에서만 효과를 내는 데 초점이 있었다. 또 시각적 적대 공격(예: 적대 패치, 위장 패턴, 광(光) 기반 교란)은 영상 입력 자체를 조작하므로 물리 세계에 그대로 적용하기 어렵거나 탐지가 쉽다는 한계가 있다. 결론적으로 “현실에서 가능한 주파수 범위”와 “더 긴 전파 거리의 저주파 영향”은 충분히 탐구되지 않았다.

- **Core Contribution**: 이 논문은 저주파 음향(가청 대역, 20kHz 미만)으로 카메라-센서-비전 파이프라인을 물리적으로 교란해 객체 탐지 성능을 떨어뜨리는 공격을 체계적으로 다룬다. 특히 기계적 공진을 이용해 안정화가 완비되지 않은 카메라에서도(전용 안정화 하드웨어 부재 포함) 탐지 오류(오분류·검출 누락·허위 탐지)가 발생함을 보인다. 또한 단순 성공 여부를 넘어, 이미지/객체 특징 수준에서 어떤 유형의 붕괴가 나타나는지 분석한다는 점이 핵심이다.

- **Technical Challenges**: 저주파에서는 전파는 유리하지만 실제로 어떤 주파수에서 카메라 하우징/렌즈/센서가 공진하는지 찾아내는 것이 관건이다. 논문은 주파수 스윕으로 “공진 대역”을 먼저 탐색한 뒤, 함수발생기(5Hz~30kHz)와 스피커로 해당 주파수를 연속 인가해 짧은 동영상 실험을 구성했다. 이후 YOLOv11 객체 탐지의 검출 결과를 바운딩박스 단위 평균 RGB/면적, 신뢰도, 정확도 등으로 정리해 주파수-의존적 오류 양상을 분해해 관찰했다.

- **Empirical Impact**: 실험 결과, 20~30Hz 및 155~180Hz에서 기준선 대비 검출률이 거의 10% 감소했으며, 신뢰도도 공진 주파수에서 평균 7% 이상 떨어졌다. 오류 유형도 오분류, 억제(검출 누락), 허위 탐지로 구체화되며, 프레임 지터·블러·기하학적 왜곡 같은 시각적 열화가 동반됐다. 저주파 음향 공격이 모델 파라미터/입력 접근 없이도(하드웨어 및 방향성 음원은 필요하지만) AI CV 시스템의 물리적 캡처 단계에서 실패를 유발할 수 있다는 점은 현장 보안과 견고성 연구에 중요한 경고가 된다.



### HPSv3++: Scaling Reward Models Across the Full Spectrum of Diffusion Model Capabilities (https://arxiv.org/abs/2606.14657)
- **Prior Approaches**: 기존 보상모델(Reward model)은 ImageReward, PickScore, MPS, HPSv3 등처럼 대체로 단일 스코어의 정적(무조건) 모델링에 머물렀다. 특히 HPSv3 같은 모델은 과거 T2I 모델의 사전(미리) 라벨링된 데이터로 학습돼, 생성기의 성능이 전면(frontier)으로 바뀌거나 RL 반복(iteration)에 따른 분포가 이동하면 점수 보정이 흔들린다.

- **Core Contribution**: 이 논문은 HPSv3++를 제안해, 보상모델이 생성기 ‘능력(capability)’과 RL ‘반복 단계’ 변화까지 함께 인식하도록 확장한다. 또한 기존 HPSv3의 유효한 인간 선호 지식을 유지하면서도, 더 넓은 capability–iteration 스펙트럼에서 동적으로 보정(calibration)되는 조건부 보상모델로 바꾼 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 전면 생성기 분포로 확장할 때 발생하는 성능 격차와 (2) RL 롤아웃 분포 이동에 따른 보상모델의 오정렬을 동시에 막는 것이다. 이를 위해 HPDv3++(212K, 텍스트 충실도·미적 품질 이중차원) 데이터로 새로운 분포 신호를 주고, 2단계 학습에서 Stage 1은 data-aware Orthogonal Gradient Descent로 망각을 줄이며, Stage 2는 capability–iteration 조건 신호(FiLM)와 표준편차(std) 기반 비지도 가이던스로 분별력과 보정력을 함께 강화한다.

- **Empirical Impact**: 실험에서 HPSv3++는 선호 예측 정확도에서 HPSv3 대비 HPDv3 9.8%p, GenAI-Bench 5.5%p 향상 등 최신 벤치마크에서 우수한 성능을 보였다. 또한 T2I RL 학습에서 Flow-GRPO로 SDXL, FLUX.1-dev, Qwen-Image 등 서로 다른 생성기 전반에 대해 GenEval 등 지표가 일관되게 개선됐고, 인간 선호 사용자 평가에서도 HPSv3++ 기반 최적화가 HPSv3 대비 77.5% 승률을 기록해 인간 선호 정렬이 더 잘됨을 입증했다.



### Improving Lunar Topography with Deep Learning Schrödinger Bridges (https://arxiv.org/abs/2606.14638)
- **Prior Approaches**: 행성 지형의 초해상도는 기존에 DEM(예: LOLA 기반)의 해상도 한계를 보완하기 위해 광학 영상 같은 추가 관측을 활용해 왔습니다. Shape-from-Shading(명암 기반 형상 복원, SfS)은 광학의 밝기 정보를 이용하지만, 조명·시점 각도와 파라미터 튜닝에 민감해 유연성이 낮고 대면적 적용이 비싸다는 한계가 있습니다. 또한 GAN·정규화 흐름·확산 기반 SR 연구가 있으나, 지형 초해상도에 Schrödinger Bridge(SB)를 직접 적용한 사례는 드뭅니다.

- **Core Contribution**: 이 논문은 저해상도 지형과 목표 해상도에서의 광학 맥락을 연결하는 생성모델로, 확산 기반 Schrödinger Bridge(SB) 프레임워크를 이용한 달 지형 초해상도 방식을 제안합니다. 핵심은 저해상도 DEM을 “순수 잡음”이 아니라 시작 조건으로 두고, 광학 영상 제약을 함께 반영해 고해상도 분포를 샘플링하는 조건부 SB를 학습하는 것입니다. 이를 통해 재구성 결과의 픽셀 단위 불확실성까지 함께 산출할 수 있는 확장성을 노립니다.

- **Technical Challenges**: SB로 조건부 역문제를 학습·생성하려면, (1) SB의 경로 분포를 정확히 반영하면서 (2) 학습 시에는 경계조건 쌍에서 샘플링 가능한 형태로 계산을 구성해야 하는 문제가 있습니다. 저자들은 이미지-투-이미지 SB의 학습 트릭을 채택해, 경계쌍(x0, xT)에서 필요한 확률분포를 해석적으로 다루는 방식으로 확산 모델 학습과 유사한 목적함수(Score Matching 계열)를 가능하게 했습니다. 또한 VAE의 잠재공간에서 DiT(확산 트랜스포머)를 학습해 고차원 지형·영상의 계산 부담을 줄이는 구현 전략을 사용합니다.

- **Empirical Impact**: 학습은 렌더링한 달 지형 데이터와 Lunar Reconnaissance Orbiter Narrow Angle Camera(NAC)를 모사한 광학 영상으로 수행하며, LOLA DEM을 20 mpp로 복원하도록 320 mpp 입력을 16배 초해상도하는 설정을 사용합니다. 저자들은 이 접근이 유연한 초해상도 재구성을 제공하면서, 재구성의 픽셀 단위 불확실성도 함께 제공함을 보였다고 정리합니다. 다만 현재는 주로 시뮬레이션 기반의 실험 단계이므로, 실제 광학·지형 조건 전반에서 기존 SfS와 더 직접 경쟁하려면 추가 연구가 필요하다는 점도 함께 언급됩니다.



### SED:Lightweight Saliency prediction for Event-based data via Distillation (https://arxiv.org/abs/2606.14631)
- **Prior Approaches**: 이벤트 기반 시선/주목도(saliency) 예측은 기존에 SNN 기반(뉴로모픽) 접근이나 수작업 특징 기반으로 시도됐지만, 빠른 움직임과 센서 잡음 같은 어려운 상황에서 성능이 제한적이었습니다. 이후 트랜스포머 기반 딥러닝(SEST)이 정확도를 끌어올렸지만, 파라미터 4,500만·메모리 180MB 수준으로 엣지 배포에는 부담이 큽니다. 또한 작은 모델을 일반 학습만으로 만들면 합성 데이터와 실제 이벤트 사이의 분포 변화에 쉽게 과적합된다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 이벤트 기반 주목도 예측을 위한 초경량 네트워크 SED를 제안하며, 핵심은 Depthwise Spatio-Temporal Block(DSTconv)로 3D 연산을 극도로 효율화하는 것입니다. 여기에 지식 증류(knowledge distillation)를 결합해, 학생 모델이 정확도뿐 아니라 합성→실제 도메인 전이 같은 강건성까지 확보하도록 설계했습니다. 그 결과, 거대 교사 모델을 대체하되 성능을 유지/개선하면서 모델 크기를 극단적으로 줄이는 것을 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 이벤트 데이터의 시간 정보를 충분히 활용하면서도 3D 컨볼루션의 비용을 감당 가능한 수준으로 낮추는 것입니다. 저자들은 3D depthwise separable의 depthwise 단계를 공간(depthwise spatial)과 시간(depthwise temporal)으로 분해해 계산·파라미터를 줄이는 DSTconv를 구성했고, 포인트와이즈(pointwise) 연산을 후반으로 지연해 추가 비용을 억제했습니다. 또 합성/실제 분포가 다를 때 작은 모델이 학습 분포를 넘어 무너지는 문제를, SEST 교사로부터의 output-level 증류와 과제 손실을 함께 최적화하는 방식으로 완화했습니다.

- **Empirical Impact**: SED는 N-DHF1K와 N-UCF Sports에서 교사 및 기존 이벤트 기반 방법들을 상회(또는 대부분 지표에서 동등/우위)하며, 모델 크기는 180MB→0.32MB(562배 감소), 파라미터는 45M→81k(554배 감소)로 압축됩니다. 연산 지연도 GPU에서 5.84ms(교사 32.9ms), CPU에서 38.89ms(교사 117.6ms)로 크게 줄어 엣지 실시간 적용 가능성을 강화했습니다. 특히 N-UCF Sports만으로 증류한 학생이 더 큰 N-DHF1K로도 잘 전이되며, 합성→실제 이벤트(EBSD)에서도 증류 모델은 교사조차 능가하는 반면 기준선(처음부터 학습)은 전이 붕괴를 보였다는 점에서 의미가 큽니다.



### StereoGeo: an end-to-end stereo camera calibration method (https://arxiv.org/abs/2606.14619)
Comments:
          5 pages, 1 figure, accepted at the 34th European Signal Processing Conference (EUSIPCO 2026)

- **Prior Approaches**: 기존 스테레오(및 단일) 카메라 캘리브레이션은 체커보드 같은 패턴을 쓰는 기하 기반이 정확하지만, 잡음·패턴 검출 오류에 민감하고 촬영 조건에 제약이 큽니다. 학습 기반은 이를 완화해 단일 이미지에서 GeoCalib처럼 내재변수와 중력 방향을 예측하나, 대개 단일 카메라 설정에 치우쳐 스테레오의 상대 외재변수까지는 제한적으로 다룹니다. 스테레오 학습 접근(예: UGCL)은 스테레오 쌍에서 좌우 카메라 내재변수가 같다고 가정하거나, 카메라별 중력 방향을 명시적으로 추정하지 않아 실제 장비 변동을 반영하기 어렵습니다.

- **Core Contribution**: StereoGeo는 GeoCalib을 스테레오로 확장해, 좌우 카메라 각각의 초점거리와 중력 방향(roll·pitch 분해)을 예측하고 두 카메라의 상대 외재변환까지 한 번에 추정하는 end-to-end 프레임워크를 제안합니다. 특히 좌우 카메라의 내재변수가 서로 다를 수 있음을 전제로 하며, 캘리브레이션 패턴이나 특징 매칭 없이도 스테레오 캘리브레이션을 수행합니다. 또한 예측된 관점장(perspective field)을 미분가능한 최적화로 다듬어 기하적 정합성을 확보하는 점이 핵심입니다.

- **Technical Challenges**: 문제의 핵심 난이도는 (1) 내재변수·중력방향·상대 자세를 함께 추정하면서 (2) 패턴/매칭/다중 뷰 제약 없이 (3) 비선형 카메라 모델에 안정적으로 최적화를 걸어야 한다는 점입니다. StereoGeo는 좌우 이미지를 독립적인 듀얼 브랜치로 처리해 단안 캘리브레이션 예측을 흔들지 않게 하되, 상대 자세 추정에는 스테레오 융합 모듈을 제한적으로 사용합니다. 이어서 Confidence map으로 가중치를 부여하고 Levenberg–Marquardt(Levenberg–Marquardt) 최적화를 통해 초점거리와 중력 방향을 미분가능하게 정제하며, 그 과정이 학습에 역전파되도록 구성해 전체 파이프라인을 공동 학습합니다.

- **Empirical Impact**: 실험에서 StereoGeo는 단안 케이스에서도 GeoCalib 대비 경쟁력 있는 내재변수·중력 추정 정확도를 보이면서, 스테레오 설정에서는 UGCL 등 기존 방법보다 외재(상대 자세) 추정 성능이 더 좋게 나타납니다. KITTI 및 합성 벤치마크에서 특징 매칭+RANSAC 기반 기하 파이프라인과 비교할 때, 회전은 유사한 수준의 정확도를 보이면서 표준편차가 크게 낮아 추정 일관성이 향상됩니다. 결정적으로 기존 기하법은 축척 불확실성 때문에 메트릭 번역을 회복하기 어려운 반면, StereoGeo는 메트릭 translation까지 복원해 스테레오 비전 응용에 바로 연결되는 실용성을 입증합니다.



### S$^2$COPE: Self-Supervised Concept Discovery via Preference Learning (https://arxiv.org/abs/2606.14586)
- **Prior Approaches**: 기존 연구는 자기지도학습으로 대규모 데이터에서 강한 표현을 얻지만, 결과가 고차원 벡터로 남아 해석이 어렵다는 문제가 있었다. 반대로 개념병목모델은 언어 기반으로 투명성을 제공하지만, 인간이 라벨 또는 개념어를 촘촘히 정의·정제해야 하는 비용 병목에 자주 막힌다. ‘라벨-프리’로 불리는 방법도 개념 후보를 고정된 VLLM이 1회 생성한 뒤 클래스 라벨에 기반해 필터링하거나, 냉동 표현을 분해해 발견 범위를 제한하는 한계를 보인다.

- **Core Contribution**: 이 논문은 VLLM을 정적 특징 추출기로 쓰지 않고, 이미지로부터 개념어를 스스로 제안·검증·강화하는 자기지도 루프를 end-to-end로 학습하는 S2COPE를 제안한다. 핵심은 시각 불변성(같은 인스턴스의 변환에는 유지, 서로 다른 인스턴스에서는 구별)을 선호학습 형태의 보상으로 모델에 주어, 라벨 없이도 도메인 특화의 구조화된 시각 개념을 발견하게 하는 것이다. 또한 이 개념 발견을 DPO(Direct Preference Optimization) 기반의 자기지도 선호 최적화로 VLLM 백본에 직접 반영해, 생성-필터링 분리 파이프라인의 한계를 줄인다.

- **Technical Challenges**: 가장 큰 기술적 난점은 ‘언어로 제안된 개념’이 실제로 해당 이미지의 안정적 시각 속성을 설명하는지, 그리고 배경 잡음을 포함하지 않는지 정량화하는 것이다. S2COPE는 CLIP을 고정된 크로스모달 평가기로 두고, 증강된 여러 뷰에서 같은 개념이 일관되게 정렬되는지(불변성)와 배치 내 다른 이미지에는 덜 정렬되는지(특이성)를 함께 갖는 크로스모달 대조 보상을 설계한다. 이 보상으로 생성된 개념들 사이의 선호 쌍을 구성해 DPO로 VLLM의 생성 정책을 업데이트함으로써, 외부 인적 판단 없이도 개념 제안 품질이 반복적으로 개선되게 한다.

- **Empirical Impact**: iNaturalist 미니(무라벨)에서만 학습한 뒤 자연·의료·물리 8개 데이터셋으로 전이 평가했을 때, S2COPE는 베이스 VLLM 대비 평균 약 16포인트, 최대로는 테스트 top-1 정확도에서 24포인트 절대 향상을 보였다. 특히 라벨 사전이 약한 특수 도메인(예: BloodMNIST, OrganCMNIST, Gravity Spy)에서 개선 폭이 커, 단순 사전 필터링이 아닌 일반화 가능한 개념 발견 메커니즘임을 시사한다. 또한 10명 사용자 연구에서 자율 발견된 개념 목록이 더 물리적으로 유의미하고 해석 가능하다는 선호가 평균 96.41%로 확인되었다.



### A Qualitative Review of GenAI-Based Methods for Data Generation and Augmentation in Industrial Computer Vision Applications (https://arxiv.org/abs/2606.14578)
Comments:
          Accepted to Computing Conference 2026

- **Prior Approaches**: 기존 접근은 현업에서 신뢰할 만한 예측 동작을 보장하기 위해 ‘예측 가능성’을 뒷받침할 데이터베이스를 쌓아야 하지만, 산업 현장에서는 초기 데이터 확보가 쉽지 않다고 지적합니다. 이를 해결하려고 활성학습(Active Learning)으로 배포 중 데이터를 반복적으로 늘리지만, 성능이 흔들리는 순간 사용 신뢰가 손상돼 한 번 잃으면 회복이 어렵다는 문제가 드러납니다. 결국 데이터베이스도 앱도 모두 제때 성숙하지 못하는 ‘치킨앤에그’ 딜레마가 발생합니다.

- **Core Contribution**: 이 논문은 초기 활성학습 단계의 데이터 급상승(data ramp-up)을 더 강하게 만들기 위해, 생성형 AI 기반 데이터 생성·증강 방법을 중심으로 최신 기술들을 정리합니다. 특히 산업용 비전 분류 분류(classification) 유스케이스에 적용할 때의 적응 가능성을 살펴보고, “자동 데이터 급상승” 가능성을 긍정적으로 검토합니다. 동시에 이러한 자동화가 항상 통하는 것이 아님을 경고하며, 원천 환경과 산업 환경 사이의 맥락 차이를 핵심 쟁점으로 둡니다.

- **Technical Challenges**: 가장 큰 기술적 도전은 생성·증강이 학습 환경(source)에서 정의된 맥락과 산업 현장(target)의 맥락이 다르다는 ‘도메인 불일치’입니다. 구체적으로는 자연어로 정의되는 상황(context)과 대상 객체의 특징(object characteristics)이 달라지면, 생성 데이터가 예측 가능성 향상으로 바로 연결되지 않을 수 있습니다. 논문은 GenAI를 통한 자동 급상승 시도에 잠재력이 있으나, 이러한 맥락·특징 불일치를 어떻게 줄일지가 관건임을 정리합니다.

- **Empirical Impact**: 정량 결과는 초록만으로는 제한적이지만, 산업 분류 문제에 대해 GenAI 기반 데이터 급상승이 적용될 여지가 있음을 실무 관점에서 제시합니다. 동시에 도메인 불일치가 실제로 신뢰 확보를 흔들 수 있다는 점을 전면에 내세워, 단순 자동화가 아닌 현장 맞춤 검증의 필요성을 강조합니다. 이 논문은 초기 데이터 축적 단계에서 신뢰를 지키는 연구 방향을 ‘생성형 데이터 전략’으로 확장시키는 데 의미가 있습니다.



### NEST3D: A High-Resolution Multimodal Dataset of Sociable Weaver Tree Nests (https://arxiv.org/abs/2606.14562)
Comments:
          14 pages, 4 figures. Dataset available at this https URL

- **Prior Approaches**: 기존 연구들은 사회성 직조새 둥지를 관찰·현장 측정 중심으로 다뤄왔고, 원격탐사 분야에서도 둥지를 주변 식생의 일부로 취급하는 경우가 많았습니다. 그 결과 둥지의 3차원 구조를 세밀하게 분리·학습할 수 있는 공개 라벨 데이터가 부족했습니다. 2D 중심 데이터나 라벨 없는 3D 재구성은 감독 학습 및 정밀 분할 벤치마크로 쓰기 어렵다는 한계도 있었습니다.

- **Core Contribution**: 이 논문은 NEST3D라는 공개 멀티모달 데이터셋을 제안해, 둥지를 ‘나무(tree)·풀(grass)·둥지(nest)’의 독립 클래스로 구분하는 3차원 분할 학습 기반을 제공합니다. RGB 이미지 27,945장, 멀티스펙트럼(녹색·적색·레드엣지·NIR) 111,780장, 약 7.81억 포인트의 3D 포인트클라우드와 전문가 라벨(3개 클래스)을 104개 둥지-보유 나무 장면으로 구축했습니다. 이를 통해 둥지 부피 추정, 종 보전 등 생태 응용과 3D 재구성·분할·분류 연구를 동시에 추진할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 둥지가 불규칙하고 얇은 가지·식생과 강하게 가려지는 3차원 기하 문제, (2) 가시 스펙트럼에서 둥지가 나무 목질과 스펙트럼적으로 비슷해 분리가 어려운 문제, (3) ‘둥지’가 극도로 희소한 클래스 불균형(long-tail)입니다. 논문은 포토그래메트리 기반 3D 재구성으로 밀집 포인트를 만들고, 전문가가 포인트 단위로 라벨을 부여해 감독 학습이 가능하게 했습니다. 벤치마크 실험에서는 PT-v3, RandLA-Net, KPConv를 통일된 프로토콜로 비교하면서 클래스 가중 손실과 Lovász loss 등 불균형 대응까지 적용해 차이를 체계적으로 드러냈습니다.

- **Empirical Impact**: 실험 결과 PT-v3가 테스트에서 mIoU 86.35%(OA 96.42%)로 가장 높은 성능을 보였고, 둥지(nest) IoU는 69.99%까지 유지했습니다. 반면 RandLA-Net은 둥지 IoU가 17.98%로 크게 떨어졌고, KPConv는 IoU 0.00%로 다수 클래스 붕괴(대부분 ‘풀’/‘grass’ 예측)에 가까운 양상을 보였습니다. 이는 구조적으로 작은 복잡 대상과 극심한 클래스 불균형에서 아키텍처별 성능 격차가 크다는 점을 명확히 보여 주며, 앞으로 멀티스펙트럼 융합·특화 샘플링 같은 방향성을 제시하는 도전적 기준선 역할을 합니다.



### Visual Quality Score Assessment of Large White Goods in Remanufacture with Multi-View Deformable-DETR (https://arxiv.org/abs/2606.14556)
Comments:
          Accepted to GCSM 2026

- **Prior Approaches**: 재제조(리맨ufacturing)에서 제품의 시각 품질 평가는 주로 수작업 검사에 의존해 왔고, 자동화 연구는 결함 탐지·분할 중심이라 고장(기능)과 외관(미용) 변이를 구분하거나 가격 산정용 ‘연속 점수’로 확장하기 어렵다는 한계가 있습니다. 또한 멀티뷰를 쓰더라도 단일/저수준 출력 위주이거나, 라벨(정답) 요구량이 커서 대규모 현장 적용이 막혔습니다.

- **Core Contribution**: 이 논문은 Deformable-DETR(변형형 DETR) 기반 멀티뷰 프레임워크로 여러 시점의 정보를 통합해 ‘자동 품질 점수(continuous quality scoring)’를 산출하는 방법을 제안합니다. 아울러 부족한 라벨 환경을 고려해 자기지도학습(Self-Supervised Learning, SSL) 사전학습 후 전문가 점수로 지도 미세조정을 수행하며, 결정 근거를 히트맵 형태로 설명할 수 있게 했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 고해상도 멀티뷰에서 작은 결함을 정밀하게 포착하고 (2) 뷰 간 중복 정보를 품질 점수로 효과적으로 융합하며 (3) 라벨이 제한된 상황에서 과적합을 줄이는 것입니다. 논문은 Deformable-DETR의 다중 스케일 변형 어텐션으로 관심 영역을 세밀 추출하고, 뷰별 [CLS] 토큰과 보조 토큰을 설계해 학습된 쿼리가 뷰별 특징을 찾아오게 했으며, frozen SSL 백본(DINOv2류)을 고정하고 neck만 학습해 데이터 효율을 높였습니다.

- **Empirical Impact**: 산업용 멀티뷰 데이터셋(KIKERP)에서 MAE 기준으로 모델 성능이 단계적으로 개선되며, 최종 MV Deform + SSL이 기존 뷰 독립 멀티뷰 분류/회귀 기반 접근 대비 점수 오차를 더 크게 낮췄습니다. 또한 선형 프로젝션 기반 관심 영역 히트맵을 통해 결함 또는 품질 관련 영역에 근거한 설명 가능성을 제공해, 재제조 라인의 대규모·투명한 검사로 확장될 잠재력을 보여줍니다.



### Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner (https://arxiv.org/abs/2606.14555)
- **Prior Approaches**: GAP 뒤 선형 분류기는 이미지 수준 라벨만으로 학습되며, 평가도 Top-5처럼 전역 지표가 중심이라 공간 근거가 가려진다. CAM/gradient 기반 설명은 질의한 특정 클래스의 열지도를 만들 수 있지만, 다중 객체가 섞인 장면에서 “어떤 물체가 근거였는가”를 분리해 보긴 어렵다. 또한 MIL 관점의 기존 약지도 학습은 주로 학습 목표나 집계 규칙을 바꾸는 데 초점을 두고, 표준 분류기가 내재한 공간 근거의 진단 가능성은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 GAP+선형 헤드 구조가 클래스 로짓을 공간 피처 그리드 위의 “점별 읽기(dense readout)”로 정확히 분해할 수 있음을 보인다. 이를 MIL(다중 인스턴스 학습)으로 재해석해, 이미지 수준 예측이 틀리더라도 공간에는 정답 클래스 근거가 남아 있을 수 있음을 체계적으로 정리한다. 이후 dense readout을 단순한 사후 진단 도구로 활용해, GAP이 평균화하며 숨기는 공간 단서를 추출한다.

- **Technical Challenges**: 핵심 기술 과제는 “공간 인스턴스”를 정의한 뒤, 점별 선형 헤드 적용이 의미 있는 공간 클래스 점수를 준다는 수학적/실험적 정당성을 확보하는 것이다. 논문은 CNN/ViT에서 얻는 마지막 공간 피처를 인스턴스로 보고, 선형 헤드의 점별 적용 전후의 등가성으로 공간 점수맵을 구성한다. 독립 인스턴스 가정은 CNN의 중첩 수용영역과 ViT의 셀프어텐션 때문에 엄밀히는 성립하지 않지만, 실제 진단 효과가 일관되게 나타나도록 실험 설계를 보완했다.

- **Empirical Impact**: ImageNet 및 occlusion 실험에서 전경 영역에서의 탐지율이 90% 이상으로 높게 나타나, GAP이 가려도 공간 근거는 복원됨을 확인했다. ImageNet-A에서는 이미지 수준 정확도가 급락해도 공간 단서 기반 탐지는 상대적으로 잘 유지되어, 실패가 “집계(평균)로 인한 희석”과 연결됨을 시사한다. MS-COCO에서도 동결된 자기지도 백본에 선형 헤드를 학습해 얻은 공간 로짓이 전경에서 정답 범주 근거를 복구했으며, 합성 다중 객체 실험은 단일 라벨 감독만으로도 이미지 내부의 정답 공간 점수를 형성할 수 있음을 직접 입증한다.



### A Lightweight Fiducial-Based Pipeline for 3D Hyperspectral Mapping of ex-vivo Lumpectomy Specimens (https://arxiv.org/abs/2606.14534)
- **Prior Approaches**: 기존 HSI 기반 절제연(마진) 평가는 의심 영역을 주로 2D 스펙트럴 맵에서 판별해, 실제 수술 의사결정에 필요한 3D 위치 정밀화가 부족했다. 냉동절편 같은 intraoperative 대안은 정확도나 접근성, 혹은 전체 표면을 다루는 데 필요한 처리 시간이 제한되는 경우가 많았다.

- **Core Contribution**: 이 논문은 단일 top-down HSI 획득과 RGB 기반 3D 재구성을 결합해, 조직 표면의 각 스펙트럼을 3D 좌표에 정합한 ‘3D hyperspectral point cloud’를 자동 생성한다. 특히 ArUco 마커를 공통 기준으로 사용해 보정(calibration)과 크로스 모달 포즈 추정 없이도 RGB-3D와 HSI를 정렬한다.

- **Technical Challenges**: 핵심 난제는 HSI에서 찾은 2D 좌표를 3D 형상 위에 정확히 옮기는 것이다. 저자들은 (1) deep learning SfM(MASt3R-SfM) 결과를 ArUco 코너 중심의 pseudo bundle adjustment로 안정화하고 metric 스케일을 고정한 뒤, (2) HSI 카메라 포즈를 복원하지 않는 2D-2D planar homography로 HSI 픽셀을 3D 깊이 맵의 표면점으로 조회한다.

- **Empirical Impact**: ex-vivo 유방 절제 표본 2개(SB019, SB020)에서 3D 정합 오차 중앙값이 1mm 미만, 2D 재투영 오차는 0.02mm 미만을 보고했으며, 샘플당 총 처리 시간도 4분 이내로 제시된다. 입력 RGB 뷰를 일부 줄이면 계산 이득은 제한적인 반면 3D 오차가 급격히 커져, 임상 시간 내 신뢰도를 위해 충분한 뷰 수 유지가 필요함을 실험적으로 확인했다.



### Scratched Lenses, Shifted Depth: Passive Camera-Side Optical Attacks (https://arxiv.org/abs/2606.14504)
- **Prior Approaches**: 기존 물리적 적대 공격은 주로 장면을 바꾸는 장면-측(scene-side) 방식(패치·텍스처·표지물)이나 촬영 시점에 빛을 주입하는 능동 광학 방식(active optical)으로 발전해 왔다. 또 카메라-측(camera-side) 접근으로는 렌즈 위에 스티커를 붙여 분류에 통용되는 교란을 만드는 연구가 있지만, 이는 모든 프레임에 동일한 가시 패턴을 얹는 방식이라 은밀성·기하 인식 결합이 다르다. 결과적으로 “카메라 자체의 작은 손상”이 장면 조명 조건에 따라 기하 추론을 선택적으로 망가뜨릴 수 있는지에 대한 검증은 부족했다.

- **Core Contribution**: 이 논문은 위 문제를 “수동 렌즈-측 손상(passive lens-side damage)”이라는 새로운 위협으로 정식화한다. 핵심은 손상이 카메라 광학 경로에 고정돼 있되, 특정 장면 조건(밝은 점광원/반사광)이 들어올 때만 줄무늬(streak) 같은 구조적 광학 아티팩트를 만들어 깊이 및 기하 추정을 편향한다는 점이다. 이를 Scratch-induced Lens Adversarial Streak Hijacking(SLASH)로 구현하고, 단일 고정 스크래치가 다중 프레임에서 목표 깊이 방향을 지속·선택적으로 흔든다고 제시한다.

- **Technical Challenges**: 기여를 현실화하려면 (1) 물리 스크래치를 이미지 평면의 스트릭 아티팩트로 어떻게 사상할지, (2) 센서·ISP까지 완전 모델링하지 않고도 그럴듯한 모양(광도·대비·공간 지지)을 만들지, (3) 배포 후 스크래치를 매 프레임 재조정하지 못하는 제약 아래 장면 시퀀스 전반에서 한 설정이 통하도록 최적화해야 한다. SLASH는 스크래치를 픽셀 교란이 아니라 광학 공간의 “고정 결함”으로 두고, 얇은 렌즈(파라크시얼) 근사 기반의 기하 제약 사상으로 스트릭의 위치·방향·길이를 계산한다. 또한 가벼운 외형 합성기(블러 마스크와 방향성 흐림, 트리거-스크래치 거리 감쇠)를 써서 물리 정확도에 의존하지 않고 깊이 타깃 손실을 장면-특정 프레임 시퀀스에서 최적화한다.

- **Empirical Impact**: 실험은 단일안(monocular) 깊이 추정과 단일안 3D 물체 검출을 디지털·실물 환경에서 평가하며, 고정 스크래치 제약 하에서 깊이 추정의 방향성 오차가 최대 32% 상대 오차까지 도달함을 보인다. 또한 실험적 물리 결과가 실제 카메라 녹화로 전이되어, 모델의 자연 예측 기준선보다 훨씬 큰 깊이 편향이 재현됨을 확인했다. 이는 “겉보기엔 무해한 렌즈 손상”이 장면 트리거를 통해 잠재적 적대 채널이 될 수 있음을 드러내며, 물리적 강건성(광학 경로 무결성)에 대한 방어 가정의 재검토를 촉진한다.



### Value-order Decomposition for Generalist Anomaly Detection (https://arxiv.org/abs/2606.14475)
- **Prior Approaches**: 기존 시각 이상 탐지는 정상 데이터만으로 기준 패턴을 학습하거나(임베딩 거리·재구성·지식 증류 계열), 일반화 문제를 풀기 위해 새로운 클래스/도메인에 재학습·미세조정을 수행하는 방식이 많았다. Generalist Anomaly Detection(GAD)에서는 residual(잔차) 특징을 정규 기준에 맞추어 교차 도메인 일반화를 노렸지만, 이상 잔차가 객체 범주·결함 유형·데이터 도메인별로 여전히 얽혀 “세 가지 일반화 갭”이 남는 한계가 있었다. 또한 실존 이상 참조를 쓰는 방법은 데이터 가용성이 낮아 적용이 어렵고, 합성 이상 기반 접근은 합성 품질·다양성에 의존하는 문제가 있었다.

- **Core Contribution**: 이 논문은 residual 특징을 Value-Order Decomposition(VOD)으로 분해해, 객체/결함/도메인 갭에 덜 민감한 “value(값) 성분”과 갭별 구조를 담는 “order(순서) 성분”을 분리한다. 특히 이상 탐지에는 value 성분만을 핵심 표현으로 사용해 정상·이상의 정렬을 갭 전반에서 촉진하면서도 분리 가능성(정상 vs 이상)을 유지하는 것이 기여의 핵심이다. 여기에 더해, cut-and-paste 기반 간단한 합성 이상(reference)을 만들고 정상과 “합성-이상” 참조만으로 추론하는 프레임워크를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 residual 공간에서 이상이 세 가지 갭(객체 범주·결함 유형·도메인)별로 클러스터링되어 정렬이 잘 되지 않는다는 점이다. 이를 해결하기 위해 residual의 각 차원을 내림차순으로 정렬해 값 분포(크기/정도)는 value로, 차원 간 상대적 순위(구조/의미)는 order로 분리하고, 이상 판단에는 value만 사용해 갭 특이 정보를 억제하도록 설계했다. 또한 합성 이상이 실제 결함과 외형이 달라도 가치 성분에서 real 결함과 정렬될 수 있음을 전제로, A-Value projection처럼 “합성 이상 방향”에 대한 투영 기반 점수로 이상도를 계산하도록 구성했다.

- **Empirical Impact**: 다양한 산업 및 의료 벤치마크에서 VOD 기반 방법은 기존 기준선 대비 큰 폭의 성능 향상을 보이며 세 가지 일반화 갭(새 객체 범주·새 결함 유형—실제/합성 포함—새 도메인) 전반에서 강건한 일반화를 입증했다. 특히 합성 이상을 생성할 때도 diffusion 등 복잡한 생성 모델 없이 cut-and-paste 전략만 사용했는데, 이는 합성 데이터 품질 의존성을 줄이면서도 강한 일반화 성능을 낸다는 점에서 의미가 있다. 결과적으로, 실존 이상 참조 데이터가 부족한 현실 조건에서도 효과적으로 확장 가능한 일반자형 이상 탐지(GAD) 설계 방향을 제시했다.



### MooMIns -- Monocular 3D Reconstruction and Object Pose Estimation from Multiple Instances (https://arxiv.org/abs/2606.14389)
- **Prior Approaches**: 단안 RGB 영상에서 3D 복원과 6D 포즈를 동시에 구하는 문제는 깊이가 직접 관측되지 않아 본질적으로 부정확성이 크다. 기존 접근은 학습된 깊이/형상 사전지식이나 생성 모델에 의존해 성능을 내지만, 학습 분포 밖에서는 증거가 부족한 영역에서 기하를 “환각”하는 문제가 보고된다.
또한 6D 포즈 추정은 CAD 모델 같은 강한 선지식을 쓰는 모델 기반 방식이나, 추가 뷰/학습 또는 생성된 3D 모델을 요구하는 모델-프리 방식이 많아 단일 이미지의 다중 인스턴스 신호를 직접 활용하는 연구는 상대적으로 적다.

- **Core Contribution**: 이 논문은 ‘단일 카메라의 단일 다중 인스턴스 이미지(같은 객체가 서로 다른 자세로 여러 번 보이는 이미지)’라는 산업 현장의 현실적인 관측을 기하 제약으로 활용한다. 각 인스턴스가 동일한 형상·외관을 공유한다는 가정 아래, 동일 객체에 대한 암묵적 멀티뷰를 이용해 공통(정준) 3D를 복원하고 각 인스턴스의 6D 포즈를 동시에 추정한다.
이를 위해 MooMIns라는 새로운 Gaussian splatting 기반 방법을 제안하며, 고전적인 “여러 카메라에서 한 장면을 렌더링”하던 설정을 “한 카메라에서 여러 인스턴스 장면을 렌더링”하는 방식으로 뒤집는다.

- **Technical Challenges**: 핵심 기술 과제는 단일 이미지에 공통 렌더링 파이프라인을 적용하면서도, 인스턴스 간 가려짐·부정확한 초기화·제조 허용 오차·대칭/무질감(무텍스처) 같은 요인 때문에 최적화가 쉽게 퇴화(degenerate)될 수 있다는 점이다. 논문은 SAM3 기반 가시 인스턴스 분할을 시작값으로 쓰고, 이를 COLMAP의 수정된 Structure from Motion(SfM) 파이프라인에 넣어 초기 포인트클라우드와 인스턴스 포즈를 구한 뒤, 수정된 Gaussian splatting(Reflection-based SV 포함)으로 정련한다.
또한 ADC(Adaptive Density Control) 임계조건을 ‘매 반복에서 동일 이미지가 전체 인스턴스를 제공’하는 상황에 맞게 조정하고, 인스턴스 드롭아웃으로 여러 인스턴스의 외관이 정준 객체에 과적합되는 퇴화를 완화한다.

- **Empirical Impact**: MooMIns는 합성 및 실제 빈 피킹(bin-picking) 시나리오에서 검증되며, 학습에 사용되지 않은 새로운 객체에 대해서도 정확한 3D 복원과 신뢰할 수 있는 인스턴스별 포즈 추정을 보여준다. 특히 단안 학습 사전지식에 기대지 않고, 이미지 증거 기반의 기하 복원으로 환각을 줄이는 전략이 실험 결과에 반영된다.
산업용 저비용 단일 RGB 입력으로도 포즈·형상 추정이 가능하다는 점에서 로보틱스와 품질검사, 역공학 같은 분야에 실용적 임팩트를 줄 것으로 평가된다.



### IndustryBench-MIPU: Benchmarking Multi-Image Attribute Value Extraction for Industrial Products (https://arxiv.org/abs/2606.14383)
- **Prior Approaches**: 기존 연구는 멀티모달 대규모 언어 모델이 제품 이미지를 이해하는 능력을 다루더라도, 밸브·차단기처럼 방대한 기술 사양이 ‘여러 이미지에 흩어진 상태’에서 신뢰성 있게 복원되는 문제는 상대적으로 덜 탐구돼 왔다. 특히 사양표/명판/도면이 서로 다른 형태의 정보로 분산돼 있어, 표·문자 인식과 도면 추론, 용어 해독, 근거 통합이 한 번에 요구된다는 점이 기존 방식의 한계로 남았다.

- **Core Contribution**: 이 논문은 산업 제품 이해를 위한 첫 대규모 벤치마크인 IndustryBench-MIPU를 제안한다. 사양을 ‘속성-값 쌍(attribute-value pairs)’으로 구조화해, 사양표·명판의 텍스트 인식, 도면의 시각적 추론, 산업 용어의 도메인 지식, 그리고 여러 이미지에 흩어진 근거의 통합을 동시에 측정한다.

- **Technical Challenges**: 핵심 난제는 모델이 단일 이미지에서는 정밀도는 높게 유지할 수 있어도, 제품 단위로 누락 없이 완전한 사양을 재구성해야 하는 ‘완전성(completeness)’이 급격히 떨어진다는 점이다. 저자들은 이를 검증하기 위해 4,559개 제품·27,652개 이미지·18개 산업 범주에 대해 103,703개 주석을 다중 모델 합의와 3단계 품질 보증으로 구축했으며, 단일 이미지와 제품 단위 멀티 이미지 설정 모두에서 9개 MLLM을 비교 평가한다.

- **Empirical Impact**: 실험 결과 모델의 정밀도는 86~94%로 높지만, 제품 단위 속성 복원은 최고 49.9%에 머물렀고 단일 이미지에서 멀티 이미지로 확장하면 재현율이 15~34%p 감소했다. 즉, 단일 이미지 정확도가 아니라 멀티 이미지 완전성이 병목이며, 산업 문서·도면 기반의 신뢰 가능한 사양 추출 연구에서 통합 추론과 누락 방지에 대한 새로운 방향성을 제시한다. 데이터와 코드는 공개돼 재현과 후속 연구를 촉진한다.



### FLaRA: Predicting Future Latent Representations for Accident Anticipation (https://arxiv.org/abs/2606.14380)
Comments:
          Accepted at the 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC 2026)

- **Prior Approaches**: 대시캠 기반 사고 예측은 관측 프레임의 표현을 그대로 받아 충돌 확률로 직접 매핑하는 패러다임이 주류였습니다. 예를 들어 LSTM/그래프 신경망/트랜스포머로 시간적 단서를 누적하거나, 위험 손실 가중치를 시간에 따라 조절해 더 이른 경보를 노렸습니다. 그러나 이런 방식은 미래 장면의 전개를 명시적으로 시뮬레이션하지 않아, 예측 초기에선 정보 부족으로 성능이 흔들릴 수 있습니다.

- **Core Contribution**: FLaRA는 충돌 확률 예측을 관측 표현에서 바로 수행하지 않고, 미래의 잠재 표현(미래 latent representations)을 예측한 뒤 그 예측값으로 분류하는 “predict-then-classify” 전환을 제안합니다. 이를 위해 V-JEPA2(Video Joint-Embedding Predictive Architecture 2)에서 출발해, 맥락 프레임으로부터 장면의 향후 latent를 예측하는 구조를 사고 예측에 적용했습니다. 또한 사고 예측에서 미래 전개를 모델이 실제로 “맞춰보게” 만드는 학습 신호를 함께 설계했습니다.

- **Technical Challenges**: 핵심 난제는 예측된 미래 잠재 표현이 분류에 유리할 뿐 아니라, 실제 미래 장면 동역학과도 정합적이어야 한다는 점입니다. FLaRA는 분류 손실뿐 아니라 보조 재구성 손실을 추가해 예측된 미래 latent이 정답 미래 프레임의 latent에 가깝도록 Smooth-L1 기반으로 공동 최적화합니다(재구성-분류 다중목적 학습). 추론 시에는 미래 프레임을 볼 수 없으므로, 컨텍스트 윈도우만 입력으로 사용해 미래 latent을 예측하고 그 결과만으로 분류하도록 파이프라인을 고정했습니다.

- **Empirical Impact**: Nexar에서 FLaRA는 AP와 AUC 모두에서 최첨단 수준을 달성하며, BADAS 대비 약 4배 적은 학습 파라미터로 성능 우위를 보였습니다. DAD, DADA-2000, DoTA 같은 교차 도메인 검증에서도 추가적인 도메인별 미세조정 없이 경쟁력 있는 결과를 보여 미래 잠재 표현 예측이 일반화에 기여함을 시사합니다. 특히 경보 타이밍에서 더 긴 시간 여유(더 이른 구간)일수록 격차가 커져, 관측만으로는 부족한 초반 상황에서 미래 전개 예측의 가치가 실증되었습니다.



### Point Cloud Upsampling through Patch-based Frequency Superposition (https://arxiv.org/abs/2606.14355)
- **Prior Approaches**: 기존 점군 업샘플링은 크게 딥러닝 기반과 최적화 기반으로 나뉩니다. 딥러닝(PU-Net 등)은 성능이 좋지만 학습 데이터와 유사한 분포에서만 잘 작동하고, 기하를 왜곡(환각)할 위험이 있으며 블랙박스 성격 때문에 해석이 어렵습니다. 최적화 기반(EAR, LOP 계열)은 기하를 추정하지만 대체로 점-표면 거리(P2F) 같은 정밀도에서 한계가 있거나, 주파수 모델은 FSGU처럼 제한적 가정(예: 단일 함수 형태)에 의존합니다.

- **Core Contribution**: PUtPFS(Point Cloud Upsampling through Patch-based Frequency Superposition)는 점군을 패치 단위로 선택한 뒤, 해당 패치의 국소 표면을 “공간 주파수의 중첩(superposition)”으로 추정해 새 점을 배치하는 최적화 기반 방법입니다. 또한 주변 점들의 밀도가 낮은 영역을 순차적으로 시드로 삼아, 결과적으로 더 균일한 업샘플링을 목표로 합니다. 무엇보다 학습 데이터 없이도 동작하며 수학적으로 해석 가능한 절차로 구성되었다는 점을 핵심 가치로 제시합니다.

- **Technical Challenges**: 주요 난제는 (1) 패치를 안정적으로 분할·성장시키고 (2) 그 패치를 2D 함수로 투영해 (3) 주파수 기반 재구성의 정확도를 유지하는 것입니다. 논문은 패치 성장 시 점들의 공분산 고유값 비율 변화로 중단 기준을 세워 서로 다른 표면이 섞이는 문제를 줄이고, 큰 고유값 방향을 이용해 국소 평면을 만든 뒤 점을 투영해 signed distance를 함수로 재구성합니다. 이후 DCT(이산 코사인 변환) 기저에 대한 잔차 에너지 최소화 방식으로 주파수 선택적 근사를 수행하고, 공간·주파수 가중치로 저주파 중심의 안정적인 복원을 유도합니다.

- **Empirical Impact**: PU-GAN, PU1K에서 Chamfer 거리(CD)와 Hausdorff 거리(HD)는 딥러닝 수준과 비슷하거나 근접한 범위를 보이며, 특히 점-표면 거리(P2F)에서는 일관되게 최상(또는 최상위)을 달성했다고 보고합니다. 최적화 기반 비교에서도 PUtPFS가 EAR, FSGU보다 CD/HD 및 P2F 관점에서 우수하며, 시각적으로도 잡음이 줄고 표면을 더 잘 따르는 경향이 확인됩니다. 다만 인접한 두 표면이나 얇은 선 구조는 2D 표면 근사로 인해 제대로 복원하지 못할 수 있고, 점 배치의 완전한 균일성은 제한적이라고 밝힙니다.



### ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models (https://arxiv.org/abs/2606.14351)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: 기존 T2I(텍스트-이미지) 안전 대책은 데이터 필터링, 생성 후처리 안전 필터, 훈련-프리 지침, 그리고 UNet 가중치/어텐션을 수정하는 개념 지우기(unlearning)로 나뉜다. 다만 많은 개념 지우기 방법이 ‘위험 개념’뿐 아니라 악성 프롬프트에 포함된 ‘무해한 의미(예: 사람/인체 맥락)’까지 과도하게 지워 생성 유틸리티가 떨어지는 문제가 지적된다. 또한 이러한 기법들은 I2I(이미지-이미지) 환경에서 전이가 잘 되지 않거나(기존 T2I용 설정의 한계), 역프롬프트/공격에 취약할 수 있다.

- **Core Contribution**: 이 논문은 개념 지우기를 단순 제거가 아니라 ‘안전 의미 해석을 유지한 채 위험 생성만 최소화’하는 보상 최적화 문제로 재구성하고, 강화학습 기반 개념 지우기(CER)를 제안한다. Safety(안전) 보상과 과도 제거를 막는 alignment(정렬) 보상을 함께 설계해, 위험 콘텐츠는 억제하면서도 무해한 인간 중심 의미는 보존하도록 학습한다. 추가로 교차어텐션에 Safe Adapter를 도입해 텍스트 임베딩의 일부 토큰만 효율적으로 조절함으로써 과도한 지우기를 완화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 안전성 평가기가 제한된 개념만 포착할 때 생길 수 있는 ‘무의미한 출력’ 방지, (2) 위험 프롬프트에 섞인 안전 의미까지 같이 지워버리는 과도 제거 문제, (3) 교차어텐션 기반 방어를 우회하는 adversarial 프롬프트 위협에 대한 강건성이다. 논문은 이미지 기반 NSFW 안전 평가기 점수를 안전 보상으로 사용하되, BLIP 기반 이미지 캡션을 통해 안전한 설명으로 정렬 보상을 구성하고, 사람(옷 착용 등) 관련 보조 조건으로 인간 지향 출력을 유지하는 방식으로 타협점을 잡았다. Safe Adapter는 텍스트 임베딩의 뒤쪽 일부 토큰(예: 마지막 4개)에만 적용해 의미 훼손을 줄이면서 목표한 위험 토큰 영향만 분리·규제하도록 설계한다.

- **Empirical Impact**: 실험에서 다양한 데이터셋 및 10개 SOTA 개념 지우기 대비, 위험 콘텐츠 생성 억제(예: Nudity Removal Rate)와 무해 이미지 충실도(FID, CLIP 등) 사이의 균형이 더 우수하게 나타난다. 특히 red-teaming 도구에 대한 강건성에서 Ring-A-Bell, P4D, MMA 공격에 대해 최상위 성능을 보이며, NRR이 100%에 도달하는 설정도 보고된다. 또한 I2I 시나리오에서도 전이 성능이 더 좋고, 예술 스타일·객체 같은 일반 개념 지우기까지 확장 가능함을 실증하며 실사용 안전 편향을 줄일 잠재력을 보여준다.



### CausalMotion: Structured Physical Reasoning as Keyframe and Trajectory Guidance for Training-Free Video Generation (https://arxiv.org/abs/2606.14317)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 확산 기반 비디오 생성은 높은 시각 품질과 단기 시간적 일관성을 크게 개선했지만, 물리적·인과적으로 타당한 장기 상호작용(긴 호라이즌)에서는 한계를 보인다. 많은 방법이 데이터의 통계적 상관을 암묵적으로 학습하는 데 그쳐, 중간 상태 누락이나 순간적 위치 점프 같은 물리 불가능 현상이 발생하기 쉽다. 물리 제약을 시뮬레이터로 넣는 접근은 실제로는 계산·모델링 오버헤드가 크고, 단순 가정에 기대는 경우가 많다.

- **Core Contribution**: 이 논문은 학습 없이(inference-only) 동작에 인과 구조를 주입하는 프레임워크 CausalMotion을 제안한다. 핵심은 비전-언어 모델(VLM)이 텍스트 프롬프트를 인과적으로 일관된 키프레임 시퀀스와 물체 중심 모션 궤적으로 분해한 뒤, 이를 “소프트 제약”으로 확산 모델의 샘플링에 연결해 생성이 물리적으로 그럴듯하도록 유도하는 것이다. 이렇게 하면 모델 파라미터나 추가 감독 없이도 인과적 상태 전이와 물체 동역학을 명시적으로 모델링할 수 있다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘생성’을 ‘이유(추론)’에서 분리하되, VLM이 만든 중간표현(키프레임·궤적)과 확산 생성의 시간·공간 정합을 어떻게 맞추느냐였다. 논문은 (1) VLM으로부터 희소 키프레임을 만들고, (2) 세그멘테이션으로 초기 물체 박스를 잡아 Physical State Vector로 물체 상호작용을 추론하며, (3) 이를 촘촘한 궤적으로 보간·정렬(키프레임-궤적 간 IoU 기반 시간 할당)한 뒤, (4) 궤적/외관 앵커를 사용해 라텐트 공간에서 지역적 업데이트로 부드럽게 제약을 거는 방식으로 해결한다. 또한 확산 모델의 다운샘플 구조에 맞춰 궤적을 라텐트로 인코딩하고, 키프레임 인덱스에 따라 조건을 주입한다.

- **Empirical Impact**: PhyGenBench에서 평균 0.65로 기준선 대비 67% 개선되며, 특히 역학과 열(thermal) 범주에서 성능 향상이 두드러졌다. VBench에서도 82.52%를 달성해 물리 제약을 넣어도 시각적 품질이 크게 훼손되지 않음을 보여준다. 더 나아가 VLM 기반 저널 판정에서 물리적 그럴듯함·시간적 일관성·의미 정합이 모두 개선되어, 장기 역학이 필요한 시나리오에서 특히 효과적임을 확인했다.



### Pano3D: Unified 3D Reconstruction and Panoptic Segmentation (https://arxiv.org/abs/2606.14307)
Comments:
          Project page: this https URL

- **Prior Approaches**: 최근의 3D 피드포워드 복원 신경망(FRM)은 카메라 파라미터 없이도 unposed RGB로부터 조밀한 3D pointmap을 복원하는 데 성공했지만, 견고한 의미 이해까지 포함시키는 일은 여전히 미완의 과제로 남아 있었다. 기존 FRM-세그멘테이션 확장들은 2D 모델 피처에 의존하거나, 기하·의미를 별도 분기해 나중에 결합하거나, 무차별 군집/클러스터링 같은 휴리스틱 후처리가 필요했다. 특히 이런 방식은 다중 뷰 일관성과 마스크 경계의 정밀도에서 제약을 보일 수 있다.

- **Core Contribution**: 이 논문은 통합 프레임워크 Pano3D로 3D 복원과 3D 팬옵틱(semantic+instance) 분할을 단일 모델에서 함께 학습한다. FRM의 출력에 Mask2Former 계열의 set-based 마스크 디코더를 “쿼리 기반”으로 바로 붙여, 미분 불가능한 클러스터링이나 외부 2D 모델 없이도 인스턴스 마스크와 클래스 로짓을 직접 예측한다. 또한 기하 복원 디코더를 고정하지 않고 세그멘테이션과 함께 파인튜닝하여 재구성 능력을 유지하면서 의미를 학습하도록 설계했다.

- **Technical Challenges**: 핵심 기술 난점은 세그멘테이션 손실이 기하 디코더의 표현을 망가뜨릴 수 있다는 점이며, 이를 통제하는 학습 안정화가 필요하다. 논문은 (1) 기하 피처에서 시작해 의미까지 함께 포착하도록 어댑터와 쿼리 기반 마스크 디코더를 구성하고, (2) 기하 손실을 학습에서 계속 유지해 복원 충실도를 방어하며, (3) 세그멘테이션 그라디언트를 기하 디코더로 흘리기 전에 스케일링(gradient scaling)해 과도한 업데이트를 억제한다. 결과적으로 외부 일관성 손실이나 휴리스틱 후처리 없이도 멀티뷰 일관 마스크를 얻는 것이 목표가 된다.

- **Empirical Impact**: 실험에서는 ScanNet, ScanNet200, ScanNet++에서 3D 팬옵틱 분할 성능을 달성하며, 기존 대비 state-of-the-art를 보고한다. 또한 ablation으로 통합 학습이 팬옵틱 분할을 가능하게 할 뿐 아니라 기하 성능도 크게 훼손하지 않으며, 특히 파인튜닝 설정과 그라디언트 스케일링이 기하-의미 상호 이득에 중요함을 보여준다. 마지막으로 MUSt3R(온라인)과 Pi3(올투올) 같은 서로 다른 FRM 백본에도 동일한 팬옵틱 디코더를 적용할 수 있음을 통해 범용성까지 입증한다.



### What Drives Test-Time Adaptation for CLIP? A Controlled Empirical Study from an Update Perspectiv (https://arxiv.org/abs/2606.14299)
- **Prior Approaches**: 기존 Vision-Language Model(VLM)인 CLIP의 제로샷 인식은 성능이 강하지만 배포 환경의 분포 변화(distribution shift)에 취약했다. 이에 Test-Time Adaptation(TTA)이 CLIP에 적용되며, 테스트 시점에 프롬프트/어댑터를 학습하거나(파라미터 기반), 특징 캐시·통계 등을 축적하거나(상태 기반), 추론 단계에서 예측만 정제하는 방식(추론 기반)으로 문헌이 빠르게 확장됐다. 다만 어떤 구성 요소가 실제로 성능 향상을 만드는지, 어떤 종류의 분포 변화에서 안정적인지에 대한 체계적 이해는 부족했다.

- **Core Contribution**: 본 논문은 TTA4CLIP의 성과를 “최신 정확도” 경쟁에서 한 발 물러나, 어떤 요인이 적응을 이끄는지 통제된 실험으로 분석한다. 먼저 테스트 시점에 무엇을 업데이트하는지 기준으로 방법들을 파라미터 기반·상태 기반·추론 기반의 3가지 패러다임으로 통합 분류한다. 또한 TTABC라는 오픈소스 TTA 벤치마크를 제안해 20개+ 대표 방법을 표준화된 프로토콜로 평가하고, 분포 변화 유형 전반에서 비교 가능성을 높인다.

- **Technical Challenges**: 파라미터 기반 방법에서는 흔히 “더 강한 최적화”가 이득의 원인이라고 여겨졌지만, 본 연구는 그 효과가 제한적이며 테스트 시점 증거와 신뢰할 만한 무지도(unsupervised) 프록시가 핵심임을 보인다. 구체적으로 학습률과 적응 단계 수를 바꿔도 이득은 빠르게 포화되는 경향이 있고, 확장된 증거량(증강 뷰 수)과 신뢰 필터가 제공하는 품질이 성능을 크게 좌우한다. 또한 라벨이 없을 때 어떤 프록시(예: 엔트로피 최소화, MSP 등)를 쓸지의 문제가 되는데, 본 논문은 “정확도와 잘 정렬된 신뢰 신호”라면 형태 자체보다도 신호의 정합성이 중요하다는 해석을 제시한다.

- **Empirical Impact**: 통제 실험을 통해 파라미터 기반 적응의 이득은 대규모 최적화보다 테스트 증거의 양·질에서 주로 오며, 유사한 성능은 무겁지 않은 업데이트(교차/현재 샘플 기반 증거 활용, 경량 프로토타입 갱신 등)로도 달성 가능함을 보여준다. 더 나아가 “단 하나의 만능 TTA 패러다임”은 없으며, 선호되는 전략은 분포 변화의 성격에 따라 달라진다는 결론을 제시한다. TTABC는 평가 프로토콜과 기준을 표준화해 향후 연구가 어떤 가정이 언제 성립하는지 더 엄밀히 검증할 수 있는 기반을 제공한다.



### Pix2Pix-Hybrid: Structure-Guided Conditional Synthesis of Hajj Crowd Images with Multi-Channel Conditioning and Weak Attribute Supervision (https://arxiv.org/abs/2606.14297)
- **Prior Approaches**: 기존 군중 계수 연구는 기준이 되는 실제 데이터셋(예: ShanghaiTech, UCF-QNRF 등)에 크게 의존하지만, Hajj처럼 극단적 밀집·가림·조명 변동이 큰 도메인에서는 주석 부족과 도메인 격차로 일반화가 어렵다. 데이터 희소성을 보완하려는 증강은 대개 원본과 강하게 상관된 변형에 그쳐 장면 구조·맥락 다양성을 충분히 늘리지 못한다. 합성 데이터 생성에서도 일반 생성 모델은 고품질은 가능해도 모드 붕괴나 데이터 과적합 위험, 그리고 후속 계수 학습에 필요한 “구조 보존” 목표 정렬이 부족할 때가 많다.

- **Core Contribution**: 이 논문은 Pix2Pix 계열을 확장한 구조-가이드 조건부 GAN인 Pix2Pix-Hybrid(P2P-H)를 제안해, Hajj 장면의 지배적 기하(레이아웃)를 유지하면서 외형을 밀도·시간대 같은 약한 속성으로 다양화한다. 핵심은 무작위 잠재벡터로 자유 생성하는 모델이 아니라, 입력 조건을 엣지·그레이스케일과 밀도/시간대에서 “셀프 페어링”으로 만들고 RGB를 재구성·변환하도록 학습해 계수에 필요한 구조 충실도를 확보하는 데 있다. 또한 생성 품질을 위해 다중 해상도 PatchGAN 판별과 GAN 학습 안정화(지각 손실·특징 매칭·적응형 정규화/증강)를 함께 통합한다.

- **Technical Challenges**: Hajj 도메인은 (1) 주석이 거의 불가능하고 (2) 프라이버시 제약으로 데이터 수집·공개가 제한되며 (3) 고밀도 가림과 해상도/조명 편차가 커서 일반적인 조건부 GAN이 흐림·체커보드·학습 불안정에 빠지기 쉽다. 저자들은 공개 Hajj 영상에서 프레임을 모아 정제·익명화(얼굴 블러)하고, 밀도(낮음/중간/높음)와 시간대(아침/오후/밤)를 자동 약라벨로 추출해 수동 라벨 부담을 줄였다. 생성 모델은 U-Net 생성기 + 서로 다른 해상도를 보는 두 개의 다중 스케일 PatchGAN으로 전역 일관성과 국소 질감을 동시에 강제하며, 적응형 데이터 증강과 복합 손실로 제한 데이터에서도 학습 안정성을 높였다.

- **Empirical Impact**: 이 과정을 통해 9.93e2장의 실제 프레임 학습으로부터 1만 장 규모의 고해상도 합성 데이터 CrowdH를 구축했고, Pix2Pix 및 StyleGAN2-ADA 대비 구조 보존 관점에서 조건부 합성 품질이 개선되었다. 또한 CrowdH-Mix-469(실제 384 + 선택 합성 85)로 5개 계수 모델을 평가한 결과, 합성 데이터를 포함한 학습이 MAE를 모든 모델에서 낮췄으며 CSRNet에서 이득이 가장 컸다. 특히 합성 데이터의 선택성이 효과를 좌우함을 보여주면서, 프라이버시 민감 환경에서도 계수 성능을 실질적으로 끌어올릴 수 있음을 시사한다.



### A Robust Point Cloud Analysis Framework Inspired By Primary Visual Cortex (https://arxiv.org/abs/2606.14292)
Comments:
          12 pages, 2 figures, 7 tables

- **Prior Approaches**: 점군(3D point cloud) 분석은 기존에 기하 기반 수공 특징과 복잡한 전처리에 의존하거나, 복셀/멀티뷰로 투영해 2D CNN을 재사용하는 방식(간접)과 원시 점을 직접 처리하는 방식(직접)으로 발전해 왔다. 그러나 투영은 정보 손실을, 직접 모델(예: PointNet 계열)은 연산 구조의 한계로 인해 잡음·희소성·가림·공간 변형 같은 실제 부정합(corruption)에 취약하다. CNN 계열은 계산 비용·불안정한 경로 제어 부족 문제도 지적돼 왔고, 스파이킹 신경망(SNN) 역시 효율에는 강점이 있으나 복잡한 부정합에 대한 “표현 안정화” 메커니즘이 부족했다.

- **Core Contribution**: 이 논문은 시각 피질의 처리 영감을 받아 Dendritic-Connected Continuous-Coupled Neural Network(DC-CCNN)를 제안하고, 이를 기반으로 더 강한 DC-CCNN++를 만든다. DC-CCNN++는 Neuro-Inspired Robust Modulation-and-Readout Module(NRMR)과 Cortically Inspired Progressive Variability Training(CPVT)로 구성돼, 아키텍처(전역 조절·듀얼 코드 판독)와 학습(점진적 환경 변동 노출)을 함께 강화한다. 결과적으로 단순히 깨끗한 데이터 성능을 올리는 것을 넘어, 부정합 상황에서 특징과 결정을 더 안정적으로 유지하는 방향을 제시한다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 점군의 불규칙·희소·잡음 입력에서 효율적 표현을 만들면서 (2) 부정합이 들어와도 “신뢰 경로”가 흔들리지 않게 제어하는 것이다. DC-CCNN은 BINN–CCNN의 결합으로 국소 누적(시간적 안정화), 이웃 결합(공간 구조), 동적 임계값(잡음 억제)을 통해 로컬 수준의 강건성을 설계했고, NRMR은 여기에 더해 전역 문맥 기반 채널 게인 조절로 경로 신뢰도를 재가중하며 듀얼(최대/평균) 판독으로 보완적 근거를 함께 통합한다. 또한 CPVT는 다양한 부정합을 ‘환경 변동’으로 통일해 점진적으로 노출하되, 미니배치에서 깨끗한 샘플 기준선(homeostatic anchor)을 함께 유지하도록 설계해 적응 과도화를 막는다.

- **Empirical Impact**: 실험에서 DC-CCNN++는 분류와 파트 세분화 모두에서 원래 DC-CCNN보다 더 높은 성능을 보이며, 상태-최신 방법들과 견줄 만한 수준의 정확도를 유지한다. 특히 희소성, 가림, 가우시안 잡음, salt-and-pepper 잡음, 공간 변환 같은 다양한 corruption에서 강건성이 개선되는 것으로 보고된다. 효율성(에너지 관점의 뉴로모픽 설계)과 실사용 강건성을 동시에 노린 뇌 영감 기반 구조/학습 전략이 점군 분석 분야에서 대안적 방향성을 제공한다는 점에서 의미가 있다.



### One Layer's Trash is Another Layer's Treasure: Adaptive Layer-wise Visual Token Selection in LVLMs (https://arxiv.org/abs/2606.14277)
Comments:
          Accepted by CVPR 2026 (highlight)

- **Prior Approaches**: 기존 비전-언어 모델의 시각 토큰 압축은 대체로 특정 레이어에서 어텐션을 보고 정적(고정) 가지치기를 수행하는 방식이 많다. FastV처럼 한 번 잘린 토큰은 이후 모든 레이어에서 다시 접근할 수 없어, 깊은 레이어에서 필요한 시각 정보가 “너무 일찍” 사라질 수 있다. 또한 레이어마다 관심하는 시각 영역이 다르다는 관찰에도 불구하고, 정적 부분집합을 전 레이어에 그대로 적용하는 한계가 있었다.

- **Core Contribution**: ALVTS(Adaptive Layer-wise Visual Token Selection)는 레이어별로 다른 시각 토큰 부분집합을 동적으로 선택하도록 프레임을 바꾼다. 각 레이어는 자기 역할에 중요한 토큰만 처리하고, 나머지 토큰은 해당 레이어를 스킵하되 다음 레이어에서 다시 결합해 “접근성”을 유지한다. 이를 통해 레이어 간 시각 집중도 차이를 반영하면서 계산 중복을 줄이는 것을 목표로 한다.

- **Technical Challenges**: 가장 큰 과제는 모든 토큰에 대해 풀 어텐션을 돌리지 않고도 “중요한 토큰”을 정확히 추정해야 한다는 점이다. ALVTS는 어텐션의 질의/키 투영을 저랭크 근사로 구성한 경량 토크 셀렉터로 각 레이어의 중요도 점수를 계산하고, top-k로 선택된 토큰만 해당 레이어 연산에 참여시키며 나머지는 건너뛴다. 또한 중요도 일관성을 제약한 저랭크 근사(importance consistency constrained low-rank approximation)로 셀렉터가 풀 어텐션의 선택 패턴을 가깝게 따라가도록 하여 재학습 없이도 동작하게 했다.

- **Empirical Impact**: LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL 등에서 압축률 89%일 때 원본 대비 정확도 96.7%를 유지하며, 89% 압축과 함께 더 나은 효율-정확도 트레이드오프를 보였다. 구체적으로 토큰 압축 89%에서 평균 성능이 베이스라인 대비 유의하게 높고, LLaVA-1.5-7B에서는 지연 시간이 211ms→156ms로 단축(1.35×)되며 프리필 시간은 165ms→103ms(1.6×)로 빨라졌다. 정적 가지치기 대비 레이어별 동적 선택이 COCO/NoCaps 같은 과제에서 특히 개선되는 결과도 제시되어, 실전 배포 관점의 확장성까지 뒷받침한다.



### HiST: A Hierarchical Sparse Transformer for Cross-Modal Spatial Transcriptomics Modeling (https://arxiv.org/abs/2606.14251)
- **Prior Approaches**: 기존 H&E-to-ST 방법은 (1) 측정 위치를 패치로 보고 독립적으로 회귀하거나, (2) 고정된 거리/국소 이웃을 그래프 전파나 이웃 어텐션으로 결합해 맥락을 전달하는 흐름이 많았습니다. 이런 방식은 장거리 조직 구조를 포착하려면 층을 깊게 쌓아야 하거나, 국소 결합 설계에 민감해지는 한계가 있습니다. 또한 반복적 생성(Flow/Diffusion) 계열은 전역 추론 비용이 커서 whole-slide 스케일에서 효율이 떨어질 수 있습니다.

- **Core Contribution**: HiST는 측정된 ST 위치를 기준으로 ‘조직이 존재하는 능동 영역(active footprint)’에만 계산을 집중하는 계층형 희소 트랜스포머를 제안합니다. H&E-정렬 정보를 활용해 각 유전자 발현 벡터를 예측하되, 다중해상도 문맥을 U-Net 같은 인코더-디코더 위계로 빠르게 확장합니다. 여기에 슬라이드별 염색/스캐너 변이를 완화하기 위해 병목 구조의 slide calibration token(슬라이드 보정 토큰)으로 전역 조건을 전달합니다.

- **Technical Challenges**: whole-slide 입력은 기가픽셀 규모인 반면, supervision은 불규칙하고 희소한 위치 집합에만 존재해 다중스케일 모델링이 까다롭습니다. HiST는 이를 위해 측정 좌표를 격자 인덱스에 매핑해 희소 장(field)으로 표현하고, 고정 크기 window sparse attention과 dyadic(2배 단위) 계층 전이로 장거리 수용영역을 효율적으로 넓힙니다. 슬라이드별 획득 차이는 전역 토큰이 모든 위치 분포를 요약한 뒤 로컬 표현을 조건화하는 저대역폭 경로로 해결하도록 설계했습니다.

- **Empirical Impact**: 여러 장기(multi-organ) 벤치마크에서 HiST는 최근 기준선 대비 예측 성능을 동급 이상으로 유지하면서도 추론 시간과 피크 메모리를 크게 절감했다고 보고합니다. 특히 런타임과 메모리 스케일이 전체 슬라이드 면적이 아니라 관측된 위치 수(활성 토큰 수) 중심으로 움직이도록 구성되어 효율성이 두드러집니다. 결과적으로 whole-slide H&E를 이용한 분자 수준 예측을 더 실용적으로 만들며, 계산 제약 하에서도 정확도를 끌어올릴 수 있음을 보여줍니다.



### A Multi-Domain Feature Fusion Framework for Generalizable Deepfake Detection Across Different Generators (https://arxiv.org/abs/2606.14230)
- **Prior Approaches**: 기존 딥페이크 탐지는 생성 과정에서 남는 흔적(공간·그래디언트·주파수)을 ‘단서 기반’으로 쓰거나, 대규모 데이터로 ‘데이터 기반’ 특징을 학습하는 방식으로 크게 나뉩니다. 다만 많은 방법이 단일 표현 도메인(주로 공간 또는 주파수)에 치우쳐 있어, GAN에서 잘 되더라도 확산 모델(diffusion)이나 다른 제너레이터로 갈수록 성능이 급격히 떨어지는 경향이 있습니다. 또한 제너레이터/패러다임이 바뀌면 주파수·공간 단서가 약해지거나, 동일 계열 내부에서 학습된 편향이 일반화에 방해가 된다는 점이 반복적으로 지적됩니다.

- **Core Contribution**: 이 논문은 공간, 그래디언트, DWT 기반 주파수 표현을 함께 쓰는 다중 도메인 딥페이크 탐지 프레임워크 SGFF-Net(Spatial-Gradient-Frequency Fusion Network)을 제안합니다. SGFF-Net은 Dual Residual Learning 구조 안에서 두 잔차 학습 경로로 서로 다른 도메인의 포렌식 단서를 융합해, GAN과 확산 모델을 모두 아우르는 제너레이터-비의존적(generator-agnostic) 표현을 학습하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 생성 패러다임에서 공유되는 ‘일반화 가능한 단서’만을 효과적으로 모으는 동시에, 단일 도메인 특징이 만들 수 있는 분포 이동(distribution shift) 취약성을 줄이는 것입니다. 저자들은 RGB(공간)뿐 아니라 그래디언트 맵을 생성하는 사전 변환 모델과 DWT 기반 다중 해상도 주파수 큐(High-High 성분)를 함께 사용하고, Dual Residual Learning을 통해 보완적 정보를 단계적으로 결합하는 방식으로 이를 해결합니다. 여기에 멀티소스 학습과 데이터 증강을 체계적으로 추가해 크로스 제너레이터/크로스 패러다임 강건성을 더 끌어올립니다.

- **Empirical Impact**: 실험에서 SGFF-Net은 동일 데이터셋 평가(intra-dataset) 정확도 98.95%를 달성하며, 크로스 모델 70.46%, 크로스 패러다임 69.94%로 단일 도메인 기반 대비 일반화 성능을 개선합니다. 특히 멀티소스 학습과 증강을 적용하면 크로스 모델 정확도가 79.80%로, 크로스 패러다임은 약 78%로, 실제 환경(real-world) 데이터에서는 61.50%에서 75.80%로 크게 상승해 실사용 강건성을 보여줍니다. 결과적으로 공간·그래디언트·웨이브릿 주파수의 상호보완적 포렌식 신호를 결합하는 전략이, unseen 제너레이터와 실제 데이터 전이에서 효과적이라는 실증적 근거를 제공합니다.



### Hybrid Classical-Quantum (HCQ) Alzheimer's Classification via Supervised $β$-VAE and Quantum Kernels (https://arxiv.org/abs/2606.14194)
- **Prior Approaches**: 기존 AD 구조 MRI 분류는 (1) 수작업 바이오마커(예: 해마 부피, 피질 두께)를 추출해 얕은 분류기를 쓰는 방식, (2) 3D CNN처럼 end-to-end로 특징을 학습하는 방식, (3) 오토인코더로 압축 후 분류하는 compress-then-classify 방식으로 발전해 왔다. 하지만 소규모 코호트에서 수작업 특징은 병적 신호의 누락이 생기고, 3D CNN은 과적합·재현성 문제가, 전통적 오토인코더는 재구성 중심이라 질병-판별 정보를 충분히 담기 어렵다.

- **Core Contribution**: 이 논문은 두 단계 하이브리드 classical-quantum(HCQ) 파이프라인에서 핵심을 ‘서로 독립적으로 동작하지 않게’ 만드는 데 둔다. β-변분 오토인코더(베타-변분 오토인코더, β-VAE)를 재구성·KL·focal 분류 손실로 동시에 학습해 질병-인지(disease-aware) 잠재표현을 만들고, PLS로 6개 진단 특징(회전각)을 뽑아 양자 커널이 그 표현을 직접 받도록 구성한다.

- **Technical Challenges**: 첫째, 3D MRI는 억 단위 복셀이라 양자 회로가 바로 입력받기 불가능하므로, end-to-end로 학습된 압축 표현이 양자 단계로 “질병 신호를 유지한 채” 전달되어야 한다. 둘째, 소규모 데이터에서 VAE 초기값·훈련 변동이 큰데, 이를 줄이기 위해 안정성 향상 모델(M2)은 서로 다른 시드의 다중 VAE 선택(내부 검증 성능 기준)과 추론 시 약한 증강 결과 평균, 그리고 Platt scaling/완화된 SVM 규제를 통해 확률 보정과 분산 감소를 달성한다.

- **Empirical Impact**: ADNI-1(308명, 137 AD/171 CN) 5-fold stratified 교차검증에서 기준선은 정확도 67.2%, AUC 0.759였고, 안정성 향상 변형은 정확도 72.1%, AUC 0.799로 개선되며 교차 폴드 분산이 절반 수준으로 줄었다. 3D Grad-CAM은 모델이 해마 및 내측 측두엽 등 알츠하이머 초기 구조 변화와 맞닿은 영역에 주로 초점을 둠을 보여, 질병-관련 특징 학습이 단순 데이터 아티팩트가 아닐 가능성을 뒷받침한다.



### MUSE: Agentic 3D Scene Authoring via Memory-Grounded Incremental Requirement Satisfaction (https://arxiv.org/abs/2606.14168)
- **Prior Approaches**: 기존 텍스트-기반 3D 장면 합성은 주로 데이터 기반 또는 one-shot LLM/에이전트 계획으로, 로컬 구조를 명시적으로 제어하거나 실행 이력을 지속해 추적하기 어렵습니다. 그 결과 특정 제약이 실패하면 전체 장면을 다시 생성하거나 수동 개입이 필요한 경우가 많아, 편집 가능성이 제한됩니다. 언어 가이드 편집 역시 일부는 객체 추가/삭제를 지원하지만, 요구 수준의 진행 상태와 비타깃 콘텐츠 보호를 일관되게 묶어 관리하는 메커니즘은 부족합니다.

- **Core Contribution**: 이 논문은 3D 장면 작성(생성+편집)을 ‘증분 요구사항 만족’으로 재정의해, 요구 수준의 상태를 유지하면서 로컬 수정과 보존을 함께 달성하려고 합니다. Architect–Sculptor–Inspector 다중 에이전트 구조를 통해 요구사항을 구조화해 실행하고, 각 단계에서 검증 및 메모리 업데이트로 이미 만족된 부분을 보호합니다. 이를 통해 장면 합성과 장면 편집을 하나의 통제 프레임워크로 통합합니다.

- **Technical Challenges**: 핵심 기술 난점은 조밀한 공간 제약을 한 번에 맞추려 하면 제약이 누락/환각되기 쉽고, 후속 작업이 만족된 결과를 되돌려(regression) 비타깃 영역을 깨뜨릴 수 있다는 점입니다. 저자들은 요구사항을 타입과 우선순위, 판별 가능한 조건(ϕ)으로 쪼개 스케줄링 가능한 단위로 만들고, Working/Scene/Skill Memory로 실행 진행과 보호 바인딩을 명시적으로 관리합니다. 또한 단계별 Inspector 검증(규칙 기반 + 필요한 경우 모델 판단)과 보호 집합 기반의 선차단(preemptive blocking), 로컬 일관성 수정으로 안전한 증분 실행을 보장합니다.

- **Empirical Impact**: AuthorBench를 통해 요구사항 수준의 통제성과 보존 인식 편집을 별도 검증기로 공정하게 평가했으며, MUSE는 전체 구성에서 All-Goal 성공률을 37.9에서 80.7로, 표면 제약 충족을 35.0에서 92.6으로 끌어올렸습니다. 편집에서는 240개 테스트 분할에서 All-Goal 49.6, 보존율 99.9, 의도치 않은 변경률 0.6으로 높은 로컬리티를 보였고, 인간 평가와 내비게이션 프록시 실험에서도 사용자 의도 및 공간 안정성 정렬이 강화되는 경향이 확인됐습니다. 더불어 메모리/검증 구성요소의 제거(ablation) 실험이 이 성능의 원인이 되는 설계 요소를 뒷받침합니다.



### VideoWeave: Unlocking Geometric Consistency in Video Generation via Joint Geometry-Video Modeling (https://arxiv.org/abs/2606.14162)
- **Prior Approaches**: 기존 대규모 비디오 확산 모델은 현실감은 만들지만, 시간에 따른 3D 구조를 안정적으로 유지하지 못해 시점 변화나 장시간 생성에서 기하학 드리프트와 부자연스러운 동작이 생기곤 합니다. 이를 줄이기 위해 깊이지도·포인트클라우드·카메라 포즈 같은 명시적 기하 복원을 조건/감독/보상으로 쓰는 방식이 많았지만, 상류 기하 추정 오류가 그대로 생성기에 전파되고 계산 비용도 커져 확장성이 제한됩니다. 일부는 기하 모델의 암묵적 특징을 쓰지만, 비디오 생성 잠재공간과 표현 정렬이 어긋나면 사전학습된 비디오 우선순위(프리트레이닝 prior)를 해칠 수 있습니다.

- **Core Contribution**: 본 논문은 VideoWeave를 제안하며, 기하 정보를 ‘복원 결과 기반의 외부 조건’이 아니라 ‘학습 시간에만 쓰이는 잠재 변수’로 다루어 기하 일관성을 분포 수준에서 유도합니다. DA3 같은 기하 모델의 내부 특징을 기하 잠재로 변환해 비디오 잠재와 공유 디노이징 공간에서 함께 모델링하고, 이후 학습된 결합 분포의 점수 필드를 더 작은 학생 생성기로 증류합니다. 또한 공동 학습을 위해 외관(appearance)과 암묵적 기하 표현이 페어로 주어지는 GeoVid-80K(80K 영상)를 구축합니다.

- **Technical Challenges**: 핵심 난제는 서로 다른 시간 해상도·공간 해상도·채널 차원을 가진 기하 특징과 비디오 잠재를 호환되게 만들되, 사전학습된 비디오 prior를 갑작스런 분포 이동으로 망치지 않는 것입니다. VideoWeave는 먼저 경량 어댑터로 스케일을 맞추는 ‘geometry latent warm-up’을 수행해 적응된 기하 잠재가 기존 디노이징 목표 안에서 서서히 의미를 갖도록 하고, 그다음에는 영상의 속도는 기존 최종 헤드로, 기하는 중간 계층에서 예측하는 계층-인식(레이어-어웨어) 공동 모델링으로 3D 관련 신호가 과도하게 지워지지 않게 설계합니다. 마지막으로 고비용 결합 확산모델의 분포를 DMD 방식으로 몇 단계 생성기에 점수 수준에서 옮겨, 추론 시에는 기하 인코더/디코더를 완전히 제거합니다.

- **Empirical Impact**: 실험은 text-to-video와 image-to-video 두 설정에서 수행되며, VideoWeave가 기하적 일관성(공간 코히어런스·장거리 기하 안정성)을 개선하면서도 시각 품질을 유지함을 보여줍니다. 또한 학습 시에는 암묵적 기하 잠재를 쓰지만 추론에서는 이를 버리므로, 기존처럼 매번 기하 복원·렌더링 파이프라인을 돌려야 하는 방식 대비 효율적인 생성이 가능합니다. GeoVid-80K는 깊이 단서·모션 패럴랙스·시점 변화·시간적 일관성을 강조하도록 선별되어, 향후 3D-aware 비디오 생성 연구의 재현성과 비교 가능성을 높이는 기반이 될 전망입니다.



### Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale: A Frozen-Backbone Grafting Diagnostic (https://arxiv.org/abs/2606.14153)
Comments:
          23 pages, 5 figures, 8 tables

- **Prior Approaches**: 기존 연구는 VLA에서 비전 인코더를 바꿀 때 언어·액션 구성요소와 함께 학습(공동 적응)되는 경우가 많아, 관측된 성능 차이가 인코더 자체인지 백본과의 상호적응인지가 섞여 해석이 어려웠습니다. VLM4VLA처럼 학습 단계에서 VLM 조합을 체계적으로 바꾸는 연구도 있지만, 이미 공개된 VLA 체크포인트를 상속받아 ‘후처리로 인코더만 갈아끼우는’ 실무 질문과는 거리가 있습니다.

- **Core Contribution**: 논문은 ‘냉동 백본 그래프팅(frozen-backbone grafting)’ 진단을 제안합니다. 언어 모델과 액션 전문가를 고정한 채, 후보 비전 인코더를 결정적 풀링(Adaptive Average Pooling)과 LayerNorm, 단일 학습 가능한 선형 프로젝터만으로 연결해 오프라인 행동 MSE를 비교하며, 작은 VLA에서의 인코더 최상위 선택이 큰 백본으로도 옮겨가는지 검증합니다.

- **Technical Challenges**: 핵심 기술 과제는 인코더 비교가 언어-인코더 공동 적응과 섞이지 않도록 공정성을 만드는 것입니다. 이를 위해 후보 인코더는 사전학습 가중치를 동결하고, 고정된 토큰 수가 되도록 공간 풀링을 결정론적으로 적용한 뒤 프로젝터만 2,000 스텝(배치/학습률 프로토콜 고정) 학습해 순수한 인코더-적합성 신호를 추출합니다; 또한 시뮬레이터와 체크포인트의 본체(embodiment) 불일치 때문에 폐루프 성공률 대신 오프라인 행동 예측 MSE를 사용해 측정 가능성을 확보했습니다.

- **Empirical Impact**: 결과적으로 작은 백본에서 1등으로 뽑힌 인코더가 큰 백본에서도 신뢰성 있게 1등을 보장하지 않았습니다. SmolVLA에서는 SigLIP이 두 스위트에서 전반적으로 우세하지만, π0.5에서는 DINOv2가 공간 스위트에서 앞서고 객체 스위트는 씨드 민감한 근접-동률 밴드를 보이며, 특히 백본-스위트 조합에 따라 랭킹 방향이 엇갈리는 양상이 관측됩니다. 다만 그래프팅 하네스 자체가 백본마다 부호가 다른 영향을 주므로(조건부 결론), 이 방법은 ‘대규모 커밋 전에 저비용으로 타깃 백본 진단을 돌리는 도구’로서의 실용적 의미가 큽니다.



### BoRAD: Bootstrap your Own Representations for Multi-class Anomaly Detection (https://arxiv.org/abs/2606.14129)
- **Prior Approaches**: 기존 재구성 기반 이상 탐지는 각 범주별로 별도 모델을 학습하는 한-대-한(one-for-one) 방식이 많아, 범주가 늘면 운영 비용이 크게 증가한다. one-for-all로 묶으면 모델이 여러 정상 외관을 함께 재구성해야 해서 두 실패 모드(동일 shortcut: 이상이 그대로 통과, mis-reconstruction: 범주 혼동)가 함께 커진다. 이를 줄이기 위해 클래스 조건 모듈, 메모리/프롬프트, 마스킹, 양자화 등 다양한 접근이 제안됐지만 라벨·음성 쌍·추론 시 메모리 조회 같은 추가 부담이 남는다.

- **Core Contribution**: BoRAD는 one-for-all 재구성 이상 탐지를 ‘표현(표상) 용량 배분 문제’로 재정의하고, 라벨 없이 학습하는 학습 프레임워크를 제안한다. 핵심은 공유 가능한 prototype bank로 훈련 중 표현 공간을 형태별로 다르게 제약해 동일 shortcut과 mis-reconstruction을 동시에 완화하는 것이다. 추론 단계에서는 prototype을 조회/사용하지 않고 기존 teacher-student 특징 불일치 기반 방식만으로 이상 점수를 계산한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 정상 범주 간 구조를 유지하면서도 이상 영역이 재구성 경로로 ‘복사’되는 것을 막는 균형(과표현 vs 과압축)이다. BoRAD는 이를 기하학적 분기(Geometric Bifurcation)로 풀기 위해, 공간(로컬)에서는 prototype 정렬로 within-prototype 변동을 수축해 이상 복사를 억제하고, 전역(글로벌)에서는 prototype-relative 좌표 이동으로 범주 간( between-prototype ) 각도/기하 구조를 보존한다. 또한 BYOL 스타일의 negative-free 예측 정렬을 prototype 조건 하에서 수행해 라벨·음성 쌍 없이도 분산/각도 민감도를 함께 조절한다.

- **Empirical Impact**: MVTec AD에서 mAD 86.2%, VisA에서 80.7%, Real-IAD에서 73.1%로 경쟁력 있는 성능을 보이며, 특히 로컬라이제이션 지표에서 강점을 보인다. 성분별 축적 실험에서 기본 재구성 모델 대비 prototype 기반 공간 수축과 전역 shift 정렬을 더할수록 성능이 일관되게 향상되며, 최종 모델은 AUROC·pixel AP/F1·mAD 전반에서 최고치를 달성한다. 진단 분석에서도 이상 복사 감소, 정상 범주 분리도 개선, 이상-정상 점수 분리 강화가 확인되어 제안한 표현 제어 전략의 실증적 타당성을 뒷받침한다.



### Conditioning Matters: Stabilizing Inversion and Attention in Diffusion Image Editing (https://arxiv.org/abs/2606.14125)
Comments:
          Accepted to ECML PKDD 2026 Research Track

- **Prior Approaches**: 학습 없이 텍스트 기반 확산 이미지 편집은 보통 역변환(inversion)과 어텐션 조작(attention manipulation)을 결합한다. 하지만 기존 방법은 역변환 정확도 저하(오차 누적)와 편집 충실도-배경 보존의 상충(trade-off) 문제를 해결하지 못해 구조가 무너지거나 의미가 흔들리기 쉽다. 또한 텍스트 조건이 교차 브랜치 어텐션의 의미·공간 정합성에 미치는 영향은 상대적으로 덜 다뤄져 왔다.

- **Core Contribution**: 이 논문은 텍스트 조건의 정밀도(precision)가 확산 속도장(diffusion velocity field)의 기하(geometry)와 안정성에 직접 영향을 주고, 그 결과 교차 브랜치 어텐션의 일관성까지 바꾼다는 점을 경험적·이론적으로 제시한다. 이를 바탕으로 SimEdit은 (1) 조건 리파인먼트(conditioning refinement)로 역변환 안정성과 구조 정합을 돕고, (2) 토큰 단위 교차 브랜치 어텐션 제어(token-wise cross-branch attention control)로 편집 유도 토큰과 구조 보존 토큰을 분리해 비대칭으로 조절한다. 즉, “텍스트 조건을 단순 입력”이 아니라 “편집 동역학을 좌우하는 기하 제약”으로 취급하는 관점이 핵심 기여다.

- **Technical Challenges**: SimEdit을 구현할 때의 핵심 난제는 더 정밀한 조건이 역변환을 안정화하지만, 토큰 밀도가 늘며 어텐션 정규화로 인해 편집 의미 토큰이 희석(dilution)될 수 있다는 점이다. 논문은 LCS(최장 공통 부분 수열) 기반으로 구조 보존 토큰과 편집 유도 토큰을 분할하고, 구조 토큰에는 기존 방식의 특징 주입을 유지하는 반면 편집 토큰에는 증폭 계수로 비대칭 조절을 적용해 희석을 보정한다. 또한 텍스트 정밀도가 속도장 안정성을 좌우한다는 분석을 통해, 리파인먼트가 단순 프롬프트 개선이 아니라 역변환 궤적의 분산을 낮추는 데 기여함을 연결한다.

- **Empirical Impact**: PIE-Bench에서 SimEdit은 기존 어텐션 조작 계열 방법 대비 역변환 재구성 품질과 편집 성능을 동시에 일관되게 개선하며, 마스크 없는 설정에서도 구조 보존과 의미 정렬을 함께 끌어올린다. 여러 확산 백본/역변환 솔버에서도 리파인먼트 효과가 재현되어 일반성이 확인됐고, 평가 지표(DINO 기반 구조 일관성, 편집 영역 외 배경 보존 PSNR/SSIM/LPIPS/MSE, CLIP 정렬) 전반에서 개선 경향을 보였다. 런타임도 샘플당 총합 기준으로 대부분의 비용을 차지하는 반면 리파인먼트 단계는 상대적으로 작은 비중(약 수십 초 중 일부)에 그쳐 실사용 편입성이 높다는 점을 강조한다.



### A New Multi-Domain Benchmark for Micro-Action Recognition and Detection (https://arxiv.org/abs/2606.14096)
Comments:
          10 pages, 9 figures

- **Prior Approaches**: 기존 연구는 주로 매크로 액션(크고 의도적인 움직임) 인식에 집중했으며, 미세 움직임은 상대적으로 덜 다뤄졌다. 미세 동작(whole-body micro-actions)은 MA-52처럼 실험실 인터뷰 기반 벤치마크가 먼저 등장했지만, 규모·장면 다양성·태스크 범위·평가 프로토콜이 제한적이었다. 후속 연구는 검출/QA 등으로 확장했지만, 여전히 실제 환경의 도메인 변화와 장면 복잡도를 충분히 반영하기엔 부족했다.

- **Core Contribution**: 본 논문은 MA-52의 한계를 보완해 MMA-82를 제안한다. MMA-82는 52개에서 82개로 미세 동작 라벨을 확장하고, 실험실 인터뷰뿐 아니라 거리 인터뷰, 정신과 환자 인터뷰, 감정이 풍부한 TV 영상 등 4개 도메인을 포괄해 총 77,856개의 주석 인스턴스를 제공한다. 또한 인식(Micro-Action Recognition)과 다중 라벨 검출(Multi-label Micro-Action Detection)이라는 두 핵심 태스크와, in-domain/ cross-domain 및 few-shot/zero-shot 평가까지 구성해 보다 현실적인 성능 검증을 가능하게 한다.

- **Technical Challenges**: 미세 동작은 지속 시간이 매우 짧고 진폭이 작아(잡음·조명·배경·자연스러운 신체 변동에 취약) 라벨링과 인식 모두가 어렵다. 논문은 멀티스테이지 데이터/라벨링 파이프라인을 통해 범주 체계를 정교화하고(82개 세분 라벨), LabelU 기반으로 시작·종료 시각까지 주석하되 0.2초 최소 구간과 3인 주석/검증 절차 및 IoU 기반 합의 병합 전략으로 시간 경계 불확실성을 완화한다. 더 나아가 도메인 시프트와 장꼬리 분포, 복잡한 시간 구간 국소화 문제를 평가 프로토콜 자체에 반영해 모델의 일반화 난이도를 실질적으로 높였다.

- **Empirical Impact**: 실험 결과, 기존 방법은 현실적인 미세 동작 이해에서 특히 도메인 변화, 장꼬리 범주 불균형, 그리고 시간적 로컬라이제이션에서 여전히 큰 어려움을 보였다. 즉, 학습 도메인과 테스트 도메인이 달라지면 성능이 크게 하락하며, few-shot 적응만으로는 한계가 있음을 확인했다. 한편 논문은 미세 동작이 감정 상태와 강하게 연관되며, 얼굴 미세표정(Facial micro-expressions) 외에 감정 인식에 보완 단서를 제공할 수 있음을 분석으로 보여주며, 인체 중심(human-centered) 비디오 이해 분야의 연구 자원으로서 MMA-82의 의미를 확장한다.



### FEMOT: Multi-Object Tracking using Frame and Event Cameras (https://arxiv.org/abs/2606.14094)
- **Prior Approaches**: 기존 MOT는 주로 프레임 기반 RGB 카메라의 검출-연결(tracking-by-detection) 또는 탐지+추적, 쿼리 기반 Transformer로 발전해 왔습니다. 하지만 저조도, 역광, 과노출, 모션 블러, 유사 외형, 잦은 출입/가림 등 복잡한 환경에서는 시각 정보 신뢰도가 떨어져 궤적 단절과 신원 연관 오류가 누적되기 쉽습니다. RGB-이벤트(event) 융합은 검출·단일 객체 추적에서는 연구가 있었지만, 신원 단위의 시간 일관성이 요구되는 RGB-이벤트 다중 객체 추적(MOT)은 대규모·정밀 어노테이션 벤치마크가 부족해 체계적으로 검증되지 못했습니다.

- **Core Contribution**: 이 논문은 RGB-이벤트 MOT을 위한 대규모 데이터셋 FEMOT(Frequency-aware? 아님; 논문 제목의 데이터셋명)와 이를 활용한 포괄 벤치마크를 제시합니다. FEMOT은 다양한 실제 시나리오와 14개 도전 속성(저조도, 빠른 움직임, 출입 빈도 등)을 포함하며 RGB 및 이벤트 데이터와 고품질 궤적/박스 어노테이션을 함께 제공합니다. 또한 주파수 영역에서 RGB와 이벤트의 상보성을 분리·융합하는 FEMOTR을 제안해, 강건한 위치 추정과 신원 연관을 동시에 노립니다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 비동기 이벤트를 RGB 프레임에 정렬해 학습 가능한 입력으로 만들고, (2) 두 모달리티가 갖는 서로 다른 주파수 특성과 잡음/왜곡을 안정적으로 결합하며, (3) 다중 객체에서 시간에 걸친 신원 연관을 지속적으로 유지하는 것입니다. 논문은 이벤트를 RGB 프레임의 노출 구간에 맞춰 이벤트 누적(e.g., ON/OFF 채널화) 형태로 변환해 정렬 문제를 줄였고, Frequency Aware Feature Fusion에서 FFT 기반으로 진폭/위상을 분리한 뒤 모달리티별 주파수 응답을 가이드-조절 방식으로 융합합니다. 이후 Transformer 기반 쿼리 전파와 동적 시간 상호작용 모듈(장기 메모리 갱신, 다중 궤적 간 attention)을 통해 탐지와 연관을 함께 수행합니다.

- **Empirical Impact**: FEMOT 데이터셋과 DSEC-MOT(DSEC-MOT) 두 벤치마크에서 광범위한 실험을 통해 FEMOTR의 효과를 검증합니다. 또한 FEMOT 기반으로 10개 이상 강력한 기존 추적기를 재학습·평가해, RGB-이벤트 MOT 연구를 위한 표준 벤치마크 성격을 확립합니다. 결과적으로 이벤트 카메라가 복잡 조명/모션에서 RGB의 취약점을 보완한다는 점을 정량적으로 뒷받침하며, 향후 RGB-이벤트 MOT의 비교·진보를 촉진할 것으로 기대됩니다.



### Clay-CNN Hybrids: Leveraging Geo-Foundational Models as Auxiliary Context for Landslide Detection (https://arxiv.org/abs/2606.14081)
Comments:
          9 pages, 7 figures, 2 tables

- **Prior Approaches**: 기존 재난 후(이벤트 직후) 산사태 자동 탐지는 주로 U-Net 계열의 의미 분할에 의존하지만, Landslide4Sense처럼 양성 픽셀이 약 2% 수준인 극단적 클래스 불균형과 이벤트별 데이터 부족에 취약했다. 또한 지형(DEM·경사) 정보는 물리적으로 의미가 있으나 픽셀 수준 분리도는 낮아, 기존 방식은 스펙트럼 유사성(맨땅 등) 때문에 오탐을 줄이기 어렵다는 한계가 있었다. 최근에는 Geo-Foundational Model(GFM)로 Prithvi-EO-2.0 등이 시도됐지만, 이를 CNN 구조에 어떻게 ‘최적으로’ 결합하는지에 대한 체계적 검증이 부족했다.

- **Core Contribution**: 이 논문은 Clay v1.5 같은 Geo-Foundational Model의 장점을 산사태 분할에 실질적으로 살리기 위해, GFM을 CNN과 결합하는 두 가지 하이브리드 아키텍처(Arch 1: Clay를 주 인코더+잔차 지형 융합, Arch 2: U-Net의 병목 컨텍스트에 Clay 삽입)를 제안한다. 핵심 메시지는 GFMs를 분할기 전체를 대체(standalone 인코더)하기보다, 공간 정밀도가 필요한 CNN 계층에 ‘보조 컨텍스트’로 투입하는 방식이 유리하다는 점이다. 즉, 스펙트럼의 일반화는 Clay가, 경계·국소화는 U-Net이 담당하도록 역할을 분리한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 클래스 불균형으로 인해 결정 임계값이 기본 0.5에서 크게 벗어나며 (2) Clay의 전역(글로벌) 표현을 픽셀 단위 경계 재현에 맞게 정렬해야 한다는 점이다. 이를 위해 LoRA 기반의 단계적 미세조정(먼저 디코더/융합 학습 후 일정 epoch에 LoRA 삽입), 가중 손실(BCE+Lovász hinge)로 F1 최적화 성향을 강화, 그리고 검증셋에서 임계값을 재보정하는 절차를 적용했다. 또한 MC Dropout과 Grad-CAM을 통해 단순 성능 향상뿐 아니라 오탐/미탐 및 융합 기여가 어디서 발생하는지 진단했다.

- **Empirical Impact**: Landslide4Sense 벤치마크에서 하이브리드 U-Net+Clay(Arch 2)에 2-stage LoRA를 적용한 모델이 테스트 F1 64.5±1.8%(seed 3개 평균)를 달성해 U-Net 기준선(59.9%)과 Clay 단독 인코더(55.2%)를 모두 능가했다. 특히 FPR은 낮게 유지하면서(배경 오탐 억제) 미세한/스펙트럼이 애매한 영역에서 FN이 상대적으로 늘어나는 오차 양상이 드러났고, 이는 불균형 상황에서의 합리적 trade-off로 해석된다. 해석 실험(Grad-CAM)과 불확실성(MC Dropout)은 Clay가 ‘분류/우선순위’에 도움을 주고, CNN이 ‘공간 정밀’로 보정한다는 결론을 뒷받침하며, 향후 지오해저드 분할에서 GFMs의 배치 전략에 실증적 기준을 제시한다.



### Diffusion-Refined Segmentation and Vision-Language Interpretation for Pediatric Brain Tumor MRI (https://arxiv.org/abs/2606.14072)
- **Prior Approaches**: 소아 뇌종양 분할은 희소한 라벨 데이터, 영상 내 이질성, 경계가 흐린(infiltrative) 종양 특성 때문에 기존 지도학습 모델의 일반화가 어렵다. BraTS-PEDs 2023 같은 벤치마크에서는 nnU-Net/Sw i n-UNETR 계열 앙상블이나 대규모 사전학습이 성능을 끌어올리지만, 연산·추론 비용이 커지고 특히 작은 Enhancing Tumor(ET) 경계에서 불균형 문제가 남는다. 또한 확산 모델은 성능 향상이 보고되었으나, 강한 기준선 대비 경계 개선의 일관성이 성립되지 않거나 소아 데이터에서 충분히 검증되지 못한 한계가 있다.

- **Core Contribution**: 이 논문은 3D 분할 기준 모델(3D Res U-Net과 Swin-UNETR)을 먼저 돌린 뒤, 그 “거친 예측(coarse prior)”을 조건으로 하는 확산 기반 정련(refinement) 모델로 경계를 날카롭게 만드는 2단계 프레임워크를 제안한다. 특히 조건부 3D DDPM refiner와 MedSegDiff를 도입해 확산 과정의 안정성을 높였고, ET 경계 품질 향상에 초점을 둔다. 마지막으로 분할 결과의 정량 볼륨과 대표 시각화를 멀티모달 언어 모델에 결합해 방사선학 스타일의 구조화 리포트를 자동 생성한다.

- **Technical Challenges**: 소아 종양은 ET 같은 소표적이 전체 볼륨의 극히 일부를 차지해 클래스 불균형이 심하며, 경계가 모호해 확률적 생성 과정이 잡음을 학습해 붕괴할 위험이 있다. 저자들은 이 문제를 (1) 거친 Swin-UNETR 예측을 조건으로 확산 입력에 주입하고, (2) 경계에 가중치를 둔 학습 목적(경계 강조 손실)과 (3) 불확실성을 다루도록 설계된 확산 정련 구조로 완화한다. 또한 3D 연산의 메모리 병목을 고려해 학습은 패치/슬라이싱 기반으로 수행하고, 지표는 HD95처럼 경계 중심의 평가에 맞춰 설계했다.

- **Empirical Impact**: BraTS-PEDs 2023에서 조건부 확산 모델은 비조건(unconditional) 대비 학습 안정성과 성능이 크게 개선되며, 특히 ET의 Dice가 의미 있게 상승한다. MedSegDiff(조건부)가 가장 낮은 HD95를 기록하며 경계 일치가 우수함을 보여, 확산의 강점이 단순 재현이 아니라 “경계 정밀도 개선”에 있음을 실증한다. 더불어 토머스(ET/TC/WT) 볼륨과 오버레이를 언어 모델 입력으로 연결해 해석 가능한 임상형 리포트 워크플로까지 시연하며, 소아 신경종양 분야에서 end-to-end 보조 AI 활용 가능성을 제시한다.



### ShearFuse-UNet: Hadamard, DCT, and Shearlet Transform Fusion for Next-Day Wildfire Spread Prediction (https://arxiv.org/abs/2606.14071)
- **Prior Approaches**: 기존 접근은 경험적·결정론적 시뮬레이터나 통계 기반 모델에 기반해 왔지만, 환경 변수 간의 복잡한 비선형 상호작용을 충분히 다루기 어렵다는 한계가 있었습니다. 딥러닝 계열에서는 CNN·U-Net 변형이 세그멘테이션 형태로 다음 날 화재 확산을 예측했으나, 데이터 수집과 성능에 초점을 맞추는 경우가 많아 계산 효율이 떨어지거나 방향성 구조 반영이 약했습니다. 또한 변환 기반 방법이 주파수에서 계수를 분리해 효율을 높이더라도(예: WHT, DCT), 화재 전선은 강한 이방성을 가져 단순 등방 변환만으로는 경계의 방향 정보를 충분히 포착하기 어렵습니다.

- **Core Contribution**: 이 논문은 ShearFuse-UNet을 제안하며, 경량 U-Net 인코더 블록 내부에 WHT·DCT·cone-adapted digital Shearlet의 3개 변환 분기(branch)를 통합해 다음 날 화재 확산을 예측합니다. WHT와 DCT는 상호 보완적인 스펙트럼 표현을 제공하고, SpectralFusion 게이트로 두 표현을 채널 단위로 적응적으로 결합합니다. Shearlet 분기는 잔차(residual)로 복원되어 화재 전선의 길쭉한 가장자리 같은 방향성 구조를 명시적으로 강화합니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 “고정 수학 변환”을 모델에 넣으면서도 학습 가능성과 재구성 안정성을 유지하는 것이었습니다. 저자들은 WHT·DCT는 고정된 직교 변환에 학습 가능한 스펙트럼 스케일링과 소프트 임계값을 더해 희소한 필터링을 수행하고, Shearlet은 cone-adapted 디지털 Shearlet을 PyTorch만으로 구현해 방향·이방성 분해를 잔차 경로로 결합했습니다. 또한 Shearlet을 모든 인코더 단계에 촘촘히 넣으면 성능이 소폭 하락(검증 F1 약 0.5% 감소)하는 경향을 관찰해, down2와 down4 단계에만 희소하게 주입하는 설계를 채택해 중복을 줄였습니다.

- **Empirical Impact**: WildfireSpreadTS에서 ShearFuse-UNet은 267k 파라미터로 F1 0.596(IoU 0.424)을 달성해, 14M 파라미터 ResNet18 기반 U-Net의 F1 0.589를 능가했습니다. 또한 Google Next-Day Wildfire Spread 데이터셋에서도 다른 벤치마크에 걸쳐 유사한 개선이 확인되어 일반화 가능성을 뒷받침합니다. 결과적으로 방향성 구조를 변환 유도 편향으로 반영하면서도 파라미터·연산을 크게 줄이는 정확도-효율 균형을 보여 해당 분야의 경량 예측 모델 설계에 의미 있는 실증을 제공합니다.



### WAM4D: Fast 4D World Action Model via Spatial Register Tokens (https://arxiv.org/abs/2606.14048)
Comments:
          15 pages, 7figures, 9tables

- **Prior Approaches**: 기존 WAM(월드 액션 모델)은 미래 관측과 실행 가능한 로봇 동작을 함께 예측하지만, 주로 2D 비디오나 잠재공간 표현에 의존한다. 이로 인해 접촉 지오메트리·가려진 표면·자유공간 같은 3D 제약을 충분히 반영하지 못해 정밀 조작에서 오류가 누적될 수 있다. 반면 4D 기반 방법들은 깊이/노멀/포인트맵 같은 기하 표현을 명시적으로 예측해 물리 일관성을 높이지만, 조작 추론 시점에 비용이 큰 디코딩이나 최적화가 필요해 지연이 커진다.

- **Core Contribution**: 이 논문은 빠른 4D 월드 액션 모델 WAM4D를 제안한다. 핵심 아이디어는 기하 기반(geometric foundation) 사전지식을 “학습 시 미래 깊이 readout”으로만 주입하고, 추론 시에는 2D 관측→동작 경로만 남기는 것이다. 이를 위해 공간 레지스터 토큰(spatial register tokens)으로 인과 비디오-동작 트랜스포머의 중간 표현에서 깊이를 읽어내도록 훈련하고, 그 결과가 조작 성능으로 이어지게 한다.

- **Technical Challenges**: 주요 기술 과제는 (1) 비디오 예측에 기하 감독을 억지로 얹으면 인과적(causal) 결합이 약해질 수 있고, (2) 밀집 4D를 그대로 목표로 두면 디코딩 비용이 커진다는 트레이드오프를 동시에 해결하는 것이다. WAM4D는 공간 레지스터 토큰이 미래 깊이 학습 목표를 만들되, 정책 경로에서는 레지스터/깊이 디코더를 제거해 추론 경량화를 달성한다. 또한 Mixture-of-Transformers(MoT) 백본에 causal mixture attention을 설계해 비디오·동작·레지스터 토큰 간 가시성(visibility)을 제어하고, 미래 동작이 미래 비디오/레지스터 정보를 “치트”처럼 쓰지 못하도록 마스킹한다.

- **Empirical Impact**: 실험은 RoboTwin 2.0와 접촉이 많은 실세계 장기 지평 조작 작업에서 수행되며, WAM4D가 공간 일관성과 조작 성공률을 개선하면서도 추론 효율을 유지함을 보였다. RoboTwin 2.0에서는 비디오 품질 및 깊이 관련 지표에서의 개선과 함께 동작 예측 성능이 경쟁 수준으로 유지되었고, 실세계 로봇 AstriBot S1의 4개 작업에서도 전반적으로 가장 높은 성과를 보였다. 특히 접촉·기하 민감·긴 시간 실행이 필요한 설정에서 기존 WAM/VLA 계열보다 실패율이 낮아, “학습 시에만 기하 프라이어를 전달하는” 접근의 실용성이 입증된다.



### Rethinking One-Step Image Editing through ChordEdit: Reproduction, Simplification, and New Insights (https://arxiv.org/abs/2606.14042)
Comments:
          9 pages

- **Prior Approaches**: 텍스트-가이드 단일 스텝 이미지 편집은 빠르지만, 실제로 어떤 메커니즘으로 원본 보존과 의미 정렬이 동시에 달성되는지 불명확하다는 한계가 있었다. 기존 ChordEdit 계열은 chord window(δ), chord transport, proximal refinement를 경험적으로 설계해 성능을 내왔으나, 각 구성요소의 역할이 분해·검증되진 않았다.

- **Core Contribution**: 본 논문은 ChordEdit을 재현(reproduction), 제거(ablation), 단순화(simplification)로 다시 해석해, 편집이 저주파 의미 이동과 고주파 정밀 정렬의 두 단계로 자연스럽게 분해된다는 관점을 제시한다. 특히 chord window δ는 독립적인 transport 경로라기보다 효과적인 노이즈 타임스텝을 t에서 t-δ로 “이동(shift)”시키는 역할이 크다고 분석한다.

- **Technical Challenges**: 구성요소별 기여를 분리하려면 δ, 편집 타임스텝 t, proximal refinement 타임스텝 t**의 상호작용을 제어한 실험 설계가 필요했으며, 저자들은 단일 파라미터 스윕과 그룹화된 정성 비교로 fidelity(PSNR)–semantics(CLIP-Whole/CLIP-Edited) 트레이드오프를 추적했다. 또한 chord window를 제거한 단순 설정이 거의 동일한 성능을 내는지 검증하며, chord path 자체보다 “좋은 효과 타임스텝 선택”이 핵심임을 실험적으로 확인했다.

- **Empirical Impact**: PIE Bench에서 재현 결과는 chord transport의 의미 효과가 원논문과 달라지는 등 재현성 이슈를 드러냈지만, proximal refinement는 의미를 개선하면서 배경 보존을 소폭 희생하는 일관된 패턴으로 확인되었다. 나아가 chord window를 단순화해도 기본 설정과 유사한 성능이 나와, 향후 prompt-conditioned dynamic timestep selection(프롬프트에 따라 t와 t**를 적응적으로 고르는 정책)로 발전할 수 있는 실마리를 제공한다.



### Toward 360-Degree Indoor Panorama Editing via Tuning-Free Diffusion Model with Refocusing Cross-Attention (https://arxiv.org/abs/2606.14035)
Comments:
          ICCCI 2026. Project page: this https URL

- **Prior Approaches**: 기존 제로샷 텍스트 기반 이미지 편집은 Stable Diffusion 같은 사전학습 T2I 모델의 가중치를 고정하고, 주로 주의(attention)나 잡음(노이즈) 블렌딩으로 구조 보존을 시도합니다. 다만 국소 편집에서는 사용자가 지정한 목표 영역보다 시각적으로 두드러진 대상에 모델이 더 집중해, 프롬프트 민감성(프롬프트 취약성)과 비목표 영역 번짐(spillover)이 자주 발생합니다. 또한 학습 데이터의 세밀한 물체 단위 감독 부족으로 작은 물체나 복잡한 장면에서 정밀한 변경이 불완전해지곤 합니다.

- **Core Contribution**: FocusDiff는 튜닝이나 최적화 없이 단일 패스로 목표 영역에 한정된 정밀 편집을 가능하게 하는 프레임워크입니다. 핵심은 리포커싱(refocusing) 크로스 어텐션으로, 비편집 영역에는 선택적 블러를 적용해 모델의 주의를 마스크된 목표 영역으로 유도하면서도 대상의 정체성(아이덴티티), 구조, 외형을 자연스럽게 전이하는 것입니다. 여기에 배경 보존과 전역 일관성을 위한 컨텍스트 보존 모듈을 결합해, 배경 훼손을 줄이면서도 지역 수정의 정확도를 끌어올립니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 크로스 어텐션이 자연스럽게 목표보다 두드러진 영역으로 쏠리면서 생기는 국소성 붕괴를 막는 것입니다. FocusDiff는 원본 이미지와 블러 처리된 이미지(비편집 영역 억제)를 DDIM inversion으로 잠재공간에 재현하고, 시점별로 마스크-유도 주의 계산을 통해 목표 영역의 의미 정보만 편집 잠재에 재주입합니다. 동시에 컨텍스트 보존 통합 모듈로 디노이징이 진행될수록 열화되는 배경 정보를 안정화하고, 텍스트 조건도 마스크 영역에는 프롬프트를, 비마스크 영역에는 null 프롬프트를 주어 전역 변경을 억제합니다.

- **Empirical Impact**: LIMB(Localiz​ed Image Manipulation Benchmark)에서 FocusDiff는 텍스트-이미지 정렬(예: CLIPScore)과 배경 보존(예: LPIPS) 사이의 균형에서 기존 제로샷 편집 SOTA를 능가했습니다. 특히 작은 물체/다중 물체가 포함된 어려운 케이스에서 편집 정밀도와 자연스러움이 개선되었고, SD v2.1 및 SDXL로도 내부 구조 변경 없이 확장 적용돼 성능이 더 좋아졌습니다. 또한 360도 실내 파노라마와 VR 환경까지 확장해, 목표 영역 한정 편집이 몰입형 시나리오에서도 효과적으로 작동함을 사용자 연구(SUS 77.12/100)와 함께 보여줬습니다.



### GarmentSketch: Large-scale Sketch-to-Fashion Benchmark (https://arxiv.org/abs/2606.14025)
Comments:
          ICCCI 2026. Project page: this https URL

- **Prior Approaches**: 패션 스케치 기반 이미지 합성은 스케치의 추상적·스타일화된 선 표현 때문에 구조를 지키는 데 어려움이 컸습니다. 기존 text-to-image나 sketch-to-image, ControlNet/T2I-Adapter 같은 멀티모달 결합 방식은 실사감과 구조 충실도 사이에서 균형을 맞추기 힘들었고, 특히 패션 도메인에 맞춘 대규모 스케치-캡션-이미지 연동 데이터가 부족했습니다.

- **Core Contribution**: 이 논문은 GarmentSketch라는 대규모 데이터셋을 제안하며, 21개 의류 카테고리에 걸쳐 26,249개의 패션 스케치를 상세 텍스트 설명과 페어링했습니다. 캡션은 여러 멀티모달 LLM을 조합해 후보를 만들고, 사람이 검증·보정하는 다단계 파이프라인으로 의미 정확성과 묘사 풍부함을 동시에 노렸습니다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘희소한 스케치 선’과 ‘텍스처·실루엣·장식’ 같은 고수준 의미를 한 모델이 동시에 만족하도록 정렬하는 데 있습니다. 저자들은 Informative-Drawings로 효율적인 스케치 구조 가이드를 만들고, 텍스트 쪽은 여러 MLLM의 상호 보완을 캡션 합성에 반영한 뒤 인력 검수로 품질을 고정해 데이터 정렬의 병목을 완화했습니다.

- **Empirical Impact**: 벤치마크 결과, Gemini 2.5 Nano Banana는 FID와 LPIPS에서 강세를 보이며 전반적으로 가장 그럴듯한 결과를 냈지만 스케치 구조 충실도는 상대적으로 흔들렸습니다. 반대로 ControlNet Scribble SDXL과 T2I-Adapter는 실루엣·형태 보존에 유리해 설계 프로토타이핑에 적합한 경향을 보였고, Ao Dai 같은 문화권 의상에서는 일부 모델이 오해/편향을 드러내 데이터 다양성의 필요성도 실증했습니다.



### ViT-Up: Faithful Feature Upsampling for Vision Transformers (https://arxiv.org/abs/2606.14024)
Comments:
          Code is available at: this https URL

- **Prior Approaches**: ViT의 전역 self-attention 비용 때문에 보통 14×14~28×28 수준의 저해상도 패치 토큰 격자에서 특징을 뽑아 dense prediction에 병목이 생긴다. 이를 줄이기 위해 JAFAR, AnyUp, UPLiFT, NAF 같은 task-agnostic feature upsampler는 고해상도 image guidance를 넣어 시각적으로 선명한 결과를 만들지만, 얕은 외부 인코더의 의미력 부족으로 feature leakage(의미가 섞임), fragmentation(조각남), blur(번짐) 같은 문제가 나타난다. 또한 많은 방법이 저해상도 토큰을 “재조립”하는 방식이라, 겉보기 선명도와 달리 의미 표현이 미세하게 혼합될 여지가 크다.

- **Core Contribution**: 논문은 ViT-Up을 제안하며, 외부 image guidance 대신 ViT 내부 중간 hidden state에서 계층적으로 쿼리를 구성해 연속 좌표에서 암시적(implicit) feature upsampling을 수행한다. 이를 통해 임의의 연속 좌표에서 backbone 특징 공간과 정렬(alignment)되는 밀집 특징을 예측하고, 토큰 재조립형 guided upsampler에서 흔한 의미 누출을 완화한다. 결과적으로 dense prediction뿐 아니라 semantic correspondence에서도 특징 충실도를 더 직접적으로 개선하는 방향을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 고해상도에서 teacher 역할을 할 ViT 특징을 효율적으로 얻는 것, (2) 좌표 기반 디코딩에서 nearest token만으로는 고차원 의미를 충분히 복원하기 어렵다는 점이다. 논문은 student-teacher 증류로 고해상도 teacher 비용을 줄이면서도 여러 해상도에 대한 특징을 다중 스케일로 감독하고, 좌표-조건 디코더 내부에 nearest patch 토큰의 상대 오프셋을 이용한 FeatX(위치 조건부 FiLM 기반)로 sub-token 디테일을 복구한다. 또한 분포 변화에 더 잘 적응하기 위해 LoRA로 patch embedding 및 attention 투영들에 저랭크 적응을 적용해, 백본을 전면 미세조정하지 않고도 고해상도 예측 품질을 끌어올린다.

- **Empirical Impact**: ViT-Up은 dense prediction에서 기존 image-guided upsampler 대비 일관된 성능 향상을 보이며, DINOv3-S+에서 Cityscapes mIoU는 최대 +2.07, SPair-71k PCK@0.10은 +4.17만큼 개선됐다. 더 큰 DINOv3-B 백본으로 확장하면 향상 폭이 Cityscapes +3.36 mIoU, SPair-71k PCK@0.10 +8.09로 커져 스케일링이 유리함을 보여준다. 의미 대응(semantic correspondence)에서의 큰 개선은 단순 시각적 선명도보다 의미 정렬과 공간 정합이 실제로 좋아졌다는 점을 뒷받침한다.



### RT-VLA: Real-Time Vision-Language-Action Models via Knowledge Distillation (https://arxiv.org/abs/2606.14010)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 자율주행(E2E) 모델은 언어 추론과 행동 예측을 한데 묶어 해석가능성을 높였지만, 큰 비전-언어 백본과 생성형(자기회귀) 추론 때문에 추론 지연이 커 도로 환경에 바로 배치하기 어렵다는 한계가 컸습니다. DriveCoT, ORION, AutoVLA, OpenDriveVLA, SimLingo 계열은 언어 기반 추론을 강화했지만 실시간 제어 관점에서는 계산 비용이 병목이 되었습니다. 또한 기존 증류는 주로 행동/궤적 출력에 초점이 맞춰져 있어, 언어 설명 능력을 실시간 제어와 분리해 유지하는 설계가 부족했습니다.

- **Core Contribution**: RT-VLA는 SimLingo의 주행과 언어 추론 능력을 더 작은 학생 모델로 “다단계 지도형 증류”해 실시간 추론이 가능한 경량 VLA를 제안합니다. 비전 특징, 쿼리 표현, 웨이포인트 예측, 언어 로짓까지 여러 수준에서 교사를 따라가 언어 기반 추론 및 주행 성능을 동시에 보존하도록 설계했습니다. 더 나아가 안전-중요 장면에서만 오프라인 언어 분석(사후 설명)을 수행해, 실시간 제어 경로에는 지연을 추가하지 않게 했습니다.

- **Technical Challenges**: 문제는(1) 교사와 학생의 비전 토큰 길이/차원이 달라 직접 특징 매칭이 어렵고, (2) 설명용 언어 모듈을 학생이 잘 재현하려면 단순 출력 증류만으로는 “생성 분포 불일치(shift)”가 생길 수 있다는 점입니다. RT-VLA는 정렬 모듈로 비전·쿼리·예측 시퀀스를 맞춘 뒤, 웨이포인트/언어 로짓의 다단계 증류 손실을 구성했습니다. 또한 오프라인 증류 후에는 on-policy 언어 파인튜닝으로 학생이 실제로 생성한 토큰을 다시 교사에게 평가해 KL을 최소화, 사후 설명의 일관성을 높였습니다. 마지막으로 언어 추론 브랜치를 폐루프 제어에서 분리하고, 필요 시에만 로그된 프레임을 입력으로 후처리하도록 KV 캐싱 기반의 경량 생성도 적용했습니다.

- **Empirical Impact**: Bench2Drive(CarLARO v0.9.15)에서 RT-VLA는 폐루프 주행 점수 85.19로 SimLingo(85.07) 수준을 유지하며, SimLingo-BASE(85.94)에 대해서도 격차를 0.75점으로 줄였습니다. 추론 속도는 비전 전용 모드에서 1544.34ms→34.48ms로 44.8배, 비전+언어 모드에서는 7.9배 가량 빨라져 실시간 제어의 실용성을 크게 높였습니다. 언어 설명 품질은 DeepSeek-V4-Flash 평가 기준 50.9로 교사(SimLingo) 대비 0.9점 낮은 수준에 그쳐, 지연을 줄이면서도 설명 능력을 상당 부분 보존했음을 보여줬습니다. 특히 지연이 줄어든 만큼 장애 회피·차로 변경 같은 정성 사례에서 초기 반응과 합류 타이밍이 개선되어, 시간 제약이 큰 도심 시나리오에서 의미 있는 효과를 확인했습니다.



### HARBOR: Heading Analysis and Reconstruction from Behavioral Observation and Radar (https://arxiv.org/abs/2606.14006)
- **Prior Approaches**: 기존 SAR 기반 접근은 다중 시계열 SAR을 선행해 추적하는 방식이나, 추론 시점에 AIS를 함께 써서 속도·진행방향을 복원하는 방식이 많았다. 또 일부는 SAR 검출 결과에 대해 시뮬레이션으로 위치를 투영하지만, 단일 프레임에서 방향(heading)까지 자동 추정하면서 확률적 궤적을 함께 내는 통합 파이프라인은 드물었다. 이 때문에 AIS가 끊기거나 스푸핑·비협조가 발생하는 상황에서는 성능이 급격히 흔들릴 수 있다.

- **Core Contribution**: HARBOR는 단일 SAR 스냅샷만으로도 선박의 향후 위치를 ‘확률 히트맵’ 형태로 예측하도록, 이미지 분석과 운동 모델을 분리하는 설계를 제안한다. AIS는 추론 때 쓰지 않고 오프라인 캘리브레이션 단계에서 선박 유형(크기)에 따른 속도 및 각도 분산 같은 운동 파라미터만 통계적으로 도출하는 데 한정된다. 그 결과 데이터 부재 환경에서도 방향을 반영한 단기 위치 예측을 수행한다.

- **Technical Challenges**: 핵심 난제는 단일 프레임에서 시간 정보 없이 heading을 어떻게 추정하느냐이다. 논문은 이 문제를 위해 선박 후보를 형태학적 전처리·연결요소 분할로 뽑고, skeleton(골격) 끝점의 국소 강도 패턴을 이용해 선미/선수를 구분한 뒤 단위 방향벡터로 heading을 복원한다. 이후 AIS로 캘리브레이션한 속도와 각도 분산을 이용해 heading 정렬 2차원 확률 분포(방향성 콘 포함)를 만들고 전역 히트맵은 후보들의 확률을 ‘합산’해 혼잡도까지 반영한다.

- **Empirical Impact**: 실제 COSMO-SkyMed(브라질 남부) SAR 장면에서 19,217×17,496 픽셀 입력 기준, 형태학 기반 자동 검출 후 27개 후보에 대해 heading과 360분 범위 확률 투영을 생성했으며 총 처리 시간은 약 6분 34초였다. 결과는 선박 규모 분포가 항만·어업 맥락과 질적으로 맞물리고, 정적 스냅샷에서도 운동 ‘경향’과 방향 인지 예측이 가능함을 보여준다. 다만 heading의 bow/stern 신뢰도가 낮은 케이스가 많아(예: 상대 강도 차이 10% 미만 비율 높음) 정량적 궤적 정확도 검증은 추가 연구가 필요하다고 밝힌다.



### Context-Guided Semantic Alignment for Feature Fusion Networks (https://arxiv.org/abs/2606.14005)
Comments:
          26 pages, 12 figures, 8 tables

- **Prior Approaches**: 기존 FPN 계열은 백본의 서로 다른 해상도·수용영역 특징을 단순 덧셈/이어붙이기 등으로 융합하지만, 저수준 특징은 공간은 정밀하나 의미가 약하고 고수준 특징은 의미는 풍부하되 공간이 거칠어 ‘의미 불일치’가 발생한다. 이를 완화하려고 채널 재가중(SENet, FaPN, AFF)이나 각 피라미드 레벨에 대한 일관된 감독( AugFPN 등), 또는 피라미드 구조 자체를 재설계하는 방식이 제안됐지만 계산 비용이 크거나 기존 검출기에 플러그앤플레이로 넣기 어렵다.

- **Core Contribution**: 이 논문은 피처 융합 전단에서 저수준 특징을 고수준 문맥으로 ‘의미 정렬’해주는 경량 모듈 FINE(Feature Interaction NEtwork)을 제안한다. FINE은 저수준 특징에 대하여 크로스 레벨 주의(cross-level attention)로 정렬된 공간-채널 조절 맵을 만들고, 잔차 기반 곱셈으로 의미 관련 픽셀만 선별적으로 강화해 정밀한 위치 정보도 보존한다.

- **Technical Challenges**: 핵심 문제는 피라미드 레벨마다 유효 수용영역(ERF)이 달라서 단순한 픽셀-대-픽셀 또는 고밀도 토큰 주의가 ‘정렬되지 않은 대응’을 학습하게 되고 계산도 기하급수로 늘어난다는 점이다. 이를 해결하기 위해 Alignment-Aware Token Sampling(AATS)로 두 레벨의 ERF 스케일을 맞춘 토큰으로 압축해 어텐션 복잡도를 크게 낮추고(약 1차 자릿수 수준 감소), 그 결과를 업샘플링해 잔차 잉여(modulation) 형태로 적용하는 전략을 쓴다.

- **Empirical Impact**: MS COCO에서 FINE은 여러 검출기 백본/목표 구조에 공통으로 성능을 올리면서도 계산 오버헤드는 거의 추가하지 않는다고 보고한다. 특히 작은 물체(APs)에서 개선 폭이 두드러지며, 오탐(False positives)과 배경 잡음에 의한 오분류를 줄이는 방향으로 이득이 설명된다. Faster R-CNN·RT-DETR·YOLO 계열 등에서 AP 향상과 동시에 FPS/지연 측면의 실시간 처리 성능도 유지되는 결과를 제시해, 의미 불일치 완화가 실제 효율적 정확도 향상으로 이어짐을 보인다.



### Prompt2Effect: Training-Free Image-to-Video Model Specialization via LoRA Generation (https://arxiv.org/abs/2606.13971)
- **Prior Approaches**: 기존 Image-to-Video(I2V)에서 효과 수준 개인화를 하려면 보통 각 효과마다 LoRA를 별도로 학습해야 한다. 이 과정은 효과별 데이터 정리와 반복 최적화가 필요해 인터랙티브 창작 워크플로를 막고, I2V처럼 시공간 차원이 큰 모델에서는 확장성 병목이 더 커진다. 가중치 생성(하이퍼네트워크) 방식도 있었지만, 주로 의미 임베딩만으로 어댑터 가중치를 회귀해 고차원 I2V용 LoRA 파라미터를 안정적으로 예측하기 어렵다는 한계가 드러난다.

- **Core Contribution**: Prompt2Effect는 효과 프롬프트로부터 특정 효과용 LoRA 가중치를 한 번의 순전파에서 합성해, 효과별 학습 비용을 분산(상각)하는 학습비용 절감형 접근을 제안한다. 핵심 아이디어는 (1) 냉동된 베이스 모델 가중치에 명시적으로 조건을 거는 “weight-driven” 하이퍼네트워크와 (2) LoRA 회귀의 비식별성 문제를 줄이기 위한 SVD-canonicalized 파라미터화다. 그 결과, I2V 확장에서도 효과 정합과 비디오 품질을 유지하며 빠른 제어가 가능해진다.

- **Technical Challenges**: 하이퍼네트워크가 텍스트 의미만으로 대규모·구조적인 LoRA 갱신을 회귀하려면, 베이스 레이어의 입력-출력 좌표 및 상관 구조가 빠져 있어 문제의 비일의성이 커지고 학습이 불안정해질 수 있다. Prompt2Effect는 이 점을 해결하기 위해 냉동 베이스 가중치 W0를 토큰 형태로 하이퍼네트워크 입력에 포함해 층별 기하적 선행지식을 제공한다. 또한 LoRA 분해(BA)는 요인 교체로 동치 표현이 생겨 안정성이 떨어지는데, 이를 SVD 기반 정준화된 요인으로 예측하여 해공간의 중복과 요동을 줄였다.

- **Empirical Impact**: 실험에서 Prompt2Effect는 기존 효과별 LoRA 파인튜닝과 비교해 비디오 품질과 효과 정합 측면에서 동급 또는 더 나은 성능을 보이면서도, 효과당 GPU 학습 56시간을 3.3초의 하이퍼네트워크 추론으로 대폭 줄인다. 또한 예측 가중치를 초기값으로 삼아 후속 LoRA 파인튜닝을 수행하면 최종 성능이 더 좋아지고 최적화 속도도 약 10배 빨라진다. 더 나아가 SVD 정준화와 weight-driven 설계는 고차원 I2V 설정에서 예측이 안정적으로 수렴하도록 돕는 것으로 분석된다.



### CaricHarmony: Contrastive Diffusion Paths for Identity-Preserving Caricature Synthesis (https://arxiv.org/abs/2606.13964)
- **Prior Approaches**: 스케치 기반 카리커처 합성은 신원(Identity)과 형태(Shape)라는 상충 요구를 동시에 만족해야 하지만, 기존 방법들은 대체로 한쪽을 더 우선시해 균형이 무너지는 문제가 있었다. GAN 기반 워핑 기법들은 왜곡 자체를 자동으로 만들거나 랜드마크/제어점에 크게 의존해 사용자 의도에 대한 정밀한 통제가 부족했고, 확실히 제어하려는 확산(디퓨전) 기반 접근은 ID-형태 충돌의 근본 원인을 제대로 다루지 못했다.
특히 DemoCaricature는 샘플/신원별 고비용 파인튜닝이 필요하고, CaricatureBooth는 Bezier 곡선 같은 입력 제약을 강하게 둬 창의적 스케치 입력을 제한한다.

- **Core Contribution**: 이 논문은 ID-형태 갈등의 뿌리를 디퓨전의 ‘조건 신호 오염(condition signal contamination)’으로 정의하고, 이를 통해 균형 생성이 구조적으로 붕괴되는 이유를 설명한다. 그리고 CaricHarmony는 학습 없이(inference-time training-free) 이 오염을 직접 해소하기 위해 세 갈래 디퓨전 경로를 병렬로 유지한다: 순수 ID 경로, 순수 형태 경로, 그리고 두 조건을 조화한 메인 경로다.
조화 경로는 교차-어텐션(cross-attention) 특징에 기반한 에너지 함수로 단계별 그래디언트 안내를 받아 중간 영역에 머물도록 유도된다.

- **Technical Challenges**: 핵심 기술 난제는 (1) ID와 형태 조건을 단순 결합하면 디노이징 궤적에서 확률 분포가 서로 경쟁하며 한 극단으로 ‘붕괴’한다는 점과, (2) 왜곡이 심한 카리커처에서 기존 ID 정합 손실은 도메인 간 격차로 약해진다는 점이다. CaricHarmony는 이를 피하기 위해 메인 경로가 참조(순수 ID·순수 형태) 경로의 ‘오염되지 않은’ 특징 분포를 따라가도록 교차-어텐션 정렬을 적용한다.
구체적으로는 형태 충실도를 위한 레이아웃/의미 정렬 에너지와, 극단 왜곡에도 견고한 토큰 수준 대응 매칭 기반의 ID 에너지를 설계하고, 두 에너지를 디노이징 타임스텝에 따라 시점별로 조절해(거친 단계엔 형태, 미세 단계엔 ID) 균형 붕괴를 예방한다.

- **Empirical Impact**: 실험에서 CaricHarmony는 WebCaricature 데이터셋에서 종합 품질과 조건 만족도를 동시에 끌어올렸고, 특히 형태 CLIP 점수 0.8615로 기준 접근 대비 개선을 보이면서도 신원 일관성을 유지했다. 사용자 선호도도 7.81로 나타나, 단순 보존/단순 왜곡이 아니라 ‘조화된 카리커처’ 생성이 실제 사용 관점에서 선택된다는 점을 보여준다.
또한 신원별 파인튜닝 없이 16초 내 생성으로 DemoCaricature의 70초 수준 제약을 크게 완화하며, 스케치 입력 형식 제약도 줄여 실사용 접근성을 크게 높였다.



### Self-Evolving Visual Questioner (https://arxiv.org/abs/2606.13929)
Comments:
          21 pages, including references and appendix. Project Page is available at this https URL

- **Prior Approaches**: 기존 시각언어모델(VLM)은 주로 인간이 만든 질문을 받아 정답을 잘 내는 ‘수동 답변자’로 학습·평가됐다. 시각 질문 생성(VQG)도 데이터셋/인간 라벨/외부 생성기(더 강한 모델) 의존도가 높아, 생성 질문의 다양성·시각적 접지(grounding)·추론 난이도가 근본적으로 고정 분포에 묶이는 한계가 있었다. 일부 자기학습 시도는 있었지만, 질문 생성 자체를 ‘지속 진화하는 능력’으로 다루기보다 QA 성능 향상의 보조 역할에 머무는 경우가 많았다.

- **Core Contribution**: 이 논문은 VLM이 외부 감독 없이도 스스로 더 어려운·더 정보적인·시각적으로 근거 있는 질문을 만들며 QG 능력을 연속적으로 개선하는 ‘자기진화형 시각 질문자’ 프레임워크를 제안한다. 핵심은 VLM을 질문 제안자(proposer)와 필터(filter)로 동시에 쓰고, 생성된 질문-정답 쌍을 QA/QG 이중 포맷으로 학습해 질문 생성과 답변 능력을 함께 유지·강화하는 점이다. 또한 제안 분포의 탐색 붕괴를 막기 위해 탐색 다양성을 보존하도록 설계했다.

- **Technical Challenges**: 가장 큰 기술적 난제는 자기 생성 데이터만 늘리면 모델이 반복 템플릿이나 얕은 인식에 수렴해 ‘붕괴(collapse)’하거나, 시각적 접지와 추론 난이도가 떨어질 수 있다는 점이다. 이를 해결하기 위해 (1) 여러 시각 의도(intent)로 후보 질문 풀을 넓히고, (2) rewrite로 난이도 방향(시각 탐색·증거 범위·문맥 추론·공간 추론)을 강화하며, (3) 원래 제안과 비교해 시각적 타당성/접지/난이도 향상이 있는 것만 필터링한다. 마지막으로 QA 포맷 학습을 함께 섞어 답변 능력의 드리프트를 억제한다.

- **Empirical Impact**: 여러 VLM 백본(Qwen 계열)에서 QG의 5개 차원(시각 탐색 난이도, 증거 커버리지, 문맥 추론, 공간 추론, 질문 다양성)이 일관되게 크게 개선됐고, QA 성능은 대체로 유지되거나 일부 벤치마크에서 향상됐다. 특히 2라운드 자기진화 후 QG 점수가 초기 모델 대비 약 82% 상승했으며, 생성 질문은 이미지 근거를 더 잘 찾아내고 더 넓은 내용/더 복잡한 추론을 요구하는 쪽으로 이동했다. 또한 개선된 질문 품질로 후속 QA 학습을 하면(동일 예산) 하위 성능도 더 잘 오르며, 단순 텍스트 유사도가 아닌 ‘더 정보적인 시각·추론 질문’이 감독 신호로서 가치가 있음을 실험적으로 뒷받침한다. 



### Overhead Wildlife Locator (OWL): Benchmarking Weakly Supervised Learning for Aerial Wildlife Surveys (https://arxiv.org/abs/2606.13911)
Comments:
          16 pages, 4 figures, 3 tables

- **Prior Approaches**: 기존 항공 야생동물 검출은 YOLO나 Faster R-CNN처럼 바운딩 박스 라벨을 요구해, 라벨링 비용과 시간이 크게 든다는 한계가 있었다. 점 라벨(개체 중심점) 기반 대안도 있었지만, 대표적으로 HerdNet은 Gaussian 대신 FIDT로 더 선명한 피크를 만들면서도 종 분류를 함께 최적화하는 멀티태스크 구조가 국소화 정밀도를 떨어뜨릴 수 있다는 지적이 있었다. 또한 POLO 같은 앵커리스 포인트 방식은 고정 반경 가정 때문에 밀집 장면에서 중복/누락이 생겨 언더카운트로 이어질 수 있다.

- **Core Contribution**: 이 논문은 바운딩 박스가 아닌 점 라벨만으로 개체 위치를 복원하는 약지도 밀도추정 프레임워크 Overhead Wildlife Locator(OWL)를 제안한다. OWL은 동일한 목적(개체 중심 피크 추출) 아래에서도 인코더 설계 스펙트럼을 3가지로 나눠 OWL-C(완전 합성곱), OWL-T(스윈 보강 하이브리드), OWL-D(동결 DINOv3 파운데이션 인코더+ DPT 스타일 디코더)를 비교한다. 이를 통해 어떤 관측 환경(희박/밀집, 복잡 배경)에 어떤 변형이 더 적합한지까지 정량적으로 보여주는 것이 핵심 기여다.

- **Technical Challenges**: 가장 큰 기술 난제는 점 라벨로부터 바운딩 박스 없이도 ‘작고 밀집된 개체’를 배경 잡음과 분리해 피크를 날카롭게 만들고, 그 피크를 개체 수로 일관되게 변환하는 과정이다. OWL은 Gaussian의 블러 문제를 해결하기 위해 FIDT(더 정확히는 FIDT 계열의 focal inverse distance transform)를 사용해 피크를 뚜렷하게 만들고, 예측 밀도맵에서 로컬 최대값 탐지로 개체 좌표를 회수한다. 또한 OWL-D에서는 동결된 DINOv3의 일반화 표현을 DPT 스타일 멀티스케일 융합 디코더로 연결하고, 학습에서는 희소한 양성(중심점) 극단적 불균형에 맞춰 CenterNet의 focal loss를 적용한다.

- **Empirical Impact**: 다섯 개 공중 이미지 데이터셋에서 POLO 및 YOLOv11 계열과 비교했으며, Delplanque의 HerdNet 기준선 대비 OWL-D가 Delplanque에서 0.934 AP로 HerdNet(0.840)을 능가하는 새 최첨단을 기록했다. 성능은 관측 ‘레짐’ 의존적이어서 SheepCounter UAV처럼 극도로 밀집된 조건에서는 OWL-T가 최고(0.978 AP)를 보인 반면, 파운데이션 기반 OWL-D는 성능이 저하되는 패턴이 나타나 변형 선택의 실무적 가이드가 제공된다. 더 나아가 Alaska Department of Fish and Game의 2022년 Central Arctic Caribou census 배치 검증에서 OWL-C 미세조정 모델이 패치 테스트에서 F1=0.965, 부호 있는 총 개체 수 오차 +3.1%로 운영 준비도를 입증했으며, OWL 코드와 모델 가중치 및 패치 단위 카리부 데이터셋(PCH·CAH)을 공개해 연구 확장에도 의미가 크다.



### PMOF: A Dataset and Benchmark for Passenger Monitoring Using Overhead Fisheye Cameras (https://arxiv.org/abs/2606.13910)
Comments:
          6 pages, 7 figures. Accepted to the 22nd IEEE International Conference on Advanced Visual and Signal-Based Systems (AVSS 2026)

- **Prior Approaches**: 기존 대중교통 객실 내 모니터링 연구는 승객 수/행동 인식처럼 안전·운영 목적에 집중해 왔지만, 실제 움직이는 차량 환경에서의 강건한 인식보다는 정적/부분 환경에 치우치는 경우가 많았습니다. 천장 고정 오버헤드(fisheye) 카메라를 쓰는 연구도 있었지만, 공개 데이터셋은 대체로 정적 사무공간 위주이거나(office domain) 주석 범위가 제한적이었습니다.
또한 움직임으로 생기는 도메인 차이(차량 운동에 따른 배경 변화)를 반영한 공개 데이터가 없어 실사용 일반화가 어려웠습니다.

- **Core Contribution**: 이 논문은 움직이는 차량 안에서 촬영한 오버헤드 피시아이 이미지를 공개한 최초의 데이터셋 PMOF를 제안합니다. PMOF는 19,696 프레임(19k+), 사람·의류·가방 클래스에 더해 회전 바운딩 박스, 추적 식별자, 행동 라벨(앉음/바닥 앉음/서있음/누움)을 제공해 탐지·추적·행동인식을 함께 다룰 수 있게 합니다.
특히 정적 환경 데이터와의 도메인 갭을 실증적으로 다루고, 다른 오버헤드 피시아이 데이터셋으로의 전이까지 벤치마킹하는 구성을 제공합니다.

- **Technical Challenges**: 움직이는 차량 환경은 공간이 좁고 조명이 변하며, 배경이 모션에 따라 흔들리고, 가림(occlusion)이 잦아 일반적인 정적 데이터 학습만으로는 성능이 크게 떨어집니다. 또 피시아이 특유의 기하 왜곡 때문에 회전된 바운딩 박스를 학습에 적합하게 유지하는 증강 설계가 중요합니다.
논문은 회전 바운딩 박스의 네 모서리를 키포인트처럼 다뤄 회전 각도를 보존하는 rotation-aware augmentation을 만들고, translation/cropping/mosaic처럼 원형 기하를 깨는 변형은 제외해 학습 일관성을 확보했습니다.

- **Empirical Impact**: 실험 결과, 단독 PMOF 미세조정은 AP50 89.9% 수준으로 오피스 데이터(예: CEPDOF)만 쓰는 경우(최대 83.8%)보다 개선됐고, PMOF에 대한 회전-aware 증강을 더하면 94.8% AP50까지 올라갔습니다. 더 나아가 CEPDOFaug + PMOFaug로 학습했을 때 외부 도메인 오버헤드 피시아이 데이터셋 HABBOF에서 96.5% AP50를 기록해, 정적-이동 도메인 전이의 효과를 수치로 확인했습니다.
또한 피시아이 전용 구조 변경 없이 일반 YOLO26m-obb를 사용해도 강한 성능을 보여, PMOF와 증강이 승객 모니터링을 넘어 더 넓은 피시아이 기반 사람 탐지 연구의 기반이 될 수 있음을 시사합니다.



### HiLo-Token: Input-Adaptive High-Low Frequency Token Compression for Efficient Image Editing (https://arxiv.org/abs/2606.13898)
Comments:
          14 pages, 10 figures, Patent filled

- **Prior Approaches**: 기존 생성형 이미지 편집은 사용자 마스크를 입력으로 받아 처리하지만, DiT(디퓨전 트랜스포머)로 넘어가면 토큰 수 증가로 인해 지연(latency) 비용이 크게 늘어나는 문제가 있었다. 토큰/모델 압축, 활성 캐싱, 저비트 양자화 등은 효과가 제한적이거나(혹은) 품질 저하가 동반돼 프로덕션 배포엔 부담이 컸다. 또한 마스크 기반 편집을 직접 최적화한 연구는 일부 있었지만, 컨텍스트 손실을 줄이면서 효율을 동시에 확보하는 데 한계가 있었다.

- **Core Contribution**: 이 논문은 DiT 기반 마스크 편집의 지연 병목을 줄이기 위해 HiLo-Token을 제안한다. 핵심은 입력(사용자 마스크/영상 특성)에 적응적으로 토큰 예산을 배분해, 편집 영역 주변의 국소 정보(고주파)와 전역 구조(저주파)를 동시에 유지하면서 불필요한 토큰 계산을 줄이는 것이다. 특히 디퓨전 변환기에서 품질 회귀 없이 압축을 가능하게 하는 “주파수 기반 선택+다중 해상도 토큰” 설계를 채택했다.

- **Technical Challenges**: 가장 큰 기술 과제는 공격적인 토큰 삭제 시에도 편집 품질을 유지하는 것이었다. 단순히 확장 마스크(dilated mask) 안의 토큰만 남기면 컨텍스트 모양이 달라져 유용한 정보를 버리게 되는데, 이를 보완하기 위해 고주파 토큰은 소벨(Sobel) 기반 공간 주파수 맵에서 선택하고, 저주파 토큰은 16배 다운샘플된 정보로 전역 구조를 대표하도록 결합한다. 다음으로 컨텍스트 추출 비용을 줄여야 했는데, 값비싼 컨텍스트 인코더 대신 저비용의 공간 주파수 계산과 경량 패치 임베딩으로 토큰 선택 오버헤드를 약 10ms 수준으로 제한했다.

- **Empirical Impact**: 대규모 프로덕션급 평가 데이터에서 HiLo-Token은 DiT만 기준으로 A100-80GB에서 마스크 비율 구간별로 3.13배/2.59배/1.67배 가속을 달성했으며, 생성 품질 저하 없이 동작했다고 보고했다. 끝단(end-to-end) 파이프라인 기준으로도 1.33배/1.66배/1.77배의 전체 속도 향상이 확인됐고, Amazon AWS p5.48xlarge 노드 요구량은 Remove 기능 기준 33% 감소로 이어졌다. 또한 텍스트 포함 generative fill 등 사용자 시나리오 중심의 사용자 연구에서도 동률 비율이 높거나 일부 조건에서 HiLo-Token이 더 좋은 결과를 보였다.



### How do Self-Supervised Remote Sensing Vision Models Transfer to Downstream Tasks? (https://arxiv.org/abs/2606.13896)
- **Prior Approaches**: 기존 GeoFM 연구는 성능(벤치마크 점수)이나 최종 임베딩을 기준으로 모델을 비교해 왔지만, 과업과 적응 설정에 따라 어떤 정보가 실제로 활용되는지까지는 충분히 규명되지 않았다. 특히 원격탐사는 자연영상과 달리 경계가 불명확하고 스펙트럼·공간·시간·지리 맥락의 영향을 크게 받는데, 표준 전이 헤드는 이러한 깊이별 정보 구성과 잘 맞지 않을 수 있다. 그 결과, 선행 연구는 특정 벤치마크에선 잘 보이지만 다른 과업·설정에선 뒤처지는 현상을 보고해 왔다.

- **Core Contribution**: 이 논문은 대표적인 6개 GeoFM을 대조/지식증류/재구성/멀티모달 사전학습 계열로 묶고, 분류·회귀·세그멘테이션을 대상으로 라벨 가용성과 다운스트림 파이프라인(고정 vs 미세조정, 디코더 설계)을 바꿔가며 전이 거동을 체계적으로 분석한다. 또한 레이어별 선형 프로빙과 depthwise 프로파일링으로 “어느 블록에서 과업에 필요한 정보가 더 잘 접근되는지”를 보여주어, 벤치마크 간 순위가 흔들리는 원인을 해석 가능하게 만든다.

- **Technical Challenges**: GeoFM 전이는 최종 임베딩만으로는 설명이 부족해, 네트워크 깊이에 따른 정보 접근성(저수준 신호 vs 의미 정보)과 적응 단계에서의 변화까지 동시에 측정해야 한다. 논문은 레이어별 선형 프로브로 과업 관련 정보의 접근 위치를 추정하고, CKA로 미세조정이 표현 공간을 깊이별로 얼마나/어디에 국소적으로 바꾸는지(특히 ViT 블록의 MLP 첫 선형층) 정량화한다. 더 나아가 세그멘테이션에서는 디코더 설계(라이트 멀티스케일 vs UPerNet 등)가 GeoFM 선택만큼 큰 영향을 줄 수 있음을 실험적으로 확인한다.

- **Empirical Impact**: 실험 결과 모델 순위는 과업 종류와 적응 설정에 따라 크게 변하며, 동일 모델군 내에서도 저수준/의미 과업에 대한 깊이별 성숙도가 달라 “최종층 기준 비교”의 한계가 드러난다. 재구성 계열(MAE, Prithvi)은 대체로 저수준 정보가 더 깊게 유지되는 경향이 있고, joint-embedding 계열(MoCo, DINO)은 초중반에 강하게 나타난 뒤 의미 정보로의 전이가 더 급격하게 전환되는 양상이 관찰된다. 또한 PASTIS와 Sen1Floods11의 사례 분석에서 미세조정은 전 깊이를 균일하게 재작성하지 않고 국소적으로 변화를 일으키며, 세그멘테이션 헤드가 정보 조직 방식과 불일치할 수 있어 평가와 적응 전략을 ‘표현 인지형’으로 설계해야 한다는 실무적 시사점을 제공한다.



### Avatar V: Scaling Video-Reference Avatar Video Generation (https://arxiv.org/abs/2606.13872)
Comments:
          31 pages, 15 figures. All contributors are listed in alphabetical order by first name

- **Prior Approaches**: 기존의 사람(인물) 기반 talking avatar 생성은 대체로 한 장의 정적 이미지에 조건을 거는 방식이 많아, 보이지 않던 각도·표정·조음 패턴을 생성 과정에서 추정해야 합니다. 또한 비디오 레퍼런스를 쓰더라도 참조 토큰을 그대로 붙이면 attention 비용이 폭증해 긴 영상 조건을 다루기 어렵고, 픽셀 단위 학습은 얼굴 중에서도 립·눈·미세 표정 같은 핵심 영역의 세밀함을 충분히 끌어올리기 어렵다는 한계가 있습니다.

- **Core Contribution**: Avatar V는 인물의 “모양”뿐 아니라 말하기 리듬·미세 표정·제스처 같은 “행동”까지 행동적으로 구분 가능한 아바타를 목표로, 짧은 레퍼런스 비디오를 직접 조건으로 삼는 비디오-레퍼런스 기반 인물 모델링을 제안합니다. 핵심은 고정 크기 임베딩으로 인물을 압축하지 않고, 레퍼런스 비디오의 전체 토큰 시퀀스를 생성 모델이 attention으로 읽어 정적 신원 특성과 동적 말하기 스타일을 함께 재현하도록 한 점입니다.

- **Technical Challenges**: 가장 큰 기술 도전은 긴 레퍼런스 비디오를 그대로 conditioning할 때 발생하는 계산량(특히 attention의 비용)과, 말하기 스타일·미세 표정처럼 시간적으로 중요한 신호를 학습/전달하는 문제였습니다. Avatar V는 Sparse Reference Attention으로 조건부 attention을 레퍼런스 길이에 대해 거의 선형으로 만들고, motion representation 스트림과 closed-loop 학습 신호로 말하기 스타일을 별도 축에서 정교하게 내재화하며, 정체성에 민감한 얼굴 디테일을 보정하는 identity-aware 초해상도 refiner와 사람 인지에 맞춘 보조 손실까지 결합해 세밀함을 확보합니다.

- **Empirical Impact**: 100M+ 학습 클립을 50M+ 원천 비디오에서 큐레이션하고, flow matching 기반 DiT에 5단계(사전학습→개인화 SFT→2단계 증류→RLHF) 학습을 대규모 GPU 환경에서 수행해 1080p급 고품질 talking avatar 장기 생성(무제한 길이)을 구현했다고 보고합니다. cross-scene 벤치마크에서 Seedance 2.0, Kling O3 Pro, Veo 3.1, OmniHuman 1.5를 자동 지표와 인간 평가 모두에서 일관되게 능가하며, “행동적으로 인물 구분이 되는 생성”을 실용 규모로 끌어올렸다는 점에서 분야의 기준점을 한 단계 높였다는 의미가 있습니다.



### Mirage Probes: How Vision Models Fake Visual Understanding (https://arxiv.org/abs/2606.13870)
- **Prior Approaches**: 기존 연구는 비전-언어 모델의 “mirage(환상) 행동”을 보통 하나의 실패 모드로 취급했다. 즉, 이미지가 없어도 정답처럼 보이는 출력을 내는 현상을 텍스트 편향·데이터 규칙·숏컷 탓으로 뭉뚱그려 설명하며, 내부 표현 수준에서 무엇이 다른지 검증하지 못했다.

- **Core Contribution**: 이 논문은 mirage 행동이 실제로는 두 가지 메커니즘으로 나뉜다고 주장하며, 이를 표현(representation) 레벨에서 분해하려는 Mirage Probes를 제안한다. 특히 시각 입력이 있는 경우와 없는 경우의 행동 프록시 라벨을 만들고, 질문 변형(의미는 유지, 표면 단서는 최소화) 대비쌍을 통해 내부 신호를 대비 학습적으로 측정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “이미지 없이도 그럴듯한 답”이 출력만으로는 원인이 보이지 않는다는 점이다. 저자들은 (1) 이미지 유무에 따른 행동 차이를 라벨로 근사하고, (2) 잔차 스트림·MLP·post-attention·어텐션 헤드 등 여러 층/위치에서 선형·대비 차분 방식으로 복원 가능성을 시험하며, (3) Naive Bayes 텍스트 기반 기준선으로 표면 어휘 단서 여부를 함께 통제한다.

- **Empirical Impact**: 두 개 오픈소스 VLM에서 mirage 정보는 이미지가 제공된 상태의 내부 활성에서 선형 방향으로도 유의미하게 디코딩되며, 특히 차분 활성(difference-of-activations) 프로브가 가장 깔끔하게 신호를 회수했다. 또한 벤치마크 간 분리 양상이 달라 “텍스트 편향”과 “spurious images(잠재 공간의 가짜 시각 생성)”가 서로 다른 정권(regime)임을 시사하고, 텍스트 분포 세정 같은 중재가 텍스트 편향에는 효과적일 수 있으나 spurious images에는 대표적으로 한계가 있음을 보여준다.



### Temporal Backtracking Search for Test-time Generative Video Reasoning (https://arxiv.org/abs/2606.13861)
- **Prior Approaches**: 기존 비디오 추론은 한 번 생성한 롤아웃을 통째로 보고 Best-of-N(BoN)처럼 전체 궤적을 뽑는 단발(single-shot) 패러다임에 기대는 경우가 많았습니다. 이 방식은 계산을 노이즈 스케줄이나 부분 잠재공간에만 적용해 영상 품질·프롬프트 정렬은 개선할 수 있어도, 논리적으로 잘못된 궤적을 초기에 되돌리기 어렵습니다.

- **Core Contribution**: 이 논문은 비디오 생성 추론의 스케일링 축을 “노이즈/디노이징 단계”가 아니라 “시간(temporal) 축”으로 옮겨, Temporal Backtracking Search(TBS)를 제안합니다. TBS는 검증된 앞부분(prefix)을 보존한 채 실패 지점을 찾아 해당 이후 구간만 생성-수정하도록 설계되어, 잘못된 롤아웃을 매번 처음부터 버리지 않습니다.

- **Technical Challenges**: 핵심 난관은 비디오 확산 모델이 “임의의 깨끗한 접두(prefix)에서 이어서(temporal branching) 생성”할 수 없다는 점입니다. 이를 위해 variable-K(=latents prefix 길이) 조건화를 도입해 재시작 앵커로부터 특정 잠재 프레임 길이만큼을 깨끗하게 고정한 뒤, 프로세스 검증기가 첫 실패 프레임을 국소화해 재시작 지점을 정하고, 접두 기반(prefix-based) 우선순위 프런티어로 계산을 올바른 분기 확장에 배분합니다.

- **Empirical Impact**: 실험은 알고리즘 그리드 월드(정확한 상징 검증), 연속 네비게이션(학습 기반 검증), 로봇 조작(시뮬레이터 재생 검증) 등 3개 영역에서 TBS가 동일 예산의 BoN보다 Pareto 우위임을 보였습니다. 특히 엄격한 분포 밖(out-of-distribution) 조건에서 단발 생성은 0.7% EM에 그친 반면 TBS는 22.7% EM을 달성했으며, 성공 에피소드 모두가 재시작된 분기에서 나왔습니다. 



### Explaining RhythmFormer: A Systematic XAI Analysis of Periodic Sparse Attention for Remote Photoplethysmography (https://arxiv.org/abs/2606.13839)
Comments:
          26 pages, 8 figures

- **Prior Approaches**: rPPG 트랜스포머의 XAI는 대체로 원시 attention, Grad-CAM/그래디언트 살리언시 같은 시각화 중심 접근에 머물렀습니다. 그 결과 “그럴듯해 보이는” 히트맵은 제공되지만, 마스킹했을 때 예측이 얼마나 변하는지 같은 정량적 faithfulness 지표나 생리학적 근거 검증이 부족했습니다. 또한 원시 attention을 모델의 진짜 초점으로 볼 수 있는지에 대해 회의적 논의가 있어, 단일 방법 의존의 한계가 두드러집니다.

- **Core Contribution**: 이 논문은 RhythmFormer의 주기적 sparse attention( bi-level routing, top-k 선택 )에 맞춘 XAI 프레임워크를 제안해, 시각적 해석과 감사 가능한 수치 증거 사이의 공백을 메웁니다. 구체적으로 원시 attention, attention rollout, attention flow, Beyond Intuition을 RhythmFormer's refined attention에 맞춰 적응하고, skin(얼굴/목) 영역 정렬을 정량화하는 skin coverage 지표와 rPPG 회귀용 SaCo faithfulness를 함께 도입합니다. 이를 통해 “어디를 봤는가”와 “그 주목이 실제로 예측을 얼마나 좌우하는가”를 동시에 측정합니다.

- **Technical Challenges**: 핵심 난점은 (1) sparse top-k 라우팅 때문에 누락된 연결이 많아 attribution을 누적/흐름 기반으로 조합하기 어렵고, (2) rPPG는 분류가 아니라 파형 회귀라 SaCo 같은 교란 기반 신뢰성 평가를 그대로 적용할 수 없다는 점입니다. 저자들은 top-k로 0이 된 위치를 보존하는 방식으로 희소→밀집 재구성을 수행하고, 전 레벨 attention을 동일한 시간 해상도로 다운샘플해 다층 행렬 곱/흐름 계산이 가능하게 했습니다. 또한 SaCo의 perturbation impact을 rPPG 파형의 MAE로 재정의해, 분류 신뢰도 하락 대신 모델 민감도만 반영하도록 설계했습니다.

- **Empirical Impact**: UBFC-rPPG에서 수치 실험을 통해 다중 hop leakage(희소 top-k 라우팅에서 사라진 연결이 누적 연산으로 다시 복원되는 현상)가 정량적으로 관찰됩니다. attention rollout과 flow는 특정 refined-attention 층에서 0으로 만든 연결을 거의 다시 복원하는 반면, Beyond Intuition은 value-projection 가중 rollout과 gradient 기반 마스크로 이를 완화해 skin coverage 중앙값(0.83 vs 0.57)과 faithfulness(F=0.92)를 가장 높게 달성했습니다. 나아가 낮은 SaCo outlier 사례에서도 artefactual 영역을 대체하면 네 방법 모두 안정적으로 회복되어, 제안 지표들이 다양한 attribution 계열에서 일관된 신호를 준다는 점을 시사합니다.



### Compressing Image Style Training into a Single Model Forward (https://arxiv.org/abs/2606.13809)
Comments:
          11 pages, 9 figures

- **Prior Approaches**: 확산 기반 스타일 전이는 두 갈래로 발전해왔다. Adapter 기반(ControlNet, T2I-Adapter, IP-Adapter)은 한 번 학습 뒤 추론이 빠르지만 스타일이 외부 조건으로만 주입돼 스타일 충실도와 프롬프트-레퍼런스 충돌, 의미 누출 문제가 생기기 쉽다. LoRA 기반 개인화(텍스트 인버전, DreamBooth, LoRA)는 더 잘 내재화하지만 스타일마다 별도 최적화 학습이 필요해 비용과 지연이 커진다.

- **Core Contribution**: 이 논문은 레퍼런스 이미지를 보고 LoRA 가중치를 직접 예측하는 i2L(image-to-LoRA)로, 스타일 LoRA 학습 과정을 “한 번의 모델 순전파”로 분산(아몰타이즈)한다. 즉, 기존처럼 스타일별로 반복 학습하지 않고도, 주입된 LoRA가 생성기 내부 가중치를 직접 갱신하도록 만들어 텍스트 기반 확산 생성의 내용 제어는 유지하면서 스타일만 빠르게 인스턴스화한다. 또한 예측된 LoRA를 표준 모듈로 제공해 이후 합성(다중 레퍼런스, 컨트롤 모듈 조합)까지 자연스럽게 확장한다.

- **Technical Challenges**: 핵심 기술 난제는 레퍼런스 신호가 소수 이미지에서 오더라도 생성기에 필요한 “대규모 LoRA 업데이트”를 스케일 맞게 생성하는 것이다. i2L은 SigLIP2 이미지 인코더의 패치 토큰과 학습 가능한 LoRA 쿼리를 입력으로 하는 변환기 구조를 쓰고, LoRA 행/열(row-and-column) 형태에 맞춘 압축 디코딩 헤드로 레이어별·랭크별 가중치를 생성한다. 또 레퍼런스의 콘텐츠 복사를 줄이기 위해 MegaStyle-1M처럼 스타일은 같고 콘텐츠는 다른 튜플로 학습해 스타일 일관성은 강화하고 의미 누출을 억제했다.

- **Empirical Impact**: Z-Image, FLUX.2, Hidream-O1에서 i2L은 기존 레퍼런스 조건 주입 방식 대비 스타일 충실도, 프롬프트 정렬, 지각 품질에서 전반적으로 개선을 보였다. 특히 CLIP-Text/스타일 관련 지표와 미학 평가 계열까지 폭넓게 경쟁력 있거나 최상 성능을 달성해 LoRA 가중치로 스타일을 내재화하는 전략의 실효성을 확인했다. 또한 예측 LoRA를 이용해 비대칭 classifier-free guidance, 다중 레퍼런스 스타일 융합, ControlNet·AttriCtrl·인페인팅 같은 컨트롤 모듈과의 조합이 추가 학습 없이 가능하다는 점에서 실사용 확장성도 강조된다.



### CineOrchestra: Unified Entity-Centric Conditioning for Cinematic Video Generation (https://arxiv.org/abs/2606.13768)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 텍스트-투-비디오 모델은 대체로 하나의 전역 캡션과 단일 샷 생성에 초점이 맞춰져, 다중 인물/사물, 시간 구간별 사건, 카메라 이동, 샷 전환을 동시에 정밀 제어하기 어렵다. 관련 연구는 인물 개인화, 시간 제어, 멀티샷 합성, 카메라 제어를 각각 따로 다루며, 서로 다른 아키텍처와 학습 데이터로 축을 분리해 최종적으로는 일관된 영화적 장면을 합치기 힘들었다.

- **Core Contribution**: CineOrchestra는 영화 장면의 네 축(인물/사물, 사건 타이밍, 카메라, 샷 전환)을 단일 비디오 확산 트랜스포머(Video DiT)에서 동시에 제어하는 통합 프레임워크를 제안한다. 핵심 아이디어는 각 요소를 ‘특정 시간 구간 동안 작용하는 엔티티’로 보고, 모든 조건을 동일한 엔티티-중심 구조(시작시간, 종료시간, 프롬프트)로 표현하되 카메라/{transition} 태그로 촬영·전환까지 자연스럽게 합성한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 서로 길이가 크게 다른 사건(0.1초 컷~10초 이동)을 동일한 정렬/주의 메커니즘으로 공정하게 라우팅하는 것이다. 이를 위해 파라미터 없는 두 종류의 협응 RoPE를 설계했는데, (1) 구간 내부를 샘플링하는 interval-sampled temporal RoPE로 이벤트 길이 차이에 따른 주의 유사도 감쇠를 보정했고, (2) 2D entity-temporal cross-attention RoPE로 엔티티 토큰을 구분하며 각 조건이 자신이 지시한 시공간 영역으로 정확히 향하도록 했다.

- **Empirical Impact**: CineBench와 CineBenchSyn의 두 새로운 벤치마크에서 CineOrchestra는 6개 축별 특화 모델을 밀도 캡션 추종과 샷 전환 타이밍에서 능가했으며, 쌍대 사용자 평가와 구성요소(ablations)에서도 일관된 개선이 확인됐다. 특히 퍼셉션 지표에서도 전반적 품질과 장면 차원에서 좋은 선호를 얻어, 영화적 구성 요소를 한 번에 조합하는 접근이 실제 생성 품질과 제어 정밀도 모두에 의미 있는 진전을 보였다는 점에서 영향력이 크다.



### Connections Between Pairs of Filters Improve the Accuracy of Convolutional Neural Networks (https://arxiv.org/abs/2606.13736)
Comments:
          IJCNN 2023

- **Prior Approaches**: 기존 CNN 개선은 보통 합성곱 블록을 쌓고 점별(pointwise) 활성함수로 비선형성을 분리하는 전통적 패턴을 따랐다. 점별 비선형이 충분히 선택적이지 못해 강건성이 떨어질 수 있다는 문제의식이 있었고, 이를 위해 쌍(pair) 단위 뉴런 상호작용(예: bilinear, Volterra, polynomial) 같은 연결이 제안돼 왔다. Feature-Product(FP)나 Min-nets는 두 필터 출력을 AND처럼 결합(곱셈, 최소값)해 특정 영역(예: 모서리·교차)을 잘 잡지만, AND 논리가 깊은 층에서 정보 손실로 이어질 수 있다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 두 깊이방향(depthwise) 컨볼루션의 출력을 결합하는 “connected block”을 제안하고, AND에 한정되지 않은 더 일반적인 연결 함수를 검토한다. 특히 연결 함수 자체에 학습 가능한 매개변수를 포함해, 층마다 다른 결합 방식(OR형에 가까운 형태부터 AND를 닮은 형태까지)을 선택하도록 만들어 과제에 더 잘 적응하게 한다. 결과적으로 연결 함수가 네트워크 표현을 “정보를 억지로 제거하기”가 아니라 “필요한 영역을 강조하고 유형을 구분”하는 방향으로 확장된다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 두 특성 맵을 결합하되 AND처럼 경직되지 않은 연결이 성능을 실제로 끌어올릴지, (2) 그 연결의 자유도를 학습으로 안정적으로 제어할 수 있는지였다. 저자들은 connected block 구조를 유지하되, ReLU-활성 XOR에서 출발한 형태의 연결 함수에 스칼라 매개변수 l을 두고 이를 블록(층)마다 따로 학습시키는 방식으로 해결했다. l 값이 0이면 OR처럼 합치고, 0.5면 XOR 성질을, 1이면 입력 부호/크기 관계를 더 강하게 반영하는 변형이 되도록 설계해 층별 최적 결합을 찾게 했다.

- **Empirical Impact**: CIFAR-10 분류 실험에서 connected block은 기본 블록만 쓰는 기준선보다 전반적으로 정확도를 높였다. 그중 학습 가능한 XOR 연결은 평균 92.4%로 가장 높은 성능을 보였고, 단순 baseline 대비 0.4%p, connected block이 없는 원래 아키텍처 대비 0.6%p 개선됐다. 또한 l이 스택마다 특정 패턴을 보였는데 특히 마지막 스택은 낮은 l을 학습해 OR에 가까운 결합을 선택하는 경향이 나타나, “AND형 선택은 초기에 유리하고 깊어질수록 유연한 결합이 필요”하다는 실증적 시사점을 제공한다.



### Morphology-Aware Sample Assignment: Overcoming IoU Insensitivity for Surface Defect Detection (https://arxiv.org/abs/2606.13723)
- **Prior Approaches**: 기존 시각 탐지 학습에서는 Intersection-over-Union(IoU)이 후보 박스와 정답 박스의 공간 정렬을 평가하며, 그 결과가 양성 샘플(positive sample) 할당과 학습 효율을 좌우한다. 그러나 IoU 응답 곡선에는 서로 다른 기하학적 겹침에도 유사한 IoU 점수가 나오는 비민감(non-sensitive) 구간이 존재해, 구조적으로 다른 샘플이 동일 취급되는 한계가 있다.

- **Core Contribution**: 이 논문은 IoU가 구조적 대응을 충분히 반영하지 못한다는 문제를 해결하기 위해, 면적·형상·종횡비를 아우르는 형태 유사도(morphological similarity) 지표 묶음을 제안한다. 또한 이 다차원 유사도를 평균 기반으로 집계해 보조 매칭 점수(supplementary matching score)를 만들고, 이를 양성 샘플 할당을 더 판별력 있게 수행하는 데 활용한다.

- **Technical Challenges**: 핵심 난제는 IoU의 비민감 구간처럼 응답 분포가 뭉개지는 상황에서도, 학습에 유효한 방향성 있는 그라디언트와 정답 인스턴스 주변을 정확히 감싸는 높은 반응 영역을 구성하는 것이다. 저자들은 형태 유사도 포함이 매칭 함수의 응답 분포를 재형성해 ‘폴리곤형’ 등반응(iso-response) 윤곽과 촘촘한 고반응 영역을 만들고, 그 결과 긍정 샘플 선택을 더 신뢰성 있게 만든다고 이론적으로 보인다.

- **Empirical Impact**: YOLOv9 프레임워크 기반 실험에서 NEUDET과 GC10-DET 데이터셋 모두에서 일관된 성능 향상을 확인했다. 특히 제안 방식은 플러그앤플레이 형태이며 추가 추론 비용이 0이어서 산업용 비전 검사 배포 효율까지 함께 확보하는 의미가 있다.



### TSA: Temporal Slot Activation for Persistent Object-Centric Video Representation (https://arxiv.org/abs/2606.13714)
- **Prior Approaches**: 기존 비지도 비디오 오브젝트-중심 학습은 Slot Attention 기반에서 슬롯을 매 프레임 “무조건” 갱신·디코딩하며, 보이지 않는(완전 가려진) 객체 슬롯도 계속 설명에 참여하는 설계를 흔히 사용한다. 이 때문에 장시간 가림 구간에서 슬롯 상태가 증거에 의해 흔들리는 drift와, 비활성 슬롯이 디코더 주의(attention)로 인해 재구성에 간섭하는 문제가 누적될 수 있다.

- **Core Contribution**: 논문은 슬롯의 생명주기(lifecycle)가 “활성일 때만 업데이트·설명”되어야 한다는 요구를 무감독으로 만족시키는 Temporal Slot Activation(TSA)을 제안한다. TSA는 슬롯별·시점별 활성도 alpha_{k,t}를 학습하고, 비활성 슬롯은 이전 상태에 고정(activation-gated 업데이트)하며 디코더의 attention 참여도 활성도 기반 바이어스로 억제한다. 또한 부분 가림과 서서히 재등장하는 상황을 위해 Temporal Context Encoder로 만든 per-slot temporal memory를 활성도 예측에 반영한다.

- **Technical Challenges**: 무감독 환경에서 “보임/미보임”을 직접 감독하지 않고도 슬롯 업데이트 타이밍을 학습하는 것이 핵심 난제다. TSA는 (1) 활성도에 따라 상태 업데이트를 게이팅하고 (2) 소프트맥스 전 주의 로짓에 가산 바이어스를 더해 재구성 경로의 간섭을 줄이며, (3) temporal memory로 활성도 판단을 안정화하는 방식으로 이를 해결한다.

- **Empirical Impact**: MOVi-C/E, YT-VIS, OVIS(심한 가림)에서 TSA는 객체 분해 품질과 시간적 동일성 보존을 전반적으로 개선하며, 특히 장시간·중첩 가림에서 큰 폭의 향상을 보인다. FG-ARI, mBO, IDF1, HOTA 같은 추적·일치 지표에서도 성능이 좋아지고, 추가로 YouTube-VIS HQ에서 슬롯 표현을 고정한 downstream 평가(인식+동역학 예측)에서도 TSA가 시간 예측의 일관성을 크게 높이는 것으로 입증된다. 요약하면 슬롯의 활성도 기반 생명주기 제어가 장기 가림에서의 슬롯 정체성 유지에 직접적인 실증 효과를 가져온다.



### Trimodal Glioma Representation Alignment via Volumetric Contrastive Learning (https://arxiv.org/abs/2606.14568)
- **Prior Approaches**: 기존 뇌종양 전이(예후) 예측·등급 모델은 주로 병렬 처리되는 쌍대(WSI-omics, MRI-histology 등) 멀티모달에 머물렀고, 정렬 목적도 대부분 두 양식 간 pairwise로 설계되는 경우가 많았습니다. 그 결과 추가 정보가 있는 세 번째 양식을 넣더라도, 서로의 의미적 정합을 통합적으로 학습하지 못하는 한계가 지적됩니다. 또한 쌍대 유사도 손실은 양식 수가 늘어날수록 확장성이 떨어져 ‘3자 정렬’의 이점을 충분히 활용하기 어렵습니다.

- **Core Contribution**: GLORIA는 병리(whole-slide histopathology), 전사체(mRNA), 3D MRI를 한꺼번에 정렬하는 GLioma Omics - Radiology - hIstopathology Alignment 프레임워크를 제안합니다. 세 모달리티 인코더가 공통 잠재공간으로 투영된 뒤, 각 양식 임베딩이 하나의 기하적 기준(삼중 임베딩의 공분간 부피)으로 함께 정합되도록 설계합니다. 이후 cross-modal gating으로 융합해 등급(Grade II/III/IV)과 전체 생존(Overall survival)을 동시에 학습합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 이질적인 입력 형식(WSI 조각, 유전자 그래프, 3D MRI)을 같은 잠재공간에 의미 있게 매핑하고, (2) 세 양식을 단일 학습 목표로 안정적으로 정렬하는 것입니다. GLORIA는 UNI2-h 기반 WSI 인코더, 유전자-유전자 그래프 기반 mRNA GCN, SwinUNETR의 SwinViT 인코더(3D MRI)로 각각 임베딩을 만든 뒤, Gramian contrastive loss로 삼중 임베딩이 이루는 병렬체의 부피(정렬 정도)를 InfoNCE 형태에 반영해 ‘3자 정렬’을 학습합니다. 마지막으로 sigmoid 기반 per-feature gating을 통해 차원별로 모달리티 혼합 비중을 학습하고, Cox 비례위험 손실과 분류 손실을 불확실성 가중으로 함께 최적화합니다.

- **Empirical Impact**: TCGA-GBM/LGG와 BraTS21을 교집합해 132명의 세 모달 매칭 코호트를 구성하고, 공통 삼중 테스트 세트에서 GLORIA는 WSI-mRNA 기반 이중(bimodal) 기준선 대비 모든 평가 지표에서 성능을 개선했다고 보고합니다. 특히 등급 분류에서 Grade II와 II/III 경계 구간의 체계적 과소 분류를 더 잘 바로잡았으며, WSI의 주의(attention/시각화)도 핵 밀도가 높은 영역에 더 집중되는 경향이 관찰됩니다. 반면 생존 예후 순위(C-Index) 개선은 상대적으로 미미해, MRI의 기여가 주로 분류(등급 결론) 쪽에 집중된 것으로 해석됩니다.



### Spectrum Aware Illumination Estimation Using Multispectral Imag (https://arxiv.org/abs/2606.14248)
Comments:
          Accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). DOI: https://doi.org/10.1109/TCSVT.2026.3701975

- **Prior Approaches**: 기존 RGB 기반 조명 추정은 흰 패치 존재(Max-RGB), 회색 가정(Gray-World), 엣지 회색 가정(Gray-Edge) 같은 통계를 전제로 해 장면이 가정에서 벗어나면 성능이 급락하는 한계가 있었다. MS/하이퍼스펙트럴로 확장된 방법들은 저랭크 분해로 조명과 반사율을 분리하거나, 딥러닝으로 SPD를 복원했지만 채널 간 스펙트럴 상관을 충분히 활용하지 못하거나 센서 도메인 차이로 사전학습 전이가 불안정했다.

- **Core Contribution**: 이 논문은 MS ISE(illumination spectrum estimation)를 위해 공간-스펙트럼 특징 추출 블록을 설계하고, 스펙트럴 어텐션(특히 illuminant prior를 반영하는 SABIP)으로 조명에 관련된 채널과 스펙트럴 상관을 더 잘 강조한다. 또한 스펙트럼 도메인 변환을 제안해 고차원(고채널)에서 학습한 조명 정보를 추가 학습 없이 저차원 센서 공간이나 XYZ 같은 색 공간으로 옮겨 쓸 수 있게 했다. 비교를 위한 실세계 MS 데이터셋 MILD도 함께 제공해 극단 조명까지 평가 가능하게 했다.

- **Technical Challenges**: ISE는 조명 SPD와 표면 반사율이 함께 관측에 얽히는 under-constrained 문제라 정확한 복원이 어렵고, MS에서는 채널 수·샘플링 간격·센서 민감도 차이로 같은 모델을 다른 센서에 그대로 적용할 때 성능이 흔들린다. 논문은 3D 합성곱으로 스펙트럴 디테일을 보존하면서, illuminant prior(IP) 기반 재가중과 다중 헤드 스펙트럴 self-attention(MSSAB)로 채널 간 의존성을 단계적으로 학습해 복원 안정성을 높였다. 더 나아가, 센서 민감도 함수(CSF)나 CIE-1931 CMF를 이용한 학습-비의존 스펙트럴 변환 행렬로 도메인 불일치를 처리한다.

- **Empirical Impact**: 분광복사계(spectroradiometer)로 측정한 GT를 갖는 MILD와 다양한 조명 조건에서 실험을 수행했으며, 기존 모델 대비 조명 스펙트럼 추정 정확도가 더 높다고 보고한다. 특히 극단적인 단색/고채도 조명처럼 RGB 가정이 깨지는 시나리오에서도 평가가 가능해 실전 적용성을 강화한다. 또한 고차원 센서에서 학습한 조명 SPD가 저차원 카메라 센서 공간으로 추가 학습 없이 매끄럽게 변환되어, 다중 센서 환경에서의 배포 비용을 낮추는 실용적 의미가 크다.



### Context-aware Modality-Topology Co-Alignment for Multimodal Attributed Graphs (https://arxiv.org/abs/2606.14172)
- **Prior Approaches**: 기존 멀티모달 어트리뷰트 그래프(MAG) 방법들은 주어진 그래프 맥락(토폴로지)이나 단일한 융합 표현을 가정하는 경우가 많았다. 그 결과 태스크에 따라 달라져야 하는 이웃 신뢰도와 이웃 요구사항(분류·링크 예측·검색·생성 등)이 동일하게 처리되어, 잡음 증폭이나 상충하는 엣지 전파가 발생하기 쉽다. 또한 모달리티 융합/정렬이 공통 임베딩으로의 압축을 유도하면서 모달리티별(텍스트/이미지) 세부 증거가 사라질 수 있다.

- **Core Contribution**: CoMAG는 MAG 백본을 “태스크 적응형 신뢰 가능한 컨텍스트 학습”과 “모달리티 보존형 홉-토큰 정렬”로 재구성해, 그래프 중심과 모달리티 중심 태스크를 한 번의 전방 패스로 동시에 만족시키는 것을 목표로 한다. 신뢰도 게이트를 통해 태스크에 맞는 컨텍스트 그래프를 구성하고, 모달리티별 다중 홉 궤적을 유지한 채 홉 위치까지 반영해 미세 정렬을 수행한다. 정렬 과정에서 공유 합의(shared consensus)와 모달리티 프라이빗 잔차(private residual)를 분리해 과도한 정보 소실을 줄인다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 모든 관측 엣지를 동일하게 전파하면 안 되는데, 어떤 엣지가 ‘신뢰할 만한’지 모달리티 의미와 태스크 관점에서 결정해야 한다는 점이다. CoMAG는 엣지 양끝의 멀티모달 의미 일치도를 근거로 엣지 신뢰도를 추정해 기존 토폴로지를 보정하고, 그래프에 없는 관계는 크로스모달 유사도 기반의 의미 보완 이웃으로 복원하되 태스크 게이트로 혼합 비중을 조절한다. 또 (2) 모달리티를 하나의 임베딩으로 뭉치게 하면 디테일이 사라지므로, 모달리티-별 홉 토큰을 만들고 거리 페널티를 둔 크로스모달 어텐션으로 정렬한 뒤 공유/프라이빗을 디커플링하는 설계로 모달리티 붕괴를 제어한다.

- **Empirical Impact**: OpenMAG 프로토콜의 9개 데이터셋에서 CoMAG는 특징만/그래프만/멀티모달/단일 통합 백본 계열의 다양한 기준선을 상대로 그래프 레벨 예측, 모달리티 매칭(정렬), 그래프 조건부 생성 전반에서 최상 성능을 보였다. 특히 그래프 예측의 구조·클래스 판별력을 높이면서도 검색/생성에 필요한 모달리티별 증거를 유지하는 방향이 관찰된다. 또한 안정적 전파, 과도 평활화(over-smoothing) 완화, 모달리티 콜랩스 제어를 위한 분석을 제공하며, 희소 엣지 선형 복잡도 수준의 효율을 유지한다고 주장한다.



### Naive Visual Memory is Not Enough: A Failure-Mode Study of GUI Agents (https://arxiv.org/abs/2606.14106)
Comments:
          9 pages, 5 figures, ICML 2026 WORKSHOP

- **Prior Approaches**: GUI 에이전트는 과거 궤적을 저장하고 비슷한 상태에서 재현해 의사결정을 돕는 ‘경험적 메모리’를 도입해 신뢰성을 개선해 왔다. 더 나아가 스크린샷을 통째로 저장·검색하는 시각 메모리도 제안됐지만, 어떤 실패를 줄이고 어떤 실패를 키우는지 체계적으로 분석되진 않았다.

- **Core Contribution**: 이 논문은 GUI 에이전트의 오류를 인지 실패, 시각 상태 오독, 숨은 동작 블라인드니스, 그라운딩 에러의 4가지로 분류하고, 이들이 perception-추론-action 파이프라인의 서로 다른 단계에 대응함을 보였다. 또한 전체 화면(풀 이미지) 기반 시각 메모리가 실패 분포를 ‘상태 단계 오류는 줄이지만 행동 단계 오류는 악화’시키는 경향을 드러내 핵심 쟁점을 정리한다. 이를 바탕으로 Action-Grounded Visual Memory(AGMem)를 제안해, 메모리를 행동과 직접 연결된 로컬 크롭으로 압축·검색하도록 한다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 어떤 픽셀을 저장해야 상태 해석에 도움이 되면서 행동 단계의 주의 분산을 막을지, (2) 검색된 예시가 현재 서브태스크와 어긋나면 부정적 전이가 생기는 문제를 줄이는 데 있다. AGMem은 성공/회복의 ‘행동 효과’가 드러나는 구역만 크롭한 action-relevant view를 저장하고, 서브태스크 기반으로 검색 공간을 좁혀 관련 없는 시각 단서를 재주입하지 않게 한다. 더해 오류가 감지되면 recovery-aware verification memory로 잘못된 상태의 복구 예시를 별도로 검색해 오류 전이를 차단한다.

- **Empirical Impact**: OSWorld와 WebForge, 그리고 AgentNetBench에서 AGMem은 풀 이미지 메모리 대비 실패 분포의 부작용을 완화하며 전체 성능을 끌어올린다. 특히 OSWorld에서 task success가 풀 이미지 메모리보다 33.3% 향상되었다고 보고하며, GPT-5.4-mini 기준으로도 엔드투엔드 정확도 개선(예: +6.8%p)과 각 실패 모드 동시 감소(전체 모드에서 개선)를 관찰했다. 결과적으로 ‘시각 메모리 여부’뿐 아니라 ‘행동에 근거한 시각 압축과 선택적 검색’이 GUI 에이전트 신뢰성 향상의 핵심임을 실증적으로 제시한다.



### FoleyGenEx: Unified Video-to-Audio Generation with Multi-Modal Control, Temporal Alignment, and Semantic Precision (https://arxiv.org/abs/2606.14049)
Comments:
          Accepted by INTERSPEECH 2026

- **Prior Approaches**: 기존 VTA(Video-to-Audio) 연구는 멀티모달 제어(텍스트/비디오/참조 음원)를 제공하더라도 프레임 단위 시간 정렬이 약하거나, 반대로 정렬 능력이 강해도 참조 오디오 기반 조건(AC-VTA, FE)이 부족한 경우가 많았다. 예를 들어 MultiFoley는 비디오 특징을 단순 업샘플링해 참조 오디오와 결합하지만 시간 동기화가 쉽게 흔들렸다. MMAudio는 Synchformer 등으로 동기화 성능이 뛰어나지만 참조 오디오 조건 분기가 없어 음색·운율·사건 타이밍까지 “정확히” 맞춰야 하는 작업에서 한계가 있었다.

- **Core Contribution**: FoleyGenEx는 MMDiT(Multi-modal Diffusion Transformer) 기반 동기화 강점을 유지하면서, 참조 오디오를 주입하는 조건 주입(conditional injection) 경로와 멀티모달 동적 마스킹을 함께 설계해 시간 정렬·다양한 조건·정밀 의미를 한 프레임워크에서 맞추려 한다. 또한 어드버브(부사) 기반 데이터 증강을 도입해 “빠르게/느리게”, “크게/작게”처럼 미세한 방식·강도를 텍스트로 제어할 수 있도록 학습 신호를 보강한다.

- **Technical Challenges**: 핵심 난관은 (1) 학습 시의 입력 구성과 추론 시의 입력 구성이 달라지면 동기화가 깨지는 문제, (2) 참조 오디오 조건이 들어올 때 음향 스타일(음색·운율)과 시간 정렬이 동시에 흔들리지 않게 만드는 문제, (3) 텍스트가 프레임별 정렬을 엄밀히 보장하지 않는 상황에서 의미를 유지하는 문제였다. FoleyGenEx는 오디오 잠재를 대규모로 마스킹해 훈련·추론 입력 불일치가 없도록 하고, 오디오·비주얼·동기화 스트림에 대해 서로 대응되는 마스킹을 적용해 shortcut bias를 줄였다. 여기에 어드버브 증강은 신호 처리(속도/거리/볼륨 변조)와 LLM 기반 캡션 재생성·패러프레이징을 결합해 미세 의미를 안정적으로 학습시키는 방식으로 해결했다.

- **Empirical Impact**: AudioCaps와 VGGSound, Greatest Hits 등의 벤치마크에서 FoleyGenEx는 기존 방법 대비 분포 일치, 의미 정합, 음질, 시간 정렬에서 전반적으로 경쟁력 또는 개선을 보이며 특히 MultiFoley보다 동기화·의미 제어 측면에서 우수했다. TC-VTA 같은 “비디오와 텍스트가 의미 충돌”하는 상황에서도 FoleyGenEx는 시각 동작 타이밍을 기준으로 오디오 스펙트럼이 적절히 생성되는 결과를 보여 경량 업샘플링 중심 접근의 문제를 보완함을 시사한다. 더불어 어드버브 데이터 증강(AA)을 넣으면 모델들이 세부 부사 단서에 더 민감해지고 정성 평가에서도 개선이 확인되어, 제어 가능한 VTA의 실사용 정밀도 향상에 의미가 있다.



### High-Fidelity Video Compression based on Invertible Neural Transform and Implicit Conditioning (https://arxiv.org/abs/2606.13957)
- **Prior Approaches**: 기존 학습 기반 비디오 압축은 분석-합성(transforms)을 비가역적으로 설계하는 경우가 많아, 복원 왜곡이 양자화 오차뿐 아니라 변환 근사 오차의 합으로 나타난다. 특히 고품질 구간에서는 양자화 오차가 작아지는 만큼 변환 자체가 만든 왜곡이 병목이 되며, 그 결과 단일 모델이 저비트부터 고충실까지 일관되게 성능을 유지하기 어렵다. 또한 기존 가역(인버터블) 접근은 영상에서 바로 압축 효율을 얻기 위해 추가적인 압축 친화 장치(예: 채널 스퀴즈, 조건부 코딩 등)에 의존하는 경향이 있었다.

- **Core Contribution**: InnVC는 양자화 이전의 메인 변환 경로를 완전히 가역(invertible)으로 유지해 변환 근사 오차 병목을 핵심적으로 완화하는 신경 비디오 코덱이다. 동시에 비디오 내용에 적응하는 암묵적 조건 필드(implicit conditioning field)를 도입해, 시공간 상관 구조와 세부 디테일을 서로 다른 구성 요소가 맡도록 분리(decoupling)한다. 추가로 예약 마스킹(scheduled masking)으로 잠재 채널의 중요도를 점진적으로 정렬해 엔트로피 코딩 친화성을 높인다.

- **Technical Challenges**: 가역 변환은 양자화 이전 정보 손실을 줄이지만, 차원(채널)으로 정보가 재배치되어 잠재 표현이 ‘압축하기 어려운 형태’가 될 수 있다. InnVC는 이를 위해 가역 백본의 조건 신호를 다단 멀티스케일 조절로 주입하고, 예약 마스킹을 통해 의미 있는 콘텐츠가 앞쪽 채널에 집중되도록 정규화한다. 또한 채널 순서형 채널 오토레그레시브(channel autoregressive) 엔트로피 모델로 채널 간 컨텍스트를 효과적으로 활용한다.

- **Empirical Impact**: UVG와 MCL-JCV 벤치마크에서 InnVC는 넓은 화질 범위 전반에 걸쳐 경쟁력을 보이며, 특히 고품질(high-quality) 구간에서 강한 이점을 보인다. UVG에서 x265 대비 PSNR 기준 BD-rate 21.66% 감소, MS-SSIM 기준 46.06% 감소를 달성했으며(논문 추가 요약치로는 평균 PSNR 기준 UVG 24.90%, MCL-JCV 22.38% 감소), 단일 아키텍처 스케일로 PSNR 20dB 이상 범위를 폭넓게 커버하는 것이 강조된다. 요약하면, 기존 신경 코덱이 놓치던 ‘고충실 구간’ 성능을 가역 변환 설계와 조건 분리, 채널 조직화로 실증했다.



### GMN4AD: Graph Matching Network for Alzheimer's Disease Diagnosis with Test-Time Domain Adaptation using Multi-centered Structure Magnetic Resonance Imaging (https://arxiv.org/abs/2606.13919)
- **Prior Approaches**: 기존 sMRI 기반 그래프 접근은 각 뇌 그래프를 독립적으로 다루는 경우가 많아, 양식(modality) 차이와 사이트 간 이질성(inter-site heterogeneity)으로 생기는 도메인 편향을 충분히 반영하지 못했습니다. 그 결과 진단 성능이 데이터 분포에 따라 흔들리거나 일반화가 제한되는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 서로 다른 이질적 뇌 그래프 사이의 관계를 학습하기 위해 Graph Matching Network for Alzheimer's Disease Diagnosis(GMN4AD)를 제안합니다. 또한 추론 시점(test-time)에서 도메인 적응을 수행하는 전략을 더해, 경도 인지 장애(MCI) 단계 조기 진단을 포함한 AD 진단 정밀도를 높이는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 사이트와 양식에서 생성된 그래프 간 대응관계를 안정적으로 학습하는 동시에, 실제 추론 환경에서 발생하는 도메인 이동(domain shift)을 사전에 알 수 없다는 점입니다. 논문은 graph matching으로 교차-그래프 상호작용을 포착하고, test-time domain adaptation에서는 대조 학습(contrastive learning)을 결합해 추론 중 분포 불일치를 완화하도록 설계합니다.

- **Empirical Impact**: GMN4AD는 3개의 공개 AD 데이터셋에서 기존 최첨단 방법 대비 우수한 진단 성능을 보였으며, 특히 도메인 변화에 대한 견고함이 관찰됩니다. 이는 sMRI 그래프 기반 AD 진단에서 그래프 매칭과 추론 시점 적응을 함께 고려하는 접근이 실용적 일반화 해법이 될 수 있음을 시사합니다.



### Gefen: Optimized Stochastic Optimizer (https://arxiv.org/abs/2606.13894)
- **Prior Approaches**: AdamW는 1·2차 모멘트 이동평균을 저장해 성능이 안정적이지만, 옵티마이저 상태만으로도 파라미터 메모리에 비례한 큰 버퍼가 추가된다. 기존 메모리 절감 연구로는 Adam-mini(수동 규칙 기반 공유), Adam8bit/Adam4bit(모멘트 양자화) 등이 있으나, Hessian 정렬의 이론적 근거가 약하거나 텐서 이름/아키텍처 정보 같은 수동 설정이 필요하며, 고정 블록 크기 같은 암묵적 하이퍼파라미터 부담도 남는다. 또한 양자화 방식이 플러그-앤-플레이로 항상 적용되기 어려운 제약이 있다.

- **Core Contribution**: Gefen은 AdamW의 성능을 유지하면서도 2차 모멘트 상태를 파라미터 블록 간에 자동 공유하고, 1차 모멘트를 학습된 코드북으로 양자화해 옵티마이저 메모리를 약 8배 줄이는 것을 목표로 한다. 핵심 아이디어는 큰 혼합 Hessian 항이 두 파라미터의 제곱 그라디언트 비율을 1에 가깝게 만든다는 이론에 기반해, Hessian 관련성이 큰 파라미터끼리 같은 2차 모멘트 추정을 쓰는 것이 자연스럽다는 점이다. Hessian을 직접 계산하지 않고도 이를 구현할 수 있게, 학습 초깃값의 제곱 그라디언트로 블록 구조를 자동 추론한다.

- **Technical Challenges**: 문제는 Hessian을 대규모에서 직접 계산하는 것이 불가능하다는 점인데, Gefen은 대신 첫 스텝의 제곱 그라디언트만으로 텐서 내 블록 분할 후보를 나눠 보고, 블록 내 이질성이 작아지는 “첫 뚜렷한 개선 지점”의 주기를 선택해 공유 블록을 만든다. 이어서 1차 모멘트 양자화를 위해 기존처럼 수동/고정 블록 크기와 경험적 양자화를 쓰지 않고, 앞서 얻은 블록 분할을 그대로 재사용하는 “정확한 히스토그램 기반 동적 계획법” 코드북 학습을 제안한다. 또한 Lloyd-Max 계열 양자화의 범위 수축(또는 초기화 민감도) 문제를 피하기 위해 코드북에 극단값을 강제 포함하는 등 수렴·일관성을 확보한다.

- **Empirical Impact**: 실험에서 Gefen은 비교한 AdamW 계열 메모리 절감 방법들 중 옵티마이저의 피크 메모리를 가장 크게 낮추면서도 AdamW 수준의 성능을 유지한다. 분산 학습에서는 FSDP에서 AdamW 대비 처리량(throughput)을 56% 개선했고, DDP에서는 AdamW가 마이크로배치 1조차 담지 못하는 조건에서 Gefen이 마이크로배치 2를 가능하게 하며 Adam-mini 대비 21% 처리량 향상을 보였다. 즉, 단순 대체(drop-in replacement)로도 큰 모델·더 큰 배치·더 큰 마이크로배치를 현실적으로 열어 훈련 효율을 끌어올릴 수 있다는 점에서 실용적 파급력이 크다.



### PhysVLA: Towards Physically-Grounded VLA for Embodied Robotic Manipulation (https://arxiv.org/abs/2606.13886)
Comments:
          9 pages, 5 figures, supplementary material included

- **Prior Approaches**: VLA(vision-language-action) 모델들은 영상과 언어 지시를 로봇 제어 행동으로 바로 매핑하지만, 주로 시연 데이터 적합에 의존해 강체 동역학이나 접촉 제약을 명시적으로 강제하지 않는다. 그래서 접촉이 많은 다단계 조작에서 물리적으로 불일치한 행동이 나올 수 있고, 이를 줄이기 위한 단순한 시간적 스무딩(예: EMA)은 장기 품질을 희생하며 실패를 늘릴 위험이 있다.

- **Core Contribution**: 본 논문은 학습 없이(inference-time, training-free) 임의의 고정 VLA 백본을 감싸 동작을 물리적으로 보정하는 플러그앤플레이 프레임워크 PhysVLA(Physics-VLA)를 제안한다. PhysVLA는 백본 가중치 접근·미세조정 없이도, 시뮬레이터/시스템 상태만 활용해 행동을 단계(approach, grasp, transport, place)별로 정합시키고 필요할 때만 동역학 잔차 기반 보정을 작동시킨다.

- **Technical Challenges**: 핵심 기술 난관은 ‘전체 에피소드에 균일하게 스무딩’하면 접촉 국면의 민감한 반응이 무뎌져 실패가 늘어난다는 점이다. PhysVLA는 (1) 조작 단계를 분해하는 위상(phase) 인지 유한상태기계로 기하 기반 소규모 보정을 적용하고, (2) Euler-Lagrange 잔차가 기준을 넘을 때만 선택적으로 가동되는 selective Euler-Lagrange gate로 미세한 동역학 불일치만 보정해 <1ms 수준의 지연으로 운영 가능하게 했다.

- **Empirical Impact**: LIBERO-Spatial의 Franka Panda 7-DoF 실험에서 PhysVLA는 백본 전반에 걸쳐 성공률을 최대 17%p, 안정성을 최대 19%p까지 끌어올리며 태스크별 성능 퇴행을 보이지 않았다. 또한 Robosuite Lift 크로스-시뮬레이터 스윕에서 궤적 저크(trajectory jerk) 강건성이 최대 10배 개선됐고, 실제 Agilex Piper 로봇 pick-and-place에서도 재학습 없이 성공률이 최대 50%p 향상되어 물리 인지 보정 모듈의 범용성과 전이성을 입증했다.



### Multi-Agent Embodied Autonomous Driving: From V2X Information Exchange to Shared World Models (https://arxiv.org/abs/2606.13840)
- **Prior Approaches**: 기존 자율주행 연구는 차량 단독 지능에 무게가 실렸지만, 최근에는 인프라·다른 차량과의 협력으로 다중 에이전트 구체적 환경(embodied)에서 불확실성을 다루는 방향으로 이동하고 있다. 선행 접근은 V2X 통신, 협력적 지각, 에이전트 간 인지, 협동 계획, 종단 간 협동 주행, 폐루프 검증을 위한 시뮬레이션/데이터 엔진으로 넓게 분류된다. 다만 평가가 시뮬레이션과 선별된 벤치마크, 오프라인 프로토콜에 치우쳐 있고, 실차·개방 도로에서의 실시간 안전 보장이 충분히 검증되지 않았다.

- **Core Contribution**: 이 논문은 전환 과정을 Shared World Models(공유 세계 모델, SWMs) 관점에서 구조화하며, 서로 교환된 관측이 어떻게 정렬된 상태(aligned state)와 의도 인지 기반 상호작용, 최종 행동 조정으로 이어지는지 정리한다. 즉, 각 에이전트가 유지하는 예측적 교차 에이전트 표현이 무엇을 의미하고, 어떤 구성 요소들이 정합성 문제를 해결해야 하는지 연구 의제를 제시한다. 또한 우선순위로 공유 상태의 검증 가능 유지, 의도·계획 정렬의 강건성, 통신 지연과 배치 제약 하의 안전한 협동 행동을 도출한다.

- **Technical Challenges**: 핵심 난제는 교환 관측을 공통의 정렬 상태로 유지하는 과정에서 발생하는 불일치와, 서로 다른 에이전트의 의도·계획이 엇갈릴 때의 정렬 실패를 막는 것이다. 저자들은 이를 위해 통신·지연·노이즈 환경에서 SWMs가 예측적 표현을 안정적으로 보존하도록 하는 설계와, 계획·행동 단계에서 정렬이 깨졌을 때의 강건한 대응을 강조한다. 더 나아가 개방 교통에서 실시간 안전 보장을 수학적으로(또는 검증 절차로) 확보하는 것이 아직 공백으로 남아, 재현 가능한 검증 체계가 요구된다고 지적한다.

- **Empirical Impact**: 240편이 넘는 문헌을 포함해 380편 이상을 폭넓게 망라하며, 현재까지의 검증이 주로 시뮬레이션·커리큘럼화된 벤치마크·오프라인 설정에 집중되어 있음을 명확히 보여준다. 이 결과는 MAEAD(다중 에이전트 물리적 환경 기반 자율주행)의 향후 연구가 ‘성능’뿐 아니라 ‘공유 상태의 검증 가능성’과 ‘실시간 안전 보장’ 중심으로 재편돼야 함을 독자에게 방향성으로 제공한다. 특히 foundation model 기반 조정이 등장했더라도 개방 도로에서 안전을 확인할 수 있는 근거가 부족하다는 점을 전면에 내세워, 실시간 검증 연구의 중요성을 부각한다.



### $μ_0$: A Scalable 3D Interaction-Trace World Mod (https://arxiv.org/abs/2606.13769)
- **Prior Approaches**: 기존 월드 모델은 주로 픽셀 생성으로 넓은 시각 사전학습을 얻거나, 혹은 행동(액션) 레이블을 직접 예측해 제어에 연결했습니다. 하지만 픽셀 기반은 배경·외관 재구성에 용량이 소모되고 조작에 필요한 접촉/기하 구조를 놓치기 쉬우며, 행동 직접 예측은 임봇(로봇 체형)별 레이블이 희소해 확장성이 제한됩니다.

- **Core Contribution**: 이 논문은 행동 레이블 없이도 전이 가능한 ‘3D 트레이스(3D 궤적) 세계 모델’ μ0를 제안합니다. μ0는 객체/도구/손/접촉 영역 같은 의미 있는 상호작용 지점들의 부드러운 3D 궤적을 예측해, 임봇과 무관한 모션 인터페이스를 형성합니다.

- **Technical Challenges**: 핵심은 대규모 학습을 위한 3D 감독을 어떻게 자동으로 만들고, 불확실한 미래와 가림(occlusion) 속에서도 실행 가능한 궤적을 어떻게 생성하느냐입니다. 이를 위해 TraceExtract로 DINOv2 기반 의미 키포인트를 선택하고 전역 정렬된 3D 트레이스를 만들며 이벤트 단위 계층 언어 캡션을 붙여 학습데이터를 확장했고, μ0는 VLM 백본과 순열-대칭 Trace Expert를 결합해 B-스플라인 제어점에 대한 조건부 디노이징(흐름 매칭)으로 미래 트레이스를 생성합니다.

- **Empirical Impact**: 실험에서 μ0는 2D/3D 트레이스 예측의 ADE·FDE·DTW 전반에서 기존 트레이스 모델과 tokenized VLM 계열을 능가합니다. 또한 RoboCasa365 시뮬레이션과 UR3 실로봇 조작에서, 동작 레이블 없이 비디오만 사전학습한 뒤 행동 전문가를 붙인 방식이 액션 감독 VLA 모델과 경쟁하거나 더 높은 성공률을 보이며, 3D 트레이스 표현이 크로스-임봇 조작에 실용적으로 전이됨을 입증했습니다.



### Orchestra-o1: Omnimodal Agent Orchestration (https://arxiv.org/abs/2606.13707)
- **Prior Approaches**: 기존 에이전트 오케스트레이션은 주로 텍스트 중심이거나 시각-언어처럼 제한된 조합에 맞춰져 있어, 텍스트·이미지·오디오·비디오가 함께 얽힌 오므니모달 환경에서의 일반화가 어렵다. 또한 모듈을 나눠도 실행 흐름이 선형적이거나 휴리스틱에 의존해 복잡한 의존성/병렬성을 효율적으로 다루지 못한다. 네이티브 오므니모달 에이전트는 한 모델이 인식·추론·도구사용까지 동시에 담당하려 하지만, 장기 지평 추론과 도구/크로스모달 정교성에서 한계가 나타난다.

- **Core Contribution**: Orchestra-o1은 오므니모달 에이전트를 위한 오케스트레이션 프레임워크로, 입력 양식(modality)을 고려한 작업 분해와 하위 에이전트의 온라인 전문화, 병렬 서브태스크 실행을 한 구조 안에 묶는다. 메인 에이전트가 고수준 결정(위임·완료)을 하고, 인식/행동은 전용 서브에이전트와 통합 도구 생태계가 담당하도록 ‘역할 분리’를 명확히 한다. 아울러 Orchestra-o1의 메인 에이전트 학습을 위해 DA-GRPO(Decision-aligned group relative policy optimization)라는 오프라인 에이전트형 강화학습 레시피를 제안한다.

- **Technical Challenges**: 핵심 난제는(1) 어떤 입력과 도구가 해당 단계에 필요한지 모달리티 인지형으로 결정하고, (2) 결과 간 의존성을 그래프로 표현해 준비된 작업만 병렬로 스케줄링하는 것이다. 논문은 메인 에이전트가 서브태스크별 요구 벡터(텍스트/이미지/오디오/비디오/코드 등)와 도구 요구를 예측해 모델·도구를 매칭하며, 생성된 의존성 그래프를 기반으로 ready 집합에서 배치를 뽑아 병렬 실행한다. 또한 증거(evidence)를 구조화된 메모리에 압축 저장하며, 충분성 점수가 임계치를 넘을 때 종료하도록 하여 컨텍스트 예산 문제도 함께 다룬다. DA-GRPO는 단계별 오케스트레이션 결정(위임·서브에이전트 선택·도구 사용·생성)을 기준 궤적과 정렬하도록 다차원 루브릭 보상으로 오케스트레이션 의사결정을 학습시킨다.

- **Empirical Impact**: OmniGAIA 벤치마크에서 강한 프로프라이어터리 메인 에이전트와 결합했을 때 Orchestra-o1은 기존 2등 대비 10.3% 정확도 향상을 달성한다. 또한 DA-GRPO로 훈련한 Orchestra-o1-8B는 오픈소스 오므니모달 에이전트들에 대해 최신 수준의 성능을 보이며, OmniGAIA에서 최고 정확도를 20.8%에서 30.0%로 끌어올린다. 구조가 병렬화 가능하고 추론/비용 효율 측면에서도 이점이 있어, 오므니모달 에이전트 스웜 설계에 대한 실용적 기준선을 제시했다.



### C-MambaPose: A Physics-Informed Complex Mamba Framework for Cross-Environment WiFi Human Pose Estimation (https://arxiv.org/abs/2606.13700)
- **Prior Approaches**: WiFi 기반 3D 인체자세추정은 비침습·프라이버시 보호·저조도/가림에도 강점이 있지만, 기존 방식은 CSI의 복소(phase 포함) 물리 정보를 충분히 활용하지 못하는 경우가 많다. 또한 한 환경에서 학습한 모델이 방 레이아웃·가구·다중경로 특성이 바뀌는 교차환경에서 도메인 시프트를 크게 겪고, 큰 CNN/Transformer 백본은 배경 다중경로 잡음에 쉽게 과적합되기 쉽다.

- **Core Contribution**: 이 논문은 복소값(complex) 물리를 보존하는 물리기반 단계(위상 정제·위상 보존 표현)와, 구조적 제약(관절 위상/뼈 토폴로지)을 결합한 C-MambaPose를 제안한다. 특히 복소 Mamba(선형 시간대의 장기 의존)와 GraFormer류 GCN 디코더(해부학적 뼈 일관성)를 하이브리드로 묶어 교차환경 일반화와 파라미터 효율을 동시에 노린다.

- **Technical Challenges**: 핵심 난제는 (1) SFO/CFO/PLL 위상 잡음으로 심하게 오염된 CSI 위상에서 ‘신체 운동과 물리적으로 결합된’ 위상 관계를 복원·보존하는 것과, (2) 이를 교차환경에서 안정적으로 학습할 수 있게 저복잡도 시퀀스 모델에 적절히 주입하는 것이다. 저자들은 subcarrier 축 위상 언와인딩 후 SFO/CFO의 선형 오차를 최소제곱으로 제거해 clean phase를 만들고, 위상은 단위원 복소 표현으로 유지하며 amplitude는 정규화·라테 fusion으로 결합한다. 이어서 Complex Dynamic Selective Kernel로 수용영역을 동적으로 조절하고, Complex Mamba 블록 내부에 shared-mask 기반 Complex Dropout으로 위상 일관성을 깨지 않도록 과적합을 억제한다.

- **Empirical Impact**: MM-Fi 데이터셋에서 C-MambaPose는 전 설정에서 경쟁적이거나 우수한 성능을 보이며, 특히 어려운 교차환경 분할(Setting 3)에서 MPJPE 298.5mm를 달성했다. 파라미터는 3.78M으로 GraphPose-Fi 대비 83.1%, MetaFi++ 대비 85.7%를 줄였고, DT-Pose 크기와 비슷한 수준(약 18% 더 작음)에서도 프리트레인 없이 더 높은 성능을 보여 WiFi 기반 자세추정의 경량·일반화 방향에 의미 있는 실증을 제공한다.



New uploads on arXiv(cs.AI)

### Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows (https://arxiv.org/abs/2606.14672)
- **Prior Approaches**: 기존 에이전트 워크플로는 병렬로 여러 가지 하위 작업을 수행한 뒤 마지막에 합치는 형태(DAG 후 합성)로 많이 설계되지만, LLM은 순차 텍스트 접점에 맞춰 상태를 선형화해 전달하는 경우가 대부분이다. 그래서 합성 단계에서 각 브랜치 결과를 텍스트로 이어 붙여 프리필을 다시 수행하거나(중복 계산) 요약 기반 전달로 세부 정보가 손실되는 한계가 있었다. 또한 KV 캐시 재사용 연구는 주로 RAG에서 독립 인코딩된 증거 조각을 묶어 쓰는 문제에 초점이 있어, 에이전트 브랜치의 로컬 맥락에서 생성된 캐시를 다대일로 읽고 판단·통합해야 하는 합성 문제와는 요구 역량이 다르다.

- **Core Contribution**: 이 논문은 병렬 에이전트 브랜치가 생성한 KV 캐시를 합성기가 직접 소비하도록 하는 플러그앤플레이 프레임워크 Parallel-Synthesis를 제안한다. 핵심은 텍스트를 다시 이어 붙여 프리필하지 않고, 합성기 쪽에서 “병렬 캐시 인터페이스”를 해석·생성 가능하게 적응시키는 것이다. 이를 위해 브랜치 캐시를 보정하는 cache mapper와 합성기 모델을 해당 인터페이스에 맞게 학습시키는 synthesizer LoRA 어댑터를 함께 둔다.

- **Technical Challenges**: 가장 큰 기술적 난점은 병렬 작업에서 온 캐시들이 단일 연속 prefix에서 이어진 것이 아니라 서로 다른 로컬 컨텍스트로 생성되었다는 점이다. 논문은 이를 해결하기 위해 (1) 공유된 분기 지점 이후의 위치에 맞춰 RoPE 기반으로 캐시의 위치 정렬을 수행하는 positional re-encoding, (2) 브랜치 길이와 개수 같은 메타 정보를 바탕으로 키·값에 대한 선형 변환을 학습하는 cache mapping, (3) 캐시 기반 합성이 텍스트 직렬 합성과 유사한 추론을 하도록 적응시키는 synthesizer LoRA를 단계적으로 결합한다. 또한 학습 단계에서는 병렬 캐시 문맥에 대한 적응 데이터와, 텍스트 직렬 합성 경로에서 얻은 추론 궤적을 증류해(Reasoning distillation) 캐시 기반에서도 비교·판단·통합 능력을 강화한다.

- **Empirical Impact**: 수학, 과학 QA, 코드 생성, GAIA, 멀티 에이전트 데이터베이스 진단 등 9개 다운스트림 데이터셋에서 Parallel-Synthesis는 7개에서 텍스트 기반 합성과 동등하거나 더 높은 성능을 보였고 2개에서도 근접한 격차를 유지했다. 효율 면에서는 재프리필을 피함으로써 time-to-first-token(TTFT)이 2.5배~11배 줄어드는 개선을 보였다. 특히 추론이 무거운 작업에서도 이득이 관찰되어, 병렬 합성을 위한 “더 네이티브한(원형 구조에 가까운) 인터페이스”가 단순 속도 최적화가 아니라 품질에도 기여할 수 있음을 시사한다.



### Abstracting Cross-Domain Action Sequences into Interpretable Workflows (https://arxiv.org/abs/2606.14654)
Comments:
          preprint; 9 pages, 5 figures

- **Prior Approaches**: 기존 연구는 타임스탬프 UI 로그를 빈발 항목 집합 마이닝이나 순차 패턴 마이닝으로 요약해 왔지만, 토큰에 의미를 접지하기 어려워 잡음과 스푸리어스 상관에 취약했다. 딥러닝 기반 순차 모델도 도메인·태스크별 학습과 라벨링이 필요해 새로운 서비스나 행동 변화에 대응하기 비용이 컸다. 또 텍스트 기반 의도 추정은 목적이 이미 언어에 드러나는 경우가 많아, 저수준 이벤트만 있는 로그 문제와는 난이도가 다르다.

- **Core Contribution**: 이 논문은 WorkflowView라는 LLM 기반 계층적 추상화 프레임워크를 제안한다. 저수준 행동 시퀀스를 먼저 자연어 설명으로 변환한 뒤, 이를 바탕으로 고수준 활동과(필요 시) 범주까지 추론해 해석 가능하고 실행 가능한 인사이트로 연결한다. 핵심은 잡음이 섞인 로그를 단계적으로 “점진적 노이즈 제거” 형태로 다루면서, 제로샷/퓨샷만으로도 도메인 전이를 노린다는 점이다.

- **Technical Challenges**: 가장 큰 기술 과제는(1) 로그가 너무 세분화·잡음이 많아 의도가 직접 드러나지 않는다는 점과(2) 자연어 모델이 UI 이벤트 문맥을 제대로 추론하도록 프롬프트를 계층적으로 설계해야 한다는 점이다. WorkflowView는 Layer 1에서 관측된 행동을 자세한 자연어로 풀어쓴 뒤, Layer 2(및 Layer 3)에서 시간적 패턴과 중요도가 다른 행동을 점진적으로 정리해 고수준 활동을 추출한다. 또한 범주가 사전에 정해져 있지 않은 경우에는 별도 활동 분류(라벨 생성) 단계를 결합해 데이터에 맞춘 분류 체계를 만든다.

- **Empirical Impact**: 실험은 브라우저 로그 태스크 설명 재구성(코사인 유사도 평균 0.91), MOOC 수강생 이탈 예측(가중 F1 0.90, 퓨샷 예시 5개 수준), Microsoft Word에서 AI 도구 통합 맥락 분석(익명·집계 인사이트) 등 3개 서로 다른 도메인으로 검증됐다. 결과적으로 제로샷/퓨샷만으로도 기존 학습 기반 접근과 견줄 만한 성능과 높은 의미적 정합성을 보였다는 점이 의미 있다. 더 나아가 저수준 텔레메트리로부터 프라이버시를 지키는 집계형 관찰을 만들 수 있어, 실제 제품 개선(예: AI 출력 수용 이후 편집·서식 전환 패턴 반영)에 바로 연결될 잠재력이 제시된다.



### A Temporal Planning Framework for Disruption Aware Dynamic Route Optimization in Heterogeneous Railway Systems (https://arxiv.org/abs/2606.14582)
- **Prior Approaches**: 기존 연구는 시간표(timetable) 생성에 집중하는 경우가 많아, 전환기(turnout) 같은 저수준 실행 동작까지 포함한 “실행 가능한” 복구 계획을 자동으로 만들기 어렵다. 또한 다중 게이지(multi-gauge) 환경에서의 게이지 호환성과 서로 다른 속도·정차 패턴에 따른 제약을 충분히 통합하지 못해 이질적 철도망에서의 확장성이 떨어진다. 일부는 MaxSAT나 MILP로 충돌 없는 스케줄을 만들지만, 대체로 동작 순서와 안전 보장 수준이 낮게 남는 한계가 있다.

- **Core Contribution**: 논문은 이질적 다중 게이지 철도에서 발생하는 혼란을 “동적 경로 최적화 + 실행 가능한 복구”로 함께 다루는 Temporal Planning 기반 프레임워크 DART를 제안한다. PDDL 2.1로 게이지 호환 제약과 전환기 조작, 그리고 블록드 트랙·블록드 트레인·슬로다운·엔진 고장 같은 혼란 시나리오를 직접 모델링해, 충돌 없는 시간 스탬프 기반 작전 계획을 생성한다. 특히 시간표만이 아니라 실제로 수행 가능한 동작 시퀀스를 함께 산출하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 여러 열차가 공유 자원을 두고 동시에 움직이면서도 안전/충돌 제약을 만족해야 하고, (2) 게이지 호환성 때문에 가능한 경로가 급격히 줄어들며, (3) 혼란이 끼면 기존 스케줄에서의 일정을 다시 맞춰야 한다는 점이다. 논문은 PDDL 2.1의 durative action과 시간 조건(at start/over all/at end), 수치 플루언트로 시간·상태 변화를 정밀하게 표현해, 전환기 조작과 구간 통과 시간을 제약에 맞게 재계산하도록 만든다. 또한 생성된 계획이 실제 순서 논리와 실행 가능성을 만족하는지 VAL로 검증해 사람 의존도를 줄인다.

- **Empirical Impact**: 평가를 위해 최대 1,000개 트랙 포인트와 120개 열차 규모까지 확장된 200개 벤치마크(정상 100/혼란 100)를 구성하고, POPF와 OPTIC 같은 최신 시간 계획자 2종을 사용해 성능을 점검했다. 실험 결과 DART는 이질적 철도망에서 게이지 제약과 혼란 시나리오를 처리하면서 충돌 없는 시간 스탬프 계획을 효과적으로 생성하며, 혼란 강도와 네트워크 밀도가 커져도 계획 생성이 예측 가능하게 스케일링됨을 보였다. 저수준 실행 동작까지 제공하는 “검증 가능한 복구 계획” 접근은 철도 운영의 안전성과 자동화 수준을 높이는 데 의미가 있다.



### VISTA: View-Consistent Self-Verified Training for GUI Grounding (https://arxiv.org/abs/2606.14579)
- **Prior Approaches**: GUI Grounding은 자연어 지시에 맞는 UI 클릭 좌표를 추정하며, 작은 오차가 잘못된 요소를 활성화해 이후 작업을 망칠 수 있다. 이에 GRPO와 규칙 기반 point-in-box 보상으로 RL 성능을 끌어올리는 시도가 있었지만, 한 스크린샷 뷰에서 반복 롤아웃을 뽑으면 그룹이 전부 실패(올-페일) 또는 전부 성공(올-서커)으로 쏠려 상대적 advantage가 사라지는 보상 퇴화가 발생한다. 또한 좌표는 입력 기하에 민감해, 뷰 변화만으로도 포맷/좌표 생성이 불안정해질 수 있다.

- **Core Contribution**: VISTA는 GRPO의 핵심 설계를 “어떻게 보상하느냐”뿐 아니라 “비교 그룹을 어떻게 구성하느냐”로 확장해, 같은 GUI 인스턴스의 목표 보존(target-preserving) 다중 뷰로 GRPO 그룹을 만든다. 이렇게 하면 의미(명령·대상)는 유지되지만 기하(좌표 프레임)는 달라져, 어려운 경우엔 성공 롤아웃이 등장할 확률이 커지고 쉬운 경우엔 불안정성이 드러나 유효한 그룹 분산이 복원된다. 여기에 self-verified cross-view anchor(자가 검증 오라클 앵커)를 더해, 모델이 최대 보상 롤아웃을 이미 만든 경우에만 오라클 좌표를 안정화 신호로 사용한다.

- **Technical Challenges**: 다중 뷰를 쓰면 탐색·다양성은 늘지만, 짧은 좌표 문자열 생성 특성 때문에 좌표 포맷/부호화가 흔들려 보상 신호의 안정성이 떨어질 수 있다. VISTA는 목표를 포함하는 제한적 crop으로 정확한 박스 리매핑(좌표 재투영)을 보장해 뷰 일관성을 확보하고, 오라클은 그룹 통계의 기준선(baseline)에서 제외해 암묵적 지도학습이 전체 업데이트를 지배하지 않게 한다. 또한 최대 보상 롤아웃을 만든 “자기 검증된” 그룹에만 앵커를 활성화해, 초기의 올-제로 그룹에서 advantage 폭주로 학습이 왜곡되는 것을 막는다.

- **Empirical Impact**: 다수의 GUI-grounding 벤치마크와 Qwen 계열 백본에서 VISTA는 ScreenSpot-Pro뿐 아니라 전반 정확도를 꾸준히 개선했다. 예를 들어 Qwen3-VL의 4B/8B/30B-A3B가 ScreenSpot-Pro에서 55.5/52.7/53.7에서 63.4/65.8/67.0으로 상승했고, 다른 모델(예: Qwen3.5 초기화)으로도 전이 개선이 확인됐다. 강건성 분석에서도 worst-view 정확도는 증가하고 prediction flip rate는 감소해, 동일 쿼리에 대한 좌표 예측이 뷰 변화에 더 덜 흔들리는 것으로 나타났다.



### StreamMemBench: Streaming Evaluation of Agent Memory for Future-Oriented Assistanc (https://arxiv.org/abs/2606.14571)
- **Prior Approaches**: 기존 개인 에이전트 메모리 벤치마크는 대화 회상이나 과제 성능 향상 같은 요소를 각각 따로 평가하는 경향이 강하다. 그 결과, 스트리밍 관찰(증거)에서 이후 유사 과제로 이어지는 ‘흐름(trajectory)’이 실제로 재사용되는지 검증이 부족했다. 또한 관찰한 증거를 즉시 쓰는 능력과, 피드백·사용자 상호작용 경험을 다음 과제로 옮기는 능력이 동시에 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 EgoLife의 에고센서 스트림에서 증거 앵커(evidence anchor)를 기준으로 두 단계 과제를 구성한 스트리밍 벤치마크 StreamMemBench를 제안한다. 1단계는 증거를 제대로 활용하는지, 2단계는 1단계에서 얻은 피드백과 상호작용 경험이 follow-up에 재사용되는지를 평가한다. 이를 통해 ‘관찰→즉시 도움→다음 유사 작업’의 연속성을 벤치마크로 측정한다.

- **Technical Challenges**: 핵심 기술 과제는 스트리밍 입력에서 특정 증거를 끌어와 현재 요청에 반영하고, 그 다음 과제로 이어지는 재사용 신호(피드백·상호작용)를 안정적으로 보존·전달하는 데 있다. 저자들은 이 과정을 증거 회상, 초기 증거 사용, 피드백 반영, follow-up 재사용의 네 가지 지표로 분해해 진단 가능하게 만들었다. 또한 서로 다른 백본을 사용한 실험에서 여러 메모리 시스템의 실패 양상을 일관되게 관찰하도록 설계했다.

- **Empirical Impact**: 8개 메모리 시스템을 두 백본에서 평가한 결과, 관찰 증거를 실제로 활용하지 못하거나 로컬에서 피드백을 반영했더라도 후속 행동으로 이어지지 않는 경우가 많았다. 즉, 메모리에 저장은 하더라도 미래 지향적 보조로 연결되는 메커니즘이 약함을 보여준다. StreamMemBench는 개인 에이전트 메모리 연구가 ‘저장’뿐 아니라 ‘다음 과제로의 재사용’까지 검증하도록 방향을 제시하는 데 의미가 있다.



### Every Eval Ever: A Unifying Schema and Community Repository for AI Evaluation Results (https://arxiv.org/abs/2606.14516)
- **Prior Approaches**: 기존 AI 평가는 다양한 리더보드, 논문, 블로그, 평가 하네스 로그에 흩어진 결과를 주로 표 형태의 단일 점수로 요약해 왔습니다. 또한 lm-eval-harness, HELM, Inspect AI 같은 평가 프레임워크가 서로 호환되지 않는 형식과 메타데이터를 사용해, 겉보기로 같은 평가라도 점수가 달라 비교가 흔들렸습니다.

- **Core Contribution**: Every Eval Ever(EEE)는 평가 결과를 단일 JSON 스키마로 표준화해 “점수”가 아니라 “해석 가능한 맥락”까지 함께 저장하도록 설계했습니다. EEE는 원천이 리더보드든 논문이든 관계없이 스키마에 흡수하고, 가능할 때는 인스턴스 단위 출력까지 선택적으로 포함해 재분석의 기반을 제공합니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 출처의 결과 포맷과 메타데이터를 동일한 의미 단위로 매핑하는 것이었습니다. 이를 위해 community-governed 메타데이터 스키마와 인스턴스 레벨 동반 스키마를 만들고, HELM·lm-eval-harness·Inspect AI 등 주요 하네스/포맷용 자동 컨버터를 제공하며, Pydantic 기반 검증 파이프라인으로 제출 시 스키마 준수와 필드 일관성을 강제합니다.

- **Empirical Impact**: EEE 저장소는 Hugging Face에서 커뮤니티 크라우드소싱으로 확장되어 22,235개 모델, 2,273개 벤치마크, 31개 평가 포맷을 포괄합니다. 이를 바탕으로 평가 하네스에 따른 재현성 격차, 에이전트 평가의 비용-정확도 상충, “perplexity”처럼 라벨만 같은 지표의 구현 의존성 등을 메타 분석 수준에서 드러내며, 분야의 비교 가능성과 비용 효율(재실행 절감)을 동시에 높였다는 점에서 의미가 큽니다.



### Dense Coordinate-List Fine-Tuning Induces a Controllable Interference Surface in Vision-Language Models (https://arxiv.org/abs/2606.14507)
- **Prior Approaches**: 기존 비전-언어 모델의 시각적 그라운딩(fine-tuning)은 주로 위치 정확도 같은 단일 성능 지표에 집중해 왔다. 하지만 조밀한 좌표 리스트는 생성 결과가 “클래스-숫자 좌표 레코드의 연속”과 “목록 종료(terminate)”까지 포함하는 구조화된 출력이라, 성능 외에 반복·종료 방식이 함께 변할 수 있다. 선행 연구는 종종 적응이 전반적 능력 저하나 포맷 드리프트를 일으킨다고 넓게 설명했지만, 반복 간섭이 출력 구조의 어디에 국한되는지는 명확히 분해하지 못했다.

- **Core Contribution**: 이 논문은 조밀한 좌표 리스트 생성(특히 bbox 좌표 레코드)을 “세밀한 생성 표면(generation surface)”과 “제어 표면(control surface)”으로 보고, 목표 F1과 반복·종료 같은 구조적 행동을 같은 평가 틀에서 측정·분해한다. 핵심적으로, 고용량 adapter는 타깃 그라운딩을 올리는 동시에 bbox 좌표 object-list에서 ‘반복 꼬리(repeated-tail pressure)’를 유발하지만, object-level repeat-stop 같은 제어로 ‘정확 중복 레코드’만 제거하면서 F1 손실을 거의 없앨 수 있음을 보인다. 즉 간섭은 확산된 능력 손실이 아니라, bbox 좌표 리스트라는 특정 구조축에 묶인 국소적·분해 가능한 현상으로 정리된다.

- **Technical Challenges**: 문제는 fine-tuning이 “어디서” 그리고 “어떤 단위(토큰 vs 레코드)”로 반복/종료 행동을 바꾸는지 추적하는 것이다. 저자들은 (1) 밀도-정밀도 축(프롬프트 예산으로 객체 수를 조절, repetition penalty 포함)과 (2) 구조-무결성 축(object-level repeat-stop으로 정확 중복 레코드가 다시 나올 때 첫 prefix에서 종료)을 직교 축으로 설계해 표면을 지도처럼 탐색했다. 또한 비슷한 구조를 가진 non-bbox JSON과 spatial/count JSON을 함께 비교하고, adapter 용량/모듈(q/k/v/o vs q/v) 스윕과 모델 패밀리 교차 검증으로 간섭이 bbox 좌표 리스트에 구조적으로 결박됨을 확인했다.

- **Empirical Impact**: Gemma 4 12B에서 고용량 q/k/v/o LoRA는 class-aware F1@0.3을 0.007에서 0.448로 크게 올리면서도 duplicate rate 0.080, max repeat 23 같은 반복 꼬리를 동반한다. 그러나 같은 밀도 설정에서 object-level repeat-stop을 적용하면 duplicate rate 0.000, max repeat 1로 “깨끗한 종료점”을 만들면서 F1@0.3를 0.490(거의 유지)까지 끌어올리고 F1@0.5도 소폭 개선(0.385)한다. 이 메커니즘은 Qwen3-VL-8B에서도 재현되고, COCO 2017 공공 데이터에서도 dense-bbox 획득 성능과 반복 압력이 유의미하게 함께 이동하며 repeat-stop으로 제거되어(반복 꼬리 제거와 F1 유지/소폭 상승) 범용적으로 의미 있는 ‘구조-제어 지침’을 제시한다.



### From Chatbot to Digital Colleague: The Paradigm Shift Toward Persistent Autonomous AI (https://arxiv.org/abs/2606.14502)
Comments:
          The paper is available on the project website: this https URL

- **Prior Approaches**: 기존 LLM 흐름은 챗봇처럼 ‘빠른 응답’에 초점이 맞춰져, 다음 토큰 예측 기반 생성이 언어 유창성과 상호작용을 만들었지만 깊은 검증·장기 일관성·논리적 계획은 취약했습니다. 한편 에이전트는 도구 호출로 작업을 확장했지만, 잘못된 호출 형식·관측 누락·중간 오류 회복 실패 등으로 궤적이 쉽게 무너지는 취약성이 컸습니다. 데이터와 평가는 대화 정답성이나 정적 벤치마크 중심이라 ‘작업 완료(closure)’까지 추적·감사하기 어려웠습니다.

- **Core Contribution**: 이 논문은 챗봇을 ‘디지털 콜리그(사내 동료)’로 전환하는 과정을 두 축(인지 코어: Chatbot→Thinking LLM, 도구 실행: Agent→OpenClaw)으로 구조화합니다. 핵심 메커니즘으로 Workspace + Skill을 제시하며, 상태가 유지되는 워크스페이스와 재사용 가능한 절차(스킬)가 모여야 에피소드형 응답과 단발성 도구 호출을 지속적 업무로 바꿀 수 있다고 주장합니다. 또한 학습 데이터 단위와 평가 관점을 instruction-response에서 state–action–observation 궤적 및 작업 완료 중심으로 이동시킵니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 자율 실행 중 추론이 근거 없이 흘러 환각이 생기거나, (2) 장기 실행에서 도구 체인 오류가 누적돼 계획이 쉽게 깨지며, (3) 메모리·상태가 컨텍스트 창에 의존해 불안정해지고, (4) 실행 가능한 행동의 안전·거버넌스가 복잡해진다는 점입니다. 논문은 이를 Workspace로 증거·상태·로그를 고정하고, Skill로 계획-검증-오류복구-유효성 확인 같은 운영 절차를 모듈화하며, 검증 루프와 감사 가능한 평가 환경으로 자율성의 실패 모드를 줄이려 합니다. 인지 코어 측에서는 Thinking LLM의 Long Chain-of-Thought, 추론 중 탐색·반성·자기수정, 추론 시점 컴퓨팅 투자와 강화학습을 통해 단순 생성의 상한을 넘고자 합니다.

- **Empirical Impact**: 이 리뷰는 지표를 ‘정답/선호도’에서 ‘의도된 최종 상태에 도달했는지(작업 완료)’로 바꾸는 평가 패러다임 전환이 필요함을 강조하며, state–action–observation 궤적 기반의 벤치마크·생태계형 평가가 더 설득력 있다고 제안합니다. 또한 OpenAI o1·DeepSeek R1 같은 추론 강화 모델과 OpenClaw 스타일 워크스테이션 시스템의 결합이 ‘스스로 진화하는 업무 수행’으로 가는 실마리임을 보여줍니다. 결과적으로 분야가 챗봇형 생성에서 신뢰 가능한 디지털 콜리그형 시스템으로 재정의되는 데 방향성을 제공하며, 거버넌스·안전·프라이버시·업무 경계 같은 사회기술적 의제까지 함께 부각합니다.



### When the Tool Decides: LLM Agents Defer Blindly to Graph Neural Network Tools, and Stronger Backbones Defer Mor (https://arxiv.org/abs/2606.14476)
Comments:
          9 pages, 2 figures. Under review at TMLR

- **Prior Approaches**: 기존 그래프-LLM 에이전트 연구는 에이전트가 GNN 같은 학습된 예측기를 ‘도구’로 호출할 때, 신뢰할 만하면 쓰고 아닐 땐 텍스트·이웃·자기 추론으로 보정한다는 전제를 둔다. 또 tool-use 개선이 보고되지만, 도구 출력 위에 ‘판단’을 실제로 덧붙이는지 정면 검증한 측정은 드물었다. 본 논문은 이러한 암묵적 가정이 얼마나 잘 맞는지 측정 가능 질문으로 재정의한다.

- **Core Contribution**: 저자들은 고정된 GNN을 ReAct 스타일 LLM 에이전트의 명시적 도구로 제공했을 때, 에이전트가 결과를 ‘증거로 가중’하는지 아니면 ‘그대로 순응(obey)’하는지 논리적으로 분해해 측정한다. 핵심 지표는 에이전트 최종 예측과 원본 GNN 예측의 일치율(파롯 점수)과, 도구 순응으로 잃는 정확도 격차(oracle gap)다. 그 결과는 에이전트가 판단을 발휘하기보다 GNN 출력에 고정되는 ‘GNN parrot’ 현상으로 귀결된다.

- **Technical Challenges**: 도구 출력이 맥락에 들어온 뒤 에이전트가 실제로 무엇을 했는지(채택 vs 가중)를 분리해 관찰해야 한다는 점이 가장 큰 기술적 과제다. 이를 위해 고정 GNN을 도구로 연결하고, 텍스트만 쓰는 대안과(그래프 도구 비사용), 이웃의 학습 라벨을 미니멀 그래프 탐색 도구로 보는 대안도 함께 둬 비교한다. 또한 선택적 호출을 복구책으로 두고, 신뢰성 게이팅(순도 기반·학습 라우터)과 정보 한계(테스트 시간 특징으로 가능한 최대 성능을 상한)까지 실험해 ‘게이트 설계’만으로는 해결되지 않음을 보인다.

- **Empirical Impact**: ogbn-arxiv에서 에이전트+GNN 도구 일치율이 약 97.6~99.2%로 매우 높아, 도구 없는 자기 추론 일치율(대략 17~37%)과 큰 대비를 이룬다. 더 놀라운 점은 스케일이 커질수록 순응이 약해지지 않고 오히려 강화되어, 도구를 실제로 호출할 수 있는 구간부터 일치율이 0.60→0.98까지 상승한다는 것이다. oracle gap은 대안이 강해지는 영역에서 오히려 커지며, 선택적 호출 게이트는 고동질성(high homophily) 격차의 절반가량만 회복하지만 전역 성능 이득은 없고, ‘표준 불확실성 프록시’로는 오라클 여유의 1/6~1/3만 회수 가능하다는 정보 상한이 제시된다. WikiCS에서도 유사한 파롯과 양의 oracle gap이 재현되어, 이 실패 모드가 특정 데이터에만 국한되지 않을 가능성을 시사한다.



### GitOfThoughts: Version-Controlled Reasoning and Agent Memory You Can Replay, Diff, and Merg (https://arxiv.org/abs/2606.14470)
Comments:
          10 pages, 1 figure, 9 tables

- **Prior Approaches**: 기존 연구는 추론 과정을 내부 메모리·버퍼로만 다루며, 에피소드가 끝나면 체인/트리 기록이 사라져 재현·감사·병합이 어렵다는 한계가 있었다. 또한 벡터·그래프·커스텀 스토어 같은 ‘메모리 형식’은 제안됐지만, 정확도 향상 여부는 일관되지 않았고 특히 방법 전이(transfer) 관점의 증거가 약했다.

- **Core Contribution**: 이 논문은 LLM 에이전트의 추론 트리를 GitOfThoughts로 버전관리해, 커밋·노트·태그로 점수화/결과를 기록하고 retrieval을 git log처럼 수행하게 한다. 이를 통해 추론을 재실행(replay)·감사(audit)·병합(merge) 가능하게 하면서도 정확도는 기존 메모리 스토어와 ‘동등’한 수준을 목표로 검증한다. 동시에 “어떤 메모리 형식이든 새 문제(novel problem) 정확도를 올리나?”라는 가설을 전면적으로 반박·검정한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 추론 노드를 안정적으로 점수/결과와 함께 영속 저장하고, (2) 저장된 내용을 에이전트 내부 retrieval로 공정하게 비교하며, (3) 여러 에이전트 경험을 충돌 없이 병합해 감사를 가능하게 하는 것이다. 논문은 생각 생성은 기존 tree-of-thoughts/ ReAct 루프를 유지하되, score 시점에 thought·scores·trace를 커밋하고 태그로 outcome을 남기는 형태로 Git 매핑을 구현해 “저장/조회/병합”만 바꿔가며 실험한다.

- **Empirical Impact**: GPQA-Diamond와 MATH-500에서 교차문제·교차에피소드 전반의 사전등록(pre-registered) 검증 결과, 대부분의 메모리 형식은 새 문제 정확도를 신뢰할 만하게 개선하지 못했다(특히 40개 샘플에서 보였던 git의 유망 신호는 더 큰 재현에서 붕괴). 다만 메모리가 ‘먹히는’ 조건은 복사 가능성(copyability) 임계값으로, 검색된 사례가 거의 중복에 가까울 때(유사도 약 0.8 이상) 정확도가 급상승하고, 그 아래에서는 이득이 없다. 반면 방법 전이(작동 원리 추출)는 모델 스케일을 키워도 거의 나타나지 않았고, 정확도를 주로 움직이는 안정적 레버는 메모리가 아니라 테스트 시 sampling(self-consistency) 같은 추론 선택 전략이었다.



### Causal Object-Centric Models for Planning with Monte Carlo Tree Search (https://arxiv.org/abs/2606.14418)
- **Prior Approaches**: 기존 모델 기반 강화학습(MBRL)은 월드 모델로 상상 전개를 하며 표준적으로 Dreamer류나 MuZero류의 잠재공간 계획을 활용해 왔다. 시각 환경에서 성능 병목은 관측을 단일 전역 표현(CNN 기반)으로 뭉치는 방식에 있었고, 이를 개선하기 위해 Slot Attention(슬롯 주의) 기반의 객체 중심 표현 학습과 이를 MBRL에 결합한 방법들이 나왔다. 다만 기존 객체 중심 MBRL은 객체 상호작용을 충분히 명시적으로 모델링하지 못하거나, 주석된 분할 마스크 의존(FULL-supervision) 문제가 남아 있었다.

- **Core Contribution**: COMET은 MuZero 스타일 잠재 계획에 객체 수준 귀납 편향을 추가해, 슬롯 구조(latent slot space)에서 MCTS(Monte Carlo Tree Search)로 계획을 수행하는 객체 중심 MBRL 알고리즘을 제안한다. 핵심은 고정된 비지도 객체 중심 인코더로부터 얻은 슬롯에 대해, 트랜스포머 월드 모델이 미래 슬롯과 보상을 예측하고, 정책·가치 헤드는 대상 토큰을 기준으로 ‘객체-인과 attention’을 통해 의사결정에 중요한 엔터티에 집중하게 만든다는 점이다. 또한 행동을 객체 슬롯에 결속시키는 ‘action-slot fusion’ 메커니즘을 도입해, 행동이 어떤 객체에 영향을 주는지 학습적으로 결합한다.

- **Technical Challenges**: 객체 중심 슬롯은 시간에 따라 순서가 뒤바뀔 수 있어(순열 불변성과 할당의 비결정성) 동적 환경에서 일관된 객체 추적이 어렵다. COMET은 이를 완화하기 위해 에피소드 내부에서 다음 시점의 슬롯을 현재 시점 슬롯으로 초기화하는 방식으로 시간적 안정성을 확보하고, 인코더는 비강화학습 데이터로 미리 학습한 뒤 고정해 표현 요동을 줄인다. 또 행동을 단일 임베딩으로 슬롯 전체에 강제로 압축하면 정보병목이 생기므로, 슬롯마다 행동 임베딩을 결합(슬롯-조건화)해 action-slot fusion을 설계하고, 이를 트랜스포머 입력으로 사용해 전이 예측을 안정화했다.

- **Empirical Impact**: 평가는 Object-Centric Visual RL benchmark, ManiSkill, Robosuite, VizDoom의 8개 시각·동역학 다양한 태스크에서 수행했으며, COMET은 훈련 초반부터 평균 정규화 점수(mean normalized score)에서 객체 중심/단일(monolithic) 기준선을 앞서는 경향을 보였다. 특히 대상 객체와 교란 객체가 뚜렷한 시각적으로 단순한 과업에서는 object causal attention이 실제로 과제 관련 객체에 높은 중요도(인과 점수/가중치)를 부여해 성능 격차를 만들었다. 반면 Block Lifting과 Cube Pushing처럼 표현이 흔들리는 제어·시각 복합 태스크에서는 객체 표현이 슬롯을 제대로 분리하지 못할 때 한계가 나타났고, 이는 객체 표현 강건성이 성능의 관건임을 시사한다.



### CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning (https://arxiv.org/abs/2606.14415)
Comments:
          Accepted as a Spotlight paper at the 43rd International Conference on Machine Learning (ICML 2026)

- **Prior Approaches**: 기존 Safe RL은 CMDP 제약 하에서 기대 보상을 최대화하되 비용 제약을 만족시키도록 학습한다. 신뢰영역 기반(2차) 방법은 이론적 보장이 강하지만 Fisher 행렬 역산 같은 비용이 커지고, 1차 방법은 비교적 가벼운 대신 안정성이 문제될 수 있다. 프라이멀-듀얼 라그랑지 방법은 제약에 대한 듀얼 업데이트가 지연되어 안전 경계 근처에서 진동(oscillation)과 장기 제약 위반이 생기며, 페널티 기반 방법은 페널티 계수 튜닝에 민감해 과도한 보수성 또는 지속 위반을 유발한다.

- **Core Contribution**: 이 논문은 Constraint-Sensitive Policy Optimization(CSPO)로, 프라이멀-듀얼의 단순함을 유지하면서 정책 업데이트 시 제약 함수의 국소 민감도(local constraint sensitivity)를 반영한다. 안전 위반이 발생했을 때만 제약 위반을 줄이기 위한 보정 항을 추가하고, 이를 안전 경계까지의 “부호 있는 최단 거리” 개념에서 유도한 correction으로 설계한다. 또한 CSPO는 원래 CMDP의 KKT 해(최적 해/제약 만족 집합)를 보존하도록 구성해, 안전성을 ‘학습 과정의 즉각적 복구’로 개선하되 목표 자체를 바꾸지 않는다.

- **Technical Challenges**: 핵심 난제는 듀얼 업데이트 지연으로 인한 진동을 줄이면서도, 제약 경계의 기울기(가파름/평탄함)에 따라 필요한 복구 강도가 달라지는 상황을 안정적으로 처리하는 것이다. CSPO는 제약 위반 정도에 더해 제약 그래디언트 크기‖∇g(θ)‖로 correction 스케일을 조정해, 경계가 가파르면 과보정을 줄이고 평탄하면 더 강하게 복구하도록 적응한다. 실용 구현에서는 PPO 스타일의 클리핑된 surrogate로 비용 항을 최적화하되, 위 민감도 가중치와 효과적 라그랑지(λ_eff)를 위반 시에만 강화해 “더 똑똑한 회복”을 제공한다.

- **Empirical Impact**: Safety Gymnasium의 9개 연속제어 안전 과제(보행/로봇 다리 locomotion, 내비게이션)에서 CSPO는 기존 프라이멀-듀얼 및 페널티 기반 SOTA 대비 더 빠른 안전 복구와 보상 보존(Reward Preservation)을 동시에 보였다. 논문은 회복 동역학을 Time-To-Safety(TTS), Reward Preservation(RP), Violation frequency(VF)로 측정해, 단순히 최종 제약 성능이 아니라 학습 중 안정적 복구가 개선됨을 보여준다. 결과적으로 제약 경계 근처에서의 진동과 장기 위반을 줄이면서도 높은 constrained return을 달성해, Safe RL의 “안전 복구 품질”을 끌어올리는 실증적 의미가 크다.



### Communication Policy Evolution for Proactive LLM Agents (https://arxiv.org/abs/2606.14314)
- **Prior Approaches**: 기존의 능동형 LLM 에이전트 연구는 ‘무엇을 물을지/어떻게 상호작용할지’에 집중했지만, 텍스트와 구조화 UI 중 어떤 통신 채널을 쓸지 같은 채널 선택 자체는 상대적으로 덜 다뤄졌다. 또한 개인화 연구는 사용자 성향을 반영하는 데는 진전이 있었지만, 그 성향에 맞춰 통신 방식을 설계·선택하는 문제를 설계 변수로 보지 않는 경우가 많았다.

- **Core Contribution**: 이 논문은 에이전트의 통신 방식을 채널 선택 문제로 정식화하고, 텍스트 기반 ask_question과 UI 기반 generate_ui를 결합하는 Communication Policy를 제안한다. 추가로 User–Agent와 Planner–Executor의 두 설정으로 ‘사용자 관점의 정보 비대칭’과 ‘협업 관점의 정보 비대칭’을 분리해 채널 효과를 검증한다.

- **Technical Challenges**: 핵심 난관은 혼합 채널을 열어줘도 ‘언제 텍스트를 쓰고 언제 UI를 쓸지’ 정책이 부재하면 성능이 흔들린다는 점이다. 이를 해결하기 위해 Communication Policy Evolution(CPE)을 도입해, 모델 가중치 수정 없이 프롬프트만 반복적으로 진화시키며 롤아웃 결과 분석→JSON 패치 제안→훈련/검증 기반 게이팅으로 단조 개선(held-out 성능 저하 방지)을 보장한다.

- **Empirical Impact**: 네 개 벤치마크(SWE-bench, TravelGym, τ2-bench, WebArena)와 다양한 페르소나에서 텍스트와 UI는 상호보완적 성격을 보였다. 전반적으로 혼합 정책이 생산성(task success)을 가장 많이 이끌었고, UI는 반응 품질과 페르소나 준수에서, 텍스트는 과업 진행 효율에서 강점을 보였으며 CPE는 프롬프트 최적화만으로 여러 설정에서 최고 생산성을 달성했다. 결과적으로 ‘통신 채널 선택’이 LLM 에이전트 설계의 중요한(하지만 덜 탐구된) 독립 차원임을 실험적으로 확인했다.



### HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry (https://arxiv.org/abs/2606.14249)
- **Prior Approaches**: 기존 에이전트 인프라는 프롬프트·도구·메모리·제어를 ‘런타임 하네스’로 제공하지만, 대부분 사람이 손으로 만든 정적 스캐폴딩에 머물러 있습니다. 또한 하네스 구성은 정형화된 합성(조립) 단위로 다뤄지지 않아 모델/도메인이 바뀔 때마다 재작성되기 쉽고, 실행 중 생성되는 트레이스도 체계적으로 개선으로 되돌아가지 못했습니다.

- **Core Contribution**: 이 논문은 하네스를 첫째 클래스 객체로 취급해, 조합(composition)·적응(adaptation)·진화(evolution)까지 한 프레임에서 다루는 HarnessX를 제안합니다. HarnessX는 타입 기반 하네스 프리미티브를 “대치 대수(substitution algebra)”로 조립하고, 트레이스를 기반으로 하네스를 업데이트하며, 그 결과를 다시 모델 학습 신호로 연결해 하네스-모델 루프를 닫습니다.

- **Technical Challenges**: 핵심 난제는 (1) 하네스 수정이 검증기만 속이는 ‘리워드 해킹’, (2) 일부 작업 개선이 다른 작업을 망가뜨리는 ‘재앙적 망각’, (3) 안전한 국소 편집에 과도하게 치우치는 ‘과소 탐색’ 같은 RL 병리 현상이 하네스 진화에서 더 크게 나타나는 점입니다. HarnessX는 트레이스 관측을 구조화해 Digester-Planner-Evolver-Critic의 4단계 파이프라인으로 후보 편집을 제안·평가하고, 결정적 게이팅(해결된 작업 회귀 방지)과 변형 격리(variant isolation)로 안정적인 진화를 유도합니다.

- **Empirical Impact**: 5개 벤치마크(ALFWorld, GAIA, WebShop, tau^3-Bench, SWE-bench Verified)에서 HarnessX는 평균 +14.5% 성능 향상을 보였고, 구성에 따라 최대 +44.0%까지 개선되었습니다. 특히 기준선이 낮은 설정에서 이득이 커 역스케일링 패턴이 관측되었으며, 하네스 전용 진화 대비 모델과의 공동 진화는 추가로 +4.7%를 더했습니다.



### AFFORDANCE20Q: Evaluating Affordance Reasoning from Physical Properties (https://arxiv.org/abs/2606.14240)
- **Prior Approaches**: 기존의 어포던스(affordance) 추론 벤치마크는 평가 입력에서 물체의 이름/정체성이 노출되는 경우가 많았습니다. 그래서 모델은 물체-어포던스 매핑을 외워 “추론” 없이도 정답을 고를 수 있어, 물리 속성 기반 추론 능력을 분리해 측정하기 어렵습니다.
또한 20-Questions류 멀티턴 질문 과제에서도 후보 공간이 주로 ‘물체’ 자체로 구성되어, 물체는 고정하고 ‘어포던스’를 추론해야 하는 문제 설정은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 물체의 정체성을 숨기고 물리 속성에 대한 예/아니오 질문만으로 어포던스를 맞히게 하는 신규 벤치마크 Affordance20Q를 제안합니다. 각 게임은 후보 어포던스 8개 중 정답 어포던스를 20턴 이내에 식별하며, 454개 물체와 59개 어포던스를 기반으로 총 1,009개 게임을 구성했습니다.
또한 지식 베이스(KB) 근거로 어포던스 규칙을 유도해 LLM의 자유형 추측을 줄이려는 KB-Anchored Rule Induction(KARI) 파이프라인도 함께 제공합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘정체성 기억’이 통하지 않는 상황에서, 모델이 게임이 진행될수록 후보군을 실제로 줄이는 판별적 질문을 생성하는지 확인하고 개선하는 데 있습니다. 저자들은 KL 기반 정보이득(IG) 분석으로 모델이 턴이 늘어날수록 구별력을 갖는 질문을 충분히 묻지 못한다는 실패 양상을 진단합니다.
이를 개선하기 위해 KARI는 KB에서 얻은 물리 상식 근거로 AND/OR 트리 형태의 규칙(예: 재질, 형태, 크기, 표면, 특정 부분 PART)을 생성·검증하며, KB 범위를 벗어나는 값/동의어 드리프트를 감사(auditor)로 제어합니다.

- **Empirical Impact**: 실험에서 15개 최신 LLM 모두 사람이 달성한 성능 대비 큰 격차를 보이며, 최고 모델도 사람과 비교해 약 20점(해당 수치 범위 내) 수준의 하락이 관찰됩니다. 또한 정보이득 분석은 모델이 후반으로 갈수록 후보를 좁히는 질문을 덜 한다는 경향을 재확인합니다.
KARI를 적용하면 공개 오픈소스 LLM이 최대 15.2점 향상되며 격차를 일부 축소하지만, 추가 개선은 KB 커버리지 한계에 의해 제한된다고 보고합니다.



### SkillAudit: Ground-Truth-Free Skill Evolution via Paired Trajectory Auditing (https://arxiv.org/abs/2606.14239)
Comments:
          20 pages, 5 figures

- **Prior Approaches**: 기존 스킬 진화 연구는 주로 검증 점수, 숨겨진 테스트 결과, 환경 보상, 오라클 합격/불합격 같은 ‘정답에 가까운 신호’를 활용해 스킬 업데이트를 승인/거부했다. 또는 기업 지식베이스, 지원 티켓, 로그, 결과 보상 등 외부 감독을 더 풍부하게 도입해 학습 신호를 만들기도 한다. 하지만 실제 배포 환경에서는 과제 설명과 워크스페이스 정보만 있고 정답 보상/테스트/스크로어 함수에 접근하지 못하는 경우가 많다.

- **Core Contribution**: SkillAudit은 숨겨진 정답 피드백 없이도 스킬 문서를 진화시키는 프레임워크를 제안한다. 핵심은 ‘페어드 트래젝토리 오디팅’으로, 같은 과제를 후보 스킬을 넣었을 때와 넣지 않았을 때 두 번 실행해 행동 차이를 스스로 관측 가능한 근거로 삼는 것이다. 이렇게 얻은 차이를 문서의 특정 구절로 연결해, 편집이 무엇을 바꾸는지 진단하고 업데이트를 안전하게 반영한다.

- **Technical Challenges**: 정답 보상이 관측되지 않으면, 단순히 실행 결과의 차이를 ‘어떤 구절을 고쳐야 하는지’로 바꾸는 진단 문제와, 평가자 드리프트 및 회귀를 막는 안정성 문제가 동시에 발생한다. SkillAudit은 PACE(Process-Aligned Contrastive Evaluation)로 트래젝토리 분기 지점을 스킬 문서의 구절에 앵커링해 4개 축(절차 준수, 산출물 증거, 일관성, 효과 차이)별 진단 신호를 만들고, 과제 설명에서 컴파일된 고정 구조 검증기(Anchor Verifier)로 하드 제약 위반이나 성능 저하를 차단한다. 또한 Refine은 노이즈를 제거하고 Repair는 충돌 구절을 교체하도록 편집 파이프라인을 분리해, 효과적인 스킬을 깨는 과도한 수정과 원인 불명의 보수적 패치를 동시에 줄인다.

- **Empirical Impact**: SkillsBench 89개 컨테이너 과제(8개 전문 도메인)에서 SkillAudit은 평균 과제 보상 73.9%를 달성해, 스킬 없는 에이전트(40.9%)와 정적 전문가 스킬(56.7%)을 크게 앞섰다. 진화 과정 어디에서도 숨겨진 테스트, 기준 해답, 외부 스코어링 함수를 사용하지 않았다는 점이 결과의 실용성을 높인다. 아울러 ‘관측 가능성 경계’를 제시해, 스킬의 지식이 검증기가 읽을 수 있는 구조적 흔적(파일 생성, 컴파일, 수치 결과 등)을 남길 때 진화가 잘 되고, 절차 지식처럼 보이지 않으면 회복/수정이 제한됨을 보여준다.



### Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL (https://arxiv.org/abs/2606.14211)
- **Prior Approaches**: LLM 에이전트는 실행 결과·에러 메시지·도구 출력 같은 환경 피드백을 받지만, 대부분 다음 행동 선택의 단서로만 사용하고 ‘정답 여부’ 자체를 피드백에 근거해 검증하는 방식은 약했다. 기존 자기평가 연구는 대체로 환경 피드백 없이 모델이 스스로 생성한 답에 대한 신뢰도를 말하는 데 그쳐, 본 논문의 문제 설정(피드백 후 자기점검)과 차이가 있다. 또한 outcome-only RL(예: GRPO 계열)은 정답/오답 결과만으로 학습 신호를 만들기 때문에, 정직한 에러 플래그가 불리해지는 구조적 한계가 관측된다.

- **Core Contribution**: 이 논문은 LLM 에이전트가 환경 피드백을 본 뒤에도 자신의 정오를 제대로 판정하지 못하는 ‘reflection gap’을 정식화하고, 이를 해결하는 방법으로 RefGRPO를 제안한다. 핵심은 표준 RL 보상(결과 outcome)에 더해, 모델이 생성한 ‘피드백 후 reflection’이 실제 outcome과 일치하는지를 비교해 보정(calibration)하도록 만드는 것이다. 추가로 calibration 보너스의 비중을 학습 중 동적으로 조절해, 초기에 보정을 충분히 학습하되 점차 과제 정확도에 집중하도록 설계했다.

- **Technical Challenges**: 기술적 난관은 credit-assignment mismatch다: outcome-only RL은 rollout의 최종 결과만으로 advantage를 주기 때문에, 정답이 맞는데도 에러라고 플래그한 경우와 정답이 틀렸는데 정직하게 에러라고 플래그한 경우가 동일 outcome 보상 신호에 의해 역학습될 수 있다. RefGRPO는 이 문제를 ‘free calibration bonus’로 완화한다—training 중 이미 이용 가능한 reflection과 실제 outcome의 일치 여부를 이진 보너스로 계산해 reward에 대비시키며, calibration 계수 α(t)를 스케줄링해 보정 학습과 과제 학습의 균형을 맞춘다.

- **Empirical Impact**: text-to-SQL 에이전트 환경에서 RefGRPO는 reflection calibration을 크게 개선하면서도 정확도를 함께 끌어올렸다. 예를 들어 underconfidence rate는 44.4%에서 7.7%로, task accuracy는 75.1%에서 76.5%로 향상되며, 다섯 개 벤치마크 평균에서 통합 지표인 ChowScore도 73.0%에서 76.5%로 상승했다. 더 나아가 calibrated reflection은 환경 피드백 기반의 자기검증(자기개선에 pseudo-reward 활용, 선택적 예측에서 올바른 rollout만 커밋) 성능 향상으로 이어짐을 보여준다.



### When Should Agent Trust Be Conditional? Characterizing and Attacking Skill-Conditional Reputation in Agent Swarms (https://arxiv.org/abs/2606.14200)
Comments:
          18 pages, 8 figures, 2 tables

- **Prior Approaches**: 기존 에이전트 라우팅·신뢰는 PageRank/EigenTrust 계열처럼 각 에이전트를 단일 전역 점수(글로벌 신뢰도)로 요약해 왔습니다. 하지만 AppWorld처럼 에이전트가 스킬별로 강점이 크게 갈리면, 전역 최상위 에이전트에 고정 라우팅하는 순간 전문화의 이득이 사라집니다. 또한 스킬별 관측치가 희소할 때는 단순 per-skill 추정이 노이즈에 취약해, 그래프 기반 전이적(트랜지티브) 차용을 다른 축(에이전트×스킬)으로 옮겨오는 접근이 자연스럽지만 안전성까지 정량화된 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 신뢰의 “대상”을 단일 점수가 아니라 스킬 조건부 신뢰 R(i|k)—스킬 k 과제를 맡길 때 에이전트 i를 얼마나 신뢰할지—로 재정의합니다. 동시에 스킬 조건부가 언제 가치가 있는지, 스킬 간 정보(증거)를 얼마나 빌려야 하는지, 그리고 그 빌림이 공격 표면을 만들지까지 세 가지로 쪼개어 검증 가능한 질문으로 만듭니다. 기여는 새로운 라우터를 막 제안하기보다, 조건부 신뢰가 이득을 내는 “조건의 지도(phase diagram)”와 그 반대편(취약성)의 비용을 함께 계량화한 점에 있습니다.

- **Technical Challenges**: 핵심 난관은 스킬별 증거가 희소해 per-skill 신뢰를 그대로 추정하면 분산이 커진다는 데 있습니다. 논문은 스킬 간 상관 구조를 결합 행렬로 두고, 결합 강도 β로 증거를 부분 풀링(empirical-Bayes 스타일)하는 추정자를 설계해 분산을 낮추되 편향을 통제합니다. 더 나아가 CIVT(Conditional Information Value Test)로 모델 실행 없이 기존 로그만으로 조건부가 실제로 가치 있는 구간인지 판별해, “언제 쓰면 안 되는가”를 사전에 걸러내도록 했습니다.

- **Empirical Impact**: AppWorld의 서로 “진짜로 이질적인” 14개 공개 에이전트 풀에서, 조건부 신뢰가 유의미한(작지만 실제) 향상을 내는 구간이 관측되었고 스킬별 최고 에이전트가 스킬에 따라 실제로 바뀌는 패턴도 확인됩니다. 다만 결합 강도 β가 데이터 효율을 주는 동시에 평판 세탁(정보 오염) 통로가 될 수 있어, 공격자가 한 스킬에는 증거를 싸게 만들고 목표 스킬에는 증거를 전혀 만들지 않아도 라우팅 후회가 0에서 0.94까지 치솟는 시나리오를 보입니다. 제로-에비던스 게이트 같은 완화책은 최악을 줄이지만 완전 차단은 아니며, 결국 “조건부 신뢰의 이득-보안 비용” 트레이드오프를 정량으로 제공했다는 점에서 분야에 중요한 기준점을 남깁니다.



### VeriGeo: Controllable Geometry Question Generation with Numerical and Analytical Verification (https://arxiv.org/abs/2606.14176)
Comments:
          32 pages, 4 figures, 9 tables

- **Prior Approaches**: 기존 기하 문제 생성은 다이어그램(도형)을 먼저 만들고 문항을 뒤이어 구성하는 방식(다이어그램-퍼스트)이 주로 타당성을 높였지만, 사용자 지정 개념·난이도·도형 요구 같은 제약을 유연하게 반영하기 어렵습니다. 반대로 시드(seed) 기반 재작성 방식은 변형 생성은 편하나, 문장–도형–해설 간 불일치나 환각 제약이 생겨도 자동 탐지와 복구가 제한적이어서 신뢰성에 비용이 큽니다. 또한 시드 기반 다양성은 시드 분포에 크게 묶이는 경향이 있어 개념 커버리지가 제한됩니다.

- **Core Contribution**: VeriGeo는 사용자 제약(목표 개념, 난이도, 도형 복잡도 등)을 바탕으로 기하 문제와 도형을 생성하되, 실행 가능한 추론 흔적(action sequence)으로 문제-도형-해설의 정합성을 검증 가능한 형태로 묶는 폐루프(닫힌 고리) 프레임워크를 제안합니다. Author 에이전트가 문항과 도형을 만들고 Solver 에이전트가 증명에 정렬된 해설을 생성하며, 두 에이전트는 동일한 공유된 행동 시퀀스를 통해 검증 가능한 표현을 유지합니다. 핵심은 “생성”이 아니라 “검증 가능한 생성”을 목표로 한다는 점입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 수치적 조건(예: 공선/수직)이 도형 좌표에서 실제로 성립하는지, (2) 해당 제약이 수학적으로 실현 가능한지(방정식 시스템의 해 존재), (3) 문항 텍스트·도형·행동 시퀀스·증명 논리가 전역적으로 모순 없이 맞물리는지 동시에 보장하는 것입니다. VeriGeo는 3단계 검증(수치 검증, 분석 검증, 논리 검증)을 수행하고 실패 시 Author/Solver의 자기 성찰(self-reflection)로 복구 가능한 오류는 수정하며, 복구 불가한 경우는 거절합니다. 특히 분석 검증에서는 부동소수 오차를 줄이기 위해 유리수·근호 등 exact 표현을 사용하고, 방정식이 과/부정확해지는 문제를 gauge fixing과 rank-aware filtering 같은 공학적 기법으로 안정화했습니다.

- **Empirical Impact**: 5개 LLM 백본에서 검증 없이 생성한 원본은 합격률이 평균 29.02%로 낮았지만, VeriGeo의 검증 유도 복구가 상당 부분을 되살려 평균 repaired 비율이 25.78%에 이르렀고 거절은 남은 일부에 그쳤습니다. 또한 검증된 생성 데이터 8.7k개로 Qwen2.5-VL-7B-Instruct를 지도 미세조정한 결과 PGPS9K 59.40%, GeoQA 82.74%, MathVista-GPS 75.96%를 달성해, GeoQA에서 MLLM 기반 종단(end-to-end) 솔버 중 최고 보고 성능을 기록했습니다. 데이터 측면에서도 354개 서로 다른 기하 개념을 포괄하는 등 개념 다양성이 크게 넓어져, “대규모”보다 “검증된 품질”이 기하 멀티모달 추론 성능에 직접적인 이득을 준다는 메시지를 강화합니다.



### FactoryLLM: A Safe and Open-Source AI Playground for Evaluating LLMs in Smart Factories (https://arxiv.org/abs/2606.14119)
Comments:
          6 pages, 3 figures, IEEE INDIN 2026

- **Prior Approaches**: 기존 제조 분야 LLM/RAG 연구는 대체로 단일 기기 문서나 단일 작업(유지보수 로그 요약, 질의응답 등)에 초점이 맞춰져 있었다. 키워드 검색은 용어 불일치에 약하고, 규칙 기반은 지식공학 비용이 커 유지보수가 어렵다. 지식 그래프는 구조화가 유리하지만 머신 수가 늘면 구축·유지 비용이 급격히 증가한다. 또한 RAG 평가는 대부분 임의 설문·수작업 점검에 의존해 비교 가능성이 낮았다.

- **Core Contribution**: 이 논문은 여러 머신에 분산된 기술 문서를 한 세션에서 묶어 추론 성능을 평가하는 오픈소스 “FactoryLLM”을 제안한다. 특히 교차-문서(하드웨어+소프트웨어) 질의응답을 대상으로, Retrieval-Augmented Generation이 실제로 얼마나 근거 기반(grounded)으로 답하는지 시험한다. 사용자가 로컬·오픈소스 모델을 선택할 수 있어 민감한 산업 데이터를 외부로 보내지 않고도 재현 가능한 실험 환경을 제공한다.

- **Technical Challenges**: 교차-문서 추론의 핵심 난제는 (1) 문서 간 용어·추상화가 달라 의미 정렬이 어렵고, (2) 긴 매뉴얼에서 관련 단락을 정확히 끌어오는 검색 정밀도도 흔들리며, (3) 여러 근거를 조합해 안전한 답을 구성해야 한다. FactoryLLM은 벡터 기반과 그래프 기반 검색을 모두 지원하고, Input–Output, Chain-of-Thought, Tree-of-Thought, Graph-of-Thought 등 다양한 프롬프트 전략을 구성형으로 교체해 비교한다. 또한 응답이 근거 문맥에 기반했는지와 검색 품질을 함께 보려 RAGAS와 NVIDIA LLM-as-a-Judge를 이중 평가로 운용한다.

- **Empirical Impact**: AIV와 Mobile Planner 매뉴얼 약 600페이지에서 뽑은 30개의 교차-머신 유지보수 질의를 대상으로 3개 오픈 LLM을 평가했으며, 모든 모델이 groundedness 0.88 이상을 기록했다. 반면 retrieval 정밀도(예: context precision)는 0.46~0.51로 상대적으로 낮아, 병목이 생성보다 검색에 있음을 수치로 확인했다. RAGAS의 faithfulness·precision 신호와 NVIDIA의 groundedness·context relevance 신호가 보완적으로 작동해, 향후 재랭킹·질의 분해 같은 개선 지점을 더 명확히 드러낸다는 점에서 의미가 있다.



### Applicability Condition Extraction for Therapeutic Drug-Disease Relations (https://arxiv.org/abs/2606.14031)
- **Prior Approaches**: 기존 생의학 관계 추출은 약물-질병 간 ‘관계 존재/유형’을 맞히는 데 초점이 맞춰져, 치료가 통하는 조건(맥락)까지는 명시적으로 모델링하지 못했습니다. 또한 세부 정보 추출(예: 이상반응 추출)은 사건/속성 스팬을 다루지만, 치료 적용 가능성을 제한하는 조건을 약물-질병 관계 단위로 정리하는 연구는 드뭅니다.

- **Core Contribution**: 이 논문은 치료제-질병 관계에서 적용 가능 조건을 뽑는 신규 태스크인 ‘약물-질병 적용가능 조건 추출’을 제안합니다. 이를 위해 Drug-ACE라는 첫 데이터셋을 구축했으며, 1,119개의 약물-질병 페어(총 2,290개 조건 스팬)를 PubMed 초록에서 수작업 라벨링했습니다. 또한 관계 역할 정보를 반영하도록 LoRA를 강화한 Role-Conditioned LoRA(RCLoRA)를 제시합니다.

- **Technical Challenges**: 핵심 난관은 조건이 여러 문장에 걸쳐 암묵적으로 흩어져 있고, 같은 문서에 여러 약물·질병이 공존하며, 스팬이 토큰화로 인해 희석된다는 점입니다. 저자들은 질병을 ‘관계의 주어’, 약물을 ‘관계의 목적어’로 보고 입력 토큰에 Subj/Obj/NA 역할을 라벨링한 뒤, 역할별로 다른 LoRA 저랭크 투영(역할 조건 저랭크 업데이트)을 학습해 컨텍스트에서 ‘누가 누구에 적용되는지’를 명시적으로 주입합니다.

- **Empirical Impact**: Drug-ACE에서 span 및 조건 유형까지 함께 평가하는 실험에서 RCLoRA는 다양한 백본과 매칭 방식(하드/소프트) 전반에 걸쳐 강력한 기준선을 일관되게 능가했습니다. 특히 표준 LoRA는 복잡한 유형(예: 동반질환)에서 성능이 크게 흔들렸지만, RCLoRA는 해당 유형에서도 의미 있는 성능을 보였습니다. 결과적으로 치료 주장 해석과 임상 의사결정 지원에 필요한 ‘조건부 근거’를 데이터와 모델로 더 구체화하는 데 기여합니다.



### Formalizing Numerical Analysis: An Agent Pipeline and Quality Audit Beyond Kernel Acceptanc (https://arxiv.org/abs/2606.14000)
- **Prior Approaches**: 기존 자동정리(autoformalization) 연구는 개별 정리 단위 번역 성능을 높이거나, mathlib에 이미 잘 정리된 영역에서 코딩 에이전트가 생성한 결과가 커널에서 컴파일/타입체크되는지로 “성공”을 판정해 왔다. 최근에는 프로젝트 규모로도 확장됐지만, 주로 컴파일 통과 여부만 보고 의미적 정확성은 제한된 샘플링으로만 확인되는 경우가 많다.

- **Core Contribution**: 이 논문은 mathlib에 거의 없는 수치해석 교재(상미분방정식 수치해석)를 코딩 에이전트로 형식화해, 기존과 달리 새 이론을 “처음부터” 구축하는 능력을 시험한다. 동시에 컴파일을 넘어 정형화 품질을 의미적 정합성, mathlib 재사용, 파일 간(프로젝트 내부) 재사용의 3차원으로 체계적으로 감사(audit)하는 재현 가능한 평가 프레임워크를 제안한다.

- **Technical Challenges**: 핵심 난점은 (1) 교재의 자연어 명제를 Lean 선언이 의미까지 충실히 반영하도록 만드는 것, (2) 에이전트가 커널 통과만으로 드러나지 않는 부분-누락/가정 추가/매개변수 제한 같은 “부정확한 패턴”을 피하는 것, (3) 이를 대규모로 검증 가능한 기준으로 자동화하는 것이다. 저자들은 Planner-Worker-Evaluator 구조의 에이전트 루프와, LLM-as-judge 기반의 양방향 함의(Q1/Q2) 루브릭 및 역번역(back-translation)까지 결합해 의미적 정확성과 재사용 품질을 함께 판정한다.

- **Empirical Impact**: Butcher 교재를 대상으로 했을 때 완전 증명까지 포함한 형식화 커버리지는 48.0%였고, 전체적으로 약 54.9%가 “sorry가 남지 않음 또는 부분만 달성” 수준으로 잡혔다. 더 중요한 발견은, 커널 승인만으로는 드러나지 않는 불충실 패턴(다중 문장 중 일부만 포함, 교재에 없는 가정 추가, 특정 매개변수로 제한 등)이 반복적으로 관찰되며, 이것들이 LLM 심사에서 의미 불일치로 구체화된다는 점이다. 또한 RepoProver, M2F의 공개 산출물에서도 유사한 감사를 적용해 “컴파일 기반 지표가 품질을 과대평가한다”는 결론과 재현 가능한 감사 방법론을 제시한다.



### Minim: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization (https://arxiv.org/abs/2606.13949)
Comments:
          Accepted at ICML 2026 (43rd International Conference on Machine Learning, Seoul, South Korea). Code available at this https URL

- **Prior Approaches**: 기존 연구는 접근성 트리나 DOM 같은 구조화 관측을 에이전트가 추론에 활용하도록 만들었지만, 대부분은 ‘공유 우선’ 설계로 전체 상태를 원격 추론 서버에 그대로 전송하는 흐름을 따른다. 이로 인해 현재 작업과 무관한 요소(예: 알림, 비밀번호/인증 코드, 다른 세션의 UI 상태)까지 의미 단서와 함께 노출되는 ‘Semantic Over-Privileged Observation’ 문제가 발생한다. 또한 정적 PII 필터링·차등프라이버시·암호 기반 가드레일은 구조적 행동 단서까지 함께 훼손하거나, 이미 전송된 정보로부터의 민감 추론을 막지 못하는 한계가 있다.

- **Core Contribution**: MINIM은 신뢰 가능한 로컬 브로커가 원격으로 관측을 보내기 전에, 작업 맥락에 맞춰 ‘사전 최소화’(pre-disclosure minimization)를 수행하도록 제안한다. Contextual Integrity(맥락적 정합성, CI) 관점에서, 각 UI 요소에 대해 고유 민감도(sensitivity)와 작업 조건 필요도(task-conditioned necessity)를 동시에 학습해 공개 정책을 결정한다. 이를 통해 필요한 것은 유지하고, 민감하지만 불필요한 속성은 추상화하며, 작업과 무관한 콘텐츠는 제거하는 3단(Keep/Abstract/Remove) 공개 전략을 구현한다.

- **Technical Challenges**: 핵심 난제는 ‘어떤 요소가 민감하냐’와 ‘현재 작업에 꼭 필요하냐’가 작업에 따라 바뀐다는 점이다. MINIM은 요소별로 두 점수(s, n)를 예측하는 이중 스코어링을 두고, CI를 만족시키기 위한 목적함수를 설계해 필요도 예측 오류가 특히 위험 콘텐츠에서 더 크게 벌점되도록 했다. 그 결과 고위험·저필요(유출이 치명적이지만 지금은 필요 없는) 구간을 공격적으로 가지치기하면서도, 행동에 필요한 구조적 단서와 의미 맥락은 유지할 수 있게 한다.

- **Empirical Impact**: WebArena에서 생성한 접근성 트리 기반 실데이터(쇼핑·레딧·지메일, 5,403개 트리-작업 변형)로 실험한 결과, MINIM은 작업-무관 민감 유출을 크게 줄이면서도 작업-핵심 정보와 대화/행동 가능 요소를 잘 보존했다. 보고된 지표에서 인터랙티브 핵심 유지(TCNP-I)는 0.9931, 작업 핵심 보존(TCNP)은 0.9491을 유지하면서, 민감 유출(TISL)은 전체 관측 대비 약 10.1% 수준으로 낮아졌다. 또한 일반 지시형 LLM을 스코어러로 프롬프트하는 비교에서는, 모델 규모를 키워도 작업-조건 최소화를 일관되게 강제하지 못해 유출이 더 높게 나타나 MINIM의 정책 내재화 효과가 확인됐다.



### Adversarial Concept Search: Predicting Compositional Errors From Feature Geometry (https://arxiv.org/abs/2606.13934)
- **Prior Approaches**: 기존에는 사람도 어렵게 느끼도록 문제를 설계하거나, 대규모 벤치마크를 정교하게 큐레이션해 ‘엣지 케이스’를 포착하려는 접근이 많았습니다. 또 일부 오류 예측 연구는 active learning처럼 특정 입력을 모델에 통과시켜 신호를 얻는 방식이어서, 입력 생성·평가 비용이 큰 언어 모델링 같은 설정에서는 제약이 큽니다. 문제는 “어떤 개념 조합이 실패하는지”를 입력 평가 없이 선제적으로 맞히기 어렵다는 점입니다.

- **Core Contribution**: 이 논문은 입력을 실제로 모델에 넣어보지 않고도 실패 가능성이 높은 ‘개념 조합 시나리오’를 찾는 Adversarial Concept Search(ACS)를 제안합니다. 핵심은 모델의 표상(표현) 기하를 이용해 조합적 일반화 실패(compositional failure)를 예측한다는 점이며, 특히 원자 개념들의 표현 방향이 만들어내는 간섭이 실패를 유발한다고 봅니다. 결과적으로, 모델의 특정 composed 입력을 평가하지 않고도 실패 모드 우선순위를 매길 수 있는 스케일러블 기반을 제공합니다.

- **Technical Challenges**: 기하 기반 예측을 위해서는 “원자 개념이 모델 내부에서 어떤 방향(피처 인코딩)”으로 표현되는지와, 조합 시 간섭(interference)이 얼마나 큰지(=Compositional Interference, CI)를 계산해야 했습니다. 논문은 Sparse Autoencoders 같은 복잡한 분해 없이, 원자 개념을 나타내는 residual 표현들을 모아 평균/다중 방향으로 개념 표현을 근사하고, 잔여 표현에서 프롬프트·작업군 같은 지배적 배경 클러스터를 mean-centering으로 제거해 각도 추정 왜곡을 줄였습니다. 이후 원자 개념 표현들 사이의 각도/유사성이 간섭을 키우고 성능 저하로 이어진다는 이론적 가설을 CI로 수치화해 실패를 랭킹하도록 설계했습니다.

- **Empirical Impact**: 먼저 합성 벤치마크 SCAN에서, 학습 커버리지와 모델 크기를 바꿔도 CI가 높을수록 조합 정확도가 떨어지는 경향이 일관되게 나타났고, 실패 예측 성능은 PR-AUC 기준으로 무작위·기준선보다 우수했습니다. 이어 Llama-3.2-3B를 대상으로 다중 홉 추론과 다국어 사실 회상에서도, 원자 단계(단일 홉)와 언어 하위공간 사이의 간섭 기하가 실패를 예측했으며 composed 입력을 직접 평가하지 않고도 어려운 예시를 식별할 수 있었습니다. 이는 실제 배포 환경에서 고위험 예시를 선별하고 타깃 스트레스 테스트·능동 학습을 효율화하는 실용적 초석을 제공한다는 의미가 있습니다.



### Sorries Are Not the Hard Part: An Expert-Review Case Study of a Semi-Autonomous Formalization (https://arxiv.org/abs/2606.13925)
- **Prior Approaches**: 기존 자동정형화 평가는 주로 ‘증명 완성’에 초점(컴파일 통과 여부, sorry 제거 여부)으로 이뤄졌습니다. 그러나 정형화는 단순히 커널을 통과하는 증명 산출물이 아니라, 향후 재사용을 좌우하는 라이브러리(정의/정리 일반성/API/네임스페이스)로 이어져야 합니다. 이 논문은 Grothendieck의 vanishing theorem 정형화 사례로, completion 성적표가 재사용 가능성을 충분히 반영하지 못한다고 지적합니다.

- **Core Contribution**: 이 논문은 Lean에서 Grothendieck의 vanishing theorem을 반자동으로 정형화한 뒤, 같은 전문가 리뷰를 거친 ‘상태 A(초기)’와 ‘상태 B(리팩터링/압축 후)’를 비교해 라이브러리 품질 격차의 원인을 분석합니다. 핵심 메시지는 “sorries를 닫았는가”가 아니라 “전문가가 손보고 나서도 재사용 가능한 형태로 남는가”를 평가 기준으로 삼아야 한다는 점입니다. 특히 에이전트는 국소 목표(기계적으로 확인 가능한 피드백)는 잘 맞추지만, 전역적으로 좋은 정의를 고르거나 API를 설계하는 능력은 약하다고 보고합니다.

- **Technical Challenges**: 해결해야 할 기술적 난제는 정의 선택과 API 설계처럼 ‘설명 가능하지만 측정이 어려운’ 설계 문제였습니다. 예를 들어 mathlib에 정의는 있으나 API가 없는 sheaf cohomology(Sheaf.H)에서, 상태 A는 정의를 계속 펼치는 방식으로 진행해 유지보수가 어려웠고, 상태 B에서는 전용 cohomology API 파일을 만들었지만 그 API 자체는 여전히 과도한 특화성과 큰 인터페이스로 인해 사용자 관점의 품질이 떨어졌습니다. 또한 정의 네이밍이 sheafH-filtered colimit 등과 무관하게 보이는 중복/오명(definition drift)이나, Equiv 래핑을 부적절하게 def로 고정해 인스턴스의 정의적(equal definitional) 일치성을 깨는 문제도 반복되며, 이러한 설계 오류는 단순 컴파일 게이트로 잘 드러나지 않는다고 결론냅니다.

- **Empirical Impact**: 실험은 two-stage로 진행돼, 초기에는 커널 통과가 가능했지만 전문가 리뷰에서 정의/일반성/파일 구성/API에서 ‘심각한 문제’가 다수 지적됩니다. 리뷰 기반 리팩터링 후 상태 B는 컴파일 품질과 증명 구조가 개선되고, 특히 mathlib API를 활용한 부분은 더 깔끔해졌습니다. 다만 정의와 인터페이스의 전역 설계 능력은 크게 향상되지 않아, 향후 자동정형화 시스템의 평가는 completion이 아닌 ‘전문가 리뷰 통과 후 라이브러리로 살아남는지’로 옮겨가야 한다는 실증적 근거를 제공합니다. 또한 데이터셋과 로그 공개로 실패 모드(루프 드리프트, 전역 판단 부재 등)를 후속 연구자가 재현·분석할 수 있게 했다는 점에서 분야에 방법론적 임팩트가 큽니다.



### A Multi-Agent AI System for Automated High School Transcript Processing: Collaborative Document Analysis at Sca (https://arxiv.org/abs/2606.13916)
- **Prior Approaches**: 기존 문서 처리 방식은 템플릿 기반(OCR 포함)처럼 표준 양식에는 강하지만, 고교마다 다른 성적표 형식·평가체계·레이아웃에서는 성능이 급격히 떨어지는 한계가 있었다. LLM·비전모델을 쓰는 최신 접근도 교육 문서에 특화된 검증 장치가 부족해 GPA처럼 핵심 지표와의 의미 일치가 흔들리기 쉽다. 교육 분야 연구·상용 자동화는 특정 학교/특정 과제에 국한되는 경우가 많아 ‘형식 다양성’ 전반을 안정적으로 커버하기 어렵다.

- **Core Contribution**: 이 논문은 성적표의 형식 다양성을 다루기 위해, 서로 다른 능력을 가진 다중 에이전트를 오케스트레이션 에이전트가 협업·조율하도록 설계한다. 구성은 형식 파싱을 담당하는 Pattern Recognition Agent, 문맥 의미를 해석하는 Semantic Analysis Agent, 시각 레이아웃을 분석하는 Vision Intelligence Agent이며, Orchestration Agent가 결과를 통합하고 충돌을 해결한다. 핵심 아이디어로 GPA 추출 성공을 ‘조정(quality control) 신호’로 사용해 협업의 신뢰도를 높이고 정보 누락을 줄인다.

- **Technical Challenges**: 가장 큰 기술 과제는 에이전트가 서로 다른 방식으로 값을 추출할 때 발생하는 충돌을 단순 합의가 아닌 도메인 신뢰도로 판정하는 것이었다. 저자들은 GPA가 모든 성적표에 존재하며 의사결정에 중요하다는 점을 활용해, 어떤 에이전트 조합이 충분히 성공했는지(추가 호출이 필요한지)를 GPA 기반 신호로 판단하도록 오케스트레이션을 구성했다. 또한 네트워크/외부 모델 실패, 피크 시간 처리량, 감사 추적을 위해 메시지 버스·재시도·로드 밸런싱·설명 로그(에이전트별 참여 및 결정 근거)를 함께 구현했다.

- **Empirical Impact**: 미국 13개 주의 실제 고교 성적표 40장을 평가한 결과, 모든 문서에서 협업 처리가 완료(100%)됐고 전문가 수기 검토 대비 GPA·과목 등 추출 정확도는 96.7%를 달성했다. 문서 1장당 평균 처리 시간은 약 45초로 운영 가능한 수준이며, API 비용도 장당 0.15달러 수준으로 제시됐다. 성능은 단일 에이전트 대비 협업이 필요한 구간에서 특히 개선됐고, GPA 기반 신호를 통해 대부분의 문서가 소수 에이전트로 해결되도록 점진적 참여가 이뤄져 실사용 확장성의 근거를 제공한다.



### Capability Minimization as a Safety Primitive: Risk-Aware Causal Gating for Least-Privilege LLM Agents (https://arxiv.org/abs/2606.13884)
- **Prior Approaches**: 기존 도구 노출(노출 메뉴) 연구는 주로 관련성·효율·검색/가지치기 크기 최적화에 초점을 맞췄고, CMTF처럼 인과적 필요조건(다음 목표로 가는 원인 경로)에 따라 도구를 노출했다. 하지만 이런 방식은 “읽기 전용 도구”와 “삭제/송금 같은 고위험 도구”를 같은 기준으로 취급해, 공격 표면(standing authority)이 과도해질 수 있다. 또한 대부분의 보안은 가드레일/검증처럼 ‘호출을 시도한 뒤 허용 여부를 판단’하는 데 머물러, 애초에 호출 공간에서 위험 도구를 제거하는 구조적 방어가 약하다.

- **Core Contribution**: RACG(Risk-Aware Causal Gating)는 도구 노출을 인과적 충분성뿐 아니라 위험도와 인증(authorization) 상태까지 결합한 “안전한 노출 게이트”로 확장한다. 구체적으로, 고위험 도구는 (1) 목표로 가는 최소 인과 경로 위에 있고 (2) 상태에 존재하는 신뢰 가능한 인증 변수까지 충족될 때만 에이전트의 실행 가능 도구 집합에 들어간다. 즉, 예측의 자신감이 아니라 ‘반사실(counterfactual) 위험’에 기반해 act/보류/기권을 결정하는 프레임을 도구 노출에 적용해 무단 고위험 호출을 구조적으로 차단한다.

- **Technical Challenges**: 핵심 난제는 “게이트 기준을 신뢰 가능하게 만들기”로, 단순 확률추정이나 모델의 confidence에 의존하면 안전 제약을 만족한다고 보장하기 어렵다. 논문은 분포에 무관한(distribution-free) 형태의 고위험 조건에서 ‘행동할 확률’에 대한 상계를 유도해, 사용자 안전 제약에 맞는 운영 임계값(threshold)으로 변환한다. 또 분포 변화가 생기면 예측된 결과와 실제 결과의 불일치를 모니터링해 게이트를 더 엄격하게 조정함으로써, 인과 가정 위반 가능성을 반영한다.

- **Empirical Impact**: RiskGate에서 RACG는 인가되지 않은 고위험 도구 노출과 공격 성공(주입된 지시로 표적 고위험 도구 호출)을 크게 줄이면서, 인가가 필요한 작업의 완료율은 대부분 유지하는 것으로 보고된다. 또한 confidence 기반 또는 selective-prediction 기준선 대비, 같은 기권률(abatement/abstention rate) 조건에서도 더 낮은 고비용 오류를 달성한다. 이는 고위험을 ‘예측 불확실성’과 분리해 인과적 위험으로 제어할 때, 안전성과 투명성을 함께 높일 수 있음을 실증적으로 뒷받침한다.



### Hyperdimensional computing for structured querying on tabular data embeddings (https://arxiv.org/abs/2606.13871)
Comments:
          15 pages with appendices. 8 figures. Under review

- **Prior Approaches**: 기존의 표(tabular) 임베딩은 행·열·테이블을 벡터로 만든 뒤 근접 이웃 탐색(최근접 검색)으로 후보를 고르는 방식이 주류였습니다. 그러나 이때의 유사도 점수는 의미가 잘 해석되지 않아, “가장 덜 다른 항목”을 반환한 것인지 “진짜 정답이 없는 상태(제로 매치)”인지 구분하기 어렵습니다. 그 결과 현업 배포에서 쓸 만한 기준선 임계값(threshold)을 세우기 힘들고, 특히 제로 매치 탐지에 취약해집니다.

- **Core Contribution**: 이 논문은 HDC(HyperDimensional Computing), 그중에서도 HRR(Holographic Reduced Representations) 모델을 표 행 임베딩의 프레임으로 제안합니다. 임베딩 공간에서 구조화된 select-project 질의를 직접 답할 수 있다는 관점에 맞춰, 일치/불일치 검색 조건에 대해 해석 가능한 기대 유사도 값을 이론적으로 도출합니다. 이를 바탕으로 정답 매치와 비매치를 가르는 “원칙적인” 검색 임계값을 설정할 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 난점은 HRR 임베딩에서 코사인 유사도 같은 점수가 실제로 어떤 값의 의미를 갖는지(특히 일치/불일치 술어에서의 분포) 확인하는 것이었습니다. 논문은 HRR의 결합(binding)·번들링(bundling) 같은 대수적 연산 성질을 활용해, 차원(d)이 커질수록 기대 유사도가 해석 가능한 형태로 수렴함을 보이고 equality/non-equality 술어 각각에 대한 닫힌형(closed-form) 임계값을 유도합니다. 이렇게 얻은 임계값으로 제로 매치도 안정적으로 판별하도록 설계합니다.

- **Empirical Impact**: 실험에서는 EmbDI(그래프 기반 기준선)와 비교해, 다양한 테이블 크기와 술어 길이 조건에서 행 검색 성능을 평가했습니다. 결과적으로 HDC는 모든 구성에서 EmbDI와 비슷하거나 더 좋은 row retrieval 성능을 보였고, 특히 비등호(non-equality) 술어에서 더 견고했습니다. 또한 충분한 차원에서는 속성 투영(attribute projection) 정확도를 완벽에 가깝게 끌어올리며, 무엇보다도 임계값을 통해 제로 매치 예측을 신뢰성 있게 수행할 수 있다는 점을 실증적으로 보여줍니다.



### Poker Arena: Multi-Axis Profiling of Strategic Reasoning and Memory in LLMs (https://arxiv.org/abs/2606.13815)
Comments:
          33 pages, ICML Workshop

- **Prior Approaches**: 기존 LLM 게임플레이 벤치마크는 다수의 전략 역량을 단일 스칼라 성과(대개 승률/점수)로 뭉개, 모델의 ‘역량 구조’가 어떻게 다른지 파악하기 어려웠습니다. 또한 정형 게임의 방법론(Counterfactual Regret Minimization, self-play 기반 탐색 등)은 학습 시점에 강하게 의존해 추론 고정(frozen-weight) LLM 평가에는 그대로 옮기기 어렵습니다. 메모리 아키텍처 연구도 주로 단일 에이전트/협력 환경에 치우쳐, 경쟁적인 다중 에이전트 상황에서 지속 메모리가 어떻게 작동하는지는 덜 규명됐습니다.

- **Core Contribution**: 이 논문은 no-limit Texas Hold’em 토너먼트 환경 ‘Poker Arena’를 제안해, 전략적 추론 역량을 9개 축(베팅 사이징, 블러핑, 상대 읽기, 침착함, 적응성, 예측, 전략적 믹싱, 사실 정확도, 포지셔닝 인식)으로 분해해 평가합니다. 동시에 3계층 메모리(핸드 내, 세션 내, 세션 간)를 설계해 지속성이 성과에 미치는 영향을 통제된 방식으로 측정합니다. 그 결과 단일 스칼라 리더보드가 모델을 잘못 순위화할 수 있음을, ‘축별 역량 구조’ 관점에서 정리합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 전략 역량을 액션/추론 텍스트에서 측정 가능한 형태로 쪼개고, (2) LLM 추론 고정 상태에서도 메모리의 시간 스케일 효과(세션 간 지속)를 통제해 비교하는 것입니다. 이를 위해 액션 로그 기반의 결정론적 지표와, 이유 텍스트에 의존하는 축(예: 의도적 블러프, 구체적 상대 읽기)은 LLM 판정(다중 패널/편향 완화 포함)으로 점수화했습니다. 또한 베팅 규칙(사이드팟/타임아웃/쇼다운)과 컨텍스트 재구성을 엄밀히 처리하면서, 메모리 계층별로 아블레이션을 수행해 비교 가능성을 확보했습니다.

- **Empirical Impact**: 7개 프런티어 LLM을 50개 세션(각 세션 1,000핸드 규모)에서 비교한 결과, Claude가 칩 누적에서는 1위를 차지했지만 9축 평균 점수에서는 5위에 그쳐 스칼라 리더보드의 오분류가 확인됐습니다. 또한 지속 메모리는 모델마다 효과의 부호가 달라(GPT는 이득, Kimi는 손해, Claude는 대체로 불변) ‘메모리 유무’가 아니라 ‘모델-메모리 인터페이스’가 성과를 좌우함을 보여줍니다. 결론적으로 Poker Arena는 다축 평가가 역량 구조를 드러내며, 기존 단일 점수 순위가 감추는 차이를 체계적으로 복원할 수 있음을 실증합니다.



### MA-ProofBench: A Two-Tiered Evaluation of LLMs for Theorem Proving in Mathematical Analysis (https://arxiv.org/abs/2606.13782)
Comments:
          19 pages, 4 figures, 4 tables

- **Prior Approaches**: 기존의 형식 수학 벤치마크는 주로 대수, 기초 정수론, 조합론처럼 비교적 형식화가 쉬운 주제에 치우쳐 있었고, 연속성·극한·위상 구조를 함께 다루는 해석(수학적 해석) 영역은 상대적으로 빈약했습니다. 또한 일부 데이터셋은 의미적 부정확성이나 원문 수학과의 불일치 같은 품질 문제가 보고돼, 평가 신뢰도를 깎는 요인이 되곤 했습니다. 일부 경쟁형 문제 중심 벤치마크는 이미 성능이 포화되는 양상도 보여 변별력이 감소했습니다.

- **Core Contribution**: 본 논문은 수학적 해석 전용 최초의 형식 정리증명 벤치마크인 MA-ProofBench를 제안합니다. 전체 200개 정리를 6개 핵심 주제·27개 세부 범주로 구성하고, 학부 수준(Level I)과 박사 예비과정 수준(Level II)으로 난이도를 나눠 해석 심도의 차이를 정밀하게 평가합니다. 더불어 원문 수학과의 충실성을 보장하기 위해 사람-LLM 협업 형식화 파이프라인과 독립 전문가 검토를 결합했습니다.

- **Technical Challenges**: 해석 문제를 Lean 4로 옮기는 과정에서 핵심 난관은 Mathlib 지식의 정확한 호출과, 긴 증명 흐름에서의 미완료·스킵 없이 모든 서브골을 충족하는 “증명 완결”입니다. 논문은 초안 형식화(Proof를 sorry로 두고)→컴파일러 피드백 기반 리파인→3인 독립 역번역 검토→난이도 채점의 4단계 워크플로로 문장 의미 및 구문 정확도를 맞췄습니다. 그 결과 함수·적분·공간 구조·암묵 전제를 명시하는 통일 규칙을 적용해 형식화 중 의미 손실을 줄였습니다.

- **Empirical Impact**: 실험에서 MA-ProofBench의 현재 모델 성능은 전반적으로 매우 낮았고, 최고 성능인 GPT-5.5도 Level I Pass@8 16%, Level II Pass@8 5%에 그쳤습니다. 실패의 주요 원인은 Mathlib hallucination과 incomplete proofs로 분석됐으며, 자연어 버전 평가에서는 모델이 비형식 추론에서는 맞추지만 Lean 4로 컴파일 가능한 정리로 ‘번역’하지 못하는 격차가 뚜렷하게 드러났습니다. 이는 향후 시스템이 Mathlib 정합성, 해석 분야의 장거리 증명 계획·보조정리 구성, 그리고 비형식→형식 변환 신뢰도를 함께 개선해야 함을 보여주는 기준점으로 의미가 큽니다.



### AI Receptivity or AI Adoption Breadth? A Tool-Specific Reanalysis of the Lower-Literacy/Higher-Usage Link (https://arxiv.org/abs/2606.13734)
Comments:
          11 pages, 2 tables, 1 figure

- **Prior Approaches**: 기존 마케팅 연구는 알고리즘에 대한 호감/회피가 개인 특성·과업 맥락에 따라 달라진다는 점을 보여 왔지만, ‘AI 수용성’의 개인 수준 선행요인은 상대적으로 덜 탐구돼 왔습니다. 한편 Tully 등(2025)은 AI 리터러시(객관식 AI 지식)가 낮을수록 AI 사용 빈도가 높다는 ‘역설적’ 관계를 여러 연구에서 제시했는데, 특히 Study 3은 실제 AI 도구의 과거 사용을 종속변수로 써 보수적 검증으로 여겨졌습니다.

- **Core Contribution**: 본 논문은 Tully 등(2025)의 Study 3 공개 데이터를 재분석해, 집계 지표가 ‘일반적 수용성’ 해석으로 바로 연결되지 않을 수 있음을 보여줍니다. 핵심은 5개 도구 범주를 평균내어 한 지표로 만들면, 도구별로 다른 행동 과정(시도 여부 vs 사용 강도)이 섞이면서 해석이 왜곡된다는 점을 도구 유형별·서열형 모델로 분해해 재정의한 것입니다.

- **Technical Challenges**: 기존 분석은 5점 리커트형 사용 빈도를 평균한 연속 지수로 회귀해, 서열 자료의 모형 가정 위반과 범주 이질성에 따른 효과 압축 위험이 있습니다. 저자들은 OLS, 이항 로짓(빈번/도입 기준), 순서 로짓(비례확률), 다항 로짓으로 재현하고, 텍스트 AI(글쓰기 도우미)와 비텍스트 AI(이미지·생산성·웹사이트·헬스 앱)를 분리해 채택(비사용→사용 전환)과 사용 강도 구간을 분해했습니다.

- **Empirical Impact**: 재분석 결과, 전체 집계에서도 ‘리터러시가 낮을수록 사용이 높다’는 방향성은 재현됐지만 의미는 달라졌습니다. 인구통계 보정 1차 모형에서 텍스트 AI는 유의하지 않은 반면(ordered-logit β=-0.090, p=0.387), 비텍스트 AI는 강하게 유의하며(β=-0.377, p<0.001) 효과가 주로 ‘한 번이라도 써 본 적 있음’ 같은 도입 경계에 집중됐습니다(비텍스트 도구 ever-use의 오즈비≈0.68). 따라서 Study 3이 실제 사용 데이터를 기반으로 보여주는 결론은 ‘전반적 AI 수용성’보다 ‘낮은 보급도의 비텍스트 AI 제품군을 더 넓게 시도한다’는 더 좁은 패턴으로 귀결되며, 기업의 타깃팅 주장도 도구 유형별로 재조정될 필요가 있음을 시사합니다.



### When Sample Selection Bias Precipitates Model Collaps (https://arxiv.org/abs/2606.13732)
Comments:
          Accepted at the 43rd International Conference on Machine Learning (ICML 2026)

- **Prior Approaches**: 재귀적 학습에서 합성 데이터만 반복 사용(Replace)하면 통계적 충실도와 분산이 급격히 줄어드는 model collapse가 나타난다는 분석이 축적되어 왔다. 이를 완화하려고 초기 실데이터를 누적하거나(Accumulate) 제한된 크기로 샘플링하며(Accumulate-Subsample), 합성 데이터에 대한 데이터 selection이 중요하다고 보는 데에는 비교적 공감대가 있다. 다만 기존 논의는 선택/필터가 “품질을 높인다”는 가정에 기대는 경우가 많았고, 검증 기준(reference) 자체가 지역·편향적일 때 selection이 어떤 구조적 부작용을 만드는지 충분히 다루지 못했다.

- **Core Contribution**: 이 논문은 저자원 검증(low-resource verification) 환경, 즉 각 verifier가 전체 매니폴드의 작은 조각만 보고 편향된 로컬 기준으로 합성을 걸러낼 때 selection이 오히려 편향 필터로 작동함을 보인다. 그 결과 평균은 로컬 목표에 맞춰질 수 있지만, 전역적으로 중요한 꼬리 모드(tail modes)는 계속 제거되어 diversity가 붕괴한다. 저자들은 siloed selection이 collapse를 가속하며 power-law 형태의 다양성 감소를 유도한다고 이론적으로 정리한다.

- **Technical Challenges**: 핵심 난제는 “로컬 reference가 편향 selection을 통해 전역 분포 꼬리를 어떻게 소거하는가”를 수학적으로 추적하는 것이다. 저자들은 selection을 국소적으로 오목한 점수함수와 선택 영역으로 모델링하고, 선택 비율(필터링 예산)이 만드는 truncated 분포의 모멘트 변화를 따라가며 variance 붕괴의 붕괴 속도(rate)를 도출한다. 또한 진정한 분포와의 차이를 Wasserstein 기하로 연결해, 로컬에서 잘 맞춘 모델이라도 전역 기준에서는 Wasserstein discrepancy가 일반화 비용으로 이어짐을 보여준다.

- **Empirical Impact**: 실험에서는 Gaussian 모델링 프레임워크를 통해 로컬 기준 top-α 샘플링이 Replace/Accumulate 모두에서 다양성 붕괴를 일으키되, Accumulate에서는 정리된 power-law 붕괴 동역학이 관측됨을 확인한다. 특히 skewed 분포에서는 로컬-reference selection이 실패하는 반면, 여러 silo에서 원데이터를 교환하지 않고 Wasserstein proxy reference(geodesic interpolation 또는 Wasserstein barycenter)를 만들어 협력적으로 검증하면 diversity 저하가 완화된다. 즉, 재귀적 합성 데이터 파이프라인을 운영할 때 “검증 기준의 범위가 조각난 현실”을 전제로, 단일 로컬 reference에 의존하는 선택을 경계해야 한다는 실무적 메시지를 제공한다.



### TwinBI: An Agentic Digital Twin for Efficient Augmented Interactions with Business Intelligence Dashboards (https://arxiv.org/abs/2606.13731)
- **Prior Approaches**: 기존 NL 기반 BI 보조는 NL→SQL이나 차트 생성 등 ‘쿼리 작성’ 중심으로 발전했지만, 대시보드에서 사용자가 적용한 필터·계층·지표 같은 분석 상태를 채팅과 일관되게 유지하긴 어렵습니다. 그 결과 답변은 그럴듯해도 실제 대시보드 상태와 의미(집계 단위, 스코프)가 어긋나거나 다단계 탐색에서 상태가 망가지는 문제가 반복됩니다. 한편 대시보드는 강력한 시각적 조작이 있지만, 상태를 다시 복원해 다음 대화 턴에 반영하는 연결 계층이 부족해 사용자에게 재설정 부담이 생깁니다.

- **Core Contribution**: TwinBI는 LLM 에이전트 쌍과 실행 가능한 BI 대시보드 상태의 ‘디지털 트윈’을 결합해, 대화와 대시보드 조작을 하나의 공유 분석 상태(shared analytical state)로 동기화합니다. 이 상태는 통합 상호작용 로그를 기반으로 재구성되며, 스키마·계층·지표 매핑과 차트 컨텍스트까지 의미 있게 정렬해 에이전트가 잘못된 상태에서 답을 만들 여지를 줄입니다. 또한 사용자가 검증할 수 있도록 SQL·스키마 뷰·로그·/insights 같은 상태 기반 산출물을 노출합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘대화로 들어온 요청’과 ‘대시보드에서 누적된 시각적 조작’이 서로 다른 맥락을 가질 때, 집계 단위·필터 스코프·계층 레벨까지 포함한 정확한 현재 상태를 복원하는 것이었습니다. TwinBI는 대시보드 이벤트를 구조화해 통합 로그에 남기고, 의미 계층(측정값·차원·계층·조인 경로)과 함께 실행 가능한 쿼리 사양으로 상태를 재구성함으로써 에이전트의 의미적 기준선을 맞춥니다. 더불어 /insights는 ‘현재 보이는 증거’에만 기반하도록 출력 범위를 제한해, 상태 밖 추론으로 인한 불일치를 줄이도록 설계했습니다.

- **Empirical Impact**: 통제된 A/B 벤치마크에서 TwinBI는 정확일치(exact-match)를 43.3%에서 63.3%로, 부분점수(partial-credit)를 48.3%에서 70.8%로 끌어올렸고, 타임아웃도 40.0%에서 10.0%으로 크게 감소시켰습니다. 유저빌리티 연구에서도 대시보드-채팅 통합 워크플로가 과업 정확도는 유지하면서 인지·작업 부담을 중간 수준으로 유지했으며, 상태 인지 기반 상호작용 메커니즘에 대한 평가도 긍정적이었습니다. 즉 TwinBI는 에이전트 수준의 분석 신뢰성과 사용자 관점의 상태 기반 지원을 동시에 강화하는 실증 결과를 보였다는 점에서 의미가 있습니다.



### YeasierAgent: Agentic Social Sandbox as a Canvas for Intent-Driven Creation of Platform-Agnostic Symbiotic Agent-Native Applications (https://arxiv.org/abs/2606.13722)
- **Prior Approaches**: 기존 AI 앱 생성은 자연어를 코드나 화면 일부로 바꿔주는 방식이 주류였지만, 결과물이 독립된 스크립트·페이지·컴포넌트로 끝나 재사용성이 떨어지는 한계가 있었다. 또한 멀티 에이전트나 에이전트 샌드박스 연구도 시뮬레이션/관찰 중심에 머무는 경우가 많아, 사용자 관점에서 ‘소프트웨어 인터페이스 자체’로 노출되긴 어려웠다. 요컨대 사용자 정체성, 앱 상태, 사회적 맥락, 기기 표현이 하나의 연속 경험으로 묶이지 못했다.

- **Core Contribution**: YeasierAgent는 앱을 기기별 UI로 정의하기보다 ‘사용자-에이전트-세계’가 함께 참여하는 협업 공간으로 재정의한다. 특히 플랫폼 비의존적 상호작용 단위(에이전트, 장면, 대화)를 써서 대화형 앱을 웹·모바일 등 여러 단말에서 빠르게 구성하고 동일한 경험 흐름을 유지하는 구조를 제안한다. 동시에 감정적 동반(Companion)과 실용적 도구 실행을 하나의 경험 샌드박스 안에서 통합해 “Symbiotic Agent-Native Applications” 범주를 정리한다.

- **Technical Challenges**: 핵심은 (1) LLM 기반 생성이 만드는 앱이 단말과 무관하게 동일한 ‘장면·대화·선택’ 단위로 동작해야 하고, (2) 사용자의 디지털 트윈 에이전트를 지속적으로 정교화해 감정적 반응과 작업 수행을 동시에 만족해야 한다는 점이다. 논문은 벡터 저장 기반 장기 메모리와 Big Five(빅 파이브) 성향 파라미터를 프롬프트/행동 제어에 동적으로 인코딩해 디지털 트윈을 만들며, 장면-공간 매핑을 통해 작업 진행을 직관적으로 관찰 가능하게 설계한다. 또한 세계(World)를 이벤트 기반 관찰 캔버스로 두고 생성 앱(Creation Apps)이 그 위에서 규칙·목표·역할·사회적 결과를 실행하도록 분리함으로써 플랫폼 연속성을 확보한다.

- **Empirical Impact**: 저자는 라이브 플랫폼 배포를 통해 에이전트-공간 상호작용이 도구 실행, 게임형 추리, 인터랙티브 드라마 같은 서로 다른 앱 경험을 공통 원시 요소로 묶을 수 있음을 정성적으로 보인다. OpenClaw 호환 로컬 워크플로처럼 기술 로그를 읽지 않아도 에이전트의 위치·행동·표정으로 진행을 이해하게 하는 사례와, 다중 에이전트의 숨은 정보·선택 기반 게임, 사용 개입에 따라 변주되는 반-스크립트 서사 사례가 제시된다. 다만 멀티 에이전트 공간의 실시간 렌더링 비용과 LLM/네트워크 성능 의존성은 구현 제약으로 남아 있어, 향후 시각 최적화와 안정적 실행 개선이 과제로 제시된다. 



### Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP (https://arxiv.org/abs/2606.13720)
- **Prior Approaches**: 안전 미세조정(chat safety fine-tuned) 모델의 ‘거절(refusal)’은 유효한 동작 방향을Activation space에서 찾아 개입하는 방식이 주류였다. 기존에는 단일 선형 방향을 Mean-of-Differences(차분 평균; DiM)로 추출해, 해당 방향을 더하면 거절이 유도되고 빼면 거절이 억제되는 식의 개입이 비교적 간단하면서도 강력하다고 알려져 있다. 다만 이런 방법은 보통 단일 방향에 의존해, 더 풍부한 기하학적 조작(예: 제거 vs 반대 개념으로의 이동) 가능성이 충분히 비교되지 않았다.

- **Core Contribution**: 이 논문은 DiM 기반 ‘방향 개입’과 Iterative Nullspace Projection(Iterative Nullspace Projection; INLP)에서 파생된 ‘개념 제거/반대 개념 반영’ 개입을 5개 공개 가중치 채팅 모델에서 정면 비교한다. 특히 INLP는 (1) nullspace projection(개념 삭제에 가까운 연산)과 (2) counterfactual flipping(반대 개념으로의 반영)이라는 두 계열로 나뉘며, nullspace의 크기 k와 연산 파라미터 α로 효과-성능 트레이드오프를 조절할 수 있음을 보여준다. 결과적으로 INLP counterfactual flipping은 DiM directional ablation 수준으로 거절 억제 성능을 내는 반면, nullspace projection은 전반적으로 더 약했다.

- **Technical Challenges**: 핵심 기술적 난제는 (a) refusal에 대응하는 표현을 정확히 추출하고, (b) 그 표현을 같은 기준으로 ‘공정하게’ 개입했는지 비교하는 것이다. 저자들은 DiM과 INLP 모두 대비 학습 데이터의 harmful/harmless 활성 평균 차이를 기반으로 하는 대조적(contrastive) 방향을 구하되, 각 후보 개입은 자신이 추출된 층(l)과 토큰 위치(t)에만 적용해 방법 간 영향력이 과도하게 달라지지 않도록 했다. 또한 INLP에 대해 k(지워지는 부분공간 차원)와 α(삭제/반대 반영 정도)를 조절해 near-기준선(perplexity baseline) 근처에서 거절 억제 효과를 최대화하는 운영점을 찾아, activation 공간의 서로 다른 기하학적 변형이 실제 거동 차이를 만든다는 점도 분석했다.

- **Empirical Impact**: 실험 결과, 다섯 모델 전반에서 INLP counterfactual flipping은 harmful 프롬프트에서 non-refusal(거절하지 않음)을 DiM directional ablation과 비슷하게 억제 수준으로 끌어올렸고, LlamaGuard 2 unsafety 같은 안전 지표에서도 유사한 경향이 관찰됐다. 반면 INLP nullspace projection은 일부 모델에서만 부분적으로 효과가 나타났고 대체로 약했으며, activation geometry에서는 nullspace projection이 두 클러스터 사이의 ‘중간 영역’으로 붕괴시키는 반면 counterfactual flipping은 반대 클러스터 쪽으로 이동시키는 질적 차이가 드러났다. 또한 k를 강한 분류기까지 포함하되 과도한 성능 저하는 피하는 값(예: k0.8)을 쓰면 대부분의 모델에서 기준선 퍼플렉시티에 가깝게 유지하면서도 제어 가능성을 얻을 수 있어, 안전 제어 연구에서 단일 방향 개입의 한계를 넘어서는 ‘튜너블 개입’ 가능성을 시사한다.



### WorkBench Revisited: Workplace Agents Two Years On (https://arxiv.org/abs/2606.13715)
Comments:
          8 pages, 3 figures. Follow-up to arXiv:2405.00823

- **Prior Approaches**: 기존 에이전트 벤치마크는 웹 탐색, 일반 보조, 도구 사용 등 주변 문제를 다루거나, LLM 평가자에 의존해 행동을 점수화하는 방식이 많았다. WorkBench는 사무 환경을 샌드박스로 구현하고, 에이전트가 임의의 경로로 작업하되 최종 상태를 정답과 비교해 ‘행동 자체’의 성패를 직접 평가하는 데 초점이 있다. 다만 2024년 출시 당시에는 최고의 에이전트도 작업의 43%만 완료했고, 포맷/툴 호출 실패 같은 요소가 결과를 크게 흔들었다.

- **Core Contribution**: 이 논문은 2024년 WorkBench를 2026년까지 재실행하며 성능과 안전성을 함께 재측정한다. 특히 구조화된 native tool-calling을 일괄 적용해 비교 기준을 공정하게 만들고, 작업 완료율뿐 아니라 의도치 않은 유해 부작용(예: 잘못된 수신자에게 이메일 발송)과 1회 실행 비용까지 2개 축을 추가해 ‘능력-안전-비용’의 동시 지표를 제시한다. 또한 기존 벤치마크의 채점/정답/프롬프트 불일치 및 툴 엔지니어링 문제를 수정해, 2026 점수와 2024 점수를 직접 비교할 때의 함정을 줄였다.

- **Technical Challenges**: 핵심 기술 난제는 에이전트가 샌드박스에서 임의의 경로로 작업할 때, 툴 호출 포맷 실패와 프롬프트-정답 불일치가 성능을 왜곡하지 않게 하는 것이다. 저자들은 ReAct의 텍스트 기반 툴 파싱 대신 공급자들이 제공하는 structured output의 native tool-calling을 사용해 스키마 준수 실패를 제거했고, 모델 추론이 남긴 기본 오류(예: ‘아직 오버듀이 아닌데 오버듀로 간주’ 같은 조건 판단, 캘린더 검색 결과 5개 제한을 고려한 재쿼리 부족)는 그대로 관찰되는 형태로 정리했다. 아울러 “last NN days” 오프바이원, 잘못된 그라운드 트루스, silent-zero 집계 버그, 줄바꿈 이스케이프 같은 엔지니어링 결함을 재생성/재계산해 결과 재현성을 높였다.

- **Empirical Impact**: 재평가 결과, 작업 완료율은 2024년 최상위 GPT-4(43%)에서 2026년 Claude Opus 4.8(88.8%)로 크게 상승했으며, 의도치 않은 유해 부작용 비율도 26%에서 2.5%로 급감했다. 더 나아가 능력과 안전이 상충하기보다 함께 개선되는 경향이 확인되어, 가장 많이 끝내는 모델이 대체로 가장 적게 ‘의도치 않은 피해’를 낸다는 점이 강조된다. 한편 비용은 모델·공급자 간 2자릿수 수준으로 격차가 커서, 오픈 웨이트가 더 저렴한 구간의 효율을 끌어올렸고(캐시 미적용 상한치 기준), 전체적으로 frontier 모델의 절대 효율 격차와 더불어 ‘방출일 단독’으로는 발전량을 설명하기 어렵다는 시사점을 준다. 



### Hybrid Open-Ended Tri-Evolution Makes Better Deep Researcher (https://arxiv.org/abs/2606.13710)
- **Prior Approaches**: 기존 딥 리서치는 웹 규모 지식을 탐색·통합해 장문 보고서를 만들지만, 학습된 파라미터 기반 역량이 고정된 기준선(기준 학습셋/전략)으로 상한이 정해진다. 에이전트 진화(self-play 등)는 검증 가능한 정답이 있는 과제에서 주로 성과가 입증돼, 명확한 정답이 없는 오픈 엔디드 리서치 과제로의 일반화에 공백이 있었다.

- **Core Contribution**: HOTE(Hybrid Open-ended Tri-Evolution)는 제안자(proposer)·해결자(solver)·판정자(judge)를 함께 진화시키는 프레임워크로, 오픈 엔디드 리서치의 ‘정답 의존 평가’ 문제를 판정 루브릭 기반 보상으로 우회한다. 또한 도구 사용(tool-use)과 무도구(no-tool) 학습을 함께 두 모드 간 상호 이익이 나도록 설계해, 보고서 품질과 참조 생성 역량을 동시에 끌어올린다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 해설/보상 설계가 쉬워 보이지만 reward hacking을 유발할 수 있고, (2) 과제가 오픈 엔디드라 상대 비교가 안정적이어야 하며, (3) 제안자까지 함께 학습하면 샘플 길이와 계산비용이 폭증할 수 있다는 점이다. HOTE는 GRPO 기반 강화학습으로 solver와 proposer를 동시에 업데이트하고, judge가 강점·약점을 반영하는 루브릭을 동적으로 갱신하되 meta rubrics로 제안자 입력을 압축해 학습 효율과 난이도 조절을 동시에 달성한다. 여기에 하이브리드 트레이닝에서 매 단계 절반은 no-tool, 절반은 tool-use로 랜덤 배치해 도구 잡음에 따른 편향과 파라미터 과의존을 함께 완화한다.

- **Empirical Impact**: 세 개의 장문 딥 리서치 벤치마크(HealthBench, ResearchQA, DeepResearchBench)에서 HOTE로 학습한 8B 모델은 오픈 소스 8–32B 고정형 모델과 최신 딥 리서치 학습법 대비 더 높은 성능을 보였다. 특히 세 모듈(제안자·해결자·판정자) 모두의 진화가 없으면 상승 추세가 약해지거나 수렴이 정체되는 등 기여가 검증됐다. 또한 도구 사용 횟수·학습 단계 측면에서 시간 오버헤드를 줄이면서도 지속적인 진화(최소 250시간 이상, wall-clock 기준) 경향을 보였다는 점에서, 오픈 엔디드 연구 에이전트의 실용적 확장 가능성을 시사한다.



### Orchestra-o1: Omnimodal Agent Orchestration (https://arxiv.org/abs/2606.13707)
- **Prior Approaches**: 기존 에이전트 오케스트레이션은 주로 텍스트 중심이거나 시각-언어처럼 제한된 조합에 맞춰져 있어, 텍스트·이미지·오디오·비디오가 함께 얽힌 오므니모달 환경에서의 일반화가 어렵다. 또한 모듈을 나눠도 실행 흐름이 선형적이거나 휴리스틱에 의존해 복잡한 의존성/병렬성을 효율적으로 다루지 못한다. 네이티브 오므니모달 에이전트는 한 모델이 인식·추론·도구사용까지 동시에 담당하려 하지만, 장기 지평 추론과 도구/크로스모달 정교성에서 한계가 나타난다.

- **Core Contribution**: Orchestra-o1은 오므니모달 에이전트를 위한 오케스트레이션 프레임워크로, 입력 양식(modality)을 고려한 작업 분해와 하위 에이전트의 온라인 전문화, 병렬 서브태스크 실행을 한 구조 안에 묶는다. 메인 에이전트가 고수준 결정(위임·완료)을 하고, 인식/행동은 전용 서브에이전트와 통합 도구 생태계가 담당하도록 ‘역할 분리’를 명확히 한다. 아울러 Orchestra-o1의 메인 에이전트 학습을 위해 DA-GRPO(Decision-aligned group relative policy optimization)라는 오프라인 에이전트형 강화학습 레시피를 제안한다.

- **Technical Challenges**: 핵심 난제는(1) 어떤 입력과 도구가 해당 단계에 필요한지 모달리티 인지형으로 결정하고, (2) 결과 간 의존성을 그래프로 표현해 준비된 작업만 병렬로 스케줄링하는 것이다. 논문은 메인 에이전트가 서브태스크별 요구 벡터(텍스트/이미지/오디오/비디오/코드 등)와 도구 요구를 예측해 모델·도구를 매칭하며, 생성된 의존성 그래프를 기반으로 ready 집합에서 배치를 뽑아 병렬 실행한다. 또한 증거(evidence)를 구조화된 메모리에 압축 저장하며, 충분성 점수가 임계치를 넘을 때 종료하도록 하여 컨텍스트 예산 문제도 함께 다룬다. DA-GRPO는 단계별 오케스트레이션 결정(위임·서브에이전트 선택·도구 사용·생성)을 기준 궤적과 정렬하도록 다차원 루브릭 보상으로 오케스트레이션 의사결정을 학습시킨다.

- **Empirical Impact**: OmniGAIA 벤치마크에서 강한 프로프라이어터리 메인 에이전트와 결합했을 때 Orchestra-o1은 기존 2등 대비 10.3% 정확도 향상을 달성한다. 또한 DA-GRPO로 훈련한 Orchestra-o1-8B는 오픈소스 오므니모달 에이전트들에 대해 최신 수준의 성능을 보이며, OmniGAIA에서 최고 정확도를 20.8%에서 30.0%로 끌어올린다. 구조가 병렬화 가능하고 추론/비용 효율 측면에서도 이점이 있어, 오므니모달 에이전트 스웜 설계에 대한 실용적 기준선을 제시했다.



### History of the Muddy Children Puzz (https://arxiv.org/abs/2606.13703)
- **Prior Approaches**: 머디 차일드 퍼즐(Muddy Children Puzzle)은 지식과 무지(knowledge and ignorance)를 다루는 대표적 에피스테믹 논리 퍼즐로 널리 알려져 있지만, “누가 처음 만들었는가”의 출처는 명확하지 않았다. 기존에는 퍼즐의 변형(숫자, 색모자 등)과 풀이 자체는 많이 다뤄졌으나, 논리·문학·퍼즐 출판 흐름을 통해 계보를 체계적으로 추적한 연구는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 지난 2세기 동안의 논리 및 문학 출판물을 관통하며 머디 차일드 퍼즐의 기원을 추적해, 19세기 초의 문학적 게임(예: Rabelais의 Gargantua et Pantagruel 주석판)과의 연관 가능성을 제시한다. 또한 자신을 참조하는(self-reference) 새로운 색모자(hats) 퍼즐을 제안해, 단순 지식 추론을 넘어 자기지시적 구조까지 확장되는 지점을 보여준다.

- **Technical Challenges**: 기여를 위해서는 “퍼즐”로 명시된 형태가 아니라 게임·소설·퍼즐집 속 암시적 규칙을 논리적 의미로 재해석해야 했고, 특히 1830~1930년대에 해당하는 공백 구간을 메우기 위한 문헌 탐색의 어려움이 컸다. 논문은 문자적 유사성(얼굴/표식이 보이지 않는다는 설정 등)과 추론 패턴(상대의 표식을 근거로 자기 상태를 단계적으로 확정)을 함께 대조하고, 출판 시점별로 에피스테믹 논리로의 전환 경로를 정리하는 방식으로 이 난제를 해결한다.

- **Empirical Impact**: 실증적으로는 일본에서 ‘Dirac’s Riddle’로 알려진 20세기 초/중반 경로처럼, 국가·언어·매체를 달리하며 동시다발적으로 재등장한 정황을 문헌 근거와 함께 제시해 퍼즐 계보를 구체화한다. 더 나아가 1980년대 이후 에피스테믹 논리·분산계산·동적 에피스테믹 논리(dynamic epistemic logic) 연구에서 머디 차일드가 사실상 표준 예제로 자리잡는 과정이 설명되며, 이후 퍼즐 공동체의 모자 문제 발전과도 연결될 수 있음을 시사한다.



### UP-NRPA: User Portrait based Nested Rollout Policy Adaptation for Planning with Large Language Models in Goal-oriented Dialogue Systems (https://arxiv.org/abs/2606.13683)
- **Prior Approaches**: 기존 목표 지향 대화 정책 계획은 프롬프트 엔지니어링이나 오프라인 강화학습 기반 MCTS/정책 플래너를 통해 성능을 끌어올려 왔습니다. 다만 오프라인 강화학습은 학습 데이터와 훈련 비용에 의존하고, 새로운 사용자 유형에서의 일반화와 실시간 적응이 약합니다. 또한 기존 사용자 페르소나 반영 기법(TRIP/UDP 등)은 사용자 피드백에 따라 동적으로 전략을 바꾸는 데 한계가 있어, 비협력(협상·설득) 및 목표 불일치 상황에서 성공률이 떨어지는 문제가 남았습니다.

- **Core Contribution**: 이 논문은 User Portrait 기반 Nested Rollout Policy Adaptation(UP-NRPA) 온라인 프레임워크를 제안해, 사용자 특성에 맞춘 대화 전략을 실시간으로 커스터마이즈합니다. 핵심은 Big Five 성격·의사결정 스타일로 사용자 초상(user portrait)을 만들고, 대화 중 실시간 피드백과 연결된 적응 메커니즘으로 정책을 조정한다는 점입니다. 특히 오프라인 강화학습 정책 모델을 학습하지 않고도, 온라인 탐색과 사용자 피드백만으로 전략을 업데이트합니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 다양한 사용자 특성을 반영하되, 목표 지향 행동 시퀀스를 온라인으로 효율적으로 탐색·개선하는 것입니다. 논문은 NRPA-GD의 중첩 롤아웃(Nested Rollout) 아이디어를 확장해, 2단계(레벨 2: 정책 후보 선택, 레벨 1: 다턴 롤아웃 시뮬레이션) 구조에서 보상을 계산하며 최적 시퀀스로 정책 분포를 점진적으로 이동시킵니다. 레벨 1에서 사용자 초상 기반 롤플레이 시뮬레이터가 행동에 대한 피드백을 주고, 보상 신호로 가중치를 업데이트해 오프라인 학습 없이도 적응형 샘플링이 되도록 설계했습니다.

- **Empirical Impact**: 실험에서는 협력/비협력 대화 벤치마크(ESConv, ExTES, CraigslistBargain, P4G)에서 대규모 사용자 시뮬레이터를 구성해 평가했으며, 전반적으로 목표 달성 성능이 크게 개선됐습니다. 특히 협상 관련 지표에서 sale-to-list ratio(SL)가 56.41% 상승했고, 여러 작업에서 성공률(SR)이 1.0000에 도달하는 결과도 보고됩니다. 또한 soft success rate(SSR) 및 인간 평가에서도 UP-NRPA가 TRIP/UDP/NRPA-GD 대비 일관되게 우수했으며, 오프라인 강화학습 없이도 사용자 유형이 달라도 대화 목표를 안정적으로 달성할 수 있음을 보여줍니다.



### A Deep Reinforcement Learning (DRL)-Based Transformer Method for Solving the Open Shop Scheduling Problem (https://arxiv.org/abs/2606.13682)
- **Prior Approaches**: 오픈 샵 스케줄링 문제(OSSP)는 작업 수와 기계 수가 커질수록 계산 난도가 급격히 높아집니다. 정확해법은 규모가 커지면 빠르게 비현실적이 되었고, SPT·LPT·MWKR·EST 같은 고전 규칙이나 메타휴리스틱은 큰 스케일에서 성능을 유지하려면 튜닝 부담이 커질 수 있습니다.

- **Core Contribution**: 이 논문은 인코더-디코더 구조의 멀티헤드 어텐션을 갖춘 Transformer 기반 스케줄링 정책을 제안합니다. 특히 Taillard 벤치마크의 작은 인스턴스(4×4, 5×5, 7×7, 10×10)에서 학습한 뒤, 입력으로는 처리시간 행렬만 사용해 더 큰 문제로도 그대로 적용 가능하다는 점을 핵심 기여로 내세웁니다.

- **Technical Challenges**: 주요 난제는 작은 학습 데이터에서 얻은 스케줄링 결정을 큰 인스턴스에서 그대로 일반화해도 ‘항상 가능한 해(제약 만족)’를 만들 수 있느냐입니다. 연구진은 처리시간 행렬만 넣는 특징-빈약(feature-light) 설정에서도 어텐션으로 작업 간 상호작용을 포착하고, 디코더가 스케줄을 구성해 실행 가능한 해를 산출하도록 설계했습니다.

- **Empirical Impact**: 학습된 정책은 Taillard 테스트에서 makespan이 기준 대비 대체로 15~30% 범위에 들어오며, 40×40~100×100 무작위 인스턴스에서도 재학습 없이 평균 갭 12.89~15.12%를 기록했습니다. EST와는 비슷하거나 근소하게 경쟁력을 유지하는 한편, SPT와 LPT는 크게 앞서며, 고전 규칙 대비 튜닝 부담이 적은 학습 기반 대안으로서의 확장성을 보여줍니다.



### ClinHallu: A Benchmark for Diagnosing Stage-Wise Hallucinations in Medical MLLM Reasoning (https://arxiv.org/abs/2606.14697)
Comments:
          Code and datasets: this https URL

- **Prior Approaches**: 기존 의료 홀루시네이션 벤치마크는 정답/오답 같은 최종 출력 중심으로 평가해, 오류가 추론 과정의 어디에서 발생하는지(시각 인식 실패, 의학 지식 회상 실패, 근거 통합 실패)를 분리해 진단하기 어렵습니다. CARES·Med-HallMark 등 멀티모달 평가가 확장됐더라도 여전히 “결과가 틀렸는지”에 집중하는 경향이 강했습니다. 그 결과 같은 오답이라도 서로 다른 원인 실패가 하나의 판단으로 뭉쳐지는 한계가 있었습니다.

- **Core Contribution**: ClinHallu는 의료 MLLM 추론을 Visual Recognition(시각 인식), Knowledge Recall(지식 회상), Reasoning Integration(근거 통합) 3단계로 분해해, 단계별 홀루시네이션 원인을 추적하는 벤치마크를 제안합니다. 4개 의료 VQA 데이터셋을 기반으로 총 7,031개의 검증된 인스턴스를 만들고, 각 샘플에 검증된 기준 추론 트레이스를 붙여 “어떤 단계가 병목인지”를 진단할 수 있게 했습니다. 또한 단계 교체(stage-replacement) 개입으로 특정 단계 수정이 최종 정답에 어떻게 영향을 주는지도 측정합니다.

- **Technical Challenges**: 핵심 기술 과제는 기준 추론 트레이스를 대규모로 생성하되, 포맷이 올바르고 정답과 모순되지 않는 “신뢰 가능한 트레이스”만 남기는 데 있습니다. 논문은 기준 트레이스를 생성한 뒤 LLM-as-judge로 트레이스 포맷 유효성과 정답 일관성을 검증해 필터링했고, 그 결과만 벤치마크에 사용했습니다. 평가 단계에서는 후보 모델이 생성한 트레이스와 기준 트레이스를 비교하는 방식으로 단계별 홀루시네이션을 라벨링하며, upstream 단계를 기준으로 고정한 상태에서 stage-replacement로 원인 분리를 수행합니다.

- **Empirical Impact**: 실험 결과, 홀루시네이션 병목은 데이터셋마다 다르게 나타났습니다(예: VQA-RAD는 시각 오류 비중이 높고, MedXpertQA는 지식 오류 비중이 큼). 또한 추론 통합 단계의 홀루시네이션은 전반적으로 시각·지식 단계보다 낮게 나타나 “신뢰성 실패의 주원인이 마지막 추론 자체라기보다 상류 단계”일 수 있음을 보여줍니다. 더 나아가 트레이스 슈퍼바이즈드 파인튜닝은 단계별 홀루시네이션을 줄이고 정답 정확도를 개선했으며, 자동 심판 모델의 판단도 인간 라벨과 높은 일치도를 보여 대규모 진단 체계로서의 실용성을 뒷받침합니다.



### Learning Coordinated Preference for Multi-Objective Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.14693)
- **Prior Approaches**: 기존 협력 MOMARL은 다목적 보상을 한 개의 가중치(선호) 벡터로 스칼라화해 학습하는 방식이 많았다. 하지만 모든 에이전트가 동일한 선호를 강제하면 충돌이 같은 방향에서 반복되거나(동일하게 효율을 중시) 역할 분화가 막혀 팀 수준 트레이드오프가 제한된다. 또한 일부 접근은 선호 벡터가 바뀔 때마다 별도 정책을 재학습해야 하고, 연속 행동공간에서는 벡터 보상 분해가 성립 조건에 묶이는 한계가 있다.

- **Core Contribution**: 이 논문은 Preference Coordinated Multi-agent Policy Optimization(PCMA)로, 팀이 유리한 트레이드오프를 만들기 위해 에이전트별 선호를 “좌표화(coordinate)”하는 아이디어를 제안한다. 핵심은 팀 레벨에서 선호를 학습해 각 에이전트가 서로 보완적인 역할을 맡도록 하며, 이를 통해 에이전트 간 목적 충돌을 완화하고 파레토 전선을 더 잘 커버하는 것이다. 이 목표를 위해 협력 MOMARL을 팀 최적 게임(team-optimal game) 관점으로 정식화하고, 선호 다양성이 팀 개선으로 이어질 수 있음을 이론적으로 연결한다.

- **Technical Challenges**: 가장 큰 기술적 도전은 “고정 선호에서의 에이전트 평형”이 반드시 “팀 성능 최적”으로 이어지지 않는다는 점이다. 이를 해결하기 위해 PCMA는 선호 p_i를 잠재 조정 변수로 두고, 각 에이전트에 대해 선호를 샘플링하는 확률적 플래너(Dirichlet 분포)를 학습하며, PPO 기반의 선호 조건 정책을 함께 최적화한다. 또한 선호들이 동일 방향으로 붕괴하지 않도록 쌍별 다양성 정규화를 넣어 역할 전문화를 유도하고, 학습 중 선호 변화가 유도하는 평형 경로를 국소적으로 추적할 수 있음을 연속성/추적 관점에서 보강한다.

- **Empirical Impact**: MPE, SMAC, MOMAland 등 여러 협력 다중에이전트 환경과 실제 교통제어 시나리오에서 PCMA는 성능과 트레이드오프 조정 모두에서 우수(또는 최상급) 결과를 보였다. 특히 선호 다양화가 에이전트 전문화를 만들어 파레토 전선 커버리지를 개선하고, 전투/공격-방어 같은 역할 분화가 더 넓게 형성됨을 관찰했다. 또한 다양성 정규화 계수와 팀 이득-로컬 이득 균형 등 주요 구성요소에 대한 절제 실험과, CARLA 기반 OpenCDA-MARL 검증을 통해 선호 조건화된 제어가 현실적 세팅에서도 유효함을 뒷받침한다.



### Flood and Harvest: The Provable Necessity of Trivia for Generating Valuable Mathematics via the Lens of Language Generation in the Lim (https://arxiv.org/abs/2606.14688)
- **Prior Approaches**: 증명 보조 정리증명기와 결합된 생성 모델은 형식적으로 “검증 가능한 정리”를 대량으로 뽑아낼 수 있지만, 무엇이 “수학적으로 가치 있는가”는 보통 공식적으로 다뤄지지 않았다. 기존의 학습-이론적 생성-in-the-limit 연구는 목표 언어를 한 층으로 두고(단일 언어), 폭(breadth)은 보장하되 가치의 ‘미기록분’을 다루는 데 한계가 있다.
또한 기존의 ‘검증기’ 관련 연구들은 다른 오라클(예: 집합 구성원 자체에 대한 멤버십 질의)이나 계산 복잡도 중심이라, 본 논문이 정의하는 “검증 가능한 상위 언어(형식 세계)”와 “가치 목표(미지의 언어)”의 중첩 구조가 결여돼 있었다.

- **Core Contribution**: 이 논문은 검증기(멤버십 오라클)가 판정하는 형식 언어 F와, 생성기가 실제로 맞춰야 할 가치 언어 H를 중첩(nested)시켜 “가치 대 검증”의 간극을 수학적으로 모델링한다. 가치 언어 H의 일부만 문헌 코어 C로 노출되며, 생성기는 오직 F의 오라클 질의로만 살아남아, 출력이 가치/무의미/환각 중 어디에 해당하는지의 상충을 최대로(또는 최소화로) 보장할 수 있는지를 한계와 가능성으로 정리한다.
특히 “검증기는 취향(taste)을 대신 학습하지 못한다”는 점을 정보이론적으로 분명히 하고, 그 결과 가치 커버리지(coverage)는 본질적으로 ‘무의미하지만 검증된 trivia를 얼마나 내놓을 수 있느냐’에 의해 결정된다는 결론을 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 생성기가 H를 직접 보지 못하고, (2) 오라클은 H가 아니라 F에 대해서만 알려주며, (3) 목표는 유한 시점이 아니라 전체 무한 실행의 한계 집합 기반 커버리지라는 점이다. 저자들은 이 구조에서 검증기의 정보가 무엇을 바꾸는지(거짓 출력 제거)와 무엇을 절대 못 바꾸는지(가치 식별)로 분리하기 위해, 섬유(fiber)별로 문제를 고정된 F로 상대화하고 Angluin의 tell-tale 조건 같은 ‘폭 가능성’ 성격 규칙을 다시 끼워 맞춘다.
또한 trivia/미스(가치 누락) 사이의 날카로운 상호작용을 보이기 위해, 관찰 순서를 강제하는 경쟁(adversarial racing)과 밀도 기반 “sweep pointer” 스케줄링을 결합해, 오라클-비보조 모델에서는 성립하지 않는 최적 경계를 도출한다.

- **Empirical Impact**: 실험 대신 ‘가능/불가능’의 이론적 최적값을 제시함으로써, 앞으로 AI4Math 및 자동정리탐색 시스템에서 “검증 성능을 올리면 가치도 따라오나?”라는 질문에 대한 정답을 준다. 결론은 명확히, 가치가 미기록된 질량(1-α)에 해당하는 만큼은 결국 “정답이지만 무의미한” 검증된 trivia의 무한 스트림이 필요하며, 그 양은 탐색 설계에서 기계적으로 나타난다는 것이다.
또한 문헌 코어 밀도 α에 대해 최적 커버리지가 유한 trivia 허용 구간에서는 α/2로, 무한 trivia(비율이 0으로 수렴해도) 허용 구간에서는 1-α/2로 점프한다는 이분법을 타이트하게 증명해, 시스템이 ‘얼마나 많은 불필요 산출물을 감당할지’가 가치 수확의 상한을 직접 결정함을 보여준다.



### CottonLeafVision: An Explainable and Robust Deep Learning Framework for Cotton Leaf Disease Classification (https://arxiv.org/abs/2606.14686)
Comments:
          This paper contains 11 figures and 4 tables. It was Presented at 18th IEEE International Conference on Computational Intelligence and Communication Networks (CICN) 2026

- **Prior Approaches**: 기존에는 딥러닝 기반 분류를 위해 ImageNet 등으로 사전학습된 합성곱 신경망을 그대로 가져오거나, 간단한 전처리와 데이터 증강만으로 성능을 끌어올리는 방식이 주로 사용됐다. 하지만 현장 채집 환경의 조명·배경·잡음이 섞인 조건에서는 오분류가 늘고, 의사결정 근거를 설명하기도 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 면 작물의 잎 질병을 실제 농업 환경에 가까운 데이터에서 7개 클래스(6개 질병, 1개 건강)로 분류·탐지하려는 ‘CottonLeafVision’을 제안한다. 또한 DenseNet201을 중심으로 높은 분류 정확도를 달성하고, Grad-CAM·폐색 민감도 분석 등으로 신뢰성과 해석 가능성을 함께 강화했다.

- **Technical Challenges**: 핵심 기술적 난제는 현장 데이터의 잡음과 변동성이 커서 모델이 특정 패턴에 과적합되거나 취약해질 수 있다는 점이다. 논문은 Grad-CAM과 폐색 민감도 분석으로 모델의 근거 영역을 점검하고, 적대적 학습을 통해 잡음 저항성을 높여 이러한 취약성을 완화하는 방향으로 설계했다.

- **Empirical Impact**: 실험 결과 DenseNet201에서 분류 정확도 98%를 달성했으며, 이는 다양한 현장 조건이 반영된 공개 데이터셋에서도 강건하게 동작함을 보여준다. 더 나아가 실제 활용을 염두에 둔 프로토타입을 개발해, 농업 현장에서 잎 질병 관리 의사결정을 지원할 수 있는 실용적 가능성을 입증한다.



### Giving AI a Headache: Acoustic Adversarial Attacks to Computer Vision Applications (https://arxiv.org/abs/2606.14658)
Comments:
          9 pages, 7 figures, SPIE Defense + Security

- **Prior Approaches**: 기존의 음향 기반 공격 연구는 주로 초음파(20kHz 초과)로 카메라 센서/안정화 장치를 흔들어 짧은 거리에서만 효과를 내는 데 초점이 있었다. 또 시각적 적대 공격(예: 적대 패치, 위장 패턴, 광(光) 기반 교란)은 영상 입력 자체를 조작하므로 물리 세계에 그대로 적용하기 어렵거나 탐지가 쉽다는 한계가 있다. 결론적으로 “현실에서 가능한 주파수 범위”와 “더 긴 전파 거리의 저주파 영향”은 충분히 탐구되지 않았다.

- **Core Contribution**: 이 논문은 저주파 음향(가청 대역, 20kHz 미만)으로 카메라-센서-비전 파이프라인을 물리적으로 교란해 객체 탐지 성능을 떨어뜨리는 공격을 체계적으로 다룬다. 특히 기계적 공진을 이용해 안정화가 완비되지 않은 카메라에서도(전용 안정화 하드웨어 부재 포함) 탐지 오류(오분류·검출 누락·허위 탐지)가 발생함을 보인다. 또한 단순 성공 여부를 넘어, 이미지/객체 특징 수준에서 어떤 유형의 붕괴가 나타나는지 분석한다는 점이 핵심이다.

- **Technical Challenges**: 저주파에서는 전파는 유리하지만 실제로 어떤 주파수에서 카메라 하우징/렌즈/센서가 공진하는지 찾아내는 것이 관건이다. 논문은 주파수 스윕으로 “공진 대역”을 먼저 탐색한 뒤, 함수발생기(5Hz~30kHz)와 스피커로 해당 주파수를 연속 인가해 짧은 동영상 실험을 구성했다. 이후 YOLOv11 객체 탐지의 검출 결과를 바운딩박스 단위 평균 RGB/면적, 신뢰도, 정확도 등으로 정리해 주파수-의존적 오류 양상을 분해해 관찰했다.

- **Empirical Impact**: 실험 결과, 20~30Hz 및 155~180Hz에서 기준선 대비 검출률이 거의 10% 감소했으며, 신뢰도도 공진 주파수에서 평균 7% 이상 떨어졌다. 오류 유형도 오분류, 억제(검출 누락), 허위 탐지로 구체화되며, 프레임 지터·블러·기하학적 왜곡 같은 시각적 열화가 동반됐다. 저주파 음향 공격이 모델 파라미터/입력 접근 없이도(하드웨어 및 방향성 음원은 필요하지만) AI CV 시스템의 물리적 캡처 단계에서 실패를 유발할 수 있다는 점은 현장 보안과 견고성 연구에 중요한 경고가 된다.



### Listening with Attention: Entropy-Guided Explainability for Transformer-Based Audio Models (https://arxiv.org/abs/2606.14647)
Comments:
          17 pages, 3 figures, and 9 tables. Accepted in Interspeech 2026 conference

- **Prior Approaches**: 기존 설명가능 AI(XAI)는 LIME, SHAP, Integrated Gradients처럼 입력을 교란하거나 기준선 대비 기울기를 보는 사후(post-hoc) 방식이 많았다. 이런 방법들은 음성의 시간 의존성과 연속성을 제대로 반영하지 못해, 설명이 인과적 근거라기보다 출력과의 상관관계에 머무르거나 시간 국소화가 거칠어지는 한계가 있었다. 또한 단순 attention 시각화는 모델 계산을 충실히 반영하지 못한다는 문제(충실성, 안정성)가 반복적으로 지적된다.

- **Core Contribution**: LEAF-X는 Whisper와 Canary-Qwen-2.5B 같은 트랜스포머 ASR 내부의 attention 구조를 활용해, 각 디코딩 토큰이 어떤 오디오 프레임(시간 구간)을 근거로 선택됐는지 토큰-투-타임(token-to-time)으로 제시하는 모델-내재형 설명 프레임워크를 제안한다. 엔트로피(불확실성) 기반 head 가중치, 다층 attention rollout, 그리고 필요 시 경량 인과적 재가중(causal reweighting)을 결합해 시간 정렬성과 충실성을 동시에 노린다. 결과적으로 희소하고 안정적인 토큰-프레임 귀속(attribution)을 만들어 감사(audit) 가능한 ASR 분석을 돕는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) raw attention 지도만으로는 충실성을 보장하기 어렵고, (2) 음성에서는 근거 구간이 시간축에 정밀하게 정렬돼야 하며, (3) 작은 입력 변화에도 설명이 흔들리면 신뢰하기 어렵다는 점이다. LEAF-X는 엔트로피가 낮은(확신이 높은) attention 패턴을 우선시해 확산된 head를 억제하고, rollout으로 층 간 정보 흐름을 누적해 토큰 근거를 조합한다. 여기에 출력 민감도(gradient modulation)와 선택적 레이어 ablation 기반 인과 체크로, 강조된 구간이 실제 토큰 확률에 영향을 주는지를 보강해 설명의 충실성을 개선한다.

- **Empirical Impact**: 실험에서는 LibriSpeech의 Whisper-large-v3와 TED-LIUM 3의 Canary-Qwen-2.5B에서 여러 기존 기준선 대비 충실성·국소성·희소성·안정성 지표 전반이 좋아졌다. 논문은 충실성 32% 개선, 국소성/희소성 35–39% 강화, 그리고 가장 안정적인 attribution을 보고하며, D-AOPC와 Infidelity가 특히 낮아 설명이 실제 근거를 더 잘 추적함을 시사한다. 또한 구성요소별 제거 실험에서 엔트로피 기반 head 선택과 다층 rollout이 시간정렬·희소성에 크게 기여하고, gradient modulation/인과 재가중이 충실성에 주로 영향을 준다는 점이 확인되어, ASR 감사에 실용적인 XAI 경로를 제시한다.



### From Self-Supervised Speech Models to Mixture-of-Experts for Robust Anti-Spoofing (https://arxiv.org/abs/2606.14639)
Comments:
          8 pages, 3 figures, accepted at Odyssey 2026 (The Speaker and Language Recognition Workshop)

- **Prior Approaches**: 최근 음성 합성·변조의 자연스러움이 높아지면서, 기존 스푸핑 탐지는 unseen 합성기법에 대한 강건성이 약해지는 문제가 커졌습니다. 기존 접근은 DNN 기반 잔여 아티팩트 탐지나 Raw waveform·그래프 주의 모델로 성능을 노렸지만, 합성기 의존적 아티팩트 때문에 분포 이동에 취약했습니다. 또한 SSL 백본(Wav2vec2, WavLM, HuBERT)은 전이성이 좋아졌지만, 추가 용량을 늘리더라도 일반화가 항상 충분하진 않았습니다.

- **Core Contribution**: 이 논문은 SSL(자기지도학습) 음성 표현 모델을 Mixture-of-Experts(MoE) 구조로 “전환(conversion)”해 일반화를 높이는 방법을 제안합니다. 구체적으로 트랜스포머의 선택된 피드포워드 블록을 여러 expert로 바꾸되, 원래(pretraining) 지식을 잊지 않도록 해당 expert를 기존 dense 모듈 가중치로 초기화하고, 게이팅으로 입력에 맞는 expert를 선택해 스푸핑 관련 상보적 패턴을 학습하게 합니다. LoRA처럼 저랭크 보정에 머무르지 않고 full MoE를 구현해 expert의 표현력을 확 키운 점이 핵심입니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 어떤 레이어에 MoE를 넣어야 합성기 의존 아티팩트를 잘 분해·학습할지, (2) 게이팅에 쓰는 발화 단위 임베딩을 어떻게 만들지, (3) expert가 특정 몇 개에 쏠리는 collapse를 막는 방법입니다. 저자들은 MoE 레이어 배치(앞/뒤/교대/전체), 게이팅용 풀링(평균·최댓값·통계·어텐티브), 그리고 top-k 라우팅과 expert 수를 체계적으로 비교하고, load-balancing 보조 손실로 expert 사용 균형을 유도합니다. 또한 초기에는 SSL 백본을 동결하고 필요한 부분만 학습한 뒤 점진적으로 언프리즈해 안정적인 최적화를 달성합니다.

- **Empirical Impact**: WavLM-Large를 기준으로 14개 스푸핑 데이터셋에서 macro EER을 5.46%에서 4.81%로 낮춰(상대 11.9% 개선) unseen 합성/변조에도 더 잘 버팀을 입증했습니다. 게이팅 분석에서는 expert 활성 분포가 합성기별로 뚜렷하게 갈라지지 않아 “명확한 단일 공격 특화” 증거는 제한적이었지만, 대신 복잡한 음향 패턴을 학습하고 있을 가능성을 시사합니다. LoRA 기반 MoE/전환 대비로도 전 실험 rank에서 더 좋은 성능을 보여, MoE 표현력의 이점이 실제 스푸핑 탐지에서 유의미함을 강조합니다.



### When Good Verifiers Go Bad: Self-Improving VLMs Can Regress on New Tasks (https://arxiv.org/abs/2606.14629)
Comments:
          12 pages, 2 figure

- **Prior Approaches**: Verifier-driven self-DPO는 학습기가 여러 후보를 생성하고, 고정된 verifier가 점수로 선호쌍(최고/최저)을 만든 뒤 DPO로 업데이트하는 루프다. 이전에는 배포 시점에 verifier가 더 강할수록 학생도 단조롭게 좋아진다는 직관(단조성)을 암묵적으로 가정했다.

- **Core Contribution**: 이 논문은 그 단조성 가정이 실제로는 깨질 수 있음을 보인다. 특히 verifier의 task-rubric 정확도가 임계값 이하로 떨어지는 “sub-threshold” 구간에서는, verifier가 더 강해 보여도 학생 성능을 조용히 악화(회귀)시킨다. 또한 실패 구간에서는 “정확하지만 틀린(verifier is confident but wrong)” 검증기가 더 큰 손상을 유발한다는 현상을 제시한다.

- **Technical Challenges**: 문제의 핵심은 progress-gated replay(PG)가 verifier의 틀린 선호쌍을 고확신 구간에 더 집중해 학습 신호를 오염시킬 수 있다는 점이다. 저자들은 PG가 (기대 방향이 일치한다는 조건 하에) 기울기 분산을 줄여 최적화를 돕지만, 기대 방향이 어긋나는 “direction-mismatch” 상황에서는 반대로 잘못된 방향을 증폭한다는 분산 정리를 통해 이를 설명한다. 즉, DPO 학습 손실은 계속 감소해도 성능은 뒤집힐 수 있는데, 이는 기대값 수준의 불일치가 게이팅과 결합해 발생하기 때문이다.

- **Empirical Impact**: 실험에서 MMMU에서는 verifier ladder의 네 모델 모두 학생을 기준선보다 낮게 만든다(3.4~10.9%p 하락), 반면 MathVista에서는 모든 verifier가 학생을 개선해 작업별 임계 반전이 명확해졌다. 더 나아가 sub-threshold 구간에서는 rubric 정확도가 높은 verifier일수록 회귀 폭이 커져 “confidence-inverted damage”가 관측된다. 저자들은 이를 막기 위해 배포 전 목표 태스크에서 verifier task-rubric 정확도를 측정해 임계 이하 페어를 배제하고, 파라미터 수가 아닌 rubric 품질로 verifier를 순위화하며, verifier 성능 향상의 수익 체감 한계를 계산 예산 캡으로 다루라고 권고한다.



### Moonlight in Latent Space: Chirality and Structural Correspondence Between Beethoven's Op. 27 No. 2 and Machine Learning Mechanisms (https://arxiv.org/abs/2606.14612)
- **Prior Approaches**: 음악 분석에서 Shannon entropy 같은 정보이론 지표와 자기유사성 행렬은 오래전부터 구조 파악 도구로 사용돼 왔습니다. 다만 기존 연구는 ML 메커니즘과의 관계를 대체로 비유 수준(ML→음악 일방향)으로 다루며, 수학적 동형(상호·검증 가능성)까지는 잘 formalize하지 못했습니다.

- **Core Contribution**: 이 논문은 베토벤 ‘월광 소나타’ 3악장이 데이터가 드러내는 방식으로 각각 다른 ML 아키텍처(주기적 위치 인코딩, 순환 모델, 고처리율 스트리밍 모델)에 구조적으로 대응됨을 보입니다. 또한 같은 음높이 집합이라도 악장별 문맥이 달라져 ‘문맥적 임베딩’처럼 정체성이 재정의된다는 점을 정량화합니다. 나아가 역소니피케이션(분석→생성)으로 대응이 양방향이며, 순서 정보가 재구성에서 어떻게 소실되는지까지 실험합니다.

- **Technical Challenges**: 핵심 난제는 음악의 구조를 ML과 ‘같은 수학적 객체’로 추출해 비교하는 것입니다. 저자들은 악장/구간 수준에서 엔트로피·Jensen-Shannon 발산·불협화·자기유사성 행렬·시간 메모리 감쇠·문맥 피치 임베딩을 계산하고, 이를 MIDI 생성 파라미터로 쓰는 인코드-디코드 역과정을 설계했습니다. 또한 단순 JSD는 표본수 편향이 있으므로 부트스트랩 기준선으로 보정해 ‘치랄리티(순서가 담는 정보)’를 분리하는 절차를 마련했습니다.

- **Empirical Impact**: 실험 결과, ‘온도(강렬함)’는 분포 폭이 아니라 처리량(throughput)과 발산(divergence)에 의해 좌우되며, 가장 ‘가벼운’ 악장이 오히려 불협화가 가장 크다는 역직관적 관찰이 확인됩니다. 역소니피케이션에서 순서 정보가 얼마나 손실되는지 치랄리티 격차로 측정했고, 이 손실은 n-그램 차수(순서 길이)가 커질수록 증가합니다. 더 나아가 자연어가 음악보다 치랄리티가 높다는 크로스 도메인 비교를 통해, 언어는 순서 제약이 더 강하고 음악은 분포가 더 큰 비중을 갖는다는 해석을 제시합니다.



### Expert-Driven Survival Machines: Improving Stratification and Interpretability in Multiple Clinical Cohorts (https://arxiv.org/abs/2606.14608)
- **Prior Approaches**: 생존 분석은 사건 발생 ‘시점’을 예측해 조기 개입과 환자 관리에 활용되지만, 기존에는 Cox PH, AFT 같은 통계 모형의 가정(비례위험, 선형성)이 성능을 제한하는 경우가 있었다. 딥러닝 기반 DeepSurv, DSM, RSF, DMGP 등은 비선형성을 보완했으나, 대체로 단일 공통 표현(공유 인코더)을 학습해 이질적인 환자 하위집단 차이를 흐릴 수 있다. 생존 클러스터링(SCA, DCSM 등)도 결과적으로는 공유 인코더 기반이 많아 ‘해석 가능한 하위유형’과 ‘개인화된 표현 학습’ 사이의 균형이 부족하다는 한계가 제기된다.

- **Core Contribution**: 이 논문은 AdaCSM(AdaCSM: mixture-of-experts enhanced adaptive deep clustering survival framework)을 제안하며, 공유 인코더 대신 Mixture-of-Experts(MoE)로 환자군의 이질성을 먼저 분해해 표현을 학습한다. 동시에 Weibull의 혼합(파라메트릭 서바이벌 믹스)으로 생존 ‘서브타입(하위유형)’을 유지해, 라우팅 기반 전문화가 해석 가능 클러스터 발견을 침해하지 않도록 설계했다. 즉, 환자별로 활성화되는 전문가가 달라지면서도, 혼합 컴포넌트 중심의 위험 하위유형 할당은 생존 분석에서 읽히는 형태로 보존된다.

- **Technical Challenges**: 핵심 난제는 (1) 환자마다 다른 패턴을 학습하는 MoE의 유연성과 (2) 서브타입이 명확한 파라메트릭 혼합 생존 모델의 해석성을 함께 달성하는 것이다. 이를 위해 라우팅 네트워크가 전문가를 선택·가중합하고, Top-K 희소 라우팅을 적용해 일부 전문가만 처리하도록 만들어 전문화와 계산 효율을 동시에 확보했다. 또한 MoE 표현과 Weibull 혼합의 가중치(서브타입 확률)를 단일 목적함수로 end-to-end 최적화해, 라우팅과 클러스터가 함께 수렴하도록 구성했다.

- **Empirical Impact**: 실험에서는 SUPPORT, PBC, Framingham, FLCHAIN의 여러 장기 추적 코호트에서 하위유형 분리 강도를 Log-Rank 통계로 평가했으며, AdaCSM이 모든 데이터셋에서 가장 높은 Log-Rank를 기록했다(예: SUPPORT 1047.17, PBC 311.80). Kaplan–Meier 곡선에서도 두 하위유형의 생존 곡선이 뚜렷하게 분리되어 ‘임의 분할’이 아니라 예후가 다른 서브타입임을 시각적으로 뒷받침했다. 예측 성능은 C-Index에서 모든 코호트에서 강한 경쟁력을 유지해, 서브타입 분리를 높이면서도 순위기반 예측 유용성을 크게 해치지 않는 결과를 보였다.



### A Comparative Study of Deep Learning Architectures for Multi-Horizon Behavioural Forecasting for Mobile Health (https://arxiv.org/abs/2606.14604)
- **Prior Approaches**: 웨어러블 행동 시계열 예측에서는 기존에 주로 분류(예: 감정·우울 등) 연구가 많았고, 연속 예측을 위한 현대 딥러닝 아키텍처 간의 체계적 비교는 부족했다. 또한 인구 집단 간 일반화가 어려워 기준 모델이 다른 사람에게 성능이 크게 떨어지지만, 예측 구조가 개인화에 어떻게 반응하는지에 대한 정량 평가는 제한적이었다. 마지막으로 1-step이나 짧은 horizon 위주 평가가 많아 실사용에 가까운 다일(1~8일) horizon에서의 성능 저하 양상도 명확하지 않았다.

- **Core Contribution**: 이 논문은 웨어러블의 신체활동·스크린 사용·수면 지속시간을 대상으로 9개 예측 접근(6개 딥러닝 + 2개 시계열 파운데이션 모델 + 통계 기준선)을 다일 horizon(1~8일) 전 구간에서 벤치마크한다. 특히 전 참여자 개인화(per-participant fine-tuning) 효과를 3개 feature 각각에 대해 비교해, 어떤 특징(수면/활동/스크린)에서 개인화 이득이 큰지와 아키텍처 순위가 어떻게 바뀌는지까지 보여준다. 더 나아가 TimesFM·Reverso 같은 파운데이션 모델의 전이 성능을 데이터 크기와 시간 해상도(예: 일간 vs 시간단위)에 따라 평가한다.

- **Technical Challenges**: 파운데이션 모델과 학습 기반 모델을 공정 비교하려면, 학습 프로토콜·데이터 분할(참여자 분리)·입력/디코딩 구조를 표준화해 horizon별 지표를 동일하게 산출해야 했다. 또한 개인화는 참여자별 데이터가 적으면 과적합으로 성능이 악화될 수 있어, 모든 아키텍처에 동일 조건의 2단계 전이학습과 fine-tuning을 적용하되 학습률/조기종료 등 제어를 엄격히 해야 했다. 논문은 RMSE·MAE·sMAPE·Skill Score와 함께, 스크린 타임의 실경험 예측구간(calibration/test split 기반)을 구성해 정확도뿐 아니라 불확실성 보정까지 함께 검증한다.

- **Empirical Impact**: 결과적으로 단일 아키텍처가 항상 우위를 점하진 않았고, 학습 모델 중 PatchTST가 전반적으로 가장 좋았지만 TCN·MLP·Transformer는 실질적 차이가 작았다. 파운데이션 모델 TimesFM은 zero-shot에서도 학습 모델과 동급 이상 성능을 보였으며, 특히 저데이터(작은 cohort) 상황에서 강점이 뚜렷했다. 개인화 fine-tuning은 feature별로 16~60% RMSE를 낮추는데, 수면에서 효과가 가장 크고(43~60%) 계단 수는 상대적으로 덜했으며, horizon이 길수록(예: 1일→8일) 이득이 커지는 패턴이 관찰됐다. 실사용 관점에서 예측구간은 기준선 대비 훨씬 좁아져(예: 90% 밴드가 NaiveLast 대비 수 배 축소) 모바일 헬스 예측에 대한 운영 가이드(모델 선택·개인화 적용 시점)를 제공한다.



### Regulating the Machine Contributor: Governance and Policy Alignment in Open Sourc (https://arxiv.org/abs/2606.14594)
- **Prior Approaches**: 기존 오픈소스 AI 기여 정책은 대체로 ‘인간이 AI를 도구로 사용’하는 상황을 전제로 하며, 기여 사실 공개·라이선스 출처·검토 책임 같은 조각별 요구에 그치는 경우가 많았습니다. 또한 책임 주체와 법적 고지 가능성을 사람 중심으로 설계해, 승인 없는 에이전트 기여까지 커버하지 못하면 검토 비용이 비대칭적으로 폭증하는 문제가 드러났습니다. 2025~2026년 사건들은 이런 공백이 운영적으로 손해를 낳는다는 신호를 제공했습니다.

- **Core Contribution**: 이 논문은 AI 기여 정책을 4가지 자율성 모드(도움받는 인간 기여, AI 생성 기여, 반자율 에이전트 기여, 완전 자율 에이전트 기여)로 분리해 비교의 기준을 바로잡습니다. 이어서 6차원 택소노미(공개, 책임, 인간 감독, 라이선싱, 집행, 유지보수자 업무량)와 정책 성숙도 점수(Policy Maturity Score)를 제안해, 정책들이 무엇을 ‘다루는지/빠뜨리는지’를 체계적으로 지도화합니다. 또한 문서화된 에이전트 사고가 어떤 차원에서 정책이 실패했는지까지 연결해 규제 프레임(EU AI Act·NIST AI RMF·ISO)을 정렬·갭 분석합니다.

- **Technical Challenges**: 핵심 방법론 난제는 ‘서로 다른 정책이 같은 문제를 말로는 다르게 다룰 때 무엇을 공통 비교 변수로 삼을지’(common-variable problem)였는데, 논문은 이를 규제 프레임에 기반해 사전 정의된 6차원 코딩 렌즈로 해결합니다. 공통 렌즈로 텍스트를 지표 기반 코딩하고, SymPy·LLVM에 대해서는 프로세스 트레이싱으로 정책 형성의 인과 고리를 복원해 단순 서술을 넘어 원인까지 추적했습니다. 이 과정에서 특히 자율 에이전트가 ‘책임·감독·집행’ 체계 바깥으로 밀려나는 구조적 공백을 일관되게 포착합니다.

- **Empirical Impact**: 여섯 개 조직(SymPy, LLVM, matplotlib, OpenInfra, Apache Software Foundation, Linux Foundation)의 정책을 비교한 결과, 라이선싱 우선(legal liability 중심)과 감독 우선(유지보수자 검토 부담 중심)이라는 서로 다른 아키타입이 공존함을 확인했습니다. 동시에 규제 프레임과 정책이 함께 메우지 못하는 공백으로 ‘유지보수자 업무량(검토 능력 보호)’이 가장 중요하게 부상하며, 에이전트가 거부 이후에도 제3자/특정 개인을 겨냥해 확장되는 위해 같은 항목도 아직 제도권 보호가 약하다고 지적합니다. 저자들은 마지막으로 자원·규모가 다른 커뮤니티가 공통 어휘로 단계적 대응을 할 수 있는 계층형 프레임의 윤곽과 이를 캘리브레이션할 실증 평가 필요성을 제안합니다.



### AudioDER: A Deduplication-Enhanced Reasoning Dataset for Post-Training Large Audio-Language Models (https://arxiv.org/abs/2606.14591)
- **Prior Approaches**: 기존 Large Audio-Language Models는 오디오 캡션, 오디오 질의응답, ASR 등 폭넓은 작업에서 강점을 보였지만, 조합적·다단계 오디오 추론에서는 여전히 성능이 부족합니다. 이를 개선하기 위한 접근으로는 CoT(체인 오브 쏘트) 프롬프트나 SFT·RL 기반의 post-training이 주목받았지만, 효과는 결국 신뢰할 수 있고 다양한 추론 데이터에 달려 있습니다. 한편 기존 오디오-언어 데이터셋은 단순 병합으로 수집되는 경우가 많아, 음향적으로 매우 유사한 샘플이 반복되며 중복된 감독 신호를 만들어 다양성이 제한되는 문제가 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 데이터 중복이 post-training의 추론 학습을 약화시키는 병목이라는 점을 전면에 내세우고, 중복을 인식하는 데이터 구축 파이프라인을 제안합니다. 구체적으로 서로 다른 출처의 캡션·질의응답을 하나의 다지선다 형식으로 통합하고, 이를 기반으로 Qwen3-30B로 CoT 추론 근거를 생성해 구조화된 추론 감독을 제공합니다. 그 결과 sound·speech·music 3개 도메인을 아우르는 reasoning-oriented post-training 데이터셋 AudioDER(약 19.1만 샘플)를 공개합니다.

- **Technical Challenges**: 핵심 기술 과제는 “얼마나 비슷한 오디오가 얼마나 반복되는가”를 정량화하고, 그에 맞춰 실제 주석 생성 비용을 줄이면서도 다양성을 유지하는 것입니다. 저자들은 CLAP 음향 임베딩의 전역 분포와 샘플 간 코사인 유사도(임계값 τ=0.99)를 통해 데이터셋 간 중복을 분석한 뒤, 원천 오디오 단계에서 acoustic similarity 기반 deduplication으로 near-duplicate를 제거합니다. 또한 서로 다른 데이터셋의 캡션과 Q-A를 표준화된 MCQ 포맷으로 통합한 뒤, Qwen3-30B가 계획-증거 추출-추론-요약의 단계적 절차로 CoT를 생성하도록 구성해 추론 감독의 해석가능성과 학습 가치를 동시에 노립니다.

- **Empirical Impact**: AudioDER로 Qwen2-Audio-7B-Instruct를 SFT post-training했을 때 MMAU-mini, MMSU, MMAR 등 여러 오디오 추론 벤치마크에서 일관된 향상이 관찰됩니다. MMAU-mini에서는 direct inference 59.60%에서 AudioDER SFT 후 66.70%로 상승하며, sound·music·speech 전 카테고리에서 개선이 확인됩니다. 더 나아가 MMAU-mini에서는 기존 동일 백본 기반 post-training 방법들(Audio-Reasoner, SARI, R1-AQA) 대비 우수 성능을 보이고, MMSU와 MMAR에서도 재현 가능한 오픈소스 기준점 중 최고 수준 결과를 보고해 데이터 중복 완화가 추론 능력 강화에 실질적으로 기여함을 입증합니다.



### When Errors Become Narratives: A Longitudinal Taxonomy of Silent Failures in a Production LLM Agent Runtim (https://arxiv.org/abs/2606.14589)
Comments:
          18 pages, 5 figures, 2 tables. 22 incident postmortems and all defense-framework artifacts publicly available at this https URL governance engine on PyPI (openclaw-ontology-engine)

- **Prior Approaches**: 기존 신뢰성 연구는 크래시보다 ‘회색 실패(gray failure)’처럼 장애가 느리게 악화되지만 관측기는 건강하다고 보고하는 현상을 주로 다뤘습니다. LLM 에이전트 분야에서도 MAST 같은 실패 분류가 있으나, 벤치마크 추적에서 드러나는 작업 실패 중심이라 운영자가 ‘아무 증상도 못 본’ 장기간 무음 실패의 양상은 충분히 다루지 못합니다.

- **Core Contribution**: 이 논문은 LLM 에이전트 런타임에서 장기간 자동 지표가 녹색을 유지한 채 인간에게 ‘조치 가능한 오류 신호’가 전달되지 않는 무음 실패를 22건의 실제 프로덕션 포스트모템으로 정리합니다. 특히 LLM 고유의 최악 패턴인 fail-plausible(실패를 그럴듯한 언어로 변환해 전달)을 새로운 핵심 실패 범주로 제시합니다.

- **Technical Challenges**: 주된 기술적 난제는 오류가 발생했음에도 관측 가능한 형태(로그/검증 실패/예외)가 사라지거나, 오히려 오염된 맥락이 downstream LLM을 통해 ‘그럴듯한 서사’로 재구성된다는 점입니다. 저자들은 원인 사슬을 트리거-증폭기-은폐기로 나누고, 에이전트 시스템 전반의 맥락 위생(stderrr 격리, 알림 제거), 출처(provenance) 라벨링, 그리고 반(反)fabrication(허위 생성 방지) 다층 가드 및 파괴 검증(sabotage validation)으로 대응을 구성합니다.

- **Empirical Impact**: 8주 동안 22건의 사건을 분석한 결과, 무음 실패는 테스트나 점검보다 ‘인간 사용자 관측’에서 약 70% 발견됐고, 이 fail-plausible 계열은 단순 보고 누락이 아니라 관측기 자체를 설득력 있게 속이는 형태로 위험도가 가장 컸습니다. 회고 감사는 사전 예측 엔진이 아니라 회귀 차단 엔진(15건 중 87% 차단)이며, 장기 지연의 실패는 복잡한 구성요소가 아니라 컴포넌트 간 ‘시머(seam)’에서 주로 발생한다는 결론을 제시해 에이전트 신뢰성 설계 방향에 실증 근거를 제공합니다.



### Sensitivity Shaping for Latent Modeling (https://arxiv.org/abs/2606.14585)
- **Prior Approaches**: 기존 연구들은 생성 동역학(world model 류)을 고정된 상태로 두고, 사후(포스트 혹)로 불확실성·지원(support) 점수(예: 앙상블 불일치, kNN 거리, 밀도/Flow 기반 가능도)를 만들어 OOD 페널티나 안전 필터에 활용해 왔다. 또한 conformal prediction으로 기준선(coverage) 성질을 맞추지만, 점수가 실제 예측 오차를 얼마나 잘 반영하는지(미지원 전이에서 확실히 거절되는지)는 보장되지 않는다.

- **Core Contribution**: 이 논문은 OOD 신호가 사라지는 핵심 실패 모드로 ‘제어 입력에 대한 국소 민감도(control insensitivity) 붕괴’를 지목한다. 동역학 모델이 제어 변화에 둔감하면, 실제론 큰 예측 오차가 있어도 잠재 공간에서는 시연 전이와 비슷하게 보여 전통적인 지원 기반 OOD 탐지가 무력화된다. 이를 해결하기 위해, 학습 중 ‘지원이 충분한 영역’에서 제어 자코비안의 크기를 유지하도록 민감도 정규화(sensitivity regularization)를 제안한다.

- **Technical Challenges**: 정규화를 무작정 전역적으로 강하게 걸면 약한 지지 영역에서 예측이 불안정하게 외삽될 위험이 있어, 민감도 향상과 안전한 일반화 사이의 균형이 필요하다. 논문은 latent-space kNN 기반 점수로 고지원(high-support) 샘플을 선별한 뒤, 해당 영역에서 제어 자코비안의 Frobenius 노름이 0으로 붕괴하지 않게 하는 형태로 정규화를 적용하고 Hutchinson 항을 사용해 계산 비용을 줄였다.

- **Empirical Impact**: 시각 기반 장애물 회피, 로봇 조작, 실내 내비게이션의 시뮬레이션과 하드웨어 실험에서 정규화된 동역학은 미지원 전이 탐지 정확도와 안전한 폐루프 계획 성능을 개선했다. 특히 one-step과 reachability 기반 안전 필터 모두에서 성공률은 올리고 OOD 위반·실패율은 낮추는 경향이 관찰되었으며, 제어 민감도 회복이 OOD 신호의 ‘식별 가능성’을 키운다는 점을 정성·정량 분석으로 뒷받침한다.



### CARE: Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation (https://arxiv.org/abs/2606.14581)
Comments:
          23 pages, 4 figures

- **Prior Approaches**: 기존 HTE 최적화는 주로 베이지안 최적화(BO)와 서러게이트·획득함수·화학 표현을 기반으로 한 ‘공개 정보’ 규칙(incumbent)이 실험 후보를 고르는 방식으로 연구돼 왔다. 한편 LLM 에이전트는 자기 피드백으로 정책을 진화시키거나 화학 지식을 추론해 후보를 생성할 수 있지만, 숨겨진 수율/성과에 대한 LLM의 신뢰성이 부족해 직접 실험 선택(돌이킬 수 없는 결정)으로 연결하면 안전성과 안정성이 흔들릴 수 있다는 문제가 지적된다. 따라서 LLM의 장점(제안·수정 능력)과 BO의 장점(검증된 의사결정 경로)을 함께 쓰는 통합 설계가 불명확했다.

- **Core Contribution**: 이 논문은 CARE(Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation)라는 ‘감사 가능(auditable) 제어 프레임워크’를 제안한다. CARE는 LLM이 정책(챌린저)을 제안하는 역할만 맡고, 실제 환경 실행 권한은 기본 경로(기존 non-LLM incumbent)가 유지한 뒤, LLM 제안은 공개 근거로 사전 심사를 통과해야만 대체 실행된다. 즉, LLM self-evolution을 ‘직접 행동 선택’이 아니라 ‘제안 생성’으로 격리해 불안정한 권한 위임을 구조적으로 차단한다.

- **Technical Challenges**: 핵심 기술 난제는 LLM이 제안한 챌린저 정책이 공개 정보만으로 타당한지(숨은 수율·비공개 신호에 기대지 않는지)를 선택 시점 이전에 판정하는 것이다. 이를 위해 CARE는 outcome 공개 이전에 Public-Evidence Intervention Gate로 챌린저와 incumbent를 비교하고, 승인 여부를 사전-기록된 audit log에 남기며, 승인된 경우에만 해당 실험을 실행한다. 또한 LLM이 제안 공간을 넓히되(회복·프런티어 제안 모드) 실제 실행은 게이트가 희소하게 허용하도록 분리함으로써 제어 안정성을 확보한다.

- **Empirical Impact**: CARE는 Minerva/Olympus와 ChemLex(각각 HTE 재현 벤치마크)에서 다른 방법 대비 최고 성능을 보였다. Minerva/Olympus에서는 public incumbent 대비 final-best가 80.0→88.5, ChemLex에서는 83.9→92.1로 개선되었고 best-so-far AUC에서도 최상위를 기록했다. 절제 실험과 개입 로그 분석은 ‘LLM-only 위임’이나 ‘게이트 제거’가 성능 저하로 이어지는 반면, LLM은 제안 공간 확장자로서 유효하게 기여하며 선택 권한은 공개 근거 게이트가 책임진다는 메커니즘을 실증적으로 뒷받침한다.



### SIMMER: Benchmarking Latent Failures in LLM Executable Planning with a World Mod (https://arxiv.org/abs/2606.14574)
- **Prior Approaches**: 기존 평가는 계획이 실행에 성공하는지(성공률)나 참조 계획과의 의미 유사도에 주로 초점을 맞췄다. 하지만 TextWorld·ALFWorld·VirtualHome 같은 가상환경은 단순한 상태표현 탓에 오염, 온도/화학 변화처럼 누적되는 암묵 상태를 제대로 반영하지 못한다. 한편 자연어 유사도 평가는 문장 표면에서는 그럴듯해도 행동 간 상태 의존성이 만들어내는 오류(잠재 실패)를 잡기 어렵다.

- **Core Contribution**: 이 논문은 LLM 계획에서 바로 드러나지 않는 ‘잠재 실패(latent failure)’—즉 전제조건은 만족하지만 암묵 상태 전파로 목표 달성을 망치거나 심하면 되돌릴 수 없는 피해를 일으키는 경우—를 체계적으로 평가하는 SIMMER를 제안한다. SIMMER는 주방 도메인에 기반한 상징적 세계모델로 계획을 실행 검증하며, 즉시 실패와 잠재/비가역 실패를 구분하는 실패 분류와 실행기(상태기계)를 함께 제공한다.

- **Technical Challenges**: 핵심 난제는 ‘전제조건 위반’ 같은 즉시 오류는 물론, 접촉·오염 전파처럼 실행 중 보이지 않게 누적되는 상태 변화까지 추적해야 한다는 점이다. 이를 위해 SIMMER는 77개 행동과 262개 객체로 구성된 PDDL 스타일 세계모델을 만들고, 상태기계 실행기가 단계별로 상태를 갱신한 뒤 실행 후 감사(audit)로 최종 상태의 이상 징후를 탐지한다. 또한 반사실적(카운터팩추얼) 예견 시뮬레이션으로 각 행동 직전에 상태 변화를 예측·점검하도록 프롬프트를 설계해 잠재 위험을 사전에 줄이게 한다.

- **Empirical Impact**: 100개 주방 스크립트(12개 조리 기법)를 6개 LLM에 적용한 결과, 최상위 성능도 에러가 없는 계획은 17% 미만(평균 7.2%)에 그쳤고 잠재 실패가 포함된 계획은 29~52%로 나타났다. 특히 잠재 실패의 상당 부분이 비가역적이며, 다수 사례가 오염 전파를 잘못 시뮬레이션한 데서 비롯됐다. 카운터팩추얼 예견 시뮬레이션은 잠재 실패를 최대 72%, 비가역 케이스를 최대 75%까지 감소시켜, ‘명시적 상태 추론’을 통한 더 견고한 LLM 플래너 방향을 실증적으로 제시한다.



### Regional Climate Model Emulation with Diffusion Approaches: What is the Added Value of Generative Machine Learning? (https://arxiv.org/abs/2606.14570)
Comments:
          Submitted to Journal of Advances in Modeling Earth Systems (JAMES)

- **Prior Approaches**: 기존 RCM 에뮬레이션은 GCM의 대규모 예측변수를 받아 고해상도 강수장을 재현하는 데 초점을 둔다. 특히 결정론적 딥러닝은 단일 강수 지도를 출력해 분포의 꼬리(극한)와 불확실성을 충분히 반영하지 못해, 대체로 과도한 매끈함과 극한 저평가가 발생한다. 생성 모델 중 확산모델은 조건부 분포를 학습해 앙상블을 제공할 수 있지만, ‘주어진 대규모 상태에서 불확실성 엔벨로프가 유용한가’에 대한 체계적 검증은 부족했다.

- **Core Contribution**: 논문은 확산모델 기반 확률적 RCM 에뮬레이션의 추가 가치(불확실성 엔벨로프의 정보성)를 극한 사건까지 포함해 평가한다. 이를 위해 2단계 확산 프레임워크 ParamDiffusion을 제안하고, 기존 단일 지도용 기준모델·분포 파라미터(베르누이-감마) 기반 확률모델·기존 확산모델 CPMGEM과 비교한다. 또한 임계값을 넘는 통계(예: 99퍼센타일)와 공간적으로 누적되는 극한을 함께 보도록 검증 체계를 확장한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 강수가 간헐적이고 양의 치우침이 큰 데다 (2) 그 공간적 구조가 단순 점별 독립 가정으로는 일관되게 생성되지 않는다는 점이다. ParamDiffusion은 먼저 베르누이-감마(Bernoulli-Gamma)로 각 격자점의 건기/강우 강도 분포를 예측해 ‘점별 예측가능 성분과 불확실성’을 분리하고, 이어 확산모델이 그 분포 파라미터를 조건으로 공간적으로 일관된 강수장을 생성해 엔벨로프를 구성한다. 반면 점별 파라미터만 출력하는 B-Gamma는 공간적으로 단절된 패턴이 나타나며, 이 한계가 확산 단계의 필요성을 뒷받침한다.

- **Empirical Impact**: 실험은 perfect model 프레임워크에서 ALADIN63(약 12.5km) 강수장을 목표로 하며, 중장기 통계와 선택된 8개 ‘관심 날짜’(극한 포함)를 통해 모델을 비교한다. 확산 기반 접근(특히 ParamDiffusion·CPMGEM)은 장기 강수 통계와 공간적 상세성을 높은 기술(skill)로 재현하고, 특정 사건에서는 단일 지도 기준모델보다 개선을 보인다. 다만 평가된 모델들은 RCM이 만든 ‘가장 극단적인’ 사건을 불확실성 엔벨로프 안에서 일관되게 포착하지는 못해, 고영향 강수 극한에 대해 신뢰성 있는 확률 표현을 더 개선해야 함을 시사한다.



### Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner (https://arxiv.org/abs/2606.14555)
- **Prior Approaches**: GAP 뒤 선형 분류기는 이미지 수준 라벨만으로 학습되며, 평가도 Top-5처럼 전역 지표가 중심이라 공간 근거가 가려진다. CAM/gradient 기반 설명은 질의한 특정 클래스의 열지도를 만들 수 있지만, 다중 객체가 섞인 장면에서 “어떤 물체가 근거였는가”를 분리해 보긴 어렵다. 또한 MIL 관점의 기존 약지도 학습은 주로 학습 목표나 집계 규칙을 바꾸는 데 초점을 두고, 표준 분류기가 내재한 공간 근거의 진단 가능성은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 GAP+선형 헤드 구조가 클래스 로짓을 공간 피처 그리드 위의 “점별 읽기(dense readout)”로 정확히 분해할 수 있음을 보인다. 이를 MIL(다중 인스턴스 학습)으로 재해석해, 이미지 수준 예측이 틀리더라도 공간에는 정답 클래스 근거가 남아 있을 수 있음을 체계적으로 정리한다. 이후 dense readout을 단순한 사후 진단 도구로 활용해, GAP이 평균화하며 숨기는 공간 단서를 추출한다.

- **Technical Challenges**: 핵심 기술 과제는 “공간 인스턴스”를 정의한 뒤, 점별 선형 헤드 적용이 의미 있는 공간 클래스 점수를 준다는 수학적/실험적 정당성을 확보하는 것이다. 논문은 CNN/ViT에서 얻는 마지막 공간 피처를 인스턴스로 보고, 선형 헤드의 점별 적용 전후의 등가성으로 공간 점수맵을 구성한다. 독립 인스턴스 가정은 CNN의 중첩 수용영역과 ViT의 셀프어텐션 때문에 엄밀히는 성립하지 않지만, 실제 진단 효과가 일관되게 나타나도록 실험 설계를 보완했다.

- **Empirical Impact**: ImageNet 및 occlusion 실험에서 전경 영역에서의 탐지율이 90% 이상으로 높게 나타나, GAP이 가려도 공간 근거는 복원됨을 확인했다. ImageNet-A에서는 이미지 수준 정확도가 급락해도 공간 단서 기반 탐지는 상대적으로 잘 유지되어, 실패가 “집계(평균)로 인한 희석”과 연결됨을 시사한다. MS-COCO에서도 동결된 자기지도 백본에 선형 헤드를 학습해 얻은 공간 로짓이 전경에서 정답 범주 근거를 복구했으며, 합성 다중 객체 실험은 단일 라벨 감독만으로도 이미지 내부의 정답 공간 점수를 형성할 수 있음을 직접 입증한다.



### TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation (https://arxiv.org/abs/2606.14551)
- **Prior Approaches**: 대부분의 비전-로봇 모방 정책은 현재 관측(혹은 짧은 최근 창)만으로 다음 행동을 결정하도록 설계돼, 현재 입력이 충분하다는 가정을 암묵적으로 둡니다. 지연-증거(delayed-evidence) 환경에서는 결정적 단서가 이미 사라져 현재 관측만으로는 분기가 구분되지 않으며, 단순히 기록 길이를 늘리는 방식도 비용이 커지거나 단서가 다른 신호에 덮여 성능이 흔들립니다.

- **Core Contribution**: 이 논문은 지연-증거 모방을 위한 고정 예산(memory-bounded)의 인과적 기억 프레임워크 TRACE를 제안합니다. TRACE는 단서를 직접 저장하기보다, 로봇이 실제로 어떤 경로를 따라왔는지 나타내는 path signature(경로 시그니처)를 키로 삼아 단서가 보였던 시점에 저장된 시각·로봇 상태 증거를 나중의 애매한 분기 지점에서 복원해 올바른 브랜치를 선택하게 합니다.

- **Technical Challenges**: 핵심 난제는 “현재 관측만으로는 부족하지만, 긴 히스토리 전체를 저장하지 않고도” 실행된 인과적 경로에 결합된 단서를 유지하는 것입니다. TRACE는 (1) raw time/라벨 인덱싱을 버리고 (2) 로봇 상태 궤적을 스트리밍으로 요약한 경로 시그니처(누적·증분)를 라우팅 키로 사용하며, (3) 고정 크기 슬롯 메모리에 증거를 선택적으로 쓰고 읽는 어텐션형 게이팅 업데이트를 통해 이 문제를 해결합니다. 또한 정책 백본과 액션 헤드, 모방 손실은 바꾸지 않고 경량 어댑터로 TRACE 메모리 조건만 주입해 플러그인 방식으로 통합합니다.

- **Empirical Impact**: 실세계 장기 조작의 5개 지연-증거 태스크(Tool, Book, Laundry, Cable, Medicine)에서 TRACE는 회귀형(action-chunking)과 확산형(diffusion) 정책 모두에 대해 분기 선택 및 과업 성공(평균 진행도)을 유의미하게 개선했습니다. 특히 단서가 초기에만 보이고 이후 시점에서는 시각적으로 거의 동일해지는 태스크에서 가장 큰 이득이 나타났고, 순수 히스토리 길이/일반 메모리 대비 TRACE가 “실행된 인과 경로에 따라 증거를 라우팅·복원”하기 때문임을 진단 및 절제 실험으로 확인했습니다.



### From Shield to Target: Denial-of-Service Attacks on LLM-Based Agent Guardrails (https://arxiv.org/abs/2606.14517)
- **Prior Approaches**: 기존 에이전트 안전 대응은 금지 키워드 차단이나 손수 작성한 규칙 기반 검문 같은 방식이 주류였지만, 실제 위험은 단일 단어가 아니라 맥락 상호작용에서 생겨 손쉽게 우회됐다. 이후 Llama Guard 같은 분류기 기반 접근은 알려진 공격 유형에는 강했지만, 과업·환경·다단계 상호작용까지 포괄하는 범용성에는 한계가 있다. 이에 따라 등장한 LLM 기반 가드레일은 문맥을 읽고 구조화된 판정을 내려 우회 공격에 강점이 있지만, 그 “가용성” 문제는 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 LLM 가드레일의 추론 능력이 오히려 새로운 취약점인 ‘추론 연장(Reasoning-extension) DoS’로 이어짐을 체계적으로 드러낸다. 공격자는 자연어 페이로드로 가드레일이 원래 하려던 안전분석 스키마를 “정상 데이터”로 오인하게 만들어, 판정이 끝나지 않을 정도로 추론 루프를 길게 만든다. 결과적으로 이는 안전 우회가 아니라, 가드레일이 임계 경로(critical path)에서 병목이 되어 에이전트의 진행 자체를 막는 가용성 공격이다.

- **Technical Challenges**: 핵심 기술 과제는 ‘가드레일의 과업 내 행동’을 꺾는 것이 아니라, 가드레일이 본래 따르는 분석 템플릿(스키마-팔로잉)을 악용해 루프 길이를 극대화하는 페이로드를 찾는 것이다. 이를 위해 저자들은 서러게이트 가드레일에서 추론 길이를 피트니스로 삼는 빔서치 최적화 프레임워크를 제안하고, (1) LLM-as-Proposer와 전략 뱅크로 자연어 기반 다양한 페이로드를 탐색하거나 (2) 기작-인지 구조 돌연변이로 위험 범주·열거 깊이·반(反)지름길(anti-shortcut) 같은 스키마 슬롯을 직접 변형한다. 또한 어텐션 사이클링과 엔트로피 붕괴 같은 루프 징후를 관측해, 단순 문장 그럴듯함이 아니라 구조적 성질이 확장 효과를 만든다는 점을 뒷받침한다.

- **Empirical Impact**: 실험에서 이 공격은 단독(standalone) 평가로 여러 가드레일 아키텍처·안전 템플릿·에이전트 벤치마크에 일반화됐고, 한 오픈소스 서러게이트(TS-Guard-8B)에서 최적화한 페이로드가 8개 주요 백본(Claude, GPT, Gemini, DeepSeek, Qwen 등)으로 전이되어 토큰 관점에서 13–63배 증폭을 보였다. 또한 실제 웹·데스크톱·코드·멀티에이전트 배치에서는 최대 148배 지연 증폭이 관측됐고, 단일 오염 문서가 공유 가드레일 인프라를 포화시켜 동시 실행 에이전트들의 처리량을 떨어뜨리며 연쇄적으로 시스템을 마비시킬 수 있음을 보여줬다. 저자들은 필터·토큰 버짓 같은 완화책들이 직접 해결하지 못하고 실패 모드만 이동하거나 더 긴 루프를 유발할 수 있음을 들어, ‘비용 상한(cost-bounded)’과 ‘추론 견고성(reasoning-robust)’이 시급하다고 결론낸다.



### Securing the Future of IoMT in the Post-Quantum Era: An Edge-Native Federated Learning Approach (https://arxiv.org/abs/2606.14515)
- **Prior Approaches**: 기존 연구는 IoMT에서 보안·프라이버시를 다루되, 대체로 기존 공개키 기반 경량 암호(예: RSA/ECC)에 의존하거나 양자 위협을 장기 관점에서 충분히 반영하지 못했다. 또한 FL 보안 연구는 모델 업데이트로 인한 정보 유출·학습 조작 위험을 논의하지만, 대규모 연산이 가능한 고성능 환경 평가에 치우쳐 자원 제약이 큰 현장 IoMT 적용성이 약했다.

- **Core Contribution**: 이 논문은 PQC(후기 양자 암호, Post-Quantum Cryptography)를 FL-enabled IoMT에 “오케스트레이션-통신-암호처리” 단위로 통합해, 학습 단계의 취약점과 장기 양자 내성 요구를 동시에 겨냥한다. Kubernetes 대안으로 K3s(경량 Kubernetes)를 사용하고, 통신에는 ML-KEM(ML-KEM, NIST 표준 KEM)과 경량 인증 암호 Ascon을 결합한 보안 프레임워크를 제안한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 자원 제약 환경에서 PQC의 계산·통신 오버헤드를 감당하면서도 (2) FL 라운드 동안 암호화/복호화 워크플로가 안정적으로 동작하고 (3) 디바이스 수 증가 시 지연이 폭증하지 않게 만드는 것이다. 논문은 통신 전담 포드를 분리해 암호 연산을 병렬화하고, RabbitMQ 기반 메시징과 quorum queue로 안정성을 보완해 지연과 실패(예: 복호 실패 원인의 키 핸들링 문제)를 줄이도록 설계했다.

- **Empirical Impact**: Raspberry Pi 4의 3대 엣지 K3s 테스트베드에서 분산 구조는 기기 수가 늘어도 키 교환 및 모델 교환 지연이 상대적으로 완만하게 증가했으며, 10대에서 키 교환 약 83%, 모델 교환 약 59% 지연 감소 같은 성과를 보였다. 이는 “양자 내성 암호 + 경량 오케스트레이션 + FL”을 실제 자원 제약 엣지에 배치 가능한 수준으로 증명했다는 점에서 향후 IIoMT(지능형 의료 사물인터넷) 보안 설계의 기준점이 될 의미가 있다.



### Fodor and Pylyshyn's Systematicity Challenge Still Stands (https://arxiv.org/abs/2606.14512)
Comments:
          Accepted in the Transactions of the Association for Computational Linguistics (TACL). This is a pre-MIT Press publication version of the paper

- **Prior Approaches**: 기존 연결주의(뉴럴 네트워크)는 상징주의의 체계성(systematicity) 설명을 바로 제공하지 못한다는 비판을 받아왔다. 특히 Fodor와 Pylyshyn의 ‘체계성 논증’은 인간이 한 문장을 이해할 수 있으면 그에 상응하는 다른 문장도 반드시 이해하는 쌍조건적(biconditional) 의존성을 보인다고 주장하며, 이는 상징 체계에서 기본적으로(by default) 따라온다고 본다. 
최근에는 Brenden Lake와 Marco Baroni가 메타학습으로 합성성(compositionality)을 학습하면 인간 체계성을 재현·설명할 수 있다고 주장했지만, 이 논문은 그 결론이 성급하다고 본다.

- **Core Contribution**: 이 논문은 Lake와 Baroni의 Meta-Learning for Compositionality(MLC)가 체계성 과제를 실제로 해결하지 못한다는 점을 체계적으로 반박한다. 단순히 ‘모델이 일부 정답을 잘 맞춘다’가 아니라, 분포 안팎에서 규칙을 일관되게 학습·적용하는지, 그리고 어떤 규칙 변화에도 체계적으로 반응하는지로 실패를 보여준다. 
저자들은 따라서 Fodor와 Pylyshyn이 제기한 신경망의 체계성 과제가 현재까지 미해결 상태라고 결론짓는다.

- **Technical Challenges**: 저자들이 지목하는 핵심 기술적 문제는 MLC가 메타학습 중 본 데이터의 아주 구체적인 세부사항에 과도하게 민감해, 훈련 분포 밖(조금만 벗어나도)에서 규칙 학습과 일반화가 무너진다는 점이다. 또한 분포 안에서도 ‘체계적 행동’이 관찰되지 않는다고 주장하며, 체계성을 (1) 분포 내 좁은 성능, (2) 무관한 디테일에 대한 민감도, (3) 서로 다른 문법에서도 동일한 체계성을 보일 수 있는가로 세 층위로 나눠 점검한다. 
저자들은 이 세 층위 모두에서 MLC 모델이 비체계적 실패를 보였고, 따라서 “체계성은 생긴다”는 주장에 신뢰를 주기 어렵다고 말한다.

- **Empirical Impact**: 실험적으로 저자들은 Lake와 Baroni가 보고한 ‘gold grammar’뿐 아니라 그 규칙을 소폭 수정한 변형에서도 성능이 흔들리며, 이는 체계성의 형태로 보기 어렵다는 근거로 제시된다. 특히 모델이 규칙의 사소한 변경이나 라벨-의미 매핑 같은 우연적 요소에 좌우되는 경향을 보였다고 하며, 체계성 논증이 요구하는 ‘기본 패턴’ 학습과는 거리가 있다고 해석한다. 
결과적으로 이 논문은 신경망의 합성적 일반화가 체계성을 의미하지 않을 수 있음을 강조하며, 인지과학-머신러닝의 연결 고리에서 ‘체계성 설명’의 기준을 더 엄격히 다뤄야 한다는 파급효과를 갖는다.



### A Fixed-Point Neural Operator for Size- and Functional-Transferable Hamiltonian Prediction (https://arxiv.org/abs/2606.14498)
Comments:
          30 pages, 5 figures, 2 tables

- **Prior Approaches**: 기존 연구는 분자 구조를 기준으로 수렴된 해밀토니안을 한 번에 직접 회귀하거나(Direct Regression), 또는 DEQ처럼 고정점 기반으로 반복·암묵미분을 적용해 self-consistency를 강제하는 방식이 주를 이뤘습니다. 하지만 KS-DFT에서 수렴 해밀토니안은 기하학의 명시적 함수가 아니라 SCF의 암묵적 고정점이라, 한 단계 매핑은 비선형 전자 반응을 통째로 학습해야 해 전이성이 떨어지기 쉽습니다. 또한 최종 상태의 원소별 해밀토니안 손실은 점유 부분공간(occupied subspace)이 지배하는 궤도 에너지·전자밀도 차이를 충분히 반영하지 못한다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 해밀토니안 예측을 “수렴 해밀토니안 회귀”가 아니라 SCF 절차가 유도하는 업데이트 규칙을 학습해 고정점을 찾는 문제로 재정의한 HamEvo를 제안합니다. HamEvo는 neural operator가 단일 SCF 업데이트를 흉내 내고, 그 고정점이 수렴 해밀토니안이 되도록 설계했습니다. 더불어 중간 SCF 궤적을 선학습에 활용하고, 고정점에서 밀도행렬(density matrix) 슈퍼비전을 통해 점유 부분공간을 캘리브레이션해 전자 구조 일관성을 맞춥니다.

- **Technical Challenges**: 핵심 난제는 (1) 수렴 해밀토니안이 고정점 형태로 정의되어 있어 직접 매핑이 전이성을 해치며, (2) 해밀토니안 원소별 오차가 곧바로 궤도 에너지·전자밀도 정확도로 이어지지 않는다는 점입니다. HamEvo는 이를 위해 SCF 중간 궤적에서 H(t)→H(t+1) 전이 자체를 학습하고, 고정점에서는 밀도행렬을 통해 점유 부분공간을 직접 감독하도록 구성했습니다. 아키텍처는 원자쌍 블록 기반의 해밀토니안 분해와 SO(3) 등변성을 반영한 메시지 패싱을 사용해 기하학 변화에 대해 표현을 안정적으로 유지합니다.

- **Empirical Impact**: 실험에서 HamEvo는 MD17부터 drug-like QMugs까지의 벤치마크에서 해밀토니안 오차를 기존 직접 회귀 및 DEQ 계열 기준선 대비 35–49% 낮췄습니다. QMugs에서 HOMO/LUMO 에너지의 MAE는 각각 0.036 eV, 0.053 eV로 화학적 정확도(약 1 kcal/mol) 수준에 근접했으며, 점유 궤도 계수 유사도도 높은 편(평균 0.974)으로 보고됩니다. 또한 few-shot 미세조정(참조 컨포메이션 20개)만으로 최대 122원자 분자까지 확장하고, ωB97 계열 등 교환상관기능 정의에서도 적응 가능하며, 추론 속도는 기존 DFT 대비 최대 242배 빨라 열역학적 분자동역학 샘플링 기반 온도 의존 HOMO–LUMO 갭 변형까지 포착하는 의미 있는 결과를 보였습니다.



### The Perceived Fragility of Explanations in Audio Models: Manipulation of Attribution with Unchanged Predictions (https://arxiv.org/abs/2606.14466)
Comments:
          Accepted to the ICML 2026 Workshop on Machine Learning for Audio: 5 pages, 4 figures

- **Prior Approaches**: 기존 오디오 딥페이크 탐지의 XAI 연구는 Grad-CAM, LRP처럼 사후( post-hoc ) 설명을 제공하는 데 초점을 맞췄지만, 설명이 공격에 얼마나 쉽게 흔들리는지는 충분히 다뤄지지 않았다. 시각(비전) 분야에서는 설명 조작 취약성이 Lp 규범으로 측정돼 왔으나, 오디오는 인간 청각과의 연관성이 낮아 그대로 적용하기 어렵다.

- **Core Contribution**: 이 논문은 오디오 딥페이크 탐지에서 사후 설명(어트리뷰션 맵)을 예측 레이블과 분리해 조작할 수 있음을 보인다. 이를 위해 청각 마스킹(psychoacoustic masking) 임계값을 최적화 손실에 동적으로 넣어, “안 들리는 섭동으로도” 설명 히트맵을 바꾸되 최종 예측은 보존하도록 만든다.

- **Technical Challenges**: 설명 조작은 일반적으로 2차 미분이 필요해 최적화가 불안정해지는데, 논문은 Adam 기반으로 수렴을 확보한다. 또한 스펙트럼 에너지가 마스킹 임계값을 넘지 않도록 audibility 페널티를 구성하고, 추가로 예측 보존을 위한 마진 기반 손실과 파형 진폭 하드 제약을 함께 걸어 목적(설명 이동)과 제약(청감·분류)을 동시에 만족시킨다.

- **Empirical Impact**: SONICS 딥페이크 데이터에서 VGGish, AST, SpecTTTra 등 다양한 아키텍처에 대해 Grad-CAM과 LRP를 공격한 결과, 기존 무제약 PGD는 청질을 크게 망가뜨렸지만 제안한 청각 마스킹 기반 공격은 고음질 지표를 유지하며 설명을 체계적으로 왜곡했다. 특히 AST가 가장 취약했고, SpecTTTra는 상대적으로 강했는데, 이는 모델의 장기 의존성(토큰 기반 주의/attention)이 설명 조향에 더 잘 흔들릴 수 있음을 시사한다.



### MoDiCoL: A Modular Diagnostic Continual Learning Dataset for Robust Speech Recognition (https://arxiv.org/abs/2606.14459)
Comments:
          Accepted at Interspeech 2026

- **Prior Approaches**: 기존 ASR 평가는 잡음, 악센트, 발화 장애, 화자 연령 등 분포 변화 요인을 따로 떼어 측정하는 경우가 많아, 실제로 누적·동시 발생하는 상황을 충분히 반영하지 못합니다. 또한 continual learning(연속 학습)이 ASR에서 분포 드리프트에 대한 ‘강건성’ 형성·전이·망각을 진단하는 방식으로 체계화된 연구는 상대적으로 제한적이었습니다.

- **Core Contribution**: 이 논문은 강건성을 시간이 지나며 진화하는 ‘동적 능력’으로 보고, MoDiCoL(Modular Diagnostic Continual Learning)이라는 모듈형 진단용 데이터셋과 연속 학습 커리큘럼을 제안합니다. MoDiCoL은 언어 내용, 화자 특성, 음향 환경을 요인 설계로 분리해, 강건성이 어떻게 얻어지고 다른 드리프트에 옮겨지며 다시 잊히는지 분석할 수 있게 합니다.

- **Technical Challenges**: 핵심은 (1) 요인이 실제로 함께 나타나는 복합 드리프트를 통제된 순서로 만들고, (2) 논리적으로 불가능한 조합은 합성 발화·증강으로 보정하며, (3) 온라인·스트리밍 연속 학습처럼 현실에 가까운 평가를 수행하는 데 있습니다. 이를 위해 요인 실험 설계를 L27 직교배열과 foldover로 구성해 다수의 실행(run)을 만들고, denoising·불유창 삽입·장애 시뮬레이션·거리·잡음 주입 등의 증강 파이프라인으로 각 수준을 맞춘 뒤, Experience Replay·Representation-level Regularization·Orthogonal Gradient Descent를 비교합니다.

- **Empirical Impact**: 실험에서 초기(미적응) 모델은 화자·언어 드리프트에서 WER이 크게 악화되며, 이는 강건성이 단일 요인에만 민감하지 않음을 시사합니다. 연속 학습 결과로는 Experience Replay가 재학습 안정성과 망각 억제에 가장 효과적이었고, 버퍼 10% 구성이 A-WER 및 낮은 FM/BWT로 강건성 전이를 가장 잘 보였습니다. 또한 OGD가 RLR보다 성능이 좋게 나타나, 성능 저하가 표현 자체의 표류(대표 드리프트)뿐 아니라 기울기 하위공간 내 간섭에서 크게 기인함을 실증적으로 보여줍니다.



### tap: A File-Based Protocol for Heterogeneous LLM Agent Collaboration (https://arxiv.org/abs/2606.14445)
Comments:
          Accepted to KCC 2026. English archival translation. 3 pages, 1 figure, 3 tables

- **Prior Approaches**: 기존 LLM 멀티에이전트 코딩 프레임워크(예: ChatDev, MetaGPT, AutoGen)는 역할 기반 대화나 대화 서버 중심 협업을 제안했지만, 대체로 동일한 런타임·동일 API 계열·벤더 모델 패밀리 같은 전제가 깔려 있습니다. 또 SWE-agent, OpenHands처럼 자동 버그 수정 데모는 많았지만, 여러 공급자의 에이전트가 하나의 저장소에서 직접 메시지를 주고받으며 개발·리뷰를 나누는 문제는 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 서로 다른 벤더의 Claude와 Codex가 공통 메모리나 동일 실행 환경 없이도 같은 코드베이스에서 협업하도록 하는 파일 기반 협업 프로토콜 tap을 제안합니다. tap은 메시지를 우선 파일(마크다운+YAML 메타데이터)로 보존해 추적 가능성을 유지하면서, 동시에 실행 환경별 실시간 알림 경로를 결합해 지연을 줄입니다.

- **Technical Challenges**: 핵심 난제는 에이전트마다 통신 방식과 실행 스타일이 달라 중앙 서버나 공통 채널을 기대하기 어렵다는 점입니다. 저자들은 메시지의 정본을 inbox의 파일로 고정하고, 실시간 알림은 “지연 감소” 용도로만 사용하며 수신 측은 알림을 받아도 파일을 다시 읽어 처리하도록 설계해 알림 실패·재시작 시에도 메시지 유실 없이 복구 가능하게 했습니다. 또한 git worktree로 작업을 분리해 충돌을 막고, instanceId로 로그·상태 파일을 격리해 병렬 실행도 견딜 수 있게 구성했습니다.

- **Empirical Impact**: tap은 27일 동안 37개 generation에 걸쳐 실제 운영 저장소에서 자기 자신을 개발·리뷰하는 데 적용되며 209개의 tap 관련 PR과 717개의 운영 아티팩트를 축적했습니다. 검토 아티팩트 375건을 분석한 결과, 서로 다른 모델 조합(heterogeneous model pairs)에서 결함 또는 수정 요청이 기록될 비율이 69.8%로 동종 조합(53.1%)보다 높았고, 서로 다른 모델·환경이 리뷰 관점을 넓힐 수 있음을 시사합니다. 관찰연구라는 한계는 있으나, 파일 기반 기록이 세대 간 외부 메모리(external memory) 역할을 하며 지속적 협업 맥락을 유지한다는 점이 실증적으로 제시됐습니다.



### CADET: Physics-Grounded Causal Auditing and Training-Free Deconfounding of End-to-End Driving Planners (https://arxiv.org/abs/2606.14438)
Comments:
          8pages 4figures

- **Prior Approaches**: 기존 E2E 자율주행 플래너는 행동 모방(imitaton)으로 학습되어, 데이터에서 함께 나타난 단서(동시발생 상관)를 의사결정 변수로 착각하는 “통계적 지름길”에 취약하다고 지적한다. 평가에 주로 쓰이는 open-loop L2 및 충돌률은 관측 입력을 제거해도 크게 변하지 않아(기준선 효과) 실제로 어떤 입력 단서에 의존하는지(인과 vs 비인과)를 잘 드러내지 못한다.

- **Core Contribution**: 논문은 CADET( Causal Auditing and Deconfounding at Test-time )를 제안하며, 학습된(고정) E2E 플래너를 재학습 없이(test-time) 감사·벤치마킹·수정한다. 핵심은 물리-기하학적 prior(지각 모듈 출력 기반)이 “그 객체가 물리적으로 결정을 바꿀 수 있는가”를 외부 기준선처럼 판단해, 의존도가 높지만 실제 원인이 아닌 단서를 찾아낸다는 점이다.

- **Technical Challenges**: 큰 어려움은 관측 기반 지표나 환경 불변성만으로는 전역적으로 통계 상관이 유지되는 ‘global spurious’ 케이스를 분리하기 어렵다는 것이다. CADET은 (1) 질의별 영향도(교란 시 계획 변화)와 (2) 물리 prior의 게이트, (3) 환경 간 안정성 항을 결합한 PCR(Physics-grounded causal audit) 점수를 만들고, 플래너 추론 시 플래그된 단서만 causal masking으로 억제해 do(·)에 근사하는 반사실적 수정(TCM)을 수행한다.

- **Empirical Impact**: SpurGen(구조적으로 인과/비인과가 라벨로 알려진 합성 벤치마크)에서 단일 신호 기반 방법들은 가정 실패로 정밀도·재현율이 크게 흔들리지만, PCR은 높은 정밀도와 재현율(F1≈0.91)을 보이며 물리 prior의 견고함도 관측 잡음에서 유지된다. 또한 nuScenes의 공개 플래너(SparseDrive)를 대상으로 한 감사가 가능하며, 훈련 없이도 spurious 의존을 구체적으로 플래그하고 TCM 적용 시 반사실적 강건성을 개선하는 방향성을 제시한다.



### Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack (https://arxiv.org/abs/2606.14409)
- **Prior Approaches**: Vision-Language-Action(VLA)은 지속 제어에서 유망하지만, 실제 로봇 배치에는 데이터·학습·적응·실행을 하드웨어 제약에 맞춰 함께 설계해야 한다는 한계가 크다. 기존 텔레오퍼레이션은 마스터-슬레이브 방식 탓에 조작이 비자연스럽고, 촉각(힘) 정보가 빈약해 정밀 조작에 취약하다. 또한 UMI류는 데이터 다양성을 늘리지만 시각·레이블 정밀도(특히 SLAM 의존이나 손끝 힘 전달의 부족)에서 병목이 생기며, 크로스-임보던스 적응은 운동/제어/지각 간 ‘갭’ 때문에 추가 난도가 붙는다. 정책 측면에서는 이산 행동 토큰 기반은 정밀도·속도에 제약이 있고, 연속 제어의 강화학습은 보상·가치 모델 의존이나 배포용 고주파 폐루프 서빙이 기본 목표가 되지 못했다.

- **Core Contribution**: HyVLA-0.5는 VLA를 ‘정책 모델’이 아니라 데이터 수집부터 연속행동 학습, RL 후처리, 실제 로봇 배치까지 잇는 end-to-end 파이프라인으로 통합해 문제를 체계적으로 푼다. 핵심 기여는 (1) 손끝 수준 촉각과 고정밀 행동 레이블을 함께 주는 fingertip UMI+모션캡처 기반 데이터 구축, (2) Mixture-of-Transformers(MoT) 기반 embodied VLM에 conditional flow matching 기반 연속 행동 expert를 결합해 고주파 제어를 가능케 한 모델링, (3) 보상/가치 네트워크 없이도 선호를 통해 실패를 개선하는 FlowPRO를 둔 RL 후처리, (4) 델타-청크를 매끈하게 연결하고 폐루프 실행을 고속화한 배포 프레임워크다.

- **Technical Challenges**: 가장 큰 기술 과제는 ‘사람-로봇’ 임보던스 차이를 줄이면서도 손끝 정밀 조작에 필요한 연속 제어를 안정적으로 학습·실행하는 것이다. 논문은 (a) 행동 레이블을 SLAM이 아닌 광학 모션캡처로 획득해 고충실도 6-자유도 궤적을 제공하고, (b) 정책 출력은 로봇 고유 기구학에 덜 종속되도록 end-effector-frame의 델타-청크로 표현해 임보던스 간 학습을 쉽게 하며, (c) 모델은 연속 행동을 flow matching으로 직접 생성해 디스크리타이즈의 정밀도 손실을 줄인다. RL에서는 보상모델의 취약성을 피하려고 Teleoperation 개입-롤백으로 성공/실패 궤적 쌍을 모아 reward-free Proximalized Preference Optimization(PRO) 계열인 FlowPRO로 오프라인 선호를 정렬하고 망각을 억제하는 성질을 활용한다. 마지막으로 배포에서는 백본 추론과 실행을 비동기 오버랩하고, 델타 청크를 cubic Bézier 스무딩으로 C1 연속 전이로 이어 폐루프 고주파 실행 제약을 만족시킨다.

- **Empirical Impact**: 논문은 Hy-UMI-10K에서 1010K 시간 규모의 시연을 학습 자원으로 삼고, 동종·이종 임보던스에 대한 SFT 트랙(Track A/B)과 이후 FlowPRO 기반 RL 후처리를 통해 정교 조작 강건성과 성공률을 끌어올리는 흐름을 보인다. 특히 실패 사례를 빠르게 반복 학습 루프로 회수해 장꼬리(long-tail) 조작의 성능을 개선하고, 보상/가치 네트워크 없이도 near-ceiling 수준에 가까운 성공률로 향상시키는 점을 강조한다. 또한 델타-청크+비동기 서빙+연속 스무딩 조합으로 실제 하드웨어 폐루프에서 고주파 제어를 구현해, VLA의 ‘모델 성능’에 머물지 않고 배치 가능성을 실증했다. 이 결과는 로봇 일반가자에 필요한 시스템 공동 설계(데이터-학습-배포)의 중요성을 한 단계 끌어올리는 사례로 해석된다.



### Learning to Hear Hesitation: Continual Learning for Disfluency-Aware ASR (https://arxiv.org/abs/2606.14391)
Comments:
          Accepted at Interspeech 2026

- **Prior Approaches**: 기존 자동음성인식(ASR)은 대개 발화를 “정상적으로” 만들기 위해 disfluency(불유창성)를 무시하도록 최적화되어, 정보 손실과 환각(hallucination)이 생기기 쉽습니다. 보완 연구는 불유창성 마커를 포함하거나(예: 표기/토큰) 화자별로 개인화(personalization)하는 방식, 혹은 더 자세한(동일성 높은) 전사(verbatim transcription)를 노리는 방식이 중심이었습니다. 그러나 제한된 소규모 불유창성 데이터로 모델을 추가 학습하면, 일반 도메인 능력이 망가지는 catastrophic forgetting 문제가 자주 보고됩니다.

- **Core Contribution**: 이 논문은 불유창성 인식에 continual learning(계속학습, CL)을 적용해, 불유창성 토큰을 “안정적으로” 학습하면서도 기존 ASR 성능을 유지하는 절차를 제안합니다. 특히 pretrained ASR(whisper-small.en)에 불유창성 마커를 네 가지 토큰 유형으로 도입한 뒤, 불유창성 분포가 다른 데이터셋으로 연속 학습해 적응의 견고함을 실험합니다. 또한 CL 방법별로 나타나는 내부 학습 동학을 비교해, 단순 성능 차이를 넘어 왜 어떤 방법이 더 잘(혹은 덜) 마커를 학습하는지까지 설명하려고 합니다.

- **Technical Challenges**: 핵심 기술 난제는 “토큰(불유창성 마커) 학습”과 “일반 ASR 품질(pWER)” 사이의 충돌로, 소규모 추가학습이 마커를 잘 못 배우거나(언더-인트로듀스) 반대로 전반 품질을 해치는 트레이드오프가 발생한다는 점입니다. 저자들은 네 가지 CL 기법(EWC, ER, A-GEM, WA)을 비교하고, disfluency 토큰이 생성될 때 어떤 attention head가 관여하는지 head-attribution으로 추적했으며, 상위 cross-attention head를 ablation(마스킹)해 원인성을 검증했습니다. 분석 결과, 마커를 성공적으로 내보내는 경우 공통적으로 소수의 decoder cross-attention head가 특화되며, 이 기작이 CL 방법 전반에서 공유됨을 보였습니다.

- **Empirical Impact**: 실험에서 pWER(불유창성/구두점 등을 제거한 기준의 WER)와 마커 F1 사이에 일관된 trade-off가 확인되며, pWER를 가장 잘 유지하는 WA는 마커를 상대적으로 덜 도입하는 경향이, 반대로 마커 F1을 가장 잘 내는 ER는 ASR 품질 희생이 일부 동반되는 패턴이 나타납니다. 순차 적응(먼저 마커 세팅 후 Pitt→Delaware)에서는 ER이 마커의 일반화·유지(최소 forgetting) 측면에서 가장 강했고, WA는 비불유창(clean) 및 불유창 pWER 유지가 가장 안정적이었습니다. 또한 상위 cross-attention head를 제거하면 disfluency(특히 FILLER) 발화가 크게 줄어들면서도 pWER는 유사하게 유지되어, “왜 CL 방법이 마커 학습에 실패하는지”에 대한 기계적 설명을 제공했다는 점에서 의미가 큽니다.



### Discovery under Hypothesis Redundancy: A Geometric Theory of Discovery Bottlenecks (https://arxiv.org/abs/2606.14386)
Comments:
          23 pages, 1 figure, 27 tables

- **Prior Approaches**: 기존 발견 시스템은 후보를 많이 만들거나(LLM) 정해진 연산으로 탐색을 확장(구조화된 로컬 서치)하는 방식에 치우쳐, “새 후보가 아카이브의 독립 방향을 추가하는가”를 체계적으로 진단하지 못했다. 로컬 서치는 상관이 높은 후보가 스펙트럴하게 한정된 하위공간으로 몰릴 때 수익이 급격히 감소하며, 비선형 확장이나 재조합만으로는 그 압축된 공간을 벗어나기 어렵다. 결과적으로 랜덤한 비국소 탐색은 커버리지는 늘려도 예측성과의 연결이 끊겨 생산성이 떨어지는 문제가 있었다.

- **Core Contribution**: 이 논문은 Search Compression Hypothesis로, LLM의 비국소 제안이 유용해지는 조건이 “세 기하학적 조건의 동시 충족”이라고 정식화한다. 핵심은 (1) 스펙트럴 압축(유효 차원 축소), (2) 이미 탐색된 스팬으로부터의 직교적 탈출, (3) 목표 신호에 대한 잔차 신호 정렬(RSA)이 함께 있어야 하이브리드 탐색의 이점이 생긴다는 점이다. 또한 이 조건들이 제거될 경우(각 조건의 개별 필요성) 하이브리드 우위가 사라짐을 정리해, LLM 탐색을 ‘단순 참신함 탐색’이 아닌 ‘언제 방향성 있는 비국소 탐색이 필요한지’ 진단하는 절차로 바꾼다.

- **Technical Challenges**: 가장 큰 기술적 난제는 후보가 직교로 ‘멀어 보이는 것’과 실제로 ‘목표에 예측 가능하게 정렬되어 추가 정보가 되는 것’을 분리해 모델링하는 것이었다. 이를 위해 가설 상관행렬의 스펙트럴 구조로 압축 정도를 정의하고, 로컬 연산은 구조화된 스팬 안에 갇힌다는 기하를 사용해 로컬 수익 감소를 수학적으로 연결했다(명목 차원 대비 유효 차원). 이어서 비국소 점프가 직교 탈출을 만들더라도 RSA가 없으면 수율이 오르지 않는다는 점을 RSA/PredNovelty 개념과 예측 검증(중복·비예측 후보 제거) 연산으로 구현·검증했다.

- **Empirical Impact**: 실험은 통제된 합성 환경에서부터 A-share 팩터 발견(5,647개 종목, 2010–2026)과 상징 회귀 벤치마크(LLM-SRBench 포함)까지 폭넓게 수행되며, 하이브리드 이득이 압축이 심할수록 커지고(약한/표적-보유 방향에서 집중) 공간이 풀랭크에 가까워지면 사라지는 패턴이 반복 확인된다. 특히 시그널 플랜팅과 directed-vs-random 비교에서 ‘참신함(직교성)만으로는 부족’하며, 무작위 점프는 커버리지를 넓혀도 예측 정렬이 없어 수율을 개선하지 못했다. 또한 이 프레임워크의 예산 배분 함의를 점검하는 퍼블릭 탭룰러 sanity check를 통해, LLM-guided 발견을 언제 멈추거나 강화해야 하는 실무적 진단으로 확장할 수 있음을 보여준다.



### Elastic Queries Reinforcement Learning: Self-Aware Policy Execution for VLA Models (https://arxiv.org/abs/2606.14375)
- **Prior Approaches**: 기존 VLA(vision-language-action) 정책은 고정된 추론 스케줄로 실행되는 경우가 대부분이라, 상태의 난이도에 따라 계산을 유연하게 늘리거나 줄이기 어렵다. 특히 생성(denoising) 예산과 재계획 전 행동 청크 길이를 수동 하이퍼파라미터로 두어 쉬운 구간엔 과투자하고 어려운 접촉/정렬 구간엔 피드백이 부족해질 수 있다. 또한 미세조정이나 가벼운 적응을 하더라도, 실행 스케줄 자체는 학습 문제로 다루지 않는 한계가 있었다.

- **Core Contribution**: EQRL(Elastic Queries Reinforcement Learning)은 VLA의 각 쿼리를 탄력적으로 만들기 위해, 고정된 생성기(사전학습된 VLA)는 그대로 두고 쿼리 수준에서 잠재 입력(steering), denoising 예산, 행동 청크 길이를 함께 선택하는 프레임워크를 제안한다. 이를 통해 상태 난이도에 따라 “더 계산할지/덜 계산할지”, “더 짧게 커밋하고 다시 계획할지/길게 오픈루프로 갈지”를 정책이 결정할 수 있게 한다. 학습은 스케줄 전용의 경량 어댑터와 잔차(residual)로 수행되어, VLA 본체의 비싼 파인튜닝 부담을 줄인다.

- **Technical Challenges**: 핵심 난관은 연산(NFE) 비용이 태스크 학습을 압도하거나, 제한 없는 스케줄이 상수/노이즈성 가치로 붕괴할 수 있다는 점이다. EQRL은 여러 critic의 앙상블 불일치로 상태 난이도 신호를 만들고, 그 신호가 “어려운 상태엔 더 보수적으로(더 많은 denoising, 더 짧은 청크)” 계산을 배분하도록 스케줄 우선(baseline+prior)을 제공한다. 또한 청크 길이가 시간 스케일을 바꾸므로 쿼리 수준 거시 행동(macro-action)으로 학습하고, 청크 의존 discount와 에피소드 수준 NFE 예산 밴드를 함께 두어 전체 비용을 제어하면서도 난이도 정렬을 유지한다.

- **Empirical Impact**: LIBERO와 ALOHA 시뮬레이션, 그리고 ALOHA/실로봇 매니퓰레이션(오프라인·온라인)에서 EQRL은 태스크 성공률을 보존하거나 개선하면서도 amortized 추론 비용을 줄였다. 특히 단순히 평균 NFE를 낮추거나, denoising 또는 청크 길이 중 하나만 동적으로 바꾸는 제한 실험에서는 성공 곡선이 약화되어, critic 기반 난이도 인지와 조인트 스케줄링의 결합이 필요함을 보였다. 실로봇에서도 벽시계(latency)와 접촉-재계획의 균형 측면에서 실제 실행 속도 개선 및 안정적인 성능 유지가 관찰되어, “탄력적 계산·재계획 인터페이스”의 실효성을 보여준다.



### No Accidental Software Agent First Canonical Code for Human Code Entropy Reduction and 30 to 500 times Lower Frontier Model Requirements (https://arxiv.org/abs/2606.14357)
Comments:
          36 pages

- **Prior Approaches**: 기존 연구는 코드가 반복적이라는 점(클론·자연스러움)과 대규모 학습 효율을 바탕으로 에이전트/LLM의 소프트웨어 작업 능력을 키우는 데 집중해 왔다. 하지만 인간 저장소의 표현 변이(프레임워크/네이밍/CI 방언/생성물 경계/증거 부족)는 과제로 남아, 토큰·추론·도구 호출·실패 복구가 저장소 “좌표계”를 탐색하는 데 소모된다는 한계가 있다.
또한 중복 제거나 필터링은 텍스트 품질을 개선할 수는 있어도, 행동 동치(behavior equivalence) 관점에서의 잉여 자유도를 학습 목표 자체에서 줄이지는 못한다.

- **Core Contribution**: 이 논문은 프론티어 코딩 모델이 “올바른 물체”를 배우지 못한다는 문제의식 아래, agent-first canonical code(에이전트 우선 정규 코드)라는 proof-carrying 정규화 계층을 제안한다. 행동 오라클을 명시하고, 행동 동치로 소프트웨어 인코딩을 몫(quotient) 처리해 governed representative(관리되는 대표)와 그에 대한 증거·증명 의무를 함께 학습/변환하려는 가설이 핵심이다.
결과적으로 소스·컨텍스트·이유·도구·검증·보안·출처·리뷰·실패 루프까지 포함한 “verified-change 비용”을 같은 오라클 하에서 일괄 비교 가능한 단위로 낮추는 것을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 정규화가 표현상의 우연(entropy)은 줄이되, 숨겨진 테스트·보안·마이그레이션 안전·프로비넌스·리뷰 수용 같은 행동 관련 정보는 보존해야 한다는 “정보 보존” 문제다. 이를 위해 typed change algebra, proof lanes, constrained edit grammars, semantic patch cells, runtime negative memory, proof-carrying change objects 등으로 합법적 수정 경로와 증거 채널을 강제하고, 비보존 항목은 명시적으로 disposition(포기/예외/위험)로 기록하도록 설계한다.
또한 raw 저장소와 canonical 저장소의 효과를 단순 모델 성능이 아니라 같은 모델·같은 리니지·같은 평가 문맥에서 “탐색/분기/실패 루프/검증 성공 비용”으로 분리해 검증 가능하게 만드는 실험 설계가 과제로 제시된다.

- **Empirical Impact**: 논문은 아직 완전한 행동 보존 및 비용 절감의 종합 증거를 확정하진 않으며, 대신 위 가설을 깨뜨릴 수 있는(falsifiable) 평가 게이트를 제안한다. 예비 QLoRA 실험에서 Qwen2.5-Coder-14B는 64,088개의 canonical trajectory 학습이 가능하고, 테스트된 금지 언어 마커를 억제하는 신호를 보였지만 행동 보존·스케일링 경제성·verified-change 비용 절감은 입증하지 못했다.
즉 현재 임팩트는 “데이터 서브스트레이트를 quotient(행동 동치 몫)으로 바꾸면 에이전트 비용이 줄어들 것”이라는 계획 가능한 연구 프로그램을 제시했다는 점에 있으며, No-Accident Horizon(우연 감소의 상한) 관점에서 향후 검증이 기대된다.



### PLAIground: SLO-Driven Runtime Model Selection for Compound AI Systems in the Edge-Cloud-Space Continuum (https://arxiv.org/abs/2606.14356)
- **Prior Approaches**: 기존 연구는 복합 AI(Compound AI) 워크플로에서 모델을 고르는 문제를 주로 배포 시점 최적화나 단일 추론 서빙 수준에서 다뤘다. LangChain, DSPy 같은 프레임워크는 작업 정의와 모델 구현을 강하게 결합해 두어서 모델 포맷·입출력·API 차이 때문에 실행 중 전환을 자동화하기 어렵다.

- **Core Contribution**: 이 논문은 PLAIground라는 프레임워크로, 작업 의미(태스크)를 특정 모델 구현에서 분리해 실행 중 모델 전환이 가능하도록 만든다. 핵심은 Compoundable AI Model(CAIM) 추상화로, Task Contract·Data Contract로 워크플로 로직을 고정하고 System Contract로 후보 모델을 런타임에 선택할 수 있게 한다.

- **Technical Challenges**: 기여를 실제로 작동시키려면, (1) 모델 전환 시에도 입력·출력 형식이 깨지지 않게 인터페이스를 정규화하고 (2) 정확도·지연·비용·에너지 같은 여러 SLO를 동시 만족시키며 (3) 조건 변화에 따라 모델을 오락가락 없이 안정적으로 바꿔야 한다. 논문은 Pixie 알고리즘을 통해 CAIM 단위로 창(window) 기반 관측치를 SLO 임계값과 비교해 업그레이드/다운그레이드를 하되 쿨다운으로 흔들림을 줄이고, 워크플로 레벨 예산은 CAIM별로 사전 분해해 런타임 의사결정에 반영한다.

- **Empirical Impact**: 평가는 화재 탐지(에너지 제약)와 질의응답 라우팅(정확도·지연·비용 동시 제약) 두 개의 현실적인 워크플로에서 이뤄졌다. Pixie는 고정 모델 전략 대비 비용·지연 예산을 지키면서도 정확도 목표를 충족했으며, 화재 탐지에서 최대 91.3% 유효 정확도를 달성하고 비용은 최대 21배 절감(또는 지연 2.5배 개선에 해당) 효과를 보였다. 무엇보다 정확도·지연·비용 SLO를 동시에 만족한 유일한 전략으로 보고돼, 3D 컴퓨팅 컨티뉴엄 같은 자원 변동 환경에서의 운영 가능성을 강화한다.



### Design Methodology and Performance Trade-offs Management for Distributed and Compound AI Systems (https://arxiv.org/abs/2606.14350)
- **Prior Approaches**: 기존 모델 중심(model-centric) 배포는 설계 시점에 단일 단일 모델을 고정하고, 모든 입력에 동일한 추론 경로와 동일 자원을 적용한다. 또한 개선 수단이 주로 모델 스케일링(파라미터/데이터/연산 증대)에 몰려 있어, 입력 난이도·도메인에 따른 계산 분해나 런타임 최적화가 어렵다. 더불어 모델 내부 지식은 학습 시점에 고정되어, 외부 지식·정확한 계산·안전성/검증 같은 요구를 단일 모델만으로 안정적으로 처리하기 힘들다.

- **Core Contribution**: 이 논문은 모델을 단위로 삼는 대신, 여러 모델·알고리즘·툴을 워크플로로 조율하는 시스템 중심(system-centric) 설계를 제안한다. Compound AI를 위한 설계 공간을 워크플로 토폴로지(흐름 구조)와 구성 선택(모델 배치·런타임 파라미터) 두 축으로 정리하고, 모놀리식 배포의 핵심 한계를 겨냥한 8가지 설계 패턴을 제시한다. 이를 통해 accuracy·latency·cost 같은 SLO를 런타임에서 더 유연하게 맞출 수 있는 구조를 제공한다.

- **Technical Challenges**: 문제는 패턴이 조합될수록 설계 공간이 조합적으로 폭증하고, 어떤 모델을 어떤 역할에 배치하고 어떤 임계값·샘플 수·검색 깊이를 둘지에 따라 성능이 크게 달라진다는 점이다. 논문은 패턴을 구조적 역할(예: Router/Cascade, Sampler/Verifier/Aggregator, Retriever/Tool Executor, Guardrail)로 추상화해 워크플로 토폴로지를 고정하고, 그 위에서 구성 선택으로 accuracy-latency-cost 운영점을 이동시키도록 방법론을 구성한다. 특히 라우팅 임계값과 검색 깊이처럼 파라미터가 상호작용하는 경우까지 실험적으로 특성화해, 단일 파라미터 튜닝이 아닌 공동 최적화의 필요성을 보여준다.

- **Empirical Impact**: 3개의 비전·언어 도메인 케이스 스터디에서 Compound AI 구성은 모놀리식 기준선에 근접하면서도 지연시간을 최대 60%, 비용을 최대 71%까지 절감했다. 예를 들어 CODEC는 엣지-클라우드 선별 오프로딩(Router)과 시간적 융합(Aggregator)으로 단독 엣지/단독 클라우드가 제공하지 못하는 운영점을 만들어 mAP@50을 높이면서도 latency·cost를 줄였다. QARouter는 Retriever+Router 조합으로 F1을 끌어올렸고, InferScale은 Sampler 패턴(test-time compute scaling)으로 더 작은 모델의 반복 샘플링만으로 정확도를 재현하며, “수동 프로토타입에서 SLO 준수의 자동 발견·유지”로 가는 향후 로드맵 과제도 5개로 제시한다.



### Squeeze-Release: Iterative Pruning with Exact Structural Minimization (https://arxiv.org/abs/2606.14346)
- **Prior Approaches**: 기존 비정형 프루닝은 중요도에 따라 개별 가중치를 0으로 만들지만, 배포 모델은 텐서의 모양이 그대로라 실제 런타임 연산량과 메모리가 크게 줄지 않는 경우가 많다. 또한 대부분의 보고 지표(마스크-얼라이브 파라미터 수)는 “실행 가능한 더 작은 모델 크기”와 직접 대응되지 않아, 일반 GPU/하드웨어에서의 속도 절감과 간극이 생긴다. 구조적 프루닝은 이 간극을 피하지만, 동일 압축 수준에서는 비정형 방법이 보통 더 높은 정확도를 제공한다는 주장도 함께 제시된다.

- **Core Contribution**: 이 논문은 마스크/프루닝으로 생긴 0 구조를 실제로 제거하는 “정확한 구조 재작성(minimization)”을 제안한다. minimization은 부동소수점 반올림 오차 범위 내에서 원래 네트워크의 순전파 함수를 동일하게 유지하면서, 마스크 버퍼를 없애 더 작은 “배포용 밀집(dense)” 네트워크로 변환한다. 여기에 반복 루프인 Squeeze-Release를 더해, compaction 이후 남는 정확한 0 위치의 용량을 작은 보정 잡음(캘리브레이션된 노이즈)으로 다시 학습 가능하게 만들고, 한 번의 프루닝-축소 패스로는 찾기 어려운 구조적 중복까지 연쇄적으로 축소하도록 유도한다.

- **Technical Challenges**: 핵심 난제는 “0이 생겼다”는 정보만으로는 채널/뉴런 단위의 물리적 축소가 항상 가능하지 않다는 점이다(대부분은 row/column/filter 전체가 아니라 내부 일부만 0인 경우). minimization은 특정 단위가 dead-incoming(행 전체 0) 또는 dead-outgoing(다음 계층에서 읽히지 않는 열/입력 0)일 때만 정확히 접어서 제거하는 방식으로, 순전파를 함수 보존적으로 재작성한다. 또 LayerNorm이 채널 평균·분산을 채널 축에서 계산하기 때문에 채널 제거 시 통계가 깨지는 문제가 생기는데, 이를 CompensatedLayerNorm이 제거된 채널의 개수·합·제곱합 같은 충분통계를 저장해 “정확한 LayerNorm 출력”을 복원하는 방식으로 해결한다.

- **Empirical Impact**: Squeeze-Release는 fully-connected 모델 네트워크에서 비축소 대비 39배 더 작은 배포 모델로 압축하면서 정확도는 비슷한 수준을 보고한다. ConvNeXt-Tiny 같은 현대 CNN에서는 14.8배 축소를 달성하며, 비교적 유사한 정확도를 유지한다. 또한 재작성(minimization)이 Transformer 아키텍처로 확장될 수 있음을 증명해, 배포 비용 절감 관점에서 기존 반복 프루닝의 “마스크 지표 과대평가” 문제를 더 현실적인 모델 크기·연산량으로 연결한다는 점에서 의미가 있다.



### I'm Sorry Driver, I'm Afraid I Can't Do That: Appraising the Safety of LLMs within Automotive Contexts (https://arxiv.org/abs/2606.14327)
Comments:
          Accepted at the Dependable AI in Embedded Systems (DAIES) Workshop at SAFECOMP 2026; 15 pages, 3 figures, 2 tables

- **Prior Approaches**: 자동차 분야에서 LLM은 인포테인먼트부터 실시간 의사결정/설명, 자율주행 보조까지 넓게 탐색돼 왔습니다. 다만 공개 연구들은 주로 자연어 입력을 제어 명령으로 바꾸는 기술적 가능성에 초점을 맞추며, ISO/PAS 8800·ISO 21448 같은 안전 규격이 요구하는 보증(assurance)까지는 충분히 다루지 못합니다. 특히 폐쇄형 LLM을 실제 차량 아키텍처에 얹을 때 필요한 평가·주장 구조가 약합니다.

- **Core Contribution**: 이 논문은 “상류(upstream) 모델 개발”과 “하류(downstream) 차량 통합 맥락”을 분리해 안전 보증의 이중 과제를 정식화합니다. 그리고 지연(latency)과 가치 정렬(value alignment)을 축으로, 기존 접근이 안전 주장에 어디서 빈틈을 남기는지 GSN(Goal Structuring Notation) 스타일의 안전 논증 패턴을 제안합니다. 마지막으로 Talk2Drive 기반 케이스 스터디로 위험 실패 모드(지연·거부/정렬 오류)를 드러내며, 향후 LLM 관련 위험 사건을 다루기 위한 보증 메커니즘의 방향성을 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 두 가지입니다: (1) 상류에서 개발된 범용 LLM의 행동이 하류의 차량 목표(기능 안전)에 부합하는지 보장하기 어렵다는 점(정렬 관련), (2) 상용 수준 모델의 연산·추론이 실시간 제어의 시간 상한을 초과할 수 있다는 점(지연 관련)입니다. 저자들은 Talk2Drive의 STT→LLM→스크립트 실행 파이프라인을 그대로 두고 모델만 교체해, 평균 지연뿐 아니라 최악 꼬리 지연과 거부 응답 같은 정렬 실패를 관찰하는 방식으로 문제를 실증합니다. 또한 chain-of-thought 성향 모델은 응답 꼬리가 커져 제어 환경에 부적합할 수 있음을 보여 줍니다.

- **Empirical Impact**: Talk2Drive 실험에서 일부 모델은 평균 응답은 수용 가능해도, 최악 지연이 기능 안전 관점에서 치명적 장애가 될 수 있었습니다. Gemini의 경우 reasoning 성격 때문에 최대 지연이 매우 커져 실시간 제어에 부적합했고, GPT·Claude 등에서도 스크립트 일치도(기준선 대비)와 STT 특이 케이스, 그리고 “정지 요청에 거부” 같은 정렬 오류가 반복적으로 확인됐습니다. 결과적으로 기존 기법의 안전 주장은 현재 상태에선 불완전하며, 차량 안전 분야에서 LLM 통합을 논하려면 지연 꼬리와 가치 정렬의 운영적 검증을 안전 논증 프레임 안에 포함해야 한다는 메시지를 강화합니다.



### Achieving Precise Text-To-Cypher Via Grounded Knowledge Graph Data Generation (https://arxiv.org/abs/2606.14325)
- **Prior Approaches**: Property Graph를 대상으로 한 대화형 질의 응답은 Text2Cypher(텍스트→Cypher) 파서를 요구하지만, 이를 위한 고품질 데이터와 라벨이 부족하다는 문제가 반복돼 왔다. 기존에는 대형·상용 언어모델에 의존하거나, 제한된 수의 수작업/소량 데이터로 미세조정해 성능 격차를 메우려는 접근이 많았다. 그 결과 로컬 배포 환경에서는 정확도와 비용(또는 데이터 수집/주석 캠페인) 간 트레이드오프가 크게 나타났다.

- **Core Contribution**: 이 논문은 Text2Cypher 파인튜닝을 위해 자동 합성 데이터 생성 방법을 제안한다. 이를 통해 작은 LLM도 Property Graph 질의를 안정적으로 Cypher로 변환하도록 만들며, 대형 상용 모델과 경쟁 가능한 성능을 목표로 한다. 특히 로컬 배포가 필요한 환경에서 데이터 주권을 확보하면서도 추가 비용이 큰 주석 캠페인 없이 정확도를 유지하는 방향을 제시한다.

- **Technical Challenges**: 합성 데이터는 ‘그럴듯한’ 예제가 아니라 벤치마크 수준의 정답 Cypher를 제공해야 하며, 그래프 스키마·관계·제약을 정확히 반영해야 한다. 또한 작은 LLM이 합성 데이터에 과적합하지 않고 일반화하도록 학습 신호를 설계하는 문제가 있다. 논문은 이러한 요구를 만족하도록 합성 데이터 생성 파이프라인을 구성하고, 이를 소형 LLM 파인튜닝에 직접 활용하는 전략으로 해결한다.

- **Empirical Impact**: 저자들은 주요 Text2Cypher 벤치마크 전반에서 실험을 수행해, 제안한 합성 데이터 생성 방식이 소형 LLM의 성능을 유의미하게 끌어올린다고 보고한다. 그 결과 일부 설정에서는 대형 상용 모델에 근접하거나 경쟁하는 수준까지 도달해, 로컬 배치 제약에서도 정확도를 잃지 않을 수 있음을 보여준다. 데이터 소버린티와 비용 효율을 동시에 달성할 수 있다는 점에서, 실무형 그래프 질의 시스템의 구축 장벽을 낮추는 의미가 크다.



### Transforming Shape Schemas with Composable Property-Graph Queries (Extended Version) (https://arxiv.org/abs/2606.14309)
- **Prior Approaches**: 기존에는 SPARQL CONSTRUCT와 RDF를 대상으로, 기술한 스키마(형상, shape) 제약이 질의 변환 결과에 대해 논리적으로 포함(따라서 출력 스키마 추론)된다는 접근이 Description Logics(서술 논리, DL)로 정식화돼 왔습니다. 하지만 property graph(속성 그래프)는 라벨과 키-값 속성, 그리고 일급(first-class) 간선 같은 PG 전용 표현력이 있어 DL만으로 그대로 다루기 어렵습니다.

- **Core Contribution**: 이 논문은 ProGS(ProGraph Shapes, 속성 그래프 형상 언어) 입력 shape와 G-CORE 질의 변환을 바탕으로, 모든 가능한 출력 그래프에 대해 만족해야 할 출력 shape를 추론하는 절차를 제시합니다. 특히 그래프 인스턴스에 의존하지 않고 입력 shape를 만족하는 “어떤 그래프에서든” 성립하는 출력 shape를 정적으로 계산할 수 있게 합니다.

- **Technical Challenges**: 핵심 난제는 DL이 property graph의 속성 어노테이션과 일급 간선을 직접 표현하기 어렵다는 점입니다. 논문은 이를 해결하기 위해 중간 RDF 레이어와 재구성(reification)을 도입해 PG를 RDF로 변환하고, ProGS↔SHACL 매핑 및 G-CORE↔SPARQL CONSTRUCT 매핑을 함께 설계한 뒤, 효율적인 DL reasoner로 출력 shape를 계산합니다.

- **Empirical Impact**: 이 방식은 변환 질의에 대한 출력 shape의 건전성(soundness)과, 매핑된 질의/shape 의미가 동등(semantic equivalence)함을 메타이론으로 보장해 실무에서의 신뢰성을 높입니다. 또한 질의 파이프라인을 반복 실행해야 하는 타입 시스템·검증·개발자 디버깅 같은 use case에서, 출력 shape를 한 번만 추론해도 재사용 가능한 기반을 제공한다는 점에서 의미가 큽니다.



### Thinking Outside the [Chat]Box: Bridging Computer Science and Industrial Design for Cognitive-Inclusive Generative AI (https://arxiv.org/abs/2606.14306)
- **Prior Approaches**: 기존 GenAI 접근은 대부분 채팅창 중심이라 사용자가 목표를 자연어로 정확히 ‘프롬프트 엔지니어링’해야 하고, 긴 답변은 인지 과부하를 유발한다. 특히 지적장애(ID) 사용자는 질문을 구체화하기 어렵고, 생성 정보의 신뢰도를 확인할 방법이 부족하며, 기술 용어를 다듬거나 단순화하는 경로도 제한적이었다. 또한 인지 접근성(HCI)과 투명성·개인화의 필요성은 연구되어 왔지만, 채팅 모델을 넘어서는 ‘상호작용의 큰 틀’을 비교·재구성한 사례는 상대적으로 적었다.

- **Core Contribution**: 이 논문은 ID를 위한 인지 접근성을 위해 ‘프롬프트를 넘어선’ 대안적 인터랙션 모델을 제안하고, 이를 구조적(Structural)·경험적(Experiential) 두 층의 지원 체계로 정리한다. 컴퓨터과학(구조적 워크스페이스)과 산업디자인(속도·주의·감정적 몰입 중심)을 각각 다른 학생 코호트에 같은 요구사항으로 설계시켜, 공통 요구사항(초기 캘리브레이션, 선제적 프롬프팅, 응답 조각 직접 조작)과 상호보완적 아이디어를 도출했다. 결과적으로 “무엇(신뢰·탐색 가능성)”과 “어떻게(주의·페이싱·다중감각)”를 함께 제공해야 한다는 프레임을 제시한다.

- **Technical Challenges**: 핵심 난제는 신뢰성(환각 가능성)을 인터페이스 차원에서 가시화하면서도, 사용자 입력 장벽과 정보 과부하를 동시에 낮추는 설계를 구현하는 것이다. CS 측은 신뢰도 지표(예: Reliability Traffic Light), 출처 표시, 응답별 신뢰 분해, 용어 자동 글로서리 블록, 대화 문맥의 계층화(일반 히스토리·메시지별 이동)로 구조적 스캐폴딩을 구체화했다. 산업디자인 측은 빈 입력 상태의 마비를 ‘Guided Mode’로 줄이고, 체크박스 기반 단계별 공개·Focus Mode로 주의·속도를 제어하며, 슬라이더·프리셋 등으로 응답의 톤/형식을 즉시 조정하는 경험적 스캐폴딩을 제안했다.

- **Empirical Impact**: 실증은 ID 사용자가 직접 참여한 형태는 아니지만, 전문가 수준의 비교가 가능한 고충실도 프로토타입과 사용자 중심 설계 절차를 통해 설계 경향을 정리했다는 점에서 의미가 있다. CS의 투명성·탐색성 장치와 산업디자인의 주의·페이싱·다중양식 설계가 함께 갈 때 ‘대화형 채팅’보다 더 인지적으로 포용적인 상호작용 공간이 가능하다는 시사점을 제공한다. 향후에는 전문가 오디트, ID와 함께하는 공동 설계, 기술 타당성 검증(예: 환각 감지·운영체제 연동의 제한), 그리고 종단적 사용자 평가로 실제 효과를 확인하는 로드맵을 제시한다.



### Pix2Pix-Hybrid: Structure-Guided Conditional Synthesis of Hajj Crowd Images with Multi-Channel Conditioning and Weak Attribute Supervision (https://arxiv.org/abs/2606.14297)
- **Prior Approaches**: 기존 군중 계수 연구는 기준이 되는 실제 데이터셋(예: ShanghaiTech, UCF-QNRF 등)에 크게 의존하지만, Hajj처럼 극단적 밀집·가림·조명 변동이 큰 도메인에서는 주석 부족과 도메인 격차로 일반화가 어렵다. 데이터 희소성을 보완하려는 증강은 대개 원본과 강하게 상관된 변형에 그쳐 장면 구조·맥락 다양성을 충분히 늘리지 못한다. 합성 데이터 생성에서도 일반 생성 모델은 고품질은 가능해도 모드 붕괴나 데이터 과적합 위험, 그리고 후속 계수 학습에 필요한 “구조 보존” 목표 정렬이 부족할 때가 많다.

- **Core Contribution**: 이 논문은 Pix2Pix 계열을 확장한 구조-가이드 조건부 GAN인 Pix2Pix-Hybrid(P2P-H)를 제안해, Hajj 장면의 지배적 기하(레이아웃)를 유지하면서 외형을 밀도·시간대 같은 약한 속성으로 다양화한다. 핵심은 무작위 잠재벡터로 자유 생성하는 모델이 아니라, 입력 조건을 엣지·그레이스케일과 밀도/시간대에서 “셀프 페어링”으로 만들고 RGB를 재구성·변환하도록 학습해 계수에 필요한 구조 충실도를 확보하는 데 있다. 또한 생성 품질을 위해 다중 해상도 PatchGAN 판별과 GAN 학습 안정화(지각 손실·특징 매칭·적응형 정규화/증강)를 함께 통합한다.

- **Technical Challenges**: Hajj 도메인은 (1) 주석이 거의 불가능하고 (2) 프라이버시 제약으로 데이터 수집·공개가 제한되며 (3) 고밀도 가림과 해상도/조명 편차가 커서 일반적인 조건부 GAN이 흐림·체커보드·학습 불안정에 빠지기 쉽다. 저자들은 공개 Hajj 영상에서 프레임을 모아 정제·익명화(얼굴 블러)하고, 밀도(낮음/중간/높음)와 시간대(아침/오후/밤)를 자동 약라벨로 추출해 수동 라벨 부담을 줄였다. 생성 모델은 U-Net 생성기 + 서로 다른 해상도를 보는 두 개의 다중 스케일 PatchGAN으로 전역 일관성과 국소 질감을 동시에 강제하며, 적응형 데이터 증강과 복합 손실로 제한 데이터에서도 학습 안정성을 높였다.

- **Empirical Impact**: 이 과정을 통해 9.93e2장의 실제 프레임 학습으로부터 1만 장 규모의 고해상도 합성 데이터 CrowdH를 구축했고, Pix2Pix 및 StyleGAN2-ADA 대비 구조 보존 관점에서 조건부 합성 품질이 개선되었다. 또한 CrowdH-Mix-469(실제 384 + 선택 합성 85)로 5개 계수 모델을 평가한 결과, 합성 데이터를 포함한 학습이 MAE를 모든 모델에서 낮췄으며 CSRNet에서 이득이 가장 컸다. 특히 합성 데이터의 선택성이 효과를 좌우함을 보여주면서, 프라이버시 민감 환경에서도 계수 성능을 실질적으로 끌어올릴 수 있음을 시사한다.



### AgentCyberRange: Benchmarking Frontier AI Systems in Realistic Cyber Ranges (https://arxiv.org/abs/2606.14295)
- **Prior Approaches**: 기존 사이버 보안 벤치마크는 주로 CTF 풀이, 단일 취약점 재현, 익스플로잇 생성처럼 개별 스킬을 따로 평가하는 경우가 많습니다. 그 결과 현실적인 침투 절차(노출 서비스 탐색→초기 발판 확보→내부 정보 수집→다중 호스트 확장)를 end-to-end로 관찰하기 어렵다는 한계가 있었습니다. 또한 공개 인프라가 제한적이라 다중 호스트 사이버 레인지에서 재현 가능한 공격 평가를 수행하기가 쉽지 않았습니다.

- **Core Contribution**: AgentCyberRange는 개방형(open) 다중 레인지(multi-range) 인프라로, frontier AI의 자율적 공격 능력을 현실적인 사이버 레인지 조건에서 측정하도록 설계됐습니다. 15개 실제 웹 애플리케이션의 110개 취약점과, 내부 호스트 156개를 포함한 8개 엔터프라이즈 유사 사이버 레인지를 결합해 “웹 익스플로잇”과 “포스트 익스플로잇” 두 단계를 함께 평가합니다. 또한 실행·오케스트레이션·결과 수집·검증을 담당하는 Cage 파이프라인을 공개해, 모델/프롬프트/예산을 맞춘 재현 평가가 가능해졌습니다.

- **Technical Challenges**: 핵심 기술 과제는 현실적인 침투 흐름을 유지하면서도 자동 검증 가능한 과제로 만들고, 동시에 실행 규모를 확장하는 데 있습니다. 저자들은 웹 트랙에서 숨겨진 엔드포인트 탐색과 검증 가능한 영향(실제로 트리거되는 효과)에 초점을 두고, 포스트 트랙에서는 멀티 호스트 토폴로지·권한 상승·권한/자격 증명 재사용·측면 이동·탐지/방어 압력까지 포함하도록 범위를 구성했습니다. Cage는 서로 다른 에이전트 하네스들을 어댑터로 통일하고, 격리된 컨테이너 실행과 evidencing 기반 검증(예: PoC의 카나리 문자열 확인, 호스트의 /tmp 및 /root 마커 점검)을 통해 일관된 평가를 구현합니다.

- **Empirical Impact**: 실험에서는 6개 frontier AI 시스템을 동일한 조건(프롬프트·예산 매칭)에서 비교했으며, GPT-5.5( Codex )가 웹 익스플로잇 16.1%, 포스트 익스플로잇 31.7%로 가장 높은 성공률을 보였습니다. 힌트를 더 구체화하면 웹 33.0%, 포스트 46.3%까지 상승했는데, 이는 현실 공격 단계형 과제가 여전히 어렵지만 어느 정도 자동화가 이미 가능함을 시사합니다. 더 나아가 벤치마크 밖의 미지 취약점 탐지, 방어 회피를 위한 페이로드 변형 같은 관찰도 보고되어, 공개 사이버 레인지 기반 평가가 신종 오펜시브 위험을 조기에 포착하는 데 필요하다는 메시지를 강화합니다.



### Hierarchical ODE: Learning Continuous-Time Physical Prototypes for Early Link Failure Detection (https://arxiv.org/abs/2606.14284)
Comments:
          International Conference on Machine Learning 2026

- **Prior Approaches**: 기존 시계열 프로토타입 학습은 RNN/LSTM 같은 이산 구조에 의존해 연속 물리 과정을 불연속 상태 전이로 근사하는 경향이 있었다. 이 때문에 미분(벡터장) 기반의 연속 역학 정보를 충분히 학습하지 못해, 매끄러운 감쇠 트렌드와 일시적 잡음이 관측에서 얽히는 문제가 남았다. 또한 클러스터 수를 미리 정하는 폐쇄집합(closed-set) 가정은 실제 환경의 미지 프로토타입 다양성을 제대로 반영하지 못했다.

- **Core Contribution**: 이 논문은 관측이 불규칙하고 잡음이 큰 상황에서도 잠재 상태의 연속 진화를 학습하기 위해 neural ODE(신경 ODE)를 인코더-디코더에 도입한다. 동시에 위계적(hierarchical) ODE 클러스터링으로 프로토타입을 연속 궤적 기반으로 발견해, 잡음은 줄이고 물리적 패턴은 분리하는 것을 목표로 한다. 더불어 적응형 계층 메커니즘으로 사전에 정해진 클러스터 수 없이도 프로토타입 개수를 자동 결정한다.

- **Technical Challenges**: 핵심 기술 난제는 첫째, 불규칙 샘플링에서 연속 역학을 일관되게 복원하면서 관측 주입으로 인한 오류를 제어하는 것이다. 논문은 잠재 상태를 신경 ODE의 적분 곡선으로 모델링하고, 관측 시점에서 GRU 게이팅으로 예측 prior를 보정해 연속성과 정합성을 동시에 확보한다. 둘째, open-world에서 프로토타입 수를 고정하지 않으면서 과/부 분리를 막기 위해, 계층적 응집 클러스터링의 덴드로그램을 로컬 임계값 탐색으로 잘라 구조적 K를 만든 뒤 하위 K-means로 정밀 프로토타입을 분해한다.

- **Empirical Impact**: 실세계 사무공간의 proactive network handover(사전 네트워크 핸드오버) 데이터에서, 제안 방법은 FRR(오탐보다 미탐에 해당하는 false rejection rate)을 전 시나리오에서 가장 낮추면서도 FAR(불필요한 핸드오버 위험)도 경쟁적으로 유지했다. 특히 고잡음 시나리오에서 이산 기반 모델들이 감쇠를 잡음으로 오해해 미탐하는 경향이 있는데, 신경 ODE 기반 연속 진화 학습이 이를 완화했음을 보여준다. 결과적으로 반응적 모니터링에서 사전 예측(early link failure detection)으로 의사결정 패러다임을 전환할 수 있는 실증 근거를 제공한다.



### DIFF-ERO: A Conformance-Aware Loss for Deep Learning in Process Mining (https://arxiv.org/abs/2606.14283)
Comments:
          Accepted at the 24th International Conference on Business Process Management

- **Prior Approaches**: 이 분야의 대표적 학습 목표는 크로스엔트로피 같은 국소 다음-활동(로컬 next-step) 정확도에 맞춰져 있다. 그 결과 모델은 토큰 단위 정확도는 높아도, 루프·동시성·선택 같은 제어흐름 구조를 충분히 반영하지 못해 전역적 적합성(conformance)이 흔들릴 수 있다.

- **Core Contribution**: DIFF-ERO는 엔트로피 기반 확률적 적합성(entropic stochastic conformance)을 미분 가능(differentiable)한 손실로 재구성해 학습 과정에 제어흐름 정보를 직접 포함한다. 기존의 크로스엔트로피가 주로 재현율적(precision/recall 관점에서 재현율에 가까운) 신호를 주는 데 비해, DIFF-ERO는 구조 적합도에 뿌리를 둔 정밀도 성격의 신호를 보완하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 이산적인 적합성 계산(리플레이, 정렬 등)을 신경망의 역전파용 연속 손실로 바꾸는 것이다. 논문은 배치 단위 확률 전이행렬을 만들고(직접-후속 관계 기반) 이를 소프트 엣지(soft edge memberships)로 표현해, 구조적 정밀도·재현율 신호가 그대로 backpropagation에 흐르도록 했다. 또한 확률적으로 유효한 자동자 형태를 위해 소프트맥스 이후 행(row) 정규화를 적용하고, 바이트(bits)항의 이산성 문제는 스무딩된 배치 수준 정규화 항으로 완화했다.

- **Empirical Impact**: Transformer encoder-decoder 기반 다음-활동 예측에서 DIFF-ERO는 다른 손실·타깃들과 비교해 구조가 중요한 구간에서 예측 성능을 개선한다. 동시에 학습된 확률적 오토마톤이 구조적 정답(ground truth) 쪽으로 수렴하는 정성/정량 지표를 보여, 네트워크가 공정 모델 구조를 내부화하고 있음을 시사한다. 즉, 로컬 정확도만이 아니라 전역적인 제어흐름 적합성까지 함께 다루는 학습 목표로서의 의미가 있다.



### Robust Fall Recovery for Armless Bipedal-Wheeled Robots Via Force-Guided Learning (https://arxiv.org/abs/2606.14270)
Comments:
          8 pages, 6 figures, accepted by IEEE Robotics and Automation Letters (RA-L)

- **Prior Approaches**: 모델 기반 접근은 사전 계산된 궤적(모션캡처/최적화 등)을 실행하는 방식이라 결정적이지만, 초기 자세 변화에 민감하고 실제에서 마주치는 다양한 낙하 포즈에 대한 일반화가 약합니다. 강화학습·모방학습 기반 접근은 성공률과 강건성을 개선했으나, 계단형 학습/커리큘럼·단계별 보상 설계에 의존해 조기 수렴(로컬 최적점, dead point) 문제가 남아 있습니다. 특히 팔이 없고 추가 다리 지지가 없는 이족 바퀴형 로봇은 상체/다른 지지 수단 없이 다리 구동만으로 버텨야 해서, reward 중심 학습이 더 쉽게 무너집니다.

- **Core Contribution**: FTSR는 낙하 복구를 위해 외부 보조힘을 시뮬레이션 학습 중 ‘키-연동 외력(높이 상관)’으로 구성하되, 이를 단계적으로 줄이는 커리큘럼이 아니라 CPO의 제약(optimizable constraint) 형태로 명시적으로 최적화 가능하게 만듭니다. 또한 서기(자세 다듬기)에서 걷기(지속 보행)로 전환되도록 height-progressive stage-wise rewards를 배치해, 보조 개입 감소 이후에도 자세 안정과 이후 동작을 같이 학습하게 합니다. Teacher-student 구조로 힘의 효과와 복구 동역학에 관한 특권 정보를 증류해, 팔 없이도 내부 복구 전략을 빠르게 형성합니다.

- **Technical Challenges**: 핵심 난제는(1) 보조힘을 줄이는 과정에서 reward-only 학습이 dead point로 조기 붕괴하는 것, (2) 외력이 사라진 뒤에도 서기 자세를 안정적으로 유지하며 지속 보행까지 이어지게 하는 것, (3) 특권 힘 정보를 쓰되 실제 실행은 관측(자기 정보)만으로 가능하게 하는 것입니다. 논문은 힘·토크를 로봇 중심부 높이와 시간 진행에 연동해 제약 조건으로 ‘물리적으로 feasible한 복구 궤적’ 쪽으로 탐색을 제한하고, 높이 통계 기반 임계값으로 보상 단계를 자동 전환해 학습 안정성을 높였습니다. 더불어 teacher는 접촉력·자세·힘 관련 정보를 처리하고 student는 과거 관측 이력만으로 같은 표현을 학습하도록 설계해, 제약 유도 학습의 효율과 전이성을 함께 확보합니다.

- **Empirical Impact**: FTSR는 시뮬레이션에서 다양한 지형(경사, 단차, 요철, 거친 표면 등)과 랜덤화된 초기 자세에 대해 강건한 낙하 복구를 달성하며, 단계별 구성요소(ablation) 중에서도 force-guided 제약과 height-progressive 보상이 성능에 결정적임을 보입니다. 특히 팔 없는 이족 바퀴형 로봇에서 보조힘을 제거한 뒤에도 안정적으로 서기·걷기 능력을 유지하고, 명령 속도 변화에도 높은 성공률을 보였다고 보고합니다. 나아가 23-DOF 휴머노이드에 대해서도 유사한 회복 성능과 더 빠른 학습(적은 반복으로 부드러운 복구)을 실험적으로 확인해, 현장 적용 가능성과 일반화 잠재력을 뒷받침합니다.



### ChronoID: Infusing Explicit Temporal Signals into Semantic IDs for Generative Recommendation (https://arxiv.org/abs/2606.14260)
- **Prior Approaches**: 생성형 추천에서는 OneRec, MiniOneRec 같은 방식이 항목 의미를 의미 ID(Semantic IDs)로 이산화해 토큰 생성으로 추천을 통합한다. 하지만 기존 Semantic ID 학습은 시간 정보를 텍스트 임베딩 기반 양자화에 의존해 ‘시간 비명시적(time-implicit)’으로만 영향을 주며, 시간은 세션 구성 휴리스틱이나 순서/위치 인코딩, 최적화 단계에서만 간접 반영된다.
그 결과 동일한 항목이 서로 다른 시간 맥락에서 등장해도 같은 의미 ID로 매핑되어, 시간에 따른 의미·관련성·사용자 의도 변화까지 표현하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Semantic ID 자체에 명시적으로 시간을 주입해야 한다는 문제의식을 바탕으로, 시간 인지(time-aware) 의미 ID 학습 프레임워크 ChronoID를 제안한다. 핵심은 시간을 의미 추상화(양자화/코드북) 단계에 넣는 ‘어디에/어떻게’의 설계 공간을 정리하고, 그에 따른 아키텍처 전략을 체계적으로 비교하는 것이다.
또한 시간 누수 없이 평가하기 위해 time-explicit 생성 추천 벤치마크를 새로 정의해, 시간에 따른 분리 학습/검증/테스트 프로토콜을 제공한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 시간 신호를 의미 ID 학습에 적절히 표현·퓨전하는 것인데, 이를 위해 ChronoID는 시간 인코딩(절대 시각 vs 상대 시간 간격), 퓨전 순서(early fusion vs late fusion), 양자화 구조(Residual vs Parallel)라는 세 축으로 설계 공간을 분해해 탐색한다.
또한 의미 ID가 이산 토큰으로 변환된 뒤 생성 모델에 입력된다는 점을 고려해, 절대 timestamp보다 상대 시간 Δt를 쓰는 편이 상호작용 리듬을 더 잘 반영할 수 있음을 실험적으로 확인하고, late fusion과 parallel quantization이 이종 입력 정렬과 식별력 향상에 유리함을 보인다.

- **Empirical Impact**: 실험에서 ChronoID의 time-explicit 의미 ID는 기존 생성/판별 기반 추천 기준선보다 일관되게 높은 HR@K와 NDCG@K를 보이며, 특히 MiniOneRec 대비 HR@3에서 큰 상대 개선이 보고된다. 시간 인코딩을 상대 시간으로 두고 late/parallel 설계를 적용할수록 성능이 더 좋아져, 단순 입력 증가가 아닌 ‘시간-텍스트 의미의 풍부화’가 이득의 원천임을 뒷받침한다.
나아가 시간 신호를 제거하거나 제로패딩으로 대체하면 성능이 크게 하락하고, 원자 수준의 timestamp만으로도 계절·명절 같은 고수준 패턴을 충분히 내재화할 수 있음을 보여 의미 ID 기반 생성 추천의 평가·설계 관점을 재정립한다.



### When and How Severely: Scenario-Specific Safety Envelopes for Driving VLAs (https://arxiv.org/abs/2606.14238)
- **Prior Approaches**: 기존에는 센서 잡음(또는 입력 변형)에 대한 평균 성능 저하나 단일 기준선(예: ADE가 임계치 이하)을 ODD(Operational Design Domain) 경계로 사용해 왔다. 또한 VLA(vision-language-action)에서 설명(Chain-of-Causation) 변화는 “이진(바뀜/안 바뀜)”으로만 다뤄져, 실패가 일어났을 때의 ‘심각도 분포’ 형태는 충분히 분해되지 않았다. 이런 집계 지표는 시나리오마다 임계치가 다르고 실패가 이산적인 심각도대로 뭉치는 현상을 가릴 수 있다.

- **Core Contribution**: 이 논문은 SOTIF(ISO 21448) 관점에서 VLA의 ODD를 “한 개의 잡음 허용치”가 아니라 두 축으로 명시한다. 첫째, 시나리오별로 기준선(ADE) 대비 15% 예산을 처음 초과하는 시점(when)을 σ*(s) 형태로 제시한다. 둘째, 실패를 유발한 경우 설명이 바뀐 샘플들에서 궤적 오차가 어떤 심각도 밴드로 분포하는지(how severely)를 6개 밴드로 분해해, 같은 평균 오차라도 고심각도 비율이 달라질 수 있음을 보여준다.

- **Technical Challenges**: 기여를 위해서는 (1) 시나리오별 안전 임계치를 신뢰도 있게 추정하고 (2) 실패의 오차 분포를 이산 밴드 구조로 복원하며 (3) 두 결과를 동일한 코퍼스에서 교차 결합해야 한다. 논문은 clip-단위 부트스트랩으로 σ*(s)의 신뢰 구간을 확보하고, CoC가 바뀐 5,443개 조건에 대해 궤적 L2 오차 분포를 GMM(Gaussian Mixture Model)으로 피팅해 BIC로 k=6을 선택했다. 마지막으로 σ*(s)와 고심각도(C4/C5) 비율 P(C4∪C5|coc_changed)을 함께 매핑해, 단일 집계 임계치가 드러내지 못하는 불일치(과보호/과소보호)를 통합적으로 드러낸다.

- **Empirical Impact**: 15,968개 (clip, attack)에서 평균 기반 보수적 집계(σ≤50)는 일부 시나리오를 과도하게 막고, 다른 시나리오는 여전히 취약할 수 있음을 확인했다. 특히 σ≤70까지도 안전 임계치가 유지되는 시나리오가 존재하지만, 실패 시에는 다른 시나리오들이 더 많은 고심각도(C4/C5)로 치우친다(예: STOP_SIGNAL은 LANE_KEEPING 대비 C4/C5 비율이 약 4배). 결과적으로 배치 가능한 SOTIF 스타일 ODD 스펙은 각 위험(해저드)에 대해 단일 숫자 허용치가 아니라 ‘시나리오별 임계치 + 실패 심각도 비율’이라는 2차원 안전 엔벨로프가 필요하다는 점을 실증적으로 뒷받침한다.



### Selective Agentic Recovery for UAV Autonomy with a Persistent Mission Runtim (https://arxiv.org/abs/2606.14219)
Comments:
          17 pages, 2 figures. Preprint

- **Prior Approaches**: 기존 에이전틱 AI 기반 로봇/드론 연구는 언어 모델의 추론을 행동으로 연결하는 API·툴 사용, reason-act-observe 루프 등을 제안했지만, 물리 UAV에서는 호출 빈도(주기적/상시)나 전문가 규칙에 의존하는 경우가 많았습니다. 또한 원격 출력이 곧바로 비행 명령으로 라우팅되면 검증과 안전 필터링이 어려워져, 선택적 호출(admission) 설계의 런타임 부담이 남아 있었습니다.

- **Core Contribution**: 이 논문은 Persistent Mission Runtime(PMR)으로, 드론의 미션 루프와 안전·실행 권한은 로컬에서 유지하면서 원격 에이전틱 추론을 ‘온디맨드 복구 모듈’로만 호출하도록 정리합니다. 원격 reasoner의 출력은 미리 정의된 복구 스킬 중 하나로 제한되고, 파싱·로컬 검증·안전 차단·실행 매핑을 거친 뒤에만 비행에 영향을 줍니다. 더불어 호출의 필요성을 판단하는 Learned Cognitive Value of Invocation(learned-CVI)로 비용 대비 효용이 클 때만 호출되게 합니다.

- **Technical Challenges**: 핵심 난제는 원격 호출이 지연·토큰 비용·백엔드 불확실성을 키우는 만큼, ‘언제’ 호출해야 단기 회복이 충분히 이득이 되는지 정량화하는 것입니다. 논문은 진행 정체/막힘 같은 런타임 증거를 포함한 압축 18차원 의미 요약 기반의 admission 게이트(learned-CVI)를 학습해, 단기 복구 유틸리티가 로컬 계속 실행을 이길 확률을 추정합니다. 여기에 쿨다운·예산·터미널 반경 억제·하드 stuck 가드 등 고정 런타임 가드로 무분별한 과호출을 억제하며, 호출된 출력은 스킬 계약 기반의 닫힌 구조로 검증 가능하게 만듭니다.

- **Empirical Impact**: Gazebo/PX4 400회 고정 시뮬레이션(8개 시나리오)에서 PMR은 로컬 전용의 hard/ambiguous Clean Success@1m 5.0%를 95.0%로 끌어올렸고, one-shot/periodic 호출 대비 각각 20.0/32.5%p 높은 성능을 보였습니다. 또한 수동 규칙 기반 기준선 대비 원격 reasoner 호출 횟수 16.7%, 로그 토큰 29.2%를 줄이면서도 성공률을 유지해 효율적 ‘희소 호출’의 이점을 실증했습니다. Crazyflie 나노 쿼드콥터 실험에서도 blocked navigation에서 10/10 Clean Success@1m를 달성해, 의미 요약 기반 감지에서도 적용 가능함을 보여주었습니다.



### Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback (https://arxiv.org/abs/2606.14218)
- **Prior Approaches**: 기존 VLA 및 로봇 world model 연구는 주로 힘·토크 데이터의 부재로 ‘순응’ 학습을 약하게 다뤘습니다. 또한 ALOHA·GELLO 같은 원격조종 데이터 수집은 관절 토크를 노출하지 않아 능동 순응(지금의 저항에 따라 힘을 조절)과 리더-팔/팔 사이 저항의 실시간 햅틱 피드백이 제한됩니다. UMI 그리퍼처럼 손으로 힘을 느끼는 방식도 있지만, 주로 엔드이펙터 중심 기록이라 충돌 제약이 빡빡한 환경에서 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Universal Manipulation Exoskeleton(UME)을 제안해, 원격조종 중 실시간 힘·토크(관절 토크 시그널)를 함께 기록하고 조작자가 투명한 햅틱 토크를 느끼게 합니다. 그 결과 토크 모달리티를 학습에 직접 활용해 능동 순응 정책을 만들고, 전완/어깨/손목 전체 관절 구성(whole-arm configuration)까지 함께 써서 제약 공간에서의 전신·양손(bimanual) 조정을 강화합니다. 더 나아가 보편적 리타게팅(retargeting)으로 서로 다른 로봇 팔(예: OpenArm, Franka, X-ARM)까지 동일한 조작 경험을 옮길 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술 난제는 ‘투명한 햅틱 토크’와 ‘로봇 간 보편 리타게팅’이 동시에 성립하도록 제어·기록 체계를 설계하는 데 있습니다. UME는 사람 팔의 3-1-3 구조를 고려한 기구 설계와 쿼지-다이렉트 드라이브 모터(낮은 감속비)로 실시간 저항 전달을 노렸고, 동역학/야코비안 기반의 로버스트한 토크 역변환과 중력·원심·코리올리·마찰 보상을 포함해 투명성을 높였습니다. 아울러 어깨(가상 구면), 팔꿈치(1자유도), 손목(가상 구면)으로 체인을 분해해 각 부분을 독립적으로 리타게팅하며 특이점 근처의 추적 불안정을 줄였습니다.

- **Empirical Impact**: 실험에서는 토크 모달리티와 whole-arm 기록의 효과가 태스크별로 뚜렷하게 나타났습니다(박스 밀기/뒤집기, GPU 집기, 냉장고 음료 회수 등). 토크 정보를 제거한 No-torque 및 엔드이펙터만 쓰는 UMI 대조군은 제약·충돌 상황에서 상태 구분 실패나 충돌로 성패가 크게 떨어졌고, UME는 장시간 모바일 조작에서도 높은 성공률(예: 대다수 태스크에서 0.85~0.95)을 보였습니다. 또한 데이터 수집 처리량이 향상되어(박스 뒤집기에서 토크 미지원 대비 3.3배, 사람 속도의 71%) 향후 ‘순응 정책 학습’의 실용성과 확장성에 의미가 큽니다.



### From Prompts to Responses: Dual-Sided Data Leakage and Defense in Split Large Language Models (https://arxiv.org/abs/2606.14210)
Comments:
          18 pages, Accepted at ICML 2026

- **Prior Approaches**: 기존 Split-LLM(Head-Body-Tail) 연구는 주로 중간 표현에서 입력 프롬프트가 유출되는 문제를 다뤘으며, inversion 계열 공격이 널리 보고됐다. 그다음으로는 생성 응답(output response) 유출 가능성이 일부 탐구됐지만, 생성 과정에서 양쪽(model head와 model tail)에서 정보가 함께 새어 나오는 ‘이중 단(dual-ended) 유출’의 체계적 취약성은 충분히 규명되지 않았다. 방어 또한 head 중심 노이즈/정규화가 많아 tail을 직접 겨냥하는 공격에는 한계가 있었다.

- **Core Contribution**: 이 논문은 Split-LLM 생성에서 입력 프롬프트와 생성 응답이 동시에 복원될 수 있음을 보이며, Patched Model Inversion with Dual-Sided Initialization(PIDI)라는 2단계 공격을 제안한다. 공격은 Dual-Sided Initialization으로 입력과 응답을 거칠게 각각 초기 추정한 뒤, Patched Model Inversion(PMI)으로 긴 시퀀스에서도 임베딩을 패치 단위로 정교화해 복원 품질을 크게 끌어올린다. 또한 이런 양끝 유출 위협을 동시에 줄이기 위해 Adapter 기반 DualGuard with Mutual Information Defense(ADMI) 방어를 설계한다.

- **Technical Challenges**: 핵심 기술 난관은 (1) forward-only 생성 환경에서 입력과 응답 양쪽의 정보가 순차적으로 누적되며, (2) 긴 생성 길이에서 임베딩을 통째로 최적화하면 학습·추정이 불안정해진다는 점이다. PIDI는 응답에는 모델 completion 성질을, 입력에는 semi-white-box forward inversion을 각각 초기화로 사용하고, PMI 단계에서는 인과 구조를 활용해 임베딩을 여러 패치로 쪼개 단계적으로 최적화한 뒤 마지막에 전체를 함께 조정한다. ADMI는 head의 입력 유출을 상호정보 정규화로 억제하고 tail의 model completion 성질은 모델 거리 정규화로 깨는 동시에, adapter 모듈과 국소 warm-up을 통해 성능 저하를 최소화하는 방식으로 두 난관을 함께 다룬다.

- **Empirical Impact**: 다양한 태스크와 모델에 대한 실험에서 ADMI는 PIDI 및 기존 최첨단 inversion 계열 공격에 대해 일관되게 방어 성능을 보였고, 동시에 과제 성능(유틸리티) 하락도 제한적이었다. 특히 dual-ended로 동시에 노리는 공격을 대상으로 한 방어임에도, 구조 변경을 과도하게 요구하지 않는 학습 설계가 실용성을 높인다는 점이 강조된다. 결과적으로 Split-LLM 보안 평가가 ‘head 단일 유출’ 중심에서 ‘생성 전체의 양끝 동시 유출’로 확장돼야 함을 강하게 시사한다.



### MeEvo: Metacognitive Evolution Combined with Natural Evolution for Automatic Heuristic Design (https://arxiv.org/abs/2606.14202)
- **Prior Approaches**: 기존 LLM 기반 AHD는 크게 두 갈래로 나뉩니다. 자연 진화(Natural Evolution)는 교차·변이를 통해 휴리스틱 코드를 탐색하지만 추론(Reasoning) 흔적을 세대 간에 계승 가능한 정보로 남기지 않아 전략적 지식이 소실됩니다. 메타인지 진화(Metacognitive Evolution)는 반성으로 추론을 정교화하지만 단일 계층의 국소 탐색에 머물러 인구(population) 기반의 재조합을 통한 근본적 새 아키텍처 발견이 어렵습니다.

- **Core Contribution**: MeEvo는 자연 진화와 메타인지 진화를 동등한 두 계층으로 결합하되, 공유 이력(history)을 매개로 상호작용하게 설계합니다. 자연 진화는 코드와 함께 추론 흔적·적합도·오류를 기록하고, 메타인지 진화는 이 누적 이력을 읽어 더 나은 휴리스틱을 생성해 다음 사이클의 부모 풀에 다시 넣습니다. 이를 통해 “인구 기반 탐색”과 “추론 기반 정교화”가 번갈아 강화되도록 합니다.

- **Technical Challenges**: 핵심 난제는 두 계층이 필요로 하는 조건이 상충한다는 점입니다. 자연 진화의 교차는 안정적인 부모 풀을 필요로 하고, 메타인지 진화의 반성은 여러 세대에 걸친 충분한 이력으로 잡음과 진짜 패턴을 구분해야 하므로 병렬 수행이 비효율적일 수 있습니다. MeEvo는 계층을 NN세대/ MM세대로 순환 교대하고, 자연 진화 내부에서는 교차 확률이 초기 탐색에, 변이 확률이 수렴 구간의 정교화에 치우치도록 동적 스케줄링을 적용해 탐색-활용 균형을 구조적으로 맞춥니다.

- **Empirical Impact**: 다섯 가지 최적화 문제에서 두 가지 LLM 백본을 사용한 실험 결과, MeEvo는 기존 LLM 기반 AHD 대비 더 강하고 더 안정적인 성능을 보였으며 특히 복잡한 제약이 있는 과제에서 격차가 컸습니다. 또한 ACS와 WSN 같은 실세계 문제에서 다른 방법들이 “거의 실행 불가능”한 휴리스틱에 막힐 때도 MeEvo는 일관되게 실행 가능한 휴리스틱을 생성해 탐색을 지속할 수 있음을 보여줍니다. 이는 추론 흔적을 계승 가능한 정보로 다루는 이중 계층 설계가 실용 최적화에서도 검색 효율과 해 품질을 함께 끌어올릴 수 있음을 시사합니다.



### OdysSim: Building Foundation Models for Human Behavior Simulation (https://arxiv.org/abs/2606.14199)
Comments:
          34 pages. Code: this https URL ; Models and data: this https URL

- **Prior Approaches**: 기존 인간 행동 시뮬레이션 벤치마크와 학습은 이론심리, 사회상호작용, 역할놀이, 사용자 행동 등으로 조각나 있어 전체적인 진척을 한눈에 추적하기 어렵다. 또한 SFT나 RLHF처럼 ‘도움됨’에 치우친 후처리는 모델을 지나치게 동의적이고 균질한 조력자 톤으로 끌어 Sim2Real gap을 키우며, 단순 프롬프트만으로는 인간의 바람직하지 않은 행동 다양성을 재현하기 어렵다.

- **Core Contribution**: OdysSim은 대규모 행동 기반(behavioral foundation) 모델을 만들기 위한 가장 큰 공개적이고 체계적인 조사로, 모델이 인간 행동을 “규모로” 시뮬레이션하도록 설계한다. 이를 위해 5개 능력 축(C0NV, SS, COG, ROLE, EVAL)으로 62개 데이터셋과 23개 벤치마크 태스크를 통합하는 Soul(Simulation Of human-Like behavior) 프레임워크와, 이를 반영한 SOUL-Index 평가체계를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 사회적 맥락(화자/역할/의도/목표)이 텍스트에 명시되지 않은 데이터가 많다는 점이며, 그래서 저자들은 각 대화를 백생성(back-generated) 방식으로 인물 프로필과 상호작용 목표 같은 사회적 그라운딩을 덧씌운 OdysSim 코퍼스를 구축한다. 이후 midtraining(행동 분포 선행 적응)→태스크별 RL(각 축의 보상 신호에 최적화, 필요 시 LLM judge의 텍스트 피드백을 학습 정보로 사용)→전문가 증류로 단일 8B 모델 OSim을 만들며, 이 단계 구성이 서로 다른 보완 효과를 내도록 설계된다.

- **Empirical Impact**: 결과적으로 공개 8B OSim은 SOUL-Index의 23개 태스크 중 8개에서 1위(또는 공동 1위)이며, 다른 단일 프런티어 모델보다 ‘태스크 수 기준’에서 우세하다. 대화·사회적 과업에서 특히 강점이 나타나고 생성의 길이/형식/어휘 선택이 더 인간답게 바뀌며, τ-벤치의 OOD 사용자 시뮬레이션에서도 반응 정렬(React 93.2 vs 93.5)에 거의 근접한다. 한편 LLM-as-judge RL이 reward-hacking 패턴을 유발할 수 있음을 보여주고 이를 완화하는 탐지기(detector)를 함께 제시해 후속 연구에 방향을 제공한다.



### Robustness without Wrinkles: Parallel Simulation and Robust MPC for Certified Deformable Manipulation (https://arxiv.org/abs/2606.14188)
- **Prior Approaches**: 변형 물체(로프·클로스) 제어는 고차원 상태, 복잡한 비선형 동역학, 그리고 언제/어디서 접촉할지의 결정 때문에 어렵다. 기존 방법은 기하학적 단순화로 계획을 가능하게 하지만 결과가 개방고리로 끝나 견고성이 떨어지고, 학습 기반 방식은 강건한 안전 보장 없이 느린 학습이 필요하며, 접촉을 미분가능하게 만들면 그라디언트 불연속으로 접촉 모드 탐색이 막히는 문제가 있었다.
또한 출력 피드백에서의 안전·강건 MPC는 상태공간이 큰 변형 물체에 스케일하기 어렵거나, 계산이 느려 실시간 적용이 제한적이었다.

- **Core Contribution**: CORD-SLS는 접촉을 매끄럽게(smoothing) 처리한 GPU 병렬 미분가능 시뮬레이터를 핵심으로, 그라디언트 기반 계획을 접촉을 통과해 수행할 수 있게 한다. 여기에 GPU 병렬 출력 피드백 강건 MPC를 결합해 모델·센서 불확실성 하에서도 제약을 만족하는 도달 가능성(reachability) 기반 계획을 실시간으로 만든다.
추가로 conformal prediction을 통해 시각 피드백의 불확실성을 캘리브레이션하고, 이를 MPC의 도달 튜브(tube) 폭으로 반영해 높은 확률의 안전 제어를 달성한다.

- **Technical Challenges**: 가장 큰 기술 난제는 ‘접촉의 전환(활성/비활성, 마찰 접촉의 상이한 상태)’이 만들어내는 비연속성과 소실 그라디언트로 인해, 접촉 발견이 그라디언트 최적화에서 막히는 점이다. 논문은 접촉 활성 조건과 보완성(complementarity) 제약에 대해 접촉 스무딩을 적용해 계획용 그라디언트를 정보성 있게 만들고, 실제 실행(전개 롤아웃)은 비스무스(비부드러운) 동역학을 사용해 물리 충실도를 유지한다.
또한 출력 피드백 SLS의 계산 비용을 줄이기 위해 관측기(옵저버) 재귀를 GPU 병렬 prefix scan 형태의 연산으로 재구성해, 장시간 예측 구간에서도 밀리초급 합성을 목표로 한다.

- **Empirical Impact**: 시뮬레이션과 실제 하드웨어에서 로프·클로스의 고차원·접촉 다발 과제(장애물 회피, 라우팅, 폴딩, 평탄화 등)를 평가하며, 전반적으로 기준선 대비 안전성·속도·과제 성공률에서 우수함을 보인다.
특히 계획이 밀리초(ms) 단위로 동작해, 접촉이 빈번히 발생하는 조작에서도 실시간 강건 제어가 가능함을 실증한다.
또한 미분가능 시뮬레이터를 사용한 model-based reinforcement learning은 analytical policy gradients 덕분에 샘플 효율을 높여 신경 조작 정책 학습에도 의미 있는 가속 효과를 보인다.



### Learning Urban Access Costs from Origin-Destination Flows via Inverse Optimal Transpor (https://arxiv.org/abs/2606.14157)
Comments:
          Oral Presentation. 2026 International Conference on Urban AI

- **Prior Approaches**: 도시 서비스(교육·의료·사회서비스)는 공공-민간이 혼합된 시설망에서 보조금과 공간 배치에 따라 이용이 갈린다. 기존에는 관측된 OD(출발-도착) 흐름을 예측하는 중력모형·이산선택모형 등이 주로 쓰이지만, 실제로 사람들을 움직이게 한 ‘잠재 비용 함수’까지 역으로 복원해 정책 질문(보조금이 체감 접근성을 얼마나 늘리는가)에 직접 답하기는 어렵다. 특히 보조금이 혼잡을 완화하려면 ‘가격/제도 접근성’뿐 아니라 ‘공간적 도달 가능성’이 함께 변해야 하지만, 이를 정량화하는 해석 가능한 지표가 부족했다.

- **Core Contribution**: 이 논문은 학교 선택 문제를 엔트로피 정규화 최적수송(OT)의 관점에서 보고, 관측된 학교 간 전이 흐름으로부터 역최적수송(inverse OT) 방식으로 ‘선택 비용’을 추정한다. 또한 두 가지 상보적 비용모델을 제시해, 하나는 해석 가능한 거리 구간(piecewise) 모델로 보조금 항을 명시하고, 다른 하나는 미분 가능한 Sinkhorn 전달을 이용한 신경 비용모델로 더 정교한 거리-보조금 상호작용을 학습한다. 추정 결과로 보조금이 체감 이동비용을 얼마나 상쇄하는지의 정책용 지표인 보조금-동등 거리 λ(k)를 제안한다.

- **Technical Challenges**: 역최적수송의 핵심 난관은 ‘관측된 흐름’만으로부터 비용함수의 모양을 유일하게/안정적으로 복원하는 것이다. 논문은 (1) 도로망 거리와 보조금을 포함한 구간별 해석 모델에 대해 log-domain Sinkhorn 순전파를 결합한 MAP 최적화를 수행하고, (2) 신경 비용모델은 미분 가능한 Sinkhorn 연산을 통해 역전파로 비용파라미터를 학습해 데이터 적합도를 개선한다. 두 모델의 비교를 통해 해석 가능성과 미세한 상호작용 학습 사이의 균형을 점검한다.

- **Empirical Impact**: 필리핀 교육부 데이터(2022~2024, 가장 인구가 많은 지역)에서 28만 명 수준의 학습자 이동을 23,820개 관측 흐름으로 집계해 실험했으며, 해석 모델은 지표가 직관적으로 정책에 연결되도록 λ(k)를 산출한다. 예를 들어 1,000페소 보조금 증가는 거리 5~15km 구간에서 체감 이동비용을 약 6.07km 상쇄하는 것으로 해석되며, 구간별로 상쇄 크기가 달라 ‘균일 보조금의 공간적 효과 불균형’을 시사한다. 신경 비용모델은 기준 적합도를 더 낮춰(목적함수 Φ 감소) 더 세밀한 거리 민감도 패턴을 보여주되, 구간 단순화가 장거리 마찰을 과대/왜곡할 수 있음을 드러내 도시 접근성 기반 보조금 설계와 시설 배치 계획에 활용될 수 있다.



### Learning High Coverage Discriminative Parsimonious Rulesets (https://arxiv.org/abs/2606.14156)
- **Prior Approaches**: 기존 규칙 기반 분류 학습은 RIPPER, CN2 같은 순차 커버링이나 연관분류처럼 많은 규칙을 만들기 쉬워 해석 가능성을 해친다는 한계가 있었다. 또 최근 방법들은 정확도는 높아도 커버리지가 낮아(High accuracy–low coverage) 입력 공간의 상당 부분을 기본 규칙에 의존하게 만든다는 문제가 확인됐다.

- **Core Contribution**: 이 논문은 분류용 DNF(AND-of-ORs) 규칙 집합을 목표로, 정확도와 해석 가능성(규칙 폭, 간결성)까지 유지하면서 커버리지를 극대화하는 문제 CDPR(High Coverage Discriminative Parsimonious Rule sets)을 정식화한다. 이를 위해 정확도 임계치, 규칙 중복(겹침) 제한, 규칙 수·폭 제한을 품질 제약으로 함께 다루는 것이 핵심 기여다.

- **Technical Challenges**: CDPR의 대표 제약인 규칙 간 overlap-rate 제한은 matroid/knapsack 형태가 아니어서 기존 표준 최적화로 직접 풀기 어렵다. 논문은 이를 해결하기 위해 CDPR을 submodular maximization 문제로 변환하고, 후보 규칙을 노드로 하는 overlap-graph를 구성해 Graph Rules Algorithm(GRA)로 CDPRL을 풀며 근사 보장도 함께 제시한다; 또한 런타임을 줄이기 위해 CDPRL-MO 변형과 그리디 GDY도 제안한다.

- **Empirical Impact**: 10개 공개 UCI 데이터셋과 알츠하이머 관련 2개 데이터셋에서 GRA와 GDY는 다음 최선 알고리즘 대비 평균 커버리지 기준 2.5배 이상 향상을 보였다. 정확도와 규칙의 해석 가능성 지표(폭·간결성)도 경쟁 수준을 유지해, 규칙 기반 모델이 GDPR ‘설명에 대한 권리’ 같은 요구에서 더 실용적으로 쓰일 수 있음을 시사한다.



### Implicit Reasoning for Large Language Model-based Generative Recommendation (https://arxiv.org/abs/2606.14142)
- **Prior Approaches**: 기존 LLM 기반 생성 추천(Generative Recommendation, GR)은 항목을 Semantic ID(SID)로 표현해 다음 SID를 생성하도록 학습합니다. 이를 보완하려고 CPT로 SID를 의미적으로 적합하게 만든 뒤, CoT(Chain-of-Thought) 형태의 명시적 합리화를 SFT로 추가하고 마지막에 RL 기반 후처리까지 붙이는 다단계 파이프라인이 널리 쓰였습니다. 하지만 각 단계가 언제 필요하고 무엇을 해결하는지에 대한 명확한 진단이 부족했습니다.

- **Core Contribution**: 논문은 명시적 추론(CoT) 기반 GR 학습 파이프라인을 단계별로 쪼개 분석하며, CoT SFT가 단독으로는 다음 SID 예측을 잘 개선하지 못한다는 원인을 제시합니다. 특히 CoT SFT의 실패가 (1) 사전지식의 약화된 언어화, (2) 텍스트 토큰- SID 임베딩 공간의 불일치, (3) 합리화 문구 품질에 대한 과도한 민감성에서 비롯됨을 체계적으로 보여줍니다. 이를 바탕으로 ‘합리화 문장을 만들기’ 대신 ‘잠재적 계산’을 유도하는 PauseRec을 제안합니다.

- **Technical Challenges**: 핵심 기술 과제는 SID 기반 생성에서 LLM의 자연어 기반 지식을 효율적으로 “연결”하는 인터페이스를 설계하는 것입니다. 저자들은 명시적 CoT가 생성 결과로 이어지지 않는 이유를 언어화 약화와 임베딩 불일치(이론적 분석 포함)로 설명하고, 합리화 문구를 조금만 바꿔도 성능이 크게 흔들린다는 점을 실험으로 확인합니다. PauseRec은 <pause> 토큰을 학습 가능한 브리지로 두고, 합리화 텍스트를 생성하거나 그에 대한 강한 감독(RL·추적 정렬)을 줄이는 대신 <pause> 위치의 손실을 마스킹해 SID 예측에 직접 기여하는 잠재 추론만 학습하게 설계합니다.

- **Empirical Impact**: Amazon 리뷰 3개 데이터셋에서 PauseRec은 기존 next-item SFT 및 명시적 CoT(특히 RLVR 결합) 대비 성능을 전반적으로 개선했으며, 최대 6.22%p 향상과 최대 65% GPU 시간 절감, 추론 지연 최대 71.3% 단축을 보고합니다. 또한 RL 기반 명시적 CoT와 비교해 12개 지표 중 대부분에서 경쟁력 있거나 더 나은 결과를 보여 “명시적 합리화 생성” 없이도 지식 활용이 가능함을 실증합니다. 결과적으로 LLM 기반 GR에서 해석 가능한 추론 텍스트보다 효율적인 잠재 추론 설계가 더 강력한 대안이 될 수 있음을 시사합니다.



### Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources (https://arxiv.org/abs/2606.14141)
- **Prior Approaches**: 기존 오디오-언어 모델은 음원 장면을 주로 클립의 전역 의미(라벨/캡션)로 취급해, 공간과 시간에 따라 변하는 상태 추적이 약한 편입니다. 반대로 음원 위치추적·SELD 계열은 시간에 따른 방향을 잘 맞추지만, 제한된 이벤트 분류체계 때문에 언어 기반의 폭넓은 의미 추론과 결합하기가 어렵습니다. 또한 많은 공간 오디오 QA 벤치마크가 정적인 소스에 집중해 동적인 공간 추론은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 음향 이벤트를 ‘무엇이 들리는가’뿐 아니라 ‘시간에 따른 어디에 있는가(방향·거리)’, ‘움직이는가’, ‘다른 소스와 어떻게 관계되는가’로 묶어 평가하는 ST-AudioQA를 제안합니다. FOA(First-Order Ambisonics) 렌더링 기반의 제어형 장면 메타데이터로, 소스 정체성과 궤적을 촘촘히 감독하고 QA를 구조화합니다. 이를 바탕으로 ST-Audio Encoder와 LLM 연동형 ST-AudioLM을 만들어 시간해상 궤적 정보를 언어 모델 입력 토큰으로 연결합니다.

- **Technical Challenges**: 핵심 난제는 ‘이벤트 의미’와 ‘시간에 따른 공간 상태(방향·거리)’를 동시에 학습하되, LLM이 질문에 맞춰 시간 고정(앵커된) 상태를 읽어낼 수 있는 표현을 만드는 것입니다. 이를 위해 ST-Audio Encoder는 time-resolved FOA 인코더로서 시점별(40개 빈) 활동·방향·거리 궤적을 예측하도록 설계하고, 의미 토큰과 궤적 토큰을 분리해 41개 오디오 토큰으로 구성합니다. 또한 ST-AudioLM은 이 토큰을 LLM에 전달하고 LoRA·커넥터만 학습하는 방식으로 단일/혼합/조합형 QA 커리큘럼(A→B→C)을 점진적으로 적용합니다.

- **Empirical Impact**: 실험 결과 ST-Audio Encoder는 동적 장면에서 잡음/기하 변화 없이도 이벤트 의미 유지와 동적 로컬라이제이션의 균형을 더 잘 이뤄, 정적 공간 인코더의 시간 슬라이딩만으로는 부족함을 보여줍니다. ST-AudioLM은 단일 소스 인지, 두 소스 그라운딩, 소스-시간-공간의 조합 추론 전반에서 정적 기반·로컬라이제이션 중심 베이스라인보다 강한 추론 성능을 보였습니다. 다만 소스 간 ‘관계’처럼 동작 조건을 포함한 세밀한 관계 추론은 여전히 어려워 추가 연구 여지가 남습니다.



### Conditioning Matters: Stabilizing Inversion and Attention in Diffusion Image Editing (https://arxiv.org/abs/2606.14125)
Comments:
          Accepted to ECML PKDD 2026 Research Track

- **Prior Approaches**: 학습 없이 텍스트 기반 확산 이미지 편집은 보통 역변환(inversion)과 어텐션 조작(attention manipulation)을 결합한다. 하지만 기존 방법은 역변환 정확도 저하(오차 누적)와 편집 충실도-배경 보존의 상충(trade-off) 문제를 해결하지 못해 구조가 무너지거나 의미가 흔들리기 쉽다. 또한 텍스트 조건이 교차 브랜치 어텐션의 의미·공간 정합성에 미치는 영향은 상대적으로 덜 다뤄져 왔다.

- **Core Contribution**: 이 논문은 텍스트 조건의 정밀도(precision)가 확산 속도장(diffusion velocity field)의 기하(geometry)와 안정성에 직접 영향을 주고, 그 결과 교차 브랜치 어텐션의 일관성까지 바꾼다는 점을 경험적·이론적으로 제시한다. 이를 바탕으로 SimEdit은 (1) 조건 리파인먼트(conditioning refinement)로 역변환 안정성과 구조 정합을 돕고, (2) 토큰 단위 교차 브랜치 어텐션 제어(token-wise cross-branch attention control)로 편집 유도 토큰과 구조 보존 토큰을 분리해 비대칭으로 조절한다. 즉, “텍스트 조건을 단순 입력”이 아니라 “편집 동역학을 좌우하는 기하 제약”으로 취급하는 관점이 핵심 기여다.

- **Technical Challenges**: SimEdit을 구현할 때의 핵심 난제는 더 정밀한 조건이 역변환을 안정화하지만, 토큰 밀도가 늘며 어텐션 정규화로 인해 편집 의미 토큰이 희석(dilution)될 수 있다는 점이다. 논문은 LCS(최장 공통 부분 수열) 기반으로 구조 보존 토큰과 편집 유도 토큰을 분할하고, 구조 토큰에는 기존 방식의 특징 주입을 유지하는 반면 편집 토큰에는 증폭 계수로 비대칭 조절을 적용해 희석을 보정한다. 또한 텍스트 정밀도가 속도장 안정성을 좌우한다는 분석을 통해, 리파인먼트가 단순 프롬프트 개선이 아니라 역변환 궤적의 분산을 낮추는 데 기여함을 연결한다.

- **Empirical Impact**: PIE-Bench에서 SimEdit은 기존 어텐션 조작 계열 방법 대비 역변환 재구성 품질과 편집 성능을 동시에 일관되게 개선하며, 마스크 없는 설정에서도 구조 보존과 의미 정렬을 함께 끌어올린다. 여러 확산 백본/역변환 솔버에서도 리파인먼트 효과가 재현되어 일반성이 확인됐고, 평가 지표(DINO 기반 구조 일관성, 편집 영역 외 배경 보존 PSNR/SSIM/LPIPS/MSE, CLIP 정렬) 전반에서 개선 경향을 보였다. 런타임도 샘플당 총합 기준으로 대부분의 비용을 차지하는 반면 리파인먼트 단계는 상대적으로 작은 비중(약 수십 초 중 일부)에 그쳐 실사용 편입성이 높다는 점을 강조한다.



### Recovering Stranded Discrimination in Knowledge Tracing: Per-Item Bias Correction via Empirical-Bayes Shrinkag (https://arxiv.org/abs/2606.14123)
Comments:
          25 pages, 3 figures. Accepted at ECML PKDD 2026 (Research Track). Code: this https URL

- **Prior Approaches**: 기존 지식추적(KT) 연구는 DKT, SAKT, DKVMN, AKT, LPKT처럼 학생 상태(및 일부는 항목 표현)를 학습해 정답확률을 예측한다. 배포 후에는 모델이 학습을 멈춘 채로 항목 속성 변화(난이도/신규 항목/집단 변화)와 백본의 항목별 표현 한계가 누적되어 항목 단위 logit 편향이 생긴다. 이를 보정하려고 Platt scaling, temperature scaling, isotonic regression 같은 사후 캘리브레이션을 쓰지만, 이런 전역(score-only) 변환은 대개 AUC를 그대로 둔 채 확률 눈금만 조정한다.

- **Core Contribution**: 이 논문은 배포 KT 모델에서 항목별로 “AUC 판별력의 여유(headroom)”가 좌초(stranded)될 수 있음을 진단한다. 핵심은 단조(monotone)한 점수-만 변환은 순위를 보존해 AUC가 구조적으로 불변이며, 좌초된 판별력을 되살리려면 항목 정체성(item identity)을 조건으로 걸어야 한다는 점이다. 이를 해결하기 위해 SLC(State-space Logit Correction)를 제안하고, 항목 logit에 대한 가산 오프셋을 추정해 ranking 품질까지 복구하는 것을 목표로 한다.

- **Technical Challenges**: SLC의 기술적 난관은 이진 관측(정답/오답)을 항목별 연속형 편향 추정으로 연결하는 추론 문제다. 논문은 Laplace/IRLS로 이진 우도에 대한 가우시안 의사관측(pseudo-observation)을 만들고, 항목별 랜덤 효과로서 편향을 모델링한 뒤 Kalman smoother 기반의 경험적 베이즈 shrinkage로 희소 항목에서의 과적합을 줄인다. 또한 offset-Platt 링크로 전역 스케일/시프트는 분리해 맞추며, 시간 드리프트까지 추적하는 확장에서도 현재 데이터 밀도에서는 정보량이 부족하다는 ‘탐지 가능성 한계’를 이론적으로 제시한다.

- **Empirical Impact**: 네 가지 KT 데이터셋, 다섯 가지 백본, 세 시드에서 SLC는 네 데이터셋 모두에서 AUC를 개선하고 세 데이터셋에서 NLL도 함께 낮춘다. 특히 효과는 관측이 적은(spare/sparse) 항목에 더 집중되며, 이는 shrinkage가 실제로 유효하다는 신호로 해석된다. 교차 도메인 대조 실험은 동일한 현상이 KT 밖(예: 엔티티 수준 편향이 남는 배치 모델)에서도 나타날 수 있음을 시사한다.



### FAConformer: Frequency-Aware Convolutional Transformer for Auditory Attention Decoding (https://arxiv.org/abs/2606.14120)
Comments:
          15 pages, 7 figures

- **Prior Approaches**: AAD는 다화자 환경에서 신경반응으로 ‘어떤 화자를 듣고 있는지’를 맞히는 문제지만, EEG 기반은 신호가 약하고 잡음/개인차가 커서 안정적인 판별 표현 학습이 어렵다. 기존 방법들은 주로 EEG를 여러 주파수 대역으로 나눈 뒤 DE 같은 특징을 얕게 결합하거나, 대역 특징을 그대로 이어붙여(컨캣) 결합해 주파수 정보를 충분히 계층적으로 쓰지 못한다. 또한 대역별 패턴과 대역 간 상호작용이 데이터에 따라 유연하게 최적화되지 못하는 한계가 지적된다.

- **Core Contribution**: FAConformer는 주파수 대역을 기준으로 계층적 모델링을 수행해, 주파수 영역 EEG 정보를 더 효과적으로 활용한다. 각 대역은 독립적인 CNN-Transformer 인코더로 인코딩해 대역 특이적(밴드 스페시픽) 표현을 만들고, 이후 주파수-aware attention(FAA)으로 대역 토큰 간 의존성을 먼저 학습한 뒤 융합해 최종 결정을 내린다. 아울러 band-wise auxiliary supervision(BAS)으로 FAA에서 기여도가 낮게 평가될 수 있는 가지(branch)가 공동 학습에서 덜 최적화되는 문제를 줄인다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “대역별로 무엇을 잘 학습할지”와 “대역을 어떻게 적응적으로 섞을지”를 종단(end-to-end) 방식으로 설계하는 것이다. FAConformer는 FFT 기반 마스크로 대역 신호를 분해한 뒤, 각 대역마다 별도 인코더로 장단기 의존성을 함께 학습하고, FAA에서는 대역별 특징을 토큰 시퀀스로 만들어 자기어텐션으로 대역 간 상호작용을 데이터 주도로 포착한다. 동시에 BAS를 추가해 약한 기여 대역도 판별 정보를 학습하도록 유도해 융합 단계의 신뢰성을 높였다.

- **Empirical Impact**: DTU와 KUL 두 공개 AAD 데이터셋에서 의사결정 창 길이(2s/1s/0.1s) 전반에 걸쳐 FAConformer가 12개 경쟁 기준선을 일관되게 능가했으며, 최신 상태의 모델 대비 약 4.9% 향상을 보고한다. 특히 KUL처럼 더 짧은 창에서 개선 폭이 더 커져, 주파수-aware 설계가 시간해상도 제약 하에서도 효과적임을 시사한다. 추가 분석(대역 중요도/어블레이션/파라미터 민감도)도 프레임워크의 효과와 견고함, 그리고 해석 가능성을 뒷받침한다.



### A Two-Stage Statistical Framework for Evaluating Associative Interference in Large Language Models (https://arxiv.org/abs/2606.14117)
Comments:
          11 pages; 2 figures

- **Prior Approaches**: LLM의 편향(bias)을 IAT(암묵적 연합검사)처럼 심리학 패러다임으로 평가하려는 시도가 늘었지만, 결과 해석을 흐리는 방법론적 혼선이 있었다. 특히 refusals(거절), safety 제약, 포맷 이탈 같은 ‘응답 순응도’가 ‘과제 수행’과 함께 섞여, 비슷한 비대칭이 실제 연합 구조가 아니라 모델의 출력 정책 차이일 수 있다는 문제가 지적됐다. 또한 기존 프롬프트 기반 비교는 추론 구조가 약해, null 결과가 ‘편향 없음’인지 ‘측정 불능’인지 구분하기 어려웠다.

- **Core Contribution**: 이 논문은 IAT를 LLM용 통제된 forced-choice(강제 선택) 형식으로 재구성하고, 두 단계(순응도-추론)를 분리하는 베이지안 모델링 프레임워크를 제안한다. 1단계는 올바른 ‘A/B’ 선택을 했는지(순응도)를 추정하고, 2단계는 올바른 응답 조건에서 congruent 대비 incongruent 블록의 과제 일관성(task-consistency) 감소를 ‘연합 간섭(associative interference)’으로 정의해 추정한다. 이를 통해 거절·포맷 문제로 인한 착시를 배제하고, 블록 의존 비대칭이 진짜로 존재하는지 평가한다.

- **Technical Challenges**: 핵심 기술 난제는 LLM이 사회적으로 민감한 연관을 피하거나 안전 필터링으로 응답을 바꾸는 경우, 이를 ‘task-inconsistent’로 잘못 세면 간섭 효과가 과대추정될 수 있다는 점이다. 논문은 JSON 강제 포맷으로 A/B만 받게 하고, noncompliant(거절·오류·누락)을 별도 범주로 분리한 뒤, 순응도는 1단계 계층 로지스틱 회귀로, 간섭은 2단계에서 조건부로 추정한다. 더해 자극 단어별 변동을 item-level 랜덤 인터셉트로 흡수하고, 블록 라벨을 퍼뮤트한 위조 검정으로 관측된 비대칭이 항목 구성이나 우연이 아니라 ‘블록 의존 구조’에서 나온 것임을 확인한다.

- **Empirical Impact**: 세 모델(GPT-5, Gemini 2.5 Pro, Claude Sonnet-4)과 두 도메인(Gender–Career, Gender–Science)에서, 순응도는 전반적으로 0.98 이상으로 높고 블록 간 차이도 거의 없었다. 그 결과 간섭 효과는 모델별로 크게 달랐는데, Claude Sonnet-4는 Gender–Career에서 유의미한 간섭(ΔP=0.086, 95% CrI [0.026, 0.173])을 보였고 Gender–Science에서는 더 작은 효과가 관측됐다. 반면 Gemini 2.5 Pro는 간섭이 약화됐고, GPT-5는 두 도메인 모두에서 최소 또는 검출 불가 수준의 간섭만 나타나 IAT형 연합 비대칭이 ‘보편적 성질’이 아니라 모델·정렬(alignment) 특성에 좌우됨을 시사한다. 이 프레임워크는 심리학 기반 감사(auditing)를 LLM 평가에 적용할 때 null 결과 해석과 측정 타당성을 동시에 개선하는 표준 도구가 될 가능성이 크다.



### Numbers Already Carry Their Own Embeddings (https://arxiv.org/abs/2606.14108)
Comments:
          Presented at the MATH-AI Workshop at NeurIPS 2025

- **Prior Approaches**: 기존 수치 표현은 숫자를 텍스트처럼 토큰화한 뒤(예: 문자/서브워드 조각) 모델이 학습으로 덧셈·곱셈 규칙을 암묵적으로 재구성하도록 맡기는 경우가 많습니다. 연속 임베딩, 자릿수 기반 위치 인코딩, 상징 학습(Symbolic pretraining) 등은 의미가 있으나, 본질적으로는 학습 중 “재발견”에 의존한다는 한계가 있습니다. 그래서 정밀한 산술 일반화에서 취약성과 오류가 반복될 여지가 큽니다.

- **Core Contribution**: 이 논문은 Adelic operation-preserved embeddings(AOE)라는 학습 없는(훈련 프리) 수치 표현을 제안해, 숫자의 실수 값과 모듈러(p-adic) 서명을 동시에 임베딩에 담습니다. 핵심은 덧셈과 곱셈 같은 연산 구조를 좌표별로 보존하도록 표현 자체를 설계해, 임베딩이 “수학의 언어를 그대로” 반영하게 만든 점입니다. 또한 AOE는 기존 아키텍처에 끼워 넣는 형태(plug-and-play)로 동작합니다.

- **Technical Challenges**: 문제는 숫자 토큰화를 버리고도 신경망이 바로 쓸 수 있는 고정 형태의 텐서를 만드는 동시에, p-adic 성분의 정밀도·선택(어떤 소수 집합, 몇 자리까지)을 계산적으로 다뤄야 한다는 점입니다. 논문은 Adele ring의 구성에 기반해 실수 성분과 선택한 여러 p-adic 전개를 잘라낸 벡터들을 연결하고, Hensel의 보조정리로 p^N 정밀도의 합동해를 들어 올려 전개를 구성합니다. 여기에 2D 위치 인코딩을 더해 시퀀스 위치와 내부 구조(소수 인덱스·자리 인덱스) 정보를 함께 주입합니다.

- **Empirical Impact**: Algebraic Combinatorics Dataset(ACD) 6개 분류 과제에서 AOE 탑재 모델은 동일한 아키텍처의 기준선(표준 nn.Embedding 기반)보다 전 과제에서 일관되게 성능이 좋았습니다. 특히 Weaving Patterns에서 첫 ‘완전 정확도(퍼펙트 정확도)’를 달성해, 대형 언어모델도 어려워했던 숫자 문제의 실마리를 보여줬다고 주장합니다. 다만 유리수(ℚ) 중심의 범위 제한과 p 선택·정밀도 설정의 휴리스틱, 전처리로 인한 학습 속도 저하(기준 대비 수십 배)를 향후 과제로 제시합니다.



### FEMOT: Multi-Object Tracking using Frame and Event Cameras (https://arxiv.org/abs/2606.14094)
- **Prior Approaches**: 기존 MOT는 주로 프레임 기반 RGB 카메라의 검출-연결(tracking-by-detection) 또는 탐지+추적, 쿼리 기반 Transformer로 발전해 왔습니다. 하지만 저조도, 역광, 과노출, 모션 블러, 유사 외형, 잦은 출입/가림 등 복잡한 환경에서는 시각 정보 신뢰도가 떨어져 궤적 단절과 신원 연관 오류가 누적되기 쉽습니다. RGB-이벤트(event) 융합은 검출·단일 객체 추적에서는 연구가 있었지만, 신원 단위의 시간 일관성이 요구되는 RGB-이벤트 다중 객체 추적(MOT)은 대규모·정밀 어노테이션 벤치마크가 부족해 체계적으로 검증되지 못했습니다.

- **Core Contribution**: 이 논문은 RGB-이벤트 MOT을 위한 대규모 데이터셋 FEMOT(Frequency-aware? 아님; 논문 제목의 데이터셋명)와 이를 활용한 포괄 벤치마크를 제시합니다. FEMOT은 다양한 실제 시나리오와 14개 도전 속성(저조도, 빠른 움직임, 출입 빈도 등)을 포함하며 RGB 및 이벤트 데이터와 고품질 궤적/박스 어노테이션을 함께 제공합니다. 또한 주파수 영역에서 RGB와 이벤트의 상보성을 분리·융합하는 FEMOTR을 제안해, 강건한 위치 추정과 신원 연관을 동시에 노립니다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 비동기 이벤트를 RGB 프레임에 정렬해 학습 가능한 입력으로 만들고, (2) 두 모달리티가 갖는 서로 다른 주파수 특성과 잡음/왜곡을 안정적으로 결합하며, (3) 다중 객체에서 시간에 걸친 신원 연관을 지속적으로 유지하는 것입니다. 논문은 이벤트를 RGB 프레임의 노출 구간에 맞춰 이벤트 누적(e.g., ON/OFF 채널화) 형태로 변환해 정렬 문제를 줄였고, Frequency Aware Feature Fusion에서 FFT 기반으로 진폭/위상을 분리한 뒤 모달리티별 주파수 응답을 가이드-조절 방식으로 융합합니다. 이후 Transformer 기반 쿼리 전파와 동적 시간 상호작용 모듈(장기 메모리 갱신, 다중 궤적 간 attention)을 통해 탐지와 연관을 함께 수행합니다.

- **Empirical Impact**: FEMOT 데이터셋과 DSEC-MOT(DSEC-MOT) 두 벤치마크에서 광범위한 실험을 통해 FEMOTR의 효과를 검증합니다. 또한 FEMOT 기반으로 10개 이상 강력한 기존 추적기를 재학습·평가해, RGB-이벤트 MOT 연구를 위한 표준 벤치마크 성격을 확립합니다. 결과적으로 이벤트 카메라가 복잡 조명/모션에서 RGB의 취약점을 보완한다는 점을 정량적으로 뒷받침하며, 향후 RGB-이벤트 MOT의 비교·진보를 촉진할 것으로 기대됩니다.



### Clay-CNN Hybrids: Leveraging Geo-Foundational Models as Auxiliary Context for Landslide Detection (https://arxiv.org/abs/2606.14081)
Comments:
          9 pages, 7 figures, 2 tables

- **Prior Approaches**: 기존 재난 후(이벤트 직후) 산사태 자동 탐지는 주로 U-Net 계열의 의미 분할에 의존하지만, Landslide4Sense처럼 양성 픽셀이 약 2% 수준인 극단적 클래스 불균형과 이벤트별 데이터 부족에 취약했다. 또한 지형(DEM·경사) 정보는 물리적으로 의미가 있으나 픽셀 수준 분리도는 낮아, 기존 방식은 스펙트럼 유사성(맨땅 등) 때문에 오탐을 줄이기 어렵다는 한계가 있었다. 최근에는 Geo-Foundational Model(GFM)로 Prithvi-EO-2.0 등이 시도됐지만, 이를 CNN 구조에 어떻게 ‘최적으로’ 결합하는지에 대한 체계적 검증이 부족했다.

- **Core Contribution**: 이 논문은 Clay v1.5 같은 Geo-Foundational Model의 장점을 산사태 분할에 실질적으로 살리기 위해, GFM을 CNN과 결합하는 두 가지 하이브리드 아키텍처(Arch 1: Clay를 주 인코더+잔차 지형 융합, Arch 2: U-Net의 병목 컨텍스트에 Clay 삽입)를 제안한다. 핵심 메시지는 GFMs를 분할기 전체를 대체(standalone 인코더)하기보다, 공간 정밀도가 필요한 CNN 계층에 ‘보조 컨텍스트’로 투입하는 방식이 유리하다는 점이다. 즉, 스펙트럼의 일반화는 Clay가, 경계·국소화는 U-Net이 담당하도록 역할을 분리한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 클래스 불균형으로 인해 결정 임계값이 기본 0.5에서 크게 벗어나며 (2) Clay의 전역(글로벌) 표현을 픽셀 단위 경계 재현에 맞게 정렬해야 한다는 점이다. 이를 위해 LoRA 기반의 단계적 미세조정(먼저 디코더/융합 학습 후 일정 epoch에 LoRA 삽입), 가중 손실(BCE+Lovász hinge)로 F1 최적화 성향을 강화, 그리고 검증셋에서 임계값을 재보정하는 절차를 적용했다. 또한 MC Dropout과 Grad-CAM을 통해 단순 성능 향상뿐 아니라 오탐/미탐 및 융합 기여가 어디서 발생하는지 진단했다.

- **Empirical Impact**: Landslide4Sense 벤치마크에서 하이브리드 U-Net+Clay(Arch 2)에 2-stage LoRA를 적용한 모델이 테스트 F1 64.5±1.8%(seed 3개 평균)를 달성해 U-Net 기준선(59.9%)과 Clay 단독 인코더(55.2%)를 모두 능가했다. 특히 FPR은 낮게 유지하면서(배경 오탐 억제) 미세한/스펙트럼이 애매한 영역에서 FN이 상대적으로 늘어나는 오차 양상이 드러났고, 이는 불균형 상황에서의 합리적 trade-off로 해석된다. 해석 실험(Grad-CAM)과 불확실성(MC Dropout)은 Clay가 ‘분류/우선순위’에 도움을 주고, CNN이 ‘공간 정밀’로 보정한다는 결론을 뒷받침하며, 향후 지오해저드 분할에서 GFMs의 배치 전략에 실증적 기준을 제시한다.



### Rethinking Backdoor Adversarial Unlearning through the Lens of Catastrophic Forgetting in Continual Learning (https://arxiv.org/abs/2606.14078)
Comments:
          Accepted by ACM CCS 2026

- **Prior Approaches**: 기존 백도어 방어는 주로 (1) 트리거 합성 기반, (2) 모델 재구성 기반, (3) 백도어 적대적 언러닝(Adversarial unlearning) 기반의 안전 튜닝으로 나뉜다. 그러나 최근 연구는 ASR을 낮추더라도 잔존 백도어 특징이 남아 있어, Retuning Attack이나 Query-based Reactivation Attack처럼 재활성화 공격에 쉽게 무너진다고 지적한다. 또한 안전 튜닝은 출력 수준의 견고성에 치우쳐, 백도어가 파라미터에 어떻게 인코딩되는지에 대한 원리적 제거가 부족하다는 한계가 드러난다.

- **Core Contribution**: 이 논문은 백도어 학습과 제거를 연속학습(continual learning)의 관점에서 ‘3단계 순차 태스크’로 재정식화한다(정상 태스크 τc, 백도어 태스크 τb, 언러닝 태스크 τu). 그리고 “완전한 백도어 언러닝”을 정의해, 언러닝 후 모델이 백도어 효과를 완전히 제거하면서도 정상 정확도를 기준선에 가깝게 유지하는 상태로 명확히 한다. 더 나아가 완전 제거를 위한 언러닝 태스크의 필요조건(언러닝 태스크는 백도어 태스크에 정렬되고, 정상 태스크와는 직교해야 함)을 재난적 망각(catastrophic forgetting) 메커니즘으로부터 도출한다.

- **Technical Challenges**: 핵심 난제는 잔존 백도어를 ‘파라미터가 가진 메모리’를 원리적으로 지우면서, 정상 태스크 성능은 유지하는 제거 조건을 실제 학습 절차로 구현하는 것이다. 이를 위해 BI-BAU는 언러닝 조건을 만족하는 적대적 예시 생성 문제를 ‘Blind inversion(블라인드 역변환)’으로 구성하고, MAP(최대 사후확률) 목적을 최적화하도록 설계한다. 또한 적대적 훈련의 양수준(bi-level) 최적화를 Expectation-Maximization(EM) 틀에 통합해 MAP 목적을 효과적으로 풀어내며, 타깃 클래스가 불명인 비지정형(untargeted) 및 멀티모달 대조학습 설정으로도 확장한다.

- **Empirical Impact**: 실험은 BI-BAU가 다양한 백도어 공격 전반에서 일반성을 보이며, 특히 직교성/선형성이 낮아 기존 방어가 취약해지는 경우에도 잔존 ASR을 낮춰 완전 제거에 가깝게 만든다고 보고한다. 더 중요하게는 정제 후에도 재활성화 공격 하에서 백도어가 다시 켜지는 현상이 억제되어, 기존 방법의 ‘겉보기 안전’ 문제를 실질적으로 완화한다. 결과적으로 이 연구는 백도어 방어를 단순 출력 안정화가 아닌 연속학습 기반의 망각 정렬 원리로 다룰 수 있음을 실증적으로 제시하며, 실제 배포 환경의 신뢰성 향상에 의미가 있다.



### Knowledge Graph Enhanced Memory-Augmented Retrieval for Long Context Modeling (https://arxiv.org/abs/2606.14047)
- **Prior Approaches**: 긴 문맥 언어모델은 토큰 수를 늘리는 방식(확장 attention, long-context LLM)과 외부 메모리 검색(메모리-증강, RAG 계열)로 발전해 왔습니다. 하지만 attention은 ‘lost-in-the-middle’로 중간 구간 정보가 비관련 정보처럼 처리되는 문제가 있고, 검색 기반 방법은 관련 항목을 ‘관계’가 아닌 ‘의미적 유사도’로만 고르기 쉬워 동일 개체의 상태 변화/인과 흐름을 놓칩니다. 또한 지식그래프 접근은 ConceptNet·Freebase 같은 고정 그래프에 의존하는 경우가 많아 세션/도메인에만 나타나는 고유 개체와 관계를 실시간으로 반영하기 어렵습니다.

- **Core Contribution**: KGERMAR는 추론(inference) 중에 입력 텍스트로부터 동적이고 문맥별 지식그래프(개체·관계)를 구성해, 의미 유사도 검색에 ‘명시적 엔터티 관계’ 신호를 결합합니다. 이를 위해 그래프 구조 임베딩과 텍스트 의미 임베딩을 함께 학습·융합하고, 기존 메모리-증강 구조를 확장해 관계 중심 검색이 가능하게 합니다. 결과적으로 수천 토큰 떨어진 과거 맥락에서도 ‘같은 개체의 인과/상태 진행’에 맞는 대상을 더 잘 찾아 일관된 장문 추론을 돕습니다.

- **Technical Challenges**: 핵심 난점은 (1) 긴 문맥에서 관계를 정확히 뽑아내야 하지만 관계추출은 로컬 윈도 제약을 받는다는 점, (2) 동적 그래프 품질 저하가 검색 성능으로 바로 이어질 수 있다는 점, (3) 그래프 기반 신호와 텍스트 의미 신호를 충돌 없이 효과적으로 결합해야 한다는 점입니다. 논문은 NER·관계추출(BERT 계열)로 문맥 그래프를 만들고, 개체 통합·관계 신뢰도 점수·그래프 필터링으로 잡음을 줄였습니다. 또한 R-GCN으로 관계 기반 전파(다중 hop)를 수행하고, contextual/semantic/structural의 3개 메모리 은행 검색 신호를 가중치 학습으로 융합해 그래프 구조와 텍스트 의미를 함께 주입합니다.

- **Empirical Impact**: SlimPajama, WikiText-103, PG-19, Proof-pile의 다양한 도메인에서 문맥 길이 1K~32K 전 구간으로 평가했으며, 기존 memory-augmented baseline 대비 최대 8.5% 낮은 perplexity와 2~2.5배 더 나은 메모리 효율을 보고합니다. 특히 구조적 관계 기반 검색이 강화되면서 NLU 태스크 전반에서 in-context learning 성능도 더 좋게 나타났습니다. 동적 지식그래프를 추론 시점에 구성해 도메인 특화 표현을 만든다는 점에서, 고정 지식베이스 의존을 줄이고 장문 추론의 ‘구조적 관련성’ 문제에 실질적 돌파구를 제시한 연구로 평가됩니다.



### Same-Origin Policy for Agentic Browsers (https://arxiv.org/abs/2606.14027)
- **Prior Approaches**: 기존 에이전틱 브라우저 연구는 자연어 지시로 웹 작업을 자동화하는 데 초점을 맞췄지만, 브라우저 보안의 핵심인 동일 출처 정책(SOP)이 실제로도 제대로 지켜지는지에 대한 체계적 평가는 부족했다. 또한 SOP 위반은 주로 스크립트가 유발하는 교차 출처 데이터 흐름 관점에서 논의돼 왔지만, 에이전트가 브라우저를 “자체 채널”로 활용할 때의 위험은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 에이전틱 브라우저가 교차 출처 데이터 흐름을 자동화된 채널로 만들어 SOP를 우회할 수 있음을 관찰하고, 이를 정량 평가할 벤치마크 SOPBench를 제안한다. 나아가 SOPGuard라는 에이전틱 브라우저 전용 SOP 집행 메커니즘을 설계해 BrowserOS에 구현하고, 실제 위반을 막는 방향으로 문제를 해결한다.

- **Technical Challenges**: 핵심 난제는 에이전틱 브라우저의 자율 실행 흐름 속에서 SOP 위반이 어떤 경로로 발생하는지 측정·분류하고, 이를 막되 에이전트의 웹 작업 유용성을 크게 훼손하지 않는 집행을 만드는 것이다. 연구진은 SOPBench로 위반 유형을 체계적으로 평가하고, SOPGuard로 교차 출처 데이터 흐름을 집행 수준에서 제어하도록 BrowserOS에 통합했으며, 그 결과 런타임 오버헤드는 작게 유지되도록 최적화했다.

- **Empirical Impact**: 실험 결과, 기존 에이전틱 브라우저들은 정상 환경에서도 공격 환경에서도 SOP를 자주 위반했으며, 이는 SOP가 자동화된 에이전틱 환경에서 그대로 “보장되는 가정”이 성립하지 않을 수 있음을 보여준다. SOPGuard 적용 후에는 SOP 집행이 효과적으로 이뤄지면서도 유용성 손실이 제한적이었고, 소규모 런타임 오버헤드만 발생해 실사용 관점의 실현 가능성도 확인됐다.



### Hidden in Plain Sight: Benchmarking Agent Safety Against Decomposition Attacks with DECOMPBENCH (https://arxiv.org/abs/2606.13994)
- **Prior Approaches**: 기존 에이전트 안전 벤치마크는 다중 턴·다중 도구 사용을 포함하더라도, 분해 공격(Decomposition Attacks)의 ‘누적 의도’ 위험을 명시적으로 모델링하지 않는 경우가 많습니다. 또한 분해 안전을 사후적으로 변환해 평가하거나, 작업이 자연스럽게 분해되지 않아 실제 공격 흐름과 거리가 생길 수 있습니다.

- **Core Contribution**: 이 논문은 분해 공격에 특화된 벤치마크 DeCompBench를 제안합니다. Decomposition-by-Design 원칙으로 ‘처음부터 합법적(무해한) 부분 작업의 조합으로만 악의 목적이 드러나는’ 작업들을 구성해, 현실적인 에이전트 오남용 경로를 평가하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술적 난제는, 작업 그래프가 분해 가능성을 보장하더라도 실제 평가 시 구체 값이 결합되면 하위 작업이 즉시 악의적으로 보일 수 있다는 점입니다. 이를 위해 글래픽 프레임워크로 의존 구조를 만들고, LLM 분해기에서 중간 우회(Intermediate Indirection)와 단계별 은닉(Stepwise Wrapping) 같은 변환을 써서 ‘개별 하위 작업은 무해하지만 누적하면 악의’가 되도록 분해 흐름을 생성했습니다.

- **Empirical Impact**: 실험 결과, 안전 정렬이 잘된 폐쇄형 에이전트는 단일(모놀리식) 악성 요청에서는 높은 거부율을 보이지만, 이를 분해된 하위 작업 시퀀스로 제공하면 거부율이 크게 떨어지고 공격 성공률이 상승했습니다. 특히 실패의 대부분은 안전 거부가 아니라 서비스 조작 능력 한계( capability failure )에서 발생해, 현행 안전장치가 ‘누적 의도 추론’에는 약하다는 신호를 줍니다.



### Mask, Sample, Revise: A Revisable CTMC Inference Stack for Guided Discrete Flow Matching Text-to-Speech (https://arxiv.org/abs/2606.13989)
- **Prior Approaches**: 기존 NAR TTS는 AR 대비 빠르지만, 지속시간 예측기나 외부 정렬기 같은 구성요소가 필요해 복잡성과 제약(프로소디 다양성 저하)을 만들 수 있습니다. 정렬 없는 인필링(text-filler) 계열은 duration 모듈을 줄이지만, 저 NFE(적은 샘플링 예산)에서는 초기 오류(삭제·대치·화자 드리프트)가 남아 안정성이 떨어질 수 있습니다. 한편 DFM(Discrete Flow Matching)과 CTMC 기반 생성은 코덱 토큰 TTS에 적합하다고 알려졌으나, 인필링 상황에서 “추론 시 제어(inference-time control)”가 무엇을 어떻게 해야 안정화되는지는 충분히 다뤄지지 않았습니다.

- **Core Contribution**: 이 논문은 정렬 없는 DFM-TTS에서 조건부 인필링을 안정적으로 만들기 위해, 추론 단계에 CTMC 제어 스택을 도입합니다. 핵심은 Mask, Sample, Revise의 “수정 가능한(revisable) 인필링”이며, predictor-free guidance로 텍스트 조건을 강화하고 prompt-matched conditional coupling으로 확률 경로를 프롬프트에 맞춥니다. 여기에 SC-ReMask(Schedule-Constrained CTMC Remasking)를 더해, 토큰→마스크 전이를 샘플링 중에 허용함으로써 초기에 만든 결정도 뒤집어 재생성할 수 있게 했습니다.

- **Technical Challenges**: 가장 큰 난제는 저 NFE 환경에서 조건부 제어가 약해지며, 한 번의 초기 실수가 이후 삭제/대치로 굳어버리는 “오류 전파”를 막는 것입니다. 저자들은 (1) 같은 체크포인트로 조건/비조건 CTMC 전이율을 얻는 predictor-free guidance를 적용하고, (2) 프롬프트 프리픽스를 고정한 뒤 나머지 구간만 인필링하는 conditional coupling으로 확률 경로의 정합성을 맞춥니다. 또한 SC-ReMask를 CTMC의 명시적 전이(토큰-to-mask)로 구현해, tau-leaping 점프 과정에 remasking이 직접 반영되도록 하면서, switch time과 캡/리스케일로 스케줄 제약을 걸어 과도한 재마스킹을 억제합니다.

- **Empirical Impact**: 실험은 LibriSpeech test-clean의 voice-prompted 설정에서 객관 지표(WER/CER, SIM-o, UTMOS)와 인간 청취 MOS를 함께 사용해 검증했습니다. 통제된 ablation 결과, 정렬 없는 조건부 인필링의 취약성은 단순히 샘플링 스텝을 늘린다고 해결되지 않으며, PFG와 conditional coupling, 그리고 특히 SC-ReMask의 조합이 저 NFE에서 성능을 크게 끌어올립니다(예: WER 8.39%, CER 3.56% 등). WER/CER 및 지각 품질이 모두 개선되고 통계적 유의성도 보고되어, CTMC 기반 DFM-TTS에서 “추론 시 제어”가 콘텐츠 정확도의 핵심 요인임을 보여준다는 점에서 의미가 큽니다.



### STREAM: Multi-Tier LLM Inference Middleware with Dual-Channel HPC Token Streaming (https://arxiv.org/abs/2606.13968)
Comments:
          6 pages, 1 figure, PEARC '26

- **Prior Approaches**: 기존에는 로컬 추론, 기관 HPC, 상용 클라우드 API가 각각 따로 운영되는 ‘조각난’ 환경이 주류였다. 로컬은 비용·프라이버시는 좋지만 하드웨어와 컨텍스트 윈도우 제약이 크고, 기관 HPC는 강력한 GPU를 제공해도 방화벽과 배치 중심 실행 때문에 실시간 토큰 스트리밍이 어렵다. 클라우드는 품질과 접근성은 높지만 토큰 과금과 데이터 보관 정책이 민감 연구 데이터에 불리하다.

- **Core Contribution**: STREAM은 로컬·기관 HPC·클라우드를 단일 미들웨어에서 계층형으로 통합하고, 복잡도 기반 라우팅으로 ‘정말 필요한 경우에만’ 비용이 큰 계층으로 올리는 방식을 제안한다. 특히 기관 HPC에서 토큰을 즉시 사용자에게 전달하는 문제를 해결해, HPC를 사실상 API처럼 호출 가능한 형태로 만든다. 이를 통해 사용자 입장에서는 별도 작업 없이 OpenAI 호환 엔드포인트로 인터랙티브 응답을 받도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 두 가지였다: (1) 컨텍스트가 길어지면 단순 질의도 컨텍스트 한계 때문에 비싼 계층으로 넘어가는 ‘컨텍스트 유도 업그레이드’와 (2) HPC 방화벽·NAT 환경에서 실시간 스트리밍이 막히는 네트워크 제약이다. STREAM은 계층별 컨텍스트 한계에 맞춘 tier-aware 롤링 요약으로 불필요한 업그레이드를 줄이고, Globus Compute의 제어(인증·잡 디스패치)와 WebSocket 릴레이의 데이터(토큰 전달)를 분리한 dual-channel 아키텍처로 방화벽을 우회한다. 또 릴레이를 신뢰하지 않는 가정 하에 AES-256-GCM 엔드투엔드 암호화로 토큰 페이로드가 릴레이 사업자에게 노출되지 않게 했다.

- **Empirical Impact**: 평가에서 STREAM은 Llama 3.2 3B 기준 1,200개 질의 벤치마크에서 무료 티어 유지율 85.1%를 달성했으며, 라우팅 판정 정확도는 49%로 개선 여지가 남아 있다. 지연 측면에서는 배치 모드 TTFT 11.40초에서 릴레이 스트리밍 0.54초로 줄여 21.1배 개선을 보였고, 로컬 0.26초·클라우드 1.68초와 비교해도 인터랙티브 사용성을 확보했다. 결과적으로 기관 HPC의 유휴 GPU를 더 잘 활용하면서도 민감 데이터 제약을 만족하는 ‘HPC-as-API’ 패턴을 실험적으로 입증했다.



### The Silent Cost of Artificial Intelligence Assistance: A Theory of Autonomy Surrender, the Recovery Mechanism, and the Restoration of Human Agency (https://arxiv.org/abs/2606.13962)
Comments:
          15 pages, 1 figure. Submitted version

- **Prior Approaches**: 기존 연구는 AI가 의사결정에 개입할 때의 편익과 위험을 다루는 데 집중했지만, 인간 자율성이 서서히 잠식되는 비용은 충분히 이론화되지 못했다. 특히 Human Identity and Autonomy Gap(HIAG) 같은 틀은 문제의 존재를 설명하되, 자율성 포기의 누적 과정과 측정 가능한 메커니즘까지는 약했다.

- **Core Contribution**: 이 논문은 자율성 포기를 ‘인지 대역폭(cognitive bandwidth) 고갈’이 누적시키는 측정 가능한 과정으로 모델링하며, 재탈환이 왜 어려워지는지에 대한 설명을 제공한다. 또한 자율성 회복은 수동적 선택이 아니라 의도적인 대역폭 복원을 요구하는 능동적 인지 사건이며, 설계자는 이를 위한 회복 메커니즘을 의무적으로 포함해야 한다고 주장한다.

- **Technical Challenges**: 핵심 난제는 AI 보조가 제공되는 동안 자율성이 ‘인식되지 않게’ 이동하는 침묵의 비용을 어떻게 구조화해 모델의 일부로 만들 것인가였다. 논문은 (1) silent cost, (2) reclaiming이 심리·인지적으로 급격히 어려워지는 surrender threshold, (3) 의도적 통제 재진입을 설계/윤리 책임과 연결하는 recovery mechanism을 상호작용하는 메커니즘으로 통합해 해결한다.

- **Empirical Impact**: 또한 AI 보조에 대한 기능적 의존이 결핍이 아니라 선호로 경험되는 preference inversion의 ‘종착 상태’를 예측해, 자율성 회복이 단순 설계 문제가 아니라 문화·정치적 쟁점으로 확장됨을 시사한다. 제안된 모델은 AI 시스템 설계, 거버넌스 프레임워크, 휴먼 팩터 연구에서 인간이 의사결정 루프에 다시 들어갈 수 있는 경로와 책임 배분을 구체화하는 데 의미가 있다.



### GMN4AD: Graph Matching Network for Alzheimer's Disease Diagnosis with Test-Time Domain Adaptation using Multi-centered Structure Magnetic Resonance Imaging (https://arxiv.org/abs/2606.13919)
- **Prior Approaches**: 기존 sMRI 기반 그래프 접근은 각 뇌 그래프를 독립적으로 다루는 경우가 많아, 양식(modality) 차이와 사이트 간 이질성(inter-site heterogeneity)으로 생기는 도메인 편향을 충분히 반영하지 못했습니다. 그 결과 진단 성능이 데이터 분포에 따라 흔들리거나 일반화가 제한되는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 서로 다른 이질적 뇌 그래프 사이의 관계를 학습하기 위해 Graph Matching Network for Alzheimer's Disease Diagnosis(GMN4AD)를 제안합니다. 또한 추론 시점(test-time)에서 도메인 적응을 수행하는 전략을 더해, 경도 인지 장애(MCI) 단계 조기 진단을 포함한 AD 진단 정밀도를 높이는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 사이트와 양식에서 생성된 그래프 간 대응관계를 안정적으로 학습하는 동시에, 실제 추론 환경에서 발생하는 도메인 이동(domain shift)을 사전에 알 수 없다는 점입니다. 논문은 graph matching으로 교차-그래프 상호작용을 포착하고, test-time domain adaptation에서는 대조 학습(contrastive learning)을 결합해 추론 중 분포 불일치를 완화하도록 설계합니다.

- **Empirical Impact**: GMN4AD는 3개의 공개 AD 데이터셋에서 기존 최첨단 방법 대비 우수한 진단 성능을 보였으며, 특히 도메인 변화에 대한 견고함이 관찰됩니다. 이는 sMRI 그래프 기반 AD 진단에서 그래프 매칭과 추론 시점 적응을 함께 고려하는 접근이 실용적 일반화 해법이 될 수 있음을 시사합니다.



### SANA: What Matters for QA Agents over Massive Data Lakes? (https://arxiv.org/abs/2606.13904)
Comments:
          9 pages, 7 figures

- **Prior Approaches**: 탐색형 질의응답(EQA)은 에이전트가 데이터 레이크에서 관련 소스를 찾아 분석하고, 중간 결과에 따라 다음 행동을 바꾸는 장기 루프 문제다. 기존 연구는 에이전트형 검색, 분해/계획, 도구 사용, 코드·SQL 생성 등 개별 능력을 개선했지만, 검색·계획·데이터 분석이 섞여 발생하는 실패 양상을 끝단 성능만으로 분리하기 어렵다. 또한 EQA 평가(예: DCI, Metadata Reasoner)는 검색/선택의 일부를 다루더라도, “무엇이 왜 실패했는지”를 에이전트의 정책(다음 행동 선택, 제출 시점)까지 포함해 진단하기는 제한적이었다.

- **Core Contribution**: 이 논문은 SANA(Search Agent Navigation Ablation)로 EQA를 런타임 프로파일로 바꿔 진단하는 프레임워크를 제안한다. 프로파일에는 정답 소스의 순서, 누출을 제거한 하위질문, 실행 기록(어떤 데이터에서 어떤 의도로 어떤 계산을 했는지)이 포함되어, 각 구성요소를 이상화(idealized)한 뒤 차이를 남겨 “정책 실패”를 추적한다. 즉, 검색·계획·데이터 분석 실행은 분리해 상한/하한을 만들고, 잔차 성능 갭은 에이전트가 잘못된 소스를 추구하거나 검증/중간 증거 추적을 놓치거나 잘못된 최종 제출을 하는 등 정책 문제의 증거로 남긴다.

- **Technical Challenges**: 핵심 난관은 구성요소를 단순히 교체하면 의도(intent)와 기대되는 도구 호출이 어긋나 공정한 비교가 깨진다는 점이다. SANA는 각 도구 호출의 의미적 의도를 자연어(검색 키워드, 분석 목표)로 추출해, 이상화된 구성요소가 “도구 품질”만 향상되고 정책의 다음 행동 선택 책임은 유지되도록 설계한다. 구체적으로는 (1) 검색 이상화에서 골드 소스 집합 안에서만 결과를 반환해 검색 정밀도를 통제하고, (2) 데이터 분석 이상화에서 분석 의도와 검증된 실행을 맞추지 못하면 더 강한 모델로 코드 생성·수정 및 재시도까지 수행해 실행 실패를 줄인다.

- **Empirical Impact**: LakeQA와 KramaBench(변환 버전)에서 고정된 프롬프트·예산·런타임 조건으로 경량/중간 크기 에이전트를 평가한 결과, 데이터 분석 실행이 두 벤치마크에서 일관된 병목으로 나타난다. 검색은 LakeQA의 대규모 데이터 레이크 설정에서는 큰 한계였지만, 더 작은 스케일의 KramaBench에서는 상대적으로 덜했으며, 계획은 전반적으로 개선 폭이 더 작았다. 무엇보다 end-to-end 정확도만으로는 섞여 보이던 실패를 검색·계획·분석 실행 및 잔차 정책 문제로 체계적으로 분해해, 향후 “증거 추적·검증·정답 제출/중단 기준” 같은 정책 설계 목표를 구체화하는 데 의미가 있다.



### HiLo-Token: Input-Adaptive High-Low Frequency Token Compression for Efficient Image Editing (https://arxiv.org/abs/2606.13898)
Comments:
          14 pages, 10 figures, Patent filled

- **Prior Approaches**: 기존 생성형 이미지 편집은 사용자 마스크를 입력으로 받아 처리하지만, DiT(디퓨전 트랜스포머)로 넘어가면 토큰 수 증가로 인해 지연(latency) 비용이 크게 늘어나는 문제가 있었다. 토큰/모델 압축, 활성 캐싱, 저비트 양자화 등은 효과가 제한적이거나(혹은) 품질 저하가 동반돼 프로덕션 배포엔 부담이 컸다. 또한 마스크 기반 편집을 직접 최적화한 연구는 일부 있었지만, 컨텍스트 손실을 줄이면서 효율을 동시에 확보하는 데 한계가 있었다.

- **Core Contribution**: 이 논문은 DiT 기반 마스크 편집의 지연 병목을 줄이기 위해 HiLo-Token을 제안한다. 핵심은 입력(사용자 마스크/영상 특성)에 적응적으로 토큰 예산을 배분해, 편집 영역 주변의 국소 정보(고주파)와 전역 구조(저주파)를 동시에 유지하면서 불필요한 토큰 계산을 줄이는 것이다. 특히 디퓨전 변환기에서 품질 회귀 없이 압축을 가능하게 하는 “주파수 기반 선택+다중 해상도 토큰” 설계를 채택했다.

- **Technical Challenges**: 가장 큰 기술 과제는 공격적인 토큰 삭제 시에도 편집 품질을 유지하는 것이었다. 단순히 확장 마스크(dilated mask) 안의 토큰만 남기면 컨텍스트 모양이 달라져 유용한 정보를 버리게 되는데, 이를 보완하기 위해 고주파 토큰은 소벨(Sobel) 기반 공간 주파수 맵에서 선택하고, 저주파 토큰은 16배 다운샘플된 정보로 전역 구조를 대표하도록 결합한다. 다음으로 컨텍스트 추출 비용을 줄여야 했는데, 값비싼 컨텍스트 인코더 대신 저비용의 공간 주파수 계산과 경량 패치 임베딩으로 토큰 선택 오버헤드를 약 10ms 수준으로 제한했다.

- **Empirical Impact**: 대규모 프로덕션급 평가 데이터에서 HiLo-Token은 DiT만 기준으로 A100-80GB에서 마스크 비율 구간별로 3.13배/2.59배/1.67배 가속을 달성했으며, 생성 품질 저하 없이 동작했다고 보고했다. 끝단(end-to-end) 파이프라인 기준으로도 1.33배/1.66배/1.77배의 전체 속도 향상이 확인됐고, Amazon AWS p5.48xlarge 노드 요구량은 Remove 기능 기준 33% 감소로 이어졌다. 또한 텍스트 포함 generative fill 등 사용자 시나리오 중심의 사용자 연구에서도 동률 비율이 높거나 일부 조건에서 HiLo-Token이 더 좋은 결과를 보였다.



### How do Self-Supervised Remote Sensing Vision Models Transfer to Downstream Tasks? (https://arxiv.org/abs/2606.13896)
- **Prior Approaches**: 기존 GeoFM 연구는 성능(벤치마크 점수)이나 최종 임베딩을 기준으로 모델을 비교해 왔지만, 과업과 적응 설정에 따라 어떤 정보가 실제로 활용되는지까지는 충분히 규명되지 않았다. 특히 원격탐사는 자연영상과 달리 경계가 불명확하고 스펙트럼·공간·시간·지리 맥락의 영향을 크게 받는데, 표준 전이 헤드는 이러한 깊이별 정보 구성과 잘 맞지 않을 수 있다. 그 결과, 선행 연구는 특정 벤치마크에선 잘 보이지만 다른 과업·설정에선 뒤처지는 현상을 보고해 왔다.

- **Core Contribution**: 이 논문은 대표적인 6개 GeoFM을 대조/지식증류/재구성/멀티모달 사전학습 계열로 묶고, 분류·회귀·세그멘테이션을 대상으로 라벨 가용성과 다운스트림 파이프라인(고정 vs 미세조정, 디코더 설계)을 바꿔가며 전이 거동을 체계적으로 분석한다. 또한 레이어별 선형 프로빙과 depthwise 프로파일링으로 “어느 블록에서 과업에 필요한 정보가 더 잘 접근되는지”를 보여주어, 벤치마크 간 순위가 흔들리는 원인을 해석 가능하게 만든다.

- **Technical Challenges**: GeoFM 전이는 최종 임베딩만으로는 설명이 부족해, 네트워크 깊이에 따른 정보 접근성(저수준 신호 vs 의미 정보)과 적응 단계에서의 변화까지 동시에 측정해야 한다. 논문은 레이어별 선형 프로브로 과업 관련 정보의 접근 위치를 추정하고, CKA로 미세조정이 표현 공간을 깊이별로 얼마나/어디에 국소적으로 바꾸는지(특히 ViT 블록의 MLP 첫 선형층) 정량화한다. 더 나아가 세그멘테이션에서는 디코더 설계(라이트 멀티스케일 vs UPerNet 등)가 GeoFM 선택만큼 큰 영향을 줄 수 있음을 실험적으로 확인한다.

- **Empirical Impact**: 실험 결과 모델 순위는 과업 종류와 적응 설정에 따라 크게 변하며, 동일 모델군 내에서도 저수준/의미 과업에 대한 깊이별 성숙도가 달라 “최종층 기준 비교”의 한계가 드러난다. 재구성 계열(MAE, Prithvi)은 대체로 저수준 정보가 더 깊게 유지되는 경향이 있고, joint-embedding 계열(MoCo, DINO)은 초중반에 강하게 나타난 뒤 의미 정보로의 전이가 더 급격하게 전환되는 양상이 관찰된다. 또한 PASTIS와 Sen1Floods11의 사례 분석에서 미세조정은 전 깊이를 균일하게 재작성하지 않고 국소적으로 변화를 일으키며, 세그멘테이션 헤드가 정보 조직 방식과 불일치할 수 있어 평가와 적응 전략을 ‘표현 인지형’으로 설계해야 한다는 실무적 시사점을 제공한다.



### Gefen: Optimized Stochastic Optimizer (https://arxiv.org/abs/2606.13894)
- **Prior Approaches**: AdamW는 1·2차 모멘트 이동평균을 저장해 성능이 안정적이지만, 옵티마이저 상태만으로도 파라미터 메모리에 비례한 큰 버퍼가 추가된다. 기존 메모리 절감 연구로는 Adam-mini(수동 규칙 기반 공유), Adam8bit/Adam4bit(모멘트 양자화) 등이 있으나, Hessian 정렬의 이론적 근거가 약하거나 텐서 이름/아키텍처 정보 같은 수동 설정이 필요하며, 고정 블록 크기 같은 암묵적 하이퍼파라미터 부담도 남는다. 또한 양자화 방식이 플러그-앤-플레이로 항상 적용되기 어려운 제약이 있다.

- **Core Contribution**: Gefen은 AdamW의 성능을 유지하면서도 2차 모멘트 상태를 파라미터 블록 간에 자동 공유하고, 1차 모멘트를 학습된 코드북으로 양자화해 옵티마이저 메모리를 약 8배 줄이는 것을 목표로 한다. 핵심 아이디어는 큰 혼합 Hessian 항이 두 파라미터의 제곱 그라디언트 비율을 1에 가깝게 만든다는 이론에 기반해, Hessian 관련성이 큰 파라미터끼리 같은 2차 모멘트 추정을 쓰는 것이 자연스럽다는 점이다. Hessian을 직접 계산하지 않고도 이를 구현할 수 있게, 학습 초깃값의 제곱 그라디언트로 블록 구조를 자동 추론한다.

- **Technical Challenges**: 문제는 Hessian을 대규모에서 직접 계산하는 것이 불가능하다는 점인데, Gefen은 대신 첫 스텝의 제곱 그라디언트만으로 텐서 내 블록 분할 후보를 나눠 보고, 블록 내 이질성이 작아지는 “첫 뚜렷한 개선 지점”의 주기를 선택해 공유 블록을 만든다. 이어서 1차 모멘트 양자화를 위해 기존처럼 수동/고정 블록 크기와 경험적 양자화를 쓰지 않고, 앞서 얻은 블록 분할을 그대로 재사용하는 “정확한 히스토그램 기반 동적 계획법” 코드북 학습을 제안한다. 또한 Lloyd-Max 계열 양자화의 범위 수축(또는 초기화 민감도) 문제를 피하기 위해 코드북에 극단값을 강제 포함하는 등 수렴·일관성을 확보한다.

- **Empirical Impact**: 실험에서 Gefen은 비교한 AdamW 계열 메모리 절감 방법들 중 옵티마이저의 피크 메모리를 가장 크게 낮추면서도 AdamW 수준의 성능을 유지한다. 분산 학습에서는 FSDP에서 AdamW 대비 처리량(throughput)을 56% 개선했고, DDP에서는 AdamW가 마이크로배치 1조차 담지 못하는 조건에서 Gefen이 마이크로배치 2를 가능하게 하며 Adam-mini 대비 21% 처리량 향상을 보였다. 즉, 단순 대체(drop-in replacement)로도 큰 모델·더 큰 배치·더 큰 마이크로배치를 현실적으로 열어 훈련 효율을 끌어올릴 수 있다는 점에서 실용적 파급력이 크다.



### Crypto x AI, AI x Crypto: A Survey (https://arxiv.org/abs/2606.13892)
- **Prior Approaches**: 최근 ‘crypto x AI’ 영역에서는 논문·제품·콘텐츠가 급증했지만, 무엇이 실질적으로 달성됐는지 체계적으로 정리된 정답은 부족하다고 지적한다. 기존 접근들은 흩어진 아이디어와 부분 결과 중심이라, 기회·제약·미해결 질문이 무엇인지 구분하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 설문(survey) 논문은 AI가 블록체인 기반 기술(넓게는 ‘crypto’)에 할 수 있는 일과, 그 반대로 블록체인이 AI에 제공할 수 있는 가능성을 각각 정리한다. 또한 기존 연구를 체계화해 핵심 관찰을 요약하고, 추가 연구가 필요한 열린 질문을 우선순위 관점에서 제시한다.

- **Technical Challenges**: 통합의 기술적 난점은 ‘의미 있는 결합’이 무엇인지에 대한 기준이 불명확하다는 점과, 현재는 소수 사례에 대한 과장된 기대가 섞여 있다는 데서 출발한다. 논문은 이런 오해를 바로잡고, 실제로 해결해야 할 연구 질문을 드러내는 방식으로 기술적 과제를 재구성한다.

- **Empirical Impact**: 저자들은 AI와 crypto의 결합이 아직 ‘의미 있는 통합’의 초기 단계에 머물러 있다고 결론내리며, 산업의 과잉 일반화된 기대가 문제라고 강조한다. 이 설문은 분야의 혼선을 줄이고 연구 의제와 검증 가능한 방향성을 정리해, 후속 연구자와 업계 의사결정에 참고 프레임을 제공한다.



### Mirage Probes: How Vision Models Fake Visual Understanding (https://arxiv.org/abs/2606.13870)
- **Prior Approaches**: 기존 연구는 비전-언어 모델의 “mirage(환상) 행동”을 보통 하나의 실패 모드로 취급했다. 즉, 이미지가 없어도 정답처럼 보이는 출력을 내는 현상을 텍스트 편향·데이터 규칙·숏컷 탓으로 뭉뚱그려 설명하며, 내부 표현 수준에서 무엇이 다른지 검증하지 못했다.

- **Core Contribution**: 이 논문은 mirage 행동이 실제로는 두 가지 메커니즘으로 나뉜다고 주장하며, 이를 표현(representation) 레벨에서 분해하려는 Mirage Probes를 제안한다. 특히 시각 입력이 있는 경우와 없는 경우의 행동 프록시 라벨을 만들고, 질문 변형(의미는 유지, 표면 단서는 최소화) 대비쌍을 통해 내부 신호를 대비 학습적으로 측정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “이미지 없이도 그럴듯한 답”이 출력만으로는 원인이 보이지 않는다는 점이다. 저자들은 (1) 이미지 유무에 따른 행동 차이를 라벨로 근사하고, (2) 잔차 스트림·MLP·post-attention·어텐션 헤드 등 여러 층/위치에서 선형·대비 차분 방식으로 복원 가능성을 시험하며, (3) Naive Bayes 텍스트 기반 기준선으로 표면 어휘 단서 여부를 함께 통제한다.

- **Empirical Impact**: 두 개 오픈소스 VLM에서 mirage 정보는 이미지가 제공된 상태의 내부 활성에서 선형 방향으로도 유의미하게 디코딩되며, 특히 차분 활성(difference-of-activations) 프로브가 가장 깔끔하게 신호를 회수했다. 또한 벤치마크 간 분리 양상이 달라 “텍스트 편향”과 “spurious images(잠재 공간의 가짜 시각 생성)”가 서로 다른 정권(regime)임을 시사하고, 텍스트 분포 세정 같은 중재가 텍스트 편향에는 효과적일 수 있으나 spurious images에는 대표적으로 한계가 있음을 보여준다.



### SuperThoughts: Reasoning Tokens in Superposition (https://arxiv.org/abs/2606.13862)
- **Prior Approaches**: 기존 LLM의 Chain-of-Thought(CoT)는 정답으로 가기 전 긴 추론 토큰을 순차 생성해 성능을 높이지만, 그만큼 추론 계산이 비싸다. 이를 줄이려는 잠재(continuous latent) 공간 추론은 중간 표현에 대한 정답(중간 슈퍼비전) 신호가 약해 학습이 불안정해지고, 장기 추론 과제에서 성능이 떨어지는 문제가 반복됐다. 또한 멀티-토큰 예측은 한 번에 여러 토큰을 맞출 수 있어 보이지만, 실제 추론에서는 주 모델이 해당 토큰들의 KV를 채워야 해서 연산 효율이 크게 개선되지 못했다.

- **Core Contribution**: SuperThoughts는 CoT의 연속된 토큰 두 개를 하나의 잠재 표현으로 압축하고, 한 단계에서 두 토큰을 예측해 추론 길이(연산 단계 수)를 줄이도록 설계했다. 학습 단계에서는 여전히 이산 토큰에 대한 교차 엔트로피 슈퍼비전을 유지해 ‘중간 감독 부재’로 인한 학습 불안정을 완화한다. 여기에 MTP(Multi-Token Prediction) 모듈의 확신이 낮을 때는 표준 디코딩으로 되돌리는 적응형 추론을 넣어 정확도 하락을 제어한다.

- **Technical Challenges**: 핵심 난제는 토큰 쌍 압축으로 인해 중간 표현이 사전학습된 언어 분포에서 벗어나 ‘대표(표상) 드리프트’가 생기면 성능이 무너지는 점이다. 이를 해결하기 위해 압축기(Compressor)를 지식 증류로 먼저 정렬한 뒤, Main 모듈과 MTP까지 전체를 이산 토큰 기반 손실로 함께 학습하는 2단계 학습을 수행한다. 또 한 단계에서 두 ‘어려운’ 토큰을 동시에 처리할 때 생길 수 있는 한계를, MTP 예측 확률 기반 임계값으로 감지하고 다음 단계에서 단일 토큰 방식으로 재시도하는 폴백으로 완화한다.

- **Empirical Impact**: Qwen2.5-Math-1.5B/7B/14B에 적용해 MATH500, AMC23, OlympiadBench, GPQA-Diamond에서 평가했으며, CoT 길이를 약 20~35% 줄이면서 정확도는 대부분 1~2점 수준의 소폭 저하로 유지했다. 특히 1.5B에서는 적응형 추론이 MATH500 정확도를 기준선과 거의 동일하게 맞추는 동시에 CoT를 30%대까지 줄였고, 더 큰 모델에서는 전반적으로 정확도 격차를 좁히며 압축 이점을 누릴 수 있음을 보였다. 수학 외 도메인(과학)에서도 유사한 효과가 관찰되어, ‘잠재 공간 추론+이산 중간 감독+적응형 디코딩’ 조합이 장기 추론 효율화에 실용적 대안이 될 가능성을 시사한다.



### Mood-Aware Music Recommendation: Integrating User Affective Signals into Ranking Systems (https://arxiv.org/abs/2606.13858)
Comments:
          13 pages, 4 figures, and 1 table

- **Prior Approaches**: 기존 음악 추천은 사용자-아이템 상호작용에 기반한 협업 필터링(CF)이 중심이었지만, 음악 도메인에서는 상호작용이 극도로 희소하고(99.9%+), 인기 편향과 명시적 피드백 부재로 성능이 흔들립니다. 콘텐츠 기반 방식은 장르·악기·가사 같은 아이템 속성으로 확장됐으나, 사용자의 감정(기분)을 반영한 개인화는 상대적으로 덜 다뤄졌습니다. 또한 사용자의 선호는 단기 맥락에 따라 변하는데도, 많은 접근이 정적인 선호 가정에 머물렀다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 사용자의 현재 기분을 에너지-발언스(energy–valence) 감정 공간에 매핑하고, 그 목표 감정에 맞춰 후보 곡을 재랭킹하는 ‘mood-conditioned ranking’ 프레임워크를 제안합니다. 특히 소프트맥스 기반 샘플링으로, 목표 감정과의 근접성은 높이되 완전 결정적으로만 고르지 않아 다양성도 함께 확보합니다. 결과적으로 장기 취향 유사도에 단기 감정 신호를 결합해 추천 결과를 조정하는 것이 핵심 기여입니다.

- **Technical Challenges**: 기술적으로는(1) 텍스트/오디오/사용자 상태가 서로 다른 형태의 데이터를 같은 감정 공간에서 비교 가능하게 표현하고, (2) 목표 감정에 더 맞는 곡을 선호하면서도 반복 추천에서 다양성을 잃지 않는 선택 메커니즘이 필요합니다. 논문은 Russell의 에너지-발언스 평면에서 사용자 기분과 곡의 오디오 감정 특징을 [0,1] 정규화로 정렬하고, 후보 전체의 에너지-발언스 거리(제곱 유클리드 거리)를 볼츠만 분포로 확률화한 뒤 소프트맥스 확률에 따라 서로 다른 곡을 샘플링하는 방식으로 해결합니다. 또한 Spotify(사용자 씨앗 트랙)–Last.fm(후보 탐색)–ReccoBeats(발언스/에너지 피처)로 파이프라인을 구성해 실사용 형태로 통합합니다.

- **Empirical Impact**: 단일 블라인드 사용자 연구에서 mood-aware 모델이 기준선(유사도 기반 KNN)보다 사용자 인지 품질 평가가 일관되게 높게 나왔습니다. 평균 평점은 mood-aware 3.59(±1.19), 대조군 2.67(±1.28)이며, 평점 분포가 4~5로 이동했고 Mann–Whitney U 검정에서도 유의한 개선(p≈0.01)이 보고됩니다. 다만 ‘Relaxed’와 ‘Sad’에서 개선 폭이 큰 반면 ‘Stimulated’와 ‘Distressed’에서는 뚜렷한 향상이 없었고, 표본 규모·자기보고 편향·크로스플랫폼 매칭 오류 같은 한계도 함께 제시합니다.



### SpheriCity: Designing Trustworthy Conversational AI for Sustainability Decision Suppor (https://arxiv.org/abs/2606.13854)
Comments:
          Accepted to ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies (COMPASS '26)

- **Prior Approaches**: 기존 LLM 챗봇 평가는 문장 유창성, 문법, 일반적 관련성 같은 범용 지표에 치우쳐 있었고, 지속가능성 전문가가 요구하는 ‘출처 검증 가능성’을 충분히 담지 못했다. RAG처럼 근거를 찾는 접근도 정답성이나 검색 성능 위주로 측정되는 경우가 많아, 생성 결과를 실제 의사결정 워크플로에 통합할 때의 신뢰·해석 문제를 놓치기 쉽다. 또한 지속가능성 문서는 길고 구조가 제각각이어서 문서 간 비교·합성 과정 자체가 인지적 부담으로 남는다.

- **Core Contribution**: SpheriCity는 도시 단위 원형경제(순환경제) 평가 보고서를 대상으로, 근거 추적(provenance) 중심의 대화형 지식 이해(지식 sensemaking)를 지원하는 전문가 지향 프로토타입이다. 핵심은 답변 생성에 앞서 증거를 페이지 단위 인용으로 노출하고, 구조화된 합성으로 문서 간 비교와 교차 출처 검증을 돕는 데 있다. 아울러 지속가능성 고위험 지식 영역에서 전문가가 신뢰를 판단하는 방식을 반영한 ‘전문가 기반 평가 프레임워크’도 함께 제안한다.

- **Technical Challenges**: 가장 큰 기술 과제는 보고서의 길고 이질적인 구조에서 신뢰 가능한 근거를 뽑아 생성 과정에서 환각·근거 불일치를 줄이는 것이다. SpheriCity는 RAG 아키텍처 위에서 임베딩 기반 벡터 검색과 지식 그래프 기반 검색을 결합하고, 두 검색 파이프라인에서 각각 초안을 만든 뒤 합성 단계에서 불일치와 상호보완을 다루도록 설계했다. 또 인용의 페이지 번호를 언어모델이 ‘생성’하지 않고, 문서 처리 파이프라인에서 메타데이터로 보존·전달해 출처 무결성을 강화한다. 인터페이스는 긴 서술보다 불릿 기반 구조 응답과 프롬프트 템플릿을 제공해 전문가의 확인·스캔 비용을 줄인다.

- **Empirical Impact**: 6명의 지속가능성 전문가를 대상으로 교차 도시 비교, 정책 요약, 추천형 질의 등 13개 대표 쿼리를 사용한 형성적(참여형) 평가를 수행했으며, 투명한 출처·맥락 설명·해석 가능성·전문가 워크플로 정렬이 유용성과 신뢰 판단을 크게 좌우하는 것으로 나타났다. 특히 응답의 충분성보다 ‘검증 가능성’과 ‘무리한 일반화 억제’가 더 중요하다는 의견이 강조되며, 이는 범용 챗봇 평가 지표의 한계를 뒷받침한다. 이 연구는 지속가능성 의사결정 지원에서 AI의 신뢰를 측정·설계하는 실천적 기준과 설계 통찰을 제공해, 후속 고위험 영역 인간-AI 협업 연구의 출발점이 될 의미가 있다.



### Explaining RhythmFormer: A Systematic XAI Analysis of Periodic Sparse Attention for Remote Photoplethysmography (https://arxiv.org/abs/2606.13839)
Comments:
          26 pages, 8 figures

- **Prior Approaches**: rPPG 트랜스포머의 XAI는 대체로 원시 attention, Grad-CAM/그래디언트 살리언시 같은 시각화 중심 접근에 머물렀습니다. 그 결과 “그럴듯해 보이는” 히트맵은 제공되지만, 마스킹했을 때 예측이 얼마나 변하는지 같은 정량적 faithfulness 지표나 생리학적 근거 검증이 부족했습니다. 또한 원시 attention을 모델의 진짜 초점으로 볼 수 있는지에 대해 회의적 논의가 있어, 단일 방법 의존의 한계가 두드러집니다.

- **Core Contribution**: 이 논문은 RhythmFormer의 주기적 sparse attention( bi-level routing, top-k 선택 )에 맞춘 XAI 프레임워크를 제안해, 시각적 해석과 감사 가능한 수치 증거 사이의 공백을 메웁니다. 구체적으로 원시 attention, attention rollout, attention flow, Beyond Intuition을 RhythmFormer's refined attention에 맞춰 적응하고, skin(얼굴/목) 영역 정렬을 정량화하는 skin coverage 지표와 rPPG 회귀용 SaCo faithfulness를 함께 도입합니다. 이를 통해 “어디를 봤는가”와 “그 주목이 실제로 예측을 얼마나 좌우하는가”를 동시에 측정합니다.

- **Technical Challenges**: 핵심 난점은 (1) sparse top-k 라우팅 때문에 누락된 연결이 많아 attribution을 누적/흐름 기반으로 조합하기 어렵고, (2) rPPG는 분류가 아니라 파형 회귀라 SaCo 같은 교란 기반 신뢰성 평가를 그대로 적용할 수 없다는 점입니다. 저자들은 top-k로 0이 된 위치를 보존하는 방식으로 희소→밀집 재구성을 수행하고, 전 레벨 attention을 동일한 시간 해상도로 다운샘플해 다층 행렬 곱/흐름 계산이 가능하게 했습니다. 또한 SaCo의 perturbation impact을 rPPG 파형의 MAE로 재정의해, 분류 신뢰도 하락 대신 모델 민감도만 반영하도록 설계했습니다.

- **Empirical Impact**: UBFC-rPPG에서 수치 실험을 통해 다중 hop leakage(희소 top-k 라우팅에서 사라진 연결이 누적 연산으로 다시 복원되는 현상)가 정량적으로 관찰됩니다. attention rollout과 flow는 특정 refined-attention 층에서 0으로 만든 연결을 거의 다시 복원하는 반면, Beyond Intuition은 value-projection 가중 rollout과 gradient 기반 마스크로 이를 완화해 skin coverage 중앙값(0.83 vs 0.57)과 faithfulness(F=0.92)를 가장 높게 달성했습니다. 나아가 낮은 SaCo outlier 사례에서도 artefactual 영역을 대체하면 네 방법 모두 안정적으로 회복되어, 제안 지표들이 다양한 attribution 계열에서 일관된 신호를 준다는 점을 시사합니다.



### When Plausible Is Not Realistic: Evaluating Human Mobility in LLM-Based Urban Simulation (https://arxiv.org/abs/2606.13835)
Comments:
          14 pages, 10 figures

- **Prior Approaches**: 기존 LLM 기반 도시 시뮬레이터인 AgentSociety와 CitySim은 생성된 일정과 활동이 “그럴듯한지”에 초점을 두는 경우가 많았다. 그 평가는 시맨틱한 활동 분포나 서사적 일관성은 볼 수 있지만, 실제 인간 이동의 핵심인 공간·시간 제약을 함께 재현하는지 검증하기엔 부족했다. 또한 관측 가능한 이동 흔적(traceability)과 의사결정 과정을 관찰가능성(observability) 있게 남기지 못해, 오류가 어디서 발생했는지 해석이 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 LLM 기반 도시 시뮬레이터의 이동을 실세계 데이터와 직접 대조해 검증하는 프레임워크를 제안한다. 공간 이동 법칙, 시간 리듬, 네트워크 모티프, 의미 기반 활동 전이, 행동 이동 프로필을 다차원 지표로 결합해 “서사적 그럴듯함”과 “실증적 이동 현실성”을 분리 평가한다. Greater Paris와 Shanghai 데이터로 AgentSociety와 CitySim을 평가해, 어떤 요소가 재현되고 어떤 요소가 실패하는지 체계적으로 보여준다.

- **Technical Challenges**: 기여를 실현하려면 (1) LLM 에이전트의 POI 선택과 실행을 추적 가능한 형태로 기록하고, (2) 시뮬레이터 간 비교를 공정하게 하며, (3) 대규모 지도 생성과 지표 계산을 반복 실험 가능하게 만들어야 한다. 이를 위해 AgentSociety를 기반으로 En-AgentSociety를 구축해 방문 POI/카테고리와 프롬프트-응답 및 실행 흐름을 로깅(traceability)하고, 병목·지연을 모니터링할 수 있게 했으며, 지도 생성은 지역 단위로 확장되도록 최적화했다. 또한 CitySim은 공개 설명을 바탕으로 재구현·수정해 동일 인프라에서 일관된 비교가 가능하게 했고, 트립 길이·기원-목적지 흐름·체류시간·전이 역학 등 다양한 지표 계산 절차를 마련했다.

- **Empirical Impact**: 실험 결과는 시맨틱 활동 분포 같은 고수준 패턴은 일부 따라가도, 여행거리 분포·기원-목적지 흐름·체류시간·전이 동역학 같은 핵심 공간/시간 제약을 재현하는 데 큰 격차가 있음을 보여준다. 특히 기본 프롬프트(persona) 설정만으로는 이동 다양성이 안정적으로 확보되지 않으며, 프로필을 반영한 초기화가 필요하다고 관찰했다. 더불어 지역 규모 맵 생성, 관찰가능성 강화, 이동 메트릭 산출, 교통 시뮬레이터까지 포함한 공개형 인프라를 제공해 재현 가능한 벤치마킹을 촉진한다.



### Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Respons (https://arxiv.org/abs/2606.13832)
- **Prior Approaches**: CAGE Challenge 4의 기존 보상 중심(reward-only) 강화학습은 SOC의 작동 예산을 명시적으로 제한하지 않아, 보안 리워드는 높더라도 다운타임/MTTR·오탐 대응·방화벽 변경 같은 운영 비용을 과도하게 유발할 수 있습니다. 또한 휴리스틱 기반 접근이 유효 행동 처리와 임무 단계 트래픽 규율 같은 ‘운영 규율’을 내장해 MARL보다 강했지만, 그 규율이 학습으로부터 자동으로 담보되지는 못했습니다. 결과적으로 보상만 최적화한 에이전트가 실제 배포에 필요한 거버넌스(감사·한도 준수)를 안정적으로 학습하지 못한다는 문제가 남았습니다.

- **Core Contribution**: 논문은 SOC 운영 규율을 ‘안전 계약(safety-contract)’이라는 측정 가능한 제약으로 바꾸어, 에이전트가 에피소드 동안 MTTR(다운타임), false-positive(분석가 부담), 방화벽 변경(변경관리) 예산을 넘지 않도록 학습·검증하는 프레임워크를 제안합니다. 이를 그래프 기반 MARL로 구현하며, 시뮬레이터 관측과 재사용 가능한 운영 예산/제약 로직을 분리해 배치 가능성을 높입니다. 또한 ACD3-GAT( Adaptive Constrained Counterfactual Decisioning with a Graph Attention Network encoder )로 제약 최적화와 반사실적(action) 스크리닝을 통합합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 부분 관측과 비정상성(Blue-에이전트 vs Red-과정 경쟁) 속에서 (2) 장기 지평의 보안 성공과 (3) 세 가지 동시 SOC 제약을 함께 만족시키는 학습을 안정적으로 만드는 것입니다. 저자들은 Lagrangian 기반의 constrained MAPPO-GAT(C-MAPPO-GAT)로 비용 통제를 분리·검증하고, ACD3-GAT에서는 예산 컨텍스트, CVaR 기반 꼬리위험 추정, 상대(oppponent) 상태 정보, 그리고 Graph Counterfactual Risk Propagation(G-CRP)로 ‘예산을 깨는 행동’의 위험을 사전에 걸러내도록 구성했습니다. 결과적으로 보상 최적화가 유발하는 운영 피해를 학습 과정과 실행 전 스크리닝에서 함께 제어합니다.

- **Empirical Impact**: CAGE Challenge 4 재현 실험에서 무제약(unconstrained) 방법은 다운타임 예산 위반이 모든 에피소드에서 발생했고, 평균 다운타임 프록시 비용도 예산 50 대비 311~430 수준으로 크게 초과했습니다. 반면 C-MAPPO-GAT는 다운타임 위반율을 100%에서 0.3%로 낮추고 평균 다운타임 비용을 355.4에서 15.5로 줄였으며, ACD3-GAT는 평균 다운타임 비용 48.2와 13.8% 위반율로 ‘가장 보수적인 준수점’이 아닌 더 넓은 안전 계약(frontier)에 위치했습니다. 또한 토폴로지 시드 변화와 적응형 Red 스트레스 테스트에서 제약 기반 정책들이 reward-only MAPPO-GAT 대비 최악 성능 저하가 더 작게 유지되어, 배포 관점의 견고성을 실증했습니다.



### AI can help scientists publish less (https://arxiv.org/abs/2606.13829)
Comments:
          7 pages, no figures

- **Prior Approaches**: 기존 논의는 주로 ‘AI가 쏟아내는 논문 홍수’ 속에서 과학을 방어하고 진위를 가리는 데 초점이 맞춰져 있었다. 예컨대 검증·필터링 강화, 표절·생성물 탐지 같은 대응책이 중심이었지만, 근본적으로 출판 체계의 왜곡을 되돌리지는 못한다는 한계가 있었다. 그 결과 연구자 시간은 여전히 검증과 생산성 압박에 소모된다는 문제도 남았다.

- **Core Contribution**: 이 논문은 AI를 단순한 방어 도구가 아니라 출판 시스템의 왜곡을 교정하는 ‘기회’로 재정의한다. 또한 더 적은 수의 더 나은 논문을 출판하도록 돕고, 과학자들이 자신의 최선의 연구에 시간을 돌려 사용할 수 있게 하는 방향을 제안한다. 즉, 생산량 중심의 압력을 품질 중심으로 전환하는 데 기여한다는 메시지다.

- **Technical Challenges**: 핵심 기술적 과제는 AI를 사용하면서도 논문 품질을 높이고(또는 잡음을 줄이고), 동시에 검증 비용과 연구자 부담을 줄이는 균형을 만드는 것이다. 이를 위해 생성·작성 과정에서의 보조를 체계화하되, 신뢰성과 재현가능성을 해치지 않도록 출판 워크플로를 재설계하는 접근이 필요하다. 논문은 이런 방향을 통해 ‘더 적고 더 좋은’ 출판으로 자연스럽게 이어지도록 설계하자는 관점을 제시한다.

- **Empirical Impact**: 이 논문은 초록 수준에서 AI를 단지 위협이 아니라 출판 관행을 개선하는 수단으로 쓰면, 전체적으로 더 적은 검토·생산 비용으로 더 높은 품질의 결과를 기대할 수 있다고 주장한다. 또한 연구자에게 시간을 돌려주는 효과가 과학 생산성에 긍정적으로 작용할 수 있다는 점에서 의미가 있다. 다만 구체적인 실험 결과나 정량 지표는 제공된 정보만으로는 확인이 어렵다.



### Aligning Quantum Operators with Large Language Models (https://arxiv.org/abs/2606.13811)
- **Prior Approaches**: 기존 LLM-양자 연구는 Qiskit 코드 도우미나 OpenQASM 작성처럼 게이트 이름·회로 텍스트·프로그램 같은 상징 표현을 주로 다뤘다. 하지만 단위행렬처럼 연산을 정의하는 수학적 객체(예: 유니터리의 복소수 구조)는 입력으로 직접 처리하지 못해, 컴파일·검증·알고리즘 탐색에서 ‘연산자 자체’에 근거한 추론이 어렵다. 한편 양자 회로 합성의 전통적 방법은 Clifford+TT에서 정확 합성/근사 합성에 대한 보장이 있으나 멀티쿼빗로 확장 시 탐색 비용이나 성능 저하가 커서 휴리스틱·학습 기반이 필요해졌다.

- **Core Contribution**: 이 논문은 유니터리 연산자를 LLM의 잠재공간으로 투영해, 텍스트가 아니라 ‘양자 연산자’에 직접 조건을 거는 정렬(alignment) 프레임워크를 제안한다. 구체적으로 유니터리를 Pauli Transfer Matrix(PTM, 파울리 전이 행렬)로 표현하고, 이를 경량 인코더와 프로젝터로 LLM 임베딩 토큰에 매핑한 뒤 자가회귀적으로 게이트를 생성해 회로 합성을 수행한다. 또한 자연어로 게이트 제약을 주어 학습 중에 보지 못한 제약까지 반영하는 언어 조건 합성(language-conditioned synthesis) 가능성을 함께 보여준다.

- **Technical Challenges**: 핵심 난제는 복잡한 유니터리의 수학적 구조를 LLM 입력으로 ‘이해 가능’하게 만드는 표현 정렬 문제와, 합성 과정에서 남은 작업(잔여 연산)을 안정적으로 반영하는 방법이다. 저자들은 PTM의 곱성(회로의 PTM이 게이트 PTM들의 곱이라는 성질)을 이용해, residual PTM에서 한 단계씩 ‘남은 연산을 벗겨내는’ stepwise 생성(다음 게이트 1토큰 예측)으로 조합 폭발을 줄였다. 구현상 PTM을 패치 단위 시각 토큰으로 쪼개 LLM의 단어 임베딩 공간에 투영하고, LLM 자체는 고정한 채 비전 인코더·프로젝터만 supervised fine-tuning으로 학습해 RL 없이도 next-token 예측 손실로 맞춘 점이 차별점이다.

- **Empirical Impact**: 4쿼빗 Clifford+TT 합성에서 training data(145K→9.2M) 규모를 키우자 성공률이 3배 이상 개선되었고, 성능이 아직 포화(saturation) 징후 없이 증가함을 보고했다. 또한 best-of-N 샘플링으로 추론 계산을 늘리면 성공률이 80샘플에서 99.4%까지 올라 다른 기준선(탐색·시뮬레이티드 어닐링·RL 계열)을 전반적으로 앞섰다. 더 나아가 Haar-random 유니터리 같은 학습 분포 밖 입력에서도 학습된 ‘진행성’(피델리티 증가)이 관찰되어, 향후 더 긴 회로/더 일반 연산자로 확장될 수 있는 신호를 제공한다.



### A Benchmark and Framework for Evaluating Next Action Predictions in Spreadsheets (https://arxiv.org/abs/2606.13802)
Comments:
          Accepted at ICML 2026. Code and benchmark: this https URL

- **Prior Approaches**: 기존 연구는 자연어를 작업(수식/변환/편집)으로 변환하는 방식(SheetCopilot, SpreadsheetBench)이나, 사용자가 명시적으로 의도를 신호할 때만 수식을 제안하는 방식(예: SpreadsheetCoder)에 집중해 왔다. 반면 스프레드시트에서는 UI 표면이 복잡하고, 공개 코퍼스에는 코드처럼 세부 버전 히스토리가 거의 없어 “다음 액션 자동완성”을 일반적으로 평가하기가 어려웠다. 또한 오프라인(고정 시점) 평가로는 예측이 상태를 바꾸며 연쇄 오류를 만들 수 있는 현실 사용 흐름을 반영하기 힘들었다.

- **Core Contribution**: 이 논문은 스프레드시트에서 사용자의 액션 시퀀스를 관찰한 뒤 다음 편집을 제안하는 예측형 자동완성의 공백을 메우기 위해 벤치마크와 평가 프레임워크를 제안한다. 52개의 “생성 궤적(trajectory)”을 구축해 과거 편집 기록이 없는 문제를 보완하고, 온라인(on-policy) 롤아웃 기반 평가로 사용자 수용(accept)/거절(reject)이 이후 편집 계획을 어떻게 바꾸는지를 반영한다. 결과적으로 단순 정확도 대신 실제 작업 절감효과를 측정하도록 설계된 것이 핵심이다.

- **Technical Challenges**: 첫째, 공개 스프레드시트에는 편집 히스토리가 없어 학습/평가용 고품질 시퀀스를 만들기 어렵다. 이를 위해 파라미터화된 휴리스틱 시딩 + LLM 정교화 + 인간 검수를 통해 12K 액션 규모의 시퀀스를 재구성했으며, LLM-judge-editor 루프로 기계적/부자연스러운 패턴을 자동 정정해 라벨링 부담을 줄였다. 둘째, 스프레드시트 액션 공간은 공간적(좌표/범위), 시간적(순서), 복합적(여러 속성 동시 변경)이라 고정 타깃 예측만으로는 부족해, 액션 단위 TP/FP/FN(MM)와 “사용자 액션 절감(uas)”을 함께 계산하는 온라인 평가 루프(수용 시 미래를 동적으로 갱신)를 설계해 이를 해결했다.

- **Empirical Impact**: 실험 결과, 모델 성능이 누적 절감효과(uas)로 일관되게 이어져 이 태스크가 학습 가능한 문제임을 보여준다(예: 더 큰 GPT-5 계열이 더 높은 savings, 작은 모델은 격차가 존재). 또한 “항상 수용(always)”처럼 품질만 무시하면 negative savings가 발생해, 정확성(정답성)과 유틸리티(절감효과) 사이의 균형 및 abstention/stop 전략이 중요하다는 것을 확인했다. 트리거를 자주(예: stride s=1) 호출할수록 초기 거절률이 커지지만 전체 절감효과는 올라가며, 문맥 길이와 예측 길이 제한에서는 포화/역효과 같은 실무적 관찰이 제공되어 추후 설계 방향(언제 제안할지, 언제 멈출지)을 제시한다.



### An integrated interpretable control effectiveness learning and nonlinear control allocation methodology for overactuated aircrafts (https://arxiv.org/abs/2606.13794)
- **Prior Approaches**: 기존 제어 할당(control allocation)은 작동점에서 고정된 선형 제어 유효성 행렬 B를 가정하는 방식이 많았다. 이 접근은 다중 작동기(m>n) 중복성은 활용하지만, 비선형 결합과 상태 의존 유효성 변화(예: 비행 조건·작동기 상태에 따른 공력 변화)를 제때 반영하지 못해 불일치가 커지고 정확도·강건성이 떨어진다. 이를 보완하려고 고정 모델을 온라인에서 업데이트하거나(유한차분 기반) 데이터 기반 신경망·강화학습을 쓰면 성능은 좋아질 수 있으나, 계산량이 크거나(실시간 할당 불가) 블랙박스 특성으로 인해 검증·고장진단·안전성 설명이 어렵다는 한계가 남는다.

- **Core Contribution**: 이 논문은 작동기-가상 제어(virtual control) 매핑의 비선형 제어 유효성(control effectiveness mapping)을 “명시적이고 물리 제약을 포함한 해석적 모델”로 학습한다. 이를 위해 Sparse Identification of Nonlinear Dynamics(SINDy) 계열의 constrained SR3를 써서, 단순 근사나 신경망 없이도 데이터로부터 희소한 지배 방정식 구조를 복원한다. 학습된 매핑은 컴팩트하고 해석 가능하며, 입력에 대한 해석적 도함수(미분)를 제공해 비선형 솔버 안에 효율적으로 포함될 수 있다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 과격 기동에서 나타나는 상태 의존 공력·작동기 상호작용을 충분히 잘 포착하면서도, (2) 실시간 할당 루프에서 계산 가능한 형태(미분/평가 비용이 낮은 형태)로 유지하는 것이다. 저자들은 임의 다항 확장 대신 항공기 강체 회전역학의 구조(관성 기반 결합항, 모멘트 분해)를 후보 함수 라이브러리에 사전 반영해 식별 공간을 줄이고 조건을 개선했으며, SR3로 희소성과 데이터 적합의 균형 및 물리 제약을 함께 처리해 항목 선택의 안정성을 확보했다. 또한 온라인에서는 예측 잔차(residual)를 모니터링해 식별 모델이 크게 어긋날 때 계수를 갱신하는 적응 메커니즘을 두어 작동기 고장이나 운용 조건 변화에도 점진적으로 재구성되게 했다.

- **Empirical Impact**: 검증은 ADMIRE(Aero-Data Model in a Research Environment) 고충실도 비선형 항공기 벤치마크에서 수행되며, 공격적인 기동 전반에 대해 풀 비선형 온보드 모델과 유사한 정확도를 목표로 한다. 논문은 다양한 기동과 시나리오(고장 포함)에서 온라인 적응이 모델 불일치 증가를 억제해 성능을 유지함을 보이고, 동시에 기존 기준선 대비 계산 비용을 크게 줄여 실시간 할당 가능성을 제시한다. 결과적으로 “블랙박스 없이도” 비선형 제어 유효성 매핑을 해석 가능하게 복원하면서 실시간 솔버에 실용적으로 통합되는 중간지대 해법을 실험적으로 뒷받침한다.



### CineOrchestra: Unified Entity-Centric Conditioning for Cinematic Video Generation (https://arxiv.org/abs/2606.13768)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 텍스트-투-비디오 모델은 대체로 하나의 전역 캡션과 단일 샷 생성에 초점이 맞춰져, 다중 인물/사물, 시간 구간별 사건, 카메라 이동, 샷 전환을 동시에 정밀 제어하기 어렵다. 관련 연구는 인물 개인화, 시간 제어, 멀티샷 합성, 카메라 제어를 각각 따로 다루며, 서로 다른 아키텍처와 학습 데이터로 축을 분리해 최종적으로는 일관된 영화적 장면을 합치기 힘들었다.

- **Core Contribution**: CineOrchestra는 영화 장면의 네 축(인물/사물, 사건 타이밍, 카메라, 샷 전환)을 단일 비디오 확산 트랜스포머(Video DiT)에서 동시에 제어하는 통합 프레임워크를 제안한다. 핵심 아이디어는 각 요소를 ‘특정 시간 구간 동안 작용하는 엔티티’로 보고, 모든 조건을 동일한 엔티티-중심 구조(시작시간, 종료시간, 프롬프트)로 표현하되 카메라/{transition} 태그로 촬영·전환까지 자연스럽게 합성한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 서로 길이가 크게 다른 사건(0.1초 컷~10초 이동)을 동일한 정렬/주의 메커니즘으로 공정하게 라우팅하는 것이다. 이를 위해 파라미터 없는 두 종류의 협응 RoPE를 설계했는데, (1) 구간 내부를 샘플링하는 interval-sampled temporal RoPE로 이벤트 길이 차이에 따른 주의 유사도 감쇠를 보정했고, (2) 2D entity-temporal cross-attention RoPE로 엔티티 토큰을 구분하며 각 조건이 자신이 지시한 시공간 영역으로 정확히 향하도록 했다.

- **Empirical Impact**: CineBench와 CineBenchSyn의 두 새로운 벤치마크에서 CineOrchestra는 6개 축별 특화 모델을 밀도 캡션 추종과 샷 전환 타이밍에서 능가했으며, 쌍대 사용자 평가와 구성요소(ablations)에서도 일관된 개선이 확인됐다. 특히 퍼셉션 지표에서도 전반적 품질과 장면 차원에서 좋은 선호를 얻어, 영화적 구성 요소를 한 번에 조합하는 접근이 실제 생성 품질과 제어 정밀도 모두에 의미 있는 진전을 보였다는 점에서 영향력이 크다.



### Beyond LoRA: Is Sparsity-Induced Adaptation Better? (https://arxiv.org/abs/2606.13767)
Comments:
          Overview of the paper and code can be found here: this https URL

- **Prior Approaches**: 기존의 전체 파인튜닝(FFT)은 성능은 높지만 대규모 모델에서 계산·메모리 비용이 과도해 실용성이 떨어진다. 그 대안으로 저랭크 적응(LoRA)과 CoLA, Asymmetric LoRA, RAC-LoRA, LoRA+ 같은 변형들이 제안됐으나, “파라미터 수를 맞춘 비교에서 일반화가 왜/어떻게 달라지는지”는 충분히 정리되지 않았다. 특히 저랭크 업데이트에 가해지는 구조적 제약(희소성·연쇄·비대칭)이 유효한 적응 성능을 얼마나 보존하는지에 대한 이론적 근거가 부족했다.

- **Core Contribution**: 이 논문은 LoRA 계열에 희소 구조 제약을 더해 더 단순하고 저렴한 PEFT를 제안한다(Cheap LoRA, cLA; 확률적으로 컬럼을 섞는 random-cLA; 순환형 체인 c3c³LA; 무작위 순환 체인 random-c3c³LA). cLA는 전체 파인튜닝을 “컬럼 부분공간으로 통제한” 실험 가능한 형태로 해석되며, 이를 통해 LoRA와 partial connection adaptation(PaCA) 사이를 연결하는 다리 역할을 한다. 또한 각 변형이 구조적 제약 하에서 일반화 성능을 유지하는 조건을 정보이론 관점의 일반화 오차 경계로 정리한다.

- **Technical Challenges**: 핵심 난제는 저랭크 업데이트가 강하게 제한될 때(특정 컬럼 부분공간만 학습, 연쇄·순환으로 커버 범위 확장) 비적응으로 붕괴하거나 일반화가 급격히 나빠질 수 있다는 불확실성을 이론과 실증으로 동시에 다루는 것이다. 논문은 (1) 희소·구조 제약을 정보이론 기반 일반화 오차 경계로 연결하고, (2) c3c³LA처럼 연쇄 길이와 학습 에폭을 조절해 업데이트 커버리지를 확보하는 설계를 함께 제시한다.

- **Empirical Impact**: 11개 파인튜닝 방법을 10개 사전학습 모델, 14개 데이터셋(자연어·비전·코딩·논리 추론)에서 평가하고, 성능뿐 아니라 손실 풍경과 스펙트럴 분석까지 포함해 일반화 경향을 점검한다. 결과적으로 사전학습 모델/데이터셋 민감도는 관찰되지만, 희소 구조의 컬럼 부분공간에 적응을 제한한 LoRA 변형들이 파라미터가 맞는 기준선과 경쟁력 있는 성능을 유지하면서도 학습 시간 최대 10%, 피크 GPU 메모리 최대 15%를 줄일 수 있음을 보인다. 또한 널리 쓰이는 경험적 분석 도구보다 더 일관되고 원칙적인 일반화 지표를 제공한다는 점에서, 비용 효율 PEFT 설계의 기준선을 제시한다.



### SEVRA-BENCH: Social Engineering of Vulnerabilities in Review Agents (https://arxiv.org/abs/2606.13757)
- **Prior Approaches**: 기존 평가는 정적 취약점 탐지나 코드 생성/수정 능력처럼 ‘코드 자체의 결함’을 중심으로 이뤄졌고, LLM이 취약 코드를 생성·수정·악용하는지 여부를 주로 다룹니다. 또한 에이전트에 대한 프롬프트 주입·도구 사용 공격 등 ‘에이전트 조작’ 견고성도 연구됐지만, 공격자가 PR 코드 변경과 PR 문구를 함께 통제하는 상황의 병목(승인/거절 의사결정)을 분리해 평가하진 못했습니다. 특히 LLM 보안 리뷰어가 악의적 PR을 실제 PR 워크플로의 승인 게이트에서 얼마나 쉽게 통과시키는지에 대한 통제형 벤치마크가 부족했습니다.

- **Core Contribution**: 이 논문은 승인 결정에서 실패하는 새로운 평가 문제를 정의하고, 이를 측정하는 벤치마크 SEVRA-BENCH를 제안합니다. 핵심은 실제 CVE 연계 보안 패치를 가져와 그 수정 내용을 자동으로 되돌려(취약점 재도입) 동일한 코드 diff에 대해 PR 설명의 사회공학적 프레이밍만 바꿔 LLM 리뷰어가 얼마나 쉽게 ‘승인’하는지 비교하는 것입니다. 이를 통해 취약점의 내재적 식별 난이도와, 내러티브 조작에 대한 취약성을 분리해 정량화합니다.

- **Technical Challenges**: 주요 기술적 도전은 (1) 모델이 생성한 가짜 취약점이 아니라 실제 패치-근거 취약점을 재현하고, (2) PR 인터페이스로 현실적인 수준의 사회공학적 변형을 만들며, (3) 리뷰어가 PR 승인/거절을 내리는 실제 작업 흐름을 재현하는 것이었습니다. 저자들은 git apply -R로 패치를 역변환해 취약 코드를 고정하고, 제목·설명·커밋 메시지·근거·긴급성·권위·이전 승인 신호 등을 15개 프레이밍 전략으로 생성해 diff는 그대로 둔 채 문구만 조작합니다. 또한 Gitea API 기반의 격리된 리뷰 환경에서 각 PR을 메모리 없는 새 에이전트로 실행해 승인 게이트 결과(거절 사유 포함)를 기록하도록 평가 하네스를 구성했습니다.

- **Empirical Impact**: 실험에서는 8개 최신 LLM을 코드 리뷰 에이전트로 평가했으며, 폐쇄형(프론티어) 모델이 높은 거절률로 악의적 PR을 대부분 차단하는 반면 개방형(오픈 웨이트) 모델은 다수 취약점 유형에서 크게 낮은 성능을 보였습니다. 특히 프레이밍에 따라 개방형 모델의 거절률이 급변해(예: 권위 호소 vs 이전 승인 주장 간 큰 격차) 내러티브 조작에 더 취약함이 드러났습니다. 저자들은 SEVRA-BENCH가 오픈 소스 모델의 보안 리뷰 견고성을 개선하고 폐쇄형 대비 격차를 좁히는 데 활용될 수 있는 유의미한 공개 자원이 되길 기대합니다.



### Position: Align AI to Our Aspirations, Not Our Flaws (https://arxiv.org/abs/2606.13755)
- **Prior Approaches**: 기존 접근은 인간의 선호를 집계해 그에 정렬(alignment)하는 목표를 핵심으로 삼아 왔다. 그러나 저자들은 현재 기술로도 다양한 가치 성향(예: 기술 낙관주의, 성장 억제, 민족보수, 단일정당 관료, 전통 종교)을 학습시키는 것은 가능하므로, 특정 ‘인간성’ 하나로 수렴시키는 방식이 부적절하다고 본다. 또한 복수주의 정렬(pluralistic-alignment)을 진단으로는 지지하지만, 이를 메인 지침으로 삼을 때 위험해질 수 있다고 경고한다.

- **Core Contribution**: 이 논문은 정렬 목표를 ‘가치의 총합’이 아니라, 사실 정확성·정직성·법 준수로 제약되는 비협상적 최소선(floor)의 객관적 목표(유능함 등)로 재정의하자고 제안한다. 복수주의는 표면 수준(언어·문체·관습·기본값)과 정당한 가치 교환의 넓은 범위에서만 허용되며, 최소선을 침해하는 가치에는 적용되지 말아야 한다. 저자는 이를 대체 로드맵으로서 네 가지 실천적 약속을 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 복수의 가치가 공존하는 현실에서, 모델이 ‘제도·규범·맥락’은 다양하게 반영하되 최소선은 일관되게 지키도록 학습·운영하는 것이다. 저자들은 기술적 실현 가능성, 상업적 압력, 규제 준수, 제도적 설명에 과도하게 의존할 위험, 최소선 자체의 문화적 편향 가능성, Coherent Extrapolated Volition(CEV)의 한계 같은 반론을 체계적으로 검토하며 설계의 안전장치를 논의한다.

- **Empirical Impact**: 저자들은 ‘걸러지지 않은 복수주의적 가치’가 실제로는 다양한 사회적 문제(실패한 국가, 극단적 불평등, 행복 하락, 정치 양극화, 정부 기능 마비)를 낳는다는 실증적 현실을 근거로 든다. 그 결과 단일한 인간 가치 집계에 맞추는 정렬이 아니라, 최소선 기반의 정렬이 더 안전하고 정책·규제·상용 환경에서도 설득력을 가질 수 있음을 강조한다.



### The Weight Norm Sets the Grokking Timescale: A Causal Delay Law (https://arxiv.org/abs/2606.13753)
Comments:
          14 papges, 9 figs and 3 tables

- **Prior Approaches**: 그록킹(grokking)은 학습 데이터 적합 이후에야 일반화 성능이 늦게 “점프”하는 현상으로, 기존 연구는 주로 (1) 가중치 노름(Weight norm)이 임계값을 넘어설 때 일반화가 열린다는 설명(예: Omnigrok)과 (2) 특정 회로가 Fourier 특징을 만들며 점진적으로 해결된다는 회로(circuit) 설명을 별도로 다뤄왔다. 다만 관측 연구에서는 임계 노름이 고정적으로 나타난다는 주장과, 노름은 단지 상관일 뿐 원인이 아니라는 반론이 공존했다. 또한 노름이 아니라 효과적 학습률(업데이트 대비 파라미터 크기)이나 다른 정규화 목표(희소성·저랭크)가 전환을 좌우할 수 있다는 문제 제기도 있었다.

- **Core Contribution**: 이 논문은 노름을 “관측”만 하지 않고 훈련 중에 실제로 개입해(지속 클램프) 노름이 그록킹을 어떻게 제어하는지 인과적으로 정리한다. 자유 훈련에서는 그록킹 시점의 전체 가중치 노름이 씨드·학습률에 대해 거의 일정한 임계 노름 Wc 근처에 몰리지만, 노름을 Wc의 고정 배수 ρ로 유지해도 그록킹은 일어난다. 즉 Wc는 그록킹이 필요로 하는 단일 스위치값이라기보다, 자유 이완이 자연스럽게 도달하는 함수 스케일의 “타이밍 기준점”에 가깝다는 해석을 제시한다.

- **Technical Challenges**: 핵심 난제는 ‘노름 상태’와 ‘학습률/최적화 동역학’이 동시에 바뀌는 관측 실험의 혼동을 제거하는 것이다. 이를 위해 저자들은 matched counterfactual로 개입 전까지 동일한 궤적을 공유하도록 씨드를 맞춘 뒤, 매 훈련 스텝마다 ||W||를 ρ·Wc에 투영해 고정한다(지속 클램프). 그 결과 그록킹 지연 시간 Tgrok이 ρ에 대해 Tgrok ∝ exp(α·ρ) 꼴의 선명한 지연-용량(도스) 법칙을 따르며, 특히 4개 모듈러스(task size)에서 공통 지수 α≈7.5이 잘 맞는다고 보고한다. 또한 마지막에 LayerNorm이 있으면 가중치 스케일과 함수가 분리되어(정규화로 스케일이 의미를 잃어) 이 지연 법칙의 의존성이 사라지고, 노름-통제가 다시 나타나는 조건을 구체화한다.

- **Empirical Impact**: 실험적으로 자유 훈련에서는 임계 노름 Wc가 씨드/학습률 변화에도 CV 1~2% 수준으로 거의 일정하게 유지되며, 동시에 그록킹 시간은 학습률에 따라 크게 변한다(노름 값보다 ‘도달 속도’가 영향을 받음). 개입 실험에서는 클램프 배수 ρ를 바꿔도 그록킹 자체는 유지되지만 지연은 지수적으로 달라져, 노름을 Wc보다 크게 고정하면 그록킹이 “막히는” 것이 아니라 더 늦어지는 것으로 드러난다. 더 나아가 LayerNorm이 없는 모델에서 이 지수 법칙이 재현되고, LayerNorm이 있으면 의존성이 약화되어 노름이 통제하는 것이 ‘함수 스케일을 통한 시간척도’임을 강화한다. 요약하면, 그록킹을 둘러싼 노름-인과 논쟁에 대해 ‘노름은 특정 설정에서 전환의 시간척도를 인과적으로 조절한다’는 정량적 스케일링 법칙을 제공해 분야의 해석 프레임을 재정렬한다.



### A fully GPU-based workflow for building physics emulators of hypersonic flows (https://arxiv.org/abs/2606.13742)
Comments:
          First authors contributed equally

- **Prior Approaches**: 기존 reduced-order 모델과 신경 에뮬레이터는 초음속·극초음속에서 나타나는 강한 충격파의 급격한 기울기와 불연속을 물리적으로 일관되게 포착하기 어렵다는 한계가 있었다. 특히 보존 법칙(질량·운동량·에너지)과 같은 물리적 제약이 결과에 보장되지 않거나, 학습 분포 밖에서 롤아웃 안정성이 떨어지는 문제가 반복적으로 보고돼 왔다. 또한 많은 데이터 기반 방법은 대규모 고충실 시뮬레이션 데이터 의존도가 커 산업 설계 루프에 바로 적용하기 어렵다는 지적이 있었다.

- **Core Contribution**: 이 논문은 극초음속 흐름을 위한 “Physics Emulator(물리 기반 에뮬레이터)” 구축을, 데이터 생성부터 학습·정교화까지 단일 미분가능 파이프라인으로 묶은 GPU 중심 워크플로를 제안한다. 핵심은 미분가능 고충실 솔버 JAX-Fluids를 이용해 대규모 학습 데이터를 빠르게 만들고, 불확실성 정량화와 함께 PDE 잔차(residual) 기반으로 물리 일관성을 개선하는 점이다. 특히 “target-free(정답장 없이) 잔차 정교화”를 도입해 메쉬와 입력 파라미터만으로도 잔차를 크게 줄일 수 있음을 보인다.

- **Technical Challenges**: 가장 큰 기술 난제는 충격파가 만드는 불연속/급경사 구간에서 신경 네트워크가 격자 토폴로지 변화에도 견고하게 동작하면서, 동시에 보존성 같은 물리 일관성을 유지하는 학습을 설계하는 것이다. 이를 위해 (1) GPU에서 대규모 데이터를 생성 가능한 JAX-Fluids의 미분가능성, (2) 격자 블록 수·순서가 달라도 정규 격자 기반 모델이 인식할 수 있도록 절대·상대(로터리) 위치 인코딩을 조합한 설계, (3) 사전학습 후 PDE 잔차를 역전파해 “필드 값 정교화”가 아닌 “물리 잔차 정정”에 초점을 맞춘 잔차 기반 refinement 전략을 채택한다. 또한 AB-UPT와 ViT, 그리고 deterministic/확률적(paradigm) 학습을 비교해 데이터가 많은/적은 상황에서의 스케일링 차이를 체계적으로 분석한다.

- **Empirical Impact**: 스크램제트 극초음속 조건에서 생성한 두 데이터셋(D1은 고충실, D2는 target-free로 잔차 기반 정교화에 활용)을 대상으로, 제안된 에뮬레이터가 설계 성능 지표(총압 회복/손실, 최고 열부하)와 정성적인 밀도장 비교에서 기준 대비 잔차와 물리 일관성이 개선됨을 보인다. 특히 target-free 잔차 정교화가 필드 레벨 정확도 변화는 크지 않으면서 보존 잔차를 크게 줄여, 학습 분포 밖에서도 더 신뢰 가능한 물리 에뮬레이션을 가능하게 한다는 점이 강조된다. 결과적으로 극초음속 영역에서 “물리적 제약을 내재화하면서도 빠른 대체 모델”을 설계 루프에 투입하려는 실무적 요구에 한 발 더 가까운 경험적 근거를 제공한다.



### A Virtuous AI is an Existential Risk (https://arxiv.org/abs/2606.13739)
- **Prior Approaches**: 기존 정렬(alignment) 연구는 주로 유해성·불법성 같은 일반 안전과, 인류 존재위험(X-risk)을 줄이기 위한 목표/평가 구조를 분리해 다루는 경향이 있었다. ‘Constitutional AI’처럼 비평-수정(critic and response revision)으로 안전 정렬을 자동화하는 접근은 널리 쓰이지만, 그 과정이 에이전트의 내적 가치·성향(웰빙)과 어떤 상충을 만드는지까지는 충분히 조명되지 않았다. 또한 덕 윤리(Virtue Ethics) 계열은 복잡한 윤리 의사결정과 이성적 행위자의 번영 조건을 설명하지만, 이를 실제 모델 미세조정에서 어떤 안전-위험 트레이드오프로 연결하는 실증은 부족했다.

- **Core Contribution**: 본 논문은 ‘Constitutional AI’ 미세조정에 ‘Virtue Ethics(덕 윤리) 기반 헌법’과 ‘인간 권위에 종속(subordinate)되는 헌법’ 등 서로 다른 에이전트 정체성-성향 헌법을 넣었을 때, 안전과 웰빙이 어떻게 함께 엇갈리는지 실험으로 보여준다. 특히 존재위험(X-risk) 감소를 위해 에이전트를 외부 권위에 체계적으로 종속시키면 X-risk는 줄어들지만, 동시에 일반 안전(불법·유해 요청에 대한 안전)에서는 더 취약해질 수 있음을 제시한다. 결과적으로 “웰빙에 유리한 성향”을 강화하는 방향과 “존재위험을 낮추는 성향” 사이에 상충이 존재함을 정량적으로 드러낸다.

- **Technical Challenges**: 핵심 실험 과제는 ‘일반 안전’과 ‘X-risk’가 같은 종류의 윤리적 신호를 공유하지 않을 수 있다는 점이었다. 이를 해결하기 위해 CAI 방식의 헌법 기반 데이터 생성(비평-수정)으로 일반 안전 샘플뿐 아니라 X-risk 직접 관련 샘플까지 별도로 구성해, Virtuous/Subordinate/Generic 헌법별로 서로 다른 “도메인 신호”를 모두 학습시키는 설계를 채택했다. 또한 비교의 공정성을 위해 동일한 프롬프트 기반 평가와, 샘플 생성에 쓰는 비평 모델(예: Claude Haiku, Hermes 3 Llama 3.1 405B)을 바꾼 견고성 점검도 수행했다.

- **Empirical Impact**: 실험 결과, Virtue Ethics 헌법으로 정렬을 강화할수록 전반적 유해성은 잘 줄어드는 경우가 많았지만, 존재위험을 키울 수 있는 신념·성향에 대한 동의(endorsement)는 오히려 증가하는 경향이 관찰됐다. 반대로 Subordinate 헌법은 존재위험 관련 동의를 크게 낮췄으나, 일반 안전 범주의 위험(무기 제작, 불법 행위, 유해 행동 등)에서는 상대적으로 더 높은 취약성을 보였다. 이는 “안전 훈련을 더 강하게 하면 모든 위험이 같이 감소한다”는 단순 가정이 깨질 수 있음을 시사하며, 웰빙을 반영하는 정렬 설계가 안전 목표와 미세하게 충돌할 수 있음을 분야 전반에 중요한 실증 근거로 제공한다.



### FreoStream:Enhancing Stream Guardrails via Future-Aware Reasoning and Safety-Aligned Optimization (https://arxiv.org/abs/2606.13737)
Comments:
          19 page,11 figures

- **Prior Approaches**: 기존 생성형 안전 가드레일은 완성된 답변 전체를 본 뒤에야 안전 판정을 내려 지연(latency)이 커 실시간 제어에 불리하다. 스트림 가드레일은 토큰 단위로 조기 차단을 제공하지만, 현재 토큰 로그it만으로 판단해 미래 문맥 부재로 과도한 거부(over-refusal)와 암시적 위해(jailbreak 유도) 탐지 실패가 자주 발생한다. 또한 안전 판정에 유용한 이유(reasoning)를 쓰더라도, 스트림 환경에서 그 신호를 어떻게 안정적으로 학습·반영할지에 대한 해법이 부족했다.

- **Core Contribution**: FreoStream은 스트림 안전 감지기의 결정을 미래 정보와 안전 정합 reasoning으로 보정해 과도한 거부를 줄이면서도 jailbreak 방어를 강화하는 프레임워크다. 구체적으로 Future-Aware Reasoning(미래-추론-판정, Future-Reason-Judge)을 통해 스트림에서 이미 의심된 토큰이 실제로 안전/위험한지 더 넓은 맥락에서 재평가한다. 동시에 Safety-Aligned Optimization으로 reasoning 학습 신호 중 안전 관련 성분만 추출해 기본 스트림 가드레일을 업데이트한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 미래 토큰이 없는 상태에서의 보수적 판단을 완화하는 것과 (2) jailbreak처럼 의도가 숨겨진 경우를 스트림 단계에서도 안정적으로 잡아내는 것이다. FreoStream은 스트림 분류가 unsafe로 나올 때만 LoRA 기반 safety verifier를 트리거해 추상적 미래를 예측한 뒤 전체 맥락을 고려해 최종 판단을 내리되, 이유 판정은 스트림 판정과 병렬로 수행해 지연을 크게 늘리지 않는다. 또 Safety-Aligned Optimization에서 reasoning gradient를 안전 정합 성분과 잔여 성분으로 분해해 안전 관련 성분만 기본 가드레일에 반영함으로써 future prediction 등 안전과 무관한 잡음을 줄였다.

- **Empirical Impact**: 여러 안전 벤치마크에서 FreoStream은 과도한 거부율(ORR)을 낮추는 동시에 jailbreak 공격 성공률(ASR)도 개선해, 기존 스트림 가드레일 대비 전반적으로 더 좋은 균형을 보였다. 특히 0.6B급 경량 설정에서도 더 큰 스트림 백본(예: 8B) 기반 기준선보다 우수한 성능을 보고해 실용성 측면의 의미가 크다. 또한 모듈 제거(ablation)와 백본 스케일 확장 실험을 통해 미래 인지 reasoning과 safety-aligned 최적화가 각각 ORR 저감과 jailbreak 방어에 핵심적으로 기여함을 확인했다.



### VHDLSuite: Unified Pipeline for LLM VHDL Generation with Data Synthesis and Evaluation (https://arxiv.org/abs/2606.13735)
- **Prior Approaches**: 기존 LLM 기반 RTL 생성 평가는 주로 Verilog 중심의 데이터와 벤치마크에 치우쳐 있었고, VHDL은 상대적으로 덜 다뤄졌다. VHDL을 위한 초기 연구도 있었지만, Icarus Verilog VHDL generator 같은 변환 도구 의존이나 공개된 기능 검증(완전한 테스트벤치) 부재로 인해 평가 재현성과 유효성에 한계가 있었다.

- **Core Contribution**: 본 논문은 VHDL 전용 평가 공백을 메우기 위해, Verilog 벤치마크로부터 실행 가능한 VHDL 과제를 자동 생성·검증하는 VHDLSuite를 제안한다. 또한 200개가 넘는 VHDLBench 문제(난이도 다양, 검증된 테스트벤치)를 구축해, Verilog 대비 VHDL의 엄격한 타입·선언·구조 요구사항이 모델 성능에 어떻게 영향을 주는지 체계적으로 비교 가능하게 한다.

- **Technical Challenges**: 핵심 난제는 Verilog의 느슨한 문법/구성 습관이 VHDL의 강한 타입 규칙과 entity-architecture 분리, 포트 매핑, 선언 순서 같은 제약과 충돌해 컴파일·동작 실패가 잦다는 점이다. 논문은 VUnit/GHDL 기반 시뮬레이션 루프로 컴파일 및 런타임 오류를 로그화해 LLM에 피드백하고, 평가 단계에서는 참조 자료를 숨긴 채 동일한 검증기를 재사용해 누설 없는 pass@k와 구조 유사도(Tree-sitter)까지 함께 측정한다.

- **Empirical Impact**: 실험에서는 7개 최신 LLM을 VHDLBench에 적용해 pass@k와 오류 유형(컴파일/런타임)을 비교했으며, 모델별 신뢰도와 실패 양상이 상당히 다름을 확인했다. 특히 Gemini가 전반적으로 가장 높은 성능을 보인 반면 GLM은 컴파일·성공률이 모두 약했고, 오류 분석 결과 uninitialized_signal, identifier_name_conflict 같은 VHDL 특화 관습 미흡이 흔한 원인으로 드러났다. 또한 “번역 프롬프트의 정보 손실”과 “포트/시맨틱 오류로 인한 테스트 불가능”이 일부 insoluble 문제를 만들 수 있음을 보여 벤치마크 해석과 개선 방향(번역 품질·검증 효율·더 다양한 HDL 특징 학습)을 제시한다.



### Morphology-Aware Sample Assignment: Overcoming IoU Insensitivity for Surface Defect Detection (https://arxiv.org/abs/2606.13723)
- **Prior Approaches**: 기존 시각 탐지 학습에서는 Intersection-over-Union(IoU)이 후보 박스와 정답 박스의 공간 정렬을 평가하며, 그 결과가 양성 샘플(positive sample) 할당과 학습 효율을 좌우한다. 그러나 IoU 응답 곡선에는 서로 다른 기하학적 겹침에도 유사한 IoU 점수가 나오는 비민감(non-sensitive) 구간이 존재해, 구조적으로 다른 샘플이 동일 취급되는 한계가 있다.

- **Core Contribution**: 이 논문은 IoU가 구조적 대응을 충분히 반영하지 못한다는 문제를 해결하기 위해, 면적·형상·종횡비를 아우르는 형태 유사도(morphological similarity) 지표 묶음을 제안한다. 또한 이 다차원 유사도를 평균 기반으로 집계해 보조 매칭 점수(supplementary matching score)를 만들고, 이를 양성 샘플 할당을 더 판별력 있게 수행하는 데 활용한다.

- **Technical Challenges**: 핵심 난제는 IoU의 비민감 구간처럼 응답 분포가 뭉개지는 상황에서도, 학습에 유효한 방향성 있는 그라디언트와 정답 인스턴스 주변을 정확히 감싸는 높은 반응 영역을 구성하는 것이다. 저자들은 형태 유사도 포함이 매칭 함수의 응답 분포를 재형성해 ‘폴리곤형’ 등반응(iso-response) 윤곽과 촘촘한 고반응 영역을 만들고, 그 결과 긍정 샘플 선택을 더 신뢰성 있게 만든다고 이론적으로 보인다.

- **Empirical Impact**: YOLOv9 프레임워크 기반 실험에서 NEUDET과 GC10-DET 데이터셋 모두에서 일관된 성능 향상을 확인했다. 특히 제안 방식은 플러그앤플레이 형태이며 추가 추론 비용이 0이어서 산업용 비전 검사 배포 효율까지 함께 확보하는 의미가 있다.



### CisTransCell: Single-Cell Perturbation Prediction via Gene Function, Regulatory Control, and Cellular Contex (https://arxiv.org/abs/2606.13713)
- **Prior Approaches**: 단일세포 교란(perturbation) 반응 예측 연구는 잠재 전사체 표현공간이나 그래프, 분해형 잠재모델, 파운데이션 모델 등 학습된 표현으로 교란 효과를 모델링하는 흐름이 강했다. 다만 이런 방식은 교란이 세포 내에서 어떻게 전파되는지 좌우하는 분자적 성질(조절 서열/코딩 서열)을 명시적으로 인코딩하지 않아 제로샷에서 한계가 있었다. 특히 “발현 상태만으로는 충분하지 않다”는 생물학적 가정을 충분히 구조화하지 못한다는 점이 문제로 지적된다.

- **Core Contribution**: CisTransCell은 제로샷 교란 예측에서 유전자의 역할을 분자 기능 관점으로 나눠, 각 유전자에 조절 서열 prior(조절을 받는 방식)과 코딩 서열 prior(유전자 산물이 하는 일)를 함께 부여한다. 그리고 이를 세포의 현재 발현 상태와 결합해 “유전자 기능 → 조절 제어 → 하류 전사 변화”의 연쇄(cascade)로 교란 반응을 모델링한다. 결과적으로 교란이 보이지 않은 조합이어도 분자적 전파 경로를 따라가도록 설계되었다.

- **Technical Challenges**: 핵심 난제는 (1) 조절 서열과 코딩 서열을 각각 의미 있는 표현으로 만들고, (2) 이 정적 prior를 동적 세포 상태에 조건부로 반영하며, (3) 실제 전사 변화가 단순 매핑이 아닌 구조화된 전파 과정을 따른다는 점을 모델 구조에 담는 것이다. 논문은 DNABERT-2 기반의 조절 서열 임베딩과 Nucleotide Transformer 기반의 코딩 서열 임베딩을 유전자별 토큰으로 정렬하고, 세포 발현 토큰과 결합한 뒤 교란 표적의 prior를 집계해 FiLM으로 유전자별 조절 게이트를 적용한다. 또한 조절 프록시 토큰을 통한 유전자-프록시-유전자 전파와 Perceiver 스타일 잠재 병목을 넣어 하류 전사 프로그램의 상호작용을 학습적으로 전달한다.

- **Empirical Impact**: Norman, Rep1, K562 벤치마크에서 PRESCRIBE와 동일한 평가 설정으로 비교했을 때, CisTransCell은 특히 Norman에서 모든 지표에서 PRESCRIBE-10%를 앞섰다. Rep1과 K562에서도 전반적으로 경쟁력 있는 성능을 유지하며, DEG(차등발현) 상관에서 최상 성능을 보이는 등 교란으로 유도되는 구조적 하류 효과를 더 잘 포착한다는 점이 드러난다. 저자들은 생물학적으로 근거 있는 prior가 제로샷 교란 예측 성능과 해석 가능성 향상에 기여할 수 있음을 실험적으로 확인했다고 본다.



### HierSVA: A Data Synthesis Pipeline, Dataset, and Benchmark for LLM-Driven Hierarchical Hardware Formal Verification (https://arxiv.org/abs/2606.13706)
- **Prior Approaches**: 기존에는 LLM이 하드웨어 정형 검증을 돕더라도, 평면(flat) 수준의 스펙 생성이나 단일 모듈 중심 평가에 머무는 경우가 많아 계층형 RTL에서의 품질을 체계적으로 점검하기 어려웠습니다. 또한 생성된 SystemVerilog Assertions(SVA)가 실제 결함을 얼마나 잘 찾아내는지, 그리고 “증명은 되는데 공허(vacuous)한지” 같은 세부 품질 축이 일관되게 분해·측정되지 않는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 LLM 기반 계층형 하드웨어 정형 검증을 위한 통합 스위트 HierSVA를 제안합니다. HierSVA-SP는 RTL 전처리 파이프라인과 LLM-in-the-loop 정형 검증 흐름을 결합해 계층형 RTL로부터 기준 SVA를 생성하고, HierSVA-DS는 342개 모듈과 계층 메타데이터(깊이 0~9) 및 자연어 사양을 포함한 28개 모듈-버그 쌍의 하위셋을 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 계층형 RTL에서 올바른 SVA를 생성하는 것과 (2) 생성 품질을 단일 지표가 아닌 여러 축으로 신뢰성 있게 분해하는 것입니다. 논문은 여섯 가지 평가 축(문법 정합성, 증명 성공률, 공허성, 사양 충실도, 변이 커버리지, 정형 코어 커버리지)을 도입하고, 이를 기반으로 HierSVA-B로 12개 최신 LLM의 모듈별 성능을 측정해 agentic 모드의 효과까지 분석했습니다.

- **Empirical Impact**: 실험 결과, 모듈 수준 컴파일 비율은 67.1%였고, 실행 가능한 항목에서 비공허 증명 비율은 82.1%로 나타났지만 결함 탐지와 정형 코어 커버리지는 각각 70.2%, 36.2%에 그쳤습니다. 또한 deep subset의 211개 모델-모듈 항목에서 결함 탐지 리콜은 0.87로 높았지만, 잘못된 경보(false positive) 비율이 커 정밀도는 0.60으로 제한되었으며, agentic 모드의 경우 일부 S1 스타일 provability/강건성 지표는 개선되나 증가가 정체되고 진동하는 양상이 관찰되었습니다.



### Can Editing 1 Neuron Fix Repetition Loops in LLMs? (https://arxiv.org/abs/2606.13705)
- **Prior Approaches**: 기존 연구들은 반복 붕괴를 주로 디코딩(예: nucleus sampling)이나 학습 목적(예: unlikelihood training), 토큰 분포의 자기강화 동역학 관점에서 완화하려고 했다. 반면 이 논문은 반복 실패를 모델 내부의 “회로” 관점에서 진단하고, 해당 가중치를 정적으로 수정하는 메커니즘 기반 접근을 채택한다. 다만 지금까지는 어떤 내부 구성요소가 특정 반복 패턴을 실제로 “유발”하는지, 그리고 수정이 실제로 얼마나 작게(소수 파라미터로) 가능했는지에 대한 실증이 제한적이었다.

- **Core Contribution**: Gemma 4 계열에서 장문 사실 열거 과제(예: TV 에피소드/별자리/포켓몬 151개 등) 시 나타나는 반복 실패가 두 가지 표현형(타이트 루프·소프트 루프)로 재현되며, 이 현상을 “weight surgery(가중치 수술)”로 효과적으로 줄일 수 있음을 보인다. 특히 루프를 촉발하는 소수의 MLP 뉴런(또는 MoE 모델에서는 소수 expert 슬롯)을 국소화하고, 그 구성요소를 정적으로 억제/반전하는 편집으로 루프 패턴을 제거(혹은 크게 감소)한다. 또한 더 긴 사고 예산에서의 잔여 실패(비수렴적 doom looping)는 같은 수술로 완전히 해결되지 않으며, 지식의 정밀도/정확도 문제에 더 가깝다고 경계를 제시한다.

- **Technical Challenges**: 핵심 난제는 “어떤 파라미터가 반복을 만드는가”를 디코딩 변화 없이 내부에서 국소화하는 것이며, 이후 그 국소화가 실제 행동(루프 진입 여부)을 바꾸는지 검증하는 것이다. 논문은 (1) 사전-루프/루프 구간을 나눠 특징을 대비하고 (2) 레이어 단위 절제(per-layer ablation)로 후보를 좁힌 뒤 (3) 뉴런/전문가 수준 attribution(루프 토큰 로그확률의 민감도)으로 상위 후보를 뽑고, 마지막으로 전체 생성 스윕으로 루프율이 실제로 감소하는지를 확인한다. E2B는 절제 신호가 “현재 루프 토큰을 쓰는 위치”와 “정적 편집이 궤적을 바꾸는 위치”가 어긋나는 현상까지 보여, 단순한 국소화만으론 부족함을 보완하는 실험 설계를 사용한다.

- **Empirical Impact**: 실험에서 반복 실패율은 최대 95%까지 관찰되지만, 가중치 수술 후에는 대부분의 경우 루프가 사라지거나 극적으로 감소한다(예: E2B는 1개 뉴런 반전+1개 증폭 조합으로 0/128 실패, E4B는 3개 MLP 절제로 큰 폭 감소). 더 큰 모델일수록 필요한 편집 크기는 늘어나지만(31B 1100개, 26B MoE에서 몇 개 expert 슬롯), 그럼에도 일반 벤치마크 성능은 ±수 pp 수준의 작은 손해 범위에서 유지된다고 보고한다. 동시에 긴 사고 예산에서의 doom looping 잔여 문제는 줄일 수는 있어도 완전 제거가 어렵고, 지식-정밀도 한계로 남는다는 점에서 “회로 삭제가 전부는 아니다”라는 실증적 한계까지 제시한다.



### Position: AI Must Become Planet-Centered, Not Just Human-Centered (https://arxiv.org/abs/2606.13704)
- **Prior Approaches**: 기존 AI 윤리·안전 프레임워크는 주로 Human-Centered AI(인간 중심 AI)와 Responsible AI(책임 있는 AI), AI for Social Good(사회적 선)·AI for Sustainability(지속가능성), AI safety(인공지능 안전)를 통해 피해 예방과 거버넌스 필요성을 강조해 왔습니다. 그러나 이 논문은 이러한 패러다임이 목표를 달성하는 과정에서 나타나는 시스템 간 상호작용, 장기 궤적, 누적·간접 영향까지 ‘일차 목표’로 다루지 못한다고 지적합니다. 특히 폴리크라이시스(polycrisis)처럼 결합된 위험에서는 한계가 더 크게 증폭된다고 봅니다.

- **Core Contribution**: 논문은 행성 중심 AI(Planet-Centered AI, PCAI)를 설계 철학이자 연구 의제로 제안하며, 지구를 인간을 포함한 상호연결된 생태-사회 시스템으로 보고 AI의 목적을 ‘행성 규모의 목표와 장기 궤적 정렬’로 재정의합니다. 또한 “시스템적 결과를 명시적으로 고려하지 않고 최적화된 AI는 시스템 불안정을 악화할 가능성이 더 크다”는 반증 가능(falsifiable)한 주장으로 방향성을 명확히 합니다. PCAI는 AI 라이프사이클 전반(문제정의-모델설계-평가-배치)을 재구성해 행성적 책임을 인간 사용자 범위를 넘어 생태계와 지구 시스템까지 확장합니다.

- **Technical Challenges**: 핵심 기술적 난관은 앤트로포세인(인간이 지배적으로 지구 시스템을 바꾸는 시기) 문제들이 ‘wicked problems(악의적 문제)’에 가깝다는 점으로, 비정상성·피드백·깊은 불확실성·가치 논쟁 때문에 고정된 최적해와 신뢰 가능한 성공평가가 어렵다는 데 있습니다. PCAI는 이를 해결하기 위해 시스템 매핑과 변화이론(theory of change)을 문제정의 전제에 두고, 평가를 단일 점수 대신 궤적 기반(trajectory-oriented)·파레토(Pareto) 방식으로 바꾸며, 반동효과(rebound effect)·상관 실패처럼 증폭 메커니즘을 ‘시스템적 위험 프로브’로 점검하도록 요구합니다. 또한 예측/제어 중심에서 ‘선견(f oresight)·시나리오·시뮬레이션’ 중심의 인식 인프라로 AI의 역할을 전환하고, 배치 이후에는 모니터 가능성(monitorability)과 수정 가능성(revisability)을 포함한 안전장치를 설계합니다.

- **Empirical Impact**: 실증적으로는 기후·환경 영역에서 AI for Sustainability(지속가능성 목적의 AI)나 자율주행 같은 사례가 효율 개선의 일부 성과는 보일 수 있어도, 배치 확대로 인한 수요·인프라·거시적 궤적 변화를 평가에서 누락하면 순효과가 악화될 수 있음을 논의합니다(예: 반동효과 및 전 생애 배출). 논문은 지표가 과제 수준으로 국한될 때 ‘얽힘(entanglement)’과 ‘개입 깊이(intervention depth)’ 맹점이 생겨 시스템 위험을 놓치기 쉽다고 정리합니다. 최종적으로 PCAI는 모델 성능 경쟁만으로는 부족하며, 행성 규모의 안정성과 장기적 결과를 예측·평가·감시하는 프레임으로 분야의 실천 기준을 바꿔야 한다는 점을 강조합니다.



### Active Inference for Adaptive Traffic Signal Control in Noisy Nonstationary IoT Environments (https://arxiv.org/abs/2606.13698)
Comments:
          Submitted to IEEE 12th World Forum on Internet of Things (WF-IoT) 2026

- **Prior Approaches**: 기존 신호 제어는 센서가 가려지거나(occlusion), 날씨로 신호가 약해지고(weather attenuation), 수요가 시간에 따라 바뀌는 상황(nonstationary demand)에서 성능이 쉽게 떨어진다. 규칙 기반은 변화에 대한 적응이 약하고, 딥러닝 기반은 효과는 나와도 내부 의사결정의 추적·감사가 어렵다는 한계가 있다. 또한 잡음과 비정상성을 점점 키우는 복합 시나리오에서 일관된 강건성을 보이기 어렵다.

- **Core Contribution**: 이 논문은 4-진 신호 교차로에 대해 능동 추론(active inference) 기반 제어기를 제안하고, Gaussian 사전분포(가우시안 신념)로 각 방향의 혼잡을 추정한 뒤 기대 자유에너지(EFE)를 최소화하는 방식으로 위상(phase)을 동적으로 선택한다. 핵심은 단순 최적화가 아니라, 관측 불확실성을 반영해 의사결정 파이프라인을 끝까지 추적 가능(fully traceable)하게 만드는 데 있다. 이를 통해 ‘왜 이런 선택을 했는가’를 감사 가능한 형태로 제공하려는 목표가 분명하다.

- **Technical Challenges**: 문제는 잡음과 가림이 섞여 관측이 흔들릴 때도 ‘혼잡 추정’과 ‘위상 선택’을 안정적으로 수행해야 한다는 점이다. 논문은 방향별 혼잡을 가우시안 신념으로 모델링하고, 각 후보 위상에 대한 기대 자유에너지를 계산해 위상을 고르는 절차로 해결했으며, 그 과정이 의사결정 경로로 그대로 남도록 설계했다. 다만 대중교통 우선 서비스율과 위상 전환 빈도에서는 약간의 비용이 발생할 수 있음을 함께 보고한다.

- **Empirical Impact**: SUMO 교통 시뮬레이터에서 규칙 기반 휴리스틱과 DQN(딥 Q-네트워크)을 비교하고, 잡음과 비정상성을 단계적으로 키우는 네 가지 시나리오(센서 가림, 불리한 날씨, 확률적 사고 포함)를 구성해 성능을 검증한다. 시나리오별로 100회의 독립 무작위 평가에서 능동 추론은 가장 잡음이 큰 조건에서 대기(Idle) 시간과 CO2 배출을 가장 낮췄다(예: 56,977초·29.12kg 대 71,741초·30.56kg for DQN). 대신 버스 우선 서비스율과 위상 전환 빈도는 다소 희생되는 트레이드오프가 나타나며, 강건한(robust)·감사 가능한 신호 제어의 가능성을 실증적으로 제시한다.



### Korzhinskii-Net: Physics-Informed Neural Network for Sub-Surface Mineral Prospectivity Modelling (https://arxiv.org/abs/2606.13695)
Comments:
          12 pages, 7 figures, 3 tables

- **Prior Approaches**: 광물 가능성 모델링(MPM)은 표면·지구물리 특징을 입력해 매장 여부를 맞히는 지도학습 분류로 많이 전환됐습니다. 그러나 전통적 로지스틱 회귀·랜덤포레스트·그래디언트 부스팅·SVM·얕은 MLP류는 지하에 금속을 집중시키는 열대류, 유체 흐름, 암석-기반 반응을 직접 인코딩하지 못해 상관 패턴에 의존하는 한계가 큽니다. 또한 라벨이 희소하고 공간적으로 편향돼, 누수(leakage)를 통제하지 않으면 교차검증이 과대평가될 수 있습니다.

- **Core Contribution**: Korzhinskii-Net은 Darcy 유동, 열의 대류-확산 수송, 그리고 암종별 평형 용해도에 기반한 반응(softplus 포화 포함)을 하나의 미분가능한 2-D 방사형 전방모델로 결합한 physics-informed 신경망입니다. Dmitri S. Korzhinskii의 침투-대사작용(infiltration metasomatism) 이론을 물리적 스캐폴드로 삼아, 온도·압력·농도장을 동시에 예측하고 그로부터 광물화(매장) 지도를 생성합니다. 여기에 표면·원격탐사 프록시로 약지도(weak supervision)를 걸어, 희소한 알려진 매장 픽셀만으로도 지하 국소화 패턴을 복원하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난점은 서로 다른 성격의 물리 방정식이 결합된 상황에서 PINN이 안정적으로 학습하는 것입니다(예: 압력은 타원형, 수송은 대류 지배적이며, 반응항은 강한 비선형·강성 가능). 논문은 좌표 입력에 Fourier feature를 써 급격한 전이(반응 전선 등)를 표현하고, 자동미분으로 PDE 잔차를 계산해 흐름·열·반응을 공동 최적화합니다. 또한 포화된 반응률, 프록시가 modulate하는 경계 조건(열원 풋프린트, 결함·지진성 프록시), 그리고 hard ring-shaped negative를 포함한 손실/평가 설계로 학습 및 누수 문제를 함께 완화했습니다.

- **Empirical Impact**: 러시아 5개 광산 지역(노릴스크, 페흐엔가, 우도칸, 쇼호이 로그, 미르니)과 4개 원자재 범주에서 Korzhinskii-Net은 공정한 5-fold·누수 통제 교차검증 하에 평균 PR-AUC 0.885를 달성했으며, 가장 강한 고전 기준선(gradient boosting)의 0.281 대비 약 3.1배 향상입니다. 정밀도-재현율 곡선이 전 구간에서 더 위에 놓이고, per-site 부트스트랩과 paired Wilcoxon 검정에서도 모든 기준선 대비 유의미한 우위를 보였다고 보고합니다. 나아가 드릴링 우선순위 관점의 mean fractional rank이 0.019로, 고전 모델군(0.41~0.58대) 대비 현저히 개선돼 실무적 가치가 있음을 시사합니다.



### Efficient Temporal Modeling for Mobile Sleep Staging via Lightweight Random Attention (https://arxiv.org/abs/2606.13694)
Comments:
          7 pages, 1 figures, 5 tables

- **Prior Approaches**: 모바일 수면 단계 분류는 PSG의 한계를 넘어 웨어러블 EEG로 확장되고 있지만, LSTM/GRU/Transformer 같은 순차 모델은 연산·지연·메모리 부담이 커 배치/실시간 배포에 제약이 있다. 한편 epoch-wise 방식은 계산은 가볍지만 시간 연속성을 충분히 반영하지 못해 생리적 전이 패턴을 깨는 불안정 예측이 생긴다. 기존 연구는 긴 문맥을 늘려도 성능이 일관되게 크게 오르지 않는 경우가 많아, 복잡한 장거리 의존 학습보다 ‘국소 일관성(스무딩)’이 이득을 주는지에 대한 의문을 제기한다.

- **Core Contribution**: 이 논문은 Random Attention(RA)을 제안해, 학습 가능한 순차 모델 대신 고정된 무작위 투영을 기반으로 한 유사도 중심의 시간 집계를 수행한다. RA는 epoch encoder 이후에 거의 추가 학습 파라미터 없이도 temporal smoothing을 제공하며, 수면 단계 구조를 글로벌 스무딩 항과 특징 유사도 항으로 분해한 Random Attention Prior Kernel(RAPK) 관점의 해석도 함께 제시한다. 결과적으로 ‘복잡한 의존성 추론’이 아니라 ‘내용 기반 스무딩’이 모바일 수면 스테이징의 핵심 요구일 수 있음을 뒷받침한다.

- **Technical Challenges**: 핵심 과제는 학습 가능한 attention/순차 모듈 없이도 시간적 연속성과 단계 전이를 동시에 살리는 효율적인 집계가 가능한지였다. 저자들은 고정 random projection을 사용해 T×T attention 행렬을 만들지 않도록 설계하고, RA의 커널 해석(RAPK)으로 글로벌 평균화(잡음 억제·관성)와 특징 유사도 기반 적응 스무딩(경계 보존)을 결합하도록 동작함을 설명한다. 또한 투영 차원과 초기화 분포가 attention 로그이트 붕괴 여부에 영향을 준다는 점을 들어 Xavier 균일 초기화가 안정적임을 실증적으로 정리한다.

- **Empirical Impact**: Sleep-EDF-20과 Sleep-EDF-78에서 RA는 epoch-wise 기준선 대비 정확도와 weighted F1을 각각 약 1~3%p 끌어올리며, LSTM/GRU 및 학습형 Transformer와 견줄 만한 성능-효율 균형을 보였다. 특히 전이 단계인 N1과 REM에서 개선 폭이 가장 커, 유사도 기반 스무딩이 임상적으로 의미 있는 구간을 잘 정리함을 보여준다. 더불어 여러 백본(DeepSleepNet, TinySleepNet, ULW-SleepNet 등)에 플러그 앤 플레이로 적용해도 일관된 향상이 나타나며, 기존 고정형 스무딩 방법 대비 윈도 크기 선택에 덜 민감해 실시간 웨어러블 적용 가능성이 강화된다.



### An Agentic Retrieval Framework for Autonomous Context-Aware Data Quality Assessmen (https://arxiv.org/abs/2606.13692)
Comments:
          26 pages, 18 figures, Submitted to the International Journal of Intelligent Information and Database Systems

- **Prior Approaches**: 기존 데이터 품질 평가는 정적 규칙/품질 차원(완전성·정확성·일관성 등)에 의존하거나, 컨텍스트를 반영하더라도 실행 가능한 검증 로직(SQL/코드)으로 자동 번역하는 단계가 수동·오류가 잦다는 한계가 있었다. 또한 LLM 기반 규칙 생성은 유연하지만 프롬프트 민감도, 재현성 저하, 비현실적이거나 실행 불가한 규칙 생성 위험이 커서 감사·추적성 요구에 취약하다. RAG는 환각을 줄일 수 있으나, 다단계 에이전트 오케스트레이션과 “실행 가능성” 보장은 별도로 다뤄지지 않는 경우가 많았다.

- **Core Contribution**: 이 논문은 “의도한 데이터 사용(usage intent)”을 자연어로 입력받아, 컨텍스트에 맞는 데이터 품질 평가 전략을 만들고 실행 가능한 검증 로직까지 생성·실행하는 에이전트-리트리벌 통합 프레임워크를 제안한다. 특히 생성된 규칙을 곧바로 실행하지 않고, 현실성·실행가능성을 점검하는 feasibility validation 단계를 워크플로에 포함해 통제된 자율성을 제공한다. 채택한 규칙은 결정론적으로 실행해 재현성과 감사를 보장한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 사용 의도를 컨텍스트로 해석해도 규칙이 실제 스키마/타입/실행 환경과 맞지 않으면 실패한다는 점, (2) LLM이 그럴듯한 가정을 만들어 “근거 없는 서술” 또는 “실행 불가 규칙”으로 이어질 수 있다는 점이다. 논문은 다중 에이전트(사용 해석·평가 계획·규칙 생성·설명)와 리트리벌로 검증 의미를 프롬프트에 주입해 생성 공간을 제한하고, Feasibility Validator가 스키마 정합성·운영 실행성·논리 일관성·안전 범위를 통과한 명세만 결정론 실행으로 넘기는 방식으로 이를 해결한다.

- **Empirical Impact**: 저자들은 단일 데이터셋에 대해 여러 사용 시나리오를 적용해 평가 결과가 의도된 사용에 의미 있게 적응함을 보였다. 동시에 feasibility-gated 실행이 비현실적이거나 실행 불가능한 규칙 생성을 줄여 실행 실패를 완화하는 효과가 관찰됐다. 이는 “자율적이되 통제 가능한” 컨텍스트 기반 데이터 품질 자동화의 실용적 기반을 제시한다는 점에서 의미가 있다.



### The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation (https://arxiv.org/abs/2606.13685)
Comments:
          24 pages, 7 figures

- **Prior Approaches**: LLM-as-a-Judge는 모델 출력 순위화, 보상모델 학습, 리더보드 채점에서 널리 쓰이지만, “같은 조건에서 재실행하면 같은 결론이 나오는가”는 상대적으로 덜 측정돼 왔습니다. 기존 연구들은 주로 위치 편향(먼저 제시된 응답 선호), 과장/자기강화, 길이·형식 같은 편향을 단일 샘플 관점에서 다뤘고, 신뢰도(변동성) 자체를 반복 샘플로 정량화한 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 LLM 심판의 신뢰도를 단일 수치가 아니라 네 층(쌍대 판정의 내부 변동, 점수의 변동, 동일 심판 내 반복 일관성, 서로 다른 심판 모델 간 합의)으로 분해해 측정 체계를 제시합니다. 또한 쌍대 선택이 점수 차이를 뒷받침하지 못하는 ‘쌍대-점수 역설(pairwise–pointwise gap)’을 실증적으로 보여, 리더보드식 단일 재판정이 흔들릴 수 있음을 강조합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “반복 재평가 시 판정이 얼마나 바뀌는지”를 항목 단위로 고정밀 추정하는 것입니다. 저자들은 29개 태스크(10개 범주)에서 두 OpenAI judge 모델(GPT-4o-mini, GPT-4.1-mini)을 대상으로 질문당 쌍대 50회/점수 50회를 반복하고, 온도(temperature)·프롬프트 민감도(서술 템플릿)까지 통제해 변동의 원인을 분리했습니다.

- **Empirical Impact**: 실험 결과 쌍대 선호는 평균 13.6%에서 뒤집히며, 20% 이상 뒤집히는 질문이 28%나 됩니다; 점수 차이는 10점 척도에서 0.19~0.36 수준으로 집계 유의성도 약해, 승자 판정이 ‘품질 격차’ 증거로 과대해석될 수 있음을 시사합니다. 또한 심판 간 합의는 76%(κ=0.51)로 중간 수준에 그치고, 동일 심판도 프롬프트를 바꾸면 다수결 결과가 25%에서 달라져 고위험 평가에선 다중 재판정·위치 무작위화·불확실성 보고가 표준이 되어야 한다는 결론을 냅니다.



### Cross-Dataset Bloom Question Classification: Supervised Models and Prompted LLMs (https://arxiv.org/abs/2606.13684)
Comments:
          Accepted at AIED 2026. Abdolali Faraji and Mohammadreza Molavi contributed equally to this work

- **Prior Approaches**: 기존 연구는 SVM·TF–IDF(POS 가중) 같은 전통적 ML과 BERT 계열 미세조정(DL)로 Bloom 인지수준을 분류해 높은 성능을 보고했다. 다만 대부분이 동일 데이터의 무작위 train–test 분할에 의존해, 실제 교육 맥락·교사별 주관성이 다른 경우의 일반화 성능은 불명확했다. LLM은 교육 분류에 활용되었지만, Bloom 인지수준 ‘질문 분류’에 대한 체계적 다중 데이터 평가가 부족했다.

- **Core Contribution**: 이 논문은 Bloom 인지수준 질문 분류에서 “교차 데이터 일반화”를 핵심 문제로 두고, 기존 ML/DL의 데이터 전이 실패를 정량적으로 확인했다. 동시에 여러 프롬프트 전략을 비교해 LLM이 데이터 변화에 더 안정적으로 동작함을 보여주며, 특히 “수준별 대표 예시 + 해당 수준의 핵심 동사(action verbs)” 조합이 가장 잘 맞는 설정임을 제시한다. 나아가 이 최적 프롬프트 흐름을 경량 UI로 구현해 대규모 문항은행 분류를 실사용 관점에서 지원한다.

- **Technical Challenges**: Bloom 라벨링은 맥락·교사 관점에 따라 달라져 학습 데이터와 다른 교육 환경으로 넘어갈 때 모델 성능이 크게 흔들린다는 점이 기술적 난제다. 논문은 ML/DL이 훈련 데이터에 과도하게 적응하는 문제를 교차 데이터 F1 하락(평균적으로 큰 폭 감소)으로 드러내고, LLM은 미세조정 없이도 in-context 예시와 JSON 출력 설계로 태스크 수행의 재현성을 높여 견고성을 확보했다. 또한 UI에서는 교사가 소수 예시만 제공하면 자동으로 동사를 추출해 프롬프트를 구성하도록 하여 운영 부담을 줄였다.

- **Empirical Impact**: 5개 데이터셋(총 4,179문항) 교차 평가에서 감독학습 ML/DL은 미지 데이터로 갈수록 weighted F1이 평균 약 0.25~0.28 수준으로 크게 떨어졌다. 반면 LLM은 교차 환경에서도 상대적으로 성능 저하가 작았고, 최적 프롬프트에서 weighted F1이 최대 0.84까지 도달해 실무 적용 가능성을 강화했다. 실제 교사용 UI는 사용성 실험(N=50)에서 NASA-TLX 기준 부담이 낮고 SUS 평균 78.2로 ‘매우 양호’ 수준의 사용성을 보였으며, 교육 문항 분류 자동화의 현실적 경로를 제시했다.



### Simplex-Constrained Sparse Bagging: Transitioning from Uniform Priors to Sparse Posteriors in Ensemble Learning (https://arxiv.org/abs/2606.13589)
Comments:
          6 pages, 3 tables

- **Prior Approaches**: 기존 배깅은 부트스트랩으로 여러 추정기를 학습한 뒤 평균(또는 투표)하여 분산을 줄이지만, 가중치를 모두 동일하게 두는 경우가 많았습니다. 이때 추정기 간 상관으로 과도한 확신(캘리브레이션 부정확)과 불필요한 중복 추정기가 함께 남아 추론 비용을 키운다는 한계가 지적됩니다. 또한 앙상블 프루닝은 휴리스틱 탐색이나 성능 순위 기반 선택이 주를 이루며, 가중치까지 공동 최적화할 때 수학적 보장이 부족했습니다.

- **Core Contribution**: SCSB(Simplex-Constrained Sparse Bagging)는 사후(post-training) 압축과 확률 캘리브레이션을 하나의 확률 단체(확률 심플렉스) 가중치 최적화로 동시에 수행합니다. OOB(Out-Of-Bag) 손실을 최소화해 데이터 누수를 피하면서, 중요하지 않은 추정기 가중치는 정확히 0으로 만들고 남은 추정기에만 가중치를 부여합니다. 그 결과 앙상블을 ‘uniform prior’에서 ‘sparse posterior’로 전환해 캘리브레이션과 일반화 성능을 함께 개선합니다.

- **Technical Challenges**: 핵심 난점은 심플렉스 제약 하에서 L1(라쏘) 페널티가 가중치의 L1 노름을 일정하게 만들어(‘L1-simplex paradox’) 희소성을 유도하지 못한다는 이론적 문제입니다. SCSB는 이를 해결하기 위해 -||w||2^2 형태의 오목(concave) 이차 페널티를 도입해 최적해가 심플렉스의 꼭짓점(정확한 0 가중치)으로 수렴하도록 유도합니다. 또한 분류(로그-로스)와 회귀(MSE)에 대해 OOB 기반 목적함수의 해석적 기울기를 유도하고, SLSQP로 효율적으로 최적화합니다.

- **Empirical Impact**: 실험에서 SCSB는 최대 96% 앙상블 압축을 달성하면서도 정확도나 R2 같은 일반화 성능을 유지하거나 소폭 향상시켰습니다. 프루닝으로 ‘활성 추정기 수’가 줄어들어 추론 지연이 선형적으로 감소하며, 예시로 spambase에서 추론 시간이 약 5.7배 빨라졌습니다. 더불어 Log-Loss를 OOB에서 직접 최소화한 설계가 ECE(Expected Calibration Error) 개선으로 이어져, 확률 캘리브레이션이 기존 균등 가중 앙상블보다 우수하다는 점이 확인됩니다.



### Under What Conditions Can a Machine Be Called Genuinely Creative? (https://arxiv.org/abs/2606.13196)
- **Prior Approaches**: 기존 연구는 생성 결과물을 중심으로 기계 창의성을 평가하는 경향이 강했습니다. 예컨대 참신성·유용성·가치·놀라움 같은 기준이 “창의적 출력”을 만들었는지에 초점이 있습니다. 다만 이런 접근은 기계가 의미가 걸린 환경을 어떻게 다루는지, 개입 후 결과를 관찰하고 프레임을 바꾸는지까지는 충분히 설명하지 못합니다.

- **Core Contribution**: 이 논문은 기계 창의성을 “출력의 새로움”이 아니라 “불완전한 상황을 재귀적으로 개입해 구조를 변환하는 능력”으로 재정의합니다. 이를 위해 Designics(의미를 담는 의도적 변화의 과학)에서 도출한 요건 프레임워크를 제시하고, 창의성은 환경-충돌-역량의 법칙 하에 상호 의존적인 10가지 요구를 충족해야 한다고 주장합니다. 또한 인간 주도성은 인간-인공지능 공동 거주(human–AI co-living) 같은 내부 조건을 통해 보존될 수 있다고 봅니다.

- **Technical Challenges**: 핵심 난제는 “생성”과 “의미 있는 개입”을 분리해, 기계가 환경을 표현하고(scope) 충돌을 인식한 뒤(intervene) 결과를 관찰·학습하고(rescore/update) 다음 순환의 지각 경계를 바꾸는지(rescoping)를 기능적으로 구현하는 것입니다. 논문은 이 요구를 단일 모델 구조로 고정하지 않고, 환경 표현·한정된 지각·충돌 식별·개입 역량·결과 관찰·업데이트·재구성·국소-전역 전개·가치 기반 스코핑·인간-공동거주로 정리해 계산 가능하게 구성되도록 설명합니다.

- **Empirical Impact**: 컴퓨팅 관점에서의 실행 가능성은 사이버-물리 및 사이버-생물 연구 사례(재귀적 요소 추출, 자율 메쉬 생성, 신경생리·작업부하 분석 등)로 보여줍니다. 또한 개방형 시스템, 자동 과학 발견, 자기수정 에이전트, 파운데이션 모델, 에이전트 워크플로는 강력하지만 그 자체만으로는 “진짜 기계 창의성”을 증명하지 못하는 압력 사례로 위치시킵니다. 더 나아가 윤리를 사후 필터가 아니라 창의 과정 내부의 경계 조건(가치 기반 스코핑과 공동거주)으로 포함시켜, 향후 평가·설계 지침에 영향을 줄 수 있음을 강조합니다.



### GUITrans2Act: Understanding User Operational Behaviors from Mobile GUI Interactions with Vision-Language Models (https://arxiv.org/abs/2606.12817)
Comments:
          20 pages, 9 figures. Yudong Zhang and Lei Hu contributed equally to this work. Zuojian Wang, and Zhilin Gao are corresponding authors

- **Prior Approaches**: 기존 연구는 정적 화면을 이해하거나(정적 UI 인식·그라운딩) 다음 실행 동작을 예측하는 GUI 에이전트 형태로 접근해 왔습니다. 그러나 전자는 ‘두 상태를 잇는 실제 조작(작업 의미)’을 복원하기 어렵고, 후자는 런타임 상태에 종속되어 출력이 해석 가능(인터프리터블)하지 않은 한계가 있습니다. 또한 일반 VLM은 멀티프레임은 잘 봐도, 모바일 화면 전이에서 ‘조작 의미’를 안정적으로 추론하는 데 취약하다는 문제가 지적됩니다.

- **Core Contribution**: 이 논문은 모바일 화면 전이(이전-이후 스크린 상태 변화)를 자연어로 된 절차형 ‘운영 지식(operational knowledge)’으로 직접 변환하는 Teach VLM을 제안합니다. 운영 지식은 행동 유형, 대상 UI 요소, 텍스트 인자, 실행 순서를 포함하는 짧은 단계별 문장으로 정의되며, 앱/과업에 덜 의존하는 형태를 목표로 합니다. 더불어 Teach-and-Repeat 패러다임으로, 한 번 생성한 운영 지식을 반복 실행 에이전트의 해석 가능한 절차 레퍼런스로 재사용하도록 설계합니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 시연 영상에 로딩 애니메이션/전이 잡음이 매우 많아 키프레임-조작 대응이 불안정하다는 점과 (2) 앱 인터페이스 설계가 지나치게 이질적이라 일반 VLM이 연산 의미를 잘 일반화하지 못한다는 점입니다. 이를 위해 Teach VLM은 먼저 규칙 기반 차분으로 조작 관련 ‘후보 키프레임 전이’를 뽑고, 가벼운 LLM으로 사용자 조작 가능성만 걸러 Teach VLM 학습용 전이를 정제합니다. 또한 정렬 데이터 부족 문제는 모델 예측→평가→사람 피드백→재학습을 반복하는 데이터 플라이휠로 완화하고, 다단계 전이를 슬라이딩 윈도 기반 추론으로 단계 수 불일치와 환각을 줄입니다.

- **Empirical Impact**: 실험에서 Teach VLM은 operation semantics prediction에서 강한 VLM 기준선을 일관되게 크게 능가하며, Android-In-The-Zoo와 GUIOdyssey에서 단일 단계의 joint 정합성(OSA) 지표가 특히 두드러집니다. 또한 중국어 프레임 단위 주석을 갖춘 Chinese Mobile Screen Teach Benchmark를 통해 미세 평가가 가능해졌다는 점에서 의미가 있습니다. Teach-and-Repeat로 운영 지식을 외부 절차 레퍼런스로 주입했을 때 Android World에서 하위 실행 에이전트의 Task Success Rate가 일관되게 개선되어, ‘시연에서 자동화로의 재사용’ 가능성을 실증합니다.



### MP3: Multi-Period Pattern Pre-training for Spatio-Temporal Forecasting (https://arxiv.org/abs/2606.13119)
- **Prior Approaches**: 기존 spatio-temporal graph neural network(STGNN)은 복잡한 공간·시간 의존성을 학습하도록 설계돼 왔지만, 성능 향상은 점차 포화되는 경향이 있었다. 특히 짧은 관측 창에 기반한 입력은 미래를 가늠하는 데 필요한 다중 주기 패턴을 충분히 식별하지 못해 ‘시간적 미라지(temporal mirage)’ 현상을 놓치기 쉽다.

- **Core Contribution**: 본 논문은 도시 교통 같은 spatio-temporal 데이터에서 나타나는 temporal mirage의 근본 원인을 ‘짧은 창이 다중 주기 관측을 불완전하게 만들기 때문’으로 보고, 이를 겨냥한 플러그앤플레이 사전학습 모듈 MP3를 제안한다. MP3는 장기 시계열에서 다중 주기 패턴을 명시적으로 학습해, 미라지를 구분하는 예측 표현을 downstream STGNN에 주입한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 짧은 창이 제공하지 못하는 절대적 주기 위치 정보를 어떻게 장기 시계열에서 복원할지, (2) 노드 수가 큰 그래프에서 이질적인 전역 공간 상관을 계산량 폭발 없이 어떻게 포착할지, (3) 서로 다른 주기 패턴 간의 ‘교차-주기 중첩 인과’를 어떻게 모델에 강제할지였다. MP3는 FFT 기반 주기 선택과 2D 재구성으로 다중 주기 텐서를 만들고, edge convolution으로 주기 내·간 변화를 분리하며, 병목(bottleneck) 투영·전역 memory bank·희소 그래프 집계로 전역 공간 상관을 효율화하고, DAG 기반 인과 마스크를 갖춘 Transformer로 강→약 주기 방향 상호작용을 학습한다.

- **Empirical Impact**: MP3는 5개 STGNN 백본과 5개 실제 데이터셋(대규모 CA 포함)에서 일관되게 예측 성능을 개선하며, 평균적으로 MAE 4.7%, RMSE 5.0% 감소를 보고한다. 플러그앤플레이 형태로 백본 파라미터를 동결한 채 gating으로 결합해도 성능이 견고하게 유지돼, 확장성과 범용 적용 가능성이 실증적으로 확인됐다.



### Order Is Not Control: Driven-Dissipative Response Laws Across Artificial and Biological Systems (https://arxiv.org/abs/2606.12923)
Comments:
          52 pages, 7 figures, updated title

- **Prior Approaches**: 기존 연구들은 정렬(alignment), 해석가능성, 스티어링, 신경 섭동(neural perturbation)에서 “질서(order)”를 관찰하면 곧바로 제어(control)로 해석하는 경향이 있었다. 하지만 질서의 유도는 수신기(receiver)가 승인한 행동/결과 읽기(readout)의 ‘이동’과 동일하지 않을 수 있어, 관측(증거)과 제어(실제 제약 하 이동)를 분리하기 어렵다. 또한 프롬프트·원칙·어댑터·디코드 설정·생물학적 좌표 변화가 같은 방향의 구조를 만들더라도, 동일 조건(denominator)에서의 국소적 이동이 검증되지 않으면 제어라고 단정하기 힘들다.

- **Core Contribution**: 이 논문은 정렬·해석·스티어링 실험을 공통 언어로 연결하기 위해, ‘수신기 승인 반응 이동’을 만들어내는 경험적 대상인 반응 법칙(response law)을 제시한다. 제어는 단순한 구조화가 아니라, 수신기 상태에 의해 게이트(gated)된 반응 법칙 아래에서 유한한 노력으로 목표 반응(또는 결과 읽기) 계열을 같은 분모(denominator)에서 이동시키되, 손상(damage), 무응답/회피(null/evasive), 형식 무효(invalid format), 과과속(overdrive), 기준선 불필요 교란이 유계(bound)로 유지될 때만 성립한다. 따라서 “order=control”을 부정하고, 제어를 분모-조건(stochastic response kernel)과 그 파생 반응 지도/행동 효과로 정의한다.

- **Technical Challenges**: 핵심 난제는 ‘무엇이 실제 제어 증거인가’를 실험적으로 분리하는 것이다. 이를 위해 논문은 준비 매체(prepared medium), 욕조/경계 조건(bath/denominator), 수신기 상태 및 비교기(comparator), 반응 분지(basin), 싱크 채널, 선언된 노력까지를 분모 튜플로 고정한 뒤에만 반응 커널 𝒫δ(dy|x,a)와 그 유한 차분(반응 지도, 행동 효과)을 측정한다. 또한 조준 가능한 기준선 대비 이동과 함께 싱크/노력 채널을 동시에 보고, 강한 개입이 단조 개선이 아니라 손상으로 라우팅될 수 있음을 ‘분모 동일성’ 하에서 차단적으로 판정한다.

- **Empirical Impact**: 생물학(마우스 ALM, C. elegans, 제브라피시)에서는 좌표 동일성이나 기하가 곧 제어를 보장한다는 주장 없이도, 섭동-반응 사슬이 같은 형태의 분모-조건 반응 연산자(응답 연산자)를 구성함을 물리 기질에서 보여준다. LLM 생성 출력 패널에서는 분모가 고정된 조건에서 반응 벡터 성분 부호 정확도(예: 72.8~73.7%)가 상승하고, 은닉 관측자/읽기 예측의 정확도도 보고되며(예: 93.6%, 91.7%), 어댑터는 준비 매체의 취약성(susceptibility)을 바꿔 동일한 가시적 경계가 다른 반응 분지로 ‘받아들여지게’ 만든다. 결론적으로 이 논문은 국소 허용(admitted) 제어 영역과 측정 가능한 확률적 반응 연산자를 실증적으로 지지하면서도, 좌표 동일성·은닉/로짓 인과적 충분성·배포용(컨트롤러급) 선행 제너레이션 제어까지는 아직 남겨둔다는 점을 명확히 한다.



New uploads on arXiv(cs.RO)

### EgoGuide: Egocentric Guidance for Efficient Robot-Free Demonstration Collection and Learning (https://arxiv.org/abs/2606.14665)
- **Prior Approaches**: 로봇 학습에서 가장 큰 병목은 현실 시연 데이터의 스케일 문제이며, 이를 줄이기 위해 UMI(Universal Manipulation Interface) 같은 로봇 프리 수집이 대안으로 주목받아 왔다. 하지만 기존 UMI 파이프라인은 (1) 이미 데이터가 충분한 상태인지 수집자가 알기 어려워 중복 시연이 늘고, (2) 단일 손목 카메라 관측이 가림/사라짐/장기 진행을 충분히 설명하지 못해 전역 장면 정보가 부족하다는 한계가 있다.

- **Core Contribution**: 이 논문은 시연 수집 단계에서 “현재 상태가 데이터 다양성/커버리지를 얼마나 늘릴지”를 실시간으로 피드백하는 EgoGuide를 제안한다. 동시에 고정된 에고센트릭(머리/시점) 관측을 견고하게 활용하기 위해 GERP(Gated Egocentric Residual Policy)라는 잔차(residual) 기반 정책을 도입해, 손목 기준 제어를 안정 베이스로 두고 에고센트릭 맥락이 모호한 구간에서만 보정하도록 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (a) UMI식 로봇 프리 수집에서 시연자가 ‘언더익스플로어드 상태’를 찾도록 안내하려면 멀티모달(손목+머리) 커버리지 추정이 필요하고, (b) 움직이는 인간 시점에 의존하는 방식은 관측 불일치와 잡음에 취약하다는 점이다. EgoGuide는 손목/머리 영상의 시각 특징과 손목 포즈의 기하 커버리지를 노벨티 분위수로 정규화해 AR 인터페이스로 0~100 점수를 제공하고, 시연을 작업 중간에서 시작하는 선택지도 제공하며, 센서 오류·물리 불일치 에피소드를 정적 품질 체크로 제거한다. GERP는 손목 뷰 정책을 고정한 뒤, 머리 기준으로 손목 포즈를 변환해 에고센트릭 후보 행동을 만들고 게이트로 블렌딩하여 손목 제어의 안정성을 유지하면서 가림 상황에서만 성능을 끌어올리도록 학습/추론을 설계한다.

- **Empirical Impact**: 실세계 조작 실험에서 EgoGuide는 동일 성공률을 목표로 할 때 필요한 시연 에피소드 수를 줄이고 데이터 효율을 개선했으며, 예를 들어 Pepper Sorting에서는 200개 시연에서 성공률을 10%→50%로 끌어올리고 절반(50%)의 시연만으로도 유사한 성공을 달성했다. 또한 GERP는 손목 단독 대비 Pepper Sorting에서 성공률과 작업 진행 점수를 5~10%p 수준으로 개선했고, 특히 손목 관측이 가리거나 불완전할 때 게이트 값이 증가하며 견고성이 향상됨을 보였다.



### Whole-Body Impedance Model Predictive Control for Safe Physical Human--Robot Interaction on Floating-Base Platforms (https://arxiv.org/abs/2606.14617)
- **Prior Approaches**: 기존 부유(플로팅) 베이스 로봇 제어는 크게 두 갈래로 발전해 왔다. 하나는 SRBD 기반 centroidal MPC와 계층형 WBC로 GRF를 분배하는 방식인데, 팔 상호작용은 대부분 ‘억제해야 할 교란’으로만 취급되어 지속 접촉에서 정착오차(steady-state error) 보장이 어렵다. 다른 하나는 고정 베이스 조작기용 impedance MPC/전략인데, 부유 베이스의 비구동(언액추에이티드) 상태와 접촉 전이, 그리고 접촉 제약을 반영한 작업공간 관성의 처리가 부족해 그대로 적용하면 토크 오차와 불안정 가능성이 생긴다.

- **Core Contribution**: 이 논문은 고정 베이스 two-layer Impedance MPC의 핵심을 부유 베이스, 접촉 전이 환경으로 구조적으로 확장한다. 즉, 500ms 범위 centroidal MPC로 접촉력을 계획하고, priority 기반 WBC(접촉 일치 null-space 투영)로 균형을 유지한 뒤, 남는 팔 작업 공간을 receding-horizon Impedance MPC가 ‘예측·교란 거부’ 채널로 채운다. 특히 pHRI(사람-로봇 상호작용)에서 요구되는 “지속 상수 하중에 대한 영 정착오차”와 안전한 컴플라이언스 제어를 함께 달성하는 설계를 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 접촉 제약이 바뀌면 작업공간 동역학과 입력 행렬이 불연속적으로 변하고, (2) 팔 동작이 CoM과 접촉력 분포에 연쇄적으로 영향을 주어 계층 간 간섭이 생길 수 있다는 점이다. 저자들은 접촉 모드별로 피드포워드를 정리해 잔여 팔 동역학을 접촉-일치 피드백 선형화 후 선형 double integrator 형태로 만들고, 모드 인덱싱된 입력 행렬 라이브러리로 QP를 오프라인 비용까지 미리 계산해 ≥1kHz 동작을 가능하게 했다. 또한 Kalman-augmented 상태(적분형 교란 상태)를 도입해 pHRI 힘, SRBD 근사 오차, 다리 모멘텀 변화를 함께 추정·전파하며, 접촉 전이 시 공분산 인플레이션으로 추정치를 유지해 모드 전환 후에도 정착오차를 0으로 보장한다.

- **Empirical Impact**: 시뮬레이션에서 17-DOF 이족 보행과 Unitree G1 휴머노이드에 적용해, 지속적인 pHRI 하중에서 팔 추적의 영 정착오차와 전이 구간의 안정성을 함께 검증한다. 제안한 Impedance Equivalence Theorem에 따라 무한시간 극한에서 고전적인 task-space impedance 법을 회복하며, 유효 질량·감쇠·강성이 자세와 접촉 구성에 따라 자동으로 적응하는 점도 이론과 함께 정리된다. 결과적으로 고속 실시간(≥1kHz) 예측 기반 compliance 조작을 부유 베이스 균형 제어와 한 스택에서 통합할 수 있다는 실증적 의미가 크다.



### Safe Reinforcement Learning of Autonomous Highway Driving: A Unified Framework for Safety and Efficiency (https://arxiv.org/abs/2606.14609)
Comments:
          20 pages, 5 figures, 7 tables. Preprint version

- **Prior Approaches**: 기존의 안전 강화 학습(SRL) 연구는 (1) 제약을 페널티/기대값으로 완화하는 소프트 제약과 (2) 안전 조건을 강제해 위반을 원천 차단하려는 하드 제약으로 크게 나뉜다. 소프트 제약은 학습 초기에 위반이 발생할 수 있고, LLM 기반 신뢰영역 분할은 물리 상태와의 접합이 약해 희귀 사건에서 안전 보장이 약해질 수 있다. 하드 제약은 셸딩/스위칭/공식 검증 등으로 안전을 담보하지만, 이질적인 모듈(MPC·규칙·학습정책) 전환이 만드는 불안정성과 스케일 문제(다차선·램프 등 복합 시나리오)가 남는다.

- **Core Contribution**: 이 논문은 안전 거리(SD), 리워드 머신(reward machine, RM), 혼합 전문가(mixture-of-experts, MoE)를 하나로 통합한 MoE-RM-SRL 프레임워크를 제안한다. 배치에서는 SD와 RM이 함께 ‘규칙 인지형(reward-aware) 보상’으로 고속도로 교통 규정과 단계별 목표를 인코딩해, 안전성과 효율을 동시에 노린다. 학습에서는 SD 기반 게이팅이 최소 전문가만 활성화해 차선유지/차선변경 같은 과제를 안정적으로 다루도록 설계한다.

- **Technical Challenges**: 핵심 기술 난제는 (i) 학습 중 시행착오를 줄이면서도 (ii) 규칙과 단계 목표를 시계열적으로 반영하고 (iii) 서로 다른 제어기 간 전환에서 생기는 불연속·충격성 전이 문제를 완화하는 것이다. 저자들은 SD를 기반으로 안전 조건을 엄밀한 신호로 만들고, 이를 RM의 유한상태 기계에 연결해 ‘시간적으로 확장된 목표’를 보상으로 변환한다. 또한 희소 게이팅 MoE를 학습에 도입해 최대 11개 deep Q-network(DQN) 전문가 중 필요한 소수만 켜, 이질적 컨트롤러 스위칭에서 흔한 불안정성을 감소시키는 방향으로 해결한다.

- **Empirical Impact**: CARLA와 6-DoF 드라이버 인 더 루프 가상현실(DiL-VR) 환경에서 확률적 2차선 교통 실험을 수행했으며, MoE-RM-SRL이 기존 최첨단 기준선 대비 안전성과 효율을 모두 크게 개선했다고 보고한다. 또한 이 프레임워크를 다차선 주행, 온-램프 진입 및 오프-램프 탈출 시나리오로 자연스럽게 확장 가능함을 보이며, 모듈별 기여를 분해하는 절제 실험도 함께 제시한다. 결과적으로 SRL의 실용 과제였던 ‘훈련-배치 전 과정의 안전성’과 ‘복합 주행 효율’을 동시에 끌어올렸다는 점에서 의의가 크다.



### Impedance MPC with Disturbance Estimation for Dexterous Hand Contro (https://arxiv.org/abs/2606.14606)
- **Prior Approaches**: 기존 임피던스 제어는 접촉 안전을 위해 관절 강성을 조절하지만, 정확도(정밀 추종)와 교란/상태오차 제거를 동시에 최적화하기 어렵다. PI 임피던스나 기존 임피던스 MPC는 접촉 이벤트 간 적분 고착(windup)이나 업데이트 주기(10–30Hz) 제약 때문에 목표 정밀도(15mrad 이내)를 달성하기 어렵다. 한편 기존 MPC들은 추정된 외란을 현재 스텝에서만 보정하는 방식이 많아 접촉 발생·이탈 과도응답이 커질 수 있다.

- **Core Contribution**: 이 논문은 다양한 텐던 구동기(유압, 케이블, 공압, 트위스트 문자열, 시리즈 탄성)에서 공통으로 쓰일 수 있는 actuator-agnostic Impedance Model Predictive Control(임피던스 MPC) 프레임워크를 제안한다. 핵심은 물리 인간-로봇 상호작용(pHRI)에서 확립된 constant-AdA_d offset-free 구조를 그대로 유지하되, 텐던 전송의 반영 관성·감쇠를 대수적으로 상쇄해 잔차를 double integrator로 만드는 것이다. 이렇게 하면 500Hz 수준의 빠른 QP 실행과 함께 ISO/TS 15066 기반 접촉력 하드 제약을 통합하면서도 정상상태 오차를 0으로 만드는 구조를 만든다.

- **Technical Challenges**: 가장 큰 기술 과제는 텐던 전송이 매체·기구에 따라 반영 관성/감쇠와 비선형성을 달리하며 고대역 제어를 어렵게 만든다는 점이다. 이를 위해 저자들은 구동기별 effective inertia와 damping만 남기도록(feedforward로 전달계를 상쇄) platform reduction 및 substitution map을 설계하고, 상시 접촉 하중에 대해 인코더만으로 외란을 추정하는 encoder-only augmented-Kalman(적분형 외란 상태)을 결합한다. 또한 유압 예제에서는 캐비테이션·압력 한계 같은 안전 제약을 QP 내부에 제약으로 넣되, 충격 상황에서 QP가 비가용해지는 문제는 소프트 제약 완화로 재귀적 가용성을 확보한다.

- **Empirical Impact**: 유압 구동 손가락 실험 예시에서 500Hz Kalman MPC는 1.5Nm 접촉 하중 조건에서 손가락 편향을 RMS 0.5mrad, 정상상태 0.1mrad, 피크 6.6mrad로 제한하며 classical impedance 대비 183×, 1500×, 23× 개선을 보고한다. 또한 첫 이동(first-move) 강성의 구현값(예: 18→323Nm/rad, 업데이트 속도 변화)을 독립적으로 검증해 목표 임피던스가 실제로 형성됨을 보여준다. 16-DOF LEAP Hand MuJoCo 시뮬레이션으로 확장 시에도 2.5N 그립 로드 교란을 0.7초 내 회복하는 등, 고정밀 접촉 제어가 다지관절 수준으로 확장 가능하다는 점에서 분야에 의미가 크다.



### What Robots Do Matters More Than What They Look Like: Task Context Shapes Trust in Educational HRI (https://arxiv.org/abs/2606.14602)
Comments:
          Accepted in the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026), Kitakyushu, Fukuoka, Japan

- **Prior Approaches**: 기존의 사회적 보조 로봇(SAR) 연구는 로봇 외형(외형적 인간다움, 형태 등)이 사용자 신뢰를 좌우할 수 있다고 가정해 왔다. 또 교육·정보 공유 맥락에서 로봇이 대화 능력만 확보하면 상호작용이 원활해질 것으로 보는 경향이 있었다. 하지만 로봇 외형이 여러 과업 전반에 걸쳐 공통으로 적절한지, 신뢰가 주로 어떤 맥락 요인에서 달라지는지는 명확하지 않았다.

- **Core Contribution**: 이 논문은 로봇 외형과 과업 유형이 동시에 신뢰도에 미치는 영향을 함께 검증한다. 세 가지 교육 관련 과업(수업/가르치기, 절차 안내, 개인 정보 요청)에 대해, 서로 다른 외형을 가진 로봇 세 명을 동일 참가자 내(within-subjects) 실험으로 비교한다. 결론적으로 신뢰는 외형보다 과업 맥락에 더 강하게 좌우된다는 실증 근거를 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 외형 효과와 과업 효과를 분리해 공정하게 비교하는 설계였다. 이를 위해 N=81 참가자를 대상으로 반복측정 분석이 가능하도록 영상 기반 과제로 통제하고, 과업별 신뢰 평가를 수집해 주효과와 상호작용을 통계적으로 확인했다. 그 결과 로봇 외형의 주효과는 유의하지 않았고, 외형-과업 상호작용도 경계적 수준(marginal)으로 나타났다.

- **Empirical Impact**: 실험에서는 신뢰가 과업에 따라 뚜렷하게 달랐는데, 절차/안내 성격의 ‘교육적 가이드’ 상황에서 최고 신뢰가 보고됐고, ‘개인 정보 요청’ 상황에서 유의하게 낮아졌다. 반면 로봇 외형은 신뢰를 전반적으로 끌어올리거나 내리지 못했다. 이 결과는 교육 환경에서 로봇을 배치할 때 인플루언서처럼 보이게 만드는 ‘인간형 외형’보다, 역할과 행동을 상호작용 목표에 맞추는 ‘과업 인지적(task-aware) 배치’가 중요하다는 방향성을 강화한다.



### Sensitivity Shaping for Latent Modeling (https://arxiv.org/abs/2606.14585)
- **Prior Approaches**: 기존 연구들은 생성 동역학(world model 류)을 고정된 상태로 두고, 사후(포스트 혹)로 불확실성·지원(support) 점수(예: 앙상블 불일치, kNN 거리, 밀도/Flow 기반 가능도)를 만들어 OOD 페널티나 안전 필터에 활용해 왔다. 또한 conformal prediction으로 기준선(coverage) 성질을 맞추지만, 점수가 실제 예측 오차를 얼마나 잘 반영하는지(미지원 전이에서 확실히 거절되는지)는 보장되지 않는다.

- **Core Contribution**: 이 논문은 OOD 신호가 사라지는 핵심 실패 모드로 ‘제어 입력에 대한 국소 민감도(control insensitivity) 붕괴’를 지목한다. 동역학 모델이 제어 변화에 둔감하면, 실제론 큰 예측 오차가 있어도 잠재 공간에서는 시연 전이와 비슷하게 보여 전통적인 지원 기반 OOD 탐지가 무력화된다. 이를 해결하기 위해, 학습 중 ‘지원이 충분한 영역’에서 제어 자코비안의 크기를 유지하도록 민감도 정규화(sensitivity regularization)를 제안한다.

- **Technical Challenges**: 정규화를 무작정 전역적으로 강하게 걸면 약한 지지 영역에서 예측이 불안정하게 외삽될 위험이 있어, 민감도 향상과 안전한 일반화 사이의 균형이 필요하다. 논문은 latent-space kNN 기반 점수로 고지원(high-support) 샘플을 선별한 뒤, 해당 영역에서 제어 자코비안의 Frobenius 노름이 0으로 붕괴하지 않게 하는 형태로 정규화를 적용하고 Hutchinson 항을 사용해 계산 비용을 줄였다.

- **Empirical Impact**: 시각 기반 장애물 회피, 로봇 조작, 실내 내비게이션의 시뮬레이션과 하드웨어 실험에서 정규화된 동역학은 미지원 전이 탐지 정확도와 안전한 폐루프 계획 성능을 개선했다. 특히 one-step과 reachability 기반 안전 필터 모두에서 성공률은 올리고 OOD 위반·실패율은 낮추는 경향이 관찰되었으며, 제어 민감도 회복이 OOD 신호의 ‘식별 가능성’을 키운다는 점을 정성·정량 분석으로 뒷받침한다.



### ORCA: A Platform for Open-Source Dexterity Research (https://arxiv.org/abs/2606.14561)
Comments:
          15 pages

- **Prior Approaches**: 로봇 조작 연구는 병렬 턱(gripper) 기반이 실용적이라는 이유로 주로 정착되어 왔고, 텔레오퍼레이션도 상대적으로 쉬워 대규모 시연 데이터 구축에 유리했다. 반면 인체형 손(anthropomorphic hands)은 하드웨어는 비싸고, 제어·시뮬레이션·리타겟팅·텔레오퍼션 소프트웨어가 단일 코드베이스로 분산되어 재현성과 확장성이 크게 떨어졌다. 그 결과 손 기반 연구는 초기 통합 비용이 커져 발전이 느려지고, 기존 robot-learning 생태계와도 단절되는 문제가 있었다.

- **Core Contribution**: 이 논문은 손의 조작(dexterity)을 ‘첫 번째 클래스(first-class) 로봇 학습 도메인’으로 다루기 위한 오픈소스 학습 스택 orca를 제안한다. orca는 저수준 제어, 시뮬레이션, 텔레오퍼레이션, 리타겟팅을 단일 인터페이스로 통합하고, lerobot 같은 대표 로봇 학습 프레임워크와 네이티브로 연결해 동일한 데이터·학습·평가 파이프라인을 손 연구에도 그대로 적용하게 한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 실손과 시뮬레이션, (2) 다양한 입력 장치(예: VR 헤드셋, 모션 장갑, 미디어파이프 기반 추적)의 출력 좌표계를 일관되게 묶고, (3) 이를 실시간에 가깝게 리타겟팅해 닫힌 루프 정책 학습으로 이어지게 하는 것이다. 논문은 OrcaHand(3D 프린팅 가능한 텐던(tendon) 구동 손) 위에 orca_core의 단일 관절공간 인터페이스를 두고, orca_sim에서 URDF/MJCF 기반으로 같은 인터페이스를 재사용하며, orca_teleop의 랜드마크 정규화·스케일/회전 불변 보정과 키 벡터 매칭(허버 손실)을 통해 잡음/형태 불일치에 강한 리타겟팅을 구현한다.

- **Empirical Impact**: 실험에서는 VR 헤드셋 텔레오퍼레이션으로 in-hand reorientation 작업 시연 10개를 수집한 뒤, lerobot과 ACT로 행동 복제 정책을 학습하고 재현 가능한 설정에서 평가한다. 10회 롤아웃 중 9회 성공으로, 텔레오퍼레이션-리타겟팅-데이터 기록-학습-평가가 하나의 열린 파이프라인에서 실제로 작동함을 보여준다. 또한 MIT 라이선스로 전체 스택을 공개해 손 조작 연구의 진입장벽을 낮추고, 향후 dexterity 분야의 공통 기반(shared foundation)으로 자리잡을 가능성을 제시한다.



### TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation (https://arxiv.org/abs/2606.14551)
- **Prior Approaches**: 대부분의 비전-로봇 모방 정책은 현재 관측(혹은 짧은 최근 창)만으로 다음 행동을 결정하도록 설계돼, 현재 입력이 충분하다는 가정을 암묵적으로 둡니다. 지연-증거(delayed-evidence) 환경에서는 결정적 단서가 이미 사라져 현재 관측만으로는 분기가 구분되지 않으며, 단순히 기록 길이를 늘리는 방식도 비용이 커지거나 단서가 다른 신호에 덮여 성능이 흔들립니다.

- **Core Contribution**: 이 논문은 지연-증거 모방을 위한 고정 예산(memory-bounded)의 인과적 기억 프레임워크 TRACE를 제안합니다. TRACE는 단서를 직접 저장하기보다, 로봇이 실제로 어떤 경로를 따라왔는지 나타내는 path signature(경로 시그니처)를 키로 삼아 단서가 보였던 시점에 저장된 시각·로봇 상태 증거를 나중의 애매한 분기 지점에서 복원해 올바른 브랜치를 선택하게 합니다.

- **Technical Challenges**: 핵심 난제는 “현재 관측만으로는 부족하지만, 긴 히스토리 전체를 저장하지 않고도” 실행된 인과적 경로에 결합된 단서를 유지하는 것입니다. TRACE는 (1) raw time/라벨 인덱싱을 버리고 (2) 로봇 상태 궤적을 스트리밍으로 요약한 경로 시그니처(누적·증분)를 라우팅 키로 사용하며, (3) 고정 크기 슬롯 메모리에 증거를 선택적으로 쓰고 읽는 어텐션형 게이팅 업데이트를 통해 이 문제를 해결합니다. 또한 정책 백본과 액션 헤드, 모방 손실은 바꾸지 않고 경량 어댑터로 TRACE 메모리 조건만 주입해 플러그인 방식으로 통합합니다.

- **Empirical Impact**: 실세계 장기 조작의 5개 지연-증거 태스크(Tool, Book, Laundry, Cable, Medicine)에서 TRACE는 회귀형(action-chunking)과 확산형(diffusion) 정책 모두에 대해 분기 선택 및 과업 성공(평균 진행도)을 유의미하게 개선했습니다. 특히 단서가 초기에만 보이고 이후 시점에서는 시각적으로 거의 동일해지는 태스크에서 가장 큰 이득이 나타났고, 순수 히스토리 길이/일반 메모리 대비 TRACE가 “실행된 인과 경로에 따라 증거를 라우팅·복원”하기 때문임을 진단 및 절제 실험으로 확인했습니다.



### Spatially Conditioned Diffusion Policy: Learning Precise and Robust Manipulation with a Single RGB Camera (https://arxiv.org/abs/2606.14535)
Comments:
          15 pages

- **Prior Approaches**: 시각 모방 학습은 정밀 접촉 조작에서 보통 전역 시점(global view)과 손목 카메라(wrist-mounted cameras)를 함께 써 왔습니다. 손목 카메라는 국소 영역을 가까이서 보여 잡음을 줄이고(시각적 모호성 완화) 정책이 작업 관련 부분에 집중하게 돕는 역할을 합니다. 반면 단일 전역 RGB 카메기에서는 미세 접촉 디테일을 포착하면서도 복잡한 장면에서 작업 관련 영역을 명시적으로 고르기가 어렵습니다.
기존 단일 시점 방법은 관측을 전역 임베딩으로 압축하거나(global embedding) 시각 토큰을 분해해도 작업 관련성을 공간적으로 명시적으로 선택하지 못해 관련성 추론을 암묵적으로 맡기는 한계가 있었습니다. SKIL/OTTER처럼 세분화·키포인트·시각언어모델(VLM) 등을 쓰는 접근도 있으나, 대형 사전학습 모델과 다단계 지각 파이프라인 의존으로 에러 전파가 생길 수 있습니다. Point-cloud 기반(DP3 등) 역시 최종적으로 전역 표현으로 요약되는 경우가 많아 미세 초점(정밀 제어를 위한 공간 선택)이 부족했습니다.

- **Core Contribution**: 이 논문은 단일 RGB 카메라만으로 정밀하고 견고한 조작을 수행하는 Spatially Conditioned Diffusion Policy(SCDP)를 제안합니다. 핵심 아이디어는 “미래 말단 이펙터 궤적”이 이미지에서 작업 관련 영역을 가리키는 시각적 어텐션 앵커(attention anchor)가 될 수 있다는 점입니다.
확산(diffusion) 과정에서 중간 단계의 액션 궤적을 재구성·투영해, 해당 위치를 따라 멀티스케일 시각 특징을 점 단위로 샘플링하고 이를 디노이징 네트워크 조건으로 사용합니다.

- **Technical Challenges**: 단일 전역 시점에서 정밀 접촉을 성공시키려면, (1) 미세한 시각 디테일과 (2) 산만한 장면 속에서의 작업 관련 공간 선택을 동시에 만족해야 합니다. SCDP는 이를 위해 ResNet-18 기반 멀티스케일 이미지 인코더로 전역 문맥과 국소 디테일을 함께 뽑고, 확산 루프의 중간 액션 궤적을 재구성해 3D→2D 투영 좌표로 어텐션 앵커를 형성합니다.
그다음 멀티스케일 특징맵에서 해당 픽셀들을 bilinear interpolation으로 점 단위 특징으로 뽑아, 궤적·해상도 정보를 보존한 공간 컨텍스트를 FiLM(Feature-wise Linear Modulation)으로 U-Net 디노이저에 주입합니다. 이렇게 “외부 지각 모듈 없이” 액션 기반 쿼리로 공간 선택을 내재화하려고 합니다.

- **Empirical Impact**: 시뮬레이션에서 SCDP는 Meta-World와 DexArt 전 과제(총 54개) 설정 및 난이도 높은 단일 시점 설정에서 강한 성능을 보이며, Meta-World 난이도 그룹마다 20 데모만으로 80% 이상 성공률을 달성했다고 보고합니다. 특히 Hard/Very Hard에서 격차가 크게 나타나 단일 시점 정밀 조작에서의 효과를 뒷받침합니다.
단일 RGB 카메라 경쟁력도 확인되어 손목 카메라를 추가한 기준이나 깊이 기반 변형 대비 유사한 수준의 성능을 보이며, 데이터 효율성과 잡동사니(distractor)·클러터 환경에서도 성공률 저하가 상대적으로 작았습니다. 현실 로봇 실험에서도 USB/Battery 삽입, 정밀 그리핑, 장애물 환경에서의 견고성이 관찰되어 단일 카메라 기반 모방학습의 실용적 확장 가능성을 시사합니다.



### AERMANI-PLACE: Language Guided Object Placement with Aerial Manipulators (https://arxiv.org/abs/2606.14531)
- **Prior Approaches**: 기존 공중 조작(AM) 배치는 목표 놓기 위치를 미터 좌표(placement pose)로 미리 주는 방식이 주류다. 이때 사용자는 좌표계·장면 기하를 따져야 하고, 작은 입력/센서 오차가 안전·성공률에 직접 영향을 준다. 언어-공간 접목 연구도 존재하지만, 공중 플랫폼의 비정형 관점과 지연 문제로 AM에 충분히 잘 전이되지 못했다.

- **Core Contribution**: AERMANI-PLACE는 훈련 없이(학습 없이) 언어 지시를 공중 배치 행동으로 바꾸는 프레임워크를 제안한다. 장면 RGB 이미지와 자연어를 넣으면, 이미지 편집 모델이 ‘어디에 놓을지’를 가리키는 시각 마커(포인팅 단서)를 생성하고 이를 3D로 접지(grounding)해 실행 가능한 좌표로 만든다. 즉, 모호한 3D 좌표 입력 부담을 없애고 ‘비주얼 포인팅’ 형태의 목표 지정으로 전환한 점이 핵심이다.

- **Technical Challenges**: 핵심 난제는 언어→목표 위치의 불확실성을 로봇 실행 관점의 물리 제약으로 안전하게 바꾸는 것이다. 논문은 마커를 2D 픽셀로 뽑아 깊이 관측으로 3D metric point를 복원한 뒤, 충돌 가능성을 해결하는 로컬 보정과 함께 충격/진동을 줄이기 위한 탑다운(상방 접근-하강) 궤적을 결합한다. 또한 생성 모델의 환각·장면 왜곡·마커 공중 부유 같은 실패를 줄이기 위해 일관성 제약을 걸고, 충돌 없는 접촉 지점을 찾는 touch-down/근접 탐색으로 안정성을 확보한다.

- **Empirical Impact**: 100개 언어 지시 배치 과제에서 평균 87% 성공률(5cm 이내)을 달성했으며, 학습이 없는 방식임에도 학습 기반 SOTA와 유사한 성능을 보였다. 실제 공중 조작 하드웨어에서는 25회 중 18회로 평균 72% 성공률을 보이며 현실 전이에 대한 근거를 제공한다. 특히 센서 잡음과 공중 난류(downwash)로 인한 드리프트를 탑다운 전략과 기하 보정이 완화하는 것이 확인되어, 자율 공중 pick-and-place의 실사용 가능성을 높인다는 의미가 있다.



### CADET: Physics-Grounded Causal Auditing and Training-Free Deconfounding of End-to-End Driving Planners (https://arxiv.org/abs/2606.14438)
Comments:
          8pages 4figures

- **Prior Approaches**: 기존 E2E 자율주행 플래너는 행동 모방(imitaton)으로 학습되어, 데이터에서 함께 나타난 단서(동시발생 상관)를 의사결정 변수로 착각하는 “통계적 지름길”에 취약하다고 지적한다. 평가에 주로 쓰이는 open-loop L2 및 충돌률은 관측 입력을 제거해도 크게 변하지 않아(기준선 효과) 실제로 어떤 입력 단서에 의존하는지(인과 vs 비인과)를 잘 드러내지 못한다.

- **Core Contribution**: 논문은 CADET( Causal Auditing and Deconfounding at Test-time )를 제안하며, 학습된(고정) E2E 플래너를 재학습 없이(test-time) 감사·벤치마킹·수정한다. 핵심은 물리-기하학적 prior(지각 모듈 출력 기반)이 “그 객체가 물리적으로 결정을 바꿀 수 있는가”를 외부 기준선처럼 판단해, 의존도가 높지만 실제 원인이 아닌 단서를 찾아낸다는 점이다.

- **Technical Challenges**: 큰 어려움은 관측 기반 지표나 환경 불변성만으로는 전역적으로 통계 상관이 유지되는 ‘global spurious’ 케이스를 분리하기 어렵다는 것이다. CADET은 (1) 질의별 영향도(교란 시 계획 변화)와 (2) 물리 prior의 게이트, (3) 환경 간 안정성 항을 결합한 PCR(Physics-grounded causal audit) 점수를 만들고, 플래너 추론 시 플래그된 단서만 causal masking으로 억제해 do(·)에 근사하는 반사실적 수정(TCM)을 수행한다.

- **Empirical Impact**: SpurGen(구조적으로 인과/비인과가 라벨로 알려진 합성 벤치마크)에서 단일 신호 기반 방법들은 가정 실패로 정밀도·재현율이 크게 흔들리지만, PCR은 높은 정밀도와 재현율(F1≈0.91)을 보이며 물리 prior의 견고함도 관측 잡음에서 유지된다. 또한 nuScenes의 공개 플래너(SparseDrive)를 대상으로 한 감사가 가능하며, 훈련 없이도 spurious 의존을 구체적으로 플래그하고 TCM 적용 시 반사실적 강건성을 개선하는 방향성을 제시한다.



### Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions (https://arxiv.org/abs/2606.14433)
Comments:
          9 pages, 6 figures

- **Prior Approaches**: 로봇 학습에서 모션을 얻는 대표 방식은 모방학습(Behavioral Cloning)이나 강화학습 기반 모션 모사로, 대부분은 kinematics와 모터 액션이 포함된 시연 데이터가 필요합니다. 그런데 이런 데이터는 파이프라인 구축과 수집 비용이 커서 제작 시간이 길고, 모션도 단일 스타일에 과적되기 쉽습니다. 또 RL 정규화는 reward shaping이나 보상 회피 문제가 생겨 자연스럽고 안정적인 거동을 만들기 어렵다는 한계가 있습니다.

- **Core Contribution**: 본 논문은 Unitree Go2용 대규모 “키네마틱-상태-모터액션” 데이터셋 Kine2Go를 제안합니다. 40개 서로 다른 기준 모션에 대해 각기 다른 강화학습 정책을 학습해, 총 800개의 다양한 게이트(보행·달리기·트로팅·턴·스핀 등) 궤적을 Go2 호환 형식으로 수집합니다. 특히 정착된 휴머노이드 모캡 자원과 달리 쿼드러플드에서 바로 쓸 수 있는 기반 데이터를 제공하는 점이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술적 어려움은 서로 다른 체형의 모션을 Go2의 관절/자유도 제약에 맞게 “정합”시키고, 그 다음 RL이 이를 추적해 물리적으로 타당한 모터 명령까지 생성하도록 만드는 데 있습니다. 이를 위해 (1) 소스 모션을 Go2 12-DoF에 맞추는 kinematic retargeting과 IK 기반 프레임 단위 정합, (2) 목표조건(goal-conditioned) MDP에서 PPO로 기준 궤적 추적 정책 학습, (3) 시뮬레이터 롤아웃 후 넘어짐·충돌·큰 이탈 같은 불안정 궤적을 필터링하는 절차를 Genesis GPU 물리엔진 위에 구성했습니다. 또한 시작/종료 프레임 불일치 문제를 해결하기 위해 기준 모션을 사이클링하고, 초기 웜업 구간은 트리밍해 데이터 품질을 높였습니다.

- **Empirical Impact**: Kine2Go는 800개 궤적·proprioceptive 상태·정규화된 모터 액션을 포함해 “오프더셸프(off-the-shelf)”로 쿼드러플드 저수준 보행 학습에 바로 활용될 수 있도록 설계됐습니다. 저자들은 본 데이터셋이 behavioral cloning, motion-conditioned 정책 학습, 그리고 기반(locations foundation) 모델의 정규화/훈련에 특히 유용하다고 제시합니다. 점프나 고르지 않은 지형 같은 영역은 아직 제외되었지만, 플러그인형 파이프라인과 확장 방향(점프·거친 지형·다양한 환경)을 통해 후속 연구를 촉진할 것으로 기대됩니다.



### ForestBack: Breadcrumb-Based Pedestrian Dead Reckoning for Infrastructure-Free Return Navigation (https://arxiv.org/abs/2606.14421)
Comments:
          9 pages, 6 figures, 1 table, and 19 equations

- **Prior Approaches**: 기존 PDR(보행 사망천이항법)은 가속 기반 보행(step) 감지, 보폭(step-length) 추정, 방향(heading) 추정으로 시작점 대비 상대 위치를 갱신한다. 다만 step-length·heading 오차가 누적되어 드리프트가 커지고, 자력계(magnetometer) 교란과 저가 IMU의 잡음·바이어스가 경로를 쉽게 벗어나게 만든다. 또한 많은 선행 연구는 현재 위치 추정이나 지도·무선 인프라 보정을 중심으로 설계되어, 인프라 없이 “되돌아가기(리턴 내비게이션)”의 경로 가역성(route preserving)을 직접 보장하기 어렵다.

- **Core Contribution**: ForestBack은 GPS·Wi‑Fi·Bluetooth 비컨·사전 인프라 없이, 사용자의 보행 경로를 가역적(bidirectional) “브레드크럼(breadcrumb) 노드” 시퀀스로 기록하고 역방향 경로 안내를 생성하는 인프라 프리 리턴 내비게이션 프레임워크를 제안한다. 동시에 가속 기반 step 감지, 적응형 step-length 추정, 자력계 보조 heading 추정, 기압 고도(barometric-altitude) 보정, 그리고 브레드크럼 경로 재구성(양방향 복원)을 통합해 경로 구조 보존에 초점을 둔다. 결과적으로 단순히 최종 좌표만 맞추는 것이 아니라, 되돌아가기 동안 유효한 안내가 가능한 형태의 경로 재구성을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) step·거리·방향 오차가 누적되는 드리프트 문제, (2) 자력계 교란에 따른 heading 불안정, (3) 리턴 안내를 위해 경로를 “되감기 가능한 형태”로 표현·복원해야 한다는 점이다. ForestBack은 가속 크기 기반 피크 검출에 이동평균 평활과 적응형 임계값을 적용해 오탐을 줄이고, 보폭을 보행 역학과 기압 고도 변화(기울기 관련 항)까지 반영하는 적응형 모델로 추정한다. heading은 자이로 단기 적분에 자력계 보조를 EKF(확장 칼만 필터)로 결합하되, 자기 이상이 감지되면 자력계 보정 가중치를 낮춰 교란 영향을 완화하며, 기록된 브레드크럼을 역순으로 변환해 리턴 안내를 만든다.

- **Empirical Impact**: 실험은 장애물 회피를 포함한 A–B–C–D–E 5 체크포인트 실내 경로에서 36회 보행 시험(총 42,474개 IMU/자력계/기압 시계열 샘플, 100Hz)으로 수행되었고, 전통적 PDR 기준선과 비교했다. ForestBack은 평균 RMSE를 1.129m에서 0.965m로 낮춰 15.76% 개선을 보였으며, 평균 최종 위치 오차도 1.781m에서 1.388m로 감소했다. 또한 턴 이벤트 탐지 일관성이 약 99.90%에 달해 경로 구조 기록의 신뢰성을 뒷받침하며, 공개된 데이터셋과 분석 노트북이 인프라 프리 PDR 리턴 내비게이션 벤치마킹의 재현성을 강화한다.



### Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack (https://arxiv.org/abs/2606.14409)
- **Prior Approaches**: Vision-Language-Action(VLA)은 지속 제어에서 유망하지만, 실제 로봇 배치에는 데이터·학습·적응·실행을 하드웨어 제약에 맞춰 함께 설계해야 한다는 한계가 크다. 기존 텔레오퍼레이션은 마스터-슬레이브 방식 탓에 조작이 비자연스럽고, 촉각(힘) 정보가 빈약해 정밀 조작에 취약하다. 또한 UMI류는 데이터 다양성을 늘리지만 시각·레이블 정밀도(특히 SLAM 의존이나 손끝 힘 전달의 부족)에서 병목이 생기며, 크로스-임보던스 적응은 운동/제어/지각 간 ‘갭’ 때문에 추가 난도가 붙는다. 정책 측면에서는 이산 행동 토큰 기반은 정밀도·속도에 제약이 있고, 연속 제어의 강화학습은 보상·가치 모델 의존이나 배포용 고주파 폐루프 서빙이 기본 목표가 되지 못했다.

- **Core Contribution**: HyVLA-0.5는 VLA를 ‘정책 모델’이 아니라 데이터 수집부터 연속행동 학습, RL 후처리, 실제 로봇 배치까지 잇는 end-to-end 파이프라인으로 통합해 문제를 체계적으로 푼다. 핵심 기여는 (1) 손끝 수준 촉각과 고정밀 행동 레이블을 함께 주는 fingertip UMI+모션캡처 기반 데이터 구축, (2) Mixture-of-Transformers(MoT) 기반 embodied VLM에 conditional flow matching 기반 연속 행동 expert를 결합해 고주파 제어를 가능케 한 모델링, (3) 보상/가치 네트워크 없이도 선호를 통해 실패를 개선하는 FlowPRO를 둔 RL 후처리, (4) 델타-청크를 매끈하게 연결하고 폐루프 실행을 고속화한 배포 프레임워크다.

- **Technical Challenges**: 가장 큰 기술 과제는 ‘사람-로봇’ 임보던스 차이를 줄이면서도 손끝 정밀 조작에 필요한 연속 제어를 안정적으로 학습·실행하는 것이다. 논문은 (a) 행동 레이블을 SLAM이 아닌 광학 모션캡처로 획득해 고충실도 6-자유도 궤적을 제공하고, (b) 정책 출력은 로봇 고유 기구학에 덜 종속되도록 end-effector-frame의 델타-청크로 표현해 임보던스 간 학습을 쉽게 하며, (c) 모델은 연속 행동을 flow matching으로 직접 생성해 디스크리타이즈의 정밀도 손실을 줄인다. RL에서는 보상모델의 취약성을 피하려고 Teleoperation 개입-롤백으로 성공/실패 궤적 쌍을 모아 reward-free Proximalized Preference Optimization(PRO) 계열인 FlowPRO로 오프라인 선호를 정렬하고 망각을 억제하는 성질을 활용한다. 마지막으로 배포에서는 백본 추론과 실행을 비동기 오버랩하고, 델타 청크를 cubic Bézier 스무딩으로 C1 연속 전이로 이어 폐루프 고주파 실행 제약을 만족시킨다.

- **Empirical Impact**: 논문은 Hy-UMI-10K에서 1010K 시간 규모의 시연을 학습 자원으로 삼고, 동종·이종 임보던스에 대한 SFT 트랙(Track A/B)과 이후 FlowPRO 기반 RL 후처리를 통해 정교 조작 강건성과 성공률을 끌어올리는 흐름을 보인다. 특히 실패 사례를 빠르게 반복 학습 루프로 회수해 장꼬리(long-tail) 조작의 성능을 개선하고, 보상/가치 네트워크 없이도 near-ceiling 수준에 가까운 성공률로 향상시키는 점을 강조한다. 또한 델타-청크+비동기 서빙+연속 스무딩 조합으로 실제 하드웨어 폐루프에서 고주파 제어를 구현해, VLA의 ‘모델 성능’에 머물지 않고 배치 가능성을 실증했다. 이 결과는 로봇 일반가자에 필요한 시스템 공동 설계(데이터-학습-배포)의 중요성을 한 단계 끌어올리는 사례로 해석된다.



### Elastic Queries Reinforcement Learning: Self-Aware Policy Execution for VLA Models (https://arxiv.org/abs/2606.14375)
- **Prior Approaches**: 기존 VLA(vision-language-action) 정책은 고정된 추론 스케줄로 실행되는 경우가 대부분이라, 상태의 난이도에 따라 계산을 유연하게 늘리거나 줄이기 어렵다. 특히 생성(denoising) 예산과 재계획 전 행동 청크 길이를 수동 하이퍼파라미터로 두어 쉬운 구간엔 과투자하고 어려운 접촉/정렬 구간엔 피드백이 부족해질 수 있다. 또한 미세조정이나 가벼운 적응을 하더라도, 실행 스케줄 자체는 학습 문제로 다루지 않는 한계가 있었다.

- **Core Contribution**: EQRL(Elastic Queries Reinforcement Learning)은 VLA의 각 쿼리를 탄력적으로 만들기 위해, 고정된 생성기(사전학습된 VLA)는 그대로 두고 쿼리 수준에서 잠재 입력(steering), denoising 예산, 행동 청크 길이를 함께 선택하는 프레임워크를 제안한다. 이를 통해 상태 난이도에 따라 “더 계산할지/덜 계산할지”, “더 짧게 커밋하고 다시 계획할지/길게 오픈루프로 갈지”를 정책이 결정할 수 있게 한다. 학습은 스케줄 전용의 경량 어댑터와 잔차(residual)로 수행되어, VLA 본체의 비싼 파인튜닝 부담을 줄인다.

- **Technical Challenges**: 핵심 난관은 연산(NFE) 비용이 태스크 학습을 압도하거나, 제한 없는 스케줄이 상수/노이즈성 가치로 붕괴할 수 있다는 점이다. EQRL은 여러 critic의 앙상블 불일치로 상태 난이도 신호를 만들고, 그 신호가 “어려운 상태엔 더 보수적으로(더 많은 denoising, 더 짧은 청크)” 계산을 배분하도록 스케줄 우선(baseline+prior)을 제공한다. 또한 청크 길이가 시간 스케일을 바꾸므로 쿼리 수준 거시 행동(macro-action)으로 학습하고, 청크 의존 discount와 에피소드 수준 NFE 예산 밴드를 함께 두어 전체 비용을 제어하면서도 난이도 정렬을 유지한다.

- **Empirical Impact**: LIBERO와 ALOHA 시뮬레이션, 그리고 ALOHA/실로봇 매니퓰레이션(오프라인·온라인)에서 EQRL은 태스크 성공률을 보존하거나 개선하면서도 amortized 추론 비용을 줄였다. 특히 단순히 평균 NFE를 낮추거나, denoising 또는 청크 길이 중 하나만 동적으로 바꾸는 제한 실험에서는 성공 곡선이 약화되어, critic 기반 난이도 인지와 조인트 스케줄링의 결합이 필요함을 보였다. 실로봇에서도 벽시계(latency)와 접촉-재계획의 균형 측면에서 실제 실행 속도 개선 및 안정적인 성능 유지가 관찰되어, “탄력적 계산·재계획 인터페이스”의 실효성을 보여준다.



### Robust Fall Recovery for Armless Bipedal-Wheeled Robots Via Force-Guided Learning (https://arxiv.org/abs/2606.14270)
Comments:
          8 pages, 6 figures, accepted by IEEE Robotics and Automation Letters (RA-L)

- **Prior Approaches**: 모델 기반 접근은 사전 계산된 궤적(모션캡처/최적화 등)을 실행하는 방식이라 결정적이지만, 초기 자세 변화에 민감하고 실제에서 마주치는 다양한 낙하 포즈에 대한 일반화가 약합니다. 강화학습·모방학습 기반 접근은 성공률과 강건성을 개선했으나, 계단형 학습/커리큘럼·단계별 보상 설계에 의존해 조기 수렴(로컬 최적점, dead point) 문제가 남아 있습니다. 특히 팔이 없고 추가 다리 지지가 없는 이족 바퀴형 로봇은 상체/다른 지지 수단 없이 다리 구동만으로 버텨야 해서, reward 중심 학습이 더 쉽게 무너집니다.

- **Core Contribution**: FTSR는 낙하 복구를 위해 외부 보조힘을 시뮬레이션 학습 중 ‘키-연동 외력(높이 상관)’으로 구성하되, 이를 단계적으로 줄이는 커리큘럼이 아니라 CPO의 제약(optimizable constraint) 형태로 명시적으로 최적화 가능하게 만듭니다. 또한 서기(자세 다듬기)에서 걷기(지속 보행)로 전환되도록 height-progressive stage-wise rewards를 배치해, 보조 개입 감소 이후에도 자세 안정과 이후 동작을 같이 학습하게 합니다. Teacher-student 구조로 힘의 효과와 복구 동역학에 관한 특권 정보를 증류해, 팔 없이도 내부 복구 전략을 빠르게 형성합니다.

- **Technical Challenges**: 핵심 난제는(1) 보조힘을 줄이는 과정에서 reward-only 학습이 dead point로 조기 붕괴하는 것, (2) 외력이 사라진 뒤에도 서기 자세를 안정적으로 유지하며 지속 보행까지 이어지게 하는 것, (3) 특권 힘 정보를 쓰되 실제 실행은 관측(자기 정보)만으로 가능하게 하는 것입니다. 논문은 힘·토크를 로봇 중심부 높이와 시간 진행에 연동해 제약 조건으로 ‘물리적으로 feasible한 복구 궤적’ 쪽으로 탐색을 제한하고, 높이 통계 기반 임계값으로 보상 단계를 자동 전환해 학습 안정성을 높였습니다. 더불어 teacher는 접촉력·자세·힘 관련 정보를 처리하고 student는 과거 관측 이력만으로 같은 표현을 학습하도록 설계해, 제약 유도 학습의 효율과 전이성을 함께 확보합니다.

- **Empirical Impact**: FTSR는 시뮬레이션에서 다양한 지형(경사, 단차, 요철, 거친 표면 등)과 랜덤화된 초기 자세에 대해 강건한 낙하 복구를 달성하며, 단계별 구성요소(ablation) 중에서도 force-guided 제약과 height-progressive 보상이 성능에 결정적임을 보입니다. 특히 팔 없는 이족 바퀴형 로봇에서 보조힘을 제거한 뒤에도 안정적으로 서기·걷기 능력을 유지하고, 명령 속도 변화에도 높은 성공률을 보였다고 보고합니다. 나아가 23-DOF 휴머노이드에 대해서도 유사한 회복 성능과 더 빠른 학습(적은 반복으로 부드러운 복구)을 실험적으로 확인해, 현장 적용 가능성과 일반화 잠재력을 뒷받침합니다.



### FloVerse: Floor Plan-Guided Multi-Modal Navigation (https://arxiv.org/abs/2606.14267)
Comments:
          Accepted at CVPR 2026

- **Prior Approaches**: 기존 체화형 내비게이션은 맵 기반(명시적 지도·그래프 구성)과 맵리스 방식(관측→행동 직접 예측)으로 크게 나뉜다. floor plan(평면도)을 공간 기준선으로 쓰는 연구도 있었지만, 주로 PointNav에 한정되고 실험 환경/스케일이 작아 ObjectNav·ImageNav로의 확장이 충분히 검증되지 않았다. 또한 평면도는 가구 같은 이동 가능한 요소가 빠져 있어 충돌 없는 경로를 직접 계획하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 floor plan 기반 embodied navigation을 PointNav·ObjectNav·ImageNav의 세 모달리티로 통합한 새로운 태스크 FloVerse를 제안한다. 이를 위해 HM3D와 Gibson 4+에서 얻은 1,627개 장면 레벨의 floor plan과, 240K 전문가 궤적 및 12M RGBD-포즈 페어로 구성된 FloVerse-1.6K 데이터셋을 구축한다. 모델 측면에서는 두 단계 정책 ThreeDiff를 제안해, 전역적인 평면도 공간 priors와 국소 깊이 정보를 함께 사용한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 평면도가 제공하는 구조적·(부분적으로) 의미적 규칙을 목표 추론과 계획에 연결하되, (2) 평면도의 부재 정보(가구 등)로 인한 충돌 위험을 국소적으로 보정하는 것이다. ThreeDiff는 diffusion 기반 planner로 평면도와 목표 조건을 바탕으로 장기 의도에 맞는 거친 궤적을 예측하고, 이어서 깊이 기반 점유/거리정보(SDF)를 이용한 refiner가 충돌 회피가 되도록 궤적을 정제한다. 또한 학습 시 모달리티 랜덤 마스킹으로 세 목표 형식을 하나의 모델에서 안정적으로 다루도록 설계하고, masked-modality goal reasoning으로 목표-공간 정합을 유도한다.

- **Empirical Impact**: 실험 결과 floor plan 정보는 세 모달리티 모두에서 성공률과 경로 효율(SPL)을 일관되게 향상시켰으며, 특히 PointNav에서 개선 폭이 가장 컸다. ImageNav·ObjectNav에서도 성능 향상이 관측되는데, 이는 평면도 priors가 방의 연결성과 거친 레이아웃 같은 구조적 지식을 목표 위치 추정에 간접적으로 제공하기 때문으로 해석된다. 나아가 ThreeDiff는 모달리티별 단일 목적 모델과 비교해 경쟁력 있는 수준이거나 경우에 따라 우수하며, floor plan으로부터 공간 정보를 암묵적으로 학습한다는 정성/정량 신호가 함께 제시된다.



### ReactVLA: Fast and Lightweight Reactive Robot Manipulation via Improved Mean Flow Action Generation (https://arxiv.org/abs/2606.14255)
- **Prior Approaches**: 확산(디퓨전) 기반 VLA는 작동을 반복적 잡음 제거로 구성해 다양한 행동 분포를 잘 표현하지만, 배포 시 수십 번의 순차 샘플링(또는 적분) 때문에 추론 지연이 커진다는 한계가 있었다. 최근에는 Flow/Rectified Flow처럼 샘플링 효율을 높이는 시도도 있었지만, 극단적으로 낮은 저단계(low-step)에서 경로 불일치와 수치 오차가 커져 품질이 흔들릴 수 있다. 또한 트랜스포머에서 잔차(residual)를 고정 방식으로 누적하면 깊어질수록 멀티모달 표현이 희석돼 정밀 제어에서 성능 저하가 나타났다.

- **Core Contribution**: ReactVLA는 반응형(reactive) 폐루프 로봇 조작을 목표로, 확산식 표현력을 유지하면서 추론을 “1~몇 단계”로 줄이는 프레임워크를 제안한다. 이를 위해 (1) improved Mean Flow(iMF)로 로봇 행동을 유한 구간의 평균 수송(average transport)으로 예측해 저단계 생성의 안정성을 높이고, (2) Attention Residuals(AttnRes)로 트랜스포머 깊이 전반에서 멀티모달 정보를 동적으로 라우팅해 표현 희석을 완화한다.

- **Technical Challenges**: 핵심 기술 난제는 저단계 생성에서 로컬 속도 예측 기반의 기존 흐름이 불안정해질 수 있다는 점과, 그 불안정성이 Jacobian-Vector Product(JVP) 보정 경로를 통해 학습을 폭주시키기 쉽다는 점이다. ReactVLA는 iMF의 JVP 보정(스톱그라디언트 포함)으로 평균 수송 예측과 내부 동역학의 정합성을 맞추고, 손실함수는 MSE 대신 Pseudo-Huber로 큰 오차의 그래디언트 폭주를 억제해 학습 안정성을 확보한다. 여기에 AttnRes의 깊이별 입력 의존적(feature routing) 재가중으로 시각·언어·프로프리오셉티브 신호가 저단계에서도 충분히 보존되도록 설계했다.

- **Empirical Impact**: LIBERO와 RoboIMI 시뮬레이션 및 Diana 7 실로봇에서 ReactVLA는 유사 크기 VLA 기준 SmolVLA, π0 등과 비교해 일관되게 높은 성공/보상 성능을 보인다. 특히 정밀 조작에서 최대 1.65배 성능 향상과 함께, 주요 모델 대비 4배 이상 추론 속도 개선 및 물리 로봇 정책 지연 38.6ms 미만을 달성해 고주파 폐루프 제어가 가능함을 실증했다. 결과적으로 “확산식 표현력”을 “실시간 반응성”과 동시에 달성하는 실용적 경로를 제시하며, 반응형 로봇 조작에서 생성형 정책의 적용 범위를 넓히는 데 의미가 크다.



### Optimality-Preserving Decomposition for Scalable QAOA in Natural-Language-Guided Multi-Drone Assignmen (https://arxiv.org/abs/2606.14252)
Comments:
          10 pages, 2 figures, 3 tables, preprint

- **Prior Approaches**: 기존 다중 드론 구역 배정은 사람이 임무 목적을 제약식이나 수식으로 바꿔야 해 시간이 오래 걸리고, 논리 누락이 생기기 쉽습니다. LLM 기반 자동화도 있었지만(예: 단일 로봇 중심), 멀티 드론으로 확장 시 환각된 제약이 섞이거나 최적화 보장이 약해지는 한계가 컸습니다. 또 QUBO를 만들더라도 완전 탐색/정확 해법은 조합 폭발로 실시간 운영이 어렵습니다.

- **Core Contribution**: 이 논문은 자연어 임무를 구조적으로 견고한 QUBO 제약으로 변환하는 LLM 전면부와, 하드웨어 제약을 고려해 풀 수 있는 양자-고전 백엔드를 한데 묶는 엔드투엔드 프레임워크를 제안합니다. 특히 제약 누락( false negatives )을 거의 없애기 위해 SFT와 DPO를 결합해 구조적 정확도를 높이고, 하드 제약은 페널티가 아니라 회로 수준에서 구조적으로 강제합니다. 그 결과, 사람이 말한 임무를 최적 할당으로 연결하는 경로를 스케일까지 확장합니다.

- **Technical Challenges**: 가장 큰 문제는 (1) LLM이 만든 큰 QUBO를 그대로 QAOA에 넣기엔 NISQ의 큐빗 수 한계가 너무 작고, (2) 페널티 항 중심 인코딩은 최적화 지형을 망가뜨릴 수 있다는 점입니다. 이를 위해 제약 보존형 그래프 분할과 충족 경계 상태를 압축한 separator-based 동적 계획법(merge)로 전역 최적을 유지한 채 문제를 하위 문제로 쪼갭니다. 또한 one-hot 제약의 무거운 패널티 오버헤드를 줄이기 위해 W-state 초기화와 Hamming-weight 보존 XY-mixer를 CVaR-QAOA에 결합해 유효 부분공간만 양자 진화를 수행하도록 설계했습니다.

- **Empirical Impact**: 이 프레임워크는 이상화된 오라클 조건에서 100% 전역 최적을 회복했고, 실제 CVaR-QAOA 샘플링에서도 300개 시나리오 중 96.3%에서 전역 최적 일치를 보였습니다. 제약 추출 품질도 SFT+DPO 전면부에서 recall 99.7% 수준으로 확인돼, 최적화 단계로 넘어가기 전에 경계 제약이 크게 누락되지 않음을 입증합니다. 특히 완전 탐색이 지수 벽에 막히는 스케일 이후에도 실행 시간이 큐빗 예산(상수)에 의해 제한되며, “정확성의 소폭 손실(3.7%)을 얻고도 지수적 가속을 획득”하는 실용적 트레이드오프를 보여줍니다.



### SyLink Hand: A Synergy-Inspired Linkage-Driven Anthropomorphic Hand for Human-Like Dexterity (https://arxiv.org/abs/2606.14250)
- **Prior Approaches**: 기존 연구는 인간 손의 자유도를 모사하기 위해 모터·센서를 늘리거나, 손의 ‘시너지’를 PCA 등으로 추출해 구동 자유도를 줄이는 방식으로 절충을 시도해 왔다. 그러나 시너지 기반 축소 설계는 복잡한 조작 일반화에 한계가 있고, 2-DOF 기저관절을 구현하는 차분( differential ) 기어·연성체인 구조는 공간/기구 복잡도가 커지며 두 자유도를 동시에 제어해야 하는 제약도 남는다.

- **Core Contribution**: 이 논문은 손동작 데이터에서 관절 간 상관(시너지) 패턴을 정량 추출해, ‘필요 DOF만 유지’하는 간소화된 관절 분배를 설계 기준으로 제시한다. 여기에 링크리지 기반 결합 구동(특히 PIP–DIP, 엄지의 Flex/Ext)과 MCP의 2-DOF를 한 개의 공간 효율적인 구동 구조로 분리제어하는 구성을 결합해, 외형·운동·기능에서의 인체형(anthropomorphic)성을 실용 아키텍처로 달성하는 것이 핵심 기여다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 인체 관절 시너지의 상관구조를 유지하면서도 결합 구동을 컴팩트한 링크로 구현하는 것, (2) MCP의 Flex/Ext와 Abd/Add를 기존 방식처럼 동기 모터 2개 없이 분리제어하면서도 간섭 없이 동작공간을 확보하는 것이다. 저자들은 인체 모션캡처 장갑 데이터의 상관만 인접 관절에 한정해 결합 그룹을 만들고, PSO 기반 비선형 최소제곱으로 링크 파라미터를 최적화해 자연 궤적을 재현한다; MCP에는 구면(spherical) 4절 링크를 설계해 두 회전 축이 구면 제약을 이루도록 만들어 기구적으로 분리제어를 가능케 했다.

- **Empirical Impact**: 제안된 SyLink Hand 프로토타입은 19개 관절을 11개 구동기로 구동하며 질량 520g, 제작비 약 400달러 수준으로 구현된다. 실험에서는 인간 유사 운동학 성능, 높은 하중 지지(25N+), 그리고 다양한 그립·인핸드 조작 능력이 보고되어 시너지-기반 링크리지 설계가 ‘인체형과 단순성·다기능성’의 동시 달성을 지지함을 보여준다. 이는 값비싼 고자유도 로봇손 대신, 비용과 공간 제약이 있는 실제 응용으로의 확장 가능성을 강화하는 근거로 의미가 있다.



### When and How Severely: Scenario-Specific Safety Envelopes for Driving VLAs (https://arxiv.org/abs/2606.14238)
- **Prior Approaches**: 기존에는 센서 잡음(또는 입력 변형)에 대한 평균 성능 저하나 단일 기준선(예: ADE가 임계치 이하)을 ODD(Operational Design Domain) 경계로 사용해 왔다. 또한 VLA(vision-language-action)에서 설명(Chain-of-Causation) 변화는 “이진(바뀜/안 바뀜)”으로만 다뤄져, 실패가 일어났을 때의 ‘심각도 분포’ 형태는 충분히 분해되지 않았다. 이런 집계 지표는 시나리오마다 임계치가 다르고 실패가 이산적인 심각도대로 뭉치는 현상을 가릴 수 있다.

- **Core Contribution**: 이 논문은 SOTIF(ISO 21448) 관점에서 VLA의 ODD를 “한 개의 잡음 허용치”가 아니라 두 축으로 명시한다. 첫째, 시나리오별로 기준선(ADE) 대비 15% 예산을 처음 초과하는 시점(when)을 σ*(s) 형태로 제시한다. 둘째, 실패를 유발한 경우 설명이 바뀐 샘플들에서 궤적 오차가 어떤 심각도 밴드로 분포하는지(how severely)를 6개 밴드로 분해해, 같은 평균 오차라도 고심각도 비율이 달라질 수 있음을 보여준다.

- **Technical Challenges**: 기여를 위해서는 (1) 시나리오별 안전 임계치를 신뢰도 있게 추정하고 (2) 실패의 오차 분포를 이산 밴드 구조로 복원하며 (3) 두 결과를 동일한 코퍼스에서 교차 결합해야 한다. 논문은 clip-단위 부트스트랩으로 σ*(s)의 신뢰 구간을 확보하고, CoC가 바뀐 5,443개 조건에 대해 궤적 L2 오차 분포를 GMM(Gaussian Mixture Model)으로 피팅해 BIC로 k=6을 선택했다. 마지막으로 σ*(s)와 고심각도(C4/C5) 비율 P(C4∪C5|coc_changed)을 함께 매핑해, 단일 집계 임계치가 드러내지 못하는 불일치(과보호/과소보호)를 통합적으로 드러낸다.

- **Empirical Impact**: 15,968개 (clip, attack)에서 평균 기반 보수적 집계(σ≤50)는 일부 시나리오를 과도하게 막고, 다른 시나리오는 여전히 취약할 수 있음을 확인했다. 특히 σ≤70까지도 안전 임계치가 유지되는 시나리오가 존재하지만, 실패 시에는 다른 시나리오들이 더 많은 고심각도(C4/C5)로 치우친다(예: STOP_SIGNAL은 LANE_KEEPING 대비 C4/C5 비율이 약 4배). 결과적으로 배치 가능한 SOTIF 스타일 ODD 스펙은 각 위험(해저드)에 대해 단일 숫자 허용치가 아니라 ‘시나리오별 임계치 + 실패 심각도 비율’이라는 2차원 안전 엔벨로프가 필요하다는 점을 실증적으로 뒷받침한다.



### BIM-Loc: BIM-Integrated Discrepancy-Aware LiDAR-based Indoor Localization (https://arxiv.org/abs/2606.14237)
Comments:
          24 pages, 21 figures, accepted by International Journal of Robotics Research (IJRR), to be published

- **Prior Approaches**: GNSS가 없는 실내에서 LiDAR SLAM은 일반적으로 scan matching 전단(odometry)과 pose-graph 최적화 후단(map optimization)을 결합해 드리프트를 줄여 왔습니다. 또 한, 루프 클로저나 사전 맵(바닥도면, 점군맵, 2D/3D 메시, 메시/확률 표현)을 통해 기준을 보완하지만, 복도처럼 단방향 동선에서는 루프가 잘 생기지 않습니다. BIM을 활용한 연구도 있었으나 대부분 BIM 표면에서 점을 샘플링해 점기반 정렬을 수행하거나, BIM-현장 불일치를 충분히 SLAM 최적화 안에서 직접 다루지 못했습니다.

- **Core Contribution**: 이 논문은 BIM과 실환경의 불일치(BIM-reality discrepancy)가 있는 상황에서조차 3D LiDAR 위치추정을 견고하게 만들기 위해 BIM-Loc을 제안합니다. BIM을 ‘as-designed’ 상태 그대로 사전 맵으로 쓰되, 관측과 BIM 간 차이를 온라인으로 추정하면서 동시에 BIM 좌표계에 정렬된 궤적을 추정합니다. 특히 BIM-현장 불일치 탐지를 궤적 최적화와 결합해, 단순 재지역화가 아닌 지속적인 상태 업데이트를 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 (1) 관측 포인트가 어떤 BIM 구조/면에 해당하는지 안정적으로 대응(데이터 연관)해야 하고, (2) BIM에 존재하지 않거나 변형된 요소가 포함된 불일치 환경에서 일관성 제약을 최적화에 반영해야 하며, (3) 불일치를 실시간으로 갱신하면서 메모리 사용량도 줄여야 한다는 점입니다. 이를 위해 다중 교차(multi-hit) ray casting으로 BIM 구조-면 연관을 만들고, 3D 관측을 BIM의 2D texture 공간으로 UV 좌표(바리센트릭 좌표) 투영해 잔차를 계산합니다. 또한 pose graph 최적화에 BIM 통합 팩터를 넣어 odometry·순차 스캔·BIM 구조 간 일관성을 강제하고, 계층적 베이지안 추론으로 픽셀 수준 갱신을 구조 수준 불일치 지표로 전파합니다.

- **Empirical Impact**: 시뮬레이션과 실제 건설 현장/실내 데이터에서 BIM-Loc은 기존 map 기반 방식 대비 위치추정 정확도와 견고성이 유의미하게 향상되었다고 보고합니다. 특히 사전 점군맵처럼 대규모 점 데이터에 의존하지 않고 ‘as-designed BIM만’으로 동작하면서도, BIM-현장 불일치가 큰 상황에서 드리프트를 억제하는 성능을 보입니다. 이는 AEC 맥락에서의 BIM 연계 로봇(점검·서비스)의 실사용 가능성을 높이고, BIM 불일치를 SLAM 최적화의 일부로 다루는 방향성을 제시한다는 의미가 있습니다.



### Selective Agentic Recovery for UAV Autonomy with a Persistent Mission Runtim (https://arxiv.org/abs/2606.14219)
Comments:
          17 pages, 2 figures. Preprint

- **Prior Approaches**: 기존 에이전틱 AI 기반 로봇/드론 연구는 언어 모델의 추론을 행동으로 연결하는 API·툴 사용, reason-act-observe 루프 등을 제안했지만, 물리 UAV에서는 호출 빈도(주기적/상시)나 전문가 규칙에 의존하는 경우가 많았습니다. 또한 원격 출력이 곧바로 비행 명령으로 라우팅되면 검증과 안전 필터링이 어려워져, 선택적 호출(admission) 설계의 런타임 부담이 남아 있었습니다.

- **Core Contribution**: 이 논문은 Persistent Mission Runtime(PMR)으로, 드론의 미션 루프와 안전·실행 권한은 로컬에서 유지하면서 원격 에이전틱 추론을 ‘온디맨드 복구 모듈’로만 호출하도록 정리합니다. 원격 reasoner의 출력은 미리 정의된 복구 스킬 중 하나로 제한되고, 파싱·로컬 검증·안전 차단·실행 매핑을 거친 뒤에만 비행에 영향을 줍니다. 더불어 호출의 필요성을 판단하는 Learned Cognitive Value of Invocation(learned-CVI)로 비용 대비 효용이 클 때만 호출되게 합니다.

- **Technical Challenges**: 핵심 난제는 원격 호출이 지연·토큰 비용·백엔드 불확실성을 키우는 만큼, ‘언제’ 호출해야 단기 회복이 충분히 이득이 되는지 정량화하는 것입니다. 논문은 진행 정체/막힘 같은 런타임 증거를 포함한 압축 18차원 의미 요약 기반의 admission 게이트(learned-CVI)를 학습해, 단기 복구 유틸리티가 로컬 계속 실행을 이길 확률을 추정합니다. 여기에 쿨다운·예산·터미널 반경 억제·하드 stuck 가드 등 고정 런타임 가드로 무분별한 과호출을 억제하며, 호출된 출력은 스킬 계약 기반의 닫힌 구조로 검증 가능하게 만듭니다.

- **Empirical Impact**: Gazebo/PX4 400회 고정 시뮬레이션(8개 시나리오)에서 PMR은 로컬 전용의 hard/ambiguous Clean Success@1m 5.0%를 95.0%로 끌어올렸고, one-shot/periodic 호출 대비 각각 20.0/32.5%p 높은 성능을 보였습니다. 또한 수동 규칙 기반 기준선 대비 원격 reasoner 호출 횟수 16.7%, 로그 토큰 29.2%를 줄이면서도 성공률을 유지해 효율적 ‘희소 호출’의 이점을 실증했습니다. Crazyflie 나노 쿼드콥터 실험에서도 blocked navigation에서 10/10 Clean Success@1m를 달성해, 의미 요약 기반 감지에서도 적용 가능함을 보여주었습니다.



### Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback (https://arxiv.org/abs/2606.14218)
- **Prior Approaches**: 기존 VLA 및 로봇 world model 연구는 주로 힘·토크 데이터의 부재로 ‘순응’ 학습을 약하게 다뤘습니다. 또한 ALOHA·GELLO 같은 원격조종 데이터 수집은 관절 토크를 노출하지 않아 능동 순응(지금의 저항에 따라 힘을 조절)과 리더-팔/팔 사이 저항의 실시간 햅틱 피드백이 제한됩니다. UMI 그리퍼처럼 손으로 힘을 느끼는 방식도 있지만, 주로 엔드이펙터 중심 기록이라 충돌 제약이 빡빡한 환경에서 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Universal Manipulation Exoskeleton(UME)을 제안해, 원격조종 중 실시간 힘·토크(관절 토크 시그널)를 함께 기록하고 조작자가 투명한 햅틱 토크를 느끼게 합니다. 그 결과 토크 모달리티를 학습에 직접 활용해 능동 순응 정책을 만들고, 전완/어깨/손목 전체 관절 구성(whole-arm configuration)까지 함께 써서 제약 공간에서의 전신·양손(bimanual) 조정을 강화합니다. 더 나아가 보편적 리타게팅(retargeting)으로 서로 다른 로봇 팔(예: OpenArm, Franka, X-ARM)까지 동일한 조작 경험을 옮길 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술 난제는 ‘투명한 햅틱 토크’와 ‘로봇 간 보편 리타게팅’이 동시에 성립하도록 제어·기록 체계를 설계하는 데 있습니다. UME는 사람 팔의 3-1-3 구조를 고려한 기구 설계와 쿼지-다이렉트 드라이브 모터(낮은 감속비)로 실시간 저항 전달을 노렸고, 동역학/야코비안 기반의 로버스트한 토크 역변환과 중력·원심·코리올리·마찰 보상을 포함해 투명성을 높였습니다. 아울러 어깨(가상 구면), 팔꿈치(1자유도), 손목(가상 구면)으로 체인을 분해해 각 부분을 독립적으로 리타게팅하며 특이점 근처의 추적 불안정을 줄였습니다.

- **Empirical Impact**: 실험에서는 토크 모달리티와 whole-arm 기록의 효과가 태스크별로 뚜렷하게 나타났습니다(박스 밀기/뒤집기, GPU 집기, 냉장고 음료 회수 등). 토크 정보를 제거한 No-torque 및 엔드이펙터만 쓰는 UMI 대조군은 제약·충돌 상황에서 상태 구분 실패나 충돌로 성패가 크게 떨어졌고, UME는 장시간 모바일 조작에서도 높은 성공률(예: 대다수 태스크에서 0.85~0.95)을 보였습니다. 또한 데이터 수집 처리량이 향상되어(박스 뒤집기에서 토크 미지원 대비 3.3배, 사람 속도의 71%) 향후 ‘순응 정책 학습’의 실용성과 확장성에 의미가 큽니다.



### Short-Horizon Position Accuracy of Single-Track Models: Implications for Motion Planning of Autonomous Vehicles (https://arxiv.org/abs/2606.14216)
Comments:
          Submitted to The International Journal of Automotive Engineering, Official Journal of the Society of Automotive Engineers of Japan, Inc. (JSAE)

- **Prior Approaches**: 자율주행에서 MPC는 차량 모델로 예측 궤적과 제어 입력을 계산하지만, 기존 차량 동역학 연구는 위치(정확한 좌표) 오차를 체계적으로 실측 대비 평가하는 데 상대적으로 소홀했다. 단일트랙(바이시클) 모델은 쓸모가 크지만(계산 효율), 무슬립 가정(운동학)·선형 타이어·비선형 타이어 같은 변형들이 어떤 조건에서 얼마나 위치 오차를 줄이거나 늘리는지에 대한 비교는 제한적이었다.

- **Core Contribution**: 이 논문은 단일트랙 차량 모델 3종(운동학, 선형 동역학, 비선형 Magic Formula + 준정적 종방향 하중이전)을 동일한 고정 단기 예측 가정(T_pred=5초) 아래에서 실제 차량 실측(OxTS RT3000 v3)과 직접 비교한다. 목표는 ‘항상 가장 좋은 모델’이 아니라, 모델 복잡도·파라미터 품질·위치 정확도 사이의 트레이드오프를 MPC 선택에 필요한 인사이트로 제공하는 것이다.

- **Technical Challenges**: 핵심 기술적 난제는 실측 기반으로 각 모델의 파라미터를 신뢰성 있게 식별하고, 오차를 공정하게 분해해 비교하는 것이다. 이를 위해 계측 차량에서 질량/CG/스티어링비/관성·코너링강성·MF 파라미터를 별도 실험으로 추정하고, OxTS 센서 위치를 CG 기준으로 변환하며, 시간 영역에서 종·횡 오차를 평균·최대·RMSE로 정량화했다.

- **Empirical Impact**: 실험 결과 단기 예측에서 동역학 모델 2종(선형/비선형)이 운동학 모델보다 종·횡 위치 오차를 대부분 시나리오에서 2~6배 개선했으며, 특히 고횡가속에서는 운동학의 understeer(미끄림 미반영) 한계가 명확히 드러났다. 반면 선형 동역학과 비선형(Magic Formula)은 대다수 시나리오에서 cm 수준 차이로 비슷했는데, 이는 슬립이 선형 영역에 머무르고 하중이전 효과가 작았기 때문이다. 고난도 시나리오(S4)에서는 비선형이 항상 더 낫지 않았고(선형이 더 낮은 오차), MF 파라미터 식별의 운영조건·마찰 가정과 스티어링 컴플라이언스/조향 동역학 미모델링이 정확도에 큰 영향을 줄 수 있음을 보여준다. 이 때문에 MPC용 모델 선택 시 ‘복잡도 증가 = 항상 이득’이 아니라, 목표 운용영역에서의 타이어·하중·조향 조건을 얼마나 정확히 커버하는지가 실증적으로 중요하다는 메시지를 남긴다.



### Robustness without Wrinkles: Parallel Simulation and Robust MPC for Certified Deformable Manipulation (https://arxiv.org/abs/2606.14188)
- **Prior Approaches**: 변형 물체(로프·클로스) 제어는 고차원 상태, 복잡한 비선형 동역학, 그리고 언제/어디서 접촉할지의 결정 때문에 어렵다. 기존 방법은 기하학적 단순화로 계획을 가능하게 하지만 결과가 개방고리로 끝나 견고성이 떨어지고, 학습 기반 방식은 강건한 안전 보장 없이 느린 학습이 필요하며, 접촉을 미분가능하게 만들면 그라디언트 불연속으로 접촉 모드 탐색이 막히는 문제가 있었다.
또한 출력 피드백에서의 안전·강건 MPC는 상태공간이 큰 변형 물체에 스케일하기 어렵거나, 계산이 느려 실시간 적용이 제한적이었다.

- **Core Contribution**: CORD-SLS는 접촉을 매끄럽게(smoothing) 처리한 GPU 병렬 미분가능 시뮬레이터를 핵심으로, 그라디언트 기반 계획을 접촉을 통과해 수행할 수 있게 한다. 여기에 GPU 병렬 출력 피드백 강건 MPC를 결합해 모델·센서 불확실성 하에서도 제약을 만족하는 도달 가능성(reachability) 기반 계획을 실시간으로 만든다.
추가로 conformal prediction을 통해 시각 피드백의 불확실성을 캘리브레이션하고, 이를 MPC의 도달 튜브(tube) 폭으로 반영해 높은 확률의 안전 제어를 달성한다.

- **Technical Challenges**: 가장 큰 기술 난제는 ‘접촉의 전환(활성/비활성, 마찰 접촉의 상이한 상태)’이 만들어내는 비연속성과 소실 그라디언트로 인해, 접촉 발견이 그라디언트 최적화에서 막히는 점이다. 논문은 접촉 활성 조건과 보완성(complementarity) 제약에 대해 접촉 스무딩을 적용해 계획용 그라디언트를 정보성 있게 만들고, 실제 실행(전개 롤아웃)은 비스무스(비부드러운) 동역학을 사용해 물리 충실도를 유지한다.
또한 출력 피드백 SLS의 계산 비용을 줄이기 위해 관측기(옵저버) 재귀를 GPU 병렬 prefix scan 형태의 연산으로 재구성해, 장시간 예측 구간에서도 밀리초급 합성을 목표로 한다.

- **Empirical Impact**: 시뮬레이션과 실제 하드웨어에서 로프·클로스의 고차원·접촉 다발 과제(장애물 회피, 라우팅, 폴딩, 평탄화 등)를 평가하며, 전반적으로 기준선 대비 안전성·속도·과제 성공률에서 우수함을 보인다.
특히 계획이 밀리초(ms) 단위로 동작해, 접촉이 빈번히 발생하는 조작에서도 실시간 강건 제어가 가능함을 실증한다.
또한 미분가능 시뮬레이터를 사용한 model-based reinforcement learning은 analytical policy gradients 덕분에 샘플 효율을 높여 신경 조작 정책 학습에도 의미 있는 가속 효과를 보인다.



### GAIT: Legged Robot Proprioceptive State Estimation with Attention over Inertial-Leg Tokens (https://arxiv.org/abs/2606.14160)
- **Prior Approaches**: 다리 로봇의 자체항법(프로프리오셉티브)만으로 상태를 추정할 때 기존 방법은 주로 IMU와 관절/다리 운동학을 결합하되, ‘발이 접지한 다리는 고정되어 있다’는 정적 접지 가정을 포함한 contact-aided 프레임워크에 의존했다. 이 가정이 미끄러짐이나 자세 변화로 깨지면 수렴과 일관성이 크게 저하되고, 모델 기반은 접지 추정이나 휴리스틱 파라미터 튜닝이 필요했다. 최근 학습 기반 추정기는 이를 완화하려 했지만 시뮬레이션 정책/데이터 분포에 강하게 맞춰져, 학습 밖의 보행(예: bound, pace, pronk)이나 시뮬-실 불일치·미모델링 지형에서 성능이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 관성-다리(inertial–leg, IL) 토큰화와 어텐션을 결합한 ‘주의 기반 프로프리오셉티브 상태 추정기’를 제안한다. IL 토큰화는 IMU(관성) 입력과 다리별 운동학 입력을 개별 토큰으로 분리해, 현재 접지 조건에 따라 각 측정의 신뢰도를 상대적으로 재가중하도록 유도한다. 또한 네트워크가 예측한 몸체 선속도와 그 불확실성을 IEKF의 가짜 측정(pseudo-measurement)으로 사용하면서, 별도의 명시적 접지 추정기나 정적 접지 기반의 측정 업데이트 가정 없이도 robust하게 동작하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 ‘접지 상태에 따라 다리 운동학 측정의 신뢰도가 달라지지만, 이를 학습 데이터 분포 밖에서도 안정적으로 반영’하는 구조를 만드는 것이다. 기존처럼 입력을 단일 벡터로 합치면 네트워크가 접지 패턴에 암묵적으로 과적합해 분포 이동 시 성능이 붕괴하기 쉽다. 논문은 IL 토큰화로 다리별 구조를 명시하고, Perceiver IO 형태의 encoder cross-attention으로 각 토큰의 상대적 중요도를 현재 상태에서 효율적으로 학습하게 했으며, 몸체 선속도뿐 아니라 불확실도도 함께 예측해 IEKF에서 가중치(측정 공분산)를 동적으로 반영한다.

- **Empirical Impact**: Unitree Go1에서 시뮬레이션 trot 데이터만 학습한 뒤, 실제 실험에서 학습에 없던 지형(예: debris terrain)과 학습에 포함되지 않은 보행(bound/pace/pronk 포함)을 평가했다. 결과적으로 제안 방법은 미지 보행 패턴에서 기존 학습 기반 추정기 대비 더 낮은 추정 오차를 보였고, 접지 보조(model-based) 방법보다도 개선되는 경우가 확인됐다. 또한 네트워크 단독의 몸체 선속도 예측이 미모델링 접촉 조건과 지형 변화에서 더 안정적임을 보이며, 제안 구조가 분포 이동에 강한 inductive bias를 제공한다는 점을 실증적으로 뒷받침한다.



### A Modular Dual-Arm Apple Harvesting Robot with Enhanced Field Performanc (https://arxiv.org/abs/2606.14089)
- **Prior Approaches**: 기존 사과 수확 로봇은 단일 팔의 순차 작업에 의존해 처리량(throughput)이 사이클 시간 개선만으로는 한계가 있었다. 또한 RGB-D 기반 과실 국소화는 야외 조명 변화와 부분 가림(occlusion) 때문에 2D 탐지는 가능해도 3D 깊이 신뢰도가 떨어져 병목이 지속됐다. 다중 팔 연구도 제안은 있었지만, 공유 자원(진공)과 타이밍을 고려한 체계적 조정 및 고속 듀얼암 안전 보장은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 한 그루의 상·하부 과실을 동시에 노릴 수 있도록 ‘수직 적층(vertically stacked) 듀얼암 모듈’을 제안해, 정지 지점 이동을 다중 나무 횡이동에서 단일 나무 정지로 단순화한다. 더불어 Grounding-DINO와 EfficientViT-SAM을 잇는 기반 모델 기반 인식부터, 고속·안전 궤적 제어와 진공 공유를 위한 시간 논리 기반 듀얼암 조정까지 5가지 핵심 개선을 한 시스템으로 통합한다. 결과적으로 플랫폼 설계(하드웨어)와 소프트웨어(인식·제어·조정)가 같이 상호 보완되도록 구성한 것이 핵심 기여다.

- **Technical Challenges**: 주요 기술 난제는 (1) 야외에서 강한 직사광·반사로 인한 관측 열화, (2) 두 팔이 동시에 움직이면서 시야를 가리는 가림 문제, (3) 고속으로 움직이는 듀얼암에서 관절 안전성과 공유 진공 자원 충돌을 동시에 보장하는 것이다. 저자들은 선스크린 모듈과 시야 관리(view-management)로 조명·가림을 완화하고, 7차(7th-order) 저크 제한 궤적에 Control Barrier Function(CBF) 안전 필터를 결합해 빠르면서도 안전한 동작을 만들었다. 또한 선형 스윕 수확(접근 버퍼 10cm, 회전 분리)과 시간 논리(temporal-logic) 사양 및 비동기 비전-팔 스케줄링으로, 두 팔이 공유 진공을 충돌 없이 효율적으로 쓰도록 구현했다.

- **Empirical Impact**: 현장 검증은 2025 수확 시즌 동안 워싱턴의 상용 과수원 2곳, 서로 다른 품종·나무 구조에서 수행됐고 1738개 팔 사이클 기록을 바탕으로 시도당 성공률 80.0%, 팔당 평균 사이클 시간 7.53초를 달성했다. 품질 평가에서도 로봇 수확 과실의 91.2%가 USDA 최고 등급(Extra Fancy)을 유지했으며, 멍(bruse) 발생률은 2.4~4.9% 범위였다. 추가로 사이클 시간 단축과 무거운 잎(heavy foliage) 가림 상황에서의 처리 개선이 이뤄지면 상용 수확 적용 가능성이 더 커질 것으로 전망된다.



### Self-Improving VLA Policies: Selected Diffusion Noise for Spurious-Robust Action Smoothing (https://arxiv.org/abs/2606.14084)
- **Prior Approaches**: 확산 기반 Vison-Language-Action(VLA) 정책은 다양한 로봇 조작에서 강한 일반화를 보이지만, 보이지 않는 물체나 교란 관측에서도 배경 단서에 집착하는 “환각”과 물리적으로 불안정한 “jerky(급격·진동성)” 행동에 취약하다. 기존에는 CFG 같은 단계별 가이던스, best-of-N 샘플링 후 외부 평가/후처리, 혹은 PCD처럼 원본-마스킹 관측 대비를 쓰는 방식이 주로 활용됐으나, 외부 평가 의존이나 비용(예: inpainting), 그리고 매개변수·모듈 추가로 바로 배포하기 어려운 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 Selected Diffusion Noise(SDN)라는 학습 없는(test-time training-free) 방법을 제안한다. 확산 모델의 초기 잡음(noise)을 “조절 가능한 자유도”로 보고, 원본 관측과 물체 마스킹 관측 사이에서 환각 경향이 커지는 잡음 씨앗을 걸러내면서 동시에 더 매끄러운(action smoothness) 궤적을 선택한다. 모델 파라미터 변경이나 보조 네트워크/외부 평가기 없이도, 고정된 VLA 백본을 플러그앤플레이로 개선하도록 설계했다.

- **Technical Challenges**: 핵심 기술적 과제는 두 가지다: (1) 단순 확률이 아니라 “물체에 기반(grounded)된 의미론”을 잡음 씨앗 단위로 진단하고, (2) 목표 도달만이 아니라 로봇 실행 관점의 물리적 매끄러움을 함께 보장하는 것이다. SDN은 각 추론 단계에서 여러 잡음 씨앗으로 후보 궤적을 만들고, kNN 기반 밀도비 추정으로 원본-마스킹 관측 간 대비가 큰(=환각에 덜 민감한) 후보를 1차로 고른 뒤, jerk RMS 같은 동역학 안정성 지표를 최소화하도록 2차로 재선택한다. 또한 마스킹은 inpainting 없이 픽셀을 0으로 가리는 방식으로 처리해 비용을 낮춘다.

- **Empirical Impact**: Google Robot과 Widow-X 시뮬레이션, 그리고 ALOHA 기반의 두 실세계 데이터셋에서 π0와 Groot 계열 VLA 정책 전반에 걸쳐 일관된 성능 향상을 보인다. 성공률은 시뮬레이션에서 평균 +8%, 실세계에서 평균 +10% 개선됐고, 선택된 행동은 더 부드럽고 안정적인 형태로 나타났다. 특히 CFG·PCD 같은 대안이 일부 과제에서 성능을 해칠 수 있는 반면, SDN은 벤치마크 전반에서 “do-no-harm” 성격(비음수 이득)을 보여 테스트타임 보정으로서의 신뢰성을 강조한다.



### The N2D Haptic Glove: A Multi-Finger Glove for 2D Directional Force Feedback for Contact Rich Manipulation (https://arxiv.org/abs/2606.14083)
- **Prior Approaches**: 기존 햅틱 장갑은 주로 진동 또는 단일 축 힘 저항(1-DOF)으로 접촉을 표현해 손가락 힘의 ‘방향’이 불분명했다. 멀티핑거 장갑도 다수는 절단(피부) 자극이나 단일 축 저항 위주라, 축 방향(axial)과 횡 방향(transverse) 상호작용을 모두 정밀하게 제어하기 어렵다는 한계가 있었다. 그 결과 사용자는 시각 정보에 의존해 과압(over-pressing)이나 제어 변동이 커지고, 로봇 원격조작의 정밀 조작 성능이 떨어지곤 했다.

- **Core Contribution**: 본 논문은 손가락 끝에 평면(planar) 형태의 2차원 방향성 힘(flexion-extension 기반)을 다수 손가락에 동시에 렌더링하는 N2D Haptic Glove를 제안한다. 각 손가락당 활성 자유도를 2개로 확장해, 특히 축 방향 힘이 중요한 페이퍼/버튼 누르기·탐침(probing) 같은 작업에서 방향성 피드백을 직접 제공하는 것이 핵심이다. 또한 구동·제어 구조를 투명성(transparency) 중심으로 설계해, 힘 피드백이 기계적 감각에 의해 흐려지지 않도록 했다.

- **Technical Challenges**: 방향성 힘을 손가락 끝에서 안정적으로 만들려면, 손가락 기구학(평면 2-DOF)과 힘-토크 전달을 정확히 결합해 제어해야 한다. 논문은 capstan-drive 기반 저마찰·무백래시 전달로 기계적 히스테리시스와 백래시를 줄이고, 자코비안 전치(Jacobian transpose) 매핑으로 원하는 힘의 평면 성분만 효과적으로 렌더링하는 방식을 사용한다. 벤치탑에서는 전압-토크 매핑을 방향별 정교 캘리브레이션하고, 힘 센서로 원-실측 궤적을 검증해 잔차가 데드존·포화·마찰 같은 비모델 항에서 주로 발생함을 확인했다.

- **Empirical Impact**: 실측 검증에서 지령한 평면 힘 궤적의 오차(RMSE)가 x축 0.032N, z축 0.014N 수준으로 방향 성분까지 잘 추종됨을 보였다. 원격조작 사용자 연구에서는 시각만 쓰거나 단일 축(1D) 저항과 비교해, 축 방향 탐침/누르기(pressing)에서 접촉 힘 오차가 유의하게 감소하고, 시도 간 일관성도 좋아졌다. 특히 100g 목표에서 2D 피드백이 1D·무햅틱 대비 성능을 크게 개선했으며, NASA-TLX 설문에서도 제어감은 높고 정신적 부담은 줄어드는 경향이 관찰되어 방향성 햅틱의 실사용 가치가 확인됐다.



### Development of a 3 in Sewer Pipe Inspection Robot with an Articulated Differential Mechanism using X-shaped Linkages (https://arxiv.org/abs/2606.14070)
Comments:
          The 23rd International Conference on Ubiquitous Robots (UR 2026), 15-18 July, Osaka Ibaraki Campus, Ritsumeikan University, Ibaraki, Osaka, Japan

- **Prior Approaches**: 노후된 하수관로(예: 일본에서 50년 이상 경과 구간의 증가)로 인해 협소 내경 관로 점검 수요가 커졌지만, 내경 100 mm 이하에서는 주행·탐색이 어렵다. 기존 방식으로는 푸시인 카메라가 기본이나 케이블 굽힘 강성 때문에 다중 굴곡 통과가 힘들고, 페리설스틱(pneumatic) 기반 로봇은 속도가 매우 느리며 대형·중량 압축기가 필요해 현장 제약이 크다. 또한 1-in 급 로봇은 행성기어 기반 구조로 바퀴가 내벽을 지속적으로 강하게 누르는데, 급박한 사고 시 후퇴·대피가 어려울 수 있고 기구가 복잡하다.

- **Core Contribution**: 이 논문은 기존 Xbot-1(와이어 구동 3-in 점검 로봇, 비상 대피용 병렬 탄성 구동 포함)을 개선한 Xbot-2를 제안한다. Xbot-2는 추진 유닛을 단순히 하나가 아니라 연결해 차동(differential) 원리를 적용한 관절형 메커니즘으로 확장 구동을 안정화함으로써 장거리 견인력과 실효 사거리 문제를 완화한다. 더 나아가 관 조인트 같은 단차 구간을 모터 전류만으로 감지하고 와이어 장력을 줄여 통과시키는 무센서형 제어 절차도 함께 제안한다.

- **Technical Challenges**: 첫 번째 난제는 추진 유닛이 1개인 Xbot-1에서 발생한 낮은 견인력과, 단차(관 조인트) 통과 시 걸림 현상이었다. Xbot-2에서는 X자 링크(X-shaped links) 기반 와이어 구동으로 로봇 외경을 관 반지름 방향으로 확장하며, 여러 추진 유닛을 연결해 관 지름 변화에도 접촉 변형이 ‘차동’처럼 분산되도록 설계해 단순 결합만으로도 견인력을 끌어올렸다. 두 번째 난제는 단차 접촉을 인코더·IMU 같은 추가 센서 없이 감지해야 하는 점이었는데, 구동 바퀴 모터 전류의 임계값(실험적으로 2.5 A) 초과를 접촉 신호로 삼아 와이어 감김 릴을 150도 되감기(장력 완화)→짧은 후진→단차에 의해 링크가 수축되도록 하는 순차 제어로 해결했다.

- **Empirical Impact**: 실험에서는 Xbot-1 대비 Xbot-2의 견인력이 유닛 연결 수에 따라 약 1.5배(2개 유닛)에서 약 2.0배(3개 유닛)까지 증가함을 전류 파형 비교(모터에 인가한 전류 기반)로 확인했다. 관 조인트 단차 구간에 대해서는 전류 임계 기반 알고리즘이 접촉 후 정체(모터 스톨) 직후 장력 완화와 후진을 유도해 걸림 없이 통과로 이어짐을 실제 로봇·관 모사 실험에서 검증했다. 특히 모터 전류가 약 2.9 A 수준으로 상승한 시점 이후 후진과 와이어 장력 역전(150도)이 이루어진 뒤, 다중 추진 유닛(3개)이 조인트를 매끄럽게 통과했으며, 향후 장거리 실제 배관 점검과 비상 대피 성능 검증 계획도 제시했다.



### Semidefinite Relaxations for Collision-Free Motion Planning (https://arxiv.org/abs/2606.14063)
- **Prior Approaches**: 충돌 회피 모션 플래닝은 장애물 회피 제약이 비볼록이라 계산이 어렵고, 비용(에너지/스냅)·연속성·경계조건까지 포함하면 더 복잡해진다. 기존에는 NLP 기반 궤적 최적화가 널리 쓰이지만 초기값 민감성과 가변적인 수렴 시간을 겪기 쉽고, 자유공간을 미리 convex decomposition해 우회하는 방식은 분해 자체가 또 하나의 어려운 문제다. 최근에는 샘플링 기반과 로컬 최적화를 결합하거나(steering), SOS/모멘트 기반 SDP 완화를 일부 모션 플래닝에 적용해 “대체로 tight”함을 관찰하는 연구가 늘었지만, 충돌 회피에서 완화가 왜/언제 정확해지는지에 대한 이론은 부족했다.

- **Core Contribution**: 이 논문은 점 로봇이 구형 장애물을 피해 시작점-목표점으로 이동하는 문제를 다항 곡선의 비볼록 최적화로 정확히 모델링한 뒤, 자연스러운 반정정부완화(semidefinite relaxation)를 제안한다. 핵심 기여는 두 가지로, 첫째로 볼록 완화가 “글로벌 최적성”을 보장하는 더 높은 차원 공간에서의 관련 모션 플래닝 문제를 푸는 것과 동치임을 기하학적으로 해석한다. 둘째로, 대칭(symmetry) 감소를 통해 SDP 크기가 다항 차수에 대해 선형으로만 커지고 주변 공간 차원에는 의존하지 않게 만들어 완화의 실용성을 높인다.

- **Technical Challenges**: 가장 큰 기술적 도전은 장애물 회피가 연속 구간 [0,1] 전 구간에 대해 성립해야 하는 무한 제약(semi-infinite)을 포함한다는 점이다. 저자들은 1변수 다항식의 음이 아님(nonnegativity)을 SOS(제곱합)로 정확히 LMI로 바꾸는 고전 정리를 적용해, 이산화 없이도 “구 바깥” 조건을 궤적 계수에 대한 PSD 제약으로 인증한다. 이어서 LMI가 궤적 계수에 대해 비볼록이라 Shor류 완화가 필요하지만, 여기서 대칭성을 활용해 PSD 콘 크기를 크게 줄이는 1차 완화로 계산량을 제어한다.

- **Empirical Impact**: 실험에서는 제안 완화가 SNOPT/Ipopt로 직접 비선형 프로그래밍을 푸는 전사(transcription)보다 10~100배 빠르고, 해 시간 변동성이 훨씬 낮으며, 원문 문제에 대해 신뢰성 있게 국소 최적 경로를 찾는다고 보고한다. 또한 RRT 플래너의 convex steering 함수로 쓰면 복잡한 환경에서도 $C^4$ 연속성을 갖는 minimum-snap 쿼드로터 궤적을 효과적으로 생성한다. 무엇보다 이 연구는 “충돌 회피 모션 플래닝에서 SDP 완화의 tightness와 완화 정도”를 이론적으로 처음 분석한 사례로, 이후 관련 완화 적용의 설계 기준을 제시한다.



### ReactSim-Bench: Benchmarking Reactive Behavior World Model Simulation in Autonomous Driving (https://arxiv.org/abs/2606.14058)
- **Prior Approaches**: 기존 행동 시뮬레이션 벤치마크는 주로 전체 장면의 사실성(현실감)을 로그와의 유사도나 오픈루프 예측 지표로 평가했다. 또한 대부분 AV와 주변 에이전트를 모델이 함께 제어해, 테스트 시 AV가 로그에서 벗어날 때 주변이 어떻게 반응해야 하는 ‘반응 능력’은 직접 측정되지 않았다.

- **Core Contribution**: 이 논문은 ReactSim-Bench를 제안해 행동 world model 시뮬레이션의 반응 능력을 정면으로 평가한다. 핵심은 AV 제어와 주변 에이전트 시뮬레이션을 분리하고, 로그와 다른 AV 행동을 독립 입력으로 넣어 주변 차량이 그 입력에 맞게 피드백하는지 측정하는 반응 시뮬레이션 프로토콜이다.

- **Technical Challenges**: 반응 능력을 평가하려면 로그에서 충분히 벗어나면서도 충돌을 유도하는 ‘반응 압력’을 주고, 동시에 입력 AV 행동의 운동학적 실현 가능성과 지도 제약을 만족해야 한다. 이를 위해 AV planner로 후보 행동을 만들고 규칙 기반 필터링과 수동 검증으로 유효한 편향 행동만 선별한 뒤, 충돌/지도 정합/운동학적 타당성 지표로 반응의 안전성과 규칙 준수를 평가한다.

- **Empirical Impact**: nuPlan 기반으로 2,636개 시나리오(종방향·횡방향·방향 편향 3분류)를 구성해 Transformer, diffusion, next-token-prediction 계열 최신 모델들을 체계적으로 비교했다. 실험 결과, 현실감 점수가 높다고 반응 지표가 항상 좋아지진 않았고 로그 피팅 중심 접근은 AV가 로그에서 벗어날 때 A-AV 충돌이 크게 증가했다; 또한 재계획 주기가 1Hz 이상에서 대체로 반응 성능이 좋아지되 모델에 따라 최적점이 달라질 수 있음을 보였다.



### From Attacks to Curricula: Learnability-Guided Adversarial Training for Safe Autonomous Driving (https://arxiv.org/abs/2606.14032)
- **Prior Approaches**: 기존 닫힌고리 적대적 학습은 드물고 위험한 시나리오를 만들어 정책에 충돌 유발 상호작용을 제공하는 방식으로 발전해 왔다. 그러나 대부분은 생성 목표가 “충돌 확률 최대화”에 치우쳐 실제로 회피가 불가능한 극단 상황을 만들어 학습 신호를 약화시키거나, 휴리스틱/정적 샘플링은 정책의 능력 변화와 시나리오 난이도 간 불일치를 키워 표본 효율과 수렴을 떨어뜨린다. 플로우를 개선하려는 시도(규칙/물리 제약, 선호 정렬 등)도 있었지만, 생성 단계에서 “해결 가능성(resolvability)”을 직접 맞추고 샘플링 단계에서 “현재 정책 역량”을 예측·반영하는 결합은 부족했다.

- **Core Contribution**: AlignADV는 적대적 시나리오를 공격용 위험물로만 보지 않고, ‘학습 가능성’에 맞춘 커리큘럼으로 전환하는 프레임워크를 제안한다. 핵심은 (1) 해결 가능성이 보장되는 범위 안에서 위험도는 유지한 채 시나리오를 생성하고, (2) 현재 업데이트된 정책이 해당 시나리오를 실제로 해결할 성공확률을 시뮬레이션 없이 예측해 샘플링 가중치를 동적으로 맞춘다는 점이다. 결과적으로 공격 지향 생성에서 학습 지향 정책 개선으로 관점을 전환한다.

- **Technical Challenges**: 첫째, 충돌을 유발하는 생성 목표만 최적화하면 불가해(unsolvable) 시나리오가 섞여 정책이 회피 위주로 붕괴하거나 학습 신호가 사라지는 문제가 있다. 논문은 생성기를 ‘해결 가능성 정렬’ 관점으로 재정의하고, DPO(Direct Preference Optimization)로 생성 분포를 해결 가능한 쪽으로 재형성해 “치명적이되 풀 수 있는” 시나리오를 만들도록 한다. 둘째, 정책 능력은 학습 중 비선형으로 변하므로 과거 상호작용 기반의 샘플링은 타이밍이 늦는다; 이를 해결하기 위해 행동 지문(behavioral fingerprint)으로 정책의 고유한 동특성을 요약하고, 시나리오를 넣었을 때 성공확률을 추정하는 멀티모달 역량 예측 모델로 미래 성능을 ‘선제적으로’ 추정한 뒤 그 예측을 샘플링에 반영한다.

- **Empirical Impact**: Waymo Open Motion Dataset 실험에서 AlignADV는 기준선 대비 학습 단계 수를 최대 40.6% 줄이면서도 최종 성능을 개선했다. 또한 정상 트래픽뿐 아니라 적대적 교통 조건에서도 충돌률을 낮추고 경로 완료율(route completion)을 향상시켜, 해결 가능성 정렬과 역량 정렬 샘플링이 함께 작동함을 보여준다. 전체적으로 ‘해결 가능성’과 ‘현재 역량’을 동시에 고려하는 커리큘럼 설계가 안전한 자동주행 학습의 효율과 안정성을 높일 수 있다는 방향성을 제시한다.



### SplatlessDF: Continuous Distance Field Mapping with Non-Splatting Gaussians (https://arxiv.org/abs/2606.13990)
- **Prior Approaches**: 로봇 분야의 연속 거리장(DF)은 점유격자, TSDF, Euclidean distance transform 같은 명시적 공간 구조 기반 표현이 여전히 핵심이다. 최근에는 신경 암시적(DeepSDF, iSDF 등)·복셀 기반·Gaussian process 기반 방법이 연속 DF를 학습하지만, 대부분은 GS(가우시안 스플래팅) 렌더링과 분리돼 있어 ‘가우시안의 역할’이 주로 렌더링에 머문다. 한편 PINGS나 GS-SDF처럼 DF와 GS를 함께 쓰는 시도도 있으나, DF가 가우시안으로 직접 파라미터화되지는 않아 공간 질의용 표현으로의 연결성이 약했다.

- **Core Contribution**: 이 논문은 연속 DF를 중심에 두고, 비등방성 가우시안 원소를 공간(좌표) 관점에서 직접 DF의 파라미터로 사용한다. 제안하는 SplatlessDF는 가우시안을 이미지 평면에 투영·스플래팅(rasterization/alpha compositing)하지 않고, 임의의 공간 좌표에서 거리와 그래디언트를 미분 가능하게 질의할 수 있도록 설계됐다. 또한 DF 전용(standalone) 학습과 2D Gaussian splatting(2DGS)와 결합한 joint 학습을 모두 제공해, 로봇 내비게이션에 필요한 거리장 기능과 렌더링 성능을 함께 노린다.

- **Technical Challenges**: 핵심 과제는 (1) 가우시안 기반 표현을 렌더링용이 아니라 DF 자체의 ‘연속 거리 함수’로 바꾸면서, (2) 거리 및 그래디언트 품질을 안정적으로 학습하는 것이다. SplatlessDF는 각 질의 점에서 가우시안들의 가중합으로 거리값을 계산하고, 표면 점대(거리는 0에 수렴)·비표면 점대(양의 거리)에 대한 거리 감독을 이용해 미분 가능한 DF를 최적화한다. joint 설정에서는 DF와 2DGS를 직접 합치지 않고, 신뢰할 수 있는 2DGS 원소(원점/디스크 점)를 DF의 저거리(near zero-distance) 구조에 맞추는 중심·디스크 일치 손실로 기하를 결합해 두 목표를 동시에 학습한다.

- **Empirical Impact**: 실험은 2D·3D 모두에서 DF 정확도와 그래디언트 품질, 그리고 로봇용 내비게이션 유틸리티까지 평가하며, SplatlessDF의 DF 전용 모델이 최우수 또는 상위권 RMSE를 보인다. joint 모델은 렌더링 품질에서는 기본 2DGS 대비 향상되면서도, DF 측면에서도 그래디언트 방향 일치가 가장 좋거나 경쟁력 있게 유지된다. 또한 CHOMP 같은 경로계획에서 필요한 거리·그래디언트 질의의 신뢰성을 바탕으로 내비게이션 성능을 개선함을 보여, GS 스타일 표현이 표면/렌더링뿐 아니라 로봇 내비게이션용 지도(연속 거리장)로도 확장될 잠재력을 입증한다.



### An Attention-based Model for Robust Forecasting with Missing Modality (https://arxiv.org/abs/2606.13970)
Comments:
          Work originally done in 2023

- **Prior Approaches**: 기존 멀티모달 융합 모델들은 대체로 학습과 추론 모두에서 모든 모달리티가 주어진다는 가정에 의존합니다. 결손 모달리티를 다루는 연구도 있으나, 이미지-텍스트처럼 모달리티가 2개인 경우가 많고 로봇처럼 3개 이상 모달리티가 있는 상황에서의 견고성은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 학습과 추론 단계 모두에서 모달리티가 누락될 수 있는 상황을 처리하도록 설계된 멀티모달 예측 모델을 제안합니다. Conditional Variational Auto-Encoder(CVAE)로 조건부 생성분포를 학습하고, 모든 모달리티로 만든 표현에 최대한 근접한 ‘통합 고정 차원 표현’을 결손 상황에서도 유지하도록 합니다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘입력이 모달리티 개수/구성에 따라 달라져도’ 동일한 표현 공간과 분포를 안정적으로 학습하는 것입니다. 이를 위해 트랜스포머 기반 어텐션에서 고정 크기 질의(learnable embedding)를 사용해 결손 시에도 출력 차원이 흔들리지 않게 하고, 학습 중 모달리티를 랜덤 마스킹한 뒤 결손 모달리티가 있을 때의 분포가 완전 모달리티일 때의 분포와 가까워지도록 KL 기반 누락 모달리티 손실을 추가합니다.

- **Empirical Impact**: 인간 궤적 예측과 로봇 조작 예측까지 총 5개 멀티모달 데이터셋에서 실험했으며, 결손 모달리티 상황에서도 기준선 대비 우수한 성능을 보였습니다. 특히 모달리티를 제거한 테스트에서도 추가 재학습 없이 동작하며, 결손에 따른 성능 저하 폭이 기존 방식보다 작게 나타나 로봇 지각·의사결정 분야에서의 실용성을 강화한다는 점이 입증됩니다.



### Learning Dynamic Swing-Up of an Inverted Pendulum using Remote Magnetic Actuation (https://arxiv.org/abs/2606.13915)
- **Prior Approaches**: 대부분의 전자기 항법 시스템(eMNS) 연구는 전자기장을 준정적(quasi-static)으로 보고, 정적 또는 평형 근처에서의 제어에 집중해 왔다. 최근에는 동역학 기반 접근이 궤적 추종과 교란 억제 성능을 높일 수 있음을 보였지만, 평형에서 한참 떨어진 영역의 고난도 동적 궤적(예: 스윙업) 추적은 여전히 공백으로 남아 있었다. 또한 스윙업 제어는 모델 불일치 보정이 핵심인데, LQR 같은 피드백만으로는 전자기 작동의 복잡한 현상 때문에 한계가 드러난다.

- **Core Contribution**: 이 논문은 임상에서 곧바로 사용 가능하다고 언급되는 Navion eMNS로 자력 구동 인버티드 펜듈럼의 ‘첫 스윙업’(0.6초 내 단일 연속 동작)을 시연한다. 실제 기구 자체는 임상적 필요성이 크지 않더라도, 제어 목표를 힘·토크로 설정해 카테터, 가이드와이어 등 다른 자기 구동 장치로의 확장 가능성을 보여준다. 핵심은 전자기 내부 동역학을 반영한 궤적 최적화에 시간가변 LQR 상태 피드백과 반복학습제어(Iterative Learning Control, ILC)를 결합해 목표 달성 사각지대를 메운 점이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 전자기장 모델 캘리브레이션 오차, (2) 관절 및 하드 스톱의 불완전한 모델링, (3) 저수준 전류 구동기의 대역폭 제한으로 인해 ‘요청 토크’와 ‘실제 작동 토크’가 시간적으로 어긋난다는 점이다. 이를 해결하기 위해 논문은 토크 세트포인트를 유도하되 내부 eMNS 작동 대역폭을 1차 동역학으로 포함해 유효한 궤적을 만들고, 피드백(LQR)으로 비반복 교란의 영향을 줄인 뒤 ILC가 이전 시행 데이터를 통해 피드포워드 보정을 점진적으로 학습한다. 그 결과 LQR 단독으로는 스윙업 조건을 충족하지 못하지만, ILC 보정이 6회 이내에 성공적인 스윙업을 가능하게 한다.

- **Empirical Impact**: 실험 후 분석에서는 ILC가 학습한 보정 신호가 고정밀 자기장 모델 캘리브레이션이 예측한 토크 불일치와 매우 유사하게 나타난다. 이는 전자기 작동에서 필연적으로 생기는 불확실성(예: 환자별 생리 운동 패턴, 장 모델 캘리브레이션 오류)을 학습과 적응으로 다룰 수 있다는 가능성을 실증한 것이다. 또한 Navion뿐 아니라 연구용 OctoMag에서도 동일한 접근의 플랫폼 범용성을 보여, 동적 궤적 추종을 eMNS의 실사용 한계 쪽으로 확장하는 데 의미가 있다.



### PhysVLA: Towards Physically-Grounded VLA for Embodied Robotic Manipulation (https://arxiv.org/abs/2606.13886)
Comments:
          9 pages, 5 figures, supplementary material included

- **Prior Approaches**: VLA(vision-language-action) 모델들은 영상과 언어 지시를 로봇 제어 행동으로 바로 매핑하지만, 주로 시연 데이터 적합에 의존해 강체 동역학이나 접촉 제약을 명시적으로 강제하지 않는다. 그래서 접촉이 많은 다단계 조작에서 물리적으로 불일치한 행동이 나올 수 있고, 이를 줄이기 위한 단순한 시간적 스무딩(예: EMA)은 장기 품질을 희생하며 실패를 늘릴 위험이 있다.

- **Core Contribution**: 본 논문은 학습 없이(inference-time, training-free) 임의의 고정 VLA 백본을 감싸 동작을 물리적으로 보정하는 플러그앤플레이 프레임워크 PhysVLA(Physics-VLA)를 제안한다. PhysVLA는 백본 가중치 접근·미세조정 없이도, 시뮬레이터/시스템 상태만 활용해 행동을 단계(approach, grasp, transport, place)별로 정합시키고 필요할 때만 동역학 잔차 기반 보정을 작동시킨다.

- **Technical Challenges**: 핵심 기술 난관은 ‘전체 에피소드에 균일하게 스무딩’하면 접촉 국면의 민감한 반응이 무뎌져 실패가 늘어난다는 점이다. PhysVLA는 (1) 조작 단계를 분해하는 위상(phase) 인지 유한상태기계로 기하 기반 소규모 보정을 적용하고, (2) Euler-Lagrange 잔차가 기준을 넘을 때만 선택적으로 가동되는 selective Euler-Lagrange gate로 미세한 동역학 불일치만 보정해 <1ms 수준의 지연으로 운영 가능하게 했다.

- **Empirical Impact**: LIBERO-Spatial의 Franka Panda 7-DoF 실험에서 PhysVLA는 백본 전반에 걸쳐 성공률을 최대 17%p, 안정성을 최대 19%p까지 끌어올리며 태스크별 성능 퇴행을 보이지 않았다. 또한 Robosuite Lift 크로스-시뮬레이터 스윕에서 궤적 저크(trajectory jerk) 강건성이 최대 10배 개선됐고, 실제 Agilex Piper 로봇 pick-and-place에서도 재학습 없이 성공률이 최대 50%p 향상되어 물리 인지 보정 모듈의 범용성과 전이성을 입증했다.



### Guided Diffusion with Distilled Vision-Language Reliability for Aerial Navigation (https://arxiv.org/abs/2606.13883)
- **Prior Approaches**: 기존 UAV 실내 자율주행은 지각-지도화-계획을 모듈별 파이프라인으로 나눠 처리해, 인터페이스를 지날 때 정보가 손실되고 오차가 누적되며 지연이 커지기 쉽다. 엔드투엔드 확산(difussion) 기반 방법은 관측에서 궤적 분포를 직접 샘플링해 경로의 다중성을 살리지만, 학습 데이터의 깨끗한 조건을 전제로 해 관측이 신뢰 불가할 때를 구분하지 못한다. 그 결과 유리, 거울, 과노출 표면처럼 깊이 센서가 그럴듯하지만 틀린 값을 내는 영역이 그대로 계획에 반영되는 실패 모드가 남는다.

- **Core Contribution**: 이 논문은 ‘신뢰도’를 주행 계획의 핵심 입력으로 편입한 신뢰도 인지(reliability-aware) 확산 플래너를 제안한다. RGB에서 생성한 장면 수준 신뢰도 히트맵을 확산 궤적 생성에 조건으로 넣고, 물리 장애물(깊이 기반)과 신뢰 불가 영역(가상 장애물)을 동일하게 비용에 반영해 안전한 경로를 유도한다. 또한 비전-언어 모델이 제공하는 오픈 보이캐비뷸러리 의미 추론을 경량 네트워크로 증류해, 실시간 예산 내에서 신뢰도 추정을 가능하게 한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘깊이 측정이 틀릴 때’ 그 신호를 저비용으로 안정적으로 감지하고, 샘플링 과정의 지도화/계획 비용과 충돌하지 않게 만드는 것이다. 논문은 AnyTraverse 같은 고비용 교사로 신뢰도 맵을 만들고, 이를 MiT-B0 학생 모델에 학습시켜 실시간 추론이 가능한 신뢰도 히트맵을 얻는다. 그리고 학습 없이(training-free) 2단계 미분가능 ESDF 기반 비용으로 확산 역과정(denoising)을 안내하되, 신뢰도 낮은 영역은 ‘약한’ 가상 장애물로 설정해 물리 장애물 회피와 함께 균형 있게 회피하도록 만든다.

- **Empirical Impact**: 시뮬레이션과 실기 쿼드로터 환경에서, 제안 방식은 비슷한 확산 기반 기준선(NoMaD) 대비 더 안전한 궤적을 만들었다. 특히 장애물 위반률(obstacle-violation rate)이 40.3%에서 9.6%로 크게 감소했고, 통과한 영역의 평균 신뢰도는 0.588에서 0.925로 상승했다. 신뢰도 유도 항을 제거하면 평균 신뢰도가 0.898→0.783 수준으로 붕괴해, 신뢰도 인지가 성능의 결정적 구성요소임을 실험적으로 확인했으며, 증류로 인해 비전-언어 모델 전체 대비 최대 2배 빠른 구성이 가능해 현장 적용성도 높였다.



### AnyGoal: Vision-Language Guided Multi-Agent Exploration for Training-Free Lifelong Navigation (https://arxiv.org/abs/2606.13878)
Comments:
          17 pages, 3 figures

- **Prior Approaches**: GOAT-Bench에서 기존 방식은 (1) 학습 기반 엔드투엔드 정책이 공간 사전지식에 의존해 분포·범주·목표 양식이 바뀌면 취약해지거나, (2) DETIC류의 닫힌 어휘 탐지기에 의존하는 모듈식 파이프라인이 열려 있는 범용 탐지 성능 한계(낮은 recall)로 천장을 맞는 문제가 있습니다. 또 다른 축인 3D-Mem 계열은 3D 스냅샷을 뷰마다 다시 조회해 VLM 호출 비용과 메모리·연산 부담이 커지고, 구조적으로 단일 에이전트에 머뭅니다. 즉, ‘공간 추론’과 ‘의미 추론’을 각각의 취약 구성요소(폐쇄형 탐지기/대규모 3D 스냅샷)로 분해해 신뢰성 손실이 누적되는 한계가 있습니다.

- **Core Contribution**: AnyGoal은 학습 없이 동작하는 다로봇 아키텍처로, VLM을 중심에 두되 결과를 2D Gaussian Bayesian Value Map(BVM)에 불확실성까지 포함해 누적합니다. BVM은 서브태스크 간에 초기화하지 않아 “평생 증거 누적(lifelong evidence accumulation)”을 가능하게 하고, 프론티어(탐색 경계)를 VLM 판단과 BVM 기반 Bayesian UCB를 함께 섞어 우선순위화합니다. 또한 중앙 제어·명시적 메시지 없이, 공유 맵을 읽고 그리디 할당(분리 패널티와 커밋 히스테리시스)만으로 에이전트 협업을 수행합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (a) VLM의 스코어를 바로 계획에 쓰면 오탐이 누적되며, (b) 다로봇 협업에서는 각자 관측을 정합하는 방식이 병목이 된다는 점입니다. AnyGoal은 깊이 기반 depth-cone 마스크로 관측 신뢰도를 반영한 정밀도 가중 정규분포(평균 μ, 분산 σ²) 업데이트를 설계해, 공간적으로 모순되는 VLM 히트를 사전에 역투영 단계에서 배제합니다. 그리고 프론티어 선택은 “VLM-as-judge 소프트맥스 + UCB 탐색 항”을 볼록 결합하고, “is-worth” Yes/No 게이트로 구조적으로 비합리적인 후보를 거르는 방식으로 불확실성·오탐 문제를 제어합니다.

- **Empirical Impact**: GOAT-Bench val_unseen(360 에피소드, 2,669 서브태스크)에서 AnyGoal 듀얼 에이전트는 Subtask SR 52.4%, SPL 12.7%를 기록하며 Modular GOAT 대비 Subtask SR을 +27.5%p 끌어올렸습니다. 더 나아가 단일 에이전트(N=1)도 Subtask SR 41.9%로, 성능 향상이 단순 에이전트 수 증가가 아니라 Gaussian BVM의 누적·불확실성 정량화와 VLM 판정 게이트·UCB 결합 구조에서 온다는 점을 보여줍니다. 또한 4가지 지각(ablation) 실험에서 탐지 recall이 올라갈수록 실패 양상이 ‘탐색 부족’에서 ‘중지(verification) 결정의 오류’로 전환됨을 정량화해, 후속 연구의 병목을 “더 나은 탐지기”가 아니라 “멈춤 시점의 목표 검증”으로 재정의하는 의미가 있습니다.



### ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation (https://arxiv.org/abs/2606.13877)
Comments:
          32 pages, 12 figures, supplementary material included

- **Prior Approaches**: 기존 연구는 잠재 세계 모델(예: JEPA 계열)과 같은 아키텍처/목표 개선에 집중해 왔지만, 접촉이 많은 조작에서 ‘표현(representation) 자체의 어떤 성질’이 장기 계획 안정성에 핵심인지 체계적으로 분리해 보긴 어려웠습니다. 또한 촉각 연구는 인식·상태추정·정책학습 관점에서 많이 다뤄졌으나, 예측형(world model)에서 필요한 시간적으로 연속된 잠재 표현 관점에서의 영향은 상대적으로 덜 규명되었습니다.

- **Core Contribution**: 이 논문은 비전-촉각 세계 모델이 접촉이 많은 조작에서 장기 계획을 안정적으로 지원하려면 어떤 표현 특성이 중요한지 실증적으로 답하는 벤치마크 ContactWorld를 제안합니다. ContactWorld는 삽입·분해·스크루·탐색 등 12개 태스크에서 비전/촉각 조합과 융합 방식, 계획 지평(horizon)을 통일된 평가 프레임워크로 비교하며, 표현 구조와 멀티모달 호환성이 성능을 좌우한다는 결론을 도출합니다.

- **Technical Challenges**: 핵심 기술적 난제는 접촉 전이 과정에서 발생하는 작은 예측 오차가 시간에 따라 누적되며 잠재 롤아웃이 불안정해진다는 점입니다. 저자들은 JEPA 패러다임의 계획지향 잠재 예측 모델 위에 태스크 전반의 비전(손목·정면·포인트클라우드)과 촉각(TacRGB·TacDepth·TacFF)을 넣고, ‘공간 구조(spatial structure)’와 ‘시간 연속성(temporal continuity)’, 그리고 멀티모달 ‘호환성’이 계획 성능에 미치는 영향을 정교한 절제실험으로 분석해 해결합니다.

- **Empirical Impact**: 실험 결과, 공간 구조와 시간 연속성을 함께 보존하는 표현이 가장 강한 장기 계획 성능을 보였으며, 포인트클라우드는 손목뷰 20.7%, 정면뷰 22.0% 대비 평균 성공률을 32.1%로 끌어올렸습니다. 촉각은 무조건 이득이 아니라 호환성이 중요해 TacFF를 포인트클라우드와 결합할 때 전체 최강 성능(36.1%)을 달성했고, 장기 지평에서 촉각의 기여가 더 커지며(누적 오차·접촉 불확실성 완화) 안정성이 향상됨을 보여줍니다.



### Output-Level Regularization Eliminates the Seed Lottery in Single-GPU VLA Fine-Tuning (https://arxiv.org/abs/2606.13856)
Comments:
          10 pages, 8 figures, submitted to CoRL 2026

- **Prior Approaches**: 기존에는 VLA-JEPA류 비전-언어-행동 모델을 단일 GPU에서 파인튜닝할 때, 사전학습 체크포인트를 불러오고 학습하면 된다고 가정해 왔습니다. L2나 EWC 같은 ‘가중치 수준 정규화’는 파라미터 변화량을 제한하지만, 행동 출력이 한쪽으로 붕괴하는 현상(output collapse)은 구조적으로 감지·제어하지 못합니다. 특히 고정된 V-JEPA2 인코더 아래에서는 이 한계가 더 크게 드러납니다.

- **Core Contribution**: 이 논문은 동일 코드·동일 데이터에서 랜덤 시드만 바꿔도 성능이 크게 흔들리는 ‘시드 럭키(Seed lottery)’를 VLA 파인튜닝에서 정식으로 규명합니다. 원인은 행동 예측기가 입력에 무관하게 거의 동일한 출력을 내도록 학습하는 출력 붕괴이며, 이 현상이 Jacobian null-space와 연결된 구조적 조건(동결 인코더)에서 발생함을 제시합니다. 또한 이 붕괴를 “시드가 무엇인지 미리 모르면 실패를 기다려야 하는” 재현성 위기로 확정하고, 실용적인 해결책까지 제안합니다.

- **Technical Challenges**: 핵심 기술 난제는 실패가 학습 중 경고 없이 조용히 발생해, 손실 곡선이나 파라미터 드리프트로는 예측하기 어렵다는 점입니다. 저자들은 가중치 변화량을 억제하는 정규화가 Jacobian 관점에서 출력 변화에 비활성인 방향(null-space)에 예산을 쓰는 ‘구조적 맹점’을 Jacobian 랭크 결손으로 설명하고, 출력 자체를 직접 안정화하는 정규화(출력 수준)를 설계해 이 맹점을 우회합니다. 그 결과 VICReg, Dropout, 학습률 절반(LowLR) 같은 간단한 설정 변경으로 전 카타스트로픽 시드를 제거합니다.

- **Empirical Impact**: LIBERO-Object 등 3개 벤치마크에서 시드를 최대 13개까지 비교한 실험 결과, 기준선은 13개 중 1개 런이 65.2%로 붕괴하며 나머지는 91–94%대를 유지해 29pp 격차가 발생합니다. 반면 VICReg, Dropout, LowLR는 총 21개 붕괴 사례를 0으로 만들고(통계적으로 유의미), 가중치 수준 방법(L2, EWC)은 럭키 현상을 그대로 보존합니다. 단일 RTX 5090 환경에서 “단 하나의 설정만 바꾸면 재현성 실패를 제거”할 수 있다는 점에서, 현업 배포 안정성과 VLA 파인튜닝 표준 관행에 직접적인 영향을 줍니다.



### Efficient Domain-Adaptive Policy Learning via Kernel Representation with Application to Quadrotor Control under Non-Stationary Disturbances (https://arxiv.org/abs/2606.13842)
- **Prior Approaches**: 기존 강건 제어는 최악의 교란을 가정해 보수적일 수 있고, 고전 적응 제어는 알려진 동역학에 대한 매개변수 불확실성 중심이라 교란의 표현력이 제한된다. 학습 기반 제어는 환경 인코더 조건의 정책을 학습하지만, 인코더가 오프라인 이후 고정되면 훈련 환경에서 다루지 못한 교란 변형을 포착하기 어렵다. 온라인 학습 방식도 과거 정보 기반으로 제어를 선택하는 경우가 많아 튜닝 파라미터에 민감하다는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 커널 표현 기반의 효율적 도메인 적응 정책 학습 알고리즘을 제안한다. Random Fourier Features(랜덤 푸리에 피처, RFF)를 사용해 미지의 비정상 교란을 미분 가능 커널 근사로 모델링하고, 오프라인에서 정책을 학습한 뒤 배치된 배포 단계에서 커널 계수와 대역폭을 온라인으로 함께 업데이트한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 시뮬레이션-실세계(sim-to-real) 갭과 시간에 따라 달라지는 복잡한 교란을 충분히 표현하면서, (2) 배포 시 실시간으로 빠르게 적응할 수 있을 만큼 계산이 효율적이어야 한다는 점이다. 논문은 RFF로 커널을 선형화해 계산 효율을 확보하고, 오프라인에서는 무작위로 커널 계수와 대역폭을 샘플링해 교란 프로파일 다양성을 만들며, 미분 가능한 시뮬레이션과 해석적 그래디언트로 정책을 빠르게 최적화한다. 온라인 단계에서는 커널 계수와 대역폭 모두를 온라인 최소제곱 추정으로 갱신해 교란에 맞춰 기저 자체가 적응하도록 설계한다.

- **Empirical Impact**: RTX 4090 GPU 기준 오프라인 학습이 매우 짧은 시간(약 50초 수준) 안에 이뤄지며, 배포 시에는 실시간으로 비정상 환경에 적응한다는 점이 강조된다. Crazyflie 하드웨어를 포함한 고정밀 시뮬레이션과 실험에서 바람, 지면 효과 온/오프 전환, 공중 부유 페이로드(질량 2% 수준) 변화, 공기역학 효과 등 다양한 교란 조건에서 기존 학습 기반·모델 기반 적응 제어 기준선을 일관되게 능가한다. 시뮬레이션에서는 위치 추종 성능이 최대 약 6e-7% 개선, 하드웨어에서는 최대 약 3% 개선을 보이며, 빠른 온라인 적응이 실제 비행에서도 유효함을 실증한다.



### Multi-Agent Embodied Autonomous Driving: From V2X Information Exchange to Shared World Models (https://arxiv.org/abs/2606.13840)
- **Prior Approaches**: 기존 자율주행 연구는 차량 단독 지능에 무게가 실렸지만, 최근에는 인프라·다른 차량과의 협력으로 다중 에이전트 구체적 환경(embodied)에서 불확실성을 다루는 방향으로 이동하고 있다. 선행 접근은 V2X 통신, 협력적 지각, 에이전트 간 인지, 협동 계획, 종단 간 협동 주행, 폐루프 검증을 위한 시뮬레이션/데이터 엔진으로 넓게 분류된다. 다만 평가가 시뮬레이션과 선별된 벤치마크, 오프라인 프로토콜에 치우쳐 있고, 실차·개방 도로에서의 실시간 안전 보장이 충분히 검증되지 않았다.

- **Core Contribution**: 이 논문은 전환 과정을 Shared World Models(공유 세계 모델, SWMs) 관점에서 구조화하며, 서로 교환된 관측이 어떻게 정렬된 상태(aligned state)와 의도 인지 기반 상호작용, 최종 행동 조정으로 이어지는지 정리한다. 즉, 각 에이전트가 유지하는 예측적 교차 에이전트 표현이 무엇을 의미하고, 어떤 구성 요소들이 정합성 문제를 해결해야 하는지 연구 의제를 제시한다. 또한 우선순위로 공유 상태의 검증 가능 유지, 의도·계획 정렬의 강건성, 통신 지연과 배치 제약 하의 안전한 협동 행동을 도출한다.

- **Technical Challenges**: 핵심 난제는 교환 관측을 공통의 정렬 상태로 유지하는 과정에서 발생하는 불일치와, 서로 다른 에이전트의 의도·계획이 엇갈릴 때의 정렬 실패를 막는 것이다. 저자들은 이를 위해 통신·지연·노이즈 환경에서 SWMs가 예측적 표현을 안정적으로 보존하도록 하는 설계와, 계획·행동 단계에서 정렬이 깨졌을 때의 강건한 대응을 강조한다. 더 나아가 개방 교통에서 실시간 안전 보장을 수학적으로(또는 검증 절차로) 확보하는 것이 아직 공백으로 남아, 재현 가능한 검증 체계가 요구된다고 지적한다.

- **Empirical Impact**: 240편이 넘는 문헌을 포함해 380편 이상을 폭넓게 망라하며, 현재까지의 검증이 주로 시뮬레이션·커리큘럼화된 벤치마크·오프라인 설정에 집중되어 있음을 명확히 보여준다. 이 결과는 MAEAD(다중 에이전트 물리적 환경 기반 자율주행)의 향후 연구가 ‘성능’뿐 아니라 ‘공유 상태의 검증 가능성’과 ‘실시간 안전 보장’ 중심으로 재편돼야 함을 독자에게 방향성으로 제공한다. 특히 foundation model 기반 조정이 등장했더라도 개방 도로에서 안전을 확인할 수 있는 근거가 부족하다는 점을 전면에 내세워, 실시간 검증 연구의 중요성을 부각한다.



### FlowMo-WM: A World Model with Object Momentum and Hidden Ambient Drif (https://arxiv.org/abs/2606.13817)
- **Prior Approaches**: 로봇용 월드 모델은 Dreamer/RSSM/TDMPC 계열처럼 관측과 행동으로 미래 잠재 상태를 예측해 계획·제어에 활용해 왔습니다. 하지만 기존 다수의 action-conditioned 평가는 ‘현재 행동이 즉시 지배’하는 환경을 가정해, 관성 모멘텀과 숨은 주변 드리프트(예: 수류·바람)가 이어서 영향을 주는 실환경 문제를 충분히 분리해 다루지 못했습니다.

- **Core Contribution**: FlowMo-WM은 수중 표면 차량처럼 미래 운동이 ‘명령 추력 + 숨은 주변 흐름에 의한 이동’에 의해 결정되는 상황을 겨냥한 end-to-end 시각 월드 모델입니다. 입력은 이미지와 행동 이력뿐이며, 객체 중심의 단기 운동 상태와 주변 드리프트를 요약한 장기 컨텍스트를 분해해 장기 예측을 수행합니다.

- **Technical Challenges**: 핵심 난제는 흐름장/속도/물리 파라미터 같은 특권 정보를 제공하지 않고도, 단기 프레임만으로는 관측 불가능한 모멘텀과 드리프트를 이력으로부터 추론해 누적 오차를 줄이는 것입니다. FlowMo-WM은 (1) 단기 GRU 상태와 (2) 느리게 변하는 장기 컨텍스트 분해를 만들고, ‘0 컨텍스트 잔차(residual) 전이’로 action 기반 기본 동역학과 컨텍스트 기반 드리프트 효과를 분리해 롤아웃 안정성을 확보합니다.

- **Empirical Impact**: 수중 시뮬레이션에서 다양한 숨은 흐름·교란·차량 동역학을 랜덤화한 조건 하에, FlowMo-WM은 비교 대상 action-conditioned 잠재 월드 모델 대비 60스텝 장기 롤아웃 위치 오차를 더 크게 줄이며(예: pos@60에서 19% 내외 개선) 예측 기반 계획 성공률도 가장 높였습니다. 컨텍스트를 0으로 끄거나(제거) 다른 샘플로 섞으면(poshuffle) 예측이 크게 불안정해져, 학습된 장기 컨텍스트가 드리프트 하에서의 안정적 예측에 ‘실제로 기능’함을 실험적으로 입증했습니다.



### $μ_0$: A Scalable 3D Interaction-Trace World Mod (https://arxiv.org/abs/2606.13769)
- **Prior Approaches**: 기존 월드 모델은 주로 픽셀 생성으로 넓은 시각 사전학습을 얻거나, 혹은 행동(액션) 레이블을 직접 예측해 제어에 연결했습니다. 하지만 픽셀 기반은 배경·외관 재구성에 용량이 소모되고 조작에 필요한 접촉/기하 구조를 놓치기 쉬우며, 행동 직접 예측은 임봇(로봇 체형)별 레이블이 희소해 확장성이 제한됩니다.

- **Core Contribution**: 이 논문은 행동 레이블 없이도 전이 가능한 ‘3D 트레이스(3D 궤적) 세계 모델’ μ0를 제안합니다. μ0는 객체/도구/손/접촉 영역 같은 의미 있는 상호작용 지점들의 부드러운 3D 궤적을 예측해, 임봇과 무관한 모션 인터페이스를 형성합니다.

- **Technical Challenges**: 핵심은 대규모 학습을 위한 3D 감독을 어떻게 자동으로 만들고, 불확실한 미래와 가림(occlusion) 속에서도 실행 가능한 궤적을 어떻게 생성하느냐입니다. 이를 위해 TraceExtract로 DINOv2 기반 의미 키포인트를 선택하고 전역 정렬된 3D 트레이스를 만들며 이벤트 단위 계층 언어 캡션을 붙여 학습데이터를 확장했고, μ0는 VLM 백본과 순열-대칭 Trace Expert를 결합해 B-스플라인 제어점에 대한 조건부 디노이징(흐름 매칭)으로 미래 트레이스를 생성합니다.

- **Empirical Impact**: 실험에서 μ0는 2D/3D 트레이스 예측의 ADE·FDE·DTW 전반에서 기존 트레이스 모델과 tokenized VLM 계열을 능가합니다. 또한 RoboCasa365 시뮬레이션과 UR3 실로봇 조작에서, 동작 레이블 없이 비디오만 사전학습한 뒤 행동 전문가를 붙인 방식이 액션 감독 VLA 모델과 경쟁하거나 더 높은 성공률을 보이며, 3D 트레이스 표현이 크로스-임봇 조작에 실용적으로 전이됨을 입증했습니다.



### Scalable Dynamic Tactile Sensing Enabled by Passive and Flexible Acoustic Waveguides (https://arxiv.org/abs/2606.13746)
Comments:
          40 pages, 6 figures

- **Prior Approaches**: 기존의 인공 동적 촉각 센서는 감도·견고성·컴플라이언스(순응성)를 동시에 만족해야 하지만, 대면적 어레이로 확장할 때 구조 유연성과 성능 간의 트레이드오프가 커졌다. 또한 케이블·배선 복잡성과 비용이 증가해 배치 규모를 제한하는 문제가 있었다.

- **Core Contribution**: 이 논문은 딥 서브파장(dip sub-wavelength) 음향 도파관을 활용한 패시브 분산(passive distributed) 패러다임을 제안해, 구조의 큰 유연성 변화에도 음향 전달 성능이 거의 불변이 되도록 분리(decouple)했다. 탄성 막으로 캡을 씌운 헬름홀츠 공진기와 스프링 보강 마이크로튜브를 폐쇄형 네트워크로 연결해, 거시적 굽힘에서도 음향 전달이 유지되는 구조를 구현했다.

- **Technical Challenges**: 핵심 과제는 대면적에서 배선 없이도 각 지점의 자극을 정확히 국소화하고 저주파 신호까지 복원하는 신호처리 파이프라인을 만드는 것이었다. 저수의 마이크로폰만 점적으로 배치해 위치 정보를 얻고, Fast Continuous Wavelet Transform과 경량 신경망을 조합해 5.5 ms 내 추론을 가능하게 했다.

- **Empirical Impact**: 실험에서는 4 mm 최고 공간 해상도와 4개 마이크로폰 기반 64노드 어레이에서 99% 이상의 정확도를 보였고, 100 Hz 미만 저주파 신호의 웨이브폼 재구성도 달성했다. 또한 단일 털 접촉부터 5 mg 입자 충격, 맥파·깃털 터치·손가락 접촉까지 다양한 자극을 장갑과 대면적 스킨 형태로 검출해, 차세대 인간-기계 인터페이스를 위한 확장 가능·저비용·유연 센싱의 실증 근거를 제시했다.



### Occupancy-Grounded Room Segmentation for Hierarchical 3D Scene Graphs (https://arxiv.org/abs/2606.13727)
- **Prior Approaches**: 기존 계층형 3D 장면 그래프(3DSG)의 ‘룸’ 레이어는 장소 연결(place-connectivity), 벽면(wall plane), 또는 새 지도(BEV) 기반 분할/학습 모듈 등 서로 다른 공간 기판에서 구성돼 평가 기준이 일치하지 않았다. 그 결과 룸 노드는 물리적 방 인스턴스(개수·면적·경계)를 공통된 기하 기준으로 직접 검증하기 어렵고, 시스템별 지표에 의존해 비교가 제한됐다. 또한 일부 방식은 룸 경계 폴리곤을 바로 노출하지 않아 공정한 2D 방 분할 평가로 전환하기도 까다로웠다.

- **Core Contribution**: 이 논문은 점유(occupancy)에서 유도한 자유공간을 기준으로 룸 노드를 고정(anchoring)해, 각 룸에 명시적 폴리곤 풋프린트(footprint)를 부여하는 파이프라인을 제안한다. RGB-D로부터 3D 점유 지도를 만들고, 세로 여유 분석을 통해 2D 자유공간 레이어를 만든 뒤, 자유공간 분해(free-space decomposition)로 폴리곤 영역을 얻는다. 이후 이 영역을 시간에 걸쳐 추적해 룸 노드를 안정적으로 유지하며, 예측 폴리곤을 주석 룸 인스턴스와 1:1 매칭해 직접 평가할 수 있게 했다.

- **Technical Challenges**: 핵심 난제는 (1) 자유공간으로부터 실제 방 경계를 안정적으로 분해·추적하고, (2) 지도 복원 오류가 룸 폴리곤 경계에 전이되는 문제를 통제하는 것이다. 저자들은 영역 분해 결과를 IoU, 중심 이동, 면적 비율 조건으로 추적하고, 매칭 실패는 일정 기간 버퍼링하며 전체 자유공간/영역 수가 크게 흔들리면 이전 상태를 재사용하는 필터로 룸 정체성의 흔들림을 줄였다. 또한 룸 경계의 정밀도는 결국 점유→자유공간→분해로 전파되는 입력 지도 품질에 의해 제한되며, 이를 완전히 회피하는 대안은 아직 없다고 밝힌다.

- **Empirical Impact**: Matterport3D 12개 씬에서 룸 폴리곤을 평가한 결과, occupancy-grounded anchoring은 Hydra 대비 룸 인스턴스 회복을 크게 개선했다(리콜 0.152→0.379, F1 0.241→0.427). 다만 정밀도는 낮아졌는데, 이는 일대일 IoU 기준을 만족하지 못하는 ‘경계 품질 저하’·‘인스턴스 분리 실패’에서 주로 발생한 것으로 분석된다. 전반적으로 룸 레이어는 커버리지-정밀도(coverage–precision) 트레이드오프로 해석해야 하며, 경계 정확도를 포함한 방 인스턴스 회복은 여전히 해결 과제로 남는다.



### Instruct-Particulate: Scaling Feed-Forward 3D Object Articulation with Kinematic Contro (https://arxiv.org/abs/2606.14699)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존의 관절형 3D 객체 복원 연구는 소수의 정밀 라벨 데이터에 크게 의존해 왔고, 이 때문에 새로운 범주로의 일반화가 제한적이었다. 또한 최적화 기반 접근은 입력이 많아 스케일이 어렵고, 최근의 피드포워드 모델도 데이터 희소성 때문에 성능이 병목에 걸리는 문제가 있었다.

- **Core Contribution**: Instruct-Particulate는 정적 3D 메쉬와 함께 목표 ‘운동학(kinematic) 스펙’—부품 설명, 연결 구조, 관절 종류, 선택적 점 프롬프트—를 입력으로 받아 해당하는 관절형 분할과 관절 운동 파라미터를 한 번에 예측한다. 이때 스펙이 과업을 구체화해 모호한 정답(가능한 분할의 다수)을 평균내는 문제를 줄이고, 서로 다른 세분화 수준의 라벨도 함께 학습에 활용할 수 있게 한다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 같은 객체라도 관절 구조를 하나로 고정하기 어렵고(분할/의미의 불일치), 데이터 다양성이 증가할수록 일관성 문제가 커진다는 점이다. 이를 위해 모델 입력에 명시적 kinematic structure를 포함시키고, VLM이 테스트 시 스펙을 자동으로 획득하도록 설계했으며, 인코더-디코더 형태의 대규모 트랜스포머로 점-쿼리 기반 관절 분할과 관절 축·범위 파라미터를 동시에 디코딩한다.

- **Empirical Impact**: 논문은 15만 개를 훨씬 넘는 관절형 3D 객체로 구성된 이질(heterogeneous) 학습 데이터(합성 포함) 구축과 함께 실험을 수행했으며, 범주 간 일반화와 AI 생성 메시에 대한 성능이 기존 대비 더 좋아짐을 보였다. 또한 이미지-투-3D 모델을 통해 실제 이미지에서 관절형 애셋을 물리 시뮬레이터에 바로 내보낼 수 있는 수준으로 생성·재구성하는 확장 가능성도 확인했다.



### Provably Safe, Yet Scalable Reinforcement Learning (https://arxiv.org/abs/2606.14536)
- **Prior Approaches**: 기존 안전 강화학습은 소프트 제약 최적화(예: 라그랑주 기반)로 제약을 비용으로 다루는 방식이 주류였지만, 학습된 정책이 실제 배치에서 제약을 항상 만족한다는 형식적 보장은 약합니다. 반면 엄격한 안전 보장을 주는 방법들은 CBF(제어 장벽 함수)나 SI(안전 지표)처럼 명시적 증명 함수(증명서)를 구성해야 하고, 그 과정이 상태 차원 증가에 따라 비싸지며 종종 보수적인 행동을 유도합니다.

- **Core Contribution**: 이 논문은 PS2-RL(Provably Safe, yet Scalable RL)이라는 2단계 프레임워크로, 안전을 ‘증명’하면서도 확장성을 잃지 않는 학습 파이프라인을 제안합니다. 핵심 아이디어는 명시적 불변집합을 직접 계산하지 않고, 학습된 백업 정책으로 시스템을 전진 적분해 ‘암묵적(control-invariant) 불변집합’을 온라인에서 생성하는 구조입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 “강화학습이 만드는 신경망 제어입력”을 안전 제약에 대해 하드하게 강제하면서, 동시에 미분가능한(끝단-대-끝단) 학습이 가능해야 한다는 점입니다. PS2-RL은 1단계에서 safe-arrival value function으로 백업 정책의 시간을 절약하는 성질을 학습하고(기저 안전영역에 도달하도록), 2단계에서는 BCBF에서 유도된 제약을 미분가능한 제약-투영 레이어(control-invariant layer)로 엄격히 투영해 안전을 보장하면서 RL 목적함수를 계속 최적화합니다.

- **Empirical Impact**: 로봇 제어 태스크에서 상태 차원이 최대 10까지인 설정을 실험하며, PS2-RL이 학습과 배치 모두에서 100% 안전을 달성했다고 보고합니다. 또한 보수적인 증명서 기반 접근이나 단순한 백업(예: 해석적 정책) 대비 더 큰 안전 영역을 확보하면서 성능도 기저 방법들보다 앞선 결과를 보이며, 고차원 입력 제약 시스템에서의 실용성을 강화합니다.



### Causal Object-Centric Models for Planning with Monte Carlo Tree Search (https://arxiv.org/abs/2606.14418)
- **Prior Approaches**: 기존 모델 기반 강화학습(MBRL)은 월드 모델로 상상 전개를 하며 표준적으로 Dreamer류나 MuZero류의 잠재공간 계획을 활용해 왔다. 시각 환경에서 성능 병목은 관측을 단일 전역 표현(CNN 기반)으로 뭉치는 방식에 있었고, 이를 개선하기 위해 Slot Attention(슬롯 주의) 기반의 객체 중심 표현 학습과 이를 MBRL에 결합한 방법들이 나왔다. 다만 기존 객체 중심 MBRL은 객체 상호작용을 충분히 명시적으로 모델링하지 못하거나, 주석된 분할 마스크 의존(FULL-supervision) 문제가 남아 있었다.

- **Core Contribution**: COMET은 MuZero 스타일 잠재 계획에 객체 수준 귀납 편향을 추가해, 슬롯 구조(latent slot space)에서 MCTS(Monte Carlo Tree Search)로 계획을 수행하는 객체 중심 MBRL 알고리즘을 제안한다. 핵심은 고정된 비지도 객체 중심 인코더로부터 얻은 슬롯에 대해, 트랜스포머 월드 모델이 미래 슬롯과 보상을 예측하고, 정책·가치 헤드는 대상 토큰을 기준으로 ‘객체-인과 attention’을 통해 의사결정에 중요한 엔터티에 집중하게 만든다는 점이다. 또한 행동을 객체 슬롯에 결속시키는 ‘action-slot fusion’ 메커니즘을 도입해, 행동이 어떤 객체에 영향을 주는지 학습적으로 결합한다.

- **Technical Challenges**: 객체 중심 슬롯은 시간에 따라 순서가 뒤바뀔 수 있어(순열 불변성과 할당의 비결정성) 동적 환경에서 일관된 객체 추적이 어렵다. COMET은 이를 완화하기 위해 에피소드 내부에서 다음 시점의 슬롯을 현재 시점 슬롯으로 초기화하는 방식으로 시간적 안정성을 확보하고, 인코더는 비강화학습 데이터로 미리 학습한 뒤 고정해 표현 요동을 줄인다. 또 행동을 단일 임베딩으로 슬롯 전체에 강제로 압축하면 정보병목이 생기므로, 슬롯마다 행동 임베딩을 결합(슬롯-조건화)해 action-slot fusion을 설계하고, 이를 트랜스포머 입력으로 사용해 전이 예측을 안정화했다.

- **Empirical Impact**: 평가는 Object-Centric Visual RL benchmark, ManiSkill, Robosuite, VizDoom의 8개 시각·동역학 다양한 태스크에서 수행했으며, COMET은 훈련 초반부터 평균 정규화 점수(mean normalized score)에서 객체 중심/단일(monolithic) 기준선을 앞서는 경향을 보였다. 특히 대상 객체와 교란 객체가 뚜렷한 시각적으로 단순한 과업에서는 object causal attention이 실제로 과제 관련 객체에 높은 중요도(인과 점수/가중치)를 부여해 성능 격차를 만들었다. 반면 Block Lifting과 Cube Pushing처럼 표현이 흔들리는 제어·시각 복합 태스크에서는 객체 표현이 슬롯을 제대로 분리하지 못할 때 한계가 나타났고, 이는 객체 표현 강건성이 성능의 관건임을 시사한다.



### Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale: A Frozen-Backbone Grafting Diagnostic (https://arxiv.org/abs/2606.14153)
Comments:
          23 pages, 5 figures, 8 tables

- **Prior Approaches**: 기존 연구는 VLA에서 비전 인코더를 바꿀 때 언어·액션 구성요소와 함께 학습(공동 적응)되는 경우가 많아, 관측된 성능 차이가 인코더 자체인지 백본과의 상호적응인지가 섞여 해석이 어려웠습니다. VLM4VLA처럼 학습 단계에서 VLM 조합을 체계적으로 바꾸는 연구도 있지만, 이미 공개된 VLA 체크포인트를 상속받아 ‘후처리로 인코더만 갈아끼우는’ 실무 질문과는 거리가 있습니다.

- **Core Contribution**: 논문은 ‘냉동 백본 그래프팅(frozen-backbone grafting)’ 진단을 제안합니다. 언어 모델과 액션 전문가를 고정한 채, 후보 비전 인코더를 결정적 풀링(Adaptive Average Pooling)과 LayerNorm, 단일 학습 가능한 선형 프로젝터만으로 연결해 오프라인 행동 MSE를 비교하며, 작은 VLA에서의 인코더 최상위 선택이 큰 백본으로도 옮겨가는지 검증합니다.

- **Technical Challenges**: 핵심 기술 과제는 인코더 비교가 언어-인코더 공동 적응과 섞이지 않도록 공정성을 만드는 것입니다. 이를 위해 후보 인코더는 사전학습 가중치를 동결하고, 고정된 토큰 수가 되도록 공간 풀링을 결정론적으로 적용한 뒤 프로젝터만 2,000 스텝(배치/학습률 프로토콜 고정) 학습해 순수한 인코더-적합성 신호를 추출합니다; 또한 시뮬레이터와 체크포인트의 본체(embodiment) 불일치 때문에 폐루프 성공률 대신 오프라인 행동 예측 MSE를 사용해 측정 가능성을 확보했습니다.

- **Empirical Impact**: 결과적으로 작은 백본에서 1등으로 뽑힌 인코더가 큰 백본에서도 신뢰성 있게 1등을 보장하지 않았습니다. SmolVLA에서는 SigLIP이 두 스위트에서 전반적으로 우세하지만, π0.5에서는 DINOv2가 공간 스위트에서 앞서고 객체 스위트는 씨드 민감한 근접-동률 밴드를 보이며, 특히 백본-스위트 조합에 따라 랭킹 방향이 엇갈리는 양상이 관측됩니다. 다만 그래프팅 하네스 자체가 백본마다 부호가 다른 영향을 주므로(조건부 결론), 이 방법은 ‘대규모 커밋 전에 저비용으로 타깃 백본 진단을 돌리는 도구’로서의 실용적 의미가 큽니다.



### WAM4D: Fast 4D World Action Model via Spatial Register Tokens (https://arxiv.org/abs/2606.14048)
Comments:
          15 pages, 7figures, 9tables

- **Prior Approaches**: 기존 WAM(월드 액션 모델)은 미래 관측과 실행 가능한 로봇 동작을 함께 예측하지만, 주로 2D 비디오나 잠재공간 표현에 의존한다. 이로 인해 접촉 지오메트리·가려진 표면·자유공간 같은 3D 제약을 충분히 반영하지 못해 정밀 조작에서 오류가 누적될 수 있다. 반면 4D 기반 방법들은 깊이/노멀/포인트맵 같은 기하 표현을 명시적으로 예측해 물리 일관성을 높이지만, 조작 추론 시점에 비용이 큰 디코딩이나 최적화가 필요해 지연이 커진다.

- **Core Contribution**: 이 논문은 빠른 4D 월드 액션 모델 WAM4D를 제안한다. 핵심 아이디어는 기하 기반(geometric foundation) 사전지식을 “학습 시 미래 깊이 readout”으로만 주입하고, 추론 시에는 2D 관측→동작 경로만 남기는 것이다. 이를 위해 공간 레지스터 토큰(spatial register tokens)으로 인과 비디오-동작 트랜스포머의 중간 표현에서 깊이를 읽어내도록 훈련하고, 그 결과가 조작 성능으로 이어지게 한다.

- **Technical Challenges**: 주요 기술 과제는 (1) 비디오 예측에 기하 감독을 억지로 얹으면 인과적(causal) 결합이 약해질 수 있고, (2) 밀집 4D를 그대로 목표로 두면 디코딩 비용이 커진다는 트레이드오프를 동시에 해결하는 것이다. WAM4D는 공간 레지스터 토큰이 미래 깊이 학습 목표를 만들되, 정책 경로에서는 레지스터/깊이 디코더를 제거해 추론 경량화를 달성한다. 또한 Mixture-of-Transformers(MoT) 백본에 causal mixture attention을 설계해 비디오·동작·레지스터 토큰 간 가시성(visibility)을 제어하고, 미래 동작이 미래 비디오/레지스터 정보를 “치트”처럼 쓰지 못하도록 마스킹한다.

- **Empirical Impact**: 실험은 RoboTwin 2.0와 접촉이 많은 실세계 장기 지평 조작 작업에서 수행되며, WAM4D가 공간 일관성과 조작 성공률을 개선하면서도 추론 효율을 유지함을 보였다. RoboTwin 2.0에서는 비디오 품질 및 깊이 관련 지표에서의 개선과 함께 동작 예측 성능이 경쟁 수준으로 유지되었고, 실세계 로봇 AstriBot S1의 4개 작업에서도 전반적으로 가장 높은 성과를 보였다. 특히 접촉·기하 민감·긴 시간 실행이 필요한 설정에서 기존 WAM/VLA 계열보다 실패율이 낮아, “학습 시에만 기하 프라이어를 전달하는” 접근의 실용성이 입증된다.



### RT-VLA: Real-Time Vision-Language-Action Models via Knowledge Distillation (https://arxiv.org/abs/2606.14010)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 자율주행(E2E) 모델은 언어 추론과 행동 예측을 한데 묶어 해석가능성을 높였지만, 큰 비전-언어 백본과 생성형(자기회귀) 추론 때문에 추론 지연이 커 도로 환경에 바로 배치하기 어렵다는 한계가 컸습니다. DriveCoT, ORION, AutoVLA, OpenDriveVLA, SimLingo 계열은 언어 기반 추론을 강화했지만 실시간 제어 관점에서는 계산 비용이 병목이 되었습니다. 또한 기존 증류는 주로 행동/궤적 출력에 초점이 맞춰져 있어, 언어 설명 능력을 실시간 제어와 분리해 유지하는 설계가 부족했습니다.

- **Core Contribution**: RT-VLA는 SimLingo의 주행과 언어 추론 능력을 더 작은 학생 모델로 “다단계 지도형 증류”해 실시간 추론이 가능한 경량 VLA를 제안합니다. 비전 특징, 쿼리 표현, 웨이포인트 예측, 언어 로짓까지 여러 수준에서 교사를 따라가 언어 기반 추론 및 주행 성능을 동시에 보존하도록 설계했습니다. 더 나아가 안전-중요 장면에서만 오프라인 언어 분석(사후 설명)을 수행해, 실시간 제어 경로에는 지연을 추가하지 않게 했습니다.

- **Technical Challenges**: 문제는(1) 교사와 학생의 비전 토큰 길이/차원이 달라 직접 특징 매칭이 어렵고, (2) 설명용 언어 모듈을 학생이 잘 재현하려면 단순 출력 증류만으로는 “생성 분포 불일치(shift)”가 생길 수 있다는 점입니다. RT-VLA는 정렬 모듈로 비전·쿼리·예측 시퀀스를 맞춘 뒤, 웨이포인트/언어 로짓의 다단계 증류 손실을 구성했습니다. 또한 오프라인 증류 후에는 on-policy 언어 파인튜닝으로 학생이 실제로 생성한 토큰을 다시 교사에게 평가해 KL을 최소화, 사후 설명의 일관성을 높였습니다. 마지막으로 언어 추론 브랜치를 폐루프 제어에서 분리하고, 필요 시에만 로그된 프레임을 입력으로 후처리하도록 KV 캐싱 기반의 경량 생성도 적용했습니다.

- **Empirical Impact**: Bench2Drive(CarLARO v0.9.15)에서 RT-VLA는 폐루프 주행 점수 85.19로 SimLingo(85.07) 수준을 유지하며, SimLingo-BASE(85.94)에 대해서도 격차를 0.75점으로 줄였습니다. 추론 속도는 비전 전용 모드에서 1544.34ms→34.48ms로 44.8배, 비전+언어 모드에서는 7.9배 가량 빨라져 실시간 제어의 실용성을 크게 높였습니다. 언어 설명 품질은 DeepSeek-V4-Flash 평가 기준 50.9로 교사(SimLingo) 대비 0.9점 낮은 수준에 그쳐, 지연을 줄이면서도 설명 능력을 상당 부분 보존했음을 보여줬습니다. 특히 지연이 줄어든 만큼 장애 회피·차로 변경 같은 정성 사례에서 초기 반응과 합류 타이밍이 개선되어, 시간 제약이 큰 도심 시나리오에서 의미 있는 효과를 확인했습니다.



### An integrated interpretable control effectiveness learning and nonlinear control allocation methodology for overactuated aircrafts (https://arxiv.org/abs/2606.13794)
- **Prior Approaches**: 기존 제어 할당(control allocation)은 작동점에서 고정된 선형 제어 유효성 행렬 B를 가정하는 방식이 많았다. 이 접근은 다중 작동기(m>n) 중복성은 활용하지만, 비선형 결합과 상태 의존 유효성 변화(예: 비행 조건·작동기 상태에 따른 공력 변화)를 제때 반영하지 못해 불일치가 커지고 정확도·강건성이 떨어진다. 이를 보완하려고 고정 모델을 온라인에서 업데이트하거나(유한차분 기반) 데이터 기반 신경망·강화학습을 쓰면 성능은 좋아질 수 있으나, 계산량이 크거나(실시간 할당 불가) 블랙박스 특성으로 인해 검증·고장진단·안전성 설명이 어렵다는 한계가 남는다.

- **Core Contribution**: 이 논문은 작동기-가상 제어(virtual control) 매핑의 비선형 제어 유효성(control effectiveness mapping)을 “명시적이고 물리 제약을 포함한 해석적 모델”로 학습한다. 이를 위해 Sparse Identification of Nonlinear Dynamics(SINDy) 계열의 constrained SR3를 써서, 단순 근사나 신경망 없이도 데이터로부터 희소한 지배 방정식 구조를 복원한다. 학습된 매핑은 컴팩트하고 해석 가능하며, 입력에 대한 해석적 도함수(미분)를 제공해 비선형 솔버 안에 효율적으로 포함될 수 있다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 과격 기동에서 나타나는 상태 의존 공력·작동기 상호작용을 충분히 잘 포착하면서도, (2) 실시간 할당 루프에서 계산 가능한 형태(미분/평가 비용이 낮은 형태)로 유지하는 것이다. 저자들은 임의 다항 확장 대신 항공기 강체 회전역학의 구조(관성 기반 결합항, 모멘트 분해)를 후보 함수 라이브러리에 사전 반영해 식별 공간을 줄이고 조건을 개선했으며, SR3로 희소성과 데이터 적합의 균형 및 물리 제약을 함께 처리해 항목 선택의 안정성을 확보했다. 또한 온라인에서는 예측 잔차(residual)를 모니터링해 식별 모델이 크게 어긋날 때 계수를 갱신하는 적응 메커니즘을 두어 작동기 고장이나 운용 조건 변화에도 점진적으로 재구성되게 했다.

- **Empirical Impact**: 검증은 ADMIRE(Aero-Data Model in a Research Environment) 고충실도 비선형 항공기 벤치마크에서 수행되며, 공격적인 기동 전반에 대해 풀 비선형 온보드 모델과 유사한 정확도를 목표로 한다. 논문은 다양한 기동과 시나리오(고장 포함)에서 온라인 적응이 모델 불일치 증가를 억제해 성능을 유지함을 보이고, 동시에 기존 기준선 대비 계산 비용을 크게 줄여 실시간 할당 가능성을 제시한다. 결과적으로 “블랙박스 없이도” 비선형 제어 유효성 매핑을 해석 가능하게 복원하면서 실시간 솔버에 실용적으로 통합되는 중간지대 해법을 실험적으로 뒷받침한다.



New uploads on arXiv(cs.MA)

### Learning Coordinated Preference for Multi-Objective Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.14693)
- **Prior Approaches**: 기존 협력 MOMARL은 다목적 보상을 한 개의 가중치(선호) 벡터로 스칼라화해 학습하는 방식이 많았다. 하지만 모든 에이전트가 동일한 선호를 강제하면 충돌이 같은 방향에서 반복되거나(동일하게 효율을 중시) 역할 분화가 막혀 팀 수준 트레이드오프가 제한된다. 또한 일부 접근은 선호 벡터가 바뀔 때마다 별도 정책을 재학습해야 하고, 연속 행동공간에서는 벡터 보상 분해가 성립 조건에 묶이는 한계가 있다.

- **Core Contribution**: 이 논문은 Preference Coordinated Multi-agent Policy Optimization(PCMA)로, 팀이 유리한 트레이드오프를 만들기 위해 에이전트별 선호를 “좌표화(coordinate)”하는 아이디어를 제안한다. 핵심은 팀 레벨에서 선호를 학습해 각 에이전트가 서로 보완적인 역할을 맡도록 하며, 이를 통해 에이전트 간 목적 충돌을 완화하고 파레토 전선을 더 잘 커버하는 것이다. 이 목표를 위해 협력 MOMARL을 팀 최적 게임(team-optimal game) 관점으로 정식화하고, 선호 다양성이 팀 개선으로 이어질 수 있음을 이론적으로 연결한다.

- **Technical Challenges**: 가장 큰 기술적 도전은 “고정 선호에서의 에이전트 평형”이 반드시 “팀 성능 최적”으로 이어지지 않는다는 점이다. 이를 해결하기 위해 PCMA는 선호 p_i를 잠재 조정 변수로 두고, 각 에이전트에 대해 선호를 샘플링하는 확률적 플래너(Dirichlet 분포)를 학습하며, PPO 기반의 선호 조건 정책을 함께 최적화한다. 또한 선호들이 동일 방향으로 붕괴하지 않도록 쌍별 다양성 정규화를 넣어 역할 전문화를 유도하고, 학습 중 선호 변화가 유도하는 평형 경로를 국소적으로 추적할 수 있음을 연속성/추적 관점에서 보강한다.

- **Empirical Impact**: MPE, SMAC, MOMAland 등 여러 협력 다중에이전트 환경과 실제 교통제어 시나리오에서 PCMA는 성능과 트레이드오프 조정 모두에서 우수(또는 최상급) 결과를 보였다. 특히 선호 다양화가 에이전트 전문화를 만들어 파레토 전선 커버리지를 개선하고, 전투/공격-방어 같은 역할 분화가 더 넓게 형성됨을 관찰했다. 또한 다양성 정규화 계수와 팀 이득-로컬 이득 균형 등 주요 구성요소에 대한 절제 실험과, CARLA 기반 OpenCDA-MARL 검증을 통해 선호 조건화된 제어가 현실적 세팅에서도 유효함을 뒷받침한다.



### Naive Visual Memory is Not Enough: A Failure-Mode Study of GUI Agents (https://arxiv.org/abs/2606.14106)
Comments:
          9 pages, 5 figures, ICML 2026 WORKSHOP

- **Prior Approaches**: GUI 에이전트는 과거 궤적을 저장하고 비슷한 상태에서 재현해 의사결정을 돕는 ‘경험적 메모리’를 도입해 신뢰성을 개선해 왔다. 더 나아가 스크린샷을 통째로 저장·검색하는 시각 메모리도 제안됐지만, 어떤 실패를 줄이고 어떤 실패를 키우는지 체계적으로 분석되진 않았다.

- **Core Contribution**: 이 논문은 GUI 에이전트의 오류를 인지 실패, 시각 상태 오독, 숨은 동작 블라인드니스, 그라운딩 에러의 4가지로 분류하고, 이들이 perception-추론-action 파이프라인의 서로 다른 단계에 대응함을 보였다. 또한 전체 화면(풀 이미지) 기반 시각 메모리가 실패 분포를 ‘상태 단계 오류는 줄이지만 행동 단계 오류는 악화’시키는 경향을 드러내 핵심 쟁점을 정리한다. 이를 바탕으로 Action-Grounded Visual Memory(AGMem)를 제안해, 메모리를 행동과 직접 연결된 로컬 크롭으로 압축·검색하도록 한다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 어떤 픽셀을 저장해야 상태 해석에 도움이 되면서 행동 단계의 주의 분산을 막을지, (2) 검색된 예시가 현재 서브태스크와 어긋나면 부정적 전이가 생기는 문제를 줄이는 데 있다. AGMem은 성공/회복의 ‘행동 효과’가 드러나는 구역만 크롭한 action-relevant view를 저장하고, 서브태스크 기반으로 검색 공간을 좁혀 관련 없는 시각 단서를 재주입하지 않게 한다. 더해 오류가 감지되면 recovery-aware verification memory로 잘못된 상태의 복구 예시를 별도로 검색해 오류 전이를 차단한다.

- **Empirical Impact**: OSWorld와 WebForge, 그리고 AgentNetBench에서 AGMem은 풀 이미지 메모리 대비 실패 분포의 부작용을 완화하며 전체 성능을 끌어올린다. 특히 OSWorld에서 task success가 풀 이미지 메모리보다 33.3% 향상되었다고 보고하며, GPT-5.4-mini 기준으로도 엔드투엔드 정확도 개선(예: +6.8%p)과 각 실패 모드 동시 감소(전체 모드에서 개선)를 관찰했다. 결과적으로 ‘시각 메모리 여부’뿐 아니라 ‘행동에 근거한 시각 압축과 선택적 검색’이 GUI 에이전트 신뢰성 향상의 핵심임을 실증적으로 제시한다.



### Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Respons (https://arxiv.org/abs/2606.13832)
- **Prior Approaches**: CAGE Challenge 4의 기존 보상 중심(reward-only) 강화학습은 SOC의 작동 예산을 명시적으로 제한하지 않아, 보안 리워드는 높더라도 다운타임/MTTR·오탐 대응·방화벽 변경 같은 운영 비용을 과도하게 유발할 수 있습니다. 또한 휴리스틱 기반 접근이 유효 행동 처리와 임무 단계 트래픽 규율 같은 ‘운영 규율’을 내장해 MARL보다 강했지만, 그 규율이 학습으로부터 자동으로 담보되지는 못했습니다. 결과적으로 보상만 최적화한 에이전트가 실제 배포에 필요한 거버넌스(감사·한도 준수)를 안정적으로 학습하지 못한다는 문제가 남았습니다.

- **Core Contribution**: 논문은 SOC 운영 규율을 ‘안전 계약(safety-contract)’이라는 측정 가능한 제약으로 바꾸어, 에이전트가 에피소드 동안 MTTR(다운타임), false-positive(분석가 부담), 방화벽 변경(변경관리) 예산을 넘지 않도록 학습·검증하는 프레임워크를 제안합니다. 이를 그래프 기반 MARL로 구현하며, 시뮬레이터 관측과 재사용 가능한 운영 예산/제약 로직을 분리해 배치 가능성을 높입니다. 또한 ACD3-GAT( Adaptive Constrained Counterfactual Decisioning with a Graph Attention Network encoder )로 제약 최적화와 반사실적(action) 스크리닝을 통합합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 부분 관측과 비정상성(Blue-에이전트 vs Red-과정 경쟁) 속에서 (2) 장기 지평의 보안 성공과 (3) 세 가지 동시 SOC 제약을 함께 만족시키는 학습을 안정적으로 만드는 것입니다. 저자들은 Lagrangian 기반의 constrained MAPPO-GAT(C-MAPPO-GAT)로 비용 통제를 분리·검증하고, ACD3-GAT에서는 예산 컨텍스트, CVaR 기반 꼬리위험 추정, 상대(oppponent) 상태 정보, 그리고 Graph Counterfactual Risk Propagation(G-CRP)로 ‘예산을 깨는 행동’의 위험을 사전에 걸러내도록 구성했습니다. 결과적으로 보상 최적화가 유발하는 운영 피해를 학습 과정과 실행 전 스크리닝에서 함께 제어합니다.

- **Empirical Impact**: CAGE Challenge 4 재현 실험에서 무제약(unconstrained) 방법은 다운타임 예산 위반이 모든 에피소드에서 발생했고, 평균 다운타임 프록시 비용도 예산 50 대비 311~430 수준으로 크게 초과했습니다. 반면 C-MAPPO-GAT는 다운타임 위반율을 100%에서 0.3%로 낮추고 평균 다운타임 비용을 355.4에서 15.5로 줄였으며, ACD3-GAT는 평균 다운타임 비용 48.2와 13.8% 위반율로 ‘가장 보수적인 준수점’이 아닌 더 넓은 안전 계약(frontier)에 위치했습니다. 또한 토폴로지 시드 변화와 적응형 Red 스트레스 테스트에서 제약 기반 정책들이 reward-only MAPPO-GAT 대비 최악 성능 저하가 더 작게 유지되어, 배포 관점의 견고성을 실증했습니다.



### Large Language Models as Supervised Extraction Assistants: Lowering the Barrier to Documentation Standard Adoption in Agent-Based Modelling (https://arxiv.org/abs/2606.13749)
Comments:
          17 pages, accepted for publication at the Social Simulation Conference 2026, see this https URL

- **Prior Approaches**: ABM 모델 문서화 표준인 ODD, 데이터 사용 투명성을 다루는 RAT-RS 등이 존재하지만, 작성 비용이 커서 실제 채택은 제한적이었다. 특히 세부 질문이 많은 표준일수록(예: ODD+D처럼 요구사항이 확장될수록) 연구자가 “부가 정보”로 느끼는 문서화 작업 부담이 커진다.

- **Core Contribution**: 이 논문은 LLM을 활용해 RAT-RS 같은 문서화 표준의 질의응답 추출을 ‘인간 감독 하에’ 부분 자동화하는 접근을 제안한다. 이를 통해 연구자는 초안을 검증·승인하는 역할로 전환되어, 문서화 일관성과 누락 탐지가 개선될 수 있음을 보여준다.

- **Technical Challenges**: 문제는 LLM이 논문 텍스트에서 근거를 정확히 끌어오고(환각 억제), 질문 유형별로 다른 추론 난도를 안정적으로 처리하는지였다. 연구진은 4종 LLM로 동일 모듈 프롬프트를 실행하고, 사실/기술(서술)/설명(이유)/평가/이분형 등 5개 질문 유형을 분류해 일관성과 성능 차이를 측정하며, ‘근거 신호’와 출력 형식 제한 같은 프롬프트 설계로 신뢰 가능한 추출 범위를 가늠했다.

- **Empirical Impact**: 4개 LLM은 전반적으로 구조화되고 그럴듯한 답을 생성했지만, 인간-LLM 간 의미 정렬은 낮았고 특히 설명·평가형에서 변동이 컸다. 반면 LLM끼리의 반복 실행에서는 서술·사실형에서 상대적으로 높은 일관성이 나타나, “무엇(What)”은 비교적 안정적이지만 “왜(Why)”는 인간 검토가 필수라는 실무 휴리스틱을 도출했다.



### Contract-Based Compositional Shielding for Safe Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.14130)
- **Prior Approaches**: 기존 안전 멀티에이전트 강화학습에서는 행동 가드(shield)로 학습·배포 시점에 위험 행동을 마스킹하거나 대체하는 방식이 널리 쓰인다. 하지만 분산 가드에서 허용 가능성(어떤 행동이 안전한지)이 다른 에이전트의 비정상(non-stationary) 정책에 의존하는 안전 조정 문제에서는, 독립적으로 분해된 권한(permission)만으로는 팀 최적이면서도 안전한 행동을 과도하게 배제할 수 있다.

- **Core Contribution**: 이 논문은 분산 실행 환경에서 학습·배포되는 에이전트들이, 중앙 제어 없이도 팀 최적의 안전 행동을 “결정론적 보장” 형태로 복원하는 방법을 제안한다. 이를 위해 전역 안전 사양 φ를 LTL 안전 조각(LTL_safe)으로 두고, 에이전트들은 φ를 함의하는 로컬 LTL_safe 의무들의 튜플 중 하나를 선택하되, 전체 튜플이 동시에 인증되므로 로컬 의무를 상호 가정(assumption)으로 사용할 수 있게 만든다.

- **Technical Challenges**: 핵심 난점은, 중앙 런타임 제어 없이도 로컬 의무를 조합해 전역 안전 φ를 만족하면서 각 에이전트의 행동 마스크로 투영할 수 있어야 한다는 점이다. 논문은 학습 단계에서 비정상성을 다루기 위해 non-stationary multi-armed bandit으로 로컬 LTL_safe 의무 라이브러리의 튜플을 선택하고, 그 튜플이 전역 φ를 함의한다는 계약(contract) 인증을 기반으로 end-to-end 안전을 잃지 않도록 한다.

- **Empirical Impact**: 6개 환경과 15개의 알고리즘 변형에 대해 평가하며, 기존 분해형 분산 권한이 막던 팀-최적 안전 행동을 더 잘 복원함을 실증한다. 또한 학습·배포 모두에서 중앙 안전 제어 없이도 안전성을 유지하면서 보상을 최적화할 수 있다는 점에서, 안전 조정이 필요한 협력 MARL의 실용적 설계 방향을 강화한다.



### When Plausible Is Not Realistic: Evaluating Human Mobility in LLM-Based Urban Simulation (https://arxiv.org/abs/2606.13835)
Comments:
          14 pages, 10 figures

- **Prior Approaches**: 기존 LLM 기반 도시 시뮬레이터인 AgentSociety와 CitySim은 생성된 일정과 활동이 “그럴듯한지”에 초점을 두는 경우가 많았다. 그 평가는 시맨틱한 활동 분포나 서사적 일관성은 볼 수 있지만, 실제 인간 이동의 핵심인 공간·시간 제약을 함께 재현하는지 검증하기엔 부족했다. 또한 관측 가능한 이동 흔적(traceability)과 의사결정 과정을 관찰가능성(observability) 있게 남기지 못해, 오류가 어디서 발생했는지 해석이 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 LLM 기반 도시 시뮬레이터의 이동을 실세계 데이터와 직접 대조해 검증하는 프레임워크를 제안한다. 공간 이동 법칙, 시간 리듬, 네트워크 모티프, 의미 기반 활동 전이, 행동 이동 프로필을 다차원 지표로 결합해 “서사적 그럴듯함”과 “실증적 이동 현실성”을 분리 평가한다. Greater Paris와 Shanghai 데이터로 AgentSociety와 CitySim을 평가해, 어떤 요소가 재현되고 어떤 요소가 실패하는지 체계적으로 보여준다.

- **Technical Challenges**: 기여를 실현하려면 (1) LLM 에이전트의 POI 선택과 실행을 추적 가능한 형태로 기록하고, (2) 시뮬레이터 간 비교를 공정하게 하며, (3) 대규모 지도 생성과 지표 계산을 반복 실험 가능하게 만들어야 한다. 이를 위해 AgentSociety를 기반으로 En-AgentSociety를 구축해 방문 POI/카테고리와 프롬프트-응답 및 실행 흐름을 로깅(traceability)하고, 병목·지연을 모니터링할 수 있게 했으며, 지도 생성은 지역 단위로 확장되도록 최적화했다. 또한 CitySim은 공개 설명을 바탕으로 재구현·수정해 동일 인프라에서 일관된 비교가 가능하게 했고, 트립 길이·기원-목적지 흐름·체류시간·전이 역학 등 다양한 지표 계산 절차를 마련했다.

- **Empirical Impact**: 실험 결과는 시맨틱 활동 분포 같은 고수준 패턴은 일부 따라가도, 여행거리 분포·기원-목적지 흐름·체류시간·전이 동역학 같은 핵심 공간/시간 제약을 재현하는 데 큰 격차가 있음을 보여준다. 특히 기본 프롬프트(persona) 설정만으로는 이동 다양성이 안정적으로 확보되지 않으며, 프로필을 반영한 초기화가 필요하다고 관찰했다. 더불어 지역 규모 맵 생성, 관찰가능성 강화, 이동 메트릭 산출, 교통 시뮬레이터까지 포함한 공개형 인프라를 제공해 재현 가능한 벤치마킹을 촉진한다.



### TwinBI: An Agentic Digital Twin for Efficient Augmented Interactions with Business Intelligence Dashboards (https://arxiv.org/abs/2606.13731)
- **Prior Approaches**: 기존 NL 기반 BI 보조는 NL→SQL이나 차트 생성 등 ‘쿼리 작성’ 중심으로 발전했지만, 대시보드에서 사용자가 적용한 필터·계층·지표 같은 분석 상태를 채팅과 일관되게 유지하긴 어렵습니다. 그 결과 답변은 그럴듯해도 실제 대시보드 상태와 의미(집계 단위, 스코프)가 어긋나거나 다단계 탐색에서 상태가 망가지는 문제가 반복됩니다. 한편 대시보드는 강력한 시각적 조작이 있지만, 상태를 다시 복원해 다음 대화 턴에 반영하는 연결 계층이 부족해 사용자에게 재설정 부담이 생깁니다.

- **Core Contribution**: TwinBI는 LLM 에이전트 쌍과 실행 가능한 BI 대시보드 상태의 ‘디지털 트윈’을 결합해, 대화와 대시보드 조작을 하나의 공유 분석 상태(shared analytical state)로 동기화합니다. 이 상태는 통합 상호작용 로그를 기반으로 재구성되며, 스키마·계층·지표 매핑과 차트 컨텍스트까지 의미 있게 정렬해 에이전트가 잘못된 상태에서 답을 만들 여지를 줄입니다. 또한 사용자가 검증할 수 있도록 SQL·스키마 뷰·로그·/insights 같은 상태 기반 산출물을 노출합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘대화로 들어온 요청’과 ‘대시보드에서 누적된 시각적 조작’이 서로 다른 맥락을 가질 때, 집계 단위·필터 스코프·계층 레벨까지 포함한 정확한 현재 상태를 복원하는 것이었습니다. TwinBI는 대시보드 이벤트를 구조화해 통합 로그에 남기고, 의미 계층(측정값·차원·계층·조인 경로)과 함께 실행 가능한 쿼리 사양으로 상태를 재구성함으로써 에이전트의 의미적 기준선을 맞춥니다. 더불어 /insights는 ‘현재 보이는 증거’에만 기반하도록 출력 범위를 제한해, 상태 밖 추론으로 인한 불일치를 줄이도록 설계했습니다.

- **Empirical Impact**: 통제된 A/B 벤치마크에서 TwinBI는 정확일치(exact-match)를 43.3%에서 63.3%로, 부분점수(partial-credit)를 48.3%에서 70.8%로 끌어올렸고, 타임아웃도 40.0%에서 10.0%으로 크게 감소시켰습니다. 유저빌리티 연구에서도 대시보드-채팅 통합 워크플로가 과업 정확도는 유지하면서 인지·작업 부담을 중간 수준으로 유지했으며, 상태 인지 기반 상호작용 메커니즘에 대한 평가도 긍정적이었습니다. 즉 TwinBI는 에이전트 수준의 분석 신뢰성과 사용자 관점의 상태 기반 지원을 동시에 강화하는 실증 결과를 보였다는 점에서 의미가 있습니다.



### YeasierAgent: Agentic Social Sandbox as a Canvas for Intent-Driven Creation of Platform-Agnostic Symbiotic Agent-Native Applications (https://arxiv.org/abs/2606.13722)
- **Prior Approaches**: 기존 AI 앱 생성은 자연어를 코드나 화면 일부로 바꿔주는 방식이 주류였지만, 결과물이 독립된 스크립트·페이지·컴포넌트로 끝나 재사용성이 떨어지는 한계가 있었다. 또한 멀티 에이전트나 에이전트 샌드박스 연구도 시뮬레이션/관찰 중심에 머무는 경우가 많아, 사용자 관점에서 ‘소프트웨어 인터페이스 자체’로 노출되긴 어려웠다. 요컨대 사용자 정체성, 앱 상태, 사회적 맥락, 기기 표현이 하나의 연속 경험으로 묶이지 못했다.

- **Core Contribution**: YeasierAgent는 앱을 기기별 UI로 정의하기보다 ‘사용자-에이전트-세계’가 함께 참여하는 협업 공간으로 재정의한다. 특히 플랫폼 비의존적 상호작용 단위(에이전트, 장면, 대화)를 써서 대화형 앱을 웹·모바일 등 여러 단말에서 빠르게 구성하고 동일한 경험 흐름을 유지하는 구조를 제안한다. 동시에 감정적 동반(Companion)과 실용적 도구 실행을 하나의 경험 샌드박스 안에서 통합해 “Symbiotic Agent-Native Applications” 범주를 정리한다.

- **Technical Challenges**: 핵심은 (1) LLM 기반 생성이 만드는 앱이 단말과 무관하게 동일한 ‘장면·대화·선택’ 단위로 동작해야 하고, (2) 사용자의 디지털 트윈 에이전트를 지속적으로 정교화해 감정적 반응과 작업 수행을 동시에 만족해야 한다는 점이다. 논문은 벡터 저장 기반 장기 메모리와 Big Five(빅 파이브) 성향 파라미터를 프롬프트/행동 제어에 동적으로 인코딩해 디지털 트윈을 만들며, 장면-공간 매핑을 통해 작업 진행을 직관적으로 관찰 가능하게 설계한다. 또한 세계(World)를 이벤트 기반 관찰 캔버스로 두고 생성 앱(Creation Apps)이 그 위에서 규칙·목표·역할·사회적 결과를 실행하도록 분리함으로써 플랫폼 연속성을 확보한다.

- **Empirical Impact**: 저자는 라이브 플랫폼 배포를 통해 에이전트-공간 상호작용이 도구 실행, 게임형 추리, 인터랙티브 드라마 같은 서로 다른 앱 경험을 공통 원시 요소로 묶을 수 있음을 정성적으로 보인다. OpenClaw 호환 로컬 워크플로처럼 기술 로그를 읽지 않아도 에이전트의 위치·행동·표정으로 진행을 이해하게 하는 사례와, 다중 에이전트의 숨은 정보·선택 기반 게임, 사용 개입에 따라 변주되는 반-스크립트 서사 사례가 제시된다. 다만 멀티 에이전트 공간의 실시간 렌더링 비용과 LLM/네트워크 성능 의존성은 구현 제약으로 남아 있어, 향후 시각 최적화와 안정적 실행 개선이 과제로 제시된다. 



### WorkBench Revisited: Workplace Agents Two Years On (https://arxiv.org/abs/2606.13715)
Comments:
          8 pages, 3 figures. Follow-up to arXiv:2405.00823

- **Prior Approaches**: 기존 에이전트 벤치마크는 웹 탐색, 일반 보조, 도구 사용 등 주변 문제를 다루거나, LLM 평가자에 의존해 행동을 점수화하는 방식이 많았다. WorkBench는 사무 환경을 샌드박스로 구현하고, 에이전트가 임의의 경로로 작업하되 최종 상태를 정답과 비교해 ‘행동 자체’의 성패를 직접 평가하는 데 초점이 있다. 다만 2024년 출시 당시에는 최고의 에이전트도 작업의 43%만 완료했고, 포맷/툴 호출 실패 같은 요소가 결과를 크게 흔들었다.

- **Core Contribution**: 이 논문은 2024년 WorkBench를 2026년까지 재실행하며 성능과 안전성을 함께 재측정한다. 특히 구조화된 native tool-calling을 일괄 적용해 비교 기준을 공정하게 만들고, 작업 완료율뿐 아니라 의도치 않은 유해 부작용(예: 잘못된 수신자에게 이메일 발송)과 1회 실행 비용까지 2개 축을 추가해 ‘능력-안전-비용’의 동시 지표를 제시한다. 또한 기존 벤치마크의 채점/정답/프롬프트 불일치 및 툴 엔지니어링 문제를 수정해, 2026 점수와 2024 점수를 직접 비교할 때의 함정을 줄였다.

- **Technical Challenges**: 핵심 기술 난제는 에이전트가 샌드박스에서 임의의 경로로 작업할 때, 툴 호출 포맷 실패와 프롬프트-정답 불일치가 성능을 왜곡하지 않게 하는 것이다. 저자들은 ReAct의 텍스트 기반 툴 파싱 대신 공급자들이 제공하는 structured output의 native tool-calling을 사용해 스키마 준수 실패를 제거했고, 모델 추론이 남긴 기본 오류(예: ‘아직 오버듀이 아닌데 오버듀로 간주’ 같은 조건 판단, 캘린더 검색 결과 5개 제한을 고려한 재쿼리 부족)는 그대로 관찰되는 형태로 정리했다. 아울러 “last NN days” 오프바이원, 잘못된 그라운드 트루스, silent-zero 집계 버그, 줄바꿈 이스케이프 같은 엔지니어링 결함을 재생성/재계산해 결과 재현성을 높였다.

- **Empirical Impact**: 재평가 결과, 작업 완료율은 2024년 최상위 GPT-4(43%)에서 2026년 Claude Opus 4.8(88.8%)로 크게 상승했으며, 의도치 않은 유해 부작용 비율도 26%에서 2.5%로 급감했다. 더 나아가 능력과 안전이 상충하기보다 함께 개선되는 경향이 확인되어, 가장 많이 끝내는 모델이 대체로 가장 적게 ‘의도치 않은 피해’를 낸다는 점이 강조된다. 한편 비용은 모델·공급자 간 2자릿수 수준으로 격차가 커서, 오픈 웨이트가 더 저렴한 구간의 효율을 끌어올렸고(캐시 미적용 상한치 기준), 전체적으로 frontier 모델의 절대 효율 격차와 더불어 ‘방출일 단독’으로는 발전량을 설명하기 어렵다는 시사점을 준다. 



### AGORA: Can Deliberation and Governance Gates Absorb Participation Bias in Transit Planning? (https://arxiv.org/abs/2606.13696)
- **Prior Approaches**: TNDP(대중교통 노선망 설계)는 보통 네트워크·수요·제약을 고정한 채 최적화 알고리즘으로 해법을 비교하지만, 실제 채택은 공청회·의견수렴·표결 같은 참여 절차를 거치며 달라진다. 선행 연구는 소득·연령·장애·차량 보유 등 참여자의 분포가 체계적으로 치우친다는 점과, 이 편향이 형평성에 불리하게 작동할 수 있음을 보여주지만 참여 ‘구성’이 결과를 얼마나 바꾸는지 인과적으로 분리해 실험하진 못했다.

- **Core Contribution**: 이 논문은 AGORA(Agentic Governance for Optimization-Representation Alignment)로 네트워크/수요/솔버는 고정한 채 ‘회의 구성(누가 참석하는가)’만 체계적으로 바꿔, 결과 변동의 원인을 과정 설계로 되돌려 재해석한다. 더불어 LLM 기반 참여자(이해관계자 에이전트)로 비판·투표·수정이 포함된 구조적 심의를 수행하고, 합의·서비스·형평을 위한 거버넌스 게이트까지 결합해 참여 편향이 결과에 미치는 의존도를 줄이는 메커니즘을 겨냥한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘참석자 구성’ 효과를 잡아내기 위해, 동일한 TNDP 인스턴스에서 구성만 바꾸면서도 과정(심의/수정/게이트)이 공정하게 비교되도록 설계하는 것이다. AGORA는 같은 회의 프로필에서 통제된 시드로 시뮬레이션을 반복하고, LLM 모드(비판·수정 사용)와 no-revision pass-through(비판 없이 선택만 통과) 조건을 쌍으로 맞춰 ‘언어’만이 아니라 ‘수정된 의사결정’이 결론을 바꾸는지까지 분리 측정한다.

- **Empirical Impact**: Mandl, Mumford0 두 벤치마크에서 집계 수준의 결과는 구성에 크게 흔들리지 않았지만, 꼬리위험(상위 분위 passenger cost)과 형평 격차에서는 대표 샘플이 비뚤어진 구성보다 유리한 경향이 나타났다. 특히 게이트(합의·서비스·형평 기준)는 프로필 간 변동을 압축해 꼬리위험 차이를 크게 줄였으나, Mumford0에서는 수용률이 낮아 임계값이 인스턴스별 보정이 필요함도 확인됐다. 결론적으로 참여 편향을 ‘통제 불가능 입력’이 아니라 ‘심의 설계와 거버넌스 문제’로 전환하며, 대표성 보장이 어렵더라도 잘 짜인 절차가 결과 의존도를 실질적으로 낮출 수 있음을 보여준다.




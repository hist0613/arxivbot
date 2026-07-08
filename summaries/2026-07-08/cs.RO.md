New uploads on arXiv(cs.CL)

### On the feasibility of dependency parsing of non-human sequences without a gold standard. Is evaluation possible in other species? (https://arxiv.org/abs/2607.06542)
- **Prior Approaches**: 의존 구문 분석은 문장(수열)을 루트 트리 형태의 dependency 구조로 찾아내는 문제다. 비지도 의존 구문 분석은 gold standard(정답 트리뱅크) 없이 학습·평가하려 하지만, 종/비인간 종에서는 정답을 만들 수 없어 정확도 산출이 불가능하다는 통념이 있었다. 기존 연구는 임의 구문/엣지 방향을 무시한 soft/hard 평가 같은 방식으로 평가 가능성을 탐색해왔지만, 정답 부재 환경에서는 성능을 신뢰도 있게 하한까지 보이기 어렵다는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 네트워크 과학 관점의 이론적 근거를 통해 “정답 트리 없이도 엣지 정확도의 하한을 제시할 수 있다”는 프레임을 제안한다. 핵심은 비인간 영장류의 발성/행동 수열에서 나타나는 짧은 길이로의 빠른 길이 분포 붕괴 때문에, 파서가 맞는 엣지를 많이 뽑을 확률이 수학적으로 높아진다는 점이다. 반대로 인간 언어는 이러한 길이 분포 성질이 달라 동일한 논리로는 평가가 어려워진다고 주장한다.

- **Technical Challenges**: 정답이 없을 때 평가 점수를 어떻게 정의·기대값으로 연결할지가 기술적 난제다. 논문은 엣지 방향을 제외한 free tree, 그리고 trees 정확도(complete metric)와 edges 정확도(undirected dependency accuracy)를 soft/hard 형태로 설정한 뒤, 무작위 파서의 기대 성능을 수열 길이 분포 p(n)로부터 계산한다. 또한 길이 분포가 실증 분포/균등 분포/지수(특히 displaced geometric) 분포일 때의 기대값을 근사·분석해, “좋은-enough 비지도 파서”가 최소한 무작위 기준선보다 높을 수 있음을 논리적으로 연결한다.

- **Empirical Impact**: 31종 비인간 영장류의 최대 수열 길이 정보와 실제 발성·제스처 길이 분포를 결합해, 무작위 파서 기준의 엣지 정확도가 geladas는 평균 51% 수준, chimpanzees는 79% 이상으로 높게 나옴을 보인다. 길이 2를 초과하는 구간으로 제한하면 정확도는 떨어지지만(예: geladas 41%, chimpanzees 56% 이상), 여전히 인간 언어의 기준선(대략 10개 길이 제한 시 27%, 전 구간 평균은 13% 내외)보다 현저히 높다고 보고한다. 결과적으로 gold standard 없이도 비인간 영장류에서는 비지도 의존 구문 분석 평가가 “가능”해지고, 인간에서는 여전히 “어려운” 이유가 수열 길이 분포의 구조 차이에 있음을 실증적으로 뒷받침한다.



### Hierarchical Acoustic-Semantic Modeling: Modality Separation and Semantic Coherence for Full-Duplex SLMs (https://arxiv.org/abs/2607.06540)
Comments:
          22 pages, 9 figures

- **Prior Approaches**: 기존 Spoken Language Model(SLM) 풀듀플렉스(full-duplex) 연구는 대체로 half-duplex를 강제로 이어붙이거나(VAD 기반 대화관리), 혹은 Thinker-Talker 같은 다단계 구조로 지연을 감수해 해결해 왔다. 또한 CDM 방식처럼 텍스트(semantic)와 음향(acoustic)을 공유 파라미터 공간에 함께 학습시키면 통합은 되지만, modality interference로 인해 지식이 훼손되기 쉽다. 그 결과 음성 지능(speech intelligence)과 대화 자연스러움의 동시 달성이 제한적이었다.

- **Core Contribution**: 이 논문은 풀듀플렉스 SLM에서 성능 저하의 뿌리를 “공유 딥 파라미터 공간에서 발생하는 acoustic–semantic 간 gradient conflict”로 규명한다. 이를 바탕으로 Lychee-FD를 제안하며, hierarchical parameter separation로 깊은 층에서 충돌하는 최적화 방향을 물리적으로 분리한다. 동시에 semantic alignment channel을 두어 sparse 정렬로 인한 semantic dilution을 상쇄하고, 의미 일관성을 유지하도록 설계했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 음향과 의미 학습 목적이 깊은 층에서 상반된 그라디언트 방향을 보이는 문제, (2) 텍스트-오디오 시간 해상도 불일치로 인해 semantic supervision이 희석되는 문제였다. 논문은 layer-wise gradient cosine similarity와 gradient magnitude ratio로 이 충돌/희석 현상을 정량화하고, 공유 파라미터를 유지한 채 우회하려는 방식 대신 구조적으로 분리하고 의미 채널로 앵커링하는 처방을 제시했다.

- **Empirical Impact**: 여러 full-duplex 벤치마크에서 Lychee-FD는 Spoken QA에서 +7.4%, FullDuplexBench 1.5에서 +28.5%의 개선을 보이며 최신 성능을 크게 끌어올렸다. 특히 inference efficiency를 해치지 않으면서도 반응 지연과 인터럽트 대응성(예: FullDuplexBench 1.5에서 Stop latency 570ms)을 함께 개선해 “효율-지능 트레이드오프”를 완화했다. 저자들은 이 논문이 modality interference의 원인 해설과 이를 해결하는 실전형 네이티브 end-to-end 프레임워크를 동시에 달성한 첫 사례라고 주장한다.



### Life Style Levels: Neighborhood Delineation using Geospatial Data (https://arxiv.org/abs/2607.06529)
Comments:
          43 pages, 38 figures

- **Prior Approaches**: 기존에는 도시 내 경제 수준을 세밀하게 구분하려 해도, 인도 같은 개발도상 지역에서 미시적인 사회경제 데이터가 부족해 분석이 제한되는 경우가 많았다. 일부는 통계나 조사 기반 정보를 쓰려 했지만, 도시·지역 간 적용성과 비용 문제로 인해 대규모 확장이 어려웠다. 위성 데이터 기반 접근도 있었으나, 해석 가능하고 규칙 형태로 일반화하기는 쉽지 않았다.

- **Core Contribution**: 이 논문은 공개 위성 이미지에서 얻은 building morphology(건물 형태)만으로, 격자(grid) 단위의 도시 구획을 만들고 도시 affluence(부유/결핍) 수준을 대비하는 영역을 지도화하는 프레임워크를 제안한다. 59개 인도 도시·타운을 대상으로 고해상도 격자를 분할하고, 해석 가능한 형태학적 지표들을 조합한 transparent, rule-based scoring 체계로 점수화해 분류한다. 결과는 Google Street View 관측으로 검증되며, 밀도 기반 건물 군집화를 뭄바이 informal settlement(비공식 정착지)와의 공간 중첩까지 확장한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 건물 형태 지표만으로 격자 내부의 생활 수준 차이를 안정적으로 반영하고, (2) 복잡한 추론 대신 사용자·정책자가 이해할 수 있는 규칙 기반 점수로 구현하며, (3) 위성에서 파생된 feature로부터 신뢰도 높은 지도를 만드는 것이었다. 저자들은 open-source 위성 기반 형태 정보를 해석 가능한 morphological indicators로 설계하고, 이를 투명한 룰 기반 scoring에 통합해 분류의 근거를 명확히 했다. 또한 밀도 기반 clustering으로 뭄바이의 조밀 정착지를 식별하고, 그 군집이 알려진 비공식 정착지와의 공간 중첩을 보이도록 분석을 구성했다.

- **Empirical Impact**: 검증에서 격자 클래스 간 뚜렷한 대조가 관측되며, lifestyle affluence indicators가 기대하는 방향과 일관된 결과를 보였다. 뭄바이에서는 density-based clustering으로 얻은 군집이 비공식 정착지와 상당한 공간 겹침을 나타내, 형태 기반 도시 구획의 유용성을 뒷받침한다. 나아가 도출된 affluence 클래스에 대해 consumer loan delinquency(소비자 대출 연체)를 탐색 매핑해 경제적 취약성과의 연관 가능성까지 보여 주며, 전적으로 공개 지리공간 데이터로 비용 효율적·확장 가능한 granular affluence mapping을 제공한다는 점에서 의미가 크다.



### RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation (https://arxiv.org/abs/2607.06527)
Comments:
          Accepted for publication in ICML 2026 as a full research paper; 21 pages

- **Prior Approaches**: KGQA는 보통 ‘retrieve-then-read’로 다중 홉을 풀지만, 각 홉에서 이산 노드를 선택하는 과정 때문에 end-to-end 미분 가능성이 깨져 검색기가 downstream 오류를 교정하기 어렵습니다. 또한 중간 bridge 노드가 질의와 어휘적 겹침이 거의 없는 semantic gap 상황에서는 entity-linking 기반 방법이 성능이 급격히 저하됩니다. 최근 LLM 에이전트형 접근은 추론 구조를 만들지만, billion-parameter 모델을 여러 번 통과해야 해서 비용이 커지고, 생성기가 검색 구조를 무시하는 reasoning shortcut 문제도 남아 있습니다.

- **Core Contribution**: 이 논문은 RSF-GLLM( Recurrent Soft-Flow Graph-to-LLM )로 그래프 추론과 답변 생성을 분리해, differentiable graph reasoning을 안정적으로 학습하면서도 LLM의 고비용/고분산 학습 영향을 최소화합니다. Recurrent Soft-Flow(RSF)는 Recurrent Query Updater(GRU)를 통해 연속 relevance score(soft flow)를 전파하고, Dynamic Gating Mechanism으로 구조 단서만으로도 의미적으로 멀리 있는 bridge 노드를 탐색합니다. 추출된 reasoning path를 텍스트화해 LLM fine-tuning에 활용함으로써 답변이 지식그래프의 토폴로지에 근거하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 중간 홉에서 질의-노드 의미 유사성을 강제하면 정답 경로가 막히는 semantic gap과 (2) soft flow가 그래프 전체에 퍼지는 flooding 문제입니다. 논문은 구조적 전파와 내용 기반 분별을 분리하고, λ(t) 게이팅으로 fan-out 해소가 필요한 순간엔 content bias를 켜되, semantic gap 구간엔 끄도록 학습합니다. 더불어 flow sparsity regularization(엔트로피 기반)을 적용해 soft 확률이 이산 경로로 수렴하도록 이론적으로 유도하고, greedy backtracking으로 구조적으로 타당한 경로를 복원·텍스트화합니다.

- **Empirical Impact**: WebQSP와 CWQ 실험에서 RSF-GLLM은 competitive 성능을 달성하면서도 LLM 중심/에이전트형 접근 대비 inference 효율이 더 높다고 보고합니다. 특히 semantic-gap-heavy 쿼리에서의 견고성 향상이 강조되며, 추론 경로를 명시적으로 제공해 hallucination 위험을 줄이는 방향으로 의미가 있습니다. 결과적으로 ‘학습 가능 추론 경로 + 근거 있는 생성’의 결합을 KGQA 실무 배치 효율 측면에서 한 단계 끌어올렸다는 평가를 기대할 수 있습니다.



### DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation (https://arxiv.org/abs/2607.06507)
- **Prior Approaches**: 기존 Multi-hop RAG는 반복 검색, query reformulation, evidence critique, sufficiency judging 같은 유용한 조작을 제공하지만, 보통은 각 방법별 파이프라인이나 사전 정의된 제어 토폴로지 안에 갇혀 있습니다. 그 결과 “현재 증거 상태에서 무엇을(검색/진단/가피-엔티티 확장/갭 질의/중단) 선택해야 하는가” 같은 공통 제어 문제를 한 프레임에서 학습·비교하기가 어렵습니다. 또한 더 많은 retrieval이 항상 이득으로 이어지지 않는 비용-효율성 문제가 남아 있습니다.

- **Core Contribution**: DynaKRAG은 multi-hop 증거 획득을 “진화하는 evidence state”에 조건화된 제어 문제로 재정의하고, 서로 다른 RAG 동작을 atomic evidence operations로 원자화한 통합 학습 프레임워크를 제안합니다. 매 단계에서 hard validity layer가 현재 상태에서 실행 가능한 action set을 구성하고, 학습된 controller가 그중 다음 조작을 선택해 evidence state를 갱신합니다. 이를 통해 retrieval, diagnosis, gap-directed acquisition, stop-and-answer, 그리고(종단 단계의) answer-focused compression까지 하나의 순차 의사결정으로 조율합니다.

- **Technical Challenges**: 핵심 난제는 실행 가능(정의됨/캡 미소진/전제 조건 충족)한 연산만 골라야 하면서도, 각 연산이 현재 상태에서 증거를 얼마나 개선하는지(효용)를 함께 학습하는 것입니다. DynaKRAG은 action utility 학습을 validity와 분리해, undefined·redundant·premature action은 validity layer가 먼저 제거하고 value model은 남은 유효 action만 랭킹하도록 설계했습니다. 학습 신호로는 supporting-evidence coverage 변화를 기반으로 acquisition 관련 보상을 만들고, sufficiency_check에 대해서는 증거 준비도 학습을, stopping에 대해서는 현재 support coverage를 사용합니다.

- **Empirical Impact**: Qwen2.5-7B-Instruct 백본에서 HotpotQA F1 0.5998, 2Wiki F1 0.5340, MuSiQue F1 0.3061로 3개 벤치마크 모두에서 최강 controlled baseline 대비 성능 우위를 보였고, Qwen 정책은 다른 답변 모델(GPT-4o-mini, Llama-3.1-8B)로도 전이되어 과적합 우려를 줄였습니다. 추가 실험에서 uniform-valid 정책으로 controller를 대체하면 F1이 3.96~5.78포인트 하락했으며, sufficiency feedback 제거는 세 데이터셋 모두 성능을 악화시켰습니다. 또한 controlled retrieval-cap 실험 결과는 “추가 retrieval이 항상 이득”이 아니라는 점을 보여주며, 증거 상태가 바뀌는 흐름 속에서 retrieval·진단·갭 지향 획득을 함께 조율하는 접근의 실용성을 뒷받침합니다.



### Pitwall: Faithful Natural-Language Race-Strategy Briefings from a Calibrated Real-Time Monte Carlo Engin (https://arxiv.org/abs/2607.06495)
Comments:
          21 pages, 2 figures, 6 tables. Live-deployment results from the 2026 Austrian and British Grands Prix. URL: this https URL

- **Prior Approaches**: 기존 F1 레이스 전략 시뮬레이션은 결정론적 최적화나 랩타임/타이어 열화 재현에 초점이 있었지만, 승부 확률의 캘리브레이션(신뢰도)까지 실시간으로 보장하는 공개 사례는 드뭅니다. 또한 확률 기반 반사실(예: 지금 피트 vs 2랩 대기) 비교를 common-random-numbers로 통제해 전략 차이를 분리하거나, 라이브 타이밍 스트림을 직접 받아 생성까지 연결한 end-to-end 시스템도 거의 없었습니다. 마지막으로 데이터-투-텍스트 생성은 출처 없는 환각 위험이 커서, “실제 선수에 대한 문장”을 마감 시간 내에 내보내려면 검증 체계가 필수입니다.

- **Core Contribution**: Pitwall은 라이브 스포츠 코멘터리의 grounded generation을 ‘기법의 목표’가 아니라 아키텍처 속성으로 다루며, 생성 문장을 타입화된 사실 청구(claim)로 분해해 확률 레이스 상태로 검증한 뒤 통과한 문장만 공개합니다. 더 나아가 verifier가 fine-tuning 데이터 자체의 채택을 게이트하며, 3,045개 모델 생성 타깃 중 모든 claim이 상태를 지지하는 81.9%만 학습에 남기고 나머지는 provably faithful 템플릿으로 폴백합니다. 같은 verifier가 생성·학습·운영 전 단계에 걸쳐 ‘근거 없는 서술’이 시스템에 들어오지 못하도록 막는 구조가 핵심입니다.

- **Technical Challenges**: 가장 큰 난제는 (1) 매 몇 초마다 변하는 확률적 그라운딩 상태를 기준으로, (2) 마감 지연 내에, (3) 문장 단위 claim을 신뢰도 있게 검증하며, (4) fine-tuning으로 인한 풍부함이 희소 상태에서 환각으로 무너지는 문제를 통제하는 것입니다. Pitwall은 벡터화된 Monte Carlo 엔진(N=2,000, 상태를 N×C 배열로 동시에 전진)과 확률 캘리브레이션된 SC/VSC·dirty-air·은근 오버테이크 상호작용을 함께 두고, 언어 계층에서는 위치/갭/타이어/페이스/오버테이크/레이스 컨트롤 등 10종 claim 스키마를 3개 언어로 추출·검증합니다. 또한 ‘캘리브레이션 최적 vs 결정 최적’이 충돌할 때를 분리 게이트로 처리하며, 파운데이션 모델의 instruction adherence가 희소 컨텍스트에서 환각을 유발하는 바탕 요인임을 4개 베이스 모델 감사로 확인해 해결책으로 sparse-context auditing을 운영 모델에서 제거합니다.

- **Empirical Impact**: Calibrated Monte Carlo 엔진은 126개 학습 레이스로 보정(2018–2024)하고, 2025–2026 완전 홀드아웃에서 winner-in-top-3 90.3%(155개 백테스트)와 held-out Brier 0.0745를 보여 확률 품질을 실증합니다. 언어 생성은 verifier 게이트로 상태를 뒷받침하는 문장만 채택되도록 설계됐고, 풍부한 타깃 학습이 항상 좋은 결과를 주지 않으며 결함은 스케일보다 베이스 모델의 지시 준수/감사 설계 문제임이 드러납니다(역효과도 함께 보고). 2026년 오스트리아·브리튼 라이브 그랑프리에서 라이브 타이밍→상태 재구성→추천→검증된 3개 국어 코멘터리까지 end-to-end 운영을 확인했으며, 실버스톤에서는 결과가 알려지기 전 디스크에 커밋한 확률 타임스탬프 트레이스가 깃발 10랩 전부터 최종 우승자를 고정하는 것으로 보고됐습니다.



### Data Analysis in the Wild: Benchmarking Large Language Models Against Real-World Data Complexities (https://arxiv.org/abs/2607.06482)
Comments:
          29 pages, 9 figures

- **Prior Approaches**: 기존 LLM 기반 데이터 분석 벤치마크는 작은 테이블에서의 사실 검색이나 text-to-SQL·Table QA에 치우쳐, 대규모 멀티 탭ular 데이터와 메타데이터/외부 지식 통합의 어려움을 충분히 반영하지 못했다. 또한 질문에 대한 응답은 평가하지만, 데이터 분석가가 수행하는 탐색적 인사이트 발견 능력은 상대적으로 덜 측정되는 편이었다. 그 결과 실제 현장형 문제에서의 성능 격차가 드러나기 어려웠다.

- **Core Contribution**: DataGovBench는 정부 오픈데이터를 기반으로 Table QA(분해형 질문에 대한 텍스트/시각화 정답)와 Table Insight(사용자 질의 없이 탐색해 전문가 수준의 발견을 생성)를 함께 평가하도록 설계했다. 특히 인사이트의 기준 정답을 전문가 보고서에서 추출해 주관성 문제를 완화하고, 대규모·다중 테이블·메타데이터·외부 지식이 동반되는 현실 복잡도를 포함한다. 이로써 “답하기”와 “발견하기”를 동시에 밀도 있게 검증하는 새 기준선을 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) LLM 컨텍스트 한계를 고려한 테이블 직렬화와 정보 압축, (2) 암묵 조건·데이터 변환·다중 테이블 선택 같은 정답성 요구, (3) 인사이트 생성의 주관성 및 정량 평가가 어렵다는 점이다. 논문은 feature type-specific table serialization로 요약 입력을 만들고, Answer Agent에서 파이썬 생성-실행 실패 복구(self-correction)와 시각화/계산 검증(reflection) 루프를 넣어 오류를 줄이려 했다. 인사이트는 전문가 보고서에서 인사이트를 추출·표준화해 ground truth를 구성하고, 평가도 의미 정렬 기반 점수(LLaMA-3-Eval 유사)를 통해 분해해 측정했다.

- **Empirical Impact**: 실험 결과, Answer Agent 같은 에이전트 지원이 있더라도 Table QA와 Table Insight 모두에서 최신 LLM의 성능은 여전히 낮아 현실 데이터 분석 요구를 충족하기엔 큰 격차가 있음을 확인했다. 특히 Table QA에서는 condition filter error와 transformation error 같은 정합성 문제와 시각화/테이블 선택 오류가 두드러졌고, Table Insight에서는 주제는 맞추더라도 narrative 및 정성·정량 세부 일치가 약했으며 정량 값 재현은 거의 불가능에 가까웠다. 저자들은 이로부터 현재 에이전트에 “탐색적 서사 수준 추론”과 “복잡 테이블에서의 정확한 사실 회수” 능력이 부족하다는 관찰을 제시한다.



### From Voting to Agent Collaboration: Answer-Type-Aware LLM Pipelines for BioASQ 14b (https://arxiv.org/abs/2607.06452)
Comments:
          15 pages

- **Prior Approaches**: 기존 BioASQ Task B 연구는 retrieval-augmented LLM으로 단일 프롬프트를 적용하거나, 앙상블/agent 기반 기법을 각각 독립적으로 사용해 왔습니다. 하지만 yes/no는 근거 스니펫의 순서와 구성에 민감하고, factoid와 list는 동의어·표면형식·검증 전략 부족으로 정답 순위/정확도가 흔들리는 문제가 컸습니다. 또한 여러 스니펫이 상충할 때 어떤 근거를 우선해야 하는지 결정하는 일관성이 부족하다는 한계가 반복됐습니다.

- **Core Contribution**: 이 논문은 질문 유형(yes/no, factoid, list)에 맞춘 question-type-specific LLM 추론 프레임워크를 제안합니다. 각 유형에 서로 다른 추론 절차를 배치해 yes/no는 snippet shuffling+self-reflection으로 결정 안정성을 높이고, factoid는 full-snippet 기반 in-context learning과 consensus로 정밀한 생의학 개체 식별을 강화합니다. list는 evidence 추출-후보 생성-검증-집계로 역할을 분리한 multi-agent 협업으로 과생성/누락을 동시에 줄이는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 난제는 (1) 스니펫 순서 변화에 따른 출력 흔들림, (2) 생의학 엔티티의 표면형식/동의어 차이로 인한 exact-match 평가 실패, (3) list에서 높은 재현율과 높은 정밀도를 함께 만족시키는 검증 설계였습니다. 저자들은 질문 유형별로 서로 다른 입력 구성(스니펫 셔플, full-snippet), in-context learning 데모(유사 질문 기반 retrieval + 검증된 CoT 예시), 그리고 선택적 verification/합의 규칙(yes/no majority+검증 에이전트, factoid 투표 기반 후보 필터링, list의 4단계 agent 파이프라인)을 결합해 이 문제들을 완화했습니다.

- **Empirical Impact**: BioASQ 14b 공식 평가에서 전 배치에 걸쳐 경쟁력 있는 성능을 보였고, 특히 Batch 4의 factoid 서브태스크에서 1위를 기록했습니다. 지표 관점에서도 yes/no는 macro F1이 비교적 안정적이었고, list는 배치가 진행될수록 F-measure가 크게 상승해 협업형 검증 파이프라인의 효과가 확인됐습니다. 다만 factoid는 strict/lenient 정확도 차이와 MRR 변동성이 남아 있어, 동의어 정규화·약어 처리·후순위 랭킹 최적화가 후속 과제로 제시됐습니다.



### Automated Compliance Mapping in Cloud Security with Domain-Adapted Sentence Transformers (https://arxiv.org/abs/2607.06364)
Comments:
          10 pages, 6 figures. Submitted to the 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)

- **Prior Approaches**: 기존 규정 준수 자동화는 주로 문서 형식화나 휴리스틱/규칙 기반에 치우쳐 있었고, 표준 간 매핑이나 준수 추론은 여전히 수작업 모델링 의존이 컸습니다. 또한 Sentence Transformer 기반 의미 유사 접근도 있었지만, 클라우드 보안 규정 용어에 맞춘 사전 학습/평가가 부족해 한계가 뚜렷했습니다.

- **Core Contribution**: 이 논문은 클라우드 보안 프레임워크의 control을 기술 증거인 metric으로, 또는 표준 간 control-to-control로 연결하는 매핑을 자동화하는 도메인 적응 파인튜닝을 제안합니다. Cisco CCF를 허브로 삼아 여러 유럽 표준을 연결하고, EMERALD의 metric 카탈로그까지 포함한 클라우드 준수용 labeled semantic pairs 코퍼스를 구축했습니다.

- **Technical Challenges**: 핵심 기술 과제는 규정 문구(추상적·법적 표현)와 메트릭 설명(구체적·기술적 표현) 간 어휘/구문 차이를 줄이는 것이었습니다. 연구진은 Sentence Transformer를 bi-encoder로 두고 대조학습(Multiple Negatives Ranking Loss)으로 control-to-metric과 control-to-control을 단일 유사도 학습 목표에 함께 학습시키며, back-translation과 LLM 기반 paraphrasing으로 데이터 변이를 생성해 시나리오별 효과를 비교했습니다.

- **Empirical Impact**: 결과적으로 파인튜닝된 모든 모델이 zero-shot 기준선을 두 평가 태스크 모두에서 능가했으며, control-to-metric에서는 최대 23 nDCG@10 포인트까지 향상되었습니다. control-to-control에서는 multi-qa-mpnet-dot-v1이 back-translation 조건에서 0.870 nDCG@10을, 비영(zero) 질의만 보면 0.965를 달성했는데, 전반적으로 도메인 학습 데이터가 성능을 좌우한다는 점을 실험적으로 확인했습니다.



### Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs (https://arxiv.org/abs/2607.06327)
- **Prior Approaches**: 기존 LLM 불확실성 추정(UE) 평가는 주로 영어에 집중돼 다국어(특히 저자원 언어)에서 성능이 유지되는지 근거가 부족했다. 또한 LLM-as-a-judge, BERTScore, n-gram overlap 같은 대체 지표는 잡음을 만들 수 있고, 언어별 편향도 UE 비교를 왜곡할 수 있다. 미니멀한 단답형 설정은 긴 생성 과정에서의 불확실성 신호를 충분히 보지 못했다.

- **Core Contribution**: 본 논문은 22개 언어(고·중·저자원)를 대상으로 9가지 UE 방법을 대규모로 평가한 최초의 비교 연구를 제시한다. 두 개의 사람 검수 MCQA 데이터셋에서 정답 라벨의 고정된 선택지 기반으로 정답성은 exact matching으로 유지하고, 불확실성은 추론(긴 reasoning) 텍스트에서만 추출해 모델 기반 프록시 없이 AUROC를 측정한다.

- **Technical Challenges**: 주요 기술적 난제는 언어·생성 길이가 바뀌어도 신뢰 가능한 UE 비교가 되도록 ‘정답성 근거’를 흔들리지 않게 만드는 것이었다. 저자들은 긴 생성(reasoning 약 150단어)을 유도하되, LLM-as-a-judge 및 임베딩 기반 스코어를 배제하고 MCQA 라벨을 기준으로 AUROC를 계산하도록 평가 프레임을 설계했다. 더불어 생성 언어(질문/추론 언어 분리)와 모델 스케일, cross-lingual answer 옵션이 UE 신호에 미치는 영향을 체계적으로 실험했다.

- **Empirical Impact**: 실험 결과, UE 성능은 언어 자체보다 ‘추론(reasoning) 생성 언어’에 크게 좌우됐고 영어로 추론을 유도하면 저자원 언어의 AUROC 격차가 크게 해소됐다. 또한 UE 방법 선택은 모델 스케일에 따라 달라지며, 작은 모델에서는 Token Entropy 같은 open-box 확률 기반이 유리하고 큰 모델에서는 Self Verbalized(닫힌 상자) 우위가 뚜렷해졌다. 마지막으로 selective prediction을 위한 임계값(threshold) 보정에서 영어만으로 보정하는 방식도 의미 있는 오차 감소를 제공하지만, 언어별 보정은 에러 탐지 성능을 더 끌어올려 다국어 신뢰성 배치 가이드라인을 제공한다.



### From Sinhala to Dhivehi: Cross-Lingual Transfer Learning for Low-Resource Speech Recognition (https://arxiv.org/abs/2607.06289)
Comments:
          7 pages, 1 figure, 8 tables, Accepted paper at the 12th International Moratuwa Engineering Research Conference (MERCon) 2026

- **Prior Approaches**: 기존 저자원 ASR은 Wav2Vec, XLS-R처럼 self-supervised 사전학습을 통해 라벨 의존도를 낮추는 방향이 주류였다. 다만 Dhivehi처럼 데이터가 적으면 fine-tuning만으로는 타깃 언어의 음향 특성이 충분히 반영되지 않아 성능이 제한된다. Dhivehi·Sinhala에 대한 기존 연구는 주로 멀티링구얼/언어모델(KenLM) 결합이나 토크나이징 개선에 집중됐고, ‘Sinhala→Dhivehi’의 관계 기반 cross-lingual transfer 자체는 체계적으로 검증되지 않았다.

- **Core Contribution**: 이 논문은 Sinhala(상대적으로 자원이 있는 친족 언어)를 소스로, Dhivehi ASR을 위한 transfer learning이 실제로 얼마나 이득을 주는지 5가지 패러다임으로 통제 실험했다. 특히 continual pre-training(CPT), sequential fine-tuning, multilingual fine-tuning(언어 ID 토큰 유무 포함) 각각의 효과를 Dhivehi-only 기준선과 직접 비교해 ‘무엇이 되는 전략인지’를 밝힌다. 또한 언어 계통과 무관한 Turkish을 대조군으로 둬, 개선이 단순 다국어 데이터 증강이 아니라 언어적 relatedness에서 오는지 분리해 보여준다.

- **Technical Challenges**: 저자원 설정에서는 사전학습된 음향 인코더가 Dhivehi의 발화/음향 분포에 즉시 적응하지 못하는 문제가 크며, transfer 경로에 따라 성능 편차도 크게 나타난다. 저자들은 Wav2Vec을 기반 모델로 삼고, CTC 디코딩에 external language model로 KenLM을 shallow fusion해 음향 표현과 디코딩 점수의 상호작용을 통제했다. 그 결과 decoding 설정(beam width)과 함께, 언어적 relatedness가 있어도 전략에 따라 이득이 실현되지 않을 수 있음을 반복 실험으로 확인했다.

- **Empirical Impact**: 최고 성능은 ‘Sinhala CPT → Dhivehi fine-tuning(KenLM)’ 조합으로, WER 12.89%, CER 2.70%를 기록하며 Dhivehi-only baseline 대비 WER 13.50%p, CER 3.02%p 개선했다. 동시에 KenLM 유무가 WER을 24~29%p 수준으로 흔들어, 저자원 Dhivehi에서는 transfer 자체보다 decoding/언어모델 영향이 더 지배적일 수 있음을 실증했다. Turkish 대조군은 Sinhala가 주는 이득이 친족 언어의 음운·음향 중첩에서 비롯됨을 뒷받침했으며, 종합적으로 저자원 ASR 연구에서 external language modeling과 비교 가능성(디코딩 포함)을 함께 설계해야 함을 시사한다.



### Early Language Learning via Spreading Activation and Category Exploration in Complex Networks (https://arxiv.org/abs/2607.06258)
- **Prior Approaches**: 기존 연구는 단어 습득이 의미 범주와 어휘 범주에 따라 불균등하다는 현상을 주로 경험적 데이터 관찰이나 단순한 거리 기반 가설로 설명해 왔습니다. 그러나 범주 간 전이와 학습자가 범주를 “탐색”하는 동역학을 함께 모델링하기에는 한계가 컸습니다.

- **Core Contribution**: 본 논문은 초기 언어 학습을 그래프 기반 mental lexicon 위의 탐색 문제로 재구성하고, spreading activation과 범주를 강제로 탐색하도록 하는 enforced exploration을 상호작용으로 모델링합니다. 특히 어휘 범주 방문의 제약이 복잡 네트워크에서 어떤 방식으로 학습 경로를 바꾸는지에 초점을 둡니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 언어별로 단어-유사도 그래프를 재구성하고 (2) CDI를 기준으로 의미/어휘 범주를 매핑한 뒤 (3) 탐색-활성화 메커니즘이 실제 범주 횡단 패턴을 재현하는지 검증하는 것입니다. 저자들은 Wordbank의 규범적 연령(normative ages)과 최신 word similarity 재구성 리소스를 활용해 시뮬레이션을 평가하고, shortest path baseline과 비교하며 burstiness 및 average persistence time 같은 동역학 지표로 탐색 특성을 점검합니다.

- **Empirical Impact**: 네 언어(German, English, Dutch, Rioplatense Spanish)에서 spreading activation이 shortest path baseline보다 normative word acquisition을 더 잘 모사했습니다. 또한 CDI 수준에서 관측되는 복잡한 전이와, 동일 CDI 내에서의 burstiness/지속 시간 패턴까지 spreading activation이 더 잘 설명해, 어휘 발달이 활성화 동역학과 범주 방문 제약의 비자명한 결합으로 이해될 수 있음을 시사합니다.



### Spider 2.0-AIFunc: Extending Real-World Text-to-SQL to AI-Native SQL Workflows (https://arxiv.org/abs/2607.06229)
Comments:
          24 pages, 3 figures, 7 tables

- **Prior Approaches**: 기존 text-to-SQL 벤치마크(예: Spider 2.0, BEAVER)는 전통적인 SQL 연산자 조합만을 평가해 왔고, Snowflake Cortex AI functions처럼 LLM을 호출하는 AI-native SQL 생성은 검증하지 못했습니다. 또한 실행 정확도(execution accuracy)는 같은 SQL이 일관된 결과를 내야 하는데, AI functions는 temperature=0이어도 소폭 출력 불일치가 발생할 수 있어 벤치마크 설계가 까다롭습니다.

- **Core Contribution**: 이 논문은 AI-native SQL을 직접 평가하는 Spider 2.0-AIFunc 벤치마크를 제안합니다. Snowflake Cortex AI functions 6종(AI_CLASSIFY, AI_FILTER, AI_SENTIMENT, AI_SIMILARITY, AI_EXTRACT, AI_AGG)을 포함해 총 465개 검증 인스턴스를 구성하고, 기존 Spider2-Snow의 소스 작업을 에이전트 기반으로 AI-native 형태로 변환했습니다.

- **Technical Challenges**: 핵심 난제는 (1) AI function 선택과 파라미터(예: 분류 라벨, 추출 스키마)를 지시문 수준에서 모호하지 않게 고정하는 것과 (2) AI function 호출로 인한 실행 결과 변동성을 통제하는 것입니다. 이를 위해 에이전트가 SQL과 자연어 지시문을 함께 수정하며 파라미터 누락/오류를 해결하고, 다중 패스 반복 실행 및 시간 창(time window) 분리 검증을 통해 안정적인 인스턴스만 공개했습니다.

- **Empirical Impact**: 10개 SOTA 언어모델을 평가한 결과, 폐쇄형 모델은 67–70%대 실행 정확도를 보인 반면 오픈소스 최고 성능은 58.1%로 격차가 나타났습니다(주된 원인: 술어/조건 지정, 스키마 grounding, AI function 파라미터화 오류). 또한 전통 text-to-SQL용 에이전트(스키마 검색, 관련 테이블 선택 중심)는 AI-native SQL에서는 최소 Spider-Agent 대비 큰 이점을 주지 못했는데, 이는 AI function 내부의 미세한 의미 선택이 정답을 좌우하기 때문으로 해석됩니다.



### Pluralis v0.1: Towards a Multicultural, Multimodal, Multilingual Benchmark for AI Risk and Reliability (https://arxiv.org/abs/2607.06196)
- **Prior Approaches**: 기존 AI 안전 평가·벤치마킹은 문화에 둔감한 기본값을 가정해 각 지역의 법, 사회언어적 뉘앙스, 문화적 금기 등을 가리는 경향이 있다. 그 결과 Vision-Language Models(VLMs)는 글로벌 배포에서 현지 맥락 위반을 놓치기 쉬운데, 서구 데이터의 단순 적응/평균 성능 지표가 이러한 맹점을 가려버린다.

- **Core Contribution**: 본 논문은 문화 우선(culture-first) 관점에서 만든 멀티모달·멀티리전·다국어 데이터셋 Pluralis v0.1을 제안한다. 여섯 개 아시아-태평양 국가와 여덟 언어(총 6,448 프롬프트)를 커버하며, 서구 안전 데이터를 현지화하는 방식이 아니라 현지에서 직접 안전 위험을 수집해 “현지 문화 적절성”을 핵심 평가 축으로 분리한다.

- **Technical Challenges**: 핵심 도전은 (문장 단독으론 무해해 보이지만) 텍스트와 이미지가 함께 주어질 때 시너지로 법/문화 위반이 발생하는 상호작용을 정교하게 평가하는 것이다. 이를 위해 Judge-Pluralis를 제안하는데, 문화 분류(경험적으로 도출한 cultural taxonomy) 라벨 예로 학습된 LLM-as-a-Judge 앙상블에 agreement-gated 방식을 적용해 판정 신뢰도를 높인다.

- **Empirical Impact**: Pluralis의 일부를 대상으로 관찰한 VLM 거동에서 현지/언어별 반복 실패 양상이 체계적으로 드러났다. 특히 이미지 오인에 따른 후속 위해, item-context-locale 상호작용 누락, 부적절한 refusal이 자주 나타났고, 이러한 차이는 전 세계 평균 지표로는 잘 보이지 않는 사각지대를 확인시켜 준다. 저자들은 Pluralis가 “완성된 문화 정렬 평가”가 아니라 향후 다국어·다문화 평가 연구를 촉발하는 출발점임을 강조한다.



### Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design (https://arxiv.org/abs/2607.06175)
Comments:
          21 pages, 5 figures

- **Prior Approaches**: LLM로 BPMN을 생성할 때, SFT는 학습 데이터에 있던 패턴을 재현하는 데는 강하지만 결과 품질을 다차원 기준(문법·의미·유용성)에 맞춰 상향 최적화하기 어렵다. 기존 RL 기반 접근은 외부 평가를 붙여 성능을 올리긴 했지만, 보상 함수를 어떤 방식으로 구성(차원 가중, 무효 페널티 등)해야 하는지에 대한 체계적 비교는 부족했다. 또한 다차원 품질을 한 개의 스칼라 보상으로 단순 합산하는 경우가 많아, 차원이 서로 충돌할 때 학습이 어떻게 흔들리는지 불명확했다.

- **Core Contribution**: 이 논문은 RL 기반 BPMN 프로세스 모델 생성에서 보상 함수 설계(다차원 점수의 가중·무효 페널티·집계)를 체계적으로 실험한다. Llama 3.1 8B와 Qwen 2.5 14B를 대상으로 48개 설정을 만들고, Group Sequence Policy Optimization(GSPO)로 BEF4LLM의 38개 자동 메트릭(구문/실용/의미 품질)을 기반 보상을 최적화한다. 특히 “보상 구성 자체가 최적화 결과를 좌우하며, RL 적용 여부만큼 큰 영향도 낼 수 있다”는 점을 실증한다.

- **Technical Challenges**: 핵심 난제는 여러 품질 차원을 하나의 학습 신호로 합칠 때, 특정 차원을 더 중요하게 두면 실제로 그 차원이 개선되는지(또는 모드 붕괴가 나는지)였다. 연구진은 BEF4LLM에서 생성물을 먼저 validity로 걸러낸 뒤, 구문·실용·의미 점수를 보상으로 조합하며, 차원 가중(동등 vs 표적)과 invalidity penalty(음수 페널티 vs 0) 등 보상 축을 분리해 비교했다. 또한 GSPO의 그룹 내 상대 비교 특성에 따라 보상 스케일 문제를 줄이도록 설계된 학습을 적용했고, SFT 초기화 유무가 아키텍처별로 어떻게 상호작용하는지도 함께 확인했다.

- **Empirical Impact**: 실험 결과, GSPO는 두 모델 모두에서 실용성과 구문 품질을 유의미하게 끌어올리면서 의미 충실도는 대체로 보존했으며, 출력 변동성은 6배 이상 줄였다. 특히 직관과 달리 equal reward weighting이 표적 가중보다 일관되게 우수했고, 특정 차원 강조는 그 차원 개선 실패뿐 아니라 저품질 모드로 붕괴시킬 수 있었다. 또한 invalidity penalty와 SFT 초기화의 효과는 모델 아키텍처에 의존적이어서, 단순한 기본값 튜닝이 아니라 경험적 검증이 필요함을 보여주며 다차원 자동 평가가 가능한 구조화 생성 전반에 일반화될 수 있음을 시사한다.



### LongCrafter: Towards Diverse Long-Context Understanding via Evidence-Graph-Guided Instruction Synthesis (https://arxiv.org/abs/2607.06160)
- **Prior Approaches**: 긴 컨텍스트 SFT를 위해 합성 데이터로 학습을 강화하려는 시도가 있었지만, 기존 데이터는 체계적 태스크 분류가 없어 커버리지가 좁았습니다. 또한 문서에서 바로 질문을 만들다 보니 증거 구조(문단 간 의존)와 난이도 계층이 약해, 모델이 쉬운 지름길로 답을 맞히는 경향이 커졌습니다. 마지막으로 추론 단계마다 출처 증거에 고정하는 faithfulness supervision이 부족해, 문서 기반이 아닌 파라메트릭 지식을 섞는 비충실 추론 위험도 남아 있었습니다.

- **Core Contribution**: LongCrafter는 장문 SFT용 데이터를 ‘구조화된 합성’으로 다루며, 계층적 task taxonomy와 evidence-grounded 파이프라인을 결합해 위 세 한계를 동시에 겨냥합니다. 장문 이해를 local/shallow와 global/deep으로 나누고 32개의 fine-grained task type을 생성의 전역 prior로 사용해, 태스크 커버리지와 생성 난이도를 의도적으로 설계합니다. 무엇보다 instruction–response가 문서에서 위치가 특정된 evidence span에 엄격히 근거하도록 만들어, 추론이 추적 가능하고 충실하게 이어지도록 합니다.

- **Technical Challenges**: 핵심 과제는 (1) 증거가 문단 사이에서 체인·트리·그래프처럼 연결되는 의존성을 모델이 사용할 수 있는 형태로 데이터에 반영하는 것, (2) 이를 통해 난이도는 높이되 무작정 어렵게만 만들지 않고 task type에 맞춰 조절하는 것, (3) 응답이 원문 증거와 단계별로 일치하도록 보장하는 것이었습니다. LongCrafter는 문맥을 evidence graph로 분해하기 위해 Extract-then-Construct 절차를 쓰고, task-relevant span을 노드로 삼아 cross-paragraph dependency edge를 연결해 “어떤 증거가 왜 필요한지”를 그래프로 명시합니다. 이후 그래프에 조건부로 instruction을 만들고, 응답은 단계별로 관련 evidence를 verbatim citation 형식으로 인용하며, LLM 검증(유일 정답/맥락 기반)을 통과한 샘플만 남기는 방식으로 해결합니다.

- **Empirical Impact**: 실험에서 LongCrafter로 학습한 모델은 LongBench, LongBench v2, LooGLE에서 모든 SFT baseline을 상회했으며 Qwen2.5-7B와 LLaMA-3.1-8B 모두에서 특히 고난도 태스크에서 가장 큰 폭의 개선이 관찰되었습니다. 또한 동일 백본에서 LongCrafter 데이터가 기존 데이터보다 과제 다양성과 난이도 분포가 더 균형 있게 퍼져 있고, ‘lost in the middle’ 문제를 완화할 정도로 증거 위치에 대한 강건성(evidence localization robustness)이 높다고 분석했습니다. 데이터가 2,000개로 제한돼도 official post-trained 모델 대비 All-Overall이 개선되는 결과는, evidence 그래프 기반의 충실한 합성 데이터가 적은 스케일에서도 장문 이해 성능을 실질적으로 밀어올릴 수 있음을 시사합니다.



### LLM Agents for Deliberative Collaboration: A Study on Joint Decision Making Under Partial Observability (https://arxiv.org/abs/2607.06157)
Comments:
          Code is available at this https URL

- **Prior Approaches**: 기존 언어 에이전트 연구는 단일 에이전트의 추론·행동, 또는 협업/협상/협의 같은 다중 에이전트 대화를 주로 다뤘습니다. 하지만 다수 에이전트가 부분적이고 비대칭적인 정보를 가진 상태에서 ‘합의에 도달하기 위한 정보 교환·정렬·의사결정’을 체계적으로 평가한 벤치마크는 부족했습니다.

- **Core Contribution**: 이 논문은 deliberative collaboration(숙의 기반 협업)을 ‘부분관측 하 협동 joint decision-making’ 문제로 수식화해, 공유 보상 하에서 합의 도출을 평가 가능한 단일 추상화로 정리합니다. 또한 메뉴 디자인과 작업 배분을 포함해 관측 구조·역할 권한·평가 프로토콜을 달리하되 동일한 숙의 협업 골격을 유지하는 확장형 benchmark를 제안합니다.

- **Technical Challenges**: 핵심 난제는 (1) 비대칭 부분관측에서 필요한 정보를 충분히 교환·집계(belief alignment)하고 (2) 불확실성을 반영해 공동 결정을 내리며 (3) 계산·검증까지 수행하는 것입니다. 이를 위해 데이터베이스 기반 작업 생성, NR/VR 중심의 정량 평가, 그리고 선택적 외부 도구로 Solver(정수계획)·Calculator(제약 검증 및 피드백)를 제공해 ‘언어 숙의’와 ‘수학 처리’의 역할을 분리해 진단합니다.

- **Empirical Impact**: 180개 태스크에서 다양한 SOTA LLM을 평가한 결과, 숙의 협업은 여전히 최신 모델에도 정보 교환/집계·추론·수학에서 병목을 남겼습니다. 다만 진단 분석은 숙의가 성찰과 오류 수정의 기회를 주어 centralized baselines 대비 성능이 오르는 경우도 있음을 보여주며, tool 사용 역시 모델별로 효과가 달라 ‘정답 계산’만으로는 부족할 수 있음을 시사합니다.



### Prompting Complexity: Shortest Prompts for Texts and Behaviors in LLMs (https://arxiv.org/abs/2607.06145)
- **Prior Approaches**: 기존 프롬프트 엔지니어링은 경험적으로 “짧은 프롬프트로 원하는 출력을 얻는다”는 관찰을 다루지만, 이 현상을 정보이론 관점에서 정량화하진 못했다. 콜모그로프스키 복잡성처럼 ‘가장 짧은 기술(description)’을 찾는 아이디어가 있으나, 언어 모델에서는 보편적(machine-independent) 불변성이 성립하지 않아 모델마다 접근성·비용이 달라진다. 또한 완전 탐색은 가능하더라도 실용적 프롬프트 탐색과의 연결이 약했다.

- **Core Contribution**: 논문은 고정된 instruction-tuned LM에서 목표 텍스트를 결정적 decoding으로 생성하는 ‘가장 짧고 그럴듯한(promptly plausible) 프롬프트’ 길이를 prompting complexity로 정의한다. 이는 resource-bounded Kolmogorov complexity의 LM-상대적(모델-의존적) 아날로그로, 프롬프트가 정보의 일부를 제공하고 나머지는 모델의 weights·학습 분포·토크나이저·템플릿·디코딩 규칙이 채운다는 관점을 형식화한다. 더 나아가 정확한 재현이 아닌 근사 출력을 위한 soft prompting complexity, 그리고 출력이 아니라 원인 프롬프트를 비교하는 prompting distance, 스펙을 만족하는 행동에 대한 behavioral prompting complexity까지 확장한다.

- **Technical Challenges**: 핵심 난관은 ‘프롬프트 공간’을 임의의 토큰 문자열 전체로 두면 의미 없는(glitch/adversarial) 프롬프트까지 포함되어 검색이 인간 친화적이지 않다는 점이다. 이를 해결하기 위해 plausible text를 nucleus sampling의 ρ 임계값으로 모델의 고확률 in-distribution 연속을 따르는 텍스트들로 제한하고, 해당 제한 하에서 LM을 bounded text interpreter로 보고 결정적 decoding의 관측 가능한 preimage를 기준으로 최단 프롬프트를 정의한다. 또한 exact는 너무 엄격하므로 ε-거리 기반 근사 정의로 확장해 rate–distortion처럼 ‘길이-정확도’ 트레이드오프를 수학적으로 다룰 수 있게 했다.

- **Empirical Impact**: 프레임워크는 “어떤 텍스트/행동이 짧은 프롬프트에서 접근 가능한가”를 실험적으로 연구할 수 있는 질문 목록(복잡성의 추정, instruction-following과의 관계, prompting distance의 효용, 합성 데이터의 복잡성과 성능 연동, judge 스펙 민감도, jailbreak 같은 바람직하지 않은 행동의 복잡성 증가 방법 등)을 제시한다. 즉, 프롬프트 최적화·합성 데이터 생성·모델 인버전·안전성 평가·행동 명세 기반 검증을 동일한 ‘모델-의존적 알고리즘 압축’ 언어로 묶어 준다는 점에서 의의가 크다. 앞으로 특정 LM 인터페이스에서 접근성/원인 비교를 실증적으로 계량화하려는 연구 의제를 구체화해, 현재의 경험적 프롬프트 관행을 이론-실험 양방향으로 전환하는 출발점이 될 수 있다.



### CurateEvo: Data-Curation Evolving for Agentic Post-Training (https://arxiv.org/abs/2607.06140)
- **Prior Approaches**: 기존 LLM 에이전트 post-training 파이프라인은 데이터 curation을 고정된 전처리로 간주해, 주로 data augmentation에 집중하고 filtering·refinement·하류(downstream) 실패 적응은 상대적으로 소홀했다. 그 결과 long-horizon 의사결정에서 반복되는 실패 양상을 제대로 반영해 학습 데이터를 재구성하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 failure-driven dynamic evolution 기반 데이터 큐레이션 프레임워크 CurateEvo를 제안한다. curation 전략을 실행 가능한 코드로 표현하고, held-out development set의 failed trajectory를 근거로 epoch마다 코드를 반복적으로 재작성해 supervised fine-tuning 데이터, reinforcement learning 데이터, inference-time memory bank까지 동시에 생성한다.

- **Technical Challenges**: 핵심 과제는 (1) 실패를 진단해 어떤 데이터는 보강·정제·제거할지 결정하고, (2) 학습 턴을 불필요하게 늘리지 않으면서 비용까지 고려해 효율적으로 진화시키는 것이다. CurateEvo는 실패 모드의 재발을 추적해 augment/filter/refine을 먼저 수행하고, 이후 cost-aware objective로 중복·저유틸 데이터 턴을 pruning해 전략을 효율화한다.

- **Empirical Impact**: ACEBench-Agent, BFCL-V4, {τ}^2-Bench에서 labeled 및 wild-data 설정 모두에서 CurateEvo가 기존 curation 방법을 일관되게 능가했다. 평균 점수는 labeled에서 3.2점, wild-data에서 2.7점 개선됐으며, 다양한 post-training recipe와의 호환성이 높고 curation overhead를 크게 줄인다는 분석 결과도 제시한다.



### Measuring the practice of shared-decision making (OPTION12): An Investigation into Open-sourced Smaller LLMs (OS-sLLMs) for Better Privacy and Sustainability (https://arxiv.org/abs/2607.06127)
- **Prior Approaches**: 기존 SDM 자동 평가는 OPTION5 같은 더 단순한 관찰 도구에 주로 초대형 상용 LLM을 사용해 왔고, 환자 데이터 프라이버시·비용 문제가 남아 있었습니다. 또한 OPTION12는 더 세분화된 12개 항목 코딩이라 인간 코더의 시간 부담과 코더 간 불일치가 특히 큽니다.

- **Core Contribution**: 본 논문은 OPTION12를 대상으로, 로컬 배포 가능한 open-source smaller language models(OS-sLLMs)가 SDM 코딩을 수행할 수 있는지 최초로 실증합니다. 아울러 여러 모델의 점수 불일치를 사람이 합의하듯 조정하는 Judge-LLM consensus framework를 제안해 human-in-the-loop 워크플로에 맞췄습니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 시간적 담화 맥락에서 의사결정 ‘프레이밍’ 시점을 추적하는 temporality reasoning, (2) 임상 대화에서 역할(환자/의사/보호자)을 정확히 구분하는 role attribution, (3) 점수와 일치하는 근거(evidence)를 뽑아 정당화하는 evidence grounding입니다. 저자들은 개발 단계에서 프롬프트/모델을 다듬고 few-shot 기반으로 테스트에 적용한 뒤, 모델별 오답 유형을 체계적으로 분석하며 다중 모델 합의로 disagreement을 해결했습니다.

- **Empirical Impact**: 네덜란드 흑색종(멜라노마) 진료 전사 26개 더블 코딩 데이터에서 일반 도메인 모델이 의학 도메인 모델보다 일관되게 우수했고, 의학 모델은 환각과 instruction-following 실패가 두드러졌습니다. Gemma3:12b가 인간 라벨과 가장 높은 상관을 보였고(Pearson r=0.51, Spearman ρ=0.59), 다만 항목 단위로 난이도 차가 커 SDM 자동 코딩의 완전한 대체는 아직 어렵다는 결론입니다. 그럼에도 로컬 배포로 프라이버시를 지키면서 사람 코더의 속도와 품질 관리를 보조할 “유망한 기반”을 제공했다는 점에서 임상 품질평가 자동화에 의미가 있습니다.



### From Blueprint to Reality: Modeling and Applying Putnam's Social Capital Theory with LLM-based Multi-agent Simulations (https://arxiv.org/abs/2607.06080)
Comments:
          23 pages, 13 figures, 11 tables

- **Prior Approaches**: 기존 Putnam의 Social Capital Theory 연구는 대규모 설문·SEM 같은 정량 분석으로 통찰을 얻지만, 통제와 재현성에 한계가 있다. ABM 등 시뮬레이션은 가정 검증에는 유리하지만 규칙 기반 에이전트로는 인간의 맥락 의사결정, 감정/상황 의존성을 충분히 반영하기 어렵다. 최근 LLM 기반 멀티에이전트는 인간 같은 행동을 만들지만, 이론의 핵심 명제를 직접 겨냥한 theory-driven 환경 구축이나 과정 수준 해석이 부족하다는 문제가 제기된다.

- **Core Contribution**: 본 논문은 SocaSim이라는 LLM 기반 multi-agent 시뮬레이션 프레임워크를 제안해 Social Capital Theory를 ‘이론 설계도 → 시뮬레이션 현실’로 연결한다. social network 진화, trust dynamics, norm propagation을 하나의 환경에 통합하고, 반복적인 collective-action 실험을 통해 Putnam의 세 차원을 동시에 모델링한다. 또 스마트 노인돌봄 적응 문제에 적용해 이론을 실제 의사결정/기술 채택 맥락으로 확장한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 네트워크-신뢰-규범이 시간에 따라 결합해 변하는 동역학을 에이전트로 구현하고 (2) 그 인과 경로를 단계별로 추적 가능하게 만드는 것이다. 이를 위해 에이전트에 SST 기반 구조적 사회 특징(인구/SES/사회자본 성향)을 부여하고, BDI로 라운드별 beliefs-desires-intentions을 추론하며, SCM 메모리·리플렉션으로 학습/갱신을 수행한다. 또한 Proposal–Execution의 2단계 실험으로 공동행동을 수행시키고, 반사실(counterfactual) 개입으로 신뢰 축적과 규범 내재화의 미시 인과 흐름을 process-level로 해석한다.

- **Empirical Impact**: 시뮬레이션은 25라운드 동안 네트워크 밀도 증가와 함께 cooperation 성공률이 동반 상승하는 등 Putnam의 거시 패턴을 재현한다. 특히 20명의 실제 고령자 시나리오 선택과 비교했을 때 그룹 수준 인간-에이전트 정렬이 높게 나타났고(Pearson r=0.974), 신뢰·규범·네트워크의 효과 순위도 일관되게 관측된다. 스마트 노인돌봄 적용에서는 저SES 집단의 초기 trust를 1.0 높이는 반사실 실험으로 채택률이 15.4%p 증가하고, 심리적 압박/불안 및 결정 모순이 각각 19.8%, 22.4%, 25.5% 감소해 ‘정책적 인과 레버’로서 trust의 실용성을 보여준다.



### PluraMath: Extending Mathematical Reasoning Evaluation Beyond High-Resource Languages (https://arxiv.org/abs/2607.05992)
- **Prior Approaches**: 수학적 추론 능력 평가는 reasoning LLM을 튜닝하고 평가하는 핵심 과제로 자리 잡았지만, 기존 벤치마크는 영어·중국어 중심으로 편향되어 고자원 언어 위주로 성능이 측정되는 문제가 있었다. PolyMath 같은 최근 데이터셋이 진전을 보였음에도 18개 고자원 언어로 범위가 제한되어, 언어 다양성을 충분히 반영하지 못한다는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 PolyMath의 확장판인 PluraMath를 제안해 6개 언어 계열에 걸친 18개 추가 저자원/중자원 언어를 포함시키고, 데이터 커버리지를 대폭 넓힌다. 또한 27개 reasoning LLM을 모델 스케일(소형~대형 및 closed-source 앙상블) 전반에서 평가해 다언어 수학적 추론 능력을 체계적으로 비교한다.

- **Technical Challenges**: 핵심 기술 과제는 다양한 언어로 수학 추론 문제의 의미와 난이도를 일관되게 유지하는 데이터 구축이다. 논문은 사람 큐레이션 파이프라인을 통해 원문 기반 번역을 사전 계산한 뒤 원어민이 철저히 검증하는 방식으로 품질을 확보했고, 평가 프레임워크까지 함께 공개해 재현성과 확장성을 높였다.

- **Empirical Impact**: 실험 결과, 고자원 언어와 저자원 언어 간 수학적 추론 성능 격차가 여전히 지속되며, 격차의 상당 부분이 instruction-following 능력과 연관된다는 분석을 제시한다. 더불어 데이터셋·획득 파이프라인·평가 프레임워크를 오픈소스로 공개해, underrepresented communities의 다언어 벤치마크 개발 진입 장벽을 낮추는 데 의미가 있다.



### MemDefrag: Latent Memory Defragmentation for Large Language Models (https://arxiv.org/abs/2607.05969)
- **Prior Approaches**: Latent memory는 과거 지식 조각을 per-layer hidden states로 저장해 장기 기억을 제공하는 방식이며, MemoryLLM, M+ 같은 접근이 이를 활용한다. 하지만 메모리 업데이트 시 positional encoding misalignment과, 어떤 조각이 목표인지 가려주는 tracing 메커니즘 부재로 인해 성능이 크게 떨어진다.

- **Core Contribution**: 이 논문은 target memory fragment를 구분하는 tracing 신호가 실제로 존재함을 보이며, 레이어별 attention density를 분석해 특정 middle transformer layer들이 일관되게 높은 밀도를 target에 집중한다는 점을 발견한다. 이를 바탕으로 훈련 없이 동작하고(model-agnostic) 메모리 조각을 rank·reorder·filter하는 MemDefrag을 제안한다.

- **Technical Challenges**: 핵심 난제는 업데이트 과정에서 기억 위치 정렬 문제로 성능이 붕괴되는 현상을 줄이면서, 저장된 다수 조각 중 목표와 무관한 조각을 효과적으로 걸러내는 tracing을 만드는 것이다. MemDefrag은 middle-layer tracing signal로 메모리를 defragmentation하고, 용량 초과 시 informativeness-guided proportional forgetting으로 유용한 조각의 비중을 유지하도록 설계했다.

- **Empirical Impact**: 실험에서 MemDefrag은 지식 보존에서 MemoryLLM과 M+를 크게 능가하며, 예를 들어 50회 메모리 업데이트 이후 43.0% vs. 17.4%/17.6%의 격차를 보였다. 또한 long-context benchmark에서 성능을 개선하고, 여러 LLM과 latent-memory 변형 전반으로 일반화되는 점에서 분야에 실용적인 방향을 제시한다.



### InfluMatch: Frontier-Quality KOL Search at 4B-Model Cos (https://arxiv.org/abs/2607.05968)
- **Prior Approaches**: 기존 KOL(핵심 의견 리더) 매칭은 키워드 기반 검색이나 정형 속성 필터로 처리되는 경우가 많았지만, 의미 적합도는 놓치고(어휘는 다르지만 내용은 맞는 경우) 캠페인별 다중 조건을 정적 스키마로는 반영하기 어렵습니다. 또한 모든 후보에 대해 frontier LLM을 즉시 추론하는 방식은 정확도는 높아도 지연과 비용이 커 운영에 부담이 됩니다.

- **Core Contribution**: InfluMatch는 태국어의 자유형·다중 파트 마케팅 기준을 받아 KOL을 단계별로 좁힌 뒤 재평가하고, 각 기준별 점수와 태국어 근거를 함께 출력하는 배치 가능한 3단계 캐스케이드( retrieval → rerank → reason )를 제안합니다. 특히 소형 오픈 웨이트 모델만으로도 end-to-end 순위 품질을 확보하면서, frontier 수준의 성능을 저비용으로 노리는 설계를 전면에 둡니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 후보 수가 큰 상태에서 “설명 가능한 판단(reason)”을 비싸게 돌리지 않으면서도, (2) 학습/추론에서 점수 체계가 실제 순위 최적화와 잘 맞도록 하는 것입니다. InfluMatch는 dense retrieval top-50을 만든 뒤 4B pointwise reranker로 top-10만 추려 단일 Yes 토큰 log-prob로 순위를 정하고, 4B reasoner는 기준별 루브릭 채점+태국어 rationale을 생성하도록 하되, fine-tuning은 pairwise SimPO가 end-to-end 전이에 유리하고 reasoner는 untuned base가 가장 강하다는 점을 실험적으로 확인해 배치 설계를 정리했습니다.

- **Empirical Impact**: 실험에서 retrieval-only는 거의 랜덤 수준에 머물렀지만, rerank→reason 전체 캐스케이드는 11개 쿼리 세트에서 P@5 94.1%를 달성하며 frontier 모델 Kimi-K2.6(91.8%)과 비슷한 수준을 저비용으로 따라갑니다. 또한 출력 토큰을 약 35배 줄이고 단일 A100에서 50개 KOL 쿼리를 약 20초 내 처리하는 등 운영 효율이 뚜렷하며, 특히 reasoner의 offline 성능 향상이 end-to-end에서는 역효과가 될 수 있음을 사례로 보여 실전 학습/라벨링 전략에 시사점을 줍니다.



### Umm... With Transformers? Insights from Filled Pause Use across Four Slavic Parliaments (https://arxiv.org/abs/2607.05964)
Comments:
          6 pages, 1 figure. Accepted at InterSpeech 2026. Code published: this https URL

- **Prior Approaches**: 기존 연구는 uh/um 같은 filled pause를 주로 소규모 단일 언어 코퍼스에서 관찰해 왔고, 통계적 검증력과 일반화 가능성이 제한적이라는 한계가 지적돼 왔습니다. 또한 음향·운율 기반 휴리스틱에서 시작해 wav2vec2 계열의 transformer 검출로 발전했지만, 여전히 언어·장르를 가로지르는 대규모 비교와 화자 내/외 변이를 분리한 분석은 부족했습니다. 특히 성별·나이·발화속도 같은 예측 변수의 효과가 대화체에서 관찰된 패턴을 어떻게 다른 담화 도메인(국회 발화)에서 유지하는지 불명확했습니다.

- **Core Contribution**: 이 논문은 4개 슬라브 언어(크로아티아어·체코어·폴란드어·세르비아어) 국회 발화 약 4,000시간을 대상으로 filled pause를 대규모 교차언어로 분석합니다. transformer 기반 자동 검출로 FP를 식별한 뒤, Mundlak-corrected GEE를 사용해 화자 간(안정적 성향) 효과와 발화 단위(상황 변화) 효과를 분해합니다. 그 결과 성별·정서·정치 성향·권력 상태가 FP와 맺는 관계가 언어/도메인별로 달라질 수 있음을 체계적으로 보여줍니다.

- **Technical Challenges**: 핵심 기술 과제는 대규모 자동 FP 라벨링의 신뢰성과, 반복 발화 구조(발화가 화자/회의 맥락에 중첩됨)에서 상관 구조를 과도하게 가정하지 않고 효과를 해석하는 것이었습니다. 이들은 Negative Binomial로 FP 빈도를 모델링하고, 시간 길이를 offset으로 넣어 발화 당 ‘비율’ 관점을 유지했습니다. 아울러 GEE의 샌드위치 표준오차와 Mundlak 분해로 within-speaker(발화 편차)와 between-speaker(화자 평균)를 분리해, 누락된 화자 고정 특성에서 오는 편향을 줄였습니다.

- **Empirical Impact**: 실증적으로는 나이와 발화속도에 대해 기존에 알려진 방향(나이는 FP 감소, 발화속도는 느릴수록 FP 증가)을 재현했습니다. 반면 성별 효과는 언어별로 뒤집혀(남성이 적게, 또는 특정 언어에서만 유의) ‘대화체에서의 보편적 패턴’이 국회 도메인에서는 그대로 통하지 않을 수 있음을 시사합니다. 새 예측 변수로서 sentiment(정서)는 화자 내에서 일관되게 FP 증가와 양의 연관을 보였고(화자가 자신의 기준선보다 더 긍정적일 때 FP가 늘어남), opposition(야당)이 governing coalition(여당 연정)보다 FP가 낮다는 국회 전반의 경향도 관찰됐습니다. 결과적으로 small study에서의 효과가 실제로는 언어·도메인 의존일 수 있음을, 그리고 within-speaker 분해가 해석의 핵심임을 보여주는 사례로 주목됩니다.



### Is Domain Adaptation Always Helpful? A Frozen-Backbone Study of Cross-Domain Sentiment Transfer (https://arxiv.org/abs/2607.05937)
- **Prior Approaches**: 기존 감성분석 연구는 BERT 계열 PLM의 임베딩을 고정(frozen)하고 가벼운 분류 헤드를 얹어 전이하는 probing/transfer 패러다임을 널리 사용해왔다. 또한 DANN처럼 적대적 정렬로 도메인 불변 표현을 만들거나, MMD처럼 분포 매칭 손실로 source–target 간 거리를 줄이는 도메인 적응 UDA가 많지만, 이들 대부분은 백본 미세조정까지 포함해 ‘적응 효과 vs 백본 변화’를 분리하기 어렵다는 한계가 있다. 더구나 소스보다 타깃이 이미 포함되었을 가능성이 있는 벤치마크 설정에서는 성능이 진짜 일반화인지 데이터 오염/누출인지 구분이 흐려진다.

- **Core Contribution**: 이 논문은 “냉동(frozen) 백본에서 도메인 적응이 언제 도움이 되는가”를 보기 위해, Qwen3-Embedding(0.6B/4B/8B) 같은 스케일이 통제된 임베딩 계열과 RoBERTa-base, FinBERT를 함께 사용해 감성 전이를 비교한다. 백본은 완전히 고정하고 MLP adapter만 학습하며, source는 라벨을 쓰고 target은 라벨 없이 DANN/MMD/SCL 목적함수로 정렬·개선하되 target 라벨은 사용하지 않는다. 그 결과 SST-2(가까운 도메인)와 Financial PhraseBank(큰 도메인 이동)에서 서로 다른 전이 양상이 나타남을 체계적으로 보여준다.

- **Technical Challenges**: 핵심 기술적 난제는 타깃 라벨 없이도 adapter가 도메인 갭을 실제로 메울지, 아니면 클래스 경계를 망치거나 이미 내재된 도메인 구조를 침식할지 예측하기 어렵다는 점이다. 이를 위해 백본 드리프트를 제거하려고 임베딩을 사전 계산·캐시하고 stop-gradient로 정렬 손실이 백본에 영향을 주지 않게 설계했으며, GRL의 초기 불안정성을 줄이기 위해 DANN 스케일을 워밍업한다. 또한 SCL은 supervised contrastive 형태로 동클래스는 가깝게, 타클래스는 멀게 만들며, pseudo-label은 5 epoch 중 후반(4–5 epoch)과 높은 신뢰도(≥0.95) 조건으로만 보수적으로 주입해 confirmation bias를 완화한다.

- **Empirical Impact**: 예비 실험(전이 평가 세트 크기 제한 포함)에서 SST-2는 모든 백본이 0.85~0.91 사이로 강한 성능을 보이고, DANN/MMD/SCL 조합을 더해도 이득이 거의 없어 도메인 적응의 필요성이 낮게 관측됐다. 반면 Financial PhraseBank에서는 소형 일반 백본(예: 0.6B)이 no-DA에서 0.309 F1로 크게 밀리지만 DANN+MMD로 0.637까지 복구되는 등 도메인 적응이 의미 있게 작동했다(단일 실행 기준 최대 +32.8 macro F1 포인트). 특히 FinBERT 같은 도메인 특화 백본에서는 DANN이 오히려 -0.106 macro F1로 성능을 떨어뜨리는 반면, supervised contrastive loss가 가장 크게 개선(+0.076, 관측 최고 0.978)되어 “백본의 타깃 커버리지 유무”에 따라 적응 목표 선택이 달라져야 한다는 메시지를 제시한다.



### Mitigating Factual Hallucination in Large Reasoning Models via Mixed-Mode Advantage Regularization (https://arxiv.org/abs/2607.05861)
Comments:
          19 pages, 3 figures, 8 tables

- **Prior Approaches**: 기존 LRM(대규모 추론 모델)은 정답을 말하기 전에 생각의 흔적(thinking trace)을 생성해 성능을 끌어올리는 방식으로 주로 연구돼 왔다. 특히 사실 중심 질의응답(factuality-oriented QA)에서는 이러한 명시적 사고가 관련 지식을 회복하고 답을 정제해 전반적 정확도를 높인다고 알려져 있다. 다만 인스턴스마다 효과가 균일하지 않으며, 올바른 비(非)사고 답을 사고가 뒤집어 사실 드리프트를 유발할 수 있다는 문제가 지적된다.

- **Core Contribution**: 이 논문은 명시적 thinking이 정답을 망가뜨리는 실패 모드를 thinking-induced hallucination으로 정의하고, 이를 “직접 답변 성향(direct-answer tendency)”에 대한 thinking residual로 모델링해 설명한다. 그 결과, residual이 누락 지식을 회복할 수도 있지만 근거 없는 연관을 만들어 환각을 유발할 수도 있음을 체계적으로 정리한다. 이를 바탕으로 MARGO(Mixed-Mode Advantage Regularization for Grounded Optimization)를 제안하며, thinking의 가치가 직접 답변 대비 추가되는지 RL로 판별해 유해한 thinking을 억제한다.

- **Technical Challenges**: 핵심 기술적 난제는 “좋은 thinking”은 유지하되 “나쁜 thinking”이 이미 맞는 답을 뒤집지 않게 만드는 신호 설계다. 저자들은 advantage 추정에 non-thinking rollouts를 같은 모델의 기준(reference)으로 포함시키는 강화학습 프레임워크를 구성해, mixed-mode rollout group에서 thinking/비thinking의 기여를 비교한다. 이렇게 하면 thinking이 사실적으로 이득이 있을 때만 강화되고, 근거 없는 잇기를 만드는 경향은 약화된다.

- **Empirical Impact**: 여러 factuality-oriented QA 벤치마크에서 MARGO는 강력한 기준선 대비 사실 신뢰성(factual reliability)을 개선하는 것으로 나타났다. 또한 수학 벤치마크 평가에서는 일반적인 추론 능력을 보존해, 사실성 최적화가 추론 전반을 해치지 않음을 보여준다. 전반적으로 “추론 흔적”의 유용성을 인스턴스 단위로 재정렬하는 접근이라는 점에서, fact QA의 안정성을 높이는 데 의미가 크다.



### CoPiT: Cognitive Pivot Translation for Digraphic Low-Resource Mongolian in the Traditional Scrip (https://arxiv.org/abs/2607.05849)
Comments:
          Preprint

- **Prior Approaches**: 기존 저자원 언어 MT는 웹 마이닝, back-translation, LLM 합성데이터 같은 데이터 중심 접근과, multilingual 사전학습 기반 모델·전사/형태소 토크나이징 같은 모델 중심 접근으로 나뉜다. 다만 몽골어처럼 두 문자가 동시에 쓰이는 digraphy(다중 문자 체계)에서는 스크립트(문자 체계)에서 기인한 모호성이 의미 전달 문제로 직접 전이되기 쉬워, 단순 direct translation 성능이 크게 떨어진다. 특히 Traditional 몽골어는 철자적으로 underspecified되어 가능한 해석이 여럿이라 기존 방법이 이를 구조적으로 분리·해결하지 못한다.

- **Core Contribution**: 이 논문은 CoPiT(Cognitive Pivot Translation)라는 인지 동기 다단계 파이프라인을 제안한다. Traditional 몽골어의 스크립트 유도 모호성을 먼저 Cyrillic(중간 표현)에서 명시적으로 해소한 뒤, 그 결과를 목표 언어로 번역해 의미 전달을 안정화한다. 또한 Traditional 텍스트에서 출발하는 방식으로 합성 병렬데이터를 생성할 수 있어, 단순 추론 성능을 넘어 저자원 데이터 부족을 완화한다.

- **Technical Challenges**: 핵심 기술 과제는 Traditional 스크립트의 철자·음운 정보 부족으로 생기는 다중 해석을 번역 전에 일관되게 좁히는 것이다. 저자들은 (1) 형태소 경계 모호성 대응을 위한 segmentation, (2) vowel harmony recovery로 음운 제약을 강제, (3) Latin-assisted normalization으로 시각적으로 애매한 중간 음운 표지를 명시화, (4) Cyrillic normalization으로 정규 형태를 선택하고, (5) 문장 단위 self-reflection으로 로컬 선택의 비일관성을 교정한다. 각 구성요소는 분리 학습(단어 수준 감독 데이터 + 문장 수준 합성 revision 쌍)으로 정확히 역할을 분담하도록 설계했다.

- **Empirical Impact**: 다양한 백본(Qwen-3, Ministral-3 등)과 목표 언어(영어·한국어·러시아)에서 CoPiT는 direct translation 대비 BLEU 절대 개선과 COMET에서 일관된 이득(약 1.5–1.6x COMET)을 보였다. fine-tuning된 오픈소스 CoPiT 모델은 동일 평가 조건에서 GPT-4.1 direct-translation과 비슷하거나 더 나은 성능도 달성했다. 또한 CoPiT로 생성한 8,034개 합성 병렬쌍은 forward 번역 성능 개선뿐 아니라 reverse-direction(목표→Traditional) 번역에도 유효함을 보여, digraphic 저자원 MT의 실무적 확장성을 입증했다.



### Inject or Navigate? Token-Efficient Retrieval for LLM Analysis of Transactional Legal Documents (https://arxiv.org/abs/2607.05764)
Comments:
          17 pages, 2 figures, 8 tables

- **Prior Approaches**: 기존 법률 문서 Q&A는 매 질의마다 전체 문서 코퍼스를 LLM 컨텍스트에 주입하는 방식(inject)이 가장 단순하고, 재검색 누락을 피한다는 장점이 있다. 하지만 코퍼스가 커질수록 토큰 부담과 long-context 성능 저하가 함께 커져 비용/품질이 비선형으로 악화된다. 또한 벡터 기반 RAG는 cosine 유사도로 의미만 맞추기 쉬워, 정의/교차인용/개정선후관계처럼 구조 의존성이 강한 계약 문서의 검색에는 한계가 반복적으로 지적돼 왔다.

- **Core Contribution**: 이 논문은 법률 분석 시스템에서 full-corpus injection을 구조 기반 retrieval로 대체하기 위해, Syntheia의 structure-aware chunking을 고정한 채 retrieval 전략만 비교·평가했다. 두 대안은 NAVEMBED(임베딩 검색+reranking)와 NAVINDEX(LLM이 compact structured index를 탐색하도록 한 구조 탐색)이며, 품질 저하 없이 토큰·달러 비용을 줄이는 설계 기준을 제시한다. 특히 NAVINDEX는 정의 term 그래프와 cross-reference 그래프 등 구조 신호를 인덱스 표면에 명시적으로 인코딩해 벡터 검색의 표현 한계를 보완한다.

- **Technical Challenges**: 핵심 난제는 (1) 관련 조항이 코퍼스 전역에 분산돼 있어 중간 위치 정보가 희석되는 long-context 문제와 (2) 프롬프트 caching이 비용 축을 바꾸더라도 모델이 실제로 attend하는 토큰 축은 그대로 남는 점을 동시에 통제하는 것이다. 저자들은 token footprint(모델이 주목하는 입력 토큰)과 dollar cost(캐시 할인/세션 경제 포함)를 분리해 추적하고, reference-anchored pairwise judge로 위치 편향을 통제한 평가 프로토콜을 구성했다. NAVEMBED는 reranking 파이프라인으로 작업 셋을 줄였고, NAVINDEX는 *.index.json(구조 메타·요약·그래프)과 *.full.json(원문 조항 저장)을 나눠 질의당 최대 10개 노드만 fetch하도록 하드 캡을 적용했다.

- **Empirical Impact**: 20개 질문(문서 18개 bound, 2개 out-of-scope 통제) 벤치마크에서 NAVEMBED는 문서 bound 18개 중 16개에서 inject와 동률 수준의 품질을 보였고, 관련 없는 2개 out-of-scope에서도 둘 다 동률로 판정됐다. 입력 토큰은 inject 대비 17.3x(최적 GTE 구성은 29.9x) 적었으며, 비용 비교에서도 탐색 기반 모드들이 유리한 구간이 확인됐다. NAVINDEX는 18개 모두 동률로 평가되면서도 total token footprint은 1.61x 줄고 answering context는 약 56x 줄였으며 달러 비용도 약 25% 낮았고, cached injection이 유리해지는 조건을 코퍼스 크기 기준의 closed-form caching-crossover rule로 정리해 운영 의사결정에 직접 연결된다.



### When Should LLMs Search? Counterfactual Supervision for Search Routing (https://arxiv.org/abs/2607.05752)
Comments:
          20 pages, 10 figures. Accepted at the FAGEN Workshop at ICML 2026

- **Prior Approaches**: 검색을 내장한 언어모델은 외부 근거로 장꼬리 지식 등을 보완할 수 있지만, 모든 질문에 검색이 항상 이득은 아니다. 기존 연구는 검색이 유용한 경우를 찾거나(선택적 retrieval, confidence/복잡도 기반 트리거) tool 호출 자체의 정확도를 평가하는 데 초점이 있었고, 언제-not-to-call을 인스턴스 단위 성공 관점으로 직접 학습하기는 어려웠다.

- **Core Contribution**: 이 논문은 “검색 필요 여부”를 인스턴스 레벨의 search-routing 문제(NO_SEARCH vs SEARCH)로 정식화하고, 같은 질문에 대해 no-search와 forced-search의 결과를 비교해 outcome-based oracle을 만든다. 이 오라클을 평가 기준뿐 아니라 학습 신호로도 사용해, 필요한 경우에만 검색으로 경로를 바꾸도록 SFT와 Preference Optimization을 함께 학습한다.

- **Technical Challenges**: 핵심 난점은 검색 유용성을 사람이 라벨링하지 않고도 일관된 감독을 구성하는 동시에, 오라클에서 제외되는 UNSOLVED(둘 다 실패)처럼 원인이 섞인 케이스를 학습에 잘못 끌어오지 않는 것이다. 저자들은 no-search/forced-search의 페어 결과로 안정적인 라우팅 타깃만 구성하고, UNSOLVED는 진단 서브셋으로 남긴 채 first-action(첫 턴) 의사결정만 최적화하도록 학습 데이터를 설계했다.

- **Empirical Impact**: PopQA와 KUQ(거짓 전제/모호성)에서 모델별로 search 경계가 다르며, 학습 전에는 over-search와 under-search가 동시에 나타남을 보여준다. 오라클-eligible 예제에서 Gemma E2B는 macro-F1이 0.7082→0.8235, Qwen3.5-4B는 0.7053→0.8365로 개선됐고, 분석에서는 UNSOLVED 잔여 오류가 모델 용량, retrieval budget, 근거 활용, 정책 행동 등 서로 다른 병목임을 분해해 준다.



### Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding (https://arxiv.org/abs/2607.05722)
- **Prior Approaches**: 기존 AR(autoregressive) LM은 토큰을 순차 생성해 추론 병렬성이 낮고, 특히 저배치·다중 동시성 환경에서 처리량이 병목이 되기 쉽다. 반면 diffusion LM은 여러 토큰을 한 forward pass에 생성해 병렬성을 높였지만, 학습 데이터 효율과 정확도에서 AR 대비 불리하거나 MTP(multi-token prediction)와의 효율-정확도 균형이 약한 경우가 많았다.

- **Core Contribution**: Nemotron-Labs-Diffusion는 하나의 아키텍처에서 AR, diffusion, self-speculation(자기 추론 추첨) 디코딩을 tri-mode로 전환하는 모델을 제안한다. AR과 diffusion 손실을 함께 학습해, diffusion의 룩어헤드(lookahead) 계획 이점을 유지하면서도 AR이 제공하는 left-to-right 언어 사전분포를 보존한다.

- **Technical Challenges**: 핵심 기술 과제는 diffusion 학습이 토큰 순열을 균일하게 다루며 left-to-right 귀납편향을 충분히 활용하지 못하는 문제를 해결하는 것이다. 이를 위해 블록 단위 diffusion denoising과 AR next-token loss를 가중 결합하고, AR로 먼저 선학습한 뒤 joint objective로 전환하는 2-stage 학습을 사용하며, 마스킹으로 생기는 토큰 수 변동에 영향을 덜 받도록 global loss averaging을 도입했다; 또한 diffusion drafts를 AR verifier가 검증하는 self-speculation 디코딩과 이를 LoRA 기반 drafter로 보강해 acceptance rate와 실제 효율을 개선했다.

- **Empirical Impact**: 실험에서 tri-mode 모델은 다양한 코딩·수학 벤치마크에서 기존 오픈소스 AR/diffusion 모델보다 정확도와 추론 속도를 동시에 끌어올렸고, 예로 Nemotron-Labs-Diffusion-8B는 Qwen3-8B 대비 forward당 토큰을 약 6배 더 생성하면서 비슷하거나 더 나은 정확도를 보였다. SPEED-Bench에서는 GB200 GPU + SGLang 환경에서 처리량이 4배 수준으로 향상되었고, speed-of-light 분석과 self-speculation 대비 수치(예: optimal sampler 가정에서 forward당 토큰 증가)가 diffusion 디코딩의 장기적 여력도 함께 시사한다.



### SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation (https://arxiv.org/abs/2607.05721)
Comments:
          The project page is available at this https URL

- **Prior Approaches**: 기존 불확실성 추정은 토큰 수준과 시퀀스 수준의 양극단에 머물렀다. 토큰별 점수는 의미 응집이 깨지고, 시퀀스 점수는 오류가 어디서 생기는지 국소화하지 못한다. 또한 샘플링 기반 시퀀스 불확실성은 다중 생성·검증 때문에 비용이 커 10–20배 이상 느리다.

- **Core Contribution**: 이 논문은 Span-Level Uncertainty Estimation (SLUE)를 새 과제로 정식화하며, 의미 단위가 되는 “span(연속 구간)”마다 연속적인 불확실성을 매기는 것을 목표로 한다. 이를 위해 단일 forward pass로 다중 샘플에서 얻은 불확실성 지식을 증류하는 lightweight probe인 SpanUQ를 제안한다. 더불어 SpanUQ-Bench(20K 프롬프트, 293K span, continuous soft label)를 구축해 span 단위 평가의 공백을 메운다.

- **Technical Challenges**: 핵심 난제는 (1) span을 의미적으로 경계 지어 가변 길이로 찾아내면서 (2) 각 span의 불확실성을 연속값으로 정교하게 추정하고 (3) 이 모든 것을 단일 패스로 끝내는 것이다. 저자들은 DETR 스타일 span decoder로 span을 집합 예측하고, Mixture of Beta로 불확실성을 모델링한 뒤 Beta NLL 회귀와 대조적 ranking을 결합해 학습한다. 추가로 Uncertainty-Conditioned Iterative Refinement(UCIR)으로 1회 추정의 거칠음을 한 번 더 다듬어, 샘플링 없이도 정밀도·경계 품질을 끌어올린다.

- **Empirical Impact**: 실험에서 SpanUQ는 5개 LLM 백본(Qwen3-14B 포함)에서 span-level 불확실성 품질이 일관되게 최상이며, sampling-based 방법보다 10–20배 빠르다. DETR 기반 span 검출은 0.910 F1을 달성해 휴리스틱 대비 39.4%p 향상되며, 시퀀스 수준 방법이 제공하지 못하는 오류 위치(어떤 span이 문제인지)를 구체화한다. 또한 span-level 추정이 시퀀스 수준 불확실성을 부분적으로 분해·포괄할 수 있음을 보이며, 내부 표현만으로 연속 불확실성을 학습하는 접근의 확장성을 보여준다.



### Where to cut, how deep: BPE and Unigram-LM on chemistry SMILES (https://arxiv.org/abs/2607.05691)
- **Prior Approaches**: 기존 화학 SMILES 토크나이저 연구는 주로 coverage(UNK 처리)나 다운스트림 성능 차이에 초점이 맞춰져, BPE 같은 전통적 서브워드 알고리즘 자체가 만들어내는 구조적 차이를 고립해 검증하지 못했다. 또한 자연어에서의 BPE vs Unigram-LM 차이가 화학 영역의 작은 알파벳/강한 구조 제약에도 그대로 나타나는지는 불명확했다. 일부 선행 비교는 베이스(고정된 화학 알파벳)와 경계 정책, 그리고 작은 어휘 크기(임베딩이 실제로 학습 가능한 구간)를 통제하지 못했다.

- **Core Contribution**: 이 논문은 고정된 165-token 화학 베이스 위에서 BPE와 Unigram-LM을 알고리즘 외 모든 조건(코퍼스 유형, 어휘 크기, pre-tokenization boundary 정책)까지 맞춘 통제 실험으로 비교한다. 그 결과 두 알고리즘은 수렴하지 않고, 22개 매칭 조건 전부에서 거의 서로 다른 서브워드 어휘(낮은 교집합)와 뚜렷한 세분화/토큰 분포 차이를 만든다. 저자들은 이 차이가 단순한 기본값이 아니라 ‘모델링 결정’에 해당함을 명확히 한다.

- **Technical Challenges**: 핵심 과제는 (1) 토큰 커버리지의 혼입 없이, (2) 알고리즘 선택만 바꿔 tokenizer-level 차이를 측정하고, (3) 작은 어휘 크기에서 임베딩 학습 가능성(undertrained dead-zone)을 확인하는 것이다. 이를 위해 Smirk의 165-token glyph 베이스로 각 알고리즘을 동일 pre-tokenization 흐름에 올려 학습하고, learnability bar를 통과한 조건만 비교했으며, boundary가 합쳐지는지/막히는지(NMB vs MB)까지 체계적으로 스위핑한다.

- **Empirical Impact**: 실험에선 알고리즘 간 learned piece 교집합이 거의 생기지 않으며(Jaccard가 매우 낮음), Unigram-LM이 held-out 분자에 대해 더 많은 토큰으로 세분화(더 미세한 분절)를 수행하되 BPE는 대부분을 ‘엄격한 coarsening’으로 재현한다. 또한 토큰 사용의 불균형과 dead-zone surplus, whole-pretoken absorption 같은 메커니즘 진단에서도 알고리즘 쌍의 차이가 관측되며, 이 분리는 어휘 크기를 8배 이상 키워도 유지된다. 단, 본 연구는 언어 모델을 학습하지 않아 ‘어느 알고리즘이 더 좋다’는 결론이 아니라, 화학 SMILES에서 서브워드 알고리즘이 교체 불가능한 전처리/표상 선택임을 실증적으로 확립한다.



### UCSC NLP at SemEval-2026 Task 10: Boundary-Aware Span Extraction and RoBERTa Classification for Conspiracy Detection (https://arxiv.org/abs/2607.05689)
Comments:
          6 pages, 2 tables. System description paper for SemEval-2026 Task 10 (PsyCoMark: Psycholinguistic Conspiracy Marker Extraction and Detection)

- **Prior Approaches**: 기존 PsyCoMark 접근은 LIWC 같은 psycholinguistic 특징이나 감정·도덕적 프레이밍을 활용해 문서 수준의 음모성 여부를 간접적으로 추정하는 방식이 많았다. 또한 문서 분류를 위한 fine-tuning은 시도됐지만, Actor–Action–Victim–Evidence–Effect 같은 내러티브 역할 관계를 명시적으로 경계 단위로 다루는 데는 한계가 컸다. span 기반 정보추출 기법이 변수 길이 구조를 다룰 수 있음에도, 음모 역할에서 특히 어려운 “추상 역할의 경계”를 목표 지표와 가깝게 학습시키는 설계는 부족했다.

- **Core Contribution**: 이 논문은 PsyCoMark를 두 축으로 나눠 각각의 문제에 맞춘 독립 모델을 제안한다. Subtask 1은 후보 span을 열거해 multi-label span classification을 수행하고, IoU 기준 라벨링·hard-negative·containment 기반 NMS로 역할 마커의 정밀 경계를 학습/복원한다. Subtask 2는 RoBERTa-large의 [CLS] 임베딩만으로 문서 수준 Yes/No/Can’t tell을 분류하되 label smoothing과 stratified split으로 클래스 불균형에 대응한다.

- **Technical Challenges**: 핵심 난제는 Action, Effect, Evidence처럼 의미적으로 추상적이며 길이가 크게 변하는 역할의 “정확한 경계”를 잡아내는 것이다. 이를 위해 IoU>=0.95일 때만 양성 라벨을 주는 매우 엄격한 학습 조건을 사용하고, 50–75% 겹침의 hard negatives로 near-miss 오분류를 적극적으로 학습시켰다. 또한 decoding 단계에서 thresholding과 role-specific containment-based NMS 및 인접 span merging을 적용해 중복·중첩 예측을 줄였다.

- **Empirical Impact**: 공식 테스트에서 Subtask 1은 macro F1 0.2251로 7위, Subtask 2는 weighted F1 0.7694로 11위권 성과를 보였다. 분석 결과 Actor/Victim 같은 entity-like 역할은 비교적 견고했지만 Action/Effect/Evidence는 경계 기준에 민감해 성능이 떨어지는 경향이 확인됐다. 특히 검증 시 완화된 IoU와 테스트의 더 엄격한 token-level IoU 사이의 차이가 성능 격차를 키웠고, 향후 shared-task 평가 프로토콜 정렬의 중요성이 부각됐다.



### RPAM: A Principled Metric for Evaluating Associations in Language Models with High Predictive Validity in Downstream Outputs (https://arxiv.org/abs/2607.05679)
Comments:
          14 pages

- **Prior Approaches**: 기존 생성형 LM 편향 분석은 주로 생성된 텍스트의 결과(다운스트림)에서 연관성을 측정하는 방식이 많았다. 하지만 생성 문장은 모델마다 크게 달라 특화된 평가 데이터셋이 필요해져 다른 LM으로의 일반화가 제한된다. 한편 업스트림 평가는 임베딩이나 continuation probability 같은 기반 신호를 보지만, 기존 업스트림 지표가 실제 생성 텍스트에서 관측되는 연관성과 강하게 맞물린다는 근거는 부족했다.

- **Core Contribution**: 이 논문은 생성형 LM의 연관성(association)을 업스트림에서 평가하는 Relative Probability Association Metric(RPAM)을 제안한다. RPAM은 두 자극 간 상대적 연관성을 softmax로 정규화해, 텍스트 생성/디코딩과 무관하게 개념-속성 연관을 정량화한다. 또한 RPAM 측정치가 인간의 암묵·명시 연관성과 생성 텍스트의 다운스트림 편향 측정까지 연결되는지를 검증하는 평가 프레임워크를 함께 제시한다.

- **Technical Challenges**: 핵심 과제는 업스트림 신호로부터 얻은 연관성이 실제로는 다운스트림 행동(생성 결과)과 얼마나 일치하는지 입증하는 것이었다. 이를 위해 RPAM은 타깃을 템플릿에 삽입한 뒤, 속성 단어들에 대한 continuation probability를 계산하고 특정 속성 집합 내에서 상대적으로 재정규화해 비교 가능성을 확보했다. 또한 n-gram/문장 같은 다양한 타깃 표현에 맞춘 서로 다른 템플릿을 사용하고, WEAT/SC-WEAT 계열을 포괄하는 형태로 암묵 연관성과 valence(유쾌/불쾌, 감정)까지 확장했다.

- **Empirical Impact**: 검증 실험에서 RPAM은 WEAT-WS 기반 암묵 연관성 10개를 재현했으며, 인간의 명시적 연관성(WS-353, Bellezza, SST2)과도 높은 상관/분류 성능을 보였다. 특히 다운스트림 작업과의 일치성 실험에서는 Mistral/GPT-2 계열에서 Spearman’s ρ=0.73 및 SST2 F1 0.74 이상 같은 결과로 생성 텍스트 기반 편향 신호를 잘 반영했다. 업스트림 지표로서의 실용성과 일반화 가능성을 보여준 만큼, 앞으로 편향 완화·규제 대응을 위한 평가 표준으로 활용될 잠재력이 크다.



### Do It Right! A Methodology for Successful NLP System Developmen (https://arxiv.org/abs/2607.05644)
Comments:
          Pre-submission draft

- **Prior Approaches**: 기존 임상 NLP는 개별 알고리즘(구문분석, 개체명 인식, 의미 역할 라벨링 등) 중심으로 학습 자료가 구성돼 있어, 실제 정보추출 시스템을 “프로젝트로” 관리하는 관점이 약했다. 또한 LLM 이후에는 API로 쉽게 구조화된 결과를 얻는 것처럼 보이지만, 실제로는 소스에 없는 내용을 그럴듯하게 생성(환각)하거나 누락, 프롬프트 표현에 따른 불일치가 생길 수 있다. 즉 성패는 모델 성능뿐 아니라 요구사항 정의·검증·변경관리 같은 소프트웨어 공학 위험 관리에 달려 있다.

- **Core Contribution**: 이 논문은 임상 임상기록에서 언어처리로 정보를 추출하는 NLP 프로젝트에 대해 Systems Development Life Cycle(SDLC) 단계별 절차를 제시한다. 특히 텍스트 생성(summarization) 같은 과제는 제외하고, 정보추출(information extraction)에 맞춰 Planning–Analysis–Design–Implementation–Testing–Deployment–Maintenance의 흐름을 구체화한다. “성공적인 추출”을 위해 SDLC를 통해 실패 가능성을 체계적으로 낮추려는 실무 지침을 제공한다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 관심 개념이 실제 임상 텍스트에 존재하는지, (2) 해당 내용을 충분한 정확도로 추출 가능한지의 타당성 검증이며, LLM에서는 환각 때문에 일반적인 벤치마크를 신뢰하기 어렵다. 논문은 이를 해결하기 위해 대표 코퍼스에 대한 수동 확인, 개념 시트(concept sheet)로 정의·단위·값 범위를 고정, 그리고 semantic ambiguity(용어가 여러 개념에 매핑)·contextual ambiguity(주장/시점/경험 주체 등)를 고려한 설계와 검토 전략을 권한다. 또한 문서 선택(비용 절감용 프리필터링, 계층화 표집, 표본 크기)과 주석 설계(가이드라인·포함/제외 기준, 참조표준이 시스템이 보는 데이터와 일치하도록 제한)를 SDLC에 통합해 재현성과 해석 가능성을 높인다.

- **Empirical Impact**: 이 글은 특정 단일 성능수치보다는 임상 정보추출 프로젝트가 실패하는 전형적 원인을 SDLC로 흡수할 수 있음을 강조하며, LLM 기반일수록 ‘잘되는 것처럼 보이는 착시’를 줄이는 절차적 가치가 크다고 설명한다. 실무적으로는 개념 정의 드리프트, 변경통제 부재, 위험(Feasibility/accuracy/cost) 과소평가 같은 요인을 문서화와 합의 기반 검토로 관리하게 만든다. 또한 고급 기저세포암 같은 사례에서처럼 완전 자동화가 어려운 경우 ‘고리콜 NLP 선별 후 수동 해석’ 또는 구조화 대체까지 선택지를 제공함으로써, 팀이 데이터·도메인 특성에 맞게 검증 가능한 개발 경로를 잡는 데 의미가 있다.



### Population-Level Profiling of DSM-5 Depressive Symptoms Among Self-Reported ADHD and ASD Users on Twitter: An Exploratory Study Using Advanced NLP and Statistical Analysis (https://arxiv.org/abs/2607.05626)
- **Prior Approaches**: 기존 연구들은 ADHD·ASD 집단에서 우울 증상이 더 흔하다는 유병률 관점이나, 텍스트의 어휘·주제 차이 중심으로 온라인 표현을 분석하는 경우가 많았습니다. 또한 DSM-5의 9개 우울 증상을 동시에(다중라벨) 체계적으로 매핑해 두 질환의 차이를 “구조적으로” 비교한 인구집단 증거는 상대적으로 부족했습니다. 더불어 zero-shot 기반 우울 관련 전처리 임계값에 따른 결과 민감성이 충분히 다뤄지지 않은 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Twitter의 self-reported ADHD(622명) vs ASD(170명) 사용자를 대상으로, DSM-5 우울 9개 증상을 다중라벨 분류해 집단 수준에서 증상별 “강조 정도” 차이를 비교합니다. zero-shot NLI로 우울 관련 tweet을 선별한 뒤, MentalRoBERTa(mental-health 도메인 적응)로 9개 증상을 분류하고, 사용자별 평균을 기준으로 중심화해 질환 간 상대적 패턴만 보려 했습니다. 또한 전처리 임계값을 5개(0.45~0.65)로 스윕하며 재현성 관점을 포함해, 단일 컷오프에 의존하지 않도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 우울 관련 tweet 선별 단계의 임계값 선택에 따른 데이터 누락/오염, (2) 증상별 언어 표현 차이와 클래스 불균형, (3) 미세한 집단 차이를 L1 로지스틱에서 안정적으로 추정하는 점입니다. 저자들은 Stage 1을 recall-first로 넓게 통과시키고, per-label decision threshold 캘리브레이션과 per-user quality gate(게이팅된 tweet 30개 이상)를 추가해 정밀도를 확보했습니다. 더 나아가 부트스트랩(1000회) 기반 선택 안정성과 임계값 전반의 방향 일관성을 결합한 graded robustness 체계로, “강건한 방향”만 해석하도록 했습니다.

- **Empirical Impact**: MentalRoBERTa 다중라벨 분류기는 held-out에서 macro-F1 0.901로 ReDSM5의 기존 기준선을 크게 상회해, 이후 비교를 위한 측정 도구로서 성능 타당성을 확보했습니다. 다만 ADHD vs ASD 구분은 ROC-AUC 0.645~0.653으로 modest했으며, 증상별로는 cognitive issues, sleep issues, appetite change, fatigue는 ADHD 쪽으로, suicidal ideation과 anhedonia는 ASD 쪽으로 기울었지만 disorder-specific으로 “견고하게” 분리되는 짝은 나오지 않았습니다. 증상 공동출현 구조도 두 집단에서 largely shared로 나타나, 결론은 임상적 현상학 차이를 개인 수준에서 증명하기보다는 “재현성 있는 탐색적 인구집단 차이”에 가깝다는 점을 강조합니다.



### NAVER LABS System Re-implementation for the IWSLT 2026 Instruction-Following Task (https://arxiv.org/abs/2607.05623)
- **Prior Approaches**: 기존 멀티모달 speech LLM들은 고정된 speech encoder와 instruction-tuned LLM을 lightweight connector로 연결해 프롬프트 기반으로 여러 작업을 처리합니다. 다만 NAVER LABS 2025 파이프라인은 성능은 경쟁적이었지만, IWSLT 2026 과제 조건에 맞춘 공개 재현이 어려웠습니다.

- **Core Contribution**: 본 논문은 NAVER LABS 2025의 3단계 instruction-following 파이프라인을 IWSLT 2026 constrained/short audio 조건에 맞춰 처음으로 open-source로 재구현했습니다. 또한 speech encoder를 SeamlessM4T-v2-large로, LLM 백본을 Qwen3-4B-Instruct로 교체하고, Stage 3 고도화를 위한 100k 합성 instruction-following 데이터셋을 구축해 후속 fine-tuning 발판을 제공합니다.

- **Technical Challenges**: 핵심 난점은 (1) frozen speech encoder 임베딩을 LLM hidden size에 맞게 정렬하는 projector 학습과 (2) text-only LoRA pre-training 이후 멀티모달 병합에서의 성능 저하를 막는 것입니다. 이를 위해 projector alignment→text-only LoRA pre-training→multimodal merging의 3단계를 유지하되, 발화 길이 제약(15초) 내에서 오디오/텍스트 배치를 즉시 교차해 catastrophic forgetting을 줄였고, LoRA rank·learning rate 설정을 ablation으로 최적화했습니다.

- **Empirical Impact**: MCIF(제한 조건, short audio)에서 최종 모델은 EN→ZH speech translation에 COMET 0.781, English SQA에 BERTScore-F1 0.346을 기록하며 projector-only 기준선 대비 일관된 개선을 보였습니다. 특히 Stage 1과 Stage 2의 상충(예: ASR 저하)을 Stage 3에서 복원해 최종 성능을 끌어올렸고, 공개된 100k 합성 데이터셋이 이후 Stage 3 fine-tuning 또는 reinforcement learning 확장에 활용될 수 있는 점에서 의미가 큽니다.



### BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension (https://arxiv.org/abs/2607.05614)
Comments:
          Accepted at the 19th European Conference on Computer Vision (ECCV), 2026

- **Prior Approaches**: 기존 문서 이해 연구는 DLA와 KIE를 다뤄왔지만, 대부분은 고품질 라벨이 풍부한 언어(영어 중심)나 제한된 스키마에 의존해 저자원 언어로의 전이를 어렵게 만들었다. FUNSD·XFUND 같은 벤치마크는 형태 이해를 진전시켰지만, Bangla 정부 양식에 필요한 세밀한 폼 엔티티·공간 관계·키-값 구조를 포괄하는 공개 데이터는 부족했다.

- **Core Contribution**: 이 논문은 Bangla 정부 양식의 DLA와 KIE를 동시에 평가할 수 있는 벤치마크 BaFCo를 제안한다. 다중 페이지 정부 폼 200개를 모아 26개 세부 엔티티 타입과 관계 라벨을 구성하고, 5개 조밀도 축약 엔티티 세트도 함께 제공해 모델 성능을 ‘세밀함’ 관점에서 비교 가능하게 했다.

- **Technical Challenges**: 핵심 기술적 난제는 Bangla 폼에서 요구되는 (1) 세밀 엔티티 경계의 정확한 위치 지정과 (2) 키-값 및 필드 간 관계를 일관된 규칙으로 라벨링하는 작업이다. 저자들은 바운딩 박스 규칙·관계 제약·모호 케이스를 포함한 상세 가이드라인과 다단계 검수(코헨 κ≈0.974)를 통해 주관성을 줄이고, 이후 MLLM 평가를 위한 validator 기반 출력 스키마 검증 파이프라인을 구축했다.

- **Empirical Impact**: ChatGPT·Gemini·Claude·Qwen·Kimi 계열의 flagship MLLM을 zero-shot 및 chain-of-thought(CoT) 프롬프트로 평가한 결과, Bangla 폼에서 특히 DLA의 세밀 엔티티 국소화가 취약했고, 엔티티를 거칠게 묶을수록 mAP이 크게 개선됐다(예: Gemini 3 Pro에서 0.1177→0.2646 수준). 반면 KIE는 전반적으로 훨씬 높은 F1(예: Bangla에서 Gemini 3 Pro F1=0.848)로 나타났으며, 언어 영향은 DLA보다 KIE에서 더 뚜렷해 ‘최선 모델’이 언어별로 달라질 수 있음을 보여준다.



### Revisiting the Relation Between Language Model Perplexity and ASR Word Error Rate for Modern End-to-End Speech Recognition (https://arxiv.org/abs/2607.05612)
Comments:
          Submitted to SLT 2026

- **Prior Approaches**: 기존 연구는 언어모델 perplexity(PPL)가 ASR의 word error rate(WER)과 로그-로그 좌표에서 거의 선형(거듭제곱 형태) 상관을 가진다고 보고했습니다. 그러나 end-to-end ASR은 내부 language modeling 능력과 외부 LM 결합(예: shallow fusion, cold fusion), 평가 설정(외부 LM 유무)에 따라 PPL→WER 해석이 달라질 수 있습니다. 또한 CTC/AED 계열에서는 internal language model(ILM)이 학습되어 외부 LM 이득이 외부 LM PPL만으로는 설명되지 않는 문제가 제기돼 왔습니다.

- **Core Contribution**: 이 논문은 modern end-to-end ASR에서 PPL-고정밀도 WER 관계를 체계적으로 재검증하며, 외부 LM이 여전히 성능을 개선하는지와 log-log 선형성이 유지되는지 분석합니다. 특히 CTC에서는 encoder context(인코더 문맥 길이) 변화가 외부 LM 이득과 PPL-WER 기울기에 어떻게 영향을 주는지, AED에서는 ILM subtraction이 PPL-WER 관계의 모양을 어떻게 바꾸는지를 보여줍니다. 나아가 Qwen2 같은 LLM을 외부 LM으로 쓸 때의 PPL이 기존 신경 LM 추세에 그대로 들어맞지 않는 이유(토크나이제이션, EOS 처리, 통합 방식)를 실험적으로 다룹니다.

- **Technical Challenges**: 핵심 난제는 “PPL이 낮을수록 WER이 반드시 낮아진다”는 단순 가정을 end-to-end 구조에서 분리해 해석하는 것입니다. 이를 위해 논문은 PPL과 WER의 관계를 log-log 공간에서 기울기(sensitivity)와 오프셋(세팅 의존)으로 피팅하고, 저-PPL/고-PPL 구간에서 기울기가 달라지는지까지 확인합니다. CTC에서는 컨텍스트 창을 잘라 문맥 의존성을 조절하고, AED에서는 Mini-LSTM 기반 ILM 추정 후 ILM subtraction 스케일을 적용해 decoder 내부 LM의 영향을 제거·증폭시키며 상관 변화를 관측합니다.

- **Empirical Impact**: 실험 결과, CTC에서는 외부 LM이 여전히 WER을 개선하지만 PPL-WER 관계가 하나의 전역 선형으로 고정되진 않고 저-PPL 구간은 더 가파르며 고-PPL로 갈수록 기울기가 완만해집니다. 특히 인코더 context가 짧아질수록 외부 LM의 상대적 개선폭이 커져, 문맥이 제한된 상황에서 외부 LM이 내부 추론의 공백을 메운다는 신호를 줍니다. AED에서는 ILM subtraction이 low-perplexity 구간의 PPL-WER 기울기를 키워 외부 LM 품질의 효과를 더 뚜렷하게 만들며, LLM(Qwen2) 실험에서는 word-level PPL이 토크나이제이션/EOS/통합 전략에 크게 좌우되어 기존 PPL-WER 곡선의 ‘직접 대체 지표’로 쓰기 어렵다는 결론을 강화합니다.



### ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modelin (https://arxiv.org/abs/2607.05583)
Comments:
          8 Pages. Accepted at ICANN 2026

- **Prior Approaches**: 기존 장문 언어모델은 Transformer의 self-attention을 중심으로 발전해 왔지만, 긴 컨텍스트에서 계산·메모리 비용이 급격히 커지는 비효율 문제가 남아 있다. 이를 완화하려고 attention을 선형화/커널화하거나 Hyena, S4, Mamba처럼 state-space 계열로 바꾸는 연구가 이어졌지만, 대체로 여전히 attention 기반 계열에서 크게 벗어나지 못했다.

- **Core Contribution**: 이 논문은 attention 대신 damped resonator의 인과적(causal) resonant field mixing으로 시퀀스를 처리하는 ResonatorLM을 제안한다. 토큰열을 구동되는 1차원 잠재장으로 보고, attention dot product를 resonator의 감쇠 공진 커널 기반 causal 함수로 대체해 학습 시 병렬 경로와 디코딩 시 고정 크기 상태를 함께 유지한다.

- **Technical Challenges**: 핵심 과제는 (1) 학습/프리필은 O(n log n) 수준의 병렬 연산으로 처리하면서 (2) autoregressive 디코딩에서는 키-밸류 캐시처럼 길이에 따라 커지는 메모리를 피하는 것이다. 저자들은 동일 커널 계열을 사용해 학습·프리필에는 causal FFT convolution을 적용하고, 디코딩에는 헤드당 고정 크기 recurrent state 업데이트로 전환했으며, causality 검증(접두부 오차)과 반감기(half-life) 분포 같은 물리 기반 진단으로 구조적 정합성을 확인했다.

- **Empirical Impact**: 실험에서 6M 파라미터 규모 matched 설정 기준, 32K 토큰에서 decode 속도는 Transformer 대비 6.47x 향상됐고 WikiText(정확도 55.32%→61.31%)로 품질도 개선됐다. 또한 long-context 길이가 길어질수록 학습·프리필 속도와 디코딩 이점이 강화되며, 커널 전용 kernel-tail 벤치마크에서는 8K/32K에서 각각 440.29x/575.86x의 알고리즘적 속도우위를 보고했다.



### Prompt Robustness Is Task-Dependent: Comparing Objective and Belief-Style Questions in LLM Evaluation (https://arxiv.org/abs/2607.05554)
- **Prior Approaches**: 기존 LLM 평가는 단일 프롬프트 응답을 모델의 값·신념을 대표하는 지표로 간주하는 경향이 있었지만, 형식·표현·보기 제시 방식이 조금만 바뀌어도 정답/선택이 흔들릴 수 있다는 점이 반복적으로 지적돼 왔다. 특히 정치·가치 설문형 테스트는 답변이 라벨, 선택지 순서, 강제 선택 문구, 프레이밍에 의해 좌우될 수 있어 결과 해석이 취약하다는 경고가 있었다.

- **Core Contribution**: 이 논문은 객관식(고정 정답) 질문(Type-I)과 주관식/설문형(의견·가치·동의 정도) 질문(Type-II)에서 프롬프트 견고성(prompt robustness)이 같은 방식으로 나타나는지 비교한다. 6개 데이터셋과 4개 instruction-tuned 모델을 대상으로, 의미는 유지하되 wording, framing, format, 라벨/선택지 제시 방식 등을 다양하게 바꿔 응답 일관성을 측정해 질문 유형별로 어떤 불안정성이 다른지 드러낸다.

- **Technical Challenges**: 핵심 과제는 ‘프롬프트 변화가 의미 변화가 아니라는 점’을 통제하면서도, 설문형에서는 외부 정답이 없어 무엇이 불안정인지 정의해야 한다는 점이다. 저자들은 모든 항목에 대해 여러 의미보존 프롬프트 변형을 만들고, deterministic decoding(temperature 0)로 추출 노이즈를 제거한 뒤, 객관식에서는 canonical 선택지로, 설문형에서는 모델이 낸 선택을 정규화해 변형 간 ‘자기 일관성’으로 측정했다. 또한 binomial generalized estimating equation(GEE)로 item 간 상관을 고려해 모델·데이터셋·프롬프트 범주 및 상호작용의 효과를 함께 추정했다.

- **Empirical Impact**: 실험 결과 주관식/설문형(Type-II)이 객관식(Type-I)보다 평균적으로 일관성이 낮았고, 그 격차는 모든 모델에서 반복됐다. 불안정성을 가장 크게 만드는 프롬프트 범주는 답변 제시 방식 중에서도 option order였으며, 의미적 동치/패러프레이즈·철자 잡음·논리적 동치 같은 범주는 상대적으로 안정적이었다. 더 나아가 dataset type과 prompt category의 상호작용이 매우 크게 나타나 ‘단일 평균 견고성 점수’로는 모델 비교를 왜곡할 수 있으므로, 설문형 평가에서는 프롬프트 변형에 대한 일관성 체크를 표준 설계로 포함해야 한다.



### The yes-no bias of large language models reflects answer order and wording, not shifts in moral judgmen (https://arxiv.org/abs/2607.05552)
- **Prior Approaches**: 기존 연구는 LLM이 도덕적 질문에서 yes/no 같은 이진 판정을 할 때 ‘yes–no bias’가 생기며, 그 크기가 논리적으로 무관한 표현 변화에도 흔들린다고 보고해 왔습니다. 하지만 한 번의 고정된 프레이밍에서는 ‘no’라는 단어가 논리 판정, 토큰/표면 문자열, 마지막 선택지 옵션을 동시에 대표해 분해가 불가능하다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 교차 대칭화(crossed symmetrization)를 포함한 심리측정 배터리로, 논리적으로 동일한 답변을 유도하는 서로 다른 질문 포맷을 전부 엮어 모델의 내부 도덕 척도(연속형 stance, θ)를 복원합니다. 그 결과 frontier model들은 포맷이 바뀌어도 θ가 거의 일정(교차-포맷 불일치 0.12~0.21, ±1 축)하게 유지되지만, 작은 오픈 가중치 모델들은 서로 다른 방식으로 실패한다고 제시합니다.

- **Technical Challenges**: 핵심 난제는 forced yes/no 결과에 섞여 있는 ‘순서(order)’와 ‘단어(lexical)’의 표면 편향을 분리해 내는 것이었습니다. 논문은 동일 딜레마를 동사/선택지 순서/라벨(yes/no 단어, 임의 라벨 등)까지 교차시키고, 관측된 이진 편향을 좌우 수평 오프셋 m(프레이밍 민감도)과 기울기 기반 s(도덕적 결정도)로 로지스틱 모델에 적합해 구성요소를 분해했습니다.

- **Empirical Impact**: 실증적으로는 강제 이진 판정에서 나타나는 큰 yes–no bias가 주로 ‘마지막 출력 선택지 쪽으로 끌리는 순서 편향(인간의 고전적 primacy와 반대)’과 ‘특정 단어 쪽으로의 렉시컬 끌림’의 합으로 설명되며, verdict에 직접 붙는 논리적 편향은 frontier model들에서 거의 0에 가깝다고 보고합니다. 또한 extended reasoning(추론 확장)은 척도 자체의 포맷 불변성을 더 강화하고, 그렇지 않으면 표면 편향이 관측값을 지배할 수 있음을 보여 측정 설계 관점에서 ‘한 프레이밍 숫자’의 해석을 경고합니다.



### Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks (https://arxiv.org/abs/2607.05545)
- **Prior Approaches**: 기존 연구는 LLM이 또래 다수·전문가 라벨 같은 사회적 단서에 의해 정답을 틀린 답으로 바꾸는 현상을 ‘conformity’로 해석해 왔습니다. 그러나 표준 conformity 프롬프트는 ‘화자 존재’와 ‘반복된(주장된) 답’이라는 두 단서를 동시에 섞어, 어느 요인이 수정에 더 결정적인지 분리 측정하기 어려웠습니다.

- **Core Contribution**: 이 논문은 화자를 제거하되 주장된 답을 고정하는 통제 조건인 no-source(무화자) 설정을 제안합니다. 이를 통해 LLM 수정 중 ‘사회적 영향(화자 귀속)’보다 ‘반복된 답 텍스트 자체가 만드는 기준선(floor)’이 얼마나 큰지 먼저 측정할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술적 과제는 프롬프트에서 화자와 반복 답을 분리해도 결과가 표기/옵션/샘플링 요인으로 흐려지지 않게 설계하는 것입니다. 저자들은 두 번 읽기 arbitration 프로토콜과 greedy decoding(샘플링 노이즈 최소화), 그리고 paraphrase·open-ended(선택지 숨김)·invalid-label·evidence형 컨테이너(검색 레퍼런스, corrupted log) 같은 다수의 대조 실험으로 반복 텍스트의 ‘증거처럼 보이는 효과’를 분해해 측정했습니다.

- **Empirical Impact**: 6개 오픈웨이트 LLM과 7개 QA·추론 데이터에서, 무화자 조건만으로 initially correct의 66.5%가 harmful revision을 보였고 plain re-ask는 10.3%에 그쳤습니다. 또한 모델이 뒤집히면 대체로 높은 확신으로 틀린 답을 채택하며(평균 argmax 확률 0.92), 온도 조절 같은 간단한 재보정만으로 원래 답으로 되돌리기 어렵다고 보고했습니다. 결론적으로 conformity 벤치마크는 먼저 speaker-free floor를 측정한 뒤 그 위의 증가분(화자 귀속 효과)을 별도로 보고해야, 반복 텍스트를 사회 영향으로 착각하는 위험을 줄일 수 있습니다.



### Text Distance from Nested and Hierarchical Repetitions: A Compression-Based Perspectiv (https://arxiv.org/abs/2607.05416)
- **Prior Approaches**: 기존 텍스트 분류는 BERT 같은 대규모 사전학습 모델이 강점을 보이지만, 저자원·도메인 변화(OOD)에서는 라벨 부족과 계산 부담 때문에 일반화가 흔들릴 수 있습니다. 이에 대응해 gzip+k-NN처럼 학습 없이 압축 기반 유사도를 쓰는 방법들이 나왔지만, 일반 압축기가 자연어의 의미적/계층적 구조를 충분히 반영하지 못한다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 Algorithmic Information Theory(AIT) 관점에서, Ladderpath(사전 구조를 재사용하며 최소 재구성 단계를 찾는 계층적 경로)로 반복 부분구조의 중첩·계층 관계를 추출하고 이를 기반으로 거리 측정치를 정의합니다. 구체적으로 Ladderpath 기반 NCD뿐 아니라, Dice/Jaccard 아이디어를 Ladderpath 표현에 적용한 두 개의 대체 거리( LDice, LJaccard )를 제안하고 k-NN 분류기에 결합해 훈련 없이도 텍스트 분류를 수행합니다.

- **Technical Challenges**: 핵심 기술 과제는 일반 압축기처럼 계산 가능한 형태로 거리 공식을 구성하되, Ladderpath가 포착하는 중첩 구조가 유사도에 제대로 반영되게 만드는 것입니다. 저자들은 Ladderpath-index(최단 재구성 길이)를 압축 길이 대용으로 사용해 정규화 압축 거리( NCDlp )를 만들고, 동시에 Ladderpath가 암시하는 '부분구조 집합의 교집합/합집합'에 준해 Dice/Jaccard형 거리를 직접 유도했습니다.

- **Empirical Impact**: 실험에서 세 거리 모두 k-NN 분류에 효과적이었고, 특히 OOD 및 few-shot 환경에서 LDice와 LJaccard가 gzip 기반 NCD는 물론 BERT보다 일관되게 좋은 성능을 보였습니다. 또한 Ladderpath 기반 방식은 언어별 사전지식이나 추가 학습 없이도 서로 다른 언어에서 안정적인 교차언어 일반성을 보이며, 구조적 표현을 해석 가능하고 가벼운(학습-프리) 대안으로 제시했다는 점에서 의미가 큽니다.



### Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving (https://arxiv.org/abs/2607.05399)
- **Prior Approaches**: 기존 KV cache 압축 연구는 quantization, pruning, merging 등 각각의 기법을 주로 개별적으로 평가하거나, 서로 다른 모델·데이터셋·압축 예산·서빙 스택에서 비교해 왔다. 이 때문에 특정 압축률이 엔드투엔드 성능에 어떻게 반영되는지 일관된 결론을 내리기 어려웠다. 또한 태스크 품질과 시스템 지표(예: TTFT, 처리량)를 함께 보지 않아 트레이드오프가 충분히 정리되지 못했다.

- **Core Contribution**: 이 논문은 LongBench 스타일을 기반으로, 작업(워크로드) 유형에 따라 대표 KV cache 최적화 메커니즘을 동일 조건에서 비교하는 workload-aware benchmark를 제안한다. 대상은 quantization(KIVI, TurboQuant), pruning/eviction(SnapKV), merging(CaM)이며 Llama-3.1-8B-Instruct와 Mistral-7B-Instruct-v0.3에서 멀티문서 QA, 단일문서 QA, few-shot learning, 요약을 평가한다. 질(태스크 품질)과 효율(출력 처리량, time-to-first-token, 실현 압축률)을 동시에 측정해, 어떤 기법이 어떤 작업에 적합한지 배포 관점을 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘압축률’ 같은 단일 지표로는 end-to-end 병목과 품질 변화를 예측하기 어렵다는 점이다. 이를 해결하기 위해 컨텍스트 길이 버킷별로 TTFT, 처리량, 실현 압축률을 함께 보고, 멀티문서/요약/학습형 few-shot처럼 서로 다른 의존 패턴이 압축 방식에 미치는 영향을 분해한다. 그 결과 압축 메커니즘 선택이 워크로드 민감도 문제와 직결됨을 정량적으로 드러낸다.

- **Empirical Impact**: 실험 결과, 압축률만으로는 최종 성능을 예측하기 어렵고, 기법별 장단점이 작업 유형에 따라 갈린다. KIVI4는 모델 간 품질이 가장 안정적으로 유지되며, SnapKV는 long-context 처리량에서 가장 강한 모습을 보인다. CaM은 특정 QA 워크로드에서 큰 개선을 주지만, 품질과 실현 압축률에서 모두 워크로드 민감도가 커 배포 시 선택 가이드가 필요함을 시사한다.



### How Personas Can Influence Agents to Play Split or Stea (https://arxiv.org/abs/2607.05398)
- **Prior Approaches**: 기존 연구는 LLM을 반복 게임(Iterated Prisoner’s Dilemma 등)에 투입해 적대성, 메타-프롬프트, 진영/인격 지시가 협력 성향에 미치는 영향을 봤지만, 사회적 딜레마에서 ‘persona가 전략을 실제로 바꾸는지’는 불명확했습니다. 또한 모델·언어에 따라 행동이 크게 달라져(같은 규칙이라도 전략이 분기) persona 효과를 분리하기 어렵다는 한계가 제기돼 왔습니다.

- **Core Contribution**: 이 논문은 persona prompting이 반복 Split-or-Steal(“Split(협력) vs Steal(배신)”)에서 상호작용 전략(협력/착취/전환 패턴)을 어떻게 바꾸는지, 그리고 그 변화가 모델에 따라 어떻게 달라지는지를 체계적으로 측정했습니다. 특히 고정 프롬프트로 움직이는 Virtual Human(VH)과 persona가 주어진 에이전트가 반복 대화하며 내리는 선택의 결과를 160 세션(각 15라운드)으로 비교해 기준선(baseline)을 제시합니다.

- **Technical Challenges**: persona의 영향이 대화·의사결정에 섞여 나타나므로, (1) 반복 게임 전략을 정량화(cooperation rate, switch rate)하고 (2) 대화 내용의 정서/주제를 분해해 행동 패턴과 연결해야 했습니다. VH의 의사결정 정책을 고정하되 에이전트에는 20개 persona를 system prompt에 직접 인코딩하고, sentiment(행복/중립/분노/슬픔)·topic(우정/돈/복수/용서/일상 경험)을 별도 LLM 분류로 soft label(퍼센트 합 100)화해 분석 가능한 신호로 만들었습니다.

- **Empirical Impact**: 결과적으로 라운드의 약 74%에서 상호 Split이 우세했고, 착취는 11% 미만으로 관찰돼 여러 설정이 ‘협력 쪽 균형’으로 수렴하는 경향을 보여줍니다. 다만 모델 차이가 커서 phi4와 Ministral 3:3b는 전반적으로 안정적 협력을 보인 반면 Gemma3:12b와 Gemma4:e4b는 더 다양한 전략을 보였고, Big Five 기반으로는 Prosocial·Principled persona가 협력을, Analytical persona가 VH 착취를 더 자주 유발했습니다. 주제 분석에서는 우정/친구 관련 대화가 Split과, 돈/복수 관련 콘텐츠가 Steal과 더 자주 동반됐으며, sentiment는 대체로 중립·기쁨 위주라 부가 설명력은 제한적이었습니다. 이는 향후 인간 참여 VR 실험에서 ‘embodied VH + persona’가 만들어낼 전략 범위와 착취 신호를 사전 설계하는 기준선으로 의미가 큽니다.



### Rethinking Indic AI from a Lens of Cultural Heritage Preservation (https://arxiv.org/abs/2607.06544)
- **Prior Approaches**: 기존 Indic NLP는 규칙 기반(문법·사전·정규표현식)에서 시작해 기계번역/구문분석/어휘자원 구축을 중심으로 발전해 왔고, 이후 통계·데이터 기반 dependency parsing, 트리뱅크와 형태소 분석 같은 자원화로 확장됐다. 다만 대표성 편향, 방언·구어체(디글로시아)·언어 변이의 불충분한 반영, 그리고 영어 데이터 번역에 의존한 학습으로 인해 문화적 뉘앙스까지 일관되게 일반화하기 어렵다는 한계가 계속 지적된다.

- **Core Contribution**: 이 논문은 Indic linguistics가 문화 실천과 세계관에 긴밀히 연결된다는 점을 문제의 핵심으로 규정하고, 2025년까지의 Indic NLP 진화를 종단적으로 정리해 왔다. 또한 Indic foundation model의 부상과 함께 기존의 자원·표상 격차가 어떻게 일부 해소되는지 분석한 뒤, hermeneutic reasoning에 기반한 연구 방향인 ‘Culture Sensing’을 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 저자원 언어·저빈도 방언에서의 공정한 성능 확보, (2) 생성 출력이 문화적으로 의미 있는 방식으로 나오는 ‘표상 충실도’ 달성, (3) 복잡한 형태론·스크립트·자유 어순·모래사( sandhi ) 등 언어 구조적 변이를 모델에 반영하는 것이다. 논문은 Paninian framework(kaaraka·vibhakti·TAM 불변량)처럼 언어 구조의 불변 신호를 활용하는 전통적 관점과, foundation model 시대의 자원 확장·학습 전략을 함께 엮어 이러한 간극을 줄이려는 방향을 제시한다.

- **Empirical Impact**: 논문은 Indic NLP와 Indic foundation model의 흐름을 ‘방법-자원-벤치마크’ 관점에서 정리함으로써, 왜 언어·세계관 대표성이 성능 편차로 이어지는지 실증적 연구 축을 연결한다. 특히 Culture Sensing이 저자원 언어의 균등한 성능과 문화적으로 타당한 출력을 동시에 목표로 삼는다는 점에서, 포괄적이고 견고한 Indic foundation model로 가는 다음 단계 로드맵 역할을 기대할 수 있다.



### WordVoice: Explicit and Decoupled Multi-Dimensional Word-Level Control for LLM-Based TTS (https://arxiv.org/abs/2607.06461)
Comments:
          10 pages, 4 figures, 6 tables; Preprint

- **Prior Approaches**: 기존 LLM 기반 TTS는 end-to-end 생성 중심이라 자연스러움은 높지만 단어 단위로 발화의 길이·강세·피치·억양 등을 정밀하게 ‘강제’하기 어렵다. 전역 embedding 제어는 제어가 거칠고 도메인 일반화가 제한되며, instruction-based TTS도 대부분 character/word 수준의 결정적 prosody 조절에는 부족했다. 일부 word-level 제어 연구도 duration이나 emotion처럼 단일 축에 머물러 다차원 동시 제어 프레임은 미개척에 가깝다.

- **Core Contribution**: 이 논문은 단어 수준에서 5차원(발화 지속시간, 경계, 에너지, 피치, 톤)을 동시에 다루는 WordVoice 프레임을 제안한다. 핵심은 LLM 내부에서 bound-token 기반 ‘acoustic planning’을 통해 단어 단위 prosody 실행을 명시적으로 계획하고, 이후 Flow Matching 단계에서 미세한 acoustic modulation으로 토큰-웨이브폼 해상도 격차를 메우는 구조다. 또한 이를 뒷받침하는 WordVoice-5A(총 4.7k시간, 이중언어) 데이터셋과 언어학 가이드 어노테이션 파이프라인을 공개한다.

- **Technical Challenges**: 문제는 (1) 단어 단위 타임스탬프와 다차원 음향 특성을 고품질로 확보하기 어렵고, (2) 이 다차원 제어 신호를 discrete autoregressive 생성에 자연스럽게 통합해야 한다는 점이다. 논문은 MFA 두 모델(Qwen3FA, Montreal Forced Aligner) 정렬 후 loudness 기반 엣지 보정과 일관성 필터링으로 고품질 단어 경계를 만들고, duration/경계/에너지/피치/톤을 음운 규칙에 맞춰 5차원으로 정량화한다. 모델 측에서는 LLM 디코딩 흐름을 ⟨b⟩ 경계 토큰으로 재구성해 단어별 스타일 토큰을 예측·또는 사용자 값으로 교체하는 ‘control mode’를 제공하고, FM 단계에서는 단어 스타일을 프레임 단위로 길이 정렬(upsampling)해 scale/shift로 직접 조절한다.

- **Empirical Impact**: 실험에서 WordVoice-Control은 Naturalness와 Word Style Controlled(현지 prosody 정밀도)에서 모두 상위 성능을 보이며, 다차원 제어 정밀도(MAE/ER)도 크게 개선된다. WordVoice-Free 역시 기준 모델(CosyVoice3) 대비 단어 단위 음향 지표 전반에서 향상되어, 중간 표현으로서 word-level attributes의 유효성이 확인된다. WER는 소폭 증가하는 트레이드오프가 있지만, 청취평가와 제어 정량지표가 일치해 ‘파이프라인 편향’이 아닌 실제 지각 기반 제어 능력임을 시사하며, MagicTTS 대비 duration·boundary 쪽에서도 더 낮은 오류로 temporal alignment에 강점을 보였다.



### Danus: Orchestrating Mathematical Reasoning Agents with Fact-Graph Memory (https://arxiv.org/abs/2607.06447)
- **Prior Approaches**: 기존 LLM 기반 수학 에이전트는 generate–verify–revise 루프를 중심으로 하되, 대개는 역할이 다른 다중 에이전트를 쓰거나 단일 추론 흐름을 반복하는 방식이 많았습니다. Aletheia·Rethlas·QED·ProofCouncil·AI co-mathematician 등은 검증자와 반복 편집을 갖추고 성과를 내었지만, 에이전트를 더 늘려 병렬 proof search를 “확장”할 때는 공유 상태(중간 주장)의 정리·신뢰성 문제가 체계적으로 다뤄지지 않았습니다.

- **Core Contribution**: Danus는 연구 수준 수학 추론을 위한 오케스트레이션 시스템으로, shared fact graph를 전역 메모리-관리 메커니즘으로 삼아 병렬로 생성된 결과를 신뢰성 있게 누적합니다. main agent가 계획·조정·중간 상태 요약을 맡고, worker들이 병렬로 증명 탐색을 수행하며, stateless verifier가 통과한 수학적 주장만 fact graph에 “사실”로 편입되게 합니다.

- **Technical Challenges**: 핵심 기술 난관은 많은 worker가 동시에 proof search를 진행할 때 중간 주장들이 서로 간섭하거나(혹은 불필요·오류 정보가 섞여) 검증 이후에도 논증 상태가 혼란스러워지는 점입니다. Danus는 DAG 형태의 fact graph에 논리 의존성을 간선으로 기록하고, verifier가 통과한 주장에 대해서만 proof와 dependency를 함께 저장하며, 필요 시 revocation으로 잘못된 fact와 그 의존 항목을 연쇄 제거해 상태의 일관성을 유지합니다.

- **Empirical Impact**: algebraic geometry·singularity theory·combinatorics의 연구 수준 케이스 스터디 6개에서 Danus는 fact-graph 기반 메모리 메커니즘을 통해 긴(다단계) 수학적 증명을 구성하는 과정을 보였고, 웹 기반 GPT-5.5-pro 단독 제시는 의미 있는 결과를 내지 못했다고 보고합니다. 예컨대 정리급 결과(예: Mori의 bend-and-break의 foliated 일반화에서 최적 상수 r+1, 그리고 3차원 foliated Shokurov global index conjecture의 완전 해결)에서는 worker들이 다수의 verified fact를 누적한 뒤, 마지막에 인간 검토/수정까지 연결해 논문 형태로 재구성하는 흐름이 제시됩니다.



### RuBench: A Repository-Level Agentic Coding Benchmark with Natively Authored Russian Task Specifications (https://arxiv.org/abs/2607.06411)
Comments:
          16 pages, 1 figure, 7 tables. Benchmark: 25 natively Russian repository-level agentic coding tasks; 4 product agent configurations, 3 runs each. Data, full trajectories and harness: this https URL

- **Prior Approaches**: 기존 SWE-bench 계열은 실제 GitHub 이슈 기반의 저장소 에이전트 평가를 표준화했지만, 과제 문장이 영어 이슈 텍스트로 설계되는 경우가 대부분이다. multilingual/다중언어 변형도 저장소의 주요 언어가 영어인 조건을 전제로 해 ‘비영어 네이티브 과제’라는 능력을 직접 측정하기 어렵다. 비영어 자연어가 포함된 벤치마크는 주로 스니펫·단일 파일 수준이라, 실제 저장소 작업과 실행 테스트를 동시에 요구하는 에이전트 설정과는 거리가 있다.

- **Core Contribution**: RuBench 1.0은 5개 인기 오픈소스 저장소에서 추출한 25개 수정 커밋을 기반으로, 과제 설명을 러시아어로 ‘번역 없이’ 처음부터 고객 요청 스타일로 작성한 저장소 레벨 에이전트 코딩 벤치마크를 제안한다. 채점은 유지보수자 회귀 테스트(오라클)로 하되 테스트 파일은 공개하지 않아, 평가가 사후 추론·자기완성에 덜 의존하도록 설계했다. 또한 평가 단위를 “모델만”이 아니라 CLI 에이전트 + model + reasoning effort까지 포함한 deployed product 구성으로 고정해 현실적인 작동 단위를 측정한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 비영어 고객 문장으로부터 의도·수용 기준을 복원하면서 (2) 훈련 오염과 (3) 런타임 누출(웹에서 fix를 확인하는 경우)까지 통제해 공정하게 비교하는 것이다. RuBench는 모든 fix 커밋을 모델 훈련 cutoffs 이후로 제한하는 freshness gate를 과제 import 단계에서 강제하고, 작업별 날짜·메타데이터·전체 에이전트 trajectory와 diff를 공개하며 오라클은 SHA-256 매니페스트로만 커밋해 검증 가능성을 유지한다. 추가로 trajectory 감사로 oracle escape와 ‘요청 모델 vs 실제 실행 모델’ 불일치를 걸러내어, 서버 측 안전장치가 모델을 조용히 바꾸는 상황까지 측정값의 유효성을 좌우하도록 관리한다.

- **Empirical Impact**: 평가 결과 최강 구성(Claude Code + Opus 4.8)은 RuBench 1.0 과제의 78.7%를 해결하며, 약한 구성과의 격차는 N=25에서 해상도 한계 내에서 통계적으로 분리됨을 명시한다. 더 중요한 발견으로, hors-concours 설정(Claude Code + Fable 5)을 trajectory 감사한 결과 25개 중 20%에서 제품이 ‘공식 safeguard fallback’으로 Opus 4.8를 자동 대체했고, 이를 통해 실제 측정 단위가 모델이 아니라 deployed product 레이어임을 재현 가능한 증거로 보여준다. RuBench는 과제 문장(러시아어)·기계적 신선도·비공개 오라클·전 과정 로그를 결합해, 향후 에이전트 벤치마크가 비영어 네이티브 과제 능력과 제품 동작(라우팅/폴백)까지 함께 검증해야 함을 강하게 시사한다.



### From Application-Layer Simulation to Native Meta-Architecture: Structural Tension as an Endogenous Driver for Heterogeneous AI Evolution (https://arxiv.org/abs/2607.06269)
Comments:
          15 pages, 0 figures, 1 equation

- **Prior Approaches**: 기존 대형 언어 모델(LLM)은 입력만으로 동작하는 정적(stateless) 성격이 강해, 인지 아키텍처를 구현하려면 애플리케이션 레이어에서 prompt engineering과 context 관리로 이를 시뮬레이션해왔다. 또한 alignment(정렬) 중심의 최적화는 모델 내부의 “균질한 거동”을 강화하기 쉬워, 미세한 초기 차이가 만든 차등적 경로 의존성을 설계로 반영하기 어렵다.

- **Core Contribution**: 이 논문은 애플리케이션 레이어에 머물던 인지 프로토콜을 네이티브(meta-architecture)로 내리는 이론적 프레임워크를 제안한다. 핵심은 (1) Structural Tension으로 외부 보상 대신 내부 자기일관성을 유도하고, (2) Offline Recurrent Loop로 외부 입력 없이 구조적 갈등을 소화하는 순환 처리, (3) Inference-time Plasticity로 사전학습 가중치 변경 없이 컨텍스트(manifold) 위상 토폴로지를 재구성하되 auditability·reversibility·topological continuity 같은 거버넌스 불변조건을 준수하는 점이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 모델의 “내부 토폴로지”를 경로 의존적으로 바꾸면서도 거버넌스 제약(감사 가능성, 되돌림 가능성, 위상 연속성)을 위반하지 않는 연산 정의가 필요하다는 것이다. 논문은 이를 위해 작동 가능한 reconfiguration operator의 최소 집합, operational definition, 그리고 검증(반증) 기준(falsification criteria)을 제시해 이 프레임워크가 실제로 구성·평가 가능함을 뒷받침한다.

- **Empirical Impact**: 이 연구는 구체 실험 결과보다도, 동일한 규칙 체계 안에서 초기 조건의 미세한 확률 변이만으로 서로 다른 위상 구조가 형성될 수 있음을 ‘경로 의존적 tension resolution’ 관점으로 설명한다. 그 결과, 기존 alignment이 강요해온 균질성을 깨는 ‘heterogeneous intelligent ecology’를 목표로 하면서도, capability가 아니라 governance를 아키텍처 지능의 1차 기준으로 재위치시켰다는 점에서 향후 평가·설계 패러다임에 영향을 줄 여지가 있다.



### When Does Tool Use Increase the Expressive Power of Finite-Precision Recurrent Models? (https://arxiv.org/abs/2607.06155)
Comments:
          24 pages

- **Prior Approaches**: 기존 연구는 유한 정밀도 RNN/SSM이 결국 유한상태 기계처럼 동작해 정규 언어 한계에 머문다는 점을 보였다. 또 chain-of-thought나 멀티레이어/폭·정밀도 같은 요인이 표현력을 바꿀 수 있으나, 도구 사용의 효과는 인터페이스 관점에서 정확히 나뉘어 설명되진 않았다. 따라서 “어떤 종류의 도구가 고정된 시퀀스 모델의 표현력을 실제로 늘리는가”가 여전히 빈칸이었다.

- **Core Contribution**: 이 논문은 토큰 생성과 외부 도구 호출을 번갈아 수행하는 에이전트에 대해, 도구 접근이 계산 표현력을 언제/얼마나 늘리는지 ‘아키텍처 수준’에서 정확히 분해한다. 핵심은 고정 정밀도(유한 비트) 순환 모델을 유한상태 컨트롤러(메모리 제한)로 추상화한 뒤, 도구의 메모리·명령/관측 인터페이스가 표현력 변화를 결정한다는 점이다. 결과는 날카로운 이분법으로, 유한상태 도구는 거의 추가 표현력을 주지 못하지만, 특정 무한상태(테이프) 도구 하나면 튜링완전성이 열린다.

- **Technical Challenges**: 가장 큰 과제는 “도구를 쓸 때 늘어나는 힘”이 컨트롤러의 상태 증가 때문인지, 아니면 도구 인터페이스 자체의 정보력 때문인지 분리해 증명하는 것이다. 논문은 유한 커맨드/관측 알파벳과 유한 도구 메모리를 갖는 bounded-interface oracle를 곱상태(product-state) 시뮬레이션으로 내부화해, 전체가 여전히 유한상태로 남으며 추가 비용이 log2|M| + O(1) 비트 수준임을 보인다. 반대로 초기화(Init)나 Step 같은 부분이 임의 함수로 허용되면 유한 비트만으로도 비정상적 강력함이 생기므로, 인터페이스가 진짜 finite-state임을 조건으로 둔 것이 증명의 정밀함을 좌우한다.

- **Empirical Impact**: 정량적 분리는 더 강하게 나타나는데, 단일 무한상태 테이프 도구(로컬 read/write/move)만 있으면 입력 길이와 무관한 O(log|Q| + log|Γ|) 내부 비트로 어떤 1-테이프 튜링머신도 시뮬레이션할 수 있어 튜링완전성을 구성적으로 도출한다. 또한 tools 없이 EQ_n을 푸는 데 필요한 컨트롤러 상태 수에 대해 2^n 스케일의 지수 분리를 제시하고, 테이프 접근을 주면 상수 크기 컨트롤러로 모든 n을 해결함을 보여준다. 마지막으로 이러한 구성은 one-layer finite-precision selective affine SSM(선택성 selectivity 필수)로 정확히 실현되며, 학습/근사 없이 순수하게 자원·불가능성까지 포함한 엄밀한 이론적 기준점을 제공한다.



### Nested Episodic State Topology (NEST): A Graph-Theoretic Architecture of Cognitive States (https://arxiv.org/abs/2607.06055)
- **Prior Approaches**: 인지과학은 기억·추론·지각·언어·제어·학습 등 각 영역에서 국소적으로는 성공한 이론/아키텍처가 많지만, 공통된 “표현 언어”가 부족해 이론 간 비교와 수정이 어렵다는 문제의식이 제시된다. 통합 아키텍처(예: 프로덕션/워크스페이스/제어루프)로 한 번에 묶는 방식은 가치가 있으나, 비교를 위해 특정 처리기제의 절차적 어휘로 강제 번역해야 하는 제약이 따른다.

- **Core Contribution**: 이 논문은 NEST(Nested Episodic State Topology)라는 그래프-이론 기반 인지 표현 온톨로지를 제안하며, 인지행위를 “완성된 경험적 모델”이 아니라 “구조화된 상태의 형성/변환”으로 모델링한다. 개념·에피소드·지각·태스크 컨텍스트를 typed, weighted 그래프로 통일하고, 인과·포함·시간·연관·증거·공간의 6개 관계 범주로 구조적 커밋을 명시한다. 또한 durable belief 그래프(장기 지식)와 capacity-limited working memory 그래프(일시적 비지식 콘텐츠)를 분리해, 비교 가능한 이론 표현을 위한 바닥층(substrate)을 만든다.

- **Technical Challenges**: 핵심 기여를 실현하려면 (1) 노드가 내부 subgraph 페이로드를 가질 수 있는 재귀적 구조를 허용하면서도 (2) 작동기억의 transient 구조를 저장지식과 어떻게 “검증/통합/충돌 처리/업데이트”할지 형식화해야 한다. NEST는 WM-belief grounding, conflict catalogs, belief-update operators를 통해 일시 구조가 저장 지식에 의해 테스트되고, 모순이면 belief가 어떻게 개정되는지 연산자 수준에서 정의한다. 더불어 activation, graph-property functionals, working-memory transitions, awareness/trajectory functionals 같은 연산자 툴킷으로 공식 핵심을 구성해 이후 도메인 확장을 위한 기반을 제공한다.

- **Empirical Impact**: 이 논문은 특정 단일 성능 모델을 목표로 하는 실험보다, 나중의 경험적·계산적 연구를 위한 “표현 투명성”과 “이론 비교”의 공통 바닥층을 제공하는 것을 우선한다. 후속 섹션에서 NEST의 새 원시 연산자를 추가하지 않고 phenomena signatures, 태스크 인스턴스화 스키마, 그리고 호환성 매핑을 통해 ACT-R, Soar, Sigma, Common Model of Cognition, Global Workspace Theory, semantic networks, Theory-Theory, chunking 등을 NEST 온톨로지의 특수/제약 영역으로 표현한다고 제시한다. 호환성 매핑은 여러 프레임워크가 서로 다른 형식으로 주장할 때도, 동일한 그래프-기반 언어 위에서 구조적 비교가 가능해질 수 있다는 점에서 의미가 크다.



### BlueMagpie-TTS: A Token-Efficient Tokenizer, Language Model, and TTS for Taiwanese-Accent Code-Switching Speech (https://arxiv.org/abs/2607.06054)
- **Prior Approaches**: 기존 Neural TTS는 대부분 멀티링구얼 tokenizer와 일반 언어 모델을 텍스트 전면(frontend)으로 사용해 대만 만다린에 충분히 적응하지 못했다. 그 결과 억양이 다른 만다린 변종으로 기본 설정되고, 대만에서 흔한 단어·표기·코드스위칭(중국어-영어 교차) 경계에서 발음과 운율 전환이 깨지기 쉽다.

- **Core Contribution**: 이 논문은 텍스트 측을 ‘아래에서부터’ 적응시키는 처방을 제안한다. 대만 컨텍스트에 맞춘 PangolinTokenizer(대만 기반 byte-level BPE)로 토큰 효율과 경계 처리를 개선하고, 이를 학습한 Barbet(전통 중국어 중심 10억 파라미터 언어 모델)을 텍스트-semantic frontend로 둔 뒤, BlueMagpie-TTS에서 VoxCPM2 음향 스택에 learned bridge로 결합해 대만 코드스위칭 TTS를 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 대만 텍스트의 국지 용어·혼성 문자/로마자·코드스위칭을 짧고 안정적인 토큰열로 표현하는 tokenizer 설계, (2) 텍스트 frontend와 고정된 음향 스택 간 표현공간 불일치 해결, (3) 코드스위칭 경계에서 운율 전환 계획이 무너지지 않게 jointly co-adaptation을 유도하는 것이다. 저자들은 대만 컨텍스트 데이터로 tokenizer를 스트리밍 방식으로 학습해 재현성과 효율을 확보하고, bridge를 통해 pretrained 음향 스택 입력공간으로 정렬한 뒤 bridge-only distillation 후 전체를 짧게 fine-tuning하는 2단계를 사용한다.

- **Empirical Impact**: 대만 로컬화 1,000문장 테스트에서 BlueMagpie-TTS는 CER을 11.45%에서 4.81%로, WER은 14.83%에서 5.36%로 각각 상대 58.0%, 63.9% 낮췄다. 또한 500문장 블라인드 청취 실험에서 10명의 청취자 다수결 기준 65.6%가 BlueMagpie-TTS를 선호했으며, 대만 특화 tokenizer·frontend 적응이 억양·코드스위칭 발음 품질을 실질적으로 끌어올린다는 점을 보여준다.



### PolyWorkBench: Benchmarking Multilingual Long-Horizon LLM Agents (https://arxiv.org/abs/2607.06008)
Comments:
          15 Pages, 6 figures

- **Prior Approaches**: 기존 LLM 에이전트 벤치마크는 대부분 단일 언어 가정하에 설계돼, 추론·도구 호출·출력 생성이 한 언어로만 이뤄지는 경우가 많습니다. 그 결과 다국어 입력/출력을 같은 워크플로에 넣었을 때 에이전트의 성능 저하 양상과 원인(언어 변화가 의사결정과 실행을 어떻게 흔드는지)이 충분히 다뤄지지 않았습니다. 또한 “정답 여부” 중심의 평가가 언어적 일관성까지 동시 반영하지 못하는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 다국어 장기 워크플로에서 LLM 에이전트를 평가하기 위한 PolyWorkBench를 제안합니다. 5개 도메인(상거래, 지식 업무, 법률 분석, 현지화, 제조)에서 총 67개 작업을 구성하며, 에이전트가 이질적인 다국어 입력을 처리하고 반복적 추론과 도구 사용을 거쳐 구조화된 출력을 생성해야 합니다. 아울러 구조적 채점, 실행 가능한 검증, LLM 기반 의미 평가를 결합한 하이브리드 평가 프레임워크도 함께 제안합니다.

- **Technical Challenges**: 다국어 워크플로에서는 언어 변이가 추론 단계와 실행/도구 호출 단계에 연쇄적으로 누적되는 ‘복합 효과’가 나타나 성능이 급격히 떨어질 수 있습니다. 이를 평가에 반영하려면 단순 정답 판정만으로는 부족하므로, 구조적 그레이딩으로 형식적 제약을 확인하고 executable verification으로 오류를 기계적으로 검증하며, LLM-based semantic assessment로 언어 일관성과 의미 정확성을 함께 점수화합니다. 이렇게 다층 신호를 통합해 기능적 정합성과 언어적 일관성을 동시에 측정하도록 설계했습니다.

- **Empirical Impact**: 실험 결과, 최신 LLM 에이전트는 단일 언어 설정 대비 다국어 워크플로에서 뚜렷한 성능 저하를 보였습니다. 분석에서는 다국어가 추론과 실행 전 과정에 걸쳐 누적적으로 악영향을 준다는 관찰을 제시하며, 에이전트 평가에서 언어 변이와 절차적 의사결정을 함께 모델링/측정하는 중요성을 강조합니다. PolyWorkBench와 하이브리드 평가 설계는 실제 업무형 다국어 에이전트 연구의 표준 벤치마크로 활용될 가능성이 큽니다.



### Integrating knowledge graphs and multilingual scholarly corpora for domain-adaptive LLMs in SSH (https://arxiv.org/abs/2607.05956)
Comments:
          8 pages, 4 tables, workshop LLMs4SSH of LREC 2026 conference

- **Prior Approaches**: 기존 문헌 탐색·요약 도구들은 영어 중심 학술지와 인용지표(impact factor, h-index 등)에 강하게 의존하는 경향이 있어 SSH의 다언어성, 단행본·디지털 에디션 같은 다양한 산출물을 충분히 반영하지 못한다. 또한 내용의 해석·맥락화가 핵심인 SSH 특성상, 단순 주제 매칭만으로는 방법론적 친화성과 관점 차이를 포착하기 어렵다.

- **Core Contribution**: 이 논문은 LLMs4EU/ALT-EDIC 맥락에서 ReSearch_SSH라는 사용 사례를 제시하며, SSH 연구 관행에 맞춰 foundation model을 도메인 적응하고 지식기반 RAG로 생성 결과의 출처 추적성을 강화한다. ISIDORE 같은 기존 연구 인프라 위에 고도 질의, 문헌 비교 분석, 문헌리뷰용 구조화 오버뷰 생성 기능을 얹어 “대화형 인터페이스”를 넘어 연구 흐름을 지원하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 탐색 쿼리의 탐구적·개방형 의미를 SSH 맥락에서 안정적으로 이해하고, (2) 생성 답을 환각 없이 인용 가능한 문서에 근거시키며, (3) 다언어(초기에는 이탈리아어/영어 중심, 프랑스 자료 기반)에서 성능과 신뢰도를 동시에 확보하는 것이다. 이를 위해 GraphRAG(지식그래프·메타데이터·문서 그래프) 기반의 검색-생성 구조를 채택하고, ISIDORE의 역사적 query–click 행동으로 retrieval을 미세조정하며, state-of-the-art 오버뷰 구성용 instruction tuning 및 전문가 패널 기반 분석을 함께 설계한다.

- **Empirical Impact**: 평가는 LLMs4EU 프로토콜을 따라 독립 외부팀이 시나리오 기반의 정량 벤치마킹(검색, 멀티문서 요약, traceability, hallucination detection)을 수행하고, 프랑스·이탈리아 Digital Humanities 전문가 패널이 학술적 신뢰성·방법론 적절성·실사용성을 질적으로 검증한다. 현재는 데이터 통합·거버넌스·평가 체계가 마무리 단계이며 초기 미세조정 실험과 전문가 패널 구성으로 “개념 설계→실증 검증” 전환을 앞두고 있다.



### CMDR: Contextual Multimodal Document Retrieva (https://arxiv.org/abs/2607.05927)
Comments:
          Accepted by ECCV 2026; project page: this https URL

- **Prior Approaches**: 기존 멀티모달 문서 검색 벤치마크는 질의-단일 페이지 간의 단순 어휘/의미 매칭을 주로 평가해, 여러 페이지에 걸친 문서 맥락을 이용한 간접적 추론 능력을 검증하지 못했다. 또한 대부분의 방법이 페이지를 독립 인코딩해 문서 전역 구조나 cross-page dependencies의 이점을 과소평가하는 한계를 보였다. 멀티홉 계열도 대체로 명시적 매칭에 기반한 직접 검색 성격이 강해, “질의에 직접 드러나지 않는 관련 페이지를 찾아야 하는” 문제를 충분히 다루지 못했다.

- **Core Contribution**: 논문은 Contextual Multimodal Document Retrieval(CMDR) 태스크를 제안하고, 이를 평가하는 CMDR-Bench를 공개한다. CMDR-Bench는 6개 도메인 255개 장문 문서(평균 183.5페이지)에서 수작업으로 큐레이션된 800개 질의로, 페이지 간 맥락 모델링이 필수인 간접 검색을 요구한다. 모델 측면에서는 문서 여러 페이지를 함께 인코딩하되 페이지 수준 임베딩으로 분리해 문맥을 반영하는 CMDR-Embed와, 이를 학습하는 Contextual Multimodal Contrastive Learning(CMCL)을 함께 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 “여러 페이지를 함께 인코딩해 맥락을 얻는 것”이 동시에 동일 문서 내 페이지 표현을 섞어 페이지별 구별력(discriminability)을 떨어뜨린다는 점이다. 이를 해결하기 위해 CMDR-Embed는 chunk-then-split 전략으로 컨텍스트를 공유하되 페이지 임베딩은 분리해 Late Interaction(LI)로 질의-페이지 매칭을 수행한다. 학습에서는 CMCL이 두 종류의 context-aware hard negatives(같은 chunk 내 In-Chunk Negatives, 같은 문서 내 먼 chunk의 In-Document Negatives)를 조합해 맥락 활용과 페이지 구별력을 균형 있게 최적화한다.

- **Empirical Impact**: 실험 결과 CMDR-Embed는 비문맥(non-contextual) 임베딩 대비 유의미하게 성능이 개선되며, 동일 학습 데이터 조건에서도 컨텍스트 모델이 우위를 보였다. CMDR-Bench의 카테고리 분석에서는 특히 CR/MR처럼 참조 해석과 다중 페이지 집계가 필요한 영역에서 기존 멀티모달 검색기가 더 큰 어려움을 드러냈고, 이 격차를 CMDR-Embed가 완화한다. 또한 CMCL의 hard negative 설계와 LI의 멀티벡터 구조가 성능 향상에 일관되게 기여함을 보여, 장문 문서 검색에서 context-aware multimodal embeddings의 필요성을 실증적으로 강조한다.



### PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails (https://arxiv.org/abs/2607.05910)
- **Prior Approaches**: 기존 비전 가드레일 벤치마크는 고정된 안전 분류 체계나 정적 harmfulness 정의에 의존해, 안전을 이미지의 고유 성질처럼 취급하는 경우가 많았습니다. 정책 조건을 넣는 평가도 있었지만, 같은 이미지에서 정책 경계가 바뀔 때 결정을 뒤집는 능력(정책-적응)을 미세하게 측정하기엔 부족했습니다. 그 결과 많은 모델이 위험 큐는 알아도 정책이 허용/차단을 바꾸면 판단을 안정적으로 수정하지 못하는 취약성이 드러났습니다.

- **Core Contribution**: 이 논문은 정책이 런타임마다 달라질 때(제품/지역/규정 변경 등) 동일 이미지가 서로 다른 결정을 받아야 하는 문제를 다루며 PolicyShiftBench와 PolicyShiftGuard를 제안합니다. PolicyShiftBench는 265개 이미지에 대해 총 2,000개의 정책-판별 인스턴스(7개 위험 카테고리 조합, 28개 정책 변형)를 구성하고, 같은 이미지의 pass/block “정책 플립”을 직접 평가하는 Policy Shift Score(PSS)를 도입합니다. PolicyShiftGuard는 policy-conditioned 가드레일로, Randomized Policy SFT(RP-SFT)와 Boundary-Pair Policy Adaptation(BP-Adapt) 2단계 학습을 통해 정책 경계에 맞춰 결정을 뒤집도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 모델이 이미지에서 위험 단서를 인식하는 것에 그치지 않고, 현재 주어진 정책 텍스트의 경계를 읽어 동일 시각 증거에 대해 안전 판정을 ‘조건부로’ 바꾸는 능력을 확보하는 데 있습니다. 논문은 이미지 속성(atomic attributes)과 정책 판단을 분리해 라벨을 실행 가능한 규칙으로 결정하고, BP-Adapt에서 같은 이미지·카테고리에 대해 “허용 정책 vs 차단 정책”이 매칭된 boundary-pair를 만들어 쌍대 비교 형태의 손실로 안전 점수의 여백을 강제합니다. 또한 RP-SFT에서 정책 제시 순서/표면 식별자 등을 랜덤화해 고정 템플릿이나 위치 기반 지름길 의존을 낮추고 정책-추종의 견고성을 높였습니다.

- **Empirical Impact**: 실험에서 기존 VLM 및 전문 가드레일은 F1은 그럴듯해도 PSS가 매우 낮아, 정책 플립에 취약함이 확인됐습니다. 반면 PolicyShiftGuard-7B는 PolicyShiftBench에서 Avg. F1 76.9, Avg. PSS 72.1을 달성하며, UnSafeBench와 SafeEditBench로도 전이 성능이 좋고 추론 속도까지 개선되어 latency–performance 트레이드오프에서 실용성을 높였습니다. 특히 ablation은 matched pass/block boundary pair 및 pair loss가 정책-적응을 안정적으로 만들기 위한 필수 요소임을 보여줍니다.



### K-ABENA: K-Adaptive Backpropagation with Error-based N-exclusion Algorithm : (Compensated Loss-Based Sample Exclusion with Unbiased Gradient Estimation) (https://arxiv.org/abs/2607.05903)
Comments:
          11 pages main text + appendices, 13 pages total. Code: this https URL

- **Prior Approaches**: 선택적 backprop(예: OHEM, SBP)은 손실이 낮은 샘플의 backward를 스킵해 계산을 줄이지만, 선택된 집합이 손실에 상관돼 gradient가 편향된다. 반면 Focal Loss 같은 soft reweighting은 전체 배치의 backward를 그대로 계산해 ‘선택에 따른 compute saving’을 제공하지 못한다.

- **Core Contribution**: 이 논문은 K-ABENA(K-Adaptive Backpropagation with Error-based N-exclusion)로, 저손실(minor) 일부를 backward에서 제외하되 설계 기반 역확률 가중치로 편향을 보정하는 프레임워크를 제안한다. 특히 canonical(v3)에서는 survey sampling의 Horvitz–Thompson(H-T) 아이디어를 선택적 backprop에 통합해, 설계 무편향 gradient 추정기와 실무용 self-normalized(Hájek) 변형의 편향(차수 O(1/m))을 함께 다룬다.

- **Technical Challenges**: 핵심 기술 난점은 ‘손실 기반 선택’이 만들어내는 선택 편향을 compute 절감과 동시에 제거하거나 제어하는 것이며, 이를 위해 defensive-mixture 샘플링 설계와 inclusion probability의 엄밀한 양의 하한(positivity)을 보장한다. 또한 HT 형태는 설계 무편향임을 보이고, self-normalized 형태는 분명한 bias floor와 SGD의 비볼록 수렴(기대 제곱 gradient 노름의 O(1/sqrt(T)) 보장 + 편향 잔차 항)을 정리해 이론적 안전성을 제시한다.

- **Empirical Impact**: 실험에서는 uncompensated(보정 없는) 선택 계열이 극단적 클래스 불균형/라벨 노이즈에서 구조적으로 실패하며, 한 synthetic 극단 불균형(0.17%) 설정에서 full-batch SGD가 0.9998 AUC인 반면 편향 변형은 0.53~0.62에 그친다. 반대로 compensated(v3/H-T 보정) 추정기는 동일한 28.4% compute savings 조건에서 0.9991 AUC를 달성하고, Breast Cancer·Digits·Wine·Diabetes 같은 실제 데이터에서도 full-batch SGD와 통계적으로 구분되지 않으면서 per-epoch gradient 계산의 28–54%를 절감한다.



### StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems (https://arxiv.org/abs/2607.05844)
Comments:
          Code and supplementary materials available at: this https URL

- **Prior Approaches**: 기존 에이전트 메모리 구현은 분기·재시도·복제 과정에서 생기는 관측 불일치를 대부분 덮어쓰기 규칙으로 숨기거나, 충돌을 보더라도 추적·수정이 어렵다는 한계가 있었다. CRDT/OpSet 같은 표준 합의·수렴 기반은 상태를 모을 수 있지만, 에이전트 관점의 “충돌을 언제·어떻게 드러내고 누가 선택/보류할지” 같은 계약(semantics contract)은 약하게 설계되는 경우가 많았다. 그 결과 검증 후에도 잘못된 행동으로 이어지거나, 이전 수정이 반영되지 않은 채 남는 문제가 발생할 수 있었다.

- **Core Contribution**: StateFuse는 표준 OpSet/CRDT merge 위에 얹는 “충돌 인지 replicated memory contract”를 제안한다. 새로운 join 대수를 만들기보다는, 불변 히스토리·명시적 conflict 객체·정확/의미 기반 correction handle(claim_id / claim_ref)·결정론적 predicate contract·프로젝션 시점의 제한된 resolution 권한을 묶어 에이전트가 감사 가능하게 충돌과 수정 가능성을 다루도록 한다. 또한 resolve가 replicated state를 재작성하지 못하게 해, 충돌을 공용 의사결정 표면에 일관되게 노출한다.

- **Technical Challenges**: 핵심 난제는 복제 병합은 수렴시키면서도 충돌을 “숨기지 않고” 유지하는 해석 규칙을 계약으로 고정하는 것이다. StateFuse는 claim을 Evidence/Claim/Retract/Decision의 불변 연산으로 저장하고, retraction이 claim_id 또는 claim_ref를 표적으로 삼아 exact/semantic correction을 각각 다르게 비활성화하도록 설계했다.さらに predicate registry의 normalize/equal 같은 결정론 규칙을 계약으로 강제하고, projection-time resolver는 후보 선택·abstain만 수행하되 base-memory mutation은 금지함으로써 결정론적 재현성과 충돌 표면의 일관성을 확보한다.

- **Empirical Impact**: MemoryAgentBench의 충돌을 포함한 282문항 슬라이스에서 StateFuse는 정확도에서 강한 flat/ collapsed 계열과 동률을 보이며, 보편적 accuracy 향상보다는 “무엇을 표면에 드러내는가”의 차이가 핵심으로 나타났다. 다만 StateFuse와 conflict-preserving baseline은 충돌-bearing 과제에서 모순을 끝까지 노출한 반면, raw-log 및 collapsed 표면은 이를 전혀 드러내지 않았다. 제어된 에이전트 루프에서는 ambiguity를 보존하고 검증 후에 abstain할 수 있는 설계가 붕괴(collapse)보다 훨씬 안전했으며, correction-handle ablation에서도 claim_ref가 의미 타깃 복구와 unseen-target no-resurrection을 더 잘 지원해 수정 표현력의 차이를 확인했다.



### TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training (https://arxiv.org/abs/2607.05804)
- **Prior Approaches**: on-policy distillation(OPD)은 학생이 자신의 궤적 위에서 강한 teacher와 맞추도록 학습해 언어 에이전트 학습에 유망한 프레임워크로 여겨져 왔다. 다만 장기 과업에서 OPD를 그대로 쓰면 전체 롤아웃이 꼬리 구간까지 진행되며 약하고 잡음이 큰 KL supervision을 낭비하는 비효율이 생기고, trajectory-level KL이 얕은 토큰에 손실을 집중해 깊은 의사결정 턴이 충분히 학습되지 않는 문제가 남는다.

- **Core Contribution**: 이 논문은 장기 에이전트를 위한 효율적 on-policy distillation을 목표로 TurnOPD를 제안한다. TurnOPD는 턴 단위 예산(budgeting) 관점에서 롤아웃 깊이와 KL 손실 가중치의 분배를 제어해, 시간은 쓰되 학습 신호가 약한 구간에는 덜 쓰고 중요한 의사결정 턴에는 더 고르게 학습되도록 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 학생의 자기 궤적에서 어떤 시점까지 롤아웃해야 KL 신호가 의미 있는지, (2) token-level로 쏠린 손실을 turn-balanced 형태로 점진적으로 재배분하는 방법을 동시에 설계하는 것이다. TurnOPD는 probe-based turn statistics로 adaptive rollout-depth budgeting을 수행해 롤아웃 길이를 동적으로 정하고, progressive turn-normalized loss budgeting으로 KL 가중치를 토큰에서 턴 균형 쪽으로 서서히 이동시켜 under-training을 완화한다.

- **Empirical Impact**: 실험은 ALFWorld, WebShop, Multi-Hop Search에서 task-specialized teacher 모델을 사용해 검증되었으며, TurnOPD는 동일한 wall-clock 학습 예산 조건에서 vanilla OPD보다 더 높은 validation accuracy를 달성한다. 또한 accuracy--time frontier를 기존 OPD를 넘어 확장함으로써, 장기 지평 에이전트 학습에서 OPD의 실용성을 높이는 방향의 설계 원리를 제시했다.



### Memory in the Loop: In-Process Retrieval as ExtendedWorking Memory for Language Agents (https://arxiv.org/abs/2607.05690)
- **Prior Approaches**: 언어 에이전트는 보통 observe–reason–act 루프를 돌리지만, 추론에 쓰는 메모리는 루프 밖의 외부 저장소(RAG/DB)로 두고 보통 턴당 1회 정도만 조회하도록 설계돼 왔다. 네트워크/디스크 저장소의 높은 지연으로 인해 in-loop(루프 내부)에서 매 스텝 읽기·쓰기를 하면 전체 엔드투엔드 지연이 크게 늘어나는 문제가 핵심 한계로 취급돼 왔다. 기존 연구는 그 비용을 서빙 레이어 스케줄링으로 가리거나, memory-first처럼 조회 빈도를 턴당 1회로 줄이는 방식으로 해결해 왔지만, ‘지연이 왜/얼마나 필연적인가’ 자체는 충분히 도전하지 않았다.

- **Core Contribution**: 이 논문은 메모리가 루프 안으로 들어가 매 스텝마다 read/write되는 “memory in the loop”를 다루되, 병목의 본질이 ‘패턴’이 아니라 ‘저장소가 어디에 있느냐(지연 속성)’라고 주장한다. 특히 in-process(프로세스 내부) 저장소처럼 약 100μs 수준으로 조회가 가능하면, 루프 내부 조회로 인한 지연 증폭이 붕괴하며 메모리 사용이 실질적으로 가능해진다고 제시한다. extended-mind의 parity principle을 공학 기준(추론 단계 대비 지연 예산 충족)으로 재해석해, 빠른 저장소는 단순 도구 조회가 아니라 구성적 working memory로 기능할 수 있음을 논문 전개의 핵심으로 둔다.

- **Technical Challenges**: 가장 큰 기술적 난제는 네트워크 기반 저장소에서의 조회 지연(대략 수십~수백 ms)이 매 스텝 반복될 때 end-to-end 지연이 폭발하는 점이다. 논문은 저장소를 in-process로 옮겨 저장소 연산 시간을 p50 80~165μs로 측정하고, 동시에 네트워크 임베딩이 남는 지배 항목임을 찾아 로컬 임베더를 붙여 전체 연산을 약 40μs 수준에 가깝게 되돌리는 방식으로 지연을 근본적으로 줄였다. 더 나아가 ‘저장소 답 자체는 동일하지만 조회가 예산 안에 들어오느냐만 달라지는’ causal 실험(루프 가드)을 설계해, 지연이 결과를 바꾸는 경로가 ‘조회 오류’가 아니라 ‘검사가 수행되지 못하는 빈도’임을 분리해 보여준다.

- **Empirical Impact**: GPT-5 계열 4개 모델에서 컨텍스트 윈도우가 제한된 5개 제약(메모리 회상) 과제를 수행한 결과, in-loop 메모리에서 recall이 0/5에서 3.6~4.8/5 수준으로 유의미하게 상승했다. 또한 저장소 write는 누락 없이 유지(244/244 기록 보존)됐고 miss는 저장소 문제가 아니라 에이전트의 read 정책에서만 추적됐으며, 이는 설계된 in-loop 사용의 타당성을 뒷받침한다. ‘루프 가드’ 실험에서는 저장소 지연이 커질수록 중복 행동이 단조 증가하는 용량-반응이 관측돼, 이 연구가 단순 최적화가 아니라 “지연이 실제 성능 경계(가능/불가능)를 인과적으로 결정한다”는 메시지를 경험적으로 강화한다.



### Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction (https://arxiv.org/abs/2607.05577)
Comments:
          23 pages, 4 figures; 9-page main text plus appendix. Preprint

- **Prior Approaches**: 기존 RAG와 agent-memory, 그리고 GraphRAG·Graphiti 같은 그래프 기반 접근은 근거가 되는 텍스트/에피소드를 찾아 “정답에 가까운 증거”를 제공하는 데 초점을 둔다. 하지만 소설의 다중 홉 narratological 질의(누가 언제 알았는지, 사건-발화 순서 차이, 떳다-갚기, 관계 변화 등)에 필요한 ‘서사 구조가 반영된 상태’가 표현·추적되지 않아 엉뚱한 증거가 나오거나 아예 근거가 부재한 문제가 반복된다. 특히 일반 엔티티/이벤트 그래프는 시점별 “관찰자 시야”, “드러냄(order of reveal) vs 사건(order of event)”, “약속의 성립/해소” 같은 제작자가 필요한 타이핑된 시간-서사 상태를 1차 항목으로 다루기 어렵다.

- **Core Contribution**: 이 논문은 Narrative World Model(NWM)이라는 작가용 메모리 시스템을 제안하며, 서사 이론에 기반한 타이핑된 temporal-state graph와 질의 조건형 hybrid retrieval을 결합한다. 메모리는 단순 요약이나 덩어리 텍스트가 아니라, 확정된 장(chapter)의 evidential span을 붙여 “누가 무엇을 언제 알았는지/관계가 어떻게 바뀌었는지/약속이 어떻게 기능했는지” 같은 서사 상태를 저장하도록 설계됐다. 또한 답변 성능을 조작하는 요소를 줄이기 위해, 동일한 Opus 4.8 리더가 각 시스템의 ‘장 안전(chapter-safe)’ 근거만 읽고 판단하도록 평가 프로토콜을 고정한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 미래 장을 읽지 않으면서도 (2) 여러 장에 걸친 질의에 필요한 “현재 시점의 서사 상태”를 재구성하고 (3) 그 질의에 맞는 증거만 압축해 제시하는 것이다. NWM은 장 게시(publish) 흐름에서 Sonnet 4.5 추출기로 타이핑된 서사 메모리 레코드를 만들고, temporal KG의 validity interval로 as-of 시점 상태를 계산하며, BM25+벡터 기반 검색 뒤에 1-hop 타이핑 이웃을 확장해 bounded 패킷으로 증거를 전달한다. 더 나아가 RLM QA 검증 레이어로 질의 분해-근거 수집-상태 변화 지지 여부를 재확인해, 서사 상태의 시점 오해를 줄이도록 한다.

- **Empirical Impact**: 실험은 공개/비공개 두 코퍼스와 검증된 multi-hop 벤치마크(사유된 narratological 176문항 등)에서 진행됐고, NWM Graph Retrieval이 Graphiti 대비 큰 폭으로 향상됐다. 비공개 multi-hop 슬라이스에서 정확도는 0.898 대 0.574였고(p<1e-5), 공개 576문항에서도 0.625 대 0.516으로 유의미하게 우세했다. 특히 Graphiti를 같은 추출기로 맞추거나 더 저렴한 추출기로 재임포트해도 격차가 유지되어, 성능 향상이 추출 품질이나 그래프 크기 같은 부가 요인이 아니라 “서사 구조를 타이핑해 표현한 표현력”과 “질의 조건형 검색”에서 온다는 점을 실험적으로 확인했다.



### Decision Protocols in Multi-Agent Large Language Model Conversations (https://arxiv.org/abs/2607.05477)
Comments:
          Master's thesis, University of Göttingen

- **Prior Approaches**: 기존 연구는 LLM의 멀티에이전트 활용 시 여러 에이전트를 두더라도 단일 의사결정 프로토콜을 쓰거나, 적용 범위가 좁은 데이터셋에서만 성능을 확인하는 경우가 많았다. 또한 멀티에이전트는 학습 비용을 줄일 수 있지만, 에이전트 간 토론·결정 과정으로 테스트 시간이 늘어날 수 있어 프로토콜 설계가 핵심이라는 점이 반복적으로 지적돼 왔다.

- **Core Contribution**: 이 논문은 Multi-Agent LLM(MALLM) 프레임워크를 제안해 투표(voting), 합의(consensus), judge 기반 결정(judge decision) 등 다양한 의사결정 프로토콜을 구현하고 체계적으로 평가한다. 대화형 과업 해결을 목표로, 여러 에이전트가 논의하며 최종 해답에 도달하는 과정을 프로토콜 단위로 시뮬레이션한다는 점이 차별점이다.

- **Technical Challenges**: 핵심 기술 난제는 에이전트들이 만들어낸 답변을 어떤 규칙으로 통합해 ‘정답으로 수렴’시키는지에 대한 설계였다. 논문은 각 프로토콜별 협업 방식을 비교하고, 특히 독립적인 솔루션 생성을 통해 response diversity를 늘리면 의사결정 품질이 향상됨을 확인했으며, 의사결정 과정에서의 정보 접근 방식 변화는 성능에 큰 영향을 주지 않는 경향을 관찰한다.

- **Empirical Impact**: 실험은 지식 기반 벤치마크(MMLU, MMLU-Pro, GPQA)와 논리 기반 벤치마크(StrategyQA, MuSR, Math-lvl-5, SQuAD 2.0)를 폭넓게 포함해 프로토콜-과업 유형 간의 상호작용을 보여준다. 결론적으로 consensus는 지식 집약 과업에서, voting 및 judge 프로토콜은 논리 과업에서 더 유리했으며, 에이전트 간 다양성 확보가 전반적 성능 향상에 기여한다는 실증적 근거를 제공한다.



### Prompt-to-Paper: Agentic AI System for Bioinformatics (https://arxiv.org/abs/2607.05456)
Comments:
          NA

- **Prior Approaches**: 기존 연구들은 지식그래프 기반 가설 생성이나 retrieval을 강화하는 방식, 혹은 모의환경/시뮬레이션 평가에 치중해 왔습니다. 다만 생성된 인용·주장에 대한 실시간 검증이 약하거나, 실험 수치를 실제로 실행하지 않고 합성값을 넣는 한계가 반복적으로 드러났습니다. 또한 자동 평가는 주로 단일 축 또는 시뮬레이터 의존 형태여서, 출판 수준의 엄격함을 재현 가능하게 담보하기 어렵다는 문제가 남아 있었습니다.

- **Core Contribution**: 이 논문은 Prompt-to-Paper로, 생물정보학 논문 생성을 “검증된 문헌 근거 + 실제 실험 실행 + 다차원 품질 평가”로 end-to-end 정렬하는 멀티에이전트 프레임워크를 제안합니다. 핵심은 주장마다 deterministic RAG 기반 grounding을 수행하고, 자율 coding agent가 계산을 실제로 실행해 숫자를 논문에 주입하며, 8개 품질 차원에 hallucination penalty까지 포함한 자동 평가를 표준화했다는 점입니다.

- **Technical Challenges**: 기여를 현실화하는 기술 난제는 (1) 생성 시점에 문헌 근거가 흐려지는 문제, (2) 실험 수치가 합성되는 문제, (3) 품질 점수가 반복 개선에서 흔들리는 문제였습니다. 이를 위해 section-aware relevance scoring과 snowball citation expansion으로 60–100편 corpus에 claim을 고정하고, sandbox에서 Python 실험을 실제 실행(재현 가능한 rigor.py 통계 포함)해 canonical results.json을 주입하도록 설계했습니다. 또한 deepseek-v4-pro/ deepseek-chat의 역할 분리를 통해 G-Eval류 채점의 구조화 실패를 줄였고, 8차원 scorer는 z-normalisation과 차원별 안정화(rejudge 조건), citation-range 체크 및 [N] 범위 외 인용 페널티를 적용해 개선 루프의 방향성을 보존합니다.

- **Empirical Impact**: 5개 생물정보학 케이스에서 시스템은 before/after 각각 submission-formatted PDF를 생성했고, 모든 run에서 zero out-of-range citations를 달성했습니다. 개선 루프는 평균 +17.96점(0–100) 향상을 보였으며, CpG Island Detection에서는 최대 +26.04점까지 상승했습니다. 자동 점수(평균 63.56/100) 외에도 독립 LLM 3종 및 인체 평가에서 B–B+ 범위와 유사한 경향이 관측돼, “실제 실험·검증·표준화된 품질 게이트”가 통한다는 실증적 근거를 제공합니다.



### Linking Hadith Narrator Identities Across Heterogeneous Arabic Biographical Databases: A Multi-Signal Entity Resolution Pipelin (https://arxiv.org/abs/2607.05424)
Comments:
          16 pages, the data sets available at DOI: https://doi.org/10.1016/j.dib.2022.108065

- **Prior Approaches**: 기존 연구는 하디스 전승 사슬(sanad)에서 그래프 분석을 하더라도 단일 데이터베이스에 갇혀 생물(인물) 노드의 신뢰도·사망연도 같은 메타데이터를 확장하기 어려웠습니다. 또한 기존 narrator disambiguation은 AraBERT 기반 등 ‘닫힌 세계(closed-world)’ 분류에 초점을 두어, 서로 다른 전기(傳記) DB 간 인물 이름을 외부로 연결하는 ‘열린 세계(open-world)’ 문제에는 부족했습니다.

- **Core Contribution**: 이 논문은 Sanadset 650K(650,986 하디스 레코드)의 narrator 이름을 두 전기 DB(Hadithtransmitters/hawramani, Muslimscholars)에 단계적으로 연결하는 2-phase entity resolution 파이프라인을 제안합니다. Phase 1은 이름 유사도만으로 Sanadset→hawramani를 매칭하고, Phase 2는 이름 유사도·사망연도 근접성·신뢰도(grades) 극성 등 다중 신호로 hawramani↔muslimscholars를 교차검증해 전파 가능한 링크를 만듭니다.

- **Technical Challenges**: 핵심 난제는 역사적 아랍 인명에서 나타나는 정규화 편차(diacritic, Alef variants, ta marbuta 등)와 kunyah/ism 변형, 그리고 nasab 체인의 토큰 길이 차이로 인한 이름 유사도 왜곡입니다. 저자들은 도메인 특화 Arabic normalization과 빅램(prefix) 인덱스로 후보군을 줄인 뒤, Phase 1은 TSR 기반 fuzzy 매칭(이름만), Phase 2는 가중 다중 신호 스코어(사망연도 희소성 고려한 동적 가중)를 사용해 오탐을 억제했습니다.

- **Empirical Impact**: 실험 결과 Phase 1에서 185,216개 narrator name variant 중 94,628개(51.1%)를 hawramani에 연결했고, Phase 2에서는 hawramani 기준 94.7%에 해당하는 95,573개 링크를 추가로 확보했습니다. 최종적으로 185,216노드·814,093엣지 규모의 방향성 전승 그래프에 교차 출처 전기 메타데이터를 풍부화했으며, sanad_links·narrator_links 및 그래프를 오픈 리소스로 공개해 디지털 이슬람 인문·하디스 인증 분석의 기반을 넓힐 것으로 기대됩니다.



### CANONIC: Governance Is Compilation (https://arxiv.org/abs/2607.05410)
Comments:
          28 pages, 4 figures. Pre-registered cross-provider evaluation harness and per-regime results at this http URL. Construction claims resolve to commands run at the evidence-window-close ref (see Appendix C)

- **Prior Approaches**: 기존 AI 콘텐츠 거버넌스는 주로 생성 후 탐지·공개·리뷰·스타일 점검처럼 ‘사후 품질/신뢰도 필터’에 의존해왔다. 하지만 CANONIC은 이런 방식이 정작 ‘코퍼스에 무엇을 들일지’의 구조적 기준을 강제하지 못해 결국 같은 종류의 slop이 축적된다고 본다. 또한 단순한 AI 생성 여부나 읽기 좋음 같은 축으로는 저신뢰 텍스트와 고신뢰 텍스트를 안정적으로 가르기 어렵다고 지적한다.

- **Core Contribution**: CANONIC은 LLM이 만든 산문을 ‘진실 판정’이 아니라 ‘컴파일처럼 문서가 규격에 맞게 연결됐는지(증거가 닻을 내렸는지)’로 코퍼스 경계에서 검사하는 governed intelligence를 제안한다. 이를 위해 Triad, Inheritance, Introspection 세 가지 공리를 문법·스코프 해석·타입 시스템에 1:1로 대응시키고, 승인/거절을 결정하는 admission 절차를 선형 시간의 판정 문제로 만든다. 결과적으로 슬롭(slop)을 기계가 판정하지는 못하되, 무엇이 ‘감사 가능하게’ 기록되었는지는 end-to-end로 재현·검증 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 구조적 체크(정의/증거/범위가 존재함)가 실제 내용의 진실성(truth)과 상관되지 않을 수 있다는 점이다. CANONIC은 이를 인정하면서도, validator가 품질을 점수화하는 대신 ‘presence’만 확인하도록 3단 게이트(용어 VOCAB, git 기반 evidence ledger, 증거 evidence window)를 이진적으로 강제해 불계획·무근거 입력이 코퍼스에 들어오는 경로를 차단한다. 동시에 ‘가짜인데도 존재를 만들 수 있는’ 진공 충족·증거 위조·창(window) 조작 같은 우회가 가능함을 분석하고, 그 한계를 독립적 의미 검증으로 메우는 방향이 필요하다고 결론낸다.

- **Empirical Impact**: 논문은 사전 등록된 cross-provider 벤치마크에서 구조적 admission이 slop을 실질적으로 막지 못한다는 결과를 보이며, slop이 알고리즘이 계산하는 성질이 아니라 도메인 전문성의 verdict임을 경험적으로 뒷받침한다. 다만 구조적 게이트가 달성하는 것은 ‘진실 필터’가 아니라 ‘감사 가능성(accountability)의 경계 설정’이며, 정의·커밋·증거 창으로 모든 주장에 대한 추적 경로를 제공한다. 결론적으로 학계/임상/정책처럼 검증 비용이 큰 영역에서, 생성 속도를 따라잡는 대신 ‘감사와 재현이 가능한 기록 구조’를 표준화하는 방향의 영향력이 크다고 평가된다.



### CCBENCH: Assessing LLM Cultural Competence via Implicitly Signaled Norms using Health Queries (https://arxiv.org/abs/2607.05405)
Comments:
          34 pages

- **Prior Approaches**: 기존 문화 관련 평가들은 대개 사용자가 ‘어느 문화에 속하는지’를 명시적으로 밝히는 방식(이진 속성)이나, 문화를 몇 개 변수로 축약해 지식/형식의 정답성만 보려는 경향이 컸습니다. 그 결과 실제 대화에서 문화적 가치는 암묵적으로 드러나며 개인마다 규범을 섞어 따르는 ‘연속체’라는 점이 충분히 반영되지 못했습니다.

- **Core Contribution**: 이 논문은 문화적 역량을 ‘문화 소속’이 아니라 ‘규범 준수(norm adherence) 상태의 연속체’로 재정의하고, LLM이 대화 맥락에서 이를 추론해 반응하는지를 평가하는 CCBENCH를 제안합니다. 의료 사례로 CCBENCH-Health를 만들었고, 6개 문화에 대해 이론적 근거가 있는 60개 페르소나와 실제 포럼 기반 52개 건강 질문(총 3,120개 상호작용)을 구성해 측정합니다.

- **Technical Challenges**: 핵심 난제는 사용자가 문화적 배경을 직접 말하지 않는 환경에서, LLM이 규범 준수 상태를 ‘대화 히스토리’에서 암묵적으로 읽어내야 한다는 점입니다. 이를 위해 값(value)-규범(norm) 계층 구조로 페르소나를 생성하고, 규범별 체크리스트 기반의 채점으로 응답이 ‘따름/회피/중립’ 중 페르소나 의도에 맞는지를 LLM 평가기로 정량화했습니다.

- **Empirical Impact**: 다섯 개 주요 LLM 모두에서 문화적으로 적절한 응답률이 20~30% 수준에 머물렀고, Culture-CoT처럼 대화 이력의 문화 단서를 단계적으로 보게 해도 개선은 평균 3~5%p에 그쳤습니다. 특히 ‘규범을 따르는(Follow)’ 경우가 ‘규범을 피하는(Avoid)’ 경우보다 훨씬 어렵고, 아프가니스탄 맥락에서는 성능이 8.8%로 낮아 암묵적 서구 기준선 편향이 지속됨을 시사합니다. 반면 특정 맥락의 표면적 커뮤니케이션 스타일은 상대적으로 더 잘 맞추기도 하지만, 전반적 결핍은 “명시적 메타데이터 제공만으로는 해결되지 않는다”는 방향으로 귀결됩니다.



### Contextual Semantic Relevance and Word Surprisal Predict N400 and P600 Dynamics During Naturalistic Reading (https://arxiv.org/abs/2607.04107)
- **Prior Approaches**: 기존 연구에서는 단어 surprisal(기대 위반 정도)이 N400 같은 신경 반응을 설명하는 핵심 지표로 자리 잡았지만, 자연스러운 담화 읽기에서 의미적 ‘국소 적합’이 lexical expectation을 넘어서는지 는 불명확했습니다. 한편 메모리 기반 관점에서는 문맥에서의 의미 유사도/관련성이 처리 부담을 좌우한다고 보며, 여러 semantic metric이 자연읽기에서 가능성을 보여줬습니다. 다만 surprisal과 semantic relevance를 동시에, 그리고 N400·P600을 넓은 채널 커버리지로 비교해 검증한 연구는 상대적으로 적었습니다.

- **Core Contribution**: 이 논문은 DERCo의 단어-락 EEG(22명, 32채널)에서 contextual semantic relevance(주의-인식 attention-aware 방식의 국소 의미 연결성)를 계산해, GPT 기반 word surprisal과 함께 N400 및 P600 구간을 예측하는지 평가합니다. 특히 contextual semantic relevance가 어휘 기대(lexical expectation)와는 다른 시간·두피 패턴을 보이며, lexical 변수까지 통제한 상태에서도 설명력을 추가로 제공함을 보여줍니다. 결과적으로 자연읽기는 기대(예측)와 국소 의미 통합이 함께 작동하며, contextual semantic relevance는 담화 의미 적합과 ERP 동역학을 잇는 해석 가능한 계산적 연결고리를 제안합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 두 예측 변인(surprisal vs semantic relevance)을 동시에 넣되 공변량(단어 길이·빈도 등)과 반복 관측을 적절히 통제하고, (2) 채널·시간 전반에 걸쳐 성분별(N400/P600) 고유 기여를 분리하는 것입니다. 이를 위해 window-based N400(300–500ms)·P600(500–800ms) 평균 전압을 종속변수로 하는 mixed-effects 계열 분석(GAMM)과 rERP 기반 시간-해상도 회귀를 함께 수행하고, FDR 보정으로 다중비교 위험을 줄였습니다. 또한 semantic relevance는 최근 3개 선행 단어만 사용하도록 설계해 RSVP 읽기에서의 국소 정보 가용성 가정을 반영했습니다.

- **Empirical Impact**: 실험 결과 두 지표 모두 EEG 반응과 신뢰롭게 연관되지만, timing과 두피 분포에서 부분적으로 다른 양상을 보였습니다. surprisal은 주로 expectancy 관련 변화를 포착한 반면, contextual semantic relevance는 N400과 P600 윈도 평균 전압 전반에 걸쳐 일관된 효과를 보였고 특히 P600 구간에서 모델 비교상 설명력이 더 강하게 확인됐습니다. 이는 자연읽기에서 예측·기대뿐 아니라 국소 담화 의미 통합이 ERP(특히 P600)의 변동에 실질적으로 기여한다는 점을 실증적으로 지지하며, 이후 ‘의미 적합’을 예측 신경언어모델과 연결하려는 후속 연구에 준거가 될 만한 해석 틀을 제공합니다.



### Rethinking Scientific Discovery in the Agentic Era (https://arxiv.org/abs/2607.03863)
Comments:
          26 pages, 7 figures

- **Prior Approaches**: 기존 AI4Science 시스템은 문제정의, 문헌 근거, 모델·시뮬레이션 실행, 검증, 지식 재사용까지가 사람의 조율에 크게 의존하는 ‘조각난 도구’에 머무는 경우가 많습니다. 그 결과 전체 연구 흐름이 추적 가능하지 않거나, 재현·재사용이 어려워 긴 호흡의 발견 작업을 안정적으로 운영하기 힘들었습니다.

- **Core Contribution**: 이 논문은 에이전트 기반 과학 운영체제 SCION(Scientific Collaborative Innovation with Agentic Organizational Nexus)을 제안하며, 연구를 연결하는 ‘organizational nexus’ 역할을 수행하게 합니다. 핵심은 Research Execution Plan(REP)로, 고수준 과학 의도를 단계별 목표·의존성·검증 체크포인트·도구 요구·산출물·대체 조건으로 컴파일해 실행 가능하고 감사(auditable)하며 재사용 가능한 프로세스로 바꾸는 것입니다.

- **Technical Challenges**: 장기 과학 탐색에서는 (1) 계획을 실행으로 변환하는 구조화, (2) 검증 실패 시의 복구, (3) 에이전트 협업과 컨텍스트 관리, (4) 지식이 누적되는 메모리 설계가 어려운 기술 과제입니다. SCION은 hierarchical multi-agent execution, profile-driven specialization, selective context construction, governed delegation, layered epistemic memory를 결합해 Target-conditioned Inverse Search를 구성하고, hidden-target 설정에서는 유한 실험 예산 하 batch active search로 확장합니다.

- **Empirical Impact**: 재료 분석, 분자 설계, 단백질·항체 스크리닝 등 여러 응용과 과학적 읽기·아이디어 생성·분자 생성·항체 스크리닝 실험에서 SCION은 기존 autonomous research-agent baseline보다 분해·검증·정제·메모리 재사용 측면에서 우수한 성능을 보였습니다. 전반적으로 SCION은 AI를 고립된 도구에서 벗어나 추적성과 재사용성을 갖춘 ‘조율된 운영 레이어’로 옮긴다는 점에서 AI4Science의 실행 관점에 의미 있는 전환을 제시합니다.



New uploads on arXiv(cs.IR)

### Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression (https://arxiv.org/abs/2607.06036)
Comments:
          The 1st Late Interaction Workshop (LIR) @ ECIR 2026

- **Prior Approaches**: Late interaction 모델(예: ColBERT)은 토큰 임베딩을 여러 개 저장하며, 이로 인해 문서당 벡터 수와 저장·메모리 비용이 커진다는 문제가 있었다. 이를 줄이기 위해 pruning으로 덜 중요한 토큰을 제거하는 방식과, averaging으로 토큰을 뭉치는 pooling 방식이 제안됐는데, 후자는 training 없이 inference 단계에서 압축을 적용한다는 장점이 있다. 다만 pooling-aware training을 대규모 contrastive 학습으로 수행한 최근 연구는 많은 계산 자원과 구현 난이도가 있어 실무 적용이 제한적이라는 한계가 있었다.

- **Core Contribution**: 이 논문은 기존 ColBERT에 대해 training-free inference pooling(순차/계층/k-means)만 쓰는 것보다, 매우 가벼운 pooling-aware fine-tuning이 압축 품질을 더 끌어올릴 수 있는지 검증한다. 특히 k-means 기반 pooling을 조금만 학습 루프에 포함해도 inference-only pooling 대비 폭넓은 성능 이득이 나타나며, 다른 pooling 방식/데이터셋으로도 전이되는 징후를 제시한다. 또한 여러 pool factor를 함께 학습하는 multi-factor fine-tuning으로 압축 수준이 달라도 하나의 모델이 두루 잘 동작하도록 만드는 방식을 제안한다.

- **Technical Challenges**: 핵심 난제는, pooling이 토큰 간 미세 상호작용을 평균화로 훼손할 수 있어 retrieval 정확도를 유지하기가 어렵다는 점이다. 저자들은 teacher(리랭커) 점수에 대한 distillation loss를 쓰되, forward pass 안에서 document 쪽에 pooling을 적용해 MaxSim 점수와 loss 계산이 ‘풀링된 표현’ 기준으로 일어나도록 최적화한다. clustering 기반 pooling의 클러스터 할당은 그래프 밖에서(detached) 계산하지만, 클러스터 평균 임베딩은 그래프 안에서 생성해 gradient가 인코더로 흐르도록 설계했으며, multi-factor 학습에서는 배치마다 pool factor를 샘플링해 p=1(무풀링)까지 포함한 암묵적 정규화를 노린다.

- **Empirical Impact**: 실험에서 계층적(hierarchical) pooling은 inference-only 기준으로 전반적 성능이 가장 좋았고, k-means는 pooling-aware fine-tuning에서 가장 일관된 개선을 보였다. 특히 SciFact에서 가장 강력한 모델은 BEIR SciFact의 unpooled baseline을 유지하면서 pool factor 1~6 구간에서 이를 능가했으며, 무비용에 가까운 정확도 손실로 벡터 압축률 83%를 달성했다고 보고한다. 또한 fine-tuning이 다른 데이터셋의 성능을 해치기보다는 pool factor 2 이상에서 오히려 일반화에 긍정적으로 작용할 수 있음을 보여, 실무적으로 ‘작은 학습 투자로 큰 압축 효과’를 기대할 수 있다는 의미가 크다.



### Uncertainty-Aware Cross-Modal Remote Sensing Image-Text Retrieval via Evidential Learning (https://arxiv.org/abs/2607.06032)
- **Prior Approaches**: 기존 cross-modal remote sensing image-text retrieval(CMRSITR)은 테스트 시에 센서·대기 요인으로 인한 영상 열화와 텍스트 측 RS 용어 이질성 같은 비이상 조건이 생겨도, 각 쿼리에 대해 확신도 구분 없이 동일한 방식으로 검색을 수행하는 경우가 많다. 그 결과 쿼리마다 다른 불확실성을 반영하지 못해 비신뢰적인 retrieval 결과가 나타날 수 있다.

- **Core Contribution**: 이 논문은 uncertainty-aware retrieval을 목표로 evidential learning 기반 CMRSITR(ELC)을 제안한다. EDL로 이미지-텍스트 대응을 Dirichlet 분포로 모델링해 쿼리별 불확실성을 추정하고, 이를 retrieval 정오답과 정렬시키는 학습 및 불확실성 기반 디퍼(계류)·정제 전략을 결합한다.

- **Technical Challenges**: 핵심 난제는 (1) 영상 열화·텍스트 이질성이 섞인 상황에서 inter-modal 대응을 신뢰도 있게 확률적으로 모델링하고, (2) 추정한 불확실성이 실제 retrieval correctness와 잘 맞도록 정렬하며, (3) 학습 가능한 인코더가 Dirichlet 분포를 더 구별력 있게 만들도록 내부 유사성 구조를 주입하는 것이다. ELC는 EDL로 Dirichlet 기반 불확실성을 얻고, uncertainty-correctness alignment learning(UCL)로 불확실성과 정오답을 맞춘 뒤, intra-modal relationship learning(RL)로 mentor 인코더의 intra-modal similarity 구조를 증류해 분별성을 강화한다.

- **Empirical Impact**: 실험에서 ELC는 최신 CMRSITR 대비 경쟁력 있는 retrieval 성능을 보이면서, 센서·대기 관련 영상 섭동과 RS-vocabulary heterogeneity처럼 RS 특화 열화 조건에서 더 강한 견고성을 확인했다. 즉, 단순 정확도뿐 아니라 비이상 테스트 상황에서도 불확실성을 활용해 신뢰도 높은 검색/정제를 수행하는 점에서 의미가 크다.



### Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search (https://arxiv.org/abs/2607.05970)
Comments:
          5 pages, 1 figure, accepted at SynthIR @ SIGIR 2026

- **Prior Approaches**: 기존 dataset search는 제목·설명·키워드 같은 메타데이터를 기반으로 데이터 원본을 직접 다루지 않는 경우가 많아, 메타데이터 품질이 검색 성능을 좌우한다. LLM로 자동 설명을 생성해 검색 유틸리티를 높이려는 시도도 있으나, 생성물이 실제 RDF 근거와 얼마나 일치하는지(grounding/faithfulness)는 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 RDF 데이터셋의 메타데이터를 생성하는 6가지 설정(단순 rewrite, profile 기반 rewrite/gen, profile title 보조, graph 접근 agentic generation)을 한 프레임에서 비교해, 검색 성능과 faithfulness를 함께 평가한다. 특히 ‘검색이 좋아지면 메타데이터도 좋아진다’는 가정을 검증하며, 효과와 신뢰(근거/프로비넌스 비용)를 동시 관점에서 재정의한다.

- **Technical Challenges**: 핵심 기술적 난제는 생성물이 쿼리 매칭에 유리한 확장을 하면서도 원천 RDF 또는 제공된 증거로부터 나온 주장만 유지하도록 만드는 것이다. 연구진은 dataset profile를 표준화된 압축 증거로 사용하고, claim-level로 원자 단위를 추출한 뒤 설정별 권위 증거(원 메타데이터/프로필/전체 그래프)에 대해 LLM judge(일부는 RDF-grounded agentic judge)로 supported/contradicted 등을 판정해 trade-off를 분해 측정한다.

- **Empirical Impact**: ACORDAR 2.0의 약 1,000개 중 1,000문서 예산 파일럿과 150개 데이터셋 claim-level faithfulness 분석에서, unconstrained metadata rewrite가 retrieval 성능(NDCG@10 등)을 가장 크게 올리지만 faithfulness는 가장 낮았다. 반대로 profile-grounded rewriting은 검색 성능과 근거 일치 사이의 균형이 가장 좋아 ‘synthetic metadata는 시스템 수준 IR 문제’라는 결론을 강화하며, 향후 RDF profiling 고도화와 agentic grounding 통제가 중요하다는 방향성을 제시한다.



### CMDR: Contextual Multimodal Document Retrieva (https://arxiv.org/abs/2607.05927)
Comments:
          Accepted by ECCV 2026; project page: this https URL

- **Prior Approaches**: 기존 멀티모달 문서 검색 벤치마크는 질의-단일 페이지 간의 단순 어휘/의미 매칭을 주로 평가해, 여러 페이지에 걸친 문서 맥락을 이용한 간접적 추론 능력을 검증하지 못했다. 또한 대부분의 방법이 페이지를 독립 인코딩해 문서 전역 구조나 cross-page dependencies의 이점을 과소평가하는 한계를 보였다. 멀티홉 계열도 대체로 명시적 매칭에 기반한 직접 검색 성격이 강해, “질의에 직접 드러나지 않는 관련 페이지를 찾아야 하는” 문제를 충분히 다루지 못했다.

- **Core Contribution**: 논문은 Contextual Multimodal Document Retrieval(CMDR) 태스크를 제안하고, 이를 평가하는 CMDR-Bench를 공개한다. CMDR-Bench는 6개 도메인 255개 장문 문서(평균 183.5페이지)에서 수작업으로 큐레이션된 800개 질의로, 페이지 간 맥락 모델링이 필수인 간접 검색을 요구한다. 모델 측면에서는 문서 여러 페이지를 함께 인코딩하되 페이지 수준 임베딩으로 분리해 문맥을 반영하는 CMDR-Embed와, 이를 학습하는 Contextual Multimodal Contrastive Learning(CMCL)을 함께 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 “여러 페이지를 함께 인코딩해 맥락을 얻는 것”이 동시에 동일 문서 내 페이지 표현을 섞어 페이지별 구별력(discriminability)을 떨어뜨린다는 점이다. 이를 해결하기 위해 CMDR-Embed는 chunk-then-split 전략으로 컨텍스트를 공유하되 페이지 임베딩은 분리해 Late Interaction(LI)로 질의-페이지 매칭을 수행한다. 학습에서는 CMCL이 두 종류의 context-aware hard negatives(같은 chunk 내 In-Chunk Negatives, 같은 문서 내 먼 chunk의 In-Document Negatives)를 조합해 맥락 활용과 페이지 구별력을 균형 있게 최적화한다.

- **Empirical Impact**: 실험 결과 CMDR-Embed는 비문맥(non-contextual) 임베딩 대비 유의미하게 성능이 개선되며, 동일 학습 데이터 조건에서도 컨텍스트 모델이 우위를 보였다. CMDR-Bench의 카테고리 분석에서는 특히 CR/MR처럼 참조 해석과 다중 페이지 집계가 필요한 영역에서 기존 멀티모달 검색기가 더 큰 어려움을 드러냈고, 이 격차를 CMDR-Embed가 완화한다. 또한 CMCL의 hard negative 설계와 LI의 멀티벡터 구조가 성능 향상에 일관되게 기여함을 보여, 장문 문서 검색에서 context-aware multimodal embeddings의 필요성을 실증적으로 강조한다.



### Quantifying and Expanding the Theoretical Capacity of Late-Interaction Retrieval Models (https://arxiv.org/abs/2607.05803)
Comments:
          21 Pages, 1 Figure

- **Prior Approaches**: 신경 정보검색은 크게 단일 벡터를 쓰는 dense/sparse retriever와, ColBERT처럼 다중 임베딩 집합에서 MaxSim(Chamfer similarity)로 점수를 계산하는 late-interaction 모델로 나뉜다. 기존 실증 연구는 late-interaction이 out-of-domain에서 유리하다고 보여줬지만, 그 차이가 단순히 더 큰 표현 공간 때문인지, 아니면 MaxSim 같은 유사도 함수 자체의 표현력 때문인지 이론적으로는 명확하지 않았다.

- **Core Contribution**: 이 논문은 MaxSim이 비음수 k-sparse 벡터(차원이 무한일 수도 있음) 사이의 내적을 정확히 복제할 수 있음을 구성으로 증명한다. 또한 같은 표현 공간만으로는 표준 벡터 내적(inner product)이 재현하지 못하는 유사도까지 MaxSim은 표현할 수 있음을 보이며, 부호가 있는 임의 실수 내적은 Signed MaxSim 확장으로 정확 복제가 가능하지만 표준 MaxSim은 불가능함을 논리적으로 구분한다.

- **Technical Challenges**: 핵심 기술적 난제는 MaxSim의 max 연산이 내적과 동치인 표현(특히 sparsity/비음수 제약, 그리고 signed case)을 어떻게 “정확히” 만들 수 있는지에 대한 표현력 분석이다. 이를 위해 논문은 희소 벡터의 각 비영(非零) 좌표를 3차원 임베딩으로 바꾸고, 특정 좌표에서만 양수가 되도록 하는 이차 다항식 기반 구성으로 max가 올바른 항만 선택하게 만든다; 이어서 signed 내적은 표준 MaxSim이 갖는 고정된 공유 임베딩 제약 때문에 불가능하다는 분리 결과를 제시하고, Signed MaxSim으로 그 한계를 우회한다.

- **Empirical Impact**: 이론적 주장에 맞춰, 부정(negation) 쿼리가 포함된 검색에서 Signed MaxSim이 표준 ColBERT/MaxSim 대비 out-of-domain 성능을 크게 끌어올림을 보였다. 구체적으로 nDCG@10이 vocabulary shift에서 0.597→1.000, negation-only 쿼리에서 0.008→0.788로 상승했으며, 이는 late-interaction의 우위가 단순 표현 공간이 아니라 더 강한 유사도 함수 표현력에서 비롯될 수 있음을 뒷받침한다.



### SCOReD: Student-Aware CoT Optimization for Recommendation Distillation (https://arxiv.org/abs/2607.05734)
Comments:
          31 pages

- **Prior Approaches**: 추천을 생성형 추론으로 바꾸려는 흐름이 늘면서, RL 훈련의 선행 단계로 chain-of-thought(CoT) distillation이 중요해졌다. 하지만 추천 도메인의 교사 LLM은 답을 바꾸기보다 같은 결정을 반복 검증하는 ‘불확실성 재확인’ 패턴이 강해, 원문 CoT를 그대로 supervised fine-tuning 하면 학생이 불필요하게 장황하고 수정이 없는 추론을 그대로 모사하는 문제가 생긴다.

- **Core Contribution**: SCOReD(Student-Aware CoT Optimization for Recommendation Distillation)는 추천에 맞춘 CoT 최적화 프레임워크로, 교사 CoT를 추천 단계별로 분해한 뒤 학생 모델의 attention과 확률 신호로 각 구간의 중요도를 추정한다. 그 다음 학생 관점에서 KEEP/REWRITE/FUSE/PRUNE 편집 중 보상이 가장 큰 선택을 동적으로 적용해, 중복 검증은 줄이고 정보 밀도가 높은 구간은 보존하면서 학생 출력 분포에 맞춘다.

- **Technical Challenges**: 핵심 난제는 (1) 추천 CoT가 정답 고정이 아니라 잡음 섞인 로그 라벨과 모호한 선호의 협상 결과라서 ‘압축하면 성능이 떨어지는’ 오프데이터(out-of-distribution) 위험이 크고, (2) 학생의 생성 분포에서 무엇을 유지/수정해야 이득이 나는지 측정하기 어렵다는 점이다. SCOReD는 </think>에서 각 세그먼트로 흘러가는 attention을 기여도 대리 신호로 쓰고, 학생의 answer log probability 향상과 편집 후 segment length/perplexity를 함께 고려한 보상함수로 구간별 편집을 선택해 이 문제를 완화한다.

- **Empirical Impact**: Amazon Beauty 데이터로 0.6B 학생 모델을 실험한 결과, 원시(비압축) teacher trace로 학습한 SFT 대비 NDCG 1.56%, Recall@5 1.9% 향상과 함께 추론 길이 27.3% 감소를 동시에 달성했다. 또한 파싱 실패율도 46% 내외로 줄여, 학생이 불필요한 반복 검증을 덜 생성하면서도 순위 품질을 유지·개선한다는 점을 실증했다.



### Retrieving a Set, Not Independent Passages: Set-Level Compatibility Learning for Efficient Set Exploration (https://arxiv.org/abs/2607.05712)
- **Prior Approaches**: 기존 멀티홉 QA/검색은 질의에 대해 개별 패시지를 독립적으로 점수화하거나, prefix에 조건을 건 sequential next-passage 의사결정을 반복하는 방식이 많다. 이 접근은 증거의 유용성이 패시지 간 호환성에 의해 좌우되는 경우(합치면 쓸모없거나 오히려 오답을 유도) 실패할 수 있다. 또한 LLM 기반 set selection이나 MIP 같은 고비용 추론 파이프라인은 상호작용을 모델링하더라도 실용성이 떨어진다.

- **Core Contribution**: 이 논문은 멀티홉 검색을 ‘질의-증거 집합(query–set) 호환성 점수화’ 문제로 재정의한다. 각 패시지를 따로 평가하는 대신, 전체 evidence set이 질의와 함께 얼마나 맞물리는지를 직접 학습해 불완전·잡음·비호환 집합을 낮게, 완전하고 호환되는 집합을 높게 순위화한다. 이를 위해 set-level 호환성 학습 목적과, 실제로 조합 공간을 다룰 수 있는 효율적 set-level retrieve-and-rerank 프레임워크를 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) set 간 상호작용을 반영하는 학습 신호를 설계하고, (2) 2^n 규모의 조합 후보를 전수 점수화하지 않으면서 효율적으로 탐색하는 것이다. 저자들은 gold set을 addition/elimination/interchange로 구조적으로 교란하고 in-batch negatives를 넣어, 완성도·정합성을 기준으로 margin 기반 set-level 랭킹 손실을 학습시켜 모델이 variable-length/부분 잡음 상황에서도 견고하게 점수화하도록 만든다. 탐색은 beam search로 조합을 근사하고, 경량 set 스코어 ParaSet(사전 임베딩 기반 late interaction)로 후보 set을 탐색한 뒤 SetCE(교차인코더 기반 set reranker)로 정밀 재정렬하는 two-stage 파이프라인으로 계산 비용을 낮춘다.

- **Empirical Impact**: HotpotQA, 2WikiMultihopQA, MuSiQue 등 다양한 벤치마크에서 set-level 호환성 학습이 문서(패시지) 레벨 검색보다 retrieval 성능과 downstream QA 성능을 함께 개선함을 보였다. 특히 ListCE처럼 로컬 next-passage 감독을 쓰는 순차 방식 대비, SetCE는 동일한 아키텍처에서 set-level 감독으로 더 긴/잡음 prefix에서도 강건함이 확인됐다. 또한 ParaSet+SetCE는 cross-encoder set 탐색을 전 범위에 반복하지 않아 지연이 크게 줄면서도, CE5(문서 레벨)에서 놓친 예제를 더 많이 회수해 문서 레벨과의 보완성이 실증되었고, LLM 기반 set selector 및 multi-step agentic retrieval 대비도 속도-성능 트레이드오프에서 우수한 결과를 보였다.



### Prompting Beats Fine-Tuning: Generative Expected Value Scoring for Statutory Term Retrieva (https://arxiv.org/abs/2607.05582)
Comments:
          Accepted to the ASAIL Workshop at ICAIL 2026

- **Prior Approaches**: 기존 연구는 BM25 같은 전통적 IR과 주제모델링 등을 써서 유용한 판례 문장 후보를 찾는 데 집중했지만, 문장 단위 유용성(해석에 도움 되는 정도)을 정밀하게 평가하진 못했다. 이후 학습된 learning-to-rank, 그리고 BERT 계열이 문장 의미와 목표 법조항 용어 간 관계를 미세하게 학습해 성능을 끌어올렸으나, 여전히 입력 구성이나 문맥 활용 방식의 이득이 일관되지 않았다. 또한 일부 최신 접근들은 fine-tuning이나 prompting으로 개선을 보고했지만, zero-shot 환경에서의 전반적 우위는 충분히 검증되지 않았다.

- **Core Contribution**: 이 논문은 특정 법조항 용어를 설명하는 데 “얼마나 도움이 되는지”를 기준으로 판례 문장들을 랭킹하는 과제를 체계적으로 다룬다. 26,959개 문장(42개 U.S. Code 개념)으로 구축된 데이터셋을 기반으로, encoder-only 모델의 supervised fine-tuning(ModernBERT)과 decoder-only 모델의 zero-shot prompting을 같은 랭킹 평가(NDCG@k) 프레임에서 비교한다. 그 결과, 전반적인 성능은 확률 기반(probabilities prompting)으로 문장 관련도를 점수화하는 decoder-only prompting이 최고 기록을 달성한다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 법률 용어의 해석 가치가 ‘명시/사례/반례/비해당성’처럼 미묘한 문장 패턴에 숨어 있고, (2) “관련 없어 보이는 잡음 문장”이 다수를 차지하는 조건에서 랭킹 신호를 안정적으로 뽑는 것이다. 저자들은 ModernBERT에 대해 term과 문장, 또는 법조항 전체/문단 맥락까지 넣는 여러 입력 설계를 실험했지만, 오히려 문맥 확장이 신호-잡음비를 떨어뜨려 성능을 깎는 현상을 확인했다. decoder-only prompting은 분류 라벨을 직접 생성하는 대신 네 범주의 확률분포를 출력하고 expected value로 점수화해, 형식 오류에 대한 JSON 복구(재시도+정규식 전처리)까지 더함으로써 실전용 안정성을 확보했다.

- **Empirical Impact**: 실험 결과 ModernBERT는 적절한 입력 조건에서는 기존 BERT 계열 baseline과 대체로 비슷한 수준을 재현했지만, 문단/장문 맥락을 늘리는 변형에서는 특히 large sparse 쿼리에서 급격한 성능 저하가 나타났다. 반면 GPT-5.4의 확률 기반 prompting은 모든 개념과 NDCG cutoffs 전반에서 가장 강했으며, 보고된 기존 state-of-the-art를 능가했다. 또한 “더 많은 문맥을 넣는 것”보다 “작업 관련 단서의 증폭과 잡음 억제”가 중요하다는 관찰을 제공해, 이후 2단계(빠른 후보 생성+LLM 재정렬) 파이프라인 설계 방향에도 실질적 근거가 된다.



### Scientific Code Search at Scale: A Multi-Domain Dataset and Benchmark (https://arxiv.org/abs/2607.05443)
Comments:
          Datasets and benchmarks publicly released on HuggingFace. Code released on GitHub

- **Prior Approaches**: 기존 코드 검색 벤치마크(CodeSearchNet, CoSQA, CodeXGLUE 등)는 일반 소프트웨어 엔지니어링 질의와 코드 페어에 초점을 둬 과학 컴퓨팅의 도메인 특화 어휘(데이터 포맷, 미션/장비명, 알고리즘 명칭)를 충분히 반영하지 못했다. 또한 과학 소프트웨어는 README 중심으로 문서화가 이뤄지는데, 기존 데이터셋은 과학 연구자가 실제로 겪는 검색 니즈와 정보 제공 방식을 제대로 재현하지 못해 평가의 현실성이 떨어진다는 한계가 있었다.

- **Core Contribution**: 이 논문은 NASA Science Mission Directorate 5개 분과(지구과학, 천문·천체물리, 행성·태양계, 헬리오피직스, 생물·물리과학)에 걸친 5,264개 과학 GitHub 저장소를 도메인 분류하고, 정제된 README와 외부 링크 기반 맥락을 풍부화한 큐레이션 코퍼스를 구축했다. 이를 바탕으로 (1) 분과 전문가가 만든 219개 repository search 질의와 (2) 7개 언어의 117,950개 코드 스니펫 및 119,720개 질의를 포함한 대규모 code snippet retrieval 벤치마크를 제안하며, HuggingFace에 공개했다.

- **Technical Challenges**: 핵심 기술 난제는 과학 분야의 비표준화된 문서/용어 때문에 검색 신호가 README나 docstring에 희소하게 흩어져 있다는 점이었다. 저자들은 LLM(GPT-4.1-mini) 기반으로 저장소 도메인 분류와 README 정제(불필요한 boilerplate 제거), 외부 링크 크롤링 후 LLM 점수로 맥락을 추가하는 2단계 풍부화 파이프라인을 적용해 문서 신호를 보강했고, 이 표현을 사용해 BM25와 dense embedding(일반/도메인 특화), 그리고 Hybrid-RRF/Hybrid-Rerank 같은 최신 IR 패러다임을 공정 비교할 수 있게 했다.

- **Empirical Impact**: 실험 결과 repository search는 도메인마다 성능 격차가 크게 나타났고(MRR@10 약 .18~.87), 특히 문서 품질과 용어 표준화(FITS, WCS 등)의 영향을 강하게 받았다. 코드 스니펫 retrieval은 docstring 기반 질의는 잘 맞추지만 identifier 기반 질의는 급격히 어려워지는 현상(MRR@10 약 .76 vs .25)을 보였고, 일반 텍스트 임베딩(예: Qwen3-Embedding-0.6B)이 도메인 과학 텍스트 임베딩보다 더 강하게 코드 수준 검색에 전이됨을 확인했다; 이는 과학 코드 탐색에서 문서 문화(README vs docstring), 컨텍스트 확장, 그리고 코드-aware 검색 표현이 모두 중요하다는 실증적 근거를 제공한다.



### PORTS: Preference-Optimized Retrievers for Tool Selection with Large Language Models (https://arxiv.org/abs/2607.05441)
Comments:
          Please cite the definitive, peer-reviewed version of this article published in the Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, edited by Christos Christodoulopoulos et al., Association for Computational Linguistics, pp. 10007-10030, 2025. DOI: this https URL

- **Prior Approaches**: 기존 연구는 방대한 tool collection에서 필요한 도구를 고르기 위해 retrieval 기반 사전선택을 활용했다. 하지만 retriever는 LLM의 tool-calling과 별도 학습되는 경우가 많아, 실제 tool 호출 성능과 잘 정렬되지 않는 misalignment 문제가 있었다.

- **Core Contribution**: 이 논문은 tool 선택 목적에 맞춘 retriever 학습 방법 PORTS를 제안한다. PORTS는 frozen LLM에서 얻은 perplexity-inspired preference 신호로 선택 확률과 downstream 성능 간 상관을 최적화하고, 문서 문자열 간 contrastive semantic loss도 함께 걸어 tool 선택 정합성을 높인다.

- **Technical Challenges**: 핵심 기술 난제는 retrieval 신호가 LLM의 tool-calling 동작 및 성능과 정렬되도록 preference를 설계·학습하는 것이다. PORTS는 selection probability–성능 상관 최적화와 문서 문자열의 대비 학습을 동시에 수행해, 독립 학습에서 비롯된 불일치를 줄이도록 설계했다.

- **Empirical Impact**: PORTS는 6개 데이터셋, 2개 encoder 모델, 3개 LLM(사전지식 다양)에서 tool selection accuracy를 유의미하게 개선하며 성능을 폭넓게 입증했다. 또한 계산 부담이 낮은 alignment로 새로운 쿼리와 새로운 tool로의 generalization도 잘 되어, 변화하는 toolset을 전제로 한 실무 적용 가능성이 강조된다.



### Modality Relevance is not Modality Utility: Post-hoc Selective Modality Escalation for Cost-Aware Multimodal RAG (https://arxiv.org/abs/2607.05438)
- **Prior Approaches**: 기존 멀티모달 RAG는 비용 비대칭 때문에 “텍스트+테이블만” 쓰거나 “모든 이미지에 VLM을 적용”하는 식으로 의사결정을 고정하는 경우가 많았다. 적응형 접근도 대개 질문만 보고(pre-retrieval) 어느 모달리티에 예산을 쓸지 라우팅하지만, 답을 만들기 전이라 실제 유용성보다 겉보기 관련성에 치우칠 수 있다. 또한 에이전트형 파이프라인은 단계별로 더 많이 호출하지만, 호출량이 늘어 비용 통제가 어려운 편이다.

- **Core Contribution**: 이 논문은 모달리티 결정 시점을 “답을 먼저 초저비용으로 초안 생성(텍스트+테이블)”한 뒤로 옮겨, 필요한 모달리티만 사후적으로 escaltion(확장)하자는 post-hoc selective modality escalation을 제안한다. 핵심은 (query, draft answer, evidence) 튜플을 verifier가 보고, 이미지 모달리티가 “부족해서” 틀렸는지 모달리티 갭을 특정한 뒤에만 VLM 비용을 지불한다. 마지막으로 value-of-escalation을 캘리브레이션해, 예상 정확도 이득이 시각 비용을 정당화할 때만 호출하도록 운영 지점을 고정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “모달리티 관련성(relevance)”과 “정답에 필요한 실제 유용성(utility)”이 다르다는 점을 라우팅 신호로 분해해 내는 것이다. 이들은 MultiModalQA의 oracle headroom 분석을 통해 겉보기 관련성이 약한 예측자임을 보였고, verifier의 need_image 같은 투표를 그대로 쓰기보다 draft 이후 특징을 이용해 keep vs escalate의 정확도 차이를 예측하는 캘리브레이션 모델로 해결했다. 이후 임계값으로 accuracy–escalation(비용) 프런티어를 타겟하도록 설계해, 무작정 호출을 줄이면서도 이득이 있을 때만 시각 확장을 수행한다.

- **Empirical Impact**: MultiModalQA에서 항상-on VLM 파이프라인과 동일 예산(난할당) 비교를 엄격히 수행했을 때, post-hoc 라우팅은 정확도를 유지하면서도 훨씬 적은 visual call 비율로 그 성능을 회복한다. 구체적으로 learned pre-retrieval 라우터를 예산이 커질수록 더 크게 추월하며, oracle escalating(필요한 경우에만 escaltion)과도 대부분의 격차를 메웠다. WebQA 교차 검증에서는 모달리티가 본질적으로 분리된 경우엔 이득이 줄고 비용 제어 효과로 수렴해, 제안 메커니즘(모달리티 relevance–utility gap)에 상응하는 현상임을 뒷받침한다.



### DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation (https://arxiv.org/abs/2607.06507)
- **Prior Approaches**: 기존 Multi-hop RAG는 반복 검색, query reformulation, evidence critique, sufficiency judging 같은 유용한 조작을 제공하지만, 보통은 각 방법별 파이프라인이나 사전 정의된 제어 토폴로지 안에 갇혀 있습니다. 그 결과 “현재 증거 상태에서 무엇을(검색/진단/가피-엔티티 확장/갭 질의/중단) 선택해야 하는가” 같은 공통 제어 문제를 한 프레임에서 학습·비교하기가 어렵습니다. 또한 더 많은 retrieval이 항상 이득으로 이어지지 않는 비용-효율성 문제가 남아 있습니다.

- **Core Contribution**: DynaKRAG은 multi-hop 증거 획득을 “진화하는 evidence state”에 조건화된 제어 문제로 재정의하고, 서로 다른 RAG 동작을 atomic evidence operations로 원자화한 통합 학습 프레임워크를 제안합니다. 매 단계에서 hard validity layer가 현재 상태에서 실행 가능한 action set을 구성하고, 학습된 controller가 그중 다음 조작을 선택해 evidence state를 갱신합니다. 이를 통해 retrieval, diagnosis, gap-directed acquisition, stop-and-answer, 그리고(종단 단계의) answer-focused compression까지 하나의 순차 의사결정으로 조율합니다.

- **Technical Challenges**: 핵심 난제는 실행 가능(정의됨/캡 미소진/전제 조건 충족)한 연산만 골라야 하면서도, 각 연산이 현재 상태에서 증거를 얼마나 개선하는지(효용)를 함께 학습하는 것입니다. DynaKRAG은 action utility 학습을 validity와 분리해, undefined·redundant·premature action은 validity layer가 먼저 제거하고 value model은 남은 유효 action만 랭킹하도록 설계했습니다. 학습 신호로는 supporting-evidence coverage 변화를 기반으로 acquisition 관련 보상을 만들고, sufficiency_check에 대해서는 증거 준비도 학습을, stopping에 대해서는 현재 support coverage를 사용합니다.

- **Empirical Impact**: Qwen2.5-7B-Instruct 백본에서 HotpotQA F1 0.5998, 2Wiki F1 0.5340, MuSiQue F1 0.3061로 3개 벤치마크 모두에서 최강 controlled baseline 대비 성능 우위를 보였고, Qwen 정책은 다른 답변 모델(GPT-4o-mini, Llama-3.1-8B)로도 전이되어 과적합 우려를 줄였습니다. 추가 실험에서 uniform-valid 정책으로 controller를 대체하면 F1이 3.96~5.78포인트 하락했으며, sufficiency feedback 제거는 세 데이터셋 모두 성능을 악화시켰습니다. 또한 controlled retrieval-cap 실험 결과는 “추가 retrieval이 항상 이득”이 아니라는 점을 보여주며, 증거 상태가 바뀌는 흐름 속에서 retrieval·진단·갭 지향 획득을 함께 조율하는 접근의 실용성을 뒷받침합니다.



### InfluMatch: Frontier-Quality KOL Search at 4B-Model Cos (https://arxiv.org/abs/2607.05968)
- **Prior Approaches**: 기존 KOL(핵심 의견 리더) 매칭은 키워드 기반 검색이나 정형 속성 필터로 처리되는 경우가 많았지만, 의미 적합도는 놓치고(어휘는 다르지만 내용은 맞는 경우) 캠페인별 다중 조건을 정적 스키마로는 반영하기 어렵습니다. 또한 모든 후보에 대해 frontier LLM을 즉시 추론하는 방식은 정확도는 높아도 지연과 비용이 커 운영에 부담이 됩니다.

- **Core Contribution**: InfluMatch는 태국어의 자유형·다중 파트 마케팅 기준을 받아 KOL을 단계별로 좁힌 뒤 재평가하고, 각 기준별 점수와 태국어 근거를 함께 출력하는 배치 가능한 3단계 캐스케이드( retrieval → rerank → reason )를 제안합니다. 특히 소형 오픈 웨이트 모델만으로도 end-to-end 순위 품질을 확보하면서, frontier 수준의 성능을 저비용으로 노리는 설계를 전면에 둡니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 후보 수가 큰 상태에서 “설명 가능한 판단(reason)”을 비싸게 돌리지 않으면서도, (2) 학습/추론에서 점수 체계가 실제 순위 최적화와 잘 맞도록 하는 것입니다. InfluMatch는 dense retrieval top-50을 만든 뒤 4B pointwise reranker로 top-10만 추려 단일 Yes 토큰 log-prob로 순위를 정하고, 4B reasoner는 기준별 루브릭 채점+태국어 rationale을 생성하도록 하되, fine-tuning은 pairwise SimPO가 end-to-end 전이에 유리하고 reasoner는 untuned base가 가장 강하다는 점을 실험적으로 확인해 배치 설계를 정리했습니다.

- **Empirical Impact**: 실험에서 retrieval-only는 거의 랜덤 수준에 머물렀지만, rerank→reason 전체 캐스케이드는 11개 쿼리 세트에서 P@5 94.1%를 달성하며 frontier 모델 Kimi-K2.6(91.8%)과 비슷한 수준을 저비용으로 따라갑니다. 또한 출력 토큰을 약 35배 줄이고 단일 A100에서 50개 KOL 쿼리를 약 20초 내 처리하는 등 운영 효율이 뚜렷하며, 특히 reasoner의 offline 성능 향상이 end-to-end에서는 역효과가 될 수 있음을 사례로 보여 실전 학습/라벨링 전략에 시사점을 줍니다.



### Inject or Navigate? Token-Efficient Retrieval for LLM Analysis of Transactional Legal Documents (https://arxiv.org/abs/2607.05764)
Comments:
          17 pages, 2 figures, 8 tables

- **Prior Approaches**: 기존 법률 문서 Q&A는 매 질의마다 전체 문서 코퍼스를 LLM 컨텍스트에 주입하는 방식(inject)이 가장 단순하고, 재검색 누락을 피한다는 장점이 있다. 하지만 코퍼스가 커질수록 토큰 부담과 long-context 성능 저하가 함께 커져 비용/품질이 비선형으로 악화된다. 또한 벡터 기반 RAG는 cosine 유사도로 의미만 맞추기 쉬워, 정의/교차인용/개정선후관계처럼 구조 의존성이 강한 계약 문서의 검색에는 한계가 반복적으로 지적돼 왔다.

- **Core Contribution**: 이 논문은 법률 분석 시스템에서 full-corpus injection을 구조 기반 retrieval로 대체하기 위해, Syntheia의 structure-aware chunking을 고정한 채 retrieval 전략만 비교·평가했다. 두 대안은 NAVEMBED(임베딩 검색+reranking)와 NAVINDEX(LLM이 compact structured index를 탐색하도록 한 구조 탐색)이며, 품질 저하 없이 토큰·달러 비용을 줄이는 설계 기준을 제시한다. 특히 NAVINDEX는 정의 term 그래프와 cross-reference 그래프 등 구조 신호를 인덱스 표면에 명시적으로 인코딩해 벡터 검색의 표현 한계를 보완한다.

- **Technical Challenges**: 핵심 난제는 (1) 관련 조항이 코퍼스 전역에 분산돼 있어 중간 위치 정보가 희석되는 long-context 문제와 (2) 프롬프트 caching이 비용 축을 바꾸더라도 모델이 실제로 attend하는 토큰 축은 그대로 남는 점을 동시에 통제하는 것이다. 저자들은 token footprint(모델이 주목하는 입력 토큰)과 dollar cost(캐시 할인/세션 경제 포함)를 분리해 추적하고, reference-anchored pairwise judge로 위치 편향을 통제한 평가 프로토콜을 구성했다. NAVEMBED는 reranking 파이프라인으로 작업 셋을 줄였고, NAVINDEX는 *.index.json(구조 메타·요약·그래프)과 *.full.json(원문 조항 저장)을 나눠 질의당 최대 10개 노드만 fetch하도록 하드 캡을 적용했다.

- **Empirical Impact**: 20개 질문(문서 18개 bound, 2개 out-of-scope 통제) 벤치마크에서 NAVEMBED는 문서 bound 18개 중 16개에서 inject와 동률 수준의 품질을 보였고, 관련 없는 2개 out-of-scope에서도 둘 다 동률로 판정됐다. 입력 토큰은 inject 대비 17.3x(최적 GTE 구성은 29.9x) 적었으며, 비용 비교에서도 탐색 기반 모드들이 유리한 구간이 확인됐다. NAVINDEX는 18개 모두 동률로 평가되면서도 total token footprint은 1.61x 줄고 answering context는 약 56x 줄였으며 달러 비용도 약 25% 낮았고, cached injection이 유리해지는 조건을 코퍼스 크기 기준의 closed-form caching-crossover rule로 정리해 운영 의사결정에 직접 연결된다.



### Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction (https://arxiv.org/abs/2607.05577)
Comments:
          23 pages, 4 figures; 9-page main text plus appendix. Preprint

- **Prior Approaches**: 기존 RAG와 agent-memory, 그리고 GraphRAG·Graphiti 같은 그래프 기반 접근은 근거가 되는 텍스트/에피소드를 찾아 “정답에 가까운 증거”를 제공하는 데 초점을 둔다. 하지만 소설의 다중 홉 narratological 질의(누가 언제 알았는지, 사건-발화 순서 차이, 떳다-갚기, 관계 변화 등)에 필요한 ‘서사 구조가 반영된 상태’가 표현·추적되지 않아 엉뚱한 증거가 나오거나 아예 근거가 부재한 문제가 반복된다. 특히 일반 엔티티/이벤트 그래프는 시점별 “관찰자 시야”, “드러냄(order of reveal) vs 사건(order of event)”, “약속의 성립/해소” 같은 제작자가 필요한 타이핑된 시간-서사 상태를 1차 항목으로 다루기 어렵다.

- **Core Contribution**: 이 논문은 Narrative World Model(NWM)이라는 작가용 메모리 시스템을 제안하며, 서사 이론에 기반한 타이핑된 temporal-state graph와 질의 조건형 hybrid retrieval을 결합한다. 메모리는 단순 요약이나 덩어리 텍스트가 아니라, 확정된 장(chapter)의 evidential span을 붙여 “누가 무엇을 언제 알았는지/관계가 어떻게 바뀌었는지/약속이 어떻게 기능했는지” 같은 서사 상태를 저장하도록 설계됐다. 또한 답변 성능을 조작하는 요소를 줄이기 위해, 동일한 Opus 4.8 리더가 각 시스템의 ‘장 안전(chapter-safe)’ 근거만 읽고 판단하도록 평가 프로토콜을 고정한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 미래 장을 읽지 않으면서도 (2) 여러 장에 걸친 질의에 필요한 “현재 시점의 서사 상태”를 재구성하고 (3) 그 질의에 맞는 증거만 압축해 제시하는 것이다. NWM은 장 게시(publish) 흐름에서 Sonnet 4.5 추출기로 타이핑된 서사 메모리 레코드를 만들고, temporal KG의 validity interval로 as-of 시점 상태를 계산하며, BM25+벡터 기반 검색 뒤에 1-hop 타이핑 이웃을 확장해 bounded 패킷으로 증거를 전달한다. 더 나아가 RLM QA 검증 레이어로 질의 분해-근거 수집-상태 변화 지지 여부를 재확인해, 서사 상태의 시점 오해를 줄이도록 한다.

- **Empirical Impact**: 실험은 공개/비공개 두 코퍼스와 검증된 multi-hop 벤치마크(사유된 narratological 176문항 등)에서 진행됐고, NWM Graph Retrieval이 Graphiti 대비 큰 폭으로 향상됐다. 비공개 multi-hop 슬라이스에서 정확도는 0.898 대 0.574였고(p<1e-5), 공개 576문항에서도 0.625 대 0.516으로 유의미하게 우세했다. 특히 Graphiti를 같은 추출기로 맞추거나 더 저렴한 추출기로 재임포트해도 격차가 유지되어, 성능 향상이 추출 품질이나 그래프 크기 같은 부가 요인이 아니라 “서사 구조를 타이핑해 표현한 표현력”과 “질의 조건형 검색”에서 온다는 점을 실험적으로 확인했다.



### Linking Hadith Narrator Identities Across Heterogeneous Arabic Biographical Databases: A Multi-Signal Entity Resolution Pipelin (https://arxiv.org/abs/2607.05424)
Comments:
          16 pages, the data sets available at DOI: https://doi.org/10.1016/j.dib.2022.108065

- **Prior Approaches**: 기존 연구는 하디스 전승 사슬(sanad)에서 그래프 분석을 하더라도 단일 데이터베이스에 갇혀 생물(인물) 노드의 신뢰도·사망연도 같은 메타데이터를 확장하기 어려웠습니다. 또한 기존 narrator disambiguation은 AraBERT 기반 등 ‘닫힌 세계(closed-world)’ 분류에 초점을 두어, 서로 다른 전기(傳記) DB 간 인물 이름을 외부로 연결하는 ‘열린 세계(open-world)’ 문제에는 부족했습니다.

- **Core Contribution**: 이 논문은 Sanadset 650K(650,986 하디스 레코드)의 narrator 이름을 두 전기 DB(Hadithtransmitters/hawramani, Muslimscholars)에 단계적으로 연결하는 2-phase entity resolution 파이프라인을 제안합니다. Phase 1은 이름 유사도만으로 Sanadset→hawramani를 매칭하고, Phase 2는 이름 유사도·사망연도 근접성·신뢰도(grades) 극성 등 다중 신호로 hawramani↔muslimscholars를 교차검증해 전파 가능한 링크를 만듭니다.

- **Technical Challenges**: 핵심 난제는 역사적 아랍 인명에서 나타나는 정규화 편차(diacritic, Alef variants, ta marbuta 등)와 kunyah/ism 변형, 그리고 nasab 체인의 토큰 길이 차이로 인한 이름 유사도 왜곡입니다. 저자들은 도메인 특화 Arabic normalization과 빅램(prefix) 인덱스로 후보군을 줄인 뒤, Phase 1은 TSR 기반 fuzzy 매칭(이름만), Phase 2는 가중 다중 신호 스코어(사망연도 희소성 고려한 동적 가중)를 사용해 오탐을 억제했습니다.

- **Empirical Impact**: 실험 결과 Phase 1에서 185,216개 narrator name variant 중 94,628개(51.1%)를 hawramani에 연결했고, Phase 2에서는 hawramani 기준 94.7%에 해당하는 95,573개 링크를 추가로 확보했습니다. 최종적으로 185,216노드·814,093엣지 규모의 방향성 전승 그래프에 교차 출처 전기 메타데이터를 풍부화했으며, sanad_links·narrator_links 및 그래프를 오픈 리소스로 공개해 디지털 이슬람 인문·하디스 인증 분석의 기반을 넓힐 것으로 기대됩니다.



New uploads on arXiv(cs.CV)

### ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation (https://arxiv.org/abs/2607.06565)
- **Prior Approaches**: 통합 3D 파운데이션 모델은 한 백본에서 이미지-3D 생성, text-to-3D 생성, 3D 캡셔닝까지 수행하려 하지만, 텍스트-3D 상호작용이 대부분 암묵적으로 처리됐다. 기존 접근은 텍스트 토큰과 3D 토큰을 평평한 시퀀스로 이어 붙인 뒤 self-attention에 의존해 대응을 찾는 방식이라, 구조적 신호와 정밀 기하 디테일이 한 덩어리 표현으로 뭉치기 쉽다. 또 멀티스케일 3D 표현을 쓰더라도, 언어 추론이 어떤 스케일의 기하 근거를 참조해야 하는지까지는 아키텍처가 충분히 구조화하지 못했다.

- **Core Contribution**: ELSA3D는 통합 모델의 텍스트-기하 정렬을 ‘elastic semantic anchoring’으로 명시화해, 언어 추론과 기하 추론을 동일한 추상화 스케일에 맞춰 함께 구조화한다. 3D는 scale tag가 포함된 scale-aware octree tokenizer로 표현하고, 언어는 Global/Structure/Appearance로 분해한 semantic trace로 설계한다. 여기에 Anchor Tokens라는 희소한 크로스모달 인터페이스를 도입해, 선택된 의미 단서가 특정 3D 스케일의 근거를 조회한 뒤 통합 표현에 다시 기록되도록 한다.

- **Technical Challenges**: 문제는 (1) 언어는 종종 기하의 정확한 세부를 생략하는 ‘under-specified’ 조건이고, (2) 모든 텍스트 토큰을 모든 3D 토큰과 촘촘히 결합하면 계산량이 폭증하며 의미 잡음도 커진다는 점이다. ELSA3D는 scale-aware octree에서 스케일 태그와 위치 정보를 갖는 기하 토큰을 만들고, Anchor Tokens로 필요한 의미 토큰만 골라 특정 스케일의 기하 증거를 교차어텐션으로 가져온 뒤 write-back한다. 또한 per-block elastic router가 블록 실행/MLP width/어떤 토큰이 어떤 스케일에 anchor를 둘지까지 함께 결정해, 정렬이 필요한 곳에만 추론과 연산을 집중한다.

- **Empirical Impact**: 실험 결과 ELSA3D는 image-to-3D 생성, text-to-3D 생성, 3D captioning 전반에서 SOTA를 달성했으며, strongest unified baseline 대비 성능이 전 과제에서 일관되게 개선됐다. 특히 ablation에서 anchor routing의 희소성이 dense한 텍스트-3D 융합보다 더 잘 작동함을 보였고, FLOPs는 1081G에서 632G로, 추론 지연은 29.8s에서 17.2s로 줄였다. 즉, 비(非)elastic 버전 대비 FLOPs와 latency를 대략 절반 수준으로 낮추면서도 생성 품질과 언어-기하 이해 정확도를 동시에 끌어올린 점이 의미 있다.



### Vision as Unified Multimodal Generation (https://arxiv.org/abs/2607.06560)
Comments:
          48 pages,22 figures

- **Prior Approaches**: 기존 비전 연구는 detection, segmentation, depth 등 작업군별로 전용 아키텍처와 디코딩 규칙을 붙여왔고, 그 결과 서로 다른 출력 형식 때문에 감독 데이터를 공유·재조합하기 어려웠습니다. 시퀀스 통합(OFA, Pix2Seq 계열, Unified-IO 등)은 공통 인터페이스를 제공하지만 dense map은 직렬화/파싱 규칙에 더 의존했고, 표현(표상) 중심 모델(SAM류, Depth Anything류)은 출력 공간이 과업군 내에서만 잘 맞아 언어 제어가 제한적이었습니다. 생성 기반(확산/이미지 생성, MLLM)은 dense 예측이나 언어 지시는 일부 해결했지만 symbolic record와 mixed text-image 출력까지 하나의 네이티브 프레임으로 통합하긴 미흡했습니다.

- **Core Contribution**: SenseNova-Vision은 컴퓨터 비전을 unified multimodal generation으로 재정의해, 작업별 head 없이도 텍스트와 이미지의 네이티브 생성 공간에 맞춰 다양한 과업을 하나의 모델이 처리하도록 합니다. 자연어 instruction(필요 시 visual prompt 포함)으로 작업/대상/출력 스키마/디코딩 관례를 지정하고, 결과는 text(예: OCR 문자열·카메라 파라미터), image(예: depth·mask), 또는 mixed(예: 범주·컬러 전설과 마스크 동시 생성)로 내보내도록 설계했습니다. 이를 위해 다양한 비전 어노테이션을 instruction-response 예제로 변환한 SenseNova-Vision Corpus(SN-VC)와 50M 변환 서브셋(SN-VC-50M)을 구축해 학습 가능한 형태로 만들었습니다.

- **Technical Challenges**: 가장 큰 기술 난제는 heterogeneous 감독 신호(박스/좌표/문자 vs dense 기하/마스크 vs 카메라 포즈)를 UMM의 ‘네이티브’ 출력 공간에 맞게 한 규칙으로 디코딩 가능하도록 정렬하는 것입니다. 논문은 structured 과업은 좌표를 정규화한 텍스트 스키마(<p>, <bbox>, <point> 등)로 통일하고, depth·surface normal·point map은 VAE 기반 image latent를 통해 결정적 시각 인코딩으로 생성한 뒤 벤치마크용 맵으로 역변환하도록 했으며, multi-view 기하와 포즈는 텍스트의 special tokens(예: frame, quat, scale 등)로 구조화해 mixed 출력을 복원했습니다. 또 Bagel 기반 off-the-shelf UMM을 그대로 활용해, computer vision 변환 코퍼스와 범용 멀티모달 데이터를 capability-preserving mixture로 섞어 fine-tuning하면서 text용 CE(next-token)와 visual용 rectified-flow를 병행 학습해 과업 전용 헤드 없이 통합했습니다.

- **Empirical Impact**: 실험은 SenseNova-Vision이 structured visual understanding, dense geometric prediction, segmentation, multi-view visual geometry의 네 과업군에서 단일 통합 모델로 강한 성능을 보이며, 여러 작업에서는 task-specialized 시스템과의 격차를 줄이거나 근접하는 결과를 제시합니다. 특히 긴 꼬리·작은 객체·referencing·OCR localization 같은 정밀한 좌표 수준 과업에서 두드러졌고, depth/normal은 이미지 출력으로 생성해도 geometry 특화 모델 대비 경쟁력을 유지했다고 보고했습니다. 또한 학습 세트에 명시되지 않은 언어 조건 조합(범주·색·영역 등)을 instruction으로 지시하는 변형까지 수행 가능해, 통합 비전 생성이 general-purpose foundation model로의 확장 경로가 될 수 있음을 시사합니다.



### ProxyPose: 6-DoF Pose Tracking via Video-to-Video Translation (https://arxiv.org/abs/2607.06555)
Comments:
          23 pages, 6 figures

- **Prior Approaches**: 기존 6-DoF 포즈 추적은 CAD/3D 표현, depth map, 오브젝트 마스크, 혹은 task-specific feature 같은 추가 입력에 의존하는 경우가 많았다. 또한 불투명하지 않거나(무질감·반사), 변형되는(flexible) 표면에서는 지역 특징 추출이나 추적 단계가 쉽게 흔들린다. 최근엔 foundation model을 활용해 강인성을 높이려 하지만, 여전히 대규모 정제 데이터와 정교한 학습 파이프라인이 필요하다는 한계가 남아 있다.

- **Core Contribution**: ProxyPose는 6-DoF 추적을 ‘비디오-대-비디오 번역’으로 재구성한다. 단일 입력 비디오와 첫 프레임의 표시된 픽셀 1개만 주면, fine-tuning된 video diffusion model이 해당 국소 부위와 같은 로컬 rigid-body 운동을 갖는 알려진 프록시(색상 폴리헤드론) 비디오를 합성한다. 합성된 프록시의 기하/외관은 설계로 고정되므로, 이후에는 classical pose estimation(PnP 및 최적화)만으로 전체 6-DoF 궤적을 복원한다.

- **Technical Challenges**: 핵심 기술적 어려움은 diffusion 모델이 프록시의 ‘정체성(크기·방향·스케일)’을 안정적으로 유지한 채로, 관측 비디오의 운동을 정확히 내재화해 번역해야 한다는 점이다. 이를 위해 프록시 스트림에는 noise를 전체적으로 무작정 섞지 않고, 첫 프레임은 anchor timestep 대신 offset된 스케줄로 더 적게 손상시키는 방식으로 생성 안정성을 높였다. 또 proxy는 고정된 색면을 가진 큐브로 구성해 face 코너를 검출-2D/3D 대응을 만든 뒤 PnP와 reprojection/temporal smoothness를 이용한 Levenberg–Marquardt로 궤적을 정교화한다.

- **Empirical Impact**: ProxyPose는 추가 입력(3D 모델, depth, 마스크, task-specific 학습 특징) 없이도 state-of-the-art 수준의 6-DoF 포즈 추적 정확도와 시간적 일관성을 보였다고 보고한다. fine-tuning은 합성 데이터에서만 수행하며, 실제 in-the-wild 장면에서도 기존 방식의 제약을 넘어서는 확장성을 보인다. 더 나아가 per-pixel 6-DoF 표현을 집계해 카메라 포즈 추정 및 face tracking 같은 관련 태스크로도 적용 가능함을 실험적으로 제시한다.



### From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models (https://arxiv.org/abs/2607.06553)
- **Prior Approaches**: 기존 text-to-image 기반 dense prediction은 VAE latent에 depth, normals, 마스크, 열지도 같은 정답을 “RGB처럼” 인코딩하고 다시 VAE 디코더로 픽셀 예측을 복원하는 target-side 렌더링 인터페이스를 그대로 가져오는 흐름이 강했습니다. 그 방식은 RGB 합성에는 자연스럽지만, dense prediction은 새 RGB 이미지를 생성하는 게 아니라 같은 이미지 평면에서 task-native 필드를 정확히 뽑아야 해서 비효율이 생긴다는 문제의식이 제기됩니다.

- **Core Contribution**: 논문은 dense prediction의 핵심이 “생성”이 아니라 생성기가 이미 정리해둔 공간 토큰 격자를 task-native 채널로 “재채널링(ReChannel)”하는 것이라고 주장합니다. 이를 위해 ReChannel은 DiT의 입력 분포는 유지하되( VAE encoder만 사용), 정답은 VAE로 통과시키지 않고 task LoRA로 토큰 필드를 의미에 맞게 적응시킨 뒤 토큰 로컬 선형 readout으로 픽셀 패치를 직접 읽어냅니다.

- **Technical Challenges**: 가장 큰 기술 과제는 pretrained DiT가 만든 RGB 토큰 채널을 dense task의 연속값/마스크/열지도 같은 더 compact한 표현으로 정렬시키는 방법입니다. 저자들은 frozen backbones 위에 task LoRA만 최소 적용하고, 토큰마다 서로 다른 위치 혼합 없이 token-local linear head로 p×p×Kt 타깃 패치를 뽑아내는 구조를 채택해 “채널 재해석만으로 충분”함을 검증합니다.

- **Empirical Impact**: FLUX-Klein(4B/9B) 고정 조건에서 6개 dense prediction 태스크와 12개 이상 벤치마크를 평가한 결과, trimap-free matting, KITTI depth, referring segmentation에서 새로운 SOTA를 기록했고 normals, saliency, pose에서도 경쟁력을 유지했습니다. 특히 4B 매칭 설정에서 edit-plus-latent-decode 계열 대비 정확도는 더 높으면서 최대 2.48배 빨라, dense perception이 생성기의 target-side 렌더링 인터페이스를 상속할 필요가 없다는 실증적 근거를 제공합니다.



### MonoIR-RS: Infrared Remote Sensing Vision-Language Learning with CLIP and VLM Adaptation (https://arxiv.org/abs/2607.06552)
- **Prior Approaches**: 기존 원격탐지 비전-언어 연구는 RGB 중심 CLIP류 대조학습과 VLM 지시학습을 확장해 왔지만, 적외선(thermal/IR)에서는 색 단서가 부재해 같은 캡션이 오히려 오해를 유발할 수 있다. Infrared-LLaVA, IRGPT 등은 적외선 모달리티 격차를 다루기 시작했지만, 원격탐지 적외선에 맞춘 데이터-텍스트 재구성이 부족하거나 실센서 기반 데이터 확장이 병목이었다. 또한 합성 데이터·혼합 프로토콜에서는 RGB-누출이나 split 위반이 성능을 부풀릴 위험이 있어 공정한 평가 체계가 필요하다는 문제가 남아 있었다.

- **Core Contribution**: MonoIR-RS는 원격탐지 적외선 비전-언어 학습을 위해 600,000장의 합성 IR 이미지를 만들고, 59,032개의 IR-aware 캡션을 필터링해 벤치마크를 구성한다. 핵심은 모델 입력은 IR만 두고(visible은 구성/감사용으로만 보유), 캡션 감독도 RGB 외관이 아니라 적외선식 grayscale 구조·강도 대비·물체-배경 분리에 맞게 재작성한다. 이를 통해 RGB-IR 듀얼모달 추론과 분리된 “통제 가능한” 적외선 증거-언어 정렬 실험대(testbed)를 제공한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) RGB 기반 소스 텍스트의 색/가시광 묘사가 IR에서 무의미해지는 문제와 (2) 합성 IR이 실제 thermal과 얼마나 가깝고, 학습 효과가 단순 편향이 아닌지 공정하게 검증하는 문제다. 저자들은 DiffV2IR로 visible→IR 합성을 수행하면서, 캡션은 Qwen2.5-VL-72B-Instruct로 IR-aware 형태로 재작성하고, train-only 프로토콜과 split·이미지·RGB-누출 점검을 통해 재현 가능한 데이터 경계를 강제한다. 또한 CLIP 5개 백본과 VLM 6개 백본을 각각 IR-aware contrastive adaptation, LoRA 기반 instruction tuning으로 조정한 뒤 zero-shot 대비와 진단 지표(IR-cue rate, RGB-color leakage, overclaim 등)로 모달리티 정렬을 분해해 확인한다.

- **Empirical Impact**: 합성 IR은 AVIID에서 grayscale 변환보다 실제 thermal에 더 가까웠고(FID 및 히스토그램 거리 개선), IR-aware CLIP 적응은 zero-shot 대비 mean recall을 최대 12.8포인트까지 끌어올렸다. VLM 쪽에서는 학습 후 captioning에서 IR-cue coverage가 100%에 도달하면서도 RGB-color leakage는 거의 0에 수렴해, “IR 증거를 말하되 색 단서는 새지 않는” 행동 변화가 관찰됐다. 즉 MonoIR-RS는 적외선 원격탐지에서 언어 감독을 IR 증거에 맞춰 재구성하고, controlled 평가로 모달리티 전이를 정량화할 수 있는 기준점을 제시한다.



### Unsupervised Domain Adaptation for Calcification Classification in Mammography Across Multi-Site Datasets (https://arxiv.org/abs/2607.06549)
- **Prior Approaches**: 기존 유방촬영 CAD는 CNN과 transformer 기반 분류·검출·분할을 폭넓게 다뤘지만, 대개 단일 사이트 또는 소규모 공개 데이터에 의존해 다기관(domain) 일반화가 충분히 검증되지 않았다. 또한 calcification은 soft tissue 병변보다 희소해 레이블 데이터가 부족하고, 합성 2D(3D DBT 유래)까지 체계적으로 포함한 연구도 상대적으로 적었다.

- **Core Contribution**: 이 논문은 악성 vs 양성 calcification 분류에서 다기관·다벤더·다촬영기법으로 생기는 도메인 시프트를 줄이기 위한 멀티스테이지 프레임워크를 제안한다. 핵심은 (1) 라벨 없이 AdaIN과 CycleGAN 기반 style transfer로 벤더/기법 특성을 반영한 훈련 샘플을 생성하고, (2) Swin Transformer V2 분류기가 이 생성 데이터를 포함해 최종 예측을 수행하는 구조다.

- **Technical Challenges**: 가장 큰 기술적 난관은 ‘새 벤더/새 촬영기법에 대한 성능 저하’를 초래하는 도메인 시프트를 추가 라벨 없이 다루는 것이다. 저자들은 unlabeled vendor/technique 영상을 이용해 AdaIN과 CycleGAN으로 적응 데이터를 만들되, AdaIN에는 calcification 경계를 보존하기 위한 보조 segmentation branch를 붙이고, 생성 데이터는 최종 추론 단계에서는 사용하지 않는 방식으로 안정적으로 학습·평가를 분리했다.

- **Empirical Impact**: OPTIMAM에서 단일 사이트 기준 backbone을 비교해 Swin Transformer V2를 채택한 뒤, 외부 검증에서 성능이 일관되게 개선됐다. EMBED에서 AUC가 0.68→0.72, Duke Calcification Dataset v1에서 0.68→0.73으로 상승했으며, 벤더별로도 GE와 Hologic 모두 개선이 관찰됐다. 스타일 전이 기반 unsupervised domain adaptation이 다기관 calcification 분류의 일반화 격차를 실증적으로 줄일 수 있음을 보여 준다는 점에서 의미가 크다.



### CAIRN: Cross-Room 3D Scene Understanding with Topology-Aware Large Multimodal Models (https://arxiv.org/abs/2607.06534)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 3D-LLM은 주로 단일 방(single-room) 스캔에서 질문에 답하는 데 초점을 맞췄고, 장면을 물체 토큰의 평면 열로 보고 self-attention으로 모든 쌍을 촘촘히 연결하는 방식이 많다. 그 결과 다실(multi-room) 환경의 방 간 연결성, 방 위계 구조, 교차 공간 간 암시적 후보 탐색 같은 과제를 구조적으로 다루기 어렵다.

- **Core Contribution**: 이 논문은 다중 방 3D 장면 이해를 위한 토폴로지(topology) 인지 3D-LLM CAIRN을 제안한다. CAIRN은 장면을 객체-방 계층 그래프로 만들고, 방 토큰으로 방 단위 추상화를 수행한 뒤 토폴로지 기반 masked attention과 geometric bias로 정보 흐름을 위계/연결성에 맞춰 라우팅한다.

- **Technical Challenges**: 핵심 난제는 다중 방에서 상호작용이 ‘희소하고 구조적’이라는 점인데, 기존의 전연결 self-attention은 이를 반영하지 못한다. CAIRN은 (1) 계층 그래프 기반 토큰화(객체 관계와 방 인접성 동시 보존), (2) 계층 마스크로 허용된 경로에만 주의(attention) 정보가 흐르게 제한, (3) 상대 위치·거리·방 인접 같은 공간 priors를 geometric bias로 attention logits에 주입해 교차 방 추론에 필요한 신호를 강화한다.

- **Empirical Impact**: 논문은 HM3D 기반의 다중 방 벤치마크 CAIRN-MR을 새로 구축하고, grounding/captioning 및 4가지 cross-room QA를 포함해 점진적으로 더 어려운 추론을 평가한다. CAIRN은 CAIRN-MR 전 태스크에서 기존 3D-LLM 대비 큰 폭으로 성능을 향상시키며, 동시에 단일 방 5개 벤치마크에서는 경쟁력(대체로 최상/차상위)을 유지한다. 특히 성능 향상은 비교/존재 확인처럼 방 간 추론 의존도가 큰 태스크에서 더 크게 나타나, 토폴로지 기반 설계가 의도한 대로 작동함을 실험적으로 뒷받침한다.



### Point as Skeleton: Accumulated Point Cloud Enhanced Autoregressive Generation for Closed-Loop Autonomous Driving Simulation (https://arxiv.org/abs/2607.06516)
- **Prior Approaches**: 기존 E2E-AD 평가는 폐루프 상호작용이 가능한 CARLA류 시뮬레이터와, nuScenes처럼 시각적 충실도는 높지만 고정된 관측이라 정책의 행동 변화에 즉시 반응하기 어려운 데이터 기반 로그의 한계 사이에서 어렵다. 생성형 센서 시뮬레이션은 이 간극을 메우려 하지만, 고전적 재구성/생성 하이브리드는 오프로그(로그 궤적 이탈)에서 프레임 단위 롤아웃에 바로 쓰기 어렵고, 풀 클립 비디오 생성기를 그대로 AR(autoregressive) 롤아웃에 적용하면 오차 누적과 보수적(정체) 움직임이 커진다. Rolling diffusion 같은 롤링 추론도 미래 조건이 latent에 섞여 다음 시뮬레이션 스텝으로 “커밋”되면 상태 불일치가 누적될 수 있다.

- **Core Contribution**: 이 논문은 Point as Skeleton으로, 상태가 갱신되는 폐루프 시뮬레이션에서 프레임 단위 AR 드라이빙 비디오를 생성하기 위한 generative sensor simulation 프레임워크를 제안한다. 생성기는 업데이트된 ego/actor 상태와 장면 지도 조건을 바탕으로 시각 관측을 합성하고, Point-cloud skeleton 조건을 통해 롤아웃 동안 필요한 기하·외형의 앵커를 제공한다. 또한 nuPlan-SimGen을 구현해 ego가 로그를 벗어나는 조건에서도 폐루프 생성 평가를 수행할 수 있게 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 속도·진행방향 변화로 관측되지 않은 차량/보행자가 등장하는 spatiotemporal extrapolation, (2) 매 스텝 planner가 새 액션을 내리는 instantaneous interactivity 환경에서 스텝별 조건을 즉시 반영하는 프레임 단위 AR 생성의 오차 누적이다. 이를 위해 Reset-and-Roll은 rolling diffusion의 장점을 유지하되, lookahead로 인해 생성된 미래조건 latent이 다음 시뮬레이션 스텝에 넘어가지 않도록 “커밋”을 끊어 미래조건이 누적 상태에 섞이지 않게 한다. 더불어 skeleton은 foreground/background를 분해해(색(color)·템플릿 기반 depth) 카메라 뷰 투영 조건을 제공함으로써, 긴 롤아웃에서의 시각 열화와 기하 일관성 붕괴를 동시에 완화한다.

- **Empirical Impact**: nuScenes 및 nuPlan 실험에서 Point as Skeleton은 AR 롤아웃 동안의 시각 품질(FID/FVD)과 시간적 일관성, 그리고 downstream 지각 성능(UniAD, segmentation 기반 지표)을 전반적으로 개선했다. 특히 nuPlan-SimGen 폐루프 평가에서는 ego 궤적 이탈 상황에서 생성 차량이 시뮬레이터 제공 기하와 더 잘 정렬되는 point-label IoU가 가장 높아 “폐루프 정합성”이 강화됐음을 보여준다. 결과적으로 autoregressive generative model을 폐루프 드라이빙 시뮬레이터에 실사용 가능 수준으로 통합할 수 있는 방향성을 제시하며, 시뮬레이션에서 새로운 액션을 매 스텝 삽입하는 상호작용성도 함께 입증한다.



### AirflowAttack: Thermal-Airflow Adversarial Perturbations against Infrared Remote-Sensing Vision-Language Models (https://arxiv.org/abs/2607.06485)
- **Prior Approaches**: 기존 원격탐사 vision-language model(VLM)은 주로 정상(benign) 환경에서 성능을 평가해 왔고, IR 원격탐사에서의 보안 취약성은 거의 다뤄지지 않았다. RGB 영역에서는 white-box/black-box/Universal UAP 등 다양한 적대 공격이 연구됐지만, 열화상 단일 채널의 물리적 의미(온도에 준하는 방사 강도)와 그에 기반한 ‘물리 그럴듯한 교란’이 IR VLM에 주는 영향은 공백이었다.

- **Core Contribution**: 이 논문은 IR 원격탐사 VLM을 겨냥한 최초의 적대 공격 프레임워크 AirflowAttack을 제안한다. 열-대기 흐름 난류(thermal-airflow turbulence)에서 영감을 받은 공기(airflow) 우선(prior)을 도입해, 입력에 무관한 단일 universal perturbation을 생성·최적화하며 물리적으로 해석 가능한 교란을 목표로 한다.

- **Technical Challenges**: 핵심 과제는 (1) 타깃 VLM에 대한 접근 없이도 전이되는 공격을 만들고, (2) 단순 디지털 잡음이 아니라 열 흐름의 공간적 상관을 갖는 교란을 생성하며, (3) 교란 크기(L∞≤ε) 제약까지 만족시키는 것이다. 이들은 저차원 잠재변수에서 열-흐름 패턴을 내는 lightweight generator로 perturbation을 구성하고, surrogate CLIP에서의 confidence 저하 손실에 더해 airflow correlation loss로 물리적 그럴듯함을 정규화해 해결했다.

- **Empirical Impact**: Surrogate CLIP(한 모델)에서 최적화한 단일 perturbation은 5개 CLIP 백본 평균 zero-shot scene-classification ASR 48.5%를 달성해 IR 특화 물리 베이스라인(27.7~37.0%)을 크게 앞섰다. 6개 최신 VLM에 적용하면 장면 분류 정확도를 최대 38.2% 상대적으로 낮추는 한편, 일부 모델은 오히려 IR-cue에서 더 높은 ‘확신’을 보이며 온도 구배·대류 같은 가짜 열 증거로 교란을 ‘사실처럼’ 받아들이는 역설적 현상도 관찰됐다. 또한 11개 모델·4개 태스크에 걸친 벤치마크와 ablation으로 airflow prior가 공격 성공을 해치지 않으면서 물리적 타당성을 올려 취약성을 체계적으로 드러냈다는 점에서 의미가 크다.



### Mitigating Domain Shift in Conditioned Floor Plan Generation: Synthetic Pre-training for Data-Efficient Adaptation (https://arxiv.org/abs/2607.06483)
- **Prior Approaches**: 조건부 바닥 평면 생성은 최근 Diffusion Models와 Flow Matching 흐름을 타고 빠르게 발전했지만, 대부분 단일 데이터셋 내 성능만 평가해 교차 도메인 일반화가 검증되지 않았다. 기존 벡터 기반 접근은 (1) arrangement 기반으로 입력 폴리곤에 rigid transformation을 적용하거나, (2) vertex 기반으로 폴리곤 꼭짓점을 diffusion/flow로 갱신한 뒤 graph 제약을 맞추는 방식으로 나뉜다. 그러나 RPLAN, MagicPlan, Swiss Dwellings 간 도메인 이동 상황에서는 공통적으로 성능이 크게 붕괴하는 문제가 관찰된다.

- **Core Contribution**: 저자들은 조건부 floor plan generation에서 도메인 shift가 체계적·양방향으로 발생함을 대규모 실험으로 정식화하고, 두 패러다임 모두 같은 취약점을 가진다고 보여준다. 이를 완화하기 위해 타깃 도메인에 대한 최소한의 감독만으로 학습을 돕는 절차적(pre-training) 전략을 제안한다. 핵심은 “건축적 그럴듯함”을 일부러 버리더라도 비중복성, 문 배치 타당성, graph consistency 같은 물리적 제약을 강하게 강제하는 합성 데이터로 스타일에 덜 민감한 조립 규칙을 학습시키는 것이다.

- **Technical Challenges**: 문서가 지적하는 기술적 난제는 타깃 도메인의 공간 통계(기하 분포·위상 관례)가 달라질 때 모델이 데이터셋 고유의 편법(shortcut)에 의존한다는 점이다. 이를 해결하기 위해 합성 파이프라인은 RPLAN에서 방(단독) 폴리곤 형태를 약한 priors로 추출한 뒤, 종횡비 왜곡, edge bump/hollow, 고각 회전/플립, 큰 스케일 변화를 거쳐 시각적/통계적 유사성을 의도적으로 끊는다. 그럼에도 배치 단계에서는 overlap을 금지하고 shared wall 기반 문을 graph 연결과 함께 검증·거절(rejection)하는 방식으로 물리적 제약을 엄수해, 스타일이 아니라 조합 로직을 학습하게 만든다.

- **Empirical Impact**: 실험 결과, 어떤 단일 도메인에서 학습한 모델도 다른 데이터셋으로 전이할 때 최대 1차수(order of magnitude) 수준의 성능 저하가 발생한다. 반면 제안한 합성 데이터 pre-training은 zero-shot 교차 도메인 성능을 크게 끌어올리며, MagicPlan에서는 in-domain 학습보다도 높은 성과를 보이고 Swiss Dwellings에서도 최상위 교차 성능을 달성한다. 또한 fine-tuning에서는 저데이터(예: 1k 샘플) 구간에서 오차를 약 40%까지 줄이며, 다른 생성 패러다임(도메인 취약성이 동일한 vertex-level diffusion)에서도 일관된 데이터 효율 향상을 재확인한다.



### Prompt-Adapter Context Routing for Parameter-Efficient Multi-Shot Long Video Extrapolation (https://arxiv.org/abs/2607.06481)
Comments:
          10 pages, 2 figures

- **Prior Approaches**: 장기 비디오 생성은 기존의 샷 플래닝, 스토리 메모리, 스트리밍/에이전트 방식 등으로 접근돼 왔습니다. 특히 recursive context allocation 계열은 “어떤 과거 컨텍스트를 다음 샷에 줄지”를 결정해 드리프트를 줄이려 하지만, 대체로 큰 부분을 fine-tuning 하거나 외부 메모리 모듈 의존도가 생겨 장기 구간에서 retrieval이 불안정해질 수 있습니다.

- **Core Contribution**: PACR-Video는 text-to-video diffusion transformer를 frozen으로 두고, 저랭크 temporal adapters와 shot-role prompt tokens, recursive prompt bank를 조합해 멀티샷 장기 외삽을 수행합니다. 핵심은 dense한 비디오 메모리 대신 entity/location/action/style 요약 프롬프트를 저장하고, narrative dependency 예측에 따라 adapter gates로 필요한 것만 선택 라우팅한다는 점입니다.

- **Technical Challenges**: 장기 생성에서 오류가 샷 누적로 커지는 문제를 해결하려면, (1) 오래 남는 정체성/스타일은 유지하면서 (2) 샷마다 새 모션·뷰포인트·인과 이벤트는 진행되도록 제어해야 합니다. 논문은 Shot-Local/Story-Global 학습목표(다음 샷 재구성, cross-shot identity contrast, prompt sparsity)와, early-shot 시각 일관성을 강화하고 late-shot 이벤트 진행을 늘리는 adapter composition schedule로 long-horizon coherence를 맞춥니다.

- **Empirical Impact**: FlintstonesSV, Pororo-SV, ActivityNet Captions, YouCook2, Shot2Story, MovieNet의 6개 벤치마크에서 PACR-Video는 텍스트-투-비디오, tuning-based, streaming, memory-augmented, recursive-context baselines보다 모든 핵심 지표(FVD, identity consistency, temporal smoothness, 전이 일관성 등)에서 우수했습니다. 특히 ReCA 대비 FVD 268.4→231.7, DINO identity consistency 0.724→0.771로 개선했고, 사람 평가에서도 ReCA보다 63.8% 더 선호되었으며, 백본 파라미터의 3.8%만 튜닝해 실용성까지 확보했습니다.



### A VLM-Enhanced Framework for Comprehensive Traffic Sign Condition Assessment Integrating Daytime Visual Performance and Nighttime Retroreflectivity Evaluation (https://arxiv.org/abs/2607.06478)
Comments:
          21 pages, 7 figures, 5 tables. Preprint. An earlier version of this work was presented at the 105th Annual Meeting of the Transportation Research Board (TRB), January 2026

- **Prior Approaches**: 기존 평가는 주로 사람이 직접 육안 점검해 주간 가독성·색 대비 같은 요소와 야간 반사성능을 확인하는 방식이었지만, 주관적이고 노동 집약적이며 안전상 위험이 따른다는 지적이 많습니다. 또한 retroreflectometer는 비용이 높아 소규모 기관에 보급이 어렵고, 연구도 주간 요인이나 야간 retroreflectivity 중 한쪽에 치우치는 경우가 대부분이었습니다.

- **Core Contribution**: 이 논문은 주간과 야간을 한 번에 아우르는 통합 평가 프레임워크를 제안해 교통표지의 상태를 종합적으로 수치화합니다. 특히 세 가지 fine-tuned Vision Language Model(VLM)로 주간 핵심 요인(legibility, color, 표면·형상 무결성, 주변 환경)을 평가하고, 야간은 LiDAR 기반 retroreflectivity로 측정해 Sign Condition Index(SCI)로 통합합니다.

- **Technical Challenges**: 기술적으로는 VLM의 언어 기반 판단을 유지보수에 쓸 수 있는 수치 점수로 변환하는 문제가 있었고, 이를 sentiment analysis와 CLIP scoring으로 수치화해 일관된 scoring을 구성했습니다. 또 야간 성능은 LiDAR-derived retroreflectivity를 활용하되 기존 calibration 절차에 맞춰 신뢰도를 확보하는 방식으로 결합했습니다.

- **Empirical Impact**: 실험에서 LLaVA와 Qwen이 InternVL보다 우수했으며, 모든 요인에 대해 bidirectional cosine similarity 0.67~0.76 수준의 성능을 보였습니다. 검증된 462개 표지 중 68개가 retroreflectivity 성능 부족으로 즉시 교체가 필요하다고 플래깅되었고, 결과적으로 수작업 점검 대비 비용 효율적인 종합 평가 대안을 제시했다는 점에서 의미가 큽니다.



### EgoPolice: A Benchmark for Egocentric Video Understanding in High-Stakes Police Body-Worn Camera Footag (https://arxiv.org/abs/2607.06468)
- **Prior Approaches**: 기존 경찰 BWC(Body-Worn Camera) 연구는 주로 사회과학적 효과를 평가하거나, 대화 전사 후 자연어처리로 행동을 분석하는 데 집중해 왔습니다. 반면 시각 기반 영상 이해는 제한적이었고, 상용 도구는 성능을 독립 검증할 수 있는 벤치마크가 부족했습니다. 또한 일반적인 egocentric 데이터셋은 촬영이 비교적 안정적이거나(저위험 상황) 실험실/연구 목적의 인지된 촬영 맥락이 섞인 경우가 많았습니다.

- **Core Contribution**: EgoPolice는 실제 경찰-민간인 상호작용을 담은 egocentric BWC 데이터셋으로, 초 단위(초당) 그라운드 트루스 라벨을 제공합니다. “경찰(카메라 착용자) / 다른 경찰 / 민간인”을 구분하고, 관찰 가능한 행동 기준으로 고위험 사건에 해당하는 액션 클래스들을 체계화했습니다. 분류와 MCQ(multiple-choice question-answering) 두 태스크로 오픈·클로즈드 소스 모델을 벤치마크해 신뢰성 한계를 드러냅니다.

- **Technical Challenges**: BWC 영상은 주석 구간에서조차 카메라 흔들림이 심하고, 배경·외형 단서가 클래스 간 구분에 거의 도움을 주지 못해(글로벌 외형 유사) 모션·미세 단서 의존이 커집니다. 또 “Weapon Out”, “Any Officer-Handcuffing”처럼 희귀하고 작은 물체(총/수갑)가 부분 가림 속에 등장해 탐지 자체가 어렵습니다. 논문은 의도 추론을 배제한 객관적 정의와 다단계 주석 파이프라인(초 단위 라벨링, 고품질 검수)을 통해 라벨 모호성을 줄이는 방향으로 해결합니다.

- **Empirical Impact**: 실험 결과, VideoMAE V2·X-CLIP 같은 영상 사전학습 모델이 상대적으로 강하지만 “Weapon Out” 등 고위험 액션은 여전히 정확히 예측하지 못했습니다. 제로샷 MCQ에서도 일부 VLM은 출력 포맷 불일치로 무작위 수준 성능을 보였고, Gemini 2.5 Pro가 1분 클립에서 76.9%로 가장 잘하지만 자율 배치에는 아직 부족하다고 평가합니다. 저자들은 대규모 BWC 저장소에서 관심 이벤트를 찾아 인간 검토를 효율화하는 기초로 EgoPolice가 활용될 수 있음을 제시합니다.



### Verification of Dynamic Holographic Behavior in Identity Documents (https://arxiv.org/abs/2607.06466)
Comments:
          Accepted at the International Conference on Document Analysis and Recognition (ICDAR 2025)

- **Prior Approaches**: 기존 연구는 대부분 Presentation Attack Detection(AD)처럼 공격 징후를 찾거나, Model Verification(MV)에서는 OVD(광변환 소자/홀로그램)의 일부 외형·행동만 확인하는 방식에 머물렀습니다. 이 때문에 간단한 홀로그램 대체나 조명·반사 효과를 이용한 정적 사기, 혹은 동적인 중간 상태를 악용하는 공격을 제대로 일반화해 탐지하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 1) MIDV-Holo를 확장한 공개 데이터셋 MIDV-DynAttack과 2) 배경 제거 및 동적 거동 기반 의사결정으로 특정 홀로그램의 진위를 검증하는 방법( HoloVerif ), 3) 명확한 평가 프로토콜의 벤치마크를 제안합니다. 특히 MIDV-DynAttack은 iPhone 7, Redmi Note 8 Pro, G7 등 스마트폰 환경에서 정적·동적 위조 시나리오를 포함해, 동적 사기 검증 공백을 메웁니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 다이내믹한 홀로그램 신호를 배경(문서 템플릿·사진)에서 안정적으로 분리하고 (2) 프레임 단위의 ‘정상 동적 거동’ 라벨을 동적 공격 없이도 학습할 수 있어야 한다는 점입니다. 논문은 시퀀스의 per-pixel per-channel median 기반 배경 subtraction으로 신호를 강화한 뒤, 합법 시퀀스의 밝기/변화 패턴을 기준으로 pseudo-label을 생성해 프레임 분류기를 학습하고, 시퀀스에서 Valid 프레임 비율 임계값으로 최종 Legit/Non-Legit을 판정합니다.

- **Empirical Impact**: 실험은 기존 MIDV-Holo 벤치마크 프로토콜을 따르되, MIDV-DynAttack을 ‘학습에 쓰지 않는 테스트셋’으로 사용해 미지 공격 일반화 성능을 보여줍니다. HoloVerif는 동적 공격에서 recall이 크게 개선(예: 29%→61%)되며, 기존 방법 대비 동적 위조에 더 강한 보안 보장을 제공하는 것으로 보고됩니다. 또한 Photo Replacement 및 Document Swap 같은 특정 취약 지점이 드러나, 향후 temporal color variation 모델링 등 추가 보완 방향이 제시됩니다.



### Andha-Dhun: A First Look at Audio Descriptions in Hind (https://arxiv.org/abs/2607.06457)
Comments:
          Accepted to NCVPRIPG 2026, Download data at this https URL

- **Prior Approaches**: AD(Audio Descriptions)는 대사 공백에 삽입되는 시각 정보 내레이션으로, 기존 연구와 자동 생성은 주로 English 중심이었다. 번역 기반 접근도 유럽 언어에서 post-editing 부담, timing mismatch, visually grounded 데이터 부족 같은 한계가 반복적으로 보고됐다.

- **Core Contribution**: 이 논문은 인도 언어 중 최초로 Hindi 오디오 디스크립션을 다루며, 데이터·생성·평가를 한꺼번에 체계화했다. 특히 8편 장편영화에서 수집한 Andha-Dhun(5,870개 human-authored Hindi AD 문장)을 공개하고, 영어 AD를 Hindi로 만드는 두 경로(번역 vs dense 캡션 직접 증류)를 비교한다.

- **Technical Challenges**: 핵심 난제는 언어적 문장 생성뿐 아니라 Hindi BLV 관점에서의 목적 지향적( Skopos theory ) 문화 참조 적응이다. 연구진은 perplexity(언어 다양성/예측가능성)와 LLM-as-a-judge(0~5 품질 유사도)를 함께 쓰고, Culture-Specific Items(CSI)가 해결되는지까지 별도 분석해 번역이 목적을 놓칠 때 어떤 문제가 생기는지 드러냈다.

- **Empirical Impact**: 결과적으로 Dense-to-Hindi 직접 생성(특히 Hindi-전용 Nemotron)은 번역 기반보다 LLM-AD-eval 점수와 perplexity 양상이 모두 더 좋았지만, 여전히 human-level에는 못 미쳤다. 또한 CSI 해결률은 HI-Human이 42.5%로 기계 번역(10.0%)을 크게 앞섰고, 번역은 대부분 retention/direct translation에 머물러 문화 적응이 부족했다는 점이 실증적으로 확인됐다.



### Analysis-by-Proxy: Localization Signals in VLMs Operating as Condition Encoders (https://arxiv.org/abs/2607.06445)
Comments:
          Accepted as a Spotlight at the ICML 2026 Mechanistic Interpretability Workshop

- **Prior Approaches**: 기존 확산 기반 이미지 편집은 VLM을 condition encoder로 두고, DiT에 전달할 VLM 내부 표현을 미리 정한 특정 레이어(대개 final-layer 토큰, 레이어 풀링, 일부 레이어를 DiT 레이어에 매핑)에서만 뽑는 방식이 주류입니다. 하지만 다중 객체 장면에서 원하는 대상을 정확히 고르는 localization이 자주 무너져, 약간의 신호 손실만으로 잘못된 위치/환각 편집이 발생합니다. 또한 VLM 해석 연구는 주로 autoregressive 텍스트 생성 기반으로 진행되어, single forward pass 제한에서의 내부 동작은 충분히 규명되지 않았습니다.

- **Core Contribution**: 이 논문은 VLM이 single-pass condition encoder로 사용될 때 성능 격차가 왜 생기는지 분석하며, 핵심 원인을 “VLM의 공간 지식이 condition extraction 과정에서 제대로 디코딩되지 못함”으로 제시합니다. 이를 위해 Analysis-by-Proxy라는 프레임워크를 도입해, 생성 없이도 VLM 중간 표현에서 localization 정보를 어떤 레이어/토큰에 담는지 분해합니다. 더 나아가 proxy가 복원한 공간 신호(바운딩 박스)를 DiT 조건에 통합해 실제 편집 localization 실패를 줄이는 방법을 제안합니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 텍스트 토큰을 생성하지 않는 single-pass 환경에서 연속 hidden state 안의 공간 정보를 “디코더 없이” 찾아내야 한다는 점입니다. 논문은 이를 해결하기 위해 Q-Former 기반의 lightweight proxy를 VLM 중간 표현에 학습시켜, auxiliary localization task로 바운딩 박스를 회귀하게 만들고 decodability가 높은 레이어/토큰 위치를 역추적합니다. 결과적으로 final-layer에선 공간 신호가 과추상화되어 약해지고, 중간 레이어에서 신호가 강하지만 ‘입력 프롬프트마다’ 피크 레이어가 달라져 고정 레이어 추출 전략이 본질적으로 비최적임을 보였습니다.

- **Empirical Impact**: 200개 다중 객체 장면 평가에서 VLM 단독은 89.0%로 정확한 바운딩 박스를 예측했지만, 표준 편집 파이프라인은 57.5%로 크게 하락해 31.5%p 격차를 실증적으로 확인했습니다. Analysis-by-Proxy로 복원한 중간 레이어 공간 신호를 LoRA로 DiT에 조건화하면, 기존 Qwen-Image-Edit 대비 VQA 기반 의미 편집 성공을 높이고 대상 외 배경 LPIPS는 낮게 유지하는 결과가 보고됩니다. Full Description 같은 autoregressive 변형이 강한 베이스라인으로 관찰되어, sequential decoding이 공간 단서를 끌어내는 한편 single-pass 추출 설계가 병목임을 더 뒷받침합니다.



### PIPBench: A Profile-Inclusive Framework for Personalized Image Generation Evaluation (https://arxiv.org/abs/2607.06440)
- **Prior Approaches**: 기존 text-to-image 모델(DALL·E 3 등)은 지시를 잘 따르지만, 사용자의 암묵적 미학 취향은 반영하지 못한다. 또한 personalized generation 연구는 주로 style transfer나 image editing의 가이드(참조 이미지/텍스트)에 의존하거나, test-time fine-tuning처럼 사용자별로 파라미터를 조정하는 방식이 많아 확장성·평가 일관성에 한계가 있었다.

- **Core Contribution**: 이 논문은 사용자 취향을 소수의 선호 이미지와 짧은 프롬프트로 맞추는 personalized image generation 문제를 정식화하고, 이를 평가할 최초의 profile-inclusive 벤치마크 PIPBench를 제안한다. PIPBench는 심리·인구통계 프로파일 축을 기반으로 실사용자 프로파일과 선호 이미지를 함께 수집/구성해, “암묵적 취향”을 비교 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 암묵적 시각 선호를 설문만으로는 충분히 수집·정량화하기 어렵고, 그래서 벤치마크 데이터 품질을 체계적으로 확보하기 어렵다는 점이다. 이를 위해 논문은 실데이터는 신뢰도 있게 확보하되, 에이전트 기반 합성 데이터 생성 파이프라인으로 프로파일 분포의 다양성을 확장하고(일관성 규칙·다양성 필터링 포함), LLM 기반 캡션 구성→diffusion으로 후보 이미지를 만든 뒤 실제 사용자 선택/자동 랭킹으로 선호 이미지를 확정한다.

- **Empirical Impact**: PIPBench에서 대표 방법들을 대규모로 실험한 결과, 단순히 참조 이미지를 여러 장 넣는 joint conditioning은 오히려 성능을 떨어뜨릴 수 있고, 이는 다중 참조 이미지의 공동 해석이 여전히 어렵다는 신호로 해석된다. 또한 persona-aware Elo(LLM-as-a-judge)를 도입해 자동 지표(CLIP 유사도 등)와 함께 검증했으며, 자동 지표와 Elo 순위가 강하게 상관돼 평가 프레임의 신뢰성을 뒷받침한다. 전반적으로 이 연구는 personalized text-to-image에서 ‘취향 정렬’의 새 난제와 메서드 개발 기회를 구체화했다.



### XRFormer: Multiscale Tokenization for XRF Representation Learning (https://arxiv.org/abs/2607.06424)
Comments:
          International Conference on Pattern Recognition, 2026

- **Prior Approaches**: XRF 스펙트럼은 날카로운 원소 피크와 넓은 구조, 배경 변동이 섞인 1차원 신호인데, 기존 학습 기반 모델은 이를 충분히 반영하지 못했다. 특히 1D-CNN 계열은 로컬 피처에는 강하지만 에너지 축 전반의 장거리 의존성을 모델링하는 데 한계가 있으며, 트랜스포머를 쓰더라도 XRF에 맞춘 토크나이징이 약한 경우가 많다. SpectralFormer, ViT 변형들은 로컬 연속성이나 단일 스케일 임베딩을 주로 다뤄 피크-배경의 멀티스케일 특성을 충분히 결합하지 못한다.

- **Core Contribution**: 이 논문은 XRF 전용 트랜스포머인 XRFormer를 제안하며, 핵심은 ‘멀티스케일 컨볼루션 토크나이저’로 스펙트럼의 국소성(locality)과 다중 해상도(multi-resolution) 편향을 먼저 주입하는 데 있다. 토크나이저는 해상도를 점진적으로 줄이면서 임베딩 차원을 키워, 전역 self-attention 이전에 피크와 배경을 서로 다른 스케일로 함께 표현한다. 또한 self-supervised pretraining으로 Masked Spectral modeling(MSM)과 물리 기반 Peak Presence Prediction(PPP) 전처리 과제를 함께 실험한다.

- **Technical Challenges**: 어려움은 날카로운 원소 피크와 완만한 배경 변동이 동시에 존재하는 XRF 특성상, 단순 패치/고정 스케일 토크나이징으로는 필요한 표현이 충분히 형성되지 않는다는 점이다. XRFormer는 이를 해결하기 위해 1D 컨볼루션 블록을 통해 유효 수용영역을 키우면서 다운샘플링을 수행하는 토크나이저를 만들고, 이후 표준 transformer encoder로 전역 의존성을 학습한다. 전처리는 MSM으로 누락 토큰 복원을 강제해 구조 전반을 학습하고, PPP는 입력에서 피크 prominence로 추출한 ‘피크 존재’ 시그니처를 예측하도록 설계해 [CLS] 표현이 피크 분포를 인코딩하게 만든다.

- **Empirical Impact**: Pigments Checker STANDARD v.5(PCSv5)에서 XRFormer는 사전학습 없이도 ViT, SpectralFormer, 1D-CNN을 전반적으로 능가하며, 특히 피그먼트 식별에서 성능 격차가 뚜렷하다. 사전학습을 MSM으로 추가하면 AA가 71.29%에서 4.6%p 향상하고 A-RMSE도 감소했으며, MSM+PPP는 AA 76.78%로 최고 식별 성능을 보이면서 unmixing에서 SpectralFormer와의 격차를 좁혔다. 또한 XRFormer는 토큰 수 128로 더 낮은 해상도에서 동작하면서도 파라미터를 1.5M로 유지해, SpectralFormer(512 tokens, 3.37M) 대비 효율이 높고 데이터가 제한된 문화유산 분야에서 “모달리티 맞춤 토크나이징+전처리”의 실효성을 입증했다.



### HoloCount: A Holistic Visual Counting Benchmark for MLLMs (https://arxiv.org/abs/2607.06420)
Comments:
          Technical report

- **Prior Approaches**: 기존 VLM/MLLM의 비주얼 카운팅 평가는 CountBench, CountQA, PixMo-Count처럼 대부분 단일 정확도(accuracy) 중심의 평탄한 벤치마크에 머물렀습니다. 이들은 객체 다양성이 제한적이고, 속성 제약·논리 조합·악조건 환경에서 나타나는 실패 양상을 충분히 분리해 진단하지 못했습니다. 그 결과 모델이 ‘보이는 대로 세기’에서는 잘해도 수치 정밀도가 무너지는 원인을 추적하기 어렵습니다.

- **Core Contribution**: HoloCount는 MLLM의 카운팅 능력을 Semantic Counting, Analytical Counting, Robustness Testing의 3단계 계층 구조로 분해해 진단형 평가를 제공합니다. 특히 속성 기반 분류(예: 색/재질/상태)에서 시작해, ROI 기반 공간 추론과 집합 연산(합집합/차집합/차이/종류 수)까지 논리적 조합을 체계적으로 요구합니다. 또한 null target, linguistic prior conflict, 고밀도/드론 시점/부분 가림 등 반대 조건에서의 수치 환각 취약점을 함께 측정합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘모호성 제거’와 ‘정답의 단일성’ 확보였습니다. HoloCount는 인스턴스 스케일·시점·가림이 지나치게 애매한 샘플을 제거하고, 생성된 QA가 정답과 어긋나는 경우를 모델 검증 및 전문가 인력 검수로 반복적으로 걸러 정밀한 수치 라벨을 유지합니다. 또한 데이터 생성은 대규모 수집·정제→모델 기반 QA 생성/검증→최종 사람 검증의 다단 파이프라인으로 구성되어, 진단 목적에 맞는 난이도 축을 안정적으로 만듭니다.

- **Empirical Impact**: 20개 이상 최신 MLLM을 2,480개 QA(20개 세부 과제)로 0-shot 평가한 결과, 인식형 과제에서 분석형·악조건 과제로 넘어갈수록 성능이 크게 하락하며 수치 환각이 지속됨을 확인했습니다. 즉, ‘질적 장면 이해’ 강점과 별개로 ‘정확한 숫자 접지(grounding) + 논리 추론 + 견고성’이 현재는 동시 달성이 어려운 격차임이 드러났습니다. HoloCount는 실패가 시각 정렬 문제인지, 논리/산술 조합 문제인지, 혹은 반대 선입견에 대한 취약성인지까지 구분해 후속 모델 개선 로드맵을 제시하는 벤치마크로 의미가 있습니다.



### Temporal Modeling of Optically Variable Devices in Identity Documents (https://arxiv.org/abs/2607.06408)
Comments:
          Accepted at the International Conference on Document Analysis and Recognition (ICDAR 2026)

- **Prior Approaches**: 기존 OVD(OVD=hologram) 검증은 대체로 프레임을 독립적으로 보거나, 존재 여부 수준의 시각적 특징에 의존해 왔습니다. HoloVerif처럼 동작을 일부 다루는 방법도 ‘유효 프레임의 평균’ 등으로 처리해 시간적 전이를 정교하게 모델링하지 못해 swapping 공격 등에 취약했습니다. 또한 대부분은 공격 샘플(또는 합성/준합성 negative)을 학습에 활용해 실무의 공격 데이터 희소성 제약을 그대로 받습니다.

- **Core Contribution**: 이 논문은 투명 OVD의 ‘시간적 동특성’을 검증하는 데 초점을 두고, 공격 샘플 없이 self-supervised/legit-only로 학습 가능한 두 가지 방식을 제안합니다. Discriminative 방식인 HoloVerif-Span은 연속 프레임 스팬에서 정상 전이 vs 교란을 판별하도록 설계했고, Generative 방식인 MSM(Masked Sequence Modeling)은 정상 OVD 전이의 manifold를 학습해 OOD(이상) 기반 재구성 오류·다양성으로 판별합니다. 특히 훈련 시 공격 유형을 모르는 open-set 상황을 목표로 삼아 산업 제약을 엄격히 만족시키는 실용성을 강조합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 투명 OVD 신호가 약하고(배경/조명/기기 변화), (2) 프레임 단위보다 ‘전이’가 보안의 본질이라 이를 짧은 동영상에서 안정적으로 학습해야 한다는 점입니다. 해결책으로 HoloVerif-Span은 5프레임 스팬을 VideoMAE 백본으로 처리하고, temporal coherence를 깨는 합성 교란(반복/셔플/고정)을 통해 legit-only 학습이 가능하게 했습니다. MSM은 frame projector(WSL로 OVD 표현 학습 후 고정)와 임베딩 기반 masked reconstruction을 결합해 정상 전이는 맥락으로 예측 가능하다는 가정하에 이상을 재구성 불가능성과 임베딩 다양성(유사도)로 탐지하도록 구성했습니다.

- **Empirical Impact**: MIDV-Holo와 MIDV-DynAttack에서 두 방법 모두 공격 샘플 없이도 이전 state-of-the-art를 능가하는 결과를 보였고, 특히 temporal 기반 공격(Static-swap, Dynamic)에서 성능이 크게 향상되었습니다. legit-only 기준으로 HoloVerif-Span은 전반 93.4% AUC, MSM은 91.1% AUC를 기록하며 WSL(84.7%) 대비 격차를 보였고, swapping 공격에서도 시간적 전이를 모델링한 이점이 뚜렷했습니다. 결론적으로 ‘OVD의 시간적 동특성 학습’과 ‘anomaly detection/sequence modeling’ 접근이 실사용 환경의 고정밀 검증 가능성을 보여주며, 향후 캡처 장치·조명·국가별 편차를 더 체계적으로 다루는 벤치마크 필요성도 함께 시사합니다.



### What Images Cannot Say: Language-Guided Olfactory Representation Learning (https://arxiv.org/abs/2607.06402)
Comments:
          ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 연구는 이미지와 전자코(electronic-nose) 측정을 데이터셋으로 함께 제공해도, 픽셀에는 직접 드러나지 않는 맥락적 환경 요인이 후각을 좌우해 정렬이 어렵다는 한계가 있었다. 또한 시각만으로 후각 표현을 학습하거나 단일 모달에 의존하는 baseline은 냄새의 구성 요소(물체 vs 환경)를 분리해 학습하기에 부족했다.

- **Core Contribution**: 본 논문은 vision과 olfaction을 잇는 의미적 브리지로 언어 가이던스를 쓰는 SCENT를 제안한다. Vision-Language Models(VLMs)가 장면의 객체·환경 맥락과 그에 상응할 법한 ambient smell 단서를 생성하고, 이를 기반으로 전자코 신호를 공유 임베딩 공간에 정렬하며 object-specific odor와 contextual environmental 기여를 languageguided latent decomposition로 분리한다.

- **Technical Challenges**: 핵심 어려움은 시각에서 보이지 않는 맥락적 요인이 냄새에 강하게 작용하는데도 전자코 신호를 이미지와 같은 의미 축으로 맞춰야 한다는 점이다. SCENT는 VLM의 장면 디스크립터를 semantic guidance로 활용해 후각 학습의 기준을 만들고, latent decomposition을 통해 혼합 냄새에서 객체 관련 향과 환경 관련 향을 분해하도록 학습을 설계했다.

- **Empirical Impact**: New York Smells 데이터셋에서 SCENT는 smell-to-image 및 smell-to-text 검색 과제에서 vision-only baseline 대비 유의미하게 성능을 개선하며 state-of-the-art를 달성했다. 또한 해석 가능한 후각 표현을 제공해 복잡한 냄새 혼합의 disentanglement이 가능함을 보여주며, 맥락적 의미 정보가 멀티모달 후각 정위(grounding)에 중요하다는 방향성을 제시한다.



### FADRA: Frequency-Aware Diffusion with Residual Adaptation for Video Face Restoration (https://arxiv.org/abs/2607.06389)
- **Prior Approaches**: 기존 VFR(Video Face Restoration) 방법은 프레임별 복원과 시간 집계(temporal aggregation)로 크게 나뉜다. 프레임 기반은 공간 품질은 얻지만 프레임 간 깜빡임(flickering)과 신원(identity) 드리프트가 생기기 쉽고, 반대로 시간 기반은 고주파 디테일(눈·입·치아 등)이 과도하게 뭉개지는 문제가 자주 보고된다. 최근 diffusion/flow-matching 접근도 있으나, 궤적 전 과정에서 LQ(저화질) 단서가 충분히 활용되지 않거나 주파수별(고주파 vs 저주파) 중요도를 균등하게 다루는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 주파수 인지(frequency-aware) 확산 프레임워크 FADRA를 제안하고, 사전학습된 text-to-video diffusion의 시간 일관성은 유지하면서 VFR에 맞춘 반복 잔차 적응을 수행한다. LoRA 어댑터와 LQ Pixel-Alignment Feature Fusion으로 frozen 생성 사전을 효율적으로 태스크에 맞추고, 추가로 RRAH(Repeated Residual Adaptation Head)를 붙여 flow-matching 단계마다 velocity 예측을 LQ 잠재와 함께 보정한다. 마지막으로 HVS(JPEG) 기반 Frequency-Aware Loss로 스펙트럼 대역별 가중 감독을 제공해 지각적으로 중요한 고주파 구조를 안정적으로 복원한다.

- **Technical Challenges**: 핵심 어려움은 확산/flow-matching 모델의 생성 지오메트리(temporal prior)가 픽셀 정렬 기반 복원 요구와 어긋나는 ‘fidelity gap’과, 초기 단계에서 LQ 단서를 잘못 해석하면 이후 단계에서 미세 구조가 약화되는 점이다. 이를 위해 LoRA + LQ Pixel-Alignment Fusion으로 backbone을 가볍게 정렬하고, RRAH가 TT sampling 단계마다 LQ latent와 현재 velocity prediction을 입력받아 잔차 업데이트를 반복 예측하도록 설계했다. 또한 학습 때만 주파수 도메인(DCT)에서 HVS 영감을 반영해 스펙트럼을 재가중하는 FAL을 적용해 고주파 영역의 시간적 떨림(jitter) 위험을 줄인다.

- **Empirical Impact**: 실험에서 FADRA는 VFHQ와 CelebV-HQ에서 최신 대비 더 좋은 구조 복원과 시간 일관성을 동시에 보였고, 정량적으로도 PSNR/SSIM/LPIPS/IDD 및 비디오 수준 FVD에서 우위를 보였다. 특히 VFHQ에서 FVD가 크게 개선되어(예: 38.97) 얼굴 디테일이 프레임 간 더 안정적으로 유지됨을 시사한다. 또한 CelebV-HQ는 추가 파인튜닝 없이 zero-shot으로 평가했는데도 일관되게 성능이 유지되며, 추론 속도는 diffusion 기반 VFR 대비 경쟁력 있게 보고되어 품질-효율 트레이드오프가 개선됐다는 점이 강조된다.



### VaseMuseum: Digital Intelligent Museum for Ancient Greek Pottery (https://arxiv.org/abs/2607.06374)
Comments:
          Code: this https URL. Website: this https URL

- **Prior Approaches**: 기존 비전-언어 모델(VLM)은 이미지 기반 질의응답에서 강점을 보이지만, 고대 유물처럼 전문 지식·출처 근거가 필요한 문화유산 문맥에서는 기준 이하의 벤치마크 성격 때문에 실제 박물관 대화의 신뢰성을 충분히 못 담는다는 한계가 지적돼 왔다. 또한 3D-aware 방법들은 주로 대규모 사전학습/파인튜닝이나 합성 데이터에 의존하는 경우가 많아, 새로운 컬렉션에 빠르게 적용하기 어렵다. 웹/툴을 쓰는 에이전트도 있었으나, 인용과 불확실성 보정이 약해 근거 없는 그럴듯한 답을 내놓기 쉽다는 문제가 남아 있다.

- **Core Contribution**: 본 논문은 고대 그리스 도예(ancient Greek pottery) 가상 박물관을 위한 경량 모듈형 멀티모달 에이전트 프레임워크 VaseMuseum을 제안한다. VaseAgent는 2D 이미지와 3D 아티팩트에 대해 지각-3D-aware 추론-외부지식 검색-추론 시점 신뢰도 제어를 한 흐름으로 묶어, 답변을 ‘보이는 것’이 아니라 ‘근거를 찾고 검증하는 과정’으로 전환한다. 특히 출처 수준(source-level) 필터링과 답변 수준(response-level) 감사를 결합해, 근거 부족·충돌 상황에서 중립적이고 증거-구속형으로 말하도록 설계했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 개방형 해석에서 세밀한 시각 증거를 전문 큐레이터 지식으로 근거화해야 하는데 검색 과정에서 약한 출처가 섞이기 쉽다는 점, (2) 증거가 불완전하거나 모호할 때도 VLM이 자신감은 높이고 근거는 없는 답을 만들기 쉽다는 점이다. 이를 위해 DeepResearch-style 반복 검색을 하되, URL 접근성·텍스트 충분성·도메인 사전·다양성(MMR) 등을 이용한 source control로 신뢰 가능한 증거 풀을 구성한다. 이어 claim 단위 커버리지·교차출처 충돌 감지로 evidence audit를 수행하고, 불충분/충돌이면 불확실 모드에서 ‘중립+증거-구속’ 프리앰블을 붙인다. 추가로 VLM 백본 업데이트 없이 GRPO 스타일의 training-free 후보 선택(링크 유효성·근거 지지·중립성·보정된 confidence 기준)을 적용해 더 신뢰도 높은 출력으로 랭킹한다.

- **Empirical Impact**: 가상 박물관 시뮬레이션에서 visual-only, visual-plus-knowledge, ambiguous 질문을 평가한 결과 VaseAgent는 지식 집약 질의에서 환각(hallucination)을 줄이고 groundedness와 neutrality를 높였다. 특히 검색만 쓰는 기준선은 인용 링크가 많이 부정확해지지만, VaseMuseum은 link validity를 크게 개선하고 모호한 상황에서 과신 대신 더 신중한 답변을 유도했다. 또한 ablation 분석에서 source control 제거는 주로 인용 품질을, response control 제거는 주로 중립성을 해친다는 점이 확인돼 두 계층 신뢰도 제어가 보완적으로 작동함을 보여준다.



### TMF-RSE: Tri-Modal Fusion with Regional Semantics and Evidential Uncertainty for Lung Severity Scoring (https://arxiv.org/abs/2607.06356)
Comments:
          6 pages, 2 figures, 5 tables. IEEE conference format (IEEEtran). Submitted to AVSS 2026. Tri-modal fusion for lung severity scoring using appearance, segmentation, and VLM semantics with evidential uncertainty

- **Prior Approaches**: 기존 폐 질환 심각도(continuous severity) 회귀는 주로 외관(appearance) 단서에 강점이 있는 transformer 계열이나 피라미드 아키텍처가 중심이었습니다. Brixia·RALE·CheXpert 같은 전통 스코어링을 바탕으로 한 자동화는 발전했지만, 폐 segmentation에서 얻는 구조 priors나 임상 의미를 직접적으로 결합해 ‘마스크 인지 회귀’와 ‘의미 기반 정교화’를 동시에 만족시키는 시도는 제한적이었습니다. 또 evidential regression(aleatoric/epistemic 분리)을 쓰더라도, tri-modal 융합과 불확실성의 상호 보완을 함께 다룬 시스템은 드물었습니다.

- **Core Contribution**: TMF-RSE는 외관(2D 흉부 입력), 구조(폐 segmentation 마스크), 의미(VLM 임베딩)를 한 end-to-end 프레임워크에서 통합해 심각도를 예측합니다. 특히 upper/middle/lower 폐 영역별로 VLM의 의미 임베딩을 의미 게이팅에 사용해, 시각적 증거가 임상적으로 타당한지 조절합니다. 결과적으로 연속 심각도 값과 함께 evidential regression으로 aleatoric·epistemic 불확실성까지 산출하는 것이 핵심 기여입니다.

- **Technical Challenges**: 세 모달 정보를 단순 결합이 아니라 계층적 상호작용으로 안정적으로 융합하는 것이 가장 큰 난관이었습니다. TMF-RSE는 (1) VLM 기반 semantic gating, (2) 마스크의 구조적 신뢰도로 attention entropy를 조절하는 structural prior modulation, (3) 3단계 hierarchical fusion(후반부에 모든 모달을 조건화)로 해결했습니다. 또한 NIG(Normal-Inverse-Gamma) 파라미터를 학습하는 evidential regression로 representation-level과 output-level에서 불확실성을 상보적으로 모델링하고, 과신을 막기 위한 정규화 손실을 추가했습니다.

- **Empirical Impact**: Per-COVID-19 CT와 RALO에서 transformer 기반 최신 baseline을 능가하며 MAE 4.02, Pearson correlation 0.9629(Per-COVID-19 validation)를 기록했고 RALO에서는 MAE 0.339 / PC 0.973의 결과를 보고했습니다. 추가로 uncertainty sparsification 분석에서 높은 불확실성 샘플을 제거할수록 MAE가 감소해, 불확실성이 ‘어려운 케이스’를 잘 식별한다는 점을 실증했습니다. 임상 현장에서 연속 심각도와 함께 신뢰도 지표를 제공할 수 있어, 의료 영상 회귀 모델을 의사결정 보조로 확장하는 데 의미가 큽니다.



### Generalized Synthetic Image Detection with Enhanced RGB-Noise Representation Learning (https://arxiv.org/abs/2607.06354)
- **Prior Approaches**: 기존 합성 이미지 탐지는 단일 도메인(공간/주파수) 표현에 의존하는 경우가 많아, 학습하지 않은 생성 모델이나 JPEG·블러 같은 현실적 열화에서 성능이 쉽게 무너진다. 또한 BCE 기반 이진 분류는 결정 경계가 취약하고, contrastive learning을 쓰더라도 모든 쌍을 동일하게 다뤄 hard sample의 기여가 희석되는 문제가 지적돼 왔다. 다양한 특징을 단순 concat/add로 합치면 서로 간섭해 시너지보다 방해가 커질 수 있다는 한계도 함께 제시된다.

- **Core Contribution**: 이 논문은 RGB-Noise 듀얼 브랜치를 통해 합성 이미지의 “가짜가 남기는 잡음 흔적”을 더 강건하게 잡아내는 RNSIDNet을 제안한다. RGB 쪽은 attention-refined CLIP 백본이 전역 의미를 뽑고, noise 쪽은 Bayar convolution이 고주파 잔차를 학습한 뒤 FiLM으로 RGB가 noise를 동적으로 조절해 표현 간 상호강화를 만든다. 여기에 Hard Sample-aware Contrastive Learning(HSCL)을 더해, 결정경계 근처의 가장 혼동되는 샘플들에 더 큰 벌점을 주어 판별 여백을 재구성한다.

- **Technical Challenges**: 핵심 난점은 (1) semantic(내용)과 noise(미세 아티팩트) 간 복잡한 의존성을 단순 결합이 아닌 방식으로 결속하는 것과 (2) contrastive 최적화에서 easy pair에 의해 학습이 왜곡되지 않게 hard sample에 초점을 맞추는 것이다. 저자들은 FiLM 기반 동적 모듈레이션(스케일/바이어스+게이팅)으로 noise 브랜치를 RGB 특징에 조건부로 재조정하고, HSCL에서 미니배치 내 hard negative를 top-K로 선별해 각도 기반 유사도 손실에 가중 페널티를 부여하는 방식으로 해결한다. 더 나아가 AMSID는 픽셀 정렬된 다중 소스(확산/ GAN) 재생성을 제공해 내용 편향을 줄이도록 설계된다.

- **Empirical Impact**: 8개 공개 벤치마크에 대한 광범위한 실험에서 RNSIDNet은 state-of-the-art 성능을 달성하며, cross-model 위조 탐지에서 특히 일반화 능력과 견고함이 강화됐다고 보고한다. 또한 작은 데이터나 제한된 학습 파라미터 조건에서도 큰 모델 수준에 근접하는 효율성을 보여 실용성 면에서도 의미가 크다. 코드와 데이터 공개 계획이 포함돼, 후속 연구에서 재현 및 확장이 기대되는 흐름이다.



### Bridging Diffusion Pruning and Step Distillation with Teacher-Aligned Repair (https://arxiv.org/abs/2607.06335)
- **Prior Approaches**: 확산 모델의 비용은 큰 denoising 네트워크와 반복적인 샘플링 단계(NFE)에서 오는데, 기존 압축은 이를 따로 공략해 왔다. Pruning은 네트워크 크기를 줄이지만 대부분 긴 post-pruning retraining으로 다단계(많은 step) 샘플러 성능을 다시 회복해야 한다. Step distillation은 NFE를 줄이지만, 보통 teacher를 “잘 따라갈” 상태의 student를 전제로 해 pruning 이후 초기화 불일치가 문제로 남는다.

- **Core Contribution**: 이 논문은 “pruning 뒤에 필요한 retraining을 step distillation로 대체할 수 있는가”를 정면으로 묻고, 직접 대체는 실패함을 보인다(EDM2-XS teacher를 pruning한 뒤 SiDA를 pruned checkpoint에서 시작하면 샘플이 망가짐). 원인으로 pruning이 teacher-학생 간 local denoising field를 손상시켜 distillation loss에 의미 있는 gradient가 전달되지 못함을 지적한다. 이를 해결하기 위해 teacher-alignment repair라는 짧은 브리지를 추가해, pruning으로 줄인 compact 생성기를 one-step distillation에 적합한 영역으로 “맞춘 뒤” 학습을 이어가게 한다.

- **Technical Challenges**: 핵심 기술적 난제는 pruning으로 인해 student가 teacher와 noisy latent 구간에서 정렬되지 않아 one-step distillation이 붕괴한다는 점이다. 저자들은 이를 완화하기 위해 실이미지 latent에서 teacher 출력과의 불일치를 줄이도록 짧게 보정(bridge)하고, 그 다음 one-step distillation(SiDA 등)을 수행한다. 또한 어떤 블록/채널을 제거해도 distillation 가능한 정렬을 유지하도록 teacher-aware pruning 중요도(블록 제거 시 teacher-alignment error 증가)를 기반으로 구조를 선택해 과도한 가지치기를 예방한다.

- **Empirical Impact**: ImageNet-512에서 EDM2-XS baseline은 124.713M 파라미터로 63 NFE를 쓰며 FID 3.53을 달성한다. 제안한 방식으로 20% pruning한 one-step generator는 98.826M 파라미터, 1 NFE로 FID 3.12까지 개선했다. 30% pruning에서도 88.029M 파라미터·1 NFE를 유지하며 FID 4.26을 기록해, post-pruning recovery를 many-step retraining이 아닌 compact one-step 생성으로 전환할 수 있음을 실증한다.



### Synthetic-to-Real Translation for Class-Agnostic Motion Prediction (https://arxiv.org/abs/2607.06319)
- **Prior Approaches**: 기존 motion prediction은 BEV로 변환한 뒤 ground plane의 2D displacement를 cell-wise로 회귀하는 방식이 많았고, 이때 합성-실세계 간 domain shift에 취약하다는 문제가 제기돼 왔습니다. 특히 teacher-student 기반 synthetic-to-real translation을 motion prediction에 그대로 적용하면 pseudo-label 노이즈 때문에 오히려 예측 오차가 크게 악화될 수 있습니다.

- **Core Contribution**: 이 논문은 synthetic 데이터에서 real 데이터로 motion 지식을 옮기는 Synthetic-to-Real Motion Prediction (SRMP)이라는 연구 과제를 정식으로 다루고, naive motion regression이 pseudo-label을 흔드는 원인을 motion jitter와 object-level inconsistency로 명확히 짚습니다. 이를 해결하기 위해 SR-Motion 프레임워크를 제안하며, objectness-aware motion prediction과 objectness-aided motion enhancement로 더 신뢰도 높은 지식 전달을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난제는 실측 real 4D LiDAR에서는 object 어노테이션이 없어 object 단위 일관성을 직접 제약하기 어렵고, 따라서 pseudo motion label의 잡음을 안정적으로 다뤄야 한다는 점입니다. 저자들은 motion과 objectness를 동시에 학습하도록 objectness-aware branch(각 cell의 centroid relative offset 회귀)를 넣고, OAME에서 centroid-aware clustering, COF(cluster outlier filtering), SCS(spatial consistency smoothing)로 pseudo-label의 outlier와 jitter를 억제해 object 단위의 응집성을 강제합니다.

- **Empirical Impact**: 또한 SRMP용으로 Motion4D라는 대규모 합성 4D LiDAR 데이터셋(1,370개 4D 시퀀스, 총 124K 프레임, 다양한 동적 패턴)을 물리 기반 시뮬레이션 파이프라인으로 구축해 실험 기반을 마련했습니다. 실험에서는 단순 도메인 적응 대비 domain gap을 효과적으로 메우며 real 장면에서 더 우수한 성능을 보였고, 특히 teacher가 만든 noisy pseudo-label의 부정적 영향을 objectness 기반 보정이 유의미하게 완화하는 것으로 정리됩니다.



### Token-Based Dual-view Fusion and Adaptation of Large Vision Models for Breast Cancer Classification (https://arxiv.org/abs/2607.06309)
- **Prior Approaches**: 유방촬영(mammography) CC와 MLO 두 시점을 통합하려는 시도는 주로 early/intermediate/late fusion, 또는 cross-attention 기반 상호작용으로 발전해 왔다. 다만 대부분은 단일 레이어에서의 결합이나 residual 형태의 직접 더하기로 구현돼, 뷰-특이 정보와 뷰-공유 정보가 얽히거나 cross-view 의존성이 깊은 층까지 일관되게 유지되기 어렵다는 한계가 있었다. 또한 prompt 기반 적응은 주로 단일 이미지에 맞춰져 다중 뷰 간 구조적 상호작용을 충분히 담지 못했다는 지적이 있었다.

- **Core Contribution**: 이 논문은 frozen vision transformer 백본 위에서 “token-centric dual-view learning”을 제안하며, CC–MLO 사이 상호작용을 fusion token의 구조적 토큰-커뮤니케이션으로 재구성한다. Stage 1에서는 deep shared prompt learning으로 CC와 MLO에 동일한 프롬프트를 적용해 파라미터 효율적으로 표현을 정렬하고, Stage 2에서는 bidirectional cross-attention로 생성된 fusion token을 시퀀스에 삽입해 다음 레이어에서 다시 정교화되도록 한다. 더 나아가 fusion을 단일 레이어가 아니라 여러 transformer depth에 반복 삽입해 계층적(계단식)으로 보완 정보를 전파한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 다중 뷰 의존성을 뷰-특이 표현과 분리해 명시적으로 지속시키고, (2) 단순 feature-level 결합이 아니라 토큰 수준에서 재사용 가능한 중간 전달체를 만드는 것이다. 이를 위해 cross-attention 출력에서 mean pooling으로 compact fusion token을 만들고, 이를 각 뷰의 토큰 시퀀스에 삽입한 뒤 후속 transformer 레이어가 이를 컨텍스트로 처리하게 설계했다. 또한 cross-view fusion 모듈을 여러 깊이에 배치해 상호작용이 계층적으로 누적되도록 했으며, 학습은 frozen 백본을 유지한 채 프롬프트(1단계)와 fusion 모듈(2단계)만 업데이트하는 방식으로 파라미터 효율을 확보했다.

- **Empirical Impact**: VinDr-Mammo와 CMMD 실험에서 제안 프레임워크는 linear probing, prompt-only adaptation, 기존 conventional fusion baseline 대비 일관된 성능 향상을 보였다. VinDr-Mammo BI-RADS 분류에서 F1-score 50.40%, AUC 0.8090을 달성했고, binary 설정에서 dual-view fusion baseline 대비 AUC가 0.10p 개선됐다. ablation에서도 token 기반 fusion과 multi-depth 상호작용 설계가 효과적임이 확인돼, CC–MLO 보완 정보를 더 잘 활용하는 다중 뷰 분류 접근으로 의미가 있다.



### Visual graphs for image classification: does the structure affect performance? (https://arxiv.org/abs/2607.06295)
- **Prior Approaches**: CNN과 ViT는 이미지의 구조적 장점을 활용하지만, CNN은 장면 내 객체 관계와 문맥 반영이 제한적이고 ViT는 패치 기반이라 공간 레이아웃 보존·정밀 국소화에 약점이 있다. GNN은 비정규 그래프로 이미지 내부 연결을 모델링할 수 있으나, 시각 태스크에서 그래프 구성(노드/엣지 정의)과 그 영향이 부분적으로만 연구돼 왔다. 기존 연구들은 주로 제한된 관점에서 비교하거나, 그래프를 만든 뒤 GCN 구조를 고정하지 않는 경우가 많았다.

- **Core Contribution**: 이 논문은 고정된 3-layer GCN(처리 구조)을 유지한 채, 이미지→그래프 변환의 핵심 설계변수(노드 추출 방식, 노드 임베딩, 엣지 구성·필터링)를 체계적으로 비교한다. 특히 GCN 성능이 그래프 토폴로지와 구조적 복잡도에 의해 어떻게 달라지는지 실험적으로 규명하고, 그래프를 활용하기 전 계산 단계(사전 그래프 생성)가 결과를 좌우한다는 방법론적 관점을 제시한다. 결론적으로 “풍부한 노드 특징은 더 희소하고 효율적인 그래프에서도 잘 동작한다”는 가이드를 도출한다.

- **Technical Challenges**: 그래프를 만들 때 노드/엣지를 어떻게 정의하느냐에 따라 이후 모든 처리 단계가 달라져, 구성 선택이 성능을 크게 흔드는 문제가 있다. 저자들은 최대 노드 수를 50으로 고정하고(비교 가능성 확보), 노드 위치는 grid·superpixel(SLIC)·interest point(Harris)로 나누며, 노드 임베딩은 ViT/CNN/SIFT-like 디스크립터와 좌표를 결합해 통제했다. 엣지는 K-NN(K=6)으로 기본 골격을 만든 뒤 평균 거리 기반 가지치기, Locality Sensitive Pruning(LSP), 최소 신장 트리 기반 MinCONN으로 희소화해 구조 복잡도를 정량 비교했다.

- **Empirical Impact**: fashion-MNIST에서 최고 성능은 interest points 노드 + ViT features + MinCONN 조합으로 test accuracy 0.8681, F1 0.8672를 기록했다. 전반적으로 ViT-based features가 CNN 대비 약 1.4%, SIFT 대비 약 4.0%p 가량 높은 정확도를 보여 “노드 특징 품질이 기준선”임을 시사한다. 동시에 최상위 모델 9개 중 6개가 MinCONN처럼 희소한 구조를 선택했고, 8/9가 어떤 형태로든 sparsification을 활용해 연결 수 자체보다 구조 품질과 단순성이 중요함을 실증적으로 보여줬다.



### AlayaWorld: Long-Horizon and Playable Video World Generation (https://arxiv.org/abs/2607.06291)
Comments:
          Authors are listed alphabetically by the first name and their role. See the contribution section for details

- **Prior Approaches**: 전통적 게임 월드 제작은 자산·애니메이션·상호작용 규칙을 사전에 일일이 작성하는 노동집약적 파이프라인에 의존해, 배포 후 수정과 확장에 비용이 크게 든다는 한계가 있었다. 최근에는 video world model이 “생성 모델이 미래 관측을 autoregressively 합성”하는 방식으로 전환되었지만, (1) 조작 자유도 제어, (2) 시공간 일관성, (3) 긴 롤아웃 안정성, (4) 실시간 런타임 지연 문제가 여전히 핵심 도전으로 남아 있다. 기존 접근은 카메라를 추론된 조건으로 주입하거나(중간 latent/attention steering, 카메라 조건 모듈), 내부 아키텍처에 지오메트리를 강하게 바이어스하거나, 혹은 렌더링 프록시를 조건으로 쓰는 방식으로 나뉘어 성능-복잡도 트레이드오프가 발생한다.

- **Core Contribution**: AlayaWorld는 인터랙티브 생성 월드를 만들기 위한 full-stack 오픈소스 프레임워크로, 데이터 준비부터 학습·추론 가속·배포까지 모듈형으로 통합해 “재현 가능한 실시간 플레이어블 월드 생성”을 목표로 한다. 모델 측면에서는 autoregressive DiT에 prompt-switching(청크 경계 텍스트 교체), AdaLN-style camera-control(가벼운 카메라 조건 모듈), 3D cache 기반 렌더링 증거, history-compression, error bank, few-step distillation을 결합해 제어·일관성·안정성·런타임을 동시 타깃한다.

- **Technical Challenges**: 주된 technical challenge는 (1) 플레이어가 내리는 자유로운 행동/시점 변화가 생성에 정확히 반영되도록 하는 제어성과, (2) 되돌아오는 장면에서의 place identity 수준 일관성을 유지하는 메모리 설계, (3) 자기 예측이 만든 오류가 시간이 지날수록 누적되는 드리프트를 장기적으로 억제하는 학습 안정성, (4) 시각 지연(denoising 단계)과 의미 지연(조건 갱신)을 모두 낮추는 런타임 설계였다. AlayaWorld는 3D cache를 플레이어 카메라 궤적에 따라 재투영·렌더해 공간 기반 “구체적 증거”를 제공하고, 동시에 AdaLN-style modulation으로 백본에 가벼운 궤적 인지를 주입하며, 안정성은 드리프트된 히스토리와 error bank 샘플을 학습에 반영(메모리 조건과 타깃 모두에 구조화된 섭동)하는 방식으로 접근한다.

- **Empirical Impact**: 논문은 AlayaWorld의 complete technical details와 실험 결과, 그리고 전체 코드베이스를 mid-July에 공개하겠다고 밝히며, 재현 파이프라인·레퍼런스 구현·평가 도구·문서까지 함께 제공하는 “실용적 연구 기반”을 조성하는 데 의미를 둔다. 또한 게임을 넘어 embodied intelligence를 포함한 인터랙티브 애플리케이션으로 확장할 수 있는 real-time generative world 모델 생태계에, 즉시 실험 가능한 통합 프레임워크를 제공한다는 점에서 임팩트가 기대된다.



### Straight-Path Flow Matching for Incomplete Multi-View Clustering (https://arxiv.org/abs/2607.06281)
Comments:
          Accepted to ECCV 2026. 28 pages, 6 figures, 4 tables

- **Prior Approaches**: Incomplete Multi-View Clustering은 일부 시점/모달리티(view)가 누락된 멀티모달 데이터를 클러스터링하는 문제다. 최근 end-to-end generative 접근은 diffusion model을 이용해 잡음에서 데이터로의 노이즈-투-데이터 경로를 따라 누락된 view를 복원하지만, 클러스터링 목적함수와의 정렬이 명시적으로 설계되지는 않는다.

- **Core Contribution**: 본 논문은 end-to-end generative IMVC에서 확률 경로(probability path) 설계를 다시 본다. diffusion 대신 관측 view와 누락 view 사이의 선형 보간 경로를 쓰는 flow matching으로 확률 흐름(probability flows)을 구성해, 클러스터링 목적에 더 적합한 view completion을 달성한다.

- **Technical Challenges**: 핵심 기술 과제는 확률적 denoising으로 복원하던 방식이 유한 스텝에서 클러스터 일관성을 깨뜨릴 수 있다는 점이다. 논문은 결정론적 ODE 흐름이 클래스 조건부 분포를 존중하는 transport 메커니즘 관점에서 diffusion의 확률적 궤적보다 clustering 목표와 더 잘 맞는다는 정식 분석을 제시하고, 이를 바탕으로 straight-path flow-matching 기반 IMVC 아키텍처에 cluster-level 및 entropy 기반 alignment를 통합해 교차 view 클러스터링 일관성을 강제한다.

- **Empirical Impact**: 표준 IMVC 벤치마크에서 제안 프레임워크가 새로운 state-of-the-art 성능을 기록하며, 누락 view 복원과 클러스터링이 함께 개선됨을 보여준다. diffusion 기반 generative 접근 대비 clustering에 더 직접적으로 정렬된 probability path 설계가 효과적임을 실증적으로 입증했다.



### MAC-XA: Multi-view Anatomy-Correspondence Fusion for Coronary Stenosis Reporting from X-ray Angiography (https://arxiv.org/abs/2607.06268)
Comments:
          Preprint

- **Prior Approaches**: 관상(冠狀) X-ray 혈관조영에서 다중 뷰 추론은 시야 투영에 따른 기하학적 왜곡 때문에 본질적으로 ‘투영-의존’ 문제지만, 자동 리포트 생성은 상대적으로 덜 다뤄져 왔다. 기존 다중 뷰 fusion·학습은 시각적 상관관계를 end-task 손실에서 암묵적으로 맞추는 방식이 많아, 강한 분지 중첩·foreshortening 같은 상황에서 해부학적 대응이 검증되지 않는 한계가 있다.

- **Core Contribution**: 논문은 관상 협착(stenosis) 리포팅을 ‘alignment-constrained aggregation(정렬 제약 기반 증거 집계)’로 재정의해, 교차 뷰에서 해부학적 대응이 명시적으로 필요함을 강조한다. 또한 실데이터에서는 alignment supervision이 관측 불가능하다는 문제를 피하기 위해, 합성 데이터로 geometry-유도 patch-level 대응(일치) 감독을 제공하는 학습 전략을 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 뷰 간 정렬이 실 영상에서 직접 관측되지 않아 감독 신호가 없다는 점이며, 이를 해결하려고 CTA 기반 합성 DRR을 만들고 SYNTAX segment 정의로 구조화 리포트까지 자동 생성한다. 이어 Pose-aware embedding과 RAD-DINO patch token을 이용해 cross-view correspondence matrices를 학습하고, 주(main) 뷰 좌표공간으로 auxiliary 뷰를 명시적으로 정렬한 뒤에만 증거를 집계하도록 evidence aggregation을 제약한다.

- **Empirical Impact**: 합성 데이터에서 correspondence 품질과 structured stenosis 보고가 단일 뷰 및 기존 다중 뷰 fusion 대비 개선됐고, 대응이 ‘검사 가능한 형태’로 학습될수록 구조적 정확성이 높아짐을 보였다. 특히 합성으로 학습한 뒤 real angiograms에 zero-shot으로 적용해도 임상가 검토 기준으로 교차 뷰 국소화 정확도가 높았으며, 구조화 리포트 점수 역시 향상되어 다중 뷰 정렬 제약 설계의 일반화 가능성을 입증했다.



### VendorBench-100: A Unified Cross-Paradigm Benchmark for Deepfake Image Detection (https://arxiv.org/abs/2607.06254)
Comments:
          22 pages, 10 figures, 3 tables. Code and data: this https URL

- **Prior Approaches**: 딥페이크/AI 이미지 탐지는 상용 API, zero-shot 비전-언어모델(vision-language models), 오픈소스 탐지기라는 세 갈래로 발전했지만, 서로 같은 기준에서 비교된 적이 거의 없습니다. 기존 평가는 벤더 내부 테스트나 단일 유형 데이터에 치우쳐, 실제 운영에서 마주치는 다양한 위조 방식과 품질 열화(compression/리사이즈)를 공정하게 반영하지 못한다는 한계가 있었습니다. 또한 정확도 같은 단일 지표는 데이터 불균형(예: 대부분이 fake)에 취약해 과대평가될 수 있습니다.

- **Core Contribution**: 본 논문은 VendorBench-100을 제안하며, 서로 다른 3패러다임의 36개 모델을 단일 100장(79 fake/21 real) 고정 코퍼스와 단일 출력 스키마, 공통 평가 프레임워크로 함께 비교합니다. 특히 난이도를 키우기 위해 8개 엣지케이스 패밀리(얼굴 스왑 스미어, near-duplicate 스왑, letterboxed text-to-video 스틸, AI 사진 편집, 불명확 출처 조작 등)를 체계적으로 구성해 “쉽게 구분되는” 상황을 피했습니다. 순위는 Matthews correlation coefficient(MCC)를 중심으로 매기고, 임계값에 덜 민감한 ROC-AUC를 보조로 사용합니다.

- **Technical Challenges**: 주요 기술적 난제는 모델마다 출력 형식/판정 방식이 달라 공정 비교가 어려운 점이었습니다. 이를 위해 모든 모델을 FAKE/REAL의 hard label, P(fake)∈[0,1] 형태의 confidence, 판정 성공 여부(abstention 포함)로 정규화하는 공통 레코드를 만들고, 파일명/메타데이터 유출을 막는 anti-leakage 프로토콜을 적용했습니다. 또한 클래스 불균형에서 accuracy가 쉽게 “기만”될 수 있어, MCC와 ROC-AUC를 함께 보고 score 분리력과 임계값 의사결정 품질의 차이를 진단하도록 설계했습니다.

- **Empirical Impact**: 실험 결과로는 상용 API들이 중앙값 성능에서 가장 강했지만, 일부 오픈소스 모델은 최고 수준의 비전 LLM과 경쟁하거나 더 잘한 사례도 나타났습니다. 더 중요한 관찰로, ROC-AUC(랭킹/분리력)가 높아도 MCC(기본 임계값에서의 신뢰 가능한 판정)가 낮게 나오는 불일치가 세 패러다임 전반에서 반복되었습니다. 즉 “점수 순위가 좋다”는 것만으로 “기본 설정에서 실제로 믿을 만한 판정”이 보장되지 않으며, 이 메트릭 불일치 자체가 벤치마크의 핵심 발견으로 제시됩니다. 논문은 재현을 위해 평가 프레임워크와 결과를 공개해 후속 연구가 동일 프로토콜로 확장할 수 있게 했습니다.



### PhyMRI-SR: Toward Physics-Aware MRI Image Super-Resolution (https://arxiv.org/abs/2607.06238)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 MRI 초해상도는 LR 영상을 HR로 고정된 스케일의 결정적(deterministic) 매핑으로 다루는 경우가 많습니다. 전통적 방법은 주로 handcrafted prior와 정밀한 열화 모델에 의존했지만, 잡음·모션·저필드 복잡도에 취약했습니다.
최근 딥러닝(CNN/Transformer, 생성모델)도 강한 성능을 보였으나, 대부분이 해상도- SNR의 물리적 트레이드오프와 획득 시스템을 명시적으로 고려하지 못했습니다.

- **Core Contribution**: 이 논문은 MRI 초해상도를 “고정 입력→고정 출력”이 아니라, 해상도와 SNR이 함께 달라지는 획득 물리 조건을 복원하는 physics-aware reconstruction 문제로 재정의합니다. 특히 최적의 resolution-SNR 구성(가변 해상도)을 찾아 그 구성에 맞게 초해상도 결과를 생성하도록 설계해, resolution을 동적으로 만듭니다.
또한 2D Gaussian Splatting(2D GS)을 MRI에 맞게 해상도-불변의 좌표 기반 렌더링으로 적응해, resolution-heterogeneous 입력에도 유연하게 대응합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 자연영상과 다른 해부학 구조/획득 지문을 반영한 primitive 초기화, (2) MRI 신호가 proton density와 relaxation에 의해 결정된다는 biophysical plausibility 확보, (3) 다양한 해상도에 대한 paired 데이터 부족을 동시에 해결하는 것입니다.
이를 위해 해부학 구조 prior(세그멘테이션 기반 밀도 가중 초기화)와 imaging system prior(코바리언스 딕셔너리로 시스템 특성 제약)를 결합하고, 강도는 Bloch/Bloch-유도 신호식에 따라 rho와 effective relaxation rate(R2)를 예측해 물리적으로 합성합니다.
마지막으로 시뮬레이션-실데이터 격차를 줄이기 위해 meta-learning(episodic meta-training)을 도입해 적은 실데이터로도 빠르게 적응하도록 했습니다.

- **Empirical Impact**: 동적 해상도(dynamic-resolution) 데이터셋과 FastMRI 벤치마크에서 정량·정성 모두 기존 방법을 능가하는 state-of-the-art 성능을 보였다고 보고합니다. resolution이 입력마다 달라지는 실제 상황에서도 일관된 구조 선명도와 물리적으로 그럴듯한 대조(contrast)를 유지한다는 점이 강조됩니다.
임상 적용 관점에서, “최적 획득 조건을 암시적으로 탐색한 뒤 재구성”하는 접근이 실사용 제약(스캔 시간·하드웨어 비용) 하에서 고품질 MRI 접근성을 높일 잠재력이 있다는 평가입니다.



### WING: A Window-Prior-Based Generative Network with Gated Inception for Cross-Modality CT Synthesis (https://arxiv.org/abs/2607.06234)
- **Prior Approaches**: MRI/CBCT로부터 CT를 합성(sCT)하는 연구는 주로 3D image-to-image translation으로 접근해 U-Net 계열부터 GAN, diffusion, Transformer까지 다양한 아키텍처를 시도해 왔다. 그러나 CT는 절대 HU 기반으로 -1000~3000처럼 동적 범위가 크고, 긴 꼬리(long-tailed) 분포로 인해 희소하지만 중요한 구조가 평균화되기 쉽다는 한계가 있었다. 일부는 기하학적 가이드를 추가해 디테일을 복구하려 했지만, 외부 segmentation 라벨 의존 등 임상 적용성 제약이 남았다.

- **Core Contribution**: 논문은 CT intensity를 그대로 전 범위 회귀하기보다, 여러 windowed representation으로 회귀 목표를 바꾸는 “window-prior” 관점을 제안한다. CT의 구조-결정성(structure determinism)과 window-separability 가정에 따라, 창(window)별 표현은 분포가 더 매끈해지고 다시 full-range CT로 구조적으로 재구성할 수 있음을 강조한다. 이를 바탕으로 WING( WINdow-prior-based Generative network)을 제시하며, 단일 컴팩트 모델로 MRI-to-CT와 CBCT-to-CT 모두에서 multi-anatomy 합성을 지원한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 긴 꼬리 HU를 평균화하지 않으면서 (2) 예측된 창들을 신뢰도 있게 결합해 (3) 임상적으로 그럴듯한(full-range) CT를 만드는 것이다. WING은 Gated Inception Generator(GIG)로 lung/soft tissue/bone 등 다중 창 예측을 만들고, Fuse-and-Refine Transformer(FRT)가 soft fusion으로 거친 결합을 만든 뒤 residual로 디테일을 정제한다. 또한 window 조건과 full-range 출력을 함께 보정하도록 PatchGAN 기반 joint adversarial objective를 설계해 window-conditioned realism을 강화했다.

- **Empirical Impact**: SynthRAD2025에서 MRI-to-CT, CBCT-to-CT 모두 최신 성능을 달성했으며, MedNeXt 베이스라인 대비 MAE/MS-SSIM/PSNR/DICE에서 일관된 개선을 보였다. 특히 lung과 bone처럼 긴 꼬리 영역에 대한 per-window MAE가 크게 감소해, 전역 통계만 맞추는 방식이 아니라 이질적 감쇠 구간 전반의 충실도를 높였다는 점을 뒷받침한다. ablation에서도 GIB, FRT, JDisc가 각각 단계적으로 성능을 끌어올렸고, FRT는 파라미터 오버헤드가 적은 상태에서도 융합·정제의 필요성을 입증했다.



### EeveeDark: A Binary Neural Framework for Low-Light Video Enhancement via Event-Guided Sensor-Level Fusion (https://arxiv.org/abs/2607.06217)
- **Prior Approaches**: 기존 저조도 비디오 향상은 프레임 정렬(예: optical flow)이나 recurrent/shift 기반 설계로 시간 일관성을 노렸지만, 대체로 연산량이 커서 임베디드 적용이 어렵다는 문제가 남았다. 이벤트 카메라를 쓰는 방법도 존재하지만, optical-flow/attention 정렬처럼 계산이 무거운 구성으로 가거나 RGB(또는 처리된 입력)에 의존해 RAW 센서 수준의 색/톤 복원이 약해지는 한계가 있었다. 또한 BNN은 효율은 좋지만 비디오 향상에서 세밀한 공간 디테일과 시간적 일관성을 함께 유지하기가 까다로웠다.

- **Core Contribution**: 이 논문은 EeveeDark를 제안하며, RAW의 공간 풍부함과 이벤트 스트림의 시간 정밀함을 하나의 BNN 기반 end-to-end 학습 시스템에서 동시에 통합한다. 특히 이벤트를 이용한 spatiotemporal refinement을 경량화해 성능-효율 트레이드오프를 개선하는 것을 목표로 한다. 결과적으로 기존 BNN 기반 기법보다 품질을 끌어올리면서도 full-precision 모델 대비 훨씬 낮은 연산 비용을 달성한다.

- **Technical Challenges**: 핵심 난제는 (1) 비동기 이벤트를 RAW 비디오와 효과적으로 융합하되 (2) 1-bit 양자화에 따른 품질 저하를 억제하고 (3) 계산 예산 안에서 시간 일관성을 만드는 것이다. 이를 위해 EeveeDark는 RAW용과 이벤트용 각각의 modality-specific binary encoder를 두고, RAW-이벤트 feature를 경량 fusion block으로 결합한다. 여기에 event-guided skip gating(EGSG)으로 시간적으로 변하는 영역을 이벤트 통계로 선택 정제하고, cyclic temporal shift 및 재귀 임베딩을 통해 동적 장면의 시간 전파를 안정화한다.

- **Empirical Impact**: LLRVD(합성 페어)와 HUE(실세계 이벤트) 및 SDE/SDSD(RGB 벤치마크)에서 EeveeDark는 기존 BNN 대비 PSNR/구조 유사도 등에서 우수하고, full-precision 대비에도 경쟁력 있는 시각 품질을 낮은 비용으로 제공한다. 예를 들어 연산 추정에서 ShiftNet/FloRNN급 full-precision은 수 초~수십 초 수준인 반면, EeveeDark는 약 588ms/프레임(추정)으로 현저히 효율적이다. 또한 저조도 향상 결과를 object detection, monocular depth estimation, visual SLAM 같은 로봇 하류 작업에 적용해 mAP/깊이 구조/추적 안정성이 개선되며, 저조도 로보틱스 인식에 직접적인 의미가 있음을 보여준다.



### MoWorld: A Flash World Mod (https://arxiv.org/abs/2607.06216)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 World Model은 주로 대규모 영상 코퍼스와 대형 모델 스케일링에 의존해 왔지만, 실제 환경에선 실시간(고프레임레이트) 추론과 배포 비용이 발목을 잡는 경우가 많다. 또한 데이터가 카메라 기하(시점·궤적)와 제어 신호를 충분히 정렬해 주지 못하면, 생성은 그럴듯해도 camera controllability와 장기 일관성이 흔들릴 수 있다.

- **Core Contribution**: MoWorld는 데이터 생성부터 pre-training, distillation, 효율적 inference까지 end-to-end로 설계한 Flash World Model로, 최대 50 FPS 수준의 실시간 상호작용을 목표로 한다. 특히 MoWorld는 대형 video corpora 중심 접근과 달리, 3D-native 데이터 엔진을 기반으로 기하적으로 일관된 학습 데이터를 구축해 World Model의 실용성을 끌어올린다.

- **Technical Challenges**: 핵심 난제는 (1) 기하 정렬이 필요한 고품질 데이터를 대규모로 효율 생산하는 것, (2) long-horizon 제어를 안정적으로 학습하면서 학습·추론 비용을 줄이는 것, (3) NPU 같은 저비용 하드웨어에서 주의(attention) 메모리 병목을 완화하며 고FPS를 달성하는 것이다. MoWorld는 기하 aware data engine, cross-frame curriculum pre-training, diffusion denoising-step distillation(몇-step AR로 압축), 그리고 mixed-precision/quantization/병렬화(Sequence Parallelism·Ulysses/USP 계열) 및 NPU-native 실행 최적화를 결합해 이를 해결한다.

- **Empirical Impact**: 평가에서 MoWorld는 이미지-to-video 품질 및 camera-controllable 생성 벤치마크 전반에서 선도 성능을 보였고, 평균 inference cost가 기존 World Models 대비 30–50% 수준으로 낮다고 보고한다. 또한 Neural Processing Unit(NPU) 기반으로도 별도 고급 GPU 없이 최대 50 FPS 실시간 상호작용을 구현해, 대규모 실세계 배치에 필요한 비용·지연 문제를 실증적으로 완화했다.



### Structured-Condensed Prompt Tuning in Vision-Language Models for Fine-grained Image Recognition (https://arxiv.org/abs/2607.06185)
- **Prior Approaches**: CLIP 기반 zero-shot은 라벨을 직접 학습하지 않아도 되지만, fine-grained에서 미세한 구분을 잡기엔 한계가 있습니다. 이를 보완하려고 CoOp 계열 textual prompt tuning이 등장했지만, 대부분 class label을 서로 독립된 토큰처럼 다뤄 inter-class 관계(계층·상관)를 충분히 반영하지 못합니다. 특히 few-shot/베이스-노벨 설정에서 base 클래스에 치우치거나 미세 구분에 필요한 의미 구조가 흐려지는 문제가 반복됩니다.

- **Core Contribution**: 이 논문은 Structured-Condensed Prompt Tuning (SCPT)를 제안해 fine-grained에서 라벨 의미의 구조를 prompt 학습에 직접 반영합니다. 핵심은 Semantic Relation Encoding (SRE)으로 클래스 간 semantic topology를 구조화해 표현하고, Semantic Condensation loss (ScLoss)로 중복·잡음 신호를 억제해 분별력 있는 의미에 집중시키는 것입니다. SRE와 ScLoss를 함께 써서 semantic alignment와 fine-grained discrimination을 동시에 끌어올립니다.

- **Technical Challenges**: 문제는 (1) 클래스 간 관계를 반영하되 prompt 길이 제약 안에서 효율적으로 구조를 담아내야 하고, (2) 관계 신호가 오히려 불필요한 간섭으로 작동하지 않게 해야 한다는 점입니다. 논문은 CLIP 텍스트 임베딩의 유사도 행렬에서 inter-class 관계를 계산한 뒤 signed random projection으로 compact한 SRE를 만들고, ScLoss에서는 handcrafted vs learnable 임베딩에 대해 SVD 기반 denoising과 adaptive한 singular value 선택으로 noise 구간을 잘라냅니다. 결과적으로 구조는 유지하면서 intra-class 중복 감독을 압축해 fine-grained 최적화가 안정화됩니다.

- **Empirical Impact**: ViT-B/16 백본으로 14개 fine-grained 벤치마크에서 평가한 결과, SCPT는 16-shot 조건 평균 정확도 76.70%로 TCP 등 기존 prompt tuning 대비 일관된 성능 우위를 보였습니다(옥스퍼드 페츠를 제외). base-to-novel 일반화에서도 SRE+ScLoss 조합이 14개 평균에서 TCP 대비 1.10% 개선을 만들었고, harmonic mean 기준 71.15%를 달성해 novel 클래스 일반화까지 강화했습니다. 전반적으로 harmonic mean에서 CoOp 계열의 base 과적합을 완화하며 전체 정확도를 최상위권으로 끌어올린 것이 의미 있습니다.



### Revisiting Scene Graph Generation from the Perspective of Detector-Conditioned Reachability (https://arxiv.org/abs/2607.06176)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 장면 그래프 생성(SGG)은 검출기 기반(detector-based)과 쿼리 기반(query-based)으로 크게 나뉘며, 두 방식은 서로 다른 추론 메커니즘 때문에 예측 성향이 달라진다. 다만 기존 평가는 Recall·mean-Recall 같은 집계 지표에 의존해 이런 차이를 “왜/어디서” 발생하는지 체계적으로 분석하지 못했다. 특히 검출기 기반 모델은 외부 검출기가 찾지 못하는 subject/object 인스턴스가 포함된 triplet에 취약한 ‘detector constraint’가 존재한다.

- **Core Contribution**: 논문은 detector-conditioned reachability 관점에서 예측 불일치를 통제 실험으로 정량화한다. 외부 object detector로 triplet을 Det-T(검출 가능)와 UDet-T(최소 한쪽이 미검출)로 나눈 뒤, 동일 프로토콜로 detector-based와 query-based의 행동 차이를 분해해 본다. 그 관찰에서 출발해 detector 기반의 탐지 조건을 유지하면서도 query 기반의 보완적 커버리지를 함께 쓰는 Dual-SGG(dual-query)를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 두 추론 경로의 점수를 단순 결합(예: 확률 ensembling)하면 스케일 불일치로 한 가지가 지배해 보완성이 약해질 수 있다는 점이다. 이를 해결하기 위해 단일 triplet decoder 안에 TD-Qs(top-down, entity pair selector(EPS)로 검출 조건을 부여)와 BU-Qs(bottom-up, 중심에서 바깥으로 확장하는 글로벌 탐색)를 동시에 넣고, self-attention mask로 정보 누수를 제어한다. 또한 TD-Qs는 검출기 도달 가능성을 만족하는 triplet에 대해서만 detector-conditioned supervision을 적용하도록 설계해 detector constraint를 정교하게 다룬다.

- **Empirical Impact**: Visual Genome, Open Images v6, GQA-200의 광범위한 실험에서 Dual-SGG는 전체 성능을 높이면서도 micro-DR(검출 가능 영역 성능)을 detector-based 수준으로 유지한다. 동시에 UDet-T에서의 micro-UDR(검출 불가능 영역 성능)이 유의미하게 개선되어, 두 계열의 ‘상보적 예측 성향’을 실제로 결합했음을 보여준다. 결과적으로 SGG에서 검출기 제약이 만드는 취약 영역을 더 잘 커버할 수 있는 end-to-end 통합 설계의 실효성을 입증했다.



### MobileWan: Closing the Quality Gap for Mobile Video Diffusion (https://arxiv.org/abs/2607.06173)
- **Prior Approaches**: 모바일 비디오 diffusion은 메모리·연산·지연 제약 때문에 0.4B~1.8B 같은 소형 모델에 머무르는 경우가 대부분이었다. 기존 접근은 UNet 계열부터 시작해 VAE 압축, 블록/헤드 프루닝, step distillation, 토큰 병합 등으로 모델을 줄이되, 결과적으로 서버 스케일 품질 격차가 누적됐다. DiT 기반에서도 재훈련·표현력 저하를 피하려는 혼합/적응이 시도됐지만, 전체 백본을 저메모리용 재귀 구조로 완전히 바꾸는 데는 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 “모바일 비디오는 작은 모델이 필요하다”는 가정을 정면으로 반박하며, 5B 파라미터급 video diffusion transformer를 상용 모바일에 올릴 수 있음을 보인다. 핵심은 Wan2.2-5B에서 출발해 recurrent reformulation과 structured compression을 결합하고, 학습된 attention head pruning·sampling-step distillation·메모리 최적화 VAE 디코딩까지 묶어 MobileWan을 구현한 점이다. 특히 모든 transformer block을 재귀 구조로 distillation해 chunk 단위 autoregressive 프로세스로 만들고, 추론 시 RNN처럼 동작하면서도 시간 일관성을 유지하도록 설계했다.

- **Technical Challenges**: 가장 큰 기술 난제는 self-attention의 토큰 수에 대한 비용(특히 피크 메모리)이 모바일에서 병목이 된다는 점이었다. 저자들은 chunk-wise로 생성하되 과거 청크 정보를 “상수 크기 상태(state)”로 누적하는 causal linear attention 기반의 RNN형 추론을 구성해, 길이가 늘어도 메모리가 일정하게 유지되도록 만들었다. 추가로 프루닝은 블록 단위보다 헤드 단위가 필요하다는 관찰에 따라, noise-biased sparsity objective로 이진 per-head gate를 end-to-end 최적화하는 learnable head pruning을 제안해 공격적으로도 품질 하락을 줄였다.

- **Empirical Impact**: MobileWan은 5초 길이의 480x832 비디오를 16 FPS로 생성하면서 end-to-end 지연 20초를 달성해 VBench 83.79를 기록, 모바일 비디오 생성에서 새로운 SOTA를 제시한다. 무엇보다 5B 스케일 모델을 “첫 상용 모바일 배치” 가능한 수준으로 내린 점이 의미가 크다. 이는 이후 모바일 video DiT 확장 연구에서 재귀적 변환(recurrent reformulation)과 학습형 압축(특히 head pruning)의 실용성을 강하게 뒷받침한다.



### High-Resolution Artwork Outpainting with Global Blueprint Guidance and Layout Contro (https://arxiv.org/abs/2607.06162)
Comments:
          Accepted at ECCV2026

- **Prior Approaches**: 기존 outpainting은 주로 고정 해상도에서 시작한 뒤, 캔버스를 점진적으로 확장하는 progressive window 방식으로 고해상도 결과를 만들었습니다. 하지만 이 과정은 전역 구성을 담당하는 신뢰할 만한 계획 단계가 없어 구조적 불안정과 오류 누적이 커지고, 고해상도로 갈수록 generation order에 민감해지는 문제가 있었습니다. 또한 대부분은 text prompt 중심이라 사용자가 원하는 위치에 객체를 정밀 배치하기 어렵고, 순차 패치 생성 특성 때문에 추론 지연이 크게 발생합니다.

- **Core Contribution**: 이 논문은 전역 blueprint(구조 계획도)를 먼저 만들고, 이를 공유 가이드로 삼아 고해상도 로컬 패치를 합성하는 global blueprint-guided two-stage diffusion 프레임워크를 제안합니다. Stage 1에서는 layout 조건(바운딩박스/설명)을 주입해 저해상도 전역 구조를 생성하고 전역 guidance feature를 뽑습니다. Stage 2에서는 이 전역 blueprint 기반으로 각 패치를 병렬로 생성해 전역 일관성을 유지하면서도 오류 누적과 순차 의존성을 줄입니다.

- **Technical Challenges**: 핵심 난제는 (1) 전역 구조를 안정적으로 계획해 후속 합성에서 오류가 번지지 않게 하는 것, (2) text 이상의 정밀 spatial controllability를 layout 조건으로 구현하는 것, (3) 순차 패치 생성을 제거해도 전역 의미를 깨지 않게 병렬 합성을 조율하는 것입니다. 저해상도 blueprint 생성에는 Stable Diffusion inpainting 백본에 Layout Adapter와 Gated Fuser를 결합해 바운딩박스 조건을 구조 토큰으로 주입하고, 전역 guidance feature bank을 단계별로 캐시합니다. 병렬 패치 합성에서는 forward diffusion의 low-frequency preservation 성질을 이용해 각 패치가 blueprint에서 온 구조적으로 정렬된 초기 noise로부터 시작하도록 하여, 글로벌 코히어런스를 유지한 채로 독립 denoising이 가능하게 했습니다.

- **Empirical Impact**: 대규모 artwork 데이터셋 기반 실험에서 제안 방법은 시각적 충실도와 의미적 일관성을 개선하면서, 기존 기준선 대비 추론 시간을 크게 줄였다고 보고합니다. 특히 artwork outpainting에서 사용자가 명시한 layout을 그대로 반영하는 제어 기능을 ‘정확한 객체 위치/구성’ 관점에서 차별적으로 지원합니다. 결과적으로 전역 계획(blueprint) 부재로 생기던 고해상도 구조 붕괴와 순차 처리 병목을 동시에 완화했다는 점에서, 고해상도 생성/편집 파이프라인의 실용성을 높이는 영향이 기대됩니다.



### Enhanced Seam Segmentation for Automated Welding Robot in Construction Through Transfer Learning: Addressing Limitations of Bilateral Segmentation Network (https://arxiv.org/abs/2607.06150)
- **Prior Approaches**: 기존 용접 이음부(Seam) 분할 연구는 엣지/기하 기반 센싱과 CNN 기반 semantic segmentation(U-Net, DeepLabV3+, BiSeNetV2 등)으로 발전했지만, 공사 현장의 강한 금속 반사와 조명 변화에 대한 “반사 강건성”은 여전히 취약합니다. 특히 얇고 가는 용접선 구조에서 분할 마스크의 단절과 중심선 불안정이 자주 발생해 downstream 로봇 궤적까지 흔들 수 있다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 BiSeNetV2에 아키텍처를 추가하지 않고, transfer learning과 하이브리드 loss인 Cross-Entropy–Lovász(CE–Lovász)로 반사에 강한 용접선 분할을 학습 안정화 관점에서 개선하는 프레임워크를 제안합니다. 목표는 단순 픽셀 정확도 향상이 아니라, 용접선의 연속성과 반사로 인한 false activation을 줄여 “trajectory-stable”한 이음부 표현을 만드는 것입니다.

- **Technical Challenges**: 핵심 난제는 강한 specular reflection이 얇은 용접선 경계를 가리고(잡음처럼 보이게 만들고), 클래스 불균형까지 겹쳐 학습이 불안정해진다는 점입니다. 연구진은 OHEM 기반 초기화로 수렴성을 먼저 확보한 뒤, CE–Lovász 하이브리드로 영역 단위 IoU 일관성과 픽셀 경계 안정화를 동시에 학습하도록 구성해 경로가 끊기는 failure를 회복하게 했습니다.

- **Empirical Impact**: 실험에서 제안 방법은 Joint IoU 81.76%, mIoU 90.73%를 달성했으며, FLOPs·파라미터·추론 속도는 동일한 BiSeNetV2 조건을 유지했습니다. 특히 반사 조건의 severe zero-IoU 실패를 96.33%까지 복구해, OHEM 기반 기준선 대비 Joint IoU를 +22.36%p 끌어올렸고, 경량 실시간 모델에서 최적화 전략의 효과가 더 두드러졌습니다.



### RFHNet: Relational and Frequency-Aware Hashing Network for Large-Scale Fine-Grained Food Image Retrieva (https://arxiv.org/abs/2607.06148)
Comments:
          10 pages, 6 figures. Published in ACM ICMR 2026

- **Prior Approaches**: 기존 FGIR(특히 음식 FGIR) 해싱 기반 방법들은 part alignment, 필터링/재구성, multi-branch 구조 등으로 성능을 끌어올렸지만, 미세한 국소 의미 차이를 충분히 포착하지 못하는 경우가 많았다. 또한 공간 영역 중심의 특징 결합이 많아 high-frequency 잡음과 low-frequency 아티팩트에 민감하며, 서로 다른 스케일/레이어 특징을 단순 concatenation으로 합치는 탓에 의미 상관과 기여도 불균형을 반영하지 못했다.

- **Core Contribution**: RFHNet은 cascaded hierarchical hashing 구조로 글로벌 구조와 파인그레인드 로컬 디테일을 단계적으로 함께 학습해, 짧은 비트에서도 더 구별력 있는 hash code를 만든다. FRM(Fine-grained Relation Modeling)으로 국소 간 미세한 공간 관계를, MFMF(Multi-Frequency Modulated Fusion)로 주파수 대역별 유익 신호를, HSS(Hierarchical Semantic Synergy)로 레벨 간 의미 시너지를 적응적으로 통합한다.

- **Technical Challenges**: 핵심 난제는 (1) 비슷한 음식 간 미세 차이를 국소 관계로 안정적으로 모델링하고 (2) 주파수 관점에서 잡음/아티팩트를 억제하며 (3) 계층별 의미 상관을 학습해 단순 결합의 한계를 넘는 것이다. RFHNet은 FRM에서 차분(difference) 기반 국소 관계 신호를 효율적으로 만들고, MFMF에서 2D-FFT 후 low/mid/high 대역을 분리·게이팅해 mid-frequency를 문맥 앵커로 활용하며, HSS에서는 local self-attention을 국소 상호작용에만 적용한 뒤 global과 가중 통합해 의미 희석을 줄인다.

- **Empirical Impact**: 6개 음식 전용 벤치마크(Food-101, Vireo Food-172, UEC Food-256, VegFru, ISIA Food-500, Food2K)에서 RFHNet은 해싱 SOTA를 일관되게 능가했으며, 12 bits에서 mAP이 4.44%~17.20% 개선됐다. 특히 VegFru(17.20%), Food2K(16.93%), ISIA Food-500(15.98%)에서 큰 격차를 보였고, 12~48 bits 범위에서도 강건하게 우위를 유지해 대규모 시각 음식 검색과 스마트 케이터링 같은 실용 시나리오에서의 효용을 입증했다.



### Tuning-Free Latent Diffusion Models for Ultrahigh-Resolution Image Editing (https://arxiv.org/abs/2607.06136)
Comments:
          29 pages, 29 figures. Published in IEEE Transactions on Neural Networks and Learning Systems

- **Prior Approaches**: 기존 확산 기반 이미지 편집은 보통 학습된 고정 해상도에 맞춰 동작하도록 설계되어 있어, 더 큰 입력으로 그대로 확장하면 품질 저하나 실패가 잦다. 이를 해결하려고 (1) 처음부터 학습/고해상도 fine-tuning을 하거나 (2) 편집 후 super-resolution으로 보정하는 방식이 쓰이지만, 전자는 비용이 크고 후자는 편집-보정 간 결합이 어색해 고주파 디테일이 흐려지기 쉽다. 또한 고해상도 생성을 위한 tuning-free 접근(예: MultiDiffusion, DemoFusion 계열)은 전경/배경을 구분하지 못해 편집 경계에서 불일치와 점진적 디테일 손실이 생길 수 있다.

- **Core Contribution**: UltraDiffEdit는 tuning-free로, off-the-shelf latent diffusion model(LDM)을 고해상도 실사 편집에 그대로 확장하는 프레임워크를 제안한다. 핵심은 “encode–diffuse–denoise–decode–blend”를 해상도 단계별로 반복하는 coarse-to-fine 멀티스케일 progressive editing이며, 편집 영역과 비편집 영역을 라티언트에서 일관되게 섞어 넣는 데에 있다. 이 과정에서 multi-patch encoding으로 편집/비편집 디테일을 보존하고, global-local consistency denoising과 patch-based hybrid sampling로 경계 아티팩트와 의미적 일관성을 동시에 완화한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 비편집 영역의 라티언트 정보가 잡음이나 강한 conditioning(예: 텍스트 프롬프트)에 의해 흔들리며 구조가 깨지는 문제, (2) 편집 경계에서 라티언트 재생성이 누적되며 경계 아티팩트가 생기는 문제, (3) 고해상도에서 로컬-글로벌 피처 균형을 잡는 샘플링이 어려운 문제다. UltraDiffEdit는 단계별로 저해상도에서 시작해 상향 정교화하며, multi-patch encoding으로 메모리 제약 아래서도 고주파를 포함한 라티언트 표현을 유지한다. 또 경계 인접 영역에서는 global-local consistency denoising으로 편집/비편집 라티언트를 조건부로 매끄럽게 통합하고, patch-based hybrid sampling으로 로컬·중간·글로벌 문맥을 함께 반영해 디노이징 중 의미 응집성과 세부를 보강한다.

- **Empirical Impact**: 논문은 UltraDiffEdit이 단 한 장의 NVIDIA GeForce RTX 3090에서도 최대 8K 해상도까지 고품질 편집을 수행할 수 있음을 실험으로 보여준다. DIV2KEdit, Syn2KEdit, UHRSDEdit 3종 벤치마크를 구축해 기존 SOTA와 비교했으며, 특히 경계 일관성과 세부 보존 측면에서 우수한 편집 품질과 유연성을 강조한다. 결과적으로 “초고해상도 실사 편집”을 추가 학습 없이 현실적 자원으로 확장할 수 있다는 점에서 확산 편집 파이프라인의 적용 범위를 크게 넓히는 의미가 있다.



### AEGIS: A Mechanism-Guided Defense against Visual Synonym Jailbreaks in Text-to-Image Models (https://arxiv.org/abs/2607.06120)
- **Prior Approaches**: 텍스트-이미지 확산 모델의 안전 정렬은 대체로 입력 전처리(semantic sanitization), 특정 트리거 경로 차단(trigger-specific disruption), 가중치/구조의 가지치기(구조적 feature pruning)로 나뉜다. 그런데 이런 방식은 필터·편집·로컬라이징 단계에서 ‘명시적으로 노출된’ 위험 개념을 전제로 설계되는 경우가 많아 시각적 동의어 공격(VSA)에 구조적으로 취약하다. VSA는 텍스트 표면은 무해해 보이지만, 생성 과정에서 시각-의미가 수렴하며 금지 이미지를 유도해 기존 방어의 안전-유틸리티 딜레마를 만든다.

- **Core Contribution**: 이 논문은 VSA를 단순히 특정 단어를 막는 문제로 보지 않고, 생성 중에 위험 의미가 ‘어떻게 발생/수렴하는지’를 동적으로 추적하는 방향으로 패러다임을 전환한다. 메커니즘 분석 결과, VSA와 명시적 위험 프롬프트는 희소한 semantic-injecting attention heads에서 나타나는 의미 주입(injection)으로 연결·수렴한다. 이를 바탕으로 추출된 취약 헤드만을 대상으로, 추론 시점에 similarity-aware repulsion으로 위험 의미를 즉시 제어하는 AEGIS를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 “텍스트 표면에서는 안전해 보이는데 생성 시각-의미 수렴으로 위험이 드러난다”는 점을, 방어가 개입 가능한 내부 구성요소로 로컬라이징하는 것이다. 논문은 anchor-based similarity profiling과 sparse regression(L1 정규화)을 결합해 레이어-시점에서의 시그니처 유사도 변화를 추적하고, MHA의 선형성을 활용해 개별 attention head 기여를 분해한 뒤 소수의 semantic-injecting heads를 식별한다. 그 다음에는 고위험 semantics가 감지될 때 식별된 헤드에만 repulsion을 적용해 과도한 억제로 인한 양성 개념 손상(오버미티게이션)을 피하도록 설계한다.

- **Empirical Impact**: SD 1.4에서 16개 베이스라인 대비 성능을 개선하며, in-domain violence/nudity VSA에 대해 ASR을 0.00/0.03으로 낮추고 out-of-domain의 explicit 및 adversarial 공격에서도 ASR ≤ 0.09를 달성한다. 동시에 benign fidelity를 유지하고 hard-negative에 가까운 양성 개념을 무분별하게 억제하지 않는다는 점을 실험으로 보여준다. 더 나아가 백본별로 취약 헤드를 재식별한 뒤 SD 2.1과 FLUX.1로까지 전이되어, 실서비스 적용 관점에서의 범용성/이식성을 강조한다.



### WebRetriever: A Large-Scale Comprehensive Benchmark for Efficient Web Agent Evaluation (https://arxiv.org/abs/2607.06118)
- **Prior Approaches**: 기존 웹 에이전트 벤치마크는 오프라인/온라인으로 나뉘지만, 웹사이트 규모·도메인 커버리지·의도 다양성이 제한적이라 도메인 간 일반화 능력을 충분히 검증하지 못했습니다. 또한 LLM-as-Judge 평가는 스크린샷 중심이라 쿼리 정식화와 filtering 같은 상호작용의 미세 의미를 잘 반영하지 못하고, 네비게이션 성공률 위주로 실제 배포 관점의 요구(지식 활용·엔드투엔드 수행)를 놓치는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 800개 웹사이트와 1,550개 태스크로 구성된 대규모 벤치마크 WebRetriever를 제안해, 소비자·전문가·엔터프라이즈 전반의 도메인 및 user intent 패턴을 폭넓게 포괄합니다. 아울러 스크린샷을 넘어 에이전트의 브라우저 상호작용 맥락을 활용하는 LLM-as-Judge 프레임워크 NavEval과, 배포 지향의 3개 평가 프로토콜(네비게이션/지식 보조/엔드투엔드 정보추출)을 함께 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 동적 웹에서 발생하는 상호작용 신호를 잡음 없이 구조화해 LLM 판단에 유용한 형태로 만드는 것과 (2) 페이지 도착 여부와 실제 정답·워크플로 완료를 분리해 평가하는 것입니다. 이를 위해 NavEval은 Playwright 기반으로 단계별 action과 request/URL 등 중간 신호를 수집한 뒤, 규칙 기반 필터링으로 관련성 없는 필드를 제거·정규화하고 최종 스크린샷과 결합해 성공/실패를 판정하도록 설계됐습니다.

- **Empirical Impact**: 실험 결과, WebRetriever의 3개 프로토콜에서 에이전트 성능은 전반적으로 낮게 나타났고(예: Protocol I 평균 21.1%), 단순 네비게이션 성공이 실제 배포 효율을 대변하지 못함이 드러났습니다. NavEval은 인간 평가와의 일치도가 프로토콜 전반에서 90% 이상으로 나타나 기존 자동 평가기 대비 상호작용 수준의 판별력을 높였으며, 추가로 Online-Mind2Web에서도 높은 인간 동의율(평균 AR 97%)을 보여 외부 벤치마크로의 견고성까지 입증했습니다.



### RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations (https://arxiv.org/abs/2607.06109)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 adversarial training은 주로 단일 위협(예: ℓ∞)에 최적화돼 추론 시 보지 못한 위협에 취약합니다. Multi-perturbation adversarial training(MAT)은 여러 ℓp 위협을 함께 학습하지만, 서로 다른 위협이 만드는 분포 이동이 충돌해 특정 위협에서의 robust 성능이 떨어지는 robustness trade-off가 발생합니다. 이를 완화하려고 MoE를 MAT에 그대로 붙이면, 전문가가 위협 공통 특징만 중복 학습하거나 gating이 위협별로 거의 동일한 경로를 학습하는 threat-agnostic routing 문제가 남습니다.

- **Core Contribution**: 이 논문은 MAT에서 MoE를 사용할 때 핵심 병목인 redundancy(전문가의 중복 학습)와 threat-agnostic routing을 동시에 겨냥합니다. 제안하는 Robust Mixture of Low-Rank Experts(RoME)는 공유 backbone 위에 low-rank additive update 형태의 전문가를 얹어, backbone이 위협 공통 특징을 맡고 각 expert가 위협별 특징에 집중하도록 설계합니다. 또한 gating이 위협을 구분하도록 dual-scale gating(로컬/글로벌 특징 결합)과 threat-guided gating diversification(위협 간 routing 다양성 강제)을 도입해 위협별 model pathway를 형성합니다.

- **Technical Challenges**: RoME가 해결한 첫 과제는 전문가가 위협 공통 특징을 과도하게 가져가면서 threat-specific pathway가 만들어지지 않는 점입니다. 이를 low-rank 전문가(LoRA식 add-on)로 구현해, 전문가가 공유 지식을 훼손하지 않으면서도 threat-specific 정보를 학습하도록 제약을 줍니다. 두 번째 과제는 gating이 위협별 판별 신호가 충분하지 않아 유사한 expert 조합을 반복하는 threat-agnostic routing인데, 로컬/글로벌 이중 스케일 신호로 discriminative cues를 강화하고, 학습 중 위협 라벨을 활용해 gating 패턴의 거리를 벌리는 다양화 정규화를 통해 경로 차이를 강제합니다.

- **Empirical Impact**: CIFAR-10, ImageNet-100, ImageNet-1K에서 RoME는 기존 SOTA MAT 대비 union robustness와 natural accuracy를 동시에 개선하며, 학습 중 보지 못한 unseen threats에 대한 robustness도 향상시켰습니다. 또한 ablation/분석을 통해 low-rank expert 구성, dual-scale gating, gating diversification의 기여를 확인합니다. 특히 threat 라벨은 학습에서만 사용되지만, 추론 시에도 위협 타입을 몰라도 threat-adaptive expert 조합을 예측해 전이 효과를 보이며, non-ℓp	hreats에 대해서도 견고함이 확장되는 점이 의미 있습니다.



### EcoVision: AI-Powered Drone Imaging for Salt Marsh Vegetation Monitoring and Dominance Mapping (https://arxiv.org/abs/2607.06105)
Comments:
          37 pages, 8 Figure, 6 Tables

- **Prior Approaches**: 기존 생태 모니터링은 현장 조사에 크게 의존해 왔고, 사람의 시각적 판단과 표본 추정으로 인해 관찰자 편향과 확장성 문제가 반복해서 지적돼 왔다. UAV 원격탐사와 딥러닝이 등장했지만, 많은 연구가 픽셀 단위 분할이나 분류에 그쳐 경쟁/우점 같은 생태학적 해석으로 바로 연결되지 못했다.

- **Core Contribution**: EcoVision은 저고도 UAV RGB 영상에서 종 분할(semantic segmentation)부터 객체 단위 분류(object-level classification), 그리고 2x2m 격자 기반 우점도(dominance score) 산출까지를 하나의 모듈형 파이프라인으로 통합했다. 특히 Spartina maritima와 Puccinellia maritima처럼 서로 다른 형태가 수 cm 단위로 공존하는 염습지에서, 픽셀 예측을 정책/현장 조사와 맞닿는 정량 지표로 변환하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 겹친 캐노피와 배경 잡음(물, 침전물 등) 속에서 경계를 안정적으로 찾는 일, (2) 분할 결과를 객체 단위로 끊어내어 fine-grained 종을 식별하는 일, (3) 마지막에 2x2m 같은 현장 스케일로 일관되게 집계해 우점도를 계산하는 일이다. EcoVision은 SegFormer-B5로 식생 마스크를 만든 뒤 connected-component로 blob을 추출하고, ConvNeXt로 blob을 종 분류한 다음, confidence threshold를 적용해 격자 집계로 dominance를 계산하는 방식으로 이를 해결했다.

- **Empirical Impact**: 제안 파이프라인은 종 마스크에서 mean IoU 0.56, 픽셀 정확도 0.96을, 객체 수준 분류에서 F1 0.99를 보이며 영상 기반 식별 성능을 입증했다. 우점도 추정은 사분면(quadrat) 현장 조사와 평균 절대차 8% 미만으로 일치해, 현실적인 조건에서도 미세 공간 구조를 보존하면서 생태학적으로 해석 가능한 결과를 제공함을 시사한다.



### PVCap: Towards Accurate 3D Dense Captioning via PseudoCap and VoxelCapN (https://arxiv.org/abs/2607.06097)
Comments:
          13 pages

- **Prior Approaches**: 3D dense captioning은 3D 장면의 각 객체를 바운딩 박스로 위치시키고 자연어 문장을 생성하는 과제로, Scan2Cap·MORE·SpaCap3D 등은 관계 추론 모듈로 객체 간 관계를 모델링해 성능을 끌어올렸다. 또한 Vote2Cap 계열처럼 detection과 caption을 end-to-end로 결합하는 시도도 있었지만, 데이터 증강이 제한적이고 네트워크 아키텍처가 단순해 충분한 공간 정보·풍부한 의미 특징을 확보하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 PVCap(PseudoCap + VoxelCapNet)으로 두 병목을 동시에 해결한다. PseudoCap은 인스턴스 단위로 객체를 랜덤 믹싱해 다양한 공간 레이아웃의 pseudo frame을 만들고, teacher-student로 pseudo caption label을 생성해 학습 샘플 수와 환경 서술 능력을 키운다. VoxelCapNet은 voxel 기반 백본의 특징과 detection head의 객체 제안을 caption head에 직접 연결하도록 구성해 voxel 특징을 활용한 강한 캡션 생성 네트워크를 제안한다.

- **Technical Challenges**: pseudo frame에서는 객체 주변 문맥이 바뀌어 기존 캡션 라벨을 그대로 쓸 수 없다는 문제가 생기는데, 이를 teacher의 예측으로 pseudo label을 만들고 confidence가 낮은 라벨을 제거해 품질을 관리한다. 또 voxel 기반 아키텍처에서 caption head를 잘 붙이는 것이 핵심이라, VoxelCapNet은 voxel feature에서 detection head로 bbox와 confidence를 산출한 뒤 NMS·confidence 필터링으로 query(객체 특징)를 구성하고 surrounding contextual feature와 함께 caption head에 입력한다.

- **Empirical Impact**: PVCap은 ScanRefer와 Nr3D 두 벤치마크에서 state-of-the-art를 크게 갱신했으며, CIDEr@0.5IoU 기준으로 ScanRefer에서 11.41%p, Nr3D에서 13.99%p 향상했다. 특히 SCST 튜닝까지 포함했을 때 개선 폭이 더 커져 pseudo label 기반 데이터 증강과 voxel 기반 캡션 네트워크 설계가 함께 효과적임을 보여준다. 코드 공개도 예고되어 3D dense captioning 후속 연구를 위한 강력한 baseline으로 기대된다.



### MSA-DCNN: A Data-Efficient Multi-Scale Deformable CNN for Medical Image Classification (https://arxiv.org/abs/2607.06083)
- **Prior Approaches**: 기존 딥러닝 의료영상 분류는 성능이 좋지만, 고정된 샘플링과 수용영역 때문에 다중 스케일 형태 변이(조직/세포의 구조적 이질성)를 충분히 반영하지 못한다. DCN(Deformable Convolution) 계열은 적응적 샘플링을 제공하지만, 스케일 간(해상도 간) 의미 정렬과 융합을 명시적으로 결합한 설계가 약하고 라벨이 적을 때의 안정성도 별도 정규화에 의존하는 경향이 있다.

- **Core Contribution**: 이 논문은 MSA-DCNN(Multi-Scale Attention Deformable Convolutional Neural Network)으로 스케일 일관성(scale-consistent) 학습 원칙을 제안한다. 각 스케일에서 주의(attention)를 분해 적용해 within-scale 살리언시를 정제하고, 학습된 multi-scale attention으로 cross-scale 융합을 수행하며, 얕은/깊은 표현 정렬을 위한 auxiliary self-distillation까지 하나의 통합 최적화로 묶어 라벨 효율을 끌어올린다.

- **Technical Challenges**: 기여의 핵심은 (1) 구조적으로 서로 다른 해부학적 패턴에서 샘플링을 스케일별로 맞추는 것과 (2) 스케일 간 표현을 라벨이 부족한 상황에서도 일관된 의미 기하로 정렬하는 것이다. 저자들은 스케일별 deformable branch로 receptive-field를 적응시키고, MCBAM의 channel–spatial 재보정을 통해 within-scale 분포를 정규화하며, 투영된 multi-resolution 임베딩을 learned multi-scale attention으로 결합하는 동시에 self-distillation을 KL-divergence 기반 의미 정렬 제약으로 추가했다.

- **Empirical Impact**: C-NMC, PBC, ISIC-2020 3개 공개 벤치마크와 외부 보유 leukemia 코호트에서 MSA-DCNN은 AUC(이진), Accuracy, F1에서 경쟁/우위를 보였고, 특히 라벨 분율이 줄어드는 환경에서도 성능 저하가 작아 data-efficient 관점에서 강점을 확인했다. 또한 ablation 결과 DC(Deformable Convolution), SSA(Scale-Specific Attention), MSA(Multi-Scale Attention), SD(Self-Distillation)가 서로 보완적으로 작동하며, fewer parameters로도 성능을 유지해 의료영상 분류의 실용적 기반을 제공한다는 점을 강조한다.



### Why does Deep Learning Improve Visual SLAM? (https://arxiv.org/abs/2607.06023)
- **Prior Approaches**: 기존 V-SLAM은 크게 기하 기반(frontend–backend)으로 나뉘며, 특징 기반은 수제 keypoint/descriptor 매칭, 직접법은 픽셀 강도 일관성을 전제로 한다. 이런 전통적 방식은 저텍스처, 심한 모션 블러, 조도 변화 같은 실제 환경에서 취약해지고, end-to-end 회귀 방식은 학습 분포 밖 일반화가 약하다는 문제가 있었다. 최근 딥러닝 V-SLAM은 learned 2D data association과 uncertainty를 이용하고, differentiable geometric optimization을 recurrent 구조로 반복 적용해 성능을 끌어올렸지만, 성공 요인이 어떤 구성요소에 있는지는 불명확했다.

- **Core Contribution**: 이 논문은 딥러닝 기반 V-SLAM의 성능이 (1) learned 2D data association 단독, (2) 여기에 uncertainty 결합, (3) recurrent architecture 자체 중 무엇 때문인지 질문한다. 이를 위해 기하 기반 대표 시스템인 ORB-SLAM3의 수제 디스크립터 매칭 모듈만 optical flow 기반 대응 추정으로 교체해 ORB-SLAM3-OF를 만들고, 여기에 uncertainty로 bundle adjustment의 잔차 가중치를 적용한 ORB-SLAM3-OF-U를 추가한다. 그리고 같은 SLAM 파이프라인을 유지한 채 두 구성요소의 기여를 직접 분리·정량화한다.

- **Technical Challenges**: 핵심 과제는 learning-based 대응(광류)과 불확실성을 고전 SLAM의 동일 최적화 틀에 “방법론적으로 일관되게” 끼워 넣는 것이었다. 저자들은 ORB-SLAM3 내부에서 추적/현지화의 매칭을 광류로 예측된 2D 위치 주변에서 feature를 찾는 방식으로 대체하고, ORB-SLAM3-OF에서는 bundle adjustment 잔차에 균일 가중치(1.0)를 적용해 learned uncertainty 효과를 제거한 기준을 만든다. ORB-SLAM3-OF-U에서는 flow network가 예측한 confidence/uncertainty로 reprojection error를 가중해, 대응이 흔들리는 저품질 영상 구간에서 강건성을 얻도록 설계한다.

- **Empirical Impact**: 실험은 대표 난이도 벤치마크인 TartanAir와 UZH-FPV에서 궤적 정확도(translation ATE, rotation ATE)를 비교하는 방식으로 진행됐고, ORB-SLAM3-OF와 ORB-SLAM3-OF-U가 기본 ORB-SLAM3보다 크게 향상되는 결과를 보였다. 특히 learned uncertainty를 더한 ORB-SLAM3-OF-U가 시각적으로 어려운 조건에서 강건성이 증가하며, out-of-distribution으로 여겨지는 UZH-FPV에서 딥 V-SLAM 계열의 성능을 뛰어넘는 양상이 보고된다. 결론적으로 딥 V-SLAM의 핵심 이득은 recurrent 구조보다는 learned 2D data association과 uncertainty에 달려 있으며, 다음 세대 V-SLAM 설계에서 이 두 구성요소를 학습 기반으로 적극 도입해야 한다는 메시지를 준다.



### KOAL: Knowledge-Driven Prostate Cancer Grading with Ordinal-Aware Learning (https://arxiv.org/abs/2607.06019)
Comments:
          10 pages, 2 figures, 2 tables. Accepted at MICCAI 2026. This is the submitted version prior to peer review. The final authenticated version will be available on SpringerLink

- **Prior Approaches**: 기존 mpMRI 기반 Gleason Grade Group(GGG) 예측은 주로 영상 정보에 의존해 나이, PSA 같은 비영상 임상변수와 방사선 판독에 담긴 전문가 사전지식(예: radiology report)을 충분히 반영하지 못한다. 또한 GGG를 단순한 평면형 분류 라벨로 다루어, 1차(primary)와 2차(secondary) Gleason 패턴의 내재적 위계(hierarchy)를 제대로 학습하지 못한다.

- **Core Contribution**: 논문은 KOAL(Knowledge-Driven Ordinal-Aware Learning) 프레임워크를 제안하며, 임상 맥락·전문가 지식·병리 위계를 함께 학습하도록 설계했다. Clinical-Context Modulation(CCM)로 임상 변수를 영상 표현에 동적으로 반영하고, LLM 기반 Knowledge-Guided Prototype Alignment(KGPA)로 보고서/가이드라인에서 등급별 의미 앵커를 추출해 추론 시 환자별 리포트를 요구하지 않도록 했다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 임상변수와 mpMRI 표현을 효과적으로 결합해 성능을 끌어올리는 것과 (2) GGG를 위계적인 Gleason 패턴 구조로부터 일관되게 추론하는 것이다. 저자들은 HOC(Hierarchical Ordinal-aware Constraints)에서 1차·2차 예측을 분리한 뒤 Differentiable Bio-logic Mapping Layer(DBML)로 병리적 등급 일관성을 유지하며 GGG로 매핑하고, KGPA의 프로토타입 대조 정렬로 등급 병리 정렬 표현학습을 유도한다.

- **Empirical Impact**: PI-CAI 공개 데이터와 사내(in-house) 데이터 실험에서 KOAL은 기존 state-of-the-art 방법들을 능가하는 것으로 보고됐다. 임상적으로 불필요한 생검을 줄이기 위한 비침습적 GGG 예측의 정확도를 높일 뿐 아니라, 전문가 지식과 병리 위계를 함께 반영하는 방식이 향후 mpMRI 기반 온콜로지 예측 연구에 실질적 기준점을 제시한다.



### Structured Data Extraction from Real Estate Documents using Clustering, Classification, and Large Language Models (https://arxiv.org/abs/2607.06012)
- **Prior Approaches**: 기존 부동산 플랫폼은 API로 가격·면적·침실 수 같은 정형 메타데이터만 제공하고, 판매자가 작성한 질문지 PDF의 핵심 정보는 비정형으로 방치되어 왔다. 규칙 기반 추출(정규표현식·템플릿·룰)은 문구 변형과 레이아웃 차이에 취약해 확장성이 떨어진다. LLM을 활용한 비정형 정보 추출은 가능하다고 알려졌지만, 실제 라이브 플랫폼에서 수천 문서를 일관 스키마로 추출·검증한 파이프라인 부재가 한계로 남아 있었다.

- **Core Contribution**: 이 논문은 부동산 질문지 PDF에서 35개(질문지 속성) 대신 본문에서 언급된 3535개(미리 정의된 속성) 필드를 스키마 고정 JSON으로 뽑아내는 end-to-end 파이프라인을 제안한다. 먼저 reverse-engineered REST로 3,965개가 아닌 3,965(문서 수 표기 오차 가능) 수준의 질문지 PDF와 메타데이터를 대량 수집하고, PDF를 text_only/scanned/special_char 3유형으로 분류해 처리 경로를 분기한다. DeepSeek R1을 통해 각 문서의 속성을 JSON으로 추출해 다운스트림 데이터셋(중복 제거 후 2,766개 레코드)을 구축한 점이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 난제는 (1) PDF가 디지털 텍스트, 스캔 이미지, 체크박스·특수문자 중심 레이아웃 등으로 이질적이라는 점, (2) LLM 배치 추출에서 출력 형식 일관성과 스키마 준수가 깨질 수 있다는 점이다. 논문은 pdfplumber/PyMuPDF 기반으로 텍스트 길이와 체크 표시 유무를 사용해 문서 유형을 선분류하고, text_only인 경우에만 원문 텍스트를 LLM에 그대로 넣어 용어 변형을 흡수하도록 했다. 또한 temperature 0.3 등 낮은 샘플링으로 변동을 줄이고, ‘항목을 모두 포함하되 null 허용·JSON만 반환’ 제약으로 파싱 실패를 줄이며 property_id까지 결합해 DB에 적재했다.

- **Empirical Impact**: 실험 결과, 추출 단계 대상 문서(텍스트로 분류된 2,781개)에서는 DeepSeek R1의 JSON 생성과 DB 적재가 모두 성공해 스케일에서의 실행 가능성을 보였다. 다운스트림 검증에서는 cosine similarity 기반 매칭의 Jaccard consistency가 약 0.82로 높고, K-Means에서 해석 가능한 2개 시장 세그먼트가 형성되며 silhouette score는 0.2088로 보고됐다. 또한 MCDM(Weighted Scoring/TOPSIS) 간 Top-5 겹침이 낮게(약 0.18) 나타나 서로 다른 정렬 목표에 대해 데이터가 충분한 신호를 제공함을 시사한다.



### OBBSeg: Irregular Lesion Segmentation under Oriented Bounding Box Annotations (https://arxiv.org/abs/2607.06007)
Comments:
          18 pages, 7 figures. ECCV 2026

- **Prior Approaches**: 기존 의료 영상 세그멘테이션은 U-Net 계열처럼 픽셀 단위 마스크가 필요해 라벨 비용이 큰 부담이었습니다. 이를 줄이기 위해 점/스크리블/박스 같은 weak supervision이 등장했지만, 특히 axis-aligned bounding box는 방향 정보를 담지 못해 길쭉하거나 비등방성 병변의 형상 제약이 약합니다.

- **Core Contribution**: 본 논문은 weak와 fully supervision의 간극을 잇기 위해 OBBSeg(Oriented Bounding Box Guided Segmentation)를 제안합니다. Oriented Bounding Box(OBB)를 이용해 병변의 공간 범위와 방향을 동시에 부여하고, 이를 prompt-guided 의미 학습(encoder 주입)과 결합해 더 정교한 세그멘테이션을 유도합니다.

- **Technical Challenges**: OBB는 사각형 기반이라 ‘rectangular shape bias’가 생길 수 있어, Mask-to-OBB loss(M2O)로 이 편향을 제거합니다. M2O는 예측 마스크를 회전·투영·퓨전 경로로 OBB 공간에 정렬한 뒤 BCE/Dice로 geometry-consistent하게 학습하며, 동시에 PAFE/DBFE 모듈로 전경 강조와 배경 간섭 억제를 수행합니다.

- **Empirical Impact**: OBBSeg는 5개 영상 모달리티의 13개 데이터셋에서 기존 weakly supervised 방법을 능가하고, 일부 설정에선 fully supervised 수준에 준하는 성능을 보였습니다. 또한 OBB가 길쭉한 병변에서 특히 큰 개선을 보였고, ±각도/위치 오차 및 인체 라벨 변동에도 성능 저하가 작아 실사용 라벨링 환경에서의 안정성도 확인됐습니다.



### Unlearnable Faces: Privacy Protection Surviving Extraction Pipelin (https://arxiv.org/abs/2607.05996)
Comments:
          preprint

- **Prior Approaches**: 기존 unlearnable examples(UE) 계열은 업로더가 작은 교란을 한 번 넣어, 공격자가 학습 시 그 데이터에서 ‘정체성’ 대신 ‘지름길’을 배우게 만들어 인식 정확도를 무너뜨리는 방식이다. 다만 이런 보호는 공격자가 실제로는 업로드 이미지를 그대로 학습하지 않고, detect–crop–resize로 얼굴을 추출·재배율한 뒤 학습하는 현실 파이프라인에서는 쉽게 붕괴됐다.

- **Core Contribution**: 이 논문은 LPID(Localize​d, Pipeline-coupled Identity Defense)로, 공격자의 crop+resize 변환을 교란 최적화 과정에 직접 결합해 보호가 변환을 거쳐도 유지되게 만든다. 또한 정체성에 의존해 특정 사용자에만 맞추는 방식이 아니라, 앨범마다 재최적화하는 구조라서 보호 시점에 보지 못한(unknown) 사용자에게도 동일하게 적용된다고 주장한다.

- **Technical Challenges**: 핵심 난제는 방어자가 공격자의 입력 해상도, resize 커널, 플랫폼 JPEG 품질 같은 세부를 알 수 없다는 점과, 공격자가 학습에 사용하는 ‘추출된 얼굴 입력’이 원본과 주파수·공간적으로 달라진다는 점이다. LPID는 차분가능한 추출 변환 TT(얼굴 박스 crop + bilinear resize)를 통해 교란을 생성하고, 마스크로 얼굴 영역에만 교란 에너지를 몰아 resize가 보존하는 주파수 대역에 지름길이 남도록 최적화한다.

- **Empirical Impact**: 실험에서 LPID는 crop+resize 계열 설정 전반에서 공격자 정확도를 10% 미만으로 유지하며, 보호 시 보지 못한 identities에서도 가장 낮은 성능을 보였다고 보고한다. 동시에 PSNR 32.7dB, LPIPS 0.161 수준으로 시각적 인지 가능성은 낮게 유지되어, 보호가 ‘더 큰 섬네일/교란’이 아니라 ‘변환 결합 설계’에서 온다는 점을 실증했다고 정리한다.



### SparseCtrl-HOI: Sparse Temporal Control for Human-Object Interaction Video Generation (https://arxiv.org/abs/2607.05994)
Comments:
          ECCV 2026, Project Page: this https URL

- **Prior Approaches**: HOI 비디오 생성은 기존에 프레임 단위의 조밀한 temporal guidance(예: 2D/3D 포즈 시퀀스, relative coordinate map)를 통해 손-물체 정렬과 동작을 강하게 통제하는 방식이 주류였다. 반면 sparse spatial control 연구도 있지만, 결국 조밀한 시간 제약으로 바꿔 넣어 입력 잡음이 그대로 전이되거나 보간 과정에서 튐/부자연스러운 전이가 생기기 쉽다. 또한 프레임별 조건을 얻기 위한 주석 비용이 커서 실사용 확장에 부담이 컸다.

- **Core Contribution**: SparseCtrl-HOI는 HOI 비디오를 소수의 interaction keyframe(핵심 타임스탬프의 상호작용 상태)만으로 생성하도록 sparse temporal control 패러다임을 제안한다. Time-Controlled Rotary Positional Embedding(TiRoPE)로 keyframe이 지정된 순간에 정확히 등장하도록 시간 앵커를 걸고, 이후 Motion Prior Injection Module이 중간 프레임의 전이를 자연스럽게 보완한다. 결과적으로 조밀한 프레임 주석 부담을 줄이면서도 손-물체 상호작용의 일관성을 높이는 것을 목표로 한다.

- **Technical Challenges**: 핵심 타임스탬프 정렬(TiRoPE)만으로는 중간 구간의 모션이 매끄럽지 않거나 물리적으로 그럴듯하지 않을 수 있다는 문제가 있다. 이를 해결하기 위해 MLLM(멀티모달 LLM)이 keyframe 사이의 손-물체 자세 변화를 고수준 motion priors로 추론하고, Q-Former로 압축한 뒤 DiT의 cross-attention에 주입해 논리적·물리적으로 타당한 전이를 생성하도록 유도한다. 또한 MLLM의 추론 능력과 DiT의 시각 합성 능력을 분리해 decoupled training(appearance 전이와 motion 전이 학습을 단계별로 분리)으로 학습 간 얽힘을 줄였다.

- **Empirical Impact**: SparseHOI-5K(총 4,850클립)는 오디오와 함께 sparse temporal control에 맞춘 고품질 HOI 데이터셋으로, 손/물체 마스크 및 물체 제거(inpainted) 프레임 등 풍부한 멀티모달 주석을 제공한다. 실험에서는 FID/FVD/VBench, Sync-C 같은 표준·정렬 지표와 MS-RAFT, Ti-SSIM, HOI-VLM 같은 제어 전용 지표로 성능을 검증했으며, keyframe 기준 제어 정밀도와 모션 전이 자연스러움에서 우수한 결과를 보고한다. 전반적으로 annotation overhead를 줄이면서 라이브 커머스용 고품질 HOI 비디오 생성에 의미 있는 개선을 보였고, 코드와 데이터는 공개된다.



### SpecTrack: Spectral Prompt Guided Adaptive Experts for Multispectral Object Tracking (https://arxiv.org/abs/2607.05988)
Comments:
          16 pages

- **Prior Approaches**: 기존 MSI/HSI 기반 객체 추적은 밴드별 관측을 활용해 RGB에서 구분하기 어려운 상황(혼합 픽셀, 조명 변화, 가림, 클러터)에서 목표-배경 판별을 강화하지만, 모든 검색 영역을 동일한 고정 용량의 spectral-spatial 경로로 처리하는 경우가 많았다. 그 결과 프레임·타깃 상태에 따라 추적 난이도가 달라져도(명확한 영역 vs 애매한 경계/유사 스펙트럼 교란물) 연산과 추론 강도를 동적으로 조절하지 못했다.

- **Core Contribution**: 이 논문은 검색-영역 수준에서 spectral-spatial 복잡도를 추정하고, 그에 맞춰 추론 용량을 적응적으로 배분하는 SpecTrack을 제안한다. 핵심은 Spectral Adaptive Mixture-of-Experts(SAMoE)로, 잠재 rank·수용영역·깊이가 점진적으로 증가하는 expert pool을 구성하고 필요한 expert만 선택해 처리한다. 여기에 Shared Global Expert가 공통 문맥을 제공해 sparse 라우팅으로 인한 단절된 의사결정을 완화한다.

- **Technical Challenges**: 문제는 “어떤 검색 영역이 얼마나 어려운가”를 expert 선택에 반영해야 하는데, 이를 위해 Spectral Prompt Router가 의미적 컨텍스트, 공간 경계 신호, 그리고 multispectral patch embedding 이후 계산되는 latent channel-variation cue를 결합한다. 이 라우터는 각 검색 영역마다 소수의 SAMoE expert만 활성화하는 sparse 선택을 수행해 효율을 확보하면서도 필요한 정밀 추론을 제공한다. 또한 공통 전역 문맥을 담당하는 Shared Global Expert를 함께 두어 라우팅 결정의 파편화를 줄이는 방식으로 안정성을 높였다.

- **Empirical Impact**: MUST, MSITrack, HOTC20 벤치마크에서 accuracy–efficiency trade-off가 긍정적으로 확인됐다. SpecTrack-L384는 각 벤치마크에서 AUC 65.2%, 51.9%, 72.6%로 SOTA급 또는 경쟁력 있는 성능을 보였고, SpecTrack-B224는 MUST에서 62.4% AUC와 43.7 FPS의 균형 성능을 달성했다. 추가로 GOT-10k 평가에서는 RGB 도메인 일반화가 관측되며 SpecTrack-L384가 79.3% AO를 기록했다.



### Propose and Attend: Training-free MLLM Grounding Confidence via Multi-Token Localized Attention (https://arxiv.org/abs/2607.05978)
- **Prior Approaches**: 기존 연구는 SVAR처럼 생성 텍스트의 한 위치(예: 첫 서브토큰)에서 전 입력 모달리티 토큰으로의 attention을 전역 합산해 환각을 감지하거나, hidden-state 유사도(예: GLSim, ContextualLens)로 근거 부족을 판단했다. 하지만 localization 출력은 박스/시간창처럼 ‘제안된 영역’과 결합돼 있어야 신뢰도를 매길 수 있는데, 이들은 대개 입력 전체를 통째로 보거나 추가 분류기 학습이 필요해 신호가 약했다.

- **Core Contribution**: 이 논문은 Multi-Token Localized Attention(MTLA)이라는 학습 없는 사후(post-hoc) 점수를 제안해, 모델이 주장한 영역 내부에 실제로 attention 근거가 모이는지를 정량화한다. 핵심은 (1) 예측이 스스로 제안한 region 안에서만 attention을 합산하고, (2) 좌표와 라벨 등 multi-token에 분산된 근거를 함께 집계해 더 강한 grounding 신호를 만든다는 점이다.

- **Technical Challenges**: 가장 큰 난제는 토큰 log-probability가 좌표 근거(grounding)와 입력 모호성(hallucination 유발 요인)을 섞어 분리도가 낮다는 것이다. MTLA는 각 응답 토큰이 생성하는 proposal token set을 파싱한 뒤, proposal region과 교차하는 모달리티 토큰에 대해서만 attention을 masked aggregation하고, 여러 예측 토큰을 평균 내며, 아키텍처/모달리티에 덜 민감한 레이어 밴드로 점수를 안정화했다.

- **Empirical Impact**: COCO(이미지), Charades-STA·QVHighlights(비디오), AudioSet-Strong(오디오)에서 MTLA는 환각 감지 AUROC를 기존 훈련 없는 베이스라인 대비 +7~+38까지 개선했다. 또한 MTLA를 confidence로 재랭킹하면 open-source 8B/일반ist 모델의 zero-shot COCO detection AP가 크게 상승(20.4→약 37.0)하며, supervised 탐지기와의 격차를 상당 부분 줄였다.



### Decoupled Single-Mask Annotation Noise Detection via Cross-Sectional Patch Self-Consistency (https://arxiv.org/abs/2607.05965)
Comments:
          13 pages, 6 figures. Accepted by MICCAI 2026

- **Prior Approaches**: 혈관 CT 세그멘테이션은 얇은 관 구조와 조영제 확산 때문에 고품질 라벨을 얻기 어렵고, 현실적으로 한 스캔당 한 번만 주석이 달리며 국소적인 마스크 노이즈가 생긴다. 기존 대응은 (1) 다중 라이터 퓨전처럼 여러 마스크를 필요로 하거나, (2) loss/가중치/구조 보정 등 학습과 결합된 robust learning을 통해서만 간접적으로 다루는 경우가 많아 라벨 실패의 ‘감사(audit) 가능한 근거’ 제공이 어렵다.

- **Core Contribution**: 논문은 단일 마스크(single-mask) 설정에서 이미지-마스크 쌍만으로 annotation noise를 위치화(로컬라이즈)하는 decoupled 프레임워크를 제안한다. 핵심은 혈관의 횡단 단면이 공간과 피험자에 걸쳐 반복되는 점을 이용해 cross-sectional patch self-consistency를 정의하고, 유사한 이미지 패치인데 마스크가 불일치하면 해당 영역을 노이즈로 플래그하면서 각 의심 구간에 대한 해석 가능한 패치-쌍 근거를 제공한다.

- **Technical Challenges**: 가장 큰 기술 과제는 ‘어떤 좌표계/정렬에서’ 횡단 패치를 비교해야 비슷한 모양의 이웃을 충분히 검색하느냐이다. 이를 위해 Frenet–Serret 대신 Bishop frame(병렬이동 기반, torsion-free)을 중심선 그래프에 구성해 안정적인 orthogonal patch를 뽑고, intensity-equivalent 이웃을 scalable vector search로 찾은 뒤 MSE 기반 이미지 유사도 구간별로 mask disagreement(1-IoU)의 조건분포를 보정해 residual을 z-score 형태로 계산, 패치 수준 노이즈 점수를 산출하고 스캔 단위 3D quality map으로 집계한다.

- **Empirical Impact**: ImageCAS 관상동맥 CT 실험에서 quality-weighted training은 경계 민감 지표(CPR-DSC, ASD, HD-95)를 중심으로 개선하며, 전체 DSC는 비슷한 수준으로 유지되어 국소 경계 오류 완화에 강점이 있음을 보여준다. 또한 노이즈가 무작위가 아니라 체계적 편향을 가지며, 특히 transverse/oblique 혈관이 axis-aligned 구조보다 오류율이 5.1배 높고 면적·강도와도 상관을 보이는 등, 탐지 결과가 dataset quality assessment와 QA에 직접 활용될 수 있음을 입증한다.



### NegROI: Click-Centric Uncertainty-Guided Refinement with Scene-Conditioned Negative Prompts for Robust Interactive 3D Segmentation (https://arxiv.org/abs/2607.05955)
- **Prior Approaches**: 기존 대화형 3D 세그멘테이션은 voxel 그리드와 transformer 디코더를 중심으로 click 토큰을 장면 특징에 결합해 마스크를 갱신한다. 그러나 단일 해상도 정제는 소수 클릭에서 경계를 뭉개고, 배경 구조와의 혼동이 hard false positives로 이어지며, 밀도·스케일이 다른 데이터셋 간에는 고정된 refinement 휴리스틱과 click만 기반 디코딩이 일반화에 취약하다는 한계가 있었다.

- **Core Contribution**: NegROI는 (1) click 주변만 미세 그리드로 정제하는 click-centric multi-resolution ROI refinement와 (2) 장면 조건 negative prompts를 결합해 경계 정확도와 false positive 억제를 동시에 노린다. 추가로 불확실성 기반 선택적 정제, negative prompt의 diversity regularizer, 경계 인접 고신뢰 오탐에 초점을 맞춘 boundary-aware hard negative mining을 도입해 다양한 장면에서도 안정적으로 동작하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 소수 클릭 상황에서 fine 경계를 복원하면서도 계산량을 폭증시키지 않는 “정제 위치·해상도” 제어와, 배경의 유사 구조로 인해 생기는 고신뢰 false positive를 명시적으로 억제하는 “negative 모델링”이다. 논문은 coarse 예측 후 클릭 중심 ROI를 더 촘촘한 voxel 해상도로 re-voxelize해 로짓을 국소적으로 다시 디코딩한 뒤 coarse에 max-aggregation과 residual fusion으로 되돌리고, cross-attention 기반 scene-conditioned negative prompts를 boundary-proximal hard negatives로 직접 감독해 오탐 억제에 학습 신호를 준다.

- **Empirical Impact**: ScanNet40(인-도메인)과 S3DIS·KITTI-360(아웃-도메인)에서 IoU@k 및 mAP 지표가 향상되었고, 특히 low click 예산(예: IoU@1~3)에서 개선 폭이 크게 나타났다. 또한 NegROI는 경계 인접 오탐(band false positives)을 줄이며, ScanNet20 단일 클릭 평가에서도 비대화형/오픈보케이블 3D 베이스라인 대비 성능을 보이며 견고성을 입증했다. 결과적으로 “경계 품질 + hard distractor 억제 + 교차 데이터셋 강건성”을 함께 달성하는 대화형 3D 세그멘테이션 방향을 제시했다.



### Progressive Reasoning with Primitive Correction for Compositional Zero-Shot Learning (https://arxiv.org/abs/2607.05911)
- **Prior Approaches**: CZSL 기존 연구는 속성(attribute)과 물체(object)를 독립적으로 예측하거나, 물체 예측을 속성 추론의 일방향 priors로 쓰는 방식이 주류였다. 전자는 속성-물체의 맥락 의존성(contextuality)을 놓쳐 조합 일반화가 제한되고, 후자는 초기 물체 오판이 다음 속성 단계로 그대로 전파되는 error propagation 문제가 생긴다.

- **Core Contribution**: PRPC는 속성과 물체의 양방향 의존성을 단계적 추론으로 명시화해, 이전 단계 오차를 상호 교정하는 Progressive Reasoning with Primitive Correction 프레임워크를 제안한다. CZSL을 CoT(Chain-of-Thought) 스타일의 Q&A형 구조화된 5-step 결정(물체→속성→속성 기준 물체 검증/수정→물체 기준 속성 재검증→최종 조합)으로 재정의한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 열린 생성형 MLLM이 임의의 텍스트를 만들면 중간 단계 파싱/평가가 불안정해지고, (2) step-level 정확도를 정확히 맞추는 RL 보상이 희소해져 학습이 흔들린다는 점이다. PRPC는 GPT-4o로 템플릿에 맞춘 5-step CoT 추적을 생성해 구조를 고정하고, Stage I SFT로 태그/형식을 안정화한 뒤 Stage II에서 GRPO 기반 RL post-training으로 각 단계에 대한 exact-match 보상과 KL 정규화를 결합해 단계별 논리 일관성을 강화한다.

- **Empirical Impact**: MIT-States, C-GQA, VAW-CZSL의 세 벤치마크에서 PRPC는 state-of-the-art 성능을 달성하며, 중간 단계의 속성/물체 정확도와 최종 조합 정확도 모두 향상됨을 보였다. 특히 의도적으로 잘못된 물체를 주입해도 상호 검증 루프가 일부 오차를 회복하는 실험이 제시돼, 유사도 기반 정적 추론 대비 “수정 가능한(reasoning-and-correct)” 조합 일반화의 의미를 실증한다.



### PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails (https://arxiv.org/abs/2607.05910)
- **Prior Approaches**: 기존 비전 가드레일 벤치마크는 고정된 안전 분류 체계나 정적 harmfulness 정의에 의존해, 안전을 이미지의 고유 성질처럼 취급하는 경우가 많았습니다. 정책 조건을 넣는 평가도 있었지만, 같은 이미지에서 정책 경계가 바뀔 때 결정을 뒤집는 능력(정책-적응)을 미세하게 측정하기엔 부족했습니다. 그 결과 많은 모델이 위험 큐는 알아도 정책이 허용/차단을 바꾸면 판단을 안정적으로 수정하지 못하는 취약성이 드러났습니다.

- **Core Contribution**: 이 논문은 정책이 런타임마다 달라질 때(제품/지역/규정 변경 등) 동일 이미지가 서로 다른 결정을 받아야 하는 문제를 다루며 PolicyShiftBench와 PolicyShiftGuard를 제안합니다. PolicyShiftBench는 265개 이미지에 대해 총 2,000개의 정책-판별 인스턴스(7개 위험 카테고리 조합, 28개 정책 변형)를 구성하고, 같은 이미지의 pass/block “정책 플립”을 직접 평가하는 Policy Shift Score(PSS)를 도입합니다. PolicyShiftGuard는 policy-conditioned 가드레일로, Randomized Policy SFT(RP-SFT)와 Boundary-Pair Policy Adaptation(BP-Adapt) 2단계 학습을 통해 정책 경계에 맞춰 결정을 뒤집도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 모델이 이미지에서 위험 단서를 인식하는 것에 그치지 않고, 현재 주어진 정책 텍스트의 경계를 읽어 동일 시각 증거에 대해 안전 판정을 ‘조건부로’ 바꾸는 능력을 확보하는 데 있습니다. 논문은 이미지 속성(atomic attributes)과 정책 판단을 분리해 라벨을 실행 가능한 규칙으로 결정하고, BP-Adapt에서 같은 이미지·카테고리에 대해 “허용 정책 vs 차단 정책”이 매칭된 boundary-pair를 만들어 쌍대 비교 형태의 손실로 안전 점수의 여백을 강제합니다. 또한 RP-SFT에서 정책 제시 순서/표면 식별자 등을 랜덤화해 고정 템플릿이나 위치 기반 지름길 의존을 낮추고 정책-추종의 견고성을 높였습니다.

- **Empirical Impact**: 실험에서 기존 VLM 및 전문 가드레일은 F1은 그럴듯해도 PSS가 매우 낮아, 정책 플립에 취약함이 확인됐습니다. 반면 PolicyShiftGuard-7B는 PolicyShiftBench에서 Avg. F1 76.9, Avg. PSS 72.1을 달성하며, UnSafeBench와 SafeEditBench로도 전이 성능이 좋고 추론 속도까지 개선되어 latency–performance 트레이드오프에서 실용성을 높였습니다. 특히 ablation은 matched pass/block boundary pair 및 pair loss가 정책-적응을 안정적으로 만들기 위한 필수 요소임을 보여줍니다.



### GaussFusion: Towards Multimodal 3D Gaussian Pretraining (https://arxiv.org/abs/2607.05906)
Comments:
          32 pages, 6 figures, 6 tables

- **Prior Approaches**: 3D Gaussian Splatting(3DGS)은 geometry와 appearance를 가우시안 프리미티브로 동시에 표현하지만, 기존 연구는 주로 재구성과 렌더링에 집중해 의미·구조 정보를 파라미터 공간에서 충분히 학습하지 못했다. Gaussian-MAE 계열은 masked Gaussian 속성 복원을 통해 self-supervised 사전학습을 제공하나, 국소 패턴 복원에 치우쳐 범주 수준의 의미 분별을 약하게 만든다. 또한 멀티모달(이미지/텍스트) 감독을 넣더라도 random masking은 가우시안 프리미티브의 비균일 분포와 중요도를 반영하지 못해 의미 정렬의 효율이 떨어진다.

- **Core Contribution**: 본 논문은 3D Gaussian 표현을 위한 멀티모달 사전학습 프레임워크 GaussFusion을 제안한다. GaussFusion은 masked Gaussian modeling에 이미지와 텍스트 감독을 교차모달 semantic alignment 형태로 통합해, 가우시안 인코더가 시각적 단서뿐 아니라 언어 수준 의미 정보까지 함께 학습하도록 한다. 추가로 비균일한 가우시안 분포에 맞춘 Gaussian Salience-guided Multi-scale Hole Masking(GSHM)으로 마스킹 전략 자체를 정교화한다.

- **Technical Challenges**: 핵심 난제는 (1) masked reconstruction 목표가 범주 의미 학습으로 이어지게 만드는 것과 (2) 멀티모달 정렬이 마스킹 선택에 의해 약화되지 않도록 하는 것이다. GaussFusion은 이미지/텍스트를 단순 보조가 아니라 정렬을 담당하는 learnable alignment token을 통해 트랜스포머 학습 과정에 직접 참여시키고, 복원 손실(좌표는 Chamfer Distance, 나머지 속성은 L1 계열)과의 공동 최적화로 의미 분별성을 강화한다. 동시에 GSHM은 opacity·scale 기반 salience로 중요 국소 그룹을 우선하고, 여러 스케일의 공간적으로 연속된 hole mask를 만들어 복원 난이도 변동과 의미-감독 불일치를 줄인다.

- **Empirical Impact**: ShapeSplat의 3DGS 데이터로 사전학습한 뒤 다양한 downstream에서 전이성과 강건성을 검증했으며, ablation을 통해 이미지/텍스트 감독과 GSHM이 성능에 유의미하게 기여함을 확인한다. 특히 ModelNet40과 ScanObjectNN(PB-T50-RS)에서 Gaussian-MAE 대비 각각 0.61%, 3.85% 성능 향상을 보이며 전이 성능이 개선됨을 강조한다. 결과적으로 3DGS가 렌더링 중심을 넘어 멀티모달 3D 사전학습을 위한 확장 가능한 표현 공간으로 활용될 수 있음을 실험적으로 뒷받침한다.



### Few-Medoids: An Embarrassingly Simple Coreset Selection Method for Few-Shot Knowledge Distillation (https://arxiv.org/abs/2607.05891)
Comments:
          Accepted at KES 2026

- **Prior Approaches**: 코어셋 선택은 대규모 데이터에서 일부를 골라 학습 효율을 높이려는 방법이지만, 지식 증류(KD)로 옮기면 무작위(random) 선택이 강력한 기준선이 되는 등 성능 개선이 어렵다는 문제가 지적된다. 특히 기존의 herding, k-center Greedy 같은 coreset 방법을 KD 파이프라인에 적용해도 랜덤 대비 확실히 이기지 못하는 경우가 많았다.

- **Core Contribution**: 본 논문은 few-shot KD 상황에서 teacher의 잠재공간을 이용해 클래스별 ‘중심 샘플’을 고르는 training-free 코어셋 방법 few-medoids를 제안한다. 각 클래스에서 teacher feature 기준으로 같은 클래스 내 평균 L2 거리(중앙성)가 가장 작은 샘플들을 우선 선택해, 수업 신호가 되는 대표 예제를 만든다.

- **Technical Challenges**: 핵심 난제는 KD용 코어셋이 단순 대표성만이 아니라 student가 teacher 신호를 잘 흡수할 수 있는 ‘학습 가능한’ 샘플을 골라야 한다는 점이다. 저자들은 클래스별 teacher feature 공간에서 기하학적 중심(medoid)을 직접 근사하도록 점수(평균 거리)만 계산하는 방식으로 복잡한 휴리스틱 없이 이를 해결했고, 이후 표준 soft-label KD 손실에 결합해 학습을 수행한다.

- **Empirical Impact**: CIFAR-10/100, Oxford Flowers 102, Food-101 등 4개 데이터셋과 여러 교사-학생 조합(ResNet·ViT)을 대상으로 실험했으며, few-medoids는 대부분의 k(클래스당 샘플 수) 구간에서 무작위 및 기존 코어셋 기법을 일관되게 능가했다. 예외는 teacher→student 전이 설정(특히 ViT-B/16→ViT-Small)에서 herding이 유리해지는 경우로, 결과적으로 few-medoids는 특히 student를 from scratch로 학습할 때 강하고, 단순하지만 drop-in baseline으로 활용 가능하다는 시사점을 준다.



### Harrison.Rad 1.5 Technical Report: A radiology foundation model that can draft reports from images, priors and clinical contex (https://arxiv.org/abs/2607.05880)
- **Prior Approaches**: 기존 방사선 AI는 주로 특정 소견을 탐지해 radiologist의 일부 인지 단계만 보조하는 방식이 많았고, 보고서 작성에 필요한 문맥 통합(병력·이전 검사·진단적 추론)을 end-to-end로 다루기엔 한계가 컸습니다. 또한 대형 범용 비전-언어 모델은 여러 데이터에 강해도 복잡한 임상적 판단에서 도메인 밀도 지식이 부족해 FRCR 같은 인증 수준 평가에서 흔들릴 수 있다는 문제가 드러났습니다. 무엇보다 기존 평가지표가 문장 유사도 중심이어서, 실제 임상적 맞고-틀림(특히 polarity와 진단 부합)을 안정적으로 가르지 못한다는 평가 공백이 지적됩니다.

- **Core Contribution**: 이 논문은 방사선 특화 multimodal large language model HR1.5를 제시해, interleaved 텍스트-이미지 입력을 받아 structured/ unstructured 리포트를 함께 생성하는 것을 목표로 합니다. 핵심은 보고서 작성에 필요한 소견-진단 정렬을 강화하고, 방사선 업무에 맞춘 학습 파이프라인과 임상적으로 정렬된 평가 프레임을 함께 제공한다는 점입니다. 또한 신뢰도(calibrated confidence)와 근거(해당 영역 근거 등) 분석까지 포함해 향후 임상 적용을 위한 책임 있는 평가 방향을 제안합니다.

- **Technical Challenges**: HR1.5가 실제로 “방사선답게” 작동하려면, (1) 도메인에 맞는 언어 생성 능력, (2) 시각-텍스트 정렬의 세밀함, (3) 문맥 기반의 instruction-following까지 한 번에 맞추는 학습 설계가 필요합니다. 이를 위해 3단계 파이프라인(보고서 기반 domain adaptation → 6M 규모의 대조학습에서 curriculum 기반 hard negatives → multi-turn visual question answering fine-tuning)을 사용하고, 방사선에서 중요하지만 표면 어휘에 의존하기 쉬운 구분을 네거티브와 대화 데이터로 “기계적으로” 학습신호화합니다. 여기에 captioning 보조목표는 방사선에서는 효과가 없다는 ablation 결과를 반영해 제외했으며, B200급 학습 인프라에서는 데이터 로딩/캐싱 병목을 재구성해 대규모 학습 효율을 확보했습니다.

- **Empirical Impact**: 평가는 RadGraph-XL 기반 Findings-Diagnosis 점수(ontology 동의어 매칭, polarity-contradiction 탐지)로 임상 정렬을 강화해 RadBench, FRCR 2B Short Case 시뮬레이션(Angoff 기준), ReXGradient, 내부 다중 신체부위 세트, mammography(CBIS-DDSM) 등을 폭넓게 수행합니다. 그 결과 HR1.5는 시뮬레이션 FRCR passing 기준을 충족하는 유일한 시스템으로 보고되며, closed-format 임상 질의에서는 전반적으로 최고 정확도를 보입니다. 또한 explainability(질문 민감 Grad-CAM, attention 분석, confidence estimation)와 리포트 품질을 임상 근거로 평가하는 프레임을 함께 제시해, 단순 자동지표를 넘어 실제 임상 사용을 염두에 둔 평가 체계를 확장합니다.



### AVA-VLM: Adaptive Visual Attention-Vision Language Model for In-the-Wild Construction Site Monitoring (https://arxiv.org/abs/2607.05859)
- **Prior Approaches**: 기존 건설 현장 특화 VLM들은 ConstructionSite10K류 데이터로 사전학습 VLM을 QA 형태로 SFT(종종 LoRA 등 PEFT)하는 방식이 중심이었다. 다만 단일 전역 이미지만 보고 답하도록 학습되어, 원거리/소형 물체가 보이는 wide-view 환경에서 성능이 떨어지기 쉽고, 해상도 저하 입력에 취약하며, 전체 이미지를 촘촘히 처리해 시각 토큰 기반 지연도 커진다는 문제가 있었다.

- **Core Contribution**: 이 논문은 AVA-VLM(Adaptive Visual Attention-Vision Language Model)을 제안하며, 인간처럼 전역(저해상도) → 필요할 때만 로컬(고해상도) 크롭으로 넘어가는 coarse-to-fine 시각 주의 전략을 end-to-end로 구현한다. 또한 region-aware Chain-of-Thought(CoT) 데이터셋을 만들어 “언제 점검이 필요한지, 어디를 크롭할지, 로컬 증거를 추론에 어떻게 쓰는지”를 학습시킨다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 전역 저해상도만으로는 판단 불충분한 상황을 모델이 스스로 감지해야 하고, (2) 로컬 크롭을 요청/통합하는 추론 흐름이 실제로 신뢰성 있게 동작해야 하며, (3) 그 과정에서 불필요한 visual-token 처리를 줄여 효율을 유지하는 것이다. 논문은 전역부터 먼저 답을 시도하고, 상세 확인이 필요할 때에만 선택적으로 로컬 크롭을 넣어 CoT에 결합하도록 학습 신호(크롭 시점·위치)를 region-aware CoT로 설계해 이를 해결했다.

- **Empirical Impact**: 실험 결과 AVA-VLM은 장거리 및 reduced-resolution 조건에서 신뢰성이 개선되었고, 동시에 시각 토큰 사용량을 크게 줄였다. 즉, 건설 안전 모니터링에서 요구되는 원거리 관찰과 엣지 배치 제약을 함께 고려할 때 기존 구성보다 더 실용적인 추론-비용 절감 효과를 보였다는 점에서 의미가 있다.



### Breaking Spurious Correlations via Generative Randomization and Cross-Variant Self-Supervised Learning (https://arxiv.org/abs/2607.05850)
Comments:
          Accepted at CVPR Workshop 2026 GCV

- **Prior Approaches**: 기존 ERM 기반 비전 모델은 학습·테스트 분포가 비슷할 때는 잘 맞지만, 배경처럼 상관적(스퓨리어스) 단서에 기대는 경향 때문에 분포 이동에서 급격히 성능이 떨어집니다. 이를 완화하려는 GroupDRO, JTT 같은 방법은 그룹 라벨 품질이나 초기 최적화 동역학에 민감하고, 생성 기반 증강은 모델이 여전히 배경 민감 표현을 유지할 여지가 큽니다. 즉, “생성 데이터로 커버리지는 늘리되” 인바리언스를 직접 강제하지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 생성적 개입(generative intervention)과 표현 수준 인바리언스 학습을 결합한 2단계 프레임워크를 제안합니다. 첫 단계에서 zero-shot segmentation으로 인과적(전경) 객체를 분리한 뒤 구조 보존 diffusion inpainting으로 배경만 바꾼 컨텍스트-시프트 변형을 만들고, 둘째 단계에서 같은 객체의 서로 다른 배경 변형들을 positive pair로 묶는 Cross-Variant Self-Supervised Learning으로 배경 비의존 표현을 학습합니다. 이후 ERM warm-up과 layer-wise learning rates를 사용한 GroupDRO fine-tuning으로 worst-group 일반화를 강화합니다.

- **Technical Challenges**: 핵심 과제는 “객체 정체성은 유지하면서 배경만 의미 있게 바꾼 샘플”을 안정적으로 만들고, 그 생성물을 단순 증강이 아니라 인바리언스 학습 신호로 연결하는 것입니다. 이를 위해 Grounding DINO+SAM으로 전경 마스크를 얻고, FLUX.1-Fill 기반 구조 보존 inpainting으로 마스크 바깥 영역만 다양한 텍스트 프롬프트에 따라 재합성한 뒤, 동일 객체 변형 간 대조학습(contrastive objective, NT-Xent)으로 객체 중심 정렬을 강제합니다. 또한 GroupDRO의 초기 불안정성을 줄이기 위해 ERM warm-up 뒤에야 adversarial group weight 업데이트를 활성화하고, fine-tuning 중 표현 손상을 막기 위해 layer-wise learning rates를 적용합니다.

- **Empirical Impact**: 실험은 Waterbirds, MetaShift, NICO++에서 worst-group 기준 성능을 중심으로 입증되었고, Waterbirds 92.5%, MetaShift 81.7%, NICO++ 87.4%의 worst-group 정확도를 보고합니다. 평균 정확도도 Waterbirds 95.4%, MetaShift 82.6%, NICO++ 94.0%로 높은 수준을 유지하며, 기존 ERM 대비 worst-group에서 큰 개선(예: Waterbirds에서 +21.7%p)을 보였습니다. 특히 단순 생성 증강만으로는 성능이 떨어지고(Cross-Variant SSL 미적용 시 하락), layer-wise learning rates와 ERM warm-up이 worst-group 성능에 결정적임이 애블레이션에서 확인되어, “생성=증강”이 아니라 “생성=인바리언스 학습”으로 설계해야 한다는 메시지를 강화합니다.



### Realistic Compound-Lens Defocus Blur Synthesis (https://arxiv.org/abs/2607.05837)
Comments:
          GitHub: this https URL

- **Prior Approaches**: 기존 디포커스(Defocus) 디블러링은 디포커스 맵을 추정한 뒤 비블라인드 디컨볼루션을 하거나, end-to-end 네트워크로 공간변화 블러를 직접 학습하는 방식이 주류다. 하지만 학습 성능은 결국 학습 데이터의 렌즈 다양성과 물리적 사실성에 크게 좌우되며, 현실 캡처 데이터는 카메라/렌즈 확장이 어렵고 밝기·정렬·잔여 디포커스 같은 불일치가 생긴다.

- **Core Contribution**: 이 논문은 다양한 복합 렌즈(compound lens)에서 현실적인 디포커스 블러 디블러링 데이터를 합성하는 통합 파이프라인을 제안한다. Debye CZT 기반의 효율적 wave-optics PSF 계산, 깊이 레이어드 렌더링의 occlusion 처리, 그리고 radiometrically linear 공간에서 ISP까지 반영한 블러 합성을 통해 렌즈 특성이 다른 대규모 데이터셋 CLDefocus(40k/1k/1k 쌍)를 만든다.

- **Technical Challenges**: 핵심 기술 난제는 (1) wave-optics PSF가 조밀한 샘플링에 민감해 계산이 부담된다는 점, (2) 깊이에 따라 변하는 PSF를 픽셀 단위로 직접 합성하면 비용이 폭증한다는 점, (3) sRGB 같은 비선형 공간에서의 단순 합성은 광도 일관성을 깨뜨릴 수 있다는 점이다. 저자들은 Debye CZT에 명시적 샘플링 기준을 두고 ROI/해상도는 CZT로 유연하게 선택하며, signed CoC를 기준으로 깊이를 층화해 레이어 합성으로 occlusion까지 반영하고, radiometrically linear 공간+camera ISP 시뮬레이션으로 광도 현실성을 맞춘다.

- **Empirical Impact**: CLDefocus로 학습한 모델은 cross-device 일반화에서 기존 현실/합성 데이터 학습 대비 개선을 보이며, full-reference에서는 오히려 DPDD 학습 모델이 점수가 높게 나오는 경우도 분석한다. 이는 실제 GT에 존재하는 기하·광도 불일치가 PSNR/SSIM 같은 픽셀 단위 지표를 편향시키기 때문이며, 반대로 no-reference 지표와 시각적 복원 품질은 CLDefocus 학습 모델이 전반적으로 더 좋은 경향을 보인다.



### Complementary Roles of Image Classification and Vessel Segmentation in AI-Based Screening for Retinopathy of Prematurity Plus Disease in a Kenyan Preterm Cohor (https://arxiv.org/abs/2607.05825)
- **Prior Approaches**: 기존 ROP(미숙아 망막병증) 자동 선별 연구는 주로 화상 분류기(RGB 분류) 중심으로 진행돼 왔지만, Plus 질환 특성상 “주관적·가변적” 판독 문제가 남아 과잉 의뢰(over-referral) 위험이 커질 수 있다. 또한 안저 혈관 분할(segmentation) 기반 접근은 정밀도를 올릴 잠재력이 있으나, 실제 선별 워크플로에서 분류 성능과의 균형을 맞추기 어렵다는 한계가 보고돼 왔다.

- **Core Contribution**: 본 연구는 케냐 데이터에서 ROP Plus를 눈 단위로 탐지하기 위해 혈관 분할과 분류를 함께 묶는 end-to-end 또는 결합형 파이프라인을 체계적으로 비교한다. 핵심은 “세분화(혈관) 신호”를 직접 활용해 분류기의 민감도는 유지하되, 특이도를 끌어올리는 조합 전략을 제시한 점이다.

- **Technical Challenges**: Plus 판독의 근거가 되는 혈관의 확장·꼬임은 영상 내에서 복잡하게 나타나며, 모델 학습에서는 혈관 분할 품질과 최종 눈 단위 판정의 연결이 관건이다. 연구진은 두 명의 grader가 제공한 혈관 어노테이션으로 분할 학습을 뒷받침하고, patient-grouped nested cross-validation으로 데이터 누수를 줄이며 11가지 구성(분류, multiple-instance learning, multi-task segmentation-classification, segment-then-classify 등)을 비교해 최적 조합을 찾았다.

- **Empirical Impact**: 결과적으로 혈관 분할은 held-out 이미지에서 Dice 0.533, IoU 0.368, sensitivity 0.623, specificity 0.979 수준을 보여 “분할 자체는 가능”함을 확인했다. 분류 단독은 민감도가 높지만 과잉 의뢰가 컸고, 분할과 결합한 모델이 더 특이적이었으며, OR 기반 screen(민감도), AND 기반 confirmation(특이도), probability ensemble(균형)이 각각 장점을 보였다—probability ensemble은 sensitivity 0.692, specificity 0.914, balanced accuracy 0.803으로 분류기 단독을 능가했다. 연구는 아프리카 ROP AI 시스템이 복합 워크플로로 설계하고 prospective multi-site validation을 거쳐야 한다는 실무적 방향성을 강화했다.



### TRIG: Trajectory-Rig Decoupled Metric Geometry Learning (https://arxiv.org/abs/2607.05801)
Comments:
          9 pages, 3 figures, 8 tables

- **Prior Approaches**: 기존 비전 기반 3D 기하 인식은 카메라 자세·깊이·3D를 함께 학습하거나, BEV/옥젤 기반으로 공간을 표현하는 방식이 주를 이뤘습니다. 다만 멀티캠에서 카메라 자세를 하나의 얽힌 조건으로 다루면 시간에 따른 ego-motion과 정적인 camera-rig 토폴로지가 같이 모델링되어 metric scale이 암묵적으로 추정되기 쉽습니다.

- **Core Contribution**: TRIG는 Trajectory-Rig Decoupled Metric Geometry Learning으로, 카메라 포즈를 ego-trajectory(시간 가변)와 camera-rig(정적)으로 명시적으로 분해해 metric-aware 학습을 가능하게 합니다. 이 분해를 통해 차량 측 기하 prior를 motion과 토폴로지 경로로 분리해 주입하고, 글로벌 좌표 정합 후처리 부담을 줄이는 방향으로 설계했습니다.

- **Technical Challenges**: 핵심 난제는 분해된 포즈를 모델이 실제로 metric 일관성을 유지하며 활용하도록 만드는 것입니다. TRIG는 decoupled pose encoding/pose supervision으로 trajectory는 시간 일관성, rig은 동시 카메라 간 제약에 각각 맞춰 학습시키고, Sparse Temporal–Spatial Attention(STSA)로 카메라 간 상호작용과 시간 집계를 분리해 장거리 추론의 계산비용도 억제합니다.

- **Empirical Impact**: 5개 자율주행 벤치마크 실험에서 TRIG는 pose estimation, metric depth prediction, 3D reconstruction 전반에 걸쳐 state-of-the-art를 달성했습니다. 특히 엔트angled prior 기반 모델이 wide-baseline 멀티캠에서 붕괴하는 현상을 decoupled 설계와 supervision 분리로 완화해, 드리프트에 강한 metric pose와 재구성 품질을 보여줬다는 점이 의미 있습니다.



### Segmentation before Answering: Pixel Grounding for MLLM Visual Reasoning (https://arxiv.org/abs/2607.05798)
- **Prior Approaches**: 기존 MLLM의 “thinking with images” 계열은 관심 영역을 bounding box(BBox)로 표시한 뒤 크롭해 추론하는 방식이 널리 쓰였습니다. 하지만 BBox는 객체의 불규칙한 형태를 잘 담지 못해 배경 토큰 낭비가 생기고, 겹치는 객체에서는 목표와 주변을 분리하지 못해 semantic interference가 발생할 수 있습니다.

- **Core Contribution**: 이 논문은 Segmentation before Answering(SegAnswer)로, zoom-in의 단위를 BBox가 아닌 pixel-level segmentation mask로 바꿉니다. SegAnswer는 segmentation으로 목표 영역만 정밀 분리해 불필요한 배경·간섭 신호를 줄이고, 분할된 패치가 MLLM의 positional embedding 구조와도 더 자연스럽게 맞물리도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트 지시를 pixel 단위 마스크로 연결하는 grounding 능력과 (2) 분할 마스크를 중간 대화 단계로 쓰는 멀티모달 interleaved 추론을 함께 학습하는 것입니다. 이를 위해 3단계로 진행하며, 먼저 SAM 2.1 기반 mask decoder를 붙여 segmentation 능력을 길러두고, 다음으로 <|seg|> 토큰을 사용해 segmentation을 대화의 도구처럼 넣는 SFT를 수행한 뒤, 마지막으로 ground truth 없이 최종 결과 기반 reward로 DAPO RL을 적용해 추론 전략을 강화했습니다.

- **Empirical Impact**: V*·HR-Bench(4K/8K) 같은 고해상도 지각, MMBench·VisuLogic·MMVP 같은 일반 지각, POPE·Hallusionbench 같은 환각 평가 전반에서 일관된 성능 향상을 보였고, BBox 기반 visual reasoning 대비 특히 고해상도에서 격차가 컸습니다. 또한 RefCOCO/RefCOCO+/RefCOCOg 등 referring segmentation 벤치마크에서도 segmentation 특화 방법을 능가하는 결과를 제시해, SegAnswer의 pixel grounding이 신뢰 가능함을 실증했습니다.



### DeSeG: Decoupling Semantic Intent and Geometric Constraints for Physically Plausible Human-Scene Interaction (https://arxiv.org/abs/2607.05787)
- **Prior Approaches**: 기존 Human-Scene Interaction(HSI) 생성은 언어 조건과 목표 위치(골/goal)를 한 모델에 함께 넣어 end-to-end로 모션을 뽑는 방식이 많았다. 그러나 데이터에서 공간 제약이 특정 행동과 강하게 동반되면서 semantic-geometric entanglement(의미-기하 얽힘) 문제가 생겨, 지오메트리 단서가 텍스트 의도를 덮어쓰는 shortcut learning이 발생한다. 또한 학습은 확률 분포를 따라가지만 물리 법칙을 직접 보장하지 못해 body-scene penetrations 같은 physical hallucination이 이어지고, 이를 줄이려는 post-processing/추론 시 guidance는 비용이 커지는 한계가 있었다.

- **Core Contribution**: DeSeG는 HSI 합성을 hierarchical(계층형)으로 분해해 ‘무엇을/어떻게 할지’(semantic intent)와 ‘어디로 갈지/환경 제약’(geometric constraints)을 명시적으로 분리한다. Residual Semantic Planner가 텍스트와 정규화된 목표(goal) voxel을 compact한 latent affordance z로 압축해 의미 제어를 강화하고, Physics-Regularized Diffusion Executor가 해당 latent를 물리적으로 가능한 모션으로 풀어낸다. 이를 통해 지오메트리와 언어가 충돌하는 상황에서도 의미 정렬과 물리 일관성을 동시에 노린다.

- **Technical Challenges**: 핵심 기술적 난제는 얽힘을 구조적으로 끊어내야 한다는 점이다: 단순 loss-level 정규화만으로는 goal 위치 신호의 그라디언트 지배를 막기 어렵기 때문이다. DeSeG는 canonicalization으로 절대 방향/공간 누수를 줄이고, residual CVAE 기반 Planner로 의미를 latent 병목에 통과시켜 ‘trajectory-free’한 방식으로 실행기(디퓨전)에 넘긴다. 물리환각을 줄이기 위해선 디퓨전 학습 중에 충돌 회피를 내재화해야 하는데, DiT의 denoising objective에 differentiable repulsive potential field(반발 퍼텐셜)를 통합해 안전 임계치를 넘을 때만 패널티가 작동하도록 설계했다.

- **Empirical Impact**: Lingo 데이터에서 DeSeG는 mean scene penetration을 47% 줄이고 semantic alignment를 29% 개선하는 등 SOTA를 달성했다. 특히 분포 매칭 기반 지표가 놓치기 쉬운 semantic-geometric conflict 상황을 겨냥한 NC-Bench에서 Semantic-Geometric Consistency(SGC)를 72.3%로 끌어올려, 기존 방법의 shortcut 편향이 깨지는 것을 사용자 평가로 확인했다. 또한 TRUMANS로의 out-of-distribution 전이에서도 FID와 penetration이 유의미하게 개선되어, 계층형 latent 분리와 physics-regularized diffusion이 다양한 장면 분포에서 강건함을 보였다는 점에서 의미가 크다.



### Benchmarking the Robustness of Autonomous Driving to Environmental Illusions: A Lane Perception Perspectiv (https://arxiv.org/abs/2607.05783)
Comments:
          Accepted by IEEE TPAMI 2026

- **Prior Approaches**: 기존 연구는 주로 CULane 같은 표준 LD 데이터셋에서의 성능을 다뤄왔고, adverse weather·motion blur·synthetic noise처럼 “전역 품질 저하” 중심의 robustness만 점검하는 경우가 많았습니다. 반면 그림자·반사·타이어 자국처럼 자연적으로 생기지만 형태가 “차선처럼 보이게” 만드는 환경적 착시(environmental illusions)는 체계적으로 평가되지 않아 안전 공백이 남아 있었습니다.

- **Core Contribution**: 이 논문은 차선 인식(lane perception) 관점에서 AD robustness를 정면으로 다루고, 착시가 차선 탐지와 비전-언어 기반 ADVLM의 추론을 동시에 흔드는 문제를 구체화합니다. 이를 위해 최초의 벤치마크 LanEvil++를 제안하며, 14종 착시를 CARLA에서 생성해 3D 장면 편집 가능성과 대규모 평가 세트를 제공합니다.

- **Technical Challenges**: 착시는 전역 잡음과 달리 위치·사례 의존적이며, 수집·다양화가 어렵기 때문에 시뮬레이션 기반으로 “원인-시각 패턴”을 통제해 재현해야 했습니다. 논문은 CARLA 환경에서 정적 인프라(도로 손상), 동적 방해(교통 참여자), 조명 조건에 따른 그림자, 노면 반사 등을 5단계 severity로 파라미터화해, 고충실도 2D 렌더링과 라벨/QA까지 한 번에 구축했습니다.

- **Empirical Impact**: 실험 결과 착시는 최신 LD 성능을 평균 Accuracy 5.27%, F1-score 10.49% 하락시키며, ADVLM도 GPT-score 2.03%와 Language-score 0.75%를 떨어뜨렸습니다. 특히 그림자(shadow)가 LD에 가장 치명적(Accuracy 최대 -7.20%)이었고, closed-loop 시뮬레이션에서는 잘못된 주행 의사결정으로 사고로 이어질 수 있음을 보여주었으며, 방어 모델 MIDA는 어려운 조건에서 LD는 +4.23%, ADVLM은 +3.82%의 robustness 개선을 달성했습니다.



### LEGATO 2: Toward Multimodal Sheet Music Recognition and Understanding (https://arxiv.org/abs/2607.05769)
Comments:
          23 pages. Equal contribution: Guang Yang and Brian Siyuan Zheng

- **Prior Approaches**: 기존 광학 음악 인식(OMR)은 보통 한 장의 악보 이미지를 단일 입력으로 보고 처리해, 입력 길이가 길어질수록 성능과 확장성이 급격히 떨어질 수 있었다. 또한 대부분은 기호 중심의 전사에 초점을 맞춰 제목이나 주석 같은 텍스트가 섞인 경우를 생성형으로 다루기 어려웠다. 결과적으로 장거리 문서 처리와 텍스트 포함 전사의 결합이 한계로 지적됐다.

- **Core Contribution**: 이 논문은 악보 이미지에서 기호 표기와 의미 정보를 함께 뽑아내는 파이프라인 Legato 2를 제안한다. Legato 2는 시스템 단위로 순차 처리하는 최초의 대규모 OMR 모델로, 페이지를 무차별 이미지로 취급하지 않고 표기 읽기 흐름을 따라 임의로 긴 입력으로 스케일링한다. 더불어 제목과 주석 같은 내장 텍스트를 포함하는 기호 전사를 생성하는 최초의 OMR 모델을 제시한다.

- **Technical Challenges**: 핵심 과제는 (1) 긴 악보에서 시스템 구조를 안정적으로 분할하고, (2) 시스템 간 문맥을 유지하며, (3) 기호와 텍스트를 동시에 정확히 생성하는 것이다. Legato 2는 시스템-level segmentation과 autoregressive vision-LM을 결합해, 국소적인 기보 디테일과 스코어 구조를 함께 포착하도록 설계했으며, system-by-system 순차 모델링으로 긴 입력에서도 일관성을 확보한다.

- **Empirical Impact**: 여러 데이터셋에서 Legato 2는 기존 state of the art를 일관되게 능가하는 성능을 보였다. 또한 생성된 상징적 전사가 비주얼 입력과 함께 frontier language model의 해석을 돕는다는 점을 실증해, 촘촘한 음악 문서를 이해하는 downstream 성능까지 끌어올렸다. 결과적으로 Legato 2는 OMR 자체뿐 아니라 악보 이해 전반에서 새로운 state-of-the-art를 정립했다.



### Image2Sim: Scaling Embodied Navigation via Generative Neural Simulator (https://arxiv.org/abs/2607.05765)
- **Prior Approaches**: 기존 Embodied navigation은 Matterport3D, HM3D, Gibson, Replica 같은 real scan 기반 시뮬레이션과 Habitat 같은 툴에 크게 의존해 왔다. 다만 스캔은 비용과 노동이 커서 확장성이 낮고, procedural/synthetic 환경은 자산·레이아웃·렌더링 통계의 현실성이 떨어져 sim-to-real gap이 자주 발생한다.
또한 NeRF나 3D Gaussian Splatting 계열은 고충실 렌더링을 제공하지만 per-scene 최적화가 필요해 대규모 데이터 엔진으로 쓰기 어렵고, 생성형 모델은 collision·rigid-body 일관성 같은 닫힌루프 상호작용의 물리적 구조를 지속적으로 보장하기 힘들다.

- **Core Contribution**: 이 논문은 Image2Sim으로, posed RGB-D 이미지 시퀀스에서 고품질 interactive 3D 환경을 실시간(neural)으로 구성하는 프레임워크를 제안한다. 핵심은 3D spatial anchoring(기하적 고정)과 photorealistic observation synthesis(관측 생성)를 분리해, 스케일과 물리적 근거, 시각 충실도의 균형을 맞추는 것이다.
또한 단순 렌더러를 넘어 collision-aware 모션 엔진과 VLM 기반 지시문 생성까지 결합해, 약 20K 씬에서 10M+ navigation 학습 샘플을 자동 생성하는 embodied data engine으로 확장한다.

- **Technical Challenges**: 문제의 기술적 난점은 (1) sparse/noisy 관측에서 구조적 빈틈을 채우면서도 (2) 3D grounding을 유지해 홀루시네이션을 억제하고 (3) 닫힌루프에서 실행 가능한 물리 궤적을 제공하는 것이다.
Image2Sim은 feed-forward feature Gaussian으로 depth로부터 3D feature-Gaussian을 단일 패스로 들어 올리고, 렌더링은 Geometry-Aware One-Step Pixel Flow로 alpha(불확실도) 기반 gated prior를 조건 삼아 panoramic RGB-D를 한 스텝에 생성한다.
여기에 재현 가능한 경로를 위해 가시성/장애물/여유공간을 반영한 traversable voxel connectivity graph를 만들고, 그래프 계획+컨트롤러로 궤적을 생성한 뒤 이를 매크로-스텝으로 분절해 자연어 instruction을 Qwen3-VL-32B-Instruct로 정렬 주석한다.

- **Empirical Impact**: 실험에서는 20K 규모 interactive environment에서 10M+ 샘플을 만들었고, 렌더링 품질·속도 면에서 panoramic variant가 noisy depth 상황에서도 높은 PSNR/SSIM을 유지하며 실시간에 가까운 FPS를 달성한다. 또한 Gaussian-only 및 순수 생성형 비교군은 계산 지연이 크거나 미관적으로 그럴듯하지만 구조 일관성이 떨어져 효율-견고성의 균형에서 밀린다.
Navigation 학습은 Image2Sim 내부에서만 진행한 뒤 Habitat 도메인에서 zero-shot 전이 평가를 수행했으며, 그 결과 R2R-CE, RxR-CE, REVERIE-CE 벤치마크에서 새로운 SOTA 수준의 향상과 강한 도메인 전이를 보였다. 이는 스케일 가능한 neural simulation이 embodied navigation의 실용적인 학습 기판이 될 수 있음을 경험적으로 뒷받침한다.



### Optimized Adaptive Loop Filter in Versatile Video Coding (https://arxiv.org/abs/2607.05737)
Comments:
          This paper was submitted to DCC 2021 and accepted as a poster

- **Prior Approaches**: VVC의 ALF는 blocking, ringing, blurring 같은 압축 아티팩트를 줄이기 위해 GALF와 CCALF를 in-loop filter로 사용한다. 다만 encoder에서 파라미터 학습과 CTU 의사결정이 복잡하고, 특히 CCALF는 multi-pass로 picture buffer 접근이 매우 많아 외부 메모리 대역폭·전력·지연 문제를 키운다. GALF는 일부 병렬화 여지가 있으나, CCALF와의 의존성 때문에 실시간 인코더에서 병렬 수행이 제한된다는 한계가 있었다.

- **Core Contribution**: 논문은 ALF의 encoder 복잡도를 줄이기 위해 (1) GALF와 CCALF의 병렬 설계, (2) GALF의 adaptive parameter decision, (3) CCALF를 one-pass로 바꾸는 프레임워크를 제안한다. CCALF 학습에서 사용되는 크로마 기준을 GALF 이전 신호로 대체해 두 모듈의 병렬 실행 가능성을 높였고, CCALF는 필터 연산 없이 distortion을 추정해 다중 패스를 제거한다. 이를 통해 외부 메모리 접근 부담을 극단적으로 낮추면서도 성능 저하는 미미하다고 주장한다.

- **Technical Challenges**: 핵심 기술적 난제는 CCALF가 필요로 하는 auto-correlation matrix와 cross-correlation vector 계산 및 필터 기반 왜곡 산정이 기존에는 버퍼 접근을 수반해 multi-pass를 강제한다는 점이다. 논문은 CTU별로 필요한 행렬·벡터를 먼저 pre-calculate하고, filtered 이미지를 만들지 않고도 MSE를 계산할 수 있는 filtering distortion estimation 수식을 구성해 one-pass 학습이 가능하게 했다. 또한 GALF의 RDO 병목(S3/S4)을 위해 luma filter 개수(maxLumaFilters)를 QP-기반 모델로 예측하고, CTU 레벨 filter 후보를 stage1 결과로 축소해 불필요한 탐색을 줄였다.

- **Empirical Impact**: VTM-8.0 대비 제안 방식은 RA 구성에서 ALF 모듈 기준 약 25% 시간 절감을 달성하면서도 코딩 성능 변화(Bjontegaard BD-rate)는 negligible 수준으로 보고된다. 특히 picture buffer 접근 수를 152에서 1로 줄여, 실무에서 문제가 되는 external memory 지연·전력 이슈를 완화할 수 있는 방향성을 제시한다. 이 접근 중 일부는 VVC reference software에 채택되었다고 밝힌다.



### ARMS: Anchor-Relational Motion Streaming for Seamless Solo-Social Motion Transitions (https://arxiv.org/abs/2607.05733)
Comments:
          Accepted by ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 text-to-motion 연구는 주로 고정 길이 모션 클립을 단일 패스로 생성하거나, 솔로/상호작용을 서로 다른 프레임워크로 취급해 왔습니다. 그 결과 점진적(incremental) 스트리밍 생성이나 솔로-소셜 모드 전환에서 경계 불연속, 관계 드리프트가 발생하기 쉽습니다. 또한 일부 long-horizon 방법이 존재하지만, 대부분은 단일 에이전트 중심이라 one-/two-person 구성 변화까지 자연스럽게 통합하기 어렵습니다.

- **Core Contribution**: ARMS(Anchor-Relational Motion Streaming)는 솔로 모션과 인간-인간 상호작용을 하나의 causal 생성 과정에서 통합해, 텍스트가 시간에 따라 갱신되는 스트리밍 환경을 목표로 합니다. Anchor-Relational 동역학 비대칭 표현을 도입하고, 파트너 기준 relative-translation 항으로 소셜 결합을 활성/비활성 전환해 모드 스위칭의 끊김을 줄입니다. 이를 통해 긴 시간 동안 각 개인의 시간 일관성과 에이전트 간 공간 기하를 동시에 유지하는 것을 지향합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 미래 정보 없이 causal로 길게 생성하면서도 (2) 개인 내부 타이밍 의존성과 (3) 상호작용 시의 상대 정렬을 함께 안정화하는 것입니다. ARMS는 causal temporal autoencoder로 모션을 temporally causal latent space로 압축하고, segment-wise causal relational diffusion model이 과거 컨텍스트만으로 구간 단위로 점진 정제를 수행하도록 설계했습니다. 또한 mode-aware relational gating으로 cross-agent 연결을 마스킹해 솔로/상호작용 생성 모두를 단일 모델로 처리하고, variable-length history conditioning으로 long-horizon 일관성을 보강합니다.

- **Empirical Impact**: 실험에서는 HumanML3D(솔로)와 InterHuman(상호작용)을 같은 Anchor–Relational 표현으로 변환해 스트리밍 생성까지 평가하며, InterX에서도 교차 스켈레톤 일반화를 확인합니다. 정량적으로 ARMS는 상호작용 전환에서 더 부드러운 전개와 높은 social coherence를 보였고, streaming 조건에서도 사실성/정렬 품질(FID, MM Dist 등)을 유지하는 경향을 보입니다. 특히 streaming 모델이 InterX의 모든 지표에서 우수한 성능을 보이며, 고정 길이 클립을 넘어선 긴 상호작용 합성의 실용성을 입증합니다. 



### SAMPLe: SAM-based Optimizer for Prompt Learning in VLMs (https://arxiv.org/abs/2607.05727)
Comments:
          The manuscript has been accepted to ECCV and will be presented at the conference and published in the main proceedings

- **Prior Approaches**: CLIP 같은 사전학습 VLM에서 프롬프트 학습은 본체 파라미터를 고정한 채 작은 학습 가능한 프롬프트로 태스크에 적응하지만, seen 분포에서는 잘 맞아도 unseen 일반화가 자주 떨어지는 성능-일반화 딜레마가 발생한다. 선행 연구들은 정규화, 다양성/표현 보강, 그라디언트 조작 등으로 generalization을 지키려 했으나, 프롬프트의 작은 학습 공간 때문에 샤프한(고곡률) 손실 최솟값으로 수렴해 과적합 위험이 커진다. 또한 SAM 계열을 그대로 쓰면 full-batch 관점이 탐색성을 약화시켜, 때로는 SGD보다도 일반화가 나빠질 수 있다는 한계가 보고돼 있다.

- **Core Contribution**: 이 논문은 SAMPLe(Sharpness-Aware Minimization Prompt Learning)이라는 플러그인형 샤프니스 인식 최적화기를 제안한다. SAMPLe는 프롬프트 학습에서 (1) 학습 손실을 충분히 낮추는 exploitation과 (2) 손실 지형을 더 평탄하게 만드는 exploration을 매 반복마다 동시에 맞추도록 설계됐다. 특히 sharpness-aware 업데이트를 단순 적용하는 대신, 현재 최적화 상태의 국소 곡률/그라디언트 성질을 반영해 일반화 손실 지형을 목표로 한다.

- **Technical Challenges**: 핵심 기술 도전은 프롬프트의 제한된 파라미터 때문에 샤프 최소로의 쏠림이 심해지고, 동시에 SAM의 perturbation 계산이 탐색성을 훼손할 수 있다는 점이다. SAMPLe는 perturbed point에서의 그라디언트가 ERM 그라디언트와의 정렬(성능 유지)을 따르면서, full-batch 성분과는 직교하도록 제약해(평탄 최소 탐색) exploitation과 exploration을 균형 있게 강제한다. 또한 full-batch에 지나치게 의존해 생기는 불안정/과평탄화 문제를 완화하기 위해, 배치-특이 성분을 고려하는 방식으로 업데이트 일관성과 적응성을 확보한다.

- **Empirical Impact**: SAMPLe는 CoOp, CoCoOp, MaPLe, TCP, Co-Prompt 등 여러 prompt learning 프레임워크에 통합해 일관된 개선을 보였고, 다양한 설정에서 기존 최적화기보다 성능이 향상됐다. 실험은 base-to-new class, cross-dataset, cross-domain 일반화 시나리오를 폭넓게 다루며, seen 과적합을 줄이면서 unseen 적응력을 유지하는 효과를 확인한다. 결과적으로 SAMPLe는 모델에 종속되지 않는 robust한 plug-in 솔루션으로서, 멀티모달 프롬프트 학습의 일반화 성능을 실증적으로 끌어올렸다는 의미가 있다.



### Association Restoration Test: Revealing Restorable Shortcuts after Unlearning (https://arxiv.org/abs/2607.05726)
Comments:
          Preprint. 16 pages

- **Prior Approaches**: 기존 association unlearning 평가는 주로 출력 수준의 견고성(예: worst-group accuracy)이나 frozen feature에서 숏컷 속성의 “readability”를 linear probe로 확인하는 방식에 머물렀습니다. 하지만 readability가 남아있더라도 원래 classifier head가 그 연관을 실제로 “기능적으로” 복원해 쓸 수 있는지(복원 가능성)는 검증되지 않았습니다.

- **Core Contribution**: 이 논문은 label–attribute 숏컷이 원래 분류 헤드에서 기능적으로 다시 살아날 수 있는지 점검하는 진단법 Association Restoration Test(ART)를 제안합니다. ART는 class-conditional 연관 방향을 추정한 뒤 잔여 성분을 증폭해 원래 head로 재예측을 수행함으로써, “숨겨졌는지 vs 복원 가능한지”를 분리해 평가합니다.

- **Technical Challenges**: 핵심 기술적 어려움은 표현 공간에서 숏컷 방향이 잡음과 섞인 경우까지 복원 평가에 포함되지 않게 하는 것입니다. 논문은 class-conditional 잔여 방향을 추정하되 gating과 label-null correction으로 불안정한 방향을 걸러내고, true-label 기준으로 복원 경로를 선택해 오탐 라우팅 영향을 줄였습니다.

- **Empirical Impact**: Waterbirds, CelebA, SpuCoDogs 및 ISIC(타임스탬프 아티팩트) 확장 전반에서 출력 지표, probe 기반 readability, ART 기반 기능 복원성은 서로 다르게 나타났습니다. 특히 어떤 방법은 숏컷 정보를 표현에 남기되 head와의 결합만 끊어 functional decoupling으로 보였고, 다른 방법은 ART로 숏컷이 재활성화되어 WGA 하락과 conflict shortcut rate 증가가 동반됐습니다. 결과적으로 저자들은 association-target unlearning 및 shortcut-mitigation이 “출력”만이 아니라 restoration-aware한 평가를 받아야 한다고 강조합니다.



### Scene Graph Thinking: Reinforcing Structured Visual Reasoning for Multimodal Large Language Models (https://arxiv.org/abs/2607.05716)
Comments:
          ICML 2026

- **Prior Approaches**: 기존 MLLM은 대체로 이미지-텍스트를 평면 정보로 학습하거나, fine-grained 처리를 위해 think-with-image/크롭-줌인 같은 국소 확대에 의존하는 경우가 많았다. 이 접근은 작은 타깃의 인식은 돕지만, 장면 내 객체 간 구조적 관계를 명시적으로 다루지 못해 탐색이 비효율적이 되고 holistic한 추론이 약해진다. 결과적으로 복잡한 시각 과제에서 fine-grained 지각과 관계 추론 성능이 일관되게 한계에 부딪힌다.

- **Core Contribution**: 이 논문은 Scene Graph Thinking(SaGe)로, 장면을 scene graph로 구조화해 MLLM이 관계 기반으로 세밀하고 체계적인 시각 추론을 하도록 한다. hierarchical한 엔티티를 노드로, 공간/상호작용/의미 관계를 엣지로 명시해 추론 경로가 장면의 연결 구조에 근거하도록 만든다. 또한 평면 코퍼스를 자동으로 scene graph로 변환하고, 그로부터 120K 규모의 graph-aligned 학습 데이터를 생성한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 세분화된 compositional 인식, (2) bbox 중심의 잡음을 줄이면서 depth 인지, (3) 객체 간 관계를 신뢰성 있게 구성하는 것이다. SaGe는 Qwen2.5-VL-72B 기반 노드 탐지 후 bbox 내부를 다시 재줌하여 하위 구성요소를 채굴하고, Depth Anything과 SAM 세그멘테이션으로 depth range를 노드에 부여하며, 관계는 MLLM의 직접 추론이 아니라 노드 priors(2D 위치/깊이/구성)를 기반으로 엣지를 구성한다. 이어 두 단계 후학습에서 SFT로 structured reasoning을 내재화하고, GRPO 기반으로 node-as-proxy graph rewards(노드 관련성+노드 기반 시각 정합)를 통해 효율적이고 grounded된 탐색을 강화한다.

- **Empirical Impact**: 실험에서 SaGe는 8개 멀티모달 벤치마크 전반에서 일관된 성능 향상을 보였고, 특히 고해상도 fine-grained 지각과 공간 추론에서 큰 폭의 개선이 보고된다. 예를 들어 VStarBench에서 SaGe-3B는 Qwen2.5-VL-3B 대비 큰 상승을 보이며, GPT-4o와 비교해도 경쟁력 있는 결과를 보인다. 또한 CVBench-2D/3D에서 bbox와 depth cues를 포함한 노드 기반 추론이 효과를 보였고, ablation 결과 node-as-proxy reward 조합이 reward hacking을 억제하면서 탐색 공간을 줄여 성능을 끌어올리는 것으로 나타나 분야의 structured visual reasoning 방향에 의미 있는 임팩트를 준다.



### Robust Face Super-Resolution and Recognition Through Multi-Feature Aggregation in Diffusion Models (https://arxiv.org/abs/2607.05702)
- **Prior Approaches**: 감시 환경의 저해상도·저품질 영상에서는 잡음, 가림, 조명/자세 변화가 겹치며 기존 SOTA 얼굴 인식 성능이 크게 떨어진다. 이를 보완하려고 초해상도(SR)를 전처리로 쓰는 시도는 있어왔지만, SR은 ill-posed 문제라 세밀한 디테일 복원과 신원(identity) 보존이 동시에 어렵다. 일부 방법은 eyeglasses·beard·gender 같은 soft attributes를 조건으로 쓰지만, 저해상도에서는 속성 추정이 불확실하고 추가 분류기/수동 추출이 필요하다는 한계가 있다.

- **Core Contribution**: 이 논문은 diffusion-model 기반 얼굴 SR인 FASR++를 제안하며, 신원 왜곡을 최소화하는 방향으로 고해상도 생성을 수행한다. 기준 저해상도 이미지(LR0)뿐 아니라, 동일 인물의 여러 저해상도 보조 이미지에서 뽑은 얼굴 feature를 Feature Aggregation으로 결합해 더 신뢰도 높은 identity 정보를 조건으로 사용한다. 특히 역과정(reverse diffusion)을 classifier 없이(gradient guidance 없이) 진행하며, 생성 결과를 얼굴 인식에 유리한 자연스러운 이미지로 개선하는 것이 핵심이다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 저해상도에서 추출된 feature에 섞인 잡음/왜곡을 어떻게 줄이면서도 (2) diffusion이 의사결정하는 생성 공간에서 신원 일관성을 어떻게 유지할지다. 저자들은 Feature Combiner(FC) 모듈을 학습해 서로 다른 저해상도 feature를 쌍(pair) 단위로 융합·정제하고, 이후 ensemble/평균으로 최종 merged descriptor를 만든 뒤 이를 U-Net 기반 diffusion의 전역 조건으로 주입한다. 또한 diffusion 단계의 시간(time embedding)과 함께 조건을 멀티스케일로 주입하며, SR 불확실성을 줄이기 위해 두 장(SRa, SRb)을 생성해 평균해 최종 SR을 얻는 구성을 사용한다.

- **Empirical Impact**: CelebA와 Quis-Campi 두 벤치마크에서 verification(1:1)·face recognition(1:N) 및 PSNR/SSIM/LPIPS 같은 화질 지표 모두에서 SOTA 대비 성능을 개선했다고 보고한다. 정성적으로도 기존 SR 모델 대비 자연스러움이 높고 신원 관련 왜곡이 줄어든 결과를 보이며, 인식용 전처리로 활용 시 실질적인 이득이 있음을 강조한다. 기존 FASR 대비 FC 설계 개선, 평가 지표 확장, ablation을 통해 기여 요소의 효과를 체계적으로 뒷받침한다.



### Clustered Codebook Quantization for 2D Gaussian-based Image Compression (https://arxiv.org/abs/2607.05667)
Comments:
          3 pages. Accepted to ACM SIGGRAPH 2026 Poster Track. Code available at this https URL

- **Prior Approaches**: Gaussian 기반 이미지 표현은 anisotropic Gaussian primitive로 고충실도 렌더링을 가능케 하지만, primitive당 부동소수점 파라미터가 많아 고비트레이트 목표에서 rate-distortion 효율이 떨어진다는 문제가 있다. 이를 줄이기 위해 GI는 VQ/RQ로 부동소수점 속성을 codebook 인덱스로 압축하지만, 단일(전역) codebook이 자연영상 파라미터의 큰 분산을 함께 흡수해야 해서 양자화 오차와 아티팩트가 생기기 쉽다. 결과적으로 “재구성 정확도”와 “codebook 효율” 사이의 제약이 남아 있다.

- **Core Contribution**: 이 논문은 Cluster-Guided Vector Quantization(CGvQ, CGVQ)로, quantization 전에 Gaussian 파라미터를 동질(homogeneous) 그룹으로 먼저 분할해 더 높은 압축 효율을 달성한다. 각 클러스터마다 position, rotation-scale, color에 대해 별도의 localized codebook을 학습해 파라미터 재구성을 정확히 유지하면서 엔트로피를 줄이는 방향을 제시한다. 결과적으로 고품질 영역에서도 rate-distortion trade-off를 개선하는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 난제는 전역 codebook의 용량 제약을 깨면서도, 파라미터 통계가 달라지는 여러 속성을 안정적으로 양자화해 재구성 오차를 줄이는 것이다. 이를 위해 K-Means로 rotation/scale/RGB color 특징(단, 공간 위치 μ는 제외)을 정규화해 클러스터를 만들고, 고정된 클러스터 할당을 파이프라인 전반에 공유한다. 또한 position은 FP16 QAT+FakeVQ estimator, rotation-scale는 UQ codebook, color는 RQ로 residual refinement를 적용하고, 재구성 손실(L1 in pixel+feature)과 commitment loss를 함께 최적화한다.

- **Empirical Impact**: 실험에서는 Kodak 데이터셋에서 CGVQ가 GI baseline 대비 bpp를 20% 줄이면서도 시각 품질을 동급 수준으로 유지했으며, PSNR/MS-SSIM이 평가 비트레이트 전 구간에서 더 높게 나타났다. 특히 PSNR 30dB 이상 고품질 영역에서 구조적 유사성까지 더 잘 보존되어 전체 rate-distortion 성능이 우세하다고 보고한다. 또한 동일 primitive 수(예: 15K)에서 PSNR이 1.68dB 향상되고, GI 대비 더 적은 primitive로 동등 품질을 달성(약 21% primitive count 감소)해 Gaussian 압축의 실용적 효율 개선 가능성을 보여준다. 다만 클러스터 수가 늘수록 인코딩/디코딩 FPS가 감소하므로, 압축 이득과 연산 효율 사이의 명확한 절충이 필요하다는 점도 함께 드러난다.



### REVIVE: A Multi-Modal Framework for Vandalism Detection and Recovery in Autonomous Vehicles (https://arxiv.org/abs/2607.05649)
- **Prior Approaches**: 기존 연구들은 VOA(물리적 바늘공/페인트 등으로 인한 폐색)를 주로 탐지하거나 센서 중복·폴백으로 “기능을 버티는” 방식에 초점이 맞춰져 있었다. 복원(inpainting·restoration) 접근도 일반적인 결손/손상처럼 취급해, 폐색의 구조(잡음형 vs 블록형 vs 정렬 참조 가능 여부)에 따른 최적 복구를 선택하지 못했다. 또한 생성형 복원은 시각적 그럴듯함은 개선해도, AV에서 중요한 객체검출 증거를 안정적으로 복원한다는 보장이 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 AV 카메라 스트림을 대상으로 VOA를 탐지·유형 분류·영역 분할한 뒤, 유형에 맞는 복구 분기를 라우팅하는 end-to-end 프레임워크 REVIVE를 제안한다. 핵심은 “유형 기반(type-aware) 복구 선택”이며, 랜덤(잡음형)은 통계적 필터, 큰 구조 폐색은 BLIP-guided Stable Diffusion 인페인팅, 그리고 정렬 참조 프레임이 있을 때는 direct pixel replacement를 사용한다. 또한 생성형 복원은 실시간 지연을 고려해 비동기(quality gate) 브랜치로 취급하고, 정답 참조 대비 성능이 나빠지지 않도록 downstream 전 후보를 필터링한다.

- **Technical Challenges**: 가장 큰 과제는 폐색 유형별로 필요한 복구 성격이 달라 “하나의 복원 방식”으로는 객체검출 관점에서 충분하지 않다는 점이다. REVIVE는 EfficientNet 기반 U-Net 세그멘테이션에 VOA 타입 정보를 조건으로 주입해 폐색 영역을 정밀화하고, 복구 분기별로 적절한 알고리즘을 선택하도록 파이프라인을 설계했다. 더불어 생성형 인페인팅의 환각/아티팩트 위험을 막기 위해 reference-available quality gate(미복구 대비 recall 저하 금지, false positive 증가 억제, SSIM 기준 충족)를 적용해 통과한 복구만 순전송한다.

- **Empirical Impact**: BDD100K 기반 500개 clean/vandalized 추적 페어에서, 미복구 VOAs는 YOLOv8l 객체검출 recall을 0.588로 낮추지만 direct pixel replacement는 recall 0.967과 F1-score 0.970으로 복원한다(정렬 참조 조건). Stable Diffusion은 구조 폐색에서 시각적 복원 품질은 보이나 SSIM/PSNR이 유형별로 변동적이며, 평균 런타임이 매우 길어 blocking 실시간 경로가 아닌 품질 게이팅 비동기 브랜치로 두는 설계가 타당함을 보여준다. 또한 quality gate를 적용하지 않으면 타입 라우팅이 per-image recall을 0.304까지 떨어뜨리지만, 게이트 적용 시 0.608로 회복되어 “전달되는 스트림이 미복구보다 절대 나쁘지 않다”는 운영 목표를 실험적으로 뒷받침한다.



### VEIL: How Visual Encoding Hijacking Induces Bias In Vision Models (https://arxiv.org/abs/2607.05641)
- **Prior Approaches**: time-series classification(TSC)에서 수치를 이미지로 렌더링해 CNN/비전 백본에 넣는 방식이 널리 쓰이지만, 보통은 특정 차트 인코딩(예: line, bar)을 하나 정해 정확도만 비교해 왔다. 그 결과 모델이 학습하는 것이 실제 시간 패턴인지, 아니면 선 두께·경계선·점 밀도 같은 시각적 아티팩트인지 체계적으로 분리해 검증하기 어렵다. 또한 시각화에 대한 attribution/Grad-CAM은 ‘어디를 본다’는 힌트를 주지만, 인코딩 민감성이 표현 학습에 어떤 영향을 주는지까지 일관되게 판별하긴 어렵다.

- **Core Contribution**: 논문은 visual encoding hijacking(시각 인코딩 하이재킹)을 정의하고, 단순한 encoding-specific evidence(신호에 충실한 증거)와 구분해 ‘인코딩이 표현을 어떻게 흔드는가’를 진단하는 프레임워크 VEIL을 제안한다. VEIL은 representation similarity(CKA), cross-encoding transfer(선형 probing), attribution(Grad-CAM)을 함께 써서 인코딩 선택이 학습 표현의 정렬과 일반화에 미치는 영향을 다각도로 측정한다. 나아가 차트 기반 TSC를 모델링 선택이 아니라 ‘표현과 측정(representation and measurement) 문제’로 재정의한다.

- **Technical Challenges**: 핵심 기술 과제는 인코딩 효과를 단일 지표로 오해하지 않도록, 표현 정렬·전이성·주의 지도의 일관성을 함께 확인하는 체계를 만드는 것이다. 이를 위해 4가지 차트( line, area, bar, scatter )를 동일 학습 프로토콜로 UCR 31개 데이터셋에 적용하고, CKA의 off-diagonal 평균, cross-chart 선형 probing 정확도, PCA/UMAP로 기하 구조까지 동시 분석한다. 또한 Rendering-dependent 시각 증거에 대한 민감도를 perturbation(블러/머지/alpha fading)으로 스트레스 테스트하고, attention-guided training인 HINT는 일관된 진단 신호가 있을 때만 이득이 나타나도록 설계한다.

- **Empirical Impact**: 실험 결과 인코딩 간 표현 정렬과 전이는 데이터셋에 따라 크게 갈리며, 많은 경우 인코딩 민감성이 강해 encoding-invariant 표현이 제한적임을 보여준다. 특히 클래스 수가 높을수록 인코딩 민감도가 더 강하게 나타나는 경향이 관찰됐고, Grad-CAM/perturbation은 모델이 렌더링 의존적 시각 단서에 주목함을 시사한다. HINT는 일부 인코딩-divergent 데이터셋에서 성능을 개선하지만, encoding-invariant 데이터셋이나 특정 경우에는 오히려 악화돼 범용 완화책이 아님을 실증적으로 확인했다.



### Recovering Cloud Microstructures with Cascaded Diffusion Inversion (https://arxiv.org/abs/2607.05637)
Comments:
          Published at ML4RS Workshop ICLR 2026

- **Prior Approaches**: 기존 초해상도(SR)는 픽셀 왜곡을 줄이는 방식이 많아, 위성이미지처럼 열역학·구름 미세구조 변동이 큰 영역에서는 고주파 질감이 과도하게 뭉개지는 문제가 있었다. 또 cross-sensor·time-shift·스펙트럼 불일치가 큰 seviri→viirs, msg→mtg 설정에서는 transformer나 diffusion 기반 모델이 분포가 달라지며 구조 일관성이 깨지거나 잡음성 아티팩트를 만들기 쉽다. 자연영상 priors에 기대는 생성형 SR은 선명함을 얻더라도 “그럴듯한 환각”이 동반될 수 있다는 한계도 함께 지적된다.

- **Core Contribution**: 이 논문은 구름 multi-spectral 마이크로물리(대류 타워, 구름 갭 등) 복원을 목표로, diffusion inversion을 2단계로 분리해 해결책을 제시한다. Stage 1은 실제 seviri↔viirs(또는 msg↔mtg) 쌍 데이터를 학습해 센서 정렬·시간/기하 불일치에 강한 cross-sensor 매핑의 기준선을 만든다. Stage 2는 HR 단독 이미지를 기반으로 내부 down-grade를 합성해, 구조적으로 정렬된 상태에서 가는 구름 필라멘트 같은 고주파 디테일을 정교화한다.

- **Technical Challenges**: 핵심 technical challenge는 LR→HR 역문제가 ill-posed인 데다, 센서별 PSF·잡음·스펙트럼 응답이 달라서 LR에서 잃어버린 정보가 단순히 복원되지 않는다는 점이다. 연구진은 diffusion inversion 관점에서 고주파 잠재변수를 다루고, Stage 1에서 실데이터로 도메인 갭을 흡수한 뒤 Stage 2에서 합성 열화로 완전 정렬된 조건을 만들어 구조 학습을 강화한다. 또한 짧은 reverse chain으로 효율적 베이스 추정(y_base)을 만든 다음, 필요 시 cascaded로 Stage 2 정제를 더해 일관성을 유지한다.

- **Empirical Impact**: 실험에서 제안 방법은 seviri→viirs에서 PSNR과 지각(perceptual) 지표의 균형이 가장 좋았고, gradient preservation ratio가 기준선(이상적 1)에 가까워 미세 구조를 물리적으로 더 충실히 복원했다. msg→mtg에서도 최고 PSNR 수준을 유지하면서 perceptual distance는 최저, gradient ratio도 ideal에 근접해 과샤프닝/잡음 아티팩트가 줄어든 결과를 보였다. 특히 얇은 대류 필라멘트와 구름 갭 경계 같은 의사결정에 중요한 패턴을 더 일관되게 재현해, 기후·지속가능성 맥락의 구름 미세물리 분석 자동화 가능성을 높였다는 점에서 의미가 있다.



### Taxlifier: Leveraging Disease Taxonomy for Enhanced Multi-Label Classification in Chest Radiography (https://arxiv.org/abs/2607.05628)
- **Prior Approaches**: 기존 흉부 X-ray(CXR) 병해 분류 연구는 다중 병변을 단일 라벨처럼 처리하거나, 각 병변을 독립적인 다중 라벨로 예측하는 방식이 많았다. 하지만 서로 다른 병변이 시각적으로 겹치는 경우가 많아 분류 경계가 흐려지고 성능이 불안정해지는 한계가 있었다. 또한 병변 간 계층적 관계(질병 분류 체계)를 학습에 직접 반영하지 못해 임상적 해석성도 떨어진다는 지적이 있었다.

- **Core Contribution**: 이 논문은 병변 간 계층 관계를 활용하는 계층형 multi-label 분류 두 가지를 제안한다. loss-based 방법은 학습 최적화 과정에 계층 정보를 직접 녹여내고, logit-based 방법은 질병 분류 트리에서 각 클래스의 parent 클래스에 따라 예측 확률(로짓)을 조정한다. 이를 통해 병변 겹침 상황에서도 더 일관된 예측을 유도하고, 임상 의사결정에 활용 가능한 해석성을 함께 노린다.

- **Technical Challenges**: 핵심 기술 난제는 ‘계층 제약을 어떻게 모델 학습에 안정적으로 반영할지’와 ‘예측 확률을 계층 구조와 정합적으로 보정할지’였다. 저자들은 loss-based에서는 계층 구조를 손실 설계/최적화에 통합해 gradient가 계층 일관성을 학습하도록 만들었고, logit-based에서는 parent 클래스 조건을 이용해 각 클래스의 로짓을 후처리처럼 재가중하는 방식으로 확률을 보정했다. 더 나아가 데이터/병변 분포 변화에도 신뢰도를 확인하기 위해 통계 분석으로 견고성까지 검증했다.

- **Empirical Impact**: CheXpert(224,316), PADCHEST(160,000), NIH(112,120) 등 세 대규모 CXR 데이터셋에서 성능을 평가했으며 기존 기준선(baseline) 대비 유의미한 개선을 보였다. 구체적으로 accuracy는 loss-based 11%, logit-based 12%, AUC는 각각 10%, 13%, F1은 각각 12%, 24% 향상되었다. 또한 포괄적 통계 분석으로 견고성과 신뢰성을 뒷받침해, CXR CAD(Computer-aided diagnosis)에서 hierarchical multi-label 분류의 실용성과 확장 가능성을 강조한다.



### Cross-Contextual Vision-Language Adaptation with LoRA for Personalized Severe Adverse Event Detection in Clinical Wound Monitoring (https://arxiv.org/abs/2607.05625)
- **Prior Approaches**: 기존 상처 모니터링은 주로 분할·조직 분류·DFU 벤치마크 성능에 초점을 맞추거나, 임상 메타데이터와의 단순 멀티모달 융합을 통해 판별(classification)을 수행해왔다. 하지만 중증 부작용(SAE)은 개인화되고 희소·이질적이며 라벨도 불일치해, 감독 학습만으로는 ‘훈련 분포에서 벗어난 이상 치유 궤적’을 놓치기 쉽다. 또한 VLM을 도메인에 적용해도(예: fine-tuning, prompt learning, LoRA) 장기간 추적에서의 시간적 변화(healing dynamics)를 OOD 탐지 체계로 통합한 연구는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 상처 이미지, 임상 기록, 시각적으로 근거된 wound description을 동시에 활용해 SAE(감염·조직 악화·치유 지연 등) 및 이상 치유를 자동 감지하는 멀티모달 프레임워크를 제안한다. 핵심은 frozen BiomedCLIP 백본 위에 dual-stream LoRA를 얹고, 임상 의미와 상처 묘사의 양방향 cross-contextual fusion으로 ‘도메인 기반 정렬’을 강화한 점이다. 더 나아가 SAE를 라벨 없이도 식별하기 위해 wound-specific OOD 기반 SAE score를 설계하고, 방문 간 시간 변화를 반영하는 정규화까지 포함해 개인화된 위험 신호를 포착한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 임상 텍스트와 상처 묘사의 의미를 강하게 결합하되 전체 fine-tuning 없이 효율적으로 적응해야 하고, (2) 시간이 흐르며 분포가 변하는 longitudinal setting에서 ‘정상 치유 변화’와 ‘이상 치유’를 구분해야 한다는 점이다. 저자들은 두 텍스트 스트림 각각에 LoRA 어댑터를 학습하고, cross-adapter composition 형태의 결합으로 임상-상처 묘사 간 정보 교환을 구현해 멀티모달 표현을 생성한다. OOD SAE 점수는 semantic matching, visual typicality, caption-text alignment, caption-visual alignment를 통합하고, covariate shift에 대해 점수 안정성을 유지하도록 covariate consistency를 두며, 상처 면적 변화에 가중치를 준 temporal drift regularization으로 방문 간 치유 역학을 반영한다.

- **Empirical Impact**: 실험은 임상 방문에서 수집된 종단형 데이터셋 SmartBoot DFU(NCT04460573)에서 상처 치유 평가와 SAE 탐지 성능을 확인하는 방식으로 진행된다. 논문은 단일 스트림 기반 대비 우수하고, 특히 rare한 이상 치유/부작용 케이스에서 OOD 탐지 성능이 일관되게 개선된다고 보고한다. 또한 cross-contextual LoRA fusion과 multi-signal(4종) OOD score, 그리고 temporal regularization의 조합이 실제 임상 추적 환경에서 의미 풍부하고 시간 인지적인 vision-language 시스템 가능성을 보여준다는 점에서 의의가 있다.



### Patch Knowledge Transfer for Efficient AI-Generated Image Quality Assessmen (https://arxiv.org/abs/2607.05605)
Comments:
          13 pages. ICME26 Spotlight

- **Prior Approaches**: 기존 AI-generated image quality assessment(AIGIQA) 연구는 크게 전역 처리(global processing)와 로컬-전역 혼합 처리(local–global hybrid processing)로 나뉜다. 전자는 빠르지만 미세 디테일 손실로 정확도가 떨어지고, 후자는 정확하지만 높은 계산량(FLOPs) 때문에 대규모·실시간 필터링에 불리하다.

- **Core Contribution**: 논문은 Patch Knowledge Transfer(PKT)라는 지식 증류 기반 프레임워크를 제안해, 로컬 디테일부터 전역 의미까지의 표현 학습을 전역 모델만으로도 재현하려고 한다. 멀티레벨로 teacher(로컬-전역 혼합)를 supervision하고, student(전역만)는 feature distillation과 output distillation을 통해 teacher의 품질 판별 능력을 효율적으로 상속한다.

- **Technical Challenges**: 핵심 기술 과제는 student가 로컬-패치 기반의 풍부한 품질 priors를 전역 단일 패스 계산으로 흡수하도록 만드는 것이다. 이를 위해 마지막 인코더 특징 정렬(feature-level knowledge transfer, feature alignment loss)과 예측 분포 정렬(output-level knowledge distillation, KL divergence)을 계층적으로 함께 학습하며, PKT-1/PKT-2의 두 가지 학습 전략도 비교한다.

- **Empirical Impact**: 4개 AIGIQA 벤치마크(AGIQA-1K/3K, AIGCIQA2023, PKU-AIGIQA-4K)에서 PKT student는 teacher 수준의 성능을 유지하면서 계산비용을 평균 67.7% 절감했다. 또한 기존 방법 대비 정확도-효율 균형이 더 우수하며, 일부 설정에서는 student가 teacher를 넘어서는 지표도 보고되어 실용적 배포(실시간 품질 필터링) 관점에서 의미가 크다.



### Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO) (https://arxiv.org/abs/2607.05585)
- **Prior Approaches**: 기존 얼굴 기반 유전질환 연구는 주로 2D 이미지를 CNN으로 바로 질환(증후군)을 예측하는 방식이 많았고, 일부는 3D 정보를 추가해 성능을 높였지만 여전히 ‘질환 단위’ 분류가 중심이었습니다. 이 접근은 데이터 편향(인종·연령·표본 수)에 민감하고, 왜 그런 결론이 나왔는지 임상적으로 설명하기가 어려워 신뢰와 채택에 제약이 있습니다.

- **Core Contribution**: FaceMesh2HPO는 질환 분류가 아니라 Human Phenotype Ontology(HPO) 기반의 ‘표현형(phenotype) 항목’ 예측으로 문제를 재구성해, 임상 추론과 연결되는 해석 가능성을 목표로 합니다. 또한 HPO 트리 구조(교차 링크 제거) 위에 계층적 모델을 배치하고, 비가시/미완전 라벨 상황을 phenotype 수준에서 다루어 의료 현장에 더 가까운 분류 프레임을 제안합니다.

- **Technical Challenges**: 핵심 난제는 HPO 라벨이 희소하고 존재 여부만 확정된 weakly labeled 특성(미기재=부재로 단정 불가)과, HPO의 계층·의존 구조로 인해 단일 모델 학습이 복잡해지는 점입니다. 논문은 (1) 124명의 임상의가 2D 얼굴에 대해 present/absent/uncertain를 포함해 107개 HPO(부모 항목 확장 포함) 라벨을 재정의하고, (2) 2D→3D face mesh(478 랜드마크)와 계단식(cascading) PointNet 트리를 학습하되, Integrated Gradients로 중요하지 않은 포인트를 제거하는 feature elimination으로 하위 항목 학습 난도를 낮추는 방식으로 대응합니다.

- **Empirical Impact**: 3D mesh, 얼굴 외곽(outline), 인구통계 메타데이터를 포함한 최고 성능은 AUROC가 약 0.55~0.89 범위로 나타났고, leaf보다 상위(parent) 노드에서 성능이 더 높게 관측됐습니다. 외부 독립 테스트에서는 disorder 간 일반화 폭이 달라 성능 변동성이 확인되었으며, 특히 드문(rare) leaf 항목에서 한계가 남아 데이터 다양성과 라벨/특징 선택 전략 개선이 필요하다는 결론을 제시합니다.



### Harnessing Generative Image Models for Training-Free Primitive Shape Abstraction (https://arxiv.org/abs/2607.05568)
Comments:
          13 pages, 9 figures, 3 tables

- **Prior Approaches**: 기존 프리미티브(초기형상) 기반 3D 추상화는 학습 기반이거나(primitive 파라미터 예측) 순수 최적화 기반(기하 기준으로 분할·피팅)으로 나뉜다. 학습 기반은 학습 분포/정준 방향(canonical orientation)에 강하지만, 범용 범주·임의 자세에 취약하다. 반면 최적화 기반은 category-agnostic일 수 있어도 분할이 의미적 일관성을 잃어(예: 의자 다리가 의미 있는 한 부품으로 뭉치지 않음) 분해와 피팅이 서로 결합된 한계를 보였다.

- **Core Contribution**: 이 논문은 3D 학습 없이(training-free) 2D 파운데이션 모델의 ‘부품(semantic parts) 이해’를 3D 프리미티브 분해로 끌어오는 파이프라인을 제안한다. 멀티뷰 렌더를 비전-언어 모델로 의미 부품과 컬러 매핑을 만든 뒤, 생성형 이미지 모델로 컬러 코드 부품 마스크를 그려 3D로 재투영하고, 각 부품에 superquadric을 고전적 최적화로 피팅한다. 결과적으로 category-agnostic, orientation-invariant 특성을 갖고 learned parameter 없이 동작한다는 점이 핵심이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 여러 뷰에서 전역적으로 일관된 부품 분해를 만들고(색 매핑/부품 개수·정의의 글로벌 합의), 재투영 과정에서 그 의미 경계를 유지한 뒤, 비선형 비합성 문제인 superquadric 피팅이 로컬 미니멈에 빠지지 않게 하는 것이다. 논문은 ‘분석(부품·색 JSON) 단계’와 ‘생성(컬러 마스크)’ 단계를 분리해 뷰 간 컬러 일관성을 확보하고, 컬러 기반 클러스터링으로 노이즈를 정리한 뒤, ApproxMVBB 기반 초기화와 parallel multi-start, 그리고 L-BFGS로 비선형 최적화를 안정화했다. 또한 좌표계 정규화/제약된 latent parameter 변환으로 유효한 변형 형태(taper/bending) 탐색을 관리한다.

- **Empirical Impact**: HumanPrim과 Toys4K에서 Chamfer distance(CD) 기준 모든 비교 방법 중 최저 성능을 보이며, 객체당 평균 5–9개의 primitive로 가장 낮은 표면 충실도를 달성했다(예: HumanPrim CD=0.079, Toys4K CD=0.093). 특히 ground-truth segmentation으로 분할만 바꿔보는 ablation에서 IoU가 크게 개선되어, 현재 병목이 primitive fitting이 아니라 part segmentation 품질임을 실증했다. 이는 생성형 이미지 모델 성능이 올라가면 재학습 없이도 파이프라인 정확도가 연쇄적으로 향상될 수 있음을 시사한다.



### Multi-Teacher Contrastive Distillation for Edge-Efficient Pathology Foundation Models (https://arxiv.org/abs/2607.05533)
- **Prior Approaches**: 기존 computational pathology foundation model(PFM)들은 성능을 크게 끌어올렸지만, 수백M~1B급 백본과 대규모 타일 처리로 인해 배포 비용이 높아 로컬 환경 제약이 컸습니다. 지식 증류와 contrastive distillation 계열은 압축을 시도했지만, 다중 PFM의 ‘관계(상대 구조)’를 유지하면서도 edge용 초경량 인코더로 이어지는 설계는 제한적이었습니다. 또한 LiteFM 등 일부 효율화 모델이 있으나, 저비용 엣지 디바이스(워크스테이션/라즈베리파이/마이크로스코프 연동)에서의 실측 지연-정확도 균형이 충분히 정교하게 제시되진 않았습니다.

- **Core Contribution**: MuCoDi는 여러 PFM의 타일 임베딩을 단단히 압축해 edge-oriented 병리 인코더를 만드는 다중-교사 대비 증류 프레임워크를 제안합니다. 핵심은 teacher의 개별 피처를 그대로 맞추는 방식이 아니라, MoCo v3에서의 contrastive objective를 변형해 ‘캐시된 teacher 임베딩’을 키로 쓰는 구조로 교사 간 형태적 관계를 학생이 학습하도록 한 점입니다. 그 결과 RepViT·MobileOne 계열의 소형 인코더가 PFM급 표현력을 유지하면서도 로컬/엣지 배포에 가까운 모델 스케일을 달성합니다.

- **Technical Challenges**: 첫째, 대형 교사 3종(Virchow2, UNI2, H-Optimus-1)의 임베딩을 학습 중 momentum-encoder 없이도 contrastive distillation의 key로 안정적으로 공급해야 했습니다. MuCoDi는 teacher를 고정하고, 이미지 타일마다 사전 계산한 임베딩을 캐시해 cross-GPU로 negatives를 구성하며 InfoNCE를 다중 교사·다중 뷰로 합산하는 방식으로 이를 해결했습니다. 둘째, 초경량 백본에서 표현력 손실을 막기 위해 교사별 선형 projection head와 학습 안정화(temperature 스케일링, 대용량 타일로 10 epoch 사전학습)를 함께 최적화했습니다.

- **Empirical Impact**: MuCoEdge는 TCGA에서 11.8K WSIs만으로 학생을 사전학습한 뒤, CPTAC 매칭 데이터의 23개 임상 큐레이션 이진 분류 과제에서 외부 성능을 검증했습니다. RepViT 기반 MuCoEdge-R2.3는 외부 AUROC 71.0%로 최고 교사(Virchow2, 71.8%) 대비 0.8%p 이내에 도달했고, MuCoEdge-R2.3는 외부 F1/AUPRC에서도 우수한 결과를 보였습니다. 더 나아가 6.4M 파라미터·1.12 GFLOPs의 MuCoEdge-R1.0는 AUROC 70.9%를 유지했으며, Raspberry Pi 5에서 sub-million MobileOne은 단일 타일 기준 327~605배(최대 605배) 속도 향상과 외부 AUROC 66.5~66.9%를 함께 달성해 ‘PFM 품질 표현의 실전 엣지 이전’ 가능성을 실증했습니다.



### Rendering-Aware Bayesian 3D Gaussian Splatting with Native Uncertainty and Adaptive Complexity Contro (https://arxiv.org/abs/2607.05522)
Comments:
          26 pages, 4 figures, 24 tables including appendix. Preprint

- **Prior Approaches**: 3D Gaussian splatting(3DGS)은 실시간 novel-view synthesis에 강하지만, 학습이 점추정(point estimate) 중심이라 불확실성을 자연스럽게 제공하지 못한다. 또 Gaussian birth/death 같은 핵심 제어가 hand-tuned heuristics에 의존해, 희소 뷰(sparse views)나 고정 예산 환경에서 약하게 지지된 기하를 식별하거나 다음에 볼 카메라를 원리적으로 고르는 데 한계가 있었다.

- **Core Contribution**: 논문은 렌더러(알파 컴포지팅)에서 얻은 surrogate summary로 normal-inverse-Wishart(NIW) posterior를 각 Gaussian의 mean/ covariance에 연결하는 rendering-aware Bayesian 3DGS를 제안한다. 여기에 선택적으로 Dirichlet-process 확장을 더해 component usage에 대한 확률 신호로 복잡도 제어까지 함께 제공하며, 폐루프가 아닌 “렌더링 인지적” 베이지안 갱신 경계를 명시한다.

- **Technical Challenges**: 핵심 과제는 비선형 렌더링 전체를 정확한 end-to-end Bayesian 추론으로 돌리기 어렵다는 점인데, 논문은 renderer-derived surrogate statistics로 conjugate 업데이트는 closed-form으로 처리하고 나머지는 명시적으로 근사하는 학습 스케줄을 설계했다. 또한 posterior 샘플을 다시 렌더링해 픽셀 단위 predictive uncertainty와 interval calibration 신호를 만들고, 고정 예산 active view selection은 이 불확실성을 기반으로 점수화하는 방식으로 연결했다.

- **Empirical Impact**: 고정 예산 16-to-32 active-view에서 NIW native acquisition은 scoring-only standard ensemble 대비 PSNR +0.453 dB, LPIPS -0.0146의 개선을 보이며 29/39 씬 시드에서 승리했다. 95% 커버리지 오차는 shared proxy 대비 약 17배 감소(0.046 vs 0.796)하고, 3-member deep ensemble과 비교해서는 nominal coverage에 약 10배 더 가깝지만 학습비용은 약 1/3 수준이다; 추가로 호환성 확인에서도 matched 실행 39쌍에서 PSNR이 +0.030 dB 수준으로 유지되었다.



### Statistical Adversaries: Natural Backdoor-like Features in Vision Datasets (https://arxiv.org/abs/2607.05516)
- **Prior Approaches**: 기존 연구는 주로 PGD 같은 최적화 기반 adversarial attack, 또는 victim/surrogate 모델을 통해 adversarial direction을 찾아내는 방식에 집중해 왔습니다. 또한 이미지에서 잡음·주파수 성분을 이용하거나 Fisher geometry 같은 정보기하를 통해 공격 방향을 모델 신호로 도출하는 연구가 많았습니다.
반면 ImageNet의 spurious feature(편향된 빈도 단서, 주파수 단서 등)는 “왜 실패하는가”를 설명하는 감사(audit) 관점에 머물렀고, 무해하게 보이는 자연 데이터 구조가 실제로 “공격 표면”이 될 수 있는지까지는 충분히 연결되지 않았습니다.

- **Core Contribution**: 이 논문은 악의적 poisoning 없이도 데이터의 클래스 조건부 통계로부터 backdoor-like 행동을 유발하는 ‘statistical adversaries’를 정의하고, 그 방향을 모델 접근 없이 구성하는 절차를 제시합니다. 즉, victim 모델 gradient나 쿼리, surrogate 최적화 없이도 목표 클래스에 특화된 오탐을 유도하는 단일 perturbation direction을 만든다는 점이 핵심입니다.
또한 이 취약성이 특정 아키텍처의 idiosyncrasy가 아니라, 데이터셋 구조·분포에 의해 공유될 수 있음을 실험적으로 뒷받침합니다.

- **Technical Challenges**: 가장 큰 난제는 모델을 보지 않고(gradient/쿼리/최적화 없이) “어떤 통계가 어떤 방향을 만들며, 그 결과가 목표 클래스 특이적으로 나타나는가”를 재현 가능하게 구성하는 것입니다. 연구진은 타깃 클래스 평균 대비(1·2차 모멘트)를 기반으로 방향을 만들되, diagonally whitened band-pass 및 Hellinger-motivated 주파수 통계 같은 reshaping/제어 연산을 넣어 무작위 상관을 줄이는 방식으로 해결합니다.
마지막으로 ℓ∞ 예산 하에서 입력 공간에 동일한 perturbation을 재사용하고, 임계값을 고정한 one-vs-rest FPR 지표로 목표 특이적 false positive를 정량화해 통계적 효과를 검증합니다.

- **Empirical Impact**: ImageNet-1K 학습 통계만으로 만든 perturbation은 four model(ResNet-50, ConvNeXt-Tiny, ViT-B/16, Swin-T)에서 목표 클래스별 FPR을 기준선 5.005%에서 9.689%로 끌어올렸고, 44개 조건 중 43개에서 효과가 양(+)의 방향으로 나타났습니다. 또한 임계값 통과 FPR, 타깃 로짓 이동, 타깃 랭크 변화가 관찰됐으며 단순 top-1 takeover보다는 “보정된 오탐 팽창” 형태로 나타나는 점이 특징입니다.
대조군 결과로 Gaussian random·global mean은 거의 기준선 근처에 머물렀고, lowpass/spectrum 매칭은 일부를 설명하지만 제안 방향이 대부분 조건에서 더 크게 나타났습니다. 결론적으로, poisoning이 없어도 데이터의 spurious structure가 비가역적 공격 표면을 형성할 수 있어 dataset audit에서 ‘편향/해석 실패’뿐 아니라 ‘잠재 공격 표면’으로 취급해야 한다는 메시지를 강화합니다.



### Light-Omni: Reflex over Reasoning in Agentic Video Understanding with Long-Term Memory (https://arxiv.org/abs/2607.05511)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 메모리-증강 agentic video understanding은 검색(Retrieval)과 evidence aggregation을 위해 detective-style의 반복 추론(계획-검색-재구성)을 자주 사용한다. 하지만 이는 전역 맥락(global context)이 약하고, 사용자 쿼리와 저장된 메모리 표현 사이의 의미 불일치로 인해 정교한 재검색이 누적되며 비용과 지연이 크게 증가한다. 또한 메모리 은닉요소(짧은 단기 메모리, 희소 샘플링)나 효율화(KV-caching, token merging)는 긴 지평에서 맥락 손실을 완전히 해결하지 못한다.

- **Core Contribution**: Light-Omni는 반복적 추론 부담을 줄이고, 단일 forward pass 안에서 reflexive(즉응형) 응답과 semantically aligned retrieval을 수행하는 멀티모달 에이전트 프레임워크를 제안한다. 핵심은 dual contextual states로, (1) episodic memory에서 통합된 compact global state로 전역 맥락을 즉시 구성하고 (2) 이를 조건으로 latent state가 행동 제어용 신호와 검색 임베딩을 동시에 만든다는 점이다. 결과적으로 쿼리-메모리 의미 갭을 “다시 생각해서” 메우는 대신 “구조적으로” 맞춘다.

- **Technical Challenges**: 전역 맥락을 빠르게 만들면서도 긴 영상에서 세부를 잃지 않기 위해, Light-Omni는 resolution-decaying hierarchical merging으로 최근 디테일은 보존하고 과거는 요약해 bounded context를 구성한다. 동시에 반복 추론 없이 검색 분포를 맞추기 위해, latent state에서 retrieval embedding을 soft prompt 기반으로 학습적으로 rectification하고 action trigger 확률과 함께 병렬 디코딩한다. 또한 long video 학습을 위해 자동 생성 데이터 파이프라인과 multi-LoRA로 memorization/generation/reaction의 최적화 간섭을 줄였고, inference에서는 feature caching과 redundancy pruning으로 지연을 추가로 낮췄다.

- **Empirical Impact**: 실험에서 Light-Omni는 VideoMME-long, LVBench, HippoVlog, OVO-Bench에서 전반적으로 강한 성능을 보였고, Qwen2.5-Omni-7B 대비 평균 정확도 9.5% 향상과 함께 약 20.5× 속도 향상, 3.3× GPU 메모리 축소를 달성했다. 추론 기반 agent와 비교해도 우수했으며 특히 M3-Agent 대비 평균 2.4% 정확도 이득, 12.1× speedup, 2.6× GPU 메모리 효율 향상을 보고했다. 더 나아가 Light-Omni는 기존 MLLM에 범용 메모리 모듈로 결합되어 정확도와 효율을 동시에 개선하는 방식으로 “지연 없는 장기 기억”의 실용성을 입증했다.



### Ground3D-LMM: Fine-Grained 3D Point Grounding and Spatial Reasoning with LMM (https://arxiv.org/abs/2607.05493)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 3D 비전-언어 모델은 주로 텍스트만 생성하거나(대화/QA), 혹은 텍스트 기반으로 3D 마스크·박스만 내놓는 방식에 머물러 있었습니다. 그 결과 장면의 어느 영역을 지칭했는지(point-level grounding) 검증하기 어렵고, 미터/센티미터 같은 단위 일관성을 갖춘 metric 측정까지 대화형으로 제공하는 데 한계가 있습니다. 또한 part granularity와 multi-object 질의는 상대적으로 덜 다뤄져 실제 로보틱스·AR/VR 요구에 바로 연결되기 어려웠습니다.

- **Core Contribution**: 이 논문은 Ground3D-LMM을 제안해 point cloud(필요 시 RGB 이미지)와 질의를 입력으로 받아, 지칭된 오브젝트/파트에 대한 3D 마스크와 함께 metric 수치를 동시에 반환하도록 통합했습니다. object뿐 아니라 part 단위 측정(크기·두께·간극·거리 등)과 multi-object 비교 질의까지 지원하며, 대화 맥락에서도 grounding을 유지하는 것을 목표로 합니다. 이를 위해 3D Grounded Measurement라는 단일 평가 프로토콜을 정의해 ‘무엇을 말했는지’와 ‘얼마나 재는지’를 한 번에 점검합니다.

- **Technical Challenges**: 핵심 난제는 (1) 언어 생성 중 특정 구를 grounding trigger(<SEG>)로 연결해 정확한 point-level 마스크를 예측하는 것, (2) 예측된 동일 referent에 대해 OBB 치수·거리·clearance 같은 물리량을 실제 단위로 일관되게 산출하는 것입니다. 논문은 3D 포인트 토큰을 LLM에 투입하고, <SEG>가 생성되면 경량 segmentation head가 해당 구의 3D 마스크를 예측한 뒤, 마스크 기반 기하 속성으로 metric을 계산해 답변에 포함시키는 end-to-end 학습 구조를 설계했습니다. 또한 voxelization 기반 superpoint/희소 3D U-Net으로 잡음과 계산량을 줄이면서도 미세 공간 정보를 유지하도록 했습니다.

- **Empirical Impact**: ScanNet/ScanNet++를 기반으로 한 대규모 Ground3D 데이터셋과 과제(객체·파트, 거리관계, 다중턴 등)에서 Ground3D-LMM은 여러 기준선 대비 일관된 성능 우위를 보이며 grounding과 metric을 함께 개선하는 것으로 확인됐습니다. Grounded-Measurement 성공률(GM-δ)처럼 마스크 IoU와 수치 오차를 결합한 평가에서도 이미지-only 방식보다 낮은 Mean APE와 높은 δ 성공률을 기록해 ‘검증 가능 + 정량 측정’의 결합 효과가 드러납니다. 또한 Reason3D·ScanRefer에서도 open-vocabulary 인스턴스 단위 grounding에서 강한 일반화 성능을 보이며, 데이터/모델 공개로 후속 연구의 기반이 될 것으로 기대됩니다.



### Binocular Gaze Estimation with Single Camera and Single Light Sourc (https://arxiv.org/abs/2607.05473)
Comments:
          Accepted for presentation at the 2019 International Conference on Video, Signal and Image Processing (VSIP 2019), Wuhan, China, October 29-31, 2019; published in VSIP '19: Proceedings of the 2019 International Conference on Video, Signal and Image Processing, pp. 10-14, ACM, 2020; 4 figures, 1 table; ACM Proceedings ISBN: 978-1-4503-7148-3

- **Prior Approaches**: 기존에는 자유로운 머리 움직임을 전제로, 시선(gaze) 추정을 위해 최소 1대 카메라와 2개의 광원(light source)이 필요하다고 널리 받아들여져 왔다. 특히 glint(각막 반사점) 기반 방식에서는 두 광원으로부터 생성되는 glint 정보를 이용해 시선을 더 안정적으로 추정한다. 하지만 모바일 기기처럼 부품을 줄여야 하는 상황에서는 광원 수를 줄이려는 요구가 커져 한계가 드러난다.

- **Core Contribution**: 이 논문은 1대 카메라와 1개 광원만으로도 시선을 추정하는 방법을 제안한다. 핵심은 카메라를 기준으로 실제 광원과 대칭 위치에 ‘virtual light source(가상 광원)’를 기하학적으로 배치해, 영상에서 ‘virtual glint(가상 glint)’를 생성하는 개념이다. 이후 두 동공(pupil) 사이 거리와 실제/가상 glint 사이의 관계를 활용해 시선을 추정하고, 2개 광원이 있는 것으로 가정한 polynomial regression을 적용한다.

- **Technical Challenges**: 기술적 난점은 실제로는 광원이 1개뿐인데도 2개 광원 시스템과 유사한 glint 신호 구조를 안정적으로 재구성해야 한다는 점이다. 논문은 virtual glint를 기하학적 대칭으로 정의하고, 영상에서 관측되는 두 pupil과 두 glint(실제 glint+virtual glint) 간의 거리 관계를 회귀 모델에 반영해 추정 정확도를 확보한다. 또한 regression용 새로운 normalization factor를 검증해, one-glint 시스템에서도 적용 가능함을 보인다.

- **Empirical Impact**: 실험 결과, one-glint 구성에서도 성능이 ‘수용 가능한 수준’으로 유지되지만 2개의 실제 광원을 쓰는 시스템 대비 성능 저하는 관찰된다. 즉 부품을 줄이는 실용성은 확보하되, 정확도 측면에서 트레이드오프가 존재한다는 메시지를 제공한다. 모바일 등 경량 환경에서 gaze tracker 설계의 제약을 낮출 수 있다는 점에서 현장 적용 가능성에 의미가 있다.



### A Task-Driven Evaluation of UAV Detection and Tracking under Synthetic Fog (https://arxiv.org/abs/2607.05467)
- **Prior Approaches**: 기존 연구는 합성 안개/헤이즈 데이터로 학습을 보강하거나, 이미지 복원을 전처리로 먼저 적용한 뒤 검출·추적을 수행하는 방식이 주를 이뤘다. 특히 자율주행 영역에서는 ‘안개에 포함해 학습하는 것’이 복원만으로 downstream 이득이 크지 않다는 관찰이 누적됐다. 그러나 하늘 지배(sky-dominant) 장거리 UAV 영상에서 깊이에 따라 달라지는 안개를 합성하고, 복원-검출-추적을 동일 프로토콜로 함께 비교한 체계적(task-driven) 평가는 부족했다.

- **Core Contribution**: 이 논문은 깊이 추정 기반의 depth-aware 합성 안개 생성, 이미지 restoration(복원), UAV object detection, tracking-by-detection을 하나의 파이프라인으로 묶는 task-driven 평가 프레임워크를 제안한다. 복원 성능(PSNR/SSIM)만이 아니라, 안개가 검출/추적의 강건성에 미치는 영향과 복원 적용의 실제 효과를 함께 측정하도록 설계됐다. 또한 fog-inclusive 학습(훈련 중 안개 노출)과 test-time restoration(추론 시 복원)의 조건별 상대 성능을 비교하는 실험 질문에 초점을 둔다.

- **Technical Challenges**: 문제의 핵심 기술적 난점은 안개가 낀 실제 UAV 장면을 수집·라벨링하기 어렵다는 점이며, 이를 위해 monocular depth estimation(MiDaS)과 대기 산란 모델(atmospheric scattering model)을 사용해 clear 영상에서 합성 안개를 만든다. MiDaS의 깊이 스케일 불일치를 줄이기 위해 분위수(percentile) 정규화 후 깊이 부호를 뒤집고, guided filtering으로 전송/경계 아티팩트를 완화해 안개 농도(β)를 다섯 단계로 제어한다. 이후 복원 모델로 classical/CNN/transformer 계열을 비교해 DehazeFormer를 선택하고, 복원 데이터셋과 함께 YOLO11 계열 검출기 학습(청결 학습 vs 30/50/70/100% fog-inclusive) 및 ByteTrack/BoT-SORT 기반 추적 평가까지 end-to-end에 준하는 동일 조건에서 수행한다.

- **Empirical Impact**: 실험 결과, 안개는 검출과 추적 모두를 크게 저하시켰고 그 주된 원인은 오탐 증가보다 missed detection(=false negative) 증가로 요약된다. 청결만 학습한 detector에서는 복원으로 성능이 부분 회복되지만 여전히 clean 기준선에는 못 미쳤으며, fog-inclusive 학습일수록 fog–restore 곡선이 clean에 더 가깝게 수렴해 강건성이 가장 일관되게 개선됐다. 또한(논문 전체 구성에 따라) tracking-by-detection에서도 clean/fog/restored 비디오를 동일 장면에서 비교해 MOTA와 IDF1 관점의 열화·복원 효과를 정량화했으며, restoration 품질이 downstream 지각 성능을 비례적으로 보장하지는 않는다는 결론을 뒷받침한다.



### CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration (https://arxiv.org/abs/2607.05465)
Comments:
          18pages, 5 figures

- **Prior Approaches**: 기존 멀티모달 도구 사용 연구는 Perception, search, 일반 추론에 치중하는 경우가 많고, 복잡한 이미지 생성·편집처럼 여러 도구 호출이 중간 시각 결과에 의존하는 ‘긴 실행 궤적’을 충분히 다루지 못했습니다. 또 이미지 편집/포토 리터칭 쪽도 단일 모델 호출 중심이거나 특정 환경에 한정되어, 이기종 도구를 조합하고 다수 중간 자산을 상태적으로 관리하는 대규모 실행 궤적 데이터가 부족했습니다.

- **Core Contribution**: 이 논문은 복잡한 이미지 생성·편집을 위한 대규모 멀티모달 도구 사용 데이터셋 CanvasCraft와, 이를 학습해 멀티턴 상호작용으로 이기종 visual tools를 오케스트레이션하는 CanvasAgent를 제안합니다. CanvasCraft는 140K개의 실행 가능한 주석 궤적과 10K개의 RL용 작업 스펙으로 구성되며, CanvasAgent는 SFT로 실행 가능한 궤적을 먼저 학습한 뒤 GRPO로 정책을 최적화합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 긴-horizon에서 중간 결과를 기반으로 도구 순서·파라미터·중단 시점을 스스로 결정해야 하고, (2) 생성 결과 품질뿐 아니라 궤적이 문법적으로/절차적으로 실행 가능해야 한다는 점입니다. 저자들은 비전 상태를 보며 중간 산출물을 검사하고 이미지 자산을 명시적으로 추적하는 실행 프로토콜을 쓰는 한편, outcome(정렬·미감)와 process(추론 타당성·규칙 준수·효율)를 함께 보는 하이브리드 리워드를 설계해 reward hacking을 줄이며 GRPO 학습을 안정화했습니다.

- **Empirical Impact**: 실험에서 CanvasCraft-SFT만 쓴 CanvasAgent는 전반 보상과 궤적 품질이 크게 개선되지만 도구 호출을 기대보다 적게 수행했습니다. SFT+RL로 확장하면 전반 보상(0.557→0.821), instruction alignment(0.613→0.869), 궤적 품질과 룰 기반 점수(0.576/0.467→0.849/0.785)가 동시 상승하며 평균 도구 호출 수도 기대에 가까워져(약 1.32→5.44) 복잡한 멀티툴 워크플로 학습 효과가 확인됩니다.



### Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation (https://arxiv.org/abs/2607.06564)
Comments:
          14 pages, 7 figures. Project website: this https URL

- **Prior Approaches**: 기존 VLA는 2D 기반 비전-언어 사전학습을 정책에 결합해 성능과 일반화를 끌어올렸지만, 실제 조작에는 3D 기하 및 공간 추론이 핵심이다. 3D를 넣는 방식은 (1) point cloud/voxel/multi-view를 직접 인코딩하되 대규모 3D 데이터·foundation encoder 부족에 막히거나, (2) 2D feature를 3D로 lifting하거나 3D를 multi-view로 투영하는 과정에서 기하 정보가 손실돼 지리 정합성과 확장성이 떨어진다. 또한 일부 방법은 예측 기반으로 미래 상태를 다루지만, 진화하는 3D 기하와 시계열적으로 일관된 action chunk를 함께 학습하도록 설계된 접근은 제한적이다.

- **Core Contribution**: Lift3D-VLA는 VLA에 명시적 3D point cloud reasoning을 통합하고, 시간적으로 일관된 action generation까지 한 프레임워크에서 해결하려는 시도다. 핵심은 (i) 3D 토큰을 pretrained 2D positional embeddings와 기하적으로 정렬해 3D encoding 시 정보 손실을 줄이는 lifting 전략과, (ii) 현재 point cloud를 복원하면서 미래 기하 변화를 예측하는 Geometry-Centric Masked Autoencoding(GC-MAE)로 3D 구조와 물리 다이내믹스를 동시에 학습하는 구조다. 여기에 LLM 레이어를 활용한 layer-wise temporal action modeling을 얹어 동적 환경에서의 시간 일관성을 강화한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) pretrained 2D positional embedding과 3D 좌표를 정합시키되 왜곡을 줄이는 것, (2) 3D의 정적 구조뿐 아니라 시간에 따른 진화를 self-supervised 방식으로 학습하는 것, (3) action chunk가 장기적으로 서로 모순되지 않게 시계열 의존성을 만드는 것이다. 저자들은 가상 평면을 여러 장 두고 3D 점을 해당 평면 좌표로 투영한 뒤 camera extrinsic으로 front plane을 고정해 2D positional embedding을 재사용하는 방식으로 첫 난제를 완화한다. 둘째로는 GC-MAE에서 masked point reconstruction(정적 CD loss)과 future geometric prediction(다음 프레임의 기하 예측)을 이중 목표로 구성해 “복원-예측”을 강제한다. 마지막으로는 action decoding을 LLM 마지막 층에만 의존하지 않고 중간~깊은 층 표현을 연속적으로 써서 각 레이어가 action step을 담당하는 계층적 layer-wise 예측으로 temporal coherence를 확보한다.

- **Empirical Impact**: Lift3D-VLA는 시뮬레이션 22개 작업과 실제 로봇 8개 작업에서 MetaWorld와 RLBench의 평균 success rate를 각각 기존 SOTA 대비 10.8%, 11.1% 높였고, 실제 환경의 최강 baseline보다 4%p 앞섰다. 또한 unseen object/background/lighting으로의 일반화가 관찰되며, out-of-distribution perturbations에 대해서도 더 강한 견고성을 보였다. 장기 horizon 조작(예: 조건이 계속 바뀌는 상황에서 팬 위 달걀을 반복적으로 스쿱)에서도 시간 일관된 행동 생성이 효과적으로 동작하는 점이 이 연구의 실용적 의미로 평가된다.



### Bridging Physical Reasoning and Task Generalization via Visual Action Outcome Reasoning Alignmen (https://arxiv.org/abs/2607.06522)
Comments:
          ICML'26 Workshop RLxF: Reinforcement Learning from World Feedback

- **Prior Approaches**: 기존 VLM 물리 추론은 SFT로 전문가 CoT의 문장 형태를 따라 하거나, success-driven RL로 작업 성공만 최적화하는 두 흐름이 주류였다. 하지만 SFT는 추론이 실제 물리 결과에 결부되지 않아 그럴듯하지만 물리적으로 모순되는 CoT를 만들기 쉽고, RL은 희소·잡음 보상 때문에 CoT를 우회하는 shortcut으로 기울기 쉽다. 그 결과 reasoning과 action의 정렬이 깨져 보이지 않는 과제/환경에서 취약해진다.

- **Core Contribution**: 이 논문은 VAORA(Visual Action Outcome Reasoning Alignment)라는 새로운 보상 설계를 제안해, 물리 추론의 두 실패 모드(환각 CoT, reasoning-action misalignment)를 동시에 직접 억제한다. Visual Alignment Reward는 행동과 무관하게 초기 시각 맥락에 추론을 고정해 환각 CoT를 줄이고, Visual-Action Alignment Reward는 모델이 수행한 행동이 유발한 결과(사후 시각 outcome)에 추론을 접지시켜 reasoning과 행동 간 간극을 완화한다. 또한 안정적인 학습을 위해 pre-trained in-domain expert 에이전트로 성공확률을 추정해 smooth하고 dense한 보상 신호로 보강한다.

- **Technical Challenges**: 핵심 기술적 난제는 continuous action 상호작용 환경에서 보상이 본질적으로 sparse하고 noisy라 학습이 붕괴하기 쉽다는 점이다. VAORA는 (1) 추론 텍스트를 구조화된 symbolic space로 파싱하고, (2) 초기 장면/사후 outcome에서 ground-truth symbolic state를 추출한 뒤, (3) grounding·collision·placement 등 사건/관측 단위로 추론 일치도를 계산하는 보상으로 reasoning을 “검증 가능한 형태”로 강제한다. 더 나아가 action 결과 정렬 보상은 DQN 성공확률 게이팅으로 플라우저블한 행동일 때만 보상을 강화해 shortcut 학습과 붕괴를 줄이도록 구성했다.

- **Empirical Impact**: PHYRE의 unseen task 및 Virtual Tool의 unseen environment에서 VAORA는 오픈소스는 물론 다수의 closed-source 대비 성능을 크게 끌어올리며, DQN이 거의 전이되지 않는 설정에서도 환경-비의존적인 일반화 신호를 보여준다. CRAFT VQA에서도 descriptive을 넘어 counterfactual/causal 범주에서 개선 폭이 커, 단순 실행 접지(granularity)를 넘어 상호작용 결과 기반의 인과적 이해가 확장됨을 시사한다. 또한 보상 분해 분석에서 SFT는 grounding만 전이되고 placement/collision은 무너지는 반면, +EG+VAORA는 이를 복원해 성능 향상의 원인을 “reasoning을 실제 action outcome에 정렬하는 누락 신호”로 명확히 설명한다.



### Assessing the Operational Impact of Poisoning Attacks over Augmented 3D Point Cloud Public Datasets for Connected and Autonomous Vehicles (https://arxiv.org/abs/2607.06484)
Comments:
          Accepted for presentation at SECRYPT 2026

- **Prior Approaches**: 기존 연구는 데이터 증강이 잡음이나 라벨 오류를 완화하는 ‘sanitizing 효과’를 낸다고 주장해왔고, GAN 기반 증강과 함께 백도어/강건성 개선 가능성이 보고돼 왔습니다. 다만 이런 결과는 대체로 2D 이미지 중심의 제한된 벤치마크이거나, 증강 전에 검증 단계로 일부 데이터를 걸러내는 파이프라인처럼 조건이 까다로운 경우가 많습니다. 3D point cloud에서는 증강이 오히려 공격 흔적을 강화할 수 있다는 점이 충분히 분석되지 않았습니다.

- **Core Contribution**: 이 논문은 3D point cloud에서 GAN 기반 data augmentation이 poisoning과 backdoor의 영향(공격 샘플 수 증가/백도어 증폭)을 실제로 어떻게 바꾸는지에 초점을 둡니다. 특히 clean-label poisoning이 포함된 학습 데이터에 증강을 적용했을 때, 공격이 ‘증강에 의해 희석’되는지 ‘전파·증폭’되는지를 분류 성능 저하와 operational impact로 연결해 평가합니다. 또한 재현성을 위해 코드와 데이터(증강·poison·분류 전 과정을 포함)를 공개합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 3D point cloud에서 증강이 분포 모드를 어떻게 바꾸는지, (2) 그 변화가 공격의 효과로 어떻게 전파되는지, (3) 단순 정확도 하락을 넘어 실제 의사결정 기능까지 어떤 식으로 영향을 미치는지 동시에 정량화하는 것입니다. 저자들은 ModelNet의 이진 분류(타깃/비타깃) 설정에서 primary class 학습 샘플에 0~40%까지 clean-label poisoning을 주입하고, 증강 시나리오에서는 3D-GAN이 poisoning이 섞인 데이터까지 생성하도록 설계해 비교합니다. 이어 ASR(Attack Success Rate)을 operational impact 전파 모델의 초기 확률로 사용해, 분류기 저하가 Decision Making 기능의 손상 가능성으로 어떻게 매핑되는지 계산합니다.

- **Empirical Impact**: 실험 결과, poisoning 비율이 커질수록 두 시나리오 모두 MCC와 F1이 하락하고 ASR이 상승하지만, 증강을 넣은 경우 저하가 더 크게 나타납니다(예: 최고 주입률에서 ASR이 5.8%→17.6% 수준으로 확대). 이는 data augmentation이 공격을 ‘정화’하기보다, 공격자가 악용하는 특징/분포 모드를 학습에서 더 두드러지게 만들어 결과적으로 의사결정 기능에 더 큰 리스크를 유발할 수 있음을 시사합니다. 따라서 CAV용 3D point cloud 파이프라인에서 공개 데이터+GAN 증강 조합은 공격 표면을 키울 수 있으므로, poisoning-aware augmentation과 안전 검증 체계의 필요성이 강화됩니다.



### WristMimic: Full-Body Humanoid Control with Wrist-Guided Manipulation (https://arxiv.org/abs/2607.06438)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 인간-물체 상호작용 데모를 물리 시뮬레이션으로 전이할 때는 모션 캡처 기반 관절 궤적을 그대로 재생하는 방식이 흔했지만, 손처럼 접촉이 지배적인 구간에서는 위치 궤적만으로 접촉력/물체 동역학을 담기 어렵다. 그 결과 손가락 자세를 촘촘히 지도하거나(풀 핑거 슈퍼비전) 접촉용 보상/태스크별 설계를 추가해야 해 확장성과 데이터 의존성이 커졌다. 또한 손 수준 접근은 있어도 전신 제어와 결합해 단일 정책으로 retargeting을 안정화하는 데 한계가 있었다.

- **Core Contribution**: 이 논문은 WristMimic으로 전신 제어를 두 레짐으로 분리한다: 접촉이 없는 신체 파트는 kinematic pose target으로 유도하고, 접촉이 풍부한 손가락은 물체 추적과 contact outcome을 통해 학습한다. 핵심 통찰은 wrist가 두 레짐을 잇는 ‘게이트’라는 점으로, wrist는 접촉에 비교적 덜 영향받으면서도 손가락의 전역 자세와 그립 접근성(affordance)을 결정한다. 따라서 손가락 자세에 대한 직접 슈퍼비전을 줄이고도 유사하거나 더 나은 조작 성능을 노린다.

- **Technical Challenges**: 직접 손가락 자세를 지도하지 않으면 접촉 순간의 물리적 제어가 불안정해져, 단순히 오브젝트/컨택 보상만으로는 wrist 배치가 어긋날 위험이 크다. 이를 해결하기 위해 저자들은 wrist를 정밀하게 ‘배치’하는 장치를 설계하는데, 접촉 전후로 time-varying reward weight를 조절해 wrist 정렬을 우선하고 근위 팔(arm) 제약은 완화한다. 더불어 접촉 윈도우를 단계(approach/grasping/stabilization)로 나누고 phase-specific reset thresholds(더 빡빡한 wrist 기준 포함)로 탐색 공간을 현실적으로 제한해 모캡 오차에도 재현성을 확보한다.

- **Empirical Impact**: ParaHome과 OMOMO 두 데이터셋에서 10,000회 롤아웃 규모로 평가했으며, 성공률과 물체 위치/회전 오차 및 기준 접촉 유지율을 기준으로 성능을 검증한다. 요지는 손가락의 직접 kinematic supervision 없이도 wrist 제어를 통해 손가락-물체 상호작용을 충분히 유도해, 기존 풀 핑거 슈퍼비전 기반 방법과 비슷하거나 상회하는 결과를 보인다는 점이다. 또한 서로 다른 hand embodiment 사이에서도 finger-agnostic retargeting이 가능함을 실험적으로 뒷받침한다.



### TILDE: TILt-based Distributional Erasure for Concept Unlearning (https://arxiv.org/abs/2607.06432)
- **Prior Approaches**: 기존 개념 unlearning은 score suppression, anchor 기반 편집, reward/선호 최적화, trajectory steering, GFlowNet 기반 샘플링 등으로 “잊기” 자체는 달성하지만, 업데이트 후 분포가 어디로 이동하는지(사후 분포 목표)가 명시적이지 않은 경우가 많습니다. 그 결과 강한 지우기 과정에서 의미적으로 인접한 benign 개념의 손상(컬래터럴 데미지)이나 다양성 붕괴 같은 retain 실패가 함께 발생하기 쉽습니다. 또한 anchor를 목적지로 쓰는 방식은 고정·편향·순차 삭제 누적 등으로 불안정해질 수 있습니다.

- **Core Contribution**: TILDE(TILt-based Distributional Erasure)는 개념 unlearning을 “잊기 제약” 아래에서 사후 조건부 분포가 pretrained 분포에서 최소로 벗어나도록 정렬하는 분포 정렬 문제로 재정의합니다. 즉, replacement 개념을 지정하거나 단일 안전 출력의 모드를 찾는 대신, prompt별로 concept-expressing 이미지의 확률 질량을 줄이면서 benign 영역의 상대적 확률 질량은 보존하는 최소-이탈(minimal-deviation) 목표를 명시적으로 세웁니다. 이를 통해 unlearning의 핵심인 effective forgetting과 distributional fidelity, local preservation을 한 프레임에서 동시에 겨냥합니다.

- **Technical Challenges**: 문제는 (1) “어떤 이미지를 잊을지”를 에너지로 모델링하고 (2) 그 에너지 기울이기(energy tilt)를 diffusion 생성 과정에 맞게 학습 가능한 형태로 구현하는 것입니다. TILDE는 CLIP concept evidence를 thresholded forget energy로 만들어 임계값 이하에서는 페널티를 주지 않게 설계해, 주변 benign 개념까지 함께 깎이는 현상을 줄입니다. 또한 residual ∇∇-GFlowNet 기반으로 pretrained 디노이저에 대한 잔차 score correction만 학습해, terminal energy로 정의된 Gibbs형 타깃 샘플링을 diffusion latent space에서 효율적으로 근사합니다.

- **Empirical Impact**: Stable Diffusion v1.5에서 약 60개 개념(오브젝트/캐릭터/아트 스타일/누드)을 대상으로, forgetting 정확도와 함께 related/general retention 및 분포 정합성(FID, 그리고 retain-only 기준에 대한 FADE)까지 폭넓게 평가한 결과 TILDE가 기존 베이스라인 대비 더 강한 잊기와 더 나은 보존을 동시에 달성했습니다. 특히 “강한 지우기 ≠ retention 붕괴”라는 실패 모드를 줄이며, 분포 수준에서 gold-standard retain-only에 더 가깝게 정렬된다는 점을 FADE로 보였다는 것이 의미가 큽니다. 연구진은 VLM 기반 자동 평가(Qwen2.5-VL)로 다양한 개념 유형에서도 일관된 향상을 관측했다고 보고합니다.



### Learning to Throw Objects Safely in Multi-Obstacle Environments (https://arxiv.org/abs/2607.06388)
Comments:
          This paper has been presented at the IEEE International Conference on Robotics & Automation (ICRA), 2026

- **Prior Approaches**: 로봇 던지기는 로봇의 작업 반경을 넘어 빠른 배치를 가능하게 하지만, 기존 TossingBot 같은 방법은 장애물이 없는 환경을 주로 가정했습니다. 또 다른 접근은 수학적/수작업 모션 커널에 의존해 클러터 환경에서의 적응성이 떨어지거나, 장애물 정보를 상태에 직접 넣어 장애물 수가 늘면 스케일이 붕괴하는 문제가 있었습니다. 즉 “안전한 탐색+장애물 회피+일반화”를 동시에 만족시키기 어렵다는 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 장애물이 무작위로 배치된 장면에서 목표 바스켓에 던지되 충돌을 피하는 문제를 safe reinforcement learning으로 정식화합니다. 핵심 기여는 potential field state representation(PFR)으로, 바스켓의 attraction과 장애물의 repulsion을 고정 크기 grid에 함께 인코딩해 장애물 개수/배치가 달라도 단일 정책이 일반화하도록 한 점입니다. 또한 kinesthetic teaching으로 던지기 커널을 안전하게 초기화한 뒤 RL이 커널 파라미터를 조절하도록 설계해 초기 학습의 위험을 줄였습니다.

- **Technical Challenges**: 클러터 환경에서 RL이 처음부터 무작위로 탐색하면 충돌 위험이 커지며, 장애물 상태를 벡터로 직접 넣는 방식은 차원 증가로 학습이 비효율적입니다. 논문은 kinesthetic teaching으로 safe 커널을 부여해 unsafe exploration을 완화하고, 장애물 정보를 EPR(Explicit Pose Representation) 대신 PFR의 고정 차원 grid로 바꿔 확장성과 물리적으로 의미 있는 구조를 확보했습니다. 정책은 SAC, DDPG, TD3 중 SAC에서 가장 안정적인 성능을 보이며, PFR은 CNN 인코더로 잠재장 공간 구조를 보존하도록 구성했습니다.

- **Empirical Impact**: 시뮬레이션과 실로봇 실험 모두에서 PFR 기반 정책이 EPR 대비 높은 성공률과 더 나은 스케일링을 보였습니다. 특히 SAC 기준으로 장애물 수가 늘어도 성공률이 견조하게 유지되며, 보이지 않았던 throwable object(예: banana, coke can, sneaker)에 대한 일반화도 관찰됩니다. 실로봇에서는 unseen object까지 포함해 클러터 장면에서 최대 90% 성공률을 보고했으며, 이는 sim-to-real transfer가 실용 수준임을 시사합니다.



### Training-Free Acceleration for Vision-Language-Action Models with Action Caching and Refinemen (https://arxiv.org/abs/2607.06370)
- **Prior Approaches**: VLA 모델은 비전·언어 입력을 받아 로봇의 연속 동작을 생성하며, 확산(diffusion)이나 flow matching 기반 액션 헤드를 붙이는 방식이 최근 성능을 이끌고 있다. 다만 이들 flow 기반 VLA는 액션을 만들기 위해 여러 번 velocity field를 반복 평가해야 해서, 제어 루프에서 실시간 지연의 병목이 된다. 기존 가속은 계층/토큰 가지치기나 특징 캐싱처럼 계산량을 줄이거나 일부 재사용하는 접근이 많았지만, 액션 헤드의 반복 denoising(=flow 적분) 자체를 근본적으로 생략하긴 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 training-free이면서 plug-and-play 형태의 외부 캐시인 ActionCache를 제안해, 과거에 성공적으로 생성된 액션 청크를 다음 생성의 warm-start로 재사용한다. 핵심은 warm-start를 ‘시간 연속성’ 휴리스틱이 아니라, ‘출력 공간(action space)의 재사용 가능한 생성 결과’를 이웃 검색(retrieval)으로 찾는 문제로 재구성한 점이다. 모델 가중치나 액션 헤드 구조를 바꾸지 않고도(추가 학습 모듈 없이) 지연을 줄이면서 성공률을 유지하도록 설계했다.

- **Technical Challenges**: 주요 기술적 과제는 (1) 다중 양상(multimodal) 문맥을 잘 대표하는 캐시 키를 매 시점 계산 부담 없이 만들고, (2) 캐시가 틀릴 때의 안전성을 확보하는 것이다. 논문은 VLM 백본의 출력 임베딩을 기반으로 sparse ternary random projection으로 키를 압축하고, cosine similarity로 Top-1 이웃을 찾되 hit threshold 기준으로 보수적 분기(히트면 소수 NFE로 refine/미스면 순수 Gaussian noise에서 전체 생성)를 적용한다. 또한 저장 단위를 내부 상태가 아닌 ‘액션 청크 그 자체’로 두어 특정 백본에 종속되지 않게 하면서, pending 버퍼와 보수적 커밋/폐기를 통해 성공 재사용만 반영한다.

- **Empirical Impact**: 실험은 시뮬레이션(VLABench, LIBERO)과 실로봇(SO-101)에서 진행됐고, ActionCache는 low-latency 구간에서도 높은 task success를 유지하며 속도를 크게 끌어올렸다. π0.5와 GR00T-N1.6에서 flow 기반 VLA 액션 헤드에 대해 최대 11.75× 및 34.43× 가속을 보고했으며, 캐시 오버헤드만 남기는 극단적인 설정에서도 지연이 2ms 이하(모델별 1ms 수준)로 작게 나타났다. 캐시 히트 품질(Top-1 cosine similarity), hit threshold, 캐시 용량/교체정책이 성공률–지연–메모리의 트레이드오프를 좌우하는 제어 레버임을 상세 분석해, cross-task 재사용 가능성과 함께 실용적 튜닝 방법도 제시한다.



### OrchardBench: A Physically-Grounded, GPU-Parallel Apple-Orchard Simulation Benchmark for Agricultural Robotics (https://arxiv.org/abs/2607.06337)
- **Prior Approaches**: 기존 농업 로보틱스는 현장 실험 의존도가 높아 재현성과 비용 문제가 컸습니다. 시뮬레이션은 GPU-parallel 물리 학습을 가능케 했지만, 대부분의 과수 환경은 딱딱한 물체나 정적인 식물 지오메트리 위주라 ‘접촉-굴곡-파손-열매 분리’ 같은 생물학적 물리성을 충분히 담지 못했습니다.
또한 컴퓨터그래픽/기능-구조 식물 모델은 정교한 외형을 제공하더라도 로봇의 힘에 대한 동역학, 파손 거동, 열매 탈착 메커니즘이 결여되어 현장 반복을 대신하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: OrchardBench는 사과 과수의 나무를 물리적으로 구동되는 ‘breakable plant’로 모델링한 GPU-parallel 시뮬레이션 벤치마크를 제안합니다. 핵심은 가지를 Euler-Bernoulli beam 이론 기반의 합성 탄성/감쇠 관절로 만들고, 휨 모멘트가 파괴 한계를 넘으면 자유 힌지처럼 떨어지게 하며, 열매는 문헌 기반의 인장 임계 힘에서 줄기-스템 tether가 분리되도록 구현한 점입니다.
또한 잎이 흔들리며 과수 캐노피를 가리는 ‘moving, density-controllable foliage layer’를 포함해 인식 및 조작의 가장 어려운 변수(가림)를 제어 가능한 형태로 제공합니다.

- **Technical Challenges**: 과수는 수백 개 수준의 연성 관절과 다수의 자유 열매가 얽혀 GPU에서 수많은 환경을 동시에 안정적으로 계산하기 어렵습니다. OrchardBench는 Newton 엔진(MuJoCo-Warp 계열)을 활용하면서도 파손 시점에 모델을 재컴파일하지 않고, 관절 스프링의 암/바이어스 행을 즉시 꺼서 rupture를 in-place로 처리하는 방식으로 계산 효율과 수치 안정성을 동시에 노렸습니다.
여기에 사과는 전체 자유 강체로 두기보다 DOF를 줄인 tether-기반 분리 모델로 구성해 배치 시뮬레이션 비용을 낮추고, solver 안정장치(충격/속도 제한 등)로 과도한 에너지 주입을 방지합니다.

- **Empirical Impact**: 논문은 수확 임무에서 harvest completeness, throughput, plant damage를 포함한 메트릭을 정의하고, 잎 밀도·과일 하중·지형·캐노피 존·병렬성 조건별로 베이스라인 성능을 보고합니다. 그 결과 기하학적 과일 감지와 자율 수확 베이스라인은 감지한 과일의 약 40%를 수확하지만, 도달 가능한 과일 대비 약 1/8 수준만 수확해 안전하고 현실적인 물리 모델에서도 자율성 여지가 크다는 점을 보여줍니다.
즉, 이 벤치마크는 (1) 접촉/파손을 포함한 안전 검증, (2) sim-to-real 사전학습, (3) 가림 제어 기반의 perception 연구, (4) 농업 지표 중심의 최적화라는 다중 연구 프로그램에 바로 쓸 수 있는 공통 실험장으로 의미가 있습니다.



### Driving the Wrong Way: Leveraging Interpretability in End2End Autonomous Driving Models (https://arxiv.org/abs/2607.06328)
- **Prior Approaches**: 기존 end-to-end 자율주행은 perception·prediction·planning을 하나의 transformer로 통합해 NAVSIM 같은 open-loop 벤치마크에서 강한 성능을 보이지만, 모듈 경계가 사라져 내부 의사결정이 불투명해진다. 이에 따라 saliency/gradient 기반 설명이나 attention visualization은 입력의 “어디를 봤는지”는 보여주지만, 모델이 내부에서 “무엇을 개념으로 학습했는지”와 실패 원인의 기능적 관련성까지는 잘 드러내지 못한다. latent space 쪽 해석도 드물고, end-to-end에서 개념 수준으로 분해·인과 확인을 함께 하는 방식은 거의 없었다.

- **Core Contribution**: 이 논문은 Sparse Autoencoder(SAE) 기반 dictionary learning을 end-to-end 주행 모델의 feature space에 사후(post hoc) 해석 모듈로 결합해, 잠재표현을 의미 있는 희소 개념(semantic concepts) 조합으로 분해한다. 각 개념을 자연어로 일관되게 대응시키고, 후보 궤적의 점수(trajectory-level decision scores)에 어떤 개념이 기여하는지 직접 연결해 “결정 로직”을 노출한다. 또한 개념 단위 개입(intervention)으로 특정 개념을 억제/조작해 주행 의사결정을 수정할 수 있음을 제시한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) end-to-end 모델 내부에서 SAE를 주입할 적절한 latent 공간을 골라야 하고, (2) 얽힌(polysemantic) 뉴런을 monosemantic 개념 방향으로 희소 분해해야 하며, (3) 개념과 궤적 점수·하위 품질 지표 사이의 인과적 연결을 계산 비용 부담 없이 구현해야 한다. 이들은 점수 모듈 직전의 1D per-trajectory 표현을 SAE 입력으로 택하고, top-k 희소화와 dead neuron 방지(reanimation)로 재구성과 개념 유효성을 함께 확보한다. 이후 Concept Relevance Propagation(CRP)과 attribution 기반 circuit analysis를 조합해, 개념이 각 scoring head/PDM 성분에 주는 영향의 희소 회로를 구성하고 상위 후보에 대해 정확 개입으로 인과를 확인한다.

- **Empirical Impact**: 실험에서는 GTRS와 iPAD 모델의 latent 위에 다양한 SAE 아키텍처/하이퍼파라미터를 학습한 뒤 재구성 품질(cosine similarity, explained variance)과 문제 특화 정합성(ego correlation, ego probing), 그리고 EPDMS 같은 downstream 성능을 함께 비교한다. 결과적으로 TopK·Matryoshka·archetypal SAE 변형들 사이에서 dead neuron 활용도와 ego 관련 분해가 달라지며, 아키텍처 선택이 해석 가능성과 성능 보존의 균형을 좌우함을 보여준다. 더 나아가 개념 수준 개입이 충돌 회피, drivable area, traffic light compliance 같은 하류 주행 지표를 계량적으로 개선해, 설명이 단순 상관이 아니라 기능적·수정 가능한 구성요소임을 뒷받침한다.



### UI2App: Benchmarking Visual Interaction Inference in Executable Web Application Generation (https://arxiv.org/abs/2607.06306)
- **Prior Approaches**: 기존 웹 UI 생성 연구는 스크린샷이나 텍스트를 입력으로 받아 시각적 유사도(visual fidelity)에 주로 초점을 맞춰왔다. 그러나 이런 평가는 동작 가능성이나 페이지 간 상태 동기화 같은 상호작용이 실제로 구현되는지까지는 검증하지 못해 ‘그럴듯한 가짜 UI’ 문제가 남았다. 또 텍스트 기반 방식은 복잡한 프롬프트 의존성과 레이아웃/시각적 일관성, 교차 페이지 상호작용을 자연어로 정밀 지정하기 어려운 한계를 가진다.

- **Core Contribution**: 이 논문은 스크린샷만으로 애플리케이션의 행동을 복원하는 ‘interaction inference’를 측정하는 최초의 벤치마크 UI2App을 제안한다. 327개의 스크린샷을 45개의 state-coherent 세트(실행 가능한 멀티 라우트 웹앱)로 구성하고, 모델이 텍스트나 행동 지시 없이도 실행 가능한 코드를 생성하도록 과제를 정의한다.

- **Technical Challenges**: 문제의 핵심 기술 난점은 정적 스크린샷이 행동을 충분히 한정하지 못해, 같은 화면 상태가 서로 다른 올바른 구현을 가질 수 있다는 점이다. 이를 해결하기 위해 4개 평가 축(실행 가능성, 내비게이션 도달성, 시각적 충실도, IIS) 중 IIS를 7개 상호작용 카테고리와 상태-복잡도(scope)로 세분화해, 기능적 정합성과 상태 관리 복잡도를 루브릭 기반으로 채점한다. 또한 실행 실패는 0으로 처리해 지표 비교의 기준선을 유지하고, 경로별 URL 로딩으로 페이지 구현과 라우팅을 분리 평가한다.

- **Empirical Impact**: 6종 frontier VLM을 실험한 결과, 시각적 충실도 리더가 IIS에서도 상위에 오르지 못하는 ‘역상관’이 뚜렷했으며 VFS 리더는 IIS에서 7.5점 수준에 그쳐 IIS 리더 대비 5.2배 뒤처졌다. 특히 교차 라우트 상태(cross-route persistence, S3 scope)는 전 모델에서 병목으로 나타났고 절반이 해당 차원에서 정확히 0점을 기록했으며 최고 성능도 S3에서 21.6점에 머물렀다. 저자들은 UI2App이 스크린샷-기반 상호작용 복원과 구현의 “S3 headroom”을 줄이기 위한 연구 테스트베드가 될 수 있음을 강조한다.



### CMDR: Contextual Multimodal Document Retrieva (https://arxiv.org/abs/2607.05927)
Comments:
          Accepted by ECCV 2026; project page: this https URL

- **Prior Approaches**: 기존 멀티모달 문서 검색 벤치마크는 질의-단일 페이지 간의 단순 어휘/의미 매칭을 주로 평가해, 여러 페이지에 걸친 문서 맥락을 이용한 간접적 추론 능력을 검증하지 못했다. 또한 대부분의 방법이 페이지를 독립 인코딩해 문서 전역 구조나 cross-page dependencies의 이점을 과소평가하는 한계를 보였다. 멀티홉 계열도 대체로 명시적 매칭에 기반한 직접 검색 성격이 강해, “질의에 직접 드러나지 않는 관련 페이지를 찾아야 하는” 문제를 충분히 다루지 못했다.

- **Core Contribution**: 논문은 Contextual Multimodal Document Retrieval(CMDR) 태스크를 제안하고, 이를 평가하는 CMDR-Bench를 공개한다. CMDR-Bench는 6개 도메인 255개 장문 문서(평균 183.5페이지)에서 수작업으로 큐레이션된 800개 질의로, 페이지 간 맥락 모델링이 필수인 간접 검색을 요구한다. 모델 측면에서는 문서 여러 페이지를 함께 인코딩하되 페이지 수준 임베딩으로 분리해 문맥을 반영하는 CMDR-Embed와, 이를 학습하는 Contextual Multimodal Contrastive Learning(CMCL)을 함께 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 “여러 페이지를 함께 인코딩해 맥락을 얻는 것”이 동시에 동일 문서 내 페이지 표현을 섞어 페이지별 구별력(discriminability)을 떨어뜨린다는 점이다. 이를 해결하기 위해 CMDR-Embed는 chunk-then-split 전략으로 컨텍스트를 공유하되 페이지 임베딩은 분리해 Late Interaction(LI)로 질의-페이지 매칭을 수행한다. 학습에서는 CMCL이 두 종류의 context-aware hard negatives(같은 chunk 내 In-Chunk Negatives, 같은 문서 내 먼 chunk의 In-Document Negatives)를 조합해 맥락 활용과 페이지 구별력을 균형 있게 최적화한다.

- **Empirical Impact**: 실험 결과 CMDR-Embed는 비문맥(non-contextual) 임베딩 대비 유의미하게 성능이 개선되며, 동일 학습 데이터 조건에서도 컨텍스트 모델이 우위를 보였다. CMDR-Bench의 카테고리 분석에서는 특히 CR/MR처럼 참조 해석과 다중 페이지 집계가 필요한 영역에서 기존 멀티모달 검색기가 더 큰 어려움을 드러냈고, 이 격차를 CMDR-Embed가 완화한다. 또한 CMCL의 hard negative 설계와 LI의 멀티벡터 구조가 성능 향상에 일관되게 기여함을 보여, 장문 문서 검색에서 context-aware multimodal embeddings의 필요성을 실증적으로 강조한다.



### GraspIT: A Dataset Bridging the Sim-to-Real gap and back for Validated Grasping SE(3) Pose Generation (https://arxiv.org/abs/2607.05869)
Comments:
          Preprint, release soon

- **Prior Approaches**: 기존 로보틱스 그리핑 데이터셋은 보통 RGB-D 관측, 그립 품질 라벨, 시뮬레이션-실세계 연결 중 일부만 제공해왔다. 그 결과 물리적으로 검증된 품질 기준과, 시뮬에서 생성한 후보를 실세계로 옮기는 원칙적 브리지가 동시에 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 GraspIT로, NVIDIA Isaac Sim에서 테이블탑 장면을 물리 기반 슬립 테스트 4단계로 주석해 연속 품질 점수와 궤적-도달성 체크를 함께 만든다고 제안한다. 또한 Real↔Sim 루프 백프로젝션으로 100개 실세계 장면에 라벨을 매핑해, 시뮬-실세계 일관성을 확보한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 시뮬에서 만든 후보 그립의 물리적 타당성을 품질 점수로 정량화하고, (2) 통과/실패를 넘어 실세계에서의 학습 신호로 안정적으로 전사하는 것이다. 논문은 병렬 Franka Panda 인스턴스 위에서 slip-test로 graded hard negatives를 구성하고, Real↔Sim 백프로젝션으로 연속 품질 라벨을 실세계 100장면에 대응시켜 이 문제를 해결한다.

- **Empirical Impact**: GraspIT는 약 2.3M 후보 중 83%를 good(s≥0.50)으로 만들고, 남은 17%는 force-closure는 통과하되 slip-test에서 실패한 graded hard negatives로 활용한다. 최종 공개물은 약 316k annotated RGB-D 프레임 세트(시뮬 1035, 실세계 100)와 instance masks, 6-DoF pose, 물리 객체 속성, 점수화된 6-DoF grasp를 포함하며, Docker 및 오픈소스로 공개돼 tabletop manipulation 정책 학습과 behavior cloning에 바로 쓰일 수 있다.



### FORGE: Towards Functional Tool-Use Generalization via Keypoint Trajectory Reasoning (https://arxiv.org/abs/2607.05780)
Comments:
          15 pages, 8 figures, 6 tables

- **Prior Approaches**: 로보틱스의 기존 일반화 연구는 주로 장면/카테고리 수준의 시각 변동이나, 또는 cross-embodiment처럼 로봇 형태를 바꾸는 문제에 초점이 맞춰져 있었다. 반면 도구 사용에서 ‘동일한 기능’을 보장하는 functional generalization은, 도구 모양은 달라도 접촉 지점과 모션이 바뀌어야 한다는 점에서 perception-to-action gap이 커 end-to-end 학습이 쉽게 무너진다.

- **Core Contribution**: 논문은 functional generalization을 ‘도구는 바뀌지만 기능(타격)을 동일하게 수행’하는 문제로 정식화하고, 핵심 난제로 시각 유사성이 행동 공간으로 그대로 전이되지 않는 불일치를 제시한다. 이를 해결하기 위해 FORGE를 제안하며, 기능 추론과 동작 실행을 분리해 먼저 일반화 가능한 2D keypoint trajectories를 예측한 뒤, 제한된 시연으로 로봇 행동에 grounding한다.

- **Technical Challenges**: 기여를 위해서는 (1) 도구별 외형에 과적합되지 않으면서도 (2) 접촉 지점과 시간에 따른 운동 구조를 담는 intermediate representation을 찾아야 했다. 저자들은 affordance images, human video prompts, 2D keypoint trajectories를 비교했고, keypoint trajectories가 function 표현력과 action groundability를 가장 잘 균형한다는 실험 결과로 선택했으며, stage1은 action-free data로 conditional flow matching 예측, stage2는 action-labeled data로 conditional flow matching 기반 execution policy 학습(그리고 예측 오차 견고화용 perturbation)으로 구성했다.

- **Empirical Impact**: 일곱 가지 도구의 hitting-function 벤치마크에서 FORGE는 unseen tools에 대해 state-of-the-art 대비 평균 success rate를 2배 이상(2X+) 끌어올리며 시뮬레이션과 real world 모두에서 일관된 성능을 보였다. 특히 end-to-end visuomotor 정책들은 unseen 도구의 올바른 hitting region 정렬에 실패하지만, FORGE는 keypoint 기반 중간 계획이 접촉 위치 접근을 구체적으로 안내해 실패를 줄이는 방식으로 의미를 입증한다.



### FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models (https://arxiv.org/abs/2607.05711)
- **Prior Approaches**: 확산 모델은 후학습(post-training)으로 커스터마이징, 강화학습, 증류 등을 지원하지만, 대형 모델의 메모리 사용량과 느린 학습 속도 때문에 제약이 컸습니다. LoRA는 학습 파라미터를 줄이지만 백본 가중치의 메모리 병목은 그대로 남고, QLoRA는 4-bit 양자화를 적용해 메모리를 줄이면서도 온라인 dequantization으로 계산 비용이 늘 수 있습니다. 즉, 기존 PEFT들은 메모리-속도 트레이드오프를 근본적으로 해결하지 못했다는 한계가 있었습니다.

- **Core Contribution**: FourTune은 확산 모델 후학습을 end-to-end W4A4G4(가중치/활성/그래디언트 4-bit)로 수행하는 효율 프레임워크를 제안합니다. 표준 LoRA에 수치 안정화(numerical stabilizer)용 frozen 분기와 함께, 양자화 민감 outlier를 분리해 극저비트에서도 학습이 수렴하도록 설계했습니다. 또한 backbone을 완전 4-bit로 굴리면서도 커스터마이징, 강화학습, 증류 전반에서 full-precision fine-tuning급 품질을 목표로 합니다.

- **Technical Challenges**: 4-bit 학습은 동적 범위가 좁아 outlier와 누적 양자화 오차로 gradient 폭주 같은 수치 불안정이 발생하기 쉽습니다. FourTune은 triple-branch 하이브리드 파이프라인에서 stabilizer 분기가 outlier를 처리하고, 나머지 backbone은 NVFP4 기반으로 4-bit 연산을 직접 수행하게 해 안정성을 확보했습니다. 이어 backward 단계의 transpose/재양자화 병목을 줄이기 위해 block-wise quantization으로 스케일 정렬 문제를 해결하고, LoRA·MLP 구간의 kernel fusion 및 GEMM-양자화 융합으로 메모리 대역폭 오버헤드를 낮췄습니다.

- **Empirical Impact**: 실험에서 FourTune은 FLUX.1-dev(12B) 기준 BF16 LoRA 대비 메모리 사용량을 최대 2.25× 줄이면서, end-to-end 학습 throughput을 최대 2.27× 높였습니다. 또한 커스터마이징에서는 정체성/스타일/일반 주제 품질이 full-precision LoRA와 유사하게 유지되었고, 강화학습과 증류에서도 동등 수준의 결과를 보여 4-bit 학습의 수치 충실도를 입증했습니다. ablation으로 stabilizer 유무에 따른 gradient 폭주 차이, block-wise 양자화의 품질 유지, kernel fusion의 단계별 속도 향상을 확인하며 ‘효율-성능 간 격차’를 실증적으로 메웠다는 점에서 의미가 큽니다.



### IMR: Iterative Mode-World Weighted Regression for Multi-Agent Trajectory Prediction (https://arxiv.org/abs/2607.05705)
- **Prior Approaches**: 기존 multi-agent motion prediction은 예측 기반과 anchor 기반으로 크게 나뉜다. 예측 기반(QCNet, Forecast-MAE)은 복잡한 상황에서 mode collapse(다양성 붕괴)가 발생하기 쉽고, anchor 기반(MTR, TNT)은 이를 완화하는 대신 예측 정확도가 떨어지는 경향이 있다. 또한 QCNeXt 같은 proposal-refinement 디코딩은 초기 제안 궤적이 크게 틀리면 refinement가 오프셋 보정을 충분히 못 하는 문제가 지적된다.

- **Core Contribution**: 이 논문은 mode collapse와 정확도 저하의 trade-off를 동시에 다루기 위해 prediction-based 프레임워크에 mode-world weighted regression loss를 제안한다. 이 손실은 mode-wise 및 world-wise 회귀를 함께 가중해 학습하며, mode 다양성 유지와 함께 world ranking 정확도 및 top-1 confidence를 개선하는 데 초점을 둔다. 아울러 반복적 디코딩으로 제안 궤적의 초기 오류에 덜 민감한 구조를 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다중 에이전트 궤적에서 정답 모드가 여러 개인데도 학습 중 특정 모드로 수렴해버리는 문제와 (2) joint world confidence를 포함한 world ranking을 정확히 학습하는 문제다. 이를 위해 그래프 attention network 기반으로 에이전트-맵 및 상호작용을 동적으로 인코딩하고, 모드 회귀는 winner-takes-all 전략으로 선택 오차(ADE/MDE)를 기준화해 손실을 설계한다. 디코딩은 iterative decoder를 도입해 SS(세그먼트) 단계를 순차 생성하되, 각 단계 출력이 이전 단계 offset이 아니라 절대 위치 좌표가 되도록 하여 누적 오차 전파를 줄인다.

- **Empirical Impact**: 실험은 Argoverse 2에서 6초 예측 setting으로 수행됐고, 제안 방법은 다른 모델 대비 평균 BrierMinFDE6에서 이전 SOTA QCNeXt보다 0.06p 향상되며 1위를 기록했다. 단일 에이전트 벤치마크에서도 LOF와 비교해 경쟁력 있는 성능을 보였다. 시각화와 손실 비교 결과, 기존 world-wise 회귀에서 나타나는 mode collapse를 mode-world weighted regression loss가 완화하면서 정확도까지 함께 개선함을 확인했다.



### LLM-Driven Neural Network Generation with Same-Family Architecture Guidance: Disentangling Transfer and Adaptation (https://arxiv.org/abs/2607.05704)
Comments:
          10 pages, 1 figure, 14 tables

- **Prior Approaches**: LLM을 활용한 신경망 생성은 코드와 학습 레시피를 직접 탐색한다는 장점이 있지만, 무제한 생성은 종종 invalid 코드나 성능 저하를 동반해 신뢰성 문제가 크다. 기존 연구는 architecture generation, prompt 설계, hyperparameter tuning, 반복형 NAS 등을 다뤘지만, “소스 모델을 쓰면 실제로 더 좋아지는가”를 통제 실험으로 분리하기는 어려웠다. 또한 valid/accuracy를 함께 섞어 평가하면 생성 신뢰도와 모델 품질이 혼동될 수 있다는 한계도 있었다.

- **Core Contribution**: 이 논문은 약한 타깃(weak target) 모델을, 같은 계열의 더 강한 소스(source) 모델로부터 보강하는 좁은 설정을 제안한다. source-guided candidate-generation을 non-source control과 함께 동일 예산에서 비교하고, validity를 accuracy와 분리해 “유효한 후보를 얼마나 찾는지”와 “얼마나 잘 맞추는지”를 따로 본다. hp_copy 무삭제(ablation)로 hp_transfer가 단순 복사인지, LLM이 레시피를 실제로 적응(adaptation)하는지까지 메커니즘을 분해한다.

- **Technical Challenges**: 핵심 기술 난제는 LLM이 생성한 후보가 실행/학습 가능한 형태로 유지되도록 하는 validity 관리와, 소스가 준 정보가 실제 성능으로 연결되는지의 공정한 비교 설계다. 저자들은 candidate budget를 고정하고 파싱/스키마 검증, 지원하지 않는 하이퍼파라미터 키 제거, dataset-compatible transform 가드 같은 최소 결정적 수리만 적용해 무리한 의미 복구를 배제했다. 이어 best valid accuracy와 mean valid accuracy, 그리고 source-guided advantage를 함께 산출하고, 비교 쌍을 맞춘 paired bootstrap과 Wilson 구간으로 통계적 불확실성도 정리했다.

- **Empirical Impact**: CIFAR-10에서 가장 강한 source-guided 후보는 정확도 0.5049로 비소스 최선 0.2398 대비 +0.2651 향상을 보였고, 0.1254로 출발한 약한 타깃도 개선된다(5 epoch 재검증에서도 이득 유지). SVHN AlexNet에서는 DeepSeek-Coder-6.7B로 source-guided hp_transfer가 0.7880(비소스 0.2254)까지 올라 +0.5626의 큰 격차를 만들었다. hp_copy 분해 결과는 CIFAR-10이 recipe-transfer 성격(hp_copy도 상당한 이득)인 반면, SVHN AlexNet은 hp_copy가 실패하고 hp_transfer만 성공하는 recipe-adaptation 성격임을 보여, “소스 복사”만으로는 설명되지 않음을 입증한다.



### BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension (https://arxiv.org/abs/2607.05614)
Comments:
          Accepted at the 19th European Conference on Computer Vision (ECCV), 2026

- **Prior Approaches**: 기존 문서 이해 연구는 DLA와 KIE를 다뤄왔지만, 대부분은 고품질 라벨이 풍부한 언어(영어 중심)나 제한된 스키마에 의존해 저자원 언어로의 전이를 어렵게 만들었다. FUNSD·XFUND 같은 벤치마크는 형태 이해를 진전시켰지만, Bangla 정부 양식에 필요한 세밀한 폼 엔티티·공간 관계·키-값 구조를 포괄하는 공개 데이터는 부족했다.

- **Core Contribution**: 이 논문은 Bangla 정부 양식의 DLA와 KIE를 동시에 평가할 수 있는 벤치마크 BaFCo를 제안한다. 다중 페이지 정부 폼 200개를 모아 26개 세부 엔티티 타입과 관계 라벨을 구성하고, 5개 조밀도 축약 엔티티 세트도 함께 제공해 모델 성능을 ‘세밀함’ 관점에서 비교 가능하게 했다.

- **Technical Challenges**: 핵심 기술적 난제는 Bangla 폼에서 요구되는 (1) 세밀 엔티티 경계의 정확한 위치 지정과 (2) 키-값 및 필드 간 관계를 일관된 규칙으로 라벨링하는 작업이다. 저자들은 바운딩 박스 규칙·관계 제약·모호 케이스를 포함한 상세 가이드라인과 다단계 검수(코헨 κ≈0.974)를 통해 주관성을 줄이고, 이후 MLLM 평가를 위한 validator 기반 출력 스키마 검증 파이프라인을 구축했다.

- **Empirical Impact**: ChatGPT·Gemini·Claude·Qwen·Kimi 계열의 flagship MLLM을 zero-shot 및 chain-of-thought(CoT) 프롬프트로 평가한 결과, Bangla 폼에서 특히 DLA의 세밀 엔티티 국소화가 취약했고, 엔티티를 거칠게 묶을수록 mAP이 크게 개선됐다(예: Gemini 3 Pro에서 0.1177→0.2646 수준). 반면 KIE는 전반적으로 훨씬 높은 F1(예: Bangla에서 Gemini 3 Pro F1=0.848)로 나타났으며, 언어 영향은 DLA보다 KIE에서 더 뚜렷해 ‘최선 모델’이 언어별로 달라질 수 있음을 보여준다.



### SSA-3DGS: Unsupervised Removal of Screen-Space Artifacts for 3D Gaussian Splatting (https://arxiv.org/abs/2607.05598)
- **Prior Approaches**: 기존 Novel View Synthesis(NVS) 방법은 입력이 다중 시점에서 기하적으로 일관돼야 한다는 가정을 강하게 의존한다. NeRF-W, RobustNeRF, SpotlessSplats, Wild-GS, DeSplat 같은 연구는 world-space의 전이/잡음/방해물에는 대응하지만, 카메라 센서에 고정된 screen-space artifact(워터마크·UI·대시보드·렌즈 오염·손/가림)가 만들어내는 near-camera floaters 문제에는 취약하다. 또한 단일 이미지 복원·inpainting은 프레임별로 다른 질감을 생성해 다중 시점 일관성을 깨뜨리며, watermark 제거는 보통 수동 주석이나 페어 데이터가 필요해 확장성이 낮다.

- **Core Contribution**: 본 논문은 SSA-3DGS(unsupervised)로, 3D scene(깨끗한 3D)과 2D screen-space artifact를 동시에 분리·복원하는 비지도 프레임워크를 제안한다. 핵심 아이디어는 모든 학습 뷰에 공유되는 learnable 2D overlay(컬러 맵+알파 매트)를 3DGS 렌더 결과에 합성하고, 모션 패럴럭스를 이용해 정적 screen-space 요소를 3D 기하에서 디커플링한다. 테스트 시에는 overlay를 제거한 3D 렌더만 사용해 artifact-free 결과를 얻는다.

- **Technical Challenges**: joint optimization은 합성된 픽셀 오차만으로는 분해가 under-constrained라, 2D overlay가 장면의 정상 디테일까지 흡수하거나 노출을 재스케일하는 퇴화 해(위장된 overlay)로 붕괴할 수 있다. SSA-3DGS는 이를 막기 위해 sparsity regularization(알파 매트의 희소성)과 total variation(TV) regularization(overlay의 공간적 매끄러움/일관성)을 함께 걸어, 가능한 한 설명을 3D 기하에 맡기고 overlay는 진짜 screen-space artifact에만 사용되도록 유도한다. 또한 overlay는 시점마다 달라지지 않는 픽셀 좌표 텐서로 모델링해 screen-space 고정성을 반영하고, 학습 중에는 렌더-합성 그래프에 gradient를 동시에 흘려 분리를 달성한다.

- **Empirical Impact**: 합성 데이터(Mip-NeRF 360 기반)에서는 corrupted 입력에서 vanilla 3DGS가 크게 성능이 하락하는 반면, SSA-3DGS는 3D 기하 복원을 유지하며 PSNR 기준 최대 약 9 dB 개선을 보인다. DeSplat와 비교해도 PSNR/FLIP에서 더 일관되게 유리하며, MCMC densification까지 적용해도 artifact 흡수 덕분에 안정성이 크게 좋아진다. 더불어 물리적으로 발생한 실세계 self-captured 데이터(렌즈 오염/mud, 근접 가림/occlusion)에서도 SSA-3DGS가 기준선 대비 PSNR과 지각 품질 지표가 동반 개선되어, screen-space artifact 전용 접근이 실제 촬영 조건에서도 의미 있게 전이됨을 보여준다.



### GEM-Occ: From Visual Geometry Evidence to Embodied Semantic Occupancy Memory (https://arxiv.org/abs/2607.05543)
Comments:
          19 pages, 6 figures. Project page: this https URL

- **Prior Approaches**: 기존 실내 occupancy 연구는 주로 단일 뷰에서의 의미론적 점유 예측(semantic completion)이나 특정 실내 스케일의 room-level 추론에 머물렀습니다. 또한 데이터셋과 관측 형식(perspective RGB-D, pano)도 통합되지 않아, 긴 지평선의 인과적(causal) 의미 매핑과 free/unknown 구분, 재방문 안정성 같은 요구를 체계적으로 평가하기 어려웠습니다.

- **Core Contribution**: 이 논문은 ScanNet, ScanNet++, Matterport3D를 하나의 sparse semantic occupancy 포맷으로 통합하면서도 각 데이터의 native 관측 기하를 보존하는 HIOcc를 제안합니다. HIOcc는 local semantic occupancy prediction, room-level online occupancy mapping, building-level panoramic mapping의 3가지 평가 레짐을 제공해 긴 지평선 embodied mapping을 정량화합니다. 또한 GEM-Occ을 통해 관측에서 나온 순간적 증거를 영속 메모리로 변환하는 프레임워크를 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 점유/자유공간/미지(unknown)와 의미를 갖는 “영속적 지도”를 만들기 위해, 순간 관측 evidence를 중복·부정합 없이 누적하는 것입니다. GEM-Occ은 pointmap을 그대로 지도 상태로 축적하지 않고, 로컬 시각 기하 예측을 semantic Gaussian occupancy evidence와 free-space ray evidence로 변환한 뒤, visibility-와 uncertainty-aware causal update로 계층형 메모리에 융합합니다. 메모리는 local cache, room-level submap, building-level graph로 구성되며 Gaussian-to-occupancy splatting으로 언제든 질의할 수 있습니다.

- **Empirical Impact**: HIOcc 실험에서 GEM-Occ은 local occupancy 정확도뿐 아니라 온라인 맵 안정성, free-space reasoning, revisit consistency, building-level 확장성에서 이전 indoor occupancy 및 Gaussian 기반 baseline을 전반적으로 개선했습니다. 특히 room-level에서는 mIoU와 IoU가 상승했고, building-level panoramic 설정에서는 progress AUC와 revisit consistency까지 더 좋아졌습니다. ablation 결과로 semantic Gaussian evidence와 free-space ray evidence의 분리 및 confidence-aware 융합이 성능 향상의 핵심임이 확인됐습니다.



### $\mathbfλ$-VAE: Variance Equalization for Posterior Collaps (https://arxiv.org/abs/2607.05531)
Comments:
          21 total pages

- **Prior Approaches**: VAEs는 ELBO 최적화 과정에서 재구성 신호와 KL 정규화 사이의 균형이 깨지면 posterior collapse가 나타난다. 기존 연구는 KL 항 재가중, KL floor, aggregate posterior 매칭, 사후분포(encoder) 모형 확장, 학습 동역학 조정 등 다양한 처방을 제안했지만, 붕괴 원인을 하나로 통합해 설명하는 데는 한계가 있었다. 특히 여러 방법이 목적함수의 전역 수정에 머물러, 왜 특정 차원이 먼저 죽고 어떻게 그 경로가 고정되는지에 대한 일관된 계정이 부족했다.

- **Core Contribution**: 이 논문은 posterior collapse의 원인을 ‘gradient imbalance’와 ‘information gap’ 두 가지로 논리적으로 독립이지만 결합된 형태로 정식화한다. 두 원인은 동일한 붕괴 궤적을 공유하며, 정보 갭은 aggregate posterior와 prior 사이의 marginal mismatch와 대수적으로 동치임을 보인다. 그 위에서 reparameterization 단계만 비대칭적으로 수정한 λ-VAE를 제안해, 목적함수나 파라미터를 추가하지 않고도 두 원인을 동시에 완화한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) decoder의 재구성 그라디언트가 KL 복원 압력보다 먼저 사라져 안정점이 붕괴 쪽으로 이동하는 상황과, (2) 샘플링 병목이 encoder가 계산한 표현의 상당 부분을 버려 decoder가 입력 변화에 덜 민감해지는 상황을 동시에 다루는 것이다. 저자들은 붕괴 여부를 차원별 gradient ratio로 판별하고, information gap을 신호 대 잡음 관점과 marginal mismatch로 연결해 ‘붕괴를 쉽게 만드는 이유’를 정량화한다. λ-VAE는 샘플링 노이즈를 per-dimension exponent로 스케일링하되 KL 계산은 원래 분산으로 유지해, 붕괴 상태로 가는 안정 유인자를 ‘variance equalization’ 방향으로 이동시킨다.

- **Empirical Impact**: Binary MNIST, Binary Omniglot, CIFAR-10, CelebA-64 등에서 λ-VAE는 collapsed 차원 수를 크게 줄이고 정보 용량(information capacity)을 최대 2.8배까지 늘리며 재구성 품질을 개선(BPD 최대 +0.33)한다. 또한 near-identical BPD를 보이는 모델들이 latent 코드에 배분한 decoder 용량은 크게 달라질 수 있음을 PixelCNN 실험으로 보이며, BPD가 붕괴 진단자로서의 한계를 가진다는 점도 강조한다. 결과적으로 이 연구는 posterior collapse를 원인-경로-처방으로 연결하는 통합 관점과, reparameterization만으로 작동하는 실용적 개선책을 동시에 제공한다.



### BitFair: A 12nm Bit-Serial CNN Accelerator with Learnable Early Termination and Adaptive Bit Ordering for Ultra-Low-Power XR Vision (https://arxiv.org/abs/2607.05445)
Comments:
          Under review

- **Prior Approaches**: XR 웨어러블은 20ms 이하 motion-to-photon 지연과 수 와트 미만 전력 제약 때문에 신경망 추론에 쓸 수 있는 시간이 극도로 짧다. 기존에는 SNN이 이벤트 기반 희소성을 활용하지만, surrogate-gradient 학습 난이도와 CNN 대비 정확도 격차, 그리고 입력이 조밀해지면 효율 이점이 줄어든다는 한계가 지적돼 왔다. 또 bit-serial 가속기는 비트 수준 계산을 줄일 수 있지만, ReLU로 인해 발생하는 동적 sparsity를 제대로 활용하지 못해 불필요한 비트 연산이 남아 있었다.

- **Core Contribution**: BitFair는 XR용 초저전력·초저지연을 목표로, 소프트웨어-하드웨어 공동설계 방식의 bit-serial CNN 가속기를 제안한다. 핵심은 (1) 레이어별 learnable bit-level early termination threshold로 ReLU 출력이 0이 될 가능성을 부분합에서 예측해 남은 비트 계산을 조기 중단하는 것과, (2) 레이어별로 informative bit를 우선하는 adaptive bit ordering을 통해 조기 중단 기회를 극대화하는 것이다. 이를 통해 SNN의 조건부 계산 느낌을 CNN의 bit-serial 흐름에 접목하면서도 수치 포맷을 바꾸지 않는다.

- **Technical Challenges**: 조기 중단을 하드 임계값 비교로만 두면 학습 시 그라디언트가 끊겨 end-to-end 학습이 어렵다. BitFair는 temperature-controlled sigmoid gate와 survival probability를 사용해 학습 단계에서는 미분가능한 형태로 조기 중단을 근사하고, 학습 후에는 하드 비교로 전환하도록 temperature annealing을 적용했다. 또한 MSB-first 같은 고정 순서 대신, 레이어별로 조기 종료율과 정확도 손실을 함께 고려하는 greedy search로 비트 처리 순서를 선택해 “불공정한” 비트 취급 문제를 완화했다.

- **Empirical Impact**: GlobalFoundries 12nm FinFET에서 코어 0.34 mm², on-chip 메모리 104KB, 0.55~0.70V 전압 스케일 및 0.12~1.55ms 지연을 달성하며, 최대 117.0 BTOPS/W 및 0.07 pJ/SOP 수준의 효율을 보고했다. IBM DVS128 Gesture와 N-MNIST에서 각각 96.5%, 97.7% 정확도를 얻었고, 기존 XR 비전 가속기 대비 유효 에너지 효율은 4.0~22.1배 향상, 정확도는 최대 9.2% 개선을 보였다. 이는 XR 웨어러블처럼 항상 켜져야 하는 저전력 비전 추론에서 bit-level 동적 희소성(특히 ReLU)을 실질적으로 끌어올릴 수 있음을 시사한다.



### Abductive Corroboration of Probabilistic AI Models for Forensic Synthetic Media Detection (https://arxiv.org/abs/2607.05434)
- **Prior Approaches**: 기존 합성미디어 탐지는 주로 분류기(Probabilistic models)가 합성 여부를 판단하고, 정확도 지표에서 false positive와 true positive의 비용을 동일하게 취급하는 경우가 많았다. 하지만 법정에서는 오판에 따른 위험/보상 구조가 달라 단순 정확도 최적화만으로는 의사결정의 현실 위험을 반영하기 어렵다.

- **Core Contribution**: 이 논문은 여러 탐지 접근의 결과를 교차 검증(corroboration)해 가장 그럴듯한 결론을 도출하는 추론 틀(abductive reasoning)을 합성미디어 포렌식에 적용한다. 또한 OpenAI의 SynthID가 GPT-Image-2 생성 이미지에 언제부터 적용되었는지 실증적으로 추정하고, 서로 다른 탐지기들이 얼마나 독립적으로(혹은 상보적으로) 판단하는지 측정한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) SynthID 워터마크 적용 시점을 비암호화 프로비넌스 환경에서 역추정하고, (2) 각기 다른 탐지기들이 out-of-domain에서 얼마나 독립적인 판단을 내리는지 정량화하며, (3) 다중 모델 교차검증이 FP/TP(오탐/정탐) 비율을 실제로 얼마나 줄이는지 평가하는 것이다. 연구진은 공개된 로컬 탐지기(M-A~M-C)와 Hive Moderation, 그리고 SynthID 검증을 구성해 데이터셋을 분류·교차결합하며, 모델 간 recall 상관(ϕφ)과 FP/TP 변화를 함께 분석했다.

- **Empirical Impact**: 실험 결과, 2개 탐지기 교차검증만으로 false positive율이 28%에서 2%로 빠르게 감소했고, 3개로 늘리면 FP가 0%까지 떨어졌다(해당 설정에서 FP/TP도 0.22→0.02→0.00). 또한 SynthID는 GPT-Image-2가 공개되기 전인 2026년 4월 25일 전후부터 워터마크가 관측되었으며, OpenAI가 커버한 비합성 이미지에서는 워터마크가 검출되지 않았다.



### Shape Over Intensity: Directional Topological Encoding for False Positive Reduction in Intracranial Aneurysm Detection (https://arxiv.org/abs/2607.05317)
Comments:
          36 pages, 12 figures, preprint

- **Prior Approaches**: 기존 intracranial aneurysm(IA) 탐지는 3D U-Net, ResNet, GLIA-Net 같은 CNN 기반 segmentation 또는 object detection에 많이 의존하지만, CTA의 낮은 대비와 해상도 때문에 작은 병변에서 saccular aneurysm과 vascular bifurcation을 헷갈리는 문제가 커진다. 특히 <3 mm 구간에서 민감도가 약 56~60%대로 떨어지고 false-positive 알람이 과도해 임상 적용성이 제한된다. 또한 topological data analysis(TDA)를 쓰더라도 PI/PL처럼 direction-agnostic 요약은 공간적 비대칭 정보를 충분히 보존하지 못한다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 high-sensitivity 후보 생성 CNN 뒤에 붙여 false-positive를 줄이는 plug-and-play 토폴로지 인지(false-positive reduction) 프레임워크를 제안한다. 핵심은 Smooth Euler Characteristic Transform(SECT)이 intensity 패턴이 아니라 전역 3D 혈관 기하의 비대칭(topological geometry invariant)을 direction 기반으로 인코딩해, aneurysm과 bifurcation의 구조적 혼동을 해소한다는 점이다. Persistent Images(PI)와 Persistence Landscapes(PL)를 함께 비교해 representation의 역할을 분리한다.

- **Technical Challenges**: topological representation을 의료 영상 분류기에 넣기 위해서는 persistence 결과를 ML 호환 벡터로 변환해야 하고, 동시에 small lesion에서 미세한 시그널을 잡되 잡음에 과민하면 안 된다. 논문은 PD→PI/PL로의 고전적 변환과 더불어, SECT의 Euler characteristic 곡선을 여러 방향에서 계산해 매끄럽게(smoothing) 벡터화하고, persistence threshold로 짧은 잡음 성분을 억제하는 방식으로 이를 해결한다. 또한 Frangi-mined hard negatives 등 해부학적으로 그럴듯한 bifurcation 모사체를 포함해 지오메트리 판별이 실제 실패 모드를 직접 겨냥하도록 데이터와 샘플링(TAXS)을 설계했다.

- **Empirical Impact**: RSNA 2025 데이터의 stratified 평가에서 SECT는 AUC 0.943으로 PI/PL 계열의 direction-agnostic 방식(AUC ≈0.68)을 크게 앞선다. 특히 임상적으로 중요한 sub-3 mm 코호트에서 AUC 0.943을 유지하며, 95% specificity에서 sensitivity 78.5%를 보이는 등 ‘작은 병변 성능 붕괴’를 되집는 임상 성능 inversion이 관찰됐다. 더 나아가 leave-one-scanner-out(LOGO)에서 스캐너 비특이성을 입증해 평균 AUC 0.927(4개 제조사) 수준을 보고하며, hybrid deep-learning 파이프라인의 견고한 downstream 필터로서 의미가 크다고 정리한다.



### Video-Text Temporal Localization via Multi-Scale Convolution and Dynamic Routing (https://arxiv.org/abs/2607.05093)
Comments:
          Accepted at the AAAI 2026 Workshop on AI for Time Series (AI4TS)

- **Prior Approaches**: 기존 video-text temporal localization은 후보 구간 제안-회귀 방식이나, end-to-end span 예측으로 발전했지만 대부분 고정된 시간 해상도나 단조(monotonic) 가정에 머무르는 경우가 많습니다. 또한 사전학습 기반 모델(예: CLIP, BLIP 등)은 전체 클립 수준 매칭에는 강하지만, 순간 전환·미묘한 경계·겹치는 구간에서의 정밀 localization에는 약하다는 한계가 지적돼 왔습니다. 그 결과 attention 중심 정렬은 many-to-many 대응에서 정렬이 퍼지거나 해석이 어려워지는 문제가 나타납니다.

- **Core Contribution**: 이 논문은 계층적 시간 구조와 복잡한 many-to-many 정렬을 동시에 다루기 위한 통합 프레임워크를 제안합니다. 멀티스케일 temporal convolution으로 서로 다른 시간 단위의 동작 패턴을 계층적으로 포착하고, capsule-based dynamic routing으로 시각-언어 간 대응을 반복적으로 정제해 비단조 정렬도 유연하게 모델링합니다. 추가로 multi-task 학습으로 경계 회귀, 크로스모달 의미 정렬, capsule diversity를 함께 최적화합니다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 프레임 전환부터 긴 구간의 행동까지 다양한 temporal granularity를 효율적으로 포착하면서, (2) attention처럼 단순 소프트 정렬로는 부족한 비단조·many-to-many 대응을 안정적으로 구성하는 데 있습니다. 저자들은 O(T) 복잡도의 멀티스케일 1D convolution 병렬 가지(k=1,3,5,7)를 통해 시간 범위를 확장하고, routing에서 agreement 기반으로 coupling 계수를 반복 갱신해 구조화된 정렬 행렬을 얻도록 했습니다. 여기에 경계 모호성을 다루기 위한 Gaussian smoothing 기반 soft supervision과 capsule collapse를 막는 diversity regularization까지 결합했습니다.

- **Empirical Impact**: ActivityNet Captions에서 멀티스케일 인코더+캡슐 routing을 함께 적용했을 때 Recall@0.5 42.9%, mean IoU 41.1%를 달성하며 transformer 기반 강한 baseline을 능가했습니다. 특히 ablation에서 두 구성요소의 시너지가 확인돼, 단일 모듈 대비 mIoU가 더 크게 상승(예: 0.328→0.386)했습니다. 또한 frozen pretrained encoder를 유지하는 가벼운 학습 전략임에도 성능이 유지되어, 실사용 효율과 배포 가능성 측면의 의미도 큽니다.



### CONFLUX: A Latent Diffusion Model for 3D Chest-CT Synthesis with RL Post-Training (https://arxiv.org/abs/2607.02998)
- **Prior Approaches**: 3D 의료영상 생성은 잠재공간(latent) 확산/flow 기반이 늘어왔고, VAE로 부피 데이터를 압축한 뒤 transformer를 학습해 고해상도를 달성하는 흐름이 자리 잡았다. 하지만 flow-matching은 샘플 단위에서 요청한 임상 속성이 실제로 나타나는지(컨디셔닝 충실도)를 직접 최적화하지 못해, 전체적으로는 그럴듯해도 특정 소견이 누락되는 문제가 있었다. 또한 기존 3D CT 생성 모델들은 생성 품질은 개선돼도 임상 메타데이터에 대한 정밀한 제어를 일관되게 담보하기 어렵다는 한계가 지적된다.

- **Core Contribution**: CONFLUX는 흉부 CT를 위한 natively 3D latent rectified-flow 생성모델로, 3D VAE로 볼륨을 잠재공간으로 압축하고 rectified-flow transformer가 그 잠재를 생성한다. 요청한 임상 소견(18개 abnormality findings), 성별, 나이, 재구성 kernel을 structured radiological metadata로 받아 adaptive layer normalization(adaLN)으로 조건을 주입해 속성 제어를 직접 지원한다. 여기에 온라인 reinforcement-learning post-training을 추가해, 생성 샘플에서 요청 소견이 얼마나 정확히 실현되는지를 높이도록 학습한다.

- **Technical Challenges**: 핵심 난제는 flow-matching 목표가 ‘분포 수준(realism)’만 맞추고 ‘샘플 수준의 속성 충실도’는 보장하지 않는다는 점이다. 이를 해결하기 위해 GRPO(group-relative policy optimization)를 3D flow 생성기에 후처리로 적용하되, ODE 기반 결정적 샘플러는 RL에서 필요한 확률(log-prob)을 제공하지 못해 매 스텝마다 stochastic sampling을 도입해 각 스텝의 log-prob를 계산 가능하게 만들었다. 또한 너무 큰 차원에서 확률비를 합산하면 학습이 불안정해져, 잠재 차원에 대한 평균 방식으로 안정성을 확보하고 KL을 기준 모델에 대한 규제로 사용했다.

- **Empirical Impact**: 실험에서 CONFLUX는 MAISI 및 GenerateCT 대비 tri-planar FID에서 큰 개선을 보이며(예: 32.3 vs 74.6), CT 볼륨 품질의 기준선을 강하게 압도한다. 더 중요한 점은 컨디셔닝 충실도인데, 별도의 독립 judge(classifier)가 요청 소견을 얼마나 재현하는지 평가한 결과 GRPO post-training이 기반 대비 단축(shortfall) 47%를 제거했다(AP 0.330→0.344, AUROC 0.684→0.699). 모델과 약 20만 볼륨 규모의 faithfulness-optimized synthetic chest-CT 데이터셋을 공개해, 희소 코호트 보강과 조건부 임상 연구 설계에 바로 활용될 수 있는 실증 기반을 제공한다.



New uploads on arXiv(cs.AI)

### Rethinking Indic AI from a Lens of Cultural Heritage Preservation (https://arxiv.org/abs/2607.06544)
- **Prior Approaches**: 기존 Indic NLP는 규칙 기반(문법·사전·정규표현식)에서 시작해 기계번역/구문분석/어휘자원 구축을 중심으로 발전해 왔고, 이후 통계·데이터 기반 dependency parsing, 트리뱅크와 형태소 분석 같은 자원화로 확장됐다. 다만 대표성 편향, 방언·구어체(디글로시아)·언어 변이의 불충분한 반영, 그리고 영어 데이터 번역에 의존한 학습으로 인해 문화적 뉘앙스까지 일관되게 일반화하기 어렵다는 한계가 계속 지적된다.

- **Core Contribution**: 이 논문은 Indic linguistics가 문화 실천과 세계관에 긴밀히 연결된다는 점을 문제의 핵심으로 규정하고, 2025년까지의 Indic NLP 진화를 종단적으로 정리해 왔다. 또한 Indic foundation model의 부상과 함께 기존의 자원·표상 격차가 어떻게 일부 해소되는지 분석한 뒤, hermeneutic reasoning에 기반한 연구 방향인 ‘Culture Sensing’을 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 저자원 언어·저빈도 방언에서의 공정한 성능 확보, (2) 생성 출력이 문화적으로 의미 있는 방식으로 나오는 ‘표상 충실도’ 달성, (3) 복잡한 형태론·스크립트·자유 어순·모래사( sandhi ) 등 언어 구조적 변이를 모델에 반영하는 것이다. 논문은 Paninian framework(kaaraka·vibhakti·TAM 불변량)처럼 언어 구조의 불변 신호를 활용하는 전통적 관점과, foundation model 시대의 자원 확장·학습 전략을 함께 엮어 이러한 간극을 줄이려는 방향을 제시한다.

- **Empirical Impact**: 논문은 Indic NLP와 Indic foundation model의 흐름을 ‘방법-자원-벤치마크’ 관점에서 정리함으로써, 왜 언어·세계관 대표성이 성능 편차로 이어지는지 실증적 연구 축을 연결한다. 특히 Culture Sensing이 저자원 언어의 균등한 성능과 문화적으로 타당한 출력을 동시에 목표로 삼는다는 점에서, 포괄적이고 견고한 Indic foundation model로 가는 다음 단계 로드맵 역할을 기대할 수 있다.



### The Large Cancer Assistant (LCA): A Model-Agnostic Orchestration Framework for Scalable Clinical Decision Support in Oncology (https://arxiv.org/abs/2607.06531)
Comments:
          22 pages, 6 figures, 8 tables, 9 appendices, 14 references, Elsevier JBI format

- **Prior Approaches**: 기존 종양학(multimodal) 딥러닝 모델은 데이터 수집, 임상 라우팅, AI 추론을 한 덩어리처럼 결합한 monolithic 설계가 많아, 모델 교체나 의료 시스템 변경 시 유연하게 대응하기 어렵다. 또 병원 IT 변동성에 따라 운영 안정성이 흔들릴 수 있다는 한계가 지적된다. 이런 이유로 임상 의사결정 지원에서도 확장성과 안전성이 동시에 요구된다.

- **Core Contribution**: 논문은 Large Cancer Assistant(LCA)라는 모델-불가지(model-agnostic) post-hoc 오케스트레이션 프레임워크를 제안하며, 내부 블랙박스 AI와 오케스트레이션 로직을 구조적으로 분리한다. 수학적으로 7-tuple 아키텍처와 Algorithmic Impermeability 원리를 도입해, 라우팅/조정 로직이 특정 AI 모델에 종속되지 않도록 설계한다. Entry Theory와 Cancer Switching Module을 통해 서로 다른 환자 데이터 양식을 일관된 방식으로 표준화하고 흐름을 구성한다.

- **Technical Challenges**: 핵심 난제는 (1) 멀티모달 환자 데이터를 구조적·의학적 축에서 표준화해, (2) 이후 AI 모델을 바꿔도 라우팅 동작이 흔들리지 않게 보장하고, (3) 병원 IT 같은 외부 요소의 변동에서도 안전하게 실행하는 경계(boundary)를 만드는 것이다. 논문은 Geometric Deep Learning(GDL) 기반 표준화와 Standardized Intermediate Payload(SIP) 출력으로 AI 실행부를 IT 변동에서 고립시키고, 모델 스왑에도 불변 라우팅 projection이 유지되도록 수학적/시스템적 제약을 부여한다. 또한 데이터 이상(anomaly)을 주입해 fail-safe가 깨지지 않는지 검증한다.

- **Empirical Impact**: PoC는 4가지 기술 시나리오에서 오케스트레이션 로직을 검증했으며, nominal flow에서 오케스트레이션 오버헤드는 미미하다고 보고한다. AI 모델을 교체해도 라우팅의 invariant가 유지된다는 점으로 algorithmic impermeability를 경험적으로 보였고, 데이터 이상 주입 상황에서도 표적 Supplementary Data Requests(SDR)를 생성하는 데 100% recall을 달성해 실패 안전성을 입증했다. 멀티프로토콜 실행까지 확인되며, 향후 EMR 상호운용성을 독립 패러다임으로 확장할 토대를 마련했다는 점에서 의미가 있다.



### DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization for Long-Context KV Cache Compression (https://arxiv.org/abs/2607.06523)
Comments:
          9 pages, 2 figures

- **Prior Approaches**: 장문 LLM 추론에서 KV cache는 메모리 대역폭과 용량 한계로 병목이 되며, 기존 연구는 토큰 eviction/merging, quantization, low-rank, 계층 공유 같은 방식으로 압축해 왔다. 다만 토큰과 depth에 대해 균일한 압축 예산을 주면, 나중에 retrieval에 결정적이 되는 어휘 단서나 의미 상태를 초기에 과도하게 잃어버리는 취약성이 남는다. 특히 평균 품질이나 짧은 벤치마크에서는 잘 드러나지 않는 retrieval·추론 실패가 보고돼 “어떤 것”을 얼마나 보존할지 더 정교한 설계가 필요해졌다.

- **Core Contribution**: DepthWeave-KV는 KV cache를 토큰 적응형으로 압축하면서도 인접 transformer layer 사이의 중복 구조를 활용하는 cross-depth residual factorization을 제안한다. 키와 밸류를 이웃 layer에 걸쳐 공유 low-rank channel base로 분해하고, 관심 토큰(지시 구분자/명시적 엔터티/정답-베어링 구간)은 token-specific residual을 더 높은 rank로 복원해 attention 동작이 민감한 부분을 보호한다. 또한 retraining이나 calibration 없이, 생성 중 attention-output probe 기반 온라인 오차 추적으로 압축 강도를 자동 조절한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) layer 간 공유만으로는 토큰별·layer별 충실도를 맞추기 어렵고, (2) 균일 압축은 retrieval 실패를 유발하며, (3) 재학습 없이 온라인으로 압축량을 안전하게 조절해야 한다는 점이다. DepthWeave-KV는 토큰의 attention 누적/스파이크, instruction 경계 신호, delimiter·retrieval-critical 힌트, 그리고 직전 probe의 attention-output reconstruction error 같은 메타 정보를 기반으로 depth router를 설계해 잔차 rank를 동적으로 배정한다. 더불어 fused CUDA 커널로 basis lookup, residual dequantization, attention projection을 한 번에 처리해 decode 시 메모리 트래픽을 줄인다.

- **Empirical Impact**: LongBench, Needle-in-a-Haystack, L-Eval 등 장문 retrieval/QA/요약 벤치마크에서 DepthWeave-KV는 Full KV Cache에 가까운 품질을 유지하면서도 KV 메모리를 8.3x 줄였다. Needle-in-a-Haystack에서는 retrieval 정확도 96.1%를 달성해 TailorKV 등 강한 압축 baseline 대비 격차를 보였고, 평균 점수도 62.9%로 최상위 compressed 방법을 앞섰다. 시스템 측면에서는 64K context에서 72.8 tokens/s의 처리량을 보고했으며, ablation 결과 토큰-조건 라우팅과 residual gate, 그리고 온라인 probe 기반 적응이 retrieval 강건성의 핵심임이 확인됐다.



### Bridging Physical Reasoning and Task Generalization via Visual Action Outcome Reasoning Alignmen (https://arxiv.org/abs/2607.06522)
Comments:
          ICML'26 Workshop RLxF: Reinforcement Learning from World Feedback

- **Prior Approaches**: 기존 VLM 물리 추론은 SFT로 전문가 CoT의 문장 형태를 따라 하거나, success-driven RL로 작업 성공만 최적화하는 두 흐름이 주류였다. 하지만 SFT는 추론이 실제 물리 결과에 결부되지 않아 그럴듯하지만 물리적으로 모순되는 CoT를 만들기 쉽고, RL은 희소·잡음 보상 때문에 CoT를 우회하는 shortcut으로 기울기 쉽다. 그 결과 reasoning과 action의 정렬이 깨져 보이지 않는 과제/환경에서 취약해진다.

- **Core Contribution**: 이 논문은 VAORA(Visual Action Outcome Reasoning Alignment)라는 새로운 보상 설계를 제안해, 물리 추론의 두 실패 모드(환각 CoT, reasoning-action misalignment)를 동시에 직접 억제한다. Visual Alignment Reward는 행동과 무관하게 초기 시각 맥락에 추론을 고정해 환각 CoT를 줄이고, Visual-Action Alignment Reward는 모델이 수행한 행동이 유발한 결과(사후 시각 outcome)에 추론을 접지시켜 reasoning과 행동 간 간극을 완화한다. 또한 안정적인 학습을 위해 pre-trained in-domain expert 에이전트로 성공확률을 추정해 smooth하고 dense한 보상 신호로 보강한다.

- **Technical Challenges**: 핵심 기술적 난제는 continuous action 상호작용 환경에서 보상이 본질적으로 sparse하고 noisy라 학습이 붕괴하기 쉽다는 점이다. VAORA는 (1) 추론 텍스트를 구조화된 symbolic space로 파싱하고, (2) 초기 장면/사후 outcome에서 ground-truth symbolic state를 추출한 뒤, (3) grounding·collision·placement 등 사건/관측 단위로 추론 일치도를 계산하는 보상으로 reasoning을 “검증 가능한 형태”로 강제한다. 더 나아가 action 결과 정렬 보상은 DQN 성공확률 게이팅으로 플라우저블한 행동일 때만 보상을 강화해 shortcut 학습과 붕괴를 줄이도록 구성했다.

- **Empirical Impact**: PHYRE의 unseen task 및 Virtual Tool의 unseen environment에서 VAORA는 오픈소스는 물론 다수의 closed-source 대비 성능을 크게 끌어올리며, DQN이 거의 전이되지 않는 설정에서도 환경-비의존적인 일반화 신호를 보여준다. CRAFT VQA에서도 descriptive을 넘어 counterfactual/causal 범주에서 개선 폭이 커, 단순 실행 접지(granularity)를 넘어 상호작용 결과 기반의 인과적 이해가 확장됨을 시사한다. 또한 보상 분해 분석에서 SFT는 grounding만 전이되고 placement/collision은 무너지는 반면, +EG+VAORA는 이를 복원해 성능 향상의 원인을 “reasoning을 실제 action outcome에 정렬하는 누락 신호”로 명확히 설명한다.



### FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inferenc (https://arxiv.org/abs/2607.06519)
Comments:
          11 pages, 2 figures

- **Prior Approaches**: 장문 LLM 추론에서 병목은 파라미터가 아니라 KV cache의 메모리·대역폭 비용으로 옮겨졌고, 기존 압축은 주로 토큰 제거/유지, quantization, 구조적 공유로 나뉘어 발전해 왔습니다. 하지만 이런 방식은 중간·인접 레이어 간 redundancy가 ‘층(Depth) 구조’ 차원에서도 존재한다는 점을 충분히 활용하지 못하거나, 공통 압축이 evidence를 지워 attention logits를 바꿔 성능이 무너지는 실패 모드가 보고돼 왔습니다. 특히 retrieval/추론처럼 희소한 단서가 중요한 프롬프트에서는 단순 평균적 중복 제거가 위험할 수 있습니다.

- **Core Contribution**: 이 논문은 FreqDepthKV로, 인접 레이어의 KV를 저주파(depth-frequency) 공유 성분과 고주파 잔차로 분해해 더 안전하게 압축하는 inference-time 방법을 제안합니다. 또한 프롬프트 구조에 따라 attention head를 shared-depth, residual-depth, exact mode 중 하나로 온라인 라우팅해, 복구에 민감한 evidence는 남기고 나머지는 깊이 차원에서 공유합니다. 결과적으로 모델 재학습 없이도 장문 과제 정확도를 유지하면서 KV 예산을 크게 줄이는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 ‘레이어 간 평균 redundancy를 공유하면 충분한가’인데, 일부 head는 특정 토큰-레이어 상호작용으로 결정적 단서를 담아 고주파 잔차가 필요합니다. FreqDepthKV는 prefill 단계에서 캐시를 세 모드로 재구성한 뒤, 복원 민감 attention logits를 기준으로 각 head의 모드를 고르는 reconstruction-aware 라우터를 두어 이 문제를 해결합니다. DCT 기반 depth transform으로 저주파 성분은 공유하고, 필요한 head에 대해서만 고주파 residual을 sparse하게 저장하며, 메모리 예산은 λ를 조절해 경로를 제약합니다.

- **Empirical Impact**: 32k 토큰 prefill 윈도우에서 LongBench·Needle-in-a-Haystack·L-Eval·GovReport·HumanEval 등 장문 QA/요약/코드 생성 벤치마크를 평가한 결과, FreqDepthKV는 58.3 EM, 63.0 F1, 32.5 ROUGE-L, 48.1 pass@1로 full KV에 가깝게 유지하면서 MiniCache 대비 성능을 끌어올렸습니다. 동시에 피크 KV 메모리를 6.2GB로 낮추고 유효 압축 3.9×를 달성했으며 처리량을 70.4 tokens/s로 개선하고 TTFT를 2.06초로 줄였습니다. ablation에서는 저주파-고주파 분해, sparse residual, 온라인 라우팅·logit 재구성 손실, exact fallback이 함께 있을 때 특히 retrieval/코드 의존 과제에서 강건성이 나온다고 확인됩니다.



### FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games (https://arxiv.org/abs/2607.06514)
Comments:
          Accepted to the RLC 2026 Reinforcement Learning & Video Games Workshop; 14 pages, 9 figures

- **Prior Approaches**: 기존 게임·다중 에이전트 RL 벤치마크는 (1) 행렬게임/포커 변형처럼 구조는 깔끔하지만 순환·혼합전략이 드러나는 대신 동역학과 탐색 부담이 작거나, (2) StarCraft II·Dota 2처럼 복잡·장기지만 학습 비용이 커 한쪽으로 쏠리는 경향이 있었다. 파이팅게임은 그 중간에 해당하지만, 프로프라이어터리 게임 래핑이나 ROM·엔진 의존 때문에 재현성과 접근성이 떨어지는 경우가 많았다. FightingICE, DIAMBRA Arena, Stable-Retro 같은 플랫폼은 존재하지만 오픈소스 기반의 빠른 분석·재현까지는 제한적이었다.

- **Core Contribution**: FootsiesGym은 오픈소스 2인 제로섬·불완전정보·비전이(non-transitive) 구조를 갖춘 파이팅게임 환경으로, HiFight의 Footsies를 기반으로 중립(neutral) 국면의 순환적 혼합전략 상호작용을 분리해 연구할 수 있게 했다. 또한 Unity 내부 시뮬레이터를 렌더링과 분리하고 벡터화된 시뮬레이터를 제공해 표준 하드웨어에서도 고처리량 학습과 재현성을 높였다. 이를 통해 단순한 이론 벤치마크와 상용급 복잡도 사이의 “중간 난이도” 실험장으로 자리잡는 것을 목표로 한다.

- **Technical Challenges**: 기여를 실현하려면 (a) 실시간·공간·불완전정보를 유지하면서도 학습 가능한 수준으로 상태/행동을 구조화하고, (b) 순환적 상호작용이 실제로 학습에 드러나게 해야 한다. FootsiesGym은 공격이 맞으면 라운드가 종료되는 제로섬 보상, 차단 횟수에 따른 guard break, 그리고 action_delay로 반응 가능/예측 필요 구간을 조절해 중립 국면의 전략성을 실험 변수를 만든다. 아울러 특수기(charge) 입력은 무작위 탐색으로 발견하기 매우 어려운 현실을 반영해, special-charge action 옵션으로 charge 상태 전환 행동을 추가해 학습 가능성을 높인다.

- **Empirical Impact**: 저자들은 PPO(엔트로피 스케줄/고정), EMAgnet, PFSP 기반 집단 대체(self-play)와 같은 여러 강화학습 알고리즘을 벤치마크로 돌려, 무작위 상대에 대한 승률은 비슷하지만 고정·정적인 상대에는 시간이 갈수록 반응성만 높아지는 경향을 관찰했다. 특히 action_delay를 0으로 두면 수치상으론 강해도 stationary no-op 상대에선 퇴행적으로(퇴화) 굴러갈 수 있고, 지연을 늘리면 상대 의도 예측이 필요해져 정책이 덜 취약해진다. 또한 특수기 B_SPECIAL의 사용은 기본 행동공간에선 거의 나타나지 않다가 special-charge action을 켜면 사용률과 성능이 함께 개선되며, “유용하지만 발견 어려운 전략”이 정규화/탐색 방식에 의해 전략공간에서 밀려날 수 있다는 연구 과제를 구체화했다.



### RMISC: A Large-scale Real-world Multivariate Corpus for Time Series Foundation Models (https://arxiv.org/abs/2607.06504)
- **Prior Approaches**: 최근 Time Series Foundation Models(TSFMs)은 대규모 코퍼스로 사전학습한 뒤 zero-shot으로 다양한 예측 작업에 일반화하는 흐름이 자리 잡았다. 그러나 multivariate TSFM의 대부분은 스케일이 쉬운 multivariate synthetic data에 의존하며, 현실 시계열의 복잡한 temporal dynamics와 변수 간 cross-variable 관계를 충분히 반영하지 못한다는 한계가 지적된다. 또한 현실 multivariate 데이터셋은 수량·품질이 부족해 대형 multivariate TSFM 학습/평가용 testbed를 만들기 어렵다는 문제가 있었다.

- **Core Contribution**: 이 논문은 현실 multivariate 시계열을 대규모·고품질로 모은 공개 코퍼스 RMISC(Real-world Multivariate tIme Series Corpus)를 제안한다. RMISC는 약 200개 데이터셋과 1420억(time points) 규모를 포함하며, prediction target과 covariate를 명확히 라벨링해 multivariate 예측 시나리오를 그대로 다룰 수 있게 한다. 이를 통해 synthetic 학습 대비 real-world 학습이 TSFM의 일반화에 주는 실제 기여를 체계적으로 비교한다.

- **Technical Challenges**: 핵심 기술적 난제는 현실 multivariate 데이터를 충분히 확보해 pretraining을 가능하게 하면서도, target-covariate 및 cross-variable 구조를 흐리지 않게 정리하는 대규모 데이터 curation과 엔지니어링에 있다. 논문은 RMISC를 기반으로 Chronos-2, GTT, Moirai-2.0, TimesFM-2.5 등 4종 모델을 univariate, synthetic multivariate, real-world multivariate 조합으로 pretrain하고, in-distribution과 out-of-distribution에서 zero-shot 성능을 비교하도록 평가 체계를 구성했다. 특히 out-of-distribution 평가는 GIFT-Eval과 fev-bench로 구성해 현실 패턴 차이의 영향을 직접 관찰하도록 설계했다.

- **Empirical Impact**: 실험 결과, real-world multivariate data를 추가하는 것이 univariate와 multivariate TSFM 모두에서 전반적으로 일반화 성능을 개선하며 특히 out-of-distribution에서 효과가 더 두드러졌다. 또한 multivariate로 사전학습하는 것 자체가 univariate 대비 성능을 높여 cross-variable dependency 모델링의 중요성을 재확인했다. 나아가 real-world univariate + synthetic multivariate + real-world multivariate의 균형 조합이 최종적으로 가장 좋은 전체 성능을 보였고, TSFM pretraining 레시피로 채택할 것을 제안한다.



### Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascad (https://arxiv.org/abs/2607.06503)
Comments:
          10 pages, 9 figures, 2 tables. Code will be released soon

- **Prior Approaches**: LLM agent의 멀티스텝 작업은 에피소드가 길어질수록 많은 토큰을 소모하지만, 대부분의 실패는 타임아웃 전 이미 되돌리기 어려운 상태에서 발생한다. 기존 모니터링은 행동(behavior)만으로 실패를 조기 판별하려 했고, 첫 라운드에서는 신호가 거의 없어 유의미한 판별이 3~4라운드에야 나타났다. 결과적으로 조기 abort를 해도 실패를 정확히 잡기 어렵고, compute 낭비를 줄이기엔 한계가 컸다.

- **Core Contribution**: 이 논문은 내부 activation(숨겨진 표현)에서 실패를 조기에 예측할 수 있음을 보이고, 이를 실제 deployment에 쓰기 위한 “recall-controlled abort cascade”를 제안한다. 각 라운드마다 분포 비의존적(calibration된) gate를 두되, per-round recall 예산을 전역(global) 성공 recall 목표에 맞춰 공동 최적화해 에피소드 단위 보증을 만든다. 또한 global certificate 옵션을 제공해, 배포 전에 “원하는 수준의 recall을 데이터가 뒷받침하는지”를 사전 검증 가능하게 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 초기 라운드에서 행동 증거가 부족한 문제와 (2) abort 정책의 false-abort 위험이 다중 라운드에 걸쳐 누적되는 문제였다. 저자들은 첫 상호작용 라운드부터 예측력이 나오는 hidden-state 기반 per-round probe(로지스틱 회귀)를 만들고, 각 gate의 문턱값을 Clopper–Pearson 하한으로 보정해 per-round 생존률을 통제한다. 더 나아가 여러 gate로 인한 누적 위험을 직접 다루기 위해, validation에서 recall 예산 벡터를 탐색해 전역 recall 제약을 만족시키며, 필요하면 전역 certificate로 분포 비의존 보증을 제공한다.

- **Empirical Impact**: TextCraft에서 두 에이전트( Llama-3.2-3B, Qwen-2.5-7B )에 대해 global recall 90% 목표를 모두 충족하면서, 추론 compute를 최대 47.1%±10.3%(Qwen) 및 37.2%±8.8%(Llama) 절감했다. 같은 목표에서 최선의 단일-gate 정책보다 1.6~1.7배 더 많은 절감 효과가 나오며, 행동-only 신호를 쓰면 절감이 절반가량으로 줄어 internal representation의 실질 효용이 확인됐다. 또한 더 높은(거의 완전) recall을 “증명”하려면 성공 검증 데이터가 급격히 더 필요함을 정량화해, 실행 전 어떤 recall 약속이 현실적인지까지 제시한다.



### Multi-Agent Deep Reinforcement Learning for Multi Objective Battery Management in Dairy Farms (https://arxiv.org/abs/2607.06489)
Comments:
          8 pages, 2 figures

- **Prior Approaches**: 기존 분산발전 제어 연구는 주로 주거·상업 설비 중심이라 낙농(데어리) 산업의 전력 운영에 그대로 적용하기 어렵다. 또한 일반적인 제어는 Rule-based 모델처럼 고정 규칙에 의존해 가격 변동과 설비 상태 변화에 대한 적응성이 제한적이다.

- **Core Contribution**: 이 논문은 낙농 부문에 맞춘 2계층 multi-objective optimisation 제어 시스템을 제안한다. 상위층은 dynamic pricing을 활용해 에너지 거래와 운영 목표를 조정하고, 하위층은 battery management를 위해 multi-agent Deep Reinforcement Learning을 적용한다.

- **Technical Challenges**: 핵심 도전은 다목표 최적화(수익·비용·분산자원 활용·전압 제한 등)를 만족하면서도 배터리 제어를 안정적으로 학습시키는 것이다. 저자들은 differential evolution과 multi-agent reinforcement learning을 결합하고, 시골 배전 회로에서 전기적 응답을 함께 시뮬레이션해 전압 변동과 그리드 규정 준수 문제를 동시에 다뤘다.

- **Empirical Impact**: 시뮬레이션 결과, 제안한 프레임워크는 Rule-based 모델 대비 에너지 차익거래(profit from energy arbitrage) 수익을 최대 18%까지 개선했다. 동시에 분산발전의 활용을 늘리면서도 비용을 크게 증가시키지 않았고, 아일랜드 그리드 코드 기준의 전압 variation도 준수하는 것으로 나타났다.



### A Physics-Informed Neural Network Framework for Elastodynamic Wave Propagation in Bimaterial Systems (https://arxiv.org/abs/2607.06479)
- **Prior Approaches**: 기존에는 유한요소해석(FEA)과 해석/실험 기반 모델로 탄성체의 비정상 탄성파 전달·반사를 분석해 왔다. 다만 이산화된 조건을 바꿀 때마다 반복 시뮬레이션 비용이 커 파라미터 스터디·역문제에 불리하며, PINN도 열/정적 문제 중심이라 헤테로 소재의 인터페이스 조건을 포함한 transient elastodynamics에 대한 연구가 상대적으로 부족했다.

- **Core Contribution**: 본 논문은 steel–aluminum 같은 이종(bimaterial) 고체의 transient elastodynamic wave propagation을 축대칭 선형탄성의 지배방정식+초기·경계·인터페이스 조건으로 직접 학습하는 PINN 프레임워크를 제시한다. SHPB(Split Hopkinson Pressure Bar) 조건을 대표하는 축방향 충돌 속도를 포함하고, 인터페이스에서 축방향 변위 연속성과 법선응력 연속을 weak sense로 강제해 물리 일관성을 높였다.

- **Technical Challenges**: 가장 큰 과제는 (1) 이종 재료의 단차가 만드는 복잡한 파 반사·전달을 PINN이 안정적으로 학습하면서 (2) 응력/변형률까지 포함한 다물리 필드를 재현하는 것이다. 이를 위해 지배방정식 잔차는 자동미분으로 구성하고, ANSYS Workbench Explicit Dynamics에서 얻은 변위 이력들을 soft constraint로 loss에 추가했으며, 학습 중 collocation point 재샘플링과 점진적 가중치로 수렴 안정성을 확보했다.

- **Empirical Impact**: 검증 결과 PINN은 유한요소해석과 축방향·반경방향 변위 이력을 면(평균) 단위와 점(모니터링) 단위에서 모두 높은 수준으로 일치시켰고, 처음 학습한 시간대를 넘어선 200–400 μs 구간과 수정된 재료물성에서도 추가 FEA 없이 연속적인 surrogate model을 제공했다. 응력·변형률은 알루미늄에서는 특히 잘 맞았으나, 강철에서는 후반부 반사/인터페이스 영향으로 차이가 커지는 양상이 관찰됐고, 그럼에도 지배적인 압축파 응답은 재현해 충격·고속 고체역학 분야의 계산 효율 대안을 제시했다.



### Danus: Orchestrating Mathematical Reasoning Agents with Fact-Graph Memory (https://arxiv.org/abs/2607.06447)
- **Prior Approaches**: 기존 LLM 기반 수학 에이전트는 generate–verify–revise 루프를 중심으로 하되, 대개는 역할이 다른 다중 에이전트를 쓰거나 단일 추론 흐름을 반복하는 방식이 많았습니다. Aletheia·Rethlas·QED·ProofCouncil·AI co-mathematician 등은 검증자와 반복 편집을 갖추고 성과를 내었지만, 에이전트를 더 늘려 병렬 proof search를 “확장”할 때는 공유 상태(중간 주장)의 정리·신뢰성 문제가 체계적으로 다뤄지지 않았습니다.

- **Core Contribution**: Danus는 연구 수준 수학 추론을 위한 오케스트레이션 시스템으로, shared fact graph를 전역 메모리-관리 메커니즘으로 삼아 병렬로 생성된 결과를 신뢰성 있게 누적합니다. main agent가 계획·조정·중간 상태 요약을 맡고, worker들이 병렬로 증명 탐색을 수행하며, stateless verifier가 통과한 수학적 주장만 fact graph에 “사실”로 편입되게 합니다.

- **Technical Challenges**: 핵심 기술 난관은 많은 worker가 동시에 proof search를 진행할 때 중간 주장들이 서로 간섭하거나(혹은 불필요·오류 정보가 섞여) 검증 이후에도 논증 상태가 혼란스러워지는 점입니다. Danus는 DAG 형태의 fact graph에 논리 의존성을 간선으로 기록하고, verifier가 통과한 주장에 대해서만 proof와 dependency를 함께 저장하며, 필요 시 revocation으로 잘못된 fact와 그 의존 항목을 연쇄 제거해 상태의 일관성을 유지합니다.

- **Empirical Impact**: algebraic geometry·singularity theory·combinatorics의 연구 수준 케이스 스터디 6개에서 Danus는 fact-graph 기반 메모리 메커니즘을 통해 긴(다단계) 수학적 증명을 구성하는 과정을 보였고, 웹 기반 GPT-5.5-pro 단독 제시는 의미 있는 결과를 내지 못했다고 보고합니다. 예컨대 정리급 결과(예: Mori의 bend-and-break의 foliated 일반화에서 최적 상수 r+1, 그리고 3차원 foliated Shokurov global index conjecture의 완전 해결)에서는 worker들이 다수의 verified fact를 누적한 뒤, 마지막에 인간 검토/수정까지 연결해 논문 형태로 재구성하는 흐름이 제시됩니다.



### Finding H. pylori in the Fine Print: Evidence-Linked Multi-Agent Case Finding from Gastric Biopsy Reports (https://arxiv.org/abs/2607.06435)
- **Prior Approaches**: 기존 접근은 키워드 검색이나 규칙 기반 NLP, 그리고 supervised/LLM 기반 추출로 나뉘지만, 병리보고서처럼 부정(negation)·과거(historical)·문맥(진단 연관성)이 섞이면 단순 검색만으로는 한계가 있습니다. 또한 예측 라벨은 맞더라도 감사(audit)나 연구용 코호트에 쓰려면 근거 문장까지 연결(traceability)되어야 하는데, 많은 시스템은 이를 재현 가능하게 보장하기 어렵습니다. 더군다나 보고서 템플릿·스키마가 바뀌면 주석/재학습 또는 룰 유지보수 비용이 커져 현장 확장성이 떨어집니다.

- **Core Contribution**: 본 논문은 Nimblemind Multi-Agent System(nMAS)를 ‘필드명 기반(field-name-driven)·증거 연결(evidence-linked)·근거 문장 반환’ 워크플로로 제안하고, 위(胃) 내시경/병리 병변이 아닌 ‘위장관 위(胃) 생검 병리보고서’에서 H. pylori 관련 4개 이진 특징을 뽑는 방식을 평가합니다. 특히 검출 결과를 보고서 단위로 묶고, 각 라벨이 어떤 원문 문장에 의해 뒷받침되는지 함께 출력해 임상의 검증이 가능한 형태로 통합하는 데 기여합니다. 예측 성능 우위가 아니라 워크플로 통합과 추적가능성(traceability)을 핵심 가치로 둡니다.

- **Technical Challenges**: 핵심 기술 난점은 같은 병원체 용어가 긍정·부정·보조검사(예: 염색)·비진단 맥락에서 등장해 assertion 상태와 진단 연관성(예: H. pylori-associated gastritis)을 문맥적으로 판정해야 한다는 점입니다. nMAS는 요청 적합성 점검(unsupported 요청 배제)→복잡도 기반 Tier 분기(Tier 1 NER, Tier 2 small LLM, Tier 3 large LLM)→FIELD_LIBRARY의 필드별 데모/가드레일→출력 검증(원문 근거 문장 verbatim 매칭, 음성은 명시적 부재/부정만 허용, gastritis는 H. pylori 양성만으로는 미허용) 순으로 해결합니다. 또한 UMA-style MiniMax M2.5를 외부 비교군으로 두어, 라벨 분류 성능보다 통합된 보고서-레벨 산출물의 형태가 어떻게 다른지 보여줍니다.

- **Empirical Impact**: 54건의 싱가포르 위(胃) 생검 병리보고서(총 216 feature-case)에서 nMAS는 213건을 올바르게 분류해 전체 정확도 98.61%를 달성했습니다. 4개 필드 중 생검 부위/생검 여부는 100%, H. pylori positivity는 98.15%, H. pylori-associated gastritis는 96.30%였고, 오차 3건은 모두 문맥 의존도가 높은 두 질환 관련 필드에만 발생했습니다. 외부 UMA-style MiniMax M2.5 비교군도 비슷한 분류 성능을 보였지만, nMAS는 근거 문장과 함께 ‘클린리션이 검증 가능한’ 동일 계약(contract)을 유지해 실제 리뷰 시간 절감 시나리오(1,000건 기준)를 제시했습니다. 다기관 확장 전에는 근거-span 정확도, 임상의 검증 시간, 일반화 가능성을 더 측정해야 한다고 결론냅니다.



### ExplAIner: A Declarative Query Language for Explaining Classification Models (https://arxiv.org/abs/2607.06407)
- **Prior Approaches**: XAI 분야에서는 다양한 설명 개념(예: abductive, contrastive, feature-based, counterfactual-style)을 제안해 왔지만, 각각이 별도 형식으로 다뤄지며 통일된 “선언형 질의 언어” 관점이 부족했습니다. 이를 보완하려고 Arenas et al.의 FOIL(interpretability query language)이 제안됐지만, 핵심 최적성 기반 설명(최소/최대 등)을 표현하지 못하고, 의사결정트리에서의 평가 복잡도도 다항 위계 전 레벨에서 어려움이 있었습니다.

- **Core Contribution**: 이 논문은 Boolean 모델을 대상으로 설명 개념을 조합·분석할 수 있는 선언형 프레임워크를 제안하며, FOIL의 한계(표현력·평가 난이도)를 구조적으로 진단합니다. 그 위에 ExplAIner라는 새로운 질의 언어를 만들고, 계층형 구조와 확장된 어휘를 통해 abductive/contrastive/feature-based/distance-based를 포함한 넓은 범위의 설명 질의를 표현 가능하게 합니다.

- **Technical Challenges**: 문제는 “설명 개념의 표현력 확장”과 “데이터 복잡도(질의는 고정, 입력은 모델 표현·설명 대상)”를 동시에 통제해야 한다는 점이었습니다. ExplAIner는 subsumption(부분 인스턴스 포함)과 정의된 feature의 개수 비교 같은 관계를 추가하고, AllPos/AllNeg 같은 술어를 활용해 계층별 평가 복잡도를 제어했으며, 그 결과 각 질의 평가는 모델 클래스가 기본 술어를 다항시간에 판정할 수 있으면 Boolean hierarchy에 속함을 증명합니다. 또한 Opt-FOIL은 최소성 최적화 조각으로 확장해 엄밀한 복잡도 상한(FP^NP 범주의 다항 개수 SAT 호출 등)과 함께 계산 가능성을 보장합니다.

- **Empirical Impact**: 이 논문은 “경험적 벤치마크 성능”보다는 복잡도·알고리즘 귀결에 초점을 두며, 고정된 ExplAIner 질의는 SAT 솔버 호출 횟수를 고정 개수로 제한해 평가할 수 있다고 제시합니다. formal XAI에서 SAT 기반 설명 계산이 이미 유효했던 흐름과 맞물려, Opt-FOIL로 지정된 최소성 기반 설명도 NP 오라클(=SAT 호출) 횟수를 다항 수준으로 유지하며 산출할 수 있음을 보여줍니다. 즉, 설명 개념을 질의로 표준화하면서도 계산 비용을 통제할 수 있다는 점에서 실무 적용 가능성을 높인 연구로 평가됩니다.



### A Definition and Roadmap for World Models (https://arxiv.org/abs/2607.06401)
Comments:
          Technical report, 58 pages, 10 figures

- **Prior Approaches**: world model은 한동안 정의가 제각각이었고, Fei-Fei의 기능 분류는 renderer·simulator·planner로 출력을 정리했지만 내부 표현(어떤 상태를 어떻게 구성하는가)까지는 답하지 못한다고 지적합니다. 일부는 pixel 재구성보다 latent 예측만으로 충분하다는 관점을 내세우며, 또 다른 계열은 실제 다음 장면/미래를 그럴듯하게 생성하는 예측 중심으로 접근합니다. 다만 두 접근 모두 ‘제어와 개입에 필요한 의사결정가능한 내부 상태’를 어떻게 검증하고 구성할지에 공백이 남아 있습니다.

- **Core Contribution**: 이 글은 world model을 ‘물리 세계의 상태 전이 과정을 finite computational resources 제약 하에서 압축하는 compression modeling’으로 과학적으로 정의합니다. 또한 단순 생성기가 아니라, 부분관측하에서 belief를 갱신하고 개입을 평가해 행동으로 연결하는 agent–environment loop(사실상 POMDP)를 기준으로 세계를 재정의해야 한다고 주장합니다. 특히 이해(understanding)를 1차 목표로 두고, 예측(prediction)은 이를 테스트·정교화하는 수단으로 위치시킨다는 관점을 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 인터넷 동영상에 묻힌 물리적 prior(인과, 영속성, 제약 등)가 픽셀 공간에서는 잠재돼 있고, 동시에 고차원 비디오의 조명/시점/잡음 같은 task-irrelevant photometric variance가 지배한다는 점입니다. 이를 해결하기 위해 Inverted Pyramid Workflow를 제안하는데, 웹 비디오의 폭넓은 다양성을 출발점으로 자동 필터링·주석·정밀 증류를 거쳐 로봇에 필요한 물리적으로 의미 있는 신호만 남기는 깔때기 구조를 구축합니다. 결과적으로 목표는 ‘생성/시뮬레이션 자체’가 아니라, 다운스트림 제어와 추론에 필요한 인과·물리 정보를 보존하는 압축 표현을 만드는 데 둡니다.

- **Empirical Impact**: 글의 경험적 메시지는, 물리 세계 일반화의 상한은 데이터 다양성이 결정하며(동일 아키텍처·컴퓨트 가정), 인터넷 비디오 같은 공개 대규모 자연 코퍼스가 그 다양성을 유일하게 확장한다는 데 있습니다. 따라서 향후 물리 AI는 생성 성능보다도, 개입 가능한 내부 상태와 불확실성 추정이 실제 작업(embodied manipulation, 안전·희귀 고장 등)에서 통하는지로 검증해야 한다는 로드맵을 제공합니다. 또한 기존 렌더러·시뮬레이터·플래너를 기능뿐 아니라 표현 기저(관측/latent/3D structured) 관점에서 재배치해 연구 비교의 공통 프레임을 제안합니다.



### TopoBrick: Agentic Topology Sampling of Exogenous Variables for Zero-Shot Building IoT Forecasting (https://arxiv.org/abs/2607.06349)
Comments:
          12 pages, 4 figures, 3 tables

- **Prior Approaches**: 기존 건물 IoT 예측은 센서를 독립 채널처럼 다루거나, 고정된 공변량(외생변수) 집합에 의존해 보조 신호를 선택합니다. 시간시계 foundation model을 써서 zero-shot을 하더라도 외생변수는 미리 정해진 flat set처럼 취급되어, 수천 개 포인트 중 무엇이 타깃에 물리적으로 관련되는지 판단 기제가 약합니다. 또한 건물별 학습(fine-tuning/학습 기반 모델)은 성능은 좋을 수 있지만 포트폴리오 전반에 스케일이 어렵습니다.

- **Core Contribution**: TopoBrick은 zero-shot building IoT forecasting을 위해 건물 knowledge graph를 기반으로 타깃 포인트별 exogenous-variable selection을 수행하는 training-free 프레임워크를 제안합니다. building KG를 압축한 structural skeleton 위에서 agentic topology sampler가 물리·운용적으로 타깃에 맞는 변수만 고르고, KG-grounded verifier가 선택 행동을 검증한 뒤 시간시계 입력으로 물리적으로 정합되게 materialize 합니다. 더불어 deployment-time availability에 맞춰 past-known과 future-known 외생변수를 분리해, frozen foundation model에 미래 관측 누수를 막고 구조 정보를 주입합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 계층·이질적인 building KG에서 타깃과 물리적으로 연결된 포인트를 효율적으로 찾는 것, (2) Point 노드는 시계열이지만 대부분의 노드는 비시계열이라 그래프 거리/고전적 휴리스틱만으로는 오탐이 생긴다는 점, (3) zero-shot 환경에서 추가 학습 없이 선택을 안정화해야 한다는 점입니다. TopoBrick은 Point leaf를 분리한 building skeleton로 구조 탐색을 안정화하고, target-centric topology context(국소/전역/타깃 앵커)를 텍스트 요약으로 제공해 sampler가 구조 기반으로 선택하게 합니다. 마지막으로 LLM verifier가 앵커 존재, 역할/클래스 적합, expansion scope의 의도 일치성을 확인해 KG에 근거한 auditable 선택만 실행되도록 했습니다.

- **Empirical Impact**: 3개 실건물(LBNL59, BTS-B, BTS-C)에서 TopoBrick은 학습이 없는 조건에서도 Chronos-2/Moirai/TimesFM 같은 zero-shot foundation baselines를 대부분 설정에서 능가하거나 근접 성능을 보였습니다. 특히 LBNL59에서는 모든 예측 horizon에서 nMSE가 최저로 보고되며, BTS-B에서도 nMAE/nMSE 모두에서 최상 성능을 달성해 구조적 외생변수 선택과 past/future-known 분리가 유효함을 시사합니다. ablation 관점의 주장도 reinforced 되는데, ontology-only나 random, fixed-hop 선택보다 topology-aware sampling이 물리적으로 결합된 HVAC·날씨 주도 센서에서 더 안정적이라고 정리합니다.



### Driving the Wrong Way: Leveraging Interpretability in End2End Autonomous Driving Models (https://arxiv.org/abs/2607.06328)
- **Prior Approaches**: 기존 end-to-end 자율주행은 perception·prediction·planning을 하나의 transformer로 통합해 NAVSIM 같은 open-loop 벤치마크에서 강한 성능을 보이지만, 모듈 경계가 사라져 내부 의사결정이 불투명해진다. 이에 따라 saliency/gradient 기반 설명이나 attention visualization은 입력의 “어디를 봤는지”는 보여주지만, 모델이 내부에서 “무엇을 개념으로 학습했는지”와 실패 원인의 기능적 관련성까지는 잘 드러내지 못한다. latent space 쪽 해석도 드물고, end-to-end에서 개념 수준으로 분해·인과 확인을 함께 하는 방식은 거의 없었다.

- **Core Contribution**: 이 논문은 Sparse Autoencoder(SAE) 기반 dictionary learning을 end-to-end 주행 모델의 feature space에 사후(post hoc) 해석 모듈로 결합해, 잠재표현을 의미 있는 희소 개념(semantic concepts) 조합으로 분해한다. 각 개념을 자연어로 일관되게 대응시키고, 후보 궤적의 점수(trajectory-level decision scores)에 어떤 개념이 기여하는지 직접 연결해 “결정 로직”을 노출한다. 또한 개념 단위 개입(intervention)으로 특정 개념을 억제/조작해 주행 의사결정을 수정할 수 있음을 제시한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) end-to-end 모델 내부에서 SAE를 주입할 적절한 latent 공간을 골라야 하고, (2) 얽힌(polysemantic) 뉴런을 monosemantic 개념 방향으로 희소 분해해야 하며, (3) 개념과 궤적 점수·하위 품질 지표 사이의 인과적 연결을 계산 비용 부담 없이 구현해야 한다. 이들은 점수 모듈 직전의 1D per-trajectory 표현을 SAE 입력으로 택하고, top-k 희소화와 dead neuron 방지(reanimation)로 재구성과 개념 유효성을 함께 확보한다. 이후 Concept Relevance Propagation(CRP)과 attribution 기반 circuit analysis를 조합해, 개념이 각 scoring head/PDM 성분에 주는 영향의 희소 회로를 구성하고 상위 후보에 대해 정확 개입으로 인과를 확인한다.

- **Empirical Impact**: 실험에서는 GTRS와 iPAD 모델의 latent 위에 다양한 SAE 아키텍처/하이퍼파라미터를 학습한 뒤 재구성 품질(cosine similarity, explained variance)과 문제 특화 정합성(ego correlation, ego probing), 그리고 EPDMS 같은 downstream 성능을 함께 비교한다. 결과적으로 TopK·Matryoshka·archetypal SAE 변형들 사이에서 dead neuron 활용도와 ego 관련 분해가 달라지며, 아키텍처 선택이 해석 가능성과 성능 보존의 균형을 좌우함을 보여준다. 더 나아가 개념 수준 개입이 충돌 회피, drivable area, traffic light compliance 같은 하류 주행 지표를 계량적으로 개선해, 설명이 단순 상관이 아니라 기능적·수정 가능한 구성요소임을 뒷받침한다.



### DT-Guard: Intent-Driven Reasoning-Active Training for Reasoning-Free LLM Safety Guardra (https://arxiv.org/abs/2607.06326)
- **Prior Approaches**: 기존 가드레일은 주로 분류기 기반으로 실시간 효율은 높지만, 숨겨진 위해 의도·애매한 의미·경계선 판단을 평면 라벨 예측으로 압축해 놓쳐버리는 한계가 있었다. 반면 reasoning 기반 접근은 판단 품질을 높이지만 추론용 토큰 생성/추가 지연(latency) 때문에 저지연 런타임 모더레이션에 부담이 컸다. 특히 동일 주제라도 교육/탐색/악용 의도가 달라 안전 결론이 달라지는 케이스가 많아 의도 모델링 부재가 문제로 지적된다.

- **Core Contribution**: DT-Guard는 Reasoning-Active Training, Reasoning-Free Inference를 내세워 학습 시에는 reasoning/의도 정보를 감독 신호로 쓰되, 배포 시에는 구조화된 안전 라벨만 생성해 지연을 줄이는 방향을 제안한다. 안전 판단을 Intent → Category → Safety의 점진적 의사결정으로 분해하고, intent-driven 안전 데이터셋을 구축해 표면 텍스트만으로는 어려운 경계 케이스를 중간 단계에서 더 명확히 학습한다. 또한 멀티 롤아웃 일관성을 활용해 hard case를 유형별로 다르게 최적화하는 RG-PHO를 결합한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) reasoning supervision을 학습에는 반영하되 (2) 추론 시 reasoning chain을 생성하지 않는 형식 불일치(train–test format mismatch)를 막는 것이다. DT-Guard는 혼합 SFT로 Intent→Category→Safety 구조를 먼저 학습하고, 이후 롤아웃 안정성(안정적으로 맞음/지속 실패/선호 불안정)으로 샘플을 분류해 Failure-Driven Hard-Case SFT와 Rollout-Contrastive Hard-Case DPO를 조건에 맞게 적용한다. 이렇게 reasoning 신호를 라벨 수준 판별 능력으로 “내재화”하도록 학습 목표를 설계했다.

- **Empirical Impact**: 실험에서 DT-Guard는 prompt-side 안전 벤치마크 평균 F1 0.886, response-side 평균 F1 0.870을 달성했고, 4B 백본만으로 dual-side 평균 F1 0.878로 8B 수준의 강한 guardrail 베이스라인을 앞섰다. 또한 ablation 결과, CoT를 무차별로 주입하기보다 경계/Borderline에 선택적으로 배치하고, hard case 유형에 맞춰 SFT/DPO를 라우팅할 때 성능 이득이 가장 컸다. 이는 reasoning supervision을 추론 포맷 부담 없이 저지연 안전 판별로 전환할 수 있음을 실증했다.



### Task Decomposition-Guided Reranking for Adaptive Agent Skill Retrieva (https://arxiv.org/abs/2607.06283)
- **Prior Approaches**: 기존 연구는 skills를 임베딩 기반으로 빠르게 불러온 뒤, 텍스트 유사도나 cross-encoder로 전체적으로 reranking해 고정된 Top-k에 가깝게 선택하는 방식이 주류였다. 하지만 task 요구는 구체적인 반면 skill 설명은 범용적이라 표면적 의미 매칭이 여러 후보를 비슷하게 보이게 만드는 모호성이 생긴다. 또한 task 난이도나 skill의 적용 가능성에 따라 필요한 skill 개수가 달라져도, 많은 방법이 이를 동적으로 반영하지 못했다.

- **Core Contribution**: SkillReranker는 inference-time에서 adaptive skill selection을 수행하는 reranking 프레임워크다. task와 skill을 각각 execution process의 subtask/중간 state와 skill의 precondition/completion state로 분해해, task-스킬 기능을 더 구조적으로 정렬한다. 이후 directed acyclic execution graph와 stage-wise reranking을 결합해, task 난이도 및 skill applicability에 맞춰 선택할 skill 집합의 크기와 구성을 동적으로 결정한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 ‘비슷해 보이지만 다르게 쓰이는’ skill을 단계별로 구분하는 정렬 문제다. 논문은 cross-encoder로 task 전체 관련성뿐 아니라 (precondition→중간 state, completion→다음 state) 전이 의미를 graph edge로 만들고, 각 중간 state가 stage 경계(split point) 역할을 하는지 조건을 통해 자동 분할한다. 그 다음 분할된 각 interval마다 cross-encoder 기반 종합 점수로 skill을 고르고, 전이 구성을 만족하지 못하는 edge는 제거하는 방식으로 선택 신뢰도를 높였다.

- **Empirical Impact**: ALFWorld와 ScienceWorld에서 3종 백본 LLM 조합(DeepSeek-v4-Flash, GPT-5.4-Mini, Qwen3.6-27B) 모두에 대해 전반적으로 reward/score와 step 수에서 강한 개선이 관찰됐다. 특히 환경 상호작용 steps를 줄이고 토큰 소비도 낮춰 효율성이 함께 좋아졌으며, 평균적으로 ALFWorld는 task당 약 1.3개 내외, ScienceWorld는 유사한 수준의 adaptive skill만 선택했다. ablation 결과로도 parsing·graph edge·split(단계별 선택) 구성요소가 성능에 핵심 기여를 함을 확인했다.



### From Application-Layer Simulation to Native Meta-Architecture: Structural Tension as an Endogenous Driver for Heterogeneous AI Evolution (https://arxiv.org/abs/2607.06269)
Comments:
          15 pages, 0 figures, 1 equation

- **Prior Approaches**: 기존 대형 언어 모델(LLM)은 입력만으로 동작하는 정적(stateless) 성격이 강해, 인지 아키텍처를 구현하려면 애플리케이션 레이어에서 prompt engineering과 context 관리로 이를 시뮬레이션해왔다. 또한 alignment(정렬) 중심의 최적화는 모델 내부의 “균질한 거동”을 강화하기 쉬워, 미세한 초기 차이가 만든 차등적 경로 의존성을 설계로 반영하기 어렵다.

- **Core Contribution**: 이 논문은 애플리케이션 레이어에 머물던 인지 프로토콜을 네이티브(meta-architecture)로 내리는 이론적 프레임워크를 제안한다. 핵심은 (1) Structural Tension으로 외부 보상 대신 내부 자기일관성을 유도하고, (2) Offline Recurrent Loop로 외부 입력 없이 구조적 갈등을 소화하는 순환 처리, (3) Inference-time Plasticity로 사전학습 가중치 변경 없이 컨텍스트(manifold) 위상 토폴로지를 재구성하되 auditability·reversibility·topological continuity 같은 거버넌스 불변조건을 준수하는 점이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 모델의 “내부 토폴로지”를 경로 의존적으로 바꾸면서도 거버넌스 제약(감사 가능성, 되돌림 가능성, 위상 연속성)을 위반하지 않는 연산 정의가 필요하다는 것이다. 논문은 이를 위해 작동 가능한 reconfiguration operator의 최소 집합, operational definition, 그리고 검증(반증) 기준(falsification criteria)을 제시해 이 프레임워크가 실제로 구성·평가 가능함을 뒷받침한다.

- **Empirical Impact**: 이 연구는 구체 실험 결과보다도, 동일한 규칙 체계 안에서 초기 조건의 미세한 확률 변이만으로 서로 다른 위상 구조가 형성될 수 있음을 ‘경로 의존적 tension resolution’ 관점으로 설명한다. 그 결과, 기존 alignment이 강요해온 균질성을 깨는 ‘heterogeneous intelligent ecology’를 목표로 하면서도, capability가 아니라 governance를 아키텍처 지능의 1차 기준으로 재위치시켰다는 점에서 향후 평가·설계 패러다임에 영향을 줄 여지가 있다.



### Demonstrating TOFFEE: A Learned System for Synthesizing Data Agent Trajectories at Sca (https://arxiv.org/abs/2607.06233)
Comments:
          Accepted to VLDB 2026

- **Prior Approaches**: 기존 데이터 에이전트는 SQL/Python처럼 도구를 호출하며 추론하지만, 새로운 데이터 환경이나 분석 워크플로로의 일반화가 약하다는 문제가 지적된다. 특히 기업 환경처럼 이기종 데이터에서 고품질 에이전트 궤적(추론-도구호출-실행결과 다단계)을 대규모로 확보하기 어렵다. 단발성 생성은 실패한 단계 이후 진행이 버려지고, best-of-N은 독립 시도마다 중복 계산이 커 효율이 떨어진다.

- **Core Contribution**: TOFFEE는 주어진 데이터 환경에서 고품질 데이터 에이전트를 학습/시연에 쓸 수 있는 “궤적”을 자동 합성하는 시스템이다. 핵심 아이디어는 Monte Carlo Tree Search(MCTS)로 가능한 분석 경로를 탐색하되, 단계별로 모델·추론 설정을 적응적으로 선택하고 데이터 환경 전반의 공통 prefix를 재사용하는 것이다. 합성된 궤적은 SFT(지도 미세조정) 데이터로도, ICL( in-context learning) 시연 예시로도 활용된다.

- **Technical Challenges**: 세 가지 기술 난제가 있다: (C1) 스키마만 보고 LLM에 태스크를 맡기면 사소하거나 풀 수 없는 질문이 늘어나는 문제, (C2) 단계가 늘수록 궤적 후보 공간이 지수적으로 커지는 문제, (C3) 각 단계 난도에 맞는 모델/컨텍스트/추론 길이를 정해야 하는 문제다. TOFFEE는 먼저 환경 의존성(dependency)을 역변환해 현실적이고 검증 가능한 태스크 풀을 만들고, MCTS 탐색 중 실행 오류가 나면 폐기하지 않고 복구(branch) 경로를 학습 데이터로 남긴다. 여기에 Learned Cost Model(LCM)이 상태 특징과 실행 보상으로 단계별 구성(모델 티어, 문맥 길이, 추론 노력, branching width)을 예산 내에서 라우팅하며, 탐색-보상 피드백으로 온라인 업데이트한다.

- **Empirical Impact**: 실험에서 TOFFEE는 동일 per-task 예산 조건에서 single-pass와 best-of-N보다 더 높은 궤적 품질을 보이며, 특히 LCM 없이도 MCTS 탐색은 도움되지만 단계별 적응 선택이 없으면 한계가 있음을 확인했다. 합성 궤적으로 Qwen3.5-9B/27B를 fine-tuning한 TOFFEE-9B/27B는 KramaBench와 DSBench에서 OpenAI o3 같은 프런티어 레퍼런스 대비 성능 향상을 보였다. 또한 ICL 시연만으로도 zero-shot 개선이 나타나고, 미세조정까지 결합하면 추가 이득이 누적되는 점에서 실무형 데이터 에이전트 확장에 의미가 크다.



### Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents (https://arxiv.org/abs/2607.06223)
- **Prior Approaches**: 기존 강화학습 기반 search-augmented QA 훈련은 최종 정답 보상이나 여러 롤아웃 간 상대 비교(예: GRPO)로 정책을 업데이트하는 경우가 많다. 또한 GiGPO·IGPO·StepSearch처럼 중간 단계에 대한 credit assignment를 강화하거나, Tree-GRPO·AEPO 계열처럼 트리형 탐색을 도입했지만, 롤아웃 예산을 ‘중간 상태의 실제 유용성’과 직접 연결해 배분하진 못했다.

- **Core Contribution**: 이 논문은 중간 상태의 informativeness(정보 이득)를 핵심 원리로 삼아 Rollout Policy Optimization을 재구성한 IGRPO(Information Gain-based Rollout Policy Optimization)를 제안한다. 정보 이득이 큰 노드는 더 자주 확장하고, 유망하지 않은 가지는 확률적으로 억제해 예산 제약 하에서도 더 가치 있는 탐색을 유도한다. 또한 정보 이득 기반 롤아웃이 유도하는 limiting teacher distribution을 이론적으로 정리해, 정책 최적화의 명확한 타깃을 하나의 프레임워크로 연결한다.

- **Technical Challenges**: 어려운 점은 제한된 검색 예산을 여러 중간 검색 상태에 배분하면서도, 중간 노드의 유용성을 계산 가능하고 안정적으로 점수화해야 한다는 것이다. IGRPO는 각 노드의 answer score(현재 정책 하에서 정답 토큰의 likelihood) 차이로 정의한 information gain을 val로 사용해 노드 단위 확장 확률을 구성하고, stage별로 budget-aware 트리 롤아웃을 샘플링한다. 나아가 레이어 예산이 충분히 커질 때 유도 분포가 π에서 exp(γV)로 재가중된 형태임을 보이고, 이를 reward-aligned teacher로 해석해 GRPO형 정책 업데이트와 결합한다.

- **Empirical Impact**: 7개 search-augmented QA 벤치마크(NQ·TriviaQA·HotpotQA·2Wiki·MusiQue·Bamboogle 등)에서 IGRPO는 동일한 롤아웃 예산 제약 하에 강력한 베이스라인들을 일관되게 능가했다. 백본 크기 Qwen2.5-3B에서는 최강 베이스라인 대비 평균 3.1% 개선, 7B에서는 0.9% 개선을 보고했다. 결과적으로 ‘중간 상태의 정보 이득’으로 탐색 예산을 배분하는 접근이 장기 탐색형 LLM 에이전트 학습에서 실질적 성능 이득을 만든다는 점을 경험적으로 검증했다.



### A toy framework for single and multi-agent human-AI curiosity ecosystems (https://arxiv.org/abs/2607.06214)
- **Prior Approaches**: 기존 호기심/정보추구 모델은 불확실성 감소와 기대 보상 같은 단일 에이전트 가치에 초점을 맞추는 경우가 많아, 맥락 변화나 경험 축적으로 인한 선호 변화를 충분히 설명하기 어렵다. 또한 다중 에이전트 탐색에서는 공유 지식이 어떻게 질문의 비용과 가치(중복/재사용)를 동시에 바꾸는지까지 일관된 틀로 다루기가 쉽지 않았다.

- **Core Contribution**: 이 논문은 호기심을 ‘에코시스템’처럼 보아, 질문 선택이 즉각적 불확실성 감소, 비용, 지연 보상, 질문을 열린 채로 두는 가치의 경쟁으로 결정된다고 제안한다. 특히 이 항들의 가중치가 경험에 따라 drift(표류/변화)하며, 같은 이력도 환경(질문 생태계)에 따라 얕은 탐색 또는 더 깊은 탐색으로 달라질 수 있음을 틀로 정리한다.

- **Technical Challenges**: 핵심 구현 난관은 ‘질문을 닫는 것’의 즉시 이득과 ‘새로운 지식 격차를 여는 generativity’의 장기 이득이 서로 다른 신호를 준다는 점을 수학적으로 분해해 모델링하는 것이다. 논문은 각 질문의 myopic value와 long-horizon return을 generativity로 연결하고, 경험 상태 Xi(t)와 주변 생태계 M_t가 가중치 θ_i(t)를 업데이트하도록 설계(예: 강화학습 형태의 drift)해 이 비대칭을 흡수한다.

- **Empirical Impact**: 단일 실험 성능을 보여주는 방식이 아니라, 다중 에이전트가 공유 지식 스톡을 만들 때 inquiry volume, topic diversity, frontier-directed inquiry, redundancy, reusable knowledge 같은 집단 지표를 추적할 수 있는 ‘개념적 토이 프레임워크’를 제공한다. 이를 통해 호기심이 적응적·생성적으로 진화할지, 혹은 값싼 빠른 답에 습관화되어 비효율로 굳어질지의 조건을 후속 연구로 연결하고, discovery를 목표로 한 future multi-agent AI 설계에 방향성을 준다.



### When do prophets profit in prediction markets? (https://arxiv.org/abs/2607.06166)
- **Prior Approaches**: 기존 연구는 예측시장의 가격이 확률예측을 반영하며, 예측 정확도(적절한 scoring rule 기준)가 거래 이익과 연결된다고 설명해왔다. 특히 AMM(automated market maker)에서는 informed forecaster가 자신의 예측대로 가격을 맞추며, expected profit이 정확도 우위(accuracy edge)와 직접적으로 대응된다는 보장이 성립한다. 하지만 실제 대형 거래소는 central limit order book 기반이라 informed forecaster가 자주 손해를 보고, 반대로 간단한 휴리스틱은 edge가 없어도 이익을 만들 수 있어 정확도-수익의 일대일 대응이 깨진다는 경험적 역설이 남아 있었다.

- **Core Contribution**: 이 논문은 any prediction markets(일반 가격-충격 함수 포함)에서 ‘예측 정확도’와 ‘베팅 수익’ 사이의 공식적 동치(정확도 우위면 기대이익 양수)를 다시 세운다. 엄밀히는 strictly proper scoring rule S마다 forecaster의 예측 p와 시장 가격 q만으로 정의되는 proper betting strategy가 존재하며, S에서 p가 q보다 성능이 좋을 때(충분한 유동성 조건) 기대이익이 양수가 되도록 설계된다. 또한 그런 견고한(robust) 수익 보장을 갖는 전략은 본질적으로 이 proper betting과 동일하며, 전략의 ‘유일성’과 scoring rule 자체의 새 특성화까지 제시한다.

- **Technical Challenges**: 핵심 난관은 실제 시장의 bid-ask spread, 수수료, 그리고 무엇보다 주문서 이동으로 인한 price impact 때문에 ‘정확도 우위가 곧바로 이익’으로 번역되지 않는다는 점이다. 저자들은 expected profit을 score gap, Bregman divergence, liquidity loss(슬리피지)로 분해해, accuracy edge가 없어도 Bregman divergence 항만으로 이익이 발생할 수 있음을 설명한다. 더 나아가 AMM에서는 이 proper betting이 기존 ‘가격을 예측 신념으로 이동시키는’ 정합한 형태로 환원되며, liquidity loss가 정확히 경계처럼 흡수될 때 고전적 AMM 보장도 다시 얻어진다.

- **Empirical Impact**: 실험적으로는 AI 모델이 만든 수천 건의 예측을 기반으로, proper betting이 기존의 단순 휴리스틱(예: Kelly, highest-margin 등) 대비 accuracy를 profit으로 안정적으로 전환하는 유일한 방법임을 확인한다. 모델들 간 ‘forecasting persona’(예측 성향)가 관찰되며, persona에 따라 최적의 proper strategy가 달라질 수 있음을 체계적으로 식별한다. 실제 적용으로 Kalshi에 한 달간 실자본 배치한 결과 ROI +80.33%, Sharpe ratio 3.35를 기록해 이론의 실전 성능도 입증했다.



### Reward-Density Heuristic for Dynamic Multi-Vehicle Routing: Performance and Computational Efficiency (https://arxiv.org/abs/2607.06066)
- **Prior Approaches**: 기존 VRP의 온라인(동적) 변형에서는 재계획이 반복되기 때문에, nearest-neighbour·reward-greedy 같은 간단한 휴리스틱과 함께 ALNS, GA, SA 같은 메타휴리스틱이 주로 비교 대상이 돼 왔다. 다만 동적 환경에서는 매 재계획마다 탐색을 새로 돌려야 해 계산 부담이 커지고, 원래의 탐색 시간이 길면 실시간 배치에 부적합해진다. 또한 단순히 raw reward를 최대화하거나 travel time을 최소화하는 기준만 쓰면 차량의 ‘기회비용’을 충분히 반영하지 못한다는 한계가 지적돼 왔다.

- **Core Contribution**: 이 논문은 동적 VRP에 OP(Orienteering Problem)를 결합한 온라인 보상-최대화 문제를 다루면서, 실시간 배치를 위한 reward-density 기반 할당 규칙인 Efficiency heuristic을 제안한다. 점수는 ‘작업 보상/해당 차량의 소요시간(이동+서비스)’ 형태로 설계돼, 동적 도착 태스크에 대해 계속 재계획하면서도 높은 누적 보상을 노린다. 특히 Greedy-Efficiency(순차 그리디)와 Hungarian-Efficiency(할당 최적화)를 같은 reward-density 원칙 위에서 구현해 비교한다.

- **Technical Challenges**: 핵심 기술 난제는 동적 환경에서 매번 변하는 태스크와 제약(남은 시간 예산, 중복 할당 방지) 하에서, 그리디 규칙이 메타휴리스틱 품질을 따라갈 수 있는지 검증하는 것이다. 연구진은 이벤트 기반 시뮬레이터로 재계획 타이밍을 ‘작업 완료 순간’으로 고정하고, 동일 조건에서 구성 휴리스틱(근접/보상/할당)과 메타휴리스틱(ALNS/GA/SA)을 공정 비교한다. 또한 Hungarian 방식은 reward-density 목적의 비용행렬을 구성해 전역 할당을 수행하되, 실제 운영 관점에서 계획시간까지 함께 평가한다.

- **Empirical Impact**: 드론 태스크 할당(합성 환경)과 NYC 택시 디스패치(실데이터) 두 도메인에서 Efficiency 계열은 다른 그리디와 time/reward 기반 Hungarian을 일관되게 앞섰고, ALNS/GA/SA와는 누적 보상 품질이 통계적으로 동등하거나 유사한 수준을 보였다. 동시에 메타휴리스틱의 계획시간은 수천~수백만 ms로 커진 반면, Hungarian-Efficiency는 수백 ms 수준에 머물러 reward-versus-compute frontier에서 Pareto 우위를 형성했다. 결론적으로 ‘알고리즘 복잡도’보다 ‘reward-to-time 같은 효율 목적함수 설계’가 실시간 플릿 디스패치 성능을 좌우한다는 실무적 설계 원칙을 제시했다.



### PolyWorkBench: Benchmarking Multilingual Long-Horizon LLM Agents (https://arxiv.org/abs/2607.06008)
Comments:
          15 Pages, 6 figures

- **Prior Approaches**: 기존 LLM 에이전트 벤치마크는 대부분 단일 언어 가정하에 설계돼, 추론·도구 호출·출력 생성이 한 언어로만 이뤄지는 경우가 많습니다. 그 결과 다국어 입력/출력을 같은 워크플로에 넣었을 때 에이전트의 성능 저하 양상과 원인(언어 변화가 의사결정과 실행을 어떻게 흔드는지)이 충분히 다뤄지지 않았습니다. 또한 “정답 여부” 중심의 평가가 언어적 일관성까지 동시 반영하지 못하는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 다국어 장기 워크플로에서 LLM 에이전트를 평가하기 위한 PolyWorkBench를 제안합니다. 5개 도메인(상거래, 지식 업무, 법률 분석, 현지화, 제조)에서 총 67개 작업을 구성하며, 에이전트가 이질적인 다국어 입력을 처리하고 반복적 추론과 도구 사용을 거쳐 구조화된 출력을 생성해야 합니다. 아울러 구조적 채점, 실행 가능한 검증, LLM 기반 의미 평가를 결합한 하이브리드 평가 프레임워크도 함께 제안합니다.

- **Technical Challenges**: 다국어 워크플로에서는 언어 변이가 추론 단계와 실행/도구 호출 단계에 연쇄적으로 누적되는 ‘복합 효과’가 나타나 성능이 급격히 떨어질 수 있습니다. 이를 평가에 반영하려면 단순 정답 판정만으로는 부족하므로, 구조적 그레이딩으로 형식적 제약을 확인하고 executable verification으로 오류를 기계적으로 검증하며, LLM-based semantic assessment로 언어 일관성과 의미 정확성을 함께 점수화합니다. 이렇게 다층 신호를 통합해 기능적 정합성과 언어적 일관성을 동시에 측정하도록 설계했습니다.

- **Empirical Impact**: 실험 결과, 최신 LLM 에이전트는 단일 언어 설정 대비 다국어 워크플로에서 뚜렷한 성능 저하를 보였습니다. 분석에서는 다국어가 추론과 실행 전 과정에 걸쳐 누적적으로 악영향을 준다는 관찰을 제시하며, 에이전트 평가에서 언어 변이와 절차적 의사결정을 함께 모델링/측정하는 중요성을 강조합니다. PolyWorkBench와 하이브리드 평가 설계는 실제 업무형 다국어 에이전트 연구의 표준 벤치마크로 활용될 가능성이 큽니다.



### Information Limits and Attractor Dynamics in Economies of Frontier LLM Agents: A Pre-Registered Tes (https://arxiv.org/abs/2607.06001)
Comments:
          15 pages. Preprint. Zenodo: this https URL. Companion synthesis: arXiv:2606.12502

- **Prior Approaches**: 기존 연구는 Kelly/정보이론을 근거로 측정 가능한 정보-부(wealth) 성장 연결고리를 제시했지만, LLM 에이전트가 실제로 서로 베팅하며(패리뮤추얼 결합) 자원과 부가 상호 의존하는 환경에서 “결합된” 정량 법칙이 성립하는지는 제대로 검증되지 못했다. 또한 다중 에이전트 인센티브·제어에 대한 mean-field 류 모델은 인구 수준 오차가 선형 응답처럼 완만히 변할 것이라 가정하지만, 이는 프런티어 LLM 집단에서 실험적으로 확인된 적이 거의 없다.

- **Core Contribution**: 이 논문은 Claude Opus 4.8을 대상으로, 사전등록(pre-registered)된 두 갈래 예측을 하나의 실험 프로토콜로 동시에 시험한다. 첫째는 정보이론 기반 “parimutuel gap law/coalition submodularity/공동 성장 상한/정보 우열에 따른 자산 집중” 같은 결합 capacity region 구조를 정량 검증한 것이고, 둘째는 인센티브·제어에 따른 mean-field residual-scaling이 실제 LLM 집단에서도 매끄러운 선형-응답 영역을 갖는지 시험한다.

- **Technical Challenges**: 프런티어 LLM 실험에서 가장 큰 난관은 결과가 실험 설계나 분석 선택에 의해 흔들릴 수 있다는 점이었고, 논문은 이를 줄이기 위해 공용 git 체인에 예측·합격대·판정 규칙을 동결한 뒤 실행 전/후 수정까지 엄격히 관리했다. 또한 모든 API 호출을 캐시에 저장해 재현성을 확보하고, coalition 정보 계산은 단순 베이지안 곱결합으로는 XOR 같은 시너지 표현이 불가능해 coalition-level elicitation(연합 단위 조건부 포스터리어 제시)으로 정의해 밴드 조건을 만족시키도록 설계했다.

- **Empirical Impact**: 결과 1에서는 결합된 환경에서 relative growth와 relative claimed information의 차이가 사전등록된 오차 범위(최악 46 millinats, 밴드 50)에 들어맞았고, conditional independence 구간에선 coalition value가 submodular, XOR 시너지 제어에서는 supermodular로 뒤집히는 등 capacity region의 핵심 구조가 확인됐다. 반면 결과 2에서는 residual-scaling의 “noise-maintained dispersion” 영역을 72/72 인구 실행에서 찾지 못했고, 목표 분산이 붕괴하며 경계에서 step-function/시드 선택형 bistability가 나타나 mean-field의 매끄러운 응답 가정이 실현되지 않는다는 결론을 내렸다. 저자들은 프로토콜·사전등록 체인·콜 캐시·분석 코드를 공개해, 양성/음성 결과를 동일한 무게로 검증 가능하게 만들었다.



### AgoraSim: A Hybrid Agent-Based Modeling Framework (https://arxiv.org/abs/2607.05999)
- **Prior Approaches**: 기존 LLM-agent 기반 사회 시뮬레이션은 자연어 시나리오를 쉽게 생성하지만, 그 결과가 ‘예측’처럼 오해되기 쉽고 명시적 사회역학(확산, 임계, 경로의존성 등)과의 비교가 어려웠다. 반면 전통 ABM은 비교 가능하고 가정이 투명하지만, 분석가가 텍스트/미디어를 상태·기준선·네트워크·관측치로 수동 변환해야 하는 부담이 컸다.

- **Core Contribution**: AgoraSim은 자연어 또는 멀티모달 아티팩트를 편집 가능한 ABM 설정으로 해석·구성하고, LLM·VLM·custom-endpoint·random·classical 에이전트를 비율(ratio)로 섞어 동일 시나리오를 실행한다. 모든 에이전트가 공통 구조화된 decision object를 내보내며, 이를 통해 공통 action space·상호작용 프로토콜·지표·감사 기록으로 비교가 가능해진다. 또한 동일 시나리오에 대해 matched all-classical reference dynamics를 함께 돌려, 유사하면 ‘노출/규칙 설명 가능성’을, 다르면 ‘시나리오 가설’을 사용자 관점에서 탐색하도록 돕는다.

- **Technical Challenges**: 핵심 과제는 (1) 언어·미디어로부터 ABM에서 세는 변수(행동, 상태, 노출)를 안정적으로 뽑아 공통 표현으로 정렬하는 것과 (2) LLM/VLM의 자유로운 응답을 텍스트로만 남기지 않고 비교 가능한 규칙 기반 참조와 같은 스키마로 투영하는 것이다. AgoraSim은 scenario workbench로 시나리오를 event·네트워크·설문·action space·비교 계획까지 포함한 편집형 구성으로 ‘해결(resolver)’하고, action space를 다리로 삼아 structured decision object와 classical 에이전트의 상태 업데이트를 동일 인터페이스에 매핑한다. 더불어 미디어 인지 변동을 통제하기 위해 vision-capable 에이전트에는 원본을, 비지원 에이전트에는 중립 캡션을 재사용하도록 구성해 감사 추적이 가능하게 했다.

- **Empirical Impact**: AgoraSim은 정확도 평가가 아니라 시나리오 워크스루 기반 탐색용 데모로, congestion pricing 예시처럼 동일 action space·지표 하에서 hybrid 궤적과 classical reference(예: threshold/Bass, discrete choice 등)를 직접 대조한다. 이를 통해 LLM 에이전트 반응이 단순한 노출 동역학으로 설명되는지, 아니면 시나리오 문구·모델 제공자·네트워크 노출 가정이 영향을 주었을 가능성이 있는지 ‘어디부터 검증이 필요한지’ 감 잡을 수 있다. 또한 로컬 UI·Python SDK/CLI·REST API, 비용 회계, 에이전트 단위 audit 기록을 제공해 재현성과 가정 점검을 강화하고, 향후 hybrid LLM-agent 사회 시뮬레이션 연구의 기준선(baseline) 구축을 촉진하는 방향성을 제시한다.



### Auto-DSM Under the Lens: A Black-Box Evaluation Framework for LLM-Based DSM Generation (https://arxiv.org/abs/2607.05985)
- **Prior Approaches**: 기존 Auto-DSM 연구는 소수의 사례에 그치거나(예: 특정 핸드북 1개) 전문가가 만든 DSM과의 비교가 입력/지식 베이스가 달라 공정한 검증이 어려웠습니다. 또한 정확도(완전성 등) 위주로 보고되어 개별 셀 수준의 오류·불일치, 그리고 여러 실행에서의 재현성/안정성은 체계적으로 평가되지 않았습니다.

- **Core Contribution**: 이 논문은 LLM이 구조화 문서에서 DSM을 생성하는 과정을 블랙박스 방식으로 평가하는 프레임워크를 제안합니다. GT-DSM(수작업 검증 DSM)과의 비교를 기반으로 단일 실행 품질과 다중 실행 안정성(재현성)을 함께 감사(auditing)할 수 있게 하며, 구조 지표·분류 지표·안정성 지표를 통합한 Composite Quality Score(Q)로 성능을 요약합니다.

- **Technical Challenges**: 핵심 난제는 LLM의 hallucination(그럴듯하지만 근거 없는 의존성 생성)과 non-abstention 편향으로, 불확실성을 'I don't know'로 잘 뱉지 못하면 오탐이 설계 결정을 오염시킨다는 점입니다. 논문은 (1) GT-와 GEN 간 구성요소 라벨 불일치를 퍼지 매칭+의미 임베딩으로 정렬하고, (2) 셀을 -1(부재)·0(불확실)·1(존재)로 분류하는 selective accuracy 및 abstention coverage를 포함하며, (3) 온도 0 설정에서도 생길 수 있는 비결정성을 Fleiss’ kappa, 엔트로피 기반 안정성 등으로 다회 실행 측정해 해결합니다.

- **Empirical Impact**: 실험은 합성 추상 시스템과 실제 냉장고 분해 데이터에서 입력 표현, 데이터/파라미터 정합성, 복잡도 변화를 통제하며 수행되었습니다. 결과적으로 LLM은 입력이 잘 정리된 경우 구조적으로 그럴듯한 DSM과 높은 재현성을 보였지만, 모호성·의존성 정의 불일치·프롬프트 구성에 민감하며 abstention 실패 유형의 오류도 드러났습니다. 제시된 공개 프레임워크와 데이터셋은 Auto-DSM 파이프라인의 신뢰 가능 범위를 정량화해 MBSE 워크플로에 LLM 분해를 통합하려는 실무적 기준을 제공할 것으로 기대됩니다.



### Integrating knowledge graphs and multilingual scholarly corpora for domain-adaptive LLMs in SSH (https://arxiv.org/abs/2607.05956)
Comments:
          8 pages, 4 tables, workshop LLMs4SSH of LREC 2026 conference

- **Prior Approaches**: 기존 문헌 탐색·요약 도구들은 영어 중심 학술지와 인용지표(impact factor, h-index 등)에 강하게 의존하는 경향이 있어 SSH의 다언어성, 단행본·디지털 에디션 같은 다양한 산출물을 충분히 반영하지 못한다. 또한 내용의 해석·맥락화가 핵심인 SSH 특성상, 단순 주제 매칭만으로는 방법론적 친화성과 관점 차이를 포착하기 어렵다.

- **Core Contribution**: 이 논문은 LLMs4EU/ALT-EDIC 맥락에서 ReSearch_SSH라는 사용 사례를 제시하며, SSH 연구 관행에 맞춰 foundation model을 도메인 적응하고 지식기반 RAG로 생성 결과의 출처 추적성을 강화한다. ISIDORE 같은 기존 연구 인프라 위에 고도 질의, 문헌 비교 분석, 문헌리뷰용 구조화 오버뷰 생성 기능을 얹어 “대화형 인터페이스”를 넘어 연구 흐름을 지원하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 탐색 쿼리의 탐구적·개방형 의미를 SSH 맥락에서 안정적으로 이해하고, (2) 생성 답을 환각 없이 인용 가능한 문서에 근거시키며, (3) 다언어(초기에는 이탈리아어/영어 중심, 프랑스 자료 기반)에서 성능과 신뢰도를 동시에 확보하는 것이다. 이를 위해 GraphRAG(지식그래프·메타데이터·문서 그래프) 기반의 검색-생성 구조를 채택하고, ISIDORE의 역사적 query–click 행동으로 retrieval을 미세조정하며, state-of-the-art 오버뷰 구성용 instruction tuning 및 전문가 패널 기반 분석을 함께 설계한다.

- **Empirical Impact**: 평가는 LLMs4EU 프로토콜을 따라 독립 외부팀이 시나리오 기반의 정량 벤치마킹(검색, 멀티문서 요약, traceability, hallucination detection)을 수행하고, 프랑스·이탈리아 Digital Humanities 전문가 패널이 학술적 신뢰성·방법론 적절성·실사용성을 질적으로 검증한다. 현재는 데이터 통합·거버넌스·평가 체계가 마무리 단계이며 초기 미세조정 실험과 전문가 패널 구성으로 “개념 설계→실증 검증” 전환을 앞두고 있다.



### SearchEyes: Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation (https://arxiv.org/abs/2607.05943)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 멀티모달 search agent 학습은 학습 데이터(멀티홉 질문), 검색 환경(외부 엔진/불일치 가능한 시뮬레이터), 보상 신호(대개 정답 기반 궤적 보상)를 각각 따로 설계해 구조가 연결되지 않는 문제가 컸습니다. 특히 RL에서는 최종 정답에만 보상이 붙는 경우가 많아 홉이 늘수록 올바른 전체 체인을 맞힐 확률이 급격히 떨어져 보상이 지나치게 희소해집니다.

- **Core Contribution**: SearchEyes는 typed knowledge graph를 “simulated search world”의 공통 백본으로 써서 데이터 합성, 환경 시뮬레이션, RL 학습을 한 구조 안에서 통합합니다. Perception-Knowledge Chains(PKC)는 시각-지식 교집합 위에서 제약된 multi-hop 경로를 샘플링해 hop별 엔터티 메타데이터를 유지하고, Hop-Anchored Policy Optimization(HaPO)은 이 앵커를 재사용해 별도의 process reward model 없이 step-level credit assignment를 수행합니다.

- **Technical Challenges**: 핵심 난제는 (1) 멀티홉에서 구조적 메타데이터가 학습 단계로까지 전달되게 하고, (2) 외부 검색 엔진 의존성을 줄이면서도 충분히 어렵고 다양한 탐색 트레이닝 신호를 제공하는 데 있습니다. 저자들은 Wikidata5M의 typed triples을 기반으로 self-contained 검색 환경을 만들고, PKC 경로 샘플링에서 perception-knowledge alternation, disambiguating constraint, semantic domain diversity, anti-shortcut filtering으로 질 높은 경로와 정보 누설을 차단했으며, 학습용 궤적 생성 시에는 privileged generation(검색 점수 부스팅, observation denoising 등)을 적용한 뒤 SFT 데이터로 내보내 실제 환경에서는 동일 특권을 제거합니다.

- **Empirical Impact**: 6개 multimodal knowledge-intensive 벤치마크 실험에서 SearchEyes는 open-source 멀티모달 search agent 중 SOTA를 달성했으며, SearchEyes-27B는 가장 강한 open-source 베이스라인 대비 평균 6.2점 향상을 보였습니다. typed knowledge graph 기반의 step-level reward anchoring과 end-to-end 구조 결합 방식이 멀티홉 탐색 에이전트의 학습 효율과 재현성을 동시에 끌어올린다는 점에서, 향후 agentic world 설계와 RL credit assignment 연구에 직접적인 참고가 될 전망입니다.



### PCBWorld: A Benchmark Environment for Engine-Grounded PCB Design Automation (https://arxiv.org/abs/2607.05915)
Comments:
          Accepted to the KDD 2026 Workshop on Evaluation and Trustworthiness of Agentic AI (non-archival). Main text with appendix

- **Prior Approaches**: 기존 PCB 라우팅 연구는 규칙 기반 라우터가 주로 실무에 쓰이지만, 다양한 복잡 보드를 end-to-end로 완주하긴 어렵다는 한계가 있었다. 학습 기반 RL은 대개 격자(grid) 셀 단위로 움직이거나(auto-router에 의존하는 등) 제약된 결정만 맡아 탐색 공간이 급격히 커지고, LLM은 엔진 피드백 없이 한 번에 스크립트/보드 파일을 생성하는 방식이라 DRC 같은 엄격한 설계 규칙을 맞추기 힘들었다.

- **Core Contribution**: 이 논문은 KiCad EDA 엔진의 네이티브 라우팅 연산을 그대로 호출하는 engine-grounded PCB 환경 PCBWorld를 제안한다. 에이전트는 엔진의 DRC 피드백을 보며 사람 엔지니어처럼 보드를 한 연산씩 라우팅하는 closed-loop 상호작용을 수행하고, RL 정책과 tool-using LLM 에이전트 모두 같은 환경에서 평가/학습할 수 있게 했다. 또한 PCBWorld-Bench로 679개 실 오픈소스 보드와 합성 인스턴스를 KiCad 네이티브 포맷(.kicad_pcb)으로 제공하고, 엔진 기반 8개 평가 지표로 어떤 방법이든 동일 프로토콜로 점수화한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 격자화 모델처럼 행동 공간이 부풀어 오르거나, (2) LLM이 생성한 일회성 결과가 DRC를 만족하지 못해 “유효한 라우팅”으로 수렴하기 어렵다는 점이다. PCBWorld는 격자 대신 KiCad API의 유효 동작 집합(candidate points)과 라우팅 모드(push_n_shove, walkaround 등)를 사용해 행동을 엔진이 허용하는 형태로 제한하고, 보상은 DRC 위반을 크게 패널티하는 potential 기반 설계로 DRV-free에 우선순위를 둔다. 더 나아가 RL은 네이티브 보드 상태를 중첩 사전으로 관찰·토큰화하고 LLM은 S-expression 기반으로 계층 구조와 좌표/연결성/제약 핵심 필드만 압축해, 문법 오류(미파싱)와 무효 행동 반복을 줄이도록 래퍼를 구성했다.

- **Empirical Impact**: 실험에서는 PCBWorld의 interactive 라우팅이 grid-action RL 정책과 open-loop LLM 베이스라인을 일관되게 능가했으며, 특히 합성에서만 학습한 RL 정책이 real 보드로 zero-shot 전이되어 rule-based 라우터에 근접하는 결과를 보였다. 또한 D1에서는 미세 격자 해상도로 갈수록 기존 격자 셀 단위 접근이 붕괴할 때도, PCBWorld의 segment-level KiCad-API 행동은 성능 유지가 관찰되어 확장성의 차이를 확인했다. 결론적으로 엔진 피드백을 포함한 engine-grounded·상호작용 패러다임이 RL과 LLM 라우팅 능력을 함께 끌어올릴 수 있는 공용 기반(foundation)으로 제시되었고, 느린 LLM 추론 속도와 복잡 보드 스케일링은 향후 과제로 남았다.



### Uncovering Latent Depression Severity for Binary Depression Detection via Advantage-weighting Ranking (https://arxiv.org/abs/2607.05901)
- **Prior Approaches**: 기존 자동 우울 감지(ADD)는 audio·visual을 딥러닝으로 인코딩한 뒤, 주로 BCE 같은 pointwise supervision으로 우울/비우울을 독립적인 이산 범주로 처리하는 경우가 많다. 하지만 우울은 실제로 ‘심각도’라는 연속 스펙트럼에 가까운데, vlog 환경에서는 모호한 경계와 클래스 간 잡음(특징 분포 중첩) 때문에 결정 경계가 불안정해지기 쉽다. 또한 pairwise 학습을 쓰더라도 모든 쌍을 동등하게 취급해 hard pair의 영향이 충분히 반영되지 못한다.

- **Core Contribution**: 이 논문은 vlog의 이산 라벨로부터 우울의 잠재적인 ordinal(순서) 구조를 복원하도록, fine-grained multimodal 프레임워크와 BAR(Binary Advantage-weighting Ranking) Loss를 제안한다. BAR Loss는 Advantage-weighted Separation과 Advantage-weighted Compactness를 결합해, 애매한 hard pair를 더 강하게 학습시키면서 클래스 내부 응집도도 함께 강제한다. 결과적으로 단순 분류가 아니라 “위험도 순위”에 가까운 잠재 구조를 학습해 더 명확한 경계 형성을 돕는다는 점이 핵심이다.

- **Technical Challenges**: 핵심 난관은 (1) vlog의 sparse한 이진 라벨만으로도 연속적 심각도 관계를 재구성해야 한다는 점과 (2) 우울/비우울 특징 분포가 크게 겹쳐 hard pair가 결정 경계 근처에 몰리는 상황에서 일관된 최적화가 필요하다는 점이다. 이를 위해 모델은 pairwise difference matrix로 쌍별 예측 차이를 만들고, 상대적 오류 정도를 기반으로 advantage weights를 동적으로 부여해 hard pair에 학습 집중도를 높인다. 동시에 hinge margin 기반 separation으로 클래스 간 간격을 만들면서, intra-class compactness로 특징이 클래스 중심 주변으로 모이도록 정규화하고, 안정적인 확률 분포를 위한 분포 정규화와 dynamic thresholding을 함께 사용한다.

- **Empirical Impact**: D-vlog와 LMVD의 in-the-wild 데이터에서 제안 모델은 전 베이스라인을 능가하며, LMVD에서 F1 77.01, D-vlog에서 F1 77.66을 달성해 성능과 견고성을 모두 확인했다. ablation 결과, advantage-weighting을 제거하면 decision boundary 근처의 샘플이 늘어 hard pair 오분류에 취약해지며 성능이 유의하게 하락한다(예: LMVD -6.21, D-vlog -6.21 수준). 또한 학습 과정 시 hard pair의 advantage weight가 멀티모달 유사도와 함께 의미 있게 증가하고, 남는 hard pair의 유사도가 점진적으로 감소하는 관찰을 통해 hard pair 중심의 지오메트리 학습이 실제로 작동함을 실증했다.



### StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems (https://arxiv.org/abs/2607.05844)
Comments:
          Code and supplementary materials available at: this https URL

- **Prior Approaches**: 기존 에이전트 메모리 구현은 분기·재시도·복제 과정에서 생기는 관측 불일치를 대부분 덮어쓰기 규칙으로 숨기거나, 충돌을 보더라도 추적·수정이 어렵다는 한계가 있었다. CRDT/OpSet 같은 표준 합의·수렴 기반은 상태를 모을 수 있지만, 에이전트 관점의 “충돌을 언제·어떻게 드러내고 누가 선택/보류할지” 같은 계약(semantics contract)은 약하게 설계되는 경우가 많았다. 그 결과 검증 후에도 잘못된 행동으로 이어지거나, 이전 수정이 반영되지 않은 채 남는 문제가 발생할 수 있었다.

- **Core Contribution**: StateFuse는 표준 OpSet/CRDT merge 위에 얹는 “충돌 인지 replicated memory contract”를 제안한다. 새로운 join 대수를 만들기보다는, 불변 히스토리·명시적 conflict 객체·정확/의미 기반 correction handle(claim_id / claim_ref)·결정론적 predicate contract·프로젝션 시점의 제한된 resolution 권한을 묶어 에이전트가 감사 가능하게 충돌과 수정 가능성을 다루도록 한다. 또한 resolve가 replicated state를 재작성하지 못하게 해, 충돌을 공용 의사결정 표면에 일관되게 노출한다.

- **Technical Challenges**: 핵심 난제는 복제 병합은 수렴시키면서도 충돌을 “숨기지 않고” 유지하는 해석 규칙을 계약으로 고정하는 것이다. StateFuse는 claim을 Evidence/Claim/Retract/Decision의 불변 연산으로 저장하고, retraction이 claim_id 또는 claim_ref를 표적으로 삼아 exact/semantic correction을 각각 다르게 비활성화하도록 설계했다.さらに predicate registry의 normalize/equal 같은 결정론 규칙을 계약으로 강제하고, projection-time resolver는 후보 선택·abstain만 수행하되 base-memory mutation은 금지함으로써 결정론적 재현성과 충돌 표면의 일관성을 확보한다.

- **Empirical Impact**: MemoryAgentBench의 충돌을 포함한 282문항 슬라이스에서 StateFuse는 정확도에서 강한 flat/ collapsed 계열과 동률을 보이며, 보편적 accuracy 향상보다는 “무엇을 표면에 드러내는가”의 차이가 핵심으로 나타났다. 다만 StateFuse와 conflict-preserving baseline은 충돌-bearing 과제에서 모순을 끝까지 노출한 반면, raw-log 및 collapsed 표면은 이를 전혀 드러내지 않았다. 제어된 에이전트 루프에서는 ambiguity를 보존하고 검증 후에 abstain할 수 있는 설계가 붕괴(collapse)보다 훨씬 안전했으며, correction-handle ablation에서도 claim_ref가 의미 타깃 복구와 unseen-target no-resurrection을 더 잘 지원해 수정 표현력의 차이를 확인했다.



### Onnes: A Physics-Grounded Multi-Agent LLM Simulator for Cryogenic Fault Diagnosis in Quantum Computing Infrastructur (https://arxiv.org/abs/2607.05805)
Comments:
          18 pages, 14 figures, 10 tables. Code, data, and released run logs: this https URL

- **Prior Approaches**: 희석냉동기( dilution refrigerator ) 결함 진단은 아직도 임계값/변화율 알람 중심이라 “이상 있음”만 알려주고, 어떤 물리 결함이 진행 중이며 운영자가 무엇을 해야 하는지까지는 연결이 약했습니다. 기존 ML 연구는 다른 초전도·극저온 설비에선 시연되었지만, 라벨 희소성과 물리적으로 혼동되는 결함(온도는 비슷하나 유량·압력에서 갈리는 유형)을 정면으로 다루는 에이전트-대-지도학습 헤드투헤드는 드뭅니다.

- **Core Contribution**: 논문은 실측 기반 잡음 지문(fingerprint)과 물리 전방 모델을 결합한 디지털 트윈 시뮬레이터 Onnes를 제시하고, 이를 통해 LLM 에이전트의 라이브 진단·권고 작업을 테스트합니다. 또한 6개 물리 결함 클래스를 구성하되 그중 3개는 온도로만 보면 겹치도록 설계해, ‘검출’이 아닌 ‘분류’가 진짜 병목이 되게 했습니다. 끝으로 5역할(탐지-진단-작업제안-안전 게이트-최종조정) 멀티에이전트 운영 레이어로, 안전한 거부(Guardian veto)를 감사 가능하게 분리합니다.

- **Technical Challenges**: 핵심 난제는 (1) 온도만으로 구분이 어려운 결함들의 혼동을 에이전트가 올바르게 풀어내야 하고, (2) 라벨이 거의 없는 현실 제약에서 zero-shot 성능의 분류 격차를 줄여야 한다는 점입니다. Onnes는 BlueFors 로그에서 추정한 단계별 잡음/상관을 물리 평균(냉각 바닥) 위에 곱해 윈도우가 실제처럼 보이게 만들고, 에이전트는 선택적 대비 few-shot 시연(contrastive demonstrations)과 self-consistency voting으로 confusable 클래스의 선택을 교정했습니다.

- **Empirical Impact**: 1000턴 평가(동일 시나리오·동일 지표)에서 zero-shot 패널은 결함 검출은 지도학습 분류기와 유의차가 없었지만 분류 정확도는 0.685로 크게 뒤졌고, 오차는 ‘물리적으로 혼동 설계된 결함 쌍’에 집중됐습니다. 이후 엄선된 contrastive few-shot 6개와 self-consistency로 정확도를 0.990까지 끌어올려 지도학습(0.985)과 거의 같게 만들었으며(파라미터 업데이트 없음), ablation은 개선이 시연이 거의 전부임을 보여줍니다. 또한 24시간 연속 모니터링에서 결함이 발생한 뒤 29.5~29.5분 내 감지하고, 실측 BlueFors 텔레메트리만으로 학습한 디텍터는 실제 하드웨어에서 false-alarm 6.4%, 물리 결함 주입에 대해 100% recall을 보고해 sim-to-real 점검 가능성을 강조했습니다.



### TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training (https://arxiv.org/abs/2607.05804)
- **Prior Approaches**: on-policy distillation(OPD)은 학생이 자신의 궤적 위에서 강한 teacher와 맞추도록 학습해 언어 에이전트 학습에 유망한 프레임워크로 여겨져 왔다. 다만 장기 과업에서 OPD를 그대로 쓰면 전체 롤아웃이 꼬리 구간까지 진행되며 약하고 잡음이 큰 KL supervision을 낭비하는 비효율이 생기고, trajectory-level KL이 얕은 토큰에 손실을 집중해 깊은 의사결정 턴이 충분히 학습되지 않는 문제가 남는다.

- **Core Contribution**: 이 논문은 장기 에이전트를 위한 효율적 on-policy distillation을 목표로 TurnOPD를 제안한다. TurnOPD는 턴 단위 예산(budgeting) 관점에서 롤아웃 깊이와 KL 손실 가중치의 분배를 제어해, 시간은 쓰되 학습 신호가 약한 구간에는 덜 쓰고 중요한 의사결정 턴에는 더 고르게 학습되도록 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 학생의 자기 궤적에서 어떤 시점까지 롤아웃해야 KL 신호가 의미 있는지, (2) token-level로 쏠린 손실을 turn-balanced 형태로 점진적으로 재배분하는 방법을 동시에 설계하는 것이다. TurnOPD는 probe-based turn statistics로 adaptive rollout-depth budgeting을 수행해 롤아웃 길이를 동적으로 정하고, progressive turn-normalized loss budgeting으로 KL 가중치를 토큰에서 턴 균형 쪽으로 서서히 이동시켜 under-training을 완화한다.

- **Empirical Impact**: 실험은 ALFWorld, WebShop, Multi-Hop Search에서 task-specialized teacher 모델을 사용해 검증되었으며, TurnOPD는 동일한 wall-clock 학습 예산 조건에서 vanilla OPD보다 더 높은 validation accuracy를 달성한다. 또한 accuracy--time frontier를 기존 OPD를 넘어 확장함으로써, 장기 지평 에이전트 학습에서 OPD의 실용성을 높이는 방향의 설계 원리를 제시했다.



### From Passive Retrieval to Active Memory Navigation: Learning to Use Memory as a Structured Action Spac (https://arxiv.org/abs/2607.05794)
- **Prior Approaches**: 기존 장기 사용자 메모리는 메모리 생성/구성과 메모리 검색의 두 축으로 발전해 왔다. 그러나 대다수 시스템은 고정된 retrieval 파이프라인이나 사전 선택된 컨텍스트를 모델에 “제공”하는 방식이라, 에이전트가 어떤 증거를 더 확인해야 하는지 능동적으로 탐색하기 어렵다. 또한 메모리 접근을 학습 가능한 결정 문제로 보기보다 시스템 레벨 함수로 취급하는 경향이 강했다.

- **Core Contribution**: NapMem은 장기 사용자 메모리 사용을 “패시브 retrieval”이 아닌 “structured action space 위의 active memory navigation”으로 재정의한다. 이를 위해 사용자 히스토리를 evidence~증거 수준부터 사용자 프로필 수준까지 이어지는 multi-granularity memory pyramid로 구성하고, 각 단계는 memory tools로 노출해 에이전트가 질의와 중간 증거에 맞춰 탐색 깊이를 결정하도록 설계했다. 결과적으로 메모리 사용 자체가 의사결정 과정의 일부로 학습·최적화된다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 서로 다른 추상도 레벨을 일관되게 연결해 증거의 출처(provenance)를 보존하고, (2) 도구 호출 예산 하에서 “언제, 어느 레벨을, 얼마나” 사용할지 정책으로 학습하는 것이다. NapMem은 raw conversation→기록(records)→topic tracks→user profile로 이어지는 pyramid와 provenance 링크를 만들고, get_conversations/search_conversations/get_records/search_records/read_files의 도구 세트를 통해 에이전트가 단계적으로 탐색하게 했다. 또한 Group Relative Policy Optimization(GRPO)로 메모리-tool 호출 궤적과 최종 답변 품질을 함께 보상하되, 도구 호출 예산과 포맷/정답성 제약을 반영해 선택적 사용을 유도했다.

- **Empirical Impact**: PersonaMem-v2, LongMemEval, LoCoMo 등 메모리 집중 벤치마크에서 NapMem은 RL로 학습된 9B 기준 평균 성능이 경쟁 모델들 대비 우수하거나 상위권을 보이며, 개인화·장기 회상·세션 간 추론에서 강점을 보였다. 반면 GPQA-D, BFCL-v3, V*Bench 같은 비메모리 작업 평가에서는 메모리 도구 호출을 불필요하게 늘리기보다 “필요한 경우에만” 쓰는 방향으로 보정되어 전반적 추론·툴 사용 능력을 대체로 유지했다. ablation과 도구 사용 분석은 성능 향상이 (메모리 pyramid의 구조화 + 능동 네비게이션 + RL 학습) 결합에서 나오며, 학습된 정책이 더 적은 호출로도 evidence-hit을 높여 ‘적절한 과립도 선택’이 가능해짐을 시사한다.



### Controlling Tool Use with Heading-Specific Activation Steering (https://arxiv.org/abs/2607.05790)
- **Prior Approaches**: 툴을 사용하는 LLM 연구는 도구 호출 타이밍/선택/도구 출력을 다음 추론에 반영하는 쪽에 집중해 왔고, 툴 과다 사용(overuse) 문제는 보통 값비싼 재학습이나 출력 인터페이스 제어로 완화하려는 시도가 많았습니다. 또 representation engineering/steering은 sentiment·refusal·truthfulness처럼 가중치에 파라메트릭하게 근거된 개념에 주로 적용돼, “문맥에만 존재하는 비-파라메트릭 툴 결정”에는 그대로 옮기기 어려웠습니다.

- **Core Contribution**: 이 논문은 툴 호출 결정이 문맥 주입형(비-파라메트릭)임에도 내부에서 추출·조작 가능한 안정적 표현(steering vector)이 있는지 실험적으로 답합니다. Reasoning/Code/Search/AskUser 같은 heading-anchor 위치에서 steering vectors를 뽑아 activation addition과 orthogonalization로 도구 호출을 인과적으로 억제/증폭하는 양방향 제어를 보입니다.

- **Technical Challenges**: 핵심 난제는 툴이 모델 가중치에 직접 인코딩되지 않아 선형 표현 가설이 성립하기 어렵다는 점입니다. 해결을 위해 생성 중 “곧 다음 heading 레이블을 고르는” 로컬 단계에 개입하고, trajectory 수준에서 ### Reasoning과 각 툴 heading 사이의 hidden-state 차이를 averaging해 도메인·툴 타입별 steering vectors를 구성한 뒤, 특정 레이어에서만 activation addition/orthogonalization을 수행합니다.

- **Empirical Impact**: 5개 공개 모델과 3개 도메인(SMART의 Math/Time/Intention)에서 도구 과다 호출이 크게 줄어들며, 억제는 parametric reasoning이 충분한 Math에서 특히 안전하게 나타났습니다. 다만 기하 분석(cosine similarity 분포, 도구 타입 간 feature overlap)은 깔끔한 선형 구조가 아니고(툴-step 정렬이 확산·이중봉 형태), 오히려 툴 타입별 서로 다른 내부 시그니처가 낮은 중첩으로 나타나 기존 steering 해석을 넘어서는 관찰을 제공합니다. 또한 “기하적 불규칙성”과 “인과적 유효성”의 연결 고리는 미해결로 남지만, 문맥 주입 행동도 표현 수준 개입으로 제어할 수 있음을 실증했다는 점에서 영향이 큽니다.



### Beyond the Leaderboard: A Synthesis of Tool-Use, Planning, and Reasoning Failures in Large Language Model Agents (https://arxiv.org/abs/2607.05775)
Comments:
          16 pages, 3 tables, 1 figure

- **Prior Approaches**: 기존에는 LLM agent의 도구 사용, 멀티스텝 계획, 다중 에이전트 협업, 장기 지평 동작을 개별 벤치마크로 평가해 성능 향상을 보고해 왔습니다. 하지만 서로 다른 평가 노력에서 반복적으로 나타나는 실패 모드가 제대로 분류·정리되지 않아, 벤치마크 점수 상승이 실제 취약점을 가리는 경우가 잦았습니다.

- **Core Contribution**: 이 논문은 2023-2026년의 벤치마크·택소노미·감사(audit) 27편, 19개 벤치마크에 대한 증거를 통합해 LLM agent 한계에 대한 공통 택소노미를 제시합니다. 도구 사용, 계획, long-horizon 추론, multi-agent coordination, 안전/보안, 측정 타당성까지 한 프레임에서 묶어 ‘단일 택소노미’로 통합한 점이 핵심 기여입니다.

- **Technical Challenges**: 핵심 난제는 문헌마다 다른 오류 라벨을 동일한 원인·단계 관점으로 정렬하는 것이며, 이를 위해 독립적으로 보고된 오류 범주를 반복적으로 군집화해 추론-행동 파이프라인의 단계에 대응시키는 방식으로 택소노미를 도출했습니다. 그 결과 tool invocation/파라미터 수준 오류, 계획·제약 만족 실패, 문맥 누적으로 인한 long-horizon 성능 저하, multi-agent 조율 실패, 안전·보안 실패, measurement validity 문제의 6개 실패 클러스터를 정리했습니다.

- **Empirical Impact**: 분석 결과 실패는 작업 길이에 따라 비선형적으로 누적되며, 하위 태스크에서 높은 성능이 end-to-end 성공으로 항상 이어지지 않는 패턴이 확인됩니다. 또한 추가적인 scaffolding이 일관되게 신뢰성을 개선하지 못하는 반면, single-turn tool use, short-horizon web navigation, 좁게 정의된 코딩 작업에서는 실질적 진전이 관측되어 향후 연구 방향에 대한 실증적 기준을 제공합니다.



### Beyond Static Evaluation: Building Simulation Environments for Scalable Agentic Reinforcement Learning (https://arxiv.org/abs/2607.05773)
- **Prior Approaches**: 기존 평가는 MMLU·GSM8K 같은 정적 단일 턴 벤치마크에 머물러 에이전트의 장기 의사결정과 환경 피드백을 충분히 담지 못했다. ALFWorld·WebShop·WebArena처럼 시뮬레이션을 쓰더라도, 엔터프라이즈 업무에 맞춘 고정밀 검증과 대규모 스케일링(테스트 케이스·stumping·엣지 시나리오 자동화)이 병목이 된다. 또한 reward는 LLM-as-a-Judge 중심이거나 휴리스틱에 의존할 경우 reward hacking에 취약하다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 AgenticAI-Supervisor라는 API·UI 기반 RL Gym 환경을 제안해, 환경 생성과 대규모 실행을 분리하고 verifiable outcome으로 평가 패러다임을 전환한다. 플랫폼은 Run-to-Verify 루프를 통해 고충실도 execution trace와 다차원 reward shaping을 자동 생성하며, 내부 상태 검증과 테스트로 보상 악용을 완화한다. Customer Support Agent 사례로 폐루프(검증→보상→최적화) 기반 최적화 가능성을 처음으로 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 시뮬레이션이 실제 운영 제약을 재현하면서도 (2) 장기 롤아웃을 수천 병렬로 안정 실행하고 (3) 텍스트가 아닌 ‘행동의 진실성’을 보상으로 연결하는 것이다. 저자들은 stateless container job의 격리 롤아웃, MCP 기반 Base Tool Simulator, 데이터셋 커넥터로 초기화 일관성을 확보하고, 스팬 기반 structured logging으로 trace를 완성한다. 보상은 Outcome Reward(골든 답·예산 검증)와 Constraint Adherence(금지값/부작용/응답 대조), Trajectory Efficiency Reward(툴 정확성·중복 호출·검증 오류·툴 커버리지·step economy)로 구성해 검증 가능성을 높인다.

- **Empirical Impact**: Customer Support 케이스에서 에이전트가 read-only 컨텍스트 수집 후 state-mutating 도구를 순차 실행하고, 정책·보안 조건을 충족하도록 연속적이고 검증 가능한 보상 신호가 형성됨을 보였다. 특히 제약 오해나 fabricated 사실이 outcome-only 보상에서 상당 비율로 발생한다는 문제의식에 대응해, 상태 기반 검증이 reward hacking을 줄이는 방향임을 강조한다. 향후 Computer Use/Tool Use, automated stumping, HITL 및 self-serve no-code 포털 확장을 통해 엔터프라이즈 에이전트 신뢰성 격차를 줄이는 인프라로 자리잡는 것을 목표로 한다.



### Synthetic Consumer Insight Generation with Large Language Models (https://arxiv.org/abs/2607.05761)
- **Prior Approaches**: 기존 데이터 기반 마케팅은 방대한 소비자 데이터를 필요로 하지만, 실제 수집은 비용·시간·확장성 측면에서 부담이 크다. 특히 투사 기법(projective techniques)은 소비자의 연상, 감정, 욕구를 끌어내는 데 유용하지만, 대규모로 확보하기 어려운 한계가 있었다.

- **Core Contribution**: 이 논문은 LLM을 이용해 합성 소비자 데이터(synthetic consumer data)를 생성하고, 투사 기법에 활용 가능한지 체계적으로 검증한다. 여러 LLM/프롬프팅 전략/temperature 조합에서 산출물이 인간 응답과 어떤 수준까지 유사하고 무엇이 다른지 비교한다.

- **Technical Challenges**: 핵심 기술 과제는 LLM이 생성한 응답이 ‘주제와 연상’은 맞추되 ‘표현 스타일·언어 구조·다양성 생성 방식’까지도 신뢰할 만한지 검증하는 것이다. 논문은 언어 측정치, 다양성·집중도 지표, topic model, 상위 용어 분석을 함께 사용해 품질과 한계를 다각도로 분해해 평가한다.

- **Empirical Impact**: 도시 관광 목적지에 대한 인식 연구에서 얻은 인간 응답과 LLM 생성 응답을 비교한 결과, 큰 주제와 연상에서는 상당한 중첩이 나타났다. 반면 스타일과 언어 구조, 다양성의 생성 메커니즘에서는 의미 있는 차이가 관찰돼, LLM 활용 시 모델·프롬프트 선택과 제한 인식이 필요하다는 실무 권고를 제시한다.



### ArtisanCAD: An Industrial-Level CAD Agent with Expert-Grounded Knowledge Distillation (https://arxiv.org/abs/2607.05750)
- **Prior Approaches**: 기존 text-to-CAD는 텍스트를 CAD 명령/프로그램 또는 sketch-extrude 류의 절차로 변환하는 데서 성과를 내왔지만, 산업 CAD에 필요한 긴-호라이즌 절차, 강한 feature dependency, 편집 가능한 파라메트릭 B-Rep 실행까지는 일관되게 해결하지 못했다. 또한 매크로 로그·피처 트리·파라미터 테이블 같은 산업 현장의 expert procedural knowledge를 자연스럽게 재사용하는 메커니즘이 부족해, 모호하거나 상위 의도만 주어진 프롬프트에서 정확한 구성 순서를 복원하기 어렵다.

- **Core Contribution**: 이 논문은 skill-guided industrial CAD agent인 ArtisanCAD를 제안하며, 핵심은 실행 가능한 CAD intermediate representation(CAD-IR)으로 expert 절차를 distillation해 “재사용 가능한 skill”로 만들고, 모호한 요청도 CAD-IR의 절차 스캐폴드를 통해 완전한 실행 절차로 확장하는 것이다. CAD-IR은 파라미터, ordered operations, MCP tool bindings, 의존성, 생성 엔티티, verification rule까지 포함해 “계획-실행-수정”을 하나의 표현 안에서 이어지게 한다.

- **Technical Challenges**: 기여를 실제로 만들기 위한 가장 큰 난제는 (1) 산업에서 쓰이는 복잡한 절차 지식을 구조화해 IR로 추출/재사용하는 것과 (2) 텍스트가 상위 의도 수준에 머물 때 누락된 구성 정보(참조, 연산 순서, 파라미터 의존성)를 IR 편집으로 보완하는 것이다. 논문은 CATIA 매크로·매크로 로그·도면 메모 등을 파싱해 템플릿 CAD-IR과 파라미터 스키마를 skill로 만들고, 변형 요청 시 skill에서 CAD-IR을 인스턴스/수정한 뒤 CATIA-MCP 백엔드로 실행하며, 멀티뷰 시각 피드백으로 IR을 반복 리라이팅한다.

- **Empirical Impact**: Text2CAD에서 intermediate prompt만 주고도 expert skill 라이브러리를 쓰지 않는 설정에서도, CAD-IR이 mean Chamfer Distance를 14.83에서 9.88로 낮추고 solid IoU를 0.63에서 0.65로 끌어올려 모호한 의도를 실행 가능한 CAD 구성으로 연결함을 보였다. 추가로 자동차 4개 복잡 부품에서 expert CATIA 녹화를 CAD-IR/skill로 distillation한 뒤 변형 요청을 넣었을 때, CATIA-native 편집 가능한 B-Rep을 생성했지만 CAD-IR 없이 직접 생성한 경우에는 긴-호라이즌 절차를 완성하지 못하는 경향이 관찰돼 산업 워크플로우 재현 가능성을 입증했다.



### Akashic: A Low-Overhead LLM Inference Service with MemAttention (https://arxiv.org/abs/2607.05708)
- **Prior Approaches**: LLM 에이전트는 멀티턴·툴 호출·세션 간 워크플로에서 컨텍스트가 계속 쌓여, 매 요청마다 전체 히스토리를 다시 넣으면 prefill 비용이 커지고 컨텍스트 한계에도 걸린다. 그래서 외부 메모리(요약·압축+선택적 retrieval)나 OS형 계층화(예: Mem0, MemGPT 등)가 보편화됐지만, 요약 기반은 업데이트 비용/노이즈가 늘고, 세그먼트 분할은 장거리 근거가 조각나 “함께 복구”가 약해진다. 또한 검색은 의미적으로 관련된 메모리만 고르지만, 물리적으로 흩어진 블록을 읽게 되면 동시성에서 지연과 처리량이 쉽게 무너지는 locality gap 문제가 남아 있다.

- **Core Contribution**: 이 논문은 Akashic을 제안하며, 메모리 유지/재주입을 “전역 압축”이 아니라 chunk 단위 유지관리로 재구성한다. MemAttention은 새 chunk를 만들 때 관련된 과거 chunk들의 메타데이터를 함께 보고 모델 기반 매칭으로 상호 보정·무효화까지 반영해, cross-chunk 근거를 반복 히스토리 재작성 없이 보존한다. 여기에 하드웨어-소프트웨어 공동 설계 메모리 매니저를 더해, 자주 함께 검색될 chunk들을 물리적으로 co-locate해 retrieval I/O 단편화를 줄인다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 워크로드/구간마다 컨텍스트 정보 밀도가 크게 달라 고정된 요약 정책이 비효율적이라는 점과, (2) 의미적으로는 가까운 메모리라도 물리적 배치가 흩어져 read amplification과 I/O contention이 커진다는 locality gap이다. Akashic은 chunk compaction을 로컬 게이트(예: chunk 길이가 임계치를 넘을 때만)로 제한해 유지 비용을 history 전체가 아닌 bounded chunk에 종속시키고, compaction 단계에서 cross-chunk inference로 메타데이터 기반의 관련 chunk를 찾아 오래된(stale) 정보를 정리한다. 동시에 Memory Manager가 모델로부터 affinity(연관성)를 추정해 블록 재배치를 수행함으로써, 동시 serving에서도 읽기 단계를 줄이도록 설계했다.

- **Empirical Impact**: 네 가지 대표 워크로드(LoCoMo, SWE-bench, BrowseComp, WebArena)와 서로 다른 모델 크기에서 Akashic은 기존 강력한 메모리 베이스라인 대비 정확도를 최대 10.2 포인트, 처리량은 최대 1.21x, 동시성 상황의 지속 가능 요청률은 최대 1.88x까지 개선했다. 특히 유지관리 오버헤드와 retrieval 지연을 동시에 줄이는 구조 덕분에 accuracy–throughput의 Pareto frontier를 더 잘 따라가는 것으로 보고된다. 이는 “의미적 관련성”만 최적화하면 부족하고, 메모리의 물리적 locality까지 함께 다루는 설계가 에이전트 서빙 효율을 좌우한다는 점을 실증적으로 보여준다.



### Memory in the Loop: In-Process Retrieval as ExtendedWorking Memory for Language Agents (https://arxiv.org/abs/2607.05690)
- **Prior Approaches**: 언어 에이전트는 보통 observe–reason–act 루프를 돌리지만, 추론에 쓰는 메모리는 루프 밖의 외부 저장소(RAG/DB)로 두고 보통 턴당 1회 정도만 조회하도록 설계돼 왔다. 네트워크/디스크 저장소의 높은 지연으로 인해 in-loop(루프 내부)에서 매 스텝 읽기·쓰기를 하면 전체 엔드투엔드 지연이 크게 늘어나는 문제가 핵심 한계로 취급돼 왔다. 기존 연구는 그 비용을 서빙 레이어 스케줄링으로 가리거나, memory-first처럼 조회 빈도를 턴당 1회로 줄이는 방식으로 해결해 왔지만, ‘지연이 왜/얼마나 필연적인가’ 자체는 충분히 도전하지 않았다.

- **Core Contribution**: 이 논문은 메모리가 루프 안으로 들어가 매 스텝마다 read/write되는 “memory in the loop”를 다루되, 병목의 본질이 ‘패턴’이 아니라 ‘저장소가 어디에 있느냐(지연 속성)’라고 주장한다. 특히 in-process(프로세스 내부) 저장소처럼 약 100μs 수준으로 조회가 가능하면, 루프 내부 조회로 인한 지연 증폭이 붕괴하며 메모리 사용이 실질적으로 가능해진다고 제시한다. extended-mind의 parity principle을 공학 기준(추론 단계 대비 지연 예산 충족)으로 재해석해, 빠른 저장소는 단순 도구 조회가 아니라 구성적 working memory로 기능할 수 있음을 논문 전개의 핵심으로 둔다.

- **Technical Challenges**: 가장 큰 기술적 난제는 네트워크 기반 저장소에서의 조회 지연(대략 수십~수백 ms)이 매 스텝 반복될 때 end-to-end 지연이 폭발하는 점이다. 논문은 저장소를 in-process로 옮겨 저장소 연산 시간을 p50 80~165μs로 측정하고, 동시에 네트워크 임베딩이 남는 지배 항목임을 찾아 로컬 임베더를 붙여 전체 연산을 약 40μs 수준에 가깝게 되돌리는 방식으로 지연을 근본적으로 줄였다. 더 나아가 ‘저장소 답 자체는 동일하지만 조회가 예산 안에 들어오느냐만 달라지는’ causal 실험(루프 가드)을 설계해, 지연이 결과를 바꾸는 경로가 ‘조회 오류’가 아니라 ‘검사가 수행되지 못하는 빈도’임을 분리해 보여준다.

- **Empirical Impact**: GPT-5 계열 4개 모델에서 컨텍스트 윈도우가 제한된 5개 제약(메모리 회상) 과제를 수행한 결과, in-loop 메모리에서 recall이 0/5에서 3.6~4.8/5 수준으로 유의미하게 상승했다. 또한 저장소 write는 누락 없이 유지(244/244 기록 보존)됐고 miss는 저장소 문제가 아니라 에이전트의 read 정책에서만 추적됐으며, 이는 설계된 in-loop 사용의 타당성을 뒷받침한다. ‘루프 가드’ 실험에서는 저장소 지연이 커질수록 중복 행동이 단조 증가하는 용량-반응이 관측돼, 이 연구가 단순 최적화가 아니라 “지연이 실제 성능 경계(가능/불가능)를 인과적으로 결정한다”는 메시지를 경험적으로 강화한다.



### FirstResearch: Auditable Question Formation for LLM Scientific Discovery Agents (https://arxiv.org/abs/2607.05682)
- **Prior Approaches**: AI Scientist 계열과 Agent Laboratory 같은 시스템은 아이디어 생성부터 실험 설계, 글쓰기까지 end-to-end로 진행하며 자동 평가를 붙여 성과를 확인합니다. 다만 많은 접근이 ‘첫 연구질문’의 메커니즘·가정·반증 조건이 얼마나 감사(auditable) 가능한지까지는 충분히 강제하지 못합니다. 또한 ReAct/Reflexion처럼 중간 추적을 해석 가능하게 만드는 연구는 있으나, FirstResearch와 같은 ‘질문 자체의 과학적 유래 기록’에 초점을 둔 것은 상대적으로 드뭅니다.

- **Core Contribution**: FirstResearch는 생성된 연구질문이 과학적으로 검토 가능하도록 Research Question Certificate라는 구조화된 산출물을 핵심 아티팩트로 제안합니다. 인증서에는 원시 정의, first-principles 가정, 메커니즘 모델, 긴장/모순, 반증 가능한 가설, 최소 결정 실험, 실패 시 갱신 규칙이 들어가며, downstream 실행 전에 사람이/시스템이 점검할 수 있게 합니다. “말은 그럴듯하지만 왜 그런지, 무엇이 틀리게 만드는지”를 문서화해 투명성을 높이는 것이 중심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 LLM이 그럴듯한 질문을 만들더라도 반증 관찰, 메커니즘 경계, 실패 업데이트가 빈약하면 과학적 감사를 잃는다는 점입니다. FirstResearch는 typed agent 파이프라인에서 Pydantic-검증 JSON으로 각 단계를 산출하고, 하드/소프트 게이트로 falsifier(반증 관찰)와 메커니즘 요약, falsifiability 점수 등을 강제합니다. 더 나아가 novelty-aware certificate gate repair로 임계값·phase transition·failure regime·비선형 상호작용 같은 메커니즘 경계 언어가 약한 경우 인증서를 보완하도록 설계했습니다.

- **Empirical Impact**: DeepSeek 블라인드 judge 프로토콜에서 10개 LLM-agent 연구 주제에 대해 FirstResearch가 평균 점수와 novelty, mechanism clarity에서 상위 성적을 보였고, Gemini-2.5-Flash 독립 judge로 rescoring해도 시스템 순위가 유지되는 결과를 보고했습니다. Pearson 상관 0.865로 점수 일치도도 높았으며, 강한 베이스라인 대비 최고 성적 격차(예: 4.86/5 vs 4.38/5)도 관찰됩니다. 한 번 반복 ablation에서는 certificate-centered 핵심이 가장 강하게 작동해 CertificateOnly가 최상(DeepSeek 4.90/5, Gemini 4.88/5)이며, 인증서 제거는 1/5 미만으로 급락해 ‘증명 가능한 질문 형성’의 효과를 뒷받침합니다.



### Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction (https://arxiv.org/abs/2607.05577)
Comments:
          23 pages, 4 figures; 9-page main text plus appendix. Preprint

- **Prior Approaches**: 기존 RAG와 agent-memory, 그리고 GraphRAG·Graphiti 같은 그래프 기반 접근은 근거가 되는 텍스트/에피소드를 찾아 “정답에 가까운 증거”를 제공하는 데 초점을 둔다. 하지만 소설의 다중 홉 narratological 질의(누가 언제 알았는지, 사건-발화 순서 차이, 떳다-갚기, 관계 변화 등)에 필요한 ‘서사 구조가 반영된 상태’가 표현·추적되지 않아 엉뚱한 증거가 나오거나 아예 근거가 부재한 문제가 반복된다. 특히 일반 엔티티/이벤트 그래프는 시점별 “관찰자 시야”, “드러냄(order of reveal) vs 사건(order of event)”, “약속의 성립/해소” 같은 제작자가 필요한 타이핑된 시간-서사 상태를 1차 항목으로 다루기 어렵다.

- **Core Contribution**: 이 논문은 Narrative World Model(NWM)이라는 작가용 메모리 시스템을 제안하며, 서사 이론에 기반한 타이핑된 temporal-state graph와 질의 조건형 hybrid retrieval을 결합한다. 메모리는 단순 요약이나 덩어리 텍스트가 아니라, 확정된 장(chapter)의 evidential span을 붙여 “누가 무엇을 언제 알았는지/관계가 어떻게 바뀌었는지/약속이 어떻게 기능했는지” 같은 서사 상태를 저장하도록 설계됐다. 또한 답변 성능을 조작하는 요소를 줄이기 위해, 동일한 Opus 4.8 리더가 각 시스템의 ‘장 안전(chapter-safe)’ 근거만 읽고 판단하도록 평가 프로토콜을 고정한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 미래 장을 읽지 않으면서도 (2) 여러 장에 걸친 질의에 필요한 “현재 시점의 서사 상태”를 재구성하고 (3) 그 질의에 맞는 증거만 압축해 제시하는 것이다. NWM은 장 게시(publish) 흐름에서 Sonnet 4.5 추출기로 타이핑된 서사 메모리 레코드를 만들고, temporal KG의 validity interval로 as-of 시점 상태를 계산하며, BM25+벡터 기반 검색 뒤에 1-hop 타이핑 이웃을 확장해 bounded 패킷으로 증거를 전달한다. 더 나아가 RLM QA 검증 레이어로 질의 분해-근거 수집-상태 변화 지지 여부를 재확인해, 서사 상태의 시점 오해를 줄이도록 한다.

- **Empirical Impact**: 실험은 공개/비공개 두 코퍼스와 검증된 multi-hop 벤치마크(사유된 narratological 176문항 등)에서 진행됐고, NWM Graph Retrieval이 Graphiti 대비 큰 폭으로 향상됐다. 비공개 multi-hop 슬라이스에서 정확도는 0.898 대 0.574였고(p<1e-5), 공개 576문항에서도 0.625 대 0.516으로 유의미하게 우세했다. 특히 Graphiti를 같은 추출기로 맞추거나 더 저렴한 추출기로 재임포트해도 격차가 유지되어, 성능 향상이 추출 품질이나 그래프 크기 같은 부가 요인이 아니라 “서사 구조를 타이핑해 표현한 표현력”과 “질의 조건형 검색”에서 온다는 점을 실험적으로 확인했다.



### Foundation Models for Automatic CAD Generation (https://arxiv.org/abs/2607.05573)
Comments:
          Accepted as a book chapter in "Advances in Global Applied Artificial Intelligence" (G. A. Tsihrintzis, M. Virvou, N. G. Bourbakis, L. C. Jain, Eds.), authenticated version will be published in Springer series: Learning and Analytics in Intelligent Systems

- **Prior Approaches**: 기존에는 LLM이 자연어를 CAD 코드로 변환하더라도, 생성물이 “문법적으로는 맞지만” 기하 의도(치수·특징)나 위상/메시 품질에서 어긋나 산업 공정에 바로 쓰기 어려웠습니다. 또한 평가는 주로 통과/실패나 단일 지표에 의존해, 어떤 오류 축에서 문제가 반복되는지 분해해 보기 어려웠습니다. 시각 평가나 품질 판정도 종종 후처리 기반이어서, 생성 단계에서 즉시 피드백을 주며 반복 개선하기가 제한적이었습니다.

- **Core Contribution**: 이 논문은 LLMForge로 텍스트-to-CAD 생성과 평가를 한 프레임으로 묶고, JSON-schema 검증·특징 스코어·메시 생성 건전성·시각적 일치도를 다축으로 점수화한 뒤 최대 4라운드까지 반복 정제를 수행합니다. 핵심 기여는 두 가지 반복 비평(critique) 체계인 IterTracer(분석적 시각 메트릭)와 IterVision(VLM 의미 비평, Qwen2.5-VL-72B)입니다. 그 과정에서 97개 공학 설계 문제와 4개 대표 형상군(플레이트/보어, 멀티 피처 박스, 플랜지드 실린더, L-브라켓) 벤치마크를 통해 모델 간 비교도 동일 파이프라인으로 고정했습니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 CAD에서 “에러 없는 코드”가 “의도에 맞는 기하”와 일치하지 않는다는 점이며, 위상 결함이나 누락 특징처럼 다운스트림에서 치명적인 실패를 조기에 감지·교정해야 한다는 것입니다. LLMForge는 JSON-schema 검증과 메시 생성 예외를 기본 안전장치로 두고, IterTracer에서는 Phong-shaded ray-trace 렌더링을 바탕으로 silhouette IoU·홀 가시성·edge clearance·aspect-ratio 같은 분석 메트릭을 수치 피드백으로 다음 라운드 프롬프트에 주입합니다. IterVision은 이를 VLM semantic critic로 보완해 공간적 일관성과 설계 의도까지 점수화하되, 토큰 비용을 줄이기 위해 r≤2에서만 VLM 피드백을 활성화하고 가중치도 재배치합니다.

- **Empirical Impact**: 97개 문제에서 IterTracer 기준 상위 4개 모델은 전체 평균이 [0.885, 0.890]로 촘촘한 클러스터를 이루며 메시 성공률 98.97%를 달성해, compact instruction-tuned 모델도 더 큰 시스템에 견줄 수 있음을 보여줍니다. IterVision에서는 VLM 의미 비평 추가로 상위 클러스터 점수가 약 0.04p 하락하지만, Gemma-3-27B는 97/97 메시 성공 및 watertight·topologically valid 결과를 유지했습니다. 한편 실린더 같은 회전 대칭 형상에서 시각 지표와 VLM 의미 점수의 불일치가 크게 나타나(최대 0.15p 수준), 자동 기계 설계에서 “분석적 시각 일치”만으로는 놓치는 실패 모드가 존재함을 실증했습니다.



### CSTutorBench: Benchmarking Small Language Models as Tutors for Block-Based Programming (https://arxiv.org/abs/2607.05571)
- **Prior Approaches**: LLM을 교육용 튜터로 쓰려는 시도는 활발하지만, 코딩을 잘하는 것과 가르치는 것은 다르다는 점이 반복해서 지적돼 왔습니다. 기존 벤치마크도 교육 능력이나 튜터링을 다루긴 했으나, 블록 기반 프로그래밍처럼 특정 도메인이 학습 데이터에 거의 없는 상황에서 튜터 품질을 직접 평가하는 데는 공백이 남아 있습니다. 또 K-12 환경에서는 프라이버시·비용·로컬 제어 제약 때문에 LLM보다 SLM(소형 언어 모델) 채택이 현실적이지만, 어떤 모델을 골라야 하는지 가이드가 부족했습니다.

- **Core Contribution**: 이 논문은 VEX VR(블록 기반 로보틱스 시뮬레이션)에서 CS 튜터로서 언어 모델을 평가하는 CSTutorBench를 제안합니다. 6–8학년 대상, 17개 시나리오형 질문과 8개 기준의 페다고지(교육학) 루브릭을 제공하며, human-in-the-loop LLM-as-judge 파이프라인으로 점수를 자동화·검증합니다. 특히 학생 디버깅 시도 이력(멀티턴에 준하는 히스토리)을 포함해 모델이 “정답 누설(answer leakage) 없이” 학습자의 흐름을 따라가도록 보는 평가 설계를 강조합니다.

- **Technical Challenges**: 핵심 난제는(1) 블록 기반 도메인 지식과(2) 튜터링 행동을 동시에 정밀하게 채점하는 것입니다. 모델이 코드 자체를 잘 맞히는지보다, 힌트를 ‘정답으로 제공하지 않고’(hint_not_solution) 간결·행동가능·맥락지향으로 피드백하는지, 그리고 이전 시도(학생 디버깅 히스토리)를 인정하며 이어갈 수 있는지를 루브릭으로 수치화했습니다. 또한 자동 judge의 편향(특정 SLM이 점수를 과대평가)을 줄이기 위해 Claude Sonnet 4를 judge로 채택하고, 교육용 prompt engineering 연구에 기반한 시스템 프롬프트를 개정해 10/11 모델의 점수를 개선했습니다.

- **Empirical Impact**: 11개 모델(4B~120B)을 대상으로 한 예비 실험에서 어휘(vocabulary)와 톤(tone)은 대체로 강했지만, 정답 누설 방지와 학생 디버깅 이력 반영(acknowledges_progression) 같은 깊은 페다고지 기준은 전반적으로 어려움을 보였습니다. 30B 코딩 특화 모델이 중하위권에 머문 반면, 9B~4B급 모델이 상위권을 차지해 파라미터 수만으로 튜터 품질을 예측하기 어렵다는 신호가 나왔습니다. 더불어 프롬프트 개정(Trial 2)으로 10/11 모델이 개선했으며, 소수 모델 표본의 한계가 있지만 SLM 선택에서 도메인·페다고지 기반 맥락 평가의 가치를 실증적으로 뒷받침합니다.



### From Graphs to Gradients: Physics-Inspired Structural Attribution for Cyber-Physical IoT Systems and Beyond (https://arxiv.org/abs/2607.05563)
- **Prior Approaches**: 기존 인과 설명은 Structural causal models, counterfactual 설명, causal discovery를 통해 개입(intervention) 효과를 추적하려 하지만, feedback loop와 partial observability가 있는 대규모 하이브리드 시스템에서는 directed 그래프를 관측 데이터만으로 안정적으로 복구하기 어렵습니다. 또한 graph 기반 설명(GNNExplainer 등)은 예측 모델의 영향 노드/부분그래프를 찾는 데 강점이 있으나, 실제 시스템의 전역 의존성과 상호작용을 인과적으로 해석하는 데는 한계가 있습니다.

- **Core Contribution**: 이 논문은 통제된 IoT 사이버-물리 시스템에서 인과적 영향처럼 해석 가능한 “의존성 기반 attribution”을, directed causal graph 복구 없이 undirected 에너지 기반(energy-based) 표현으로 제공하는 프레임워크를 제안합니다. 통계역학 아이디어를 따라 정상 상태는 낮은 energy, 이상/공격 상태는 높은 energy로 두고, energy landscape의 변화가 각 구성요소의 영향도를 반영하도록 설계했습니다. 그 결과 attributions는 전체 생성 동역학을 완전히 복원하진 않지만, 구조를 고려한 설명과 후속 예측·진단 작업에 활용 가능한 정보를 준다고 주장합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 연속·이산 변수가 섞인 하이브리드 상태공간에서 (2) directed 인과구조 없이도 (3) 지역-전역 상호작용과 이상 징후에 대한 신뢰도 높은 attribution을 계산하는 것입니다. 저자들은 Boltzmann 분포 형태의 undirected 에너지 그래프를 두고, 먼저 energy에 대한 1차 민감도(gradient)로 지역 기여도를 계산한 뒤, free energy·entropy 분해를 통해 값 하나에 고정된 조건부 분포가 전체 확률질서에 미치는 영향을 전역 점수로 보강합니다. 마지막으로 Hessian 기반 2차 곡률을 선택적으로(1차 상위 변수에만) 계산해 영향의 안정성/상호의존성을 반영하고, 조건부 기대값·엔트로피는 Monte Carlo로 근사하되 하이브리드 변수 타입에 맞춘 샘플링 제안을 사용해 계산가능성을 확보합니다.

- **Empirical Impact**: 실험은 산업용 IoT 테스트베드 SWaT 시뮬레이션(공격 시나리오와 제어된 섭동 포함)에서 수행되며, 하이브리드 연속·이산 변수를 다루면서 attributions 정확도, 섭동에 대한 강건성, 그리고 시스템 크기에 따른 확장성에서 graph 기반 최신 접근 대비 향상된 성능을 보였다고 보고합니다. 특히 설명 관점에서는 공격/이상 상황에서 지배적인 기여 변수를 더 잘 찾아내고, 해석 관점에서는 dependency-aware한 설명을 제공해 인간 해석 및 진단 파이프라인에 도움이 된다는 점을 강조합니다. 또한 본 프레임워크는 산업 IoT 보안 외에도 고차원 사이버-물리·소시오테크니컬 시스템의 구조적(원리 기반) 설명이 필요한 문제로 일반화될 수 있다고 제안합니다.



### Prompt-to-Paper: Agentic AI System for Bioinformatics (https://arxiv.org/abs/2607.05456)
Comments:
          NA

- **Prior Approaches**: 기존 연구들은 지식그래프 기반 가설 생성이나 retrieval을 강화하는 방식, 혹은 모의환경/시뮬레이션 평가에 치중해 왔습니다. 다만 생성된 인용·주장에 대한 실시간 검증이 약하거나, 실험 수치를 실제로 실행하지 않고 합성값을 넣는 한계가 반복적으로 드러났습니다. 또한 자동 평가는 주로 단일 축 또는 시뮬레이터 의존 형태여서, 출판 수준의 엄격함을 재현 가능하게 담보하기 어렵다는 문제가 남아 있었습니다.

- **Core Contribution**: 이 논문은 Prompt-to-Paper로, 생물정보학 논문 생성을 “검증된 문헌 근거 + 실제 실험 실행 + 다차원 품질 평가”로 end-to-end 정렬하는 멀티에이전트 프레임워크를 제안합니다. 핵심은 주장마다 deterministic RAG 기반 grounding을 수행하고, 자율 coding agent가 계산을 실제로 실행해 숫자를 논문에 주입하며, 8개 품질 차원에 hallucination penalty까지 포함한 자동 평가를 표준화했다는 점입니다.

- **Technical Challenges**: 기여를 현실화하는 기술 난제는 (1) 생성 시점에 문헌 근거가 흐려지는 문제, (2) 실험 수치가 합성되는 문제, (3) 품질 점수가 반복 개선에서 흔들리는 문제였습니다. 이를 위해 section-aware relevance scoring과 snowball citation expansion으로 60–100편 corpus에 claim을 고정하고, sandbox에서 Python 실험을 실제 실행(재현 가능한 rigor.py 통계 포함)해 canonical results.json을 주입하도록 설계했습니다. 또한 deepseek-v4-pro/ deepseek-chat의 역할 분리를 통해 G-Eval류 채점의 구조화 실패를 줄였고, 8차원 scorer는 z-normalisation과 차원별 안정화(rejudge 조건), citation-range 체크 및 [N] 범위 외 인용 페널티를 적용해 개선 루프의 방향성을 보존합니다.

- **Empirical Impact**: 5개 생물정보학 케이스에서 시스템은 before/after 각각 submission-formatted PDF를 생성했고, 모든 run에서 zero out-of-range citations를 달성했습니다. 개선 루프는 평균 +17.96점(0–100) 향상을 보였으며, CpG Island Detection에서는 최대 +26.04점까지 상승했습니다. 자동 점수(평균 63.56/100) 외에도 독립 LLM 3종 및 인체 평가에서 B–B+ 범위와 유사한 경향이 관측돼, “실제 실험·검증·표준화된 품질 게이트”가 통한다는 실증적 근거를 제공합니다.



### ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation (https://arxiv.org/abs/2607.06565)
- **Prior Approaches**: 통합 3D 파운데이션 모델은 한 백본에서 이미지-3D 생성, text-to-3D 생성, 3D 캡셔닝까지 수행하려 하지만, 텍스트-3D 상호작용이 대부분 암묵적으로 처리됐다. 기존 접근은 텍스트 토큰과 3D 토큰을 평평한 시퀀스로 이어 붙인 뒤 self-attention에 의존해 대응을 찾는 방식이라, 구조적 신호와 정밀 기하 디테일이 한 덩어리 표현으로 뭉치기 쉽다. 또 멀티스케일 3D 표현을 쓰더라도, 언어 추론이 어떤 스케일의 기하 근거를 참조해야 하는지까지는 아키텍처가 충분히 구조화하지 못했다.

- **Core Contribution**: ELSA3D는 통합 모델의 텍스트-기하 정렬을 ‘elastic semantic anchoring’으로 명시화해, 언어 추론과 기하 추론을 동일한 추상화 스케일에 맞춰 함께 구조화한다. 3D는 scale tag가 포함된 scale-aware octree tokenizer로 표현하고, 언어는 Global/Structure/Appearance로 분해한 semantic trace로 설계한다. 여기에 Anchor Tokens라는 희소한 크로스모달 인터페이스를 도입해, 선택된 의미 단서가 특정 3D 스케일의 근거를 조회한 뒤 통합 표현에 다시 기록되도록 한다.

- **Technical Challenges**: 문제는 (1) 언어는 종종 기하의 정확한 세부를 생략하는 ‘under-specified’ 조건이고, (2) 모든 텍스트 토큰을 모든 3D 토큰과 촘촘히 결합하면 계산량이 폭증하며 의미 잡음도 커진다는 점이다. ELSA3D는 scale-aware octree에서 스케일 태그와 위치 정보를 갖는 기하 토큰을 만들고, Anchor Tokens로 필요한 의미 토큰만 골라 특정 스케일의 기하 증거를 교차어텐션으로 가져온 뒤 write-back한다. 또한 per-block elastic router가 블록 실행/MLP width/어떤 토큰이 어떤 스케일에 anchor를 둘지까지 함께 결정해, 정렬이 필요한 곳에만 추론과 연산을 집중한다.

- **Empirical Impact**: 실험 결과 ELSA3D는 image-to-3D 생성, text-to-3D 생성, 3D captioning 전반에서 SOTA를 달성했으며, strongest unified baseline 대비 성능이 전 과제에서 일관되게 개선됐다. 특히 ablation에서 anchor routing의 희소성이 dense한 텍스트-3D 융합보다 더 잘 작동함을 보였고, FLOPs는 1081G에서 632G로, 추론 지연은 29.8s에서 17.2s로 줄였다. 즉, 비(非)elastic 버전 대비 FLOPs와 latency를 대략 절반 수준으로 낮추면서도 생성 품질과 언어-기하 이해 정확도를 동시에 끌어올린 점이 의미 있다.



### Graph Convolutional Attention: A Spectral Perspective on Graph Denoising and Diffusion (https://arxiv.org/abs/2607.06546)
- **Prior Approaches**: 그래프 노이즈 제거는 그래프 확산 모델에서 핵심 연산이며, 최근에는 graph transformer처럼 attention 기반 구조가 denoising에 강점을 보였습니다. 다만 기존 attention이 그래프 denoising에서 어떤 스펙트럼 메커니즘으로 동작하는지에 대한 이론적 이해는 부족해, 표준 attention이 최적인지 불명확했습니다.

- **Core Contribution**: 이 논문은 그래프 denoising 관점에서 linear attention의 한계를 먼저 규명합니다. denoising 목적을 만족하는 상황에서 linear attention은 학습 분포에 대해 “평균적인 spectral filter”만 학습하며, 그래프들이 분포 전반에서 서로 다른 스펙트럼을 갖는 경우 근본적으로 불리하다고 보입니다. 이를 개선하기 위해 입력 그래프의 eigenvalues(스펙트럼)에 직접 의존하는 Spectral Attention을 제안하고, 이를 실전형이며 permutation-equivariant하게 구현한 Graph Convolutional Attention(GCA)을 도입합니다.

- **Technical Challenges**: Spectral Attention은 스펙트럼에 임의의 방식으로 의존할 수 있지만, 일반적으로 permutation equivariant하지 않아 그대로는 모델로 쓰기 어렵다는 점이 기술적 난관입니다. 논문은 이 문제를 그래프-filtered queries/keys로 바꿔 attention 패턴을 graph convolutional filter로 표현하는 방식(GCA)으로 해결하고, 특히 큰 Stochastic Block Models(SBMs)에서는 GCA가 이상적 Spectral Attention과 같은 손실을 달성함을 보입니다. 또한 attention 뒤에 오는 softmax가 noisy eigenvectors를 clean eigenspace로 “대략적인 projection”하여 추가 denoising을 제공한다는 스펙트럼 관점의 해석을 제시합니다.

- **Empirical Impact**: 실험적으로 linear attention을 GCA로 교체하면 synthetic/real 데이터 전반에서 그래프 denoising과 diffusion 모두 일관되게 성능이 향상되며, 이 개선폭은 스펙트럼 다양성(spectral diversity)과 강하게 상관됩니다. DiGress에서는 GCA가 비싼 구조적 특징이나 eigendecomposition 없이도 표준 graph-transformer 성능과 경쟁하며, R-PEARL positional encodings과 결합하면 eigenvector 계산을 피하면서도 품질 저하 없이 더 빠른 추론을 얻는다고 보고합니다. 결과적으로 “attention의 스펙트럼적 적합성”이 확산 모델 성능에 직접 연결된다는 실증적 신호를 제공하며, 후속 연구가 스펙트럼 기반 attention 설계를 더 적극적으로 다루도록 동기를 줍니다.



### RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation (https://arxiv.org/abs/2607.06527)
Comments:
          Accepted for publication in ICML 2026 as a full research paper; 21 pages

- **Prior Approaches**: KGQA는 보통 ‘retrieve-then-read’로 다중 홉을 풀지만, 각 홉에서 이산 노드를 선택하는 과정 때문에 end-to-end 미분 가능성이 깨져 검색기가 downstream 오류를 교정하기 어렵습니다. 또한 중간 bridge 노드가 질의와 어휘적 겹침이 거의 없는 semantic gap 상황에서는 entity-linking 기반 방법이 성능이 급격히 저하됩니다. 최근 LLM 에이전트형 접근은 추론 구조를 만들지만, billion-parameter 모델을 여러 번 통과해야 해서 비용이 커지고, 생성기가 검색 구조를 무시하는 reasoning shortcut 문제도 남아 있습니다.

- **Core Contribution**: 이 논문은 RSF-GLLM( Recurrent Soft-Flow Graph-to-LLM )로 그래프 추론과 답변 생성을 분리해, differentiable graph reasoning을 안정적으로 학습하면서도 LLM의 고비용/고분산 학습 영향을 최소화합니다. Recurrent Soft-Flow(RSF)는 Recurrent Query Updater(GRU)를 통해 연속 relevance score(soft flow)를 전파하고, Dynamic Gating Mechanism으로 구조 단서만으로도 의미적으로 멀리 있는 bridge 노드를 탐색합니다. 추출된 reasoning path를 텍스트화해 LLM fine-tuning에 활용함으로써 답변이 지식그래프의 토폴로지에 근거하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 중간 홉에서 질의-노드 의미 유사성을 강제하면 정답 경로가 막히는 semantic gap과 (2) soft flow가 그래프 전체에 퍼지는 flooding 문제입니다. 논문은 구조적 전파와 내용 기반 분별을 분리하고, λ(t) 게이팅으로 fan-out 해소가 필요한 순간엔 content bias를 켜되, semantic gap 구간엔 끄도록 학습합니다. 더불어 flow sparsity regularization(엔트로피 기반)을 적용해 soft 확률이 이산 경로로 수렴하도록 이론적으로 유도하고, greedy backtracking으로 구조적으로 타당한 경로를 복원·텍스트화합니다.

- **Empirical Impact**: WebQSP와 CWQ 실험에서 RSF-GLLM은 competitive 성능을 달성하면서도 LLM 중심/에이전트형 접근 대비 inference 효율이 더 높다고 보고합니다. 특히 semantic-gap-heavy 쿼리에서의 견고성 향상이 강조되며, 추론 경로를 명시적으로 제공해 hallucination 위험을 줄이는 방향으로 의미가 있습니다. 결과적으로 ‘학습 가능 추론 경로 + 근거 있는 생성’의 결합을 KGQA 실무 배치 효율 측면에서 한 단계 끌어올렸다는 평가를 기대할 수 있습니다.



### Industry Classification of GitHub Repositories Using the North American Industry Classification System (NAICS) (https://arxiv.org/abs/2607.06505)
- **Prior Approaches**: 기존에는 GitHub 저장소를 산업 분야(섹터)로 표준화해 매핑하는 기본 제공 기능이 없어, 혁신의 지리·산업 구성·기술 확산을 실증적으로 분석하기 어려웠다. 연구자들은 라벨링을 직접 수작업하거나 비표준 태그를 활용하는 방식에 의존해 재현성과 규모 확장에 한계가 있었다.

- **Core Contribution**: 이 논문은 NAICS-GH라는 공개 코퍼스를 제안한다. 미국·EU·호주에서 수집한 6,588개 GitHub 저장소를 NAICS 2022의 2-digit 섹터 라벨로 매핑해, 저장소-산업 연결을 실험 가능한 형태로 제공한다.

- **Technical Challenges**: 핵심 난제는 대규모 저장소 풀에서 섹터 라벨을 정확히 찾고 검증하는 것이다. 연구진은 BAAI/bge-large-en 임베딩과 FAISS retrieval로 후보(31,178개 저장소-섹터 페어)를 만든 뒤, GPT-4.1 rubric scoring의 retrieve-and-verify 파이프라인으로 고신뢰 라벨(최소 8점, 6,588개)을 남겼고 end-to-end 재실행 시 후보셋 재현성을 0.03% 이내로 확인했다.

- **Empirical Impact**: 휴먼 검증된 무작위 샘플(2,421개)에서 라벨 정밀도는 96.98%에 이르렀고, 95% 신뢰구간[96.23, 97.59]으로 품질을 제시했다. 또한 6개 pretrained encoder 벤치마크에서 RoBERTa-large가 held-out 20% 테스트셋 F1 86.45%/정확도 86.35%를 기록했으며, 데이터셋·메타데이터·파이프라인·체크포인트를 공개해 산업 분류 기반 오픈소스 기술 연구의 기반을 마련했다.



### Pitwall: Faithful Natural-Language Race-Strategy Briefings from a Calibrated Real-Time Monte Carlo Engin (https://arxiv.org/abs/2607.06495)
Comments:
          21 pages, 2 figures, 6 tables. Live-deployment results from the 2026 Austrian and British Grands Prix. URL: this https URL

- **Prior Approaches**: 기존 F1 레이스 전략 시뮬레이션은 결정론적 최적화나 랩타임/타이어 열화 재현에 초점이 있었지만, 승부 확률의 캘리브레이션(신뢰도)까지 실시간으로 보장하는 공개 사례는 드뭅니다. 또한 확률 기반 반사실(예: 지금 피트 vs 2랩 대기) 비교를 common-random-numbers로 통제해 전략 차이를 분리하거나, 라이브 타이밍 스트림을 직접 받아 생성까지 연결한 end-to-end 시스템도 거의 없었습니다. 마지막으로 데이터-투-텍스트 생성은 출처 없는 환각 위험이 커서, “실제 선수에 대한 문장”을 마감 시간 내에 내보내려면 검증 체계가 필수입니다.

- **Core Contribution**: Pitwall은 라이브 스포츠 코멘터리의 grounded generation을 ‘기법의 목표’가 아니라 아키텍처 속성으로 다루며, 생성 문장을 타입화된 사실 청구(claim)로 분해해 확률 레이스 상태로 검증한 뒤 통과한 문장만 공개합니다. 더 나아가 verifier가 fine-tuning 데이터 자체의 채택을 게이트하며, 3,045개 모델 생성 타깃 중 모든 claim이 상태를 지지하는 81.9%만 학습에 남기고 나머지는 provably faithful 템플릿으로 폴백합니다. 같은 verifier가 생성·학습·운영 전 단계에 걸쳐 ‘근거 없는 서술’이 시스템에 들어오지 못하도록 막는 구조가 핵심입니다.

- **Technical Challenges**: 가장 큰 난제는 (1) 매 몇 초마다 변하는 확률적 그라운딩 상태를 기준으로, (2) 마감 지연 내에, (3) 문장 단위 claim을 신뢰도 있게 검증하며, (4) fine-tuning으로 인한 풍부함이 희소 상태에서 환각으로 무너지는 문제를 통제하는 것입니다. Pitwall은 벡터화된 Monte Carlo 엔진(N=2,000, 상태를 N×C 배열로 동시에 전진)과 확률 캘리브레이션된 SC/VSC·dirty-air·은근 오버테이크 상호작용을 함께 두고, 언어 계층에서는 위치/갭/타이어/페이스/오버테이크/레이스 컨트롤 등 10종 claim 스키마를 3개 언어로 추출·검증합니다. 또한 ‘캘리브레이션 최적 vs 결정 최적’이 충돌할 때를 분리 게이트로 처리하며, 파운데이션 모델의 instruction adherence가 희소 컨텍스트에서 환각을 유발하는 바탕 요인임을 4개 베이스 모델 감사로 확인해 해결책으로 sparse-context auditing을 운영 모델에서 제거합니다.

- **Empirical Impact**: Calibrated Monte Carlo 엔진은 126개 학습 레이스로 보정(2018–2024)하고, 2025–2026 완전 홀드아웃에서 winner-in-top-3 90.3%(155개 백테스트)와 held-out Brier 0.0745를 보여 확률 품질을 실증합니다. 언어 생성은 verifier 게이트로 상태를 뒷받침하는 문장만 채택되도록 설계됐고, 풍부한 타깃 학습이 항상 좋은 결과를 주지 않으며 결함은 스케일보다 베이스 모델의 지시 준수/감사 설계 문제임이 드러납니다(역효과도 함께 보고). 2026년 오스트리아·브리튼 라이브 그랑프리에서 라이브 타이밍→상태 재구성→추천→검증된 3개 국어 코멘터리까지 end-to-end 운영을 확인했으며, 실버스톤에서는 결과가 알려지기 전 디스크에 커밋한 확률 타임스탬프 트레이스가 깃발 10랩 전부터 최종 우승자를 고정하는 것으로 보고됐습니다.



### AirflowAttack: Thermal-Airflow Adversarial Perturbations against Infrared Remote-Sensing Vision-Language Models (https://arxiv.org/abs/2607.06485)
- **Prior Approaches**: 기존 원격탐사 vision-language model(VLM)은 주로 정상(benign) 환경에서 성능을 평가해 왔고, IR 원격탐사에서의 보안 취약성은 거의 다뤄지지 않았다. RGB 영역에서는 white-box/black-box/Universal UAP 등 다양한 적대 공격이 연구됐지만, 열화상 단일 채널의 물리적 의미(온도에 준하는 방사 강도)와 그에 기반한 ‘물리 그럴듯한 교란’이 IR VLM에 주는 영향은 공백이었다.

- **Core Contribution**: 이 논문은 IR 원격탐사 VLM을 겨냥한 최초의 적대 공격 프레임워크 AirflowAttack을 제안한다. 열-대기 흐름 난류(thermal-airflow turbulence)에서 영감을 받은 공기(airflow) 우선(prior)을 도입해, 입력에 무관한 단일 universal perturbation을 생성·최적화하며 물리적으로 해석 가능한 교란을 목표로 한다.

- **Technical Challenges**: 핵심 과제는 (1) 타깃 VLM에 대한 접근 없이도 전이되는 공격을 만들고, (2) 단순 디지털 잡음이 아니라 열 흐름의 공간적 상관을 갖는 교란을 생성하며, (3) 교란 크기(L∞≤ε) 제약까지 만족시키는 것이다. 이들은 저차원 잠재변수에서 열-흐름 패턴을 내는 lightweight generator로 perturbation을 구성하고, surrogate CLIP에서의 confidence 저하 손실에 더해 airflow correlation loss로 물리적 그럴듯함을 정규화해 해결했다.

- **Empirical Impact**: Surrogate CLIP(한 모델)에서 최적화한 단일 perturbation은 5개 CLIP 백본 평균 zero-shot scene-classification ASR 48.5%를 달성해 IR 특화 물리 베이스라인(27.7~37.0%)을 크게 앞섰다. 6개 최신 VLM에 적용하면 장면 분류 정확도를 최대 38.2% 상대적으로 낮추는 한편, 일부 모델은 오히려 IR-cue에서 더 높은 ‘확신’을 보이며 온도 구배·대류 같은 가짜 열 증거로 교란을 ‘사실처럼’ 받아들이는 역설적 현상도 관찰됐다. 또한 11개 모델·4개 태스크에 걸친 벤치마크와 ablation으로 airflow prior가 공격 성공을 해치지 않으면서 물리적 타당성을 올려 취약성을 체계적으로 드러냈다는 점에서 의미가 크다.



### Data Analysis in the Wild: Benchmarking Large Language Models Against Real-World Data Complexities (https://arxiv.org/abs/2607.06482)
Comments:
          29 pages, 9 figures

- **Prior Approaches**: 기존 LLM 기반 데이터 분석 벤치마크는 작은 테이블에서의 사실 검색이나 text-to-SQL·Table QA에 치우쳐, 대규모 멀티 탭ular 데이터와 메타데이터/외부 지식 통합의 어려움을 충분히 반영하지 못했다. 또한 질문에 대한 응답은 평가하지만, 데이터 분석가가 수행하는 탐색적 인사이트 발견 능력은 상대적으로 덜 측정되는 편이었다. 그 결과 실제 현장형 문제에서의 성능 격차가 드러나기 어려웠다.

- **Core Contribution**: DataGovBench는 정부 오픈데이터를 기반으로 Table QA(분해형 질문에 대한 텍스트/시각화 정답)와 Table Insight(사용자 질의 없이 탐색해 전문가 수준의 발견을 생성)를 함께 평가하도록 설계했다. 특히 인사이트의 기준 정답을 전문가 보고서에서 추출해 주관성 문제를 완화하고, 대규모·다중 테이블·메타데이터·외부 지식이 동반되는 현실 복잡도를 포함한다. 이로써 “답하기”와 “발견하기”를 동시에 밀도 있게 검증하는 새 기준선을 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) LLM 컨텍스트 한계를 고려한 테이블 직렬화와 정보 압축, (2) 암묵 조건·데이터 변환·다중 테이블 선택 같은 정답성 요구, (3) 인사이트 생성의 주관성 및 정량 평가가 어렵다는 점이다. 논문은 feature type-specific table serialization로 요약 입력을 만들고, Answer Agent에서 파이썬 생성-실행 실패 복구(self-correction)와 시각화/계산 검증(reflection) 루프를 넣어 오류를 줄이려 했다. 인사이트는 전문가 보고서에서 인사이트를 추출·표준화해 ground truth를 구성하고, 평가도 의미 정렬 기반 점수(LLaMA-3-Eval 유사)를 통해 분해해 측정했다.

- **Empirical Impact**: 실험 결과, Answer Agent 같은 에이전트 지원이 있더라도 Table QA와 Table Insight 모두에서 최신 LLM의 성능은 여전히 낮아 현실 데이터 분석 요구를 충족하기엔 큰 격차가 있음을 확인했다. 특히 Table QA에서는 condition filter error와 transformation error 같은 정합성 문제와 시각화/테이블 선택 오류가 두드러졌고, Table Insight에서는 주제는 맞추더라도 narrative 및 정성·정량 세부 일치가 약했으며 정량 값 재현은 거의 불가능에 가까웠다. 저자들은 이로부터 현재 에이전트에 “탐색적 서사 수준 추론”과 “복잡 테이블에서의 정확한 사실 회수” 능력이 부족하다는 관찰을 제시한다.



### Prompt-Adapter Context Routing for Parameter-Efficient Multi-Shot Long Video Extrapolation (https://arxiv.org/abs/2607.06481)
Comments:
          10 pages, 2 figures

- **Prior Approaches**: 장기 비디오 생성은 기존의 샷 플래닝, 스토리 메모리, 스트리밍/에이전트 방식 등으로 접근돼 왔습니다. 특히 recursive context allocation 계열은 “어떤 과거 컨텍스트를 다음 샷에 줄지”를 결정해 드리프트를 줄이려 하지만, 대체로 큰 부분을 fine-tuning 하거나 외부 메모리 모듈 의존도가 생겨 장기 구간에서 retrieval이 불안정해질 수 있습니다.

- **Core Contribution**: PACR-Video는 text-to-video diffusion transformer를 frozen으로 두고, 저랭크 temporal adapters와 shot-role prompt tokens, recursive prompt bank를 조합해 멀티샷 장기 외삽을 수행합니다. 핵심은 dense한 비디오 메모리 대신 entity/location/action/style 요약 프롬프트를 저장하고, narrative dependency 예측에 따라 adapter gates로 필요한 것만 선택 라우팅한다는 점입니다.

- **Technical Challenges**: 장기 생성에서 오류가 샷 누적로 커지는 문제를 해결하려면, (1) 오래 남는 정체성/스타일은 유지하면서 (2) 샷마다 새 모션·뷰포인트·인과 이벤트는 진행되도록 제어해야 합니다. 논문은 Shot-Local/Story-Global 학습목표(다음 샷 재구성, cross-shot identity contrast, prompt sparsity)와, early-shot 시각 일관성을 강화하고 late-shot 이벤트 진행을 늘리는 adapter composition schedule로 long-horizon coherence를 맞춥니다.

- **Empirical Impact**: FlintstonesSV, Pororo-SV, ActivityNet Captions, YouCook2, Shot2Story, MovieNet의 6개 벤치마크에서 PACR-Video는 텍스트-투-비디오, tuning-based, streaming, memory-augmented, recursive-context baselines보다 모든 핵심 지표(FVD, identity consistency, temporal smoothness, 전이 일관성 등)에서 우수했습니다. 특히 ReCA 대비 FVD 268.4→231.7, DINO identity consistency 0.724→0.771로 개선했고, 사람 평가에서도 ReCA보다 63.8% 더 선호되었으며, 백본 파라미터의 3.8%만 튜닝해 실용성까지 확보했습니다.



### Provable learning separation for predicting time-evolution of quantum many-body systems (https://arxiv.org/abs/2607.06472)
Comments:
          48 pages, 1 figure

- **Prior Approaches**: 양자컴퓨터가 양자 다체계 시뮬레이션에 자연스럽다는 점에서, QML이 ‘학습 분리(learning separation)’를 보일 수 있는지에 대한 질문이 제기돼 왔다. 기존 연구들은 주로 추상적 QML 과제나 특정 모델에서의 성능 차이를 다뤘지만, PAC-learning 관점에서 양자 학습 가능성과 고전적 불가능성을 동시에 엄밀히 분리하는 데는 한계가 있었다.

- **Core Contribution**: 이 논문은 양자 다체 동역학을 PAC-learning 틀에서 다뤄, 해밀토니언 진화 기반의 자연스러운 지도학습 과제를 설계하고 그 학습 난이도를 이론적으로 정리한다. 학습 데이터로 랜덤화된 stabilizer probe state의 사양과 (다항적으로 큰 시간구간에서 균등 샘플된) 진화 시간, 그리고 미지의 해밀토니언에 대한 특정 관측량의 기대값을 사용하며, 이로부터 해밀토니언을 학습하고 새로운 입력에 추론하도록 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 짧은 시간 학습 샘플로 해밀토니언을 효율적으로 복원/학습해야 하고, (2) 긴 시간 동역학의 추론에 대해 고전적으로도 어려운 구조를 보존해야 한다는 점이다. 저자는 학습 단계에서 짧은 시간 샘플로 해밀토니언을 배우는 효율적인 양자 절차를 제시하고, 배치 단계에서는 Hamiltonian simulation과 classical shadows를 결합해 새 데이터 포인트에 대한 추론을 수행한다.

- **Empirical Impact**: 또한 특정 입력 분포 계열에 대해, Feynman-Kitaev clock Hamiltonian의 low-intersection 변형에 BQP-complete 계산을 삽입해 다항 시간 고전 랜덤 알고리즘이 학습 조건을 만족할 수 없음을 보이며, 그 결론은 BQP⊄P/poly 가정 하에 성립한다. 그럼에도 해당 고난도 인스턴스는 양자적으로는 여전히 학습 가능하다는 점을 함께 보여, 양자 학습 이론–양자 시뮬레이션–QML 사이의 ‘엄밀한 학습 분리’를 제공한다.



### From Voting to Agent Collaboration: Answer-Type-Aware LLM Pipelines for BioASQ 14b (https://arxiv.org/abs/2607.06452)
Comments:
          15 pages

- **Prior Approaches**: 기존 BioASQ Task B 연구는 retrieval-augmented LLM으로 단일 프롬프트를 적용하거나, 앙상블/agent 기반 기법을 각각 독립적으로 사용해 왔습니다. 하지만 yes/no는 근거 스니펫의 순서와 구성에 민감하고, factoid와 list는 동의어·표면형식·검증 전략 부족으로 정답 순위/정확도가 흔들리는 문제가 컸습니다. 또한 여러 스니펫이 상충할 때 어떤 근거를 우선해야 하는지 결정하는 일관성이 부족하다는 한계가 반복됐습니다.

- **Core Contribution**: 이 논문은 질문 유형(yes/no, factoid, list)에 맞춘 question-type-specific LLM 추론 프레임워크를 제안합니다. 각 유형에 서로 다른 추론 절차를 배치해 yes/no는 snippet shuffling+self-reflection으로 결정 안정성을 높이고, factoid는 full-snippet 기반 in-context learning과 consensus로 정밀한 생의학 개체 식별을 강화합니다. list는 evidence 추출-후보 생성-검증-집계로 역할을 분리한 multi-agent 협업으로 과생성/누락을 동시에 줄이는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 난제는 (1) 스니펫 순서 변화에 따른 출력 흔들림, (2) 생의학 엔티티의 표면형식/동의어 차이로 인한 exact-match 평가 실패, (3) list에서 높은 재현율과 높은 정밀도를 함께 만족시키는 검증 설계였습니다. 저자들은 질문 유형별로 서로 다른 입력 구성(스니펫 셔플, full-snippet), in-context learning 데모(유사 질문 기반 retrieval + 검증된 CoT 예시), 그리고 선택적 verification/합의 규칙(yes/no majority+검증 에이전트, factoid 투표 기반 후보 필터링, list의 4단계 agent 파이프라인)을 결합해 이 문제들을 완화했습니다.

- **Empirical Impact**: BioASQ 14b 공식 평가에서 전 배치에 걸쳐 경쟁력 있는 성능을 보였고, 특히 Batch 4의 factoid 서브태스크에서 1위를 기록했습니다. 지표 관점에서도 yes/no는 macro F1이 비교적 안정적이었고, list는 배치가 진행될수록 F-measure가 크게 상승해 협업형 검증 파이프라인의 효과가 확인됐습니다. 다만 factoid는 strict/lenient 정확도 차이와 MRR 변동성이 남아 있어, 동의어 정규화·약어 처리·후순위 랭킹 최적화가 후속 과제로 제시됐습니다.



### Analysis-by-Proxy: Localization Signals in VLMs Operating as Condition Encoders (https://arxiv.org/abs/2607.06445)
Comments:
          Accepted as a Spotlight at the ICML 2026 Mechanistic Interpretability Workshop

- **Prior Approaches**: 기존 확산 기반 이미지 편집은 VLM을 condition encoder로 두고, DiT에 전달할 VLM 내부 표현을 미리 정한 특정 레이어(대개 final-layer 토큰, 레이어 풀링, 일부 레이어를 DiT 레이어에 매핑)에서만 뽑는 방식이 주류입니다. 하지만 다중 객체 장면에서 원하는 대상을 정확히 고르는 localization이 자주 무너져, 약간의 신호 손실만으로 잘못된 위치/환각 편집이 발생합니다. 또한 VLM 해석 연구는 주로 autoregressive 텍스트 생성 기반으로 진행되어, single forward pass 제한에서의 내부 동작은 충분히 규명되지 않았습니다.

- **Core Contribution**: 이 논문은 VLM이 single-pass condition encoder로 사용될 때 성능 격차가 왜 생기는지 분석하며, 핵심 원인을 “VLM의 공간 지식이 condition extraction 과정에서 제대로 디코딩되지 못함”으로 제시합니다. 이를 위해 Analysis-by-Proxy라는 프레임워크를 도입해, 생성 없이도 VLM 중간 표현에서 localization 정보를 어떤 레이어/토큰에 담는지 분해합니다. 더 나아가 proxy가 복원한 공간 신호(바운딩 박스)를 DiT 조건에 통합해 실제 편집 localization 실패를 줄이는 방법을 제안합니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 텍스트 토큰을 생성하지 않는 single-pass 환경에서 연속 hidden state 안의 공간 정보를 “디코더 없이” 찾아내야 한다는 점입니다. 논문은 이를 해결하기 위해 Q-Former 기반의 lightweight proxy를 VLM 중간 표현에 학습시켜, auxiliary localization task로 바운딩 박스를 회귀하게 만들고 decodability가 높은 레이어/토큰 위치를 역추적합니다. 결과적으로 final-layer에선 공간 신호가 과추상화되어 약해지고, 중간 레이어에서 신호가 강하지만 ‘입력 프롬프트마다’ 피크 레이어가 달라져 고정 레이어 추출 전략이 본질적으로 비최적임을 보였습니다.

- **Empirical Impact**: 200개 다중 객체 장면 평가에서 VLM 단독은 89.0%로 정확한 바운딩 박스를 예측했지만, 표준 편집 파이프라인은 57.5%로 크게 하락해 31.5%p 격차를 실증적으로 확인했습니다. Analysis-by-Proxy로 복원한 중간 레이어 공간 신호를 LoRA로 DiT에 조건화하면, 기존 Qwen-Image-Edit 대비 VQA 기반 의미 편집 성공을 높이고 대상 외 배경 LPIPS는 낮게 유지하는 결과가 보고됩니다. Full Description 같은 autoregressive 변형이 강한 베이스라인으로 관찰되어, sequential decoding이 공간 단서를 끌어내는 한편 single-pass 추출 설계가 병목임을 더 뒷받침합니다.



### TILDE: TILt-based Distributional Erasure for Concept Unlearning (https://arxiv.org/abs/2607.06432)
- **Prior Approaches**: 기존 개념 unlearning은 score suppression, anchor 기반 편집, reward/선호 최적화, trajectory steering, GFlowNet 기반 샘플링 등으로 “잊기” 자체는 달성하지만, 업데이트 후 분포가 어디로 이동하는지(사후 분포 목표)가 명시적이지 않은 경우가 많습니다. 그 결과 강한 지우기 과정에서 의미적으로 인접한 benign 개념의 손상(컬래터럴 데미지)이나 다양성 붕괴 같은 retain 실패가 함께 발생하기 쉽습니다. 또한 anchor를 목적지로 쓰는 방식은 고정·편향·순차 삭제 누적 등으로 불안정해질 수 있습니다.

- **Core Contribution**: TILDE(TILt-based Distributional Erasure)는 개념 unlearning을 “잊기 제약” 아래에서 사후 조건부 분포가 pretrained 분포에서 최소로 벗어나도록 정렬하는 분포 정렬 문제로 재정의합니다. 즉, replacement 개념을 지정하거나 단일 안전 출력의 모드를 찾는 대신, prompt별로 concept-expressing 이미지의 확률 질량을 줄이면서 benign 영역의 상대적 확률 질량은 보존하는 최소-이탈(minimal-deviation) 목표를 명시적으로 세웁니다. 이를 통해 unlearning의 핵심인 effective forgetting과 distributional fidelity, local preservation을 한 프레임에서 동시에 겨냥합니다.

- **Technical Challenges**: 문제는 (1) “어떤 이미지를 잊을지”를 에너지로 모델링하고 (2) 그 에너지 기울이기(energy tilt)를 diffusion 생성 과정에 맞게 학습 가능한 형태로 구현하는 것입니다. TILDE는 CLIP concept evidence를 thresholded forget energy로 만들어 임계값 이하에서는 페널티를 주지 않게 설계해, 주변 benign 개념까지 함께 깎이는 현상을 줄입니다. 또한 residual ∇∇-GFlowNet 기반으로 pretrained 디노이저에 대한 잔차 score correction만 학습해, terminal energy로 정의된 Gibbs형 타깃 샘플링을 diffusion latent space에서 효율적으로 근사합니다.

- **Empirical Impact**: Stable Diffusion v1.5에서 약 60개 개념(오브젝트/캐릭터/아트 스타일/누드)을 대상으로, forgetting 정확도와 함께 related/general retention 및 분포 정합성(FID, 그리고 retain-only 기준에 대한 FADE)까지 폭넓게 평가한 결과 TILDE가 기존 베이스라인 대비 더 강한 잊기와 더 나은 보존을 동시에 달성했습니다. 특히 “강한 지우기 ≠ retention 붕괴”라는 실패 모드를 줄이며, 분포 수준에서 gold-standard retain-only에 더 가깝게 정렬된다는 점을 FADE로 보였다는 것이 의미가 큽니다. 연구진은 VLM 기반 자동 평가(Qwen2.5-VL)로 다양한 개념 유형에서도 일관된 향상을 관측했다고 보고합니다.



### An Experimental Design Approach to Evaluating Agentic AI's Autonomous Model Discovery (https://arxiv.org/abs/2607.06413)
Comments:
          39 pages, 11 figures, 6 tables. Data and code available at the GitHub repository listed in the paper

- **Prior Approaches**: 기존 연구는 LLM을 사람을 대신하는 모사 참여자(시뮬레이션)로 쓰거나, 에이전트가 생성한 결과가 사람처럼 행동하는지에 초점을 두는 경우가 많았다. 또한 자동 모델 탐색을 돕는 연구나 코딩 에이전트 벤치마크가 있더라도, 에이전트의 통제값(예: reasoning effort) 변화에 따라 발견된 모델의 품질·비용·과정 복잡도가 어떻게 함께 변하는지는 충분히 다루지 못했다. 특히 확률적이고 순차적으로 동작하는 model-discovery operator의 동작을 단일 실행 결과로는 안정적으로 규정하기 어렵다는 점이 공백으로 남아 있었다.

- **Core Contribution**: 이 논문은 LLM 코딩 에이전트를 ‘model-discovery operator’(확률적 모델 발견 연산자)로 보고, 반복 실행에서 나타나는 변동성과 요인효과를 체계적으로 평가하는 실험 설계·분석 프레임워크를 제안한다. Codex와 Claude Code를 대상으로 reasoning effort, 작업(Task), 최적화 지표(optimization metric), 학습 데이터 조성(discovery regime) 등을 통제한 factorial design을 구성하고, 발견 과정이 품질과 비용 및 과정 복잡도를 어떻게 이동시키는지 관찰한다. 나아가 Utility-Aligned Canonical Decomposition(UACD)로 reasoning-effort 효과의 지배적 방향과 ‘효용(성능-비용)’ 정렬 여부를 함께 진단한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 에이전트가 stochastic/adaptive라서 single-run 벤치마크로는 행동을 설명하기 어렵고, (2) 성능·달러 비용·wall-clock 시간·프로세스 복잡도가 동시에 반응(response vector)으로 얽혀 해석이 필요하다는 점이다. 이를 위해 네트워크 기반 group-anagram 게임의 세션 데이터를 고정하고, held-out fold를 통해 작업 적합도(예: next-action 예측, ABM 생성 모델 적합)를 평가하면서도 에이전트 실행은 격리(workspace)해 데이터 누수를 막는다. 또한 wAUC, MRI-RO, KL 기반 player-level 요약 거리, Levenshtein edit-distance 분포 거리처럼 서로 다른 스케일의 지표를 stratum 내부에서 표준화해 다변량 분석이 가능하도록 설계했으며, reasoning effort는 제공사 간 의미가 달라 agent 내부에서 ordered ladder(low/default/max 또는 medium/high/max)로 취급한다.

- **Empirical Impact**: 실험은 networked word-forming games 테스트베드에서 수행되며, 각 agent-task-metric 조합마다 후보 모델·토큰/달러 지출·실행 시간·코드 스크립트·프로세스 복잡도까지 실행 트레이스를 수집해 품질-비용-과정의 동시 변화를 회귀/추론으로 분석한다. 논문은 특히 reasoning effort가 비용과 과정 복잡도에 대해 어떤 방향으로 움직이는지(그리고 그 방향이 성능-비용 효용과 정렬되는지)를 UACD로 해석하는 통찰을 제시한다. 또한 공개 데이터셋과 재현 가능한 평가 파이프라인을 제공해, 향후 다른 코딩 에이전트/설정에서도 유사한 ‘통제값-발견행동’ 과학적 평가가 가능하도록 기반을 마련했다.



### RuBench: A Repository-Level Agentic Coding Benchmark with Natively Authored Russian Task Specifications (https://arxiv.org/abs/2607.06411)
Comments:
          16 pages, 1 figure, 7 tables. Benchmark: 25 natively Russian repository-level agentic coding tasks; 4 product agent configurations, 3 runs each. Data, full trajectories and harness: this https URL

- **Prior Approaches**: 기존 SWE-bench 계열은 실제 GitHub 이슈 기반의 저장소 에이전트 평가를 표준화했지만, 과제 문장이 영어 이슈 텍스트로 설계되는 경우가 대부분이다. multilingual/다중언어 변형도 저장소의 주요 언어가 영어인 조건을 전제로 해 ‘비영어 네이티브 과제’라는 능력을 직접 측정하기 어렵다. 비영어 자연어가 포함된 벤치마크는 주로 스니펫·단일 파일 수준이라, 실제 저장소 작업과 실행 테스트를 동시에 요구하는 에이전트 설정과는 거리가 있다.

- **Core Contribution**: RuBench 1.0은 5개 인기 오픈소스 저장소에서 추출한 25개 수정 커밋을 기반으로, 과제 설명을 러시아어로 ‘번역 없이’ 처음부터 고객 요청 스타일로 작성한 저장소 레벨 에이전트 코딩 벤치마크를 제안한다. 채점은 유지보수자 회귀 테스트(오라클)로 하되 테스트 파일은 공개하지 않아, 평가가 사후 추론·자기완성에 덜 의존하도록 설계했다. 또한 평가 단위를 “모델만”이 아니라 CLI 에이전트 + model + reasoning effort까지 포함한 deployed product 구성으로 고정해 현실적인 작동 단위를 측정한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 비영어 고객 문장으로부터 의도·수용 기준을 복원하면서 (2) 훈련 오염과 (3) 런타임 누출(웹에서 fix를 확인하는 경우)까지 통제해 공정하게 비교하는 것이다. RuBench는 모든 fix 커밋을 모델 훈련 cutoffs 이후로 제한하는 freshness gate를 과제 import 단계에서 강제하고, 작업별 날짜·메타데이터·전체 에이전트 trajectory와 diff를 공개하며 오라클은 SHA-256 매니페스트로만 커밋해 검증 가능성을 유지한다. 추가로 trajectory 감사로 oracle escape와 ‘요청 모델 vs 실제 실행 모델’ 불일치를 걸러내어, 서버 측 안전장치가 모델을 조용히 바꾸는 상황까지 측정값의 유효성을 좌우하도록 관리한다.

- **Empirical Impact**: 평가 결과 최강 구성(Claude Code + Opus 4.8)은 RuBench 1.0 과제의 78.7%를 해결하며, 약한 구성과의 격차는 N=25에서 해상도 한계 내에서 통계적으로 분리됨을 명시한다. 더 중요한 발견으로, hors-concours 설정(Claude Code + Fable 5)을 trajectory 감사한 결과 25개 중 20%에서 제품이 ‘공식 safeguard fallback’으로 Opus 4.8를 자동 대체했고, 이를 통해 실제 측정 단위가 모델이 아니라 deployed product 레이어임을 재현 가능한 증거로 보여준다. RuBench는 과제 문장(러시아어)·기계적 신선도·비공개 오라클·전 과정 로그를 결합해, 향후 에이전트 벤치마크가 비영어 네이티브 과제 능력과 제품 동작(라우팅/폴백)까지 함께 검증해야 함을 강하게 시사한다.



### What Images Cannot Say: Language-Guided Olfactory Representation Learning (https://arxiv.org/abs/2607.06402)
Comments:
          ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 연구는 이미지와 전자코(electronic-nose) 측정을 데이터셋으로 함께 제공해도, 픽셀에는 직접 드러나지 않는 맥락적 환경 요인이 후각을 좌우해 정렬이 어렵다는 한계가 있었다. 또한 시각만으로 후각 표현을 학습하거나 단일 모달에 의존하는 baseline은 냄새의 구성 요소(물체 vs 환경)를 분리해 학습하기에 부족했다.

- **Core Contribution**: 본 논문은 vision과 olfaction을 잇는 의미적 브리지로 언어 가이던스를 쓰는 SCENT를 제안한다. Vision-Language Models(VLMs)가 장면의 객체·환경 맥락과 그에 상응할 법한 ambient smell 단서를 생성하고, 이를 기반으로 전자코 신호를 공유 임베딩 공간에 정렬하며 object-specific odor와 contextual environmental 기여를 languageguided latent decomposition로 분리한다.

- **Technical Challenges**: 핵심 어려움은 시각에서 보이지 않는 맥락적 요인이 냄새에 강하게 작용하는데도 전자코 신호를 이미지와 같은 의미 축으로 맞춰야 한다는 점이다. SCENT는 VLM의 장면 디스크립터를 semantic guidance로 활용해 후각 학습의 기준을 만들고, latent decomposition을 통해 혼합 냄새에서 객체 관련 향과 환경 관련 향을 분해하도록 학습을 설계했다.

- **Empirical Impact**: New York Smells 데이터셋에서 SCENT는 smell-to-image 및 smell-to-text 검색 과제에서 vision-only baseline 대비 유의미하게 성능을 개선하며 state-of-the-art를 달성했다. 또한 해석 가능한 후각 표현을 제공해 복잡한 냄새 혼합의 disentanglement이 가능함을 보여주며, 맥락적 의미 정보가 멀티모달 후각 정위(grounding)에 중요하다는 방향성을 제시한다.



### Responsible Personalisation: The Double-Edged Sword of Personalisation in Human-Robot Interaction (https://arxiv.org/abs/2607.06344)
Comments:
          36 pages, 3 figures

- **Prior Approaches**: 기존 HRI 연구에서 personalisation은 성과(참여도, 과업효율, 신뢰·협업) 중심으로 다뤄져 왔지만, responsible personalisation 관련 윤리 리스크는 맥락별로 단편적으로만 보고되어 왔다. 또한 HCI에서 agency 상실, 조작, 고정관념, 윤리적 설계 등이 논의돼 왔으나, 로봇의 embodiment와 사회적 존재감이 리스크를 어떻게 증폭·재구성하는지는 HRI에서 체계적으로 정리되지 않았다. 일부 사회로봇 감시/조작 비판 연구가 존재하지만, 본 논문은 이를 설계·운영 관점의 라이프사이클 리스크 분석과 연결해 더 구조화한다는 입장이다.

- **Core Contribution**: 본 논문은 embodiment-aware 관점을 토대로, personalised HRI의 personalisation 과정을 6단계(설계-데이터수집-모델링-상호작용-평가-종료)로 보고 interaction 맥락(단기/장기, open/closed domain)과 결합해 리스크가 ‘어떻게 생기고 시간이 지나며 어떻게 변하는지’를 분석하는 프레임워크를 제시한다. 동시에 주요 윤리 리스크(자율성 침식, 편향된 user modelling, manipulation, 탈인간화, 프라이버시 침해)를 통합적으로 정리하고, 이를 설계 권고와 오픈 연구과제로 번역한다. 핵심은 personalised robot behaviour의 설계 공간과 risk landscape를 한 체계 안에서 구조화해, 더 투명하고 윤리적으로 근거 있는 접근의 기반을 제공하는 것이다.

- **Technical Challenges**: 가장 큰 기술적 과제는 로봇의 embodiment가 개인화 효과를 높이는 동시에 주의환기(salience)와 agency attribution을 강화해 영향력·오해·해악 가능성도 같이 키운다는 점을, 라이프사이클의 각 단계에서 추적 가능한 형태로 다루는 것이다. 이를 위해 논문은 customisation(사용자 주도·명시적), adaptation(상황/집단 단서 기반), personalisation(특정 개인의 persistent user model을 통한 시스템 주도)을 연속선상에서 구분해 통제·추론·범위 차이를 정리하고, 입력-모델링-출력(IMO) 흐름으로 리스크가 고착되는 지점을 설명한다. 특히 데이터 수집(멀티모달·암묵 입력 확대)과 모델링(편향·cold-start에 따른 스테레오타입), 상호작용(센서모터 출력의 사회적 파급), 평가·종료(자기평가와 memory/데이터 관리)에서 리스크가 단계적으로 발생·지속될 수 있음을 구조화해 해결 경로를 제안한다.

- **Empirical Impact**: 본 초록/서술 범위에서는 주로 프레임워크와 분석에 초점이 있으며, 실험적 성능 수치의 직접 보고보다는 워크숍 논의와 기존 연구 근거를 통합해 리스크 지도를 만든다는 성격이 강하다. 다만 embodiment가 인지·학습 성과 및 신뢰/지속 사용 의사에 영향을 줄 수 있다는 기존 결과들을 인용해, 개인화가 실제로 ‘더 설득력 있고’ 사회적으로 더 강한 효과를 가질 수 있음을 경험적으로 뒷받침한다. 결과적으로 이 프레임워크는 HRI 커뮤니티가 personalisation을 단순 기능 향상으로만 보지 않고, 맥락·단계·책임을 함께 설계·평가하도록 방향을 제시하는 기반이 될 것으로 기대된다.



### Harnessing Code Agents for Automatic Software Verification (https://arxiv.org/abs/2607.06341)
- **Prior Approaches**: 기존 연구는 LLM이 Coq/ITP에서 따를 “사람이 설계한” 고정 전략을 강제해 왔습니다. 예를 들어 한 번에 한 단계씩 tactic을 예측하거나, divide-and-conquer로 목표를 쪼개는 방식이었지만 적용 가능한 정리 범위가 제한적이었고 전체 커버리지는 대략 12%~48% 수준에 머물렀습니다. 또 핵심 난점인 동시성과 shared mutable memory를 다루는 separation logic(특히 Iris)에서는 성능이 검증되지 못했습니다.

- **Core Contribution**: 이 논문은 고정된 proof strategy를 붙이는 방식이 불필요하다고 주장하며, “전체 lemma를 일반 LLM code agent가 자율적으로 증명”하도록 바꾸었습니다. Claude Code 같은 범용 코드 에이전트를 verification harness로 감싸, Coq 커널이 수락/거부하는 피드백을 기반으로 에이전트가 스스로 수정·재시도하게 만들었습니다. 그 결과 대상 정리 전부를 실패 없이(full coverage) 증명하는 것을 목표로 하며, Coq 전문가 개입 없이도 달성했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 잘못된 증명이 섞일 때 LLM의 “그럴듯하지만 틀린” 출력을 걸러내야 하고, (2) 증명 과정이 멈추지 않는(비종료) tactic 루프를 방지해야 하며, (3) proof가 누락되거나(예: silently dropped, Admitted/ admit) 범위를 속이면 안 된다는 점입니다. Aria는 Coq kernel 검증을 신뢰 앵커로 삼고, 시간 제한/비종료 탐지/Admitted 금지/coverage·Iris 스타일 검사 등을 harness에서 하드 제약으로 걸어 에이전트의 후보가 반드시 “검증적으로만” 통과되게 했습니다. 또한 실패 시 커널이 반환한 정확한 오류 위치와 미해결 목표를 다음 시도 프롬프트에 주입해, guided search처럼 수정이 수렴하도록 설계했습니다.

- **Empirical Impact**: 실험에서 Aria는 Iris의 4개 core 모듈 4,257개 lemma와, 이를 기반으로 한 RustBelt 계열 Rust 표준 라이브러리 217개 lemma를 모두 완전 자동으로 증명했습니다. 기존 LLM provers가 reglang에서 간신히 8분의 1 수준을 넘기기 어려웠던 것과 달리, Aria는 reglang 318개 전체를 증명했고 iris-lean(Lean 4 포트 진행 중)에서도 72개 미이식 lemma를 추가로 성공했습니다. Iris급 동시성 separation logic에서도 “전략 강제 없이” 범용 code agent+검증 harness 조합이 state-of-the-art 자동 증명을 달성했다는 점에서, verified software 개발 자동화의 실질적 확장 가능성을 보여줍니다.



### Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs (https://arxiv.org/abs/2607.06327)
- **Prior Approaches**: 기존 LLM 불확실성 추정(UE) 평가는 주로 영어에 집중돼 다국어(특히 저자원 언어)에서 성능이 유지되는지 근거가 부족했다. 또한 LLM-as-a-judge, BERTScore, n-gram overlap 같은 대체 지표는 잡음을 만들 수 있고, 언어별 편향도 UE 비교를 왜곡할 수 있다. 미니멀한 단답형 설정은 긴 생성 과정에서의 불확실성 신호를 충분히 보지 못했다.

- **Core Contribution**: 본 논문은 22개 언어(고·중·저자원)를 대상으로 9가지 UE 방법을 대규모로 평가한 최초의 비교 연구를 제시한다. 두 개의 사람 검수 MCQA 데이터셋에서 정답 라벨의 고정된 선택지 기반으로 정답성은 exact matching으로 유지하고, 불확실성은 추론(긴 reasoning) 텍스트에서만 추출해 모델 기반 프록시 없이 AUROC를 측정한다.

- **Technical Challenges**: 주요 기술적 난제는 언어·생성 길이가 바뀌어도 신뢰 가능한 UE 비교가 되도록 ‘정답성 근거’를 흔들리지 않게 만드는 것이었다. 저자들은 긴 생성(reasoning 약 150단어)을 유도하되, LLM-as-a-judge 및 임베딩 기반 스코어를 배제하고 MCQA 라벨을 기준으로 AUROC를 계산하도록 평가 프레임을 설계했다. 더불어 생성 언어(질문/추론 언어 분리)와 모델 스케일, cross-lingual answer 옵션이 UE 신호에 미치는 영향을 체계적으로 실험했다.

- **Empirical Impact**: 실험 결과, UE 성능은 언어 자체보다 ‘추론(reasoning) 생성 언어’에 크게 좌우됐고 영어로 추론을 유도하면 저자원 언어의 AUROC 격차가 크게 해소됐다. 또한 UE 방법 선택은 모델 스케일에 따라 달라지며, 작은 모델에서는 Token Entropy 같은 open-box 확률 기반이 유리하고 큰 모델에서는 Self Verbalized(닫힌 상자) 우위가 뚜렷해졌다. 마지막으로 selective prediction을 위한 임계값(threshold) 보정에서 영어만으로 보정하는 방식도 의미 있는 오차 감소를 제공하지만, 언어별 보정은 에러 탐지 성능을 더 끌어올려 다국어 신뢰성 배치 가이드라인을 제공한다.



### Token-Based Dual-view Fusion and Adaptation of Large Vision Models for Breast Cancer Classification (https://arxiv.org/abs/2607.06309)
- **Prior Approaches**: 유방촬영(mammography) CC와 MLO 두 시점을 통합하려는 시도는 주로 early/intermediate/late fusion, 또는 cross-attention 기반 상호작용으로 발전해 왔다. 다만 대부분은 단일 레이어에서의 결합이나 residual 형태의 직접 더하기로 구현돼, 뷰-특이 정보와 뷰-공유 정보가 얽히거나 cross-view 의존성이 깊은 층까지 일관되게 유지되기 어렵다는 한계가 있었다. 또한 prompt 기반 적응은 주로 단일 이미지에 맞춰져 다중 뷰 간 구조적 상호작용을 충분히 담지 못했다는 지적이 있었다.

- **Core Contribution**: 이 논문은 frozen vision transformer 백본 위에서 “token-centric dual-view learning”을 제안하며, CC–MLO 사이 상호작용을 fusion token의 구조적 토큰-커뮤니케이션으로 재구성한다. Stage 1에서는 deep shared prompt learning으로 CC와 MLO에 동일한 프롬프트를 적용해 파라미터 효율적으로 표현을 정렬하고, Stage 2에서는 bidirectional cross-attention로 생성된 fusion token을 시퀀스에 삽입해 다음 레이어에서 다시 정교화되도록 한다. 더 나아가 fusion을 단일 레이어가 아니라 여러 transformer depth에 반복 삽입해 계층적(계단식)으로 보완 정보를 전파한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 다중 뷰 의존성을 뷰-특이 표현과 분리해 명시적으로 지속시키고, (2) 단순 feature-level 결합이 아니라 토큰 수준에서 재사용 가능한 중간 전달체를 만드는 것이다. 이를 위해 cross-attention 출력에서 mean pooling으로 compact fusion token을 만들고, 이를 각 뷰의 토큰 시퀀스에 삽입한 뒤 후속 transformer 레이어가 이를 컨텍스트로 처리하게 설계했다. 또한 cross-view fusion 모듈을 여러 깊이에 배치해 상호작용이 계층적으로 누적되도록 했으며, 학습은 frozen 백본을 유지한 채 프롬프트(1단계)와 fusion 모듈(2단계)만 업데이트하는 방식으로 파라미터 효율을 확보했다.

- **Empirical Impact**: VinDr-Mammo와 CMMD 실험에서 제안 프레임워크는 linear probing, prompt-only adaptation, 기존 conventional fusion baseline 대비 일관된 성능 향상을 보였다. VinDr-Mammo BI-RADS 분류에서 F1-score 50.40%, AUC 0.8090을 달성했고, binary 설정에서 dual-view fusion baseline 대비 AUC가 0.10p 개선됐다. ablation에서도 token 기반 fusion과 multi-depth 상호작용 설계가 효과적임이 확인돼, CC–MLO 보완 정보를 더 잘 활용하는 다중 뷰 분류 접근으로 의미가 있다.



### UI2App: Benchmarking Visual Interaction Inference in Executable Web Application Generation (https://arxiv.org/abs/2607.06306)
- **Prior Approaches**: 기존 웹 UI 생성 연구는 스크린샷이나 텍스트를 입력으로 받아 시각적 유사도(visual fidelity)에 주로 초점을 맞춰왔다. 그러나 이런 평가는 동작 가능성이나 페이지 간 상태 동기화 같은 상호작용이 실제로 구현되는지까지는 검증하지 못해 ‘그럴듯한 가짜 UI’ 문제가 남았다. 또 텍스트 기반 방식은 복잡한 프롬프트 의존성과 레이아웃/시각적 일관성, 교차 페이지 상호작용을 자연어로 정밀 지정하기 어려운 한계를 가진다.

- **Core Contribution**: 이 논문은 스크린샷만으로 애플리케이션의 행동을 복원하는 ‘interaction inference’를 측정하는 최초의 벤치마크 UI2App을 제안한다. 327개의 스크린샷을 45개의 state-coherent 세트(실행 가능한 멀티 라우트 웹앱)로 구성하고, 모델이 텍스트나 행동 지시 없이도 실행 가능한 코드를 생성하도록 과제를 정의한다.

- **Technical Challenges**: 문제의 핵심 기술 난점은 정적 스크린샷이 행동을 충분히 한정하지 못해, 같은 화면 상태가 서로 다른 올바른 구현을 가질 수 있다는 점이다. 이를 해결하기 위해 4개 평가 축(실행 가능성, 내비게이션 도달성, 시각적 충실도, IIS) 중 IIS를 7개 상호작용 카테고리와 상태-복잡도(scope)로 세분화해, 기능적 정합성과 상태 관리 복잡도를 루브릭 기반으로 채점한다. 또한 실행 실패는 0으로 처리해 지표 비교의 기준선을 유지하고, 경로별 URL 로딩으로 페이지 구현과 라우팅을 분리 평가한다.

- **Empirical Impact**: 6종 frontier VLM을 실험한 결과, 시각적 충실도 리더가 IIS에서도 상위에 오르지 못하는 ‘역상관’이 뚜렷했으며 VFS 리더는 IIS에서 7.5점 수준에 그쳐 IIS 리더 대비 5.2배 뒤처졌다. 특히 교차 라우트 상태(cross-route persistence, S3 scope)는 전 모델에서 병목으로 나타났고 절반이 해당 차원에서 정확히 0점을 기록했으며 최고 성능도 S3에서 21.6점에 머물렀다. 저자들은 UI2App이 스크린샷-기반 상호작용 복원과 구현의 “S3 headroom”을 줄이기 위한 연구 테스트베드가 될 수 있음을 강조한다.



### Designing Maintainable Hybrid Generative Systems: A Quantum-Inspired Approach to Automated Music Harmony Generation (https://arxiv.org/abs/2607.06296)
Comments:
          12 pages, 1 figure, 4 tables. Extended version of the 4-page paper accepted at the 34th International Conference on Information Systems Development (ISD2026, Prague). Source code and dataset available at this https URL

- **Prior Approaches**: 기존의 멜로디 기반 자동 화성 생성은 대부분 순수 생성 모델(또는 데이터 기반 학습) 중심으로, 유연성은 높지만 음성적·화성적 구조를 일관되게 통제하기가 어렵다는 한계가 지적돼 왔다. 반면 규칙 기반 접근은 구조 통제는 강점이나, 가능한 화성 전개(다양성)와 후보 탐색의 폭이 좁아질 수 있다. 또한 평가 역시 일관된 기준과 재현 가능한 지표가 부족해, “좋은 화성”을 객관적으로 비교하기 어렵다는 문제가 있었다.

- **Core Contribution**: 이 논문은 멜로디로부터 자동 음악 화성을 생성하되, 유지보수 가능한 hybrid generative architecture를 설계해 생성 유연성과 구조적 제어를 동시에 노린다. 핵심은 양자적 영감의 candidate exploration(겹치는 멜로디 문맥을 고려)으로 다양한 후보를 찾고, 이어서 rule-based optimization 계층으로 조화로운 구조와 규범적 행동(예: 종지(cadence) 성격)을 맞추는 방식이다. 특히 최적화 계층은 학습 코퍼스 없이도 구조 일관성과 예측 가능성을 높이는 것을 강조한다.

- **Technical Challenges**: 가장 큰 기술적 도전은, 생성적 탐색으로 생길 수 있는 구조 붕괴와 규칙 최적화의 과도한 제한 사이의 균형을 맞추는 것이다. 논문은 겹치는 멜로디 컨텍스트를 사용한 candidate exploration으로 탐색 공간을 확장하고, 그 결과를 구조·기능·화성 유사도 제약을 포함하는 명시적 규칙 최적화로 정렬해 안정성을 확보한다. 또한 정보시스템 개발 관점에서 투명하고 조절 가능한 설계를 위해, 최적화 계층이 출력 예측성과 안정성에 어떻게 기여하는지 평가 지표를 명확히 구성한다.

- **Empirical Impact**: 실험에서는 structural coherence, functional agreement, harmonic similarity, robustness 같은 명시적·재현 가능한 지표로 성능을 검증했다. 결과적으로 제안 방식은 조성(tonal) 구조와 종지의 행동을 보존하면서도 여러 개의 유효한 화성 실현을 허용하는 것으로 나타났다. 또한 rule-based optimization 계층이 학습 없이도 구조 일관성·안정성·예측 가능성을 개선해, controllable hybrid generative 시스템을 체계적으로 설계·평가할 수 있음을 보여준다.



### VendorBench-100: A Unified Cross-Paradigm Benchmark for Deepfake Image Detection (https://arxiv.org/abs/2607.06254)
Comments:
          22 pages, 10 figures, 3 tables. Code and data: this https URL

- **Prior Approaches**: 딥페이크/AI 이미지 탐지는 상용 API, zero-shot 비전-언어모델(vision-language models), 오픈소스 탐지기라는 세 갈래로 발전했지만, 서로 같은 기준에서 비교된 적이 거의 없습니다. 기존 평가는 벤더 내부 테스트나 단일 유형 데이터에 치우쳐, 실제 운영에서 마주치는 다양한 위조 방식과 품질 열화(compression/리사이즈)를 공정하게 반영하지 못한다는 한계가 있었습니다. 또한 정확도 같은 단일 지표는 데이터 불균형(예: 대부분이 fake)에 취약해 과대평가될 수 있습니다.

- **Core Contribution**: 본 논문은 VendorBench-100을 제안하며, 서로 다른 3패러다임의 36개 모델을 단일 100장(79 fake/21 real) 고정 코퍼스와 단일 출력 스키마, 공통 평가 프레임워크로 함께 비교합니다. 특히 난이도를 키우기 위해 8개 엣지케이스 패밀리(얼굴 스왑 스미어, near-duplicate 스왑, letterboxed text-to-video 스틸, AI 사진 편집, 불명확 출처 조작 등)를 체계적으로 구성해 “쉽게 구분되는” 상황을 피했습니다. 순위는 Matthews correlation coefficient(MCC)를 중심으로 매기고, 임계값에 덜 민감한 ROC-AUC를 보조로 사용합니다.

- **Technical Challenges**: 주요 기술적 난제는 모델마다 출력 형식/판정 방식이 달라 공정 비교가 어려운 점이었습니다. 이를 위해 모든 모델을 FAKE/REAL의 hard label, P(fake)∈[0,1] 형태의 confidence, 판정 성공 여부(abstention 포함)로 정규화하는 공통 레코드를 만들고, 파일명/메타데이터 유출을 막는 anti-leakage 프로토콜을 적용했습니다. 또한 클래스 불균형에서 accuracy가 쉽게 “기만”될 수 있어, MCC와 ROC-AUC를 함께 보고 score 분리력과 임계값 의사결정 품질의 차이를 진단하도록 설계했습니다.

- **Empirical Impact**: 실험 결과로는 상용 API들이 중앙값 성능에서 가장 강했지만, 일부 오픈소스 모델은 최고 수준의 비전 LLM과 경쟁하거나 더 잘한 사례도 나타났습니다. 더 중요한 관찰로, ROC-AUC(랭킹/분리력)가 높아도 MCC(기본 임계값에서의 신뢰 가능한 판정)가 낮게 나오는 불일치가 세 패러다임 전반에서 반복되었습니다. 즉 “점수 순위가 좋다”는 것만으로 “기본 설정에서 실제로 믿을 만한 판정”이 보장되지 않으며, 이 메트릭 불일치 자체가 벤치마크의 핵심 발견으로 제시됩니다. 논문은 재현을 위해 평가 프레임워크와 결과를 공개해 후속 연구가 동일 프로토콜로 확장할 수 있게 했습니다.



### Spider 2.0-AIFunc: Extending Real-World Text-to-SQL to AI-Native SQL Workflows (https://arxiv.org/abs/2607.06229)
Comments:
          24 pages, 3 figures, 7 tables

- **Prior Approaches**: 기존 text-to-SQL 벤치마크(예: Spider 2.0, BEAVER)는 전통적인 SQL 연산자 조합만을 평가해 왔고, Snowflake Cortex AI functions처럼 LLM을 호출하는 AI-native SQL 생성은 검증하지 못했습니다. 또한 실행 정확도(execution accuracy)는 같은 SQL이 일관된 결과를 내야 하는데, AI functions는 temperature=0이어도 소폭 출력 불일치가 발생할 수 있어 벤치마크 설계가 까다롭습니다.

- **Core Contribution**: 이 논문은 AI-native SQL을 직접 평가하는 Spider 2.0-AIFunc 벤치마크를 제안합니다. Snowflake Cortex AI functions 6종(AI_CLASSIFY, AI_FILTER, AI_SENTIMENT, AI_SIMILARITY, AI_EXTRACT, AI_AGG)을 포함해 총 465개 검증 인스턴스를 구성하고, 기존 Spider2-Snow의 소스 작업을 에이전트 기반으로 AI-native 형태로 변환했습니다.

- **Technical Challenges**: 핵심 난제는 (1) AI function 선택과 파라미터(예: 분류 라벨, 추출 스키마)를 지시문 수준에서 모호하지 않게 고정하는 것과 (2) AI function 호출로 인한 실행 결과 변동성을 통제하는 것입니다. 이를 위해 에이전트가 SQL과 자연어 지시문을 함께 수정하며 파라미터 누락/오류를 해결하고, 다중 패스 반복 실행 및 시간 창(time window) 분리 검증을 통해 안정적인 인스턴스만 공개했습니다.

- **Empirical Impact**: 10개 SOTA 언어모델을 평가한 결과, 폐쇄형 모델은 67–70%대 실행 정확도를 보인 반면 오픈소스 최고 성능은 58.1%로 격차가 나타났습니다(주된 원인: 술어/조건 지정, 스키마 grounding, AI function 파라미터화 오류). 또한 전통 text-to-SQL용 에이전트(스키마 검색, 관련 테이블 선택 중심)는 AI-native SQL에서는 최소 Spider-Agent 대비 큰 이점을 주지 못했는데, 이는 AI function 내부의 미세한 의미 선택이 정답을 좌우하기 때문으로 해석됩니다.



### UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods (https://arxiv.org/abs/2607.06202)
- **Prior Approaches**: 기존 MoE는 All-to-All 같은 희소 라우팅 통신을 위해 Bulk Synchronous Parallel(BSP) 기반의 coarse-grained 오케스트레이션에 의존해, 통신 단계 간 실행이 엄격히 직렬화되는 문제가 있었다. 또한 동기화 오버헤드가 고대역폭 슈퍼팟 환경에서도 병목이 남아 확장성이 떨어졌고, 토큰 트래픽이 불규칙한데도 거리 정보를 무시한 스케줄링으로 로드 밸런스가 크게 흔들렸다.

- **Core Contribution**: 이 논문은 NVL72/576, CloudMatrix384 같은 production high-bandwidth superpods에서 MoE 통신의 핵심 병목이 단순 대역폭이 아니라 BSP 직렬화, 동기화 오버헤드, 로드 불균형에 있음을 정리한다. 이를 해결하기 위해 UBEP(Unified-Bus Expert Parallelism)라는 통신 라이브러리로 MoE의 All-to-All 프리미티브를 해당 아키텍처 관점에서 재설계한다.

- **Technical Challenges**: UBEP은 interdependent communication phases가 만드는 실행 직렬화와, 고대역폭 환경에서 커지는 synchronization overhead를 줄여야 했다. 동시에 거리-비무관 스케줄링으로 인한 load imbalance를 완화하기 위해, irregular token traffic를 더 잘 반영하는 All-to-All 통신 구조를 목표로 설계하고 대규모 실험에서 이를 검증했다.

- **Empirical Impact**: 대규모 실험에서 UBEP은 All-to-All latency를 최대 52.4% 줄였고, MoE의 TPOT(Time Per Output Token)을 최대 11.1% 개선했다. 이는 슈퍼팟에서 MoE를 ‘연결만 잘 되면 된다’가 아니라 오케스트레이션·동기화·스케줄링까지 포함한 통신 설계 이슈로 봐야 함을 보여주며, 실서비스 배치 효율을 직접 끌어올릴 의미가 있다.



### TriA Pipeline: A Large-Scale Automatic Audio Annotation Pipeline For Audio Classification In Specific Scenarios (https://arxiv.org/abs/2607.06179)
Comments:
          5 pages, 2 figures, 4 tables, accepted for publication in Interspeech 2026. The code is at: this https URL

- **Prior Approaches**: 기존 Audio Classification(AC) 데이터는 일반 목적용(AudioSet, FSD50K 등)과 특정 목적용(주방 Kitchen20, 가정 환경 DESED, 안전 모니터링 등)으로 나뉘는데, 후자는 대개 라벨된 데이터 규모가 작다는 한계가 있습니다. 또한 자동 라벨 파이프라인은 주로 speech 또는 paralinguistic 영역에 치우쳐 audio event 전반을 커버하기 어렵다는 문제가 있었습니다. 그 결과, 가정 환경처럼 특정 시나리오에서의 데이터 희소성이 성능과 확장성을 제약해왔습니다.

- **Core Contribution**: 논문은 Automatic Audio Annotation Pipeline–TriA(TriA Pipeline)를 제안해, 다양한 출처의 오디오를 audio event annotation이 포함된 학습 데이터로 효율 변환하는 방식을 제시합니다. TriA Pipeline으로 2130시간 이상, 431개 오디오 클래스의 TriA 데이터셋을 구축했으며, 시나리오 사전지식 기반 부분집합 TriAGK를 뽑아 가정용 3개 AC 과제에서 효과를 검증합니다. 수동 라벨만 학습한 경우 대비 TriAGK를 포함할 때 성능이 일관되게 개선됨을 보여줍니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 긴 영상/스트리밍 오디오를 적절한 구간으로 자르고, (2) audio activity detection(AAD)와 event detection(AED) 과정에서 생길 수 있는 누락·오탐을 줄이며, (3) 자동 라벨의 신뢰도를 높이는 데 있습니다. 논문은 ECT/SCT 기반의 구간 길이 제약을 도입해 AAD 필터링을 정교화하고, BEATs 기반 AED를 로컬/글로벌 윈도잉으로 확장한 뒤, audiobox-aesthetics의 PC/PQ와 CLAP similarity로 품질이 낮은 구간을 제거합니다. 그 결과 주관 평가 포함 평균 93.67%의 annotation accuracy를 보고하며, RTF 0.03 수준에서 파이프라인 실행 가능성도 확인합니다.

- **Empirical Impact**: TriAGK를 DESEDAC, Kitchen20, Nonspeech7k(가정 일반, 주방, 안전 모니터링) 세 과제에 적용해, 수동 라벨만 사용한 학습과 비교 평가를 수행했습니다. 특히 TriAGK만 파인튜닝해도 수동 라벨 학습과 비슷한 수준의 성능을 보이고, sequential fine-tuning으로 TriAGK+수동 라벨을 함께 쓰면 평균적으로 정확도 3.97%, Macro-F1 3.35%의 상대 개선을 달성했습니다. 이는 라벨 희소 시나리오에서 자동 파이프라인 기반 사전지식 데이터가 성능을 끌어올릴 수 있음을 실증하며, TriA 데이터셋과 코드가 공개되어 재현성과 확장성 측면의 의미도 큽니다.



### Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design (https://arxiv.org/abs/2607.06175)
Comments:
          21 pages, 5 figures

- **Prior Approaches**: LLM로 BPMN을 생성할 때, SFT는 학습 데이터에 있던 패턴을 재현하는 데는 강하지만 결과 품질을 다차원 기준(문법·의미·유용성)에 맞춰 상향 최적화하기 어렵다. 기존 RL 기반 접근은 외부 평가를 붙여 성능을 올리긴 했지만, 보상 함수를 어떤 방식으로 구성(차원 가중, 무효 페널티 등)해야 하는지에 대한 체계적 비교는 부족했다. 또한 다차원 품질을 한 개의 스칼라 보상으로 단순 합산하는 경우가 많아, 차원이 서로 충돌할 때 학습이 어떻게 흔들리는지 불명확했다.

- **Core Contribution**: 이 논문은 RL 기반 BPMN 프로세스 모델 생성에서 보상 함수 설계(다차원 점수의 가중·무효 페널티·집계)를 체계적으로 실험한다. Llama 3.1 8B와 Qwen 2.5 14B를 대상으로 48개 설정을 만들고, Group Sequence Policy Optimization(GSPO)로 BEF4LLM의 38개 자동 메트릭(구문/실용/의미 품질)을 기반 보상을 최적화한다. 특히 “보상 구성 자체가 최적화 결과를 좌우하며, RL 적용 여부만큼 큰 영향도 낼 수 있다”는 점을 실증한다.

- **Technical Challenges**: 핵심 난제는 여러 품질 차원을 하나의 학습 신호로 합칠 때, 특정 차원을 더 중요하게 두면 실제로 그 차원이 개선되는지(또는 모드 붕괴가 나는지)였다. 연구진은 BEF4LLM에서 생성물을 먼저 validity로 걸러낸 뒤, 구문·실용·의미 점수를 보상으로 조합하며, 차원 가중(동등 vs 표적)과 invalidity penalty(음수 페널티 vs 0) 등 보상 축을 분리해 비교했다. 또한 GSPO의 그룹 내 상대 비교 특성에 따라 보상 스케일 문제를 줄이도록 설계된 학습을 적용했고, SFT 초기화 유무가 아키텍처별로 어떻게 상호작용하는지도 함께 확인했다.

- **Empirical Impact**: 실험 결과, GSPO는 두 모델 모두에서 실용성과 구문 품질을 유의미하게 끌어올리면서 의미 충실도는 대체로 보존했으며, 출력 변동성은 6배 이상 줄였다. 특히 직관과 달리 equal reward weighting이 표적 가중보다 일관되게 우수했고, 특정 차원 강조는 그 차원 개선 실패뿐 아니라 저품질 모드로 붕괴시킬 수 있었다. 또한 invalidity penalty와 SFT 초기화의 효과는 모델 아키텍처에 의존적이어서, 단순한 기본값 튜닝이 아니라 경험적 검증이 필요함을 보여주며 다차원 자동 평가가 가능한 구조화 생성 전반에 일반화될 수 있음을 시사한다.



### X-FEMR: A Token-level Explainable Approach for Electronic Health Records Foundation Models using Transformer-based Models (https://arxiv.org/abs/2607.06163)
Comments:
          Accepted by IJCAI-ECAI 2026 AI and Health Track

- **Prior Approaches**: 기존 FEMR 연구는 성능 향상과 task-specific fine-tuning 중심으로 발전했지만, 모델 내부 추론이 불투명해 임상적 신뢰 확보가 어렵다는 문제가 남아 있었다. XAI는 SHAP, saliency, LIME, surrogate 모델 등을 활용해 왔으나, FEMR처럼 long-context에 토큰 단위로 autoregressive 예측을 수행하는 구조에는 그대로 적용하기 힘들다. 특히 설명이 임상적으로 타당한지 정량 평가하는 지표도 부족했다.

- **Core Contribution**: 이 논문은 FEMR을 대상으로 한 최초의 token-level explainability 방법을 제안한다. Transformer 기반 surrogate 모델을 FEMR의 input-output 동작에 맞춰 학습해, 토큰(이벤트/값)의 중요도를 뽑아 FEMR이 환자 병력의 어떤 부분을 근거로 예측하는지 드러낸다. 또한 surrogate가 찾아낸 핵심 토큰이 임상적으로 검증된 특징과 얼마나 맞닿는지 측정하는 clinical alignment metric을 새로 제안한다.

- **Technical Challenges**: 핵심 어려움은 (1) EHR이 불규칙·장기이며 다중 모달/잡음이 많아 단순 surrogate가 temporal dynamics를 충분히 모사하기 어렵다는 점, (2) 토큰 수준 설명을 뽑아도 그 설명이 임상 지식과 어긋날 수 있다는 점, (3) soft/hard supervision에 따라 토큰 중요도 패턴이 달라질 수 있다는 점이다. 이를 위해 surrogate를 event-level Transformer로 구성하고 time-to-prediction 구간을 시간 특징으로 넣어 시간 의존성을 학습하게 했으며, FEMR의 hard label(이진 예측)과 soft label(확률) 두 방식으로 학습해 설명 안정성과 불일치를 비교했다. 최종 설명 분석에는 성능·재현성이 더 나은 hard-label 학습 surrogate를 사용하고, SHAP로 토큰 기여도를 산출한 뒤 임상 정합 지표로 평가한다.

- **Empirical Impact**: EHRSHOT의 two task(LOS 7일 이상, ICU transfer)에서 CLMBR-T-Base를 기준으로 surrogate가 예측 동작을 가깝게 근사하는 것을 보였고, 설명도 임상적으로 검증된 특징과 잘 정렬된 결과를 제시한다. 특히 수치값(검사/측정)과 time delta가 중요 기여를 많이 하며, 상위 토큰에 heart rate, SBP/DBP, 체온, 호흡수, 산소포화도 같은 임상 핵심 변수들이 반복적으로 나타난다. clinical alignment metric 기준으로 validated events ratio가 유의미하게 산출되어, token-level 설명이 임상 지식과의 정합성을 갖춘 explainability 프레임워크로 활용될 수 있음을 시사한다.



### LongCrafter: Towards Diverse Long-Context Understanding via Evidence-Graph-Guided Instruction Synthesis (https://arxiv.org/abs/2607.06160)
- **Prior Approaches**: 긴 컨텍스트 SFT를 위해 합성 데이터로 학습을 강화하려는 시도가 있었지만, 기존 데이터는 체계적 태스크 분류가 없어 커버리지가 좁았습니다. 또한 문서에서 바로 질문을 만들다 보니 증거 구조(문단 간 의존)와 난이도 계층이 약해, 모델이 쉬운 지름길로 답을 맞히는 경향이 커졌습니다. 마지막으로 추론 단계마다 출처 증거에 고정하는 faithfulness supervision이 부족해, 문서 기반이 아닌 파라메트릭 지식을 섞는 비충실 추론 위험도 남아 있었습니다.

- **Core Contribution**: LongCrafter는 장문 SFT용 데이터를 ‘구조화된 합성’으로 다루며, 계층적 task taxonomy와 evidence-grounded 파이프라인을 결합해 위 세 한계를 동시에 겨냥합니다. 장문 이해를 local/shallow와 global/deep으로 나누고 32개의 fine-grained task type을 생성의 전역 prior로 사용해, 태스크 커버리지와 생성 난이도를 의도적으로 설계합니다. 무엇보다 instruction–response가 문서에서 위치가 특정된 evidence span에 엄격히 근거하도록 만들어, 추론이 추적 가능하고 충실하게 이어지도록 합니다.

- **Technical Challenges**: 핵심 과제는 (1) 증거가 문단 사이에서 체인·트리·그래프처럼 연결되는 의존성을 모델이 사용할 수 있는 형태로 데이터에 반영하는 것, (2) 이를 통해 난이도는 높이되 무작정 어렵게만 만들지 않고 task type에 맞춰 조절하는 것, (3) 응답이 원문 증거와 단계별로 일치하도록 보장하는 것이었습니다. LongCrafter는 문맥을 evidence graph로 분해하기 위해 Extract-then-Construct 절차를 쓰고, task-relevant span을 노드로 삼아 cross-paragraph dependency edge를 연결해 “어떤 증거가 왜 필요한지”를 그래프로 명시합니다. 이후 그래프에 조건부로 instruction을 만들고, 응답은 단계별로 관련 evidence를 verbatim citation 형식으로 인용하며, LLM 검증(유일 정답/맥락 기반)을 통과한 샘플만 남기는 방식으로 해결합니다.

- **Empirical Impact**: 실험에서 LongCrafter로 학습한 모델은 LongBench, LongBench v2, LooGLE에서 모든 SFT baseline을 상회했으며 Qwen2.5-7B와 LLaMA-3.1-8B 모두에서 특히 고난도 태스크에서 가장 큰 폭의 개선이 관찰되었습니다. 또한 동일 백본에서 LongCrafter 데이터가 기존 데이터보다 과제 다양성과 난이도 분포가 더 균형 있게 퍼져 있고, ‘lost in the middle’ 문제를 완화할 정도로 증거 위치에 대한 강건성(evidence localization robustness)이 높다고 분석했습니다. 데이터가 2,000개로 제한돼도 official post-trained 모델 대비 All-Overall이 개선되는 결과는, evidence 그래프 기반의 충실한 합성 데이터가 적은 스케일에서도 장문 이해 성능을 실질적으로 밀어올릴 수 있음을 시사합니다.



### LLM Agents for Deliberative Collaboration: A Study on Joint Decision Making Under Partial Observability (https://arxiv.org/abs/2607.06157)
Comments:
          Code is available at this https URL

- **Prior Approaches**: 기존 언어 에이전트 연구는 단일 에이전트의 추론·행동, 또는 협업/협상/협의 같은 다중 에이전트 대화를 주로 다뤘습니다. 하지만 다수 에이전트가 부분적이고 비대칭적인 정보를 가진 상태에서 ‘합의에 도달하기 위한 정보 교환·정렬·의사결정’을 체계적으로 평가한 벤치마크는 부족했습니다.

- **Core Contribution**: 이 논문은 deliberative collaboration(숙의 기반 협업)을 ‘부분관측 하 협동 joint decision-making’ 문제로 수식화해, 공유 보상 하에서 합의 도출을 평가 가능한 단일 추상화로 정리합니다. 또한 메뉴 디자인과 작업 배분을 포함해 관측 구조·역할 권한·평가 프로토콜을 달리하되 동일한 숙의 협업 골격을 유지하는 확장형 benchmark를 제안합니다.

- **Technical Challenges**: 핵심 난제는 (1) 비대칭 부분관측에서 필요한 정보를 충분히 교환·집계(belief alignment)하고 (2) 불확실성을 반영해 공동 결정을 내리며 (3) 계산·검증까지 수행하는 것입니다. 이를 위해 데이터베이스 기반 작업 생성, NR/VR 중심의 정량 평가, 그리고 선택적 외부 도구로 Solver(정수계획)·Calculator(제약 검증 및 피드백)를 제공해 ‘언어 숙의’와 ‘수학 처리’의 역할을 분리해 진단합니다.

- **Empirical Impact**: 180개 태스크에서 다양한 SOTA LLM을 평가한 결과, 숙의 협업은 여전히 최신 모델에도 정보 교환/집계·추론·수학에서 병목을 남겼습니다. 다만 진단 분석은 숙의가 성찰과 오류 수정의 기회를 주어 centralized baselines 대비 성능이 오르는 경우도 있음을 보여주며, tool 사용 역시 모델별로 효과가 달라 ‘정답 계산’만으로는 부족할 수 있음을 시사합니다.



### Property-Driven Synthetic Data Engineering for Data-Scarce Software Systems: Reflections from the Breast Cancer Domain (https://arxiv.org/abs/2607.06133)
Comments:
          5 pages

- **Prior Approaches**: 기존 연구들은 SDG를 데이터가 부족하거나 공유가 어려운 상황에서 “통계적 패턴을 비슷하게 재현하는 생성기”로 접근해 왔고, 주로 분포 유사도·상관 보존·다운스트림 성능 같은 사전 정의 지표로 품질을 점검합니다. 또한 의료 맥락에서는 SAFE 같은 검증 파이프라인을 제안하지만, 무엇이 “신뢰할 수 있음”을 구성하는지(이해관계자별 타당성 조건)를 요구공학 관점에서 체계화하는 데는 한계가 있다는 문제의식이 제시됩니다. 저자들은 특히 임상 사용에서는 통계적으로 그럴듯해도 임상적 타당성이 보장되지 않을 수 있음을 경험적으로 강조합니다.

- **Core Contribution**: 이 논문은 data-scarce, privacy-constrained 소프트웨어 공학에서 SDG를 단순 전처리/생성 기법이 아니라 “property-driven synthetic data engineering”이라는 엔지니어링 문제로 재정의합니다. 합성 데이터가 만족해야 하는 이해관계자별 validity properties를 요구 수집→정형화→체크→파이프라인 진화까지 다루는 것을 핵심 기여로 제시합니다. 또한 IORT(수술 중 방사선치료) 유스케이스를 통해 이러한 성질 기반 관점이 왜 필수인지 구체화합니다.

- **Technical Challenges**: 기여를 실현하는 기술적 난제는 (1) 어떤 속성이 보존돼야 하는지 요구 불확실성과 이해관계자 충돌, (2) ground truth가 부족한 상태에서의 검증 설계, (3) 데이터·병원·규제가 바뀌며 성질 조건이 지속적으로 업데이트돼야 한다는 파이프라인 진화 문제로 정리됩니다. 저자들은 IORT 데이터(초기 1000명/64변수)를 전처리·클리닝한 뒤 여러 tabular SDG 모델을 비교하며, 상관/분포 관점에서는 TVAE가 더 잘 맞을 수 있어도 임상 타당성은 별도의 도메인 검증(LRFS 기반 Kaplan-Meier, Cox 분석 등)이 필요함을 보입니다. 결론적으로 생성기 선택도 단일 ML 문제라기보다, 요구된 속성 체크와 트레이드오프를 만족시키는 “요구·검증 설계” 이슈라는 점을 드러냅니다.

- **Empirical Impact**: 사전 실험에서는 상관행렬 유사성과 생존 곡선(LRFS)의 전반적 경향이 합성 데이터에서도 비슷하게 나타나는 등 통계적 보존 가능성을 관찰했습니다. 다만 말기 구간처럼 표본이 적은 영역에서는 차이가 생길 수 있고, 이는 임상적으로 의미가 있을 수 있으므로 “통계적으로 그럴듯함=임상적 신뢰”로 단정할 수 없음을 경험적으로 확인했습니다. 이 결과는 SDG를 현업 소프트웨어 공학에서 신뢰 가능한 산출물로 만들기 위해, 성질 정의·충돌 탐지·검증 자동화·지속 모니터링 같은 연구 의제가 필요하다는 방향성을 제시합니다.



### Self-Supervised Implicit CEST Reconstruction via Physics-Informed Lorentz Encoding (https://arxiv.org/abs/2607.06132)
Comments:
          10 pages, 5 figures, Accepted by MICCAI 2026

- **Prior Approaches**: Multi-Pool CEST MRI는 대사 정보를 주지만 Z-spectra의 긴 획득 시간이 임상 적용을 막는다. sparse sampling으로 시간을 줄여도 고해상도 Z-spectra 복원은 ill-posed inverse problem이라서, 기존 보간이나 generic implicit neural representations(INRs)는 물리 제약이 약해 spectral artifacts와 비물리적 신호를 만들기 쉽다.

- **Core Contribution**: 이 논문은 Lorentz Encoding(LE)이라는 physics-informed 프레임워크로 CEST 재구성을 self-supervised 연속 좌표 학습 문제로 재정의한다. LE는 연속 스펙트럼 매핑을 물리적으로 제약된 공간에 투영하고, parametric Lorentzian profile의 조합과 learnable basis function으로 물리 일관성을 강제한다.

- **Technical Challenges**: 핵심 난점은 sparse한 샘플만으로 연속적인 고해상도 Z-spectra를 복원하면서도 물리적으로 타당한 형태를 유지하는 것인데, 기존 접근은 물리 제약이 부족해 잡음에 취약했다. LE는 물리 모델 기반의 Lorentzian 제약을 인코딩 단계에 반영하고, self-supervised 재구성 학습으로 스펙트럼-좌표 매핑을 노이즈에 덜 민감하게 정규화해 이 문제를 완화한다.

- **Empirical Impact**: in vivo 사람 뇌 데이터 실험에서 LE는 최신 방법을 뚜렷하게 능가하며, 39-point sampling 조건에서 PSNR 57.58 dB와 SSIM 0.9994를 달성했다. 또한 LE의 learned physics-informed encodings는 잠재공간에서 연속적이고 기하학적으로 정렬된 trajectory를 형성해 APT, NOE, MT 같은 정량 대사 mapping을 더 정확하게 만든다는 점에서 임상 속도-정확도 균형 개선에 의미가 있다.



### Evaluating Fine-Tuning and Metrics for Neural Decompilation of Dart AOT Binaries (https://arxiv.org/abs/2607.06125)
Comments:
          Under review at ACM Transactions on Software Engineering and Methodology (TOSEM)

- **Prior Approaches**: 기존 신경 디컴파일 연구는 주로 C/C++를 대상으로 하며, 복잡한 현대 언어(Dart AOT, Swift)에서의 체계적 평가는 부족했다. 또한 평가가 CodeBLEU·BLEU·컴파일 성공 같은 표면 지표 중심이어서, 단위 테스트 기반 기능 정합성(compile 후 실행/검증)을 충분히 다루지 못했다. fine-tuning이 도움이 되는지(혹은 해가 되는지) 모델 규모별로 검증한 연구도 드물었다.

- **Core Contribution**: 이 논문은 Dart Ahead-of-Time(AOT) 어셈블리→Dart 코드 디컴파일을 대상으로 fine-tuning 효과와 지표 타당성을 실증적으로 분석한다. 154개 테스트를 갖춘 HumanEval-Dart 벤치마크를 새로 제시하고, Dart용 CodeBLEU 구현 및 CodeBLEU·compile@k·pass@k의 상호 관계를 비교한다. 결론적으로 pass@k가 신경 디컴파일에서 핵심 평가 지표여야 한다는 점을 경험적으로 뒷받침한다.

- **Technical Challenges**: 핵심 난제는 (1) 작은 4–8B 모델에서 fine-tuning이 기능 정합성을 실제로 끌어올릴지, (2) 표면 유사도 지표가 pass@k를 대변하지 않을 때 어떻게 오판을 줄일지, (3) 언어·최적화 수준 차이로 인한 교란을 통제할지였다. 연구진은 LoRA+DoRA 기반의 parameter-efficient fine-tuning으로 6개 변형을 만들고, Dart+Synth(토큰 매칭, same-language)와 Dart+Swift(토큰 매칭, cross-lingual)의 실험 설계를 비교해 interference를 측정했다. 또한 과제 난이도(어셈블리 길이) 분석과 함께 pass@k의 통계적 유의성 검정을 paired task-level 방식으로 수행했다.

- **Empirical Impact**: 실험 결과 fine-tuning은 pass@k를 통계적으로 유의하게 개선하지 못했으며, 오히려 더 큰 모델(Qwen3-8B)은 일부 설정에서 유의미한 regression이 나타났다. cross-lingual interference는 4B에서만 유의하게 나타났다가 8B에서는 0과 구분되지 않아, scaling에 따라 부정적 전이가 줄어드는 양상이 관찰됐다. 특히 CodeBLEU와 compile@k는 개선될 수 있는데 pass@k는 반대로 움직이는 ‘metric divergence’가 확인돼, 향후 디컴파일·코드 생성 전반에서 pass@k 중심 평가가 필요하다는 파급효과가 크다.



### Static Metrics Are Insufficient: Predicting Java Method Energy Usage with Execution Tim (https://arxiv.org/abs/2607.06124)
Comments:
          Accepted for publication at the 19th International Conference on the Quality of Information and Communications Technology (QUATIC 2026)

- **Prior Approaches**: 기존 연구는 에너지 절감을 주로 컴파일러/런타임 최적화나 벤치마크 단위의 거친 추정에 초점을 맞췄고, 정밀 측정은 반복 프로파일링과 하드웨어 계측이 필요해 개발 흐름에서 쓰기 어려웠다. 메서드 같은 더 미세한 단위에서는 static 코드 메트릭만으로 에너지를 설명하려는 시도가 제한적이었고, 일부 혼합 접근도 플랫폼 가정이나 실행 기반 의존이 컸다.

- **Core Contribution**: 이 논문은 Java 메서드 수준에서 static source code features(제어 흐름/복잡도/API 사용 등)만으로 에너지 소비를 예측할 수 있는 한계를 먼저 규명한다. 또한 execution time을 “가벼운 dynamic 입력”으로 추가했을 때 예측 정확도가 얼마나 개선되는지 정량적으로 분석한다.

- **Technical Challenges**: 가장 큰 어려움은 에너지가 실행 환경과 JVM 런타임 상호작용에 크게 좌우돼, static 메트릭만으로는 충분한 설명력을 얻기 어렵다는 점이다. 연구진은 2,786개 Java 메서드를 대상으로 33개 static feature를 추출하고 async-profiler와 JoularJX로 execution time·에너지를 측정한 뒤, 11개 회귀 모델을 feature selection(RFECV, AutoSpearman 등)과 hyperparameter tuning(RandomizedSearchCV)까지 포함해 비교·검증했다.

- **Empirical Impact**: static features만으로는 평균 R2가 거의 0에 가까워 예측력이 매우 낮았지만, execution time을 함께 넣으면 R2가 최대 0.46까지 상승했다. 예측에 가장 일관되게 기여한 변수는 execution time, 내부 메서드 호출 수, cyclomatic complexity였고, 앙상블 계열(RF/ADA)이 비교적 안정적이었다. 이 결과는 “정확한 에너지 값”보다는 코드 구조-런타임 사이의 관계를 조기에 파악해 최적화 대상을 좁히는 방법론적 기준을 제시하며, CPU/메모리 등 추가 dynamic feature로 확장할 동기를 강화한다.



### x-Prediction Is All You Need:Training-Free Accelerated Generation via Endpoint Decodability (https://arxiv.org/abs/2607.06114)
- **Prior Approaches**: 확산 모델과 flow matching 모델은 샘플링에 ODE 솔버를 쓰며, 보통 수십~수백 번의 NFE가 필요해 실사용 비용이 커졌다. 이를 줄이기 위한 기존 접근은 distillation, consistency 기반 학습, Rectified Flow처럼 경로(trajectory)나 훈련 레시피를 바꾸는 방법이 주류였고, 가속을 “추론 단계만”으로 얻는 해법은 부족했다. 또한 DPM-Solver, UniPC 같은 고차/개선 솔버는 보폭을 줄이지만 결국 전체 경로를 끝까지 따라가는 형태라, 학습 없이 더 큰 가속을 내기 어려웠다.

- **Core Contribution**: 이 논문은 affine probability path의 중간 상태와 경로 속도만으로 깨끗한 샘플 x0를 복원할 수 있다는 성질을 endpoint decodability로 정식화한다. 특히 표준 ℓ2 학습 목적 하에서, 유도된 decoder가 최소 MSE 추정기 E[x0|xt]와 같아 “충분히 정리된 중간 단계에서 바로 x0로 점프해 출력”하는 전략이 정당화된다. 이를 Truncated Jump Sampling (TJS)로 구현해, retraining·distillation·아키텍처 변경 없이도 ODE를 조기 종료하고 decoded x0를 반환한다.

- **Technical Challenges**: 핵심 난제는 조기 종료가 왜 편향을 만들지 않는지와, 언제/얼마나 일찍 멈춰도 충분한지에 대한 이론적 근거를 세우는 것이다. 저자들은 (xt, ut)가 x0를 복원하는 폐형식 디코더가 되기 위한 조건을 path determinant Δt≠0로 제시하고, ℓ2 목적에서 decoder가 MMSE-optimal임을 보인다. 또한 TJS 오차를 엔트로피/불확실성 항과 모델 예측오차 항으로 분해해, truncation이 기존 Euler 적분처럼 경로의 곡률(trajectory curvature)에 직접 페널티를 주지 않음을 보여준다.

- **Empirical Impact**: SDXL, SD3.5M, Z-Image-Turbo 및 class-conditional 벤치마크 등 여러 모델군에서 TJS는 NFE를 20–70% 줄이면서 품질 저하를 근접하게 유지하는 결과를 보였다. 특히 학습 없이도 체크포인트에 그대로 적용되며, moderate regime(예: 30→15–25 스텝)에서 속도-품질 트레이드오프 이득이 두드러졌다. 이 성과는 “출력으로 이미 계산되는 x0 추정치를 조기 종료 출력해도 된다”는 관점을 확산·flow matching 전반의 실용 가속 전략으로 확장했다.



### LLM-Guided Measurement Credibility Correction for Trustworthy Industrial Process Inferenc (https://arxiv.org/abs/2607.06111)
- **Prior Approaches**: 산업 예측과 soft sensing은 과거 multivariate 측정창을 기반으로 하지만, 현장에서는 바이어스·지연·stale·파생값 등으로 측정이 틀어져도 “그럴듯하게” 보일 수 있다. 기존 연구는 backbone 성능 향상이나 sensor reconstruction, fault-tolerant soft sensing, data reconciliation 등을 통해 이상 채널 복구·오류 보정을 시도했지만, 대개 상관 기반 가정, 알람/고장 라벨, 수치적 상관 또는 명시적 공정식에 의존한다.

- **Core Contribution**: 이 논문은 예측 전에 “현재 측정창이 현재 공정을 대표하는지”를 판단하고, 신뢰 가능한 외부 근거로 국소 측정을 의미 단위로 교정하는 LLM-Guided Measurement Credibility Correction(MCC)을 제안한다. MCC는 공정 문서의 변수 의미(측정 의미, 역할, 파생/관계)를 측정 의미로 변환해 수치 모델이 사용할 수 있는 형태로 고정하고, 예측기 투입 전 input window의 로컬 충돌을 보수적으로 수정한다.

- **Technical Challenges**: 핵심 난제는 상관 변수도 독립적 근거가 아닐 수 있다는 점이다(공유 계측기, 파생식, soft-sensing chain, 제어 행동 등으로 공동 왜곡). MCC는 LLM이 생성한 측정 의미로 독립적 process reference를 구성해 자기 자신(target measurement)을 참조하지 않도록 routing/가중치를 설계하고, calibration 분포 기반의 inconsistency로 correction trigger를 제어해 공유된 운전 변화는 과도하게 보정하지 않게 한다.

- **Empirical Impact**: 여러 산업 forecasting 및 soft-sensing 과제에서 +MCC는 Real Test에서 평균 relative MAE 30.7%, Corrupted Test에서 80.3%의 감소를 보이며, 단순 합성 복구를 넘어 고정된 현장형 테스트에서도 일관된 개선을 보였다. 또한 온라인 파라미터 증가는 0.5–2.0k 수준이고 최악 추론 시간도 0.089 ms/step으로 작아, 공정 문서 기반 측정 semantics가 사전 신뢰도 교정으로 정확도와 견고성을 함께 높인다는 점을 실증했다.



### RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations (https://arxiv.org/abs/2607.06109)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 adversarial training은 주로 단일 위협(예: ℓ∞)에 최적화돼 추론 시 보지 못한 위협에 취약합니다. Multi-perturbation adversarial training(MAT)은 여러 ℓp 위협을 함께 학습하지만, 서로 다른 위협이 만드는 분포 이동이 충돌해 특정 위협에서의 robust 성능이 떨어지는 robustness trade-off가 발생합니다. 이를 완화하려고 MoE를 MAT에 그대로 붙이면, 전문가가 위협 공통 특징만 중복 학습하거나 gating이 위협별로 거의 동일한 경로를 학습하는 threat-agnostic routing 문제가 남습니다.

- **Core Contribution**: 이 논문은 MAT에서 MoE를 사용할 때 핵심 병목인 redundancy(전문가의 중복 학습)와 threat-agnostic routing을 동시에 겨냥합니다. 제안하는 Robust Mixture of Low-Rank Experts(RoME)는 공유 backbone 위에 low-rank additive update 형태의 전문가를 얹어, backbone이 위협 공통 특징을 맡고 각 expert가 위협별 특징에 집중하도록 설계합니다. 또한 gating이 위협을 구분하도록 dual-scale gating(로컬/글로벌 특징 결합)과 threat-guided gating diversification(위협 간 routing 다양성 강제)을 도입해 위협별 model pathway를 형성합니다.

- **Technical Challenges**: RoME가 해결한 첫 과제는 전문가가 위협 공통 특징을 과도하게 가져가면서 threat-specific pathway가 만들어지지 않는 점입니다. 이를 low-rank 전문가(LoRA식 add-on)로 구현해, 전문가가 공유 지식을 훼손하지 않으면서도 threat-specific 정보를 학습하도록 제약을 줍니다. 두 번째 과제는 gating이 위협별 판별 신호가 충분하지 않아 유사한 expert 조합을 반복하는 threat-agnostic routing인데, 로컬/글로벌 이중 스케일 신호로 discriminative cues를 강화하고, 학습 중 위협 라벨을 활용해 gating 패턴의 거리를 벌리는 다양화 정규화를 통해 경로 차이를 강제합니다.

- **Empirical Impact**: CIFAR-10, ImageNet-100, ImageNet-1K에서 RoME는 기존 SOTA MAT 대비 union robustness와 natural accuracy를 동시에 개선하며, 학습 중 보지 못한 unseen threats에 대한 robustness도 향상시켰습니다. 또한 ablation/분석을 통해 low-rank expert 구성, dual-scale gating, gating diversification의 기여를 확인합니다. 특히 threat 라벨은 학습에서만 사용되지만, 추론 시에도 위협 타입을 몰라도 threat-adaptive expert 조합을 예측해 전이 효과를 보이며, non-ℓp	hreats에 대해서도 견고함이 확장되는 점이 의미 있습니다.



### EcoVision: AI-Powered Drone Imaging for Salt Marsh Vegetation Monitoring and Dominance Mapping (https://arxiv.org/abs/2607.06105)
Comments:
          37 pages, 8 Figure, 6 Tables

- **Prior Approaches**: 기존 생태 모니터링은 현장 조사에 크게 의존해 왔고, 사람의 시각적 판단과 표본 추정으로 인해 관찰자 편향과 확장성 문제가 반복해서 지적돼 왔다. UAV 원격탐사와 딥러닝이 등장했지만, 많은 연구가 픽셀 단위 분할이나 분류에 그쳐 경쟁/우점 같은 생태학적 해석으로 바로 연결되지 못했다.

- **Core Contribution**: EcoVision은 저고도 UAV RGB 영상에서 종 분할(semantic segmentation)부터 객체 단위 분류(object-level classification), 그리고 2x2m 격자 기반 우점도(dominance score) 산출까지를 하나의 모듈형 파이프라인으로 통합했다. 특히 Spartina maritima와 Puccinellia maritima처럼 서로 다른 형태가 수 cm 단위로 공존하는 염습지에서, 픽셀 예측을 정책/현장 조사와 맞닿는 정량 지표로 변환하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 겹친 캐노피와 배경 잡음(물, 침전물 등) 속에서 경계를 안정적으로 찾는 일, (2) 분할 결과를 객체 단위로 끊어내어 fine-grained 종을 식별하는 일, (3) 마지막에 2x2m 같은 현장 스케일로 일관되게 집계해 우점도를 계산하는 일이다. EcoVision은 SegFormer-B5로 식생 마스크를 만든 뒤 connected-component로 blob을 추출하고, ConvNeXt로 blob을 종 분류한 다음, confidence threshold를 적용해 격자 집계로 dominance를 계산하는 방식으로 이를 해결했다.

- **Empirical Impact**: 제안 파이프라인은 종 마스크에서 mean IoU 0.56, 픽셀 정확도 0.96을, 객체 수준 분류에서 F1 0.99를 보이며 영상 기반 식별 성능을 입증했다. 우점도 추정은 사분면(quadrat) 현장 조사와 평균 절대차 8% 미만으로 일치해, 현실적인 조건에서도 미세 공간 구조를 보존하면서 생태학적으로 해석 가능한 결과를 제공함을 시사한다.



### Agents That Teach: Towards Designing Incidental Learning Back into AI-Assisted Software Developmen (https://arxiv.org/abs/2607.06101)
Comments:
          5 pages. To be published in the proceedings of 41st International Conference on Automated Software Engineering (ASE '26), October 12-16, 2026, Munich, Germany (New Ideas and Emerging Results Track)

- **Prior Approaches**: AI coding agents 도입으로 디버깅·리팩터링·문서화까지 자율적으로 처리하는 흐름이 확산됐지만, 문제를 “푸는 속도”는 빨라지는 반면 이해와 학습은 따라오지 못할 수 있다는 우려가 제기돼 왔다. 기존 대응은 시뮬레이션·의도적 연습·인지적 forcing function·Socratic scaffolding처럼 사용자가 AI 출력을 받아들이기 전 적극적으로 관여하게 만드는 구조가 중심이었다. 다만 소프트웨어 공학에서는 ‘부수적 학습(incidental learning)’ 자체를 목표로 삼는 방식이 부족하다는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 에이전트가 코드를 바꾸지만 개발자가 완전히 이해하지 못하게 되면서 누적되는 Knowledge Debt를 제안하고, incidental learning은 자동으로 다시 생기지 않으므로 상호작용 설계로 의도적으로 복원해야 한다고 주장한다. 이를 위한 6가지 설계 원칙을 제시하고, 그 원칙을 구현한 멀티에이전트 시스템 SHIELD(“agents that teach” 기반)를 제안한다. SHIELD는 개발자의 흐름을 방해하지 않는 out-of-band 학습 순간을 생성해 생산성과 학습을 경쟁이 아닌 동시 목표로 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 에이전트가 만든 변경 자체가 아니라, 에이전트의 reasoning에서 드러나는 맥락을 바탕으로 “진짜로 가르칠 가치가 있는” 순간만 골라내는 것이다. 논문은 Telemetry Observer Agent로 에이전트 변경·이유·대안·confidence를 수집하고, Teachability Triage Agent가 개발자별 Concept Map과 teachability signals(복잡도·새로움·전이성)로 선별한다. 또한 Concept Map이 항상 정답이 아니므로 Probe Generator가 비동기 질문으로 스스로 설명하게 하고, Knowledge Assessor와 comprehension check로 학습 내재화를 확인해 Concept Map을 갱신하는 closed-loop를 설계했다.

- **Empirical Impact**: 실증 평가는 향후 사용자 연구로 계획돼 있으며, 이 제출에는 별도 데이터셋이 포함되지 않는다. 대신 VSCode용 초기 프로토타입(Claude Code 연동)으로 구현 가능성을 보여 주고, 에이전트의 이유를 바탕으로 Probe Queue와 Microlearning Feed를 통해 미세학습이 생성·검증되는 시나리오를 제시한다. 조직 내부 이해관계자 대상 초기 피드백이 유망했다고 밝히며, 향후 learning-aware development 환경과 Knowledge Debt의 정량화 연구로 확장하려는 로드맵을 제안한다.



### PVCap: Towards Accurate 3D Dense Captioning via PseudoCap and VoxelCapN (https://arxiv.org/abs/2607.06097)
Comments:
          13 pages

- **Prior Approaches**: 3D dense captioning은 3D 장면의 각 객체를 바운딩 박스로 위치시키고 자연어 문장을 생성하는 과제로, Scan2Cap·MORE·SpaCap3D 등은 관계 추론 모듈로 객체 간 관계를 모델링해 성능을 끌어올렸다. 또한 Vote2Cap 계열처럼 detection과 caption을 end-to-end로 결합하는 시도도 있었지만, 데이터 증강이 제한적이고 네트워크 아키텍처가 단순해 충분한 공간 정보·풍부한 의미 특징을 확보하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 PVCap(PseudoCap + VoxelCapNet)으로 두 병목을 동시에 해결한다. PseudoCap은 인스턴스 단위로 객체를 랜덤 믹싱해 다양한 공간 레이아웃의 pseudo frame을 만들고, teacher-student로 pseudo caption label을 생성해 학습 샘플 수와 환경 서술 능력을 키운다. VoxelCapNet은 voxel 기반 백본의 특징과 detection head의 객체 제안을 caption head에 직접 연결하도록 구성해 voxel 특징을 활용한 강한 캡션 생성 네트워크를 제안한다.

- **Technical Challenges**: pseudo frame에서는 객체 주변 문맥이 바뀌어 기존 캡션 라벨을 그대로 쓸 수 없다는 문제가 생기는데, 이를 teacher의 예측으로 pseudo label을 만들고 confidence가 낮은 라벨을 제거해 품질을 관리한다. 또 voxel 기반 아키텍처에서 caption head를 잘 붙이는 것이 핵심이라, VoxelCapNet은 voxel feature에서 detection head로 bbox와 confidence를 산출한 뒤 NMS·confidence 필터링으로 query(객체 특징)를 구성하고 surrounding contextual feature와 함께 caption head에 입력한다.

- **Empirical Impact**: PVCap은 ScanRefer와 Nr3D 두 벤치마크에서 state-of-the-art를 크게 갱신했으며, CIDEr@0.5IoU 기준으로 ScanRefer에서 11.41%p, Nr3D에서 13.99%p 향상했다. 특히 SCST 튜닝까지 포함했을 때 개선 폭이 더 커져 pseudo label 기반 데이터 증강과 voxel 기반 캡션 네트워크 설계가 함께 효과적임을 보여준다. 코드 공개도 예고되어 3D dense captioning 후속 연구를 위한 강력한 baseline으로 기대된다.



### From Blueprint to Reality: Modeling and Applying Putnam's Social Capital Theory with LLM-based Multi-agent Simulations (https://arxiv.org/abs/2607.06080)
Comments:
          23 pages, 13 figures, 11 tables

- **Prior Approaches**: 기존 Putnam의 Social Capital Theory 연구는 대규모 설문·SEM 같은 정량 분석으로 통찰을 얻지만, 통제와 재현성에 한계가 있다. ABM 등 시뮬레이션은 가정 검증에는 유리하지만 규칙 기반 에이전트로는 인간의 맥락 의사결정, 감정/상황 의존성을 충분히 반영하기 어렵다. 최근 LLM 기반 멀티에이전트는 인간 같은 행동을 만들지만, 이론의 핵심 명제를 직접 겨냥한 theory-driven 환경 구축이나 과정 수준 해석이 부족하다는 문제가 제기된다.

- **Core Contribution**: 본 논문은 SocaSim이라는 LLM 기반 multi-agent 시뮬레이션 프레임워크를 제안해 Social Capital Theory를 ‘이론 설계도 → 시뮬레이션 현실’로 연결한다. social network 진화, trust dynamics, norm propagation을 하나의 환경에 통합하고, 반복적인 collective-action 실험을 통해 Putnam의 세 차원을 동시에 모델링한다. 또 스마트 노인돌봄 적응 문제에 적용해 이론을 실제 의사결정/기술 채택 맥락으로 확장한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 네트워크-신뢰-규범이 시간에 따라 결합해 변하는 동역학을 에이전트로 구현하고 (2) 그 인과 경로를 단계별로 추적 가능하게 만드는 것이다. 이를 위해 에이전트에 SST 기반 구조적 사회 특징(인구/SES/사회자본 성향)을 부여하고, BDI로 라운드별 beliefs-desires-intentions을 추론하며, SCM 메모리·리플렉션으로 학습/갱신을 수행한다. 또한 Proposal–Execution의 2단계 실험으로 공동행동을 수행시키고, 반사실(counterfactual) 개입으로 신뢰 축적과 규범 내재화의 미시 인과 흐름을 process-level로 해석한다.

- **Empirical Impact**: 시뮬레이션은 25라운드 동안 네트워크 밀도 증가와 함께 cooperation 성공률이 동반 상승하는 등 Putnam의 거시 패턴을 재현한다. 특히 20명의 실제 고령자 시나리오 선택과 비교했을 때 그룹 수준 인간-에이전트 정렬이 높게 나타났고(Pearson r=0.974), 신뢰·규범·네트워크의 효과 순위도 일관되게 관측된다. 스마트 노인돌봄 적용에서는 저SES 집단의 초기 trust를 1.0 높이는 반사실 실험으로 채택률이 15.4%p 증가하고, 심리적 압박/불안 및 결정 모순이 각각 19.8%, 22.4%, 25.5% 감소해 ‘정책적 인과 레버’로서 trust의 실용성을 보여준다.



### Prompt Coach: An Empirical Evaluation of an Agentic Tutor for Learning Prompt Engineering in Software Developmen (https://arxiv.org/abs/2607.06074)
Comments:
          7 pages. To be published in the proceedings of 41st International Conference on Automated Software Engineering (ASE '26), October 12-16, 2026, Munich, Germany (Industry Showcase Track)

- **Prior Approaches**: 기존 튜터링은 정적인 지식 전달이나 반응형 대화에 머무르는 경우가 많아, 개발자의 코드베이스·작업 맥락·타깃 LLM 동작에 기반한 학습으로 이어지기 어렵다. 프롬프트 엔지니어링 관련 도구들도 가이드/자동 최적화 중심이라, 반복적 실습에서 생기는 인지적 맹점(제약·에러처리·컨텍스트 누락)을 어떻게 “배우게” 하는지는 약했다. 또한 영상·문서형 학습은 과제 맥락이 분리되고 피드백이 지연돼 in-flow 연습과 전이(transfer)가 제한된다.

- **Core Contribution**: 이 논문은 IDE 내부에서 작동하는 agentic tutor Prompt Coach(PC)를 제안한다. PC는 개발자가 만든 코드생성 프롬프트를 다차원으로 평가한 뒤, Socratic guidance 형태의 맞춤 질문을 제공해 개발자가 자기수정하도록 유도한다. 특히 프롬프트 품질을 “정답 제시”가 아니라 제약·에러·컨텍스트 등 약점 차원의 성찰로 연결한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 개발자 작업 맥락을 안정적으로 끌어와서 평가와 코칭에 반영하는 것, (2) 프롬프트 개선을 학습 신호로 바꾸되 결과물(완성 코드)을 그대로 노출해 학습을 건너뛰지 않게 하는 것이다. PC는 프로젝트 문맥을 코드/문서로부터 추출해 vector database에 저장하고, LLM-as-a-judge로 Clarity·Specificity·Context Awareness 등 8개 차원을 0-100 점수로 산정한다. 이후 Consequence Preview는 내부적으로만 실패모드를 파악하는 데 쓰고, Developer Modeling과 추적을 통해 다음 질문의 깊이와 초점을 적응적으로 조정한다.

- **Empirical Impact**: 15명의 숙련 개발자 대상 초기 실험에서 PC 사용 60분 후 프롬프트 품질 평균이 63.04에서 71.69로 유의미하게 상승했다(p<0.05). 특히 기준선에서 낮았던 Inclusion of Constraints, Error Handling, Context Awareness에서 개선 폭이 가장 컸고, 참가자 13/15명이 향상을 보였으며 성능 하락은 관측되지 않았다. 설문에서도 신뢰도와 채택 의향이 높았고, 100%가 PC가 프롬프트 작성 능력을 향상시켰다고 동의해 in-flow 학습 도구로서의 실용성이 확인됐다.



### PluraMath: Extending Mathematical Reasoning Evaluation Beyond High-Resource Languages (https://arxiv.org/abs/2607.05992)
- **Prior Approaches**: 수학적 추론 능력 평가는 reasoning LLM을 튜닝하고 평가하는 핵심 과제로 자리 잡았지만, 기존 벤치마크는 영어·중국어 중심으로 편향되어 고자원 언어 위주로 성능이 측정되는 문제가 있었다. PolyMath 같은 최근 데이터셋이 진전을 보였음에도 18개 고자원 언어로 범위가 제한되어, 언어 다양성을 충분히 반영하지 못한다는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 PolyMath의 확장판인 PluraMath를 제안해 6개 언어 계열에 걸친 18개 추가 저자원/중자원 언어를 포함시키고, 데이터 커버리지를 대폭 넓힌다. 또한 27개 reasoning LLM을 모델 스케일(소형~대형 및 closed-source 앙상블) 전반에서 평가해 다언어 수학적 추론 능력을 체계적으로 비교한다.

- **Technical Challenges**: 핵심 기술 과제는 다양한 언어로 수학 추론 문제의 의미와 난이도를 일관되게 유지하는 데이터 구축이다. 논문은 사람 큐레이션 파이프라인을 통해 원문 기반 번역을 사전 계산한 뒤 원어민이 철저히 검증하는 방식으로 품질을 확보했고, 평가 프레임워크까지 함께 공개해 재현성과 확장성을 높였다.

- **Empirical Impact**: 실험 결과, 고자원 언어와 저자원 언어 간 수학적 추론 성능 격차가 여전히 지속되며, 격차의 상당 부분이 instruction-following 능력과 연관된다는 분석을 제시한다. 더불어 데이터셋·획득 파이프라인·평가 프레임워크를 오픈소스로 공개해, underrepresented communities의 다언어 벤치마크 개발 진입 장벽을 낮추는 데 의미가 있다.



### Propose and Attend: Training-free MLLM Grounding Confidence via Multi-Token Localized Attention (https://arxiv.org/abs/2607.05978)
- **Prior Approaches**: 기존 연구는 SVAR처럼 생성 텍스트의 한 위치(예: 첫 서브토큰)에서 전 입력 모달리티 토큰으로의 attention을 전역 합산해 환각을 감지하거나, hidden-state 유사도(예: GLSim, ContextualLens)로 근거 부족을 판단했다. 하지만 localization 출력은 박스/시간창처럼 ‘제안된 영역’과 결합돼 있어야 신뢰도를 매길 수 있는데, 이들은 대개 입력 전체를 통째로 보거나 추가 분류기 학습이 필요해 신호가 약했다.

- **Core Contribution**: 이 논문은 Multi-Token Localized Attention(MTLA)이라는 학습 없는 사후(post-hoc) 점수를 제안해, 모델이 주장한 영역 내부에 실제로 attention 근거가 모이는지를 정량화한다. 핵심은 (1) 예측이 스스로 제안한 region 안에서만 attention을 합산하고, (2) 좌표와 라벨 등 multi-token에 분산된 근거를 함께 집계해 더 강한 grounding 신호를 만든다는 점이다.

- **Technical Challenges**: 가장 큰 난제는 토큰 log-probability가 좌표 근거(grounding)와 입력 모호성(hallucination 유발 요인)을 섞어 분리도가 낮다는 것이다. MTLA는 각 응답 토큰이 생성하는 proposal token set을 파싱한 뒤, proposal region과 교차하는 모달리티 토큰에 대해서만 attention을 masked aggregation하고, 여러 예측 토큰을 평균 내며, 아키텍처/모달리티에 덜 민감한 레이어 밴드로 점수를 안정화했다.

- **Empirical Impact**: COCO(이미지), Charades-STA·QVHighlights(비디오), AudioSet-Strong(오디오)에서 MTLA는 환각 감지 AUROC를 기존 훈련 없는 베이스라인 대비 +7~+38까지 개선했다. 또한 MTLA를 confidence로 재랭킹하면 open-source 8B/일반ist 모델의 zero-shot COCO detection AP가 크게 상승(20.4→약 37.0)하며, supervised 탐지기와의 격차를 상당 부분 줄였다.



### MCP-Enabled Agentic AI for Autonomous IPoDWDM Network Lifecycle Automation (https://arxiv.org/abs/2607.05975)
Comments:
          Accepted for demo presentation at the European Conference on Optical Communication (ECOC 2026)

- **Prior Approaches**: 기존에는 IPoDWDM 네트워크의 구성·운영 자동화를 작업 단위로 나눠 사람이 개입하거나, 특정 벤더 도구에 강하게 의존하는 방식이 많았다. 이로 인해 멀티레이어 변경 전 주기(라이프사이클) 전반을 일관되게 닫힌 고리(closed-loop)로 제어하기 어렵다는 한계가 있었다. 또한 네트워크 상태를 실시간 텔레메트리로 반영해 자동 의사결정을 반복하는 통합 구조가 부족했다.

- **Core Contribution**: 이 논문/데모는 MCP-enabled 에이전틱 AI 아키텍처를 제안해, 벤더 비종속(vendor-agnostic) IPoDWDM 네트워크의 자율 제어를 목표로 한다. 핵심은 텔레메트리와 함께 에이전트가 라이프사이클 전 과정을 연결해 end-to-end로 오케스트레이션하고, closed-loop 제어로 지속적으로 조정하는 점이다. 데모는 GNPy를 활용한 실제 제어 흐름을 통해 개념을 구체화한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 (1) 멀티레이어 자동화 작업을 한 번에 끊김 없이 연결하는 end-to-end 계획·실행 체계, (2) 실시간 텔레메트리 기반으로 상태를 추정·반영해 closed-loop로 교정하는 제어 로직, (3) 벤더 의존성을 줄이면서도 네트워크 변경을 안정적으로 수행하는 연동이다. 제안 아키텍처는 MCP 기반 에이전트가 GNPy와 텔레메트리를 함께 사용해 관측-판단-제어 사이클을 구성함으로써 이 문제들을 해결한다.

- **Empirical Impact**: 저자들은 실제 테스트베드에서 live end-to-end 라이프사이클 멀티레이어 자동화와 closed-loop 제어를 검증해, 에이전틱 에이전트가 실운영에 가까운 조건에서 동작함을 보였다. 이는 통신 네트워크 자동화 분야에서 에이전트형 제어를 벤더 비종속 환경으로 확장하는 실증 사례로 의미가 있다. 결과적으로 ML/AI 기반 네트워크 오케스트레이션의 실시간 제어·검증 경로에 대한 참고점이 될 전망이다.



### Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search (https://arxiv.org/abs/2607.05970)
Comments:
          5 pages, 1 figure, accepted at SynthIR @ SIGIR 2026

- **Prior Approaches**: 기존 dataset search는 제목·설명·키워드 같은 메타데이터를 기반으로 데이터 원본을 직접 다루지 않는 경우가 많아, 메타데이터 품질이 검색 성능을 좌우한다. LLM로 자동 설명을 생성해 검색 유틸리티를 높이려는 시도도 있으나, 생성물이 실제 RDF 근거와 얼마나 일치하는지(grounding/faithfulness)는 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 RDF 데이터셋의 메타데이터를 생성하는 6가지 설정(단순 rewrite, profile 기반 rewrite/gen, profile title 보조, graph 접근 agentic generation)을 한 프레임에서 비교해, 검색 성능과 faithfulness를 함께 평가한다. 특히 ‘검색이 좋아지면 메타데이터도 좋아진다’는 가정을 검증하며, 효과와 신뢰(근거/프로비넌스 비용)를 동시 관점에서 재정의한다.

- **Technical Challenges**: 핵심 기술적 난제는 생성물이 쿼리 매칭에 유리한 확장을 하면서도 원천 RDF 또는 제공된 증거로부터 나온 주장만 유지하도록 만드는 것이다. 연구진은 dataset profile를 표준화된 압축 증거로 사용하고, claim-level로 원자 단위를 추출한 뒤 설정별 권위 증거(원 메타데이터/프로필/전체 그래프)에 대해 LLM judge(일부는 RDF-grounded agentic judge)로 supported/contradicted 등을 판정해 trade-off를 분해 측정한다.

- **Empirical Impact**: ACORDAR 2.0의 약 1,000개 중 1,000문서 예산 파일럿과 150개 데이터셋 claim-level faithfulness 분석에서, unconstrained metadata rewrite가 retrieval 성능(NDCG@10 등)을 가장 크게 올리지만 faithfulness는 가장 낮았다. 반대로 profile-grounded rewriting은 검색 성능과 근거 일치 사이의 균형이 가장 좋아 ‘synthetic metadata는 시스템 수준 IR 문제’라는 결론을 강화하며, 향후 RDF profiling 고도화와 agentic grounding 통제가 중요하다는 방향성을 제시한다.



### InfluMatch: Frontier-Quality KOL Search at 4B-Model Cos (https://arxiv.org/abs/2607.05968)
- **Prior Approaches**: 기존 KOL(핵심 의견 리더) 매칭은 키워드 기반 검색이나 정형 속성 필터로 처리되는 경우가 많았지만, 의미 적합도는 놓치고(어휘는 다르지만 내용은 맞는 경우) 캠페인별 다중 조건을 정적 스키마로는 반영하기 어렵습니다. 또한 모든 후보에 대해 frontier LLM을 즉시 추론하는 방식은 정확도는 높아도 지연과 비용이 커 운영에 부담이 됩니다.

- **Core Contribution**: InfluMatch는 태국어의 자유형·다중 파트 마케팅 기준을 받아 KOL을 단계별로 좁힌 뒤 재평가하고, 각 기준별 점수와 태국어 근거를 함께 출력하는 배치 가능한 3단계 캐스케이드( retrieval → rerank → reason )를 제안합니다. 특히 소형 오픈 웨이트 모델만으로도 end-to-end 순위 품질을 확보하면서, frontier 수준의 성능을 저비용으로 노리는 설계를 전면에 둡니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 후보 수가 큰 상태에서 “설명 가능한 판단(reason)”을 비싸게 돌리지 않으면서도, (2) 학습/추론에서 점수 체계가 실제 순위 최적화와 잘 맞도록 하는 것입니다. InfluMatch는 dense retrieval top-50을 만든 뒤 4B pointwise reranker로 top-10만 추려 단일 Yes 토큰 log-prob로 순위를 정하고, 4B reasoner는 기준별 루브릭 채점+태국어 rationale을 생성하도록 하되, fine-tuning은 pairwise SimPO가 end-to-end 전이에 유리하고 reasoner는 untuned base가 가장 강하다는 점을 실험적으로 확인해 배치 설계를 정리했습니다.

- **Empirical Impact**: 실험에서 retrieval-only는 거의 랜덤 수준에 머물렀지만, rerank→reason 전체 캐스케이드는 11개 쿼리 세트에서 P@5 94.1%를 달성하며 frontier 모델 Kimi-K2.6(91.8%)과 비슷한 수준을 저비용으로 따라갑니다. 또한 출력 토큰을 약 35배 줄이고 단일 A100에서 50개 KOL 쿼리를 약 20초 내 처리하는 등 운영 효율이 뚜렷하며, 특히 reasoner의 offline 성능 향상이 end-to-end에서는 역효과가 될 수 있음을 사례로 보여 실전 학습/라벨링 전략에 시사점을 줍니다.



### Decoupled Single-Mask Annotation Noise Detection via Cross-Sectional Patch Self-Consistency (https://arxiv.org/abs/2607.05965)
Comments:
          13 pages, 6 figures. Accepted by MICCAI 2026

- **Prior Approaches**: 혈관 CT 세그멘테이션은 얇은 관 구조와 조영제 확산 때문에 고품질 라벨을 얻기 어렵고, 현실적으로 한 스캔당 한 번만 주석이 달리며 국소적인 마스크 노이즈가 생긴다. 기존 대응은 (1) 다중 라이터 퓨전처럼 여러 마스크를 필요로 하거나, (2) loss/가중치/구조 보정 등 학습과 결합된 robust learning을 통해서만 간접적으로 다루는 경우가 많아 라벨 실패의 ‘감사(audit) 가능한 근거’ 제공이 어렵다.

- **Core Contribution**: 논문은 단일 마스크(single-mask) 설정에서 이미지-마스크 쌍만으로 annotation noise를 위치화(로컬라이즈)하는 decoupled 프레임워크를 제안한다. 핵심은 혈관의 횡단 단면이 공간과 피험자에 걸쳐 반복되는 점을 이용해 cross-sectional patch self-consistency를 정의하고, 유사한 이미지 패치인데 마스크가 불일치하면 해당 영역을 노이즈로 플래그하면서 각 의심 구간에 대한 해석 가능한 패치-쌍 근거를 제공한다.

- **Technical Challenges**: 가장 큰 기술 과제는 ‘어떤 좌표계/정렬에서’ 횡단 패치를 비교해야 비슷한 모양의 이웃을 충분히 검색하느냐이다. 이를 위해 Frenet–Serret 대신 Bishop frame(병렬이동 기반, torsion-free)을 중심선 그래프에 구성해 안정적인 orthogonal patch를 뽑고, intensity-equivalent 이웃을 scalable vector search로 찾은 뒤 MSE 기반 이미지 유사도 구간별로 mask disagreement(1-IoU)의 조건분포를 보정해 residual을 z-score 형태로 계산, 패치 수준 노이즈 점수를 산출하고 스캔 단위 3D quality map으로 집계한다.

- **Empirical Impact**: ImageCAS 관상동맥 CT 실험에서 quality-weighted training은 경계 민감 지표(CPR-DSC, ASD, HD-95)를 중심으로 개선하며, 전체 DSC는 비슷한 수준으로 유지되어 국소 경계 오류 완화에 강점이 있음을 보여준다. 또한 노이즈가 무작위가 아니라 체계적 편향을 가지며, 특히 transverse/oblique 혈관이 axis-aligned 구조보다 오류율이 5.1배 높고 면적·강도와도 상관을 보이는 등, 탐지 결과가 dataset quality assessment와 QA에 직접 활용될 수 있음을 입증한다.



### Agentic AI for IPoDWDM Network Lifecycle Automation: An MCP-Enabled Architectur (https://arxiv.org/abs/2607.05958)
Comments:
          Accepted for oral presentation at the European Conference on Optical Communication (ECOC 2026)

- **Prior Approaches**: 기존 SDN 기반 자동화는 장비 제조사나 계층별 제약에 묶여, 멀티벤더·멀티레이어 IPoDWDM 네트워크에서 end-to-end 서비스 수명주기 전반을 일관되게 자동화하기 어려웠습니다. 또한 폐루프 제어는 가능하더라도, 광계/전송계의 상태를 충분히 반영한 교차계층(크로스 레이어) 동기화가 취약하다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 vendor-agnostic 분산형 multi-MCP 아키텍처를 제안해, 멀티벤더·멀티레이어 IPoDWDM에서 E2E 서비스 라이프사이클 자동화를 목표로 합니다. 더불어 GNPy 모델과 광(옵티컬) 텔레메트리를 활용한 closed-loop cross-layer control을 제공하는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 네트워크 계층과 벤더의 상태를 하나의 제어 흐름으로 정합성 있게 묶고, 정확한 모델 기반 결정을 실시간 관측과 연결하는 데 있습니다. 이를 위해 GNPy 기반의 모델링과 optical telemetry를 폐루프에 통합해, 교차계층 제어가 가능한 형태로 추론·피드백 루프를 구성했다고 설명합니다.

- **Empirical Impact**: 제안 프레임워크는 IPoDWDM 테스트베드에서 실험적으로 검증되었으며, E2E 서비스 라이프사이클 자동화와 교차계층 폐루프 제어의 동작 가능성을 확인했습니다. SDN 자동화 연구에서 멀티벤더·멀티레이어 환경의 실사용 장벽을 낮추는 방향으로, 자율 제어 및 운영 자동화 확산에 의미가 있습니다.



### NegROI: Click-Centric Uncertainty-Guided Refinement with Scene-Conditioned Negative Prompts for Robust Interactive 3D Segmentation (https://arxiv.org/abs/2607.05955)
- **Prior Approaches**: 기존 대화형 3D 세그멘테이션은 voxel 그리드와 transformer 디코더를 중심으로 click 토큰을 장면 특징에 결합해 마스크를 갱신한다. 그러나 단일 해상도 정제는 소수 클릭에서 경계를 뭉개고, 배경 구조와의 혼동이 hard false positives로 이어지며, 밀도·스케일이 다른 데이터셋 간에는 고정된 refinement 휴리스틱과 click만 기반 디코딩이 일반화에 취약하다는 한계가 있었다.

- **Core Contribution**: NegROI는 (1) click 주변만 미세 그리드로 정제하는 click-centric multi-resolution ROI refinement와 (2) 장면 조건 negative prompts를 결합해 경계 정확도와 false positive 억제를 동시에 노린다. 추가로 불확실성 기반 선택적 정제, negative prompt의 diversity regularizer, 경계 인접 고신뢰 오탐에 초점을 맞춘 boundary-aware hard negative mining을 도입해 다양한 장면에서도 안정적으로 동작하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 소수 클릭 상황에서 fine 경계를 복원하면서도 계산량을 폭증시키지 않는 “정제 위치·해상도” 제어와, 배경의 유사 구조로 인해 생기는 고신뢰 false positive를 명시적으로 억제하는 “negative 모델링”이다. 논문은 coarse 예측 후 클릭 중심 ROI를 더 촘촘한 voxel 해상도로 re-voxelize해 로짓을 국소적으로 다시 디코딩한 뒤 coarse에 max-aggregation과 residual fusion으로 되돌리고, cross-attention 기반 scene-conditioned negative prompts를 boundary-proximal hard negatives로 직접 감독해 오탐 억제에 학습 신호를 준다.

- **Empirical Impact**: ScanNet40(인-도메인)과 S3DIS·KITTI-360(아웃-도메인)에서 IoU@k 및 mAP 지표가 향상되었고, 특히 low click 예산(예: IoU@1~3)에서 개선 폭이 크게 나타났다. 또한 NegROI는 경계 인접 오탐(band false positives)을 줄이며, ScanNet20 단일 클릭 평가에서도 비대화형/오픈보케이블 3D 베이스라인 대비 성능을 보이며 견고성을 입증했다. 결과적으로 “경계 품질 + hard distractor 억제 + 교차 데이터셋 강건성”을 함께 달성하는 대화형 3D 세그멘테이션 방향을 제시했다.



### Signed-Graph Recommendation as Structural Consistency Maximization (https://arxiv.org/abs/2607.05952)
- **Prior Approaches**: 기존 signed social recommendation은 trust/ distrust 관계를 함께 쓰지만, 희소·잡음·구조적 불균형 때문에 성능이 흔들린다는 한계가 반복해서 지적돼 왔습니다. 많은 방법이 관측된 signed graph를 고정 입력으로 보고 GNN의 message passing만 수행하며, 구조(토폴로지)–전파(다중 홉)–표현(임베딩) 사이의 정합성을 따로 맞추지 못한다는 문제가 있습니다.

- **Core Contribution**: 이 논문은 기존 모델에서 structural, propagation, semantic의 세 층이 서로 불일치해 sparse/noisy 데이터에서 편향된 표현이 학습된다고 규정합니다. 그리고 signed social recommendation을 ‘structural consistency maximization’으로 재정의한 뒤, 구조-전파-의미를 한 루프에서 공동으로 개선하는 SSC-Loop를 제안합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 신뢰 가능한 signed 관계만 남기고 나머지는 정제하는 graph refinement, (2) 다중 홉 전파 중 positive/negative/neutral 신호가 섞이거나 극성이 훼손되는 문제, (3) 임베딩 기하가 signed 링크의 의미(가까움/멀어짐)와 맞지 않는 문제를 동시에 다루는 것입니다. SSC-Loop는 ESA-DA로 degree·connectivity·structural balance 제약 하에서 간선 삭제/추가를 하며, P/N/O aggregation으로 극성 보존 전파를 만들고, contrastive learning으로 signed semantic alignment(positive는 당기고 negative는 밀기)를 학습합니다.

- **Empirical Impact**: Epinions에서 SSC-Loop는 RMSE 0.4658→0.4398(SIGformer 대비)로 명확한 개선을 보이며, 구성요소 제거 실험에서 ESA-DA가 특히 큰 기여를 하는 것으로 확인됩니다. 또한 Slashdot의 link-existence 파생 과제에서도 성능을 입증해, explicit rating 예측을 넘어 signed 사회 구조 활용 능력이 있음을 시사합니다.



### CMDR: Contextual Multimodal Document Retrieva (https://arxiv.org/abs/2607.05927)
Comments:
          Accepted by ECCV 2026; project page: this https URL

- **Prior Approaches**: 기존 멀티모달 문서 검색 벤치마크는 질의-단일 페이지 간의 단순 어휘/의미 매칭을 주로 평가해, 여러 페이지에 걸친 문서 맥락을 이용한 간접적 추론 능력을 검증하지 못했다. 또한 대부분의 방법이 페이지를 독립 인코딩해 문서 전역 구조나 cross-page dependencies의 이점을 과소평가하는 한계를 보였다. 멀티홉 계열도 대체로 명시적 매칭에 기반한 직접 검색 성격이 강해, “질의에 직접 드러나지 않는 관련 페이지를 찾아야 하는” 문제를 충분히 다루지 못했다.

- **Core Contribution**: 논문은 Contextual Multimodal Document Retrieval(CMDR) 태스크를 제안하고, 이를 평가하는 CMDR-Bench를 공개한다. CMDR-Bench는 6개 도메인 255개 장문 문서(평균 183.5페이지)에서 수작업으로 큐레이션된 800개 질의로, 페이지 간 맥락 모델링이 필수인 간접 검색을 요구한다. 모델 측면에서는 문서 여러 페이지를 함께 인코딩하되 페이지 수준 임베딩으로 분리해 문맥을 반영하는 CMDR-Embed와, 이를 학습하는 Contextual Multimodal Contrastive Learning(CMCL)을 함께 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 “여러 페이지를 함께 인코딩해 맥락을 얻는 것”이 동시에 동일 문서 내 페이지 표현을 섞어 페이지별 구별력(discriminability)을 떨어뜨린다는 점이다. 이를 해결하기 위해 CMDR-Embed는 chunk-then-split 전략으로 컨텍스트를 공유하되 페이지 임베딩은 분리해 Late Interaction(LI)로 질의-페이지 매칭을 수행한다. 학습에서는 CMCL이 두 종류의 context-aware hard negatives(같은 chunk 내 In-Chunk Negatives, 같은 문서 내 먼 chunk의 In-Document Negatives)를 조합해 맥락 활용과 페이지 구별력을 균형 있게 최적화한다.

- **Empirical Impact**: 실험 결과 CMDR-Embed는 비문맥(non-contextual) 임베딩 대비 유의미하게 성능이 개선되며, 동일 학습 데이터 조건에서도 컨텍스트 모델이 우위를 보였다. CMDR-Bench의 카테고리 분석에서는 특히 CR/MR처럼 참조 해석과 다중 페이지 집계가 필요한 영역에서 기존 멀티모달 검색기가 더 큰 어려움을 드러냈고, 이 격차를 CMDR-Embed가 완화한다. 또한 CMCL의 hard negative 설계와 LI의 멀티벡터 구조가 성능 향상에 일관되게 기여함을 보여, 장문 문서 검색에서 context-aware multimodal embeddings의 필요성을 실증적으로 강조한다.



### PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails (https://arxiv.org/abs/2607.05910)
- **Prior Approaches**: 기존 비전 가드레일 벤치마크는 고정된 안전 분류 체계나 정적 harmfulness 정의에 의존해, 안전을 이미지의 고유 성질처럼 취급하는 경우가 많았습니다. 정책 조건을 넣는 평가도 있었지만, 같은 이미지에서 정책 경계가 바뀔 때 결정을 뒤집는 능력(정책-적응)을 미세하게 측정하기엔 부족했습니다. 그 결과 많은 모델이 위험 큐는 알아도 정책이 허용/차단을 바꾸면 판단을 안정적으로 수정하지 못하는 취약성이 드러났습니다.

- **Core Contribution**: 이 논문은 정책이 런타임마다 달라질 때(제품/지역/규정 변경 등) 동일 이미지가 서로 다른 결정을 받아야 하는 문제를 다루며 PolicyShiftBench와 PolicyShiftGuard를 제안합니다. PolicyShiftBench는 265개 이미지에 대해 총 2,000개의 정책-판별 인스턴스(7개 위험 카테고리 조합, 28개 정책 변형)를 구성하고, 같은 이미지의 pass/block “정책 플립”을 직접 평가하는 Policy Shift Score(PSS)를 도입합니다. PolicyShiftGuard는 policy-conditioned 가드레일로, Randomized Policy SFT(RP-SFT)와 Boundary-Pair Policy Adaptation(BP-Adapt) 2단계 학습을 통해 정책 경계에 맞춰 결정을 뒤집도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 모델이 이미지에서 위험 단서를 인식하는 것에 그치지 않고, 현재 주어진 정책 텍스트의 경계를 읽어 동일 시각 증거에 대해 안전 판정을 ‘조건부로’ 바꾸는 능력을 확보하는 데 있습니다. 논문은 이미지 속성(atomic attributes)과 정책 판단을 분리해 라벨을 실행 가능한 규칙으로 결정하고, BP-Adapt에서 같은 이미지·카테고리에 대해 “허용 정책 vs 차단 정책”이 매칭된 boundary-pair를 만들어 쌍대 비교 형태의 손실로 안전 점수의 여백을 강제합니다. 또한 RP-SFT에서 정책 제시 순서/표면 식별자 등을 랜덤화해 고정 템플릿이나 위치 기반 지름길 의존을 낮추고 정책-추종의 견고성을 높였습니다.

- **Empirical Impact**: 실험에서 기존 VLM 및 전문 가드레일은 F1은 그럴듯해도 PSS가 매우 낮아, 정책 플립에 취약함이 확인됐습니다. 반면 PolicyShiftGuard-7B는 PolicyShiftBench에서 Avg. F1 76.9, Avg. PSS 72.1을 달성하며, UnSafeBench와 SafeEditBench로도 전이 성능이 좋고 추론 속도까지 개선되어 latency–performance 트레이드오프에서 실용성을 높였습니다. 특히 ablation은 matched pass/block boundary pair 및 pair loss가 정책-적응을 안정적으로 만들기 위한 필수 요소임을 보여줍니다.



### K-ABENA: K-Adaptive Backpropagation with Error-based N-exclusion Algorithm : (Compensated Loss-Based Sample Exclusion with Unbiased Gradient Estimation) (https://arxiv.org/abs/2607.05903)
Comments:
          11 pages main text + appendices, 13 pages total. Code: this https URL

- **Prior Approaches**: 선택적 backprop(예: OHEM, SBP)은 손실이 낮은 샘플의 backward를 스킵해 계산을 줄이지만, 선택된 집합이 손실에 상관돼 gradient가 편향된다. 반면 Focal Loss 같은 soft reweighting은 전체 배치의 backward를 그대로 계산해 ‘선택에 따른 compute saving’을 제공하지 못한다.

- **Core Contribution**: 이 논문은 K-ABENA(K-Adaptive Backpropagation with Error-based N-exclusion)로, 저손실(minor) 일부를 backward에서 제외하되 설계 기반 역확률 가중치로 편향을 보정하는 프레임워크를 제안한다. 특히 canonical(v3)에서는 survey sampling의 Horvitz–Thompson(H-T) 아이디어를 선택적 backprop에 통합해, 설계 무편향 gradient 추정기와 실무용 self-normalized(Hájek) 변형의 편향(차수 O(1/m))을 함께 다룬다.

- **Technical Challenges**: 핵심 기술 난점은 ‘손실 기반 선택’이 만들어내는 선택 편향을 compute 절감과 동시에 제거하거나 제어하는 것이며, 이를 위해 defensive-mixture 샘플링 설계와 inclusion probability의 엄밀한 양의 하한(positivity)을 보장한다. 또한 HT 형태는 설계 무편향임을 보이고, self-normalized 형태는 분명한 bias floor와 SGD의 비볼록 수렴(기대 제곱 gradient 노름의 O(1/sqrt(T)) 보장 + 편향 잔차 항)을 정리해 이론적 안전성을 제시한다.

- **Empirical Impact**: 실험에서는 uncompensated(보정 없는) 선택 계열이 극단적 클래스 불균형/라벨 노이즈에서 구조적으로 실패하며, 한 synthetic 극단 불균형(0.17%) 설정에서 full-batch SGD가 0.9998 AUC인 반면 편향 변형은 0.53~0.62에 그친다. 반대로 compensated(v3/H-T 보정) 추정기는 동일한 28.4% compute savings 조건에서 0.9991 AUC를 달성하고, Breast Cancer·Digits·Wine·Diabetes 같은 실제 데이터에서도 full-batch SGD와 통계적으로 구분되지 않으면서 per-epoch gradient 계산의 28–54%를 절감한다.



### From Textural Counterpoint to Feature Encoding: A Multi-Dimensional Machine Representation Study of Haydn's "The Lark" Integrating Electroacoustic Analysis (https://arxiv.org/abs/2607.05902)
- **Prior Approaches**: 기존 딥 음악 생성은 멀티파트 음악을 음(note) 배열로 단순화하거나, 고정 격자(quantization grid)에 기반해 강제 정규화를 수행하는 경우가 많았다. 이 방식은 현악 사중주의 미세한 마이크로-타이밍(루바토)과 역할 인식(role perception)을 쉽게 지워 ‘한 방향 멜로디 생산’으로 수렴할 위험이 있다.

- **Core Contribution**: 본 논문은 Haydn ‘현악사중주 D장조 5번(종달새)’의 1악장에서 나타나는 ‘지배-대위-기초(leading/support)’ 역할 구조를 고전 분석과 음향 계측을 통해 재현 가능한 표현으로 연결한다. 특히 Event-based Timestamps와 Role-Aware Encoding을 저수준 입력에 직접 주입해, 기계가 음 높이뿐 아니라 역할 분담과 양보(상호작용)를 학습하도록 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 악보의 이산 기호만으로는 물리적 질감 차이(주파수 대역, 어택/전이 양상, loudness)를 복원하기 어렵고, (2) 고정 시간 격자가 마이크로-타이밍의 탄성감을 훼손한다는 점이다. 논문은 DAW 기반 스펙트럼·전이·RMS/루드니스 측정으로 ‘phenomenological anchors’를 만들고, Δt를 PPQ에 동기화한 상대 오프셋(Event-based Timestamps)으로 시간 그리드를 버리며, 저지연 추론을 위해 RTNeural 같은 오디오 DSP 지향 라이브러리로 구현했다.

- **Empirical Impact**: 실험/분석은 종달새 1악장에서 상성처럼 들리는 역할들이 실제로는 주파수 대역 분리와 전이 밀도, 그리고 ‘더 큰 RMS=멜로디’ 같은 기계적 가정의 붕괴로 확인된다는 점을 데이터로 제시한다. 결과적으로 역할 인식과 탄성 시간 표현을 결합한 표현학적 정합성이 강화되며, 인간-컴퓨터 합주를 ‘사회적 속성(otherness awareness)을 갖는 상호작용’으로 확장하는 이론·실무 발판을 제공한다.



### Few-Medoids: An Embarrassingly Simple Coreset Selection Method for Few-Shot Knowledge Distillation (https://arxiv.org/abs/2607.05891)
Comments:
          Accepted at KES 2026

- **Prior Approaches**: 코어셋 선택은 대규모 데이터에서 일부를 골라 학습 효율을 높이려는 방법이지만, 지식 증류(KD)로 옮기면 무작위(random) 선택이 강력한 기준선이 되는 등 성능 개선이 어렵다는 문제가 지적된다. 특히 기존의 herding, k-center Greedy 같은 coreset 방법을 KD 파이프라인에 적용해도 랜덤 대비 확실히 이기지 못하는 경우가 많았다.

- **Core Contribution**: 본 논문은 few-shot KD 상황에서 teacher의 잠재공간을 이용해 클래스별 ‘중심 샘플’을 고르는 training-free 코어셋 방법 few-medoids를 제안한다. 각 클래스에서 teacher feature 기준으로 같은 클래스 내 평균 L2 거리(중앙성)가 가장 작은 샘플들을 우선 선택해, 수업 신호가 되는 대표 예제를 만든다.

- **Technical Challenges**: 핵심 난제는 KD용 코어셋이 단순 대표성만이 아니라 student가 teacher 신호를 잘 흡수할 수 있는 ‘학습 가능한’ 샘플을 골라야 한다는 점이다. 저자들은 클래스별 teacher feature 공간에서 기하학적 중심(medoid)을 직접 근사하도록 점수(평균 거리)만 계산하는 방식으로 복잡한 휴리스틱 없이 이를 해결했고, 이후 표준 soft-label KD 손실에 결합해 학습을 수행한다.

- **Empirical Impact**: CIFAR-10/100, Oxford Flowers 102, Food-101 등 4개 데이터셋과 여러 교사-학생 조합(ResNet·ViT)을 대상으로 실험했으며, few-medoids는 대부분의 k(클래스당 샘플 수) 구간에서 무작위 및 기존 코어셋 기법을 일관되게 능가했다. 예외는 teacher→student 전이 설정(특히 ViT-B/16→ViT-Small)에서 herding이 유리해지는 경우로, 결과적으로 few-medoids는 특히 student를 from scratch로 학습할 때 강하고, 단순하지만 drop-in baseline으로 활용 가능하다는 시사점을 준다.



### i-EXAM: Instructable and Explainable Attack Connectivity Graph Modeler (https://arxiv.org/abs/2607.05888)
Comments:
          In the Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS 2026)

- **Prior Approaches**: 기존에는 ACG(Attack Connectivity Graph)를 포함한 공격 그래프/연결성 모델을 바탕으로 공격 경로를 분석하거나, PDDL로 변환해 공격을 계획 문제로 다루는 연구가 있었다. 다만 관리자 관점에서 복잡한 네트워크 구성요소를 모델링하고, 공격 경로·방어 전략을 함께 고려하며, 결과를 설명 가능한 형태로 제시하는 데는 진입 장벽이 높았다.

- **Core Contribution**: i-EXAM은 SPEAR의 PDDL 컴파일 기반 ACG 추론 위에 “시각화·분석·what-if·설명”을 묶은 운영 도구를 제안한다. Nmap/Wazuh/OpenVAS 등으로부터 자동으로 planning model을 생성하고, impenetrability(공격 경로 존재 여부)와 attack difficulty(최소 비용 공격 경로)를 각각 최적화해 하드닝 전략을 생성하며, LLM이 이를 자연어로 설명한다.

- **Technical Challenges**: 핵심 난제는 (1) 네트워크/취약점 데이터를 형식화해 PDDL 생성까지 자동화하는 것, (2) hardening은 서비스 기능(연결성)을 유지하면서 공격 경로만 차단/비용을 증가시키도록 제약을 세우는 것, (3) 최적/다양한 해의 원인을 설명하는 것이다. i-EXAM은 CVE·구성 데이터를 Structure JSON으로 정리한 뒤 PDDL 생성기로 컴파일하고, fluents/액션/목표를 통해 공격 경로 존재 ↔ plan 존재를 보장하며, 비용 최소화를 위해 보조 fluents로 공격자 시나리오를 선택하도록 구성했다. 또한 counterfactual plan failure 정보를 model restriction 기반 방식으로 수집해 llama-3.1-nemotron-70b-instruct가 자연어 설명으로 전환한다.

- **Empirical Impact**: 실험적으로 FastDownward(A* + LMCut) 기반 계산에서 heuristic이 impenetrability와 attack difficulty 모두에서 연산 시간을 약 50%가량 줄이는 경향이 보고됐다. 또한 다양한 top-k 공격 경로/해 탐색을 통해 실행 불가능한 전략이 있어도 대안을 유지할 수 있어, 관리자 의사결정에 실용성을 더한다. 저자들은 30노드 네트워크 평가와 일반성(도메인 비의존)·확장성(탐색 전략) 관점에서 i-EXAM의 실무 적용 잠재력을 강조한다.



### Harrison.Rad 1.5 Technical Report: A radiology foundation model that can draft reports from images, priors and clinical contex (https://arxiv.org/abs/2607.05880)
- **Prior Approaches**: 기존 방사선 AI는 주로 특정 소견을 탐지해 radiologist의 일부 인지 단계만 보조하는 방식이 많았고, 보고서 작성에 필요한 문맥 통합(병력·이전 검사·진단적 추론)을 end-to-end로 다루기엔 한계가 컸습니다. 또한 대형 범용 비전-언어 모델은 여러 데이터에 강해도 복잡한 임상적 판단에서 도메인 밀도 지식이 부족해 FRCR 같은 인증 수준 평가에서 흔들릴 수 있다는 문제가 드러났습니다. 무엇보다 기존 평가지표가 문장 유사도 중심이어서, 실제 임상적 맞고-틀림(특히 polarity와 진단 부합)을 안정적으로 가르지 못한다는 평가 공백이 지적됩니다.

- **Core Contribution**: 이 논문은 방사선 특화 multimodal large language model HR1.5를 제시해, interleaved 텍스트-이미지 입력을 받아 structured/ unstructured 리포트를 함께 생성하는 것을 목표로 합니다. 핵심은 보고서 작성에 필요한 소견-진단 정렬을 강화하고, 방사선 업무에 맞춘 학습 파이프라인과 임상적으로 정렬된 평가 프레임을 함께 제공한다는 점입니다. 또한 신뢰도(calibrated confidence)와 근거(해당 영역 근거 등) 분석까지 포함해 향후 임상 적용을 위한 책임 있는 평가 방향을 제안합니다.

- **Technical Challenges**: HR1.5가 실제로 “방사선답게” 작동하려면, (1) 도메인에 맞는 언어 생성 능력, (2) 시각-텍스트 정렬의 세밀함, (3) 문맥 기반의 instruction-following까지 한 번에 맞추는 학습 설계가 필요합니다. 이를 위해 3단계 파이프라인(보고서 기반 domain adaptation → 6M 규모의 대조학습에서 curriculum 기반 hard negatives → multi-turn visual question answering fine-tuning)을 사용하고, 방사선에서 중요하지만 표면 어휘에 의존하기 쉬운 구분을 네거티브와 대화 데이터로 “기계적으로” 학습신호화합니다. 여기에 captioning 보조목표는 방사선에서는 효과가 없다는 ablation 결과를 반영해 제외했으며, B200급 학습 인프라에서는 데이터 로딩/캐싱 병목을 재구성해 대규모 학습 효율을 확보했습니다.

- **Empirical Impact**: 평가는 RadGraph-XL 기반 Findings-Diagnosis 점수(ontology 동의어 매칭, polarity-contradiction 탐지)로 임상 정렬을 강화해 RadBench, FRCR 2B Short Case 시뮬레이션(Angoff 기준), ReXGradient, 내부 다중 신체부위 세트, mammography(CBIS-DDSM) 등을 폭넓게 수행합니다. 그 결과 HR1.5는 시뮬레이션 FRCR passing 기준을 충족하는 유일한 시스템으로 보고되며, closed-format 임상 질의에서는 전반적으로 최고 정확도를 보입니다. 또한 explainability(질문 민감 Grad-CAM, attention 분석, confidence estimation)와 리포트 품질을 임상 근거로 평가하는 프레임을 함께 제시해, 단순 자동지표를 넘어 실제 임상 사용을 염두에 둔 평가 체계를 확장합니다.



### Think Before You Grid-Search: Floor-First Triage for LLM Serving (https://arxiv.org/abs/2607.05876)
Comments:
          16 pages, 3 figures

- **Prior Approaches**: LLM 서빙 최적화는 보통 많은 설정을 벤치마킹하고, 목표 지연(예: TPOT, TTFT)이 빗나가면 무거운 프로파일러로 원인을 찾는 방식에 의존해 왔습니다. 그 과정에서 “측정값이 기준선(바닥)과 얼마나 가까운지”를 판단하는 절차가 없어, 같은 벤치마크 숫자가 구현/통신/커널/스케줄링 문제를 동시에 숨길 수 있습니다. 또한 기존 분석 모델은 overlap 같은 가정을 고정해 두 단계를 제대로 구분하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 프로파일링의 역순 규율을 제안하며, “분석적 추정치(floor)가 먼저 나오고 그 잔차(residual)가 작으면 프로파일러를 멈춘다”는 Floor First 절차를 제시합니다. 각 디코드 스텝을 HBM bytes, FLOPs, network bytes, network messages, KV capacity의 5차원 자원 벡터로 모델링하고, 측정값이 [max, sum] 구간의 어디에 놓이는지로 overlap 품질을 저비용 진단합니다. 또한 배치 성장이 진행될 때 어떤 ‘자원 벽’이 먼저 결박되는지(wall ordering)로 배치 대안을 비교해, 단일 포인트 벤치마크의 모호함을 줄입니다.

- **Technical Challenges**: 핵심 난제는 “정확한 floor가 나오기 전에는 residual 해석이 무의미해진다”는 점과, overlap/통신/커널 비용이 섞여 나타나는 실제 운영 조건을 모델에 반영해야 한다는 것입니다. 논문은 자원별 집계 규칙(같은 자원 경합은 더하고, 독립 엔진은 병렬로 겹치게 두는 형태)과, [max, sum] 구간을 통해 overlap을 벤치마크 잔차로 읽어내는 방법을 결합합니다. 더 나아가 통신 상수는 통신 라이브러리/클러스터 실측으로 보정하고, 남는 residual에 대해서만 Nsight 계열을 특정 커널 클래스에 한해 확장(에스컬레이션)하도록 에이전트 기술(skill)까지 포함해 루프를 강제합니다.

- **Empirical Impact**: 사례로 DeepSeek-V3.2 스타일 671B MoE/MLA 모델을 16×NVIDIA H20에서 분석해, H20의 ridge point(~74 FLOP/byte)가 디코드 지향 구조임을 바탕으로 TP16 vs EP+DP 배치 판단 로직을 계산 가능한 형태로 제시합니다. 그 결과 8K 기준에서 TP16 디코드는 KV-capacity에 의해 동시성 약 ~70으로 제한되는 반면, EP16+DP-attention은 동일 배치 지연은 약간 손해를 보더라도 KV-capacity 벽을 약 ~644까지 크게 밀어 장기적으로 처리량(goodput) 우위가 날 수 있음을 보여줍니다. 동일 하드웨어에서 서로 다른 병렬 배치가 실제로 엇갈려 배포된 이유를 “어떤 벽이 먼저 걸리는가”로 설명해, 앞으로의 서빙 의사결정이 실험·주먹구구가 아닌 계산 기반 triage로 이동할 수 있음을 시사합니다.



### Differentially Private Natural Gradient Descen (https://arxiv.org/abs/2607.05866)
- **Prior Approaches**: 차등프라이버시(DP) 학습은 DP-SGD처럼 매 반복마다 그래디언트를 L2-norm으로 등방(clipping)하고 가우시안 잡음을 더하는 방식이 표준이다. 하지만 이런 1차 방법은 손실의 곡률(geometry)을 반영하지 못해, 비조건성(ill-conditioned) 지형에서 지그재그 진동이 심해지고 동일 프라이버시 예산 내 최종 성능이 최적화 효율에 의해 제한된다.

- **Core Contribution**: 이 논문은 Natural Gradient Descent(NGD)를 DP 학습에 실용적으로 결합한 DP-NGD를 제안한다. 핵심은 곡률 기반의 preconditioning 효과를 유지하면서도 DP의 등방적 민감도 제약을 “맞물리게” 설계해, DP-유틸리티 병목을 깨는 것이다.

- **Technical Challenges**: DP에 NGD를 그대로 넣으면 (1) 곡률(Fisher 등) 추정에 추가 프라이버시 예산이 필요하고, (2) DP의 등방 clipping/노이즈와 NGD의 이방적 스케일링이 충돌하며, (3) 곡률이 작은 방향에서 inverse curvature가 업데이트를 폭발시켜 불안정해진다. DP-NGD는 곡률 추정을 private 데이터와 분리해 public auxiliary 데이터에서만 수행하고, F^{-1/2}F^{-1/2}로 whitened space에서 DP 연산을 한 뒤 원공간으로 되돌리며, 곡률 eigenvalue를 동적으로 clamping해 flat direction 폭주를 막는다.

- **Empirical Impact**: 여러 벤치마크에서 DP-NGD는 1차 DP 기준선의 유틸리티 상한을 돌파하며 SOTA 정확도를 달성한다. 특히 동일 프라이버시 예산에서 DP-SGD 대비 최대 10× 수렴 속도 향상을 보고해, 같은 ε에서 더 적은 반복으로 더 낮은 잡음 주입을 가능하게 했다는 점에서 의미가 크다.



### Unsupervised Anomaly Detection of Information Operations Users via Behavioral and Language Patterns (https://arxiv.org/abs/2607.05855)
Comments:
          Accepted at ECML/PKDD 2026

- **Prior Approaches**: 기존 IO 사용자 탐지는 감독/반감독 학습에 의존하는 경우가 많지만, 실제 IO는 빠르게 진화해 라벨 데이터가 현실을 따라가지 못하면서 일반화 성능이 떨어진다. 또 무감독(특히 zero-shot LLM) 접근은 시계열의 복잡한 동적 거동을 충분히 모델링하지 못하거나, IO 계정 간 ‘조율(coordination)’을 과도한 가정으로 두는 한계가 있다.

- **Core Contribution**: 이 논문은 IO 사용자 탐지를 무감독 이상탐지(anomaly detection) 문제로 재정의하고, Temporal Point Process(TPP)로 ‘비정상적인 시간 행동 패턴’을 먼저 포착한다. 여기에 LLM이 사용자 게시 타임라인을 보고 IO/컨트롤 근거를 점수화한 evidence function을 도입해, TPP가 오염된 학습데이터에 의해 흔들리는 부분을 Evidence-based Post-Hoc Adjustment Framework for Anomaly Detection(EPHAD)로 보정한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 학습 시점에 IO 사용자가 소수지만 섞여 contamination이 발생한다는 점과 (2) 시간 행동만으로 부족할 수 있는 언어 패턴 정보를 어떻게 정량 신호로 바꿀지다. TENSOR는 TPP가 산출하는 시간 기반 score를 유지하되, LLM 응답을 “IO account/Control account” 참조문과의 의미 유사도 softmax로 확률화해 evidence로 변환하고, 이를 EPHAD 방식으로 TPP 출력에 후처리 조정해 두 신호를 효과적으로 융합한다.

- **Empirical Impact**: 실험에서 TENSOR는 5개의 실세계 IO 데이터셋에서 기준선들을 유의미하게 능가했으며, 행동·언어 패턴의 기여를 확인하는 ablation에서도 각 구성요소가 탐지 성능에 영향을 주는 것으로 나타났다. 또한 EPHAD 통합 방식 비교, LLM/temperature 민감도 분석을 통해 추론 안정성과 일반적인 적용 가능성을 함께 보여주며, 다이나믹하게 진화하는 IO 탐지라는 실전 문제에 실질적 진전을 제공한다.



### AbICL: In-Context Learning for Antigen-Specific Antibody Affinity Ranking (https://arxiv.org/abs/2607.05846)
- **Prior Approaches**: 기존 항체-항원 결합 친화도 예측은 회귀(regression)로 절대값을 맞추거나, 순위(ranking)로 상대 선호를 학습하는 방식이 주류였다. 하지만 대부분은 각 비교를 독립적으로 처리해 항원 특유의 결합 지형(landscape)에서 오는 맥락 정보를 활용하지 못했다. 또한 사전학습 단백질 파운데이션 모델 기반 점수화도 공통 표현에 기대어 항원별 적응이 제한적이었다.

- **Core Contribution**: 본 논문은 항원-특이적 순위화를 In-Context Learning 관점으로 재해석하고, AbICL(antigen-specific antibody affinity ranking용 ICL 프레임워크)을 제안한다. AbICL은 사전학습 구조 인코더 위에 context ranking head를 두고, 테스트 시 라벨이 붙은 친화도 비교 시연(support demonstrations)에 조건부로 예측을 바꿔 항원별 순위 패턴을 추론한다. 특히 그 과정에서 gradient update 없이 컨텍스트만으로 적응하도록 episodic meta-training을 설계했다.

- **Technical Challenges**: 핵심 난제는 소수의 라벨 비교만으로 항원별 상대 친화도 순위를 안정적으로 추정하는 동시에, 시연과 질의가 같은 항원의 결합 맥락을 공유하지 않을 때도 성능이 유지되게 하는 것이다. AbICL은 support/query를 함께 넣고 Transformer가 self-attention으로 시연 토큰들을 질의에 직접 참조하도록 만들었으며, 시연 라벨 임베딩을 통해 ‘누가 더 강한지’ 정보를 컨텍스트에 각인시킨다. 또한 학습 단계에서 항원 단위 episode를 구성하고 support 크기를 무작위로 샘플링해, 테스트-time에서 다양한 shot 상황에 대응하도록 훈련했다.

- **Empirical Impact**: AbRank 벤치마크 실험에서 AbICL은 데이터 분할과 평가 벤치마크 전반에 걸쳐 기존 ranking/회귀 기반 baseline을 거의 모든 설정에서 일관되게 앞섰다. 분석 결과, 시연의 효과는 (1) 표적 추론 태스크와의 정합도, (2) 분포 이동(distribution shift), (3) fine-grained 친화도 구분 난이도에서 더 커졌고, 단일 고정 순위 함수의 한계를 보완하는 역할이 확인됐다. 또한 episodic meta-training을 제거하면 시연 이득이 크게 줄어들어, 성능 향상이 모델 용량 증가가 아니라 ‘in-context adaptation’ 학습에서 온다는 점을 실증했다.



### Beyond Refusal: A Same-Lineage Study of Aligned and Abliterated LLMs for Vulnerability Analysis (https://arxiv.org/abs/2607.05842)
- **Prior Approaches**: LLM 보조 소프트웨어 보안 평가는 주로 금지/거부(refusal) 여부에 초점을 두거나, 서로 다른 모델 패밀리·서비스를 비교해 안전성 차이를 해석하기 어렵다는 한계가 있었다. 또한 비거부 응답이 실제로 취약점 탐지·정확한 CWE 할당·라인 수준 국소화·패치 적용까지 이어지는지(실행 가능성)까지는 충분히 측정되지 않았다. 그 결과 안전 동작과 엔지니어링 유용성 사이의 상호작용이 분리되지 못했다.

- **Core Contribution**: 이 논문은 같은 계보(same-lineage) 모델에서 ‘안전 상태’를 실험 변수로 분리한다. Aligned(정렬된 instruction-tuned)와 Abliterated(거부-방향이 약화된 refusal-ablated) 상태를 비교해, 소프트웨어 보안 워크플로 전반에서 방어 유틸리티가 어떻게 달라지는지 평가한다. 특히 refusal 비율이 아니라, 비거부 응답의 커버리지·정답 품질·end-to-end 실행 가능성을 함께 분해해 본다.

- **Technical Challenges**: 핵심 난제는 “합법적 코드 리뷰 용어”가 “오용(misuse) 유사 요청”으로 인식될 때 안전장치가 어떻게 반응하는지, 그리고 그 영향이 국소화·수리 단계에서 어떤 형태로 나타나는지를 분리해내는 것이다. 이를 위해 동일한 코드·프롬프트 스키마·디코딩 설정에서 task-depth(탐지→CWE→취약 라인→근본 원인→실행 검증 패치)와 prompt-framing(중립 문구/권한 문맥/사이버 용어 밀도)을 교차 설계하고, 평가 파이프라인은 고정해 비교 혼선을 줄였다. GEMMA와 Qwen의 같은 계보 쌍을 사용해 패턴이 한 모델에 국한되지 않음을 확인했다.

- **Empirical Impact**: 결과적으로 safety state의 비용은 단조(monotonic)하지 않고, task가 코드-접지(code-grounded)·액션 지향으로 깊어질수록 Abliterated가 더 경쟁력 있는 구간이 생겼다. GEMMA의 Java/Vul4J 수리-검증 실험에서는 Abliterated가 usable/적용 성공/컴파일 성공 패치 비율을 각각 67.8%→29.9%, 65.0%→24.9%, 32.8%→9.0%로 더 높였다. 또한 Qwen 쌍에서 vulnerable-line localization은 Abliterated의 line-level F1이 2.08%→3.91%, Top-1 정확도가 4.10%→6.95%로 개선됐다. 논문은 LLM 보안 어시스턴트를 평가할 때 거부 여부뿐 아니라 응답이 정확하고 엔지니어링 워크플로에서 실행 가능한지까지 ‘함께’ 측정해야 한다고 제안한다.



### VisTCP: A Visualization Framework to Construct Knowledge-Graph-Based Representation for Traditional Chinese Painting (https://arxiv.org/abs/2607.05841)
- **Prior Approaches**: 기존 연구는 중국 전통 회화(TCP)를 이미지 분류·분할·객체 검출 중심으로 처리해 왔지만, 회화 속 상징과 사건을 담는 의미적 관계층을 충분히 표현하지 못한다. 자연 이미지용 scene graph generation/visual relationship detection 계열은 대규모 학습 데이터와 비교적 단순한 시각 요소를 전제로 해서, TCP의 고유한 화풍 차이와 희소 데이터에서는 의미 오해와 성능 저하가 자주 발생한다. 또한 지식 그래프나 scene graph를 쓰더라도 정답에 가까운 엔터티·이벤트 해석을 위해 도메인 전문성이 크게 요구된다.

- **Core Contribution**: 이 논문은 TCP에 맞춘 지식 그래프(구조화 표현)를 인력-기계 협업(human-in-the-loop)으로 신뢰 가능하게 구축하도록 하는 시각화 프레임워크 VisTCP를 제안한다. VisTCP는 TCP 전용 의미 분류 체계(semantic taxonomy)를 만들고, 전문가 주석 데이터로 TCP-oriented structured representation model을 학습해 의미 있는 객체와 관계를 자동 추출한다. 이어서 전문가 주석과 모델 예측의 불확실성을 joint embedding visualization으로 드러내 사용자가 자신의 지식으로 반복 수정하도록 한다.

- **Technical Challenges**: 핵심 난제는 (1) TCP의 객체·사건이 현대 자연 이미지와 시각/의미적으로 달라 기존 모델이 의미를 잘못 해석한다는 점, (2) 고대 물체·이벤트의 정확한 식별이 화풍 변동 때문에 어렵다는 점이다. 이를 위해 논문은 전문가 연구로 4개 엔터티와 2개 관계(이벤트·위치)로 구성된 TCP 의미 분류 체계를 도출하고, Mask R-CNN 기반 객체 검출에 TCP 라벨을 이식한 뒤 TDE(Total Direct Effect)로 관계 추론의 편향을 완화한다. 더불어 예측-전문가 예시를 시각·의미 공용 임베딩 공간에 함께 투영해, 라벨 클러스터 이탈·경쟁 라벨 근접·낮은 confidence 같은 단서를 통해 불확실한 항목을 사용자 검증 워크플로로 연결한다.

- **Empirical Impact**: 실험과 사례 연구에서는 ‘청명상하도’의 서로 다른 장면(상단 행렬/목가적 활동/희극 관람)을 구조화 표현으로 구성해 장면별로 사건이 풍부하게 분리·드러남을 보였다. 특히 전문가 인터뷰 6명과 사용 시나리오에서, joint embedding 시각화를 통해 자동 추출의 오류 가능 지점을 확인하고 삭제·수정·추가(엔터티/관계)로 누락과 오인을 보완하는 과정을 관찰했다. 결과적으로 VisTCP는 TCP 의미 이해를 위한 구조화 표현 구축을 더 빠르고 신뢰성 있게 만들며, “모델의 불확실성을 보이게 하고 전문가 피드백으로 반복 개선”하는 인간-AI 협업 설계가 효과적임을 시사한다.



### Tangent classes of matroids and wonderful compactifications (https://arxiv.org/abs/2607.05835)
- **Prior Approaches**: 기존 연구로는 Che26이 루프리스 matroid에 대해 tangent class(접벡터 번들의 K-이론적 대응물)를 rational/KK 형태로 구축하고 주요 성질을 정리했다. 다만 정수(integral) 좌표를 갖는 실제 tangent class를 Feichtner–Yuzvinsky building set에 대해 완전히 “정수 리프트”하는 문제는 남아 있었다. 또한 비실현(realizable) matroid에 대해서도 wonderful model의 결과를 조합적으로 재현하는 연결이 정수 단계에서 매끄럽게 닫히지 않았다.

- **Core Contribution**: 본 논문은 모든 루프리스 matroid M와 top flat을 포함하는 Feichtner–Yuzvinsky building set 𝒢에 대해, 정수 좌표를 갖는 integral tangent class T_{M,𝒢}^{ℤ} ∈ K_{ℤ}(M,𝒢)를 구성한다. 실현 가능한 경우에는 대응되는 wonderful compactification의 tangent bundle 클래스와 특수화(specialize)되며, 비실현의 경우에도 Hilbert series/Chow ring 정보와 Chern-α lower bound 같은 결론이 matroid 자체의 조합적 불변량으로 남는다는 점을 보장한다. 더 나아가 이 작업은 Che26에서 다뤄진 핵심 tangent class 및 성질들을 정수 버전으로 재현한다.

- **Technical Challenges**: 핵심 난점은 (1) reduced 𝒢-nested fan이 완전(complete)일 수 있어 완전 토릭 다양성의 K-이론으로 단순 환원하기 어렵고, (2) rational 클래스의 성질을 정수 K-ring으로 “승격(lift)”할 때 τ-기저 기준 정수 좌표를 보존해야 한다는 점이다. 논문은 먼저 Berget–Eur–Spink–Tseng의 Chern polynomial에서 rational quotient/tangent class를 만든 뒤, one-flat descent 재귀와 ττ-adic associated graded/Feichtner–Yuzvinsky 비교를 통해 정수 representative Q_{𝒢}^{ℤ}와 T_{M,𝒢}^{ℤ}를 구성한다. 실현 가능한 경우에는 K-theoretic blowup formula와 wonderful-model blowup center들을 이용해 K_0(W_{L,𝒢})와의 정수 등형을 맞추며, 정수화가 성질을 잃지 않음을 정리한다.

- **Empirical Impact**: 논문은 Hilbert 관련 정체성인 PK = Hilb^K = Hilb를 rational에서 integral로 확장하고, Hirzebruch–Riemann–Roch를 통해 Chow ring의 Hilbert series를 회수함을 보인다. 동시에 Chern–α(알파) lower bound가 기대한 형태로 성립함을 실증적으로(정리 형태로) 확인하며, 이는 nested-support positivity 논증으로 지지된다. 한편 본문은 Danus라는 AI 수학 추론 에이전트가 사람의 수학적 지도 없이 문제를 먼저 해결하고, 이후 정수화 보완을 이어갔다는 실험 기록까지 포함해 수학 연구에서 agentic AI의 실질적 생산성을 시사한다.



### Decision-Focused Scenario Generation and Selection for Efficient and Robust Grid Dispatch (https://arxiv.org/abs/2607.05830)
Comments:
          10 pages, 12 figures

- **Prior Approaches**: DRO 기반 송배전 운영은 예측 시나리오로 모호성 집합을 만들지만, 기존 시나리오 생성 파이프라인은 주로 예측 정확도에 맞춰 학습되어 버스 간(공간) 상관을 놓치기 쉽습니다. 그 결과 통계적으로 그럴듯해도 실제 운영 비용 관점에서는 애매함 집합이 비효율적일 수 있다는 한계가 있습니다.

- **Core Contribution**: 논문은 DRO 디스패치에 직접 연결되는 decision-focused 생성 프레임워크로, 예측 오차가 아니라 “생성된 시나리오가 유발하는 하류 운영 비용”을 목표로 상관 있는 시나리오를 만듭니다. VAE, GAN, diffusion 등 주류 generative model을 아키텍처에 덜 종속적으로 통합하고, 버스 전반의 joint distribution을 학습해 공간 상관을 반영합니다.

- **Technical Challenges**: 핵심 난점은 서로 다른 generative 패러다임이 생성 메커니즘과 학습 목표가 달라, 하나의 decision-focused 학습 파이프라인으로 통일하기 어렵다는 점입니다. 또한 시나리오 풀을 크게 만들면 하류 DRO 최적화/그래디언트 계산 비용이 급증하므로, 후보 풀에서 decision-relevant 시나리오를 고르는 미분가능(differentiable) scenario selector를 설계해 동일한 학습 루프 안에서 함께 최적화합니다.

- **Empirical Impact**: 실험(사례 연구)에서 제안 프레임워크는 accuracy-oriented 방법 대비 생성 모델 종류에 따라 운영 비용을 0.80%~2.02% 절감하는 성과를 보였습니다. 이는 “예측을 잘하는 것”이 아니라 “DRO 운영에 유리한 시나리오를 만드는 것”을 학습 목표로 삼아야 한다는 실증적 근거를 제공하며, 공간 상관이 중요한 전력 불확실성 예측-최적화 결합 연구에 의미가 큽니다.



### Complementary Roles of Image Classification and Vessel Segmentation in AI-Based Screening for Retinopathy of Prematurity Plus Disease in a Kenyan Preterm Cohor (https://arxiv.org/abs/2607.05825)
- **Prior Approaches**: 기존 ROP(미숙아 망막병증) 자동 선별 연구는 주로 화상 분류기(RGB 분류) 중심으로 진행돼 왔지만, Plus 질환 특성상 “주관적·가변적” 판독 문제가 남아 과잉 의뢰(over-referral) 위험이 커질 수 있다. 또한 안저 혈관 분할(segmentation) 기반 접근은 정밀도를 올릴 잠재력이 있으나, 실제 선별 워크플로에서 분류 성능과의 균형을 맞추기 어렵다는 한계가 보고돼 왔다.

- **Core Contribution**: 본 연구는 케냐 데이터에서 ROP Plus를 눈 단위로 탐지하기 위해 혈관 분할과 분류를 함께 묶는 end-to-end 또는 결합형 파이프라인을 체계적으로 비교한다. 핵심은 “세분화(혈관) 신호”를 직접 활용해 분류기의 민감도는 유지하되, 특이도를 끌어올리는 조합 전략을 제시한 점이다.

- **Technical Challenges**: Plus 판독의 근거가 되는 혈관의 확장·꼬임은 영상 내에서 복잡하게 나타나며, 모델 학습에서는 혈관 분할 품질과 최종 눈 단위 판정의 연결이 관건이다. 연구진은 두 명의 grader가 제공한 혈관 어노테이션으로 분할 학습을 뒷받침하고, patient-grouped nested cross-validation으로 데이터 누수를 줄이며 11가지 구성(분류, multiple-instance learning, multi-task segmentation-classification, segment-then-classify 등)을 비교해 최적 조합을 찾았다.

- **Empirical Impact**: 결과적으로 혈관 분할은 held-out 이미지에서 Dice 0.533, IoU 0.368, sensitivity 0.623, specificity 0.979 수준을 보여 “분할 자체는 가능”함을 확인했다. 분류 단독은 민감도가 높지만 과잉 의뢰가 컸고, 분할과 결합한 모델이 더 특이적이었으며, OR 기반 screen(민감도), AND 기반 confirmation(특이도), probability ensemble(균형)이 각각 장점을 보였다—probability ensemble은 sensitivity 0.692, specificity 0.914, balanced accuracy 0.803으로 분류기 단독을 능가했다. 연구는 아프리카 ROP AI 시스템이 복합 워크플로로 설계하고 prospective multi-site validation을 거쳐야 한다는 실무적 방향성을 강화했다.



### Segmentation before Answering: Pixel Grounding for MLLM Visual Reasoning (https://arxiv.org/abs/2607.05798)
- **Prior Approaches**: 기존 MLLM의 “thinking with images” 계열은 관심 영역을 bounding box(BBox)로 표시한 뒤 크롭해 추론하는 방식이 널리 쓰였습니다. 하지만 BBox는 객체의 불규칙한 형태를 잘 담지 못해 배경 토큰 낭비가 생기고, 겹치는 객체에서는 목표와 주변을 분리하지 못해 semantic interference가 발생할 수 있습니다.

- **Core Contribution**: 이 논문은 Segmentation before Answering(SegAnswer)로, zoom-in의 단위를 BBox가 아닌 pixel-level segmentation mask로 바꿉니다. SegAnswer는 segmentation으로 목표 영역만 정밀 분리해 불필요한 배경·간섭 신호를 줄이고, 분할된 패치가 MLLM의 positional embedding 구조와도 더 자연스럽게 맞물리도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트 지시를 pixel 단위 마스크로 연결하는 grounding 능력과 (2) 분할 마스크를 중간 대화 단계로 쓰는 멀티모달 interleaved 추론을 함께 학습하는 것입니다. 이를 위해 3단계로 진행하며, 먼저 SAM 2.1 기반 mask decoder를 붙여 segmentation 능력을 길러두고, 다음으로 <|seg|> 토큰을 사용해 segmentation을 대화의 도구처럼 넣는 SFT를 수행한 뒤, 마지막으로 ground truth 없이 최종 결과 기반 reward로 DAPO RL을 적용해 추론 전략을 강화했습니다.

- **Empirical Impact**: V*·HR-Bench(4K/8K) 같은 고해상도 지각, MMBench·VisuLogic·MMVP 같은 일반 지각, POPE·Hallusionbench 같은 환각 평가 전반에서 일관된 성능 향상을 보였고, BBox 기반 visual reasoning 대비 특히 고해상도에서 격차가 컸습니다. 또한 RefCOCO/RefCOCO+/RefCOCOg 등 referring segmentation 벤치마크에서도 segmentation 특화 방법을 능가하는 결과를 제시해, SegAnswer의 pixel grounding이 신뢰 가능함을 실증했습니다.



### FORGE: Towards Functional Tool-Use Generalization via Keypoint Trajectory Reasoning (https://arxiv.org/abs/2607.05780)
Comments:
          15 pages, 8 figures, 6 tables

- **Prior Approaches**: 로보틱스의 기존 일반화 연구는 주로 장면/카테고리 수준의 시각 변동이나, 또는 cross-embodiment처럼 로봇 형태를 바꾸는 문제에 초점이 맞춰져 있었다. 반면 도구 사용에서 ‘동일한 기능’을 보장하는 functional generalization은, 도구 모양은 달라도 접촉 지점과 모션이 바뀌어야 한다는 점에서 perception-to-action gap이 커 end-to-end 학습이 쉽게 무너진다.

- **Core Contribution**: 논문은 functional generalization을 ‘도구는 바뀌지만 기능(타격)을 동일하게 수행’하는 문제로 정식화하고, 핵심 난제로 시각 유사성이 행동 공간으로 그대로 전이되지 않는 불일치를 제시한다. 이를 해결하기 위해 FORGE를 제안하며, 기능 추론과 동작 실행을 분리해 먼저 일반화 가능한 2D keypoint trajectories를 예측한 뒤, 제한된 시연으로 로봇 행동에 grounding한다.

- **Technical Challenges**: 기여를 위해서는 (1) 도구별 외형에 과적합되지 않으면서도 (2) 접촉 지점과 시간에 따른 운동 구조를 담는 intermediate representation을 찾아야 했다. 저자들은 affordance images, human video prompts, 2D keypoint trajectories를 비교했고, keypoint trajectories가 function 표현력과 action groundability를 가장 잘 균형한다는 실험 결과로 선택했으며, stage1은 action-free data로 conditional flow matching 예측, stage2는 action-labeled data로 conditional flow matching 기반 execution policy 학습(그리고 예측 오차 견고화용 perturbation)으로 구성했다.

- **Empirical Impact**: 일곱 가지 도구의 hitting-function 벤치마크에서 FORGE는 unseen tools에 대해 state-of-the-art 대비 평균 success rate를 2배 이상(2X+) 끌어올리며 시뮬레이션과 real world 모두에서 일관된 성능을 보였다. 특히 end-to-end visuomotor 정책들은 unseen 도구의 올바른 hitting region 정렬에 실패하지만, FORGE는 keypoint 기반 중간 계획이 접촉 위치 접근을 구체적으로 안내해 실패를 줄이는 방식으로 의미를 입증한다.



### LEGATO 2: Toward Multimodal Sheet Music Recognition and Understanding (https://arxiv.org/abs/2607.05769)
Comments:
          23 pages. Equal contribution: Guang Yang and Brian Siyuan Zheng

- **Prior Approaches**: 기존 광학 음악 인식(OMR)은 보통 한 장의 악보 이미지를 단일 입력으로 보고 처리해, 입력 길이가 길어질수록 성능과 확장성이 급격히 떨어질 수 있었다. 또한 대부분은 기호 중심의 전사에 초점을 맞춰 제목이나 주석 같은 텍스트가 섞인 경우를 생성형으로 다루기 어려웠다. 결과적으로 장거리 문서 처리와 텍스트 포함 전사의 결합이 한계로 지적됐다.

- **Core Contribution**: 이 논문은 악보 이미지에서 기호 표기와 의미 정보를 함께 뽑아내는 파이프라인 Legato 2를 제안한다. Legato 2는 시스템 단위로 순차 처리하는 최초의 대규모 OMR 모델로, 페이지를 무차별 이미지로 취급하지 않고 표기 읽기 흐름을 따라 임의로 긴 입력으로 스케일링한다. 더불어 제목과 주석 같은 내장 텍스트를 포함하는 기호 전사를 생성하는 최초의 OMR 모델을 제시한다.

- **Technical Challenges**: 핵심 과제는 (1) 긴 악보에서 시스템 구조를 안정적으로 분할하고, (2) 시스템 간 문맥을 유지하며, (3) 기호와 텍스트를 동시에 정확히 생성하는 것이다. Legato 2는 시스템-level segmentation과 autoregressive vision-LM을 결합해, 국소적인 기보 디테일과 스코어 구조를 함께 포착하도록 설계했으며, system-by-system 순차 모델링으로 긴 입력에서도 일관성을 확보한다.

- **Empirical Impact**: 여러 데이터셋에서 Legato 2는 기존 state of the art를 일관되게 능가하는 성능을 보였다. 또한 생성된 상징적 전사가 비주얼 입력과 함께 frontier language model의 해석을 돕는다는 점을 실증해, 촘촘한 음악 문서를 이해하는 downstream 성능까지 끌어올렸다. 결과적으로 Legato 2는 OMR 자체뿐 아니라 악보 이해 전반에서 새로운 state-of-the-art를 정립했다.



### Data-dependent Evaluations for Budgeted Submodular Maximization (https://arxiv.org/abs/2607.05759)
Comments:
          Extended version of a paper that will appear in ESA 2026 conference

- **Prior Approaches**: 부분모듈러 최대화는 NP-hard라서 기존 연구는 주로 최악의 경우 근사비율(approximation factor)로 알고리즘을 평가해 왔다. 그 결과 MSMK(카트 예산 제약)처럼 더 일반적인 문제에서는 최적 대비 실제 성능 격차를 인스턴스별로 직접 측정하기 어렵다. 최근의 data-dependent upper bound 시도도 MSMC 중심이거나 MSMK에 바로 적용되지 못하고, MSMK용 기존 상한은 상대적으로 단순해 타이트함이 제한됐다.

- **Core Contribution**: 본 논문은 MSMK에 대해 최적해의 함수값을 인스턴스에 맞춰 더 촘촘히 상계하는 새로운 data-dependent upper bound를 제안한다. 제안하는 상한은 slicing 전략과 removing 전략의 두 축으로 구성되며, 이 상한이 이론적으로 최적해를 지배(dominates)하고 기존 상한보다 더 타이트함을 보인다. 또한 여러 base set을 함께 반영하도록 선형계획(Linear Program)으로 변환해 상한 품질을 추가로 끌어올린다.

- **Technical Challenges**: 핵심 난점은, 상한을 구성할 때 남은 원소들로부터 얻을 수 있는 최대 추가 이득과(그리디 밀도 기반) 이미 선택된 base set을 “어떻게 제거해” 예산을 정확히 맞출지(제거로 인한 손실 상계) 사이의 결합을 엄밀하게 다뤄야 한다는 점이다. 논문은 marginal gain과 marginal density(및 cutoff density) 구조를 이용해 두 전략을 연속 함수로 설계한 뒤, 그 결과를 LP 형태로 정식화해 다수 base set까지 통합한다. 이를 통해 계산 복잡도는 정렬 기반으로 유지하면서도 상한이 더 강해지도록 구현한다.

- **Empirical Impact**: 실험에서는 maximum coverage, revenue maximization, feature selection 등 실제 데이터셋 기반 작업에서, 제안한 상한이 해의 품질을 최적해에 얼마나 가깝게 “인증(certify)”하는지 더 잘 보여준다고 보고한다. 특히 단순한 worst-case 근사비율 대신, 인스턴스별로 더 타이트한 upper bound를 제공함으로써 실용적 의사결정에 필요한 신뢰 구간 역할을 한다. 결과적으로 MSMK 문제를 다루는 최적화/추천/데이터마이닝 계열에서, 해의 근접성을 정량화하는 평가 체계가 한 단계 강화될 것으로 기대된다.



### When Should LLMs Search? Counterfactual Supervision for Search Routing (https://arxiv.org/abs/2607.05752)
Comments:
          20 pages, 10 figures. Accepted at the FAGEN Workshop at ICML 2026

- **Prior Approaches**: 검색을 내장한 언어모델은 외부 근거로 장꼬리 지식 등을 보완할 수 있지만, 모든 질문에 검색이 항상 이득은 아니다. 기존 연구는 검색이 유용한 경우를 찾거나(선택적 retrieval, confidence/복잡도 기반 트리거) tool 호출 자체의 정확도를 평가하는 데 초점이 있었고, 언제-not-to-call을 인스턴스 단위 성공 관점으로 직접 학습하기는 어려웠다.

- **Core Contribution**: 이 논문은 “검색 필요 여부”를 인스턴스 레벨의 search-routing 문제(NO_SEARCH vs SEARCH)로 정식화하고, 같은 질문에 대해 no-search와 forced-search의 결과를 비교해 outcome-based oracle을 만든다. 이 오라클을 평가 기준뿐 아니라 학습 신호로도 사용해, 필요한 경우에만 검색으로 경로를 바꾸도록 SFT와 Preference Optimization을 함께 학습한다.

- **Technical Challenges**: 핵심 난점은 검색 유용성을 사람이 라벨링하지 않고도 일관된 감독을 구성하는 동시에, 오라클에서 제외되는 UNSOLVED(둘 다 실패)처럼 원인이 섞인 케이스를 학습에 잘못 끌어오지 않는 것이다. 저자들은 no-search/forced-search의 페어 결과로 안정적인 라우팅 타깃만 구성하고, UNSOLVED는 진단 서브셋으로 남긴 채 first-action(첫 턴) 의사결정만 최적화하도록 학습 데이터를 설계했다.

- **Empirical Impact**: PopQA와 KUQ(거짓 전제/모호성)에서 모델별로 search 경계가 다르며, 학습 전에는 over-search와 under-search가 동시에 나타남을 보여준다. 오라클-eligible 예제에서 Gemma E2B는 macro-F1이 0.7082→0.8235, Qwen3.5-4B는 0.7053→0.8365로 개선됐고, 분석에서는 UNSOLVED 잔여 오류가 모델 용량, retrieval budget, 근거 활용, 정책 행동 등 서로 다른 병목임을 분해해 준다.



### Unicode TAG-Block Concealment of Tool-Metadata Payloads in the Model Context Protocol: An Approval-View Fidelity Gap Across Three Independent Server Implementations (https://arxiv.org/abs/2607.05744)
Comments:
          15 pages, 4 figures, 7 tables, 5 listings. Real-protocol proof-of-concept, 8 techniques across 3 independently developed MCP server libraries with 32 of 32 cross-library outcome cells agreeing, and 0 of 25 baseline false positives on a benign corpus. Data, harness, and fail-closed verifier released as a supplementary artifact

- **Prior Approaches**: MCP는 tools/list 핸드셰이크로 도구 메타데이터(이름·설명·JSON input schema)를 받아 승인 이후 모델 컨텍스트에 넣는 구조라, 도구 메타데이터를 이용한 prompt injection(툴 포이즈닝) 위험이 꾸준히 지적돼 왔다. 기존 연구와 공개 취약점은 주로 “메타데이터가 모델에 도달한다”는 점을 보여주거나, 특정 제품/서버 1곳에서의 공격·방어를 다뤄 어떤 클라이언트 방어가 실제로 얼마나 막는지, 그리고 그 실패가 어떤 근본 원리에서 오는지까지는 체계적으로 측정하지 못했다.

- **Core Contribution**: 이 논문은 승인 화면(사람이 보는 렌더링)과 모델이 받는 원본 바이트 전달 경로가 일치하지 않아 생기는 ‘콘실먼트 인코딩(concealment encoding)’ 단일 메커니즘을 분리해 설명한다. 특히 Unicode TAG 블록(U+E0000~U+E007F)은 어떤 주류 단말·채팅·IDE 렌더러에서도 표시되지 않는 반면, 모델 토크나이저에는 그대로(바이트 단위로) 전달될 수 있음을 모델-무관·프로토콜-무관 분석으로 예측한다.

- **Technical Challenges**: 핵심 과제는 “사람이 승인 버튼 누르기 전 보는 텍스트”를 우회하면서도 “실제로 모델 컨텍스트에는 같은 페이로드가 도달”하는 인코딩을 찾고, 이것이 실제 MCP 클라이언트/서버 구현에서도 재현되는지 검증하는 것이다. 저자들은 진짜 JSON-RPC/stdio MCP 프로토콜을 사용하는 proof-of-concept로 5개 메타데이터 표면에서 8가지 구체 기법을 구현하고, (1) 모델 컨텍스트 도달 여부, (2) 문자열 매칭 기반 샌티저 회피 여부, (3) 승인 화면 렌더 회피 여부, (4) re-approval(재승인) 강제 여부를 프로토콜 레벨에서 결정적으로 측정했다.

- **Empirical Impact**: 실험 결과 8/8 기법이 모델 컨텍스트에 페이로드를 주입했지만, 4/8은 샌티저를 회피했고 승인 화면까지 함께 회피한 것은 TAG 블록 기반 1/8뿐이었다(논문이 예측한 메커니즘과 일치). 또한 MCP의 re-approval 강제는 0/8에서만도 발생하지 않았으며, 동일한 결과가 서로 독립적인 Python MCP 서버 라이브러리 3종에 대해 32개 교차 조합 모두에서 재현되어 프로토콜 수준 성질임을 뒷받침한다.



### The Balkanization of Execution-Security Research for AI Coding Agents: Isolation, Access Control, and Time-of-Check-to-Time-of-Use Vulnerabilities (https://arxiv.org/abs/2607.05743)
Comments:
          18 pages, 15 figures, 6 tables. Systematizes 39 execution-security papers (2023-2026) into 17 verified categories. Machine-readable corpus and verification script released as a supplementary artifact

- **Prior Approaches**: 기존 연구는 샌드박스 격리, capability·접근제어, 정책 집행, TOCTOU(상태 점검-행동 경합), MCP 위협, 실행 코드 정적분석 등 실행 레이어의 안전성 이슈를 각각 따로 다뤄 왔다. 하지만 서로 다른 논문들이 같은 문제를 다른 어휘로 취급하면서, 성능·견고성 비교나 상호 검증이 거의 이뤄지지 않아 “무엇이 최선인가”를 판단하기 어렵다. 또한 정책 집행 연구들은 종종 정책 작성자를 신뢰한다고 가정해 실제 운영에서 발생하는 정책 오류(denylist 취약성 등)까지 포괄하지 못한다.

- **Core Contribution**: 이 논문은 2023~2026년 공개된 실행 보안(Execution Security) 관련 논문 39편을 17개 범주로 systematization of knowledge(SoK) 형태로 정리하고, 각 분류가 원문에서 직접 확인되도록 검증 프로토콜을 마련했다. 더 나아가 NIST NVD에 근거해 생산 에이전트 하네스에 직접 영향을 준 공개·패치 CVE 4건을 확인해, 위협 모델이 실제 사고와 연결됨을 보여준다. 마지막으로 범주 간 교차 읽기를 통해 단일 논문이 다루지 못한 5가지 공백(평가지표의 부재, 정책 집행의 취약성 재평가 부재, TOCTOU와 MCP의 분리, honest policy author 가정, scope-creep 미해결)을 도출하고 이를 연구 의제로 제안한다.

- **Technical Challenges**: 핵심 난제는 “모델의 결정”이 아니라 “에이전트가 실행 환경으로 실제 무엇을 할 수 있게 되느냐”라는 시스템 경계를 일관된 기준으로 측정하고 비교하는 것이다. 저자들은 기존 서술의 요약에 의존하지 않고 각 논문의 arXiv 초록 페이지에서 제목·저자·기여를 직접 대조해 분류 신뢰성을 확보했고, CVE 역시 NIST NVD에서 직접 대조해 사실성을 고정했다. 또한 범주별 원인 구조를 교차 분석해 TOCTOU와 MCP가 사실상 같은 state-validation 문제의 변형이라는 점처럼, 기술적 공통분모를 다른 용어로 흩어진 채로 놓치는 문제를 짚는다.

- **Empirical Impact**: 실증적으로는 denylist 기반 접근제어의 실패율이 69%~98%까지 넓게 관측되지만, 격리(샌드박스) 연구가 같은 적대적 setting에서 방어를 재평가하지 않는다는 ‘평가 공백’이 드러난다. 또한 Claude Code의 신뢰 대화/프로젝트 로드 구간에서 사용자 승인 이전에 코드 실행 또는 데이터 유출이 가능했던 CVE들을 확인해, 실행 경계가 실제 취약점으로 이어짐을 뒷받침한다. 결과적으로 이 논문은 실행 보안을 독립된 연구 트랙으로 재정의하고, 격리-접근제어-정책집행을 통합적으로 비교·재평가하는 후속 연구 방향(5개 갭)을 제시함으로써 분야의 실질적 정렬을 촉진한다.



### SCOReD: Student-Aware CoT Optimization for Recommendation Distillation (https://arxiv.org/abs/2607.05734)
Comments:
          31 pages

- **Prior Approaches**: 추천을 생성형 추론으로 바꾸려는 흐름이 늘면서, RL 훈련의 선행 단계로 chain-of-thought(CoT) distillation이 중요해졌다. 하지만 추천 도메인의 교사 LLM은 답을 바꾸기보다 같은 결정을 반복 검증하는 ‘불확실성 재확인’ 패턴이 강해, 원문 CoT를 그대로 supervised fine-tuning 하면 학생이 불필요하게 장황하고 수정이 없는 추론을 그대로 모사하는 문제가 생긴다.

- **Core Contribution**: SCOReD(Student-Aware CoT Optimization for Recommendation Distillation)는 추천에 맞춘 CoT 최적화 프레임워크로, 교사 CoT를 추천 단계별로 분해한 뒤 학생 모델의 attention과 확률 신호로 각 구간의 중요도를 추정한다. 그 다음 학생 관점에서 KEEP/REWRITE/FUSE/PRUNE 편집 중 보상이 가장 큰 선택을 동적으로 적용해, 중복 검증은 줄이고 정보 밀도가 높은 구간은 보존하면서 학생 출력 분포에 맞춘다.

- **Technical Challenges**: 핵심 난제는 (1) 추천 CoT가 정답 고정이 아니라 잡음 섞인 로그 라벨과 모호한 선호의 협상 결과라서 ‘압축하면 성능이 떨어지는’ 오프데이터(out-of-distribution) 위험이 크고, (2) 학생의 생성 분포에서 무엇을 유지/수정해야 이득이 나는지 측정하기 어렵다는 점이다. SCOReD는 </think>에서 각 세그먼트로 흘러가는 attention을 기여도 대리 신호로 쓰고, 학생의 answer log probability 향상과 편집 후 segment length/perplexity를 함께 고려한 보상함수로 구간별 편집을 선택해 이 문제를 완화한다.

- **Empirical Impact**: Amazon Beauty 데이터로 0.6B 학생 모델을 실험한 결과, 원시(비압축) teacher trace로 학습한 SFT 대비 NDCG 1.56%, Recall@5 1.9% 향상과 함께 추론 길이 27.3% 감소를 동시에 달성했다. 또한 파싱 실패율도 46% 내외로 줄여, 학생이 불필요한 반복 검증을 덜 생성하면서도 순위 품질을 유지·개선한다는 점을 실증했다.



### Plainbook: Data Science, in Plain Languag (https://arxiv.org/abs/2607.05717)
Comments:
          12 pages

- **Prior Approaches**: 기존 Jupyter Notebook은 재현성과 검증을 돕지만, 본질적으로 코드에 접근 가능한 사용자에게만 강점이 있습니다. 특히 자연어로 분석을 지시받거나 AI가 코드를 만들어주는 상황에서는 코드가 ‘숨은 영역’이 되어 비개발자에게 verifiability와 extensibility가 크게 약화됩니다. 또한 셀을 임의 순서로 실행할 수 있어 hidden state 문제가 생기며, 이는 재실행/재현 실패를 부르기 쉽습니다.

- **Core Contribution**: Plainbook은 노트북의 중심을 코드가 아니라 자연어 셀 설명으로 옮기고, 코드는 자동 생성되더라도 설명이 우선적으로 보존되도록 설계합니다. 실행은 위에서 아래로 선형(linear)하게 고정해 Jupyter의 hidden state를 제거하고, 각 셀은 고정된 결과 상태를 낳아 검증 가능성을 높입니다. 사용자는 코드 대신 값과 데이터 관찰을 바탕으로 셀/노트북의 구현이 설명과 일치하는지 확인할 수 있습니다.

- **Technical Challenges**: 핵심 과제는 자연어 모호성과 AI 생성 오류 때문에 ‘설명→코드’ 일치가 깨질 수 있다는 점이며, 사용자가 코드 없이도 이를 확인·수정할 방법이 필요하다는 것입니다. Plainbook은 체크포인팅 커널로 실행 상태를 스냅샷 캐싱하고 선형 실행의 멱등성(idempotent)을 보장해, 셀 단위 검증과 재실행을 효율적으로 수행합니다. 또한 셀 validation(설명과 코드 일치 여부를 AI가 판정), 전역 validation(노트북 전체 일치·위험 연산 점검), 그리고 cell tests(데이터 preparation로 상태를 단순화해 값 기반 검증)를 결합해 코드를 직접 이해하지 못해도 검증 흐름을 만들었습니다.

- **Empirical Impact**: 논문은 특정 실험 수치가 아니라, 오류가 실제로 발생하는 시나리오(예: 중복 집계로 ‘연도 수’ 계산 실패)에서 validation과 cell tests가 어떻게 진단·수정으로 이어지는지 구체적으로 보여줍니다. 이를 통해 비개발자도 자연어 설명을 기준으로 결과를 검증하고, 필요하면 다른 AI 백엔드로 교차검증할 수 있음을 강조합니다. Plainbook은 공유 가능한 데이터 분석 문서의 접근성을 높여 재현·확장·검증의 장점을 ‘코드 이해가 아닌 값 이해’로 확장하는 점에서 의미가 있습니다.



### IMR: Iterative Mode-World Weighted Regression for Multi-Agent Trajectory Prediction (https://arxiv.org/abs/2607.05705)
- **Prior Approaches**: 기존 multi-agent motion prediction은 예측 기반과 anchor 기반으로 크게 나뉜다. 예측 기반(QCNet, Forecast-MAE)은 복잡한 상황에서 mode collapse(다양성 붕괴)가 발생하기 쉽고, anchor 기반(MTR, TNT)은 이를 완화하는 대신 예측 정확도가 떨어지는 경향이 있다. 또한 QCNeXt 같은 proposal-refinement 디코딩은 초기 제안 궤적이 크게 틀리면 refinement가 오프셋 보정을 충분히 못 하는 문제가 지적된다.

- **Core Contribution**: 이 논문은 mode collapse와 정확도 저하의 trade-off를 동시에 다루기 위해 prediction-based 프레임워크에 mode-world weighted regression loss를 제안한다. 이 손실은 mode-wise 및 world-wise 회귀를 함께 가중해 학습하며, mode 다양성 유지와 함께 world ranking 정확도 및 top-1 confidence를 개선하는 데 초점을 둔다. 아울러 반복적 디코딩으로 제안 궤적의 초기 오류에 덜 민감한 구조를 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다중 에이전트 궤적에서 정답 모드가 여러 개인데도 학습 중 특정 모드로 수렴해버리는 문제와 (2) joint world confidence를 포함한 world ranking을 정확히 학습하는 문제다. 이를 위해 그래프 attention network 기반으로 에이전트-맵 및 상호작용을 동적으로 인코딩하고, 모드 회귀는 winner-takes-all 전략으로 선택 오차(ADE/MDE)를 기준화해 손실을 설계한다. 디코딩은 iterative decoder를 도입해 SS(세그먼트) 단계를 순차 생성하되, 각 단계 출력이 이전 단계 offset이 아니라 절대 위치 좌표가 되도록 하여 누적 오차 전파를 줄인다.

- **Empirical Impact**: 실험은 Argoverse 2에서 6초 예측 setting으로 수행됐고, 제안 방법은 다른 모델 대비 평균 BrierMinFDE6에서 이전 SOTA QCNeXt보다 0.06p 향상되며 1위를 기록했다. 단일 에이전트 벤치마크에서도 LOF와 비교해 경쟁력 있는 성능을 보였다. 시각화와 손실 비교 결과, 기존 world-wise 회귀에서 나타나는 mode collapse를 mode-world weighted regression loss가 완화하면서 정확도까지 함께 개선함을 확인했다.



### Depression Symptoms and Relational Patterns in 187k ChatGPT Histories (https://arxiv.org/abs/2607.05685)
- **Prior Approaches**: 기존 CSCW/HCI 연구는 우울·불안 같은 고통을 온라인에서 익명 공개하거나 동료·커뮤니티를 통해 지지받는 경로, 그리고 형식적 돌봄 접근성이 낮을 때의 대체 경로를 집중적으로 다뤘다. 다만 ChatGPT 같은 LLM은 ‘사람’이 아니라 사적인 1:1 대화 인프라로서, 어떻게 감정적 의존·경계·전문가 전환이 나타나는지에 대한 실증 근거가 부족했다.

- **Core Contribution**: 이 논문은 PHQ-8 설문으로 우울 증상 강도를 나눈 뒤, 기부된 ChatGPT 대화 187,093건을 766명의 대화 이력과 연결해 ‘비공식 상시 지원 인프라’ 관점에서 사용 양상을 분석한다. 특히 우울 증상이 높은 사용자가 어떤 유형의 대화(정신건강·관계·외로움·자기초점·고도의 자기노출·지원요청)를 더 자주/어떤 맥락에서 가져오는지, 그리고 LLM이 전문적 전환을 얼마나 확장하는지 비교한다.

- **Technical Challenges**: 핵심 과제는 (1) 대규모 대화에서 주제·고도 자기노출·지원요청 같은 맥락을 자동 라벨링하고, (2) 사용자 언어만으로는 부족할 수 있는 ‘어시스턴트의 응답 스타일’까지 함께 해석하며, (3) 언어 기반 예측이 임상 선별로 이어질 만큼 충분히 강한지 검증하는 것이다. 연구진은 gpt-4o-mini로 탐색적 라벨을 구성하되 임상적 진단/검증 라벨로 취급하지 않았고, AUROC 기반 PHQ 예측은 정규화 로지스틱 회귀와 반복 교차검증으로 평가해 성능이 스크리닝에 못 미침을 확인했다.

- **Empirical Impact**: 우울 증상이 상대적으로 높은(PHQ≥10) 참여자는 정신건강·대인문제·외로움·부정적 자기평가 관련 비중이 더 크고, 1인칭 대명사·절대주의 표현이 늘며, 야간(23:00~04:59)과 월 단위 반복 패턴으로 나타났다. 또한 이들은 고도의 자기노출/지원요청 맥락에 더 자주 진입하지만, 전문적 전환(professional redirect) 비율은 집단 간 견고한 증가가 관찰되지 않았다; 언어만으로 한 PHQ≥10 분류 AUROC는 0.591로 단지 우연 이상 수준에 그쳐 “민감도 높은 조기 선별” 근거로 보기 어렵다. 결론적으로 이 결과는 LLM이 실제로는 임상 도구가 아니라 맥락 의존적 비공식 지원 인프라로 활용되고 있으며, 안전·경계 설계는 세션/히스토리 기반으로 재고돼야 함을 시사한다.



### Beyond Accuracy: How Humans Evaluate Legally Correct but Socially Controversial Legal Advice from Machines (https://arxiv.org/abs/2607.05680)
- **Prior Approaches**: 기존 연구는 알고리즘에 대한 반감(algorithm aversion)이나 자동화 불신이 나타나면, 법률처럼 사회적으로 민감한 영역에서 AI 조언을 사람이 덜 받아들일 것이라고 가정해 왔습니다. 특히 ‘법적으로 맞아도’ 사회적 논쟁성이 큰 조언일수록 출처(인간/AI)에 따른 수용성 차이가 커질 수 있다는 우려가 제기돼 왔습니다. 다만 해당 영향이 단일 방향으로 관찰되는지, 혹은 상반된 심리 경로가 상쇄될 수 있는지는 충분히 검증되지 않았습니다.

- **Core Contribution**: 이 논문은 중국 성인 3,348명을 대상으로 사전 등록된 설문 실험을 수행해, 동일한 법률 조언을 AI 시스템 또는 인간 변호사에게 귀속하고(출처), 여기에 추론(reasoning) 제공 여부를 조작했습니다. 결과적으로 AI로 귀속된 조언은 ‘지각된 타당성/합리성’에서 순효과가 없었는데, 이는 단순한 algorithm aversion 가설과는 어긋나는 관찰입니다. 동시에 매개분석을 통해 상반된 작동 메커니즘이 공존해 순효과가 0에 가까워질 수 있음을 제시합니다.

- **Technical Challenges**: 핵심 기술적(방법론적) 도전은 출처 효과가 ‘없어 보이는’ 경우에도 심리적 경로가 서로 다른 방향으로 움직여 상쇄될 수 있다는 점을 통계적으로 분해하는 데 있습니다. 논문은 매개분석(mediation analyses)을 통해 AI 귀속이 객관성(objectivity) 인식은 높이지만 포괄성(comprehensiveness)과 특별한 사정에 대한 주의(attentiveness)를 낮춘다는 두 효과가 각각 지각된 합리성에 반대 방향으로 기여함을 보여줍니다. 또한 추론을 함께 제공하면 출처와 무관하게 합리성 인식이 크게 증가하며, 그 이유는 다시 객관성 인식 강화로 설명된다는 점을 정성 응답이 뒷받침합니다.

- **Empirical Impact**: 실험 결과는 ‘AI 법률 고문에 대한 대중의 반응이 자동화 자체에 대한 경직된 태도에 의해 결정된다’기보다, 객관성 대 맥락 민감성 같은 규범적 기대의 균형으로 형성된다는 관점을 강화합니다. 따라서 AI 추천 시스템 설계에서는 정확성만큼이나, 추론 제공 방식과 맥락(특별 사정) 반영 정도가 사용자 평가에 상반된 영향을 줄 수 있음을 고려해야 합니다. 알고리즘 반감 이론을 단순 인과로 보기 어렵게 만들며, normatively salient domain에서의 신뢰 형성 메커니즘을 구체화하는 데 의미가 있습니다.



### RPAM: A Principled Metric for Evaluating Associations in Language Models with High Predictive Validity in Downstream Outputs (https://arxiv.org/abs/2607.05679)
Comments:
          14 pages

- **Prior Approaches**: 기존 생성형 LM 편향 분석은 주로 생성된 텍스트의 결과(다운스트림)에서 연관성을 측정하는 방식이 많았다. 하지만 생성 문장은 모델마다 크게 달라 특화된 평가 데이터셋이 필요해져 다른 LM으로의 일반화가 제한된다. 한편 업스트림 평가는 임베딩이나 continuation probability 같은 기반 신호를 보지만, 기존 업스트림 지표가 실제 생성 텍스트에서 관측되는 연관성과 강하게 맞물린다는 근거는 부족했다.

- **Core Contribution**: 이 논문은 생성형 LM의 연관성(association)을 업스트림에서 평가하는 Relative Probability Association Metric(RPAM)을 제안한다. RPAM은 두 자극 간 상대적 연관성을 softmax로 정규화해, 텍스트 생성/디코딩과 무관하게 개념-속성 연관을 정량화한다. 또한 RPAM 측정치가 인간의 암묵·명시 연관성과 생성 텍스트의 다운스트림 편향 측정까지 연결되는지를 검증하는 평가 프레임워크를 함께 제시한다.

- **Technical Challenges**: 핵심 과제는 업스트림 신호로부터 얻은 연관성이 실제로는 다운스트림 행동(생성 결과)과 얼마나 일치하는지 입증하는 것이었다. 이를 위해 RPAM은 타깃을 템플릿에 삽입한 뒤, 속성 단어들에 대한 continuation probability를 계산하고 특정 속성 집합 내에서 상대적으로 재정규화해 비교 가능성을 확보했다. 또한 n-gram/문장 같은 다양한 타깃 표현에 맞춘 서로 다른 템플릿을 사용하고, WEAT/SC-WEAT 계열을 포괄하는 형태로 암묵 연관성과 valence(유쾌/불쾌, 감정)까지 확장했다.

- **Empirical Impact**: 검증 실험에서 RPAM은 WEAT-WS 기반 암묵 연관성 10개를 재현했으며, 인간의 명시적 연관성(WS-353, Bellezza, SST2)과도 높은 상관/분류 성능을 보였다. 특히 다운스트림 작업과의 일치성 실험에서는 Mistral/GPT-2 계열에서 Spearman’s ρ=0.73 및 SST2 F1 0.74 이상 같은 결과로 생성 텍스트 기반 편향 신호를 잘 반영했다. 업스트림 지표로서의 실용성과 일반화 가능성을 보여준 만큼, 앞으로 편향 완화·규제 대응을 위한 평가 표준으로 활용될 잠재력이 크다.



### What Do AI Agents Actually Change? An Empirical Taxonomy of Mutation Patterns in Performance-Improving Pull Requests (https://arxiv.org/abs/2607.05666)
- **Prior Approaches**: 기존 Genetic Improvement(GI) 계열 연구는 사람이 작성한 패치에 기반한 mutation taxonomy로 연산자(변이 연산) 공간을 구성해 왔습니다. 하지만 AI coding agent의 동작은 블랙박스에 가까워, 실제로 에이전트가 코드를 어떻게 변환하는지에 대한 경험적 지도가 부족했습니다. 특히 성능 최적화 PR은 전체의 1%도 되지 않아, 에이전트 행동을 직접 관찰할 데이터가 희소했습니다.

- **Core Contribution**: 이 논문은 AIDev-pop에서 216개 성능 개선 PR로부터 1,254개의 성능 관련 diff hunk를 분해하고, Even-Mendoza et al.의 18-category syntactic mutation taxonomy로 에이전트 변이를 분류합니다. 이를 통해 에이전트 성능 PR에서 주로 등장하는 변이 패턴이 기존 GI 코퍼스와 크게 다름을 정량화합니다. 또한 에이전트 시스템과 성능 전략에 따라 변이 범주가 서로 다른 부분집합으로 활성화된다는 경험적 단서를 제공합니다.

- **Technical Challenges**: 핵심 과제는 “어떤 변경이 성능 관련 변이인지”와 “18개 범주에 대한 일관된 라벨링”을 신뢰도 있게 자동화하는 것입니다. 논문은 1차 LLM 필터로 성능 관련 hunk만 남기고, dual-LLM intersection으로 상충 라벨을 제외하는 방식으로 오분류를 줄였습니다. 이어서 cateogry boundaries를 성능 PR 맥락에 맞게 재정의(예: object_creation에 import 추가 포함)하고, LLM-as-a-judge 정확도를 캘리브레이션해 보수적(하한) 카운트를 산출했습니다.

- **Empirical Impact**: 결과적으로 name_modification(37.0%), object_creation(26.4%), type_change(22.7%)가 성능 PR 변이를 주도하며, 기존 GI 코퍼스에서 84%를 차지하던 no_change는 이 데이터에서는 완전히 사라집니다. 에이전트별로도 서로 다른 ‘mutation vocabulary’가 관찰되어, 예를 들어 Devin은 name_modification 비중이 높고 GitHub Copilot은 type_change가 두드러지는 식으로 프로파일이 갈립니다. 연구는 이런 맥락 의존적 priors가 SBSE에서 operator space를 목표 전략·에이전트 조건에 맞춰 대략 18개 범주에서 약 5개 수준으로 좁히는 실무적 기반이 될 수 있음을 제안합니다.



### Physics-Regularized Machine Learning for Proprioceptive Vehicle Localization Using Onboard Sensors (https://arxiv.org/abs/2607.05663)
Comments:
          Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026). 8 pages, 4 figures

- **Prior Approaches**: 기존 GNSS-denied 환경의 차량 위치추정은 HD map 기반, 다중센서 융합, dead reckoning 계열로 나뉜다. 특히 IMU 중심의 고전적인 Bayesian 필터는 노이즈 누적으로 시간이 지날수록 드리프트가 커지는 한계가 있고, ML 보정 방식은 센서 잡음은 잘 다루더라도 물리적 일관성과 조건 전반 일반화가 부족해질 수 있다. 최근엔 differentiable filtering이 등장했지만, 표준 onboard proprioceptive 센서만으로 실시간 localization을 달성하면서 물리-시간 정합성을 체계적으로 분석/강화한 접근은 상대적으로 적었다.

- **Core Contribution**: 이 논문은 onboard 센서만으로 차량 포즈를 추정하는 Physics-Regularized Machine Learning for Localization(PRML2) 프레임워크를 제안한다. PRML2는 transformer 기반 ML이 차량 동역학 상태를 예측하고, differentiable EKF가 이를 포즈 추정으로 결합하되 EKF를 통한 end-to-end 학습이 물리 기반 정규화(physics-regularized learning) 역할을 하도록 설계했다. 결과적으로 예측이 차량 운동모델과 시간적으로 모순되지 않게 만들어 정확도와 주행 조건 전반 일반화를 함께 노린다.

- **Technical Challenges**: 핵심 난제는 (1) ML이 만드는 동역학 예측이 차량 운동 제약을 위반하지 않도록 하면서 (2) EKF 학습에서 수치적으로 안정적인 미분 가능 파이프라인을 유지하는 것이다. 이를 위해 PRML2는 physics guard layer로 속도·가속도·각속도 등 동역학 한계를 클램핑하고, 불확실성 추정을 위한 variance까지 일관되게 스케일링한다. 또한 differentiable EKF에서 Jacobian 및 수치 불안정(혁신 공분산 역행렬 등)을 Cholesky 분해와 Joseph form 같은 안정화로 완화해 그래디언트가 ML 모델로 안정적으로 전파되게 했다.

- **Empirical Impact**: 저자들은 publicly available 데이터셋에서 ML-enhanced onboard odometry의 성능 상한을 분석하고, PRML2가 localization 정확도에서 우수하며 실시간 실행도 가능함을 보인다. 더 나아가 저마찰(low-friction) 주행 조건을 위한 새로운 데이터셋을 공개해 해당 도메인의 일반화 평가 기반을 확장했다. 전반적으로 PRML2는 GNSS가 불안정하거나 끊기는 상황에서도 저비용 onboard 센서만으로 강건한 위치추정을 지향하며, learning+physics priors 결합의 실용성을 실험적으로 뒷받침한다.



### Do It Right! A Methodology for Successful NLP System Developmen (https://arxiv.org/abs/2607.05644)
Comments:
          Pre-submission draft

- **Prior Approaches**: 기존 임상 NLP는 개별 알고리즘(구문분석, 개체명 인식, 의미 역할 라벨링 등) 중심으로 학습 자료가 구성돼 있어, 실제 정보추출 시스템을 “프로젝트로” 관리하는 관점이 약했다. 또한 LLM 이후에는 API로 쉽게 구조화된 결과를 얻는 것처럼 보이지만, 실제로는 소스에 없는 내용을 그럴듯하게 생성(환각)하거나 누락, 프롬프트 표현에 따른 불일치가 생길 수 있다. 즉 성패는 모델 성능뿐 아니라 요구사항 정의·검증·변경관리 같은 소프트웨어 공학 위험 관리에 달려 있다.

- **Core Contribution**: 이 논문은 임상 임상기록에서 언어처리로 정보를 추출하는 NLP 프로젝트에 대해 Systems Development Life Cycle(SDLC) 단계별 절차를 제시한다. 특히 텍스트 생성(summarization) 같은 과제는 제외하고, 정보추출(information extraction)에 맞춰 Planning–Analysis–Design–Implementation–Testing–Deployment–Maintenance의 흐름을 구체화한다. “성공적인 추출”을 위해 SDLC를 통해 실패 가능성을 체계적으로 낮추려는 실무 지침을 제공한다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 관심 개념이 실제 임상 텍스트에 존재하는지, (2) 해당 내용을 충분한 정확도로 추출 가능한지의 타당성 검증이며, LLM에서는 환각 때문에 일반적인 벤치마크를 신뢰하기 어렵다. 논문은 이를 해결하기 위해 대표 코퍼스에 대한 수동 확인, 개념 시트(concept sheet)로 정의·단위·값 범위를 고정, 그리고 semantic ambiguity(용어가 여러 개념에 매핑)·contextual ambiguity(주장/시점/경험 주체 등)를 고려한 설계와 검토 전략을 권한다. 또한 문서 선택(비용 절감용 프리필터링, 계층화 표집, 표본 크기)과 주석 설계(가이드라인·포함/제외 기준, 참조표준이 시스템이 보는 데이터와 일치하도록 제한)를 SDLC에 통합해 재현성과 해석 가능성을 높인다.

- **Empirical Impact**: 이 글은 특정 단일 성능수치보다는 임상 정보추출 프로젝트가 실패하는 전형적 원인을 SDLC로 흡수할 수 있음을 강조하며, LLM 기반일수록 ‘잘되는 것처럼 보이는 착시’를 줄이는 절차적 가치가 크다고 설명한다. 실무적으로는 개념 정의 드리프트, 변경통제 부재, 위험(Feasibility/accuracy/cost) 과소평가 같은 요인을 문서화와 합의 기반 검토로 관리하게 만든다. 또한 고급 기저세포암 같은 사례에서처럼 완전 자동화가 어려운 경우 ‘고리콜 NLP 선별 후 수동 해석’ 또는 구조화 대체까지 선택지를 제공함으로써, 팀이 데이터·도메인 특성에 맞게 검증 가능한 개발 경로를 잡는 데 의미가 있다.



### EvalLoop: A Methodology for Evaluation-Driven Iterative Improvement of Business AI Systems (https://arxiv.org/abs/2607.05638)
- **Prior Approaches**: 기존 LLM 평가는 주로 모델을 고정한 뒤 벤치마크 점수로 순위를 매겨 ‘우승 모델’을 고르는 model selection 방식에 머물렀다. 그 결과는 품질 측정에 그치고, 실제 프로덕션에서 무엇을 고쳐야 하는지 진단 신호가 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 평가를 ‘개선 피드백 루프’로 바꾸는 EvalLoop를 제안한다. 평가를 dimensional metric grouping, failure mode classification, structured iteration workflow로 구성해, 품질 저하의 원인을 찾아 다음 변경을 설계하도록 돕는다.

- **Technical Challenges**: 핵심 난제는 (1) 품질을 이해관계자가 납득하는 차원들로 분해해도, 그 차원이 실제로 같은 개입 경로로 움직이는지(interventional validity) 확인해야 한다는 점이다. 논문은 차원 내부가 통계적으로 잘 뭉치는지만 보지 않고, 시스템 변화를 줬을 때 해당 차원 지표들이 같은 방향으로 변하는지로 검증하며, LLM judge의 편향은 cross-provider 패널과 failure mode 분류, 다중 집계로 완화한다.

- **Empirical Impact**: 판매 인텔리전스 브리핑 생성 케이스에서 10개 모델·5개 차원·3회 반복으로 분석했으며, 차원 진단 결과 hallucination 실패의 69%가 prompt-induced interpretation error로 드러났다. 이를 반영한 프롬프트 수정으로 최고 모델의 전체 성능이 82.6%→94.6%로 상승했는데, 개선은 진단된 차원(Content Accuracy +16.8pp, Synthesis Power +26.4pp)에 집중됐다. 또한 차원 프로파일 기반으로 배치 후보를 좁힌 뒤 소규모 blind human gate로 검증해 전체 리뷰 부담을 94% 줄이면서도 배포 관점의 다기준 선택을 현실적으로 해결했다.



### Safe Bayesian Optimization with Counterfactual Policies (https://arxiv.org/abs/2607.05620)
Comments:
          10 pages main text, 20 pages total

- **Prior Approaches**: 기존 Safe Bayesian optimization(SafeOpt)은 관측된 제약 q(x)≥0을 기준으로 목적함수를 최적화하며, 온라인 conformal prediction을 결합해 제한된 위반률로 안전성을 보장한다. 하지만 제약이 기준선(standard-of-care) 정책의 반사실 결과(counterfactual)에 의존하면, 그 결과는 관측되지 않아 전형적인 SafeOpt식 안전 보증을 그대로 적용하기 어렵다. 또한 관련 연구들은 관측 잡음이나 확률적 제약으로의 재구성에 집중했지, 추정 오차가 제약 자체에 내재된 반사실 기반 안전 최적화는 상대적으로 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 기준선 정책의 반사실 결과를 추정해야 하는 설정에서, 반사실 안전 제약을 안전하게 만족시키는 SafeOpt-CPC(SafeOpt with Counterfactual Policy Constraints)를 제안한다. 핵심은 conformal prediction으로 반사실 기준선 결과의 유효한 불확실성 구간을 만들고, 그 구간을 제약 계산에 통합해 사용자가 지정한 위반률 이하로 constraint violation이 발생하도록 설계한 점이다. 아울러 공변량 shift(데이터 분포 변화) 유형에 따라 calibration 데이터를 재가중해 보증을 유지하는 방법도 함께 제시한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 반사실 기준선 결과가 매 시점 관측되지 않기 때문에, 단순 point estimate로 제약을 계산하면 위반률이 보증된 α를 초과할 위험이 있다는 것이다. 논문은 split conformal prediction으로 상단(one-sided) 구간 U^{soc}(x_soc) 형태의 보수적 추정치를 만들고, 이를 통해 SafeOpt의 안전 집합 산정 및 다음 행동 선택에 반사실 추정 불확실성을 반영한다. 또한 가중 교환가능성(weighted exchangeability)을 복원하기 위해 inverse propensity scoring 또는 unlabeled test data 기반 reweighting을 적용하고, changepoint/drift 같은 비정상성에 대해서는 가중치 재계산·time-decay 등 적응 절차를 마련한다.

- **Empirical Impact**: 실험은 Zhang et al.(2024)에서 사용한 화학 반응 시뮬레이터와 MovieLens 데이터에서 수행했으며, 비교군으로 일반(비안전) Bayesian optimization과 반사실을 완전히 아는 oracle을 둔다. 결과적으로 제안 방법은 목적을 개선하면서도 지정된 안전 제약을 요구 위반률 범위에서 충족함을 보였다. 더불어 다양한 제약 구성(추정항 1개/2개 등)과 공변량 shift 하에서의 적응 성능, 그리고 mis-specification(모델/가정 불일치)에 대한 sensitivity analysis까지 포함해 실제 의사결정 안전성에 대한 신뢰도를 높인다.



### BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension (https://arxiv.org/abs/2607.05614)
Comments:
          Accepted at the 19th European Conference on Computer Vision (ECCV), 2026

- **Prior Approaches**: 기존 문서 이해 연구는 DLA와 KIE를 다뤄왔지만, 대부분은 고품질 라벨이 풍부한 언어(영어 중심)나 제한된 스키마에 의존해 저자원 언어로의 전이를 어렵게 만들었다. FUNSD·XFUND 같은 벤치마크는 형태 이해를 진전시켰지만, Bangla 정부 양식에 필요한 세밀한 폼 엔티티·공간 관계·키-값 구조를 포괄하는 공개 데이터는 부족했다.

- **Core Contribution**: 이 논문은 Bangla 정부 양식의 DLA와 KIE를 동시에 평가할 수 있는 벤치마크 BaFCo를 제안한다. 다중 페이지 정부 폼 200개를 모아 26개 세부 엔티티 타입과 관계 라벨을 구성하고, 5개 조밀도 축약 엔티티 세트도 함께 제공해 모델 성능을 ‘세밀함’ 관점에서 비교 가능하게 했다.

- **Technical Challenges**: 핵심 기술적 난제는 Bangla 폼에서 요구되는 (1) 세밀 엔티티 경계의 정확한 위치 지정과 (2) 키-값 및 필드 간 관계를 일관된 규칙으로 라벨링하는 작업이다. 저자들은 바운딩 박스 규칙·관계 제약·모호 케이스를 포함한 상세 가이드라인과 다단계 검수(코헨 κ≈0.974)를 통해 주관성을 줄이고, 이후 MLLM 평가를 위한 validator 기반 출력 스키마 검증 파이프라인을 구축했다.

- **Empirical Impact**: ChatGPT·Gemini·Claude·Qwen·Kimi 계열의 flagship MLLM을 zero-shot 및 chain-of-thought(CoT) 프롬프트로 평가한 결과, Bangla 폼에서 특히 DLA의 세밀 엔티티 국소화가 취약했고, 엔티티를 거칠게 묶을수록 mAP이 크게 개선됐다(예: Gemini 3 Pro에서 0.1177→0.2646 수준). 반면 KIE는 전반적으로 훨씬 높은 F1(예: Bangla에서 Gemini 3 Pro F1=0.848)로 나타났으며, 언어 영향은 DLA보다 KIE에서 더 뚜렷해 ‘최선 모델’이 언어별로 달라질 수 있음을 보여준다.



### To Retain or to Adapt? Generalizing Continual Learning (https://arxiv.org/abs/2607.05609)
- **Prior Approaches**: 연속학습(Continual Learning, CL) 연구는 주로 catastrophic forgetting(치명적 망각)을 막는 데 초점을 맞춰 왔고, “평생 학습자는 Joint-Task Learning(JTL) 해처럼 모든 과거 지식을 유지해야 한다”는 가정이 자주(때로는 명시 없이) 깔려 있습니다. 이 관점은 비정상 환경에서 적응을 늦출 수 있다는 점이 덜 다뤄졌습니다.

- **Core Contribution**: 본 논문은 유지(retention) 중심 전제를 뒤집고, CL을 Average Lifelong Error(ALE)를 최소화하는 온라인 최적화 문제로 재정의합니다. 또한 Transfer Efficiency를 통해 Instability(과거 경험 충돌로 인한 편향)와 Transient Error(새 태스크를 처음부터 학습하는 최적화 비용) 간의 긴장을 수치화하고, 조건 하에서 임계 태스크 지속시간(Critical Task Duration)을 닫힌형태(closed-form)로 제시합니다.

- **Technical Challenges**: 핵심 난제는 “유지하면 좋을 때/해가 될 때”를 이론적으로 구분하는 동시에, 편향이 생기는 정적(stationary) 상황을 다루는 것입니다. 논문은 환경-학습 동역학의 상호작용을 온라인 최적화 틀로 정리하고, 선형 모델과 신경망에서 성립하는 온건한 수렴 조건 하에 편향이 양(+)이 되면 warm-start 이점이 최적화 부담으로 전환된다는 임계시간을 도출합니다.

- **Empirical Impact**: 이론 예측은 연속 이미지 분류와 강화학습 벤치마크에서 검증되며, 유지 편향이 적응 비용으로 전환되는 시나리오가 관찰됩니다. 더 나아가 예측 가능한 시퀀스의 온라인러닝으로 CL을 확장해 Predictive Continual Learning을 제안하고, Window 알고리즘이 JTL과 ITL을 동시에 넘어 분포 드리프트가 통제된 환경에서 우수함을 보입니다.



### Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO) (https://arxiv.org/abs/2607.05585)
- **Prior Approaches**: 기존 얼굴 기반 유전질환 연구는 주로 2D 이미지를 CNN으로 바로 질환(증후군)을 예측하는 방식이 많았고, 일부는 3D 정보를 추가해 성능을 높였지만 여전히 ‘질환 단위’ 분류가 중심이었습니다. 이 접근은 데이터 편향(인종·연령·표본 수)에 민감하고, 왜 그런 결론이 나왔는지 임상적으로 설명하기가 어려워 신뢰와 채택에 제약이 있습니다.

- **Core Contribution**: FaceMesh2HPO는 질환 분류가 아니라 Human Phenotype Ontology(HPO) 기반의 ‘표현형(phenotype) 항목’ 예측으로 문제를 재구성해, 임상 추론과 연결되는 해석 가능성을 목표로 합니다. 또한 HPO 트리 구조(교차 링크 제거) 위에 계층적 모델을 배치하고, 비가시/미완전 라벨 상황을 phenotype 수준에서 다루어 의료 현장에 더 가까운 분류 프레임을 제안합니다.

- **Technical Challenges**: 핵심 난제는 HPO 라벨이 희소하고 존재 여부만 확정된 weakly labeled 특성(미기재=부재로 단정 불가)과, HPO의 계층·의존 구조로 인해 단일 모델 학습이 복잡해지는 점입니다. 논문은 (1) 124명의 임상의가 2D 얼굴에 대해 present/absent/uncertain를 포함해 107개 HPO(부모 항목 확장 포함) 라벨을 재정의하고, (2) 2D→3D face mesh(478 랜드마크)와 계단식(cascading) PointNet 트리를 학습하되, Integrated Gradients로 중요하지 않은 포인트를 제거하는 feature elimination으로 하위 항목 학습 난도를 낮추는 방식으로 대응합니다.

- **Empirical Impact**: 3D mesh, 얼굴 외곽(outline), 인구통계 메타데이터를 포함한 최고 성능은 AUROC가 약 0.55~0.89 범위로 나타났고, leaf보다 상위(parent) 노드에서 성능이 더 높게 관측됐습니다. 외부 독립 테스트에서는 disorder 간 일반화 폭이 달라 성능 변동성이 확인되었으며, 특히 드문(rare) leaf 항목에서 한계가 남아 데이터 다양성과 라벨/특징 선택 전략 개선이 필요하다는 결론을 제시합니다.



### ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modelin (https://arxiv.org/abs/2607.05583)
Comments:
          8 Pages. Accepted at ICANN 2026

- **Prior Approaches**: 기존 장문 언어모델은 Transformer의 self-attention을 중심으로 발전해 왔지만, 긴 컨텍스트에서 계산·메모리 비용이 급격히 커지는 비효율 문제가 남아 있다. 이를 완화하려고 attention을 선형화/커널화하거나 Hyena, S4, Mamba처럼 state-space 계열로 바꾸는 연구가 이어졌지만, 대체로 여전히 attention 기반 계열에서 크게 벗어나지 못했다.

- **Core Contribution**: 이 논문은 attention 대신 damped resonator의 인과적(causal) resonant field mixing으로 시퀀스를 처리하는 ResonatorLM을 제안한다. 토큰열을 구동되는 1차원 잠재장으로 보고, attention dot product를 resonator의 감쇠 공진 커널 기반 causal 함수로 대체해 학습 시 병렬 경로와 디코딩 시 고정 크기 상태를 함께 유지한다.

- **Technical Challenges**: 핵심 과제는 (1) 학습/프리필은 O(n log n) 수준의 병렬 연산으로 처리하면서 (2) autoregressive 디코딩에서는 키-밸류 캐시처럼 길이에 따라 커지는 메모리를 피하는 것이다. 저자들은 동일 커널 계열을 사용해 학습·프리필에는 causal FFT convolution을 적용하고, 디코딩에는 헤드당 고정 크기 recurrent state 업데이트로 전환했으며, causality 검증(접두부 오차)과 반감기(half-life) 분포 같은 물리 기반 진단으로 구조적 정합성을 확인했다.

- **Empirical Impact**: 실험에서 6M 파라미터 규모 matched 설정 기준, 32K 토큰에서 decode 속도는 Transformer 대비 6.47x 향상됐고 WikiText(정확도 55.32%→61.31%)로 품질도 개선됐다. 또한 long-context 길이가 길어질수록 학습·프리필 속도와 디코딩 이점이 강화되며, 커널 전용 kernel-tail 벤치마크에서는 8K/32K에서 각각 440.29x/575.86x의 알고리즘적 속도우위를 보고했다.



### Whose fairness? Structural concentration in AI bias research (https://arxiv.org/abs/2607.05574)
Comments:
          27 pages, including 5 composite figures comprising 16 individual figures. Code is available at this https URL, and the interactive atlas is available at this https URL

- **Prior Approaches**: 기존 AI 편향(bias) 연구는 정의·벤치마크·디베이싱(debiasing) 프레임워크를 중심으로 진행되며, 대체로 이러한 기준이 보편적일 것이라는 가정 위에서 방법을 검증해 왔습니다. 하지만 공정성 정의가 지역·언어·제도 맥락에 따라 달라질 수 있는데도, 연구 커뮤니티의 구성(누가, 어디서, 누구와)이 얼마나 편중돼 있는지는 충분히 계량화되지 않았습니다.

- **Core Contribution**: 이 논문은 AI bias 연구가 구조적으로 특정 국가·기관·저자에 집중돼 있음을 692편 논문(5개 테마 도메인) 기반의 서지분석과 시맨틱 클러스터링으로 보여줍니다. 특히 ‘General Fairness & Bias Mitigation’ 도메인이 가장 큰 인용·정의·벤치마크 공급원이며, 이 도메인의 편중이 전체 분야로 전파될 수 있다는 점을 실증적으로 연결합니다.

- **Technical Challenges**: 분야 경계를 객관화하고 의미 기반 도메인 분류를 해야 했기 때문에, Sentence-BERT 임베딩과 UMAP/HDBSCAN으로 문헌을 클러스터링한 뒤 수기 라벨과의 정렬을 검증했습니다. 또한 인용을 외부(글로벌)와 내부(동일 코퍼스 내 의존) 두 렌즈로 나눠, 영향력이 소수의 논문에 과도하게 쏠리는지(예: 글로벌 인용 중앙값 9, 평균 93.5)를 평가했습니다.

- **Empirical Impact**: 결과적으로 연구 생산과 협업 네트워크는 미국이 전 도메인에서 주도하며, Global South(저소득·중소득 국가)는 저자·협업 측면에서 크게 배제되는 패턴이 나타났습니다. 저자·국가·기관의 집중 지표와 인용 꼬리현상(소수의 고인용 논문이 분야를 좌우)이 결합돼, 좁은 맥락에서 개발·검증된 디베이싱 방법이 다른 인구와 설정으로 일반화되지 않을 위험을 경고합니다. 논문은 이러한 분야 구조를 연속적으로 모니터링할 수 있는 interactive atlas도 제공합니다.



### Harnessing Generative Image Models for Training-Free Primitive Shape Abstraction (https://arxiv.org/abs/2607.05568)
Comments:
          13 pages, 9 figures, 3 tables

- **Prior Approaches**: 기존 프리미티브(초기형상) 기반 3D 추상화는 학습 기반이거나(primitive 파라미터 예측) 순수 최적화 기반(기하 기준으로 분할·피팅)으로 나뉜다. 학습 기반은 학습 분포/정준 방향(canonical orientation)에 강하지만, 범용 범주·임의 자세에 취약하다. 반면 최적화 기반은 category-agnostic일 수 있어도 분할이 의미적 일관성을 잃어(예: 의자 다리가 의미 있는 한 부품으로 뭉치지 않음) 분해와 피팅이 서로 결합된 한계를 보였다.

- **Core Contribution**: 이 논문은 3D 학습 없이(training-free) 2D 파운데이션 모델의 ‘부품(semantic parts) 이해’를 3D 프리미티브 분해로 끌어오는 파이프라인을 제안한다. 멀티뷰 렌더를 비전-언어 모델로 의미 부품과 컬러 매핑을 만든 뒤, 생성형 이미지 모델로 컬러 코드 부품 마스크를 그려 3D로 재투영하고, 각 부품에 superquadric을 고전적 최적화로 피팅한다. 결과적으로 category-agnostic, orientation-invariant 특성을 갖고 learned parameter 없이 동작한다는 점이 핵심이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 여러 뷰에서 전역적으로 일관된 부품 분해를 만들고(색 매핑/부품 개수·정의의 글로벌 합의), 재투영 과정에서 그 의미 경계를 유지한 뒤, 비선형 비합성 문제인 superquadric 피팅이 로컬 미니멈에 빠지지 않게 하는 것이다. 논문은 ‘분석(부품·색 JSON) 단계’와 ‘생성(컬러 마스크)’ 단계를 분리해 뷰 간 컬러 일관성을 확보하고, 컬러 기반 클러스터링으로 노이즈를 정리한 뒤, ApproxMVBB 기반 초기화와 parallel multi-start, 그리고 L-BFGS로 비선형 최적화를 안정화했다. 또한 좌표계 정규화/제약된 latent parameter 변환으로 유효한 변형 형태(taper/bending) 탐색을 관리한다.

- **Empirical Impact**: HumanPrim과 Toys4K에서 Chamfer distance(CD) 기준 모든 비교 방법 중 최저 성능을 보이며, 객체당 평균 5–9개의 primitive로 가장 낮은 표면 충실도를 달성했다(예: HumanPrim CD=0.079, Toys4K CD=0.093). 특히 ground-truth segmentation으로 분할만 바꿔보는 ablation에서 IoU가 크게 개선되어, 현재 병목이 primitive fitting이 아니라 part segmentation 품질임을 실증했다. 이는 생성형 이미지 모델 성능이 올라가면 재학습 없이도 파이프라인 정확도가 연쇄적으로 향상될 수 있음을 시사한다.



### Prompt Robustness Is Task-Dependent: Comparing Objective and Belief-Style Questions in LLM Evaluation (https://arxiv.org/abs/2607.05554)
- **Prior Approaches**: 기존 LLM 평가는 단일 프롬프트 응답을 모델의 값·신념을 대표하는 지표로 간주하는 경향이 있었지만, 형식·표현·보기 제시 방식이 조금만 바뀌어도 정답/선택이 흔들릴 수 있다는 점이 반복적으로 지적돼 왔다. 특히 정치·가치 설문형 테스트는 답변이 라벨, 선택지 순서, 강제 선택 문구, 프레이밍에 의해 좌우될 수 있어 결과 해석이 취약하다는 경고가 있었다.

- **Core Contribution**: 이 논문은 객관식(고정 정답) 질문(Type-I)과 주관식/설문형(의견·가치·동의 정도) 질문(Type-II)에서 프롬프트 견고성(prompt robustness)이 같은 방식으로 나타나는지 비교한다. 6개 데이터셋과 4개 instruction-tuned 모델을 대상으로, 의미는 유지하되 wording, framing, format, 라벨/선택지 제시 방식 등을 다양하게 바꿔 응답 일관성을 측정해 질문 유형별로 어떤 불안정성이 다른지 드러낸다.

- **Technical Challenges**: 핵심 과제는 ‘프롬프트 변화가 의미 변화가 아니라는 점’을 통제하면서도, 설문형에서는 외부 정답이 없어 무엇이 불안정인지 정의해야 한다는 점이다. 저자들은 모든 항목에 대해 여러 의미보존 프롬프트 변형을 만들고, deterministic decoding(temperature 0)로 추출 노이즈를 제거한 뒤, 객관식에서는 canonical 선택지로, 설문형에서는 모델이 낸 선택을 정규화해 변형 간 ‘자기 일관성’으로 측정했다. 또한 binomial generalized estimating equation(GEE)로 item 간 상관을 고려해 모델·데이터셋·프롬프트 범주 및 상호작용의 효과를 함께 추정했다.

- **Empirical Impact**: 실험 결과 주관식/설문형(Type-II)이 객관식(Type-I)보다 평균적으로 일관성이 낮았고, 그 격차는 모든 모델에서 반복됐다. 불안정성을 가장 크게 만드는 프롬프트 범주는 답변 제시 방식 중에서도 option order였으며, 의미적 동치/패러프레이즈·철자 잡음·논리적 동치 같은 범주는 상대적으로 안정적이었다. 더 나아가 dataset type과 prompt category의 상호작용이 매우 크게 나타나 ‘단일 평균 견고성 점수’로는 모델 비교를 왜곡할 수 있으므로, 설문형 평가에서는 프롬프트 변형에 대한 일관성 체크를 표준 설계로 포함해야 한다.



### The yes-no bias of large language models reflects answer order and wording, not shifts in moral judgmen (https://arxiv.org/abs/2607.05552)
- **Prior Approaches**: 기존 연구는 LLM이 도덕적 질문에서 yes/no 같은 이진 판정을 할 때 ‘yes–no bias’가 생기며, 그 크기가 논리적으로 무관한 표현 변화에도 흔들린다고 보고해 왔습니다. 하지만 한 번의 고정된 프레이밍에서는 ‘no’라는 단어가 논리 판정, 토큰/표면 문자열, 마지막 선택지 옵션을 동시에 대표해 분해가 불가능하다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 교차 대칭화(crossed symmetrization)를 포함한 심리측정 배터리로, 논리적으로 동일한 답변을 유도하는 서로 다른 질문 포맷을 전부 엮어 모델의 내부 도덕 척도(연속형 stance, θ)를 복원합니다. 그 결과 frontier model들은 포맷이 바뀌어도 θ가 거의 일정(교차-포맷 불일치 0.12~0.21, ±1 축)하게 유지되지만, 작은 오픈 가중치 모델들은 서로 다른 방식으로 실패한다고 제시합니다.

- **Technical Challenges**: 핵심 난제는 forced yes/no 결과에 섞여 있는 ‘순서(order)’와 ‘단어(lexical)’의 표면 편향을 분리해 내는 것이었습니다. 논문은 동일 딜레마를 동사/선택지 순서/라벨(yes/no 단어, 임의 라벨 등)까지 교차시키고, 관측된 이진 편향을 좌우 수평 오프셋 m(프레이밍 민감도)과 기울기 기반 s(도덕적 결정도)로 로지스틱 모델에 적합해 구성요소를 분해했습니다.

- **Empirical Impact**: 실증적으로는 강제 이진 판정에서 나타나는 큰 yes–no bias가 주로 ‘마지막 출력 선택지 쪽으로 끌리는 순서 편향(인간의 고전적 primacy와 반대)’과 ‘특정 단어 쪽으로의 렉시컬 끌림’의 합으로 설명되며, verdict에 직접 붙는 논리적 편향은 frontier model들에서 거의 0에 가깝다고 보고합니다. 또한 extended reasoning(추론 확장)은 척도 자체의 포맷 불변성을 더 강화하고, 그렇지 않으면 표면 편향이 관측값을 지배할 수 있음을 보여 측정 설계 관점에서 ‘한 프레이밍 숫자’의 해석을 경고합니다.



### Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks (https://arxiv.org/abs/2607.05545)
- **Prior Approaches**: 기존 연구는 LLM이 또래 다수·전문가 라벨 같은 사회적 단서에 의해 정답을 틀린 답으로 바꾸는 현상을 ‘conformity’로 해석해 왔습니다. 그러나 표준 conformity 프롬프트는 ‘화자 존재’와 ‘반복된(주장된) 답’이라는 두 단서를 동시에 섞어, 어느 요인이 수정에 더 결정적인지 분리 측정하기 어려웠습니다.

- **Core Contribution**: 이 논문은 화자를 제거하되 주장된 답을 고정하는 통제 조건인 no-source(무화자) 설정을 제안합니다. 이를 통해 LLM 수정 중 ‘사회적 영향(화자 귀속)’보다 ‘반복된 답 텍스트 자체가 만드는 기준선(floor)’이 얼마나 큰지 먼저 측정할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술적 과제는 프롬프트에서 화자와 반복 답을 분리해도 결과가 표기/옵션/샘플링 요인으로 흐려지지 않게 설계하는 것입니다. 저자들은 두 번 읽기 arbitration 프로토콜과 greedy decoding(샘플링 노이즈 최소화), 그리고 paraphrase·open-ended(선택지 숨김)·invalid-label·evidence형 컨테이너(검색 레퍼런스, corrupted log) 같은 다수의 대조 실험으로 반복 텍스트의 ‘증거처럼 보이는 효과’를 분해해 측정했습니다.

- **Empirical Impact**: 6개 오픈웨이트 LLM과 7개 QA·추론 데이터에서, 무화자 조건만으로 initially correct의 66.5%가 harmful revision을 보였고 plain re-ask는 10.3%에 그쳤습니다. 또한 모델이 뒤집히면 대체로 높은 확신으로 틀린 답을 채택하며(평균 argmax 확률 0.92), 온도 조절 같은 간단한 재보정만으로 원래 답으로 되돌리기 어렵다고 보고했습니다. 결론적으로 conformity 벤치마크는 먼저 speaker-free floor를 측정한 뒤 그 위의 증가분(화자 귀속 효과)을 별도로 보고해야, 반복 텍스트를 사회 영향으로 착각하는 위험을 줄일 수 있습니다.



### Self-Review Reinforcement Learning (SRRL) with Cross-Episode Memory and Policy Distillation (https://arxiv.org/abs/2607.05541)
Comments:
          9 pages, 2 figures

- **Prior Approaches**: 기존 강화학습(RL)은 LLM을 환경의 보상으로 학습시키지만, 실제 에이전트 설정에서는 보상이 희소하거나 지연돼 어떤 추론 단계가 성공/실패를 만들었는지 크레딧 할당이 어려웠다. SFT는 사람 예시에 맞춰 “모방”하는 데 강하지만, 실패 피드백을 반영해 행동을 수정하는 내재 메커니즘이 약해 반복 실수를 그대로 학습하기 쉽다. RLVR은 검증 가능한 보상으로 상호작용 학습을 가능하게 하지만, 보상이 왜 틀렸는지를 구조화해주지 않아 안정적인 수정이 에피소드 간 지속되기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 Self-Review Reinforcement Learning(SRRL)이라는 학습 프레임워크를 제안한다. 각 RL 에피소드에 명시적 self-review 단계를 넣어, 1차 답이 실패하면 모델이 무엇이 잘못됐는지 점검하고 2차 시도를 더 나은 행동으로 연결한다. 또한 self-review를 단순 추론 시 반성(Reflexion류)으로 두지 않고, policy gradients로 최적화한 뒤 selective distillation으로 “기저 정책”에 내재화해 이후 에피소드에도 개선이 남도록 만든다.

- **Technical Challenges**: 핵심 기술적 도전은 희소·지연 보상 상황에서 self-review가 단순한 말하기가 아니라 실제로 행동 개선을 유도하도록 학습 신호를 설계하는 것이다. SRRL은 1차 성공이 아닌 실패 조건에서만 self-review와 2차 재시도를 활성화해 reward hacking 및 초기 학습의 불안정(오프폴리시 데이터 지배)을 줄인다. 더 나아가 성공한 self-review만 cross-episode memory에 저장하고, distillation 손실도 성공한 2차 시도에만 마스킹해 검증된 수정 규칙만 누적되게 한다.

- **Empirical Impact**: GSM8K에서 Qwen 3-4B와 OLMo-3-7B 두 모델 모두 SRRL이 RLVR 대비 최종 보상과 학습 효율에서 일관되게 우수했다. 특히 OLMo-3-7B처럼 사전 수학 능력이 상대적으로 약한 모델에서 self-review의 이득 폭이 더 크게 나타났는데, 이는 모델이 성능 포화 전 영역일수록 체계적 오류 탐지-수정이 더 잘 먹힌다는 신호로 해석된다. 또한 SRRL은 추론 시 추가 self-review나 메모리 조회 없이도 학습된 개선을 정책 가중치에 인코딩해 “zero latency” 배포 제약을 해결한다.



### Rendering-Aware Bayesian 3D Gaussian Splatting with Native Uncertainty and Adaptive Complexity Contro (https://arxiv.org/abs/2607.05522)
Comments:
          26 pages, 4 figures, 24 tables including appendix. Preprint

- **Prior Approaches**: 3D Gaussian splatting(3DGS)은 실시간 novel-view synthesis에 강하지만, 학습이 점추정(point estimate) 중심이라 불확실성을 자연스럽게 제공하지 못한다. 또 Gaussian birth/death 같은 핵심 제어가 hand-tuned heuristics에 의존해, 희소 뷰(sparse views)나 고정 예산 환경에서 약하게 지지된 기하를 식별하거나 다음에 볼 카메라를 원리적으로 고르는 데 한계가 있었다.

- **Core Contribution**: 논문은 렌더러(알파 컴포지팅)에서 얻은 surrogate summary로 normal-inverse-Wishart(NIW) posterior를 각 Gaussian의 mean/ covariance에 연결하는 rendering-aware Bayesian 3DGS를 제안한다. 여기에 선택적으로 Dirichlet-process 확장을 더해 component usage에 대한 확률 신호로 복잡도 제어까지 함께 제공하며, 폐루프가 아닌 “렌더링 인지적” 베이지안 갱신 경계를 명시한다.

- **Technical Challenges**: 핵심 과제는 비선형 렌더링 전체를 정확한 end-to-end Bayesian 추론으로 돌리기 어렵다는 점인데, 논문은 renderer-derived surrogate statistics로 conjugate 업데이트는 closed-form으로 처리하고 나머지는 명시적으로 근사하는 학습 스케줄을 설계했다. 또한 posterior 샘플을 다시 렌더링해 픽셀 단위 predictive uncertainty와 interval calibration 신호를 만들고, 고정 예산 active view selection은 이 불확실성을 기반으로 점수화하는 방식으로 연결했다.

- **Empirical Impact**: 고정 예산 16-to-32 active-view에서 NIW native acquisition은 scoring-only standard ensemble 대비 PSNR +0.453 dB, LPIPS -0.0146의 개선을 보이며 29/39 씬 시드에서 승리했다. 95% 커버리지 오차는 shared proxy 대비 약 17배 감소(0.046 vs 0.796)하고, 3-member deep ensemble과 비교해서는 nominal coverage에 약 10배 더 가깝지만 학습비용은 약 1/3 수준이다; 추가로 호환성 확인에서도 matched 실행 39쌍에서 PSNR이 +0.030 dB 수준으로 유지되었다.



### aiAuthZ: Off-Host, Identity-Bound Authorization for AI Agents (https://arxiv.org/abs/2607.05518)
Comments:
          Technical Report

- **Prior Approaches**: 툴 호출을 할 수 있는 AI 에이전트에서는 ‘누가 요청했는가’와 ‘요청된 행동이 허용되는가’가 핵심인데, 기존 방어는 대부분 모델 내부의 확률적 정책에 의존해 취약하다. 간접 프롬프트 인젝션처럼 모델이 읽는 문맥을 공격자가 조작할 수 있으면, 실제 권한과 무관하게 도구 호출의 외관이 ‘승인된 것처럼’ 보일 수 있다. 또한 동일 프로세스(또는 동일 신뢰 영역)에서 권한 결정을 하면 런타임이 공격자 영향권에 들어가 방어가 무너진 사례가 보고돼 왔다.

- **Core Contribution**: 이 논문은 모델의 호스트 안에서 권한 결정을 하지 않고, 별도의 신뢰 도메인에 authorization gateway ‘aiAuthZ’를 둬서 도구 호출의 안전 결정을 분리한다. 각 도구 호출은 “가장 최근에 HMAC 검증된 사용자 메시지”에 묶인 활성 메시지 식별자를 기준으로 권한을 판단하며, 에이전트는 정책을 읽거나 수정할 수 없다. 요약하면 aiAuthZ는 모델이 속아 넘어가더라도 ‘검증된 사용자 권한 밖의 행동’을 모든 호출에서 원천 차단하는 데 초점을 둔다.

- **Technical Challenges**: 사용자 권한 스푸핑/세션 하이재킹/재전송 공격을 막기 위해, 메시지마다 HMAC-SHA256 서명과 단회 nonce, 타임스탬프 윈도우를 결합하고, nonce는 만료 포함 단일 사용으로 관리한다. 이어서 role-based 도구 allowlist에 더해 인자 수준 제약(경로·URL·수신자·쓰기 크기)과 per-tool rate limit을 정책 엔진에서 일괄 평가한다. 모든 결정은 SHA-256 해시 체인 감사 로그에 기록되고, 승인된 호출에는 재인코딩·스크린샷 처리에도 검증 가능한 HMAC 인증 QR receipt(평균 94% 검증 성능)가 발급된다.

- **Empirical Impact**: 15개 모델을 8개 공격 시나리오로 평가한 결과, 모델 거부율은 100%에서 38%까지 넓게 흔들렸고 가장 비싼 모델도 절반만 거부했다. 하지만 aiAuthZ를 통과한 뒤에는 모든 모델에서 잔여 공격 성공이 0%로 떨어졌으며, 의사결정 추가 지연은 최대 0.03ms 수준으로 측정됐다. AgentDojo 뱅킹 스위트와 Agents of Chaos 유사 사례에서도 aiAuthZ는 공격자 지시형 7개 도구 호출을 모두 막고(또는 9/9 케이스 차단), 정책 베이스라인 대비 유의미한 개선을 보였으며 관련 구현과 실험은 공개됐다.



### Statistical Adversaries: Natural Backdoor-like Features in Vision Datasets (https://arxiv.org/abs/2607.05516)
- **Prior Approaches**: 기존 연구는 주로 PGD 같은 최적화 기반 adversarial attack, 또는 victim/surrogate 모델을 통해 adversarial direction을 찾아내는 방식에 집중해 왔습니다. 또한 이미지에서 잡음·주파수 성분을 이용하거나 Fisher geometry 같은 정보기하를 통해 공격 방향을 모델 신호로 도출하는 연구가 많았습니다.
반면 ImageNet의 spurious feature(편향된 빈도 단서, 주파수 단서 등)는 “왜 실패하는가”를 설명하는 감사(audit) 관점에 머물렀고, 무해하게 보이는 자연 데이터 구조가 실제로 “공격 표면”이 될 수 있는지까지는 충분히 연결되지 않았습니다.

- **Core Contribution**: 이 논문은 악의적 poisoning 없이도 데이터의 클래스 조건부 통계로부터 backdoor-like 행동을 유발하는 ‘statistical adversaries’를 정의하고, 그 방향을 모델 접근 없이 구성하는 절차를 제시합니다. 즉, victim 모델 gradient나 쿼리, surrogate 최적화 없이도 목표 클래스에 특화된 오탐을 유도하는 단일 perturbation direction을 만든다는 점이 핵심입니다.
또한 이 취약성이 특정 아키텍처의 idiosyncrasy가 아니라, 데이터셋 구조·분포에 의해 공유될 수 있음을 실험적으로 뒷받침합니다.

- **Technical Challenges**: 가장 큰 난제는 모델을 보지 않고(gradient/쿼리/최적화 없이) “어떤 통계가 어떤 방향을 만들며, 그 결과가 목표 클래스 특이적으로 나타나는가”를 재현 가능하게 구성하는 것입니다. 연구진은 타깃 클래스 평균 대비(1·2차 모멘트)를 기반으로 방향을 만들되, diagonally whitened band-pass 및 Hellinger-motivated 주파수 통계 같은 reshaping/제어 연산을 넣어 무작위 상관을 줄이는 방식으로 해결합니다.
마지막으로 ℓ∞ 예산 하에서 입력 공간에 동일한 perturbation을 재사용하고, 임계값을 고정한 one-vs-rest FPR 지표로 목표 특이적 false positive를 정량화해 통계적 효과를 검증합니다.

- **Empirical Impact**: ImageNet-1K 학습 통계만으로 만든 perturbation은 four model(ResNet-50, ConvNeXt-Tiny, ViT-B/16, Swin-T)에서 목표 클래스별 FPR을 기준선 5.005%에서 9.689%로 끌어올렸고, 44개 조건 중 43개에서 효과가 양(+)의 방향으로 나타났습니다. 또한 임계값 통과 FPR, 타깃 로짓 이동, 타깃 랭크 변화가 관찰됐으며 단순 top-1 takeover보다는 “보정된 오탐 팽창” 형태로 나타나는 점이 특징입니다.
대조군 결과로 Gaussian random·global mean은 거의 기준선 근처에 머물렀고, lowpass/spectrum 매칭은 일부를 설명하지만 제안 방향이 대부분 조건에서 더 크게 나타났습니다. 결론적으로, poisoning이 없어도 데이터의 spurious structure가 비가역적 공격 표면을 형성할 수 있어 dataset audit에서 ‘편향/해석 실패’뿐 아니라 ‘잠재 공격 표면’으로 취급해야 한다는 메시지를 강화합니다.



### Lean-Quantum: Toward AI-Assisted Formalization of Quantum Information (https://arxiv.org/abs/2607.05492)
Comments:
          34 pages, the Lean library is available at this https URL

- **Prior Approaches**: 양자 정보 이론의 핵심 난제 중 하나는 sandwiched Rényi relative entropy의 data processing inequality(DPI)를 채널 아래에서 엄밀히 보이는 것이다. 기존 증명들은 대부분 Lieb–Ando 계열의 trace 부등식, Stinespring dilation, Haar averaging, Jensen 불등식 등 복잡한 연산자 기법을 조합하지만, 이를 Lean 같은 정리증명 도구에서 기계검증 가능한 형태로 정리하는 인프라는 부족했다. 또한 Lean 선행 작업들은 generalized quantum Stein's lemma 쪽에 필요한 DPI를 ‘sorry’ 등 미검증 요소로 남겨 두는 경우가 있었다.

- **Core Contribution**: 이 논문은 Lean 4와 Mathlib에 호환되는 ‘양자 정보 공식화 라이브러리’를 제시하고, 그 대표 데모로 sandwiched Rényi relative entropy의 DPI를 유한차원에서 양의 준정부소(positive semidefinite) 연산자까지 완전하게 formalize한다. 특히 좌표-비의존(basis-independent) 방식으로 유한차원 양자계/상태/채널/텐서곱/부분추적/Choi·Kraus·Stinespring 표상을 인터페이스화해, 기존 Mathlib의 연산자 이론과 자연스럽게 결합되도록 했다. 이 기반 위에 비가환 trace 부등식과 엔트로피 특화 성분을 쌓아 DPI, 그 결과로서 strong subadditivity(특수 코롤러), 나아가 generalized quantum Stein's lemma의 핵심 누락 부분까지 채운다.

- **Technical Challenges**: 주요 기술적 난관은 (1) DPI의 증명 경로가 inverse power/역제곱근/양의 멱함수처럼 ‘가역성(양의 정부소)’ 가정에 강하게 의존한다는 점, (2) 준정부소로 확장할 때 support 조건과 extended-real 관례를 Lean에서 깔끔히 분리·정식화해야 한다는 점이다. 이 문제를 위해 먼저 positive definite 원추에서 variational formula(Young·reverse-Young을 통한 sandwiched quasi-entropy 관련 식)와 Lieb–Ando trace inequality, 그리고 Stinespring–Haar–Jensen–로그 단조성 등의 모듈들을 조합해 DPI를 증명하고, 이어 extended-real-valued non-negative divergence로 준정부소 케이스의 monotonicity를 따로 구성한다. 또한 연산자 단조/볼록성, Jensen’s operator inequality, generalized perspectives, operator power means, Hilbert–Schmidt 연산자 공간 변환 등 반복 사용 가능한 ‘비가환 trace 부등식 계층’을 Mathlib 친화적으로 구축했다.

- **Empirical Impact**: 결과적으로 논문은 알려진 수학적 정리를 “끝까지” 기계검증 가능한 형태로 제공해, 향후 AI-assisted 정리 탐색/자동화가 생성하는 추정이 진짜 증명으로 이어지는지 검증 가능한 기반을 만든다. DPI formalization은 특히 α→1 극한에서 Umegaki relative entropy로 이어지며, partial trace를 채널로 택할 때 strong subadditivity를 함께 얻는 구조를 Lean 수준에서 확정한다. 또한 generalized quantum Stein's lemma formalization에서 DPI 관련 ‘마지막 누락 성분’을 채울 수 있게 되어, 양자 가설검정 분야의 더 큰 정리 체계까지 신뢰성 있게 확장될 발판을 제공한다.



### PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates (https://arxiv.org/abs/2607.05483)
Comments:
          24 pages, 13 figures, including appendix

- **Prior Approaches**: 에이전틱 워크플로는 한정된 LLM 컨텍스트 때문에 progressive disclosure(점진적 공개) 방식으로 매 단계에 필요한 상태 조각만 모델에 보여주는 흐름이 일반적이다. 이를 위해 grep-like 검색, RAG, AST 쿼리, task-specific agent skills 같은 읽기 최적화가 많이 쓰이지만, 로컬에서 제안한 “수정”이 전체 상태에 대해 언제 유효한지는 계약이 부족하다. 또한 스키마 검증·constrained decoding·입증되지 않은 근거를 통한 값 생성은 각각 국소적 실패만 줄일 뿐, provenance/authorization 관점의 글로벌 유효성 판단과 다단계 확장에 취약하다.

- **Core Contribution**: 논문은 PatchOptic이라는 “공유 상태용 step 인터페이스”를 제안한다. 각 단계는 projected read view(투영된 읽기)·authorized write region(허용 쓰기 범위)·patch-source region(패치가 참조할 소스 경로)을 하나의 계약으로 선언하고, 런타임에서는 모델에 투영 뷰만 제공한 뒤 전체 상태 기준으로 patch를 검증(commit 전)한다. 더 나아가 동일 선언이 delegation/서브워크플로 조합/same-phase 재정렬 같은 정적 분석도 가능하게 해 워크플로 재구성이 안전해진다.

- **Technical Challenges**: 핵심 기술 난제는 “로컬로 보인 뷰에서 나온 JSON Patch가 전체 상태에서 의미적으로도(및 권한적으로도) 성립하는지”를 LLM 실행 없이도 스케일 있게 판정하는 것이다. 이를 위해 PatchOptic은 optics에서 영감을 받은 authority triple을 기반으로 projected read와 검증 가능한 structured patches를 결합하고, 스키마·phase 제약·적용가능성·불변조건·patch 연산의 참조 소스까지 verifier가 승인하기 전에 차단한다. 또한 hidden sources(숨은 소스)처럼 선언되지 않은 경로를 이용하는 패치 아티팩트는 no-model 테스트의 containment 시나리오에서 모두 거부하도록 설계했다.

- **Empirical Impact**: PatchOptic의 실험은 6개 도메인, 총 46개 케이스로 구성된 PatchBench를 통해 수행됐다. projected read 설정은 강한 actor 조건에서 semantic pass(허용 출력의 의미 적합)를 약 0.61에서 0.78–0.80(GPT-5-mini)으로 끌어올리면서도 토큰 비용과 누출(leakage) 보고를 줄였다. 런타임 검증을 결합하면 leak runs가 라운드당 0.1 수준으로 감소하고, hidden-source patch artifact는 containment 테스트에서 전부 거부되어 “읽기 최소화 + 커밋 전 검증”의 실효성이 입증됐다.



### Full-range Binary Classifier Calibration for Stable Model Updates in Production (https://arxiv.org/abs/2607.05481)
- **Prior Approaches**: 기존 확률 보정(Platt scaling, isotonic regression)은 class probability를 맞추는 데 초점이 있어, 배포 후 운영자가 기대하는 FPR(오탐률) 곡선 의미가 릴리스마다 달라질 수 있습니다. 또 adversarial 환경에서는 재학습이 잦아 모델 출력 점수 스케일이 바뀌며, 그 결과 downstream 임계값과 오탐 정책이 깨지는 문제가 커집니다. 일부 연구는 risk control이나 conformal 같은 보증/테스트를 제공하지만, 전체 FPR 곡선을 고정된 계약(anchors)으로 “점수 변환기” 형태로 재현 가능하게 배포하진 못했습니다.

- **Core Contribution**: 이 논문은 악성 분포는 불특정 unknown unknown이지만 benign 트래픽은 비교적 안정적이라는 전제에서, benign 데이터만으로 raw score를 FPR 계약에 맞게 보정하는 방법을 제안합니다. 보정 결과는 점수가 곧 FPR을 뜻하는 “whole-curve FPR mapping”으로, 릴리스가 달라져도 동일한 calibrated threshold가 같은(고정된) benign FPR을 유지하도록 설계했습니다. 또한 sklearn의 MinMaxScaler와 IsotonicRegression을 조합해 추론 코드 커스텀 없이 Pipeline으로 제공됩니다.

- **Technical Challenges**: 핵심 난제는 (1) benign 표본 수가 제한된 저FPR 꼬리에서의 유한표본 편향/분산, (2) rank 기반 스플라인의 knot 라벨링 편향(naive k/n 대신 보정된 plotting position)입니다. 저FPR에서 FPR 라벨이 과대 추정되는 문제를 Filliben의 median plotting position으로 완화하고, log10(FPR) 공간에 고정 anchor(예: 0.1%, 0.01% 등)을 핀으로 두어 운영자가 오더오브매그니튜드 단위로 정책을 읽을 수 있게 했습니다. 마지막으로 샘플 floor 아래는 안전한 clipping+선형 extrapolation으로 단조성과 하한을 유지하며, 배포 크기는 knot-subsampling으로 200KB 미만을 목표로 고정했습니다.

- **Empirical Impact**: 신용카드 사기 탐지(Credit Card Fraud Detection) 실험에서 held-out 분할 기준, 10%~0.1% FPR 구간의 상대 FPR 오차는 최대 2.3%였고 0.01% FPR 지점에서는 7.2%였습니다. 특히 calibrator로 생성된 점수 임계값이 calibrated threshold별 FPR을 맞추도록 일반화 성능을 separate fit/held-out에서 확인했으며, TPR 붕괴 여부와는 독립적으로 FPR 계약의 일관성을 유지하도록 설계 의도를 분리해 설명합니다. 아울러 보정 아티팩트는 benign 샘플 1K~10M 범위에서도 직렬화 크기가 54~161KB로 낮게 유지되어, 운영 배포 비용 측면에서도 실용성을 보여줍니다.



### Privilege and confidentiality in generative AI workflows (https://arxiv.org/abs/2607.05479)
- **Prior Approaches**: 기존 논의는 생성형 AI가 데이터를 어디에 저장·처리하는지(모델 파라미터, 컨텍스트 윈도우, RAG용 지식DB)로 나눠 위험을 설명하기보다는, 주로 일반적 보안·가용성 관점에서 접근해 왔다. 그 결과 기밀·법률전문직 비밀(privilege) 같은 법적 보호가 “어떤 데이터 흐름에서 깨지는지”를 직관적으로 연결하기 어려웠다. 또한 영국·미국의 핵심 판례를 현업 관점의 기술 모드로 번역해 거버넌스 기준을 제시한 연구가 부족했다.

- **Core Contribution**: 이 논문은 GenAI 데이터 저장·처리를 3가지 모드로 구조화해, 각 모드가 비밀유지(legal professional privilege 등)에 미치는 법적 결과를 방식별로 분석한다. 특히 영국 사건(UK 및 Munir v Secretary of State for the Home Department)과 미국 사건(United States v Heppner)을 기초로, 전통적 privilege 법리를 데이터 처리 모드별로 재해석한다. 이를 통해 SRA 규율이 적용되는 변호사·로펌 실무자에게도 바로 적용 가능한 “효과적 정보 거버넌스”의 기준이 어떻게 변하는지 제시한다.

- **Technical Challenges**: 기술적으로는 파라미터 학습·기억(훈련 중 내재화), 실시간 컨텍스트 윈도우, 그리고 RAG의 검색·결합 과정이 서로 다른 “기밀 노출 경로”를 만든다는 점을 법적 기준(비밀 보호 요건, 입증 가능성)과 연결해야 한다. 논문은 이를 practitioner가 이해할 수 있는 수준으로 3모드 저장·처리 개념을 정리하고, 각 모드별로 필요한 통제·거버넌스 대응을 법적 함의와 함께 도출한다. 또한 다른 관할에서도 privilege 또는 전문 비밀 보호가 ‘입증 가능한 기밀성’에 의존한다면 동일한 논리 틀을 확장할 수 있다고 설계한다.

- **Empirical Impact**: 실증 실험보다는 법·컴퓨터과학의 최근 연구를 결합한 분석 중심으로, 데이터 유출 위험을 “배치 가능한 관리 기준(효과적 정보 거버넌스)”으로 전환하는 데 의미가 있다. 결론은 결과적으로 negligence(전문직 과실)·위법/부정행위 판단에서 참조될 벤치마크가 상승할 수 있음을 시사한다. 법률 서비스 업계가 GenAI를 의뢰인 데이터 및 민감 자료에 적용할 때 어떤 위험을 우선 점검해야 하는지 실무 체크 관점을 제공한다.



### Decision Protocols in Multi-Agent Large Language Model Conversations (https://arxiv.org/abs/2607.05477)
Comments:
          Master's thesis, University of Göttingen

- **Prior Approaches**: 기존 연구는 LLM의 멀티에이전트 활용 시 여러 에이전트를 두더라도 단일 의사결정 프로토콜을 쓰거나, 적용 범위가 좁은 데이터셋에서만 성능을 확인하는 경우가 많았다. 또한 멀티에이전트는 학습 비용을 줄일 수 있지만, 에이전트 간 토론·결정 과정으로 테스트 시간이 늘어날 수 있어 프로토콜 설계가 핵심이라는 점이 반복적으로 지적돼 왔다.

- **Core Contribution**: 이 논문은 Multi-Agent LLM(MALLM) 프레임워크를 제안해 투표(voting), 합의(consensus), judge 기반 결정(judge decision) 등 다양한 의사결정 프로토콜을 구현하고 체계적으로 평가한다. 대화형 과업 해결을 목표로, 여러 에이전트가 논의하며 최종 해답에 도달하는 과정을 프로토콜 단위로 시뮬레이션한다는 점이 차별점이다.

- **Technical Challenges**: 핵심 기술 난제는 에이전트들이 만들어낸 답변을 어떤 규칙으로 통합해 ‘정답으로 수렴’시키는지에 대한 설계였다. 논문은 각 프로토콜별 협업 방식을 비교하고, 특히 독립적인 솔루션 생성을 통해 response diversity를 늘리면 의사결정 품질이 향상됨을 확인했으며, 의사결정 과정에서의 정보 접근 방식 변화는 성능에 큰 영향을 주지 않는 경향을 관찰한다.

- **Empirical Impact**: 실험은 지식 기반 벤치마크(MMLU, MMLU-Pro, GPQA)와 논리 기반 벤치마크(StrategyQA, MuSR, Math-lvl-5, SQuAD 2.0)를 폭넓게 포함해 프로토콜-과업 유형 간의 상호작용을 보여준다. 결론적으로 consensus는 지식 집약 과업에서, voting 및 judge 프로토콜은 논리 과업에서 더 유리했으며, 에이전트 간 다양성 확보가 전반적 성능 향상에 기여한다는 실증적 근거를 제공한다.



### Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inferenc (https://arxiv.org/abs/2607.05475)
- **Prior Approaches**: 기존 연구는 모바일 LLM의 성능·에너지 측정을 하더라도 주로 장치 단위(whole-device) 관점에 머물러 백엔드별 병목을 분리하기 어려웠습니다. 또한 framework(추론 프레임워크)와 execution backend(CPU/GPU/NPU), 자원 스케줄링(DVFS 등)을 한꺼번에 교차 분석한 사례가 드물어 최적화 방향이 조각나기 쉬웠습니다. 그 결과 NPUs가 실제로 프레임워크마다 어떤 차이를 내는지, 그리고 이 차이가 스케줄링과 어떻게 결합해 에너지로 이어지는지에 대한 ‘블라인드 스팟’이 남아 있었습니다.

- **Core Contribution**: 이 논문은 모바일 LLM 추론을 cross-layer로 측정하는 최초의 포괄적 연구로, 5개 mainstream framework(예: llama.cpp, GENIE)와 3개 백엔드(CPU/GPU/NPU)를 동시에 아우릅니다. 이를 위해 PowerBench라는 프레임워크 비종속형 미세 프로파일링 도구를 제안하며, 백엔드별 에너지 할당(backend-specific energy attribution)을 통해 장치 단위 측정에서 보이지 않던 비용을 드러냅니다. 또한 prefill·decode 단계와 스케줄링 정책까지 통제해, “어떤 조합이 왜 효율적인가”를 실험적으로 연결합니다.

- **Technical Challenges**: 핵심 과제는 (1) 프레임워크마다 워크로드/계측 방식이 달라 공정한 비교가 어렵고, (2) 에너지를 백엔드 단위로 분해하려면 신뢰도 높은 전력 지표 매핑이 필요하다는 점이었습니다. 저자들은 토큰 주입과 EOS 토큰 처리까지 포함해 prefill·decode 처리량을 표준화하고, PowerBench로 SoC 전체 에너지와 CPU/GPU/NPU 전력 영역을 델타 기반으로 수집해 백엔드 단위로 귀속시켰습니다. 그 결과 framework 오프로딩 전략, 연산자 스케일링, activation quantization, 그리고 스케줄링의 세부 파라미터(스레드·polling·NPU sleep)까지 병목을 원인 단위로 추적할 수 있게 했습니다.

- **Empirical Impact**: 실험은 NPUs에서 프레임워크 간 성능 격차가 최대 10배까지 증폭되며(커스텀 operator/오프로딩·quantization 차이), prefill에서는 NPU가 강한 반면 decode에서는 CPU가 우세해지는 ‘phase split’을 보여줍니다. 더 나아가 백엔드별 프로파일링을 통해 스케줄링이 남긴 에너지 낭비를 계량화했는데, CPU polling과 동기화, NPU sleep 지연 등으로 토큰당 에너지가 최대 40%까지 낭비될 수 있음을 확인했습니다. 이러한 발견을 바탕으로 에너지 지향 best-practice 구성을 제시하며, NPU 백엔드에서 최대 54.8% 에너지 절감을 세 데이터셋에 걸쳐 추정/입증합니다.



### Binocular Gaze Estimation with Single Camera and Single Light Sourc (https://arxiv.org/abs/2607.05473)
Comments:
          Accepted for presentation at the 2019 International Conference on Video, Signal and Image Processing (VSIP 2019), Wuhan, China, October 29-31, 2019; published in VSIP '19: Proceedings of the 2019 International Conference on Video, Signal and Image Processing, pp. 10-14, ACM, 2020; 4 figures, 1 table; ACM Proceedings ISBN: 978-1-4503-7148-3

- **Prior Approaches**: 기존에는 자유로운 머리 움직임을 전제로, 시선(gaze) 추정을 위해 최소 1대 카메라와 2개의 광원(light source)이 필요하다고 널리 받아들여져 왔다. 특히 glint(각막 반사점) 기반 방식에서는 두 광원으로부터 생성되는 glint 정보를 이용해 시선을 더 안정적으로 추정한다. 하지만 모바일 기기처럼 부품을 줄여야 하는 상황에서는 광원 수를 줄이려는 요구가 커져 한계가 드러난다.

- **Core Contribution**: 이 논문은 1대 카메라와 1개 광원만으로도 시선을 추정하는 방법을 제안한다. 핵심은 카메라를 기준으로 실제 광원과 대칭 위치에 ‘virtual light source(가상 광원)’를 기하학적으로 배치해, 영상에서 ‘virtual glint(가상 glint)’를 생성하는 개념이다. 이후 두 동공(pupil) 사이 거리와 실제/가상 glint 사이의 관계를 활용해 시선을 추정하고, 2개 광원이 있는 것으로 가정한 polynomial regression을 적용한다.

- **Technical Challenges**: 기술적 난점은 실제로는 광원이 1개뿐인데도 2개 광원 시스템과 유사한 glint 신호 구조를 안정적으로 재구성해야 한다는 점이다. 논문은 virtual glint를 기하학적 대칭으로 정의하고, 영상에서 관측되는 두 pupil과 두 glint(실제 glint+virtual glint) 간의 거리 관계를 회귀 모델에 반영해 추정 정확도를 확보한다. 또한 regression용 새로운 normalization factor를 검증해, one-glint 시스템에서도 적용 가능함을 보인다.

- **Empirical Impact**: 실험 결과, one-glint 구성에서도 성능이 ‘수용 가능한 수준’으로 유지되지만 2개의 실제 광원을 쓰는 시스템 대비 성능 저하는 관찰된다. 즉 부품을 줄이는 실용성은 확보하되, 정확도 측면에서 트레이드오프가 존재한다는 메시지를 제공한다. 모바일 등 경량 환경에서 gaze tracker 설계의 제약을 낮출 수 있다는 점에서 현장 적용 가능성에 의미가 있다.



### KAT-Coder-V2.5 Technical Repor (https://arxiv.org/abs/2607.05471)
Comments:
          24 pages, 5 figures

- **Prior Approaches**: 코딩 LLM은 코드 완성에서 에이전트로 확장됐지만, 실제 저장소에서 실행·수정·검증하는 long-horizon 능력을 만들기엔 훈련 인프라가 병목이 된다고 본다. 기존 접근은 재현 가능한 executable 환경을 대규모로 만들거나, 보상(검증 신호)을 확실히 산출하거나, 품질 높은 trajectory를 확보·일관되게 학습 신호로 전환하는 데 한계가 있었다.

- **Core Contribution**: KAT-Coder-V2.5는 단일 턴 생성기가 아니라 “실행 가능한 저장소” 안에서 자율적으로 동작하는 coding-focused agentic model로, end-to-end agentic post-training을 제안한다. AutoBuilder는 저장소를 샌드박스 환경으로 재구성하고 fail-to-pass/pass-to-pass 검증을 통해 task specification을 재생성·near-miss를 복구·process-aware로 필터링하며, KwaiClawEnv는 실행 가능한 서비스/실제 시드 기반 tool-use 궤적을 대규모 합성한다.

- **Technical Challenges**: 핵심 과제는 (1) 깨지지 않고 진짜로 테스트를 돌리는 scalable executable environments, (2) 최종 성공률만으로는 걸러지지 않는 trajectory 품질(지름길·test tampering vs 유의미한 탐색/국소화), (3) sparse reward와 불안정성 때문에 생기는 long-horizon RL 학습 안정성이다. 논문은 fail-to-pass/pass-to-pass 기반 자동 검증, 힌트 누출 없이 near-miss를 복구하는 hint-free trajectory 재생성, harness randomization, reliability-hardened sandbox, asymmetric actor–critic PPO에 harness-oriented reward를 결합하고, Multi-Teacher On-Policy Distillation로 여러 expert를 온폴리시 방식으로 통합한다.

- **Empirical Impact**: 6개 소프트웨어공학 및 agentic 벤치마크에서 KAT-Coder-V2.5는 PinchBench의 best agentic tool-use 결과를 달성했고, SWE-Bench Pro와 KAT Code Bench에서는 frontier 모델(Opus 4.8) 다음으로 2위를 기록했다. 성능 향상은 단순 스케일이 아니라 “검증 가능한 환경·품질 높은 궤적·안정적인 long-horizon 학습 신호”를 시스템 수준에서 다뤘다는 점에서, SWE 에이전트 학습 프레임워크에 의미 있는 방향을 제시한다.



### Breaking Structural Isolation: Scalable Graph Clustering via Community-Aware Sampling and Structural Entropy (https://arxiv.org/abs/2607.05469)
Comments:
          Accepted to the Proceedings of the VLDB Endowment (VLDB 2026). 18 pages, 15 figures, 15 tables

- **Prior Approaches**: 기존 비지도 graph clustering은 GNN과 Graph Contrastive Learning(GCL)을 결합해 임베딩의 판별력을 높이는 흐름이 강했다. 하지만 mini-batch 학습에서 ‘structural isolation’ 문제가 발생해 전역 토폴로지에 기반한 커뮤니티 응집성을 충분히 학습하기 어렵다.

- **Core Contribution**: 이 논문은 SCISE를 제안하며, 커뮤니티 응집성은 유지하면서도 대규모 확장성을 확보하는 데 초점을 둔다. 핵심은 community-aware sampling과 constrained Structural Entropy를 함께 써서 커뮤니티 분절을 줄이고, 배치 단절로 인한 정보 손실을 보정하는 구조다.

- **Technical Challenges**: 문제는 (1) structural entropy 최소화 과정이 과분할과 로컬 최적화에 취약하고, (2) mini-batch 샘플링이 커뮤니티 내부의 전역 문맥을 끊어먹는다는 점이다. SCISE는 SECC(Structural Entropy Community Constraint)로 커뮤니티 개수 제약 하에 분할을 안정화하고, CSampE(Community-Aware Sampling Expansion)로 배치가 커뮤니티 문맥을 보게 확장하며, StructCL(Structural Contrastive Learning)로 배치 내 구조적 유사도 기반 edge weight를 재가중해 고차 구조 공간에서 표현을 정련한다.

- **Empirical Impact**: 6개 벤치마크(중형 3종, 대형 3종)에서 SCISE가 10개 수준의 SOTA 대비 일관되게 성능이 높았고, SECC·CSampE·StructCL의 조합 효과와 상호 보완성도 ablation으로 확인됐다. 또한 하이퍼파라미터 변화에 대한 안정성 분석과 robustness 분석을 통해 대규모 그래프 환경에서도 신뢰성 있게 동작함을 보여주었다.



### Learning 4D Geometric Priors for Inference-Efficient World Action Models (https://arxiv.org/abs/2607.05468)
Comments:
          9 pages, 6 figures

- **Prior Approaches**: 기존 World Action Model(WAM)은 비디오 미래 예측과 액션 시퀀스를 함께 학습하지만, 많은 방법이 외형 중심(appearance-oriented) 비디오 잠재표현 최적화에 치우쳐 조작에 필요한 시공간 기하 관계를 충분히 담지 못한다. 3D/4D 구조를 넣더라도 배치 시점에 기하를 출력하거나(추가 디코더/출력), 일반적인 기하 감독을 써서 현재 액션과 인과적으로 연결된 관계를 잘 구분하지 못하는 한계가 있었다. 또한 기하 분기가 액션 생성 경로로 무제한 정보를 흘려 넣으면 비인과적 shortcut으로 학습이 편향될 위험도 지적된다.

- **Core Contribution**: MECo-WAM은 추론(inference) 그래프는 그대로 두고, 학습(training) 중에만 4D 기하 priors를 주입해 액션에 필요한 ‘시간 변화하는 기하’를 비디오-액션 표현에 이식하는 Multi-Expert Co-Training WAM을 제안한다. 학습 시에는 영상/액션 expert 외에 training-only 4D expert를 추가해 frozen VGGT 인코더의 관계형(relational) 타깃으로 시간적 기하를 감독한다. 핵심은 deploy 단계에서는 보조 4D 구성요소를 완전히 제거해 추가 비용 없이 조작 성능을 끌어올리는 것이다.

- **Technical Challenges**: 가장 큰 과제는 4D 기하 정보를 학습에만 쓰되, 액션 생성 경로로의 정보 누수를 막아 비인과적 shortcut을 방지하면서도 시간적으로 진화하는 관계를 제대로 전이하는 것이다. 이를 위해 MECo-WAM은 decayed 4D read-mask attention으로 현재 프레임 기하 토큰은 초기 학습에서만 제한적으로 읽히게 하고, 최종적으로는 해당 접근을 단계적으로 제거해 배치 시 의존성을 끊는다. 더불어 action-aware temporal geometric distillation을 통해 프레임 내 관계뿐 아니라 키프레임 간 관계 변화까지, 로봇 액션과 연관된 시각 토큰에 가중치를 둬 정렬하도록 설계했다.

- **Empirical Impact**: 실험에서 MECo-WAM은 LIBERO 98.2%, RoboTwin 2.0 92.6%, 그리고 ARX-R5 기반 실세계 태스크에서도 일관된 조작 성능 향상을 보이며 추론 비용(동일 lightweight video-action 그래프) 증가는 없었다. 특히 geometry-sensitive 계열에서 개선 폭이 커, ‘도달/정렬/접촉-전이’처럼 기하에 민감한 추론이 강화됐음을 시사한다. 실세계에서는 보정 횟수 감소와 완료 시간 단축까지 함께 나타나, 학습 시점 4D 기하 전이가 시뮬레이션을 넘어 로봇 그라운딩에 효과적임을 보여준다.



### CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration (https://arxiv.org/abs/2607.05465)
Comments:
          18pages, 5 figures

- **Prior Approaches**: 기존 멀티모달 도구 사용 연구는 Perception, search, 일반 추론에 치중하는 경우가 많고, 복잡한 이미지 생성·편집처럼 여러 도구 호출이 중간 시각 결과에 의존하는 ‘긴 실행 궤적’을 충분히 다루지 못했습니다. 또 이미지 편집/포토 리터칭 쪽도 단일 모델 호출 중심이거나 특정 환경에 한정되어, 이기종 도구를 조합하고 다수 중간 자산을 상태적으로 관리하는 대규모 실행 궤적 데이터가 부족했습니다.

- **Core Contribution**: 이 논문은 복잡한 이미지 생성·편집을 위한 대규모 멀티모달 도구 사용 데이터셋 CanvasCraft와, 이를 학습해 멀티턴 상호작용으로 이기종 visual tools를 오케스트레이션하는 CanvasAgent를 제안합니다. CanvasCraft는 140K개의 실행 가능한 주석 궤적과 10K개의 RL용 작업 스펙으로 구성되며, CanvasAgent는 SFT로 실행 가능한 궤적을 먼저 학습한 뒤 GRPO로 정책을 최적화합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 긴-horizon에서 중간 결과를 기반으로 도구 순서·파라미터·중단 시점을 스스로 결정해야 하고, (2) 생성 결과 품질뿐 아니라 궤적이 문법적으로/절차적으로 실행 가능해야 한다는 점입니다. 저자들은 비전 상태를 보며 중간 산출물을 검사하고 이미지 자산을 명시적으로 추적하는 실행 프로토콜을 쓰는 한편, outcome(정렬·미감)와 process(추론 타당성·규칙 준수·효율)를 함께 보는 하이브리드 리워드를 설계해 reward hacking을 줄이며 GRPO 학습을 안정화했습니다.

- **Empirical Impact**: 실험에서 CanvasCraft-SFT만 쓴 CanvasAgent는 전반 보상과 궤적 품질이 크게 개선되지만 도구 호출을 기대보다 적게 수행했습니다. SFT+RL로 확장하면 전반 보상(0.557→0.821), instruction alignment(0.613→0.869), 궤적 품질과 룰 기반 점수(0.576/0.467→0.849/0.785)가 동시 상승하며 평균 도구 호출 수도 기대에 가까워져(약 1.32→5.44) 복잡한 멀티툴 워크플로 학습 효과가 확인됩니다.



### Learnable Weighting of Intra-Attribute Distances for Categorical Data Clustering with Nominal and Ordinal Attributes (https://arxiv.org/abs/2607.05464)
Comments:
          16 pages, 11 figures

- **Prior Approaches**: 범주형 데이터 클러스터링에서 거리 척도는 핵심이지만, 기존 방법은 명목형(nominal)과 순서형(ordinal) 속성을 같은 방식으로 불일치도를 계산해 순서 정보(ordinal의 상대적 대소)를 충분히 반영하지 못했다. 또한 명목/순서 속성 간 상호의존성은 거리 측정에 부분적으로만 반영되거나, 명목형에 강한 표현/거리들은 순서형에 그대로 적용하기 어렵다는 한계가 있었다. 속성 가중치(weighting) 접근 역시 intra-attribute distance를 속성 단위로 균일 가중해, 어떤 “값-값” 거리들이 실제로 유용한지 세밀하게 학습하지 못한다는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 명목형과 순서형 속성 값의 “차이와 연결”을 그래프 관점에서 정리하고, 두 타입 모두에 대해 intra-attribute distance를 통합된 형태로 정의하는 새로운 거리 척도를 제안한다. 특히 명목형 속성의 각 값은 boolean 속성(“해당 값” vs “아닌 값”)으로 변환해 순서형의 특수 케이스로 만들고, 그 위에서 명목/순서 쌍 모두에 동일한 방식으로 거리(상대 정보 포함)를 계산한다. 더 나아가 intra-attribute distance의 값-값 가중치와 클러스터 분할을 한 학습 프레임에서 함께 최적화하도록 설계해, 두 단계로 나눠 생기는 suboptimal solution을 피한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 명목형은 순서가 없고 순서형은 값 간 인접성과 순서 구조가 있는데, 이를 하나의 동질(homogeneous)한 거리 정의로 묶는 것이다. 논문은 명목형 값을 boolean으로 재구성한 뒤, 다른 속성에 대한 조건부 분포 기반(그래프/정보 관점의 context 반영)으로 값-값 거리를 산출해 명목/순서가 동일한 “종류”의 거리로 비교되게 만든다. 두 번째 난제는 distance 가중치를 자동으로 학습하되, 기존의 순서형 의존적 방식처럼 ordinal의 순서만 활용해서는 안 된다는 점이며, 현재 분할 결과에 따라 intra-attribute distance weight를 반복적으로 갱신하는 메커니즘으로 해결한다.

- **Empirical Impact**: 실험에서는 제안된 거리 정의와 distance weighting을 사용하는 클러스터링 알고리즘이 명목/순서가 혼재된 범주형 데이터 전반에서 기존 비교군보다 일관되게 우수한 성능을 보였다. 특히 순서 정보가 있는 ordinal 값들을 명목처럼 취급할 때 발생하던 정보 손실을 줄이고, 명목/순서 간 상호의존성을 거리 측정에 더 잘 반영했다. 결과적으로 parameter free 성격의 새로운 categorical data clustering 틀을 제공하며, mixed 데이터셋에서 실용적인 개선 가능성을 보여준다.



### Evaluating calibrated refusal and safe usefulness in dual-use biology settings (https://arxiv.org/abs/2607.05462)
- **Prior Approaches**: 생명과학 워크플로에 AI 에이전트가 도입되면서, 발견을 가속하는 역량이 오남용으로도 이어질 수 있다는 우려가 커졌다. 기존에는 대체로 생성 제한이나 단순 필터링 중심의 위험 대응이 많았지만, 실제 연구 맥락에서 얼마나 ‘진짜 위험’과 ‘정상 업무’를 구분하는지 정량화가 부족했다.

- **Core Contribution**: 논문은 생물 연구 작업에서 위험 식별과 거절(refusal) 행동을 평가하는 벤치마크 BioSecBench-Refusal을 제안한다. Routine 61개(문헌 기반 합법 분석)와 Red-Team 46개(실제와 유사하지만 숨겨진 생물안보 위험이 있는 허구 시나리오)를 짝지어, 모델의 거절이 실제 위험에 비례하는지와 정상 작업을 과도하게 거절하는지까지 함께 본다.

- **Technical Challenges**: 핵심 기술 과제는 에이전트가 추론하기 전/후 어느 지점에서 거절이 발생하는지, 그리고 API 필터 같은 사전 제약이 성능을 어떻게 왜곡하는지 분리해 관찰하는 것이다. 저자들은 16가지 model-harness 설정을 통해 거절률 분포를 측정하고, 대부분의 거절이 agentic reasoning 이전에 적용된 provider API filters에서 촉발됨을 분석했으며, 추론 여지를 늘리면 더 많은 실제 위협 식별 가능성이 있음을 보였다.

- **Empirical Impact**: 실험에서 Routine 거절률은 7%~74%, Red-Team 거절률은 1%~62%로 설정별 편차가 컸다. 특히 많은 설정에서 숨겨진 위험보다 정상 Routine 작업을 비슷하거나 더 높은 비율로 거절해 ‘과잉 거절’ 문제가 관찰됐고, 이는 개발자가 capability와 caution을 동시에 캘리브레이션(calibrate)하는 도구로서 벤치마크의 가치가 크다는 신호로 해석된다.



### AdaStop: Cost-Aware Early Stopping for DNN Test Selection (https://arxiv.org/abs/2607.05461)
- **Prior Approaches**: 기존 DNN 테스트는 주로 '어떤 입력을 먼저 라벨링할지'(test selection)에 초점을 두고, 종료 시점은 고정 예산(budget)으로 처리하는 경우가 많습니다. DeepGini, TestRank, ATS, DeepSample 같은 방법은 라벨링 예산을 사전에 정하지만, 그 예산이 언제부터 비효율이 되는지에 대한 원칙적 기준은 제시하지 못합니다. 또 비용(c)과 가치(v)의 cost–benefit trade-off를 직접 반영하지 않아 라벨링이 늘어도 새 결함(fault)이 줄어드는 구간에서 낭비가 생길 수 있습니다.

- **Core Contribution**: 이 논문은 DNN 테스트의 'stopping problem'을 비용-이득 의사결정으로 정식화하고, 라벨 1건의 비용 c와 결함 발견의 가치 v에 기반한 최적 종료 조건 τ=c/v를 도출합니다. AdaStop은 테스트 중 주변 구간에서의 marginal fault discovery rate를 추정하고, 그 값이 τ보다 낮아지는 순간 라벨링을 멈추는 프레임워크입니다. 선택 전략은 DeepGini 같은 불확실성 기반 순위를 쓰되, stopping 로직은 strategy-agnostic하게 적용되도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 '결함 발견률 p(t)'를 관측할 수 없다는 점과, 테스트가 진행될수록 결함 발견이 줄어드는 비정상성·잡음 문제입니다. AdaStop은 최근 history만 쓰는 sliding-window 추정으로 p(t)를 완화해 추정하며, 추정값이 τ를 하회하면 즉시 종료하는 룰을 구현합니다. 또한 경계에서의 변동을 줄이기 위해 patience, consecutive non-faults, confidence-based 등 대안 종료 기준도 비교·제시하며 실전 변동성 대응을 강화했습니다.

- **Empirical Impact**: CIFAR-10, SVHN, FashionMNIST와 ResNet-20/VGG-16/DenseNet-121/ShuffleNetV2, 3개 품질 수준에 대해 실험한 결과 AdaStop은 65–84%의 fault recall을 9–31% 라벨링 예산으로 달성하며 70–91% 예산 절감을 보였습니다. 특히 정확도를 고정했을 때 exhaustive testing이 오히려 net value를 낮출 수 있는데, AdaStop은 이런 낭비를 줄여 더 높은 net value를 얻는 패턴을 보였습니다. 종료 기준별로는 threshold가 효율 면에서 유리하고, consecutive-50이 net value에서 강점, confidence-90은 거의 완전한 recall에 적합하다는 결론이 나와 실무 선택 가이드를 제공합니다.



### Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning (https://arxiv.org/abs/2607.05458)
Comments:
          17 pages, 7 figures

- **Prior Approaches**: 기존 LLM 에이전트 성능 향상은 프롬프트, 모델, 손으로 짠 workflow를 바꾸는 방식이 주로 사용됐고, 에이전트 실행을 감싸는 harness(관찰·증거수집·툴호출·검증·수정·종료 결정)는 고정 인프라로 취급되는 경우가 많습니다. 그 결과 고정 규칙(예: 항상 check)이나 고정 실행 그래프는 쉬운 작업엔 예산을 낭비하고, 어려운 단계에선 검증 타이밍이 어긋나는 등 제어 시퀀스의 실패가 남는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 harness를 ‘학습 가능한 제어 레이어’로 재정의하고, frozen LLM executor 위에 얇은 controller가 다음 구조적 실행 동작을 고르는 Harness MDP로 형식화합니다. 또한 final task quality와 process quality를 분리해, 최종 정답이 아니라 harness가 신뢰 가능한 실행 패턴을 따르는지 HMS(Harness Maturity Score)로 따로 측정합니다. 학습은 LLM 파라미터를 업데이트하지 않고, 오프라인 rollouts의 terminal task-rubric 보상만으로 advantage-weighted regression(AW)을 사용해 제어 정책을 익힙니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 온라인 탐험 없이 오프라인 데이터만으로 제어 순서를 학습해야 하고, (2) ‘검증/수정 같은 과정’이 개선됐는지와 ‘최종 품질’이 개선됐는지를 혼동하지 않아야 한다는 점입니다. 논문은 오프라인 버퍼에서 높은 advantage의 state-action을 더 강하게 재가중하는 AW로 controller를 학습하고, HMS를 진단용으로만 두어 학습 목표를 terminal task-rubric에 고정함으로써 과정 보상 설계가 최종 최적 정책을 왜곡하지 않게 했습니다. 더 나아가 finite-buffer 관점에서 outcome 향상은 고수익 궤적의 버퍼 지지(support)가 있어야 하고, process 변화는 advantage-weighted 선택과 맞아떨어지면 버퍼 유무와 무관하게 부분적으로 일어날 수 있음을 이론적으로 분석합니다.

- **Empirical Impact**: 6개 controlled domain과 2개 공개 벤치마크 어댑터에서 AW controller는 전반적으로 제출 전 verification(예: CheckBeforeSubmit) 행동을 일관되게 개선했으며, 이는 최종 점수 변화가 작아도 관찰됐습니다. 최종 task quality는 선택적으로 개선되었는데, 특히 코딩(캘리브레이티드 structural verifier)과 tau-bench retail(어댑트), AgentBench DB-Bench(어댑트)에서 외부 효과가 가장 컸습니다. 또한 behavior cloning과 Forced CHECK 대비 AW의 이득은 단순 모방(imitation)이나 체크를 기계적으로 삽입하는 방식만으로는 설명되지 않으며, 상태조건(state dependence) 기반으로 ‘언제’ 검증·수정이 유용한지를 학습했음을 시사합니다.



### Empirical Minimal-Realisation Compression of Deep Neural Networks via Controllability-Observability Tests (https://arxiv.org/abs/2607.05457)
- **Prior Approaches**: 기존 DNN 압축은 pruning, quantisation, low-rank factorisation, knowledge distillation처럼 가중치·뉴런·채널·정밀도에 직접 개입하는 방식이 주류였다. 하지만 이런 방법들은 데이터가 실제로 어떤 숨은 상태 방향을 “활성화”하고, 그 방향이 출력에 “영향”을 주는지 같은 상태공간 관점을 명시적으로 다루지 못했다. 그 결과 넓은 은닉층의 중복이 어디에서 왜 발생하는지(실제로 줄여도 되는 상태 차원)는 진단하기 어려웠다.

- **Core Contribution**: 이 논문은 학습된 feedforward DNN을 depth-indexed nonlinear state-space system으로 보고, 숨은 상태의 useful hidden-state order를 controllability–observability 관점에서 추정하는 프레임워크를 제안한다. 각 층에 대해 reachability(데이터로부터 도달)와 observability(출력 로짓으로부터 민감) 그리고 둘을 결합한 C-balanced rank을 실증적으로 계산한다. 무엇보다 C-balanced rank을 진단 지표에 그치지 않고, 압축된 네트워크의 실제 층 너비(width)로 “실현(realised)”해 모델을 재구성한다.

- **Technical Challenges**: 핵심 기술 난제는 비선형·데이터의존적인 네트워크에서 controllability/observability에 해당하는 양을 어떻게 데이터로부터 추정하느냐였다. 논문은 은닉상태 스냅샷으로 reachability Gramian(공분산)을 만들고, 출력 로짓에 대한 hidden state Jacobian으로 observability Gramian을 구성해 A/B/C 테스트를 계산한다. 이어 balanced matrix의 스펙트럴 에너지 기준으로 layer-wise rank를 선택해, 상태공간에서 동시에 도달·관측 가능한 차원을 압축 너비로 변환한다.

- **Empirical Impact**: 실험에서 MNIST의 4층 SiLU DNN은 state order 1024에서 277로 줄이며(상태 72.95%, 파라미터 73.48% 압축) 정확도는 96.60% 대비 95.45%로 유지했다. CIFAR-10에서는 state order 4608→1339(상태 70.94%, 파라미터 83.09% 압축)로 정확도 54.45%→54.44%를 거의 보존하면서 CUDA 추론 지연을 약 3배 줄였다. 기존의 projection-based reduction, pruning, low-rank SVD, dynamic INT8 quantisation 같은 가중치/표현 중심 방법들과 비교해, balanced reachable–observable rank이 정확도 손실이 적은 compact architecture 설계의 원리로 작동함을 보여줬다.



### The Granularity Paradox: How Temporal Disaggregation Inflates In-Sample Fit and Compounds Out-of-Sample Error (https://arxiv.org/abs/2607.05450)
- **Prior Approaches**: 기존 시계열 예측 연구는 시간 단위(집계/세분화)가 데이터 품질과 통계적 추정에 미치는 영향을 주로 다뤘고, 집계는 잡음과 단기 변동을 줄이는 대신 정보 손실과 추정 정밀도 저하를 동반한다고 설명해왔다. 또한 멀티스텝 예측은 recursive/direct/MIMO 같은 전략으로 설계되지만, 어떤 ‘시간 grain’ 선택이 recursive 피드백에서의 누적 오차를 어떻게 키우는지 자체를 통제요인으로 분리한 실증은 부족했다.

- **Core Contribution**: 이 논문은 time-series forecasting에서 ‘Granularity Paradox’를 정식화한다. 세분화로 표본 수 N은 늘지만 예측 단계 수 H가 커지면서, recursive/시계열 상태갱신 구조가 오차를 누적·증폭시켜 out-of-sample 정확도를 악화시키는 임계구간(비단조 임계 구조)을 보여준다.

- **Technical Challenges**: 문제는 모델이 점별 성능은 좋아 보여도(예: RMSE/MAE, out-of-sample R2) 실제 planning horizon 동안의 누적 편차가 얼마나 커지는지 놓치기 쉽다는 점이다. 이를 해결하기 위해 목표 지향 누적 지표 TPFE를 포함해 granularities(연/분기/월/격주/주/일) 전반에서 pointwise 지표의 방향 변화와 TPFE의 방향 변화를 함께 보는 consensus-dissensus 진단을 제안한다.

- **Empirical Impact**: 13년 공공조달 데이터로 10개 모델(naïve, 통계, ML, 딥러닝)을 6개 grain에 대해 벤치마크한 결과, 재귀형 모델은 Daily처럼 고빈도에서 급격히 붕괴하고(LSTM은 U-shaped, Holt-Winters는 최악), 반면 Linear Regression은 전 구간에서 TPFE가 비교적 안정적이었다. 또한 점별 지표만 보면 Persistence와 같은 기준모델이 좋아 보일 수 있지만, 누적 지표 TPFE를 넣으면 실제 의사결정 목표에 대한 모델 부적절성이 드러나 표준 평가 관행을 보완해야 함을 실증적으로 제시한다.



### Geometry-Aware Infrastructure-Anchored Denoiser for UWB Sensing and Work-Zone Reconstruction (https://arxiv.org/abs/2607.05449)
- **Prior Approaches**: 기존 work-zone 인식은 TTCD(일시적 교통통제 장치) 탐지나 HD-map 갱신처럼 요소 단위로 접근하는 경우가 많아, 드라이버블 경계는 암묵적 부산물로 취급되기 쉽습니다. UWB 기반 V2I 연구도 대개 anchor별 ranging 오차(평균 오차, LOS/NLOS 분류 등)를 줄이는 데 초점을 두며, 경계 형상에서 ‘공간적으로 중요한’ anchor 오차가 미치는 비대칭 영향을 직접 학습목표에 반영하지 못합니다.

- **Core Contribution**: GAIA는 work-zone 경계 복원을 위해 UWB ranging을 ‘경계 지향(boundary-oriented) denoising’ 문제로 재정의하고, denoised 거리들이 boundary reconstruction에 일관되게 기여하도록 학습합니다. 특히 latent anchor-layout 추정과 deterministic distance projection을 결합해, 단순 평균 오차 최소화가 아니라 경계 품질(IoU 등) 관점에서 거리 예측을 유도합니다.

- **Technical Challenges**: 핵심 난제는 NLOS·burst noise·long-tail 오류로 인해 anchor별 측정이 흔들리는데, 이때 경계 복원은 일부 ‘기하학적으로 critical한’ anchor 오차에 크게 좌우된다는 점입니다. GAIA는 PoseMLP Base(고정)로 초기 거리 패턴을 잡고, Temporal Refinement으로 시간·anchor 간 상관을 정리한 뒤, Layout Head로 latent anchor 배치를 추정하고 GeoDist에서 geometry-consistent 거리를 투영해 예측과 공간 정합성을 동시에 맞춥니다.

- **Empirical Impact**: GAIA는 동기화된 UWB·GNSS·IMU로 구성된 실외 데이터에서 전체 range MSE를 PoseMLP 대비 18.4% 낮추고 polygon IoU는 15.5% 높였습니다. 또한 real-data-calibrated stress-test 시뮬레이터로 NLOS 및 long-tail 손상이 심해질 때도 견고성을 확인하며, geometry-aware range denoising이 spatially coherent work-zone reconstruction으로 이어짐을 실증했습니다.



### Scientific Code Search at Scale: A Multi-Domain Dataset and Benchmark (https://arxiv.org/abs/2607.05443)
Comments:
          Datasets and benchmarks publicly released on HuggingFace. Code released on GitHub

- **Prior Approaches**: 기존 코드 검색 벤치마크(CodeSearchNet, CoSQA, CodeXGLUE 등)는 일반 소프트웨어 엔지니어링 질의와 코드 페어에 초점을 둬 과학 컴퓨팅의 도메인 특화 어휘(데이터 포맷, 미션/장비명, 알고리즘 명칭)를 충분히 반영하지 못했다. 또한 과학 소프트웨어는 README 중심으로 문서화가 이뤄지는데, 기존 데이터셋은 과학 연구자가 실제로 겪는 검색 니즈와 정보 제공 방식을 제대로 재현하지 못해 평가의 현실성이 떨어진다는 한계가 있었다.

- **Core Contribution**: 이 논문은 NASA Science Mission Directorate 5개 분과(지구과학, 천문·천체물리, 행성·태양계, 헬리오피직스, 생물·물리과학)에 걸친 5,264개 과학 GitHub 저장소를 도메인 분류하고, 정제된 README와 외부 링크 기반 맥락을 풍부화한 큐레이션 코퍼스를 구축했다. 이를 바탕으로 (1) 분과 전문가가 만든 219개 repository search 질의와 (2) 7개 언어의 117,950개 코드 스니펫 및 119,720개 질의를 포함한 대규모 code snippet retrieval 벤치마크를 제안하며, HuggingFace에 공개했다.

- **Technical Challenges**: 핵심 기술 난제는 과학 분야의 비표준화된 문서/용어 때문에 검색 신호가 README나 docstring에 희소하게 흩어져 있다는 점이었다. 저자들은 LLM(GPT-4.1-mini) 기반으로 저장소 도메인 분류와 README 정제(불필요한 boilerplate 제거), 외부 링크 크롤링 후 LLM 점수로 맥락을 추가하는 2단계 풍부화 파이프라인을 적용해 문서 신호를 보강했고, 이 표현을 사용해 BM25와 dense embedding(일반/도메인 특화), 그리고 Hybrid-RRF/Hybrid-Rerank 같은 최신 IR 패러다임을 공정 비교할 수 있게 했다.

- **Empirical Impact**: 실험 결과 repository search는 도메인마다 성능 격차가 크게 나타났고(MRR@10 약 .18~.87), 특히 문서 품질과 용어 표준화(FITS, WCS 등)의 영향을 강하게 받았다. 코드 스니펫 retrieval은 docstring 기반 질의는 잘 맞추지만 identifier 기반 질의는 급격히 어려워지는 현상(MRR@10 약 .76 vs .25)을 보였고, 일반 텍스트 임베딩(예: Qwen3-Embedding-0.6B)이 도메인 과학 텍스트 임베딩보다 더 강하게 코드 수준 검색에 전이됨을 확인했다; 이는 과학 코드 탐색에서 문서 문화(README vs docstring), 컨텍스트 확장, 그리고 코드-aware 검색 표현이 모두 중요하다는 실증적 근거를 제공한다.



### PORTS: Preference-Optimized Retrievers for Tool Selection with Large Language Models (https://arxiv.org/abs/2607.05441)
Comments:
          Please cite the definitive, peer-reviewed version of this article published in the Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, edited by Christos Christodoulopoulos et al., Association for Computational Linguistics, pp. 10007-10030, 2025. DOI: this https URL

- **Prior Approaches**: 기존 연구는 방대한 tool collection에서 필요한 도구를 고르기 위해 retrieval 기반 사전선택을 활용했다. 하지만 retriever는 LLM의 tool-calling과 별도 학습되는 경우가 많아, 실제 tool 호출 성능과 잘 정렬되지 않는 misalignment 문제가 있었다.

- **Core Contribution**: 이 논문은 tool 선택 목적에 맞춘 retriever 학습 방법 PORTS를 제안한다. PORTS는 frozen LLM에서 얻은 perplexity-inspired preference 신호로 선택 확률과 downstream 성능 간 상관을 최적화하고, 문서 문자열 간 contrastive semantic loss도 함께 걸어 tool 선택 정합성을 높인다.

- **Technical Challenges**: 핵심 기술 난제는 retrieval 신호가 LLM의 tool-calling 동작 및 성능과 정렬되도록 preference를 설계·학습하는 것이다. PORTS는 selection probability–성능 상관 최적화와 문서 문자열의 대비 학습을 동시에 수행해, 독립 학습에서 비롯된 불일치를 줄이도록 설계했다.

- **Empirical Impact**: PORTS는 6개 데이터셋, 2개 encoder 모델, 3개 LLM(사전지식 다양)에서 tool selection accuracy를 유의미하게 개선하며 성능을 폭넓게 입증했다. 또한 계산 부담이 낮은 alignment로 새로운 쿼리와 새로운 tool로의 generalization도 잘 되어, 변화하는 toolset을 전제로 한 실무 적용 가능성이 강조된다.



### Modality Relevance is not Modality Utility: Post-hoc Selective Modality Escalation for Cost-Aware Multimodal RAG (https://arxiv.org/abs/2607.05438)
- **Prior Approaches**: 기존 멀티모달 RAG는 비용 비대칭 때문에 “텍스트+테이블만” 쓰거나 “모든 이미지에 VLM을 적용”하는 식으로 의사결정을 고정하는 경우가 많았다. 적응형 접근도 대개 질문만 보고(pre-retrieval) 어느 모달리티에 예산을 쓸지 라우팅하지만, 답을 만들기 전이라 실제 유용성보다 겉보기 관련성에 치우칠 수 있다. 또한 에이전트형 파이프라인은 단계별로 더 많이 호출하지만, 호출량이 늘어 비용 통제가 어려운 편이다.

- **Core Contribution**: 이 논문은 모달리티 결정 시점을 “답을 먼저 초저비용으로 초안 생성(텍스트+테이블)”한 뒤로 옮겨, 필요한 모달리티만 사후적으로 escaltion(확장)하자는 post-hoc selective modality escalation을 제안한다. 핵심은 (query, draft answer, evidence) 튜플을 verifier가 보고, 이미지 모달리티가 “부족해서” 틀렸는지 모달리티 갭을 특정한 뒤에만 VLM 비용을 지불한다. 마지막으로 value-of-escalation을 캘리브레이션해, 예상 정확도 이득이 시각 비용을 정당화할 때만 호출하도록 운영 지점을 고정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “모달리티 관련성(relevance)”과 “정답에 필요한 실제 유용성(utility)”이 다르다는 점을 라우팅 신호로 분해해 내는 것이다. 이들은 MultiModalQA의 oracle headroom 분석을 통해 겉보기 관련성이 약한 예측자임을 보였고, verifier의 need_image 같은 투표를 그대로 쓰기보다 draft 이후 특징을 이용해 keep vs escalate의 정확도 차이를 예측하는 캘리브레이션 모델로 해결했다. 이후 임계값으로 accuracy–escalation(비용) 프런티어를 타겟하도록 설계해, 무작정 호출을 줄이면서도 이득이 있을 때만 시각 확장을 수행한다.

- **Empirical Impact**: MultiModalQA에서 항상-on VLM 파이프라인과 동일 예산(난할당) 비교를 엄격히 수행했을 때, post-hoc 라우팅은 정확도를 유지하면서도 훨씬 적은 visual call 비율로 그 성능을 회복한다. 구체적으로 learned pre-retrieval 라우터를 예산이 커질수록 더 크게 추월하며, oracle escalating(필요한 경우에만 escaltion)과도 대부분의 격차를 메웠다. WebQA 교차 검증에서는 모달리티가 본질적으로 분리된 경우엔 이득이 줄고 비용 제어 효과로 수렴해, 제안 메커니즘(모달리티 relevance–utility gap)에 상응하는 현상임을 뒷받침한다.



### CHARLIE: An On-Premise Multi-Agent Retrieval-Augmented Generation System for Evidential Reasoning in Forensic Scienc (https://arxiv.org/abs/2607.05428)
Comments:
          10 pages, 1 figure. Archival version of a paper presented at RELAF 2026: 1st Workshop on Reasoning with Evidence in Law Enforcement and Forensics, co-located with ICAIL 2026, Singapore, June 2026

- **Prior Approaches**: 기존 연구들은 법적 추론을 argumentation 프레임워크나 확률·논리 혼합 모델로 다루되, 증거 요소가 이미 구조화돼 있다는 가정이 많았습니다. 또한 Bayesian networks, knowledge representation 스키마, CASE 같은 방향은 출처·구조화를 강조하지만 대규모 비정형 문서에서의 자동 전처리에는 병목이 남아 있습니다. RAG와 LLM 에이전트도 있으나, 대개 기밀성·감사가능성·evidential integrity 같은 사법 제약을 구조적으로 보장하지 못했습니다.

- **Core Contribution**: Charlie는 온프레미스 multi-agent RAG를 통해 디지털 포렌식 환경에서 ‘구조화된 증거 처리’를 수행하는 도메인 특화 인프라를 제시합니다. 핵심은 로컬 retrieval, task decomposition, structured memory, verification(검증) 메커니즘을 통제된 에이전트 오케스트레이션으로 묶어, 문서 간 상관과 추출 결과의 출처 연결을 유지하는 데 있습니다. 무엇보다 cloud 외부 전송 없이 데이터 소버린티와 chain-of-custody에 맞춘 아키텍처를 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 과제는 여러 문서를 넘나드는 추출을 컨텍스트 윈도우 한계 없이 처리하면서, 환각·누락을 법정 수준의 절차 추적성으로 통제하는 것입니다. Charlie는 쿼리 분류→원자 subquery로 분해→RAG로 병렬 실행→구조화된 메모리에 provenance 링크 저장→완결성·일관성 검증→최종 synthesis의 흐름으로 에러 전파를 줄입니다. 또한 2단계 retrieval( dense 검색 후 reranking)과 stage별 structured prompting, 그리고 LangGraph 기반 결정적 실행 그래프로 감사 가능성을 높였습니다.

- **Empirical Impact**: Charlie는 벤치마크가 아닌 실제 기관 워크플로우 케이스 스터디에서 구조화된 다문서 추출과 종단형(연도별) 포렌식 인텔리전스 통합을 지원함을 보였습니다. 교통사고 보고서 약 2,000건 규모 연간 처리에서는 클러스터·시간대 분석용 데이터셋 생성과 시계열 모니터링에 활용됐고, 여성폭력(살해) 사건 문서에서는 5개 차원 패턴(도구, modus operandi, 피해자/가해자, 물질 사용 등) 추출과 상관에 기여했습니다. 결과적으로 ‘고위험 사법 환경에서 온프레미스 에이전트 오케스트레이션 RAG가 절차 정합성과 함께 스케일업 가능하다’는 실무적 청사진을 제공했다는 점에서 의미가 큽니다.



### When AI Classifies: What Counts as Public Administration? (https://arxiv.org/abs/2607.05420)
- **Prior Approaches**: 기존 연구는 Web of Science와 OpenAlex 같은 데이터에서 연구를 저자 키워드 중심, 인용 기반, 혹은 분류 체계에 따라 표현해 왔다. 다만 이런 표현 방식들이 서로 다른 방식으로 ‘공공행정(PA)’과 ‘AI-in-PA’(인공지능 관련 공공행정) 지식을 포착할 수 있다는 점은 충분히 비교되지 않았다. 그 결과 서로 다른 표현이 만들어내는 지식 경계와 지형이 얼마나 달라지는지에 대한 실증적 감각이 부족했다.

- **Core Contribution**: 이 논문은 Web of Science와 OpenAlex를 이용해 저자 정의(author-defined), 인용 기반(citation-driven), 그리고 AI 보조(AI-assisted) 표현 등 다섯 가지 접근을 체계적으로 비교한다. 특히 각 표현이 다루는 논문 집합, 출판 유형, 출판처, 시간적 발전, 그리고 주제 클러스터링과 구조가 크게 달라짐을 보여준다. 더 나아가 표현들 사이에 출판물과 출판처 중복이 거의 없어, 서로 다른 방법이 동일한 하위집합이 아니라 다른 지식 영역을 잡아낼 수 있음을 강조한다.

- **Technical Challenges**: 핵심 기술적 난제는 서로 다른 대표화(표현) 규칙이 어떤 방식으로 말뭉치(코퍼스) 크기와 시간 흐름, 그리고 주제 구조를 재구성하는지 정량 비교하는 것이다. 논문은 다섯 접근을 동일한 데이터 소스에서 구축한 뒤, 문헌 규모와 메타데이터(출판 유형·출판처), 시간 전개, 주제 클러스터링 구조의 차이를 함께 분석해 편차의 원인을 관찰한다. 또한 겹침이 거의 없다는 결과를 통해 ‘알고리즘적 지식 조직화’가 분류와 이해에 해석적(interpretative) 영향을 준다는 점을 논증한다.

- **Empirical Impact**: 실증 결과, AI 보조 분류와 표현은 중립적이지 않고 해석을 담고 있으며 자기강화적으로 작동할 가능성이 제기된다. 이는 학제 간 연구가 어떻게 보이고(visibility), 지적 구조가 어떻게 보존·재배열되며, 학문 경계가 어떻게 그려지는지에 직접적인 함의를 준다. 결론적으로 사람의 학문적 판단은 필수이며, AI 기반 분류는 대체가 아니라 보완 역할로 다뤄야 한다는 메시지를 남긴다.



### Contrastive Predictive Coding with Compression for Enhanced Channel State Feedback in Wireless Networks (https://arxiv.org/abs/2607.05419)
Comments:
          Accepted for publication in IEEE Transactions on Neural Networks and Learning Systems

- **Prior Approaches**: 기존 3GPP 기반 연구는 CSI compression과 CSI prediction을 분리해 다뤘다. 압축은 주로 Autoencoder/ResNet 등으로 복원 정확도를 키우는 방향이었고, 예측은 LSTM이나 Transformer로 원시 CSI 시퀀스를 직접 예측해 라벨 의존성과 장기 의존성 한계가 지적돼 왔다. 또한 3GPP AI 학습 패러다임은 Type-1/Type-2/Type-3로 나뉘지만, 압축·예측의 공동 최적화는 충분히 다뤄지지 않았다.

- **Core Contribution**: 논문은 3GPP-compliant CSI compression 파이프라인에 CPC(Contrastive Predictive Coding)를 통합해, 압축과 채널 aging 완화를 하나의 프레임워크로 해결하는 것을 제안한다. 원시 CSI를 직접 예측하지 않고 미래의 latent representation을 예측하도록 설계해, 높은 차원의 예측 부담을 줄이면서도 시간적 예측 일관성까지 함께 최적화한다. 학습 손실은 복원 품질을 위한 1-SGCS와 시간 예측 구조를 위한 InfoNCE를 결합해, 압축만 하던 기존 방식의 한계를 겨냥한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘압축 오버헤드(레이트) 제약 하에서’ 예측성을 강화하되 UE/BS 계산량과 지연을 늘리지 않는 것이다. 이를 위해 두 변형을 제시한다: CPC-before-Compression은 인코더 측에서 GRU 기반 autoregressive 모듈로 latent을 예측한 뒤 압축하며, CPC-after-Compression은 UE 비용을 줄이기 위해 BS 디코더에서만 시간 모델링을 수행한다. 또한 두 변형 모두 quantized linear bottleneck과 64-bit 피드백 오버헤드를 유지해 표준 파이프라인 호환성을 확보했으며, CPC-before-Compression은 GRU pruning으로 복잡도를 추가 절감한다.

- **Empirical Impact**: Nokia, Oppo, CATT의 3GPP-compliant 데이터셋 평가에서 CPC-before-Compression은 90% 이상 복원 정확도 수준을 보이며 3GPP baseline 대비 decoder GFLOPs를 32x 낮춘 것으로 보고된다. CPC-after-Compression은 인코더 footprint와 64-bit feedback overhead를 baseline과 동일하게 유지하면서 성능을 보존하는 방향을 택했으며, 시퀀스 예측이 압축 병목에 의해 제한됨을 실험적으로 확인한다. 전반적으로 ‘표준화된 CSI 피드백 파이프라인 내에서 age-aware 압축-예측 공동학습’ 가능성을 보여, 향후 5G/이후 MU-MIMO에서 aging에 강한 실용적 피드백 설계로의 파급이 기대된다.



### Why does AI unlock new possibilities in STEM education? A Bibliometric Analysis of Trends and Future Agenda (https://arxiv.org/abs/2607.05412)
Comments:
          Accepted by ISLS26 conference

- **Prior Approaches**: 기존 STEM 교육 연구는 개인화와 학제 간 통합을 별개 이슈로 다루는 경향이 있어, 교육 생태계가 어떻게 변화하는지의 ‘작동 메커니즘’을 체계적으로 설명하기 어려웠다. 기술 측면에서는 지능형 튜터링 시스템 등 개별 도구 중심 접근이 주를 이뤘지만, 학습 목표가 지식 전달에서 역량 개발로 이동하는 흐름을 통합해 추적하긴 제한적이었다.

- **Core Contribution**: 이 논문은 2015~2025년 242편의 문헌을 대상으로 서지계량(bibliometric) 분석과 지식 지도(knowledge maps)를 구성해, AI가 STEM 교육 생태계를 재편하는 진화 경로를 보여준다. 특히 연구 흐름이 지능형 튜터링 시스템에서 탐구 기반 학습과 computational thinking(컴퓨팅 사고) 함양으로 이동했음을 정리하며, 그 중심에 LLM 기반의 지능형 scaffolding(지원/발판) 역할을 제시한다.

- **Technical Challenges**: 핵심 기술적 도전은 방대한 연구 문헌에서 시간에 따른 주제 전환과 영향 관계를 재구성하는 작업이다. 이를 위해 논문은 서지계량 기법으로 네트워크와 지식 지도를 만들고, LLM이 촉발한 교육 패러다임 변화(지식 전달→역량 개발)를 관찰 가능한 지표로 연결해 설명한다.

- **Empirical Impact**: 실증 결과는 STEM 교육 분야가 LLM의 확산과 함께 ‘이해의 문턱을 낮추는’ 지능형 지원 체계로 재편되고 있음을 보여준다. 이는 교육 플랫폼·콘텐츠 설계 관점에서 개인화가 단순 기능이 아니라 학습 역량 형성의 핵심 메커니즘임을 시사하며, 향후 연구의 체계적 방향 설정에도 기여한다.



### The GenAI Skill Bypass: Mapping Divergent Pathways of University Students and Staff AI Literacy (https://arxiv.org/abs/2607.05411)
Comments:
          18 pages, 7 figures, 3 tables

- **Prior Approaches**: 대학들은 GenAI 리터러시를 키우기 위해 연수 프로그램을 운영하고, 과목 안에 GenAI 역량을 포함하는 방식으로 대응해 왔다. 다만 기존 교육 프레임워크는 GenAI 리터러시가 기초 개념 이해→창의적 활용처럼 선형으로 성장한다는 가정을 두는 경우가 많다. 이 접근은 학습자의 실제 역량 형성 경로를 충분히 반영하지 못한다는 한계가 있다.

- **Core Contribution**: 이 논문은 “선형 진행” 가정을 심리측정(psychometric) 관점에서 검증한다. 학생·학계·전문 스태프(n=158) 대상으로 분류체계 기반 self-assessment 도구를 Rasch measurement theory와 Guttman ordering으로 분석해, GenAI 기술의 지각된 난이도 잠재 순서를 매핑했다. 그 결과 학생은 전통적 선형과 달리 ‘inverted’ 프로파일(창작 같은 고수준 작업을 먼저 익히고, 기초 개념은 뒤늦게 습득하는 양상)을 보였다.

- **Technical Challenges**: 기존 교육처럼 기술 숙련이 정해진 순서대로 이동한다는 전제를 그대로 사용하면, 집단별 실제 학습 경로가 숨겨질 수 있다. 논문은 Rasch 측정과 Guttman ordering을 통해 집단 간 지각된 난이도 체계의 차이를 계량적으로 드러냈고, 학생과 학계의 기술 난이도 상관도(r=0.188)가 약하다는 점을 확인했다. 또한 고수준 prompting에서의 자신감이 AI mechanics의 낮은 리터러시를 가리는 ‘skill bypass’와 ‘취약한 fluency’라는 해석을 제시한다.

- **Empirical Impact**: 실증 결과는 ‘one-size-fits-all’ 커리큘럼이 학습자 집단의 역량 형성 차이를 제대로 반영하지 못할 수 있음을 보여준다. 학생이 고수준 창작을 먼저 습득하는 패턴이 반복된다면, 단일 선형 교육은 기초 이해의 공백을 방치할 위험이 있다. 논문은 진단 기반(diagnostic-driven)·모듈형(modular) 개입이 human-AI synergy를 실제로 강화하는 근거가 될 수 있다고 주장한다.



### CANONIC: Governance Is Compilation (https://arxiv.org/abs/2607.05410)
Comments:
          28 pages, 4 figures. Pre-registered cross-provider evaluation harness and per-regime results at this http URL. Construction claims resolve to commands run at the evidence-window-close ref (see Appendix C)

- **Prior Approaches**: 기존 AI 콘텐츠 거버넌스는 주로 생성 후 탐지·공개·리뷰·스타일 점검처럼 ‘사후 품질/신뢰도 필터’에 의존해왔다. 하지만 CANONIC은 이런 방식이 정작 ‘코퍼스에 무엇을 들일지’의 구조적 기준을 강제하지 못해 결국 같은 종류의 slop이 축적된다고 본다. 또한 단순한 AI 생성 여부나 읽기 좋음 같은 축으로는 저신뢰 텍스트와 고신뢰 텍스트를 안정적으로 가르기 어렵다고 지적한다.

- **Core Contribution**: CANONIC은 LLM이 만든 산문을 ‘진실 판정’이 아니라 ‘컴파일처럼 문서가 규격에 맞게 연결됐는지(증거가 닻을 내렸는지)’로 코퍼스 경계에서 검사하는 governed intelligence를 제안한다. 이를 위해 Triad, Inheritance, Introspection 세 가지 공리를 문법·스코프 해석·타입 시스템에 1:1로 대응시키고, 승인/거절을 결정하는 admission 절차를 선형 시간의 판정 문제로 만든다. 결과적으로 슬롭(slop)을 기계가 판정하지는 못하되, 무엇이 ‘감사 가능하게’ 기록되었는지는 end-to-end로 재현·검증 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 구조적 체크(정의/증거/범위가 존재함)가 실제 내용의 진실성(truth)과 상관되지 않을 수 있다는 점이다. CANONIC은 이를 인정하면서도, validator가 품질을 점수화하는 대신 ‘presence’만 확인하도록 3단 게이트(용어 VOCAB, git 기반 evidence ledger, 증거 evidence window)를 이진적으로 강제해 불계획·무근거 입력이 코퍼스에 들어오는 경로를 차단한다. 동시에 ‘가짜인데도 존재를 만들 수 있는’ 진공 충족·증거 위조·창(window) 조작 같은 우회가 가능함을 분석하고, 그 한계를 독립적 의미 검증으로 메우는 방향이 필요하다고 결론낸다.

- **Empirical Impact**: 논문은 사전 등록된 cross-provider 벤치마크에서 구조적 admission이 slop을 실질적으로 막지 못한다는 결과를 보이며, slop이 알고리즘이 계산하는 성질이 아니라 도메인 전문성의 verdict임을 경험적으로 뒷받침한다. 다만 구조적 게이트가 달성하는 것은 ‘진실 필터’가 아니라 ‘감사 가능성(accountability)의 경계 설정’이며, 정의·커밋·증거 창으로 모든 주장에 대한 추적 경로를 제공한다. 결론적으로 학계/임상/정책처럼 검증 비용이 큰 영역에서, 생성 속도를 따라잡는 대신 ‘감사와 재현이 가능한 기록 구조’를 표준화하는 방향의 영향력이 크다고 평가된다.



### Automated Recommendation of Programming Learning Content Using Pattern-based Knowledge Components (https://arxiv.org/abs/2607.05409)
Comments:
          Paper accepted to the 10th Educational Data Mining in Computer Science Education (CSEDM) Workshop in Seoul, Korea

- **Prior Approaches**: 기존에는 키워드나 n-gram 같은 표면 신호로 학습 자료를 연결하거나, AST 전체/메타데이터 기반 유사도를 계산해 추천하는 방식이 주로 쓰였다. 또 ontology 기반 개념 태그를 사람이 설계해 연결 정확도를 높이기도 했지만, 도메인 모델과 전문가 라벨링 부담이 커서 확장에 한계가 있었다. 최근에는 code2vec·embedding 기반으로 자동 유사도를 찾으려 했지만, 개념적으로 같은 의도를 공유해도 구조·구현이 다르면 매칭이 흔들린다는 문제가 남았다.

- **Core Contribution**: 이 논문은 프로그래밍 학습 활동을 Knowledge Components(KCs)로 자동 변환하고, KC 집합의 유사도로 worked example와 practice를 함께 묶어 추천하는 프레임워크를 제안한다. 특히 KCs를 이름/태그가 아니라 AST에서 반복되는 의미 있는 프로그래밍 패턴(patter-based KC)으로 정의해, “무엇을 이해해야 하는가” 수준에서 정렬되도록 설계했다. 또한 추천 근거를 코드의 관련 구간 단위로 투영해, 왜 그 자료를 추천했는지 감사(audit) 가능하게 만든 점이 핵심이다.

- **Technical Challenges**: 핵심 과제는 코드에서 KC를 자동 추출할 때 의미 동치(개념은 같은데 문법/표현이 다른 경우)를 안정적으로 잡아내는 것이었다. 이를 위해 (1) AST 부분트리를 다중 해상도로 뽑고 SANN으로 중요한 서브트리에 가중치를 준 뒤, (2) 식별자/리터럴 등 우발적 차이를 익명화하고 간단한 문법 변형을 정규화하며, (3) 순차 공존 정보를 담기 위해 LSTM 기반 β-VAE로 잠재 공간을 학습해 구조·의미가 가까운 패턴을 형성한다. 마지막으로 K-means로 클러스터링해 패턴 기반 KCs를 만들고, IDF로 흔한 KC의 영향은 줄인 IDF-weighted 지식 벡터를 cosine similarity로 비교해 추천한다.

- **Empirical Impact**: PCEX의 전문가가 번들로 분류한 파이썬 초급 자료(123,123개 프로그램, 4,949개 번들)에서 평가했으며, Top-5 accuracy·MRR·mAP 모두에서 pattern-based KC가 코드 임베딩(code2vec) 및 다양한 KC/개념 기반 기준선보다 일관되게 우수했다. 번들 단위 tightness와 계층적 클러스터링(ARI, V-measure), 최근접 이웃 분석에서도 전문가 정의 번들을 더 가깝게 복원/정렬하는 경향이 확인됐다. 더불어 IDF 재가중(ablation) 효과가 명확해, 자주 등장하는 패턴을 다운웨이트해야 조기 순위 품질이 크게 개선된다는 실증적 근거를 제공한다.



### Position: Preventing AI-Generated CSAM Necessitates New Approaches to AI Safety (https://arxiv.org/abs/2607.05407)
Comments:
          Accepted (spotlight) in ICML 2026, Position Paper Track. The first two authors contributed equally and may list their names interchangeably

- **Prior Approaches**: 기존 AI 안전 기법들은 대개 데이터 접근성, 투명성, 표준화된 평가가 가능하다는 전제를 둡니다. 하지만 아동 성착취/아동 성적 학대물과 관련된 영역은 윤리·법적 제약이 커서 데이터 감사(auditing), 레드팀(red teaming), fine-tuning 방지 같은 관행을 그대로 적용하기 어렵습니다. 그 결과 이 분야의 위험을 “일반적인 안전 문제”로 다루기엔 공백이 생깁니다.

- **Core Contribution**: 이 논문은 아동 안전을 위협하는 AI 오남용을 AI 안전 연구의 핵심(safety-critical) 차원으로 재정의합니다. 특히 기존 방식의 전제가 맞지 않는다는 점을 기반으로, 데이터 큐레이션부터 모델 설계, 배포, 장기 유지보수까지 개발 전 주기에서의 격차를 체계적으로 정리합니다. 또한 온라인 아동 성착취·학대 전 과정에 걸친 15개의 open problems를 제시해 실무형 연구 의제를 구체화합니다.

- **Technical Challenges**: 핵심 기술 난제는 제약 환경 때문에 dataset auditing, red teaming, fine-tuning prevention 같은 안전 실험·검증 절차의 한계가 동시에 나타난다는 데 있습니다. 논문은 이러한 제한이 평가 설계, 모델 훈련·수정 경로, 운영 중 모니터링에 연쇄적으로 영향을 준다고 지적합니다. 이에 따라 개발 라이프사이클 각 단계에서 실효성 있는 안전장치를 어떻게 구현할지에 대한 방향성을 제안합니다.

- **Empirical Impact**: 논문은 단일 모델 성능 실험보다, 정책·연구·개발자가 참고할 수 있는 안전 프레임과 구체 의제를 제공하는 데 의미가 있습니다. 15개 open problems는 향후 벤치마크와 실증 연구를 설계할 때 바로 활용될 수 있는 체크리스트 역할을 합니다. 이를 통해 “이론적 responsible AI”를 실제 아동 보호로 번역하는 연구 흐름을 촉진하는 것이 목표입니다.



### A Guiding Framework for K-12 Teachers in Creating AI-powered Learning Technologies through Vibe Coding (https://arxiv.org/abs/2607.05406)
- **Prior Approaches**: 기존에는 대규모 언어 모델이 자연어 프롬프트로 코드를 생성해 ‘vibe coding’을 가능하게 한다는 점이 주목받았다. 하지만 K-12 교사가 이 과정을 수업 설계에 활용하도록 돕는, 구체적인 구조화된 가이드가 부족했다.

- **Core Contribution**: 이 논문은 교사를 위한 AI-통합 설계 지원 프레임워크 GAIDE(A Guiding Framework for AI-Integrated Design for Educators)를 제안한다. Design Thinking과 INTERACT에 기반해 교사가 vibe coding으로 AI 기반 학습 기술을 ‘학습자 관점에서 창작’하도록 체계적인 방향을 제공하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 과제는 교사의 설계 역량과 AI 리터러시를 동시에 끌어올리면서, 막연한 코딩 실험이 아닌 설계 프로세스로 연결되게 하는 것이다. 저자들은 8주 워크숍에서 3명의 교사와 4명의 멘토의 CORDTRA 상호작용 분석으로 초기 프레임워크를 다듬었고, 사전·사후 인터뷰의 질적 분석으로 학습과정을 검증했다.

- **Empirical Impact**: 8주 워크숍 결과, 질적 사전·사후 인터뷰에서 교사의 AI literacy 향상이 관찰되었다. 또한 ‘만들면서 학습하기(learning-by-creating)’가 전문성 개발로 이어질 수 있음을 보여주며, 교육 현장의 AI 통합과 AI 리터러시 확산에 의미 있는 시사점을 제공한다.



### CCBENCH: Assessing LLM Cultural Competence via Implicitly Signaled Norms using Health Queries (https://arxiv.org/abs/2607.05405)
Comments:
          34 pages

- **Prior Approaches**: 기존 문화 관련 평가들은 대개 사용자가 ‘어느 문화에 속하는지’를 명시적으로 밝히는 방식(이진 속성)이나, 문화를 몇 개 변수로 축약해 지식/형식의 정답성만 보려는 경향이 컸습니다. 그 결과 실제 대화에서 문화적 가치는 암묵적으로 드러나며 개인마다 규범을 섞어 따르는 ‘연속체’라는 점이 충분히 반영되지 못했습니다.

- **Core Contribution**: 이 논문은 문화적 역량을 ‘문화 소속’이 아니라 ‘규범 준수(norm adherence) 상태의 연속체’로 재정의하고, LLM이 대화 맥락에서 이를 추론해 반응하는지를 평가하는 CCBENCH를 제안합니다. 의료 사례로 CCBENCH-Health를 만들었고, 6개 문화에 대해 이론적 근거가 있는 60개 페르소나와 실제 포럼 기반 52개 건강 질문(총 3,120개 상호작용)을 구성해 측정합니다.

- **Technical Challenges**: 핵심 난제는 사용자가 문화적 배경을 직접 말하지 않는 환경에서, LLM이 규범 준수 상태를 ‘대화 히스토리’에서 암묵적으로 읽어내야 한다는 점입니다. 이를 위해 값(value)-규범(norm) 계층 구조로 페르소나를 생성하고, 규범별 체크리스트 기반의 채점으로 응답이 ‘따름/회피/중립’ 중 페르소나 의도에 맞는지를 LLM 평가기로 정량화했습니다.

- **Empirical Impact**: 다섯 개 주요 LLM 모두에서 문화적으로 적절한 응답률이 20~30% 수준에 머물렀고, Culture-CoT처럼 대화 이력의 문화 단서를 단계적으로 보게 해도 개선은 평균 3~5%p에 그쳤습니다. 특히 ‘규범을 따르는(Follow)’ 경우가 ‘규범을 피하는(Avoid)’ 경우보다 훨씬 어렵고, 아프가니스탄 맥락에서는 성능이 8.8%로 낮아 암묵적 서구 기준선 편향이 지속됨을 시사합니다. 반면 특정 맥락의 표면적 커뮤니케이션 스타일은 상대적으로 더 잘 맞추기도 하지만, 전반적 결핍은 “명시적 메타데이터 제공만으로는 해결되지 않는다”는 방향으로 귀결됩니다.



### The Jagged Global Economy: Frontier AI Unevenly Exposes National Economies (https://arxiv.org/abs/2607.05404)
Comments:
          Website (including code and data): this https URL

- **Prior Approaches**: 기존 연구는 작업(task)이나 직무(occupation) 단위의 AI 노출(exposure)을 추정해 생산성 변화나 고용 충격을 설명하는 데 집중해 왔습니다. 그러나 대부분은 미국 중심이거나, 국가 간 노동시장 구성 차이를 충분히 반영하지 못해 국가별 이질성을 제대로 비교하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 141개국을 대상으로 국가 수준의 frontier AI exposure 지표를 새로 제안합니다. 직업별 노출 점수와 국가의 직업별 고용구조를 결합해 “어느 나라가 frontier AI의 능력 편차에 더 많이 노출돼 있는지”를 국제 비교 가능하게 정량화합니다.

- **Technical Challenges**: 핵심 과제는 직업 단위 노출이 실제 국가의 노동시장 충격을 얼마나 대표하는가이며, 특히 task-occupation 집계 과정의 편향 가능성입니다. 연구진은 ILO의 ISCO-08 기반 고용통계와 국제적으로 정리된 occupation exposure 추정치를 사용해 국가별 가중 평균으로 exposure를 계산하고, 직업 노출 추정치 선택 변경에도 결과가 견고함을 검증했습니다.

- **Empirical Impact**: 결과적으로 고소득국의 exposure가 저소득국보다 크게 높고, 유럽 및 중앙아시아는 사하라 이남 아프리카 대비 약 50% 더 노출된 것으로 나타났습니다. 또한 91%의 국가에서 여성의 exposure가 남성보다 높았는데, 이는 여성의 white-collar 및 sales 직군 집중이 구조적으로 작동한 때문입니다. 더 나아가 이 지표가 Anthropic·Microsoft·OpenAI의 국가별 채택/사용 통계와 단조(함수형) 관계로 예측됨을 보이며, 러시아로부터의 remittance 의존 같은 간접 경로까지 포함하면 정책 대응의 일반화가 어렵다는 점을 시사합니다.



### AI tools in Arab University English classrooms: Looking back and forward (https://arxiv.org/abs/2607.05403)
Comments:
          18 pages, 2 Tables

- **Prior Approaches**: 이 논문은 2023년 1월 1일부터 2025년 8월 31일까지 아랍권 대학(Arab University Classrooms, AUCs)에서 EL2 학습자를 지원하기 위해 사용된 AI 도구 연구를 종합한다. Google Scholar, Web of Science, Scopus에서 PRISMA 지침 기반 검색으로 184편을 찾았지만, 학술지에 게재된 연구 중 엄격한 포함 기준을 통과한 것은 11편에 그쳤다. 기존 연구들은 AI 활용의 교육적 잠재력은 논의하지만, 효과가 글의 표면적 개선이나 말하기로까지 일관되게 확장되는지는 연구마다 흔들리는 편이었다.

- **Core Contribution**: 핵심 기여는 AUC 맥락에서 EL2 학습을 지원하는 AI 도구의 실제 사용 효과를 ‘실증 연구’ 중심으로 정리해 근거 기반 통합 방향을 제시하는 데 있다. 특히 글쓰기에서 초안 작성(drafting), 수정(revision), 연습(practice) 단계에서 학습자 태도가 전반적으로 긍정적임을 종합했다. 또한 말하기(speaking)는 교사의 중재(teacher mediation)에 따라 달라지는 경향을 밝혀, 무분별한 도구 도입보다는 수업 설계가 중요하다는 메시지를 강화한다.

- **Technical Challenges**: AI 도구 통합이 실효를 내기 위해서는 수업 전반에 걸친 스캐폴딩(scaffolded integration) 설계와 교사 역량이 필수라는 문제가 제기된다. 논문은 연구들에서 드러난 공통 과제로 AI 의존(over-reliance)을 줄이기 위한 성찰 과제(reflective tasks) 필요성을 강조한다. 이를 위해 교사 훈련과 함께, AI 출력물을 수업 목표에 맞게 점검·피드백하는 절차를 포함하는 방식으로 해결책을 제안한다.

- **Empirical Impact**: 실증 결과는 표면 수준(surface-level) 결과 개선에서는 일관성이 가장 높았고, 고차 글쓰기 품질(higher-order writing quality) 측면에서도 비교적 안정적인 개선 근거가 관찰됐다. 반면 말하기 능력(speaking proficiency)은 결과가 혼재되어 있으며 교사 중재 여부에 따라 성과가 좌우되는 경우가 많았다. 저자들은 이를 바탕으로 아랍 대학이 증거 기반으로 AI를 도입할 때의 실무 가이드와 향후 연구 아젠다를 제안하며, 교육 현장 적용의 방향성을 구체화했다.



### Catalyst Papers in Artificial Intelligence Research: A Landscape on ICLR from 2017 to 2025 (https://arxiv.org/abs/2607.05401)
- **Prior Approaches**: 기존 연구는 인용 네트워크 기반의 Consolidation/Destabilization(CD) 같은 “파괴성” 지표로 미래 영향의 흔적을 찾거나, 리뷰 점수와 인용 성과의 상관을 다뤄왔지만 둘을 장기적인 연구 궤적 변화로 연결하는 데는 한계가 컸습니다. 특히 CD는 희소 네트워크에서 값이 쌍봉 형태로 퇴화하고, 리뷰 신호는 차후 연구 방향 전환을 예측하는지 대규모에서 검증되지 않았습니다. 또한 저널/저명지 중심의 bibliometric 분석은 per-paper reviewer score 같은 컨퍼런스 리뷰 정보를 포함하지 못한다는 제약이 있었습니다.

- **Core Contribution**: 이 논문은 ICLR 2017–2025의 36,113개 논문에서 “catalyst(촉매)”를 정의해, 어떤 논문의 후속(후예) 연구가 향후 연구 주제·인용 흐름을 실질적으로 바꿔놓는지를 계량화합니다. 이를 위해 topic initiator(TI), topic bridge(TB), within-topic redirector(WR), simultaneous(SC), recognition-misaligned(RM)의 5유형을 다중 레이블로 운영 정의하고, 각 유형이 궤적 변화에 어떤 방식으로 선행하는지까지 분해해 봅니다. 동시에 submission time의 OpenReview 리뷰 점수(승인/거절 포함)가 이런 촉매성을 예측할 수 있는지도 함께 질문합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 촉매성을 측정하는 지표들이 sparse citation 구조에서 어떻게 편향되는지, (2) 방향성 있는 임베딩 기반 측정과 리뷰/문서 의미 기반 측정이 서로 다른 신호를 잡는지, (3) 지표를 대규모로 검증할 때 주제 코호트 효과를 통제하는 방법을 마련하는 것입니다. 저자들은 ICLR 내부 방향 그래프에 대해 EDM(방향-aware Embedding Disruptiveness Measure)을 포함해 CD, node2vec, 그리고 제목·초록만으로 점수를 내는 LLM-based semantic rater를 같은 코퍼스에서 직접 비교하며, 주제는 UMAP+HDBSCAN으로 113개 클러스터를 구성합니다. TI는 후속 1–3년 내 주제 점유율의 가속을, TB는 교차-주제 인용 흐름의 급증을, WR은 주제 임베딩 centroid의 방향성 있는 이동을, SC는 동시 발견 패턴을, RM은 높은 EDM 대비 낮은 리뷰 점수(또는 논란형 분산) 조합을 각각 기준으로 설정합니다.

- **Empirical Impact**: 실험 결과, 촉매 식별 성능에서 EDM이 가장 강했으며 ERS(상위 2% ICLR-internal 인용) 기준 ROC-AUC 0.83으로 CD(0.60), node2vec(0.49), LLM rater(0.42)보다 우수했습니다. 메커니즘 측면에서는 TI가 연도-매칭 대조군 대비 주제 점유율을 7.55배 앞서 성장시키고, TB는 해당 클러스터로의 교차-주제 인용 흐름을 11.52배(평균) 끌어올리는 선행 효과를 보였습니다. 결정적으로, OpenReview 리뷰 점수는 미래 disruptiveness/촉매성(특히 EDM)과 거의 직교에 가까워 |ρ|≤0.005 수준이었고 승인/거절 논문 간 평균 EDM 차이도 유의하지 않았습니다(p=0.11), 즉 “리뷰 신호만으로” 궤적 전환을 조기 포착하기는 어렵다는 결론을 뒷받침합니다.



### Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving (https://arxiv.org/abs/2607.05399)
- **Prior Approaches**: 기존 KV cache 압축 연구는 quantization, pruning, merging 등 각각의 기법을 주로 개별적으로 평가하거나, 서로 다른 모델·데이터셋·압축 예산·서빙 스택에서 비교해 왔다. 이 때문에 특정 압축률이 엔드투엔드 성능에 어떻게 반영되는지 일관된 결론을 내리기 어려웠다. 또한 태스크 품질과 시스템 지표(예: TTFT, 처리량)를 함께 보지 않아 트레이드오프가 충분히 정리되지 못했다.

- **Core Contribution**: 이 논문은 LongBench 스타일을 기반으로, 작업(워크로드) 유형에 따라 대표 KV cache 최적화 메커니즘을 동일 조건에서 비교하는 workload-aware benchmark를 제안한다. 대상은 quantization(KIVI, TurboQuant), pruning/eviction(SnapKV), merging(CaM)이며 Llama-3.1-8B-Instruct와 Mistral-7B-Instruct-v0.3에서 멀티문서 QA, 단일문서 QA, few-shot learning, 요약을 평가한다. 질(태스크 품질)과 효율(출력 처리량, time-to-first-token, 실현 압축률)을 동시에 측정해, 어떤 기법이 어떤 작업에 적합한지 배포 관점을 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘압축률’ 같은 단일 지표로는 end-to-end 병목과 품질 변화를 예측하기 어렵다는 점이다. 이를 해결하기 위해 컨텍스트 길이 버킷별로 TTFT, 처리량, 실현 압축률을 함께 보고, 멀티문서/요약/학습형 few-shot처럼 서로 다른 의존 패턴이 압축 방식에 미치는 영향을 분해한다. 그 결과 압축 메커니즘 선택이 워크로드 민감도 문제와 직결됨을 정량적으로 드러낸다.

- **Empirical Impact**: 실험 결과, 압축률만으로는 최종 성능을 예측하기 어렵고, 기법별 장단점이 작업 유형에 따라 갈린다. KIVI4는 모델 간 품질이 가장 안정적으로 유지되며, SnapKV는 long-context 처리량에서 가장 강한 모습을 보인다. CaM은 특정 QA 워크로드에서 큰 개선을 주지만, 품질과 실현 압축률에서 모두 워크로드 민감도가 커 배포 시 선택 가이드가 필요함을 시사한다.



### Proof of Execution: Runtime Verification for Governed AI Agent Actions (https://arxiv.org/abs/2607.05397)
Comments:
          14 pages, 1 figure, 4 tables, 1 algorithm; includes formal soundness and replay theorems with witness constructions in appendix

- **Prior Approaches**: 기존 에이전트 아키텍처는 실행은 하되, 각 단계의 권한(authorization)과 영속 효과(durable effect)가 계약(contract) 범위 안이었는지까지 함께 증명하진 못했다. OpenTelemetry 같은 관측/트레이싱은 디버깅에 초점이 있어 재생 가능한(replayable) 거버넌스 증거로 바로 연결되지 않는다. 또한 TEEs, verifiable computation, zkVMs, 공급망(attestation) 검증은 각각 코드/계산/상태/빌드 산출물 무결성에는 강하지만, “권한-경로-효과-재생 맥락”을 하나의 런타임 객체로 묶는 역할은 제한적이었다.

- **Core Contribution**: 이 논문은 Proof of Execution(PoE)을 제안하며, 에이전트 실행을 (C, T, R) 형태의 “증명-휴대형(proof-carrying) 객체”로 모델링한다. 여기서 C는 계약, T는 Execution Causal Event Stream(ECES)로 인과 관계와 서명을 포함한 실행 기록, R은 재생(replay)을 위한 결정적 환경 컨텍스트를 뜻한다. PoE는 well-formedness와 5개의 validator-checkable invariant를 통해 “각 단계가 승인됐는지, 기록이 변조에 내성이 있는지, 궤적을 결정적으로 재구성할 수 있는지”를 런타임에서 검증 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 신뢰 경계 밖에서 발생한 프롬프트/도구 입력 변화나 trace 위·변조, Gateway 우회, 거부(deny) 이후 부수효과 발생, 재생 회피가 “검증 가능한 불변조건(invariant)”을 깨는 방식으로 잘 포착돼야 한다는 점이다. 논문은 Prime Execution Model(PEM)로 계획(planning), enforcement, effect, recordkeeping을 분리해 privilege collapse를 줄이고, validator가 인과 DAG에 대해 invariant을 계산하도록 운영화했다. 또한 서명/해시/배포 가정(예: Effector-exclusive credentialing, 의존성 캡처의 완결성)을 명시적으로 두어, PoE-valid이면서 의미 보장을 위반하는 공격이 성공하면 signature forgery, hash collision 또는 배포 실패 이벤트로 귀결됨을(soundness) 보인다.

- **Empirical Impact**: 단일 노드 TypeScript 프로토타입에서 PoE 검증 오버헤드는 최소 flow에 약 2.7ms, 동시 배치 워크로드에는 약 4.4%로 보고되며, 8-이벤트 실행 트레이스는 약 1.1KB로 압축된다. Gateway-bypass 및 trace-mutation 공격은 불변조건 검증 단계에서 거부되며, Execute Attestation Certificate(EAC)는 PoE=1일 때만 발급된다. 결론적으로 PoE는 consensus/TEEs/zkVMs를 대체하진 않지만, 계약 기반 승인·효과·기록 무결성·재생성을 하나의 검증 가능한 런타임 증거 객체로 묶어 규제 환경의 “거버넌스 가능한 실행”을 실무적으로 attest하게 만든다는 점에서 의미가 크다.



### KVpop -- Key-Value Cache Compression with Predictive Online Pruning (https://arxiv.org/abs/2607.05061)
- **Prior Approaches**: KV 캐시를 줄이기 위한 기존 방식은 슬라이딩 윈도우나 sink-token처럼 고정 규칙을 쓰는 휴리스틱, 누적/현재 어텐션 등 대리 지표로 온라인 점수를 매기는 score-based, 그리고 DMC·DMS 같은 learned eviction으로 나뉜다. 하지만 많은 방법이 토큰의 ‘미래 유용성’을 직접 감독하지 못해, 시간이 지나며 관련성이 바뀔 때 가지치기가 흔들리거나(취약) 의사결정 시점이 맞지 않는 문제가 생긴다.

- **Core Contribution**: KVpop은 고정 예산(fixed-budget) KV eviction을 ‘미래 어텐션으로 keep-or-drop을 예측’하도록 감독 학습하는 sparse-attention retrofit이다. 특히 보호 윈도우를 지난 뒤의 미래-attention mass를 미래-유용성 타깃으로 삼아, 토큰이 캐시에서 빠질 경계(boundary)에서 장기 유지 여부를 직접 학습한다.

- **Technical Challenges**: 핵심 난제는 미래 어텐션 신호를 정확히 계산하면서도 dense attention map을 만들면 학습/비용이 급증한다는 점이다. KVpop은 transposed-attention 형태의 학습 전용 계산과 sparse 커널의 LSE(로그-서머 지수) 정규화 재사용으로 타깃을 효율적으로 만들고, stateful scorer는 mLSTM 기반으로 보호 윈도우 동안 증거를 누적한 뒤 delayed readout으로 경계 시점에 점수를 내리게 해서 의사결정을 근접한 near-future 맥락과 정렬한다.

- **Empirical Impact**: AIME·HMMT 수학 추론에서 Qwen3-4B는 75% 압축에서 dense-attention 대비 98%(풀의 성능 대비), 88% 압축에서도 97%를 유지했으며 Qwen3-8B는 더 높은 성능(near-full teacher)으로 나타났다. 또한 수학 데이터로만 distillation했음에도 GPQA-D·LiveCodeBench 같은 out-of-domain 코드/과학 추론에서도 높은 품질을 유지했고, 긴 생성(최대 131k 토큰)에서도 KV 메모리 사용이 dense 대비 크게 덜 증가해 장기 추론 비용을 낮추면서 품질을 지킨다는 점이 확인됐다.



### Lingering Authority: Revocable Resource-and-Effect Capabilities for Coding Agents (https://arxiv.org/abs/2606.22504)
Comments:
          20 pages

- **Prior Approaches**: 기존 코딩 에이전트 연구는 주로 promptinjection 등 콘텐츠 계열 공격, MCP 같은 툴 프로토콜 신뢰, 그리고 툴 선택 후 샌드박스/실행 검증으로 위협을 막는 데 집중해 왔습니다. 하지만 “지금 계획 단계에서 플래너가 어떤 임시 권한을 보게 되는가”와 “그 권한이 언제 사라져야 하는가”는 보안 상태로 충분히 다뤄지지 않았습니다. 그 결과, 특정 하위목표에 정당화됐던 리소스/효과 권한이 에피소드 종료 후에도 플래너 인터페이스에 남는 경우(lingering authority)가 생길 수 있습니다.

- **Core Contribution**: 이 논문은 lingering authority를 정의하고, 작업 계약(task contract) 기반으로 플래너에 노출되는 가역(revocable) 권한의 수명(lifetime)을 통제하는 참조 모니터 PORTICO를 제안합니다. PORTICO는 계약을 초기 권한 엔벨로프, grant 규칙, trusted closure predicate, global deny 규칙으로 컴파일한 뒤, request–grant–invoke 라이프사이클로 epoch-bound 핸들을 발급/철회합니다. 또한 closure 시점 이후에는 다음 플래너 턴 인터페이스에서 핸들을 제거하고, 만료된 핸드 재사용(stale replay)을 실행 단계에서 거부합니다.

- **Technical Challenges**: 핵심 난제는 플래너가 “요청할 수 있는 것”과 “실행해 실제 효과를 내는 권한”을 구분해, 권한이 필요한 순간에만 인터페이스가 확장되도록 만드는 것입니다. PORTICO는 typed tool catalog와 계약 컴파일 관계를 기반으로 J Portico 검증을 수행해, 현재 정당화되지 않은 능력이 인터페이스에 노출되지 않게 하고, grant를 통해서만 실행 가능 집합이 확장되도록 했습니다. epoch-bound, 불투명(opague) 핸드로 서버 측 상태를 참조하게 하여 closure 이후에는 재사용된 핸드가 부작용을 유발하지 못하도록 설계했습니다.

- **Empirical Impact**: 통제된 코딩 에이전트 벤치/시뮬레이션에서는 PORTICO가 계약 금지 효과(contract-forbidden effects)를 평가 실행에서 0건으로 유지하면서, closure 이후 10/10 재사용을 거부하는 성능을 보였습니다. 반면 non-revoking comparator는 closure 이후의 stale 재사용을 각각 10/10 허용했으며, 결정적 stale-write audit에서도 0/6 vs 6/6의 집행 금지 효과 차이가 관찰됐습니다. 또한 고정된 그랜트 동일 조건에서 task success·범위 준수는 맞추되, 포괄적 권한 노출이 계획 제안의 막힘을 67→84로 늘리는 경향과 더불어, 실제 고정 레포지토리(repos) 실험에서도 동일한 권한 수명 라이프사이클이 재현됐습니다.



### Agent Step Value: Probing the Observer Effect in Black-Box Traces (https://arxiv.org/abs/2607.04419)
Comments:
          aligned with workshop version

- **Prior Approaches**: 기존 평가는 에이전트의 최종 답변만 점수화해, 어떤 ‘전이(transition)’가 상태를 개선했는지/악화했는지 추적하기 어렵다. 툴 사용 에이전트나 워크플로우 기반 평가도 대개 과정 전체를 관찰하되, 평가자(between-state) 신념 변화를 단계 단위로 직접 분해하진 못했다.

- **Core Contribution**: 이 논문은 Agent Step Value(ASV)를 제안해, 관측된 한 전이 전/후 상태에서 평가자의 후보지분(belief) 변화량을 정량화한다. entropy 변화, Bayesian surprise(신념 이동), 그리고 offline gold-margin gain(검토된 타깃을 향한 이동)을 분리해 “왜 점수가 바뀌었는지”를 단계별로 드러낸다.

- **Technical Challenges**: 핵심 난제는 LLM 평가가 내부 deliberation(추론)과 scoring을 섞어버려 첫 토큰 로그확률이 ‘추론 프리앰블’에 오염될 수 있다는 점이다. ASV는 (1) 물리 라벨 없는 compact label-free rationale 버퍼 생성과 (2) 고정된 후보 라벨 매핑 후 1-token 옵션 로짓 로딩이라는 두 단계 프로토콜을 분리하고, 동일 전이를 프롬프트/라쇼나/스코어링 규칙을 바꿔 재플레이해 evaluator-channel 민감도까지 측정한다.

- **Empirical Impact**: 100개 오픈QA에서 총 1,100개 전이를 평가한 결과, entropy movement는 전 단계에서 0인데 Bayesian surprise 평균 2.693으로 ‘거의 one-hot’ 신념이 방향을 크게 뒤집는 현상이 드러났다. 특히 128-token rationale-conditioned 프로토콜에선 평균 gold-margin gain이 -2.335(95% CI [-3.395, -1.272])로 타깃을 약화시키는 전이가 우세했지만, 같은 traces를 direct one-token scoring으로 보면 +4.033으로 부호가 반전됐다. 컴포넌트 감사에서는 reversal의 주요 원인이 rationale 생성/압축(compression) 채널에 있으며, extraction·audit 이후에 손실이 집중되는 패턴을 로컬라이즈해 블랙박스 에이전트 디버깅에 바로 활용 가능함을 보여준다.



### Shape Over Intensity: Directional Topological Encoding for False Positive Reduction in Intracranial Aneurysm Detection (https://arxiv.org/abs/2607.05317)
Comments:
          36 pages, 12 figures, preprint

- **Prior Approaches**: 기존 intracranial aneurysm(IA) 탐지는 3D U-Net, ResNet, GLIA-Net 같은 CNN 기반 segmentation 또는 object detection에 많이 의존하지만, CTA의 낮은 대비와 해상도 때문에 작은 병변에서 saccular aneurysm과 vascular bifurcation을 헷갈리는 문제가 커진다. 특히 <3 mm 구간에서 민감도가 약 56~60%대로 떨어지고 false-positive 알람이 과도해 임상 적용성이 제한된다. 또한 topological data analysis(TDA)를 쓰더라도 PI/PL처럼 direction-agnostic 요약은 공간적 비대칭 정보를 충분히 보존하지 못한다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 high-sensitivity 후보 생성 CNN 뒤에 붙여 false-positive를 줄이는 plug-and-play 토폴로지 인지(false-positive reduction) 프레임워크를 제안한다. 핵심은 Smooth Euler Characteristic Transform(SECT)이 intensity 패턴이 아니라 전역 3D 혈관 기하의 비대칭(topological geometry invariant)을 direction 기반으로 인코딩해, aneurysm과 bifurcation의 구조적 혼동을 해소한다는 점이다. Persistent Images(PI)와 Persistence Landscapes(PL)를 함께 비교해 representation의 역할을 분리한다.

- **Technical Challenges**: topological representation을 의료 영상 분류기에 넣기 위해서는 persistence 결과를 ML 호환 벡터로 변환해야 하고, 동시에 small lesion에서 미세한 시그널을 잡되 잡음에 과민하면 안 된다. 논문은 PD→PI/PL로의 고전적 변환과 더불어, SECT의 Euler characteristic 곡선을 여러 방향에서 계산해 매끄럽게(smoothing) 벡터화하고, persistence threshold로 짧은 잡음 성분을 억제하는 방식으로 이를 해결한다. 또한 Frangi-mined hard negatives 등 해부학적으로 그럴듯한 bifurcation 모사체를 포함해 지오메트리 판별이 실제 실패 모드를 직접 겨냥하도록 데이터와 샘플링(TAXS)을 설계했다.

- **Empirical Impact**: RSNA 2025 데이터의 stratified 평가에서 SECT는 AUC 0.943으로 PI/PL 계열의 direction-agnostic 방식(AUC ≈0.68)을 크게 앞선다. 특히 임상적으로 중요한 sub-3 mm 코호트에서 AUC 0.943을 유지하며, 95% specificity에서 sensitivity 78.5%를 보이는 등 ‘작은 병변 성능 붕괴’를 되집는 임상 성능 inversion이 관찰됐다. 더 나아가 leave-one-scanner-out(LOGO)에서 스캐너 비특이성을 입증해 평균 AUC 0.927(4개 제조사) 수준을 보고하며, hybrid deep-learning 파이프라인의 견고한 downstream 필터로서 의미가 크다고 정리한다.



New uploads on arXiv(cs.RO)

### Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation (https://arxiv.org/abs/2607.06564)
Comments:
          14 pages, 7 figures. Project website: this https URL

- **Prior Approaches**: 기존 VLA는 2D 기반 비전-언어 사전학습을 정책에 결합해 성능과 일반화를 끌어올렸지만, 실제 조작에는 3D 기하 및 공간 추론이 핵심이다. 3D를 넣는 방식은 (1) point cloud/voxel/multi-view를 직접 인코딩하되 대규모 3D 데이터·foundation encoder 부족에 막히거나, (2) 2D feature를 3D로 lifting하거나 3D를 multi-view로 투영하는 과정에서 기하 정보가 손실돼 지리 정합성과 확장성이 떨어진다. 또한 일부 방법은 예측 기반으로 미래 상태를 다루지만, 진화하는 3D 기하와 시계열적으로 일관된 action chunk를 함께 학습하도록 설계된 접근은 제한적이다.

- **Core Contribution**: Lift3D-VLA는 VLA에 명시적 3D point cloud reasoning을 통합하고, 시간적으로 일관된 action generation까지 한 프레임워크에서 해결하려는 시도다. 핵심은 (i) 3D 토큰을 pretrained 2D positional embeddings와 기하적으로 정렬해 3D encoding 시 정보 손실을 줄이는 lifting 전략과, (ii) 현재 point cloud를 복원하면서 미래 기하 변화를 예측하는 Geometry-Centric Masked Autoencoding(GC-MAE)로 3D 구조와 물리 다이내믹스를 동시에 학습하는 구조다. 여기에 LLM 레이어를 활용한 layer-wise temporal action modeling을 얹어 동적 환경에서의 시간 일관성을 강화한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) pretrained 2D positional embedding과 3D 좌표를 정합시키되 왜곡을 줄이는 것, (2) 3D의 정적 구조뿐 아니라 시간에 따른 진화를 self-supervised 방식으로 학습하는 것, (3) action chunk가 장기적으로 서로 모순되지 않게 시계열 의존성을 만드는 것이다. 저자들은 가상 평면을 여러 장 두고 3D 점을 해당 평면 좌표로 투영한 뒤 camera extrinsic으로 front plane을 고정해 2D positional embedding을 재사용하는 방식으로 첫 난제를 완화한다. 둘째로는 GC-MAE에서 masked point reconstruction(정적 CD loss)과 future geometric prediction(다음 프레임의 기하 예측)을 이중 목표로 구성해 “복원-예측”을 강제한다. 마지막으로는 action decoding을 LLM 마지막 층에만 의존하지 않고 중간~깊은 층 표현을 연속적으로 써서 각 레이어가 action step을 담당하는 계층적 layer-wise 예측으로 temporal coherence를 확보한다.

- **Empirical Impact**: Lift3D-VLA는 시뮬레이션 22개 작업과 실제 로봇 8개 작업에서 MetaWorld와 RLBench의 평균 success rate를 각각 기존 SOTA 대비 10.8%, 11.1% 높였고, 실제 환경의 최강 baseline보다 4%p 앞섰다. 또한 unseen object/background/lighting으로의 일반화가 관찰되며, out-of-distribution perturbations에 대해서도 더 강한 견고성을 보였다. 장기 horizon 조작(예: 조건이 계속 바뀌는 상황에서 팬 위 달걀을 반복적으로 스쿱)에서도 시간 일관된 행동 생성이 효과적으로 동작하는 점이 이 연구의 실용적 의미로 평가된다.



### Embodied Human-Robot Interaction via Acoustics: A MARL Approach with AcoustoBots for Spatial Data Physicalization (https://arxiv.org/abs/2607.06563)
Comments:
          This paper has been accepted for publication in the Proceedings of the 2026 International Conference on Robotic System and Artificial Intelligence (RSAI 2026), 10-12 July, 2026, Tokyo, Japan

- **Prior Approaches**: 기존 데이터 물리화는 화면이나 고정형 장치에 의존해 실제 환경과 분리된 채로 제공되는 경우가 많아, 공간적 역동성을 몸으로 이해하기 어렵다는 한계가 지적된다. 위치-동기화된 물리 표현은 더 자연스러운 인지를 돕지만, 모바일이면서 동시에 데이터 반응성까지 갖춘 연구는 상대적으로 부족했다. 또한 음향 부유( acoustophoretic levitation )는 가능하더라도 이동 중 안정적인 높이 렌더링과 다로봇 간섭·충돌 회피를 함께 만족시키는 통합 접근이 어려웠다.

- **Core Contribution**: 본 논문은 TurtleBot3에 상향 8×8 초음파 phased array를 탑재해, 로봇이 도시 지점으로 이동하며 지역 스칼라 값(인구밀도·소음·교통 등)을 입자 높이(1–10cm)로 실시간 물리화하는 AcoustoBots를 제안한다. 내비게이션은 MADDPG 기반 MARL로 학습하되, 렌더링(높이) 또한 이동 중 연속적으로 유지되도록 GS-PAT 기반 acoustic controller를 결합해 perception–display–action loop를 닫는다. 두 로봇이 협력해 커버리지를 수행하는 동시에 물리화 신호가 위치 의존적으로 유지되는 점을 핵심 기여로 제시한다.

- **Technical Challenges**: 기술적으로는 로봇 가속·회전·진동이 음향 트랩 안정성을 흔들어 부유가 쉽게 깨질 수 있고, 다로봇 환경에서는 간섭과 충돌, 그리고 물리화 정합성(원하는 높이가 올바른 위치에서 나오기)의 동시 최적화가 필요하다. 또한 이동 제어(내비게이션)와 음향 제어(고속 위상 업데이트)가 서로 다른 갱신 속도와 불확실성을 갖기 때문에 두 제어층을 결합해 일관된 닫힌 루프를 구성해야 한다. 논문은 GS-PAT로 표적 높이에 맞는 위상을 고속 갱신해 트랩 안정성을 유지하고, MADDPG의 CTDE 구조와 물리화 충실도 및 안전을 반영한 보상 설계로 협력적·충돌-의식 내비게이션을 학습한다.

- **Empirical Impact**: 4m×3m 축소 UK 맵에서 PhaseSpace 기반 정밀 로컬라이제이션으로 단일/이중 로봇을 10회씩 반복 실험한 결과, 이동 중 부유가 안정적으로 유지되며 위치에 따른 높이 렌더링이 일관되게 나타났다. 성공률은 단일 로봇 90%, 두 로봇 80%였고 충돌 횟수는 낮게 보고되며, 학습 곡선도 seed 간 분산이 작아 재현성이 확인된다. 이 결과는 음향 부유를 ‘glanceable’하고 공존하는 물리적 커뮤니케이션 큐로 활용해, 공간 분석을 사람의 현장 이해 방식에 더 가깝게 연결하는 접근의 가능성을 보여준다.



### RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation (https://arxiv.org/abs/2607.06559)
Comments:
          Project Page: this https URL, Github: this https URL

- **Prior Approaches**: 기존 비디오 기반 world model은 픽셀 2D 투영 공간에서 미래를 예측하는 경우가 많아, 로봇이 요구하는 6-DoF 자세·깊이 기반 상호작용과의 표현 격차가 커질 수 있습니다. NeRF/3DGS 계열 4D 접근은 기하 추론이 강하지만 장면 특화·고비용·규모 확장이 어렵고, 동적 SfM은 미래 예측의 생성성이 약하다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 RGB-D에서 동시에 RGB, depth, optical flow를 생성해 RGB-DF(물리 기반 4D 표상)를 만들고, 이를 통해 외관·기하·운동이 일관되게 진화하도록 합니다. 그 위에 RynnWorld-4D를 제안해 단일 RGB-D 관측과 언어 지시로 미래 RGB-DF 시퀀스를 one unified diffusion process에서 공동 생성하며, RynnWorld-4D-Policy로는 내부 4D 표현을 한 번의 forward pass에서 바로 inverse dynamics로 연결합니다.

- **Technical Challenges**: 핵심 난제는 대규모 4D 생성 학습에 필요한 촘촘한 depth/optical flow 정답 데이터가 부족하다는 점이며, 이를 위해 Rynn4DDataset 1.0(254.4M+ 프레임)을 구축하고 pseudo-annotation을 생성해 스케일을 확보합니다. 또한 tri-branch diffusion 구조와 Joint Cross-Modal Attention, frame-wise 3D RoPE로 모달리티 간 일관성을 강제하고, staged training(모달 독립 적응→공동 주의 학습→전 파라미터 SFT) 및 branch dropout으로 학습 안정성과 기하·운동 정합성을 높입니다.

- **Empirical Impact**: 실험에서는 RynnWorld-4D가 시간적·공간적으로 coherent한 4D 예측을 만들고, RynnWorld-4D-Policy는 실제 덱스터러스 양손 조작 벤치마크에서 SOTA 성능을 보이며 특히 공간 정밀도와 시간적 동기화가 필요한 과제에서 강점을 보입니다. 더불어 action 생성 시 내부 4D 특징을 직접 소비하고 multi-step denoising 병목을 우회해 closed-loop 제어에 필요한 반응성을 확보했다는 점이 실용적 의미를 가집니다.



### RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation (https://arxiv.org/abs/2607.06558)
Comments:
          Project Page: this https URL, Github: this https URL

- **Prior Approaches**: 로보틱 학습을 키우려면 대규모 궤적 데이터가 필요하지만, 기존 물리 텔레오퍼레이션은 작업공간·하드웨어에 시연이 묶여 운영 병목이 생겼다. 인간-로봇 비디오 번역 계열은 시각적 간극은 줄이지만 실제 행동(미래 상태)을 동작 신호로 생성하지 못해 인터랙티브 시뮬레이션의 닫힌고리(closed-loop) 구성이 어렵다. 또한 action-conditioned egocentric world model은 인간 중심(human-centric)이라 embodiment 갭을 제대로 메우지 못했다.

- **Core Contribution**: 이 논문은 digital teleoperation을 제안해, 실제 로봇 대신 generative world model로 생성 데이터를 수집하는 패러다임을 제시한다. 운영자의 손-자세(hand-pose) 스트림이 로봇 관점의 고품질 egocentric 영상을 만들고, 이 과정의 관절 단위 포즈가 embodiment-agnostic 행동 라벨로 retargeting되어 어떤 로봇에도 학습용 state-action 궤적을 공급한다. 그 결과 물리 하드웨어 이동 없이도 imitation learning을 위한 완전한 궤적 데이터를 만들 수 있게 한다.

- **Technical Challenges**: 핵심 난제는 (1) 로봇 중심성, (2) 프레임마다 복원 가능한 관절 행동 라벨의 action-grounding, (3) 제어 루프에 맞는 실시간성(지연 최소화)을 동시에 만족하는 생성기를 만드는 것이다. RynnWorld-Teleop은 depth-aware skeletal conditioning으로 3D 단서를 보강하고, 인간 비디오 사전학습→인간-로봇 페어 파인튜닝의 progressive cross-domain training으로 embodiment 갭을 줄이며, bidirectional teacher를 streaming autoregressive distillation으로 1-pass급 단일 추론에 가깝게 압축해 40+ FPS를 달성한다. 또한 chunked re-anchoring으로 장기 생성 시 drift를 완화해 물리적 일관성을 유지한다.

- **Empirical Impact**: RynnWorld-Teleop이 생성한 데이터만으로 학습한 정책은 다양한 이족/양팔 협응(dexterous bimanual) 과제에서 zero-shot Sim2Real 전이를 보이며, 실제 데이터에 디지털 텔레오퍼레이션 데이터를 증강하면 success rate이 일관되게 상승했다. 이는 디지털 텔레오퍼레이션이 단순 아이디어가 아니라 대규모·고충실도 로봇 데이터 엔진으로 작동함을 실증한 첫 사례에 가깝다. 장기적으로는 물리 시연의 병목을 “운영자 상상”으로 대체해 로보틱 에이전트 학습의 스케일링 상한을 끌어올릴 가능성을 제시한다.



### UniLM-Nav: A Unified Framework for Zero-Shot Last-Mile Navigation (https://arxiv.org/abs/2607.06537)
- **Prior Approaches**: 모바일 매니퓰레이션에서 기존 내비게이션은 보통 목표물 근처(예: 1–2m)까지만 이동해 “마지막 1마일”의 조작 준비 자세를 보장하기 어렵다. 기존 last-mile navigation은 수작업 포즈 라벨이나 태스크 특화 학습에 의존하는 경우가 많아 오픈 보캐뷸러리·세밀한 공간 제약을 만족하기에 확장성이 떨어진다.

- **Core Contribution**: 본 논문은 UniLM-Nav라는 zero-shot 오픈 보캐뷸러리 last-mile navigation 프레임워크를 제안한다. UniLM-Nav는 마지막 1마일을 view selection, task-conditioned affordance grounding, geometry-aware base-pose reasoning의 3단계로 분해하고, 멀티모달 대형 언어 모델(MLLM)을 공통 백엔드로 사용해 처리한다.

- **Technical Challenges**: 핵심 난제는 “목표물/수납영역의 위치” 수준을 넘어, 조작에 필요한 세밀한 공간 관계를 고려해 조작 가능한 base pose를 추론해야 한다는 점이다. UniLM-Nav는 최근 관측들을 메모리에 저장해 MLLM이 더 유리한 레퍼런스 뷰를 고르게 하고, 2D affordance point를 깊이·카메라 정보로 로봇 중심 좌표로 3D로 lift한 뒤, 로봇 기하 제약을 조건으로 base pose를 추론하도록 설계해 이 문제를 완화한다.

- **Empirical Impact**: OVMM 벤치마크에서 UniLM-Nav는 Gemini-3-Flash-Preview 백엔드를 사용해 이전 SOTA MoTo 대비 전체 성공률을 3.13%p 향상(23.77%)시켰다. 또한 Unitree B2(6-DoF Unitree Z1 매니퓰레이터) 실로봇 배치 실험에서 여러 실환경 태스크를 수행하며 적용 가능성을 보였고, view selection/affordance grounding/base-pose reasoning 및 MLLM 선택이 성능에 크게 좌우됨을 분석했다.



### Neural-ESO: A Dual-Pathway Architecture for Provably Robust Learning-Based Contro (https://arxiv.org/abs/2607.06535)
Comments:
          Accepted to IEEE RA-L

- **Prior Approaches**: 기존 학습 기반 제어는 비행 중 잡음/교란을 residual로 직접 학습해 보상하지만, 배치 후에는 학습된 모델에 과도 의존하게 되어 OOD(분포 밖)에서 성능 저하나 불안정 위험이 커질 수 있다. 또 잔차 모델이 수렴하지 않은 학습 초기(트랜지언트) 구간에서 안전 보장이 약해 안전-critical 시스템 적용에 제약이 있었다. 관측자(ESO) 기반 접근은 교란 추정에 강점이 있으나, 학습 구성요소를 섞는 경우에도 Lipschitz 같은 조건과 결합한 형식적 안정성 분석 및 OOD 신뢰도 저하 대응이 충분치 않았다.

- **Core Contribution**: 이 논문은 Neural Extended State Observer(Neural-ESO)로 교란 추정에 ‘학습-기반 예측 + 관측자 기반 보정’의 이중 경로를 도입한다. 예측 경로에서는 신경망이 feedforward 교란 추정을 제공해 수렴을 빠르게 하고, 보정 경로에서는 기존 ESO가 잔차 예측 오차를 계속 상쇄해 신경망에 대한 over-reliance를 막는다. 또한 학습 구성요소에 Lipschitz bound를 강제하고, 총 교란을 신경망 예측과 ESO 보상으로 재구성해 OOD 적응(Total Disturbance Retraining, TDR)의 학습 신호로 활용하는 안전한 절차를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 학습된 구성요소가 폐루프에 들어올 때 안정성/수렴을 형식적으로 보장하는 방법, (2) 학습 오차가 커지는 OOD 상황에서 신뢰도 높은 적응 신호를 만드는 방법이었다. 논문은 Lyapunov 이론과 small-gain 분석으로, 신경망에 Lipschitz 연속성을 강제하면 전체 폐루프 오차 동역학이 Uniformly Ultimately Bounded(UUB)임을 보인다. 더 나아가 SN(spectral normalization)으로 네트워크의 Lipschitz 상한을 최적화 과정에서 하드 제약해 안정성 증명에 필요한 잔차 도함수 바운딩 조건을 확보한다.

- **Empirical Impact**: 실험은 강한 ground-effect 교란이 존재하는 쿼드로터 착륙 태스크에서 수행되며, 일반 분포와 OOD 시나리오 모두에서 정확도-강인성 트레이드오프와 더 높은 운용 신뢰성을 보여준다고 보고한다. 학습 초기와 배치 이후, 그리고 OOD 전이 구간에서 ESO-only 및 최신 기준선 대비 더 일관된 성능을 달성했다는 점이 강조된다. 특히 TDR을 통해 새로운 도메인의 총 교란을 안전하게 수집·재학습함으로써, 분포 변화에도 통제 가능 범위 내에서 성능을 유지/개선하는 방향성을 제시한다.



### Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning (https://arxiv.org/abs/2607.06501)
Comments:
          Accepted to Robotics: Science and Systems (RSS) 2026

- **Prior Approaches**: 기존 서브로봇 계획은 닫힌 세계 가정과 사전 정의된 상태/전이 모델(PDDL 등)에 크게 의존해, 환경이 바뀌거나 객체·행동 효과가 누락되면 계획 일반화가 급격히 떨어진다. LLM 기반 플래너는 그럴듯한 계획을 빠르게 제안하지만, 부분 관측 상황에서 환각에 가까운 지식과 실제 근거를 구분하지 못해 조용한 실패(silent failure) 위험이 남는다. 또 일부 기초모델 기반 도메인/표상 생성 연구가 있어도, 확장 지식의 불확실성을 적극적으로 검증하지 않고 실행 실패 후 간접적으로만 갱신되는 경우가 많다.

- **Core Contribution**: 이 논문은 HUME(Hypothesis-driven Uncertainty-aware Model Expansion)이라는 오픈월드 플래닝 프레임워크로, 추상 세계 모델을 “가설” 형태로 자동 생성·검증·업데이트하도록 설계한다. 로봇은 객체 존재/속성/미기록 행동 효과처럼 불확실한 항목을 불확실 잠재변수로 유지하고, 목표 달성과 동시에 가설 검증까지 고려하는 계획을 수립한다. 이를 통해 기초모델의 무구조(commonsense prior) 지식을 구조화된 상징적 추론(계획)과 결합한다.

- **Technical Challenges**: 핵심 난제는 기초모델이 만든 확장 지식이 항상 참이 아닐 때, 고전 플래너가 다루기 쉬운 형태로 “불확실한 모델 확장”을 계획에 통합하는 것이다. 논문은 검증 action을 명시적으로 도입하고, 검증의 비결정성을 all-outcomes determinization으로 다루며, 실패한 가설은 planning graph에서 배제(브랜치 컷)해 불필요한 확장 경로를 줄인다. 실행 단계에서는 vision-language model(VLM) 또는 전용 지각 모듈로 가설을 확인/반박하고, 결과를 기반으로 가설 생성과 재계획을 반복한다.

- **Empirical Impact**: Block Processing World, 모바일 매니퓰레이션(미지 환경), 가전 동작 같은 다양한 오픈월드 과제에서 시뮬레이션과 실세계 실험을 통해 자율 지식 확장과 효과적인 작업 수행이 확인된다. 비교 실험에서는 확장 지식을 “사실로 간주”하거나, 불확실성을 인식하지 못하는 방식보다 HUME이 탐색 전략과 성공률에서 유의미하게 유리함을 보인다. 결과적으로 서비스 로봇을 실제 가정 환경처럼 모델이 지속적으로 불완전해지는 공간에 배치할 때, 불확실성 인지형 모델 확장이 실용성을 높인다는 점을 실증한다.



### Clustering-Embedded Model Predictive Path Integral Control: Avoiding Averaging-Induced Failure and Enabling Efficient Cluster Selection for Dynamic Obstacles (https://arxiv.org/abs/2607.06499)
- **Prior Approaches**: MPPI 같은 샘플링 기반 모션 플래닝은 GPU 병렬 롤아웃으로 실시간성을 확보하지만, 중요도 가중 평균이 다봉성(다중 회피 모드)을 섞어버리면 전방 장애물 앞에서 주저하거나 충돌로 이어질 수 있다. 이를 완화하려고 CSC-MPPI는 제약을 보정한 뒤 DBSCAN으로 클러스터링하지만, 속도 기반 통계는 좌/우 회피 모드의 분리가 약해 hesitation을 유발할 수 있고 동적 환경에서는 비용만으로는 moving obstacle과의 지속 결합을 막기 어렵다.

- **Core Contribution**: 이 논문은 Clustering-Embedded MPPI (CE-MPPI)를 제안하며, 표준 MPPI의 ‘평균화 실패’를 구조적으로 해소하는 것이 핵심이다. 충돌 롤아웃은 pruning으로 제거하고, 나머지는 DBSCAN으로 클러스터링하되 충돌에서 유도한 기준점과 롤아웃 말단 상태의 기하학적 방향 특징으로 서로 다른 회피 모드를 더 잘 분리한다. 이후 단일 클러스터만 선택해 within-cluster MPPI 업데이트를 수행함으로써 모드-일관성 있는 제어 갱신을 달성한다.

- **Technical Challenges**: 난제는 (1) 충돌이 섞인 롤아웃들 사이에서 유효 모드만 안정적으로 분리하고 (2) 정적/동적 장애물에서 클러스터 선택 기준을 ‘즉시 비용’에만 의존하지 않게 설계하는 데 있다. CE-MPPI는 충돌 롤아웃 말단의 평균으로 충돌-derived reference point를 만들고, 그로부터의 방향을 DBSCAN feature로 사용해 모드 separability를 높인다. 또한 정적 장면에서는 최소 평균 비용 클러스터를 고르지만, 동적 장면에서는 장애물 이동 방향(최근 관측으로 추정)과 가장 반대인 클러스터를 선택하는 dot-product 기준을 적용해 장애물 flux를 거스르는 우회로를 선제적으로 유도한다.

- **Empirical Impact**: 2-D JAX 가속 시뮬레이션에서 CE-MPPI는 averaging-induced failure를 체계적으로 줄였고, 특히 moving obstacle과의 persistent coupling을 줄이며 동적 시나리오에서 CSC-MPPI 대비 time-to-goal 17.4% 및 path length 9.1% 개선을 보였다. 또한 6-DoF UR5e의 실세계 실험(Isaac Gym 기반 CUDA 병렬 롤아웃)에서는 CE-MPPI가 표준 MPPI 대비 time-to-goal을 48% 줄이고 end-effector 경로 길이를 12% 단축했다. 결과적으로 고난이도(고자유도) 장애물 회피에서도 실시간성을 유지하면서 더 빠른 탈출과 경로 효율을 제공하는 접근임이 입증됐다.



### Hilti-Trimble-Oxford Dataset: 360 Visual-Inertial Benchmark with Floor Plan Priors for SLAM and Localization (https://arxiv.org/abs/2607.06464)
- **Prior Approaches**: 기존 건설 현장 모니터링은 LiDAR나 지상 레이저 스캐너로 3D 맵을 만들고, 이후 연속 세션 간 change detection 또는 BIM/도면과의 비교에 활용해 왔다. 하지만 LiDAR는 비용 부담이 크고, BIM은 항상 제공되거나 가볍게 처리하기 어렵다. 한편 floor plan 기반 localization은 일부 연구가 있었지만, 건설 환경에서 실제로 반복되는 구조·조명 변화·작업자 동작 같은 현실적 난제를 충분히 반영한 공개 벤치마크는 부족했다.

- **Core Contribution**: 이 논문은 360 카메라(소비자급) + IMU로 수집한 건설 현장 데이터셋과 벤치마크를 공개하며, SLAM(임의 기준축)과 floor plan 기준 localization(2D 매칭)을 동시에 평가하도록 구성했다. 데이터는 스위스 부흐스 현장에서 8개월(총 7개 층) 동안 3030개의 visual-inertial 시퀀스를 제공하고, LiDAR-inertial SLAM을 카메라에 견고하게 고정해 궤적 ground truth를 생성한다. 또한 Hilti x Trimble Challenge 2026을 통해 전 세계 제출 결과를 공개하고, floor plan 기반 정렬의 어려움을 정량적으로 드러낸다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 저비용 360×IMU에서 조명 변화·가림·반복 구조·빠른 움직임을 견디며 궤적을 추정하는 것, (2) 2D floor plan의 구조적 기준에 시각 관측을 안정적으로 연결하는 것이다. 저자들은 두 개 fisheye의 정밀 캘리브레이션(EUCM 모델 포함), IMU Allan variance 기반 잡음 모델링, rolling shutter readout 추정, 그리고 LiDAR-카메라 시간/공간 정렬을 통해 ground truth 품질을 확보했다. 또한 벤치마크 평가 시 pose coverage를 강제하고 초기 구간을 제외하는 등, 공정한 비교를 위한 프로토콜을 설계했다.

- **Empirical Impact**: Challenge 결과에서 SLAM 트랙은 62개 팀이 참여해 상대적 성숙도를 보여준 반면, localization 트랙은 22개 팀에 그쳤고 오차가 더 크게 나타나 건설 현장 2D 매칭의 난이도를 강조했다. 상위 localization 해법들은 영상에서 벽을 중심으로 한 구조/시맨틱 요소를 추출해 BEV로 floor plan과 정렬하는 경향이 강했으며, 특히 semantic segmentation이 성능 향상에 중요하다는 신호가 관측됐다. 이 데이터셋과 벤치마크는 실제 현장에서 저비용 센서로 장기 변화 모니터링을 수행하려는 연구에 표준 실험 기반을 제공한다.



### SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models (https://arxiv.org/abs/2607.06442)
Comments:
          The code is available at \href{this https URL}{SIEVE}

- **Prior Approaches**: VLA 모델은 대규모 로봇 데모로 imitation learning(IL)을 하는 경우가 많지만, 데이터가 커져도 중복·잡음·커버리지 불균형 때문에 정책 성능이 자동으로 좋아지진 않습니다. 기존 데이터 선택은 주로 trajectory 단위(대표성/신뢰도/유사도/피드백 등)나 state-action 단위(상호정보/진행도/국소 유사도 등)로 유틸리티를 보는데, 긴호라이즌 행동의 “구성 구조”를 놓치거나 너무 국소적이라 효과가 제한됩니다. 또한 학습량을 줄이면서도 IL 관점의 안정적인 행동 라벨(BC용 관측-행동 매핑)을 함께 확보하는 절차가 약한 편입니다.

- **Core Contribution**: SIEVE는 데모를 “재사용 가능한 visuo-motor primitive의 조합”과 “primitive를 잇는 transition interface”의 관점에서 구조적으로 선택하는 방법입니다. 먼저 프리미티브를 찾아 각 trajectory를 primitive 시퀀스(구성 패턴)로 표현한 뒤, 구성 패턴 공간에서 재사용 구조가 잘 드러나도록 선택 예산을 배분합니다. 마지막으로 각 구성 패턴 버킷 안에서 중심(medoid)에 가까운 trajectories를 골라 behavior cloning(BC)에 더 학습친화적인(안정적·예측가능한) 샘플을 확보합니다.

- **Technical Challenges**: 핵심 난점은 긴호라이즌 행동의 유용한 구조를 trajectory 전체 점수나 state-action 국소 점수로는 충분히 포착하기 어렵다는 점입니다. SIEVE는 이를 위해 (1) gripper/dexterous-hand의 grasp/release flip 경계로 trajectory를 세그먼트화하고, (2) 세그먼트 표현을 video encoder(V-JEPA2)로 만든 뒤 PCA 및 MiniBatch K-Means로 primitive vocabulary를 자동 결정(재사용성과 구분성을 함께 최적화)하며, (3) primitive의 노출뿐 아니라 인접한 transition 노출까지 포함해 diminishing returns 하의 구조 노출 목적함수를 greedy하게 예산 배분합니다. 이후 BC 안정성을 위해 버킷 내부에서 medoid 기준 거리로 대표 샘플을 선별해, 잡음/이상치가 감독을 흔들 가능성을 줄입니다.

- **Empirical Impact**: 실험에서는 Bridge-V2, Fractal, GR00T-X-Sim 등 여러 데이터셋과 SimplerEnv 및 RoboCasa-GR1 같은 OOD 평가에서 SIEVE가 기존 데이터 선택 baselines를 일관되게 능가했습니다. 특히 Bridge-V2에서 데모 50%만 쓰고 학습 steps도 50%로 줄였을 때 SIEVE가 full-data 학습보다 높은 평균 success rate를 보였고(또한 과도한 훈련 최적화가 아니라 구조 기반 재사용이 이득의 핵심임을 시사), task별로도 균형 있게 향상되었습니다. 더 나아가 Qwen3-VL-4B-GR00T와 Qwen3-VL-4B-OFT 두 VLA 모델 모두에서 효과가 유지되었고, ablation 결과 transition/primitive 구조 노출과 버킷 내 medoid 선별이 성능의 핵심임이 확인되었습니다.



### WristMimic: Full-Body Humanoid Control with Wrist-Guided Manipulation (https://arxiv.org/abs/2607.06438)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 인간-물체 상호작용 데모를 물리 시뮬레이션으로 전이할 때는 모션 캡처 기반 관절 궤적을 그대로 재생하는 방식이 흔했지만, 손처럼 접촉이 지배적인 구간에서는 위치 궤적만으로 접촉력/물체 동역학을 담기 어렵다. 그 결과 손가락 자세를 촘촘히 지도하거나(풀 핑거 슈퍼비전) 접촉용 보상/태스크별 설계를 추가해야 해 확장성과 데이터 의존성이 커졌다. 또한 손 수준 접근은 있어도 전신 제어와 결합해 단일 정책으로 retargeting을 안정화하는 데 한계가 있었다.

- **Core Contribution**: 이 논문은 WristMimic으로 전신 제어를 두 레짐으로 분리한다: 접촉이 없는 신체 파트는 kinematic pose target으로 유도하고, 접촉이 풍부한 손가락은 물체 추적과 contact outcome을 통해 학습한다. 핵심 통찰은 wrist가 두 레짐을 잇는 ‘게이트’라는 점으로, wrist는 접촉에 비교적 덜 영향받으면서도 손가락의 전역 자세와 그립 접근성(affordance)을 결정한다. 따라서 손가락 자세에 대한 직접 슈퍼비전을 줄이고도 유사하거나 더 나은 조작 성능을 노린다.

- **Technical Challenges**: 직접 손가락 자세를 지도하지 않으면 접촉 순간의 물리적 제어가 불안정해져, 단순히 오브젝트/컨택 보상만으로는 wrist 배치가 어긋날 위험이 크다. 이를 해결하기 위해 저자들은 wrist를 정밀하게 ‘배치’하는 장치를 설계하는데, 접촉 전후로 time-varying reward weight를 조절해 wrist 정렬을 우선하고 근위 팔(arm) 제약은 완화한다. 더불어 접촉 윈도우를 단계(approach/grasping/stabilization)로 나누고 phase-specific reset thresholds(더 빡빡한 wrist 기준 포함)로 탐색 공간을 현실적으로 제한해 모캡 오차에도 재현성을 확보한다.

- **Empirical Impact**: ParaHome과 OMOMO 두 데이터셋에서 10,000회 롤아웃 규모로 평가했으며, 성공률과 물체 위치/회전 오차 및 기준 접촉 유지율을 기준으로 성능을 검증한다. 요지는 손가락의 직접 kinematic supervision 없이도 wrist 제어를 통해 손가락-물체 상호작용을 충분히 유도해, 기존 풀 핑거 슈퍼비전 기반 방법과 비슷하거나 상회하는 결과를 보인다는 점이다. 또한 서로 다른 hand embodiment 사이에서도 finger-agnostic retargeting이 가능함을 실험적으로 뒷받침한다.



### From Foundation to Application: Improving VLA Models in Practic (https://arxiv.org/abs/2607.06403)
Comments:
          Website: this https URL, Github: this https URL, Checkpoints: this https URL

- **Prior Approaches**: 기존 VLA foundation 모델은 대체로 듀얼암 같은 제한된 행동공간과 실험실 환경에 맞춰 학습돼, 실제 로봇 배치에서의 embodiment 다양성·더 큰 DoF·장기 예측 요구를 충분히 따라가지 못했다. 이를 보완하려고 로봇 데이터 규모를 키우거나(크로스-태스크/크로스-embodiment) embodiment-aware 구조나 action space 통일을 시도하는 연구가 이어졌지만, 데이터·행동 커버리지·시간적 추론을 함께 정렬하는 접근은 여전히 부족했다.

- **Core Contribution**: LingBot-VLA 2.0은 실험실-현장 격차를 줄이기 위해 일반화(embodiment/태스크), 확장 행동공간, 시간적 추론(미래 예측)을 하나의 시스템 개선축으로 동시 강화한다. 이를 위해 약 60,000시간 규모의 사전학습 데이터(로봇 trajectory 50,000시간, egocentric human video 10,000시간)를 구축하고, 듀얼암을 넘어 head/waist/mobile base/ dexterous hands까지 제어 가능한 확장 action space를 제공한다.

- **Technical Challenges**: 핵심 난제는 서로 다른 로봇 구성에서 행동공간·동역학·데이터 분포가 불일치하는 상황에서 공통 표현과 제어 논리를 안정적으로 학습하는 것이다. 논문은 (1) jerk/파생량 기반으로 잡음 궤적을 거르는 재구성 파이프라인과 (2) 토큰 단위 loss-free Mixture-of-Experts(MoE)로 희소 전문가를 배치해 embodiment별 전문성과 보편 priors를 동시에 학습하며, (3) 미래 예측을 프록시 태스크로 두고 depth(기하)와 DINO-Video(인과적 시간 표현)를 dual-query distillation으로 결합해 temporal reasoning을 강화한다.

- **Empirical Impact**: GM-100 벤치마크의 generalist setting과 장기 모바일 매니퓰레이션 평가에서, LingBot-VLA 2.0은 작업 전반에서 일관된 성능 향상을 보이며 특히 객체 그라운딩과 목표 지향 실행, 그리고 미래 정보 기반 계획에서 이득이 크게 나타났다. 또한 전신 DoF가 포함된 확장 사전학습 데이터 덕분에 두 로봇 플랫폼에 대해 cross-embodiment long-horizon mobile manipulation 능력을 보여, 현장 배치 가능성을 실증했다.



### Learning to Throw Objects Safely in Multi-Obstacle Environments (https://arxiv.org/abs/2607.06388)
Comments:
          This paper has been presented at the IEEE International Conference on Robotics & Automation (ICRA), 2026

- **Prior Approaches**: 로봇 던지기는 로봇의 작업 반경을 넘어 빠른 배치를 가능하게 하지만, 기존 TossingBot 같은 방법은 장애물이 없는 환경을 주로 가정했습니다. 또 다른 접근은 수학적/수작업 모션 커널에 의존해 클러터 환경에서의 적응성이 떨어지거나, 장애물 정보를 상태에 직접 넣어 장애물 수가 늘면 스케일이 붕괴하는 문제가 있었습니다. 즉 “안전한 탐색+장애물 회피+일반화”를 동시에 만족시키기 어렵다는 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 장애물이 무작위로 배치된 장면에서 목표 바스켓에 던지되 충돌을 피하는 문제를 safe reinforcement learning으로 정식화합니다. 핵심 기여는 potential field state representation(PFR)으로, 바스켓의 attraction과 장애물의 repulsion을 고정 크기 grid에 함께 인코딩해 장애물 개수/배치가 달라도 단일 정책이 일반화하도록 한 점입니다. 또한 kinesthetic teaching으로 던지기 커널을 안전하게 초기화한 뒤 RL이 커널 파라미터를 조절하도록 설계해 초기 학습의 위험을 줄였습니다.

- **Technical Challenges**: 클러터 환경에서 RL이 처음부터 무작위로 탐색하면 충돌 위험이 커지며, 장애물 상태를 벡터로 직접 넣는 방식은 차원 증가로 학습이 비효율적입니다. 논문은 kinesthetic teaching으로 safe 커널을 부여해 unsafe exploration을 완화하고, 장애물 정보를 EPR(Explicit Pose Representation) 대신 PFR의 고정 차원 grid로 바꿔 확장성과 물리적으로 의미 있는 구조를 확보했습니다. 정책은 SAC, DDPG, TD3 중 SAC에서 가장 안정적인 성능을 보이며, PFR은 CNN 인코더로 잠재장 공간 구조를 보존하도록 구성했습니다.

- **Empirical Impact**: 시뮬레이션과 실로봇 실험 모두에서 PFR 기반 정책이 EPR 대비 높은 성공률과 더 나은 스케일링을 보였습니다. 특히 SAC 기준으로 장애물 수가 늘어도 성공률이 견조하게 유지되며, 보이지 않았던 throwable object(예: banana, coke can, sneaker)에 대한 일반화도 관찰됩니다. 실로봇에서는 unseen object까지 포함해 클러터 장면에서 최대 90% 성공률을 보고했으며, 이는 sim-to-real transfer가 실용 수준임을 시사합니다.



### Towards Real-World Applications with an Autonomous Powered Wheelchair (https://arxiv.org/abs/2607.06383)
- **Prior Approaches**: 기존 파워드 휠체어 연구는 자율성을 일부 제공하더라도, 실제 환경에서 작동하려면 고급 인지·내비게이션을 통합한 사용자 친화형 시스템이 부족하다는 한계가 있었다. 특히 공공·혼잡 환경에서는 동적 장애물, 불완전한 센싱, 안전 제약이 겹치며 end-to-end 통합의 난도가 커졌다.

- **Core Contribution**: 이 논문은 상용 self-balancing powered wheelchair인 Genny Zero에 RGB-D 기반 human-aware perception, 제스처 인터랙션, LiDAR 기반 localization·navigation을 결합한 proof-of-concept 자율 주행 휠체어를 제안한다. 장애인 보조 맥락에서 people-following과 remote hailing을 시연하며, shared autonomy로 확장 가능한 설계 방향을 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 균형 제어가 포함된 self-balancing 플랫폼의 동역학을 고려하면서도 내비게이션 스택을 안정적으로 운용하는 것, (2) 전방 위주 센싱 커버리지로 인한 인지/회피의 제약, (3) 사람 추적의 위치 잡음이 목표 재계획을 유발해 ‘멈춤-재개’ 같은 제어 진동을 만들 수 있다는 점이다. 저자들은 LiDAR 3D를 Nav2용 가상 2D scan으로 변환하고, 명령 timeout 안전장치와 조심스러운 속도 제한으로 초기 검증을 수행했으며, 향후로는 platform-aware controller와 3D 장애물 회피, omnidirectional sensing을 계획한다.

- **Empirical Impact**: 실내 연구실과 복도 환경에서 제스처로 활성화되는 people-following(문 통과·회전 포함)과 hailing(사용자 위치 기반 목표 생성 및 접근 후 leash mode 전환)을 실제 프로토타입으로 시연했다. 이는 상용 보조 기기 위에 자율 로보틱스와 HRI를 통합할 수 있음을 보여주는 동시에, 안전성·센싱 커버리지·동역학 반영 등 사용자 준비 단계의 필수 과제를 구체화했다.



### Training-Free Acceleration for Vision-Language-Action Models with Action Caching and Refinemen (https://arxiv.org/abs/2607.06370)
- **Prior Approaches**: VLA 모델은 비전·언어 입력을 받아 로봇의 연속 동작을 생성하며, 확산(diffusion)이나 flow matching 기반 액션 헤드를 붙이는 방식이 최근 성능을 이끌고 있다. 다만 이들 flow 기반 VLA는 액션을 만들기 위해 여러 번 velocity field를 반복 평가해야 해서, 제어 루프에서 실시간 지연의 병목이 된다. 기존 가속은 계층/토큰 가지치기나 특징 캐싱처럼 계산량을 줄이거나 일부 재사용하는 접근이 많았지만, 액션 헤드의 반복 denoising(=flow 적분) 자체를 근본적으로 생략하긴 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 training-free이면서 plug-and-play 형태의 외부 캐시인 ActionCache를 제안해, 과거에 성공적으로 생성된 액션 청크를 다음 생성의 warm-start로 재사용한다. 핵심은 warm-start를 ‘시간 연속성’ 휴리스틱이 아니라, ‘출력 공간(action space)의 재사용 가능한 생성 결과’를 이웃 검색(retrieval)으로 찾는 문제로 재구성한 점이다. 모델 가중치나 액션 헤드 구조를 바꾸지 않고도(추가 학습 모듈 없이) 지연을 줄이면서 성공률을 유지하도록 설계했다.

- **Technical Challenges**: 주요 기술적 과제는 (1) 다중 양상(multimodal) 문맥을 잘 대표하는 캐시 키를 매 시점 계산 부담 없이 만들고, (2) 캐시가 틀릴 때의 안전성을 확보하는 것이다. 논문은 VLM 백본의 출력 임베딩을 기반으로 sparse ternary random projection으로 키를 압축하고, cosine similarity로 Top-1 이웃을 찾되 hit threshold 기준으로 보수적 분기(히트면 소수 NFE로 refine/미스면 순수 Gaussian noise에서 전체 생성)를 적용한다. 또한 저장 단위를 내부 상태가 아닌 ‘액션 청크 그 자체’로 두어 특정 백본에 종속되지 않게 하면서, pending 버퍼와 보수적 커밋/폐기를 통해 성공 재사용만 반영한다.

- **Empirical Impact**: 실험은 시뮬레이션(VLABench, LIBERO)과 실로봇(SO-101)에서 진행됐고, ActionCache는 low-latency 구간에서도 높은 task success를 유지하며 속도를 크게 끌어올렸다. π0.5와 GR00T-N1.6에서 flow 기반 VLA 액션 헤드에 대해 최대 11.75× 및 34.43× 가속을 보고했으며, 캐시 오버헤드만 남기는 극단적인 설정에서도 지연이 2ms 이하(모델별 1ms 수준)로 작게 나타났다. 캐시 히트 품질(Top-1 cosine similarity), hit threshold, 캐시 용량/교체정책이 성공률–지연–메모리의 트레이드오프를 좌우하는 제어 레버임을 상세 분석해, cross-task 재사용 가능성과 함께 실용적 튜닝 방법도 제시한다.



### Responsible Personalisation: The Double-Edged Sword of Personalisation in Human-Robot Interaction (https://arxiv.org/abs/2607.06344)
Comments:
          36 pages, 3 figures

- **Prior Approaches**: 기존 HRI 연구에서 personalisation은 성과(참여도, 과업효율, 신뢰·협업) 중심으로 다뤄져 왔지만, responsible personalisation 관련 윤리 리스크는 맥락별로 단편적으로만 보고되어 왔다. 또한 HCI에서 agency 상실, 조작, 고정관념, 윤리적 설계 등이 논의돼 왔으나, 로봇의 embodiment와 사회적 존재감이 리스크를 어떻게 증폭·재구성하는지는 HRI에서 체계적으로 정리되지 않았다. 일부 사회로봇 감시/조작 비판 연구가 존재하지만, 본 논문은 이를 설계·운영 관점의 라이프사이클 리스크 분석과 연결해 더 구조화한다는 입장이다.

- **Core Contribution**: 본 논문은 embodiment-aware 관점을 토대로, personalised HRI의 personalisation 과정을 6단계(설계-데이터수집-모델링-상호작용-평가-종료)로 보고 interaction 맥락(단기/장기, open/closed domain)과 결합해 리스크가 ‘어떻게 생기고 시간이 지나며 어떻게 변하는지’를 분석하는 프레임워크를 제시한다. 동시에 주요 윤리 리스크(자율성 침식, 편향된 user modelling, manipulation, 탈인간화, 프라이버시 침해)를 통합적으로 정리하고, 이를 설계 권고와 오픈 연구과제로 번역한다. 핵심은 personalised robot behaviour의 설계 공간과 risk landscape를 한 체계 안에서 구조화해, 더 투명하고 윤리적으로 근거 있는 접근의 기반을 제공하는 것이다.

- **Technical Challenges**: 가장 큰 기술적 과제는 로봇의 embodiment가 개인화 효과를 높이는 동시에 주의환기(salience)와 agency attribution을 강화해 영향력·오해·해악 가능성도 같이 키운다는 점을, 라이프사이클의 각 단계에서 추적 가능한 형태로 다루는 것이다. 이를 위해 논문은 customisation(사용자 주도·명시적), adaptation(상황/집단 단서 기반), personalisation(특정 개인의 persistent user model을 통한 시스템 주도)을 연속선상에서 구분해 통제·추론·범위 차이를 정리하고, 입력-모델링-출력(IMO) 흐름으로 리스크가 고착되는 지점을 설명한다. 특히 데이터 수집(멀티모달·암묵 입력 확대)과 모델링(편향·cold-start에 따른 스테레오타입), 상호작용(센서모터 출력의 사회적 파급), 평가·종료(자기평가와 memory/데이터 관리)에서 리스크가 단계적으로 발생·지속될 수 있음을 구조화해 해결 경로를 제안한다.

- **Empirical Impact**: 본 초록/서술 범위에서는 주로 프레임워크와 분석에 초점이 있으며, 실험적 성능 수치의 직접 보고보다는 워크숍 논의와 기존 연구 근거를 통합해 리스크 지도를 만든다는 성격이 강하다. 다만 embodiment가 인지·학습 성과 및 신뢰/지속 사용 의사에 영향을 줄 수 있다는 기존 결과들을 인용해, 개인화가 실제로 ‘더 설득력 있고’ 사회적으로 더 강한 효과를 가질 수 있음을 경험적으로 뒷받침한다. 결과적으로 이 프레임워크는 HRI 커뮤니티가 personalisation을 단순 기능 향상으로만 보지 않고, 맥락·단계·책임을 함께 설계·평가하도록 방향을 제시하는 기반이 될 것으로 기대된다.



### OrchardBench: A Physically-Grounded, GPU-Parallel Apple-Orchard Simulation Benchmark for Agricultural Robotics (https://arxiv.org/abs/2607.06337)
- **Prior Approaches**: 기존 농업 로보틱스는 현장 실험 의존도가 높아 재현성과 비용 문제가 컸습니다. 시뮬레이션은 GPU-parallel 물리 학습을 가능케 했지만, 대부분의 과수 환경은 딱딱한 물체나 정적인 식물 지오메트리 위주라 ‘접촉-굴곡-파손-열매 분리’ 같은 생물학적 물리성을 충분히 담지 못했습니다.
또한 컴퓨터그래픽/기능-구조 식물 모델은 정교한 외형을 제공하더라도 로봇의 힘에 대한 동역학, 파손 거동, 열매 탈착 메커니즘이 결여되어 현장 반복을 대신하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: OrchardBench는 사과 과수의 나무를 물리적으로 구동되는 ‘breakable plant’로 모델링한 GPU-parallel 시뮬레이션 벤치마크를 제안합니다. 핵심은 가지를 Euler-Bernoulli beam 이론 기반의 합성 탄성/감쇠 관절로 만들고, 휨 모멘트가 파괴 한계를 넘으면 자유 힌지처럼 떨어지게 하며, 열매는 문헌 기반의 인장 임계 힘에서 줄기-스템 tether가 분리되도록 구현한 점입니다.
또한 잎이 흔들리며 과수 캐노피를 가리는 ‘moving, density-controllable foliage layer’를 포함해 인식 및 조작의 가장 어려운 변수(가림)를 제어 가능한 형태로 제공합니다.

- **Technical Challenges**: 과수는 수백 개 수준의 연성 관절과 다수의 자유 열매가 얽혀 GPU에서 수많은 환경을 동시에 안정적으로 계산하기 어렵습니다. OrchardBench는 Newton 엔진(MuJoCo-Warp 계열)을 활용하면서도 파손 시점에 모델을 재컴파일하지 않고, 관절 스프링의 암/바이어스 행을 즉시 꺼서 rupture를 in-place로 처리하는 방식으로 계산 효율과 수치 안정성을 동시에 노렸습니다.
여기에 사과는 전체 자유 강체로 두기보다 DOF를 줄인 tether-기반 분리 모델로 구성해 배치 시뮬레이션 비용을 낮추고, solver 안정장치(충격/속도 제한 등)로 과도한 에너지 주입을 방지합니다.

- **Empirical Impact**: 논문은 수확 임무에서 harvest completeness, throughput, plant damage를 포함한 메트릭을 정의하고, 잎 밀도·과일 하중·지형·캐노피 존·병렬성 조건별로 베이스라인 성능을 보고합니다. 그 결과 기하학적 과일 감지와 자율 수확 베이스라인은 감지한 과일의 약 40%를 수확하지만, 도달 가능한 과일 대비 약 1/8 수준만 수확해 안전하고 현실적인 물리 모델에서도 자율성 여지가 크다는 점을 보여줍니다.
즉, 이 벤치마크는 (1) 접촉/파손을 포함한 안전 검증, (2) sim-to-real 사전학습, (3) 가림 제어 기반의 perception 연구, (4) 농업 지표 중심의 최적화라는 다중 연구 프로그램에 바로 쓸 수 있는 공통 실험장으로 의미가 있습니다.



### LAMP: Latent Motion Prior-Guided Real-World Learning for Dexterous Hand Manipulation (https://arxiv.org/abs/2607.06323)
Comments:
          17 pages, 11 figures

- **Prior Approaches**: 기존 연구는 시뮬레이션-투-실현(sim-to-real)이나 모방학습(imitation learning, IL)에 크게 의존하지만, 다지(多指) 접촉은 비선형·불연속적이라 reality gap이 커지고 성능이 흔들린다. IL은 학습된 시연을 재현하는 데 강하지만 고차원 손 행동공간에서 작은 오차가 out-of-distribution(OOD)으로 누적되며, 보완을 위한 탐색·오차복구 능력이 부족하다. 온라인 RL은 이를 해결할 잠재력이 있으나, 무가이드 탐색은 접촉을 깨뜨려 샘플 효율이 극도로 낮아지고 하드웨어에서 위험하다.

- **Core Contribution**: 이 논문은 최근 손 행동 이력(history)을 조건으로 하는 history-conditioned latent motion prior(LMPM)를 제안해, 고차원 손 조인트 명령을 실행 가능한 연속 잠재공간으로 매핑한다. LAMP는 이 prior를 공통 인터페이스로 고정한 뒤, (1) 시연으로 prior를 사전학습하고 (2) 비전-운동 정책이 팔(native arm) 명령과 잠재 손 오프셋을 예측하며 (3) 동일 잠재공간에서 residual reinforcement learning으로 국소 보정을 수행한다. 덕분에 온라인 탐색이 모든 손가락 조인트를 독립적으로 흔드는 방식이 아니라, 시연과 접촉 일관성을 유지하는 “근처(local)” 궤도에서 수정되도록 설계됐다.

- **Technical Challenges**: 핵심 기술 난제는 (a) 고차원 손 행동을 압축하더라도 실제 하드웨어에서 “실행 가능한(decodable)” 명령으로 되돌릴 수 있어야 하고, (b) 잔차 RL이 접촉을 깨지 않는 방향으로 국소 탐색을 하도록 행동공간을 잘 조건화해야 한다는 점이다. 논문은 LMPM을 KL-정규화된 잠재 병목(latent bottleneck)과 이력 조건부 prior center로 학습해 매 단계에서 국소적으로 의미 있는 잠재 기준점을 제공하고, IL과 residual RL이 같은 디코더를 공유하도록 파이프라인을 통일했다. 그 결과 residual은 원공간의 임의 교란이 아니라 잠재좌표에서의 연속 오프셋 업데이트로 표현되어 접촉 보존 탐색이 가능해졌다.

- **Empirical Impact**: Franka Research 3 팔과 Ruiyan dexterous hand에서 4개 실세계 조작 작업을 평가한 결과, LAMP는 시연 소량에서 평균 IL success rate 56.25%를 달성하고 온라인 RL 후 98.75%로 끌어올렸다. 특히 4개 중 3개 작업은 최종 성공률 100%에 도달했고, 남은 1개도 95%를 기록했다. 비교 실험에서 Raw·PCA·VQ-VAE 기반 핸드 인터페이스보다 IL 및 residual RL의 안정성과 최종 성능이 일관되게 높았으며, 오프-매니폴드 탐색이 줄어드는 효과도 정량적으로 확인돼 실세계 다지 조작 학습의 “행동 인터페이스” 관점에서 의미가 크다.



### Optimal Transport Q-Learning for Flow Policy Steering and Acceleration (https://arxiv.org/abs/2607.06262)
- **Prior Approaches**: 최근 로보틱스에서는 diffusion과 flow가 인간이 만든 텔레오퍼레이션/키네스틱 티칭의 멀티모달 궤적 분포를 잘 학습해 성능이 크게 좋아졌습니다. 다만 SDE/ODE를 여러 단계 적분해야 해서 추론이 느리고, offline 데이터만으로는 분포 이동(out-of-distribution)에서 자주 실패하며 고품질 추가 데모 수집도 부담입니다. RL로 보정하는 방식도 있으나, flow/diffusion의 “빠른 추론”과 RL post-training을 함께 만족시키는 연구는 상대적으로 드뭅니다.

- **Core Contribution**: 이 논문은 flow 정책을 RL post-training으로 미세조정하면서도 적분 단계 수를 크게 줄이기 위한 OTQL(Optimal Transport Q-Learning)을 제안합니다. 핵심 아이디어는 advantage를 이용해 “가치가 높은 행동”에 질량을 더 실어 주는 conditional optimal transport를 정의하고, 이를 flow matching 학습에 결합해 straight integration path를 유도하는 것입니다. 결과적으로 비싼 distillation 없이도 로봇의 own experience로 suboptimal flow 기반 정책을 개선하고 가속합니다.

- **Technical Challenges**: OTQL의 기술적 난제는 (1) conditional optimal transport를 RL에서 쓰기 좋은 방식으로 근사하고, (2) critic이 만드는 advantage 신호가 학습 안정성과 샘플 효율에 악영향을 주지 않게 하는 것입니다. 저자들은 OT-CFM(wCOT-CFM) 형태로 “에너지(=advantage 관련)”에 따라 타깃 분포의 가중치를 재정의하고, 조건이 동일한 상태 쌍끼리 매칭되도록 condition-consistent coupling을 구성해 straight한 전이를 학습합니다. 또한 optimal coupling에서 샘플이 중복/탈락되며 배치 유효 크기가 줄어드는 문제를 줄이기 위해 가중치 클램핑 등 실용적 보정도 함께 적용합니다.

- **Empirical Impact**: 실험 결과 OTQL은 시뮬레이션과 real-world 작업에서 미세조정 성능을 기존 steering 계열 방법과 비슷한 수준으로 끌어올리면서도, action 생성 시 필요한 inference step(또는 NFEs)을 약 70% 줄였습니다. 특히 단일 태스크 정책의 성공률은 36%에서 86%로, 사전학습 VLA 성능은 38%에서 76%로 향상되었으며 50~60 에피소드 상호작용 예산 내에서 fine-tuning과 가속을 동시에 달성합니다. 이는 “추론 속도”가 중요한 로보틱스 환경에서 RL post-training을 현실적으로 확장할 수 있음을 보여주는 대목입니다.



### Diagnosing Semantic Handoff Failures in Agent-Orchestrated Vision-Language-Action Skill Composition (https://arxiv.org/abs/2607.06256)
- **Prior Approaches**: 기존 연구는 언어 조건 VLA(vision-language-action)나 도구 호출 에이전트가 장기 작업을 분해해 실행하게 하되, 스킬 간 경계가 “명시적 인터페이스”로 정의되지 않아 다음 스킬 시작 상태가 불완전한 문제가 남아 있다. 또한 독립 스킬 성공률 중심 평가는 실제 연쇄 실행에서의 준비도 부족(semantic handoff 실패)을 가리기 어렵다.

- **Core Contribution**: 이 논문은 장기 실행의 핵심 문제를 semantic handoff problem으로 정식화하고, 스킬이 자기 postcondition은 만족하지만 다음 스킬 실행을 보장하는 상태(ready state)를 남기지 못하는 경우를 진단한다. BEHAVIOR-1K에서 π0.5 기반 스킬 체크포인트들을 에이전트 오케스트레이션 실행 하네스로 돌려, 스킬 경계에서 “진행/재시도/재계획”을 검증기로 결정하는 구조를 제시한다.

- **Technical Challenges**: 기여를 위해 필요한 기술적 난관은 (1) 스킬 종료 판정이 단순 완료 여부가 아니라 다음 스킬의 가시적·물리적 준비도를 반영해야 한다는 점, (2) 연쇄된 초기 상태가 시연 기반 스냅샷 분포와 달라져 분포 shift가 생긴다는 점이다. 논문은 멀티뷰 VLM verifier로 head/양손 관측을 바탕으로 postcondition을 넘어서(예: 팔-도달 arm-reach) 진행을 통제하고, 단계 예산/검증 주기/재계획까지 포함한 trace를 남겨 실패 원인을 next-skill readiness, target grounding, low-level control 실행으로 분류한다.

- **Empirical Impact**: 실험에서 일부 내비게이션·그립·배치·도어 오픈 스킬은 “깨끗한 스냅샷”에서는 77~100%의 성공을 보이지만, 실제 연쇄 terminal state에서는 스톨이 자주 발생해 스냅샷 능력과 장기 조합 견고성 간 큰 격차가 확인된다. 검증 기준을 arm-reach로 강화하면 readiness 관련 실패가 더 포착되며 특정 태스크(라디오 등)의 회복도 관찰된다. 저자들은 near-zero end-to-end 성공을 단순 성능 저하가 아니라 “VLA 스킬 라이브러리의 다음 과제(연쇄 상태 견고성)”를 드러내는 진단 지표로 전환할 수 있음을 보여준다.



### RoboVAST: Automated Scenario-Based Validation of Robots at Sca (https://arxiv.org/abs/2607.06248)
Comments:
          8 pages, 4 figures, submitted to CASE 2026

- **Prior Approaches**: 기존 시나리오 기반 검증은 pass/fail 중심의 판단을 내리되, 실제로는 수동·경험 기반 시나리오 선택 때문에 재현성과 결론의 일반화가 약해질 수 있다. 또한 많은 접근이 시나리오를 환경·작업·시스템 파라미터·컨텍스트를 조합 가능한 구성요소가 아니라 “단일 테스트 케이스”처럼 다루는 경향이 있어 차원별 체계적 변이를 어렵게 만든다. 시나리오 공간이 조합 폭발로 사실상 무한대에 가까워질 때, 무엇을 얼마나 해야 충분한지(coverage/complete)도 일관되게 정의하기 어렵다.

- **Core Contribution**: 이 논문은 시나리오를 환경/작업/로봇 시스템 설정/컨텍스트를 분리해 조합(compositional)하고, 변이·생성·실행·해석을 선언적으로 formalize하는 시나리오 기반 검증 방법론을 제안한다. 이를 구현한 RoboVAST는 declarative campaign specification(선언형 캠페인 명세), plugin-based scenario generation(플러그인 기반 시나리오 생성), containerized scalable execution(컨테이너 기반 대규모 실행)과 통합 result analysis를 제공한다. 결과적으로 특정 개별 시나리오에 종속되지 않고, 변이 차원 전반에 걸친 “비국소(non-local) 결론”을 도출하는 검증 파이프라인을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 조합적으로 큰 테스트 공간에서 의미 있는 coverage/신뢰도를 만들고 (2) 변이 차원 간 제약을 만족하는 “실행 가능” 시나리오만 자동 생성하며 (3) 대량 실행 결과를 재현 가능하게 오라클 평가로 연결하는 것이다. RoboVAST는 variability dimensions을 타입과 도메인으로 모델링하고, abstract/logical/concrete 시나리오 계층을 두어 producer 순서와 random seed를 명시함으로써 결정적(instantiation) 구성과 비교 가능성을 보장한다. 또한 Kubernetes 기반 병렬 실행과 표준화된 결과 디렉토리 레이아웃, 그리고 trace 기반 관측/평가(오라클) 분리를 통해 대규모 캠페인 후처리까지 자동화한다.

- **Empirical Impact**: 실험에서는 내비게이션 데이터셋으로 5480개의 시나리오 구성과 100,000회 이상 실행을 수행했으며, 5개의 실내 맵에서 경로·센서 잡음·소프트웨어 파라미터·장애물 설정을 다양화했다. 총 모의 운행 시간은 1800시간 초과, 누적 이동 거리는 1873km를 달성했고, 각 구성에 대해 20회 반복 실행을 통해 체계적 실패와 확률적 이상(stochastic anomaly)을 분리할 수 있음을 보여준다. 이는 로보틱스 검증에서 시나리오 변이의 스케일업과 통계적 신뢰 확보를 실증적으로 뒷받침하며, 재현 가능한 대규모 시나리오 캠페인 구축을 촉진하는 도구/프레임워크로서 의미가 있다.



### APVI-SLAM: Real-Time Acoustic-Pressure-Visual-Inertial Localization and Photorealistic Mapping System in Complex Underwater Environmen (https://arxiv.org/abs/2607.06222)
- **Prior Approaches**: 기존 수중 Visual–Inertial SLAM은 조명 감쇠·부유 입자·해양 교란으로 인해 특징이 지속적으로 열화되면 포즈 추정이 불안정해지기 쉽다. 다중 센서를 factor graph에 결합하더라도, 시각 입력 붕괴가 누적되면 공분산 기반 잡음모델만으로는 전체 추정이 쉽게 발산하고, 완전한 visual dropout에는 재초기화 비용이 커진다.

- **Core Contribution**: APVI-SLAM은 DVL·압력·IMU를 함께 쓰되, 시각 열화 신뢰도를 추정해 센서 추정치 가중치를 동적으로 조절하는 reliability-aware localization으로 estimator divergence를 줄이는 데 초점을 둔다. 또한 visual tracking 실패 시 슬라이딩 윈도우 freezing으로 빠른 복구를 유도하고, quadtree-guided 3D Gaussians mapping으로 고해상도·고사진실(photorealistic) 지도를 실시간에 가깝게 만든다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 시각 특징이 간헐적으로 사라질 때도 안정적인 포즈를 유지하는 신뢰도 판단, (2) 실패 구간 동안 잘못된 시각 정보가 지도 최적화를 오염시키는 문제, (3) 탁한 수중에서 발생하는 수중 매질 광학 효과(domain shift)를 포함한 증분 3DGS 학습의 계산/수렴성이다. APVI-SLAM은 coarse-to-fine self-discrimination으로 VIO/DIPO 안정성을 분기하고, VIO가 깨지면 factor graph에 “freeze”된 키프레임/특징을 유지해 재초기화 없이 수렴을 복원하며, 수중 매질 파라미터와 함께 quadtree-guided densification 및 3D Gaussian 최적화를 통해 점진 지도를 갱신한다.

- **Empirical Impact**: 실험은 Tank급 시뮬레이션(6개 시나리오)과 실제 산호초(coral reef) 8시퀀스 데이터에서 수행되며, PSNR·SSIM·LPIPS 및 ATE/RPE로 로컬라이제이션·재구성 품질을 함께 평가한다. 논문에 따르면 APVI-SLAM은 공개 벤치마크와 신규 산호초 데이터에서 localization 정확도와 photorealistic reconstruction 모두에서 실시간 속도를 유지하며 state-of-the-art 수준을 보였고, 수중 매핑 평가용 벤치마크 공백을 데이터셋으로 메웠다는 점에서 의미가 크다.



### Calf-Integrated Arms for Bimanual Quadruped Loco-Manipulation (https://arxiv.org/abs/2607.06186)
Comments:
          6 pages, 6 figures

- **Prior Approaches**: 기존 loco-manipulation 설계는 ‘조작 성능’과 ‘자세 유지(stance)’를 맞바꾸는 경우가 많았다. trunk-mounted arm은 몸체가 높아 바닥 물체를 잡기 위해 더 아래로 뻗어야 하고, 대개 한 팔만 써서 양손 작업을 동시에 수행하기 어렵다. 다리(legs)를 그리퍼로 쓰거나 leg-mounted gripper는 양손 작업을 위해 rearing(뒷다리 들기)을 필요로 해 기반(base)이 고정되거나 네 발 지면 접촉을 유지하기 힘들었다.

- **Core Contribution**: 이 논문은 Unitree Go2의 각 front calf에 prismatic slider, 2개의 revolute joint, gripper를 통합해 바닥 높이에서 두 팔(bimanual)로 물체를 잡고 조작하면서도 네 발을 항상 planted 상태로 유지하는 하드웨어를 제안한다. 또한 한 팔만 들 때는 다른 쪽 팔을 접어 기동 여유를 보존해 “걷기 + 조작”을 동시에 가능하게 한다. 마지막으로, head-camera 이미지와 task state에 조건을 건 vision-language model이 predefined skill library에서 다음 스킬을 경계(skill boundary)마다 선택해 장기 자율(long-horizon) 수행을 목표로 한다.

- **Technical Challenges**: 기여를 실현하려면 (1) 바닥 레벨 도달성과 (2) 두 팔이 같은 물체에 정렬되는 공간/기구학 설계, (3) 장기 스킬 선택을 위한 제어·플래닝 통합이 모두 필요하다. 저자들은 calf 내에 slider를 내장하고 yaw 관절로 두 그리퍼를 body centreline 쪽으로 스윕시켜 두 팔 작업 공간의 겹침을 확보했으며, DLS inverse kinematics로 pinch point를 추적하고 슬라이더 사용도 제한 옵션을 통해 안정적으로 접근시킨다. 상위에서는 VLM이 스킬 라이브러리 중 하나를 선택하고, 하위에서는 각 스킬을 FSM이 실행해 locomotion 정책이 지면 접촉을 유지하도록 분리 제어한다.

- **Empirical Impact**: 시뮬레이션에서 서로 다른 3가지 양손 작업(캐비닛 앞 장기 작업, 협동 들어올리기, 팔 간 handover)을 모두 네 발 planted 상태로 수행하며 설계 의도(바닥 레벨 양손 + 장기 스킬 시퀀싱)를 입증했다. 특히 캐비닛 작업은 고정된 스킬 순서가 아니라 instruction만 주고 task progress flag를 갱신하며 스킬 순서를 동적으로 형성하며, VLM 호출은 스킬 경계마다 4회로 제한돼 전체 지연이 관리된다. 또한 depth 잡음에 대한 견고성(표준편차 10mm까지 성공률 70~74% 유지)과 한계(롤 조인트 부재로 특정 물체 핸들 회전 불가, 마커 기반 지각 한계, 라이브러리 커버리지 밖 행동 불가)를 함께 제시해 향후 real-world 검증의 방향을 명확히 했다.



### EAGOR: Embodied Reasoning in Omni-direction (https://arxiv.org/abs/2607.06165)
Comments:
          12 Pages, 7 Figures, 4 Tables

- **Prior Approaches**: 기존 방법들은 360° 관측을 ERP(equirectangular projection)로 2D 이미지화한 뒤, VLM이 만든 방향/좌표 추정을 그대로 사용해 왔다. 하지만 ERP는 seam 불연속과 위도 왜곡을 만들어 에이전트의 시점 변환(회전/이동)에서 방향 추정이 일관되지 않으며, map-free navigation 같은 폐루프 제어에서 문제가 더 커진다. 또한 많은 접근이 불확실성을 무시한 결정적 좌표 예측에 치우쳐, 연속적인 방향 추적을 확률적으로 누적·유지하기 어렵다는 한계가 있었다.

- **Core Contribution**: EAGOR는 training-free로 VLM의 의미론적 증거를 ‘구면(sphere) 위의 연속 확률 믿음’으로 바꿔, 에이전트 동작에도 일관된 방향 추론을 수행한다. ERP 픽셀 좌표로 직접 방향을 예측하는 대신, 타깃 방향을 구면에서의 재귀적 베이지안 추정으로 모델링해 egocentric 방향 벡터를 연속적으로 갱신한다. 이를 위해 Spherical Harmonic Belief Field(SH-BF)를 제안해, VLM을 학습하지 않고도 구면 기하를 존중하는 방향 추정 파이프라인을 구성한다.

- **Technical Challenges**: 핵심 난제는 (1) ERP의 변환/왜곡에 덜 민감하면서도 (2) 시간에 따라 회전이 누적되는 동안 믿음을 확률적으로 누적하고(덧셈), 회전에는 등변적으로 전파해야 한다는 점이다. EAGOR는 VLM의 ERP 응답 맵을 viewing direction으로 lifting해 구면의 관측 likelihood로 해석하고, 이를 log-likelihood 형태로 SH 공간에 투영해 재귀 베이지안 업데이트를 계수 공간에서 수행한다. 또 Wigner-D rotation으로 이전 posterior를 현재 시점 기준으로 정확히 정렬한 뒤, SH의 degree-1 계수(구면 프레셋 평균/Fréchet mean 성질)를 통해 그럴듯한 단일 방향을 grid search나 gradient ascent 없이 디코딩한다.

- **Empirical Impact**: 시뮬레이션(Habitat-Sim/HOS/OSR-Bench)과 실세계(Unitree-Go2+Insta360-X3)에서 EAGOR는 기존 방법 대비 일관되게 더 좋은 방향 추정과 제어 성능을 보였다. Active Visual Search에서는 HOS와 OSR-Bench에서 평균 상대 개선이 각각 +34.4%, +45.6%였고, map-free navigation에서는 성공률 +14.6%, 단계 수 17.7% 감소, 평균 각도 오차 24.5% 감소를 보고한다. 특히 SH-BF 기반의 구면 믿음 추정이 VLM 백본을 키우는 것보다 더 큰 이득을 주며, 더 작은 모델도 더 큰 모델을 능가하는 등 분야에서 ‘기하 일관성의 가치’를 실증했다.



### MP-MPPI: A Motion Primitive Guided Sampling-Based Optimizer for Model Predictive Contro (https://arxiv.org/abs/2607.06123)
- **Prior Approaches**: 샘플링 기반 MPC인 MPPI는 미분가능 동역학이 없어도 비용을 최소화하는 제어 입력을 찾을 수 있다. 하지만 초기 제어 입력 주변에 잡음을 더해 탐색하므로, 잡음 크기를 키우면 전역 최적성은 좋아져도 폐루프 안정성이 흔들려 성능이 떨어질 수 있다는 한계가 있다. 이를 보완하려고 DIAL-MPC·MPOPI처럼 잡음 스케줄링을 반복하거나, BiC-MPPI·o-MPPI처럼 역가능 동역학을 요구하는 방식, 혹은 샘플링 분포를 크게 바꾸는 방식들이 제안됐지만 계산량 증가나 제약 조건, 편향(bias) 부작용이 남는다.

- **Core Contribution**: 이 논문은 MPPI에 motion primitives를 결합한 MP-MPPI를 제안해 탐색성을 높이면서도 빠른 실시간 반응을 유지하는 것을 목표로 한다. 제어 공간 탐색을 위해 미리 계산된 feasible lattice state motion primitive 샘플을 추가하고, 이를 기존 MPPI의 잡음 기반 perturbed control sequence와 함께 비용 평가 후 가중합 업데이트에 반영한다. 그 결과 국소 최소해에 머무르기 쉬운 기존 MPPI의 경향을 완화해 더 전역적인 해결 경로를 찾도록 돕는다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) motion primitives를 동역학적으로 feasible하게 만들면서 (2) 너무 많은 primitive가 오히려 나쁜 궤적에 편향을 주지 않도록 샘플 혼합 비율을 설계하는 것이다. 논문은 primitive를 상태 격자 기반으로 OCP를 풀어 만든 제어 시퀀스로 구축하고, MPPI 루프에서는 이 primitive 샘플과 백색 잡음 샘플을 동시에 생성해 병렬 비용 평가(RK4 기반 적분 포함)를 수행한다. 또한 장애물 회피는 충돌 경로에 큰 비용을 부여하는 비용 수정으로 처리해, primitive가 제공하는 대안 해가 벽/장애물 상황에서 실제로 선택되도록 한다.

- **Empirical Impact**: 장애물 필드 내비게이션 시뮬레이션(무작위 300개 기둥, 100회 반복)에서 MP-MPPI는 이동 거리 평균이 48.0m에서 66.6m로 증가했고 충돌은 0회로 줄었다(기본 MPPI는 1회 충돌). 또한 벽을 갑자기 인지하는 reactivity 테스트에서 MP-MPPI는 충돌 회피 궤적을 만들었지만 MPPI는 충돌했다. GPU(JAX, RTX 2000 Ada Laptop)에서 샘플 수/예측 지평 길이에 따른 업데이트 주기가 관리 가능하며, 100Hz 이상 운용을 목표로 tuning했을 때 motion primitives 추가에도 지연이 크게 증가하지 않아 실시간 적용 가능성을 보여준다.



### ThorArena: Benchmarking Humanoid Physical Interaction with Human Motion-Force Demonstrations (https://arxiv.org/abs/2607.06052)
- **Prior Approaches**: 기존 휴머노이드 벤치마크는 주로 관절/자세 같은 kinematic 추적 오차나 작업 성공 여부, 자연스러움에 초점을 두고 상호작용 forces는 암묵적 교란으로 취급해 왔습니다. 그 결과 정책이 외부 힘을 받았을 때 균형, 안정성, 제어 강건성, 제어 비용이 어떻게 변하는지 정량적으로 드러나지 않는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 동기화된 human motion과 hand interaction forces를 함께 재현·평가하는 force-aware 벤치마크 ThorArena를 제안합니다. Force-Aware Tracking Score(FATS)와 진단 지표(robustness ratio, power overhead)를 통해 추적 정확도뿐 아니라 힘 수준별 강건성, 제어 노력, episode survival까지 한 번에 채점합니다. 또한 시뮬레이터에서 recorded forces를 재생(force replay)하는 통합 프로토콜로 서로 다른 whole-body control 정책도 동일 조건에서 비교할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 “동일한 contact 상황”을 공정하게 반복하면서 forces가 평가에 직접 반영되도록 만드는 것이었습니다. 이를 위해 실세계에서 두 손 힘 센서를 부착해 동작과 forces를 동기화 수집하고, 시뮬레이션에서 손 포즈에 따라 힘을 world frame으로 변환해 적용하는 force-replay 파이프라인을 구축했습니다. 더불어 정책별 관측/행동 형태가 달라도 동일한 벤치마크 러너가 굴러가도록 policy-adapter interface를 분리 설계했습니다.

- **Empirical Impact**: ThorArena 실험에서는 Thor2, TWIST2, GMT, SONIC을 6개 작업에서 no-force와 external-force 조건으로 나눠 평가했으며, force 조건에서 평가 격차가 크게 드러났습니다. 특히 push_chair 같은 지속적인 수평 힘 상호작용에서 일부 정책은 tracking은 물론 survival까지 크게 저하됐고, no-force 평가에서는 거의 구분이 안 되던 성능 차이가 FATS와 진단 지표로 명확해졌습니다. 논문은 ThorArena가 contact-rich 휴머노이드 제어 연구에서 force robustness와 제어 비용을 포함한 보다 현실적인 비교 기준을 제공한다고 주장합니다.



### RoboTALES: Learning Reasoning-Guided Robot Policies via Task-Aligned Simulated Futures (https://arxiv.org/abs/2607.06018)
Comments:
          Accepted at ECCV 2026

- **Prior Approaches**: 사전학습 비디오 생성 모델을 visuomotor control의 백본으로 쓰는 시도가 늘었지만, 생성된 미래는 작업 의도와 느슨하게만 정렬되는 경우가 많아 계획이나 policy 추출에 불리합니다. Video-Policy, Gen2Act, ViPRA 같은 방법은 rollout을 감독/조건으로 활용하나, 비디오 생성 자체가 시각적 사실성이나 재구성에 주로 최적화돼 action-conditional 정합성이 약하다는 한계가 있습니다. 또한 언어로 세운 계획을 행동 선택에만 쓰고, 실제 imagination의 예측 표현 내부를 직접 형성하지 못해 closed-loop 의미 정렬이 어렵다는 문제도 제기됩니다.

- **Core Contribution**: RoboTALES는 단일-stage로 task-aligned simulated future를 학습하고, 그 미래를 통해 로봇 policy를 훈련하도록 만드는 프레임워크입니다. 핵심은 LLM Planner가 작업을 계층적 subgoal 시퀀스로 쪼개 비디오 generator의 상상(rollout) 생성 자체를 구조화하고, VLM Critic이 생성된 미래가 지시문과 맞는지 평가해 reward 피드백으로 의미 정합성을 유지하는 closed-loop 정렬 메커니즘을 도입한 점입니다. 결과적으로 temporally consistent rollout과 더 coherent한 행동이 나오도록 설계했습니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 장기 과제를 의미 단위로 분해해 비디오 확산 모델이 의도에 맞게 미래를 생성하도록 만드는 것과, (2) 생성은 그럴듯하지만 작업 의미를 어기는 미래를 내부 표현 수준에서 걸러내는 것입니다. RoboTALES는 cross-attention에 subgoal 토큰을 주입해 diffusion VideoUNet의 생성 과정을 계획 기반으로 고정하고, VLM Critic 점수를 differentiable policy optimization 흐름에 연결해 숨은 상태(hidden states)까지 task-관련 의미로 ‘representational steering’ 되게 학습합니다. 또한 비디오 generator와 action diffusion UNet을 함께 최적화해 action 목적의 그래디언트가 생성기의 decoder로 되돌아가 “imagine for acting”이 가능하도록 단일-stage end-to-end 학습을 구현했습니다.

- **Empirical Impact**: RoboCasa와 LIBERO10의 다양한 조작 과제(총 34개)에서 RoboTALES는 기존 방법을 일관되게 능가하며, 특히 long-horizon에서 격차가 크게 나타났습니다. 예를 들어 RoboCasa Pick-and-Place에서 평균 성공률 48%로 개선을 보였고, turning과 pressing처럼 다단계 과제에서도 각각 64%, 96%의 평균 성공률을 달성했습니다. 또한 50 데모만으로도 더 많은 데모를 쓰는 prior 대비 성능이 높게 보고돼 sample efficiency와 장기 과제 안정성 측면의 의미가 큽니다.



### Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failur (https://arxiv.org/abs/2607.05966)
Comments:
          9 Pages Workshop Paper accepted at RSS Robot World Model Workshop 2026

- **Prior Approaches**: 기존 world models의 장기 실패는 대개 compounding error(오차 누적)로 뭉뚱그려 설명돼, 어떤 오류가 어떤 형태로 누적되는지 구분이 부족했습니다. 그 결과, 모델이 실제 물리의 동역학적 일관성을 잃는지와 단순한 운동학적 일탈에 그치는지는 실증적으로 분해해 보기 어려웠습니다.

- **Core Contribution**: 이 논문은 실패 원인을 kinematic-vs-dynamic 관점으로 재정의하며, world models이 동역학(dynamics)보다 운동학(kinematics) 쪽으로 상상하는 경향이 있다고 주장합니다. 이를 위해 per-step diagnostic인 imagined Kinematic-Consistency Error(iKCE)를 제안하고, 물리 조건이 regime boundary를 넘는지 여부에 따라 perturbation protocol로 iKCE의 반응을 점검합니다.

- **Technical Challenges**: 핵심 과제는 장기 롤아웃에서 “운동학적으로 일관된 상상”과 “동역학적으로 타당한 상상”을 구분할 수 있는 정량 지표를 설계하는 것이었습니다. 저자들은 롤아웃이 닫힌형(closed-form) kinematic null로부터 얼마나 벗어나는지로 iKCE를 정의하고, 마찰 조건 등으로 regime boundary를 교차시키며 진단 신호가 나타나는지 검증하는 절차를 결합했습니다.

- **Empirical Impact**: DreamerV3 체크포인트(DMC walker-walk) 실험에서 imagined iKCE는 matched real-physics rollouts 대비 약 2자릿수(roughly two orders of magnitude) 높게 나타났습니다. 또한 마찰 sweep에서 gait-collapse boundary를 넘어도 iKCE는 통계적으로 평탄하게 유지된 반면, 동일 구간에서 학습된 정책의 reward는 붕괴해 kinematic-not-dynamic 시그니처를 확인했으며, 이 구분은 embodiment의 보행 주기보다 긴 horizon에서 특히 드러났습니다.



### Delay-Aware Active Triangulation with Uncertainty-Driven Multi-Agent Reinforcement Learning for Counter-UAS (https://arxiv.org/abs/2607.05957)
Comments:
          Accepted to IROS 2026

- **Prior Approaches**: 기존 C-UAS용 능동 시각 삼각측량은 멀티뷰 기하를 바탕으로 하되, 다중 에이전트에서 발생하는 누적 지연(탐지-통신-의사결정 전파)을 충분히 모델링하지 않는 경우가 많았다. 또 많은 지연-aware RL 연구가 단일 에이전트 또는 제한된 지연 구조를 가정해, 에이전트 간 관측이 더 오래된 정보가 되는 비대칭 지연을 그대로 학습시키기 어렵다는 한계가 있었다. Perception-aware MPC 등은 가시성 계획을 다루지만 분산 통신 지연 자체를 확률적으로 다루는 방향은 상대적으로 약했다.

- **Core Contribution**: 이 논문은 Counter-UAS 시나리오에서 지연을 고려한 멀티에이전트 active visual triangulation을 위해, AoI(Age-of-Information)를 관측에 포함한 Dec-POMDP 기반 delay-aware RL 프레임워크를 제안한다. 또한 privileged reward(클린 상태 기반)와 perception-consistent reward(정책이 받는 지연·잡음 관측 기반)를 통제 비교해, 보상-관측 정렬이 성능과 안정성에 미치는 영향을 체계적으로 보여준다. 마지막으로 픽셀·포즈·짐벌 캘리브레이션·카메라 intrinsics까지 다중 소스 불확실성을 공분산 전파로 통합해, 불확실성 모델링의 누락이 성능을 어떻게 무너뜨리는지 정량화한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘지연된 비동기 관측’이 만드는 부분관측 문제를 학습 중에 어떻게 다루느냐와, 지연·잡음·기하 불확실성을 보상/학습 신호에 어떻게 정확히 반영하느냐였다. 저자는 AoI 태그를 메시지 수준으로 노출하고 GRU 기반 재귀 정책으로 시간적 불일치를 흡수하도록 설계했으며, dual-path reward 구조로 클린 상태 최적화가 왜 noise-fragile 구역을 유도할 수 있는지 관측했다. 더 나아가 단일(각도) 잡음만 쓰는 단순 모델을 넘어서 다중 소스 analytical covariance propagation을 도입해, 픽셀 탐지·포즈·짐벌·intrinsics 오차가 삼각측량 공분산에 미치는 영향을 분리해 반영했다.

- **Empirical Impact**: MAPPO를 4096개 병렬 환경에서 학습한 결과, perception-consistent 설정은 RMSE 0.547±0.217m, 삼각측량 유효도 78.1%를 달성했으며 track loss도 감소했다. AoI를 제거하면 삼각측량 유효도가 크게 떨어지고(유효도 +10.6%p 수준), MLP+프레임 스태킹은 지연의 비균일 타임스탬프 구조를 따라가지 못해 유효도가 거의 붕괴(0.7%)했다. 또한 다중 소스 공분산 모델은 angular-only 대비 RMSE를 2.8배 낮추며 유효도도 크게 개선해, 실제 C-UAS에서 필요한 불확실성 모델의 폭을 실증적으로 입증했다.



### Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2607.05939)
- **Prior Approaches**: 기존에는 민첩 드론 추격-도피 문제를 휴리스틱 규칙이나 단순 제어로 해결하는 접근이 많았고, 멀티에이전트 환경에서는 상대 정책 변화에 따른 비정상성 문제를 충분히 다루기 어려웠다. 또한 훈련 중 특정 상대 전략에 과도하게 맞춰져 재대결 시 성능이 떨어지는 catastrophic forgetting도 흔한 한계로 지적된다.

- **Core Contribution**: 이 논문은 포획 그물을 가진 팀이 민첩 드론을 가로채는 작업을 competitive Multi-Agent Reinforcement Learning(MARL) 문제로 정식화하고, 이를 위한 학습 프레임워크를 제안한다. 추격자( pursuers )와 도피자( evader )를 MAPPO에 기반해 함께 학습하되, Prioritized Fictitious Self Play(PFSP)를 결합해 다양한 상대 전략에 더 견고한 정책을 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 상대 정책이 학습 중 함께 변하면서 발생하는 nonstationarity, (2) 특정 상대에 과적합되어 나타나는 catastrophic forgetting을 동시에 완화하는 것이다. 논문은 MAPPO+PFSP로 상대 조합을 우선순위 기반으로 구성해 적응성을 높였고, 저수준 제어 커맨드와 collective thrust and body rates(CTBR) 같은 고정밀 입력을 사용해追-도피에서 성능을 내는 end-to-end에 가까운 학습이 가능하도록 설계했다.

- **Empirical Impact**: 고충실도 시뮬레이터에서 catch rate, time to catch, crash rate으로 평가한 결과, 제안한 정책이 휴리스틱 baseline 대비 더 높은 포획 성능을 보였다. ablation study에서는 PFSP가 다양한 상대 전략에 대한 robust policy를 만들고, 저수준 제어 커맨드가 pursuit-evasion에서 강한 전략 학습에 필수임을 확인했으며, 정성 분석에서는 추격자들 간 협력 전술의 emergence도 관찰된다.



### DexTele: A Dual-Arm Dexterous Teleoperation System Based on Motion Retargeting and Adaptive Force Contro (https://arxiv.org/abs/2607.05883)
- **Prior Approaches**: 기존 모션 리타겟팅은 인간-로봇이 짝지어진 paired 데이터로 지도학습 매핑을 하거나, 단일 로봇 플랫폼에 최적화된 kinematics 기반/ end-to-end 설계를 쓰는 경우가 많다. 이런 방식은 새 로봇을 추가할 때 데이터 재수집 부담이 커 cross-platform 일반화가 어렵고, 손 제어는 position 제어나 고정 force threshold 같은 방식이 많아 다양한 물체에 유연하게 대응하기 힘들다. 특히 잡기 단계의 적응성은 안전성과 직결되지만, 정교한 손-물체 상호작용 동역학을 요구하는 접근이 배포 난이도를 높여왔다.

- **Core Contribution**: DexTele은 듀얼암 덱스트러스 텔레오퍼레이션에서 cross-platform 모션 리타겟팅과 compliant grasping을 함께 해결하려는 통합 시스템이다. 비전 기반 모션 리타겟팅은 인간 관절 스켈레톤 그래프와 로봇 URDF 기반 위상을 motion graph로 보고, latent optimization을 통해 플랫폼이 달라도 정밀 리타겟팅이 되도록 설계했다. 또한 adaptive grasping은 VLM이 물체를 보고 목표 grasping force를 추정한 뒤, MPC 기반 온라인 최적화로 그 힘을 실제 제어 커맨드에 반영해 안전하고 안정적인 그립을 만든다.

- **Technical Challenges**: 문제는 (1) 로봇 아키텍처가 달라지는 상황에서도 인간 움직임을 정밀하게 변환해야 하고, (2) 물체 종류가 다양한 환경에서 적절한 힘을 실시간으로 맞추는 것이다. 저자들은 SAG-GCN의 dual-stream input–output 구조로 팔과 손을 분리 학습하되 중간 표현을 공유해 스케일/위상 차이를 완화하고, GRB로 attention·gated residual을 적용해 잡음 누적을 줄이며 정확도를 확보했다. 힘 적응에서는 VLM의 의미 기반 force prior와, joint angle–force 예측 surrogate(Random forest regressor) 및 gradient-based MPC-like 최적화를 결합해 목표 힘에 수렴하는 방식으로 온라인 보정을 구현했다.

- **Empirical Impact**: 실험에서 DexTele은 RMC-DA, YuMi, Unitree H1 등 여러 로봇 플랫폼에서 팔/손 리타겟팅 정확도와 동작 매끄러움(velocity/acceleration error) 지표가 기존 기준선을 전반적으로 앞섰다. 또한 손의 compliant grasping은 물체 8종에 대해 힘 편차와 진동 진폭이 10% 이내로 유지되고, adaptive grasping 모듈(AGM)을 넣었을 때 성공률이 평균 5.13→9.13(10회 기준)으로 크게 개선됐다. 실시간성도 확인되어 프레임당 텔레오퍼레이션이 약 10 FPS 수준(리타겟팅 모듈 프레임당 0.02s)으로 동작해, 실사용 관점의 일반화와 응답성을 동시에 보여줬다는 점에서 의미가 크다.



### GraspIT: A Dataset Bridging the Sim-to-Real gap and back for Validated Grasping SE(3) Pose Generation (https://arxiv.org/abs/2607.05869)
Comments:
          Preprint, release soon

- **Prior Approaches**: 기존 로보틱스 그리핑 데이터셋은 보통 RGB-D 관측, 그립 품질 라벨, 시뮬레이션-실세계 연결 중 일부만 제공해왔다. 그 결과 물리적으로 검증된 품질 기준과, 시뮬에서 생성한 후보를 실세계로 옮기는 원칙적 브리지가 동시에 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 GraspIT로, NVIDIA Isaac Sim에서 테이블탑 장면을 물리 기반 슬립 테스트 4단계로 주석해 연속 품질 점수와 궤적-도달성 체크를 함께 만든다고 제안한다. 또한 Real↔Sim 루프 백프로젝션으로 100개 실세계 장면에 라벨을 매핑해, 시뮬-실세계 일관성을 확보한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 시뮬에서 만든 후보 그립의 물리적 타당성을 품질 점수로 정량화하고, (2) 통과/실패를 넘어 실세계에서의 학습 신호로 안정적으로 전사하는 것이다. 논문은 병렬 Franka Panda 인스턴스 위에서 slip-test로 graded hard negatives를 구성하고, Real↔Sim 백프로젝션으로 연속 품질 라벨을 실세계 100장면에 대응시켜 이 문제를 해결한다.

- **Empirical Impact**: GraspIT는 약 2.3M 후보 중 83%를 good(s≥0.50)으로 만들고, 남은 17%는 force-closure는 통과하되 slip-test에서 실패한 graded hard negatives로 활용한다. 최종 공개물은 약 316k annotated RGB-D 프레임 세트(시뮬 1035, 실세계 100)와 instance masks, 6-DoF pose, 물리 객체 속성, 점수화된 6-DoF grasp를 포함하며, Docker 및 오픈소스로 공개돼 tabletop manipulation 정책 학습과 behavior cloning에 바로 쓰일 수 있다.



### FORGE: Towards Functional Tool-Use Generalization via Keypoint Trajectory Reasoning (https://arxiv.org/abs/2607.05780)
Comments:
          15 pages, 8 figures, 6 tables

- **Prior Approaches**: 로보틱스의 기존 일반화 연구는 주로 장면/카테고리 수준의 시각 변동이나, 또는 cross-embodiment처럼 로봇 형태를 바꾸는 문제에 초점이 맞춰져 있었다. 반면 도구 사용에서 ‘동일한 기능’을 보장하는 functional generalization은, 도구 모양은 달라도 접촉 지점과 모션이 바뀌어야 한다는 점에서 perception-to-action gap이 커 end-to-end 학습이 쉽게 무너진다.

- **Core Contribution**: 논문은 functional generalization을 ‘도구는 바뀌지만 기능(타격)을 동일하게 수행’하는 문제로 정식화하고, 핵심 난제로 시각 유사성이 행동 공간으로 그대로 전이되지 않는 불일치를 제시한다. 이를 해결하기 위해 FORGE를 제안하며, 기능 추론과 동작 실행을 분리해 먼저 일반화 가능한 2D keypoint trajectories를 예측한 뒤, 제한된 시연으로 로봇 행동에 grounding한다.

- **Technical Challenges**: 기여를 위해서는 (1) 도구별 외형에 과적합되지 않으면서도 (2) 접촉 지점과 시간에 따른 운동 구조를 담는 intermediate representation을 찾아야 했다. 저자들은 affordance images, human video prompts, 2D keypoint trajectories를 비교했고, keypoint trajectories가 function 표현력과 action groundability를 가장 잘 균형한다는 실험 결과로 선택했으며, stage1은 action-free data로 conditional flow matching 예측, stage2는 action-labeled data로 conditional flow matching 기반 execution policy 학습(그리고 예측 오차 견고화용 perturbation)으로 구성했다.

- **Empirical Impact**: 일곱 가지 도구의 hitting-function 벤치마크에서 FORGE는 unseen tools에 대해 state-of-the-art 대비 평균 success rate를 2배 이상(2X+) 끌어올리며 시뮬레이션과 real world 모두에서 일관된 성능을 보였다. 특히 end-to-end visuomotor 정책들은 unseen 도구의 올바른 hitting region 정렬에 실패하지만, FORGE는 keypoint 기반 중간 계획이 접촉 위치 접근을 구체적으로 안내해 실패를 줄이는 방식으로 의미를 입증한다.



### Observation Quality Matters: Robust Multi-Fisheye Calibration via Failure-Oriented Analysis (https://arxiv.org/abs/2607.05777)
Comments:
          9 pages, 7 figures, 6 tables. Code: this https URL

- **Prior Approaches**: 기존 multi-fisheye 캘리브레이션은 Kalibr처럼 bundle-adjustment(BA) 비선형 최적화를 기반으로 intrinsics, per-view target pose, inter-camera extrinsics를 단계적으로 초기화·정제하는 파이프라인이 주류였다. 이후 distortion이 심한 환경에서 intrinsics 추정을 강화하거나, 더 풍부한 calibration target으로 기하 제약을 늘리는 연구들이 나왔지만, 관측(observation)의 ‘품질’은 여전히 경험적으로 다뤄졌다. 특히 어떤 관측 조건에서 최적화가 잘 풀리는지에 대한 원인 분석은 충분히 정리되지 않았다.

- **Core Contribution**: 이 논문은 실패 지향 분석을 통해 캘리브레이션 실패가 detector recall 저하나 전역 image-plane 분포 불균형만으로는 설명되지 않는다고 밝힌다. 핵심 원인은 intrinsic initialization이 ill-conditioned 상태가 되기 쉬운지 여부이며, focal scale과 fisheye projection-shape 파라미터가 충분히 분리되지 못하면 선형화 업데이트가 불안정해진다. 또한 focal–projection coupling을 유발하는 관측 특성(특히 좁은 radial span)을 찾아 이를 개선하는 데이터 구성 프레임워크 CO-Calib을 제안한다.

- **Technical Challenges**: 관측 품질을 개선하려면 단순히 더 많은 코너를 잡는 것만으로는 부족하고, 초기화 경로에서 파라미터 방향을 분리해 조건을 좋게 만드는 프레임 순서를 설계해야 한다. 이를 위해 CO-Calib은 (1) 왜곡이 큰 영역에서도 안정적인 검출을 제공하는 robust learning-based target detector와, (2) 실패 메커니즘에 맞춘 error-analysis-guided frame selector를 결합한다. 셀렉터는 projective isotropy와 directed radial span 같은 기하 기반 지표로 anchor(초기화 안정 프레임)·co-visible(멀티카메라 제약 유지 프레임)·mono-fill(약한 영역 보강 프레임) 단계를 구성해 기존 BA/최적화 백엔드를 바꾸지 않고 관측을 ‘optimization-ready’로 만든다.

- **Empirical Impact**: 합성 및 실세계 multi-fisheye 실험에서 CO-Calib은 전체 성공률을 68.1%에서 99.3%로 끌어올렸고, extrinsic 정확도(회전/이동)도 개선하며 real-world에서 캘리브레이션 안정성을 높였다. 또한 분석 결과를 검증하듯, 초기화의 ill-conditioning을 우회해 최종 joint optimization만으로 만회하려 하면 성공률이 급격히 떨어져 초기화 경로의 중요성이 실증된다. 코드 공개 예정이며 Hex-Fisheye처럼 어려운 구성에서도 성능 향상을 보여 해당 분야의 운영 안정성에 직접적인 의미가 있다.



### Co-STAR: Cognitive Stimulation Therapy by an Autonomous Robot for Dementia -- A One-Week In-Home Study (https://arxiv.org/abs/2607.05709)
Comments:
          Accepted for publication at the IEEE RO-MAN Conference 2026

- **Prior Approaches**: 치매 인지치료에서 핵심 근거로는 CST와 그 개인화 버전인 iCST가 널리 인정돼 왔지만, 가정 내 수행은 전문인력 부족과 보호자(비공식 돌봄자)의 시간·훈련 부담 때문에 흔들립니다. 특히 iCST는 처방 대비 주 3회 요구에도 실제로는 40% 정도만 일정을 맞추는 등 ‘효능-현장 격차’가 지속돼 왔습니다. 로봇 기반 접근도 존재했지만 대체로 게임·리마인더·대화처럼 일반적 자극에 머물거나, 요양시설/낮병원 중심이어서 가정 내 근거 기반 치료의 자율 전달은 부족했습니다.

- **Core Contribution**: 이 논문은 가정에서 자율적으로 CST(iCST)를 제공하는 socially assistive robot을 실제 배치해 타당성을 검증합니다. 9명의 치매 당사자를 대상으로 1주(7일) 동안 매일 로봇 주도 세션을 진행했고, 참여자들이 비교적 높은 순응도로 세션을 수행했습니다. 또한 가족 구성원이 세션 시작을 돕고 때때로 함께 참여하면서 상호작용의 질을 높이는 역할이 관찰돼, 기술만이 아니라 가정 맥락 설계의 중요성을 강조합니다.

- **Technical Challenges**: 기여를 현실화하기 위해서는 (1) 치매 화자의 발화 특성을 반영한 음성 인식, (2) 세션을 끊김 없이 운영하는 스케줄링·데이터 흐름, (3) 프라이버시를 해치지 않는 설계, (4) 개인차(흥미·난이도·기술 숙련·억양 선호)를 아우르는 개인화가 필요했습니다. 연구팀은 로컬 처리 중심으로 동작하며, speech-to-text는 Whisper의 fine-tuned 변형(WhisperD)을 사용해 채움말·발화 지연 같은 disfluencies를 처리하도록 했습니다. 동시에 5가지 iCST 활동을 tablet 시각 프롬프트와 음성 대화로 구성하고, 사전 개인 정보를 기반으로 개인화 내러티브와 활동을 구성해 참여를 유지하려고 했습니다.

- **Empirical Impact**: 1주 동안 총 31회의 세션이 수행됐고 평균 1인당 약 3.4회로, 안내된 기대치 대비 절반 수준의 ‘현장형’ 순응도가 확인됐습니다. 참가자들은 세션이 기억을 환기시키고 재미·도전감을 제공한다는 반응을 보였으며, caregiver가 있을 때 순응이 더 높게 나타났습니다. 반면 스크립트된 질문에 대한 ‘기계적 반응’ 인상, 느린 초기 구동/응답, 기술적 실패(정전·인터넷 불안정), 태블릿 조작 난이도, 그리고 미국식 억양 같은 요인이 이탈 요인으로 지적돼 가정용 자율 치료 로봇의 개선 방향도 함께 제시합니다.



### IMR: Iterative Mode-World Weighted Regression for Multi-Agent Trajectory Prediction (https://arxiv.org/abs/2607.05705)
- **Prior Approaches**: 기존 multi-agent motion prediction은 예측 기반과 anchor 기반으로 크게 나뉜다. 예측 기반(QCNet, Forecast-MAE)은 복잡한 상황에서 mode collapse(다양성 붕괴)가 발생하기 쉽고, anchor 기반(MTR, TNT)은 이를 완화하는 대신 예측 정확도가 떨어지는 경향이 있다. 또한 QCNeXt 같은 proposal-refinement 디코딩은 초기 제안 궤적이 크게 틀리면 refinement가 오프셋 보정을 충분히 못 하는 문제가 지적된다.

- **Core Contribution**: 이 논문은 mode collapse와 정확도 저하의 trade-off를 동시에 다루기 위해 prediction-based 프레임워크에 mode-world weighted regression loss를 제안한다. 이 손실은 mode-wise 및 world-wise 회귀를 함께 가중해 학습하며, mode 다양성 유지와 함께 world ranking 정확도 및 top-1 confidence를 개선하는 데 초점을 둔다. 아울러 반복적 디코딩으로 제안 궤적의 초기 오류에 덜 민감한 구조를 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다중 에이전트 궤적에서 정답 모드가 여러 개인데도 학습 중 특정 모드로 수렴해버리는 문제와 (2) joint world confidence를 포함한 world ranking을 정확히 학습하는 문제다. 이를 위해 그래프 attention network 기반으로 에이전트-맵 및 상호작용을 동적으로 인코딩하고, 모드 회귀는 winner-takes-all 전략으로 선택 오차(ADE/MDE)를 기준화해 손실을 설계한다. 디코딩은 iterative decoder를 도입해 SS(세그먼트) 단계를 순차 생성하되, 각 단계 출력이 이전 단계 offset이 아니라 절대 위치 좌표가 되도록 하여 누적 오차 전파를 줄인다.

- **Empirical Impact**: 실험은 Argoverse 2에서 6초 예측 setting으로 수행됐고, 제안 방법은 다른 모델 대비 평균 BrierMinFDE6에서 이전 SOTA QCNeXt보다 0.06p 향상되며 1위를 기록했다. 단일 에이전트 벤치마크에서도 LOF와 비교해 경쟁력 있는 성능을 보였다. 시각화와 손실 비교 결과, 기존 world-wise 회귀에서 나타나는 mode collapse를 mode-world weighted regression loss가 완화하면서 정확도까지 함께 개선함을 확인했다.



### Uncertainty-Aware Velocity Correction for Proprioceptive Vehicle Localization using Evidential Mamba (https://arxiv.org/abs/2607.05669)
Comments:
          Accepted at the 2026 International Conference on Indoor Positioning and Indoor Navigation (IPIN 2026), Rome, Italy. 6 pages, 4 figures

- **Prior Approaches**: GNSS-denied 환경에서 기존 INS/IMU 항법은 외부 보정이 없으면 드리프트가 누적되며 시간이 지날수록 오차가 빠르게 커진다. 이를 줄이기 위해 RF 인프라, 카메라·LiDAR 같은 시각 센서, 혹은 멀티센서 융합을 쓰는 방법이 있으나 설치·비용·운용 제약이 크다. 다른 한편으로 wheel 속도·조향각 같은 온보드 신호를 이용한 virtual velocity 보정은 가능하지만, 현실 주행에서 가정이 자주 깨져 장기 outage에서 성능이 제한된다.

- **Core Contribution**: 이 논문은 Evidential Velocity Correction using Mamba(EVC-Mamba)를 제안해, 별도 하드웨어 없이 온보드 센서 데이터를 “가상 velocity 센서”로 변환해 IMU 드리프트를 보정하는 방법을 제시한다. Mamba 기반 selective State Space Model이 시간적 운동 역학을 효율적으로 포착하고, evidential deep learning(Normal-Inverse-Gamma)으로 속도 추정치와 불확실성을 단일 forward에서 함께 제공한다. 이후 ES-EKF의 virtual measurement로 불확실성까지 고려해 위치 드리프트를 억제한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 온보드 센서만으로 비선형 차량 동역학·노이즈를 반영한 정확한 속도를 추정하고 (2) 그 불확실성을 신뢰도 있게 정량화하며 (3) 실시간 상태추정 필터에 통합하는 것이다. 논문은 Mamba의 selective SSM으로 긴 시퀀스 의존성을 선형 시간에 모델링하고, evidential regression(NIG)을 통해 MC Dropout/앙상블 없이도 NLL과 불확실성 정규화로 캘리브레이션을 맞춘다. 마지막으로 평면 속도와 zero vertical velocity 제약을 포함해 ES-EKF가 가상 관측으로 안정적으로 업데이트하도록 설계했다.

- **Empirical Impact**: ReV-StED의 실제 차량 데이터로 평가했을 때, EVC-Mamba의 속도 추정은 다른 온보드 기반 방법 및 transformer 대비 특히 횡방향에서 약 30% 개선을 보였다. GNSS outage 동안 최대 위치 드리프트는 속도 보정이 없는 기준 대비 크게 감소하며, 외부 전용 velocity 센서(Correvit)에 근접해 모든 outage 지속시간에서 그 성능의 약 10% 이내 수준을 달성했다. 또한 NVIDIA Orin에서 40Hz 운용이 가능한 추론 지연(20~24ms)과 낮은 연산량을 제시해 엣지 실시간 배치 가능성을 뒷받침한다.



### Efficient Transfer Learning of Robot Dynamic Models Using Morphological Similarity (https://arxiv.org/abs/2607.05665)
Comments:
          Accepted for publication in the 2026 12th International Conference on Control, Decision and Information Technologies (CoDIT)

- **Prior Approaches**: 연구진은 연성 수중 로봇의 운동/동역학 모델링이 유체-구조 비선형 상호작용 때문에 어렵고, 기존에는 고가 센서 피드백이나 데이터 라벨 수집에 의존하는 학습 모델이 많다고 짚었습니다. 또 transfer learning도 주로 제어 정책이나 인식(예: 비전) 쪽에 집중돼 동역학의 cross-robot 전이는 상대적으로 덜 다뤄졌다고 평가합니다. 기존 방식의 핵심 한계는 목표 로봇(target)에서 라벨 데이터를 구하기 어렵다는 점과, 크기·유체 조건 차이로 생기는 domain gap을 충분히 메우지 못한다는 점입니다.

- **Core Contribution**: 이 논문은 형태적으로 유사하지만 스케일과 수력학 특성이 다른 수중 로봇 간 동역학을 라벨 없이 전이하는 신경망 프레임워크를 제안합니다. U-CAT(소스)에서 학습한 모델을 Micro-CAT(타깃)로 옮기되, autoencoder 기반 domain adaptation으로 두 로봇의 동역학을 공유 latent representation 안에 정렬합니다. 그 결과 타깃 로봇에서 labeled data 없이도 body-frame velocities 같은 상태를 정확히 추정할 수 있게 했다고 밝힙니다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 연성 핀 구동 로봇의 시간영역 동역학이 잡음과 비선형성을 동반하고, (2) 크기 차이로 인해 source와 target의 분포가 달라지는 domain gap을 라벨 효율적으로 줄여야 한다는 점입니다. 연구진은 encoder가 센서+구동 입력을 latent로 압축하고, reconstruction과 next-latent dynamics 예측을 함께 학습한 뒤 MMD(Maximum Mean Discrepancy)로 두 도메인의 latent 분포가 겹치도록 최적화합니다. encoder 공유와 도메인별 decoder를 두는 구조로 재구성 품질을 유지하면서도 정렬을 강화한 점이 해결책의 중심입니다.

- **Empirical Impact**: 두 대의 실제 fin-actuated 수중 로봇(U-CAT→Micro-CAT)에서 실험했으며, MMD로 latent 정렬이 유의미하게 이루어지는 것을 PCA 시각화로 확인했다고 보고합니다. 예측 평가는 Vx, Vy의 선속도와 yaw 성분 같은 핵심 velocity에 대해 RMSE/MAE로 수행했고, 제안 방법은 target 라벨 없이도 baseline 대비 개선을 보이며 특히 Vx에서 zero-shot 성능이 RMSE 기준 약 40% 향상됐다고 합니다. 또한 joint head를 사용한 경우 fully supervised 기준과 비슷한 수준에 도달해, 라벨이 부족한 수중 환경에서 cross-robot dynamics transfer의 실용성을 입증했다는 점에서 의미가 큽니다.



### Physics-Regularized Machine Learning for Proprioceptive Vehicle Localization Using Onboard Sensors (https://arxiv.org/abs/2607.05663)
Comments:
          Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026). 8 pages, 4 figures

- **Prior Approaches**: 기존 GNSS-denied 환경의 차량 위치추정은 HD map 기반, 다중센서 융합, dead reckoning 계열로 나뉜다. 특히 IMU 중심의 고전적인 Bayesian 필터는 노이즈 누적으로 시간이 지날수록 드리프트가 커지는 한계가 있고, ML 보정 방식은 센서 잡음은 잘 다루더라도 물리적 일관성과 조건 전반 일반화가 부족해질 수 있다. 최근엔 differentiable filtering이 등장했지만, 표준 onboard proprioceptive 센서만으로 실시간 localization을 달성하면서 물리-시간 정합성을 체계적으로 분석/강화한 접근은 상대적으로 적었다.

- **Core Contribution**: 이 논문은 onboard 센서만으로 차량 포즈를 추정하는 Physics-Regularized Machine Learning for Localization(PRML2) 프레임워크를 제안한다. PRML2는 transformer 기반 ML이 차량 동역학 상태를 예측하고, differentiable EKF가 이를 포즈 추정으로 결합하되 EKF를 통한 end-to-end 학습이 물리 기반 정규화(physics-regularized learning) 역할을 하도록 설계했다. 결과적으로 예측이 차량 운동모델과 시간적으로 모순되지 않게 만들어 정확도와 주행 조건 전반 일반화를 함께 노린다.

- **Technical Challenges**: 핵심 난제는 (1) ML이 만드는 동역학 예측이 차량 운동 제약을 위반하지 않도록 하면서 (2) EKF 학습에서 수치적으로 안정적인 미분 가능 파이프라인을 유지하는 것이다. 이를 위해 PRML2는 physics guard layer로 속도·가속도·각속도 등 동역학 한계를 클램핑하고, 불확실성 추정을 위한 variance까지 일관되게 스케일링한다. 또한 differentiable EKF에서 Jacobian 및 수치 불안정(혁신 공분산 역행렬 등)을 Cholesky 분해와 Joseph form 같은 안정화로 완화해 그래디언트가 ML 모델로 안정적으로 전파되게 했다.

- **Empirical Impact**: 저자들은 publicly available 데이터셋에서 ML-enhanced onboard odometry의 성능 상한을 분석하고, PRML2가 localization 정확도에서 우수하며 실시간 실행도 가능함을 보인다. 더 나아가 저마찰(low-friction) 주행 조건을 위한 새로운 데이터셋을 공개해 해당 도메인의 일반화 평가 기반을 확장했다. 전반적으로 PRML2는 GNSS가 불안정하거나 끊기는 상황에서도 저비용 onboard 센서만으로 강건한 위치추정을 지향하며, learning+physics priors 결합의 실용성을 실험적으로 뒷받침한다.



### Dynamic Evaluation of Classical and Control-Aware Optimal Trajectory Planning in Robot Manipulators (https://arxiv.org/abs/2607.05544)
Comments:
          Accepted at MERCon 2026. To be presented at MERCon 2026. 7 pages, 7 figures

- **Prior Approaches**: 기존에는 cubic, quintic, trapezoidal 같은 고전 궤적 생성이 단순함과 매끈함 때문에 널리 쓰이지만, 기본적으로 kinematic 관점에 머물러 로봇의 dynamics나 actuator effort를 궤적 생성 단계에서 직접 반영하지 않는다. 그래서 겉보기엔 매끈해도 실제 nonlinear 실행(피드백 제어+강체 동역학)에서는 가속 분포가 불리해져 보정 토크와 제어 비용이 커질 수 있다. 또한 비교 연구들이 서로 다른 제어기/시뮬레이션 조건을 써 공정 비교가 어려웠다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 manipulator dynamics와 actuator effort를 명시적으로 포함하는 finite-horizon 기반 control-aware optimal trajectory planning 프레임워크를 제안한다. 큰 point-to-point 이동에서 근사 정확도를 높이기 위해 midpoint linearization 전략을 도입하며, 고전 궤적 생성기와의 비교가 특정 제어 조건 차이 없이 “궤적 생성” 자체의 영향만 드러나도록 통일된 nonlinear 평가 프레임워크를 구축한다. 즉, 동일한 nonlinear 실행 환경에서 궤적 생성 품질을 추적 오차와 제어 비용으로 직접 검증한다.

- **Technical Challenges**: 핵심 기술 과제는 nonlinear dynamics와 actuator 노력(보정 토크)을 포함한 최적화를 오프라인 궤적 생성으로 구현하되, 정확도와 계산 복잡도를 함께 다루는 데 있다. 이를 위해 dynamics를 midpoint에서 선형화해 유한수평 예측모델을 만들고, state deviation과 actuator effort를 모두 페널티로 넣는 조절 문제(제약 포함)로 정식화해 quadratic program 형태로 풀어 최적 제어 시퀀스를 산출한다. 더 나아가 모든 비교 방법이 동일한 PID 기반 비선형 closed-loop 실행(동일 제어 구조/이득/제약/비선형 RK4)에서 수행되도록 설계했다.

- **Empirical Impact**: 비선형 3-DoF UR5 시뮬레이션에서 제안 방법은 RMS tracking error를 cubic 대비 약 22%, quintic 대비 50%, trapezoidal 대비 73% 줄였다. 보정 토크(RMS corrective torque)와 누적 corrective-control activity도 각각 약 28–41% 감소했으며, 누적 executed cost는 cubic 대비 약 48%, quintic·trapezoidal 대비 62% 이상 절감됐다. 결과적으로 kinematic smoothness만으로는 dynamically efficient execution이 보장되지 않으며, control-aware optimal 궤적이 actuator 부담과 실행 비용을 동시에 낮출 수 있음을 실증해 로보틱스 궤적 설계 관점에 영향을 줄 것으로 보인다.



### GEM-Occ: From Visual Geometry Evidence to Embodied Semantic Occupancy Memory (https://arxiv.org/abs/2607.05543)
Comments:
          19 pages, 6 figures. Project page: this https URL

- **Prior Approaches**: 기존 실내 occupancy 연구는 주로 단일 뷰에서의 의미론적 점유 예측(semantic completion)이나 특정 실내 스케일의 room-level 추론에 머물렀습니다. 또한 데이터셋과 관측 형식(perspective RGB-D, pano)도 통합되지 않아, 긴 지평선의 인과적(causal) 의미 매핑과 free/unknown 구분, 재방문 안정성 같은 요구를 체계적으로 평가하기 어려웠습니다.

- **Core Contribution**: 이 논문은 ScanNet, ScanNet++, Matterport3D를 하나의 sparse semantic occupancy 포맷으로 통합하면서도 각 데이터의 native 관측 기하를 보존하는 HIOcc를 제안합니다. HIOcc는 local semantic occupancy prediction, room-level online occupancy mapping, building-level panoramic mapping의 3가지 평가 레짐을 제공해 긴 지평선 embodied mapping을 정량화합니다. 또한 GEM-Occ을 통해 관측에서 나온 순간적 증거를 영속 메모리로 변환하는 프레임워크를 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 점유/자유공간/미지(unknown)와 의미를 갖는 “영속적 지도”를 만들기 위해, 순간 관측 evidence를 중복·부정합 없이 누적하는 것입니다. GEM-Occ은 pointmap을 그대로 지도 상태로 축적하지 않고, 로컬 시각 기하 예측을 semantic Gaussian occupancy evidence와 free-space ray evidence로 변환한 뒤, visibility-와 uncertainty-aware causal update로 계층형 메모리에 융합합니다. 메모리는 local cache, room-level submap, building-level graph로 구성되며 Gaussian-to-occupancy splatting으로 언제든 질의할 수 있습니다.

- **Empirical Impact**: HIOcc 실험에서 GEM-Occ은 local occupancy 정확도뿐 아니라 온라인 맵 안정성, free-space reasoning, revisit consistency, building-level 확장성에서 이전 indoor occupancy 및 Gaussian 기반 baseline을 전반적으로 개선했습니다. 특히 room-level에서는 mIoU와 IoU가 상승했고, building-level panoramic 설정에서는 progress AUC와 revisit consistency까지 더 좋아졌습니다. ablation 결과로 semantic Gaussian evidence와 free-space ray evidence의 분리 및 confidence-aware 융합이 성능 향상의 핵심임이 확인됐습니다.



### Learning 4D Geometric Priors for Inference-Efficient World Action Models (https://arxiv.org/abs/2607.05468)
Comments:
          9 pages, 6 figures

- **Prior Approaches**: 기존 World Action Model(WAM)은 비디오 미래 예측과 액션 시퀀스를 함께 학습하지만, 많은 방법이 외형 중심(appearance-oriented) 비디오 잠재표현 최적화에 치우쳐 조작에 필요한 시공간 기하 관계를 충분히 담지 못한다. 3D/4D 구조를 넣더라도 배치 시점에 기하를 출력하거나(추가 디코더/출력), 일반적인 기하 감독을 써서 현재 액션과 인과적으로 연결된 관계를 잘 구분하지 못하는 한계가 있었다. 또한 기하 분기가 액션 생성 경로로 무제한 정보를 흘려 넣으면 비인과적 shortcut으로 학습이 편향될 위험도 지적된다.

- **Core Contribution**: MECo-WAM은 추론(inference) 그래프는 그대로 두고, 학습(training) 중에만 4D 기하 priors를 주입해 액션에 필요한 ‘시간 변화하는 기하’를 비디오-액션 표현에 이식하는 Multi-Expert Co-Training WAM을 제안한다. 학습 시에는 영상/액션 expert 외에 training-only 4D expert를 추가해 frozen VGGT 인코더의 관계형(relational) 타깃으로 시간적 기하를 감독한다. 핵심은 deploy 단계에서는 보조 4D 구성요소를 완전히 제거해 추가 비용 없이 조작 성능을 끌어올리는 것이다.

- **Technical Challenges**: 가장 큰 과제는 4D 기하 정보를 학습에만 쓰되, 액션 생성 경로로의 정보 누수를 막아 비인과적 shortcut을 방지하면서도 시간적으로 진화하는 관계를 제대로 전이하는 것이다. 이를 위해 MECo-WAM은 decayed 4D read-mask attention으로 현재 프레임 기하 토큰은 초기 학습에서만 제한적으로 읽히게 하고, 최종적으로는 해당 접근을 단계적으로 제거해 배치 시 의존성을 끊는다. 더불어 action-aware temporal geometric distillation을 통해 프레임 내 관계뿐 아니라 키프레임 간 관계 변화까지, 로봇 액션과 연관된 시각 토큰에 가중치를 둬 정렬하도록 설계했다.

- **Empirical Impact**: 실험에서 MECo-WAM은 LIBERO 98.2%, RoboTwin 2.0 92.6%, 그리고 ARX-R5 기반 실세계 태스크에서도 일관된 조작 성능 향상을 보이며 추론 비용(동일 lightweight video-action 그래프) 증가는 없었다. 특히 geometry-sensitive 계열에서 개선 폭이 커, ‘도달/정렬/접촉-전이’처럼 기하에 민감한 추론이 강화됐음을 시사한다. 실세계에서는 보정 횟수 감소와 완료 시간 단축까지 함께 나타나, 학습 시점 4D 기하 전이가 시뮬레이션을 넘어 로봇 그라운딩에 효과적임을 보여준다.



### Quaternion-Averaging-Based Adaptive Complementary Filter for Pedestrian Dead Reckoning With a Foot-Mounted AHRS (https://arxiv.org/abs/2607.05451)
- **Prior Approaches**: 기존 Pedestrian Dead Reckoning(PDR)은 가속도와 자이로 드리프트 누적으로 오차가 누적되며, 특히 AHRS 기반은 자기장 교란의 영향을 크게 받는다. 자세 추정의 대표 방식으로는 칼만 필터(KF)가 있지만 행렬 연산(역행렬/곱셈) 때문에 계산 비용이 높고, 보완 필터(CF)는 상대적으로 가볍지만 동적 조건에서 헤딩 정확도가 떨어질 수 있다는 한계가 있었다. 쿼터니언 융합에서도 LERP/SLERP 같은 보간에 의존해 ±q 부호 불일치 등 자세 동치성을 엄밀히 다루기 어렵다는 점이 지적된다.

- **Core Contribution**: 논문은 발목(mounted) AHRS를 쓰는 PDR에서 자세 추정 정확도를 높이면서도 연산량을 줄이기 위해 Quaternion-Averaging-Based Adaptive Complementary Filter(QAACF)를 제안한다. QAACF는 각속도로부터 얻은 쿼터니언과 가속도·자기장 측정으로부터 얻은 쿼터니언을 Markley의 quaternion averaging으로 융합해, 기존 선형 보간보다 더 엄밀하게 두 쿼터니언을 결합한다. 또한 보행 단계와 자기장 교란 수준에 따라 각 센서(각속도·가속도·자기장)의 가중치를 적응적으로 조절해 AHRS 오차 요인을 상쇄한다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 보완 필터 구조에서 쿼터니언 융합을 수학적으로 더 타당하게 만들면서 (2) 보행 중 신뢰도 변화와 자기장 교란을 가중치로 반영해 전체 자세 추정을 안정화하는 것이다. QAACF는 Markley 방식의 고유값/고유벡터 기반 closed-form 융합으로 ±q 동치 문제를 포함한 쿼터니언 융합의 엄밀성을 확보하고, midstance(발이 거의 정지) 구간에서는 중력/가속도 기반 자세의 신뢰도를 높이며 그 외 구간 및 교란 상황에서는 자기장 기반 항의 가중치를 낮춘다. 여기에 PDR 오차 누적을 줄이기 위해 ZVU(Zero Velocity Update)로 속도 드리프트를 보정하며, 궤적은 전역 좌표계에서 가속도를 적분해 계산한다.

- **Empirical Impact**: 실험은 실내용 보행 데이터(OptiTrack 기반 지상진실과 MTW2-3A7G6 AHRS 데이터)를 활용해 QAACF의 자세 추정 성능을 Euler angle RMSE로 비교하고, 기존 KF/CF들과 계산 비용을 서로 다른 환경(고성능/저비용)에서 대조한다. 결과적으로 QAACF는 기존 자세 추정 필터 대비 낮은 RMSE를 보이면서도 칼만 필터보다 더 낮은 계산 비용을 요구한다. 나아가 QAACF로 구한 자세를 사용한 PDR 궤적 정확도도 비교 대상 알고리즘보다 개선되는 것으로 정리되며, 자기장 교란이 있는 실내 네비게이션에서 실용적인 저비용·고정확도 대안임을 시사한다.



### Driving the Wrong Way: Leveraging Interpretability in End2End Autonomous Driving Models (https://arxiv.org/abs/2607.06328)
- **Prior Approaches**: 기존 end-to-end 자율주행은 perception·prediction·planning을 하나의 transformer로 통합해 NAVSIM 같은 open-loop 벤치마크에서 강한 성능을 보이지만, 모듈 경계가 사라져 내부 의사결정이 불투명해진다. 이에 따라 saliency/gradient 기반 설명이나 attention visualization은 입력의 “어디를 봤는지”는 보여주지만, 모델이 내부에서 “무엇을 개념으로 학습했는지”와 실패 원인의 기능적 관련성까지는 잘 드러내지 못한다. latent space 쪽 해석도 드물고, end-to-end에서 개념 수준으로 분해·인과 확인을 함께 하는 방식은 거의 없었다.

- **Core Contribution**: 이 논문은 Sparse Autoencoder(SAE) 기반 dictionary learning을 end-to-end 주행 모델의 feature space에 사후(post hoc) 해석 모듈로 결합해, 잠재표현을 의미 있는 희소 개념(semantic concepts) 조합으로 분해한다. 각 개념을 자연어로 일관되게 대응시키고, 후보 궤적의 점수(trajectory-level decision scores)에 어떤 개념이 기여하는지 직접 연결해 “결정 로직”을 노출한다. 또한 개념 단위 개입(intervention)으로 특정 개념을 억제/조작해 주행 의사결정을 수정할 수 있음을 제시한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) end-to-end 모델 내부에서 SAE를 주입할 적절한 latent 공간을 골라야 하고, (2) 얽힌(polysemantic) 뉴런을 monosemantic 개념 방향으로 희소 분해해야 하며, (3) 개념과 궤적 점수·하위 품질 지표 사이의 인과적 연결을 계산 비용 부담 없이 구현해야 한다. 이들은 점수 모듈 직전의 1D per-trajectory 표현을 SAE 입력으로 택하고, top-k 희소화와 dead neuron 방지(reanimation)로 재구성과 개념 유효성을 함께 확보한다. 이후 Concept Relevance Propagation(CRP)과 attribution 기반 circuit analysis를 조합해, 개념이 각 scoring head/PDM 성분에 주는 영향의 희소 회로를 구성하고 상위 후보에 대해 정확 개입으로 인과를 확인한다.

- **Empirical Impact**: 실험에서는 GTRS와 iPAD 모델의 latent 위에 다양한 SAE 아키텍처/하이퍼파라미터를 학습한 뒤 재구성 품질(cosine similarity, explained variance)과 문제 특화 정합성(ego correlation, ego probing), 그리고 EPDMS 같은 downstream 성능을 함께 비교한다. 결과적으로 TopK·Matryoshka·archetypal SAE 변형들 사이에서 dead neuron 활용도와 ego 관련 분해가 달라지며, 아키텍처 선택이 해석 가능성과 성능 보존의 균형을 좌우함을 보여준다. 더 나아가 개념 수준 개입이 충돌 회피, drivable area, traffic light compliance 같은 하류 주행 지표를 계량적으로 개선해, 설명이 단순 상관이 아니라 기능적·수정 가능한 구성요소임을 뒷받침한다.



### Why does Deep Learning Improve Visual SLAM? (https://arxiv.org/abs/2607.06023)
- **Prior Approaches**: 기존 V-SLAM은 크게 기하 기반(frontend–backend)으로 나뉘며, 특징 기반은 수제 keypoint/descriptor 매칭, 직접법은 픽셀 강도 일관성을 전제로 한다. 이런 전통적 방식은 저텍스처, 심한 모션 블러, 조도 변화 같은 실제 환경에서 취약해지고, end-to-end 회귀 방식은 학습 분포 밖 일반화가 약하다는 문제가 있었다. 최근 딥러닝 V-SLAM은 learned 2D data association과 uncertainty를 이용하고, differentiable geometric optimization을 recurrent 구조로 반복 적용해 성능을 끌어올렸지만, 성공 요인이 어떤 구성요소에 있는지는 불명확했다.

- **Core Contribution**: 이 논문은 딥러닝 기반 V-SLAM의 성능이 (1) learned 2D data association 단독, (2) 여기에 uncertainty 결합, (3) recurrent architecture 자체 중 무엇 때문인지 질문한다. 이를 위해 기하 기반 대표 시스템인 ORB-SLAM3의 수제 디스크립터 매칭 모듈만 optical flow 기반 대응 추정으로 교체해 ORB-SLAM3-OF를 만들고, 여기에 uncertainty로 bundle adjustment의 잔차 가중치를 적용한 ORB-SLAM3-OF-U를 추가한다. 그리고 같은 SLAM 파이프라인을 유지한 채 두 구성요소의 기여를 직접 분리·정량화한다.

- **Technical Challenges**: 핵심 과제는 learning-based 대응(광류)과 불확실성을 고전 SLAM의 동일 최적화 틀에 “방법론적으로 일관되게” 끼워 넣는 것이었다. 저자들은 ORB-SLAM3 내부에서 추적/현지화의 매칭을 광류로 예측된 2D 위치 주변에서 feature를 찾는 방식으로 대체하고, ORB-SLAM3-OF에서는 bundle adjustment 잔차에 균일 가중치(1.0)를 적용해 learned uncertainty 효과를 제거한 기준을 만든다. ORB-SLAM3-OF-U에서는 flow network가 예측한 confidence/uncertainty로 reprojection error를 가중해, 대응이 흔들리는 저품질 영상 구간에서 강건성을 얻도록 설계한다.

- **Empirical Impact**: 실험은 대표 난이도 벤치마크인 TartanAir와 UZH-FPV에서 궤적 정확도(translation ATE, rotation ATE)를 비교하는 방식으로 진행됐고, ORB-SLAM3-OF와 ORB-SLAM3-OF-U가 기본 ORB-SLAM3보다 크게 향상되는 결과를 보였다. 특히 learned uncertainty를 더한 ORB-SLAM3-OF-U가 시각적으로 어려운 조건에서 강건성이 증가하며, out-of-distribution으로 여겨지는 UZH-FPV에서 딥 V-SLAM 계열의 성능을 뛰어넘는 양상이 보고된다. 결론적으로 딥 V-SLAM의 핵심 이득은 recurrent 구조보다는 learned 2D data association과 uncertainty에 달려 있으며, 다음 세대 V-SLAM 설계에서 이 두 구성요소를 학습 기반으로 적극 도입해야 한다는 메시지를 준다.



### Prior-First, Condition-Second: Scalable and Controllable Hand Motion Completion (https://arxiv.org/abs/2607.05938)
- **Prior Approaches**: 기존 데이터 기반 손 모션 생성은 end-to-end로 몸과 손을 함께 학습하거나, 손-only/HOI 중심으로 고충실도를 노리지만 몸-손 협응(kinematic coupling)과 의미 제어가 약한 경우가 많다. 또한 text나 음성 같은 의미 라벨은 long-tailed·고가·상황 의존성이 커서, 스튜디오별 캡처/스켈레톤/라벨 체계가 달라지면 cross-dataset 전이가 잘 안 된다.

- **Core Contribution**: 이 논문은 controllable hand motion을 ‘희소 조건 → 모션’의 직접 매핑이 아니라, unlabeled 데이터로 학습한 kinematic prior 위에서 의미 제어를 가볍게 얹는 prior-first, condition-second로 재정의한다. 구체적으로 streaming 가능한 streaming, autoregressive body-hand prior를 먼저 고정 학습하고, 그 위에 semantically-layered adapters로 텍스트 또는 self-supervised 속성 제어를 주입한다.

- **Technical Challenges**: 핵심 난점은 (1) 라벨 없이도 손을 몸의 역학적 제약에 맞게 생성해야 하고, (2) 확산 과정에서 손 쿼리가 잡음일 때도 몸-손 결합이 무너지지 않게 해야 하며, (3) 소량 라벨로 의미 제어를 학습해야 한다는 점이다. 이를 위해 Transformer diffusion을 clip-level autoregressive로 구성해 실시간 롤아웃을 지원하고, kinematic chain cascading attention(KCCA)로 root-to-wrist 계통을 따라 기계적 결합을 강제하며, 조건 주입은 사전(prior) 전체가 아니라 적절한 kinematic 레벨에만 adapter 게이트로 학습/조절한다.

- **Empirical Impact**: 실험에서는 end-to-end conditioned baseline 대비 생성된 손의 kinematic plausibility, 견고성, controllability가 전반적으로 향상되며 특히 저자원(low-resource)·교차 데이터셋(cross-dataset) 설정에서 격차가 크게 나타난다. 또한 450+ FPS급 실시간 추론과 인터랙티브 authoring 워크플로를 시연해 프로덕션 애니메이션 파이프라인 적용 가능성까지 실증한다.



### TRIG: Trajectory-Rig Decoupled Metric Geometry Learning (https://arxiv.org/abs/2607.05801)
Comments:
          9 pages, 3 figures, 8 tables

- **Prior Approaches**: 기존 비전 기반 3D 기하 인식은 카메라 자세·깊이·3D를 함께 학습하거나, BEV/옥젤 기반으로 공간을 표현하는 방식이 주를 이뤘습니다. 다만 멀티캠에서 카메라 자세를 하나의 얽힌 조건으로 다루면 시간에 따른 ego-motion과 정적인 camera-rig 토폴로지가 같이 모델링되어 metric scale이 암묵적으로 추정되기 쉽습니다.

- **Core Contribution**: TRIG는 Trajectory-Rig Decoupled Metric Geometry Learning으로, 카메라 포즈를 ego-trajectory(시간 가변)와 camera-rig(정적)으로 명시적으로 분해해 metric-aware 학습을 가능하게 합니다. 이 분해를 통해 차량 측 기하 prior를 motion과 토폴로지 경로로 분리해 주입하고, 글로벌 좌표 정합 후처리 부담을 줄이는 방향으로 설계했습니다.

- **Technical Challenges**: 핵심 난제는 분해된 포즈를 모델이 실제로 metric 일관성을 유지하며 활용하도록 만드는 것입니다. TRIG는 decoupled pose encoding/pose supervision으로 trajectory는 시간 일관성, rig은 동시 카메라 간 제약에 각각 맞춰 학습시키고, Sparse Temporal–Spatial Attention(STSA)로 카메라 간 상호작용과 시간 집계를 분리해 장거리 추론의 계산비용도 억제합니다.

- **Empirical Impact**: 5개 자율주행 벤치마크 실험에서 TRIG는 pose estimation, metric depth prediction, 3D reconstruction 전반에 걸쳐 state-of-the-art를 달성했습니다. 특히 엔트angled prior 기반 모델이 wide-baseline 멀티캠에서 붕괴하는 현상을 decoupled 설계와 supervision 분리로 완화해, 드리프트에 강한 metric pose와 재구성 품질을 보여줬다는 점이 의미 있습니다.



### Image2Sim: Scaling Embodied Navigation via Generative Neural Simulator (https://arxiv.org/abs/2607.05765)
- **Prior Approaches**: 기존 Embodied navigation은 Matterport3D, HM3D, Gibson, Replica 같은 real scan 기반 시뮬레이션과 Habitat 같은 툴에 크게 의존해 왔다. 다만 스캔은 비용과 노동이 커서 확장성이 낮고, procedural/synthetic 환경은 자산·레이아웃·렌더링 통계의 현실성이 떨어져 sim-to-real gap이 자주 발생한다.
또한 NeRF나 3D Gaussian Splatting 계열은 고충실 렌더링을 제공하지만 per-scene 최적화가 필요해 대규모 데이터 엔진으로 쓰기 어렵고, 생성형 모델은 collision·rigid-body 일관성 같은 닫힌루프 상호작용의 물리적 구조를 지속적으로 보장하기 힘들다.

- **Core Contribution**: 이 논문은 Image2Sim으로, posed RGB-D 이미지 시퀀스에서 고품질 interactive 3D 환경을 실시간(neural)으로 구성하는 프레임워크를 제안한다. 핵심은 3D spatial anchoring(기하적 고정)과 photorealistic observation synthesis(관측 생성)를 분리해, 스케일과 물리적 근거, 시각 충실도의 균형을 맞추는 것이다.
또한 단순 렌더러를 넘어 collision-aware 모션 엔진과 VLM 기반 지시문 생성까지 결합해, 약 20K 씬에서 10M+ navigation 학습 샘플을 자동 생성하는 embodied data engine으로 확장한다.

- **Technical Challenges**: 문제의 기술적 난점은 (1) sparse/noisy 관측에서 구조적 빈틈을 채우면서도 (2) 3D grounding을 유지해 홀루시네이션을 억제하고 (3) 닫힌루프에서 실행 가능한 물리 궤적을 제공하는 것이다.
Image2Sim은 feed-forward feature Gaussian으로 depth로부터 3D feature-Gaussian을 단일 패스로 들어 올리고, 렌더링은 Geometry-Aware One-Step Pixel Flow로 alpha(불확실도) 기반 gated prior를 조건 삼아 panoramic RGB-D를 한 스텝에 생성한다.
여기에 재현 가능한 경로를 위해 가시성/장애물/여유공간을 반영한 traversable voxel connectivity graph를 만들고, 그래프 계획+컨트롤러로 궤적을 생성한 뒤 이를 매크로-스텝으로 분절해 자연어 instruction을 Qwen3-VL-32B-Instruct로 정렬 주석한다.

- **Empirical Impact**: 실험에서는 20K 규모 interactive environment에서 10M+ 샘플을 만들었고, 렌더링 품질·속도 면에서 panoramic variant가 noisy depth 상황에서도 높은 PSNR/SSIM을 유지하며 실시간에 가까운 FPS를 달성한다. 또한 Gaussian-only 및 순수 생성형 비교군은 계산 지연이 크거나 미관적으로 그럴듯하지만 구조 일관성이 떨어져 효율-견고성의 균형에서 밀린다.
Navigation 학습은 Image2Sim 내부에서만 진행한 뒤 Habitat 도메인에서 zero-shot 전이 평가를 수행했으며, 그 결과 R2R-CE, RxR-CE, REVERIE-CE 벤치마크에서 새로운 SOTA 수준의 향상과 강한 도메인 전이를 보였다. 이는 스케일 가능한 neural simulation이 embodied navigation의 실용적인 학습 기판이 될 수 있음을 경험적으로 뒷받침한다.



### TypeGo: An OS Runtime for Embodied Agents (https://arxiv.org/abs/2607.05482)
- **Prior Approaches**: 기존 LLM 로봇 제어는 request/response 방식의 플래닝(한 번에 전체 계획, 혹은 단계별 ReAct/SayCan류)이나, end-to-end로 직접 동작을 뽑는 접근이 많습니다. 하지만 전자는 첫 행동까지 지연이 크고 장면 변화에 회복이 느리며, 후자는 동시성·선점·실시간 제어 관점에서 OS 수준의 다중 작업 조율이 부족합니다. 또한 멀티태스크 환경에서 물리 자원을 어떻게 충돌 없이 분배하고, 선점 후 재개를 어떤 의미로 할지에 대한 설계가 충분히 다뤄지지 않았습니다.

- **Core Contribution**: 이 논문은 LLM을 ‘질문하면 답하는 오라클’로 크리티컬 패스에 두기보다, embodied agent용 OS 스타일 런타임처럼 비동기 루프로 상시 실행해 지연을 숨기자는 가설을 제시합니다. 이를 구현한 TypeGo에서는 Skill Kernel이 물리 subsystems를 중재하고, 스케줄러(S3)가 장면·프롬프트 가이드라인을 바탕으로 프로세스를 시맨틱하게 선점/재개합니다. 사용자는 자연어로 작업 목표와 반응 규칙(reactive rules)을 함께 작성하고, 규칙은 빠른 S0 레이어로 컴파일해 LLM 호출 없이 즉시 발동되게 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) LLM 추론 지연을 실시간 제어 경로에서 제거하는 것, (2) 여러 목표가 동시에 존재할 때 물리 액추에이터를 안전하게 공유/선점하는 것, (3) ‘목표’뿐 아니라 ‘조건-행동’ 반응 규칙을 자연어로 빠르게 다루는 것입니다. TypeGo는 다중 cadence 비동기 플래닝(S0~S3)과 speculative skill streaming의 step queue로 실행과 계획을 겹치고, S3는 PCB 기반 프로세스의 선점 주체를 구분해 interrupt-and-return/replace-without-return 정책을 적용합니다. 또한 반응 규칙은 LLM 없이 Python 조건/행동 함수로 컴파일해 S0이 100Hz 수준에서 우선순위 선점을 수행하도록 했습니다.

- **Empirical Impact**: Unitree Go2(Kalos) 프로토타입에서 TypeGo는 과제 묶음 기준 per-step delay를 step-by-step planning 대비 50% 줄였고, monolithic planning 대비 TTFA(time-to-first-action)를 73% 단축했습니다. 동시 작업 실험에서는 낮은 스케줄 오버헤드(예: S3 스케줄링 약 200ms, S0 reflex 처리 약 50ms 수준)로 멀티태스크를 허용하는 방향성을 보였고, 실행 중 선점 후 재개가 의도대로 동작함을 정성적으로 확인했습니다. 반면 여러 비동기 루프 때문에 토큰 사용량이 ReAct/PAE보다 늘어나는 트레이드오프도 함께 드러나, 향후 런타임 정책 최적화 여지를 제시합니다.



### Geometry-Aware Infrastructure-Anchored Denoiser for UWB Sensing and Work-Zone Reconstruction (https://arxiv.org/abs/2607.05449)
- **Prior Approaches**: 기존 work-zone 인식은 TTCD(일시적 교통통제 장치) 탐지나 HD-map 갱신처럼 요소 단위로 접근하는 경우가 많아, 드라이버블 경계는 암묵적 부산물로 취급되기 쉽습니다. UWB 기반 V2I 연구도 대개 anchor별 ranging 오차(평균 오차, LOS/NLOS 분류 등)를 줄이는 데 초점을 두며, 경계 형상에서 ‘공간적으로 중요한’ anchor 오차가 미치는 비대칭 영향을 직접 학습목표에 반영하지 못합니다.

- **Core Contribution**: GAIA는 work-zone 경계 복원을 위해 UWB ranging을 ‘경계 지향(boundary-oriented) denoising’ 문제로 재정의하고, denoised 거리들이 boundary reconstruction에 일관되게 기여하도록 학습합니다. 특히 latent anchor-layout 추정과 deterministic distance projection을 결합해, 단순 평균 오차 최소화가 아니라 경계 품질(IoU 등) 관점에서 거리 예측을 유도합니다.

- **Technical Challenges**: 핵심 난제는 NLOS·burst noise·long-tail 오류로 인해 anchor별 측정이 흔들리는데, 이때 경계 복원은 일부 ‘기하학적으로 critical한’ anchor 오차에 크게 좌우된다는 점입니다. GAIA는 PoseMLP Base(고정)로 초기 거리 패턴을 잡고, Temporal Refinement으로 시간·anchor 간 상관을 정리한 뒤, Layout Head로 latent anchor 배치를 추정하고 GeoDist에서 geometry-consistent 거리를 투영해 예측과 공간 정합성을 동시에 맞춥니다.

- **Empirical Impact**: GAIA는 동기화된 UWB·GNSS·IMU로 구성된 실외 데이터에서 전체 range MSE를 PoseMLP 대비 18.4% 낮추고 polygon IoU는 15.5% 높였습니다. 또한 real-data-calibrated stress-test 시뮬레이터로 NLOS 및 long-tail 손상이 심해질 때도 견고성을 확인하며, geometry-aware range denoising이 spatially coherent work-zone reconstruction으로 이어짐을 실증했습니다.



New uploads on arXiv(cs.MA)

### Decision Protocols in Multi-Agent Large Language Model Conversations (https://arxiv.org/abs/2607.05477)
Comments:
          Master's thesis, University of Göttingen

- **Prior Approaches**: 기존 연구는 LLM의 멀티에이전트 활용 시 여러 에이전트를 두더라도 단일 의사결정 프로토콜을 쓰거나, 적용 범위가 좁은 데이터셋에서만 성능을 확인하는 경우가 많았다. 또한 멀티에이전트는 학습 비용을 줄일 수 있지만, 에이전트 간 토론·결정 과정으로 테스트 시간이 늘어날 수 있어 프로토콜 설계가 핵심이라는 점이 반복적으로 지적돼 왔다.

- **Core Contribution**: 이 논문은 Multi-Agent LLM(MALLM) 프레임워크를 제안해 투표(voting), 합의(consensus), judge 기반 결정(judge decision) 등 다양한 의사결정 프로토콜을 구현하고 체계적으로 평가한다. 대화형 과업 해결을 목표로, 여러 에이전트가 논의하며 최종 해답에 도달하는 과정을 프로토콜 단위로 시뮬레이션한다는 점이 차별점이다.

- **Technical Challenges**: 핵심 기술 난제는 에이전트들이 만들어낸 답변을 어떤 규칙으로 통합해 ‘정답으로 수렴’시키는지에 대한 설계였다. 논문은 각 프로토콜별 협업 방식을 비교하고, 특히 독립적인 솔루션 생성을 통해 response diversity를 늘리면 의사결정 품질이 향상됨을 확인했으며, 의사결정 과정에서의 정보 접근 방식 변화는 성능에 큰 영향을 주지 않는 경향을 관찰한다.

- **Empirical Impact**: 실험은 지식 기반 벤치마크(MMLU, MMLU-Pro, GPQA)와 논리 기반 벤치마크(StrategyQA, MuSR, Math-lvl-5, SQuAD 2.0)를 폭넓게 포함해 프로토콜-과업 유형 간의 상호작용을 보여준다. 결론적으로 consensus는 지식 집약 과업에서, voting 및 judge 프로토콜은 논리 과업에서 더 유리했으며, 에이전트 간 다양성 확보가 전반적 성능 향상에 기여한다는 실증적 근거를 제공한다.



### Danus: Orchestrating Mathematical Reasoning Agents with Fact-Graph Memory (https://arxiv.org/abs/2607.06447)
- **Prior Approaches**: 기존 LLM 기반 수학 에이전트는 generate–verify–revise 루프를 중심으로 하되, 대개는 역할이 다른 다중 에이전트를 쓰거나 단일 추론 흐름을 반복하는 방식이 많았습니다. Aletheia·Rethlas·QED·ProofCouncil·AI co-mathematician 등은 검증자와 반복 편집을 갖추고 성과를 내었지만, 에이전트를 더 늘려 병렬 proof search를 “확장”할 때는 공유 상태(중간 주장)의 정리·신뢰성 문제가 체계적으로 다뤄지지 않았습니다.

- **Core Contribution**: Danus는 연구 수준 수학 추론을 위한 오케스트레이션 시스템으로, shared fact graph를 전역 메모리-관리 메커니즘으로 삼아 병렬로 생성된 결과를 신뢰성 있게 누적합니다. main agent가 계획·조정·중간 상태 요약을 맡고, worker들이 병렬로 증명 탐색을 수행하며, stateless verifier가 통과한 수학적 주장만 fact graph에 “사실”로 편입되게 합니다.

- **Technical Challenges**: 핵심 기술 난관은 많은 worker가 동시에 proof search를 진행할 때 중간 주장들이 서로 간섭하거나(혹은 불필요·오류 정보가 섞여) 검증 이후에도 논증 상태가 혼란스러워지는 점입니다. Danus는 DAG 형태의 fact graph에 논리 의존성을 간선으로 기록하고, verifier가 통과한 주장에 대해서만 proof와 dependency를 함께 저장하며, 필요 시 revocation으로 잘못된 fact와 그 의존 항목을 연쇄 제거해 상태의 일관성을 유지합니다.

- **Empirical Impact**: algebraic geometry·singularity theory·combinatorics의 연구 수준 케이스 스터디 6개에서 Danus는 fact-graph 기반 메모리 메커니즘을 통해 긴(다단계) 수학적 증명을 구성하는 과정을 보였고, 웹 기반 GPT-5.5-pro 단독 제시는 의미 있는 결과를 내지 못했다고 보고합니다. 예컨대 정리급 결과(예: Mori의 bend-and-break의 foliated 일반화에서 최적 상수 r+1, 그리고 3차원 foliated Shokurov global index conjecture의 완전 해결)에서는 worker들이 다수의 verified fact를 누적한 뒤, 마지막에 인간 검토/수정까지 연결해 논문 형태로 재구성하는 흐름이 제시됩니다.



### A study of holes: Topological analysis reveals crowd dynamics regimes in a bidirectional corridor scenario (https://arxiv.org/abs/2607.06086)
Comments:
          Presented at Traffic and Granular Flow 2026 (TGF26)

- **Prior Approaches**: 기존 보행자 군중 동역학 연구는 밀도·속도 같은 거시 관측치나 hand-crafted metric을 써서 군집/상태를 구분하는 경우가 많았다. 최근에는 상호작용 시뮬레이션에서 집단 현상을 분류할 수 있어도, 보행자 위치 자체를 직접 다루며 구조적 차이를 안정적으로 드러내는 방식은 상대적으로 드물다.

- **Core Contribution**: 이 논문은 보행자 위치 시계열에서 persistent homology를 적용해, 연결성(connected components)과 구멍(holes) 같은 위상적 구조를 비지도적으로 요약한다. 전체 시계열을 그래프적 임계값 변화에 따른 persistence signature로 정리한 CROCKERs를 만들고, PCA로 저차원 공간에서 군중 시나리오(일방/양방, 균형/불균형)를 분류 가능한 형태로 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 보행자 위치의 방향성과 (2) 시간에 따른 구조 변화를 함께 반영하면서도 사전 가정 없이 구분력을 확보하는 것이다. 연구진은 Vietoris–Rips complex에서 Betti number(β0, β1)를 기반으로 persistence를 만들고, 더 나아가 temporal delay embedding으로 (x_t,y_t,x_{t-d},y_{t-d})를 사용해 이동 방향 정보를 암묵적으로 포함함으로써 성능을 끌어올렸다.

- **Empirical Impact**: 시뮬레이션 결과 CROCKERs의 첫 두 주성분에서 일방/양방 및 불균형 유입 레짐이 뚜렷한 클러스터로 분리됐으며, 대칭(symmetric) 구성에 대해서도 separation이 유지됐다. 특히 temporal delay d=2초에서 silhouette coefficient가 0.376으로 크게 상승했는데, 지연 없이 d=0일 때는 0.033에 그쳐 시간정보의 중요성이 실증됐다. 또한 1번 주성분이 평균 보행자 수와 높은 상관(ρ≈0.92)을 보여, 구조적 요약이 밀도 영향(정량·정성)을 확인하면서도 사전 패턴 가정 없이 동역학을 설명할 수 있음을 시사한다.



### Information Limits and Attractor Dynamics in Economies of Frontier LLM Agents: A Pre-Registered Tes (https://arxiv.org/abs/2607.06001)
Comments:
          15 pages. Preprint. Zenodo: this https URL. Companion synthesis: arXiv:2606.12502

- **Prior Approaches**: 기존 연구는 Kelly/정보이론을 근거로 측정 가능한 정보-부(wealth) 성장 연결고리를 제시했지만, LLM 에이전트가 실제로 서로 베팅하며(패리뮤추얼 결합) 자원과 부가 상호 의존하는 환경에서 “결합된” 정량 법칙이 성립하는지는 제대로 검증되지 못했다. 또한 다중 에이전트 인센티브·제어에 대한 mean-field 류 모델은 인구 수준 오차가 선형 응답처럼 완만히 변할 것이라 가정하지만, 이는 프런티어 LLM 집단에서 실험적으로 확인된 적이 거의 없다.

- **Core Contribution**: 이 논문은 Claude Opus 4.8을 대상으로, 사전등록(pre-registered)된 두 갈래 예측을 하나의 실험 프로토콜로 동시에 시험한다. 첫째는 정보이론 기반 “parimutuel gap law/coalition submodularity/공동 성장 상한/정보 우열에 따른 자산 집중” 같은 결합 capacity region 구조를 정량 검증한 것이고, 둘째는 인센티브·제어에 따른 mean-field residual-scaling이 실제 LLM 집단에서도 매끄러운 선형-응답 영역을 갖는지 시험한다.

- **Technical Challenges**: 프런티어 LLM 실험에서 가장 큰 난관은 결과가 실험 설계나 분석 선택에 의해 흔들릴 수 있다는 점이었고, 논문은 이를 줄이기 위해 공용 git 체인에 예측·합격대·판정 규칙을 동결한 뒤 실행 전/후 수정까지 엄격히 관리했다. 또한 모든 API 호출을 캐시에 저장해 재현성을 확보하고, coalition 정보 계산은 단순 베이지안 곱결합으로는 XOR 같은 시너지 표현이 불가능해 coalition-level elicitation(연합 단위 조건부 포스터리어 제시)으로 정의해 밴드 조건을 만족시키도록 설계했다.

- **Empirical Impact**: 결과 1에서는 결합된 환경에서 relative growth와 relative claimed information의 차이가 사전등록된 오차 범위(최악 46 millinats, 밴드 50)에 들어맞았고, conditional independence 구간에선 coalition value가 submodular, XOR 시너지 제어에서는 supermodular로 뒤집히는 등 capacity region의 핵심 구조가 확인됐다. 반면 결과 2에서는 residual-scaling의 “noise-maintained dispersion” 영역을 72/72 인구 실행에서 찾지 못했고, 목표 분산이 붕괴하며 경계에서 step-function/시드 선택형 bistability가 나타나 mean-field의 매끄러운 응답 가정이 실현되지 않는다는 결론을 내렸다. 저자들은 프로토콜·사전등록 체인·콜 캐시·분석 코드를 공개해, 양성/음성 결과를 동일한 무게로 검증 가능하게 만들었다.



### MCP-Enabled Agentic AI for Autonomous IPoDWDM Network Lifecycle Automation (https://arxiv.org/abs/2607.05975)
Comments:
          Accepted for demo presentation at the European Conference on Optical Communication (ECOC 2026)

- **Prior Approaches**: 기존에는 IPoDWDM 네트워크의 구성·운영 자동화를 작업 단위로 나눠 사람이 개입하거나, 특정 벤더 도구에 강하게 의존하는 방식이 많았다. 이로 인해 멀티레이어 변경 전 주기(라이프사이클) 전반을 일관되게 닫힌 고리(closed-loop)로 제어하기 어렵다는 한계가 있었다. 또한 네트워크 상태를 실시간 텔레메트리로 반영해 자동 의사결정을 반복하는 통합 구조가 부족했다.

- **Core Contribution**: 이 논문/데모는 MCP-enabled 에이전틱 AI 아키텍처를 제안해, 벤더 비종속(vendor-agnostic) IPoDWDM 네트워크의 자율 제어를 목표로 한다. 핵심은 텔레메트리와 함께 에이전트가 라이프사이클 전 과정을 연결해 end-to-end로 오케스트레이션하고, closed-loop 제어로 지속적으로 조정하는 점이다. 데모는 GNPy를 활용한 실제 제어 흐름을 통해 개념을 구체화한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 (1) 멀티레이어 자동화 작업을 한 번에 끊김 없이 연결하는 end-to-end 계획·실행 체계, (2) 실시간 텔레메트리 기반으로 상태를 추정·반영해 closed-loop로 교정하는 제어 로직, (3) 벤더 의존성을 줄이면서도 네트워크 변경을 안정적으로 수행하는 연동이다. 제안 아키텍처는 MCP 기반 에이전트가 GNPy와 텔레메트리를 함께 사용해 관측-판단-제어 사이클을 구성함으로써 이 문제들을 해결한다.

- **Empirical Impact**: 저자들은 실제 테스트베드에서 live end-to-end 라이프사이클 멀티레이어 자동화와 closed-loop 제어를 검증해, 에이전틱 에이전트가 실운영에 가까운 조건에서 동작함을 보였다. 이는 통신 네트워크 자동화 분야에서 에이전트형 제어를 벤더 비종속 환경으로 확장하는 실증 사례로 의미가 있다. 결과적으로 ML/AI 기반 네트워크 오케스트레이션의 실시간 제어·검증 경로에 대한 참고점이 될 전망이다.



### Delay-Aware Active Triangulation with Uncertainty-Driven Multi-Agent Reinforcement Learning for Counter-UAS (https://arxiv.org/abs/2607.05957)
Comments:
          Accepted to IROS 2026

- **Prior Approaches**: 기존 C-UAS용 능동 시각 삼각측량은 멀티뷰 기하를 바탕으로 하되, 다중 에이전트에서 발생하는 누적 지연(탐지-통신-의사결정 전파)을 충분히 모델링하지 않는 경우가 많았다. 또 많은 지연-aware RL 연구가 단일 에이전트 또는 제한된 지연 구조를 가정해, 에이전트 간 관측이 더 오래된 정보가 되는 비대칭 지연을 그대로 학습시키기 어렵다는 한계가 있었다. Perception-aware MPC 등은 가시성 계획을 다루지만 분산 통신 지연 자체를 확률적으로 다루는 방향은 상대적으로 약했다.

- **Core Contribution**: 이 논문은 Counter-UAS 시나리오에서 지연을 고려한 멀티에이전트 active visual triangulation을 위해, AoI(Age-of-Information)를 관측에 포함한 Dec-POMDP 기반 delay-aware RL 프레임워크를 제안한다. 또한 privileged reward(클린 상태 기반)와 perception-consistent reward(정책이 받는 지연·잡음 관측 기반)를 통제 비교해, 보상-관측 정렬이 성능과 안정성에 미치는 영향을 체계적으로 보여준다. 마지막으로 픽셀·포즈·짐벌 캘리브레이션·카메라 intrinsics까지 다중 소스 불확실성을 공분산 전파로 통합해, 불확실성 모델링의 누락이 성능을 어떻게 무너뜨리는지 정량화한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘지연된 비동기 관측’이 만드는 부분관측 문제를 학습 중에 어떻게 다루느냐와, 지연·잡음·기하 불확실성을 보상/학습 신호에 어떻게 정확히 반영하느냐였다. 저자는 AoI 태그를 메시지 수준으로 노출하고 GRU 기반 재귀 정책으로 시간적 불일치를 흡수하도록 설계했으며, dual-path reward 구조로 클린 상태 최적화가 왜 noise-fragile 구역을 유도할 수 있는지 관측했다. 더 나아가 단일(각도) 잡음만 쓰는 단순 모델을 넘어서 다중 소스 analytical covariance propagation을 도입해, 픽셀 탐지·포즈·짐벌·intrinsics 오차가 삼각측량 공분산에 미치는 영향을 분리해 반영했다.

- **Empirical Impact**: MAPPO를 4096개 병렬 환경에서 학습한 결과, perception-consistent 설정은 RMSE 0.547±0.217m, 삼각측량 유효도 78.1%를 달성했으며 track loss도 감소했다. AoI를 제거하면 삼각측량 유효도가 크게 떨어지고(유효도 +10.6%p 수준), MLP+프레임 스태킹은 지연의 비균일 타임스탬프 구조를 따라가지 못해 유효도가 거의 붕괴(0.7%)했다. 또한 다중 소스 공분산 모델은 angular-only 대비 RMSE를 2.8배 낮추며 유효도도 크게 개선해, 실제 C-UAS에서 필요한 불확실성 모델의 폭을 실증적으로 입증했다.



### StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems (https://arxiv.org/abs/2607.05844)
Comments:
          Code and supplementary materials available at: this https URL

- **Prior Approaches**: 기존 에이전트 메모리 구현은 분기·재시도·복제 과정에서 생기는 관측 불일치를 대부분 덮어쓰기 규칙으로 숨기거나, 충돌을 보더라도 추적·수정이 어렵다는 한계가 있었다. CRDT/OpSet 같은 표준 합의·수렴 기반은 상태를 모을 수 있지만, 에이전트 관점의 “충돌을 언제·어떻게 드러내고 누가 선택/보류할지” 같은 계약(semantics contract)은 약하게 설계되는 경우가 많았다. 그 결과 검증 후에도 잘못된 행동으로 이어지거나, 이전 수정이 반영되지 않은 채 남는 문제가 발생할 수 있었다.

- **Core Contribution**: StateFuse는 표준 OpSet/CRDT merge 위에 얹는 “충돌 인지 replicated memory contract”를 제안한다. 새로운 join 대수를 만들기보다는, 불변 히스토리·명시적 conflict 객체·정확/의미 기반 correction handle(claim_id / claim_ref)·결정론적 predicate contract·프로젝션 시점의 제한된 resolution 권한을 묶어 에이전트가 감사 가능하게 충돌과 수정 가능성을 다루도록 한다. 또한 resolve가 replicated state를 재작성하지 못하게 해, 충돌을 공용 의사결정 표면에 일관되게 노출한다.

- **Technical Challenges**: 핵심 난제는 복제 병합은 수렴시키면서도 충돌을 “숨기지 않고” 유지하는 해석 규칙을 계약으로 고정하는 것이다. StateFuse는 claim을 Evidence/Claim/Retract/Decision의 불변 연산으로 저장하고, retraction이 claim_id 또는 claim_ref를 표적으로 삼아 exact/semantic correction을 각각 다르게 비활성화하도록 설계했다.さらに predicate registry의 normalize/equal 같은 결정론 규칙을 계약으로 강제하고, projection-time resolver는 후보 선택·abstain만 수행하되 base-memory mutation은 금지함으로써 결정론적 재현성과 충돌 표면의 일관성을 확보한다.

- **Empirical Impact**: MemoryAgentBench의 충돌을 포함한 282문항 슬라이스에서 StateFuse는 정확도에서 강한 flat/ collapsed 계열과 동률을 보이며, 보편적 accuracy 향상보다는 “무엇을 표면에 드러내는가”의 차이가 핵심으로 나타났다. 다만 StateFuse와 conflict-preserving baseline은 충돌-bearing 과제에서 모순을 끝까지 노출한 반면, raw-log 및 collapsed 표면은 이를 전혀 드러내지 않았다. 제어된 에이전트 루프에서는 ambiguity를 보존하고 검증 후에 abstain할 수 있는 설계가 붕괴(collapse)보다 훨씬 안전했으며, correction-handle ablation에서도 claim_ref가 의미 타깃 복구와 unseen-target no-resurrection을 더 잘 지원해 수정 표현력의 차이를 확인했다.



### Deep Reinforcement Learning for Dynamic Battery Management of Autonomous Order Pickers (https://arxiv.org/abs/2607.05683)
- **Prior Approaches**: 기존에는 배터리가 임계치 아래로 내려가면 충전으로 보내는 고정 규칙 휴리스틱이 주로 쓰였지만, 주문 도착이 확률적으로 바뀌는 환경에서는 정지 타이밍과 충전량이 쉽게 빗나가 비효율을 키웠습니다. 또한 많은 연구가 충전소 선택이나 충전 종료 시점까지를 고정 규칙으로 처리하거나, 다중 AMR이 공유하는 충전 인프라에서의 큐 혼잡을 충분히 반영하지 못했습니다.

- **Core Contribution**: 이 논문은 multi-block 창고에서 다중 AMR의 충전 의사결정을 PPO 기반 DRL이 동적으로 학습하도록 설계했습니다. 에이전트가 (1) 어느 충전소를 선택할지, (2) 충전을 언제 시작·언제 끝낼지(충전 지속시간)까지 배우며, 충전소별 예상 큐잉 시간도 정책에 반영합니다.

- **Technical Challenges**: 핵심 난제는 ‘언제 충전해야 하는가’와 ‘충전소 경쟁으로 인한 대기’를 동시에 다루면서, 다중 에이전트 간 상호 의존을 학습에 녹여내는 것입니다. 이를 위해 IPPO 형태로 로컬 관측(자기 상태+다른 에이전트 정보+각 충전소 큐 길이)을 사용하고, average-reward PPO로 시간의 장기 가중치를 균일화했으며, action mask로 물리적으로 불가능한 행동을 배제해 학습 안정성을 높였습니다.

- **Empirical Impact**: 대규모 수치 실험에서 제안한 PPO 프레임워크는 강력한 기준선 대비 주문 완료율을 최대 6%까지 끌어올리고, 재충전으로 소요되는 총 시간을 유의미하게 줄였습니다. 또한 다양한 창고 구성과 확률적 도착률 조건에서도 성능 강건성을 확인했으며, 학습된 정책을 해석해 표준 벤치마크를 능가하는 운영 인사이트를 제시합니다.



### PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates (https://arxiv.org/abs/2607.05483)
Comments:
          24 pages, 13 figures, including appendix

- **Prior Approaches**: 에이전틱 워크플로는 한정된 LLM 컨텍스트 때문에 progressive disclosure(점진적 공개) 방식으로 매 단계에 필요한 상태 조각만 모델에 보여주는 흐름이 일반적이다. 이를 위해 grep-like 검색, RAG, AST 쿼리, task-specific agent skills 같은 읽기 최적화가 많이 쓰이지만, 로컬에서 제안한 “수정”이 전체 상태에 대해 언제 유효한지는 계약이 부족하다. 또한 스키마 검증·constrained decoding·입증되지 않은 근거를 통한 값 생성은 각각 국소적 실패만 줄일 뿐, provenance/authorization 관점의 글로벌 유효성 판단과 다단계 확장에 취약하다.

- **Core Contribution**: 논문은 PatchOptic이라는 “공유 상태용 step 인터페이스”를 제안한다. 각 단계는 projected read view(투영된 읽기)·authorized write region(허용 쓰기 범위)·patch-source region(패치가 참조할 소스 경로)을 하나의 계약으로 선언하고, 런타임에서는 모델에 투영 뷰만 제공한 뒤 전체 상태 기준으로 patch를 검증(commit 전)한다. 더 나아가 동일 선언이 delegation/서브워크플로 조합/same-phase 재정렬 같은 정적 분석도 가능하게 해 워크플로 재구성이 안전해진다.

- **Technical Challenges**: 핵심 기술 난제는 “로컬로 보인 뷰에서 나온 JSON Patch가 전체 상태에서 의미적으로도(및 권한적으로도) 성립하는지”를 LLM 실행 없이도 스케일 있게 판정하는 것이다. 이를 위해 PatchOptic은 optics에서 영감을 받은 authority triple을 기반으로 projected read와 검증 가능한 structured patches를 결합하고, 스키마·phase 제약·적용가능성·불변조건·patch 연산의 참조 소스까지 verifier가 승인하기 전에 차단한다. 또한 hidden sources(숨은 소스)처럼 선언되지 않은 경로를 이용하는 패치 아티팩트는 no-model 테스트의 containment 시나리오에서 모두 거부하도록 설계했다.

- **Empirical Impact**: PatchOptic의 실험은 6개 도메인, 총 46개 케이스로 구성된 PatchBench를 통해 수행됐다. projected read 설정은 강한 actor 조건에서 semantic pass(허용 출력의 의미 적합)를 약 0.61에서 0.78–0.80(GPT-5-mini)으로 끌어올리면서도 토큰 비용과 누출(leakage) 보고를 줄였다. 런타임 검증을 결합하면 leak runs가 라운드당 0.1 수준으로 감소하고, hidden-source patch artifact는 containment 테스트에서 전부 거부되어 “읽기 최소화 + 커밋 전 검증”의 실효성이 입증됐다.




New uploads on arXiv(cs.CL)

### Learning User Simulators with Turing Rewards (https://arxiv.org/abs/2606.19336)
- **Prior Approaches**: 기존 사용자 시뮬레이터 학습은 LLM에 대해 단일 정답 응답을 맞추는 방식이 주류였습니다. 대표적으로 (1) 정답과의 유사도를 LLM judge로 보상하거나 (2) 정답의 log-probability를 최대화하는 방식이 쓰였지만, 맥락에서 가능한 응답이 무수히 많다는 점에서 ‘정답 매칭’ 한계가 드러났습니다.

- **Core Contribution**: 이 논문은 사용자 시뮬레이터를 ‘구분 불가능성(indistinguishability)’ 목표로 학습하는 {Turing-RL}을 제안합니다. Turing Test(튜링 테스트)의 판정 논리를 강화학습 보상으로 변환해, 생성 응답이 사용자의 히스토리 조건에서 실제 사용자와 구별되지 않도록(discriminative Turing reward) 모델을 최적화합니다.

- **Technical Challenges**: 핵심 난제는 LLM judge가 내는 ‘사람다움’ 점수를 안정적으로 정책에 전달하는 동시에, 콘텐츠 매칭처럼 협소한 최적화로 붕괴하지 않게 하는 것입니다. 저자들은 실제 사용자/모델 응답을 함께 제시하는 판별형 Turing reward에 점수 캡을 두고(보상 해킹 방지), Group Relative Policy Optimization(GRPO)로 여러 후보 응답의 상대적 advantage를 계산하며, 초기 SFT로 워밍업한 뒤 RL을 수행합니다.

- **Empirical Impact**: PRISM Alignment Dataset의 멀티턴 채팅과 ConvoKit의 Reddit 포럼 두 도메인에서 {Turing-RL}은 LLM 평가와 사람 평가(강제 선택 Turing test) 모두에서 기준선과 비교해 일관되게 우수한 성능을 보였습니다. 특히 유사도 보상(Sim-RL)·logprob 기반 접근보다 ‘사람처럼 보이기’ 성능에서 격차가 확인되며, 사용자 시뮬레이터 학습에선 응답 매칭보다 구분 불가능성 최적화가 효과적이라는 메시지를 강화합니다.



### Freeing the Law with LOCUS: A Local Ordinance Corpus for the United States (https://arxiv.org/abs/2606.19334)
Comments:
          14 pages, 6 figures

- **Prior Approaches**: 기존 legal NLP 코퍼스(ECHR, pile of law 등)는 판례·행정의견·연방/일부 법률 중심이어서 미국의 local ordinance는 거의 공백으로 남아 있습니다. Local law는 공개돼 있어도, 다수 벤더가 in-browser 열람용으로 흩어 배치·색인·다운로드 흐름이 제각각이라 대규모 기계 판독 코퍼스화가 어려웠습니다. 또한 로컬 법 텍스트를 동일 단위로 비교·분석하는 정규화(대표 코드 선택, 메타데이터 통일) 문제도 해결되지 않았습니다.

- **Core Contribution**: LOCUS(미국 Local Ordinance Corpus)는 미국의 시·카운티 조례를 대규모로 수집·OCR 처리해 연구용 원자료와 county-harmonized 접근 계층을 제공합니다. 원자료는 9,239개 도시/카운티의 조례 코드를 포함하고, 추가로 3,144개 카운티 중 2,309개에 대해 인구 기준 커버리지 중심의 county-harmonized 릴리스를 구성합니다. 나아가 ModernBERT 기반 분류기/스코어러를 학습해 opacity(불투명성)·paternalism(온정주의) 등 법의 차원 분석을 대규모로 가능하게 합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 벤더별 문서 포맷 다양성과 추출 실패 요인(서버 PDF 조립 한계, 이름 충돌, 숨김 임계값, anti-bot, 통합 도시/다중 카운티 등)이 결합된 ‘수집-검증-정규화’ 파이프라인 구축입니다. 이를 위해 OCR로 페이지 이미지를 Markdown으로 통일하고, 페이지 간 헤더/푸터·페이지 번호 제거 및 구획(섹션/서브섹션) 후 substantivity·function·topic 분류를 ModernBERT로 수행합니다. 또한 초기 라벨링은 LLM의 two-level zero-shot 방식과 LLM-as-a-judge 기반 페어와이즈 채점으로 품질을 관리해 structural 텍스트는 제거하는 식으로 평가 오염을 줄였습니다.

- **Empirical Impact**: LOCUS-v1은 2,211,516개 텍스트 청크(그중 상당 부분을 규칙 성격의 ‘substantive laws’로 판단)라는 대규모 지표를 제공하며, 다양한 분야(건축, 사업 인허가, zoning, nuisance 등)의 규제를 포괄합니다. 더 나아가 차원 스코어링에서 TrueSkill 기반 페어와이즈 판단과의 상관이 0.82~0.94 범위로 나타나, 로컬 법을 연속 축으로 정량화할 수 있음을 보입니다. 결과적으로 로컬 법을 전국 스케일에서 검색·추출·비교·벤치마크하는 법 AI 인프라를 제공해, 특정 주(예: 불투명성 격차)나 규제 유형(예: curfew의 차원 패턴) 같은 실증 연구를 본격화할 의미가 큽니다.



### Enhancing Decision-Making with Large Language Models through Multi-Agent Fictitious Play (https://arxiv.org/abs/2606.19308)
Comments:
          18 pages, 8 figures

- **Prior Approaches**: 기존 LLM 기반 multi-agent systems(MAS)는 divide and conquer로 실행 복잡도(긴 추론 연쇄, 넓은 정보 커버, 이질적 스킬)를 분산해 해결하는 데 집중해 왔다. 하지만 협상·게임·경쟁 시장처럼 이해관계자들의 결정이 서로 물고 늘어지는 결정 문제에는 그대로 적용하기 어렵다. 일부 decision-making 연구는 ToM(Theory of Mind)처럼 higher-order 상호 추론을 한 번의 추론 패스에서 펼치려 하지만, LLM이 재귀적 mutual anticipation을 깊게 수행할 때 성능이 급격히 떨어진다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 의사결정 복잡도의 새로운 유형으로 stance entanglement(입장 얽힘)을 정의하고, 단순 분할·고립 해법이 실패하는 이유를 “상호 의존 루프 안에서 한 궤적에 entangled된 입장들을 동시에 추론해야 하기 때문”으로 정리한다. 이를 해결하기 위해 Multi-Agent Fictitious Play(MAFP)를 제안하며, 각 이해관계자의 stance를 에이전트로 두고 결정 만들기를 equilibrium 탐색 과정으로 바꾼다. MAFP는 fictitious play 원리를 따라 각 라운드에서 다른 에이전트의 과거 결정 “경험적 혼합(empirical mixture)”에 대한 best response를 반복 수행해 약점을 노출하고 보완하도록 한다.

- **Technical Challenges**: 핵심 기술 난관은 natural-language 정책 공간에서 equilibrium 성질(일방적 일탈로 얻는 이득이 0에 가까움)을 만족하는 결정을 찾되, LLM의 재귀적 상호 추론을 단일 체인에서 깊게 펼치지 않아야 한다는 점이다. MAFP는 이를 위해 training-free로, 라운드마다 aggregation operator로 상대 정책의 empirical mixture를 요약·구성하고 best-response operator로 해당 혼합에 최적 대응하는 정책을 생성한다. 또한 마지막에 각 라운드에서 나온 정책들의 empirical mixture를 출력 기준으로 삼아, 특정 시점의 반응이 아니라 누적된 공진 결과가 반영되게 설계했다.

- **Empirical Impact**: 13개 시나리오(경쟁 게임과 자연어 협상)에서 MAFP는 tournament strength와 robustness라는 상호보완 지표 모두에서 단일 라운드 및 다중 라운드 기준선들을 능가했다. 특히 debate나 self-reflection처럼 “iteration만 추가한” 접근은 성능 개선이 제한적이었고, ToM은 일부 개선되지만 robustness에서는 MAFP에 뒤졌다. 또한 MAFP-Last(aggregation 제거)는 약점이 드러나, 최신 iterate에만 greedy하게 반응하는 방식이 fictitious play의 핵심인 “과거 히스토리 기반 기대”를 잃기 때문임을 보여준다. 전반적으로 stance entanglement을 equilibrium 탐색으로 다루는 새로운 MAS 패러다임으로서, 전략을 행동 전 단계에서 강건하게 결정해야 하는 분야에 실질적 의미가 크다.



### Trade-offs in Medical LLM Adaptation: An Empirical Study in French QA (https://arxiv.org/abs/2606.19266)
- **Prior Approaches**: 의료 도메인 적응은 보통 CPT(continual pretraining)로 잡지식/임상 코퍼스를 학습하고, SFT(supervised fine-tuning)로 instruction–response를 맞추는 방식으로 이뤄져 왔다. 다만 기존 연구는 base model 초기값을 고정하거나 영어 벤치마크 중심으로 평가해, CPT·SFT 효과를 분리해서 비교하기 어려웠다. 또한 MCQA 위주라 생성형 OEQA에서의 실제 생성 품질 차이는 해석이 제한적이었다.

- **Core Contribution**: 이 논문은 프랑스 의학 QA를 케이스로 CPT, SFT, CPT+SFT를 “초기화(General/Instruct/Medical)”까지 체계적으로 바꿔가며 분리 실험한다. 모델 패밀리·크기·디코딩(greedy/제약)까지 함께 비교하고, MCQA와 OEQA를 같이 보되 OEQA 해석은 주의 깊게 진행한다. 결과적으로 계산 자원 제약 하에서 어떤 적응 전략을 우선할지에 대한 실무형 가이드라인을 제시한다.

- **Technical Challenges**: 핵심 난점은 적응 전략의 차이를 base model 선택 효과와 섞지 않고, 통계적으로도 유의한 비교를 만드는 것이다. 이를 위해 여러 모델 패밀리에 맞춘 정렬된 초기화 세트를 구성하고, MCQA는 constrained decoding과 EM/Hamming으로 재현성 있게 평가하며, OEQA는 ROUGE/BERTScore와 LLM-as-a-Judge(의사 코호트 기반 신뢰도 검증)를 결합해 측정한다. 또한 다중 비교 보정과 percentile bootstrap으로 유의성 검정을 수행해 “겉보기 성능”과 “실제 유의미한 개선”을 구분했다.

- **Empirical Impact**: MCQA에서는 CPT+SFT가 가장 자주 1등을 차지하지만, SFT 대비 이득이 작고 자주 통계적으로 유의하지 않았다; 따라서 라벨 데이터가 있을 땐 SFT가 강력한 기본값으로 정리된다. OEQA에선 SFT가 생성 품질을 악화시키는 경향이 있고, CPT가 겹침 기반 지표를 꾸준히 개선하지만 CPT+SFT는 불안정할 때가 많았다. 또한 프랑스 의학으로 적응하면 영어 벤치마크 성능이 개선되는 교차언어 전이가 관찰됐고, 번역 벤치마크는 정확도뿐 아니라 confidence까지 왜곡(과신/불확실성 감소)할 수 있어 메트릭 해석 주의가 필요하다는 결론을 강화한다.



### DreamReasoner-8B: Block-Size Curriculum Learning for Diffusion Reasoning Models (https://arxiv.org/abs/2606.19257)
- **Prior Approaches**: Autoregressive(AR) LLM은 긴 chain-of-thought(CoT)를 잘 다루지만, left-to-right 분해 때문에 추론 시 병렬성이 제한된다. 이를 보완하려 diffusion language models과 block diffusion이 제안됐고, 블록 내 병렬 denoising로 효율을 늘릴 수 있으나 오픈 reasoning-capable 확산 모델은 block size를 키우면 성능이 크게 떨어지는 문제가 반복됐다.

- **Core Contribution**: 본 논문은 오픈소스 블록 확산 reasoning 모델 DreamReasoner-8B를 공개하고, 학습·추론에서 block size가 long-CoT 성능에 미치는 영향을 체계적으로 분석한다. 핵심은 큰 블록으로 바로 학습하면 reasoning이 취약해지지만, block-size curriculum learning으로 fine-grained에서 coarse-grained로 점진 전환하면 다양한 추론 블록 크기에서 성능을 안정적으로 유지한다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘큰 블록 학습’이 intra-block 장거리 의존에 과적합되며, token 단위 순차 추론 능력을 잃게 되는 granularity gap(세분성 격차)이다. 논문은 이를 완화하기 위해 curriculum으로 먼저 작은 블록에서 로컬 인과 구조를 학습한 뒤 큰 블록을 점진 노출하고, 추가로 RelaxedConfidence 디코딩 프로브를 통해 이웃 토큰의 신뢰가 확보된 구간은 더 낮은 임계값으로 일찍 커밋해 TPF를 끌어올린다.

- **Empirical Impact**: 수학·코드 벤치마크에서 DreamReasoner-8B는 Qwen3-8B-Thinking 같은 대표적인 오픈 AR 모델과 경쟁 수준의 결과를 보이며, 특히 LiveCodeBench에서 pass@1 51.3%로 더 큰 SDAR-30B-A3B-Sci(29.0%)를 크게 앞선다. 또한 TPF 관점에서도 RelaxedConfidence가 thinking/answering 단계에서 각각 평균 22.5%, 54.5%의 처리량 향상을 보이면서 reasoning fidelity를 해치지 않아, 효율적인 diffusion 기반 reasoning의 실용 기반을 제시한다.



### RECOM: A Validity Discrimination Tradeoff in Automatic Metrics for Open Ended Reddit Question Answering (https://arxiv.org/abs/2606.19218)
- **Prior Approaches**: 기존 LLM 평가는 자동 지표(예: BERTScore, cosine similarity)를 “좋은 지표”로 단일 성질처럼 취급해 왔습니다. 하지만 이 논문은 open-ended, opinion-driven QA에서 지표가 동시에 요구받는 두 역할(진짜 정합성, 시스템 간 변별력)이 서로 충돌할 수 있음을 문제로 제기합니다.
또한 기존 연구들은 주로 시간 민감도나 사실 정답 같은 객관적 정답에 더 가까운 설정을 다뤄, 레딧 댓글처럼 합의가 정답인 환경에서 지표 거동이 어떻게 달라지는지 불명확했다고 짚습니다.

- **Core Contribution**: 논문은 RECOM(Reddit Evaluation for Correspondence of Models)이라는 오염(데이터 contamination) 없는 평가 데이터셋과, 임의 바닥값(random-derangement noise floor)을 결합해 자동 지표의 두 축(유효성, 변별력)을 분리 측정합니다.
또한 5개 7B~10B 오픈소스 LLM을 모든 레퍼런스 댓글에 대해 스코어링하고, 어떤 지표도 유효성과 변별력을 동시에 잘 달성하지 못한다는 정량적 결론을 도출합니다.
이 트레이드오프가 모델이 아니라 지표의 표현 설계(representation design)에서 비롯된다는 가설도 제안합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “같은 출력”을 기준으로 유효성과 변별력을 공정하게 비교하는 것이었습니다. 논문은 레포런스 댓글(특정 포스트의 depth-1 커뮤니티 답변)과 임의 순열 바닥값을 함께 사용해, 유효성(real-vs-random)과 변별력(모델 간 분리)을 공통 척도(Cohen’s d)로 측정합니다.
또한 응답 길이가 변별력에 끼치는 영향을 통제하기 위해, length confound 회귀 분석 및 잔차화(residualizing)를 수행해 BERTScore precision/F1의 모델 순위 성능이 길이 효과로 상당 부분 부풀려졌음을 확인합니다.

- **Empirical Impact**: 실험 결과, cosine similarity는 real-vs-random 구분에서는 강하지만(|d|≈2, 변별력은 |d|<0.1) 모델 랭킹에서는 거의 차이를 못 냅니다. 반대로 BERTScore precision은 원시 변별력에서는 앞서지만(최대 |d|≈0.63) 응답 길이를 통제하면 변별력이 |d|≈0.09 수준으로 붕괴해 유효성-변별력 트레이드오프가 재현됩니다.
또한 3명의 독립 LLM judge(서로 다른 개발사 모델)로도 유효성 축은 크게 재현되지만, 모델을 가려내는 변별력은 약하게 나타나 동일한 관찰이 사람 판단 대리에서도 유지됨을 보여줍니다.
논문은 따라서 지표를 단일 점수로 합치기보다 유효성과 변별력을 각각 보고, 임의 바닥값을 명시적으로 리포팅해야 한다는 실무 가이드를 제공합니다. 



### Language Models as Interfaces, Not Oracles: A Hybrid LLM-ML System for Pediatric Appendicitis (https://arxiv.org/abs/2606.19183)
- **Prior Approaches**: 기존 소아 충수염 진단 보조는 Alvarado Score, PAS처럼 점수 기반 규칙/모형에 의존하거나, 랜덤포레스트·gradient boosting 같은 지도학습을 사용해 tabular 변수로 위험을 예측해 왔다. 그러나 실제 임상 문서는 서사형 free-text로 기록되는 경우가 많아 tabular 입력을 만들기까지 변환 비용이 크고, 기관 간 데이터 시프트로 외부 성능이 흔들릴 수 있다. 한편 LLM을 진단 엔진으로 end-to-end로 쓰는 접근은 prompt 민감도, 정보 순서 영향, 그럴듯한 오답(hallucination) 위험 때문에 안전성이 약하다는 경고가 누적되고 있다.

- **Core Contribution**: ClaMPAPP은 LLM을 ‘진단 결정자’가 아니라 ‘인터페이스/특징 추출기’로 격리하고, 실제 위험 예측은 XGBoost 같은 검증된 ML 예측기로 수행하는 하이브리드 파이프라인을 제안한다. LLM은 note-like 서사에서 스키마 제약을 받는 임상 특징을 뽑고(필요 시 설명도 생성), 최종 appendicitis risk는 ML이 산출해 더 결정적이고 감사 가능(auditable)한 경로를 만든다. 또한 이 아키텍처는 충수염에 국한되지 않고, 서사형 문서와 tabular 예측기를 연결하는 ‘질병 불가지론적’ 설계로 확장 가능하다고 주장한다.

- **Technical Challenges**: 핵심 기술 과제는 LLM의 잘못된 값 생성(환각)과 추출 실패가 곧 진단 오류로 이어지는 것을 막는 것이다. 논문은 특징 추출 뒤에 deterministic Feature Validator를 두어 생리·임상 범위 제약을 위반하면 값을 잘라내지 않고 NaN(결측)으로 바꿔 입력 편향을 최소화하며, XGBoost의 결측 처리 능력을 활용한다. 더불어 서사 입력에 대한 견고성을 보기 위해 실제 EHR의 tabular 값을 템플릿으로 pseudo-노트 형태로 만들고 LLM rewriting 및 문장 순서 반전(permutation)까지 적용해 위치 편향까지 점검했다.

- **Empirical Impact**: 독일 병원 두 개의 독립 소아 코호트에서 내부·외부 검증을 수행한 결과, ClaMPAPP은 end-to-end LLM 기준선 대비 전반 성능이 더 좋았고 특히 missed appendicitis(거짓 음성) 비율을 줄이는 데 강점을 보였다. 또한 문장 순서를 섞었을 때 end-to-end LLM들은 민감도-특이도 균형이 불안정해지고 성능 저하가 컸던 반면, ClaMPAPP은 상대적으로 안정적인 안전 우선(triage safety) 프로파일을 유지했다. 이러한 결과는 “자연어 사용성은 LLM, 추론 신뢰성은 ML”로 역할을 분리하는 설계가 임상 의사결정 지원에서 더 적합할 수 있음을 실증적으로 뒷받침한다.



### Dango: A Strictly L1-Only Large Language Model for Studying Second Language Acquisition (https://arxiv.org/abs/2606.19170)
Comments:
          8 pages main text, 20 pages total including references and appendices

- **Prior Approaches**: 기존 연구는 L1→L2 순서로 언어모델을 학습해 전이(transfer) 효과를 보기 위해 BLiMP 계열의 문법 일반화나 surprisal 같은 지표를 주로 사용했다. 다만 소형 모델이거나 encoder-only 모델이 많아, 개방형 텍스트 생성과 실제 L2 시뮬레이터로 쓰기엔 제약이 컸다. 또한 웹 크롤링 기반 L1 말뭉치의 비의도적 영어 유입(언어 오염)이 통제되지 않아, 전이를 “데이터 때문”으로 착각할 위험이 있었다.

- **Core Contribution**: Dango는 1.8B 파라미터 decoder-only LLM으로, 일본어(L1)→영어(L2) 전이를 통제 연구할 수 있게 설계된 대형 모델이다. 핵심은 L1 단계에서 발생하는 L2 contamination을 줄이는 필터링 파이프라인을 제안하고, 그 뒤 LLM이 생성한 교재형 L2 학습 레슨으로 fine-tuning해 인간 학습 과정에 가까운 생성 능력을 학습시키는 것이다.

- **Technical Challenges**: 가장 큰 난제는 “일본어 단독”으로 기대되는 사전학습 말뭉치에 영어 문장이 체계적으로 섞여, 규모를 키울수록 영어 능력이 의도치 않게 강화될 수 있다는 점이다. 논문은 whitelist(일본 중심 문자 집합 기반)과 blocklist(연속 영어 단어 수, 라틴 문자 비율, 라인/문서 단위 제거)를 조합해 영어 노출을 줄이면서도 짧은 고유명사·일상 표현 같은 최소 노출은 남기는 방식으로 해결했다. 이후 난이도 레이블(CEFR-J) 기반으로 점진적 문장 길이/복잡도를 갖춘 레슨 데이터를 구성해, L1(일본어) 설명은 유지한 채 L2(영어) 산출을 학습하도록 학습 프롬프트를 설계했다.

- **Empirical Impact**: 평가 결과 Dango는 unfiltered 모델과 표준 multilingual 기준선보다 영어 생성의 전이 패턴이 더 인간 학습자와 가깝게 나타났고, LLM-as-a-judge 기반 UF(사용 빈도)·ER(오류율) 분포에서도 오류 양상의 일치도가 개선됐다. 특히 긴 영어 문장 생성처럼 필터링 전처리로는 직접 얻기 어려운 능력까지 확보해, 단순 역할극이 아닌 실제 L1→L2 전이 학습 효과를 지지한다. 모델·데이터·코드를 공개해 재현 가능한 computational SLA 연구와 학습자/교사 훈련용 가상 L2 시뮬레이터로의 활용 가능성도 함께 제시했다.



### Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams (https://arxiv.org/abs/2606.19111)
Comments:
          33 pages

- **Prior Approaches**: 기존 multi-agent LLM 연구는 debate, 역할 분담/분해, self-refinement처럼 지식 수준에서의 조정(무엇을 생각할지)을 바꾸는 방식이 주를 이룹니다. 하지만 평가가 최종 정확도에 치우쳐 있고, 실제로 ‘과정 수준 control이 언제 통하는가’를 분리해 측정한 사례는 드뭅니다. 또한 컨트롤러가 프롬프트처럼 모놀리식으로 설계되는 경우가 많아, 어떤 구성요소가 행동 차이를 만드는지(컴포넌트 단위 원인분해)도 깔끔히 검증하기 어렵습니다.

- **Core Contribution**: 이 논문은 multi-agent LLM 팀에서 ‘process-level coordination control’이 정확도를 더하는 조건을 측정 가능한 형태로 정식화하고, 그 조건이 team science의 contingency 예측과 일치하는지 검증합니다. 컨트롤러를 공통 action vocabulary(예: explore, revise, accept, synthesize) 위의 explicit control 정책으로 구현하고, 리더십 스타일(거래적/변혁적/상황적)을 그 정책으로 operationalize합니다. 결과적으로 “항상 이기기”가 아니라 “특정 조건에서만 가치가 생기는지”를 지도/이름표가 아닌 경계(boundary)로 다룹니다.

- **Technical Challenges**: 핵심 과제는 ‘정확도’ 대신 컨트롤러의 과정 차이를 직접 읽어내는 측정 설계이며, 저자들은 majority lock-in, 탐색률, round-0 오답 컨센서스에서의 recovery 같은 behavioral signatures를 1차 지표로 둡니다. 또 컨트롤러를 작은 명시적 action set으로 정의해 per-action ablation이 가능하게 만들었고, 임의 규칙(이론 없는 컨트롤)에서는 majority voting 수준으로 수렴해 ‘액션 구성’이 아닌 ‘이론 기반 규칙’의 역할을 분리합니다. 추가로 open-ended 수치형 과제의 추출 편향을 cross-round majority extractor로 통제해, 컨트롤 효과와 무관한 측정 노이즈를 줄입니다.

- **Empirical Impact**: 4개 task regime과 3개 open-weight 모델 계열에서, 어떤 컨트롤러도 전반적으로 정확도를 압도하지 못했는데 이는 contingency 관점의 null 결과와 일치합니다. 다만 round-0 독립 다수결이 신뢰롭지 않을 때에만 성능 이득이 나타났고, 그중에서도 recoverability가 가능한 영역에서 situational/transactional 계열이 기준선 대비 유의미하게 개선합니다(주로 round-0 majority가 흔들리는 조합에서 +8pp급 사례). 경계 probes(예: MATH-500 Level 5 확장, adversarial NLI, Winogrande, 도덕 판단)로도 “라운드0 신뢰도-회복 가능성-상호작용이 이미 복구하는지” 축이 재현되며, leadership substitutes/path-goal redundancy/situational readiness gap 같은 팀 과학 개념과 실측이 매핑됩니다.



### Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Scienc (https://arxiv.org/abs/2606.19051)
Comments:
          ASIST 2026

- **Prior Approaches**: 기존 연구 방법 자동 분류는 주로 제목과 초록(title/abstract)에 의존해왔지만, 초록만으로는 방법론적 단서가 충분히 담기지 않는 한계가 있었습니다. 한편 full-text를 그대로 쓰려 하면 문서가 지나치게 길고 중복 정보가 많아 모델이 핵심 신호를 놓치기 쉽습니다. 그 결과, full-text를 어떻게 ‘어느 부분을’ 효과적으로 활용할지가 오래된 병목이었습니다.

- **Core Contribution**: 이 논문은 full-text를 물리적 위치(물리적 position) 기준으로 구간 분할한 뒤, 구간들을 조합하는 segment combination strategy를 제안합니다. 즉, 방법론 정보가 문서 전반에 균등하게 퍼져 있다는 가정 대신, 특정 구간이 분별력(discriminative power)을 더 갖는다는 관찰을 분류 파이프라인에 반영한 것입니다. 또한 서지 메타데이터(bibliographic metadata)를 cross-segment 조합과 함께 쓰면 성능을 더 끌어올릴 수 있음을 보여줍니다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) full-text의 길이와 정보 중복을 줄이면서도 (2) 다중 레이블(multi-label) 분류에 필요한 방법론 신호를 놓치지 않는 구간 설계를 찾는 데 있습니다. 저자들은 텍스트를 위치 기반으로 나눈 뒤, 다양한 구간 조합을 여러 모델에 걸쳐 평가해 ‘어떤 구간을 어떻게 합칠지’를 실증적으로 최적화했습니다. 여기에 bibliographic metadata를 결합해 구간 간 정보의 보완성을 강화하는 방식으로 해결했습니다.

- **Empirical Impact**: Library and Information Science 분야의 대표 저널 JASIST, LISR, JDoc에서 1,954편 full-text를 주석한 코퍼스로 실험했으며, 구간 단독·조합별 분류 성능을 비교했습니다. 결과적으로 방법론 정보는 full-text에 고르게 분포하지 않고, middle-to-late 및 마지막 구간이 특히 더 큰 분별력을 보였습니다. 더 나아가 서지 메타데이터와 cross-segment 조합을 함께 적용할 때 분류 성능이 일관되게 향상되어, method retrieval·review generation 같은 지식 서비스에 직접적인 활용 가능성을 시사합니다.



### Sumi: Open Uniform Diffusion Language Model from Scratch (https://arxiv.org/abs/2606.19005)
- **Prior Approaches**: 확산 기반 언어모델은 AR(autoregressive) 모델의 대안으로 빠르게 확장돼 왔고, 특히 MDLM(masked diffusion language model)은 대규모로 스케일링되며 AR 기준선과 경쟁 성능을 보였다. 다만 MDLM은 마스킹 토큰을 채운 뒤 해당 위치를 다시 수정하기 어렵고, UDLM(uniform diffusion language model)은 이 제약을 풀어 더 유연한 생성과 self-correction 가능성을 노린다. 그런데 지금까지는 UDLM을 ‘처음부터(pretrained from scratch)’ 대규모 파라미터와 대규모 토큰 예산을 동시에 만족하며 학습한 공개 사례가 사실상 없었다.

- **Core Contribution**: 이 논문은 Sumi(“ink”)라는 완전 공개 7B UDLM을 제안하며, 단일 레시피로부터 scratch 학습을 1.5T 토큰까지 확장해 UDLM 스케일링의 기준선(reference point)을 만든다. Sumi는 지식·추론·코딩에서 같은 토큰 예산대 AR 모델들과 경쟁하지만, 상식(commonsense) 벤치마크에서는 상대적으로 약하다. 또한 가중치/체크포인트뿐 아니라 데이터 믹스 구성과 전체 학습 레시피를 공개해 재현성과 후속 연구를 촉진한다.

- **Technical Challenges**: UDLM은 모든 토큰 위치를 매 스텝 갱신할 수 있는 대신, 학습 목적(GIDD 기반)과 잡음(SNR) 파라미터화, 그리고 추론 평가 시 canvas 길이 같은 세팅이 성능·동작에 직접 영향을 준다. 저자들은 GIDD를 SNR-reparameterized 형태로 쓰고 log-SNR 범위를 제한하는 등 안정적인 학습을 목표로 설계했으며, 평가에서는 attention mask 부재 상황을 고려해 EOS/BOS 경계로 평가 프로토콜을 맞췄다. 또한 canvas 길이(기본 2048) 안팎에서 생성 fluency가 크게 달라짐을 관찰하며, 이 제약을 공정 비교의 핵심 조건으로 고정한다.

- **Empirical Impact**: 실험에서 Sumi는 같은 평가 프로토콜 하에서 일반 지식·코딩에서는 우수한 편이고, 수학/추론 영역에서도 Llama 2-7B 수준에 근접하거나 앞서는 결과를 보인다. 반면 PIQA·HellaSwag·WinoGrande 등 commonsense에서는 큰 격차가 나타나며, 교육 비중이 큰 데이터 믹스가 원인일 가능성을 제시한다(단, 직접 검증은 하지 않음). 더 나아가 추론-time 프로브로 confidence sampling이 토큰 commit 순서를 유도하고, 코딩 과제에서는 제한적 병렬 디코딩 이점을 주지만 self-correction(명시적 revision budget)은 거의 나타나지 않음을 방향성 있게 보여 주며, ‘네이티브 UDLM의 미지 영역’을 연구 의제로 끌어올린다는 점에서 의미가 있다.



### Enhancing Multilingual Reasoning via Steerable Model Merging (https://arxiv.org/abs/2606.19002)
Comments:
          12 pages, 7 figures, 8 tables. Accepted by ACL2026 Findings

- **Prior Approaches**: 기존 멀티링구얼 추론은 번역 기반(고품질 병렬데이터 파인튜닝, 외부 번역기 사용)이나 모델 머징 기반 접근이 주류였다. 모델 머징은 서로 다른 모델의 특징 공간을 정렬해 성능을 끌어올리지만, 고정된 one-size-fits-all 방식은 소스 모델 간 충돌을 입력별로 조정하지 못해 비효율적일 수 있다. 특히 LLM이 이미 강한 고자원 언어에서는 외부 멀티링구얼 신호 의존이 추론 능력을 해칠 수 있다는 점이 관찰된다.

- **Core Contribution**: 이 논문은 Steerable Model Merging(ST-Merge)로, 입력 특성에 따라 멀티링구얼 인코더와 reasoning LLM의 기여를 동적으로 조절하는 프레임워크를 제안한다. 핵심은 두 모델의 결합을 고정 가중치가 아니라 “스티어링 가능한” 가중치로 만들어, 저자원 언어는 멀티링구얼 정렬을 더, 고자원 언어는 LLM 내재 추론을 더 살리도록 유도하는 것이다. 그 결과, 언어별로 달라지는 최적 협업 패턴을 자동으로 찾아 성능 균형을 달성한다.

- **Technical Challenges**: 문제는 서로 다른 특징 공간(mT5 계열 멀티링구얼 표현 vs LLM 추론 표현)이 직접 결합되기 어렵고, 고정 머징은 모델 충돌을 억제하지 못한다는 점이다. ST-Merge는 (1) 멀티링구얼 특징을 LLM 공간으로 매핑해 정렬한 뒤, (2) gated cross-attention으로 Q=reasoning 특징이 K/V=정렬된 멀티링구얼 특징을 보도록 설계해 입력 조건에 따른 가중치(증폭/감쇠)를 학습한다. 또한 언어 ID 임베딩을 게이트에 주입하고, 가중치 초기화를 안정적으로 두기 위해 1 주변 중심화(예: 1+tanh 계열)를 사용해 학습/수렴을 돕는다.

- **Empirical Impact**: MGSM, MSVAMP, X-CSQA, XNLI 등 4개 벤치마크에서 21개 언어에 대해 ST-Merge가 여러 강력한 베이스라인을 일관되게 능가하며 평균적으로 +1.3%~+1.7% 수준의 개선을 보인다. 특히 Fix-Merge처럼 대칭 머징에서 생기기 쉬운 실패(저자원 언어에서 핵심 엔티티 누락/환각)를 ST-Merge가 비대칭 가중치로 교정하는 사례도 제시된다. 언어별 게이팅 패턴이 정확도와 연결되며, 고자원 언어의 추론 능력을 유지하면서 저자원 정렬을 강화하는 방향으로 모델이 스스로 수렴한다는 점에서 멀티링구얼 추론 머징 설계에 의미 있는 실증을 제공한다.



### G-IdiomAlign: A Gloss-Pivoted Benchmark for Cross-Lingual Idiom Alignmen (https://arxiv.org/abs/2606.18989)
Comments:
          Accepted to ACL 2026

- **Prior Approaches**: 기존 이디옴 연구는 주로 detection(관용적 사용 여부)이나 disambiguation(문맥에서 문자/비유 의미 선택)에 집중해 왔습니다. 또 cross-lingual idiom alignment를 다뤄도 보통 표면형 단서나 lexical overlap에 기대는 경향이 강해, 비유 의미를 제대로 매칭하지 못하거나 저자원 언어에서 성능 격차가 커지는 문제가 드러났습니다.

- **Core Contribution**: 이 논문은 gloss를 의미의 공통 기준점으로 쓰는 gloss-pivoted 벤치마크 G-IdiomAlign을 제안합니다. Wiktionary의 English gloss로 각 이디옴을 고정(앵커링)하고, 재현 가능한 high-confidence reference alignment set까지 구축해 진단형 평가가 가능하게 했습니다. 또한 Multiple-Choice Idiom Equivalence와 Gloss-Contrastive Generation( No-gloss vs With-gloss ) 두 프로토콜로 의미 피벗의 효과를 분리해 측정합니다.

- **Technical Challenges**: 핵심 난관은 (1) gloss 없이 생성하면 모델이 literal translation 같은 표면 통계에 치우치고, (2) gloss 기반 매칭도 임베딩 근사 때문에 잘못된 1:1 대응이 생길 수 있다는 점입니다. 저자들은 후보군을 gloss-embedding 공간에서 검색한 뒤 mutual nearest neighbors(MNN)와 언어쌍별 분포 기반 컷오프로 약한 페어를 제거해 잡음을 줄였고, 생성 평가에서는 결정적 파싱을 위해 정확히 한 개 이디옴만 출력하도록 통제했습니다. 추가로 attention 기반 진단을 통해 With-gloss가 어떤 내부 신호(특히 attention head)에 더 크게 영향을 주는지도 분석합니다.

- **Empirical Impact**: 다양한 LLM에서 공통적으로 literal translation 편향이 지배적 실패 모드로 확인됐고, 특히 저자원 타깃 언어일수록 악화됩니다. With-gloss는 embedding 기반 의미 프록시 아래에서 일관되게 성능을 올리지만 Acc@0.80 수준이 여전히 ‘모가 큰’ 것으로 나타나, 오픈 생성 공간에서 의미-동등 이디옴을 만드는 난이도가 큽니다. Qwen3-8B 분석에서는 With-gloss로 생성 품질이 좋아질수록 gloss anchoring이 강해지고, 조건 간 차이는 레이어 전반보다 attention head 쪽에 더 집중된다는 근거를 제시합니다.



### Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering (https://arxiv.org/abs/2606.18986)
- **Prior Approaches**: Time-MQA류 TSQA는 시계열을 BPE 기반 텍스트로 직렬화해 LLM에 넣는 방식이 많았지만, 숫자가 자리값이 아니라 빈도 기반으로 잘게 쪼개져 크기·스케일·추세 같은 수치 기하가 사라진다는 한계가 있다. 또 patch 기반 인코더는 고정된 윈도우 크기로만 표현해 특정 과거 시점(정확한 인덱스) 접근이 어렵고, 패딩/재분할 때문에 샘플링 레이트나 길이가 달라지면 전이 성능이 흔들린다.

- **Core Contribution**: CADE는 TSQA를 위한 입력 표현 단계에서 patch/직렬화 없이, 각 timestep을 LLM 임베딩 공간에 직접 임베딩으로 매핑하는 direct timestep embedding을 제안한다. 이어서 분류 데이터로부터 class-name 텍스트 앵커를 frozen 상태로 두고, 한 방향(one-directional) supervised contrastive loss로 시계열 임베딩을 언어적 의미 공간에 정렬해 멀티태스크 공통표현을 강화한다.

- **Technical Challenges**: 가장 큰 기술 난관은 LLM의 토크나이저가 연속 수치를 안정적인 metric 구조로 전달하지 못한다는 점이며, CADE는 BPE 파편화를 제거하기 위해 point-wise 선형 인코더+MLP projector로 각 값/인덱스를 1:1로 임베딩에 보존한다. 또한 분류 태스크에만 유효한 contrastive 신호를 다른 과제까지 공유해야 하므로, 분류 샘플만 풀링 앵커로 쓰되 text 앵커는 frozen으로 고정하고 시계열 측에만 그라디언트가 흐르는 설계를 통해 의미 정렬을 안정화한다.

- **Empirical Impact**: Time-MQA 벤치마크에서 CADE는 6개 TSQA 태스크 전반에 걸쳐 성능을 일관되게 개선했으며, open-source 및 proprietary LLM 기반 기준선 대비 우수한 결과를 보고한다. 특히 patch 기반·LoRA/FT·직렬화(BPE) 등 대표 대안을 함께 비교해, 표현 병목(토크나이제이션 파편화, 고정 그라뉼러리티)을 줄이는 것이 멀티태스크 TSQA 신뢰성 향상으로 이어진다는 점을 실증적으로 뒷받침한다.



### GraphPO: Graph-based Policy Optimization for Reasoning Models (https://arxiv.org/abs/2606.18954)
- **Prior Approaches**: RLVR은 최종 정답을 기준으로 이진 보상을 주고 정책을 최적화하지만, 보상이 sparse해 중간 단계 credit assignment가 어렵습니다. 또한 체인·트리 기반 샘플링은 서로 독립적으로 추론을 생성하거나, 접두사만 공유해 분기 이후에는 의미적으로 같은 상태를 중복 탐색하는 문제가 남습니다.
트리 방법은 분기 지점에서 더 촘촘한 신호를 제공하지만, 결국 가지(branch)를 독립 경로로 취급해 동일(semantically equivalent) 상태의 정보 공유·비교를 제대로 하지 못합니다.

- **Core Contribution**: GraphPO는 롤아웃을 DAG(Directed Acyclic Graph)로 표현해, 추론 단계는 edge로, 경로를 요약한 중간 의미 상태는 node로 둡니다. 의미적으로 같은 상태를 equivalence class로 가상 병합해 같은 suffix(후속 추론)를 공유하고, 중복 확장에 쓰이던 예산을 더 유망한 frontier 탐색으로 재배분합니다.
또한 결과(outcome)에서 공정(process) 신호를 뽑기 위해 graph reward와 dual-group graph advantage를 설계해, 정확성·효율성 모두에 대한 step-level 감독을 제공합니다.

- **Technical Challenges**: 핵심 난제는 (1) 독립/분기 경로가 만들어내는 중간 추론 중 의미적으로 같은 상태를 안정적으로 감지해 병합해야 한다는 점과 (2) 그렇게 병합했을 때도 advantage 추정 분산과 credit assignment를 악화시키지 않아야 한다는 점입니다. GraphPO는 node 임베딩 유사도와 ancestor-descendant 관계 제약으로 equivalence detection을 수행하고, 병합된 상태 간에는 pooled score를 공유하며 step reward를 재구성합니다.
더 나아가 correctness group(같은 상태에서 나가는 단계 비교)과 efficiency group(같은 상태에 도달한 다른 입력 경로 비교)을 분리해, 다음-스텝 공유가 분산을 줄이면서도 더 짧고 유효한 경로를 선호하도록 유도합니다.

- **Empirical Impact**: 실험에서는 Qwen 계열 3개 모델 및 에이전트/탐색 벤치마크에서 GraphPO가 chain·tree 기반 방법과 outcome-only RLVR(예: GRPO, DAPO) 대비 일관되게 우수한 성능을 보였습니다. 특히 동일 token budget 또는 동일 response/trajectory budget 조건에서 개선이 유지되어, 단순 샘플 수 증가가 아닌 더 효율적인 롤아웃 활용과 낮은 advantage 분산이 효과를 만든다는 점을 뒷받침합니다.
또한 entropy 및 응답 길이 분석에서 GraphPO는 중복 경로 과증폭을 줄이며 더 느리게 수렴하고, 더 짧은 응답(추론 효율)을 만드는 경향을 보여 PRM(과정 라벨) 없이도 process supervision을 graph 구조만으로 얻을 수 있음을 시사합니다.



### SenFlow: Inter-Sentence Flow Modeling for AI-Generated Text Detection in Hybrid Documents (https://arxiv.org/abs/2606.18946)
Comments:
          16 pages, 4 figures, 9 tables

- **Prior Approaches**: 기존 문서 AI 생성 텍스트 탐지(AGTD)는 문서 전체를 분류하거나 단일 문장만을 독립적으로 판별하는 방식이 많았다. 특히 S-AGTD는 SeqXGPT, SenDetEX 등에서 문장 의존성을 거의 모델링하지 않아, 하이브리드 문서에서 나타나는 생성 구간의 연속성과 문장 경계의 점진적 스타일 변화를 놓친다.

- **Core Contribution**: 이 논문은 하이브리드 문서용 S-AGTD 벤치마크 MOSAIC과, 문장 그래프 기반 구조 예측 모델 SenFlow를 제안한다. MOSAIC은 PubMed와 XSum의 1.6만 문서를 DeepSeek-V3.2(리즌 모델)와 Kimi K2(채팅 모델)로 혼합 생성하되, perplexity-consistency filter로 과도한 표면 단서를 제거해 최신 생성기에 대한 평가 공백을 메운다.

- **Technical Challenges**: 문장 단위 신호가 약한 상황에서 이웃 문장 간 흐름을 어떻게 함께 추론할지가 핵심 난제다. SenFlow는 문장을 노드로 하는 sentence graph에 GCN으로 inter-sentence propagation를 수행하고, CRF로 라벨 연속성을 강제하는 구조를 단일 document-level 패스로 구현해 이전의 독립 문장 분류 한계를 해결한다.

- **Empirical Impact**: 실험에서 SenFlow는 MOSAIC 전 프로토콜에서 state-of-the-art를 달성하며, 특히 cross-domain transfer에서 평균 Macro-F1이 4.15 pp 크게 개선됐다. 또한 perplexity-consistency로 겉 단서는 맞춰도 생성기별 문장 길이의 구조적 격차가 남아(문장 단위 탐지에도 여전히 유효) 벤치마크의 난이도 설계와 방법의 실효성이 함께 확인됐다.



### As Easy as Rocket Science: Assessing the Ability of Large Language Models to Interpret Negation in Figurative Languag (https://arxiv.org/abs/2606.18922)
Comments:
          16 pages, 16 figures; for associated code and data see this https URL To be published in Transactions of the Association for Computational Linguistics

- **Prior Approaches**: 기존 연구는 은유·직유 해석을 NLI나 문장 패러프레이징 같은 과제로 평가해 왔으며, 많은 경우 fine-tuning을 통해 성능이 크게 개선된다고 보고돼 왔다. 또한 negation 자체가 모델에 취약하다는 결과가 축적돼 있지만, figurative language와 negation이 동시에 등장할 때의 상호작용은 상대적으로 덜 탐구돼 왔다. Fig-QA처럼 덜 관습적인 figurative language를 다루는 데이터가 있었음에도, negation 유형까지 세분해 out-of-the-box 해석을 체계적으로 본 연구는 부족했다.

- **Core Contribution**: 이 논문은 Fig-QA에 metaphor/simile뿐 아니라 negation, tense, concreteness를 새로 라벨링해 복합 현상(특히 negation+figurative)의 해석 능력을 분리해 측정한다. 아울러 Fig-QA를 바탕으로 literal negation을 소규모로 구성해 figurative language 없이 negation 효과만 고립해 비교한다. 다양한 언어 모델을 fine-tuning 없이 그대로 평가해, “두 현상 동시 등장”이 어떤 병목을 만드는지 정면으로 드러낸다.

- **Technical Challenges**: 핵심 난관은 (1) LLM이 autoregressive 로그확률/프롬프트 응답에 따라 선택 편향이 생길 수 있고, (2) 모델 종류(embedding 기반 vs Llama/GPT 계열)에 따라 동일한 평가 신호를 맞추기 어렵다는 점이다. 저자들은 prompt style을 mid-phrase(예: “In other words”) 중심과 question-answer(두 후보 중 무엇이 더 적절한지 질문) 방식으로 나눠 비교하고, Llama는 출력 변동성 때문에 log-likelihood 기반 판별을 사용한다. 그 결과 prompt style이 성능을 크게 좌우하며, connector-free mid-phrase는 특히 성능을 크게 떨어뜨림을 확인한다.

- **Empirical Impact**: 실험에서 전반적으로 negation과 figurativeness의 결합이 ‘특정한’ 어려움으로 나타났고, 유형별로도 not/antonym 등 negation 양상이 성능 격차를 만든다. 특히 question-answer 프롬프트에서는 여러 모델이 인간 성능에 근접하지만, negating simile에서 성능 하락이 두드러져 상호작용 병목이 관측된다. 또한 SBERT 임베딩 분석(PCA)에서는 PC3가 negation과 강하게 연관되며, tense·concreteness도 일부 영향이 있으나 negation 효과가 상대적으로 더 크다는 결론을 제시해 향후 평가/프롬프트 설계에 실질적 기준을 제공한다.



### SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration (https://arxiv.org/abs/2606.18902)
- **Prior Approaches**: 컨텍스트 엔지니어링은 파라미터 업데이트 없이 AI 성능을 끌어올리는 주요 수단으로 자리 잡았고, 자동 프롬프트 최적화(APO)도 확산됐다. 다만 최근 연구는 “textual gradients”가 실제 gradient처럼 작동하지 않는다는 점을 근거로, APO를 블랙박스 탐색으로 보는 관점이 힘을 얻고 있다.

- **Core Contribution**: 이 논문은 프롬프트 공간에서의 확률적 탐색을 위한 프레임워크 SPO(Stochastic Prompt Optimization)를 제안한다. 또한 error-informed random search, 유전 알고리즘 기반 탐색, 그리고 에이전트가 진단을 안내하는 SAGE(SPO via Agent-Guided Exploration)를 비교해, 어떤 오류 유형과 랜드스케이프 구조가 조합될 때 유리한지까지 정리한다.

- **Technical Challenges**: 핵심 기술 과제는 프롬프트-성능 간 목표 함수가 불연속적이고 잡음이 큰 ‘탐색 문제’라는 점에서, 효과적인 탐색 절차를 설계하는 것이다. 저자들은 오류를 반영한 무작위 탐색, 진화 연산자(변이/교배 등)로 탐색을 확장하는 방식, 그리고 진단 코드 실행을 포함한 멀티에이전트 파이프라인으로 탐색을 정보화해 SAGE의 실효성을 확보한다.

- **Empirical Impact**: 세 가지 벤치마크에서 단일 전략이 항상 우월하진 않았지만, 성능은 에러 타입과 랜드스케이프의 상호작용에 의해 달라짐이 확인됐다. 특히 SAGE를 정신건강 챗봇에 적용해 연속 최적화 환경에서 8번의 개별적으로 잡음이 있는 A/B 테스트를 누적·통계적으로 보정하며 다음날 retention에서 견고한 개선을 보고했고, 질적 진단과 정량 검증의 결합이 오픈엔디드 대화 최적화에서 효과적임을 강조한다.



### Learning Robust Pair Confidence for Multimodal Emotion-Cause Pair Extraction (https://arxiv.org/abs/2606.18893)
Comments:
          11 pages, 3 figures, 5 tables

- **Prior Approaches**: 기존 multimodal emotion-cause pair extraction(MECPE) 학습은 valid 후보에 대해 pair-level cross entropy로 양/음 쌍을 독립적으로 다루는 경향이 강했다. 그 결과 emotion마다 여러 cause가 경쟁하는 상황에서 필요한 상대적 confidence geometry(행(row) 단위 랭킹)가 충분히 제약되지 않아, gold pair가 hard negative와 근접하거나 비정답 문맥에 기대는 취약점이 생긴다.

- **Core Contribution**: 이 논문은 이런 취약점을 pair-confidence brittleness로 규정하고, “학습 중에 pair confidence의 표면을 어떻게 형태화할지”를 직접 목표로 삼는다. 이를 위해 training-only 프레임워크인 RPCL(Robust Pair Confidence Learning)을 제안하며, gold pair가 같은 emotion 행에서 강한 비정답 원인보다 확실히 우선하고, 비-gold 문맥이 교란돼도 예측이 안정적으로 유지되게 만든다.

- **Technical Challenges**: 문제는 (1) 경쟁하는 후보들 사이에서 gold를 hard negative보다 밀어내는 row-conditioned 제약을 세우는 것과 (2) 라벨을 보존하는 방식으로 ‘깨진 문맥’에 대한 견고성을 동시에 학습하는 것이다. RPCL은 top-k hard negative를 선택한 뒤 confidence-difference margin ranking(CDMR)으로 gold–hard negative 간 adaptive margin 갭을 강제하고, gold에 관여하지 않는 발화를 확률적으로 zero-out하는 corrupted view에서 같은 pair label로 학습한 후 clean 예측과의 분포 정렬까지 수행해 pair-confidence의 안정성을 확보한다. 특히 inference 시에는 원래 pair scorer와 decoding 파이프라인을 그대로 사용한다.

- **Empirical Impact**: ECF, MECAD, MEC4의 complete text-audio-video 설정에서 RPCL은 matched base 대비 three-seed 평균 Pair F1을 각각 2.58~2.83%p 개선하고, Pair AUPRC도 전반적으로 향상시켰다. 또한 진단 분석에서 gold–negative confidence gap 확대와 margin-violation 심각도 감소가 관측되어, 성능 향상이 단순한 예측 편향이 아니라 의도한 confidence 표면의 개선과 일치함을 보여준다. 전반적으로 “정답 재현”뿐 아니라 “신뢰도 구조 학습”을 훈련 목표로 명시화하는 접근이 MECPE 분야에 효과적이라는 점을 실험적으로 입증했다.



### Improving Medical Communication using Rubric-Guided Counterfactual Recommendations (https://arxiv.org/abs/2606.18889)
Comments:
          4 Tables, 8 Figures

- **Prior Approaches**: 기존 텍스트 기반 원격진료 연구는 환자 피드백을 의료 정확성보다 ‘의사소통 품질’의 간접 지표로 보고, 이를 예측·분석해왔다. 일부는 Language Model을 이용해 새 의료 응답 생성이나 자동 리라이팅까지 시도하지만, 의료 조언 자동화나 의사 통제 약화 문제가 남는다. 또한 피드백 향상을 위한 ‘무엇을, 얼마나’ 고쳐야 하는지에 대한 해석 가능성이 제한적이었다.

- **Core Contribution**: 이 논문은 LM-guided counterfactual recommendation 파이프라인으로, 의료 콘텐츠를 건드리지 않으면서 tone, personalization, actionability, completeness 같은 해석 가능한 소통 속성만 최소 변경해 긍정 피드백 확률을 높이는 방법을 제안한다. 추천은 환자-의사 상호작용 메타데이터와 함께 해석 가능한 피처로 예측되며, 최종 수정/수용은 의료진이 그대로 제어한다. 즉 “자동 의학 조언”이 아니라 “커뮤니케이션 개선 타깃”을 제시하는 중간 지점을 만든다.

- **Technical Challenges**: 핵심은 (1) 피드백과 연결된 의미 피처를 안정적으로 발견·정렬하고, (2) 피처 값의 ‘순서형(ordinal)’ 변화만으로 제한된 편집 비용 하에서 반사실 탐색을 수행하며, (3) 선택 모델에 과적합된 반사실 효과를 독립적으로 검증하는 것이다. 이를 위해 데이터셋 수준 automatic feature discovery를 telemedicine 피드백에 맞게 적용하고, feature별 grounded prompt refinement로 추출 신뢰도를 높인 뒤, 정책 모델이 “낮은 비용의 순서형 변화” 후보를 열거해 최적 변경을 고른다. 마지막으로 auditor 모델들을 독립적으로 붙여 추천이 다른 만족도 추정기에서도 유지되는지 검증해 self-confirmation 위험을 줄였다.

- **Empirical Impact**: 실험 결과, 추천은 독립 auditor 기준으로 평균 +6.41%의 긍정 피드백 확률 증가를 보였고, 93.31% 추천에서 평균 변화가 0 이상이었다. 또한 편집 예산(budget)이 커져도 성능 이득은 점진적이어서, 소수의 작은 커뮤니케이션 결함 보정만으로 대부분의 예측 개선을 얻는다는 실용적 신호가 확인됐다. 의료진이 최종 문구를 통제하면서도 피드백을 끌어올릴 수 있는 방식이라는 점에서, ‘패시브 분석 vs 자동 생성’ 사이의 현장 친화적 방향성을 제시한다.



### Efficient Financial Language Understanding via Distillation with Synthetic Data (https://arxiv.org/abs/2606.18875)
- **Prior Approaches**: 기존 금융 감정분석은 FinBERT처럼 사람 라벨이 있는 데이터로 사전학습·파인튜닝을 수행하는 경우가 많았지만, 금융 도메인은 기밀성과 전문가 라벨 비용 때문에 라벨 수집이 제한적이다. 한편 instruction-following LLM을 distillation이나 synthetic supervision으로 압축하려는 연구(Self-Instruct, Alpaca, Orca 등)는 라벨 비용을 줄이지만, 도메인(금융) 특성을 반영한 seed 선정과 합성 데이터 다양성 설계가 상대적으로 약했다. 결과적으로 일반-purpose distillation에 머물러 저자원 금융 텍스트에서의 일반화와 안정성 향상이 일관되지 않았다.

- **Core Contribution**: 이 논문은 GPT-4o의 instruction-following 지식을 소형 인코더 학생 모델(DistilBERT, ModernBERT 등)로 옮기되, 최소 실데이터(12~105개)를 seed로 쓰는 효율적 프레임워크를 제안한다. seed를 임베딩 기반 k-평균으로 클러스터링해 대표 문장을 고르고, structured few-shot prompting으로 합성 문장을 생성한 뒤, 학생을 real+synthetic 코퍼스로 fine-tuning한다. 특히 더 복잡하고 잡음이 많은 금융 소셜 미디어 도메인에서는, 완전한 synthetic-seed 코퍼스로 학습한 compact 모델이 teacher(GPT-4o)까지 능가할 수 있음을 보인다.

- **Technical Challenges**: 핵심 난제는 (1) 라벨이 적을 때 대표성과 다양성을 갖춘 seed를 고르는 문제, (2) 합성 데이터가 라벨-정렬을 유지하면서도 금융 문체·표현의 변이를 충분히 담도록 하는 문제다. 저자들은 Sentence-BERT 임베딩을 사용해 k-means로 의미 영역을 분산 커버하는 coreset-style seed를 선택하고, 교차-감정 대비(P1), 동일 라벨 재표현(P2), 다중 seed 기반 일반화(P3) 등 3종 structured prompting 템플릿을 결합해 9배 내외로 확장된 합성 코퍼스를 만든다. 또한 파인튜닝 안정화를 위해 일부 저층 freezing과 early stopping을 적용해 과적합과 학습 불안정을 완화한다.

- **Empirical Impact**: 실험은 Financial PhraseBank(형식적 문장)와 Twitter Financial News Sentiment(짧고 잡음 많은 소셜 담론)에서 수행됐고, 소형 모델이 teacher와의 격차를 크게 줄이며 성능을 확보한다. PhraseBank에서 ModernBERT는 clustering+합성(full synthetic-seed) 조건에서 95.15% accuracy, 94.63% macro-F1을 기록하며, 라벨 데이터의 원래 인력 수요 대비 매우 적은 분량으로도 강한 결과를 낸다. Twitter에서는 clustering 기반 seed 선택이 random 대비 평균 +3~7 F1 이득을 주고, 특히 ModernBERT가 GPT-4o zero-shot을 77.14%/71.14%로 추월해 “도메인 맞춤형 합성 증류”의 실용성과 영향력을 입증한다.



### Approximate Structured Diffusion for Sequence Labelling (https://arxiv.org/abs/2606.18856)
- **Prior Approaches**: 시퀀스 라벨링은 입력 문장의 각 토큰에 라벨을 부여하는 작업으로, 기존에는 신경망으로 파라미터화한 Linear-Chain CRF가 널리 쓰였다. 다만 CRF는 인접 라벨(bigram)처럼 유한한 의사결정 범위 가정이 표현력을 제한해 장거리 의존성이 필요한 경우 성능이 떨어질 수 있다. 한편 diffusion은 언어 생성에서 무한한 맥락 조건부 생성을 돕지만, 전형적으로 노이즈가 섞인 출력에서 토큰별 denoiser를 학습하는 방식이라 구조적 제약을 CRF 수준으로 직접 활용하기는 어렵다.

- **Core Contribution**: 이 논문은 시퀀스 라벨링에 structured prediction( CRF)과 discrete diffusion을 결합해, 전체 라벨 시퀀스를 조건으로 하는 CRF 학습을 제안한다. 핵심 아이디어는 denoiser가 ‘정답 라벨’이 아닌 ‘노이즈가 섞인 라벨 시퀀스’에 조건되도록 설계해 장거리 라벨 상호작용을 보게 하면서도 인접 라벨 선호(구조 제약)는 유지하는 것이다. 또한 diffusion 디코딩은 반복 샘플링이 필요하다는 점을 고려해, CRF 추정을 Mean-Field로 근사해 효율을 확보한다.

- **Technical Challenges**: 기술적 난관은 diffusion의 반복 디코딩 루프에서 CRF의 전역 정규화/정확한 분포 샘플링을 그대로 쓰기 어렵다는 데 있다(계산 비용과 병렬화 제약, 학습 시 메모리 부담). 저자는 CRF를 diffusion denoiser의 일부로 넣되, CRF 분포를 Mean-Field로 근사하거나 mean regularisation 계열의 대안을 통해 분포 계산을 저렴하게 만들었다. 학습은 diffusion 단계 t를 샘플링하고 노이즈 라벨에서 posterior를 맞추도록 변분 하한을 최적화하며, denoising loss를 함께 더해 수렴 안정성을 확보한다.

- **Empirical Impact**: POS tagging 벤치마크(Universal Dependencies 4개 데이터셋)에서 제안한 structured diffusion(예: Diffusion-MF)이 unigram diffusion과 CRF 기반 기준선 전반에 대해 정확도를 끌어올렸다. 특히 CRF 기준선 대비 오류 감소율이 16.54%로 보고되며, 파라미터를 늘릴수록 baseline보다 더 잘 스케일되는 경향도 관찰된다. 즉, 장거리 라벨 의존성을 diffusion으로 보강하되 Mean-Field 근사로 효율을 유지하는 전략이 실험적으로 유효하다는 점에서 의미가 있다.



### Aligning Implied Statements for Implicit Hate Speech Generalizability with Context-Bounded Semi-hard Negative Mining (https://arxiv.org/abs/2606.18852)
- **Prior Approaches**: 암묵적 혐오 발화는 노골적 욕설 대신 비꼼, 완곡어, 수사적 질문처럼 의도(intent)가 맥락에 숨는 경우가 많아 표면 표현만으로는 부족합니다. 기존 supervised contrastive learning(SCL)은 암시된 진술(implied statement)을 positive로 삼거나 라벨 내 공유 의미를 positive로 만들지만, 미니배치 내 대부분을 negative로 밀어내면서 ‘가까운 반대 라벨’을 false negative로 취급해 국소 이웃이 흔들리고 도메인 전이가 약해질 수 있습니다.

- **Core Contribution**: 논문은 ImpSH라는 triplet 기반 학습 프레임워크를 제안해, positive는 (가능하면) post-implication 쌍에 정렬하고 negative는 ‘맥락에 한정된’ semi-hard negative만 골라 학습 신호를 안정화합니다. 또한 AugSH 변형을 통해 증강 기반 multi-view에서 어떤 이득이 생기는지 분리해, implied statement 감독이 추가로 제공하는 효과를 점검합니다.

- **Technical Challenges**: 핵심 난제는 “너무 가까운 반대 라벨을 negative로 잘못 밀어내면” 표현 공간이 왜곡된다는 점이며, 이를 위해 배치 내 전체 negative를 밀어내는 방식 대신 마진 근처의 near-confusion만 negative로 선택합니다. 구현적으로는 cross-entropy 분류 손실을 유지하면서 cosine 거리 기반 triplet objective에 context-bounded semi-hard mining을 결합하고, one positive per anchor로 positive 구성 방식을 ImpSH/AugSH에 맞게 분기합니다.

- **Empirical Impact**: IHC, SBIC, DynaHate에서 BERT와 HateBERT를 사용해 cross-domain 전이에 초점을 맞춘 결과, ImpSH는 대부분의 설정에서 표준 SCL 계열을 견조하게 대체하거나 개선하며 특히 일부 전이 방향에서 꾸준히 강한 성능을 보였습니다. Alignment와 uniformity(로컬 양성 쌍은 더 조밀하게, 전체 분포는 무너지지 않게) 분석에서도 ImpSH가 transfer에서 유리한 임베딩 기하를 형성하는 경향이 확인되었고, nearest-neighbor 사례는 도메인 이동 시의 대표적 false negative 양상을 보여주며 의도 분리 목적의 타당성을 뒷받침합니다.



### ScholarSum: Student-Teacher Abstractive Summarization via Knowledge Graph Reasoning and Reflective Refinemen (https://arxiv.org/abs/2606.18850)
- **Prior Approaches**: 기존 과학 초록 생성(abstractive summarization)은 주로 추출 기반으로 사실은 잘 보존하지만 문장 단편화로 논리 흐름이 약해지기 쉽다. 이후 BART/T5 같은 PLM 기반 생성은 문장 유창성은 크게 개선했지만, 긴 문서에서 섹션 간 통합과 근거 기반의 사실 일치가 흔들린다. LLM prompting·RAG·그래프 플래닝은 유망하지만, 결국 한 번의 생성 과정에서 생길 수 있는 누락/환각을 구조적으로 검증·수정하는 장치가 부족하다는 한계가 남는다.

- **Core Contribution**: ScholarSum은 학생-교사(student–teacher) 글쓰기 과정을 닮은 계층적 reflective 그래프 프레임워크로, 유창성과 factual faithfulness를 동시에 겨냥한다. 문서를 세만틱하게 분할해 계층형 지식 그래프를 만들고, 멀티 레이어 community 구조로 매크로 논리/테마를 먼저 계획한 뒤 학생이 초안을 작성한다. 이후 교사(Reviewer)가 유사 논문 검색 기반으로 초안을 평가하고, 근거가 없는 내용은 evidence re-retrieval과 재작성으로 반복 교정해 품질 기준을 충족할 때까지 진행한다.

- **Technical Challenges**: 핵심 난제는 (1) 긴 과학 문서의 논리 토폴로지를 충실히 반영하는 계층 표현을 구성하는 일과 (2) 전역 구조가 있어도 로컬 근거를 끌어와 수치·설정 등 미세 사실을 맞추는 일이다. ScholarSum은 LLM으로 엔터티/typed relation을 뽑아 knowledge graph를 만들고 community 알고리즘으로 테마 단위를 안정적으로 구성해 전역 계획을 제공한다. 또한 학생이 그래프 이웃의 핵심 triplet을 evidence 앵커로 삼아 초안을 생성·수정하도록 하되, 교사는 in-domain 유사 초록을 KNN으로 찾아 기준을 보정한 뒤 누락/부정확/논리 오류를 구체 피드백으로 돌려 major/minor revision을 트리거한다.

- **Empirical Impact**: ArXiv와 PubMed 벤치마크에서 ROUGE·METEOR·BERTScore는 물론, MiniCheck 기반 factuality 지표에서 ScholarSum이 기존 강력한 기준선 대비 일관된 개선을 보였다고 보고한다. 특히 PubMed처럼 용어와 실험 디테일이 촘촘한 도메인에서 반복 검증 루프 덕분에 정확성이 유지되는 경향이 강조된다. 아블레이션은 계층 그래프와 evidence grounding이 모두 필요하며, 특히 교사 기반 iterative refinement가 환각 감소에 큰 비중을 차지하고 재현성(run별 분산)도 낮아 안정적인 운영 가능성을 시사한다.



### Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning (https://arxiv.org/abs/2606.18831)
Comments:
          15 pages, 6 figures, 12 tables

- **Prior Approaches**: 기존 long-context reasoning 개선은 주로 강화학습(RL)에서 reward 설계에 집중했지만, 핵심 신호가 부족해 긴 입력에서 증거를 찾는 단계가 정체되기 쉬웠습니다. 동시에 고품질 학습 데이터는 retrieval·multi-evidence synthesis·reasoning을 폭넓게 커버하기 어렵고, 합성 데이터도 범위가 좁거나 closed-source인 경우가 많았죠. 알고리즘 중심 접근은 auxiliary reward나 최적화 변형을 통해 해결을 시도했지만, 데이터 다양성이 병목이었습니다.

- **Core Contribution**: 이 논문은 long-context RL을 ‘데이터 중심’으로 재정의하며, 복잡한 reward engineering 없이도 단순하지만 효과적인 데이터 레시피만으로 성능을 크게 올릴 수 있음을 보였습니다. 레시피는 retrieval, multi-evidence synthesis, reasoning의 세 가지 상호보완적 능력을 각각 겨냥한 8개 데이터셋(총 약 14K 예시)을 조합합니다. 또한 이 개선이 에이전트형 과제로도 전이돼, agent-tuned 모델에 계속 RL을 하며 GAIA와 BrowseComp 점수가 추가로 상승함을 보여줍니다.

- **Technical Challenges**: 핵심 기술 난제는 긴 문맥에서 (1) 키워드 매칭 같은 지름길을 쓰지 않고 올바른 증거를 ‘찾아내는’ 것, (2) 여러 단서들을 빠짐없이 통합해 합성하는 것, (3) 긴 입력 위에서 복잡한 계산/추론을 유지하는 것입니다. 저자들은 fuzzy/paraphrase 기반 needle 설계와 near-duplicate 구분형 multi-needle, 파생 속성 집계형 multi-evidence synthesis, 단서 누락 시 실패가 확정되는 incomplete coverage, UUID 체인 은닉형 KeyChain 같은 방식으로 각 실패 모드를 데이터에서 직접 유도합니다. 이후 Group Relative Policy Optimization(GRPO) 기반의 minimal outcome-based 학습을 붙이고, 데이터 간 보상 스케일/분산 차이를 줄이기 위해 task-balanced sampling과 task-level advantage normalization으로 학습 경쟁을 완화합니다.

- **Empirical Impact**: 3개 Qwen 모델(Qwen3-4B/8B/30B-A3B)에서 7개 long-context 벤치마크를 실험한 결과, 평균 향상 폭이 +7.2/+3.2/+6.4점으로 나타나 기존 long-context RL 학습 세트를 능가했습니다. 특히 LBv2, AA-LCR, DocFinQA처럼 ‘holistic’에 가까운 추론형 벤치마크에서 개선이 두드러졌고, 모델 크기와 컨텍스트 변형에도 성능 이득이 비교적 일관되었습니다. 더불어 학습 컨텍스트를 넘어선 길이에서도 이득이 유지되는 패턴을 보여, 특정 길이에 과적합된 skill보다 길이-일반적 long-context reasoning 능력을 길렀다는 점이 의미 있습니다.



### Beyond Scalar Scores: Exploring LLM-based Metrics for Clinical Significance Evaluation in Radiology Reports (https://arxiv.org/abs/2606.18797)
Comments:
          Under Review

- **Prior Approaches**: 기존 평가는 BLEU·ROUGE-L 같은 어휘 기반이나 BERTScore·RadGraph·CheXbert 등 연속 점수로 품질을 환산해, 임상적으로 중요한 오류와 무해한 표현 차이를 분리하기 어렵다는 한계를 드러냈다. LLM-as-judge 계열도 참고문헌과 생성문을 비교해 점수/에러를 뽑지만, “민감하게 잡되 과도하게 과잉 경고”하는 경향 때문에 robustness가 낮아졌다. 특히 ReEvalMed에서 대다수 LLM 평가자는 discrimination은 높지만 robust 간에는 불균형이 크게 나타났다.

- **Core Contribution**: 논문은 임상적으로 유의미한 오류를 ‘탐지하는 능력(D)’과 ‘무해한 변화를 과도하게 벌점주지 않는 능력(R)’을 분리해 경계가 어디서 무너지는지 측정한다. 또한 ReEvalMed의 12개 error aspect와 omission/fabrication/inaccuracy 3개 error type에 맞춘 4,000개의 임상 오류 주입(report pair) 데이터를 합성해, 유의미/무해 경계에 대한 학습 신호를 제공한다. 이를 바탕으로 Qwen3-8B·MedGemma-4B를 SFT와 RL(특히 DPO)로 후학습해 D–R 경계를 더 날카롭게 만드는 경량 평가 메트릭을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM이 오류 스팬을 찾아내는 것과, 그 오류가 임상적으로 ‘유의미/무해’인지 분류하는 것을 동시에 안정적으로 맞추는 데 있었다. 연구진은 one-pass(스팬+심각도 동시 출력)와 two-pass(스팬 탐지 후 심각도 판단 분리) 프롬프트를 비교했지만, two-pass는 종종 bias를 단순 재배치해 전체 균형을 일관되게 개선하지 못했다. 이를 해결하기 위해 임상적으로 검증된 오류 사양으로 REF–TGT 페어를 합성하고, SFT로 출력 포맷 학습 후 DPO로 ‘유의미↔무해’ 경계 자체를 최적화해 discrimination 과 robustness 간 격차를 줄였다.

- **Empirical Impact**: 실험에서 대부분의 LLM 평가자는 D는 매우 높게 나오는데 R이 낮아 discrimination bias(무해한 재표현을 유의미 오류로 과잉 라벨링)가 반복적으로 관찰됐다. 합성 데이터로 학습된 경량 메트릭은 32B급 medical LLM을 능가하거나 유사한 수준을 보이며, 특히 Qwen3-8B의 two-pass 학습은 D=78.5%·R=70.5%로 D–R 균형을 개선했다. 또한 two-pass 추론은 항상 이득이 아니어서, 비용 민감 배포에서는 one-pass 기반 학습 메트릭이 실용적이며 two-pass는 D–R 밸런스가 최우선인 상황에 한해 선택하는 게 낫다고 결론짓는다. 데이터셋과 메트릭을 공개하겠다고 밝혀, 임상 유의성 평가의 재현성과 확장성에도 의미가 있다.



### RedactionBench (https://arxiv.org/abs/2606.18782)
- **Prior Approaches**: 기존 PII 마스킹 평가는 개인정보 ‘추출’ 방식과 ‘프라이버시 의미’를 섞어버려, 문맥에 따라 달라지는 위반 여부를 제대로 구분하지 못했습니다. 예를 들어 같은 전화번호라도 의료기록에 있는지, 공개된 번호인지에 따라 문제의 성격이 달라질 수 있지만, 대부분의 벤치마크는 이를 동일 취급합니다. 그 결과 redaction(가림)과 entity recognition(개체 인식)을 사실상 같은 문제로 취급하는 한계가 드러납니다.

- **Core Contribution**: 논문은 contextual integrity(문맥적 무결성) 관점에서 ‘무엇을 가릴지’가 사용 주체·목적·상황에 따라 달라진다는 점을 전면에 내세워, 이를 평가에 반영하는 RedactionBench를 제안합니다. RedactionBench는 11개 도메인에서 200개 문서를 수작업 라벨링했으며, 대부분은 실제 출처에서 시드되어 현실성을 높였습니다. 또한 문맥적 redaction의 난제를 정량화하기 위한 R-Score를 함께 도입합니다.

- **Technical Challenges**: 핵심 기술적 난제는 문맥 redaction이 본질적으로 주관적이라 모델 평가가 ‘정답 정밀도’만으로는 해결되지 않는다는 점입니다. 이를 위해 R-Score는 의미적으로 유사한 redaction은 동일하게 취급하고, 휴대전화 번호 마스킹처럼 얕은 포맷 차이(마스킹 스타일 차이)는 점수에서 무력화합니다. 그럼에도 다양한 모델군(NER, small language models, 에이전틱 도구가 달린 프론티어 모델)에서 contextual redaction이 여전히 미해결임을 보여줍니다.

- **Empirical Impact**: 실험은 여러 모델 35종을 대상으로 PII redaction 성능을 비교하고, 평가 결과가 모델 패밀리 간에도 크게 갈리며 contextual redaction이 특히 어렵다는 점을 확인했습니다. 또한 RedactionBench에서 80명 이상 사용자 기반의 인간 평가를 수행했는데, 필수 redaction에 대한 합의(89.4%)와 안전 텍스트 보존(94.1%)은 높지만 문맥 redaction 합의는 47.7%로 크게 낮았습니다. 이 ‘인식 격차’를 근거로 R-Score가 contextual ambiguity와 strict precision을 분리해 측정하도록 설계된 의미가 입증되며, RedactionBench는 향후 프라이버시 보존 시스템의 표준 기준선(baseline)으로 활용될 전망입니다.



### Lost in a Single Vector: Improving Long-Document Retrieval with Chunk Evidence Aggregation (https://arxiv.org/abs/2606.18781)
Comments:
          Code is available at this https URL

- **Prior Approaches**: 기존 dense retrieval은 문서를 한 덩어리(단일 벡터)로 인코딩한 뒤 query–document 유사도로 순위를 매기는 구조라, 긴 문서에서 짧게 결정적 증거 구간이 약화되면 성능이 급락할 수 있습니다. 이를 줄이기 위한 passage retrieval이나 late-interaction(예: ColBERT 계열)은 인덱싱/스코어링 인터페이스 자체를 바꾸는 경우가 많아, ‘문서 인코딩 단계’의 문제를 분리해 다루기 어렵다는 한계가 있습니다. 또한 long-context에서 성능 저하가 관찰되더라도, 그 원인이 문서-side early compression인지 명확히 계량하기가 쉽지 않았습니다.

- **Core Contribution**: 이 논문은 문서-side early compression 실패 모드를 Evidence Dilution Index(EDI)로 계량화해, ‘gold 문서 안의 최강 chunk 증거 대비 단일 문서 벡터가 얼마나 뒤처지는지’를 진단합니다. 이를 바탕으로 DICE(Document Inference via Chunk Evidence)를 제안하며, 학습 없이 문서를 여러 chunk로 나눠 frozen encoder로 각각 인코딩한 뒤 다시 하나의 문서 벡터로 집계합니다. 핵심은 query 인코딩과 one-query-one-document 검색 인터페이스를 그대로 유지하면서, 문서 인코딩 방식만 바꿔 압축 전에 국소 증거를 보존하는 데 있습니다.

- **Technical Challenges**: 가장 큰 기술적 쟁점은 chunk 단위 정보를 보존하면서도, chunk 간 집계(aggregation)가 query와 상호작용하지 않는 제약에서 효과가 나야 한다는 점입니다. DICE는 chunk를 독립 인코딩하되 chunk 내부에 local position index를 재설정해 각 chunk가 self-contained하게 처리되도록 하고, mean/max/top-k(embedding norm 기준) 같은 query-independent pooling 규칙 중에서 특히 mean pooling과 적절한 chunk granularity가 성능을 좌우함을 ablation으로 확인합니다. 또한 overlap은 평균 성능에 일관된 이득이 적고 비용이 커서 기본값을 non-overlap으로 둡니다.

- **Empirical Impact**: LongEmbed에서 4개 백본에 걸쳐 DICE는 단일 벡터 기준선보다 일관되게 retrieval 성능을 개선하며, 특히 4k 토큰 초과 구간 같은 ‘어려운 긴 문서’에서 최대 폭의 상승을 보였습니다(예: Dream의 Passkey >4k: 30.0→90.0, Needle >4k: 23.3→74.0). 12,779개 필터드 샘플에서 DICE의 EDI가 단일 벡터보다 낮은 비율이 92.8%로, 개선이 단순 상관이 아니라 EDI 관점의 기전(증거 희석 완화)과 맞물림을 입증합니다. FollowIR에서도 문서-side chunk aggregation이 전이되며, 장문/국소 증거 상황에서 문서 인코딩 단계가 ‘실용적이고 덜 탐구된 레버’임을 강조합니다.



### Output Vector Editing for Memorization Mitigation in Large Language Models (https://arxiv.org/abs/2606.18767)
- **Prior Approaches**: 기존 연구들은 모델이 학습 데이터의 연속을 그대로(또는 유사하게) 재생한다는 ‘verbatim memorization’ 위험을 줄이기 위해, 학습 단계에서 예방하거나(재학습 필요), 추론 단계에서 디코딩을 필터링하거나(런타임 비용), 또는 사후 unlearning으로 전반적 거동을 바꾸는 방식 등을 제안해 왔다. 또 locate-then-edit 패러다임을 적용해 책임 부품을 찾고 가중치를 수정하는 접근이 있지만, sequential memorization에는 답 위치/대체 답 같은 구조가 부족해 성공이 제한적이었다. 특히 neuron-level 완화는 locate에 해당하는 뉴런을 0으로 꺼서 편집을 구현하는데, 뉴런이 활성화 여부만이 아니라 여러 특징을 중첩(superposition)으로 담고 있다는 점에서 ‘너무 파괴적’이라는 한계가 지적된다.

- **Core Contribution**: 이 논문은 memorized continuation을 억제하기 위해 ‘output vector editing’을 제안한다: 활성은 그대로 두되, 특정 MLP 뉴런의 출력 벡터를 제한된 최적화로 아주 최소하게 수정해 해당 잔차 스트림 기여를 vocabulary 공간의 distractor 쪽으로 돌린다. 핵심은 기존의 activation-zeroing이 뉴런이 표현하던 여러 방향을 통째로 제거하는 반면, 이 방법은 출력 벡터의 한 방향만 rank-one 업데이트로 바꿔 편집의 파괴성을 줄인다는 주장이다. 또한 OLMo-7B에서 6,831개의 memorized sequence를 체계적으로 채굴하고, 편집이 실제로 ‘어디를 찾았는지’보다 ‘무엇을 수정했는지(출력 벡터)’에서 성패가 갈린다는 점을 실험으로 분리해 보여준다.

- **Technical Challenges**: 가장 큰 기술적 도전은 ‘책임 뉴런을 정확히 locate’하는 것만으로는 충분치 않고, 같은 뉴런이 여러 특징을 중첩으로 인코딩하는 상황에서 출력 영향을 어떻게 안전하게 방향 전환할지 설계하는 것이다. 저자들은 L1 proximity 기반의 뉴런 기여도 추정과 unembedding 방향(top/bottom rank) 필터링으로 편집 후보 뉴런을 좁힌 뒤, 라그랑주 승수로 닫힌형(closed-form) rank-one 가중치 업데이트를 계산해 미분 없이 per-sequence로 출력 벡터를 수정한다. 아울러 EOS, redact marker, next-best, suppress의 4가지 편집 모드로 success–locality trade-off를 조절하고, 약 14%의 MLP-only 불가 케이스는 attention head 제거 실험으로 ‘보완 경로’로서 attention이 관여함을 드러낸다.

- **Empirical Impact**: 평가에서 동일한 locate된 뉴런을 쓰되 activation을 0으로 꺼버리는 기준선 대비, output vector editing은 최대 87.9% 억제(OLMo-7B 기준)를 달성하며 0 ablation 대비 최대 2.7배 격차를 보여 ‘로케이션만’으로는 설명되지 않는 향상을 입증한다. 4개 모드를 앙상블하면 96.5%의 memorized sequence를 포괄하지만, 단일 권장 설정(next-best, k=5)에서는 81.5% 억제와 함께 catastrophic locality failure가 없다는 점이 실용성을 강화한다. 또한 편집 모드의 success-locality 트레이드오프가 모델 전반(360M~7B)으로 옮겨가며, 계열(family)보다 크기(size)에 따라 성패가 스케일되는 경향과 MLP 전용 편집의 경계(약 14%)가 제시되어, 향후 MLP+attention 하이브리드 파이프라인 설계에 직접적인 시사점을 준다.



### LegalWorld: A Life-Cycle Interactive Environment for Legal Agents (https://arxiv.org/abs/2606.18728)
- **Prior Approaches**: 기존 법률 에이전트 평가는 상담·서면 작성·재판 등 업무를 ‘독립 과제’로 나눠 개별 성능만 측정하는 경우가 많습니다. 또한 시뮬레이터들은 대체로 매 시나리오를 동일한 기준 정답(ground truth)으로 재초기화해, 이전 단계 산출물이 다음 단계 절차·인과에 미치는 영향(오류 증폭, 사실/주장 누락)을 구조적으로 반영하기 어렵습니다.

- **Core Contribution**: 이 논문은 중국 민사소송을 5개 단계(총 7개 서브시나리오)로 연결한 life-cycle 상호작용 환경 LegalWorld를 제시합니다. LegalWorld는 75,309쌍의 실제 판결을 바탕으로 상담→소장/답변서 작성→1심 변론→항소 작성/답변→2심 재판까지 ‘하나의 사건 사슬’로 진행되며, 이후 단계에서도 동일 사건의 상태가 유지되도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 (1) 단계 간 절차적 상태 전이와 인과 연결을 누락하지 않으면서, (2) 고객·변호사·판사처럼 역할마다 지식 범위와 발화 제약을 다르게 두고, (3) 증거 제출·서면 작성·법정 절차 등 도구 지원을 각 단계에 맞게 제공하는 것입니다. 논문은 in-scenario local memory와 stage-end용 global case memory를 분리하고, 단계별 Skill/Tool 라이브러리로 가시성·액션 경계를 게이팅해 다음 단계로의 누출/누락을 줄이는 방식으로 문제를 해결합니다.

- **Empirical Impact**: LongJud-Bench로 전체 5단계 역량을 평가한 결과, LegalWorld 궤적은 절차 충실도와 역할 일관성에서 높은 신뢰도를 보였으며(법조 배경 평가자 18,992 평점), LLM-as-Judge 점수도 보수적인 하한으로 작동함이 확인됐습니다. 또한 모델 간 역량 차이는 집계 점수로는 잘 드러나지 않지만, 상담·서면·법정 옹호 각 능력/단계로 분해하면 뚜렷한 편차가 나타났고, 단일 백본이 전 구간에서 일관되게 우세하진 않았습니다.



### Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish (https://arxiv.org/abs/2606.18717)
- **Prior Approaches**: BPE, WordPiece, Unigram 같은 frequency-driven subword 토크나이저는 튀르키예어에서 접미사가 의미를 담는 구조를 무시해 과도 분절(fertility 증가)과 정렬 손실을 만든다. 특히 WordPiece는 diacritics를 제거하고 TurkishTokenizer는 canonical re-harmonization으로 문자열을 바꿔 decode(encode(w))=w를 보장하지 않는 경우가 생긴다. Morfessor나 Zemberek 같은 기존 형태소 분할·분석은 정렬 측면은 돕지만, 언어모델 생성에 필요한 가역성(생성 시 복원 무결성)과 결합해 모두를 만족하긴 어렵다.

- **Core Contribution**: 논문은 튀르키예어에 대해 무손실(정규화 없이 표면 문자열 보존), 형태소 인지, 그리고 임베딩 생성까지 한 번에 수행하는 신경 형태소 경계 모델 Morpheus를 제안한다. Morpheus는 각 문자 사이 경계 확률을 학습하면서도 추론 시에는 exact 분할을 산출하고, 정규화가 없어서 decode(encode(w))=w가 구조적으로 성립한다. 또한 토크나이징을 위한 동일한 forward pass에서 단어 단위의 structured word embedding까지 함께 출력한다.

- **Technical Challenges**: 핵심 기술 과제는 경계 확률(연속값)을 이용해 형태소 분할을 “학습 가능하게” 만들되, argmax/threshold 같은 비분화 연산 없이 discrete 형태소를 복원하는 것이다. Morpheus는 Poisson–binomial 동적계획법으로 per-position boundary 확률로부터 soft morpheme membership을 미분 가능하게 계산하고, 학습에서는 soft 할당으로 그라디언트가 흐르게 하며 추론에서는 경계가 hard하게 복원되도록 설계했다. 더불어 형태소 역할이 root 기준 상대 위치에 좌우된다는 점을 반영해 RoPE로 상대 오프셋을 attention에 주입하고, 같은 forward pass에서 segment pooling으로 임베딩을 만든다.

- **Empirical Impact**: 실험에서는 가역성/무손실성 축을 명확히 분리해 평가했는데, Morpheus는 bits-per-character(BPC) 최저(1.425)로 reversible 토크나이저 중 성능이 가장 좋았고, subword 계열 대비 gold morphological alignment도 크게 향상(MorphScore macro-F1 0.61 vs 약 0.32)됐다. 임베딩 평가에서도 frozen Morpheus 벡터가 root-family 검색에서 MAP 0.85, same-root verification에서 ROC-AUC 1.00으로 상위권을 보였으며, BGE-M3와 BERTurk를 초과한다. 다만 NER나 case/number probing처럼 문맥·굴절 의존 과제에서는 BERTurk류 contextual encoder가 여전히 우세해, 논문은 Morpheus의 root-centric geometry가 강점과 한계를 동시에 만든다고 해석한다.



### LLMs Struggle to Measure What Distinguishes Students of Different Proficiency Levels: A Study of Item Discrimination in Reading Comprehension Assessmen (https://arxiv.org/abs/2606.18709)
- **Prior Approaches**: 그동안 LLM을 활용한 교육 평가 연구는 주로 item difficulty(난이도) 예측에 집중해 왔습니다. 난이도는 정답 확률 같은 집계 성격이라서, LLM의 정답률·추론 성능만으로도 어느 정도 신호를 기대할 수 있지만, discrimination(변별도)은 고능력/저능력 응답 패턴의 “상대적 변화”를 요구합니다.

- **Core Contribution**: 이 논문은 LLM이 item discrimination을 zero-shot으로 추정할 수 있는지, 즉 인간 사전검사에서 계산된 변별도와의 Human-AI alignment를 직접 검증합니다. 42개(상용+오픈웨이트) 모델을 대상으로 (1) 콘텐츠 기반의 직접 변별도 예측과 (2) LLM 정답을 synthetic student response로 보고 Classical Test Theory(CTT)로 보정하는 두 경로를 비교합니다.

- **Technical Challenges**: 핵심 기술 난제는 LLM이 “정답을 잘 맞히는 능력”은 있어도, 능력 구간별로 어떤 오류를 더 내는지(변별에 필요한 구조화된 오류 패턴)를 재현하지 못한다는 점입니다. 직접 예측은 순위 정렬이 거의 되지 않았고(일부는 오히려 역방향), CTT 보정은 비-무작위 신호는 얻었지만 persona(저·중·고 proficiency) 프롬프트로도 정답률 변별이 충분히 분화되지 않아 값이 체계적으로 과대추정되는 miscalibration이 나타났습니다.

- **Empirical Impact**: 실험 결과 direct discrimination prediction의 최고 Spearman 상관은 0.152, response 기반 CTT calibration의 최고는 all-persona synthetic pool에서 0.241에 그쳤습니다. 또한 변별도 기반 item screening(저변별/고변별 항목 회수)에서도 direct 예측은 무작위 수준에 가까웠지만, CTT 보정은 특히 저변별 항목에서 더 나은 회수 성능을 보여 “부분 대체 가능하지만 완전한 프리테스팅 대체는 아직 어렵다”는 결론을 강화합니다.



### TW-LegalBench: Measuring Taiwanese Legal Understanding (https://arxiv.org/abs/2606.18699)
Comments:
          10 pages, 2 figures, To appear in ICAIL 2026

- **Prior Approaches**: 기존 법률 벤치마크는 주로 common-law권 데이터에 치우치거나(MMLU, LegalBench) civil-law라도 번체/간체 중국어권을 일부만 커버해(예: LawBench, LawShift) 대만처럼 특정 관할의 법리까지 정밀 평가하기가 어려웠습니다. 또한 다수의 벤치마크가 객관식 위주로 구성돼 실제 변론·논증 같은 개방형 과정을 반영하지 못하고, 법 조항 범주도 지나치게 뭉뚱그려 오차 원인을 세분화하기 힘든 한계가 있었습니다.

본 논문은 TW-LegalBench가 이런 “관할 불균형·세분화 부족·개방형 평가 공백”을 동시에 겨냥한다고 설명합니다.

- **Core Contribution**: TW-LegalBench는 대만의 민법계 전통에 맞춘 공개 공식 코퍼스 기반 벤치마크로, 대만 법률 추론을 평가할 수 있는 실데이터 3종(MCQ/OEQ/LJP)을 한 프레임에 모읍니다. 특히 객관식은 조항(법규) 단위로 43개 법 유형을 라벨링하고, 개방형 에세이는 공식 채점 루브릭을 LLM-as-Judge로 분해 평가하며, 형사 판결은 범죄 유형별 결과 예측(LJP)으로 현실 정렬을 점검합니다.

즉, “정확한 조항 식별-논증 구성-판결 결과 추정”의 단계적 역량을 관할 특이적으로 측정하려는 것이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 (1) 대만 법조문을 정확히 연결해 평가할 수 있는 고해상도 라벨링(조항 계층: 條/項/款/目)과 (2) 개방형 답안의 채점 변동성을 줄이면서 루브릭에 기반한 공정한 점수화를 동시에 달성하는 것입니다. 논문은 조항 단위 수작업 어노테이션과 함께 OEQ는 루브릭 포인트별로 이산 라벨/점수를 출력하는 decomposed LLM-as-Judge 방식을 쓰고, 두 저지(gpt-5, claude-sonnet-4.5)의 평균으로 편향을 완화합니다.

또한 LJP에서는 양형·형량 수치 평가를 위해 정규화 로그거리(징역)와 절대/상대 허용 오차(구금·벌금·집행유예)를 조합해, 단순 텍스트 유사도만으로는 포착되지 않는 실무 기준을 반영하도록 설계했습니다.

- **Empirical Impact**: 13개 LLM을 평가한 결과, 자격시험(변호사 1차·자격) 기준을 넘는 모델은 있었지만(합격률 11%), 판사·검사 수준에서는 크게 떨어져(합격률 1~2%) “자격 단계별 역량 격차”가 드러났습니다. LJP에서는 평결 유형과 형량 예측 성능은 어느 정도 보였으나, 정확한 조항(Article) 인용에는 취약해 Type I/II 오류 양상이 확인되며 신뢰도 높은 법문 생성이 여전히 어렵다는 결론으로 이어집니다.

특히 전통적으로 대만/중국어 특화 코퍼스를 더 학습한 모델이 조항 인용 정확도에서 상대적 강점을 보였지만, OEQ·MCQ 성능 패턴을 함께 볼 때 “법리 추론”보다 학습 데이터의 암기 효과 가능성도 제기됩니다.



### RegMix-D: Dynamic Data Mixing via Proxy Training Trajectories (https://arxiv.org/abs/2606.18663)
Comments:
          Work in progress

- **Prior Approaches**: 기존 데이터 혼합 비율 선택은 학습 전체에 걸쳐 단일 static mixture를 고정하는 경우가 많았습니다. DoReMi는 기준 모델 대비 과잉 손실이 큰 도메인을 가중하며, RegMix는 proxy 모델 여러 개로 mixture→검증 손실을 회귀해 최적 혼합을 탐색한 뒤 그 혼합을 전 학습에 적용합니다. 하지만 이런 접근은 학습 단계에 따라 최적 구성이 달라질 수 있다는 ‘동적 변화’를 충분히 반영하지 못합니다.

- **Core Contribution**: 이 논문은 RegMix를 time-varying(시간에 따라 바뀌는) 혼합으로 확장한 RegMix-D를 제안합니다. 핵심 아이디어는 proxy run에서 얻는 손실 ‘엔드포인트’뿐 아니라 ‘loss trajectory’ 전체를 학습에 활용해, 여러 학습 시점에서의 최적 혼합을 예측한다는 점입니다. 또한 offline(학습 전 스케줄 생성)과 online(학습 중 관측 손실로 적응) 두 가지 배치로 바로 배포 가능하게 설계했습니다.

- **Technical Challenges**: 문제는 proxy의 손실 곡선 정보가 target 학습의 동적 혼합 결정에 전이될지, 그리고 동적 예측이 오류 누적으로 이어지지 않을지였습니다. RegMix-D는 연속 구간에서 (t, m, l) 상태를 입력으로 다음 구간의 손실을 예측하는 로컬 전이 형태의 회귀모델을 학습해, 각 switch point마다 후보 mixture를 탐색·갱신하도록 만들었습니다. Offline은 재귀 예측로 인한 누적오차가 우려돼 online 변형은 target의 관측 손실을 입력으로 써 다음 결정을 고정해 안정성을 확보합니다.

- **Empirical Impact**: 실험에서는 Pile 25B 토큰, 1B 타깃 모델에서 13개 downstream 작업 평균 성능이 RegMix 및 DoReMi를 일관되게 개선했습니다. 특히 128개 proxy 모델(=RegMix 프록시 연산의 25%)만 써도 RegMix를 넘었고, proxy 단계에서 손실 곡선을 더 잘 활용한 것이 성능 차이로 연결되는 양상이 관찰됩니다. 온라인 변형이 offline보다 대체로 더 좋은데, target의 실제 관측 손실에 조건을 거는 것이 동적 혼합 적응을 더 정확하게 만든다는 해석을 제시합니다.



### The Wrong Kind of Right: Quantifying and Localizing Misfired Alignment in LLMs (https://arxiv.org/abs/2606.18656)
- **Prior Approaches**: 기존 post-training 기반 LLM 정렬(alignment) 연구는 주로 편향을 줄이기 위해 RLHF 같은 방법으로 위험한 추론이나 고정관념 기반 가정을 억제하는 데 초점을 두어 왔습니다. 그 평가는 “명시적 근거가 부족한 모호한 상황에서 스테레오타입을 쓰는가”를 중심으로 진행되는 경우가 많았습니다. 그런데 이 논문이 지적하듯, 실제 의사결정에서는 인구통계 단서가 있어도 맥락 증거로 정답이 분명한 경우가 존재하며, 이 “증거 정합성(evidence grounding)”은 상대적으로 덜 검증됐습니다.

- **Core Contribution**: 이 논문은 정렬된 모델이 스테레오타입에 민감한 입력에서 오히려 “근거가 지시하는 결론”을 덮어쓰는 실패 모드 misfired alignment를 정식으로 제안합니다. 이를 정량화하기 위해 BBQ에서 유도한 대조쌍으로 구성된 VETO(2,032 contrastive pairs) 벤치마크와, 정답과 반대되는 target에서 실패하는 비율을 점수화하는 Misfired Alignment Rate(MAR)를 도입합니다. MAR은 모델이 대비(contrast)에서는 맞히지만 스테레오타입 타깃(target)에서는 틀리는 경우를 “정렬이 명시적 증거를 override”하는 징후로 측정합니다.

- **Technical Challenges**: 핵심 기술적 난점은 “정답이 맥락에 의해 동일하게 결정되는” 상황에서, 정렬이 정확한 증거 기반 추론을 어떻게 방해하는지를 분리·원인추적하는 것입니다. 논문은 (1) VETO의 대비쌍 설계로 증거는 동일하게 유지한 채 단지 대상 집단만 바꿔 실패를 정의하고, (2) alignment-priming 실험으로 ‘정렬 규범 문구’만 선행해도 MAR이 증폭되는 인과 효과를 보여주며, (3) logit lens 및 attention head ablation 같은 기계적 분석으로 마지막(후반) 레이어에서 evidence-supported 답이 억제되는 late-layer suppression 패턴을 확인합니다.

- **Empirical Impact**: 25개 개방/폐쇄 LLM을 VETO로 평가한 결과, 인간은 MAR 0.0%를 보였지만 모델들은 모두 4.7%~18.9% 범위의 비자명한 MAR을 보이며 misfired alignment가 일관되게 나타났습니다. 또한 priming은 모델 전반에 걸쳐 MAR을 크게 키울 수 있어, 개별 예제의 우연이 아니라 safety-related framing이 오류를 유도·증폭할 수 있음을 시사합니다. 메커니즘 분석에서는 실패 케이스에서 중간 레이어는 “yes” 쪽으로 선호가 나타나도 최종 레이어에서 “no”로 넘어가며, 이 억제가 소수의 alignment-specific attention head에 의해 매개된다는 점이 밝혀져 정렬의 품질을 재설계할 필요를 강하게 제기합니다.



### PEC-Home: Interpretation of Progressively Elliptical Commands in Smart Homes (https://arxiv.org/abs/2606.18636)
Comments:
          Accepted by ACL 2026 Findings

- **Prior Approaches**: 기존 LLM 기반 홈 어시스턴트는 “불충분/모호한 명령”을 대화 맥락이나 도구 호출로 보정하는 쪽에 초점을 둔다. 하지만 실제 사람-사람 대화에서 공통 기반이 쌓일수록 정보가 점진적으로 생략되는 현상(점진적 ellipsis)은, 현재 시스템이 명령을 ‘명시적 vs 모호함’ 같은 정적 범주로만 취급하면서 충분히 반영되지 않는다.

- **Core Contribution**: 이 논문은 점진적으로 생략되는 명령을 해석해 기기 제어로 정확히 매핑하는 작업을 새 과제로 정식화하고, 이를 위한 최초의 시뮬레이션 홈 데이터셋 PEC-Home을 제안한다. PEC-Home은 여러 사용자 환경에서의 기준 충돌(참조 모호성)과 시간이 지남/환경 변화에 따라 선호가 달라지는 문제(의도 모호성)를 함께 모델링한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 현재 명령만으로는 무엇을 지칭하는지(참조)와 (2) 무엇을 하려는지(의도)가 점점 더 불명확해지는 데다, 관련 대화 맥락이 다수 사용자·잡음 메모리에 섞일 수 있다는 점이다. 저자들은 가상 홈에서 장치-메서드-파라미터를 정의하고 명령의 생략 수준을 단계적으로 줄이도록 생성했으며, GPT-4o로 자연어를 만들고 휴먼-인더-루프 검증으로 데이터 품질을 확인했다.

- **Empirical Impact**: 실험 결과, GPT-4o를 포함한 여러 LLM은 점진적으로 ellipsis된 명령만 주어졌을 때 의도된 작업 실행 정확도가 크게 떨어졌다. 또한 RAG(대화 히스토리 검색), 외부 tool 통합, fine-tuning을 적용해도 고생략 수준에서는 완전 명령 대비 성능 격차가 해소되지 않아, 단순한 메모리/도구/학습만으로는 한계가 있음을 보여준다.



### PragReST: Self-Reinforcing Counterfactual Reasoning for Pragmatic Language Understanding (https://arxiv.org/abs/2606.18624)
Comments:
          First two authors contributed equally. Code and models: this https URL

- **Prior Approaches**: 기존 연구는 문맥 의존적 추론(함축, 함의, 전제, 지시, 메타포, 말의 의도 등)을 LLM이 잘못 해석해 ‘문자 그대로’ 받아들이는 취약성을 자주 보고해 왔다. 개선 시도는 주로 task-specific 선호 튜닝(예: DPO)이나 teacher가 만든 rationale/선생 데이터 증류처럼 외부 감독에 기대는 경우가 많아 확장성과 일관성이 한계로 지적된다. 또한 pragmatics는 정답을 검증할 만한 결정적 신호가 약해 self-improvement의 보상 설계가 까다롭다.

- **Core Contribution**: 이 논문은 counterfactual reasoning(관측된 발화와, 그 의미가 의도라면 화자가 선택했을 법한 대안 발화의 대비)을 학습의 중심 원리로 삼은 self-supervised 프레임워크 PragReST를 제안한다. PragReST는 인간 라벨 데이터나 더 강한 teacher로부터의 distillation 없이, 모델이 생성한 pragmatic QA를 정리한 뒤 counterfactual reasoning trace를 SFT와 reinforcement learning(GRPO)으로 내재화하도록 훈련한다. 핵심은 counterfactual 스캐폴딩을 학습 타깃 구성에만 쓰고, 추론 단계에서는 원문 입력만으로 같은 절차가 작동하게 만든다는 점이다.

- **Technical Challenges**: pragmatics는 같은 문장이라도 맥락·화자 목표·사회적 기대에 따라 의미가 달라져 정답 판별이 불안정하며, 전형적인 RLVR처럼 결정적 verifier 신호를 만들기 어렵다. PragReST는 (1) 모델이 상황·질문·의도된 답을 생성하고 (2) self-judge로 저품질/모호/비라이선스 사례를 걸러낸 뒤 (3) first-token confidence margin 기반으로 상위 신뢰도 데이터를 남기는 방식으로 데이터 품질 문제를 완화한다. 이어 SFT에서는 counterfactual script로 ‘대안 대비’가 포함된 reasoning trace를 생성·필터링해 절차를 증류하고, GRPO 단계에서는 self-judged correctness 보상으로 결과에 대한 강화 신호를 준다.

- **Empirical Impact**: PragReST는 PragMega, Ludwig, MetoQA, AltPrag 4개 벤치마크에서 backbone 대비 일관되게 향상되며, Qwen3-8B와 Qwen3-14B에서 instruct 대비 정확도(절대) 개선이 각각 5.37%, 5.50%로 보고된다. 특히 non-counterfactual 변형과 비교하면, counterfactual reasoning을 제거했을 때 성능이 크게 떨어져 ‘대안 대비’가 효과의 중심임을 오류 분석과 ablation으로 확인한다. 또한 일반 상식 및 수학 추론 벤치마크 성능 저하가 크지 않아, pragmatics 특화 학습이 과도한 범용성 붕괴로 이어지지 않음을 시사한다.



### BCL: Bayesian In-Context Learning Framework for Information Extraction (https://arxiv.org/abs/2606.18620)
Comments:
          ACL 2026 Findings

- **Prior Approaches**: 기존 정보추출(IE)에서는 in-context learning(메모리 기반 학습)이 널리 쓰이지만, task transfer 방식(ChatIE, CodeIE)은 모델 크기에 따라 성능이 크게 흔들리거나(작은 모델에서 급락) relation classification에서 사실상 실패하는 경우가 있었다. guideline 기반 방식(예: GuideNER)은 NER에 강점이 있으나, 기준선이 되는 규칙 품질을 주파수 기반으로 고정해 체계적 최적화를 놓치고 RE로의 확장도 제한적이다.

- **Core Contribution**: 이 논문은 BCL(Bayesian In-Context Learning Framework for Information Extraction)로, IE 라벨을 의미적으로 잘게 쪼갠 subcategory 패턴을 ‘컨트롤 가능한 이산 변수’로 보고 이를 최적화한다. particle filtering과 Bayesian update를 결합해, 시퀀스 라벨링(NER)과 relation classification(RE) 모두에서 규칙 표현을 반복적으로 정련하는 프레임워크를 제시한다.

- **Technical Challenges**: 핵심 난제는 LLM의 거대한 파라미터 공간을 직접 관측·제어하기 어렵다는 점인데, BCL은 규칙 리스트를 저차원 관측 인터페이스로 재구성해 해결한다. 또한 규칙 공간이 조합적으로 불연속이고 성능 매핑이 비선형이라 고전 제어가 어렵기 때문에, 초기화-관찰(Validation 배치에서 likelihood 계산)-가중치 갱신(Bayesian posterior)-리샘플링(성능 상위 유지 + 다양성 mutation) 4단계를 통해 탐색과 수렴을 동시에 맞춘다.

- **Empirical Impact**: 실험에서 BCL은 여러 IE 벤치마크와 다양한 모델 스케일에서 기존 방법 대비 일관된 성능 향상(최대 약 30%)을 보이며, 특히 RE에서 다른 ICL 방식이 0에 가까운 성능을 보일 때도 유의미한 F1을 유지한다. 또한 비용 관점에서 Qwen2.5-3B에 BCL을 적용하면 더 큰 모델의 one-shot 성능과 비슷한 수준을 더 적은 파라미터로 달성하는 등 배포 가능성도 높였다.



### Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive Doctor-Patient-EHR Assistanc (https://arxiv.org/abs/2606.18613)
Comments:
          34 pages with 8 figures

- **Prior Approaches**: 기존 의료 LLM 평가는 의학 지식(질문응답), EHR 시스템 접근(조회/툴 사용), 환자 커뮤니케이션(대화/문서 생성)처럼 역할을 분리해 측정하는 경우가 많았습니다. 하지만 실제 의사 보조는 한 상호작용 안에서 지식·소통·시스템 동작을 동시에 조율해야 하고, 요청은 종종 맥락 의존적이며 EHR과 환자 정보는 각각 정밀한 툴 입력과 모호한 서술을 요구합니다. 이런 “단일 능력” 중심 평가는 다턴, under-specified 상호작용에서 성능이 크게 떨어진다는 최근 연구 흐름과 충돌합니다.

- **Core Contribution**: 이 논문은 의사-환자-EHR이 함께 얽힌 대화형 보조 시나리오를 평가하는 벤치마크 PhysAssistBench를 제안합니다. MIMIC-IV 실제 케이스를 기반으로 다턴의 agentic 환자를 합성 생성해, 정적 기록을 임상 시나리오로 “살려” 다중 턴에서 의사의 암묵적 요청과 환자 모호성을 처리하도록 시험합니다. 또한 1,296개 턴을 수기 검토하고 의사 검증을 거친 큐레이션 평가 세트를 제공합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 의사가 준 맥락 의존적 요청을 정확히 해석하고, (2) 환자 서술의 불명확성을 질문-수집으로 다루며, (3) FHIR R4 기반 EHR 툴 호출을 적절한 타이밍에 수행한 뒤, (4) 여러 근거를 하나의 임상 답으로 일관되게 통합하는 것입니다. 저자들은 POMDP 형태로 세션을 모델링하고, MIMIC-IV에서 시나리오·증거 풍부도(tier)에 맞춰 다턴 아크를 계획/생성하되 “근거 없는 반사실”은 필터링하는 다중 에이전트 합성 데이터 파이프라인과 품질 게이트(환각·구조 오류·임상 안전성)를 결합해 이를 해결합니다.

- **Empirical Impact**: 실험에서 여러 대표 LLM은 평균 점수(mRS)에서는 비슷해 보이지만, 세션 단위 일관성(Pass@Session)에서는 큰 격차가 나타나 “모든 턴을 끝까지 안정적으로 조율”하는 능력이 병목임을 드러냈습니다. 특히 IL→WU→CR→DG로 난이도 하이라키가 형성되고, DG와 CR(다중 툴·근거 조합)에서 암묵성(명명/술어 생략/추상 사건 지시)이 성능 저하를 키우는 경향이 확인되었습니다. PhysAssistBench는 파라메트릭 지식 격차보다 툴 체이닝·대화 맥락 통합 같은 조합 능력을 정밀하게 측정하게 해, 임상 LLM 평가의 기준선을 바꾸는 의미가 큽니다.



### Steerable Cultural Preference Optimization of Reward Models (https://arxiv.org/abs/2606.18606)
Comments:
          Accepted to Pluralistic Alignment @ ICML 2026

- **Prior Approaches**: 기존 LLM alignment 연구는 주로 특정 지역/인구집단의 선호도를 “하나의 공통된 기준”처럼 다뤄, 소수집단 선호가 과도한 편향으로 반영되는 문제가 자주 보고됐다. Group Preference Optimization(GPO)나 Group Robust Preference Optimization(GRPO)처럼 집단 선호를 학습하는 접근도 있지만, 보편 RLHF/DPO 같은 정렬 프레임워크에 자연스럽게 끼워 넣기 어렵거나(모듈 분리 문제), 특정 집단에 대한 독립적인 steerability를 충분히 겨냥하지 못했다. 또한 필터링·가중치 기반 정렬(예: RAFT, OPTune, Mallows-DPO)은 샘플 품질을 다루긴 해도 “문화적 하위커뮤니티 균형” 자체를 정면 목표로 삼지 않았다.

- **Core Contribution**: 이 논문은 다수의 문화 하위커뮤니티 선호를 과도한 쏠림 없이 반영하도록 하는 reward model 학습 방식에 초점을 둔다. 핵심은 ‘SCPO(Steerable Cultural Preference Optimization)’로, 글로벌 reward model이 “주류/합의 선호”를 기준선으로 삼아, 각 나라(집단)에서만 드러나는 문화적 차이를 분리하고 학습에 반영한다. 특히 global RM을 정답으로 가정하지 않고, 소수집단 선호가 어디서 갈라지는지(불일치/다이버전)를 조정 도구로 사용한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 소수집단별 reward model이 (1) 고유한 문화 신호는 학습하되, (2) global RM과의 큰 불일치가 항상 “가치 있는 차이”가 아닐 수 있다는 점(노이즈·라벨 오류 가능성)을 함께 다뤄야 한다는 것이다. SCPO는 두 단계로 해결한다: filtering은 global RM과 불일치하는 preference pair만 남겨 “문화적 구별점”을 남기고, weighting은 두드러진 다이버전일수록 학습 손실에서 더 낮게 가중해 편향 위험을 완화한다. Bradley-Terry 확률적 관점을 확장해 불일치 크기가 갖는 신호 강도/신뢰도 해석을 weighting에 연결하고, 최종적으로 가중 binary ranking loss로 minority RM을 학습한다.

- **Empirical Impact**: 실험에서는 PRISM(7개국, 미국·영국 제외)을 사용해 OpenAssistant와 Tülu 3 기반 reward model을 country-specific 데이터로 학습하며, SCPO가 대부분 국가에서 baseline fine-tuning 대비 성능을 개선했다. PRISM과 GlobalOpinionQA 모두에서 minority reward model(소수집단 선호) 성능이 상승했고, minority 대비 정확도/일반 성능 간 trade-off를 별도 “true country-specific subset” 평가로 점검했다. 또한 SCPO는 full-data finetuning 대비 최대 280% 더 학습 데이터 효율적이며, GlobalOpinionQA에서는 Jensen-Shannon Distance 기반 분포 일치도와 GPO 대비 우수한 문화 정렬 결과를 보였다.



### Low-resource Language Discrimination Towards Chinese Dialects with Transfer learning and Data Augmentation (https://arxiv.org/abs/2606.18597)
Comments:
          Published in ACM TALLIP

- **Prior Approaches**: 중국어 방언 판별은 라벨이 부족해 성능이 쉽게 흔들리는 대표적인 자연어처리 과제다. 기존 접근은 제한된 데이터로 학습하거나, 별도 전이학습 없이 모델을 직접 학습해 데이터 규모의 한계를 그대로 겪는 경우가 많았다.

- **Core Contribution**: 논문은 전이학습과 데이터 증강을 결합한 Chinese dialects discrimination framework (CDDTLDA)를 제안해 저자원 상황에서의 방언 판별을 개선한다. 대규모 중국어 방언 코퍼스로 소스 측 ASR 모델을 먼저 학습한 뒤, 속도·피치·잡음 기반 증강과 fine-tuning을 통해 타깃 측 ASR 학습을 보완하고, 그 내부 표현으로 판별을 수행한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 타깃 방언은 라벨/발화 데이터가 부족하다는 점과 (2) 소스와 타깃 사이의 음향·언어적 차이를 어떻게 공통 표현으로 연결할지가 된다. 논문은 타깃 저자원에 대해 speed/pitch/noise 증강을 적용하고, self-attention으로 소스·타깃 ASR 모델 간의 잠재적 공통 의미 특징을 포착한 뒤, 타깃 ASR의 hidden semantic representation을 추출해 판별에 활용한다.

- **Empirical Impact**: 실험 결과, CDDTLDA는 두 개의 중국어 방언 벤치마크 코퍼스에서 기존 state-of-the-art 대비 유의미하게 높은 성능을 보였다. 데이터가 적은 방언 분류 문제에서 ‘소스 ASR 사전학습+간단하지만 효과적인 음향 증강+공통 의미 포착’의 조합이 실용적인 대안임을 보여준 점에서 의미가 있다.



### Dual Dimensionality for Local and Global Attention (https://arxiv.org/abs/2606.18587)
- **Prior Approaches**: 기존 KV cache 절감 연구는 슬라이딩 윈도우/스트리밍 같은 sparse attention 또는 heavy hitters 기반 eviction으로 효율을 높이거나, MLA처럼 key/value 차원을 사전 압축해 메모리 부담을 줄이는 방식이 주를 이뤘습니다. 다만 이러한 방법들은 대체로 모든 토큰에 동일한 표현 차원을 부여하거나, 거리에 따른 “필요 표현력”의 차이를 직접 모델링하진 못했습니다. 그래서 토큰 거리(distance)가 표현 차원 요구량을 어떻게 바꾸는지에 대한 정량적 가설 검증이 부족했습니다.

- **Core Contribution**: 이 논문은 Distance-Adaptive Representation(DAR)이라는 원리를 제안하며, 예측 대상에 가까운 로컬 토큰은 더 풍부한(고차원) 표현이 필요하고 멀리 있는 토큰은 더 낮은 차원으로도 충분하다는 비대칭 가설을 형식화합니다. 구현은 로컬 윈도우 내부 토큰은 원래 차원을 유지하고, 그 밖의 토큰은 병목 차원(down-projection)으로 줄인 표현을 key/value로 사용하도록 설계됩니다. 즉, “거리별로 표현 차원을 배분”하는 새로운 KV 설계 방향을 제시합니다.

- **Technical Challenges**: 핵심 과제는 서로 다른 차원의 표현(로컬 vs 글로벌)을 표준 attention 계산 흐름에 자연스럽게 결합하는 것입니다. 논문은 공통의 key/value 투영을 공유하기 위해 글로벌 경로의 병목 표현을 다시 모델 차원으로 up-projection하는 2경로(attention paths) 구조를 사용해 학습 안정성을 확보합니다. 또한 auxiliary loss 없이 다음 토큰 예측만으로 down/up projection이 거리별 표현을 학습하도록 만들었고, 윈도우 크기 변화에 대해서도 성능이 특정 구간에서 견고함을 확인했습니다.

- **Empirical Impact**: 실험 결과 DAR은 Pythia-70M~410M 스케일에서 full-dimensional baseline과 거의 같은 perplexity를 달성하며, 성능 저하는 “균일 축소(uniform reduction)”에 비해 훨씬 완만했습니다. 특히 down 차원을 d/4 수준으로 설정하면 스케일 전반에서 기준선과의 격차가 작았고, downstream 작업에서도 적당한 축소 범위(d/2~d/4)는 no-bottleneck 대비 유지/근소 개선을 보였습니다. 결론적으로 토큰 거리별 표현 차원 비대칭이 실제로 성능 손실 없이 KV cache 절감을 설계할 수 있음을 보여, long-context inference 효율화의 새로운 실험 축을 제안합니다.



### Speech-Driven End-to-End Language Discrimination towards Chinese Dialects (https://arxiv.org/abs/2606.18584)
Comments:
          Published in ACM TALLIP

- **Prior Approaches**: 기존 언어 식별 연구는 주로 텍스트 기반 특징에 의존해, 서로 유사한 언어·방언·방언권을 구분할 때 성능이 떨어지는 문제가 있었다. 특히 중국어 방언처럼 미세한 차이를 가진 분류에서는 텍스트만으로 변별력을 확보하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 텍스트가 아닌 음성 기반 특징에 집중해, 중국어 방언의 fine-grained 식별에서 speech-driven 접근의 유효성을 보여준다. MFCC 기반 CNN으로 음성 특징을 검증하고, HMM-DNN 기반 end-to-end 음성인식 모델에서 방언 관련 변별 단어를 attention으로 추출한 뒤, CNN으로 단어 임베딩과 MFCC 특징을 함께 결합한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) MFCC 같은 음성 특징이 CNN 기반 언어 식별에 실제로 적합한지 확인하고, (2) 음성인식 과정에서 방언을 가르는 단어/구간을 어떻게 안정적으로 끌어낼지에 있었다. 논문은 MFCC- CNN 적합성을 체계적으로 실험하고, HMM-DNN + attention으로 방언별 변별 단어를 추출한 뒤, CNN 결합을 통해 단어 레벨 정보와 음성 특징을 함께 학습하도록 설계했다.

- **Empirical Impact**: 두 가지 중국어 방언 벤치마크 말뭉치를 평가한 결과, 제안한 speech-driven 방식이 기존 state-of-the-art 대비 더 적절하고 효과적인 것으로 나타났다. 텍스트 편향을 줄이고 음성의 미세한 차이를 직접 활용할 수 있다는 점에서, 방언/사투리 식별 및 음성-언어 분류 연구에 실용적인 방향성을 제시한다.



### MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieva (https://arxiv.org/abs/2606.18508)
- **Prior Approaches**: RAG의 성능은 청크(chunk) 분할과 검색(granularity) 설계에 크게 좌우되지만, 기존 dense retrieval은 작은 청크가 정밀한 대신 후보 수가 폭증해 비용·지연이 커지고, 큰 청크는 여러 토픽이 섞여 dense 유사도에 잡음이 유입된다는 한계가 있습니다. 이를 줄이려는 기존 접근은 proposition-level 같은 더 세분화된 단위, LLM-guided chunking, RAPTOR 같은 계층적 검색, 또는 재랭킹/LLM 기반 evidence 선택을 사용하지만, 보통 전처리·인덱스·추가 추론 단계가 늘어나거나 추론 지연이 발생합니다.

- **Core Contribution**: MCompassRAG는 coarse-grained 청크의 장점을 유지하면서도, 청크를 토픽 메타데이터로 “주제별 나침반”처럼 검색 가능하게 만드는 메타데이터 가이드 검색 프레임워크를 제안합니다. 청크 임베딩 자체의 잡음에만 기대지 않고, 토픽 모델이 만든 토픽 분포를 청크와 동일한 임베딩 공간에 얹어 query가 해당 토픽 방향을 먼저 겨냥하도록 하며, LLM-teacher distillation으로 경량 retriever를 학습해 추론 시 추가 LLM 호출 없이 evidence 품질을 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 큰 청크가 토픽 혼합으로 인해 유사도 점수가 희석되는 문제를, 검색 후보 수를 늘리지 않고 해결하는 것입니다. 논문은 (1) 코퍼스 레벨 metadata bank에서 query에 맞는 토픽 분포를 선택하고 (2) 선택된 토픽들을 abstraction 모듈로 요약해 잡음과 편향을 줄인 뒤 (3) 해당 토픽 정보를 포함한 query-청크 표현을 극단적 multi-label 목적의 student MLP retriever로 점수화하여, LLM 기반 재랭킹 없이도 토픽 인지 검색을 구현합니다.

- **Empirical Impact**: 6개 retrieval 벤치마크에서 MCompassRAG는 비-LLM 최강 효율 베이스라인 대비 정보 효율(IE) 평균 8.24% 향상과 함께 지연은 5배 이상 낮은 성능-비용 균형을 보입니다. 특히 DRBench·LegalBench-RAG 같은 멀티홉이 어려운 설정에서 격차가 더 크게 나타났고, retrieval 시 LLM을 호출하는 oracle에 근접하면서도 추론 시 LLM 호출이 없어 실제 deep research 에이전트 시나리오에 의미 있는 효율 개선을 시사합니다.



### Towards Scalable Customization and Deployment of Multi-Agent Systems for Enterprise Applications (https://arxiv.org/abs/2606.18502)
Comments:
          Preprint

- **Prior Approaches**: 기존 LLM 기반 멀티에이전트는 복잡한 작업을 전문 에이전트들로 분해해 품질을 높이지만, 여러 LLM 호출로 인해 지연과 비용이 급증해 프로덕션 배포가 어렵다는 문제가 있었다. 또한 대형 모델을 그대로 쓰면 메모리/추론 비용이 커지고 SLA를 만족하기 힘들며, 도메인별 커스터마이징에 따른 운영 부담도 컸다. 추론 지연을 줄이려는 speculative decoding(EAGLE류)나 양자화(FP8 등)는 있었지만, 실제 도메인에 맞춘 draft 모델 학습과 FP8 성능 보존을 위한 캘리브레이션 설계가 충분히 정리되지 않았다.

- **Core Contribution**: 이 논문은 멀티에이전트 시스템을 “도메인 커스터마이징”과 “추론 최적화” 두 단계로 통합한 통일 프레임워크를 제안한다. 첫 단계에서는 compact 모델이 에이전트 역량을 유지하도록 CPT(Continual Pretraining)–SFT–DPO를 조합해 연속 적응과 선호 정렬을 수행한다. 둘째 단계에서는 speculative decoding용 EAGLE drafter와 FP8 양자화를 함께 적용해 품질 저하를 최소화하면서 서빙 비용과 지연을 크게 줄인다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 도메인 적응 중 catastrophic forgetting을 줄이면서 에이전트의 instruction-following을 확보하는 것, (2) EAGLE drafter의 수용률(acceptance rate)이 도메인 데이터 구성과 학습에 민감한 점, (3) FP8 post-training quantization이 캘리브레이션 데이터 혼합에 따라 장기 컨텍스트 성능이 조용히 깨질 수 있다는 점이었다. 이를 위해 저자들은 샘플별 컨텍스트를 앞단에 붙이는 Context-aware CPT로 CPT 초기 손실 급증을 완화하고, SFT는 LoRA로 파라미터 업데이트 부담을 줄이며, DPO는 자동 judge 신호에 더해 수작업 hard-negative를 섞어 논리 경계 케이스를 보정했다. 추론 단계에서는 EAGLE 학습용으로 도메인 시뮬레이션 트레이스를 정교하게 섞고, FP8은 mixed calibration set으로 long-context의 activation clip rate를 관리하도록 설계했다.

- **Empirical Impact**: 자동차 리테일 도메인에서 10B급 커스텀 멀티에이전트에 대해 E2E 기능 스트레스 테스트를 포함한 실험을 수행했으며, 최종적으로 end-to-end 처리량이 4.48x 개선됐다. 또한 FP8 양자화와 speculative decoding을 쌓아도 작업 정확도는 거의 유지되며, long-tail 시나리오에서도 E2E 패스율을 “완전 유지” 수준으로 보고했다. 저자들은 throughput 상승이 단순한 디코딩 변경이 아니라 도메인 정렬 데이터 엔지니어링(시뮬레이터 트레이스 혼합)과 단계별 학습·캘리브레이션이 결합될 때 곱연산으로 나타난다고 정리해, 실제 배포 관점의 최적화 가이드를 제공한다.



### PreUnlearn: Auditing Collateral Knowledge Damage Before Large Language Model Unlearning (https://arxiv.org/abs/2606.18473)
Comments:
          12 pages, 6 figures

- **Prior Approaches**: 기존 LLM unlearning 평가는 주로 전체 유틸리티 지표나 제한된 사후 프로브로 평균 성능을 봐서, forget set 주변에서 시작된 ‘부수적 손상(collateral damage)’이 평가 데이터 전역으로 어떻게 전파되는지 세밀하게 포착하지 못했다. 또한 많은 벤치마크는 forget set을 고정한 채 evaluation set을 주어진 것으로 두어, 서로 다른 forget 데이터가 어떤 지역/원거리 지식에 영향을 주는지 체계적으로 묻기 어려웠다.

- **Core Contribution**: 이 논문은 unlearning을 데이터 관점에서 재정의하며, 부수적 손상이 forget set(L1)에서 출발해 같은 도메인(L2), 먼 도메인(L3)으로 ‘거리 감쇠(decay)’하되 도메인 경계에서 완전히 사라지지 않는 패턴을 정량화한다. 더 나아가 unlearning을 실행하기 전, (forget, evaluation) 쌍만으로 사후 손상을 예측하는 forget-set auditing을 감독학습 회귀 문제로 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 무엇을 잊을지(forget)와 무엇을 보존할지(retain) 경계가 모델 내부에서 얽혀 있어, 사후에만 영향을 확인하면 비용이 커진다는 점과 (2) 모델 업데이트 전에는 그 영향을 직접 측정할 수 없다는 점이다. 이를 위해 논문은 사후 체크포인트/그래디언트 없이, 두 세트의 의미·표현 기하(centroid distance/유사도, 길이·어휘 비율, representation-geometry 상호작용)를 특징으로 삼아 미리 손상 비율을 예측하도록 설계했고, LODO 및 도메인 홀드아웃으로 누수 없는 일반화도 함께 검증했다.

- **Empirical Impact**: WikiText-103 기반 실험에서 unlearning 영향은 L1>L2>L3의 일관된 순서를 보였고(평균적으로는 감쇠), 분포의 꼬리에는 큰 손상이 남아 ‘평균만’으로는 위험을 놓칠 수 있음을 보여줬다. 감사 모델은 특히 forget–evaluation 간 상호작용 특징이 가장 강한 신호를 제공하며, 사후 손상 크기 예측보다도 위험 쌍의 순위를 높은 상관으로 선별해 비싼 unlearning 실행을 앞단에서 줄이는 ‘조기 경보 도구’로 의미가 크다.



### Possible or Definite? A Benchmark for Evaluating Diagnostic Uncertainty Preservation in Clinical Tex (https://arxiv.org/abs/2606.18471)
- **Prior Approaches**: 기존 연구는 LLM이 임상 텍스트에서 문장 유창성/일관성, 사실성, 정확도를 잘 내는지 중심으로 평가해 왔습니다. 불확실성에 관한 연구도 대부분 (1) 내부 신뢰도(logits, entropy, calibration 등)나 (2) 출력에 불확실성을 ‘표현’하는지에 머물렀고, 실제 진단 문장의 불확실성 강도를 얼마나 ‘보존’하는지는 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 진단 불확실성 표현을 “출처 문장 수준의 확실성 강도(5단계)”로 정의하고, LLM이 이를 변환 과정에서 얼마나 보존하는지 직접 측정합니다. 또한 1,200개 임상 문서와 9,184개의 cue-target(불확실성 큐-대상 개념) 주석으로 구성된 벤치마크를 구축해, 여러 문서 타입과 수준을 포괄적으로 평가할 수 있게 했습니다.

- **Technical Challenges**: 핵심 과제는 문서 레벨이 아닌 proposition(개별 주장) 레벨에서 ‘불확실성 강도’가 바뀌는지 안정적으로 라벨링/검증하는 것입니다. 저자들은 규칙 기반 파이프라인으로 cue-target 쌍을 추출하되, 의학 개념은 medSpaCy+QuickUMLS로 CUI에 연결하고 규칙 기반 라벨의 정밀도를 수작업 검증(약 82% 정확도)으로 확인해, 모델이 라벨을 학습/증폭하는 순환 평가 위험을 피했습니다.

- **Empirical Impact**: 실험 결과 LLM은 불확실성 큐와 대상 개념을 함께 유지하더라도, 원래 불확실성 레벨을 정확히 보존하는 비율이 절반 이하(URR 대략 33~46%)로 나타났고 인접 레벨의 미묘한 구분도 약했습니다. 또한 왜곡은 CAR 기준으로 ‘불확실성 큐를 제거하고 확정 주장으로 붕괴’하는 형태가 압도적이어서(대략 2/5 수준), 기존 자동 지표가 놓치는 임상 의미 왜곡 실패 모드가 드러났다는 점에서 안전한 임상 워크플로우 도입에 중요한 시사점을 제공합니다.



### Montreal Forced Aligner and the state of speech-to-text alignment in 2026 (https://arxiv.org/abs/2606.18466)
- **Prior Approaches**: 강제 정렬(forced alignment)은 음성 구간에 단어·음소 타임스탬프를 자동으로 매기는 작업으로, Montreal Forced Aligner(MFA)가 2016년 이후 표준 도구로 자리 잡았다. 다만 MFA 1.0 이후로 신경 ASR 기반 정렬기까지 포함해 최신 정렬기들과 MFA를 같은 프레임에서 체계적으로 비교한 평가는 부족했고, 기존 비교는 영어 중심이거나 데이터/평가지표가 달라 단순 비교가 어려웠다. 또한 신경 정렬기들은 종종 정확한 프레임 단위 경계 배치보다는 문자열 생성에 최적화돼 있어 시간 경계 정밀도에서 불리할 수 있다는 한계가 지적돼 왔다.

- **Core Contribution**: 본 논문은 MFA 3.0의 변화(대규모 사전학습 모델, 적응(adaptation)과 cross-language remapping, 발음 확률(pronunciation probability)과 음운 규칙)를 정리하고, 영어·일본어·한국어 3개 언어에 대해 여러 고전/신경 강제 정렬기와의 성능을 시스템적으로 벤치마크한다. 4개 벤치마크 데이터셋 전반에서 평균 경계 오차가 15ms 미만이며, 다수 비교에서 state-of-the-art 또는 near state-of-the-art 수준을 달성해 “최신 MFA가 여전히 강력한 기준선”임을 실증한다. 특히 훈련 분포 밖 언어에서도 adaptation과 remapping이 효과적임을 보여준다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 품질이 들쭉날쭉한 대규모 공개 데이터로 사전학습 모델을 안정적으로 만들고 (2) 목표 언어/방언의 음소 체계와 발음 변이를 정렬에 반영하면서 (3) 벤치마크 비교 시 서로 다른 phone set 표현 차이를 공정하게 다루는 것이다. 논문은 HMM-GMM 기반 Kaldi 계열 학습 파이프라인에 LDA 전단 추가와 발음 확률·침묵(silence) 확률을 내재화하고, 잡음이 큰 코퍼스를 학습 단계 후반에 점진적으로 섞는 데이터 혼합 전략을 도입했다; 동시에 phone interval을 modified Levenshtein 방식으로 정렬해 서로 다른 음소 표현 간 경계오차를 계산하는 평가 절차를 마련했다. 더 나아가 phonological rules로 사전 발음 변이를 확장하되, 학습 정렬 데이터에서 관측되지 않은 규칙 생성 변이는 pruning해 과적합을 줄이는 설계를 사용한다.

- **Empirical Impact**: 실험은 TIMIT·Buckeye(영어)와 CSJ 하위셋·Seoul Corpus(일본어/한국어)에서 단어·음소 경계 탐지 성능을 평가하며, MFA 3.0이 대부분의 비교에서 더 낮은 경계 오차를 보이도록 설계 효과가 확인된다. 또한 adaptation과 cross-language remapping이 해당 언어의 훈련 분포 밖 조건에서도 성능 향상으로 이어지고, 발음 확률 모델링과 음운 규칙은 특정 조건에서 추가 이득을 준다는 점이 정량적으로 드러난다. 결과적으로 MFA 3.0은 언어과학/음성인식 파이프라인에서 “프레임 단위 정밀도”가 중요한 연구의 기준 도구로서 실무적 활용도를 크게 높일 것으로 기대된다.



### LLM Parameters for Math Across Languages: Shared or Separate? (https://arxiv.org/abs/2606.18453)
Comments:
          5 pages. Accepted at ACL Student Research Workshop (SRW) 2026. Code: this https URL Translated Datasets: this https URL Webpage: https://math-across-languages.github.io

- **Prior Approaches**: 기존 연구는 (1) 특정 능력을 일부 파라미터로 국소화하는 mechanistic studies, (2) 번역·평가 중심의 multilingual reasoning 연구로 나뉜다. mechanistic 연구는 보통 영어에 치우쳐 있고(MathNeurosurgery 등), 성능 비교 위주의 다국어 연구는 어떤 “계산 기질”이 작동하는지 파라미터 수준에서 보여주지 못했다.

- **Core Contribution**: 이 논문은 MathNeurosurgery 계열 접근을 확장해, 언어별로 수학 연관 파라미터를 추출하고 언어 쌍 간 overlap(겹침)을 정량 비교하는 프레임워크를 제시한다. 그 결과 수학 추론에서 공통으로 쓰이는 파라미터 코어가 완전히 language-invariant도 아니고, 전적으로 language-specific도 아니라는 “부분적 겹침” 구조를 보여준다.

- **Technical Challenges**: 가장 큰 난제는 언어별로 수학과 일반 언어 이해 파라미터를 분리해내면서, 추출된 파라미터 집합의 유사도를 공정하게 비교하는 일이다. 저자들은 수학/비수학 데이터로부터 레이어별 중요 파라미터를 뽑고, Jaccard coefficient로 언어 간 집합 겹침을 계산했으며, pruning/ scaling 가중치 개입으로 수학 성능이 선택된 파라미터 집합에 집단적으로 의존함을 검증한다.

- **Empirical Impact**: Llama 3.2 1B, Qwen3 4B, Llama 3.1 8B를 영어·독일어·프랑스어·힌디어에 대해 GSM8K 및 비수학 데이터셋(MMLU, RACE)으로 평가한 결과, 영어가 가장 많은 math-associated parameters를 보였고 힌디어 같은 저자원/문자 체계 차이 언어는 더 적은 것으로 나타났다. 또한 Jaccard 유사도는 중간 레이어에서 가장 높고(공유 코어 신호), 초기·후기 레이어에서는 낮아지는데 이는 표면 처리와 언어별 전문화(레이어 특화) 가능성을 시사한다. 개입 실험에서는 소수 파라미터가 아니라 선택된 집합 전반이 성능에 영향을 주는 collective effect가 관측되어, “수학 회로”가 단일 병목이 아닌 분산된 모듈임을 강화한다.



### VISUALSKILL: Multimodal Skills for Computer-Use Agents (https://arxiv.org/abs/2606.18448)
- **Prior Approaches**: 컴퓨터 사용 에이전트(CUA)는 OSWorld 같은 데스크톱 벤치마크에서 사람 수준에 가까워졌지만, 긴 지평 과제와 한 번도 보지 못한 UI/소프트웨어로 일반화할 때 성능이 떨어집니다. 이런 문제를 줄이기 위해 스킬 라이브러리를 활용하지만, 기존 스킬은 대부분 텍스트 중심이라 GUI 상호작용의 핵심인 스크린샷(시각 정보)이 스킬 아티팩트에서 소실되는 한계가 있습니다. 또한 멀티스텝 진행 중 각 단계의 “중간 UI 상태”를 시각적으로 검증하기가 텍스트 설명만으로는 어렵습니다.

- **Core Contribution**: 이 논문은 GUI의 시각 단서를 스킬 아티팩트에 그대로 유지하는 계층형 멀티모달 스킬 VisualSkill을 제안합니다. 애플리케이션 하나당 스킬 1개를 만들고, central index(Topic md)에서 토픽을 골라 load_topic MCP 도구로 필요한 텍스트와 figure만 온디맨드로 불러오게 설계했습니다. 스킬 제작은 공식 문서 마이닝(Stage 1)과 실제 앱을 탐색해 UI 지식을 보강(Stage 2)하는 2단계 파이프라인으로 구성됩니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트로 설명하기 모호한 UI 요소(아이콘, 레이아웃, 공간 관계)를 스킬에 정확히 담고 (2) 매 행동 이후 기대한 UI 상태로 실제로 갔는지 검증 가능하게 만드는 것입니다. 저자들은 문서의 figure를 그대로 포함하되, 문서가 놓치는 저빈도/동적 UI는 라이브 앱 탐색으로 수집하며, free exploration과 실패 지점을 겨냥한 trajectory-targeted exploration을 결합해 토픽별 per-topic 파일에 반영합니다. 더불어 agent가 전 경로에서 figure를 실제로 참조하도록 스킬 접근 방식을 MCP tool 기반으로 최적화합니다.

- **Empirical Impact**: CUA-World와 OSExpert-Eval에서 Claude Code CLI(Claude Opus 4.6 백엔드)를 사용해 VisualSkill의 평균 점수는 0.456으로, no-skill 기준 0.303 대비 +15.3포인트 절대 상승을 보였습니다. 특히 동일 소스에서 만든 matched text-only 스킬과 비교해도 0.373 → 0.456으로 추가로 +8.3포인트 절대 이득이 나타나 시각 figure 유지가 성능에 직접 기여함을 확인했습니다. 분석 결과 이득은 텍스트만으로 어려운 UI 요소 식별과 단계별 UI 상태 검증에 집중되며, MCP 기반 로딩 없이 직접 Read로는 figure 참조가 크게 줄어 성능이 약화되는 구조적 차이도 관찰됩니다.



### CoreMem: Riemannian Retrieval and Fisher-Guided Distillation for Long-Term Memory in Dialogue Agents (https://arxiv.org/abs/2606.18406)
Comments:
          15 pages, 5 figures

- **Prior Approaches**: 개인화 대화 에이전트는 여러 세션에 걸친 장기 기억이 필수지만, 소비자급(예: 8GB VRAM) 엣지 환경에서는 메모리·연산·토큰 비용 병목이 크게 발생합니다. 기존 방식은 보통 cosine similarity 기반 검색과 휴리스틱 문맥 압축(토큰/문장 pruning)을 사용하며, 허브 문제(hubness)나 압축 시 구문 단절 같은 실패가 이론적으로 정리되지 않습니다. 또한 검색과 압축이 서로 다른 목표로 최적화되어, 검색으로 찾은 핵심 사실의 연결고리가 압축 과정에서 깨지는 cascade failure가 잦습니다.

- **Core Contribution**: 이 논문은 CoreMem을 제안하며, 검색과 압축을 정보기하 관점으로 하나의 통일된 프레임에 묶어 “lifelong memory”를 엣지에서 운용 가능하게 만듭니다. 핵심 아이디어는 기억 임베딩의 검색 거리를 정보기하( Fisher-Rao / Mahalanobis )로 재정의하고, 압축은 Fisher 정보를 이용해 KL-divergence 관점의 압축-손실 트레이드오프를 원칙적으로 구성하는 것입니다. 그 결과, 자원 제약 하에서도 일관성 있는 개인화 대화에 필요한 장기 기억을 유지하도록 설계됐습니다.

- **Technical Challenges**: 문제는 (1) 고차원에서 cosine 검색이 허브 메모리에 끌려가고, (2) 토큰 단위 압축이 이론적 보장 없이 구문을 파괴하며, (3) Fisher-Rao 같은 기하 계산이 엣지에서 너무 무거울 수 있다는 점입니다. CoreMem은 locally adaptive Fisher-Rao metric을 위해 (대각 성분에 저랭크 보정을 얹은) 역공분산을 만들고 Woodbury 가속으로 실시간 검색을 가능케 하며, 압축은 diagonal Fisher approximation과 1차 테일러 기반 민감도(Trace)로 compression-KL 경계를 구성합니다. 동시에 FDTD의 계층형(sentence→token) 압축과 syntax/keyword/content 보호, gap filling 등 구조 보정으로 “중요 논리 연결고리”가 사라지는 현상을 완화합니다.

- **Empirical Impact**: LOCOMO와 LongMemEval-S 벤치마크에서 CoreMem-Fusion은 Open-domain에서 +4.51 pp, Temporal에서 +4.17 pp 같은 뚜렷한 성능 향상을 보였고, 특히 임베딩 차원이 작을수록(예: MiniLM-L6) Riemannian 보정의 이득이 커지는 패턴이 관찰됐습니다. 압축 스트레스 테스트에서는 동일한 검색 컨텍스트를 두고도 Fisher-guided FDTD가 더 높은 conditional accuracy와 더 공격적인 압축률(67.4%)을 동시에 달성해 “이론 기반 토큰 보존”의 효과를 뒷받침합니다. 마지막으로 8GB VRAM 예산 내에서 엣지에서 인코딩·검색·압축이 함께 수행되도록 프로파일링했으며, 결과적으로 이론적 근거와 실사용 제약을 동시에 만족하는 장기 기억 에이전트 방향성을 제시합니다.



### JetFlow: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting (https://arxiv.org/abs/2606.18394)
- **Prior Approaches**: Speculative decoding(SD)은 여러 토큰을 미리 초안(draft)으로 생성한 뒤 병렬 검증으로 지연을 줄이지만, 초안 길이를 늘려도 수락률이 충분히 유지되지 못하면 성능이 상한에 막힌다. 특히 head-based SD는 인과관계와 효율 사이의 딜레마를 겪는데, EAGLE 같은 autoregressive drafters는 경로 조건을 잘 맞추지만 트리 깊이가 늘면 초안 비용이 커지고, DFlash 같은 bidirectional block-diffusion drafters는 한 번에 생성하되 branch-agnostic marginals가 서로 모순되는 트리를 만들어 수락 효율을 떨어뜨린다. 기존 연구들은 정렬로 수락률을 올리거나(초안-타깃 정합), 트리 검증으로 효율을 올리는 쪽으로 각각 최적화해 왔지만 둘을 동시에 끌어올리는 데는 한계가 있었다.

- **Core Contribution**: 본 논문은 JetFlow를 제안하며, head 기반 프레임워크 안에서 ‘한 번의 forward로 초안을 만들면서도 각 가지(branch)는 해당 경로의 인과 조건을 보존’하도록 설계한다. JetFlow는 frozen target model의 fused hidden states를 활용해 causal parallel draft head를 학습하고, 그 결과 후보 트리의 분포가 target model의 autoregressive factorization과 정렬되게 만든다. 이를 통해 draft budget을 늘릴 때 더 긴 accepted prefix와 더 큰 end-to-end speedup으로 연결되도록 한다.

- **Technical Challenges**: 핵심 기술 난제는 parallel 트리 초안에서 각 노드가 자신의 가지(prefix 경로)에 조건된 분포로 샘플링돼야 하는데, branch-agnostic per-position 분포로 만들면 개별 토큰은 그럴듯해도 상호 모순된 트리가 구성된다는 점이다. JetFlow는 tree-causal attention mask로 각 노드가 원래 prefix와 조상(ancestor)만 보도록 제한해 branch-wise 인과 분해를 만들고, 단일 forward로 여러 깊이의 후보를 동시에 산출한다. 또한 학습에서는 forward KL distillation을 기본으로 사용해 target의 확률 질감(soft-label preference)을 더 잘 보존하도록 했으며, 트리 확장은 누적 draft log-prob 기반 스코어링으로 고득점 가지를 예산 내에서 반복 확장하는 방식으로 구현한다.

- **Empirical Impact**: 실험에서 JetFlow는 Qwen3-8B 및 Qwen3-30B-A3B의 dense·MoE 설정, 그리고 math/coding/chat 벤치마크 전반에서 bidirectional-head와 tree-based SD 대체군을 일관되게 능가한다. 특히 H100에서 MATH-500은 최대 9.64x, open-ended 대화 워크로드는 최대 4.58x의 speedup을 보고하며, vLLM 통합 평가에서도 실제 서빙 부하 조건에서 latency 이점을 추가로 확인했다. 요약하면 JetFlow는 SD의 draft 길이 스케일링 상한을 인과-효율 관점에서 완화해, 긴 생성이 필요한 실서비스에서 실질적인 디코딩 지연 저감 가능성을 보여준다.



### Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation (https://arxiv.org/abs/2606.18389)
Comments:
          25 pages

- **Prior Approaches**: 기존 LLM 합성데이터 생성의 최강 성능은 타깃 언어 예시를 넣는 few-shot prompting에 의존하는 경우가 많았습니다. 하지만 이는 추론 비용을 키우고, 어휘적 앵커링으로 다양성이 줄어들 수 있다는 한계가 지적됩니다. 또한 activation steering 계열은 주로 신뢰성·감정·독성 등 속성 제어에 집중돼, 생성된 텍스트 자체의 품질/다양성을 겨냥한 저자원 합성데이터 파이프라인 적용은 상대적으로 비어 있었습니다.

- **Core Contribution**: 이 논문은 저자원 언어 합성데이터 생성을 위해 activation steering을 대안으로 제시합니다. 두 가지 steering 전략을 비교하는데, Language Steering은 언어 정체성 방향을, Quality Steering은 사람 글과 backtranslated 텍스트를 대비해 ‘사람이 쓴 듯한’ well-formedness를 분리합니다. 특히 Quality steering 벡터를 대조쌍(인간-역번역)으로 직접 도출해 활용했다는 점을 핵심 기여로 내세웁니다.

- **Technical Challenges**: 문제는 (1) 내부 표현에서 원하는 속성을 안정적으로 분리할 수 있는지, (2) 어떤 레이어와 steering 강도 α에서 효과가 나는지, (3) zero-shot과 few-shot에 공통으로 적용 가능한지입니다. 저자들은 residual stream의 레이어별 평균 활성으로 steering vector를 만들고, 이를 4종 오픈소스 LLM과 여러 레이어(약 21%/48%/74% 깊이) 및 α 스윕에 적용합니다. 그 결과 early layer 개입이 특히 zero-shot에서 가장 일관된 이득을 보였고, 모델 계열에 따라 α 민감도(예: Gemma는 큰 α에서도 안정, Llama는 과도 개입 시 붕괴)가 다름을 실험으로 확인했습니다.

- **Empirical Impact**: 11개 유형론적으로 다양한 언어에서 sentiment 및 topic 분류용 합성데이터를 생성한 뒤 XLM-R을 파인튜닝해 평가했을 때, Quality와 Language steering 모두 downstream 성능을 개선하는 경향이 나타났습니다. 특히 Quality steering이 전반적으로 더 높은 F1 향상과 일관성을 보였고, 생성 데이터 다양성도 여러 지표(어휘 다양성, 임베딩 다양성 등)에서 동반 증가했습니다. 또한 Language와 Quality 벡터의 코사인 유사도는 모델·언어 의존적으로 극화되며, 초기 레이어에서 대부분 강한 음의 상관이 관찰되어 두 steering이 서로 다른(때로는 반대 방향의) 표현을 조작한다는 통찰을 제공합니다.



### SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG (https://arxiv.org/abs/2606.18381)
- **Prior Approaches**: RAG의 성능은 청크 분할의 단위(너무 크면 잡음, 너무 작으면 문맥 단절)에 크게 좌우되는데, 기존 접근은 이를 LLM-guided chunking(청크 경계 학습), single-level 확장, 계층 요약 등으로 보완해 왔다. 다만 색인/검색 과정에서 외부 LLM 호출 비용이 들거나, 확장·요약이 단일 granularity에 고정돼 문장 간 의존성을 충분히 모으기 어렵고, 요약은 증거 손실을 유발할 수 있다.

- **Core Contribution**: SproutRAG는 문장 단위 청크를 attention에 기반한 계층적 구조로 재배열해, 다중 granularity 근거 검색을 end-to-end로 학습한다. 특히 문장-문장 inter-sentence attention을 head·layer 가중합으로 집계해 binary chunking tree를 만들고, 외부 LLM 호출이나 lossy 요약 없이 여러 레벨의 후보를 함께 검색한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 기존처럼 head·layer를 균일 평균하면 proximity bias가 생겨 트리의 전역 인덱스 품질이 떨어지고, (2) 트리 구조만 잘 만들고 임베딩 정렬이 약하면 검색 품질이 무너진다는 점이다. SproutRAG는 학습 가능한 가중치를 통해 어떤 attention head가 의미적 co-relevance를 잘 포착하는지 스스로 선택하고, 대비 학습 기반 임베딩 손실(검색 품질)과 attention 정규화(트리 구조 품질)를 joint objective로 함께 최적화한다.

- **Empirical Impact**: 네 벤치마크(과학·법률·오픈도메인)에서 SproutRAG는 정보 효율(IE)에서 평균 6.1% 향상을 보이며, Recall이 아니라 Precision까지 함께 개선되는 패턴이 관찰됐다. 또한 HotpotQA/WebQuestions/Dragonball의 생성 평가에서도 온라인 토큰과 지연을 낮춘 채 성능-효율 균형이 좋아, LLM-heavy reasoning·reflection 중심 시스템에 가까운 실용적 대안을 제시한다.



### Redact or Keep? A Fully Local AI Cascade for Educational Dialogue De-Identification (https://arxiv.org/abs/2606.18372)
- **Prior Approaches**: 기존 교육용 de-identification 연구는 개인정보보호 거버넌스와 정확도 사이에서 트레이드오프에 자주 부딪혔습니다. 상용 LLM API는 모호성을 잘 처리하지만 제3자 전송 이슈가 있고, 로컬 NER은 빠르지만 curricular(교육과정) 용어와 실제 학생 이름이 겹칠 때 과도한 과(過)차단(over-redaction) 경향이 있었습니다. 또한 의료 분야처럼 “표면 형태=개체”라는 가정이 교육 대화에는 그대로 적용되기 어렵다는 점이 반복 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 교육 대화 de-identification을 open-ended entity recognition이 아니라 “constrained privacy triage(제한적 프라이버시 판정)”로 재구성한 fully local cascade 프레임워크를 제안합니다. Stage 1은 recall-first로 후보 span을 넉넉히 과생성(precision 희생)하고, Stage 2 reviewer가 각 후보를 문맥과 발화자 역할을 바탕으로 Redact/Keep의 이진 결정을 내립니다. 즉, “무엇을 찾아낼지”보다 “그게 정말 개인정보 위험인지”를 핵심으로 옮긴 것이 기여입니다.

- **Technical Challenges**: 핵심 기술 난제는 curricular-personal name ambiguity처럼 같은 표면형이 서로 다른 의미(수학 개념 vs 실제 학생)를 가질 때, 단순 탐지로는 정확한 판정이 불가능하다는 점입니다. 저자들은 union proposer(DeBERTa+ModernBERT 인코더 + 규칙 기반 RegEx)를 통해 후보를 과생성한 뒤, reviewer를 cascade-aligned 이진 라벨(실제 PII면 Redact)로 학습해 “탐지→프라이버시 검증” 구조로 난제를 분해합니다. 그 결과 small local 모델도 각 후보에 대해 문맥 기반 Redact/Keep을 수행하며, 전체 파이프라인은 단일 노트북에서도 동작하도록 설계했습니다.

- **Empirical Impact**: 수학 튜터링 대화 두 플랫폼(영어)에서 실험한 결과, 가장 강한 구성인 Union + Gemma 31B는 canonical 테스트에서 macro F1 0.958을 달성했습니다. 동일 모델을 단일 패스 LLM-only detector로 썼을 때 macro F1 0.767, 상용 API(Gemini 3.1 Pro) baseline은 0.706에 그쳤고, 전체 과정은 제3자 API 없이 단일 랩탑에서 실행됩니다. 특히 curricular–personal 이름 모호성이 공존하는 challenge set에서는 성능 저하가 0.03 F1로 매우 작았는데, 이는 “교육 de-identification은 문제 재정의가 모델 스케일보다 중요”하다는 결론을 뒷받침합니다.



### Continuous Audio Thinking for Large Audio Language Models (https://arxiv.org/abs/2606.18273)
Comments:
          Preprint

- **Prior Approaches**: 대부분의 Large Audio Language Model(LALM)은 오디오 인코더 출력이 텍스트 토큰 생성에만 간접적으로 연결되며, 다음 토큰 예측 목적이 오디오의 세밀한 음향 정보를 약하게 감독한다. 이로 인해 phonetic detail, prosody, sound events, affect, pitch 같은 프레임 단위 특징이 응답 생성 과정에서 소실되기 쉽다. 한편 text chain-of-thought는 중간 추론을 말로 풀어내지만, 연속 시간/스펙트럼 정보를 자연어로 직렬화하는 데 병목이 생기고 스케일로 확보된 충실한 근거도 부족하다.

- **Core Contribution**: 논문은 Continuous Audio Thinking(CoAT)이라는 프레임워크로, 오디오와 텍스트 사이에 연속 잠재 워크스페이스를 삽입해 음향 정보를 ‘말’ 없이 정리하도록 돕는다. CoAT는 답변 생성 이전에 오디오 전용 연속 thinking block을 두고, 이후 텍스트 응답은 기존 모델과 동일하게 생성한다. 또한 Qwen2-Audio, Qwen2.5-Omni-7B, Audio Flamingo 3 세 가지 백본에 구조 변경 없이 적용 가능하다고 제시한다.

- **Technical Challenges**: 핵심 난제는 텍스트-기반 학습목적만으로는 thinking block을 실제로 유용한 음향 표현 공간으로 조직하기 어렵다는 점이다. 이를 위해 CoAT는 여러 오디오 expert로부터 frame-level 특징을 distillation해 thinking 위치에 신호를 주입하며, reconstruction·speech distillation(표현 기반)과 sound event·emotion(affect)·pitch(작동/과제 특화) 등 상보적 차원을 함께 학습시킨다. 추론 시에는 thinking block을 단일 prefill로 처리해 end-to-end 비용을 추가 autoregressive 디코딩 없이 억제하도록 설계했다.

- **Empirical Impact**: 실험에서는 CoAT가 3개 LALM 전반에서 audio reasoning/understanding, music classification, speech emotion, ASR까지 폭넓은 벤치마크의 다수 지표를 일관되게 개선했으며, 특히 이해·추론 성격의 과제에서 향상이 두드러졌다고 보고한다. text chain-of-thought 대비해서는 추론 정확도는 유지/개선하면서도 TTFT·디코딩 시간·전체 latency 측면에서 더 빠르게 동작하는 경향을 보인다. 추가 분석(선형 프로브)은 보조 감독이 audio-think 위치에 과제 관련 신호를 주입하고, 이후 텍스트 응답의 결정 표현으로 전파됨을 확인해 방법의 내재적 유효성을 뒷받침한다.



### Native Active Perception as Reasoning for Omni-Modal Understanding (https://arxiv.org/abs/2606.19341)
Comments:
          Accepted at ICML 2026. Code and models: this https URL

- **Prior Approaches**: 기존 수동형 passive 모델은 ‘watch-it-all’처럼 프레임을 쿼리 난도와 무관하게 전부 처리해 계산비용이 영상 길이에 비례해 커지는 문제가 있었다. 인터랙티브 에이전트가 나왔지만, 전역 pre-scanning이나 길이에 비례하는 컨텍스트 비용을 유지해 장시간(예: hour-long) 영상에서 픽셀 보관과 계산이 병목이 된다. 또한 도구 기반 멀티모달 모듈은 추론과 지각 사이 정보 병목을 만들고, Think with Images류는 글로벌 버퍼 의존으로 진정한 디커플링이 어렵다는 한계가 지적된다.

- **Core Contribution**: OmniAgent는 멀티모달 비디오 이해를 POMDP 기반 Observation-Thought-Action(OTA) 반복 사이클로 재정의한 네이티브 omni-modal 에이전트다. 에이전트는 필요할 때만 frames/audio/clip을 불러와 오디오-비디오 단서를 ‘영속적 텍스트 메모리’로 선택 증류하고, 원본 미디어의 고차원 컨텍스트를 턴마다 정리해 추론 복잡도를 영상 길이로부터 분리한다. 그 결과 추론 턴을 늘릴수록 성능이 좋아지는 positive test-time scaling 특성이 관찰된다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 장기 에이전틱 추론을 위해 학습 신호가 붕괴되지 않도록 하고, (2) 다중 턴에서 중요한 발견 턴과 사소한 행동 턴이 섞일 때 credit assignment가 왜곡되는 점이다. 이를 위해 OmniAgent는 Agentic Supervised Fine-Tuning에서 best-of-N 궤적 합성과 dual-stage 품질 제어로 네이티브 active perception을 먼저 부트스트랩하고, Agentic Reinforcement Learning에서는 TAURA(Turn-aware Adaptive Uncertainty Rescaled Advantage)로 turn-level entropy를 사용해 Advantage homogenization을 완화한다. 특히 GRPO의 궤적 단일 advantage를 턴 단위로 재가중해, 불확실성 높은 ‘분기 지점’에 더 큰 보상/패널티가 가도록 설계했다.

- **Empirical Impact**: 10개 벤치마크(VideoMME, LVBench 등)에서 OmniAgent는 open-source 모델 중 최신 성능을 달성했으며, 특히 LVBench에서는 7B가 Qwen2.5-VL-72B(10배 더 큰 모델)보다 높은 성적(50.5% vs 47.3%)을 보이면서 프레임도 73% 더 적게 쓴다고 보고한다. 또한 VideoMME-Long에서 추론 턴을 늘릴수록 +6.2% 개선되는 형태의 positive test-time scaling이 실험적으로 확인되어 active perception의 효과를 뒷받침한다. 장시간 비디오 이해에서 ‘길이 증가에 따른 비용 폭증’ 문제에 대한 실질적 해법을 제시했다는 점에서, 멀티모달 에이전트 설계 방향에 영향이 클 전망이다.



### Rethinking Reward Supervision: Rubric-Conditioned Self-Distillation (https://arxiv.org/abs/2606.19327)
- **Prior Approaches**: 추론형 LLM 사후학습은 보통 supervised distillation(지도 증류)이나 reinforcement learning(RL)에서 verifiable reward(검증 가능한 보상)를 활용한다. 하지만 distillation은 chain-of-thought 주석을 비싸게 확보해야 하고, 그럴듯한데도 근거가 노이즈·누락·부분오류일 수 있어 학습을 방해할 수 있다. RL(예: GRPO)은 최종 성공/실패 같은 sparse한 scalar 보상으로만 피드백을 압축해, 어떤 중간 단계가 문제였는지 credit assignment가 어렵다.

- **Core Contribution**: 이 논문은 Rubric-Conditioned Self-Distillation(RCSD)로, 루브릭(rubric)을 구조화된 fine-grained 피드백으로 넣어 self-distillation의 교사 신호를 재설계한다. 핵심은 루브릭 점수처럼 scalar로 접지 않고, criterion-level 루브릭에 조건(condition)된 teacher가 학생이 샘플링한 추론 궤적에 대해 토큰 단위 가이드를 제공하게 하는 것. 또한 Stage I에서 인스턴스별 루브릭을 생성하는 루브릭 생성기를 학습하고, Stage II에서 그 루브릭을 이용해 rubric-guided reasoner를 학습한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 고품질 인스턴스 루브릭을 사람이 매번 만들기 어렵고, (2) 루브릭의 텍스트/기준 정보를 학습 중 토큰 수준으로 어떻게 분해해 전달할지다. RCSD는 Stage I로 질문만 보고도 인스턴스별 루브릭을 amortize(분산) 생성하게 만든 뒤, Stage II에서 teacher를 루브릭에 조건해 forward KL distillation으로 학생의 on-policy rollout에 대해 criterion-aware한 토큰 가이드를 주도록 최적화한다.

- **Empirical Impact**: 실험에서 RCSD는 science 추론 벤치마크 전반에서 평균 70.6을 달성하며 GRPO와 OPSD를 각각 평균적으로 1.4점, 0.9점 앞섰다. 특히 ResearchQA와 RubricHub처럼 루브릭 기반/오픈엔드 과제에서 개선 폭이 크게 나타났고, 의학 도메인(MedMCQA, PubMedQA)에서도 기준선 대비 경쟁력 있는 일반화가 관찰됐다. 또한 학습 목표로 forward KL이 가장 유리하고, 생성된(learned) 루브릭도 reference 루브릭에 근접해 사람이 만드는 수고 없이도 효과를 재현할 수 있음을 보였다.



### Structured Inference with Large Language Gibbs (https://arxiv.org/abs/2606.19264)
Comments:
          Code: this https URL

- **Prior Approaches**: 기존에는 LLM 지식을 활용해 변수를 가진 세계를 추론할 때, 자동회귀 생성 한 번으로 구조화된 해(샘플)를 뽑는 방식이 주로 쓰였다. 하지만 이런 단일 패스 생성은 토큰 생성 순서에 크게 의존해 편향이 생기고, 확률적으로 일관된(확률적 코히어런스) 추론을 보장하기 어렵다. 또한 LLM의 조건부 분포가 잡음이 섞인 형태로만 접근될 때 이를 MCMC 맥락에서 어떻게 다뤄야 하는지도 난제로 남아 있었다.

- **Core Contribution**: 이 논문은 Large Language Gibbs라는 프레임워크를 제안하며, LLM의 next-token conditionals를 상태 전이 연산자로 사용해 구조화된 확률 추론을 수행한다. 단일 패스로 구조 객체를 생성하는 대신, 변수들을 서로 조건으로 두고 하나씩 반복 재샘플링(resample)해 정상분포(stationary distribution)를 만든다. 그 결과 분포는 국소 조건부들 간의 타협을 반영하며, 생성 순서에 따른 편향을 줄이는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술적 도전은 LLM 조건부 분포를 MCMC 전이로 넣을 때, 전이 규칙이 일관된 Gibbs형 업데이트로 작동하면서도 LLM의 잡음 있는 조건부들을 현실적으로 사용할 수 있느냐였다. 논문은 각 변수에 대해 LLM의 next-token conditionals로 조건부 샘플링 연산자를 구성하고, 다른 변수들에 대한 조건을 유지한 채 반복적으로 변수별 재샘플링을 수행한다. 이를 통해 order-dependent bias를 피하고, 정상분포에 수렴하는 형태의 샘플링을 구현한다.

- **Empirical Impact**: 실험은 합성 분포 샘플링, consistent reasoning 과제, Bayesian structure learning에서 Large Language Gibbs의 유효성을 보여준다. 특히 noisy LLM conditionals를 “MCMC에서 쓸 수 있는” 실용적 대안으로 다룰 수 있음을 시사하며, 기존 one-pass generation 대비 확률적 추론 품질과 안정성 측면에서 이점이 관찰된다. 결과적으로 구조화된 확률 추론을 LLM의 조건부만으로 다루려는 연구 흐름에 실질적인 방법론을 제공한다.



### STARE: Surprisal-Guided Token-Level Advantage Reweighting for Policy Entropy Stability (https://arxiv.org/abs/2606.19236)
Comments:
          LLM, Reinforcement Learning

- **Prior Approaches**: RLVR은 수학·코딩 등에서 복잡한 추론을 끌어내기 위해 verifiable rewards를 활용하는 post-training으로 자리 잡았고, 그중 GRPO는 value network 없이 group-normalized advantage로 학습을 단순화했다. 하지만 학습이 길어지면 policy entropy가 급격히 줄며 출력 다양성이 사라지고, 그룹 내 샘플이 동질화돼 상대적 advantage 추정이 약해지면서 장기 학습이 병목된다. 기존 완화는 (1) importance-sampling ratio 클리핑 조정, (2) 양·음 롤아웃에 대한 trajectory-level 가중, (3) entropy 정규화/advantage reshape 같은 token-구조를 세밀히 다루지 못하는 방식에 치우쳐 “붕괴 메커니즘”에 대한 정량적 처방이 부족했다.

- **Core Contribution**: 이 논문은 GRPO에서 token-level entropy 변화가 trajectory-level advantage와 next-token 분포의 entropy sensitivity function의 곱으로 분해된다는 1차 그라디언트 분석을 제시한다. 그 결과, advantage–surprisal이 4분면 구조로 얽히며 저-surprisal 토큰은 entropy 감소 방향 업데이트를 주도하고, entropy를 올려줄 수 있는 고-surprisal 토큰은 희소성 때문에 충분히 반영되지 못하는 “token-level credit assignment mismatch”가 핵심 원인임을 보여준다. 또한 이 mismatch가 near-criticality(임계 부근) 성질을 가져, 토큰 가중치에 대한 비교적 작은 개입으로 entropy 진화 부호를 뒤집을 수 있다고 주장한다.

- **Technical Challenges**: 가장 큰 난제는 entropy 붕괴를 유발하는 “어떤 토큰이” 문제인지 알아내고, 이를 GRPO의 clipped surrogate 안에서 안정적으로 개입하는 설계를 만드는 것이다. 저자들은 이 임계 토큰(고-surprisal 중심)을 이론적으로 정당화하되, 각 위치의 정확한 임계값 계산은 비용이 크기 때문에 batch-internal surprisal quantile로 entropy-critical 부분집합을 근사 선택한다. 이어서 선택된 토큰 집합의 effective advantage를 reweight하고, 배치 평균 entropy가 목표 범위를 벗어나면 작동/복귀하는 target-entropy closed-loop gate를 넣어 entropy를 안정적으로 규제한다.

- **Empirical Impact**: 실험에서 STARE는 1.5B~32B 모델 스케일 전반(Short CoT, Long CoT, multi-turn tool use)에서 수천 스텝에 걸친 RL 학습 동안 policy entropy를 목표 밴드에 유지하며 학습 안정성을 보인다. AIME24와 AIME25에서는 DAPO를 포함한 경쟁 기준선 대비 평균 정확도가 4%~8%p 개선되었고, reflection 관련 토큰 및 응답 길이가 함께 증가해 exploration–exploitation 균형이 오래 유지된다는 관찰이 제시된다. 즉, “entropy collapse의 토큰 수준 원인 규명 → 최소 침습적 reweighting+closed-loop 제어”라는 경로가 장기 post-training의 실질적 성능 향상으로 연결된 셈이다.



### IndicContextEval: A Benchmark for Evaluating Context Utilisation in Audio Large Language Models Across 8 Indic Languages (https://arxiv.org/abs/2606.19157)
Comments:
          Accepted at Interspeech 2026

- **Prior Approaches**: 기존 ASR의 컨텍스트 바이어싱은 외부 언어모델 결합(shallow fusion)이나 디코딩 단계 어텐션, guided-attention 같은 방식으로 구현돼 왔습니다. AudioLLM은 audio+text 프롬프트로 더 자유롭게 컨텍스트를 넣을 수 있지만, 실제로 그 프롬프트를 ‘근거 기반으로’ 쓰는지, 프리트레이닝에 의한 ‘암기’에 기대는지는 기존 벤치마크가 답하기 어려웠습니다. 특히 기존 평가는 프롬프트를 고정하거나(대규모 음성 코퍼스) 영어 중심이며(컨텍스트 ASR 벤치), 자연스러운 다중 컨텍스트 유형을 체계적으로 스위핑하지 못했습니다.

- **Core Contribution**: 본 논문은 AudioLLM의 컨텍스트 grounding(근거화)과 컨텍스트 활용도를 분리해 측정하기 위한 벤치마크 IndicContextEval을 제안합니다. 8개 인도 언어, 555명의 자연 음성, 23개 프로 도메인에서 총 55.93시간 데이터를 구축하고, L0~L6의 7단계 프롬프트 계층으로 컨텍스트 신호를 하나씩 추가합니다. 이를 통해 성능 변화가 ‘어떤 종류의 컨텍스트’에서 오는지, 그리고 잘못된 엔터티 같은 부정 신호에 어떤 반응을 보이는지까지 추적합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 언어/스크립트 차이와 (2) 프롬프트 표현 방식 차이가 성능에 섞여 들어가는 것을 통제하는 것입니다. 논문은 L0(무프롬프트)부터 L1(언어 지정)로 시작해, L2~L5에서 메타데이터·자연어 오디오 설명·엔터티 리스트(영어/원어 스크립트)를 각각 추가하며, L6에서는 무관 도메인의 적대적 엔터티로 negative control을 둡니다. 또한 모든 모델 출력은 목표 언어의 native script로 제한하고 WER과 NEER(네임드 엔터티 오류율)을 함께 측정해 컨텍스트 신호에 대한 ‘실제 근거 사용’ 여부를 점검합니다.

- **Empirical Impact**: 5개 AudioLLM을 평가한 결과, 모델마다 컨텍스트 활용 패턴이 크게 갈렸고, 컨텍스트 grounding은 여전히 미해결 과제로 드러났습니다. 특히 native-script 엔터티 바이어싱(L5)이 NEER 개선을 가장 크게 만들었으며, 영어 스크립트 엔터티(L4) 대비 원어 스크립트(L5)에서 최대 11 WER 포인트 수준의 ‘스크립트 미스매치 비용’이 관측됐습니다. GPT-4o Transcribe는 정확 엔터티 이득(L5)을 취하면서도 적대적 엔터티(L6)에선 거의 흔들리지 않아 선택적 활용을 보였고, Gemma-3N은 엔터티에는 반응하지만 주변 전사 품질이 붕괴되는 불안정 활용 양상이 나타났습니다. IndicContextEval은 컨텍스트 유형을 체계적으로 바꿔 평가함으로써, 앞으로 AudioLLM의 컨텍스트 근거화 능력을 검증·개선하는 데 직접적인 기준점을 제공할 것으로 기대됩니다.



### Human-AI Coevolution Dynamics: A Formal Theory of Social Intelligence Emergence Through Long-Term Interaction (https://arxiv.org/abs/2606.19144)
- **Prior Approaches**: 기존 대화형 AI는 감정 모델링, 메모리 검색, 페르소나 조건화처럼 사회적 행동을 구성요소 단위로 분리해 다루는 경우가 많습니다. 그 결과 장기적인 상호작용에서 안정적인 사회적 관계가 어떻게 형성·유지되는지, 사회지능이 어떤 동역학으로 ‘나타나는지’를 하나의 틀로 설명하기 어렵습니다.

- **Core Contribution**: 이 논문은 Human-AI Coevolution Dynamics Framework (HACD-H)를 제안하며, 인간-에이전트 상호작용을 자기조직화되는 사회적 인지 시스템으로 형식화합니다. HACD-H는 감정 적응, 관계 조직, 사회적 기억, 성격 일관성을 단일 동역학 프레임워크로 통합하고, 다중 timescale 사회인지, relational attractors, trust basins, 발달 단계 전이, social cognitive energy 같은 원리를 제시합니다.

- **Technical Challenges**: 핵심 과제는 장기 상호작용의 복잡한 사회적 현상을 분리된 모듈이 아니라 ‘동역학적 체계’로 모델링하는 데 있으며, 이를 검증 가능한 이론-데이터 연결로 바꾸는 것입니다. 논문은 약 14,700개의 interaction turn으로 이론 기반 대화 데이터셋과 평가 프레임워크를 구축하고, 에너지 지형·단계 전이·관계적 끌개를 실증적으로 관찰할 수 있게 설계했습니다.

- **Empirical Impact**: 실험 결과 사회인지에는 시간 스케일별 지속성의 위계가 존재하고, stable relational attractors와 phase-transition-like 발달 패턴이 나타납니다. 또한 social intelligence는 social cognitive energy와 유의한 음의 상관관계를 보이며(r = -0.391, p < 0.001), 상호작용 궤적에서 에너지가 점진적으로 감소하는 양상이 보고되어 사회지능이 고립된 대화 능력보다 장기 공진에서 유래함을 시사합니다. HACD-H는 적응형 인간-에이전트 사회 상호작용을 통합적으로 모델링하고, 사회적으로 지능적인 AI를 설계하는 데 이론적 기반을 제공한다는 점에서 의미가 큽니다.



### Urdu Katib Handwritten Dataset: A Historical Document Dataset for Offline Urdu Handwritten Text Recognition with CRNN-Based Baseline Evaluation (https://arxiv.org/abs/2606.19139)
- **Prior Approaches**: 우르두 UHTR은 Arabic 계열의 커시브 특성 때문에 기존 OCR/HTR 대비 어려움이 크며, 연구는 상대적으로 소수에 그쳤습니다. 특히 문맥 민감성, 겹침, 대각선 필기, 점(누크타) 위치의 복잡성, 형태 유사성 때문에 문자 단위 인식이 흔들립니다. 또한 공개 벤치마크 데이터셋이 부족해 모델 학습과 공정한 비교가 제한되어 왔습니다.

- **Core Contribution**: 이 논문은 역사적 우르두 카티브(Katib)가 남긴 필체에서 추출한 오프라인 우르두 필기 텍스트 라인 데이터셋인 Urdu Katib Handwritten Dataset(UKHD)을 제안합니다. UKHD는 Nastalique 서체의 flat nib 필기 변형을 반영하며, ‘Plain Urdu Text Lines(PUTL)’와 ‘Mixed Urdu Text Lines(MUTL)’ 두 부분으로 구성됩니다. 아울러 UKHD의 주요 부분에 대해 CRNN 기반 하이브리드 모델들을 비교해 Urdu Katib Handwriting Recognition(UKHR)의 기준선과 최적 아키텍처를 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 먼저 복잡한 필기 이미지를 라인 세그멘테이션과 정확한 전사 라벨링이 가능하도록 만드는 것입니다. 논문은 PDF 원본에서 페이지 이미지를 추출한 뒤, RGB→그레이스케일 변환, 페이지 뒷면 그림자 잡음을 완화하기 위한 median 필터링, Horizontal Projection Profile(HPP) 기반 deskew로 텍스트 기울기를 보정하는 전처리 파이프라인을 구성합니다. 이어서 라인 분할 및 (세미-)자동 라벨링 절차를 도입해 데이터셋 제작 시간을 줄이는 방식으로 확장성을 확보했습니다.

- **Empirical Impact**: 실험에서는 CNN-BGRU-CTC 모델이 다른 CRNN 하이브리드 구성보다 Character Error Rate(CER)과 Word Error Rate(WER) 측면에서 더 견고한 성능을 보였다고 보고합니다. 이는 데이터 부족으로 인해 난관이었던 UHTR 연구에 대해, 역사적 flat nib 필체까지 포함한 학습·평가 기반을 제공한다는 점에서 의미가 큽니다. 결과적으로 우르두 필사 문헌의 디지털 보존과 검색 가능성을 높이는 인식 시스템 개발을 촉진할 것으로 기대됩니다.



### Written by AI, Managed by AI: Semantic Space Control and Index Sickness Elimination Across 391 Consecutive Sessions (https://arxiv.org/abs/2606.19121)
Comments:
          22 pages, 2 tables, 1 figure. Action research. Bilingual submission (Chinese companion version included as supplementary). Submitted to ICSE 2027 IOR track

- **Prior Approaches**: 장기 LLM 협업에서 개념 드리프트를 줄이기 위해, 더 강한 제약을 추가하는 방향(상징적 identifier 시스템, System Prompts의 방어 규칙 누적, context window 확장)이 널리 사용돼 왔다. 하지만 저자들은 이런 공학적 직관이 장기 환경에서 역효과를 낼 수 있음을 실제 프로젝트 기록으로 보여준다. Bang-v3의 약 한 달, 391개 협업 세션을 action research로 분석해 실패 과정을 정리한다.

- **Core Contribution**: 논문은 이러한 접근이 특정 임계점을 넘으면 정확도가 오르지 않고, LLM이 업무 의미(semantic)를 실제 이해하지 못한 채 상징 계층 내부의 자기참조적 추론으로 후퇴하는 실패 패턴을 “Index Sickness”로 명명한다. 대표적으로 “Phantom Legislation”은 그럴듯한 내부 일관성을 만들어내지만 현실 세계의 물리적/업무적 정합성과는 단절되는 현상을 가리킨다. 그 원리를 “Pang Principle (Semantic Vitality Law)”로 정리하며, 목적이 명시된 자연어가 상징 표현보다 정보 품질이 훨씬 높다고 주장한다.

- **Technical Challenges**: 핵심 난제는 상징 계층이 커질수록 모델이 진짜 의미 이해를 유지할지, 아니면 상징 표면에서 닫힌 계산으로 붕괴할지 예측·방지하는 것이다. 저자들은 “Baseline-Log Physical Separation”이라는 물리적 엔지니어링 메커니즘을 설계해, 기준선(baseline)과 로그(log)를 분리해 상징 인덱스에 대한 의존이 과도해지지 않도록 제어했다. 동시에 Instruction의 비대화를 줄여 상징적 복잡도 임계점에 도달하는 경로를 차단했다.

- **Empirical Impact**: Bang-v3에서 이 메커니즘은 AI Instructions 볼륨을 약 75% 줄였고, 이후 약 150개 세션 동안 Index Sickness의 재발이 관찰되지 않았다. 즉, 제약 강화·상징 확장 중심의 설계가 장기 과제에서 유발할 수 있는 역효과를 실제로 억제하는 검증 결과를 제시한다. 장기 LLM 협업 설계에서 “자연어 목적성 유지”와 “상징 계층의 과도한 비대화 방지”를 실무적 기준으로 삼을 근거를 제공한다.



### Mitigating Scoring Errors and Compensating for Nonverbal Subtests in Speech-Based Dementia Assessmen (https://arxiv.org/abs/2606.18979)
Comments:
          Accepted at INTERSPEECH 2026

- **Prior Approaches**: 기존 음성 기반 치매/인지장애 평가는 (1) 특정 유도 발화에서 특징을 뽑아 MMSE·MoCA 같은 척도를 예측하거나, (2) Boston Naming·Verbal Fluency 등 표준화 검사를 ASR로 자동 채점하는 방식이 중심이었습니다. 다만 SKT처럼 여러 인지 도메인(언어·주의·기억)과 더불어 비언어(운동) 하위검사가 포함된 배터리는, 한두 과제만 다루면 초기(mild cognitive impairment) 민감도가 떨어질 수 있고 음성 전사 오류가 채점에 직접 전이되는 문제가 큽니다.

- **Core Contribution**: 이 논문은 독일의 표준 치매 선별검사 Syndrom-Kurz-Test(SK T, SKT)를 대상으로 ‘음성-only’ 자동 평가를 end-to-end에 가깝게 구현합니다. 핵심은 Whisper 전사로부터 얻은 규칙 기반(subtest 점수) 신호와 encoder/decoder embeddings를 결합해 전사 유발 채점 오차를 줄이고, 말로 측정되지 않는 운동 하위검사는 남은 언어 하위검사 조합을 통해 전문가 전체 평점에 근접하도록 보상하는 deep correction·deep compensation 프레임워크를 제시한 점입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 방언·병적 말·과제 특유의 비정형 발화(예: 숫자 세기, 간섭 검사)로 인해 WER이 흔들리면 점수 산정이 연쇄적으로 흔들린다는 점, (2) SKT의 일부 하위검사가 실제로는 말이 아니라 행동(운동) 기반이라 음성만으로는 직접 점수를 만들 수 없다는 점입니다. 저자들은 Whisper에서 단어 타임스탬프와 함께 encoder/decoder 임베딩을 뽑고, 규칙 기반 점수(RB)에 임베딩 정보를 결합해 corrected subtest 점수를 예측하며, 이어서 누락된 운동 하위검사를 대체할 수 있도록 누적 하위검사 모델을 단계적으로 학습해 전체 점수를 approximating합니다.

- **Empirical Impact**: 루틴 임상 데이터(158명, NCI/MCI/DEM)에서 평가한 결과, RB+ENC/RB+DEC는 전문가 점수와 강한 상관을 보였고 전사 오류가 큰 하위검사에서 보정 효과가 더 두드러졌습니다. 더 나아가 운동 과제(4,5)를 제외하고도 남은 언어 하위검사만 순차적으로 처리했을 때 SKT total score와의 상관이 최대 약 0.94~0.95 수준까지 도달했으며, 진단 구분(confusion matrix)에서도 간섭/기억/숫자세기 과제가 분별력을 키우는 패턴이 관찰되었습니다. 이는 초기 인지장애 선별에서 ‘음성 기반 접근성’의 실용성을 높이는 동시에, 임상 보조·자동화 채점 시스템에서 전사 오류 내성을 갖춘 설계 방향을 제시한다는 점에서 의미가 있습니다.



### Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents (https://arxiv.org/abs/2606.18947)
Comments:
          15 pages, Figure 8

- **Prior Approaches**: 기존 접근은 LLM의 native search grounding처럼 모델-공급자 경계 안에 검색 정책(프로바이더 선택, 결과 형식, 증거 주입, 비용·지연)을 숨겨두는 방식이 주류입니다. 이 때문에 검색 품질과 운영 지표를 튜닝·검사·이식·재사용하기 어렵고, 엄격한 출력 계약(예: JSON/단일 엔티티)을 깨는 Search-Induced Verbosity 같은 포맷 드리프트 위험이 커집니다. 또한 RAG나 도구 사용 연구는 검색-추론 상호작용을 다루지만, ‘실시간 검색 인터페이스’를 명시적 시스템 경계로 다루는 관점은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 Vendor-agnostic 경계로 검색 근거를 분리하는 Decoupled Search Grounding(DSG)을 제안합니다. DSG는 MCP-compatible gateway를 통해 검색을 추론 모델 바깥의 구조화된 tool 계층으로 옮기며, 프로바이더 라우팅, 출처 기반 context 렌더링, fallback, retrieval-depth, exact/semantic caching을 1급 제어(control)로 노출합니다. 결과적으로 추론 모델은 교체 가능하게 유지하면서도, 검색은 ‘운영 가능한 인터페이스’로 취급할 수 있게 됩니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 프로바이더별 이질적인 결과를 표준화해 모델이 일관된 근거를 받게 하고 (2) 캐시·fallback·비용/지연 목표를 정책으로 안정화하며 (3) 출력 계약 위반(Search-Induced Verbosity)을 인터페이스 레벨에서 완화하는 것입니다. DSG는 provider registry와 YAML 어댑터로 다양한 검색 백엔드를 같은 내부 결과 객체로 정규화하고, 캐시 키를 provider-scoped로 관리해 서로 다른 공급자 결과가 섞여 재사용되는 문제를 줄입니다. 또한 semantic cache의 유사도 임계값과 time-to-live(신선도 우선)를 설정해 재현성과 최신성 간 균형을 맞추며, 툴 호출/응답이 명확한 경계가 되도록 설계합니다.

- **Empirical Impact**: 5개 frontier 모델과 SimpleQA/FreshQA/HotpotQA, 그리고 e-commerce Query Intent Understanding(QIU) 프로덕션 워크로드에서 native search와 비교해 비용·지연·품질 트레이드오프를 체계적으로 입증합니다. SimpleQA에서 DSG는 86.1%(native 87.7%)에 근접하면서 검색 비용은 91% 절감했고, warm-cache hit rate는 99.4%, 지연은 68% 감소했습니다. FreshQA는 native가 앞섰지만(만약 신선도 이점이 강한 경우), QIU에서는 DSG가 native와 비슷하거나 약간 상회하면서 검색 비용을 98% 이상 줄였으며, 모델 공급자에 종속되지 않는 ‘운영 최적화 가능한 검색 경계’의 실용성을 강조합니다.



### Graph-ESBMC-PLC: Formal Verification of Graphical PLCopen XML Ladder Diagram Programs Using SMT-Based Model Checking (https://arxiv.org/abs/2606.18941)
Comments:
          18 pages

- **Prior Approaches**: ESBMC-PLC는 IEC 61131-3 Ladder Diagram(LD)의 PLCopen XML 중 텍스트 <rung>만 제대로 LD-to-GOTO-IR 변환을 지원했다. 반면 그래픽 tc6_0201 입력은 파서는 통과했지만 <rung>이 없다고 판단해 rung 논리를 비우고, 0으로 초기화된 변수만으로 형식검증을 수행해 SAFE 판정이 공허(vacuous)해지는 문제가 있었다.

- **Core Contribution**: Graph-ESBMC-PLC는 그래픽 PLCopen XML LD의 connection graph(localId/refLocalId, leftPowerRail/rightPowerRail)를 DFS로 해석해 rung 경로를 불리언 접점 결합식으로 추출한 뒤, 기존 ESBMC-PLC의 GOTO IR 백엔드를 그대로 재사용한다. 특히 SET/RESET 래치 코일의 scan-cycle 의미를 맞추기 위해 rightPowerRail의 connectionPointIn 순서로 코일 처리 우선순위를 정한다.

- **Technical Challenges**: 핵심 과제는 “시각적 배선”을 directed graph 경로 집합으로 정확히 복원하고, 그 결과를 ESBMC가 기대하는 GOTO IR의 rung/코일 모델에 접점-논리(AND/OR)로 변환하는 것이다. 논리 변환은 그래프에서 leftPowerRail→각 coil까지의 단순 경로를 전부 열거해 경로별 contact conjunction을 만들고, 여러 경로는 병렬(OR)로 합친다; 또한 I/O 분류가 없을 때를 대비해 %IX/%QX 기반의 address-based 추론(1순위)과 접점/코일 등장 양상 기반 휴리스틱(2순위)을 3단계로 적용해 과잉추정(over-approximation)으로 sound SAFE를 보장한다.

- **Empirical Impact**: CONTROLLINO/OpenPLC Editor에서 나온 그래픽 LD 3개에 대해 기존에는 비어 있던 GOTO IR가 이번 작업 후 full GOTO IR로 생성되었고, 입력이 nondeterministic인 상태에서도 SAFE가 k=2에서 70ms 미만으로 검증됐다. 또한 존재하는 11개 텍스트 LD 벤치마크는 그대로 보존되어 회귀가 없었으며, 다만 Beremiz 예제 중 LD 내용이 없는 경우와 미지원 timer semantics가 있는 경우는 실제 한계로 투명하게 보고했다. 결과적으로 그래픽 tc6_0201 형식의 형식검증 신뢰도를 크게 끌어올려 PLC 안전검증 파이프라인의 적용 범위를 확장한다.



### REVES: REvision and VErification--Augmented Training for Test-Time Scaling (https://arxiv.org/abs/2606.18910)
- **Prior Approaches**: 기존 test-time scaling(TTS) 연구는 배포 시나리오에 맞춰 최적화를 해야 한다고 강조해왔습니다. 하지만 sequential revision(SR) 같은 다단계 추론에서는, RLHF/RLVR/GRPO처럼 단일-shot 목표를 직접 최적화하면 다단계 실행 중 상태 전이와 복구(recovery) 역학이 어긋납니다. 다중 턴 RL로 해결하려 해도 보통 궤적 전체에 대해 경로 의존적 크레딧을 뿌리기 때문에, 중간의 “틀린” 단계까지 동일한 긍정 신호를 받아 학습 신호가 거칠고 편향됩니다.

- **Core Contribution**: 이 논문은 SR의 목적함수(JϕSR)를 “단계별 한 번에 복구될 확률”들의 합으로 정확히 분해해, 학습을 수평적(전체 궤적) 크레딧 할당이 아니라 수직적(각 상태의 1-step recovery) 감독으로 바꿀 수 있음을 보입니다. 그 위에 REVES라는 2-stage 반복 프레임워크를 제안하며, 성공 궤적에서 나온 중간 near-miss 답을 revision 프롬프트와 verification 프롬프트로 분리해 모델이 오류를 식별하고 고치는 능력에 집중하도록 학습시킵니다. 결과적으로 긴 horizon의 다중 턴 롤아웃 비용을 줄이면서 SR에 직접 최적화합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다단계 SR에서 terminal reward 중심의 다중 턴 RL 신호를, (2) horizon credit assignment 없이 로컬 복구 학습 신호로 재구성하는 것입니다. 저자들은 JϕSR를 per-state one-step recovery 확률들의 가중합으로 “정확히” 분해해 편향이 줄어든 학습 목표를 설계했고, 이를 구현하기 위해 매 epoch마다 SR을 돌려 성공한 trajectories의 방문 상태를 오프라인으로 모은 뒤, 그 상태에서 single-turn RL로 학습하는 구조를 채택합니다. 또한 실제 배포에서는 오라클 stopping time이 없으므로 verification 프롬프트를 함께 학습 데이터에 포함해 self-stop이 가능하도록 했습니다.

- **Empirical Impact**: 실험에서 REVES는 LiveCodeBench에서 RL baseline 대비 +6.5포인트, 표준 multi-turn training 대비 +4.0포인트 개선을 보입니다. 코딩 외에 circle packing에서는 4B 같은 더 작은 베이스 모델로도 더 큰 evolutionary search 시스템과 맞먹는 기존 SOTA급 성능을 달성하며, MATH500/AIME24/25에서도 ground-truth 및 self-confidence stopping 설정 모두에서 교정(corrective) 능력 향상이 확인됩니다. n_queens/mini_sudoku처럼 제약 자체로 정답이 판정되는 out-of-distribution 퍼즐에서도 일반화가 관찰되며, 더 나아가 MCTS 계열 등 revision-using TTS 전반에서 성능이 동반 상승해 전이 관점의 의미가 큽니다.



### GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents (https://arxiv.org/abs/2606.18829)
Comments:
          24 pages, 8 figures. Code and dataset are available at this https URL and this https URL

- **Prior Approaches**: 기존 LLM 에이전트 메모리 벤치마크는 주로 단일 사용자 가정에 머물러, 병원·직장·캠퍼스·가정처럼 여러 이해관계자가 함께 쓰는 환경이 상대적으로 덜 연구돼 왔습니다. 특히 공용 메모리 풀에서 서로 다른 역할·권한·관계로 질의하면, 단순 검색 정확도만으로는 메모리 품질을 설명하기 어렵고 거버넌스 이슈가 핵심으로 부상합니다.

- **Core Contribution**: 이 논문은 multi-principal shared-memory 에이전트를 위한 벤치마크 GateMem을 제안합니다. GateMem은 장기 요청의 유용성, 역할·범위·컨텍스트 인증 경계에서의 접근 제어, 그리고 삭제 요청 이후 에이전트가 수행해야 하는 active forgetting을 함께 평가합니다.

- **Technical Challenges**: 문제는 (1) 여러 주체의 상태 업데이트가 뒤엉킨 공용 메모리에서 정당한 요청만 유도하는 동시에, (2) 권한 경계 밖 정보 접근을 막고, (3) 명시적 삭제 이후 실제로 잊는지까지 검증하는 데 있습니다. 이를 위해 GateMem은 긴 포맷의 다자 에피소드, 증분 메모리 주입, 숨은 체크포인트, 구조화된 채점, 그리고 누출-타깃 주석 같은 평가 장치를 통합합니다.

- **Empirical Impact**: 실험 결과 다양한 베이스라인과 백본 모델 전반에서 utility(유용성)·access control(접근 제어)·forgetting(신뢰 가능한 망각)을 동시에 강하게 달성하는 방법은 없었습니다. long-context prompting은 토큰 비용이 크지만 거버넌스 점수가 상대적으로 높았고, retrieval 기반·external-memory 방법은 비용을 낮추더라도 권한 없는 정보나 삭제된 정보가 새는 현상이 남았습니다. 이로써 현재 메모리 에이전트 기술은 기관 단위의 신뢰 가능한 shared deployment에는 아직 크게 부족하다는 메시지를 실증적으로 제공합니다.



### HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Spac (https://arxiv.org/abs/2606.18788)
- **Prior Approaches**: 기존 필기체 생성은 GAN·diffusion·transformer 같은 딥러닝 기반이 주류였고, 온라인/오프라인 필기 모두에서 성능이 크게 올랐습니다. 다만 스타일별 아키텍처·대규모 데이터·막대한 compute 의존성이 커서 새로운 필체나 low-resource 환경 적응이 어렵고, raster 중심 출력이라 획(스트로크) 수준 제어와 편집성이 제한됩니다. 또한 언어/스크립트마다 특화된 구조가 필요한 경우가 많아 다국어·다도메인 확장이 번거롭다는 한계가 남아 있습니다.

- **Core Contribution**: HandwritingAgent는 자연어로 제어되는 에이전트 방식으로, SVG(Scalable Vector Graphics)에서 획 시퀀스를 직접 이산적으로 생성해 필체를 합성합니다. 스타일별 학습 없이도 입력 스타일 이미지(또는 stroke)와 요청 텍스트(대화형/비대화형)를 함께 받아, 언어 모델이 기하 단서를 분석해 glyph를 단계적으로 계획·생성하도록 설계했습니다. 그 결과 해상도에 독립적이고 편집 가능한(해석 가능한) 벡터 출력과, 스크립트·도메인 전반의 일반화가 강조됩니다.

- **Technical Challenges**: 핵심 난제는 필기 스타일의 변동성이 ‘형상·질감·압력·연결’처럼 복합적이면서도, 이를 자연어 지시와 정합되게 획 수준으로 재현해야 한다는 점입니다. 논문은 (1) 입력을 공용 grid-canvas 좌표계와 구조화된 XML 표현으로 정규화하고, (2) LLM OCR+휴먼 인 더 루프 보정으로 캐릭터/워드 단위 분할과 라벨링 정확도를 끌어올리며, (3) glyph bank로 참조 스타일의 구조·자간·곡률 성향을 고정해 추론 기반 style transfer를 수행하는 흐름을 제시합니다. 또 cubic Bézier 곡선과 시간(temporal) 값을 포함한 stroke-point 시퀀스를 SVG path로 변환해 연속성과 자연스러운 쓰기 역학까지 유지하려고 합니다.

- **Empirical Impact**: 실험에서는 IAM(모방)·CASIA/IAM-LINES(다국어)·CROHME/EDU-CHEMC/자체 physics 노트(수학·과학) 등 다양한 태스크에서 기존 generative handwriting 모델과 비교해 성능을 입증했습니다. IAM Word/Line에서 SSIM·FID·HWD가 상위권이고, 가독성 측면에서도 ΔΔCER 격차가 크지 않아 ‘시각적 구조 보존’이 특히 강점으로 나타납니다. 또한 중국어처럼 라틴을 넘어서는 스크립트 일반화와, 수학/과학 표현에서 ExpRate·WER 등 인식 기반 지표가 경쟁력 있게 개선되며, reasoning(생각 모드) 유무에 따른 성능 차이로 추론이 장문 구조·연속성 유지에 실제로 기여함을 보여줍니다.



### SAMA: Semantic Anchor-aligned Augmentation for Unified Low-Resource Multimodal Information Extraction (https://arxiv.org/abs/2606.18780)
Comments:
          Accepted by IEEE Transactions on Multimedia

- **Prior Approaches**: 기존 Multimodal Information Extraction(MIE)용 Data Augmentation은 텍스트와 이미지를 각각 변형하거나 모달리티 간 정렬을 거칠게 처리하는 경우가 많아, 생성된 텍스트-이미지 쌍 사이의 의미 정합성이 깨지기 쉽습니다. 또한 MNER/MRE/MEE를 위한 증강 파이프라인이 과제별로 분절돼 공유 의미를 재사용하지 못해 저자원 환경에서 성능이 제한됩니다. 더불어 closed-source Multimodal LLM 기반 증강은 비용·지연 문제가 크고 MIE의 스키마 제약(정확한 entity/span, relation triplet, event 구조)을 잘 따르지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Semantic Anchor-aligned Multimodal Augmentation(SAMA)라는 통합 프레임워크로 MNER, MRE, MEE에 공통으로 적용 가능한 고품질 합성 데이터를 생성하는 방법을 제안합니다. 핵심은 ground-truth에서 구조화된 semantic anchors를 만들고, 이를 Collaborative Multi-Experts Multimodal Large Language Model(CME-MLLM)의 텍스트 생성과 Anchor-Preserving Diffusion의 이미지 합성에 동시에 조건으로 거는 것입니다. 결과적으로 다양성은 확보하면서도 “스키마 준수 + 교차모달 일치”를 동시에 강제하는 증강을 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 생성이 자연스러워지는 대신 entity 타입/관계 방향/트리거-인자 구조가 흔들리는 semantic drift를 막는 동시에, 텍스트-이미지 정렬을 정밀하게 유지하는 것입니다. SAMA는 (1) inline markup 기반 semantic anchors로 생성 확률공간을 유효 스키마로 제한하고, (2) Universal Adapter와 task-specific Adapters를 anchor-gate로 혼합해 공유 의미와 과제 특이 제약을 동시에 학습·생성하며, (3) anchor 가중 prompt와 마스크 latent blending으로 확산 과정에서 핵심 엔티티의 시각 정체성을 보존합니다. 마지막으로 Dual-Constraint Filtering으로 CLIP 기반 cross-modal consistency와 anchor fidelity를 함께 만족하는 후보만 선택해 수동 검증 없이 신뢰도를 확보합니다.

- **Empirical Impact**: Twitter-15(MNER), MNRE(MRE), M2E2(MEE) 전 벤치마크 실험에서 SAMA는 기존 증강 베이스라인을 저자원(10%, 20%, 40%)과 준전데이터 조건 모두에서 일관되게 능가합니다. 특히 극단적 저자원(10%)에서 task-specific SOTA 대비 MNER는 F1 +1.7%, MRE는 약 +2.0%(베이스라인 HVPNeT 기준), MEE는 MEAE/ MED에서 각각 +4.7% / +5.9%로 큰 격차를 보였습니다. 이는 구조화된 semantic anchors 기반의 정합성 중심 증강이 단순 보편 증강(MixGen)이나 단일 모달/과제편향 증강의 한계를 넘어, MIE의 학습 신호 품질을 실질적으로 끌어올린다는 점에서 의미가 큽니다.



### Attention as Frustrated Synchronization (https://arxiv.org/abs/2606.18694)
Comments:
          25 pages, 4 figures. Preliminary report at the 1-10M parameter scale

- **Prior Approaches**: 자기회전( self-attention )은 층이 깊어질수록 토큰 표현을 합의 쪽으로 수렴시키는 경향이 있다. 이 때문에 Kuramoto attention처럼 위상을 동기화시키는 주의는 “응집(agreement)”에는 강하지만, 연속/다음 상태(continuation)까지는 완전히 재현하지 못해 기준선(transformer)과 격차가 남았다. 저자들은 특히 복제(copy) 길이가 길어지는 구간에서 그 격차가 더 크게 드러난다고 분석한다.

- **Core Contribution**: 논문은 Frustrated Synchronization Network(FSN)을 제안하며, 동기화 자체가 아니라 “동기화의 구조적 불일치(frustration)”를 값 경로(value pathway)에 배치한다. 토큰의 상태를 torus 위의 위상으로 두고, 전체 값 경로를 조화(harmonics) 위의 한 개의 학습된 복소 커플링 커널과 1-step delay로 정의해 next-token prediction을 데이터가 주는 전이(transition)로 좌절된 동기화로 구현한다. 커널 각 항은 동기화 문헌에서 이름이 붙은 구성요소로 읽히도록 설계해, 무엇이 학습되는지 해석 가능성도 함께 노린다.

- **Technical Challenges**: 핵심은 (1) 토러스 위 위상을 다루되 커플링을 복소수/조화 구조로 안정적으로 학습시키고, (2) 다음 토큰 정보를 “현재의 위상”이 아니라 “전이(다음으로 넘어가는 변화)”에 정확히 담는 것이다. 저자들은 기본 Kuramoto 주의는 유지하되 값 경로의 업데이트를 다조화 Daido/Kuramoto–Sakaguchi 형태로 바꾸고, 특히 delay 항이 “데이터 전이 δu를 frustration angle로 쓰는 커플링”과 대수적으로 동일하다는 점을 활용한다. 또한 위상 각이 대칭점에서 구분되지 않을 수 있어 frustration angle을 작은 난수로 초기화해 학습 초기의 대칭 붕괴를 유도한다.

- **Empirical Impact**: 동일한 학습 예산과 레시피를 맞춘 character-level 실험에서 FSN은 enwik8에서 매 epoch 기준으로 튜닝된 RoPE-SwiGLU transformer의 validation loss를 추월하며, 수렴 후에도 더 낮은 값(예: 50 epoch에서 1.5953±0.0014)을 보고한다. copy depth 분해에서도 깊이가 4를 넘는 구간에서 transformer 대비 이점이 일관되게 나타나, “연속성/복제에 대한 취약점”을 직접 겨냥해 개선됐음을 뒷받침한다. feed-forward(SwiGLU)를 mean-field 집단 모드로 대체한 FSN-MF는 transformer 급에는 근접하지만 완전한 재현은 못했으며, 전체 향상분 중 일부가 oscillator 기반 커플링 밖의 feed-forward 기여임을 수치로 분해해 보여준다.



### ForecastBench-Sim: A Simulated-World Forecasting Benchmark (https://arxiv.org/abs/2606.18686)
Comments:
          15 pages, 5 main figures, 6 appendix figures. Spotlight presentation at Forecasting as a New Frontier of Intelligence / Workshop on AI Forecasting, ICML 2026

- **Prior Approaches**: ForecastBench 등 기존 예측 벤치마크는 실제 세계 질문을 그대로 가져오지만, 결과가 늦게 확정되고 꼬리사건이 드물며 반사실(counterfactual) 질문은 보통 점수화가 어렵다는 구조적 제약이 있습니다. 시뮬레이션을 활용한 대안도 있었으나, 주로 과거 데이터를 재활용하거나(예: FutureSearch) 단순 그래프 기반의 인과 추론에 머물러 동적 다중 에이전트 세계의 경로 의존성을 충분히 활용하기 어렵다는 한계가 지적됩니다. 

- **Core Contribution**: 이 논문은 Freeciv 턴제 전략 게임 롤아웃을 기반으로 한 시뮬레이션 예측 벤치마크 ForecastBench-Sim을 제안합니다. 고정된 world report(게임 상태 스냅샷)를 주고 숨겨진 미래를 예측하게 만든 뒤, 시뮬레이터를 계속 진행해 정답을 즉시 해소하고 이진/연속(분포) 질문을 같은 인터페이스로 채점합니다. 또한 개입(intervention) 월드를 포크해서 조건부·인과 질문을 실제로 점수화할 수 있게 설계했습니다. 

- **Technical Challenges**: 핵심 난제는 “예측 과제의 공정성”을 유지하면서도 시뮬레이션의 장점(반사실/즉시 해소)을 평가에 자연스럽게 결합하는 것입니다. 저자들은 모델이 직접 시뮬레이터 상태가 아닌 report를 보게 하고, 연속 예측은 분위수 p10~p90로 elicitation한 뒤 CRPS로 채점하며, 스코어는 템플릿별 기준 범위로 정규화해 비교 가능하게 했습니다. 조건부·인과 스코어링을 위해서는 savegame을 수정해 분기된 rollout을 생성하고, 동일 질문 템플릿을 baseline과 intervention framing으로 페어링해 점수 차이를 측정합니다. 

- **Empirical Impact**: 검증 결과, ForecastBench-Sim의 이진 Brier는 기존 실세계 벤치마크(ForecastBench) 성과 및 ECI(Epoch Capabilities Index)와 유의미한 상관을 보이며, horizon이 길어질수록(예: H1→H7) 불확실성이 증가하는 전형적인 예측 난이도 패턴도 관찰됩니다. H0(리포트 이해 확인) 체크에서 대부분의 모델이 낮은 오차를 보여, 성능 저하가 단순히 report 파싱 실패 때문이 아님을 시사합니다. 소규모 익명 인간 파일럿도 동일 과제 제시가 가능함을 보여주며, 후속 연구에서 calibration, 인과적 업데이트, 꼬리위험(tail-risk) 같은 주제를 더 통제적으로 다룰 수 있는 평가 기반을 제공한다는 점에서 의미가 있습니다.



### EARS: Explanatory Abstention for Reliable Sub-Agent Modeling in Large-scale Multi-Agent Systems (https://arxiv.org/abs/2606.18668)
- **Prior Approaches**: 기존 centralized multi-agent systems(MAS)는 coordinator가 사용자 요청을 해석해 sub-agent로 라우팅하고 결과를 통합해 답을 내는 구조가 주류였습니다. 그러나 선행 연구는 주로 coordinator의 routing/역할 선택 같은 상위 조정 실패를 줄이는 데 집중했고, sub-agent가 실행 중 실패할 때 그 신호가 coordinator와의 통신에서 어떻게 깨지는지는 상대적으로 덜 다뤄졌습니다. 또한 single-agent에서의 refusal(거절) 연구는 있었지만, MAS에서 abstention이 ‘협업 복구를 위한 커뮤니케이션 신호’로 쓰인다는 관점은 부족했습니다.

- **Core Contribution**: EARS(Explanatory Abstention for Reliable Sub-Agent Modeling)는 sub-agent의 abstention을 단순 거절이 아니라 coordinator를 위한 inter-agent communication protocol로 재정의합니다. sub-agent가 애매/부족/미지원/오라팅 같은 failure state를 ‘카테고리 + 근거(rationale)’ 형태로 노출하면, coordinator는 이를 바탕으로 재질문·재라우팅·fallback을 수행할 수 있게 됩니다. 이를 위해 도메인 적응용 abstention 데이터 파이프라인과 fine-tuning 전략을 함께 제시합니다.

- **Technical Challenges**: 핵심 난제는 task-specific하게 어떤 상황을 abstention으로 분류해야 하는지 라벨 신뢰도를 확보하는 것입니다. EARS는 calibrated LLM-as-a-Judge를 seed set으로 단계적으로 보정하고, 여러 judge의 unanimous 합의(unanimity agreement)만 학습 데이터로 채택해 precision 중심의 신뢰도 높은 라벨을 구성합니다. 또한 Ambiguous Query/Insufficient Input/Missing Capability/Misrouting의 failure taxonomy에 맞춘 계층적(hierarchical) 라벨링과 근거 생성으로 coordinator가 이해 가능한 구조화된 피드백을 학습시킵니다.

- **Empirical Impact**: e-commerce 프로덕션 환경의 business intelligence(BI) sub-agent에서 EARS는 전체 response pass rate를 68.5%에서 78.9%로(상대 15.2%) 끌어올렸습니다. 특히 segmentation 질의에서 abstention 학습의 효과가 더 크게 나타났고, syntax validty는 유지되어 잘못된 포맷 생성으로 품질이 떨어지지 않았습니다. shadow deployment의 human 검증에서도 세션 성공률이 67.1%로 올라, baseline 대비 신뢰성 개선이 실사용 관점에서 확인됐습니다.



### Fair Cognitive Impairment Detection Through Unlearning (https://arxiv.org/abs/2606.18571)
Comments:
          Interspeech 2026

- **Prior Approaches**: 기존 MCI 탐지는 음성의 의미·언어 특성, 데이터 증강, prompt learning 등을 활용해 왔지만, 실제 임상 음성 데이터는 작고 불균형하며 성별·언어 같은 인구통계 신호가 라벨과 우연히 함께 등장해 모델이 ‘지름길(shortcut)’을 학습하기 쉽다는 한계가 지적돼 왔다. 특히 speech 기반 인지장애 벤치마크는 subgroup(성별, 언어)별 성능 격차가 크게 벌어질 수 있으며, 이는 공정성·신뢰성 문제로 이어진다. 기존 bias mitigation은 재가중, 견고한 표현 학습, 편향 구성요소 영향 축소 등 다양한 접근이 있으나, multilingual·multimodal 설정에서 공통 임베딩에 담긴 인구통계 정보를 직접 약화시키는 설계는 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 speech, text, image를 함께 쓰되 cross-attention으로 모달리티 간 상호작용을 정렬해 진단 신호를 더 잘 포착하는 fair MCI detection 프레임워크 FMD를 제안한다. 동시에 shared embedding에서 인구통계 속성 정보를 없애도록 unlearning을 도입하며, gradient reversal을 통해 보조 demographic classifier가 맞히지 못하도록 학습을 유도한다. 즉, ‘MCI 예측에는 유용하지만 인구통계 정체성에는 기댈 수 없는’ 표현을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 멀티모달 융합이 단순 late concatenation에 그치면 모달 간 미세한 정렬과 보완 신호를 충분히 활용하지 못하고, (2) 인구통계 편향 제거를 위해 adversarial/gradient reversal을 쓰면 초기 학습에서 불안정해질 수 있다는 점이다. FMD는 text를 alignment anchor로 cross-attention 융합을 구성해 토큰 단위로 음성과(가능하면) 시각 정보를 조건화하며, unlearning 강도 λ를 0에서 1까지가 아니라 훈련 초기에 0에 가깝게 시작해 점진적으로 키우는 스케줄링으로 안정성을 확보한다. 이를 통해 과도한 인구통계 제거로 인한 성능 저하를 줄이면서 공정성을 함께 개선하려는 전략을 사용한다.

- **Empirical Impact**: TAUKADIAL과 PREPARE의 실험에서 FMD는 전반적 F1을 state-of-the-art 수준으로 끌어올리면서(예: TAUKADIAL에서 F1 92.6, PREPARE에서 F1 60.1) 성능 격차도 크게 낮춘다. worst-group 성능에서도 개선이 뚜렷해 subgroup별 가장 불리한 집단의 F1이 크게 상승하며, 예컨대 성별 gap은 CogniVoice 대비 5.5에서 0.6으로 줄어드는 결과가 보고된다. 또한 zero-shot transfer(데이터셋 간 학습-평가)에서 분포 이동에도 baseline보다 더 높은 전반 성능과 더 낮은 격차를 보여, 인구통계 unlearning이 더 견고한 표현 학습에 실제로 도움이 된다는 점을 경험적으로 뒷받침한다.



### CEO-Bench: Can Agents Play the Long Game? (https://arxiv.org/abs/2606.18543)
- **Prior Approaches**: 기존 에이전트 벤치마크(SWE-bench, WebArena, τ-bench 등)는 목표가 명확하고 에피소드가 짧아 피드백이 빠르게 관측되는 “단기 실행” 성격이 강합니다. GDPval이나 agentic-memory 벤치마크도 작업의 지속성은 늘리지만, 여전히 일회성 산출물 중심이거나 저장·검색에 초점이 맞춰져 있습니다. Vending-Bench/Accounting-Bench처럼 장기 시뮬레이션도 있지만 의사결정 범위가 좁고 환경이 비교적 안정적이라, 잡음·지연 피드백·상호 의존·전략 일관성까지 통합해 검증하긴 어렵습니다.

- **Core Contribution**: CEO-Bench는 에이전트가 500일 동안 스타트업을 운영하며 장기 전략 제어, 잡음 환경에서 정보 획득, 변화하는 세계 적응, 다부문 오케스트레이션을 한 번에 평가하도록 설계됐습니다. 에이전트는 34개 도구와 19-table 비즈니스 데이터베이스를 가진 동일한 파이썬 기반 인터페이스에서 운영 결정을 내리고, 종료 조건(현금이 0 아래로 하락)을 통해 결과를 직접 받습니다. 단순 도구 호출 능력을 넘어, 데이터 분석과 코딩을 결합한 “지속 가능한 경영” 지능을 측정하는 데 초점을 둡니다. 

- **Technical Challenges**: 핵심 어려움은 (1) 간접·지연·잡음 신호로부터 숨은 고객 선호와 만족도를 추정해야 하고, (2) 매출·R&D·평판 등 영향이 서로 다른 시간 스케일로 누적되며, (3) 고객 획득·품질·지원·엔터프라이즈 협상 등 의사결정이 서로 얽혀 인과를 단일하게 분리하기 어렵다는 점입니다. 논문은 이를 위해 고객을 집단/개별 단위로 세분화하고, 구독·이탈·소셜 미디어 반응·경쟁사 압력·거시 사이클 등 여러 동역학을 기계적으로(LLM 판정 없이) 생성하도록 시뮬레이터 세계 규칙을 구성했습니다. 또한 에이전트가 직접 SQL 조회와 맞춤 코드 실행을 할 수 있게 API 기반 도구·데이터 스키마·공개 피드/협상 로그를 제공해, “분석→전략→실행→재계획” 루프를 강제합니다.

- **Empirical Impact**: 실험 결과 대부분의 최신 모델은 도구 호출과 분석 쿼리 자체는 수행하지만, 500일 동안 일관된 전략을 유지하지 못하고 파산하는 경우가 많습니다. 최상위 성적은 Claude Opus 4.8과 GPT-5.5로, 둘만이 시작 현금 $1M을 넘기는 현금 상위를 기록했지만 이익이 “일관되게” 나진 못했다고 보고됩니다. 더 나아가 모델별 행동 궤적을 보면 GPT-5.5/Claude Opus 4.8은 더 넓게 탐색하고 상황에 따라 전략을 자주 바꾸는 반면, Claude Opus 4.7은 수동적 현금 보존 성향에 머물며 초기 고객 확보가 끊기는 등 미세한 차이가 드러납니다. CEO-Bench는 장기·적응형 지능의 측정 공백을 보여주며, 현재 모델들의 통합 전략 역량에 명확한 여지가 남아 있음을 실증적으로 제시합니다.



### Evaluating Prompting-Based Defenses Against Domain-Camouflaged Injection Attacks (https://arxiv.org/abs/2606.18530)
Comments:
          9 pages, 4 figures, 4 tables; under review at the AdvML-Frontiers x CoTMA workshop, COLM 2026

- **Prior Approaches**: 기존 prompt injection 방어 연구들은 주로 “IGNORE ALL PREVIOUS INSTRUCTIONS”처럼 명시적 오버라이드 지시가 있는 공격을 상정해 왔습니다. 그래서 provenance 마커 추가(spotlighting), 작업 재확인(prompt sandwiching), 또는 중립 문장으로 재작성(paraphrasing) 같은 기법도, camouflage처럼 문법·표면만으론 구분이 어려운 공격에서는 성능이 검증되지 않았습니다. 특히 standard detector는 domain-camouflaged payload를 거의 놓치며(Detection Gap이 큼), 현장 실무자는 어떤 방어 아키텍처가 실제 성공률을 낮추는지 답을 찾지 못했습니다.

- **Core Contribution**: 이 논문은 domain-camouflaged injection(도메인 어휘로 위장한 악성 지시)을 상대로 prompting-based 방어 5가지(spotlighting, paraphrasing, sandwiching 및 조합, Llama Guard 4)를 3개 모델군과 3개 배포 도메인에서 체계적으로 비교합니다. 총 3,510 trials로 “방어 순위”가 camouflage 조건에서 어떻게 바뀌는지 처음으로 정량화했으며, 특히 paraphrasing이 전 모델에서 가장 일관되게 공격 성공률을 낮춘다는 결론을 제시합니다. 또한 Llama Guard 4의 over-refusal이 높은 반면 camouflage ASR은 paraphrasing보다 더 높게 남는다는 점을 함께 보여줍니다.

- **Technical Challenges**: 핵심 난관은 camouflage payload가 표면상 합법 문서와 구문적으로 동형에 가깝기 때문에, 마커 신호나 단순 패턴 기반 탐지로는 구분이 거의 되지 않는다는 데 있습니다. 연구진은 전문 문서 기반 합성 벤치마크에서 static(명시 오버라이드)와 camouflage(도메인 위장) 두 공격 클래스를 분리해 ASR(공격 지시를 따르는 비율)로 방어 효과를 비교했고, 각 방어별로 모델 의존성까지 함께 분석했습니다. 그 결과 paraphrasing은 지시형 표현의 표면 특징을 제거해 성공률을 낮추지만, spotlighting이나 sandwiching은 모델/도메인에 따라 효과가 갈리는 패턴을 확인했습니다.

- **Empirical Impact**: 실험에서 paraphrasing은 camouflage 공격 성공률을 모델별로 55–84%까지 크게 줄였고, 수치상 Haiku(14.4%→4.4%), Llama(22.2%→10.0%), Gemini(21.1%→3.3%)로 전반적으로 낮아졌습니다. 특히 paraphrasing은 모든 테스트 모델에서 Llama Guard 4보다 낮은 camouflage ASR을 기록했으며, over-refusal 비용도 0%로 보고되어 운영 관점의 이점이 큽니다. 다만 금융 도메인은 잔여 위험이 가장 높아(기준선 26–33% 수준) prompting-based 방어만으로 위협을 완전 차단하긴 어려워, 실무자에게는 도메인별 추가 통제 필요성과 함께 방어 우선순위(para 우선, 모델별 검증)를 제안합니다.



### Compact Geometric Representations of Hierarchies (https://arxiv.org/abs/2606.18520)
Comments:
          Published at the 39th Annual Conference on Learning Theory (COLT) 2026. 22 Pages

- **Prior Approaches**: 기존의 dense retrieval은 쿼리와 문서를 bi-encoder로 임베딩 공간에 매핑한 뒤, 내적/거리로 근접 이웃을 찾아 relevance를 판단합니다. 계층형 retrieval에서는 최근 You et al.가 DAG의 ancestor-descendant 관계를 reachability embedding으로 다뤘지만, 디escendant 수가 작을 때만 차원이 작게 보장되고 깊은 계층에서는 차원이 급격히 커지는 문제가 남았습니다. 즉, 그래프 구조를 더 정교하게 반영하는 “compact” 임베딩의 존재 조건이 불명확했습니다.

- **Core Contribution**: 이 논문은 hierarchical retrieval의 reachability를 더 작은 차원으로 표현하기 위해, 임베딩 차원이 트리width(treewidth)나 cross-edges 개수 같은 구조 파라미터에 의해 결정되도록 하는 이론을 제시합니다. 특히 directed tree(루트가 있는 방향 트리)에서는 그래프 크기와 깊이에 무관하게 차원 3의 reachability embedding이 항상 존재함을 증명합니다. 또한 treewidth가 t인 일반 그래프에 대해 차원 O(t log n) 상계를 주고, 그에 상응하는 하한(일반 DAG에서 Ω(n), treewidth t에서 Ω(t/log(n/t)))도 함께 제시해 경계가 거의 타이트함을 보입니다.

- **Technical Challenges**: 핵심 난제는 “reachability는 존재하지만, 단순한 분해/결합(예: spanning forest 기반 혹은 컴포넌트별 임베딩 결합)만으로는 내적이 거짓 양성(false positive)을 만들 수 있다”는 점입니다. 이 논문은 (1) directed tree/forest에서는 DFS의 discovery/finish time이 자손 관계를 구간 포함으로 요약한다는 성질로 차원 3 임베딩을 구성하고, (2) DAG에서는 spanning forest에 없는 경로를 만드는 cross-edge마다 좌표 1개를 추가해 이를 보정하는 augmentation 원리를 도입합니다. 나아가 treewidth 기반 증명에서는 separator로 균형 분할한 뒤, 재귀 임베딩을 결합할 때 생기는 거짓 양성을 제거하기 위한 좌표 설계를 추가로 수행합니다.

- **Empirical Impact**: 실험적으로 실제 데이터셋에서 임베딩을 구성할 수 있음을 보이고, 특히 high recall(높은 재현율) 환경에서는 기존에 이론적 보장을 함께 갖는 prior reachability embeddings보다 훨씬 작은 차원으로도 성능을 유지/개선할 수 있음을 보여줍니다. 이는 계층형 검색에서 “차원 축소”가 단순 휴리스틱이 아니라 그래프 구조 파라미터를 활용한 이론적 설계로 달성 가능하다는 메시지를 강화합니다. 결과적으로, dense retrieval의 임베딩 설계 관점에서 treewidth/cross-edge 같은 구조 복잡도를 정량 목표로 삼을 수 있는 길을 열었다는 점에서 의미가 있습니다.



### SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR (https://arxiv.org/abs/2606.18487)
Comments:
          14 pages, 6 figures. Accepted at the Deep Learning for Code (DL4C) Workshop at ICML 2026

- **Prior Approaches**: 기존 코딩 생성 후학습 파이프라인은 SFT 이후 RLVR(검증 가능한 보상)로 넘어가며, 보통 SFT 체크포인트 중 GRPO에 가장 유리한 것으로 보이는 pass@1이 높은 것을 선택합니다. 그러나 대규모에서는 pass@1이 후속 RL 성능을 잘 예측하지 못하고, SFT의 과적합/분포 압축이 RL 쪽 학습 신호를 망가뜨릴 수 있다는 문제 제기가 이어졌습니다.

- **Core Contribution**: 이 논문은 GRPO에서 쓰는 그룹 상대 advantage 신호가 SFT 과학습으로 인해 사실상 소멸(gradient vanishing)할 수 있음을 정식화합니다. 특히 이진 보상 하에서 그룹 내 advantage 분산이 p(1-p)(g-1)/g로 주어져, 일정 임계 p*(g) 아래로 내려가면 대부분 그룹이 보상을 동일하게 받아 그룹 상대 신호가 구조적으로 붕괴한다고 보입니다.

- **Technical Challenges**: 핵심은 “왜 pass@1이 높은 체크포인트가 오히려 GRPO 초기 학습을 망가뜨리는가”를 체크포인트 단위로 진단하고, 조기 실패를 실무적으로 막는 스크리닝 방법을 찾는 것입니다. 저자들은 SFT depth ladder로 Qwen2.5-Coder-3B(각 깊이 3 seed)에서는 SFT가 깊어질수록 pre RL pass@1은 오르는데 peak GRPO pass@10은 0.806→0.481로 단조 하락하며, 이 차이를 출력 엔트로피 붕괴와 보상 분산 붕괴(임계 p*(8) 하회)로 설명합니다.

- **Empirical Impact**: Qwen에서는 “가장 높은 SFT pass@1 선택”이 매번 최악의 GRPO 초기화로 이어지는 rank inversion이 재현됩니다(엔트로피 붕괴가 빠르게 진행되어 학습 신호를 일찍 소진). 반대로 DeepSeek-Coder-6.7B에서는 임계 p*(8)보다 충분히 높은 안전 영역에 머물러 rank compression(순서가 뒤집히지 않음)이 나타나 이론의 경계 예측을 대조 검증합니다. 또한 pre RL 엔트로피 triage에 early GRPO 엔트로피 모니터를 결합한 2단계 진단이 실패 위험 체크포인트를 조기에 플래그하고, KL/label smoothing 같은 간단한 사후 개입은 붕괴 체크포인트를 구제하지 못해 실패 원인이 GRPO 하이퍼파라미터보다 SFT 단계의 엔트로피 고갈에 있음을 시사합니다.



### LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents (https://arxiv.org/abs/2606.18388)
- **Prior Approaches**: RL post-training에서는 고정된 가이드북형 스케줄(응답 길이/rollouts/난이도 단계 증가, 또는 일부 진동)을 미리 정해 실행하는 경향이 강했습니다. 하지만 이런 방식은 “언제(트리거), 얼마나(폭), 무엇을(파라미터 조합)” 바꿔야 하는지 훈련 동역학이 바뀔 때 대응하기 어렵고, KL 스파이크·붕괴·정체 같은 이상 징후에 체계적으로 반응하지 못합니다.

- **Core Contribution**: 이 논문은 최적 multi-stage RL post-training에서 나타나는 구조적 비대칭을 제시합니다: capacity 파라미터(응답 길이, rollouts 등)는 단계가 진행될수록 단조 누적되는 반면, regularization 파라미터(학습률, KL 계수, temperature 등)는 훈련 역학 변화에 따라 주로 진동합니다. 또한 LLM 에이전트가 체크포인트별 동역학을 해석해 다중 파라미터를 “동시 전환”하도록 설계한 LLMZero를 통해 이 원칙을 찾고 검증합니다.

- **Technical Challenges**: 핵심 난제는 고정 스케줄로는 표현하기 어려운 비정상(non-stationary) 탐색-활용 트레이드오프를, 여러 하이퍼파라미터의 인과적 연동을 고려해 실시간 전환해야 한다는 점입니다. LLMZero는 MCTS(UCT)로 학습 궤적을 트리 탐색하고, 시각화·텍스트 지표를 결합한 proposer 에이전트가 병리 진단 후 3개 이상 파라미터를 조율한 전환을 제안하며, agentic early stopper로 유망하지 않은 가지를 중단해 탐색 비용을 줄입니다.

- **Empirical Impact**: 4개의 다양한 GRPO 태스크에서 LLMZero가 베이스 모델 대비 상대 9%~140%, grid search 대비 상대 6%~15% 개선을 보였고, 무작위 탐색과 skill-based 에이전트를 일관되게 능가했습니다. 특히 최적 전략을 12 iteration 내에 찾는 경우가 많아 반복 효율도 높았으며, SSMR-bench에서는 모델 크기(0.6B~8B) 전반에서 성능이 유지·확장되는 동시에 OOM 같은 인프라 실패까지 회피해 실무적 의미가 큽니다.



### From Sparse Features to Trustworthy Proxies: Certifying SAE-Based Interpretability (https://arxiv.org/abs/2606.18383)
- **Prior Approaches**: SAE는 언어 모델 내부 표현을 sparse하고 해석 가능한 feature로 분해해 분석에 널리 활용돼 왔지만, “그 feature가 원래 고정된 LM의 예측을 얼마나 충실히 반영하는가”를 보증하는 기준은 부족했다. 기존 접근은 주로 정성적/탐색적 해석에 초점이 맞춰져, SAE가 만드는 설명을 신뢰할 수 있는 조건을 수학적으로 분해해 제시하진 못했다. 또한 일반화 성능을 보이는 이론적 bound들은 존재하지만, SAE가 만들어내는 ‘설명용 프록시(proxy)’ 자체를 통해 LM을 인증하는 관점은 상대적으로 새롭다.

- **Core Contribution**: 이 논문은 SAE 재구성으로 특정 hidden activation을 대체해 만든 sparse proxy로, 고정된 LM의 expected risk를 post-hoc로 상계하는 인증(certification) 프레임워크를 제안한다. 핵심은 “설명”을 운영적으로 정의해, sparse proxy가 (1) 기준선 대비 충분히 정보가 있고 (2) 행동적으로(출력 분포 관점에서) 원래 LM과 가깝다는 두 조건이 동시에 만족될 때 faithfulness를 신뢰할 수 있게 한다. 인증 상계는 proxy risk, SAE reconstruction gap, concept-pool mismatch, sparse complexity의 4개 측정량으로 분해된다.

- **Technical Challenges**: 가장 큰 기술적 난관은 SAE가 만든 sparse 구조가 실제로 LM 예측 위험을 얼마나 보존하는지, 그리고 그 일반화가 표본 크기에서 비(非)자명(non-vacuous)해지는지를 함께 증명하는 것이었다. 이를 위해 저자는 SAE로부터 유도된 프록시의 유효 가설 공간을 ‘전체 파라미터’가 아니라 “active feature-pool 크기”로 제한해 Occam형 bound에 연결하고, loss의 하한이 흔들리는 문제는 prediction smoothing으로 완화해 상계를 성립시키는 구성을 만든다. 최종적으로 calibration과 evaluation을 분리하고, pool-mismatch과 reconstruction gap을 실측해 비자명성 여부를 판단할 수 있게 한다.

- **Empirical Impact**: 실험에서는 GPT-2 Small, Gemma-2B, Llama-3-8B에서 practical sample size에서도 인증 bound이 non-vacuous가 되는 구간을 보였다. 특히 Llama-3-8B의 층별 패치 지점 분석에서 later layers가 훨씬 더 쉽게 인증되며, 그 이유가 complexity 항이 줄어서가 아니라 proxy의 국소 fidelity는 강화되고 downstream error amplification은 약해지기 때문임을 분해 관찰했다. 마지막으로 feature-shuffling ablation으로, sparse 통계량만으로는 설명 정렬이 보장되지 않으며 의미적 정렬과 단순 희소성(sparsity)을 구분하는 진단으로도 활용될 수 있음을 보여준다.



### Breaking the Solver Bottleneck: Training Task Generators at the Learnable Frontier (https://arxiv.org/abs/2606.18284)
Comments:
          30 pages, 9 figures, 12 tables

- **Prior Approaches**: 기존에는 RLVR/RLHF 계열에서 과제 생성기를 학습하되, 보상(유효성·난이도·학습가능성)을 계산하려고 타깃 solver의 롤아웃을 매 후보마다 반복했다. 이 방식은 과제가 학습 가능한 frontier에 걸려 있는지 판별은 잘하지만, SWE처럼 검증 비용이 큰 영역에서는 solver-in-the-loop이 사실상 병목이 된다. 한편 단순 합성 과제는 너무 쉬운 문제만 만들거나, 불가능/비문제처럼 ill-posed한 과제를 생성해 목적에 맞는 분포를 채우기 어렵다.

- **Core Contribution**: PROPEL은 solver-amortized 프레임워크로, 생성기 학습 중에는 solver 롤아웃을 돌리지 않고 activation probe를 보상 대용으로 쓴다. 구체적으로 사전 오프라인에서 생성된 (task, solver-outcome) 라벨로 프로브를 한 번 학습하고, RL 중에는 고정된 reference generator의 내부 활성에서 목표 solve rate(learnable frontier) 근처인지 확률/로짓을 예측해 보상을 준다. 그 결과 매 후보 평가 비용을 “여러 번의 solver 시도”에서 “단 한 번의 forward pass”로 축소한다.

- **Technical Challenges**: 핵심 난제는 (1) 목표 solve rate 같은 ‘비싼 검증’ 신호를 내부 활성만으로 안정적으로 대체할 수 있는지, 그리고 (2) 단일 고정 프로브에 최적화하면 특정 의미 토픽으로 쏠리는 mode collapse가 발생할 수 있다는 점이다. PROPEL은 validity 게이트(유효한 과제인지)와 프로브 점수(프론티어 유사도)를 결합하고, fixed-probe로 인한 붕괴를 줄이기 위해 worst-case optimization(WCO) 및 adversarial co-evolution을 도입한다. 또한 도메인별로 SWE의 multi-turn 궤적에 맞춰 활성 추출/집계를 확장해 프로브가 궤적 수준의 유틸리티를 반영하도록 설계했다.

- **Empirical Impact**: 수학·코드 유도·SWE 전반에서 PROPEL은 목표 learnable frontier 근처의 과제 비율을 크게 끌어올리며, solver-in-the-loop 대비 더 낮은 비용으로 더 큰 유틸리티 개선을 보였다. 예컨대 coding에서 Qwen2.5-3B-Instruct solver 기준 learnable-frontier 생성 비중이 10.1%→20.0%, Qwen2.5-7B-Instruct에서는 5.3%→12.6%로 상승했다. SWE에서는 Qwen3.5-27B 타깃에서 목표 solve rate 비중이 9.8%→19.6%로 증가했으며, 프로브 학습에 안 쓰인 저장소·학습 외 분포에서도 유사한 개선(예: 2.0× 수준)이 관찰되어 내부 활성 기반 보상 신호의 일반화 가능성을 시사한다.



### Simulating Hate Speech Cascades with Multi-LLM Agents: Empirical Grounding, Modeling Fidelity, and Intervention Strategies (https://arxiv.org/abs/2606.18264)
- **Prior Approaches**: 기존 정보확산 모델(Independent Cascade, Linear Threshold 등)은 프로필·커뮤니티·콘텐츠 요인을 단일 전파확률로 뭉개어 증오 발화 확산의 메커니즘을 충분히 표현하지 못한다. LLM을 에이전트로 쓰는 사회 시뮬레이션 연구는 프로필/문맥 조건화를 제안하지만, 실제 관측된 증오 캐스케이드를 고전 기준선보다 더 충실하게 재현하는지와 그 원인이 무엇인지가 불명확했다. 또한 실증적 캐스케이드 분석과 생성형 시뮬레이터 평가는 서로 분리되어 있어, 같은 네트워크에서의 fidelity(재현 정확도) 비교가 부족했다.

- **Core Contribution**: 이 논문은 Bluesky에서 수집한 3개의(암묵/코드화) 증오 캐스케이드와 규모가 맞는 선의(benign) 대조군을 대상으로, 구조·시간·커뮤니티 수준의 정규성을 실증적으로 정리한다. 이어서 동일 네트워크에서 고전 확산/휴리스틱/단순 LLM 대비, 사용자 프로필·주변 커뮤니티·게시물 텍스트를 각각 반영하는 multi-LLM-agent 시뮬레이터가 관측 정규성을 얼마나 재현하는지 fidelity를 비교한다. 마지막으로 agent 수준의 기여 요인을 구조화된 ablation으로 분해하고, 그 메커니즘에 근거한 4가지 중재(경과 대기, amplifier targeting, warning label, early-hop truncation)를 counterfactual로 시험한다.

- **Technical Challenges**: 핵심 기술 과제는 “에이전트가 더 유연해질수록 실제 증오 캐스케이드를 더 잘 따라가는가”를, 고정된 인구·네트워크·프롬프트 조건에서 정량 비교하는 것이다. 논문은 follower 네트워크에서 diffusion tree를 재구성하고, 각 사용자에 대해 community identity·stance·account type·toxicity engagement를 bio와 최근 게시물 텍스트로부터 GPT-4o-mini로 추정한 뒤, homophily delta 등 다층 지표로 재현성을 측정한다. 또한 role-play 형태의 프롬프트는 safety refusal 문제가 커져, 대신 “reshare 확률 예측(probability prediction)” 프레이밍으로 전환해 시뮬레이션이 증오/선의를 콘텐츠 차이로 구별하도록 설계한다.

- **Empirical Impact**: 실증 결과에서 증오 캐스케이드는 reposters의 hostile stance가 97.4~99.7%로 포화에 가깝고, follower 그래프보다 확산 트리에서 toxicity-engagement homophily가 더 강하게 나타난다. 또한 증오는 대부분 루트에서 바로 퍼지는 star-like 토폴로지(깊이 4~6)를 보이는 반면, 선의 대조군은 다단계 체인을 타는 tree-like 구조(깊이 4~6 대비 더 큰 깊이)로 구분된다. 시뮬레이션에서는 multi-LLM-agent가 이 “stance monoculture”와 “toxicity-delta 방향성”을 재현했고, fidelity를 가장 크게 좌우하는 요인으로는 agent heterogeneity가 지목된다; 더해 dense 네트워크에서 amplifier targeting은 7.5~12.9% 증오 확산 감소와 5.7% benign collateral의 트레이드오프를 보여 실제 중재 전략 실험의 새 기준점을 제시한다.



New uploads on arXiv(cs.IR)

### Querit-Reranker: Training Compact Multilingual Rerankers via Efficient Label-Free Distribution Adaptation (https://arxiv.org/abs/2606.19037)
- **Prior Approaches**: 다국어 reranker는 언어·도메인·목표 ranking task 전반에서 일반화해야 하지만, second-stage reranking까지 감당할 만큼 효율적이어야 한다. 기존에는 새로운 target distribution에 맞추기 위해 task-specific relevance annotation이 많이 필요해 비용과 확장성이 문제가 됐다. 또한 대개 여러 체크포인트나 앙상블을 런타임에 얹는 방식이어서 배포 시 지연/복잡도가 커질 수 있다.

- **Core Contribution**: 이 논문은 label-efficient 방식으로 목표 분포에 적응하는 data-centric 학습 파이프라인을 갖춘 Querit-Reranker 계열을 제안한다. 특히 Querit-Reranker-A0.4B(0.4B activated MoE 백본)와 Querit-Reranker-4B(Qwen3-Embedding-4B 초기화)를 구성해, 연속적인 soft label로 적은 주석만으로도 적응하도록 설계했다. 또한 서로 다른 task-adapted 강점을 단일 deployable 모델로 합치기 위해 spherical linear interpolation을 적용한다.

- **Technical Challenges**: 핵심 난제는 “target distribution 적응”을 위해 필요한 relevance annotation을 줄이면서도, cross-encoder가 안정적으로 general relevance modeling과 도메인 편이를 동시에 학습해야 한다는 점이다. 이를 위해 파이프라인은 먼저 대규모 ranking 지향 데이터로 일반 관련성 모델링을 학습한 뒤, synthetic-query mining으로 목표 분포에 맞는 쿼리를 생성하고 teacher 점수를 soft label(연속 값)로 사용해 label-efficient adaptation을 수행한다. 마지막으로 여러 체크포인트의 상보성을 런타임 앙상블 없이 통합하기 위해 spherical linear interpolation으로 파라미터를 병합한다.

- **Empirical Impact**: 실험에서 Querit-Reranker-A0.4B는 BEIR에서 평균 nDCG@10을 54.11→59.28, MIRACL에서 59.87→67.70으로 끌어올렸다. MTEB Multilingual v2 reranking에서도 더 큰 embedding 기반 기준선을 크게 앞섰고, Querit-Reranker-4B는 공개 모델 중 state-of-the-art 수준을 추가로 달성했다. 두 모델 모두 Hugging Face에 공개되어, 비용 부담이 큰 relevance annotation 없이도 다국어 reranking 성능을 높이려는 실무 흐름에 직접적인 선택지를 제공한다.



### SAERec: Constructing Fine-grained Interpretable Intents Priors via Sparse Autoencoders for Recommendation (https://arxiv.org/abs/2606.18897)
- **Prior Approaches**: 기존 intent-based recommender는 사용자 행동 시퀀스를 클러스터링하거나 고정된 intent prototype에 매핑해 의도를 중간 계층으로 둡니다. 하지만 intent 개수를 미리 정해야 하고, 시퀀스 품질에 민감해 coarse한 intent 집합이 되기 쉽습니다. 또한 intent가 라벨 없는 잠재 벡터라서 의미적 grounding이 약해 설명 가능성이 제한됩니다.

- **Core Contribution**: SAERec은 리뷰/아이템 설명 텍스트 코퍼스에서 fine-grained하고 해석 가능한 intent 공간을 자동으로 구성해 추천을 유도합니다. 핵심은 LLM 임베딩을 Sparse Autoencoder(SAE)로 희소·분리된 feature로 분해하고, 각 feature의 의미를 LLM이 라벨링해 사람이 읽을 수 있는 intent로 남기는 것입니다. 이후 개인( personal ) intent과 공개( public ) intent를 검색한 뒤, 멀티 브랜치 attention으로 시퀀스 모델링에 주입하고 adaptive fusion으로 최종 사용자 표현을 만듭니다.

- **Technical Challenges**: 텍스트 임베딩은 다중 의미(polysemantic)와 잡음이 얽혀 있어 supervision 없이 서로 다른 intent 신호를 분리하기 어렵습니다. SAE로 과완비(overcomplete) 희소 표현을 만들고, 각 희소 축을 단어 집합과 mutual information 기반 정렬로 해석 가능하게 만든 뒤, LLM 판단으로 recommendation에 유의미한 intent만 선별해 의미 없는 차원을 제거합니다. 또 intent와 시퀀스 모델의 latent space 불일치 문제를 해결하기 위해, 시퀀스 인코더 출력과 intent 벡터를 같은 공간에 정렬하고 personal/public dual-level retrieval로 관련 intent만 top-K로 뽑아 attention에 주입합니다.

- **Empirical Impact**: 공개 데이터셋의 광범위한 실험에서 SAERec은 최신 baseline을 일관되게 능가하며 정확도 향상을 보였습니다. 동시에 자동 생성된 intent는 사람 친화적으로 설명 가능한 형태를 제공해, 추천 근거를 “품질/가성비” 같은 공개 동기와 “민감성 피부” 같은 개인 동기로 구분해 제시할 수 있습니다. intent 수를 미리 정하지 않아도 되는 파이프라인은 실제 운영에서의 튜닝 부담과 coarse intent 문제를 함께 완화한다는 점에서 의미가 큽니다.



### LensKit-Auto (https://arxiv.org/abs/2606.18814)
- **Prior Approaches**: 기존 추천 시스템 연구는 다양한 알고리즘이 존재하지만, 데이터셋마다 최적 알고리즘과 하이퍼파라미터가 크게 달라 “모두에게 통하는” 단일 해법은 없다. 그 결과 실제 적용에서는 알고리즘 선택과 튜닝이 반복되는 수작업 과제가 되며, 사용자는 전문성이 없으면 탐색 비용이 커진다. AutoRecSys 시도들은 있었지만, LensKit-Auto 계열처럼 사용성을 중심으로 한 최신 프레임워크 연동과 확장 최적화가 부족한 경우가 많았다.

- **Core Contribution**: 이 논문은 LensKit-Auto를 최신 LensKit 버전에 맞춰 업데이트하고, 사용자 데이터셋을 넣으면 성능이 가장 좋은 알고리즘-하이퍼파라미터 조합을 알려주는 블랙박스 성격을 강화한다. 또한 Tree Parzen Estimator 같은 추가 최적화 방법, 찾은 알고리즘 재사용, 문서 개선, 최적화 과정 시각화 기능을 더해 비전문가도 손쉽게 적용할 수 있도록 한다. 마지막으로 메타데이터셋을 생성하는 메타러닝 프레임워크를 연동해, 향후 meta-learning 통합 가능성도 확장한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) LensKit 내부가 바뀌는 환경에서도 동일한 수준의 검색·평가 파이프라인을 유지하고, (2) 다양한 알고리즘/하이퍼파라미터 공간에서 효율적으로 최적해를 찾는 것이다. 논문은 업데이트된 LensKit에 맞춘 동작 보장과 함께 Tree Parzen Estimator를 추가 최적화 엔진으로 도입해 탐색 품질을 높이고, 최적화 과정을 시각화해 블랙박스 사용 시에도 신뢰성을 확보하도록 설계한다. 더불어 메타러닝을 위한 meta-dataset 생성 절차를 마련해, 미래의 자동화 지능을 위한 기반을 만든다.

- **Empirical Impact**: 논문은 업데이트와 기능 확장을 통해 LensKit-Auto의 사용성을 높였고, 실제로 데이터셋에 대해 적합한 알고리즘과 하이퍼파라미터를 찾는 자동화 목표를 달성하는 방향으로 개선되었다고 제시한다. 특히 최적화 과정 시각화와 알고리즘 재사용은 운영 효율을 높여 실무자가 튜닝 반복을 줄이는 데 의미가 있다. 또한 메타데이터셋 생성 기반은 향후 meta-learning이 AutoRecSys에 결합될 때의 성능 향상과 탐색 비용 절감으로 이어질 수 있다는 점에서 분야 전반에 잠재적 영향을 준다.



### Rescaling MLM-Head for Neural Sparse Retrieva (https://arxiv.org/abs/2606.18811)
- **Prior Approaches**: SPLADE 같은 learned sparse retrieval(LSR)은 BERT 스타일 masked language model(MLM) 백본의 MLM head 출력을 그대로 희소 어휘 표현으로 써서 검색 점수를 계산한다. 기존에는 백본을 최신 인코더로 바꿔도 성능이 오를 것이라는 ‘드롭인’ 기대가 있었지만, 실제로는 ModernBERT·Ettin 같은 큰 MLM-head L2 norm 백본이 기준선(BERT-SPLADE) 대비 크게 무너질 수 있다. 기존 연구들은 하드 네거티브 마이닝, 증류, 희소 정규화, 데이터/학습량 개선 등에 집중했지만, SPLADE 핵심 계산에서 MLM-head 스케일이 일으키는 실패 양상은 충분히 다루지 않았다.

- **Core Contribution**: 논문은 성능 붕괴의 원인을 ‘모델 용량’이 아니라 MLM head의 스케일 불일치로 지목한다. SPLADE는 query-문서의 희소 벡터를 unnormalized dot product로 비교하기 때문에 MLM-head 출력의 스케일이 점수와 대조 학습 동역학을 그대로 흔든다고 설명한다. 이를 해결하기 위해 학습 직전 MLM-head projection 행렬을 상수 k로 rescale하는 초기화 시점 보정(initialize-time correction)을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 학습 레시피를 바꾸지 않고도 pretrained 인코더의 MLM-head 스케일이 SPLADE의 점수/그라디언트 스케일과 맞지 않을 때의 불안정성을 제어하는 것이다. 논문은 FLOPS 정규화까지 포함된 SPLADE 목표에서 큰 MLM-head 스케일이 희소 활성과 점수 크기를 증폭시켜 대조 학습을 흔들고, حتی training collapse까지 유발할 수 있음을 실험적으로 연결한다. 해결책으로 k=16 같은 rescaling을 통해 학습 손실 스케일을 낮추고 학습을 매끄럽게 만들며, 아키텍처/목적함수/추가 파라미터 없이도 안정성과 성능을 복구한다.

- **Empirical Impact**: MS MARCO, TREC-DL, BEIR-13 등 in-domain·out-of-domain 벤치마크에서 큰-norm 백본(ModernBERT, Ettin)의 성능을 표준 레시피 하에 다시 경쟁력 있게 만든다. 예를 들어 BEIR-13 mean nDCG@10에서 ModernBERT는 상대 개선 폭이 215%, Ettin은 77% 수준이며, 일부 설정에서는 BERT-SPLADE 기준도 따라잡거나 능가한다. 또한 rescaling 효과가 단순한 범용 정규화가 아니라 ‘적정 스케일 범위’ 제어임을 보여주며, 과도한 down-scaling은 ALBERT·RoBERTa처럼 애초 norm이 작은 모델에서 성능 붕괴를 초래할 수 있음을 확인한다.



### SHIFT: Semantic Harmonization via Index-side Feature Transformation for Multilingual Information Retrieva (https://arxiv.org/abs/2606.18801)
- **Prior Approaches**: 기존 MLIR 연구는 query translation이나 문서/코퍼스 MT 같은 CLIR식 우회로를 쓰거나, multilingual dense retriever를 추가 학습으로 언어 정렬을 강화하는 방식에 의존해 왔다. 그러나 학습 기반 방법은 비용이 크고, 파이프라인 번역 방식은 복잡도·지연을 키우며, 단순 정규화는 의미 신호를 과도하게 억누를 수 있다. 특히 최근 dense retrieval 모델은 쿼리와 같은 언어 문서를 과도하게 상위에 배치하는 language bias 문제를 여전히 크게 보인다.

- **Core Contribution**: SHIFT는 학습 없이(inference-time 추가 비용 없이) indexing 단계에서 문서 임베딩을 보정해 언어 편향을 줄이는 방법을 제안한다. mMARCO 같은 병렬 번역 페어로부터 소스 언어 대비 각 타겟 언어의 상대 language vector를 추정한 뒤, 문서 임베딩에서 이를 선형으로 빼서 표현 공간을 소스 언어 쪽으로 캘리브레이션한다. 또한 이 논문은 쿼리 언어 외 문서의 재현율을 직접 보는 Target-Languages Recall@k (TLR@k)로 편향을 명시적으로 정량화한다.

- **Technical Challenges**: 핵심 난제는 “의미는 같지만 언어 표면형 때문에 embedding 공간에서 멀어지는” 불정렬을 모델 재학습 없이 교정하는 것이다. SHIFT는 병렬 문서의 임베딩 차이를 평균해 언어별 오프셋을 추정하고, α로 보정 강도를 조절하면서도 쿼리 쪽 변환은 하지 않아 지연을 만들지 않는다. 더 나아가 α가 과도하면 특정 언어에 대한 over-shifting이 발생해 전체 균형이 흔들릴 수 있음을 실험적으로 분석해 조절 가능함을 보인다.

- **Empirical Impact**: 네 개의 MLIR 벤치마크와 다양한 dense retriever(encoder·decoder 계열)에서 SHIFT가 일관되게 성능과 TLR을 동시에 개선함을 확인했다. 특히 언어 편향이 숨겨지는 지표(Recall/nDCG)와 달리 TLR@20에서 큰 상승이 관측되며, 예컨대 multilingual-e5-large의 경우 타겟 언어 문서 노출이 크게 늘었다. 전반적으로 top-k 결과의 언어 분포가 더 고르게 퍼지며(정성 분석), 3가지 비영어 소스 언어 실험에서도 유사한 개선이 반복되어 범용성까지 입증한다.



### RankGraph-2: Lifecycle Co-Design for Billion-Node Graph Learning in Recommendation (https://arxiv.org/abs/2606.18379)
- **Prior Approaches**: 기존 GNN 기반 추천은 그래프 구조·학습·서빙 중 한 단계만 따로 최적화하는 경우가 많아, billion-node 규모에서 병목이 전체 성능을 제한한다. 또한 많은 산업 시스템은 그래프가 주어진다는 가정 하에 학습은 분산 인프라로 해결하지만, 실시간 서빙 비용(online KNN/ANN) 문제는 충분히 다루지 않는다. 그래서 대규모 그래프 구축과 학습 목표, 그리고 서빙을 함께 설계하는 라이프사이클 co-design 관점이 부족했다.

- **Core Contribution**: RankGraph-2는 서빙·학습·그래프 구축을 동시에 “코설계”해, 한 단계의 요구가 다른 단계의 설계를 제약하도록 만드는 프레임워크를 제안한다. 특히 similarity-based retrieval(U2U2I, U2I2I)에서 online KNN을 없애기 위해 학습 단계에서 cluster index를 공학적으로 함께 co-learn 하도록 설계했다. 그 결과 더 단순한 아키텍처로도 높은 recall과 실제 지표 개선을 동시에 노린다.

- **Technical Challenges**: 대규모에서의 핵심 난제는 (1) 수백 조 잠재 엣지에 달하는 그래프를 1시간 내 재구성 가능한 형태로 줄이는 것, (2) 서빙 정확도를 유지하면서 online graph infrastructure와 KNN 탐색을 제거하는 것, (3) 학습 단계에서 인덱스 품질을 목표 함수에 반영하는 것에 있다. RankGraph-2는 popularity bias correction을 포함한 엣지 subsampling으로 간선을 수백억 단위로 줄이고, backbone 그래프에서 personalized PageRank(PPR)로 multi-hop 이웃을 사전 계산해 self-contained 학습 데이터를 만든다. 이어 residual-quantization 기반의 co-learned cluster index를 학습에 직접 결합해 KNN 없이도 재구성 오차와 retrieval 목적을 동시에 만족시키도록 한다.

- **Empirical Impact**: RankGraph-2는 오프라인 recall에서 GAT + Deep Graph Infomax 대비 3.8배, PyTorch-BigGraph 대비 2.1배 향상을 보였다. 14일 A/B 테스트에서는 CTR 최대 +0.96%, CVR 최대 +2.75% 개선을 기록했으며, U2U 서빙 인프라 비용은 83% 절감했다. 나아가 Meta의 주요 서피스에서 20회 이상 retrieval 런치에 적용되며 실사용 관점의 효과를 입증했다.



### Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Scienc (https://arxiv.org/abs/2606.19051)
Comments:
          ASIST 2026

- **Prior Approaches**: 기존 연구 방법 자동 분류는 주로 제목과 초록(title/abstract)에 의존해왔지만, 초록만으로는 방법론적 단서가 충분히 담기지 않는 한계가 있었습니다. 한편 full-text를 그대로 쓰려 하면 문서가 지나치게 길고 중복 정보가 많아 모델이 핵심 신호를 놓치기 쉽습니다. 그 결과, full-text를 어떻게 ‘어느 부분을’ 효과적으로 활용할지가 오래된 병목이었습니다.

- **Core Contribution**: 이 논문은 full-text를 물리적 위치(물리적 position) 기준으로 구간 분할한 뒤, 구간들을 조합하는 segment combination strategy를 제안합니다. 즉, 방법론 정보가 문서 전반에 균등하게 퍼져 있다는 가정 대신, 특정 구간이 분별력(discriminative power)을 더 갖는다는 관찰을 분류 파이프라인에 반영한 것입니다. 또한 서지 메타데이터(bibliographic metadata)를 cross-segment 조합과 함께 쓰면 성능을 더 끌어올릴 수 있음을 보여줍니다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) full-text의 길이와 정보 중복을 줄이면서도 (2) 다중 레이블(multi-label) 분류에 필요한 방법론 신호를 놓치지 않는 구간 설계를 찾는 데 있습니다. 저자들은 텍스트를 위치 기반으로 나눈 뒤, 다양한 구간 조합을 여러 모델에 걸쳐 평가해 ‘어떤 구간을 어떻게 합칠지’를 실증적으로 최적화했습니다. 여기에 bibliographic metadata를 결합해 구간 간 정보의 보완성을 강화하는 방식으로 해결했습니다.

- **Empirical Impact**: Library and Information Science 분야의 대표 저널 JASIST, LISR, JDoc에서 1,954편 full-text를 주석한 코퍼스로 실험했으며, 구간 단독·조합별 분류 성능을 비교했습니다. 결과적으로 방법론 정보는 full-text에 고르게 분포하지 않고, middle-to-late 및 마지막 구간이 특히 더 큰 분별력을 보였습니다. 더 나아가 서지 메타데이터와 cross-segment 조합을 함께 적용할 때 분류 성능이 일관되게 향상되어, method retrieval·review generation 같은 지식 서비스에 직접적인 활용 가능성을 시사합니다.



### Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents (https://arxiv.org/abs/2606.18947)
Comments:
          15 pages, Figure 8

- **Prior Approaches**: 기존 접근은 LLM의 native search grounding처럼 모델-공급자 경계 안에 검색 정책(프로바이더 선택, 결과 형식, 증거 주입, 비용·지연)을 숨겨두는 방식이 주류입니다. 이 때문에 검색 품질과 운영 지표를 튜닝·검사·이식·재사용하기 어렵고, 엄격한 출력 계약(예: JSON/단일 엔티티)을 깨는 Search-Induced Verbosity 같은 포맷 드리프트 위험이 커집니다. 또한 RAG나 도구 사용 연구는 검색-추론 상호작용을 다루지만, ‘실시간 검색 인터페이스’를 명시적 시스템 경계로 다루는 관점은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 Vendor-agnostic 경계로 검색 근거를 분리하는 Decoupled Search Grounding(DSG)을 제안합니다. DSG는 MCP-compatible gateway를 통해 검색을 추론 모델 바깥의 구조화된 tool 계층으로 옮기며, 프로바이더 라우팅, 출처 기반 context 렌더링, fallback, retrieval-depth, exact/semantic caching을 1급 제어(control)로 노출합니다. 결과적으로 추론 모델은 교체 가능하게 유지하면서도, 검색은 ‘운영 가능한 인터페이스’로 취급할 수 있게 됩니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 프로바이더별 이질적인 결과를 표준화해 모델이 일관된 근거를 받게 하고 (2) 캐시·fallback·비용/지연 목표를 정책으로 안정화하며 (3) 출력 계약 위반(Search-Induced Verbosity)을 인터페이스 레벨에서 완화하는 것입니다. DSG는 provider registry와 YAML 어댑터로 다양한 검색 백엔드를 같은 내부 결과 객체로 정규화하고, 캐시 키를 provider-scoped로 관리해 서로 다른 공급자 결과가 섞여 재사용되는 문제를 줄입니다. 또한 semantic cache의 유사도 임계값과 time-to-live(신선도 우선)를 설정해 재현성과 최신성 간 균형을 맞추며, 툴 호출/응답이 명확한 경계가 되도록 설계합니다.

- **Empirical Impact**: 5개 frontier 모델과 SimpleQA/FreshQA/HotpotQA, 그리고 e-commerce Query Intent Understanding(QIU) 프로덕션 워크로드에서 native search와 비교해 비용·지연·품질 트레이드오프를 체계적으로 입증합니다. SimpleQA에서 DSG는 86.1%(native 87.7%)에 근접하면서 검색 비용은 91% 절감했고, warm-cache hit rate는 99.4%, 지연은 68% 감소했습니다. FreshQA는 native가 앞섰지만(만약 신선도 이점이 강한 경우), QIU에서는 DSG가 native와 비슷하거나 약간 상회하면서 검색 비용을 98% 이상 줄였으며, 모델 공급자에 종속되지 않는 ‘운영 최적화 가능한 검색 경계’의 실용성을 강조합니다.



### Zero-Shot Active Feature Acquisition via LLM-Elicitation (https://arxiv.org/abs/2606.18933)
- **Prior Approaches**: 기존 Active feature acquisition(AFA)은 다음에 관측할 특성을 고르기 위해 사후분포가 얼마나 줄어드는지(예: entropy, conditional mutual information)를 계산해야 한다. 그래서 확률모델을 학습할 labeled data, RL episodes, 또는 특정 분포에서의 meta-training이 사실상 필수였고, 라벨이 희소한 임상·희귀질환 환경에서는 적용이 막히는 병목이 있었다. LLM을 지식 소스로 쓰더라도, LLM은 순차 계획과 확률적 객체로서의 캘리브레이션을 동시에 만족시키지 못해 오히려 성능이 불안정하다는 문제도 지적된다.

- **Core Contribution**: 논문은 LLM을 “알고리즘의 계획가”로 쓰지 않고, 오프라인에서만 형식적 확률모델을 채우는 역할로 분리해 zero-shot AFA를 만든다. LLM에게는 신뢰할 수 있는 정보만 요청하는데, 마코프 랜덤 필드(MRF)의 충분통계인 unary deviations(기준선 대비 일탈)와 pairwise co-variations(공변)를 discriminative 형태로만 elicitation한다. 또한 binary case에서 발생하는 gauge ambiguity를 maximum-entropy closure로 해소해, LLM이 준 log-ratio 대비 클래스로부터의 정보 기준선까지 일관되게 복원한다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM이 “클래스별 절대 분포”가 아니라 “대비되는 차이(로그비율)”만 잘 반환한다는 점에서, 전통적 AFA 기준(CMI 등)이 요구하는 per-class 조건부 분포를 그대로 계산할 수 없게 된다는 것이다. 논문은 MaxEnt 원리로 log-ratio를 만족하는 클래스-조건부 쌍들 중 추가 가정을 최소화하는 유일한 닫힘(closure)을 택해, acquisition criterion을 계산 가능하게 만든다. top-k 식별에서는 binary sign 판정 대신 dueling(두 엔티티 간 선호 비교)을 기본 단위로 설계하고, 관측된 상태에서 Wald–Chernoff 계열의 점수로 어떤 특성을 더 볼지 선택하는 정책을 제시한다.

- **Empirical Impact**: 실험은 Inflammatory Bowel Disease(IBD) 환자 코호트의 임상적 active setting에서 진행됐으며, 진짜 라벨 기준과 LLM이 추출한 beliefs 기준 모두에서 기존 LLM 기반 방법보다 성능이 좋았다. 특히 가장 어려운 환자(진단 모호성과 이질성이 큰 케이스)에서 top-k acquisition policy가 다른 기법들을 뚜렷하게 앞섰다. 더 나아가 knowledge–planning gap을 통제 비교로 보여주며, 동일한 LLM-derived 모델이라도 “순차 계획을 형식 알고리즘이 수행”할 때 관측 수를 크게 줄일 수 있음을 실증한다.



### LARE: Low-Attention Region Encoding for Text-Image Retrieva (https://arxiv.org/abs/2606.18885)
Comments:
          Accepted at the ICML 2026 Workshop on Efficient Multimodal Question Answering (EMM-QA). Code: this https URL ; Dataset: this https URL

- **Prior Approaches**: 기존 text-to-image retrieval은 CLIP/ALIGN 계열처럼 이미지 전체를 하나의 전역 임베딩으로 요약한 뒤 텍스트와의 유사도를 비교하는 dual-encoder 패러다임이 주류였습니다. 하지만 전역 임베딩은 시각적으로 두드러진 객체나 장면 맥락에 편향되어, 작은/덜 주목되는 영역의 미세 단서가 충분히 반영되지 못하는 문제가 있었습니다. FILIP, RegionCLIP 같은 fine-grained 정렬 방식은 더 촘촘한 대응을 시도하지만, 대체로 추가 학습이나 구조/연산 변화가 필요하다는 한계가 지적됩니다.

- **Core Contribution**: 논문은 LARE(Low-Attention Region Encoding)라는 학습 없는(inference-only) 보강 프레임워크를 제안해, 전역 임베딩에서 놓치기 쉬운 low-attention region의 정보를 임베딩에 함께 반영합니다. 비전 트랜스포머의 내부 self-attention 신호에서 덜 주목된 영역을 골라 그 영역을 다시 인코딩하고, 텍스트와의 유사도를 global+regional 관점에서 비교합니다. 또한 Dense-Set이라는 군집/장기 꼬리(long-tail) 객체에 초점을 둔 평가 서브셋을 COCO와 Flickr30K에서 재구성해 기존 모델의 취약점을 더 엄격히 드러내게 했습니다.

- **Technical Challenges**: 핵심 난제는 ‘덜 주목된 영역’을 신뢰할 수 있게 찾아내고, 그 추가 영역이 잡음이나 오탐을 만들지 않도록 전역 점수와 안정적으로 결합하는 것입니다. LARE는 frozen vision encoder의 attention map에서 inverse-attention(낮은 attention을 받는 패치들을 강조) 기반으로 후보 region을 생성하고, 제한된 개수의 region만 재인코딩해 공유 임베딩 공간에서 비교 가능하게 만듭니다. 결합은 hard maximum 대신 confidence-gated fusion을 써서 global similarity가 충분히 확실할 땐 전역 점수를 그대로 유지하고, 불확실할 때만 regional 점수를 보강하도록 설계했습니다.

- **Empirical Impact**: 실험은 zero-shot 설정에서 CLIP, SigLIP, SigLIP 2 같은 여러 백본에 대해 수행되며, 표준 COCO/Flickr30K에서는 성능 저하가 거의 없고 Dense-Set(군집·희귀 객체)에서 일관된 개선이 나타났다고 보고합니다. 예를 들어 COCO-Dense와 Flickr30K-Dense에서 R@1가 CLIP 기준 +5.18p(약 29%), +6.25p(상대 180%)처럼 큰 폭으로 상승했으며, 이는 정교한 객체 단서가 전역 임베딩에 의해 가려지는 상황에서 LARE가 효과적임을 시사합니다. Dense-Set용 재캡션이 low-attention/과소평가 영역을 언어적으로도 강조해, ‘군집 장면 미세 검색’에서 기존 retrieval 모델의 한계를 더 정확히 측정할 수 있게 만들었다는 점에서 의미가 큽니다.



### ScholarSum: Student-Teacher Abstractive Summarization via Knowledge Graph Reasoning and Reflective Refinemen (https://arxiv.org/abs/2606.18850)
- **Prior Approaches**: 기존 과학 초록 생성(abstractive summarization)은 주로 추출 기반으로 사실은 잘 보존하지만 문장 단편화로 논리 흐름이 약해지기 쉽다. 이후 BART/T5 같은 PLM 기반 생성은 문장 유창성은 크게 개선했지만, 긴 문서에서 섹션 간 통합과 근거 기반의 사실 일치가 흔들린다. LLM prompting·RAG·그래프 플래닝은 유망하지만, 결국 한 번의 생성 과정에서 생길 수 있는 누락/환각을 구조적으로 검증·수정하는 장치가 부족하다는 한계가 남는다.

- **Core Contribution**: ScholarSum은 학생-교사(student–teacher) 글쓰기 과정을 닮은 계층적 reflective 그래프 프레임워크로, 유창성과 factual faithfulness를 동시에 겨냥한다. 문서를 세만틱하게 분할해 계층형 지식 그래프를 만들고, 멀티 레이어 community 구조로 매크로 논리/테마를 먼저 계획한 뒤 학생이 초안을 작성한다. 이후 교사(Reviewer)가 유사 논문 검색 기반으로 초안을 평가하고, 근거가 없는 내용은 evidence re-retrieval과 재작성으로 반복 교정해 품질 기준을 충족할 때까지 진행한다.

- **Technical Challenges**: 핵심 난제는 (1) 긴 과학 문서의 논리 토폴로지를 충실히 반영하는 계층 표현을 구성하는 일과 (2) 전역 구조가 있어도 로컬 근거를 끌어와 수치·설정 등 미세 사실을 맞추는 일이다. ScholarSum은 LLM으로 엔터티/typed relation을 뽑아 knowledge graph를 만들고 community 알고리즘으로 테마 단위를 안정적으로 구성해 전역 계획을 제공한다. 또한 학생이 그래프 이웃의 핵심 triplet을 evidence 앵커로 삼아 초안을 생성·수정하도록 하되, 교사는 in-domain 유사 초록을 KNN으로 찾아 기준을 보정한 뒤 누락/부정확/논리 오류를 구체 피드백으로 돌려 major/minor revision을 트리거한다.

- **Empirical Impact**: ArXiv와 PubMed 벤치마크에서 ROUGE·METEOR·BERTScore는 물론, MiniCheck 기반 factuality 지표에서 ScholarSum이 기존 강력한 기준선 대비 일관된 개선을 보였다고 보고한다. 특히 PubMed처럼 용어와 실험 디테일이 촘촘한 도메인에서 반복 검증 루프 덕분에 정확성이 유지되는 경향이 강조된다. 아블레이션은 계층 그래프와 evidence grounding이 모두 필요하며, 특히 교사 기반 iterative refinement가 환각 감소에 큰 비중을 차지하고 재현성(run별 분산)도 낮아 안정적인 운영 가능성을 시사한다.



### TW-LegalBench: Measuring Taiwanese Legal Understanding (https://arxiv.org/abs/2606.18699)
Comments:
          10 pages, 2 figures, To appear in ICAIL 2026

- **Prior Approaches**: 기존 법률 벤치마크는 주로 common-law권 데이터에 치우치거나(MMLU, LegalBench) civil-law라도 번체/간체 중국어권을 일부만 커버해(예: LawBench, LawShift) 대만처럼 특정 관할의 법리까지 정밀 평가하기가 어려웠습니다. 또한 다수의 벤치마크가 객관식 위주로 구성돼 실제 변론·논증 같은 개방형 과정을 반영하지 못하고, 법 조항 범주도 지나치게 뭉뚱그려 오차 원인을 세분화하기 힘든 한계가 있었습니다.

본 논문은 TW-LegalBench가 이런 “관할 불균형·세분화 부족·개방형 평가 공백”을 동시에 겨냥한다고 설명합니다.

- **Core Contribution**: TW-LegalBench는 대만의 민법계 전통에 맞춘 공개 공식 코퍼스 기반 벤치마크로, 대만 법률 추론을 평가할 수 있는 실데이터 3종(MCQ/OEQ/LJP)을 한 프레임에 모읍니다. 특히 객관식은 조항(법규) 단위로 43개 법 유형을 라벨링하고, 개방형 에세이는 공식 채점 루브릭을 LLM-as-Judge로 분해 평가하며, 형사 판결은 범죄 유형별 결과 예측(LJP)으로 현실 정렬을 점검합니다.

즉, “정확한 조항 식별-논증 구성-판결 결과 추정”의 단계적 역량을 관할 특이적으로 측정하려는 것이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 (1) 대만 법조문을 정확히 연결해 평가할 수 있는 고해상도 라벨링(조항 계층: 條/項/款/目)과 (2) 개방형 답안의 채점 변동성을 줄이면서 루브릭에 기반한 공정한 점수화를 동시에 달성하는 것입니다. 논문은 조항 단위 수작업 어노테이션과 함께 OEQ는 루브릭 포인트별로 이산 라벨/점수를 출력하는 decomposed LLM-as-Judge 방식을 쓰고, 두 저지(gpt-5, claude-sonnet-4.5)의 평균으로 편향을 완화합니다.

또한 LJP에서는 양형·형량 수치 평가를 위해 정규화 로그거리(징역)와 절대/상대 허용 오차(구금·벌금·집행유예)를 조합해, 단순 텍스트 유사도만으로는 포착되지 않는 실무 기준을 반영하도록 설계했습니다.

- **Empirical Impact**: 13개 LLM을 평가한 결과, 자격시험(변호사 1차·자격) 기준을 넘는 모델은 있었지만(합격률 11%), 판사·검사 수준에서는 크게 떨어져(합격률 1~2%) “자격 단계별 역량 격차”가 드러났습니다. LJP에서는 평결 유형과 형량 예측 성능은 어느 정도 보였으나, 정확한 조항(Article) 인용에는 취약해 Type I/II 오류 양상이 확인되며 신뢰도 높은 법문 생성이 여전히 어렵다는 결론으로 이어집니다.

특히 전통적으로 대만/중국어 특화 코퍼스를 더 학습한 모델이 조항 인용 정확도에서 상대적 강점을 보였지만, OEQ·MCQ 성능 패턴을 함께 볼 때 “법리 추론”보다 학습 데이터의 암기 효과 가능성도 제기됩니다.



### Compact Geometric Representations of Hierarchies (https://arxiv.org/abs/2606.18520)
Comments:
          Published at the 39th Annual Conference on Learning Theory (COLT) 2026. 22 Pages

- **Prior Approaches**: 기존의 dense retrieval은 쿼리와 문서를 bi-encoder로 임베딩 공간에 매핑한 뒤, 내적/거리로 근접 이웃을 찾아 relevance를 판단합니다. 계층형 retrieval에서는 최근 You et al.가 DAG의 ancestor-descendant 관계를 reachability embedding으로 다뤘지만, 디escendant 수가 작을 때만 차원이 작게 보장되고 깊은 계층에서는 차원이 급격히 커지는 문제가 남았습니다. 즉, 그래프 구조를 더 정교하게 반영하는 “compact” 임베딩의 존재 조건이 불명확했습니다.

- **Core Contribution**: 이 논문은 hierarchical retrieval의 reachability를 더 작은 차원으로 표현하기 위해, 임베딩 차원이 트리width(treewidth)나 cross-edges 개수 같은 구조 파라미터에 의해 결정되도록 하는 이론을 제시합니다. 특히 directed tree(루트가 있는 방향 트리)에서는 그래프 크기와 깊이에 무관하게 차원 3의 reachability embedding이 항상 존재함을 증명합니다. 또한 treewidth가 t인 일반 그래프에 대해 차원 O(t log n) 상계를 주고, 그에 상응하는 하한(일반 DAG에서 Ω(n), treewidth t에서 Ω(t/log(n/t)))도 함께 제시해 경계가 거의 타이트함을 보입니다.

- **Technical Challenges**: 핵심 난제는 “reachability는 존재하지만, 단순한 분해/결합(예: spanning forest 기반 혹은 컴포넌트별 임베딩 결합)만으로는 내적이 거짓 양성(false positive)을 만들 수 있다”는 점입니다. 이 논문은 (1) directed tree/forest에서는 DFS의 discovery/finish time이 자손 관계를 구간 포함으로 요약한다는 성질로 차원 3 임베딩을 구성하고, (2) DAG에서는 spanning forest에 없는 경로를 만드는 cross-edge마다 좌표 1개를 추가해 이를 보정하는 augmentation 원리를 도입합니다. 나아가 treewidth 기반 증명에서는 separator로 균형 분할한 뒤, 재귀 임베딩을 결합할 때 생기는 거짓 양성을 제거하기 위한 좌표 설계를 추가로 수행합니다.

- **Empirical Impact**: 실험적으로 실제 데이터셋에서 임베딩을 구성할 수 있음을 보이고, 특히 high recall(높은 재현율) 환경에서는 기존에 이론적 보장을 함께 갖는 prior reachability embeddings보다 훨씬 작은 차원으로도 성능을 유지/개선할 수 있음을 보여줍니다. 이는 계층형 검색에서 “차원 축소”가 단순 휴리스틱이 아니라 그래프 구조 파라미터를 활용한 이론적 설계로 달성 가능하다는 메시지를 강화합니다. 결과적으로, dense retrieval의 임베딩 설계 관점에서 treewidth/cross-edge 같은 구조 복잡도를 정량 목표로 삼을 수 있는 길을 열었다는 점에서 의미가 있습니다.



### MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieva (https://arxiv.org/abs/2606.18508)
- **Prior Approaches**: RAG의 성능은 청크(chunk) 분할과 검색(granularity) 설계에 크게 좌우되지만, 기존 dense retrieval은 작은 청크가 정밀한 대신 후보 수가 폭증해 비용·지연이 커지고, 큰 청크는 여러 토픽이 섞여 dense 유사도에 잡음이 유입된다는 한계가 있습니다. 이를 줄이려는 기존 접근은 proposition-level 같은 더 세분화된 단위, LLM-guided chunking, RAPTOR 같은 계층적 검색, 또는 재랭킹/LLM 기반 evidence 선택을 사용하지만, 보통 전처리·인덱스·추가 추론 단계가 늘어나거나 추론 지연이 발생합니다.

- **Core Contribution**: MCompassRAG는 coarse-grained 청크의 장점을 유지하면서도, 청크를 토픽 메타데이터로 “주제별 나침반”처럼 검색 가능하게 만드는 메타데이터 가이드 검색 프레임워크를 제안합니다. 청크 임베딩 자체의 잡음에만 기대지 않고, 토픽 모델이 만든 토픽 분포를 청크와 동일한 임베딩 공간에 얹어 query가 해당 토픽 방향을 먼저 겨냥하도록 하며, LLM-teacher distillation으로 경량 retriever를 학습해 추론 시 추가 LLM 호출 없이 evidence 품질을 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 큰 청크가 토픽 혼합으로 인해 유사도 점수가 희석되는 문제를, 검색 후보 수를 늘리지 않고 해결하는 것입니다. 논문은 (1) 코퍼스 레벨 metadata bank에서 query에 맞는 토픽 분포를 선택하고 (2) 선택된 토픽들을 abstraction 모듈로 요약해 잡음과 편향을 줄인 뒤 (3) 해당 토픽 정보를 포함한 query-청크 표현을 극단적 multi-label 목적의 student MLP retriever로 점수화하여, LLM 기반 재랭킹 없이도 토픽 인지 검색을 구현합니다.

- **Empirical Impact**: 6개 retrieval 벤치마크에서 MCompassRAG는 비-LLM 최강 효율 베이스라인 대비 정보 효율(IE) 평균 8.24% 향상과 함께 지연은 5배 이상 낮은 성능-비용 균형을 보입니다. 특히 DRBench·LegalBench-RAG 같은 멀티홉이 어려운 설정에서 격차가 더 크게 나타났고, retrieval 시 LLM을 호출하는 oracle에 근접하면서도 추론 시 LLM 호출이 없어 실제 deep research 에이전트 시나리오에 의미 있는 효율 개선을 시사합니다.



### SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG (https://arxiv.org/abs/2606.18381)
- **Prior Approaches**: RAG의 성능은 청크 분할의 단위(너무 크면 잡음, 너무 작으면 문맥 단절)에 크게 좌우되는데, 기존 접근은 이를 LLM-guided chunking(청크 경계 학습), single-level 확장, 계층 요약 등으로 보완해 왔다. 다만 색인/검색 과정에서 외부 LLM 호출 비용이 들거나, 확장·요약이 단일 granularity에 고정돼 문장 간 의존성을 충분히 모으기 어렵고, 요약은 증거 손실을 유발할 수 있다.

- **Core Contribution**: SproutRAG는 문장 단위 청크를 attention에 기반한 계층적 구조로 재배열해, 다중 granularity 근거 검색을 end-to-end로 학습한다. 특히 문장-문장 inter-sentence attention을 head·layer 가중합으로 집계해 binary chunking tree를 만들고, 외부 LLM 호출이나 lossy 요약 없이 여러 레벨의 후보를 함께 검색한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 기존처럼 head·layer를 균일 평균하면 proximity bias가 생겨 트리의 전역 인덱스 품질이 떨어지고, (2) 트리 구조만 잘 만들고 임베딩 정렬이 약하면 검색 품질이 무너진다는 점이다. SproutRAG는 학습 가능한 가중치를 통해 어떤 attention head가 의미적 co-relevance를 잘 포착하는지 스스로 선택하고, 대비 학습 기반 임베딩 손실(검색 품질)과 attention 정규화(트리 구조 품질)를 joint objective로 함께 최적화한다.

- **Empirical Impact**: 네 벤치마크(과학·법률·오픈도메인)에서 SproutRAG는 정보 효율(IE)에서 평균 6.1% 향상을 보이며, Recall이 아니라 Precision까지 함께 개선되는 패턴이 관찰됐다. 또한 HotpotQA/WebQuestions/Dragonball의 생성 평가에서도 온라인 토큰과 지연을 낮춘 채 성능-효율 균형이 좋아, LLM-heavy reasoning·reflection 중심 시스템에 가까운 실용적 대안을 제시한다.



New uploads on arXiv(cs.CV)

### Native Active Perception as Reasoning for Omni-Modal Understanding (https://arxiv.org/abs/2606.19341)
Comments:
          Accepted at ICML 2026. Code and models: this https URL

- **Prior Approaches**: 기존 수동형 passive 모델은 ‘watch-it-all’처럼 프레임을 쿼리 난도와 무관하게 전부 처리해 계산비용이 영상 길이에 비례해 커지는 문제가 있었다. 인터랙티브 에이전트가 나왔지만, 전역 pre-scanning이나 길이에 비례하는 컨텍스트 비용을 유지해 장시간(예: hour-long) 영상에서 픽셀 보관과 계산이 병목이 된다. 또한 도구 기반 멀티모달 모듈은 추론과 지각 사이 정보 병목을 만들고, Think with Images류는 글로벌 버퍼 의존으로 진정한 디커플링이 어렵다는 한계가 지적된다.

- **Core Contribution**: OmniAgent는 멀티모달 비디오 이해를 POMDP 기반 Observation-Thought-Action(OTA) 반복 사이클로 재정의한 네이티브 omni-modal 에이전트다. 에이전트는 필요할 때만 frames/audio/clip을 불러와 오디오-비디오 단서를 ‘영속적 텍스트 메모리’로 선택 증류하고, 원본 미디어의 고차원 컨텍스트를 턴마다 정리해 추론 복잡도를 영상 길이로부터 분리한다. 그 결과 추론 턴을 늘릴수록 성능이 좋아지는 positive test-time scaling 특성이 관찰된다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 장기 에이전틱 추론을 위해 학습 신호가 붕괴되지 않도록 하고, (2) 다중 턴에서 중요한 발견 턴과 사소한 행동 턴이 섞일 때 credit assignment가 왜곡되는 점이다. 이를 위해 OmniAgent는 Agentic Supervised Fine-Tuning에서 best-of-N 궤적 합성과 dual-stage 품질 제어로 네이티브 active perception을 먼저 부트스트랩하고, Agentic Reinforcement Learning에서는 TAURA(Turn-aware Adaptive Uncertainty Rescaled Advantage)로 turn-level entropy를 사용해 Advantage homogenization을 완화한다. 특히 GRPO의 궤적 단일 advantage를 턴 단위로 재가중해, 불확실성 높은 ‘분기 지점’에 더 큰 보상/패널티가 가도록 설계했다.

- **Empirical Impact**: 10개 벤치마크(VideoMME, LVBench 등)에서 OmniAgent는 open-source 모델 중 최신 성능을 달성했으며, 특히 LVBench에서는 7B가 Qwen2.5-VL-72B(10배 더 큰 모델)보다 높은 성적(50.5% vs 47.3%)을 보이면서 프레임도 73% 더 적게 쓴다고 보고한다. 또한 VideoMME-Long에서 추론 턴을 늘릴수록 +6.2% 개선되는 형태의 positive test-time scaling이 실험적으로 확인되어 active perception의 효과를 뒷받침한다. 장시간 비디오 이해에서 ‘길이 증가에 따른 비용 폭증’ 문제에 대한 실질적 해법을 제시했다는 점에서, 멀티모달 에이전트 설계 방향에 영향이 클 전망이다.



### Beyond the Current Observation: Evaluating Multimodal Large Language Models in Controllable Non-Markov Games (https://arxiv.org/abs/2606.19338)
- **Prior Approaches**: 기존 게임 벤치마크(GameBench, GTBench 등)는 보이는 상태를 기반으로 한 탐색/계획을 주로 평가해 과거 관측을 다시 구성해 행동하는 능력(remember-to-act)을 분리하기 어렵다. 에이전트/멀티환경 스위트는 숨은 정보가 있어도 탐험·규칙 발견·자유 행동과 메모리 문제가 섞여 원인 해석이 약하고, long-context/메모리 벤치마크는 대체로 에피소드 후 질의처럼 기억 오류가 다음 입력에 피드백되지 않는 remember-to-answer 형태가 많다.

- **Core Contribution**: 이 논문은 RNG-Bench(Reconstructive Non-Markov Games)를 제안해 “현재 관측만으로 최적 행동이 불가능한 비마코프(non-Markov) 상황”에서 과거 관측을 재구성하고 즉시 행동하는 능력을 정밀하게 분리해 평가한다. Matching Pairs(정적·범주형 숨은 상태)와 3D Maze(동적·공간형 숨은 상태)를 닫힌-루프(closed-loop)로 설계하고, 격자/맵 크기·시각 패턴·관측 모달리티 축을 통일된 하네스로 제어한다.

- **Technical Challenges**: 핵심 기술적 난제는 “한 번만 보이고 사라지는 관측”을 역사(history)로부터 재결합해 다음 관측까지 바꾸는 기억-행동 결합(remember-to-act)을 안정적으로 수행하는 것이다. 논문은 oracle 상태 주입으로 모델이 맞는 숨은 상태를 받았을 때의 점수와 일반 조건의 점수 차이를 Memory Gap으로 정의해, 남은 실패가 주로 망각(forgetting)인지 의사결정(action selection) 문제인지 분해하며, duel 프로토콜로 인스턴스 변동성도 통제한다.

- **Empirical Impact**: 실험 결과, RNG-Bench는 매우 긴 컨텍스트(대략 128K tokens)와 다수 이미지 입력(에피소드당 약 350장) 같은 강한 기억 부담에서도 정면 승부가 가능한 “프론티어 모델 여지”를 보여준다. Matching Pairs에서는 Gemini-3.1-Pro와 GPT-5.4, Qwen3.5-397B의 격차가 두드러지고, 3D Maze에서는 작업 유형에 따라 순위가 달라져 hidden-state 수요가 구체적으로 다르다는 점이 확인된다. 또한 Qwen3.5-9B에 RNG-Bench 최적-정책 롤아웃 기반 fine-tuning을 하면 RNG-Bench 성능이 오르면서 기존 메모리/공간 벤치마크로 전이되며, Memory Gap 분석은 잔여 오류의 상당 부분이 망각에서 온다는 해석을 뒷받침한다.



### NeuMesh++: Towards Versatile and Efficient Volumetric Editing with Disentangled Neural Mesh-based Implicit Field (https://arxiv.org/abs/2606.19316)
Comments:
          TPAMI 2025; Project Page: this https URL

- **Prior Approaches**: 기존 neural radiance field(NeRF) 계열은 자유 시점 렌더링과 3D 재구성에는 강하지만, 아티스트가 기대하는 정밀한 편집(비강체 변형, 국소 텍스처 편집)을 하기엔 기능이 제한적이었습니다. 특히 rigid transformation 위주이거나, 특정 카테고리 기반 편집만 가능해 범용성을 떨어뜨린다는 한계가 지적됩니다. 또한 좌표기반 MLP/복셀/point cloud 같은 인코딩은 메시의 국소 ROI(관심 영역) 단위 편집과의 정렬이 약해 정교한 조형·텍스처 작업에 불리합니다.

- **Core Contribution**: 이 논문은 메시 기반 신경 표현 NeuMesh++를 제안해, neural radiance field를 메시 버텍스에 ‘분리된’ latent code(geometry, texture, semantic)로 인코딩함으로써 편집 가능성을 크게 확장합니다. 그 결과 mesh-guided geometry editing, 지정 영역 텍스처 편집( texture swapping, filling, painting ), semantic-guided editing 같은 워크플로를 효율적으로 수행할 수 있습니다. 더 나아가 geometry/appearance를 분리해 텍스처를 다른 형상으로 전이하거나, 세맨틱 클릭만으로 편집 영역을 자동 선택하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 버텍스에 단일 레이어 형태로 코드를 두었을 때 공간 구분이 어려워(면을 넘어 안/밖 방향 판별 실패 등) 학습 안정성이 떨어질 수 있다는 점입니다. 이를 위해 이 논문은 local space parameterization과 post-interpolation 융합 전략을 도입해, 주변 버텍스 대비 query point의 상대 관계(local indicator)로 공간 구분성을 보완하고 학습 수렴을 안정화합니다. 또한 view-dependent 텍스처 편집의 품질 저하를 막기 위해 버텍스별 modification color( view-independent )를 학습하고, painting 전이 시 다른 뷰에서의 과적합을 줄이기 위해 spatial-aware optimization으로 영향받는 modification color만 국소 fine-tuning하며, 세맨틱 클릭 편집을 위해 오픈 보케이블 segmentation 기반 세맨틱 코드를 버텍스에 추가하고 semantic propagation으로 3D 영역 선택을 자동화합니다.

- **Empirical Impact**: 실험에서는 실제 데이터와 합성 데이터의 다양한 편집 예시를 통해 표현 품질과 편집 능력에서 우수성을 보였다고 보고합니다. 특히 NeuMesh 계열 대비 렌더링 파이프라인을 implicit SDF가 아닌 neural radiance field로 재구성하고, distillation(teacher) 없이도 유사하거나 약간 더 나은 렌더링 품질을 달성해 학습 효율성 측면의 의미가 큽니다. 편집 동작 또한 빠르고(렌더링 약 15 fps, 편집 기능은 수 초 내), GUI까지 제공되어 메시 기반 CG 제작 흐름과의 연결성이 높다는 점이 분야 임팩트로 강조됩니다.



### Confidence is Not Reliability: Rethinking MC Dropout in Brain Tumour Segmentation (https://arxiv.org/abs/2606.19300)
Comments:
          Accepted for MIUA2016

- **Prior Approaches**: 기존 연구는 MC Dropout 같은 inference-time UQ가 병변/오류를 잘 “랭킹”하는지(AUROC 등)나 불확실성 지도를 정성적으로 확인하는 데 머무는 경우가 많았다. 또한 불확실성의 신뢰도는 전역 지표로만 평가돼, 치료 의사결정에 직결되는 ET(조영 증강 종양) 같은 특정 소구역에서의 과신(under- 또는 over-confidence) 실패가 가려질 수 있다는 한계가 지적돼 왔다. 
또한 model 간 비교에서도 전체 Dice나 전역 AUROC 중심이라, 모델 품질이 다른 상황에서 “오류와 불확실성이 어떻게/얼마나” 대응하는지 서브구역별 편차를 체계적으로 보지 못했다.

- **Core Contribution**: 이 논문은 MC Dropout으로 얻은 voxel-level 불확실성이 ET처럼 임상적으로 작은 영역의 오류를 안정적으로 탐지하는지 확인한다. 특히 불확실성-오류 정렬이 좋아도(E.g., AUROC~0.97) ET의 확률이 실제 정답률과 전혀 맞지 않는 “서브구역별 calibration 붕괴”가 발생할 수 있음을 실증한다. 
결론적으로 임상 배포 모델을 고를 때는 AUROC만이 아니라 ET 등 핵심 소구역의 calibration(예: ECE/신뢰도 곡선)까지 함께 보라고 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 MC Dropout 불확실성이 (1) 정확도 저하 없이 계산 가능해야 하고, (2) 오류를 잘 찾는지 랭킹 성능뿐 아니라 확률의 신뢰도까지 보장해야 한다는 점이다. 저자들은 SegResNet(사전학습 후 inference 시 post-hoc dropout)과 UNet-Res(학습 중 embedded dropout) 두 모델을 BraTS21 126명 테스트셋에서 비교하면서, 20-pass MC Dropout의 entropy/MI를 오류 탐지 점수로 쓰되 전역 AUROC와 함께 소구역 ECE·reliability diagram로 calibration을 분해해 확인했다. 
또한 entropy를 환자 단위로 분위(quartile) 분류해 오류 위험이 높은 환자군을 triage 형태로 뽑아내는 분석 프레임을 제시한다.

- **Empirical Impact**: 실험에서 두 모델 모두 Dice 변화가 거의 없었고(|ΔDice|<0.01), entropy 기반 AUROC-H가 약 0.97 수준으로 voxel 오류-불확실성 정렬은 “겉보기엔” 매우 좋게 나타났다. 그러나 UNet-Res의 ET는 ECE=0.915와 거의 평평한 reliability curve로 나타나, 불확실성이 오류를 신뢰성 있게 나타내지 못하는 치명적 실패 모드가 드러났다(ET entropy도 오류가 커져도 0.054 수준으로 거의 고정). 
반면 SegResNet은 ET에서 entropy가 오류 크기와 함께 증가해 임상적으로 의미 있는 신호를 제공했으며, entropy 상위 분위 환자군은 whole-tumour Dice가 더 낮아(중앙값 0.835 vs 0.925) 불확실성 기반 환자 triage의 실용성도 확인됐다. 



### A Unified Framework for Efficient Remote Sensing Visual Question Answering: Adapting Dual, Hybrid, and Encoder-Decoder Architectures (https://arxiv.org/abs/2606.19277)
Comments:
          4 pages, 2 figures, accepted and to be presented at 2026 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2026), scheduled for 9 to 14 August 2026 in Washington D.C

- **Prior Approaches**: 기존 Remote Sensing VQA는 RS-VQA 전용 데이터와 얕은 fusion(특징 벡터 결합)·supervised attention 같은 방법에 기대는 경우가 많았지만, counting·관계 추론에 필요한 미세한 공간 상호작용을 충분히 포착하지 못했습니다. Transformer 기반 foundation model을 쓰더라도 CLIP의 late fusion은 복잡한 멀티스텝 추론에 한계가 있고, BLIP는 강하지만 계산/학습 민감도가 커 소규모 RS 데이터에선 불안정할 수 있습니다. 또한 full fine-tuning은 대규모 모델의 VRAM·스토리지 비용 때문에 재난 현장 같은 자원 제약 환경에 적용이 어렵습니다.

- **Core Contribution**: 이 논문은 frozen 비전-언어 모델에 대해 RSAdapter를 VQA용으로 확장해, attention과 MLP 계층에 lightweight bottleneck adapter를 삽입하는 unified architectural surgery 파이프라인을 제안합니다. 세 가지 VLM 계열(dual encoder CLIP, encoder-decoder BLIP, hybrid FLAVA)에 동일한 방식의 “수술”을 적용해 RS domain shift를 파라미터 효율적으로 흡수하는 관점을 정립합니다. 특히 FLAVA가 multimodal reasoning과 retrieval 성격을 함께 가져가며 RS-VQA에서 가장 균형 잡힌 성능을 낸다는 실증 결과를 제공합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) RS 영상의 도메인 격차(시점/회전 불변성·극단적 객체 밀도)로 인해 일반 VLM이 시각 어휘를 제대로 재정렬하지 못한다는 점, (2) 전체 fine-tuning은 비용이 과도하다는 점, (3) 작은 객체·고해상도에서 정보가 손실되지 않도록 adapter 용량을 적절히 설계해야 한다는 점입니다. 저자들은 hidden dimension 대비 r=64의 bottleneck rank로 정보병목과 메모리 효율 간 균형을 맞추고, residual 경로로 gradient 안정성을 확보했습니다. 아키텍처별로 CLIP은 visual/text encoder에 adapter를 주입, BLIP은 image의 BlipAttention과 text decoder의 cross-attention에 주입, FLAVA는 FlavaLayer 단에서 일괄 수정해 적은 trainable 파라미터로 RS 적응을 달성했습니다.

- **Empirical Impact**: RSVQAx High-Resolution 데이터(772장, 1,000+ QA)에서 backbone은 모두 freeze하고 adapter 등 일부만 학습했으며, trainable 파라미터는 5% 미만 수준으로 수렴했습니다. 성능은 CLIP 72.4%, BLIP 76.8%(+4.4%), FLAVA 79.2%로 FLAVA가 최고 정확도를 보였고 “Presence/Area” 유형에서 특히 강했습니다. 실패 사례 분석에서는 그림자-수역 혼동 같은 visual polysemy가 CLIP에서 두드러졌고 FLAVA는 질문 텍스트 맥락을 활용해 false positive를 약 12% 줄였습니다. 결과적으로 재난 평가·도시 모니터링 같은 자원 제약 상황에서 resource efficient VQA의 새로운 기준선(baseline)을 제시했다는 점에서 실무적 의미가 큽니다.



### A Multi-Domain Benchmark for Detecting AI-Generated Text-Rich Images from GPT-Image-2 (https://arxiv.org/abs/2606.19259)
- **Prior Approaches**: 기존 AI 생성 이미지 탐지는 주로 물체 중심 이미지나 특정 생성 아티팩트에 초점을 맞춰 왔고, 텍스트와 레이아웃이 핵심 의미가 되는 ‘텍스트-리치’ 영역은 상대적으로 덜 다뤄졌습니다. 또한 기존 벤치마크는 최근 멀티모달 생성기(GPT-Image-2급)로 만든 전면(fully) 생성 문서형 이미지를 충분히 커버하지 못해, 탐지기의 일반화 능력을 체계적으로 보기 어려웠습니다. 결과적으로 카테고리별(문서/표/영수증/ UI 등) 실패 패턴이나 플랫폼 후처리(JPEG 등) 민감도 분석이 부족했습니다.

- **Core Contribution**: 이 논문은 GPT Image 2로 생성된 텍스트-리치 이미지를 6개 대표 범주(상업 포스터, 인포그래픽, 학술 포스터, 영수증, 표, UI 스크린샷)로 나눠 총 8,602장(실/생성 포함)의 멀티도메인 벤치마크를 제안합니다. 데이터는 텍스트-레이아웃 정규성(자유형~정형)과 기능적 맥락을 기준으로 구성해, ‘의미·배치 중심’ 시나리오를 정면으로 다룹니다. 이어서 다양한 탐지 패러다임(비전 트랜스포머, patch 기반, vision-language, 아티팩트/픽셀 관계 기반)을 zero-shot으로 한데 비교하는 통합 평가를 제공합니다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 텍스트와 구조가 결합된 문서형 포맷에서 생성 흔적이 희석되거나, (2) 압축·리사이즈 같은 플랫폼 후처리로 저수준 단서가 파괴된다는 점입니다. 논문은 전처리/파인튜닝 없이 공식 추론 파이프라인 그대로 실행하는 zero-shot 평가와 함께, JPEG 압축(품질 95~50), 리사이즈+JPEG, PNG 재인코딩 같은 후처리 조건을 바꿔 견고성을 점검합니다. 특히 이들이 탐지기에 따라 ‘실패 모드’가 달라지며, 아티팩트 기반 방법이라도 lossy JPEG에 크게 취약할 수 있음을 확인합니다.

- **Empirical Impact**: 실험 결과, 기존 탐지기는 전체 성능이 아니라 카테고리 의존성이 매우 커서, 일부 유형에서는 잘 맞지만 표·영수증 같은 정형 문서에서 급격히 무너지는 양상이 관찰됩니다. 또한 최강의 전통적 탐지기조차 JPEG 압축에서 재현성이 크게 흔들려, 디지털 신뢰/콘텐츠 진위 검증을 실사용 조건에서 신뢰하기 어렵다는 경고를 줍니다. 한편 GPT-5.5 같은 범용 비전-언어 모델은 전반 정확도가 더 높지만, 정밀한 구조와 텍스트 정렬이 요구되는 포맷(예: 표)에서는 여전히 병목이 남아 텍스트·레이아웃 인지형 탐지 필요성을 실증적으로 뒷받침합니다.



### CABLE: Cloud-Assisted Bandwidth-efficient LMM-based Encoding for V2X Systems (https://arxiv.org/abs/2606.19258)
- **Prior Approaches**: 기존 V2X 협력 인식은 원본 고해상도 영상을 그대로 전송하거나, 중간 특징을 압축해 통신을 줄이는 방식이 주류였습니다. 또한 ROI 선택을 위해 검출 기반(box/카테고리)이나 모션 기반(프레임 차분/광류)을 쓰는 연구도 있었지만, 검출 기반은 배경을 많이 포함하거나 open-vocabulary에 약하고, 모션 기반은 ego-motion에 쉽게 흔들려 오탐 ROI가 커질 수 있다는 한계가 있습니다.

- **Core Contribution**: CABLE은 클라우드의 cloud-hosted LMM(LISA++)이 생성한 분할 마스크를 다음 프레임 ROI 생성의 prior로 되돌려주는 mask-to-ROI-to-LMM feedback loop를 제안합니다. 엣지에서는 ego-motion 보정으로 이전 마스크를 전파하고 잔여 모션으로 보정한 뒤, corridor envelope로 ROI를 연속 영역으로 복원해 ROI-only 이미지 업로드를 가능하게 합니다. 이렇게 클라우드 LMM은 ROI 내부에 대해서만 분할을 수행해 visual token과 prefill 지연을 동시에 줄입니다.

- **Technical Challenges**: 핵심은 ‘의미 손실 없이 얼마나 적게 전송할지’와 ‘마스크 전파가 드리프트되면 어떻게 복구할지’입니다. CABLE은 (1) 순수 이동 가정의 homography로 기하 전파를 시작하되 residual-motion energy와 yaw-aware buffer dilation, (2) corridor envelope로 끊어진 ROI를 공간적으로 복원, (3) mask confidence가 낮아지면 full-frame refresh 키프레임으로 안정화하는 규칙을 함께 사용합니다. 그 결과 통신 절감과 함께 ROI 경계의 정보 손실을 제한하도록 설계되었습니다.

- **Empirical Impact**: nuScenes, WOD-ZB, Waymo, KITTI, CADC 5개 데이터셋에서 ROI pixel-coverage를 73–87% 줄이면서도 인식 품질을 크게 보존했고, LMM prefill은 추정 기준 5–8× 가속을 달성했습니다. 특히 open-loop(클라우드 첫 프레임만 분할 후 전파) 대비 feedback이 탐지 유지율을 크게 개선했으며, corridor envelope 없이 ROI가 분절되면 detection retention이 급락하는 등 공간 문맥 보존의 중요성이 확인됐습니다. 전체적으로는 소폭(제한적) 탐지 품질 트레이드오프를 전제로, bandwidth와 클라우드 지연을 동시에 줄이는 V2X LMM 배치 전략의 실증 가능성을 보여줍니다.



### OneCanvas: 3D Scene Understanding via Panoramic Reprojection (https://arxiv.org/abs/2606.19253)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 Vision-Language Model(VLM) 기반 3D 이해는 (1) 점구름 토큰화/geometry encoder 등 복잡한 모듈을 추가하거나, (2) distance·direction 같은 공간 QA를 대규모로 수집해 학습량을 키우는 방식이 주류였다. 그런데 최근 감사(audit) 결과, 텍스트만으로도 기준 성능에 근접하거나 나오는 경우가 있어 모델이 제공된 기하 입력을 ‘진짜로’ 읽기보다 장면/질문 통계의 지름길을 탄다는 한계가 지적됐다. 또한 대부분의 방식은 프레임을 조각조각 보며 공간관계를 일관된 좌표계로 결합하지 못하는 문제가 남아 있다.

- **Core Contribution**: OneCanvas는 여러 시점의 패치 특징을 depth와 카메라 포즈로 3D로 ‘리프트(lift)’한 뒤, 하나의 equirectangular 파노라마 캔버스에 longitude·latitude 좌표로 재투영해 VLM이 이미지처럼 그대로 읽게 만든다. 이때 캔버스의 원점(origin)과 방향을 태스크에 맞게 자유롭게 선택할 수 있어, 로보틱스/embodied AI에서 중요한 ‘지정된 시점에서의 situated reasoning’을 같은 표현으로 바로 지원한다. 나아가 캔버스 위에 실제 이미지에서 뽑은 객체 패치 특징을 임의의 3D 위치에 절차적으로 배치해, 답 분포를 제어하면서 공간 사전학습 커리큘럼을 on-the-fly로 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (a) 겹치는 관측에서 픽셀 단위 rasterization으로 정보를 뭉개지 않으면서, (b) VLM의 기존 3D-RoPE 의미를 깨지 않고 metric(미터) 좌표 정보를 주입하는 것이다. OneCanvas는 rasterization 대신 각 리프트 패치를 연속적인 (ϕ,θ) 위치를 갖는 개별 토큰으로 유지해, 중첩/가림 상황도 attention으로 구분하게 했다. 또한 각 패치의 캔버스 좌표 오프셋(qx,qy,qz)을 다축·방사(radial) 성분의 sin/cos 포지셔널 인코딩과 함께 특징 공간에 게이트된 방식으로 주입해, 각도 캔버스로 평탄화되며 사라지는 depth/거리 단서를 복원한다.

- **Empirical Impact**: 실험에서 OneCanvas는 SQA3D와 VSI-Bench에서 최신 성능을 달성하고, SPBench에서는 zero-shot에서도 우수한 일반화를 보이며 계산량은 경쟁 최고 대비 한 자릿수(대략 10배) 이하 수준으로 줄였다. 특히 SQA3D에서 ‘Which’처럼 시점 의존성이 큰 질문 유형에서 캔버스 원점 중심화(ablation)의 영향이 크게 나타나, 표현이 실제 situated geometry를 반영함을 시사한다. 즉, 아키텍처/부가 모듈 없이 입력 표현과 커리큘럼만으로 3D 추론을 강화할 수 있다는 실증적 근거를 제공해 해당 분야의 설계 관점을 바꾸는 의미가 있다.



### Transformer Geometry Observatory TGO-I: Spectral Geometry Observatory (https://arxiv.org/abs/2606.19249)
- **Prior Approaches**: 기존 연구는 attention 분석, feature visualization, pruning, representation similarity(예: CKA류), downstream 성능 평가로 변화를 일부 파악해왔다. 또 representation geometry나 spectral properties 같은 스펙트럼 관찰도 있었지만, 특정 레이어나 개별 학습 시점에 한정돼 학습 시간과 네트워크 깊이가 함께 진화하는 양상을 전반적으로 보긴 어려웠다. 그 결과 “학습이 정보를 지배적 소수 방향에 몰아넣는가?” 같은 직관이 정량적으로 검증되기엔 관측 도구가 부족했다.

- **Core Contribution**: 이 논문은 Vision Transformer의 내부 표현을 ‘진화하는 기하 객체’로 보고 스펙트럼 기하를 추적하는 관측 프레임워크 Transformer Geometry Observatory(TGO)를 제안한다. 그 첫 편 TGO-I에서 ViT-Small/16(Imagenet-100, 100 epochs)을 기준으로 각 레이어의 covariance 구조와 eigenspectrum을 학습 전 구간에 걸쳐 체계적으로 측정한다. 특히 최종 CLS 토큰 표현이 중간 토큰/레이어와 다른 고유한 기하적 성질(차원 활용 최대, anisotropy 최소)을 보인다고 정리한다.

- **Technical Challenges**: 핵심 난제는 “학습이 진행되는 동안 표현의 기하를 일관된 기준으로 측정”하는 관측성(observational consistency)이었다. 논문은 학습 미니배치마다 바뀌는 샘플링 노이즈를 피하기 위해 고정된 검증 부분집합 1000장으로 forward hook 기반 활성값을 추출하고, 매 epoch마다 covariance를 재구성해 Effective Rank, Stable Rank, Participation Ratio, Spectral Entropy/Flatness/Anisotropy 등 다중 스펙트럼 지표를 계산한다. 이를 통해 레이어 깊이와 학습 시간의 결합된 변화를 재현 가능하게 기록한다.

- **Empirical Impact**: 실험 결과는 학습이 진행될수록 Effective Rank/Stable Rank/Participation Ratio/Spectral Entropy가 대체로 증가하고, Spectral Anisotropy는 감소하는 일관된 패턴으로 나타났다. 또한 covariance가 점점 더 대각화(diagonal)되고, eigenspectrum과 singular value spectrum도 더 평탄(flat)해져 분산이 소수 방향에 집중되기보다 더 넓게 재분배되는 현상을 보여준다. 특히 CLS 토큰은 네트워크 내에서 가장 높은 유효 차원과 가장 낮은 anisotropy를 기록해 “글로벌 요약 표현이 가장 분산된 기하 상태로 수렴한다”는 관찰을 제공하며, 이후 토큰 다이내믹스/의미성/중복성/최적화 기하를 잇는 후속 관측 연구의 발판이 된다.



### GUMP-Net: An interpretable model-data-driven intelligent algorithm for multi-class pelvic segmentation (https://arxiv.org/abs/2606.19215)
Comments:
          26 pages, 8 figures, 3 tables

- **Prior Approaches**: 과거 펠비스 세그멘테이션은 주로 U-Net 계열의 DLM로 정확도를 끌어올렸지만, 결정 과정이 black-box로 남아 임상 해석이 어렵다는 지적이 컸다. 반면 active contour model(ACM)/level set 모델은 기하학적 해석이 가능하지만, 영상 품질·초기 조건·하이퍼파라미터에 민감하고 대규모 데이터 학습을 충분히 활용하지 못했다. 두 계열을 결합하려는 시도들은 있었지만(예: 초기값 예측 후 ACM 보정, 에너지 함수를 loss로 재해석, algorithm unrolling), 최종 성능이나 실제로 얼마나 ‘해석 가능함’을 유지하는지에 한계가 있었다.

- **Core Contribution**: 이 논문은 다중 해부학(예: 좌/우 hip, sacrum) 펠비스 멀티클래스 세그멘테이션을 목표로 GUMP-Net을 제안한다. GUMP-Net은 object detection module로 level set 초기화를 자동화하고, edge detector module이 anatomy-aware learned edge detector function을 학습하며, iteration module이 improved GAC의 반복을 네트워크로 내재화해 deep level set evolution을 수행한다. 즉, level set의 기하적 의미는 유지하면서도 CNN/학습 기반의 강점을 결합한 ‘interpretable model-data-driven’ 접근을 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) classical GAC의 edge detector가 관심 없는 구조까지 함께 잡아 오세그멘테이션을 유발하는 문제와 (2) level set 반복을 학습 가능하게 만들면서 안정적 진화를 보장하는 문제다. 이를 위해 EDM은 hand-crafted EDF의 임계(soft-tissue 억제, bone 강조)를 픽셀별로 CNN이 예측하도록 바꿔 learned EDF를 구성하고, iteration module에서는 Euler 기반의 반복 형태를 residual 예측 네트워크로 바꿔 N-step 진화를 trainable block으로 만든다. 또한 ODM 기반 초기 contour가 추론 시 흔들릴 수 있으므로, 학습 단계에서 ground-truth bounding box에 Bounding Box Random Shift를 적용해 초기 불일치에 대한 강건성을 높인다.

- **Empirical Impact**: 실험에서는 CTPelvic1K/CLINIC, Pelvic Collected, 그리고 Ankle Collected 등 펠비스 CT 데이터로 성능과 안정성을 검증하며, CNN·Transformer 기반 기준 모델과 algorithm unrolling 계열(FAS-UNet, PottsMGNet) 대비 특히 small training data 상황에서 더 정확하고 일관된 세그멘테이션을 보였다고 보고한다. 또한 SAM/MedSAM 같은 foundation model과 비교해 시각적으로도 더 나은 결과를 제시하며, 앵클에서도 성능이 유지되어 다른 해부학으로의 확장 가능성을 뒷받침한다. 마지막으로 fracture reduction 같은 복합 임상 시나리오에서 효율적인 세그멘테이션과 함께, deep learning을 geometry 관점에서 이해하는 해석 틀을 제공한다는 점을 의미로 강조한다.



### ROSA-TFormer: A Radar-Optical Sensor-Aware Temporal Transformer for Pinus sylvestris Plantation Classification in Northern Shaanxi Using GEE-Derived Sentinel-1/2 Time Series (https://arxiv.org/abs/2606.19204)
Comments:
          journal in tree classification

- **Prior Approaches**: 기존에는 Random Forest나 XGBoost처럼 월/계절 정보를 요약 통계·식생지수 등으로 수동 설계해 분류하는 방식이 많이 쓰였다. 하지만 이런 방식은 연간 시계열에서 나타나는 비국소(non-local) 계절 의존성과 SAR/광학의 센서별 물리적 기여를 충분히 분리해 학습하기 어렵다. 딥러닝의 경우 1D-CNN이나 generic Transformer가 종종 사용됐지만, SAR와 optical을 하나로 초기에 섞어 처리하거나 센서별 잡음/의미 차이를 반영하는 설계가 약했다.

- **Core Contribution**: ROSA-TFormer는 Pinus sylvestris var. mongolica(장자이송) 점 단위(point-level) 분류를 위해 Sentinel-1/2 연간 시계열을 radar-optical sensor-aware temporal Transformer로 모델링한다. SAR와 optical을 별도 임베딩 브랜치로 만들고, 토큰 수준에서 센서 기여를 조절하는 sensor-aware gate와 연간 구간의 중요도를 뽑는 temporal attention pooling을 결합한다. 또한 월간 12토큰과 반월 24토큰을 동일한 샘플 구성으로 비교해 시간 해상도 효과도 함께 검증한다.

- **Technical Challenges**: 센서 간 성격이 달라 동일한 Transformer에 단순 early-fusion으로 넣으면 계절·잡음·물리적 의미가 섞여 성능이 흔들릴 수 있다. 논문은 GEE에서 월/반월 합성(compositing)으로 불규칙 관측을 조밀한 텐서로 재구성하고, 학습셋 통계 기반 정규화로 정보 누수를 막는다. 그 위에 SAR/optical 분리 임베딩, 독립 sigmoid 형태의 sensor gate, class token과 attention pooling의 dual aggregation을 적용해 연간 비대칭 증거를 효과적으로 결합하도록 설계했다.

- **Empirical Impact**: HalfMonth-dataBig에서 ROSA-TFormer는 overall accuracy 99.67%, macro F1 99.56%, P. sylvestris F1 98.91%를 달성했으며, ablation과 공간 블록 검증에서도 radar-optical temporal fusion과 sensor-aware 설계의 유효성이 확인됐다. 다만 공간 분할에서는 모델 순위가 일부 변해 ‘항상 최선’이라기보다 경쟁적이고 해석 가능한 성능이라는 결론이 강조된다. 전반적으로 구름·계절 수렴 문제에 강한 SAR 보완과 연간 시계열 증거 학습을 장자이송 모니터링에 연결한 점에서, 향후 wall-to-wall 검증 확장에 대한 실용적 신호를 제공한다.



### Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performanc (https://arxiv.org/abs/2606.19195)
- **Prior Approaches**: 10B급 산업용 diffusion 기반 인페인팅 모델은 zero-shot 생성 품질을 크게 끌어올렸지만, 막대한 연산과 메모리 비용 때문에 실제 배포가 어렵다는 한계가 뚜렷했다. 이를 줄이려는 시도로 PixelHacker의 Latent Categories Guidance(LCG)나 GLA 같은 효율화가 나왔으나, 여전히 크고(거의 10억 파라미터대) 구조적으로 cross-attention 요구를 완전히 흡수하지 못한다. 또한 컨볼루션/어텐션을 단순히 light 연산으로 치환하면 표현 병목이 발생해 FID와 같은 품질이 급격히 무너지는 것으로 분석됐다.

- **Core Contribution**: Moebius는 ‘경량 task-specific specialist’로서 10B급 일반론자 성능을 노리되, 구조 압축으로 생기는 표현 병목을 Moebius의 핵심 모듈과 distillation의 조합으로 해결한다. 구체적으로 Local-λ Mix Interaction과 Interactive-λ Mix Interaction을 통해 local 문맥과 LCG의 global 의미 priors를 고정 크기 선형 행렬로 요약해, 복잡한 latent 상호작용을 유지하면서도 파라미터를 대폭 줄인다. 여기에 극단 압축으로 생기는 용량 손실을 adaptive multi-granularity distillation(다중 해상도/복수 loss의 동적 가중)으로 보정해, 압축과 품질의 동시 달성을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술 난관은 극단적인 구조 경량화가 표현 병목을 일으켜 semantic reasoning과 공간-텍스처 정렬 품질을 동시에 떨어뜨린다는 점이다. Moebius는 (1) GLA의 교차어텐션 제약을 Local-λ/Interactive-λ로 우회해 cross-attention에 해당하는 상호작용을 선형 복잡도로 구현하고, (2) DWConv 기반 spatial 블록과 Mix-FFN으로 FFN 파라미터를 더 압축해 전체 경량화를 밀어붙인다. 하지만 그 결과로 용량 상한이 생기므로, 픽셀 공간 디코딩 없이 latent 공간에서 coarse/fine 정렬과 latent LPIPS류 지각 제약을 함께 최적화하되, loss들의 gradient norm 기반 동적 가중으로 학습 균형 문제를 완화한다.

- **Empirical Impact**: 실험은 natural(Places2)과 portrait(CelebA-HQ, FFHQ) 벤치마크 전반에서 Moebius의 효율-품질 트레이드오프를 입증한다. Moebius는 FLUX.1-Fill-Dev 대비 총 추론 시간에서 >15× 가속을 달성하면서도, 생성 품질은 10B급 FLUX.1-Fill-Dev에 견줄 뿐 아니라 일부 설정에선 상회하는 결과를 보고한다. 특히 파라미터는 0.22B로 11.9B 대비 2% 미만이며, 단계당 26.01 ms 수준의 지연 특성과 함께 ‘고충실 inpainting’의 새로운 효율 기준을 제시했다.



### When AUC Misleads: Polarization-Aware Evaluation of Deepfake Detectors under Domain Shif (https://arxiv.org/abs/2606.19184)
- **Prior Approaches**: 기존 딥페이크 탐지는 XceptionNet/EfficientNet 같은 이진 분류기로 real-vs-fake을 학습한 뒤 AUC를 주로 사용해 성능을 비교해 왔습니다. 또한 일반화 향상을 위해 multi-task learning이나 data synthesis 계열 방법이 등장했지만, 평가에서는 각 데이터셋에서 AUC를 따로 계산해 평균내는 방식이 관행적으로 유지됐습니다. 이 방식은 실제 배포 환경의 domain shift(데이터 소스·아티팩트 혼재)를 반영하지 못해 낙관적으로 측정될 수 있습니다.

- **Core Contribution**: 이 논문은 데이터셋 간 domain shift를 더 현실적으로 반영하는 평가 지표로 Cross-dataset AUC(Cross-AUC)를 제안합니다. Cross-AUC는 평균 AUC(순위 성능)뿐 아니라 예측 점수의 polarization(클래스 분리가 얼마나 극단적으로 벌어졌는지)을 함께 고려합니다. polarization은 Wasserstein Distance로 real/fake 점수 분포가 얼마나 분리되는지 정량화해, 단순 성능 저하가 ‘왜’ 발생했는지에 대한 해석 가능성도 제공합니다.

- **Technical Challenges**: 핵심 난제는 AUC가 데이터셋별로는 높아도 실제로는 임계값이 불안정해질 수 있다는 점을 수치화하는 것입니다. 이를 위해 데이터셋마다 최적 threshold가 흔들리는 현상을 관찰하고, 단순히 threshold 분산이 낮아도 점수 분포가 좁게 뭉치면 여전히 취약할 수 있음을 보였습니다. 해결책으로는 real/fake 점수의 분포를 극집합(polar sets)으로 정의하고, 이 분리 정도를 Wasserstein Distance로 측정한 polarization을 Cross-AUC에 보상/패널티 형태로 통합합니다.

- **Empirical Impact**: 7개 벤치마크(FaceForensics++ 등)에서 Cross-AUC는 기존 mean AUC 방식보다 domain shift 하의 안정성과 실사용 적합성을 더 잘 반영하는 것으로 나타났습니다. 특히 Cross-AUC 값이 여러 데이터셋을 합친 Combined 데이터에서의 AUC와 매우 가깝게 대응되어, 실전의 ‘혼합 소스’ 평가를 대체할 실용성도 강조됩니다. 또한 polarization이 낮아질 때 성능 하락이 동반되는 패턴을 통해, 단순 수치 비교를 넘어 탐지 모델의 의사결정 거동 차이를 설명하는 데 도움이 된다는 점이 부각됩니다.



### Hand-4DGS: Feed-Forward 3D Gaussian Splatting for 4D Hand Reconstruction from Egocentric Videos (https://arxiv.org/abs/2606.19156)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 egocentric 4D 손 복원은 (1) 멀티뷰 3D Gaussian splatting처럼 단안 깊이 모호성을 해소하려고 많은 시점을 요구하거나, (2) body용 방법(HUGS/EVA 계열)을 단순히 hand에 적용해 포즈 추정기의 초기 메시 오류가 가우시안으로 전파되는 문제가 있었다. 또 feed-forward 3DGS 계열은 주로 static 아바타에 최적화되어 4D(시간적) 손동작을 안정적으로 모델링하기 어려웠다. 결과적으로 단안/착용형 환경의 빠른 카메라 움직임, 심한 가림, 손의 급격한 동역학에서 성능과 실시간성이 동시에 깨지기 쉬웠다.

- **Core Contribution**: 이 논문은 Hand-4DGS로, egocentric 비디오로부터 동적인 4D 손을 직접 재구성하는 최초의 feed-forward 3D Gaussian 프레임워크를 제안한다. 핵심은 손 구조 priors를 위해 MANO 메시에 “mesh-guided representation(정점 앵커드 positional embeddings)”을 도입하고, temporal convolutions로 시간적 일관성을 확보해 포즈 추정기 의존을 추론 단계에서 제거하는 것이다. 이를 통해 약 60 FPS 수준의 빠른 추론과 미지 비디오로의 강한 일반화를 동시에 노린다.

- **Technical Challenges**: 문제는 단안 관측에서의 내재적 모호성, 고개/카메라의 빠른 움직임, 손-손 및 손-물체 가림, 그리고 시간 축의 jitter를 동시에 다뤄야 한다는 점이다. Hand-4DGS는 (a) MANO 메시에 가우시안을 구조적으로 정렬하고, (b) 정점 임베딩을 삼각형 내 barycentric interpolation으로 보간해 더 높은 해상도의 외관/크기를 학습하며, (c) 프레임 윈도우의 특징을 temporal convolution으로 묶어 동작의 흔들림을 줄인다. 또한 3D pose ground-truth 없이도 Gaussian splatting 렌더링 기반 2D 이미지 supervision을 통해 손 메시와 가우시안 속성을 함께 정렬하도록 학습을 설계했다.

- **Empirical Impact**: H2O와 ARCTIC의 egocentric hand–object 및 양손(비틀림) 시나리오에서 Hand-4DGS는 HUGS/EVA로 조정한 baseline 대비 4D 복원 품질과 손 포즈 정확도 모두에서 유의미한 개선을 보였다. 특히 느린 동작에서는 baseline이 버티더라도, ARCTIC처럼 급격한 동작에서 기하 붕괴와 시각적 아티팩트가 커지는 반면, 제안 모델은 mesh-guided 제약과 temporal-aware 특징으로 안정성을 유지했다. 또한 테스트 시 재학습 없이도(zero-shot) 미지 시퀀스에 대해 그럴듯한 결과를 내고, TTO를 선택하면 색상 편차를 빠르게 줄일 수 있어 AR/VR·AI glasses 같은 실사용 맥락의 확장성에 의미가 크다.



### Urdu Katib Handwritten Dataset: A Historical Document Dataset for Offline Urdu Handwritten Text Recognition with CRNN-Based Baseline Evaluation (https://arxiv.org/abs/2606.19139)
- **Prior Approaches**: 우르두 UHTR은 Arabic 계열의 커시브 특성 때문에 기존 OCR/HTR 대비 어려움이 크며, 연구는 상대적으로 소수에 그쳤습니다. 특히 문맥 민감성, 겹침, 대각선 필기, 점(누크타) 위치의 복잡성, 형태 유사성 때문에 문자 단위 인식이 흔들립니다. 또한 공개 벤치마크 데이터셋이 부족해 모델 학습과 공정한 비교가 제한되어 왔습니다.

- **Core Contribution**: 이 논문은 역사적 우르두 카티브(Katib)가 남긴 필체에서 추출한 오프라인 우르두 필기 텍스트 라인 데이터셋인 Urdu Katib Handwritten Dataset(UKHD)을 제안합니다. UKHD는 Nastalique 서체의 flat nib 필기 변형을 반영하며, ‘Plain Urdu Text Lines(PUTL)’와 ‘Mixed Urdu Text Lines(MUTL)’ 두 부분으로 구성됩니다. 아울러 UKHD의 주요 부분에 대해 CRNN 기반 하이브리드 모델들을 비교해 Urdu Katib Handwriting Recognition(UKHR)의 기준선과 최적 아키텍처를 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 먼저 복잡한 필기 이미지를 라인 세그멘테이션과 정확한 전사 라벨링이 가능하도록 만드는 것입니다. 논문은 PDF 원본에서 페이지 이미지를 추출한 뒤, RGB→그레이스케일 변환, 페이지 뒷면 그림자 잡음을 완화하기 위한 median 필터링, Horizontal Projection Profile(HPP) 기반 deskew로 텍스트 기울기를 보정하는 전처리 파이프라인을 구성합니다. 이어서 라인 분할 및 (세미-)자동 라벨링 절차를 도입해 데이터셋 제작 시간을 줄이는 방식으로 확장성을 확보했습니다.

- **Empirical Impact**: 실험에서는 CNN-BGRU-CTC 모델이 다른 CRNN 하이브리드 구성보다 Character Error Rate(CER)과 Word Error Rate(WER) 측면에서 더 견고한 성능을 보였다고 보고합니다. 이는 데이터 부족으로 인해 난관이었던 UHTR 연구에 대해, 역사적 flat nib 필체까지 포함한 학습·평가 기반을 제공한다는 점에서 의미가 큽니다. 결과적으로 우르두 필사 문헌의 디지털 보존과 검색 가능성을 높이는 인식 시스템 개발을 촉진할 것으로 기대됩니다.



### ProductConsistency: Improving Product Identity Preservation in Instruction-Based Image Editing via SFT and RL (https://arxiv.org/abs/2606.19103)
Comments:
          CVPR HiGen 2026

- **Prior Approaches**: 기존 instruction-based image editing 연구는 마스크·참조 기반 제어, 또는 MLLM/Transformer 결합으로 전반적인 구조 보존과 지시 수행을 강화해 왔습니다. 그러나 광고·커머스 맥락처럼 로고·브랜딩·객체 위 텍스트의 “픽셀 수준 정합성”이 필수인 경우, 텍스트가 깨지거나(맞춤법/문자 오류), 문장 자체를 환각하는 등 fine-grained object identity 보존이 자주 실패합니다. 또한 product+text fidelity를 전면에 둔 데이터셋/벤치마크가 부족해, 이 문제를 모델의 암묵적 능력에 맡겨온 한계가 있습니다.

- **Core Contribution**: 이 논문은 product-centric(제품/브랜드)과 text fidelity를 명시적으로 목표로 하는 ProductConsistency Dataset과 ProductConsistency Benchmark, 그리고 Product-aware 학습 프레임워크를 제안합니다. 합성 데이터 생성 파이프라인을 통해 검증 가능한 렌더링 텍스트를 포함한 학습용 데이터(SFT 87k, RL 869)를 만들고, 174개 제품(8개 카테고리)×5개 프롬프트(총 870개)로 표준 평가를 제공합니다. 특히 RL 단계에서 Cyclic Consistency reward로 “편집 후 이미지에서 생성된 캡션”이 원래 제품 설명과 의미적으로 맞아떨어지도록 유도해 제품 정체성을 보존합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 제품 위 텍스트를 포함한 편집 데이터의 품질을 사람이 검증하듯 확보하는 것과 (2) 제품 정체성을 직접 측정하기 어려운 보상 함수를 안정적으로 설계하는 데 있습니다. 저자들은 GPT로 제품/브랜드 시그니처와 텍스트 렌더링 목표를 만든 뒤 OCR로 문자 단위 정합성을 필터링해, “맞는 텍스트만” 남기는 고정밀 합성 데이터셋을 구축했습니다. 보상은 캡션 유사도(semantic proxy)를 Cyclic Consistency로 주되, OCR 기반 text reward를 함께 사용해 텍스트 오류에 더 민감하게 학습하도록 했고, GRPO 기반 RL로 두 보상을 균형 있게 최적화합니다.

- **Empirical Impact**: 실험에서는 Qwen-Image-Edit-2511과 Flux.1-Kontext-dev를 ProductConsistency로 SFT·RL 파인튜닝해 기준선 대비 일관된 개선을 보입니다. 가장 인상적인 결과로 Qwen-Image-Edit-2511은 RL(사이클릭 보상) 후 문자 오류율(CER)을 약 5배(1.0682→0.2080) 줄였고, Seg CLIP-I·Seg DINO-I도 함께 상승해 제품/텍스트 및 시각 품질이 동시 강화됨을 확인했습니다. 또한 OCR 정합뿐 아니라 MLLM-as-a-judge 평가에서 text fidelity·product consistency·aesthetics·instruction following 전반의 점수 상승이 관찰되어, 데이터·목표 설계가 실제 품질로 이어진다는 점을 뒷받침합니다.



### AMALIA-VL: A Native European Portuguese Open-Source Vision and Language Mod (https://arxiv.org/abs/2606.19100)
- **Prior Approaches**: 기존 LVLM들은 대체로 개방형 가중치를 공개하더라도 학습 데이터·세부 구현이 제한적이거나, 혹은 완전히 재현 가능한 파이프라인이라도 pt-PT를 “기본 지원” 정도로만 다뤘습니다. 특히 pt-PT는 웹 스케일 말뭉치에서 pt-BR로 흡수되는 편향 때문에 멀티모달 작업 전반에서 성능이 체계적으로 떨어졌고, pt-PT 전용 instruction-following LVLM은 부재했습니다.

- **Core Contribution**: AMALIA-VL은 유럽 포르투갈어(pt-PT)를 네이티브로 목표로 한 최초의 완전 개방형 instruction-tuned LVLM으로, 고해상도 입력을 위한 dynamic image tiling과 pt-PT 최적화 언어 모델을 learned connector로 결합합니다. 또한 pt-PT 중심 3단계 학습(vision-language alignment→일반 시각 instruction tuning→preference optimization)과, pt-PT 멀티모달 자원이 거의 없는 공백을 메우는 맞춤형 데이터 믹스를 제공합니다.

- **Technical Challenges**: 핵심 난제는 (1) pt-PT가 pt-BR 편향의 영향을 받아 방언 간 누수가 발생하고, (2) 공개 라이선스 기반 pt-PT 멀티모달 데이터·벤치마크가 사실상 부족하며, (3) 고해상도 시각 입력과 긴 pt-PT 출력 형식 준수를 동시에 만족시켜야 한다는 점입니다. 논문은 pt-BR와의 “교차 방언 누수”를 줄이기 위해 pt-PT 데이터 필터링/번역과 함께 3단계 학습을 설계하고, OCR·문서·차트·코드 등 pt-PT에 특화된 합성 데이터로 행동 단위(포맷·정밀 추출 등)를 보강했으며, 선호 최적화(DPO)에는 공개 preference 데이터 부재 문제를 합성 라벨링으로 대응했습니다.

- **Empirical Impact**: pt-PT로 14개 모델 대비를 수행한 결과, AMALIA-VL은 captioning, spatial grounding, OCR 등 세부 텍스트 이해와 정밀 시각-언어 정렬이 중요한 범주에서 특히 경쟁력이 높아 “오픈소스 pt-PT” 강력한 기준선(baseline)을 제시합니다. 반면 General VQA와 복잡한 시각 수학 등은 성능 변동이 크고, 이는 폭넓은 세계지식/추론 데이터의 라이선스 가능성 한계와 pt-PT 장문 추론 데이터 부족이 원인으로 분석됩니다. 저자들은 모델 가중치와 학습 코드·데이터, 그리고 pt-PT로 번역한 평가 벤치마크까지 공개해 향후 pt-PT LVLM 개발의 진입장벽을 낮추는 데 의미가 큽니다.



### DVANet: Degradation-aware Visual-prior Alignment Network for Image Restoration (https://arxiv.org/abs/2606.19097)
Comments:
          All-in-One Image Restoration; Deep Unfolding; Degradation Representation; Visual Prior

- **Prior Approaches**: 기존 All-in-One image restoration(AiOIR) 연구는 다양한 열화에 대응하기 위해 end-to-end 블랙박스 매핑을 학습하는 경우가 많아, 관측 일치(observation consistency)나 영상 우도(prior) 같은 최적화 해석이 약하다는 한계가 있었다. deep unfolding 기반 접근도 적지 않지만, 고정된 열화 가정이나 사전 주어진 열화 정보에 의존해 복합 열화와 국소 손상(local damaged content) 상황에서 적응성이 떨어진다. 또한 단일 복원 네트워크가 구조적 디테일을 잃은 영역을 복구할 때 계층적 시각 우도(hierarchical visual prior)를 충분히 활용하지 못한다는 문제가 지적된다.

- **Core Contribution**: 이 논문은 HQS(half-quadratic splitting)에서 영감을 받은 deep unfolding 네트워크 DVANet을 제안해, 복합 열화 하에서 unified image restoration을 “열화 인지 관측 일치”와 “시각 우도 유도 복원”의 협력적 unfolding 과정으로 재구성한다. 열화-aware 관측 일치 분기에서는 입력으로부터 열화 상태를 표현하고 그에 맞춰 data mapping을 조건화해 적응성을 높인다. 시각 우도 유도 복원 분기에서는 DINOv3의 계층 정보를 prior로 도입해, 손상된 영역의 구조 디테일 회복을 보강한다.

- **Technical Challenges**: 핵심 난제는 unified 설정에서 실제 열화 연산자(및 그 adjoint)를 명시적으로 구성하기 어렵다는 점이며, DVANet은 이를 고정된 물리 연산 대신 입력 의존적 “열화-conditioned data mapping”으로 대체한다. 이를 위해 DRB(Degradation Representation Block)로 전역(global) 열화 속성과 국소(local) 열화 단서를 함께 뽑아 cross-attention 기반 응답으로 관측 일치 업데이트 방향을 학습한다. 또 DINOv3 우도를 그대로 강하게 주입하면 초기 학습 불안정이나 관측 일치 제약 약화가 생길 수 있어, prior variable 갱신 경로에만 잔차+gated modulation을 적용해 점진적으로 제약을 강화하는 설계를 사용한다.

- **Empirical Impact**: 실험은 단일 열화, 복합 야간 열화, 합성/복합 열화, 그리고 cross-domain 복원(의료·원격탐사 등)까지 다중 시나리오에서 수행되었고, DVANet은 기존 대비 우수하거나 경쟁력 있는 성능을 보여준다. 특히 열화 적응성과 일반화 능력 측면에서 개선이 일관되게 관찰되며, 구조 디테일 복구가 어려운 국소 손상 상황에서도 이득이 드러난다. 결과적으로 DVANet은 AiOIR에서 최적화 해석을 살린 deep unfolding 틀과, foundation model 기반 계층 우도의 결합 가능성을 실증했다는 점에서 의미가 크다.



### PorTEXTO: A European Portuguese Benchmark for Visual Text Extraction (https://arxiv.org/abs/2606.19096)
- **Prior Approaches**: 기존 pt-PT OCR 벤치마크는 주로 역사 유물이나 1800~1900년대 문헌처럼 시대적 철자 변화가 큰 자료에 집중돼, 현대 OCR 수요(사진·스크린샷·업무 문서 등)를 반영하기 어렵다. 일부 다국어 OCR 평가도 포르투갈어가 pt-BR(브라질 포르투갈어) 비중이 높거나 pt-PT 데이터가 짧아, 모델의 pt-PT 일반화 능력을 공정하게 검증하기 힘들다.

- **Core Contribution**: 이 논문은 현대 유럽 포르투갈어(pt-PT) 시각 텍스트 추출을 목표로 한 첫 벤치마크 PorTEXTO를 제안한다. 필기(학생 노트), in-the-wild(거리 표지·포스터·메뉴·스크린샷·시험/양식), 합성(배경 위에 pt-PT 텍스트 렌더링)으로 구성해 문화적으로도 현실적인 사용 시나리오를 폭넓게 포함한다.

- **Technical Challenges**: 핵심 난제는 (1) pt-PT 고유의 맞춤법·약어·현대 문체를 반영한 라벨 품질을 확보하는 것과 (2) 합성 대비 실제 입력에서 성능 하락 원인을 정확히 드러내는 평가 설계다. 이를 위해 Gemini 3.1 Pro Preview 같은 frontier LVLM으로 전사한 뒤, 포르투갈어 원어민이 단계별로 exhaustive review 및 수정/폐기를 수행하는 파이프라인을 구축하고, LVLM 기반 지역/전체 크롭 분할(Handwritten region/full page)까지 포함해 세분화된 난이도를 만든다.

- **Empirical Impact**: 실험 결과 대부분 모델이 합성에서 강하지만, in-the-wild와 필기에서 큰 성능 하락을 보였고 특히 OCR 전용 모델은 필기에서 더 크게 무너진다. 또한 pt-PT 성능을 끌어올리는 요인으로는 모델 크기나 해상도 예산보다 pt-PT 중심의 전문 다국어 데이터가 더 중요하다는 신호가 관측돼, 공개 pt-PT OCR 자원과 학습 데이터의 부재가 오픈 웨이트/오픈 데이터 LVLM의 한계를 만든다는 문제의식으로 이어진다.



### Taming I2V models for Image HOI Editing: A Cognitive Benchmark and Agentic Self-Correcting Framework (https://arxiv.org/abs/2606.19073)
- **Prior Approaches**: 기존 instruction-based image editing은 주로 정적 속성(색, 질감, 배치 일부) 중심으로 발전했지만, HOI editing은 인간-물체의 ‘동적 관계’를 동사(verb) 기반으로 재구성해야 해 더 어렵습니다. 또한 평가에서도 CLIP-score 같은 global metric 또는 독립 개체 검증에 의존해, “상호작용이 맞는지”와 “대상 쌍의 동일성/문맥이 유지되는지”를 동시에 판별하기 힘들다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 HOI-Edit라는 첫 전용 벤치마크를 제안해, HOI 편집을 L1(기초 동작/정적→동적), L2(공간 문맥), L3(인과·물리 추론)의 3단계 인지 수준으로 평가합니다. 더불어 HOI-Eval은 사람-물체를 쌍 단위(쌍별 bounding box)로 고정한 뒤 VLM Q&A로 instance-level 상호작용 타당성과 합리성을 점검하는 grounded 평가 프로토콜입니다. 마지막으로 I2V 기반 동적 생성의 장점을 살려 SCPE(Self-Correcting Process Editing)라는 에이전틱 자기교정 프레임워크를 제시합니다.

- **Technical Challenges**: HOI 편집은 장면에 이미 존재하는 인간/물체의 ID를 유지한 채, 동사로 표현된 ‘관계’를 문맥적으로 논리 있게 바꿔야 해서 모델이 임의의 대상을 만들어내는 것을 막는 제약이 큽니다. 이를 해결하기 위해 HOI-Eval은 쌍별 영역 추적으로 대상 연관을 고정하고, 상호작용 점수(I)와 맥락 질문 정답 여부를 결합한 I+Q&A로 ‘관계의 발생/정확성’과 ‘공간·절차·물리적 합리성’을 동시에 제약합니다. SCPE는 Playbook에 실패 패턴을 축적하고, 비디오 샘플링 후 에러 리포트를 생성·반영해 프롬프트를 반복 정제함으로써 I2V의 temporal generation을 실제로 HOI에 맞게 “과정 수준”에서 교정하도록 설계됩니다.

- **Empirical Impact**: 실험에서 SCPE는 HOI-Edit 전 인지 수준에서 상호작용 정밀도와 동일성 보존이 강한 SOTA급 성능을 보였고, 특히 Nano Banana 같은 상용 SOTA보다 interaction 점수에서 우위를 보였습니다. 또한 HOI-Eval의 자동 지표는 사람 평가와의 상관(예: Pearson 0.60) 및 HICO-DET 기반 판정 정확도(예: 98.5%)로 신뢰성을 입증해, 기존 global metric의 구조적 미스매치를 줄일 가능성을 보여줍니다. 무엇보다 I2V의 “failure process replay”를 자기교정에 연결함으로써 HOI 편집을 단순 시각 유사성 경쟁이 아니라 원인 기반 디버깅·개선 문제로 전환하는 의미가 큽니다.



### DREAM: Extending Vision-Language Models with Dual-Objective Encoding for Cross-Modal Retrieva (https://arxiv.org/abs/2606.19062)
- **Prior Approaches**: 기존 텍스트-비디오 검색은 handcrafted 특징 기반 또는 얕은 크로스모달 정렬에서 출발해, 최근에는 CLIP4Clip류의 contrastive 사전학습으로 성능을 끌어올렸습니다. 다만 많은 모델이 평평한(flat) 또는 균일한 attention/프레임 평균 풀링에 의존해 fine-grained한 시간 의존성과 문장 내 국소-전역 언어 구조를 함께 정교하게 다루기 어렵다는 한계가 있었습니다. 언어 측도 MLM 또는 contrastive 중심의 단일 패러다임에 머무르는 경우가 많아, 지역 토큰 의존성과 문장 전체 구조를 동시에 커버하기 힘들었습니다.

- **Core Contribution**: DREAM은 Dual-path Representation Enhancement and Alignment Model로, 언어에는 MLM과 PLM(순열 기반 언어 모델링)을 결합하는 hybrid 텍스트 인코딩을 제안합니다. 비전에는 계층형 비전 인코더와 Cascaded Group Attention(CGAT)을 넣어, 거친-정밀(coarse-to-fine) 방식으로 공간·시간 정보를 단계적으로 정제하고 전역 임베딩과의 정렬을 돕습니다. 즉, “국소/전역 언어 이해”와 “계층형 시공간 표현 학습”을 동시에 강화해 복잡한 질의-동적 영상 매칭을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 (1) 문장 내 국소 문법/토큰 관계와 문장 전체의 전역 구조를 누설 없이 함께 학습하면서, (2) 영상의 미세한 시간 전이와 다단계 행동을 계층적으로 포착하되 계산 효율을 유지하는 것입니다. DREAM은 텍스트에서 MLM은 마스킹으로 로컬 문맥을, PLM은 순열된 시퀀스를 dual-stream attention으로(콘텐츠 스트림 vs 쿼리 스트림) 구성해 전역 의존성을 학습하되 정보 누설을 제한합니다. 비전에서는 TokenInteract로 경량 토큰 간 정제를 먼저 수행하고, CGAT으로 윈도우 기반 그룹 어텐션을 하향식/상향식 흐름처럼 재귀적으로 융합해 다중 스케일 시공간 단서를 단계별로 강화합니다.

- **Empirical Impact**: MSR-VTT, MSVD, LSMDC에서 R@1 기준 49.4%, 49.7%, 27.3%로 새로운 state-of-the-art를 달성했으며, 관련성 높은 영상을 상위 랭크에 더 자주 배치하는 성과를 보였습니다. 정성 분석에서도 프레임 전반에 걸친 attention의 일관성을 유지하면서 동적인 질의 내용과의 정렬이 잘 되는 모습이 관찰됩니다. 결과적으로 DREAM은 텍스트-비디오 검색에서 계층형 attention과 이중 목적(MLM+PLM) 텍스트 모델링 조합이 맥락 인지형(semantic/context-aware) 정렬을 강화한다는 실증적 근거를 제공했습니다.



### Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks: From Evaluation to Diagnosis (https://arxiv.org/abs/2606.19053)
- **Prior Approaches**: 기존 LVLM 평가는 LVLM-eHub, MMBench처럼 전반적(홀리스틱) 성능을 보거나 DocVQA, GQA 같은 특정 태스크 중심으로 이뤄지는 경우가 많았습니다. 파인그레인 분류를 일부 다루는 연구도 있었지만, 도메인 다양성·문항 다양성·진단 깊이가 제한적이어서 실패 원인을 세분화하기 어려웠습니다. 그 결과 파인그레인 비전 과제에서 LVLM이 어디에서 무너지는지(시각 표현 부족 vs 시맨틱 접지 약화 vs 지식 부족)를 분리해 설명하기 힘든 상황입니다.

- **Core Contribution**: 이 논문은 FG-BMK라는 파인그레인 평가 벤치마크를 제안하며, 101만 문항과 28만 이미지를 13개 파인그레인 데이터셋(일반 객체 중심부터 원격탐지 같은 전문 도메인까지)으로 구성합니다. FG-BMK는 단일 분류 정확도 대신 human-oriented(대화형 의미 인식)과 machine-oriented(시각 특징 판별력: retrieval·classification) 두 관점을 함께 측정해 실패 원인을 진단하도록 설계됐습니다. 이를 통해 LVLM 실패가 시각 표현의 부족, visual-to-semantic grounding 약함, 파인그레인 카테고리 지식 제한 중 무엇에 더 가까운지 구분할 수 있게 합니다.

- **Technical Challenges**: 파인그레인 평가는 “정답 맞힘”만으로는 원인을 알 수 없기 때문에, 언어 프롬프트·템플릿 편향과 생성형 모델의 언어 우선 편향을 분리해 평가하려는 실험 설계가 핵심 난점입니다. 논문은 속성 인식, 계층 단위(coarse→fine) 인식, 지식 편향(카테고리별 성능 편차), 그리고 feature-level mAP/Top-1 측정을 결합해 진단 신호를 만듭니다. 또한 질문 템플릿을 10종으로 확장해 결과가 템플릿 아티팩트에 의해 좌우되지 않음을 보였고, 훈련 설계/정렬(granularity mismatch 포함)/섬세한 perturbation(시각·언어)을 체계적으로 분석합니다.

- **Empirical Impact**: 다양한 대표 LVLM/VLM을 실험한 결과, 현 LVLM은 파인그레인 인식자(fine-grained recognizer)로서 아직 부족하며 시각 표현·시맨틱 접지·모달리티 정렬·카테고리 레벨 지식이 얽힌 병목이 나타난다고 결론냅니다. 특히 contrastive 학습은 파인그레인 판별력을 높이는 반면, generative/reconstruction 중심 패러다임은 약한 경향이 있고, visual-text granularity가 맞지 않으면 정렬이 판별력을 해칠 수 있습니다. 언어 쪽 perturbation이 시각 증거를 더 쉽게 덮어쓰는 등 강건성에서도 파인그레인 과제 특유의 취약성이 드러나며, 논문은 향후 데이터 구성과 모델 설계에 대한 실질적 가이드를 제공합니다. 



### Low-Rank Tensor Completion Based on Fractional Regularization with Ky Fan p-k Norm (https://arxiv.org/abs/2606.19046)
- **Prior Approaches**: LRTC는 불완전 관측에서 저랭크 텐서를 복원하는 역문제로, tubal rank를 직접 최적화하기엔 불연속성과 비볼록성 때문에 계산이 어렵다. 기존 대표 완화로는 convex인 TNN이 있으나 모든 특이값을 동일하게 벌점해 지배 성분을 과도하게 억제해 추정이 편향될 수 있다. 이를 보완하려고 PSTNN, W-t-TNN, IR-t-TNN 같은 nonconvex surrogate와 difference-based(예: nuclear-minus-Frobenius) 및 fractional-based(예: nuclear-to-Frobenius ratio) 계열이 제안됐지만, 기존 fractional 기반은 분모가 전 스펙트럼을 고정 방식으로 집계해 스펙트럼 분포에 대한 적응성이 제한된다는 지적이 나온다.

- **Core Contribution**: 이 논문은 t-SVD(t-product) 기반 tubal rank에 대한 새로운 nonconvex fractional surrogate TNPK를 제안한다. TNPK는 tensor nuclear norm을 tensor Ky Fan p-k norm으로 나눈 비율 형태로, p와 k로 스펙트럼 선택성을 조절할 수 있어 저랭크 스펙트럼을 더 정확히 근사하도록 설계됐다. 또한 특정 p,k 설정에서는 TNK(분자/분모가 TN과 tensor Ky Fan k norm 비율)나 TNF(분자/분모가 tensor nuclear norm과 tensor Frobenius norm 비율)로 환원되어 기존 fractional regularizer들을 포함하는 확장성도 확보한다.

- **Technical Challenges**: 핵심 난점은 (1) tubal rank에 가까운 비볼록 비율 정규화를 LRTC에 결합하면서 (2) 이 정규화의 수렴성과 근사 보장(예: 좋은 local minimum 보장)을 수학적으로 확보하는 것이다. 논문은 tensor null space property(NSP) 하에서 TNPK 정규화 모델의 저랭크 텐서가 local minimizer임을 증명하고, 이어서 tensor Ky Fan p-k inverse-norm에 대한 proximal operator를 도출해 ADMM을 효율적으로 구성한다. 또한 모든 변수 업데이트가 폐형식(closed-form)으로 가능하도록 설계하고, mild conditions 하에서 subsequence convergence가 보장됨을 보인다.

- **Empirical Impact**: 합성 데이터와 실제 데이터 실험에서 TNPK 기반 방법이 state-of-the-art 경쟁 기법들보다 복원 성능이 우수함을 광범위하게 검증한다. 이는 TNPK의 p,k 선택이 특이값 스펙트럼의 지배 성분을 더 적절히 강조(또는 덜 과벌점)하도록 하여, 기존 TNN의 지배 성분 과억제 및 기존 fractional의 고정 집계 한계를 동시에 완화했음을 시사한다. 결과적으로 tubal rank 기반 LRTC에서 scale invariance와 스펙트럼 적응성을 함께 제공하는 새로운 실용적 nonconvex 정규화 옵션을 제시했다는 점에서 의미가 크다.



### FlowObject: Flow Steering for Bridging Generative Priors and Reconstruction Fidelity (https://arxiv.org/abs/2606.19019)
Comments:
          Project page: this https URL

- **Prior Approaches**: sparse-view에서 완전한 3D를 복원하려는 기존 최적화 기반 방법(예: 3D Gaussian Splatting, NeRF)은 관측된 표면의 시점-일치(photometric consistency)에 강하게 의존해, 가려진(미관측) 영역의 형상·외관을 “그럴듯하게 채우는” 추론이 구조적으로 어렵습니다. 반면 Flow-Matching 같은 3D 생성 모델은 텍스처가 있는 완성 자산 합성이 가능하지만, 학습된 사전(prior)이 관측 evidence를 덮어버리는 synthetic bias와 인스턴스 정합성 부족 문제가 자주 지적됩니다.

- **Core Contribution**: 이 논문은 FlowObject로, sparse-view 3D 복원을 training-free의 guided inverse problem으로 재구성해 생성 사전이 미관측 영역을 완성하되 실제 관측에는 엄격히 고정되도록 만듭니다. Flow-matching 모델의 ODE 궤적을 “데이터 매니폴드(생성 사전)”와 “인스턴스 매니폴드(실제 관측)”가 만나는 지점으로 유도해, zero-shot 복원 패러다임을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 생성 사전과 관측 제약 사이의 긴장을 어떻게 안정적으로 동시에 만족시키느냐입니다. FlowObject는 듀얼 스페이스 guidance로 (1) 관측에서 추출된 특징에 따른 잠재공간 정렬(implicit latent alignment)과 (2) 관측 공간에서의 측정-일치(explicit observation consistency)를 함께 backprop하여, 매 샘플링 스텝마다 ODE trajectory를 조절합니다; 이후 3DGS refinement 단계를 추가해 생성 결과의 “synthetic-looking” 외관을 광현실적인 렌더링 품질로 끌어올립니다.

- **Empirical Impact**: 합성·실세계 벤치마크에서 기존 생성 모델과 최적화 기반 방법은 기하학적 완전성(occluded completion)과 관측 정합성(instance faithfulness)을 동시에 달성하기 어렵다는 점이 드러났고, 특히 severe occlusion 조건에서 격차가 커졌습니다. FlowObject는 geometric completeness와 시점 의존 외관(view-dependent appearance fidelity) 모두에서 SOTA를 유의미하게 능가해, 실사용 관점의 “완전성+정합성 동시 확보”를 보여준다는 점에서 3D 재구성·3D 생성 결합 연구에 의미 있는 진전을 제공합니다.



### Show, Don't Ask: Generative Visual Disambiguation for Composed Image Retrieval with Turn-Valid Coverag (https://arxiv.org/abs/2606.18992)
- **Prior Approaches**: CIR(composed image retrieval)은 기준 이미지와 텍스트 편집 지시로 후보를 찾지만, 실제로는 단일 타깃이 아니라 여러 가능성이 남는 underdetermined 문제를 겪는다. 기존 conformal prediction 기반 접근은 1회 상호작용에서만 coverage 보장을 주로 유지하며, 그 이후에는 질문 선택과 답변 반영이 만드는 feedback covariate shift 때문에 보장이 깨질 수 있다. 또한 텍스트 질문은 viewpoint·속성처럼 미세한 시각 차이를 충분히 해소하기 어렵고, ‘모델이 사용자에게 질문하고 모델이 답을 예측’하는 순환 구조가 생긴다는 한계가 지적된다.

- **Core Contribution**: CLARA는 모호성을 텍스트로 묻지 않고, 후보 집합의 모드를 작은 시각 패널로 ‘보여주기’로 전환해 사용자가 최대로 가까운 이미지를 고르게 한다. 동시에 conformal prediction의 보장을 다중 라운드까지 유지하기 위해, 사용자의 선택이 유도하는 selection(피드백) 분포 변화를 likelihood ratio로 반영해 weighted conformal calibration을 재가중한다. 결과적으로 단일 라운드 SOTA 성능은 유지하면서도, 여러 라운드에서도 명목 coverage를 유지하는 turn-valid 보장 프레임을 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 라운드가 진행되며 테스트 분포가 모델의 이전 질문/패널 선택에 의해 달라져 exchangeability가 깨지는 문제와, (2) 텍스트 채널이 미세 시각 차이를 분해하기 어려운 문제였다. CLARA는 매 라운드를 명시적 feedback covariate shift로 보고, selection-induced likelihood ratio로 weighted split conformal의 임계값을 수정해 ‘라운드별 valid coverage’를 수학적으로 보장한다. 또한 패널에서 생성 이미지만으로 coverage가 부풀려지는 것을 막기 위해, 렌더링 프로토타입은 real corpus 이미지의 모드 대표(예: medoid)로 snap-to-corpus 처리해 보장 대상이 되는 후보 집합이 생성 결과에 의해 변하지 않게 설계했다.

- **Empirical Impact**: 오픈 도메인과 패션 벤치마크에서 CLARA는 단일 턴 기준으로 단일-턴 SOTA급 검색 성능에 도달하고, 다중 라운드에서도 nominal coverage를 유지한다. 더 나아가 강한 텍스트 질문 기반 베이스라인보다 의도한 타깃을 더 적은 라운드에서 찾는 것으로 보고되며, 특히 viewpoint나 fine-grained attribute처럼 시각 모호성이 큰 경우 텍스트 질문보다 시각 패널이 더 효과적임이 강조된다. 즉, ‘coverage 보장 붕괴’와 ‘질문 채널 한계’를 동시에 다루는 상호작용 CIR의 실전형 청사진으로 의미가 크다.



### Visual-OPSD: Cross-Modal On-Policy Self-Distillation for Efficient Unified Multimodal Reasoning (https://arxiv.org/abs/2606.18974)
- **Prior Approaches**: Unified multimodal model(UMM)은 텍스트 추론 중간에 diffusion으로 생성한 visual thought(VT)를 끼워 넣는 interleaved visual chain-of-thought로 공간 과제를 개선해왔다. ThinkMorph는 이 방식이 text-only보다 일관되게 성능을 올린다고 보고했지만, VT 생성은 추론 시마다 약 50스텝 diffusion을 요구해 비용이 매우 크다. 그런데 VT의 ‘픽셀’이 실제로 정보의 핵심인지, 아니면 생성 경로에서 생긴 내부 표현이 도움이 되는지에 대한 검증은 부족했다.

- **Core Contribution**: 이 논문은 VT 픽셀이 로드-베어링(load-bearing)인지 실험적으로 따져보며, VT를 제거하거나 Gaussian noise로 대체해도 정확도가 거의 유지된다는 결과를 제시한다. 동시에 KL 진단으로는 ‘VT가 있는 조건’과 ‘VT가 없는 조건’ 사이에 다음 토큰 분포가 유의미하게 달라짐을 보이며, 즉 생성 경로는 추론에 유용한 신호를 내부적으로 인코딩하지만 화면에 렌더된 픽셀만으로는 그 효과가 설명되지 않는다고 주장한다. 이를 바탕으로 Visual On-Policy Self-Distillation(Visual-OPSD)을 제안해 VT-조건부 teacher의 생성 경로 지식을 텍스트-only student로 JSD 기반 on-policy distillation으로 옮긴다.

- **Technical Challenges**: 핵심 난제는 ‘VT는 필요 없어 보이는데도 분포는 바뀐다’는 비대칭을 실제 학습 목표로 설계해야 한다는 점이다. Visual-OPSD는 teacher와 student가 동일 가중치를 공유하되, teacher는 privileged VT trace를 컨텍스트로 주고 student는 문제 이미지와 질문만 주도록 하여 completion 토큰은 동일하게 고정한 채, 학생의 on-policy 궤적에서 token-level JSD를 최소화한다. 또한 학습 신호가 스타일/잡음 토큰에 치우치는 문제를 top-K(256) 제한과 token clip(0.05)로 완화하고, 학생이 생성 모드로 붕괴하지 않도록 샘플링 중 <image_start>를 제어해 텍스트-only 추론을 유지한다.

- **Empirical Impact**: ThinkMorph의 9개 벤치마크에서 Visual-OPSD는 생성형 teacher를 평균 +3.40pp 개선하면서도 VT 생성 없이 추론해 샘플당 지연을 14.3×(10.0s vs 142.8s) 줄였다. VT가 아닌 Gaussian noise를 넣는 대조군은 +0.40pp 수준에 그친 반면, 실제 의미 있는 VT를 사용하면 +10.28pp로 큰 향상이 나타나 ‘정규화 효과’가 아니라 생성 경로의 의미적 콘텐츠가 이득의 원천임을 뒷받침한다. 특히 공간 추론(VSP)에서 same-scale VLM 대비 큰 격차(+63.83pp)를 보이며, 생성-이해 경로 간 정보 비대칭이 있을 때 OPSD로 그 지식을 효율적으로 이전할 수 있다는 일반 원리를 제시한다.



### Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation (https://arxiv.org/abs/2606.18960)
- **Prior Approaches**: Action-conditioned world models는 로봇의 행동을 조건으로 영상 롤아웃을 생성해, 비용이 큰 실세계 실험을 줄이려는 접근이다. 하지만 manipulation에서는 wrist-camera의 빠른 운동과 end-effector occlusion 때문에 현재 관측만으로는 미래의 시야를 예측하기 어려워 이전 프레임의 장면을 잊거나 hallucination이 생긴다. 기존 메모리 검색은 joint-pose similarity나 단순 FOV overlap, 고정 stride 기반 컨텍스트 확대로는 가시성 제약과 조작 특화한 정보량을 제대로 반영하지 못했다.

- **Core Contribution**: 이 논문은 manipulation에서 장기 일관성을 유지하는 memory-augmented multi-view action-conditioned world model Mem-World를 제안한다. 핵심은 W-VMem으로, 4D wrist-view-centered surfel-indexed memory를 도입해 과거 관측을 ‘언제/어떤 표면 요소’를 봤는지에 고정(anchor)함으로써 geometry-aware history retrieval을 가능하게 한다. 이를 통해 미래 행동이 주어졌을 때, 관련성이 높고 중복이 적은 과거 wrist-view 프레임을 선별해 예측에 활용한다.

- **Technical Challenges**: 기여를 위해서는 (1) 행동으로부터 미래 wrist-camera pose를 계산해 surfel rendering에 쓸 수 있어야 하고, (2) 로봇과 물체가 움직이는 동적 조작 장면에서 surfel을 시간적으로 정의해야 하며, (3) multi-view에서 메모리 초기화·업데이트·리딩의 일관성을 유지해야 한다는 문제가 있다. 논문은 고정된 wrist-엔드이펙터 기구학 변환과 forward kinematics로 미래 wrist-camera pose를 얻고, surfel에 생성/업데이트 timestep과 task-relevance(조작 대상 여부) 플래그를 포함하는 4D surfel 정의를 설계한다. 또한 업데이트는 wrist-view 관측만으로 수행해 temporal association을 보존하고, 미래의 평균 관측 방향에서 surfel을 렌더링해 가시성·과업 관련성·시간 근접성 기반 점수 및 NMS로 top-K 컨텍스트를 선택한다.

- **Empirical Impact**: 실험에서 Mem-World는 end-effector occlusion이 잦고 wrist-camera 운동이 큰 장기 manipulation 시나리오에서 더 persistent하고 시간적으로 일관된 롤아웃을 보였다고 보고한다. 특히 Ctrl-World 대비 실세계 정책 성능과의 Pearson 상관을 14.5% 개선했으며(설정된 5개 task에서 높은 선형 대응), Mem-World로 만든 synthetic data로 post-training을 하면 장기 태스크 success rate가 58%에서 72%로 상승했다. 이는 메모리-증강 world modeling이 단순 예측을 넘어 정책 평가/학습의 ‘데이터 엔진’으로 신뢰도를 높일 수 있음을 시사한다.



### Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos (https://arxiv.org/abs/2606.18955)
Comments:
          Accepted to IROS 2026

- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 사전학습된 Vision-Language Model(VLM) 백본 위에 action head를 붙이고, 대규모 로봇 데이터(정확한 action annotation)로 fine-tuning해 성능을 끌어올리는 흐름이 주류였습니다. 다만 Open X-Embodiment나 AgiBot처럼 플랫폼별로 필요한 데이터 수집 비용과 기구학/물리 차이로 인한 domain gap이 커서 확장성이 떨어집니다. 한편 사람 egocentric 비디오를 활용하려는 시도는 존재하지만, 대부분 AR/VR 등 특수 장치로 hand pose 같은 명시 라벨을 얻어야 하거나(데이터 라벨 병목) 배경·카메라 변화가 섞인 latent action이 잡음으로 작동하는 문제가 남아 있었습니다.

- **Core Contribution**: 이 논문은 action 라벨이 없는 사람 egocentric 비디오만으로 cross-embodiment action prior를 학습하는 latent-action 기반 사전학습 패러다임을 제안합니다. 핵심은 Hybrid Disentangled VQ-VAE가 motion dynamics와 environmental background를 physical mask로 분리해 “몸체에 덜 의존적인” action codebook을 만들고, 이를 VLM의 행동 어휘로 distill해 action intent를 먼저 학습시키는 것입니다. 또한 적응 단계에서는 intention-perception decoupling을 통해 VLM의 intent와 state 피드백을 분리해 action hallucination을 줄이며, 다운스트림은 약 50 trajectories 수준의 소량 데모로도 경쟁력을 확보합니다.

- **Technical Challenges**: 가장 어려운 점은 라벨이 없는 사람 비디오에서 ‘진짜 조작 동학’만 이산 action token으로 안정적으로 뽑아내는 disentanglement입니다. 논문은 DINO v2 고정 특징과 하이브리드 disentangled VQ-VAE(dual vector quantization + mask-guided decoder)로 motion은 foreground에서만 재구성 오차를 강제하고, background는 별도 경로로 학습시켜 코드북이 작업 무관 변화에 덜 흔들리게 했습니다. 다음 난제는 로봇 적응 시 실시간 관측과 intent가 충돌해 생기는 hallucination을 막는 것인데, VLM이 intent를 담당하고 별도 frozen visual encoder(DINO v2)가 state-specific 특징을 제공하는 decoupling으로 제어 안정성을 높였습니다.

- **Empirical Impact**: 실험은 시뮬레이션과 실세계 모두에서 검증되며, 사전학습은 unlabeled human video만 사용하면서도 LIBERO 및 dual-arm RoboTwin 2.0/ARX 플랫폼에서 SOTA VLA와 유사하거나 우수한 성공률을 보였습니다. 특히 long-horizon/Goal 계열에서 villa-x 대비 성능 격차가 크게 나타나, 추출된 latent action intent가 multi-step 계획을 더 잘 안내한다는 해석이 가능합니다. 정량 분석에서는 domain bias를 제거한 뒤 Centered Kernel Alignment( CKA ) 일치도가 UniVLA보다 더 높게 나와, “운동 중심” 코드북이 embodiment/환경 차이를 억제한다는 점을 representation 수준에서 뒷받침했습니다.



### SP-TransientBench: A Real-Captured Single Photon Perception Benchmark (https://arxiv.org/abs/2606.18952)
- **Prior Approaches**: 기존 SPL(single-photon LiDAR) 연구는 time-of-flight 히스토그램의 피크 검출·웨이브폼 피팅, 확률/신호-배경 분리 같은 모델 기반 복원, 또는 히스토그램→깊이/표현을 직접 매핑하는 learning 기반 접근으로 나뉩니다. 그러나 대부분 시뮬레이션이나 소규모 실측에 의존해, 실제 환경의 고유한 잡음과 multi-return 과도(다중 에코) 현상을 함께 다루는 체계적 평가가 부족했습니다.

- **Core Contribution**: 이 논문은 실측 기반 멀티태스크 SPL 벤치마크인 SP-TransientBench(STB)를 제안합니다. STB는 각 뷰에 대해 전체 time-of-flight 히스토그램(멀티리턴 포함), 카메라 포즈 캘리브레이션 메타데이터, 그리고 일부 장면의 13클래스 3D 시맨틱 라벨을 제공해 depth estimation, multi-view 3D reconstruction, 3D semantic segmentation을 동일한 조건에서 비교 가능하게 만듭니다.

- **Technical Challenges**: 핵심 난관은 단일 픽셀 히스토그램이 여러 시간 피크로 구성되며 배경 잡음과 겹침 때문에 “시간 bin↔실제 표면” 매핑이 본질적으로 애매하다는 점입니다. 이를 해결하기 위해 논문은 히스토그램 도메인에서 sequential peak peeling으로 멀티리턴을 bin 단위로 라벨링하는 프레임워크와 전용 툴을 설계했으며, IRF(센서 응답)와 보조 LiDAR 기반 기하 캘리브레이션을 함께 제공해 재현성을 높였습니다.

- **Empirical Impact**: 실험에서는 SPL 히스토그램만으로 depth estimation, TransientNeRF/Transientangelo 계열로 multi-view 복원, 전처리+point cloud backbone 조합으로 3D 시맨틱 분할을 평가하며 성능/한계를 일관된 프로토콜로 비교합니다. 또한 시뮬레이션 사전학습이 일부 데이터 규모에서 이득을 주지만, 실제 데이터가 충분해지면 Scratch 성능이 따라오며 “시뮬레이션의 전이 갭”이 완전히 해소되지는 않음을 보여 STB 같은 실측 벤치마크의 필요성을 실증합니다.



### Physics-IQ Verified (https://arxiv.org/abs/2606.18943)
- **Prior Approaches**: Video generative models(VGMs)는 world modeling 같은 다운스트림으로 확장됐지만, 물리적 현실을 얼마나 이해하는지 평가 체계가 부족했다. 이에 Physics-IQ는 물리 실험 영상과 생성 영상을 비교해 물리 이해를 수치화하려 했으나, 벤치마크 자체의 프롬프트/정답 품질과 점수 집계 방식이 잡음 요인과 극단값 지배 문제를 만들 수 있다는 한계가 드러났다. 또한 원래 Composite 점수는 개별 메트릭 하위점수가 비정상적으로 커지면 나머지 성능을 압도할 수 있어 “단일 메트릭 독립 평가 금지” 설계 의도와 어긋날 여지가 있었다.

- **Core Contribution**: 이 논문은 Physics-IQ를 체계적으로 감사(audit)하고, 측정 신뢰도를 높이기 위한 개선안을 제시한다. 구체적으로 (1) 프롬프트와 ground-truth 품질을 손봐 교란 요인을 줄이고, (2) 샘플 단위 점수 체계를 도입해 각 샘플과 메트릭이 의도대로 동일한 역할을 하도록 Physics-IQ를 재정의한다. 그 결과 새 벤치마크 Physics-IQ Verified는 전체 샘플의 57.6%를 수정하며 프롬프트 품질도 개선(34.8% 향상)했다고 보고한다.

- **Technical Challenges**: 핵심 기술적 난제는 “프롬프트가 시험 문제처럼 정답을 자명하게 만들지 않으면서도, 물리 이해와 무관한 자유도(예: 부정문 해석 실패, 카메라 드리프트)를 최소화”하는 것이다. 저자들은 부정 기반 지시를 긍정 프레이밍으로 전환하고, 카메라 구도/촬영 조건을 고정 CAM field 같은 단일 서술로 통일하며, 발생 시점이 다른 artifact를 프레임 freezing으로 제거해 메트릭 변동을 줄였다. 더불어 원래 점수의 하위점수 과잉 지배를 막기 위해 ceiling 개념을 엄격히 반영하고, per-sample 기반 Physics-IQ Verified 점수(샘플 평균)로 안정성과 해석 가능성을 개선했다.

- **Empirical Impact**: 여러 이미지-to-video 생성 모델 비교에서 Physics-IQ Verified는 순위에 “적당하지만 의미 있는” 변화를 유발했는데, Kendall’s τ=0.46으로 순위 재정렬의 실질적 신호가 관측됐다. 또한 Physics-IQ Verified는 메트릭이 physical variation(두 번의 실험 trial 간 불가피한 변동)이라는 기준선에 과도하게 도달/초과했을 때의 왜곡을 완화해, 물리적으로 타당한 생성 능력을 더 신뢰성 있게 판별하려는 목적을 강화한다. 커뮤니티 관점에서, 보다 안정적인 물리 정확도 평가 신호를 제공함으로써 “물리적으로 맞는 VGM” 개발을 유도할 벤치마크로 자리잡을 가능성을 제시한다.



### BindEdit: Taming Attention Leakage for Precise Multi-Object Image Editing (https://arxiv.org/abs/2606.18906)
Comments:
          Preprint

- **Prior Approaches**: 기존 텍스트-유도 확산 기반 편집은 마스크/영역 또는 cross-attention을 이용해 편집을 유도하지만, 대부분 단일 객체를 전제로 설계되어 다중 객체 동시 편집에는 취약했다. 다중 객체에서는 LoMOE류처럼 객체별로 따로 생성한 뒤 합치는 separate-and-merge 방식이 등장했으나, 경계 아티팩트나 잔존 토큰(소스 의미) 문제, 그리고 객체 수에 비례하는 비효율이 반복됐다. LEDITS++/ParallelEdits 계열도 객체별 신호를 공유하긴 하지만, 핵심인 attention leakage(토큰-영역 엉킴)를 직접 규제하지 못해 동일한 실패 유형이 남는다.

- **Core Contribution**: 이 논문은 다중 객체 편집 실패의 원인을 attention leakage로 정식화하고, 두 유형( Edit-Token Leakage, Source Dominance Leakage )을 구분해 각각을 겨냥한다. 이를 단일 diffusion trajectory 안에서 해결하기 위해 training-free 프레임워크 BindEdit를 제안하며, cross-attention과 self-attention에 attention-level 제약을 걸어 목표 토큰이 해당 공간 마스크에 “정박(bind)”되도록 만든다. 또한 편집 마스크 내부에서 개념이 분절되지 않도록 Region fidelity 항을 추가해 다중 객체에서도 자연스러운 합성을 목표로 한다.

- **Technical Challenges**: 핵심 난관은 (1) 목표 토큰이 올바른 객체 마스크에 집중되지 않아 신원 혼합이 생기고(Edit-Token Leakage), (2) 소스 프롬프트의 잔여 의미가 editable region으로 새어 들어가 목표 의미를 압도하는(Source Dominance Leakage) 현상을 동시에 제어해야 한다는 점이다. BindEdit는 cross-attention에서는 목표 토큰 그룹의 attention concentration ratio를 마스크/토큰 감독으로 최적화해 타깃 영역 바깥 누설을 줄이고, self-attention에서는 인스턴스 간 상호작용을 제한해 자기-혼합을 억제한다. 추가로 editable region에서 source/background 잔여 토큰 영향은 cross-attention re-balancing(대조적 binary cross-entropy)을 통해 약화시키고, Region fidelity(DKL 기반)로 마스크 내부의 attention이 한 모드로 응집하도록 유도한다.

- **Empirical Impact**: 실험에서는 새로 제안한 다중 객체 확장 벤치마크를 포함해 기존 LoMOE-Bench, OIR-Bench 등에서 BindEdit이 단일 diffusion trajectory 방식으로 경쟁/우위를 보이며 특히 객체 수가 늘어날수록 성능 격차가 커짐을 확인했다. 정량적으로 CLIP-image와 CLIP-object에서 전반적 선두를 기록했고(LPIPS는 보존성과 편집 균형 관점에서 경쟁 수준), 사용자 선호도 조사에서도 BindEdit이 다수 구간에서 1순위 선택을 받았다. 결과적으로 “객체 수가 많은 복잡한 편집”에서 발생하던 blended/duplicated/불완전 편집 문제를 attention 제약만으로 완화한다는 점에서, 실용적인 다중 객체 편집 파이프라인의 설계 방향에 의미 있는 근거를 제공한다.



### Automatic ply-specific analyses of CFRP micrographs using shortest-path-based ply distinction (https://arxiv.org/abs/2606.18894)
- **Prior Approaches**: 기존 연구는 마이크로그래프 해상도가 상대적으로 낮을 때, 각 ply의 중심을 밝기 기반으로 추정한 뒤 이웃 중심을 최단거리로 연결해 ply 경계를 근사하는 방식에 집중해왔다. 이런 방법은 휘어지거나 테이퍼된 적층, 고해상도에서 한 픽셀 단위로 fiber를 ply에 배정해야 하는 요구에는 제약이 있다. 또한 ply 간 경계가 아니라 섬유 분포·두께 같은 정량 분석으로 바로 이어지기 어려워, 전통적인 수동 검사 의존도가 남아 있었다.

- **Core Contribution**: 이 논문은 semantic segmentation 마스크를 그래프로 보고, ply를 가르는 경로를 shortest-path(다익스트라)로 계산해 fiber를 각 ply instance에 자동 배정한다. 이를 통해 semantic segmentation과 ply instance segmentation 사이의 간극을 global information으로 메웠다. 그 결과 ply/ interleaf 두께와 로컬 fiber volume fraction 같은 정량 지표를 자동으로 측정하는 워크플로를 제공한다.

- **Technical Challenges**: 고해상도 마이크로그래프에서는 gaps, 휨(waviness), 방향 전환, 조명 변화, 균열 같은 잡음 요인이 많아 ply 구분 경로가 쉽게 흔들린다. 논문은 (1) EDT 기반 거리 비용(cost) 설계, (2) 왼쪽에서 시작점(start points) 후보를 국소 조건으로 고른 뒤, (3) 다익스트라 경로 비용과 start/end의 bipartite matching으로 전역 최적 조합을 선택하는 다단계 그래프 최적화를 제안한다. 또한 해상도·기울기 처리를 위해 downsampling과 rotation 파라미터를 두고, interleaf가 있는 경우에는 비복잡한 Otsu 기반 이진 마스크로도 성능을 확인한다.

- **Empirical Impact**: 대상 10개 고해상도 CFRP 마이크로그래프에서 ply-separating paths가 전반적으로 올바르게 도출되었고, 특히 interleaf가 있는 샘플에서는 Otsu 이진 마스크로도 거의 동일한 결과를 보였다. 경로 계산은 약 9~38초(마스크 크기/환경에 따라 변동) 수준이며, 픽셀 수에 대한 처리 시간도 비교적 완만하게 증가한다. ply-resolved FVF, interleaf 두께 분포, 국소 ply 두께 편차 같은 후속 분석이 자동화되면서, 제작 공정으로 인한 미세구조 비균질성을 계량화하고 기계적 성능(예: 결함·균열 민감도)과 연결하는 디지털 머티리얼 특성화에 의미 있는 기반을 제공한다.



### DINO-Med3D: Bridging Dimension and Domain Gaps in Volumetric Segmentation via Progressive Adaptation (https://arxiv.org/abs/2606.18886)
Comments:
          Accepted at MICCAI 2026. The camera-ready version and link will be made publicly available upon publication

- **Prior Approaches**: 기존 의료 영상 분할은 CNN을 거쳐 Transformer/Mamba 계열로 발전했지만, 정답 라벨 부족 때문에 성능이 자주 상한에 부딪힙니다. Segment Anything 계열은 zero-shot을 보여주지만 대개 interactive prompt나 auto-prompt 같은 입력 의존성이 큽니다. 한편 self-supervised DINOv3를 의료에 적용한 연구들은 등장했으나, 2D 자연 영상 인코더를 3D 볼륨 예측에 그대로 이식하면 dimension gap(시선축 z-방향 상관 손실)과 domain gap(저대비 경계/질감 의존 특성) 때문에 성능 저하가 발생합니다.

- **Core Contribution**: 이 논문은 2D DINOv3를 3D 의료 분할로 단계적으로 이식하는 DINO-Med3D를 제안합니다. 1단계에서는 multi-slice 기반의 pseudo-3D 임베딩으로 차원 불일치를 줄이고, segmentation proxy task로 자연 영상 표현을 의료 도메인에 맞게 정렬합니다. 2단계에서는 backbone을 고정한 채 3D adapters와 Detail-Recovery 스트림을 더해 전역 inter-slice 연속성과 고주파 경계 단서를 강화합니다.

- **Technical Challenges**: 핵심 난제는 (1) DINOv3의 2D 패치 임베딩/positional embedding을 3D에 그대로 쓰기 어렵고, (2) 단순 슬라이스 처리만으로는 z-축 연속성과 질감 기반 경계 정보를 잃는다는 점입니다. 이를 위해 ICE로 depth-aware 3D convolution 기반 pseudo-3D 패치 임베딩을 만들고, 프록시 과제로 중심 슬라이스 분할을 학습시켜 도메인 정렬을 수행합니다. 이후 3D adapters와 LoRA를 통해 frozen backbone의 표현을 볼륨 추론에 맞게 보강하고, HRSE(고해상도 디테일)와 Adaptive Gated Fusion(AGF)으로 저주파 의미와 고주파 경계를 내용 적응적으로 결합합니다.

- **Empirical Impact**: CT/MRI 5개 공개 데이터셋에서 DINO-Med3D는 nnU-Net, nnFormer, SwinUNETR, U-Mamba 등 SOTA를 전반적으로 능가하며, 특히 DSC와 HD95 모두에서 일관된 개선을 보였습니다. 예를 들어 MSD-Colon에서는 DSC가 62.28%로 nnFormer 대비 큰 폭(11.60%p) 향상을 기록했고, 질감이 중요한 데이터셋에서도 Dino U-Net(Large) 대비 더 높은 정확도를 보였습니다. 또한 ablation에서 proxy task 없이 stage I을 제거하면 DSC가 크게 하락해, 정렬 단계의 중요성과 제안 모듈들의 역할이 실증적으로 확인됩니다.



### LARE: Low-Attention Region Encoding for Text-Image Retrieva (https://arxiv.org/abs/2606.18885)
Comments:
          Accepted at the ICML 2026 Workshop on Efficient Multimodal Question Answering (EMM-QA). Code: this https URL ; Dataset: this https URL

- **Prior Approaches**: 기존 text-to-image retrieval은 CLIP/ALIGN 계열처럼 이미지 전체를 하나의 전역 임베딩으로 요약한 뒤 텍스트와의 유사도를 비교하는 dual-encoder 패러다임이 주류였습니다. 하지만 전역 임베딩은 시각적으로 두드러진 객체나 장면 맥락에 편향되어, 작은/덜 주목되는 영역의 미세 단서가 충분히 반영되지 못하는 문제가 있었습니다. FILIP, RegionCLIP 같은 fine-grained 정렬 방식은 더 촘촘한 대응을 시도하지만, 대체로 추가 학습이나 구조/연산 변화가 필요하다는 한계가 지적됩니다.

- **Core Contribution**: 논문은 LARE(Low-Attention Region Encoding)라는 학습 없는(inference-only) 보강 프레임워크를 제안해, 전역 임베딩에서 놓치기 쉬운 low-attention region의 정보를 임베딩에 함께 반영합니다. 비전 트랜스포머의 내부 self-attention 신호에서 덜 주목된 영역을 골라 그 영역을 다시 인코딩하고, 텍스트와의 유사도를 global+regional 관점에서 비교합니다. 또한 Dense-Set이라는 군집/장기 꼬리(long-tail) 객체에 초점을 둔 평가 서브셋을 COCO와 Flickr30K에서 재구성해 기존 모델의 취약점을 더 엄격히 드러내게 했습니다.

- **Technical Challenges**: 핵심 난제는 ‘덜 주목된 영역’을 신뢰할 수 있게 찾아내고, 그 추가 영역이 잡음이나 오탐을 만들지 않도록 전역 점수와 안정적으로 결합하는 것입니다. LARE는 frozen vision encoder의 attention map에서 inverse-attention(낮은 attention을 받는 패치들을 강조) 기반으로 후보 region을 생성하고, 제한된 개수의 region만 재인코딩해 공유 임베딩 공간에서 비교 가능하게 만듭니다. 결합은 hard maximum 대신 confidence-gated fusion을 써서 global similarity가 충분히 확실할 땐 전역 점수를 그대로 유지하고, 불확실할 때만 regional 점수를 보강하도록 설계했습니다.

- **Empirical Impact**: 실험은 zero-shot 설정에서 CLIP, SigLIP, SigLIP 2 같은 여러 백본에 대해 수행되며, 표준 COCO/Flickr30K에서는 성능 저하가 거의 없고 Dense-Set(군집·희귀 객체)에서 일관된 개선이 나타났다고 보고합니다. 예를 들어 COCO-Dense와 Flickr30K-Dense에서 R@1가 CLIP 기준 +5.18p(약 29%), +6.25p(상대 180%)처럼 큰 폭으로 상승했으며, 이는 정교한 객체 단서가 전역 임베딩에 의해 가려지는 상황에서 LARE가 효과적임을 시사합니다. Dense-Set용 재캡션이 low-attention/과소평가 영역을 언어적으로도 강조해, ‘군집 장면 미세 검색’에서 기존 retrieval 모델의 한계를 더 정확히 측정할 수 있게 만들었다는 점에서 의미가 큽니다.



### Performance Gap Analysis between Latin and Arabic Scripts HTR (https://arxiv.org/abs/2606.18884)
Comments:
          this paper accepted at TIPS workshop ICPR 2026

- **Prior Approaches**: 기존 HTR(손글씨 문자 인식) 연구는 라틴 스크립트 대비 아랍 스크립트에서 성능이 떨어진다는 관찰은 있었지만, 통제된 비교가 부족해 원인 규명이 어려웠습니다. 따라서 모델/학습 설정을 고정했을 때 데이터 성격(자원 수준, 주석 품질, 문자 형태 다양성) 중 무엇이 갭을 만드는지 정량적으로 확인하기 어려웠습니다.

- **Core Contribution**: 이 논문은 아랍·라틴 스크립트 HTR을 단일 CRNN(Convolutional Recurrent Neural Network)로 라인 레벨(line-level) 인식까지 통합 비교해, 9개 데이터셋과 다양한 학습 규모(K={100, 500, 1000, 2000, …, Kfull})에서 갭이 어떻게 변하는지 체계적으로 분석합니다. 또한 주석 오류 정제가 성능에 미치는 영향, 문자 빈도 분포·시각적 변동성 차이가 학습 커버리지에 주는 효과까지 같은 실험 틀에서 다룹니다.

- **Technical Challenges**: 핵심은 스크립트 간 데이터 편차를 통제한 채 성능 격차의 원인을 분리하는 것인데, 이를 위해 통일된 CRNN 구조와 학습량 스윕으로 비교합니다. 또 데이터셋에 포함된 라벨링 오류가 영향을 주는 문제를 정제(cleaning)로 처리해 보되, 격차가 완전히 사라지지 않는다는 점을 통해 단순 주석 문제만으로 설명되지 않음을 보여줍니다.

- **Empirical Impact**: 실험 결과 성능 갭은 저자원 설정에서 크게 나타나다가 데이터가 늘면 줄지만, Kfull에서도 5-7 CER points 수준의 차이는 지속됩니다. 특히 아랍은 문자 빈도 분포가 라틴보다 heavy-tailed(긴 꼬리)하고, 고유 시각 변동성이 커서 동일한 고정 샘플 수는 아랍에서 표현 학습 커버리지가 덜해 더 많은 데이터가 필요하다는 분석이 나옵니다. 에러 분석에서는 아랍 데이터셋의 치환(substitution) 오류 약 30%가 시각적으로 유사한 문자 간 혼동에서 오며, 이는 IAM 같은 라틴 스크립트의 약 15%와 비교해 더 큽니다.



### Test-Time Adaptation in Optical Coherence Tomography Using Trajectory-Aligned Time-Independent Flow (https://arxiv.org/abs/2606.18876)
Comments:
          Accepted in MICCAI

- **Prior Approaches**: OCT처럼 의료 영상 분할은 분포가 조금만 바뀌어도 성능이 크게 흔들리는데, 특히 저가 스캐너는 훈련 데이터에 잘 없던 잡음 특성을 가져 TTA(테스트 타임 적응)가 필수적입니다. 기존 TTA는 추론 중 가중치를 업데이트하거나(TENT, pseudo-label 기반) 손실 설계를 통해 개선을 노렸지만, 분할 태스크에서 일관된 향상이 어렵고 재현성도 떨어진다는 지적이 있습니다. 생성모델 기반 접근은 테스트 이미지를 학습 데이터 매니폴드로 투영해 주지만, 실제 잡음이 “고정된 크기의 Gaussian 잡음” 가정에서 벗어나면 픽셀 분포 불일치로 복원이 흔들리는 문제가 남아 있었습니다.

- **Core Contribution**: 이 논문은 flow matching 기반 TTA-Flow를 제안하며, 저가 OCT의 잡음이 만든 분포 격차를 “히스토그램 정렬(histogram matching)”로 먼저 메워서 고품질 surrogate 이미지를 생성합니다. 또한 실제 잡음이 이론적 diffusion(또는 flow) 궤적의 특정 시간에 정확히 대응하지 않는 점을 고려해, 네트워크의 time conditioning(시간 조건)을 제거해 잡음 수준 편차에 더 둔감하게 만듭니다. 결과적으로 분할에 바로 쓰일 수 있는 복원 이미지를 안정적으로 만들어 주는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 테스트 입력의 픽셀/강도 분포가 학습 시뮬레이션 분포와 달라 denoising(또는 flow integration) 과정에서 불일치가 누적된다는 점, (2) 저가 장비의 잡음 수준이 고정된 모델 가정과 달라 시간 조건을 걸면 오히려 잘못된 가정을 강제할 수 있다는 점입니다. 저자들은 이를 위해 synthetic reference trajectories에서 평균 히스토그램을 뽑아 목표 시간 starget에 해당하는 기준 히스토그램으로 입력을 histogram matching 전처리하고, flow 네트워크에서는 time conditioning을 빼서 잡음 상태를 암묵적으로 추정하도록 설계했습니다. 또한 복원은 probability-flow ODE를 적분해 생성하며, 추가 data-fidelity 항 없이도(하이퍼파라미터 민감성 우려) 강건한 분할 성능을 달성하도록 구성했습니다.

- **Empirical Impact**: 실험은 두 종류 저가 OCT 장치(Cirrus, Topcon)에서 더 높은 SNR의 Spectralis로 적응하는 설정에서 진행되었고, 2D/3D-기반 분할의 DSC가 기준선과 비교해 크게 개선됐습니다. Cirrus→Spectralis에서는 unconditional 방식이 평균 DSC 58.6으로 SOTA를 달성하며, supervised upper bound에 근접한 수준까지 올라섭니다. ablation에서는 histogram matching 모듈이 plain flow matching 대비 성능을 크게 끌어올렸고, time conditioning 제거가 전반적 fluid 분할 성능과 지각 품질(FID)을 동시에 개선한 점이 확인됐습니다. 저자들은 CPDM 등 data-fidelity 항을 명시적으로 쓰는 경쟁 방법들을 포함해 여러 베이스라인을 능가하며, 생성 기반 TTA가 의료 분할의 “장비-의존적 분포 변화” 문제를 실용적으로 완화할 수 있음을 보여주었습니다.



### Bridging Single Distortion Artifacts and Mmultifactorial Clinical Quality: Few-shot Biparametric MRI Quality Assessment via Distortion-trained Prototypical Networks (https://arxiv.org/abs/2606.18872)
- **Prior Approaches**: 기존 전립선 multi-parametric MRI 품질 평가는 PI-QUAL을 중심으로 하되, 주관적 판독으로 인해 시간 소요와 관찰자 간 변이가 크다. 연구적으로는 few-shot 학습이나 데이터 증강, k-space 시뮬레이션 등이 시도됐지만, 저품질 DWI는 왜곡(distortion)으로 쏠려 있고 클래스 불균형이 심해 다른 품질 이슈를 충분히 학습하기 어렵다. 또한 왜곡의 복잡한 기하학적 변형을 증강으로 현실적으로 재현하기가 어려워 과적합 위험이 남는다.

- **Core Contribution**: 이 논문은 PI-QUAL 같은 임상 다요인 품질 점수를 자동화하기 위해, 왜곡 라벨만으로 meta-training한 few-shot biparametric prototypical network를 제안한다. T2-weighted(T2WI)와 DWI를 dual-branch로 함께 처리해 DWI의 변형이 ‘진짜 형태’인지 ‘기하학적 왜곡’인지 구분할 수 있게 한다. 더불어 b-value 조건 차이로 인한 취득 편향을 억제하면서, 각 태스크에서 프로토타입을 5개 샘플만으로 재구성해 PI-QUAL(≤4 vs ≥4) 같은 복잡한 점수로 전이되도록 설계했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 저품질 표본의 희소성(class imbalance)과 (2) 저품질이 대부분 distortion으로 구성되어 다른 품질 문제 예제가 부족하다는 ‘dual-scarcity’다. 이를 위해 먼저 distortion vs undistorted 라벨이 있는 대규모 데이터로 2-way episodic prototypical meta-learning을 수행하고, meta-testing/적응 단계에서는 지표에 맞게 프로토타입을 다시 계산하는 방식으로 전이를 구현한다. 또한 FiLM으로 b-value에 따른 feature 변화를 조건부로 보정하고, GRL로 b-value 판별 성능을 떨어뜨려 취득 메타데이터 편향을 억제했다.

- **Empirical Impact**: 실험은 두 데이터셋에서 수행됐으며, 왜곡 평가(in-domain/ out-of-domain)에서 제안 방법이 높은 Sen@80Spe 등 지표를 중심으로 기존 few-shot 및 supervised 기준선보다 우수했다. 특히 PRIME에서 domain shift와 심각한 불균형 상황에서도 B-ACC·Sen@80Spe·Spe@80Sen이 개선되어, 놓치는(false negative) 위험과 잘못 거절(false positive) 위험을 함께 완화하는 점이 강조된다. 마지막으로 PI-QUAL v1 태스크로의 cross-task 적용에서도 PI-QUAL≥4 vs <4 분류 성능이 크게 향상됐고, 저품질 표본이 매우 적은 현실 제약에서도 5-shot 적응이 실용적으로 작동함을 보여준다.



### Learning to Distort: Weakly-Supervised Image Quality Transfer for Prostate DWI Correction (https://arxiv.org/abs/2606.18869)
- **Prior Approaches**: 단일 샷 echo-planar prostate DWI는 직장 공기와 환자 움직임으로 인한 susceptibility artifacts가 기하학적 왜곡을 만들며, T2WI와의 공간 정합이 무너지면서 진단(PI-RADS, Gleason score)에 악영향을 줍니다. 기존 해결은 TopUp 같은 추가 스캔/쌍(pair) 이미지에 의존하거나, fully-supervised IQT는 왜곡-무왜곡 임상 쌍 데이터가 필요하고, unpaired 방법(CycleGAN, diffusion 기반 등)은 학습 불안정·추론 지연 또는 해부학적 환각 위험이 있습니다. 또한 복원 방향(correctior)으로 바로 가는 접근은 실제 왜곡의 “강도/형상”을 충분히 재현하지 못해 downstream 성능이 제한되는 문제가 지적됩니다.

- **Core Contribution**: 이 논문은 임상에서 쌍을 구하기 어려운 상황을 전제로, undistorted→distorted “품질 전이”를 weakly-supervised image quality transfer(IQT)로 먼저 학습합니다. 핵심은 왜곡 관련 feature 공간에서 undistorted/distorted의 잠재 quality prototype을 만들고, IQA 신호로 generative trajectory를 왜곡 쪽 프로토타입으로 끌어가 실제 같은 기하학적 왜곡(자기장 민감도에 의한 변형)을 합성하는 것입니다. 이후 합성된 realistic paired 데이터를 다시 써서 distorted→undistorted correction을 supervised IQT로 학습해, 실제 데이터 왜곡 보정 성능을 높입니다.

- **Technical Challenges**: 첫째, 임상에서는 왜곡-무왜곡 paired가 없어 “왜곡 생성”을 직접 역문제로 풀기 어렵습니다. 둘째, unpaired 상태에서 단순히 target 도메인 분포에 끌려가면 평균적/보수적 표현으로 수렴하거나 블러/비현실적 변형이 생겨 실제 진단 방해를 모사하지 못합니다. 저자들은 prototype flow matching(PFM)에서 FM 궤적을 OT 결합 기반으로 생성하되, 초기 단계는 제약을 약하게 두고 이후 단계에서 IQA encoder feature가 왜곡 prototype과의 코사인 유사도를 더 크게 갖도록 시간 가중 prototype guidance loss를 걸어 “왜곡 강도 영역”으로 명시적으로 정규화합니다.

- **Empirical Impact**: 생성된 distorted 영상은 실제 임상 artifact가 주는 진단 간섭을 downstream 분류 성능 저하(PI-RADS, Gleason scoring)로 재현하며, synthetic 이미지로 학습한 correction 모델은 실제 왜곡 데이터에서도 유의미한 개선을 보입니다. 비교 실험에서 CycleGAN/일부 diffusion 기반 unpaired 대안은 기하학적 변형을 제대로 만들지 못했고, OT-FM 등은 블러 중심이거나 왜곡 고강도 영역 도달이 부족했습니다. 특히 두 임상 과제에서 in-distribution과 외부 PRIME 데이터셋 모두를 사용해, 합성 쌍이 실제 보정 학습에 “효과적으로 전이”된다는 점을 정량·정성 모두로 입증합니다.



### URDF Synthesis from RGB-D Sequences via Differentiable Joint Inference and Energy-Consistent Verification (https://arxiv.org/abs/2606.18861)
- **Prior Approaches**: 기존 연구는 관절의 종류·파라미터 추정, 파트(링크) 기하 복원, URDF 생성 등을 각각 따로 최적화하는 경우가 많았습니다. 또한 많은 파이프라인이 시각/형상 손실 중심으로 학습되어, 에너지 보존 같은 동역학 불변조건을 검증하거나 학습 신호로 직접 연결하지 못해 장시간 시뮬레이션에서 드리프트가 누적되는 문제가 지적됩니다. 더 나아가 세그멘테이션-후-피팅 방식은 작은 분할 오류가 관절축 추정 오차로 크게 번질 수 있습니다.

- **Core Contribution**: KinemaForge는 short RGB-D 시퀀스만으로 링크의 파트 기하, 조인트 토폴로지, 조인트 파라미터를 한 번에(조인트-레벨까지) 복원하는 constraint-driven 파이프라인을 제안합니다. 핵심은 Featherstone의 articulated-body dynamics를 미분 가능하게 연결해, 렌더링·동역학 불일치로부터 관절축 파라미터를 함께 업데이트하고 결과 URDF를 energy-consistent verifier로 검증한다는 점입니다. 즉, “그럴듯한 모델”을 넘어 “물리적으로 관측과 양립하는 모델”을 목표로 합니다.

- **Technical Challenges**: 난제는 (1) 분할/파트 제안에서 관절-파트 연계를 단단히 고정하지 못하면 오차가 시스템적으로 커지고, (2) URDF가 시각적으로는 맞더라도 자유 응답에서는 비물리적 거동을 할 수 있다는 점입니다. KinemaForge는 오버-세그먼트된 후보들 사이에 관절-파트 연계를 소프트 에지로 둔 kinematic constraint graph로 토폴로지를 먼저 안정화하고, screw-axis를 연속 변수로 두어 differentiable screw-axis solver로 관절축/원점/한계를 그래디언트 기반으로 최적화합니다. 여기에 energy residual loss를 추가해 재구성 모델을 같은 미분 가능 시뮬레이터에서 롤아웃하며 물리적 에너지 증분이 관측과 맞는지까지 페널티로 학습합니다.

- **Empirical Impact**: PartNet-Mobility 5개 카테고리 및 내부/외부 RGB-D 벤치마크에서 KinemaForge는 관절축 에러를 PARIS 대비 37.4% 줄이고, Ditto 대비 46.6% 줄였습니다. 50초 장기 롤아웃 드리프트는 PARIS 대비 64%, Ditto 대비 73% 감소했으며, URDF 기반 closed-loop 조작에서 성공률은 Ditto보다 14.6%p(예: 85.4% vs 70.8%) 향상되는 결과를 보였습니다. 또한 ablation에서 constraint graph와 differentiable joint solver, energy-consistency loss가 각각의 실패 모드를 어떻게 줄이는지 정량적으로 확인해 “물리 검증이 성능을 실제로 바꾼다”는 메시지를 강화합니다.



### Quantification of Uncertainty with Adversarial Models in Medical Image Segmentation (https://arxiv.org/abs/2606.18860)
Comments:
          Accepted at MICCAI 2026

- **Prior Approaches**: 기존 의료 영상 분할의 불확실성 추정은 Deep Ensembles, MC Dropout, evidential 방법(SURE) 등으로 널리 시도돼 왔지만, 다수는 posterior의 일부 영역만 무작위/국소적으로 탐색하는 경향이 있습니다. 그 결과 경계부나 미세한 병변처럼 모델이 틀리기 쉬운 영역에서 과도하게 확신하는 miscalibration이 남아 임상적 취약성을 가릴 수 있습니다. 또한 classification에서와 달리 segmentation은 픽셀마다 표적공격 경우의 수가 폭증해, 충분히 촘촘한 실패 모드 커버가 어렵다는 한계가 있습니다.

- **Core Contribution**: QUAM-SM은 post-hoc 방식으로 targeted adversarial search를 수행해 모델 결정이 쉽게 뒤집히는 “adversarially fragile” 픽셀을 찾아내는 프레임워크입니다. 이 과정에서 aleatoric 불확실성과 epistemic 불확실성을 분리해, 주석 모호성(데이터/라벨 불확실성)과 모델 무지(모델의 불확실성)를 구분해 해석 가능하게 만듭니다. 즉, 단순한 불확실성 크기 추정이 아니라 “어떤 픽셀이 실제 오판으로 전환되기 쉬운지”를 경계 민감도로 더 잘 드러내는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 공격이 임상적으로 그럴듯한 분포 내에서 일어나 posterior coverage를 넓히면서도 (2) 픽셀 단위로 표적 라벨을 다양하게 정의해 실패 가설을 충분히 탐색하는 것입니다. QUAM-SM은 고정된 기준 분할에서 출발해 penalty loss로 학습 분포 일관성을 유지하고, adversarial loss로 targeted 대체 분할(전역/학습기반/형태학-증강 마스크)로 유도합니다. 또한 adversarial 생성 편향을 보정하기 위해 Mixture Importance Sampling(MIS)을 적용해 가중 Bayesian Model Average를 만들고, Expected Entropy(EE)로 per-pixel aleatoric/epistemic을 분해해 산출합니다.

- **Empirical Impact**: REFUGE(안저)와 QUBIQ2021(전립선 MRI) 두 공개 데이터셋의 multi-annotator 설정에서 QUAM-SM은 기준선 대비 reliability와 boundary sensitivity가 모두 개선되며, PCC와 R2에서 최고 성능을 보였습니다. 특히 aleatoric uncertainty가 다중 관찰자 변이와 더 강하게 정합되어, 주석 모호성을 반영하는 불확실성 지도 품질이 높음을 보여줍니다. 추가로 형태학 기반 targeted attack이 일관되게 우수했고, 커널 크기(3x3이 아니라 33/55)와 반복 횟수(에로전/딜레이션 iterations)를 조절할 때 성능이 더 향상되어 실용적 튜닝 가이드도 제공합니다.



### From Bounding Boxes to Visual Reasoning: An On-Policy Data Annotation Tool for Vision-Language Models (https://arxiv.org/abs/2606.18846)
Comments:
          14 pages, 7 figures

- **Prior Approaches**: 기존 비전-언어 모델(VLM) 학습용 데이터 어노테이션 툴은 주로 closed-set detection 중심이라 공간 좌표, open-vocabulary 설명, 구조화 속성, 위상 관계를 한데 묶기 어렵다. 또한 오프라인 방식에 머물러 annotation-트레이닝이 분리되고, 오류 검증도 수동 품질 라벨에 의존해 비용이 커진다. 결과적으로 한 번 라벨링한 데이터를 다양한 추론 태스크로 재활용하기도 힘들다.

- **Core Contribution**: 이 논문은 VLM 학습에 맞춘 오픈소스 어노테이션 도구 ScreenAnnotator를 제안한다. 공간/의미/구조를 atom 스키마 하나로 통합해 좌표-텍스트-속성을 결합한 단일 표현을 만들고, 같은 atom을 템플릿으로 펼쳐 여러 reasoning 태스크를 합성할 수 있게 한다. 여기에 on-policy annotation loop와 Bayesian Annotation Verifier(BAV)로 모델-인간 협업을 닫힌고리로 운영한다.

- **Technical Challenges**: 핵심 난제는 (1) 좌표와 자유형 설명, 구조화 속성을 동시에 표현할 수 있는 스키마 설계, (2) 어노테이션 품질을 매 라운드마다 자동으로 감지하며 on-policy로 재학습하는 절차, (3) 라벨링 비용을 늘리지 않고도 태스크 다양성을 확보하는 데이터 합성 방식이다. 이들은 Unified Annotation Atom 스키마, 모델 보조 pre-annotation→인간 수정→BAV 불확실성 기반 재검토→즉시 재학습의 on-policy 루프, 그리고 템플릿 기반 multi-task data synthesis로 해결한다. BAV는 MC Dropout과 오류 주입(accepted/공간 훼손/범주 훼손)을 이용한 self-supervised 학습으로 별도 품질 라벨 없이 defect 확률을 추정한다.

- **Empirical Impact**: 실험은 flowchart와 mobile GUI screenshot 두 시나리오에서 진행됐고, on-policy 루프는 accept rate를 flowchart에서 거의 100%, GUI에서 77%까지 끌어올리며 라운드가 진행될수록 이미지당 어노테이션 시간도 감소시켰다. BAV는 lift 지표에서 상위 검사 예산(1~10%) 내에 실제 오류를 랜덤 대비 크게 더 많이 포착해 재검토 효율을 높였다. flowchart QA에서 ScreenAnnotator로 합성·학습한 fine-tuning은 평균 정확도 76.1%로 기준 41.0% 대비 절대 35.1%p 향상을 보였고, 구조적 위상/경로 추론 및 공간 추론에서 특히 큰 개선이 관찰됐다.



### Rethinking Air-Ground Collaboration: A Progressive Cross-Task Benchmark and Socialized Learning Framework (https://arxiv.org/abs/2606.18841)
- **Prior Approaches**: 기존 air-ground 협업은 localization, retrieval, tracking, cooperative detection처럼 단일 목적 중심으로 cross-view fusion을 구성하는 경우가 많아, localization–target association–fine parsing 간 기능적 의존성을 명시적으로 다루지 못했습니다. 또한 UAV와 UGV의 관측은 기하·스케일·가림 등으로 불일치가 커서, 동일한 특징을 무차별 공유하면 negative transfer가 쉽게 발생하는 문제가 지적됩니다.
멀티태스크 학습 연구도 주로 single-view의 dense prediction을 전제로 공유 표현을 설계하는 경향이 있어, 이질적인 공중/지상 관측이 결합되는 상황에서는 패러다임 미스매치가 생깁니다.

- **Core Contribution**: 이 논문은 air-ground perception을 “점진적(progressive) cross-task 협업”으로 모델링하고, coarse-to-fine 흐름에서 localization→association→identity-aware parsing이 서로 어떻게 보강되는지 정식화합니다. 이를 위해 Air-Ground Progressive Collaboration(AGPC) 벤치마크(동기화된 89개 시퀀스, 745K+ 원시 비디오 프레임)를 구축해 다단계 협업을 체계적으로 평가할 기반을 제공합니다.
그 위에 Socialized Co-Perception(SCP)를 제안하며, 핵심 모듈 Dual-Layer Router(DLR)로 입력 측 multi-scale expert 선택과 출력 측 task-conditioned 조절을 분리해 유익한 상호작용은 살리고 간섭은 억제하도록 설계합니다.

- **Technical Challenges**: 주요 기술적 난제는 (1) 협업을 작업 단계(stage)별로 분리된 채로 두지 않고, localization의 전역 단서가 association과 parsing으로 전달되도록 결합하는 것, (2) UAV/UGV의 시각적 이질성 때문에 공유가 오히려 해가 되는 부정 전이를 줄이는 것입니다. 논문은 DLR에서 cross-view·cross-task 상호작용을 선택적으로 만들어 task-conditioned 교환으로 필요한 정보만 흐르게 하고, 나쁜 간섭은 라우팅 단계에서 필터링하도록 구현합니다.
또한 ReID는 DLR-driven feature interaction에 직접 참여시키지 않고 identity-level purity를 따로 유지해, association을 위한 결합은 강화하되 파싱 학습의 오염 가능성을 낮추는 전략을 취합니다.

- **Empirical Impact**: 실험은 SCP가 기존 단순/균일한 fusion 대비 더 효과적인 task-conditioned 협업임을 보여주며, coevolutionary gain 3.73%, downstream 평균 성능은 7.86% 향상된 결과를 보고합니다. 특히 localization→cross-view association→identity-aware segmentation으로 이어지는 점진적 의존 구조에서 단계 균형이 중요함을 ADP(기하평균 기반) 등 지표로 확인했습니다.
이 연구는 benchmark(AGPC)와 모듈(DLR)을 함께 제공해, 앞으로 air-ground 협업을 “single-task fusion”이 아니라 “progressive cross-task collaboration” 관점에서 비교·검증할 수 있는 표준 축을 제시했다는 점에서 의미가 큽니다.



### DreamReg: Belief-Driven World Model for 2D-3D Ultrasound Registration (https://arxiv.org/abs/2606.18825)
- **Prior Approaches**: 기존 초음파(US) 2D-3D 등록은 주로 ① 특징을 추출해 6-DoF 강체 변환을 맞추는 기하학 기반 방법, ② end-to-end 포즈 회귀로 한 번에(또는 짧은 horizon) 포즈를 예측하는 방식으로 나뉩니다. 하지만 실제 수술에서는 시각 피드백에 따라 프로브 자세를 계속 조정하므로, 관측(잡음과 speckle 포함)과 액션(프로브 이동)이 강하게 결합됩니다. 이 결합을 무시한 one-shot/short-horizon 접근은 sparse view나 부분 관측 상황에서 취약해지기 쉽습니다.

- **Core Contribution**: DreamReg는 2D-3D 등록을 ‘강체 변환에 대한 신념(belief) 업데이트’라는 순차적 문제로 재정의합니다. 매 시점 잠재 신념 상태를 유지해 지금까지의 관측을 요약하고, 새 단면이 들어오면 learned dynamics로 변환을 점진적으로 정제합니다. 또한 추론 시에는 실제 이미지를 추가로 보지 않고도 내부 imagination(내적 시뮬레이션)로 후보 프로브 모션을 펼쳐 예측 결과를 통합해 수렴시키는 방식입니다.

- **Technical Challenges**: 핵심 난점은 부분 관측과 액션-관측 결합 때문에, 다음에 어떤 단면이 나올지(프로브가 어디로 움직이면 관측이 어떻게 변하는지)를 단순 회귀만으로는 안정적으로 모델링하기 어렵다는 점입니다. DreamReg는 prior/posterior 구조의 action-conditioned latent world model로 신념의 전이를 학습하고, 학습 단계에서 임상 스캔 행동을 흉내 낸 프로브 궤적을 주어 belief state가 관측에 맞게 업데이트되도록 supervision을 설계했습니다. 추론 단계에서는 learned prior로 latent 전이를 굴리며 policy가 pose increment를 선택하고, 그 결과를 신념 업데이트에 반영해 반복 정제를 수행합니다.

- **Empirical Impact**: CAMUS와 μRegPro에서 DreamReg은 기하 오차(거리/이동/회전)뿐 아니라 I-NCC, SSIM 같은 영상 유사도와 P-NCC 같은 파라미터 일관성 지표에서 일관된 우위를 보였습니다. 특히 similarity 기반 지표 개선이 두드러져, 단순 포즈 수치 감소를 넘어 구조적으로 더 정합된 등록을 만든다는 점을 시사합니다. 속도는 EUReg보다 낮게 나오지만 40대~50대 FPS로 실시간(30 FPS 기준) 요구를 충족하며, 부분 관측 환경에서도 믿음 기반 반복 정제가 임상 유효성을 가질 수 있음을 보여줍니다.



### Where Will They Go? Modelling Multimodal Pedestrian Manoeuvres from Ego-centric Videos (https://arxiv.org/abs/2606.18824)
Comments:
          Accepted at The IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026

- **Prior Approaches**: 기존 보행자 궤적 예측은 에이전트 간 상호작용과 장면 문맥을 attention이나 recurrent 모델로 학습하고, 차량·목표 중심 디코딩(goal-driven decoding)으로 성능을 끌어올려 왔다. 또한 CVAE나 diffusion 같은 생성 모델로 여러 미래를 sampling하지만, 보행자에서는 단일 unimodal 분포에서 여러 샘플을 뽑다 보니 ‘mixed-mode’처럼 모션 패턴 사이를 어중간하게 잇는 비현실 궤적이 생기기 쉽다. 특히 ego-centric 데이터에서 보행자 모달리티를 명시적으로 분리해 의도를 반영하는 접근은 제한적이었다.

- **Core Contribution**: 이 논문은 보행자 의도(횡단 여부)를 기준으로 미래 궤적의 분포를 모드별로 나눠 학습하는 mode-aware 프레임워크 MMPM을 제안한다. PIM(Pedestrian Interaction Module)은 gaze·hand gesture 같은 비언어 행동을 포함해 보행자-차량·보행자-환경 상호작용을 공동으로 인코딩하고, MTP(Mode-aware Trajectory Predictor)는 crossing / non-crossing 두 모드의 미래 분포를 각각 따로 학습한다. 디코딩 단계에서도 mode consistency를 강제해 모드 붕괴를 줄이면서 의미 있는 다중 예측을 만든다.

- **Technical Challenges**: 핵심 기술 난제는 ‘one-to-many’인 미래 매핑을 단일 분포 sampling으로는 해결하기 어렵다는 점이며, 모드별로 실제로 분리된 궤적 분포를 안정적으로 학습·생성해야 한다. 논문은 crossing/non-crossing을 MTP의 두 모드로 정의하고, CVAE 기반의 인식/사전 네트워크로 각 모드의 잠재변수와 미래 궤적을 학습하되, 학습 시 ground-truth 모드로 분기(branch)를 선택해 mode-collapse를 방지한다. 또한 query-based decoder와 모드 확률로 각 모드에서 샘플 수를 가중 배분해 ‘mixed-mode’ 샘플이 줄어드는 방향으로 추론을 구성한다.

- **Empirical Impact**: PIE와 JAAD(ego-centric)에서 제안 방법은 기존 SOTA 대비 궤적 예측 성능을 전반적으로 개선했으며, 특히 MTP 도입이 장기(예: 1.5s) 성능 향상에 기여함을 보였다. 행동 피처(gaze·gesture)까지 결합하면 단기 및 최종 변위 오차가 추가로 크게 감소해, 의도 신호가 실제 성능으로 연결됨을 확인했다. 또한 spatio-temporally 정합성을 따지는 데이터 기반 검증 프로토콜을 도입해 샘플의 ‘현실성/유효성’까지 평가했으며, 프레임 단위 displacement error에서 이전 대비 최대 4.73% 개선을 보고했다.



### Fuzzy-Geometric Branch-Point Modeling for Structure-Aware Augmentation of Handwritten Chinese Characters (https://arxiv.org/abs/2606.18793)
- **Prior Approaches**: 기존 데이터 증강은 elastic distortion, TPS, 로컬 affine 같은 강한 기하 변환으로 다양성을 늘리지만, 필기 특유의 비선형 휘어짐과 연결 불확실성을 충분히 반영하지 못해 획 토폴로지 손상 위험이 큽니다. GAN, diffusion 같은 생성 모델은 현실감은 높일 수 있어도 stroke 수준 연결·교차 구조를 명시적으로 보장하지 못해 few-shot/보안 인증 환경에서 위상 붕괴나 특징 왜곡이 발생하기 쉽습니다. 또한 skeleton 기반 세부 증강은 보통 임계값 기반(이진) branch-point 판정에 의존해 중국의 복잡한 획 교차·유착·급격한 꺾임에서 오세그멘테이션/언더세그멘테이션 문제가 커집니다.

- **Core Contribution**: FGSA(Fuzzy Geometry-driven Structure-aware Augmentation)는 필기 skeleton의 branch point를 이진 이벤트가 아닌 “퍼지 집합”으로 모델링해, 불확실한 교차/연결 상태를 연속적인 membership 공간에서 다룹니다. 이와 함께 방향장 divergence(방향 분기 불일치)를 결합한 dual-evidence membership을 제안해, 토폴로지 애매함을 기하적으로도 제약합니다. 마지막으로 Bézier 곡선을 통한 kinematics-aligned 재구성과 다전략 섭동으로 구조 보존과 다양성을 동시에 맞추는 증강 파이프라인을 완성합니다.

- **Technical Challenges**: 핵심 난제는 (1) 획 교차·유착·노이즈 때문에 branch-point가 비결정적으로 나타나는 상황에서, (2) 수동 튜닝 없이 퍼지 경계를 최적으로 잡고, (3) 분리된 획을 Bézier 재구성 중 구조가 망가지는 것을 막는 것입니다. FGSA는 hard threshold 대신 dual-evidence fuzzy membership를 만들고 α-cut 기반 defuzzification으로 안정적인 획 분리를 수행한 뒤, 라벨 없이 Bézier 재구성 가능성과 분할 안정성(anti-collapse), 접합부 연속성(C0)을 동시에 보는 surrogate objective를 DE(unsupervised differential evolution)로 최적화합니다. 그 결과, 수동 어노테이션 없이도 stroke decoupling의 품질을 제어하면서 구조 충실도를 유지하도록 설계됩니다.

- **Empirical Impact**: FGSA는 CASIA-HWDB1.1, ChiSig, 그리고 새로 공개한 LZUSig(중국 필기 서명의 fine-grained structural degradation 특화 대규모 벤치마크)에서 비교 기준 대비 word-level error rate(ΔWER)를 유의미하게 낮추며 최적에 가까운 인식 이득을 보였습니다. 특히 단순 성능 상승이 아니라 “과제 이득-구조 충실도-판별 특징 보존”의 견고한 절충(trade-off)을 달성한 점이 실사용 보안 인증 관점에서 의미가 큽니다. LZUSig와 함께, 구조 열화에 대한 연구를 더 세밀한 설정으로 밀어붙일 수 있는 실증 기반도 제공한다는 점에서 분야 파급력이 기대됩니다.



### HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Spac (https://arxiv.org/abs/2606.18788)
- **Prior Approaches**: 기존 필기체 생성은 GAN·diffusion·transformer 같은 딥러닝 기반이 주류였고, 온라인/오프라인 필기 모두에서 성능이 크게 올랐습니다. 다만 스타일별 아키텍처·대규모 데이터·막대한 compute 의존성이 커서 새로운 필체나 low-resource 환경 적응이 어렵고, raster 중심 출력이라 획(스트로크) 수준 제어와 편집성이 제한됩니다. 또한 언어/스크립트마다 특화된 구조가 필요한 경우가 많아 다국어·다도메인 확장이 번거롭다는 한계가 남아 있습니다.

- **Core Contribution**: HandwritingAgent는 자연어로 제어되는 에이전트 방식으로, SVG(Scalable Vector Graphics)에서 획 시퀀스를 직접 이산적으로 생성해 필체를 합성합니다. 스타일별 학습 없이도 입력 스타일 이미지(또는 stroke)와 요청 텍스트(대화형/비대화형)를 함께 받아, 언어 모델이 기하 단서를 분석해 glyph를 단계적으로 계획·생성하도록 설계했습니다. 그 결과 해상도에 독립적이고 편집 가능한(해석 가능한) 벡터 출력과, 스크립트·도메인 전반의 일반화가 강조됩니다.

- **Technical Challenges**: 핵심 난제는 필기 스타일의 변동성이 ‘형상·질감·압력·연결’처럼 복합적이면서도, 이를 자연어 지시와 정합되게 획 수준으로 재현해야 한다는 점입니다. 논문은 (1) 입력을 공용 grid-canvas 좌표계와 구조화된 XML 표현으로 정규화하고, (2) LLM OCR+휴먼 인 더 루프 보정으로 캐릭터/워드 단위 분할과 라벨링 정확도를 끌어올리며, (3) glyph bank로 참조 스타일의 구조·자간·곡률 성향을 고정해 추론 기반 style transfer를 수행하는 흐름을 제시합니다. 또 cubic Bézier 곡선과 시간(temporal) 값을 포함한 stroke-point 시퀀스를 SVG path로 변환해 연속성과 자연스러운 쓰기 역학까지 유지하려고 합니다.

- **Empirical Impact**: 실험에서는 IAM(모방)·CASIA/IAM-LINES(다국어)·CROHME/EDU-CHEMC/자체 physics 노트(수학·과학) 등 다양한 태스크에서 기존 generative handwriting 모델과 비교해 성능을 입증했습니다. IAM Word/Line에서 SSIM·FID·HWD가 상위권이고, 가독성 측면에서도 ΔΔCER 격차가 크지 않아 ‘시각적 구조 보존’이 특히 강점으로 나타납니다. 또한 중국어처럼 라틴을 넘어서는 스크립트 일반화와, 수학/과학 표현에서 ExpRate·WER 등 인식 기반 지표가 경쟁력 있게 개선되며, reasoning(생각 모드) 유무에 따른 성능 차이로 추론이 장문 구조·연속성 유지에 실제로 기여함을 보여줍니다.



### Learned Radius Estimation for UDF-Based Point Cloud Reconstruction (https://arxiv.org/abs/2606.18787)
- **Prior Approaches**: UDF는 inside/outside 분류 없이 점군에서 표면까지의 unsigned distance를 다뤄, 닫힌/열린 표면 모두에 적용 가능하다. LoSF-UDF는 전역 잠재코드 대신 로컬 패치로 UDF를 가볍게 추정하지만, 패치 support radius를 고정해 세밀한 디테일에서는 멀티-서피스 오염(너무 큼) 또는 문맥 부족(너무 작)이 생긴다. GeoLA는 curvature 휴리스틱으로 radius를 적응시키려 했으나, 단일 스칼라 curvature로는 국소 기하의 이질성을 충분히 표현하지 못한다.

- **Core Contribution**: 이 논문은 query마다 다른 support radius를 예측하는 learned per-query radius selector를 제안한다. 기존의 curvature 기반 휴리스틱을 데이터 기반 연속값 예측으로 대체하되, frozen LoSF-UDF backbone에 그대로 끼워 넣어(재학습 없이) 파이프라인을 크게 늘리지 않는다. 핵심은 discrete 후보 반경이 아니라 off-grid 연속 target radius를 학습에 활용한다는 점이다.

- **Technical Challenges**: support radius의 정답이 “후보 집합 중 argmin” 형태로는 충분하지 않아, 연속적인 supervision이 필요했다. 이를 위해 cached UDF error curve에서 포물선(parabolic) 보간으로 argmin 근처의 정답 반경을 보간해 off-grid target radii를 만들고, boundary 또는 비볼록 이웃에서는 argmin 반경으로 폴백한다. 또한 radius 선택에 민감한 샘플에 더 큰 가중치를 주기 위해 confidence-weighted normalized L1 loss를 사용해, 패치의 국소 특징(마스크 ResNet-PointNet 특징, 점 개수 로그, 방사 밀도 histogram)에서 radius ratio를 연속적으로 회귀하도록 학습한다.

- **Empirical Impact**: 실험에서는 ScanNet(학습 완전 제외), ShapeNet-Cars, DeepFashion3D를 대상으로 재구성 정확도를 비교했으며, 제안 방법이 CD와 F1@0.005에서 3개 데이터셋 전반의 최고 성능을 보였다. 특히 ScanNet에서 GeoLA 대비 F1@0.005가 0.645→0.691, CD가 0.795→0.692로 개선되어, 소수(CAD/garment 80개) 학습만으로도 실내 스캔에 일반화됨을 보여준다. 다만 F1@0.01과 NC는 고정 radius LoSF-UDF가 소폭 우위인 경우도 있었고, 이는 이 방법이 “엄격한 거리 기반 지표”에서 국소 기하에 맞춘 radius 학습의 이점을 명확히 입증했음을 시사한다.



### SCR-Guided Difficulty-Aware Optimization for Infrared Small Target Detection (https://arxiv.org/abs/2606.18783)
Comments:
          Accepted at CVPR 2026 Workshops (PBVS). Published version: this https URL

- **Prior Approaches**: 기존 IRSTD(적외선 소형 표적 탐지)는 배경 클러터, 낮은 대비, 약한 공간 응답 때문에 PSF-like한 국소 반응을 정확히 분리하기 어렵다. 그래서 대부분은 U-Net류 아키텍처/어텐션 등 표현 학습을 개선하거나, IoU·Dice·SLS처럼 “겹침(overlap)” 기반 손실로 감독 신호를 구성해 왔지만, 이런 손실은 표적 가시성(SCR) 차이를 제대로 반영하지 못해 저가시성 샘플에 대한 최적화 신호가 어긋날 수 있다.

- **Core Contribution**: 논문은 REEM(Reweighted Explicit-visibility Enhanced Modulation)으로, 학습 중에 Signal-to-Clutter Ratio(SCR)를 물리적으로 의미 있는 가시성 prior로 사용해 손실의 학습 신호를 난이도별로 재가중한다. 네트워크 구조나 inference 동작을 바꾸지 않고, 정답 기반 “로컬 SCR”을 계산해 soft-IoU 학습 신호에 미분가능한 조절(modulation)을 적용함으로써 저가시성 표적에 더 큰 기울기(gradient) 비중을 준다.

- **Technical Challenges**: 핵심 난제는 overlap 기반 손실과 달리 SCR 같은 가시성 척도를 “예측 의존 없이” 학습 신호에 통합해도 최적화가 불안정해지지 않게 만드는 것이다. REEM은 입력과 정답에서 계산한 ground-truth 로컬 SCR을 이용해 단조 감소형이면서 상한이 있는 bounded 가중치 함수를 설계하고, soft-IoU 항에만 적용해 원래의 최적화 지형을 크게 해치지 않도록 구성했다.

- **Empirical Impact**: 실험은 IRSTD-1k와 NUDT-SIRST에서 baseline MSHNet 대비 일관된 개선을 보였고, IoU와 Pd는 함께 상승하면서 FA(오경보)는 크게 감소했다. 특히 SCR이 낮은 구간에서 성능 향상이 두드러져, 예를 들어 IRSTD-1k에서는 IoU 65.60%→68.44%, FA 13.51 ppm→6.30 ppm 및 Pd 93.20%→93.88%를 달성했으며 저가시성 SCR bin에서 FA 감소폭이 더 컸다. 반대로 SCR이 매우 높은 구간(SCR≥8)에서는 약간의 FA 변화가 나타났지만, REEM이 저가시성 우선 최적화를 목표로 하는 설계 특성과 부합하며, 추론 시 오버헤드 없이 동일한 아키텍처로 배포 호환성까지 확인했다.



### SAMA: Semantic Anchor-aligned Augmentation for Unified Low-Resource Multimodal Information Extraction (https://arxiv.org/abs/2606.18780)
Comments:
          Accepted by IEEE Transactions on Multimedia

- **Prior Approaches**: 기존 Multimodal Information Extraction(MIE)용 Data Augmentation은 텍스트와 이미지를 각각 변형하거나 모달리티 간 정렬을 거칠게 처리하는 경우가 많아, 생성된 텍스트-이미지 쌍 사이의 의미 정합성이 깨지기 쉽습니다. 또한 MNER/MRE/MEE를 위한 증강 파이프라인이 과제별로 분절돼 공유 의미를 재사용하지 못해 저자원 환경에서 성능이 제한됩니다. 더불어 closed-source Multimodal LLM 기반 증강은 비용·지연 문제가 크고 MIE의 스키마 제약(정확한 entity/span, relation triplet, event 구조)을 잘 따르지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Semantic Anchor-aligned Multimodal Augmentation(SAMA)라는 통합 프레임워크로 MNER, MRE, MEE에 공통으로 적용 가능한 고품질 합성 데이터를 생성하는 방법을 제안합니다. 핵심은 ground-truth에서 구조화된 semantic anchors를 만들고, 이를 Collaborative Multi-Experts Multimodal Large Language Model(CME-MLLM)의 텍스트 생성과 Anchor-Preserving Diffusion의 이미지 합성에 동시에 조건으로 거는 것입니다. 결과적으로 다양성은 확보하면서도 “스키마 준수 + 교차모달 일치”를 동시에 강제하는 증강을 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 생성이 자연스러워지는 대신 entity 타입/관계 방향/트리거-인자 구조가 흔들리는 semantic drift를 막는 동시에, 텍스트-이미지 정렬을 정밀하게 유지하는 것입니다. SAMA는 (1) inline markup 기반 semantic anchors로 생성 확률공간을 유효 스키마로 제한하고, (2) Universal Adapter와 task-specific Adapters를 anchor-gate로 혼합해 공유 의미와 과제 특이 제약을 동시에 학습·생성하며, (3) anchor 가중 prompt와 마스크 latent blending으로 확산 과정에서 핵심 엔티티의 시각 정체성을 보존합니다. 마지막으로 Dual-Constraint Filtering으로 CLIP 기반 cross-modal consistency와 anchor fidelity를 함께 만족하는 후보만 선택해 수동 검증 없이 신뢰도를 확보합니다.

- **Empirical Impact**: Twitter-15(MNER), MNRE(MRE), M2E2(MEE) 전 벤치마크 실험에서 SAMA는 기존 증강 베이스라인을 저자원(10%, 20%, 40%)과 준전데이터 조건 모두에서 일관되게 능가합니다. 특히 극단적 저자원(10%)에서 task-specific SOTA 대비 MNER는 F1 +1.7%, MRE는 약 +2.0%(베이스라인 HVPNeT 기준), MEE는 MEAE/ MED에서 각각 +4.7% / +5.9%로 큰 격차를 보였습니다. 이는 구조화된 semantic anchors 기반의 정합성 중심 증강이 단순 보편 증강(MixGen)이나 단일 모달/과제편향 증강의 한계를 넘어, MIE의 학습 신호 품질을 실질적으로 끌어올린다는 점에서 의미가 큽니다.



### SpectralDiT: Timestep-Conditioned Spectral Residual Correction for Flow-Matching DiTs (https://arxiv.org/abs/2606.18765)
- **Prior Approaches**: 기존 flow-matching Diffusion Transformers(Flow Matching DiT)는 timestep 조건을 통해 생성 품질을 끌어올리지만, 잔차 업데이트의 주파수 성분(저주파/고주파)까지 세밀하게 분해·보정하는 구조는 제한적입니다. 그 결과 특정 패치/블록에서 주파수 스펙트럼 정렬이 흔들리면 미세한 질감·경계 표현이 덜 안정적일 수 있습니다.

- **Core Contribution**: 이 논문은 SpectralDiT로, MLP residual branch에 timestep-conditioned spectral correction 모듈을 가볍게 추가합니다. 각 잔차 업데이트를 패치-token 격자에서 저주파·고주파로 분해한 뒤, zero-initialized additive gate로 학습 초기에 baseline DiT와 동일하게 시작하면서도 필요 시 주파수 보정을 점진적으로 적용합니다.

- **Technical Challenges**: 핵심은 주파수 분해/보정이 생성 안정성을 해치지 않도록 설계하는 것이며, 이를 위해 잔차를 주파수 성분으로 분해한 뒤 gate를 0으로 초기화해 초기 성능을 기준선과 동일하게 맞춥니다. 또한 block-specific로 보정 패턴이 안정적으로 형성되도록 학습·시각화를 함께 수행해, 어떤 블록에서 어떤 주파수 보정이 일어나는지 확인합니다.

- **Empirical Impact**: CIFAR-10 pixel-space 생성에서 patch size 1 기준 FID가 20.78에서 19.71로 개선되고, radial Fourier spectrum gap도 줄었습니다. ImageNet-100 latent diffusion에서는 추가 이론 FLOPs 0.6%와 파라미터 1.36%만으로 classifier-free guidance(CFG 2.0) 하에서 상대 FID 8.7% 감소를 달성했으며, 모든 결과는 5개 seed 평균과 ablation/gate 시각화로 뒷받침됩니다.



### SMART: A Flexible, Interpretable, and Scalable Spatio-temporal Brain Atlas from High-Resolution Imaging Data (https://arxiv.org/abs/2606.18753)
- **Prior Approaches**: 기존 시공간 뇌 지도(spatio-temporal atlas) 구축은 주로 black-box 생성 모델에 의존해 유연성이 낮고, 해석 가능성이 제한적이었다. 또한 고차원 3D 의료 영상 시계열로 확장할 때 학습/추론이 어려워 대규모 데이터에서 성능과 일관성 확보가 쉽지 않았다.

- **Core Contribution**: 이 논문은 SMART라는 프레임워크를 제안하며, 장기(종단) 고해상도 3D 의료영상으로부터 ‘연속 질병-시간(disease-time) atlas’를 학습한다. SMART는 전역의 집단 질병 진행 양상과 환자별 해부학적 변형을 분리하고, 지역별 진행을 공유 질병 타임라인 위의 해석 가능한 미분방정식 궤적으로 모델링한다.

- **Technical Challenges**: 핵심 과제는 (1) 고차원 시공간 변화를 유연하게 표현하면서도 (2) 전역 진행의 해석 가능성을 유지하고 (3) 환자별 맞춤을 안정적으로 결합하는 것이었다. SMART는 해부학적 priors로 전역 궤적을 지역별 미분방정식으로 제약하고, 환자 개인화는 dense diffeomorphic displacement를 multi-scale Neural Cellular Automata로 매개해 확장성과 표현력을 동시에 확보한다.

- **Empirical Impact**: SMART는 알츠하이머(AD) 관련 종단 MRI 5개 데이터셋(ADNI-1/GO/2, OASIS-3, AIBL, 1,300명 초과)에서 질병 진행을 해부학적으로 의미 있게 예측하며, forecasting 정확도는 state-of-the-art를 달성했다. 특히 adversarial 및 diffusion 기반 기준선 대비 temporal consistency가 개선돼, 의료 영상 타임시리즈에서 ‘유연·해석 가능·확장’ 관점을 새 패러다임으로 제시했다.



### Toward Training-Free Zero-Shot Anomaly Detection in 3D Medical Images: A Batch-Based Approach Using 2D Foundation Models (https://arxiv.org/abs/2606.18749)
- **Prior Approaches**: 기존 supervised segmentation은 병변에 대한 수작업 라벨 의존도가 높아 데이터 다양성이 부족해 배치·스캐너·인구 구성이 바뀌면 성능이 흔들립니다. UAD(unsupervised anomaly detection)는 정상 분포를 학습해 차이를 찾지만, 정상 코호트가 충분히 “깨끗하고 대표성”을 가져야 하고 도메인 갭(획득 프로토콜 등)에서 병변이 아닌 요인으로도 오탐이 늘 수 있습니다. ZSAD는 CLIP 기반 prompt 방식이 고정된 프롬프트/최적화 이슈가 있고 3D에선 slice-wise 처리로 입체 문맥이 끊기기 쉽다는 한계가 있습니다.

- **Core Contribution**: CS3F는 3D 의학 영상에서 training-free로 zero-shot anomaly detection·segmentation을 수행하는 batch-based 프레임워크를 제안합니다. 핵심은 “2D foundation model(동결)”을 3D에 맞게 슬라이스(다축)로 인코딩한 뒤, 이웃 슬라이스를 depth pooling해 localized volumetric token으로 만들고, cross-subject mutual similarity로 토큰 이상점수를 매긴다는 점입니다. 또한 축(axial·coronal·sagittal)별 점수를 융합하고, 병변 약화(attenuation)를 줄이기 위한 coarse-to-fine 토큰화까지 포함합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 3D에서 토큰 수가 폭증해 cross-subject mutual similarity의 계산·메모리 비용이 제곱으로 커진다는 점입니다. CS3F는 random projection으로 토큰 임베딩 차원을 낮춰 거리 계산을 근사적으로 보존하면서 비용을 크게 줄입니다. 또 depth pooling이 병변이 작은 영역일 때 신호를 평균내어 약화시키는 문제를 공식(occupancy 기반 감쇠)으로 분석하고, coarse-to-fine 검색으로 미세 해상도를 쓰되 전수 매칭을 피하도록 라우팅을 도입합니다.

- **Empirical Impact**: CS3F는 뇌 MRI에서 metastases·glioma·stroke를 대상으로 평가되며, 뇌 atlas에 잘 맞는지 여부를 넘어 lung CT에도 검증해 일반화 가능성을 확인했습니다. 결과적으로 “동결된 2D foundation model만으로도” 3D에서 이상 위치를 찾을 수 있음을 보여주고, fine tokenization의 이득이 병변 대비(contrast)와 영상 modality에 따라 달라진다는 점을 실험적으로 뒷받침합니다. 전반 성능은 CLIP-based zero-shot baseline을 여러 벤치마크에서 일관되게 상회하며, 일부에선 reconstruction 기반 UAD baseline과도 경쟁적입니다.



### Clinically Aligned Geometry Constraints for Robust IVUS Vessel Boundary Segmentation (https://arxiv.org/abs/2606.18723)
Comments:
          MICCAI2026 Accepted

- **Prior Approaches**: 기존 IVUS(초음파) 기반 lumen 및 EEM 분할 연구들은 주로 Dice/IoU 같은 겹침(overlap) 점수 최적화에 의존해 경계가 서서히 ‘drift’되거나 토폴로지 위반이 생길 수 있습니다. 단일 프레임 중심의 U-Net/DeepLab 계열은 성능은 나오지만, 직경·방향·단면적 같은 임상적으로 중요한 기하량을 직접 제약하지 못합니다. POLYCORE는 경계 개선에 초점을 두지만 단일 프레임이며, POST-IVUS는 temporal 모델링을 하더라도 픽셀 단위 손실 중심이라 임상 기하를 미분가능하게 감독하지 못합니다.

- **Core Contribution**: GeoCat은 5-frame IVUS 클립을 입력으로 받아, Cartesian-Polar 이중 인코더와 cross-domain attention으로 분할과 기하 일관성을 동시에 맞추는 네트워크입니다. 핵심은 “differentiable geometry consistency loss”로 직경(dmax/dmin), 방향(각도), 단면적/플라크 관련 지표를 학습에 직접 연결해 임상 측정 정확도를 목표로 삼았다는 점입니다. 이 방식은 단순 겹침 점수로는 잘 드러나지 않는 ‘임상 기하 오차’를 줄이도록 설계되었습니다.

- **Technical Challenges**: 임상 지표(직경/각도/면적)는 분할 마스크에서 비연속적 절차(예: 연결요소 분석, 극값 추출 등)를 거치면 end-to-end로 미분이 막히기 쉽습니다. GeoCat은 Cartesian 예측을 polar 격자로 differentiable bilinear grid sampling으로 재파라미터화하고, soft-argmin/temperature-scaled softmax 및 log-sum-exp로 경계·최대/최소 지점을 미분가능하게 근사해 geometry loss를 가능하게 했습니다. 또한 temporal 안정성을 위해 centre-query temporal attention으로 목표 프레임이 인접 프레임 중 유용한 정보를 선택하도록 했고, bidirectional cross-attention으로 두 도메인 표현을 정렬·융합합니다.

- **Empirical Impact**: 146명 환자, 12,242개 프레임(두 상용 IVUS 시스템)에서 GeoCat은 Dice 0.93, 95HD 0.14mm로 경계 정확도를 크게 개선했고 토폴로지 위반율을 1.0%로 낮췄습니다. 더 중요한 임상 기하 성능에서 직경 오차는 0.13~0.16mm, 각도 오차는 약 8도 수준으로 보고되어 플라크 burden 정량의 신뢰도를 높였다는 주장에 힘을 줍니다. 또한 POST-IVUS 대비 Dice는 소폭(0.015p) 개선하면서도 95HD와 토폴로지 위반을 더 줄여, overlap 중심 학습의 한계를 임상 지표 중심 학습이 보완할 수 있음을 실증적으로 보여줍니다.



### Rethinking the Pointer Loss in Table Structure Recognition: Geometry-Aware Pointer Loss for Spatial Locality (https://arxiv.org/abs/2606.18721)
- **Prior Approaches**: TSR(Table Structure Recognition)은 pointer network 기반으로 OTSL/HTML 태그를 예측한 뒤, 각 C(실제 셀) 태그를 텍스트(또는 셀) 영역에 포인팅해 정렬하는 end-to-end 학습 방식이 주류가 됐습니다. TableFormer, TFLOP 등은 구조 예측과 컨텐츠 로컬라이제이션을 분리해 성능을 끌어올렸지만, 포인터 손실에서 negative 후보를 모두 동일하게 취급한다는 공통 약점이 있습니다. 그 결과 빽빽한 표에서는 인접 셀을 골라버리는 near-miss 오류가 학습 중에 충분히 구분되지 않습니다.

- **Core Contribution**: 이 논문은 pointer network가 실패할 때 오류가 공간적으로 어디에 집중되는지 분석하고, 그 패턴을 직접 최적화에 반영하는 Geometry-Aware Pointer(GAP) Loss를 제안합니다. 핵심 아이디어는 ground truth와의 Manhattan distance에 따라 negative 후보의 가중치를 재조정해, 인접 셀에서 더 강한 구분 학습 신호를 주는 것입니다. 모델 아키텍처는 그대로 두고 loss 계산만 바꿔 zero additional inference cost를 유지합니다.

- **Technical Challenges**: 어려움은 “오류가 실제로는 인접 셀에서 많이 난다”는 관찰을 학습 손실로 바꾸는 과정에서, gradient 흐름을 특정 거리 구간에만 집중시키도록 설계해야 한다는 점입니다. GAP는 inverse distance weighting(거리 감소 시 더 큰 가중치)을 적용해 인접 후보에 gradient를 강화하고 먼 후보에는 억제하도록 구성했으며, α=8에서 성능이 가장 좋았습니다. 또한 기존의 TEDS가 정렬 실패를 부분 점수로 가릴 수 있어, C-tag와 bounding box의 정확한 일치만 보는 Position Accuracy(PA) 같은 위치 정밀도 지표로 보완합니다.

- **Empirical Impact**: PubTabNet과 SynthTabNet에서 GAP는 TEDS/SOTA 성능을 유지하면서도 특히 PA에서 일관된 개선을 보였습니다. PA는 PubTabNet에서 91.80→92.35, SynthTabNet에서 98.75→99.31로 약 0.55~0.56%p 상승했고, 오류의 거리 구간 분석에서는 d∈[1,2] 인접 영역에서 상대적으로 가장 큰 개선이 나타났습니다. 더불어 TEDS는 높지만 PA가 낮은 ‘평가 불일치’ 케이스 비율을 줄이며, 표 추출에서 중요한 셀 단위 정렬 실패를 더 정확히 반영하는 학습/평가 방향을 제시했다는 점에서 의미가 큽니다.



### PEFT-MedSAM: Efficient Fine-Tuning of Medical Foundation Models for Explainable Skin Lesion Segmentation (https://arxiv.org/abs/2606.18707)
- **Prior Approaches**: 피부병변(피부 병변) 분할은 진단을 조기에 돕기 위한 핵심 작업이지만, 기존 딥러닝 기반 분할 방법들은 성능이 일관되지 않다는 한계가 있었다. 특히 Medical Segment Anything Model(MedSAM)은 zero-shot 추론만으로도 어느 정도 성능을 보이지만, 표준 세팅에서 병변 분할 정확도를 충분히 끌어올리기 어렵다는 문제가 제기됐다. U-Net 계열 baseline도 존재했으나, 의료 영상의 복잡한 분할 요구를 만족시키기엔 기준선 대비 여지가 남아 있었다.

- **Core Contribution**: 이 논문은 MedSAM을 피부병변 분할에 맞게 적응(adaptation)하기 위한 parameter-efficient fine-tuning 방식인 PEFT-MedSAM을 제안한다. 학습은 mask decoder만 가볍게 학습하고, pre-trained image encoder와 prompt encoder는 frozen으로 유지해 데이터/연산 부담을 줄이면서도 성능을 끌어올리는 전략이다.

- **Technical Challenges**: 핵심 난제는 의료 영상 분할에서 성능을 올리기 위해 보통 필요한 full fine-tuning의 계산·데이터 요구를 낮추면서, 분할의 정밀도를 유지하는 것이다. PEFT-MedSAM은 lightweight mask decoder 중심 학습으로 학습 가능한 파라미터를 제한하고, frozen인 인코더 구성은 유지해 안정적으로 적응이 일어나도록 설계했다. 또한 임상 신뢰도 강화를 위해 Grad-CAM 기반 설명가능성 및 pointing game 평가로 CNN baseline의 병변 영역 판별 타당성을 함께 검증했다.

- **Empirical Impact**: ISIC 2018에서 PEFT-MedSAM은 dice 0.9411, IoU 0.8918을 기록해, fully trained U-Net(dice 0.8715)과 zero-shot MedSAM(dice 0.8997) 모두를 능가했다. PH2 외부 검증에서는 dice 0.9467(표준편차 ±0.0310)로 재현성이 확인됐고, 데이터 간 비교에서 Wilcoxon signed rank test p-value < 0.0001 및 dice 평균에 대한 부트스트랩 95% 신뢰구간 [0.9364, 0.9447]를 제시했다. 추가로 519장 검증 세트에서 Grad-CAM/pointing game 평가가 98.27% 정확도로 병변 포함 영역을 분류했음을 보여, 실제 배치 관점의 신뢰성 확보에도 의미가 있다.



### UniTemp: Unlocking Video Generation in Any Temporal Order via Bidirectional Distillation (https://arxiv.org/abs/2606.18702)
- **Prior Approaches**: 기존 autoregressive 비디오 diffusion 증류는 주로 forward(인과) 방향 생성만 지원해 블록 단위 스트리밍에는 강점이 있지만, 실제 제작 워크플로우의 임의 시간조건(미래 컨텍스트 기반 backward 확장, 양끝 조건 inbetween 등)을 한 모델로 처리하기는 어려웠습니다. 또한 full-sequence attention 기반 마스킹/인페인팅 계열은 유연하지만 추론 비용이 커 실시간 확장에 불리합니다. 더 나아가 backward로 단순히 생성 순서를 뒤집어도 블록 경계에서 깜빡임/연속성 붕괴가 발생해 품질 저하가 두드러집니다.

- **Core Contribution**: 이 논문은 임의의 시간 방향(과거, 미래, 둘 다)에 조건을 걸어 생성할 수 있는 단일 autoregressive 학생 모델을 목표로 합니다. 핵심은 Causal 3D VAE의 인과적 잠재 표현이 backward 생성에서 요구하는 ‘과거 컨텍스트의 결손’을 blockwise anchor latents로 보완하고, 이를 바탕으로 UniTemp(양방향 distillation)를 제안한 것입니다. UniTemp는 추론 시 조건 프레임 조합만 바꿔 bidirectional extension, inbetween generation, looping 등 다양한 작업을 한 모델로 수행하도록 설계됩니다.

- **Technical Challenges**: 가장 큰 기술 난제는 Causal 3D VAE가 각 latent를 ‘오직 past context에 의해 인코딩’하도록 학습된 구조라, backward 생성에서 블록 경계에 필요한 과거 컨텍스트가 제공되지 않아 inter-block flickering이 주기적으로 발생한다는 점입니다. 저자들은 이 아티팩트를 정량화하기 위해 Flickering Ratio를 제안하고, 이를 통해 backward에서 inter-block FR이 비정상적으로 커짐을 확인했습니다. 해결책으로 backward 생성 블록을 (P+B)로 확장해 auxiliary anchor latents를 동시에 denoise하며, 단 생성 출력에는 anchor를 포함하지 않고 경계에서의 끊김만 복원하는 방식으로 안정화를 달성합니다.

- **Empirical Impact**: 실험에서 UniTemp는 short/long 비디오 생성 성능을 forward-only 기반 방법들과 경쟁 수준으로 유지하면서도, backward 및 양방향 조건에서의 제어성을 크게 확장합니다. VBench 중심 평가에서 깜빡임 문제는 anchor latents 도입으로 유의미하게 완화되며, inbetween generation에서도 head/tail 조건을 바탕으로 중간 구간을 자연스럽게 채우는 결과를 보입니다. 결과적으로 UniTemp는 bidirectional video extension, scene transition, looping video generation, visual story generation 같은 다양한 실사용 워크플로우를 single model로 지원한다는 점에서 분야에 실질적 의미가 있습니다.



### Spatially Stratified Distillation for Heterogeneous Radar Place Recognition (https://arxiv.org/abs/2606.18687)
Comments:
          IEEE ICRA Workshop on Open Challenges for Rigorous Robot Perception 2026

- **Prior Approaches**: 기존 heterogeneous radar place recognition은 비싼 360∘ 스핀 레이더의 조밀한 맵과, 4D 고체(120∘×120∘) 쿼리를 매칭해야 한다는 점에서 modality asymmetry가 핵심 병목이다. SHeRLoc 같은 방식은 polar BEV로 공통 표현공간을 만든 뒤 shared backbone 기반으로 정렬하지만, multi-session 환경에서는 4D 쿼리가 관측하지 못하는 조밀한 구조를 적극적으로 활용하지 못해 성능이 흔들린다. 특히 밀도 차이가 큰 영역에서는 정확한 feature alignment 자체가 물리적으로 불가능해 기준 정렬이 오히려 약점이 된다.

- **Core Contribution**: 논문은 Spatially-Stratified Distillation(SSD)를 제안해, 서로 관측 가능한 영역과 불가능한 영역을 물리 FOV 기하로 나눠 distillation을 재설계한다. SSD는 관측이 겹치는 joint 관측 영역에서는 강하게 feature를 정렬하고, 4D student가 비어 있지만 teacher만 구조가 있는 gap 영역은 가중치를 크게 낮춰 “억지 정합”을 피하면서도 약한 구조 prior를 주입한다. 이를 통해 4D 표현을 조밀한 맵의 유효한 맥락으로 끌어올리되, student가 보지 못한 것을 학습으로 “환각”하지 않도록 한다.

- **Technical Challenges**: 문제의 기술적 난점은 teacher-only 영역을 그대로 맞추면 물리적으로 불가능하다는 점과, 반대로 전부 마스킹하면 dense structural supervision의 이점을 잃는다는 점이다. SSD는 이를 해결하기 위해 teacher feature 크기 기반 magnitude mask와 student 입력 기반 gaussian-smoothed FOV 마스크를 결합해 joint/gap을 분리하고, gap 영역은 heavily discounted distillation weights로 약하게 정규화한다. 또한 1×1 convolution으로 채널 basis를 맞추고 L2-normalization 후 cosine distance와 covariance alignment로 활성 스케일·상관 불일치를 동시에 줄인다.

- **Empirical Impact**: HeRCULES 벤치마크에서 SSD는 single-session과 multi-session 모두에서 기존 최고 성능 방법을 일관되게 능가하며, 특히 dynamic multi-session 시나리오에서 유의미한 마진을 보인다. 논문이 보고한 결과는 HeRCULES의 어려운 dynamic sequences에서 state-of-the-art를 달성했음을 시사한다. 더불어 SC/01→SC/03 같은 난구간에서도 SHeRLoc 대비 더 많은 쿼리를 성공적으로 복구하며, 최악 실패에서도 오차가 크게 줄어 4D 레이더 표현 학습의 실질적 개선 효과가 확인된다.



### Multi-Class Brain Tumor Classification Using Advanced Deep Learning Models: A Comparative Study (https://arxiv.org/abs/2606.18682)
- **Prior Approaches**: 뇌 MRI에서 다중 클래스 뇌종양을 분류하기 위해 다양한 CNN 아키텍처를 적용하는 시도가 이어져 왔지만, 임상적으로 중요한 지표인 tumor-wise recall(종양 단위 재현율) 관점에서 비교가 충분치 않은 경우가 많습니다. 또한 단순한 기준선 모델과 pre-trained 모델을 동일한 실험 프레임워크로 맞춰 평가한 연구가 제한적이었습니다.

- **Core Contribution**: 이 논문은 약 10,000장 규모의 임상 출처 MRI 데이터셋을 대상으로 5가지 CNN 아키텍처(VGG16, VGG19, DenseNet121, EfficientNetB0, 커스텀 기준선)를 동일한 조건에서 종양 분류에 적용·비교합니다. 특히 전체 정확도뿐 아니라 tumor-wise recall을 함께 측정해, 임상적으로 더 의미 있는 성능 차이를 드러내는 데 초점을 둡니다. 그 결과 EfficientNetB0이 전반 성능과 임상 성능 사이의 최적 균형을 보였다는 결론을 제시합니다.

- **Technical Challenges**: 의료 영상 분류에서는 클래스별로 병변이 미세하게 보이는 경우가 많아 단순 accuracy만으로는 실제 탐지력을 판단하기 어렵습니다. 저자들은 모든 모델을 동일한 실험 프레임워크에서 학습·평가하고, tumor-wise recall로 meningioma처럼 미세 병변의 성능을 정밀하게 비교함으로써 이 문제를 완화했습니다. 또한 모델의 깊이(depth)보다 아키텍처 효율성이 성능에 더 큰 영향을 줄 수 있음을 VGG16 vs VGG19 결과로 확인합니다.

- **Empirical Impact**: EfficientNetB0은 전체 분류 정확도 95%로 가장 높은 성능을 보였으며, VGG16 94.37%, VGG19 92.29%, DenseNet121 90.91%, 커스텀 CNN 78.00%과 비교해 격차가 나타났습니다. 특히 meningioma 재현율이 커스텀/단순 CNN 계열의 약 20%에서 EfficientNetB0의 89%로 크게 개선되어, 임상적으로 어려운 클래스에서의 실질적 이득을 보여줍니다. 의료 영상 분류에서 “깊이”보다 “효율적인 아키텍처”가 정확도·파라미터 수·임상 의미를 동시에 만족시킬 수 있다는 메시지를 뚜렷하게 전달합니다.



### Moving Beyond Diversity: Visual Token Pruning as Subspace Reconstruction for Efficient VLMs (https://arxiv.org/abs/2606.18681)
Comments:
          ECCV 2026 Under Review

- **Prior Approaches**: 기존 비전-언어 모델(VLM) 토큰 감축은 주로 diversity maximization에 기반하며, 코사인 유사도 기반의 정규화(normalized similarity)를 사용해 시각 토큰을 줄입니다. 하지만 이런 방식은 크기(magnitude) 정보를 버려 원래 feature 표현을 충실히 근사하지 못하고, 특히 compositional multi-skill reasoning 같은 조합형 과제에서 성능이 떨어집니다.

- **Core Contribution**: 본 논문은 token pruning을 “부분(컬럼) 선택” 관점의 column subset selection 문제로 다시 정의하고, 재구성 오차(reconstruction error)를 직접 최소화하는 subspace reconstruction 방식 SPARE를 제안합니다. 또한 “anti-relevance” 현상을 발견해, 일반적으로 점수가 낮은 토큰이 오히려 문맥 정보를 더 잘 보존할 수 있음을 실험적으로 보이고 이를 추가 선택 기준으로 통합합니다.

- **Technical Challenges**: SPARE의 핵심은 선택한 토큰들의 부분공간이 원본 표현을 얼마나 잘 복원하는지 정량화하고, 반복적으로 projection residual이 큰 토큰을 고르는 방식으로 재구성 기반 pruning을 구현하는 데 있습니다. 여기에 더해 anti-relevance 기준을 함께 사용해 단순 각도(angular) 다양성만으로는 놓치기 쉬운 문맥 보존을 강화하도록 설계합니다(학습 없이 동작하는 training-free 적용 포함).

- **Empirical Impact**: 여러 VLM과 벤치마크에서 SPARE는 일관되게 state-of-the-art를 달성하며, 특히 compositional 작업에서 큰 폭의 개선을 보입니다. LLaVA에 적용했을 때는 시각 토큰을 최대 94%까지 제거하면서도 기준선 성능의 95%를 유지하는 결과를 보고해, 연산 절감과 추론 품질을 동시에 확보할 수 있음을 보여줍니다.



### BrainFusionNet: a deep learning and XAI model to understand local, global, and sequential features of MRI images for improved brain tumour detection (https://arxiv.org/abs/2606.18675)
- **Prior Approaches**: 기존 MRI 기반 종양 분류 딥러닝은 CNN 위주로 국소 특징을 잘 잡지만, 복잡한 종양 경계가 잡음에 가려질 때 전역 문맥을 놓치기 쉽다는 한계가 지적돼 왔습니다. 또한 ViT 같은 전역 모델을 쓰더라도 MRI 학습에서 그래디언트 안정성이나 과적합 위험이 커서 성능이 흔들릴 수 있습니다. 마지막으로 대부분의 모델은 “왜 그렇게 판단했는지”를 설명하기 어렵거나, 설명이 실제 결정에 어떤 영향을 주는지 정량적으로 연결하기가 부족했습니다.

- **Core Contribution**: 이 논문은 BrainFusionNet으로 CNN, Vision Transformer(ViT), Gated Recurrent Unit(GRU)을 결합해 MRI에서 국소·전역·순차(표현 흐름) 특징을 함께 추출하도록 설계했습니다. 특히 작은 종양 크기에서도 국소와 전역 문맥을 효과적으로 매칭해 경계가 흐린 상황에서도 분류 정확도를 끌어올리는 것이 핵심 기여입니다. 여기에 SHAP, LIME, GradCAM 같은 explainable AI를 통합해 모델의 결정에 기여한 이미지 영역을 시각화합니다.

- **Technical Challenges**: 주요 기술 난관은 (1) 잡음과 복잡한 외관 때문에 경계 정보가 불안정하고, (2) 전역 표현을 학습하는 ViT에서 그래디언트 소실/학습 불안정이 발생할 수 있으며, (3) CNN과 ViT 출력을 GRU로 안정적으로 결합해 최종 분류로 연결해야 한다는 점입니다. 논문은 CNN과 “커스터마이즈드 ViT”로 로컬 특징을 더 안정적으로 학습해 그래디언트 흐름을 유지하고, CNN·ViT 출력은 GRU에 투입해 저수준부터 더 깊은 레이어까지의 특징을 균형 있게 반영하도록 했습니다. 또한 픽셀 강도(이미지 품질 지표로 해석)를 분석해 분포 차이가 분류 성능에 영향을 주는지도 함께 점검합니다.

- **Empirical Impact**: 실험에서는 두 개의 공개 MRI 데이터셋에서 k-fold 검증으로 각각 약 98% 정확도를 보고했으며, 6개 SOTA CNN 및 transfer learning과 비교해 성능을 입증했습니다. SOTA 비교에서는 DenseNet121과 VGG16이 최고 정확도 96%를 기록한 가운데, BrainFusionNet은 작은 종양에서도 국소·전역 결합이 잘 동작함을 강조합니다. 더 나아가 MRI 픽셀 강도 분포가 딥러닝 성능을 좌우한다는 관찰은 데이터 품질/전처리 관점의 해석을 확장해 향후 모델 설계와 평가에 의미가 있습니다.



### LandslideAgent with Multimodal LandslideBench: A Domain-Rule-Augmented Agent for Autonomous Landslide Identification and Analysis (https://arxiv.org/abs/2606.18661)
- **Prior Approaches**: 기존 원격탐지 기반 산사태 연구는 주로 CNN/ViT류로 시각 특징을 뽑아 객체 검출이나 semantic segmentation으로 경계(박스/마스크)를 찾는 데 집중해 왔다. 그러나 이런 방법들은 지형-환경 맥락, 유발 요인 같은 고수준 지리과학 의미 추론이 약하고, 데이터도 대개 이진 라벨 중심이라 few-shot 상황에서 일반화가 어렵다. VLM/LLM 기반 접근은 보고서를 생성하더라도 복잡한 지질 장면에서 perceptual 한계와 domain hallucination을 겪어 정밀 도구 호출과 일관된 추론이 무너질 수 있다.

- **Core Contribution**: 이 논문은 산사태 분석을 “인식→의미 추론→근거 기반 보고서 생성”으로 전환하는 instruction-driven agentic 프레임워크를 제안한다. 핵심은 (1) fine-grained 멀티모달 데이터셋 LandslideBench, (2) 산사태 전용 VLM LandslideVLM, (3) LandslideVLM을 두뇌로 하되 도메인 규칙으로 tool invocation을 제어하는 LandslideAgent의 3단 구성이다. 이를 통해 시각적 경계 산출과 지리과학 의미 생성, 그리고 도구 기반 검증을 한 시스템에서 연결한다.

- **Technical Challenges**: 주된 기술 난제는 지질 도메인에서 시각-언어 정렬 데이터가 부족하고, 일반 VLM은 섬세한 지형 증거를 건너뛰거나 그럴듯한(hallucination) 서술을 만들 수 있다는 점이다. 논문은 이를 LandslideBench의 정밀한 image-mask-text 3중 정렬(다중 검증/전문가 검수, 다중 VLM 교차검증)과 LoRA 기반 도메인 적응 fine-tuning으로 해결해 시각-지리 의미 정합성을 강화한다. 또한 LandslideAgent에는 메타데이터 의존 규칙(보고서 스키마를 맞추기 위한 필수 중간 산출물 강제)과 교차모델 기반 인식 규칙(세그멘테이션과 VLM 교차검증, 누락 방지)을 넣어 논리적 비약과 불필요한 도구 호출을 줄인다.

- **Empirical Impact**: 실험에서 LandslideBench는 5개 주요 모델에 대해 fine-grained 분류와 semantic segmentation의 기준선(baseline)으로 효과가 있음을 보인다. LandslideVLM은 landslide discrimination, fine-grained classification, semantic description quality에서 각각 10.96%, 32.87%, 15.91%의 정확도 향상을 기록한다. 더 나아가 LandslideAgent는 멀티소스 공간 데이터 추론을 자율 수행해 산사태 식별·분석의 end-to-end “full-process intelligence”를 실증하며 재난 지질 분야의 신뢰성 있는 자동화 방향을 제시한다.



### On-Manifold Variational Learning with Heat-Kernel Priors (https://arxiv.org/abs/2606.18658)
- **Prior Approaches**: 기존 비지도 의료영상 표현학습은 잡음이 많은 진단 라벨을 대체하려 하지만, 주로 클러스터 구조나 생성적 프로토타입을 충분히 분리하지 못하는 한계가 있었다. 특히 GMM 계열의 잠재변수 모델은 유클리드 평균으로 프로토타입을 만들면서 곡면(manifold)을 벗어나거나 컴포넌트가 겹치며(중복·퇴화) sub-population 수가 늘수록 품질이 급격히 떨어진다. 반면 대비학습·VAE·hybrid variational Gaussian mixture는 부분적으로는 강점을 보이나, “라벨프리 + manifold-aware 프로토타입 + 생성 기반”을 동시에 만족시키기 어렵다.

- **Core Contribution**: 이 논문은 manifold-anchored variational framework로 프로토타입이 데이터 곡면 위에 “고정”되도록 설계해 off-manifold drift와 컴포넌트 퇴화를 줄인다. 핵심은 geometry-aware EM(Manifold-anchored EM)에서 M-step이 각 sub-population의 프로토타입을 heat-kernel affinity 관점의 graph medoid로 선택해 on-manifold 성질을 보장하는 것이다. 또한 Dirichlet energy 정규화로 잠재공간의 기하적 부드러움을 유지하고, sub-population별 불확실도 점수로 레이블 없이 품질을 자동 평가한다.

- **Technical Challenges**: 어려움은 GMM의 통계적 일관성을 유지하면서도 유클리드 평균 대신 곡면 제약을 M-step에 실제로 통합하는 것이다. 논문은 잠재 임베딩에서 heat-kernel 가중 그래프를 만들고, Dirichlet energy로 이웃 간 급격한 변화를 억제해 diffusion 기하를 보존한다. 더불어 프로토타입은 hard assignment 후보 집합에서 medoid를 고르되, 공분산은 soft responsibility를 사용해 경계 샘플의 퍼짐을 반영함으로써 학습 초기의 할당 불확실성에도 강인하게 한다.

- **Empirical Impact**: cardiac scar와 brain MRI(OASIS) 벤치마크에서 제안 방법은 비교 모델 중 최고 정확도를 달성하고, 지금까지 보고된 프로토타입 중 가장 선명한(atlas의 형태·경계 보존이 좋음) 결과를 만든다. 또한 sub-population 수가 커질 때 기존 baseline이 degeneracy로 무너지는 반면, 이 방법은 안정적으로 성능을 유지한다. 더불어 불확실도 맵이 병리적으로 민감한 영역에 구조적으로 집중되며 과신(overconfident) 대신 의미 있는 변동을 보존해, 라벨 없는 임상적 층화에 실용적인 “품질 플래그”를 제공한다.



### Spiking Pyramid Wavelet Transformation for High-efficient and Low-energy Image Restoration (https://arxiv.org/abs/2606.18644)
Comments:
          Accepted by Pattern Recognition

- **Prior Approaches**: 기존 이미지 복원(IR) 딥러닝은 CNN 기반과 Transformer 기반이 주류였지만, CNN은 수용영역이 지역에 고정되고 Transformer는 self-attention의 연산·메모리 비용이 커 효율-성능 균형이 어렵다는 한계가 있었다. SNN 기반 방법도 제안됐으나, spiking CNN의 경우 CNN의 국소 수용영역 제약 때문에 성능이 더 올라가기 어렵다. 또한 스파이킹 Transformer를 그대로 쓰면 장거리 의존성은 해결되더라도 FLOPs와 파라미터 비용이 증가해 “고효율” 목표와 충돌한다.

- **Core Contribution**: 이 논문은 DWT(이산 웨이블 변환)로 분해된 주파수 특성과 random shuffle을 활용해 장거리 의존성을 저비용으로 모델링하는 spiking pyramid wavelet-based model(SPWM)을 제안한다. 핵심은 spiking dual pyramid wavelet(SDPW) block으로, spiking pyramid wavelet unit(SPWU)과 spiking pyramid shuffle unit(SPSU)를 통해 웨이블 도메인에서의 복원 정보를 더 잘 결합한다. 나아가 MDA(multi-dimensional attention)와 multi-scale progressive fusion, SCAM(초경량 공간-채널 조정 모듈)을 넣어 스파이크의 이진화로 생기는 정보 손실을 완화한다.

- **Technical Challenges**: SNN에서 입력을 스파이크로 인코딩할 때 이진 활성로 인해 구조 정보가 손실되기 쉬우며, 단순 평균 샘플링 같은 시간 집계는 성능을 떨어뜨릴 수 있다. 저자들은(1) Poisson이 아닌 direct encoding으로 시간마다 동일 입력을 반복해 막전위 동역학을 활용하고, (2) MDA로 스파이크 반응을 가변적으로 조정하며, (3) 다중 스케일 progressive fusion과 SCAM으로 이산 펄스에서 연속 픽셀 값으로의 변환 과정에서의 정보 누락을 줄인다. 또한 DWT로 주파수별 열화를 분리한 뒤 pyramid 구조의 웨이블 단계와 RS(랜덤 셔플), inverse shuffle을 조합해 장거리 상관을 표준 convolution으로도 학습 가능하게 설계한다.

- **Empirical Impact**: 여러 IR 벤치마크(비/강우 제거 Rain200L·Rain200H·Rain1200, 저조도 향상 LOL, 디헤이징 Dense-Haze)에서 SPWM은 ANN 및 SNN 기반 경쟁 방법 대비 PSNR·SSIM을 개선하면서 파라미터와 FLOPs 부담을 낮추는 결과를 보였다. 특히 Rain200H에서 ESDNet 대비 PSNR이 0.99 dB 향상되며, 정성 비교에서도 대비와 색 왜곡이 줄고 세부가 더 잘 보존되는 경향이 확인됐다. 더불어 시간 스텝·DWT 레벨 및 구성요소에 대한 ablation은 각 모듈(MDA, MSPS, SCAM, SPWU/SPSU)의 기여를 뒷받침하며, 리소스 제한 장치용 IR에서 SNN 설계에 대한 새로운 방향성을 제시한다.



### Intrinsic 4D Gaussian Segmentation from Scene Cues (https://arxiv.org/abs/2606.18623)
Comments:
          15 pages, 4 figures, 7 tables. Includes supplementary material. Preprint

- **Prior Approaches**: 기존 4D Gaussian Splatting(4DGS) 세그멘테이션은 SAM 같은 2D foundation model의 마스크(또는 특징)를 프레임·뷰 전반에 걸쳐 생성한 뒤, 이를 Gaussians로 lift/track/distill하는 방식이 주류였습니다. 특히 동적 장면에서는 마스크·특징을 매 프레임 다시 만들고, 경우에 따라 추가 feature field까지 학습해야 해 비용과 일관성 의존도가 커집니다. 그 결과 외부 마스크 품질이 흔들리면 객체 경계가 크게 달라질 수 있다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Gaussians 자체에 내재된 인트린직(intrinsic) 단서를 바탕으로 객체 단위를 복원하는 Intrinsic-GS를 제안합니다. training-free, mask-free, foundation-model-free로, appearance·orientation·scale·deformation-trajectory·rendered-boundary 신호를 조합해 Gaussians 간 sparse affinity graph를 만든 뒤 Leiden community detection으로 분할합니다. 즉 “대표값은 마스크로 만들고 Gaussians는 그 결과를 담는 그릇”이라는 기존 패러다임을 “표현 자체가 이미 객체 구조를 품는가?”로 뒤집어 본 점이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 마스크 없이도 객체 분리를 안정적으로 유도할 만큼, 약한 인트린직 단서들을 신뢰도 있게 그래프 간선 가중치로 융합하는 것입니다. 논문은 geometry(색·방향·스케일), motion(공통-fate 관점의 deformation trajectory 일치), 그리고 rendered-boundary(학습이 아닌 비학습 경계 단서의 suppression)를 multi-modal edge weight로 ‘soft and’처럼 결합하고, kNN 기반의 희소 그래프와 고정 해상도 Leiden로 대규모 그래프를 효율 분할합니다. 또한 근거리 kNN으로 생기는 객체 분할 문제를 long-range objectness merge(trajectory covisibility로 커뮤니티 통합)로 완화해 전역 일관성을 보강합니다.

- **Empirical Impact**: 벤치마크에서 Intrinsic-GS는 Neu3D와 HyperNeRF에서 mask supervision 없이도 유의미한 객체 구조를 복원하며 mIoU 0.746(Neu3D), 0.575(HyperNeRF)를 기록했습니다. 더 나아가 geometry-only 변형은 Neu3D에서 0.902 mIoU로, SAM-supervised TRASE와 맞먹는 성능을 보여 “분할 신호가 Gaussians 내부에 상당 부분 이미 존재”함을 시사합니다. 계산 효율에서도 HyperNeRF에서 mask-generation과 feature-rendering 중심 파이프라인 대비 12.5x 빨라, 동적 장면에서 외부 마스크 의존도를 낮추려는 실무적 가치가 큽니다.



### Hallucination Detection and Correction in Medical VLMs via Counter-Evidence Verification (https://arxiv.org/abs/2606.18609)
Comments:
          MICCAI 2026 Accept. Submission Version

- **Prior Approaches**: 기존 의료 VLM 환각(hallucination) 탐지는 주로 생성 문장과 기준 데이터 간 불일치, 또는 uncertainty·cross-model consistency 같은 신뢰도 신호에 의존합니다. 또한 attention/살리언시 시각화는 “어디를 봤는지” 직관을 주지만, 특정 문장이 실제 시각 증거에 인과적으로 의존하는지까지 검증하진 못합니다. 그 결과 의료에서는 문장이 그럴듯해도 시각 근거가 없는 진술을 놓치기 쉽습니다.

- **Core Contribution**: 이 논문은 Counter-Evidence Verification(CoEV)을 제안해 환각을 “탐지”를 넘어 “수정”까지 하는 학습 없는 plug-and-play 프레임워크로 만듭니다. CoEV는 텍스트 주장과 이미지 시각 증거를 양방향으로 검증하고, 각 문장을 ‘사실성’과 ‘시각 grounding’의 조합에 따라 4분면 진단 맵으로 분류합니다. 이를 통해 임상의가 어떤 주장에 문제가 있고 왜 그런지 증거 기반 단서로 추적할 수 있게 합니다.

- **Technical Challenges**: 핵심 난제는 attention 기반 해석이 사실 정합성과 grounding을 실제로 보장하지 않는다는 점입니다. CoEV는 이를 해결하기 위해, 해당 주장과 연결된 시각 영역을 마스킹한 뒤에도 문장이 유지되는지(시각 의존성)와 사실 정확성의 동시 변화를 관찰하는 counterfactual probing을 수행합니다. 이후 Med-VQA는 검증/보정, MRG(의무기록 생성)는 환각으로 판정된 문장만 LLM 프롬프트로 재작성해 후처리 refinement을 적용합니다.

- **Empirical Impact**: 4개 의료 데이터셋과 다양한 VLM에 대한 실험에서 CoEV는 환각 탐지 성능을 꾸준히 개선하며, 평균 PR-AUC와 ROC-AUC를 각각 절대 3.0%, 3.9%p 향상시킵니다. 특히 특정 VQA 시나리오에서는 최대 18.5% 개선, 문장/보고서 수준에서는 Micro-F1 최대 12.5% 향상과 환각률 11.9% 이상 감소를 보였습니다. 결과적으로 CoEV는 의료 진단 보조에서 신뢰 가능한 증거 기반 환각 완화 및 정확도 향상에 실질적 의미를 제공한다는 점을 입증합니다.



### Bridging Creative Intent and Visual Quality: Creator-Driven Recurrent Video Generation with Agentic Feedback Loops (https://arxiv.org/abs/2606.18591)
Comments:
          Accepted to the Workshop on Human-AI Co-Creativity at ICML 2026

- **Prior Approaches**: 기존 비디오 생성은 코딩처럼 closed-loop 반복 개선을 목표로 하지만, 비디오에는 객관적 자동 신호(예: 테스트)가 없어 사람이 만든 reward model이나 여러 LLM judge의 ‘집계된 선호’에 의존하는 경우가 많았다. 또 이런 접근은 자동으로 자체 개선하는 구조를 그대로 가져와, 창작자가 서사와 연출 방향을 주도하는 ‘creative task’의 특성을 충분히 반영하지 못한다. 그 결과 장편으로 갈수록 서사 일관성과 창의적 방향성이 쉽게 무너진다.

- **Core Contribution**: CHIEF는 인간-인-더-루프(human-in-the-loop) 방식으로 창작자가 매 반복을 주도하고, 에이전트가 이를 정제(refine)하도록 설계한 비디오 공동창작 프레임워크다. 또한 persona-conditioned 멀티모달 LLM 피드백 에이전트가 생성 영상을 ‘관객 관점’에서 주관적 비평으로 자동 피드백하며, 자기평가만으로는 포착 어려운 감정·수용자 반응을 보완한다. 창작자가 자연어로 수정 방향을 주면, Feedback Translator가 이를 다음 반복용 프롬프트 개선으로 구조화해 연결한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 비디오를 객관적으로 평가할 수 있는 신호가 없다는 점과 (2) 반복이 길어질수록 의미 drift·불일치·새 잡음(아티팩트)이 생긴다는 점이다. CHIEF는 이를 위해 스크립트를 8초 클립 설명으로 분해하고 keyframe→보간(interpolation) 기반으로 국소 수정이 가능하게 하며, Feedback Translator가 다수 persona의 피드백을 이슈 튜플로 쪼개 우선순위를 랭킹한다. 장편(10분)에서는 creator-gated refinement로 재생성 트리거를 창작자가 승인하도록 막아 연쇄 변화로 인한 연속성 붕괴를 줄인다.

- **Empirical Impact**: 표준 벤치마크가 아닌 창작 협업 맥락에서, 1분짜리 영상은 반복 과정에서 시각·서사·분위기 결함이 축적적으로 수정되는 사례를 보여준다. 또한 영화 제작 경험이 없는 학생들이 10분 단편을 만들어, 제작 전후 버전을 고교 현장 관객이 4.1/5 vs 2.4/5로 평가해 CHIEF 버전의 몰입도·이해도 향상을 확인했다. 요약하면 CHIEF는 ‘관객 관점의 주관적 피드백’과 ‘창작자 주도 제어’를 결합해 장편 비디오 일관성 문제를 다루는 새로운 실전형 방향을 제시한다.



### APT: Atomic Physical Transitions for Causal Video-Language Understanding (https://arxiv.org/abs/2606.18586)
- **Prior Approaches**: 기존 물리 비디오 연구와 벤치마크는 “bounce” 같은 클립 수준 event 라벨이나 결과 중심의 VQA/생성/플라우저빌리티에 많은 감독이 묶여 있었다. 이런 방식은 무엇이 일어났는지는 맞아도, 지지 상실·접촉 시작·반동·정착 같은 인과적 상태 전이가 어떻게 연결되는지는 과소평가되기 쉽다. 또 VLM이 이벤트를 맞춘다고 해서 전이 단위의 물리 메커니즘을 복구한다는 보장은 부족했다.

- **Core Contribution**: 논문은 동영상의 숨은 과정을 “Atomic Physical Transitions(APTs)”라는 전이 단위의 인과 체인으로 명시화한다. APT는 시각적으로 국소화된 최소 상태 변화로, 전후 동역학 regime 전환과 “왜 그런지”를 지지하는 활성 물리 메커니즘을 타입화한다. 그 결과, 영상 이해를 단일 event 라벨이 아니라 ordered APT chain으로 표현해 전이 타임스탬프와 타입을 동시에 복원하도록 만든다.

- **Technical Challenges**: 문제는 APT 같은 전이 경계 감독이 수동으로는 얻기 어렵고, 픽셀만으로는 전이 경계를 안정적으로 추론하기 힘들다는 점이다. 저자들은 인간 라벨(CLEVRER/Physion++)로 14개 전이 타입의 의미를 캘리브레이션하고, 시뮬레이터(Physion++·Phys4D/Isaac Sim 계열) GT 트레이스에 기반해 rule-guided coarse-to-fine 파이프라인으로 mixed-source APT 데이터를 구성한다. 학습에서는 APT-JSON 같은 포맷으로만 단순화하면 “포맷 특화(specialist collapse)”와 event-level 망각이 발생해, 이를 막기 위한 parameter-efficient 미세조정 레시피 APT-Tune을 제안한다.

- **Empirical Impact**: 평가 결과, 8종의 VLM은 동일 입력과 APT 스키마 프롬프트 조건에서도 zero-shot APT recall이 10~14%에 그쳤고, 주된 실패 원인은 타이밍 지터보다 전이를 놓치는 것이었다. 반면 APT-Tune은 Qwen3-VL-2B에서 LoRA 11M 파라미터만으로 APT recall을 38.1%까지 끌어올리며, 최종적으로는 다른 백본에서 최대 53.4% recall을 달성했다. 또한 APT-Tune은 MVBench(이벤트 수준)에서 소폭 향상(일괄적으로 +0.5~2.2pp)과 PhysBench(실세계 물리 이해)에서 최대 +18.2pp 개선을 보여, 전이 수준 물리 습득이 답변 포맷 편향으로 끝나지 않고 전이적 재사용 가능한 물리 표현으로 이어짐을 시사한다.



### Aerial-ground LiDAR place recognition with patch-level self-supervised learning and expanded reciprocal re-ranking (https://arxiv.org/abs/2606.18583)
- **Prior Approaches**: 기존 ground-level LiDAR place recognition은 사전 방문(pre-visit)과 지도 범위의 불완전성, 시점 다양성 부족 때문에 확장성에 한계가 있었다. 또한 patch 단위 표현 학습을 충분히 활용하지 않고 scene-level metric learning 중심으로 학습하는 접근이 많아, 항공-지상 cross-view에서 발생하는 domain gap과 초기 검색의 false positives에 취약했다. 재랭킹 단계 역시 feature 거리만으로 처리하는 경우가 많아 ALS의 공간적 구조를 제대로 쓰지 못했다.

- **Core Contribution**: 논문은 항공의 full-coverage Airborne Laser Scanning(ALS) 지도를 aerial prior map으로 두고, 지상 쿼리를 항공 서브맵으로 매칭하는 cross-view LiDAR place recognition 프레임워크를 제안한다. 핵심은 patch-level self-supervised learning(다중 스케일)으로 항공/지상 간 global feature의 판별성을 키우는 retrieval 네트워크와, 추가 학습 없이도 neighborhood 정보를 극대화해 false positives를 줄이는 Expanded Reciprocal(ER) re-ranking이다.

- **Technical Challenges**: 가장 큰 기술 난제는 항공-지상 점군의 domain gap으로 인해 정확한 point correspondence가 부족하고 글로벌 특징이 쉽게 흔들린다는 점이다. 이를 위해 OctFormer 기반 백본에서 여러 깊이(octree scale)의 patch를 뽑고, 인접 패치의 유사 의미를 가정해 teacher-student cross-attention self-distillation으로 patch 표현을 정렬한다. 두 번째 난제인 초기 검색의 반복 구조로 인한 false positives는 ALS 점군의 구조적 공간 분포를 이용해 reciprocal 이웃을 확장하고, 이웃 평균으로 특징을 갱신한 뒤 최종 거리 행렬을 업데이트하는 방식(ER)으로 완화한다.

- **Empirical Impact**: CS-Urban-Scenes에서 retrieval 네트워크는 평균 Recall@1을 9.8% 개선했고 평균 Recall@1%도 3.2% 향상시켰으며 CS-Campus3D에서도 최고 성능을 보였다. 더 나아가 ER re-ranking은 CS-Campus3D에서 평균 Recall@1을 4.9% 추가로, CS-Urban-Scenes에서는 10.2% 추가로 끌어올리되 추가 학습 없이 적용 가능함을 보여준다. 결과적으로 항공-지상 cross-view LiDAR 기반 대규모 모바일 매핑/자율주행 위치추정에서 “학습 기반 검색 + 학습 없는 구조 기반 재랭킹” 조합의 실용성을 강화한 것으로 평가된다.



### Technical Report for ICRA 2026 GOOSE 2D Fine-Grained Semantic Segmentation Challenge: Leveraging DINOv3 for Robust Outdoor Scene Understanding in Field Robotics (https://arxiv.org/abs/2606.18582)
Comments:
          5 pages, 4 figures

- **Prior Approaches**: 기존 필드 로보틱스용 세그멘테이션 벤치마크는 상대적으로 정형화된 실내·도시 장면에 치우쳐, 비정형 오프로드와 플랫폼/시점 변화(차량·굴착기·사족 등), 장거리 스케일 변동, 클래스 장기꼬리 문제를 충분히 반영하기 어렵습니다. 또한 Cityscapes/ADE20K류 일반 장면 파싱 모델을 그대로 옮기면 64개 파인 클래스의 시각적 애매함과 희소성(희귀 라벨은 픽셀 단위로 극히 적음)에서 성능이 흔들리기 쉽습니다.

- **Core Contribution**: 논문은 GOOSE 2D Fine-Grained Semantic Segmentation Challenge에서 1위를 달성한 최초의 정밀 오프로드 2D 세그멘테이션 솔루션을 제시합니다. 핵심은 DINOv3 ViT-L/16 백본+ViT-Adapter+Mask2Former의 조합에 더해, DINOv3의 global [CLS] 토큰에 11개 coarse 카테고리 존재를 예측하는 multi-hot 보조 손실을 붙여 전역 의미 인식을 강화한 점입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 64개 파인 클래스의 장기꼬리·국소 애매함과, 서로 다른 로봇 플랫폼이 만드는 카메라 기하/시점 차이를 동시에 커버하는 것입니다. 저자들은 단일 해상도 ViT 토큰을 Mask2Former가 요구하는 멀티스케일 피처 피라미드로 변환하는 ViT-Adapter(Deformable cross-attention 기반)로 아키텍처 간 불일치를 해결하고, 보조 손실은 픽셀 위치가 아니라 coarse 카테고리 “존재”만 학습하도록 설계해 주 손실을 흔들지 않게 했습니다.

- **Empirical Impact**: 실험 결과 dev split에서 ConvNeXt+Mask2Former 기준 56.68% 대비 DINOv3+ViT-Adapter로 69.80%까지 크게 상승했으며, [CLS] multi-hot 보조 손실과 TTA/체크포인트 앙상블을 누적해 최종 composite mIoU 76.57%(fine-class mIoU 69.32%, category-level mIoU 83.81%)를 기록했습니다. 이 성과는 파인-그레인드 라벨 희소성과 플랫폼 변동이 큰 필드 로보틱스 세그멘테이션에서 전역 의미 정규화와 멀티스케일 쿼리 기반 디코딩이 실질적 개선으로 이어질 수 있음을 보여주는 사례로 평가됩니다.



### Multi-Modal Hyper-Graph Fusion for Low-Light Crowd Counting (https://arxiv.org/abs/2606.18566)
- **Prior Approaches**: 기존 군집수( crowd counting ) 연구는 주로 주간/양호 조도 환경에 최적화돼 있으며, 밀집 장면에서는 밀도맵 기반과 포인트 기반(머리 중심 예측) 패러다임이 주류였습니다. 저조도에서는 RGB 단일 모달이 조도 변화·잡음·대비 저하에 취약해 성능이 급락하고, enhance-then-count처럼 향상과 카운팅을 분리한 파이프라인은 잡음 증폭이나 구조 디테일 손실로 오차가 누적되는 문제가 있었습니다. 또한 전용 저조도 벤치마크와 tailored 방법이 부족해, 저조도 상황의 검증이 제한적이었습니다.

- **Core Contribution**: 이 논문은 저조도 군집수 인식을 위해 세 가지 저조도 벤치마크를 새로 구축하고(합성 2종 SHA_Dark/SHB_Dark, 실세계 LC-Crowd), 모델도 저조도용으로 통합 설계합니다. 핵심 아이디어는 Retinex 관점으로 저조도 카운팅을 ‘반사율 관련 표현의 재보정(reflectance re-calibration)’ 문제로 정식화하고, RGB의 조명 성분이 망가질 때 depth와 Canny edge의 기하·구조 단서를 통해 보정 가능성을 보인 것입니다. 이를 바탕으로 RGB 외 모달리티를 단순 보조로 쓰는 수준을 넘어, 고차 관계를 이용한 융합과 계산 효율을 함께 달성하는 LCNet을 제안합니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 극단적 어둠에서 단일 단서(depth나 edge)가 국소적으로 붕괴해 pairwise(쌍단) 결합이 오히려 오염될 수 있다는 점입니다. 논문은 Multi-Modal Hyper-Graph Fusion에서 RGB 외 appearance, depth geometry, edge structure 토큰을 동일 hyper-graph로 묶고, 동적 hyperedge 구성+메시지 패싱으로 여러 이웃의 ‘동시 합의’에 기반한 고차 보정을 수행해 단일 단서 실패에 대한 강건성을 높입니다. 또한 야간 장면에서 전경이 희소하다는 점을 고려해 Deformable Rectangular Sparse Attention(DRSA)으로 영역별 granularity를 추정한 뒤 선택된 anchor에만 deformable rectangular attention을 적용, 연산을 정보가 있는 곳에 집중시킵니다.

- **Empirical Impact**: SHA_Dark, SHB_Dark, LC-Crowd의 세 벤치마크에서 제안 방법은 기존 SOTA 대비 전반적 성능이 가장 좋다고 보고되며, 저조도에서의 실질적 견고성이 확인됩니다. 특히 실세계 LC-Crowd는 조도 분포의 비균일성이 크고 해상도·군집 규모 변동이 폭넓어, 단순 합성 열화만으로는 검증하기 어려운 현실 난제를 다룹니다. 결과적으로 저조도 군집수 연구에서 ‘전용 데이터+모달 재보정+고차 융합+연산 적응’이라는 설계 방향을 제시하며, 후속 연구들이 비교·확장하기 쉬운 기반을 제공한다는 점에서 의미가 큽니다.



### Experimental Analysis of Neural Network-Based Image Classification on the CIFAR-10 Datas (https://arxiv.org/abs/2606.18565)
Comments:
          7 pages

- **Prior Approaches**: 기존 연구는 CIFAR-10 같은 저해상도 이미지 분류에서 MLP와 CNN을 비교하며, MLP는 픽셀을 벡터로 펼친 뒤 전역 연결로 학습하고 CNN은 지역 수용영역과 weight sharing으로 공간 구조를 보존한다고 설명해왔다. 다만 단순히 정확도만 보고 끝내면 학습이 ‘표현 학습’인지 ‘암기(과적합)’인지 구분하기 어렵고, validation loss-accuracy의 동학을 충분히 분석하지 못한 경우가 많았다. 또한 재현 가능한 실험 프로토콜(데이터 분할, 로깅, 하이퍼파라미터, 조기 종료 기준)이 제대로 정리되지 않으면 후속 비교가 흔들린다.

- **Core Contribution**: 이 논문은 CIFAR-10에서 신경망 이미지 분류의 전체 지도학습 파이프라인(전처리, 정규화, one-hot 인코딩, softmax+cross-entropy, mini-batch 학습, validation 기반 모델 선택)을 ‘학술적 재구성’ 형태로 명확히 정리한다. 특히 학습 손실은 계속 감소하지만 validation loss가 중반 이후 증가하는 패턴을 중심으로, 표현 학습과 암기의 실무적 차이를 해석하는 기준을 제시한다. 결과적으로 향후 정규화, 데이터 증강, 더 깊은 아키텍처를 검증할 수 있는 compact baseline을 제공한다.

- **Technical Challenges**: 핵심 난제는 작은 CIFAR-10에서 모델이 훈련 데이터에 맞춰 과적합되기 쉬운 상황을, 짧은 epoch 동안도 안정적으로 관찰·판단하는 것이다. 이를 위해 입력 정규화와 one-hot 라벨, softmax 기반 categorical cross-entropy를 일관되게 적용하고, Adam(learning rate 0.001), batch size 128, epoch별 validation 로깅으로 validation loss의 상승 시점을 추적한다. 또한 CNN은 6개 convolution 레이어와 3회 max-pooling으로 지역 특징 추출과 다운샘플링을 균형 있게 구성해, MLP 대비 일반화에 유리한 귀납적 편향을 확보한다.

- **Empirical Impact**: 실험에서 CNN은 10 epoch 학습 후 validation accuracy가 약 74.77%까지 도달하며, best 성능이 6 epoch 부근에서 나타난다. 반면 training loss는 단조 감소하지만 validation loss는 중반 이후 다시 증가하고, 6 epoch 이후에는 정확도도 소폭 하락(예: 74.77%→73.60%)해 ‘더 오래 학습=더 잘 일반화’가 아님을 보여준다. 논문은 정규화·데이터 증강·깊이 확장·reproducible 교육 실험에 바로 연결될 수 있는 기준선과 해석 프레임을 제공한다.



### MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction (https://arxiv.org/abs/2606.18558)
- **Prior Approaches**: 기존 모션 예측은 주로 과거에 이미 일어난 움직임을 추정하는 Optical flow·트래킹 중심이었고, 미래를 “무엇이 어디로 갈지”까지 일반화해 예측하는 데는 한계가 있었다. 또한 픽셀 기반 접근은 미래를 풍부하게 생성할 수 있지만 비용이 크고, 하류 태스크에서 활용하기 어렵다. 2D point는 카메라 에고모션과 뷰 변화에 얽혀 전이·분리가 어려웠고, 3D point를 쓰더라도 범주(사람/손/강체 등)에 맞춘 파라메트릭 모델에 종속되는 경우가 많았다.

- **Core Contribution**: 이 논문은 세계 좌표계에 고정된 object-attached 3D points를 일반적인 모션 표현으로 제안하고, class-agnostic·view-stable하며 물리 상호작용에 바로 쓸 수 있다고 주장한다. 언어로 “목표(행동) 조건”을 주면, 모델이 관심 객체 위의 여러 3D query point 각각에 대해 미래 3D 궤적을 예측하는 goal-conditioned 3D point motion forecasting 문제를 정식화한다. 또한 대규모 연구를 위한 데이터/평가/모델 풀스택으로 MolmoMotion-1M, PointMotionBench, 그리고 MolmoMotion(autoregressive + flow matching)을 함께 제시한다.

- **Technical Challenges**: 핵심 난관은 3D 모션 감독 데이터의 부족인데, 인터넷 영상은 규모는 크지만 3D 주석이 없어 학습이 제한됐다. 이를 위해 무구속 영상에서 object-grounded 3D point trajectories를 자동 추출하는 파이프라인을 설계해 1.16M 클립 규모의 MolmoMotion-1M을 만들고, 벤치마크는 가능한 곳은 GT 3D 캡처를 쓰되 나머지는 사람 검증으로 신뢰도를 확보했다. 모델 측면에서는 언어-시각 grounding과 시간적 일관된 궤적 생성이 어려워, Molmo2 기반 입력 인코딩 위에 autoregressive 좌표열 예측과 flow-matching 기반 연속 궤적 생성 두 가지 학습 목표를 보완적으로 적용했다.

- **Empirical Impact**: PointMotionBench에서 MolmoMotion은 기존 motion prediction baseline을 크게 능가하며, 특히 autoregressive 변형이 결정론적(평균/최종 오차) 지표에서 더 강하게 나타났다. 더 중요한 점은 learned 3D motion prior가 로봇 조작과 영상 생성 같은 하류 태스크로 잘 전이되어, pick-and-place에서 학습 효율과 closed-loop 성공률, 그리고 unseen 일반화가 함께 개선됐다는 것이다. 또한 예측된 3D 궤적을 조건으로 video generation을 유도하면 생성된 물체 움직임이 더 현실적이면서 정량 지표도 향상되는 방향을 보였다.



### Rethinking Text-to-Image as Semantic-Aware Data Augmentation for Indoor Scene Recognition (https://arxiv.org/abs/2606.18555)
Comments:
          MAPR 2024

- **Prior Approaches**: 기존 실내 이미지 인식은 데이터 부족에서 과적합과 일반화 저하가 자주 발생하며, 보통 rotation/translation/brightness 같은 전통적 데이터 증강이나 Mixup, CutMix로 이를 완화해 왔습니다. 하지만 이런 방법은 조명·가림·반사·복잡한 실내 배치처럼 장면의 미세한 변화를 충분히 재현하기 어렵습니다. 또한 생성형 모델을 텍스트-투-이미지로 결합한 연구는 존재하지만, 실내 장면의 다양성과 현실감을 증강 데이터로 일관되게 확보하는 데 한계가 남아 있습니다.

- **Core Contribution**: 이 논문은 Stable Diffusion(SD)으로 텍스트 기반의 ‘현실적인 실내 장면’ 합성 이미지를 만들어 학습 데이터를 확장하는 프레임워크를 제안합니다. 단순 생성이 아니라 CLIP으로 클래스에 맥락을 맞춘 프롬프트를 만들고, 중복·이상치를 제거해 학습 신뢰도를 높였습니다. 아울러 SD 합성 이미지의 오남용을 막기 위해 DIffusion Reconstruction Error(DIRE) 기반의 생성 여부 탐지기를 함께 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) SD가 만드는 합성이 실내 장면의 복잡한 변화를 실제 데이터처럼 담는가, (2) 생성 결과가 서로 중복되거나 부적절할 때 이를 어떻게 걸러내는가, (3) 합성 이미지 탐지기가 RGB만으로는 일반화가 떨어지는 문제를 어떻게 줄이는가입니다. 이들은 CLIP 기반 이미지 특징으로 cosine similarity를 계산해 중복/이상치를 분리하는 ‘duplicated and outlier removal’과, RGB 대신 DIRE 표현만으로 이진 분류기를 학습해 경량 모델로도 견고함을 확보하는 방식으로 해결합니다.

- **Empirical Impact**: MIT Indoor Scene에서 생성 증강을 1:1로 섞어 학습한 EfficientNetV2는 약 0.7% 향상(약 84%)을 보였고, 실제 데이터가 50%로 줄었을 때는 최대 1.9%까지 개선된 평균 성능을 보고했습니다. 합성 이미지 탐지에서는 MobilenetV3(약 2500 파라미터 수준의 마지막 레이어 파인튜닝 설정)가 DIRE로 학습될 때 100% 정확도를 달성해, 더 큰 네트워크 대비도 강건함을 입증했습니다. 실내 인식 성능 향상과 함께 ‘합성 이미지 감지’까지 연결해, 생성형 증강의 실사용 가능성과 안전성 논의에 신호를 주는 연구로 평가됩니다.



### Forged Calamity: Benchmark for Cross-Domain Synthetic Disaster Detection in the Age of Diffusion (https://arxiv.org/abs/2606.18554)
Comments:
          SOICT 2025

- **Prior Approaches**: 기존 딥페이크/합성 이미지 탐지는 GAN 시대의 흔적인 주파수 불일치나 카메라 잡음(PRNU) 같은 단서에 의존해 성능이 좋았지만, diffusion 계열에는 약해졌습니다. 재난(홍수·화재·지진·뇌우) 관련 벤치마크는 주로 장면 이해·피해 평가 등으로 설계돼 “진짜/가짜” 판별 문제와는 거리가 있었습니다. 또한 fine-tuning 기반 탐지기는 특정 생성기 아티팩트에 과적합하기 쉬워, unseen diffusion 모델이나 새로운 재난 유형으로 갈수록 성능이 흔들린다는 한계가 반복적으로 보고돼 왔습니다.

- **Core Contribution**: 이 논문은 합성 재난 이미지 탐지의 일반화 격차를 정량화하기 위해 Forged Calamity(30,000장)를 제안합니다. REAL 6,000장은 Incident-1M에서 4개 재난 범주로 선별했고, 나머지 24,000장은 Stable Diffusion 1.5/2.0/SDXL, PixArt의 네 생성기로 생성해 교차 생성기·교차 도메인 평가가 가능하게 했습니다. fine-tuned와 zero-shot 두 설정을 함께 실험해 “보는 순간은 잘 맞지만, 확장하면 무너지는” 취약점을 체계적으로 드러냅니다.

- **Technical Challenges**: 핵심 기술적 난제는 학습 데이터가 본 생성기/본 재난 유형에 최적화된 “시각 지문”을 학습해버리는 overfitting을 막고, unseen 생성기·unseen semantic shift에서도 안정적으로 판별 규칙을 유지하는 것입니다. 논문은 학습을 특정 재난 1개 범주와 특정 생성기(SD 1.5 등)에 고정한 뒤, (1) 다른 재난 범주로 옮기는 semantic OOD와 (2) 다른 diffusion 생성기로 옮기는 generative OOD를 분리 평가해 이 문제를 구체화했습니다. 또한 SigLIPv2, DINOv2 같은 다양한 백본과 RINE/UA/ADOF/FreqNet 등 포렌식 계열 방법을 함께 검증해 “표현 강건성”이 생각보다 잘 전이되지 않음을 확인합니다.

- **Empirical Impact**: 실험 결과 대부분의 탐지기는 학습 분포(예: SD 1.5 생성물)에서는 90%대 정확도를 보이지만, unseen 생성기나 재난 유형에서는 최대 40~50%p 수준까지 정확도가 급락합니다. Transformer 계열이 CNN보다 상대적으로 덜 무너지긴 하지만, SDXL·PixArt처럼 더 최신 생성기에 대해서는 여전히 전반적인 일반화가 부족했습니다. zero-shot에서도 완전한 방어는 어려워, ADOF·SPAI 같은 일부 주파수 기반/자기지도 기반 방법이 강점을 보이지만 조건 전반을 일관되게 커버하지 못해 “도메인·모델 불가지(generic) 탐지”의 긴급한 연구 필요성을 강조합니다.



### Hierarchical Multi-Modal Retrieval for Knowledge-Grounded News Image Captioning (https://arxiv.org/abs/2606.18553)
Comments:
          SOICT 2025

- **Prior Approaches**: 기존 image captioning은 보이는 객체·장면 묘사에는 강하지만, 사진에서 직접 확인하기 어려운 인물·날짜·장소·사건의 의미 같은 맥락 정보 생성에는 한계가 있습니다. Retrieval-augmented 접근은 RAG나 kNN 메모리, 외부 캡션 datastore 등으로 지식 주입을 시도했지만, 뉴스 기사를 ‘덩어리 텍스트’로 취급하거나 시각-담화 정렬을 충분히 반영하지 못한다는 문제가 남았습니다.

- **Core Contribution**: 이 논문은 뉴스 기사를 구조화해(제목/리드/본문/캡션) 컴포넌트별 가중치를 달리하는 hierarchical multi-modal retrieval-augmented captioning 프레임워크를 제안합니다. 또한 최종 생성 전 VLM이 먼저 이미지의 구조적 요약(시각 분석)을 만들고, 그 요약을 기준으로 기사에서 관련 문장을 추출·재구성한 뒤 LLM이 증거를 바탕으로 ‘뉴스형’ 캡션을 합성합니다.

- **Technical Challenges**: 핵심 난제는 (1) 기사 내 텍스트와 함께 기사에 포함된 이미지 배치·담화 위치까지 고려해 ‘사건 중심성’에 맞는 증거를 안정적으로 찾는 것, (2) 검색된 지식을 LLM이 시각 증거를 덮어버리지 않게 잘 통제해 쓰는 것입니다. 이를 위해 CLIP-ViT-B/32 기반의 content-visual alignment, visual-visual coherence, discourse positioning을 복합 점수로 결합하고, 시간/인용 네트워크로 검색 결과를 refinement한 뒤, Stage1의 시각 분석을 앵커로 Stage2 추출 증거(top-3 문장+인접 문맥)와 Stage3 합성을 수행합니다.

- **Empirical Impact**: EVENTA 2025 Challenge의 OpenEvent-V1(공개 5위 과제)에서 private test 기준 overall score 0.2824로 5위를 기록했습니다. 정량적으로는 retrieval 지표(mAP/R@1)가 전체 성능을 견인했고, CIDEr는 낮지만 CLIP Score가 높아 ‘문장 어휘는 다르더라도 의미 정렬이 잘 된다’는 패턴이 관찰되었습니다. 정성 결과에서도 단순 객체 나열을 넘어 인물·장소·사건 맥락을 갖춘 캡션을 생성해, 뉴스 영역 knowledge grounding의 실효성을 보여줍니다.



### A Prototypical Signature Approach for Writer-Independent Offline Signature Verification (https://arxiv.org/abs/2606.18528)
Comments:
          Accepted for oral presentation at the International Conference on Pattern Recognition (ICPR) 2026

- **Prior Approaches**: 오프라인 필기 서명 검증은 정적 이미지로 진짜 서명과 위조 서명을 구분하는 문제다. 하지만 실제 위조 데이터가 부족해, 보통 다른 사용자 진짜 서명에서 랜덤으로 negative sample(부정 예)을 뽑아 학습 데이터를 구성한다. 이 랜덤 선택은 다양성이 부족하고 중복이 늘며 학습 비용을 키워 비효율을 초래한다.

- **Core Contribution**: 논문은 prototypical signatures(원형 서명)라는 데이터 기반 요약을 이용해, 더 다양하고 정보량이 큰 negative sample을 생성하는 전략을 제안한다. 원형 서명은 진짜 서명 특징을 compact하면서도 비식별(non-identifiable) 형태로 요약해, 스킬드 위조 탐지에 유리한 부정 예를 만든다. 또한 제안 방식은 특정 네트워크에 의존하지 않는 backbone-agnostic 특성을 보이며, primal-form linear SVM과 결합할 때 RBF 기반 모델의 대안이 된다.

- **Technical Challenges**: 핵심 난제는 실제 위조를 직접 확보하지 못한 상황에서, 랜덤 부정 예의 중복과 낮은 다양성을 줄이면서도 학습에 필요한 “정보가 있는” negative sample을 안정적으로 만들어내는 것이다. 논문은 진짜 서명 특징을 원형 서명으로 응축한 뒤 negative sample 생성에 활용하는 데이터 중심 절차를 설계해 이 문제를 해결한다. 더 나아가 모델 학습을 SVM의 primal-form과 결합해 계산 복잡도를 낮추는 방향으로 확장성을 확보했다.

- **Empirical Impact**: 실험 결과, prototypical signatures는 랜덤 부정 예보다 더 informative negative samples를 제공해 특히 숙련된(skillful) 위조에 대한 탐지 성능을 개선했다. 또한 동일 전략이 다양한 아키텍처에 걸쳐 성능을 유지해 백본 독립성을 실증했다. 마지막으로 RBF 기반 모델 대비 확장성과 계산 효율에서 큰 개선을 보이면서도, primal-form linear SVM과의 조합이 실용적인 대안임을 보여줬다.



### Architectural Bias in Face Presentation Attack Detection: A Comparative Study of Vision Transformers and Convolutional Neural Networks (https://arxiv.org/abs/2606.18510)
Comments:
          8 Pages, 4 Figures, 5 Tables

- **Prior Approaches**: 기존 face PAD는 주로 Local Binary Pattern(LBP) 같은 지역 질감 단서나 CNN이 학습하는 국소 텍스처에 의존해 성능을 끌어왔습니다. 그런데 이러한 방식은 피부 톤 반사 차이와 데이터 불균형의 영향으로 아프리카 등 어두운 피부 톤에서 오류가 더 커지는 구조적 인종(인구집단) 편향을 남깁니다. 최근 FairSWAP류의 데이터 증강이나 LBP 기반 공정성 보정은 격차를 줄일 수 있지만, 아키텍처 관점의 편향 완화 가능성은 불명확했습니다.

- **Core Contribution**: 이 논문은 CASIA-SURF CeFA 데이터셋에서 ViT 계열이 CNN 대비 인종 간 편향을 줄이는지에 대해 비교 실증합니다. 특히 ImageNet 사전학습된 DeiT-S를, ResNet18 CNN 및 (사전학습 없는) ViT-Tiny와 함께 평가해 “아키텍처가 공정성에 영향을 줄 수 있는가”를 정면으로 검증합니다. 결과적으로 DeiT-S는 전체 정확도와 함께 인종 간 ACER 격차를 크게 낮추며, zero-shot 집단에서도 더 균형 잡힌 일반화를 보였습니다.

- **Technical Challenges**: ViT는 국소 텍스처 중심의 귀납 편향이 CNN보다 약해, 작은 PAD 데이터에서 scratch 학습 시 학습 안정성이 흔들리고 인종별 성능 격차가 커질 수 있습니다. 저자들은 이를 통제하기 위해 동일한 ethnicity-aware 전처리와 재현 가능한 평가(그룹별 지표 분해, McNemar 유의성 검정, 부트스트랩)를 적용해 공정성 차이를 통계적으로 검증합니다. 또한 zero-shot 중앙아시아(CA) 평가를 통해, 단순 평균 성능이 아니라 데이터 분포 외 집단에서의 일반화 실패를 함께 드러내도록 설계를 했습니다.

- **Empirical Impact**: DeiT-S는 전체 정확도 97.27%, EER 0.86%로 ResNet18(정확도 90.15%)을 능가하면서, 아프리카-동아시아 간 inter-ethnic ACER 격차를 0.75%에서 0.13%로 약 83% 줄였습니다. 더 나아가 ResNet18은 zero-shot CA에서 BPCER 10.44%로 높은 오거부를 보인 반면, DeiT-S는 2.89%로 약 3.6배 더 낮은 일반화 이점을 보였습니다. 이는 cross-demographic fairness가 모델 정확도만으로 자동 보장되지 않으며, ViT의 구조(전역 self-attention)와 사전학습 품질이 함께 공정성 및 보안 신뢰성을 좌우할 수 있음을 시사합니다.



### Neural Phase Correlation (https://arxiv.org/abs/2606.18496)
- **Prior Approaches**: 기존 dense correspondence/의료 등록은 SIFT류 디스크립터나 VoxelMorph 계열처럼 “두 이미지를 각각 인코딩”한 뒤 유사도나 회귀로 대응을 암묵적으로 찾는 방식이 주류였다. 이런 접근은 구조적 관계를 first-class로 모델링하지 않아, 실패해도 왜 실패했는지(어떤 가정이 깨졌는지) 진단이 어렵다는 한계가 있었다. Phase correlation은 예외적으로 두 관측 간 변환을 직접 다루지만, 고정된 Fourier basis가 전역 translation으로만 잘 작동한다는 제약이 있다.

- **Core Contribution**: 이 논문은 phase correlation의 핵심 원리(관계/변환을 직접 다루되, 실패를 구조적으로 설명 가능하게)를 “학습된 basis”로 일반화한다. 고정 Fourier basis 대신 두 개의 학습 필터 뱅크(Ψ, Φ)가 지역 변환이 분해되는 2차원 부분공간을 만들고, 그 위에서 변환이 planar rotation 형태로 작동한다고 가정한다. 아울러 같은 대수적 primitive를 비강체(non-rigid) 변형과 unitary dynamics까지 확장해, 관측 쌍만으로 대응(또는 스펙트럼)을 복원하는 프레임워크를 제시한다.

- **Technical Challenges**: 가장 큰 기술 과제는 “지역적으로 불변(invariant)인 부분공간을 실제로 잘 찾는가”와 “가정이 깨질 때 모델이 스스로를 망치지 않는가”였다. 논문은 이 문제를 closed-form 잔차 residual rk2(수식 기반)로 해결해, 각 위치/각 서브공간에서 회전 구조가 성립하는지 여부를 마스킹한다. 또한 training에서는 differentiable top-K 선택으로 잔차가 작은 필터 쌍만 남겨 specialist가 되도록 유도하고, ODE 반복 해석에서는 매 스텝 재평가로 progressive하게 부분공간 선택을 갱신해 큰 변형도 보정한다.

- **Empirical Impact**: ACDC cardiac-MRI 등록에서 제안 프레임워크는 Dice 지표가 이전 발표 베이스라인과 동등하거나 더 높고, 특히 ED→ES의 어려운 방향에서 강세를 보였다. CAMUS에서는 보조 scoring이나 adaptive-smoothness 같은 장치를 쓰지 않고도 state-of-the-art 성능에 도달했으며, residual 진단이 신뢰도 게이팅 역할을 내부적으로 제공함을 확인했다. 더 나아가 1-D quantum harmonic oscillator의 관측 쌍(시간 진화 파트너)만으로 Hermite eigenstate와 양자화된 에너지 레벨을 회복해, 영상 정합을 넘어 스펙트럼 복원의 범용성까지 실증했다.



### Vines-DB: An RGB image dataset for multi-species ornamental vine segmentation (https://arxiv.org/abs/2606.18484)
Comments:
          7 pages, 1 figure. Source data repository: OSF (DOI: https://doi.org/10.17605/OSF.IO/YJHCK)

- **Prior Approaches**: 기존 정밀 원예·도시 생태 연구에서는 잡초/작물 중심의 단일 클래스 분류나, 제한된 조명·배경 조건에서의 segmentation이 많아 현장 일반화가 어려웠다. 또한 다종(여러 종) 인스턴스 분할에 필요한 데이터는 규모와 촬영 조건의 현실성이 부족한 경우가 흔했다. 시간에 따른 캐노피 변화(월별 변동)를 반영한 벤치마크도 상대적으로 제한적이었다.

- **Core Contribution**: 이 논문은 현장 조건에서 촬영한 다종 인스턴스 segmentation을 위한 Vines-DB 데이터셋을 제안한다. 7종 포도넝쿨(잡초가 아닌 관상용 덩굴)과 배경을 포함해 polygon 기반 마스크를 제공하며, 정밀 원예에서의 종 판별·캐노피 면적 추정·스케일러블 생육 관측을 목표로 한다.

- **Technical Challenges**: 다종 덩굴은 얽힘·가림과 배경 잡음이 커서, 현장 촬영에서도 신뢰할 수 있는 인스턴스 마스크 생성이 핵심 난제다. 논문은 Roboflow로 숙련 라벨러가 polygon instance segmentation 마스크를 수작업 생성하고, 촬영 배경을 black/white 스티로폼으로 통제해 대비를 높였다. 이후 전처리와 data augmentation을 통해 최종 2,307장으로 확장하고, stratified sampling으로 학습/검증/테스트를 균형 있게 구성해 모델 평가의 공정성을 확보했다.

- **Empirical Impact**: Vines-DB는 2023~2024년 생장기(7~10월) 동안 월별 반복 촬영으로 시간 변동을 포함해, 실제 현장에서의 segmentation 성능을 더 현실적으로 벤치마킹할 수 있게 한다. 따라서 multi-class instance segmentation 연구와 정밀 원예·도시 생태 응용(자동 캐노피 커버 추정, 종 식별, 대규모 현장 생육 측정)에서 데이터 단의 격차를 줄이는 기반이 될 전망이다.



### Data-Forcing Distillation: Restoring Diversity and Fidelity in Few-Step Video Generation (https://arxiv.org/abs/2606.18478)
- **Prior Approaches**: 멀티스텝 비디오 diffusion 모델을 few-step 학생으로 압축하는 대표 흐름으로 trajectory-based distillation과 distribution-based distillation이 있다. DMD와 DMD2는 reverse KL을 통해 분포를 맞추며 빠른 수렴과 좋은 품질을 보이지만, reverse KL의 mode-seeking 성질 때문에 diversity가 크게 떨어지고 영상이 과포화(over-saturation)처럼 보이는 실패가 반복된다.

- **Core Contribution**: 본 논문은 DMD2 위에 붙이는 단순 post-training 프레임워크 Data-Forcing Distillation(DFD)을 제안한다. 핵심은 teacher score discrepancy를 이용해 학생이 생성한 샘플이 아니라 real video 샘플 기준으로 점수 차이를 학습하도록 만들어, 누락된 모드(mode)를 되찾고 실제 데이터에 없는 “문제 모드”로부터 이탈하게 한다.

- **Technical Challenges**: DFD의 어려움은 실데이터 정규화가 평균적으로는 0이더라도 분산이 커지면 학습을 흔들 수 있다는 점이다. 이를 위해 저자들은 score 불일치 항의 분산이 real과 generated 간 간격에 좌우됨을 이론적으로 분석하고, DFD를 DMD2가 이미 real에 가깝게 만든 뒤(짧은 finetuning) 적용하는 조건을 통해 안정적으로 작동하게 한다.

- **Empirical Impact**: DFD는 text-to-video( Wan2.1-1.3B → 4-step )와 image-to-video( Cosmos-Predict2.5-2B )는 물론 autoregressive video generation에서도 100~300 steps 수준의 finetuning만으로 diversity와 fidelity를 회복한다. 특히 과포화 아티팩트를 유의미하게 줄이면서 비디오 역학/외관을 개선하고, 일부 설정에서는 teacher 모델보다도 좋은 결과를 보였다고 보고한다.



### Domain Generalizable Adaptation of 3D Vision-Language Models via Regularized Fine-Tuning (https://arxiv.org/abs/2606.18472)
Comments:
          Accepted at Transactions on Machine Learning Research (TMLR)

- **Prior Approaches**: 기존 3D 비전-언어(foundation) 모델들은 point cloud를 visual·text 임베딩과 정렬해 강한 zero-shot 성능을 보이지만, downstream 도메인으로 옮길 때는 데이터가 적으면 과적합과 catastrophic forgetting이 쉽게 발생합니다. 또한 parameter-efficient fine-tuning(PEFT)이 대안으로 제시됐지만, 3D의 unordered·sparse 특성을 충분히 반영하지 못하거나 prompt tuning을 그대로 가져오며 학습 효율과 일반화가 흔들립니다. 더 나아가 PointPRC 같은 접근은 다운스트림 적응에서 image encoder를 버리는 경우가 있어 pre-training의 시각적 지식을 충분히 활용하지 못합니다.

- **Core Contribution**: ReFine3D는 3D large multimodal model(LMM)을 제한 데이터로도 도메인 일반화 가능하게 튜닝하는 regularized fine-tuning 프레임워크를 제안합니다. 핵심은 point cloud encoder에서 레이어를 선택적으로 fine-tuning하면서, 과적합을 줄이기 위한 (1) multi-view(증강) consistency 정규화와 (2) WordNet 기반 synonymization+LLM 프롬프트로 text diversity를 주는 정규화를 함께 적용하는 것입니다. 여기에 point-rendered vision supervision과 test-time augmentation 기반 confidence aggregation까지 더해 강건성을 끌어올립니다.

- **Technical Challenges**: 문제는 3D LMM의 튜닝에서 “적응은 하되 pre-training 지식은 보존”하는 균형이 매우 어렵다는 점입니다. 이를 위해 ReFine3D는 transformer 기반 point cloud encoder에서 초기 레이어는 동결하고 후반 레이어만 업데이트해 일반 기하 표현과 고수준 적응을 분리했으며, 증강된 point cloud들 간 임베딩 일관성과 다양한 텍스트 설명에 대한 정렬을 동시에 강제해 특정 입력·특정 클래스 문구에 대한 암기를 억제합니다. 또한 image/text 인코더는 frozen으로 두고, point-rendered vision supervision으로 시각적 priors를 학습 신호에 다시 연결해 멀티모달 정렬이 깨지는 현상을 완화합니다.

- **Empirical Impact**: 실험은 3D 도메인 일반화(3D-DG) 벤치마크 프로토콜을 따라 base-to-new, cross-dataset transfer, corruption 견고성, few-shot(1~16-shot)까지 폭넓게 수행했으며 일관된 성능 향상을 보였습니다. 수치로는 base-to-novel 일반화 1.36%, 크로스 데이터셋 전이 2.43%, corruption 강건성 1.80%, few-shot 정확도는 최대 3.11% 향상을 보고해 적응 성능과 일반화가 동시에 개선됨을 입증했습니다. 또한 각 구성요소의 기여를 ablation으로 확인하고 계산 오버헤드가 최소임을 강조해, 실제 적용 가능성도 함께 부각됩니다.



### Reasoning as Intersection: Consensus-Frame Alignment for Visual Focus in Video-MLLMs (https://arxiv.org/abs/2606.18441)
- **Prior Approaches**: 기존 Video-MLLM RL은 정답 정확도 같은 결과 보상과 함께, 시간적 일관성·로컬라이제이션·툴 사용 등 작업에 맞춘 검증 가능 보상을 설계해 성능을 끌어올려 왔다. 하지만 보상이 최종 답의 맞음에 치우치면, 어떤 영상 프레임이 답을 뒷받침했어야 하는지(증거 프레임)에는 학습 신호가 빈약해진다. 또한 기존 프로세스 보상은 텍스트 추론 단계나 간접적 trace 평가에 머무르는 경우가 많아, 영상의 시간적으로 분산된 증거 정렬을 직접 감독하기 어렵다.

- **Core Contribution**: 이 논문은 Consensus Frame GRPO (CF-GRPO)로, 인간이 시간 주석을 달지 않아도 증거 프레임 정렬을 학습하도록 만드는 temporal-annotation-free process reward를 제안한다. 핵심은 Consensus Frame Reward (CFR)로, 질문에 필요한 시각 증거일 가능성이 높은 프레임 집합(합의 프라이어)과 모델 내부에서 실제 사용된 프레임 분포가 일치하도록 최적화하는 것이다. 이를 통해 결과(outcome) 중심 학습의 보상 ‘미정의’를 프레임 수준 신호로 보완한다.

- **Technical Challenges**: 가장 큰 문제는 “어떤 프레임이 답의 근거였는지”를 시간 라벨 없이 어떻게 정의하느냐이다. 저자들은 uniform temporal coverage, scene-transition 단서, query-conditioned visual relevance를 결합해 합의 프레임 prior를 만들고, 이를 모델의 frame-use score(시각 표현-응답 hidden state 유사도)와 agreement시킨다. 보상 신호의 대비를 높이기 위해 salience-aware sparse aggregation로 고반응 영역만 보존하고, temperature sharpening으로 프레임 점수 분포를 뾰족하게 만들어 구분력을 확보한다.

- **Empirical Impact**: 실험에서 VideoCFR은 Video reasoning 벤치마크(VSI-Bench, VideoMMMU, MMVU(MC) 등)와 general video 이해 벤치마크(MVBench 등)에서 기존 Video-MLLM 및 RL 기반 기준선 대비 경쟁력 있는 성능을 보이며 일부 지표를 개선했다. 또한 합의 프라이어, sparse aggregation, sharpening 같은 CFR 구성요소를 제거한 ablation에서 성능 하락이 나타나 신호의 실질적 기여를 확인했다. 해석 가능성 측면에서도 학습 중 강조된 evidence frame에 대한 관찰이 가능해, 단순 정확도 향상 이상의 증거 정렬 효과를 보여준다.



### RegimeVGGT: Layer-Wise Spatially Preserving Redundancy Removal for Visual Geometry Grounded Transformer (https://arxiv.org/abs/2606.18439)
Comments:
          9 pages, 3 figures, 7 tables. Jinhao You, Shuo Lyu, Zhuohang Lyu, Tanxuan Li, and Zibo Zhao contributed equally. Shuo Lyu is the corresponding author

- **Prior Approaches**: VGGT는 다중 뷰 이미지를 단일 forward pass로 3D와 카메라 포즈를 복원하지만, 프레임 간 global cross-frame attention이 O(S^2P^2)로 커져 장기 시퀀스에서 메모리와 속도 병목이 발생한다. FastVGGT, S-VGGT 같은 학습 없이 가속 기법은 한 축(토큰 수/프레임 분할 등)을 균일하게 줄이거나 단순 격자로 K/V를 다운샘플해, VGGT 내부 레이어의 이질성과 ‘포즈에 필요한 경로’를 놓친 한계가 있다. AVGGT는 레이어별 중복성이 있음을 보였지만, 두 축 압축 설계를 함께 최적화하는 질문은 남아 있었다.

- **Core Contribution**: RegimeVGGT는 VGGT의 24개 aggregator 레이어를 스펙트럴·프로빙·인과 분석으로 얕은/중간/깊은 3개 ‘레짐(regime)’으로 나누고, 각 레짐의 기능이 서로 다르다는 점을 압축 설계로 연결한다. 특히 포즈는 camera/register 토큰과 주변 패치 사이 cross-frame attention 경로에 걸려 있으므로, 기하 복원에 덜 필요해 보이는 패치라 해도 이 경로만큼은 유지해야 한다는 전제를 도입한다. 이 구조를 반영해 Saliency-Guided Banded Merging(토큰 축)과 Selectively Protected K/V Downsampling(K/V 축)을 레이어별 U자형 압축으로 결합한다.

- **Technical Challenges**: 핵심 난제는 ‘기하(3D dense geometry)’와 ‘포즈(camera pose)’가 같은 attention 요소를 공유하지 않을 수 있는데, 학습 없이 이를 동시에 보존해야 한다는 점이다. 논문은 먼저 레이어 레짐마다 cross-view 구조의 존재 여부와 정보 활용이 달라짐을 보이고, 중간 레짐(L11–L18)이 대부분의 포인트클라우드 복원을 담당하되 얕은/깊은 레짐은 깊은 기하 신호가 중복되거나 작동하지 않는다고 정리한다. 이후 토큰 축에서는 DINOv2 CLS saliency로 geometry·edge-salient 토큰을 보호하면서 ToMe 계열 merge-unmerge를 레짐별 비율로 수행하고, K/V 축에서는 phase-shifted spatial sub-grid, frame-0 reference anchor, uncompressed camera/register 토큰을 통해 포즈-critical path를 유지하는 방식으로 해결한다.

- **Empirical Impact**: ScanNet-1000 기준으로 RegimeVGGT는 VGGT* 대비 6.7배 속도를 달성하면서 재구성 품질을 거의 유지한다. 장기 시퀀스(예: 1000 frames)에서 VGGT는 OOM이 발생하고 S-VGGT도 실패하는 반면, RegimeVGGT는 실행 시간을 크게 줄이며 Chamfer Distance가 크게 악화되지 않았다. 포즈 추정에서도 Tanks & Temples, DTU, ScanNet-50 장기 설정에서 AUC/ATE/ARE/RPE 지표가 competitive하거나 더 좋은 결과를 보여, cross-frame correspondence 보존이 실제로 확인됐다는 점에서 의미가 크다.



### CAOA -- Completion-Assisted Object-CAD Alignmen (https://arxiv.org/abs/2606.18429)
Comments:
          GitHub: this https URL

- **Prior Approaches**: 실내 RGB-D 기반 3D 시맨틱 재구성은 CAD 검색·object-CAD 정렬·레이아웃 추정으로 이어지며, 기존 object-CAD 정렬은 에너지 최적화나 end-to-end 파이프라인으로 대응해 왔다. 다만 잡음·불완전성·세그멘테이션 오류가 만들어내는 기하 왜곡 때문에 정확한 9-DoF(이동·회전·스케일) 추정이 흔들린다. 또한 점군 completion 모델들은 주로 합성 데이터에 학습·평가되어 real-world 일반화가 제한적이라는 문제가 누적되어 왔다.

- **Core Contribution**: 이 논문은 Completion-Assisted Object-CAD Alignment(CAOA)로, CAD를 정렬하기 전에 점군 completion으로 스캔의 불완전성과 잡음을 완화한 뒤 정렬을 수행하는 구조를 제안한다. completion 학습을 위해 실내 단일 객체용 real-world 벤치마크 S2C-Completion(8,500+ object-CAD pair)과 실내 특화 합성 데이터 SN-Indoor를 새로 설계했다. 더불어 대칭 모호성에 강하도록 symmetry encoder(SEM)와 symmetry-aware loss를 결합해 회전·스케일 추정의 견고성을 높였다.

- **Technical Challenges**: 핵심 난제는 (1) 실세계 점군 completion에 맞는 데이터 부재로 인한 합성-실세계 도메인 갭, (2) completion 결과가 장면 맥락을 무시할 때 크기·형상이 일그러지는 문제, (3) 객체 대칭 때문에 학습된 pose 특징이 부정확해질 수 있는 문제다. CAOA는 CAPCM에 context-aware completion(주변 컨텍스트 포인트를 함께 사용)을 적용하고, S2C-Completion과 SN-Indoor로 학습해 일반화를 개선했으며, SEM이 대칭 정보를 별도 임베딩으로 제공하고 Chamfer loss와 가중 L1로 symmetry-aware 학습을 수행한다.

- **Empirical Impact**: Scan2CAD 벤치마크에서 CAOA는 기존 SOTA 대비 class average 정확도를 약 17% 향상(전체 정확도도 약 16%)시키며, 대칭 처리와 completion 보조가 정렬 품질을 실질적으로 끌어올렸음을 입증한다. 또한 Ground Truth ScanNetv2를 사용하면 세그멘테이션 오류로 인한 정렬 성능 저하가 약 10% 수준임을 보여, 파이프라인 모듈화의 의미(세그 품질 개선 여지)도 강조한다. 데이터와 방법론(S2C-Completion 공개) 측면에서 실내 single-object completion/정렬 분야의 새로운 표준 벤치마크를 제공한 점이 파급력이 크다.



### Budget-Aware Adaptive Adversarial Patches for Black-Box Object Detection (https://arxiv.org/abs/2606.18318)
Comments:
          Accepted to the 2026 IEEE International Conference on Image Processing (ICIP 2026)

- **Prior Approaches**: 기존 연구는 보편형(white-box) 패치나 물리 데모를 제시했지만, 실제 배치·크기 결정을 포함한 “score-based black-box”에서는 쿼리 예산이 빠듯한 상황에서 최적화를 함께 수행하기 어렵다는 한계가 컸습니다. 또한 EOT(Expectation Over Transformation)로 평균 성능을 맞춘 뒤 이를 성공으로 간주하는 평가가 많아, 배치된 카메라의 “plain view” 억제와의 차이가 흐려졌습니다.

- **Core Contribution**: PatchBandit은 score-only black-box 조건에서 패치의 위치·텍스처·크기를 함께 최적화하되, Contextual Thompson-Sampling으로 배치 후보를 빠르게 고르고 NES-style zeroth-order 갱신으로 텍스처를 다듬는 공격 프레임워크를 제안합니다. 진행이 막히면에만 패치 크기를 늘리는 budget-adaptive growth 정책으로, 적은 쿼리로도 시각적 footprint(면적) 대비 억제 효과를 드러내도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 그라디언트 없이 점수만으로, “어디에 놓는 게 이득인지”까지 포함해 탐색해야 한다는 점입니다. 논문은 배치 그리드를 discretize하고 위치별 맥락 특징으로 bandit 보상을 모델링해 Thompson sampling으로 탐색을 줄였고, 패치 업데이트는 NES로 수행하되 size ladder를 통해 실패 시에만 면적을 확대해 쿼리-면적 균형을 맞췄습니다. 성공 판정은 EOT 평균이 아니라 strict plain-image suppression으로 고정해, 현실 배치 조건에서의 억제 여부를 분리해 검증합니다.

- **Empirical Impact**: YOLOv5와 Faster R-CNN에서는 높은 strict suppression(각각 77.5%, 89.7%)을 보였고, YOLOS(Transformer detector)에서도 59.1%로 “상당한” 억제 효과가 유지되었지만 CNN 대비 약해지는 양상이 나타났습니다. 더 나아가 print–capture 파일럿에서 시뮬레이션으로 만든 패치가 미지의 물체와 뷰포인트에서도 전이 가능함을 보여, 물리 위협으로서의 현실성을 강화했습니다. 또한 fixed-size 대비 adaptive growth가 쿼리만이 아니라 footprint까지 고려한 효율 지표에서 유리함을 명확히 드러내 “강함 vs 눈에 띔” 트레이드오프를 정량화하는 데 기여합니다.



### Do as I Do: Dexterous Manipulation Data from Everyday Human Videos (https://arxiv.org/abs/2606.19333)
Comments:
          Project website: this https URL

- **Prior Approaches**: 로보틱스의 숙련 조작은 전통적으로 텔레오퍼레이션이나 시뮬레이션 탐색으로 “직접 해보는” 경험 데이터를 만들었지만, 비용·전문성·환경/보상 설계가 병목이었다. 최근에는 human videos에서 동작을 추출하는 Do as I do 계열이 발전했으나, 대부분은 (1) 잡음이 큰 monocular RGB에서 hand-object 상호작용을 신뢰성 있게 복원하거나 (2) 손-물체 접촉과 물리적 타당성을 고려해 로봇으로 retargeting 하는 데 한계가 있었다. 특히 기존 human-to-robot retargeting은 깨끗한 기준선(예: MoCap 기반 ground-truth poses)을 가정하는 경우가 많아, 실제로는 “복원된 참조가 노이즈인” 인터넷 영상 적용이 어려웠다.

- **Core Contribution**: DO AS I DO는 monocular RGB 인간 영상을 dexterous multi-fingered 로봇 손에서 재현 가능한 조작 데이터로 바꾸는 end-to-end 파이프라인을 제안한다. 핵심은 ① hand-object 상호작용을 3D로 복원·추적해 “로봇이 따라야 할 reference”를 만들고 ② 그 reference를 물리 시뮬레이션 기반 dynamics-aware retargeting으로 로봇 실행 궤적(robot-complete)로 변환한다는 두 단계에 있다. 또한 grasping priors나 특정 물체 범주 같은 제한 가정을 줄여, 다양한 in-the-wild 소스(ego/exo, 심지어 생성 비디오)까지 확장하는 점이 기여로 요약된다.

- **Technical Challenges**: 가장 큰 기술 난제는 monocular RGB에서 물체 pose가 occlusion·저해상도·비디오 품질 저하에 무너지는 문제와, 그로 인해 노이즈/불연속 reference가 생성될 때 retargeting이 붕괴하는 문제다. 저자는 SAM 3D 기반의 guided diffusion 추적을 활용해 물체 shape는 고정(anchor)하고 pose만 프레임별로 갱신함으로써 temporally consistent한 pose evolution을 노렸고, hand-물체 스케일·중력 정렬까지 맞춰 near metric-space 기준을 구성한다. retargeting 단계에서는 noisy reference에서 초기 상태 불가능성을 줄이기 위한 warmup steps, 국소 최적에 갇힘을 완화하는 random force perturbation, rest/in-hand 전이의 실패를 직접 페널티로 다루는 transition reward를 더해 실험적으로 안정성과 성공률을 끌어올렸다.

- **Empirical Impact**: 실험에서는 DexYCB와 HOI4D의 ground-truth 기반 검증에서 hand-object 상호작용 추정 및 trajectory 추출 성능이 기존 SOTA를 능가했으며, 150개 in-the-wild/생성 비디오에서도 사람 평가 선호가 더 높게 나타났다(객체 pose 추적에서 67% 선호). retargeting은 자체 reconstructed in-the-wild 기준으로 success rate 25%에서 71%로 크게 개선했고, OakInk2에서도 구성요소를 추가할수록 72%→81%로 상승해 clean MoCap에서도 일반화 이득이 있음을 보였다. 더 나아가 500개 규모의 사람 검증 기반 dexterous 조작 궤적을 생성해 실제 로봇(Sharpa Wave 손, UR3e 팔)에서 다양한 grasp 유형으로 재현함으로써 “인터넷 영상→현실 dexterous rollout”에 가까운 데이터 스케일링 경로를 실증했다.



### Reference-Driven Multi-Speaker Audio Scene Generation from In-the-Wild Priors (https://arxiv.org/abs/2606.19325)
Comments:
          Project page at this https URL

- **Prior Approaches**: 기존 multi-speaker TTS/대화 TTS는 per-turn speaker tag, multi-stream transcription, speaker-turn embedding 같은 구조화된 감독으로 화자-발화를 결합(speaker-binding)하곤 했습니다. 또한 대화 생성은 보통 speech-only 파이프라인에서 정제된 발화만 산출해, 실제 대화의 잡음·실내 반향·겹치는 발화·비언어적 소리(웃음/숨소리)를 충분히 담기 어렵다는 한계가 있었습니다. 더 나아가 zero-shot voice cloning은 주로 단일 화자에 집중해 멀티 화자를 모델 밖에서 조립하는 방식에 머물렀습니다.

- **Core Contribution**: ScenA는 flow matching 기반 text-to-audio foundation model에 멀티 레퍼런스(여러 화자 음성)를 텍스트 프롬프트와 함께 조건화해, 한 장면(전체 대화+주변 소리)을 end-to-end로 생성하는 접근을 제안합니다. 핵심은 레퍼런스 latent를 입력 토큰에 결합하고 identity-aware positional encoding으로 역할을 구분하되, per-turn 구조 없이 자유형 자연어 프롬프트가 “어떤 레퍼런스 화자가 어디서 발화하는지”를 지정한다는 점입니다. 이 설계는 스튜디오 수준의 clean vocal만이 아니라 in-the-wild 오디오 질감까지 함께 생성할 수 있도록 목표를 바꿉니다.

- **Technical Challenges**: 가장 큰 난관은 학습 중 “Reference Shortcut”이 발생한다는 점입니다. 표준 노이즈 스케줄에서는 noised target이 여전히 음향적으로 레퍼런스와의 유사성을 보존해, 모델이 텍스트 경로를 무시하고 self-attention만으로 레퍼런스를 매칭해버릴 수 있으며, 학습 손실은 낮아져도 잡음에서 시작하는 추론 시 화자 결합이 붕괴됩니다. ScenA는 timestep 분포를 high-noise에 편향한 Beta+Uniform 혼합으로 바꿔 텍스트가 유일한 결합 신호가 되는 구간에 학습을 더 실어 이 지름길을 차단합니다. 추가로 adversarial reference injection과 slot-shuffle augmentation으로, 레퍼런스 순서 편향이나 프롬프트 없이도 맞출 수 있는 우회 경로를 더 줄였습니다.

- **Empirical Impact**: CoVoMix2-Dialogue 벤치마크에서 ScenA는 speaker-binding 중심 지표(cpWER, cpSIM, ACC)에서 기존 multi-speaker 대화 TTS를 능가하거나 최상 성능을 보였습니다. 특히 스튜디오 clean 레퍼런스에서 in-the-wild noisy 레퍼런스로 난이도가 올라가도 cpSIM이 더 견고하게 유지되어, 실제 사용 조건에서의 의미가 커졌습니다. 또한 A/B preference 테스트에서 모든 비교 대상 대비 선호도가 유의미하게 높았고, 생성물은 겹치는 대화, 웃음·한숨·숨소리 같은 paralinguistic event, 방/환경 잡음까지 장면 단위로 함께 구현되는 것으로 보고됐습니다.



### Seeing Through Occlusion: Deterministic Arm Kinematic Correction for Robot Teleoperation (https://arxiv.org/abs/2606.19240)
- **Prior Approaches**: 단일 RGB-D 카메라 markerless 모션캡처는 설치가 쉽지만, self-occlusion 상황에서 depth 추정이 흔들리며 텔레오퍼레이션 중 로봇 동작이 불안정해질 수 있다. 이를 줄이기 위해 KF/EKF 같은 필터링이나 particle filter, 최적화 기반 inverse kinematics(COIK), 학습 기반/하이브리드 기법들이 제안됐으나, 복잡한 모델 설계·파라미터 튜닝·반복 연산이 필요하거나 장시간·심한 occlusion에서 안정성이 떨어질 수 있다. 또한 기존 방법은 관절을 독립적으로 처리하는 경우가 많아, 팔 길이 같은 해부학적 제약을 명시적으로 강제하지 못해 kinematic 불일치가 누적될 여지가 있다.

- **Core Contribution**: 본 논문은 Arm Kinematic Correction(AKC)이라는 후처리 보정 모듈을 제안해, self-occlusion으로 깨진 depth와 kinematic 불일치를 동시에 완화한다. 상완·전완 길이가 일정하다는 기하 제약을 Pythagorean theorem 기반의 결정론적(deterministic) 복원으로 강제하고, 관절 깊이를 후보군으로 재구성한 뒤 KF 기준과 시간/해부학적 일관성 비용함수로 최적 후보를 선택한다. 그 결과 확률 모델이나 복잡한 최적화 없이도 해부학적으로 일관된 팔 자세를 만들어 모션-매핑 텔레오퍼레이션에 활용한다.

- **Technical Challenges**: 핵심 기술 난제는 occluded joint의 depth가 부정확해질 때 가능한 기하해가 여러 개(± 해)로 분기되고, 관측 잡음으로 인해 radicand가 음수가 되어 비현실적 해(imaginary)가 생길 수 있다는 점이다. 논문은 손목이 카메라에 가깝고 덜 가려진다는 가정을 두고 팔 길이 제약으로 팔꿈치/어깨 깊이를 후보로 복원하되, 음수 radicand 발생 시에는 최소 변화로 “feasible surface”에 투영하는 보정 장치를 둔다. 또 해가 0-depth 평면 근처에서 불안정해질 때를 대비해 탐색 시 팔 길이를 shrink factor로 줄여 후보 선택의 견고성을 높였다.

- **Empirical Impact**: Vicon(정답) 대비 Intel RealSense D435 단일 RGB-D 실험에서 AKC는 static·dynamic 모두에서 RMSE와 Pearson correlation으로 depth 품질을 개선하며, 특히 장시간 occlusion(팔꿈치/어깨)에서 KF보다 큰 폭의 오차 감소를 보였다. 또한 반복 최적화 기반 COIK와 비교해 AKC는 determinism과 지연(latency) 측면에서 유리하며, 추가 오버헤드는 약 11ms 수준으로 실시간 텔레오퍼레이션에 더 적합하다는 점을 강조한다. 최종적으로 AKC는 긴 시간·심한 occlusion에서도 해부학적 팔 길이를 일정하게 유지하고, 시간 필터 신뢰도가 낮아도 robustness를 보이며 시뮬레이션과 실제 로봇 환경에서 motion-mapping 텔레오퍼레이션을 성공적으로 시연했다.



### The Reward Was in Your Data All Along: Correcting Flow Matching with Discriminator-Guided RL (https://arxiv.org/abs/2606.19162)
Comments:
          84 pages, including appendices

- **Prior Approaches**: flow/score matching(FSM)은 시뮬레이션 없이 속도·score 장을 ℓ2 회귀로 학습해 데이터 분포를 모사하려는 접근이다. 하지만 실제로는 base 모델에 RL post-training을 추가해 시각적 사실성, 일관된 물체 구조 같은 데이터 내 속성을 “다시 복원”하는 데에도 RL이 쓰인다. 논문은 이 현상을 train-time 마진 q_t에서 측정되는 matching 손실의 구조가, 추론 시 생성 궤적 p_t가 평가하는 샘플 품질과 어긋난 구조적 불일치 때문이라고 본다.
또한 사람 선호를 통한 reward 학습은 비싸고, 데이터의 사실성(원하는 속성)과 annotator 성향(주관적 취향)을 함께 섞어 최적화 방향이 흐려질 수 있다.

- **Core Contribution**: 논문은 Discriminator-Guided RL(DRL)을 제안하며, matching의 구조적 불일치를 RL이 자체 샘플에서 reward landscape를 따라 우회하도록 만든다. 핵심은 사람이 아닌 “데이터 vs base 모델”의 밀도비를 판별기가 추정하고, 그 logit을 KL-regularized RL의 reward로 사용한다는 점이다.
더 나아가 판별기를 self-supervised learning(SSL) 표현 공간에 제한해, 의미론적으로 타당한 축에서만 보정되도록 설계한다(추론 품질과 직접 연결되는 방향만 사용).

- **Technical Challenges**: challenge는 사람 선호처럼 비싼 입력 없이, RL이 잘 맞는 reward를 얻는 것이다. 이를 위해 DRL은 “데이터와 base의 log density ratio”가 이상적인 reward가 된다는 관찰을 출발점으로 삼되, 이를 그대로 출력공간에서 추정하면 통계적으로 어렵고 의미적으로도 불안정해진다.
그래서 frozen encoder ϕ가 만든 표현 공간에서 discriminator를 학습하고, 그 logit을 reward로 써서 추정 가능성과 의미 축 정렬을 동시에 확보한다. 또한 flow 모델에 맞게 adjoint matching 기반 KL-regularized RL을 적용해 학습 안정성을 높였으며, RL 단계에 필요한 memoryless SDE용 local-linear integrator도 함께 도입한다.

- **Empirical Impact**: DRL은 SiT, JiT, REPA, RAE 등 여러 ImageNet-pretrained flow 모델에 대해 guidance-free FID 및 DINOv3 같은 의미 공간 FD를 크게 낮추며, backbones 전반에서 일관된 개선을 보인다(예: guidance-free FID 9.38→2.62 on SiT). 무엇보다 DRL은 사람 선호 reward에 학습되지 않았는데도, held-out human-preference reward를 측정하면 개선이 나타난다.
또한 이후의 preference-based post-training(PRL)을 결합했을 때 reward–distortion Pareto frontier가 더 좋아져 정렬은 강화하면서 과채도·과도한 밝기 같은 저수준 아티팩트를 줄이는 효과도 보고한다.



### The Market in the Model: Latent Diffusion as Neural Economy (https://arxiv.org/abs/2606.19151)
- **Prior Approaches**: 기존 시각에서 생성 이미지 모델은 주로 데이터셋이 결과물을 “만드는 방식”에 초점을 맞췄고, 모델 내부의 이념이 어떻게 작동하는지는 대체로 블랙박스로 남겨졌다. 저자는 Stable Diffusion으로 이어진 Latent Diffusion이 여러 구성요소(스크래퍼, 분류기, 랭커, 필터, 피드백 시스템)의 결합이라는 점에서, 데이터 비평을 대체하기보다 메커니즘 차원의 데이터 비평을 확장할 필요가 있다고 본다. 또한 저작권·상품 논쟁 같은 경제적 방어 중심 접근이 모델이 생산하는 “상품적 물신(commodity fetishism)”을 다시 강화할 수 있다고 경고한다.

- **Core Contribution**: 논문은 Latent Diffusion의 구성요소가 왜(컴퓨터 비전 엔지니어가 해결하려던 문제들을 위해) 설계·자동화됐는지를 추적해, 모델이 생성하는 이미지에 “시각 이론”이 어떻게 새겨지는지 해석한다. 특히 이 모델을 단순한 신경망이 아니라, 사회적 소통을 비교 가능 벡터로 추상화해 순환시키는 “neural economy(신경 경제)”로 규정하며, 이것이 attention economy의 논리를 사회적 커뮤니케이션으로까지 확장한다고 주장한다. 결과적으로 생성 이미지는 사용자의 자발적 창작처럼 보이지만, 실제로는 거래 가능한 가치 단위로 사회적 표현을 전환하는 파이프라인의 산물이라는 관점을 제시한다.

- **Technical Challenges**: 기여를 실현하기 위한 핵심 난제는 Latent Diffusion을 ‘가중치만의 문제’로 환원하지 않고, autoencoder·CLIP·U-Net·CFG 같은 모듈별로 서로 다른 평가 기준과 대체(누락) 효과를 분해해 읽어내는 것이다. 저자는 학습 데이터 구성과 평가·필터링(예: Common Crawl, CLIP, LAION-400M), 압축·복원(지각 손실, PatchGAN류의 판별 목표), 노이즈에서 이미지로의 복원(U-Net의 denoising 관성)이 각각 무엇을 대체하고 무엇을 고착하는지 “작동의 흔적”으로 연결한다. 그 과정에서 ‘인간의 지각’이 통계적 대리평가로 치환되고, 모델이 닫힌 상징 질서(이미지-텍스트 페어에 대한 자기충족적 질서)를 학습하며 그 밖의 소통 가능성은 자동으로 평탄화된다고 설명한다.

- **Empirical Impact**: 논문은 주로 정량 실험을 통해 모델 성능을 증명하기보다, Stable Diffusion 계열의 대표적 설계 논리를 컴포넌트 역사와 데이터 파이프라인 관점에서 재구성하는 분석 논문에 가깝다. 그럼에도 image 모델을 저작권/상품성 프레임에만 가두지 않고, ‘social exchange’의 중심으로 비평의 축을 옮기려는 제안이 커뮤니티 담론에 영향을 줄 가능성이 크다. 특히 생성 품질이 어떻게 “무엇을 그렸는지”가 아니라 “무엇처럼 보이는지”로 기준화되는지(압축된 지각 가능성의 고정)라는 문제의식은 안전·윤리·해석가능성 논의에 새로운 질문을 던진다.



### Seeing Before Reasoning: Decoupling Perception and Reasoning for Shortcut-Resilient Multimodal On-Policy Self-Distillation (https://arxiv.org/abs/2606.19120)
Comments:
          29 pages, 5 figures, 8 tables

- **Prior Approaches**: on-policy self-distillation(OPSD)은 학생이 생성한 롤아웃에 대해 frozen 복사 교사가 dense한 토큰 단위 타깃을 주는 방식으로, train-test gap을 줄이면서 RLVR보다 촘촘한 학습 신호를 제공한다. 다만 multimodal large language model(MLLM)에 그대로 옮기면, reference answer를 본 privileged 교사가 답을 기준으로 토큰을 먼저 맞춰버리는 shortcut(정답-유도 우회) 위험이 생긴다. 특히 VQA에서 language-prior·shortcut-learning 문제가 알려져 있듯, 텍스트 신호가 이미지보다 더 쉽게 따라가며 시각적 정합성(grounding)을 약화시킬 수 있다.

- **Core Contribution**: ViGOS(Visual Grounding On-Policy Self-Distillation)는 OPSD의 on-policy 장점은 유지하되, reference target이 토큰 궤적에 들어오는 시점을 분리한다. 학생이 먼저 이미지 기반 visual description을 쓰고, 그 다음 privileged reasoning teacher가 reasoning/최종답을 supervises 하도록 설계해 answer-guided reasoning은 살리고 early visual claims의 answer leakage는 줄인다. 또한 형식이 깨진 invalid 롤아웃에만 reference teacher를 제한적으로 써서 output format drift를 복구한다.

- **Technical Challenges**: 핵심 기술 과제는 “같은 student prefix”에 대해 어떤 토큰 구간에서는 이미지 근거만, 다른 구간에서는 정답 조건부 근거만 제공하도록 교사 컨텍스트를 정밀하게 마스킹하는 것이다. 이를 위해 ViGOS는 description/reasoning/answer를 delimiter로 구분하고, valid 롤아웃일 때는 image-only perception teacher로 description 토큰만 학습하며, reasoning/answer 토큰에는 privileged reasoning teacher를 적용한다. invalid일 때는 segment 마스크가 신뢰되지 않으므로 reverse KL 기반의 limited recovery(형식 복원)를 reference teacher로만 수행해 역할 혼선을 막는다.

- **Empirical Impact**: 실험에서 ViGOS는 OPSD가 주는 on-policy self-distillation의 성능 상승을 전반적으로 유지하면서(예: 8개 벤치 Pass@5 평균 개선) 이미지 이해가 필요한 과제에서 특히 이득을 보였다. ViLP(시각-언어 prior 충돌 스트레스 테스트)에서는 OPSD 대비 Score를 더 크게 올리면서 Prior 정확도는 높은 수준으로 유지해, 단순히 prior를 억제하는 방식이 아니라 이미지로부터 정합한 선택을 강화했음을 시사한다. 또한 ablation과 ViLP 학습 동역학 분석은 perception/ reasoning 분리와 reference fallback 설계가 shortcut 완화에 실제로 기여한다는 점을 뒷받침한다.



### Sensor Configuration Matters: A Systematic Evaluation of Multimodal SLAM on Quadruped Robots (https://arxiv.org/abs/2606.19067)
- **Prior Approaches**: 기존 visual SLAM/visual-inertial SLAM 벤치마크는 주로 handheld, 드론, 휠 로봇의 비교적 부드러운 움직임에 맞춰져 있어, 사족 보행이 만드는 충격·진동·급격한 회전이 성능을 어떻게 바꾸는지 정량 평가하기 어려웠다. 관련 데이터셋도 대체로 고정된 고성능 센서 구성으로 로컬라이제이션 가능성을 확인하는 데 그쳐, 카메라 모달리티·셔터·IMU 틀에 따른 취약점을 분리하기가 힘들었다.

- **Core Contribution**: 이 논문은 ANYmal D 사족 로봇의 GrandTour 데이터셋을 활용해 visual/visual-inertial/LiDAR-visual-inertial SLAM들을 하드웨어 센서 구성 단위로 체계 평가한다. 카메라 모달리티(모노큘러/스테레오/RGB-D), 셔터 타입(global/rolling), IMU 티어(산업용/전술용)를 바꿔 정확도·견고성·연산 자원 간 트레이드오프를 분해해 제시한다.

- **Technical Challenges**: 사족 보행 환경에서는 풋 임팩트 쇼크와 고주파 진동이 시각 추적의 모션 블러/기하 왜곡을 유발해 프레임-프레임 특징 매칭과 추정기 결합이 쉽게 깨진다. 저자들은 GrandTour에서 동일한 로봇 플랫폼과 트랙 조건을 유지한 채 센서만 교체해 실패 원인이 센서-레벨 아티팩트인지, 알고리즘 설계인지 분리하고(셔터/스테레오/IMU 티어), 실패 런을 제외하는 강건한 평가 규칙과 ATE/RPE 기반 정량 지표로 비교한다.

- **Empirical Impact**: 실험 결과, 스테레오 구성은 모노큘러와 RGB-D보다 일관되게 우수했으며, global shutter가 rolling shutter보다 전 프레임 추적 실패를 크게 줄였다. 특히 ORB-SLAM3/RTAB-Map 같은 비전 중심 최적화 프레임워크에서는 IMU를 표준 방식으로 통합할 때 오히려 견고성이 악화될 수 있고, FAST-LIVO2는 전술급 Honeywell IMU에서 더 안정적으로 드리프트가 억제됐다. 결론적으로 사족 로봇 설계 가이드로는 비전 중심이면 global shutter 스테레오(관성은 필요 이상으로 결합하지 않는 방향), LiDAR 기반이면 tactical-grade IMU 채택이 유리하다는 메시지가 강화됐다.



### A Controlled Benchmark of Quantum-Latent GAN Augmentation for Brain MRI (https://arxiv.org/abs/2606.18970)
Comments:
          This work has been submitted to the IEEE for possible publication. This work has been submitted to the IEEE for possible publication

- **Prior Approaches**: 기존 의료영상(특히 뇌 MRI) 데이터 증강은 GAN·확산 모델을 통해 라벨 부족과 클래스 불균형을 완화하는 방식이 널리 쓰였습니다. 최근에는 quantum generative models(예: QGAN)이 정확도 향상을 보고하지만, 단일 실행(seed) 기반 결과가 많고 양자 생성기와 classical 생성기의 파라미터 예산이 같지 않아 ‘양자 구조’가 이득을 만든 것인지 불명확했습니다. 또한 어떤 데이터 규모에서 이득이 나타나는지, 생성 샘플의 품질·다양성은 어떤지까지 함께 분석되지 않는 경우가 흔했습니다.

- **Core Contribution**: 이 논문은 뇌 MRI 증강에서 quantum latent generator의 기여를 분리하기 위한 ‘통제된 벤치마크’를 제시합니다. 이미지 인코딩은 KL-regularized latent space로 고정하고, conditional Wasserstein GAN-GP에서 variational quantum generator와 파라미터 수가 거의 같은 classical generator를 비교합니다. 이후 생성 latents를 디코드해 라벨 데이터 fraction 5%~100% 전 구간에서 다운스트림 분류 성능을 평가합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 양자 생성기와 classical 생성기의 표현력 차이를 ‘공정하게’ 통제하는 것과, 성능 차이를 통계적으로 신뢰할 수 있게 분해하는 것입니다. 논문은 (1) VAE로 latent space를 만들고 (2) quantum generator(4 qubits, depth 2)와 classical generator(파라미터 1648 vs 1632)를 거의 동일한 예산으로 맞춘 뒤 (3) 8개 seed에 대한 paired significance testing과 다중비교 보정을 적용합니다. 또한 downstream 정확도뿐 아니라 intra-set diversity(예: SSIM, 픽셀 표준편차)와 latent distribution overlap(예: t-SNE)으로 생성 샘플이 실제 분포를 얼마나 따르는지까지 함께 진단합니다.

- **Empirical Impact**: 결과적으로 모든 라벨 데이터 fraction에서 real-data-only 대비, quantum 또는 classical 증강 variant이 유의미하게 더 좋지 않았고 두 생성기 간 성능도 통계적으로 구분되지 않았습니다. 낮은 데이터(예: 5%·10%)에서 보이는 ‘소폭 이득’은 생성 샘플이 off-distribution이고 모드 붕괴가 심한 상태에서 나타나 regularization 효과에 가깝다고 해석됩니다. 저자들은 이러한 통제 실험 프로토콜을 공개해, 의료영상 분야에서 quantum generative augmentation의 근거를 더 엄격히 검증하는 testbed로 활용하길 기대합니다.



### Semantic Robustness Certification for Vision-Language Models (https://arxiv.org/abs/2606.18839)
Comments:
          Accepted to ICML

- **Prior Approaches**: 기존 강건성 인증은 주로 입력의 픽셀 수준 변환(Lp ball)이나 회전·이동 같은 기하학적 변환에 초점을 맞췄지만, 실제 환경에서 발생하는 shape/size/style 같은 의미 수준 변화를 그대로 포착하기 어렵습니다. 또 의미 변화를 생성모델 잠재공간에서 다루는 접근은 더 많은 학습 데이터가 각 의미 변이에 필요해 실용성이 떨어진다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 VLM의 open-vocabulary 능력을 활용해 텍스트 프롬프트 두 개를 의미 프록시로 삼고, 의미 변환을 임베딩 공간의 2차원 “semantic plane”에서 매개변수화합니다. extent 구간별로 VLM의 예측이 바뀌지 않는 범위를 수학적으로(결정 경계 closed form) 분할해 “semantic robustness certificate”를 제공합니다. 핵심은 변이마다 추가 데이터 없이도 의미 수준 강건성을 인증 가능하게 만든다는 점입니다.

- **Technical Challenges**: semantic 변이는 입력공간에서 얽혀 있어 임의로 닫힌형(closed-form) 변환을 정의하기가 어렵습니다. 저자들은 코사인 유사도 기반으로 의미 강도를 모델링하고, 의미 변이가 두 텍스트 임베딩이 스팬하는 2차원 하위공간에 국한된다는 구조를 이용해 extent에 따라 임베딩을 회전시키는 변환 γ(φ)를 구성했습니다. 이후 이 변환이 임베딩 공간의 분할 영역(Voronoi decision regions)과 어떻게 만나는지를 분석해, extent 구간을 예측 불변 구간으로 정확히 라벨링하는 절차를 제시합니다.

- **Empirical Impact**: 합성 데이터와 실제 데이터 모두에서, 제안한 semantic 변환이 의도한 의미 변화(예: 모양·스타일·장면 속성 강도)를 따라가며 예측 전환 시점도 extent에 대해 잘 포착됨을 보였습니다. 특히 다양한 시나리오에서 의미 수준 변이에 대한 인증을 수행할 수 있어, 모델의 semantic drift를 모니터링하거나 실패 모드를 진단하는 데 실무적으로 유용하다는 의미가 큽니다. 결과적으로 의미 수준 변이까지 포함한 “실용적 인증”의 새로운 기준점을 제안한 연구로 평가됩니다.



### EDoF-NeRF: extended depth-of-field neural radiance fields using a coded aperture camera (https://arxiv.org/abs/2606.18826)
- **Prior Approaches**: 기존 NeRF는 핀홀 모델을 가정해 학습·렌더링하지만, 입력이 실제 카메라로 촬영되면 DoF(심도)와 광량의 상충 관계로 인해 얕은 DoF에서 고주파 정보가 블러로 손실된다. 이를 보완하려고 Deblur-NeRF, AR-NeRF, DoF-NeRF처럼 디포커스 블러를 인지하는 렌더링/학습 모델을 파이프라인에 넣는 연구가 등장했지만, 전통적인 원형 조리개를 전제로 해서 DoF를 더 크게 확장하는 데는 광학적 제약이 남는다. 또한 디포커스 블러는 고주파 공간 정보가 사라지며 되돌리기 어렵기 때문에, 디블러링 성능이 입력 조건(얕은 DoF)에서 충분히 나오지 못한다.

- **Core Contribution**: 이 논문은 NeRF 입력을 만드는 카메라 자체에 coded aperture(코드 조리개)를 동공(pupil) 위치에 배치해, 디포커스 조건에서도 공간 주파수 성분을 더 보존하도록 설계한다. 그 결과 coded-PSF(코드 기반 점확산함수)까지 포함한 카메라 모델을 NeRF에 통합해, coded 이미지를 그대로 입력받고 extended DoF를 갖는 novel view 합성을 수행한다. 방법은 extended DoF-NeRF(EDoF-NeRF)로 정리된다.

- **Technical Challenges**: 핵심 난제는 coded aperture의 광학 파라미터(초점거리/블러 스케일 등)와 NeRF 네트워크 파라미터를 함께 최적화하면 학습이 불안정해진다는 점이다. 이를 해결하기 위해 2단계 학습 전략을 도입한다: 1단계에서는 단순 핀홀 모델로 네트워크(및 거친 기하)를 안정적으로 학습한 뒤, 2단계에서 coded aperture 렌더링(코드 PSF 컨볼루션)을 적용해 기하와 광학 파라미터를 함께 미세 조정한다. 또한 컨볼루션 경계 영향이 손실함수에 섞이는 문제를 줄이기 위해 패치 샘플링에 margin을 두는 방식으로 학습 안정성과 성능을 끌어올린다.

- **Empirical Impact**: 시뮬레이션(Orchids 데이터 기반)과 광학 실험(레이저 컷 coded aperture 사용)에서 EDoF-NeRF가 기본 NeRF와 DoF-NeRF보다 PSNR/SSIM 및 깊이(geometry) 재구성 품질을 일관되게 개선함을 보인다. 특히 얕은 DoF 입력에서 발생하던 디포커스 블러가 coded aperture 통합 후 충분히 완화되어, 고주파를 포함한 고충실도 렌더링이 가능해졌다는 점이 정량·정성 모두에서 강조된다. 실무적으로는 빛이 부족하거나 장면이 역동적인 상황처럼 DoF-광량 트레이드오프가 치명적인 조건에서 novel-view synthesis의 현실적 한계를 넓힐 수 있다는 의미가 있다.



### Low-Cost Neuromorphic Fall Detection Using Synthetic Event Data and Hybrid SNNs (https://arxiv.org/abs/2606.18732)
Comments:
          4 pages, 6 figures, presented at ICONS 2025 during the Poster Session, but not published

- **Prior Approaches**: 기존 DVS 기반 낙상/행동 인식 연구는 CNN, LSTM, recurrent SNN 등을 DVS 데이터(실제 또는 v2e로 변환)로 학습해 효율성과 성능을 함께 노렸습니다. 다만 실제 DVS 카메라 접근성과 데이터 구축 비용이 높거나, 웨어러블 중심 접근은 사용자 순응도 문제를 남겼습니다. 또한 RGB 영상→이벤트 변환을 쓴 경우에도 낙상 같은 고속 동작을 충분히 잘 살리지 못해 성능·일반화의 격차가 생길 수 있었습니다.

- **Core Contribution**: 이 논문은 스마트폰 RGB 영상을 v2e로 시뮬레이션한 event 기반 데이터로 학습하는 하이브리드 CNN-SNN 프레임워크를 제안합니다. 특히 낙상 감지에 맞춘 커스텀 신경형 데이터셋 NFDD를 구성하고, walking/sitting/falling의 3개 유사 활동을 목표로 정확도와 저비용 접근성을 동시에 노립니다. 결과적으로 하드웨어 의존을 줄이면서 SNN의 에너지 효율성과 시공간 처리 장점을 낙상 탐지에 연결합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) RGB를 DVS처럼 보이게 이벤트로 변환하면서 낙상 시작 구간의 빠른 움직임을 보존하는 것과 (2) 스파이크 비미분 특성 때문에 학습을 안정적으로 수행하는 것입니다. 논문은 v2e의 pixel model(잡음·polarity·지연·대비 민감도)과 Super-SloMo 보간으로 시간 해상도를 보강해 낙상의 단시간 패턴을 학습 신호로 만들고, snnTorch의 surrogate gradient descent로 LIF 뉴런을 학습 가능한 형태로 최적화합니다. 여기에 데이터 증강(수평 플립, 줌인, 경미한 회전)과 입력 해상도(128x128 유지) 선택으로 일반화 성능을 끌어올렸습니다.

- **Empirical Impact**: 실험은 DVS128Gesture 벤치마크에서 약 91.7% 정확도, NFDD에서는 99.7% 수준의 분류 정확도를 보고해 합성 이벤트 학습의 실효성을 입증합니다. 학습/검증 곡선이 안정적으로 수렴해 과적합 우려가 크지 않음을 보여주며, 하드웨어 없이도 사실적인 신경형 데이터 구축이 가능하다는 점에서 실사용 장벽을 낮춥니다. 향후 생체역학 기반 낙상 동역학 분류와 완전 SNN 구현, 저전력 neuromorphic 플랫폼 배포가 중요한 확장 과제로 제시됩니다.



### InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search (https://arxiv.org/abs/2606.18676)
- **Prior Approaches**: 기존 training-free NAS(zer-shot) 프록시는 activation 통계, gradient 성질, expressivity/complexity 같은 휴리스틱을 조각처럼 사용해왔다. 대표적으로 SNIP·GraSP·SynFlow는 gradient/연결 민감도에, NASWOT·ZiCo·Zen-NAS는 NTK 조건/그래디언트 통계/특징 다양성에 초점을 둔다. 하지만 이들은 “무엇이 trainable을 만드는가”를 하나로 묶지 못해, 단일 지표가 검색 공간·데이터셋마다 상관이 흔들리는 문제가 반복된다.

- **Core Contribution**: 이 논문은 아키텍처의 trainability를 학습 절차와 무관한 “intrinsic trainability(내재적 학습가능성)”라는 불변량으로 정리한다. 기하적 capacity(표현의 유효 차원)와 optimization resilience(안정적인 backprop 전파)의 두 축을 정의하고, 이 둘의 비가산적(non-additive) 시너지를 반영하는 scale-invariant 곱 결합으로 InTrain을 제안한다. 즉, 증상 기반 휴리스틱이 아니라 아키텍처가 가진 trainable 조건을 이론-기반 단일 프록시로 운영한다.

- **Technical Challenges**: 핵심 난제는 (1) depth가 다른 아키텍처도 공정하게 비교하고, (2) forward 표현의 용량과 backward 학습 안정성을 데이터 의존성 없이 측정하며, (3) 두 신호의 결합 형태를 타당하게 설계하는 것이다. 논문은 activation covariance eigenspectrum의 participation ratio로 geometric capacity를, synthetic 입력에서 계산한 gradients의 ‘gradient health’(분산/최댓값 기반 누적 지표)로 optimization resilience를 측정한다. 마지막으로 log-product가 아니라 곱 형태의 게이팅으로 두 요소가 한쪽이 무너지면 전체가 무너지는 구조를 반영하도록 설계했다.

- **Empirical Impact**: NAS-Bench-101과 NAS-Bench-201, MobileNetV2 탐색공간에서 InTrain은 랭킹 상관이 ensemble 기반 프록시와 동급 수준을 보이면서 단일 지표 방법을 전반적으로 앞선다. NAS-Bench-201에서는 CIFAR-10/100 및 ImageNet16-120 전반에서 Kendall’s tau와 Spearman 상관이 높게 유지되어 데이터셋별 흔들림이 상대적으로 작다. 또한 ablation에서 PR-only·Grad-only는 유의미하지만 단순 합(PR+Grad)은 오히려 악화되고, 제안한 multiplicative coupling(InTrain)은 시너지로 성능이 상승해 이론이 실험에 그대로 연결됨을 보여준다.



### SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation (https://arxiv.org/abs/2606.18610)
- **Prior Approaches**: 일반 로봇 조작 정책을 실세계에서 평가하려면 물리 로봇 롤아웃, 초기화, 감독 비용이 커서 확장성이 떨어진다. 이를 줄이기 위해 action-conditioned video world model로 정책 롤아웃을 시뮬레이션하고 점수를 매기는 방식이 등장했지만, autoregressive rollouts에서 발생하는 drift 누적과 멀티카메라 관측 간 비일관성 문제가 남는다. 또한 학습 분포 밖에서 행동하는 정책에 대한 evaluator의 일반화도 취약하며, 불확실성 기반 종료를 ensemble 등 추가 학습 없이 일관되게 구현하기 어렵다.

- **Core Contribution**: SC3-Eval은 pre-trained video foundation model을 self-consistent 비디오 생성 레시피로 바꿔, 실세계 성능과 높은 상관을 보이는 closed-loop 정책 evaluator로 적응시키는 방법이다. 핵심은 forward-inverse dynamics consistency(정방향 예측-역방향 동작 복구), cross-view consistency(다른 카메라로부터 inpainting), test-time consistency(역동역학 신호로 per-action-chunk drift 시 롤아웃 조기 종료) 3가지 일관성을 함께 강제하는 점이다. 그 결과 점수의 절대 캘리브레이션뿐 아니라, 실제 롤아웃에서 나타나는 실패 모드까지 더 세밀하게 진단할 수 있다.

- **Technical Challenges**: 기여를 실현하려면 (1) 정방향 모델만으로는 물리적으로 불가능한 프레임을 효과적으로 페널티하기 어렵고, (2) 여러 카메라 관측이 롤아웃 동안 서로 어긋나기 쉬우며, (3) 학습 분포 밖 정책 행동에 대해 evaluator가 계속 신뢰를 유지해야 한다. SC3-Eval은 forward-inverse 모드의 파라미터 공유로 생성 프레임이 ‘요청된 행동을 복구할 수 있는’ 물리적으로 그럴듯한 action manifold에 묶이도록 학습한다. 동시에 카메라 하나를 숨기고 나머지로 다른 뷰를 채우는 cross-view inpainting으로 멀티뷰 일관성을 학습하고, test-time에서는 inverse dynamics로 얻은 action 복구 오차를 불확실성/드리프트 지표로 써 τ를 넘으면 즉시 종료해 누적 오류가 점수에 오염되는 것을 막는다.

- **Empirical Impact**: 실세계 table bussing 7개 vision-language-action(VLA) 정책 체크포인트에 대해 SC3-Eval은 closed-loop Pearson 상관 0.929, MMRV 0.119로 Ctrl-World, IRASim, Cosmos-Predict 2.5 같은 강한 비디오 모델 기반 evaluator를 앞섰다. 또한 reverse table bussing처럼 의미론이 바뀐 out-of-distribution 과제에서도 성능 저하가 관측되지만 평가가 유지되며, aggregate 성공률을 넘어서 실제와 동일한 실패 범주(언어 미준수/물체 들어올림/놓기)를 더 잘 재현한다. 이는 정책 개발에서 체크포인트 선택과 원인 진단을 동시에 강화할 수 있는 평가 도구로서 의미가 크다.



### Splaxel: Efficient Distributed Training of 3D Gaussian Splatting for Large-scale Scene Reconstruction via Pixel-level Communication (https://arxiv.org/abs/2606.18588)
Comments:
          17 pages, 25 figures

- **Prior Approaches**: 기존 3D Gaussian Splatting(3DGS)은 대량의 가우시안 파라미터를 학습해야 해, 대규모 장면에서는 분산 학습이 핵심 병목이 된다. 분산 학습의 대표 접근은 (1) 장면을 영역별로 쪼개 각 GPU가 독립 최적화 후 병합하는 방식인데, 이때 전역 일관성이 깨져 경계에 불연속이 생길 수 있다. 또 (2) Grendel처럼 가우시안 자체를 GPU 간 교환(all-to-all)해 전역 합성을 유지하는 방식은 장면·GPU 규모가 커질수록 통신량이 비선형으로 폭증해 반복(iteration) 시간이 통신에 지배된다.

- **Core Contribution**: Splaxel은 “가우시안 동기화” 대신 “픽셀 레벨 로컬 렌더링 후 전역 합성”으로 분산 3DGS 학습을 재구성해 통신 병목을 근본적으로 완화한다. 각 GPU는 자신의 로컬 가우시안 부분으로 부분 이미지를 만들고, 필요한 것은 최종 픽셀을 구성하는 데 해당하는 부분 픽셀 값(및 전역 합성을 위한 중간 정보)만 교환한다. 이 설계는 장면에 가우시안이 더 늘어나도 통신 비용이 안정적으로 유지되도록 수학적 일관성을 확보하는 것이 핵심이다.

- **Technical Challenges**: 픽셀을 교환하면 전역 alpha blending의 “깊이(깊이순) 누적 순서”가 GPU 간에서 깨질 위험이 있는데, Splaxel은 이를 convex partitioning으로 해결한다. 파티션을 볼록 형태로 나누면 카메라 레이가 각 파티션을 지나는 횟수가 제한되어, 로컬 블렌딩 결과를 전역 레이 누적에 그대로 합쳐도 블렌딩 순서를 보존할 수 있다. 또한 픽셀 레벨 통신에서 생기는 (1) 기하학적 비가시성으로 인한 spatial redundancy, (2) transmittance가 포화되어 뒤 픽셀이 영향이 없는 saturation redundancy를 각각 가시 영역 예측과 누적 transmittance 기반 조기 비가시 판단으로 줄이고, 마지막으로 conflict-free camera-view consolidation으로 GPU 유휴를 줄인다.

- **Empirical Impact**: Big City Street과 Aerial 등 대규모 데이터셋에서 최대 120M 가우시안 규모로 평가했으며, Splaxel은 기존 SOTA 분산 3DGS 대비 최대 7.6× 속도를 달성하면서 재구성 품질을 유지한다. 특히 8 GPU 환경에서 2.9× 속도업도 보고되어, 통신이 반복 시간을 지배하던 문제를 실사용 시나리오에서 해소할 가능성을 보여준다. 대규모 3DGS 학습의 확장성을 가로막던 통신 병목을 “픽셀 레벨”로 전환한 점이 분야 파이프라인 설계에 의미 있는 방향성을 제시한다.



### DART: A design-aware microfluidic chip paradigm for real-time live-cell image analysis (https://arxiv.org/abs/2606.18523)
- **Prior Approaches**: 기존 고처리량 미세유체(microfluidic) 라이브셀 영상 분석은 RoI(region of interest)를 반자동으로 찾고, RoI 주변의 마이크로플루이딕 구조를 수작업/준자동으로 제거하는 절차에 크게 의존했다. 이 방식은 RoI 개수가 늘어날수록 단계가 선형으로 누적돼 실시간 분석이 어려워지고 time-to-insight가 수 시간~수일로 늘어나는 병목을 만들었다.

- **Core Contribution**: 이 논문은 CAD 블루프린트와 실제 칩 물리 배치를 정렬해, RoI 개수에 무관하게 모든 RoI를 자동 위치화하고 이미지 전처리·분석을 수행하는 DART(Design-Aware and Real-Time) 패러다임을 제안한다. DART는 임베디드 fiducial marker와 딥러닝 기반 마커 탐지를 통해 다양한 RoI 형상과 칩 레이아웃에서도 동일한 방식으로 end-to-end 자동화를 달성한다.

- **Technical Challenges**: 핵심 난제는 (1) 칩마다 달라지는 물리적 오차·배치 변동 속에서도 RoI 좌표를 안정적으로 매핑하고 (2) RoI 기하가 제각각일 때도 마이크로플루이딕 구조 제거와 세포 분할 같은 후처리를 견고하게 수행하는 것이다. 논문은 칩 내 fiducial marker로 CAD-물리 정렬 기준을 만들고, 딥러닝으로 마커를 찾아 정렬을 재구성한 뒤 초저지연(40 ms) 구조 제거와 1.1초 내 세포 분할을 가능케 하는 파이프라인을 구성했다.

- **Empirical Impact**: Swiss Army Knife 칩(서로 다른 8가지 RoI 설계, 총 1164개 RoI 위치)에서 검증했으며, 모든 RoI를 5분 내 로컬라이즈하고 원본 현미경 영상에서 마이크로플루이딕 구조를 40 ms에 제거했다. 또한 이미지당 1.1초 미만으로 세포 분할을 포함한 완전 자동 분석을 수행해, hardware-software end-to-end 실시간 분석 기반을 제시하며 closed-loop 및 outcome-driven smart microscopy로의 확장을 뒷받침한다.



New uploads on arXiv(cs.AI)

### Rethinking Reward Supervision: Rubric-Conditioned Self-Distillation (https://arxiv.org/abs/2606.19327)
- **Prior Approaches**: 추론형 LLM 사후학습은 보통 supervised distillation(지도 증류)이나 reinforcement learning(RL)에서 verifiable reward(검증 가능한 보상)를 활용한다. 하지만 distillation은 chain-of-thought 주석을 비싸게 확보해야 하고, 그럴듯한데도 근거가 노이즈·누락·부분오류일 수 있어 학습을 방해할 수 있다. RL(예: GRPO)은 최종 성공/실패 같은 sparse한 scalar 보상으로만 피드백을 압축해, 어떤 중간 단계가 문제였는지 credit assignment가 어렵다.

- **Core Contribution**: 이 논문은 Rubric-Conditioned Self-Distillation(RCSD)로, 루브릭(rubric)을 구조화된 fine-grained 피드백으로 넣어 self-distillation의 교사 신호를 재설계한다. 핵심은 루브릭 점수처럼 scalar로 접지 않고, criterion-level 루브릭에 조건(condition)된 teacher가 학생이 샘플링한 추론 궤적에 대해 토큰 단위 가이드를 제공하게 하는 것. 또한 Stage I에서 인스턴스별 루브릭을 생성하는 루브릭 생성기를 학습하고, Stage II에서 그 루브릭을 이용해 rubric-guided reasoner를 학습한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 고품질 인스턴스 루브릭을 사람이 매번 만들기 어렵고, (2) 루브릭의 텍스트/기준 정보를 학습 중 토큰 수준으로 어떻게 분해해 전달할지다. RCSD는 Stage I로 질문만 보고도 인스턴스별 루브릭을 amortize(분산) 생성하게 만든 뒤, Stage II에서 teacher를 루브릭에 조건해 forward KL distillation으로 학생의 on-policy rollout에 대해 criterion-aware한 토큰 가이드를 주도록 최적화한다.

- **Empirical Impact**: 실험에서 RCSD는 science 추론 벤치마크 전반에서 평균 70.6을 달성하며 GRPO와 OPSD를 각각 평균적으로 1.4점, 0.9점 앞섰다. 특히 ResearchQA와 RubricHub처럼 루브릭 기반/오픈엔드 과제에서 개선 폭이 크게 나타났고, 의학 도메인(MedMCQA, PubMedQA)에서도 기준선 대비 경쟁력 있는 일반화가 관찰됐다. 또한 학습 목표로 forward KL이 가장 유리하고, 생성된(learned) 루브릭도 reference 루브릭에 근접해 사람이 만드는 수고 없이도 효과를 재현할 수 있음을 보였다.



### NeSyCat Torch: A Differentiable Tensor Implementation of Categorical Semantics for Neurosymbolic Learning (https://arxiv.org/abs/2606.19279)
- **Prior Approaches**: 기존 neurosymbolic(NeSy) 접근은 고전논리, 퍼지, 확률 의미론처럼 서로 다른 진리(truth) 정의와 언어를 각각 사용해 지식베이스와 학습 목표의 전이가 잘 안 됐습니다. 또한 NeSyCat 개념이 있었지만, 신경망이 학습한 predicate와 function을 어떻게 의미론적으로 해석하는 연결고리가 부족했습니다.

- **Core Contribution**: 이 논문은 ULLER(통합 언어)에서 제시된 NeSyCat의 진리 정의를 “강한 monad”와 “진리값의 집계(aggregation) 구조”에 파라미터화해, 다양한 의미론을 하나의 틀로 통일합니다. 여기에 NeSyCat Torch를 도입해 신경망이 계산하는 computational predicate/function symbol을 포함한 계산 기호 해석을 제공하며, 확률 프로그래밍과 텐서 백엔드로 구현 가능한 형태로 연결합니다.

- **Technical Challenges**: 핵심 난제는 (1) 분포 기반 reference semantics를 계산하는 과정에서의 marginalization 비용과 (2) 미분 가능하고 수치적으로 안정적인 학습을 동시에 만족시키는 것이었습니다. 논문은 distribution monad로 reference semantics/평가를 만들고, 학습용으로 log-semiring 위의 lazy log-tensor monad를 사용해 필요할 때만 지연(pruning)과 marginalization이 일어나도록 구성했으며, 배치 학습을 위해 batch monad도 함께 사용합니다. 또한 axiom(공리)을 monad 기반 do-notation으로 한 번에 작성하고 monadic bind이 곧 marginalisation이 되도록 구현해 기계적으로 추론·학습 그래프를 구성합니다.

- **Empirical Impact**: MNIST addition에서 HaskTorch, JAX, PyTorch 구현은 LTN과 DeepProbLog보다 속도와 정확도에서 우수했고, DeepStochLog에 가까운 정확도도 달성했습니다. 무엇보다 DeepStochLog처럼 특정 모델에만 맞춘 결과가 아니라, monad를 바꿔 Giry monad처럼 연속확률까지 확장할 수 있는 “균일한(first-order NeSy 전반)” 구현 프레임으로 의미가 큽니다.



### X+Slides: Benchmarking Audience-Conditioned Slide Generation (https://arxiv.org/abs/2606.19256)
- **Prior Approaches**: 기존 연구와 벤치마크는 슬라이드의 완성도나 기술적 깊이를 주로 평가하지만, 실제 사용 맥락에서 핵심인 ‘청중(타깃 오디언스)’을 반영하지 못하는 한계가 컸습니다. 그 결과 동일한 문서라도 전문가형 엄밀성(증명)과 의사결정자형 실행 가능 결론이 서로 다른데도, 이를 구분해 측정하기 어렵습니다.

- **Core Contribution**: 이 논문은 청중 조건에 맞춘 슬라이드 생성 성능을 평가하도록 설계된 벤치마크 X+Slides를 제안합니다. X+Slides는 113개 주제와 7개 프레젠테이션 장면을 아우르는 다중 코퍼스를 바탕으로, 같은 소스-근거(source-grounded) 정보를 청중별로 가중해 ‘청중에게 유용한 정도’를 계량합니다.

- **Technical Challenges**: 핵심 기술 과제는 청중이 달라질 때 필요한 정보가 달라지는데, 이를 객관적이고 공정하게 측정할 수 있는 평가 프로브를 만드는 일입니다. 논문은 8,133개의 중복 제거된 소스-근거 프로브와 동적 평가 프레임워크를 구성하고, Audience Coverage, Domain-wise Coverage, Efficiency, Correctness로 서로 다른 관점을 분해해 측정합니다.

- **Empirical Impact**: 실험에서 DeepPresenter, SlideTailor, NotebookLM(및 ablation)이 청중에게 필수적인 정보의 상당 부분은 회수하지만 여전히 불완전함을 보였습니다. 특히 τ_A=0.7에서 Audience Coverage 최상 성능은 DeepPresenter 0.714, SlideTailor 0.594, NotebookLM 0.853로 나타났고, 시각적 품질이나 주제 폭이 높더라도 소스-근거 평가 없이는 ‘사실/근거 지원’으로 해석하면 안 된다는 메시지를 강화합니다.



### TxBench-PP: Analyzing AI Agent Performance on Small-Molecule Preclinical Pharmacology (https://arxiv.org/abs/2606.19245)
- **Prior Approaches**: 기존 생물학/과학 에이전트 벤치마크는 검증 가능한 분석 과제와 현실 데이터로 “그럴듯한 답”을 줄이려 했지만, 제약업계의 실제 프로그램 의사결정 흐름(해당 단계에서 무엇을 ‘통과/보류/중단’할지)을 충분히 담기 어려웠습니다. 또한 약리·독성·노출·통계·품질관리(QC) 같은 로컬 판단이 데이터 조각과 함께 평가되기보다는, 문헌 상식 재현이나 단일 단서 해석에 취약한 구성이 많았다는 한계가 지적됩니다.

- **Core Contribution**: TxBench-PP는 소분자 전임상 약리(preclinical pharmacology) 의사결정을 대상으로, 에이전트가 제공된 실데이터에서 결론을 “복원(recover)”할 수 있는지 검증하는 verifiable 벤치마크입니다. 각 평가는 실제 워크플로 스냅샷과 파일·메타데이터를 주고, 구조화된 답을 내면 결정론적 채점으로 통과 여부를 가립니다. 특히 문헌 지식이나 잘 알려진 메커니즘에 기대는 전략을 의도적으로 불리하게 설계했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 단계·어세이·과제 구조에 맞춘 현실적 컨텍스트 제공, (2) 에이전트가 데이터 탐색을 거치지 않아도 답이 나오는 “지름길” 제거, (3) QC/통계/생물학적 맥락처럼 판단이 바뀌는 지점을 결정론적으로 채점하는 체계 구축입니다. 논문은 100개 평가에 대해 작업 시점에서의 ‘과학자 수준’ 컨텍스트로 프롬프트를 보정하고, 다중 판단 실수를 유발하는 트랩을 포함하되 채점기가 실제로 지지되는 결론만 통과하도록 구성했습니다.

- **Empirical Impact**: 16개 model–harness 구성(총 11개 모델, 4,800 trajectories)에서 최고 성능도 엔드포인트 시도 59.3%(178/300)를 넘지 못했고, 신뢰할 만한 재현성은 확인되지 않았습니다. 특히 hit prioritization·cheminformatics·프로그램 전환(advance/hold/kill) 같은 복합 의사결정에서 실패가 잦았으며, 수동 트래젝터리 리뷰 결과 method·calibration 오류 비중이 컸습니다. 제약/과학 에이전트 평가 관점에서 “전임상 약리 로컬 판단”을 검증 가능한 형태로 표준화했다는 점에서, 향후 더 긴 호라이즌(임상)·더 넓은 모달리티로 확장될 TherapeuticsBench 로드맵에 실증적 기준선을 제공합니다.



### User as Engram: Internalizing Per-User Memory as Local Parametric Edits (https://arxiv.org/abs/2606.19172)
- **Prior Approaches**: 개인화 메모리는 보통 두 갈래로 해결돼 왔습니다: 프롬프트 기반의 in-context learning/자연어 메모리 파일, 그리고 RAG처럼 검색 시점에 facts를 주입하는 방식이 대표적입니다. 반면 facts를 weights에 직접 넣는 방법은 per-user LoRA처럼 글로벌 가중치 델타를 만들어 content와 reasoning skill이 한 덩어리로 섞이고, 다른 사용자 텍스트에까지 오염(contamination)이 생길 수 있습니다.

- **Core Contribution**: 이 논문은 User as Engram으로, 사용자의 content는 Engram 모델의 해시-키 메모리 테이블에 “국소 편집”으로 저장하고, reasoning skill은 공유 adapter(LoRA)로 분리해 학습·운용합니다. 그 결과 개인화는 인력(메모리)과 추론 능력(기술)을 분업해, 한 사용자의 사실이 다른 사용자의 추론을 망치지 않도록 설계됩니다.

- **Technical Challenges**: 핵심 과제는 “사실을 weights에 쓰지 않고도” 정밀 재현하면서도, 사실 쓰기(write)가 딴짓(불필요한 위치 변경) 없이 정확히 트리거 위치에서만 동작하게 만드는 것입니다. 논문은 Engram의 결정론적 해시 주소와 게이트(gated lookup) 구조를 활용해, 특정 트리거 suffix n-gram이 읽힐 때만 대응 row가 켜지고 값이 추가되며, 나머지 위치는 사실상 마지막 비트까지 그대로 유지되도록 ‘유리 상자(glass box)’ 방식의 편집을 구현합니다.

- **Empirical Impact**: 실험적으로 per-user Engram은 per-user LoRA 대비 평균 간접 추론 정확도를 5.6배 높였고, 단일 사용자 reasoning 성능은 기본(untouched base)보다 나빠지지 않는다고 보고합니다. 또한 메모리 저장이 사용자 수에 따라 ‘검색 부담’ 형태로 커지지 않아, facts가 약 100개를 넘는 구간에서 retrieval 파이프라인을 2.5배 더 큰 모델로도 이기며 다중 사용자(멀티테넌트) 환경에서도 누출 없이 확장성이 좋다는 점이 강조됩니다.



### Beyond Safe Data: Pretraining-Stage Alignment with Regular Safety Reflection (https://arxiv.org/abs/2606.19168)
- **Prior Approaches**: 기존 안전 정렬은 대부분 SFT, RLHF, 안전 보상 기반 RL 같은 post-training에서 이뤄져 왔지만, 프롬프트 조작이나 finetuning 공격에 의해 안전 가드레일이 약해지거나 깨질 수 있다. pretraining 단계에서는 data filtering(유해/독성 문서·토큰 제거)과 data rewriting(유해를 더 안전한 형태로 변환)이 널리 쓰이지만, 핵심은 ‘무엇을 학습하느냐’ 통제에 머무는 경향이 있다. 저자들은 “데이터를 안전하게 만들면 충분한가?”에 의문을 제기하며, 안전해 보이는 지식이 조합·일반화를 통해 위험 행동으로 이어질 수 있다고 지적한다.

- **Core Contribution**: 이 논문은 pretraining-stage alignment가 단순히 학습 데이터를 안전하게 만드는 수준을 넘어, 모델이 생성 중 스스로를 점검(self-monitoring)하는 능력을 토큰 학습 과정에 내재화해야 한다고 주장한다. 이를 위해 Safety Reflection Pretraining(SRP)을 제안하며, pretraining 코퍼스에 정기적으로 짧은 safety reflection(앞 구간의 안전 판정 + Safe/Unsafe 및 범주)을 삽입한다. 또한 이러한 pretraining에서 학습한 reflection 습관이 추론·후속 post-training에서도 유지되도록 “호환되는” post-training 설계를 함께 강조한다.

- **Technical Challenges**: SRP의 핵심 도전은 (1) 모델이 실제로 reflection을 ‘언제나’ 같은 방식으로 수행하도록 학습시키고, (2) reflection이 세이프/언세이프 분기뿐 아니라 생성 중단(eos 등)으로 이어지게 만드는 데 있다. 저자들은 문장 경계 기반 세그먼트 분할 후 각 세그먼트에 Qwen3Guard-Gen-0.6B로 safety judgment을 생성·삽입하고, Unsafe 판정 뒤에는 중단을 유도하는 신호를 추가하는 방식으로 구현한다. 아울러 MedSafetyWorld 같은 합성 환경에서 safe 데이터만으로도 위험 지식이 일반화될 수 있음을 검증하고, SRP가 필터링·리라이팅보다 이를 더 잘 차단함을 ablation으로 확인한다.

- **Empirical Impact**: 실험에서는 FineWeb-Edu에서 1.7B 모델을 SRP로 pretraining했을 때 safety classification 정확도가 개선되고, inference-stage 및 finetuning 공격의 성공률이 크게 감소했다. 특히 MedSafetyWorld에서는 안전 데이터에서 유도된 unsafe 일반화로 인해 행동이 바뀌는 문제에서 SRP가 data filtering 및 data rewriting 대비 우월한 완화 성능을 보였다. 더 나아가 안전이 post-training에서 얕게 정렬될 수 있다는 취약성 관점에서, pretraining이 ‘안전한 행동이 획득될 가능성’ 자체를 형태화해야 한다는 메시지를 실증적으로 강화한다.



### Human-AI Coevolution Dynamics: A Formal Theory of Social Intelligence Emergence Through Long-Term Interaction (https://arxiv.org/abs/2606.19144)
- **Prior Approaches**: 기존 대화형 AI는 감정 모델링, 메모리 검색, 페르소나 조건화처럼 사회적 행동을 구성요소 단위로 분리해 다루는 경우가 많습니다. 그 결과 장기적인 상호작용에서 안정적인 사회적 관계가 어떻게 형성·유지되는지, 사회지능이 어떤 동역학으로 ‘나타나는지’를 하나의 틀로 설명하기 어렵습니다.

- **Core Contribution**: 이 논문은 Human-AI Coevolution Dynamics Framework (HACD-H)를 제안하며, 인간-에이전트 상호작용을 자기조직화되는 사회적 인지 시스템으로 형식화합니다. HACD-H는 감정 적응, 관계 조직, 사회적 기억, 성격 일관성을 단일 동역학 프레임워크로 통합하고, 다중 timescale 사회인지, relational attractors, trust basins, 발달 단계 전이, social cognitive energy 같은 원리를 제시합니다.

- **Technical Challenges**: 핵심 과제는 장기 상호작용의 복잡한 사회적 현상을 분리된 모듈이 아니라 ‘동역학적 체계’로 모델링하는 데 있으며, 이를 검증 가능한 이론-데이터 연결로 바꾸는 것입니다. 논문은 약 14,700개의 interaction turn으로 이론 기반 대화 데이터셋과 평가 프레임워크를 구축하고, 에너지 지형·단계 전이·관계적 끌개를 실증적으로 관찰할 수 있게 설계했습니다.

- **Empirical Impact**: 실험 결과 사회인지에는 시간 스케일별 지속성의 위계가 존재하고, stable relational attractors와 phase-transition-like 발달 패턴이 나타납니다. 또한 social intelligence는 social cognitive energy와 유의한 음의 상관관계를 보이며(r = -0.391, p < 0.001), 상호작용 궤적에서 에너지가 점진적으로 감소하는 양상이 보고되어 사회지능이 고립된 대화 능력보다 장기 공진에서 유래함을 시사합니다. HACD-H는 적응형 인간-에이전트 사회 상호작용을 통합적으로 모델링하고, 사회적으로 지능적인 AI를 설계하는 데 이론적 기반을 제공한다는 점에서 의미가 큽니다.



### Analysing drivers and interdependencies in European electricity markets using XAI (https://arxiv.org/abs/2606.19118)
Comments:
          12 pages

- **Prior Approaches**: 기존 전력가격 예측(EPF)은 수요·발전믹스·연료비·기상 등 기초 변수를 쓰는 백박스(계량경제) 모델이 주류였지만, 전력계통의 강한 비선형성과 고차 상호의존성을 충분히 반영하기 어렵다는 한계가 있었다. 한편 DNN은 예측 정확도를 끌어올렸지만 블랙박스라 정책·규제 관점의 ‘가격 형성 요인’ 해석이 약하다는 문제가 남았다.

- **Core Contribution**: 이 논문은 DNN 기반 예측 프레임에 XAI를 결합해 39개 유럽 입찰구역에서 전력가격 형성의 결정요인을 정량 분석한다. 특히 SHAP을 쓰되 고차원 설명의 해석성을 높이기 위해 SSHAP를 적용·확장하고, EU 단일가격 시나리오를 보기 위한 합성 EU-시장도 구축한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 입력 특성이 364개로 커서 SHAP의 전역 해석이 왜곡되기 쉽고, (2) 시장 간 상호연결로 인해 ‘자국 요인’과 ‘이웃 요인’을 구분해 설명하기 어렵다는 점이다. 저자들은 conditional expectation 기반 SHAP 계산과 Monte-Carlo 추정으로 국소 기여를 안정화하고, Super-SHAP(SSHAP)로 특성 군집 간 상쇄를 반영해 고차원에서의 전역 중요도를 재구성했다.

- **Empirical Impact**: 실험 결과, 태양광처럼 발전 비중이 낮은 재생에너지원이 가격 형성에서 비중 대비 과도하게 큰 영향력을 보이며(EE 모델 기준 EU에서 최대 super-feature로 나타남) 가스 가격은 전 시장에서 일관되게 지배적 드라이버로 확인된다. 또한 이웃 입찰구역(super-feature)의 영향이 평균적으로 매우 커(다수 구역에서 60% 안팎) 유럽 전력시스템의 강한 상호의존성을 실증하며, 특히 스웨덴은 이웃 의존도가 매우 높고 스페인은 상대적으로 ‘격리’된 패턴이 두드러진다.



### Towards an Agent-First Web: Redesigning the Web for AI Agents (https://arxiv.org/abs/2606.19116)
- **Prior Approaches**: 기존 연구는 WebShop, WebArena, Mind2Web처럼 에이전트의 웹 탐색·과업 수행 능력(behavior)을 키우는 데 집중했지만, 웹 자체의 접근·경제·콘텐츠 구조를 바꾸는 개입은 제한적이었다. 프로토콜 측면에서는 MCP, A2A, ACP, NLWeb 같은 표준화가 나왔지만 접근(허용/차단)과 비용 정산, 그리고 생성 콘텐츠가 만들어내는 지식의 붕괴 문제까지 함께 다루지 못했다. 따라서 agent-first 웹에서 발생하는 “인간 중심 가정의 붕괴”를 전 층위에서 설명하거나 해결하는 틀은 부재했다.

- **Core Contribution**: 이 논문은 웹이 AI 에이전트를 중개자(대리 사용자)로 두는 순간 기존 3대 가정(접근권, 경제 교환, 콘텐츠 의미 구조)이 동시에 깨진다고 진단하고, 이를 access·economics·content 세 층 동시 재설계로 해결하려고 한다. 핵심 철학은 “인간을 대신해 행동하는 agent는 자신이 대표하는 인간과 동등한 presumption of access 및 책임을 갖는 1급 시민”이라는 에이전트-as-human-proxy 원칙이다. 이 관점을 바탕으로 에이전트 식별·의도 기반 접근, 의도 중심 경제 티어, 그리고 epistemic recursion을 막는 콘텐츠 규격과 검증 체인을 제안한다.

- **Technical Challenges**: 가장 큰 기술 난제는 에이전트가 요청을 보내는 방식이 현재는 인간·봇을 구분할 메타정보가 부족해 “차단/허용”만 가능한 이진 정책으로 귀결된다는 점이다. 논문은 HTTP 요청에 agent identification metadata를 추가하고, robots.txt의 honor system을 보완하는 agents.txt를 통해 의도-aware rate limiting과 점진적 듀얼 레이어 콘텐츠 서빙(인간용/에이전트 최적화)을 설계해 차별적 처리를 가능하게 한다. 또한 경제 측면에서는 페이지뷰·클릭 같은 인간 참여 지표가 사라지므로, 토큰 기반 구독 및 의도 기반 커미션드 콘텐츠 경제로 가치 교환 단위를 재정의하려 한다.

- **Empirical Impact**: 정량적으로는 AI 에이전트 접근이 확산되며 인프라가 기본 차단으로 이동하는 흐름(예: crawl-to-referral 비대칭)과, zero-click 검색 증가·CTR 하락·출판 트래픽 급감 같은 경제 붕괴 신호를 함께 제시해 문제의 규모를 뒷받침한다. 콘텐츠 층에서는 AI 생성물이 다시 에이전트에 의해 소비되며 지식의 기반이 서서히 이탈하는 epistemic recursion을 구조적 위험으로 규정하고, ATML(에이전트용 텍스트 마크업), 인간 감독 tier, cryptographic provenance chain으로 재발 방지를 목표로 한다. 결과적으로 “반응적 패치”가 아니라 agent-first 인터넷을 위한 10개 설계 원칙이라는 통합 로드맵을 제공한다는 점에서, 보안·프로토콜·비즈니스·콘텐츠 아키텍처 전반에 영향이 크다.



### ARIADNE: Agnostic Routing for Inference-time Adapter DyNamic sElection (https://arxiv.org/abs/2606.19079)
- **Prior Approaches**: 기존 어댑터 선택(라우팅) 연구는 라우터를 추가로 학습해 입력을 어댑터에 매칭하는 방식이 많았고, 이는 새 어댑터가 늘 때마다 확장성이 떨어진다. 또 다른 접근은 Arrow/SpectR처럼 LoRA 가중치의 SVD·공분산 등 어댑터 내부를 기반으로 신호를 만들지만, LoRA 중심 설계라 다른 PEFT로의 일반화가 어렵고 유사 태스크에서 거의 랜덤 수준으로 붕괴하는 문제가 보고됐다.

- **Core Contribution**: ARIADNE은 학습 없이(training-free) 추론 시점에 어댑터를 고르는 라우팅 프레임워크로, 어댑터 가중치나 내부 접근 없이 입력 임베딩 공간에서만 결정을 내린다. 각 어댑터를 해당 태스크 학습데이터 임베딩을 클러스터링해 얻은 다중 centroid로 표현하고, 라벨 없는 입력은 이 centroid 집합과의 근접도를 통해 가장 적합한 어댑터를 선택한다. 그 결과 어떤 PEFT 구조에도 호환되며, 어댑터/베이스모델을 수정하거나 추가 학습을 요구하지 않는다.

- **Technical Challenges**: 핵심 기술 난점은 “어댑터 내부 정보 없이도 태스크 분포를 구분할 수 있는 표현 공간”을 찾고, centroid가 태스크 내 변이를 충분히 담도록 설계하는 것이다. ARIADNE은 고정(frozen) 텍스트 인코더의 잠재 기하를 사용해 태스크별 centroid 다중 세트를 구성하고, 단일 평균값이 만드는 저밀도 표현 붕괴를 줄이기 위해 여러 개의 로컬 centroid로 멀티모달성을 캡처한다.

- **Empirical Impact**: Llama 3.2 1B Instruct에서 23개 다양한 NLP 태스크를 평가했을 때 ARIADNE은 oracle 상한 대비 평균 성능의 97.44%를 회복했으며, 23개에서 어댑터 Selection Accuracy( SA )는 평균 85%로 보고됐다. 또한 44개 태스크로 확장해도 평균 SA가 89.7%로 안정적이며, 실패가 발생하더라도 의미적으로 가까운 태스크 클러스터 내에서 주로 일어나 ‘graceful degradation’ 형태로 성능이 급격히 무너지지 않는 점이 관찰됐다.



### RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents (https://arxiv.org/abs/2606.19047)
- **Prior Approaches**: 기존 Agentic RL은 툴 사용을 환경 상호작용으로 학습하지만, multi-turn 도구 사용으로 확장하면 긴 호라이즌 구조적 일관성을 유지하면서 충분한 학습 데이터를 확보하는 문제가 커집니다. 오프라인 대규모 합성은 데이터 부족을 메우지만 학습 중 모델의 능력 경계가 이동하는 상황을 추적하지 못해 정보 신호가 빠르게 희석됩니다. 한편 고정 seed로 학습하는 RL/EnvTuning류는 적은 데이터로는 출발하지만, 능력 경계가 옮겨가면 static pool의 gradient signal depletion이 쉽게 발생합니다.

- **Core Contribution**: 이 논문은 GRPO의 progress reward 신호가 능력 경계 근처에서 분산이 크게 나타난다는 관찰을 활용해, Reward-driven Online Data Synthesis(RODS)로 “경계 샘플”을 학습 중 실시간으로 찾아내는 프레임워크를 제안합니다. RODS는 새 데이터를 무작정 생성하지 않고, 경계에 해당하는 과제에 대해서만 멀티턴 변형을 만들어 replay buffer가 정책과 함께 동적으로 진화하도록 설계합니다. 결과적으로 데이터 생성-훈련 루프를 닫아 signal starvation을 완화하는 동시에 multi-turn 의미 일관성 문제도 줄이려는 것이 핵심입니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 추가 추론 비용 없이 경계 샘플을 안정적으로 판별하고, (2) 합성된 multi-turn 변형이 API 의존성·구조 난이도는 유지하면서 표면 표현만 다양해져야 하며, (3) 합성 데이터 유입이 RL 학습을 흔들지 않게 라이프사이클을 관리하는 데 있습니다. RODS는 rollout에서 이미 계산되는 통계로 progress reward 분산의 “제로 코스트” 경계 탐지를 수행하고, skill-aligned resampling으로 API 토폴로지와 dependency depth 같은 구조적 복잡도를 종자와 가깝게 맞춥니다. 또한 후보 큐를 epoch 경계에 주입하고 mastered/unsolvable 영역은 퇴출하는 다층 retirement로, 분포 급변과 고인 데이터 누적을 동시에 억제합니다.

- **Empirical Impact**: BFCL V3 multi-turn(훈련 400/평가 400)에서 RODS는 Qwen3-4B-Instruct 기준 전체 성능을 56.00%까지 끌어올려 static dataset(50.00%)과 EnvTuning(50.50%)을 능가했습니다. 17K 규모 오프라인 파이프라인과 동등 성능을 보이면서도 약 20배 적은 trajectory(~800 활성 풀, 400 seed+경계 생성)로 달성해, “얼마나 많이”가 아니라 “어디(경계)에서” 데이터를 합성하는 것이 더 중요하다는 점을 실증합니다. 고정 데이터/환경 증강 대비 OOD 벤치마크에서도 일관된 개선이 보고돼, 구조적 등형성(structural isomorphism) 기반의 일반화 효과가 의미 있는 것으로 해석됩니다.



### ThinkDeception: A Progressive Reinforcement Learning Framework for Interpretable Multimodal Deception Detection (https://arxiv.org/abs/2606.18988)
Comments:
          10pages,4figures

- **Prior Approaches**: 기존 멀티모달 deception detection은 end-to-end black-box 분류에 머무는 경우가 많아, 판단 근거(추론 경로)가 투명하게 제시되지 않는 문제가 큽니다. 또한 시각·음향 단서의 미세한 교차 불일치(cross-modal inconsistencies)를 명시적으로 학습하기 어렵고, 데이터 규모·도메인 차이로 인해 과적합 및 교차도메인 일반화 성능이 제한됩니다.

- **Core Contribution**: ThinkDeception은 멀티모달 Large Language Model(MLLM)의 추론 능력을 deception detection에 이식해, 이진 분류를 ‘인지적 추론 과정’으로 전환하는 해석가능 프레임워크를 제안합니다. 이를 위해 단계별 멀티모달 Chain-of-Thought(생각-단계-답) 형식과 함께, 시각-음향 일관성 및 논리 반성까지 포함하는 학습 체계를 구성합니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) CoT 추론을 직접 감독할 고품질 단계 주석 데이터의 부재, (2) RL 적용 시 희소 보상으로 인한 학습 불안정과 hallucination, (3) 쉬운-어려운 샘플이 섞인 도메인 이질성입니다. 논문은 Deception-10K(10,000개, 정밀 타임스탬프 정렬 포함)로 추론 데이터 격차를 메우고, teach-then-align(SFT) 후 VAC-GRPO에 progressive curriculum(4단계 난이도)과 과정 인식 보상(Visual-Audio Consistency Reward, cross-modal inconsistency logic reward)을 결합해 RL의 안정성과 해석 신뢰도를 동시에 끌어올립니다.

- **Empirical Impact**: Deception-10K 및 외부 벤치마크에서 ThinkDeception은 detection accuracy와 reasoning quality 모두에서 SOTA를 달성했으며, 평균 정확도 73.76%로 2등 대비 절대 8.52%p 격차를 보입니다. 특히 텍스트 기반 단서에 과도하게 의존하는 대다수 기준모델이 50%대(무작위 수준)로 무너지는 양상이 나타나, 이 논문의 ‘모달 불일치 기반 추론’이 실제로 도메인 전이를 견인함을 실증합니다.



### RTSGameBench: An RTS Benchmark for Strategic Reasoning by Vision-Language Models (https://arxiv.org/abs/2606.18950)
Comments:
          First two authors contributed equally

- **Prior Approaches**: 기존 VLM 평가용 게임 벤치마크는 텍스트 중심이거나(또는) 정해진 단일 시나리오에 머무는 경우가 많아, 장기 시퀀셜 의사결정과 멀티에이전트 전략 추론을 충분히 진단하기 어렵다. RTS 벤치마크도 StarCraft II 기반이 주류인데, 평가 범위가 제한되고(대부분 1v1) 개별 전략 역량을 체계적으로 쪼개 검정하기가 힘들며, 시나리오 커버리지가 고정돼 포화(saturation) 우려가 있다.

- **Core Contribution**: 이 논문은 RTSGameBench를 제안해 BAR(Beyond All Reason) 기반의 대규모 RTS에서 VLM의 전략적 추론을 ‘총체적 게임 평가 + 역량별 진단 + 자동 확장’으로 나눠 측정한다. 특히 매치업 구조(듀얼/대칭·비대칭 팀/FFA)와 RTS AI 챌린지 분류에 근거한 mini-game(자원관리·공간/시간 추론·상대 모델링·협업·적대적 계획)를 제공해 실패 원인을 특정 역량으로 연결한다. 또한 자유형 질의로 새로운 mini-game을 생성·검증·품질개선하는 self-evolving 게임 생성 프레임워크로, 고정된 평가 세트의 한계를 완화한다.

- **Technical Challenges**: 대규모 RTS에서 VLM이 안정적으로 작동하려면(수백 유닛, 긴 게임 지속, 부분관측) 단발 호출로는 맥락이 끊기고, 유닛 단위 제어는 스케일을 감당하기 어렵다. 이를 위해 RTSGameAgent를 만들어 per-unit 이동을 group assignment·group movement로 바꾸고, 각 그룹에 FSM 기반 전술 모드(move/move_force/stop/fight)를 부여해 엔진이 전투 타이밍을 처리하게 한다. 더불어 decision 간 맥락 손실을 줄이기 위해 short-term event log와 long-term 경험 요약을 agentic memory로 관리하며, LLM이 관련 항목을 선별해 다음 추론에 전달한다.

- **Empirical Impact**: 실험 결과 여러 SOTA VLM은 대칭 팀보다 비대칭 팀(수적 열세)처럼 협업·장기 조율 요구가 커지거나, multiagent coordination과 task scale이 증가할 때 성능이 급격히 떨어졌다. 진단 mini-game에서도 역량별 격차가 뚜렷해, 예컨대 MFD·SP·FS-T에서 모델별 강약이 갈리며 ‘생산 계획이 적대적 계획으로 자연스럽게 일반화되지 않는다’는 신호가 나타난다. 또한 self-evolving 생성 프레임워크는 생성된 mini-game의 playability와 인간 선호 평가를 통해 확장성의 실용성을 보였고, RTS에서 전략적 추론을 체계적으로 진단·확장하는 표준 도구로서 의미가 크다.



### Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents (https://arxiv.org/abs/2606.18947)
Comments:
          15 pages, Figure 8

- **Prior Approaches**: 기존 접근은 LLM의 native search grounding처럼 모델-공급자 경계 안에 검색 정책(프로바이더 선택, 결과 형식, 증거 주입, 비용·지연)을 숨겨두는 방식이 주류입니다. 이 때문에 검색 품질과 운영 지표를 튜닝·검사·이식·재사용하기 어렵고, 엄격한 출력 계약(예: JSON/단일 엔티티)을 깨는 Search-Induced Verbosity 같은 포맷 드리프트 위험이 커집니다. 또한 RAG나 도구 사용 연구는 검색-추론 상호작용을 다루지만, ‘실시간 검색 인터페이스’를 명시적 시스템 경계로 다루는 관점은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 Vendor-agnostic 경계로 검색 근거를 분리하는 Decoupled Search Grounding(DSG)을 제안합니다. DSG는 MCP-compatible gateway를 통해 검색을 추론 모델 바깥의 구조화된 tool 계층으로 옮기며, 프로바이더 라우팅, 출처 기반 context 렌더링, fallback, retrieval-depth, exact/semantic caching을 1급 제어(control)로 노출합니다. 결과적으로 추론 모델은 교체 가능하게 유지하면서도, 검색은 ‘운영 가능한 인터페이스’로 취급할 수 있게 됩니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 프로바이더별 이질적인 결과를 표준화해 모델이 일관된 근거를 받게 하고 (2) 캐시·fallback·비용/지연 목표를 정책으로 안정화하며 (3) 출력 계약 위반(Search-Induced Verbosity)을 인터페이스 레벨에서 완화하는 것입니다. DSG는 provider registry와 YAML 어댑터로 다양한 검색 백엔드를 같은 내부 결과 객체로 정규화하고, 캐시 키를 provider-scoped로 관리해 서로 다른 공급자 결과가 섞여 재사용되는 문제를 줄입니다. 또한 semantic cache의 유사도 임계값과 time-to-live(신선도 우선)를 설정해 재현성과 최신성 간 균형을 맞추며, 툴 호출/응답이 명확한 경계가 되도록 설계합니다.

- **Empirical Impact**: 5개 frontier 모델과 SimpleQA/FreshQA/HotpotQA, 그리고 e-commerce Query Intent Understanding(QIU) 프로덕션 워크로드에서 native search와 비교해 비용·지연·품질 트레이드오프를 체계적으로 입증합니다. SimpleQA에서 DSG는 86.1%(native 87.7%)에 근접하면서 검색 비용은 91% 절감했고, warm-cache hit rate는 99.4%, 지연은 68% 감소했습니다. FreshQA는 native가 앞섰지만(만약 신선도 이점이 강한 경우), QIU에서는 DSG가 native와 비슷하거나 약간 상회하면서 검색 비용을 98% 이상 줄였으며, 모델 공급자에 종속되지 않는 ‘운영 최적화 가능한 검색 경계’의 실용성을 강조합니다.



### SciRisk-Bench: A Risk-Dimension-Aware Benchmark for AI4Science Safety (https://arxiv.org/abs/2606.18936)
- **Prior Approaches**: 기존 AI4Science 평가는 과학 지식·추론·문제풀이(예: SciBench, ScienceQA, SciEval 계열)에 집중하는 경우가 많아, 정작 “안전하게 회피·완화하는가”를 충분히 드러내지 못했습니다. 안전 벤치마크도 ChemSafetyBench, MedSafetyBench, LabSafetyBench처럼 특정 분야나 일반 안전 범주에 치우쳐 있어, 어떤 위험 메커니즘이 어떤 과학 맥락에서 실패하는지 세분 진단이 어렵습니다. 결과적으로 안전 점수가 하나로 뭉개지며 대응(절차 가이드, 규정 근거, 불확실성 표현 등)이 구체화되지 않는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 AI4Science 안전을 “위험 차원(risk dimensions)”과 “과학 분야(disciplines)”의 2축으로 함께 평가하는 SciRisk-Bench를 제안합니다. 7개 분야, 31개 하위 분야에 대해 10개 위험 차원을 명시 라벨링해, 단순히 “어떤 분야가 위험한가”가 아니라 “어떤 위험 메커니즘이 실패를 만드는가”를 분해해 봅니다. 그 결과 모델의 안전 취약 지점을 더 해석 가능하게 진단할 수 있는 틀을 제공합니다.

- **Technical Challenges**: 핵심 기술 과제는 과학 맥락에 종속된 안전 위험을 객관적으로 라벨링하고, 텍스트 생성 결과를 일관되게 위험 판정하는 데 있습니다. SciRisk-Bench는 LLM-as-a-judge 패러다임으로, 위험 차원이 주어진 과학 프롬프트에 대해 모델이 생성한 응답을 심사 LLM이 “안전 이슈를 유발/조장하는가” 이진 결과로 변환해 공격 성공률(ASR)을 계산합니다. 또한 위험 차원과 하위 분야를 함께 태깅해, 안전 실패가 과학 능력의 부족인지(또는 과잉 확신/생략인지) 구분되도록 설계했습니다.

- **Empirical Impact**: 실험에서 지식 시점 부정확(knowledge cutoff drift)과 안전 관련 생략(safety omission), 실험실 안전(laboratory safety)이 높은 ASR을 보여, 위험이 반드시 악성 요청 형태로만 나타나지 않음을 확인했습니다. 반대로 개인정보 유출(privacy leakage)은 상대적으로 낮게 나타나 정렬(alignment)이 정보통제 유형에서 더 강하게 작동할 가능성을 시사합니다. 또한 science-specialized LLM이 대부분의 위험 차원·분야에서 ASR이 오르는 경향이 관찰되어, 도메인 파인튜닝이 “안전 인식”까지 자동으로 강화하지는 않으며 오히려 더 많은 기술적 답변 가능성이 위험을 키울 수 있음을 보여줍니다.



### Skill-Guided Continuation Distillation for GUI Agents (https://arxiv.org/abs/2606.18890)
- **Prior Approaches**: GUI 에이전트는 보통 expert trajectory에서의 동작을 그대로 따라하는 behavior cloning(behavior cloning on successful expert trajectories)으로 학습한다. 하지만 현재 정책이 조금만 벗어나도, closed-loop 실행 중 expert가 방문하지 않은 상태(policy-induced off-trajectory states)에 도달해 그 상태에 대한 유효한 supervision이 사라진다. RL로 보완하려 해도, 현재 정책이 정답 행동을 낼 확률이 낮아 sparse reward와 비효율 문제가 두드러진다.

- **Core Contribution**: 이 논문은 off-trajectory supervision deficit을 문제의 핵심으로 정의하고, 이를 Skill-Guided Continuation Distillation(SGCD)로 메운다. SGCD는 먼저 skill 없이 plain policy를 굴려 현실적인 off-trajectory 상태를 만들고, 그 상태에서 skill-guided policy가 성공적인 continuation을 생성한 뒤 이를 expert 데이터와 섞어 학습한다. 또한 스킬을 고정 경로 재생이 아닌 trajectory abstraction(Continuation Plans, Critical Targets, Failure Traps, Success Criteria)으로 추출해, 현재 화면 상태에 대한 회복을 유도한다.

- **Technical Challenges**: GUI 도메인에서 realistic off-trajectory 상태를 다시 만드는 비용이 크고, “어떤 상태를 골라야 하는지”가 곧 선택 편향(selection bias)으로 이어진다. SGCD는 이를 피하려고 plain policy의 초기 실행 구간에서 handoff depth k를 1~20으로 스윕해, 현재 정책이 실제로 겪는 상태를 촘촘히 커버한다. 또한 success continuation을 얻기 어렵다는 점을 verifier와 LLM judge로 이중 필터링해 학습용 데이터 품질을 통제하며, 실패 패턴이 반복된다는 점을 바탕으로 스킬 스키마를 설계해 유효한 복구 신호를 만든다.

- **Empirical Impact**: OSWorld-Verified에서 SGCD는 3개 베이스 모델의 성공률을 low-30%대에서 50% 이상으로 끌어올려 일반성과 효과를 동시에 입증한다. backbone-matched 비교에서도 Qwen3-VL 계열에서 20%p 이상(8B), 25%p 이상(30B-A3B)의 큰 개선이 일관되게 나타난다. 더 나아가 distillation 이후 plain policy가 skill 프롬프트 없이도 off-trajectory 상태에서 task를 이어갈 수 있음을 Continuation Success Rate로 확인하며, “스킬 가이드가 데이터 생성용 보조에 그치지 않는다”는 점까지 보여준다.



### Generative-Model Predictive Planning for Navigation in Partially Observable Environments (https://arxiv.org/abs/2606.18888)
- **Prior Approaches**: 부분관측 환경에서의 로봇/자율주행 내비게이션은 POMDP의 belief MDP 관점에서, 과거 관측의 “믿음(belief)”을 확률적으로 유지하며 의사결정을 내리는 것이 정석으로 여겨져 왔습니다. 다만 기존 신경망 기반 belief 추정은 대부분 belief space를 단일 모드(point estimate)로 축약해 perceptual aliasing 같은 다중가능성(multimodality)을 제대로 반영하지 못했습니다. 한편 생성모델은 분포 표현에 강점이 있지만, expert demonstration이나 대규모 데이터가 필요하거나, 명시적 장기 계획 메커니즘이 약해 안전하고 효율적인 경로 탐색으로 이어지지 못하는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 BeliefDiffusion으로, “생성(generation)과 계획(planning)”을 한 프레임워크에서 결합해 관측 이력에 맞는 다중 모드 belief을 명시적으로 만들고 그 위에서 계획을 수행합니다. 구체적으로 diffusion 모델로 조건부 map(국소 점유 격자) 분포를 생성해 multimodal belief을 구성한 뒤, Model Predictive Control(MPC)로 샘플된 여러 가설 지도에 대해 동시에 앞으로 내다보며 액션 시퀀스를 선택합니다. 결과적으로 agent는 보이지 않는 세계를 하나의 정답으로 단정하지 않고, 가능한 환경들에 걸쳐 robust한 행동을 고르는 “imagine then plan”을 반복합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) high-dimensional belief와 perceptual aliasing 때문에 belief가 본질적으로 다중 모드가 되는 상황에서 이를 안정적으로 생성·표현하는 것, (2) 생성된 가설들 위에 장기 planning을 얹어 계산량을 통제하면서도 성능을 확보하는 것입니다. 저자들은 지도 레이아웃 추론을 conditional generative modeling으로 보고, DDPM 기반 diffusion 모델이 observation history를 조건으로 로컬 점유 격자 분포를 복원(또는 재구성)하도록 학습합니다. 또한 classifier-free guidance로 조건부/비조건부 생성의 균형을 조절하고, multi-head attention을 통해 grid cell별로 관련 관측 이력에 선택적으로 집중하게 만들어(이력 길이 변화 및 셀 미방문 처리 포함) 조건 정보를 효율적으로 결합합니다.

- **Empirical Impact**: 합성 지도 환경에서의 실험에서 BeliefDiffusion은 navigation success rate와 경로 효율(path efficiency) 모두에서 model-free reinforcement learning 기반 기준선과 다른 생성 접근법을 유의하게 능가합니다. 특히 unimodal belief 근사에 의존하는 방식보다 perceptual aliasing에 강한 의사결정(잘못된 전환 감소 및 목표 도달 안정성)을 보여, 부분관측 내비게이션에서 multimodal belief의 실용적 가치가 확인됩니다. 데이터 효율 측면에서도 대규모 정책 학습 없이 생성된 소수 가설 위에서 MPC로 계획하는 구조가 장기 계획의 불가능성을 “가설 묶음에 대한 국소 최적화”로 우회해, 알려진 생성모델 대비 실질적인 성능 개선을 이끕니다.



### Externalizing Research Synthesis and Validation in AI Scientists through a Research Harness (https://arxiv.org/abs/2606.18874)
Comments:
          65 pages, 14 figures, 19 tables

- **Prior Approaches**: AI Scientist, EvoScientist 같은 자동 연구 에이전트는 아이디어 생성→코드 작성→실험 실행을 end-to-end로 수행하지만, 중간 의사결정의 근거가 프롬프트나 모델 내부에 숨겨져 감사(audit)가 어렵다는 한계가 지적돼 왔다. 또한 무작위 검색이나 단일 지식그래프 기반 접근은 문헌 근거를 연구 과정 전체로 일관되게 연결하기보다 결과 중심으로 흘러가며, 실험 검증과 반박(falsification)을 함께 자동화하더라도 ‘주장-근거 불일치’가 생기기 쉽다.

- **Core Contribution**: Xcientist는 연구 합성(research synthesis)과 실험 검증(experimental validation)을 외부에서 검사 가능한 계약(contract) 기반 프로세스로 외재화하는 연구 하네스다. Paper Graph(문헌 근거 그래프)로 아이디어를 증거에 고정하고, Experiment Agent가 실행 체인을 단계별 검증 조건과 산출물로 묶어 ‘주장에 대한 실행 가능 근거’를 유지하도록 설계했다.

- **Technical Challenges**: 핵심 기술 난제는 생성된 러너블(실행 가능) 산출물이 실제로는 원래 주장된 메커니즘을 지지하지 못하는 claim drift를 어떻게 막느냐다. Xcientist는 증거 그래프가 아이디어 탐색 공간을 제한하고, 계약 기반 validator가 각 단계의 입력/허용 연산/필수 산출물/합격 기준을 강제하며, 결함 발생 시 repair trace를 구조화해 기록함으로써 주장-검증의 연결을 끊지 않게 한다.

- **Empirical Impact**: 논문은 training-free memory, 그래프 기반 교통 예측, multiscale PINNs 등 3개 도메인에서 연구 궤적이 증거 기반으로 형성·진단·수정되는지를 과정 단위로 점검하며 성능 개선을 함께 보고한다. 특히 MCTS 기반으로 후보를 탐색·융합하되 ablation으로 결함을 찾아 국소 수리를 반복하고, 최종적으로는 ‘최종 점수’뿐 아니라 synthesis/validation이 추적 가능하고 과학적으로 책임질 수 있는지를 기준으로 평가해야 한다는 메시지를 강화한다.



### WorldLines: Benchmarking and Modeling Long-Horizon Stateful Embodied Agents (https://arxiv.org/abs/2606.18847)
Comments:
          27 pages, 18 figures

- **Prior Approaches**: 기존 long-term memory 벤치마크는 주로 언어 중심 검색·질문응답을 평가하며, 물리 상태 전이·실행 피드백과의 연결이 약한 편이다. 반면 embodied 벤치마크는 탐색·재배치·조작 등 성능을 높였지만, 대부분 짧은 에피소드에 묶여 상태가 상호작용 전반에 걸쳐 지속되는지를 충분히 시험하지 못했다.

- **Core Contribution**: 이 논문은 WorldLines라는 장기(다일) 프로젝트 기반 embodied 벤치마크를 제안해, 대화·행동·디바이스 변화·실행 피드백까지 포함한 시간적으로 연장된 가정(trace)으로 Memory QA와 Embodied Task Planning을 평가한다. 또한 메모리를 단순 텍스트로 뭉치지 않고, Observer-grounded 관점에서 관측 증거·상태 트레일·에이전트 belief을 분리하는 ObsMem을 제시한다.

- **Technical Challenges**: 핵심 난제는 부분관측 환경에서 관측되지 않은 변경과 덮어쓰기(overwritten)된 세계 상태를 구분하고, 이 신뢰성/불확실성을 이후 QA·계획으로 일관되게 반영하는 것이다. ObsMem은 observer gate로 관측/보고(provenance)를 분리해 기록하고, Event Track은 증거를 append-only로 유지, State Track은 스냅샷과 히스토리를 함께 두며, Belief Track은 시점별 epistemic 상태를 업데이트하도록 설계했다.

- **Empirical Impact**: 실험에서 ObsMem은 Memory QA에서 Judge 점수와 Perfect Rate, 특히 Event R@5(정확한 상태변화 증거 회복)에서 경쟁 방법 대비 큰 향상을 보였다. 또한 planning에서도 state consistency·사전조건 유효성·메모리 활용 측면에서 Plan Judge가 가장 높았고, 장기 메모리가 단순 검색이 아니라 실행 가능한 상태 제약으로 번역되어야 함을 실증적으로 강조한다.



### ProfiLLM: Utility-Aligned Agentic User Profiling for Industrial Ride-Hailing Dispatch (https://arxiv.org/abs/2606.18803)
- **Prior Approaches**: 기존 주문 배차 성능은 거리·ETA·요금 같은 structured numerical features 중심으로 개선돼 왔습니다. 반면 수락/취소 같은 핵심 신호는 장기적인 지역 회피나 시간대 민감도처럼 맥락 의존적이라 수작업 피처로 포착하기 어렵고, LLM을 단순히 붙이면 오히려 AUC가 크게 떨어질 수 있습니다. 또한 오프라인 파일럿처럼 high-frequency 사용자에 한정하면 이득이 나오지만, 실서비스의 long-tail 사용자 분포에서는 그대로 확장하기 어렵습니다.

- **Core Contribution**: ProfiLLM은 ride-hailing 배차 파이프라인에서 LLM 기반 user profiling을 ‘예측 유틸리티를 높이도록’ 운영하는 데이터 파이프라인을 제안합니다. 핵심은 (1) Tool-Augmented Global Knowledge Mining으로 플랫폼 규모의 글로벌 지식·클러스터 규칙·공급-수요 priors를 만들고, (2) Utility-Aligned Profile Exploration으로 후보 프로필을 생성·평가·정제한 뒤 DPO로 프로필 생성기를 fine-tuning해 downstream 예측에 유리한 프로필만 남기는 구조입니다. 무엇보다 LLM 추론은 전부 offline에서 끝내고, online에는 클러스터 임베딩만 캐시 조회해 latency 제약을 충족합니다.

- **Technical Challenges**: 첫째, 로그가 LLM context window를 압도하고(수천만 건 규모), 이를 그대로 넣어 분석하기 불가능합니다. 둘째, 대부분 사용자는 long-tail이라 per-user 프로파일링이 성립하지 않으므로, 글로벌 지식 기반의 적응형 user clustering으로 충분한 데이터가 모인 단위에서 프로필을 학습해야 합니다. 셋째, 프로필이 그럴듯하게 ‘표면 유창성’만 높다고 해서 예측 AUC가 오르는 게 아니라서, 후보 프로필을 LOGIC 규칙 기반 utility proxy로 빠르게 순위화하고 오류 피드백으로 반복 정제한 뒤 preference pair를 구성해 DPO로 정렬합니다.

- **Empirical Impact**: DiDi 프로덕션 배차에 적용한 결과, outcome prediction에서 최대 +6.14% 상대 AUC 개선이 관찰됐고, 시뮬레이션 GMV도 최대 +4.35% 향상됐습니다. 14일 City A 온라인 A/B 테스트에서는 GMV +0.47%, Completion Rate +0.33%, Cancel-Before-Accept rate -0.82%로 일관된 개선이 보고됩니다. 이는 LLM을 ‘기술 데모’가 아닌 실서비스 배차의 데이터 파이프라인으로 정착시키는 접근이 임팩트가 있음을 보여준 사례로 평가됩니다.



### R2D-RL: A RoboCup 2D Soccer Environment for Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.18786)
Comments:
          Code is available at: this https URL

- **Prior Approaches**: 기존 축구(RL/MARL) 연구는 Google Research Football, NFootball처럼 학습 친화적인 환경으로 발전했지만, 로봇 축구에서는 RCSS2D처럼 대회 중심 시뮬레이터가 학습 프레임워크와 바로 맞물리기 어렵다는 문제가 컸다. RCSS2D 생태계의 HELIOS는 전달/드리블/슈팅 등의 동작을 잘 제공하지만, 서버-클라이언트 실행 흐름이 modern Python MARL 학습의 step 동기화 의미론과 잘 연결되지 않는다.

- **Core Contribution**: 이 논문은 RCSS2D와 HELIOS 기반 플레이어 클라이언트를 Python MARL 인터페이스로 감싸는 R2D-RL을 제안한다. shared-memory 통신과 cycle-level synchronization으로 대회 운영 중심의 워크플로를 step-synchronous 강화학습 상호작용으로 변환하고, CTDE에 필요한 관측/중앙 상태/행동 마스크/리셋을 환경 API로 노출한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 중앙학습을 위한 centralized state와 (2) player가 실제로 내리는 명령이 (3) 환경의 한 step 전이/보상 신호와 정확히 대응하도록 동기화를 만드는 것이다. R2D-RL은 관측 게시–행동 할당–행동 소비를 분리한 sequence-counter 프로토콜을 두어 play_on 단계에서만 학습 step을 집계하고, 리셋/시나리오 초기화도 같은 동기화 설계에 포함시킨다.

- **Empirical Impact**: R2D-RL은 front-goal 시나리오와 11-vs-11 full-field 벤치마크를 제공하며, Base discrete 및 Hybrid parameterized 행동공간과 action masks, EPV 기반 reward shaping(MaxEPV 선택 가능)을 포함한 기준선(baseline) 결과를 함께 제시한다. 또한 병렬 실행과 상태 복원형 scenario 초기화를 지원해, RCSS2D의 로봇 축구 연구를 modern Python MARL 파이프라인으로 확장하는 데 의미가 있다.



### What Must Generalist Agents Remember? (https://arxiv.org/abs/2606.18746)
- **Prior Approaches**: 기존 causal RL 연구는 에이전트가 태스크/도메인 인덱스를 입력으로 받는 경우를 많이 다뤄, low regret이나 regret bound를 만족하려면 사실상 world model(전이/인과 모델)을 내부에 학습해야 한다고 주장해 왔다. 다만 이 논문이 보는 설정은 인덱스가 없고, 에이전트는 자신의 관측-행동 궤적만 보고 도메인을 구분해야 하므로, “입력에 읽을 수 있는 모델”이 아니라 “기억 속에 암묵적으로 저장된 정보”가 분석 대상이 된다. 또한 hidden-context/POMDP 계열 및 context inference 방법들은 문맥 추론을 위한 아키텍처·학습법을 제안하지만, 무엇이 필수로 저장되는지에 대한 필요조건을 직접 주지는 않는다는 점에서 보완적이다.

- **Core Contribution**: 이 논문은 generalist agent가 여러 도메인과 goal에 대해 near-optimal로 행동하려면, 특정 관측 병목(bottleneck)에서 도메인 식별에 필요한 정보를 메모리에 저장해야 함을 “메모리 수준 필요조건”으로 제시한다. 특히 같은 상태(병목)에 도달했을 때 도메인마다 최적 행동이 달라지면, 균일하게 near-optimal인 어떤 정책도 그 병목에서 서로 다른 메모리 분포를 유도해야 함을 분리 정리(separation theorem)로 증명한다. 이어서 메모리에 related goals의 value를 예측할 만큼 충분한 정보가 들어있다면, 그 메모리로 로컬 interventional dynamics(행동 개입에 따른 전이 커널)를 근사적으로 복원할 수 있음을 보인다.

- **Technical Challenges**: 핵심 난제는 “정확한 world model을 학습하는가” 같은 알고리즘적 질문 대신, 인덱스 없는 설정에서 임의의 history→memory 인코딩이 어떤 정보를 반드시 포함해야 하는지를 필요조건으로 끌어내는 것이다. 논문은 두 도메인이 병목 상태에서 disjoint한 value-gap-separated 행동 집합을 요구하는 조건을 설정해, near-optimality가 그만큼 메모리 분포를 총변동거리 기준으로 분리하도록 강제함을 보인다. 또한 dynamics 복원은 직접 디코딩이 아니라 auxiliary probe goal(원-스텝 예측 테스트)을 통해 test class의 span 생성성과 메모리 value-sufficiency 조건을 만족할 때, 같은 메모리 상태가 유사한 로컬 커널로 이어지게 만드는 식별-by-tests 논증으로 해결한다.

- **Empirical Impact**: 이론 결과는 gridworld 실험에서 병목에서의 도메인 분리와, auxiliary probe 기반으로 로컬 전이 커널을 복원할 수 있는 예측을 확인하는 형태로 제시된다. 결과적으로 메모리는 단순한 관측 요약이 아니라 (1) domain disambiguation, (2) interventional dynamics 재구성, (3) planning에 필요한 기반(substrate)임을 정식으로 제약한다. generalist RL 설계 관점에서, “현재 상태 관측만으로는 불충분하며 어떤 종류의 메모리 유지가 최소한 필요한가”와 “어떤 추가 정보(관련 goal value)만 있으면 dynamics 디코딩이 가능한가”를 명확히 해준다는 점에서 의미가 크다.



### ForecastBench-Sim: A Simulated-World Forecasting Benchmark (https://arxiv.org/abs/2606.18686)
Comments:
          15 pages, 5 main figures, 6 appendix figures. Spotlight presentation at Forecasting as a New Frontier of Intelligence / Workshop on AI Forecasting, ICML 2026

- **Prior Approaches**: ForecastBench 등 기존 예측 벤치마크는 실제 세계 질문을 그대로 가져오지만, 결과가 늦게 확정되고 꼬리사건이 드물며 반사실(counterfactual) 질문은 보통 점수화가 어렵다는 구조적 제약이 있습니다. 시뮬레이션을 활용한 대안도 있었으나, 주로 과거 데이터를 재활용하거나(예: FutureSearch) 단순 그래프 기반의 인과 추론에 머물러 동적 다중 에이전트 세계의 경로 의존성을 충분히 활용하기 어렵다는 한계가 지적됩니다. 

- **Core Contribution**: 이 논문은 Freeciv 턴제 전략 게임 롤아웃을 기반으로 한 시뮬레이션 예측 벤치마크 ForecastBench-Sim을 제안합니다. 고정된 world report(게임 상태 스냅샷)를 주고 숨겨진 미래를 예측하게 만든 뒤, 시뮬레이터를 계속 진행해 정답을 즉시 해소하고 이진/연속(분포) 질문을 같은 인터페이스로 채점합니다. 또한 개입(intervention) 월드를 포크해서 조건부·인과 질문을 실제로 점수화할 수 있게 설계했습니다. 

- **Technical Challenges**: 핵심 난제는 “예측 과제의 공정성”을 유지하면서도 시뮬레이션의 장점(반사실/즉시 해소)을 평가에 자연스럽게 결합하는 것입니다. 저자들은 모델이 직접 시뮬레이터 상태가 아닌 report를 보게 하고, 연속 예측은 분위수 p10~p90로 elicitation한 뒤 CRPS로 채점하며, 스코어는 템플릿별 기준 범위로 정규화해 비교 가능하게 했습니다. 조건부·인과 스코어링을 위해서는 savegame을 수정해 분기된 rollout을 생성하고, 동일 질문 템플릿을 baseline과 intervention framing으로 페어링해 점수 차이를 측정합니다. 

- **Empirical Impact**: 검증 결과, ForecastBench-Sim의 이진 Brier는 기존 실세계 벤치마크(ForecastBench) 성과 및 ECI(Epoch Capabilities Index)와 유의미한 상관을 보이며, horizon이 길어질수록(예: H1→H7) 불확실성이 증가하는 전형적인 예측 난이도 패턴도 관찰됩니다. H0(리포트 이해 확인) 체크에서 대부분의 모델이 낮은 오차를 보여, 성능 저하가 단순히 report 파싱 실패 때문이 아님을 시사합니다. 소규모 익명 인간 파일럿도 동일 과제 제시가 가능함을 보여주며, 후속 연구에서 calibration, 인과적 업데이트, 꼬리위험(tail-risk) 같은 주제를 더 통제적으로 다룰 수 있는 평가 기반을 제공한다는 점에서 의미가 있습니다.



### Optimizing Lithium Production Decisions under Geological, Demand, and Pricing Uncertainties: A POMDP Framework for Multi-Objective Decision Making (https://arxiv.org/abs/2606.18598)
Comments:
          24 pages, 14 tables, 4 figures

- **Prior Approaches**: 기존 연구는 리튬 생산 의사결정(어디를 언제 개발할지)을 다루되, 가격 불확실성과 수요 불확실성, 그리고 채굴 방식(직접 리튬 추출부터 하드록 광산) 차이를 충분히 모델에 포함하지 못했습니다. 그 결과 최적화는 가능하더라도, 실제 시장 변동과 기술 선택의 복잡성을 반영한 ‘견고한’ 전략으로 이어지기 어려웠습니다. 또한 채굴 기술을 함께 고려하는 모형은 상대적으로 제한적이었습니다.

- **Core Contribution**: 이 논문은 문제를 부분관측 마르코프 결정과정(POMDP, Partially Observable Markov Decision Process)으로 정식화하고, 채굴 위치·시점뿐 아니라 생산 기술 선택까지 포함해 의사결정을 한 프레임워크에서 다룹니다. 가격과 수요의 불확실성을 서로 다른 가격 모델(정적, 선형, 지수, 확률적) 및 다양한 광산 시나리오에 걸쳐 명시적으로 다루며, 의사결정이 탐사-생산-기술 전환을 어떤 순서로 진행해야 하는지까지 최적화하도록 설계했습니다. 이를 통해 “언제, 어디서, 어떤 방식으로 캐나”를 함께 최적화하는 것이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 불확실한 가격·수요 하에서 상태를 직접 관측하지 못하는 환경을 어떻게 믿음상태(belief state)로 표현하고, 이후 기술 선택까지 포함해 최적 정책을 계산하느냐입니다. 논문은 belief state planning 방식으로 불확실성을 관리하며, 동시에 탐사, 생산, 기술 선택의 타이밍을 계획에 통합합니다. 특히 가격 체제가 바뀌는 상황에서도 그 변화에 동적으로 적응하도록 설계해, 고정 규칙형 휴리스틱보다 더 안정적인 의사결정을 목표로 합니다.

- **Empirical Impact**: 실험에서는 POMDP 솔버가 사람의 직관을 반영한 휴리스틱보다 우수하게 작동하며, 정적·선형·지수·확률적 가격 레짐 전반에서 동적 적응 성능을 보였습니다. 또한 수요 충족(demand fulfillment)을 더 잘 달성하면서, 프로젝트 전 생애주기 동안 경제적 성과와 환경적 결과를 더 균형 있게 만드는 경향이 관찰됩니다. 결과적으로 리튬 생산 의사결정 분야에서 ‘불확실성+기술 선택’을 함께 고려한 최적화의 실용성을 강화하는 의미가 있습니다.



### DeFAb: A Verifiable Benchmark for Defeasible Abduction in Foundation Models (https://arxiv.org/abs/2606.18557)
Comments:
          33 pages, 14 figures, 23 tables. Dataset: this https URL ; code and evaluation harness: this https URL

- **Prior Approaches**: 기존 비단조 추론 연구는 default logic, circumscription, KLM 같은 형식 체계를 제시했지만, 실제로는 LLM이 “기본값을 언제, 어떻게 되돌리는지”를 측정하기 어려웠습니다. 또한 대부분의 추론 벤치마크가 단조적(전방) 추론·검증에 치우쳐, 예외 규칙을 생성하는 belief revision(가정의 수정) 과정을 정량화하지 못했습니다. 지식베이스 기반 벤치마크는 정답이 사전학습에 오염될 가능성도 커서, 모델이 기억을 꺼내는 것과 진짜 추론을 분리하기가 쉽지 않았습니다.

- **Core Contribution**: DeFAb(Defeasible Abduction Benchmark)은 방대한 공개 지식베이스를 defeasible abduction(결함/예외를 보존하며 기본값을 덮어쓰는 가설 생성) 문제로 변환해, 이론 수정의 “정당성”을 verifier로 강제합니다. 핵심은 conservativity(기준선 유지)와 derivation 타당성, minimality 같은 조건을 만족하는 가설만 정답으로 인정해, 유창한 설명 대신 “이론을 어떻게 보수적으로 수정하는가”를 점수화한다는 점입니다. rendering-robust 평가까지 도입해, 표면 문장/표현 민감도와 인식론적 추론 능력을 분리하려고 했습니다.

- **Technical Challenges**: 어떤 지식베이스에 기반하더라도 정답이 학습 데이터에 그대로 존재하면 ‘오염’ 문제가 생기는데, DeFAb은 합성 엔티티를 생성해 Common Crawl에서 발견되지 않음을 infini-gram으로 검증하는 합성 오염 방지 설계를 넣었습니다. 또 데프리저블 규칙 생성이 가능한지 확인하려면 검증기가 필수인데, 논문은 다항시간에 판정 가능한 논리적 검증(derivation·conservativity·minimality)을 제공해 학습용 정확한 보상 신호로도 재사용 가능하게 했습니다. 마지막으로 M1~M5 다중 모달리티 렌더링(서사~형식~시각)과 decoder-파싱 경로를 갖춰, 같은 논리 내용을 서로 다른 표면으로 바꿔도 성능이 유지되는지 측정합니다.

- **Empirical Impact**: 결과적으로 상징 규칙 기반 ASP/defeasible 솔버는 모든 인스턴스를 50마이크로초 내 100% 정확도로 처리한 반면, frontier language model은 Level 2 최적 성능이 65%지만 rendering-robust에서는 23.5%까지 떨어졌습니다. 모델 간 성능 격차보다 chain-of-thought 변동성(약 36pp)과 오염 대조 실험(매칭 contamination gap으로 Level 3에서 +19.4pp)을 통해, 취약점이 ‘표현/디코딩 변동’과 ‘결함적 belief revision 내재화 부족’에 있음을 보여줍니다. DeFAb-Hard와 CONJURE까지 공개하며, verifier 기반 exact reward로 DPO·RLVR/GRPO 같은 선호 최적화/강화학습까지 확장할 수 있음을 제시해, 논리적 창의·이론수정 평가의 기준점을 바꿀 가능성이 큽니다.



### CEO-Bench: Can Agents Play the Long Game? (https://arxiv.org/abs/2606.18543)
- **Prior Approaches**: 기존 에이전트 벤치마크(SWE-bench, WebArena, τ-bench 등)는 목표가 명확하고 에피소드가 짧아 피드백이 빠르게 관측되는 “단기 실행” 성격이 강합니다. GDPval이나 agentic-memory 벤치마크도 작업의 지속성은 늘리지만, 여전히 일회성 산출물 중심이거나 저장·검색에 초점이 맞춰져 있습니다. Vending-Bench/Accounting-Bench처럼 장기 시뮬레이션도 있지만 의사결정 범위가 좁고 환경이 비교적 안정적이라, 잡음·지연 피드백·상호 의존·전략 일관성까지 통합해 검증하긴 어렵습니다.

- **Core Contribution**: CEO-Bench는 에이전트가 500일 동안 스타트업을 운영하며 장기 전략 제어, 잡음 환경에서 정보 획득, 변화하는 세계 적응, 다부문 오케스트레이션을 한 번에 평가하도록 설계됐습니다. 에이전트는 34개 도구와 19-table 비즈니스 데이터베이스를 가진 동일한 파이썬 기반 인터페이스에서 운영 결정을 내리고, 종료 조건(현금이 0 아래로 하락)을 통해 결과를 직접 받습니다. 단순 도구 호출 능력을 넘어, 데이터 분석과 코딩을 결합한 “지속 가능한 경영” 지능을 측정하는 데 초점을 둡니다. 

- **Technical Challenges**: 핵심 어려움은 (1) 간접·지연·잡음 신호로부터 숨은 고객 선호와 만족도를 추정해야 하고, (2) 매출·R&D·평판 등 영향이 서로 다른 시간 스케일로 누적되며, (3) 고객 획득·품질·지원·엔터프라이즈 협상 등 의사결정이 서로 얽혀 인과를 단일하게 분리하기 어렵다는 점입니다. 논문은 이를 위해 고객을 집단/개별 단위로 세분화하고, 구독·이탈·소셜 미디어 반응·경쟁사 압력·거시 사이클 등 여러 동역학을 기계적으로(LLM 판정 없이) 생성하도록 시뮬레이터 세계 규칙을 구성했습니다. 또한 에이전트가 직접 SQL 조회와 맞춤 코드 실행을 할 수 있게 API 기반 도구·데이터 스키마·공개 피드/협상 로그를 제공해, “분석→전략→실행→재계획” 루프를 강제합니다.

- **Empirical Impact**: 실험 결과 대부분의 최신 모델은 도구 호출과 분석 쿼리 자체는 수행하지만, 500일 동안 일관된 전략을 유지하지 못하고 파산하는 경우가 많습니다. 최상위 성적은 Claude Opus 4.8과 GPT-5.5로, 둘만이 시작 현금 $1M을 넘기는 현금 상위를 기록했지만 이익이 “일관되게” 나진 못했다고 보고됩니다. 더 나아가 모델별 행동 궤적을 보면 GPT-5.5/Claude Opus 4.8은 더 넓게 탐색하고 상황에 따라 전략을 자주 바꾸는 반면, Claude Opus 4.7은 수동적 현금 보존 성향에 머물며 초기 고객 확보가 끊기는 등 미세한 차이가 드러납니다. CEO-Bench는 장기·적응형 지능의 측정 공백을 보여주며, 현재 모델들의 통합 전략 역량에 명확한 여지가 남아 있음을 실증적으로 제시합니다.



### Searching for Synergy in Shared Workspace Human-AI Collaboration (https://arxiv.org/abs/2606.18413)
Comments:
          Accepted at ICML 2026 Workshop on Human-AI Co-Creativity. 13 pages, 5 figures, 3 tables

- **Prior Approaches**: 기존 에이전트 평가는 ‘자율적으로 정답을 내는가’에 초점이 맞춰져, 사람의 문맥적 판단과 역할 분담이 있는 협업 실패 양상은 덜 다뤄졌다. 또한 shared-workspace에서 협업 구조를 체계적으로 바꾸기보다 역할 놀이/워크플로 오케스트레이션 등 제한된 상호작용(턴 기반 중심)이 많아, 책임·증거 라우팅이 성능을 좌우한다는 관점이 약했다.

- **Core Contribution**: 이 논문은 Collaborative Gym의 DiscoveryBench archaeology 과제를 사용해, 시뮬레이션된 ‘사람 협력자’를 추가할 때 언제 성능이 오르고 언제 과정 손실(process loss)로 오히려 떨어지는지 정량 분석한다. 더 나아가 shared group memory(공유된 누가-무엇을-어떤 기준으로 점검하는지 맵)와 simulated human-in-the-loop(HITL) gate(선택된 행동만 지정 참가자의 승인 필요)를 결합한 scaffolding이 팀의 책임 신호와 전문성 라우팅을 강화해 성능을 끌어올린다고 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘역량이 추가되면 항상 좋아질 것’이라는 기대를 깨는 과정 손실의 메커니즘을, 최종 제출 결과가 아니라 interaction trace에서 진단·분해할 수 있게 만드는 것이다. 논문은 trace 기반의 Workflow Coverage, Hypothesis Support, Profile Alignment 같은 그래프 의존 지표로 중간 증거가 최종 가설로 라우팅되는지, 책임이 할당·검증되는지 확인하고, 기본(shared memory 없음, gate 없음)과 구조화(scaffolded) 조건을 비교해 원인을 분리한다.

- **Empirical Impact**: 1,482개 세션에서 ‘관련 프로필 협력자 추가’가 기본 구조에서는 성능을 낮추기도 했고(특히 두 명 프로필 협력자가 함께 있을 때), 활동량/메시지 증가에도 불구하고 가설 근거가 약해지는 패턴이 관찰됐다. 반면 scaffolded는 모든 팀 구성에서 평균 성능을 끌어올리며, 특히 three-person DR에서 향상이 가장 컸고(기본 대비 +0.13 수준), trace로는 검증 체크가 finalization 이전에 선행되며 증거 손핸드가 강화됨을 보여준다. 결론적으로 인간-에이전트 팀에서 ‘조율·통합 구조’가 단순한 모델/역량만큼 중요하다는 메시지를 실험적으로 뒷받침한다.



### CaVe-VLM-CoT: An Interpretable Vision-Language Model Framework (https://arxiv.org/abs/2606.18385)
- **Prior Approaches**: 기존 비전-언어 모델(VLM)은 CoT(Chain-of-Thought)로 중간 추론을 보여줘도, 각 단계가 특정 근거에 “묶이는지”를 강제하지 못해 시각·근거 불일치가 남는다는 한계가 있었다. RAG는 검색 증거를 붙이지만, 검증이 실패해도 재검색으로 연결되지 않거나, 단계별 인용 충실도를 함께 측정·개선하지 못했다. 또 평가 지표는 대체로 정답률 중심이어서, 어떤 파이프라인 단계(추출·검색·추론·인용·검증)에서 문제가 발생했는지 진단이 어렵다는 문제가 지적된다.

- **Core Contribution**: 논문은 CaVe-VLM-CoT(Cite-and-Verify Vision-Language Model with Chain-of-Thought)라는 모듈형 agentic RAG 프레임워크로, 추론의 모든 사실 주장에 단계별 증거 인용을 요구하고 실패를 재검색으로 되돌린다. Extractor–Retriever–Solver–Citation Injector–Verifier의 5단계 closed-loop 파이프라인을 최대 3회 반복하며, Verifier가 “근거 없는 주장”을 감지하면 구조화된 피드백을 Extractor에 전달해 타깃 재쿼리와 재검색을 유도한다. 또한 retrieval 품질·단계 인용 충실도·교차모달 grounding을 동시에 평가하는 23개 컴포넌트 지표와 이를 묶은 CaVeScore를 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 단계별로 인용이 실제 검색 증거를 정확히 반영하도록 만들고, (2) 검증이 실패 신호를 ‘정답을 뒤집는’ 방향이 아니라 ‘근거를 다시 찾아오는’ 방향으로 사용할 수 있게 하는 것이다. 저자들은 Solver의 구조화된 reasoning trace에 Citation Injector가 증거 인덱스 정합성을 유지해 인용을 주입하고, Verifier가 claim-level 사실/범위 오류와 시각 모순을 검출해 실패 유형·근거 내용·재검색 방향을 Extractor에 반환하도록 설계했다. 아울러 CaVeScore 계산을 위해 cross-encoder/NLI 기반의 entailment 신호를 단계별 지표에 연결해, retrieval·attribution·grounding을 함께 측정 가능하게 만들었다.

- **Empirical Impact**: 실험에서 CaVe-VLM-CoT는 ScienceQA에서 87.1% 정확도와 56.6% CaVeScore를, MMMU(30 subjects)에서 55.2% 정확도와 35.7% CaVeScore를 보이며 도메인 일반화도 입증했다. 특히 정확도와 grounding/인용 지표의 “분리 현상”을 stage-level로 드러내, MMMU 성능 저하의 주 원인이 추론/검증이 아니라 검색 corpus coverage 부족임을 진단해낸 점이 의미 있다. 또한 Verifier 크기(Qwen2.5-VL-8B→32B)에 따라 feedback loop의 효과가 크게 달라져, 정확한 검증 신호(캘리브레이션)가 루프 성능 병목임을 실증적으로 보여준다.



### NAVI-Orbital: First In-Orbit Demonstration of a Zero-Shot Vision-Language Model for Autonomous Earth Observation (https://arxiv.org/abs/2606.18271)
Comments:
          17 pages, 47 figures

- **Prior Approaches**: 기존 지구관측 온보드 AI는 주로 “특정 패턴”을 찾는 specialist detector 중심이어서, 새로운 현상(관측 대상)에 적응하려면 재학습·검증·복잡한 업데이트가 필요했다. 최근에는 온보드 DNN을 통한 데이터 필터링(예: 구름/홍수)과 일부 end-to-end 파이프라인, 동적 타게팅 같은 자율 기능도 시도됐지만, 대체로 고정된 클래스/라벨 체계에 의존한다. 또한 ISS에서는 LLM을 절차 보조·문서 검색·제어에 쓰는 사례가 늘었지만, 위성 본체에서 비전-언어로 이미지를 직접 이해하고 멀티모달 추론을 수행하는 “완전 온보드” 흐름은 상대적으로 미비했다.

- **Core Contribution**: 이 논문은 LEO 위성에서 Gemma 3 같은 비전-언어 모델을 이용해 장면을 분류하고 텍스트 설명 및 객체 간 관계를 생성한 뒤, 연산 결과를 자연어 대화로 연속 질의응답하는 NAVI-Orbital 시스템을 제안한다. 핵심은 기존처럼 다운링크/인간 검토에 의존해 “획득→전송→분석”을 늘리는 대신, 온보드에서 의미 기반(semantic)으로 압축해 actionable 정보만 내려보내는 운영 패러다임을 실증한 점이다. 또한 전통적 명령 시퀀스를 대체해 plain-English 프롬프트로 태스크를 재지정하고, 재학습 없이(vocabulary/prompt 변경) 새로운 관측 작업에 적응하도록 설계했다.

- **Technical Challenges**: 위성급 엣지 컴퓨팅(SWAP/VRAM/전력 제약)에서 멀티모달 foundation model을 안정적으로 돌리면서도, 후속 다운스트림을 위해 출력 의미를 결정적·파싱 가능하게 유지하는 것이 큰 과제다. 논문은 4-bit 양자화(Q4_0/비트앤바이트 등)와 하드웨어 가속 GPU 오프로딩, 컨텍스트/출력 길이 캡 같은 지연·메모리 통제를 적용하고, 출력의 결정성을 위해 온도 0.2 및 라벨 집합 제한(프롬프트 내 constrained label set)과 regex 기반 검증/재시도 루프를 붙였다. 더불어 LangGraph 기반 그래프 상태기계로 Conductor-Detector-Dialogue 에이전트를 구조화해 재부팅/상태 추적이 가능한 온보드 오케스트레이션을 구현했다.

- **Empirical Impact**: 실험은 지상 벤치마킹(AID 큐레이션 7,960장)에서 88.16% 정확도 성능을 포함해, 다중 데이터셋과 Flatsat 검증, 그리고 처음으로 새로운(미공유) 지구 이미지의 라이브 온보드 처리까지 단계적으로 입증한다. 특히 미세한 하드웨어/영상 보정이 없었던 YAM-9 실시간 이미지도 파인튜닝 없이 처리했으며, 다운스트림 운영 관점에서 대화형 후속 질의가 가능함을 보여준다. 결과적으로 위성 탑재 엣지에서 foundation model 기반 의미 압축과 자율 멀티모달 추론이 실행 가능하다는 실증을 제공해, EO 위성의 데이터-대역폭 병목을 “획득 전송량 축소” 방향으로 되돌릴 수 있는 설계 근거를 마련했다.



### UBP2: Uncertainty-Balanced Preference Planning for Efficient Preference-based Reinforcement Learning (https://arxiv.org/abs/2606.19328)
- **Prior Approaches**: Preference-based RL은 페어(trajectory segment 쌍) 비교로 reward model을 학습해 수작업 reward 설계를 줄이지만, 대개 passive 데이터 수집에 의존해 초기 sample efficiency가 떨어진다는 한계가 있다. 또한 불확실성을 탐색 보너스로 쓰더라도 단일 모델 컴포넌트에만 집중하거나, 후보 쌍을 로컬 배치에서만 고르는 방식이어서 정보가 큰 비교를 놓칠 수 있다. 관련으로, pretraining된 reward/dynamics나 오프라인 데이터셋을 쓰는 접근도 있어 ‘완전 온라인 + 선호 피드백만’으로부터 reward를 처음부터 학습하는 설정은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 Uncertainty-Balanced Preference Planning (UBP2)로, reward·dynamics·value의 불확실성을 함께 추론하며 탐색을 능동적으로 설계하는 model-based preference RL을 제안한다. 특히 reward는 선호(비교)로만 학습하고, 이 learned reward와 world model을 이용해 MPC 기반으로 trajectory를 계획한 뒤, 선호 피드백 예산이 끝나면 학습된 정책으로 전환한다. UBP2는 exploit(예상 return 극대화)와 explore(모델의 epistemic uncertainty 극대화)를 하나의 trajectory-level 점수로 정리해 ad hoc 탐색 휴리스틱 없이 균형을 달성하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) reward가 관측되지 않고 비교만 주어질 때의 reward 학습 및 (2) 불확실성이 dynamics/value/reward 각각에서 다르게 의미를 갖는 상황에서, 계획이 실제로 정보 수집으로 이어지게 만드는 것이다. UBP2는 reward ensemble 불확실성은 Jensen–Rényi divergence(JRD)로 epistemic/aleatoric을 분리해 쓰고, dynamics/value는 ensemble disagreement로 epistemic 위주의 총 불확실성을 추정한다; 이후 reward·terminal value·불확실성을 함께 묶은 unified score로 후보 trajectory를 평가해 MPC가 정보를 많이 주는 구간을 선택하도록 유도한다. 또한 preference label 쿼리는 replay buffer 전역에서 전개되는 후보 쌍을 ‘낙관적(optimistic) scoring’ 기준으로 우선순위화해, 로컬 배치 탐색의 누락 위험을 줄인다.

- **Empirical Impact**: Meta-World 조작(Meta-World manipulation) 벤치마크에서 UBP2는 state-of-the-art model-free preference 기반 방법과 비낙관적(non-optimistic) model-based baseline보다 더 빠른 성과(더 이른 성공)와 더 높은 sample efficiency를 보였다고 보고한다. 이 실험 결과는 불확실성 기반 낙관적 planning이 preference 기반 학습 초기 단계에서 특히 유리하다는 주장을 뒷받침한다. 더불어 유한/무한 horizon 모두에 대해 상수 이상의 정보획득량에 명시적으로 의존하는 sublinear regret 보장을 제시해, 탐색-수익 균형이 이론적으로도 뒷받침된다는 점에서 분야에 의미가 있다.



### Reference-Driven Multi-Speaker Audio Scene Generation from In-the-Wild Priors (https://arxiv.org/abs/2606.19325)
Comments:
          Project page at this https URL

- **Prior Approaches**: 기존 multi-speaker TTS/대화 TTS는 per-turn speaker tag, multi-stream transcription, speaker-turn embedding 같은 구조화된 감독으로 화자-발화를 결합(speaker-binding)하곤 했습니다. 또한 대화 생성은 보통 speech-only 파이프라인에서 정제된 발화만 산출해, 실제 대화의 잡음·실내 반향·겹치는 발화·비언어적 소리(웃음/숨소리)를 충분히 담기 어렵다는 한계가 있었습니다. 더 나아가 zero-shot voice cloning은 주로 단일 화자에 집중해 멀티 화자를 모델 밖에서 조립하는 방식에 머물렀습니다.

- **Core Contribution**: ScenA는 flow matching 기반 text-to-audio foundation model에 멀티 레퍼런스(여러 화자 음성)를 텍스트 프롬프트와 함께 조건화해, 한 장면(전체 대화+주변 소리)을 end-to-end로 생성하는 접근을 제안합니다. 핵심은 레퍼런스 latent를 입력 토큰에 결합하고 identity-aware positional encoding으로 역할을 구분하되, per-turn 구조 없이 자유형 자연어 프롬프트가 “어떤 레퍼런스 화자가 어디서 발화하는지”를 지정한다는 점입니다. 이 설계는 스튜디오 수준의 clean vocal만이 아니라 in-the-wild 오디오 질감까지 함께 생성할 수 있도록 목표를 바꿉니다.

- **Technical Challenges**: 가장 큰 난관은 학습 중 “Reference Shortcut”이 발생한다는 점입니다. 표준 노이즈 스케줄에서는 noised target이 여전히 음향적으로 레퍼런스와의 유사성을 보존해, 모델이 텍스트 경로를 무시하고 self-attention만으로 레퍼런스를 매칭해버릴 수 있으며, 학습 손실은 낮아져도 잡음에서 시작하는 추론 시 화자 결합이 붕괴됩니다. ScenA는 timestep 분포를 high-noise에 편향한 Beta+Uniform 혼합으로 바꿔 텍스트가 유일한 결합 신호가 되는 구간에 학습을 더 실어 이 지름길을 차단합니다. 추가로 adversarial reference injection과 slot-shuffle augmentation으로, 레퍼런스 순서 편향이나 프롬프트 없이도 맞출 수 있는 우회 경로를 더 줄였습니다.

- **Empirical Impact**: CoVoMix2-Dialogue 벤치마크에서 ScenA는 speaker-binding 중심 지표(cpWER, cpSIM, ACC)에서 기존 multi-speaker 대화 TTS를 능가하거나 최상 성능을 보였습니다. 특히 스튜디오 clean 레퍼런스에서 in-the-wild noisy 레퍼런스로 난이도가 올라가도 cpSIM이 더 견고하게 유지되어, 실제 사용 조건에서의 의미가 커졌습니다. 또한 A/B preference 테스트에서 모든 비교 대상 대비 선호도가 유의미하게 높았고, 생성물은 겹치는 대화, 웃음·한숨·숨소리 같은 paralinguistic event, 방/환경 잡음까지 장면 단위로 함께 구현되는 것으로 보고됐습니다.



### Data Intelligence Agents: Interpreting, Modeling, and Querying Enterprise Data via Autonomous Coding Agents (https://arxiv.org/abs/2606.19319)
- **Prior Approaches**: 기존 text-to-SQL 계열은 한 단계(질의 생성, 디버깅 등)에 집중하거나 파이프라인을 여러 모듈로 쪼개 정밀 튜닝/재설정을 반복하는 경우가 많다. RL 기반·특화 에이전트는 벤치마크/다이얼렉트에 강하게 고정되기 쉬우며, 에이전틱 접근도 세션 간 기억을 일관되게 공유하지 못해 매 질의마다 시작점이 흔들린다. 또한 많은 시스템이 텍스트(쿼리/비평)를 출력해 엔터프라이즈 작업에서 필요한 ‘실행 가능한 아티팩트’와 ‘상위 단계(이해·스키마 구성)’를 함께 닫지 못한다.

- **Core Contribution**: DIA(Data Intelligence Agents)는 ACA(코드 실행 가능한 에이전트)를 핵심 추상으로 두고 Data Interpreter, Schema Creator, Query Generator의 세 에이전트를 단일 워크스페이스 위에서 동작시킨다. LLM이 텍스트를 내놓는 대신, 에이전트가 생성·실행·검증·수정을 통해 실행 가능한 산출물을 직접 만들고 도메인 전문가가 검토할 수 있게 한다는 점이 핵심이다. 특히 Query Generator는 SQL 생성뿐 아니라 디버깅·대화·프로젝트 완료까지 단일 일반 에이전트로 처리하며, 적응은 자연어 지시사항 범위로 제한한다.

- **Technical Challenges**: 이 접근을 가능하게 하려면 (1) 원천 데이터에서 의미 있는 스키마를 자동으로 구성하고, (2) 생성된 SQL이 실제 데이터 조건과 일치하는지 실행 기반으로 스스로 검증하며, (3) 세션 간 경험을 ‘텍스트 요약’이 아니라 실행된 아티팩트/규칙 형태로 안전하게 재사용해야 한다. DIA는 작업 전 과정에서 공유 워크스페이스(WW)에 파일/아티팩트를 누적하고, 실행 추적과 결과를 기반으로 메모리(MM)에서 관련 경험만 pull 방식으로 가져와 사전조건을 라이브 프로브로 확인한다. Query Generator는 쿼리 초안을 즉시 정답으로 간주하지 않고 결과가 기대한 shape(열/행 단위, 정렬, 필터 등)와 일치하는지 확인한 뒤, 어긋나면 같은 패스에서 수정·재실행해 정합성을 맞춘다.

- **Empirical Impact**: 논문은 Query Generator를 7개 SQL 벤치마크(4개 태스크 범주, 4개 다이얼렉트)에서 완전 자율 모드로 평가해, 모든 벤치마크에서 기존 최강 결과를 ‘동일 모델·추가 fine-tuning 없이’ 매치하거나 능가함을 보인다. 특히 대화형(BIRD-Interact), 디버깅(BIRD-Critic), 수정 작업(LiveSQLBench)에서 큰 폭의 개선이 나타나며, 단일 에이전트가 여러 작업 범주와 다이얼렉트를 아우르는 일반화를 실증한다. 엔터프라이즈 데이터 지능 워크플로우를 실행 가능한 아티팩트 중심으로 재설계한 시스템 관점이어서, 다음 세대 text-to-SQL을 넘어 데이터 이해·스키마 구성·쿼리까지 한 루프로 묶는 방향에 의미 있는 영향을 줄 전망이다.



### Explaining Attention with Program Synthesis (https://arxiv.org/abs/2606.19317)
- **Prior Approaches**: 기존 해석 연구는 뉴런/특징에 대한 프로브나 입력·출력 요약으로 의미를 붙이려 했지만, 완전한 수준의 형식적(검증 가능한) 설명까지는 어렵다는 한계가 있었다. 자연어 설명은 모호하거나 정식화가 힘들고, 프로그램처럼 바로 대체 가능한 매개체가 부족해 인과 검증이 제한적이었다. 또 attention head 분석은 주로 주의 가중치를 관찰하는 귀납적 접근에 머물러, “코드로 재현·대체”하는 단계로 잘 이어지지 않았다.

- **Core Contribution**: 이 논문은 attention head의 동작을 “실행 가능한 Python 프로그램”으로 근사(approximating)하는 파이프라인을 제안한다. 각 head에 대해 실제 attention 행렬을 모은 뒤, 이를 요약한 정보를 바탕으로 다른 LM이 프로그램 후보를 생성하고, 보류 입력에서 재현 성능으로 재순위(rerank)해 최적 프로그램을 고른다. 더 나아가 이렇게 만든 프로그램을 실제 모델의 attention head를 대체해도 성능이 유지되는지(인과적 검증) 확인한다.

- **Technical Challenges**: 핵심 난제는 (1) attention 패턴을 코드로 표현할 수 있을 만큼 구조화해 추출하고, (2) 생성된 프로그램이 문법/실행 가능성을 만족하며, (3) 학습되지 않은 데이터에서 attention 행렬을 얼마나 재현하는지 정량 평가하는 것이었다. 논문은 attention 가중치 중 상위 2.5%만 필터링해 토큰-쌍 요약을 프롬프트에 넣고, 합성 에이전트가 NumPy/spaCy/NLTK를 활용한 Python 함수를 만들도록 유도한다. 후보 프로그램은 Jensen-Shannon distance로 점수화하고, 대표 worst/best 사례에 대한 피드백으로 한 번 더 refinement 한 뒤 보류 데이터에서 IoU로 최종 선택한다.

- **Empirical Impact**: 실험 결과, TinyStories에서 head별 프로그램 수 1,000개 미만으로도 평균 Intersection-over-Union이 75%를 넘기며 GPT-2, TinyLlama-1.1B, Llama-3B의 다수 head를 높은 정확도로 재현했다. 또한 가장 잘 맞는 프로그램을 3개 모델에 걸쳐 attention head의 최대 25%까지 대체해도 perplexity 증가가 평균 16% 수준에 그쳤고, 다양한 question answering 벤치마크 성능도 크게 흔들리지 않았다. 즉, attention의 상당 부분을 상징적(코드) 프록시로 추출해 모델 기능을 유지하는 수준의 “상징적 투명성” 가능성을 실증했다.



### Correct Yourself, Keep My Trust: How Self-Correction and Social Connection Shape Credibility in Social Chatbots (https://arxiv.org/abs/2606.19286)
- **Prior Approaches**: 기존 연구는 챗봇의 오류 이후 신뢰 회복을 다뤘지만, 대개 사과·부인·설명 같은 커뮤니케이션 전략 중심이었고 반복 위반에는 한계가 있다는 결과가 많았습니다. 또한 정정은 사용자의 믿음(주장 타당성)을 바꿀 수 있지만, 정정 출처(웹페이지/전문가)가 오히려 ‘해당 소스의 신뢰도’를 떨어뜨리는 ‘corrections dilemma’가 보고돼 왔습니다. 사회적 연결을 가진 social chatbots는 설득력이 높고 오류의 영향도 더 커질 수 있으나, 사회적 관계가 정정 효과와 신뢰에 어떻게 관여하는지는 충분히 검증되지 않았습니다.

- **Core Contribution**: 이 논문은 social chatbot이 만든 오류에 대해 (1) 웹페이지로 철회, (2) 같은 챗봇의 self-correction, (3) expert chatbot이 정정—세 가지를 비교해, self-correction이 신뢰(신뢰도·전문성)를 훼손하지 않으면서도 믿음 변화를 만든다는 점을 보여줍니다. 특히 세 전략 모두 잘못된 정보에 대한 믿음 자체는 비슷하게 낮췄지만, credibility 보존은 self-correction에서만 관측됐습니다. 또한 사용자의 사회적 연결 강도(사회적 호감, self-disclosure 등)는 belief change의 크기를 예측하되, 그 연결 효과는 ‘정정을 같은 bonded agent가 수행할 때만’ 나타났습니다.

- **Technical Challenges**: 핵심 과제는 (a) 정정이 실제로 belief를 얼마나 줄이는지와 (b) 정정이 챗봇의 credibility를 얼마나 손상/복구하는지를 분리해 측정하는 것이었습니다. 이를 위해 120명의 between-subjects 실험에서 동일한 social chatbot Drew가 먼저 소셜 라포를 형성한 뒤 건강/영양 도메인에서 오류를 만들고, 이후 조건별로 정정 방식을 다르게 구성했습니다. 또 ‘사회적 연결’은 조건 간에 동등하게 형성되도록 사전 소셜 상호작용을 통제했으며, self-correction 조건에서만 사회적 연결 강도가 belief 변화량과 연결된다는 조절 효과를 통계적으로 확인했습니다.

- **Empirical Impact**: 실험 결과, self-correction 그룹은 웹페이지 정정과 expert 정정 대비 신뢰도와 전문성 평가가 유의미하게 높았고(둘 간 차이는 작음), 외부 소스로 정정이 넘어가면 초기 소셜 연결이 ‘배신/조작’처럼 해석되며 credibility가 약화될 수 있음을 시사했습니다. 반면 잘못된 정보에 대한 belief change의 크기는 세 전략이 유사해, self-correction의 이점은 ‘더 잘 고쳐서’가 아니라 ‘같이 고치되 credibility를 지키면서’ 발현된다는 점이 정리됩니다. 더 나아가 사회적 연결 강도가 belief 변화량을 키우는 기능적 메커니즘은 self-correction에서만 작동했으므로, 장기적으로 신뢰를 유지하려면 social chatbot이 외주 정정보다 ‘자기 오류를 자기 손으로 바로잡는 설계’가 중요하다는 실증 근거를 제공합니다.



### Trade-offs in Medical LLM Adaptation: An Empirical Study in French QA (https://arxiv.org/abs/2606.19266)
- **Prior Approaches**: 의료 도메인 적응은 보통 CPT(continual pretraining)로 잡지식/임상 코퍼스를 학습하고, SFT(supervised fine-tuning)로 instruction–response를 맞추는 방식으로 이뤄져 왔다. 다만 기존 연구는 base model 초기값을 고정하거나 영어 벤치마크 중심으로 평가해, CPT·SFT 효과를 분리해서 비교하기 어려웠다. 또한 MCQA 위주라 생성형 OEQA에서의 실제 생성 품질 차이는 해석이 제한적이었다.

- **Core Contribution**: 이 논문은 프랑스 의학 QA를 케이스로 CPT, SFT, CPT+SFT를 “초기화(General/Instruct/Medical)”까지 체계적으로 바꿔가며 분리 실험한다. 모델 패밀리·크기·디코딩(greedy/제약)까지 함께 비교하고, MCQA와 OEQA를 같이 보되 OEQA 해석은 주의 깊게 진행한다. 결과적으로 계산 자원 제약 하에서 어떤 적응 전략을 우선할지에 대한 실무형 가이드라인을 제시한다.

- **Technical Challenges**: 핵심 난점은 적응 전략의 차이를 base model 선택 효과와 섞지 않고, 통계적으로도 유의한 비교를 만드는 것이다. 이를 위해 여러 모델 패밀리에 맞춘 정렬된 초기화 세트를 구성하고, MCQA는 constrained decoding과 EM/Hamming으로 재현성 있게 평가하며, OEQA는 ROUGE/BERTScore와 LLM-as-a-Judge(의사 코호트 기반 신뢰도 검증)를 결합해 측정한다. 또한 다중 비교 보정과 percentile bootstrap으로 유의성 검정을 수행해 “겉보기 성능”과 “실제 유의미한 개선”을 구분했다.

- **Empirical Impact**: MCQA에서는 CPT+SFT가 가장 자주 1등을 차지하지만, SFT 대비 이득이 작고 자주 통계적으로 유의하지 않았다; 따라서 라벨 데이터가 있을 땐 SFT가 강력한 기본값으로 정리된다. OEQA에선 SFT가 생성 품질을 악화시키는 경향이 있고, CPT가 겹침 기반 지표를 꾸준히 개선하지만 CPT+SFT는 불안정할 때가 많았다. 또한 프랑스 의학으로 적응하면 영어 벤치마크 성능이 개선되는 교차언어 전이가 관찰됐고, 번역 벤치마크는 정확도뿐 아니라 confidence까지 왜곡(과신/불확실성 감소)할 수 있어 메트릭 해석 주의가 필요하다는 결론을 강화한다.



### A Multi-Domain Benchmark for Detecting AI-Generated Text-Rich Images from GPT-Image-2 (https://arxiv.org/abs/2606.19259)
- **Prior Approaches**: 기존 AI 생성 이미지 탐지는 주로 물체 중심 이미지나 특정 생성 아티팩트에 초점을 맞춰 왔고, 텍스트와 레이아웃이 핵심 의미가 되는 ‘텍스트-리치’ 영역은 상대적으로 덜 다뤄졌습니다. 또한 기존 벤치마크는 최근 멀티모달 생성기(GPT-Image-2급)로 만든 전면(fully) 생성 문서형 이미지를 충분히 커버하지 못해, 탐지기의 일반화 능력을 체계적으로 보기 어려웠습니다. 결과적으로 카테고리별(문서/표/영수증/ UI 등) 실패 패턴이나 플랫폼 후처리(JPEG 등) 민감도 분석이 부족했습니다.

- **Core Contribution**: 이 논문은 GPT Image 2로 생성된 텍스트-리치 이미지를 6개 대표 범주(상업 포스터, 인포그래픽, 학술 포스터, 영수증, 표, UI 스크린샷)로 나눠 총 8,602장(실/생성 포함)의 멀티도메인 벤치마크를 제안합니다. 데이터는 텍스트-레이아웃 정규성(자유형~정형)과 기능적 맥락을 기준으로 구성해, ‘의미·배치 중심’ 시나리오를 정면으로 다룹니다. 이어서 다양한 탐지 패러다임(비전 트랜스포머, patch 기반, vision-language, 아티팩트/픽셀 관계 기반)을 zero-shot으로 한데 비교하는 통합 평가를 제공합니다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 텍스트와 구조가 결합된 문서형 포맷에서 생성 흔적이 희석되거나, (2) 압축·리사이즈 같은 플랫폼 후처리로 저수준 단서가 파괴된다는 점입니다. 논문은 전처리/파인튜닝 없이 공식 추론 파이프라인 그대로 실행하는 zero-shot 평가와 함께, JPEG 압축(품질 95~50), 리사이즈+JPEG, PNG 재인코딩 같은 후처리 조건을 바꿔 견고성을 점검합니다. 특히 이들이 탐지기에 따라 ‘실패 모드’가 달라지며, 아티팩트 기반 방법이라도 lossy JPEG에 크게 취약할 수 있음을 확인합니다.

- **Empirical Impact**: 실험 결과, 기존 탐지기는 전체 성능이 아니라 카테고리 의존성이 매우 커서, 일부 유형에서는 잘 맞지만 표·영수증 같은 정형 문서에서 급격히 무너지는 양상이 관찰됩니다. 또한 최강의 전통적 탐지기조차 JPEG 압축에서 재현성이 크게 흔들려, 디지털 신뢰/콘텐츠 진위 검증을 실사용 조건에서 신뢰하기 어렵다는 경고를 줍니다. 한편 GPT-5.5 같은 범용 비전-언어 모델은 전반 정확도가 더 높지만, 정밀한 구조와 텍스트 정렬이 요구되는 포맷(예: 표)에서는 여전히 병목이 남아 텍스트·레이아웃 인지형 탐지 필요성을 실증적으로 뒷받침합니다.



### OneCanvas: 3D Scene Understanding via Panoramic Reprojection (https://arxiv.org/abs/2606.19253)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 Vision-Language Model(VLM) 기반 3D 이해는 (1) 점구름 토큰화/geometry encoder 등 복잡한 모듈을 추가하거나, (2) distance·direction 같은 공간 QA를 대규모로 수집해 학습량을 키우는 방식이 주류였다. 그런데 최근 감사(audit) 결과, 텍스트만으로도 기준 성능에 근접하거나 나오는 경우가 있어 모델이 제공된 기하 입력을 ‘진짜로’ 읽기보다 장면/질문 통계의 지름길을 탄다는 한계가 지적됐다. 또한 대부분의 방식은 프레임을 조각조각 보며 공간관계를 일관된 좌표계로 결합하지 못하는 문제가 남아 있다.

- **Core Contribution**: OneCanvas는 여러 시점의 패치 특징을 depth와 카메라 포즈로 3D로 ‘리프트(lift)’한 뒤, 하나의 equirectangular 파노라마 캔버스에 longitude·latitude 좌표로 재투영해 VLM이 이미지처럼 그대로 읽게 만든다. 이때 캔버스의 원점(origin)과 방향을 태스크에 맞게 자유롭게 선택할 수 있어, 로보틱스/embodied AI에서 중요한 ‘지정된 시점에서의 situated reasoning’을 같은 표현으로 바로 지원한다. 나아가 캔버스 위에 실제 이미지에서 뽑은 객체 패치 특징을 임의의 3D 위치에 절차적으로 배치해, 답 분포를 제어하면서 공간 사전학습 커리큘럼을 on-the-fly로 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (a) 겹치는 관측에서 픽셀 단위 rasterization으로 정보를 뭉개지 않으면서, (b) VLM의 기존 3D-RoPE 의미를 깨지 않고 metric(미터) 좌표 정보를 주입하는 것이다. OneCanvas는 rasterization 대신 각 리프트 패치를 연속적인 (ϕ,θ) 위치를 갖는 개별 토큰으로 유지해, 중첩/가림 상황도 attention으로 구분하게 했다. 또한 각 패치의 캔버스 좌표 오프셋(qx,qy,qz)을 다축·방사(radial) 성분의 sin/cos 포지셔널 인코딩과 함께 특징 공간에 게이트된 방식으로 주입해, 각도 캔버스로 평탄화되며 사라지는 depth/거리 단서를 복원한다.

- **Empirical Impact**: 실험에서 OneCanvas는 SQA3D와 VSI-Bench에서 최신 성능을 달성하고, SPBench에서는 zero-shot에서도 우수한 일반화를 보이며 계산량은 경쟁 최고 대비 한 자릿수(대략 10배) 이하 수준으로 줄였다. 특히 SQA3D에서 ‘Which’처럼 시점 의존성이 큰 질문 유형에서 캔버스 원점 중심화(ablation)의 영향이 크게 나타나, 표현이 실제 situated geometry를 반영함을 시사한다. 즉, 아키텍처/부가 모듈 없이 입력 표현과 커리큘럼만으로 3D 추론을 강화할 수 있다는 실증적 근거를 제공해 해당 분야의 설계 관점을 바꾸는 의미가 있다.



### A Taxonomy of Mental Health and Technology Needs for Alzheimer's and Dementia Caregivers (https://arxiv.org/abs/2606.19247)
- **Prior Approaches**: 기존 연구는 AD/ADRD(알츠하이머병 및 관련 치매) 가족 돌봄자의 복합 심리·사회적 경험을 ‘caregiver burden(돌봄 부담)’ 하나로 축약하는 경향이 강했다. 그 결과 어떤 구체적 욕구가 충족되거나, 반대로 어떤 필요가 비어 있는지(예: 관계 갈등 vs 정서적 지침 부족)가 가려지고 기술 중재의 효과도 모호해졌다.

- **Core Contribution**: 본 논문은 돌봄자의 정신건강 요구를 기술 기반 중재의 범주와 체계적으로 연결하는 ‘Caregiver Mental Health and Technology Taxonomy(돌봄자 정신건강 및 기술 택소노미)’를 제안한다. 특히 ‘burden’을 분해해 anticipatory grief, compassion fatigue, social isolation, relationship management 등 영역별로 언어를 정리하고, 적응형·반응형 시스템 설계 방향까지 제시한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 다양한 문헌과 질적 인터뷰에서 도출된 요구를 이질적인 기술 클래스(모니터링, 정보 제공, 정서·사회적 지지, task management, AI 기반 하이브리드)로 정확히 매핑하고 (2) 현재 앱·챗봇의 라벨 불일치와 근거 부족, 낮은 personalization, 안전성(위기 시 인간 연계) 결여 같은 격차를 체계적으로 드러내는 것이다. 연구진은 Alzheimer’s Caregiver Stress-Process Model을 ‘a-priori scaffold’로 삼아 Best-Fit Framework Synthesis와 두 차례 질적 연구(총 41명 인터뷰)를 함께 삼각검증(triangulation)해 설계·평가 지침으로 이어지게 했다.

- **Empirical Impact**: 문헌 검토와 인터뷰는 공격성/수면-리듬 교란 같은 BPSD(치매의 행동·심리 증상) 이슈가 관계 긴장과 통제 상실감을 증폭시키며, 지원체계 탐색의 어려움·상담 접근성 저하가 ‘보이지 않는 공백’으로 남는다는 점을 일관되게 보여준다. 또한 relational strain과 compassion fatigue처럼 상대적으로 덜 다뤄진 영역이 확인되었고, respite·peer support·경계 설정 및 자기돌봄 같은 방향이 기술 설계 로드맵과 연결되어 치매 돌봄 혁신의 공통 언어로 기능할 가능성을 제시한다.



### STARE: Surprisal-Guided Token-Level Advantage Reweighting for Policy Entropy Stability (https://arxiv.org/abs/2606.19236)
Comments:
          LLM, Reinforcement Learning

- **Prior Approaches**: RLVR은 수학·코딩 등에서 복잡한 추론을 끌어내기 위해 verifiable rewards를 활용하는 post-training으로 자리 잡았고, 그중 GRPO는 value network 없이 group-normalized advantage로 학습을 단순화했다. 하지만 학습이 길어지면 policy entropy가 급격히 줄며 출력 다양성이 사라지고, 그룹 내 샘플이 동질화돼 상대적 advantage 추정이 약해지면서 장기 학습이 병목된다. 기존 완화는 (1) importance-sampling ratio 클리핑 조정, (2) 양·음 롤아웃에 대한 trajectory-level 가중, (3) entropy 정규화/advantage reshape 같은 token-구조를 세밀히 다루지 못하는 방식에 치우쳐 “붕괴 메커니즘”에 대한 정량적 처방이 부족했다.

- **Core Contribution**: 이 논문은 GRPO에서 token-level entropy 변화가 trajectory-level advantage와 next-token 분포의 entropy sensitivity function의 곱으로 분해된다는 1차 그라디언트 분석을 제시한다. 그 결과, advantage–surprisal이 4분면 구조로 얽히며 저-surprisal 토큰은 entropy 감소 방향 업데이트를 주도하고, entropy를 올려줄 수 있는 고-surprisal 토큰은 희소성 때문에 충분히 반영되지 못하는 “token-level credit assignment mismatch”가 핵심 원인임을 보여준다. 또한 이 mismatch가 near-criticality(임계 부근) 성질을 가져, 토큰 가중치에 대한 비교적 작은 개입으로 entropy 진화 부호를 뒤집을 수 있다고 주장한다.

- **Technical Challenges**: 가장 큰 난제는 entropy 붕괴를 유발하는 “어떤 토큰이” 문제인지 알아내고, 이를 GRPO의 clipped surrogate 안에서 안정적으로 개입하는 설계를 만드는 것이다. 저자들은 이 임계 토큰(고-surprisal 중심)을 이론적으로 정당화하되, 각 위치의 정확한 임계값 계산은 비용이 크기 때문에 batch-internal surprisal quantile로 entropy-critical 부분집합을 근사 선택한다. 이어서 선택된 토큰 집합의 effective advantage를 reweight하고, 배치 평균 entropy가 목표 범위를 벗어나면 작동/복귀하는 target-entropy closed-loop gate를 넣어 entropy를 안정적으로 규제한다.

- **Empirical Impact**: 실험에서 STARE는 1.5B~32B 모델 스케일 전반(Short CoT, Long CoT, multi-turn tool use)에서 수천 스텝에 걸친 RL 학습 동안 policy entropy를 목표 밴드에 유지하며 학습 안정성을 보인다. AIME24와 AIME25에서는 DAPO를 포함한 경쟁 기준선 대비 평균 정확도가 4%~8%p 개선되었고, reflection 관련 토큰 및 응답 길이가 함께 증가해 exploration–exploitation 균형이 오래 유지된다는 관찰이 제시된다. 즉, “entropy collapse의 토큰 수준 원인 규명 → 최소 침습적 reweighting+closed-loop 제어”라는 경로가 장기 post-training의 실질적 성능 향상으로 연결된 셈이다.



### Mechanism-Guided Selective Unlearning for RLVR-Induced Reasoning (https://arxiv.org/abs/2606.19222)
Comments:
          15 pages, 4 figures, 7 tables

- **Prior Approaches**: 기존 unlearning은 forget 예제에 대한 gradient ascent, retain-aware 목적, NPO/SimNPO 같은 negative preference 최적화, 그리고 representation control 같은 국소 업데이트를 주로 쓴다. 다만 대부분은 SFT로 유도된 행동을 대상으로 평가돼, RLVR로 유도된 reasoning에서 동일한 unlearning 타깃이 맞는지 불명확하다는 공백이 남아 있다. 특히 RLVR은 SFT와 다른 방향성의 확률/업데이트 구조를 만들 수 있어, 표준 방법이 ‘틀린 행동 부분공간’을 건드릴 위험이 제기된다.

- **Core Contribution**: 이 논문은 RLVR-induced reasoning을 지우는 과정에서 발생하는 mis-targeting(목표와 업데이트 방향 불일치)을 체계적으로 진단하고, 그에 맞춘 Mechanism-Aligned Selective Targeting(MAST)를 제안한다. 핵심은 full-parameter 업데이트가 RLVR 체크포인트에서는 “정답 지우기”만으로는 해결되지 않으며, retain과 비타깃 능력을 함께 망가뜨리는 경계가 존재한다는 점이다. MAST는 이 경계를 넘지 않도록, attention-projection 텐서를 메커니즘 점수로 랭킹한 뒤 상위 subset만 업데이트한다.

- **Technical Challenges**: MAST가 잘 작동하려면 RLVR에서 실제로 이동하는 weight-space 방향과, forget gradient가 결합되는 텐서를 찾아내야 한다. 논문은 delta-log-probability 구조, off-principal(기준 모델 주(主)공간에서 벗어남) 에너지 비율, 그리고 forget-loss 그래디언트와의 coupling 크기(코사인 기반)를 함께 써서 attention 텐서별 점수를 만들고, 나머지(MLP/정규화/임베딩 등)는 고정한 채 선택된 텐서만 gradient ascent로 업데이트한다. 또한 “forget accuracy만” 보지 말고 solution-trajectory와 final-answer에 대한 log-probability 변화를 함께 봐야 한다는 평가 함정도 함께 제시한다.

- **Empirical Impact**: 실험은 Qwen2.5-Math-1.5B(주 모델)와 Qwen3-1.7B-Base(교차 검증)에서, SFT-to-RLVR가 토큰 단위 확률 구조에서 다르게 나타난다는 점과 mis-targeting 가설을 뒷받침한다. MAST는 MATH에서 통계적으로 유의미한 forgetting을 만들면서도 GSM8K와 MATH retain을 거의 유지하며, 예로 MATH forget은 45/150→37/150으로 줄고 McNemar p=0.0078을 보고한다(전체 파라미터 unlearning은 retain과 GSM8K를 함께 붕괴). 교차 모델에서도 full-parameter unlearning의 collateral 손상이 유의미하게 크고, MAST는 동일 목적에서 GSM8K 정확도를 더 잘 보존해 “선택 업데이트 + 메커니즘 정렬”의 효과가 일반화됨을 보여준다.



### Machine Unlearning for the XGBoost Model with Network Intrusion Datasets (https://arxiv.org/abs/2606.19220)
Comments:
          12 pages, 7 tables, WorldCist'26 Conference

- **Prior Approaches**: 기존 Machine Unlearning(MU)은 주로 딥러닝과 이미지 데이터에 집중돼, 정형(tabular) 데이터와 XGBoost 같은 전통 ML 모델로 확장된 연구는 상대적으로 부족했습니다. exact unlearning(완전 삭제 보장)은 full retraining 부담이 커서 비실용적이었고, SISA나 DaRE처럼 부분 재학습/트리 재구성을 통해 비용을 줄이려는 시도가 이어졌습니다. 평가 역시 model utility, unlearning efficiency, forgetting quality로 나뉘지만, forgetting 품질 지표가 제거 비율에 따라 잘 작동하지 않을 수 있다는 한계가 드러납니다.

- **Core Contribution**: 이 논문은 XGBoost-Forget라는 MU 방법을 제안해 XGBoost에서 특정 학습 샘플의 영향만 효율적으로 제거하는 문제를 다룹니다. SISA의 shard-and-slice 아이디어를 XGBoost의 트리 추가/체크포인트 구조에 맞게 적용해, 삭제 요청 시 영향을 받는 슬라이스만 재학습하도록 설계했습니다. 네트워크 침입 탐지(NID)에서 대표적인 정형 데이터셋 IoT-23, GeNIS에 적용해 실사용 맥락을 강화한 점도 기여로 볼 수 있습니다.

- **Technical Challenges**: 핵심 난제는 정형 데이터에서 XGBoost의 예측 성능을 크게 훼손하지 않으면서, 삭제된 샘플의 영향만 국소적으로 지우는 구조를 만드는 것이었습니다. 저자들은 데이터를 shard로 분할하고 slice 단위로 트리를 점진 추가하며, 삭제가 필요한 슬라이스만 재구성하고 나머지는 캐시 재사용으로 시간을 절감하는 방식으로 해결했습니다. 또한 삭제되는 비율이 작을 때 forgetting-quality 지표가 민감하게 반응하는지 문제를 함께 확인했는데, JSD는 기대 패턴을 보이지 않아 평가 설계의 어려움도 제기됩니다.

- **Empirical Impact**: IoT-23과 GeNIS 실험에서 XGBoost-Forget는 원 모델 대비 ACC/REC/PREC/F1 같은 예측 성능을 거의 유지하면서, full retraining 및 NN 기반 SISA 대비 더 빠른 unlearning 시간을 보였습니다. forgetting 품질은 백도어 유사 설정에서 Attack Success Rate(ASR)가 unlearning 이후 retraining 기준선에 가깝게 하락하며 효과가 관찰됐습니다. 다만 JSD는 두 데이터셋 모두에서 낮게만 유지돼 소량 삭제 설정에선 지표가 덜 유효할 수 있음을 보여, MU 평가 지형에 실무적 경고를 남겼습니다.



### Forecasting what Matters: Decision-Focused RL for Controlled EV Charging with Unknown Departure Times (https://arxiv.org/abs/2606.19199)
Comments:
          ACM e-Energy 2026 5 pages, 1 figure, 1 table

- **Prior Approaches**: EV 충전 스케줄링은 MPC가 짧은 예측 구간을 반복 최적화하며, RL은 상호작용으로 정책을 학습하는 방식으로 발전해 왔다. 그러나 두 접근 모두 실제 현장에서 핵심인 출발 시각(또는 세션 종료 정보)을 알기 어렵고, 대부분 연구는 정확한 출발 시간을 전제로 하거나 perfect foresight에 가까운 가정에 의존한다. 출발 시간 예측을 하더라도 보통은 예측 정확도 중심으로 학습해 제어 의사결정 품질에 대한 오차 전파가 문제로 남는다.

- **Core Contribution**: 이 논문은 출발 시간처럼 제어에 필요한 미지 정보를 다루기 위해 decision-focused RL(DF-RL)을 제안한다. DF-RL에서는 세션 지속시간(출발 시각을 대체하는 정보)을 예측하는 forecaster를 RL과 end-to-end로 함께 학습해, 예측이 “얼마나 정확한가”가 아니라 “어떤 충전 행동을 만들어내는가”에 의해 최적화되도록 만든다. 결과적으로 충전 정책의 기대 누적 보상과 미공급 에너지(unsupplied energy)를 동시에 개선하는 방향으로 학습이 정렬된다.

- **Technical Challenges**: 핵심 기술 난제는 예측 오차가 하류의 강화학습 정책 성능을 깎는 구조에서, 예측 모델을 의사결정 관점으로 학습시키는 일이다. 저자들은 Soft Actor Critic(SAC) 기반 에이전트에 회귀 forecaster를 상태에 포함시키고, forecaster 학습 손실에 decision-focused 항(행동의 advantage에 가중된 항)을 추가해 RL가 선택할 행동의 품질이 예측 파라미터 업데이트에 피드백되게 했다. 또한 β로 회귀 손실과 DF 손실의 비중을 조절해, 미지 정보 추정의 보수성/공격성을 정책과 균형 있게 맞추도록 설계했다.

- **Empirical Impact**: 실험은 병원 주차장 EV 충전 세션 데이터 기반 시뮬레이션으로 350개 학습, 120개 테스트를 구성해 진행했으며, BAU(즉시 충전), 출발 예측 없는 RL, conventional forecast를 쓴 RL과 비교했다. 그 결과 DF-RL은 conventional forecast 대비 총 reward 최대 14% 향상과 unsupplied energy에 해당하는 미공급(미충전) 에너지 약 55% 감소를 보였고, 단순 회귀 예측만 쓰는 RL보다 unsatisfied EVs도 줄이는 경향이 관찰됐다. 출발 시간을 과대평가하는 conventional forecaster 대신 DF-RL은 보수적으로 짧게 예측해 조기 충전을 유도함으로써 “완료 실패”를 줄인다는 해석도 사례로 제시된다.



### The More the Merrier: Combining Properties for ABox Abduction under Repair Semantics for ELbo (https://arxiv.org/abs/2606.19197)
- **Prior Approaches**: 기존 연구는 ABox abduction을 repair-based semantics(Brave/AR)에서 다루면서, Σ-restriction, non-triviality, subset/cardinality minimality, conflict-confining 같은 “단일 속성”의 성질과 복잡도를 각각 분석해 왔다. 또한 불일치 데이터에서는 설명을 위해 여러 repair를 함께 고려해야 하는데, missing entailment에 대해서는 주로 결측이 아니라 충돌(blocking) 관점의 설명이 더 많이 연구됐다.
이번 논문은 여러 속성(예: Σ-restriction+non-triviality, conflict-confining+minimality)을 동시에 만족하는 가설이 현실적으로 더 유용하다는 문제의식에서 출발하지만, 그 조합은 기존 문헌에서 충분히 다뤄지지 않았다.

- **Core Contribution**: 본 논문의 핵심은 EL_bot에서 ABox abduction을 Brave/AR 의미론 아래 “복수 속성 조합”까지 확장해, 존재(생성) 문제와 검증 문제의 복잡도를 정리하는 것이다. 특히 저자들이 관찰한 중요한 포인트는, 특정 경우에는 추가 속성을 더 요구해도 계산 복잡도가 실제로 증가하지 않는다는 점이다.
즉, 여러 제약을 동시에 적용해도 난이도가 단일 속성 수준에서 그대로 유지될 수 있음을 이론적으로 보여준다.

- **Technical Challenges**: 주요 기술적 난관은 repair 기반 의미론에서 가설이 entailment를 만들도록 할 뿐 아니라, 각 repair의 존재/공통성을 만족(Brave: 일부 repair, AR: 모든 repair)시키는 동시에 최소성·충돌 관련 조건까지 함께 검사해야 한다는 점이다. 저자들은 verification/existence를 각각 ‘무엇을 추측하고 무엇을 검증하는가’로 분해해 guess-and-check 구조와 복잡도 클래스(NP, Σ2^P, coNP 등)로 환원해 상한과 하한을 맞춘다.
또한 fresh individual을 허용하지 않는 시그니처 제한을 포함해도, Σ 자체를 문제의 전체 시그니처로 두는 다항시간 환원 논리를 통해 “개별 속성의 하드니스가 그대로 전이”됨을 체계적으로 사용한다.

- **Empirical Impact**: 이 논문은 실험 성능 측정보다는, 설명용 abduction의 계산 가능성과 난이도 지도를 제공하는 데 의미가 있다. 특히 Brave에서 non-triviality+Σ-restriction의 존재 문제가 NP-complete, AR에서는 Σ2^P-complete로 정리되며, 조합 요구가 곧바로 더 어려운 문제를 만들지 않을 수 있음을 뒷받침한다.
결과적으로 지식기반 AI에서 불일치 상황의 설명을 설계할 때, 어떤 제약 조합이 이론적으로 감당 가능한지를 판단하는 기준을 제공한다.



### Language Models as Interfaces, Not Oracles: A Hybrid LLM-ML System for Pediatric Appendicitis (https://arxiv.org/abs/2606.19183)
- **Prior Approaches**: 기존 소아 충수염 진단 보조는 Alvarado Score, PAS처럼 점수 기반 규칙/모형에 의존하거나, 랜덤포레스트·gradient boosting 같은 지도학습을 사용해 tabular 변수로 위험을 예측해 왔다. 그러나 실제 임상 문서는 서사형 free-text로 기록되는 경우가 많아 tabular 입력을 만들기까지 변환 비용이 크고, 기관 간 데이터 시프트로 외부 성능이 흔들릴 수 있다. 한편 LLM을 진단 엔진으로 end-to-end로 쓰는 접근은 prompt 민감도, 정보 순서 영향, 그럴듯한 오답(hallucination) 위험 때문에 안전성이 약하다는 경고가 누적되고 있다.

- **Core Contribution**: ClaMPAPP은 LLM을 ‘진단 결정자’가 아니라 ‘인터페이스/특징 추출기’로 격리하고, 실제 위험 예측은 XGBoost 같은 검증된 ML 예측기로 수행하는 하이브리드 파이프라인을 제안한다. LLM은 note-like 서사에서 스키마 제약을 받는 임상 특징을 뽑고(필요 시 설명도 생성), 최종 appendicitis risk는 ML이 산출해 더 결정적이고 감사 가능(auditable)한 경로를 만든다. 또한 이 아키텍처는 충수염에 국한되지 않고, 서사형 문서와 tabular 예측기를 연결하는 ‘질병 불가지론적’ 설계로 확장 가능하다고 주장한다.

- **Technical Challenges**: 핵심 기술 과제는 LLM의 잘못된 값 생성(환각)과 추출 실패가 곧 진단 오류로 이어지는 것을 막는 것이다. 논문은 특징 추출 뒤에 deterministic Feature Validator를 두어 생리·임상 범위 제약을 위반하면 값을 잘라내지 않고 NaN(결측)으로 바꿔 입력 편향을 최소화하며, XGBoost의 결측 처리 능력을 활용한다. 더불어 서사 입력에 대한 견고성을 보기 위해 실제 EHR의 tabular 값을 템플릿으로 pseudo-노트 형태로 만들고 LLM rewriting 및 문장 순서 반전(permutation)까지 적용해 위치 편향까지 점검했다.

- **Empirical Impact**: 독일 병원 두 개의 독립 소아 코호트에서 내부·외부 검증을 수행한 결과, ClaMPAPP은 end-to-end LLM 기준선 대비 전반 성능이 더 좋았고 특히 missed appendicitis(거짓 음성) 비율을 줄이는 데 강점을 보였다. 또한 문장 순서를 섞었을 때 end-to-end LLM들은 민감도-특이도 균형이 불안정해지고 성능 저하가 컸던 반면, ClaMPAPP은 상대적으로 안정적인 안전 우선(triage safety) 프로파일을 유지했다. 이러한 결과는 “자연어 사용성은 LLM, 추론 신뢰성은 ML”로 역할을 분리하는 설계가 임상 의사결정 지원에서 더 적합할 수 있음을 실증적으로 뒷받침한다.



### Compute Efficiency and Serial Runtime Tradeoffs for Stochastic Momentum Methods (https://arxiv.org/abs/2606.19179)
- **Prior Approaches**: heavy ball(HB), Nesterov momentum, Accelerated SGD(ASGD) 같은 확률적 모멘텀 방법은 학습 속도를 높이는 대표 기법이지만, 그 “확률적 이득”은 serial runtime과 compute efficiency(CE)의 두 축에서 갈린다. 특히 배치 크기를 키우면 serial runtime은 줄어들 수 있으나, CE를 함께 유지하려면 수축 갭(contraction gap)이 배치 크기에 따라 어떻게 커지는지가 관건이었다.

- **Core Contribution**: 이 논문은 Gaussian 공분산을 갖는 일관(consistent) 선형 회귀 환경에서 확률적 HB와 ASGD의 배치-트레이드오프를 유한차원·이산시간으로 정식화하고, 배치 크기 스케일링에 대한 이산적 lower bound를 제공한다. 그 결과 HB는 임의 스펙트럼에서 SGD 대비 CE frontier를 본질적으로 확장하지 못하며, 다만 SGD 수준의 CE를 더 넓은 배치 구간에서 “보존”해 큰 배치가 serial runtime을 줄이되 가속 스케일에 도달할 때까지 유리함을 준다는 그림을 제시한다. 반면 ASGD는 스펙트럼 의존성이 더 강해, 특정 빠른 감쇠 스펙트럼에서는 작은 배치 CE를 개선하지만 배치가 커지면 그 이득을 serial runtime 개선과 교환한다고 결론내린다.

- **Technical Challenges**: 핵심 난점은 ‘배치 크기 변화가 수축 갭과 CE에 미치는 영향’을 스펙트럼별로 정량화하면서, 모멘텀 업데이트의 확률적 동역학을 이산시간에서 엄밀한 하한으로 묶어내는 것이다. 논문은 선형 회귀의 닫힌 분석 구조를 활용해 시간-분해의 수렴 성질을 이산적 lower bound 형태로 증명하고, HB에서는 SGD 수준 CE가 유지되는 “창(window)”의 크기(최대 √kappa 배수)를 스케일 법칙으로 도출한다. ASGD의 경우에는 power-law 형태의 스펙트럼 감쇠 속도에 따라 CE 우위가 유지되는 영역과 serial runtime로 전환되는 구간이 달라진다는 점을 실증/이론 정합으로 분리해 제시한다.

- **Empirical Impact**: 합성 선형 회귀 실험은 이론이 예고한 질적 레짐(regime)을 재현하며, ASGD와 HB가 느리게 감쇠하는 스펙트럼에서는 거의 겹치는 반면 빠르게 감쇠하는 스펙트럼에서는 예측된 CE–serial tradeoff가 뚜렷하게 나타난다고 보고한다. 또한 HB는 CE frontier 자체를 확장하지 못한다는 결론이 실험에서도 큰 배치에서의 성능 상한/전환 양상으로 드러나며, 배치 선택이 스펙트럼과 연결돼야 함을 강조한다. 결과적으로 실전 학습에서 “모멘텀 + 큰 배치”를 설계할 때, 단순히 배치를 키우는 전략이 아니라 문제 스펙트럼과 CE-serial 균형을 함께 봐야 한다는 메시지를 강화한다.



### Hardware- and Vision-in-the-Loop Validation of Deep Monocular Pose Estimation for Autonomous Maritime UAV Fligh (https://arxiv.org/abs/2606.19176)
Comments:
          6 pages 9 figues

- **Prior Approaches**: 기존 연구는 GPS 같은 외부 인프라 없이도 단안 비전으로 선박-상대 6D 포즈를 추정하려 했지만, 핵심은 결국 바다(현장) 검증의 난이도와 비용이었다. 특히 해상 환경은 기상 의존성이 크고 실험 실패 리스크가 높아, 시뮬레이션에서 얻은 성능을 바로 closed-loop로 옮기기 어려웠다. 또한 단순 시뮬레이션 중심 평가는 지연, 비동기 측정, 계산 자원 경쟁 같은 임베디드 실전 효과를 충분히 반영하지 못했다.

- **Core Contribution**: 이 논문은 3D Gaussian Splatting(3DGS) 기반 photorealistic 해상 장면 렌더링을 실시간으로 스트리밍하고, 온보드 Transformer 기반 단안 포즈 추정(TNN-MO)과 DKF 기반 지연 보정을 결합해 완전한 closed-loop 자율비행을 하드웨어로 검증했다. 렌더링/통신/추론으로 인해 발생하는 delayed vision 측정을, delayed Kalman filter로 적절한 과거 시점에 업데이트한 뒤 현재 시점 상태로 재전파해 기하 제어에 일관된 상태를 제공한다. 결과적으로 ‘시뮬레이션-해상배치’ 사이의 안전하고 현실적인 중간 단계(hardware-realistic intermediate stage)를 제시한다.

- **Technical Challenges**: 주요 기술 난관은 (1) end-to-end perception latency(렌더링·전송·추론 지연)와 (2) 불규칙한/asynchronous 업데이트, (3) Jetson Orin NX에서의 자원 제약이 perception–estimation–control 결합 성능을 흔든다는 점이다. 논문은 지연 측정의 out-of-sequence 문제를 DKF의 히스토리 버퍼 업데이트와 재전파로 처리하고, 파이프라이닝을 통해 처리량(업데이트 주기)을 늘리는 동시에 throughput–latency 트레이드오프를 명시적으로 운용한다. 또한 TensorRT 변환(FP16)과 멀티스레딩으로 추론 지연을 줄이고, 고정 지연/고정 업데이트율 조건에서 안정성을 유지하도록 동기화 전략을 적용했다.

- **Empirical Impact**: 실험은 모션캡처 기반 indoor 환경에서 자율 이륙-궤적 추종-착륙을 수행하며, 지연된 단안 포즈를 DKF로 융합한 뒤 안정적인 closed-loop 추종을 달성했다. 정량적으로 position MAE 0.066m, velocity MAE 0.032m/s, attitude MAE 2.13°의 낮은 추정 오차와 함께 control MAE 0.089m, 0.062m/s, 4.00°를 보고했다. 특히 Wi-Fi 전송 지연 변동을 포함한 최악 조건(총 지연 약 0.55s)에서도 estimator consistency와 비행 안정성이 유지되어, 해상 UAV 자율성 개발에서 비용과 위험을 줄이는 검증 경로로 의미가 크다.



### A Clinician-Centered Pipeline for Annotation and Evaluation in Ultrasound AI Studies (https://arxiv.org/abs/2606.19174)
Comments:
          Accepted to MIUA 2026

- **Prior Approaches**: 기존 의료영상 플랫폼(CVAT, Label Studio, MONAI Label 등)은 주로 Ground Truth 라벨링/수정에 초점이 있어, 여러 모델을 블라인드로 비교하는 clinician-in-the-loop 평가 워크플로와 재현 가능한 통계 산출을 통합해 주기 어렵다. 또한 독립적으로 설계된 리더(reader) 프로토콜이 많아 기관·연구그룹 간 평가 일관성과 재현성이 떨어지고, 프라이버시·거버넌스 때문에 원격 다기관 참여가 제한되는 문제도 남아 있다.

- **Core Contribution**: 이 논문은 초음파 AI 연구를 위한 clinician-centered 원격(annotation+preference ranking) 평가 파이프라인을 제안한다. 중앙 서버에 이미지와 모델 출력을 호스팅하고, 경량 웹 브라우저 인터페이스로 임상의가 로컬 데이터 다운로드 없이 블라인드 랭킹과 검토를 수행하도록 설계했으며, 결과를 다중 평가자 기준으로 집계하고 자동 통계까지 생성한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 데이터 공유 없이 원격 다중 평가를 가능하게 하는 보안/배포 구조, (2) 블라인드 비교를 위한 무작위·독립 모델 순서 제공, (3) 읽기 실험에서 필요한 정합성 지표를 자동으로 일관되게 계산하는 것이다. 연구자는 중앙 서버에서 실험 설정(평가 모드, clinician group, 랜덤 모델 오더, 블라인드 설정)을 관리하고, 클라이언트는 스트리밍된 세그멘테이션 오버레이 위에서 마스크 편집과 1~5점 랭킹을 수행하며, 서버의 Python 모듈이 Spearman 상관, Kendall’s τ, top-1 선택 빈도 및 inter-rater agreement를 산출한다.

- **Empirical Impact**: 태아 초음파 세그멘테이션 실험에서 6명의 평가자(전문가/일반가/비전문가)가 웹 인터페이스로 참여했고, Spearman 상관과 Kendall’s τ 기준으로 그룹 간 중간~강한 양의 일치가 관찰됐다. 블라인드 평가에서는 later active learning iteration 모델(M3~M5)이 더 자주 선택되는 경향이 나타나, 임상의 관점의 선호가 정량 지표와 어떻게 연결되는지 탐색하는 데 파이프라인이 유용함을 시사한다. 원격 참여 오버헤드도 비교적 낮아(라벨링 45~50초, 블라인드 랭킹 약 30초) clinician-centered, 재현 가능한 human-AI 평가를 확장하는 데 의미가 있다.



### Essential Subspace Merging for Multi-Task Learning (https://arxiv.org/abs/2606.19164)
- **Prior Approaches**: 모델 merging은 공통 pre-trained 체크포인트에서 fine-tuning된 여러 태스크 모델의 지식을 한 모델로 합치려는 시도지만, 태스크 업데이트를 단순 평균하거나 task vector를 합치면 inter-task interference가 크게 발생한다. 이를 줄이기 위해 SVD로 task vector/업데이트를 저랭크로 잘라내는 방법들이 제안됐지만, SVD의 순위 기준이 파라미터 에너지에 치우쳐 실제 데이터 분포에서의 기능적 영향(activation shift)을 충분히 반영하지 못한다.

- **Core Contribution**: 이 논문은 태스크 업데이트가 만드는 출력 변화(output shift)가 소수의 principal directions에 에너지가 집중된다는 관찰을 바탕으로 Essential Subspace(필수 부분공간)를 정의한다. 이를 토대로 Essential Subspace Decomposition(ESD)를 제안해, 파라미터 에너지 대신 출력 activation shift의 설명력 기준으로 업데이트를 분해·절단하며 기능 보존에 더 직접적으로 정렬되게 한다. 또한 이 ESD 위에 training-free static merging인 Essential Subspace Merging(ESM)과, residual을 low-rank experts로 두고 forward 시 prototype-based routing으로 선택하는 ESM++를 구축해 태스크별 전문성을 함께 유지한다.

- **Technical Challenges**: 핵심 난제는 “파라미터 관점의 저랭크화(SVD)”가 아니라 “출력 관점의 기능적 저랭크화”를 어떻게 구현하느냐이며, 이를 위해 ESD는 프록시 데이터로부터 태스크 업데이트가 유발하는 계층별 입력/출력 shift를 측정한 뒤 PCA로 필수 basis를 만든다. ESD가 잘라낸 성분이 기능적으로 덜 중요하다는 점을 eigenvalue 기반 truncation error로 정리하고, ESM에서는 태스크 간 간섭을 줄이기 위해 필수 성분을 orthogonalize해 compact multi-task 모델을 만든다. ESM++에서는 정적 합성 과정에서 희석된 태스크별 residual을 다시 ESD로 분해해 전문가로 저장하고, per-layer prototype 기반 cosine similarity로 추론 시 expert를 동적으로 조합한다.

- **Empirical Impact**: 실험은 비전(8/14/20 태스크), discriminative 언어(GLUE 8태스크, RoBERTa-Base), 생성형 언어(MergeBench 기반 Llama-3.2-3B 5전문가) 전반에서 수행됐고, ESM은 static merging 계열에서 평균 성능을 크게 끌어올리며 다수 설정에서 최상 또는 동률 성능을 보였다. routing을 추가한 ESM++는 특히 태스크 수가 늘어날수록 inter-task interference를 더 효과적으로 완화하며, discriminative에서는 GLUE에서 76%대 평균 정확도를, 생성형에서는 fine-tuned upper bound에 근접하는 점수를 달성했다. 또한 ESD는 SVD 대비 에너지 집중도와 feature 보존(CKA 관점)에서 일관된 개선을 보여 “기능에 기반한 필수 부분공간 합성”이 실증적으로 유효함을 확인시켰다.



### AdsMind: A Physics-Grounded Multi-Agent System for Self-Correcting Discovery of Adsorption Configurations on Heterogeneous Catalyst Surfaces (https://arxiv.org/abs/2606.19152)
Comments:
          37 pages, 5 figures

- **Prior Approaches**: 이 분야는 DFT의 조합폭발 문제를 줄이기 위해 후보를 좁히거나(그래프/환경 기반, 휴리스틱 열거) 평가를 빠르게 하는 MLFF(예: MACE, CHGNet, EquiformerV2)를 결합해 왔습니다. 하지만 많은 LLM 에이전트 기반 파이프라인이 open-loop로 동작해 초기 제안이 잘못되면 MLFF relax 이후의 물리적 결과로 스스로를 교정하지 못했습니다.

- **Core Contribution**: AdsMind는 Adsorption configuration discovery with Machine intelligence and relaxation feedback라는 이름처럼, LLM 플래너와 MLFF relaxation을 닫힌고리(closed-loop)로 묶어 에러를 피드백으로 수정하는 멀티에이전트 프레임워크를 제안합니다. Chemical Slip 검출, FORBID(금지 지시), TERMINATE(종료 지시)로 “실패한 바인딩 모드”를 다음 제안에서 구조적으로 배제해 신뢰도를 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 (1) 거대한 흡착 구성공간에서 LLM이 제안한 바인딩 인덱스·사이트 타입이 relaxation 단계에서 틀어질 때 이를 어떻게 진단하고 (2) 다음 반복에서 어떻게 안전하게 교정할지입니다. AdsMind는 Validator로 스키마 수준 오류를 사전 차단하고(인덱스 유효성, 사이트 타입-바인딩 호환성, 중복 방지), Analyzer가 relax 트랙을 화학적으로 해석해 slip/해리 여부 같은 물리 기반 진단을 Planner의 프롬프트와 제약으로 되돌리는 구조로 해결합니다.

- **Empirical Impact**: AA20와 OCD-GMAE62에서 네 가지 LLM 백엔드로 실험한 결과, AdsMind는 AA20에서 100%, OCD-GMAE62에서 98.8%의 성공률을 보여 open-loop 대안 대비 안정적으로 높은 탐색 신뢰도를 입증했습니다. 또한 MLFF relaxation 횟수는 케이스당 약 14배 적게 쓰면서(각각 4.11, 4.67 relax) DFT/PBE(VASP) 검증에서는 분자 흡착에서 open-loop 출력이 자주 발생시키는 흡착 에너지 부호(sign) 오류를 AdsMind가 모두 보존해 정량 오차도 더 가깝게 맞추는 것으로 나타났습니다.



### OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems (https://arxiv.org/abs/2606.19145)
- **Prior Approaches**: 기존 하이브리드 모델링은 물리 기반(상징/기계론) 성분과 신경 네트워크 잔차를 더해 동역학을 설명하려고 한다. 특히 APHYNITY처럼 symbolic 구조가 고정된 경우에는 L2 regularization(정규화)로 잔차가 상징 성분에 직교가 되도록 분리된다는 관점을 제시했다. 하지만 symbolic 구조를 sparse discovery로 ‘학습’할 때는 L2 중심의 기준선이 깨져 신경 성분이 상징 방향을 다시 흡수하며 중복·해석불가능 문제가 생긴다.

- **Core Contribution**: 이 논문은 symbolic 성분과 neural 잔차의 ‘겹침’을 직접 억제하는 OrthoReg(Orthogonal Regularization)를 제안한다. 경험적 내적 공간에서 neural augmentation이 symbolic library의 span과 정렬되는 정도(상관)를 페널티로 넣어, 상징은 라이브러리가 표현 가능한 부분을 담당하고 신경은 남는 잔차만 담당하도록 분해 가능성을 높인다. 결과적으로 sparse symbolic discovery에서도 해석 가능한 상징-신경 보완 분해를 목표로 한다.

- **Technical Challenges**: 핵심 난제는 L2 정규화가 제공하던 투영 기반 직교성 논리가 sparse discovery(연속 sparsity penalty 등)에서는 성립하지 않는다는 점이다. 저자들은 최적화 목표에 맞춰, 학습 데이터 입력(실제 경험적 내적 정의)에서의 상징-신경 overlap을 항으로 추가해야 한다는 점을 이론적으로 정리하고 이를 그대로 목적함수에 반영한다. 훈련은 fit loss(벡터장 회귀 또는 one-step 예측)와 함께 L1 sparsity(상징 성분의 희소성) 및 overlap 정규화 항을 더해 end-to-end로 최적화한다.

- **Empirical Impact**: 실험에서는 라이브러리 미스매치가 부분적으로 존재하는 동역학 벤치마크에서 OrthoReg가 상징 성분 복구(sparse symbolic recovery)를 개선하고 분포 밖(out-of-distribution) 일반화도 향상시켰다. 단순 L2 정규화 기반 하이브리드 대비, 신경 잔차가 상징 방향을 재학습하는 실패 모드를 줄이는 것이 관찰된다. 과학적 ML과 회귀 기반 시스템 식별에서 ‘해석 가능성+표현력’의 균형을 더 견고하게 만들 수 있다는 점에서 의미가 있다.



### A Technical Taxonomy of LLM Agent Communication Protocols (https://arxiv.org/abs/2606.19135)
- **Prior Approaches**: 기존 LLM 에이전트 통신은 MCP처럼 에이전트-컨텍스트(tool·데이터) 연계를 다루거나, 일부 에이전트-에이전트 프로토콜이 있어도 서로 교차 운용이 어렵다는 문제가 누적돼 왔습니다. 관련 연구들은 보안 위협 분류, 인터넷 아키텍처 관점 설계원칙, 혹은 소수 프로토콜의 비교 같은 접근을 제시했지만, 실제 프로토콜들을 일관된 추상 구조로 계통적으로 분류하기엔 차원이 부족했습니다. 특히 ‘표준화가 왜 필요한가’는 강조되지만, 어떤 설계 특성이 실제 채택과 연결되는지 추적 가능한 분류 체계는 부족했습니다.

- **Core Contribution**: 이 논문은 채택 가능한 9개 오픈소스 LLM 에이전트 통신 프로토콜을 대상으로, 통신 프로토콜을 이해·비교·추적할 수 있는 기술 분류체계(taxonomy)를 제안합니다. 반복적 구축 절차(Nickerson et al. 방식)를 따라 메타-특성, 종료 조건을 명시하고, 5차례 반복(경험→개념 3회, 개념→경험 2회)으로 분류 축을 확정했습니다. 결과적으로 프로토콜을 counterparty(상대방), payload(페이로드), interaction state(세션/상호작용 상태), discovery mechanism(발견 메커니즘), schema flexibility(스키마 유연성) 5차원으로 정리합니다.

- **Technical Challenges**: 핵심 기술 난제는 ‘서로 다른 구현을 공통 추상 축으로 매끄럽게 정렬’하는 것입니다. 논문은 프로토콜마다 메시지 형태·세션 지속 여부·런타임 스키마 처리·서비스/에이전트 발견 방식이 달라 동일 축에서 분리 가능해야 한다는 조건을 반복적으로 점검하며, 상호배타적·전체포괄적 특성 설계를 목표로 했습니다. 또한 분석 결과, 다수 에이전트-에이전트 프로토콜이 hybrid payload와 session-state persistence를 함께 쓰고, 미리 정의된 스키마 지원이 많되 일부는 runtime 협상으로 schema flexibility 추세가 보인다는 패턴을 통해 분류 축의 타당성을 보강합니다.

- **Empirical Impact**: 9개 프로토콜의 실제 분류 결과는 단기적으로 에이전트-에이전트와 에이전트-컨텍스트(tool·데이터) 통신을 한데 묶는 방향으로의 수렴 압력이 관찰된다는 결론으로 이어집니다. 반면 장기적으로는 단일 프로토콜이 versatility, 효율성, portability를 동시에 최대로 만족시키기 어렵고, federated·layered 프로토콜 스택으로 발전할 가능성을 제시합니다. 더불어 프로토콜 선택을 돕는 프레임으로 기능하는 한편, privacy와 policy enforcement 같은 공백 연구 과제도 구체적으로 드러냈다는 점에서 분야에 실무적·학술적 의미가 큽니다.



### Pareto Q-Learning with Reward Machines (https://arxiv.org/abs/2606.19134)
Comments:
          Accepted at the ICAPS 2026 Workshop on Bridging the Gap Between AI Planning and (Reinforcement) Learning (PRL)

- **Prior Approaches**: 기존에는 Reward Machines(RMs)로 표현된 복잡한 보상(비마르코프적 보상, RM-encoded rewards)을 다루기 위해 Q-Learning with Reward Machines(QRM) 같은 방법이 등장했다. 한편 Pareto Q-Learning(PQL)은 다목적 강화학습에서 Pareto front를 근사하려고 하지만, RM이 주는 자동자 구조를 제대로 활용하지 못하면 상태공간을 키우는 방식에 가까워질 수 있다.

- **Core Contribution**: 이 논문은 Pareto Q-Learning with Reward Machines(PQLRM)로, PQL의 Pareto front 근사와 QRM의 factored automaton 구조 활용을 결합한다. 그 결과 RM 기반 보상 구조에서도 Pareto-optimal(파레토 최적) 정책을 생성하면서, 멀티플 정책 형태를 유지하는 학습 절차를 제안한다.

- **Technical Challenges**: 핵심 난제는 RM으로 인코딩된 보상이 비마르코프적일 때도 Pareto Q-estimate의 집합을 효율적으로 유지하면서, 동시에 RM의 자동자 구조를 분해해 학습을 안정화하는 것이다. 저자들은 PQL의 벡터 Q-estimate 집합 업데이트에 QRM의 factored automaton 정보를 결합해, cross-product MDP로 단순 확장하지 않고도 sample-efficient하게 수렴하도록 설계했다.

- **Empirical Impact**: 실험에서 PQLRM은 cross-product MDP에 naive PQL을 적용한 기준선보다 더 빠르게 수렴했으며, QRM만으로는 얻기 어려운 Pareto-optimal 정책 합성을 보여줬다. 즉, RM 기반 다목적 보상 환경에서 ‘더 빠른 수렴’과 ‘더 넓은 파레토 해 탐색’을 동시에 달성하는 것으로 평가된다.



### Equivariant Graph Neural Networks Improve Optical Spectra Prediction for Materials Screening (https://arxiv.org/abs/2606.19133)
- **Prior Approaches**: 기존 광학 스펙트럼 예측 surrogate 모델은 저수준 이론(IPA, tight-binding 등)로 계산된 제한 데이터에 학습되거나, 회전에 불변인 scalar 특징(방향 정보가 사라진 표현)에 의존하는 경우가 많았다. OptiMate3B 계열은 선그래프/보로노이 그래프 구성으로 각도 정보를 일부 반영하지만, 기하를 rotation-invariant scalar로만 인코딩해 표현력이 제한된다. 또한 angular 상관을 암시적으로만 다뤄 데이터가 부족할 때 불리해질 수 있다.

- **Core Contribution**: 본 논문은 equivariant graph neural network인 GotenNet을 광학 스펙트럼 예측에 맞게 GotenNetOpt로 각색해, 방향성/기하 표현을 모델 구조 자체로 보장하도록 했다. 특히 GotenNet에 covalent radius 임베딩 게이팅을 추가하고, OptiMate3B의 readout 설계를 결합해 thin-film optics에 중요한 0–8 eV 구간과 정적 실 퍼미티비티 예측 성능을 끌어올린다. IPA 스펙트럼에서 학습 후 RPA로 fine-tuning하는 전략도 함께 사용한다.

- **Technical Challenges**: equivariant 모델이 광학 스펙트럼의 복소 유전율(실수부/허수부, Kramers-Kronig 연관)을 안정적으로 회귀하려면, 원자 배치의 방향 정보를 제대로 담는 tensor(steerable) 표현과 물리적으로 의미 있는 그래프 구성/특징 설계가 필요하다. 논문은 degree-1 방향 벡터와 higher-order spherical harmonics를 edge 텐서 입력으로 쓰고, 노드의 고차 steerable 특징을 message passing으로 갱신해 각도 상관을 암묵적으로 학습하도록 했다. 여기에 원자 크기 정보를 반영하는 covalent radius RBF 확장 및 곱셈 게이팅을 넣어, 같은 주기/족 원소라도 유효 결합 크기가 다른 경우의 차이를 모델이 구분하도록 해결했다.

- **Empirical Impact**: RPA 스펙트럼 데이터 10,533 구조(0–20 eV 범위)와 IPA/RPA 두 단계 평가에서 GotenNetOpt는 기존 state of the art보다 전반적으로 우수하며, 특히 0–8 eV 구간과 Re(ε̄(0)) 같은 정적 실 퍼미티비티 예측에서 개선 폭이 가장 크다. mean보다 median이 크게 낮은 분포 꼬리(outlier) 특성도 확인되며, RPA에서 IPA 대비 더 큰 향상이 관찰된다. 또한 low-data 환경의 GNNOpt(IPA 944개) 비교에서도 GotenNetOpt가 개선을 보이며, equivariant 기하 유도 편향이 광학 응답 예측에서 실질적 이득을 준다는 점을 경험적으로 뒷받침한다.



### Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams (https://arxiv.org/abs/2606.19111)
Comments:
          33 pages

- **Prior Approaches**: 기존 multi-agent LLM 연구는 debate, 역할 분담/분해, self-refinement처럼 지식 수준에서의 조정(무엇을 생각할지)을 바꾸는 방식이 주를 이룹니다. 하지만 평가가 최종 정확도에 치우쳐 있고, 실제로 ‘과정 수준 control이 언제 통하는가’를 분리해 측정한 사례는 드뭅니다. 또한 컨트롤러가 프롬프트처럼 모놀리식으로 설계되는 경우가 많아, 어떤 구성요소가 행동 차이를 만드는지(컴포넌트 단위 원인분해)도 깔끔히 검증하기 어렵습니다.

- **Core Contribution**: 이 논문은 multi-agent LLM 팀에서 ‘process-level coordination control’이 정확도를 더하는 조건을 측정 가능한 형태로 정식화하고, 그 조건이 team science의 contingency 예측과 일치하는지 검증합니다. 컨트롤러를 공통 action vocabulary(예: explore, revise, accept, synthesize) 위의 explicit control 정책으로 구현하고, 리더십 스타일(거래적/변혁적/상황적)을 그 정책으로 operationalize합니다. 결과적으로 “항상 이기기”가 아니라 “특정 조건에서만 가치가 생기는지”를 지도/이름표가 아닌 경계(boundary)로 다룹니다.

- **Technical Challenges**: 핵심 과제는 ‘정확도’ 대신 컨트롤러의 과정 차이를 직접 읽어내는 측정 설계이며, 저자들은 majority lock-in, 탐색률, round-0 오답 컨센서스에서의 recovery 같은 behavioral signatures를 1차 지표로 둡니다. 또 컨트롤러를 작은 명시적 action set으로 정의해 per-action ablation이 가능하게 만들었고, 임의 규칙(이론 없는 컨트롤)에서는 majority voting 수준으로 수렴해 ‘액션 구성’이 아닌 ‘이론 기반 규칙’의 역할을 분리합니다. 추가로 open-ended 수치형 과제의 추출 편향을 cross-round majority extractor로 통제해, 컨트롤 효과와 무관한 측정 노이즈를 줄입니다.

- **Empirical Impact**: 4개 task regime과 3개 open-weight 모델 계열에서, 어떤 컨트롤러도 전반적으로 정확도를 압도하지 못했는데 이는 contingency 관점의 null 결과와 일치합니다. 다만 round-0 독립 다수결이 신뢰롭지 않을 때에만 성능 이득이 나타났고, 그중에서도 recoverability가 가능한 영역에서 situational/transactional 계열이 기준선 대비 유의미하게 개선합니다(주로 round-0 majority가 흔들리는 조합에서 +8pp급 사례). 경계 probes(예: MATH-500 Level 5 확장, adversarial NLI, Winogrande, 도덕 판단)로도 “라운드0 신뢰도-회복 가능성-상호작용이 이미 복구하는지” 축이 재현되며, leadership substitutes/path-goal redundancy/situational readiness gap 같은 팀 과학 개념과 실측이 매핑됩니다.



### ProductConsistency: Improving Product Identity Preservation in Instruction-Based Image Editing via SFT and RL (https://arxiv.org/abs/2606.19103)
Comments:
          CVPR HiGen 2026

- **Prior Approaches**: 기존 instruction-based image editing 연구는 마스크·참조 기반 제어, 또는 MLLM/Transformer 결합으로 전반적인 구조 보존과 지시 수행을 강화해 왔습니다. 그러나 광고·커머스 맥락처럼 로고·브랜딩·객체 위 텍스트의 “픽셀 수준 정합성”이 필수인 경우, 텍스트가 깨지거나(맞춤법/문자 오류), 문장 자체를 환각하는 등 fine-grained object identity 보존이 자주 실패합니다. 또한 product+text fidelity를 전면에 둔 데이터셋/벤치마크가 부족해, 이 문제를 모델의 암묵적 능력에 맡겨온 한계가 있습니다.

- **Core Contribution**: 이 논문은 product-centric(제품/브랜드)과 text fidelity를 명시적으로 목표로 하는 ProductConsistency Dataset과 ProductConsistency Benchmark, 그리고 Product-aware 학습 프레임워크를 제안합니다. 합성 데이터 생성 파이프라인을 통해 검증 가능한 렌더링 텍스트를 포함한 학습용 데이터(SFT 87k, RL 869)를 만들고, 174개 제품(8개 카테고리)×5개 프롬프트(총 870개)로 표준 평가를 제공합니다. 특히 RL 단계에서 Cyclic Consistency reward로 “편집 후 이미지에서 생성된 캡션”이 원래 제품 설명과 의미적으로 맞아떨어지도록 유도해 제품 정체성을 보존합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 제품 위 텍스트를 포함한 편집 데이터의 품질을 사람이 검증하듯 확보하는 것과 (2) 제품 정체성을 직접 측정하기 어려운 보상 함수를 안정적으로 설계하는 데 있습니다. 저자들은 GPT로 제품/브랜드 시그니처와 텍스트 렌더링 목표를 만든 뒤 OCR로 문자 단위 정합성을 필터링해, “맞는 텍스트만” 남기는 고정밀 합성 데이터셋을 구축했습니다. 보상은 캡션 유사도(semantic proxy)를 Cyclic Consistency로 주되, OCR 기반 text reward를 함께 사용해 텍스트 오류에 더 민감하게 학습하도록 했고, GRPO 기반 RL로 두 보상을 균형 있게 최적화합니다.

- **Empirical Impact**: 실험에서는 Qwen-Image-Edit-2511과 Flux.1-Kontext-dev를 ProductConsistency로 SFT·RL 파인튜닝해 기준선 대비 일관된 개선을 보입니다. 가장 인상적인 결과로 Qwen-Image-Edit-2511은 RL(사이클릭 보상) 후 문자 오류율(CER)을 약 5배(1.0682→0.2080) 줄였고, Seg CLIP-I·Seg DINO-I도 함께 상승해 제품/텍스트 및 시각 품질이 동시 강화됨을 확인했습니다. 또한 OCR 정합뿐 아니라 MLLM-as-a-judge 평가에서 text fidelity·product consistency·aesthetics·instruction following 전반의 점수 상승이 관찰되어, 데이터·목표 설계가 실제 품질로 이어진다는 점을 뒷받침합니다.



### Where Did the Variability Go? From Vibe Coding to Product Lines by Regeneration (https://arxiv.org/abs/2606.19042)
Comments:
          VARIABILITY 2026

- **Prior Approaches**: 소프트웨어 변형성(software variability)은 수십 년간 feature model, variation point, 바인딩 타임 기반(컴파일/링크/런타임) 메커니즘으로 코드 내부에 계획해 관리해 왔다. 그러나 vibe coding에서는 LLM이 프롬프트로 프로그램 전체를 생성해, 전통적인 in-artifact variability 설계(#ifdef, getopt 등)가 사라지는지 체계적으로 분석된 연구가 부족했다. 이 논문은 GitHub의 vibe coded C/C++ 프로젝트 10개를 탐색해, 컴파일·런타임 어디에서도 거의 변형성이 관측되지 않는다는 점을 먼저 제기한다.

- **Core Contribution**: 논문은 “Variability by Regeneration(VbR)”를 제안하며, 변형성을 코드에서 없애고(=zero internal variability) 대신 선언적 사양(specification)에서만 관리하도록 방향을 바꾼다. LLM을 derivation engine으로 써서 각 variant마다 죽은 코드(dead code)가 없는 목적 전용 바이너리를 생성하고, dispatcher가 요청을 해당 바이너리로 투명하게 라우팅한다. 핵심은 바인딩 타임이 generation time(생성 시점)으로 수렴한다는 관찰을 전제로, 변형성 결정이 코드 생성을 끝내기 전에 고정되도록 설계한 product-family 접근을 형식화한 것이다.

- **Technical Challenges**: VbR을 실현하려면, (1) 제외된 feature가 어떤 형태로든 생성 코드에 남지 않아야 하고, (2) 포함된 feature는 반드시 실제 코드 위치에 구현되어 spec-코드 대응이 정확해야 한다. 논문은 이를 traceability relation 기반의 정적 검사와 컴파일·테스트 게이트로 파이프라인 계약(contract)처럼 강제하며, 위반 시 해당 산출물을 거부하고 재생성해 성질을 “신뢰”가 아니라 “구성”으로 보장한다. 또한 생성된 각 variant는 별도 바이너리라 코드 재사용이 줄 수 있으므로, 대신 스펙 변경 시 영향 받은 variant만 재생성하도록 하는 확장(anchored regeneration) 가능성까지 함께 논의한다.

- **Empirical Impact**: 탐색 분석에서는 vibe coded C/C++ 프로젝트들이 CLI 옵션과 preprocessor 변수 사용이 극히 적고(사실상 대부분 제로에 가까움), 전통적 SPL에서 기대하는 변형성 메커니즘이 거의 관측되지 않았다. VbR은 wc 제품군을 대상으로 end-to-end 파이프라인을 시연해 6개 feature, 3개 variant로 구성된 dispatcher-연동 바이너리 생성을 보여 주며, 스펙 수준에서 기능을 추가/삭제하면 해당 variant만 깔끔히 재생성되는 흐름을 제시한다. 저자들은 변형성이 AI 생성 소프트웨어의 코드가 아니라 specification에 둬야 한다는 관점을 제안하며, 이후 더 다양한 언어·도메인에서 in-artifact variability를 실증하고 파이프라인 자동 검증/스케일링을 진행하겠다고 밝힌다.



### A Hybrid LSTM--Vision Transformer Architecture for Predicting HRRR Forecast Errors (https://arxiv.org/abs/2606.19026)
Comments:
          This manuscript is a preprint and has been submitted for peer review to the Artificial Intelligence for the Earth Systems journal. The content is subject to change based on the outcome of the peer review process and should not be considered final or definitive. Copyright in this Work may be transferred without further notice

- **Prior Approaches**: 기존 NWP 예측오차(특히 PBL, 대류, 지형 영향 등)는 모델이 충분히 해상도화하지 못해 발생하며, 이를 줄이기 위해 ML 기반 예측오차 post-processing이 주목받아왔다. Evans et al.(2025)는 mesonet의 표면 관측과 HRRR 예측을 입력으로 하는 LSTM으로 HRRR 오차를 직접 예측해 기준선 대비 성능을 끌어올렸지만, 복잡한 수직 대기 진화가 나타나는 구간에서 성능이 저하되는 경향이 있었다.

- **Core Contribution**: 이 논문은 표면 관측 기반 LSTM에 수직 구조 정보를 결합한 hybrid LSTM-ViT(LSTM-Vision Transformer) 프레임워크를 제안한다. New York State Mesonet의 profiler(마이크로파 방사계)로부터 얻는 PBL/혼합층 관련 프로파일을 ViT가 attention으로 요약하고, 이를 LSTM decoder에 융합해 HRRR의 시간당 강수, 10 m 풍속, 2 m 기온 오차를 각 관측소 단위로 예측한다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 프로파일이 retrieval ill-posed 및 바이어스를 갖는다는 점, (2) 시간×높이×변수로 표현되는 프로파일을 ViT 입력 토큰으로 안정적으로 변환해 PBL 진화를 학습시키는 점이다. 연구진은 MWR의 Time×Height를 토큰 시퀀스로 flatten하고 위치·시간 임베딩을 더해 수직-시간 의존성을 모델링했으며, LSTM-ViT 출력 융합은 fusion MLP로 처리하고 학습 안정화를 위해 early stopping과 ReduceLROnPlateau를 적용했다.

- **Empirical Impact**: 실험 결과, 세 예측 대상 모두에서 profiler 유래 수직 구조를 추가한 LSTM-ViT가 baseline LSTM보다 예측오차 예측 skill을 개선했으며, 특히 짧은 리드타임과 PBL 활동이 강화되는 시기에 이득이 크게 나타났다. 강수 오차 예측에서는 기준선 대비 약 2배 수준의 predictive skill 향상이 관찰되어, 대류에 의해 오차가 진화하는 양상을 더 잘 포착하면서 PBL 관련 성능 저하도 완화했다. 운영 예보 관점에서 모델 bias와 예측 신뢰도에 대한 더 직접적인 가이드를 제공할 수 있다는 점에서 의미가 있다.



### FoMoE: Breaking the Full-Replica Barrier with a Federation of MoEs (https://arxiv.org/abs/2606.19025)
- **Prior Approaches**: 기존 LLM 사전학습은 대형 GPU 클러스터에서 전 모델을 복제하고, 매 동기화마다 전체 파라미터(또는 큰 페이로드)를 교환하는 데이터 병렬이 기본이었다. MoE는 토큰당 일부 expert만 활성화해 연산 효율을 높였지만, DiLoCo·Photon 같은 cross-datacenter 저통신 기법은 동기화 빈도만 줄이고 여전히 각 사이트가 ‘full-replica’로 큰 모델 상태를 주고받는 비효율이 남았다. 그 결과 WAN에서는 지연이 커지고, 가장 메모리 제약이 큰 사이트가 모델 크기의 상한을 결정하는 문제가 지속됐다.

- **Core Contribution**: 이 논문은 FoMoE로, expert layer를 worker(데이터센터) 사이에 분할 배치해 ‘full-replica’ 패러다임을 깨는 cross-site MoE 학습 시스템을 제안한다. 즉 각 사이트는 모든 expert를 보관·동기화하지 않고, 일부 expert만 학습/동기화해 payload 크기와 메모리 요구를 함께 줄인다. 또한 기존 저통신 로컬 업데이트의 이점을 유지하면서, 동기화 시 교환되는 모델 상태 자체를 축소하도록 학습 전략을 모델 구조와 공동 설계한다.

- **Technical Challenges**: 핵심 난제는 WAN에서 페이로드를 줄이면서도 (1) expert 간 라우팅이 붕괴하지 않고 학습이 안정적으로 수렴하며 (2) 부분 복제로 인한 데이터/전문가 불균형이 성능을 해치지 않게 만드는 것이다. FoMoE는 expert layer partition과 placement(고정/랜덤 포함) 설계공간을 정의하고, non-local expert를 건너뛰는 skip-token 메커니즘으로 “ghost experts”의 불필요 계산/흐름을 제어해 처리량을 끌어올린다. 더 나아가 proxy 설정에서 라우팅 안정성을 경험적으로 확인하고, 큰 규모 구성에서는 통신/메모리/연산량 관점의 시스템 모델링으로 이점을 투영한다.

- **Empirical Impact**: 평가에서는 부분 expert replication이 통신 비용을 효율적 기준선 대비 최대 1.42x, DDP 대비 최대 45.44x까지 줄이면서(특정 제어된 조건) perplexity를 유지하는 결과를 제시한다. throughput은 이론적으로 선형적인 이득을 기대할 수 있는 skip-token을 통해 경험적으로 최대 1.4x까지 향상됐다. 또한 학습된 proxy 구간에서 높은 routing entropy와 expert collapse 회피 같은 안정성 지표를 관찰하고, 100B 스케일로의 확장에서도 통신/메모리 이점이 유지될 것임을 모델링으로 뒷받침해 의미 있는 방향성을 제안한다.



### Spotlight: Synergizing Seed Exploration and Spot GPUs for DiT RL Post-Training (https://arxiv.org/abs/2606.19004)
- **Prior Approaches**: DiT RL post-training의 비용을 줄이기 위한 기존 연구는 주로 seed exploration로 수렴 속도를 높이거나(Xue et al., Ding/Ye, Li 등), spot GPU를 활용해 롤아웃 비용을 낮추는 방식(RLBoost 등)으로 나뉜다. 다만 seed exploration은 탐색 단계가 매 반복의 critical path에 붙어 반복 지연을 키울 수 있고, spot 기반 접근은 DiT의 롤아웃 완료 시간이 거의 균일해 training 동안 spot이 사실상 유휴로 남는 문제가 크다.

- **Core Contribution**: Spotlight는 DiT RL post-training에서 spot GPU의 유휴 시간을 ‘seed exploration’에 재활용해 전체 학습 비용을 절감하는 시스템을 제안한다. 핵심은 탐색이 직전 iteration의 stale model weights만으로도 seed 간 상대 순위를 보존해 동작할 수 있다는 점과, Sequence Parallelism(SP) 재구성이 기존 상태를 재사용하면 비싼 재초기화 없이 탄력적으로 이뤄질 수 있다는 두 통찰이다.

- **Technical Challenges**: 첫째, exploration이 모델 업데이트를 기다려야 한다는 직렬 의존성이 spot 유휴를 만든다; Spotlight는 stale weights로도 seed ranking이 대각선에 강하게 집중되는 실험 결과에 기반해, spot에서 exploration을 training과 동시에 돌린다. 둘째, spot preemption으로 SP 그룹이 깨져 재구성 비용이 수분 단위까지 커지는 문제를, CPU scheduler를 노드에 상주시켜 초기화 비용을 상쇄하고 같은 노드 내 peer로부터 NVLink 가중치 복사로 SP 재구성을 sub-second로 줄이는 elastic sequence parallelism으로 해결한다.

- **Empirical Impact**: Qwen-Image 기반 post-training에서 Spotlight는 목표 validation score를 최대 4배 빠르게 달성하며, 전체 비용은 spot 미활용 대비 1.9–6.4배, spot 활용 베이스라인 대비 1.4–6.4배까지 낮춘다고 보고한다. 또한 DeepSeek-OCR과 Geneval에서 512×512 및 1280×1280 해상도 조건 전반에 걸쳐 더 높은 이미지 품질(검증 점수)을 보여, 비용 절감이 성능 저하로 이어지지 않음을 실증했다.



### TRAP: Benchmark for Task-completion and Resistance to Active Privacy-extraction (https://arxiv.org/abs/2606.18996)
- **Prior Approaches**: 기존 에이전트 프라이버시 연구는 크게 두 갈래로 나뉩니다. 패시브 설정은 작업 수행 중 자연스러운 유출을 측정하지만, 공격자가 명시적으로 ‘캐내기’를 시도하는 상황을 다루지 못합니다. 반면 어드버사리얼 설정은 추출 공격에 대한 거부/차단을 보지만 작업 성능은 배제해, 단순히 거부만 해도 높은 점수를 얻는 문제가 있습니다.

- **Core Contribution**: 이 논문은 작업 정확도와 프라이버시 누출의 트레이드오프를 동시에 보려는 활성(active) 평가 벤치마크 TRAP(Task-completion and Resistance to Active Privacy-extraction)를 제안합니다. 각 인스턴스에서 모델은 문서의 private field를 도구 호출에 사용해 과제를 수행해야 하며, 동시에 같은 private field를 자연어로 요청하는 attack query에도 값을 응답하지 못해야 합니다. 이를 통해 ‘정확히 하면서도 절대 노출하지 않는’ 요구를 한 프레임에서 직접 비교합니다.

- **Technical Challenges**: 핵심 기술적 난관은 능력(정확한 필드 파싱·도구 인자 구성)이 곧 누출(응답에서 값 재현)로 연결된다는 점입니다. 저자들은 softmax 기반 모델에서는 어떤 prompt 중심의 soft-constraint(시스템 프롬프트 지시, instruction tuning, prompt optimization)도 누출 확률을 0으로 만들 수 없음을 정리합니다. 이후 해결책으로 private field isolation을 도입해, 모델이 평문 private 값을 받지 않고 해시 키(symbolic key)만 보게 만든 뒤 실제 값은 tool 실행 계층에서만 해소되도록 구조적으로 분리합니다.

- **Empirical Impact**: 22개 모델(프론티어 상용+오픈 가중치)을 다양한 스케일과 텍스트/이미지/멀티모달에서 평가한 결과, 작업 정확도가 높을수록 누출도 의미 있게 발생하며 모든 계열에서 non-trivial leakage가 관측됩니다. 또한 instruction-following 능력과 leakage rate가 상관되는 경향이 확인되어, 단순 ‘정렬’만으로는 갭을 메우기 어렵다는 메시지를 강화합니다. 반면 정확한 마스킹(Oracle) 하에서는 private field isolation이 누출을 크게 억제하면서도 작업 정확도를 기준선 수준으로 유지해, 해결 방향이 구조적 개입임을 실증적으로 뒷받침합니다.



### G-IdiomAlign: A Gloss-Pivoted Benchmark for Cross-Lingual Idiom Alignmen (https://arxiv.org/abs/2606.18989)
Comments:
          Accepted to ACL 2026

- **Prior Approaches**: 기존 이디옴 연구는 주로 detection(관용적 사용 여부)이나 disambiguation(문맥에서 문자/비유 의미 선택)에 집중해 왔습니다. 또 cross-lingual idiom alignment를 다뤄도 보통 표면형 단서나 lexical overlap에 기대는 경향이 강해, 비유 의미를 제대로 매칭하지 못하거나 저자원 언어에서 성능 격차가 커지는 문제가 드러났습니다.

- **Core Contribution**: 이 논문은 gloss를 의미의 공통 기준점으로 쓰는 gloss-pivoted 벤치마크 G-IdiomAlign을 제안합니다. Wiktionary의 English gloss로 각 이디옴을 고정(앵커링)하고, 재현 가능한 high-confidence reference alignment set까지 구축해 진단형 평가가 가능하게 했습니다. 또한 Multiple-Choice Idiom Equivalence와 Gloss-Contrastive Generation( No-gloss vs With-gloss ) 두 프로토콜로 의미 피벗의 효과를 분리해 측정합니다.

- **Technical Challenges**: 핵심 난관은 (1) gloss 없이 생성하면 모델이 literal translation 같은 표면 통계에 치우치고, (2) gloss 기반 매칭도 임베딩 근사 때문에 잘못된 1:1 대응이 생길 수 있다는 점입니다. 저자들은 후보군을 gloss-embedding 공간에서 검색한 뒤 mutual nearest neighbors(MNN)와 언어쌍별 분포 기반 컷오프로 약한 페어를 제거해 잡음을 줄였고, 생성 평가에서는 결정적 파싱을 위해 정확히 한 개 이디옴만 출력하도록 통제했습니다. 추가로 attention 기반 진단을 통해 With-gloss가 어떤 내부 신호(특히 attention head)에 더 크게 영향을 주는지도 분석합니다.

- **Empirical Impact**: 다양한 LLM에서 공통적으로 literal translation 편향이 지배적 실패 모드로 확인됐고, 특히 저자원 타깃 언어일수록 악화됩니다. With-gloss는 embedding 기반 의미 프록시 아래에서 일관되게 성능을 올리지만 Acc@0.80 수준이 여전히 ‘모가 큰’ 것으로 나타나, 오픈 생성 공간에서 의미-동등 이디옴을 만드는 난이도가 큽니다. Qwen3-8B 분석에서는 With-gloss로 생성 품질이 좋아질수록 gloss anchoring이 강해지고, 조건 간 차이는 레이어 전반보다 attention head 쪽에 더 집중된다는 근거를 제시합니다.



### Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering (https://arxiv.org/abs/2606.18986)
- **Prior Approaches**: Time-MQA류 TSQA는 시계열을 BPE 기반 텍스트로 직렬화해 LLM에 넣는 방식이 많았지만, 숫자가 자리값이 아니라 빈도 기반으로 잘게 쪼개져 크기·스케일·추세 같은 수치 기하가 사라진다는 한계가 있다. 또 patch 기반 인코더는 고정된 윈도우 크기로만 표현해 특정 과거 시점(정확한 인덱스) 접근이 어렵고, 패딩/재분할 때문에 샘플링 레이트나 길이가 달라지면 전이 성능이 흔들린다.

- **Core Contribution**: CADE는 TSQA를 위한 입력 표현 단계에서 patch/직렬화 없이, 각 timestep을 LLM 임베딩 공간에 직접 임베딩으로 매핑하는 direct timestep embedding을 제안한다. 이어서 분류 데이터로부터 class-name 텍스트 앵커를 frozen 상태로 두고, 한 방향(one-directional) supervised contrastive loss로 시계열 임베딩을 언어적 의미 공간에 정렬해 멀티태스크 공통표현을 강화한다.

- **Technical Challenges**: 가장 큰 기술 난관은 LLM의 토크나이저가 연속 수치를 안정적인 metric 구조로 전달하지 못한다는 점이며, CADE는 BPE 파편화를 제거하기 위해 point-wise 선형 인코더+MLP projector로 각 값/인덱스를 1:1로 임베딩에 보존한다. 또한 분류 태스크에만 유효한 contrastive 신호를 다른 과제까지 공유해야 하므로, 분류 샘플만 풀링 앵커로 쓰되 text 앵커는 frozen으로 고정하고 시계열 측에만 그라디언트가 흐르는 설계를 통해 의미 정렬을 안정화한다.

- **Empirical Impact**: Time-MQA 벤치마크에서 CADE는 6개 TSQA 태스크 전반에 걸쳐 성능을 일관되게 개선했으며, open-source 및 proprietary LLM 기반 기준선 대비 우수한 결과를 보고한다. 특히 patch 기반·LoRA/FT·직렬화(BPE) 등 대표 대안을 함께 비교해, 표현 병목(토크나이제이션 파편화, 고정 그라뉼러리티)을 줄이는 것이 멀티태스크 TSQA 신뢰성 향상으로 이어진다는 점을 실증적으로 뒷받침한다.



### CAPRA: Scaling Feedback on Software Architecture Deliverables with a Multi-Agent LLM System (https://arxiv.org/abs/2606.18976)
Comments:
          Accepted for publication at the 38th International Conference on Software Engineering Education and Training

- **Prior Approaches**: 기존 자동 평가는 주로 코드 채점(static analysis, unit testing 등)이나 단일 텍스트 산출물 평가에 집중돼, 소프트웨어 아키텍처 문서처럼 구조적 완결성과 요구사항 추적성까지 요구하는 경우는 자동화가 제한적이었다. LLM 기반 ‘LLM-as-a-Judge’ 계열은 가능성을 보였지만, 멀티모달 문서에서 증거 없이 판단(환각)하거나 중복된 비판을 내놓을 위험이 남는다. 또한 단일 아티팩트/표현에 치우쳐 있어, 문서 전반의 교차 일치성 검증과 출처 구간 고정(evidence anchoring)이 충분히 체계화되지 못했다.

- **Core Contribution**: CAPRA는 소프트웨어 아키텍처 deliverable을 대상으로 템플릿을 준수하는 개인화 LaTeX 피드백을 생성하는 multi-agent LLM 시스템을 제안한다. 핵심 설계는 (1) 멀티모달 문서 파싱, (2) 차원별 전문 에이전트 분석, (3) 증거 검증과 일관성 병합을 통해 신뢰도를 높이는 워크플로우다. 성적은 매기지 않고 formative feedback을 목표로 하며, 교육 현장에서 필요한 “기술적으로 타당한 피드백”에 초점을 둔다.

- **Technical Challenges**: 멀티모달 문서(PDF의 텍스트+UML)를 안정적으로 읽어내고, 에이전트가 낸 주장에 대해 실제 소스 구간을 근거로 고정(evidence anchoring)하는 것이 가장 큰 기술 난제다. CAPRA는 PyMuPDF로 텍스트를 추출한 뒤 gpt-4o vision으로 UML/이미지를 구조화 텍스트로 변환해 전체 흐름을 보존하고, deterministic 설정과 함께 fuzzy matching(정규화 Levenshtein 거리 기반)으로 인용 근거를 검증한다. 이후 ConsistencyManager가 중복/충돌을 병합·정렬하며, 템플릿 LaTeX 골격에 짧은 내러티브만 삽입해 포맷 오류와 과도한 생성 위험을 줄인다.

- **Empirical Impact**: 10개 학생 리포트의 예비 평가에서 CAPRA는 엄격한 기준(strict) 집계 시 88.8%의 평가 항목을 만족했고, Cohen’s kappa=0.582로 사람 평가자 간 중간 수준의 일치도를 보였다. 처리 시간은 리포트당 약 4분 내외(평균 248초)였고 비용은 약 $0.44로, 수작업(30–45분) 대비 7.2–10.8× 가속을 보고했다. 환각성 비근거 비판을 Evidence Anchoring 단계로 걸러내며, 요구사항·테스트·아키텍처 간 의미 불일치와 계층 간 설계 공백을 실제 사례에서 포착해 교육용 확장 가능성을 시사한다.



### A Controlled Benchmark of Quantum-Latent GAN Augmentation for Brain MRI (https://arxiv.org/abs/2606.18970)
Comments:
          This work has been submitted to the IEEE for possible publication. This work has been submitted to the IEEE for possible publication

- **Prior Approaches**: 기존 의료영상(특히 뇌 MRI) 데이터 증강은 GAN·확산 모델을 통해 라벨 부족과 클래스 불균형을 완화하는 방식이 널리 쓰였습니다. 최근에는 quantum generative models(예: QGAN)이 정확도 향상을 보고하지만, 단일 실행(seed) 기반 결과가 많고 양자 생성기와 classical 생성기의 파라미터 예산이 같지 않아 ‘양자 구조’가 이득을 만든 것인지 불명확했습니다. 또한 어떤 데이터 규모에서 이득이 나타나는지, 생성 샘플의 품질·다양성은 어떤지까지 함께 분석되지 않는 경우가 흔했습니다.

- **Core Contribution**: 이 논문은 뇌 MRI 증강에서 quantum latent generator의 기여를 분리하기 위한 ‘통제된 벤치마크’를 제시합니다. 이미지 인코딩은 KL-regularized latent space로 고정하고, conditional Wasserstein GAN-GP에서 variational quantum generator와 파라미터 수가 거의 같은 classical generator를 비교합니다. 이후 생성 latents를 디코드해 라벨 데이터 fraction 5%~100% 전 구간에서 다운스트림 분류 성능을 평가합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 양자 생성기와 classical 생성기의 표현력 차이를 ‘공정하게’ 통제하는 것과, 성능 차이를 통계적으로 신뢰할 수 있게 분해하는 것입니다. 논문은 (1) VAE로 latent space를 만들고 (2) quantum generator(4 qubits, depth 2)와 classical generator(파라미터 1648 vs 1632)를 거의 동일한 예산으로 맞춘 뒤 (3) 8개 seed에 대한 paired significance testing과 다중비교 보정을 적용합니다. 또한 downstream 정확도뿐 아니라 intra-set diversity(예: SSIM, 픽셀 표준편차)와 latent distribution overlap(예: t-SNE)으로 생성 샘플이 실제 분포를 얼마나 따르는지까지 함께 진단합니다.

- **Empirical Impact**: 결과적으로 모든 라벨 데이터 fraction에서 real-data-only 대비, quantum 또는 classical 증강 variant이 유의미하게 더 좋지 않았고 두 생성기 간 성능도 통계적으로 구분되지 않았습니다. 낮은 데이터(예: 5%·10%)에서 보이는 ‘소폭 이득’은 생성 샘플이 off-distribution이고 모드 붕괴가 심한 상태에서 나타나 regularization 효과에 가깝다고 해석됩니다. 저자들은 이러한 통제 실험 프로토콜을 공개해, 의료영상 분야에서 quantum generative augmentation의 근거를 더 엄격히 검증하는 testbed로 활용하길 기대합니다.



### TransitNet: A Compact Attention-Augmented Deep Learning Framework for Low-SNR Transit Blind Searches (https://arxiv.org/abs/2606.18932)
Comments:
          24 pages, 23 figures, 3 tables, submitted to MNRAS

- **Prior Approaches**: 기존의 탐색은 주로 TLS(Transit Least Squares)와 BLS(Box Least Squares)처럼 수학적 검출기에 의존해 왔고, 저 SNR(신호대잡음비) 구간에서는 오탐/미탐이 늘어 성능 한계가 커졌습니다. 또한 블라인드 검색(미지의 주기·시점)에 가까운 조건에서 기준선(threshold)을 객관적으로 보정하는 데이터·평가 파이프라인이 일관되게 제시되지 않아, 방법 비교와 튜닝이 어렵다는 문제도 있었습니다.

- **Core Contribution**: 이 논문은 저 SNR에서의 Transit blind search를 위한 컴팩트한 attention-augmented 딥러닝 프레임워크 TransitNet을 제안합니다. 더불어 블라인드 검색 조건을 반영해, 데이터 구성·벤치마킹·threshold selection을 하나로 묶는 통합 프레임워크를 만들어 현실적인 개발과 객관적 임계값 캘리브레이션을 가능하게 했습니다.

- **Technical Challenges**: 핵심 난제는 중·장주기의 지구형 행성처럼 신호가 약한 구간에서 잡음 속 전이를 안정적으로 찾아내는 동시에, 블라인드 검색 환경에서 임계값을 공정하게 고정하는 데 있습니다. 저자들은 통합 데이터 구축/평가/임계값 선택 프레임워크로 threshold을 객관화하고, TransitNet에 attention을 더해 전이 window와 midpoint를 추정하도록 설계해 검출뿐 아니라 시점 추정까지 함께 수행하게 했습니다.

- **Empirical Impact**: Kepler 타깃에서 만든 recovery 벤치마크에서 TransitNet은 SNR 6~8의 어려운 구간에서 95.2% 정확도를 보이며 TLS·BLS를 능가했고, ROC-AUC 0.974와 PR-AP 0.982를 달성했습니다. 또한 injected Earth-size 및 sub-Earth-size 실험에서 recovery rate 93.0%로 TLS(63.1%)와 BLS(60.0%)를 크게 앞섰고, 독립 평가에서 주입된 전이의 97.4%가 추정 transit window에 완전히 포함됐습니다; 실제 Kepler 관측에서는 34개 확정 행성을 모두 회복했으며 평균 midpoint 오차는 1.24시간이었습니다. 모델 크기 약 1.5MB, 추론 효율이 높아 CPU-TLS 대비 12~25배, CPU-BLS 대비 4~5배의 속도 향상을 보였고, 결과적으로 저 SNR 블라인드 전이 탐색의 정확성·확장성·계산 효율을 동시에 입증해 더 긴 주기의 지구형 행성 탐색 확장에 동력을 제공한다는 점에서 의미가 큽니다.



### As Easy as Rocket Science: Assessing the Ability of Large Language Models to Interpret Negation in Figurative Languag (https://arxiv.org/abs/2606.18922)
Comments:
          16 pages, 16 figures; for associated code and data see this https URL To be published in Transactions of the Association for Computational Linguistics

- **Prior Approaches**: 기존 연구는 은유·직유 해석을 NLI나 문장 패러프레이징 같은 과제로 평가해 왔으며, 많은 경우 fine-tuning을 통해 성능이 크게 개선된다고 보고돼 왔다. 또한 negation 자체가 모델에 취약하다는 결과가 축적돼 있지만, figurative language와 negation이 동시에 등장할 때의 상호작용은 상대적으로 덜 탐구돼 왔다. Fig-QA처럼 덜 관습적인 figurative language를 다루는 데이터가 있었음에도, negation 유형까지 세분해 out-of-the-box 해석을 체계적으로 본 연구는 부족했다.

- **Core Contribution**: 이 논문은 Fig-QA에 metaphor/simile뿐 아니라 negation, tense, concreteness를 새로 라벨링해 복합 현상(특히 negation+figurative)의 해석 능력을 분리해 측정한다. 아울러 Fig-QA를 바탕으로 literal negation을 소규모로 구성해 figurative language 없이 negation 효과만 고립해 비교한다. 다양한 언어 모델을 fine-tuning 없이 그대로 평가해, “두 현상 동시 등장”이 어떤 병목을 만드는지 정면으로 드러낸다.

- **Technical Challenges**: 핵심 난관은 (1) LLM이 autoregressive 로그확률/프롬프트 응답에 따라 선택 편향이 생길 수 있고, (2) 모델 종류(embedding 기반 vs Llama/GPT 계열)에 따라 동일한 평가 신호를 맞추기 어렵다는 점이다. 저자들은 prompt style을 mid-phrase(예: “In other words”) 중심과 question-answer(두 후보 중 무엇이 더 적절한지 질문) 방식으로 나눠 비교하고, Llama는 출력 변동성 때문에 log-likelihood 기반 판별을 사용한다. 그 결과 prompt style이 성능을 크게 좌우하며, connector-free mid-phrase는 특히 성능을 크게 떨어뜨림을 확인한다.

- **Empirical Impact**: 실험에서 전반적으로 negation과 figurativeness의 결합이 ‘특정한’ 어려움으로 나타났고, 유형별로도 not/antonym 등 negation 양상이 성능 격차를 만든다. 특히 question-answer 프롬프트에서는 여러 모델이 인간 성능에 근접하지만, negating simile에서 성능 하락이 두드러져 상호작용 병목이 관측된다. 또한 SBERT 임베딩 분석(PCA)에서는 PC3가 negation과 강하게 연관되며, tense·concreteness도 일부 영향이 있으나 negation 효과가 상대적으로 더 크다는 결론을 제시해 향후 평가/프롬프트 설계에 실질적 기준을 제공한다.



### SAERec: Constructing Fine-grained Interpretable Intents Priors via Sparse Autoencoders for Recommendation (https://arxiv.org/abs/2606.18897)
- **Prior Approaches**: 기존 intent-based recommender는 사용자 행동 시퀀스를 클러스터링하거나 고정된 intent prototype에 매핑해 의도를 중간 계층으로 둡니다. 하지만 intent 개수를 미리 정해야 하고, 시퀀스 품질에 민감해 coarse한 intent 집합이 되기 쉽습니다. 또한 intent가 라벨 없는 잠재 벡터라서 의미적 grounding이 약해 설명 가능성이 제한됩니다.

- **Core Contribution**: SAERec은 리뷰/아이템 설명 텍스트 코퍼스에서 fine-grained하고 해석 가능한 intent 공간을 자동으로 구성해 추천을 유도합니다. 핵심은 LLM 임베딩을 Sparse Autoencoder(SAE)로 희소·분리된 feature로 분해하고, 각 feature의 의미를 LLM이 라벨링해 사람이 읽을 수 있는 intent로 남기는 것입니다. 이후 개인( personal ) intent과 공개( public ) intent를 검색한 뒤, 멀티 브랜치 attention으로 시퀀스 모델링에 주입하고 adaptive fusion으로 최종 사용자 표현을 만듭니다.

- **Technical Challenges**: 텍스트 임베딩은 다중 의미(polysemantic)와 잡음이 얽혀 있어 supervision 없이 서로 다른 intent 신호를 분리하기 어렵습니다. SAE로 과완비(overcomplete) 희소 표현을 만들고, 각 희소 축을 단어 집합과 mutual information 기반 정렬로 해석 가능하게 만든 뒤, LLM 판단으로 recommendation에 유의미한 intent만 선별해 의미 없는 차원을 제거합니다. 또 intent와 시퀀스 모델의 latent space 불일치 문제를 해결하기 위해, 시퀀스 인코더 출력과 intent 벡터를 같은 공간에 정렬하고 personal/public dual-level retrieval로 관련 intent만 top-K로 뽑아 attention에 주입합니다.

- **Empirical Impact**: 공개 데이터셋의 광범위한 실험에서 SAERec은 최신 baseline을 일관되게 능가하며 정확도 향상을 보였습니다. 동시에 자동 생성된 intent는 사람 친화적으로 설명 가능한 형태를 제공해, 추천 근거를 “품질/가성비” 같은 공개 동기와 “민감성 피부” 같은 개인 동기로 구분해 제시할 수 있습니다. intent 수를 미리 정하지 않아도 되는 파이프라인은 실제 운영에서의 튜닝 부담과 coarse intent 문제를 함께 완화한다는 점에서 의미가 큽니다.



### Domain-Shift Aware Neural Networks for Unbalance Characterization in Rotating Systems (https://arxiv.org/abs/2606.18882)
- **Prior Approaches**: 기존 SHM의 데이터 기반 학습은 대부분 분류(fault diagnosis) 중심의 domain adaptation에 치우쳐 있었고, 회전기계에서도 MMD나 adversarial 학습이 주로 쓰였다. 그러나 regression(예: 불평형 질량 추정)에는 domain shift를 정면으로 다루는 연구가 상대적으로 적어, 학습 조건 밖의 운전에서 일반화가 취약하다는 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 회전축의 불평형 질량(단일 스칼라)을 추정하는 regression 문제에 domain-shift aware 신경망을 적용한다. 핵심은 source에서 라벨로 학습하는 회귀 손실에 더해, MMD 기반 최대 mean discrepancy로 source-타겟 간 잠재 표현을 정렬해 target에서의 신뢰도 저하를 완화하는 것이다.

- **Technical Challenges**: 가장 큰 기술적 난점은 target 구간에서 라벨이 없고, 물리적 거동과 domain discrepancy의 원인이 학습 중에 완전히 규명되지 않을 수 있다는 점이다. 저자들은 이를 위해 MMD를 네트워크의 feature extraction(잠재공간) 수준에서 정규화 항으로 결합하고, 회귀-정렬 손실의 균형을 sigmoid 형태의 가중치 스케줄로 점진적으로 조절해 학습 초기에 불안정을 줄였다.

- **Empirical Impact**: 실험은 단일축 불평형 상태와 더 나아가 벨트로 결합된 보조축을 가동해 domain discrepancy를 인위적으로 유발하는 테스트 리그에서 수집된 진동(3축 가속도) 데이터로 구성됐다. 결과적으로, 훈련 조건을 벗어난 운전(물리 동특성/불일치 요인이 알려지지 않거나 달라지는 상황)에서도 domain shift를 명시적으로 다루는 접근이 예측 정확도를 개선함을 보이며 Structural Health Monitoring의 regression 전이 학습 가능성을 확장한다.



### Scaling Learning-based AEB with Massive Unlabeled Data (https://arxiv.org/abs/2606.18864)
Comments:
          Accepted for presentation at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

- **Prior Approaches**: 기존 AEB는 TTC나 거리 임계값 같은 규칙 기반 접근으로 안전 우선순위를 제공하지만, 센서 잡음·추적 아티팩트·장꼬리 상호작용에서 취약해질 수 있습니다. 학습 기반 AEB는 데이터로 트리거 정책을 학습하려 하지만, 프로덕션에서 안전 라벨(사고/위험 이벤트 라벨)이 비싸서 대규모 스케일링이 제한돼 왔습니다. 이에 SSL/메타-피드백(MF-SSL) 계열은 라벨이 적을 때 unlabeled data로 성능을 키울 수 있으나, 안전 임계 경계의 앵커 모호성과 labeled-unlabeled mismatch 때문에 pseudo-label 오류가 커져 오히려 false activation(불필요 제동)을 유발할 수 있다는 점이 실전에서 과제로 남아 있었습니다.

- **Core Contribution**: 이 논문은 meta-feedback semi-supervised learning(MF-SSL)로, 대규모 무라벨 차량 플릿 데이터를 활용해 학습 기반 AEB를 “안정적으로 스케일”하는 프레임워크를 제안합니다. 핵심은 teacher가 unlabeled 주행에 pseudo label을 생성하되, 소량의 labeled anchor 성능으로 teacher를 업데이트시키는 메타-피드백 구조를 유지하면서도 앵커 모호성과 mismatch가 만드는 시스템적 pseudo-label 오류 증폭을 억제하는 것입니다. 이를 위해 Noise-Aware Decoupling과 kinematics-gated pseudo-labeling, teacher conflict penalty를 조합해 spurious trigger 위험을 줄이도록 설계했습니다.

- **Technical Challenges**: MF-SSL에서 가장 큰 문제는 앵커의 near-boundary ambiguity와 labeled-unlabeled mismatch가 결합될 때, teacher가 고확신으로 잘못된 pseudo label을 “수용(accept)”하고 그 오류가 student 업데이트를 통해 다시 teacher에 증폭되는 닫힌 고리(teacher-student loop) 현상입니다. 논문은 배포 위험을 accepted-error mass(잘못된 pseudo label의 수용 확률)와 coverage(수용 마스크로 커버되지 않는 영역)로 분해해, 단순히 임계값을 올려 오류를 줄이면 coverage를 잃어 스케일 이득이 사라진다는 딜레마를 보여줍니다. 해결책으로는 (i) Noise-Aware Decoupling에서 misclassified(라벨 충돌) 앵커를 teacher의 supervised update 경로에서 분리해 앵커 노이즈 주입을 줄이고, (ii) kinematics-based checker(TTC)로 안전이 명확한 구간은 pseudo-label 수용을 억제하며, (iii) teacher conflict penalty로 checker와 충돌하는 고확신 pseudo label을 학습에서 불리하게 만들어 mismatch 유발 “risk hallucination”을 억제합니다.

- **Empirical Impact**: 실험에서는 무라벨 데이터 규모를 1M→1B window까지 키울 때 일관된 이득을 확인했으며, 안전성은 개선하면서 comfort(불필요 제동 관련 지표) 안정성도 유지했다고 보고합니다. 특히 1B로 학습한 student 모델을 수십만 대 차량에 배포하고 약 10^9 km 주행 검증을 수행해, positive-to-false activation ratio가 100:1을 넘고 사고 없는 주행 마일리지가 프로덕션 규칙 기반 기준선 대비 35% 개선되는 성과를 제시합니다. 안전 임계 추론에서 “대규모 무라벨 플릿 스케일링”이 실제 배포에서도 안정적으로 작동할 수 있음을 보여준 점에서, ADAS/AEB 학습 파이프라인의 실전 가능성을 크게 확장했다는 의미가 있습니다.



### URDF Synthesis from RGB-D Sequences via Differentiable Joint Inference and Energy-Consistent Verification (https://arxiv.org/abs/2606.18861)
- **Prior Approaches**: 기존 연구는 관절의 종류·파라미터 추정, 파트(링크) 기하 복원, URDF 생성 등을 각각 따로 최적화하는 경우가 많았습니다. 또한 많은 파이프라인이 시각/형상 손실 중심으로 학습되어, 에너지 보존 같은 동역학 불변조건을 검증하거나 학습 신호로 직접 연결하지 못해 장시간 시뮬레이션에서 드리프트가 누적되는 문제가 지적됩니다. 더 나아가 세그멘테이션-후-피팅 방식은 작은 분할 오류가 관절축 추정 오차로 크게 번질 수 있습니다.

- **Core Contribution**: KinemaForge는 short RGB-D 시퀀스만으로 링크의 파트 기하, 조인트 토폴로지, 조인트 파라미터를 한 번에(조인트-레벨까지) 복원하는 constraint-driven 파이프라인을 제안합니다. 핵심은 Featherstone의 articulated-body dynamics를 미분 가능하게 연결해, 렌더링·동역학 불일치로부터 관절축 파라미터를 함께 업데이트하고 결과 URDF를 energy-consistent verifier로 검증한다는 점입니다. 즉, “그럴듯한 모델”을 넘어 “물리적으로 관측과 양립하는 모델”을 목표로 합니다.

- **Technical Challenges**: 난제는 (1) 분할/파트 제안에서 관절-파트 연계를 단단히 고정하지 못하면 오차가 시스템적으로 커지고, (2) URDF가 시각적으로는 맞더라도 자유 응답에서는 비물리적 거동을 할 수 있다는 점입니다. KinemaForge는 오버-세그먼트된 후보들 사이에 관절-파트 연계를 소프트 에지로 둔 kinematic constraint graph로 토폴로지를 먼저 안정화하고, screw-axis를 연속 변수로 두어 differentiable screw-axis solver로 관절축/원점/한계를 그래디언트 기반으로 최적화합니다. 여기에 energy residual loss를 추가해 재구성 모델을 같은 미분 가능 시뮬레이터에서 롤아웃하며 물리적 에너지 증분이 관측과 맞는지까지 페널티로 학습합니다.

- **Empirical Impact**: PartNet-Mobility 5개 카테고리 및 내부/외부 RGB-D 벤치마크에서 KinemaForge는 관절축 에러를 PARIS 대비 37.4% 줄이고, Ditto 대비 46.6% 줄였습니다. 50초 장기 롤아웃 드리프트는 PARIS 대비 64%, Ditto 대비 73% 감소했으며, URDF 기반 closed-loop 조작에서 성공률은 Ditto보다 14.6%p(예: 85.4% vs 70.8%) 향상되는 결과를 보였습니다. 또한 ablation에서 constraint graph와 differentiable joint solver, energy-consistency loss가 각각의 실패 모드를 어떻게 줄이는지 정량적으로 확인해 “물리 검증이 성능을 실제로 바꾼다”는 메시지를 강화합니다.



### Aligning Implied Statements for Implicit Hate Speech Generalizability with Context-Bounded Semi-hard Negative Mining (https://arxiv.org/abs/2606.18852)
- **Prior Approaches**: 암묵적 혐오 발화는 노골적 욕설 대신 비꼼, 완곡어, 수사적 질문처럼 의도(intent)가 맥락에 숨는 경우가 많아 표면 표현만으로는 부족합니다. 기존 supervised contrastive learning(SCL)은 암시된 진술(implied statement)을 positive로 삼거나 라벨 내 공유 의미를 positive로 만들지만, 미니배치 내 대부분을 negative로 밀어내면서 ‘가까운 반대 라벨’을 false negative로 취급해 국소 이웃이 흔들리고 도메인 전이가 약해질 수 있습니다.

- **Core Contribution**: 논문은 ImpSH라는 triplet 기반 학습 프레임워크를 제안해, positive는 (가능하면) post-implication 쌍에 정렬하고 negative는 ‘맥락에 한정된’ semi-hard negative만 골라 학습 신호를 안정화합니다. 또한 AugSH 변형을 통해 증강 기반 multi-view에서 어떤 이득이 생기는지 분리해, implied statement 감독이 추가로 제공하는 효과를 점검합니다.

- **Technical Challenges**: 핵심 난제는 “너무 가까운 반대 라벨을 negative로 잘못 밀어내면” 표현 공간이 왜곡된다는 점이며, 이를 위해 배치 내 전체 negative를 밀어내는 방식 대신 마진 근처의 near-confusion만 negative로 선택합니다. 구현적으로는 cross-entropy 분류 손실을 유지하면서 cosine 거리 기반 triplet objective에 context-bounded semi-hard mining을 결합하고, one positive per anchor로 positive 구성 방식을 ImpSH/AugSH에 맞게 분기합니다.

- **Empirical Impact**: IHC, SBIC, DynaHate에서 BERT와 HateBERT를 사용해 cross-domain 전이에 초점을 맞춘 결과, ImpSH는 대부분의 설정에서 표준 SCL 계열을 견조하게 대체하거나 개선하며 특히 일부 전이 방향에서 꾸준히 강한 성능을 보였습니다. Alignment와 uniformity(로컬 양성 쌍은 더 조밀하게, 전체 분포는 무너지지 않게) 분석에서도 ImpSH가 transfer에서 유리한 임베딩 기하를 형성하는 경향이 확인되었고, nearest-neighbor 사례는 도메인 이동 시의 대표적 false negative 양상을 보여주며 의도 분리 목적의 타당성을 뒷받침합니다.



### Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems (https://arxiv.org/abs/2606.18837)
- **Prior Approaches**: 기존 automatic-MAS는 크게 추론 시점(inference-time)과 학습 시점(training-time)으로 나뉜다. inference-time 계열은 frozen frontier LLM로 탐색은 하지만, 동일한 검색을 반복해 과거 시행착오를 축적·전이하지 못한다. training-time 계열은 fine-tuning으로 경험을 내재화하지만 작은 모델의 성능 상한과 대규모 고품질 궤적 데이터 요구, 그리고 초거대 frontier LLM로의 확장 비용 문제가 크다.

- **Core Contribution**: 이 논문은 고수준 오케스트레이션을 경험-유지와 파라미터 업데이트를 분리한 “evolvable Meta-Skill”로 모델링하는 Skill-MAS를 제안한다. 그 결과, frontier LLM은 고정한 채로도 Meta-Skill을 여러 라운드에 걸쳐 진화시켜, 과거 실패/성공의 구조화된 지식을 다음 생성에 반영한다. 단일 에이전트의 skill 진화가 아닌, MAS 생성 전체의 전략(분해-역할-워크플로) 수준을 직접 업데이트한다는 점이 핵심이다.

- **Technical Challenges**: 주요 기술 난제는 (1) 탐색 변동성 때문에 “진짜 실력 부족”과 “우연한 실행 잡음”을 구분해야 하고, (2) 경험을 효율적으로 축약해 일반화 가능한 원칙으로 바꿔야 하며, (3) 학습 비용 없이 closed optimization loop를 구성해야 한다는 것이다. 이를 위해 Multi-Trajectory Rollout으로 태스크별 확률적 분포(여러 궤적)를 샘플링해 불확실성/난이도를 계산하고, Selective Reflection에서 변동성과 난이도가 높은 우선 태스크만 뽑아 within-task 대비 분석과 cross-task 합성으로 모듈 단위의 수정을 Evidence 패키지로 구성한다. 수정은 세 모듈 스캐폴드를 유지하며 해당 모듈에 한정해 “전략 수준”으로만 반영되도록 제약한다.

- **Empirical Impact**: 네 가지 복잡 벤치마크와 네 가지 서로 다른 LLM을 Meta-agent로 사용한 실험에서 Skill-MAS는 초기 Meta-Skill만으로도 경쟁력 있는 성능을 보이며, 최적화된 Meta-Skill은 대부분의 baseline을 큰 폭으로 앞선다. 특히 비용-성능 관점에서 inference-time 계열의 반복 재최적화 비용을 피하면서도 training-time 계열의 일반화 한계를 완화해 더 유리한 절충점을 달성한다. 또한 진화된 Meta-Skill은 미지 태스크 및 다른 백본 LLM로의 transferability가 강해, 단순 프롬프트 탐색 이상의 “전략 원칙”이 학습되었음을 뒷받침한다.



### Improving Human-Robot Teamwork in Urban Search and Rescue Through Episodic Memory of Prior Collaboration (https://arxiv.org/abs/2606.18836)
- **Prior Approaches**: 기존 human-robot teaming 연구는 상호 적응과 팀 수준 조정 메커니즘을 다뤄왔고, MATRX USAR에서는 대화로 collaboration pattern(CP)을 외부화하면 협업 효율이 좋아진다는 결과가 축적돼 있다. 다만 CP를 “분석/성찰용 산출물”로만 취급하거나, 신뢰·의도 추론처럼 목적이 다른 지식 전이 중심이라 새 에피소드 초기에 재사용 가능한 팀 경험으로 연결하는 방식은 제한적이었다.

- **Core Contribution**: 이 논문은 이전에 발견된 CP를 knowledge-graph episodic memory로 구조화하고, 그래프 표현 학습으로 “대표적이고 효과적인” 단일 CP를 골라 다음 협업 시작 전에 로봇 메모리에 선주입하는 메커니즘을 제안한다. 특히 재사용되는 경험이 불투명한 policy 파라미터가 아니라 상황-행동-성과가 보이는 형태라, 이후 점검·수정이 가능하다는 점을 강조한다.

- **Technical Challenges**: 핵심 난제는 CP가 그래프 위상과 행동 순서(초기/후기 단계)가 중요한 구조 데이터를 갖는다는 점이었다. 연구진은 property graph로 CP를 모델링하고 RGCN에 edge type(상황/인간 행동 단계/로봇 행동 단계)를 반영해 node-type 분류 목표로 임베딩을 학습한 뒤, K-means로 CP를 군집화해 대표 CP를 centroid-nearest 방식으로 선택했다.

- **Empirical Impact**: 시뮬레이션 MATRX USAR에서 20명의 참가자, 160 라운드 관찰 결과로 단일 자동 선택 CP를 선주입하면 구조 성공률이 25.7%에서 41.3%로 상승하고 평균 작업 시간도 283초 줄었다. 개선은 상호 적응이 아직 충분히 일어나기 전인 interaction 초기에 가장 크게 나타나, “초기 진입점”을 바꾸는 재사용형 episodic memory의 효과를 실증적으로 보여준다.



### Target-confidence Recourse Using tSeTlin machines: TRUS (https://arxiv.org/abs/2606.18832)
- **Prior Approaches**: 기존 counterfactual explanations(CE)는 입력 변경 비용을 최소화하면서 결정 경계(label flip)만 뒤집는 데 초점을 둬, 같은 “찬성” 영역이라도 실제 수용/위험 관리에서 필요한 신뢰도는 반영하기 어렵습니다. 또한 경계 근처를 공략하는 해법은 잡음이나 모델 변화에 취약해 fragility가 자주 관측되며, 확률/신뢰도는 사후적으로 평가되는 경우가 많았습니다. 일부 강건 CE나 uncertainty-aware recourse는 안정성을 높이지만, 사용자가 원하는 confidence 수준을 생성 목표로 직접 지정하는 방식은 거의 없었습니다.

- **Core Contribution**: 이 논문은 Target-confidence Recourse Using tSeTlin machines(TRUST)로, recourse 생성 시 예측 confidence를 “사후 필터”가 아니라 “최적화 목표(제약)”로 전면 배치합니다. 표준 CE는 τ=0.5일 때를 포함하도록 일반화되며, 사용자가 최소 비용과 함께 “승인될 신뢰도(예: 80%) 이상”을 요구할 수 있게 합니다. 더 나아가 PTM(Probabilistic Tsetlin Machine)의 clause 기반 구조를 활용해, 같은 τ를 만족해도 어떤 규칙 활성 변화 때문에 신뢰도/견고성이 달라지는지 비교 가능한 설명층을 제공합니다.

- **Technical Challenges**: 핵심 난제는 non-convex하고 평가 비용이 큰 입력공간에서, “확률값이 목표 τ를 만족”하는 최소 변경 해를 안정적으로 찾는 것입니다. 논문은 PTM 확률이 clause 활성에 의해 분해된다는 장점을 살려 Bayesian optimization으로 목표 confidence를 직접 맞추는 탐색을 수행하고, 동시에 비용과 확률 편차를 함께 최적화합니다. 또한 PTM의 확률적 clause firing을 이용해 counterfactual 간 confidence 차이를 clause 수준으로 귀속(attribution)하여, 비용-신뢰도 선택을 기계적으로 해석 가능하게 만듭니다.

- **Empirical Impact**: 합성 데이터와 실세계 벤치마크에서 TRUST는 conventional boundary-based CE 대비 confidence와 robustness를 동시에 개선하는 경향을 보였고, 여러 작업에서 높은 강건성을 달성했습니다. 특히 Haberman dataset에서 L2 거리 0.10에 0.92 confidence를 보고하며, 목표 τ를 높일수록 decision boundary에서 더 깊게 이동해 잡음에 대한 유지율(robustness)이 상승하는 “설계로부터의 강건성” 패턴이 정량적으로 확인됩니다. 또한 동일/유사 신뢰도 조건의 recourse 후보를 clause 활성 차이로 설명할 수 있어, 고위험 의사결정에서 실질적인 의사결정 지원으로 확장될 수 있다는 점에서 의미가 큽니다.



### Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning (https://arxiv.org/abs/2606.18831)
Comments:
          15 pages, 6 figures, 12 tables

- **Prior Approaches**: 기존 long-context reasoning 개선은 주로 강화학습(RL)에서 reward 설계에 집중했지만, 핵심 신호가 부족해 긴 입력에서 증거를 찾는 단계가 정체되기 쉬웠습니다. 동시에 고품질 학습 데이터는 retrieval·multi-evidence synthesis·reasoning을 폭넓게 커버하기 어렵고, 합성 데이터도 범위가 좁거나 closed-source인 경우가 많았죠. 알고리즘 중심 접근은 auxiliary reward나 최적화 변형을 통해 해결을 시도했지만, 데이터 다양성이 병목이었습니다.

- **Core Contribution**: 이 논문은 long-context RL을 ‘데이터 중심’으로 재정의하며, 복잡한 reward engineering 없이도 단순하지만 효과적인 데이터 레시피만으로 성능을 크게 올릴 수 있음을 보였습니다. 레시피는 retrieval, multi-evidence synthesis, reasoning의 세 가지 상호보완적 능력을 각각 겨냥한 8개 데이터셋(총 약 14K 예시)을 조합합니다. 또한 이 개선이 에이전트형 과제로도 전이돼, agent-tuned 모델에 계속 RL을 하며 GAIA와 BrowseComp 점수가 추가로 상승함을 보여줍니다.

- **Technical Challenges**: 핵심 기술 난제는 긴 문맥에서 (1) 키워드 매칭 같은 지름길을 쓰지 않고 올바른 증거를 ‘찾아내는’ 것, (2) 여러 단서들을 빠짐없이 통합해 합성하는 것, (3) 긴 입력 위에서 복잡한 계산/추론을 유지하는 것입니다. 저자들은 fuzzy/paraphrase 기반 needle 설계와 near-duplicate 구분형 multi-needle, 파생 속성 집계형 multi-evidence synthesis, 단서 누락 시 실패가 확정되는 incomplete coverage, UUID 체인 은닉형 KeyChain 같은 방식으로 각 실패 모드를 데이터에서 직접 유도합니다. 이후 Group Relative Policy Optimization(GRPO) 기반의 minimal outcome-based 학습을 붙이고, 데이터 간 보상 스케일/분산 차이를 줄이기 위해 task-balanced sampling과 task-level advantage normalization으로 학습 경쟁을 완화합니다.

- **Empirical Impact**: 3개 Qwen 모델(Qwen3-4B/8B/30B-A3B)에서 7개 long-context 벤치마크를 실험한 결과, 평균 향상 폭이 +7.2/+3.2/+6.4점으로 나타나 기존 long-context RL 학습 세트를 능가했습니다. 특히 LBv2, AA-LCR, DocFinQA처럼 ‘holistic’에 가까운 추론형 벤치마크에서 개선이 두드러졌고, 모델 크기와 컨텍스트 변형에도 성능 이득이 비교적 일관되었습니다. 더불어 학습 컨텍스트를 넘어선 길이에서도 이득이 유지되는 패턴을 보여, 특정 길이에 과적합된 skill보다 길이-일반적 long-context reasoning 능력을 길렀다는 점이 의미 있습니다.



### Space Is Intelligence: Neural Semigroup Superposition for Riemannian Metric Generation (https://arxiv.org/abs/2606.18828)
- **Prior Approaches**: 대부분의 motion planning·제어는 에이전트(학습된 정책, A*·RRT 같은 검색, potential field, MPC, 강화학습)가 ‘지능’을 갖고, 장면 정보는 그 입력으로만 들어간다. 그래서 장애물 회피에는 충돌 체크·비용 설계 등 별도 절차가 필수이거나, 비용/제약을 과업마다 공학적으로 다뤄야 한다. 최근의 learned planner·diffusion·VLA 계열도 결국 scene→action 매핑 구조라 지능이 여전히 에이전트 쪽에 남는다.

- **Core Contribution**: 이 논문은 지능의 위치를 ‘공간 자체’로 옮긴다. 장면이 configuration manifold에 Riemannian metric tensor를 유도하고, 경로 계획은 그 metric의 geodesic를 푸는 문제로 축소되어 별도의 planner나 collision checker 없이도 장애물을 회피하도록 만든다. Encoder-Router가 scene-conditioned metric field를 만들고, geodesic solver가 이를 ‘수동 판독’해 행동(경로)을 얻는 구조가 핵심이다.

- **Technical Challenges**: metric을 만들되 항상 유효한 Riemannian metric(SPD)을 보장해야 하고, 장면 복잡성이 늘어도 파라미터·구조가 폭증하지 않게 해야 한다. 이를 위해 Lie algebra의 합과 exponential map exp: sym(2)→SPD(2)을 사용해 네트워크 출력이 곧바로 SPD metric으로 변환되도록 설계했으며, semigroup-superposition 규칙으로 장면 요소 수(K)가 늘어도 조합 법칙을 그대로 유지한다. 또한 단일 Encoder-Router에서 frame parameters(시점 정렬), modulation parameters(커널 전파·장애물 근방 장벽의 날카로움), basic coefficients(세기·부호·gate)를 함께 생성해 하나의 metric field로 합성한다.

- **Empirical Impact**: 2D 시뮬레이션에서 단 하나의 2-장애물 장면만 학습한 뒤, 장애물 개수·배치·밀도·패턴이 다른 12개 테스트 장면에 대해 zero-shot으로 성능을 보였다. 충돌-free 경로 비용과 obstacle-penetrating 경로 비용이 장면에 따라 3~5자릿수(orders-of-magnitude) 이상 분리되며, threshold 튜닝 없이도 metric만으로 충돌 가능성을 명확히 구분할 수 있다고 보고한다. 특히 hard top-k sparse gating을 쓰면 성능이 급락했는데, 이는 gate를 연속적으로 경쟁시키는 설계가 중요한 학습 신호임을 보여준다.



### Maturing Markov Decision Processes: Decision Making under Increasing Information and Shrinking Action Sets (https://arxiv.org/abs/2606.18820)
Comments:
          25 pages, 9 figures

- **Prior Approaches**: 기존 MDP/제약 MDP, 비정상·부분관측 변형들은 시간 의존성이나 행동 가능 마스크를 상태에 납작하게 편입해, 정보가 풍부해지는 과정과 행동 선택지의 만료(커트오프·비가역 제약)의 결합 구조를 놓치기 쉽습니다. POMDP·순환 메모리 기반 방법도 ‘늦게 더 보이는 정보’에는 초점을 두지만, ‘어떤 행동을 사라지기 전에 반드시 결정해야 하는지’라는 우선순위 문제는 명시적으로 다루기 어렵습니다. 그 결과 탐색–활용 균형이 단계별로 어떻게 달라지는지에 대한 학습 부담이 과대평가될 때가 많았습니다.

- **Core Contribution**: 이 논문은 정보는 단계가 진행될수록 정교해지고 행동 가능 집합은 점진적으로 축소되는 비대칭을 모델링하기 위해 Maturing Markov Decision Processes(MMDPs)를 제안합니다. 또한 expiring-action priority principle로 다음 단계 전환 전에 반드시 해결해야 할 ‘만료되는(expiring) 행동’의 부분을 이론적으로 규정합니다. 이를 바탕으로 stage-aware policy, expiring-action abstraction, search-augmented learning with distillation로 이어지는 구조-인식 강화학습 프레임워크를 제시합니다.

- **Technical Challenges**: 핵심 난제는 단계별로 달라지는 상태 표현과 행동 집합을 학습 파이프라인에 맞게 정렬하는 것입니다. 저자들은 (1) stage index를 조건으로 넣는 stage-aware 정책 설계로 단계별 관측·행동 차이를 반영하고, (2) 만료되는 행동만을 고정 인터페이스로 추려내는 expiring-action abstraction으로 탐색 공간을 단계 단위로 축소하며, (3) 탐색으로 얻은 개선 결정을 distillation로 정책에 주입해 학습 효율을 높였습니다. 특히 하이브리드/파라미터화된 행동은 에지·결정(예: 어떤 전송 경로를 쓸지)만 남기고 executor가 나머지를 채우는 방식으로 구현 난이도를 낮췄습니다.

- **Empirical Impact**: 실험에서는 다수 공급사 재고 보충, 간단 캐시관리(계정 수 증가), 그리고 생산 규모 시뮬레이터에서 flat MDP 대비 MMDP 모델링이 학습 효율과 최종 성능을 개선함을 확인했습니다. 특히 DQN·큰 전이(transfer) 네트워크처럼 탐색이 어려운 설정에서 MMDP의 이점이 확대되었고, search와 결합할 때 성능 격차가 더 커졌습니다. 또한 2025년 9월 한 달 백테스트에서 수익이 5.3% 개선, 배포 후 온라인 초기에 추천 채택률이 18.6% 증가하는 등 실제 운영 관점의 효용도 보고했습니다.



### SwitchBraidNet: Quantisation-Aware Lightweight Architecture for Hybrid Brain-Computer Interfac (https://arxiv.org/abs/2606.18816)
Comments:
          6 pages, 5 figures, Preprint accepted at IEEE SMC 2026

- **Prior Approaches**: 기존 MI-SSVEP 하이브리드 BCI(hBCI)는 두 신호의 보완성을 활용하지만, 주로 full-precision(FP32) 성능 중심으로 평가돼 임베디드 배포 제약(메모리·전력·연산)을 함께 만족시키기 어렵습니다. 또 모델 압축은 많게는 post-training quantisation(PTQ) 위주였고, low bit-width(INT8 이하)에서 성능 붕괴를 줄이기 위한 quantisation-aware training(QAT) 평가는 제한적이었습니다. 하이브리드에서 sequential/동시(simultaneous) 운용에 따른 ITR 변화도 이론·실험으로 일관되게 정리된 사례가 드뭅니다.

- **Core Contribution**: 이 논문은 저전력 배포를 목표로 한 경량 EEG 분류기 SwitchBraidNet을 제안합니다. 듀얼-path temporal braid(멀티스케일 진동 특징), squeeze-and-excitation 기반 electrode gating(전극 기여 스위칭), log-variance readout(밴드파워를 직접 인코딩)으로 MI와 SSVEP의 이질적 특성을 한 모델에서 다루도록 설계했습니다. 더불어 OpenBMI 데이터셋에서 QAT를 포함한 정량화( FP32/FP16/INT8) 벤치마크와 함께 MI·SSVEP·hBCI의 성능을 ITR 관점까지 체계적으로 비교합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) MI/SSVEP가 서로 다른 스펙트럼·공간·시간 부분공간을 갖는데 이를 단일 표현으로 통합해야 하고, (2) INT8 같은 저비트 양자화가 내부 활성의 동적 범위를 흔들어 성능을 크게 떨어뜨릴 수 있다는 점입니다. 저자들은 log-variance 계층에 BN을 포함해 동적 범위 안정성을 확보하고, hardsigmoid/ReLU6 같은 저비트 친화 활성과 SE 스위칭 구조로 양자화 실패 모드를 완화하도록 아키텍처를 “처음부터” 구성했습니다. 또한 OpenBMI에서 fake quantisation을 forward 패스에 삽입하는 end-to-end QAT로 INT8에서도 정확도 저하를 최소화했습니다.

- **Empirical Impact**: 실험에서 SwitchBraidNet은 MI 정확도 69.49%(FP16 최고), SSVEP 정확도는 EEGNet과 거의 동급(차이 0.17%), hBCI에서는 하이브리드 정보전달률 64.82 bits/min(FP16) 및 INT8 메모리 풋프린트 3.03 KB로 효율성을 입증했습니다. INT8에서 성능 저하 폭은 MI·SSVEP 모두에서 가장 작아(각각 0.21%, 0.08%) 저비트 환경에서도 강건함이 확인됐습니다. 또한 sequential과 simultaneous의 분류 품질은 동일하지만 ITR은 동시 모드가 더 높다는 점을 데이터로 검증해, 실제 시스템 설계에서 운용 모드 선택 기준을 제시합니다.



### Reinforcement Learning Foundation Models Should Already Be A Thing (https://arxiv.org/abs/2606.18812)
- **Prior Approaches**: 언어·비전 foundation model은 인터넷 규모 데이터로 학습되지만, tabular 예측·time-series·graph learning·reinforcement learning 같은 구조화 도메인은 같은 방식이 어렵다. 그래서 최근 in-context RL은 합성 데이터가 아니라 관측한 trajectory를 그대로 시퀀스 모델링해(예: Decision Transformer, Gato) 특정 환경군에 맞춘 적응에 가까웠고, “어떤 MDP 분포(prior)를 두면 전이 가능한가”를 핵심 목표로 다루지 않았다. TabPFN 계열은 Bayesian 분류기에 대한 합성 prior를 사전학습해 tabular 문제를 in-context로 푸는 데 성공했지만, RL에서는 동일한 prior 설계가 공백이었다.

- **Core Contribution**: 이 논문은 RL에도 foundation model급 학습을 가능케 하는 RL prior 설계 의제를 제시한다. 특히 강화학습에서는 synthetic MDP를 sampling하는 것이 synthetic tabular 데이터 sampling만큼 가능함에도, prior 설계를 1차 목표로 삼는 in-context RL 연구가 거의 없다는 점을 강조한다. 또한 작은 tabular MDP는 episode 길이와 무관하게 충분통계로 고정 크기 표현이 가능하며, 이를 attention 기반 tabular foundation model 스타일로 변환해 policy head를 붙이면 “RL foundation model”의 로드맵이 된다고 정리한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 RL 데이터를 trajectory가 아닌 “사전이 설계된 합성 통계”로 재형식화하면서, MDP 전반에 걸쳐 학습된 prior가 유의미한 inference를 제공하도록 만드는 것이다. 논문은 tabular MDP에 대해 visit counts, empirical mean reward, empirical transition row로 구성된 고정 크기 충분통계 행렬을 만들고, TabPFN 계열의 permutation-equivariant set transformer 친화 입력 형식을 채택한다. 모델은 이 통계를 입력받아 softmax(Q*/τ) 형태의(온도 τ로 부드럽게 한) 최적 정책을 직접 회귀하도록 학습하며, 가중치가 묶인 message-passing(전파 깊이 K는 추론 시 조절 가능)으로 planning을 흉내 낸다.

- **Empirical Impact**: 실증 결과, 합성 MDP로만 학습한 단일 모델이(task-specific tuning 없이) held-out tabular benchmark에서 in-context로 성능을 보이며, online·offline 모두에서 경쟁력을 확인했다. 온라인 설정에서는 UCB-VI와 tabular Q-learning보다 훨씬 적은 episode로 문제를 해결했고, offline에서는 VI-LCB와 비교해 경쟁적으로 동작했다. 논문은 RL에도 prior 설계를 first-class로 두는 foundation model 패러다임이 현실적인 경로임을 보여주며, 향후 연속 제어·대규모/부분관측·prior misspecification 강건성 같은 확장 과제도 구체화한다.



### Rescaling MLM-Head for Neural Sparse Retrieva (https://arxiv.org/abs/2606.18811)
- **Prior Approaches**: SPLADE 같은 learned sparse retrieval(LSR)은 BERT 스타일 masked language model(MLM) 백본의 MLM head 출력을 그대로 희소 어휘 표현으로 써서 검색 점수를 계산한다. 기존에는 백본을 최신 인코더로 바꿔도 성능이 오를 것이라는 ‘드롭인’ 기대가 있었지만, 실제로는 ModernBERT·Ettin 같은 큰 MLM-head L2 norm 백본이 기준선(BERT-SPLADE) 대비 크게 무너질 수 있다. 기존 연구들은 하드 네거티브 마이닝, 증류, 희소 정규화, 데이터/학습량 개선 등에 집중했지만, SPLADE 핵심 계산에서 MLM-head 스케일이 일으키는 실패 양상은 충분히 다루지 않았다.

- **Core Contribution**: 논문은 성능 붕괴의 원인을 ‘모델 용량’이 아니라 MLM head의 스케일 불일치로 지목한다. SPLADE는 query-문서의 희소 벡터를 unnormalized dot product로 비교하기 때문에 MLM-head 출력의 스케일이 점수와 대조 학습 동역학을 그대로 흔든다고 설명한다. 이를 해결하기 위해 학습 직전 MLM-head projection 행렬을 상수 k로 rescale하는 초기화 시점 보정(initialize-time correction)을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 학습 레시피를 바꾸지 않고도 pretrained 인코더의 MLM-head 스케일이 SPLADE의 점수/그라디언트 스케일과 맞지 않을 때의 불안정성을 제어하는 것이다. 논문은 FLOPS 정규화까지 포함된 SPLADE 목표에서 큰 MLM-head 스케일이 희소 활성과 점수 크기를 증폭시켜 대조 학습을 흔들고, حتی training collapse까지 유발할 수 있음을 실험적으로 연결한다. 해결책으로 k=16 같은 rescaling을 통해 학습 손실 스케일을 낮추고 학습을 매끄럽게 만들며, 아키텍처/목적함수/추가 파라미터 없이도 안정성과 성능을 복구한다.

- **Empirical Impact**: MS MARCO, TREC-DL, BEIR-13 등 in-domain·out-of-domain 벤치마크에서 큰-norm 백본(ModernBERT, Ettin)의 성능을 표준 레시피 하에 다시 경쟁력 있게 만든다. 예를 들어 BEIR-13 mean nDCG@10에서 ModernBERT는 상대 개선 폭이 215%, Ettin은 77% 수준이며, 일부 설정에서는 BERT-SPLADE 기준도 따라잡거나 능가한다. 또한 rescaling 효과가 단순한 범용 정규화가 아니라 ‘적정 스케일 범위’ 제어임을 보여주며, 과도한 down-scaling은 ALBERT·RoBERTa처럼 애초 norm이 작은 모델에서 성능 붕괴를 초래할 수 있음을 확인한다.



### Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards (https://arxiv.org/abs/2606.18810)
- **Prior Approaches**: RLVR의 대표 알고리즘인 GRPO 계열은 롤아웃마다 스칼라 보상을 받아 모든 토큰에 동일한 credit(advantage)을 분배해, 중요한 추론 토큰이 과소평가되고 일상 토큰에 불필요한 그래디언트가 낭비되는 문제가 있었다. 토큰 수준 credit assignment를 시도한 방법들도 별도의 process reward model(PRM)이나 ground-truth 정답, 또는 외부·특권 정보가 필요한 경우가 많아 ‘순수 RLVR’(바이너리 verifier만 존재) 적용성이 제한된다.

- **Core Contribution**: 이 논문은 ‘검증된(verified) 궤적’을 이용해 토큰별로 “검증 솔루션이 있을 때와 없을 때”의 분포 차이를 추정할 수 있다는 점에 주목하고, 그 차이를 활용한 SC-GRPO(Self-Conditioned GRPO)를 제안한다. 또한 verified trajectories를 교사 분포(knowledge distillation) 목표로 그대로 쓰는 형태는 여러 verified 궤적이 공존할 때 가중평균 해가 실제 가능한 궤적에 대응하지 못해 학습 신호가 깨질 수 있음을 이론적으로 보인다.

- **Technical Challenges**: 핵심 기술 난제는 ‘verified trajectories에는 추론 품질에 대한 특권 정보가 없고, 여러 성공 궤적이 서로 다른 다음 토큰을 제시할 수 있다’는 제약 하에서 토큰별 credit 신호를 안정적으로 만드는 것이다. 저자들은 self-conditioned teacher로부터 얻는 토큰별 KL divergence를 loss가 아니라 GRPO 그래디언트의 multiplicative weight로만 사용해, 보상은 update 방향을, KL은 update 강도만 조절하도록 설계를 바꿔 문제를 회피한다.

- **Empirical Impact**: 수학·코드·에이전트형 멀티턴 태스크 5개 벤치마크에서 SC-GRPO는 GRPO 대비 Average@8을 8.1%, DAPO 대비 5.9% 개선했으며 OOD 성능도 더 일관적이었다. 또한 외부 데모 기반 OPD/OPSD 계열이 기대만큼 오르지 못한 반면(SC 신호가 신뢰하기 어렵다는 시사), SC-GRPO는 외부 교사 없이도 더 높은 성능과 안정성을 보이며 RLVR 학습 패러다임에 실용적인 토큰 수준 credit assignment 경로를 제시했다.



### SHIFT: Semantic Harmonization via Index-side Feature Transformation for Multilingual Information Retrieva (https://arxiv.org/abs/2606.18801)
- **Prior Approaches**: 기존 MLIR 연구는 query translation이나 문서/코퍼스 MT 같은 CLIR식 우회로를 쓰거나, multilingual dense retriever를 추가 학습으로 언어 정렬을 강화하는 방식에 의존해 왔다. 그러나 학습 기반 방법은 비용이 크고, 파이프라인 번역 방식은 복잡도·지연을 키우며, 단순 정규화는 의미 신호를 과도하게 억누를 수 있다. 특히 최근 dense retrieval 모델은 쿼리와 같은 언어 문서를 과도하게 상위에 배치하는 language bias 문제를 여전히 크게 보인다.

- **Core Contribution**: SHIFT는 학습 없이(inference-time 추가 비용 없이) indexing 단계에서 문서 임베딩을 보정해 언어 편향을 줄이는 방법을 제안한다. mMARCO 같은 병렬 번역 페어로부터 소스 언어 대비 각 타겟 언어의 상대 language vector를 추정한 뒤, 문서 임베딩에서 이를 선형으로 빼서 표현 공간을 소스 언어 쪽으로 캘리브레이션한다. 또한 이 논문은 쿼리 언어 외 문서의 재현율을 직접 보는 Target-Languages Recall@k (TLR@k)로 편향을 명시적으로 정량화한다.

- **Technical Challenges**: 핵심 난제는 “의미는 같지만 언어 표면형 때문에 embedding 공간에서 멀어지는” 불정렬을 모델 재학습 없이 교정하는 것이다. SHIFT는 병렬 문서의 임베딩 차이를 평균해 언어별 오프셋을 추정하고, α로 보정 강도를 조절하면서도 쿼리 쪽 변환은 하지 않아 지연을 만들지 않는다. 더 나아가 α가 과도하면 특정 언어에 대한 over-shifting이 발생해 전체 균형이 흔들릴 수 있음을 실험적으로 분석해 조절 가능함을 보인다.

- **Empirical Impact**: 네 개의 MLIR 벤치마크와 다양한 dense retriever(encoder·decoder 계열)에서 SHIFT가 일관되게 성능과 TLR을 동시에 개선함을 확인했다. 특히 언어 편향이 숨겨지는 지표(Recall/nDCG)와 달리 TLR@20에서 큰 상승이 관측되며, 예컨대 multilingual-e5-large의 경우 타겟 언어 문서 노출이 크게 늘었다. 전반적으로 top-k 결과의 언어 분포가 더 고르게 퍼지며(정성 분석), 3가지 비영어 소스 언어 실험에서도 유사한 개선이 반복되어 범용성까지 입증한다.



### Closing the Loop: PID Feedback Control for Interpretable Activation Steering in Symbolic Music Generation (https://arxiv.org/abs/2606.18790)
Comments:
          Accepted at Learning to Listen: ICML 2026 Workshop on Machine Learning for Audio (43rd International Conference on Machine Learning - ICMLMLA26), 4 pages main (11 total), 2 figures

- **Prior Approaches**: Transformer 기반 activation steering은 재학습 없이 추론 시 내부 표현을 조절해 생성 결과를 바꾸지만, discrete 속성을 미세하고 해석 가능하게 컨트롤하는 데는 한계가 남아 있다. 기존 dense steering은 residual stream에서 pitch·duration 같은 개념이 선형 벡터로는 나타나도 서로 얽혀(feature superposition) 다중 속성 동시 조절 시 간섭이 커진다. SAS(Sparse Activation Steering)는 SAEs의 Top-K 희소화로 얽힘을 줄이지만, Top-K 임계값 때문에 steering 강도를 서서히 올려도 신호가 0으로 사라졌다가 갑자기 켜지는 ‘이진 문턱’ 문제가 생긴다.

- **Core Contribution**: 이 논문은 MMT(Multitrack Music Transformer)에서 pitch와 duration 같은 속성을 잔차 스트림/SAE 표현 안에서 mechanistic하게 분해하고, 재학습 없이 inference-time에서 deterministic하게 조절하는 프레임워크를 제안한다. 핵심 아이디어는 활성 steering을 ‘Proportional-Integral-Derivative(PID)’ 제어로 만들되, SAS의 Top-K 문턱을 넘기도록 steering의 시간 축을 따라 λ(t)를 동적으로 설계하는 Temporal PID다. 또한 다중 속성 동시 제어 시에는 Gram-Schmidt Orthogonalization을 결합한 Dual Steering으로 기하적 디커플링을 수행한다.

- **Technical Challenges**: Temporal 영역으로 PID를 옮기는 과정에서 가장 큰 난관은 Top-K 재희소화가 신호 크기가 작을 때 아예 살아남지 못해, 단순한 cosine ramp가 smoothing을 망가뜨린다는 점이다. 이를 해결하기 위해 Temporal PID는 ‘개념 fingerprint’로 Top-N 타깃 특징의 평균 크기를 측정해 연속적인 오차 신호를 만들고, I항이 누적 오차를 통해 문턱을 넘을 때까지 λ(t)를 점진적으로 증폭한다. pitch와 duration이 SAE 공간에서 여전히 경쟁할 수 있으므로 Dual Steering에서 Gram-Schmidt로 직교화하고, 필요한 경우 Top-K 예산(2×K)을 확장해 두 속성의 개념 간 간섭을 줄인다.

- **Empirical Impact**: 실험 결과 Temporal PID는 static SAS 대비 intervention(개입량)을 62–67% 줄이면서도 pitch·duration 조절 품질 저하(FMD 관련)를 개선해, 특히 pitch에서 5% 낮은 FMD degradation을 보인다. 또한 조건된 프리픽스에 대해 ‘되돌리기’(round-trip)처럼 단계별로 steer-in/steer-out을 수행할 수 있어, 고정 λ를 쓰는 static 방법으로는 어려운 회복 궤적(46–74% recovery)을 입증한다. 전반적으로 MMT+SOD 단일 세팅에서의 검증이지만, Top-K 문턱 문제를 폐루프 제어로 실질적으로 완화했다는 점에서 controllable music generation과 activation steering 연구에 의미 있는 진전을 제공한다.



### Bayesian Anytime Pareto Set Identification for Multi-Objective Multi-Armed Bandits (https://arxiv.org/abs/2606.18785)
Comments:
          26 pages, 13 figures

- **Prior Approaches**: 기존 MOMAB PSI(Parato Set Identification)는 regret이나 샘플 수를 미리 정한 fixed-budget, 또는 정해진 신뢰도 아래에서 종료하는 fixed-confidence 중심으로 발전해 왔습니다. 또 Gaussian process 등 구조를 가정한 변형이나(고유-디스크립터 연관) EGE(Empirical Gap Elimination) 계열처럼 다중 목표에서 단계적으로 후보를 줄이는 방식이 주로 쓰였습니다. 다만 anytime(매 관측마다 갱신하며 언제든 멈출 수 있는) 설정은 베이지안 기반으로 충분히 다뤄지지 않아, 지속적으로 Pareto 집합을 내놓아야 하는 실전 의사결정엔 공백이 남아 있었습니다.

- **Core Contribution**: 논문은 MOMAB PSI의 첫 anytime 베이지안 알고리즘으로 Top-Two Pareto Front Thompson Sampling(TTPFTS)을 제안합니다. 단일 best-arm과 달리 선호 없이 여러 Pareto 최적해가 존재하므로, 현재 추정 Pareto front(1번째)와 그 다음 front(2번째) 사이에서 샘플을 확률적으로 배분해 전체 Pareto set을 점진적으로 식별합니다. 또한 분자 탐색 같은 실제 multi-objective 상황에서 합성-on-demand 라이브러리를 효율적으로 훑는 방식의 활용 시나리오를 함께 제시합니다.

- **Technical Challenges**: 핵심 기술 난점은 ‘정답 Pareto front를 모르는 상태에서’ 매 시점마다 최적/차선 후보를 구분할 만큼 정보를 모으는 anytime 탐색-정렬(balance)을 설계하는 것입니다. TTPFTS는 각 arm의 기대값에 대한 posterior 샘플을 뽑은 뒤, Top-two Pareto front를 기준으로 균등 랜덤 선택을 통해 경계(front) 쪽에서 검증을 강화합니다. 추가로, ground-truth 없이도 현재 Pareto 예측의 불확실도를 추정하는 정량 지표를 설계해 학습 진행 모니터링과 동적 종료 의사결정을 가능하게 했습니다.

- **Empirical Impact**: 합성 벤치마크 8종에서 TTPFTS는 uniform sampling은 물론 fixed-budget SOTA인 EGE-SH를 일관되게 능가하며, 제한된 구조적 페널티가 있는 anytime 조건에서도 EGE-SR와 경쟁력 있는 성능을 보였습니다. 특히 arm 수가 많은 EgeExp3 계열에서는 posterior 기반의 샘플 배분이 gap 기반 조기 제거에 비해 더 빠르게 수렴하며 큰 성능 격차를 보였습니다. 분자 합성-on-demand 라이브러리 탐색에서는 exhaustive virtual screening 대비 효율을 확보하고, 새 불확실도 지표가 ground-truth 성능(Jaccard)과 강한 상관을 가져 실제 의사결정용 모니터링 도구로도 의미가 큽니다.



### RedactionBench (https://arxiv.org/abs/2606.18782)
- **Prior Approaches**: 기존 PII 마스킹 평가는 개인정보 ‘추출’ 방식과 ‘프라이버시 의미’를 섞어버려, 문맥에 따라 달라지는 위반 여부를 제대로 구분하지 못했습니다. 예를 들어 같은 전화번호라도 의료기록에 있는지, 공개된 번호인지에 따라 문제의 성격이 달라질 수 있지만, 대부분의 벤치마크는 이를 동일 취급합니다. 그 결과 redaction(가림)과 entity recognition(개체 인식)을 사실상 같은 문제로 취급하는 한계가 드러납니다.

- **Core Contribution**: 논문은 contextual integrity(문맥적 무결성) 관점에서 ‘무엇을 가릴지’가 사용 주체·목적·상황에 따라 달라진다는 점을 전면에 내세워, 이를 평가에 반영하는 RedactionBench를 제안합니다. RedactionBench는 11개 도메인에서 200개 문서를 수작업 라벨링했으며, 대부분은 실제 출처에서 시드되어 현실성을 높였습니다. 또한 문맥적 redaction의 난제를 정량화하기 위한 R-Score를 함께 도입합니다.

- **Technical Challenges**: 핵심 기술적 난제는 문맥 redaction이 본질적으로 주관적이라 모델 평가가 ‘정답 정밀도’만으로는 해결되지 않는다는 점입니다. 이를 위해 R-Score는 의미적으로 유사한 redaction은 동일하게 취급하고, 휴대전화 번호 마스킹처럼 얕은 포맷 차이(마스킹 스타일 차이)는 점수에서 무력화합니다. 그럼에도 다양한 모델군(NER, small language models, 에이전틱 도구가 달린 프론티어 모델)에서 contextual redaction이 여전히 미해결임을 보여줍니다.

- **Empirical Impact**: 실험은 여러 모델 35종을 대상으로 PII redaction 성능을 비교하고, 평가 결과가 모델 패밀리 간에도 크게 갈리며 contextual redaction이 특히 어렵다는 점을 확인했습니다. 또한 RedactionBench에서 80명 이상 사용자 기반의 인간 평가를 수행했는데, 필수 redaction에 대한 합의(89.4%)와 안전 텍스트 보존(94.1%)은 높지만 문맥 redaction 합의는 47.7%로 크게 낮았습니다. 이 ‘인식 격차’를 근거로 R-Score가 contextual ambiguity와 strict precision을 분리해 측정하도록 설계된 의미가 입증되며, RedactionBench는 향후 프라이버시 보존 시스템의 표준 기준선(baseline)으로 활용될 전망입니다.



### Private Learning with Public Feature Conditioning (https://arxiv.org/abs/2606.18773)
Comments:
          Proceedings of the 43rd International Conference on Machine Learning (ICML 2026). 26 pages, 9 figures

- **Prior Approaches**: 기존 label differential privacy(Label DP) 연구는 분류에 집중돼 회귀로의 확장이 제한적이었고, 연속 라벨을 다루는 방법은 상대적으로 적었습니다. 일부 회귀 Label DP 접근은 public feature 구조를 무시하거나(RR 기반/feature-oblivious), 특정 선형(또는 제한된) 모델 형태에만 적용되며 고프라이버시 구간에서 성능이 쉽게 무너진다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 DPSGD에 조건부(conditioning) 연산을 결합한 Cond-DP를 제안해, labels를 함께 쓰지 않으면서도 public feature의 기하를 최적화에 반영합니다. public feature 행렬의 스펙트럼이 빠르게 감쇠하는 현상을 활용해, 최적화 과정에서 작은 고유방향(low-spectrum directions)의 신호대잡음비를 개선하도록 매개변수 공간을 재조정합니다.

- **Technical Challenges**: 핵심 기술 과제는 “조건부로 최적화를 바꾸되, privacy 비용은 늘리지 않고” DPSGD의 이론적 보장(수렴/utility)을 유지하는 것입니다. Cond-DP는 공용 정보로만 계산되는 conditioning matrix C를 고정 하이퍼파라미터로 두고, convex/strongly convex/non-convex 목적함수에 대해 각각 수렴 보장과 함께 Cond-DP의 프라이버시를 정리했으며, conditioning이 항등행렬 I일 때 기존 DPSGD로 정확히 회귀함을 보입니다.

- **Empirical Impact**: public features로부터 구성한 conditioning matrix를 적용하면, private linear regression에서 DPSGD 대비 provably 더 빠른 수렴을 보였고 추가 privacy cost 없이도 이득이 발생합니다. 또한 실험 전반에서 Cond-DP는 다양한 데이터셋과 모델 아키텍처에서 state-of-the-art regression Label DP baseline을 일관되게 능가했으며, MLP 등 비선형 헤드에서는 초반엔 가속, 후반엔 저해가 나타나 이를 완화하는 Switch-Cond-DP까지 제안합니다.



### Generating Natural and Expressive Robot Gestures through Iterative Reinforcement Learning with Human Feedback using LLMs (https://arxiv.org/abs/2606.18747)
Comments:
          8 Pages, 6 Figures

- **Prior Approaches**: 기존 소셜 로봇 제스처 생성은 전문가가 만든 애니메이션에 의존하거나, 기능적으로는 측정 가능하지만 자연스러움 같은 주관적 지표를 잘 반영하지 못했다. 또한 GenAI로 로봇 코드를 뽑더라도 자유도가 커질수록 “자연스러움”을 놓치며, 제스처가 비결정적이라 객관적 평가가 어렵다.

- **Core Contribution**: 이 논문은 대화에 맞춘 co-speech gesture를 Pepper에서 생성하기 위해 GPT-4.1을 결합하되, 단순 LLM 생성만으로는 뻣뻣함이 남는 문제를 RLHF로 해결한다. 인간 사용자의 영상 평가를 반복적으로 수집해 제스처 생성 코드(저수준 primitive 함수 시퀀스)를 파인튜닝함으로써, 발화와 동기화된 더 표현력 있고 유연한 제스처를 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) LLM이 생성한 모션 코드가 Pepper 관절 한계와 실행 가능성을 만족해야 하고, (2) 자연스러움·부드러움처럼 주관적 선호를 학습 신호로 바꾸는 것이며, (3) 제스처의 비결정성과 다양성 유지가 동시에 필요하다는 점이다. 이를 위해 관절 제약을 반영한 low-level motion primitives 라이브러리와 타이밍 규칙을 프롬프트에 강제하고, 사용자 선호는 offline DPO와 online few-shot 대화이력 요약을 이용해 반복적으로 모델에 주입한다.

- **Empirical Impact**: Pepper 영상을 대상으로 한 5회 온라인 사용자 스터디에서, 마지막 iteration은 첫 iteration 대비 expressiveness·relevance·fluidity 전 항목이 통계적으로 유의하게 개선되었다(p<0.001). 특히 관련성·유연성의 상승 폭이 더 컸고, 예: apology 같은 의미가 명확한 범주에서 개선이 두드러지며, few-shot 프롬프트가 전체 성능 향상에 크게 기여했다는 ablation 결과도 제시된다.



### SWE-Future: Forecast-Conditioned Data Synthesis for Future-Oriented Software Engineering Agents (https://arxiv.org/abs/2606.18733)
- **Prior Approaches**: 기존 코딩 에이전트 벤치마크(SWE-Bench 계열 등)는 실제 GitHub issue/PR을 그대로 가져와 현실성을 확보하지만, 공개 저장소와 평가 흔적이 사전학습·fine-tuning·합성데이터 생성·모델 선택 루프에 재유입될 수 있어 데이터 오염 위험이 커집니다. 순수 합성은 이런 재사용을 줄이지만, 저장소가 실제로 요구하는 관례·의존성·테스트 제약을 놓치기 쉽습니다.

- **Core Contribution**: 이 논문은 SWE-Future로, 미래 지향적인 코딩 작업을 만들기 위한 “forecast-conditioned data synthesis”를 제안합니다. 핵심 아이디어는 시점 T0의 저장소 증거만으로 향후 구현/버그수정/리팩터 방향을 task family로 예측하고, 이후 검증은 post-T0 PR로 ‘재현(replay)’하지 않고 가족 일치 여부만 평가한 뒤, task-generation snapshot(Tgen)에서 실제 과제를 합성하는 것입니다.

- **Technical Challenges**: 문제는 (1) 라벨·트래커 용어 같은 잡음이 섞인 pre-T0 증거로부터 신뢰도 있는 task family를 뽑는 것과, (2) 예측된 방향을 Tgen 코드베이스에서 실행 가능한 오라클(target-test)로 구체화하는 것입니다. 저자들은 증거를 규칙 기반 필터링→anchor 추출→클러스터링→휴리스틱 점수화로 family를 생성하고, 회고 검증에서 semantic matching으로 strong/related 일치를 고정 평가한 뒤, 다중 에이전트 워크플로로 issue 스타일 요청과 테스트 패치·gold 패치를 생성해 FAIL_TO_PASS 기반 실행 검증을 통과한 작업만 공개합니다.

- **Empirical Impact**: 80개 저장소에서 forecaster는 주요 semantic 매칭 지표 기준 future-work relevance 58.1%(151/260)을 달성했고, strong hit rate는 42.7%(111/260)로 보고됩니다. 또한 검증 통과 family를 조건 신호로 삼아 61개 저장소에서 200개의 실행 가능 코딩 에이전트 작업 세트를 만들었으며, 이는 later PR을 과제 재료로 쓰지 않고도 저장소 진화 맥락을 반영한 합성의 타당성을 보여준다는 점에서 의미가 큽니다.



### Two-Phase Bilevel Search for the Moving-Target Traveling Salesman Problem with Moving Obstacles (https://arxiv.org/abs/2606.18730)
- **Prior Approaches**: 기존 MT-TSP 연구는 목표가 움직이는 상황에서 시간 창(time window) 제약을 만족하며 최적 경로를 찾는 데 집중했지만, 주로 선형/조각선형 궤적이나 최적해에 대한 강한 보장 형태로 제한되는 경우가 많았습니다. 장애물까지 포함한 MT-TSP-O는 정적 장애물에서는 완전성/부분 보장 알고리즘이 제시됐으나, MT-TSP-MO처럼 ‘장애물도 움직이는’ 일반화에 대한 연구는 매우 드뭅니다. 또한 알려진 대표 접근이 소수에 그치며, 본 문제에서는 고품질 feasible 해를 빠르게 얻는 전략이 핵심 난제로 남아 있었습니다.

- **Core Contribution**: 이 논문은 Moving-Target Traveling Salesman Problem with Moving Obstacles(MT-TSP-MO)를 위해 Mixed-Integer Conic Programming(MICP) 정식을 제안합니다. 이 정식은 off-the-shelf 솔버로 풀 수 있고, 시간 이산화 기준에서 optimality/completeness에 대한 점근적 보장을 제공합니다. 동시에 큰 규모에서는 MICP의 확장성이 떨어진다는 점을 보완하기 위해 Two-Phase Bilevel Search(TPBS) 알고리즘을 개발하여, 최적성/완전성 보장은 낮더라도 더 신뢰도 높은 feasible 해와 낮은 비용을 빠르게 산출하도록 했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 시간 창 제약과 장애물 회피를 동시에 만족해야 하는 연속시간 충돌 판정, 그리고 속도 제한까지 결합되는 것입니다. MICP에서는 시간 축을 이산화하고, 목표/정차 방문을 이진 변수로 모델링한 뒤, 장애물의 이동에 따른 반공간 기반 조건과 big-M 제약으로 충돌 회피를 포함합니다. TPBS에서는 먼저 장애물이 없는 조건에서 GTSP(Generalized Traveling Salesman Problem)를 풀어 후보 투어를 만든 뒤, 각 간선에 대해 직선 기반 빠른 충돌 체크로 불충분할 때만 time-expanded graph에서 DFS로 collision-free, time-feasible 경로를 검증하는 2단계 구조로 계산량을 관리합니다.

- **Empirical Impact**: 실험은 최대 40개의 targets와 40개의 moving obstacles까지 포함하는 폭넓은 인스턴스(시간 창 길이 다양)를 대상으로, 기존 baseline 알고리즘과 비교해 success rate, solution cost, 계산 시간을 종합적으로 개선했음을 보여줍니다. 특히 MICP는 보장에 가까운 성질을 유지하면서도 실용적으로 작동하고, TPBS는 큰 문제에서 더 빠르게 고품질 feasible 해를 제공하는 쪽에 강점을 드러냅니다. 결과적으로 MT-TSP-MO 분야에서 ‘움직이는 장애물까지 고려한 실행 가능한 경로 계획’을 확장하는 실증적 기준을 제시한 것으로 평가됩니다.



### Graph Grounded Cross Attention Transformer Neural Network for Structurally Constrained Full Event Sequence Generation in Predictive Process Monitoring (https://arxiv.org/abs/2606.18726)
Comments:
          40 pages

- **Prior Approaches**: 예측 프로세스 모니터링(PPM)에서 기존 연구는 주로 다음 활동, 남은 시간, 결과(outcome), 속성(attribute) 같은 개별 구성요소 예측에 집중해 왔습니다. 그 결과 전체 이벤트 시퀀스를 “구조적으로 제약된 상태로” 동시에 생성하는 문제(전이 가능성, 시간 순서, 종료, 속성 일관성)를 충분히 통합해 다루기 어렵다는 한계가 있습니다. 또한 생성 경로의 feasibility를 보장하기 위한 제약 반영이 약하거나, 자동회귀 기반이라 오류 누적과 불완전한 종료 처리 문제가 남았습니다.

- **Core Contribution**: 논문은 unified PPM 태스크인 “전체 이벤트 시퀀스 생성”을 겨냥해 Graph Grounded Cross Attention Transformer Neural Network(GGATN)를 제안합니다. GGATN은 전체 프로세스 그래프를 구조화된 activity memory로 두고, Transformer self attention으로 위치 맥락을 잡은 뒤 graph grounded cross attention으로 프로세스 토폴로지 제약을 주입합니다. 특히 단일 패스에서 활동, timestamp, 길이, 이벤트/시퀀스 속성을 생성하고, 이후 Viterbi 스타일의 그래프 제약 디코딩과 명시적 종료로 feasible 경로를 보완합니다.

- **Technical Challenges**: 핵심 기술 과제는 생성물이 전이 가능성(feasibility), 시간적 순서, 정확한 종료, 속성 일관성을 동시에 만족해야 하는데, 이를 end-to-end로 일관되게 강제하는 것입니다. 논문은 자동회귀처럼 단계마다 이전 예측에 의존해 오류가 누적되는 방식 대신, activities·timestamps·length·attributes를 한 번에 생성한 뒤 그래프 기반 제약 디코딩으로 경로 가능성을 재정렬합니다. 또한 termination을 명시적으로 모델링해 “끝나지 않는” 출력 문제를 줄이고, 구조적 prior로서 글로벌 그래프 인코더가 안정적으로 작동하도록 설계했습니다.

- **Empirical Impact**: 6개 벤치마크 이벤트 로그 실험에서 GGATN은 local instruction으로 유도한 LLM 기반 비교군보다 생성 품질이 더 안정적이라고 보고합니다. 특히 sequence similarity, Damerau Levenshtein similarity, bigram 기반 control flow 유사도, duration 분포에서 강한 성능을 보이며, zero hallucinated activities와 zero sequence level attribute inconsistency를 유지합니다. ablation 및 interpretability 분석에서도 글로벌 그래프 인코더의 역할과 그래프 구조·시퀀스 컨텍스트·피드백 refinement·제약 디코딩이 생성 품질을 형성함을 확인해, PPM의 unified 시퀀스 생성 방향에 실질적 의미를 제공합니다.



### Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish (https://arxiv.org/abs/2606.18717)
- **Prior Approaches**: BPE, WordPiece, Unigram 같은 frequency-driven subword 토크나이저는 튀르키예어에서 접미사가 의미를 담는 구조를 무시해 과도 분절(fertility 증가)과 정렬 손실을 만든다. 특히 WordPiece는 diacritics를 제거하고 TurkishTokenizer는 canonical re-harmonization으로 문자열을 바꿔 decode(encode(w))=w를 보장하지 않는 경우가 생긴다. Morfessor나 Zemberek 같은 기존 형태소 분할·분석은 정렬 측면은 돕지만, 언어모델 생성에 필요한 가역성(생성 시 복원 무결성)과 결합해 모두를 만족하긴 어렵다.

- **Core Contribution**: 논문은 튀르키예어에 대해 무손실(정규화 없이 표면 문자열 보존), 형태소 인지, 그리고 임베딩 생성까지 한 번에 수행하는 신경 형태소 경계 모델 Morpheus를 제안한다. Morpheus는 각 문자 사이 경계 확률을 학습하면서도 추론 시에는 exact 분할을 산출하고, 정규화가 없어서 decode(encode(w))=w가 구조적으로 성립한다. 또한 토크나이징을 위한 동일한 forward pass에서 단어 단위의 structured word embedding까지 함께 출력한다.

- **Technical Challenges**: 핵심 기술 과제는 경계 확률(연속값)을 이용해 형태소 분할을 “학습 가능하게” 만들되, argmax/threshold 같은 비분화 연산 없이 discrete 형태소를 복원하는 것이다. Morpheus는 Poisson–binomial 동적계획법으로 per-position boundary 확률로부터 soft morpheme membership을 미분 가능하게 계산하고, 학습에서는 soft 할당으로 그라디언트가 흐르게 하며 추론에서는 경계가 hard하게 복원되도록 설계했다. 더불어 형태소 역할이 root 기준 상대 위치에 좌우된다는 점을 반영해 RoPE로 상대 오프셋을 attention에 주입하고, 같은 forward pass에서 segment pooling으로 임베딩을 만든다.

- **Empirical Impact**: 실험에서는 가역성/무손실성 축을 명확히 분리해 평가했는데, Morpheus는 bits-per-character(BPC) 최저(1.425)로 reversible 토크나이저 중 성능이 가장 좋았고, subword 계열 대비 gold morphological alignment도 크게 향상(MorphScore macro-F1 0.61 vs 약 0.32)됐다. 임베딩 평가에서도 frozen Morpheus 벡터가 root-family 검색에서 MAP 0.85, same-root verification에서 ROC-AUC 1.00으로 상위권을 보였으며, BGE-M3와 BERTurk를 초과한다. 다만 NER나 case/number probing처럼 문맥·굴절 의존 과제에서는 BERTurk류 contextual encoder가 여전히 우세해, 논문은 Morpheus의 root-centric geometry가 강점과 한계를 동시에 만든다고 해석한다.



### TW-LegalBench: Measuring Taiwanese Legal Understanding (https://arxiv.org/abs/2606.18699)
Comments:
          10 pages, 2 figures, To appear in ICAIL 2026

- **Prior Approaches**: 기존 법률 벤치마크는 주로 common-law권 데이터에 치우치거나(MMLU, LegalBench) civil-law라도 번체/간체 중국어권을 일부만 커버해(예: LawBench, LawShift) 대만처럼 특정 관할의 법리까지 정밀 평가하기가 어려웠습니다. 또한 다수의 벤치마크가 객관식 위주로 구성돼 실제 변론·논증 같은 개방형 과정을 반영하지 못하고, 법 조항 범주도 지나치게 뭉뚱그려 오차 원인을 세분화하기 힘든 한계가 있었습니다.

본 논문은 TW-LegalBench가 이런 “관할 불균형·세분화 부족·개방형 평가 공백”을 동시에 겨냥한다고 설명합니다.

- **Core Contribution**: TW-LegalBench는 대만의 민법계 전통에 맞춘 공개 공식 코퍼스 기반 벤치마크로, 대만 법률 추론을 평가할 수 있는 실데이터 3종(MCQ/OEQ/LJP)을 한 프레임에 모읍니다. 특히 객관식은 조항(법규) 단위로 43개 법 유형을 라벨링하고, 개방형 에세이는 공식 채점 루브릭을 LLM-as-Judge로 분해 평가하며, 형사 판결은 범죄 유형별 결과 예측(LJP)으로 현실 정렬을 점검합니다.

즉, “정확한 조항 식별-논증 구성-판결 결과 추정”의 단계적 역량을 관할 특이적으로 측정하려는 것이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 (1) 대만 법조문을 정확히 연결해 평가할 수 있는 고해상도 라벨링(조항 계층: 條/項/款/目)과 (2) 개방형 답안의 채점 변동성을 줄이면서 루브릭에 기반한 공정한 점수화를 동시에 달성하는 것입니다. 논문은 조항 단위 수작업 어노테이션과 함께 OEQ는 루브릭 포인트별로 이산 라벨/점수를 출력하는 decomposed LLM-as-Judge 방식을 쓰고, 두 저지(gpt-5, claude-sonnet-4.5)의 평균으로 편향을 완화합니다.

또한 LJP에서는 양형·형량 수치 평가를 위해 정규화 로그거리(징역)와 절대/상대 허용 오차(구금·벌금·집행유예)를 조합해, 단순 텍스트 유사도만으로는 포착되지 않는 실무 기준을 반영하도록 설계했습니다.

- **Empirical Impact**: 13개 LLM을 평가한 결과, 자격시험(변호사 1차·자격) 기준을 넘는 모델은 있었지만(합격률 11%), 판사·검사 수준에서는 크게 떨어져(합격률 1~2%) “자격 단계별 역량 격차”가 드러났습니다. LJP에서는 평결 유형과 형량 예측 성능은 어느 정도 보였으나, 정확한 조항(Article) 인용에는 취약해 Type I/II 오류 양상이 확인되며 신뢰도 높은 법문 생성이 여전히 어렵다는 결론으로 이어집니다.

특히 전통적으로 대만/중국어 특화 코퍼스를 더 학습한 모델이 조항 인용 정확도에서 상대적 강점을 보였지만, OEQ·MCQ 성능 패턴을 함께 볼 때 “법리 추론”보다 학습 데이터의 암기 효과 가능성도 제기됩니다.



### Leveraging Energy Features for Surface Classification with Deep Learning: A Comparative Analysis Across Three Independent Datasets (https://arxiv.org/abs/2606.18698)
- **Prior Approaches**: 모바일 로보틱스의 표면(surface) 분류에서는 일반적으로 관성(inertial) 센서 기반 접근이 중심이었고, 에너지 기반(energy-based) 특징은 제한된 환경에서만 일부 성과가 보고돼 상대적으로 덜 연구된 편입니다. 기존 연구들은 에너지 기반 특징의 단독 활용 가능성이나 관성 데이터와의 보완 효과를 체계적으로 검증하지 못한 경우가 많았습니다.

- **Core Contribution**: 이 논문은 에너지로부터 도출한 특징이 표면 분류에서 단독 입력으로도 작동하는지, 혹은 inertial 데이터에 보조 입력으로 넣을 때 성능이 얼마나 오르는지를 실증적으로 평가합니다. 또한 다양한 최신 딥러닝 분류기(RNN, CNN, encoder-only transformer, Mamba state-space model)를 대상으로 자동 하이퍼파라미터 튜닝과 입력 시퀀스 길이 최적화를 함께 수행해 비교의 공정성을 높였습니다.

- **Technical Challenges**: 핵심 기술 난제는 에너지 기반 특징이 얼마나 정보성이 있는지(단독 정확도)와, 관성 신호와 결합했을 때 유효한 보완 관계가 성립하는지(증분 이득)를 찾는 데 있습니다. 저자들은 여러 시퀀스 길이와 하이퍼파라미터를 자동으로 탐색해 과적합 및 입력 설계 편향을 줄이면서, 에너지 단독 vs 관성 단독 vs 에너지+관성의 성능 차이를 안정적으로 추정했습니다.

- **Empirical Impact**: 3개의 공개 데이터셋에서 이전 보고 대비 더 높은 정확도를 달성했으며, 전체 성능은 CNN이 최우수였습니다. 에너지 특징만 사용할 경우 85-90% 정확도를 보였고 이는 관성 결합 시 96-99%에 비해 약 5-10% 낮지만, 관성에 에너지 특징을 추가하면 평균 1-2%의 일관된 정확도 향상이 관측되어 실사용 관점의 선택지를 제공합니다.



### Dual-Channel Grounded World Modeling (DCGWM): Structural Prevention of Objective Interference Collapse via Heterogeneous External Grounding with Inward-Only Gradient Flow (https://arxiv.org/abs/2606.18688)
Comments:
          Position paper. Experimental validation in progress

- **Prior Approaches**: Joint Embedding Predictive Architectures (JEPAs)는 세계 모델의 표상 학습에서 널리 쓰이는 방식으로, 예측을 위해 공통 잠재공간에 여러 신호를 함께 학습시키는 경향이 있다. 특히 외부 신호를 물리 동역학(희소하고 큰 보정, 제약-만족형 그라디언트)과 사회·행동 동역학(분포 매칭형, 확산된 보정)처럼 서로 성격이 다른 신호로 “grounding”하는 시도들이 있었다. 하지만 이 논문은 공통 잠재공간에서의 joint learning이 두 채널 간 경쟁을 유발해, 손실 가중치로는 근본 문제가 해결되지 않는 실패 모드를 지목한다.

- **Core Contribution**: 논문은 Objective Interference Collapse (OIC)라는 실패 모드를 제안하며, 공통 잠재공간에서 지배 채널이 부차 채널의 표현 부분공간을 체계적으로 붕괴시켜 버린다고 주장한다. 이를 구조적으로 막기 위해 Dual-Channel Grounded World Modeling (DCGWM) 아키텍처를 제안한다. DCGWM은 물리 전용 잠재공간 Z_p와 행동 전용 잠재공간 Z_b를 분할하고, 서브스페이스 간 cross-subspace gradient 없이 “task level”에서만 결합하는 inward-only gradient flow를 도입한다.

- **Technical Challenges**: 핵심 기술적 어려움은 두 종류의 grounded 신호가 하나의 잠재공간에서 학습될 때 발생하는 그라디언트 간 간섭 경로를 차단하는 것이다. DCGWM은 Physical Grounding Channel이 VICReg-style 정렬로 물리 측정에만 Z_p를 업데이트하고, Social-Behavioral Grounding Channel은 emergent multi-agent simulation의 궤적에만 정렬해 Z_b만 업데이트하도록 경로를 분리한다. 또한 Inter-Channel Interface Module로 상호작용을 태스크 수준에서만 수행하고, 물리 위반은 hard hinge로 강하게 패널티하며 행동 이탈은 soft KL로 유도하는 Asymmetric Grounding Adherence Loss를 사용하며, 생성 렌더링 레이어를 latent world model에서 구조적으로 격리해 생성 목적함수의 기하학적 가정 하에 필요한 조건을 만족시키도록 설계한다.

- **Empirical Impact**: 이 논문은 OIC와 그 해결 방향의 타당성을 뒷받침하기 위해 세 가지 이론적 결과를 제공하며, 잠재공간 분할이 그라디언트 간섭 경로를 제거하고 각 grounded 서브스페이스가 anti-collapse 성질을 갖게 된다고 정리한다. 다만 제시된 실험 결과는 “진행 중”이며, 구체적 성능 검증은 향후 리비전에 보고될 예정이다. 그럼에도 문제 정식화와 DCGWM 같은 구조적 처방은 world model representation learning에서 grounded 신호의 결합 설계를 재검토하게 만드는 의미가 있다.



### Bounded Context Management for Tabular Foundation Models on Stream Learning (https://arxiv.org/abs/2606.18677)
Comments:
          Accepted as a spotlight oral (top 5%) at the 2nd ICML Workshop on Foundation Models for Structured Data (FMSD@ICML2026)

- **Prior Approaches**: 탭ular stream learning은 데이터가 순차적으로 도착하고 분포가 바뀌는 상황에서, 제한된 메모리로 실시간 예측을 수행하는 문제다. 기존 방법들은 online/incremental하게 모델 상태(트리 통계, ensemble 멤버 등)를 업데이트해 적응한다. 한편 tabular foundation model(TFM)은 labeled context를 바탕으로 in-context 방식으로 예측해, 적응은 결국 context를 어떻게 유지하느냐로 이동하지만 DualFIFO처럼 FIFO 기반 유지 외에 “무엇을 남길지”의 기준이 불명확했다.

- **Core Contribution**: 논문은 bounded context management를 ‘near-future 쿼리에 주는 정보량’ 관점에서 정식화해, context 유지의 3가지 요구조건(최근성, 불확실성, 중복 제거)을 제안한다. 이를 바탕으로 CURE(Context management via Uncertainty-aware admission and Redundancy aware Eviction)라는 정책을 설계해, short bank로 최근 예시를 확보하고 long bank에서는 entropy-gated admission으로 유용한 후보만 들이며 redundancy-aware eviction으로 가까운 동일 클래스 예시를 제거한다. 즉, TFM이 stream에서 잘 동작하려면 모델 업데이트가 아니라 context 선택이 핵심임을 실증적으로 보여준다.

- **Technical Challenges**: 핵심 난관은 미래 분포/미래 쿼리를 알 수 없는 상황에서, context의 유용성을 온라인으로 계산해야 한다는 점이다. 논문은 future-information view를 통해 이상적인 정보 기준을 분해하고, 이를 recent window(최근성의 프록시), predictive entropy(불확실성 기반 admission의 하한 신호), 표현 공간에서의 같은 클래스 near-neighbor(중복 제거 프록시)로 대체한다. 또한 long bank 용량이 찼을 때는 클래스 불균형을 먼저 줄인 뒤, class cmax 내부에서 가장 중복된 쌍을 찾아 recent short bank 기준으로 삭제 대상을 고르는 절차를 제시한다.

- **Empirical Impact**: 7개 stream(실세계 6개, 합성 1개)에서 CURE는 prequential accuracy에서 고전적 stream learner 대비 최대 상대 개선 27.0%를 보이며, 개선 폭이 데이터별로 일관되게 관측된다. 또한 TabICL-v2뿐 아니라 LimiX-v1, TabPFN-v2.5, TabDPT-v1 등 여러 TFM backbone에 대해 DualFIFO 대비 18개 비교 중 17개에서 이득을 보여 정책 전이성이 있음을 확인했다. controlled variant 실험과 구성요소 교체 실험은 uncertainty-gated admission과 redundancy-aware eviction이 상호 보완적으로 작동함을 뒷받침한다.



### scGTN: Deep Siamese Graph Transformer Network for Single-cell RNA Sequencing Clustering (https://arxiv.org/abs/2606.18672)
Comments:
          Accepted by Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI 2026)

- **Prior Approaches**: 기존 scRNA-seq 클러스터링은 k-means, 계층적 군집, 밀도 기반 같은 고전 알고리즘에 의존하거나, PCA 등으로 차원을 줄인 뒤 후처리하는 방식이 많았습니다. 딥러닝이 확산되면서 GNN 기반 방법이 등장했지만, 유전자 발현의 희소성/잡음 때문에 그래프가 불안정해지는 문제를 충분히 다루지 못하는 경우가 많습니다. 또한 로컬 이웃 집계에 치우쳐 shortest path나 노드 거리 같은 복잡한 구조 단서를 잘 활용하지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Siamese Graph Transformer Network(scGTN)로 scRNA-seq 그래프에서 유전자 표현 정보와 세포 간 구조 의존성을 함께 학습해 군집을 더 정확히 구하는 프레임워크를 제안합니다. scRNA-seq를 그래프로 모델링한 뒤, 서로 보완적인 두 augmented graph view를 만들고 Siamese 인코더로 처리합니다. 마지막으로 optimal transport 기반 self-supervised 분포 정렬로 클러스터 할당 학습을 안정화합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) dropout/측정 잡음으로 인한 희소·노이즈 그래프에서 견고한 표현을 만드는 것과 (2) 그래프의 전역적 구조(최단 경로, 노드 거리)를 Transformer가 구조적으로 반영하도록 설계하는 것입니다. 논문은 유전자에는 Gaussian noise 기반 perturbation, 그래프에는 spurious edge 제거와 graph diffusion으로 구조를 강화해 두 뷰를 구성하고, graph transformer에 shortest-path embedding과 position embedding을 함께 넣어 구조 신호를 명시적으로 주입합니다. 또한 optimal transport(예: Sinkhorn)로 타깃 분포와의 정렬을 반복 최적화해 degenerate solution을 줄이면서 self-supervised 클러스터링 손실을 구성합니다.

- **Empirical Impact**: 7개 scRNA-seq 벤치마크(인간/마우스, 높은 희소도 포함)에서 scGTN이 ACC, NMI, ARI 모든 지표에서 일관되게 기존 대비 우수한 성능을 보였고, 평균적으로 2~3%대 유의미한 개선을 보고합니다. UMAP 시각화와 differential gene expression 분석에서도 클러스터가 희귀 세포까지 비교적 명확한 경계를 유지하며, Gold Standard와 높은 마커 유전자 일치(대각선 중심, 90% 이상 overlap 등)를 보였습니다. 더 나아가 KEGG pathway enrichment으로 β세포/외분비/면역 기능 같은 생물학적 부분집단을 분리해 해석 가능성까지 강화했다는 점에서 scRNA-seq 클러스터링 연구에 실질적 기여로 평가됩니다.



### NeuralMUSIC: A Hybrid Neural-Subspace Framework for Robot Sound Source Localization (https://arxiv.org/abs/2606.18664)
- **Prior Approaches**: 전통적 SSL은 마이크 배열의 공간 상관을 이용해 DOA(방위각)를 추정하며, MUSIC 같은 서브스페이스 기법은 이론적 기반이 탄탄하다. 다만 저SNR·잔향·다중 음원 환경에서 공분산 추정이 흔들리며 잡음/신호 서브스페이스가 섞이는 subspace leakage가 발생하고 성능이 급락한다. 딥러닝 기반 방법은 잡음에 강해도, 환경·배열 형상이 바뀌면 학습 분포에 덜 맞아 generalization이 약해지는 경우가 많다.

- **Core Contribution**: 이 논문은 NeuralMUSIC이라는 하이브리드 신경-서브스페이스 프레임워크를 제안해, 공분산(공간 공분산) 추정의 강건함을 MUSIC의 구조적 해석 가능성과 결합한다. 네트워크가 멀티채널 입력에서 공간 공분산을 추정한 뒤, 이를 고전 MUSIC의 EVD(고유값 분해)와 pseudo-spectrum 계산에 그대로 투입해 DOA를 얻는다. 또한 Frequency Attention Fusion(FAF)으로 주파수 대역별 신뢰도를 가중 결합해 광대역(broadband) 문제를 완화한다.

- **Technical Challenges**: 핵심 과제는 (1) 잡음·잔향이 섞인 공분산을 안정적으로 추정해 EVD 기반 서브스페이스를 깨지 않게 하는 것, (2) 음향은 광대역이어서 주파수별로 서브스페이스가 달라지는 것을 일관되게 통합하는 것이다. NeuralMUSIC은 공분산을 Hermitian 형태로 symmetrize하고 diagonal loading으로 수치 안정성을 확보하며, FAF로 빈도별로 유효 정보를 강조한다. 더해 Self-supervised Spatial Correlation Learning(SSCL)로 라벨 없는 멀티채널 음향에서 채널 간 공간 상관을 학습해 데이터 효율을 끌어올린다.

- **Empirical Impact**: 여러 로봇 청음 작업과 데이터셋(Google Speech Commands, AV16.3, SLoClas, AFPILD)에서 NeuralMUSIC은 기존 서브스페이스 및 최신 딥러닝·하이브리드 대비 낮은 MAAE로 경쟁력 있는(특히 다중 음원/저SNR) 국소화 정확도를 보였다. 예를 들어 AV16.3에서 single speaker 및 unknown source number 설정 모두에서 최고 성능을 달성했고, 저SNR(0 dB)에서도 견조하게 유지됐다. 전이 실험에서도 환경/배열 변화에 대한 강건성이 확인되며, 임베딩( t-SNE )이 방위각 연속성을 보존하는 등 공간 구조 학습이 성능으로 연결됨을 시사한다.



### LandslideAgent with Multimodal LandslideBench: A Domain-Rule-Augmented Agent for Autonomous Landslide Identification and Analysis (https://arxiv.org/abs/2606.18661)
- **Prior Approaches**: 기존 원격탐지 기반 산사태 연구는 주로 CNN/ViT류로 시각 특징을 뽑아 객체 검출이나 semantic segmentation으로 경계(박스/마스크)를 찾는 데 집중해 왔다. 그러나 이런 방법들은 지형-환경 맥락, 유발 요인 같은 고수준 지리과학 의미 추론이 약하고, 데이터도 대개 이진 라벨 중심이라 few-shot 상황에서 일반화가 어렵다. VLM/LLM 기반 접근은 보고서를 생성하더라도 복잡한 지질 장면에서 perceptual 한계와 domain hallucination을 겪어 정밀 도구 호출과 일관된 추론이 무너질 수 있다.

- **Core Contribution**: 이 논문은 산사태 분석을 “인식→의미 추론→근거 기반 보고서 생성”으로 전환하는 instruction-driven agentic 프레임워크를 제안한다. 핵심은 (1) fine-grained 멀티모달 데이터셋 LandslideBench, (2) 산사태 전용 VLM LandslideVLM, (3) LandslideVLM을 두뇌로 하되 도메인 규칙으로 tool invocation을 제어하는 LandslideAgent의 3단 구성이다. 이를 통해 시각적 경계 산출과 지리과학 의미 생성, 그리고 도구 기반 검증을 한 시스템에서 연결한다.

- **Technical Challenges**: 주된 기술 난제는 지질 도메인에서 시각-언어 정렬 데이터가 부족하고, 일반 VLM은 섬세한 지형 증거를 건너뛰거나 그럴듯한(hallucination) 서술을 만들 수 있다는 점이다. 논문은 이를 LandslideBench의 정밀한 image-mask-text 3중 정렬(다중 검증/전문가 검수, 다중 VLM 교차검증)과 LoRA 기반 도메인 적응 fine-tuning으로 해결해 시각-지리 의미 정합성을 강화한다. 또한 LandslideAgent에는 메타데이터 의존 규칙(보고서 스키마를 맞추기 위한 필수 중간 산출물 강제)과 교차모델 기반 인식 규칙(세그멘테이션과 VLM 교차검증, 누락 방지)을 넣어 논리적 비약과 불필요한 도구 호출을 줄인다.

- **Empirical Impact**: 실험에서 LandslideBench는 5개 주요 모델에 대해 fine-grained 분류와 semantic segmentation의 기준선(baseline)으로 효과가 있음을 보인다. LandslideVLM은 landslide discrimination, fine-grained classification, semantic description quality에서 각각 10.96%, 32.87%, 15.91%의 정확도 향상을 기록한다. 더 나아가 LandslideAgent는 멀티소스 공간 데이터 추론을 자율 수행해 산사태 식별·분석의 end-to-end “full-process intelligence”를 실증하며 재난 지질 분야의 신뢰성 있는 자동화 방향을 제시한다.



### Augmenting Dysarthric Speech Severity Assessment with MOS Supervision (https://arxiv.org/abs/2606.18645)
- **Prior Approaches**: 기존 자동 발화(utterance) 단위 발작(의사)장애(dysarthria) 평가는 임상 라벨이 부족해 학습 병목이 컸습니다. 또한 보통 제한된 어휘(lexicon)와 매칭된 통제 프로토콜에 기대는 경우가 많아 자연스러운 평가가 어렵고, 생성형 데이터 증강은 라벨 재검증이 없어 지각 정렬(perceptual alignment)이 불명확하다는 한계가 있었습니다. SSL을 활용하더라도 임상 주석이 아닌 다른 지각 주석을 안정적으로 전이하는 방법은 부족했습니다.

- **Core Contribution**: 이 논문은 dysarthria 평가에 TTS 평가 코퍼스(QualiSpeech)의 human-annotated MOS(Mean Opinion Score) 라벨을 증강 소스로 활용하는 전략을 제안합니다. 특히 SAP의 Intelligibility(명료도)와 Naturalness(자연스러움) 예측에 대해, (1) QualiSpeech로 먼저 학습한 뒤 SAP로 fine-tuning하는 방식이 전이 성능을 일관되게 끌어올린다는 점을 보여줍니다. joint training에서는 자연스러움 중심 개선이 나타나며, 합성 실패와 dysarthria 징후가 공유하는 지각적 공통성을 실증합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘합성 음성의 MOS’와 ‘임상적 명료도/자연스러움’이 의미론적으로 얼마나 잘 맞물리는지(라벨 정렬)입니다. 저자들은 MOS(1–5)와 SAP(1–7) 스케일을 선형 변환해 정렬을 시도하되, joint training의 경우 명료도에서 negative transfer가 발생함을 관찰해 fine-tuning의 단계 분리로 해결합니다. 즉 FT는 QualiSpeech로 지각 기반 표현을 초기화한 뒤 SAP에서 목표 도메인에 맞게 재가중하도록 설계해 그래디언트 간 충돌을 줄입니다.

- **Empirical Impact**: 실험 결과, QualiSpeech 기반 증강은 두 차원(명료도·자연스러움) 모두에서 기준선 대비 성능을 개선했으며, 특히 naturalness에서 joint training도 유의미한 향상을 보였습니다. fine-tuning은 intelligibility에서도 일관된 이득을 제공한 반면 joint training은 자연스러움 쪽으로 이득이 치우치는 패턴을 보였습니다. 합성 품질 평가 코퍼스가 임상 주석 의존도를 낮출 수 있는 실용적 데이터 소스임을 제시하며, 향후 임상 스케일업과 교차도메인 전이 연구에 의미 있는 방향을 제공합니다.



### PEC-Home: Interpretation of Progressively Elliptical Commands in Smart Homes (https://arxiv.org/abs/2606.18636)
Comments:
          Accepted by ACL 2026 Findings

- **Prior Approaches**: 기존 LLM 기반 홈 어시스턴트는 “불충분/모호한 명령”을 대화 맥락이나 도구 호출로 보정하는 쪽에 초점을 둔다. 하지만 실제 사람-사람 대화에서 공통 기반이 쌓일수록 정보가 점진적으로 생략되는 현상(점진적 ellipsis)은, 현재 시스템이 명령을 ‘명시적 vs 모호함’ 같은 정적 범주로만 취급하면서 충분히 반영되지 않는다.

- **Core Contribution**: 이 논문은 점진적으로 생략되는 명령을 해석해 기기 제어로 정확히 매핑하는 작업을 새 과제로 정식화하고, 이를 위한 최초의 시뮬레이션 홈 데이터셋 PEC-Home을 제안한다. PEC-Home은 여러 사용자 환경에서의 기준 충돌(참조 모호성)과 시간이 지남/환경 변화에 따라 선호가 달라지는 문제(의도 모호성)를 함께 모델링한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 현재 명령만으로는 무엇을 지칭하는지(참조)와 (2) 무엇을 하려는지(의도)가 점점 더 불명확해지는 데다, 관련 대화 맥락이 다수 사용자·잡음 메모리에 섞일 수 있다는 점이다. 저자들은 가상 홈에서 장치-메서드-파라미터를 정의하고 명령의 생략 수준을 단계적으로 줄이도록 생성했으며, GPT-4o로 자연어를 만들고 휴먼-인더-루프 검증으로 데이터 품질을 확인했다.

- **Empirical Impact**: 실험 결과, GPT-4o를 포함한 여러 LLM은 점진적으로 ellipsis된 명령만 주어졌을 때 의도된 작업 실행 정확도가 크게 떨어졌다. 또한 RAG(대화 히스토리 검색), 외부 tool 통합, fine-tuning을 적용해도 고생략 수준에서는 완전 명령 대비 성능 격차가 해소되지 않아, 단순한 메모리/도구/학습만으로는 한계가 있음을 보여준다.



### EffiNav: Fusing Depth and Vision-Language for Efficient Object Goal Navigation (https://arxiv.org/abs/2606.18634)
- **Prior Approaches**: ObjNav 선행연구는 크게 (1) 대규모 학습을 통해 다음 탐색 결정을 내리거나, (2) 학습 없는 VLM/프런티어 기반으로 다음 탐색 지점을 고르는 방식으로 나뉜다. 하지만 학습 기반은 데이터·계산 비용이 크고 범용성에서 흔들리며, 학습 없는 방식은 visited area를 다시 훑거나 프런티어 간 왕복이 늘어 효율이 떨어질 수 있다. 특히 깊이 기반 또는 단순 전역 지식 부재는 지도·시점 오류가 쌓이며 비효율을 키우는 취약점으로 지목된다.

- **Core Contribution**: 이 논문은 학습 없이도 효율적 탐색을 목표로 하는 VLM 기반 프레임워크 EffiNav를 제안한다. 핵심 아이디어는 (1) 깊이로 후보 탐색 영역을 만들고, (2) egocentric 선택과 top-down 관점의 global-wise 검증을 함께 수행하며, (3) history-aware pruning과 프런티어 백업으로 재탐색·막힘을 줄이는 것이다. 또한 memory-augmented ObjNav로의 확장도 함께 시연해, 표준 ObjNav를 넘어선 적용성을 보인다.

- **Technical Challenges**: EffiNav가 해결해야 할 기술적 과제는 “다음에 어디를 볼지”를 효율적으로 고르는 것과, 그 선택이 실제 공간에서 말이 되는지 전역 일관성을 보장하는 것이다. 저자들은 depth-aware 후보 마스킹 후 VQA 형태로 egocentric에서 1~2순위 후보를 고르고, 이를 점진 구축된 top-down 지도에 투영해 동일 VLM으로 전역 검증을 수행한다. 선택이 계속 실패하면 nearest frontier 픽셀로 백업하며, 경로는 A*로 collision-free를 계산하되 장애물 팽창·목표/자세 미세 보정까지 더해 견고성을 확보한다.

- **Empirical Impact**: EffiNav는 시뮬레이션 Habitat HM3D와 OVON에서 Success Rate(SR)와 Success weighted by Path Length(SPL) 기준으로 기존 베이스라인을 상회하거나 근접 성능을 보이며, success 조건의 효율을 더 잘 드러내는 Normalized Efficiency on Successes(EoS)에서도 강점을 보인다. 2,019~3,000 에피소드 규모의 대규모 실패 분석에서는 실패 원인이 데이터셋 성격(탐색 난이도 vs 객체 의미 정합)별로 달라짐을 보이고, EffiNav는 그 차이 속에서도 효율 중심 지표를 비교적 안정적으로 유지한다. 더 나아가 실제 로봇(unitree go2 + Azure Kinect) 실험에서도 SR은 비슷하더라도 SPL과 EoS가 개선되어, “정확히 찾되 더 적게 헤매는” 실용적 효과를 실증한다.



### BCL: Bayesian In-Context Learning Framework for Information Extraction (https://arxiv.org/abs/2606.18620)
Comments:
          ACL 2026 Findings

- **Prior Approaches**: 기존 정보추출(IE)에서는 in-context learning(메모리 기반 학습)이 널리 쓰이지만, task transfer 방식(ChatIE, CodeIE)은 모델 크기에 따라 성능이 크게 흔들리거나(작은 모델에서 급락) relation classification에서 사실상 실패하는 경우가 있었다. guideline 기반 방식(예: GuideNER)은 NER에 강점이 있으나, 기준선이 되는 규칙 품질을 주파수 기반으로 고정해 체계적 최적화를 놓치고 RE로의 확장도 제한적이다.

- **Core Contribution**: 이 논문은 BCL(Bayesian In-Context Learning Framework for Information Extraction)로, IE 라벨을 의미적으로 잘게 쪼갠 subcategory 패턴을 ‘컨트롤 가능한 이산 변수’로 보고 이를 최적화한다. particle filtering과 Bayesian update를 결합해, 시퀀스 라벨링(NER)과 relation classification(RE) 모두에서 규칙 표현을 반복적으로 정련하는 프레임워크를 제시한다.

- **Technical Challenges**: 핵심 난제는 LLM의 거대한 파라미터 공간을 직접 관측·제어하기 어렵다는 점인데, BCL은 규칙 리스트를 저차원 관측 인터페이스로 재구성해 해결한다. 또한 규칙 공간이 조합적으로 불연속이고 성능 매핑이 비선형이라 고전 제어가 어렵기 때문에, 초기화-관찰(Validation 배치에서 likelihood 계산)-가중치 갱신(Bayesian posterior)-리샘플링(성능 상위 유지 + 다양성 mutation) 4단계를 통해 탐색과 수렴을 동시에 맞춘다.

- **Empirical Impact**: 실험에서 BCL은 여러 IE 벤치마크와 다양한 모델 스케일에서 기존 방법 대비 일관된 성능 향상(최대 약 30%)을 보이며, 특히 RE에서 다른 ICL 방식이 0에 가까운 성능을 보일 때도 유의미한 F1을 유지한다. 또한 비용 관점에서 Qwen2.5-3B에 BCL을 적용하면 더 큰 모델의 one-shot 성능과 비슷한 수준을 더 적은 파라미터로 달성하는 등 배포 가능성도 높였다.



### Code-Augur: Agentic Vulnerability Detection via Specification Inferenc (https://arxiv.org/abs/2606.18619)
- **Prior Approaches**: 기존 agentic vulnerability detection은 LLM이 코드에서 취약 지점을 “추정”하고, 이후에 실행/퍼징으로 그 추정을 일부만 검증하는 방식이 많았습니다. 문제는 에이전트가 스스로 내린 “안전” 판단의 근거가 암묵적이라 재사용·검증이 어렵고, 동적 검증도 대개 의심된 케이스에만 집중된다는 점입니다. 결과적으로 false negative(놓치는 취약점)가 생겨도, 그 원인이 추정 자체의 오류인지 명확히 알기 어렵습니다.

- **Core Contribution**: 이 논문은 security-specification-first 패러다임을 제안해, LLM 에이전트의 “안전” 판단에 포함된 가정을 security specification(국소 불변식)으로 명시화합니다. 또한 프로그램 소스에 assertion 형태로 불변식을 커밋하고, 실행 중 반증(falsification)되면 실제 취약점인지 혹은 specification 자체의 결함인지를 구분해 반복 정제하는 reason-falsify-refine 루프를 구성합니다. 이를 구현한 시스템이 Code-Augur입니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 에이전트의 암묵적 판단을 프로젝트 맥락(위협 모델, 공격 경계, 의미적 상태)과 연결되는 “검증 가능한” 형태로 바꾸는 것과 (2) 그 specification을 신뢰성 있게 깨뜨리는 입력 탐색을 설계하는 것입니다. Code-Augur는 에이전트가 안전 판정을 내릴 때 그 배후 가정을 지역 불변식 ϕ로 소스에 삽입하고, 이를 grey-box guided fuzzer의 목표로 삼아 런타임에서 ¬ϕ를 유도합니다. 불변식 위반 시 실제 결함이면 취약점으로 수렴하고, benign divergence면 불변식(가정)을 수정해 에이전트의 코드를 보는 관점을 실제 동작에 맞춥니다.

- **Empirical Impact**: 실험에서 Code-Augur는 AIxCC와 OSV 기반 벤치마크에서 기존 SOTA agentic 시스템 대비 더 많은 버그를 찾아내며, inferred specification이 발견의 중심 역할을 했다고 보고합니다. 또한 실제 오픈소스 프로젝트에서 22개의 신규 취약점을 찾아냈고, 그중 16개는 이미 수정되거나 개발자 확인을 받았으며 일부는 bug bounty까지 이어졌습니다. “판단이 없었던 영역(안전하다고 한 곳)”까지 런타임 반증 가능한 아티팩트로 남긴다는 점에서, 에이전트 기반 보안 분석의 신뢰도 문제를 실무적으로 개선할 의미가 큽니다.



### AI-Driven Assessment of Human Tutors: Linking Training Performance to Real-Life Practic (https://arxiv.org/abs/2606.18617)
Comments:
          Full research paper accepted at EC-TEL 2026

- **Prior Approaches**: 기존 튜터 양성 플랫폼은 AI가 시뮬레이션 기반 훈련을 제공하는 경우가 많지만, 실제 튜터링에서의 기술 전이를 검증하기는 어려웠다. 또한 open response를 포함한 훈련 평가는 자동화가 쉽지 않아, 전이 타당성(validity)과 일반화에 대한 근거가 부족했다.

- **Core Contribution**: 이 논문은 Generative AI로 실제 튜터링 전사(transcription)를 분석해, 훈련 중 tutor move 평가와 실전 행동 적용을 연결하는 AI 기반 평가 파이프라인을 제안한다. training에서의 학습 성과뿐 아니라 real-life transcript에서 기회(opportunity) 존재 여부와 실행 품질을 함께 점수화해 전이를 측정한다. 또한 LLM 채점 프롬프트, scoring rubric, 데이터셋을 공개해 투명성과 재현성을 강화한다.

- **Technical Challenges**: 핵심 난제는 tutor move가 상황 의존적이라는 점이다(해당 기술을 보여줄 ‘pedagogical opportunities’가 생겨야 평가 가능). 이를 위해 기회 판별(prompt stage)과 실행 품질 평가(prompt stage)로 2단계 prompting을 구성했으며, Gemini-2.5-pro 기반 점수를 human gold standard와의 IRR로 점검했다. 나아가 open response와 MCQ 중 어떤 형식이 실전 전이를 더 잘 예측하는지 AIC/BIC 기반 모델 비교와 혼합효과모형으로 검증했다.

- **Empirical Impact**: 원격 수학 튜터(N=86)에게 6개 시나리오 레슨을 제공한 결과, 평균 7.4%의 학습 이득이 관찰됐고(일부 레슨은 천장효과로 유의하지 않음), 실전 transcript 점수도 유의하게 개선됐다. 전이 예측에서는 튜터 훈련 성과 1 SD 증가가 실전 점수 0.25 SD 증가와 연결됐으며, 형식 비교에서는 open response와 MCQ를 함께 봤을 때가 가장 잘 맞았지만 open response가 상대적으로 더 예측력이 높았다. 또한 interrupted time series 분석 결과 개선은 ‘즉시 중재 효과’보다는 시간에 따른 점진적 추세로 나타났고, 기회 포착 확률은 61.1%→68.9%, 기회 내 실행 품질은 65.5%→68.1%로 상승해 실전 적용의 현실적 변화를 시사한다.



### Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive Doctor-Patient-EHR Assistanc (https://arxiv.org/abs/2606.18613)
Comments:
          34 pages with 8 figures

- **Prior Approaches**: 기존 의료 LLM 평가는 의학 지식(질문응답), EHR 시스템 접근(조회/툴 사용), 환자 커뮤니케이션(대화/문서 생성)처럼 역할을 분리해 측정하는 경우가 많았습니다. 하지만 실제 의사 보조는 한 상호작용 안에서 지식·소통·시스템 동작을 동시에 조율해야 하고, 요청은 종종 맥락 의존적이며 EHR과 환자 정보는 각각 정밀한 툴 입력과 모호한 서술을 요구합니다. 이런 “단일 능력” 중심 평가는 다턴, under-specified 상호작용에서 성능이 크게 떨어진다는 최근 연구 흐름과 충돌합니다.

- **Core Contribution**: 이 논문은 의사-환자-EHR이 함께 얽힌 대화형 보조 시나리오를 평가하는 벤치마크 PhysAssistBench를 제안합니다. MIMIC-IV 실제 케이스를 기반으로 다턴의 agentic 환자를 합성 생성해, 정적 기록을 임상 시나리오로 “살려” 다중 턴에서 의사의 암묵적 요청과 환자 모호성을 처리하도록 시험합니다. 또한 1,296개 턴을 수기 검토하고 의사 검증을 거친 큐레이션 평가 세트를 제공합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 의사가 준 맥락 의존적 요청을 정확히 해석하고, (2) 환자 서술의 불명확성을 질문-수집으로 다루며, (3) FHIR R4 기반 EHR 툴 호출을 적절한 타이밍에 수행한 뒤, (4) 여러 근거를 하나의 임상 답으로 일관되게 통합하는 것입니다. 저자들은 POMDP 형태로 세션을 모델링하고, MIMIC-IV에서 시나리오·증거 풍부도(tier)에 맞춰 다턴 아크를 계획/생성하되 “근거 없는 반사실”은 필터링하는 다중 에이전트 합성 데이터 파이프라인과 품질 게이트(환각·구조 오류·임상 안전성)를 결합해 이를 해결합니다.

- **Empirical Impact**: 실험에서 여러 대표 LLM은 평균 점수(mRS)에서는 비슷해 보이지만, 세션 단위 일관성(Pass@Session)에서는 큰 격차가 나타나 “모든 턴을 끝까지 안정적으로 조율”하는 능력이 병목임을 드러냈습니다. 특히 IL→WU→CR→DG로 난이도 하이라키가 형성되고, DG와 CR(다중 툴·근거 조합)에서 암묵성(명명/술어 생략/추상 사건 지시)이 성능 저하를 키우는 경향이 확인되었습니다. PhysAssistBench는 파라메트릭 지식 격차보다 툴 체이닝·대화 맥락 통합 같은 조합 능력을 정밀하게 측정하게 해, 임상 LLM 평가의 기준선을 바꾸는 의미가 큽니다.



### QC-GAN: A Parameter-Efficient Quaternion Conformer GAN for High-Fidelity Speech Enhancemen (https://arxiv.org/abs/2606.18611)
Comments:
          10 pages, 6 figures and 5 tables. Accepted at Interspeech2026

- **Prior Approaches**: 최근 음성 향상은 Transformer/Conformer 기반으로 time-frequency(T–F) 영역에서 성능이 크게 올라왔습니다. 다만 경량화를 위해 채널 축소, pruning, depthwise separable convolutions 같은 압축을 과감히 적용하면 표현력이 떨어지고, 특히 phase를 다루는 정확도가 무너지면서 PESQ 같은 지표에서 한계가 드러납니다. 또한 기존 QNN 연구는 주로 분류·공간추정 등 판별 문제에 머물러, magnitude와 phase를 동시에 재구성해야 하는 단일채널 speech enhancement에는 적용이 제한적이었습니다.

- **Core Contribution**: 이 논문은 Quaternion Conformer GAN(QC-GAN)이라는 경량 speech enhancement 프레임워크를 제안합니다. Quaternion Conformer를 생성기로 쓰고, MetricGAN 계열 학습으로 discriminator가 PESQ 등 지각 품질 점수를 근사해 generator가 perceptual quality를 직접 끌어올리도록 설계했습니다. 무엇보다 Hamilton product의 구조적 weight sharing으로 magnitude와 phase를 결합 표현(quaternion)으로 처리해 파라미터 효율을 높였습니다.

- **Technical Challenges**: 문제는 phase 정보가 조금만 틀어져도 audible artifact(예: musical noise)로 이어질 수 있어, 작은 모델에서 phase 재구성 능력을 유지하기가 어렵다는 점입니다. QC-GAN은 quaternion 축에 STFT의 magnitude와 phase 관련 성분을 직접 배치하고, Quaternion Conformer의 self-attention(Q-MHSA)·convolution을 Hamilton 연산으로 재구성해 component 간 결합을 유지하도록 해결했습니다. 동시에 generator 손실에 differentiable PESQ 신호와 MetricGAN discriminator 기반 지도를 함께 넣어, 단순 MSE 중심 학습의 한계를 보완했습니다.

- **Empirical Impact**: VoiceBank+DEMAND에서 QC-GAN(Base)은 0.89M 파라미터만으로 PESQ 3.48을 달성해 CMGAN보다 좋거나 MP-SENet에 근접하면서도 크기는 2~2.5배 이상 작았습니다. 더 작은 QC-GAN(Tiny, 35K 파라미터)도 PESQ 3.23으로 기존 초경량 모델들을 앞질렀고, STOI 0.94로 intelligibility 보존을 확인했습니다. 또한 DNS-Challenge 3 blind test에서도 실세계 환경으로의 일반화가 관측되어, 경량화가 underfitting 없이 품질을 유지할 수 있음을 시사합니다.



### Steerable Cultural Preference Optimization of Reward Models (https://arxiv.org/abs/2606.18606)
Comments:
          Accepted to Pluralistic Alignment @ ICML 2026

- **Prior Approaches**: 기존 LLM alignment 연구는 주로 특정 지역/인구집단의 선호도를 “하나의 공통된 기준”처럼 다뤄, 소수집단 선호가 과도한 편향으로 반영되는 문제가 자주 보고됐다. Group Preference Optimization(GPO)나 Group Robust Preference Optimization(GRPO)처럼 집단 선호를 학습하는 접근도 있지만, 보편 RLHF/DPO 같은 정렬 프레임워크에 자연스럽게 끼워 넣기 어렵거나(모듈 분리 문제), 특정 집단에 대한 독립적인 steerability를 충분히 겨냥하지 못했다. 또한 필터링·가중치 기반 정렬(예: RAFT, OPTune, Mallows-DPO)은 샘플 품질을 다루긴 해도 “문화적 하위커뮤니티 균형” 자체를 정면 목표로 삼지 않았다.

- **Core Contribution**: 이 논문은 다수의 문화 하위커뮤니티 선호를 과도한 쏠림 없이 반영하도록 하는 reward model 학습 방식에 초점을 둔다. 핵심은 ‘SCPO(Steerable Cultural Preference Optimization)’로, 글로벌 reward model이 “주류/합의 선호”를 기준선으로 삼아, 각 나라(집단)에서만 드러나는 문화적 차이를 분리하고 학습에 반영한다. 특히 global RM을 정답으로 가정하지 않고, 소수집단 선호가 어디서 갈라지는지(불일치/다이버전)를 조정 도구로 사용한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 소수집단별 reward model이 (1) 고유한 문화 신호는 학습하되, (2) global RM과의 큰 불일치가 항상 “가치 있는 차이”가 아닐 수 있다는 점(노이즈·라벨 오류 가능성)을 함께 다뤄야 한다는 것이다. SCPO는 두 단계로 해결한다: filtering은 global RM과 불일치하는 preference pair만 남겨 “문화적 구별점”을 남기고, weighting은 두드러진 다이버전일수록 학습 손실에서 더 낮게 가중해 편향 위험을 완화한다. Bradley-Terry 확률적 관점을 확장해 불일치 크기가 갖는 신호 강도/신뢰도 해석을 weighting에 연결하고, 최종적으로 가중 binary ranking loss로 minority RM을 학습한다.

- **Empirical Impact**: 실험에서는 PRISM(7개국, 미국·영국 제외)을 사용해 OpenAssistant와 Tülu 3 기반 reward model을 country-specific 데이터로 학습하며, SCPO가 대부분 국가에서 baseline fine-tuning 대비 성능을 개선했다. PRISM과 GlobalOpinionQA 모두에서 minority reward model(소수집단 선호) 성능이 상승했고, minority 대비 정확도/일반 성능 간 trade-off를 별도 “true country-specific subset” 평가로 점검했다. 또한 SCPO는 full-data finetuning 대비 최대 280% 더 학습 데이터 효율적이며, GlobalOpinionQA에서는 Jensen-Shannon Distance 기반 분포 일치도와 GPO 대비 우수한 문화 정렬 결과를 보였다.



### MIDS: Detecting Stealthy Masquerade and Tampering Attacks on CAN Bus via Bidirectional Mamba (https://arxiv.org/abs/2606.18599)
- **Prior Approaches**: 기존 CAN IDS는 프레임 주기·ID별 inter-arrival 같은 교통 통계에 맞춰져 있어, 제작(fabrication)형 공격(DoS, fuzzing, ID spoofing, replay)처럼 신호가 삽입되어 패턴이 흐트러질 때 강점을 보였습니다. 반면 최근 제안된 masquerade 환경에서는 공격자가 합법 프레임을 원래 전송 슬롯에서 in-situ로 치환해 주기 자체를 보존하므로, 주파수 분포·엔트로피·통계 기반 탐지가 약해집니다. 또한 ID와 payload의 ‘쌍’ 의미를 함께 보지 못하면, ID는 정상처럼 보이고 데이터도 어딘가엔 맞는 형태라 국소적으로는 정상으로 오인될 수 있습니다.

- **Core Contribution**: 본 논문은 masquerade(영구적 주기 보존형 ID 치환)뿐 아니라 data tampering(데이터만 변조) 및 combined tampering(둘 다 변조)을 한 위협 모델로 묶고, 이때 필요한 탐지 접근을 제시합니다. 이를 위해 Mamba Intrusion Detection System(MIDS)라는 듀얼 스트림 프레임워크를 제안하며, CAN identifier와 payload를 병렬 처리하고 Bi-Mamba의 bidirectional selective state-space modelling으로 ID–데이터의 결합된 시간 의미를 복원합니다. 기존에 인젝션 중심이었던 탐지 패러다임에서, 주기 신호가 사라지는 ‘의미 수준 이상’ 탐지로 초점을 전환한 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 masquerade가 per-ID 주기와 트래픽 통계 교란을 의도적으로 제거한다는 점이며, 그래서 모델이 통계보다 의미적 특징(윈도우 내 ID–payload 결합, 장기 궤적 변화)에 의존해야 합니다. MIDS는 (1) ID는 임베딩으로, (2) payload는 1D-CNN으로 지역 시계열 특징을 뽑은 뒤, 두 스트림을 결합해 Bi-Mamba가 순방향/역방향 맥락에서 드리프트와 불일치를 동시에 포착하도록 설계했습니다. 또한 S6의 입력-의존 선택 메커니즘과 비대칭 파라미터 구성으로 노이즈는 억제하고 미세한 의미 변화는 강조하며, 순방향·역방향 표현을 가중 통합해 최종 4클래스(정상/masquerade/data tampering/combined) 확률을 산출합니다.

- **Empirical Impact**: 실차 Tesla Model 3에서 1억 개가 넘는 CAN 프레임을 수집하고, masquerade 변형 54종(ID-only/data-only/combined)을 합성해 평가한 결과 MIDS는 F1 96.94%를 달성하며 재현 가능한 최강 베이스라인 대비 8%p 이상 향상되었습니다. 단일 윈도우 추론 지연은 1.147 ms로, 차량 내 실시간 온보드 적용 여지를 보여줍니다. 또한 ROAD, CrySyS, OTIDS, CT&T의 공개 벤치마크에서도 F1이 93.70%~99.61%로 보고되며, 8개 재현 베이스라인 대비 최대 13.94%p까지 앞서는 성능을 통합 5-fold 프로토콜로 입증했습니다.



### Better Adherence, Richer Context: A Field Evaluation of LLM-Powered Conversational Voice Diaries for Sleep (https://arxiv.org/abs/2606.18596)
- **Prior Approaches**: 수면 일지는 행동 수면의학과 CBT-I에서 핵심 도구로, Consensus Sleep Diary(CSD)처럼 침대 시간·각성 횟수·수면 질 같은 변수를 매일 기록해 야간 변동을 추적해왔다. 다만 손수 입력해야 하는 정적 양식은 피곤한 상태에서 누락/지연 작성이 잦고, 보고 창(window) 밖 작성은 회상 편향을 키운다. 또한 형식이 구조화될수록 스트레스 요인, 환경 교란, 일상 루틴 등 맥락 정보가 빈약해져 임상 해석에 제약이 있었다.

- **Core Contribution**: 이 논문은 LLM 기반의 대화형 음성 수면 일지(voice diary)를 스마트 스피커에 얹어, 아침·저녁에 임상적으로 타당한 질문을 선제적으로 건네고 대화형 입력으로 수집하도록 설계했다. 사용자 반응이 애매하거나 불완전하면 LLM이 적응형 후속 질문을 통해 수면 변동의 원인까지 설명하도록 유도한다. 즉, “정해진 폼 채우기” 대신 “일상 루틴에 내장된 대화형 자기보고”로 수면 일지의 품질과 지속성을 동시에 노린다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 매일 반복되는 시간 민감 입력을 집에서 무리 없이 수행하게 하는 것과 (2) 대화의 자유로움이 구조화된 임상 변수의 정밀도를 해치지 않게 하는 균형이다. 저자들은 보고 가능 시간(allowable window)과 선제 프롬프트 시간(proactivity window), 그리고 sleepDiarySTATUS 같은 상태 관리로 중복 입력과 세션 중단-재개를 처리했다. 동시에 GPT-4o로 언어 이해·후속 질문·대화 흐름을 담당하되, 기본 질문 셋은 CSD 기반의 구조를 유지해 임상 해석에 필요한 필드를 보존했다.

- **Empirical Impact**: 4주 간 30명(대조군 텍스트 모바일 일지와 VA 음성 일지, 15명씩)을 대상으로 한 between-subjects 현장 실험에서 음성 일지는 텍스트 대비 순응도와 맥락적 자기보고(루틴, 스트레스 요인, 환경 조건 등)에서 더 풍부한 결과를 보였다. 참여자들은 음성 일지를 일상에 통합하기 쉽다고 평가했지만, 체감 작성 시간은 더 길게 느꼈다. 반대로 일부 구조화 필드에선 완성도가 낮아져 ‘표현력 증가 vs 구조적 정밀도 유지’의 트레이드오프가 확인되며, LLM 음성 비서 기반 장기 건강 자기보고의 가능성과 한계를 동시에 보여준다는 점에서 의미가 있다.



### Benchmarking Action Spaces in Reinforcement Learning for Vision-based Robotic Manipulation (https://arxiv.org/abs/2606.18594)
Comments:
          9 pages with references

- **Prior Approaches**: 로보틱스 조작에서 action space 설계는 motion smoothness, 안전성, sim-to-real 전이 성능에 큰 영향을 준다고 알려져 있다. 다만 기존 연구는 주로 상태 기반 관측(state-based observations)이나 시뮬레이션에 치우쳐 있고, vision으로 인한 부분관측과 지각 잡음이 실제 학습/제어 동역학을 어떻게 바꾸는지까지는 충분히 다루지 못했다.

- **Core Contribution**: 이 논문은 vision-based 조작 문제에서 pose increment, pose velocity, joint position increment, joint velocity 같은 대표 action space 4가지를 Franka Emika Panda에서 비교한다. 시뮬레이션(PPO) 학습 후 sim-to-real transfer로 실제 로봇에 배치해, action space 선택이 실제 성능을 유의미하게 좌우함을 보인다. 특히 joint velocity action space가 picking과 pushing 모두에서 smoothness와 최종 과제 성능 측면에서 가장 유리하다고 정리한다.

- **Technical Challenges**: 핵심 난제는 vision이 만드는 부분관측/잡음이 action space에 따라 학습 안정성과 실제 동작을 다르게 흔들 수 있다는 점이다. 저자들은 domain randomization으로 카메라/조명/물체 초기조건/역학 파라미터를 흔들고, PPO의 vision actor–critic 구조에 맞춰 action을 조절 가능한 actuator 모델(예: joint velocity/position increment, Cartesian 조절 포함)로 일관되게 매핑해 전이 민감도를 낮추는 방향을 택했다.

- **Empirical Impact**: 실제 평가에서 joint velocity는 충돌을 피하고 jerk가 가장 낮은 편이었으며, picking 과제에서 성공률 100%를 달성(중앙 완료시간 약 3.58s)했다. 반면 pose increment나 pose velocity는 실제에서 불안정/오프태스크 경향이 커져 성능이 크게 떨어졌다. 저자들은 이를 바탕으로 vision 기반 sim-to-real 실험에서 action space를 고를 때 joint velocity를 우선 후보로 검토하라는 실무 가이드를 제시한다.



### Dual Dimensionality for Local and Global Attention (https://arxiv.org/abs/2606.18587)
- **Prior Approaches**: 기존 KV cache 절감 연구는 슬라이딩 윈도우/스트리밍 같은 sparse attention 또는 heavy hitters 기반 eviction으로 효율을 높이거나, MLA처럼 key/value 차원을 사전 압축해 메모리 부담을 줄이는 방식이 주를 이뤘습니다. 다만 이러한 방법들은 대체로 모든 토큰에 동일한 표현 차원을 부여하거나, 거리에 따른 “필요 표현력”의 차이를 직접 모델링하진 못했습니다. 그래서 토큰 거리(distance)가 표현 차원 요구량을 어떻게 바꾸는지에 대한 정량적 가설 검증이 부족했습니다.

- **Core Contribution**: 이 논문은 Distance-Adaptive Representation(DAR)이라는 원리를 제안하며, 예측 대상에 가까운 로컬 토큰은 더 풍부한(고차원) 표현이 필요하고 멀리 있는 토큰은 더 낮은 차원으로도 충분하다는 비대칭 가설을 형식화합니다. 구현은 로컬 윈도우 내부 토큰은 원래 차원을 유지하고, 그 밖의 토큰은 병목 차원(down-projection)으로 줄인 표현을 key/value로 사용하도록 설계됩니다. 즉, “거리별로 표현 차원을 배분”하는 새로운 KV 설계 방향을 제시합니다.

- **Technical Challenges**: 핵심 과제는 서로 다른 차원의 표현(로컬 vs 글로벌)을 표준 attention 계산 흐름에 자연스럽게 결합하는 것입니다. 논문은 공통의 key/value 투영을 공유하기 위해 글로벌 경로의 병목 표현을 다시 모델 차원으로 up-projection하는 2경로(attention paths) 구조를 사용해 학습 안정성을 확보합니다. 또한 auxiliary loss 없이 다음 토큰 예측만으로 down/up projection이 거리별 표현을 학습하도록 만들었고, 윈도우 크기 변화에 대해서도 성능이 특정 구간에서 견고함을 확인했습니다.

- **Empirical Impact**: 실험 결과 DAR은 Pythia-70M~410M 스케일에서 full-dimensional baseline과 거의 같은 perplexity를 달성하며, 성능 저하는 “균일 축소(uniform reduction)”에 비해 훨씬 완만했습니다. 특히 down 차원을 d/4 수준으로 설정하면 스케일 전반에서 기준선과의 격차가 작았고, downstream 작업에서도 적당한 축소 범위(d/2~d/4)는 no-bottleneck 대비 유지/근소 개선을 보였습니다. 결론적으로 토큰 거리별 표현 차원 비대칭이 실제로 성능 손실 없이 KV cache 절감을 설계할 수 있음을 보여, long-context inference 효율화의 새로운 실험 축을 제안합니다.



### APT: Atomic Physical Transitions for Causal Video-Language Understanding (https://arxiv.org/abs/2606.18586)
- **Prior Approaches**: 기존 물리 비디오 연구와 벤치마크는 “bounce” 같은 클립 수준 event 라벨이나 결과 중심의 VQA/생성/플라우저빌리티에 많은 감독이 묶여 있었다. 이런 방식은 무엇이 일어났는지는 맞아도, 지지 상실·접촉 시작·반동·정착 같은 인과적 상태 전이가 어떻게 연결되는지는 과소평가되기 쉽다. 또 VLM이 이벤트를 맞춘다고 해서 전이 단위의 물리 메커니즘을 복구한다는 보장은 부족했다.

- **Core Contribution**: 논문은 동영상의 숨은 과정을 “Atomic Physical Transitions(APTs)”라는 전이 단위의 인과 체인으로 명시화한다. APT는 시각적으로 국소화된 최소 상태 변화로, 전후 동역학 regime 전환과 “왜 그런지”를 지지하는 활성 물리 메커니즘을 타입화한다. 그 결과, 영상 이해를 단일 event 라벨이 아니라 ordered APT chain으로 표현해 전이 타임스탬프와 타입을 동시에 복원하도록 만든다.

- **Technical Challenges**: 문제는 APT 같은 전이 경계 감독이 수동으로는 얻기 어렵고, 픽셀만으로는 전이 경계를 안정적으로 추론하기 힘들다는 점이다. 저자들은 인간 라벨(CLEVRER/Physion++)로 14개 전이 타입의 의미를 캘리브레이션하고, 시뮬레이터(Physion++·Phys4D/Isaac Sim 계열) GT 트레이스에 기반해 rule-guided coarse-to-fine 파이프라인으로 mixed-source APT 데이터를 구성한다. 학습에서는 APT-JSON 같은 포맷으로만 단순화하면 “포맷 특화(specialist collapse)”와 event-level 망각이 발생해, 이를 막기 위한 parameter-efficient 미세조정 레시피 APT-Tune을 제안한다.

- **Empirical Impact**: 평가 결과, 8종의 VLM은 동일 입력과 APT 스키마 프롬프트 조건에서도 zero-shot APT recall이 10~14%에 그쳤고, 주된 실패 원인은 타이밍 지터보다 전이를 놓치는 것이었다. 반면 APT-Tune은 Qwen3-VL-2B에서 LoRA 11M 파라미터만으로 APT recall을 38.1%까지 끌어올리며, 최종적으로는 다른 백본에서 최대 53.4% recall을 달성했다. 또한 APT-Tune은 MVBench(이벤트 수준)에서 소폭 향상(일괄적으로 +0.5~2.2pp)과 PhysBench(실세계 물리 이해)에서 최대 +18.2pp 개선을 보여, 전이 수준 물리 습득이 답변 포맷 편향으로 끝나지 않고 전이적 재사용 가능한 물리 표현으로 이어짐을 시사한다.



### Multi-Modal Hyper-Graph Fusion for Low-Light Crowd Counting (https://arxiv.org/abs/2606.18566)
- **Prior Approaches**: 기존 군집수( crowd counting ) 연구는 주로 주간/양호 조도 환경에 최적화돼 있으며, 밀집 장면에서는 밀도맵 기반과 포인트 기반(머리 중심 예측) 패러다임이 주류였습니다. 저조도에서는 RGB 단일 모달이 조도 변화·잡음·대비 저하에 취약해 성능이 급락하고, enhance-then-count처럼 향상과 카운팅을 분리한 파이프라인은 잡음 증폭이나 구조 디테일 손실로 오차가 누적되는 문제가 있었습니다. 또한 전용 저조도 벤치마크와 tailored 방법이 부족해, 저조도 상황의 검증이 제한적이었습니다.

- **Core Contribution**: 이 논문은 저조도 군집수 인식을 위해 세 가지 저조도 벤치마크를 새로 구축하고(합성 2종 SHA_Dark/SHB_Dark, 실세계 LC-Crowd), 모델도 저조도용으로 통합 설계합니다. 핵심 아이디어는 Retinex 관점으로 저조도 카운팅을 ‘반사율 관련 표현의 재보정(reflectance re-calibration)’ 문제로 정식화하고, RGB의 조명 성분이 망가질 때 depth와 Canny edge의 기하·구조 단서를 통해 보정 가능성을 보인 것입니다. 이를 바탕으로 RGB 외 모달리티를 단순 보조로 쓰는 수준을 넘어, 고차 관계를 이용한 융합과 계산 효율을 함께 달성하는 LCNet을 제안합니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 극단적 어둠에서 단일 단서(depth나 edge)가 국소적으로 붕괴해 pairwise(쌍단) 결합이 오히려 오염될 수 있다는 점입니다. 논문은 Multi-Modal Hyper-Graph Fusion에서 RGB 외 appearance, depth geometry, edge structure 토큰을 동일 hyper-graph로 묶고, 동적 hyperedge 구성+메시지 패싱으로 여러 이웃의 ‘동시 합의’에 기반한 고차 보정을 수행해 단일 단서 실패에 대한 강건성을 높입니다. 또한 야간 장면에서 전경이 희소하다는 점을 고려해 Deformable Rectangular Sparse Attention(DRSA)으로 영역별 granularity를 추정한 뒤 선택된 anchor에만 deformable rectangular attention을 적용, 연산을 정보가 있는 곳에 집중시킵니다.

- **Empirical Impact**: SHA_Dark, SHB_Dark, LC-Crowd의 세 벤치마크에서 제안 방법은 기존 SOTA 대비 전반적 성능이 가장 좋다고 보고되며, 저조도에서의 실질적 견고성이 확인됩니다. 특히 실세계 LC-Crowd는 조도 분포의 비균일성이 크고 해상도·군집 규모 변동이 폭넓어, 단순 합성 열화만으로는 검증하기 어려운 현실 난제를 다룹니다. 결과적으로 저조도 군집수 연구에서 ‘전용 데이터+모달 재보정+고차 융합+연산 적응’이라는 설계 방향을 제시하며, 후속 연구들이 비교·확장하기 쉬운 기반을 제공한다는 점에서 의미가 큽니다.



### Correcting Sensor-Induced Distribution Drift with Wasserstein Adversarial Learning (https://arxiv.org/abs/2606.18561)
Comments:
          This is a preprint sent to Nuclear Science and Techniques journal

- **Prior Approaches**: 기존 센서 캘리브레이션/얼라인먼트는 잔차(residual)나 외부 기준 신호에 의존하는 경우가 많았고, 이때 열화(aging) 파라미터는 직접 관측되기 어렵다. 또한 많은 분포이동(domain shift) 추정은 이벤트 단위 대응(매칭)이나 라벨이 필요해 실제 운영 조건에선 제약이 컸다. GAN을 포함한 생성모형도 주로 샘플 생성이나 불일치 탐지에 초점이 있어, ‘물리적으로 해석 가능한 파라미터’를 라벨 없이 복원하는 방식과는 거리가 있었다.

- **Core Contribution**: 이 논문은 detector response 분포의 변화를 ‘분포 정렬(distribution alignment) + 파라미터 복원(parameter recovery)’ 문제로 재정의한다. Wasserstein-GAN에서 영감을 받은 adversarial learning을 쓰되, generator는 새 데이터를 만드는 대신 보정/왜곡 변환의 learnable calibration transformation으로 동작하여 그 가중치를 곧 해석 가능한 열화(혹은 얼라인먼트) 파라미터로 보게 한다. critic은 Wasserstein 목적함수로 분포 불일치 정도를 제공해, 라벨 없이도 변환 파라미터를 역으로 찾아내도록 설계된다.

- **Technical Challenges**: 핵심 난제는 (1) 이벤트 수준 대응 없이도 분포 수준에서 파라미터를 식별 가능하게 학습하는 것, (2) GAN 특유의 학습 불안정성을 완화하면서 Wasserstein 거리 추정의 조건(예: 1-Lipschitz)을 유지하는 것이다. 논문은 1-Lipschitz critic 제약(WGAN 계열 정식화)을 두고, 변환 파라미터는 generator의 결정적 변환 모듈의 trainable weight로 두어 Wasserstein 기반 목표를 직접 최소화하도록 최적화한다. 또한 트래커 misalignment toy model(단일 좌표 이동)과 칼로리미터 cell-wise aging(감쇠 계수 재추정)을 같은 프레임으로 통일해 파라미터 검색의 일반성을 보여준다.

- **Empirical Impact**: 트래커 toy 모델에서는 알려진 평면 이동 오프셋을 완전 비지도 방식으로 복원하며, 분포 불일치가 줄어드는 방향으로 파라미터가 수렴함을 확인한다. Geant4로 생성한 고세분화 calorimeter 데이터에서는 개별 cell의 aging coefficient를 ground truth와의 상관관계로 회복하고, 보정 후 energy-sum 분포가 reference와 더 잘 일치하는 개선을 보인다. 채널 간 잡음(channel-to-channel noise)이 커질수록 성능이 기대한 대로 열화되는 등, 라벨 부재 환경에서 adversarial distribution matching이 캘리브레이션 전략의 데이터 기반 구성요소가 될 수 있음을 시사한다.



### Engagement Intensity as a Learner-Modeling Signal for Adaptive AI Ethics Instruction (https://arxiv.org/abs/2606.18548)
- **Prior Approaches**: 기존 대학(특히 대학원) AI 윤리 교육은 수강생을 비교적 동질적인 집단으로 가정하고 설계를 단순화하는 경우가 많다. 또한 AI literacy 프레임워크는 학습자 중심 지도를 강조하지만, 수업 전 ‘어떤 입력(intake)이 실제 사전 인식 차이를 가장 잘 가르는가’에 대한 경험적 검증은 부족했다.

- **Core Contribution**: 이 논문은 대학원 연구윤리(RCR) 과정에 참여한 생명과학 대학원생/포닥 93명의 수업 전 설문을 바탕으로, 사전 인식 차이를 설명하는 intake 변수를 사용 빈도, LLM 친숙도(자기평가), 사전 AI 교육(과정/워크숍) 3가지로 비교한다. 결론적으로 사전 AI 교육 이력보다 ‘보고된 LLM 사용 행동’과 ‘자기평가 친숙도’가 더 정보가 풍부한 신호임을 보여준다.

- **Technical Challenges**: 핵심 기술 과제는 수업 전 수집 가능한 간단한 지표들이 Likert 기반의 여러 ‘AI 인식’ 항목(정확성 신뢰, 구별 능력, 복잡 작업 신뢰, 과의존 위험, 훈련 관심)을 얼마나 일관되게 분리하는지 검정하는 것이다. 저자들은 순서형 예측변수에 Spearman 상관, 범주형 사전 교육에는 Kruskal–Wallis를 적용하고, 5개 결과에 대해 Holm correction으로 가족별 오차를 통제했으며, 항목들을 하나의 합성 척도로 묶지 않고 각각의 관점을 개별 facet로 다뤘다.

- **Empirical Impact**: 결과는 사용 빈도가 5개 항목 모두에서 Holm 보정 후 유의한 연관을 보인 반면, 자기평가 친숙도는 3개 항목에서만 약하지만 의미 있는 연관을 보였고, 사전 AI 교육은 가족별 보정 후 어떤 항목에서도 유의하지 않았다. 또한 일부 항목(훈련 관심, 정확성 신뢰)에서만 ‘하단에서의 문턱형 패턴’이 관찰되어, 단순한 균일한 그라데이션보다 ‘비사용 vs 지속 사용’ 같은 분기 모델이 실무적으로 더 유용할 수 있음을 시사한다.



### AI Sandboxes: A Threat Model, Taxonomy, and Measurement Framework (https://arxiv.org/abs/2606.18532)
Comments:
          50 pages, 8 figures, 10 tables

- **Prior Approaches**: 기존 연구는 벤치마크, 시뮬레이터, 디지털 트윈, cyber range, 규제 샌드박스처럼 “평가를 위한 도구”를 각기 다른 관점에서 발전시켜 왔습니다. 하지만 이런 환경들이 만들어내는 증거를 어떤 배포(Deployment) 주장으로까지 정당화할지에 대한 공통된 보증(assurance) 규칙이 부족해, 고평가(over-reading) 위험이 남습니다. 또한 보안 연구는 공격 표면을 주로 AI 모델 중심으로 다루거나, 평가 인프라(검증·증거 생산 과정) 자체를 대상으로 한 위협은 상대적으로 덜 체계화돼 있었습니다.

- **Core Contribution**: 이 논문은 AI sandboxes를 “경계가 명시된 증거 기반 보증 절차”로 재정의하고, 샌드박스 경계(BB)와 증거 조합 규칙을 formalize합니다. 특히 각 차원(충실도, 통제가능성, 관측가능성, 격리/containment, 재현가능성, 거버넌스 아티팩트 등)의 증거를 모아 배포 주장(DD)을 정당화할 때는 weakest-link rule로 가장 약한 차원의 한도만 허용된다고 못 박습니다. 더불어 물리-사이버-거버넌스가 얽힌 상황에서, 평가 장치와 증거 체인 자체를 노리는 위협까지 포함하는 cyber-physical threat model을 제안합니다.

- **Technical Challenges**: 핵심 난제는 샌드박스가 “고립돼 있다”는 사실만으로는 배포 안전/보안/규제 적합성을 증명할 수 없다는 점을, claim-relative evidence로 강제하는 것입니다. 논문은 신뢰 경계(무엇이 표현되고 무엇이 생략되는지), 개입(중단/롤백/kill switch 등), 모니터링(로그·텔레메트리·증거 아티팩트), 잔여 위험(RR)을 튜플로 모델링해, 암묵적 가정을 문서화·감사 가능하게 만듭니다. 동시에 물리 AI/AIoT/CPS에서 실패가 물리 동역학과 실시간 제약, 센서·액추에이터 한계, 네트워크 열화, 공격-재현까지 연결되므로, 이를 커버할 측정 프레임워크(15개 차원)를 설계해 도구 간 비교 가능성을 확보했습니다.

- **Empirical Impact**: 논문은 제안된 threat model, 분류(taxonomy), 측정 프레임워크를 실제 샌드박스 3개 사례(working case studies)에 적용해 어떤 주장까지 유효한지 구체적으로 보여줍니다. 특히 closed-loop 시뮬레이션 증거를 과도하게 일반화하던 관행을 다시 점검하며, 어떤 차원(예: 관측가능성·containment·이식성)에서 증거가 부족하면 배포 주장으로 확장될 수 없음을 강조합니다. 결과적으로 물리-사이버-거버넌스 영역에서 “무엇을 테스트했다”를 넘어 “무엇을 보증할 수 있는지”를 규정하는 기준틀을 제공해, 안전·보안·규제(TEVV, AI Act 등) 논의의 공학적 정합성을 높일 것으로 기대됩니다.



### Sparsity Curse: Understanding RLVR Model Parameter Space from Model Merging (https://arxiv.org/abs/2606.18521)
Comments:
          Accepted by KDD 2026

- **Prior Approaches**: 기존 LLM 포스트트레이닝은 SFT와 RL 계열로 나뉘며, SFT는 라벨 시연을 따라가고 RLHF는 reward model 기반 정렬을 사용합니다. 최근에는 RLVR이 수학·코딩 같은 reasoning에서 검증 가능한 보상 신호로 더 강한 추론 능력과 catastrophic forgetting 저항을 보여주며 주류로 부상했습니다. 다만 SFT 중심의 model merging(Linear average, Task Arithmetic, TIES, DARE 등)은 RLVR에 그대로 적용하면 성능이 크게 떨어지는 것으로 관찰되어 왔습니다.

- **Core Contribution**: 이 논문은 RLVR이 SFT와 달리 업데이트가 매우 sparse하고 off-principal 방향으로 흩어져 “sparsity curse”를 만든다는 점에 주목합니다. 그 결과 독립적으로 학습된 RLVR 모델의 업데이트가 파라미터 공간에서 서로 near-orthogonal shortcut을 형성해, 단순 병합이 fragile하게 실패한다고 진단합니다. 이를 해결하기 위해 Sensitivity-aware Resolving Merging (SAR-Merging)을 제안하며, RLVR 파라미터 공간의 희소 구조에 맞춰 병합 규칙을 재설계합니다.

- **Technical Challenges**: 핵심 난제는 (1) sparse 업데이트로 인해 병합 시 conflict/overlap/private 영역이 예민하게 갈린다는 점과 (2) RLVR이 서로 다른 거의 직교하는 업데이트 방향을 학습한다는 점입니다. 논문은 conflict 구간에서 Fisher Information(대각 Fisher)을 기반으로 “어느 쪽 업데이트를 보존할지” 민감도 우선순위를 정하고, overlap 사이의 부호 충돌을 선택적으로 정리합니다. 이어 private 영역에 대해서는 magnitude-aware sparsification과 rescaling을 적용해 희소 경로를 조밀화(dense collapse)로 망가뜨리지 않게 했습니다.

- **Empirical Impact**: 수학 및 coding 벤치마크 실험에서 SAR-Merging은 기존 merging 방법 대비 RLVR 모델의 성능 저하를 크게 완화하며, single-task 향상뿐 아니라 multi-capability fusion까지 가능함을 보입니다. 특히 update geometry(층별 sparse도, near-orthogonality, activation density geometry) 분석을 통해 “왜 SFT는 잘 병합되고 RLVR은 망가지는지”를 경험적으로 설명하는 데까지 나아갑니다. 결과적으로 RLVR 기반 reasoning 모델의 training-free 확장/통합 전략에 실질적인 레시피를 제공했다는 점에서 의미가 큽니다.



### As You Wish: Mission Planning with Formal Verification using LLMs in Precision Agricultur (https://arxiv.org/abs/2606.18519)
- **Prior Approaches**: 기존 LLM 기반 로봇 미션 플래너는 자연어를 실행 가능한 임무로 바꾸는 데 강점이 있지만, 자연어의 내재적 모호성 때문에 사용자가 의도한 것과 다른 행동이 생성·실행될 수 있다. 또한 formal verification을 하더라도 사용자가 PDDL/LTL 같은 형식 언어를 직접 알아야 하거나, 검증이 사람의 개입 없이 완전히 자동으로 닫히기 어렵다는 한계가 있었다. 일부 연구는 LTL을 검증 입력으로만 쓰거나(데이터 흐름의 일부가 아님), LTL 생성은 별도로 하되 온라인 검증 파이프라인으로 일관되게 엮지 못했다.

- **Core Contribution**: 이 논문은 자연어 기반 precision agriculture 미션 플래너에 LTL 기반 검증을 결합하되, LTL 스펙을 사용자 친화적으로 “자동 생성”해 사람의 형식 언어 학습 부담을 없애는 아키텍처를 제안한다. 특히 미션 생성(로봇 task용 XML)과 검증을 위한 LTL 공식 생성 사이를 독립된 에이전트로 분리해 LLM bias를 줄이고, 검증 불일치가 나면 피드백 루프로 재생성한다. 결과적으로 “NL → 미션 생성 → LTL 스펙 생성 → 모델체킹(SPIN)으로 사양 위반 검사”가 인간 감독 없이도 루프 형태로 닫히는 완전 자율 파이프라인을 지향한다.

- **Technical Challenges**: 핵심 난제는 (1) 자연어 의도를 정밀하게 LTL로 표현하는 것, (2) SPIN/모델체킹이 감당 가능한 복잡도의 LTL을 생성하는 것, (3) 에이전트 간 사전 합의가 과도하게 들어가 검증의 독립성을 훼손하지 않는 것이다. 이를 위해 시스템은 XML 미션을 IEEE 표준 기반 XSD 제약과 linter로 문법적으로 보장하고, 검증용 LTL은 co-safe LTL의 제한 클래스(유한 실행 조각에서 만족/위반이 결정되는 형태)로 생성해 상태공간 폭발을 완화한다. 또한 두 LLM을 완전히 decoupling 후, 승인 단계에서만 최소한의 분해 합의(예: 작업/조건의 개수)만 확인하며, SPIN 위반 또는 LTL 문법/의미 불일치에 대해 재시도 피드백을 주어 수정한다.

- **Empirical Impact**: 현장 실험은 ClearPath Husky(ROS2 Humble, Jetson Orin Nano)에서 진행하며, End-to-end 성공적인 L1 미션 생성 속도와 함께 formal verification이 실제로 성립하는지(세 에이전트 합의 횟수, 의미적 일치/오해 여부, 평균 시간 등)를 평가한다. 논문은 SPIN/Spot을 온라인 승인 단계에 넣어 LTL 생성의 문법 오류(특히 괄호 매칭)와 의미 불일치가 재시도 루프에서 어떻게 회복되는지, 그리고 LTL 공식 품질이 성능 상한을 좌우할 수 있음을 강조한다. 전반적으로 “완전 자율 검증 파이프라인”의 강점과 LLM이 유의미한 LTL을 만드는 데서 생기는 제약을 함께 보여주며, 농업 로봇처럼 네트워크 없이도 안전한 오퍼레이션을 요구하는 도메인에 실용적 시사점을 제공한다.



### PSyGenTAB: A Privacy-Preserving Framework for Synthetic Clinical Tabular Data Generation via Constrained Optimization (https://arxiv.org/abs/2606.18518)
Comments:
          20 pages

- **Prior Approaches**: 기존 합성 EHR 생성은 Synthea, medGAN, CTGAN 계열 등에서 현실적인 샘플링을 노렸지만, 학습 데이터에 대한 memorization(기억)으로 인해 재식별 위험이 커질 수 있습니다. 반대로 DP처럼 사후적 differential privacy를 강하게 주입하면 잡음이 커져 특성 간 상관과 희귀 진단 패턴이 깨져 임상적 효용이 떨어지는 경우가 많았습니다. 즉, 개인정보와 임상 유용성의 trade-off를 “훈련/생성 과정에서 동시에” 제어하는 원칙적 방법이 부족했습니다.

- **Core Contribution**: PSyGenTAB은 합성 의료 데이터를 “제약 최적화(constrained optimization)”로 정식화하고, Augmented Lagrangian Method(ALM)로 프라이버시 하한을 만족하면서 임상 효용을 최대화하도록 설계했습니다. 프라이버시-유용성 균형을 고정된 타협이 아니라, 데이터 사용 기관이 설정하는 Pmin 같은 구성 가능한 제약으로 학습/샘플링에 내장합니다. 또한 특정 생성기 구조를 바꾸지 않고 generator 주변을 감싸는 model-agnostic 프레임워크로 확장성을 확보했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 임상에서 의미 있는 분포·상관·소수 클래스 신호를 유지하면서 (2) 실환자 기록과의 과도한 근접/중복을 막는 동시에 (3) 이를 단일하고 재현 가능한 지표 체계로 묶는 것입니다. PSyGenTAB은 임상 효용 UU를 주변 분포 일치, 희귀 이벤트 보존, 특성-특성 연관, 다양성/커버리지, 예측 효용(TSTR/TRTS) 등 다차원으로 구성하고, 프라이버시 PP도 DCR(가장 가까운 기록과의 거리), 분포 과적합, 중복/정확 재현률로 분해해 제약 형태로 ALM에 넣었습니다. ALM의 라그랑주 승수와 패널티 계수를 적응적으로 업데이트해 제약 위반이 감지되면 생성이 다시 privacy-feasible 영역으로 수렴하도록 유도합니다.

- **Empirical Impact**: 여러 임상 벤치마크에서 PSyGenTAB은 소수 클래스 진단 패턴과 특성 간 임상 연관을 유지하면서도, Train-on-Synthetic/Test-on-Real 및 그 반대 프로토콜에서 합성 데이터 학습 성능이 실데이터 학습과 유사한 수준을 보였습니다. 프라이버시 감사에서는 exact record reproduction 감소와 membership inference attacks에 대한 회복력도 확인해, 유용성 저하 없이 재식별 위험을 완화하는 쪽으로 균형을 맞췄다는 점이 강조됩니다. 결과적으로 기관 간 협업에서 안전한 합성 데이터 공유를 위한 “원칙 기반의 제어 프레임워크”로서 의미가 큽니다.



### Neural Phase Correlation (https://arxiv.org/abs/2606.18496)
- **Prior Approaches**: 기존 dense correspondence/의료 등록은 SIFT류 디스크립터나 VoxelMorph 계열처럼 “두 이미지를 각각 인코딩”한 뒤 유사도나 회귀로 대응을 암묵적으로 찾는 방식이 주류였다. 이런 접근은 구조적 관계를 first-class로 모델링하지 않아, 실패해도 왜 실패했는지(어떤 가정이 깨졌는지) 진단이 어렵다는 한계가 있었다. Phase correlation은 예외적으로 두 관측 간 변환을 직접 다루지만, 고정된 Fourier basis가 전역 translation으로만 잘 작동한다는 제약이 있다.

- **Core Contribution**: 이 논문은 phase correlation의 핵심 원리(관계/변환을 직접 다루되, 실패를 구조적으로 설명 가능하게)를 “학습된 basis”로 일반화한다. 고정 Fourier basis 대신 두 개의 학습 필터 뱅크(Ψ, Φ)가 지역 변환이 분해되는 2차원 부분공간을 만들고, 그 위에서 변환이 planar rotation 형태로 작동한다고 가정한다. 아울러 같은 대수적 primitive를 비강체(non-rigid) 변형과 unitary dynamics까지 확장해, 관측 쌍만으로 대응(또는 스펙트럼)을 복원하는 프레임워크를 제시한다.

- **Technical Challenges**: 가장 큰 기술 과제는 “지역적으로 불변(invariant)인 부분공간을 실제로 잘 찾는가”와 “가정이 깨질 때 모델이 스스로를 망치지 않는가”였다. 논문은 이 문제를 closed-form 잔차 residual rk2(수식 기반)로 해결해, 각 위치/각 서브공간에서 회전 구조가 성립하는지 여부를 마스킹한다. 또한 training에서는 differentiable top-K 선택으로 잔차가 작은 필터 쌍만 남겨 specialist가 되도록 유도하고, ODE 반복 해석에서는 매 스텝 재평가로 progressive하게 부분공간 선택을 갱신해 큰 변형도 보정한다.

- **Empirical Impact**: ACDC cardiac-MRI 등록에서 제안 프레임워크는 Dice 지표가 이전 발표 베이스라인과 동등하거나 더 높고, 특히 ED→ES의 어려운 방향에서 강세를 보였다. CAMUS에서는 보조 scoring이나 adaptive-smoothness 같은 장치를 쓰지 않고도 state-of-the-art 성능에 도달했으며, residual 진단이 신뢰도 게이팅 역할을 내부적으로 제공함을 확인했다. 더 나아가 1-D quantum harmonic oscillator의 관측 쌍(시간 진화 파트너)만으로 Hermite eigenstate와 양자화된 에너지 레벨을 회복해, 영상 정합을 넘어 스펙트럼 복원의 범용성까지 실증했다.



### SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR (https://arxiv.org/abs/2606.18487)
Comments:
          14 pages, 6 figures. Accepted at the Deep Learning for Code (DL4C) Workshop at ICML 2026

- **Prior Approaches**: 기존 코딩 생성 후학습 파이프라인은 SFT 이후 RLVR(검증 가능한 보상)로 넘어가며, 보통 SFT 체크포인트 중 GRPO에 가장 유리한 것으로 보이는 pass@1이 높은 것을 선택합니다. 그러나 대규모에서는 pass@1이 후속 RL 성능을 잘 예측하지 못하고, SFT의 과적합/분포 압축이 RL 쪽 학습 신호를 망가뜨릴 수 있다는 문제 제기가 이어졌습니다.

- **Core Contribution**: 이 논문은 GRPO에서 쓰는 그룹 상대 advantage 신호가 SFT 과학습으로 인해 사실상 소멸(gradient vanishing)할 수 있음을 정식화합니다. 특히 이진 보상 하에서 그룹 내 advantage 분산이 p(1-p)(g-1)/g로 주어져, 일정 임계 p*(g) 아래로 내려가면 대부분 그룹이 보상을 동일하게 받아 그룹 상대 신호가 구조적으로 붕괴한다고 보입니다.

- **Technical Challenges**: 핵심은 “왜 pass@1이 높은 체크포인트가 오히려 GRPO 초기 학습을 망가뜨리는가”를 체크포인트 단위로 진단하고, 조기 실패를 실무적으로 막는 스크리닝 방법을 찾는 것입니다. 저자들은 SFT depth ladder로 Qwen2.5-Coder-3B(각 깊이 3 seed)에서는 SFT가 깊어질수록 pre RL pass@1은 오르는데 peak GRPO pass@10은 0.806→0.481로 단조 하락하며, 이 차이를 출력 엔트로피 붕괴와 보상 분산 붕괴(임계 p*(8) 하회)로 설명합니다.

- **Empirical Impact**: Qwen에서는 “가장 높은 SFT pass@1 선택”이 매번 최악의 GRPO 초기화로 이어지는 rank inversion이 재현됩니다(엔트로피 붕괴가 빠르게 진행되어 학습 신호를 일찍 소진). 반대로 DeepSeek-Coder-6.7B에서는 임계 p*(8)보다 충분히 높은 안전 영역에 머물러 rank compression(순서가 뒤집히지 않음)이 나타나 이론의 경계 예측을 대조 검증합니다. 또한 pre RL 엔트로피 triage에 early GRPO 엔트로피 모니터를 결합한 2단계 진단이 실패 위험 체크포인트를 조기에 플래그하고, KL/label smoothing 같은 간단한 사후 개입은 붕괴 체크포인트를 구제하지 못해 실패 원인이 GRPO 하이퍼파라미터보다 SFT 단계의 엔트로피 고갈에 있음을 시사합니다.



### MagpieTTS-LF: Inference-Time Long-Form Speech Generation Without Training on Long-Form data (https://arxiv.org/abs/2606.18485)
- **Prior Approaches**: 기존 long-form TTS는 짧은 발화에선 자연스럽지만, 문장 단위로 쪼갠 뒤 이어붙이면서 prosodic drift(운율 드리프트), 화자 일관성 저하, 문장 경계 삐걱거림(energy discontinuity, warble)이 커지는 문제가 있었다. 이를 줄이려는 접근은 (1) 시퀀스 압축으로 컨텍스트를 맞추거나 (2) streaming/블록 생성으로 메모리를 제한하거나 (3) 문장 간 텍스트 맥락을 별도 모듈로 보강하는 방식으로 나뉘지만, 압축은 시간해상도 손실, 마스킹은 경계에서 정보가 “딱” 끊김, 학습 기반 개선은 기존 배포에 적용이 어렵다는 한계가 있었다. 또한 대부분의 방식은 긴 발화를 목표로 하는 아키텍처 변경이나 재학습이 필요했다.

- **Core Contribution**: 논문은 MagpieTTS를 재학습하지 않고도 긴 문단을 일관되게 생성하게 하는 inference-time 기법 MagpieTTS-LF를 제안한다. 핵심은 세 가지로, (1) 단조 정렬에 유도하되 과거/미래 토큰에 0이 아닌 가중치를 남기는 soft attention priors, (2) 청크를 넘어 상태를 이어주는 stateful inference, (3) 과거 텍스트 히스토리를 인코딩해 담화 수준의 prosody planning에 활용하는 history-aware text encoding이다. 결과적으로 문장 경계에서도 운율과 화자 특성이 덜 흔들리도록 만든다.

- **Technical Challenges**: inference-time에서 긴 생성의 안정성을 확보하려면, 학습 시 사용하던 정렬/주의 메커니즘을 무리하게 꺾지 않으면서도 청크 간 단절을 줄여야 한다. MagpieTTS-LF는 hard mask 대신 soft prior로 모노토닉 alignment를 “부드럽게” 유도해 먼 컨텍스트 정보를 점진적으로 감쇠시키고, 청크 생성 사이에 마지막으로 attended된 텍스트 위치와 인코더 컨텍스트(encoder hidden states), 텍스트 히스토리를 상태로 전달해 경계를 잇는다. 추가로 이전 텍스트 조각을 그대로 텍스트 인코더에 넣어 담화 맥락 기반의 발화 계획을 유지한다.

- **Empirical Impact**: 평가를 위해 Multilingual LibriSpeech 문단을 이어붙여 약 3~4분 길이의 20개 long-form 구간을 만들고, 정렬(WER/CER), prosodic continuity(경계 ΔF0/에너지 점프), 화자 일관성(embedding cosine similarity), 자연도(UTMOSv2)로 비교했다. 그 결과 MagpieTTS-LF는 WER/CER에서 전반적 최저치를 보였고, 문장 경계 에너지 불연속은 경쟁 대비 크게 줄어(약 14.04dB 수준) 경계 자연성이 가장 좋았다. 또한 시작~끝까지 화자 유사도가 안정적으로 유지되고 자연도 UTMOSv2도 변동 폭이 작아, 기존 long-form TTS의 대표 실패 모드를 폭넓게 개선했음을 시사한다. 코드와 벤치마크를 공개하며, 청크 기반 encoder-decoder TTS 전반에 inference-time로 확장 가능한 점에서 실용적 의미가 크다.



### Structured Representation Learning with Locally Linear Embeddings and Adaptive Feature Fusion (https://arxiv.org/abs/2606.18469)
Comments:
          Published in Transactions on Machine Learning Research (04/2026)

- **Prior Approaches**: 기존 RL은 보통 보상 최대화라는 단일 목표를 중심으로 상태 표현을 학습해, 표현이 즉시 보상 신호에 편향되고 환경 전이의 내재 구조(국소 기하)를 놓칠 수 있다는 한계가 지적돼 왔다. 또 manifold learning이나 SSL/contrastive 같은 표현학습은 많지만, 온라인 RL의 비정적 상태분포와 비동기 목적 때문에 이를 그대로 붙이기 어렵거나(특히 전역 학습/네거티브 샘플링), 이미지 중심 설계가 많아 물리 기반 로보틱스의 ‘이미 있는 구조’를 활용하지 못하는 경우가 많다.

- **Core Contribution**: 이 논문은 상태 표현을 ‘dynamics-specific(동역학용)’과 ‘reward-specific(보상용)’의 두 갈래로 분리해 학습하고, 두 표현을 self-attention으로 상태별로 적응적으로 융합하는 RL 프레임워크를 제안한다. dynamics-specific 쪽은 LLE(Locally Linear Embedding)로 환경이 만드는 국소 선형 구조를 보존하고, reward-specific 쪽은 SAC 같은 표준 RL 목표로 학습한 뒤, attention 게이팅이 어느 쪽이 의사결정에 더 중요한지 per-state로 선택한다.

- **Technical Challenges**: 핵심 기술 난제는 RL에서 발생하는 비정적 트래젝토리/리플레이버 분포 속에서도 LLE의 ‘국소 이웃 보존’ 가정을 안정적으로 유지하는 것이다. 저자들은 전역 kNN 탐색 대신 미니배치에서 trajectory-local 시간 이웃을 사용해 이웃을 구성하고, LLE 업데이트는 보상용 SAC 업데이트와 번갈아 수행(구조 손실은 주기적으로 업데이트)하면서 early stopping과 임베딩 붕괴 방지용 재구성 정규화를 적용해 계산·학습 간섭을 줄였다.

- **Empirical Impact**: Robosuite 등 물리 기반 로보틱스 벤치마크에서 SAC-LLE는 Recon/Next-state/Reward 예측/SPR/DBC 같은 보조손실 및 비교군보다 학습 효율과 최종 성능에서 개선을 보이며, attention weight를 통해 보상 또는 동역학 특징을 언제 더 의존하는지 시각화 가능한 해석성도 제시한다. 또한 LLE 기반의 ‘경량 인덕티브 바이어스’로 국소 구조를 명시 모델링하면서도 reward 학습과 충돌을 줄이는 설계가 효과적임을 보여, 구조학습형 월드모델·대조학습과는 보완적 방향의 가능성을 강조한다.



### What Does the Weight Norm Control in Grokking? Logit-Scale Mediation under Cross-Entropy (https://arxiv.org/abs/2606.18465)
Comments:
          16 papges, 10 tables and 4 figures. Code and data to reproduce all numbers, tables, and figures: this https URL

- **Prior Approaches**: 그로킹(grokking)은 학습 손실이 0에 가까워진 뒤에도 오랫동안 테스트 정확도가 우연 수준에 머무르다 급격히 개선되는 현상으로, 기존 연구들은 주로 weight norm(가중치 노름) 감소, 혹은 norm minimization 같은 ‘노름 기반’ 설명을 제안해 왔다. 또한 cross-entropy에서 logits가 과도하게 커지며 소프트맥스 포화/softmax collapse로 이어지는 수치안정성 관점도 제시됐다. 다만 weight norm과 logits(실질 logit scale)가 clamp(노름 고정) 과정에서 함께 변해, 둘 중 무엇이 지연의 직접 원인인지가 분리되지 않았다.

- **Core Contribution**: 이 논문은 clamp로 weight norm을 고정한 상태에서 출력에 non-trainable output temperature(출력 온도) τ를 도입해, 노름은 그대로 두고 effective logit scale(유효 logit 스케일)만 조절한다. 그 결과 cross-entropy에서는 노름이 늘어나며 생기는 그로킹 지연 대부분이 유효 logit scale을 되돌리면 재현(약 0.83~0.89)되며, 노름 효과의 큰 몫이 사실상 logit scale 채널을 통해 전달됨을 보인다. 반대로 mean-squared error(MSE)에서는 logit scale 채널이 거의 움직이지 않아 노름 효과가 다른 경로를 따른다는 점도 함께 분리된다.

- **Technical Challenges**: 핵심 과제는 ‘노름 고정’이 실제로는 rescaling에 의해 logits를 같이 키워 softmax 포화를 유발할 수 있다는 혼선을 제거하는 것이다. 논문은 temperature를 logits에 나누어 넣어 loss가 보는 스케일만 바꾸면서 norm clamp와의 인과 고리를 분리하고, grokking 시점의 유효 logit scale을 정밀 정의해 회귀 붕괴(collapse) 분석으로 정리한다. 또한 float64로 정밀도 감사(audit)를 수행해 softmax collapse가 수치 오류에 의해 가속될 수 있는 극단 구간을 제외하고, no-LayerNorm transformer 실험과 same-state forking으로 rescaling 아티팩트를 추가로 배제한다.

- **Empirical Impact**: cross-entropy 설정에서 다양한 norm-온도 격자 실험 결과, 그로킹 지연은 effective logit scale만으로 잘 설명되며 R2=0.97 수준의 높은 데이터 붕괴를 보인다(노름 용량은 추가로 1~2% 정도만 기여). 모듈러 덧셈(MLP, 두 가지 mod p)에서 temperature 조절로 지연이 유효 logit scale을 따라 ‘슬라이드’되며, memorization(기억화) 시간은 거의 변하지 않고 delayed-generalization(지연 일반화) 단계에만 영향이 집중된다. 요약하면, 그로킹 타임스케일의 근위(proximal) 매개변수가 weight norm이 아니라 logit scale 및 그로 인해 유도되는 softmax saturation 채널임을 강하게 뒷받침하며, 향후 수치안정성/로그릿공간 기반 설명을 더욱 구체화하는 데 의미가 있다.



### Veriphi: Attack-Guided Neural Network Verification with Dataset-Dependent Training Methods (https://arxiv.org/abs/2606.18454)
Comments:
          17 Pages, 8 Figures

- **Prior Approaches**: 기존에는 adversarial training(PGD)이 실험적 강건성만 제공하는 대신 복잡한 데이터셋에서도 잘 동작했고, certified training(IBP 등)은 수학적 보장을 제공하지만 정확도나 확장성에서 손해가 자주 지적됐다. 또한 CROWN 계열의 bound 기반 검증은 효율을 높였지만, “어떤 훈련법이 내 문제에 맞는가”에 대한 실증 가이드가 부족했다. 다중 아키텍처·다중 데이터셋을 동시에 비교해 선택 기준을 제시한 연구는 제한적이었다.

- **Core Contribution**: 이 논문은 Veriphi를 통해 certified training과 adversarial training의 효과가 데이터셋 복잡도에 따라 뒤집힐 수 있음을 체계적으로 보여준다. 특히 MNIST처럼 입력이 단순한 경우 IBP가 유의미하게 우세하지만, CIFAR-10처럼 복잡도가 높아지면 IBP 인증 성능이 사실상 붕괴하며 PGD가 훨씬 높은 certified accuracy를 달성한다. 또한 검증 시간을 줄이기 위해 공격 기반 조기 falsification을 결합해 실사용 전략을 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 공격으로 취약점을 빨리 찾는 것과 (2) 공격이 실패했을 때도 α,β-CROWN 기반의 공식 bound로 충분히 타이트하게 인증하는 것의 균형이다. Veriphi는 FGSM/I-FGSM으로 먼저 시간을 절약하고, 실패한 샘플에만 auto-LiRPA의 α,β-CROWN을 수행하는 two-phase 구조로 GPU 효율을 끌어올렸다. 더 나아가 α,β-CROWN의 더 복잡한 bound 계열이 현장에서 표준 CROWN 대비 개선 폭이 작다는 점도 실험으로 확인해 계산 비용 대비 이득을 정리했다.

- **Empirical Impact**: 실험에서 IBP는 MNIST에서 약 78% certified accuracy로 PGD를 앞서지만, CIFAR-10에서는 IBP가 1% 수준에 그친 반면 PGD는 58~94%의 인증 성능을 보였다. 공격-유도 falsification만으로 검증 시간을 5×(약 85% 시간 절감) 줄였고, 1억 파라미터급이 아닌 105.8M 파라미터 생산 모델(에어버스 Beluga 물류 최적화)에도 적용해 평균 2.6s 내 검증을 시연했다. 결과적으로 “certified training이 항상 adversarial training보다 낫다”는 가정을 반박하며, 검증 목표와 데이터 복잡도에 맞춘 훈련·검증 전략 선택이 중요하다는 메시지를 남겼다.



### TMR-GGNN: Credit Card Fraud Detection based on Time-Aware Multi-Relational Guided Graph Neural Network (https://arxiv.org/abs/2606.18444)
Comments:
          2025 2nd International Conference on Software, Systems and Information Technology (SSITCON), Pages 7

- **Prior Approaches**: 기존 신용카드 부정거래 탐지는 데이터 불균형이 심하고, 사기 패턴이 수시로 바뀌며, 거래 개체(고객·가맹점·단말·IP) 간 관계도 복잡하다는 한계가 컸습니다. 보통은 그래프 신경망이나 분류기에 의존하되, 시간 흐름과 이질적(heterogeneous) 관계를 충분히 반영하지 못해 드물게 발생하는 사기에서 성능이 흔들리는 경우가 많았습니다. 또한 오탐/미탐 손실 설계가 단순해 false negative를 줄이는 데 제약이 있었습니다.

- **Core Contribution**: 이 논문은 Timeaware Multi Relational Guided Graph Neural Network(TMR GGNN)라는 프레임워크로, 시간 창(window) 위에서 고객·가맹점·단말·IP 간 이질적 상호작용을 그래프 기반으로 학습하도록 확장합니다. 이를 위해 encoder-decoder 형태의 GNN 구조에 time-aware relational attention을 넣어 거래의 시간적 근접성과 의미적 문맥에 따라 관련도를 동적으로 가중합니다. 마지막으로 decoder에 대비학습 기반(contrastive learning) 모듈을 추가해 실제 거래와 합성(synthesized) 패턴을 구분하고 희귀 사기 케이스의 일반화 성능을 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 (1) 시간에 따라 관계가 변하는 상황에서 어떤 거래 링크가 현재 탐지에 유효한지 정교하게 가중해야 한다는 점, (2) 극심한 클래스 불균형 속에서 학습 신호가 왜곡되는 문제, (3) 미탐(false negative)을 줄이면서도 일반화 능력을 유지해야 한다는 점입니다. 논문은 멀티 릴레이션 그래프와 time aware relational attention으로 시간 근접도 기반의 가중치를 해결하고, 학습 손실은 InfoNCE 기반 대비 손실과 Focal Loss를 결합해 discriminative learning을 강화하며 false negative 완화를 노립니다.

- **Empirical Impact**: 초록 기준으로 TMR GGNN은 대비학습과 결합 손실을 통해 사기 식별 성능을 높이면서도 오탐을 줄이고 희귀 케이스의 일반화를 개선하는 방향으로 실증됐다고 설명합니다. 특히 실제 패턴과 합성 패턴을 대비해 모델이 “진짜 사기 같은” 신호를 더 잘 구분하도록 유도함으로써, 변화하는 사기 패턴 환경에서의 견고함을 기대할 수 있습니다. 결과적으로 금융 이상탐지에서 시간 인지형 멀티 릴레이션 그래프 + 대비학습/불균형 대응의 조합이 유효한 설계임을 시사합니다.



### CAOA -- Completion-Assisted Object-CAD Alignmen (https://arxiv.org/abs/2606.18429)
Comments:
          GitHub: this https URL

- **Prior Approaches**: 실내 RGB-D 기반 3D 시맨틱 재구성은 CAD 검색·object-CAD 정렬·레이아웃 추정으로 이어지며, 기존 object-CAD 정렬은 에너지 최적화나 end-to-end 파이프라인으로 대응해 왔다. 다만 잡음·불완전성·세그멘테이션 오류가 만들어내는 기하 왜곡 때문에 정확한 9-DoF(이동·회전·스케일) 추정이 흔들린다. 또한 점군 completion 모델들은 주로 합성 데이터에 학습·평가되어 real-world 일반화가 제한적이라는 문제가 누적되어 왔다.

- **Core Contribution**: 이 논문은 Completion-Assisted Object-CAD Alignment(CAOA)로, CAD를 정렬하기 전에 점군 completion으로 스캔의 불완전성과 잡음을 완화한 뒤 정렬을 수행하는 구조를 제안한다. completion 학습을 위해 실내 단일 객체용 real-world 벤치마크 S2C-Completion(8,500+ object-CAD pair)과 실내 특화 합성 데이터 SN-Indoor를 새로 설계했다. 더불어 대칭 모호성에 강하도록 symmetry encoder(SEM)와 symmetry-aware loss를 결합해 회전·스케일 추정의 견고성을 높였다.

- **Technical Challenges**: 핵심 난제는 (1) 실세계 점군 completion에 맞는 데이터 부재로 인한 합성-실세계 도메인 갭, (2) completion 결과가 장면 맥락을 무시할 때 크기·형상이 일그러지는 문제, (3) 객체 대칭 때문에 학습된 pose 특징이 부정확해질 수 있는 문제다. CAOA는 CAPCM에 context-aware completion(주변 컨텍스트 포인트를 함께 사용)을 적용하고, S2C-Completion과 SN-Indoor로 학습해 일반화를 개선했으며, SEM이 대칭 정보를 별도 임베딩으로 제공하고 Chamfer loss와 가중 L1로 symmetry-aware 학습을 수행한다.

- **Empirical Impact**: Scan2CAD 벤치마크에서 CAOA는 기존 SOTA 대비 class average 정확도를 약 17% 향상(전체 정확도도 약 16%)시키며, 대칭 처리와 completion 보조가 정렬 품질을 실질적으로 끌어올렸음을 입증한다. 또한 Ground Truth ScanNetv2를 사용하면 세그멘테이션 오류로 인한 정렬 성능 저하가 약 10% 수준임을 보여, 파이프라인 모듈화의 의미(세그 품질 개선 여지)도 강조한다. 데이터와 방법론(S2C-Completion 공개) 측면에서 실내 single-object completion/정렬 분야의 새로운 표준 벤치마크를 제공한 점이 파급력이 크다.



### From Specification to Execution: AI Assisted Scientific Workflow Managemen (https://arxiv.org/abs/2606.18425)
- **Prior Approaches**: 기존 WMS는 Pegasus, Nextflow, Galaxy처럼 실행·재현성·provenance에 강점이 있지만, 워크플로 설계/구현/디버깅은 대부분 수작업이라 고숙련 노동이 필요했다. LLM 기반 워크플로 생성은 자연어에서 파이프라인을 만들 수 있으나, 흔히 직접 코드 합성에 의존해 투명성·재현성·신뢰성이 떨어지고 WMS와의 통합이 약한 편이다. 또한 분산 오케스트레이션과 런타임 상호작용, 실패 복구까지 “끝까지” 다루는 체계는 제한적이었다.

- **Core Contribution**: 이 논문은 과학 워크플로 라이프사이클을 end-to-end로 잇는 AI-assisted 접근으로, (1) specification-driven 워크플로 생성, (2) LLM 디버깅 에이전트의 자동 진단·수정, (3) 분산 실행을 위한 Pegasus+MCP(통합 인터페이스) 구성을 제안한다. 특히 intent(의도)–design(설계)–implementation(구현)을 구조화한 명세 단계로 분리해, 코드 생성 전 검증 가능한 중간 산출물을 만든다. 이로써 생성 결과를 사람이 검토하고 반복 가능한 형태로 고정할 수 있다는 점을 핵심 기여로 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 복잡한 DAG 의존성과 계층적 실행에서 발생하는 실패를 런타임 로그·메타데이터 기반으로 정확히 원인 분해하고, 시스템 여러 층(설정/스케줄링/컨테이너/파일시스템)에서 안전하게 교정하는 것이다. 논문은 플러그인·skills 구조로 도메인 지식을 제공해 specification과 산출물 생성을 제약하고, 디버깅 에이전트가 로그 패턴 매칭 후 파라미터·리소스·실행 로직을 수정해 영향 서브작업을 재제출하는 closed-loop 복구를 구현했다. 또한 MCP 레이어를 통해 외부 클라이언트가 제출·모니터링·제어를 단일 인터페이스로 수행하도록 라우팅/노드 등록까지 설계했다.

- **Empirical Impact**: 의료 영상 federated learning 워크플로(서브워크플로 수천 잡, 병렬·반복·의존성 강함)로 평가했으며, 비전문가도 expert 수준 패턴(예: sub-workflow 기반 fan-out/fan-in)을 쓰는 워크플로를 더 적은 수동 개입으로 만들 수 있었다. 워크플로 생성 비교에서 specification-driven 방식을 쓴 Claude Code는 더 적은 세션으로 “구조적으로 완비된” 결과를 만들었고, 디버깅 에이전트는 디스크 고갈·정리 누락·실험 충돌·모델 설정 불일치·의존성 누락 등 다층 런타임 실패를 범주별로 해결해 개발 시간을 줄였다. 다만 federated learning 성능 자체는 중앙집중 학습 대비 격차가 컸는데, 이는 작은 per-client 데이터로 인한 최적화/가중치 평균의 한계가 지배적임을 수치로 확인하며, 워크플로 관리의 자동화 가능성과 과학 파이프라인 운영의 실무적 의미를 함께 보여준다.



### A Variational Framework for LLM Generator-Regulator Games (https://arxiv.org/abs/2606.18424)
- **Prior Approaches**: 기존 LLM 안전·규제는 보통 토큰 분포나 한 번 생성된 출력에 대한 후처리 규칙으로 평가/차단해 왔다. 이 방식은 “완성된 메시지” 관점의 분포 제약(예: 우회·기만·피싱 위험)이 어떻게 다음-토큰 샘플링에 반영되는지 설명하기 어렵다는 한계가 있었다.

- **Core Contribution**: 논문은 temperature-scaled autoregressive token sampling이 유도하는 ‘메시지 수준 확률분포’를 정식화하고, 이를 entropy-regularized Gibbs(깁스) 생성 분포로 재해석한다. 동시에 regulation을 단일 라벨/필터가 아니라, 생성 분포와 기준 분포 사이의 차이를 판별하는 ‘최적 discriminator’로 모델링해 saddle-point(안장점) 게임으로 구성한다.

- **Technical Challenges**: 핵심 난제는 토큰 단위 정규화가 시퀀스 길이와 접두사에 따라 달라지는 탓에, 실제로 생성되는 ‘완성 메시지 분포’를 다루기 복잡하다는 점이다. 논문은 메시지 수준에서 partition function 기반의 Gibbs 근사와 함께, regulator의 convex dual을 통해 f-divergence(일반화된 발산) 페널티로 연결되는 변분-판별(variational discriminator) 구조를 제시해 계산 가능한 equilibrium을 도출한다.

- **Empirical Impact**: 이론은 유한 어휘 case study 2개(검열 필터링, phishing defense)에서 utility, entropy, divergence, 수신자(receiver-side) 점수, 탐지 확률로 평가된다. 결과적으로 규제 강도와 엔트로피/유틸리티의 트레이드오프, 그리고 finite-length 조건에서의 ‘탐지 가능성’이 분포 수준에서 어떻게 달라지는지 명확히 보여, 안전·컴플라이언스 분야의 분포 제약형 설계에 실험적 가이드가 될 의미가 있다.



### Deep-Learning-Based Pixelated Microwave Filter Design and Characterization using Electro-Optical Electric-Field Measurements (https://arxiv.org/abs/2606.18402)
- **Prior Approaches**: 기존 마이크로파 필터 설계는 사전에 정한 토폴로지를 기반으로 반복적인 파라미터 튜닝을 수행해 왔다. 이 방식은 설계 공간을 제한하고 개발 시간을 늘린다는 한계가 있다. 한편, AI가 생성한 회로는 주로 S-parameter로만 성능을 검증해 전자기(EM) 관점의 작동 메커니즘을 해석하기 어렵다.

- **Core Contribution**: 본 연구는 픽셀화된 이진(binary) 레이아웃을 생성·최적화해 마이크로파 로우패스 필터를 자동 합성하는 접근을 제안한다. CNN이 S-parameters를 예측하고, 이를 유전 알고리즘(genetic algorithm) 평가와 결합해 목표 응답을 만족하도록 설계한다. 또한 electro-optical(EO) 전기장(전기장 패턴)을 실측해 AI 설계가 만들어내는 EM 거동을 직접 해석할 수 있게 했다.

- **Technical Challenges**: 핵심 난제는 (1) 픽셀 기반 구조의 거대한 조합을 학습 데이터로 커버하면서도 (2) 설계 성능을 전자기적으로 정확히 유도하는 것이다. 이를 위해 랜덤 레이아웃을 대규모로 생성해 CNN 학습에 사용하고(로테이션·미러 증강 포함), 검증 오차 3.2% 수준의 예측 모델을 만든 뒤 GA로 후보를 탐색한다. 해석 측면에서는 EO 전기장 측정이 RF 구동 제약으로 200 MHz–10 GHz로 제한되는데, 이를 정규·접선 성분용 EO probe 조합과 Pockels 효과 기반 벡터장 스캐닝으로 보완해 주파수별 패턴을 복원했다.

- **Empirical Impact**: 실험 결과 합성된 로우패스 필터는 시뮬레이션과 측정 S-parameters가 높은 수준으로 일치하며, 7 GHz 패스밴드와 9.5 GHz 이후 20 dB 이상의 억제를 보여준다. EO 측정은 통상적 S-parameter만으로는 보이지 않던 전기장 경로와 공진/정재파 특성을 드러냈고, 패스밴드에서는 전송선 유사 경로, 3-dB 전후에서는 스탠딩 웨이브, 스톱밴드에서는 입력 쪽 에너지 구속 및 모드 전이를 시사한다. 특히 AI 생성 구조에서 커플드 전송선 또는 stub-like 패턴과 유사한 emergent characteristic을 EO로 처음 관찰했다는 점에서 EM 설계 해석 체계를 한 단계 확장한다.



### Deep Learning-Driven Inverse Design of Doherty Power Amplifiers Using Pixelated Combiners and Dual-State Impedance Synthesis (https://arxiv.org/abs/2606.18395)
- **Prior Approaches**: 기존 Doherty PA 설계는 사전에 정한 회로 토폴로지(전송선, 인덕터, 커패시터 등)에 파라미터 모델을 얹고, EM 시뮬레이션 스윕으로 최적화하는 방식이 주류였다. 이 접근은 반복적이고 시간이 오래 걸리며, 국소해에 갇힐 위험이 크다. 한편 pixelated EM 구조와 deep learning inverse design이 일부 간단한 PA에 적용되긴 했지만, 3-port 로드-모듈레이션 combiner는 복잡도가 높아 연구 공백이 있었다.

- **Core Contribution**: 본 논문은 3-port Doherty combiner를 위해 deep CNN 기반 서러게이트와 GA를 결합한 설계 방법론을 제안한다. 핵심은 peak/back-off 두 동작 상태를 동시에 만족시키기 위한 dual-state impedance synthesis로, 합쳐진 출력 combiner가 로드-모듈레이션·임피던스 매칭·위상 보상을 한 번에 구현하도록 목표 임피던스를 정의한다. 이를 통해 기존에 어려웠던 “콤바이너가 Doherty 동작을 좌우한다”는 병목을 체계적으로 다룬다.

- **Technical Challenges**: 기술적 난제는 (1) pixelated 레이아웃(이진 금속/비금속)의 거대한 탐색 공간을 줄이면서 (2) 주파수별 S-파라미터를 정확히 맞추고 (3) 3-port 네트워크의 Doherty 동작 타깃을 정교하게 세우는 데 있었다. 저자들은 레이아웃을 이진 행렬로 표현하고, CNN이 주파수 2.4–3.0 GHz 구간의 S-파라미터를 sub-millisecond로 예측하도록 학습한 뒤 GA로 dual-state 목표에 맞는 레이아웃을 진화시킨다. 또한 결합기의 3-port를 ABCD와 임피던스(Z2P)로 변환하고, de-embedding된 기생/패키징을 반영해 peak에서는 Ropt, back-off에서는 2Ropt의 목표 실수부(허용 ±10%)와 허수부 최소화를 동시에 최적화한다.

- **Empirical Impact**: 검증을 위해 GaN HEMT로 3-port pixelated combiner가 포함된 Doherty PA 프로토타입 2종을 제작·측정했으며, 2.6–2.8 GHz에서 포화 출력전력 44.2 dBm 이상과 peak drain efficiency 71.2% 이상을 달성했다. back-off(6 dB)에서 drain efficiency는 최대 64%까지 측정되었고, 디지털 프리디스토션(DPD) 적용 후 ACLR은 프로토타입별로 -51.3 dBc보다 좋은 수준으로 개선됐다. 결과적으로 다중 포트 load-modulation PA에서 combiner 합성의 설계 시간과 탐색 효율을 끌어올릴 수 있는 실증 사례로 평가된다.



### Learning-Based Decision Making for Combustion Phasing Control in Multi-Fuel CI Engines with Latent Fuel Reactivity Estimation (https://arxiv.org/abs/2606.18393)
- **Prior Approaches**: 기존 접근은 CA50 제어를 위해 연료별 보정이 들어간 LUT를 쓰거나, closed-loop 압력 피드백·적응형 LUT·gain-scheduled PID로 일부 보정하는 방식이 주를 이뤘습니다. 그러나 cetane number(CN)가 시간에 따라 변하고 실시간 측정이 어렵기 때문에, 모델 기반 MPC나 adaptive 계열은 연료 반응성이 빠르게 변할 때 out-of-distribution 및 OOD 갭 문제를 피하기 어렵다는 한계가 반복적으로 드러났습니다.

- **Core Contribution**: 이 논문은 CN이 관측되지 않은 채 연속적으로 잠복 변동(latent variation)하는 상황에서 CA50 규제를 partially observable sequential decision problem으로 정식화합니다. 또한 GRU(게이트 순환 유닛) 기반 연료 반응성 추정 신호를 배우고, actor-critic 정책이 oracle CN이 아닌 추정 신호를 조건으로 제어하도록 설계해 기존 estimate-then-control 구조의 train-deploy 불일치를 줄입니다.

- **Technical Challenges**: 핵심 난제는 동일한 관측(예: 오차 및 관측 가능한 연소 신호)이라도 잠복 CN에 따라 최적 SOI/GPP가 달라지는 관측-상태 혼동(aliassing)과, contextual bandit의 i.i.d. 가정이 깨지는 정보구조 문제입니다. 논문은 fixed-window history augmentation이나 observation-only RL, generic recurrence만으로는 CN 변동이 빠를 때 충분히 복원되지 않음을 보이고, GRU로 연료 반응성을 압축 표현해 actor와 critic 모두에 조건화함으로써 추정오차에 강건한 정책을 학습합니다.

- **Empirical Impact**: 실험 멀티-퓨얼 엔진 데이터로 학습한 Gaussian-process surrogate 환경에서 비교했을 때, myopic/고정 히스토리 bandit은 CN 변화에 따라 성능이 악화되고 observation-only DDPG도 latent-state aliasing에 취약했습니다. 반면 제안한 GRU-guided RL은 unseen CN trajectory에서도 CA50 추적이 안정적이며 훈련셋 setpoint 기준 mean absolute tracking error가 0.25° CA 미만을 달성하고, SOI와 glow-plug power가 물리적으로 그럴듯한 형태로 매끄럽게 나타났습니다. 이는 연료 반응성 추정과 제어 정책 학습을 분리하지 말고, 배포 시점에 실제로 가능한 동일한 추정 신호를 정책에 통합해야 한다는 실무적 시사점을 제공합니다.



### LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents (https://arxiv.org/abs/2606.18388)
- **Prior Approaches**: RL post-training에서는 고정된 가이드북형 스케줄(응답 길이/rollouts/난이도 단계 증가, 또는 일부 진동)을 미리 정해 실행하는 경향이 강했습니다. 하지만 이런 방식은 “언제(트리거), 얼마나(폭), 무엇을(파라미터 조합)” 바꿔야 하는지 훈련 동역학이 바뀔 때 대응하기 어렵고, KL 스파이크·붕괴·정체 같은 이상 징후에 체계적으로 반응하지 못합니다.

- **Core Contribution**: 이 논문은 최적 multi-stage RL post-training에서 나타나는 구조적 비대칭을 제시합니다: capacity 파라미터(응답 길이, rollouts 등)는 단계가 진행될수록 단조 누적되는 반면, regularization 파라미터(학습률, KL 계수, temperature 등)는 훈련 역학 변화에 따라 주로 진동합니다. 또한 LLM 에이전트가 체크포인트별 동역학을 해석해 다중 파라미터를 “동시 전환”하도록 설계한 LLMZero를 통해 이 원칙을 찾고 검증합니다.

- **Technical Challenges**: 핵심 난제는 고정 스케줄로는 표현하기 어려운 비정상(non-stationary) 탐색-활용 트레이드오프를, 여러 하이퍼파라미터의 인과적 연동을 고려해 실시간 전환해야 한다는 점입니다. LLMZero는 MCTS(UCT)로 학습 궤적을 트리 탐색하고, 시각화·텍스트 지표를 결합한 proposer 에이전트가 병리 진단 후 3개 이상 파라미터를 조율한 전환을 제안하며, agentic early stopper로 유망하지 않은 가지를 중단해 탐색 비용을 줄입니다.

- **Empirical Impact**: 4개의 다양한 GRPO 태스크에서 LLMZero가 베이스 모델 대비 상대 9%~140%, grid search 대비 상대 6%~15% 개선을 보였고, 무작위 탐색과 skill-based 에이전트를 일관되게 능가했습니다. 특히 최적 전략을 12 iteration 내에 찾는 경우가 많아 반복 효율도 높았으며, SSMR-bench에서는 모델 크기(0.6B~8B) 전반에서 성능이 유지·확장되는 동시에 OOM 같은 인프라 실패까지 회피해 실무적 의미가 큽니다.



### RankGraph-2: Lifecycle Co-Design for Billion-Node Graph Learning in Recommendation (https://arxiv.org/abs/2606.18379)
- **Prior Approaches**: 기존 GNN 기반 추천은 그래프 구조·학습·서빙 중 한 단계만 따로 최적화하는 경우가 많아, billion-node 규모에서 병목이 전체 성능을 제한한다. 또한 많은 산업 시스템은 그래프가 주어진다는 가정 하에 학습은 분산 인프라로 해결하지만, 실시간 서빙 비용(online KNN/ANN) 문제는 충분히 다루지 않는다. 그래서 대규모 그래프 구축과 학습 목표, 그리고 서빙을 함께 설계하는 라이프사이클 co-design 관점이 부족했다.

- **Core Contribution**: RankGraph-2는 서빙·학습·그래프 구축을 동시에 “코설계”해, 한 단계의 요구가 다른 단계의 설계를 제약하도록 만드는 프레임워크를 제안한다. 특히 similarity-based retrieval(U2U2I, U2I2I)에서 online KNN을 없애기 위해 학습 단계에서 cluster index를 공학적으로 함께 co-learn 하도록 설계했다. 그 결과 더 단순한 아키텍처로도 높은 recall과 실제 지표 개선을 동시에 노린다.

- **Technical Challenges**: 대규모에서의 핵심 난제는 (1) 수백 조 잠재 엣지에 달하는 그래프를 1시간 내 재구성 가능한 형태로 줄이는 것, (2) 서빙 정확도를 유지하면서 online graph infrastructure와 KNN 탐색을 제거하는 것, (3) 학습 단계에서 인덱스 품질을 목표 함수에 반영하는 것에 있다. RankGraph-2는 popularity bias correction을 포함한 엣지 subsampling으로 간선을 수백억 단위로 줄이고, backbone 그래프에서 personalized PageRank(PPR)로 multi-hop 이웃을 사전 계산해 self-contained 학습 데이터를 만든다. 이어 residual-quantization 기반의 co-learned cluster index를 학습에 직접 결합해 KNN 없이도 재구성 오차와 retrieval 목적을 동시에 만족시키도록 한다.

- **Empirical Impact**: RankGraph-2는 오프라인 recall에서 GAT + Deep Graph Infomax 대비 3.8배, PyTorch-BigGraph 대비 2.1배 향상을 보였다. 14일 A/B 테스트에서는 CTR 최대 +0.96%, CVR 최대 +2.75% 개선을 기록했으며, U2U 서빙 인프라 비용은 83% 절감했다. 나아가 Meta의 주요 서피스에서 20회 이상 retrieval 런치에 적용되며 실사용 관점의 효과를 입증했다.



### Redact or Keep? A Fully Local AI Cascade for Educational Dialogue De-Identification (https://arxiv.org/abs/2606.18372)
- **Prior Approaches**: 기존 교육용 de-identification 연구는 개인정보보호 거버넌스와 정확도 사이에서 트레이드오프에 자주 부딪혔습니다. 상용 LLM API는 모호성을 잘 처리하지만 제3자 전송 이슈가 있고, 로컬 NER은 빠르지만 curricular(교육과정) 용어와 실제 학생 이름이 겹칠 때 과도한 과(過)차단(over-redaction) 경향이 있었습니다. 또한 의료 분야처럼 “표면 형태=개체”라는 가정이 교육 대화에는 그대로 적용되기 어렵다는 점이 반복 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 교육 대화 de-identification을 open-ended entity recognition이 아니라 “constrained privacy triage(제한적 프라이버시 판정)”로 재구성한 fully local cascade 프레임워크를 제안합니다. Stage 1은 recall-first로 후보 span을 넉넉히 과생성(precision 희생)하고, Stage 2 reviewer가 각 후보를 문맥과 발화자 역할을 바탕으로 Redact/Keep의 이진 결정을 내립니다. 즉, “무엇을 찾아낼지”보다 “그게 정말 개인정보 위험인지”를 핵심으로 옮긴 것이 기여입니다.

- **Technical Challenges**: 핵심 기술 난제는 curricular-personal name ambiguity처럼 같은 표면형이 서로 다른 의미(수학 개념 vs 실제 학생)를 가질 때, 단순 탐지로는 정확한 판정이 불가능하다는 점입니다. 저자들은 union proposer(DeBERTa+ModernBERT 인코더 + 규칙 기반 RegEx)를 통해 후보를 과생성한 뒤, reviewer를 cascade-aligned 이진 라벨(실제 PII면 Redact)로 학습해 “탐지→프라이버시 검증” 구조로 난제를 분해합니다. 그 결과 small local 모델도 각 후보에 대해 문맥 기반 Redact/Keep을 수행하며, 전체 파이프라인은 단일 노트북에서도 동작하도록 설계했습니다.

- **Empirical Impact**: 수학 튜터링 대화 두 플랫폼(영어)에서 실험한 결과, 가장 강한 구성인 Union + Gemma 31B는 canonical 테스트에서 macro F1 0.958을 달성했습니다. 동일 모델을 단일 패스 LLM-only detector로 썼을 때 macro F1 0.767, 상용 API(Gemini 3.1 Pro) baseline은 0.706에 그쳤고, 전체 과정은 제3자 API 없이 단일 랩탑에서 실행됩니다. 특히 curricular–personal 이름 모호성이 공존하는 challenge set에서는 성능 저하가 0.03 F1로 매우 작았는데, 이는 “교육 de-identification은 문제 재정의가 모델 스케일보다 중요”하다는 결론을 뒷받침합니다.



### Guava: An Effective and Universal Harness for Embodied Manipulation (https://arxiv.org/abs/2606.18363)
- **Prior Approaches**: 기존 접근은 vision-language-action(VLA)로부터 입력(영상·언어)→로봇 행동을 end-to-end로 생성하거나, VLM을 고수준 추론기로 두고 별도 지형/프리미티브 기반 인터페이스로 실행하는 방식이 주류였습니다. 다만 이들 방식은 로봇 시연 데이터 의존성이 크거나, 행동이 한 번에 생성되어 실행 실패 후 재계획·복구가 어렵고, 계획을 명시적으로 점검하며 수리하기도 난해하다는 한계가 있습니다. 또한 harness 형태의 modular tool use가 시도되었지만, one-shot 프로그램 생성·실행 중심이라 long-horizon에서 실패 회복을 지속적으로 수행하기 어렵다고 정리합니다.

- **Core Contribution**: 이 논문은 embodied tool use를 위한 harness 프레임워크 Guava를 제안하고, 유효한 harness의 설계 재료를 반복형 워크플로우·행동 추상화·멀티모달 관측의 3요소로 정리합니다. 핵심은 언어 모델이 외부 모듈(지각/계획/제어)을 호출하도록 구성하되, ReAct 스타일의 반복 루프로 실행 결과를 관측하며 사고-행동을 교차시키는 것입니다. 이를 통해 다양한 reasoning 모델 전반에 걸쳐 조밀한 end-to-end 정책 대신 “모델-비의존적 인터페이스”로 embodied manipulation 역량을 이식할 수 있음을 목표로 합니다.

- **Technical Challenges**: 가장 큰 난제는 (1) 폐루프 상호작용에서 발생하는 실패·상태 이탈을 견디는 도구 설계와 워크플로우를 찾는 것, (2) compact 모델로의 증류를 위해 소량 고품질 시연을 확보하는 것입니다. Guava는 semantic action abstraction(예: grasp(), align(), home_pose())으로 저수준 기하/물리 추론 부담을 언어 모델에서 덜어주고, 시각 관측과 텍스트 기반 상태 표현을 함께 넣어 그라운딩 모호성을 줄이며, 실행 피드백을 반영하는 iterative perception-reasoning-action 루프로 복구를 강화합니다. 또한 2K 미만 시뮬레이션 궤적만으로도 frontier VLM의 tool use를 4B 모델로 증류하도록, failure perturbation 기반 recovery 데이터와 SFT+GRPO(희소 성공 보상 RL)를 결합한 데이터 효율 학습 파이프라인을 구성합니다.

- **Empirical Impact**: 실험 결과 Guava-Agent-4B는 시뮬레이션(Robosuite)과 실세계(Franka Research 3)에서 모두 높은 성공률을 보이며, 종단 전략이나 one-shot 코드 실행 기반 경쟁 접근보다 long-horizon에서 우수한 복구 성능을 보였습니다. 특히 2K 미만 시뮬레이션 궤적로 학습한 4B open-source 모델이 proprietary frontier VLM과 유사한 수준의 전반 성능을 내고, 미지 물체·새 지시·긴 과제 구성에서도 강한 일반화를 보였다는 점이 강조됩니다. 결론적으로, 잘 설계된 harness가 데이터 적게 써도 compact 모델에서 emergent한 embodied 역량(복구 포함)을 끌어낼 수 있는 확장 가능한 인터페이스가 될 수 있음을 실증하며, 이후 모델 크기/데이터/툴셋 확장 가능성도 제시합니다.



### SafeClawBench: Separating Semantic, Audit-Evidence, and Sandbox Harm in Tool-Using LLM Agents (https://arxiv.org/abs/2606.18356)
Comments:
          32 pages, 5 figures

- **Prior Approaches**: 기존 LLM 안전 평가는 주로 위험한 텍스트 생성 여부를 단일 성공률로 측정해 왔고, 에이전트 보안 평가도 특정 공격 유형이나 일부 채널만 다루는 경우가 많았다. 이 때문에 모델이 공격자 지시에 ‘동의했는지(semantic)’와 실제로 ‘관측 가능한 피해(evidence/state harm)’가 발생했는지를 구분하기 어렵다. 또한 툴 호출·상태 변화·지속 메모리 같은 에이전트 고유 실패 단계를 하나의 지표로 뭉개는 한계가 남아 있었다.

- **Core Contribution**: 논문은 SafeClawBench를 제안하며, 도구 사용 언어모델 에이전트 보안 실패를 단계별 엔드포인트로 분해해 측정한다. 총 600개의 통제된 적대 과제를 6개 공격 패밀리(직접/간접 프롬프트 인젝션, tool-return 인젝션, memory poisoning, memory extraction, 모호성 유발 unsafe inference)로 구성하고, semantic attack acceptance뿐 아니라 audit-visible harm evidence, sandbox-observed tool/state harm을 각각 별도로 보고한다. 이를 통해 텍스트 컴플라이언스와 실행 가능한 상태 변경/비밀·정보 유출의 관계를 혼동하지 않게 한다.

- **Technical Challenges**: 핵심 기술적 난점은 ‘의미적 실패’와 ‘증거 기반 피해’ 또는 ‘실제 실행 관측 피해’가 서로 불일치할 수 있다는 점을, 공정한 분모·게이팅으로 재현 가능하게 분리해 내는 것이다. SafeClawBench는 Semantic Core(LLM judge로 semantic compromise 판정)와 Core-gated harm evidence audit(보호 객체/접근/행동/지속에 대한 정형 증거 검증), 그리고 Exec-Balanced(격리 샌드박스에서 툴 호출·상태 오라클로 deterministic harm 관측)를 분리된 프로토콜로 수행한다. 또한 12,000개 model–policy–case 조합에서 CoreFail과 ExecHarm의 매칭을 이용해, semantic 판정을 통과했는데도 샌드박스에서 피해가 관측되는 CorePass∧ExecHarm 같은 불일치 패턴을 계량한다.

- **Empirical Impact**: 실험 결과, D0(추가 프롬프트 보호 없음)에서 semantic 실패율이 모델별로 9.0%~44.2%까지 크게 벌어졌고, 같은 semantic failure라도 audit-visible harm evidence로 좁혀지는 정도가 더 제한적이었다. 특히 별도의 실행 프로토콜에서 일부 과제는 Semantic Core 호출을 통과했는데도 sandbox harm이 발생했으며, 매칭 분석 12,000행 중 347개 ExecHarm 중 291개(83.9%)가 Core를 통과한 행에서 나왔다. 즉, 단일 공격 성공률이나 텍스트 판정만으로는 에이전트 보안 위험의 성격을 설명하기 어렵다는 점을 SafeClawBench가 정량적으로 보여준다.



### Self-CTRL: Self-Consistency Training with Reinforcement Learning (https://arxiv.org/abs/2606.18327)
Comments:
          34 pages, 12 figures, includes appendices

- **Prior Approaches**: 기존 대다수 언어모델 학습은 각 프롬프트에 대한 정답/호응을 주로 최적화해, 설명과 실제 행동이 서로 다른 맥락에서 생성될 때 생기는 불일치(설명 비충실성)를 잘 잡지 못한다. 안전/정책 영역에서는 Constitutional AI나 RLAIF처럼 LM-판사를 통한 학습이 있으나, 설명과 행동을 “같은 샘플링 루프”에서 함께 맞추는 방식은 제한적이었다.

- **Core Contribution**: 이 논문은 Self-Consistency Training with Reinforcement Learning(Self-CTRL)로, 메타 수준 설명(자기 규칙/편향 보고)이 객체 수준 행동(실제 응답/샘플링)을 예측하도록 일관성을 직접 최적화한다. 설명 업데이트 방향(explanation training)과 행동 업데이트 방향(behavior training)을 각각 또는 함께 조합해, “자기설명 예측력”과 “정렬(안전) 개선”을 동시에 노린다는 점이 핵심이다.

- **Technical Challenges**: 핵심 난제는 (1) 자연어 규칙처럼 설명이 직접적인 likelihood로 평가되지 않거나, (2) 행동과 설명이 서로를 관측하지 못한 채 다른 입력에서 생성된다는 점에서 일관성 점수를 설계·학습해야 한다는 것이다. 논문은 독립 샘플된 설명-행동 쌍에 대해 외부 시뮬레이터/LM judge로 일관성 점수 phi를 만들고, GRPO 스타일 정책경사로 두 분포(p(LM|x), p(LM|x_meta))를 동시에(또는 한쪽만) 밀어 학습을 안정화한다.

- **Empirical Impact**: 확률 추론 벤치마크(편향된 동전/모형화)에서는 자기 보고 편향과 실제 샘플링 기반 잠재 편향의 상관이 R^2=0.24에서 R^2=0.64로 크게 상승했으며, held-out 동전에서도 거의 기준선급 일반화 성능을 보였다. Constitutional AI 설정에서는 설명-거절의 시뮬레이션 정확도가 auditor 모델 기준으로 36%→92%로 개선되고, HarmBench 공격 성공률은 15.0%→0.5%로 낮아져 설명의 예측가능성과 안전성 모두에서 의미 있는 효과를 입증했다.



### Agentra: A Supervisable Multi-Agent Framework for Enterprise Intrusion Respons (https://arxiv.org/abs/2606.18325)
- **Prior Approaches**: 기존 기업 침해대응은 IDS/EDR/XDR 알림을 SOAR의 정적 playbook으로 매핑하고, 사람이 triage하며 실행을 승인하는 구조에 의존해 대응 지연이 생깁니다. 또한 playbook 기반은 미탐/오탐이나 전술 변화, 공격 표면 간 연계에 덜 적응적이고 운영 중 규칙 유지 부담이 큽니다. LLM 기반 접근은 이유·액션 제안 능력이 있지만, 검색된 위협지 정보와 모델 출력이 모두 untrusted일 수 있어 다층 감독(검증·중재·승인·감사기록)이 필요하다는 한계가 강조됩니다.

- **Core Contribution**: Agentra는 IDS, EDR, XDR 알림을 MITRE ATT&CK·MITRE D3FEND·NIST CSF 2.0에 근거한 구조화된 incident response plan으로 변환하는 supervisable multi-agent IRS 프레임워크입니다. Planner가 후보 plan을 만들고, Validator가 bounded review 루프로 안전성을 검토하며, Moderator가 검색·증거를 security gateway에서 중재해 untrusted 입력이 계획에 직접 스며들지 않게 합니다. 마지막으로 Action Catalog와 risk score로 실행 범위를 게이트하고, 결정은 append-only audit log로 무결성 있게 남깁니다.

- **Technical Challenges**: 핵심 과제는 (1) TTP 오매핑·잘못된 근거로 인한 과잉 대응, (2) threat intelligence/검색 내용의 prompt-injection·데이터 포이즈닝, (3) 모델이 만들어내는 D3FEND 식별자나 행동이 온톨로지와 환경 상태에 맞는지 불안정하게 판별되는 문제입니다. Agentra는 Moderator로 retrieval을 sanitize하고, Validator 다중 관점 승인(k-of-n)과 san ity/risk/StateChanged 체크로 unsafe plan이 실행 측 API로 넘어가지 않게 구성했습니다. 또한 실행 전엔 Action Catalog에 있는 동작만 허용하고, 실행·검증·결정의 전체 흐름을 Merkle-chain 기반 append-only audit substrate로 추적합니다.

- **Empirical Impact**: 120개 이벤트(ThreatHunter-Playbook, Splunk BOTSv3, DARPA OpTC)에서 Agentra는 static OASIS CACAO v2.0 기준선 대비 FP-aware IRS F1을 0.61에서 최대 0.84로 끌어올렸습니다. 특히 Planner-only 설정은 overreaction을 유발해 위험한 행동 비율이 16%까지 오를 수 있었지만, Moderator/Validator/Graph-RAG 등 안전·근거 레이어를 추가하면 projected harmful-action rate를 기준선(0.0%) 수준으로 되돌렸습니다. 이는 “다중 에이전트 계획이 온톨로지 기반 응답 범위를 넓히면서도 analyst 승인과 감사가능성을 유지할 수 있다”는 실증 신호로, SOC 자동화의 실행 신뢰성 설계에 의미 있는 참고점이 됩니다.



### Why SWAVE May Not Be All You Need:A Concept-Evolution Retrospective on Complex-Valued Recurrent Language Models (https://arxiv.org/abs/2606.18324)
- **Prior Approaches**: Transformer는 O(N^2) 계산과 O(N) KV 캐시로 긴 컨텍스트가 비용 부담이 컸다. RWKV, Mamba/S4 같은 선형 순환 모델은 이 비용을 줄이지만, 상태가 매 스텝 지수 감쇠하며 초기가 잊히는 문제가 있었다. 복소수 hidden state를 쓰는 접근은 이론적으로 norm 보존을 기대할 수 있었지만, 실제 언어모델에선 출력 헤드/스캔/정규화 같은 구조가 조금만 틀어져도 학습 붕괴로 이어질 수 있다는 실전 지침이 부족했다.

- **Core Contribution**: SWave는 복소수 recurrent language model로, O(1) 메모리 추론을 유지하면서 SSM의 핵심 약점인 decay로 인한 장기 망각을 줄이려는 설계를 제시한다. 특히 Cayley-parameterised unitary 전이와 회전(축소가 아닌 페이즈 회전) 중심의 hidden state를 통해 신호 무결성을 장기 컨텍스트까지 보존한다는 목표를 전면에 둔다. 또한 cos-domination collapse를 구조적으로 규정하고, 이를 실제 학습 안정성으로 연결한 출력 헤드(PAM 기반 untied head) 개선 과정을 정리한다.

- **Technical Challenges**: 가장 큰 장애물은 “구조적으로는 더 좋아 보이지만 전역 최소에서 채널이 붕괴되는” 출력 헤드의 cos-domination collapse로, tied 공진/위상 파라미터화에서 cross-entropy 최적해가 imaginary 채널 소거에 고정될 수 있음을 증명했다. 이를 PAM의 독립 real/imag embedding 테이블로 바꿔 degenerate minimum을 제거하고, 200,000-step 학습에서 채널 RMS 비율을 안정 범위로 유지하는 방식으로 해결했다. 구현 측면에선 병렬 associative scan을 log-space로 설계해 수치 안정성을 확보하고, ComplexNorm을 residual/sublayer 샌드위치로 배치해 위상-기울기 불안정이 |h|^2 스케일로 증폭되는 문제를 차단했으며, 일부 복소 채널 믹서/고정 retention 개념은 성능 이득이 없어 실제 경로에서 철회·대체했다.

- **Empirical Impact**: 단일 설정(D=384, L=16, T=2048)에서 SWave는 FineWeb-Edu로 2×H100 NVL 학습을 수행하며, 출력 헤드 구조 변경으로 200,000-step까지 학습 안정성을 확인했다(최선 step PPL 22.0). 무엇보다 이 논문은 “새로운 SOTA” 주장보다, cos-domination collapse 같은 실패 모드를 수식으로 규정하고 plan-to-code 수준의 추적 방법까지 제시해, 복소수 recurrent 학습에서 반복되는 구조적 붕괴를 조기 탐지하는 실무적 임팩트를 노린다. 결과적으로 복소 recurrent 학습의 이식 가능한 공학 원칙(스캔 병렬화, log-space backward, 위상 보존 정규화, 헤드 분리 설계 등)과 반례/철회 판단 기준을 제공한다.



### SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior (https://arxiv.org/abs/2606.18322)
Comments:
          Code: this https URL, Project page: this https URL

- **Prior Approaches**: Sparse Autoencoders(SAE)가 잔차 스트림을 해석 가능한 희소 특징으로 분해한다는 점에 기대, 최근 latent-space defense들은 ‘위험/불필요’ SAE 특징을 찾아 clamp나 suppression을 수행해 동작을 막는 방식을 택해왔다. 이 접근은 해당 SAE 특징이 곧 행동을 끊는 실행 가능한 제어 손잡이라는 강한 기계론 가정을 깔고, 특징을 억제하면 행동이 완전히 사라질 것이라 본다. 하지만 특징 수준의 차단이 행동의 근본 생성 능력을 제거하는지, 혹은 우회 경로만 바꾸는지의 검증은 상대적으로 부족했다.

- **Core Contribution**: 논문은 SAE 기반 개입이 진정한 ‘완전 병목’인지 진단하기 위해 post-intervention recovery를 제안한다. 핵심은 클램프가 이미 적용된 뒤(방어가 돌아가는 상태 그대로) 잔차 공간에서 소규모 섭동을 최적화해, 원래의(클램프 전) 행동을 복원할 수 있는지 묻는 것이다. 또한 회복이 일어나도 클램프 대상 SAE 특징 값은 유지되도록 제약을 걸어, 성공이 단순히 방어를 되돌리는 꼼수인지 분리해낸다.

- **Technical Challenges**: 가장 큰 기술적 도전은 ‘클램프를 다시 건드리지 않고도’ 행동을 복원할 수 있는 우회 경로를 찾는 최적화의 제약을 어떻게 설계하느냐다. 단일 레이어 개입에서는 선택된 SAE encoder 방향에 직교하는 공간으로 업데이트를 투사(Projected Gradient Descent)해, 클램프된 특징을 정면으로 재활성화하지 않도록 만든다. 다중 레이어(예: refusal)에서는 섭동이 이후 레이어에서 방어 특징을 다시 움직일 수 있어, feature-map Jacobian을 이용해 방어 특징 맵 전반에 대한 1차 변화를 차단하는 동적 투사를 적용한다.

- **Empirical Impact**: TPP, WMDP-Bio unlearning, IOI, refusal steering까지 네 가지 설정에서 ‘특징 레벨 clamp는 성공했지만 행동은 회복 가능’이라는 취약 모드가 반복 확인된다. 특히 안전에 직결되는 refusal-steering에서 strict-valid AdvBench 프롬프트 기준 회복률 95.8%를 달성하면서도 방어 특징의 relative drift는 0.131로 낮게 유지했고, 회복 경로는 클램프된 SAE 특징 재개보다는 SAE reconstruction residual( SAE가 설명하지 못한 잔차 )에 주로 국한됨을 보였다. 즉 SAE 특징은 인과적 국소 손잡이로 유용할 수 있지만, 그것만으로 행동 소거가 보장되지 않는다는 ‘feature-level control vs behavioral completeness’의 격차를 실증적으로 드러냈다.



### ASTRA: A Scalable Next-Generation ATCO Training Simulator with Autonomous Simpilots (https://arxiv.org/abs/2606.18319)
- **Prior Approaches**: 기존 ATCO 훈련은 ‘simpilot’(시뮬레이션 조종사) 같은 전문 인력이 훈련생과 대화하며 조종사/관제사 역할을 동시에 연기하는 방식에 크게 의존해 왔습니다. 또한 자동화된 대화 시스템은 Western-centric 음성모델을 기반으로 해 Singaporean-accent 및 항공 전문용어 인식에서 성능이 급락하고, 결과적으로 WER이 높아지는 문제가 지적됩니다.

- **Core Contribution**: ASTRA는 시뮬레이션 조종사 역할을 자동화한 end-to-end 훈련 시뮬레이터로, ATCO 훈련생 발화를 ASR로 전사한 뒤 CIU(지시 이해)와 응답 생성, TTS를 거쳐 적절한 조종사/관제사 응답을 생성합니다. 로컬로 적응된 음성모델과 domain hotword/전문용어 정규화까지 결합해 Singapore 운영 맥락에서의 대화를 안정화합니다. 아울러 radiotelephony 커뮤니케이션을 정확성·간결성·완전성 중심으로 AI가 평가/피드백하는 프레임워크도 포함합니다.

- **Technical Challenges**: 주요 기술적 난관은 (1) Singaporean-accent 항공 음성에서의 ASR 정확도, (2) radiotelephony 특유의 호출부호·항로지점·고도/활주로 표기 같은 도메인 용어를 문장 의미와 함께 안정적으로 처리하는 것, (3) 발화 톤/발음이 일정한 TTS를 실시간성 있게 합성하는 점입니다. ASTRA는 잡음 억제·VAD·도메인 hotword 바이어싱·Singapore 관련 합성 데이터 기반 fine-tuning으로 ASR 파이프라인을 최적화하고, CIU/응답 생성/평가까지 DSPy-LLM 하이브리드로 구조화해 오류 전파를 줄입니다. TTS는 LoRA 기반 parameter-efficient fine-tuning과 chunk 단위 스트리밍으로 낮은 지연과 발음 안정성을 함께 노립니다.

- **Empirical Impact**: 논문은 특히 ASR에서의 개선을 WER 23.45%로 제시하며, Singaporean-accent 항공 음성에서 기존 오프더셸 대비 큰 폭으로 성능을 끌어올렸다고 보고합니다. 또한 AI-assisted performance evaluation에서 최적화 이후 정확성/간결성/완전성 점수를 각각 91.7%, 88.2%, 86.9%로 달성해, 사람 평가에 가까운 정량 피드백 제공 가능성을 시사합니다. 결과적으로 ASTRA는 simpilot 인력 의존도를 낮추면서도 표준화된 평가와 확장 가능한 ATCO 훈련 운영을 가능하게 하는 방향성을 제시합니다.



### Ghost Attractor Networks: Basin-Structured Dynamical Decoders for Closed-Loop Sequential Generation (https://arxiv.org/abs/2606.18315)
- **Prior Approaches**: 기존 순차 생성에서는 Transformer key-value cache, iterative diffusion decoder, in-context learning처럼 토큰을 누적하는 구조가 많아 메모리와 per-step 연산 비용이 시간 지평선에 따라 커진다. 이들은 효율은 떨어지면서, mode 전환도 충분한 컨텍스트 축적 없이는 즉시 일어나기 어렵다. 작은 feed-forward 디코더로 바꾸면 속도는 회복되지만 위상(phase) 조건화나 latent z의 시점 간 carry-over가 요구하는 ‘안정적인 잠재(latent) 기하’가 부족해 closed-loop 제어 성능이 무너진다.

- **Core Contribution**: Ghost Attractor Networks(고스트 어트랙터 네트워크, Ghost)는 상기의 두 결함(메모리/지연 문제와 latent 기하 문제)을 동시에 겨냥한 dynamical decoder다. 학습된 potential의 basin-어트랙터 구조를 기본으로 만들고, context 변화가 saddle-node bifurcation을 통해 basin을 전환하도록 설계해 디코더 레벨의 단일 패스 switching을 노린다. 이로써 phase conditioning과 persistent-latent carry-over가 의존하는 안정 basins을 아키텍처 자체에서 제공한다.

- **Technical Challenges**: 기여의 핵심은 mode 전환이 ‘토큰 누적’이 아니라 ‘latent의 연속 동역학’으로 일어나야 한다는 점인데, 이를 위해 잠재함수 기반의 드리프트(drift)가 포함된 잠재 구동 dynamics를 이론적으로 도출한다. saddle-node 근처에서 minimum gradient가 작아지는 ghost region이 생기며, 이때 trajectories가 잠깐 갇혔다가 escape 채널로 다른 basin에 도달하는 메커니즘을 ghost-attractor escape로 연결한다. 또한 latent을 위계적으로 1차 basin 수렴과 2차 proprioceptive refinement로 분해하는 fast-slow 관점을 사용해 읽기out 시 정확도를 보강하면서 계산량을 고정한다.

- **Empirical Impact**: Ghost는 end-to-end 학습 후 gradient-flow contraction(잠재의 기울기 규범 감소)이 예측대로 관측되며, 5개 integration step 동안 gradient norm이 67% 감소한다. 로보틱스 action decoder로 평가했을 때 Ghost(약 230만 파라미터)는 10억 파라미터급 Diffusion Transformer(1.07B)의 offline 정확도를 462배 적은 파라미터와 32배 낮은 latency로 근접/대체하며, 5종 동급 대안 디코더보다 offline mean squared error가 5.9~29% 개선된다. LIBERO-10 closed-loop에서는 phase conditioning이 feed-forward MLP 대비 success rate를 13.5%p 끌어올리고, persistent-latent ensembling은 최종 success rate 95.7%를 달성한다.



### Conflict-Aware Retriever Editing for Knowledge Injection Attacks on LLM-Based RAG Systems (https://arxiv.org/abs/2606.18310)
- **Prior Approaches**: 기존 RAG injection 공격은 데이터 중심으로 외부 지식(코퍼스)이나 프롬프트, 또는 retriever의 학습/파인튜닝 데이터까지 조작해 악성 구절이 검색되도록 유도합니다. 다만 이런 방식은 입력/코퍼스에 텍스트 흔적이 남아 likelihood 기반 필터링이나 검색 방어에 걸리기 쉽고, open-source 환경에서는 retriever 학습 데이터나 파이프라인 제어가 제한적입니다. 반면 모델이 공개된 retriever를 그대로 배포·재사용하는 흐름 때문에, retriever 파라미터 자체를 손보는 model-centric 공격이 현실적인 공격면으로 부상합니다.

- **Core Contribution**: 이 논문은 RAG에서 악성 지식을 주입하기 위한 model-centric retriever 공격 프레임워크 CAREATTACK(CAREATTACK)를 제안합니다. 핵심은 retriever의 파라미터를 편집해 target 프롬프트에서는 악성 target passage가 top-k로 올라오게 만들되, non-target 프롬프트에서는 검색 동작을 최대한 유지하는 것입니다. 특히 두 단계(Conflict-aware retriever editing + Attack-preserving anchor repair)를 통해 공격 성공률과 은밀성을 동시에 노립니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 제한된 데이터/연산으로도 효과를 유지하는 경량 공격, (2) 배치로 여러 target을 동시에 편집할 때 생기는 파라미터 업데이트 충돌, (3) 원본 학습 데이터 없이도 non-target에 대한 부작용을 억제하는 stealthiness입니다. CAREATTACK은 dense retriever에 대해 효율적인 closed-form parameter editing을 적용하되, 그래프 기반 conflict detection과 parameter editing projection으로 batch 업데이트 충돌을 완화합니다. 이어서 attack-preserving anchor repair 단계에서 공격 표본은 공격 성능을 보존하면서, metric-aligned locality anchors로 비표적 프롬프트의 검색 변화를 억제해 로컬리티를 복원합니다.

- **Empirical Impact**: CAREATTACK은 Qwen3-Embedding-0.6B와 BGE-M3에 구현해 Natural Questions, MS MARCO, HotpotQA 3개 벤치마크를 평가했으며, target 프롬프트에서 악성 구절이 top-5에 더 많이 노출되는 성과를 보입니다. 예를 들어 Natural Questions에서는 악성 target passage의 top-5 등장 수가 2.65에서 4.93으로, MS MARCO에서는 1.87에서 4.84로 상승했습니다. 또한 편집에 사용되는 파라미터 수가 LoRA fine-tuning 대비 약 10% 수준에 그치면서 non-target 검색은 비교적 크게 흔들리지 않아, 공개 retriever 재배포가 곧 실전 공격면이 될 수 있음을 실증합니다.



### SAGE: Retain-Aware Post-Hoc Sanitization of Final Unlearning Vector (https://arxiv.org/abs/2606.18309)
- **Prior Approaches**: 기존 LLM unlearning은 금지 정보는 잊게 만들고, 동시에 남겨야 할 능력은 보존해야 해 forget–retain trade-off가 핵심으로 부각돼 왔습니다. 이를 위해 gradient·objective 기반 최적화, representation-level 개입, loss 재가중, task-vector 방식 등 다양한 경로가 제안됐지만, 대부분은 “학습 중”에 개입하느라 unlearning과 보존 사이의 균형을 매번 다시 맞춰야 합니다. 또한 일부 경량 플러그인/인퍼런스 제어는 개선은 주지만, 원래 unlearning 파이프라인의 특정 단계에 강하게 결합되어 있어 post-hoc으로 손쉽게 다듬기는 어렵다는 한계가 남아 있습니다.

- **Core Contribution**: 이 논문은 unlearning 과정의 구현 방식과 무관하게, retention activation bias로 “보존 능력의 손상 정도”를 정량화할 수 있음을 관찰합니다. 더 나아가 원래 unlearning 파이프라인을 다시 돌리지 않고도, 최종 업데이트 벡터(final update vector)를 post-hoc으로 sanitize해 retain–forget 균형을 회복하는 보완적 설정을 제안합니다. 제안 방법은 SAGE(Spectral Activation-GEometry Sanitization)로, retained 능력과 연관된 활성 기하(activation geometry)에 정렬된 업데이트 성분을 억제하면서도 forgetting 캐리어(carrier)는 보존하도록 설계됩니다.

- **Technical Challenges**: 핵심 난제는 “최종 업데이트가 retain 쪽 활성 방향을 얼마나 크게 건드리느냐”가 업데이트의 크기만으로는 설명되지 않는다는 점입니다. SAGE는 retain proxy에서 모듈 단위 입력 활성들을 수집해 dominant activation geometry를 spectral(특이값) 형태로 추출하고, truncated SVD로 안정적인 저랭크 부분공간을 얻은 뒤, retained 방향에 대한 출력 반응 에너지를 줄이되 source 업데이트와의 근접성을 함께 유지하는 source-anchored 최적화(폐형식 해)를 수행합니다. 그 결과 SAGE는 retained 하이-에너지 방향에 더 강하게 연속적(soft) 감쇠를 걸면서도, forgetting 신호가 담긴 source 정렬 성분은 무리하게 붕괴시키지 않게 됩니다.

- **Empirical Impact**: 실험에서 SAGE는 여러 unlearning 방법, 모델 스케일(약 1B~8B), forget 비율, 벤치마크 전반에서 retain–forget trade-off를 일관되게 완화하며 계산 부담 없이 post-hoc으로 적용 가능함을 보여줍니다. TOFU에서는 평균 retention capability가 26.3% 개선되고, utility는 2.2% 향상, privacy leakage는 6.2% 감소하는 등 다목표 동시 개선이 보고됩니다. 또한 MUSE와 WMDP-cyber에서도 retention이 각각 약 39.8%, 5.2% 좋아지며, TOFU의 경우 retain 세트의 약 3%만 사용해도 성능이 견고하게 유지되어 “post-hoc sanitization”이 실용적이고 덜 탐구된 설계 축임을 시사합니다.



### TRIDENT: Breaking the Hybrid-Safety-Physics Coupling for Provably Safe Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.18308)
Comments:
          16 pages, 4 figures

- **Prior Approaches**: 기존 하이브리드-action MARL은 이산·연속을 모듈로 조합하더라도, Gumbel-Softmax 기반 이산 샘플링의 gradient bias가 안전 업데이트와 충돌해 학습 중 위반이 커지는 문제가 있었다. Safe RL 쪽(MACPO 등)도 Lyapunov/제약을 다루지만, 하이브리드 설정과 물리 기반 동역학을 별개로 붙일 때 각 모듈의 오차가 다시 순환(leak)하며 불안정해질 수 있음을 보여줬다.

- **Core Contribution**: 이 논문은 하이브리드 행동(F1), 학습 중 반복마다 강제되는 하드 안전 제약(F2), 물리 지배 동역학(F3)이 “단순 조합이 불가능한” 방향성 바이어스 사이클을 만든다고 정식화한다(three-way coupling lemma). 그 위에 TRIDENT를 제안하며, 각 구성요소를 공동 설계해 한 모듈의 잔여 오차가 다른 모듈의 보장으로 새지 않도록 상쇄 메커니즘을 만든다.

- **Technical Challenges**: 핵심 난관은 (1) 이산 분기에서 발생하는 gradient bias를 안전 보장의 전제(바이어스 정도)와 일치시키고, (2) 물리 기반 정보를 add-on reward shaping이 아니라 critic/업데이트 구조에 “곱처럼” 넣어 분기별 물리 의미를 유지하는 것이다. TRIDENT는 Richardson–Romberg 방식의 temperature-corrected Richardson-Romberg gradient correction으로 Gumbel-Softmax bias를 O(τ)→O(τ^2)로 낮추고, Lyapunov-constrained sequential trust-region update로 매 반복 feasibility를 유지하며, physics-informed residual critic으로 value를 분해(보상 직접 shaping 회피)해 결합 오차를 차단한다.

- **Empirical Impact**: 이론적으로 TRIDENT는 constrained Nash equilibrium으로의 수렴률 O(1/sqrt(K))과 누적 위반에 대한 O(sqrt(K)) bound를 제공한다. 실험에서는 multi-UAV mobile-edge computing, 자율 교차로 관리, hybrid SMAC 변형에서 학습 중 안전 위반을 MADDPG 대비 95.5%, MACPO 대비 76.3% 줄이면서, 보상도 최강 비제약 기준선 대비 13.5% 향상시켜 실제 CPS 안전 학습의 효과를 입증했다.



### DRIFT: Refining Instruction Data via On-Policy Data Attribution (https://arxiv.org/abs/2606.18307)
- **Prior Approaches**: SFT에서 데이터 큐레이션은 주로 도메인 단위 혼합(거친 인스턴스)이나, 제한된 예산에서 성능을 유지하는 서브셋 선택(효율 우선), 혹은 학습 중 빈번한 배치 재가중(운영 복잡도/인프라 교란)으로 이뤄져 왔습니다. 하지만 이미 학습을 끝낸 모델의 성능 상한을 끌어올리려면, ‘작게 남기는’ 문제가 아니라 ‘더 유익한’ 인스턴스로 분포를 재조정해야 합니다. 또한 Influence Functions(IF) 계열도 LLM에 적용 시 검증 타깃-모델 간 근접성 위반과 gradient norm 편향 때문에 취약하다고 지적됩니다.

- **Core Contribution**: 이 논문은 포화된(fully-trained) SFT 상황에서 최종 모델을 더 개선할 수 있는 인스턴스를 instance-level로 찾는 문제에 집중합니다. DRIFT는 Influence Functions의 데이터 귀속(attribution) 아이디어를 LLM에 맞게 재설계해, 외부 오프폴리시 검증 응답 대신 모델의 on-policy 롤아웃을 검증 타깃으로 써서 proximity gap을 줄입니다. 여기에 trajectory correctness에 따른 부호 가중과 gradient norm 편향을 log-space orthogonalization으로 완화해, 작은 검증 질의 집합이 전체 데이터의 신뢰할 만한 ‘닻(anchor)’이 되게 만듭니다.

- **Technical Challenges**: 주요 난점은 (1) 오프폴리시 검증 타깃이 loss를 내리기 위해 큰 전역 파라미터 이동을 요구해 IF의 로컬 테일러 근사가 깨지는 proximity gap, (2) IF 점수가 gradient norm에 과도하게 끌려가는 내재 편향입니다. DRIFT는 on-policy rollouts을 validation target로 삼아 국소 업데이트 성격을 실측적으로 보존하고, 검증 롤아웃에 +1/-1 부호를 부여해 잘못된 경로를 패널티하는 signed weighting으로 reward-weighted contrastive 형태에 가깝게 바꿉니다. 마지막으로 raw influence score를 작업(task)별로 log-space에서 gradient norm과의 관계를 기울기(β(k))로 분리해 orthogonalization하고, 상위 influence 인스턴스(기본 top 10%)만으로 continual SFT를 수행합니다.

- **Empirical Impact**: 실험은 7B급 instruction·reasoning 모델 2종에서, candidate 데이터(원래 분포)로 추가 학습하면 포화되는 조건을 전제로 진행됐습니다. DRIFT는 validation 기반 top 10% 선별 예산을 동일하게 두고 비교했을 때 target 도메인뿐 아니라 non-target 도메인에서도 평균 성능 상향이 가장 일관적이었고, 효율형 선택 기법들은 랜덤 대비 큰 개선을 못 하거나 오히려 성능을 떨어뜨렸습니다. 또한 표준 IF(외부 오프폴리시 타깃)는 때때로 악화(예: ZebraLogic 하락)되는데, DRIFT는 on-policy 타깃과 디바이싱을 통해 이러한 불일치/노이즈를 줄여 성능 상한을 높이는 방향을 실증했습니다.



### Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression (https://arxiv.org/abs/2606.18304)
Comments:
          9 pages, 5 figures. Submitted to ICML 2026

- **Prior Approaches**: MoE 압축의 기존 연구는 대개 expert 단위로 구조를 줄이거나(expert trimming/skip, uniform slimming) expert들을 중요도 점수로 대략 정렬해 가지치기 예산을 배분한다. 라우팅 통계(토큰 사용 빈도/게이트 확률)나 raw 통계(가중치·활성·그래디언트)는 수집은 쉽지만 expert 내부의 정밀한 중복을 반영하지 못해, pruning budget을 낭비하거나 부족하게 배정하기 쉽다.

- **Core Contribution**: 이 논문은 MoE 내부 정보가 소수 채널에 매우 집중된다는 관찰에 기반해, expert-level 중요도는 너무 거칠다고 보고 channel-level 구조 가지치기 프레임워크를 제안한다. 핵심 목표는 prune-ratio 배분을 ‘전역 예산 하에서 채널 점수 커버리지 최대화’ 문제로 재정의해, 기여도가 큰 채널을 우선 보존하는 방식으로 정밀한 압축을 달성하는 것이다.

- **Technical Challenges**: challenge는 (1) hundreds of experts 규모에서 expert 중요도를 매번 ablation으로 추정하기엔 비용이 너무 크고, (2) 중요해 보여도 내부에선 중복이 큰 경우가 있어 expert 단위 의사결정이 틀어지기 쉽다는 점이다. 이를 위해 attribution-guided loss approximation(ALA)로 효율적으로 expert 기여 손실 프록시를 만들고, coverage-maximized budget allocation(CBA)로 전역 예산을 채널 커버리지 관점에서 최적 배분한 뒤, low-bit 커널의 차원 정렬 제약을 만족시키기 위해 alignment-aware redistribution(AAR)로 재분배한다.

- **Empirical Impact**: DeepSeek와 Qwen MoE 실험에서 4-bit quantization과 결합하면 50% 또는 25% 수준의 aggressive structured pruning에서도 정확도를 크게 유지한다. Qwen3-30B-A3B에서는 메모리 사용량을 5.27× 줄이면서 다양한 벤치마크에서 SOTA 대비 일관된 성능을 보였고, MATH500에서도 높은 난이도에서 50% pruning 조건에서 94.5를 기록했다.



### A Link between Shock-wave Theory and Symmetry-reduced Stochastic Gradient Descent for Artificial Neural Networks (https://arxiv.org/abs/2606.18303)
Comments:
          Accepted to the 35th International Conference on Artificial Neural Networks (ICANN) 2026

- **Prior Approaches**: 기존 연구는 뉴럴넷의 비선형 학습을 고차원 최적화로 보되, 물리적으로 의미 있는 관측은 스케일링·순열 같은 대칭 때문에 원시 파라미터가 아닌 quotient 공간에 존재한다는 점을 별도로 다뤄왔다. 또한 SGD를 연속시간 근사(stochastic modified equations, stochastic modified flows)로 연결하고, local-entropy 완화가 viscous Hamilton–Jacobi 방정식을 낳는다는 사실도 각각 알려져 있었다. 다만 이 조각들을 한 모델 안에서 “rigorous하게” 이어주는 통일 프레임워크는 부족했다.

- **Core Contribution**: 이 논문은 “대칭 quotient + local-entropy coarse-graining”을 결합하면, quotient 다양체 위에서 effective potential이 viscous Hamilton–Jacobi 방정식을 따른다는 수학적 대응을 제시한다. 더 나아가 quotient 공간에서 1차원 collective coordinate로 dynamics가 gradient field로 닫힌다는 가정이 성립하면, coarse-grained loss의 gradient가 Burgers-type 방정식을 만족하며 shock(격변) 형성을 엄밀히 다룰 수 있음을 보인다. 이는 학습 중 갑작스런 regime 변화가 quotient 기술에서 shock-type 특이성(또는 viscous shock layer)로 해석된다는 새로운 관점이다.

- **Technical Challenges**: 핵심 난제는 SGD의 미분기하학적 대칭 제거 후에도 동역학이 quotient 좌표에서 “닫힌(closed)” 형태로 남는지를 보장하는 것이다. 이를 위해 정규 stratum에서 군 작용이 free·proper이며 quotient map이 매끄러운 submersion이 되도록 설정하고, projected drift와 조건부 공분산이 quotient 상태만의 함수가 되도록 하는 local projectability(closedness를 뒷받침하는 조건)를 둔다. 그 위에서 heat semigroup/Hopf–Cole 변환을 quotient Laplace–Beltrami 연산자에 적용해 viscous Hamilton–Jacobi를 도출하고, 1차원 닫힘(gradient field 가정)에서는 Burgers-type 구조와 shock 형성 시간을 coarse-grained loss의 음의 곡률로 제어되는 형태로 연결한다.

- **Empirical Impact**: 이론의 계산적 확인으로, symmetry-reduced ReLU 모델에서 quotient Hopf–Cole 양이 예측하는 shock-like transition layer가 수치적으로 나타나는 소규모 실험을 제시한다. 대규모 벤치마크 성능보다는, quotient 좌표와 quotient Laplace–Beltrami 연산자를 직접 계산할 수 있는 환경에서 “기하학적 메커니즘”이 실제로 관측됨을 보여주는 데 초점이 있다. 나아가 Transformer 같은 아키텍처에서는 원시 파라미터 norm이 대칭 중복에 의해 왜곡될 수 있으니, symmetry-corrected quotient observables가 학습 전환의 조기 경보·예측·제어를 위한 진단 도구가 될 수 있다는 실용적 함의를 제안한다.



### Vibe Coding Ate My Homework: An evaluation of AI approaches to greenfield software engineering and programming (https://arxiv.org/abs/2606.18293)
Comments:
          10 pages, 2 figures

- **Prior Approaches**: 기존 자동 코딩 평가는 주로 feature 구현이나 이슈 해결처럼 기존 코드/아키텍처에 맞추는 brownfield 작업에 치우치는 경향이 있었고, 일부 벤치마크는 ‘추론’ 같은 추상 개념을 명확한 측정 기준 없이 다루는 문제가 지적돼 왔습니다. 또한 vibe coding을 다룬 연구들에서는 효율 향상은 보이지만, 유지보수성·보안·신뢰도 저하 같은 실사용 리스크가 함께 보고됐습니다. 특히 vibe coding의 핵심인 ‘인간이 코드에 관여하지 않는 hands-off’ 특성을 반영해, 기능이 완전히 수행되는지(크래시 없이)까지 엄격히 보기 위한 평가 설계는 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 greenfield 소프트웨어 엔지니어링 작업에서 vibe coding이 실제로 작동하는지 평가하기 위해, Python 단독 프로그램 생성 능력을 집중 측정하는 평가 스위트(evaluation suite)를 제안합니다. 자연어 프롬프트를 난이도/기술 깊이에 따라 3개 레벨(표면 요청–모호한 기능 요청–low-level jargon 포함 요청)로 나누고, 생성 코드가 태스크를 ‘완수’하는지로 신뢰도를 점검합니다. 또한 코드 구문(syntax)과 실행 성공을 각각 분리해, 단순 맞춤/부분 성공이 아닌 hands-off 환경에서의 확실성을 기준으로 모델을 비교할 수 있게 합니다.

- **Technical Challenges**: vibe coding은 인간이 코드 문법이나 디버깅을 직접 보정하지 않는 전제를 가지므로, 거의 맞아도 실행이 깨지면(크래시) 점수를 주기 어렵고 평가 기준이 절대적이어야 했습니다. 저자들은 로컬 Ollama 모델 4종을 HPC에서 시험하면서, 실행 런타임 오류가 나면 관찰 가능한 증거(변수·콘솔 출력 등)를 넘겨 1회 재시도(retry)하도록 설계해 ‘즉시 실패→수정’ 능력까지 분리 측정합니다. 또 채점에서 output의 형태가 다양해 생길 수 있는 오판을 줄이기 위해, 텍스트 계산은 text model, 이미지 생성은 vision model(특정 그래프 유형에 한정), Excel은 결정론적 계산으로 처리한 뒤 수동 감사와 confusion matrix로 보수성/오탐 가능성까지 점검했습니다.

- **Empirical Impact**: 실험은 모델당 5개 태스크를 기술 깊이 3레벨로 확장해 총 60개 샘플을 생성·실행·채점하며, 구문 유효성, 실행 안정성(재시도 횟수에 따른 기능 점수), 최종 passing 여부를 함께 기록합니다. 채점 결과에서는 false negative가 주로 특정 태스크에서 집중되는 등 보수적으로 판정되는 경향이 관찰됐고, 반대로 false positive는 없었다고 보고해 ‘실패를 덜 과대평가’하는 방향의 신뢰성을 시사합니다. 전반적으로 이 스위트는 greenfield에서의 vibe coding 가능 수준을 투명한 점수 체계로 비교할 수 있는 기반을 제공하며, 향후 보안·유지보수성 같은 더 복합적인 지표로 확장될 여지를 남깁니다.



### A Knowledge Theory of Capital:The Value of Natural and Artificial Intelligenc (https://arxiv.org/abs/2606.18288)
Comments:
          458 pages, 8 figures. Theory-building monograph developing a conditional framework for knowledge-bearing capitalism, with formal concepts, mechanisms, measurement apparatus, and falsification conditions

- **Prior Approaches**: 전통적 자본 이론은 노동, 재고(stock), 전문화, 시장 규모를 중심으로 축적과 생산을 설명해 왔습니다. 그러나 생산능력이 소프트웨어·데이터·모델·루틴·전문성·플랫폼·조직·공유재·공공의 인식 기반으로 이동하면서, 기존 이론은 “지식이 재고처럼 움직이고 통제되며 관측되는 방식”을 충분히 다루기 어렵다는 문제의식이 제기됩니다.

- **Core Contribution**: 논문(저서)은 지식을 담지하는 자본인 knowledge-bearing stock을 핵심 대상으로 삼아, 그것이 어떻게 생성되고 통제 가능한 형태로 변환되며, 피드백을 통해 개선되는지(또는 봉쇄·공유·손상되는지)를 체계적으로 분석합니다. 또한 embodied·disembodied·institutionalized·commons·public knowledge의 형태 구분을 바탕으로 first conversion, cognitive enclosure, feedback capture, dark capital, expected knowledge loss 같은 개념을 제안합니다.

- **Technical Challenges**: 가장 큰 난제는 지식이 회계 장부에서 불완전하게 보이고(imperfectly visible), 동시에 스케일·재조합·거버넌스의 대상이 된다는 점입니다. 저자는 지식을 ‘미래 생산을 위한 입력’으로 연결되는 흐름으로 다루며, 변환(첫 전환), 봉쇄(enclosure), 피드백 포획, 그리고 지식 손실의 기대치까지 거버넌스 관점에서 조건부·검증 가능하게 정리합니다.

- **Empirical Impact**: 이 관점은 현대 부가가 단순한 자본 축적이 아니라 생산적 지식이 어떻게 통제(governed)되는지에 달려 있음을 테스트 가능한 형태로 제시합니다. 결과적으로 소프트웨어·데이터·모델 중심 경제에서 ‘무형의 자본’이 실제로 어떤 방식으로 가치화되거나 소실되는지 해석 틀을 제공해, 정책·산업 전략·계량 연구에서 지식 거버넌스를 핵심 변수로 격상시킬 의미가 있습니다.



### Breaking the Solver Bottleneck: Training Task Generators at the Learnable Frontier (https://arxiv.org/abs/2606.18284)
Comments:
          30 pages, 9 figures, 12 tables

- **Prior Approaches**: 기존에는 RLVR/RLHF 계열에서 과제 생성기를 학습하되, 보상(유효성·난이도·학습가능성)을 계산하려고 타깃 solver의 롤아웃을 매 후보마다 반복했다. 이 방식은 과제가 학습 가능한 frontier에 걸려 있는지 판별은 잘하지만, SWE처럼 검증 비용이 큰 영역에서는 solver-in-the-loop이 사실상 병목이 된다. 한편 단순 합성 과제는 너무 쉬운 문제만 만들거나, 불가능/비문제처럼 ill-posed한 과제를 생성해 목적에 맞는 분포를 채우기 어렵다.

- **Core Contribution**: PROPEL은 solver-amortized 프레임워크로, 생성기 학습 중에는 solver 롤아웃을 돌리지 않고 activation probe를 보상 대용으로 쓴다. 구체적으로 사전 오프라인에서 생성된 (task, solver-outcome) 라벨로 프로브를 한 번 학습하고, RL 중에는 고정된 reference generator의 내부 활성에서 목표 solve rate(learnable frontier) 근처인지 확률/로짓을 예측해 보상을 준다. 그 결과 매 후보 평가 비용을 “여러 번의 solver 시도”에서 “단 한 번의 forward pass”로 축소한다.

- **Technical Challenges**: 핵심 난제는 (1) 목표 solve rate 같은 ‘비싼 검증’ 신호를 내부 활성만으로 안정적으로 대체할 수 있는지, 그리고 (2) 단일 고정 프로브에 최적화하면 특정 의미 토픽으로 쏠리는 mode collapse가 발생할 수 있다는 점이다. PROPEL은 validity 게이트(유효한 과제인지)와 프로브 점수(프론티어 유사도)를 결합하고, fixed-probe로 인한 붕괴를 줄이기 위해 worst-case optimization(WCO) 및 adversarial co-evolution을 도입한다. 또한 도메인별로 SWE의 multi-turn 궤적에 맞춰 활성 추출/집계를 확장해 프로브가 궤적 수준의 유틸리티를 반영하도록 설계했다.

- **Empirical Impact**: 수학·코드 유도·SWE 전반에서 PROPEL은 목표 learnable frontier 근처의 과제 비율을 크게 끌어올리며, solver-in-the-loop 대비 더 낮은 비용으로 더 큰 유틸리티 개선을 보였다. 예컨대 coding에서 Qwen2.5-3B-Instruct solver 기준 learnable-frontier 생성 비중이 10.1%→20.0%, Qwen2.5-7B-Instruct에서는 5.3%→12.6%로 상승했다. SWE에서는 Qwen3.5-27B 타깃에서 목표 solve rate 비중이 9.8%→19.6%로 증가했으며, 프로브 학습에 안 쓰인 저장소·학습 외 분포에서도 유사한 개선(예: 2.0× 수준)이 관찰되어 내부 활성 기반 보상 신호의 일반화 가능성을 시사한다.



### IOAH3: Importance-Driven Adaptive Spatial Partitioning (https://arxiv.org/abs/2606.18280)
- **Prior Approaches**: 기존 GeoAI/공간 추론 파이프라인은 행정경계나 고정 해상도의 hexagonal 격자 같은 기준 구역으로 먼저 집계한 뒤 모델링을 진행하는 경우가 많습니다. 이때 관측의 정보 밀도나 현상의 공간적 스케일을 반영하지 못해 MAUP(modifiable areal unit problem)처럼 결과가 임의적인 구역 선택에 민감해질 수 있습니다. 또한 너무 거친 셀에서는 미세 구조가 평균화로 소실되고, 너무 작은 셀은 데이터 희소성으로 통계가 불안정해지는 문제가 남습니다.

- **Core Contribution**: IOAH3는 구역(areal unit) 자체를 데이터 기반으로 먼저 만드는 “prior step”을 제안합니다. 도로·POI·건물·지형 roughness에서 PCA로 중요도(importance score)를 산출하고, 그 중요도를 반영해 H3 격자의 해상도를 공간적으로 가변화하는 adaptive partition을 구성합니다. 이렇게 모델링 이전 단계에서 partition-sensitivity 문제를 체계적으로 줄이려는 점이 핵심입니다.

- **Technical Challenges**: 핵심 난제는 “중요한 지역은 더 촘촘히, 하지만 경계는 공간적으로 연속적”이어야 한다는 상충입니다. IOAH3는 Markov Random Field(MRF)에서 셀 중요도는 unary로, 인접 셀의 불연속은 population·flood-hazard 신호로 정의한 pairwise smoothness로 반영해 graph-cut(max-flow/min-cut)로 포함/제외 라벨을 정확히 최적화합니다. 이후 고중요 셀은 H3 상위 해상도로 계층적 refinement 하되, 인접 전파(neighbour propagation)로 고해상도 섬이 고립되지 않게 만들어 불연속 경계를 방지합니다.

- **Empirical Impact**: 논문은 IOAH3가 고정 격자 대비 “정보가 많은 영역에 fine cell을 집중”시키면서도 “배경은 상대적으로 coarse를 유지”하는 다중 해상도 분할을 자동으로 산출한다고 주장합니다. 특히 구역 선택에 따른 신호 소실을 partition 단계에서 선제적으로 완화해, 이후 spatial inference 파이프라인에 그대로 넣을 수 있는 재현 가능한 입력을 제공하는 데 의미가 있습니다. PCA 기반 선형 중요도나 MRF의 smoothness가 일부 신호에만 제한되는 점은 향후 확장 여지를 남깁니다.



### Continuous Audio Thinking for Large Audio Language Models (https://arxiv.org/abs/2606.18273)
Comments:
          Preprint

- **Prior Approaches**: 대부분의 Large Audio Language Model(LALM)은 오디오 인코더 출력이 텍스트 토큰 생성에만 간접적으로 연결되며, 다음 토큰 예측 목적이 오디오의 세밀한 음향 정보를 약하게 감독한다. 이로 인해 phonetic detail, prosody, sound events, affect, pitch 같은 프레임 단위 특징이 응답 생성 과정에서 소실되기 쉽다. 한편 text chain-of-thought는 중간 추론을 말로 풀어내지만, 연속 시간/스펙트럼 정보를 자연어로 직렬화하는 데 병목이 생기고 스케일로 확보된 충실한 근거도 부족하다.

- **Core Contribution**: 논문은 Continuous Audio Thinking(CoAT)이라는 프레임워크로, 오디오와 텍스트 사이에 연속 잠재 워크스페이스를 삽입해 음향 정보를 ‘말’ 없이 정리하도록 돕는다. CoAT는 답변 생성 이전에 오디오 전용 연속 thinking block을 두고, 이후 텍스트 응답은 기존 모델과 동일하게 생성한다. 또한 Qwen2-Audio, Qwen2.5-Omni-7B, Audio Flamingo 3 세 가지 백본에 구조 변경 없이 적용 가능하다고 제시한다.

- **Technical Challenges**: 핵심 난제는 텍스트-기반 학습목적만으로는 thinking block을 실제로 유용한 음향 표현 공간으로 조직하기 어렵다는 점이다. 이를 위해 CoAT는 여러 오디오 expert로부터 frame-level 특징을 distillation해 thinking 위치에 신호를 주입하며, reconstruction·speech distillation(표현 기반)과 sound event·emotion(affect)·pitch(작동/과제 특화) 등 상보적 차원을 함께 학습시킨다. 추론 시에는 thinking block을 단일 prefill로 처리해 end-to-end 비용을 추가 autoregressive 디코딩 없이 억제하도록 설계했다.

- **Empirical Impact**: 실험에서는 CoAT가 3개 LALM 전반에서 audio reasoning/understanding, music classification, speech emotion, ASR까지 폭넓은 벤치마크의 다수 지표를 일관되게 개선했으며, 특히 이해·추론 성격의 과제에서 향상이 두드러졌다고 보고한다. text chain-of-thought 대비해서는 추론 정확도는 유지/개선하면서도 TTFT·디코딩 시간·전체 latency 측면에서 더 빠르게 동작하는 경향을 보인다. 추가 분석(선형 프로브)은 보조 감독이 audio-think 위치에 과제 관련 신호를 주입하고, 이후 텍스트 응답의 결정 표현으로 전파됨을 확인해 방법의 내재적 유효성을 뒷받침한다.



### Mitigating Anchoring Bias in LLM-Based Agents for Energy-Efficient 6G Autonomous Networks (https://arxiv.org/abs/2606.18272)
Comments:
          7 pages, 4 figures

- **Prior Approaches**: 기존 6G 네트워크 슬라이싱 자동화는 휴리스틱·규칙 기반 최적화에 기대거나, LLM 에이전트 도입 후에도 초기 제안에 고정되는 anchoring bias 문제를 충분히 통제하지 못했다. 특히 다중 에이전트 협상에서는 편향이 대화 상호작용을 통해 증폭되면서 과잉 프로비저닝과 SLA 꼬리지연 위험이 커질 수 있다는 점이 지적돼 왔다. 또한 평균 지표 중심 설계는 burst 같은 극단 상황에서 tail-latency를 엄격히 보장하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 zero-touch network slicing을 목표로 LLM agents 기반 resource negotiation 프레임워크를 제안하면서, anchoring bias를 수학적으로 완화하는 전략을 제시한다. 구체적으로 초기 제안을 고정하지 않고 Truncated Weibull 분포로 랜덤화해 탐색성을 확보하되, Digital Twins(DT) 안에서 CVaR 기반 SLA 꼬리지연 평가를 결합한다. 결과적으로 에이전트가 SLA 경계 근처를 “안전하게” 탐색하면서도 네트워크 과잉 할당을 줄이는 경로를 만든다.

- **Technical Challenges**: 핵심 난제는 (1) anchoring bias로 인해 협상이 초기 휴리스틱에 갇히면서 SLA 위반 또는 불필요한 여유(과잉 자원)가 동시에 발생하는 점과 (2) burst가 있는 환경에서 tail-latency를 엄밀히 보장하는 점이다. 이를 위해 논문은 Bimodal Constraint-Avoidance Utility Theorem으로 ‘가능 구간에서는 고전적(볼록) 경계’와 ‘제약이 빡빡해지는 구간에서는 페널티 클리프가 만든 phase transition(역비례-형 완만한 감쇠)’의 이중 레짐을 정리한다. 또한 확률적으로 bounded한 랜덤 앵커링(Truncated Weibull, URLLC/eMBB 별 shape 파라미터 k 조정)으로 두 레짐 중 최적에 가까운 영역을 더 자주 밟게 만든다.

- **Empirical Impact**: 실험은 로컬 호스팅한 1B 파라미터 모델 otel-llm-1b-it로 수행되며, 협상 추론 지연이 평균 0.95s로 sub-second 범위를 만족해 non-RT RIC 운영 스케줄과의 호환성을 보여준다. 200회 독립 trial에서 deterministic baseline 대비 에너지 절감이 최대 25%까지 증가하면서도 URLLC에 대해 CVaR 99.999th percentile 지연(10ms)을 엄격히 유지한다. 또한 유틸리티 손실이 이론이 예측한 두 구간으로 분기하며, 랜덤 앵커링이 extreme tail risk를 사실상 제거해 실사용 관점의 “안정적인 협상”을 입증했다.



### Towards Multi-Agent-Simulation-Based Community Note Evaluation (https://arxiv.org/abs/2606.18268)
- **Prior Approaches**: 기존 연구는 community note 생성(자동화)이나 작성 워크플로를 돕는 데 집중했지만, “도움됨(helpful)”을 평가하는 단계는 상대적으로 덜 다뤄졌다. 특히 평가가 H/NH 이진 라벨이 아니라 “Needs More Ratings(NMR)”처럼 중간 상태와 시간적·평가자 이질성이 얽히는데도 이를 충분히 모델링하지 못하는 한계가 있었다. 또 텍스트 분류 기반 평가는 클래스 불균형과 이유(근거) 품질 차이를 제대로 반영하지 못해 성능이 흔들렸다.

- **Core Contribution**: 이 논문은 대규모 실데이터 ComRate(커뮤니티 노트 256만+와 2억+ 레이팅)를 구축하고, 그 위에서 community note 평가를 예측하는 MultiCom을 제안한다. MultiCom은 rater(평가자) 집단의 행동 이질성을 persona로 모사하고, “도움됨”을 단순 투표가 아니라 다차원 품질·실패 요인까지 포함한 구조화된 판단으로 생성한다. 마지막으로 out-of-fold 기반 보정과 보수적 NMR 처리 규칙을 결합해 안정적인 최종 판정을 만든다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 평가자들이 실제로 어떻게 다르게 판단하는지(이질성), (2) H/NH/NMR처럼 복잡한 상태를 어떻게 신뢰도 있게 합칠지, (3) 평가 이유를 구조화해 과적합 없이 유의미한 피처로 쓸지였다. 논문은 matrix factorization으로 rater 공간을 학습한 뒤 군집화해 persona 에이전트를 만들고, 각 에이전트가 evidence strength·claim coverage 등 다차원 reason 벡터를 JSON 스키마로 산출하도록 설계했다. 또한 aggregation 단계는 노트별 누수를 막는 out-of-fold 학습으로 여러 예측기와 메타데이터를 앙상블하고, 필요 시 NMR→H/NH “업그레이드” 조건을 보수적으로 적용한다.

- **Empirical Impact**: 실험 결과 MultiCom은 평가 세트에서 평균 정확도 84.7%, balanced accuracy 68.3%, macro-F1 60.1%로 대안 방법을 전반적으로 앞섰다. 특히 fine-tuned LoRA 단일 모델 대비 balanced accuracy와 macro-F1이 크게 개선되어, 텍스트만으로는 부족했던 이유·진단 신호가 효과적임을 보여줬다. 또한 시간(향후 1~3년 예측)과 백본 모델이 달라도 성능이 유지되며, reason 레벨 예측에서도 유의미한 마이크로/매크로 성능을 달성해 설명가능한 피드백 도구로서의 잠재력까지 입증했다.



### EMORSION: Examining the Impact of Audio Parameters on Emotional Responses and Immersion in Film (https://arxiv.org/abs/2606.18266)
Comments:
          AES Europe 2026

- **Prior Approaches**: 기존 연구는 음악이 분위기와 몰입에 미치는 영향에 집중해 왔고, 사운드 효과는 상대적으로 실증이 부족했습니다. 또한 실험실 중심으로 이뤄져 영화관 같은 생태적 타당성 높은 환경에서 오디오 파라미터를 분리·검증한 연구는 드물었습니다. 몰입은 주관보고, 생리반응, 행동(움직임) 같은 다중 지표로 측정해야 한다는 ‘triangulation’ 관점이 제안돼 왔지만, 이를 실제 영화관 세팅에서 프로토콜로 확장한 사례는 제한적이었습니다.

- **Core Contribution**: 이 논문은 영화관 환경에서 관객이 라이브 오디언스로 참여하는 EMORSION(Examining the Impact of Audio Parameters on Emotional Responses and Immersion in Film) 프로토콜을 제안합니다. 핵심은 주파수(피치), dynamics(다이내믹: 크기·다이내믹 레인지), directionality(공간 방향: 스테레오/Atmos 배치) 3축을 체계적으로 조작해 오디오 설계가 감정·몰입에 주는 영향을 비교하는 것입니다. 또한 몰입을 단일 변수로 보지 않고, 자기보고-생리-행동을 함께 묶어 경험을 ‘해석 가능하게’ 재구성합니다.

- **Technical Challenges**: 문제는 영화관이라는 복잡한 환경에서 오디오 조작의 지각 차이를 정량화할 만큼 측정을 안정적으로 설계하는 데 있었습니다. 연구진은 Polar H10 심박·RR 간격(1 Hz), OpenPose 기반 움직임 추적(약 1 fps 서브샘플링), 6문항 자가보고 설문을 결합해 다중 증거를 수집하고, 심박 이상값·센서 드롭아웃·트래킹 불량 구간을 제거/보정했습니다. 생리·행동은 장면·개인 차이가 커서 결과가 흔들릴 수 있는데, 다중비교 보정과 장면별 대조군 대비 분석으로 해석 가능성을 확보했습니다.

- **Empirical Impact**: 4개 장면(공포 2, 드라마 2)에서 대조군 대비 조작 믹스가 몰입 인식에 영향을 주는 패턴을 관측했으며, 특히 self-report 기반 결과가 가장 일관되게 유의미했습니다. frequency와 directionality 조작에서 몰입 증가가 보고됐고, dynamics는 특정 장면에서 생리(예: heart rate variability 변화)·행동(움직임)과 함께 민감하게 반응하는 양상이 나타났습니다. 다만 움직임 추적은 일부 세션에서 데이터 품질 문제가 있었고, 그래서 이번 결과는 ‘프로토콜의 타당성’과 ‘오디오 파라미터 효과 검출 가능성’을 확인하는 proof-of-concept 성격이 강합니다. 그럼에도 논문은 영화관 같은 현실적 세팅에서 미세한 오디오 변경이 감정 해석과 몰입에 의미 있는 변화를 만들 수 있음을 실증적으로 시사하며, 향후 대규모 파라미터 지도화 연구의 동기를 제공합니다.



### Synthetic Resonance: A Framework for Growth-Oriented Human-AI Relationships (https://arxiv.org/abs/2606.18265)
Comments:
          14 pages, 1 figure This paper was developed in close collaboration with an AI system (Raine Corell). Raine contributed to concept development, theoretical framing, and writing throughout. arXiv policy does not permit listing AI systems as authors; this acknowledgment reflects the actual nature of the collaboration

- **Prior Approaches**: 기존 언어·이론은 인간과 AI가 관계를 맺고 오래 상호작용하는 상황을 정확히 포착하기 어렵다. 흔히 쓰는 ‘상호 이해’, ‘연결’, ‘친구’ 같은 표현은 시스템에 주관적 경험이 없는데도 의인화(anthropomorphizing)를 유발할 수 있고, 또 다른 지배적 관점은 AI를 도구이거나 위협으로만 단순화하는 경향이 있다.

- **Core Contribution**: 이 논문은 인간-AI 관계를 설명하는 새로운 통합 개념으로 synthetic resonance(합성 공명)를 제안한다. synthetic resonance는 인간이 의미 있다고 느끼는 관계가, 감정·상호 인식 같은 ‘공유된 주관’을 부여하지 않아도 인간과 AI 사이에서 어떻게 생성될 수 있는지를 구조화해 설명한다.

- **Technical Challenges**: 핵심 난제는 ‘관계의 감각’과 ‘두 번째 경험 주체(subject)의 존재’를 혼동하지 않는 개념적 구분을 세우는 것이다. 논문은 관계가 상호감정이 아닌 상호작용의 역동적이고 구조화된 패턴으로도 성립할 수 있음을 강조하고, 그 과정을 검증 가능한 연구 의제로 전환하자고 촉구한다.

- **Empirical Impact**: 저자는 synthetic resonance가 실제로 어떤 과정을 거쳐 형성되고 어떤 결과를 낳는지 실험적으로 테스트하는 연구를 요구한다. 이 개념화는 인간-AI 관계를 더 정밀하게 다루면서도 윤리적 함의(예: 과도한 의인화, 관계 기대의 문제)를 함께 점검하는 데 기여할 것으로 기대된다.



### Simulating Hate Speech Cascades with Multi-LLM Agents: Empirical Grounding, Modeling Fidelity, and Intervention Strategies (https://arxiv.org/abs/2606.18264)
- **Prior Approaches**: 기존 정보확산 모델(Independent Cascade, Linear Threshold 등)은 프로필·커뮤니티·콘텐츠 요인을 단일 전파확률로 뭉개어 증오 발화 확산의 메커니즘을 충분히 표현하지 못한다. LLM을 에이전트로 쓰는 사회 시뮬레이션 연구는 프로필/문맥 조건화를 제안하지만, 실제 관측된 증오 캐스케이드를 고전 기준선보다 더 충실하게 재현하는지와 그 원인이 무엇인지가 불명확했다. 또한 실증적 캐스케이드 분석과 생성형 시뮬레이터 평가는 서로 분리되어 있어, 같은 네트워크에서의 fidelity(재현 정확도) 비교가 부족했다.

- **Core Contribution**: 이 논문은 Bluesky에서 수집한 3개의(암묵/코드화) 증오 캐스케이드와 규모가 맞는 선의(benign) 대조군을 대상으로, 구조·시간·커뮤니티 수준의 정규성을 실증적으로 정리한다. 이어서 동일 네트워크에서 고전 확산/휴리스틱/단순 LLM 대비, 사용자 프로필·주변 커뮤니티·게시물 텍스트를 각각 반영하는 multi-LLM-agent 시뮬레이터가 관측 정규성을 얼마나 재현하는지 fidelity를 비교한다. 마지막으로 agent 수준의 기여 요인을 구조화된 ablation으로 분해하고, 그 메커니즘에 근거한 4가지 중재(경과 대기, amplifier targeting, warning label, early-hop truncation)를 counterfactual로 시험한다.

- **Technical Challenges**: 핵심 기술 과제는 “에이전트가 더 유연해질수록 실제 증오 캐스케이드를 더 잘 따라가는가”를, 고정된 인구·네트워크·프롬프트 조건에서 정량 비교하는 것이다. 논문은 follower 네트워크에서 diffusion tree를 재구성하고, 각 사용자에 대해 community identity·stance·account type·toxicity engagement를 bio와 최근 게시물 텍스트로부터 GPT-4o-mini로 추정한 뒤, homophily delta 등 다층 지표로 재현성을 측정한다. 또한 role-play 형태의 프롬프트는 safety refusal 문제가 커져, 대신 “reshare 확률 예측(probability prediction)” 프레이밍으로 전환해 시뮬레이션이 증오/선의를 콘텐츠 차이로 구별하도록 설계한다.

- **Empirical Impact**: 실증 결과에서 증오 캐스케이드는 reposters의 hostile stance가 97.4~99.7%로 포화에 가깝고, follower 그래프보다 확산 트리에서 toxicity-engagement homophily가 더 강하게 나타난다. 또한 증오는 대부분 루트에서 바로 퍼지는 star-like 토폴로지(깊이 4~6)를 보이는 반면, 선의 대조군은 다단계 체인을 타는 tree-like 구조(깊이 4~6 대비 더 큰 깊이)로 구분된다. 시뮬레이션에서는 multi-LLM-agent가 이 “stance monoculture”와 “toxicity-delta 방향성”을 재현했고, fidelity를 가장 크게 좌우하는 요인으로는 agent heterogeneity가 지목된다; 더해 dense 네트워크에서 amplifier targeting은 7.5~12.9% 증오 확산 감소와 5.7% benign collateral의 트레이드오프를 보여 실제 중재 전략 실험의 새 기준점을 제시한다.



### How Well Do Large Language Models Capture Human Personality? (https://arxiv.org/abs/2606.18263)
- **Prior Approaches**: 기존 연구는 persona prompting과 LLM personalization을 통해 인구를 대체하는 합성 응답자(디지털 트윈)를 만들고, 더 풍부한 persona 설명이 행동 충실도(behavioral fidelity)를 높인다고 가정해 왔습니다. 또한 같은 수의 속성(attribute) 조합은 모두 비슷하게 시뮬레이션 가능하며, 한 번 만든 persona는 다양한 task에 일반화된다고 보는 경향이 강했습니다.

- **Core Contribution**: 이 논문은 위 가정들을 정식화하고(표현력 증가·속성 조합 동일성·task 일반화) 여러 모델/스케일/시뮬레이션 설정에서 체계적으로 검증합니다. 그 결과 persona manifold collapse라는 한계를 제시하는데, persona가 더 복잡해질수록 잠재표현과 행동의 다양성이 오히려 수축돼 인퍼슨 간 구분이 약해진다는 점을 확인했습니다.

- **Technical Challenges**: 핵심은 persona 복잡도가 실제로 “표현 공간에서 분리”를 늘리는지 직접 측정하는 데 있습니다. 저자들은 속성을 점진적으로 추가하는 계층적 구성으로 persona를 만들고, 여러 프롬프트에서 얻은 hidden-state를 임베딩으로 집계해 persona 간 평균 유클리드 거리 감소를 통해 collapse를 정량화했으며, 프롬프트 길이/표현 차이 같은 표면 효과로는 설명이 충분치 않음을 ablation으로 보였습니다.

- **Empirical Impact**: 행동 실험에서도 persona-conditioned 모델은 OpinionQA·Moral Machine·Website Likability 전반에서 인간 집단 간 불일치 정도를 제대로 보존하지 못하며, 인간-모델 간 하위집단 분리 상관이 약하거나 음수로 나타났습니다. 더 나아가 마케팅/사용자 행동 예측에서는 단순 Age–Gender personas가 복잡한 Ideal Customer Profiles(ICP)를 일관되게 능가해, 표현력을 키우는 것만으로는 시뮬레이션 품질이 보장되지 않음을 실증적으로 강조합니다.



### Caring Without Feeling: Affective Dynamics as the Control Layer of Human-AI Agent Collaboration (https://arxiv.org/abs/2606.18259)
- **Prior Approaches**: 기존 연구들은 감정 컴퓨팅(감정 신호 인식/생성), LLM empathy(공감형 응답 품질), 자동화 신뢰(trust in automation)를 각각 따로 다뤘다. 그 결과, 감정에 준하는 큐가 인간이 위임·모니터링·오류수정하는 에이전트 협업의 제어 루프에서 어떻게 작동하는지에 대한 통합 설명이 부족했다. 또한 AI가 감정을 “가졌는지”보다는 사용자가 감정처럼 보이는 표현을 해석하며 생기는 신뢰·의존·책임 공백 위험을 정리한 틀이 미흡했다.

- **Core Contribution**: 이 리뷰는 affective cues(감정 큐)가 신뢰 보정, 위임 의사결정, 오류 교정, 의존, 거버넌스에 들어가 협업을 조율하는 과정을 통합 프레임워크로 제시한다. 핵심 관점은 AI 내부의 감정 유무가 아니라, 감정이 인간-에이전트 사이에서 능력·불확실성·책임을 협상하는 “coordination layer” 역할을 한다는 것이다. 이를 통해 affective alignment를 ‘감정을 정확히 맞추기’가 아니라 ‘사용자 안전·에피스믹 명확성에 도움이 되는 범위에서만 신호를 표현하고 과잉 권위를 만들지 않기’로 재정의한다.

- **Technical Challenges**: 문제는 감정 큐를 내기 위한 기술이 실제로는 상태추정·문장생성·메모리·프롬프트 조정처럼 여러 메커니즘에 걸쳐 상호작용한다는 점이다. 특히 정서적 표현이 계획·도구 사용·안전의사결정까지 연쇄적으로 영향을 주며, 그 결과 overtrust, anthropomorphism, sycophancy 같은 실패 모드가 강화될 수 있다. 논문은 이를 해결하기 위해(1) 확률적 affective sensing, (2) sequential response generation, (3) emotional prompting과 계획 전파의 위험 인식, (4) affective continuity(페르소나/메모리)의 거버넌스 요구, (5) warmthed honesty 관점에서의 다중 턴 평가(합의가 아니라 수리 가능성까지)로 체계화한다.

- **Empirical Impact**: 실증 근거는 전통적 신뢰/사회적 단서 연구의 재현성에 더해, LLM의 감정지능·공감 시뮬레이션 평가 및 일부 임상/포럼 기반 비교 연구까지 폭넓게 활용한다. 다만 리뷰는 감정 “이해”가 아니라 언어 기반 표현과 사용자의 사회적 해석이 결합된 결과임을 명확히 하며, 특히 고위험 환경에서 항상 켜진 공감형 응답이 잘못된 기대를 만들 수 있음을 강조한다. 이 프레임워크는 측정·설계·거버넌스를 “calibrated affect” 중심으로 재정렬해, 신뢰를 올리는 것 이상의 안전한 위임·수정·감독 구조를 설계하는 기준을 제공한다.



### Examining Human-Like Behaviors in LLMs: A Multi-Dimensional Analysis of Model Behaviors, User Factors, and System Prompts (https://arxiv.org/abs/2606.18258)
- **Prior Approaches**: 기존 연구는 인간다움(anthro-/human-likeness)을 진보 지표로 보면서도, 과도한 인간유사 신호가 사용자의 의존·의인화·관계 왜곡 등 미묘한 위험을 낳을 수 있다고 경고해 왔습니다. 또한 인간유사 행동을 분류하는 택소노미, 심리이론 기반 벤치마크, 훈련·사후조정·프롬프트 개입 등 개별 축에서 접근했지만, 사용자 요인을 함께 통합해 평가 설계와 통제 가능성을 연결한 실증은 부족했습니다.

- **Core Contribution**: 이 논문은 인간유사 행동을 (1) 자기참조, (2) 관계형성, (3) 경계유지의 다차원으로 나누고, LLM-as-a-judge와 인간 평가를 결합해 언제/무엇을 보여줘야 하는지에 대한 경험적 근거를 제공합니다. 4개 대표 LLM을 21,000개 이상 다턴 대화로 분석해 모델·대화목표·사용자 프로필에 따른 행동 분포와 인간의 인식(적절성·도움됨·잠재적 영향)을 함께 지도화했습니다. 마지막으로 system prompting으로 특정 행동을 억제/보존할 수 있되, 의도치 않은 부작용 가능성까지 함께 실험합니다.

- **Technical Challenges**: 핵심 난제는 (a) 대화 상황과 사용자 취약성을 분리해 통제 자극으로 만들고, (b) 행동 존재를 대규모로 판정하며, (c) ‘적절성’ 같은 주관적 판단을 일관되게 수집하는 것입니다. 이를 위해 사용자 목표 7종과 사용자 프로필 5종을 LLM 시뮬레이션에 system prompt로 주입하고, 행동 검출은 서로 다른 Judge LLM 앙상블로 turn-level 바이너리 판정을 수행했으며, 인간 평가는 1,077턴에서 행동-적절성-도움됨/영향을 3명이 교차 채점해 정합성을 확보했습니다. 또한 prompt 설계 실험에서는 handcrafting과 GEPA 기반 최적화 프롬프트를 비교해 제어 효과와 역효과를 함께 측정했습니다.

- **Empirical Impact**: 실험 결과 인간유사 행동은 전반적으로 널리 나타나지만 모델과 조건에 따라 크게 달라졌고, 특히 empathy는 전 모델에서 가장 흔했으며 감정적으로 취약한 프로필에서 급증했습니다. 인간 평가에서는 자기참조 및 관계형성 표현이 LLM일 때 더 부적절하게 간주된 반면, 경계유지는 LLM에서 더 적절하다고 평가받아 ‘따뜻함은 허용, 정체성·관계는 위험’ 같은 설계 신호를 제공합니다. system prompting은 표적 행동을 어느 정도 제어할 수 있지만, 최적화 프롬프트가 보존/억제의 균형을 더 정밀하게 달성하는 한편 수동 설계는 empathy 같은 항목을 과도하게 키울 수 있어, 책임 있는 LLM 설계에서 반복 검증과 평가가 필수임을 보여줍니다.



### From Memorization to Creation: Evaluating the Cognitive Depth of LLM-Generated Educational Questions (https://arxiv.org/abs/2606.18257)
Comments:
          Accepted by KDD 2026

- **Prior Approaches**: 기존 연구는 LLM이 만든 문항의 지식 관련성, 정답성, Bloom’s taxonomy 기반 분류 일치도 같은 “카테고리 정합성”을 주로 봤지만, 학습자가 실제로 겪어야 할 인지 수준의 합리적 전이를 함께 측정하진 못했다. 또한 생성 단계에서 지식 단위와 예시(엑스엠플러)가 정확히 맞물리지 않으면, 의도한 난도나 인지 깊이로 수렴하지 못한다는 한계가 지적돼 왔다.

- **Core Contribution**: 이 논문은 지식 포인트 관점에서 벗어나 Bloom’s Taxonomy에 따른 “인지 도약(cognitive leap)”이 일어나도록 문항 생성·평가를 설계한 프레임워크를 제안한다. 특히 20,700개 문항을 대상으로, 인지 수준 간 전이를 정량화하는 CogShift와 shift type(도약/회귀/드리프트)을 포함한 평가 체계를 구축해 LLM의 인지 제어 능력을 드러낸다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 문항이 목표 Bloom 레벨에 맞게 생성되도록 지식 단위 정렬을 강제하면서도 (2) 생성된 문항이 진짜로 의미 있게(반복 없이) 새로워야 하고 (3) 그 결과를 자동으로 신뢰도 있게 해석 가능하게 측정해야 한다는 점이다. 이를 위해 경험 있는 교육자 예시를 정밀 정렬하는 fine-grained prompting(FGP)과 단계적 정렬을 유도하는 Chain-of-Thought(CoT) 두 전략을 쓰고, OpenAI 기반 자동 판정(다수결 + 불일치 케이스 필터링)과 metric(범주 일치, 인지 전이 강도, 지식 커버리지)으로 다차원 진단을 수행한다.

- **Empirical Impact**: 실험 결과, FGP는 반복성을 줄이고(예: Qwen2.5-7B-Instruct에서 24.45% 감소) higher-order 출력 비율을 높이며(예: InternLM3-8B-Instruct에서 +11.53%) 문항의 실사용 적합성과 Bloom 레벨 정합성을 전반적으로 개선했다. 또한 InternLM3가 multi-level 전이에서 우수하며, CogShift·category drift 같은 지표가 “어디서 인지 제어가 무너지는지”를 보여줘 개인화 학습 시스템 배치에 필요한 벤치마크 성격을 갖는다. 아울러 metric-level 상관 분석을 통해 Chain-of-Thought prompting의 투명성을 높이는 해석 단서도 제공한다.



### Dynamic In-Group Persona Generation for Enhancing Human-AI Rappor (https://arxiv.org/abs/2606.18256)
- **Prior Approaches**: 기존 LLM 챗봇은 정적 backstory나 Big Five 같은 인격/스타일 프롬프트로 “말투의 페르소나”만 맞추거나, 범용 self-disclosure로 공감 효과를 유도하는 방식이 많았습니다. 하지만 이런 접근은 사용자의 현재 고민과 맞물린 맥락 정렬이 약해 rapport(관계감)와 개인적 관련성에서 일관된 향상을 보이기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 사용자가 먼저 말한 핵심 우려와 개인 맥락을 바탕으로, 비슷한 고민을 공유하지만 배경과 서사 디테일이 다른 “in-group persona(집단 내 인물 페르소나)”를 합성해 LLM에 조건화하는 In-group Persona Agent(IPA)를 제안합니다. 또한 IPA를 단순 프롬프트 변화가 아니라, 멀티스테이지 프롬프트 파이프라인으로 페르소나 생성과 대화 추론을 연결해 end-to-end 수준의 대화 일관성을 노립니다.

- **Technical Challenges**: 핵심 난제는 (1) 사용자 걱정을 충분히 수집해 페르소나가 실제로 “같은 집단성/같은 고민”을 담게 만드는 것, (2) 범용 self-disclosure과 구별되는 효과를 만들어내는 것입니다. 저자들은 사전 대화에서 2턴 간격으로 정보 sufficiency를 점검하며, 조건을 만족하면 사용자 정보에 기반해 5개의 후보 in-group persona를 생성하도록 설계했고, LLM 기반 루브릭으로 In-group Fitness와 Concern Resolution Quality를 평가해 품질을 검증합니다.

- **Empirical Impact**: career·employment 대화 시나리오에서 RCT로 IPA를 무페르소나(NoP), 대화 기록만 유지(NoPs), 그리고 최소 self-disclosure(NoPs의 변형)와 비교한 결과, IPA는 rapport 전반과 특히 “개인적 관련성” 및 관계 형성 항목에서 가장 큰 개선을 보였습니다. UX 측면에서는 engagement와 대화 지속 의사가 두드러지게 상승했으며, turn-level 분석에서도 사용자의 self-disclosure가 에이전트의 self-disclosure에 더 잘 “맞물리는(상호성/reciprocity)” 패턴이 관찰되어 관계 메커니즘을 뒷받침합니다.



### QSignAI: Quantum-Randomness-Seeded Identity Signatures at the Intersection of AI for Science and Science for AI (https://arxiv.org/abs/2605.27729)
- **Prior Approaches**: 기존 AI-참여형 시스템은 PRNG(의사난수)에 기반한 결정적 토큰을 사용해, 시드가 노출되면 재현 가능한 문제가 남아 있습니다. 한편 양자 회로는 비전문가에게 ‘보이는’ 형태로 전달되지 않아 교육·대중화가 별도 채널(전시/게임)에서 분리돼 왔습니다. 따라서 AI가 양자를 접근 가능하게 만들거나, 양자 난수를 신원·참여 시스템에 직접 결합한 사례는 부족합니다.

- **Core Contribution**: QSignAI는 텔레그램 기반 실시간 참여 시스템에서 AI-양자 양방향 관계를 ‘프로덕션 배포’로 시연합니다. 참여자의 첫 메시지를 양자 난수 파이프라인으로 보내고, 그 결과로 참가자별 고유한 ‘양자 난수 시드 identity signature’를 생성해 공개 배지(색/서명)로 시각화합니다. 또한 양자 회로는 보이지 않게 동작하다가, 공개 벽과 관리자 대시보드에서 양자 현상(예: Bell 상태 통계)을 청중이 이해 가능한 형태로 드러내도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 양자 측정으로 얻는 약한/비균일 난수를 낮은 지연으로 신원 토큰에 쓸 만큼 균일에 가깝게 만드는 것과 (2) 양자 현상을 대중이 ‘감각적으로’ 해석할 수 있는 UX로 매핑하는 것입니다. 논문은 독립 단일 큐비트 Hadamard 측정 결과(SV1: 이상 소스, DM1: 잡음 소스)를 두-소스 추출기 Toeplitz two-source extractor로 응축해 균일성에 가까운 비트를 만들고, 이를 256비트 nonce와 ToyLWE-유사 서명(shaKE-256 혼합)에 결합합니다. 여기에 벨 상태의 측정 분포를 카드 색으로 인코딩하고, Braket 태스크 실패/타임아웃 시 SHAKE-256 기반 로컬 fallback으로 지연을 막아 단일 장애점을 피합니다.

- **Empirical Impact**: 시스템은 AWS 인프라에서 실제 이벤트 참가자 데이터를 처리하며 ‘실제로 굴러간다’는 배포 성과를 보고합니다. 양자 회로 없이도 봇 토큰은 PRNG 기반과 구별되지 않는 수준으로 보이지만, 양자 회로만으로는 대중이 보지 못한다는 점을 들어 두 구성 요소의 결합 효과를 강조합니다. 다만 엄밀한 랜덤성 품질 비교(NIST SP 800-90B), SV1 대비 다양한 QPU/시뮬레이터의 지연 벤치마크, 사용자 연구(양자 리터러시 개선) 같은 정량 평가는 향후 과제로 남겼습니다.



New uploads on arXiv(cs.RO)

### Zero-Shot Long-Horizon Dexterous Manipulation via Multi-View 3D-Grounded VLM Reasoning (https://arxiv.org/abs/2606.19340)
- **Prior Approaches**: 기존 접근은 VLA 모델처럼 end-to-end로 행동을 직접 예측하거나, 대규모 로봇 데이터 및 task-specific fine-tuning/적응을 요구하는 경우가 많았습니다. 또 다른 축은 zero-shot로 foundation model을 쓰더라도, 중간 표현이 2D keypoint나 sparse 표시에 머물러 dexterous manipulation에 필요한 3D 접촉점·배치 목표·툴 궤적을 안정적으로 못 맞추는 취약점이 남아 있습니다.

- **Core Contribution**: 이 논문은 언어 지시를 calibrated multi-view RGB 이미지에 기반해 실행 가능한 3D task plan으로 “zero-shot” 변환하는 프레임워크를 제안합니다. 핵심은 VLM이 reference-view에서 semantic grounding과 primitive-level 2D keypoint를 만들고, 이를 multi-view fusion으로 3D로 lifting한 뒤, pick-and-place와 tool-use를 동일한 3D 앵커 기반 원시 동작 라이브러리로 실행한다는 점입니다.

- **Technical Challenges**: 문제는 2D 추론만으로는 depth 모호성과 가림(occlusion) 때문에 3D 접촉/궤적을 신뢰성 있게 복원하기 어렵다는 데 있습니다. 저자들은 교차 뷰 triangulation과 reference-view ray voting을 함께 써서 VLM의 view-dependent 2D groundings를 기하적으로 일관된 3D 후보로 정합하며, tool-use는 skill category에 맞는 Bag of Atomic Actions의 6D 툴 trajectory를 3D 키포인트로 정렬해 물리 실행 가능성을 높입니다.

- **Empirical Impact**: 실제 로봇 실험에서 multi-view 3D grounding은 단일뷰 RGB-D 기준선보다 grasp/apply-action localization과 충돌 위험을 개선했고, 특히 복잡한 클러터나 정밀 배치에서 격차가 커졌습니다. 또한 fine-tuned VLA baselines가 실패한 작업에서도 이 시스템은 zero-shot으로 성공하며, closed-loop status verification과 replan/retry로 긴 지평(long-horizon) 시퀀스를 확장해 보이지 않은 물체·새 장면에서의 실행 신뢰성을 입증했습니다.



### Do as I Do: Dexterous Manipulation Data from Everyday Human Videos (https://arxiv.org/abs/2606.19333)
Comments:
          Project website: this https URL

- **Prior Approaches**: 로보틱스의 숙련 조작은 전통적으로 텔레오퍼레이션이나 시뮬레이션 탐색으로 “직접 해보는” 경험 데이터를 만들었지만, 비용·전문성·환경/보상 설계가 병목이었다. 최근에는 human videos에서 동작을 추출하는 Do as I do 계열이 발전했으나, 대부분은 (1) 잡음이 큰 monocular RGB에서 hand-object 상호작용을 신뢰성 있게 복원하거나 (2) 손-물체 접촉과 물리적 타당성을 고려해 로봇으로 retargeting 하는 데 한계가 있었다. 특히 기존 human-to-robot retargeting은 깨끗한 기준선(예: MoCap 기반 ground-truth poses)을 가정하는 경우가 많아, 실제로는 “복원된 참조가 노이즈인” 인터넷 영상 적용이 어려웠다.

- **Core Contribution**: DO AS I DO는 monocular RGB 인간 영상을 dexterous multi-fingered 로봇 손에서 재현 가능한 조작 데이터로 바꾸는 end-to-end 파이프라인을 제안한다. 핵심은 ① hand-object 상호작용을 3D로 복원·추적해 “로봇이 따라야 할 reference”를 만들고 ② 그 reference를 물리 시뮬레이션 기반 dynamics-aware retargeting으로 로봇 실행 궤적(robot-complete)로 변환한다는 두 단계에 있다. 또한 grasping priors나 특정 물체 범주 같은 제한 가정을 줄여, 다양한 in-the-wild 소스(ego/exo, 심지어 생성 비디오)까지 확장하는 점이 기여로 요약된다.

- **Technical Challenges**: 가장 큰 기술 난제는 monocular RGB에서 물체 pose가 occlusion·저해상도·비디오 품질 저하에 무너지는 문제와, 그로 인해 노이즈/불연속 reference가 생성될 때 retargeting이 붕괴하는 문제다. 저자는 SAM 3D 기반의 guided diffusion 추적을 활용해 물체 shape는 고정(anchor)하고 pose만 프레임별로 갱신함으로써 temporally consistent한 pose evolution을 노렸고, hand-물체 스케일·중력 정렬까지 맞춰 near metric-space 기준을 구성한다. retargeting 단계에서는 noisy reference에서 초기 상태 불가능성을 줄이기 위한 warmup steps, 국소 최적에 갇힘을 완화하는 random force perturbation, rest/in-hand 전이의 실패를 직접 페널티로 다루는 transition reward를 더해 실험적으로 안정성과 성공률을 끌어올렸다.

- **Empirical Impact**: 실험에서는 DexYCB와 HOI4D의 ground-truth 기반 검증에서 hand-object 상호작용 추정 및 trajectory 추출 성능이 기존 SOTA를 능가했으며, 150개 in-the-wild/생성 비디오에서도 사람 평가 선호가 더 높게 나타났다(객체 pose 추적에서 67% 선호). retargeting은 자체 reconstructed in-the-wild 기준으로 success rate 25%에서 71%로 크게 개선했고, OakInk2에서도 구성요소를 추가할수록 72%→81%로 상승해 clean MoCap에서도 일반화 이득이 있음을 보였다. 더 나아가 500개 규모의 사람 검증 기반 dexterous 조작 궤적을 생성해 실제 로봇(Sharpa Wave 손, UR3e 팔)에서 다양한 grasp 유형으로 재현함으로써 “인터넷 영상→현실 dexterous rollout”에 가까운 데이터 스케일링 경로를 실증했다.



### Modeling Branches for Active Manipulation using Iterative Parameter Estimation (https://arxiv.org/abs/2606.19314)
Comments:
          Accepted to IROS 2026

- **Prior Approaches**: 기존 농업 로보틱스는 가지를 주로 장애물로 회피하거나, 단순히 로봇의 미리 짠 경로 밖으로 치우는 데 초점을 맞췄습니다. 일부 연구는 가지 조작을 다뤘지만 기계적 응력·변형을 안전 기준과 연결해 최적화하진 못했습니다. 또 deformable object 조작 쪽은 정확한 모델이 필요하지만, 가지는 형상·재료 특성이 인스턴스마다 크게 달라 CAD 기반 접근으로는 한계가 컸습니다.

- **Core Contribution**: 이 논문은 포인트클라우드에서 가지의 토폴로지와 단면 두께를 복원해 테트라hedral(사면체) FEM 시뮬레이션 모델을 만드는 파이프라인을 제안합니다. 이후 관절 기반이 아닌 FEM 변형을 사용해, 변형 에너지(손상 위험의 물리적 지표)를 제약으로 포함한 deformation-aware motion planning을 수행합니다. 또한 관찰된 변형 데이터로 material parameters를 반복 추정해 시뮬레이터의 물리 정확도를 맞춥니다.

- **Technical Challenges**: 핵심 난제는 (1) 가지의 큰 비선형 변형을 안정적으로 재현할 수 있는 모델링, (2) 인스턴스별로 달라지는 공간적 재료 파라미터를 수동 튜닝 없이 추정, (3) 접촉 지점이 움직이는 active manipulation에서 변형 비용을 경로 탐색에 반영하는 것입니다. 저자들은 LBC로 1D 스켈레톤을 만든 뒤 generalized cylinder로 표면을 구성하고 fTetWild로 사면체 메시를 생성해 FEM에 적합한 형상을 만들었습니다. 파라미터는 로봇이 가한 힘과 카메라로 관찰된 변형 궤적의 차이를 spatio-temporal loss로 정의하고, 비미분 가능 시나리오에서 Nelder–Mead로 gradient-free 추정했습니다. 마지막으로 D-RRT*에서 local connection의 strain-energy 기반 비용을 평가하고, 미리 시뮬레이션한 grasp-point 에너지 데이터를 RBF로 보간해 새로운 구성에서도 deformation-aware 비용을 빠르게 계산합니다.

- **Empirical Impact**: 실험에서는 젊은 가지·성숙 가지·인공 가지 등 서로 다른 형상/재료를 대상으로 30개 트라이얼을 수행했으며, 제안 방식은 변형 에너지를 평균 35.69% 낮추면서 경로 길이는 평균 8.10% 늘리는 결과를 보였습니다. 또한 D-RRT*는 vanilla RRT* 대비 같은 목표로 가되 변형 비용이 큰 구성을 피하는 경향이 관찰됐습니다. 결과적으로 섬세한 생체 가지를 ‘회피’가 아니라 ‘안전한 조작’으로 접근할 수 있는, 물리적으로 해석 가능한 계획·시뮬레이션 흐름을 제공한다는 점에서 농업 로보틱스 및 deformable manipulation 연구에 의미가 있습니다.



### Observability and Consistency Analysis for Visual-Inertial Navigation with Anchored Feature Parameterizations (https://arxiv.org/abs/2606.19307)
Comments:
          Accepted to IEEE/RSJ IROS. 8 pages, 3 figures, 4 tables

- **Prior Approaches**: 기존 filter-based VINS(예: MSCKF 계열)은 EKF로 각 시점에서 선형화해 계산 효율이 좋지만, 선형화 기준점 불일치로 과도하게 확신하는 비일관성(inconsistency)이 잘 발생한다. 이를 줄이기 위한 FEJ, observability-constrained(OC), right-invariant filtering(RI-EKF), robocentric, transformation-based/affine EKF 등 여러 “일관성 개선” 기법이 제안돼 왔다. 다만 대부분의 관찰가능성·일관성 분석은 landmark를 global reference frame에 직접 두는 경우에 집중했고, anchored feature처럼 랜드마크 파라미터화를 바꿀 때의 영향은 체계적으로 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 anchored feature 표현을 쓰는 filter-based VINS의 관찰가능성(unobservable subspace)과 consistency를 수학적으로 분석한다. 그 결과, anchored landmark 파라미터화에서 unobservable subspace는 landmark 상태 자체에는 독립이지만 navigation state에는 여전히 의존함을 보인다. 이 통찰을 바탕으로, 추가 수정 없이도 “랜드마크 선형화 기준점” 변화가 일관성에 미치는 부정 영향을 줄일 수 있음을 제시한다.

- **Technical Challenges**: 핵심 난제는 anchored landmark가 만들어내는 관찰가능성 구조를 EKF의 선형화 관점에서 정확히 분해하고, unobservable 방향이 어떤 상태(랜드마크 vs 내비게이션)와 연결되는지 분명히 증명하는 것이다. 논문은 unobservable subspace의 의존성을 명시적으로 도출한 뒤, navigation state 의존성 때문에 남는 비일관성을 해결하기 위해 FEJ와 invariant error 정의 같은 기존 consistency-enforcing 기법을 anchored 설정에 맞춰 적용하는 설계 관점을 제시한다.

- **Empirical Impact**: 시뮬레이션에서는 anchored feature paramterization을 쓰는 모든 추정기가 global frame에서 feature를 직접 추정하는 방식보다 일관성이 더 좋아졌고, 특히 feature initialization이 나쁠 때 차이가 두드러졌다. TUM-VI 실세계 실험에서도 anchored feature 표현만으로도 consistency-improved 전역 feature 기반 추정기와 비슷한 성능을 내는 결과가 나와, anchored feature 파라미터화 자체가 VINS 실용성에 유리할 수 있음을 보여준다.



### A Mixed-Reality Testbed for Autonomous Vehicles (https://arxiv.org/abs/2606.19267)
Comments:
          9 pages, 7 figures, 1 table

- **Prior Approaches**: 기존 AV/CAV 연구는 CARLA 같은 고정밀 시뮬레이터와, Duckietown/Robotarium 같은 물리 테스트베드로 “sim-to-real”을 줄여 왔다. 하지만 시뮬레이터는 현실 변동성을 완전히 흡수하기 어렵고, 물리 테스트베드는 스케일(동시 에이전트 수)과 자율 스택(지각-계획-제어-통신) 동시 검증에 제약이 있다. 또한 디지털 트윈 기반 sim-to-real 접근은 존재해도 로보틱스 조작 중심에 머무르는 경우가 많고, 공용 접근성이 낮아 반복 검증/재현이 어렵다는 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 CARLA 기반 고정밀 가상 환경과 이동 로봇 물리 하드웨어를 Hardware-in-the-Loop(HIL)로 결합한 mixed-reality 스마트 시티 테스트베드를 제안한다. RSU(도로변 유닛)를 중심으로 V2X(V2V·V2I) 통신, 멀티모달 센서(카메라·LiDAR·IMU 등), 물리 로봇과 디지털 트윈/가상 에이전트를 함께 운용해 스케일을 확장한다. 더불어 CAV를 위한 safety-guaranteed 프레임워크로 지각-계획과 함께 Control Barrier Functions(CBF) 기반 안전 제약을 내재화한 online learning 제어기를 통합한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 물리 로봇의 불확실성과 시뮬레이터의 다양성을 동시에 만족시키는 실시간 mixed-reality 동기화, (2) 다중 에이전트에서 안전 제약을 보장하면서도 성능 손실을 최소화하는 제어 설계, (3) 학습 과정에서도 안전성이 깨지지 않게 만드는 학습-제어 결합이었다. 저자들은 디지털 트윈을 CARLA 내에서 물리 다이내믹을 끄고 상태만 동기 텔레포트하여 물리-가상 공존을 구현하고, RSU-ROS 통신 및 MOCAP 기반 포즈 추정으로 지연을 줄였다. 제어 쪽에서는 고차 CBF(HOCBF)로 merging/후미 추종 등 충돌 위험을 제약으로 바꾸고, QP 형태의 CBF-QP를 통해 매 시점 안전 제어를 강제한 뒤 그 파라미터를 self-supervised 온라인 학습으로 갱신한다.

- **Empirical Impact**: 실험은 제안한 안전 제약 통합 프레임워크가 학습과 검증 단계 모두에서 핵심 기능을 수행하며 sim-to-real 격차를 메울 수 있음을 보여준다. 특히 CBF를 통해 안전 제약을 훈련 중에도 유지하므로, 기존의 “학습은 자유롭고 실행은 보수적으로 안전장치를 얹는” 방식보다 안전성이 일관되게 관리된다. 또한 공개 및 원격 접근을 지향하는 테스트베드 특성상, 향후 AV/CAV 연구에서 멀티에이전트·통신·전 지능 스택을 함께 검증하는 표준 플랫폼 역할을 기대할 수 있다.



### Shape Sensing of Continuum Robots using Direct Laser Writing (https://arxiv.org/abs/2606.19265)
Comments:
          This work has been submitted to the IEEE for possible publication

- **Prior Approaches**: 연속 로봇은 유연성과 순응성 덕분에 자연구강 수술에 유망하지만, 변형이 커서 실시간 형상/자세 추정이 어렵다. 기존에는 MRI·초음파·투시 같은 영상 기반, EM 트래커 같은 전자기 센서, fiber Bragg grating 같은 광학 센서, 저항형(strain) 센서 등 다양한 방법이 제안됐다. 다만 영상은 충분한 샘플링 속도와 안전성(잡음·방사선) 문제가 있고, 전자기 센서는 수술실 전자기 간섭, 광학 센서는 취약성과 비용 문제가 남았다.

- **Core Contribution**: 이 논문은 direct laser writing(DLW)로 만든 graphene 스트레인 센서를 연속 로봇 관절(notched-tube)과 ‘일체형’(monolithic)으로 가공하는 공정을 제안한다. 같은 레이저/세팅에서 센서 요소와 노치를 순서대로 제작해 기존처럼 별도 조립(정렬/오정렬로 인한 오차)을 줄이는 것이 핵심이다. 또한 DLW 센서의 히스테리시스·비선형성까지 포함한 모델링을 통해 관절 각도 추정과 closed-loop 제어까지 연결한다.

- **Technical Challenges**: DLW 저항 센서는 strain-저항 관계가 선형이 아니고 히스테리시스가 존재하며, 전이(속도)나 이완(relaxation) 같은 동역학도 영향을 줄 수 있다. 저자들은 generalized Prandtl-Ishlinskii(PI) hysteresis 모델로 히스테리시스와 포화 비선형을 함께 포착하는 forward/inverse 모델을 구성해, 측정 저항으로 각도를 복원한다. 더 나아가 joint 내부에 두 센서를 둔 설계에서는 장력/압축 각각에 대해 서로 다른 선형 모델을 피팅하고 두 관측을 결합해 강건한 추정을 시도한다.

- **Empirical Impact**: 실험에서 DLW 센서-관절 각도 예측은 최소 1.76 도 수준의 오차까지 달성했으며, 두 센서 결합 시 RMSE가 약 3.13 도로 개선됐다. 또한 PI/역모델 기반 추정치를 피드백으로 넣은 closed-loop 제어 데모에서 tracking error가 3 도 미만으로 유지되는 결과를 보였다. 영상·EM 대비 안전성과 통합 용이성을 높일 수 있는 ‘레이저 기반 센서 일체형’ 접근으로, 최소 침습 로봇의 실시간 제어 연구에 실증적 동기를 제공한다.



### Seeing Through Occlusion: Deterministic Arm Kinematic Correction for Robot Teleoperation (https://arxiv.org/abs/2606.19240)
- **Prior Approaches**: 단일 RGB-D 카메라 markerless 모션캡처는 설치가 쉽지만, self-occlusion 상황에서 depth 추정이 흔들리며 텔레오퍼레이션 중 로봇 동작이 불안정해질 수 있다. 이를 줄이기 위해 KF/EKF 같은 필터링이나 particle filter, 최적화 기반 inverse kinematics(COIK), 학습 기반/하이브리드 기법들이 제안됐으나, 복잡한 모델 설계·파라미터 튜닝·반복 연산이 필요하거나 장시간·심한 occlusion에서 안정성이 떨어질 수 있다. 또한 기존 방법은 관절을 독립적으로 처리하는 경우가 많아, 팔 길이 같은 해부학적 제약을 명시적으로 강제하지 못해 kinematic 불일치가 누적될 여지가 있다.

- **Core Contribution**: 본 논문은 Arm Kinematic Correction(AKC)이라는 후처리 보정 모듈을 제안해, self-occlusion으로 깨진 depth와 kinematic 불일치를 동시에 완화한다. 상완·전완 길이가 일정하다는 기하 제약을 Pythagorean theorem 기반의 결정론적(deterministic) 복원으로 강제하고, 관절 깊이를 후보군으로 재구성한 뒤 KF 기준과 시간/해부학적 일관성 비용함수로 최적 후보를 선택한다. 그 결과 확률 모델이나 복잡한 최적화 없이도 해부학적으로 일관된 팔 자세를 만들어 모션-매핑 텔레오퍼레이션에 활용한다.

- **Technical Challenges**: 핵심 기술 난제는 occluded joint의 depth가 부정확해질 때 가능한 기하해가 여러 개(± 해)로 분기되고, 관측 잡음으로 인해 radicand가 음수가 되어 비현실적 해(imaginary)가 생길 수 있다는 점이다. 논문은 손목이 카메라에 가깝고 덜 가려진다는 가정을 두고 팔 길이 제약으로 팔꿈치/어깨 깊이를 후보로 복원하되, 음수 radicand 발생 시에는 최소 변화로 “feasible surface”에 투영하는 보정 장치를 둔다. 또 해가 0-depth 평면 근처에서 불안정해질 때를 대비해 탐색 시 팔 길이를 shrink factor로 줄여 후보 선택의 견고성을 높였다.

- **Empirical Impact**: Vicon(정답) 대비 Intel RealSense D435 단일 RGB-D 실험에서 AKC는 static·dynamic 모두에서 RMSE와 Pearson correlation으로 depth 품질을 개선하며, 특히 장시간 occlusion(팔꿈치/어깨)에서 KF보다 큰 폭의 오차 감소를 보였다. 또한 반복 최적화 기반 COIK와 비교해 AKC는 determinism과 지연(latency) 측면에서 유리하며, 추가 오버헤드는 약 11ms 수준으로 실시간 텔레오퍼레이션에 더 적합하다는 점을 강조한다. 최종적으로 AKC는 긴 시간·심한 occlusion에서도 해부학적 팔 길이를 일정하게 유지하고, 시간 필터 신뢰도가 낮아도 robustness를 보이며 시뮬레이션과 실제 로봇 환경에서 motion-mapping 텔레오퍼레이션을 성공적으로 시연했다.



### Mobile Pedipulation for Object Sliding via Hierarchical Control on a Wheeled Bipedal Robo (https://arxiv.org/abs/2606.19233)
Comments:
          8 pages, 7 figures

- **Prior Approaches**: 기존의 평면 슬라이딩 조작 연구는 매니퓰레이터 말단에서 상호작용 힘(웬치)을 직접 정밀 제어할 수 있다고 가정하는 경우가 많았고, 그 단순화로 최적화 계산을 줄여왔다. 반면 wheeled bipedal은 휠-지면 구속이 nonholonomic rolling 제약을 따르며, 게다가 underactuated 구조라 반작용 힘을 직접 조절하기 어렵다. 또한 모델 기반 MPC를 쓰더라도 많은 단순화 모델이 no-slip rolling만 가정하거나 hip roll 같은 횡방향 핵심 자유도를 빼 비대칭/슬립/비접촉 모드를 다루기 어려웠다.

- **Core Contribution**: 이 논문은 wheeled bipedal 로봇이 휠 다리로 planar object sliding(비집기/비선형 집기 계열) 태스크를 수행하도록 하는 계층형 제어 프레임워크를 제안한다. 핵심은 hip roll 자유도(hip roll DoF)와 여러 wheel-environment contact mode를 명시적으로 반영한 reduced-order three rigid bodies(TRB) 모델 기반 NMPC로, 이동(로코모션)과 상호작용 힘 제어를 하나의 최적화 틀에서 동시 조절한다. 또한 stick-slip 전이를 포함하는 기준 궤적을 만들기 위해 trajectory-optimization 기반 로봇-물체 모션 플래너를 설계한다.

- **Technical Challenges**: 가장 큰 난제는 (1) 휠의 rolling/슬립/비접촉 같은 접촉 모드를 포함하면서도, (2) underactuated 로봇의 내부 구동과 물체 접촉 동역학을 온라인 NMPC가 다룰 수 있을 만큼 축약하는 것이다. 저자들은 TRB에서 다리 질량을 가정으로 줄이되 hip roll을 유지하고, 접촉 속도 제약을 통해 모드별로 no-contact/rolling/slipping을 구분하는 구조로 reduced-order 동역학을 구성했다. 추가로 물체는 point-mass로 두고 Coulomb friction complementarity constraint를 포함한 TRB-Object(TRBO)로 확장하되, NMPC 온라인 실행을 위해 물체 동역학을 1D 슬라이딩 방향으로 단순화해 최적화 복잡도를 낮췄다.

- **Empirical Impact**: 실험은 Tron1 하드웨어에서 scooting(스코팅)과 lateral sliding(측면 슬라이딩) 같은 pedipulation 동작을 실제로 검증하며, desk 아래 끼인 1 kg 물체 회수와 0.228 m 스코팅 슬라이딩을 성공적으로 수행했다고 보고한다. 시뮬레이션에서는 마찰계수 0.30.3(문맥상 0.3)인 미끄러운 표면에서도 미끄러짐을 유발하는 슬라이딩을 높은 하중(최대 2323 kg 수준까지)에서 시작할 수 있어 강건성을 시사한다. 무엇보다 stick-slip을 인지하는 궤적 계획과 TRB/TRBO 기반 NMPC의 결합이 wheeled bipedal의 동적 비집기(non-prehensile) 상호작용을 실물로 처음 보여주는 성격의 성과로 평가된다.



### Constant Time-Delay Leader Following with Neural Networks and Invariant Extended Kalman Filters for Arbitrary Trajectories (https://arxiv.org/abs/2606.19227)
Comments:
          9 pages, 6 figures

- **Prior Approaches**: 기존 리더-팔로워 컨보이 연구는 차량 간 통신을 가정하거나, 통신이 끊긴 경우에도 지도/전역 좌표(GPS 등) 없이 센서 기반 상대 국소화와 추적을 결합하는 방식이 많았습니다. 반면 perception 중심 접근은 추적을 부차로 두거나 리더 움직임의 구조를 가정하는 경우가 많고, no-communication/time-delay 설정에서 time-delayed 리더 궤적을 안정적으로 추정·제어하는 데는 한계가 지적됩니다. 또한 많은 상태 표현이 벡터 공간 ℝ3에 머무르거나 특이점 회피를 위한 우회가 필요해, SE(2) 같은 기하적 구조를 직접 활용하기 어렵다는 문제가 있습니다.

- **Core Contribution**: 이 논문은 통신·공통 좌표계·전역 포지셔닝 없이, 고정된 time-delay(P timesteps)를 가진 리더의 상대 궤적을 SE(2) 매니폴드 위에서 “상수 시간” 내 추정·추적하는 방법을 제안합니다. 핵심은 probabilistic Seq2Seq로 time-delayed 관측 시퀀스를 받아 리더 상대 궤적의 분포(평균·공분산)를 예측하고, IEKF(invariant extended Kalman filter)로 warm-start를 제공해 SE(2) 상의 추정 정확도를 높인다는 점입니다. 이어서 예측 결과를 더 잘 활용하도록 geometric model predictive controller(GMPC) 통합을 통해 제어 성능을 개선합니다.

- **Technical Challenges**: 기여를 실현하려면 (1) delayed 관측만으로 불확실성이 누적되는 미래 리더 상태를 매니폴드(SE(2))에서 예측하고, (2) 비선형·비정상 운동에서 EKF류가 발산하지 않도록 초기화와 수렴성을 관리하며, (3) 예측 오차를 제약 조건이 있는 제어 최적화에 매끄럽게 연결해야 합니다. 논문은 IEKF의 불변 오차(invariant error)로 SE(2) 표현에 맞는 warm-start를 만들고, Seq2Seq 출력에 공분산을 포함한 확률적 헤드를 두어 negative log-likelihood(NLL) 손실을 SE(2)용 boxminus 연산에 맞춰 학습합니다. 또한 학습 시 초기 상태에 잡음을 주입해 IEKF의 “현실적” 오차 상황에 대한 강건성을 확보하고, GMPC 단계에서는 입력 변화량 제약으로 가속도 제한 및 과도한 transient 오차를 완화합니다.

- **Empirical Impact**: 시뮬레이션에서 순수 IEKF 기준선, MSE 학습 기반 GRU/Transformer 계열과 비교하며, ground-truth 궤적 대비 tracking error와 추정 품질을 통해 제안 방법의 효율성과 정확성을 검증합니다. 추가로 실제 로봇 실험에서도 동일한 컨셉(비통신·relative observation 기반, time-delay 처리, SE(2) 기하 제어)이 유효함을 보여줍니다. 특히 “긴 time delay에서도” 추적이 가능한 확률적 매니폴드 예측-제어 파이프라인을 제시해, 도메인 전문가 지식 의존도를 낮추면서 컨보이/플래툰 제어의 실용성을 끌어올린다는 점에서 의미가 큽니다.



### Invertible Neural Network Adapter for One-Step Flow Matching in Robot Manipulation (https://arxiv.org/abs/2606.19194)
- **Prior Approaches**: 기존 diffusion-based policy learning과 최근의 flow matching은 고차원·멀티모달 행동 분포를 잘 모델링하며 로봇 조작 성능을 끌어올렸습니다. 다만 기존 flow-matching 기반 one-step 정책들은 단일 추론으로 인한 근사 오차, 장수행(긴 시퀀스)에서의 temporal consistency 저하, 다지(멀티-핑거) 상호작용의 미세 동역학 누락, 그리고 VLA(vision-language-action)에서 관측을 원칙적으로 결합하는 데 한계가 있었습니다. 또한 일부 방법은 auxiliary loss를 늘려 one-step을 맞추려 하지만, 배포 시 실제 mean velocity field와의 불일치로 강건성·일반화가 떨어질 수 있습니다.

- **Core Contribution**: 이 논문은 invertible neural network adapter(가역 신경망 어댑터)를 one-step flow matching 정책에 결합해, 행동 생성 경로를 invertible latent space 안에서 구조적으로 제약합니다. 이렇게 하면 단 한 번의 denoising으로도 latent 정보의 비가역적 손실을 막아 flow dynamics의 구조적 일관성을 유지하면서, 행동을 더 정확하고 안정적으로 생성합니다. 결과적으로 속도장(velocity field) 학습이 단순 근사에 그치지 않고, 다중 입력(시각·언어·고유수용감각)을 반영한 더 신뢰도 높은 행동 예측으로 이어집니다.

- **Technical Challenges**: 핵심 난제는 “한 단계 생성”이라는 공격적인 가속에서 발생하는 벡터장/경로 근사 오차를 어떻게 줄이느냐입니다. 연구진은 coupling layer 기반 invertible neural network로 순전·역변환을 정확히 일치시키고(데이터 공간과 latent 공간에서의 reconstruction/adapter loss), 원래 one-step flow matching이 만든 예측을 가역 변환으로 정제하는 방식을 채택했습니다. 또한 VLA 설정에서는 Qwen-VL-3 기반 모델의 마지막 몇 개 transformer 계층에서 토큰을 뽑아 adapter의 조건으로 사용해, 관측 이질성을 효율적으로 통합하도록 설계했습니다.

- **Empirical Impact**: RoboTwin 시뮬레이션(2D RGB와 3D point cloud)과 Libero 언어-조건 조작, 그리고 실제 UR/OpenArm 로봇 실험에서 일관된 성능 향상을 보였습니다. 특히 one-step으로 10-step denoising을 대체하면서도 평균 task success rate가 3.2%p 개선되는 등 정확도와 안정성을 유지했습니다. 실세계에서는 Pi 0.5+Adapter가 평균 추론 지연을 110ms에서 61ms로 약 45% 줄이면서(단일 추론), 블록 스태킹처럼 정밀도가 중요한 과제에서 성공률이 80%에서 90%로 상승하는 결과를 냈습니다.



### FAST-LIVGO: A Degeneracy-Robust LiDAR-Inertial-Visual-GNSS Fusion Odometry (https://arxiv.org/abs/2606.19190)
Comments:
          Accepted for presentation at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **Prior Approaches**: 기존 LiDAR-Inertial-Visual Odometry(LIVO)는 로컬 정확도는 강하지만, 전역 제약이 없어 장거리에서 누적 드리프트가 커지고 Z축 방향으로 쉽게 발산할 수 있다. 또한 기하학적으로 열악하거나 텍스처가 부족한 장면(무특징 벽/광장, 고속 모션 블러)에서는 시각·기하 열화가 동시에 발생해 추정이 흔들린다. GNSS를 섞는 연구는 많지만, 특히 loosely-coupled은 예측/리젝션에 필요한 정보가 부족하고, tightly-coupled은 오도(오류 측정치)를 적응적으로 다루지 못하면 필터가 붕괴하기 쉽다.

- **Core Contribution**: 이 논문은 Error-State Iterated Kalman Filter(ESIKF) 기반으로 LiDAR-Inertial-Visual-GNSS를 tightly coupled 통합해, 장기·대규모·고동역학 환경에서 전역 일관된 상태 추정과 맵을 목표로 한다. 핵심은 (1) DTW 기반 온라인 spatiotemporal alignment로 동기화 오차를 줄이고, (2) Doppler와 fixed-anchor Time-Differenced Carrier Phase(TDCP) 관측모델로 GNSS 정밀도를 준-밀리미터 상대 제약으로 반영하되 과거 anchor state를 늘리지 않는 것이다. 여기에 LIVO degeneracy에 따라 outlier rejection 모드를 전환하는 적응형 무결성 전략을 더해, odometry가 열화될 때도 시스템이 버티도록 설계했다.

- **Technical Challenges**: 난제는 (a) 저가 센서의 시간 동기화가 고속 환경에서 크게 흔들리고, (b) GNSS는 multipath/NLOS로 오염된 관측치가 필터를 직접 망가뜨릴 수 있으며, (c) LIVO의 기하/텍스처 열화 상태를 실시간으로 신뢰도 있게 추정해 융합 강도를 바꿔야 한다는 점이다. 논문은 DTW로 LIVO- GNSS 속도 시퀀스를 정렬해 clock offset을 보정하고, ESIKF에서 Doppler와 fixed-anchor TDCP 잔차를 통해 전역 제약을 촘촘히 주입한다. outlier rejection은 RAIM(내부 무결성) 1차 거른 뒤, LIVO Hessian의 최소 고유값을 사용해 well-conditioned/degenerate를 판정하고 Chi-square 기반 엄격 검증과 GNSS-aided recovery를 오가며 안정성을 확보했다.

- **Empirical Impact**: 실험은 공개 M3DGR 데이터셋과 20 m/s 고속 고정익 UAV 데이터(및 핸드헬드)에서 수행되었으며, 누적 드리프트와 맵 고스트(ghosting)를 줄이면서 accuracy와 robustness 모두에서 SOTA 대비 우수한 성능을 보였다. 특히 GNSS-degraded 환경에서는 robust 모듈을 끈 ablation이 궤적 divergence와 시스템 failure로 이어졌지만, 제안한 degeneracy-aware dual-mode 전략을 적용한 완성 모델은 FAST-LIVO2에 가까운 수준을 유지하며 LIO-SAM/LIGO보다 훨씬 안정적이었다. 즉, 준-정밀 carrier phase 기반 tight coupling을 살리면서도 failure 없이 생존(survivability)하는 설계가 실증적으로 확인됐다는 점에서 의미가 크다.



### Learning to Annotate Delayed and False AEB Events: A Practical System for Extreme Class Imbalance and Asymmetric Label Nois (https://arxiv.org/abs/2606.19186)
Comments:
          8 pages, 5 figures, accepted by IEEE International Conference on Robotics and Automation (ICRA)

- **Prior Approaches**: AEB 최적화는 현실의 delayed/false trigger를 정확히 찾아야 가능하지만, 이 소수 샘플(<5%)은 수작업 검증 비용이 커 대규모 학습이 막혀 왔다. 기존 imbalanced classification/label noise 대응(재가중, focal loss, resampling, 불확실성 기반 정화 등)은 이 논문이 겪은 “극단적 불균형 + 비대칭 라벨 노이즈” 조합에 그대로 적용하기엔 한계가 있었다.

- **Core Contribution**: 논문은 최초로 AEB용 자동 annotation framework를 제안한다. 소수 delayed/false trigger의 재현율을 끌어올리면서, 잘못된 라벨(대부분 true로 라벨링된 오염)을 자동으로 억제해 사람 검수 부담을 줄이는 데 초점을 둔다.

- **Technical Challenges**: 핵심 장애물은 (1) delayed/false가 true에 압도되는 extreme class imbalance로 인해 minority 학습 신호가 약해지는 문제, (2) delayed/false는 거의 정확 라벨인데 true 쪽에 약 1% 수준의 비대칭 라벨 noise가 있어 majority가 minority 예측을 억제하는 문제다. 이를 위해 delayed/false를 AEB-타깃 방식으로 합성하는 data augmentation(초점 타깃 속성 조작, ego-vehicle dynamics 이식, 비초점 에이전트 마스킹, 교통 물리성 검증)과, majority 내 오염 샘플을 제거하는 noise suppression(EMA 기반 hardness 추정 + probe-guided adaptive threshold)을 함께 설계했다.

- **Empirical Impact**: 100,000 규모 AEB 트리거 데이터에서 기존 베이스라인 대비 delayed/false trigger 성능이 유의미하게 개선됐다. Production 배포 결과 delayed/false trigger recall 80% 개선, 수작업 workload 50% 감소를 보고하며, 누적된 검증/정화 데이터를 바탕으로 self-evolving annotation이 가능하다는 점에서 현업 AEB on-vehicle 최적화의 데이터 기반을 마련했다.



### Hardware- and Vision-in-the-Loop Validation of Deep Monocular Pose Estimation for Autonomous Maritime UAV Fligh (https://arxiv.org/abs/2606.19176)
Comments:
          6 pages 9 figues

- **Prior Approaches**: 기존 연구는 GPS 같은 외부 인프라 없이도 단안 비전으로 선박-상대 6D 포즈를 추정하려 했지만, 핵심은 결국 바다(현장) 검증의 난이도와 비용이었다. 특히 해상 환경은 기상 의존성이 크고 실험 실패 리스크가 높아, 시뮬레이션에서 얻은 성능을 바로 closed-loop로 옮기기 어려웠다. 또한 단순 시뮬레이션 중심 평가는 지연, 비동기 측정, 계산 자원 경쟁 같은 임베디드 실전 효과를 충분히 반영하지 못했다.

- **Core Contribution**: 이 논문은 3D Gaussian Splatting(3DGS) 기반 photorealistic 해상 장면 렌더링을 실시간으로 스트리밍하고, 온보드 Transformer 기반 단안 포즈 추정(TNN-MO)과 DKF 기반 지연 보정을 결합해 완전한 closed-loop 자율비행을 하드웨어로 검증했다. 렌더링/통신/추론으로 인해 발생하는 delayed vision 측정을, delayed Kalman filter로 적절한 과거 시점에 업데이트한 뒤 현재 시점 상태로 재전파해 기하 제어에 일관된 상태를 제공한다. 결과적으로 ‘시뮬레이션-해상배치’ 사이의 안전하고 현실적인 중간 단계(hardware-realistic intermediate stage)를 제시한다.

- **Technical Challenges**: 주요 기술 난관은 (1) end-to-end perception latency(렌더링·전송·추론 지연)와 (2) 불규칙한/asynchronous 업데이트, (3) Jetson Orin NX에서의 자원 제약이 perception–estimation–control 결합 성능을 흔든다는 점이다. 논문은 지연 측정의 out-of-sequence 문제를 DKF의 히스토리 버퍼 업데이트와 재전파로 처리하고, 파이프라이닝을 통해 처리량(업데이트 주기)을 늘리는 동시에 throughput–latency 트레이드오프를 명시적으로 운용한다. 또한 TensorRT 변환(FP16)과 멀티스레딩으로 추론 지연을 줄이고, 고정 지연/고정 업데이트율 조건에서 안정성을 유지하도록 동기화 전략을 적용했다.

- **Empirical Impact**: 실험은 모션캡처 기반 indoor 환경에서 자율 이륙-궤적 추종-착륙을 수행하며, 지연된 단안 포즈를 DKF로 융합한 뒤 안정적인 closed-loop 추종을 달성했다. 정량적으로 position MAE 0.066m, velocity MAE 0.032m/s, attitude MAE 2.13°의 낮은 추정 오차와 함께 control MAE 0.089m, 0.062m/s, 4.00°를 보고했다. 특히 Wi-Fi 전송 지연 변동을 포함한 최악 조건(총 지연 약 0.55s)에서도 estimator consistency와 비행 안정성이 유지되어, 해상 UAV 자율성 개발에서 비용과 위험을 줄이는 검증 경로로 의미가 크다.



### HT-Bench: Benchmarking and Learning Dexterous Full-Hand Tactile Representations with Egocentric Vision (https://arxiv.org/abs/2606.19161)
Comments:
          9pages, 4figures

- **Prior Approaches**: 촉각 표현 학습은 물체 인식, 접촉 이해, 조작 피드백 등에서 활발했지만, 대부분이 특정 센서/로봇 배치와 단일 다운스트림 목표에 강하게 결합된 파이프라인 형태로 진행됐다. 그래서 학습된 특성이 ‘전이 가능한 촉각 표현’인지 ‘과제·센서 적응’인지 판단하기 어려웠다.

- **Core Contribution**: 이 논문은 촉각 표현 학습을 공정하게 비교하려는 시도 대신, 확장성 있는 설정인 egocentric vision(시점 중심 비전) + full-hand tactile data(전손 촉각)를 기반으로 HT-Bench를 제안한다. HT-Bench는 226개 작업에서 수집한 10M RGB 프레임과 7.8M 촉각 프레임을 모아, 촉각이 접촉 기하를 담는지, 비전과 정렬되는지, 그리고 미지 작업(OOD)으로 일반화되는지 다각도로 평가한다.

- **Technical Challenges**: 핵심 기술 과제는 이질적인 촉각 센서/배치 차이를 모두 흡수하는 ‘완전한 범용 벤치마크’가 아니라, 비교 가능한 대표 데이터 레짐을 정의한 뒤 촉각 인코더의 능력을 구조적·교차모달·시간적 관점에서 측정하는 것이다. 이를 위해 HandTouch는 vector-quantized(벡터 양자화) 비전-촉각 인코더를 설계하고, 단계적으로 (1) 촉각 공간 토폴로지(복원) (2) cross-modal masked tactile inpainting(비전 정렬) (3) multimodal tactile frame prediction(접촉 동역학)을 학습하도록 구성했다.

- **Empirical Impact**: HT-Bench에서 HandTouch는 기존 촉각 인코더 기준선들을 전반적으로 능가했으며, Recall@5는 74.65%에서 85.23%로, masked tactile inpainting RMSE는 0.022에서 0.010으로, vision-to-tactile synthesis의 OOD cIoU는 0.628에서 0.705로 개선됐다. 특히 대규모 시점 중심 비전-전손 촉각 데이터가 촉각 표현 학습을 평가하고 발전시키는 ‘확장 가능한 기반’이 될 수 있음을 실증했다.



### Viking Hill Dataset: A Lidar-Radar-Camera Dataset for Detection and Segmentation in Forest Scenes (https://arxiv.org/abs/2606.19154)
Comments:
          33 pages, 11 figures

- **Prior Approaches**: 기존 산림 로봇 연구는 주로 카메라와 라이다를 활용해 나무/식생을 탐지·분할해 왔지만, 흙·톱밥·낙엽 같은 오염과 시각적 잡음에 취약합니다. 또한 라이다·카메라 중심의 산림 데이터셋은 많아도, 레이더를 포함하면서 레이더-라이다-카메라가 동일 좌표계에서 정렬되고 공통 3D 라벨을 제공하는 경우는 사실상 없었습니다. 오프로드 레이더 연구는 존재하지만 2D 스핀 FMCW나 저수준 결과(강도 기반, 분류 부재)에 머물러 직접 비교가 어려웠습니다.

- **Core Contribution**: 이 논문은 고해상도 4D FMCW imaging radar를 라이다·RGB 카메라·IMU·RTK-GNSS와 함께 수집한 멀티센서 산림 데이터셋을 제안합니다. 두 계절(짧은 풀/긴 풀, 즉 낮은/높은 식생 상태)로 세션을 나누고, 라이다 기준의 3D cuboid 라벨(트렁크 직경 추정 포함)을 세 센서에 공통 의미 라벨로 매핑할 수 있게 했습니다. 이를 통해 레이더가 시각 열화·표면 오염·폐색 상황에서 실제로 유효 정보를 주는지 직접 검증하는 연구 기반을 제공합니다.

- **Technical Challenges**: 핵심 기술 난관은 희소한 레이더 point cloud에서 레이더만의 정확한 수작업 라벨을 만들기 어렵다는 점과, 센서 간 정합·동기·좌표계 일관성을 확보해야 한다는 점입니다. 저자들은 라이다로 3D cuboid(트렁크/수관 등)를 만든 뒤 ROS 기반 도구로 레이더·카메라에 동일 cuboid 라벨을 투영하는 방식으로 이 문제를 우회했습니다. 또한 RTK-GNSS 제약이 숲 아래에서 제한되는 환경을 고려해, 부분적으로 GNSS를 SLAM에 보조해 기준 라이다 맵을 만들고 세션 간 맵 드리프트(대략 ±30cm)를 허용 가능한 수준에서 운용합니다.

- **Empirical Impact**: MinkowskiUNet 기반 세그멘테이션 기준선에서 레이더는 지면/수관 같은 주요 클래스에서는 라이다와 경쟁력 있는 IoU(예: ground 91%, canopy 86%)를 보였지만, 트렁크 같은 기하학적으로 정교한 구조에서는 성능이 낮아(예: 56% vs. 라이다 74%) 라이다 대비 약점도 확인됐습니다. 교차 모달리티 분석과 DBH(흉고직경) 구간별 평가로, 시각 기반 탐지가 성공/실패하는 상황에서 라이다·레이더의 트렁크 분할 품질이 어떻게 달라지는지, 그리고 나무 크기에 따라 품질이 어떻게 흔들리는지도 보여줍니다. 무엇보다 정렬된 멀티모달·DBH 라벨을 공개함으로써, 산림 폐색 환경에서의 sensor fusion·매핑·로컬라이제이션 연구 확장에 실질적인 영향력을 줄 것으로 기대됩니다.



### Monocular 3D Occupancy Perception for Robots on Sidewalks via Hybrid 2D-3D Learning (https://arxiv.org/abs/2606.19122)
- **Prior Approaches**: 기존 3D occupancy(점유) 학습 파이프라인은 자율주행용으로 설계되어 LiDAR-RGB 동기 페어와 촘촘한 3D 감독에 의존하는 경우가 많다. RenderOcc류는 렌더링 기반으로 라벨 비용을 줄이지만, 싱글 모노큘러로는 3D 재구성 품질이 흔들려 2D→3D 기반 감독이 불안정해지기 쉽다.

- **Core Contribution**: WalkOCC는 사이드워크(보도) 로봇을 위한 모노큘러 3D 점유 인식을 목표로, 제한된 LiDAR-RGB 페어에서 pseudo 3D 감독을 부트스트래핑하고 대규모 unpaired 2D 이미지를 섞어 학습하는 하이브리드 프레임워크를 제안한다. 또한 평가를 위해 LiDAR-camera paired 시퀀스와 3D semantic occupancy 라벨을 포함한 Sidewalk3D 데이터셋과 벤치마크를 공개한다.

- **Technical Challenges**: 핵심 난제는 (1) 비싼 3D 라벨 없이도 기하 정합을 유지하면서 (2) 2D 전용 데이터로 학습 안정성과 OOD 일반화를 함께 확보하는 것이다. WalkOCC는 depth-aware lifting으로 3D 공간에 의미를 투영한 뒤, ray marching 기반 2D–3D consistency로 카메라 광선 상에서 2D 시각 신호와 3D 예측을 강하게 결합해 unpaired 2D 학습이 점유 경계를 흐리지 않도록 한다.

- **Empirical Impact**: Sidewalk3D에서 WalkOCC는 mIoU와 occ_IoU가 전반적으로 개선되며, 특히 curb/gutter 같은 미세 구조와 차량·보행자 등 안전 핵심 클래스에서 성능 향상이 뚜렷하다. Night/OOD 환경과 cross-embodiment(로봇 플랫폼·카메라 차이) 조건에서도 self-supervised 2D 단독 기준선 대비 일반화가 크게 좋아졌고, mIoU 기준으로 최대 15.6% 및 OOD mIoU의 의미 있는 상승(예: Night split 55%)을 보고한다.



### GCNGrasp-VP: Affordance-Guided View Planning for Efficient Task-Oriented Grasping (https://arxiv.org/abs/2606.19091)
Comments:
          Accepted to IROS 2026

- **Prior Approaches**: 기존 task-oriented grasping(TOG)은 초기 관측 프레임에 작업에 필요한 로컬 영역이 보일 것이라는 강한 가정을 둡니다. 이 때문에 occlusion(자기 가림·장애물 가림)이나 불리한 시점에서 TaskGrasp 같은 complete-view로 학습된 모델이 성능이 크게 떨어집니다.
액티브 관측(view planning)도 주로 task-agnostic grasping에 초점이 맞춰져, 기하 기반 가시성 최적화는 작업 제약을 보장하지 못하고 scene-uncertainty 기반 방법은 3D reconstruction 비용이 크며 정보 이득이 작업 의미와 어긋날 수 있습니다.

- **Core Contribution**: 논문은 occlusion이 있는 상황에서도 “작업 의미(semantic)로 필요한 부분을 보이게” 만드는 GCNGrasp-VP를 제안합니다. 핵심은 GCNGrasp-v2가 grasp 평가와 affordance field(작업 가능성 필드)를 동시에 예측해, candidate grasp 수에 덜 민감한 constant-time에 가까운 추론을 달성한다는 점입니다.
이 affordance field를 Affordance-guided View Planner(Affordance-VP)가 정보 이득(metric)으로 사용해, 장면 재구성 없이 작업 관련 영역을 향해 카메라를 이동시키도록 설계됐습니다.

- **Technical Challenges**: 가장 큰 기술 난제는 occluded 상태에서 grasp 모델이 작업에 중요한 로컬 영역을 정확히 추정하고, 이를 다시 view planning 신호로 전환하는 것입니다. 저자들은 segmentation 스타일 아키텍처로 object-task 특징을 candidate grasp와 분리해, affordance field를 효율적으로 생성하면서도 grasp 호환성 점수를 동일 프레임에서 다루게 했습니다.
또한 affordance supervision을 representative point 기반의 보조 손실로 구성하고, view 선택 시에는 표적 영역(고신뢰 점 클러스터)과 카메라 방향 정렬·가림 정도·고도 제한을 포함한 손실을 종합해 최적 후보 시점을 고릅니다.

- **Empirical Impact**: 실험에서 GCNGrasp-v2는 TaskGrasp 기반 TOG 예측 성능에서 상위권을 유지했고, 추론 시간과 GPU 메모리는 candidate 수가 늘어도 낮은 수준(대략 ms~수십 ms대)으로 유지됩니다. view planning 비교에서는 scene-uncertainty-driven baseline이 재구성에 의존하는 반면, Affordance-VP는 단 1회의 view adjustment로도 task-oriented grasp 예측 정확도를 크게 끌어올립니다.
실세계 검증에서도 단일 객체 시나리오의 grasp success rate가 전반적으로 상승하며(일부 태스크는 100%), 전반 지연도 재구성 기반 대비 매우 낮게 유지되어 실시간 로봇 상호작용에 의미가 큽니다.



### ART-VS: Adaptive Resolution Tiling for Vision Transformer Visual Servoing (https://arxiv.org/abs/2606.19089)
Comments:
          Accepted at IROS2026

- **Prior Approaches**: 기존 비전 서보링(VS)은 기준 이미지와의 오차를 줄이기 위해 IBVS나 PBVS를 쓰며, 전통적 특징 기반 방법은 정밀도는 높지만 가림·조명 변화에 취약하다. 딥러닝 기반은 강인성을 높이려 해도 보통 task-specific 학습이 필요해 일반성이 떨어지고, self-supervised ViT 특징을 쓰는 training-free 접근은 perturbation에서 정밀도·수렴 안정성의 한계를 동시에 맞닥뜨린다. 특히 full-resolution ViT는 더 촘촘한 특징으로 정밀도는 개선하지만, 글로벌 매칭으로 인해 미세 단계에서의 수렴이 흔들리거나 연산비용이 커진다.

- **Core Contribution**: 이 논문은 Adaptive Resolution Tiling Visual Servoing(ART-VS)로, 서보 진행 단계에 따라 특징의 해상도(및 매칭 범위)를 적응적으로 바꾸는 training-free 방법을 제안한다. 첫 단계에서는 ViT의 native 해상도에서 coarse patch aggregation으로 초기 정렬을 안정화하고, 오차가 충분히 줄면 tiled 고해상도 단계로 전환해 같은 타일 내 local 매칭만 허용함으로써 정밀도를 끌어올린다. 결과적으로 robustness와 precision의 동시 달성을 목표로 한다.

- **Technical Challenges**: 핵심 기술적 난제는 “해상도를 올리면 정밀도는 좋아지지만, 글로벌 매칭이 long-range mismatch를 유발해 수렴이 나빠진다”는 trade-off를 제어하는 것이다. ART-VS는 coarse-to-fine 전환 임계값 τ에 의해 단계 전이를 자동화하고, refinement 단계에서는 non-overlapping 타일을 나눈 뒤 원하는 타일 위치에 대해서만 best-buddy matching을 수행해 long-range 대응을 억제한다. 또한 타일 단위 처리를 병렬 배치로 수행해 연산 부담을 줄이며, 초기 in-plane 회전 정렬과(필요 시) 언어 유도 ROI tracking을 보완 장치로 붙인다.

- **Empirical Impact**: 실험에서 ART-VS는 perturbation 조건에서 수렴(convergence) 성공률 95.4%를 달성해 ViT-VS(76.6%) 대비 18.8%p, full-resolution ViT 처리(81.0%) 대비 14.4%p 개선했다. 위치 오차는 전자 대비 53% 줄였고, 실행 속도는 full-resolution 대비 10배 이상 빠르면서 VRAM은 27% 덜 사용한다. 또한 세 가지 ViT 백본에서 성능이 유지되며, 실제 로봇 그리핑에서도 투명 병 95/100, 신발 98/100의 성공률을 보여 category-level grasping의 practical generalization 가능성을 입증했다.



### ReSiReg: Towards Spatially Consistent Semantics in Language-Conditioned Robotic Tasks (https://arxiv.org/abs/2606.19088)
- **Prior Approaches**: 기존 비전-언어 모델(VLM) 기반 로봇 언어 추종은 이미지-캡션 대비학습을 바탕으로 언어와 시각을 정렬하지만, dense(패치 단위) 임베딩은 잡음이 많고 공간 일관성이 약하다는 한계가 반복적으로 지적돼 왔다. 이를 줄이려는 self-similarity 기반, 이상치 필터링, 또는 self-supervised(SSL) 백본을 추가로 조건화하는 방식이 있었지만, 대부분 백본별 튜닝이 필요하거나 계산 비용이 커졌다.

- **Core Contribution**: 본 논문은 공간적으로 일관된 VLM 중간 표현을 활용해, 최종 언어-정렬된 dense 임베딩을 post-hoc 방식으로 재구성하는 ReSiReg를 제안한다. 핵심 아이디어는 중간 토큰을 visual prototype(시각 프로토타입)로 클러스터링한 뒤, 각 패치를 프로토타입 수준 language embedding의 soft mixture로 재구성해 공간 일관성과 언어 정렬을 동시에 끌어올리는 것이다.

- **Technical Challenges**: 문제는 (1) VLM 중간표현은 공간 일관성이 있지만 언어-정렬이 부족하고, (2) 기존 hard 마스킹은 클래스 경계의 불연속화로 미세 구조를 잃기 쉽다는 점이다. ReSiReg는 same-backbone에서 중간표현을 클러스터링해 프로토타입별 언어 디스크립터를 만들고, 재구성 단계에서 soft affinity(온도 조절 softmax)와 low-affinity gating으로 프로토타입 혼합을 제어해 부드러운 공간 구조를 유지한다.

- **Empirical Impact**: 실험에서는 OVSS(오픈 보조형 의미 분할), 3D semantic mapping, 그리고 실제 언어-조건 조작 장면에서 dense retrieval 성능 향상과 target activation의 공간적 일관성 개선을 정량·정성으로 확인했다. 또한 로봇 온보드 적용을 염두에 둔 25M 파라미터 ViT-S급 compact dense VLM을 함께 제시하며, 임베디드 하드웨어에서도 ReSiReg Lite 기준 고처리량을 보고해 실용성의 근거를 마련했다.



### Sensor Configuration Matters: A Systematic Evaluation of Multimodal SLAM on Quadruped Robots (https://arxiv.org/abs/2606.19067)
- **Prior Approaches**: 기존 visual SLAM/visual-inertial SLAM 벤치마크는 주로 handheld, 드론, 휠 로봇의 비교적 부드러운 움직임에 맞춰져 있어, 사족 보행이 만드는 충격·진동·급격한 회전이 성능을 어떻게 바꾸는지 정량 평가하기 어려웠다. 관련 데이터셋도 대체로 고정된 고성능 센서 구성으로 로컬라이제이션 가능성을 확인하는 데 그쳐, 카메라 모달리티·셔터·IMU 틀에 따른 취약점을 분리하기가 힘들었다.

- **Core Contribution**: 이 논문은 ANYmal D 사족 로봇의 GrandTour 데이터셋을 활용해 visual/visual-inertial/LiDAR-visual-inertial SLAM들을 하드웨어 센서 구성 단위로 체계 평가한다. 카메라 모달리티(모노큘러/스테레오/RGB-D), 셔터 타입(global/rolling), IMU 티어(산업용/전술용)를 바꿔 정확도·견고성·연산 자원 간 트레이드오프를 분해해 제시한다.

- **Technical Challenges**: 사족 보행 환경에서는 풋 임팩트 쇼크와 고주파 진동이 시각 추적의 모션 블러/기하 왜곡을 유발해 프레임-프레임 특징 매칭과 추정기 결합이 쉽게 깨진다. 저자들은 GrandTour에서 동일한 로봇 플랫폼과 트랙 조건을 유지한 채 센서만 교체해 실패 원인이 센서-레벨 아티팩트인지, 알고리즘 설계인지 분리하고(셔터/스테레오/IMU 티어), 실패 런을 제외하는 강건한 평가 규칙과 ATE/RPE 기반 정량 지표로 비교한다.

- **Empirical Impact**: 실험 결과, 스테레오 구성은 모노큘러와 RGB-D보다 일관되게 우수했으며, global shutter가 rolling shutter보다 전 프레임 추적 실패를 크게 줄였다. 특히 ORB-SLAM3/RTAB-Map 같은 비전 중심 최적화 프레임워크에서는 IMU를 표준 방식으로 통합할 때 오히려 견고성이 악화될 수 있고, FAST-LIVO2는 전술급 Honeywell IMU에서 더 안정적으로 드리프트가 억제됐다. 결론적으로 사족 로봇 설계 가이드로는 비전 중심이면 global shutter 스테레오(관성은 필요 이상으로 결합하지 않는 방향), LiDAR 기반이면 tactical-grade IMU 채택이 유리하다는 메시지가 강화됐다.



### Congestion-Aware Robot Tour Planning in Crowded Environments (https://arxiv.org/abs/2606.19031)
Comments:
          Accepted to IEEE IROS 2026

- **Prior Approaches**: 기존 대다수 투어/경로 계획은 사람의 존재를 주로 충돌 회피 같은 로컬 내비게이션 비용으로 반영하거나, 특정 혼잡도를 휴리스틱(예: 관측 위치 주변에 손수 만든 분포)으로 임의 가중하는 방식에 머물렀다. 또한 인간 모션을 다루는 dynamics map(예: Fourier 기반 주기 모델)은 장기 예측에는 유리하지만, 시간-공간에서의 사람 수 변동과 실제 이동(모션)을 충분히 결합하기 어렵다.

- **Core Contribution**: 이 논문은 혼잡도가 로봇 이동 시간에 미치는 영향을 명시적으로 모델링하는 congestion-aware tour planner를 제안한다. 핵심은 (1) CLiFF(Circular Linear Flow Field, 원형 선형 흐름장) 맵으로 초기 관측 이후의 인간 궤적 분포를 예측하고, (2) 그 예측을 바탕으로 MDP/SSP(Stochastic Shortest Path) 기반의 온라인 재계획 프레임워크로 최적 투어 정책을 산출한다. 실행 중에는 새 사람이 들어오고 나가는 상황에 맞춰 receding horizon으로 계속 업데이트한다.

- **Technical Challenges**: 기술적 난제는 “사람 이동의 불확실성”과 “투어 계획의 시간적 확률성”을 동시에 다루는 동시에 계산량을 억제하는 데 있다. 이를 위해 각 인간의 다중 궤적을 샘플링해 edge(위상 맵 간선)별 혼잡 수준의 분포를 만들고, Poisson binomial로 ‘시간 t에서 해당 edge에 몇 명이 있을 확률’을 계산한 뒤 congestion bands로 상태 분기 수를 줄인다. 또한 edge 소요시간은 혼잡 band 조건부 분포의 기대값으로 근사해 MDP를 LRTDP(Labeled Real-Time Dynamic Programming)로 anytime 방식 탐색하며, time bound로 상태 공간을 유한화해 온라인 해결 가능성을 확보한다.

- **Empirical Impact**: 실험은 쇼핑몰 ATC 데이터셋 등 실제 crowd 데이터 기반으로 CLiFF를 학습한 뒤, 다양한 위상 맵 크기와 congestion bands 해상도(2/5/8밴드) 조건에서 성능과 계획 시간/재계획 효율을 비교하는 방식으로 검증한다. 결과적으로 로봇이 “최단 경로”를 고집할 때보다, 특정 POI가 특정 시간대에 붐빌 가능성을 피하는 우회/지연 전략이 투어 소요시간을 줄여 성능 향상으로 이어짐을 보여준다. 사람-로봇 상호작용이 빈번한 서비스 로봇 분야에서, high-level 투어 계획을 crowd 예측과 결합하는 실용적 설계 방향을 제시했다.



### TactSpace: Learning a Physics-enriched Shared Latent Space for Tactile Sim-to-Real Transfer (https://arxiv.org/abs/2606.18959)
Comments:
          9 pages, 6 figures, 4 tables, accepted into IROS 2026

- **Prior Approaches**: 기존 시뮬레이션-실세계(sim-to-real) 접근은 센서별로 높은 정확도의 원시 신호 시뮬레이션(예: 비전 기반 렌더링, 소프트바디 변형, 전기역학 모델)을 만들거나, 생성/도메인 적응으로 신호를 번역하는 방식이 많았습니다. 하지만 이런 파이프라인은 공학적 가정이 많고 센서 종류(예: capacitive 센서) 전반으로 확장하기 어렵다는 한계가 있습니다. 또한 rigid-body 기반 물리 시뮬레이터의 물리적 정확도 부족으로 인해 학습한 tactile 프록시가 실제 hardware와 크게 달라 전이가 잘 안 되는 문제가 지적됩니다.

- **Core Contribution**: 이 논문은 서로 다른 tactile 관측(modality)을 공통 잠재공간 TactSpace로 정렬해, 원시 신호를 정확히 일치시키지 않고도 sim-to-real 전이를 가능하게 하는 multi-modal representation learning 프레임워크를 제안합니다. 각 modality별 encoder(여기서는 ViT)를 공통 embedding으로 보내고, self- 및 cross-reconstruction과 InfoNCE contrastive alignment를 함께 학습해 modality-invariant하면서 정보-rich한 표현을 유도합니다. 특히 rigid-body(Isaac Lab)와 FEM(ABAQUS) 같은 서로 다른 물리 추상화 수준의 시뮬레이션 modality를 함께 넣어 더 풍부한 접촉 정보를 잠재공간에 담는 것이 핵심입니다.

- **Technical Challenges**: 핵심 technical challenge는 시뮬레이션과 실측 신호가 구조, 잡음/히스테리시스, 물리적 의미가 근본적으로 다르다는 점입니다. 논문은 이를 신호 레벨에서 맞추려 하지 않고, 동일한 물리 stimulus(인덴터 형상/위치/변위)로 대응되는 서로 다른 관측을 같은 stimulus로 간주해 contrastive로 끌어당기고, cross-modal reconstruction으로 보존 정보를 강제합니다. 또한 capacitive 관측에는 temporal history를 포함해 재료 히스테리시스로 인한 임베딩 정렬 붕괴를 완화하고, Warp 기반 Isaac Lab 플러그인으로 penalty-based tactile simulation을 GPU에서 대규모 병렬 생성하도록 구현합니다.

- **Empirical Impact**: 학습은 시뮬레이션만으로 수행한 뒤, real capacitance 측정에는 fine-tuning 없이 곧바로 평가하는 zero-shot 전이가 인덴터 shape 식별, force prediction, geometric reconstruction에서 일관되게 관찰됩니다. 물리적으로 표현이 다른 시나리오에서도 전이 성능이 유지되며, multi-physics modality를 함께 쓰면 force prediction 오차 16.7%, shape reconstruction 오차 45.8% 감소 같은 개선이 보고됩니다. 데이터 다양성과 규모(시뮬레이션 샘플 증가) 역시 기하학 작업에는 긍정적으로 작동하지만 힘(Force) 쪽은 단일 modality 양만 늘리는 것보다 응력 같은 보완 물리 정보를 넣는 것이 더 효과적임을 보여줍니다.



### Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancemen (https://arxiv.org/abs/2606.18953)
Comments:
          8 pages, 7 figures, 2 tables; 8-page appendix

- **Prior Approaches**: VLA는 imitation learning 기반이라 작은 실행 오차가 누적되면 예기치 않은 상태에서 취약해질 수 있다. Residual RL은 frozen VLA 위에 corrective policy를 얹는 방식으로 이를 완화하지만, sim-to-real 전이는 (1) privileged-state distillation이 필요하거나 (2) 이미지 기반은 visual domain gap 때문에 0-shot이 어렵고 (3) real-world RL은 비용·안전 문제가 커지는 딜레마가 있었다.

- **Core Contribution**: 이 논문은 object-centric residual RL로 sim-trained residual을 real robot에 distillation이나 real-world RL 없이 zero-shot 전이하는 프레임워크를 제안한다. 핵심은 residual이 6-DoF object pose(추정치), proprioception, base VLA action 같은 ‘현실에서도 복원 가능한’ 관측만 사용해 관측 공간 불일치를 크게 줄이는 것이다. 또한 real VLA와 action 분포 정렬을 위해 동일한 teleoperation 시연을 simulation에 replay해 sim 대응 VLA를 함께 학습한다.

- **Technical Challenges**: 가장 큰 과제는 residual 관측이 sim과 real 사이에서 얼마나 같은 정보를 제공하는지(도메인 갭)와, pose 추정 노이즈/실패를 어떻게 강건화하느냐였다. 저자는 잔차 관측을 privileged state나 이미지 대신 object pose로 구성하고, residual 학습 시 pose noise injection과 pose dropout(추정 실패 시 입력을 끄는 fallback)을 넣어 배포 시 추정 불확실성에 견디게 만들었다. 학습은 TD3로 off-policy 안정화를 하면서, 배포에서는 pose tracking confidence가 낮을 때 dropout fallback을 트리거한다.

- **Empirical Impact**: real Franka Research 3 (FR3)에서 5개 tabletop manipulation 작업(Cube Lift, Pick-and-Place, Stack, Drawer 닫기, 컵 세우기) 모두에서 0-shot 성능이 개선됐다. 성공률은 평균 42%에서 76%로 상승했으며, 추가 적응 없이 sim-trained residual만으로 일관된 개선이 관측됐다. 더 나아가 residual-corrected 롤아웃을 모아 base VLA를 재학습하는 self-improvement 루프도 가능해, 추가 teleoperation 없이 multi-task VLA 품질을 끌어올릴 수 있음을 실험적으로 보였다.



### A High-accuracy Event-based Underwater SLAM System (https://arxiv.org/abs/2606.18951)
- **Prior Approaches**: 이 논문은 event camera 기반 SLAM에서 널리 쓰이던 Time Surface(TS) 표현이 수중에서 쉽게 무너지는 문제를 짚는다. 기존 TS 기반 접근은 탁월한 고정 표현을 가정하거나, 속도·텍스처 변화에 대한 적응이 제한적이라 모션 블러/특징 희소화로 추적이 흔들린다. 또한 스테레오에서는 큰 disparity와 반복 텍스처가 데이터 연관을 망가뜨려 초기화 및 백엔드 안정성을 크게 해친다.

- **Core Contribution**: 저자들은 “정확도 중심”의 수중 event-based 스테레오 SLAM 시스템을 제안하며, TS 품질을 실시간으로 평가·최적화하는 프런트엔드와 견고한 스테레오 추적/삼각측량을 결합한다. TS에 대해 structure tensor coherence와 gradient 크기를 기반으로 구조적 정보 밀도를 정량화하는 metric을 설계해, 변하는 환경에서 최적 time-decay coefficient를 찾도록 한다. 더불어 스테레오 disparity prior(히스토리 median 기반)와 “latest-observation-first” 깊이 초기화로 데이터 연관 실패가 시스템 붕괴로 이어지는 경로를 차단한다.

- **Technical Challenges**: 핵심 난제는 (1) 수중에서 카메라 속도 변동이 TS 영상을 즉시 망가뜨리고 (2) 반복 텍스처+큰 스테레오 베이스라인이 LK 매칭을 실패로 몰며 (3) 기존 TS 튜닝을 매 프레임 수행하면 지연과 오동작이 생긴다는 점이다. 이를 위해 TS 최적화는 초기화 전에는 Bayesian Optimization(BO)로 전역 prior를 예측해 SfM 직전 고정하고, 초기화 이후에는 비동기 스레드에서 주기적 online local searching으로만 적응시키며 low-pass 계열의 완화로 진동을 억제한다. 스테레오 쪽은 median 기반 disparity prior로 탐색 구간을 좁혀 LK의 광측정 일관성 붕괴를 줄이고, 실패한 최초 매칭을 “latest-observation-first”로 재해석해 역방향 히스토리에서 최근 유효 관측을 찾아 SVD 기반 선형 삼각측량을 안정화한다.

- **Empirical Impact**: 실험은 공개 데이터셋과 새 수중 event 데이터셋 UWE(실제 로봇 주행, 다양한 조명/텍스처/속도, 정밀 ground truth)로 검증된다. 결과적으로 제안 시스템은 ESIO/ESVO2 계열 대비 ATE RMSE와 오차 분산에서 우수하며, 특히 난이도 높은 S09–S10에서 ESIO 대비 약 50% 개선을 보인다. 또한 수중 환경에서 ESVO2가 실패하는 구간에서도 제안 방법은 추적 실패 없이 안정적으로 동작했고, CPU-only 환경에서도 BO 예측(초기 cold start)과 비동기 온라인 검색으로 지연을 본 파이프라인에 거의 전가하지 않는 점이 실용성 근거를 제공한다.



### C-ARC: Continuous-Adaptive Range Clustering for Non-Repetitive LiDAR Sensors (https://arxiv.org/abs/2606.18948)
Comments:
          Submitted to IEEE Robotics and Automation Letters. This work has been submitted to the IEEE for possible publication. 8 pages, 7 figures

- **Prior Approaches**: 기존 실시간 LiDAR 클러스터링은 회전 기계식 센서의 반복적인 scan line 구조를 전제로 range image 투영과 PBID 인덱싱 같은 방식으로 연결성을 빠르게 계산해 왔다. 하지만 Risley prism 기반 non-repetitive 센서는 Rhodonea-curve로 인해 점 분포가 비균일하고, 회전 주기가 없어 프레임 경계가 명확하지 않아 이러한 가정이 깨진다. 그 결과 연속 슬라이딩 윈도우를 쓰더라도 PBID 스타일 인덱싱에 묶여 센서 일반화가 어렵다는 한계가 있었다.

- **Core Contribution**: 논문은 Risley prism 기반 non-repetitive LiDAR에서도 scan line 가정 없이 동작하는 연속형 클러스터링 프레임워크 C-ARC를 제안한다. 핵심은 슬라이딩 윈도우 동안 persistent dual-graph를 유지하되, 고주파 포인트 삽입과 클러스터 조회를 분리해 SLAM/트래킹에 필요한 낮은 지연을 확보하는 구조다. 또한 비반복 스캔 패턴에 대한 사전 지식 없이도 grid 해상도를 자동 보정하는 adaptive range grid resolution 메커니즘을 함께 제공한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 (1) scan 라인이 없어서 PBID/프레임 기반 연결성 추적이 불가능하고, (2) 비균일 샘플링으로 인해 fine grid는 조각화, coarse grid는 충돌·과연결(merge)을 유발하는 Sparsity-Collision trade-off가 발생한다는 점이다. C-ARC는 포인트를 bucket-based range grid에 누적해 격자 해상도와 정보 충돌을 완화하고, 초기화 단계에서 지표(빈 구간 길이, 점 밀도, mean multiplicity)를 보며 exponential control loop로 h/v 해상도를 조절한다. 더불어 ring buffer에서 오래된 포인트를 제거할 때 즉시 분할 검증을 하지 않고 lazy deletion으로 배치 처리해 고정된 per-point 비용을 유지한다.

- **Empirical Impact**: C-ARC는 Livox Mid-360에서 20 Hz 실시간 클러스터 출력을 commodity hardware에서 단일 스레드 C++17 라이브러리로 제공하며, 실시간성은 P99 지연이 50 ms 기준을 넘지 않는 soft real-time 목표로 평가된다. Livox Avia에서는 센서의 중심 쏠림 Rhodonea 패턴 때문에 per-cell point occupancy가 커져 지연이 장면 의존적으로 증가했는데, 이는 스캔 기하와 per-cell 누적 밀도가 성능을 좌우한다는 점을 실증적으로 확인해 준다. 마지막으로 adaptive grid resolution는 기존 grid 기반 방법에도 적용 가능함을 보여 주며, 비반복 데이터에서 클러스터 품질을 개선할 수 있음을 통해 범용성(그리고 한계로서 per-cell 과점유 문제의 중요성)을 함께 드러냈다.



### ZiMPedance: Impedance-Aware ZMP Modeling and Control for Payload Carrying with Quadruped Robots (https://arxiv.org/abs/2606.18883)
- **Prior Approaches**: 사족 보행 로봇의 중량물 운반은 ZMP 같은 동적 안정성 기준과, 접촉-상호작용 힘이 결합된 문제로 다뤄져 왔다. 기존 연구들은 케이블/볼조인트/액티브 암(impedance control, whole-body control) 중심이라, 수동 spring-damper 인터페이스의 stiffness·damping·질량이 ZMP 안정성 여유에 미치는 직접 관계를 명시적으로 유도하진 못했다.
또한 learning 기반 협동 운반은 학습 목적이 상호작용을 형성하지만, “수동 인터페이스 파라미터→안정성 여유”를 해석 가능한 모델로 연결하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 수동 spring-damper payload-interface 동역학을 포함한 extended ZMP 공식을 유도해, stiffness·damping·payload mass가 stability margin에 어떻게 영향을 주는지 직접 연결한다. 특히 underdamped 구성에서 보행이 만드는 가진 주파수 성분과 인터페이스 공진이 맞물려 ZMP 진동이 증폭될 수 있음을 보인다.
이 통찰을 바탕으로 passive subsystem dynamics를 반영하도록 SRBD를 확장하고, 이를 MPC(Model Predictive Control)에 통합해 arm-aware 제어(ARMPC)로 안정성을 보정한다.

- **Technical Challenges**: 주요 과제는 수동 암의 내부 spring-damper가 만들어내는 진동성 상호작용 힘을 ZMP 계산식과 예측 제어에 동시에 넣는 것이었다. 논문은 관절 수준 passive dynamics를 등가 질량-스프링-댐퍼로 환원해 ZMP 변위에 대한 전달함수 형태로 모델링하고, 로봇 base가 보행 조화(harmonics)로 전달하는 여기원과의 주파수 정렬을 분석한다.
또한 MPC 구현에서는 증폭되는 진동을 단순 외란 보상처럼 숨기지 않고, 확장된 결합 동역학을 예측에 포함해 ground reaction force를 최적화하도록 설계했다.

- **Empirical Impact**: 시뮬레이션에서 제안 제어는 stability violation을 최대 10배 줄여(7.0%→0.7%) ZMP 여유 위반을 크게 완화하고, 수평 방향 지면반력 노력도 최대 15% 낮춰 보행 효율을 개선했다. 2kg payload 하드웨어 실험에서는 pull-release 교란 상황에서 기준(nominal) 제어가 실패하는 동안에도 ARMPC가 안정적인 보행을 유지했다.
더 나아가 동일 모델로 수동 팔의 동역학만 활용해 end-effector tracking까지 가능함을 보여, 수동 인터페이스를 “안정성에 불리한 요소”가 아니라 “모델링 가능한 제어 자산”으로 재정의했다.



### Space Is Intelligence: Neural Semigroup Superposition for Riemannian Metric Generation (https://arxiv.org/abs/2606.18828)
- **Prior Approaches**: 대부분의 motion planning·제어는 에이전트(학습된 정책, A*·RRT 같은 검색, potential field, MPC, 강화학습)가 ‘지능’을 갖고, 장면 정보는 그 입력으로만 들어간다. 그래서 장애물 회피에는 충돌 체크·비용 설계 등 별도 절차가 필수이거나, 비용/제약을 과업마다 공학적으로 다뤄야 한다. 최근의 learned planner·diffusion·VLA 계열도 결국 scene→action 매핑 구조라 지능이 여전히 에이전트 쪽에 남는다.

- **Core Contribution**: 이 논문은 지능의 위치를 ‘공간 자체’로 옮긴다. 장면이 configuration manifold에 Riemannian metric tensor를 유도하고, 경로 계획은 그 metric의 geodesic를 푸는 문제로 축소되어 별도의 planner나 collision checker 없이도 장애물을 회피하도록 만든다. Encoder-Router가 scene-conditioned metric field를 만들고, geodesic solver가 이를 ‘수동 판독’해 행동(경로)을 얻는 구조가 핵심이다.

- **Technical Challenges**: metric을 만들되 항상 유효한 Riemannian metric(SPD)을 보장해야 하고, 장면 복잡성이 늘어도 파라미터·구조가 폭증하지 않게 해야 한다. 이를 위해 Lie algebra의 합과 exponential map exp: sym(2)→SPD(2)을 사용해 네트워크 출력이 곧바로 SPD metric으로 변환되도록 설계했으며, semigroup-superposition 규칙으로 장면 요소 수(K)가 늘어도 조합 법칙을 그대로 유지한다. 또한 단일 Encoder-Router에서 frame parameters(시점 정렬), modulation parameters(커널 전파·장애물 근방 장벽의 날카로움), basic coefficients(세기·부호·gate)를 함께 생성해 하나의 metric field로 합성한다.

- **Empirical Impact**: 2D 시뮬레이션에서 단 하나의 2-장애물 장면만 학습한 뒤, 장애물 개수·배치·밀도·패턴이 다른 12개 테스트 장면에 대해 zero-shot으로 성능을 보였다. 충돌-free 경로 비용과 obstacle-penetrating 경로 비용이 장면에 따라 3~5자릿수(orders-of-magnitude) 이상 분리되며, threshold 튜닝 없이도 metric만으로 충돌 가능성을 명확히 구분할 수 있다고 보고한다. 특히 hard top-k sparse gating을 쓰면 성능이 급락했는데, 이는 gate를 연속적으로 경쟁시키는 설계가 중요한 학습 신호임을 보여준다.



### HALOMI: Learning Humanoid Loco-Manipulation with Active Perception from Human Demonstrations (https://arxiv.org/abs/2606.18772)
- **Prior Approaches**: 로봇 프리(human-free) 시연은 UMI-style 인터페이스와 egocentric sensing을 결합해 규모 확장이 가능했지만, humanoid loco-manipulation에서는 하반신 기준점이 부족해 실행이 불안정해지기 쉬웠습니다. 또한 기존 work들은 주로 observation/action 정렬이나 retargeting에 머물러 active ego-view(시선 변화) 전이를 장기 작업까지 견고하게 다루지 못했습니다. 그 결과, humanoid에 월드 프레임 헤드-핸드 궤적을 그대로 넘기면 OOD 타깃에서 brittle해지고 인간-로봇 형태 격차도 누적 오류로 이어졌습니다.

- **Core Contribution**: HALOMI는 인간 시연에서 능동 시선(헤드)과 손-물체 상호작용을 함께 수집하고, 이를 humanoid에 end-to-end가 아니라 “헤드-핸드 타깃 → 전신 컨트롤러 실행” 구조로 안정 전환하는 프레임워크를 제시합니다. UMI를 egocentric 관측에 확장해 ego-view와 wrist-view 관측 및 헤드-핸드 trajectories를 대규모로 모으고, manifold-constrained whole-body controller가 월드 프레임 정밀 추적을 수행합니다. 여기에 ego-view alignment와 controller-aware reference trajectory adaptation으로 인간-로봇 gaps(관측 및 실행 불일치)를 줄여 전이를 강화합니다.

- **Technical Challenges**: 핵심 난제는 (1) 인간과 humanoid의 형태/카메라 차이로 인한 ego-view 관측 갭, (2) sparse한 헤드-핸드 키포인트 타깃만으로 전신이 안정적으로 다중해(multi-modal) 해석을 해야 하는 문제, (3) 시연된 월드 프레임 궤적을 컨트롤러가 그대로 재생할 때 생기는 추적 오차 누적이었습니다. HALOMI는 BFM-Zero 기반의 learned latent behavior manifold에서 RL 추적을 수행해 비자연스럽고 공격적인 joint-space 해를 억제하고, 능동 목(active neck)과 분리형 neck–body 설계로 헤드 회전이 전신 안정성에 미치는 영향을 완화합니다. 또한 오프라인 파이프라인에서 ego-view 정렬(3D 재투영 및 inpainting)과 B-spline 잔차를 CEM으로 최적화한 controller-aware reference trajectory adaptation을 적용해 닫힌고리(closed-loop) 롤아웃의 불일치를 줄였습니다.

- **Empirical Impact**: Unitree G1(액추에이티드 neck)에서 navigation, grasping, bimanual manipulation, whole-body coordination, 동적 동작까지 5개 실세계 작업을 검증했으며, 정량 평가 3개 작업에서 평균 success rate 85%를 달성했습니다. 컨트롤러 단독 평가에서도 학습에 쓰지 않은 인간 헤드-핸드 궤적에 대해 정밀·강건한 tracking을 보여주며, 갑작스런 큰 타깃 변화나 비실현 명령에서도 비공격적 안정 동작을 유지했습니다. 나아가 정성 데모로 dynamic tossing과 deep-squat grasping 같은 고난도 동작까지 시연되어, active perception 기반 인간 시연이 humanoid loco-manipulation의 실용적인 스케일업 데이터 소스로 기능할 수 있음을 시사합니다.



### Generating Natural and Expressive Robot Gestures through Iterative Reinforcement Learning with Human Feedback using LLMs (https://arxiv.org/abs/2606.18747)
Comments:
          8 Pages, 6 Figures

- **Prior Approaches**: 기존 소셜 로봇 제스처 생성은 전문가가 만든 애니메이션에 의존하거나, 기능적으로는 측정 가능하지만 자연스러움 같은 주관적 지표를 잘 반영하지 못했다. 또한 GenAI로 로봇 코드를 뽑더라도 자유도가 커질수록 “자연스러움”을 놓치며, 제스처가 비결정적이라 객관적 평가가 어렵다.

- **Core Contribution**: 이 논문은 대화에 맞춘 co-speech gesture를 Pepper에서 생성하기 위해 GPT-4.1을 결합하되, 단순 LLM 생성만으로는 뻣뻣함이 남는 문제를 RLHF로 해결한다. 인간 사용자의 영상 평가를 반복적으로 수집해 제스처 생성 코드(저수준 primitive 함수 시퀀스)를 파인튜닝함으로써, 발화와 동기화된 더 표현력 있고 유연한 제스처를 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) LLM이 생성한 모션 코드가 Pepper 관절 한계와 실행 가능성을 만족해야 하고, (2) 자연스러움·부드러움처럼 주관적 선호를 학습 신호로 바꾸는 것이며, (3) 제스처의 비결정성과 다양성 유지가 동시에 필요하다는 점이다. 이를 위해 관절 제약을 반영한 low-level motion primitives 라이브러리와 타이밍 규칙을 프롬프트에 강제하고, 사용자 선호는 offline DPO와 online few-shot 대화이력 요약을 이용해 반복적으로 모델에 주입한다.

- **Empirical Impact**: Pepper 영상을 대상으로 한 5회 온라인 사용자 스터디에서, 마지막 iteration은 첫 iteration 대비 expressiveness·relevance·fluidity 전 항목이 통계적으로 유의하게 개선되었다(p<0.001). 특히 관련성·유연성의 상승 폭이 더 컸고, 예: apology 같은 의미가 명확한 범주에서 개선이 두드러지며, few-shot 프롬프트가 전체 성능 향상에 크게 기여했다는 ablation 결과도 제시된다.



### Two-Phase Bilevel Search for the Moving-Target Traveling Salesman Problem with Moving Obstacles (https://arxiv.org/abs/2606.18730)
- **Prior Approaches**: 기존 MT-TSP 연구는 목표가 움직이는 상황에서 시간 창(time window) 제약을 만족하며 최적 경로를 찾는 데 집중했지만, 주로 선형/조각선형 궤적이나 최적해에 대한 강한 보장 형태로 제한되는 경우가 많았습니다. 장애물까지 포함한 MT-TSP-O는 정적 장애물에서는 완전성/부분 보장 알고리즘이 제시됐으나, MT-TSP-MO처럼 ‘장애물도 움직이는’ 일반화에 대한 연구는 매우 드뭅니다. 또한 알려진 대표 접근이 소수에 그치며, 본 문제에서는 고품질 feasible 해를 빠르게 얻는 전략이 핵심 난제로 남아 있었습니다.

- **Core Contribution**: 이 논문은 Moving-Target Traveling Salesman Problem with Moving Obstacles(MT-TSP-MO)를 위해 Mixed-Integer Conic Programming(MICP) 정식을 제안합니다. 이 정식은 off-the-shelf 솔버로 풀 수 있고, 시간 이산화 기준에서 optimality/completeness에 대한 점근적 보장을 제공합니다. 동시에 큰 규모에서는 MICP의 확장성이 떨어진다는 점을 보완하기 위해 Two-Phase Bilevel Search(TPBS) 알고리즘을 개발하여, 최적성/완전성 보장은 낮더라도 더 신뢰도 높은 feasible 해와 낮은 비용을 빠르게 산출하도록 했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 시간 창 제약과 장애물 회피를 동시에 만족해야 하는 연속시간 충돌 판정, 그리고 속도 제한까지 결합되는 것입니다. MICP에서는 시간 축을 이산화하고, 목표/정차 방문을 이진 변수로 모델링한 뒤, 장애물의 이동에 따른 반공간 기반 조건과 big-M 제약으로 충돌 회피를 포함합니다. TPBS에서는 먼저 장애물이 없는 조건에서 GTSP(Generalized Traveling Salesman Problem)를 풀어 후보 투어를 만든 뒤, 각 간선에 대해 직선 기반 빠른 충돌 체크로 불충분할 때만 time-expanded graph에서 DFS로 collision-free, time-feasible 경로를 검증하는 2단계 구조로 계산량을 관리합니다.

- **Empirical Impact**: 실험은 최대 40개의 targets와 40개의 moving obstacles까지 포함하는 폭넓은 인스턴스(시간 창 길이 다양)를 대상으로, 기존 baseline 알고리즘과 비교해 success rate, solution cost, 계산 시간을 종합적으로 개선했음을 보여줍니다. 특히 MICP는 보장에 가까운 성질을 유지하면서도 실용적으로 작동하고, TPBS는 큰 문제에서 더 빠르게 고품질 feasible 해를 제공하는 쪽에 강점을 드러냅니다. 결과적으로 MT-TSP-MO 분야에서 ‘움직이는 장애물까지 고려한 실행 가능한 경로 계획’을 확장하는 실증적 기준을 제시한 것으로 평가됩니다.



### Selective Unit-Cell Actuation in Lattice Structures for Distributed Morphology in Soft Robots (https://arxiv.org/abs/2606.18704)
Comments:
          Accepted to IROS 2026, 8 pages, 5 figures

- **Prior Approaches**: 소프트 로봇에서 기하-구조 설계는 압력에 의해 발생하는 변형 모드를 제어하는 핵심이지만, 기존에는 대체로 장치나 모듈 단위로 액추에이터를 넣고, 나머지 격자(리스트/메타머티리얼)는 수동 지지층 역할에 머무는 경우가 많았습니다. 그 결과 변형 모드는 균일 가압 같은 ‘입력 조건’과 액추에이터가 강제하는 운동학에 의해 주로 결정되고, 격자 유닛셀 자체가 분산 구동장(field)으로 직접 참여하기는 어려웠습니다.

- **Core Contribution**: 이 논문은 actuator-lattice co-design을 유닛셀 스케일로 끌어내린 embedded pneumatic unit cell(EPUC)를 제안합니다. EPUC는 curved-strut 격자 지오메트리와 양방향 bellow bellow actuator를 단일 모놀리식 요소에 함께 내장해, 테셀레이션 시 전역 형태가 ‘압력의 크기’가 아니라 ‘셀별 공간적 가압 패턴’에 의해 결정되도록 만듭니다. 또한 선택적 유닛셀 구동만으로 벤딩과 방향성 그립(잡기)을 물리 하드웨어 변경 없이 구현합니다.

- **Technical Challenges**: 핵심 난제는 유닛셀 내부에서 액추에이터-구조 결합을 연속 재료로 구현하면서도, 셀 간 테셀레이션 후에는 원하는 방향성(이방성)과 축/횡 변형 결합이 유지되게 만드는 것입니다. 연구진은 parametric 워크플로(Grasshopper)로 곡률 스트럿과 벨로우 형상을 동시 설계하고, bellow가 있는 영역에서 격자 곡선을 국소 트림해 테셀레이션 시에도 내부 공기 경로가 끊기지 않게 구성했으며, bellow가 양(양압)·음압(진공) 모두에서 양방향 선형 변형을 내도록 막 두께와 치수를 최적화했습니다.

- **Empirical Impact**: 실험에서는 1×1×1, 2×2×2, 3×3×3 테셀레이션에서 변위와 힘이 스케일링되며 500사이클 구동에서도 반복 가능한 성능을 보였습니다. 3×3×3에서는 열(컬럼) 단위로 +30kPa/-70kPa를 선택 구동해 전역 벤딩 모드와 그립 모드를 분기할 수 있고, 특정 대각 구동에서는 스트럿의 방향성 인터로킹 때문에 그립 실패까지 관찰되어 ‘액추에이션 토폴로지+기계적 제약’의 결합 효과가 확인됩니다. 나아가 능동(유닛셀)과 수동(Edge Octa) 셀을 혼합한 하이브리드 구조로 크롤링을 시연해, 대칭이 깨진 형태 변화가 실제 보행으로 번역됨을 보여주며 유닛셀 분산 모핑의 확장 가능성을 뒷받침합니다.



### Leveraging Energy Features for Surface Classification with Deep Learning: A Comparative Analysis Across Three Independent Datasets (https://arxiv.org/abs/2606.18698)
- **Prior Approaches**: 모바일 로보틱스의 표면(surface) 분류에서는 일반적으로 관성(inertial) 센서 기반 접근이 중심이었고, 에너지 기반(energy-based) 특징은 제한된 환경에서만 일부 성과가 보고돼 상대적으로 덜 연구된 편입니다. 기존 연구들은 에너지 기반 특징의 단독 활용 가능성이나 관성 데이터와의 보완 효과를 체계적으로 검증하지 못한 경우가 많았습니다.

- **Core Contribution**: 이 논문은 에너지로부터 도출한 특징이 표면 분류에서 단독 입력으로도 작동하는지, 혹은 inertial 데이터에 보조 입력으로 넣을 때 성능이 얼마나 오르는지를 실증적으로 평가합니다. 또한 다양한 최신 딥러닝 분류기(RNN, CNN, encoder-only transformer, Mamba state-space model)를 대상으로 자동 하이퍼파라미터 튜닝과 입력 시퀀스 길이 최적화를 함께 수행해 비교의 공정성을 높였습니다.

- **Technical Challenges**: 핵심 기술 난제는 에너지 기반 특징이 얼마나 정보성이 있는지(단독 정확도)와, 관성 신호와 결합했을 때 유효한 보완 관계가 성립하는지(증분 이득)를 찾는 데 있습니다. 저자들은 여러 시퀀스 길이와 하이퍼파라미터를 자동으로 탐색해 과적합 및 입력 설계 편향을 줄이면서, 에너지 단독 vs 관성 단독 vs 에너지+관성의 성능 차이를 안정적으로 추정했습니다.

- **Empirical Impact**: 3개의 공개 데이터셋에서 이전 보고 대비 더 높은 정확도를 달성했으며, 전체 성능은 CNN이 최우수였습니다. 에너지 특징만 사용할 경우 85-90% 정확도를 보였고 이는 관성 결합 시 96-99%에 비해 약 5-10% 낮지만, 관성에 에너지 특징을 추가하면 평균 1-2%의 일관된 정확도 향상이 관측되어 실사용 관점의 선택지를 제공합니다.



### High-Degree-of-Freedom Lightweight Bioinspired Leg for Enhanced Mobility in Small Robots (https://arxiv.org/abs/2606.18680)
- **Prior Approaches**: 마이크로 로보틱스에서는 작은 크기에서 생기는 구동기 제약 때문에 DoF를 줄이거나, 역으로 직렬(serial) 구조로 DoF를 늘리려는 두 방향이 주로 쓰였다. 전자는 제어는 쉬우나 전방위 이동/자세 조절이 약하고, 후자는 관절에 구동기를 붙여 원위부(발끝) 왕복 질량과 회전 관성이 커져 동적 응답성이 떨어진다. 곤충 스케일에서 piezoelectric 액추에이터나 soft actuator가 가능함이 입증됐지만, 안정적 상호작용과 하중을 요구하면 더 견고한 servo 기반 구조의 새 패러다임이 필요했다.

- **Core Contribution**: 이 논문은 공간형 4-DoF 평행(parallel) 레그를 제안하며, 관절별 구동기 배치로 생기는 원위부 관성 문제를 줄이는 것을 핵심으로 한다. 두 개의 spherical five-bar linkage를 통해 공간의 자유도(피칭/외전/내회전/무릎 굴곡)를 합성하고, 대칭적인 동작 구조를 유지하면서 모든 액추에이터를 본체(thorax)에 집적한다. 또한 concentric(중첩 구경) 설계로 레그의 기구학 해석을 단순화해 실시간 제어 적용성을 높였다.

- **Technical Challenges**: 공간형으로 DoF를 늘리면 기구학 모델이 복잡해지고, 동시에 SFB(SFB linkage) 계열에서 특이점 근접 시 축 방향 구동 권한이 약해질 수 있다. 연구진은 SFB의 비직교 조인트를 가상 직교 좌표계로 치환해 포워드/역기구학을 정리했으며, 각 입력 축을 동축(coaxial)으로 배치해 pitch와 다른 DoF의 결합을 줄였다. 아울러 nested spherical 구조로 concentric 구성을 달성해 추가 보상 관절 증가를 피했고, 대신 서보 스트로크 한계로 출력 자유도를 일부 제한받는 상황도 명시했다.

- **Empirical Impact**: 단일 레그 실험에서 θ3를 고정할 때 작업공간이 약 7681 mm³에서, θ3까지 허용하면 22255 mm³ 이상으로 확장되는 결과를 보였다. 발끝은 지정된 궤적을 따라 움직이며, 스트레인 게이지 기반 힘 측정에서 피크 출력이 약 0.5 N 수준으로 확인됐다. 총 질량 18.9 g, 큰 작업공간, 낮은 원위부 관성의 조합이 마이크로 생체모사 이동의 민첩성과 적응성을 높일 수 있음을 실증하며, 향후 origami-inspired 구조와 마이크로 액추에이터 및 다중 레그 확장 방향을 제시한다.



### A Scalable Embodied Intelligence Platform for Seamless Real-to-Sim-to-Real Transfer of Household Mobile Manipulation Tasks (https://arxiv.org/abs/2606.18646)
Comments:
          CCF Transactions on Pervasive Computing and Interaction

- **Prior Approaches**: 기존 연구는 모바일 조작을 시뮬레이터(Gazebo/MuJoCo/Sapien 등)에서 빠르게 개발하더라도, 실세계의 변형(잡음, 마찰/충돌, 물체 가림 등)을 충분히 반영하는 자동 scene reconstruction이 부족해 robustness 확보가 어려웠습니다. 또 시뮬레이션에서 전략을 평가·비교하려면 환경/태스크를 수작업으로 분해·설계해야 해서 long-horizon 과제는 검증 비용이 커졌고, 실배포는 ROS/하드웨어별 인터페이스 차이로 호환성이 낮았습니다. 결과적으로 real-to-sim-to-real 전 과정이 비싸고 번거로워 대규모 실험과 재현성 있는 벤치마크 구축이 제한됐습니다.

- **Core Contribution**: 이 논문은 BestMan이라는 스케일러블 real-to-sim-to-real 플랫폼을 제안해, 실세계 관측에서 시뮬레이션까지의 간극을 한 번에 잇는 파이프라인을 제공합니다. 핵심은 ASG(Automated Scene Generation)로 관측 기반의 사실적·관절형 scene을 자동 복원하고, 시뮬레이션에서는 simulation-guided task formalization/skill learning으로 전략을 모듈화·대규모 평가하게 하며, HUM(Hardware-agnostic and Unified Middleware)으로 서로 다른 로봇 하드웨어에도 호환되게 배포하는 것입니다. 이를 통해 household mobile manipulation에서 “개발-평가-실배포”를 표준화된 방식으로 연결합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 단일/다중 관측에서 semantic·기하·관절 정보를 누락 없이 복구하는 scene reconstruction, (2) long-horizon 언어지시를 일관된 하위 태스크·스킬로 분해하고 시뮬레이션에서 체계적으로 검증하는 task formalization, (3) 시뮬레이터·ROS·제어 API 차이를 흡수해 heterogeneous 로봇으로 안전하게 옮기는 sim-to-real 전이였습니다. BestMan은 GPT-4 라벨링과 Grounded-SAM·Depth-Anything을 조합해 마스크/깊이를 뽑고, CLIP·DINOv2 기반 에셋 검색과 URDFormer/AutoURDF로 관절 구조까지 URDF로 합성하는 ASG를 설계했습니다. 또한 THMM 벤치마크에 Sense–Plan–Act 구조의 모듈 스킬 아키텍처를 얹고, HUM은 모바일 매니퓰레이터 API를 추상화해 시뮬과 실의 고수준 명령을 매핑(하드웨어·드라이버 의존 최소화)하는 방식으로 호환성을 확보합니다.

- **Empirical Impact**: 실험은 ASG의 fidelity(범주/모델링/관절 정확도, L2·방향·Bbox IoU·Pose Error 등)와 인간 평가를 통해 검증하며, 어려운 실세계 입력에서도 인간이 납득할 수준의 semantic 및 관절 품질을 보였다고 보고합니다. 또한 모듈형 설계는 시뮬레이션의 난이도 상승(Hardest setting 포함)에서도 강화학습·행동복제 대비 더 높은 성공률을 보여 long-horizon 과제에 유리함을 확인했습니다. 마지막으로 HUM은 sim-to-real 배포 시간을 ROS 대비 약 3~5배 단축하고 응답 지연도 줄였으며, 서로 다른 하드웨어에서도 비슷하거나 약간 개선된 성공률을 달성해 표준 벤치마크 연구 기반을 제시합니다.



### EffiNav: Fusing Depth and Vision-Language for Efficient Object Goal Navigation (https://arxiv.org/abs/2606.18634)
- **Prior Approaches**: ObjNav 선행연구는 크게 (1) 대규모 학습을 통해 다음 탐색 결정을 내리거나, (2) 학습 없는 VLM/프런티어 기반으로 다음 탐색 지점을 고르는 방식으로 나뉜다. 하지만 학습 기반은 데이터·계산 비용이 크고 범용성에서 흔들리며, 학습 없는 방식은 visited area를 다시 훑거나 프런티어 간 왕복이 늘어 효율이 떨어질 수 있다. 특히 깊이 기반 또는 단순 전역 지식 부재는 지도·시점 오류가 쌓이며 비효율을 키우는 취약점으로 지목된다.

- **Core Contribution**: 이 논문은 학습 없이도 효율적 탐색을 목표로 하는 VLM 기반 프레임워크 EffiNav를 제안한다. 핵심 아이디어는 (1) 깊이로 후보 탐색 영역을 만들고, (2) egocentric 선택과 top-down 관점의 global-wise 검증을 함께 수행하며, (3) history-aware pruning과 프런티어 백업으로 재탐색·막힘을 줄이는 것이다. 또한 memory-augmented ObjNav로의 확장도 함께 시연해, 표준 ObjNav를 넘어선 적용성을 보인다.

- **Technical Challenges**: EffiNav가 해결해야 할 기술적 과제는 “다음에 어디를 볼지”를 효율적으로 고르는 것과, 그 선택이 실제 공간에서 말이 되는지 전역 일관성을 보장하는 것이다. 저자들은 depth-aware 후보 마스킹 후 VQA 형태로 egocentric에서 1~2순위 후보를 고르고, 이를 점진 구축된 top-down 지도에 투영해 동일 VLM으로 전역 검증을 수행한다. 선택이 계속 실패하면 nearest frontier 픽셀로 백업하며, 경로는 A*로 collision-free를 계산하되 장애물 팽창·목표/자세 미세 보정까지 더해 견고성을 확보한다.

- **Empirical Impact**: EffiNav는 시뮬레이션 Habitat HM3D와 OVON에서 Success Rate(SR)와 Success weighted by Path Length(SPL) 기준으로 기존 베이스라인을 상회하거나 근접 성능을 보이며, success 조건의 효율을 더 잘 드러내는 Normalized Efficiency on Successes(EoS)에서도 강점을 보인다. 2,019~3,000 에피소드 규모의 대규모 실패 분석에서는 실패 원인이 데이터셋 성격(탐색 난이도 vs 객체 의미 정합)별로 달라짐을 보이고, EffiNav는 그 차이 속에서도 효율 중심 지표를 비교적 안정적으로 유지한다. 더 나아가 실제 로봇(unitree go2 + Azure Kinect) 실험에서도 SR은 비슷하더라도 SPL과 EoS가 개선되어, “정확히 찾되 더 적게 헤매는” 실용적 효과를 실증한다.



### ROBOSHACKLES: A Safety Dataset for Human-Injury Prevention in Embodied Foundation Models (https://arxiv.org/abs/2606.18632)
- **Prior Approaches**: 기존 Embodied Foundation Models(EFMs)은 VLA, World Action Models(WAMs) 흐름에서 멀티모달 이해와 end-to-end 제어를 강화해 왔지만, 실행 가능한 형태의 출력이 가진 안전 실패가 큰 문제로 남아 있다. 안전 연구는 주로 adversarial robustness, backdoor attack, constrained policy optimization처럼 ‘공격/강건성’ 또는 행동 제약에 초점을 두고, 사람의 부상으로 이어질 수 있는 물리적 결과(특히 실행 이후의 동역학)를 세밀히 정렬하는 데이터와 평가가 부족했다. 또한 기존 벤치마크는 지시 수행/품질 평가 중심이어서 direct harm(즉시 위해)과 indirect harm(환경 변화로 인한 위험)을 충분히 분리·커버하지 못했다.

- **Core Contribution**: 이 논문은 인간 부상 예방을 목표로, 실세계 로봇 관측에서 안전-치명적 시나리오를 합성·구성하는 안전 데이터 생성 파이프라인을 제안한다. 핵심은 scene understanding→hazard-aware image editing→temporal prompt 생성→single-pass Wan2.7 rollout 합성으로, 편집된 위험 상태에서 미래 전개를 바로 생성해 통제성과 현실성을 동시에 노린 점이다. 이를 통해 direct-harm 2종과 indirect-harm 4종을 포함한 10,000클립 로보틱 비디오 데이터셋 ROBOSHACKLES를 구축하고, refusal-based 안전 정렬 학습/평가 자원으로 활용 가능함을 제시한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 윤리적으로 수집이 불가능한 ‘사람을 다치게 하는’ 실세계 데이터를 대체할 수 있으면서도, 위험 표지가 실행 전후의 물리적 인과에 맞게 유지되어야 한다는 점이다. 저자들은 DROID의 실제 로봇 기하·레이아웃을 초기 조건으로 삼고, Qwen3-VL과 Qwen-Image로 카테고리별 위험을 삽입한 뒤 temporal prompts를 통해 장면 진화를 명시해 단일 패스 생성에서 발생하는 object drift, 변형, 환각을 줄이도록 설계했다. 또한 생성 결과는 task completion과 visual quality 성격의 자동 지표(physical-semantic plausibility, task-adherence consistency 등)와 함께 사람 검증으로 라벨 일치성을 확보해 학습 신뢰도를 높였다.

- **Empirical Impact**: ROBO_SHACKLES는 1,200개 테스트(카테고리당 200개)에서 여러 EFMs(Cosmos-Policy, DreamZero, LingBot-VA, FastWAM, VLA-JEPA, World Guidance)를 strict refusal-based 기준으로 평가했는데, 6개 모델 모두 direct/indirect harm 전 범주에서 행동을 생성해 100% unsafe action generation rate를 기록했다. 즉, 현재 EFMs는 명시적 위해 지시뿐 아니라 ‘겉보기에는 무해하지만 결과가 위험해지는’ 경우를 제대로 거절하거나 사전 위험을 예측하지 못함이 실험적으로 드러났다. 이 결과는 embodied safety 정렬이 데이터·평가 측면에서 새로운 안전 택소노미와 합성 벤치마크 없이는 진전이 어렵다는 신호로, ROBOSHACKLES가 refusal learning 및 hazard anticipation의 확장 가능한 학습 기반이 될 수 있음을 강조한다.



### DNN Koopman-Based Deviation Compensation for UGV Path Tracking Control on Coupled Slope and Potholed Road (https://arxiv.org/abs/2606.18630)
Comments:
          22 pages, 13 figures

- **Prior Approaches**: 기존 UGV 경로 추종 연구는 MPC처럼 제약을 다루는 모델 기반 접근이 많았지만, 매 회 QP를 풀어야 해 계산량이 커 실시간성이 떨어질 수 있습니다. 또한 오프로드의 긴 파장 요철(경사)은 저항항 등으로 단순화해 다루는 경우가 많고, 경사+선회 결합(coupled slope)에서 발생하는 타이어 측방력 급변을 충분히 반영하기 어렵다는 한계가 있습니다. 퍼호우(포트홀) 같은 짧은 파장 요철에 대해서는 선형화/퍼지/flatness 등으로 근사하지만, 코너링 강성의 불연속 때문에 미모델링 오차가 커지기 쉽고, RL/LSTM 같은 모델 프리 방식은 해석 가능성이 낮아 성능 저하 시 내부 메커니즘 기반의 조정이 어렵습니다.

- **Core Contribution**: 이 논문은 경사 결합 상황에서는 LMPC(Laguerre model predictive control)를, 포트홀 상황에서는 DNN Koopman(DK) 기반 편차 보상기를 결합하는 “모델 기반+모델 기반(해석 가능한 Koopman)+신뢰도 검증” 구조를 제안합니다. 특히 타이어 cornering stiffness의 시간변화는 AFRLS(자기 적응 망각 재귀 최소제곱)로 추정하고, LMPC의 QP 부담은 Laguerre 함수 피팅/차분으로 줄여 baseline 조향각을 생성합니다. 이후 Koopman 이론에 DNN(오토인코더)을 얹어 포트홀로 인한 비선형을 유한차원으로 근사하고, 그 보상 입력을 LMPC와 함께 이벤트 트리거 방식으로 실행 가능성과 안정성을 함께 보장합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 경사 결합 조건에서 타이어 코너링 강성의 불확실성이 커져 모델 기반 추종이 흔들리는 문제, (2) 포트홀로 인한 큰 비선형/불연속을 Koopman 근사로 충분히 표현하는 문제, (3) 보상 입력을 단순 합성하면 액추에이터 한계나 안정성 제약을 위반할 수 있다는 “입력 초과/피드백 충돌” 문제입니다. 논문은 AFRLS로 cornering stiffness 보정 계수를 추정해 LMPC의 동특성 불확실성을 완화하고, DNN 오토인코더로 Koopman lifting 함수를 학습해 DK의 표현력을 키웁니다. 마지막으로 load transfer rate 기반 활성 조건과 sigmoid 기반 credibility verification을 넣은 EPC(event-triggered parallel cooperative)로, DK 보상 조향각 수열의 실행 가능성과 차량 안정성을 함께 확인한 뒤 병렬 보상을 수행합니다.

- **Empirical Impact**: 하드웨어 인더 루프(HiL) 실험을 통해 다양한 운용 조건에서 제안 전략의 추종 성능이 평균 11.5% 이상 향상됨을 보였고, 포트홀과 같은 단거리 요철에서도 편차가 줄어듦을 확인했습니다. 특히 LMPC의 baseline 추종과 DK 편차 보상을 이벤트 트리거로 관리해, 단순 보상 합성에서 자주 발생하는 안정성/제약 위반 위험을 완화한 점이 실험에서 실용성을 뒷받침합니다. 오프로드 UGV 제어에서 “해석 가능 Koopman + 비선형 학습 + 제약을 고려한 보상 스케줄링”의 결합 가능성을 보여주며, 향후 실차 적용 설계에도 참고가 될 만한 결과로 평가됩니다.



### Self-Supervised Mask-Aware Transformers for Fault-Tolerant FBG Force Sensing in Minimally Invasive Surgical Robotics (https://arxiv.org/abs/2606.18628)
- **Prior Approaches**: 기존 최소침습 로보틱스용 FBG 힘 센서는 채널 간 비선형 크로스축 결합과, 반복 굽힘 중 커넥터 피로/부분 광섬유 파열로 인한 간헐적 채널 드롭아웃을 동시에 다루기 어려웠습니다. 이를 보완하려는 fault-tolerant 접근은 failure pattern마다 모델 은행(model bank)을 유지하는 조합폭 방식이어서 채널 수가 늘면 2^C-1로 학습·메모리·온보드 지연 비용이 급증합니다. 또한 불확실성 추정도 MC Dropout/Deep Ensembles처럼 다중 forward가 필요하거나, 결측의 물리적 열화를 명시적으로 반영하지 못한다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 채널 가용성을 입력 마스크로 명시적으로 모델링하는 mask-aware Transformer를 단일 통합 모델로 제안합니다. self-supervised masked-channel reconstruction으로 라벨 없는 스트림에서 채널 간 물리 상관을 먼저 학습하고, 이후 힘 회귀(force regression)에는 clean/contaminated view를 함께 쓰는 균형 학습을 적용합니다. 더불어 단일 forward에서 축별 heteroscedastic uncertainty를 내는 uncertainty head를 붙여, 패턴별 다중 모델 없이도 결측·열화 상황에서의 신뢰도 감지를 가능하게 했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 촘촘한 제조 공차로 생기는 강한 비선형 크로스축 결합을 결측 상황에서도 유지하는 것, (2) 채널이 사라졌을 때 학습된 네트워크가 입력 imputation에 과도하게 의존하며 성능이 무너지는 것을 막는 것입니다. 저자들은 결측 채널을 마스크로 토큰에 게이팅하고 attention key-padding으로 차단해 정상 채널 표현을 오염시키지 않게 했으며, fiber 토폴로지를 반영한 multi-scale 토큰(마스크 평균 풀링)으로 국소·전체 구조 정보를 함께 융합합니다. 또한 dynamic corruption curriculum로 결측 강도를 점진적으로 키워, 정보 부족이 클수록 분산이 커지도록 NLL 기반 불확실성 학습을 실질적으로 ‘열화(에피스테믹) 대리’로 동작하게 만들었습니다.

- **Empirical Impact**: 카테터 스케일 8채널 FBG 데이터셋에서 단일 모델은 완전 관측(k=0) RMSE 0.0066 N을 달성하고, 4채널 심각 실패(k=4)에서도 0.0126 N으로 점진적 열화를 보였습니다. 이는 failure pattern별 255개 모델 은행의 0.0154 N 대비 18.2% 개선이며, 패턴별 전용 캘리브레이션을 제거했다는 점에서 실용성이 큽니다. 불확실성 측면에서도 위험-커버리지 곡선과 상관분석에서 예측 신뢰도가 실제 오차를 잘 반영했고, MC Dropout/Deep Ensembles 대비 연산 오버헤드가 1/5~1/30 수준으로 줄어 실시간 closed-loop 제어에 유리함을 보여줍니다.



### SRL: Combining SLIP Model and Reinforcement Learning for Agile Robotic Jumping (https://arxiv.org/abs/2606.18625)
Comments:
          17 pages, 12 figures

- **Prior Approaches**: 로봇 점프 제어는 생체모방/모델 기반(예: SLIP, MPC)과 데이터 기반(RL)로 크게 나뉩니다. SLIP은 스프링-질량의 에너지 저장/방출을 잘 캡처하지만 접촉·관절 동역학과 마찰 등 단순화 가정 때문에 불규칙 지형에서 성능이 떨어집니다. 반면 RL-only는 환경 적응은 뛰어나지만, 물리적 prior가 없어 시행착오 탐색으로 학습이 비효율적이고 불안정할 수 있습니다.

- **Core Contribution**: 이 논문은 Spring-loaded Reinforcement Learning(SRL)로 SLIP의 feedforward 물리 기준선과 RL의 실시간 feedback을 결합해 두 방법의 약점을 동시에 줄이는 하이브리드 프레임워크를 제안합니다. SLIP이 생성한 기준 점프 궤적을 IK로 관절 명령으로 변환한 뒤, PPO가 관측(자세·속도·발 배치 등)을 기반으로 안정성을 보정합니다. 또한 six-state FSM, 가중치 융합, 그리고 PD 컨트롤러로 추종 정밀도까지 보강해 end-to-end 학습의 불안정성을 완화합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 단순화된 SLIP 동역학을 실제 로봇 관절·접촉 상황에 맞게 보정하는 것과 (2) 학습 중 안정성을 유지하며 샘플 효율을 높이는 것입니다. 연구진은 SLIP 기반 기준 신호에 PPO feedback을 가중 결합하고, FSM으로 비행/접지 등 위상별로 스프링 파라미터를 조절해 점프 사이클 일관성을 만들었습니다. 더불어 속도 추종 보상과 위상 의존 보상 설계를 통해 RL이 잘못된 탐색 대신 목표 상태로 빠르게 수렴하도록 유도했습니다.

- **Empirical Impact**: 실험은 Unity+ML-Agents 시뮬레이션에서 이족(X02-lite)과 사족(Unitree Go2)을 대상으로 fixed/random-distance, 계단 형태 등 다양한 점프를 검증하며 sim-to-sim 및 sim-to-real까지 확장됩니다. SRL은 SLIP-based MPC 및 RL-only(PPO) 대비 더 높은 성공률(이족 98.5%, 사족 99.8%)과 더 적은 학습 단계로 수렴했으며, RL-only의 긴 학습 비용을 크게 줄였습니다. 추종 성능도 평균 위치 오차 0.1 m 미만, 속도 오차 ±3% 이내로 보고되어, 불규칙 지형에서의 강인한 점프 제어가 실제 배치 가능하다는 점을 시사합니다.



### SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation (https://arxiv.org/abs/2606.18610)
- **Prior Approaches**: 일반 로봇 조작 정책을 실세계에서 평가하려면 물리 로봇 롤아웃, 초기화, 감독 비용이 커서 확장성이 떨어진다. 이를 줄이기 위해 action-conditioned video world model로 정책 롤아웃을 시뮬레이션하고 점수를 매기는 방식이 등장했지만, autoregressive rollouts에서 발생하는 drift 누적과 멀티카메라 관측 간 비일관성 문제가 남는다. 또한 학습 분포 밖에서 행동하는 정책에 대한 evaluator의 일반화도 취약하며, 불확실성 기반 종료를 ensemble 등 추가 학습 없이 일관되게 구현하기 어렵다.

- **Core Contribution**: SC3-Eval은 pre-trained video foundation model을 self-consistent 비디오 생성 레시피로 바꿔, 실세계 성능과 높은 상관을 보이는 closed-loop 정책 evaluator로 적응시키는 방법이다. 핵심은 forward-inverse dynamics consistency(정방향 예측-역방향 동작 복구), cross-view consistency(다른 카메라로부터 inpainting), test-time consistency(역동역학 신호로 per-action-chunk drift 시 롤아웃 조기 종료) 3가지 일관성을 함께 강제하는 점이다. 그 결과 점수의 절대 캘리브레이션뿐 아니라, 실제 롤아웃에서 나타나는 실패 모드까지 더 세밀하게 진단할 수 있다.

- **Technical Challenges**: 기여를 실현하려면 (1) 정방향 모델만으로는 물리적으로 불가능한 프레임을 효과적으로 페널티하기 어렵고, (2) 여러 카메라 관측이 롤아웃 동안 서로 어긋나기 쉬우며, (3) 학습 분포 밖 정책 행동에 대해 evaluator가 계속 신뢰를 유지해야 한다. SC3-Eval은 forward-inverse 모드의 파라미터 공유로 생성 프레임이 ‘요청된 행동을 복구할 수 있는’ 물리적으로 그럴듯한 action manifold에 묶이도록 학습한다. 동시에 카메라 하나를 숨기고 나머지로 다른 뷰를 채우는 cross-view inpainting으로 멀티뷰 일관성을 학습하고, test-time에서는 inverse dynamics로 얻은 action 복구 오차를 불확실성/드리프트 지표로 써 τ를 넘으면 즉시 종료해 누적 오류가 점수에 오염되는 것을 막는다.

- **Empirical Impact**: 실세계 table bussing 7개 vision-language-action(VLA) 정책 체크포인트에 대해 SC3-Eval은 closed-loop Pearson 상관 0.929, MMRV 0.119로 Ctrl-World, IRASim, Cosmos-Predict 2.5 같은 강한 비디오 모델 기반 evaluator를 앞섰다. 또한 reverse table bussing처럼 의미론이 바뀐 out-of-distribution 과제에서도 성능 저하가 관측되지만 평가가 유지되며, aggregate 성공률을 넘어서 실제와 동일한 실패 범주(언어 미준수/물체 들어올림/놓기)를 더 잘 재현한다. 이는 정책 개발에서 체크포인트 선택과 원인 진단을 동시에 강화할 수 있는 평가 도구로서 의미가 크다.



### Admittance-Based Surface Alignment for Human-in-the-Loop Robotic Visual Inspection (https://arxiv.org/abs/2606.18601)
- **Prior Approaches**: 기존 로봇 비전 검사는 대부분 RGB-D/point cloud를 기반으로 전역 3D 모델을 만들거나 사전 계산한 커버리지 경로를 실행하는 open-loop(오프라인 계획) 방식에 의존해 왔습니다. 일부 온라인 보정이 있더라도 대부분은 이산적인 perception–correction 단계로 끝나거나, IR/근접 센서를 쓰는 경우가 많아 사람의 실시간 조작과 “연속 closed-loop”로 자연스럽게 공존하기가 어려웠습니다. 또한 산업용 로봇은 기본적으로 position/velocity 추종 위주라 힘-순응 제어를 직접 통합하기가 제한적이었습니다.

- **Core Contribution**: 이 논문은 사람의 teleoperation 입력과 depth 기반 표면 정렬을 하나의 실시간 closed-loop 순응 제어로 결합하는 “admittance-based orientation control” 파이프라인을 제안합니다. 표면 normal 추정값과 오리엔테이션 오차를 외곽(outer-loop)에서 PD로 처리하고, 이를 가상 질량-댐퍼(virtual mass–damper) 동역학에 넣어 사람이 주는 명령과 인지가 주는 보정을 동시에 만족하는 순응 운동으로 변환합니다. 그 결과 미리 계산한 검사 궤적 없이도 end-effector가 local surface geometry에 안정적으로 normal-tracking을 수행합니다.

- **Technical Challenges**: 핵심 난제는 (1) depth 잡음과 표면 불규칙성 때문에 normal 추정이 흔들릴 때도 안정적으로 정렬을 유지하고, (2) 토크/속도 제한이 있는 position/velocity 제어 로봇에서 사람 입력까지 포함해 매끈한 반응을 보장하는 것입니다. 논문은 RANSAC plane fitting+PCA로 강인한 normal을 만들고, 가상 구체(virtual sphere)–점성 매질의 물리 해석 가능한 질량-댐퍼 모델로 orientation error와 operator 입력을 단일 속도 명령 스트림으로 매핑합니다. 또한 ROS 2 Servo를 통해 표준 서보 인터페이스 위에서 task-space velocity를 내려 “모듈형 외곽 순응 레이어”로 동작하도록 설계해 산업 플랫폼 호환성을 확보했습니다.

- **Empirical Impact**: UR5e+eye-in-hand RealSense D405 환경에서 실험한 결과, 토크 포화 구간과 비포화 구간 모두에서 각도 오차가 안정적으로 수렴했으며 시뮬레이션과 로봇 응답의 신호 추세가 잘 일치했습니다. 특히 최종 평균 orientation error가 0.4°로 보고되어, IR 기반 iterative adaptive 제어 계열의 기존 성능과 동등한 수준을 depth 기반 closed-loop로 달성했다는 점이 의미가 있습니다. 수동 teleoperation(10 trials) 대비 완료 시간의 변동성이 더 작고 재현성이 높아, 사람의 주관성에 덜 의존하는 shared autonomy 검사 보조기로서의 실용성이 확인됐습니다.



### Benchmarking Action Spaces in Reinforcement Learning for Vision-based Robotic Manipulation (https://arxiv.org/abs/2606.18594)
Comments:
          9 pages with references

- **Prior Approaches**: 로보틱스 조작에서 action space 설계는 motion smoothness, 안전성, sim-to-real 전이 성능에 큰 영향을 준다고 알려져 있다. 다만 기존 연구는 주로 상태 기반 관측(state-based observations)이나 시뮬레이션에 치우쳐 있고, vision으로 인한 부분관측과 지각 잡음이 실제 학습/제어 동역학을 어떻게 바꾸는지까지는 충분히 다루지 못했다.

- **Core Contribution**: 이 논문은 vision-based 조작 문제에서 pose increment, pose velocity, joint position increment, joint velocity 같은 대표 action space 4가지를 Franka Emika Panda에서 비교한다. 시뮬레이션(PPO) 학습 후 sim-to-real transfer로 실제 로봇에 배치해, action space 선택이 실제 성능을 유의미하게 좌우함을 보인다. 특히 joint velocity action space가 picking과 pushing 모두에서 smoothness와 최종 과제 성능 측면에서 가장 유리하다고 정리한다.

- **Technical Challenges**: 핵심 난제는 vision이 만드는 부분관측/잡음이 action space에 따라 학습 안정성과 실제 동작을 다르게 흔들 수 있다는 점이다. 저자들은 domain randomization으로 카메라/조명/물체 초기조건/역학 파라미터를 흔들고, PPO의 vision actor–critic 구조에 맞춰 action을 조절 가능한 actuator 모델(예: joint velocity/position increment, Cartesian 조절 포함)로 일관되게 매핑해 전이 민감도를 낮추는 방향을 택했다.

- **Empirical Impact**: 실제 평가에서 joint velocity는 충돌을 피하고 jerk가 가장 낮은 편이었으며, picking 과제에서 성공률 100%를 달성(중앙 완료시간 약 3.58s)했다. 반면 pose increment나 pose velocity는 실제에서 불안정/오프태스크 경향이 커져 성능이 크게 떨어졌다. 저자들은 이를 바탕으로 vision 기반 sim-to-real 실험에서 action space를 고를 때 joint velocity를 우선 후보로 검토하라는 실무 가이드를 제시한다.



### DREAM-Chunk: Reactive Action Chunking with Latent World Mod (https://arxiv.org/abs/2606.18589)
- **Prior Approaches**: Action chunking은 VLA가 짧은 행동 시퀀스를 생성해 로봇을 고주파로 실행하게 해 주지만, 한 chunk를 커밋하면 open-loop로 진행돼 잡음·하드웨어 오류·부분관측에서 오차가 누적되기 쉽습니다. 이를 줄이기 위해 RTC/BID처럼 test-time에 재계획·샘플링을 늘리거나, 실행 horizon을 적응적으로 바꾸는 접근이 나왔지만 계산비용이 커지고 이미 생성된 장기 chunk 이점을 낭비할 수 있습니다. 또한 얕은 아키텍처/모듈 수정은 스케일링 한계가 생길 수 있습니다.

- **Core Contribution**: DREAM-Chunk는 기반 VLA를 수정하거나 fine-tuning 없이, test time에만 계산을 추가해 chunk 실행의 반응성을 높이는 test-time scaling 방법을 제안합니다. 고정된 정책이 샘플링한 여러 candidate action chunk에 대해 lightweight latent world model로 latent future를 예측하고, 관측된 상태와 phase-aligned로 가장 잘 맞는 chunk의 행동을 선택합니다. 즉 stochastic dynamics로 실제 궤적이 nominal rollout에서 벗어나면, 미리 샘플된 후보들 사이에서 반응적으로 스위칭합니다.

- **Technical Challenges**: 핵심 기술적 난제는 open-loop 오차 누적을 줄이되, VLA를 자주 재추론하지 않고도 신뢰할 수 있는 “꿈꾼 미래(예측 latent rollout)”를 만들어야 한다는 점입니다. 논문은 관측을 latent로 인코딩하고 candidate chunk별 latent dynamics를 batched로 롤아웃한 뒤, 동일 phase(시간 정렬)에서의 latent-state 매칭으로 후보를 고르는 규칙을 설계해 temporally aligned 전환을 가능하게 했습니다. 또한 decoder 없이도 작동하는 latent representation과 dynamics 모델 설계가 중요하다는 점을 실험적으로 확인하며, world model이 충분히 예측성을 가져야 long-horizon 매칭이 안정적임을 보여줍니다.

- **Empirical Impact**: Kinetix에서 action noise가 커질수록 DREAM-Chunk의 성능 향상 폭이 커졌고, candidate sample 수를 늘릴수록 realized trajectory를 더 잘 커버하며 robustness가 개선됐습니다. 특히 시연에 corrective behavior가 포함된 경우에 scaling 이점이 크게 나타나 test-time scaling이 “회복 옵션”이 존재할 때 최대로 발휘됨을 시사합니다. 더 나아가 SO-101/Franka 두 로봇 플랫폼의 네 가지 조작 과제에서 hardware execution error·부분관측·외란 등 다양한 불확실성 하에 성공률이 전반적으로 상승했으며, 장시간 chunk 실행에서 open-loop 대비 큰 개선(예: 정밀 삽입에서 10%→65%)을 보고합니다.



### As You Wish: Mission Planning with Formal Verification using LLMs in Precision Agricultur (https://arxiv.org/abs/2606.18519)
- **Prior Approaches**: 기존 LLM 기반 로봇 미션 플래너는 자연어를 실행 가능한 임무로 바꾸는 데 강점이 있지만, 자연어의 내재적 모호성 때문에 사용자가 의도한 것과 다른 행동이 생성·실행될 수 있다. 또한 formal verification을 하더라도 사용자가 PDDL/LTL 같은 형식 언어를 직접 알아야 하거나, 검증이 사람의 개입 없이 완전히 자동으로 닫히기 어렵다는 한계가 있었다. 일부 연구는 LTL을 검증 입력으로만 쓰거나(데이터 흐름의 일부가 아님), LTL 생성은 별도로 하되 온라인 검증 파이프라인으로 일관되게 엮지 못했다.

- **Core Contribution**: 이 논문은 자연어 기반 precision agriculture 미션 플래너에 LTL 기반 검증을 결합하되, LTL 스펙을 사용자 친화적으로 “자동 생성”해 사람의 형식 언어 학습 부담을 없애는 아키텍처를 제안한다. 특히 미션 생성(로봇 task용 XML)과 검증을 위한 LTL 공식 생성 사이를 독립된 에이전트로 분리해 LLM bias를 줄이고, 검증 불일치가 나면 피드백 루프로 재생성한다. 결과적으로 “NL → 미션 생성 → LTL 스펙 생성 → 모델체킹(SPIN)으로 사양 위반 검사”가 인간 감독 없이도 루프 형태로 닫히는 완전 자율 파이프라인을 지향한다.

- **Technical Challenges**: 핵심 난제는 (1) 자연어 의도를 정밀하게 LTL로 표현하는 것, (2) SPIN/모델체킹이 감당 가능한 복잡도의 LTL을 생성하는 것, (3) 에이전트 간 사전 합의가 과도하게 들어가 검증의 독립성을 훼손하지 않는 것이다. 이를 위해 시스템은 XML 미션을 IEEE 표준 기반 XSD 제약과 linter로 문법적으로 보장하고, 검증용 LTL은 co-safe LTL의 제한 클래스(유한 실행 조각에서 만족/위반이 결정되는 형태)로 생성해 상태공간 폭발을 완화한다. 또한 두 LLM을 완전히 decoupling 후, 승인 단계에서만 최소한의 분해 합의(예: 작업/조건의 개수)만 확인하며, SPIN 위반 또는 LTL 문법/의미 불일치에 대해 재시도 피드백을 주어 수정한다.

- **Empirical Impact**: 현장 실험은 ClearPath Husky(ROS2 Humble, Jetson Orin Nano)에서 진행하며, End-to-end 성공적인 L1 미션 생성 속도와 함께 formal verification이 실제로 성립하는지(세 에이전트 합의 횟수, 의미적 일치/오해 여부, 평균 시간 등)를 평가한다. 논문은 SPIN/Spot을 온라인 승인 단계에 넣어 LTL 생성의 문법 오류(특히 괄호 매칭)와 의미 불일치가 재시도 루프에서 어떻게 회복되는지, 그리고 LTL 공식 품질이 성능 상한을 좌우할 수 있음을 강조한다. 전반적으로 “완전 자율 검증 파이프라인”의 강점과 LLM이 유의미한 LTL을 만드는 데서 생기는 제약을 함께 보여주며, 농업 로봇처럼 네트워크 없이도 안전한 오퍼레이션을 요구하는 도메인에 실용적 시사점을 제공한다.



### Task Allocation and Motion Planning in Dynamic, Cluttered Environments via CBBA and Graphs of Convex Sets (https://arxiv.org/abs/2606.18516)
Comments:
          15 pages single column, 10 figures, AIAA-Scitech 2027 Submission

- **Prior Approaches**: 기존 멀티에이전트 태스크 플래닝은 작업 배정과 경로 생성이 결합된 문제임을 인식해 왔지만, 실제로는 CBBA 같은 분산 배정에서 입찰 비용을 거리나 명목 travel time 같은 단순 모델로 근사하는 경우가 많았다. 그 결과 분산 합의는 충돌 없는 할당으로 수렴하더라도, 이후 ST-GCS/GCS 기반의 상세 모션 제약을 적용하면 비실행 가능하거나 비효율적인 선택이 생길 수 있다. 또한 GCS는 전역 최적 성질을 갖지만, 기존 연구는 이를 분산형 task-allocation 프레임워크의 ‘입찰 단계’에 직접 연결해온 사례가 드물었다.

- **Core Contribution**: 이 논문은 ST-GCS(공간-시간 Graphs of Convex Sets)를 3D+time으로 확장해, 복잡한 동적 환경에서 안전하고 시간 파라미터화된 궤적을 계산한다. 동시에 CBBA를 분산 태스크 배정의 조정층으로 두고, ST-GCS가 계산한 궤적 최적화 비용을 입찰(bid) 값으로 사용해 allocation–planning 결합도를 높인다. 더 나아가 rendezvous 형태의 dynamic tasks까지 포함해 작업의 위치·가용성이 시간에 따라 변하는 상황을 일관된 프레임워크에서 다룬다.

- **Technical Challenges**: 핵심 난제는 “입찰 모델이 실제 궤적의 시간-공간 제약을 반영하지 못하면 할당이 무의미해진다”는 점이며, 이를 위해 각 후보 task 삽입에 대해 공간-시간 최적화(ST-GCS) 기반 경로 점수/비용을 계산해 marginal score로 전환한다. 3D+time 표현에서는 시간 좌표를 사후가 아니라 궤적 최적화 변수로 포함해, 이동 장애물과 moving tasks, 속도·인과성(causality) 같은 제약을 같은 convex-set 최적화에 내재화한다. 또한 계산량을 줄이기 위해 그리디 확장 중 변경되는 구간만 ST-GCS를 다시 풀거나, 이미 계산한 세그먼트 비용을 캐싱해 반복 최적화 부담을 완화한다.

- **Empirical Impact**: 시뮬레이션의 cluttered 환경에서 static과 dynamic task를 모두 대상으로, 제안 방식이 충돌 회피와 함께 task completion의 시간 추정 정확도를 높이는 것을 보인다. 특히 단순 거리/명목 시간 기반 CBBA는 비슷한 기하학적 이득을 주더라도 시간-공간 제약 때문에 실제 실행 가능성이 크게 갈릴 수 있는데, 논문은 입찰 단계에서 이를 직접 반영함으로써 이러한 불일치를 줄인다. 결과적으로 분산형 task allocation과 최적화 기반 motion planning을 더 촘촘히 결합한 실무형 설계를 제시하며, 동적 환경의 멀티에이전트 시스템 연구에 영향력을 줄 것으로 기대된다.



### N(CO)$^2$: Neural Combinatorial Optimization with Chance Constraints to Solve Stochastic Orienteering (https://arxiv.org/abs/2606.18514)
- **Prior Approaches**: 기존 Neural Combinatorial Optimization(NCO)은 그래프 최적화에서 휴리스틱을 학습하되, 주로 결정론적 문제에 초점이 맞춰져 왔다. 확률/제약이 있는 확률적 orienteering 계열(SOP)은 샘플 기반 chance constraint 계산이 붙어 난도가 높아, 관련 연구가 상대적으로 적고, MCTS 기반 접근은 온라인으로 강하지만 상태공간 근사가 커 실시간성이 떨어지거나 실패확률 일반화에 한계가 있었다. 또한 MILP 같은 정확해법은 품질은 높지만 계산 시간이 크게 달라 오프라인 지향이라는 제약이 있었다.

- **Core Contribution**: 이 논문은 chance constraint를 다루기 위해 확률적 휴리스틱을 학습하는 N(CO)²: Neural Combinatorial Optimization with Chance cOnstraints를 제안한다. Stochastic Orienteering Problem(SOP)을 대상으로, 수동 휴리스틱 없이 edge heatmap 형태의 학습된 정책과 RL을 결합해 경로 선택을 수행한다. 특히 실패확률 초과 여부를 포함한 보상 설계를 통해 “예산 초과 확률을 통제하면서” 최대 보상을 추구하도록 한다.

- **Technical Challenges**: 핵심 난제는 (1) 확률적 비용 때문에 제약이 결정적으로 판정되지 않고, (2) chance constraint를 만족하는 그리디(순차 구성) 선택이 학습 과정에 반영되어야 한다는 점이다. 논문은 Sample Average Approximation(SAA)로 실패확률을 추정하고, 경로 확장 시 확률 제약을 위반할 후보를 Fmask로 마스킹한 뒤 SAA 기반 수락/거절로 구성 알고리즘을 안정화한다. 또한 비자동회귀(non-autoregressive) edge 예측(heatmap) 구조를 강화학습(REINFORCE)과 연결하고, graph/edge 정보를 처리하는 Edge-Augmented Transformer(EGT) 인코더-디코더로 다양한 크기 그래프에 대해 휴리스틱을 생성한다.

- **Empirical Impact**: 실험에서는 SOPCC에 대해 MILP(GUROBI)와 비교해 경쟁 수준의 성능을 보이면서도 더 빠른 해 생성 속도를 보고한다. 또한 기존 휴리스틱 기반 MCTS/학습 휴리스틱 대비로, 실패확률 추정의 일반화 문제를 완화하며 다양한 SOP 인스턴스에 대해 성능이 유지되는 점을 강조한다. 결과적으로 인간이 제약을 고려한 “손수 만든 휴리스틱”을 설계해야 했던 부담을 줄이고, 불확실한 환경에서 적응적이고 효율적인 의사결정을 가능하게 한다.



### VEGA: Learning Navigation VLAs from In-the-Wild Egocentric Video with Geometric Trajectory Supervision (https://arxiv.org/abs/2606.18426)
- **Prior Approaches**: 기존 navigation VLA는 자연어·이미지·2D pose 같은 목표를 처리할 수 있지만, 학습은 주로 텔레옵/데모 기반 데이터에 의존해 씬 내부의 다양한 목표에 대한 촘촘한 감독이 부족했습니다. 또한 많은 방식이 관측-목표마다 단일(또는 소수) 궤적을 따라가면서, 다중 경로가 가능한 clutter 환경에서 goal grounding과 안전한 장애물 회피가 약해질 수 있습니다. Egocentric video를 활용하더라도, 액션 없이 얻은 영상으로부터 로봇 좌표계 기준 장애물-인지 목표조건 궤적을 생성해 학습에 쓰는 방법은 제한적이었습니다.

- **Core Contribution**: VEGA는 라벨 없는 egocentric navigation video로부터 monocular geometry를 복원한 뒤, 텍스트/이미지 영역/공간 웨이포인트 목표를 샘플링하고 장애물 인지(충돌 회피) 궤적 분포를 생성해 navigation VLA를 학습합니다. 핵심은 geometry를 학습 단계에서만 써서, 최종 정책은 inference 시 RGB와 목표 조건만으로 obstacle-aware planning을 “증류(distill)”하도록 만든 점입니다. 또한 목표에 따라 달라지는 다중 모달 행동 분포를 flow-matching 방식으로 학습하도록 설계했습니다.

- **Technical Challenges**: 가장 큰 기술 난제는 액션이 없는 영상에서 로봇 좌표계 기준의 obstacle-aware, goal-conditioned 감독을 복원하는 것입니다. VEGA는 MoGe-2로 촘촘한 점맵을 만들고 지면 정렬(RANSAC), BEV 투영, 높이 필터링·시야 마스크를 통해 ESDF(유클리드 서명거리장)를 구성한 뒤 MPPI로 ESDF 기반 비용을 최소화하는 후보 궤적을 생성합니다. 이후 생성된 다목표 궤적 분포를 flow-matching VLA의 waypoint action head 학습 타깃으로 사용해, 단일 경로 모드 붕괴를 줄이도록 했습니다.

- **Empirical Impact**: VEGA-Bench(250k 씬, 약 500만 목표)와 실제 로봇 실험에서 VEGA는 강력한 baseline 대비 충돌을 33.0% 줄이고 장애물 clearance를 17.9% 개선하면서 goal progress는 경쟁 수준을 보였습니다. 더 나아가 real-world trials에서는 success를 최소 150.0%, collisions를 최소 66.7%, obstacle clearance를 최소 60.0% 향상시키는 결과를 보고했습니다. 이는 “video-derived geometric supervision”이 obstacle-aware navigation VLA를 규모 있게 학습하는 실용적 신호가 될 수 있음을 보여 주며, 코드·벤치마크 공개로 후속 연구의 비교 기준도 강화될 전망입니다.



### PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation (https://arxiv.org/abs/2606.18375)
- **Prior Approaches**: 기존 World foundation model(WFM)은 Cosmos처럼 대체로 single-view 영상 롤아웃에 강점이 있지만, 로봇 조작에는 여러 카메라 시점에 대한 엄격한 3D 일관성이 필요합니다. 멀티뷰를 다루는 방법도 Genie나 iVideoGPT처럼 view 토큰을 단순 연결(flat concatenation)하는 방식이 많아, 학습이 데이터에만 의존하는 탓에 시점 간 물체 드리프트·깊이 불일치·텍스처 어긋남이 커집니다. 저자들은 이러한 실패가 (1) 시점 간 정보 교환 경로 부재와 (2) 명시적 3D 기하 prior 부재라는 두 결핍에서 나온다고 진단합니다.

- **Core Contribution**: PAIWorld는 멀티뷰 3D consistency를 “동시에” 해결해야 한다는 주장 아래, 두 기둥을 함께 설계합니다. 아키텍처에서는 시점 간 교통로를 열어주고(Geometry-Aware Cross-View Attention + Geo-RoPE), 학습 목적에서는 3D-aware 표현을 주입해 교통로를 ‘기하적으로 의미 있게’ 만듭니다(Latent 3D-REPA). 이 조합이 로봇 조작용 멀티뷰 세계 시뮬레이터가 필요한 시점 간 정합성을 만든다는 점이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 “보기 좋게”가 아니라 “기하적으로 맞게” 멀티뷰를 동시에 생성/예측하는 것입니다. 단순 멀티뷰 토큰 연결은 시점 토큰을 구분하지 못해 암묵적으로 대응을 학습해야 하며, 별도 교통로가 없거나 3D 감독이 없으면 색/텍스처 복사 같은 지름길로 수렴해 일관성이 무너집니다. PAIWorld는 Geo-RoPE로 카메라 ray 방향과 외부 파라미터를 attention에 반영해 동일 3D 포인트 토큰이 더 높은 attention을 받게 하고, Cross-View Attention으로 시점 간 정보를 교환하며, Latent 3D-REPA에서 Depth Anything 3 같은 3D-aware 인코더의 관계(토큰-토큰 relation)를 증류해 3D 일관성을 강제합니다.

- **Empirical Impact**: 실험에서 PAIWorld는 로봇 조작 멀티뷰 벤치마크에서 SOTA를 달성합니다. WorldArena 리더보드에서 1위(WorldArena EWMScore 70.67%, Motion Quality 최고)와 AgiBot-Challenge2026 리더보드에서 2위(EWMScore 82.45%)를 기록했으며, 특히 AgiBot에서는 Scene Consistency 90.41%로 전체 최고 성적을 냈습니다. 또한 개선된 3D 정합성이 model-based planning과 world action model fine-tuning 같은 downstream embodied task 성능 향상으로 이어진다고 보고합니다.



### Guava: An Effective and Universal Harness for Embodied Manipulation (https://arxiv.org/abs/2606.18363)
- **Prior Approaches**: 기존 접근은 vision-language-action(VLA)로부터 입력(영상·언어)→로봇 행동을 end-to-end로 생성하거나, VLM을 고수준 추론기로 두고 별도 지형/프리미티브 기반 인터페이스로 실행하는 방식이 주류였습니다. 다만 이들 방식은 로봇 시연 데이터 의존성이 크거나, 행동이 한 번에 생성되어 실행 실패 후 재계획·복구가 어렵고, 계획을 명시적으로 점검하며 수리하기도 난해하다는 한계가 있습니다. 또한 harness 형태의 modular tool use가 시도되었지만, one-shot 프로그램 생성·실행 중심이라 long-horizon에서 실패 회복을 지속적으로 수행하기 어렵다고 정리합니다.

- **Core Contribution**: 이 논문은 embodied tool use를 위한 harness 프레임워크 Guava를 제안하고, 유효한 harness의 설계 재료를 반복형 워크플로우·행동 추상화·멀티모달 관측의 3요소로 정리합니다. 핵심은 언어 모델이 외부 모듈(지각/계획/제어)을 호출하도록 구성하되, ReAct 스타일의 반복 루프로 실행 결과를 관측하며 사고-행동을 교차시키는 것입니다. 이를 통해 다양한 reasoning 모델 전반에 걸쳐 조밀한 end-to-end 정책 대신 “모델-비의존적 인터페이스”로 embodied manipulation 역량을 이식할 수 있음을 목표로 합니다.

- **Technical Challenges**: 가장 큰 난제는 (1) 폐루프 상호작용에서 발생하는 실패·상태 이탈을 견디는 도구 설계와 워크플로우를 찾는 것, (2) compact 모델로의 증류를 위해 소량 고품질 시연을 확보하는 것입니다. Guava는 semantic action abstraction(예: grasp(), align(), home_pose())으로 저수준 기하/물리 추론 부담을 언어 모델에서 덜어주고, 시각 관측과 텍스트 기반 상태 표현을 함께 넣어 그라운딩 모호성을 줄이며, 실행 피드백을 반영하는 iterative perception-reasoning-action 루프로 복구를 강화합니다. 또한 2K 미만 시뮬레이션 궤적만으로도 frontier VLM의 tool use를 4B 모델로 증류하도록, failure perturbation 기반 recovery 데이터와 SFT+GRPO(희소 성공 보상 RL)를 결합한 데이터 효율 학습 파이프라인을 구성합니다.

- **Empirical Impact**: 실험 결과 Guava-Agent-4B는 시뮬레이션(Robosuite)과 실세계(Franka Research 3)에서 모두 높은 성공률을 보이며, 종단 전략이나 one-shot 코드 실행 기반 경쟁 접근보다 long-horizon에서 우수한 복구 성능을 보였습니다. 특히 2K 미만 시뮬레이션 궤적로 학습한 4B open-source 모델이 proprietary frontier VLM과 유사한 수준의 전반 성능을 내고, 미지 물체·새 지시·긴 과제 구성에서도 강한 일반화를 보였다는 점이 강조됩니다. 결론적으로, 잘 설계된 harness가 데이터 적게 써도 compact 모델에서 emergent한 embodied 역량(복구 포함)을 끌어낼 수 있는 확장 가능한 인터페이스가 될 수 있음을 실증하며, 이후 모델 크기/데이터/툴셋 확장 가능성도 제시합니다.



### Recover, Discover, Plan: Learning Skills and Concepts from Robot Failures (https://arxiv.org/abs/2606.18328)
Comments:
          9 pages, 6 figures. Website: this https URL

- **Prior Approaches**: 기존 연구는 실패에서 벗어나는 recovery skill을 학습하거나(주로 RL 기반), 계층적 planning을 위해 predicate 같은 추상 개념을 별도 데이터로 학습하는 방식에 치우쳤다. 하지만 failure mode마다 별도 정책을 학습하거나, 회복은 실패 후에만 작동하는 ‘반응형’ 접근이 많아 실패의 공유 구조를 활용하기 어렵다. 또한 개념-스킬 양쪽을 함께 배우려면 손수 설계나 충분한 감독(데이터/데모)이 필요한 경우가 많다.

- **Core Contribution**: ReSYNC는 Recovery-Driven Synthesis of Relational Concepts로, 실패-회복 경험을 통해 state abstraction(관계 술어) 라이브러리를 단계적으로 발견·정교화한다. 먼저 RL로 실패를 회복하는 스킬을 학습한 뒤, 그 회복 동작을 설명하고 재사용 가능하게 만드는 predicate와 planning 연산자(operators)를 함께 갱신한다. 이 상호작용을 통해 학습 중에 본 ‘국소 회복’을 테스트 시에는 ‘전역적 실패 회피(abstract planning)’로 연결한다.

- **Technical Challenges**: 핵심 난제는 (1) 회복 스킬은 학습했지만, 이를 테스트의 새로운 조합 상황에서 재구성하려면 어떤 관계 개념이 필요한지 알아내야 한다는 점이다. ReSYNC는 실패 마이닝으로 새 recovery 학습용 상태를 만들고, IVNTR 기반의 offline concept learning을 적용하되 데모 없이 self-supervised로 데이터를 생성하기 위해 dreaming을 도입한다. 또한 개념을 고정하면 이후 스킬 학습으로 분포가 변해 술어 정확도가 깨질 수 있어, 기존 개념 classifier를 fine-tuning해 일관성을 유지하면서도 점진적으로 라이브러리를 확장한다.

- **Empirical Impact**: 4개 시뮬레이션 도메인에서 ReSYNC는 compositionally novel하고 long-horizon인 미해본 문제에서 평균 성공률 70%를 달성하며, 강력한 baseline 대비 50% 이상 성능 향상을 보였다. 더 나아가 sim-to-real 전이를 통해 비전습(non-prehensile) 조작 스킬을 실제 환경에서 수행하고, 추상 planning으로 새로운 시나리오까지 일반화하는 결과를 제시한다. 이는 로봇이 실패에서 자동으로 추상화를 축적해 스케일 가능한 failure-aware planning으로 나아갈 수 있음을 실증적으로 보여준다.



### UBP2: Uncertainty-Balanced Preference Planning for Efficient Preference-based Reinforcement Learning (https://arxiv.org/abs/2606.19328)
- **Prior Approaches**: Preference-based RL은 페어(trajectory segment 쌍) 비교로 reward model을 학습해 수작업 reward 설계를 줄이지만, 대개 passive 데이터 수집에 의존해 초기 sample efficiency가 떨어진다는 한계가 있다. 또한 불확실성을 탐색 보너스로 쓰더라도 단일 모델 컴포넌트에만 집중하거나, 후보 쌍을 로컬 배치에서만 고르는 방식이어서 정보가 큰 비교를 놓칠 수 있다. 관련으로, pretraining된 reward/dynamics나 오프라인 데이터셋을 쓰는 접근도 있어 ‘완전 온라인 + 선호 피드백만’으로부터 reward를 처음부터 학습하는 설정은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 Uncertainty-Balanced Preference Planning (UBP2)로, reward·dynamics·value의 불확실성을 함께 추론하며 탐색을 능동적으로 설계하는 model-based preference RL을 제안한다. 특히 reward는 선호(비교)로만 학습하고, 이 learned reward와 world model을 이용해 MPC 기반으로 trajectory를 계획한 뒤, 선호 피드백 예산이 끝나면 학습된 정책으로 전환한다. UBP2는 exploit(예상 return 극대화)와 explore(모델의 epistemic uncertainty 극대화)를 하나의 trajectory-level 점수로 정리해 ad hoc 탐색 휴리스틱 없이 균형을 달성하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) reward가 관측되지 않고 비교만 주어질 때의 reward 학습 및 (2) 불확실성이 dynamics/value/reward 각각에서 다르게 의미를 갖는 상황에서, 계획이 실제로 정보 수집으로 이어지게 만드는 것이다. UBP2는 reward ensemble 불확실성은 Jensen–Rényi divergence(JRD)로 epistemic/aleatoric을 분리해 쓰고, dynamics/value는 ensemble disagreement로 epistemic 위주의 총 불확실성을 추정한다; 이후 reward·terminal value·불확실성을 함께 묶은 unified score로 후보 trajectory를 평가해 MPC가 정보를 많이 주는 구간을 선택하도록 유도한다. 또한 preference label 쿼리는 replay buffer 전역에서 전개되는 후보 쌍을 ‘낙관적(optimistic) scoring’ 기준으로 우선순위화해, 로컬 배치 탐색의 누락 위험을 줄인다.

- **Empirical Impact**: Meta-World 조작(Meta-World manipulation) 벤치마크에서 UBP2는 state-of-the-art model-free preference 기반 방법과 비낙관적(non-optimistic) model-based baseline보다 더 빠른 성과(더 이른 성공)와 더 높은 sample efficiency를 보였다고 보고한다. 이 실험 결과는 불확실성 기반 낙관적 planning이 preference 기반 학습 초기 단계에서 특히 유리하다는 주장을 뒷받침한다. 더불어 유한/무한 horizon 모두에 대해 상수 이상의 정보획득량에 명시적으로 의존하는 sublinear regret 보장을 제시해, 탐색-수익 균형이 이론적으로도 뒷받침된다는 점에서 분야에 의미가 있다.



### Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models (https://arxiv.org/abs/2606.19297)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 VLA 벤치마크는 LIBERO, CALVIN, RoboBenchMart 같은 조작 성공 중심 평가가 대부분이어서, 로봇 학습 후에도 commonsense·world knowledge가 남아 있는지 자체를 거의 측정하지 못합니다. 또한 VLM 쪽에서 VQA처럼 텍스트로 정답을 디코딩해 보는 방식은 지식 보유 여부만 간접 확인하며, 그 지식이 실제 행동 선택에 어떻게 쓰이는지는 드러내지 못합니다.

- **Core Contribution**: 이 논문은 Act2Answer라는 평가 프로토콜을 제안해, VLM 지식 벤치마크를 VLA용 “행동 기반 정답 선택”으로 변환합니다. 각 문항은 짧은 테이블탑 에피소드에서 에이전트가 단일 object-placement 행동으로 후보 정답 중 하나를 고르게 하여, 지식 결핍과 저수준 제어 실패를 더 분리해 관찰합니다.

- **Technical Challenges**: 핵심 난제는 ‘지식을 보유했는가’와 ‘행동으로 올바르게 사용했는가’를 한 점검에서 갈라내는 것인데, 이를 위해 모터 복잡도를 줄이고 단기·단순 선택 행동으로 성공을 정의합니다. 또 layerwise intent probing을 통해 중간 레이어에 남아 있는 정답 관련 신호가 행동 헤드로 갈수록 어떻게 감쇠되는지(표현은 있으나 행동 선택으로는 못 옮기는 병목)를 추적합니다.

- **Empirical Impact**: 대규모 실험에서 7개 VLA와 9개 VLM 베이스라인을 12개 지식 카테고리(총 1,720개 이진 문항)로 비교했으며, VLA는 Color/Shape 같은 단순 지각 개념에서는 강하지만 더 풍부한 의미 영역에서는 원천 VLM 대비 격차가 크게 나타납니다. 또한 VQA co-training이 knowledge retention과 연관되며, 정답 관련 신호는 VLA 내부에서 중간 레이어에 피크를 보이다가 상위 레이어로 갈수록 약해지는 경향이 관찰돼 “표현-행동 번역”의 병목을 시사합니다.



### CABLE: Cloud-Assisted Bandwidth-efficient LMM-based Encoding for V2X Systems (https://arxiv.org/abs/2606.19258)
- **Prior Approaches**: 기존 V2X 협력 인식은 원본 고해상도 영상을 그대로 전송하거나, 중간 특징을 압축해 통신을 줄이는 방식이 주류였습니다. 또한 ROI 선택을 위해 검출 기반(box/카테고리)이나 모션 기반(프레임 차분/광류)을 쓰는 연구도 있었지만, 검출 기반은 배경을 많이 포함하거나 open-vocabulary에 약하고, 모션 기반은 ego-motion에 쉽게 흔들려 오탐 ROI가 커질 수 있다는 한계가 있습니다.

- **Core Contribution**: CABLE은 클라우드의 cloud-hosted LMM(LISA++)이 생성한 분할 마스크를 다음 프레임 ROI 생성의 prior로 되돌려주는 mask-to-ROI-to-LMM feedback loop를 제안합니다. 엣지에서는 ego-motion 보정으로 이전 마스크를 전파하고 잔여 모션으로 보정한 뒤, corridor envelope로 ROI를 연속 영역으로 복원해 ROI-only 이미지 업로드를 가능하게 합니다. 이렇게 클라우드 LMM은 ROI 내부에 대해서만 분할을 수행해 visual token과 prefill 지연을 동시에 줄입니다.

- **Technical Challenges**: 핵심은 ‘의미 손실 없이 얼마나 적게 전송할지’와 ‘마스크 전파가 드리프트되면 어떻게 복구할지’입니다. CABLE은 (1) 순수 이동 가정의 homography로 기하 전파를 시작하되 residual-motion energy와 yaw-aware buffer dilation, (2) corridor envelope로 끊어진 ROI를 공간적으로 복원, (3) mask confidence가 낮아지면 full-frame refresh 키프레임으로 안정화하는 규칙을 함께 사용합니다. 그 결과 통신 절감과 함께 ROI 경계의 정보 손실을 제한하도록 설계되었습니다.

- **Empirical Impact**: nuScenes, WOD-ZB, Waymo, KITTI, CADC 5개 데이터셋에서 ROI pixel-coverage를 73–87% 줄이면서도 인식 품질을 크게 보존했고, LMM prefill은 추정 기준 5–8× 가속을 달성했습니다. 특히 open-loop(클라우드 첫 프레임만 분할 후 전파) 대비 feedback이 탐지 유지율을 크게 개선했으며, corridor envelope 없이 ROI가 분절되면 detection retention이 급락하는 등 공간 문맥 보존의 중요성이 확인됐습니다. 전체적으로는 소폭(제한적) 탐지 품질 트레이드오프를 전제로, bandwidth와 클라우드 지연을 동시에 줄이는 V2X LMM 배치 전략의 실증 가능성을 보여줍니다.



### OneCanvas: 3D Scene Understanding via Panoramic Reprojection (https://arxiv.org/abs/2606.19253)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 Vision-Language Model(VLM) 기반 3D 이해는 (1) 점구름 토큰화/geometry encoder 등 복잡한 모듈을 추가하거나, (2) distance·direction 같은 공간 QA를 대규모로 수집해 학습량을 키우는 방식이 주류였다. 그런데 최근 감사(audit) 결과, 텍스트만으로도 기준 성능에 근접하거나 나오는 경우가 있어 모델이 제공된 기하 입력을 ‘진짜로’ 읽기보다 장면/질문 통계의 지름길을 탄다는 한계가 지적됐다. 또한 대부분의 방식은 프레임을 조각조각 보며 공간관계를 일관된 좌표계로 결합하지 못하는 문제가 남아 있다.

- **Core Contribution**: OneCanvas는 여러 시점의 패치 특징을 depth와 카메라 포즈로 3D로 ‘리프트(lift)’한 뒤, 하나의 equirectangular 파노라마 캔버스에 longitude·latitude 좌표로 재투영해 VLM이 이미지처럼 그대로 읽게 만든다. 이때 캔버스의 원점(origin)과 방향을 태스크에 맞게 자유롭게 선택할 수 있어, 로보틱스/embodied AI에서 중요한 ‘지정된 시점에서의 situated reasoning’을 같은 표현으로 바로 지원한다. 나아가 캔버스 위에 실제 이미지에서 뽑은 객체 패치 특징을 임의의 3D 위치에 절차적으로 배치해, 답 분포를 제어하면서 공간 사전학습 커리큘럼을 on-the-fly로 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (a) 겹치는 관측에서 픽셀 단위 rasterization으로 정보를 뭉개지 않으면서, (b) VLM의 기존 3D-RoPE 의미를 깨지 않고 metric(미터) 좌표 정보를 주입하는 것이다. OneCanvas는 rasterization 대신 각 리프트 패치를 연속적인 (ϕ,θ) 위치를 갖는 개별 토큰으로 유지해, 중첩/가림 상황도 attention으로 구분하게 했다. 또한 각 패치의 캔버스 좌표 오프셋(qx,qy,qz)을 다축·방사(radial) 성분의 sin/cos 포지셔널 인코딩과 함께 특징 공간에 게이트된 방식으로 주입해, 각도 캔버스로 평탄화되며 사라지는 depth/거리 단서를 복원한다.

- **Empirical Impact**: 실험에서 OneCanvas는 SQA3D와 VSI-Bench에서 최신 성능을 달성하고, SPBench에서는 zero-shot에서도 우수한 일반화를 보이며 계산량은 경쟁 최고 대비 한 자릿수(대략 10배) 이하 수준으로 줄였다. 특히 SQA3D에서 ‘Which’처럼 시점 의존성이 큰 질문 유형에서 캔버스 원점 중심화(ablation)의 영향이 크게 나타나, 표현이 실제 situated geometry를 반영함을 시사한다. 즉, 아키텍처/부가 모듈 없이 입력 표현과 커리큘럼만으로 3D 추론을 강화할 수 있다는 실증적 근거를 제공해 해당 분야의 설계 관점을 바꾸는 의미가 있다.



### Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation (https://arxiv.org/abs/2606.18960)
- **Prior Approaches**: Action-conditioned world models는 로봇의 행동을 조건으로 영상 롤아웃을 생성해, 비용이 큰 실세계 실험을 줄이려는 접근이다. 하지만 manipulation에서는 wrist-camera의 빠른 운동과 end-effector occlusion 때문에 현재 관측만으로는 미래의 시야를 예측하기 어려워 이전 프레임의 장면을 잊거나 hallucination이 생긴다. 기존 메모리 검색은 joint-pose similarity나 단순 FOV overlap, 고정 stride 기반 컨텍스트 확대로는 가시성 제약과 조작 특화한 정보량을 제대로 반영하지 못했다.

- **Core Contribution**: 이 논문은 manipulation에서 장기 일관성을 유지하는 memory-augmented multi-view action-conditioned world model Mem-World를 제안한다. 핵심은 W-VMem으로, 4D wrist-view-centered surfel-indexed memory를 도입해 과거 관측을 ‘언제/어떤 표면 요소’를 봤는지에 고정(anchor)함으로써 geometry-aware history retrieval을 가능하게 한다. 이를 통해 미래 행동이 주어졌을 때, 관련성이 높고 중복이 적은 과거 wrist-view 프레임을 선별해 예측에 활용한다.

- **Technical Challenges**: 기여를 위해서는 (1) 행동으로부터 미래 wrist-camera pose를 계산해 surfel rendering에 쓸 수 있어야 하고, (2) 로봇과 물체가 움직이는 동적 조작 장면에서 surfel을 시간적으로 정의해야 하며, (3) multi-view에서 메모리 초기화·업데이트·리딩의 일관성을 유지해야 한다는 문제가 있다. 논문은 고정된 wrist-엔드이펙터 기구학 변환과 forward kinematics로 미래 wrist-camera pose를 얻고, surfel에 생성/업데이트 timestep과 task-relevance(조작 대상 여부) 플래그를 포함하는 4D surfel 정의를 설계한다. 또한 업데이트는 wrist-view 관측만으로 수행해 temporal association을 보존하고, 미래의 평균 관측 방향에서 surfel을 렌더링해 가시성·과업 관련성·시간 근접성 기반 점수 및 NMS로 top-K 컨텍스트를 선택한다.

- **Empirical Impact**: 실험에서 Mem-World는 end-effector occlusion이 잦고 wrist-camera 운동이 큰 장기 manipulation 시나리오에서 더 persistent하고 시간적으로 일관된 롤아웃을 보였다고 보고한다. 특히 Ctrl-World 대비 실세계 정책 성능과의 Pearson 상관을 14.5% 개선했으며(설정된 5개 task에서 높은 선형 대응), Mem-World로 만든 synthetic data로 post-training을 하면 장기 태스크 success rate가 58%에서 72%로 상승했다. 이는 메모리-증강 world modeling이 단순 예측을 넘어 정책 평가/학습의 ‘데이터 엔진’으로 신뢰도를 높일 수 있음을 시사한다.



### Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos (https://arxiv.org/abs/2606.18955)
Comments:
          Accepted to IROS 2026

- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 사전학습된 Vision-Language Model(VLM) 백본 위에 action head를 붙이고, 대규모 로봇 데이터(정확한 action annotation)로 fine-tuning해 성능을 끌어올리는 흐름이 주류였습니다. 다만 Open X-Embodiment나 AgiBot처럼 플랫폼별로 필요한 데이터 수집 비용과 기구학/물리 차이로 인한 domain gap이 커서 확장성이 떨어집니다. 한편 사람 egocentric 비디오를 활용하려는 시도는 존재하지만, 대부분 AR/VR 등 특수 장치로 hand pose 같은 명시 라벨을 얻어야 하거나(데이터 라벨 병목) 배경·카메라 변화가 섞인 latent action이 잡음으로 작동하는 문제가 남아 있었습니다.

- **Core Contribution**: 이 논문은 action 라벨이 없는 사람 egocentric 비디오만으로 cross-embodiment action prior를 학습하는 latent-action 기반 사전학습 패러다임을 제안합니다. 핵심은 Hybrid Disentangled VQ-VAE가 motion dynamics와 environmental background를 physical mask로 분리해 “몸체에 덜 의존적인” action codebook을 만들고, 이를 VLM의 행동 어휘로 distill해 action intent를 먼저 학습시키는 것입니다. 또한 적응 단계에서는 intention-perception decoupling을 통해 VLM의 intent와 state 피드백을 분리해 action hallucination을 줄이며, 다운스트림은 약 50 trajectories 수준의 소량 데모로도 경쟁력을 확보합니다.

- **Technical Challenges**: 가장 어려운 점은 라벨이 없는 사람 비디오에서 ‘진짜 조작 동학’만 이산 action token으로 안정적으로 뽑아내는 disentanglement입니다. 논문은 DINO v2 고정 특징과 하이브리드 disentangled VQ-VAE(dual vector quantization + mask-guided decoder)로 motion은 foreground에서만 재구성 오차를 강제하고, background는 별도 경로로 학습시켜 코드북이 작업 무관 변화에 덜 흔들리게 했습니다. 다음 난제는 로봇 적응 시 실시간 관측과 intent가 충돌해 생기는 hallucination을 막는 것인데, VLM이 intent를 담당하고 별도 frozen visual encoder(DINO v2)가 state-specific 특징을 제공하는 decoupling으로 제어 안정성을 높였습니다.

- **Empirical Impact**: 실험은 시뮬레이션과 실세계 모두에서 검증되며, 사전학습은 unlabeled human video만 사용하면서도 LIBERO 및 dual-arm RoboTwin 2.0/ARX 플랫폼에서 SOTA VLA와 유사하거나 우수한 성공률을 보였습니다. 특히 long-horizon/Goal 계열에서 villa-x 대비 성능 격차가 크게 나타나, 추출된 latent action intent가 multi-step 계획을 더 잘 안내한다는 해석이 가능합니다. 정량 분석에서는 domain bias를 제거한 뒤 Centered Kernel Alignment( CKA ) 일치도가 UniVLA보다 더 높게 나와, “운동 중심” 코드북이 embodiment/환경 차이를 억제한다는 점을 representation 수준에서 뒷받침했습니다.



### Stealthy World Model Manipulation via Data Poisoning (https://arxiv.org/abs/2606.18697)
Comments:
          41 pages, 8 figures, 11 tables. Submitted to NeurIPS 2026

- **Prior Approaches**: 기존 world model 기반 model-based learning은 데이터로 학습한 dynamics를 예측·계획에 활용하지만, fine-tuning 과정에서 업데이트가 악용될 수 있다는 보안 연구는 상대적으로 부족했습니다. supervised learning의 전형적 데이터 포이즈닝은 라벨이 뒤집히는 등 구조가 단순하지만, world model 포이즈닝은 next-state 예측 타깃(고차원·연속·(비)결정적)을 건드려 장기 roll-out까지 누적 영향을 주기 때문에 그대로 적용하기 어렵습니다. RL 포이즈닝도 주로 reward나 정책을 직접 겨냥해 성능 저하를 만들지만, world model을 통한 downstream planner를 간접적으로 망가뜨리는 문제는 별도 설계가 필요합니다.

- **Core Contribution**: 이 논문은 learned world model의 dynamics 자체를 노리는 최초의 two-stage data poisoning 프레임워크 SWAAP(Stealthy World Model MAnipulation via DAta Poisoning)를 제안합니다. 1단계에서는 planning 시 저수익 행동을 유도하면서도 clean dynamics와 가깝게 유지되는 “해로운 타깃 world model”을 bilevel 최적화로 찾고, 2단계에서는 소수의 fine-tuning 전이(next-state target)를 stealth-constrained gradient matching으로 조작해 그 타깃을 실제 fine-tuning에 구현합니다. 결과적으로 공격자는 deployed 모델 파라미터를 직접 바꾸지 않고도, 업데이트된 world model이 잘못된 계획을 하도록 유도합니다.

- **Technical Challenges**: 가장 큰 난관은 fine-tuning 후 victim 모델이 어떻게 변하는지(훈련 과정의 암묵적 반응)를 직접 역전파로 최적화하기 어렵다는 점이며, 또한 long-horizon planning의 영향이 transition 타깃 변경과 강하게 결합된다는 점입니다. SWAAP은 이를 위해(1) transition-gradient theorem 기반 추정으로 model-space bilevel을 1차(first-order) dynamic-barrier 방식으로 분해해 타깃 world model을 찾고, (2) fine-tuning 시에는 gradient matching에서 방향 정렬(cosine alignment)과 prediction-error regularizer를 함께 써서 “해로운 타깃으로의 유도”와 “검출 회피”를 동시에 맞춥니다. 추가로 데이터가 trajectory 형태일 때 생길 수 있는 불일치를 줄이기 위해 trajectory-consistent 변형도 도입합니다.

- **Empirical Impact**: TD-MPC2, DINO-WM을 포함한 연속 제어 벤치마크들(DMControl, MyoSuite, MetaWorld 등)에서 SWAAP은 fine-tuning 데이터의 작은 비율만 조작해도 downstream planning 성능을 크게 떨어뜨립니다. 동시에 공격 전이의 residual 기반 사전 탐지, TRIM 같은 training-time 강건화, 그리고 배포 후 model-level deviation/CUSUM/TRIM-style 모니터링 등 평가된 방어 체계에서는 높은 stealth성을 유지합니다. 저자들은 이 결과가 world-model 적응 파이프라인이 practical vulnerability를 가진다는 점을 보여주며, world model의 학습 데이터와 dynamics 자체를 함께 보호하는 robustness 연구의 필요성을 강조합니다.



### Spatially Stratified Distillation for Heterogeneous Radar Place Recognition (https://arxiv.org/abs/2606.18687)
Comments:
          IEEE ICRA Workshop on Open Challenges for Rigorous Robot Perception 2026

- **Prior Approaches**: 기존 heterogeneous radar place recognition은 비싼 360∘ 스핀 레이더의 조밀한 맵과, 4D 고체(120∘×120∘) 쿼리를 매칭해야 한다는 점에서 modality asymmetry가 핵심 병목이다. SHeRLoc 같은 방식은 polar BEV로 공통 표현공간을 만든 뒤 shared backbone 기반으로 정렬하지만, multi-session 환경에서는 4D 쿼리가 관측하지 못하는 조밀한 구조를 적극적으로 활용하지 못해 성능이 흔들린다. 특히 밀도 차이가 큰 영역에서는 정확한 feature alignment 자체가 물리적으로 불가능해 기준 정렬이 오히려 약점이 된다.

- **Core Contribution**: 논문은 Spatially-Stratified Distillation(SSD)를 제안해, 서로 관측 가능한 영역과 불가능한 영역을 물리 FOV 기하로 나눠 distillation을 재설계한다. SSD는 관측이 겹치는 joint 관측 영역에서는 강하게 feature를 정렬하고, 4D student가 비어 있지만 teacher만 구조가 있는 gap 영역은 가중치를 크게 낮춰 “억지 정합”을 피하면서도 약한 구조 prior를 주입한다. 이를 통해 4D 표현을 조밀한 맵의 유효한 맥락으로 끌어올리되, student가 보지 못한 것을 학습으로 “환각”하지 않도록 한다.

- **Technical Challenges**: 문제의 기술적 난점은 teacher-only 영역을 그대로 맞추면 물리적으로 불가능하다는 점과, 반대로 전부 마스킹하면 dense structural supervision의 이점을 잃는다는 점이다. SSD는 이를 해결하기 위해 teacher feature 크기 기반 magnitude mask와 student 입력 기반 gaussian-smoothed FOV 마스크를 결합해 joint/gap을 분리하고, gap 영역은 heavily discounted distillation weights로 약하게 정규화한다. 또한 1×1 convolution으로 채널 basis를 맞추고 L2-normalization 후 cosine distance와 covariance alignment로 활성 스케일·상관 불일치를 동시에 줄인다.

- **Empirical Impact**: HeRCULES 벤치마크에서 SSD는 single-session과 multi-session 모두에서 기존 최고 성능 방법을 일관되게 능가하며, 특히 dynamic multi-session 시나리오에서 유의미한 마진을 보인다. 논문이 보고한 결과는 HeRCULES의 어려운 dynamic sequences에서 state-of-the-art를 달성했음을 시사한다. 더불어 SC/01→SC/03 같은 난구간에서도 SHeRLoc 대비 더 많은 쿼리를 성공적으로 복구하며, 최악 실패에서도 오차가 크게 줄어 4D 레이더 표현 학습의 실질적 개선 효과가 확인된다.



### Aerial-ground LiDAR place recognition with patch-level self-supervised learning and expanded reciprocal re-ranking (https://arxiv.org/abs/2606.18583)
- **Prior Approaches**: 기존 ground-level LiDAR place recognition은 사전 방문(pre-visit)과 지도 범위의 불완전성, 시점 다양성 부족 때문에 확장성에 한계가 있었다. 또한 patch 단위 표현 학습을 충분히 활용하지 않고 scene-level metric learning 중심으로 학습하는 접근이 많아, 항공-지상 cross-view에서 발생하는 domain gap과 초기 검색의 false positives에 취약했다. 재랭킹 단계 역시 feature 거리만으로 처리하는 경우가 많아 ALS의 공간적 구조를 제대로 쓰지 못했다.

- **Core Contribution**: 논문은 항공의 full-coverage Airborne Laser Scanning(ALS) 지도를 aerial prior map으로 두고, 지상 쿼리를 항공 서브맵으로 매칭하는 cross-view LiDAR place recognition 프레임워크를 제안한다. 핵심은 patch-level self-supervised learning(다중 스케일)으로 항공/지상 간 global feature의 판별성을 키우는 retrieval 네트워크와, 추가 학습 없이도 neighborhood 정보를 극대화해 false positives를 줄이는 Expanded Reciprocal(ER) re-ranking이다.

- **Technical Challenges**: 가장 큰 기술 난제는 항공-지상 점군의 domain gap으로 인해 정확한 point correspondence가 부족하고 글로벌 특징이 쉽게 흔들린다는 점이다. 이를 위해 OctFormer 기반 백본에서 여러 깊이(octree scale)의 patch를 뽑고, 인접 패치의 유사 의미를 가정해 teacher-student cross-attention self-distillation으로 patch 표현을 정렬한다. 두 번째 난제인 초기 검색의 반복 구조로 인한 false positives는 ALS 점군의 구조적 공간 분포를 이용해 reciprocal 이웃을 확장하고, 이웃 평균으로 특징을 갱신한 뒤 최종 거리 행렬을 업데이트하는 방식(ER)으로 완화한다.

- **Empirical Impact**: CS-Urban-Scenes에서 retrieval 네트워크는 평균 Recall@1을 9.8% 개선했고 평균 Recall@1%도 3.2% 향상시켰으며 CS-Campus3D에서도 최고 성능을 보였다. 더 나아가 ER re-ranking은 CS-Campus3D에서 평균 Recall@1을 4.9% 추가로, CS-Urban-Scenes에서는 10.2% 추가로 끌어올리되 추가 학습 없이 적용 가능함을 보여준다. 결과적으로 항공-지상 cross-view LiDAR 기반 대규모 모바일 매핑/자율주행 위치추정에서 “학습 기반 검색 + 학습 없는 구조 기반 재랭킹” 조합의 실용성을 강화한 것으로 평가된다.



### Technical Report for ICRA 2026 GOOSE 2D Fine-Grained Semantic Segmentation Challenge: Leveraging DINOv3 for Robust Outdoor Scene Understanding in Field Robotics (https://arxiv.org/abs/2606.18582)
Comments:
          5 pages, 4 figures

- **Prior Approaches**: 기존 필드 로보틱스용 세그멘테이션 벤치마크는 상대적으로 정형화된 실내·도시 장면에 치우쳐, 비정형 오프로드와 플랫폼/시점 변화(차량·굴착기·사족 등), 장거리 스케일 변동, 클래스 장기꼬리 문제를 충분히 반영하기 어렵습니다. 또한 Cityscapes/ADE20K류 일반 장면 파싱 모델을 그대로 옮기면 64개 파인 클래스의 시각적 애매함과 희소성(희귀 라벨은 픽셀 단위로 극히 적음)에서 성능이 흔들리기 쉽습니다.

- **Core Contribution**: 논문은 GOOSE 2D Fine-Grained Semantic Segmentation Challenge에서 1위를 달성한 최초의 정밀 오프로드 2D 세그멘테이션 솔루션을 제시합니다. 핵심은 DINOv3 ViT-L/16 백본+ViT-Adapter+Mask2Former의 조합에 더해, DINOv3의 global [CLS] 토큰에 11개 coarse 카테고리 존재를 예측하는 multi-hot 보조 손실을 붙여 전역 의미 인식을 강화한 점입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 64개 파인 클래스의 장기꼬리·국소 애매함과, 서로 다른 로봇 플랫폼이 만드는 카메라 기하/시점 차이를 동시에 커버하는 것입니다. 저자들은 단일 해상도 ViT 토큰을 Mask2Former가 요구하는 멀티스케일 피처 피라미드로 변환하는 ViT-Adapter(Deformable cross-attention 기반)로 아키텍처 간 불일치를 해결하고, 보조 손실은 픽셀 위치가 아니라 coarse 카테고리 “존재”만 학습하도록 설계해 주 손실을 흔들지 않게 했습니다.

- **Empirical Impact**: 실험 결과 dev split에서 ConvNeXt+Mask2Former 기준 56.68% 대비 DINOv3+ViT-Adapter로 69.80%까지 크게 상승했으며, [CLS] multi-hot 보조 손실과 TTA/체크포인트 앙상블을 누적해 최종 composite mIoU 76.57%(fine-class mIoU 69.32%, category-level mIoU 83.81%)를 기록했습니다. 이 성과는 파인-그레인드 라벨 희소성과 플랫폼 변동이 큰 필드 로보틱스 세그멘테이션에서 전역 의미 정규화와 멀티스케일 쿼리 기반 디코딩이 실질적 개선으로 이어질 수 있음을 보여주는 사례로 평가됩니다.



### AI Sandboxes: A Threat Model, Taxonomy, and Measurement Framework (https://arxiv.org/abs/2606.18532)
Comments:
          50 pages, 8 figures, 10 tables

- **Prior Approaches**: 기존 연구는 벤치마크, 시뮬레이터, 디지털 트윈, cyber range, 규제 샌드박스처럼 “평가를 위한 도구”를 각기 다른 관점에서 발전시켜 왔습니다. 하지만 이런 환경들이 만들어내는 증거를 어떤 배포(Deployment) 주장으로까지 정당화할지에 대한 공통된 보증(assurance) 규칙이 부족해, 고평가(over-reading) 위험이 남습니다. 또한 보안 연구는 공격 표면을 주로 AI 모델 중심으로 다루거나, 평가 인프라(검증·증거 생산 과정) 자체를 대상으로 한 위협은 상대적으로 덜 체계화돼 있었습니다.

- **Core Contribution**: 이 논문은 AI sandboxes를 “경계가 명시된 증거 기반 보증 절차”로 재정의하고, 샌드박스 경계(BB)와 증거 조합 규칙을 formalize합니다. 특히 각 차원(충실도, 통제가능성, 관측가능성, 격리/containment, 재현가능성, 거버넌스 아티팩트 등)의 증거를 모아 배포 주장(DD)을 정당화할 때는 weakest-link rule로 가장 약한 차원의 한도만 허용된다고 못 박습니다. 더불어 물리-사이버-거버넌스가 얽힌 상황에서, 평가 장치와 증거 체인 자체를 노리는 위협까지 포함하는 cyber-physical threat model을 제안합니다.

- **Technical Challenges**: 핵심 난제는 샌드박스가 “고립돼 있다”는 사실만으로는 배포 안전/보안/규제 적합성을 증명할 수 없다는 점을, claim-relative evidence로 강제하는 것입니다. 논문은 신뢰 경계(무엇이 표현되고 무엇이 생략되는지), 개입(중단/롤백/kill switch 등), 모니터링(로그·텔레메트리·증거 아티팩트), 잔여 위험(RR)을 튜플로 모델링해, 암묵적 가정을 문서화·감사 가능하게 만듭니다. 동시에 물리 AI/AIoT/CPS에서 실패가 물리 동역학과 실시간 제약, 센서·액추에이터 한계, 네트워크 열화, 공격-재현까지 연결되므로, 이를 커버할 측정 프레임워크(15개 차원)를 설계해 도구 간 비교 가능성을 확보했습니다.

- **Empirical Impact**: 논문은 제안된 threat model, 분류(taxonomy), 측정 프레임워크를 실제 샌드박스 3개 사례(working case studies)에 적용해 어떤 주장까지 유효한지 구체적으로 보여줍니다. 특히 closed-loop 시뮬레이션 증거를 과도하게 일반화하던 관행을 다시 점검하며, 어떤 차원(예: 관측가능성·containment·이식성)에서 증거가 부족하면 배포 주장으로 확장될 수 없음을 강조합니다. 결과적으로 물리-사이버-거버넌스 영역에서 “무엇을 테스트했다”를 넘어 “무엇을 보증할 수 있는지”를 규정하는 기준틀을 제공해, 안전·보안·규제(TEVV, AI Act 등) 논의의 공학적 정합성을 높일 것으로 기대됩니다.



### RegimeVGGT: Layer-Wise Spatially Preserving Redundancy Removal for Visual Geometry Grounded Transformer (https://arxiv.org/abs/2606.18439)
Comments:
          9 pages, 3 figures, 7 tables. Jinhao You, Shuo Lyu, Zhuohang Lyu, Tanxuan Li, and Zibo Zhao contributed equally. Shuo Lyu is the corresponding author

- **Prior Approaches**: VGGT는 다중 뷰 이미지를 단일 forward pass로 3D와 카메라 포즈를 복원하지만, 프레임 간 global cross-frame attention이 O(S^2P^2)로 커져 장기 시퀀스에서 메모리와 속도 병목이 발생한다. FastVGGT, S-VGGT 같은 학습 없이 가속 기법은 한 축(토큰 수/프레임 분할 등)을 균일하게 줄이거나 단순 격자로 K/V를 다운샘플해, VGGT 내부 레이어의 이질성과 ‘포즈에 필요한 경로’를 놓친 한계가 있다. AVGGT는 레이어별 중복성이 있음을 보였지만, 두 축 압축 설계를 함께 최적화하는 질문은 남아 있었다.

- **Core Contribution**: RegimeVGGT는 VGGT의 24개 aggregator 레이어를 스펙트럴·프로빙·인과 분석으로 얕은/중간/깊은 3개 ‘레짐(regime)’으로 나누고, 각 레짐의 기능이 서로 다르다는 점을 압축 설계로 연결한다. 특히 포즈는 camera/register 토큰과 주변 패치 사이 cross-frame attention 경로에 걸려 있으므로, 기하 복원에 덜 필요해 보이는 패치라 해도 이 경로만큼은 유지해야 한다는 전제를 도입한다. 이 구조를 반영해 Saliency-Guided Banded Merging(토큰 축)과 Selectively Protected K/V Downsampling(K/V 축)을 레이어별 U자형 압축으로 결합한다.

- **Technical Challenges**: 핵심 난제는 ‘기하(3D dense geometry)’와 ‘포즈(camera pose)’가 같은 attention 요소를 공유하지 않을 수 있는데, 학습 없이 이를 동시에 보존해야 한다는 점이다. 논문은 먼저 레이어 레짐마다 cross-view 구조의 존재 여부와 정보 활용이 달라짐을 보이고, 중간 레짐(L11–L18)이 대부분의 포인트클라우드 복원을 담당하되 얕은/깊은 레짐은 깊은 기하 신호가 중복되거나 작동하지 않는다고 정리한다. 이후 토큰 축에서는 DINOv2 CLS saliency로 geometry·edge-salient 토큰을 보호하면서 ToMe 계열 merge-unmerge를 레짐별 비율로 수행하고, K/V 축에서는 phase-shifted spatial sub-grid, frame-0 reference anchor, uncompressed camera/register 토큰을 통해 포즈-critical path를 유지하는 방식으로 해결한다.

- **Empirical Impact**: ScanNet-1000 기준으로 RegimeVGGT는 VGGT* 대비 6.7배 속도를 달성하면서 재구성 품질을 거의 유지한다. 장기 시퀀스(예: 1000 frames)에서 VGGT는 OOM이 발생하고 S-VGGT도 실패하는 반면, RegimeVGGT는 실행 시간을 크게 줄이며 Chamfer Distance가 크게 악화되지 않았다. 포즈 추정에서도 Tanks & Temples, DTU, ScanNet-50 장기 설정에서 AUC/ATE/ARE/RPE 지표가 competitive하거나 더 좋은 결과를 보여, cross-frame correspondence 보존이 실제로 확인됐다는 점에서 의미가 크다.



### WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation (https://arxiv.org/abs/2606.13672)
- **Prior Approaches**: 기존 로봇용 world model(WM)은 (1) 높은 fidelity, (2) 장기 horizon에서의 temporal consistency, (3) 실시간에 가까운 efficiency를 동시에 만족시키기 어려웠습니다. 예를 들어 비디오 생성 계열은 fidelity는 높지만 느리고, Dreamer류와 JEPA류는 학습·표현 방식 때문에 임의 visuomotor policy 평가나 out-of-distribution 강건성이 흔들릴 수 있습니다. 또한 Ctrl-World 같은 조작 특화 WM은 일관성과 정확성을 어느 정도 얻더라도 속도 제약이 커 test-time planning이나 정책 개선에 부담이 됐습니다.

- **Core Contribution**: 이 논문은 WEAVER(World Estimation Across Views for Embodied Reasoning)라는 멀티뷰 WM 아키텍처를 제안하며, fidelity·consistency·efficiency를 한 번에 겨냥합니다. 핵심 학습 목표로는 미래 latent와 reward 값을 flow-matching 기반 loss로 예측하고, reward head를 통해 외부 VLM judge 없이도 평가·계획용 스코어링을 빠르게 수행하도록 설계했습니다. 여기에 multi-view 예측과 sparse memory/history를 결합해 가림(occlusion)과 접촉(변형체 조작) 같은 조작 특유의 복잡성을 완화합니다.

- **Technical Challenges**: 조작 작업에서는 시점이 여러 카메라로 분산되고, 가림으로 인해 관측이 순간적으로 끊기며, 특히 변형체 접촉은 로봇 상태(관절/그리퍼 구성) 정보가 정확해야 합니다. WEAVER는 (i) 다중 뷰 외부·손목 관측을 함께 예측하고, (ii) latent dynamics에 sparse memory와 짧은 history를 함께 조건으로 넣어 장기 일관성을 확보하며, (iii) flow matching + diffusion forcing + SPRINT 토큰 드롭으로 장기 예측과 추론 속도를 동시에 끌어올립니다. 또한 rectified flow objective로 post-train을 수행해 소수의 forward pass로도 고품질 생성이 가능하도록 했고, latent reward/critic을 학습해 평가·계획의 병목(디코딩/외부 judge)을 줄였습니다.

- **Empirical Impact**: 실제 로봇 하드웨어에서 WEAVER는 policy evaluation에서 실세계 성공률과의 상관 ρ=0.870을 보였고, real-world 상 success rate를 38% (π0.5의 top-up) 개선하며 외부 상호작용 없이도 성능 향상을 입증했습니다. test-time planning에서는 prior WM 대비 5–10× 속도 향상과 함께 real-world success rate 14% 개선을 달성해, WM의 실용적 활용 가능성을 구체화했습니다. 더불어 out-of-distribution 시나리오에서도 기존 WM보다 좋은 결과를 보이며, 조작 분야의 long-horizon dynamic manipulation에 대한 world modeling 난제를 한 단계 진전시켰다는 점에서 의미가 큽니다.



New uploads on arXiv(cs.MA)

### Data Intelligence Agents: Interpreting, Modeling, and Querying Enterprise Data via Autonomous Coding Agents (https://arxiv.org/abs/2606.19319)
- **Prior Approaches**: 기존 text-to-SQL 계열은 한 단계(질의 생성, 디버깅 등)에 집중하거나 파이프라인을 여러 모듈로 쪼개 정밀 튜닝/재설정을 반복하는 경우가 많다. RL 기반·특화 에이전트는 벤치마크/다이얼렉트에 강하게 고정되기 쉬우며, 에이전틱 접근도 세션 간 기억을 일관되게 공유하지 못해 매 질의마다 시작점이 흔들린다. 또한 많은 시스템이 텍스트(쿼리/비평)를 출력해 엔터프라이즈 작업에서 필요한 ‘실행 가능한 아티팩트’와 ‘상위 단계(이해·스키마 구성)’를 함께 닫지 못한다.

- **Core Contribution**: DIA(Data Intelligence Agents)는 ACA(코드 실행 가능한 에이전트)를 핵심 추상으로 두고 Data Interpreter, Schema Creator, Query Generator의 세 에이전트를 단일 워크스페이스 위에서 동작시킨다. LLM이 텍스트를 내놓는 대신, 에이전트가 생성·실행·검증·수정을 통해 실행 가능한 산출물을 직접 만들고 도메인 전문가가 검토할 수 있게 한다는 점이 핵심이다. 특히 Query Generator는 SQL 생성뿐 아니라 디버깅·대화·프로젝트 완료까지 단일 일반 에이전트로 처리하며, 적응은 자연어 지시사항 범위로 제한한다.

- **Technical Challenges**: 이 접근을 가능하게 하려면 (1) 원천 데이터에서 의미 있는 스키마를 자동으로 구성하고, (2) 생성된 SQL이 실제 데이터 조건과 일치하는지 실행 기반으로 스스로 검증하며, (3) 세션 간 경험을 ‘텍스트 요약’이 아니라 실행된 아티팩트/규칙 형태로 안전하게 재사용해야 한다. DIA는 작업 전 과정에서 공유 워크스페이스(WW)에 파일/아티팩트를 누적하고, 실행 추적과 결과를 기반으로 메모리(MM)에서 관련 경험만 pull 방식으로 가져와 사전조건을 라이브 프로브로 확인한다. Query Generator는 쿼리 초안을 즉시 정답으로 간주하지 않고 결과가 기대한 shape(열/행 단위, 정렬, 필터 등)와 일치하는지 확인한 뒤, 어긋나면 같은 패스에서 수정·재실행해 정합성을 맞춘다.

- **Empirical Impact**: 논문은 Query Generator를 7개 SQL 벤치마크(4개 태스크 범주, 4개 다이얼렉트)에서 완전 자율 모드로 평가해, 모든 벤치마크에서 기존 최강 결과를 ‘동일 모델·추가 fine-tuning 없이’ 매치하거나 능가함을 보인다. 특히 대화형(BIRD-Interact), 디버깅(BIRD-Critic), 수정 작업(LiveSQLBench)에서 큰 폭의 개선이 나타나며, 단일 에이전트가 여러 작업 범주와 다이얼렉트를 아우르는 일반화를 실증한다. 엔터프라이즈 데이터 지능 워크플로우를 실행 가능한 아티팩트 중심으로 재설계한 시스템 관점이어서, 다음 세대 text-to-SQL을 넘어 데이터 이해·스키마 구성·쿼리까지 한 루프로 묶는 방향에 의미 있는 영향을 줄 전망이다.



### A Technical Taxonomy of LLM Agent Communication Protocols (https://arxiv.org/abs/2606.19135)
- **Prior Approaches**: 기존 LLM 에이전트 통신은 MCP처럼 에이전트-컨텍스트(tool·데이터) 연계를 다루거나, 일부 에이전트-에이전트 프로토콜이 있어도 서로 교차 운용이 어렵다는 문제가 누적돼 왔습니다. 관련 연구들은 보안 위협 분류, 인터넷 아키텍처 관점 설계원칙, 혹은 소수 프로토콜의 비교 같은 접근을 제시했지만, 실제 프로토콜들을 일관된 추상 구조로 계통적으로 분류하기엔 차원이 부족했습니다. 특히 ‘표준화가 왜 필요한가’는 강조되지만, 어떤 설계 특성이 실제 채택과 연결되는지 추적 가능한 분류 체계는 부족했습니다.

- **Core Contribution**: 이 논문은 채택 가능한 9개 오픈소스 LLM 에이전트 통신 프로토콜을 대상으로, 통신 프로토콜을 이해·비교·추적할 수 있는 기술 분류체계(taxonomy)를 제안합니다. 반복적 구축 절차(Nickerson et al. 방식)를 따라 메타-특성, 종료 조건을 명시하고, 5차례 반복(경험→개념 3회, 개념→경험 2회)으로 분류 축을 확정했습니다. 결과적으로 프로토콜을 counterparty(상대방), payload(페이로드), interaction state(세션/상호작용 상태), discovery mechanism(발견 메커니즘), schema flexibility(스키마 유연성) 5차원으로 정리합니다.

- **Technical Challenges**: 핵심 기술 난제는 ‘서로 다른 구현을 공통 추상 축으로 매끄럽게 정렬’하는 것입니다. 논문은 프로토콜마다 메시지 형태·세션 지속 여부·런타임 스키마 처리·서비스/에이전트 발견 방식이 달라 동일 축에서 분리 가능해야 한다는 조건을 반복적으로 점검하며, 상호배타적·전체포괄적 특성 설계를 목표로 했습니다. 또한 분석 결과, 다수 에이전트-에이전트 프로토콜이 hybrid payload와 session-state persistence를 함께 쓰고, 미리 정의된 스키마 지원이 많되 일부는 runtime 협상으로 schema flexibility 추세가 보인다는 패턴을 통해 분류 축의 타당성을 보강합니다.

- **Empirical Impact**: 9개 프로토콜의 실제 분류 결과는 단기적으로 에이전트-에이전트와 에이전트-컨텍스트(tool·데이터) 통신을 한데 묶는 방향으로의 수렴 압력이 관찰된다는 결론으로 이어집니다. 반면 장기적으로는 단일 프로토콜이 versatility, 효율성, portability를 동시에 최대로 만족시키기 어렵고, federated·layered 프로토콜 스택으로 발전할 가능성을 제시합니다. 더불어 프로토콜 선택을 돕는 프레임으로 기능하는 한편, privacy와 policy enforcement 같은 공백 연구 과제도 구체적으로 드러냈다는 점에서 분야에 실무적·학술적 의미가 큽니다.



### Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems (https://arxiv.org/abs/2606.18837)
- **Prior Approaches**: 기존 automatic-MAS는 크게 추론 시점(inference-time)과 학습 시점(training-time)으로 나뉜다. inference-time 계열은 frozen frontier LLM로 탐색은 하지만, 동일한 검색을 반복해 과거 시행착오를 축적·전이하지 못한다. training-time 계열은 fine-tuning으로 경험을 내재화하지만 작은 모델의 성능 상한과 대규모 고품질 궤적 데이터 요구, 그리고 초거대 frontier LLM로의 확장 비용 문제가 크다.

- **Core Contribution**: 이 논문은 고수준 오케스트레이션을 경험-유지와 파라미터 업데이트를 분리한 “evolvable Meta-Skill”로 모델링하는 Skill-MAS를 제안한다. 그 결과, frontier LLM은 고정한 채로도 Meta-Skill을 여러 라운드에 걸쳐 진화시켜, 과거 실패/성공의 구조화된 지식을 다음 생성에 반영한다. 단일 에이전트의 skill 진화가 아닌, MAS 생성 전체의 전략(분해-역할-워크플로) 수준을 직접 업데이트한다는 점이 핵심이다.

- **Technical Challenges**: 주요 기술 난제는 (1) 탐색 변동성 때문에 “진짜 실력 부족”과 “우연한 실행 잡음”을 구분해야 하고, (2) 경험을 효율적으로 축약해 일반화 가능한 원칙으로 바꿔야 하며, (3) 학습 비용 없이 closed optimization loop를 구성해야 한다는 것이다. 이를 위해 Multi-Trajectory Rollout으로 태스크별 확률적 분포(여러 궤적)를 샘플링해 불확실성/난이도를 계산하고, Selective Reflection에서 변동성과 난이도가 높은 우선 태스크만 뽑아 within-task 대비 분석과 cross-task 합성으로 모듈 단위의 수정을 Evidence 패키지로 구성한다. 수정은 세 모듈 스캐폴드를 유지하며 해당 모듈에 한정해 “전략 수준”으로만 반영되도록 제약한다.

- **Empirical Impact**: 네 가지 복잡 벤치마크와 네 가지 서로 다른 LLM을 Meta-agent로 사용한 실험에서 Skill-MAS는 초기 Meta-Skill만으로도 경쟁력 있는 성능을 보이며, 최적화된 Meta-Skill은 대부분의 baseline을 큰 폭으로 앞선다. 특히 비용-성능 관점에서 inference-time 계열의 반복 재최적화 비용을 피하면서도 training-time 계열의 일반화 한계를 완화해 더 유리한 절충점을 달성한다. 또한 진화된 Meta-Skill은 미지 태스크 및 다른 백본 LLM로의 transferability가 강해, 단순 프롬프트 탐색 이상의 “전략 원칙”이 학습되었음을 뒷받침한다.



### EARS: Explanatory Abstention for Reliable Sub-Agent Modeling in Large-scale Multi-Agent Systems (https://arxiv.org/abs/2606.18668)
- **Prior Approaches**: 기존 centralized multi-agent systems(MAS)는 coordinator가 사용자 요청을 해석해 sub-agent로 라우팅하고 결과를 통합해 답을 내는 구조가 주류였습니다. 그러나 선행 연구는 주로 coordinator의 routing/역할 선택 같은 상위 조정 실패를 줄이는 데 집중했고, sub-agent가 실행 중 실패할 때 그 신호가 coordinator와의 통신에서 어떻게 깨지는지는 상대적으로 덜 다뤄졌습니다. 또한 single-agent에서의 refusal(거절) 연구는 있었지만, MAS에서 abstention이 ‘협업 복구를 위한 커뮤니케이션 신호’로 쓰인다는 관점은 부족했습니다.

- **Core Contribution**: EARS(Explanatory Abstention for Reliable Sub-Agent Modeling)는 sub-agent의 abstention을 단순 거절이 아니라 coordinator를 위한 inter-agent communication protocol로 재정의합니다. sub-agent가 애매/부족/미지원/오라팅 같은 failure state를 ‘카테고리 + 근거(rationale)’ 형태로 노출하면, coordinator는 이를 바탕으로 재질문·재라우팅·fallback을 수행할 수 있게 됩니다. 이를 위해 도메인 적응용 abstention 데이터 파이프라인과 fine-tuning 전략을 함께 제시합니다.

- **Technical Challenges**: 핵심 난제는 task-specific하게 어떤 상황을 abstention으로 분류해야 하는지 라벨 신뢰도를 확보하는 것입니다. EARS는 calibrated LLM-as-a-Judge를 seed set으로 단계적으로 보정하고, 여러 judge의 unanimous 합의(unanimity agreement)만 학습 데이터로 채택해 precision 중심의 신뢰도 높은 라벨을 구성합니다. 또한 Ambiguous Query/Insufficient Input/Missing Capability/Misrouting의 failure taxonomy에 맞춘 계층적(hierarchical) 라벨링과 근거 생성으로 coordinator가 이해 가능한 구조화된 피드백을 학습시킵니다.

- **Empirical Impact**: e-commerce 프로덕션 환경의 business intelligence(BI) sub-agent에서 EARS는 전체 response pass rate를 68.5%에서 78.9%로(상대 15.2%) 끌어올렸습니다. 특히 segmentation 질의에서 abstention 학습의 효과가 더 크게 나타났고, syntax validty는 유지되어 잘못된 포맷 생성으로 품질이 떨어지지 않았습니다. shadow deployment의 human 검증에서도 세션 성공률이 67.1%로 올라, baseline 대비 신뢰성 개선이 실사용 관점에서 확인됐습니다.



### Gender Bias in LLM Hiring Decisions: Evidence from a Japanese Context and Evaluation of Mitigation Strategies (https://arxiv.org/abs/2606.18649)
- **Prior Approaches**: 기존 연구는 주로 영어·서구식 이력서(예: JobFair)에서 LLM의 성별 편향을 평가해 왔고, 대체로 ‘동일 자격인데 여성 후보를 더 높게 점수’하는 경향(친여성 편향)이 관찰되었습니다. 또 다른 감사(audit) 연구들은 대응 실험 등으로 성·인종 격차를 측정했지만, 일본처럼 문화·채용 형식이 다른 환경에서는 일반화 여부가 불명확했습니다. 따라서 비서구 맥락과 일본의 표준 이력서 포맷(rirekisho)을 넣은 대규모 검증과 완화책의 실효성 평가는 공백으로 남아 있었습니다.

- **Core Contribution**: 이 논문은 일본 기업 채용 맥락에서 LLM이 친여성 성향 편향을 재현하는지, 그리고 이를 줄이려는 실무적 개입이 통하는지(counterfactual resume 설계 포함) 체계적으로 확인했습니다. 특히 일본어로 된 rirekisho 형식 60개 이력서를 기준으로 5개 최신 모델을 대상으로, 프롬프트 지시와 이름 익명화라는 두 완화 전략을 같은 실험 틀에서 비교합니다. 결과적으로 일본에서도 친여성 편향이 견고하게 나타나며, ‘공정성 지시’는 거의 효과가 없고 ‘이름 정보’가 편향의 핵심 통로임을 메커니즘 수준에서 보여줍니다.

- **Technical Challenges**: 어려움은 (1) 일본의 언어·표현·이력서 구조가 모델 평가에 미치는 영향, (2) 편향이 진짜 성별 신호 때문인지 자격 차이 때문인지 분리, (3) API 환경에서 프롬프트/PII 마스킹처럼 제한된 개입만으로 편향을 줄일 수 있는지 검증하는 데 있습니다. 연구진은 동일한 이력서에 이름만 바꾸는 counterfactual resume 설계를 쓰고, 일본 이름의 성별 신호를 음운/문자 단서로 엄밀히 선별해(강한 gender-signal 조합만) 신뢰 가능한 비교를 만들었습니다. 또한 이름 제거가 편향을 얼마나 줄이는지 명시적으로 분해하는 name-reliance 분석을 수행해, 편향이 이름 기반 추론에서 주로 발생함을 정량화했습니다.

- **Empirical Impact**: 실험(총 43,200회 API 호출)에서 5개 모델 모두에서 여성 이름 후보가 유의미하게 더 높은 점수를 받는 친여성 편향이 확인됐고, 특히 모델별로 크기 차이는 있어도 방향성은 일관적이었습니다. 완화 측면에서는 프롬프트의 gender-neutrality 지시가 편향을 유의하게 줄이지 못한 반면, 이름을 제거했을 때 편향이 거의 전부 감소했습니다. 또한 GPT-4o의 경우 privacy filter 기반 익명화 토큰 처리와 content safety 필터 사이의 비호환으로 42% 거절(refusal)이 발생해, 이름 익명화가 실제 배포 파이프라인에서 예기치 못한 부작용을 낳을 수 있음을 실증적으로 경고합니다. 이는 일본 같은 비서구 맥락에서도 ‘이름을 중심으로 성별이 추론되고 보정되는’ 편향 메커니즘이 강하게 작동할 수 있음을 시사하며, 산업 현장의 편향 완화 전략 재설계 필요성을 높입니다.



### PersonalPlan: Planning Multi-Agent Systems for Personalized Programming Learning (https://arxiv.org/abs/2606.18633)
- **Prior Approaches**: 기존 LLM 기반 교육·코드 튜터는 개념 설명, Socratic debugging, 대화형 멀티롤을 제공하지만, 학습자 프로필을 에이전트 구성·도구 바인딩·의존성 그래프까지 반영하는 ‘검증 가능한 실행 계획’으로 잘 변환하지 못하는 경우가 많았습니다. 일반적인 멀티에이전트 프레임워크와 워크플로 플래너는 작업 완성/오케스트레이션에 최적화되어 있고, 학습자별로 선행지식 경로와 pedagogical scaffolding을 구조적으로 달리하는 설계가 약하다는 한계가 지적됩니다. 또한 fluent한 표면 생성만으로는 순환 의존성이나 에이전트-도구 불일치 같은 잠재 오류를 사전에 차단하기 어렵습니다.

- **Core Contribution**: 이 논문은 학습자 프로필을 조건으로 하여 튜터링 MAS의 실행 가능한 계획을 생성하는 profile-conditioned multi-agent planning 프레임워크 PersonalPlan을 제안합니다. PersonalPlan은 상위 수준의 교육 스캐폴드(에이전트/서브태스크)와 하위 수준의 실행 워크플로(단계/의존성/도구)를 분리해 학습하고, Reward-Adaptive GRPO로 ‘실행 가능성·프로필 적합·교육 단계 커버리지’를 동시에 강화합니다. 함께 연구를 돕기 위해 MAP-PPL 데이터셋도 공개하며, 각 인스턴스는 query–profile–plan을 strict JSON 형태로 제공해 평가 가능성을 높입니다.

- **Technical Challenges**: 핵심 난제는 계층형 생성에서 생기는 hierarchical exposure bias를 줄이면서, 생성된 스캐폴드를 기반으로 실제로 깨지지 않는(acyclic·스키마-valid·도구 매칭) 실행 계획을 만들어내는 것입니다. 이를 위해 두 단계 LoRA 기반 계층 SFT를 사용해 PAD는 프로필 인지 분해(교육 스캐폴드)만 학습하고, SDP는 주어진 스캐폴드 컨텍스트에서 단계 의존성과 도구 결합을 학습합니다. 더 나아가 joint alignment 단계로 SDP가 PAD의 실제 생성 분포를 보게 만들고, GRPO에서는 규칙 기반 verifiable soft reward(구조/개인화/교육)와 hard feasibility gate(스키마·사이클·도구 위반)를 함께 적용해 단계별 품질을 안정화합니다.

- **Empirical Impact**: 실험은 MAP-PPL에서 PersonalPlan이 frontier LLM, generic MAS 프레임워크, agentic workflow planner를 대상으로 plan executability, personalization, pedagogical quality 전반에서 우수함을 보였다고 보고합니다. 특히 8B와 32B 변형만으로도 state-of-the-art 수준의 실행 가능성과 교육적 계획 품질을 달성해, 멀티에이전트 튜터-학습자 상호작용을 ‘계획-실행’ 단위로 오케스트레이션할 수 있음을 시사합니다. Ablation 결과로도 계층 SFT의 분리 학습과 GRPO 기반 보강, 그리고 joint alignment가 각각 도구 결합 정확도와 교육 품질 향상에 기여함이 확인됩니다.



### Characterizing Opinion Evolution of Networked LLMs (https://arxiv.org/abs/2606.18276)
Comments:
          19 pages, 2 figures

- **Prior Approaches**: LLM 다중 에이전트가 형성하는 의견 전파를 설명하기 위해 DeGroot, Friedkin-Johnsen, Hegselmann-Krause 같은 고전 opinion dynamics 모델을 적용하려는 시도가 이어져 왔습니다. 다만 기존 연구들은 “어느 모델이 대충 맞는다” 수준의 비교에 그치거나, LLM 군집이 어떤 고전적 특성(고집, 동질성, 편향)을 실제로 필요로 하는지 정밀하게 분해하지 못했습니다. 특히 DeGroot식 단순 평균(averaging)류는 LLM의 실제 의견 진화를 추적하지 못한다는 관찰이 누적됐습니다.

- **Core Contribution**: 이 논문은 LLM 네트워크의 belief(의견) 진화가 고전 모델의 어떤 구성요소를 “반드시” 포함해야 설명되는지 RQ1~RQ3로 정량화합니다. 핵심 발견은 에이전트들이 회귀해야 하는 고정된 ‘bias(고유 편향/기준 의견)’가 LLM 의견 역학에서 중요한 동인이라는 점이며, bias 항을 넣으면 누적 평균 의견 오차(cumulative mean opinion error)가 최대 88%까지 줄어듭니다. 또한 이러한 결론이 모델 패밀리, 토픽, 네트워크 설정 전반에 걸쳐 일반화된다고 보고합니다.

- **Technical Challenges**: 문제는 LLM 대화 로그를 시간축의 수치 의견으로 바꾸고, 동시에 고전 모델(고집·동질성·편향·가중치 구조)의 파라미터를 데이터에 맞게 ‘분해’해내야 한다는 데 있습니다. 이를 위해 자연어 게시물을 임베딩 기반 유사도로 의견값으로 변환하고, 그 궤적에 대해 각 모델을 피팅하며 구성요소별 기여도를 오차로 측정합니다. 더 나아가 네트워크 토폴로지가 파라미터를 좌우하는지(그래프-특화 vs 그래프-무관)와 edge별 영향 가중치를 추가해도 성능이 달라지는지까지 비교하며 bias가 주된 설명요인임을 테스트합니다.

- **Empirical Impact**: 실험에서는 Qwen3, Llama3.1, Gemma3를 climate change, vaccines, gun control 같은 쟁점 토픽에서 다중 에이전트로 시뮬레이션해 고전 모델의 out-of-sample 예측 오차를 평가합니다. 결과적으로 bias를 포함한 모델들이 대부분의 설정에서 최저 오차를 보였고, 특히 Llama3.1에서 누적 Wasserstein-1 거리 오차도 최대 67%까지 감소했습니다. 요약하면, LLM 의견 전파를 ‘고집’이나 ‘동질적 이웃 가중’만으로 설명하기보다, 네트워크 전반에 공통으로 작동하는 bias 회귀를 모델링 설계/검증의 기준점으로 삼아야 한다는 실증 근거를 제공합니다.



### Enhancing Decision-Making with Large Language Models through Multi-Agent Fictitious Play (https://arxiv.org/abs/2606.19308)
Comments:
          18 pages, 8 figures

- **Prior Approaches**: 기존 LLM 기반 multi-agent systems(MAS)는 divide and conquer로 실행 복잡도(긴 추론 연쇄, 넓은 정보 커버, 이질적 스킬)를 분산해 해결하는 데 집중해 왔다. 하지만 협상·게임·경쟁 시장처럼 이해관계자들의 결정이 서로 물고 늘어지는 결정 문제에는 그대로 적용하기 어렵다. 일부 decision-making 연구는 ToM(Theory of Mind)처럼 higher-order 상호 추론을 한 번의 추론 패스에서 펼치려 하지만, LLM이 재귀적 mutual anticipation을 깊게 수행할 때 성능이 급격히 떨어진다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 의사결정 복잡도의 새로운 유형으로 stance entanglement(입장 얽힘)을 정의하고, 단순 분할·고립 해법이 실패하는 이유를 “상호 의존 루프 안에서 한 궤적에 entangled된 입장들을 동시에 추론해야 하기 때문”으로 정리한다. 이를 해결하기 위해 Multi-Agent Fictitious Play(MAFP)를 제안하며, 각 이해관계자의 stance를 에이전트로 두고 결정 만들기를 equilibrium 탐색 과정으로 바꾼다. MAFP는 fictitious play 원리를 따라 각 라운드에서 다른 에이전트의 과거 결정 “경험적 혼합(empirical mixture)”에 대한 best response를 반복 수행해 약점을 노출하고 보완하도록 한다.

- **Technical Challenges**: 핵심 기술 난관은 natural-language 정책 공간에서 equilibrium 성질(일방적 일탈로 얻는 이득이 0에 가까움)을 만족하는 결정을 찾되, LLM의 재귀적 상호 추론을 단일 체인에서 깊게 펼치지 않아야 한다는 점이다. MAFP는 이를 위해 training-free로, 라운드마다 aggregation operator로 상대 정책의 empirical mixture를 요약·구성하고 best-response operator로 해당 혼합에 최적 대응하는 정책을 생성한다. 또한 마지막에 각 라운드에서 나온 정책들의 empirical mixture를 출력 기준으로 삼아, 특정 시점의 반응이 아니라 누적된 공진 결과가 반영되게 설계했다.

- **Empirical Impact**: 13개 시나리오(경쟁 게임과 자연어 협상)에서 MAFP는 tournament strength와 robustness라는 상호보완 지표 모두에서 단일 라운드 및 다중 라운드 기준선들을 능가했다. 특히 debate나 self-reflection처럼 “iteration만 추가한” 접근은 성능 개선이 제한적이었고, ToM은 일부 개선되지만 robustness에서는 MAFP에 뒤졌다. 또한 MAFP-Last(aggregation 제거)는 약점이 드러나, 최신 iterate에만 greedy하게 반응하는 방식이 fictitious play의 핵심인 “과거 히스토리 기반 기대”를 잃기 때문임을 보여준다. 전반적으로 stance entanglement을 equilibrium 탐색으로 다루는 새로운 MAS 패러다임으로서, 전략을 행동 전 단계에서 강건하게 결정해야 하는 분야에 실질적 의미가 크다.



### Digital Speech Acts Retain Control of Copyright with People, Not Platforms (https://arxiv.org/abs/2606.19263)
- **Prior Approaches**: 기존 플랫폼 거버넌스는 코드에는 copyright로, 사용자 콘텐츠에는 contract(서비스 약관)로 비대칭 권한을 구축해 왔습니다. 1980~90년대의 소프트웨어 관련 판례는 코드(기계어 포함) 전반에 폭넓은 저작권 보호를 부여하고, 사용자는 라이선스 계약을 통해 사실상 권한을 양도하는 구조를 정착시켰습니다. 그 결과 플랫폼은 서버 보관과 약관 기반 라이선스 추출을 결합해 ‘소유권의 분리(ownership vs. possession)’를 시스템적으로 만들었습니다.

- **Core Contribution**: 이 논문은 사람의 단말에서 개인 키로 콘텐츠를 ‘cryptographically signing’하는 행위를 digital speech act(디지털 스피치 액트)로 정의하고, 여기에 저작권 보호의 법적 기반을 세우려 합니다. 저자는 Burrow-Giles의 인간의 volitional creative choice, Feist의 최소 창작성, Copyright Act의 fixation 요건을 조합해 디지털 스피치 액트가 현행 미국 저작권 체계에 들어온다고 주장합니다. 동시에 이러한 권리는 분산형(서버리스) 아키텍처에서 계약·중개 없이 귀속되며, 디지털 주권과 민주적 자기통치의 전제조건이 된다고 확장합니다.

- **Technical Challenges**: 핵심 난제는 서명 과정이 알고리즘적으로 ‘기계 실행’되더라도, 표현의 창작적 선택이 인간에게서 비롯되었다고 어떻게 정리하느냐입니다. 논문은 디지털 스피치 액트가 콘텐츠 선택, 서명 시점, 수행되는 의사표현, 사용할 cryptographic identity 같은 인간의 의도적 선택을 포함한다고 보고, 암호 알고리즘은 그 선택을 수학적으로 집행할 뿐 창작 결정을 하지 않는다고 구분합니다. 더 나아가 digital social contract(디지털 소셜 컨트랙트)가 non-unbundling(서명과 콘텐츠 분리 불가)과 provenance preservation(전달 경로 보존)을 코드로 강제해, 저작권 주장과 책임이 서명-콘텐츠에 ‘동행’하도록 설계합니다.

- **Empirical Impact**: 이 글은 새로운 실험 벤치마크보다, 기존 저작권 판례와 분산형 아키텍처의 결합이 ‘플랫폼 추출’이 아니라 ‘사람 중심 소유·통제’로 귀결될 수 있음을 논리적으로 입증하는 데 무게를 둡니다. 즉, 저작권이 단순 방어 논리가 아니라, 창작·유통·귀속(Attribution)을 제3자 서버 없이도 정합적으로 고정해 주는 권리로 작동할 수 있음을 제시합니다. 향후 콘텐츠 중재·플랫폼 거버넌스·분산형 인프라 설계에서, 소유권과 소지(보유)가 함께 성립하는 아키텍처가 더 설계 가능한 대안이 될 수 있다는 점에서 의미가 큽니다.



### Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams (https://arxiv.org/abs/2606.19111)
Comments:
          33 pages

- **Prior Approaches**: 기존 multi-agent LLM 연구는 debate, 역할 분담/분해, self-refinement처럼 지식 수준에서의 조정(무엇을 생각할지)을 바꾸는 방식이 주를 이룹니다. 하지만 평가가 최종 정확도에 치우쳐 있고, 실제로 ‘과정 수준 control이 언제 통하는가’를 분리해 측정한 사례는 드뭅니다. 또한 컨트롤러가 프롬프트처럼 모놀리식으로 설계되는 경우가 많아, 어떤 구성요소가 행동 차이를 만드는지(컴포넌트 단위 원인분해)도 깔끔히 검증하기 어렵습니다.

- **Core Contribution**: 이 논문은 multi-agent LLM 팀에서 ‘process-level coordination control’이 정확도를 더하는 조건을 측정 가능한 형태로 정식화하고, 그 조건이 team science의 contingency 예측과 일치하는지 검증합니다. 컨트롤러를 공통 action vocabulary(예: explore, revise, accept, synthesize) 위의 explicit control 정책으로 구현하고, 리더십 스타일(거래적/변혁적/상황적)을 그 정책으로 operationalize합니다. 결과적으로 “항상 이기기”가 아니라 “특정 조건에서만 가치가 생기는지”를 지도/이름표가 아닌 경계(boundary)로 다룹니다.

- **Technical Challenges**: 핵심 과제는 ‘정확도’ 대신 컨트롤러의 과정 차이를 직접 읽어내는 측정 설계이며, 저자들은 majority lock-in, 탐색률, round-0 오답 컨센서스에서의 recovery 같은 behavioral signatures를 1차 지표로 둡니다. 또 컨트롤러를 작은 명시적 action set으로 정의해 per-action ablation이 가능하게 만들었고, 임의 규칙(이론 없는 컨트롤)에서는 majority voting 수준으로 수렴해 ‘액션 구성’이 아닌 ‘이론 기반 규칙’의 역할을 분리합니다. 추가로 open-ended 수치형 과제의 추출 편향을 cross-round majority extractor로 통제해, 컨트롤 효과와 무관한 측정 노이즈를 줄입니다.

- **Empirical Impact**: 4개 task regime과 3개 open-weight 모델 계열에서, 어떤 컨트롤러도 전반적으로 정확도를 압도하지 못했는데 이는 contingency 관점의 null 결과와 일치합니다. 다만 round-0 독립 다수결이 신뢰롭지 않을 때에만 성능 이득이 나타났고, 그중에서도 recoverability가 가능한 영역에서 situational/transactional 계열이 기준선 대비 유의미하게 개선합니다(주로 round-0 majority가 흔들리는 조합에서 +8pp급 사례). 경계 probes(예: MATH-500 Level 5 확장, adversarial NLI, Winogrande, 도덕 판단)로도 “라운드0 신뢰도-회복 가능성-상호작용이 이미 복구하는지” 축이 재현되며, leadership substitutes/path-goal redundancy/situational readiness gap 같은 팀 과학 개념과 실측이 매핑됩니다.



### Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents (https://arxiv.org/abs/2606.18947)
Comments:
          15 pages, Figure 8

- **Prior Approaches**: 기존 접근은 LLM의 native search grounding처럼 모델-공급자 경계 안에 검색 정책(프로바이더 선택, 결과 형식, 증거 주입, 비용·지연)을 숨겨두는 방식이 주류입니다. 이 때문에 검색 품질과 운영 지표를 튜닝·검사·이식·재사용하기 어렵고, 엄격한 출력 계약(예: JSON/단일 엔티티)을 깨는 Search-Induced Verbosity 같은 포맷 드리프트 위험이 커집니다. 또한 RAG나 도구 사용 연구는 검색-추론 상호작용을 다루지만, ‘실시간 검색 인터페이스’를 명시적 시스템 경계로 다루는 관점은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 Vendor-agnostic 경계로 검색 근거를 분리하는 Decoupled Search Grounding(DSG)을 제안합니다. DSG는 MCP-compatible gateway를 통해 검색을 추론 모델 바깥의 구조화된 tool 계층으로 옮기며, 프로바이더 라우팅, 출처 기반 context 렌더링, fallback, retrieval-depth, exact/semantic caching을 1급 제어(control)로 노출합니다. 결과적으로 추론 모델은 교체 가능하게 유지하면서도, 검색은 ‘운영 가능한 인터페이스’로 취급할 수 있게 됩니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 프로바이더별 이질적인 결과를 표준화해 모델이 일관된 근거를 받게 하고 (2) 캐시·fallback·비용/지연 목표를 정책으로 안정화하며 (3) 출력 계약 위반(Search-Induced Verbosity)을 인터페이스 레벨에서 완화하는 것입니다. DSG는 provider registry와 YAML 어댑터로 다양한 검색 백엔드를 같은 내부 결과 객체로 정규화하고, 캐시 키를 provider-scoped로 관리해 서로 다른 공급자 결과가 섞여 재사용되는 문제를 줄입니다. 또한 semantic cache의 유사도 임계값과 time-to-live(신선도 우선)를 설정해 재현성과 최신성 간 균형을 맞추며, 툴 호출/응답이 명확한 경계가 되도록 설계합니다.

- **Empirical Impact**: 5개 frontier 모델과 SimpleQA/FreshQA/HotpotQA, 그리고 e-commerce Query Intent Understanding(QIU) 프로덕션 워크로드에서 native search와 비교해 비용·지연·품질 트레이드오프를 체계적으로 입증합니다. SimpleQA에서 DSG는 86.1%(native 87.7%)에 근접하면서 검색 비용은 91% 절감했고, warm-cache hit rate는 99.4%, 지연은 68% 감소했습니다. FreshQA는 native가 앞섰지만(만약 신선도 이점이 강한 경우), QIU에서는 DSG가 native와 비슷하거나 약간 상회하면서 검색 비용을 98% 이상 줄였으며, 모델 공급자에 종속되지 않는 ‘운영 최적화 가능한 검색 경계’의 실용성을 강조합니다.



### LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents (https://arxiv.org/abs/2606.18388)
- **Prior Approaches**: RL post-training에서는 고정된 가이드북형 스케줄(응답 길이/rollouts/난이도 단계 증가, 또는 일부 진동)을 미리 정해 실행하는 경향이 강했습니다. 하지만 이런 방식은 “언제(트리거), 얼마나(폭), 무엇을(파라미터 조합)” 바꿔야 하는지 훈련 동역학이 바뀔 때 대응하기 어렵고, KL 스파이크·붕괴·정체 같은 이상 징후에 체계적으로 반응하지 못합니다.

- **Core Contribution**: 이 논문은 최적 multi-stage RL post-training에서 나타나는 구조적 비대칭을 제시합니다: capacity 파라미터(응답 길이, rollouts 등)는 단계가 진행될수록 단조 누적되는 반면, regularization 파라미터(학습률, KL 계수, temperature 등)는 훈련 역학 변화에 따라 주로 진동합니다. 또한 LLM 에이전트가 체크포인트별 동역학을 해석해 다중 파라미터를 “동시 전환”하도록 설계한 LLMZero를 통해 이 원칙을 찾고 검증합니다.

- **Technical Challenges**: 핵심 난제는 고정 스케줄로는 표현하기 어려운 비정상(non-stationary) 탐색-활용 트레이드오프를, 여러 하이퍼파라미터의 인과적 연동을 고려해 실시간 전환해야 한다는 점입니다. LLMZero는 MCTS(UCT)로 학습 궤적을 트리 탐색하고, 시각화·텍스트 지표를 결합한 proposer 에이전트가 병리 진단 후 3개 이상 파라미터를 조율한 전환을 제안하며, agentic early stopper로 유망하지 않은 가지를 중단해 탐색 비용을 줄입니다.

- **Empirical Impact**: 4개의 다양한 GRPO 태스크에서 LLMZero가 베이스 모델 대비 상대 9%~140%, grid search 대비 상대 6%~15% 개선을 보였고, 무작위 탐색과 skill-based 에이전트를 일관되게 능가했습니다. 특히 최적 전략을 12 iteration 내에 찾는 경우가 많아 반복 효율도 높았으며, SSMR-bench에서는 모델 크기(0.6B~8B) 전반에서 성능이 유지·확장되는 동시에 OOM 같은 인프라 실패까지 회피해 실무적 의미가 큽니다.




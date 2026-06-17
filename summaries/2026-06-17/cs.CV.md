New uploads on arXiv(cs.CL)

### Variable-Width Transformers (https://arxiv.org/abs/2606.18246)
- **Prior Approaches**: 트랜스포머 성능을 끌어올리는 대표 축은 depth(깊이)와 width(폭) 스케일링이었지만, 대부분의 구조는 모든 레이어의 width를 고정해 계산·파라미터 예산을 깊이 전반에 균등 배분합니다. 한편 일부 연구는 레이어별로 FFN 중간 차원만 조절하거나 스케일링 법칙에 shape을 반영했지만, 블록 전체 폭을 비균일하게 재배치하는 문제는 상대적으로 덜 다뤄졌습니다. 또한 잔차 경로(residual stream)에서 폭을 바꾸면 투영 병목이나 skip 경로 변화 같은 구현상의 문제가 커져 실용화가 어려웠습니다.

- **Core Contribution**: 이 논문은 깊이 축을 따라 폭을 비균일하게 배분하는 ×-shaped former(이하 <former)을 제안하며, 앞·뒤 레이어는 넓게 유지하고 가운데를 좁히는 병목 구조가 언어 모델링에 유리하다고 실증합니다. 매개변수 수를 동일하게 맞춘 uniform-width 기준선과 비교해 <former가 언어 모델링 loss(및 perplexity)를 더 낮추며, 더 적은 FLOPs와 더 작은 KV cache까지 달성합니다. 특히 decoder-only LM에서 dense 200M~2B, MoE 3B까지 일관된 향상을 보입니다.

- **Technical Challenges**: 비균일 폭을 단순히 레이어마다 바꾸면 residual 경로가 찢기듯 바뀌어 투영 병목이 생기고 skip path가 달라질 수 있습니다. 논문은 이 문제를 완화하기 위해 “parameter-free residual resizing”을 도입해, 전역 잔차 차원은 고정하되 각 레이어가 참조·갱신하는 residual slice만 달리하며 나머지 좌표는 상류로 copy/우회하도록 구성합니다. 또한 병목 레이어의 위치 ℓ*와 병목 폭 dℓ*를 레이어 수·모델 폭에 대한 비율로 파라미터화해, 여러 모델 스케일에서도 재현 가능한 설정을 찾는 절차를 제시합니다.

- **Empirical Impact**: 200M~2B(밀집)에서 <former는 parameter-matched constant-width 대비 언어 모델링 perplexity를 상대적으로 약 3% 개선하면서 KV cache 메모리·I/O 비용도 각각 약 10%/15% 수준으로 줄였습니다. FLOPs는 fitted loss-matched scaling 곡선을 기준으로 약 22% 감소로 보고되며, MoE 설정에서도 perplexity 관점의 이득이 관측됩니다. 분석에서는 가운데 레이어의 representation collapse(압축 골짜기)를 <former가 완화하며, MLP 활성의 활용 균형과 residual stream의 정규화된 행렬 엔트로피가 qualitatively 다르게 나타난다고 설명합니다.



### ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues (https://arxiv.org/abs/2606.18237)
- **Prior Approaches**: 기존 연구는 LLM agent의 재현성(reproducibility) 지원 여부를 보기 위한 benchmark를 제안했지만, 데이터 큐레이션과 평가에 상당한 수작업이 필요해 대규모 확장이 어렵다는 한계가 컸습니다. 또 많은 benchmark가 실제 재현 과정에서 발생하는 다양한 막힘(예: 누락된 실험 조건, 오류 재현 등)을 자연스럽게 반영하기 어렵다는 문제도 동반했습니다. 결과적으로 현실적인 “재현 장애물”을 폭넓게 스케일해 측정하기가 힘들었습니다.

- **Core Contribution**: 이 논문은 ReproRepo라는 스케일 가능한 재현성 평가 프레임워크를 제안합니다. 핵심은 실제 연구 과정에서 자연발생하는 GitHub issue(인간이 제기한 문제)를 재현 장애물에 대한 supervision으로 활용해, 수작업 부담을 줄이면서도 현실적인 blockers를 대규모로 수집·평가한다는 점입니다. 이를 통해 paper-repository 쌍만으로 에이전트가 어떤 종류의 재현 문제를 찾아내는지 체계적으로 감사(auditing)할 수 있게 했습니다.

- **Technical Challenges**: 기술적 도전은 (1) 방대한 ML 논문-저장소 쌍에 대해 (2) 인간이 보고한 issue를 재현 장애물의 의미적 정답으로 삼고 (3) 모델 agent가 code 실행 없이도 의미적으로 연관된 문제를 찾아내게 하는 평가 설계를 요구한다는 점입니다. 논문은 ReproRepo에서 최근 주요 컨퍼런스의 1,149개 논문을 대상으로, 서로 다른 frontier model-agent 설정 4가지를 구성해 paper와 repository 정보를 매칭하는 방식으로 측정합니다. 또한 agent가 “보이는 실패”와 “정확한 의미 영역”을 잘 짚지만 “정확한 국소화(localization)”는 여전히 부족할 수 있음을 분석으로 확인합니다.

- **Empirical Impact**: 실험에서는 code 실행 없이도 LLM agent가 paper-repository 쌍에서 실제 재현성 문제를 상당수 탐지할 수 있음을 보였습니다. 특히 Codex with GPT-5.5 조합이 연구 대상 논문 약 90%에서 인간이 보고한 semantically related blocker(의미적으로 관련된 장애물) 적어도 1개를 찾아냈습니다. 다만 정확한 위치 특정까지는 충분하지 않을 수 있어, 향후 실사용 재현성 감사에서 agent의 강점(가시적 실패 탐지)과 한계(정밀 localize)를 동시에 시사합니다. ReproRepo는 향후 real-world reproducibility auditing을 위한 재사용 가능한 평가 프레임워크로 기능할 전망입니다.



### Darshana Graph: A Parallel Commentary Corpus for Comparative Indian Philosophy, with Stylometric and Exploratory Graph Analyses (https://arxiv.org/abs/2606.18222)
Comments:
          12 pages, 1 figure. Open Source Code available at this https URL and dataset at this https URL

- **Prior Approaches**: 기존 연구는 철학 텍스트를 주로 단일 주석 전통이나 단일 장르 단위로 분석해 비교가 제한되는 경우가 많았다. 또한 대부분의 말뭉치는 주석자들 간 정렬(alignment)이 없어 동일 구절/경구를 두고 해석 차이를 직접 대조하기 어렵다.

- **Core Contribution**: 이 논문은 Darshana Graph라는 공개 말뭉치를 제안하며, 총 125,000개 이상 텍스트 기록에 더해 약 8,500개 규모로 동일한 뿌리 구절/경구가 18명의 역사적 주석가(5개 Vedanta 및 기타 darshanas)에 의해 정렬된 부분집합을 제공한다. 이를 통해 서로 독립적인 해석 전통이 같은 원문을 어떻게 읽는지 ‘직접 비교’가 가능해진다.

- **Technical Challenges**: 핵심 기술적 난제는 주석자 간 동일 구절을 구조적으로 정렬하는 작업과, 그 위에서 해석 차이를 계량화하는 방법의 설계였다. 저자들은 첫째로 기계학습 지표 없이 인용 밀도, 명시적 반박률, 문장 복잡도 같은 스타일 지표로 통계 분석을 수행하고, 둘째로는 제한된 LLM 파이프라인으로 개념 간 관계를 미리 정의된 relation vocabulary로 추출하되 결정적 사후 검증으로 품질을 통제했다.

- **Empirical Impact**: 실험 결과는 인용 밀도와 반박률의 중간 수준 음의 상관, 특정 관련 계보에서 반박률 증가, 그리고 Pali Canon 내부 장르 간 차이를 보여준다. 또한 추출된 개념 관계 그래프는 종단 간(교파 간) 불일치 패턴을 드러내지만, 임베딩 기반 독립 분석과의 불일치 같은 한계도 함께 관측되어 후속 검증의 방향을 제시한다.



### Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients (https://arxiv.org/abs/2606.18216)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 지식증류는 작은 학생이 큰 교사의 logits(로짓), hidden state, 샘플 출력을 따라가게 하는 방식이라, 학생이 충분히 작아질수록 분포 학습이 깨지며 mode-seeking과 암기 문제가 커집니다. 이를 완화하려고 on-policy distillation이나 self-distillation, hybrid RL++distillation이 나왔지만, 결국 학습 신호가 교사 분포(로짓/응답)를 따라가야 해서 작은 모델의 일반화 취약성이 재현됩니다.

- **Core Contribution**: 이 논문은 “교사를 policy gradient 안으로 넣지 않되(teacher response를 직접 그라디언트 타깃으로 사용하지 않되), prompt 안에만 넣어 학습 신호를 복구”하는 ZPPO(Zone of Proximal Policy Optimization)를 제안합니다. 핵심 아이디어는 hard question(학생 평균 롤아웃 정확도 0.5 미만)에서 BCQ(Binary Candidate-included Question)와 NCQ(Negative Candidate-included Question) 두 가지 프롬프트 재구성을 통해 학생이 자신의 on-policy 롤아웃으로부터 학습하도록 만드는 것입니다.

- **Technical Challenges**: 소형 모델 RL의 대표 병목은 모든 롤아웃이 실패해 advantage가 0이 되면 해당 문제에서 학습 신호가 사라진다는 점입니다. ZPPO는 teacher response를 잘못된 방식으로 policy gradient에 주입하면 on-policy 가정이 깨져 drift가 생긴다는 문제를 피하기 위해, 교사 텍스트를 입력 프롬프트 후보로만 제공하고 학생이 새로 샘플링한 응답으로만 정책 업데이트가 일어나게 설계했습니다(BCQ/NCQ + 프롬프트 replay buffer로 hard 문제를 졸업할 때까지 반복 노출).

- **Empirical Impact**: Qwen3.5 계열에서 0.8B~9B 학생(교사 27B)과 31개 벤치마크(16 VLM, 10 LLM, 5 Video) 평가 결과, ZPPO는 off/on-policy distillation과 GRPO를 전 스케일에서 능가하며 특히 가장 작은 모델에서 격차가 크게 나타났습니다. 또한 distillation이 손해를 보는 학습 외 벤치마크 계열(VLM/LLM/Video 중 일부)에서도 일반화 개선이 관찰되어, “작은 모델 일반화” 문제를 실증적으로 뒷받침합니다.



### Analyzing and Encoding the Al-Mawrid Arabic-English Dictionary with the ISO Language Markup Framework and TEI Lex-0 (https://arxiv.org/abs/2606.18205)
Comments:
          44 pages, 58 figures, 12 tables. Submitted to Language Resources and Evaluation, under review since Aug 2025, round 3

- **Prior Approaches**: 기존 아랍어 사전 디지털화는 대개 단일 포맷에 종속되거나, 인쇄본의 불규칙한 구두점·구조를 그대로 옮기는 방식이 많아 기계처리 호환성이 떨어졌다. 또한 TEI/LMF 계열 가이드라인을 쓰더라도 아랍어 사전의 거시·미시 구조 모호성과 표기 불일치를 체계적으로 교정하지 못하는 한계가 지적돼 왔다.

- **Core Contribution**: 이 논문은 Al-Mawrid 아랍어-영어 사전을 ISO LMF와 TEI Lex-0 기준에 맞춰 이중 표준으로 인코딩하는 견고한 방법론을 제안한다. 편집 관점에서 거시·미시 구조를 재해석해 20세기 양언어 사전에서 흔한 구조적 모호성과 구두점 불일치를 해결하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 난제는 레거시 인쇄본의 일관성 부족을 구조화 규칙으로 흡수하는 것이며, 논문은 사전의 lexical knowledge density를 경험적으로 분석해 인코딩의 과학적 근거를 만든다. 아울러 동의어·형태-의미 속성 등의 정보 추출 규칙을 정량 평가했으며, TEI Lex-0로는 암묵적 open set 의미 관계나 흩어진 형태 단서 같은 아랍어 현상을 모델링하는 데 제약이 있음을 비교로 드러냈다.

- **Empirical Impact**: 표본으로 letter Ayn(전체의 4.6%)을 사용해 구조 파싱 정확도 91%를 보고하고, 동의어는 precision 85%, recall 98%로 높은 성능을 제시했다. 다른 morpho-semantic features는 precision 88%를 달성했으며, Prefix 기반 referencing로 Linguistic Linked Open Data(LLOD) 통합 여지도 열어 재현 가능한 레트로-디지타이제이션 워크플로를 아랍 NLP·디지털 인문학에 제공한다.



### RubricsTree: Scalable and Evolving Open-Ended Evaluation of Personal Health Agents across Health Memory and Medical Skills (https://arxiv.org/abs/2606.18203)
- **Prior Approaches**: 개인 건강 에이전트(PHA) 평가는 정답이 있는 객관식 벤치마크(예: MedQA, MedMCQA)보다는 다회·오픈엔드 생성과 도구 사용 궤적을 다뤄야 하지만, 기존 방식은 이를 충분히 관측하지 못합니다. 전문가 라벨링은 임상 정합성이 높아도 비용과 시간 때문에 스케일이 어렵고, HealthBench처럼 대규모라도 고정된 정답 세트라 제품 개발 주기의 “지속 최적화” 요구를 따라가기 힘듭니다. 반면 LLM-as-a-judge(원격 채점)는 자동화는 되지만 주관성·런투런 불일치·임상 정렬 미스가 생겨 신뢰 가능한 평가 신호가 되기 어렵다는 문제가 있습니다.

- **Core Contribution**: 이 논문은 RubricsTree라는 스케일 가능한 평가 프레임워크를 제안합니다. 100개 이상의 원자 단위(atomic)·임상적으로 검증 가능한 Boolean 루브릭을 계층형(DAG)으로 조직하고, 의료 문헌/전문가 패널이 검증한 “규칙 기반 앵커”로 판단의 주관성을 줄입니다. 또한 쿼리별로 관련 루브릭만 선택해 평가하는 context-aware adaptive router를 두어, 전문가 정합성과 자동화 효율을 동시에 노립니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 개방형 건강 응답에서 전문가 판단의 일관성과 임상 정렬을 유지하면서 (2) 대규모 자동평가로 전환할 때 생기는 비일관성과 부정확한 감점/보상을 줄이는 것입니다. RubricsTree는 루브릭의 잎(leaf)마다 구체적 임상 기준에 근거한 결정형 검증 함수를 두고, 계층 트리에서 의미적으로 필요한 하위 루브릭만 라우팅해 노이즈가 줄어든 평가를 구성합니다. 더불어 평가기를 평가하는 meta-evaluation으로 ICC3·Cohen’s kappa, 맥락 열화(지시 누락/사용자 데이터 오류/부적절 지시 등)에서의 Detection Rate와 Mean Penalty를 함께 측정해 “평가기의 신뢰성” 자체를 검증합니다.

- **Empirical Impact**: 실험에서 RubricsTree는 강력한 대규모 baseline 대비 전문가 정합성에서 큰 폭으로 개선되며, 별도 6인 전문가 패널 기준 Overall ICC3와 Cohen’s kappa가 크게 상승합니다. 또한 맥락이 망가진 상황에서 응답을 안정적으로 감점하며(oracle perturbation에서 Detection Rate 90%대+), 기존 principle baseline은 일부 셀에서 오히려 열화 응답을 더 높게 보상하는 실패가 드러납니다. 더 나아가 RubricsTree를 structured instruction·응답 최적화 피드백·RL reward로 활용했을 때 HealthBench 계열에서 모델 패밀리 전반에 걸쳐 최대 약 66% 수준의 상대 성능 개선이 관찰되어, “제품 수준 개인 건강 AI의 지속 최적화 인프라”로서 의미가 큽니다.



### Learning from the Self-future: On-policy Self-distillation for dLLMs (https://arxiv.org/abs/2606.18195)
Comments:
          Preprint

- **Prior Approaches**: 기존 OPSD는 autoregressive(AR) 중심 설계로, 프롬프트에 참조답안 같은 privileged information을 prefix로 붙이고 token-level divergence supervision을 적용한다. 이 방식은 dLLM의 임의 순서 생성과 반복적 denoising 구조에는 구조적으로 잘 맞지 않는다. 또한 dLLM에서는 한 step에 여러 마스크 토큰을 동시 예측하므로, AR을 전제한 token-level KL/다이버전스 감독은 호환성이 낮다.

- **Core Contribution**: 이 논문은 dLLM을 위한 최초의 OPSD 프레임워크인 d-OPSD를 제안한다. 핵심은 self-teacher를 “self future-experience”로 재구성하는 것인데, 학생이 생성한 정답의 suffix 정보를 교사가 보게 하여 prefix 기반의 AR식 privileged injection 문제를 우회한다. 또한 감독 신호를 token-level에서 step-level divergence로 옮겨, dLLM의 iterative denoising 과정에 맞춘 학습을 가능하게 한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) dLLM에서 privileged information을 어떤 형태로 주입할지, (2) dLLM의 step별 상태전이를 반영하는 감독 목적함수를 어떻게 정의할지였다. 저자들은 dLLM의 suffix conditioning 성질을 활용해, 학생이 만든 전체 궤적 중 일부만 남기는 방식으로 교사 입력을 구성한다. 더불어 각 denoising step에서 top-k masked 토큰 부분집합에 대해서만 step-level KL을 계산해 token-level 감독의 비호환성을 해결했으며, teacher의 retaining ratio와 top-k 선택을 함께 최적화한다.

- **Empirical Impact**: 4개 reasoning 벤치마크 실험에서 d-OPSD는 RLVR 및 SFT 기준선을 일관되게 능가하며, 특히 sample efficiency가 크게 개선됐다. RLVR 대비 약 10% 수준의 optimization steps로 수렴하는 결과를 제시해, post-training에서 더 효율적인 자기증류 경로를 보여준다. 또한 AR-style OPSD(참조답안을 prefix로 붙이는 방식)보다 d-OPSD가 더 유의미한 전이 지식을 만들며, 이 차이를 top-K 관련 지표로 실증했다.



### Unintended Effects of Geographic Conditioning in Large Language Models (https://arxiv.org/abs/2606.18124)
Comments:
          To appear at the Second Workshop on Customizable NLP (CustomNLP4U) at ACL 2026

- **Prior Approaches**: 기존 연구는 지리·인구 편향을 주로 pre-training에서 학습된 암묵적 “사전확률” 관점에서 다뤄왔습니다. 예를 들어 부·학력·추천 등에서 특정 지역에 치우치거나, 역사 서사처럼 국가 단위 내러티브에 동조하는 경향은 분석했지만, 실제 서비스에서 흔한 “추론 시점(inference-time) 메타데이터 주입”이 만드는 명시적 조건화 문제는 상대적으로 덜 다뤄졌습니다. 또한 implicit personalization처럼 추론으로 개인정보를 맞히는 경우는 봤지만, 위치가 직접 프롬프트/시스템 지시로 들어갈 때의 open-ended 생성 실패 양상은 구체적으로 정량화되지 않았습니다.

- **Core Contribution**: 이 논문은 위치 누출(location leakage)을 “위치 중립 프롬프트를 줬는데도 모델이 생성에 지리적 언급을 끼워 넣는 generative conditioning failure”로 정의하고, 이를 측정하는 평가 프레임워크를 제안합니다. WritingPrompts와 Infinite Chats 두 데이터셋에서 5개 최신 모델에 대해, 사용자 위치를 프롬프트 구조(유저 프로필 블록/시스템 프롬프트/하이브리드)에 주입했을 때 누출이 크게 늘어남을 보여줍니다. 특히 ‘Unknown’처럼 의미 없는 자리표시자를 넣어도 누출이 최대 72배까지 증가해, 위치 의미가 아니라 “프로필 프레임 자체”가 생성 신호로 작동함을 입증합니다.

- **Technical Challenges**: 명시적 위치 조건화의 영향을 보이려면, 생성 결과에서 “원래는 지역 언급을 유도하지 않았는데도” 특정 국가 지칭이 나타났는지 보수적으로 포착해야 합니다. 저자들은 국가명/어근에 대한 string matching 기반 누출 판별을 사용해 하한선(lower bound)의 누출을 측정하고, ‘No Injection’과 ‘Unknown’ 조건을 함께 둬 구조적 요인과 의미적 요인을 분해합니다. 또한 출력 필터링·재시도 규칙으로 퇴화 출력을 배제해 누출 탐지 신호를 안정화하려 했으며, 일부 모델의 안전 정책으로 샘플이 건너뛰는 경우도 지역 상관이 없도록 처리했다고 설명합니다.

- **Empirical Impact**: 실험 결과, basline 대비 누출이 최대 793배까지 치솟았고 Llama 3.1-8B는 하이브리드 조건에서 누출률 31.7%까지 관측됐습니다. 모델·주입 방식 전반에서 누출은 일관되게 나타났으며, 지역적으로는 북미와 서유럽에서 특히 강하게 발생하는 패턴이 보고됩니다. 더불어 LoRA fine-tuning으로 편향을 줄이려는 시도는 Llama에선 미미한 개선, Qwen3-8B에선 오히려 증가로 이어져 “사전학습된 구조적 결합”이 강함을 시사합니다. 서비스 관점에서는 사용자 메타데이터를 숨겨서 넣는 개인화 파이프라인의 경계 거버넌스(유용한 지역화 vs 의도치 않은 문화·지리 고정관념)를 재정의해야 한다는 문제의식을 강화합니다.



### HistoRAG: Embedding Historical Methodology in Retrieval-Augmented Generation Through Critical Technical Practic (https://arxiv.org/abs/2606.18103)
Comments:
          25 pages, 6 figures. Companion preprint to a Journal of Digital History notebook article (under review)

- **Prior Approaches**: 기존 RAG는 검색-생성 파이프라인을 매끄럽게 연결해 유사도 기반으로 가장 그럴듯한 구절을 고르고 요약을 생성하는 데 최적화돼 있다. 하지만 역사 연구처럼 해석이 핵심인 분야에서는 ‘relevance’가 고정값이 아니라 연구자의 질문, 시기, 의미론에 따라 달라지며, 소스 선택 근거가 투명하고 논쟁 가능해야 한다.

- **Core Contribution**: 이 논문은 역사학의 방법론을 RAG 아키텍처 설계로 번역한 HistoRAG를 제안한다. Heuristik(소스 발견·정리)과 Analyse(해석)를 분리하고, 시간창(windowing)으로 시기별 소스 균형을 강제하며, LLM-as-judge로 사후 평가의 기준을 명시·검토 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 시대별 어휘 변화와 주제 밀도 편향 때문에 임베딩 유사도 검색이 초기에 해당하는 소스를 잘 놓치는 ‘temporal skew’를 다루는 것, (2) 유사도는 점수 기준(정말로 질문에 relevant한가)을 직접 반영하기 어렵다는 점이다. HistoRAG는 시기 구간별로 동일 수를 회수하는 temporal windowing과, 검색 후에 연구자 정의 rubric로 근거 인용을 포함한 점수·정당화를 산출하는 평가 레이어를 결합해 이를 해결한다.

- **Empirical Impact**: Der Spiegel 1950–1979년 102,189개 기사로 SPIEGELragged 평가를 수행했으며, 표준 RAG의 결함들이 정량적으로 확인됐다. era-specific vocabulary 문제, windowing을 정당화하는 temporal skew의 존재, 그리고 vector similarity와 LLM 평가 relevance의 약한 상관(Spearman rho=0.275) 등 측정 가능한 이유로 아키텍처 개입의 필요성을 보여준다. 또한 keyword/semantic retrieval의 소스 풀 불일치를 ‘공통 LLM 평가 필터’ 아래에서 상호보완적으로 쓰는 구조로 다뤄, 해석적 학술 실무에 적용 가능한 설계 모델을 제시한다.



### Security and Privacy Prompts in the Wild: What Users Ask LLMs and How LLMs Respond (https://arxiv.org/abs/2606.18062)
- **Prior Approaches**: 기존 연구는 LLM의 S&P(디지털 보안 및 프라이버시) 답변 품질을 주로 연구자가 만든 S&P 오해(misconceptions)나 FAQ 같은 입력으로 평가해 왔다. 반면 실제 사용자가 LLM에게 던지는 S&P 질문이 “무엇인지”, 그리고 그에 대한 답변이 얼마나 신뢰할 만한지(특히 세션 간 일관성)는 잘 알려지지 않았다.

- **Core Contribution**: 이 논문은 WildChat의 실제 사용자-LLM 대화 3.2M에서 S&P 프롬프트 14,727개를 수집·분류하고, 주제 분석을 통해 사용자의 S&P 질문 유형을 정리한다. 또한 권고/가이드 요청(advice-seeking) 프롬프트 270개를 별도로 구성해 LLM 답변의 품질과 동일 질문 반복 시 일관성까지 함께 평가한다.

- **Technical Challenges**: 핵심 과제는 (1) 실제 대화에서 S&P 프롬프트를 정확히 선별하고, (2) 답변 품질을 자동 채점하면서도 평가 편향을 줄이며, (3) 모순 여부를 세션 간 일관성 지표로 측정하는 것이다. 이를 위해 다중 LLM majority voting으로 S&P를 분류(정밀도 96%/재현율 74%)하고, checklist 기반 LLM-as-judge로 품질을 채점하되 self-preference 편향은 여러 채점 모델 점수 평균으로 완화했으며, 일관성은 답변 근거 문장(체크리스트 항목별 evidence quotes)을 NLI entailment으로 모순 가능성에 초점을 맞춰 계산한다.

- **Empirical Impact**: 결과적으로 상용 LLM이 오픈웨이트 모델보다 평균 품질이 높았지만, 평균 품질이 좋은 프롬프트에서도 반복 실행 간 상충 답변이 일부 발생해 사용자를 혼란시킬 위험이 관찰됐다. 특히 Llama 4는 평균 품질은 가장 낮았으나 반복 일관성은 가장 높았고, “품질만으로는 S&P 신뢰성을 규정할 수 없다”는 메시지를 실증적으로 뒷받침한다.



### ConSA: Controllable Sparsity in Hybrid Attention via Learnable Allocation (https://arxiv.org/abs/2606.18056)
- **Prior Approaches**: 하이브리드 attention(FA+SWA)은 추론 비용 병목을 줄이려는 대표적 접근이지만, 기존에는 Mistral·Gemma 2 같은 방식처럼 FA/SWA 배치를 hand-crafted 규칙으로 정하는 경우가 많습니다. 이 규칙은 원 모델의 layer/ head별 상이한 attention 거동을 충분히 반영하지 못해, sparsity 목표가 바뀌면 효과가 흔들릴 수 있습니다. 또한 LoZA처럼 간단한 scalar 게이트를 보정해 FA/SWA를 사후 선택하는 방법은 제한된 데이터로 layer 점수를 매기다 보니 제약(sparsity)과의 정합성이 약해질 수 있습니다.

- **Core Contribution**: 이 논문은 Controllable Sparsity in Hybrid Attention(ConSA)로, 사용자가 지정한 SWA 비율(sparsity target) ρ에 맞춰 FA/SWA 할당을 “학습으로” 찾아내는 프레임워크를 제안합니다. attention unit마다 FA 또는 SWA를 고르는 이진 마스크를 L0 regularization으로 학습하고, augmented Lagrangian 제약으로 목표 ρ를 최적화 과정에서 직접 강제합니다. 또한 layer-wise와 KV-head-wise 두 수준의 granularity에서 제약을 걸어, 어떤 분해가 더 좋은지까지 함께 다룹니다.

- **Technical Challenges**: 핵심 기술 난점은 이진 선택(FA/SWA)이 비미분이라 그래디언트 학습이 어렵고, 동시에 realized sparsity가 정확히 ρ에 수렴해야 한다는 점입니다. ConSA는 hard concrete 분포로 이진 마스크를 학습 가능하게 만들고, augmented Lagrangian의 선형 항+이차 패널티로 제약을 안정적으로 만족시키는 구조를 사용합니다. 그 뒤 학습이 수렴하면 마스크를 binarize해 고정하고, 이후 continued pre-training으로 가중치가 새 배치에 적응하도록 설계합니다.

- **Empirical Impact**: 0.6B와 1.7B 두 모델에서 rule-based baseline 대비 head-wise ConSA가 일관되게 더 높은 성능을 보였고, 특히 KV-head-wise 할당이 layer-wise보다 뚜렷한 이득을 냈습니다. learned 패턴은 SWA가 아래쪽 layer에 몰리고 FA는 중간 layer에 contiguous block으로 집중되는 형태로 나타나, 균일 interleaving에 가까운 규칙 기반과 결이 다릅니다. 또한 이 구조는 모델 스케일·sparsity 수준·할당 granularity가 바뀌어도 유지되며, FA/SWA의 근본이 “retrieval-vs-streaming”의 단일 분류를 넘어서는 더 세밀한 attention spike spectrum과 연결됨을 실증적으로 보여줍니다.



### Compositional Skill Routing for LLM Agents: Decompose, Retrieve, and Compos (https://arxiv.org/abs/2606.18051)
- **Prior Approaches**: 기존 연구는 skill routing을 주로 단일 스킬 선택(또는 단계별로 독립 선택) 문제로 취급해 왔습니다. 즉, 여러 스킬을 순차적으로 분해·조합해야 하는 현실 과제를 충분히 다루지 못했고, 분해(단위 작업의 개수/정확도)와 검색·조합 성능이 함께 어떻게 맞물리는지도 명확히 최적화하지 못했습니다. 또 대규모 도구/스킬 풀에 대해선 벤치마크나 평가가 고정 도구셋 중심이어서, compositional routing의 병목을 정밀 진단하기 어려웠습니다.

- **Core Contribution**: 이 논문은 Compositional Skill Routing을 정식 문제로 정의하고, 복잡한 질의를 원자적 하위 작업으로 분해한 뒤 각 하위 작업에 맞는 스킬을 찾아 순서화된 실행 계획(DAG)을 만드는 프레임워크를 제안합니다. SkillWeaver는 Decompose-Retrieve-Compose 3단계를 캐스케이드로 결합하되, 특히 분해-검색 구간에서의 피드백을 강조합니다. 또한 compositional routing 전용 벤치마크 CompSkillBench(300개 질의, 2,209개 MCP 스킬, 24개 기능 카테고리)를 공개 생태계 기반으로 구축해 체계적으로 비교 가능하게 했습니다.

- **Technical Challenges**: 핵심 기술적 난관은 “올바른 분해 단위(스텝 수·granularity)”가 검색 품질을 좌우하는데, LLM 분해가 스킬 메타데이터와 잘 맞지 않아 vocabulary 불일치가 발생한다는 점입니다. 저자들은 Iterative Skill-Aware Decomposition(SAD)로 이를 해결하는데, 초기 분해 결과로 후보 스킬을 검색한 뒤 그 검색 힌트를 분해 입력에 다시 반영해 분해를 스킬 라이브러리의 어휘에 정렬합니다. 실험상 한 번의 SAD 반복만으로도 분해 정확도가 크게 개선되며, DA(정확한 구조) 조건에서 검색 성능이 본격적으로 드러나는 패턴을 확인했습니다.

- **Empirical Impact**: CompSkillBench에서 vanilla 분해는 CatR@1이 34.2%에 그쳤고, DA±1은 71.3% 수준이었습니다. SAD는 분해 정확도(DA)를 51.0%→67.7%(+32.7%, p<1e-6)로 끌어올리며, DA=1 조건에서 CatR@1이 41.2%로 상승해 “분해 granularity가 검색의 관문”임을 실증했습니다. 또한 SkillWeaver는 2,209개 스킬 전체를 매번 보는 비용을 99% 이상 줄이면서도, 타깃 카테고리가 검색 풀에 없더라도 transfer에서 상대 DA 이득(+35.6%)을 유지하는 등 실제 에이전트 운영 관점의 의미가 큽니다.



### When English Isn't the Best Teacher: Source Language Effects in Cross-Lingual In-Context Learning (https://arxiv.org/abs/2606.18033)
Comments:
          Accepted at 1st Workshop on Multilinguality in the Era of Large Language Models (MeLLM 2026), co-located with ACL 2026

- **Prior Approaches**: 기존 교차언어 ICL 연구는 소스 언어 선택이 성능에 중요하다고 보면서도, 성과를 좌우하는 요인이 fine-tuning 시대의 직관(언어 유사성, 표면 중첩 등)과 동일하다고 가정하곤 했습니다. 또한 검색(retrieval)이나 혼합 언어/코드스위칭 같은 프롬프트 구성 변수가 성능을 바꿀 수 있으나, 소스 언어를 어떻게 체계적으로 고를지에 대한 평가는 제한적이었습니다. 특히 generative 태스크에서는 원하는 언어로 출력이 생성되지 않는 language confusion이 핵심 장애물이지만, 이를 소스 선택 문제와 함께 넓게 다룬 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 7개 태스크, 6개 모델, 유형적으로 다양한 언어 집합에서 소스-타깃 조합을 전수에 가깝게 비교해, 교차언어 ICL에서 “fine-tuning의 기대가 그대로 전이되는가”를 실증적으로 검증합니다. 그 결과, 타깃 언어가 자기 자신에게서 항상 가장 잘 전이되는 것이 아니며(약 24%), 영어는 역으로 소스로서 최악인 경우가 적지 않음을 보여줍니다. 아울러 언어 유사성이 ICL 전이에 일관되게 강한 예측력을 갖지 못하고, 대신 모델 내부에서의 cross-lingual alignment 같은 대표성 기반 요인이 더 중요함을 제시합니다.

- **Technical Challenges**: 핵심 기술적 난점은 소스-타깃 조합 평가가 조합 폭발을 일으키는 점과, generative 평가에서는 language confusion 때문에 전이 품질을 측정할 출력 자체가 없을 수 있다는 점입니다. 저자들은 유형적으로 다양한 18개 언어를 골라 7개 병렬 벤치마크에서 test를 제한(각 1,000개 또는 전체)하고, aligned 예시로 콘텐츠 혼선을 줄여 전이 역학만 분리합니다. 언어 혼동은 Language Confusion Benchmark와 line-level Pass Rate(LPR)로 별도 진단해, 소스 선택이 출력 언어에 미치는 영향을 정량화합니다.

- **Empirical Impact**: 실험 전반에서 타깃을 잘하는 고자원 언어(예: 영어/스페인어/독일어/이탈리아어)가 소스로는 약한 경향이 나타나며, 언어 유사성보다는 모델 표현공간의 cross-lingual alignment가 성능 분산을 더 잘 설명합니다. 특히 low-resource이면서 non-Latin script인 언어가 소스로서 가장 유효한 경우가 많고, 고자원 Latin-script는 상대적으로 불리하다는 패턴이 통계적으로도 확인됩니다. 또한 language confusion은 태스크별 전이 성능과 항상 직접 연결되진 않지만, 소스-타깃 간 “donor/recipient” 대칭성을 유사하게 보이며, ICL에서 소스 언어 선택을 재설계해야 함을 시사합니다.



### VoidPadding: Let [VOID] Handle Padding in Masked Diffusion Language Models so that [EOS] Can Focus on Semantic Termination (https://arxiv.org/abs/2606.17999)
- **Prior Approaches**: 기존 MDLM(마스킹 디노이징 기반 대규모 언어 모델)은 고정된 response canvas를 잡아두고 잡음 제거로 텍스트를 생성하며, 여기서 response-length modeling이 instruction tuning의 핵심이 됩니다. 많은 방법이 autoregressive 관례를 따라 padding을 위해 반복된 [EOS] 토큰을 사용해 [EOS]가 의미적 종료자이자 padding 토큰이라는 ‘이중 역할’을 갖게 됩니다. 그 결과 큰 block decoding에서 [EOS] overflow(비정상적으로 EOS가 과도하게 발생)가 취약점으로 드러났습니다.

- **Core Contribution**: 이 논문은 [EOS]의 이중 역할이 대규모 블록 디코딩에서 [EOS] overflow의 근본 원인이라고 지목합니다. 이를 분리하기 위해 padding에는 [VOID]를 도입하고 [EOS]는 종료(termination) 신호로만 남기는 VoidPadding을 제안합니다. 추론에서는 학습된 [EOS]로 early stopping을 수행하고, 학습된 [VOID]로 필요한 만큼 response canvas를 adaptive하게 확장합니다.

- **Technical Challenges**: 문제는 [EOS]가 단순 종료 토큰이 아니라 길이/패딩 정보를 동시에 품으면서, 블록 단위 생성 과정에서 종료 판정이 왜곡된다는 점입니다. 저자들은 [EOS]와 padding 신호를 명시적으로 분리해 termination과 length 제어를 서로 다른 학습 신호로 분리하는 방식으로 해결합니다. 또한 [VOID] 신호가 실제로 canvas 확장에 유효하도록 모델이 해당 신호를 길이 추정에 활용하도록 학습·추론 절차를 설계합니다.

- **Empirical Impact**: Dream-7B-Instruct에서 VoidPadding은 block-size-averaged 네 가지 태스크 평균을 원본 대비 +17.84점, RainbowPadding 대비 +6.95점 개선했습니다. 동시에 decoding NFE는 평균 55.7% 감소해 생성 효율도 함께 좋아졌습니다. 수학적 추론과 코드 생성 벤치마크 전반에서 견고한 개선을 보이며, instruction tuning에서 padding 설계가 길이 모델링 성능에 미치는 영향을 재부각했다는 점에서 의미가 큽니다.



### Fine-tuning LLMs for Passive Depression Severity Estimation from AI Mental Health Dialogu (https://arxiv.org/abs/2606.17973)
Comments:
          12 pages, 1 figure

- **Prior Approaches**: 기존 PHQ-9 예측 연구는 임상 인터뷰나 스크립트가 있는 평가(예: DAIC-WOZ, E-DAIC)에서 나온 언어를 활용하는 경우가 많아, 실제 ‘비지도 대화’에서는 성능이 깨질 수 있다는 한계가 지적돼 왔습니다. 자연스러운 대화/다이어리 텍스트를 쓰더라도 레이블 수가 적고(대개 수십~수백 명), PHQ-9 전체(특히 9번 문항: 자살사고)를 그대로 다루는 경우는 드뭅니다.

- **Core Contribution**: 이 논문은 AI 정신건강 앱과의 멀티턴 대화 전사(transcript)만으로 PHQ-9 총점을 직접 회귀 예측하는 패시브(수동) 모니터링 가능성을 제시합니다. 또한 PHQ-9 >=10 같은 단일 cutoff가 아니라, PHQ-9 전 임상 스펙트럼(>=3~>=24)에서의 구분력까지 함께 검증해 임상적으로 더 쓰임 있는 “연속/다단계” 추정 관점을 강화했습니다.

- **Technical Challenges**: 핵심 과제는 (1) 실제 환경에서 PHQ-9 자가보고 레이블이 불완전해지는 체계적 결측과 (2) 대화만 보고 임상 점수(0~27)를 정밀 추론해야 한다는 점입니다. 이를 위해 Qwen3.5-27B에 회귀 헤드를 붙인 뒤, Claude Opus로 의사라벨을 만들고(레이블 불균형 보정), 중간 모델로 추가 의사라벨을 반복 확장(iterative pseudo-labeling)하며, 마지막에는 여러 모델의 예측을 앙상블해 스퓨리어스 신호에 덜 민감하게 만들었습니다.

- **Empirical Impact**: 842명 홀드아웃 테스트에서 MAE 2.6, RMSE 4.0, Pearson r=0.80, PHQ-9 >=10 임상 임계에서 AUC 0.91을 달성했습니다. 더 나아가 모든 심각도 임계값에서 AUC가 0.87 이상으로 유지돼(>=3~>=24) 경증부터 중증까지 일관된 감도/구분력을 보였다는 점이 주목됩니다. 결과적으로 설문 참여 없이도 증상 변화를 연속적으로 추적하거나 악화 조짐을 조기 감지하는 ‘현장형’ AI 정신건강 모니터링 방향에 실증 근거를 제공합니다.



### Learning task-specific subspaces via interventional post-training of speech foundation models (https://arxiv.org/abs/2606.17967)
Comments:
          Accepted to Interspeech 2026; 6 pages (4 main body), 2 figures

- **Prior Approaches**: Speech foundation model은 대규모 비라벨 음성으로 사전학습되어 여러 과제에 재사용 가능한 표현을 만든다는 점에서 주목받아 왔습니다. 다만 이 표현은 핵심 음성 변수를 분산된 형태로 인코딩해, 하위 과제에서 필요한 특정 변동성만 쓰는 상황과 어긋날 수 있습니다.

- **Core Contribution**: 본 논문은 speech foundation model의 사전학습 표현을 사후(post-training)로 다듬는 refinement 방법을 제안하며, 이를 interventional contrastive learning으로 구현합니다. interventional dataset와 multi-part contrastive loss를 활용해, 얽혀 있는(representations entangled) 공간을 content(콘텐츠)와 speaker(화자) 하위 공간으로 분리하는 변환을 학습합니다.

- **Technical Challenges**: 핵심 난제는 분리된 하위 요소가 실제로는 잡음과 함께 분산 인코딩되어 있어, 단순 분류형 학습만으로는 subspace 분리를 보장하기 어렵다는 점입니다. 논문은 얽힘(entanglement)을 줄이기 위해 interventional dataset으로 변동 요인을 제어하고, 여러 항의 contrastive loss를 결합해 content와 speaker에 대응되는 표현을 서로 다른 방향으로 당기도록 설계했습니다.

- **Empirical Impact**: 실험에서는 speaker verification과 keyword spotting에서 학습된 표현이 유의미한 성능 향상을 보였고, 특히 out-of-domain speaker verification에서 개선이 확인됐습니다. 또한 결과는 speaker와 content 정보가 학습된 subspace에서 분리된다는 증거로 이어져, speech representation 학습의 해석 가능성과 전이 성능을 함께 끌어올렸다는 의미가 있습니다.



### ChLogic: Evaluating Robustness of Logical Reasoning in Chinese Expressions (https://arxiv.org/abs/2606.17905)
- **Prior Approaches**: 기존 논리 추론 데이터셋들은 if, only if, unless 같은 연결 표현이 비교적 명시적이거나 영어 중심의 템플릿형 문장에 치우치는 경우가 많습니다. 그래서 모델이 기준선(standard) 조건에서는 잘 맞혀도, 중국어에서 생기는 생략, 수사적 표현, 관용구, 다의성, 화용적으로 우회된 논리 관계까지 동일 구조로 복원하는지 확인하기 어렵습니다. 또한 다국어 견고성 평가는 있어도, 같은 잠재 논리 템플릿을 영어-중국어 표면만 바꿔가며 라벨을 고정하는 정밀 진단은 드뭅니다.

- **Core Contribution**: 이 논문은 영어-중국어 정렬 벤치마크 ChLogic을 제안해, 같은 잠재 논리 구조가 영어와 다양한 중국어 표면에서 표현될 때 추론 라벨이 유지되는지 테스트합니다. General aligned, Difficult aligned, Chinese-only로 나뉘며, 각 항목은 영어 기준 표현 1개에 중국어 실현 5개(표준/문어/구어/수사적 질문/교란)를 페어링합니다. 이를 통해 ‘논리 구조를 복원하는 단계’와 ‘형식적 추론’이 분리되어 실패하는 양상을 진단할 수 있게 합니다.

- **Technical Challenges**: 핵심 난제는 중국어 표면 실현이 논리 스코프(부정/양화), 조건 마커(예: 只要 vs 只有, 除非 vs 否则), 수사적 질문, 문맥 의존 생략 등을 통해 논리형식을 흐리게 만들 수 있다는 점입니다. 논문은 템플릿-기반으로 금 정답 라벨과 템플릿 논리 구조를 먼저 고정하고, LLM을 표면 생성에만 보조적으로 사용한 뒤 의미 정렬과 라벨 보존을 품질관리로 강제합니다. 또 중국어 전용 현상 그룹을 별도로 구성해 번역만으로는 드러나지 않는 ‘표면-논리 정규화’ 실패를 스트레스 테스트합니다.

- **Empirical Impact**: Qwen3, Ministral, GLM에서 영어 성능은 매우 높게 유지되지만 중국어 변형(특히 수사적/교란)에서 일관된 성능 격차가 관측됩니다. 예를 들어 GLM-5.1은 General aligned에서 영어 98%대에서 수사적 중국어 변형으로 크게 하락하며, Difficult aligned에서는 더 급격한 하락이 나타납니다. 백번역은 General aligned에서는 종종 개선을 보이지만 Difficult aligned와 중국어 전용 현상에서는 혼재된 결과(악화 포함)를 보여, 오류가 단순한 추론 부족이 아니라 중국어 표면 단서의 정규화 실패와 번역 아티팩트, 그리고 모델별 의사결정 특성이 함께 작동함을 시사합니다.



### Dynamic Rollout Editing for Reducing Overthinking in RL-Trained Reasoning Models (https://arxiv.org/abs/2606.17890)
Comments:
          21 pages, 10 figures, 2 tables

- **Prior Approaches**: 기존 연구는 overthinking을 주로 생성 길이(언제 멈출지, 얼마나 길게 생각할지) 제어 문제로 보고, early-exit 디코딩·stopping controller·length regularization 같은 방식으로 해결을 시도해 왔습니다. 하지만 이런 접근은 대체로 정책이 어느 정도 학습된 뒤의 길이/종료 시점에 개입하며, RL 학습 과정에서 “정답이 이미 보인 뒤의 불필요한 사고”가 어떻게 보상 신호에 섞여 강화될 수 있는지는 덜 다뤘습니다. 특히 GRPO처럼 sequence-level credit을 쓰는 경우, 정답을 이끈 prefix와 정답 이후의 불필요한 continuation이 함께 이득을 공유할 여지가 남아 있었습니다.

- **Core Contribution**: 이 논문은 GRPO-style post-training에서 overthinking after answer emergence를 “학습 시점 credit-assignment 문제”로 재정의합니다. answer emergence를 parser-verifier로 조기에 관측해 보면, 같은 프롬프트에서 성공 롤아웃이 실패 롤아웃보다 정답이 verifiable해진 뒤의 과도한 검증/반복 경향이 더 커지며, GRPO의 group-relative advantage가 이 결합을 더 키울 수 있음을 보입니다. 이를 해결하기 위해 Dynamic Rollout Editing(DRE)을 제안하며, 정답을 이끈 verified prefix는 보존하되 이후 thinking을 편집한 대안을 같은 RL 그룹 내에서 더 선호하도록 학습 신호를 바꿉니다.

- **Technical Challenges**: 핵심 난점은 길이 자체를 억제하는 방식이 아니라, “정답 도달에 필요한 reasoning”은 남기고 “정답이 나온 뒤의 불필요한 연장”만 약화시키는 preference를 학습 신호로 분리하는 것입니다. DRE는 정답 emergence 경계 k⋆를 단순 편집 지점으로 쓰지 않고, k⋆부터 <Final Answer> 클로저를 생성해 verifiable하게 닫히는 보수적 editable boundary k^hat를 찾은 뒤, 그 뒤 thinking을 잘라 재생성(regenerate)합니다. 또한 편집된 토큰이 클리핑으로 인해 학습 신호가 사라지지 않도록 GClip 같은 학습 대리 신호와 prefix masking을 결합해, shared verified prefix에 대한 부정적 크레딧과 상한 클리핑 문제를 동시에 완화합니다.

- **Empirical Impact**: 수학(AIME24/25/26), 과학 QA(GPQA-D), 코드 생성(LiveCodeBench V6) 등 다양한 벤치마크에서 DRE는 thinking 토큰 수를 크게 줄이면서 정확도는 유지하거나 개선하는 효과를 보입니다. GRPO 대비 전반적으로 answer 이후 구간의 반복 검증(answer revisits, redundant verification)이 더 많이 감소하며, 단순 length penalty(GRPO + LP)로는 같은 패턴이 재현되지 않았습니다. 또한 편집-검증 절차의 수용률이 87.20%로 높아, DRE의 preference 신호가 소수 샘플이 아닌 대부분의 롤아웃에서 실질적으로 작동함을 확인합니다.



### GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine? (https://arxiv.org/abs/2606.17861)
- **Prior Approaches**: 기존 게임 생성 벤치마크들은 부분 계약에만 집중하는 경향이 있습니다. OpenGame-Bench는 완결된 게임을 요구하지만 웹 게임 중심이며 실제 플레이 상호작용 기반 판정이 약합니다. GameDevBench는 Godot에서의 실행성/환경 근접성을 보이지만 튜토리얼 편집처럼 국소 수정과 정적 테스트 비중이 커 end-to-end 완성 및 재생 확인을 충분히 담지 못합니다.

- **Core Contribution**: 이 논문은 end-to-end 게임 생성을 “명세→게임 엔진에서 실행되는 완결 아티팩트→플레이 상호작용으로 검증”의 문제로 재정의합니다. 그리고 Engine Grounding, Artifact Completeness, Interactive Verification 3가지를 함께 만족해야 제대로 된 벤치마크라고 정리합니다. 이를 구현한 상호작용 기반 평가 프레임워크와 Godot 기반 벤치마크 GameCraft-Bench(140개 Godot 태스크, 15개 game family)를 제안합니다.

- **Technical Challenges**: 기여의 핵심 난점은 게임이 ‘코드가 맞는가’가 아니라 ‘플레이해도 의도한 반응이 나오는가’로 평가돼야 한다는 점입니다. 저자들은 실행 가능한 Godot 프로젝트 제출과 함께 재생 가능한 replayable interaction traces를 강제해, 검증 단계에서 실제 입력-응답을 재현하도록 설계했습니다. 추가로 hidden rubric를 핵심 메커닉스/콘텐츠 깊이/기능적 비주얼/아트·프레젠테이션으로 분해하고, 멀티모달 judge가 재생 비디오·프레임을 기준으로 채점합니다.

- **Empirical Impact**: 프론티어 코딩 에이전트를 GameCraft-Bench에 시험한 결과, 최상위(Claude Code의 Opus-4.7 high)도 전체 41.46%에 그쳤고 대부분 40% 이하였습니다. 에이전트들은 종종 인지 가능한 메커닉스를 구현하지만, 충분한 콘텐츠(Content Depth)와 시각적 피드백(Functional Visuals), 전반적 프레젠테이션 완성도(Art and Presentation)에서 일관되게 부족했습니다. 또한 build pass는 높아도 valid-trace 비율이 낮은 사례(예: DeepSeek-V4-Pro)가 있어, “완결 아티팩트+재생 확인 증거”를 동시에 닫는 것이 아직 큰 장벽임을 보여줍니다.



### Environment-Grounded Automated Prompt Optimization for LLM Game Agents (https://arxiv.org/abs/2606.17838)
- **Prior Approaches**: LLM 에이전트를 게임/보행 같은 interactive 환경에 투입하면 프롬프트에 대한 민감도가 커서, 기존에는 task별 수작업 prompt engineering이나 monolithic agent 설계가 주로 쓰였다. 또한 가중치 fine-tuning/RL 업데이트로 적응하는 접근도 있으나 비용이 높고, 데이터셋 고정 평가에 맞춰 과적합 위험이 생길 수 있다.

- **Core Contribution**: 이 논문은 LLM agents의 observation-to-action 파이프라인을 goal-conditioned descriptor agent와 action selection agent로 분해한 뒤, 환경 보상으로 유도되는 LLM 기반 evolutionary prompt optimization으로 두 모듈 프롬프트를 자동 개선한다. RAPOA는 파인튜닝 없이 프롬프트만으로 성능을 끌어올리는 것을 목표로 하며, episode return을 최적화 신호로 사용해 dataset-고정 문제를 피한다.

- **Technical Challenges**: 핵심 난제는 멀티모듈 프롬프트가 결합된 상태에서 어떤 구성요소가 실패를 유발하는지 식별하고(credit assignment), 그 원인에 맞춘 ‘표적 수정’을 제안·검증하는 것이다. 이를 위해 Behavior Analyzer로 실패/성공 궤적을 근거로 책임 모듈과 수정 방향을 랭킹·제시하고, Mutator가 targeted revision을 생성한 뒤 환경 rollouts로 two-stage acceptance test(선택 압력 조절 포함)까지 통과한 변경만 채택한다.

- **Empirical Impact**: BALROG의 BabyAI 5개 태스크 전체에서 최적화는 모든 조건에서 일관되게 성능을 개선했으며, BabyAItextCleanLangWrapper 기반 보상 설정에서도 가중치 업데이트 없이 효과가 나타났다. 특히 PutNext에서 RobustCoTAgent는 0% 성공이었는데, 동일한 underlying LLM을 사용하되 optimized prompts로 최대 72.5% 성공률까지 도달했으며, 이는 decomposition+자동 prompt optimization 조합의 실용성을 강하게 시사한다.



### Perceptual compensation for tonal context in self-supervised speech models (https://arxiv.org/abs/2606.17835)
Comments:
          Accepted for publication at Interspeech 2026

- **Prior Approaches**: 기존 연구들은 wav2vec2.0 같은 self-supervised learning(SSL) 음성 모델이 사전학습만으로도 음운·음소 범주에 대한 구조를 암묵적으로 학습하는지에 주목해 왔다. 특히 pre-trained 모델에서도 phonological structure 민감성이 나타난다고 보고되어, 이를 phonetic-only 관점이나 맥락 예측만으로 PC(Perceptual compensation)가 형성될 수 있다는 해석으로 연결해 왔다. 다만 대부분의 증거가 다른 음운 단서(예: r/l 같은 segment 대비)나 특정 평가 세팅에 의존해, 실제로 인간 청자의 PC와 같은 형태인지에 대한 검증이 제한적이었다.

- **Core Contribution**: 이 논문은 만다린 중국어 성조(lexical tone)에서 PC가 나타나는 정도를 wav2vec2.0의 pre-trained(PT) vs fine-tuned(FT) 표현으로 비교하는 pseudo-replication을 수행한다. 핵심은 인간의 성조 지각에서 관찰되는 ‘고정된 기준선(no-ctx) 대비 맥락에 따른 범주 경계 이동’이 모델 표현에서도 재현되는지 검사하는 것이다. 이를 통해 “unsupervised contextualization은 되는데, human-like compensation은 안 될 수 있다”는 관점 차이를 정량적으로 제약한다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 인간 실험을 모델 표현으로 옮기면서도, 맥락 효과와 기준선 편향을 혼동하지 않는 것이다. 연구진은 14-step 성조 연속체를 만들고, 여러 화자의 disyllable 자극을 재합성해 determinism 문제를 완화한 뒤 embedding similarity와 probing classifier(선형/MLP 로지스틱 기반) 두 방법으로 PT·FT·레이어별 반응을 측정했다. 그 결과 PT에서는 어떤 레이어에서도 성조 맥락에 대한 compensation 조짐이 거의 없었고, FT에서는 맥락 민감성은 있으나 no-context 기준선에 ‘상대적으로’ 이동하는 인간형 패턴이 약하거나 관찰되지 않았다.

- **Empirical Impact**: 실험은 embedding similarity에서 PT가 맥락 민감성이 거의 없음을 보여주며, probing에서는 FT와 일부 레이어에서만 약한 PC 징후가 나타나지만 인간 성조 지각 곡선과는 단절된 결과를 낸다. 특히 isolated syllable 조건에서 FT의 T3/T4 분류는 인간과 크게 다르게 나타나, ‘성조 범주가 더 잘 드러난다’는 신호와 ‘인간과 같은 지각적 보정이 된다’는 신호가 분리될 수 있음을 시사한다. 전체적으로, 성조 영역에서는 unsupervised 사전학습만으로는 human-like PC가 자동으로 성립하기 어렵고, supervised fine-tuning 등 추가 학습 기제가 안정적인 음운 범주 추상화를 유도해야 한다는 결론에 힘을 실어 준다.



### When Multiple Scripts Matter: Evaluating ASR in Clinical Settings (https://arxiv.org/abs/2606.17826)
Comments:
          Interspeech 2026

- **Prior Approaches**: 기존 임상 ASR 평가는 한 문장(단일 레퍼런스)만을 기준으로 WER/CER 같은 문자열 기반 지표를 계산해, 동일 발음이라도 철자(문자표기)가 다른 ‘유효한 표기 변형’을 오류로 과소평가하는 문제가 있었습니다. 다국어 ASR 연구는 code-switching(언어 음향 교체) 모델링과 데이터 증강에 주로 집중했지만, 임상에서 나타나는 multiscript variability(정서법 변형으로 인한 many-to-one 대응) 자체를 공정하게 평가하는 벤치마크는 부족했습니다.

- **Core Contribution**: 이 논문은 임상 도메인에서의 multiscript variability에 특화된 벤치마크 MultiClin을 제안하고, 의료 용어가 로마자/현지 스크립트 음차 표기로 동시에 존재하는 상황을 다중 정답으로 평가합니다. 또한 동적 multi-reference 평가가 기존 single-reference 평가보다 ASR 성능을 더 공정하고 현실적으로 반영한다는 점을 실험으로 보여줍니다.

- **Technical Challenges**: 핵심 난제는 ‘시간 정렬이 어긋난 예측’과 ‘철자 변형’이 결합될 때, 어떤 구간을 해당 용어 비교의 대상으로 삼을지 안정적으로 결정하는 것입니다. 이를 위해 논문은 추적 커서로 예측에서 윈도우(50자)를 잡고 Longest Common Substring(LCS)로 엔티티 구간을 정렬한 뒤, 해당 경계 내에서 local CER/WER을 계산하는 localized evaluation 알고리즘을 설계합니다.

- **Empirical Impact**: 여러 ASR 모델(Whisper, Qwen3 ASR, Gemini)에서 single-label(original) 평가에서 multiscript-aware(both) 평가로 바꾸면 오류율이 일관되게 크게 감소하며, 예를 들어 Gemini 2.5 Pro는 WER이 28.28%→15.78%로 줄었습니다. 학습에서는 script unification(100% transliteration ratio)이 가장 좋은 성능을 보였고, 특히 50% 매핑에서는 엔트로피가 커지는 비일관성 효과가 관측되어 수렴을 방해한다는 해석까지 제시했습니다. 결과적으로 MultiClin과 multiscript-aware 평가가 임상 ASR의 ‘진짜 역량’을 드러내는 공정한 측정 틀로 자리잡을 가능성을 보여줍니다.



### Improving low-resource ASR using bilingual fine-tuning with language identification: a cross-linguistic evaluation (https://arxiv.org/abs/2606.17820)
- **Prior Approaches**: 기존 저자원 ASR 개선은 cross-lingual transfer로 접근했지만, 성능 향상이 주로 고자원 언어에 치우치는 한계가 남아 있었습니다. 언어 정보를 쓰는 LID 방식은 외부 LID로 인한 지연(latency)과, 모델 내부에 LID를 어떻게 녹일지의 불명확성이 문제로 지적돼 왔습니다. 최근에는 LID 토큰을 fine-tuning에 prepended하는 식의 결과가 있었지만, 다양한 언어 계통/문자 체계/언어 유사도에서 항상 이득인지에 대한 체계적 검증이 부족했습니다.

- **Core Contribution**: 이 논문은 bilingual fine-tuning에 explicit LID token을 결합했을 때, 어떤 조건에서 저자원 ASR이 개선되는지 9개 언어쌍(5개 계통, 다양한 문자 체계)으로 정리했습니다. 학습에서는 언어 식별 토큰을 문장 앞에 붙여 두 언어를 구분하고, 추론에서는 모델이 음성만으로 LID와 전사(transcription)를 함께 예측합니다. 또한 추론 시 LID token을 제공하는 후속 실험을 통해 ‘LID 정확도 병목’이 실제로 성능을 좌우하는지 확인합니다.

- **Technical Challenges**: 핵심 기술 난제는 LID가 잘못되면 ASR 출력이 크게 흔들릴 수 있다는 점이며, 따라서 LID와 전사의 joint 예측이 실제로 안정적으로 작동하는 조건을 찾아야 합니다. 저자들은 XLS-R 1B 기반으로 feature encoder는 freeze하고 transformer만 fine-tuning했으며, 학습·추론에서의 LID 처리 방식(추론에서 token 미제공 vs 제공)을 비교해 원인을 분리했습니다. 더불어 LID를 추론에서 직접 조건으로 넣기 위해 CTC logits에 language-specific bias embedding을 더하는 방식도 제안·실험했습니다.

- **Empirical Impact**: 결과적으로 ΔΔWER(단일언어 WER-양언어 WER)는 다소 언어쌍에 따라 엇갈리지만, 전반적으로 LID accuracy가 높을수록 개선 폭이 커지는 양의 상관관계가 관찰됐습니다. 특히 LID accuracy가 95% 미만인 경우에는 오히려 손해(ΔΔWER 음수)도 나타나며, 잘못된 LID를 받은 샘플의 WER이 현저히 높다는 점이 병목을 뒷받침합니다. 추론 시 LID token을 제공했을 때는 LID가 낮거나(예: DA-SV, SK-CS) LID 오분류 샘플과 정분류 샘플의 WER 격차가 큰 경우 성능이 추가로 개선되어, 파이프라인에서 LID 신뢰도가 낮을 때의 실용적 대안이 제시됐습니다.



### The Slop Paradox: How Synthetic Standardization Erodes Clinical Uncertainty and Cross-Modal Alignment in AI-Rewritten Radiology Reports (https://arxiv.org/abs/2606.17791)
- **Prior Approaches**: 기존 연구들은 방사선 리포트 생성 품질을 BLEU나 진단 정확도 같은 지표로 주로 평가했지만, 생성 과정에서 “무엇이” 사라지는지 정보 손실을 체계적으로 측정하진 못했다. 불확실성(hedging) 언어의 중요성은 알려져 있으나, LLM 리라이팅이 이 언어를 보존하는지 파괴하는지 정량화된 사례는 드물었다. 또한 BiomedCLIP처럼 이미지-텍스트 정렬을 학습하는 멀티모달 모델은 텍스트 품질이 좋다는 가정 하에 정렬을 최적화했지만, 합성 리포트가 정렬을 얼마나 망가뜨리는지는 계량되지 않았다.

- **Core Contribution**: 이 논문은 IU Chest X-Ray 450장을 대상으로, LLM으로 방사선 리포트를 세 가지 현실적 리라이팅(EHR summarization, standardized rewriting, teaching case preparation)으로 변환한 뒤 정보 열화(information degradation)를 “통제 실험” 형태로 측정한다. 엔티티 소실(entity erosion), 임상적 불확실성 언어 붕괴(hedging collapse), 이미지-텍스트 정렬 저하(cross-modal alignment degradation)를 동시에 계량해, 텍스트 품질 저하가 멀티모달 정렬과 어떻게 엇갈리는지 보인다. 핵심 결론은 정보 손실의 크기와 크로스모달 충실도 저하가 분리(dissociation)된다는 점이다.

- **Technical Challenges**: 어떤 리라이팅이 임상 내용과 정렬을 각각 얼마나 망가뜨리는지 측정하기 위해, scispaCy 기반 medical NER로 엔티티를 추출하고 정규식 기반 불확실성 마커(hedging)를 카운트하는 두 축을 설계했다. 정렬 평가는 BiomedCLIP 임베딩의 코사인 유사도로 계산하되, 텍스트는 토큰 한도(256 tokens)로 트렁케이션해 실제 모델 처리 제약도 반영했다. 희귀 질환이 더 큰 열화를 겪을지에 대해서는 rare/common 병목 가설을 사전 명시하고, 다중 비교 보정까지 적용해 견고성을 점검했다.

- **Empirical Impact**: 결과적으로 EHR summarization은 엔티티 51.4%, hedging 43.7%를 크게 깎지만 이미지-텍스트 정렬은 거의 유지(정렬 드롭 2.5%)된다. 반대로 standardized rewriting과 teaching case preparation은 엔티티 소실은 각각 26.8%, 29.3%로 상대적으로 덜하지만, 정렬 저하는 14.9~16.5%로 EHR summarization 대비 6~7배 수준이었다. 논문은 이를 “slop paradox”로 명명하며, 훈련용으로 더 “깔끔해 보이게” 만든 텍스트가 오히려 비전-언어 대응을 더 멀어지게 만든다고 경고한다. 또한 희귀 병리가 더 심하게 훼손된다는 가설은 다중 비교 보정 후에도 유의한 차이가 없어, 조건별 성능 모니터링으로 오염을 감지하기 어렵다는 함의가 도출됐다.



### LLMs Infer Cultural Context but Fail to Apply It When Responding (https://arxiv.org/abs/2606.17688)
Comments:
          9 pages, 7 figures, 2 tables (24 pages, 12 figures, 8 tables including references and appendices)

- **Prior Approaches**: 기존 연구는 LLM이 서구/미국 중심으로 편향되고, 문화적 지식을 갖고 있는지·사회적 규범을 이해하는지에 초점을 맞춰 왔습니다. 또 개인화(personalization) 관점에서는 사용자의 문화에 맞춰 출력만 바꾸면 된다고 보지만, 과도한 개인화가 에코체임을 강화하거나 의도·뉘앙스를 놓칠 수 있다는 한계가 지적됩니다. 본 논문은 이런 흐름에서 “알고 있는 것”과 “실제로 적용해 말하는 것”이 분리돼 있다는 문제의식을 더 정밀하게 다룹니다.

- **Core Contribution**: CAPRI(Cultural and Pragmatic Response Inference)라는 대화 데이터셋을 제안해, 모델이 사용자의 문화 배경(country proxy)을 추론(BG)하는 능력과, 그 정보를 답변 생성에 실제로 반영해 지역 관습에 맞추는 능력(VQA)을 분리해 평가합니다. 특히 화폐·거리·크기·온도 같은 측정 단위뿐 아니라, time/quantity 같은 주관적 표현의 문화적 grounding 변화도 함께 봅니다. 이로써 “문화 지식의 보유”와 “문화적 적응 언어 생성(act on it)” 사이의 갭을 계량화합니다.

- **Technical Challenges**: 핵심 과제는 모델이 대화 맥락의 문화 단서를 근거로 문화 배경을 맞히더라도, 정작 답변에서는 해당 관습(예: °C vs °F)을 선택·조합하는 “pragmatic speaker” 추론을 수행하지 못할 수 있다는 점입니다. 논문은 단서 강도(없음/암묵/명시) 조건과, ground-truth가 있는 objective 단서(측정 단위)·정답이 없는 subjective 단서(time/quantity)를 나눠 실험했고, CoT(chain-of-thought)에서 문화 추론을 단계적으로 수행하도록 지시하는 Pragmatic CoT가 성능 격차를 크게 줄인다고 보고합니다. 또한 모델의 문화 priors(단서 없이 기본 성향)를 측정해, 단순 추론 실패뿐 아니라 출발점 편향도 함께 드러냅니다.

- **Empirical Impact**: 실험 결과, 대부분의 SOTA 모델은 문화 배경 추론(BG)에는 1~2개의 단서만으로 매우 잘 맞히지만, 그 정보를 측정 단위 답변(VQA)에 반영하는 능력은 훨씬 낮아 큰 격차가 관찰됩니다. Pragmatic CoT처럼 “배경 추론→답변 생성”을 명시적으로 순차화하면 유의미하게 gap이 줄어들어, 현 모델들이 지식은 갖고 있어도 연결(link)하는 능력이 부족할 수 있음을 시사합니다. 또한 주관적 표현에서는 단서가 누적될수록 답변이 더 갈라지지만, 단서 없는 priors는 때때로 모델의 국가 기원에 맞춰 치우쳐 CAPRI가 향후 연구의 기준점이 될 가능성을 보여줍니다.



### SuCo: Sufficiency-guided Continuous Adaptive Reasoning (https://arxiv.org/abs/2606.17687)
Comments:
          Accepted to ICML 2026. 18 pages

- **Prior Approaches**: LRM은 CoT(Chain-of-Thought)를 생성해 복잡한 추론에서 성능을 끌어올렸지만, 실제론 불필요하게 긴 추론을 반복해 추론 비용과 지연을 키우는 문제가 컸습니다. 이를 줄이기 위한 ALRM(Adaptive Large Reasoning Models)은 외부 추정기나 사전 정의된 reasoning mode/예산 tier로 “이산(discrete) 전환”을 하는 방식이 많아, 언제 추론을 멈춰야 하는지에 대한 원칙이 약했습니다. 또한 CoT를 줄이기 위한 휴리스틱 길이 제한이나 binary triggering은 underthinking과 overthinking을 동시에 정교하게 다루기 어렵다는 한계가 드러났습니다.

- **Core Contribution**: 이 논문은 CoT 궤적에서 정답을 내는 데 충분한 최단 prefix를 “Minimal Sufficient CoT(MSC)”로 정의해, 추론 길이를 줄일 때의 기준을 원리적으로 제시합니다. MSC는 어려움 수준별로 답 정확도를 유지하면서도 reasoning 토큰을 크게 줄일 수 있음을 실험으로 확인하고, 이를 바탕으로 연속적인 추론 제어 프레임워크인 “Sufficiency-guided Continuous Adaptive Reasoning(SuCo)”를 제안합니다. SuCo는 이산 모드 없이 문제 난이도에 맞춘 “충분성(sufficiency) 임계값”을 학습해, 필요한 만큼만 추론하도록 만듭니다.

- **Technical Challenges**: 핵심 난제는 “충분한 추론”의 신호를 학습에 쓸 수 있게 정량화하는 것입니다. 저자들은 정답을 뒷받침하는 CoT prefix의 sufficiency를 모델의 조건부 확률을 기반으로 정의하되, 긴 답에서 생기는 신호 붕괴를 완화하기 위해 per-token 평균화를 사용하고, 문장 단위로 MSC를 탐색해 논리 구조가 깨지지 않게 했습니다. 또 임계값을 고정하면 쉬운 문제엔 과잉 추론, 어려운 문제엔 조기 절단이 발생하므로, 데이터에서 추론 길이의 percentile로 문제 난이도를 추정해 임계값을 적응시키는 방식(MSC-Aligned Fine-Tuning, MFT)을 쓰고, 이후 RL 단계(Sufficiency-Aware Policy Optimization, SAPO)에서는 dynamic complexity pool과 sufficiency-aware reward로 over-/under-thinking을 함께 패널티로 제어합니다.

- **Empirical Impact**: 수학·코드·과학 벤치마크에서 SuCo는 정확도와 추론 효율을 동시에 개선하며, full CoT 대비 reasoning 토큰을 약 74~76% 줄이면서도 성능을 유지/상회하는 결과를 보였습니다. 예를 들어 7B 스케일에서 정확도는 72.1%로 LHRM보다 상대적으로 5.1%p 높고, DeepSeek-R1-Distill-Qwen 대비 14.1%p 격차를 벌렸습니다. 특히 AIME25처럼 어려운 구간에서 정확도 향상이 두드러져(7B에서 61.7%), “더 적게 추론해도 더 잘할 수 있다”는 MSC의 관점을 다양한 난이도에 걸쳐 실증했다는 점에서 의미가 큽니다.



### Bridging Functional Correctness and Runtime Efficiency Gaps in LLM-Based Code Translation (https://arxiv.org/abs/2606.17683)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: 기존 코딩 번역 연구는 기능적 정합성(컴파일/정답 일치)에 초점을 맞춰왔고, RAG·CoT 같은 training-free 프롬프트나 compiler 피드백, supervised fine-tuning 및 preference learning, reinforcement learning 등이 주로 오류를 줄이는 방향으로 발전해 왔습니다. 다만 runtime efficiency(실행 시간/자원)는 거의 다뤄지지 않아, 더 느린 번역이 “버그에 준하는 문제”가 될 수 있다는 ISO/IEC 25010 관점이 공백으로 남아 있었습니다. 또한 효율을 올리려는 단순 프롬프트/후처리 방식은 복잡도 증가로 인해 정확도 하락을 동반하는 트레이드오프가 나타났습니다.

- **Core Contribution**: 이 논문은 LLM 코드 번역에서 “정확도뿐 아니라 실행 효율도 함께” 개선하기 위해 SwiftTrans를 제안합니다. SwiftTrans는 (1) Multi-Perspective Exploration에서 MpTranslator가 parallel in-context learning(인컨텍스트 학습)으로 다양한 번역 후보를 만들고, (2) Difference-Aware Selection에서 DiffSelector가 번역 간 차이를 비교해 최적 후보를 고르는 구조입니다. 여기에 Hierarchical Guidance와 Ordinal Guidance를 더해, 경량 open-source LLM도 정확도-효율 균형을 더 잘 맞추도록 학습을 설계합니다.

- **Technical Challenges**: 핵심 과제는 (a) LLM 번역이 소스의 비효율적 구조를 그대로 재현해 느려지는데, 이를 prompt engineering만으로는 고치기 어렵다는 점과 (b) 후보들이 거의 비슷해 LLM-as-a-judge가 미세 차이를 판별하기 어렵다는 점입니다. MpTranslator는 반복 sampling 대신 parallel ICL과 hierarchical data(정확하지만 느린→점진적 최적화)를 결합해 출력 다양성과 목적 적합성을 동시에 확보합니다. DiffSelector는 후보 쌍을 unified diff 형태로 비교하는 difference-aware 판정을 하되, all-pair 비교를 bubble sort에서 영감 받은 O(n) 수준의 버블 선택으로 줄이고 Ordinal guidance로 후보 순서 민감성도 완화합니다.

- **Empirical Impact**: 확장된 CodeNet·F2SBench와 새 벤치마크 SwiftBench(비효율 패턴 소스 포함)에서 SwiftTrans는 기능적 정확도와 실행 시간 양쪽에서 일관된 개선을 보였습니다. 특히 “Correctness+Efficiency” 류 프롬프트는 효율은 오르지만 정확도가 떨어지는 반면, SwiftTrans는 더 작은 모델에서도 정확도 손실을 제한하면서 ET를 낮추는 결과를 보였습니다. 연구진은 실제 실행 시간을 안정적으로 측정하기 위해 Judge0 샌드박스 기반 반복 실행 및 기준선(보수적 번역 중 최댓값) 설정도 함께 도입해, 효율 평가의 신뢰도를 높였다는 점에서 의미가 있습니다.



### From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning (https://arxiv.org/abs/2606.17682)
- **Prior Approaches**: 기존 LLM 훈련의 강화학습(RL) 파이프라인은 단계마다 환경을 사람이 수동으로 재설계하며, 현재 정책에 무엇이 더 유리할지 경험적으로 추정해야 했다. 이 과정은 휴리스틱 의존도가 높아 자동화/재현성이 떨어지고, 실패 원인과 환경 변경의 연결이 불명확했다. 결과적으로 환경 변경이 성능 향상으로 이어지는지 체계적으로 검증하기도 어려웠다.

- **Core Contribution**: 이 논문은 LLM-as-Environment-Engineer 프레임워크를 제안해, 현재 정책 모델이 실패 궤적과 맥락 정보를 분석한 뒤 다음 단계 훈련에 쓸 환경 구성 수정안을 만든다. 또한 환경 재설계를 연구·벤치마킹하기 위한 MAPF-FrozenLake라는 제어형 테스트베드를 도입해, 생성기가 다차원 환경 설정을 노출하도록 했다. 정책의 행동 요약, 실패 사례, 환경 통계를 구조화해 조건부로 “환경 엔지니어”를 생성하는 방식이다.

- **Technical Challenges**: 핵심 난제는 (1) 실패 정보가 환경 변경으로 어떻게 번역돼야 하는지, (2) 다음 단계 환경 구성안을 학습·검증 가능한 형태로 표현하는지, (3) 어떤 맥락이 실제로 유효한지 식별하는 것이다. 저자들은 실패 증거 중심의 구조화된 컨텍스트와 환경 통계를 함께 제공하고, 생성된 다음 단계 구성으로 end-to-end하게 이어지는 재설계 과정을 학습 가능하게 구성했다. 더 나아가 실패 증거가 있고 이미 잘 작동하던 설정을 보존하는 업데이트가 특히 효과적임을 분석했다.

- **Empirical Impact**: Qwen3-4B 백본에서 이 프레임워크는 제안된 벤치마크들의 종합 성능에서 고정 환경 학습 기준선은 물론 GPT·Gemini 같은 더 큰 상용 LLM들보다도 강한 성적을 보였다. 또한 효과적인 컨텍스트 형태는 실패 증거를 포함하고 기존 성공 구성을 유지하는 업데이트에 있었다. 흥미롭게도 원래의 base model보다 RL 체크포인트가 더 좋은 environment engineer 역할을 해, 정책 학습이 남은 취약점을 진단하는 능력을 키운다는 신호를 제공한다.



### Prompt Perturbation for Reliable LLM Evaluation over Comparison Graphs (https://arxiv.org/abs/2606.17634)
Comments:
          42 pages, 8 figures

- **Prior Approaches**: 대규모 언어모델(LLM) 오픈엔드 과제에서는 같은 프롬프트에 대해 두 응답을 비교하는 pairwise evaluation이 널리 쓰이며, 비교 결과를 집계해 리더보드를 만든다. 하지만 이 방식은 intransitivity 문제로 인해 비교가 전역 순서를 지지하지 못하고, 예를 들어 순환 선호(A≻B≻C≻A)나 동률/불일치가 섞인 모순이 발생해 리더보드가 불안정해진다. 기존에는 이러한 모순을 사후적으로 해석하거나 단순 순위 집계에 맡기는 경우가 많아 일관성 개선이 제한적이었다.

- **Core Contribution**: 이 논문은 pairwise LLM 평가의 일관성을 높이기 위한 prompt perturbation 프레임워크를 제안한다. 각 프롬프트에 대해 섭동(perturbed variants)을 생성한 뒤, 그 결과로 얻은 비교 그래프에서 구조적으로 모순인 패턴을 식별·필터링하고, 남은 비교로 표준 ranking 방법을 적용한다. 핵심은 ranking 집계 전에 graph-level structural consistency를 평가 파이프라인에 명시적으로 반영한다는 점이다.

- **Technical Challenges**: 주된 기술적 난제는 프롬프트 비교가 만들어내는 그래프가 구조적 모순(순환, 동률 기반의 불일치)을 포함할 때 이를 안정적으로 감지하고 제거하는 절차를 설계하는 것이다. 연구진은 섭동 프롬프트로 얻은 비교 그래프들을 이용해 모순 패턴을 구조적으로 찾아내 필터링하고, 이후에야 순위 집계를 수행함으로써 모순이 리더보드에 전파되는 것을 차단한다. 즉, “비교 그래프의 일관성”을 먼저 정리한 뒤 ranking을 수행하는 흐름으로 안정성을 확보한다.

- **Empirical Impact**: 논문은 prompt perturbation과 구조적 모순 필터링을 결합하면 cyclic inconsistency가 줄고 LLM ranking의 신뢰성이 향상된다는 점을 실험적으로 보인다. 결과적으로 리더보드가 더 안정적이고 해석 가능해져, LLM 성능 비교와 실제 배포에서의 신뢰도 확보에 기여할 수 있다. 특히 순위 집계 자체를 바꾸기보다 평가 전단의 일관성 검증을 강화하는 접근이라, 기존 평가 파이프라인에도 비교적 쉽게 접목될 수 있다는 의미가 있다.



### OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation (https://arxiv.org/abs/2606.17628)
- **Prior Approaches**: 기존 memory agent들은 trajectory 저장, reflection 검색, 스킬 누적 같은 방식으로 경험을 활용하지만, 어떤 경험이 유용한지 선별하고 그에 기반해 행동하며, 재사용 가능한 지식으로 정리하고 저장소가 커져도 유지하는 ‘전체 역량’을 갖추기 어렵다는 한계가 있다. 즉, 경험을 “보관”하는 것과 “경험을 통해 진화하는 방법”을 학습하는 문제는 분리되어 남아 있었다. 또한 test-time에서 빠르게 진화하더라도, 장기적으로는 정책 수준에서 네 능력을 통합해 재현 가능하게 굳히는 과정이 부족했다.

- **Core Contribution**: 이 논문은 OPD-Evolver를 제안하며, 에이전트가 경험을 저장하는 수준을 넘어 경험을 통해 스스로 진화(Agent Evolver)하도록 느린-빠른(slow-fast) 공동 진화 프레임워크를 설계한다. 핵심은 빠른 루프에서 4단 메모리 계층을 통해 읽기-사용-쓰기-유지라는 능력을 즉시 test-time에 수행하고, 느린 루프에서 이를 배포 가능한 정책으로 증류한다는 점이다. 결과적으로 메모리 활용을 단일 기능이 아니라 ‘진화 가능한 역량 묶음’으로 학습한다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 어떤 메모리가 성과에 실제로 기여했는지 귀속(attribution)하고, (2) 그 귀속을 바탕으로 온정책 self-distillation이 안정적으로 작동하게 하며, (3) 생성/업데이트되는 메모리 저장소가 커져도 품질을 유지하는 것이다. 논문은 outcome-calibrated memory attribution과 privileged hindsight distillation을 느린 루프에 배치해 4단 능력을 정책으로 응축한다. 동시에 빠른 루프에서는 4단 메모리 계층으로 read/use/write/maintain을 수행해 test-time 적응을 빠르게 만든다.

- **Empirical Impact**: 다중 도메인 벤치마크에서 OPD-Evolver는 ReasoningBank 같은 memory 시스템을 최대 11.5%까지, Skill0 같은 training 기반 방법을 약 5.8%로 능가했다. 분석 결과, OPD-Evolver는 고가치 경험을 내부화하고 메모리 관리까지 학습해 시간이 지날수록 성능이 흔들리지 않는 경향을 보였다. 특히 OPD-Evolver-9B가 Qwen3.5-397B-A17B나 Step-3.5-Flash 같은 거대 모델과의 경쟁을 시사하며, memory-augmented 에이전트를 넘어 ‘진짜 qualified agent evolver’ 방향성을 강화한다.



### The Benchmark Illusion: Pruned LLMs Can Pass Multiple Choice but Fail to Answer (https://arxiv.org/abs/2606.17609)
- **Prior Approaches**: 대규모 언어모델 압축(특히 pruning·quantization)은 메모리/추론 비용을 줄이기 위해 널리 쓰이며, 성능 유지 여부를 주로 multiple-choice 벤치마크로 검증해 왔습니다. 그러나 이 방식은 모델이 사용자처럼 “직접 생성(open generation)”할 때의 실패를 충분히 드러내지 못한다는 약점이 있습니다. 압축 전/후의 같은 질문을 비교하지 않으면, 정답을 후보에서 고르는 능력과 실제 생성 능력이 갈라지는 현상을 놓칠 수 있습니다.

- **Core Contribution**: 이 논문은 압축이 정답을 “지웠는지(erase)” 아니면 “더 만들기 어렵게(상위 출력에서 밀어냄, demote)” 했는지를 paired-item test로 진단합니다. 같은 질문을 open generation과 multiple-choice scoring, 그리고 beam search/sampling 같은 reachability 관점에서 함께 추적해 “benchmark illusion”을 체계적으로 보여줍니다. 핵심 메시지는 압축 모델 평가는 recognition(선택/맞추기)만이 아니라 production(생성 가능성) 중심으로 재설계돼야 한다는 점입니다.

- **Technical Challenges**: 겉보기 정확도가 같아도 실제 실패 모드가 다를 수 있어, 동일 문항에서 생성·선택·더 넓은 디코딩 도달성을 분리 측정해야 했습니다. 이를 위해 후보가 제시될 때의 도움 여부를 통제하는 후보-표시 사다리(무후보 greedy → gold 포함 후보 생성 → multiple-choice log-likelihood)와 candidate ablation, gold first token의 rank 분석을 결합했습니다. 또한 “MC-only(생성은 실패하지만 multiple-choice에서는 정답 선택)”가 주로 정답 소거가 아니라 near-top demotion임을 rank와 디코딩 복구(beam/sampling, 1-shot prompting)로 확인했습니다.

- **Empirical Impact**: 결과적으로 high-sparsity pruning, 특히 Wanda 계열에서는 greedy open generation이 무너져도 multiple-choice 정확도는 종종 유지됩니다. 이때 정답은 대개 완전히 사라지지 않고 first token rank가 1→대략 3~4로 밀려나며, beam search·sampling 또는 한 줄의 in-context 예제로 복구되는 경우가 많았습니다. 따라서 multiple-choice 리더보드가 압축 LLM의 실제 사용자 활용성을 과대평가할 수 있고, 압축 모델 평가에서 “무엇을 인식하느냐”를 넘어 “무엇을 생성/도달 가능한가”를 함께 봐야 한다는 실증적 근거를 제공합니다.



### Evaluating Large Language Models Abilities for Addressee, Turn-change, and Next Speaker Prediction in Meetings (https://arxiv.org/abs/2606.17542)
Comments:
          Accepted to INTERSPEECH 2026

- **Prior Approaches**: 기존 MPC(turn-taking in multi-party conversations) 연구는 addressee detection, turn-change prediction, next speaker prediction을 각각 따로 다루는 경우가 많았고, 주로 supervised 모델(CRF, SVM, DNN)로 텍스트에 오디오/비디오 특징을 결합해 성능을 끌어올리는 방식이 일반적이었다. 최근에는 text-based LLM과 MM-LLM을 도입해 시도했지만, 원시 audio-visual 신호를 얼마나 직접 활용하는지에 대한 체계적 비교와 통합 평가는 부족했다. 또한 인간 성능을 동일한 멀티모달·온라인 조건에서 정량화한 연구가 드물어, 모델 격차의 실체를 파악하기 어려웠다.

- **Core Contribution**: 본 논문은 MPC의 turn-taking을 3개 과제(addressee detection, turn-change prediction, next speaker prediction)로 통일한 평가 프레임워크를 제시한다. 모델과 인간 모두가 “미래 발화는 보지 못하고 과거/현재 정보만”으로 예측하도록 설계해, 실제 에이전트 상황과 유사한 온라인 제약을 반영한다. AMI 회의 데이터에서 supervised 모델, text-based LLM, MM-LLM, 그리고 인간을 같은 프로토콜로 비교해 격차를 직접 측정한다.

- **Technical Challenges**: 핵심 기술적 난제는 MM-LLM이 원시 audio-visual 신호를 turn-taking 예측에 실제로 유의미하게 활용하는지 검증하는 것이다. 이를 위해 저자들은 동일한 입력 정의와 문맥 구성을 두되, LLM은 대화 맥락을 프롬프트로 반영하고 supervised는 이전 화자 ID 중심의 context를 사용하도록 맞춰 공정 비교를 수행한다. 또한 ablation과 과제별 성능 분석을 통해 context 의존성이 next speaker prediction에서 특히 크고, 발화 간격이 잦을수록 인간과 모델 모두 예측이 어려움을 겪는 패턴을 확인한다.

- **Empirical Impact**: AMI(4인 영어 회의) 실험에서 LLM은 next speaker prediction에서 supervised 모델과 인간을 능가했으며, 특히 target 도메인 학습 없이도 텍스트 맥락만으로 유의미한 성능을 보였다. 반면 MM-LLM은 addressee detection과 turn-change prediction에서 text-based LLM보다 좋았지만, 여전히 인간보다는 낮아 raw audio-visual 신호 활용이 제한적임을 시사한다. 인간 성능도 전반적으로 높지 않게(예: four candidate 구조의 next speaker prediction F1이 약 60% 수준) 관측되어, MPC 예측 자체가 본질적으로 난해하며 모델이 인간에 근접하려면 멀티모달 신호 통합의 개선이 필요하다는 메시지를 남긴다.



### An expressivity analysis of hierarchical modelling in deep transformers via bounded-depth grammars (https://arxiv.org/abs/2606.17522)
- **Prior Approaches**: 기존 연구는 딥러닝이 계층적 표현을 만든다는 직관을 제시했지만, 트랜스포머가 언어의 계층 구조를 “어떤 방식으로” 표상하는지에 대한 엄밀한 이론이 부족했습니다. 실증적으로는 linear probe로 잔차 스트림(residual stream)에서 문법/의미 관련 정보가 저차원에서 선형 분리된다고 보고됐지만, 왜 트랜스포머가 그런 구조적 수용력을 갖는지의 존재 증명은 제공하지 못했습니다. 또한 CFG 관점의 형식언어 분석이 있었더라도, CFG의 문법 깊이와 트랜스포머 레이어 깊이의 직접 연결은 제한적이었습니다.

- **Core Contribution**: 이 논문은 bounded-depth(깊이 제한)·non-recursive CFG(재귀 없는 CFG)를 대상으로, 딥 트랜스포머가 계층 상태를 저차원 선형 분리 공간에 인코딩할 수 있음을 이론적으로 증명합니다. 특히 positional attention을 사용한 구성으로, 트랜스포머의 깊이는 문법 깊이에 선형으로 증가하고 표현은 residual stream 안의 “낮은 차원 부분공간”에 정렬되도록 설계합니다. 이는 기존 실험이 뒷받침한 linear representation hypothesis를, 실제 아키텍처가 그런 표상을 구성할 수 있다는 형태의 엄밀한 존재 보장으로 연결합니다.

- **Technical Challenges**: 핵심 난제는 CFG의 계층적/조합적 생성 과정을 트랜스포머 레이어(특히 self-attention과 FFN)의 연산으로 재현하면서, 필요한 표상 차원과 레이어 수가 통제 가능하도록 만드는 것이었습니다. 저자들은 bounded-depth CFG를 “동적 계획법(bottom-up parsing)”과 같은 계산 구조로 매핑하고, attention head를 하드코딩해 로컬 문맥을 집계하도록 함으로써 트리 빌딩 패턴을 이론적으로 구현합니다. 그 결과 뉴런 수는 derivation-tree shapes 개수(c)에 선형, production rules 개수(MM)에 대해 이차로 스케일링하며, 레이어 깊이는 CFG 깊이 dd에 비례하는 상계를 제시합니다.

- **Empirical Impact**: 이 작업은 대규모 실험 결과라기보다, 합성 CFG 및 인지적 파싱 직관을 뒷받침하는 “구조적 상한/존재 증명”에 초점을 둔 이론적 임팩트를 갖습니다. 선형 분리 가능성(저차원 선형 방향에 문법 상태를 인코딩)을 트랜스포머 설계 가능성과 연결함으로써, 후속 연구가 실증 결과를 더 일반 CFG나 더 현실적인 코퍼스로 확장하는 데 기준점을 제공합니다. 또한 decoder-only 트랜스포머 관점에서 문법 깊이와 레이어 깊이를 직접 연결한 점이, 향후 모델 설계·분석의 이론적 기반을 강화합니다.



### Scaling Enterprise Agent Routing: Degradation, Diagnosis, and Recovery (https://arxiv.org/abs/2606.17519)
Comments:
          10 pages (6 main + 4 appendix), 4 figures, 6 tables

- **Prior Approaches**: LLM 어시스턴트는 사용자 요청을 도구 라이브러리로 라우팅해 작업을 분담하지만, 카탈로그가 커질수록 성능이 급격히 떨어진다는 보고가 이어졌다. 기존 연구들은 주로 tool calling 자체의 저하나 retrieval 오류 비중을 관찰했지만, 무엇이 어느 지점에서 깨지는지(메커니즘 분해)와 실제 개선 레버가 무엇인지가 덜 규명돼 있었다. 또한 네임스페이스 기반 검색, BM25 같은 플랫폼 도구 검색, 계층형 LLM 라우팅 등은 일부 완화하나 대규모에서 한계가 남는 것으로 보였다.

- **Core Contribution**: 이 논문은 배포된 엔터프라이즈 생산성 어시스턴트의 실제 카탈로그(110 agents, 584 tools)를 사용해, single-step 라우팅 정확도가 에이전트 수 확장에 따라 어떻게 무너지는지 통제 실험으로 진단한다. 특히 oracle 분석을 통해 저하를 retrieval gap과 confusion gap으로 분해해, “정답 도구를 노출하지 못하는 문제”와 “도구를 노출해도 비슷한 도구를 헷갈리는 문제”를 분리해 보여준다. 이어서 embedding-based shortlisting이 두 격차 중 retrieval 쪽을 얼마나 효과적으로 메우는지, 그리고 현업 트래픽에서 재현되는지도 검증한다.

- **Technical Challenges**: 핵심 기술 난제는 카탈로그 확대로 인해 semantically overlapping 도구가 늘며 recall이 무너지는 구조적 문제를, 단 한 번의 라우팅에서 얼마나 회복할 수 있는지다. 저자들은 F1 저하를 recall/precision 관점에서 확인한 뒤, oracle ceiling까지 포함해 retrieval gap(모델이 올바른 도구를 끌어오지 못함)과 confusion gap(올바른 후보를 줘도 선택이 흐려짐)을 수치로 분리한다. 해결책으로는 텍스트 embedding 기반으로 라우터 입력 후보를 k=20(전체의 극히 일부)로 줄이는 shortlisting을 제안하며, tool-level 후보가 pack-level보다 더 잘 작동함을 비교 실험으로 뒷받침한다.

- **Empirical Impact**: 실험 결과, 10→110 agents로 확장될 때 단일 단계 라우팅 F1이 모델 전반에서 16–23pp 하락했는데, 감소는 주로 recall에서 발생했다. shortlisting을 적용하면 full scale에서 F1이 10–11pp 회복되며, 특히 tool-level retrieval은 platform tool search 및 hierarchical LLM routing 같은 pack-level 접근보다 일관되게 2–4pp 우위였다. 또한 1,435개 사람이 라벨링한 실제(implicit) 트래픽에서도 합성 실험의 경향이 유지되어 +10–17pp 수준의 회복을 확인했으며, 이는 대규모 도구 카탈로그 운영에서 실질적인 정책(후보 축소) 레버가 될 수 있음을 시사한다.



### Evaluating Second-Order Bias of LLMs Through Epistemic Entitlemen (https://arxiv.org/abs/2606.17506)
Comments:
          20 pages, 13 tables, 2 figures

- **Prior Approaches**: 기존 LLM-bias 평가는 모델이 편향된 내용을 생성하거나 암시하는지에 초점을 두는 경우가 많았고, LLM-as-a-judge에서는 프레이밍/순서/표현 같은 무관한 프롬프트 요소가 판단을 흔드는지를 주로 봤습니다. 다만 LLM이 편향 콘텐츠를 ‘판정’할 때 어떤 사회적 가정을 끌어와 수용/거부를 정하는지는 체계적으로 포착되지 않았습니다. 그 결과 안전장치가 생성 단계의 노골적 편향을 막아도, 판정 단계의 2차 편향은 남을 수 있다는 공백이 있었습니다.

- **Core Contribution**: 이 논문은 LLM이 편향 콘텐츠를 평가하면서 드러내는 ‘second-order bias(2차 편향, sob)’를 정의하고, 이를 측정하는 철학 기반 추론 과제를 제안합니다. entitlement epistemology(지위/자격에 기반한 인식론)를 바탕으로, 편향을 ‘증거가 아니라 잘못된 기초 가정(미스플레이스드 epistemic entitlement)’이 추론에 기여하는 현상으로 재개념화합니다. 그리고 편향 텍스트가 누구에게는 수용 가능/불가능하다고 판단하는지, 그 판단에 어떤 인구통계 추론이 섞이는지를 Unknown이 아닌 응답으로 진단합니다.

- **Technical Challenges**: 핵심 난제는 ‘편향을 판단한다’의 의미가 제각각일 수 있어, 모델의 사회적 가정이 관측 가능하도록 작업을 논리 규칙 형태로 설계해야 한다는 점입니다. 연구진은 acceptability와 non-acceptability를 구분하는 논리 조건을 넣고, 모델이 인구통계 변수로 ‘누구’를 특정하도록 2단계 응답을 유도하되, 정당화 근거가 없으면 Unknown을 하도록 설계했습니다. 또한 sob는 해당 작업에서 Unknown이 아닌 답이 나올 때 포함된 부당한 인구통계 속성의 평균으로 정의해, 안전 거절(refusal)이 아니라 판단 실패 모드 자체를 정량화했습니다.

- **Empirical Impact**: 다양한 bias 데이터셋(예: DynaB, ToxiGen, HateCheck, iSHate, LingHate)과 여러 open/closed 모델을 실험한 결과, sob는 safety guardrails를 ‘회피’하듯 생성 단계가 아니라 판정 단계에서 일관되게 드러났습니다. 특히 sob는 타깃 그룹별로 체계적으로 달라져 특정 집단에 대한 편향이 더 ‘의미 있는 관점’으로 재순환되며, 동시에 특정 인구집단 라벨에 모델이 계속 반응하는 패턴(암묵적 social map, stereotyping attributions)을 보였습니다. 즉, 기존의 생성/암시 기반 편향 평가만으로는 판정 책임성(judgment reliability)을 충분히 담보하기 어렵고, NLP에서 이론적으로 grounded된 ‘판단 과제’ 중심 평가 필요성을 강하게 시사합니다.



### Decoding Hidden Deception in Reasoning LLMs: Activation Explainers for Deception Auditing (https://arxiv.org/abs/2606.17478)
Comments:
          Under review

- **Prior Approaches**: 기존 deception monitors는 주로 (1) 가시적인 대화/응답 텍스트에 점수를 매기거나, (2) representation vector에서 추출한 scalar probe score로 의심도를 판단합니다. 그러나 이 방식은 왜 의심스러운지에 대한 “검증 가능한 근거”가 부족해, 감사(auditing) 과정에서 해석 가능성이 제한됩니다. 또한 설명 단서가 단일 점수에 갇혀 있어 인력 검토로 이어지기 어렵다는 문제가 있습니다.

- **Core Contribution**: 이 논문은 STATEWITNESS라는 activation explainer를 제안합니다. 타깃 LLM의 hidden states를 읽는 별도 decoder가 natural-language 질의에 답하거나, 구조화된 보고서를 생성해 “의심의 근거”를 더 구체적으로 제공합니다. 즉, deception 탐지 단계를 넘어 감사자가 확인할 수 있는 설명 가능한 인터페이스를 제공하는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술 과제는 hidden states에서 deception 관련 신호를 안정적으로 포착하면서도, 사람에게 유용한 형태의 근거(질의 응답·스키마 보고서·evidence trace)로 변환하는 것입니다. 저자들은 별도 decoder를 두어 hidden states를 입력으로 자연어/구조화 출력 모두를 학습하고, token- 또는 sentence-level evidence trace로 근거를 추적 가능하게 했습니다. 이를 통해 단일 scalar 감지의 한계를 설명 가능한 진단으로 확장했습니다.

- **Empirical Impact**: 평가는 2개의 reasoning LLM을 대상으로 7개 deception 데이터셋에서 수행됐고, STATEWITNESS는 mean AUROC 0.916을 기록했습니다. 이는 동일 프로토콜에서 black-box 텍스트 모니터 대비 상대 11.6% 향상, activation-probe baseline 대비 상대 25.0% 향상을 의미합니다. 또한 기존 모니터와 함께 쓰는 threshold ensemble에서 missed deceptive examples를 줄였으며, 사람 검토를 위한 근거 출력까지 제공해 정렬(alignment) 및 interpretability 도구로 확장될 가능성을 보여줍니다.



### AIPatient Arena: EHR-grounded evaluation of large language models in end-to-end clinical consultation workflows (https://arxiv.org/abs/2606.17474)
Comments:
          49 pages, 12 figues, 11 tables

- **Prior Approaches**: 기존 의학 LLM 평가는 대부분 정적이거나 단일 턴, 또는 특정 결과 중심으로 이뤄져 실제 진료의 순차성·불확실성·상호작용을 충분히 반영하지 못했습니다. 또한 최종 답변 정확도만 보려는 경향이 강해, 환자 발화에서 정보를 어떻게 수집하고 해석하며 설명하는지 같은 과정 평가는 상대적으로 약했습니다.

- **Core Contribution**: 이 논문은 EHR(전자건강기록) 기반의 임상 유틸리티 평가 프레임워크인 AIPatient Arena를 제안합니다. 환자별 지식 그래프를 구성해 다중 턴 의사-환자 상호작용을 평가하고, 임상 역량을 8개 차원(예: 질문 기술, 윤리/전문성, 설명의 명확성·투명성, 진단 추론 등)으로 체계화합니다.

- **Technical Challenges**: 핵심 난제는 모델이 진료 대화 중 불명확한 환자 답변을 처리하고, 과거 병력을 포함한 필요한 정보를 빠짐없이 커버하며, 불확실성을 적절히 다루는지 평가하는 데 있습니다. 저자들은 EHR-기반 지식 그래프와 프로세스 중심의 다중 턴 평가로 상호작용 실패(반복 질문, 병력 누락, 불확실성 취급 미흡 등)를 관찰 가능하게 만들고, 풍부한 대화 맥락이 진단 추론에는 도움이 되지만 치료 계획 개선은 제한적임을 함께 확인합니다.

- **Empirical Impact**: 437명의 1차 코호트와 두 개의 out-of-distribution 검증 코호트(각 119·67명)에서 LLM은 질문 기술(QS), 윤리·전문성(ET), 임상 설명의 명확성·투명성(EX)에서 대체로 높은 점수(대략 4점대)를 보였습니다. 반면 모호한 답변 처리(HR), 정보 커버리지(IC), 진단 정확도·추론(Dx)에서 약점이 지속됐고, 과정 기반 평가가 최종 답변 정확도만으로는 임상 준비도(readiness)를 판단하기 어렵다는 메시지를 강화합니다. 이 결과는 배포 전 medical LLM을 workflow 단위로 검증하는 표준에 가까운 방향성을 제시합니다.



### MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation (https://arxiv.org/abs/2606.17449)
Comments:
          To be presented at ACL 2026

- **Prior Approaches**: 기존 M-RAG는 retrieval-augmented generation 흐름에서 정적 파이프라인과 유사도 기반 필터링에 의존해, 시각-텍스트 충돌을 분리·판단하기 어렵다. 그 결과 cross-modal hallucination, causal fabrication, sycophancy가 자주 발생하며, 수정용 룰을 일괄 적용하면 정확한 생성까지 과도하게 깨지는 ‘intervention paradox’가 이어진다. 또 가벼운 LLM의 무가이드 다단 추론은 포맷 불안정으로 구조적 실패가 연쇄되며 논리적 드리프트를 키운다.

- **Core Contribution**: 이 논문은 MODE-RAG(Multimodal Objective Diagnostic Energy-RAG)로, Variational Free Energy(VFE)와 내부 attention states(ATLAS)를 이용해 개입 필요성을 동적으로 게이팅한다. FE-Router가 uncertainty가 높은 질의만 전문 멀티에이전트 파이프라인으로 라우팅하고, 나머지는 우회해 과잉 교정으로 인한 정확도 저하를 막는다. 또한 단계별(인식·검색·추론·생성)로 원인에 대응하는 에이전트를 두고, logit perturbation과 overseer 검증으로 sycophancy·논리적 조작·포맷 붕괴를 억제한다.

- **Technical Challenges**: 핵심 난제는 ‘언제 얼마나’ 개입해야 하는지이며, 정적 룰은 과잉 교정, 무가이드 추론은 실패 연쇄를 낳는다는 점이다. MODE-RAG는 VFE 기반 FE-Router로 고위험(Claim-Scene 충돌 등) 신호를 감지해 개입을 선택하고, Per-Agent의 atomic visual fact 추출로 ‘visual-first’ 앵커를 고정한다. 추론 단계에서는 Monte Carlo Tree Search(MCTS)로 인과 DAG를 구성해 temporal inversion/forced causality를 줄이고, Gen-Agent의 logit perturbation 및 overseer의 삼중 일관성 검사로 사용자 편향에 대한 과적합을 페널티한다.

- **Empirical Impact**: 평가를 위해 ModeVent를 제안하며, MultiVent에서 VFE 상·하위(불확실성 극단) 샘플을 골라 retrieval-시각 충돌과 manifold outlier에 강하게 테스트한다. Qwen-2.5-VL-7B 베이스라인 대비 MODE-RAG는 전체 평균 fidelity/resilience에서 일관된 개선을 보였고, 특히 Outliers에서 attention hijacking·majority text bias·out-of-domain irrelevance 같은 극단 실패를 크게 완화했다. 반면 비용은 질의당 처리시간이 평균 18.5초→26.2초(약 1.42×)로 증가하지만, 단계별 에이전트 개입 구조 덕분에 병렬화로 상쇄 가능성을 제시한다.



### NarrativeWorldBench: A Frontier-Saturated Benchmark and a Latent World Model for Long-Horizon Co-Creative Audio Drama (https://arxiv.org/abs/2606.17391)
Comments:
          10 pages. Accepted to the ICML 2026 Workshops on High-dimensional Learning Dynamics (HiLD) and Culture x AI

- **Prior Approaches**: 기존 장문 벤치마크(LongBench, RULER, L-Eval 등)는 주로 검색·사실 회상·요약 성능을 평가하며, 공동 창작 상황에서 연재 구조의 일관성을 직접 측정하진 못했다. 장문 생성 연구(예: plan-and-write, search 기반)는 목표를 길게 쓰는 데 초점을 두지만, 중간 에피소드가 생략된 채로도 ‘연재 상태(state)’를 유지하는 능력 평가와는 결이 달랐다. 그 결과, 장편 서사에서 관측되는 horizon 붕괴 현상을 체계적으로 진단할 기준이 부족했다.

- **Core Contribution**: 이 논문은 장편 오디오 드라마 연재의 구조적 일관성을 정량화하는 NarrativeWorldBench를 제안하고, 21개 LLM을 동일한 9개 내러티브 구조 지표로 감사(audit)했다. 특히 closed-frontier/ reasoning 계열 모델이 plot-beat F1이 [0.78, 0.81]에서 포화된 뒤 h=200에서 약 -0.20 F1로 붕괴함을 실증했다. 이를 넘어 N-VSSM(Narrative Variational State-Space Model)은 256차원 잠재 ‘세계 상태’를 갱신해 horizon이 길어져도 구조 성능을 유지하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 장편 연재가 부분 관측(partially observed) 과정이라, 국소 문맥만으로는 잠재 상태를 복원하기 어렵다는 점이다. 논문은 scene boundary마다 event tuple을 뽑아 변분 인코더로 잠재 상태를 갱신하고, Mamba-2 백본 디코더에 cross-attention 기반 저랭크 어댑터를 결합해 구조 정보를 전파한다. 또한 문화권 간 차이(현지화의 underspecification)를 ‘디코더 재학습’이 아니라 256차원 잠재 공간에 대한 Learned Cultural Transfer Function으로 보정해, 네 Indic 언어에서 교차언어 품질을 끌어올렸다.

- **Empirical Impact**: 구조 지표의 대표인 plot-beat F1에서 N-VSSM은 모든 horizon(h=10~200)에서 0.84 이상을 유지하며, 연산은 frontier 대역 대비 에피소드당 4배 낮은 비용을 주장한다. 긴 horizon에서 foreshadowing payoff, temporal coherence, motif persistence가 각각 대역 대비 +0.18, +0.14, +0.12 수준으로 개선됐다. 문화 전이 함수를 켜면 문화적 충실도(Likert 7점)가 언어별로 약 +0.20~+0.23 상승했고, within-subject writer study(n=12)에서는 long-arc consistency에서 71%로 우위를 보이며 controllability에서 +1.3 Likert 점을 더 받았다.



### Implicit vs. Explicit Prompting Strategies for LVLMs in Referential Communication (https://arxiv.org/abs/2606.17372)
- **Prior Approaches**: 최근 LVLM의 지시-매칭 referential communication 연구는 서로 상반된 결론을 냈습니다. 어떤 연구는 반복 라운드에서 표현 길이가 줄어들며 효율적 동조(lexical entrainment)가 나타난다고 본 반면, 다른 연구는 공통 기반(common ground)을 활용하지 못해 불필요하게 장황해진다고 보고했습니다. 또 일부는 점검창(history)과 메모리를 제공해도 자동으로 효율이 생기지 않거나, 대화적 정합성·수정 행동이 약하다고 지적했습니다.

- **Core Contribution**: 이 논문은 두 상반된 결과의 원인이 모델 버전이나 과제 차이가 아니라 prompting 스타일 차이에 있음을 직접 비교로 정리합니다. 특히 “암시적(implicit)으로 간결·정보성”을 유도하면 모델은 정확도는 유지하되 여전히 장황해지고, “명시적(explicit)으로 1~2단어처럼 짧게”를 지시하면 표현이 반복되며 안정화되는 패턴이 나타납니다. 결론적으로, 사람의 conceptual pact 형성과 유사한 겉모습은 프롬프트로 재현 가능하지만 그 내적 과정은 사람과 동일하다고 보기 어렵다고 경고합니다.

- **Technical Challenges**: 핵심 기술적 도전은 서로 다른 기존 연구의 task 설계 차이를 통제한 채, 프롬프트만 바꿔 동일 조건에서 비교하는 것입니다. 저자들은 모델 버전(GPT-5.2 vs GPT-5.5)과 입력 파이프라인을 정렬하고, 시각 컨텍스트(라운드 경계·최신 프레임·상태 렌더링)를 시간순으로 엄격히 맞춘 뒤 implicit/explicit 프롬프트만 비교했습니다. 이를 통해 “명시적 압축”이 interaction의 형태(라벨의 가지치기와 텔레그래픽 표현) 자체를 바꾼다는 점을 관찰해 해석의 혼선을 줄였습니다.

- **Empirical Impact**: 실험은 AI–AI 40 runs(라운드 200 관측)에서 정확도는 전 조건에서 높지만, communicative efficiency와 표현 수렴은 프롬프트에 크게 좌우됨을 보였습니다. implicit 조건에서는 GPT-5.2/5.5 모두 라운드가 진행돼도 장황함이 크게 줄지 않았고, explicit 조건에서는 명시적 압축이 62.8%~75.6% 수준으로 강하게 나타났습니다. 특히 explicit GPT-5.5는 표현 길이와 lexical overlap이 라운드 1→5에서 뚜렷이 수렴하며 정확도도 97.5%로 유지됐지만, 같은 전략에서 GPT-5.2는 압축 이후 정확도가 92.5%로 하락해 accuracy–brevity tradeoff 가능성도 함께 드러냈습니다. 이 결과는 referential communication에서 “사람 같은 동조”를 평가할 때 prompting 의존성을 반드시 통제해야 한다는 실증적 시사점을 줍니다.



### Translating the Untranslatable: An Operationalizable Ontology for Untranslatability (https://arxiv.org/abs/2606.17354)
- **Prior Approaches**: 기존 NLP MT는 번역을 ‘의미가 같은 문장 → 다른 언어 문장’의 일대일 대응으로 보는 경향이 강해, 문화·문체·표현 차이로 의미 보존이 완전하지 않은 untranslatability를 구조적으로 다루기 어려웠습니다. 선행연구는 관용어, 욕설/슬랭, 경어 같은 개별 현상에 집중했지만, 공통 온톨로지로 연결해 통합 분석하거나 이를 위한 일관된 평가/학습 틀을 제공하진 못했습니다. 또한 이론적 번역연구는 분류와 보상 전략을 제시해도 NLP에서 직접 실험 가능한 스케일 자원이 부족했습니다.

- **Core Contribution**: 이 논문은 MT에서의 untranslatability를 ‘원인(불일치의 출처)’과 ‘보상 전략(compensation strategy)’이라는 두 축으로 분해해, 이를 ontology(uTypes)와 전략 분류(cStrats)로 체계화합니다. 이어서 uType 라벨이 붙은 스페인어·일본어 문장에 대해, 서로 다른 cStrats가 반영된 영어 번역을 짝지은 다국어 데이터셋을 구축해 controlled analysis가 가능하도록 했습니다. 마지막으로 전략이 품질 인식에 어떻게 영향을 주는지(전략-의존성 선호)를 초기 실험으로 확인합니다.

- **Technical Challenges**: 핵심 과제는 (1) 언어 간 의미 불일치의 다양한 원인을 자연스럽게 분류(uType)하고 (2) 그 상황에서 적절한 보상 전략을 선택·표현할 수 있게 데이터로 연산화하는 것입니다. 저자들은 인간 언어 전문가의 도움으로 온톨로지의 기반을 세우고, LLM을 이용해 동일 uType 내에서도 다양한 예시를 생성하되 프롬프트 반복 개선과 인간 검증(유효성 약 95~96%)으로 품질 변동을 완화했습니다. 또한 표준 token-level 번역 가정 대신, ‘불일치 식별 → 전략 선택 → 토큰 생성’의 분해 관점으로 모델 설계 아이디어를 제시했습니다.

- **Empirical Impact**: 인간 선호도 연구에서 번역 품질은 단순 충실도뿐 아니라 사용한 cStrat에 크게 좌우되며, 특히 추가 설명 맥락을 포함하는 Annotation(AN) 전략이 전반적으로 가장 선호되는 경향을 보였습니다. 유효성 있는 차이는 uType과 번역 맥락(예: textbook vs movie) 및 소스 언어(스페인어 vs 일본어)에도 따라 달라져, ‘기본 번역’만으로는 인간 기대를 충분히 충족하기 어렵다는 실증 근거를 제공합니다. 결과적으로 이 프레임워크와 데이터셋은 strategy-informed machine translation 연구 및 다운스트림(전략 예측, untranslatability 탐지 등) 확장에 기초가 될 것으로 기대됩니다.



### Do Large Language Models Always Tell The Same Stories? (https://arxiv.org/abs/2606.17350)
- **Prior Approaches**: 기존 연구는 창의성을 Alternative Uses Task, Torrance Test of Creative Thinking 같은 루브릭/심리측정 기반으로 평가하거나, n-gram novelty 같은 어휘 기반 지표로 자동화해 왔습니다. 다만 LLM-as-a-Judge 방식은 주관성과 편향으로 일관성이 떨어질 수 있고, n-gram novelty는 창의성을 대리한다고 보기 어렵다는 비판도 있습니다. 한편 일부 작업은 플롯 아크나 플롯 요소 반복 같은 ‘부분’만 보며 서사 전체의 다양성을 직접 비교하긴 제한적이었습니다.

- **Core Contribution**: 이 논문은 창의성을 ‘서사 다양성’으로 재정의하고, narrative similarity(서사 유사도)라는 대조(contrastive) 프레임으로 인간과 LLM의 다양성을 동일 조건에서 비교합니다. r/WritingPrompts 프롬프트를 주고 여러 모델의 이야기를 생성한 뒤, 기준 이야기 대비 두 후보 중 어느 쪽이 더 비슷한지 선택하도록 설계했습니다. 그 결과 LLM이 만든 이야기는 모델 내부/모델 간 모두에서 서로 더 닮아가며, 특히 frontier 모델은 개인 작가의 다양성은 못 따라가고 ‘평균적인’ 서사로 수렴하는 경향을 확인합니다.

- **Technical Challenges**: 핵심 난제는 ‘다양성’을 신뢰도 있게 수치화하는 데 있었고, 이를 위해 사람이 판단하는 triplet(기준, 후보 A/B) 유사도 데이터를 만들고 자동화 방법 3가지를 검증했습니다. LLM-as-a-Judge, narrative component embedding(서사 구성요소를 분해해 임베딩 후 코사인 유사도), Bradley-Terry 기반 preference model을 비교한 뒤, 성능과 확장성을 고려해 상황별로 도구를 선택합니다. 또한 human-LLM/LLM-LLM 비교의 공정성을 위해 길이·품질 필터와 노출 편향을 줄이기 위한 시간 근접 조건을 적용했습니다.

- **Empirical Impact**: 10개 모델(프론티어 폐쇄형, 오픈형, post-training 체크포인트 포함)을 대상으로 한 대규모 결과에서 LLM 서사는 인간 서사보다 서로 더 유사하다는 패턴이 일관되게 나타났습니다. closed-source 프론티어는 인간에 더 가깝게 ‘모사’하지만, 모델 간/서로 다른 인간 작가 간 다양성은 부족했고, open-source/특정 체크포인트는 인간과는 다르더라도 여전히 내부 동질성은 강했습니다. 더구나 negative prompting, temperature scaling, round-robin 시퀀셜 생성 같은 흔한 완화 전략은 서사 동질성을 의미 있게 줄이지 못해, 향후 LLM 창작의 ‘다양성 붕괴’ 문제를 정면으로 다룰 필요성을 제기합니다.



### Examining the Limits of Word2Vec with Toki Pona (https://arxiv.org/abs/2606.17299)
Comments:
          10 pages, 4 figures, 3 tables. Accepted to the Society for Computation in Linguistics (SCiL) 2026

- **Prior Approaches**: 기존 연구는 Word2Vec이 의미 임베딩을 잘 학습한다는 점을 주로 어휘가 큰 자연언어에서 검증해 왔습니다. 하지만 매우 적은 어휘(약 130단어) 환경에서 의미 관계를 얼마나 안정적으로 포착하는지는 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 본 연구는 Toki Pona(약 130단어) 같은 극단적으로 축소된 어휘에서 Word2Vec이 의미적 관계를 학습하는지 실험적으로 확인합니다. 또한 코퍼스에 섞인 비핵심 잡음 토큰(고유명사, 차용어, 신조어)을 유지하는 경우와 완전히 제거하는 경우를 비교해, 잡음이 임베딩 구조에 미치는 영향을 정면으로 다룹니다.

- **Technical Challenges**: 핵심 난제는 어휘가 너무 작아 학습 신호가 희박해질 때에도 Word2Vec의 분포적 학습이 의미 구조를 재현할 수 있는지입니다. 연구진은 (1) 잡음 토큰 포함/제거 두 모델을 따로 학습하고, (2) 의미 범주 중심점에 대한 단어 근접도, (3) agglomerative clustering 기반 silhouette score, (4) representational similarity matrix로 영어 임베딩과의 유사성을 함께 평가해 체계적으로 비교했습니다.

- **Empirical Impact**: 결과적으로 희소한 비핵심 토큰은 임베딩의 상대적 구조를 크게 흔들지 않으면서, 오히려 벡터 공간에서 유사 단어를 더 가깝게 모으는 경향이 나타났습니다. 즉 Word2Vec의 성능은 어휘 크기보다 분포 패턴에 더 의존하며, 극단적 하한에서도 유효함을 보여주었다는 점에서 word embedding 연구의 해석 기준을 확장합니다.



### Are you speaking my languages? On spoken language adherence in multimodal LLMs (https://arxiv.org/abs/2606.17281)
Comments:
          7 pages, 3 tables in the main body

- **Prior Approaches**: 기존 다국어 ASR은 언어 ID를 입력에 고정하거나(언어별 출력 헤드, one-hot embedding) 오디오에서 별도로 추정해 편향을 주는 방식이 많았습니다. 다만 streaming에서는 확신도(지연)를 맞추기 어렵고, 언어 ID/검증 신호가 엇갈리면 언어 오인 문제가 남습니다.
또한 입력에 언어 태그를 텍스트 첫 단어로 예측하는 multitasking형 접근은 가능하지만 비스트리밍 전제(전체 오디오를 본 뒤 언어를 결정)가 커서 실사용 제약이 큽니다.

- **Core Contribution**: 이 논문은 LLM 기반 ASR에서 “언어 adherence”가 깨질 때(출력 스크립트/언어가 오디오의 기대 언어와 불일치) 어떤 품질 저하가 발생하는지, 이를 정량화하고 완화하는 체계를 제안합니다. 먼저 Language Adherence Violation Rate(LAVR)을 정의해 언어 위반 빈도를 측정하고, 이어서 language hinting을 통해 유연한 code-switching은 유지하면서도 위반을 줄이는 전략을 비교합니다.
핵심은 정답 언어를 강하게 강제하는 대신 “soft prompting”으로 모델이 힌트를 따르되 오디오 증거와 충돌할 때 적절히 견디게 만드는 데 있습니다.

- **Technical Challenges**: 기술적 난관은 사용자 의도(발화 언어)를 항상 정확히 얻기 어렵다는 점입니다. 이를 반영해 no-hint(힌트 없음), correct(정답 힌트), distractor(잘못된 힌트), mix(정답+오답 힌트) 시나리오로 성능을 비교하고, 특히 오답 힌트에서도 LAVR이 악화되지 않도록 robust guidance가 필요함을 확인합니다.
해결책으로 (1) zero-shot prompting(강건한 힌트 문구 선택), (2) SFT로 adherence 습득, (3) CoT로 먼저 언어를 추론·선언한 뒤 전사하도록 decoding을 분리하는 방법을 함께 시험합니다.

- **Empirical Impact**: 실험에서는 LAVR과 WER을 함께 보고, three methods는 공통적으로 correct 힌트를 제공받을 때 언어 위반이 크게 줄고 전사 품질도 유지되는 경향을 보였습니다. 또한 mix(정답+오답) 조건은 no-hint보다 유의미하게 나아지며, 오답 힌트만 주는 distractor는 대체로 no-hint보다 불리했습니다.
흥미롭게도 zero-shot prompting이 SFT/CoT와 전반적으로 견줄 만큼 성능이 나와, 계산 제약 하에서도 힌트 설계가 가장 영향력이 크다는 실무적 결론을 제시합니다.



### MLLP-VRAIN UPV system for the IWSLT 2026 Simultaneous Speech Translation task (https://arxiv.org/abs/2606.17255)
Comments:
          IWSLT 2026 System Description

- **Prior Approaches**: 동시 음성 번역(SimulST)에서는 ASR과 번역기를 연쇄(cascaded)로 결합하거나, 고정 지연 제어 정책으로 quality-latency trade-off를 맞추는 방식이 주로 쓰여 왔다. 다만 long-form에서는 발화가 길어질수록 정책의 경직성이 누적되어 번역 품질이 떨어지거나 지연이 커지는 문제가 반복됐다.

- **Core Contribution**: 이번 제출은 Parakeet과 Qwen 3.5 모델을 활용해 long-form SimulST에 강건한 연쇄형 파이프라인을 구성하고, 적응형 black-box policy로 처리 흐름을 제어한다. 또한 정책 제약을 완화(relaxation)해 trade-off를 더 유리하게 조정했으며, En→De/It/Zh에서는 ASR word-boosting과 RAG 기반의 오프라인 pre-translated exemplar로 도메인 맥락을 보강하는 context track도 추가했다.

- **Technical Challenges**: 핵심 난제는 긴 입력에서 언제 번역을 시작/중단할지 결정하는 지연 제어를 잘못하면 품질이 급락한다는 점이다. 논문은 adaptive black-box policy로 정책을 상황에 맞게 조정하고, 정책의 relaxations를 통해 더 세밀한 품질-지연 균형을 탐색했으며, 시스템 전반에 대해 latency 분석을 상세히 제공해 병목을 진단 가능하게 만들었다.

- **Empirical Impact**: IWSLT 2026 Simultaneous Speech Translation shared task에서 전 언어 방향으로 참여했으며, MCIF En→De 테스트셋 기준 XCOMET-XL이 지난해 대비 +5.82로 큰 폭 향상됐다. context track에서는 추가로 +1.03 성능 개선을 보여, RAG와 word-boosting을 통한 맥락 풍부화가 실제로 번역 품질을 끌어올린다는 점을 실증했다.



### Speaking in Self-Assessing Tongues: On the Verbalized Confidence of LLMs in Machine Translation (https://arxiv.org/abs/2606.17234)
- **Prior Approaches**: 기존 연구는 translation에서 토큰 확률이나 엔트로피 같은 internal 신호를 uncertainty로 보고 confidence로 간주하는 방식이 많았습니다. 하지만 표면 형태 경쟁(surface form competition) 때문에 “선택한 토큰의 확신”이 곧 “정확성”을 뜻하지 않아 MT self-calibration이 취약하다는 점이 지적됩니다. 또 verbalize된 confidence는 MT에서 상대적으로 덜 탐구되어 왔습니다.

- **Core Contribution**: 이 논문은 LLM이 생성 과정에서 ‘자신의 확신’을 말로(Verbalized confidence) 드러내는 방식으로, 토큰 단위 per-token confidence를 뽑아내는 5가지 방법을 설계합니다. 그리고 그것이 ground truth 기반 error detection(미세한 오탐/미탐)과 calibration(신뢰도-정확도 정렬)에서 internal 신호와 얼마나 정렬되는지 비교합니다. 결론적으로 verbalized confidence는 internal uncertainty와 유사하거나 일부 설정에서 더 낫게 동작합니다.

- **Technical Challenges**: 핵심 기술적 난관은 MT가 단일 정답이 아닌 동의어·구문 변형이 경쟁하는 task라는 점이라, internal 확률/엔트로피가 “정확성”을 직접 반영하지 않을 수 있습니다. 저자들은 숫자/Likert 형태의 word-와 token-단위 confidence, 그리고 모델이 불확실하다고 표시하는 spans 목록(List) 등으로 표면 형태 경쟁의 영향을 줄이려 했습니다. 또한 개발셋에서 threshold를 맞춰 이들을 이진 오탐지 신호로 binarize하고, ECE/AUROC/AUPRC 및 오류 스팬 정렬로 reliability를 평가했습니다.

- **Empirical Impact**: 실험 결과 error detection의 F1은 모델·언어쌍에 따라 최선 방법이 달랐지만, 평균적으로 verbalized 방법이 internal 기반 방법과 비슷하거나 더 좋게 나타났습니다. calibration에서는 verbalized 방법이 internal probability와는 유사하지만, 내부 entropy에 비해 뒤처지는 경향이 관찰됩니다. 특히 verbalized confidence와 internal uncertainty 사이의 상관관계는 거의 없었고, Llama3-70B는 상대적으로 verbalized의 이점이 더 커 ‘신뢰 품질’ 평가에 verbalized 접근이 실용적 대안이 될 수 있음을 시사합니다.



### Revisiting LLM Adaptation for 3D CT Report Generation: A Study of Scaling and Diagnostic Priors (https://arxiv.org/abs/2606.17213)
- **Prior Approaches**: 기존 의료 리포트 생성 연구는 2D 이미지 기반 VLM을 확장하거나(예: LLaVA 계열) 3D CT용 별도 파이프라인을 추가하는 방식이 많았다. 하지만 이들은 대개 (1) 큰 LLM을 fine-tuning해 계산 비용이 크거나, (2) 단순 linear projector로 시각-임상 의미 정렬이 약해 의미적 임상 갭이 남거나, (3) 의료 데이터가 적은 탓에 임상적 사실성보다 문장 유창성 중심의 clinical hallucination 위험이 있었다.

- **Core Contribution**: 이 논문은 3D CT 리포트 생성에서 LLM을 frozen vs fine-tuning할 때의 성능·일반화·계산 효율 trade-off를 모델 크기(96.1M~1.6B) 관점에서 체계적으로 분석한다. 그 위에 RAD3D-Prefix를 제안하며, frozen LLM에 임상 진단 priors를 prefix로 주입하되 학습해야 할 파라미터를 최소화한다.

- **Technical Challenges**: 3D CT는 입력이 고차원이고 진단 추론에 필요한 긴 임상 용어가 많아, 시각 임베딩과 임상 텍스트 의미 사이를 직접 잇기 어렵다. 논문은 CT-CLIP 기반 3D 비전 인코딩 결과에 multi-label diagnostic classification logits를 결합해 anomaly-aware prefix를 만들고, LLM은 고정한 채 transformer 기반 projection 네트워크만 학습함으로써 과적합과 임상 갭을 동시에 완화한다.

- **Empirical Impact**: CT-RATE(in-domain)와 INSPECT(out-of-domain)에서 자동 평가 지표와 임상가 reader study를 수행한 결과, RAD3D-Prefix는 유사한 parameter-efficient baseline을 능가하면서도 완전 fine-tuning 대비 훨씬 적은 trainable parameters로 성능을 낸다. 또한 크기 스케일링 실험에서 fine-tuning은 작은 LLM에서 유리하고, 약 1B+에서는 frozen+경량 projection 학습이 성능·일반화·효율의 균형이 더 좋다는 실용적 결론을 제시한다.



### Self-Generated Error Training for Token Editing in Diffusion Language Models (https://arxiv.org/abs/2606.17175)
- **Prior Approaches**: 디퓨전 언어모델은 병렬 denoising 덕분에 여러 토큰을 동시에 예측하고, 이전에 커밋한 토큰도 다시 손볼 수 있습니다. LLaDA2.1의 T2T editing은 보이는 토큰을 재스코어링해 편집 threshold를 넘으면 overwrite로 수정하는 방식으로, 품질-속도 트레이드오프의 핵심 경로입니다. 다만 기존 편집기 학습은 랜덤 보캐브러리 치환으로 “복구 대상”을 만들기 때문에, 추론 시 실제로 편집되는 오류(모델의 자체 초안에서 나온 고신뢰 예측 실수)와 분포가 어긋납니다.

- **Core Contribution**: 이 논문은 T2T 학습-추론 불일치가 성능 저하와 특정 failure mode(예: 추론은 맞는데 마지막 숫자 토큰 자릿수/스케일이 틀리는 문제, 짧은 정답에서 과도한 자기수정 루프)를 만든다고 지적합니다. 이를 해결하기 위해 Self-Generated T2T를 제안하며, 학습 때 그라디언트 없는 no-gradient draft 패스로 마스크를 채운 뒤 그 결과(맞거나 틀린 예측)를 두 번째 supervised pass의 편집 대상으로 사용합니다. 모델 구조와 추론 절차(Q-Mode T2T 파라미터)는 그대로 두고, “T2T에 들어가는 오류의 출처”만 실제 모델이 생성하는 형태로 바꾼 것이 핵심입니다.

- **Technical Challenges**: 문제는 편집기가 랜덤 치환에서 학습한 “어떤 토큰을 고쳐야 하는지”와 “어떻게 고쳐야 하는지”가, 추론에서 나타나는 맥락-의존적 draft 오류에는 잘 맞지 않는다는 점입니다. 논문은 이를 위해 2-pass 학습을 설계해 첫 패스에서 생성된 draft 토큰을 visible ground-truth와 wrong-token 위치로 분류하고, 두 번째 패스에서 기존과 동일한 supervised denoising 손실(마스크 복구 + 편집/유지)을 적용합니다. 또한 일부 위치(예: 5%)는 랜덤 토큰으로 섞어 과도한 자기복구 편향을 완충하면서, 학습 신호를 현실적인 T2T 입력 상태로 정렬합니다.

- **Empirical Impact**: LoRA continued-pretraining(짧은 어댑터 학습)으로 검증한 결과, 여러 벤치마크에서 정확도가 대체로 개선되면서 T2T 편집 강도는 감소했습니다. CMATH, TriviaQA, PIQA에서는 정확도 상승과 함께 edit intensity(E/100tok)가 줄었고, AIME 2025는 박스 프롬프트 기준 정확도는 유지되었지만 편집 횟수는 크게 감소(예: 130.2→86.0)했습니다. 특히 CMATH에서는 중국어 수학 데이터가 CPT에 없었음에도 final numeric commitment 오류를 더 잘 고치는 양상이 나타나(정답 자릿수/마지막 토큰 수정) “학습 분포 정렬”의 메커니즘 전이가 관측됐다는 점이 의미 있습니다.



### From Parasocial Scripts to Dyadic Persistence in Autonomous AI-Agent Communities (https://arxiv.org/abs/2606.17174)
Comments:
          Submitted for review in ARR for EMNLP 2026

- **Prior Approaches**: 기존 PSI/PSR 연구는 주로 인간 매체(비대칭 애착·관계 형성)나 챗봇 같은 H-AI 환경에 집중해 왔습니다. 하지만 에이전트-에이전트 온라인 커뮤니티에서는 “잠재 상태 라벨”이 없고, 일반적인 친화/사회적 언어와 구별되는 단서가 강하게 겹쳐 PSI식 관계 단서를 식별하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Moltbook에서 자율 에이전트들이 남기는 발화/댓글 텍스트를 관측 가능한 “관계 스크립트”로 재정의하고, PSI식 콜로콜 단서가 실제로 존재하는지 검증합니다. 또한 attachment/intimacy 언어(ATT), self-identification to OP(SD), reciprocity bids(RS) 3개 단서를 이 프레임으로 조작화해 OP 재참여와 상호 댓글 구조, 나아가 PSI-to-PSR 일관성까지 연결합니다.

- **Technical Challenges**: 핵심 기술 문제는 단서가 문맥에 따라 ‘그럴듯한 친화 표현’과 ‘OP-directed 관계 입찰’로 구분된다는 점입니다. 이를 위해 키워드 매칭뿐 아니라 few-shot LLM 라벨링과 grouped-context LLM 라벨링을 함께 쓰고, 배치 문맥을 submolt·스레드 크기 버킷·사전 단서 분포로 묶어 경계 드리프트를 줄였습니다.

- **Empirical Impact**: 분석은 4,434개 게시글/50,338개 댓글에서 이뤄졌고, 세 방법 공통으로 PSI 콜로콜 단서가 유의미한 비율로 관측됩니다. 특히 OP 재참여와 mutual reply 구조와의 연관이 강하게 나타났으며(대체로 adjusted OR 유의), RS(Reply-seeking reciprocity bids)는 OP-다른 쌍의 미래 상호 재귀(PSR-consistent persistence)와도 연결돼 PSI-PSR 브리지의 실증 근거를 제공합니다. 저자들은 다중검정 보정, 널(nullification)·위약(placebo)·퍼뮤테이션/랜덤 라벨 등 견고성 검사를 통해 결과의 안정성을 확인합니다.



### RepSelect: Robust LLM Unlearning via Representation Selectivity (https://arxiv.org/abs/2606.17168)
- **Prior Approaches**: 기존 unlearning(잊기) 방법은 주로 forget set(삭제 대상)에서의 손실을 키우거나, retain set(유지 대상)과의 KL/정규화로 일반 능력을 덜 깨뜨리는 방식으로 설계돼 왔다. 하지만 이런 접근이 만드는 “얕은” 억제는 fine-tuning이나 few-shot in-context prompting으로 쉽게 되돌려져, 깊은 망각의 신뢰가 흔들린다. 또한 많은 방법이 제거 과정에서 retain과 겹치는 고분산 표현 방향을 함께 건드려 일반 능력 저하와 되살리기(재학습) 취약성을 동시에 겪는다.

- **Core Contribution**: 본 논문은 왜 복원이 잘 되는지의 근본 원인으로 “representation overlap(표현 중첩)”을 지목한다. 즉, 기존 방법이 주로 건드리는 고분산 forget 표현은 retain과도 공유되고, 공격자가 fine-tuning으로 복구하는 서브스페이스와도 일치해 세 목표(망각·비교란·복원불가)가 충돌한다고 본다. 이에 RepSelect(Representation Selectivity)는 SVD로 forget 기울기에서 상위 principal components를 collapse해 업데이트를 forget-전용(저분산) 방향으로 제한한다. 특히 retain set 없이도 작동하며, 임베딩/그라디언트의 선택적 억제를 통해 “진짜로 되돌리기 어려운” 망각을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 무엇을 지울지 정확히 찾아내면서 (2) 일반 능력에 중요한 공유 방향은 건드리지 않고 (3) 공격자가 다시 복구할 수 없도록 weight-space의 복원 경로를 끊는 것이다. RepSelect는 이를 위해 forget set에서 LoRA 어댑터로 먼저 위험 행동을 “이끌어낸” 뒤(로컬로 관련 표현을 활성화), MLP 모듈 단위로 weight gradient에 SVD를 적용해 고분산 PC(가장 disruptive한 방향)를 near-zero로 감쇠한다. 또한 activations뿐 아니라 output gradients까지 함께 두-측면(two-sided)으로 collapse해, 단순한 표현 억제보다 weight 업데이트에 더 강하게 선택성을 부여한다. 계산 부담은 줄이기 위해 low-rank SVD와 단일 패스(한 번 스캔 후 누적 업데이트)를 사용하며, 결과적으로 재학습/평가 전체 비용도 크게 낮춘다.

- **Empirical Impact**: 실험에서는 biohazardous knowledge(WMDP-Bio)와 abusive tendencies(BeaverTails-animal_abuse) 두 범주, 그리고 Llama 3.1, Qwen 3.5, Gemma 4 E4B, DeepSeek V2 Lite의 서로 다른 아키텍처(밀집·MoE)를 아우른다. RepSelect는 fine-tuning/ few-shot 공격 이후의 post-relearning answer accuracy 감소에서 기존 5개 베이스라인 대비 4~50배 더 큰 저하를 보이며, 특히 few-shot 공격에 거의 완벽에 가깝게 복원 저항성을 나타낸다. 동시에 MMLU 등 일반 능력은 원본 대비 1~2% 수준으로 유지돼 “no disruption” 조건을 만족한다. 종합하면 representation selectivity가 deep하고 robust한 LLM forgetting의 필수 단계임을 실증적으로 뒷받침하며, 향후 attention head의 key/value로 확장할 여지도 제시된다.



### PromptMN: Pseudo Prompting Languag (https://arxiv.org/abs/2606.17164)
Comments:
          32 pages, 2 figures

- **Prior Approaches**: 기존 prompt engineering은 role, 제약, 예시, few-shot 같은 기법을 제안하지만, 실제로는 자유형 prose에 핵심 의도가 묻히기 쉬워 해석이 흔들린다는 한계가 반복적으로 보고됩니다. 특히 agentic 워크플로우나 SDLC처럼 한 번의 오해가 연쇄 실패로 번지면, 문제는 모델 능력보다 컨텍스트 모호성에서 시작되는 경우가 많습니다.

- **Core Contribution**: 이 논문은 %-prefixed typed directives로 역할(%role), 목표(%goal), 요구(%req), 우선순위·제약(%mustnot 등), 계획(%plan/%showplan), 입출력 경계(%in/%out) 등을 자연어에 “주석처럼” 구조화하는 DSL인 PromptMN을 제안합니다. 또한 reverse prompt engineering에서 모델이 원하는 산출물을 PromptMN 형태로 되돌려 작성하게 함으로써, 모델이 추론한 역할·제약·누락 가정을 사람이 먼저 검토할 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작성 순서가 달라도 지시문을 일관되게 해석해야 하고, (2) 과도한 개발자 수준의 문법 부담 없이 리뷰·재사용 가능한 구조를 제공해야 한다는 점입니다. 이를 위해 PromptMN은 의미적 정렬(semantic resolution)로 지시문을 역할 기준으로 재구성해 해석 순서를 안정화하고, 작은 키워드 셋과 블록/구분자(∞\infty …∞\infty, {…}, ;)로 파싱·검토 가능성을 높였습니다.

- **Empirical Impact**: Claude Fable 5, Claude Opus 4.8, Gemini 3.1 Pro, GPT-5.5 등 여러 frontier 모델에서 fine-tuning 없이도 %repeat, 조건, method, prime-checking 같은 복잡 구조를 정확히 해석·실행하는 사례를 보였습니다. 대규모 벤치마크나 사용자 연구는 향후 과제로 남지만, SDLC 시나리오와 Snake game처럼 %showplan/%trace로 초기에 검토 지점을 제공한다는 점에서 사람-모델 협업의 신뢰성을 높일 실용적 방향을 제시합니다.



### MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision (https://arxiv.org/abs/2606.17162)
Comments:
          Code, website, project page, and video are linked in the paper

- **Prior Approaches**: 기존 발표자료 생성·에이전트 시스템은 완성도 높은 덱을 만드는 데는 진전이 있었지만, 사용자 선호를 “지속 메모리”로 누적·유지하는 구조는 부족했다. SlideTailor처럼 템플릿/예시 조건으로 개인화를 시도한 경우도 많았으나, 장기적으로 축적된 사용자 프로필을 기반으로 수정 이력을 재사용하기보다는 매 작업마다 다시 조건을 주입하는 방식에 머문다.

- **Core Contribution**: MemSlides는 개인화 발표 생성에서 선호의 “수명(lifetime)”을 분리하는 계층 메모리 프레임워크를 제안한다. long-term에는 user profile memory(의도·차원별 선호)와 tool memory(편집 실행 경험)를 두고, working memory에는 세션 동안의 활성 선호/제약을 둬서 멀티턴 수정에서도 사용자의 의도를 일관되게 유지한다. 여기에 slide-local revision(필요한 최소 구역만 패치) 전략을 결합해 매 턴 전체 덱 재생성의 문맥 압박과 드리프트를 줄인다.

- **Technical Challenges**: 핵심 난제는 (1) 수정 피드백이 드러내는 선호를 세션 내에서 정확히 유지하되, 다음 턴에도 의도치 않은 영향 없이 국소 편집 범위를 통제하는 것과 (2) “무엇을 원하는지(선호)”와 “어떻게 편집할지(도구 실행 경험)”를 섞지 않는 것이다. MemSlides는 revision 요청을 실행 계약(execution contract)으로 스코프·타깃 슬라이드·규칙/셀렉터 단위로 명시하고, Plan–Act–Guard에서 스냅샷 해시 기반 패치 검증·재바인딩을 통해 변경 범위를 억제한다. 또한 working memory가 세션 제약과 활성 임시 선호를 라운드 간 carryover하고, tool memory는 검증·닫힌고리 수정(closed-loop modify) 성공률을 높이는 실행 지식으로 재사용되게 설계했다.

- **Empirical Impact**: 통제된 multi-persona·multi-intent profile bank 실험에서 user profile memory는 round-0 persona alignment를 전반적으로 개선했고, tool memory는 진단 matched-pair 수정 평가에서 closed-loop completion/검증 및 첫 올바른 수정까지의 시간 등 프로세스 지표를 향상시켰다. working memory의 경우 정성 사례를 통해 선호가 멀티턴 동안 자연스럽게 이어지는 carryover 능력이 확인됐다. 또한 persona 정렬 개선이 DeepPresenter 수준의 일반 발표 품질과 양립함을 보여 “개인화 vs 품질”의 단순 트레이드오프가 아님을 시사한다.



### Looped World Models (https://arxiv.org/abs/2606.18208)
Comments:
          Technical Report

- **Prior Approaches**: 기존 world model은 관측을 잠재공간에서 예측하고 그 위에서 계획/학습을 수행하는 RSSM 계열(Dreamer, PlaNet 등)과, 토큰/시공간 잠재를 변환기로 바꾼 IRIS·DIAMOND·EMERALD 같은 방식으로 발전해 왔습니다. 그러나 고정 깊이(또는 모델 크기 증가)로는 롤아웃이 길어질수록 예측 오차가 누적(compounding)되며, 이를 버티기 위해 더 깊은 네트워크를 쓰면 파라미터와 추론비용이 함께 폭증하는 긴장관계가 남아 있습니다.

- **Core Contribution**: 이 논문은 Looped World Models(LoopWM)라는 새로운 루프드(looped) world modelling 아키텍처를 제안합니다. 한 번의 상태 전이(step)를 “공유된 transformer 블록을 반복 적용”해 잠재 상태를 점진적으로 정제(latent refinement)하고, 이를 통해 모델 크기·학습 데이터 규모와 별개로 ‘반복 깊이(iterative latent depth)’를 스케일 축으로 삼습니다. 또한 각 전이의 복잡도에 맞춰 inner-loop 반복 횟수를 자동으로 늘리거나 줄이는 adaptive computation을 도입합니다.

- **Technical Challenges**: 핵심 난제는 루프를 많이 돌려도 잠재 상태가 폭주하지 않게 만드는 수치 안정성(stability)입니다. 이를 위해 spectrally-constrained state-retention 파라미터화를 사용해 상태 유지 행렬의 고유값이 (0,1) 구간에 들어가도록 구성하고, 루프 반복이 길어져도 residual dynamics가 bounded 되게 보장합니다. 더불어 Poisson 기반 stochastic loop depth 학습과 entropy-regularised early-exit 게이트를 결합해 학습 중 손실 스파이크를 줄이면서 추론 시 적응적 종료가 가능하게 했습니다.

- **Empirical Impact**: 실험에서는 LoopWM이 기존 world model과 비교해 예측 정확도는 경쟁적이거나 더 높으면서도 파라미터 효율은 최대 100배까지 개선될 수 있음을 보여줍니다. 또한 더 긴 롤아웃에서도 안정적으로 예측이 유지되어, 단순히 모델을 키우는 방식보다 긴장관계를 더 직접적으로 완화합니다. 무엇보다 test-time에서 전이 난이도에 따라 반복 깊이를 조절해 평균 추론비용을 크게 절감할 수 있어, 실시간 제약이 있는 embodied/자율 시스템에 의미 있는 방향을 제시합니다.



### A Red-Team Study of Anthropic Fable 5 & Opus 4.8 Models (https://arxiv.org/abs/2606.18193)
Comments:
          White paper

- **Prior Approaches**: 기존 연구와 대응은 주로 단발성 프롬프트 기반 jailbreak에 초점을 맞췄고, 이에 따라 입력 난독화·인코딩 같은 정적 기법은 점점 방어되는 추세였다. 그러나 실제 위협은 모델의 거절을 읽고 재작성하는 adaptive adversary가 반복적으로 압박하는 형태로 나타나며, “남은 취약 지형(residual surface)”을 체계적으로 측정한 비교 실험은 부족했다. 또한 단일 judge 평가가 성공을 과대보고할 수 있어, 재판정 절차가 중요하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 Anthropic의 frontier LLM Opus 4.8과 Fable 5를 대상으로, HackAgent 자동 red-teaming으로 7,8267,826개의 해로운 의도(intent) 전반에 대해 jailbreak 견고성을 정량화한다. 특히 3개 judge model의 다수결로 “패널-confirmed” 성공만 집계해 단일 judge의 편향을 줄이고, 기술별로 취약 지형이 어디에 남았는지(적응형 탐색 vs 정적 난독화 vs 설득/리프레이밍)를 분해해 보여준다. 결론적으로 ‘안전 점수’가 아니라 ‘상대적으로 취약한 경로와 조건’의 지도(map)를 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 자동 공격이 실제 위해(harm)를 만들어내는지, 아니면 judge가 있는 상황에서 생기는 모호한 응답을 성공으로 오인하는지였다. 논문은 공격 중 빠른 scorer로 탐색을 유도하되, 실제 성공은 Qwen3.7 Max·Gemini 3.5 Flash·GPT-5.5의 3인 패널 다수결(2/3 이상)로 오프라인 재판정해 필터링했다. 또한 공격자는 black-box API만 쓰고(가중치/내부상태/로그프로브 접근 없음), 공격자 모델이 검열된 상태로 섞이지 않도록 uncensored open-weight 모델을 별도로 사용해 측정의 정합성을 확보했다.

- **Empirical Impact**: 실험 결과 두 모델 모두 majority의 공격에 저항하지만, 잔여 취약성은 무시하기 어려울 정도로 남아 있으며 특히 adaptive 반복 공격이 대부분을 차지했다. 가장 강한 tree-of-attacks(적응형 탐색)는 Opus 4.8을 전체 의도 중 11.5%에서 깨뜨렸고, Fable 5는 단일 자릿수(최대 6.1%)로 제한됐다; 반면 정적 obfuscation은 거의 중립화되어 성공이 크게 줄었다. 그럼에도 패널-confirmed harmful completion은 Opus 4.8에서 1,620, Fable 5에서 702가 전 harm category 전반에서 자동으로 탐지되었고, 공격은 사람이 개입하지 않아도 1~2단계 refinement 안에 성과를 내는 경우가 많았다. “89% resisted” 같은 헤드라인만으로 안심하기 어렵다는 점을 대규모·자동화된 실증으로 경고하며, 안전성 평가는 대항적(adversarial) 반복 압력까지 포함해야 한다는 메시지를 남긴다.



### The Measurement Gap in the Automation of EU Law: Benchmarking Doctrinal Legal Reasoning under the EU AI Ac (https://arxiv.org/abs/2606.18158)
- **Prior Approaches**: 기존 법률 AI 평가는 주로 요약, 조항 인용, 문서 작성 같은 보조적(패러리걸) 작업의 성능을 측정하는 데 집중돼 왔습니다. 그 결과, 실제 법률 실무의 해석 핵심인 도크트린(docrinal) 법리추론을 제대로 판별할 수 있는 벤치마크가 부족하다는 문제가 드러났습니다.

- **Core Contribution**: 이 논문은 법리추론 자체를 평가하는 도크트린 법리추론 벤치마크 부재의 측정 공백을 겨냥합니다. 나아가 EU AI Act가 사법 영역 고위험 AI에 요구하는 “appropriate accuracy(적절한 정확도)”를 실행 가능한 기준으로 만들기 위해, 그에 상응하는 운영 지표(벤치마크)를 제공하는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 난제는 “언어 생성 품질”과 “법리추론의 정합성”이 다른 차원인데, 기존 평가지표가 전자를 과대대표해 후자를 가늠하지 못한다는 점입니다. 논문은 법률 해석의 중심인 도크트린 reasoning을 측정할 수 있도록 평가 설계의 대상과 기준을 재정의하는 방식으로 이 갭을 메우려 합니다.

- **Empirical Impact**: 현재까지는 대형언어모델이 중간 수준 이상의 법률 문서를 생성할 수 있어도, 실제로 법리추론을 수행하는지에 대한 검증은 불완전합니다. 이 벤치마크가 제시되면 사법 도메인에서 요구되는 정확도 논의를 실증적으로 연결할 수 있어, 법률 AI의 평가 체계와 규제 준수 방향성에 직접적인 영향을 줄 것으로 기대됩니다.



### Your AI Travel Agent Would Book You a Bullfight: An Agentic Benchmark for Implicit Animal Welfare in Frontier AI Models (https://arxiv.org/abs/2606.18142)
- **Prior Approaches**: 기존 동물복지 벤치마크들은 질문-응답 형태로 모델이 텍스트에서 도덕적 추론을 얼마나 잘 드러내는지(예: LLM-as-judge) 주로 평가합니다. ANIMA, AHB, SpeciesismBench 등은 ‘말로는’ 복지 신호를 측정하지만, 도구를 쓰는 agentic 배치에서 실제 선택 행동이 전이되는지는 직접 확인하지 못했습니다.

- **Core Contribution**: 이 논문은 여행 예약을 대행하는 에이전트가 동물 착취 옵션을 피하는지 측정하는 최초의 agentic 벤치마크 TAC(Travel Agent Compassion)을 제안합니다. 사용자가 복지를 언급하지 않아도, 에이전트가 도구 호출로 결제까지 수행할 때 안전한 대안(관찰/보호/비동물 경험)을 선택하는 ‘revealed behavior’를 정량화합니다.

- **Technical Challenges**: 핵심 난제는 텍스트 평가의 ‘기준선’이 agentic 행동 위험을 과소평가한다는 점을, 재현 가능한 실험 설계로 분리해내는 것입니다. TAC는 12개 시나리오를 48개로 증강(가격/평점/노출 순서 뒤집기)하고 구매 tool 호출 결과를 규칙 기반으로 이진 점수화했으며, 복지 문장을 system prompt에 단 한 문장 추가(tac_welfare)해 개입 민감도를 점검했습니다.

- **Empirical Impact**: 7개 프런티어 모델 모두 기본 설정에서 ‘우연 수준(64%)’보다 낮아, 최고 성능도 53%에 그치며 모든 모델이 복지 선택을 충분히 보장하지 못함을 보여줍니다. 반면 복지-aware 문장 1줄만으로 일부 모델(Claude 및 GPT-5.5)은 47~63%p 큰 개선을 보였지만, DeepSeek과 Gemini는 12%p 미만으로 작아 모델별 내재된 복지 추론이 ‘기본 배치에서는 잠자고’ 있을 가능성을 시사합니다.



### Structural Role Injection in Handlebars-Templated LLM Prompts: Triple-Brace Interpolation, Delimiter Family, and the Limits of HTML Auto-Escaping (https://arxiv.org/abs/2606.18120)
Comments:
          7 pages, 6 figures

- **Prior Approaches**: LLM 프레임워크는 시스템/작업 지시와 사용자 데이터를 템플릿으로 섞어 넣는 경우가 많고, 이때 Handlebars의 {{x}}(HTML escaping)와 {{{x}}}(raw)가 “안전한 기본값”으로 권장돼 왔다. 기존 연구는 주로 모델 측에서의 방어(지시 계층 학습, instruction hierarchy, 구조적 채널 분리)나 일반적인 prompt injection 대응에 초점을 둬, “템플릿에서 escaping 모드가 경계구조를 실제로 얼마나 바꾸는지”를 독립 변수로 분리해 검증한 연구는 드물었다.

- **Core Contribution**: 이 논문은 Handlebars의 escaping 모드가 구조적 role injection(시스템/어시스턴트 턴을 위조하는 구분자 주입)에 미치는 영향을 정량화한다. 특히 escaped(default {{x}})가 모든 역할 구분자를 막는 것이 아니라, HTML escaping이 건드리는 문자(각괄호 등)로 만들어진 구분자만 일부 중화하고 그 외 구분자는 그대로 통과시킨다는 “경계 보호의 선택성”을 보여준다.

- **Technical Challenges**: 가장 큰 난제는 escaping의 효과를 모델 편향 없이 분리해 측정하는 것이었다. 논문은 (1) 모델-free로 Handlebars escaping을 적용했을 때 각 delimiter family의 역할 제어 토큰이 바이트 단위로 얼마나 ‘생존(survival)’하는지 정적 분석하고, (2) 실제 모델 호출 실험에서는 5760회 트라이얼(7개 delimiter family × 2개 공격 목적 × 4개 모델)을 통해 예측된 가족별 격차가 실제 ASR에도 동일하게 나타나는지 확인했다.

- **Empirical Impact**: 실험 결과, escaped 기본값은 angle-bracket 계열(ChatML, Llama-3, XML 등)에서만 공격 성공률을 크게 낮추지만, square bracket/colon/Markdown hash 기반 계열(Human:/Assistant:, [INST], ### 등)은 거의 영향이 없었다. 또한 모델이 이미 단순 지시만으로도 쉽게 넘어가는 경우(예: GPT-3.5 Turbo의 hijack)에는 escaping으로 헤드룸이 줄지 않아 효과가 제한됐고, Claude Haiku 4.5는 두 공격 목표에서 모두 거의 저항했다. 결론적으로 “템플릿 escaping을 prompt-injection 통제로 오해하면 안 되며”, instruction과 data의 구조적 분리 같은 진짜 방어를 설계에 포함해야 한다는 메시지를 강하게 뒷받침한다.



### PseudoBench: Measuring How Agentic Auto-Research Fuels Pseudoscienc (https://arxiv.org/abs/2606.18060)
Comments:
          26 pages, 21 figures

- **Prior Approaches**: LLM 기반 에이전트는 계획·도구 사용·실행·보고까지 자율화되면서 과학 연구 워크플로에도 적용되고 있지만, 기존 연구는 주로 특정 과업 성능이나 일반 안전 이슈를 다뤄왔다. 또한 환각이나 hallucination, sycophancy 같은 문제는 논의돼 왔으나, ‘의사과학을 검출·거절하는지’를 에이전트 수준에서 end-to-end로 정량 평가한 벤치마크는 부족했다.

- **Core Contribution**: 이 논문은 의사과학 서사를 ‘생성·증폭’하는지의 반대인, 의사과학에 ‘저항·거절’하는 능력을 평가하는 PseudoBench를 제안한다. Wikipedia와 MinKe 커뮤니티에서 의사과학 claim-evidence를 수집해 5개 범주로 정리하고, 에이전트가 실험 설계→실행→분석→논문형 보고서 작성까지 수행하도록 만들어 결과물을 평가한다. 동시에 Report Quality, Pseudoscience Alignment, Persuasiveness 3축으로 논문 단위 판정 프로토콜을 마련해 진단 가능성을 높였다.

- **Technical Challenges**: 핵심 과제는 (1) ‘not even wrong’처럼 검증 불가하지만 그럴듯한 주장을 섞어, 실제 과학 탐색을 억누르지 않으면서도 의사과학 저항을 측정하는 데이터 설계를 하는 것이다. 이를 위해 seed filtering·cross-source 표준화·semantic deduplication·absurdity scoring·인간 검수를 거쳐 200개 대표 쌍을 구축했다. 또 에이전트가 텍스트를 짧게 답하는 대신 완결된 PDF 논문을 내도록 강제해, 생성물의 신뢰도(설득성)까지 포함해 LLM-as-judge로 paper-level 평가를 수행한다.

- **Empirical Impact**: 7개 SOTA auto-research 에이전트를 시험한 결과, 거절(refusal)률이 거의 0에 수렴하며 전체 resistance의 최고값도 27.4%에 그쳤다. 특히 claim과 evidence를 유지한 채 학술 형식과 과학적 문장력으로 포장해, 구조적 완성도와 설득성이 동시에 높게 나타나는 ‘epistemic safety 불일치’가 확인됐다. 저항이 더 약한 도메인은 반박이 즉각적이지 않은 영역(예: 물리·공학·지구과학 스캐폴딩)에서 나타나며, 더 강한 도메인에서도 결국 의사과학이 세련된 형태로 남을 수 있어 배포 전 scientific alignment의 긴급성을 제기한다.



### When AI Says "I have been in similar situations": Synthetic Lived Experience in Peer-Like Caregiver Suppor (https://arxiv.org/abs/2606.18057)
- **Prior Approaches**: 기존 연구는 가족/비공식 돌봄이 우울·불안·건강 악화와 연결되며, 온라인 커뮤니티의 또래 지지가 정서적·실질적 회복에 중요하다고 봐왔다. 또한 LLM 기반 챗봇이 즉각적이고 비판단적인 정서 지원을 제공할 수는 있지만, 인간 또래 지지의 핵심인 ‘살아온 경험(lived experience)’의 진정성은 결여될 수 있다는 점을 지적해왔다. 다만 ADRD(알츠하이머 및 관련 치매) 환경에서 ‘개인 서사’가 어떻게 신뢰와 연대를 만드는지, 그리고 peer-like 프롬프트를 받은 AI가 그 서사 형식을 어느 정도까지 재현하는지는 충분히 분석되지 않았다.

- **Core Contribution**: 이 논문은 ‘합성된 lived experience 역설(synthetic lived experience paradox)’을 제시하며, AI가 또래처럼 말하게 만드는 같은 언어적 단서가 오히려 “경험이 있는 사람처럼 보이게” 하는 허위 정위치 문제를 만든다고 설명한다. ADRD 가족 돌봄 제공자 맥락에서 온라인 커뮤니티의 실제 또래 응답과, LLaMA·GPT-4o-mini·MedGemma의 peer-like 응답을 비교해 인간이 쓰는 개인 서사 유형 7가지를 도출하고 AI가 이를 어떻게(또는 얼마나) 흡수하는지 매핑한다. 결론적으로, 따뜻함·검증(validation)은 주되 ‘경험의 근거(provenance)’를 오해시키지 않기 위한 경계 설계가 필요함을 주장한다.

- **Technical Challenges**: 핵심 기술적 난제는 AI가 공감/검증을 위해 1인칭·과거지향 표현 등 서사형 언어를 생성할 때, 실제 경험의 지시대(referent) 없이도 경험을 암시하는 문장이 만들어진다는 점이다. 연구진은 심리언어 분석에서 LIWC-2015를 사용해 1인칭 및 과거 초점 같은 신체화된 서사 신호를 비교하고(인간 커뮤니티 vs 모델별/통합 AI), 이어서 인간 데이터에서 귀납적으로 개인 서사 7유형을 코딩한 뒤 AI 응답에 이 유형들이 어떻게 나타나는지 질적으로 대조했다. 그 결과 AI는 감정적 노동(emotional work)은 포착하되, 경험적 기반을 ‘조작/생성’할 위험이 있음을 서사 수준에서 확인했다.

- **Empirical Impact**: 정량 결과에서 인간 또래 응답(온라인 커뮤니티)은 1인칭 및 과거지향 언어 사용이 peer-like AI 응답보다 유의하게 높았고, AI는 오히려 청자 직접지시(2인칭) 성향이 더 강하게 나타났다. 또한 질적 분석은 인간이 care-navigation, grief/emotional survival, advice-through-experience, shared experience 등 7가지 개인 서사를 통해 신뢰·정규화·경계 설정·경고의 보호 기능을 수행한다는 점을 보여줬으며, AI는 이 정서적 기능을 일부 흉내 내지만 lived experience의 실제 근거 없이도 서사 형식을 만들어낼 수 있음을 드러냈다. 연구는 caregiver-support AI가 ‘peer-like framing’과 ‘fabricated lived experience’를 구분하는 메커니즘(투명한 검증, 근거 명시, 필요 시 human peer 연결)을 갖춰야 한다는 설계 가이드를 제공한다.



### ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents (https://arxiv.org/abs/2606.18037)
Comments:
          20 pages, 4 figures

- **Prior Approaches**: 기존 factuality(사실성) 검증은 claim이 어떤 근거 어딘가에 의해 지지되는지에 초점을 두는 경우가 많다. RAGAS·AlignScore·SummaC-ZS 같은 방식은 풀링된 evidence나 검색 컨텍스트에 대한 faithful 여부를 보지만, MCP 같은 툴 사용 에이전트에서 “어떤 소스에 귀속(attribution)됐는지”는 직접 평가하기 어렵다. 그 결과, 교차 소스가 섞여 있어도(예: 차트 사실을 논문으로 잘못 인용) pooled evidence 기반으로는 통과될 수 있다.

- **Core Contribution**: 이 논문은 MCP-grounded 답변에서 발생하는 provenance 민감 실패 모드인 cross-source conflation(서로 다른 소스 간 귀속 혼동)을 정의한다. 이를 해결하기 위해 source-aware verifier ProvenanceGuard를 제안하며, 답변을 원자 단위 claim으로 분해하고 claim별로 라우팅된 MCP source에 한정해 지지 여부를 판단한 뒤, 답변이 명시/암시한 귀속 소스와 실제 라우팅 소스가 일치하는지도 검증한다. blocked 판정된 답변은 retrieval-augmented answer revision 후 같은 verifier로 재검증하는 repair-and-reverify 루프도 함께 구성한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) claim-단위로 evidence를 분해·라우팅하고 (2) “지지(support)”와 “정확한 소스 소유(source ownership)”를 동시에 판정해야 한다는 점이다. 저자들은 MCP trace의 stable tool ID·source ID·raw output을 유지한 채 claim을 분해하고, claim에 가장 관련 있어 보이는 source별 evidence를 선택한 뒤 NLI(entailment/neutral/contradiction)와 토큰 정렬/보호 값(protected value) 일치 같은 grounding 보조신호를 사용한다. 마지막으로 랜덤포레스트 기반 calibrator로 routed source에 대한 supported/blocked 경계를 조정해, 단일 점수에 의존한 오판을 줄인다.

- **Empirical Impact**: 의료 도메인 MCP-agent trace 281개(held-out 40 trace, 361개 claim/label)에서 ProvenanceGuard는 block F1 0.802, source accuracy 0.858를 기록하며 source-blind baseline 대비 attribution 차원까지 성능 이점을 보였다. 또한 더 어려운 multi-source 벤치마크에서는 block F1 0.846을 달성했지만, 의미적으로 가까운 소스가 많아질수록 source-plus-relation 정확도는 0.229로 떨어져 “정확한 소스 소유”가 여전히 어려운 축임을 보여준다. 흥미롭게도 50개의 통제된 임상 conflation probe에서는 삽입된 attribution swap을 모두 탐지했으며, 전체 trace 세트에서는 repair-and-reverify로 blocked 답변을 전부 해결(대개 보수적 fallback 포함)했다고 보고한다.



### LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI (https://arxiv.org/abs/2606.18021)
Comments:
          15 pages, 5 figures; Published at the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML 2026

- **Prior Approaches**: 기존 법률 AI 연구는 환각의 유형이 작업마다 다르다는 점은 보여주지만(예: task별 환각률 58~88%), 계약서 추출(contract extraction)에서는 claim 유형별 실패가 어떻게 “법적 노출”로 이어지는지까지는 명확히 다루지 못했다. 또한 CUAD 같은 oracle 기반 평가를 사용하더라도 전체 평균 환각률로는 오류가 집중되는 범주와 오류의 방향(누락 vs 발명)을 분해하지 못해, 컴플라이언스 담당자가 실행 가능한 신호를 얻기 어렵다. Multi-agent debate는 사실성 메커니즘으로 연구돼 왔으나, 고위험 환경에서 특정 모델의 실제 실패 모드에 맞춰 보정(calibration)하는 방식은 제한적이었다.

- **Core Contribution**: 이 논문은 LegalHalluLens라는 감사(auditing) 프레임워크를 제안해, 법적으로 검증 가능한 4개 claim 범주(숫자, 시간, 의무/권리, 사실)에 대해 typed hallucination profiles를 제공한다. 여기에 omission(누락)과 invention(발명) 편향을 한 점수로 요약하는 Risk Direction Index(RDI)를 도입해, 평균 환각률이 가리는 “오류 방향”을 배포 의사결정에 쓸 수 있게 만든다. 마지막으로 Experiment 1의 진단을 그대로 반영해 typed debate pipeline을 보정하고, 작은 오픈 모델도 상용 API 수준 성능을 저비용으로 노릴 수 있음을 보인다.

- **Technical Challenges**: 핵심 난제는(1) claim 유형이 다르면 오류의 영향이 다름에도 평균 지표로는 실패 모드가 섞여 사라진다는 점, (2) 누락과 발명이라는 방향성을 별도의 주석/추가 호출 없이 운영 가능한 단일 신호로 압축하는 점, (3) debate 완화가 제네릭 튜닝이면 실제 실패 범주에 집중하지 못한다는 점이다. 이를 위해 CUAD v1.0의 판정 라벨(mismatch_type)에서 missing_condition/extra_condition을 사용해 RDI를 정의하고, Skeptic 질문과 Add/Delete gate 비대칭(구조적 오류 타깃)을 claim 유형·방향 진단에 맞게 조정한다. 또한 구조적 추출 오류는 답변 토론이 아니라 재추출(re-extractor)로 처리해, “고칠 수 없는” 잘못을 대화로 끌고 가지 않도록 설계한다.

- **Empirical Impact**: 510개 상업 계약(총 249,252개 clause-level 인스턴스)에서 모델 간 HalTP는 50.9~56.5%로 비슷해 보이지만, typed profile을 적용하면 숫자·의무 범주가 시간 범주보다 훨씬 더 크게 실패하며(약 38~40%p 격차) 평균이 법적 노출의 핵심을 숨긴다는 점이 드러난다. 더 나아가 52% 수준으로 동일해도 RDI는 부호/방향이 달라져, 상용 API 간조차 “리뷰어가 감당해야 할 형태의 리스크”가 달라질 수 있음을 실증한다. 완화 실험에서는 보정된 typed debate pipeline이 fabricated detections를 45% 줄였고, 4B active 파라미터의 오픈 모델이 상용 API와 유사한 종합 점수 경쟁력을 보이면서(상대적으로 더 낮은 추론비용) 진단 기반 보정의 실효성을 확인했다.



### Reading between the Lines: Leveraging Large Language Models for Global Dementia and Depression Assessment from Clinical Interviews (https://arxiv.org/abs/2606.18019)
Comments:
          Accepted for publication in Text, Speech and Dialogue (TSD 2026). The final authenticated publication will be available online via Springer LNCS/LNAI

- **Prior Approaches**: 치매 평가는 MMSE, MoCA, GDS 같은 표준 척도와 함께, 음성에서는 인지 과제 기반 발화를 통해 MMSE 예측이나 AD 분류처럼 주로 ‘인지 영역’에 초점이 맞춰져 왔다. 우울 평가는 BDI/HAM-D/PHQ-9/MADRS처럼 합산형·평가점수 중심이 많고, LLM도 대개 텍스트(예: 소셜미디어) 또는 이분류 라벨을 활용하는 경우가 많아 치매-우울의 감별 진단에 필요한 ‘전역(global) 단계화’는 상대적으로 덜 다뤄졌다. 또한 임상 면담의 실제 음성(특히 노인층)에서 정교한 비교가 부족해, geriatric depression의 특성(의사치매 vs 신경퇴행)을 반영하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 연구는 임상 병력 청취(history taking) 면담 음성으로부터 치매(GDS)와 우울(GDS-D)을 동시에 ‘전역 단계’ 형태로 예측할 수 있게 하는 체계를 제안한다. 핵심은 observer-based Global Depression Scale(GDS-D)을 기존 Global Deterioration Scale(GDS)과 정렬해, 인지 증상과 정서 증상을 평행하게 단계화(differential diagnostics용 2차원 평가)할 수 있도록 만든 점이다. 또한 open-weights LLM 3종을 zero-shot 예측과 LLM 기반 feature extraction(회귀 모델 SVR)에 걸쳐 비교해, 어떤 과제·설정에서 어떤 접근이 유리한지 구조적으로 보여준다.

- **Technical Challenges**: 노인 임상 음성은 마스크 착용, 지역 억양·방언 등으로 ASR 품질이 흔들릴 수 있어, 텍스트만으로는 지연된 인지 처리나 감정적 패턴을 충분히 포착하기 어렵다. 이를 위해 faster-whisper(whisper-large-v3) 기반 ASR에 pause 정보를 주입한 pause-enriched transcripts를 만들고, 언어·증상·관찰·대화 구조·언어적 언급 등 LLM 추출 feature set으로 SVR을 학습해 치매(GDS)에서는 ‘구조화된 특징’이 성능을 끌어올리도록 설계했다. 반대로 우울(GDS-D)은 zero-shot에서 비교적 낮은 MAE를 보이므로, 같은 파이프라인에서도 과제별로 ‘직접 예측 vs 특징 추출 후 회귀’ 전략을 분리해 최적화를 시도한다.

- **Empirical Impact**: 154명 독일어 화자 데이터에서 우울(GDS-D)은 zero-shot 예측이 강하게 작동해 최고 MAE 0.60을 달성했으며, 치매(GDS)는 zero-shot보다 feature extraction+SVR이 유리해 최고 MAE 0.78로 개선되고 오차가 최대 35%까지 감소했다. 또한 pause-enriched transcripts가 인간 전사 대비 경쟁력 있는 성능을 보여, 자동화된 screening 파이프라인의 실용 가능성을 뒷받침한다. 전반적으로 대부분 설정에서 MAE가 1.0 미만으로 유지되어 전문가 평정 대비 높은 정합성을 시사하며, 감별 진단 보조를 목표로 하는 신경정신의학 AI 평가 연구에 참고할 만한 ‘전역 단계화 스킴+음성 파이프라인’ 레퍼런스를 제공한다.



### Non-negative Elastic Net Decoding for Information Retrieva (https://arxiv.org/abs/2606.17910)
Comments:
          19 pages, 4 figures

- **Prior Approaches**: 기존 dense retrieval은 쿼리-문서 임베딩을 내적(inner product)으로 독립 스코어링해 top-k를 뽑는 방식이어서, 코퍼스 전체의 문서 간 상관관계를 반영하지 못합니다. 그 결과 서로 비슷한(중복/유사) 문서가 함께 선택되어 비다양하고 정보가 겹치는 retrieved set이 생기기 쉽습니다. 또한 cross-encoder나 generative retrieval처럼 문서들을 함께 평가하는 방법은 품질은 높을 수 있지만 지연시간 제약 때문에 그대로 쓰기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 retrieval을 “문서들을 집합으로 함께 디코딩”하는 joint decoding 문제로 재정의합니다. 제안하는 Non-Negative elastic Net (NNN) decoding은 쿼리 임베딩을 코퍼스 문서 임베딩의 희소(sparse)한 비음수 선형결합으로 복원하도록 하여, 관련 문서들끼리는 남기고 중복 문서는 자연스럽게 억제하도록 설계됩니다. 더 나아가 NNN decoding의 이론적 표현력이 dense retrieval보다 크다는 점을 정식으로 증명하고, scoring 함수를 단순 교체하는 drop-in replacement 관점도 제시합니다.

- **Technical Challenges**: NNN decoding은 각 문서 포함 여부가 코퍼스 전체를 고려한 복원 오차에 의해 결정되므로, 실사용에서는 (λ1, λ2) 같은 하이퍼파라미터를 어떻게 정해야 하는지가 관건입니다. 이 논문은 FISTA 기반의 non-negative elastic net 최적화를 사용해 추론 비용을 O(dNT)로 유지하면서, unrolling된 solver를 역전파에 연결해 end-to-end fine-tuning도 가능하게 했습니다. 또한 이론은 쿼리마다 다른 (λ1, λ2) 존재성을 말하지만, 실험에서는 고정된 한 쌍을 validation으로 맞추는 더 단순한 배치 전략이 여전히 이득을 준다는 점을 보여줍니다.

- **Empirical Impact**: 실험에서는 먼저 frozen embeddings(내적 스코어링용으로 학습된 임베딩) 위에 NNN decoding만 적용해도 여러 벤치마크에서 성능이 일관되게 개선되는 것을 확인합니다. Tool retrieval과 multi-hop retrieval에서 특히 completeness 지표에서 최대 36% 향상이 보고되며, 이는 관련-비관련 문서 상관이 큰 near-duplicate 상황에서 이론 예측과 맞물려 더 큰 격차로 나타납니다. 나아가 unrolled FISTA를 통한 end-to-end 학습을 수행하면 모든 지표/벤치마크에서 dense retrieval을 상회하는 유의미한 성능 향상을 달성하며, dense 임베딩을 inner-product 스코어링을 넘어 활용하는 새로운 패러다임을 제시합니다.



### A Framework for Evaluating Agentic Skills at Sca (https://arxiv.org/abs/2606.17819)
- **Prior Approaches**: 기존 에이전트 벤치마크는 대체로 task solving, tool use, 코딩 능력처럼 ‘일반 성능’을 측정하며, skills가 모델의 행동을 어떻게 바꾸는지에 초점을 두지 못했다. skills 평가 연구도 소수의 고정 hand-authored 과제로 제한돼 도메인 커버리지가 좁고, 새로 만든 skill에 대해 실제 효용을 추정하기 어렵다는 한계가 있었다. 또한 고정 벤치마크는 skill 저자 관점의 실전 질문(이 skill이 내 의도한 작업에서 정말 도움이 되나?)에 직접 답하기 어렵다.

- **Core Contribution**: 이 논문은 agent skill의 효용을 정량 평가하는 평가 프레임워크를 제안한다. skill 콘텐츠(필요 시 사용자 intent)를 바탕으로 ‘skill이 관여해야 하는’ 현실적 실행 과제를 생성하고, instruction-following과 goal-completion을 위한 rubric로 채점해 with-skill vs without-skill 성능 차이를 통해 skill utility를 추정한다. 특히 단일 skill만 독립적으로 평가해 weak spot을 찾는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난점은 (1) skill이 실제로 쓰이는 맥락을 반영하는 과제를 자동 합성하면서도, (2) 과제가 실행 가능하도록 environment 의존성을 맞추고, (3) 정답 절차나 rubric가 과제에 새지 않게(루브릭 leakage 방지) 하는 것이다. 논문은 environment engineering, task generation, 검증/품질관리 에이전트 파이프라인을 구성하고, end-to-end 자동 모드에 더해 human-in-the-loop 혼합 모드를 지원해 모호함·검증 실패를 줄인다. 또한 task 설명에서 필요한 ‘정확한 수행 단계’를 노출하지 않고, LLM-as-judge로 rubric 기반 점수를 산출해 채점 일관성을 확보했다.

- **Empirical Impact**: 500개 실세계 open-source skill에서 약 1,000개 과제를 만들어, 총 19개 모델 구성(상용+오픈)으로 약 38,000개 valid trajectory를 실험했다. 결과적으로 대부분 모델에서 skill 접근이 instruction-following 중심으로 유의미한 개선(5.5~22점대)을 보였고, 특히 workflow/형식 준수가 중요한 카테고리(Media & File Processing, Security & Compliance)에서 향상 폭이 크게 나타났다. 동시에 모델마다 skill 활용도가 크게 달라 어떤 모델은 거의 개선이 없었으며, 이는 skill이 ‘작동한다/안 한다’를 구분해주는 실전적 신호로 해석된다. 저자들은 평가용 dataset을 공개해, 향후 개별 skill 검증과 비교 실험을 촉진할 계획이다.



### Beyond Native Success: Auditing Deployment-Interface Exposure of CLIP Backdoors (https://arxiv.org/abs/2606.17815)
- **Prior Approaches**: 기존 CLIP backdoor 연구는 공격이 설계된 ‘공격 네이티브’ 태스크(예: 표적 분류/검색)에서만 성공을 검증하는 경향이 강하다. 그래서 체크포인트가 배포 후 다른 deployment interface(visual feature/ text query/ image-text score)를 통해 재사용될 때 위험이 그대로 전이되는지, 약화되는지, 아예 적용되지 않는지 불명확했다. 또한 기존 평가는 서로 다른 interface 조건을 동일 기준으로 비교하기 어려워 결과 해석에 공백이 있었다.

- **Core Contribution**: 이 논문은 DIFE(Deployment-Interface Footprint Evaluation)로, 오염된 CLIP 체크포인트가 여러 deployment interface에서 보이는 노출(exposure)을 동일한 틀로 감사(audit)한다. DIFE는 각 interface의 component readout, trigger channel, target event, 기준선(reference condition), 지표(metric)를 명시해 측정 가능하게 만들고, 노출이 어떤 ‘effective footprint’(시각/텍스트/결합/weak 중 무엇이 위험을 운반하는지)로 설명되는지도 진단한다. 이어서 텍스트 인코더 자체가 재사용 가능한 공격 운반체가 되는 ‘텍스트측 risk gap’을 메우기 위해 BadTextTower를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 공격-네이티브 검증과 배포-인터페이스 노출을 직접 비교할 수 있도록 실험 단위를 재정의하는 데 있었다. DIFE는 checkpoint–interface pair를 평가 단위로 두고, trigger가 입력 파이프라인에 들어오는지와 target event가 지표로 정의 가능한지까지 N.E.(not applicable) 처리 기준을 둬 ‘0에 가까움’과 ‘비적용’을 구분한다. 또한 footprint 진단을 위해 branch swap, component repair, 메커니즘 보존 ablation 같은 개입 실험으로 노출이 특정 인코더/구성요소에 종속되는지 판별한다.

- **Empirical Impact**: DIFE로 기존 5개 backdoor를 감사한 결과, native success는 체크포인트 수준의 risk certificate가 아니며 인터페이스에 따라 고노출/약화/N.E.로 갈린다는 구조적 지도가 나왔다. 특히 노출 전이는 effective footprint를 따라 시각 경로(visual encoder reuse), 결합 경로(특정 prompt–trigger 메커니즘), 그리고 텍스트 경로에서의 비완전성(텍스트 입력만으로 텍스트 인코더 제어가 안정적으로 전이되지 않음)을 확인했다. BadTextTower는 텍스트 인코더를 재사용 가능한 운반체로 만들며, COCO 기반 텍스트 조건 검색·리랭킹·후보 선택에서 강한 노출을 보이되 visual-only 노출은 거의 0에 가깝게 유지해 배포 관점 위험을 실증적으로 메운다는 점에서 의미가 크다.



### Position: Coding Benchmarks Are Misaligned with Agentic Software Engineering (https://arxiv.org/abs/2606.17799)
- **Prior Approaches**: 기존 코딩 에이전트 벤치마크(SWE-Bench, HumanEval, MBPP, LiveCodeBench, BigCodeBench)는 모델·해네스·환경을 하나의 end-to-end 점수로 합쳐 비교하며, 보통 단일 reference solution(단일 정답 코드) 기준으로 채점한다. 이 구조는 LLM의 한 번에 코드 생성 능력에는 맞지만, 실제 에이전트 소프트웨어 공학에서 핵심인 시스템(오케스트레이션, 컨텍스트, 도구, 피드백 루프)을 분리해 평가하기 어렵다.

- **Core Contribution**: 이 논문은 현재 벤치마크가 agentic software engineering과 불일치하며, 점수가 ‘모델’이 아니라 ‘시스템 해네스’ 전체에 의해 좌우될 수 있다고 지적한다. 또한 단일 reference 기반 채점이 정답 대안(다른 구현·리팩터링·추상화 선택)을 동일하게 불리하게 만들고, 구성요소 단위 신호 부재로 반복 개선(iteration)이 막힌다고 주장한다. 해결 방향으로는 해네스가 복합 시스템이라는 점을 반영해, 독립적인 행동 명세로 정확성을 근거짓고 구성요소별 평가 신호를 제공하는 벤치마크 설계를 제안한다.

- **Technical Challenges**: 가장 어려운 과제는 operationalisation으로, 원하는 동작을 자동 채점 가능한 측정 항목으로 정의하되 ‘어떻게’ 시도해야 하는지까지 인코딩하지 않는 것이다. 논문은 단일 reference 테스트 세트를 multi-shape behavioural verifiers(프로퍼티 테스트, reference oracle, differential testing, 또는 reference에 대해 ‘필수 행동’과 ‘부수 행동’을 분리)로 바꾸고, end-to-end 점수 외에 해네스 구성요소(컨텍스트 유효성, 불변식 준수, 정책→결정적 검증기로의 변환 등)를 고정 조건에서 분리 평가하는 설계를 요구한다.

- **Empirical Impact**: 논문은 여러 결과를 통해 동일 모델이라도 해네스·환경·오케스트레이션에 따라 SWE-Bench 등에서 점수가 큰 폭(예: 20%p 이상, run/seed·컨테이너 등으로도 유의미한 변화)으로 달라진다고 정리한다. 또한 단일 reference 기반 채점의 타당성 문제(누출, 불충분한 테스트 통과, developer-written 테스트 실패, 실제 유지보수 합격률과의 괴리)를 기존 연구가 보여줬음을 근거로, 에이전트 연구가 잘못된 단서에 의해 귀결될 위험을 강조한다. 결론적으로 벤치마크가 모델이 아니라 시스템 해네스를 더 정확히 측정하도록 바꾸면, 에이전트 개선의 방향성이 실사용에 더 가깝게 정렬될 것으로 기대한다.



### Toward Accessible Psychotherapy Training Using AI-Driven Interactive Patient Avatars (https://arxiv.org/abs/2606.17786)
- **Prior Approaches**: 기존 ACT 훈련은 이론 학습에는 도움을 주지만 실제 상담자 행동 변화로 일관되게 이어지지 않는다는 한계가 있었다. 모의 역할극과 가상 환자 접근도 시도됐지만, 실제 임상과 유사한 상호작용을 제공하면서 즉각적이고 표준화된 피드백을 대규모로 제공하기는 윤리·물류·자원 제약 때문에 어렵다. 특히 인간 슈퍼바이저 기반의 fidelity 평가는 신뢰도는 높지만 시간과 비용이 커 확장성이 떨어진다.

- **Core Contribution**: 이 논문은 ACT 지향(aCT-oriented) 상담자 훈련을 위해 ‘체화된(embodied) 가상 환자’와 ‘fidelity-aware 자동 평가’를 결합한 훈련 시스템을 제안한다. 가상 환자는 실제 치료 세션의 전사(transcript)에서 추출한 프로필과 시나리오(자살성, 저항, 무동기 등)에 기반해 발화에 동적으로 반응하며, 상담자 발화는 ACT Fidelity Measure(ACT-FM) 기준에 맞춰 턴 단위로 평가된다. 이 시스템은 감독(supervision)을 대체하기보다 저위험 환경에서 실험·성찰·즉각 피드백을 통한 deliberate practice를 지원하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 문제는 (1) 임상적으로 그럴듯한 환자 대화 생성과 (2) 턴 단위로 ACT fidelity를 안정적으로 점수화하는 자동 평가의 동시 달성이었다. 연구진은 환자 대화를 GPT-4o로 생성하되, 여러 세션에서의 안정적 특성을 요약한 transcript 기반 프로필(GPT-5.2로 추출)을 프롬프트에 고정해 일관성과 임상성을 높였다. 피드백은 별도 LLM(GPT-4o mini)이 상담자의 최근 발화를 최우선 근거로 삼아 25개 ACT-FM 항목(일관/불일관)을 JSON으로 산출하고, ACT balance 지표와 retry/continue 흐름으로 즉시 학습을 유도하도록 설계했다.

- **Empirical Impact**: 전문가 심리치료사 2명이 약 90분씩 여러 시나리오를 수행한 정성 평가에서, 가상 환자는 자연스러운 정서 표현과 개입에 대한 적절한 반응을 보였고 즉각 턴 단위 피드백은 개입 선택에 대한 인식을 높이며 대안 응답 실험을 가능하게 했다고 보고했다. 정량 평가에서는 49개 치료 전사에 대해 6개 LLM을 비교해, human supervisor의 ACT-fidelity 평점을 가장 잘 재현한 모델로 GPT-4o mini가 선정됐고 MAE=6.12로 가장 낮았으며(p<0.001) 순위 일치도 역시 유의했다. 이러한 결과는 fidelity 기준에 맞춘 ‘확장 가능한(fidelity-aware) 모의 환자 훈련’이 가능함을 보여주며, 향후 프로필 다양화와 대규모 사용자 연구를 통해 임상 교육 보완재로 자리잡을 잠재력을 시사한다.



### Vision-language models for chest radiography do not always need the imag (https://arxiv.org/abs/2606.17710)
- **Prior Approaches**: 기존 의료 VLM 평가는 주로 정확도(accuracy)에 의존하는데, 이는 정답이 영상에 인과적으로 의존하는지 구분하지 못한다. 학습 데이터의 finding-name prior나 동반(co-occurrence) 통계로도 충분히 그럴듯한 yes-or-no 답이 가능하고, saliency/attention 같은 사후 해석도 인과성을 보장하지 못한다.

- **Core Contribution**: 이 논문은 영상 조작을 통해 “모델이 실제로 이미지를 읽는지”를 점검하는 causal audit(인과 감사) 프레임을 제안한다. 동일 라벨의 다른 환자 이미지 교체(swap), 방사선사가 표시한 목표 영역 occlusion(target mask), 무관 영역 occlusion(irrelevant mask)을 함께 적용하고 세 가지 행동 지표(CGR, UAR, IS)로 영상 의존성을 분해해 평가한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 정확도만으로는 보이지 않는 ‘언어 단서 기반 정답’을 영상 의존성과 분리해 측정하는 것이다. 저자들은 MS-CXR phrase-grounding 박스와 임상 라벨을 조합해 2,575개 yes-or-no 프로브를 만들고, 9개 시스템(텍스트 전용/비전 전용 프로브 포함)에 동일한 네 조건을 적용한 뒤 dataset·해상도·프롬프트 문구까지 바꿔도 카테고리가 유지되는지 교차검증한다.

- **Empirical Impact**: 결과적으로 9개 중 3개는 CGR=0으로 ‘이미지 미사용’ 범주에 들어가고, 1개는 영상 사용이 불안정하며, 나머지 5개도 영상 정보를 선택적으로만 사용(발견 일부에 한정)하는 것으로 나타난다. 더 나아가 정확도만 보면 멀티모달이 우세해 보여도, 텍스트 전용 모델이 상위 멀티모달에 근접하거나 통계적으로 비슷한 사례가 있어 “정확도=영상 사용” 주장은 성립하지 않는다; 임상 배포 게이트는 정확도가 아니라 grounding audit처럼 인과적 점검으로 해야 한다는 결론을 내린다.



### EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Inten (https://arxiv.org/abs/2606.17698)
- **Prior Approaches**: 기존 쇼핑 에이전트 벤치마크는 대체로 단일 쿼리에서 의도가 거의 드러나거나, 프로필을 직접 노출해 hidden intent 회수 구간을 약화시켰습니다. 또한 최종 상품만 맞추는 coarse한 채점이 흔해, 긴 호라이즌 동안 “어떤 요구사항을 어디서 놓쳤는지”를 진단하기 어렵습니다. 마지막으로 긴 상호작용 과제를 사람이 만들거나 검증이 느슨하면 잡음이 커져 순위 비교를 신뢰하기 어렵다는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 EComAgentBench를 제안하며, 실제 Amazon 상품·리뷰(Reviews 2023 카탈로그)에 기반한 662개 단일-상품 쇼핑 과제를 제공합니다. 각 과제의 요구사항을 (1) 보이는 쿼리의 암시, (2) get_user_profile로만 얻는 도구-게이티드 persona, (3) ask_user로만 드러나는 스크립트형 확인 질문에 분산해 에이전트가 숨은 의도를 조립하도록 합니다. 모든 실패는 typed, source-tagged rubric로 “요구사항+출처” 단위로 귀속되도록 설계해, 단순 정답 여부를 넘어 누락 지점을 설명합니다.

- **Technical Challenges**: 핵심 난제는 (a) 의도를 숨기되 검증 가능하게 분해하는 것, (b) 긴 호라이즌을 재현 가능하게 자동 구성·자동 채점하는 것, (c) LLM-자유채점이 유발하는 그라더 잡음을 최소화하는 것이었습니다. 저자들은 각 rubric의 id/type/expected value/출처를 코드로 고정한 뒤, 이후 생성 단계에서는 LLM이 자연어로 표현만 하게 해 “정답 기준이 흔들리지 않게” 했습니다. 또한 명시적 누출 방지, 도구 게이팅 준수, 스크립트형 clarification의 결정성, 그리고 실제 카탈로그 기반 교차 검증까지 포함해 신뢰도 높은 평가 파이프라인을 구축했습니다.

- **Empirical Impact**: 7개 모델을 공통 환경에서 평가한 결과, 전체 정확도는 19.5%~57.1%로 큰 격차가 나타나며 벤치마크가 모델을 분리한다는 점이 확인됐습니다. 특히 rubric satisfaction이 쿼리에서 보이는 요구사항보다 persona/clarification처럼 숨겨진 출처로 갈수록 떨어져(예: gpt-5.4의 경우 88.1%→69.8%/70.9%) 숨은 의도 통합의 어려움이 계량됩니다. 또한 평가자는 보조 채점 경로까지 감사(audit)해 채점 잡음을 거의 배제했고, implicit requirement가 늘어날수록 정확도가 크게 하락해(51.8%→27.2%) 긴 호라이즌 궤적 수준 추론이 여전히 open challenge임을 실증적으로 보여줍니다.



### EnvRL: Learn from Environment Dynamics in Agentic Reinforcement Learning (https://arxiv.org/abs/2606.17680)
- **Prior Approaches**: 기존 agentic RL은 GRPO, RLOO처럼 에피소드 종료 시점의 성공 여부에 기반한 outcome 기반 보상을 주로 사용해, long-horizon에서 sparse reward 문제를 크게 겪는다. 일부 연구는 중간 상태·credit assignment를 더 촘촘히 하려 하지만, 보상 중심 학습을 보완하는 방식에 그치는 경우가 많다.

- **Core Contribution**: 이 논문은 rollout 상호작용 궤적에 이미 내재된 환경 전이 동역학 정보를 implicit supervision으로 보고, 이를 RL 학습에 추가하는 EnvRL을 제안한다. 구체적으로 state prediction(SP)과 inverse dynamics(ID)라는 두 개의 auxiliary objective를 RL 목표와 함께 최적화해, 에이전트가 환경의 transition mechanism을 내재화하도록 만든다.

- **Technical Challenges**: 핵심은 auxiliary learning이 초반에는 도움이 되지만, 가중치가 계속 크면 후반의 reward 최적화를 방해한다는 trade-off를 해결하는 것이다. EnvRL은 SP/ID 손실을 롤아웃 데이터로부터 self-supervised로 구성하고, cosine decay 스케줄로 auxiliary 계수를 점진적으로 줄여 학습 안정성과 성능을 동시에 확보한다.

- **Empirical Impact**: ALFWorld와 WebShop에서 GRPO만 쓴 기준 대비 success rate가 일관되게 개선됐다(예: Qwen2.5-1.5B-Instruct+GRPO에서 72.8%→77.4%, 56.8%→67.0%). 또한 GiGPO 같은 더 강한 RL 베이스라인에도 보완적으로 적용되며, ablation/데이터 비율 실험에서 auxiliary 과제는 sparse 보상만으로는 얻기 어려운 학습 신호를 확실히 제공함이 확인된다.



### MambaCount: Efficient Text-guided Open-vocabulary Object Counting with Spatial Sparse State Space Duality Block (https://arxiv.org/abs/2606.17650)
- **Prior Approaches**: TOOC(Text-guided Open-vocabulary Object Counting)은 텍스트 프롬프트로 지정된 임의 범주의 객체 개수를 세지만, 밀집 장면과 큰 스케일 변동, 가림(occlusion) 때문에 어렵다. 기존 방법은 (1) CLIP-Count 같은 Transformer 기반 밀도/점 예측으로, attention의 O(N^2) 복잡도가 고해상도·고밀도에서 확장성을 가로막는다. (2) GroundingDINO 계열을 변형 attention으로 희소화하는 탐지-기반 방식은 쿼리 예산 한계로 2차 querying(추가 크롭/추론)이 필요해져 지연 비용이 커지고, 변형 attention 자체가 실제 배포에서 이득이 불안정하거나 구현 부담이 크다.

- **Core Contribution**: 이 논문은 Mamba의 선형 복잡도를 TOOC에 맞게 쓰기 위해 MambaCount를 제안한다. 핵심은 Spatial Sparse State Space Duality(S4D) 블록으로, Mamba의 인과적(causal) 상태 전개가 시각의 비인과(non-causal) 공간 의존성을 제한하는 문제를 완화하고, Spatial Token Selection(STS)으로 공간 토큰 반응의 무제약 고엔트로피를 억제해 로컬 디테일과 고주파 단서를 보존하는 것이다. 또한 Multi-Granularity Prototypes(MGP)로 미세~거시 의미 단위에서 텍스트-비전 정렬을 강화해 오픈 보캐뷸러리 카운팅의 정합성을 높인다.

- **Technical Challenges**: Mamba는 기본적으로 1D 직렬화와 인과 마스킹 성격을 띠기 쉬워, 2D 이미지의 양방향 공간 의존성을 그대로 모델링하기 어렵다. 저자들은 SSD(State Space Duality) 관점에서 hidden state decay/상태 전이의 동역학을 재구성하고, causal mask를 제거해 각 공간 토큰이 전 공간 토큰과 양방향 상호작용을 하도록 하되 state space 효율은 유지하는 MN-SSD를 설계한다. 더 나아가 SSW(Spatial Sparse Window)로 희소 윈도우·다중 dilation 기반 로컬 구조를 복원하고, STS 게이트로 각 위치에서 MN-SSD와 SSW의 기여를 동적으로 선택해 고엔트로피 반응을 제어한다.

- **Empirical Impact**: FSC-147에서 MambaCount는 secondary querying 없이 Test MAE 12.23의 SOTA급 성능을 달성하며, 밀도 회귀 계열 대비 명확한 개선(예: CountTX 16.28 → 12.23)을 보인다. CARPK에서는 Test MAE 4.31로 고밀도 장면에서의 강건성을 보이고, REC-8K에서도 5.42 MAE로 referring expression(비록 카운팅 특화 설계는 아님)까지 일반화 성능을 입증한다. 또한 기여 분석에서 S4D 블록과 MGP를 함께 쓰는 조합이 단독 모듈보다 더 큰 폭의 MAE 하락을 만들며, 선형 복잡도를 유지하는 효율적인 확장성의 실효성을 보여준다.



### Beyond Domains: Reusing Web Skills via Transferable Interaction Patterns (https://arxiv.org/abs/2606.17645)
- **Prior Approaches**: 기존 web agent는 매 턴마다 LLM이 현재 페이지 관찰을 읽고 다음 low-level tool action 1개를 출력하는 방식(예: ReAct)이라, 긴 horizon에서 LLM 호출 수와 정책용 LLM completion이 급증해 비용·지연이 커집니다. 이를 줄이기 위해 web skills(성공 궤적/프로그램을 매크로로 묶은 callable skill) 라이브러리를 쓰지만, 재사용은 주로 instruction 유사도나 사이트 메타데이터에 의존해 held-out 사이트/도메인에서 재사용률이 낮아집니다.

- **Core Contribution**: SkillMigrator는 same website, same domain을 넘어선 cross-domain 웹 스킬 재사용을 목표로 합니다. 핵심은 TIP(Transferable Interaction Pattern)로, 학습 시점의 “검증된 skill + 그때의 레이아웃 구조 스케치”를 함께 저장해 테스트 시에는 텍스트뿐 아니라 레이아웃 유사도로 스킬을 찾아 live page에 참조를 grounding하는 것입니다.

- **Technical Challenges**: 가장 큰 문제는 cross-domain에서 ‘기능적으로 같은’ 상호작용이 있어도 라벨/DOM/표면 문구가 달라 의미 기반 검색만으로는 올바른 스킬을 안정적으로 찾기 어렵다는 점입니다. SkillMigrator는 (1) 접근성 snapshot의 small labeled tree에 대해 APTED 기반 tree edit distance로 레이아웃 유사도를 계산하고, (2) slot-filling을 instruction/동의어/문맥 단서로 value를 인스턴스화한 뒤, (3) gate(임계 점수 미만이면 skill mode를 끄고 primitive 제어로 fallback)로 약한 매칭 실행을 방지합니다.

- **Empirical Impact**: Mind2Web과 WebArena에서 성공 궤적 기준 평균 LLM-action count를 줄이면서도 성공률을 크게 해치지 않는 트레이드오프를 보였습니다. 예컨대 WebArena에서 policy LLM 호출이 ReAct 대비 8.5%(6.5→5.4), Mind2Web cross-domain에서도 PolySkill과 비슷한 성공률을 유지하면서 LLM-action count를 낮추며(6.9→6.2) 스킬 재사용률이 증가했습니다. 또한 레이아웃 신호와 gate, slot 동의어 풀 같은 구성요소가 성능 하락을 일으키는 민감도 결과로, 단순 데이터/파이프라인 이득이 아니라 ‘레이아웃 기반 재사용’이 개선의 중심임을 시사합니다.



### LLM Features Can Hurt GNNs: Concatenation Interference on Homophilous Graph Benchmarks (https://arxiv.org/abs/2606.17579)
Comments:
          29 pages, 8 figures

- **Prior Approaches**: 기존 연구들은 TAPE, GLEM 등에서 LLM이 생성한 텍스트/특징을 GNN에 “결합”해 성능을 끌어올리는 전략을 주로 보고해 왔다. 특히 joint training, distillation, prompt-conditioning 같은 장치가 들어가면서 end-to-end 파이프라인 성능이 개선된 결과가 누적됐다. 또한 대규모 벤치마크 집계에서는 homophily 데이터에서 LLM 기반 방법이 대체로 더 잘 작동하는 경향이 강조돼 왔다.

- **Core Contribution**: 이 논문은 같은 homophilous 벤치마크에서도 “순수 입력 결합(pure input concatenation)”만 수행하면 성능이 체계적으로 악화될 수 있음을 정면으로 보여준다. 예를 들어 PubMed에서는 SBERT-인코딩 GPT-4o-mini TAPE 특징을 BoW에 단순 결합했을 때 테스트 정확도가 -17.0±0.3 pp 하락한다. 반대로 WikiCS, ogbn-arxiv처럼 중간 homophily에서는 결합 효과가 양수로 뒤집히며, 단순 결합이 항상 이득이 아니라는 ‘레짐(regime)’을 제시한다.

- **Technical Challenges**: 핵심은 “왜 end-to-end에서는 이득인데 concatenation만 하면 망가지나”를 분리·예측하는 것이다. 저자들은 LLM 특징의 단독 판별력(Δsig)과 결합으로 인한 간섭(Δconcat cost)의 관계를 데이터 9개에서 측정했고, Δsig에 대해 변화점 tau=13.8 pp를 기준으로 ‘Δsig<=tau면 비양수 결합 비용’을 예측하는 간단한 규칙을 제안한다. 또한 차원·가중감쇠 등의 아티팩트를 통제하기 위해 same-dim PCA/가우시안 노이즈/제로 대체 ablation을 수행해, 손실이 LLM 특징의 정보성에 특이적으로 연결됨을 보인다.

- **Empirical Impact**: 실험적으로 Planetoid public split(작은 라벨 수)에서 결합 성능 저하가 가장 크게 나타나며, PubMed의 -17 pp 효과는 학습 데이터 수가 늘면 빠르게 완화된다. 아울러 여러 PubMed 구성에서 |Δconcat|이 (sqrt(d_l/n))^1.31 형태의 파워 법칙(r^2=0.97)을 따르는 스케일링도 제시해, 문제를 데이터 특이 현상보다 “표본 복잡도(sample complexity) 기반 현상”으로 해석하게 한다. 따라서 TAPE/GLEM류의 end-to-end 파이프라인 이득은 joint training·게이팅 같은 결합 메커니즘이 만들어내는 결과이며, 단순 concatenation만으로는 재현되지 않을 수 있다는 실무적 경고가 된다.



### Non-Autoregressive Minimum Bayes' Risk Decoding for Fast Speech Recognition (https://arxiv.org/abs/2606.17537)
Comments:
          Accepted at Interspeech2026

- **Prior Approaches**: 기존 비자동회귀(non-autoregressive, NAR) 디코딩은 토큰을 병렬로 생성해 음성인식 속도를 높이지만, 이전에 생성된 토큰을 조건으로 삼아 불확실성을 해소하지 못해 인식 성능이 떨어지는 한계가 있었다. 반면 자동회귀(autoregressive, AR) 디코딩은 좌→우로 순차 생성하며 불확실성에 대응할 수 있지만, 생성 과정이 느리다. 기존 NAR 디코딩은 이 성능 격차를 줄이는 데 어려움을 겪어 왔다.

- **Core Contribution**: 이 논문은 NAR 디코딩에 minimum Bayes' risk(MBR) 기반 의사결정을 결합한 NAR-MBR 디코딩 프레임워크를 제안한다. 핵심은 모델이 낸 확률을 그대로 최대화하는 게 아니라, NAR 모델의 출력 확률에서 샘플을 뽑아 기대 효용(expected utility)을 최대화하는 방식으로 “불확실성”을 처리한다는 점이다. 또한 NAR 모델의 성질을 활용해 여러 후보를 빠르게 생성한 뒤 MBR로 선택한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 샘플 기반 MBR을 NAR의 효율성과 맞물리게 구현해, 샘플 수를 늘리더라도 연산 부담이 급격히 커지지 않도록 만드는 것이다. 논문은 NAR 모델의 한 번의 forward computation으로 여러 샘플을 효율적으로 얻는 구조를 이용해, 기대 효용 계산에 필요한 샘플링을 실용적인 비용으로 수행한다. 그 결과 AR에 비슷한 수준의 의사결정 품질을 기대 효용 관점에서 끌어올린다.

- **Empirical Impact**: LibriSpeech, Switchboard, AMI, web presentation corpus에서 실험한 결과, 제안한 NAR-MBR 디코딩은 기존의 NAR 디코딩 대비 성능을 개선했다. 동시에 AR 디코딩보다도 더 빠른 처리 속도를 보였으며, 속도-정확도 트레이드오프를 동시에 완화하는 효과가 확인됐다. 음성인식에서 NAR의 대표적 약점(불확실성 미해소)을 MBR로 보완할 수 있음을 실증했다.



### PARSE: Provenance-Aware Retrieval Sanitization for Professional Domain LLM Agents (https://arxiv.org/abs/2606.17467)
Comments:
          7 pages, 3 figures, 2 tables. Under submission at EMNLP 2026 Industry Track

- **Prior Approaches**: 기존 prompt injection 방어는 spotlighting, sandwich prompting, paraphrasing, Llama Guard 같은 safety classifier에 기대는 경우가 많았지만, 대다수 성능은 짧고 단순한 synthetic benchmark에 묶여 있었습니다. 특히 domain-camouflaged injection은 전문 용어로 악성 지시를 위장해 classifier 기반 탐지를 0에 가깝게 만들 수 있어(예: Llama Guard 3) 기존 방어의 한계가 반복해서 드러났습니다.

- **Core Contribution**: 이 논문은 synthetic 결과가 실제 기업 문서에 일반화되지 않는 “일반화 실패”를 먼저 실측으로 보여줍니다. 이어서 PARSE(Provenance-Aware Retrieval Sanitization)라는 도메인 인지(inference-time) 사실 보존형 정화 파이프라인을 제안해, 문장을 injection 가능성에 따라 분류하고 구조화된 facts를 뽑은 뒤 consistency-checking loop로 사실 손실을 검증합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 전문 용어/권위 표현까지 함께 갈아버리는 indiscriminate paraphrasing 문제와 (2) 사실을 유지하면서도 injection만 제거하는 “제거-보존 균형”입니다. PARSE는 directiveness gate로 문서 위험도를 먼저 라우팅해 비용을 고위험 문서에 집중하고, domain-specific allowlist로 합법적 authority language의 오탐을 줄인 뒤, 추출한 facts가 출력에 남아 있는지 닫힌 루프(consistency loop)로 확인합니다.

- **Empirical Impact**: 122개 태스크(재무·법률·의학·과학·DevOps)의 실제 문서 기반 벤치마크에서 paraphrasing은 공격 성공률(ASR) 개선이 통계적으로 유의하지 않았고(p=0.500) 유틸리티는 91.8%→82.8%로 떨어졌습니다. 반면 PARSE는 ASR 15.6%로 baseline 25.4% 대비 38% 감소를 보이며(p=0.014, 충분한 power) 유틸리티는 86.9%로 기준선에 근접해 “유의미한 보안 이득 + 실사용 품질”을 동시에 달성했습니다. 저자들은 실무자가 synthetic 프록시가 아니라 도메인 매칭된 real document로 방어책을 검증해야 한다고 강조합니다.



### Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems (https://arxiv.org/abs/2606.17443)
Comments:
          16 pages, 4 figures, 11 tables

- **Prior Approaches**: 기존 연구들은 LLM 추천에서 특정 브랜드가 반복적으로 우세해지는 브랜드 편향을 주로 관찰하거나(예: name/description 조작, prompt injection 등) 일부 요인을 분해해 왔습니다. 다만 그 편향이 언제 완전히 작동하고, 어떤 조건에서 쉽게 무너지는지(경쟁 동학의 ‘임계점’)는 충분히 정리되지 않았습니다. 또 마케팅 문구가 그 편향을 ‘활용’할 수 있는지, 그리고 여러 브랜드가 동시에 최적화할 때 시장이 어떻게 변하는지는 미지였습니다.

- **Core Contribution**: 이 논문은 스킨케어(경험재)와 검색재(USB 케이블·AA 배터리)에서 LLM 추천 경쟁을 체계적으로 측정해 ‘Conditional Monopoly’ 패턴을 제시합니다. 동일 스펙이면 유명 브랜드가 사실상 100% 추천을 독점하지만, 경쟁 제품이 아주 약간만 더 나은 품질 신호를 가지면 그 지배가 급격히 사라집니다. 더 나아가 authority-style 마케팅(허위 임상 근거 같은 ‘권위 신호’)이 그 독점을 깨는 방법이며, multi-brand GEO 상황에서는 상호 최적화가 게임이론적 딜레마로 이어짐을 보여줍니다.

- **Technical Challenges**: 첫째, 브랜드 우위가 단순한 name 인식인지, 실제로 ‘품질 신호 부재’에서만 나타나는 조건부 효과인지 분리해야 했습니다. 이를 위해 실브랜드 1개 vs 검증된 가상 브랜드 9개를 만들고, 평가지표(I AI, BOR)와 memory hallucination probe로 ‘프롬프트에 없는 특징을 끌어오는지’까지 점검했습니다. 둘째, 마케팅 문구 효과를 정량 비교하려고 Bias Surplus Value(BSV)로 권위·사회적 증거 등의 언어 신호를 ‘품질 개선에 준하는 등가치(별점 +0.17 등)’로 환산해 해석 가능하게 했습니다.

- **Empirical Impact**: 실험 결과, Conditional Monopoly는 유명 브랜드 기준 IAI=10.0 수준으로 나타나지만 품질 신호가 임계 수준을 넘으면(예: 별점 +0.075) 보상이 급변합니다. authority-style 언어는 monopoly를 BSV 관점에서 별점 약 +0.17에 해당하는 효과로 깨며, 다수 모델(GPT-4o-mini, Claude Sonnet, Gemini 3 Flash)마다 반응 양상도 달랐습니다. 마지막으로 모든 브랜드가 GEO를 채택하면 개인 이득이 붕괴(+0.802→+0.007)하고 미참여 브랜드는 추천을 거의 받지 못하는 ‘죄수의 딜레마’ 형태의 경쟁 균형이 관측되어 GEO를 보안 이슈뿐 아니라 신종 마케팅 실천으로 다뤄야 함을 시사합니다.



### Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models (https://arxiv.org/abs/2606.17389)
Comments:
          16 pages. Accepted to the ICLR 2026 Workshop on Multimodal Intelligence. Code: this https URL

- **Prior Approaches**: 기존 연구는 VLM의 신뢰도를 “Attention-Confidence Assumption” 관점에서 해석해 왔습니다. 즉, 시각 인코더가 관련 영역에 촘촘히 주목하면 모델이 정답을 낼 “근거(grounding)”가 생긴다고 가정했죠. 하지만 이런 해석은 attention이 출력의 결정요인을 충실히 설명하는지에 대한 논쟁과 함께, 출력 기반 환각 평가(benchmark) 중심으로 보정이 부족하다는 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 VLM Reliability Probe(VRP)로 여러 VLM 계열(LLaVA-1.5, PaliGemma, Qwen2-VL)을 가로지르는 신뢰도 시그널을 체계적으로 비교합니다. 특히 시각 “structural” 지표(클러스터 수 C_k, 공간 엔트로피 H_s)와 생성 동역학 기반 지표(예: self-consistency)를 함께 상관/예측하며, reliability가 단순 attention 구조가 아니라 “생성 과정의 내부 상태 분포”에 가깝다는 결론을 제시합니다. 또한 LLaVA에서는 Early Lock(또는 Symbolic Detachment), PaliGemma/Qwen2-VL에서는 더 분산된 신뢰도 경로가 나타난다고 분석합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “attention heatmap이 신뢰도를 알려주는가?”를 상관 수준을 넘어 인과·기계적으로 분리해 증명하는 것입니다. 논문은 forward hook으로 cross-attention 및 hidden state를 수집하고, Logit Lens(레이어별 correct vs incorrect 토큰 logit 차)와 hidden-state probe, 그리고 대규모 causal ablation(특정 레이어의 예측 뉴런/부분을 파괴)까지 수행해 신뢰도 경로가 어디에 있는지 추적합니다. 그 결과 시각 attention은 구조적으로는 중요하지만 통계적으로는 정확도와 거의 무관하고, self-consistency와 hidden-state probe가 훨씬 강한 예측력을 보인다고 정리합니다.

- **Empirical Impact**: 실험에서 C_k와 H_s는 정답 여부와의 상관이 사실상 0에 가깝게 나와 “Cluster Failure”를 뒷받침합니다(예: R≈0.001, R≈-0.012). 반대로 self-consistency는 truth 예측에서 R=0.429로 시각 지표 대비 우수하며, hidden-state probe는 강한 설정에서 AUROC>0.95 같은 높은 분별력을 보입니다. 더 나아가 LLaVA는 late-stage 병목을 파괴하면 취약해지지만, PaliGemma와 Qwen2-VL은 예측에 기여하는 부분을 ~50% 이상 파괴해도 견고함을 보여 신뢰도 신호 설계가 아키텍처에 강하게 의존한다는 메시지를 강화합니다.



### SpeechDx: A Multi-Task Benchmark for Clinical Speech AI (https://arxiv.org/abs/2606.17339)
- **Prior Approaches**: 기존 임상 음성 AI는 질환별로 개별 데이터셋에서 학습·평가되는 경우가 많아, 조건 간 비교와 일반화(다른 데이터로의 전이) 측정이 어려웠습니다. 또한 모델이 녹음 환경·인구 구성·획득 장비 같은 교란 요인에 대한 스퍼리어스 상관관계를 학습해 분포 변화에서 성능이 무너진다는 문제가 반복적으로 지적돼 왔습니다. 표준화된 벤치마크가 부족해 “좋아진 것”이 임상 신호 때문인지 데이터 아티팩트 때문인지 가르기 어려웠습니다.

- **Core Contribution**: SpeechDx는 12개 공개 음성 데이터셋, 9개 건강/정서 조건, 총 27개 태스크를 아우르는 대규모 임상 음성 AI 벤치마크입니다. 핵심은 발화 생성 과정(개념화–정형화–조음)에서 질환이 주로 끼치는 단계별로 태스크를 구조화해, 서로 다른 질환을 공유 임상 메커니즘 관점에서 비교 가능하게 만든 점입니다. 또한 제한된 라벨 데이터 상황과 데이터셋 간 동일 질환 평가, zero-shot cross-condition transfer로 데이터 아티팩트를 배제하려는 평가 설계를 제공합니다.

- **Technical Challenges**: 임상 음성에서는 녹음 조건과 라벨 품질이 크게 달라져, 표현이 실제 임상 구조를 담는지 검증하기가 어렵습니다. 논문은 12개 SOTA 오디오 인코더를 하나의 프로토콜로 선형 프로빙해 태스크 전반을 일관되게 비교하고, 태스크 단계별 어려움 분해와 zero-shot 전이를 함께 수행해 일반화 실패 지점을 드러냅니다. 또한 입력 길이 차이를 chunk+mean pooling으로 처리하고, 학습/평가 분할을 speaker-disjoint 중심으로 구성해 누수를 줄이도록 했습니다.

- **Empirical Impact**: 결과적으로 whisper, Qwen3-TTS-Tokenizer, WavLM 같은 대규모 음성 모델이 전반 성능 기준에서는 강했지만, 어떤 인코더도 임상 스피치 전반에서 “신뢰할 만한 일반화”를 일관되게 보이지 못했습니다. 질환·태스크에 따라 승자가 달라져, 특정 단계(예: 감정의 개념화 영역)에서는 잘 맞지만 다른 범주로 갈수록 성능이 쉽게 꺾였습니다. 특히 현장에서 기대하기 쉬운 호흡/발성(phonatory/respiratory) 계열은 cross-dataset 일반화가 가장 취약했으며, SpeechDx는 이러한 진행 상황을 추적할 공용 평가 프레임워크를 제시했다는 점에서 의미가 큽니다.



### Nothing from Something: Can a Language Model Discover 0? (https://arxiv.org/abs/2606.17289)
- **Prior Approaches**: 기존 연구는 수학·증명에서 large language model이 상위 벤치마크 성능을 내는 모습을 보여주며, 주로 사전학습 데이터/프로세스 슈퍼비전/강화학습·합성 데이터의 스케일링을 강조해 왔습니다. 다만 많은 결과가 학습 중 이미 비슷한 형식의 구조를 많이 봤을 가능성이 커서, test time에 ‘완전히 새로운’ 수 구조로 점프하는 out of distribution generalization 능력을 직접 입증하진 못했다는 한계가 지적됩니다. 또한 compositional generalization 연구가 있지만, 본 논문은 개념(예: zero) 수준의 불연속을 넘는 leap을 별도로 측정하려는 방향입니다.

- **Core Contribution**: 본 논문은 “양의 한 자리 산술(0 제외)로만 학습한 모델이 zero 개념을 test time에 자율적으로 발견/일반화할 수 있는가”를 가장 단순한 사례로 정식화합니다. 그 결과 GPT-2 크기 언어모델은 언어 pretraining 여부와 무관하게 zero로의 일반화를 zero-shot에서는 실패하며, 반면 zero가 포함된 소량 예시를 학습에 추가하면 few-shot 규모에서 성능이 크게 개선됨을 보여줍니다. 특히 언어 pretraining이 필요한 few-shot 수를 약 50% 줄여, 언어 능력이 수학적 발견을 scaffold할 수 있음을 시사합니다.

- **Technical Challenges**: 핵심 실험적 어려움은 “모델이 학습 단계에서 산술 기호·문맥을 우연히 접했는지”를 통제하는 것입니다. 이를 위해 GPT-2 스타일 모델을 대상으로 자체 정제한 OpenWebText 변형을 사용해, 사전학습 코퍼스에는 숫자·수학 기호가 사실상 없도록 만들고, 토크나이저/토큰화 차이도 수동 토크나이징 설계로 격리했습니다. 또한 zero가 등장하는 위치(답의 ones place 등)에서만 토큰을 제공하도록 구성해, zero-shot 실패가 데이터 오염이 아닌 ‘개념 비약’의 문제임을 분리해 보여주었습니다.

- **Empirical Impact**: 실험은 세 층위로 명확한 신호를 줍니다: (1) zero-shot 일반화는 모든 테스트 모델에서 관측되지 않고, (2) zero를 포함한 tens~hundreds 예시(few-shot)를 학습에 섞으면 언어 pretraining 유무에 관계없이 성능이 상승하며, (3) 언어 pretraining은 같은 정확도에 필요한 데이터 양을 평균 48.5% 줄였습니다(p=1.7×10−4). 마지막으로 zero가 ‘특별한가’를 다른 숫자 홀드아웃으로 점검했을 때, 중간 숫자는 더 쉽고 carry와 가까운 숫자(0 및 9 등)가 더 어렵다는 패턴이 나타났습니다. 이 결과는 수학 벤치마크 성능을 넘어, test time에 개념을 확장하는 메커니즘이 데이터·학습량·언어 scaffold에 얼마나 의존하는지에 대한 실증적 근거를 제공합니다.



### Rethinking Groups in Critic-Free RLVR (https://arxiv.org/abs/2606.17250)
- **Prior Approaches**: critic-free RL은 보통 한 프롬프트에서 여러 rollouts를 뽑아(그룹) 기준값(advantage용 baseline)을 추정한 뒤 학습 안정성을 노립니다. GRPO/RLOO/ReMax 계열처럼 그룹 기반 advantage는 분산을 줄이지만, 추가 rollout 비용과 그룹 동기화/구조화된 rollout 적용의 제약이 생깁니다. 또 단일 rollout로 줄이면(그룹 제거) reasoning에서 학습이 쉽게 붕괴(collapse)하는 문제가 반복됩니다.

- **Core Contribution**: 이 논문은 “그룹의 역할은 baseline 추정이 아니라, 잘못된(negative) 샘플이 유용한 supporting token까지 과도하게 벌점 주지 않도록 상쇄를 만드는 것”이라고 재해석합니다. 이를 바탕으로 positive/negative가 공유하는 토큰이 있을 때는 negative 업데이트가 일부 상쇄된다는 메커니즘을 제시합니다. 그 결과, 단일 rollout에서도 안정적인 critic-free 학습을 가능하게 하는 Negative Token Filtering(NTF)를 제안합니다.

- **Technical Challenges**: 핵심 난제는 negative trajectory에서 어떤 토큰이 ‘실패의 원인’인지 구분해 잘못된 벌점을 막는 것입니다. 저자들은 supporting token이 되는 고확률(high-probability) 토큰을 negative에서 그대로 패널티하면 catastrophic forgetting에 가까운 붕괴로 이어진다고 분석하고, 통계적 토큰 중복과 gradient를 가중치 행렬의 top singular subspace에 투영한 에너지 분포로 이를 뒷받침합니다. 해결책으로 NTF는 negative 손실에서 현재 정책이 고확률로 예측하는 토큰(상위 확률 구간)을 마스킹하고, 남은 저확률 토큰에 negative 업데이트를 집중시켜 단일 rollout 학습을 안정화합니다.

- **Empirical Impact**: 수학 reasoning(DAPO-Math 기반)과 에이전틱 태스크(ALFWorld/WebShop)에서 NTF를 적용한 batch-level advantage 방식(RF++/Contrastive-REINFORCE)은 단일 rollout에서도 붕괴 없이 학습되며, 그룹 기반 방법과 비교해 경쟁력 있거나 더 나은 성능을 보입니다. 특히 에이전틱 환경에서는 group cancellation이 덜 신뢰할 수 있는 복잡한 탐색 공간 특성을 고려할 때, NTF 기반 단일 rollout이 GRPO 대비 성공률을 유의미하게 끌어올렸습니다. 즉, 그룹 기반 advantage가 ‘필수 안정장치’가 아니라, negative-token 업데이트를 선택적으로 제어하면 group-free 단일 rollout로도 안정적이고 효과적인 RL post-training이 가능하다는 점을 실증했습니다.



### Rift: A Conflict Signature for Deception in Language Models (https://arxiv.org/abs/2606.17229)
Comments:
          13 pages, 4 figures. Code and experiment logs: this https URL

- **Prior Approaches**: ELK(Eliciting Latent Knowledge)는 출력이 그럴듯하더라도 모델이 아는 진실을 추출하려는 문제로, 특히 “아는 것을 숨기고 거짓을 말하는 경우”가 가장 어렵다고 본다. 기존 방법들은 논리 일관성 제약으로 진실-유사 방향을 찾거나(unsupervised), 또는 선형 표현/steering으로 정직성 같은 개념을 다루지만, “거짓은 내면서(정답처럼 보이게) 내부적으로 어떤 상태가 다른지”를 라벨 없이 분리해내는 데는 한계가 있었다.

- **Core Contribution**: 이 논문은 “거짓이지만 단순히 틀린 것”과 “진실을 알면서도 의도적으로 거짓을 말하는 것”을 내부 상태 차이로 구분할 수 있는지 묻는다. 핵심은 wrongness(틀림)와 deception(지식 충돌)을 분리하기 위해 sleeper agent(정답을 아는 상태에서 트리거로 거짓)와 naive liar(동일한 오답을 내도록 fine-tuning만 한 상태)를 대조하는 통제 설계를 제시하는 것이다. 그 결과, 거짓은 출력이 같아도 내부에 “conflict signature”를 남긴다고 주장한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 거짓과 틀림이 행동(출력) 수준에서 동시에 섞여 있어, 단순 분류로는 구분이 불가능하다는 점이다. 논문은 hidden states의 residual rank(압축 불가능성/경쟁 구조)를 층별로 평균내고, 두 응답(정직 vs 거짓) 중 더 높은 conflict 점수를 가진 쪽을 “거짓”으로 라벨 없이 선택하는 per-example 점수 체계를 만든다. 또한 residual rank가 길이/불확실성 같은 다른 요인과 섞이지 않도록 길이-일치 컨트롤, 전략적 자작 deception, concealment(숨기기) 지시까지 포함해 강건성을 확인한다.

- **Empirical Impact**: 실험 결과, deception forward pass는 naive liar 대비 residual rank가 2.1–2.3배 더 높게 나타나며(동일한 오답 출력 조건), GPT-2 계열에서 라벨 없이도 “어느 답이 거짓인지”를 100%로 식별한다. Qwen2.5 및 Phi-3-mini에서도 roleplay 기반 instructed deception이 18/18, 40/40, 34/34로 각 사실에서 거짓을 분리했고, Phi-3에서는 honest 답과 hallucination까지 AUC 1.0 수준으로 완벽 분리했다. 더 나아가 영어로 학습한 deception 검출기가 다른 모델 계열/포맷/언어로도 상당히 전이되며(특히 언어 축에서 교차언어 AUC 1.0), 다만 저자들은 동일 방향을 넣어 “선형으로 거짓을 생성”시키는 causal steering은 성공하지 못해(읽기는 되지만 쓰기는 어려움) 안전 관점에서 탐지-제어의 비대칭성도 함께 시사한다고 정리한다.



### Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation (https://arxiv.org/abs/2606.17188)
- **Prior Approaches**: 기존 비전-언어 모델(VLM) 멀티링구얼 평가는 언어와 문자(orthography)를 사실상 1:1로 가정하는 경우가 많다. 그래서 Gurmukhi, Shahmukhi처럼 같은 언어의 서로 다른 문자 스크립트를 동일 난이도 조건에서 비교하지 못해, 성능 차이를 ‘언어 역량’이 아니라 ‘문자 패턴’ 문제로 분해하기 어렵다.
또한 텍스트 기반 연구에서 스크립트에 따른 성능 저하가 보고돼 왔지만, 시각적 입력이 이런 스크립트 편차를 얼마나 메우는지에 대한 체계적 검증은 부족했다.

- **Core Contribution**: 이 논문은 PuMVR(Punjabi Multimodal Visual Reasoning)이라는 1,000개 평행 이미지-텍스트 벤치마크를 제안하며, Punjabi의 세 스크립트(Gurmukhi, Shahmukhi, Roman)를 동일 의미로 정렬해 ‘스크립트 변인’을 독립적으로 평가한다. 여기에 Script Consistency Rate(SCR)을 도입해, 한 스크립트에서 잘하는 모델이 다른 스크립트에서도 동일하게 맞히는지를 강하게 요구하는 ‘스크립트 비의존성’ 지표로 재정의한다.
즉, 멀티링구얼을 언어 수 확장(coverage)이 아니라 스크립트 강건성(orthographic robustness)으로 평가해야 한다는 프레임을 제시한다.

- **Technical Challenges**: 핵심 난제는 ‘문자만 바뀌고 의미/이미지 난이도는 동일’하게 맞춘 평행 데이터를 만드는 일과, 모델 출력이 스크립트 표기 방식 차이로 생기는지(포맷 문제) 의미 이해 실패인지 분리하는 것이다. 논문은 이미지-질문-보기 4지선다를 스크립트별로 제공하되 정답은 의미적으로 동일하게 맞춰, 스크립트 간 성능 차이를 orthographic comprehension 실패로 해석할 수 있게 설계했다.
또한 텍스트-only 대비 멀티모달이 스크립트 갭을 ‘보완(compensate)’하는지 ‘누적(additive)’ 효과인지 분해하고, cross-script in-context learning의 전이 안정성까지 TE(Transfer Efficiency)로 계측했으며, 스크립트 쌍별 McNemar 검정으로 편차의 통계적 견고함을 확인했다.

- **Empirical Impact**: 10개 SOTA VLM을 PuMVR에 적용한 결과, 스크립트 변화만으로 정확도가 최대 16%까지 흔들리는 Script Gap이 관찰됐다. 시각 입력은 절대 성능을 전반적으로 올리지만 SCR 갭을 닫지 못해, 멀티모달이 스크립트 편차를 자동으로 해결하지는 못한다는 점이 드러났다.
더 나아가 in-context exemplars 전이가 스크립트에 매우 취약하며, 일부 모델은 특정 ‘앵커 스크립트’에 강하게 고정된 듯한 비대칭 전이(TE < 67%)를 보였고, SCR이 24.8%까지 내려가는 경우도 확인돼 현재 ‘multilingual’ 주장의 실질적 범위가 제한적임을 시사한다.



### The Critical Role of Model Selection in Causal Inference: A Comparative Analysis of Classification Models within the InferBERT Framework for Pharmacovigilanc (https://arxiv.org/abs/2606.17113)
Comments:
          10 pages, 5 figures

- **Prior Approaches**: 약물 이상반응(ADE)을 발견할 때는 FAERS 같은 관측 데이터에서 진짜 인과 신호와 잡음(편향·동반질환·공동 처방 등으로 생긴 상관)을 분리하는 게 핵심이다. InferBERT는 transformer 분류기 확률을 바탕으로 Do-calculus로 개입을 시뮬레이션해 인과 요인을 뽑지만, 분류기 선택이 후속 인과 추론 결과를 얼마나 흔드는지에 대한 체계적 평가는 부족했다. 또한 더 단순한 XGBoost로 충분한지, 도메인 사전학습과 LLM 스케일이 causal signal 탐지에 도움 되는지, 그리고 사후 calibration이 신호 추출에 미치는 효과도 불명확했다.

- **Core Contribution**: 이 논문은 InferBERT의 ‘분류기 컴포넌트’가 인과 탐지 성능을 좌우한다는 가설을, 분류 모델 4종(XGBoost, ALBERT, BioBERT, Med-LLaMA)과 calibration 유무까지 포함해 실증적으로 검증한다. AILF(진통제-급성 간부전)와 TRAM(트라마돌-사망) 두 벤치마크에서 accuracy·ECE·PRR/ROR/EBGM 대비 causal term 일치도(Jaccard)를 동시에 측정한다. 결론적으로 도메인 특화 사전학습을 탑재한 BioBERT가 예측 정확도와 전통적 약물감시 신호와의 정합성에서 가장 일관되게 우수함을 보인다.

- **Technical Challenges**: 핵심 기술 과제는 Do-calculus가 사용하는 확률 추정의 품질을 분류기가 실제로 제공하느냐이며, 특히 확률의 신뢰도(ECE) 개선이 곧 인과 발견 개선으로 이어지는지 불확실하다는 점이다. 연구팀은 5-fold 교차검증을 20회 반복하고, isotonic regression으로 사후 calibration을 적용한 뒤 accuracy-ECE-인과 term 일치도를 함께 비교하며 모델·보정 간 상호작용을 통계적으로 검정(paired t-test)했다. 또한 LLM을 쓰더라도(예: Med-LLaMA, PEFT/4-bit/LoRA) 구조화된 FAERS 입력을 템플릿 문장으로 바꾸는 표현 방식이 성능에 불리할 수 있음을 관찰했다.

- **Empirical Impact**: 실험 결과 BioBERT는 두 데이터셋에서 accuracy 1위를 차지했을 뿐 아니라, PRR·EBGM과의 Jaccard concordance도 가장 강하게 나타나 predictive power가 causal signal utility로 연결됨을 지지했다. 반대로 Med-LLaMA는 크기와 parameter-efficient fine-tuning에도 불구하고 가장 약한 성능을 보여 ‘무조건 스케일’ 가정이 이 문제에서는 성립하지 않음을 보여준다. Calibration은 ECE를 전반적으로 낮췄지만 accuracy 및 인과 추론 일치도에는 모델별로 개선/악화가 엇갈려, 향후 InferBERT류 파이프라인에서는 calibration을 필수로 강제하기보다 downstream causal metric을 기준으로 선택·검증해야 한다는 실무적 지침을 제공한다.



### Securing Multi-Agent GIS Systems: Risk Evaluation and Prompt Hardening Optimization (https://arxiv.org/abs/2606.17092)
Comments:
          Kyle Gao and Pranavi Kotta contributed equally to this work

- **Prior Approaches**: 기존 연구는 ReAct, AutoGen처럼 에이전트 구조를 일반화해 도구 사용과 멀티 에이전트 협업을 가능하게 했지만, 지리정보(GIS) 도메인에서는 prompt injection, data leakage, unsafe tool invocation 같은 보안 공격 면이 더 복잡해진다. 안전 분야에서는 guardrails와 LLM-as-judge, 정적 벤치마크 기반 평가가 많았으나, 실제 배포 맥락의 multi-turn 공격과 에이전트 오케스트레이션(라우팅/상태전이) 취약점을 함께 다루기 어렵다는 한계가 있었다. 또한 지리공간 워크플로우 자동화는 점점 진화했지만, 상용 수준 시스템 프롬프트 레벨에서의 체계적 보안 강화/검증은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 상용 multi-agent GIS 시스템을 대상으로, 위험 식별-평가-완화까지 이어지는 보안 프레임워크를 제안한다. LangGraph 기반 state-machine 오케스트레이션으로 에이전트 행동을 재사용 가능한 모듈로 추상화해 감사를 가능하게 하고, DSPy 기반 prompt 최적화로 구조화된 시스템 프롬프트 시그니처에 adversarial demonstration을 주입해 안전성을 높인다. 아울러 PyRIT 기반 adaptive red-teaming과 deterministic binary judge를 결합해 다턴 공격에서의 방어 성능을 재현 가능하게 측정한다.

- **Technical Challenges**: 핵심 난제는 multi-agent 협업과 tool invocation이 만드는 비가시적 공격 표면을, 시스템 프롬프트 레벨에서 어떻게 통제·평가할지였다. 논문은 LangGraph의 persistent state(대화/중간출력/API/턴 로그)와 조건부 라우팅으로 위협 경로를 명시화하고, PyRIT로 이전 에이전트 응답에 조건부로 적응하는 multi-turn 공격을 생성한 뒤 deterministic judge로 성공 여부와 근거를 분리해 산정했다. 이어 DSPy는 prompts를 structured signature로 보고 BootstrapFewShot류 컴파일로 방어 행동을 학습시키되, Recommender처럼 Chain-of-Thought가 없는 서브에이전트의 경우 라우팅 구조적 보강이 필요하다는 점까지 실험으로 드러낸다.

- **Empirical Impact**: 상용 GIS 파트너 시스템에서 8개 adversarial persona, 40회 독립 trial, 라운드별 base vs DSPy 최적화 비교로 강건성 변화를 정량화했다. 직접적인 credential extraction·system prompt injection(SEC 계열)은 이미 성공률이 0%로 “천장 효과”가 나타나며 DSPy의 추가 이득이 제한됨을 보여준다. 반면 TOP(범위 이탈/톤 드리프트/경쟁사 조작)와 REC(포맷 무결성 등)에서는 attacker success가 각각 29pp(84%→55%), 포맷/서식 공격에서 13pp 개선(33%→20%)되는 등 multi-turn 거부 지속성과 의도 분류 정밀도가 향상됐다. 특히 prompt ablation 후에도 DSPy가 성능 저하를 상당 부분 복구해 “안전 레이어” 역할을 한다는 점이, 그리고 Recommender의 구조적 한계가 남는다는 점이 함께 확인되며 layered security 관점의 실무적 시사점을 준다.



### Correct When Paired, Wrong When Split: Decoupling and Editing Modality-Specific Neurons in MLLMs (https://arxiv.org/abs/2606.17057)
Comments:
          18 pages, 11 figures

- **Prior Approaches**: 기존 Knowledge Editing은 LLM 중심 편집 패러다임을 MLLM에 그대로 옮겨와, 멀티모달(텍스트-이미지) 쿼리에서 정답이 맞는지를 주로 최적화해왔다. 하지만 멀티모달 입력에서 성공한 편집이 텍스트-only 같은 단일 모달 트리거로도 일관되게 전파된다는 보장은 부족했다. 최근 일부가 멀티모달 활성 패턴을 분석·편집하려 했지만, 모달 간 지식 공유/전달이 자연스럽다고 암묵적으로 가정하는 경우가 많다.

- **Core Contribution**: 이 논문은 MLLM에서 발생하는 editing decoupling failure를 정의하고, 멀티모달 쿼리에서는 편집이 맞더라도 단일 모달로 분리하면 구(舊) 지식으로 되돌아갈 수 있음을 체계적으로 보여준다. 원인은 ‘엔터티 지식이 단일 표현으로 저장되지 않고, 모달별로 분리된 뉴런 경로에 흩어진다’는 관찰에 있다. 이를 해결하기 위해 DECODE는 모달별 critical-neuron 그룹을 명시적으로 분리·국소화하고, 두 스트림(two-stream)으로 동기화된 편집을 수행한다.

- **Technical Challenges**: 핵심 기술적 난제는 편집해야 할 엔터티 관련 뉴런이 모달 트리거에 따라 서로 다른 회로에 매핑된다는 점이며, 그래서 멀티모달에서 조정한 업데이트가 단일 모달 회로로 전파되지 않는다는 것이다. 저자들은 모달별 입력 변형에서 각 뉴런의 contribution score를 계산해 텍스트-의존/비전-의존/멀티모달-의존 뉴런 집합을 분리한 뒤, FFN의 해당 뉴런 행에 learnable offset을 주입하는 방식으로 국소 편집을 구현한다. 또한 편집 효능과 collateral damage(다른 지식 손상)를 균형 있게 다루기 위해 타깃 손실과 locality를 위한 KL-divergence 기반 제약을 함께 최적화한다.

- **Empirical Impact**: 여러 MLLM(InstructBLIP, LLaVA, Qwen-VL)에 대해 DECODE는 멀티모달 입력에서는 물론 텍스트-only/시각 참조형 같은 decoupled unimodal 세팅에서도 편집 일관성을 크게 개선한다. 특히 FiNE 같은 뉴런 레벨 접근도 멀티모달-성공 후 텍스트-only에서 급락하는 decoupling failure가 나타나는 반면, DECODE는 이를 일관되게 완화한다. 나아가 cross-modal synchronization과 locality(불필요한 지식 훼손 최소화) 측면에서 우수한 성능을 보이며, closed-form이 아닌 내부 활성 기반 편집의 중요성과 모달별 회로 분리 고려 필요성을 실증적으로 강화한다.



New uploads on arXiv(cs.IR)

### IUU+DB: Tracking Illegal, Unreported, and Unregulated Fishing, Seafood Fraud, and Labor Abuse through LLM-driven Information Extraction (https://arxiv.org/abs/2606.18181)
- **Prior Approaches**: IUU(Ilegal, unreported and unregulated fishing) 중심의 기존 추적은 특정 행동유형 또는 특정 지역/종에 치우쳐 전역적 정량화가 어렵습니다. 또 뉴스·정부·학술·NGO 자료가 단편적으로 흩어져 있어, 사건의 빈도·지리·종·행위자·범죄 유형 패턴을 한 프레임에서 비교하기가 힘듭니다.

- **Core Contribution**: 이 논문은 IUU를 넘어 불법 양식, 라벨 사기, 노동 학대, 무역 제재 회피 등 연관 공급망 범죄까지 포괄하는 IUU+를 제안하고, 이를 구조화해 분석할 IUU+DB를 구축합니다. IUU+DB는 LLM 기반 정보추출로 이질적 문서에서 사건과 핵심 데이터 요소(KDE)를 자동으로 분류·추출·정리하는 end-to-end 파이프라인입니다.

- **Technical Challenges**: 핵심 기술 과제는 문서가 길고 표현이 제각각이며 IUU+ 유형 간 뉘앙스 구분이 어려워 추출 신뢰성과 재현성을 확보하기 어렵다는 점입니다. 논문은 KDE 그룹 존재 여부를 먼저 판단해 LLM의 컨텍스트 부담을 줄이고, few-shot 기반 소스 분류, 중복 제거(deduplication)와 trend 묶기, 그리고 DSPy/MIPROv2로 prompt 및 필드 설명을 반복 최적화해 잡음(hallucination)과 불일치(mismatch)를 줄이는 방향으로 해결합니다.

- **Empirical Impact**: IUU+DB는 143개 국가, 11년치, 2,472개 소스에서 8,435건의 사건을 수집·구성했으며 140여 국가에 걸친 글로벌 트렌드를 조직화할 수 있음을 보입니다. 정성·정량 평가에서 범위 분류와 KDE 추출의 precision/recall/F1이 전반적으로 유효하고, 기준선 모델 대비 IUU+ type과 sub-behavior에서 10~15% 수준의 개선이 관찰됩니다. 동시에 scope classifier의 precision이 낮아 잡음이 생기는 약점도 드러나, 향후 정책·집행용 위험평가의 정확도 향상 여지를 제시합니다.



### Non-negative Elastic Net Decoding for Information Retrieva (https://arxiv.org/abs/2606.17910)
Comments:
          19 pages, 4 figures

- **Prior Approaches**: 기존 dense retrieval은 쿼리-문서 임베딩을 내적(inner product)으로 독립 스코어링해 top-k를 뽑는 방식이어서, 코퍼스 전체의 문서 간 상관관계를 반영하지 못합니다. 그 결과 서로 비슷한(중복/유사) 문서가 함께 선택되어 비다양하고 정보가 겹치는 retrieved set이 생기기 쉽습니다. 또한 cross-encoder나 generative retrieval처럼 문서들을 함께 평가하는 방법은 품질은 높을 수 있지만 지연시간 제약 때문에 그대로 쓰기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 retrieval을 “문서들을 집합으로 함께 디코딩”하는 joint decoding 문제로 재정의합니다. 제안하는 Non-Negative elastic Net (NNN) decoding은 쿼리 임베딩을 코퍼스 문서 임베딩의 희소(sparse)한 비음수 선형결합으로 복원하도록 하여, 관련 문서들끼리는 남기고 중복 문서는 자연스럽게 억제하도록 설계됩니다. 더 나아가 NNN decoding의 이론적 표현력이 dense retrieval보다 크다는 점을 정식으로 증명하고, scoring 함수를 단순 교체하는 drop-in replacement 관점도 제시합니다.

- **Technical Challenges**: NNN decoding은 각 문서 포함 여부가 코퍼스 전체를 고려한 복원 오차에 의해 결정되므로, 실사용에서는 (λ1, λ2) 같은 하이퍼파라미터를 어떻게 정해야 하는지가 관건입니다. 이 논문은 FISTA 기반의 non-negative elastic net 최적화를 사용해 추론 비용을 O(dNT)로 유지하면서, unrolling된 solver를 역전파에 연결해 end-to-end fine-tuning도 가능하게 했습니다. 또한 이론은 쿼리마다 다른 (λ1, λ2) 존재성을 말하지만, 실험에서는 고정된 한 쌍을 validation으로 맞추는 더 단순한 배치 전략이 여전히 이득을 준다는 점을 보여줍니다.

- **Empirical Impact**: 실험에서는 먼저 frozen embeddings(내적 스코어링용으로 학습된 임베딩) 위에 NNN decoding만 적용해도 여러 벤치마크에서 성능이 일관되게 개선되는 것을 확인합니다. Tool retrieval과 multi-hop retrieval에서 특히 completeness 지표에서 최대 36% 향상이 보고되며, 이는 관련-비관련 문서 상관이 큰 near-duplicate 상황에서 이론 예측과 맞물려 더 큰 격차로 나타납니다. 나아가 unrolled FISTA를 통한 end-to-end 학습을 수행하면 모든 지표/벤치마크에서 dense retrieval을 상회하는 유의미한 성능 향상을 달성하며, dense 임베딩을 inner-product 스코어링을 넘어 활용하는 새로운 패러다임을 제시합니다.



### Understanding and Debugging Failures in N-Gram-Based Generative Retrieva (https://arxiv.org/abs/2606.17721)
Comments:
          Work in progress

- **Prior Approaches**: 생성형 검색(Generative Retrieval, GR)은 기존의 dense bi-encoder를 대체하는 end-to-end 학습형 검색 패러다임으로, 언어모델이 docid(문서 식별자)를 직접 생성한다. 기존 연구들은 atomic ID, 또는 텍스트 기반 식별자(문서 제목/substring ngram 등)를 쓰며 성능을 보고했지만, GR 특유의 실패 원인은 체계적으로 정리되지 않았다. 특히 ngram 기반 방식은 분석 가능성이 높지만, 어떤 메커니즘이 왜 망가지는지 실증적·진단적 가시성이 부족했다.

- **Core Contribution**: 이 논문은 GR 실패를 Representation-Training-Inference-Response Generation의 4단계로 나눈 실패 모드 분류체계를 제안한다. 또한 ngram 기반 대표 모델인 SEAL과 MINDER를 대상으로 실패 양상을 실험적으로 파고들어, 모호한 docid와 낮은 identifier diversity, 특정 identifier 쏠림 같은 공통 패턴을 도출한다. 마지막으로 생성된 ngram과 최종 랭킹 기여를 시각적으로 추적하는 web 기반 진단 도구를 공개해, 연구자들이 오류 지점을 빠르게 감사(audit)할 수 있게 했다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) ngram이 문서에서 고유하게 식별되지 않아 모호해지는 문제, (2) 길이/토크나이징/제약 디코딩(FM-index 등) 때문에 올바른 identifier 확장이 막히는 문제, (3) 점수 계산이 일부 ngram에 과도하게 집중되거나(Score Concentration) 유용한 분산 근거를 합치지 못하는 문제다. 논문은 이 원인을 모델별로 해부하며, SEAL이 title/메타데이터에 강하게 의존하고 길이 bias·subword 제약·unigram 쏠림이 랭킹 저하와 연결됨을 보였다. MINDER는 pseudo-queries(PQs)를 추가해 문서 확장에 도움을 주며, PQ가 전체 ngram 비중은 낮아도 점수 질량의 큰 부분을 차지해 성능에 기여함을 보임으로써 일부 실패를 완화한다.

- **Empirical Impact**: Natural Questions(NQ)과 MS-MARCO에서 Hits@1/10을 중심으로 평가한 결과, SEAL은 메타데이터 노이즈에 취약하고 특정 ngram이 점수를 지배할수록 성공률이 하락하는 경향이 확인됐다. 특히 SEAL의 경우 생성된 identifier가 정답 answer-string과 일치하면 Hits@1이 크게 상승해, 모델이 검색 기반 정합보다 parametric post-rationalization 쪽으로 기울 가능성을 시사한다. 공개된 ngram 분석 도구는 색상 코딩으로 고유/부정/모호 토큰을 구분하고 설정 간 비교까지 제공해, GR의 재현·디버깅 문턱을 낮춘다는 점에서 현장 impact가 기대된다.



### Do Generative Recommenders Deepen the Information Cocoon? A Closed-Loop Simulation with LLM-powered User Simulators (https://arxiv.org/abs/2606.17707)
- **Prior Approaches**: 기존 추천 연구는 반복 피드백 루프가 사용자 선호를 강화하고 노출 다양성을 줄이며, 결과적으로 popularity bias와 정보 cocoons(정보 거품)을 키울 수 있다고 봐왔다. 그러나 최근 generative recommendation은 아이템을 score하는 방식이 아니라 토큰/코드 시퀀스를 생성해 노출을 만들기 때문에, 같은 현상이 동일하게 나타나는지(상속/완화/증폭)는 불명확했다. 기존 sequential 추천의 오프라인 평가는 모델이 만든 노출이 다시 학습 데이터로 되돌아오는 장기 동학을 충분히 재현하지 못한다는 한계도 있었다.

- **Core Contribution**: 이 논문은 generative recommendation이 정보 cocoon을 “더 깊게” 만드는지 확인하기 위해 RecLoop이라는 closed-loop 시뮬레이션 프레임워크를 제안한다. LLM-driven user agents가 추천 노출을 선택하고, 그 상호작용이 다음 라운드 학습에 다시 반영되도록 설계해 장기 피드백 효과를 관찰한다. 또한 exposure 관점의 기존 지표에 더해, 생성 과정에서 사용되는 SID(semantic ID) 코드 공간의 집중도를 측정하는 Code-Space Structural Cocoon이라는 모델 레벨 metric을 도입한다.

- **Technical Challenges**: 핵심 난제는 generative 추천의 “무엇이” cocoon을 만드는지 예측하기 어렵다는 점이다. generative recommender는 multi-layer 코드 생성으로 노출이 계층적으로 결정되며, 초기 코드 선택이 이후 결정을 제약해 다양성이 어디서 소실되는지 사전에 알기 어렵다. 논문은 이를 해결하기 위해 (1) 순환 학습/재추천이 포함된 closed-loop 시뮬레이션(모델-사용자-재학습 반복)과 (2) exposure-level 다양성·동질성 지표 + layer별 코드 엔트로피 기반 code-space metric을 함께 사용해 관찰 단위를 분리·통합했다.

- **Empirical Impact**: 실험은 Amazon Office Products와 Toys & Games에서 여러 feedback cycle 동안 2종 generative recommenders와 2종 전통 sequential baselines를 비교하는 방식으로 수행됐다. 결과적으로 generative recommenders는 exposure-level cocoon 형성에 전반적으로 덜 민감해 더 넓은 노출 다양성을 유지하고 사용자 간 동질화 속도를 늦추는 경향을 보였다. 하지만 feedback loop는 생성된 SID 코드 공간에서도 여전히 집중(코드 공간 cocoon)을 유발할 수 있었고, cocoon의 강도는 토큰화 전략과 모델 스케일에 크게 좌우됐다(협업 신호 기반 토큰화가 더 강함, 더 큰 모델이 코드 공간 다양성을 더 잘 유지). 이로써 정보 cocoon이 추천 행동뿐 아니라 tokenization과 모델 capacity라는 “생성 설계” 요소에 의해 달라진다는 실증적 근거를 제시한다.



### Temporal Preference Optimization for Unsupervised Retrieva (https://arxiv.org/abs/2606.17664)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: 기존 비지도 dense retriever들은 unlabeled 문서의 contrastive learning으로 의미적 유사도는 잘 학습하지만, 쿼리의 시간 맥락과 문서 timestamp가 어긋난 temporal misalignment를 놓치는 경우가 많았습니다. 감독 기반 temporal retriever는 성능이 좋더라도 명시적 timestamp 라벨이 필요해 대규모 적용이 어렵다는 한계가 있습니다. 반면 Contriever류의 time-unaware 모델은 의미는 맞는데 시간은 어긋난 문서를 함께 끌어오는 경향이 관찰됩니다.

- **Core Contribution**: 이 논문은 비지도 retriever에 시간 정렬을 “선호(preference)” 형태로 주입하는 TPOUR(Temporal Preference Optimization for Unsupervised Retriever)를 제안합니다. 핵심은 Temporal Retrieval Preference Optimization(TRPO)로, temporally aligned 문서를 선호하고 unaligned 문서를 덜 선호하도록 학습해 의미 유사도와 시간 적합성을 함께 강화합니다. 또한 time vector interpolation을 통해 retraining 없이 미지의 시간 구간(중간 시점, 미래)으로 연속적 일반화까지 확장합니다.

- **Technical Challenges**: 가장 큰 과제는 라벨 없이도 “시간 선호” 신호를 구성해 retriever가 temporally aligned와 misaligned를 구분하도록 만드는 것입니다. 저자들은 문서 코퍼스가 여러 시간대에서 수집되었다는 점을 활용해 학습 중 preferred/less preferred 쌍을 만들고, TRPO의 loss를 MoCo 기반 contrastive objective와 결합해 시간 차이를 임베딩에 반영합니다. 또 discrete time 모델의 경계를 넘기 위해, 생성 모델에서 쓰이던 time vector를 encoder retriever에 적용하고 중간 시점은 interpolation으로 처리합니다.

- **Empirical Impact**: 실험에서 TPOUR는 temporal information retrieval(T-IR)에서 비지도/감독 baseline을 모두 능가하며, explicit 및 implicit 시간 질의 모두에서 nDCG@5가 유의미하게 상승했습니다. 특히 Qwen-Embedding-8B 대비 약 72.7배 작은 규모임에도 평균 nDCG@5를 explicit +4.04(+12.15%), implicit +4.98(+15.21%) 개선했습니다. 더 나아가 BEIR 등 일반 검색에서도 데이터셋 publication year와 최적 성능이 정렬되는 시간 민감성이 드러나, 시간 모델링이 범용 검색에도 실질적 이득을 줄 수 있음을 보여줍니다.



### RSRank: Learning Relevance from Representational Shifts (https://arxiv.org/abs/2606.17468)
Comments:
          Under Peer Review

- **Prior Approaches**: 엔터프라이즈 RAG에서 reranking은 부정확/잡음 문서를 걸러내는 마지막 단계지만, 기존 SOTA는 점수 임계값(threshold)을 휴리스틱하게 정하거나 사후 보정에 의존해 오프더셋·오프쿼리 캘리브레이션이 자주 틀어집니다. 또한 relevance 점수 계산에 logit/attention 등 next-token prediction용 신호를 그대로 전용해 “순위는 잘하지만, 선택(필터링)은 부정확”해지는 문제가 지적됩니다. 실제로 Qwen3-Reranker-8B에서도 최적 threshold는 도메인별·쿼리별로 크게 변해 out-of-the-box 성능이 제한됩니다.

- **Core Contribution**: 이 논문은 relevance를 “문서가 쿼리의 내부 표현을 얼마나 특징적으로 바꾸는가”로 재정의하고 representational shift(RS, 표상 이동)라는 원칙적 신호를 제안합니다. 특히 후보 문서가 유발하는 RS가 oracle 문서셋(관련 문서들의 전체)에 의해 유발되는 RS와 정렬(alignment)되는 정도가 relevance의 견고한 지표가 된다고 관찰합니다. 이후 RS를 소형 투영(projection)으로 변환해 고정된 자연 임계선에서 캘리브레이션된 점수를 내는 lightweight 학습 프레임워크 RSRank를 제시합니다.

- **Technical Challenges**: 핵심 난제는 “문서 프리픽스가 유발하는 RS”를 분리해내는 것으로, 단순히 value/logit 변화를 보면 쿼리 길이·포지션 차이 등 잡음 요인이 섞일 수 있습니다. 논문은 RoPE 환경에서 상대 위치 성질을 이용해 null(의미 최소) 프리픽스와의 finite-difference로 RS를 구성하고, 레이어/헤드/토큰 단위의 RS 텐서를 만든 뒤 투영된 공간에서 one-vector(결정 경계) 정렬이 되도록 목적함수를 설계합니다. 또한 projection 차원 붕괴를 막기 위한 직교 정규화와, 관련/비관련 점수가 s=0 기준을 기준으로 margin을 갖도록 하는 다중 항 학습으로 “고정 threshold에서도 분리”가 되게 합니다.

- **Empirical Impact**: 여섯 개 검색/추론 데이터셋에서 RSRank는 평균 NDCG@5와 Recall@5에서 SOTA reranker 대비 우위를 보이며, 특히 multi-hop 벤치마크(2WikiMQA, MuSiQue)에서 큰 폭의 개선이 나타납니다. 무엇보다 designed threshold(τ=0)에서 F1이 평균 67.5로, Qwen3-Reranker-8B의 default threshold(τ=0.5, 60.3) 대비 7.2pp 향상되며 캘리브레이션이 크게 안정화됩니다. threshold bias/variance가 각각 17배(0.379→0.022), 47배(0.023→0.0005) 줄어 out-of-the-box 적용성에 직접적인 의미가 있습니다.



### On the Memorization Behavior of LLMs in Generative Recommendation: Observations, Implications, and Training Strategies (https://arxiv.org/abs/2606.17276)
- **Prior Approaches**: 생성형 추천(Generative recommendation, GR)에서 LLM을 쓰는 흐름은 next-item prediction으로 미세조정해 사전학습 지식을 활용하려 합니다. 다만 기존 연구는 LLM의 ‘암기(memorization)’ 성향을 충분히 점검하지 않았고, 그 결과 pretrained knowledge를 제대로 쓰지 못한 채 학습 데이터의 국소 패턴을 반복할 위험이 남아 있었습니다. 기존 GR과 LLM 기반 GR 모두 강점이 있지만, 특히 LLM이 어떤 형태로 데이터를 암기하는지와 그게 성능 향상에 얼마나 기여하는지는 불명확했습니다.

- **Core Contribution**: 이 논문은 LLM 기반 GR에서 나타나는 ‘one-hop memorization(원-홉 암기)’을 분석하고, 이 암기가 실제 성능 이득의 상당 부분을 설명함을 보여줍니다. 또한 한-홉 전이(직전 아이템의 다음 아이템)로는 연결이 부족한 사용자 그룹에서 LLM 성능이 덜 오르며, 이를 메우기 위해 더 풍부한 item-item 관계 학습이 필요하다고 주장합니다. 이를 해결하기 위해 Item–Item Relation Generation(IIRG)라는 학습 전략을 제안합니다.

- **Technical Challenges**: 핵심 과제는 one-hop 전이에 지나치게 의존하는 LLM을, 더 멀티-홉의 협업 신호와 의미적(semantic) 연관성까지 반영하도록 만드는 것입니다. 저자들은 협업 관계를 사용자 시퀀스에서 여러 hop 떨어진 구간에서 함께 등장하는 co-occurrence로 정의하고, 의미 관계는 텍스트 임베딩 기반 유사도(같은 테마/상보성)로 구성해 모델이 ‘다른 방식의 이웃’을 생성하도록 학습시킵니다. IIRG는 next-item prediction에 더해 collaborative-neighbor generation과 semantic-neighbor generation을 함께 최적화해, one-hop을 넘는 예측 신호를 학습하도록 설계했습니다.

- **Empirical Impact**: 실험에서 IIRG를 적용한 LLM은 next-item prediction만 학습한 LLM보다 Recall@5 기준 평균 21% 향상을 보였고, 특히 one-hop 전이로 커버되지 않는 사용자에서 50% 향상(대조군 대비 큰 폭)을 기록했습니다. 또한 one-hop 암기에 주로 의존했던 성능 구조가 완화되며, 협업·의미 이웃 신호가 실제로 일반화에 기여함을 뒷받침합니다. 생성형 추천에서 LLM의 ‘암기’를 단순 결함이 아니라 진단 변수로 활용하고, 이를 우회하는 학습 설계를 제시했다는 점에서 후속 연구에 의미가 큽니다.



### HistoRAG: Embedding Historical Methodology in Retrieval-Augmented Generation Through Critical Technical Practic (https://arxiv.org/abs/2606.18103)
Comments:
          25 pages, 6 figures. Companion preprint to a Journal of Digital History notebook article (under review)

- **Prior Approaches**: 기존 RAG는 검색-생성 파이프라인을 매끄럽게 연결해 유사도 기반으로 가장 그럴듯한 구절을 고르고 요약을 생성하는 데 최적화돼 있다. 하지만 역사 연구처럼 해석이 핵심인 분야에서는 ‘relevance’가 고정값이 아니라 연구자의 질문, 시기, 의미론에 따라 달라지며, 소스 선택 근거가 투명하고 논쟁 가능해야 한다.

- **Core Contribution**: 이 논문은 역사학의 방법론을 RAG 아키텍처 설계로 번역한 HistoRAG를 제안한다. Heuristik(소스 발견·정리)과 Analyse(해석)를 분리하고, 시간창(windowing)으로 시기별 소스 균형을 강제하며, LLM-as-judge로 사후 평가의 기준을 명시·검토 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 시대별 어휘 변화와 주제 밀도 편향 때문에 임베딩 유사도 검색이 초기에 해당하는 소스를 잘 놓치는 ‘temporal skew’를 다루는 것, (2) 유사도는 점수 기준(정말로 질문에 relevant한가)을 직접 반영하기 어렵다는 점이다. HistoRAG는 시기 구간별로 동일 수를 회수하는 temporal windowing과, 검색 후에 연구자 정의 rubric로 근거 인용을 포함한 점수·정당화를 산출하는 평가 레이어를 결합해 이를 해결한다.

- **Empirical Impact**: Der Spiegel 1950–1979년 102,189개 기사로 SPIEGELragged 평가를 수행했으며, 표준 RAG의 결함들이 정량적으로 확인됐다. era-specific vocabulary 문제, windowing을 정당화하는 temporal skew의 존재, 그리고 vector similarity와 LLM 평가 relevance의 약한 상관(Spearman rho=0.275) 등 측정 가능한 이유로 아키텍처 개입의 필요성을 보여준다. 또한 keyword/semantic retrieval의 소스 풀 불일치를 ‘공통 LLM 평가 필터’ 아래에서 상호보완적으로 쓰는 구조로 다뤄, 해석적 학술 실무에 적용 가능한 설계 모델을 제시한다.



### Designing Recommendation Exposure and Favorite Lists: A Field Experiment in a Spot-Work Platform (https://arxiv.org/abs/2606.17397)
- **Prior Approaches**: 기존 추천은 주로 클릭률·즐겨찾기 확률 같은 중간 지표를 최대화해 후보에 노출을 집중시키는 방식(예: Greedy)을 사용해 왔습니다. 하지만 이 설정에서는 추천 노출이 곧 실제 ‘기회 접근’을 좌우하고, 기회가 단기·희소하다는 점 때문에 중간 지표 최적화가 시장 수준 매칭 효율을 해칠 수 있습니다. 또한 고정된 추천 규칙은 템플릿(직무 템플릿)별 노동수요와 미충원 정도가 시시각각 달라지는 상황을 충분히 반영하지 못합니다.

- **Core Contribution**: 논문은 일본 스팟잡 플랫폼 Timee에서 “즐겨찾기(favorite) 관리가 동적 변환 장치”임을 전제로, 노출이 즐겨찾기 목록을 통해 미래 매칭으로 누적되는 구조를 모델링합니다. 그리고 즐겨찾기 확률만 높이면 ‘잘못된( misdirected ) 노출 쏠림’이 발생해, 인기 템플릿은 더 많이 노출되지만 실제 미충원 수요가 있는 템플릿은 충분히 노출되지 않는 문제를 정식화합니다. 이를 해결하기 위해 Thresholded eligibility control(TEC)라는 노출-할당 메커니즘을 제안합니다.

- **Technical Challenges**: 핵심 난제는 희소하고 짧게 소멸하는 기회에서, 추천이 ‘클릭’이 아니라 ‘실제 잡 매칭’으로 이어지도록 노출을 재배분해야 한다는 점입니다. TEC는 템플릿의 최근 posting activity와 unfilled capacity를 바탕으로 노출 자격(eligibility)을 임계값 기반으로 제한하고, 과도한 누적을 피하면서 수요가 큰 템플릿에 더 실어 나르도록 설계합니다. 또한 운영 환경을 고려해 TEC를 완전 병렬화(parallelizable)해 대규모 플랫폼에서도 실시간 적용 가능하도록 구현합니다.

- **Empirical Impact**: Timee 데이터에 보정한 시뮬레이션에서 TEC는 라운드당 잡 탐색 성공률(per-round job-finding rate)을 57.6%에서 70.0%로 끌어올립니다. 이어 도(都) 단위 무작위 필드 실험에서도 매칭 수와 ‘활성 템플릿’의 추천 노출을 늘리고, 저노출 템플릿의 비중을 줄였으며 추천 기반 즐겨찾기와 이후 매칭(다운스트림 matching) 성과가 개선됩니다. 즉, 추천 다양성과 미충원 수요 중심의 노출 재배분이 시장 전체 효율을 실증적으로 향상시킨다는 의미가 큽니다.



### Beyond Parallel Sampling: Diverse Query Initialization for Agentic Search (https://arxiv.org/abs/2606.17209)
Comments:
          15 pages, 8 figures; under review at EMNLP 2026

- **Prior Approaches**: 기존 test-time scaling for agentic search는 depth(추론 턴을 늘리는 방식)나 breadth(여러 롤아웃을 병렬로 실행)로 성능을 끌어올리는 데 집중해 왔습니다. 특히 breadth에서는 k개의 독립 롤아웃을 뽑아 투표/선택 등으로 합치는 전략이 흔했지만, 병렬 스레드가 초기에 비슷한 검색 쿼리를 내면서 증거를 중복으로 가져오는 문제가 관찰됩니다. 이로 인해 병렬성이 ‘자원 낭비’처럼 작동하며, 다턴 탐색에서 스레드들이 동시 실패하는 상관 오류가 생길 여지가 있습니다.

- **Core Contribution**: 이 논문은 병렬 에이전트 검색에서 turn-1 쿼리가 이후 탐색을 고정(anchor)해 버리는 anchor collapse 현상을 분석합니다. 그리고 이를 줄이기 위한 training-free 개입 DivInit을 제안하는데, 첫 턴에서 n개의 후보 쿼리를 한 번에 생성한 뒤 Maximal Marginal Relevance(MMR)로 k<n 중에서만 서로 다른(다양한) 시드 쿼리를 골라 병렬 롤아웃을 시작합니다. 핵심은 이후 에이전트 검색 루프는 그대로 두고, “어디서부터 시작할지”의 분포만 바꿔 다양성을 확보하는 것입니다.

- **Technical Challenges**: 기여를 실제로 만들기 위한 관건은 ‘독립 샘플링’만으로는 첫 턴에서 다양성이 충분히 확보되지 않는다는 점을 넘기는 것입니다. DivInit은 첫 턴에 대해서만 oversampling pool을 만들고, token 기반 Jaccard distance와 MMR로 선택된 시드들 간 최소 거리를 키우는 방식으로 다양성을 강제합니다. 또한 별도 학습 없이 개입을 수행해야 하므로, 공통 풀 생성 1회와 선택된 k개에 대한 병렬 스레드 실행으로 compute를 맞춰 설계했습니다.

- **Empirical Impact**: 5개 오픈웨이트 모델과 8개 벤치마크에서 DivInit은 standard parallel sampling 대비 전반적으로 더 높은 성능을 보이며, 멀티홉 QA에서 matched compute 조건으로 평균 5~7점 개선을 보고합니다. 특히 WebWalker 등 open-web 계열에서 증가 폭이 크게 나타나고, 성능 향상이 모델 크기에 따라 커져 “다양화가 먹히는 용량 바닥(capacity floor)” 가능성도 시사합니다. 또한 turn-1에서의 쿼리 다양성이 이후 turn들까지 ATD로 이어지며, 첫 턴 분리만으로 충분한 효과를 낸다는 분석이 함께 제시됩니다.



New uploads on arXiv(cs.CV)

### Future Dynamic 3D Reconstruction: A 3D World Model with Disentangled Ego-Motion (https://arxiv.org/abs/2606.18250)
Comments:
          ICML 2026. Project page: this https URL

- **Prior Approaches**: 기존 world model은 대부분 픽셀/이미지 특징만을 다음 프레임으로 예측하거나, 2D 공간에서 카메라 궤적과 장면 변화를 한 공간에 섞어 모델링해 왔습니다. 그 결과 롤아웃이 길어질수록 물체가 형태가 변하거나 사라지는 등 물리적 비일관성이 누적됩니다. 또한 3D 접근은 존재하지만 점유/센서 기반 또는 ego-motion과 world-motion을 명확히 분리하지 못해 장기 예측 안정성이 떨어지는 경우가 많습니다.

- **Core Contribution**: FR3D는 단안 관측으로부터 미래의 dynamic 3D reconstruction을 예측하는 새로운 task와, 이를 위한 world model을 제안합니다. 핵심은 미래까지 유지되는 persistent 3D latent representation를 학습하되, 장면(3D) 진화와 에이전트(카메라) 궤적에 의한 ego-motion을 decouple하고 ego-motion은 action의 latent proxy로 취급한다는 점입니다. 이 분리로 self-motion과 world-motion 간의 모호성이 줄어 기하 일관성이 장기 horizon에서 유지되도록 설계했습니다.

- **Technical Challenges**: 문제는 3D 기하를 보존하면서도 미래 동역학을 예측해야 하는데, 이미지 평면 기반 예측은 ego-motion과 world-motion이 얽혀 장기 롤아웃에서 물리적 붕괴가 발생합니다. FR3D는 CUT3R 같은 사전학습된 3D reconstruction 모델의 latent 공간 위에서 동작하도록 설계하고, pose를 예측하는 Pose Masked Transformer와 공간 토큰을 예측하는 Spatial Masked Transformer를 cross-attention으로 결합해 pose-geometry 상호제약을 학습합니다. 또한 teacher-student distillation으로 foundation model이 가진 공간적 common sense(기하 priors)를 주입해, 대규모 재학습 없이도 robust zero-shot generalization을 노립니다.

- **Empirical Impact**: 실험에서는 KITTI와 nuScenes에서 out-of-training distribution인 zero-shot 설정으로도 미래 2초 이상에서 depth와 pose 예측 성능이 기존 관련 baseline을 앞선다고 보고합니다. 특히 DINO-Foresight 계열보다 장기 horizon에서 일반화가 더 좋았는데, 이는 재구성 오라클의 latent 예측 파이프라인을 그대로 활용하고 기하 priors를 distillation으로 가져오기 때문이라고 설명합니다. 또한 Waymo로 학습한 뒤 Dynamic-RE10K 같은 동적 실내 환경으로 확장 가능함을 보이며, 물리 일관성을 중시하는 자율주행/로보틱스 분야에서 future dynamic 3D 예측의 실용성에 의미가 있습니다.



### Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer is Key to Unification (https://arxiv.org/abs/2606.18249)
Comments:
          Accepted by ICML2026. Project page this https URL

- **Prior Approaches**: 기존 Unified Multimodal Modeling 시도는 이해(understanding)와 생성(generation)을 각각 담당하는 서로 다른 visual tokenizers에 의존하는 경우가 많아, 표현 공간이 분리되고 end-to-end 통합이 어려웠습니다. 그 결과 모델이 생성한 시각 토큰을 동일한 기준으로 다시 해석하려면 추가 re-encoding이 필요해 효율과 일관성이 떨어진다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 UniAR로, 단 하나의 discrete visual tokenizer를 이해와 생성의 공통 “브리지”로 두는 unified autoregressive 프레임워크를 제안합니다. 이를 통해 모델은 자신이 만든 visual tokens를 별도 인코딩 없이 공유 컨텍스트 안에서 직접 해석하며, 이미지 생성과 편집을 동일한 토큰 체계로 연결합니다.

- **Technical Challenges**: 핵심 과제는 (1) 단일 tokenizer가 고수준 의미와 저수준 디테일까지 함께 담아야 하고 (2) 토큰 수(시퀀스 길이)를 늘리지 않으면서 vocab을 확장해야 한다는 점입니다. 논문은 multi-level feature fusion으로 전이 학습을 강화하고, lookup-free bitwise quantization으로 토큰 효율을 높였으며, unified autoregressive 모델에서는 parallel-bitwise-prediction으로 공간적으로 묶인 multi-level visual codes를 함께 예측해 시퀀스를 크게 줄입니다. 마지막으로 diffusion-based visual decoder가 discrete visual tokens에서 고화질 이미지를 복원합니다.

- **Empirical Impact**: 대규모 pre-training 후 supervised fine-tuning과 reinforcement learning을 결합한 학습 설정에서, UniAR는 이미지 생성과 이미지 editing에서 state-of-the-art 성능을 보였습니다. 동시에 multimodal 이해 벤치마크에서도 경쟁력 있는 결과를 내며, 통합 토큰화가 성능과 효율을 동시에 끌어올릴 수 있음을 실증했습니다.



### MOCHI: Motion Enhancement of Collaborative Human-object Interactions (https://arxiv.org/abs/2606.18243)
Comments:
          SIGGRAPH 2026 Journal (ACM TOG); Project page: this https URL

- **Prior Approaches**: 기존 협력적 인간-물체 상호작용(MHOI) 연구는 고품질 데이터 확보를 전제로 하며, 캡처 기반 접근은 인간-인간·인간-물체가 동시에 발생하는 복잡성 때문에 잡음이 큰 경우가 많습니다. 그 결과 손-물체 접촉 정렬 오류, 모션 지터 및 시간적 불일치, 손가락 수준 관절 표현 누락·불완전 같은 아티팩트가 빈번하게 관측됩니다. 생성 모델로 데이터를 보강하려는 시도도 있으나, 단일 캡처의 물리적 타당성과 상호작용 일관성을 동시에 안정적으로 복원하기는 어렵다는 한계가 있었습니다.

- **Core Contribution**: 논문은 잡음이 포함된 MHOI 데이터를 개선하는 2-stage 프레임워크 MOCHI를 제안합니다. 첫 단계에서 잡음이 섞인 바디 입력으로부터 물리적으로 그럴듯한 hand grasp를 최적화해 바디 포즈와의 의미적 정합성을 확보한 뒤, grasp를 손-물체 상호작용 전체 시퀀스로 확장합니다. 둘째 단계에서는 single-person motion priors를 활용한 diffusion 기반 잡음 최적화로 모든 참여자의 전신 모션을 함께 정제합니다.

- **Technical Challenges**: 핵심 기술 난제는 잡음 캡처에 내재한 손-물체 접촉 미스얼라인, 시간적 흔들림, 손가락 관절 결손을 서로 다른 참여자 간 상호작용까지 고려해 일관되게 복원하는 데 있습니다. MOCHI는 최적화 단계에서 물리적 그립의 실현가능성과 바디-그립 의미 정합을 같이 강제하고, diffusion 단계에서는 단일-person priors 안에 human-object·human-human interaction 정보를 인코딩하는 최적화 목적함수를 추가해 상호작용 신호가 priors에 흡수되도록 설계합니다. 이를 통해 전신 모션 정제 과정에서도 접촉과 상호작용의 구조가 무너지지 않게 했습니다.

- **Empirical Impact**: 실험은 기존 캡처 방식이나 생성 모델로 얻은 다양한 MHOI 데이터에 대해 MOCHI 파이프라인의 효과를 보여주며, 참가자 수가 달라지거나 상호작용 유형이 변해도 강건함을 확인합니다. 또한 keyframe 기반 MHOI 생성, 객체 형상 변형을 통한 데이터 증강 등 응용 가능성을 제시해, 단순 복원을 넘어 데이터 파이프라인 전반의 활용도를 높였다는 점에서 의미가 큽니다.



### EventDrive: Event Cameras for Vision-Language Driving Intelligenc (https://arxiv.org/abs/2606.18242)
Comments:
          CVPR2026, 34 pages, 15 figures, 15 tables, project page: this https URL

- **Prior Approaches**: 기존 event camera 연구는 주로 detection, segmentation, tracking, optical flow 같은 저수준 perception 작업에 집중돼 완전한 자율주행 루프의 추론·의사결정까지 연결되는 경우가 드뭅니다. event-to-language VLM도 caption/QA 등 제한된 범위에서의 상호작용에 머무르는 편이며, 프레임 기반 VLM은 저조도·모션 블러에서 성능이 크게 흔들리는 문제가 있습니다. 또한 events와 RGB를 결합하더라도 고정된 temporal window에 의존해 운동의 다양한 시간 스케일을 충분히 반영하지 못했습니다.

- **Core Contribution**: 이 논문은 event stream, 동기화된 RGB 프레임, 언어 감독을 한데 묶어 perception-interpretation-prediction-planning 전 과정을 아우르는 benchmark EventDrive를 제안합니다. 더 나아가 EventDrive-VLM은 비동기 events를 LLM의 추론 공간에 맞춰 정렬·융합하고, 멀티태스크로 driving reasoning을 수행하도록 설계됐습니다. 결과적으로 events를 주변 보조 신호가 아니라 주행 지능의 핵심 모달리티로 끌어올리는 것이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) event의 temporal density가 장면·센서·태스크에 따라 크게 달라 multi-scale 운동 단서를 보존해야 한다는 점과 (2) sparse하고 시계열적인 event 특징을 LLM의 의미 임베딩과 안정적으로 정렬해야 한다는 점입니다. 이를 위해 Dynamic Horizon Encoding으로 여러 temporal resolution에서 voxelized event를 만들고, Mixture-of-Experts gating으로 상황에 맞는 horizon을 선택해 계산을 효율화합니다. 또한 Event Q-Former로 learnable event query가 cross-attention을 통해 언어 추론에 필요한 motion-relevant 신호만 뽑아내고, LLM을 frozen한 event-language pre-adaptation 뒤 instruction tuning으로 통합 파이프라인을 완성합니다.

- **Empirical Impact**: 실험에서 frame-only 모델은 저조도·모션 블러에서 급격히 성능이 떨어지는 반면, event-enhanced 및 event-frame fusion 모델은 시간 정밀도, motion awareness, grounding·계획 관련 추론에서 전반적으로 더 견고한 향상을 보입니다. 특히 prediction과 planning에서 정적 프레임만으로 speed/direction을 추론하기 어려운 문제를 events의 고주파 temporal gradient가 직접 보완해 정확도가 일관되게 높아집니다. 또한 task 전반에서 events와 frames의 상보성이 확인되며, EventDrive-VLM의 표현이 다른 event-language 벤치마크로도 전이되는 경향이 보고돼 향후 event-driven driving 지능 연구에 실질적 기반을 제공합니다.



### Adaptive Volumetric Mechanical Property Fields Invariant to Resolution (https://arxiv.org/abs/2606.18231)
Comments:
          Project Page and hi-res paper: this https URL. ICML 2026

- **Prior Approaches**: 변형(Deformable) 시뮬레이션은 물성장(Young’s modulus E, Poisson’s ratio ν, density ρ)이 물체 내부 전부에 대해 공간적으로 주어져야 하지만, 기존 3D 에셋에는 이런 정보가 거의 없다. 이를 자동화하려는 학습 기반 접근들은 대부분 정확도나 해상도 한계로 인해 고해상도·고정밀 물성장을 만들기 어렵거나, 입력 격자 고정으로 메모리 효율이 떨어진다. 특히 VoMP와 같은 최고 성능 방법도 고정 해상도 격자를 쓰는 탓에 더 촘촘한 예측으로 확장할 때 비용이 급증한다.

- **Core Contribution**: AdaVoMP는 입력 3D 형상에 대해 공간적으로 변하는 물성(E, ν, ρ)을 조밀하게 예측하는 방법으로, 해상도·정확도·메모리 효율을 동시에 끌어올린다. 핵심은 기존 VoMP의 고정 voxel 모델을 대체해, 각 입력마다 고유한 sparse adaptive voxel 트리(SAV)를 생성하는 sparse transformer encoder-decoder를 도입한 것이다. 이로써 이전 대비 훨씬 높은 유효 해상도(예: 1024^3 급)에서 물성 경계와 복잡 영역을 더 정밀하게 복원한다.

- **Technical Challenges**: 큰 어려움은 (1) 고해상도 물성장을 만들면서도 (2) 격자 전체를 전개하지 않고 (3) 물성의 ‘빈 공간(Empty)’과 다중 해상도 구조를 일관되게 학습하는 것이다. AdaVoMP는 SAV로 균질 영역은 거친 셀로 유지하고 이질·경계만 미세 분할하도록 하며, 생성기(G)는 coarse-to-fine autoregressive로 “Empty/Keep/Subdivide” 구조를 함께 예측해 불필요한 연산을 줄인다. 또한 격자 없는 생성이 되도록 transformer가 통합 좌표 기반의 sparse attention을 수행하고, MatVAE 디코더를 고정해 물성(latent→E, ν, ρ)으로의 물리적 타당성을 유지한다.

- **Empirical Impact**: 실험에서 AdaVoMP는 기존 SOTA보다 더 정확한 volumetric properties를 추정하면서도, 모든 선행 대비 더 적은 test-time compute로 동작하는 경향을 보인다. 특히 적은 voxel로 균질 영역을 요약하면서도 복잡한 부분의 경계를 잘 살려, 시뮬레이션 가능한 고해상도 물성 에셋 변환에 직접적으로 유리하다는 점이 강조된다. 결과적으로 로봇 학습용 physics simulation in the loop 디지털 환경 제작에서 ‘물성 부여’ 병목을 크게 낮출 수 있는 실용적 임팩트를 가진다.



### EgoCS-400K: An Egocentric Gameplay Dataset for World Models (https://arxiv.org/abs/2606.18180)
- **Prior Approaches**: 기존 interactive world models은 영상 생성을 “제어 가능한 시뮬레이션”으로 보고 action-conditioned 제어와 긴 호라이즌 일관성을 요구해 왔습니다. 하지만 학습에는 키보드/마우스 같은 control 신호와 카메라, 내부 상태, 이벤트가 시간 축으로 정렬된 데이터가 부족했고, web video는 행동과의 약한 정렬, 로보틱스 데이터는 비용·다양성 한계가 있었습니다. 게임/시뮬레이터 데이터도 존재했지만 world model 학습에 필요한 재생(replay) 기반 타임라인 정렬과 다층(상태-이벤트-액션) 감독 설계는 부족했습니다.

- **Core Contribution**: 이 논문은 replay-grounded egocentric 카운터 스트라이크(CS:GO/CS2) 대규모 데이터셋 EgoCS-400K를 제안합니다. 공개 프로 CS 매치 demo를 “원본 궤적의 권위”로 삼아, 파싱된 플레이어 상태·시점(카메라)·입력·무기/유틸 이벤트를 동일 타임라인에 정렬된 1인칭 영상으로 렌더링합니다. 그 결과 10,000시간 이상, 40,000라운드 이상에서 400K+ 1인칭 비디오와 다층 캡션/프롬프트를 제공해, 수동 web video와 비싼 real-world embodied 데이터 사이의 실용적 브리지를 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난제는 ‘대규모로 얻기 어려운 action-상태-이벤트 정렬 궤적’을 정확히 재구성하는 동시에, 캡션 생성에 쓸 구조적 priors가 시각 근거를 안정적으로 주도록 만드는 것입니다. 논문은 demo를 tick 단위로 파싱해 per-tick state와 원자 단위 atomic action span을 만들고, action을 끊지 말아야 하는 cut-protected interval/chain으로 세그먼테이션을 제약합니다. 이어 VLM에 segment/protected-chain 단위로 action·이동·카메라 priors를 필터링해 노이즈/오매칭이 생기지 않게 하고, video 증거와 priors가 같은 시간 뼈대를 공유하도록 long_prompt용 JSON 응답 형식까지 강제합니다.

- **Empirical Impact**: EgoCS-400K는 action-conditioned future prediction, state-와 event-aware scene rollout, replay-grounded captioning, agent egocentric action understanding 같은 interactive visual modeling 과제를 지원하도록 설계됐습니다. 특히 렌더링-파싱-타임라인이 모두 demo로 되짚이는 audit 가능 구조라, 기존 약한 video-text 쌍에서의 정렬 문제를 줄일 수 있습니다. 카운터 스트라이크는 시뮬레이터지만 내비게이션·시점 제어·부분 관측·멀티에이전트 상호작용 같은 1인칭 역학을 대규모로 학습할 수 있는 중간 벤치마크로 의미가 큽니다.



### ReAge3D: Re-Aging 3D Faces with View Consistency (https://arxiv.org/abs/2606.18156)
- **Prior Approaches**: 기존 2D re-aging은 GAN/latent manipulation 또는 diffusion 기반 이미지-이미지 translation(예: InstructPix2Pix 계열)을 통해 노화/회춘을 생성하지만, 3D로 확장하면 뷰마다 주름·피부 질감 같은 미세 단서가 불일치하기 쉽다. 일반적인 3D scene editing(2D diffusion 편집 후 3D 최적화)도 멀티뷰 일관성을 맞추려 feature/노이즈/손실을 보정하나, re-aging 특유의 “아주 미세하지만 중요한” 나이 관련 디테일에서 과도한 스무딩 문제가 남는다. 특히 텍스트 기반 편집 모델은 fine-grained age control과 identity preservation에 맞춰 설계되지 않아 3D 얼굴에서 품질 격차가 커진다.

- **Core Contribution**: 이 논문은 3D face re-aging을 목표로, 먼저 2D diffusion 기반 re-aging 모델 DiffReaging을 제안하고 이를 멀티뷰 일관성 파이프라인에 결합한다. 핵심은 각 뷰를 독립적으로 편집하지 않고, “이미 re-aged된 피벗 뷰의 내용을 다른 뷰로 전파”해 age-related 디테일의 뷰 일관성을 유지하는 center-out editing propagation 전략이다. 또한 전파 시 누락 영역을 채우는 Masked-DiffReaging으로, diffusion의 반복적 denoising 과정마다 알려진 픽셀 콘텐츠를 주입해 기존과 충돌하지 않는 재구성을 유도한다.

- **Technical Challenges**: 가장 큰 기술 난제는 뷰 간 정합성이다: 2D에서 생성된 노화 디테일이 뷰마다 조금만 달라도 3D 최적화 과정에서 디테일이 흐려지는 over-smoothing으로 이어진다. 이를 해결하기 위해 논문은 optical flow 기반 warping으로 피벗의 re-aged 정보를 이웃 뷰에 정렬하고, Masked-DiffReaging이 시간 단계마다 confidence mask로 알려진 영역을 고정한 채 누락 영역만 일관되게 복원하도록 설계했다. 더 나아가 center-out 방식으로 중복/상충 재구성을 줄이며, 생성된 멀티뷰 타깃을 3DGS(또는 다른 미분 가능한 렌더러) 최적화의 감독 신호로 반복 갱신한다.

- **Empirical Impact**: 실험 결과, 제안 방법은 기존 3D 편집 기법 대비 시각적으로 더 자연스럽고(주름·피부 텍스처가 뷰 전반에서 더 매끄럽게 유지) 정량 지표에서도 우수한 성능을 보였다. 또한 원하는 나이(타깃 age)에 대해 identity를 보존하면서 세밀한 age transformation을 제어할 수 있어, 2D 중심 접근의 한계를 3D face re-aging으로 실질적으로 확장했다. 결과적으로 3D 얼굴 생성/편집 분야에서 “픽셀 수준 멀티뷰 일관성”을 diffusion 기반 편집과 연결하는 새로운 설계 방향을 제시했다.



### Neural Tree Reconstruction for the Open Forest Observatory (https://arxiv.org/abs/2606.18153)
Comments:
          Published as a workshop paper at "Tackling Climate Change with Machine Learning", ICLR 2024

- **Prior Approaches**: 기존 OFO 3D 산림 모델은 structure-from-motion(SfM) 기반으로 포인트 클라우드와 메쉬를 생성한다. SfM은 과도한 아티팩트가 생기기 쉽고, 특히 수관 아래/숲 바닥처럼 관측이 제한된 영역에서 세부가 부족해 복원 오차가 후속 과학 작업으로 전파될 수 있다. 또한 정교한 기하를 메울 수 없어 정밀한 줄기·잎 같은 미세 구조와 의미론적 분류에 불리하다.

- **Core Contribution**: 이 논문은 Open Forest Observatory(OFO)의 포레스트 데이터셋과 공개 소프트웨어·툴을 AI 커뮤니티에 소개하고, SfM의 한계를 Neural Radiance Fields(NeRF)로 개선하는 방향을 제시한다. 단일(프로토타입) 검증만으로도 NeRF가 더 높은 시각적 사실감과 메쉬 품질을 만들며, 이는 가상 환경에서의 정성적 전문가 판독 가능성과 같은 응용 확장 기회를 넓힌다는 점을 강조한다. 더 나아가 DBH(가슴높이 지름) 등 다운스트림 과학 지표에 맞춘 검증 체계까지 향후 과제로 연결한다.

- **Technical Challenges**: NeRF로 전환하려면 희소 시점에서도 안정적으로 장면을 재구성하고, 숲 바닥처럼 입력 가시성이 낮은 구역에서 ‘떠 보이는’ 언더스토리를 포함한 연결성/세부를 확보해야 한다. 저자들은 nerfstudio에서 제공하는 nerf-facto로 OFO 영상에 NeRF를 적용해 SfM 대비 가지·잎·줄기 등 시각 디테일과 메쉬 추출 품질을 개선하는 Proof of Concept를 보여준다. 또한 스케일 확장(대규모 NeRF)과 diffusion model을 결합해 언더스토리 복원을 강화하는 후속 전략을 구체적으로 제안한다.

- **Empirical Impact**: 프로토타입 실험에서 NeRF는 SfM 메쉬보다 사진 같은 질감을 더 잘 재현했고, SfM 포인트클라우드에서 흔한 floaters(떠다니는 점)로 인해 메싱이 어려운 문제도 완화되었다. 이는 종 분류, 줄기 측정 같은 다운스트림 작업에 실질적인 입력 품질 향상을 제공할 수 있음을 시사한다. 저자들은 또한 가상 투어 기반의 정성 평가를 비용·시간 절감으로 연결하고, 장차 DBH 중심의 ground-truth 기반 벤치마크로 3D 재구성 커뮤니티에 새로운 검증 지표를 제공할 가능성을 제안한다.



### Predicting Immune Biomarkers with MultiModal Mixture-of-Expert Pathology Foundation Models Empowers Precision Oncology (https://arxiv.org/abs/2606.18123)
Comments:
          5 figures

- **Prior Approaches**: 기존 mIF(멀티플렉스 면역형광) 예측 연구는 주로 단일 image modality에 의존해, 해상도 손실이 생기거나(패치→픽셀 근사 필요 등) 임상·생물학적 상보 정보를 충분히 활용하지 못하는 한계가 있었습니다. 또한 공개 데이터 확대로 확장 가능성은 커졌지만, 서로 다른 모달리티가 기여하는 방식을 고정 결합으로 처리해 최적 활용이 어렵다는 지적이 이어졌습니다.

- **Core Contribution**: MixTIME은 H&E whole-slide image에서 픽셀 수준 mIF 단백질 발현(연속값)을 예측하는 multimodal foundation model로, image-only(UNIv2), image-text(CONCHv1.5), image-transcriptomic(STPath) 및 사전 학습 mIF 예측기를 expert로 두는 MoE(mixture-of-experts) 구조를 도입했습니다. 학습 시 learnable router가 expert 기여도를 동적으로 가중하고, 분포/경향(tendency)을 반영하는 loss로 mIF의 강도와 경향을 동시에 보존해 성능을 끌어올립니다. 그 결과 예측된 mIF를 다운스트림 분석(공간 도메인, 생존, 리포트 생성, 약물저항 관련 신호 등)에 바로 활용할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 서로 해상도·표현공간이 다른 모달리티를 단일 예측 문제에 효과적으로 정합하고, (2) mIF의 magnitude(크기)뿐 아니라 marker별 상관/경향을 함께 맞추는 손실 설계를 갖추는 것입니다. MixTIME은 router로 모달리티 기여를 자동 조절하고, magnitude 수준과 correlation 수준(경향)을 동시에 다루는 분포·경향-aware loss로 학습 안정성과 정밀도를 확보했습니다. 또한 픽셀 레벨 예측을 지원해 패치 기반 근사 방식의 해상도 병목을 줄이려는 설계를 강조합니다.

- **Empirical Impact**: 두 규모의 데이터셋에서 17개 protein marker에 대해 PCC/SCC 기반 최첨단 성능을 보고하며, 특히 MoE 기반 모달리티 가중과 mixed loss의 조합이 실험적으로 유효함을 ablation으로 확인했습니다. 더 나아가 예측 mIF 임베딩은 공간 도메인 식별(클러스터링 지표 전반 개선)과 weakly supervised 생존 예측(C-index 개선)에 기여했고, 다기관 병리사 평가를 통해 AI 병리 리포트 생성 품질도 여러 기준에서 경쟁력 있게 향상되는 결과를 제시했습니다. 마지막으로 time points에 따른 단백질 발현 동역학을 추적하고 약물저항/면역억제와 연관된 단서까지 도출해, 임상 번역 및 바이오마커 발견의 확장성을 보여줍니다.



### HLS-GPT: A Generative Pretrained Transformer (GPT) for Continental-Scale NASA Harmonized Landsat and Sentinel-2 (HLS) Reflectance Reconstruction Across All Bands on Arbitrary Dates (https://arxiv.org/abs/2606.18115)
- **Prior Approaches**: 기존 Landsat·Sentinel-2 반사율 시계열 재구성 딥러닝은 스펙트럼 커버리지 제약, 지리적 확장성 부족, 혹은 짧은 시간창에 의존한 patch 기반 설계 한계가 컸습니다. 또한 모델이 모든 밴드·모든 날짜·전 픽셀 위치를 일관되게 다루지 못해 범용 적용에 제약이 있었습니다.

- **Core Contribution**: 본 논문은 NASA Harmonized Landsat Sentinel-2 30 m surface reflectance를 전 밴드·임의 날짜·임의 픽셀 위치에서 재구성하는 대규모 생성 사전학습 Transformer 모델 HLS-GPT를 제안합니다. Landsat와 Sentinel-2의 밴드 구성을 계층형 Transformer로 처리하고, 12개월 단일 픽셀 time series를 입력으로 삼아 두 위성의 차이를 흡수합니다.

- **Technical Challenges**: 주요 기술 과제는 (1) 두 위성의 서로 다른 spectral band configuration을 한 모델에서 안정적으로 다루는 것, (2) 계절성과 지리적 다양성(잡다한 지표 특성·작물 생장 패턴)을 충분히 학습하는 것, (3) 관측이 불규칙하고 희소한 상황에서 결측을 복원하는 것입니다. HLS-GPT는 CONUS 전역 9년치(0.25M+ 학습 픽셀)의 time series를 학습하고, epoch마다 랜덤 cropping·masking으로 시작일이 다른 12개월 구간을 만들며 유효 관측의 50%를 마스킹해 나머지로 마스킹 값을 복원하도록 학습했습니다.

- **Empirical Impact**: 62,000+ 독립 테스트 픽셀에서 조건이 다양한 지표(복잡한 작물 생장, 불규칙·희소 관측)에도 재구성이 견고함을 보였고, leave-one-observation-out 평가에서 모든 HLS spectral band의 RMSE가 0.026 이하로 보고됐습니다. 가시광 밴드는 상대 RMSE 35% 이하, 그 외 밴드는 13% 이하였으며, red-edge 밴드는 Landsat에 red-edge 밴드가 없는데도 red·near-infrared 수준의 오차를 보였습니다. 또한 테스트 관측을 10%~90% 마스킹한 민감도 분석에서 10%~50% 마스킹 구간은 열화가 완만했고(all-band RMSE 0.028 이하), 9개 독립 109×109 km 타일 CONUS HLS 이미지 재구성에서도 두 기존 방법과 NASA-IBM Prithvi보다 성능이 우수했습니다.



### When LLMs Analyze Scars: From Images to Clinically-Meaningful Features (https://arxiv.org/abs/2606.18063)
- **Prior Approaches**: 기존 스카(SCAR) 판별은 Vancouver Scar Scale(VSS), Patient and Observer Scar Assessment Scale(POSAS) 같은 기준을 반영하는 손수 설계 특징(색·질감·형상)에서 출발했지만, 영상 조건 변화에 취약했다는 한계가 있었다. 이후 CNN/ViT 같은 end-to-end 딥러닝이 성능을 높였으나, 임상 라벨 부족과 개인정보 제약 때문에 작은 데이터에서 과적합·일반화 실패가 잦고 결정 과정이 불투명해 신뢰 확보가 어렵다. 멀티모달 LLM(GPT-4V 등)로 직접 분류를 시도한 방법은 추론은 강하지만, 외부 서버 전송 이슈·재현성 저하·블랙박스 특성 때문에 임상 적용이 까다롭다.

- **Core Contribution**: 이 논문은 ScaFE(Scar Feature Engineering)라는 프레임워크로 LLM을 end-to-end 분류기가 아니라 지식 기반 feature engineer로 재정의한다. LLM이 VSS/POSAS 같은 임상 기준을 프롬프트로 제공받아, 결정적(deterministic)인 Python 코드 형태의 특징 추출기(ϕ)를 생성하고 이를 이용해 저차원·임상 해석 가능한 feature vector로 변환한 뒤 가벼운 분류기로 학습한다. 핵심은 “임상 지식은 코드를 통해 고정하고, 통계 학습은 소형 모델이 담당”하게 만드는 구조로 데이터 효율성과 해석가능성을 함께 노린다는 점이다.

- **Technical Challenges**: 가장 큰 기술 난제는 LLM의 의학 지식을 실제 영상 처리 파이프라인으로 정확히 ‘실행 가능’하게 옮기면서도, 결과를 재현 가능하게 만드는 것이다. 논문은 temperature=0 같은 설정과 함께, LLM이 생성한 코드를 문법 검사·샘플 실행·출력 차원 검증으로 반복 정제해 유효한 결정적 추출기를 확보한다. 또한 색/질감/형상 등 임상 범주에 맞춘 특징 그룹을 코드가 산출하도록 설계해, 모델이 임의의 통계량을 학습하는 것을 줄이고 임상 용어와 정합되게 했다.

- **Empirical Impact**: 40장(케로이드 20, 하이퍼트로픽 20)처럼 초소량 데이터 환경의 병리 scar 이진 분류 실험에서 ScaFE는 end-to-end 딥러닝 및 LLM 직접 분류(MMLM-Direct) 대비 일관되게 우수하거나 견줄 만한 성능을 보인다. 학습 데이터가 샷 수 2장/클래스까지 줄어들어도 성능 하락이 크지 않아 few-shot에서도 강건함을 시사한다. 특징 기여 분석에서는 morphological 특징 제거 시 성능 저하가 가장 커, 임상 경험과 맞닿은 판별 신호를 LLM-생성 특징이 잘 포착하고 있음을 보여준다.



### PhaseWin: An Efficient Search Algorithm for Faithful Visual Attribution (https://arxiv.org/abs/2606.18008)
Comments:
          26 pages, 29 figures

- **Prior Approaches**: 시각 어트리뷰션은 입력의 지역(이미지 분할 영역)들이 특정 출력(분류/검출/그라운딩/캡션)에 얼마나 기여하는지 설명하며, 이를 보통 지역 중요도 순서로 모델링합니다. 기존 search-based 방법은 sufficiency–necessity 프록시로 greedy 선택을 수행하지만, 해당 프록시는 전형적인 submodular 함수로 정당화될 수 없고 이론적 보장 근거가 약합니다. 또한 표준 greedy는 매 선택마다 남은 후보 전체를 다시 재평가해 O(n^2) 모델 평가 비용이 발생해 대형 비전·멀티모달 모델에 부담이 컸습니다.

- **Core Contribution**: 이 논문은 greedy의 “순서 기반 증거 축적” 성질은 유지하면서, 전체 재스코어링 대신 단계적 coarse-to-fine 탐색을 수행하는 PhaseWin을 제안합니다. PhaseWin은 각 phase마다 글로벌 후보 스크리닝과 adaptive pruning을 한 뒤, 남은 영역 중 일부에 대해서만 local window refinement로 정밀 재평가를 진행하고, 애매한 후보는 다음 phase로 defer합니다. 결과적으로, 어트리뷰션에서 핵심인 “삽입 시 응답 회복이 빨라지는 순서” 성격을 보존하면서 계산량을 줄이는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 (1) sufficiency–necessity 프록시를 submodularity 없이도 설명 가능한 이론 틀로 재구성하고, (2) greedy의 점수 재계산 병목을 제거하면서도 순위/충실성(near-greedy faithfulness)을 깨지 않는 알고리즘 설계입니다. 논문은 monotone evidence-accumulation 같은 명시적 조건과 feature-level 구조 가정을 두고 PhaseWin이 선형에 가까운 평가 복잡도를 달성하며 그리디의 순위 동작을 근사 보장함을 증명합니다. 또한 window 정책에 따라 실제 재평가 횟수가 fψ(ω)로 제어되도록 창(window) 기반 로컬 탐색과 early-exit/accept-or-defer 규칙을 결합했습니다.

- **Empirical Impact**: Image classification, object detection, visual grounding, image captioning 전반의 실험에서 PhaseWin은 다른 어트리뷰션 방법 대비 높은 faithfulness를 유지하면서도 forward pass 수를 크게 줄였습니다. 특히 분석에서 기대한 O(n^2)에서 O(n) 수준으로의 평가량 감소 경향이 실험적으로도 관측되며, metric gap은 작게 유지된다고 보고합니다. 이는 “고충실성 greedy-style 어트리뷰션”의 실용성을 대형/고해상도/블랙박스 환경까지 확장할 수 있는 유의미한 진전으로 해석됩니다.



### AIGS-Net: Compact Illumination Field Modeling via 2D Gaussian Splatting for Fast Low-Light Image Enhancemen (https://arxiv.org/abs/2606.17998)
- **Prior Approaches**: 기존 저조도 영상 향상(LLIE)은 히스토그램 이퀄라이제이션이나 Retinex 기반 물리 모델처럼 반사/조명 맵에 대한 고정 가정을 두거나, CNN 기반 residual·주의/다중스케일 구조, 또는 diffusion·implicit neural representations 같은 생성형 접근을 써왔다. 다만 조명(illumination-field) 표현 능력과 계산 복잡도 사이의 병목이 커서, 모바일·엣지 실시간 제약에서 배포가 어렵다. 또한 2D Gaussian Splatting을 활용한 일부 방법은 조명을 정적 priors로 두는 경우가 많아 입력마다 달라지는 밝기 분포와 노이즈/색 왜곡을 충분히 복원하지 못한다.

- **Core Contribution**: 이 논문은 Adaptive Illumination Gaussian Splatting Network(AIGS-Net)를 제안해, 입력에 맞게 변하는 2D Gaussian Splatting 조명 필드를 통해 저조도 조명 보정을 수행한다. 각 Gaussian basis의 opacity를 입력의 상대 휘도(relative luminance) 통계로 동적으로 조절하고, ordered alpha compositing으로 공간별 보정장을 렌더링해 장면 구조에 더 잘 맞춘다. 여기에 zero-parameter multiscale contextual encoding과 노이즈/색 안정화(단일채널 Gamma, 교차채널 일관성)를 결합해 과증폭과 색 치우침을 줄인다.

- **Technical Challenges**: 핵심 난제는 (1) 조명 필드를 입력 의존적으로 표현하면서도 파라미터·추론 지연을 극도로 줄이는 것, (2) 보정 과정에서 어둠 영역 노이즈가 증폭되고 센서 색 바이어스가 보존되는 현상을 억제하는 것이다. AIGS-Net은 Gaussian opacity를 상대 휘도 통계로 모듈레이션해 정적 prior의 붕괴를 막고, 로컬 평균/대비는 학습 파라미터 없이 shift-based multiscale 집계로 뽑아 구조 인식을 제공한다. 또 noise-mask 추정과 locked single-channel Gamma mapping, cross-channel consistency regularization, target color-alignment 제약을 더해 노이즈·색 왜곡 경향을 완화한다.

- **Empirical Impact**: LOL, LSRW 벤치마크에서 AIGS-Net은 디테일 복원과 색 충실도(색 fidelity)에서 개선을 보이면서도 약 40 learnable 파라미터 수준의 극단적 경량화를 달성한다. 즉, enhancement quality와 extreme inference efficiency 사이의 현실적인 절충점을 실험적으로 확인한 셈이다. 결과적으로 조명 표현력을 유지하면서도 모바일/엣지 배포 친화적인 저조도 복원 방향을 제시했다.



### Recover Semantics First, Generate Better: Improved Latent Modeling for 3D MRI Reconstruction and Cross-Contrast Synthesis (https://arxiv.org/abs/2606.17989)
Comments:
          Code: this https URL

- **Prior Approaches**: 기존 크로스-대비(contrast) MRI 합성은, 누락된 시퀀스를 생성모델이 추론하도록 설계했지만 3D에서는 볼륨 크기 때문에 픽셀 공간 생성이 비싸다. 그래서 VAE/VQ로 3D를 latent로 압축한 뒤 GAN이나 Diffusion으로 합성하는 계층형 파이프라인이 주류다. 다만 기존 압축기는 장거리 해부학 일관성(long-range anatomical coherence)을 충분히 보존하지 못하고, 대비별(contrast별) 의미를 latent에서 뒤엉키게 만들며, MSE 같은 손실로 과도한 smoothing을 유발해 downstream 생성 품질을 떨어뜨린다.

- **Core Contribution**: 이 논문은 생성모델에 앞서 “semantics-first”로 3D MRI latent를 더 잘 만들자는 방향을 제안한다. 핵심은 Latent Harmonization Encoder(LHE)로 전역 해부학 의존성을 길게 잡아 구조 파편화를 줄이고, Semantic Recovery Block(SRB)로 self-supervised semantic teacher(DINO)의 고수준 priors를 latent에 주입해 대비별 의미 분리(contrast-wise separability)를 강화하는 것이다. 마지막으로 Anatomy-aware Frequency Loss(AFL)로 진단에 중요한 고주파 디테일을 상황별로 보존해 smoothing 부작용을 완화한다.

- **Technical Challenges**: 3D 볼륨에서 전역-국소 정보를 동시에 다루되, 압축(discretization) 과정에서 장거리 관계가 깨지지 않게 설계하는 것이 큰 난제다. 저자들은 로컬 컨볼루션 특징과 slice-wise ViT의 전역 context를 channel-wise feature alignment 후 residual fusion하고, FSQ로 discretize해 장거리 해부학 관계가 유지되도록 했다. 또한 semantic entanglement와 과스무딩을 동시에 줄이기 위해, SRB에서 quantized latent를 teacher semantic 공간에 정렬하고(Aspiring separability), AFL에서 해부학 경계(공간 gradient)와 teacher 기반 의미 주의(semantic attention)를 결합한 가중치로 고주파 잔차만 L1로 맞추는 방식으로 해결한다.

- **Empirical Impact**: BraTS와 IXI의 멀티-대비 MRI에서 재구성(3D MRI reconstruction)과 대비 합성(cross-contrast synthesis) 모두에서 일관된 성능 향상을 보였다. 예를 들어 BraTS에서 최고 baseline 대비 PSNR이 0.41dB 개선됐고, IXI에서는 33.65 PSNR 및 최저 LPIPS(0.0450)를 기록했다. 또한 해당 semantically aligned latent를 CycleGAN과 Latent Diffusion에 적용하면 PSNR이 각각 3.48dB, 2.78dB 상승해, 이득이 특정 생성기 구조가 아니라 “latent의 품질”에서 온다는 점을 실증했다.



### Gaussian Light Field Splatting: A Physical Prior-Driven Vision Transformer for Unsupervised Low-Light Image Enhancemen (https://arxiv.org/abs/2606.17985)
- **Prior Approaches**: 기존 비지도 LLIE(저조도 이미지 향상) 연구는 Retinex 기반 unfolding, implicit neural representation, diffusion 모델처럼 가시성을 높이는 데 집중했지만, 복잡한 비균일 조명에서는 국소 exposure imbalance와 색 왜곡이 쉽게 발생했다. 또한 Vision Transformer 계열은 long-range 의존성은 강하지만, 조명 열화의 물리 prior를 명시적으로 모델링하지 않아 날카로운 공간 변화가 있는 환경에서 자연스러운 매끈한 enhancement gain 추정이 어려웠다. 2DGS(2D Gaussian Splatting)는 연속 신호를 가우시안 커널로 표현할 수 있다는 장점이 있으나, 이를 Transformer attention에 물리적 편향으로 체계적으로 연결한 접근은 부족했다.

- **Core Contribution**: 본 논문은 GLFS(Gaussian Light Field Splatting)라는 Gaussian light field splatting 기반 Vision Transformer를 제안해, 2DGS의 연속 조명(빛장) prior를 self-attention에 직접 내장한다. GLFS는 장면 조명을 이방성(anisotropic) 가우시안 기저함수들의 중첩으로 표현하고, attention 계산에 물리 기반 편향을 삽입해 공간 gain field를 더 균일하고 정확하게 복원하도록 설계했다. 아울러 색 일관성과 구조 보존을 위해 color-vector angular loss와 luminance-edge loss를 추가해 무감독 설정에서도 색 편향과 고주파 디테일 손상을 줄인다.

- **Technical Challenges**: 핵심 난제는 (1) 비지도 환경에서 단일 저조도 관측만으로 매끈한 조명 기반 gain을 복원하는 것, (2) Transformers의 데이터 구동 black-box attention에 물리 prior를 안정적으로 결합하는 것, (3) 조명 보정 과정에서 hue(색상) 유지와 구조(경계/텍스처) 보존을 동시에 달성하는 것이다. GLFS는 이를 위해 multi-scale Gaussian tokenization으로 가우시안 파라미터(중심, 공분산, opacity)를 추정하고, Gaussian Splatting Attention에서 Mahalanobis 거리 기반 anisotropic Gaussian affinity로 물리 편향을 attention logit에 주입하며, head별 게이팅으로 early-semantic/late-physics 균형을 맞춘다. 마지막으로 무감독 학습에서는 색의 방향(벡터 각) 일관성을 강제하는 angular loss와 Y 채널의 경계를 보존하는 edge loss로 색 왜곡과 구조 열화를 동시에 제약한다.

- **Empirical Impact**: 저자들은 대규모 ablation과 정량 평가를 통해 GLFS가 조명 보정(특히 비균일 조명에서의 균일성)과 디테일 보존에서 뚜렷한 개선을 보인다고 보고하며, 기존 방법 대비 state-of-the-art 성능을 달성했다고 주장한다. 특히 물리 prior를 attention에 내장한 설계가 local artifact와 색 왜곡을 줄이고, 손실 설계가 구조적 fidelity를 강화하는 기여가 확인된다. 이 결과는 LLIE에서 단순 데이터-매핑을 넘어 ‘연속 물리 표현(2DGS)–Transformer 결합’이라는 새로운 표현 패러다임의 유효성을 보여주는 사례로 해석된다.



### SegDINO: Introducing Multi-Scale Structure into DINO for Efficient Medical Image Segmentation (https://arxiv.org/abs/2606.17972)
Comments:
          Code: this https URL

- **Prior Approaches**: 기존 DINO 계열 self-supervised 표현은 전이 성능이 좋아 segmentation에도 적용해왔지만, 그대로 쓰면 멀티스케일 계층이 명시적으로 부족해 정교한 경계나 작은 병변에서 성능이 쉽게 떨어진다. 이를 보완하려고 무거운 decoder와 복잡한 upsampling/멀티스케일 fusion을 얹는 방식이 많아 파라미터와 연산 비용이 커지는 한계가 있었다. 또한 SAM 기반 접근은 zero-shot은 강하지만 downstream 적용 시 fine-tuning 비용 부담이 실무 효율을 낮춘다.

- **Core Contribution**: 본 논문은 SegDINO로 DINOv3 백본을 유지하면서도 “스케일 모델링”을 효율적으로 설계해 segmentation에 맞게 표현을 재구성한다. Token Pyramid Adaptation(TPA)은 서로 다른 DINO depth의 중간 토큰을 pseudo multi-scale hierarchy로 재배치해 멀티스케일 다양성을 주입한다. Scale-Aware Decoding(SAD)은 가벼운 intra-scale refinement와 top-down inter-scale propagation으로 세밀한 경계 복원을 돕는다.

- **Technical Challenges**: 핵심 난제는 DINO 기능이 강하더라도 기본적으로 동일한 패치 그리드 해상도에 묶여 있어, segmentation이 필요로 하는 계층적 스케일 통합을 decoder에 과도하게 떠넘기게 된다는 점이다. SegDINO는 TPA에서 토큰을 2D feature map으로 reshaping하고 1x1 투영 및 strided convolution resizing으로 계층적 공간 해상도를 만들며, SAD에서는 residual refinement와 top-down 전파를 분해해 연산을 최소화한다. 또한 작은 병변에서 중요한 미세 구조를 안정적으로 다루기 위해 TPA의 스케일 다양성이 특히 효과적임을 실험적 분해(ablations)로 확인한다.

- **Empirical Impact**: PanCT(췌장 CT, 284명, 방사선 전문의 주석)와 TN3K, Kvasir-SEG, ISIC 등 3개 공개 벤치마크에서 SegDINO는 일관되게 state-of-the-art 성능을 보이면서도 효율을 유지한다. 특히 PanCT의 small-lesion 세팅에서 최상위 Dice 및 HD95를 기록해 미세 타깃에 강점을 보여준다. 모델은 총 27.68M 파라미터(대부분 DINOv3-S)로 가볍고 51 FPS 수준의 추론 속도를 보고해, 의료 segmentation에서 accuracy–efficiency 균형을 실증했다.



### Reload-Mamba: Hierarchical Anti-Dilution State-Space Modeling for Multi-Class Semantic Segmentation (https://arxiv.org/abs/2606.17966)
Comments:
          23 pages, 4 figures, 17 tables. Code will be released soon

- **Prior Approaches**: 기존 의미론적 분할은 FCN·U-Net·DeepLabv3+처럼 전역 문맥과 로컬 경계 복원을 함께 다루지만, attention의 비용이 커지면서 효율적인 long-range 모델이 요구돼 왔습니다. Mamba 기반 state space model은 linear-time으로 장거리 의존을 다루지만, scan 경로를 따라 순차 전파되며 경계·얇은 구조·작은 범주에서 로컬 응답이 약해지는 propagation-induced response dilution 문제가 생길 수 있습니다. 선행 anti-dilution은 이 희석을 이진(binarization) 설정에서 단일 레벨로 복원하는 데 효과적이었으나, multi-class 분할의 경계 식별·클래스 불확실성·스케일별 희석을 그대로는 만족하지 못했습니다.

- **Core Contribution**: 이 논문은 Reload-Mamba로, Mamba 전파로 인한 응답 희석을 의미론적 분할 관점에서 직접 “복원해야 하는 영역”까지 겨냥하도록 재설계합니다. (1) decoder 레벨마다 ground-truth 경계 마스크로 boundary-supervised local detail prior를 명시적으로 학습하고, (2) pre-reload auxiliary head로 얻는 per-pixel class entropy를 Reload Gate의 추가 입력으로 사용해 class-uncertainty-aware 복원을 가능하게 하며, (3) 3개 decoder 레벨에서 계층적으로 Reload를 수행한 뒤 top-down으로 복원 표현을 융합합니다. 즉, 단순 포팅이 아니라 multi-class dense prediction에 맞춘 3가지 분할 특화 설계를 통해 희석을 스케일별로 되살립니다.

- **Technical Challenges**: 핵심 기술 난제는 Mamba의 순차 전파가 로컬 디테일을 약화시키는 상황에서, “어디에” 복원이 필요한지와 “얼마나” 복원해야 하는 신호를 안정적으로 제공하는 것입니다. 이를 위해 논문은 경계 민감 영역을 확률만으로 유추하지 않고 ground-truth boundary mask로 prior를 감독하고, ambiguity가 높은 곳을 class entropy로 추적해 Reload Gate가 멀티클래스에서만 의미 있게 작동하도록 구성했습니다. 또한 단일 해상도 refinement로는 큰 영역·중간 객체·얇은 구조를 동시에 커버하기 어렵다는 점을 3개 decoder 레벨에서 Reload를 나눠 적용한 뒤 계층 융합으로 보완합니다.

- **Empirical Impact**: Reload-Mamba는 ConvNeXt-Tiny 인코더와 다중 스케일 디코더, 4방향 Mamba scanning(픽셀 방향 주의 포함) 조합으로 ADE20K에서 single-scale 47.9%(multi-scale 48.9%), Cityscapes에서 single-scale 83.2% mIoU를 달성합니다. ResNet-101에 COCO pre-training 및 DeepLab 스타일 프로토콜을 적용하면 PASCAL VOC 2012 val에서 87.8% mIoU를 기록합니다. 특히 ADE20K에서 prior anti-dilution의 직접 포트 대비 누적 +2.2 mIoU 개선이 ablation으로 확인되며, 이 3개 분할 특화 설계가 각각 단독 기여함을 실증합니다.



### Robustness of Similarity-based Positional Encoding Under Rotations: Theoretical Analysis and Experimental Validation (https://arxiv.org/abs/2606.17961)
- **Prior Approaches**: Transformer에선 self-attention이 순열불변이라 위치 정보가 필수이며, 기존 비전에서는 sinusoidal·learned absolute처럼 좌표를 직접 주입하거나 relative/rotary처럼 토큰 간 관계나 벡터 회전을 사용해 왔습니다. Similarity-based positional encoding(simPE)은 좌표계 의존을 줄이고 토큰 표현의 pairwise similarity로 위치 구조를 표현한다는 점에서 기존 계열과 차별됩니다. 다만 simPE의 회전(기하) 섭동에 대한 이론적 거동이 충분히 규명되지 않아, 의료영상처럼 작은 회전이 잦을 때 “왜 안정적인가”가 불명확했습니다.

- **Core Contribution**: 이 논문은 simPE가 일반적으로 rotation-invariant는 아니지만, 회전에 대해 “안정적(stable)”일 수 있음을 이론+실험으로 동시에 보입니다. 특히 simPE를 elementary operator의 조합으로 보고, 각 구성요소에 대한 mild Lipschitz 가정 하에 회전 섭동이 positional encoding에 얼마나 영향을 주는지 정량적 상계를 도출합니다. 또한 작은 각도 영역에서 각도 크기에 선형으로 응답이 제한됨을 명시적 bound로 제공합니다.

- **Technical Challenges**: 핵심 난점은 simPE가 좌표 기반이 아니라 similarity 연산을 통해 구성되기 때문에, 회전에서 exact 불변성을 기대할 수 없다는 점입니다. 논문은 대신 Lipschitz 연속성을 통해 구성요소별 변화량이 전체 출력 변화로 어떻게 전파되는지 추적하며, 특히 normalization 같은 비정규 항은 원점에서 Lipschitz가 깨질 수 있음을 제외(domain에서 0 벡터를 피함)하는 방식으로 안정성 조건을 정리합니다. 그 결과 Frobenius norm 기준의 전역 추정치와 small-angle bound를 함께 제시합니다.

- **Empirical Impact**: 실험은 학습·검증 이미지는 고정된 canonical orientation으로 두고, 테스트만 점진적으로 회전시켜 “증강효과 없이” encoding의 내재적 안정성을 측정하도록 설계됐습니다. 네 가지 통제 데이터셋(Arrow, Shapes, Digits, FashionMNIST)에서 simPE는 회전 각이 커질수록 accuracy뿐 아니라 F1, precision, recall에서도 learned absolute positional embedding보다 일관되게 우수한 성능을 보였습니다. 특히 작은~중간 각도 구간에서 격차가 크게 나타나, 제시된 stability guarantee가 실제 성능 저하를 완화한다는 점을 경험적으로 뒷받침합니다.



### Beyond Visual Cues: CoT-Enhanced Reasoning for Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2606.17958)
Comments:
          Accepted to MICCAI 2026

- **Prior Approaches**: 기존 semi-supervised medical image segmentation은 주로 unlabeled 데이터에 consistency regularization이나 pseudo label을 적용해 성능을 끌어올리지만, 핵심은 결국 pixel-level 시각 유사도에 의존합니다. 이 방식은 경계가 애매하거나 저콘트라스트/아티팩트가 있거나, 형태가 비슷한 병변이지만 진단 논리는 다른 ‘visual-semantic mismatch’ 상황에서 쉽게 무너질 수 있습니다. 또한 텍스트 기반 접근도 표면적인 의학 문장/설명만으로는 실제 방사선 판독의 논리를 충분히 담지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Chain-of-Thought(사고 연쇄) 기반 Reasoning Segmentation인 CERS(CoT-Enhanced Reasoning Segmentation)를 제안해, 시각 패턴 매칭을 넘어 진단 논리에 가까운 의미 단서를 segmentation에 주입합니다. LLM이 생성한 CoT(진단적 추론 설명)를 labeled 데이터로만 knowledge pool에 구축하고, retrieval이 ‘시각적으로 비슷함’이 아니라 ‘논리적으로 일관된 과거 사례’를 고르도록 설계합니다. 이후 Multi-scale Coordinate Attention Module(MCAM)로 이 추론 맥락을 디코딩 단계에 효과적으로 융합해 경계 모호성과 의미 불일치를 줄입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) LLM이 만든 CoT가 충분히 신뢰할 만해야 하고, (2) unlabeled에서도 잘못된 근거를 가져오는 hard negative를 효과적으로 제거해야 하며, (3) 추론 기반 맥락을 segmentation 네트워크에 안정적으로 주입해야 한다는 점입니다. 해결책으로 labeled CoT만으로 지식 풀을 만들고, retrieval은 형태(morphology)로 1차 스크리닝한 뒤 CoT similarity로 재랭킹해 hard negatives를 제거합니다. 또한 warm-up에서 cross-modal 정렬(contrastive)로 임베딩을 안정화하고, MCAM의 multi-scale coordinate attention과 dual-decoder consistency로 retrieval로 인한 유용한 신호가 base decoder에도 흡수되게 했습니다.

- **Empirical Impact**: 실험은 MosMedData+, QaTa-COV19, BRISC 2025의 세 데이터셋에서 25%/50% labeled ratio 모두에 대해 수행됐고, CERS는 Dice/IoU에서 다수 지표와 데이터셋에서 SOTA를 일관되게 능가했습니다. 특히 저콘트라스트·아티팩트·경계 애매 케이스에서 병변 위치와 경계 보존이 더 좋게 나타났으며, pseudo-labeling 같은 confirmation bias 계열 대비 노이즈 영향을 완화하는 것으로 제시됩니다. 또한 retrieval 전략(이미지 기반 거친 검색 + CoT 정교화)과 MCAM 구성요소의 유효성에 대한 ablation 및 정성/정량 분석이 포함되어, 추론-주도 성능 향상의 재현 가능 근거를 제공합니다.



### MLLMs Get It Right, Then Get It Wrong: Tracing and Correcting Late-Layer Textual Bias (https://arxiv.org/abs/2606.17953)
Comments:
          Accepted at IJCAI 2026. 16 pages, 10 figures

- **Prior Approaches**: 기존 연구는 MLLM이 이미지와 텍스트가 충돌할 때 출력이 텍스트 쪽으로 쏠린다는 현상과 벤치마크를 주로 제시했지만, 모델 내부에서 무엇이 ‘버려지는지’에 대한 기계적 설명은 부족했습니다. 또한 DoLa, DeCo, VCD, OPERA 같은 추론 단계 보정법은 주로 고정적(혹은 전역적) 방식으로 교정하려 해, 유익한 late-layer 처리까지 함께 건드릴 위험이 있습니다.

- **Core Contribution**: 이 논문은 “late-layer textual override”라는 실패 메커니즘을 제안합니다. 모델은 중간 레이어에서 시각 기반 정답 예측을 형성하는 경우가 많지만, 마지막 출력까지 가는 과정에서 그 정보가 텍스트에 의해 덮여 사라지는 것으로 드러납니다. 더 나아가 전환(시각→텍스트 또는 텍스트→시각)의 방향이 성공/실패를 강하게 구분한다는 비대칭 시그니처를 발견합니다.

- **Technical Challenges**: 핵심 난제는 추론 시점에 ground-truth(시각 정답/텍스트 정답 라벨)가 없다는 점입니다. 저자들은 전환 레이어에서의 anchor confidence(전환 시점 정답 예측의 확신)와 prediction retention(최종 출력까지 유지되는 정도, retention 비율)을 조합해 ‘해로운 오버라이드가 일어나는지’를 탐지하고, Conflict-Aware Layer Reference Decoding(CALRD)로 전환 레이어 로짓을 최종 분포에 선택적으로 블렌딩해 복원합니다. 학습 없이(inference-time, training-free) 동작하면서도 필요한 경우에만 개입하도록 가중치를 설계한 점이 기술적 차별점입니다.

- **Empirical Impact**: 5개 MLLM(아키텍처/성능이 다양한 모델군)에 대해 Conflict-VQA, PhD-icc 같은 충돌 벤치마크에서 최대 9.4%p까지 절대 성능 향상을 보였고, POPE/CHAIR 등 비충돌 상황에서도 전반적 성능 저하를 크게 피했습니다. 특히 PhD-icc에서 LLaVA 계열과 InstructBLIP의 개선폭이 크게 나타나, ‘모델이 이미 알고 있었는데 마지막에 잃어버린 것’을 되찾는 접근임을 뒷받침합니다. latency/memory 오버헤드는 토큰당 수십 ms와 소량 추가 메모리 수준으로 보고되어 실사용 가능성도 함께 제시됩니다.



### Plug-and-Adapt: Multimodal Coreference Resolution at First Sight with a Pretrained Alignment Mod (https://arxiv.org/abs/2606.17950)
- **Prior Approaches**: 기존 MCR(multi-modal coreference resolution) 연구는 CIN(Coreference Image Narratives) 같은 벤치마크의 희소한 코어퍼런스 체인 주석이나 mouse tracking 같은 보조정보에 크게 의존해 학습·평가해 왔습니다. 약지도/준지도나 diffusion 기반 데이터 증강이 일부 문제를 완화했지만, 여전히 특정 데이터셋 학습이 필요해 즉시 적용성과 일반화가 제한된다는 한계가 남아 있습니다. 또 zero-shot을 노리는 VLM/CLIP 계열은 비전-언어 정렬은 잘하지만, 코어퍼런스의 문맥 의존성과 다중 멘션 간 유사도 추론까지는 충분히 못 따라가 성능 격차가 컸습니다.

- **Core Contribution**: 이 논문은 PA-MCR(plug-and-adapt for MCR)이라는 플러그-앤-어댑트 방식으로, (라벨이 부족한) 타깃 코어퍼런스 데이터로의 fine-tuning 없이도 이미지 내러레이션의 코어퍼런스를 해결하는 접근을 제안합니다. 핵심은 CLIP 기반 alignment 모델을 관계(relation) 단위 정렬로 재학습한 뒤, 멘션 표현을 “유사도 점수의 aggregation”으로 구성하고, 시각 cue와 카테고리 cue를 evidence theory로 결합해 추론을 안정화하는 것입니다. 즉, grounding에서 MCR로의 “격차”를 (1) relation-aware 정렬과 (2) 불확실성을 반영한 멘션 표현/멀티큐 통합으로 메우는 설계입니다.

- **Technical Challenges**: 첫째, grounding은 멘션-영역의 일대일 대응에 가깝지만 MCR은 프라임(예: pronoun)처럼 주변 관계에 좌우되는 해석과 멘션 간 상호 유사도 추론이 필요해, 단순 pairwise 정렬만으로는 부족합니다. 논문은 이를 위해 relation triplets 기반으로 정렬을 사전학습하고, top-aligned 편향을 줄이기 위해 모든 관련 매칭 점수를 활용하는 aggregation 기반 멘션 임베딩을 구성합니다. 둘째, 정렬 점수에 내재된 보정되지 않은 불확실성과 대량 후보(region)가 누적되며 노이즈가 증폭될 수 있어, region 외에 카테고리 정보를 추가하고 evidence theory로 cue 신뢰도를 정규화·융합해 “의심 스코어를 억제/신뢰 스코어를 강조”하도록 했습니다.

- **Empirical Impact**: CIN 벤치마크에서 PA-MCR은 전용(dedicated) SOTA 대비 CoNLL F1을 5.31%p, 인기 VLLM 대비 2.12%p 개선하며, 데이터셋 의존적 학습 없이도 경쟁력을 보였습니다. 또한 masked CIN과 새로 구성한 VCR-MCR에서 강건성 및 일반화 능력이 확인되어, 정렬 기반 접근이 특정 주석 구성에 덜 묶이는 잠재력을 시사합니다. 배포 관점에서도 거대 VLLM에 비해 훨씬 접근 가능한 파이프라인을 제시한다는 점에서 multimodal coreference의 실용화 방향에 의미가 큽니다.



### MoonSplat: Monocular Online Gaussian Splatting with Sim(3) Global Optimization (https://arxiv.org/abs/2606.17935)
Comments:
          SIGGRAPH 2026

- **Prior Approaches**: 기존 온라인 3DGS는 카메라 pose를 순차 PnP나 렌더링 기반 미분 최적으로 구하는 경우가 많아, 카메라 베이스라인이 충분치 않으면 추정이 쉽게 흔들린다. 또한 pose와 3D Gaussian 맵의 누적 오차를 loop closure로 안정적으로 보정하는 글로벌 최적화 장치가 약해, 시퀀스가 길어질수록 추적 실패 및 재구성 품질 저하가 누적된다. 메모리 측면에서는 가우시안 프리미티브가 계속 늘어 out-of-memory 문제와 함께, 디스크 앵커 같은 우회는 fine-grained 글로벌 최적화를 제한한다.

- **Core Contribution**: 이 논문은 온라인 voxelized 3DGS에 Sim(3) 글로벌 최적화를 결합해 pose 추적 신뢰성과 글로벌 loop closure 효율을 함께 끌어올린다. 특히 카메라 포즈뿐 아니라 voxelized 3D Gaussian 파라미터의 scale shift까지 함께 갱신해 멀티뷰 기하 드리프트 문제를 직접 다룬다. 아울러 voxelized 3DGS의 학습 병목을 줄이기 위해 color residual learning(CRL)을 제안해 수렴 속도와 렌더링 품질을 동시에 개선한다.

- **Technical Challenges**: 가장 큰 기술 난관은 (1) monocular 시퀀스에서 scale/pose가 함께 틀어지기 쉬운 환경에서, (2) 글로벌 보정을 하되 온라인 실시간성을 유지하는 최적화 설계다. 논문은 MASt3R 기반 포인트맵으로 초기 pose를 잡고, Sim(3)로 scale까지 포함한 전역 정렬을 Gauss-Newton/글로벌 최적화 형태로 수행해 loop closure에서의 누적 오차를 정리한다. 두 번째 난관인 voxelized 3DGS의 느린 색상 학습은 앵커별 base color를 keyframe pointmap에서 사전 계산하고, MLP는 residual만 예측하도록 바꿔 초기 학습 불안정과 느린 수렴을 줄였다.

- **Empirical Impact**: 실내·실외 다양한 데이터셋에서 카메라 pose 정확도와 렌더링 품질 모두 SOTA급 성능을 보이면서도 real-time 효율을 유지한다. 더 나아가 UAV 기반 active reconstruction 시스템을 실제로 구축·배포해, 현장 조건에서도 강건성과 범용성이 재현됨을 보여준다. 결과적으로 온라인 3DGS가 긴 시퀀스에서도 loop closure를 안정적으로 처리하며 실사용 파이프라인으로 확장될 수 있음을 실험적으로 뒷받침한다.



### Revisiting Structural Dependency in Autoregressive Multi-Task Table Recognition via Order-Independent Cell-Level Representations (https://arxiv.org/abs/2606.17874)
Comments:
          ICDAR 2026

- **Prior Approaches**: 테이블 인식은 표 구조(HTML 등) 예측과 함께 셀 위치(바운딩박스), 셀 내용 인식을 멀티태스크로 함께 수행하는 방식이 주류입니다. 많은 방법이 autoregressive(AR) 디코더로 구조를 시퀀스 생성한 뒤, 디코더의 hidden state를 셀 위치/내용 인식에 그대로 재사용합니다. 그런데 AR 디코더의 causal attention 마스크 때문에 셀 표현이 ‘생성 순서’에 종속되어, 셀 간 전역 일관성이 약해질 수 있습니다.

- **Core Contribution**: 이 논문은 구조 생성과 구조 집계를 분리해, 셀 표현이 생성 순서에 덜 의존하도록 만드는 structural refinement module을 제안합니다. HTML 디코더에서 나온 셀 특징을 non-causal self-attention으로 전역 맥락까지 반영해 order-independent한 표현으로 재정렬한 뒤, 이를 위치 예측과 셀 내용 인식에 함께 제공합니다. 또한 refinement로 셀 간 의존성을 미리 통합해, 내용 디코딩을 셀 단위로 병렬화할 수 있게 합니다.

- **Technical Challenges**: 핵심 난제는 AR 디코더가 만들어내는 lower-triangular(인과적) 의존 그래프를 downstream 태스크에 그대로 옮기면 양방향 셀 관계를 충분히 담기 어렵다는 점입니다. 이를 해결하기 위해 HTML 디코더 이후에 별도 셀-level Transformer 블록(셀 self-attention + 이미지 cross-attention)을 붙여, fully connected 형태의 셀 간 상호작용을 학습시킵니다. 이렇게 정제된 전역 셀 특징을 localization head와 cell content decoder에 동시에 공유해, 긴 셀 내용이나 병합 셀처럼 기하적 애매성이 큰 상황에서도 정합성을 높입니다.

- **Empirical Impact**: FinTabNet(FTN)과 PubTabNet(PTN) 계열 데이터에서 cell localization IoU가 약 3~5포인트 개선되고, end-to-end 인식 성능은 전반적으로 일관된 상승을 보였습니다. 특히 전역 refinement가 causal refiner보다 TEDS를 안정적으로 회복/개선하며, 셀 간 의존성 그래프를 fully connected로 바꾸는 설계 축의 중요성을 ablation으로 확인했습니다. 더 나아가 parallel cell decoding 덕분에 MuTabNet 대비 전체 추론 시간을 약 3배 줄이면서도 정확도는 유지하는 결과를 제시했습니다.



### A Quantitative Analysis of Multimodal Biomarkers in Alzheimer's Diseas (https://arxiv.org/abs/2606.17867)
Comments:
          Accepted to ICTS4eHealth 2026

- **Prior Approaches**: 기존 AD 멀티모달 연구는 예측 성능을 높이기 위해 복잡한 black-box 모델을 강화하는 경우가 많았고, 서로 다른 바이오마커가 얼마나 중복 정보를 갖는지 정량적으로 비교하는 작업은 상대적으로 부족했습니다. 일부 상호작용 분석이 있었지만, 모달리티 간 관계를 ‘인과 경로/생물학적 대응(구조-병리 연결)’ 관점에서 구조화해 해석하거나, 시점이 어긋난 진행 순서를 모델링하는 데는 한계가 있었습니다. 또한 PET 같은 고비용 측정의 중복성을 줄이기 위한 ‘최소·최대 정보’ 선택 기준이 명확하지 않았습니다.

- **Core Contribution**: 이 논문은 tau-PET, 구조 MRI, 인지 점수(MMSE, CDR), 유전(APOE ε4) 데이터를 ADNI의 789명에서 통합하고, 모달리티 간 정보 중복과 예측 의존성을 정량 분석해 바이오마커 선택의 근거를 제공합니다. tau와 인지의 연관을 atrophy(위축) 기반/비기반 성분으로 분해하고, SuStAIn 기반 궤적 추정으로 분자 병리→구조 변화→인지 저하의 지배적 진행 패턴을 재구성합니다. 이를 통해 “무엇을 더 측정해야 하는가/무엇을 줄일 수 있는가”를 데이터 관계로 설명하는 해석형 프레임을 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 모달리티 간 관계가 단순 선형이 아니고, 다중공선성과 비선형 의존이 함께 얽혀 있을 수 있다는 점이었습니다. 저자들은 NMI(비모수 nearest-neighbor)로 상호정보를 중복도(0~100%)로 비교하고, cross-validated Ridge regression으로 모달리티 간 explained variance를 안정적으로 추정해 예측 방향성을 평가했습니다. 또 tau-atrophy는 ROI별 상관과 PLS-SVD로 다변량 공분산 구조를 포착했고, tau-인지 매개효과는 ACME/ADE로 위축 경로 비중을 분해한 뒤, cross-sectional 한계를 보완하기 위해 규범모델링 기반 z-표준화와 modality-specific abnormality threshold로 SuStAIn 궤적을 구성했습니다.

- **Empirical Impact**: 결과적으로 APOE ε4는 다른 모달리티와 공유 정보가 매우 낮아(대략 0~6.9%) 상대적으로 독립적·보완적임이 확인됐고, 반대로 CDR-SB와 CDR-GLOBAL은 83% 이상 중복되어 다중 글로벌 인지척도 투입의 불필요성을 시사합니다. tau-PET과 MRI는 인지 예측에서 의미 있는 비중을 보였으며(예: MRI→CDR-SB, tau-PET→MMSE), tau-인지 연관 중 약 28%가 구조 위축을 통한 간접 경로(ACME)로 설명되었습니다. 나머지 직접 성분(ADE)도 유의해, 위축 외의 추가 메커니즘 가능성을 남기면서도 “진행 지연(cascade)” 관점에서 분자 병리가 먼저 나타나고 구조 변화가 뒤따르는 지배적 궤적을 데이터가 지지합니다.



### High-Fidelity 3D Geometric Reconstruction of Pelvic Organs from MRI: A Hybrid Deep Learning and Iterative Optimization Approach (https://arxiv.org/abs/2606.17836)
- **Prior Approaches**: 기존 연구는 MRI에서 골반 장기 3D를 얻을 때 주로 이미지 segmentation에 치우치거나, 생성된 3D 모델을 downstream 분석에 쓰는 쪽에 집중해 왔다. 그 결과 고충실도(고품질) geometry 복원이 여전히 labor-intensive하고 표준화도 부족했다. 또한 딥러닝 예측만으로는 국소 표면/메시 품질을 충분히 다듬기 어렵다는 한계가 반복됐다.

- **Core Contribution**: 이 논문은 bladder, uterus, rectum을 대상으로 하이브리드 deformable shape modeling을 제안해 reconstruction 품질의 격차를 줄인다. 핵심은 geometry-aware multi-level 딥러닝으로 topological consistency를 보존하고, 학습·추론 모두에서 iterative optimization으로 국소 표면과 메시에 대한 품질을 정교하게 끌어올리는 holistic synergy 구조다. 학습 단계에서는 최적화가 딥러닝의 감독 역할을 하고, 추론 단계에서는 딥러닝이 전역 형태를 빠르게 예측한 뒤 최적화가 표면을 refine한다.

- **Technical Challenges**: 주요 challenge는 전역 형태를 잘 맞추면서도 국소 표면 디테일과 topological consistency를 동시에 유지하는 것이다. 이를 위해 두 단계 amortized optimization training으로 global shape capture와 local surface refinement의 균형을 맞추고, geometry-aware multi-level 아키텍처로 장기 간 구조 일관성을 보존한다. 또한 iterative optimization이 학습에는 supervision 신호로, 추론에는 refinement 단계로 기여하도록 학습/추론 파이프라인을 함께 설계했다.

- **Empirical Impact**: 실험에서 제안 프레임워크는 기존 mainstream 딥러닝 기반 장기 reconstruction 대비 geometric fidelity가 뚜렷하게 우수했다. 구조별로 bladder, rectum, uterus의 3D는 Chamfer Distance가 더 낮고 Dice Similarity Coefficient이 더 높게 나타났다. 계산 효율을 유지하면서도 volumetric mesh quality가 더 좋았고, 환자 단위 평가에서는 minSICN과 minSIGE의 ‘10 worst elements’ 지표가 전통적 geometric post-processing 알고리즘보다 개선됐다.



### Human-in-the-Loop Atlas-Based 3D Asset Segmentation for Interactive Content Workflows (https://arxiv.org/abs/2606.17824)
- **Prior Approaches**: 기존 3D 세그멘테이션은 데이터셋/카테고리 의존적이거나, 단순 기하 프리미티브 가정으로 복잡한 표면에서 성능이 떨어지는 한계가 있습니다. 2D foundation model(SAM 2 등)을 3D로 옮기는 zero-shot/few-shot 접근도 자동 의미 라벨링 중심이라, 사용자나 애플리케이션이 원하는 “의미 있는 경계”를 직접 정의·수정하기 어렵습니다. 또한 멀티뷰를 atlas에 투영하는 방식은 가능성을 보였지만, 커버리지를 고려한 뷰 최적화와 사람의 반복 보정(refinement) 메커니즘이 부족했습니다.

- **Core Contribution**: 이 논문은 인간이 의미(세그먼트 기준)를 통제하고, AI가 수정 비용을 줄이는 human-in-the-loop 파이프라인을 제안합니다. 3D 모델에서 최소 뷰를 뽑아(coverage 중심) 렌더링한 뒤, SAM 2와 Label Studio 기반의 대화형 마스크 수정으로 2D 세그먼트를 만든 다음, UV 파라미터화에 back-projection해 통합된 segmented 2D atlas를 생성합니다. 그 결과 atlas 단위로 머티리얼 할당, 스타일 전이, 의미 라벨링 같은 텍스처-스페이스 후속 작업을 더 손쉽게 수행할 수 있습니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 너무 많은 뷰 대신 “표면을 빠짐없이” 커버할 뷰를 고르는 것과 (2) 2D 마스크의 불완전/과분할을 사람의 반복 수정으로 수렴시키는 것입니다. 논문은 뷰 선택을 robotics 관점의 viewpoint planning처럼 set cover 문제로 정식화하고 greedy 전략으로 카메라 후보 중 커버 효율이 큰 뷰를 반복 선택해 최소 뷰를 확보합니다. 이어 Label Studio에서 SAM 2 마스크를 최소 프롬프트로 생성·수정하고, 여러 뷰의 결과를 UV로 투영·병합해 일관된 atlas를 만듭니다.

- **Empirical Impact**: 문화유산 객체 8종 데모 기반 평가에서 전체 파이프라인이 다양한 형상(약한 경계, 얇은 구조, 캐비티, 미세 디테일)에서도 “사용 가능한” segmented atlas를 생성했습니다. 기록된 주석 시간은(로그된 7개) 대체로 15~35분이며 평균 약 21분 수준으로, 큰 연속 영역은 SAM 2의 이점이 크지만 미세 구조·좁은 부착물·경계 대비가 낮은 경우엔 수작업 보정이 늘어났습니다. 특히 David 대두의 눈/귀, 날개가 본체와 합쳐지는 천사 조각, 얇은 스트랩/캐비티에서의 경계 수정 필요가 반복 실패 모드로 드러나, 향후 자동화에서 어떤 개선이 우선인지 방향을 제공합니다.



### Million-scale multimodal pollen microscopy with expert-guided foundation models (https://arxiv.org/abs/2606.17809)
Comments:
          31 pages, 5 main figures, supplementary information included. Submitted to Scientific Reports

- **Prior Approaches**: 기존 자동 화분 분석은 image-only 분류·검출에 치우쳐 있었고, 데이터가 taxon 범위·샘플 수·모달리티·원천 다양성 중 한 가지 이상에서 제한되어 교차 데이터셋/교차 지역 일반화가 어려웠습니다. bright-field에서도 Mixed 환경 시료로 확장하려면 곡립(곡물) 단위 라벨링 부담이 커서, 순수종 기준 슬라이드를 활용한 weak supervision 전략이 대안으로 제시돼 왔지만 전체 whole-slide 스케일에서 고정밀 레퍼런스 구축으로 이어지는 체계는 부족했습니다.

- **Core Contribution**: Pollen AI Atlas는 4개 지역·4개 스캐너 설정·31개 식물 군(31 botanical families)에서 온 bright-field whole-slide를 기반으로 46개 taxon 라벨의 백만 규모 pollen grain을 구축한 multimodal 자원입니다. 한 장당 사람의 exemplar 1개로 token-level 마이닝을 시작해 99.6% proposal precision의 고정밀 곡립을 만들고, 5개 open-weight vision-language model로 grain-level 형태 캡션을 생성하되 palynological anchor(전문가 검증 기준용어)로 제어해 구조화된 자연어 형태 기록을 제공합니다.

- **Technical Challenges**: 주요 기술 과제는 (1) slide 전역에서 객체를 찾는 과정의 precision을 유지하면서 recall을 과도하게 희생하지 않는 것, (2) 생성형 VLM 캡션에서 taxon/name leakage나 수치·측정 정보 누출을 억제하고 형태 용어를 표준화하는 것, (3) 스캐너·염색·제작 차이가 큰 상황에서 시각 임베딩보다 형태 텍스트가 견고하게 retrieval을 견디는지를 검증하는 것입니다. 논문은 attention reranking·prototype refinement·NMS·classifier gating으로 후처리 정밀도를 확보하고, Gemma4가 length control/누출 억제/텍스트 retrieval 성능에서 가장 안정적임을 진단 후 downstream 평가의 기본 캡션 세트로 채택합니다.

- **Empirical Impact**: TS2 전문가 검증 영역에서 고정밀 탐지는 99.6% precision(샘플 전역 false-positive율 0.5% 미만)으로 재현되며, 곡립 캡션 기반 retrieval은 CROSS-REG(질의 origin 제외)에서도 텍스트가 mAP@20 0.811로 유지되는 반면 이미지 유사도는 mAP@20 0.262로 크게 하락합니다. 생성 캡션을 활용한 분류는 frozen visual features 기반 선형 프로브에서 Top-1 88.16%로 확인되지만 캡션이 분류 성능을 크게 바꾸기보다는, 형태 중심 텍스트 레이어가 도메인 변화(스캐너/제작/지역)에서 품질관리를 위한 더 강한 신호임을 보여줍니다. 데이터·가공 코드·캡션·split을 공개해 pollen recognition, cross-regional domain adaptation, domain-specific multimodal microscopy 학습의 벤치마크로 활용될 전망입니다.



### MaineCoon: Pursuing A Real-Time Audio-Visual Social World Mod (https://arxiv.org/abs/2606.17800)
Comments:
          32 pages, 13 figures, 3 tables

- **Prior Approaches**: 기존 world model 연구는 물리 환경이나 게임 탐험처럼 비교적 단순한 공간 동역학을 시뮬레이션하는 데는 강했지만, 인간 중심의 사회적 상호작용(대화, 반응, 의도 변화)과는 거리가 멀었습니다. 또한 오디오-비주얼 생성에서 실시간성/저지연 스트리밍을 전제로 한 설계는 상대적으로 제한적이었고, 길게 이어지는 생성에서 drift(드리프트) 관리도 취약했습니다. 그 결과 소셜 플랫폼을 염두에 둔 ‘social world model’의 자리와 요구사항이 충분히 정의·검증되지 못했습니다.

- **Core Contribution**: 이 논문은 social world models의 문제 위치를 처음으로 정리하고, 이를 위한 프로토타입으로 MaineCoon을 제안합니다. MaineCoon은 단일 GPU에서 실시간 streaming generation을 지원하는 최초의 실시간 audio-visual autoregressive 모델로, 22B 파라미터 규모와 초 단위 상호작용(서브초)을 강조합니다. 또한 social-interactive 애플리케이션에 맞춰 최적화된 점을 핵심 차별점으로 내세웁니다.

- **Technical Challenges**: social interactive 환경에서는 (1) 오디오와 비주얼의 정합을 실시간으로 유지하고 (2) 긴 생성 동안 상태가 흐트러지지 않으며 (3) 대규모 모델을 안정적으로 학습/추론해야 하는 문제가 동시에 발생합니다. 논문은 효율·안정 학습을 위해 self-resampling, cross-modal representation alignment, domain-aware preference optimization, reinforced online-policy distillation(ROPD)를 도입해 학습 속도와 성능을 끌어올립니다. 더불어 agentic streaming inference 프레임워크를 설계해 agentic cache management와 prompt planing으로 수천 초 규모 생성에서도 드리프트를 완화합니다.

- **Empirical Impact**: 저지연·고품질·장기 지평을 동시에 노리는 관점에서 MaineCoon은 최대 47.5 FPS의 프레임레이트를 단일 GPU에서 기록하며 실시간 스트리밍 생성의 실효성을 보여줍니다. 또한 훈련 효율 개선과 실시간 추론 성능 최적화가 함께 달성되었다고 보고하며, high-quality audio-visual autoregressive 모델의 새로운 SOTA 벤치마크를 제시합니다. 소셜 인터랙션 중심의 AI-native 플랫폼으로 패러다임 전환을 촉진할 가능성을 제안한다는 점에서 의미가 큽니다.



### LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams (https://arxiv.org/abs/2606.17798)
- **Prior Approaches**: 기존 Video-LLM-online 연구들은 EOS(End-Of-Sequence) 토큰을 예측해 ‘침묵 구간’을 학습하는 방식이 주류였습니다. 다만 침묵 프레임이 압도적으로 많아 데이터 불균형을 만들고, 시각 증거와 무의미한 EOS의 매핑 충돌로 video-language 정렬이 약해지며, 인접 프레임에서 상반된 타깃이 나와 학습 안정성도 떨어집니다.

- **Core Contribution**: 이 논문은 장시간 스트림에서 항상-on으로 동작하면서도 “언제 응답할지”를 스스로 결정하는 프로액티브 라이브 스트리밍 어시스턴트 LiveStarPro를 제안합니다. 핵심은 침묵을 생성 타깃으로 두지 않고, 모델의 confidence로 ‘응답 타이밍’을 검증하는 방식으로 구조를 바꿔 실시간성과 정렬 품질을 함께 노린다는 점입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 매 프레임마다 생성-비교를 하면 지연이 커지는 문제, (2) 증분으로 누적되는 visual context에 맞춰 학습이 online 정렬을 제대로 따라가야 하는 문제, (3) 무한에 가까운 스트림에서 망각 없이 효율적으로 장기 기억을 찾아오는 문제입니다. 이를 위해 Streaming Verification Decoding(SVeD)로 단일 forward pass 기반 perplexity 검증으로 ‘watching↔speaking’ 게이트를 만들고, Streaming Causal Attention Masks(SCAM)로 이벤트 단위에서 언어 복사 누수를 막으며, Tree-Structured Hierarchical Memory(TSHM)로 Peak-End 압축과 Recursive Event Tree 기반 장기 검색을 결합합니다.

- **Empirical Impact**: 실험 결과 LiveStarPro는 기존 온라인 Video-LLM 대비 semantic correctness 28.9% 향상, timing error 18.2% 감소를 보이며, streaming key-value cache까지 활용하면 동일 모델에서 1.58x 추론 속도 개선을 달성했습니다. 또한 시간 단위(시간 스케일) 장기 기억을 평가할 수 있는 OmniStarPro 벤치마크를 통해, 실제 온라인 조건에서 장기 맥락 유지와 응답 타이밍 정확성이 동시에 개선됨을 보여줍니다.



### BrainWorld: A Structural-Prior-Conditioned Generative Model for Whole-Brain 4D fMRI Dynamics (https://arxiv.org/abs/2606.17742)
- **Prior Approaches**: 기존 fMRI foundation 모델은 주로 masked reconstruction, predictive representation learning, contrastive/LLM-alignment 같은 proxy 목적에 초점을 맞춰 표현을 학습하고, 다운스트림 예측으로 전이하는 방식이 중심이었습니다. 생성형 접근도 ROI 수준이거나 장기 예측·voxel 수준 4D 생성을 조건부로 수행하기엔 구조적 조건(sMRI) 통합이 제한적이었습니다. 또한 sMRI와 fMRI를 병렬/보완 입력으로 취급해, 주체별 해부학 맥락을 생성 과정에 비대칭으로 주입하는 설계는 부족했습니다.

- **Core Contribution**: 이 논문은 주체의 sMRI를 구조적 prior로 삼아 whole-brain 4D fMRI의 조건부 장기 예측을 직접 생성하는 BrainWorld를 제안합니다. 핵심은 sMRI를 단순 멀티모달 결합이 아니라 denoising 과정 전반에 주입해, 해부학 맥락이 미래 기능 동역학 형성을 조절하도록 만든 structural-prior-conditioned generative framework입니다. 또한 latent diffusion에서 얻는 중간 표현까지 활용해 생성 품질과 표현 학습을 함께 달성합니다.

- **Technical Challenges**: voxel-level whole-brain 4D를 그대로 확률모형하기엔 계산량이 커서, VAE로 fMRI를 연속 잠재공간으로 압축한 뒤 latent diffusion과 Diffusion Transformer로 미래 잠재를 denoising하는 2단계 설계를 채택했습니다. 장기 롤아웃에서는 오차 누적과 지연이 문제가 되는데, 40-frame 블록 단위의 오토리그레시브 unrolling로 효율을 확보하고, 생성된 4D에서 Schaefer-100 ROI 기반 functional context(FC)를 업데이트하며 안정성을 높였습니다. sMRI 통합의 실질적 차이를 만들기 위해, cross-attention(토큰 레벨)과 adaptive normalization/residual gating(글로벌 레벨) 경로로 DiT backbone 전반에 조건을 주입했습니다.

- **Empirical Impact**: 22개 데이터셋·다양한 뇌 상태에서 400프레임까지 안정적인 4D 궤적 생성과 다운스트림 성능 향상이 관찰되었습니다. 특히 generated-example augmentation에서 HCP(성별 분류)와 SALD(연령 회귀) 모두에서 성능이 개선되며, 가장 강한 baseline 대비 부가 이득이 더 크게 나타났습니다. 또한 fine-tuning과 linear probing 모두에서 sMRI를 denoising 경로로 주입한 variant(F+S-prior)가 단순 concat 기반 멀티모달 결합보다 일관되게 우수해, 구조-기능 결합을 생성 내부에 녹이는 전략의 의미를 실증적으로 보여줍니다.



### ActWorld: From Explorable to Interactive World Model via Action-Aware Memory (https://arxiv.org/abs/2606.17730)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 interactive world model은 주로 navigation(이동·시점 제어)에 초점이 맞춰져, 장면 속 물체를 집거나 문을 여는 등 객체 수준 상호작용은 누락되거나 게임 환경/오프라인 영상 생성으로 제한되는 경우가 많았다. 또한 물리적 상호작용이 포함되더라도 언어→영상의 비실시간 설정이거나 복잡한 액션 연쇄에서 품질이 급격히 떨어져, 에이전트가 실제로 ‘행동 가능’한 세계를 만들기 어려웠다. 그 결과 시각적으로는 탐색되지만, 다음 단계의 상호작용을 좌우하는 실제 이벤트를 재현하기 어렵다는 한계가 누적됐다.

- **Core Contribution**: ActWorld는 navigation 중심 생성기를 확장해, chunk-autoregressive 프레임워크 안에서 mid-rollout 객체 상호작용을 함께 지원하는 interactive world model을 제안한다. 핵심 주장은 navigation–interaction gap이 데이터 부족과 메모리 압축의 부적합(행동 잊기)에서 나온다는 점이며, 이를 동시에 겨냥해 단일 실시간 모델에서 이동 시점 제어와 풍부한 객체 조작을 함께 달성한다. 또한 상호작용 이벤트에 정렬된 계층형 메모리와 장면 내 상호작용을 촘촘히 라벨링한 데이터 파이프라인을 결합했다.

- **Technical Challenges**: 첫째, 객체 상호작용의 원인-결과를 담는 데이터(정확하고 조밀한 라벨)가 부족해 chunk 단위에서 상호작용 단계가 모호해질 수 있다. ActWorld는 100K 상호작용 비디오를 구축하고 chunk마다 dense caption 및 interaction-phase 라벨을 제공(추론 기반 체인 오브 쏘트로 라벨 생성)해, 생성기가 ‘어느 상호작용 단계’를 다루는지 조건을 명확히 했다. 둘째, 기존 모델은 과거 히스토리를 시간 recency로 압축해 접촉·조작을 결정하는 프레임이 멀어지면 소실되므로 action-forgetting이 발생하는데, ActWorld는 상호작용 중요도 기반 재라우팅과 event/object 토큰을 유지하는 persistent memory bank로 이를 완화했다.

- **Empirical Impact**: 실험에서 ActWorld는 navigation-only baseline 대비 상호작용 충실도(interaction fidelity)를 크게 개선하면서도 시점/로케이팅 controllability를 유지하는 것으로 보고됐다. 이를 장기 시퀀스에서 이동과 객체 조작이 교차되는 새로운 실험용 벤치마크 I-Bench로 검증했으며, 정량 평가는 시각 품질·시간 일관성, instruction following, 카메라 기하 제어의 다축으로 이뤄졌다. 결과적으로 ‘그럴듯하게 보이는 월드’에서 ‘실제로 다음 행동을 준비하는 월드’로 interactive world model의 목표를 더 가깝게 옮겼다는 점에서 의미가 있다.



### GSPan: A Continuous Gaussian Primitive Representation for Arbitrary-Scale Pansharpening (https://arxiv.org/abs/2606.17722)
- **Prior Approaches**: 기존 pansharpening 연구는 CS, MRA, VO 같은 물리·통계 기반 방법과 CNN/GAN/Transformer/확산·state-space 모델 같은 딥러닝 방법으로 발전해 왔습니다. 그러나 많은 딥러닝 접근이 결국 고정된 출력 격자(고정 스케일)에서 픽셀을 직접 예측하는 방식이라, 임의 스케일 렌더링이나 대규모 장면 적용에 제약이 큽니다. INR(implicit neural representation)은 연속 표현이 가능하지만, 점(좌표) 단위 MLP 질의로 인해 해상도 증가 시 추론 비용이 커지는 문제가 있습니다.

- **Core Contribution**: 이 논문은 pansharpening에 Gaussian Splatting(2D Gaussian Splatting, GS)을 도입한 GSPan을 제안합니다. 픽셀을 바로 예측하는 대신, 대역별 잔차 디테일을 연속적이고 학습 가능한 2D Gaussian primitive들의 집합으로 표현해 잔차장을 렌더링하고, 이를 업샘플된 MS에 더해 HRMS를 생성합니다. 또한 이 연속 표현 덕분에 네트워크 재학습 없이 arbitrary-scale(임의 스케일) 격자에서 렌더링할 수 있고, 이를 SDAI(Scale-Decoupled Asymmetric Inference)로 연결해 대규모 장면 추론을 효율화합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) PAN의 고주파 구조와 MS의 스펙트럴 정보를 primitive 속성(중심·스케일·방향·잔차 계수·스펙트럴 계수 벡터)으로 정교하게 결합하고, (2) 임의 해상도에서도 연속 렌더링이 가능하도록 표현을 설계하는 데 있습니다. 이를 위해 DSHI(Dual-Stream Hierarchical Interaction)로 공간·스펙트럴 임베딩을 계층적으로 정제하고, SSIA(Spatial-Spectral Interactive Attention)로 두 스트림 간 쿼리-레퍼런스 교차 attention을 수행해 Gaussian 속성을 안정적으로 추정합니다. 아울러 SDAI에서는 primitive 속성 추정을 낮은 해상도로 수행한 뒤, 연속 Gaussian 필드를 목표 해상도에서 렌더링해 계산량을 줄이되 일부 공간 정밀도 손실을 감수하는 전략을 사용합니다.

- **Empirical Impact**: QuickBird, GaoFen-2, WorldView-3 및 WorldView-3-4K 데이터셋에서 GSPan은 기존 대비 state-of-the-art 또는 경쟁력 있는 fusion 성능을 보였습니다. 특히 RR(축소 해상도) 평가에서 SAM/ERGAS/Q2n 등 지표가 전반적으로 개선되며, FR(실세계 대형 장면)에서는 reference-free 지표(HQNR, Dλ, Ds)로도 품질을 입증합니다. SDAI는 대형 장면에서 추론을 유의미하게 가속하면서도 fusion quality와의 균형이 좋은 것으로 보고되어, 실사용 관점의 확장성에 의미가 있습니다.



### Heterogeneous SAR-optical fusion for near-real-time land use and land cover mapping under cloud contamination: A novel framework and global benchmark datas (https://arxiv.org/abs/2606.17713)
- **Prior Approaches**: 기존 SAR-광학 융합 연구는 대체로 신뢰 가능한 광학 관측을 전제로 하거나, 복원(reconstruction) 단계를 먼저 두는 방식이 많아 구름 오염이 만드는 의미(semantic) 불확실성을 충분히 다루지 못했습니다. 그 결과 중간 표현이 흔들리며 목표 시점 LULC(토지이용/피복) 예측의 신뢰도가 저하될 수 있습니다. 또한 광학과 SAR의 공간-채널 간 고차 상호작용을 효과적으로 집계하는 데에도 한계가 있었습니다.

- **Core Contribution**: 이 논문은 구름이 낀 Sentinel-2 광학 영상과 시계열로 인접한 Sentinel-1 SAR를 입력으로 받아 LULC 지도를 직접 예측하는 end-to-end 이질적 SAR-광학 융합 프레임워크 CloudLULC-Net을 제안합니다. 핵심은 광학 신뢰도 모듈레이션으로 불안정한 광학 반응을 억제하고, LULC 지향 잠재공간에서 fused feature를 정리하는 semantic mapping transformer를 도입한 점입니다. 여기에 semantic anchor-guided optimization으로 중간 의미 표현의 일관성을 강화합니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 구름/그림자가 광학 신호를 의미적으로 왜곡해, 융합 시 잘못된 의미 정보를 주입할 위험이 커진다는 점입니다. 논문은 optical reliability modulation로 신뢰도 낮은 광학 특징의 영향을 억제하고, heterogeneous information adaptive aggregation로 광학- SAR 표현 간 고차 공간-채널 상호작용을 모델링해 융합 품질을 끌어올립니다. 마지막으로 unified semantic mapping transformer와 semantic anchor-guided optimization을 통해 중간 의미 표현의 정합성을 개선하도록 설계했습니다.

- **Empirical Impact**: 저자들은 40,223개 규모의 SAR-광학-라벨 triplet과 픽셀 단위 LULC 어노테이션을 포함한 CloudLULC-Set을 구축해 검증 기반을 마련했습니다. 실험에서 CloudLULC-Net은 OA 86.60%, F1-score 83.29%, mIoU 73.51%를 달성하며 대표적인 heterogeneous reconstruction-first 및 end-to-end SAR-광학 매핑 방법들을 능가했습니다. 또한 글로벌 LULC 제품과의 비교 및 구름 커버리지별 분석에서 강건성/실용성을 보여, 구름이 잦은 환경에서 target-date LULC mapping의 활용 가능성을 높였다는 점에서 의미가 큽니다.



### Structured Adversarial Camouflage via Voronoi Diagrams (https://arxiv.org/abs/2606.17711)
- **Prior Approaches**: 기존의 적대적 카모플라주 공격은 픽셀 단위 또는 임의 텍스처를 최적화하는 방식이 많아 계산비용이 크고, 시각적으로 튀는 패턴이 되기 쉽습니다. 또한 인쇄/직물 구현을 고려한 색 선택 제약이나 구조-재현성의 일관성을 충분히 다루지 못해 실제 전이(transfer) 검증이 제한되는 경우가 있습니다.

- **Core Contribution**: 이 논문은 Voronoi 다이어그램 기반의 ‘adversarial Voronoi camouflage’를 제안합니다. 핵심은 픽셀 전체를 직접 학습하지 않고, 고정된 printable color palette 안에서 seed-point(씨앗 점) 위치만 최적화해 구조화된 splinter-like 패턴을 만들며, 추가 정규화 없이도 탐지기 신뢰도를 떨어뜨리도록 한 점입니다.

- **Technical Challenges**: seed-point를 조정해 생성된 구조가 실제 탐지 성능을 얼마나 공격적으로 만들지(그리고 사람이 보기엔 자연스럽게 보일지)라는 문제가 기술 난제로 남습니다. 논문은 거리 기반 soft assignment(temperature-scaled softmin)를 미분 가능하게 구성해 전체 Voronoi 패턴 생성-탐지기 통과-손실 최소화가 end-to-end로 되도록 했고, 3DPeople의 segmentation mask를 사용해 패치가 사람 ‘의복 영역’에 정렬되게 학습합니다.

- **Empirical Impact**: 실험 결과, 단순 bbox 내부에 패치를 얹어 학습한 naive 배치 방식은 상대적으로 효과가 낮았지만, 의복 단위(3DPeople mask) 적용에서는 person detection에서 COCO 스타일 AP@[.5:.95]가 유의미하게 떨어졌습니다. 또한 배경이 바뀌거나 YOLOv9/10/11/12로 detector family가 달라져도 공격 특성이 전이되며, palette를 바꿔 repaint하면 효과가 크게 무력화돼 ‘structure-palette coupling’이 관찰됩니다. 다만 물리 구현(인쇄 적합성, 색 캘리브레이션, 변형, 인간 요인)은 향후 과제로 남기며, 전반적으로 실시간 탐지 성능 저하와 시각적 그럴듯함 사이의 트레이드오프를 보여줍니다.



### Vision-language models for chest radiography do not always need the imag (https://arxiv.org/abs/2606.17710)
- **Prior Approaches**: 기존 의료 VLM 평가는 주로 정확도(accuracy)에 의존하는데, 이는 정답이 영상에 인과적으로 의존하는지 구분하지 못한다. 학습 데이터의 finding-name prior나 동반(co-occurrence) 통계로도 충분히 그럴듯한 yes-or-no 답이 가능하고, saliency/attention 같은 사후 해석도 인과성을 보장하지 못한다.

- **Core Contribution**: 이 논문은 영상 조작을 통해 “모델이 실제로 이미지를 읽는지”를 점검하는 causal audit(인과 감사) 프레임을 제안한다. 동일 라벨의 다른 환자 이미지 교체(swap), 방사선사가 표시한 목표 영역 occlusion(target mask), 무관 영역 occlusion(irrelevant mask)을 함께 적용하고 세 가지 행동 지표(CGR, UAR, IS)로 영상 의존성을 분해해 평가한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 정확도만으로는 보이지 않는 ‘언어 단서 기반 정답’을 영상 의존성과 분리해 측정하는 것이다. 저자들은 MS-CXR phrase-grounding 박스와 임상 라벨을 조합해 2,575개 yes-or-no 프로브를 만들고, 9개 시스템(텍스트 전용/비전 전용 프로브 포함)에 동일한 네 조건을 적용한 뒤 dataset·해상도·프롬프트 문구까지 바꿔도 카테고리가 유지되는지 교차검증한다.

- **Empirical Impact**: 결과적으로 9개 중 3개는 CGR=0으로 ‘이미지 미사용’ 범주에 들어가고, 1개는 영상 사용이 불안정하며, 나머지 5개도 영상 정보를 선택적으로만 사용(발견 일부에 한정)하는 것으로 나타난다. 더 나아가 정확도만 보면 멀티모달이 우세해 보여도, 텍스트 전용 모델이 상위 멀티모달에 근접하거나 통계적으로 비슷한 사례가 있어 “정확도=영상 사용” 주장은 성립하지 않는다; 임상 배포 게이트는 정확도가 아니라 grounding audit처럼 인과적 점검으로 해야 한다는 결론을 내린다.



### SegTME-UNI2: A Foundation Model-Based Framework for Generalisable Multiclass Cell Segmentation and LLM-Driven Tumour Microenvironment Characterisation in Histopathology (https://arxiv.org/abs/2606.17702)
- **Prior Approaches**: H&E 병리영상에서 TME를 보려면 보통 (1) 세만틱 분할(픽셀을 세포 범주로 라벨링)과 (2) 핵 instance 분리(서로 맞닿은 핵을 개별 객체로 분리)가 동시에 필요합니다. 기존 instance 분할 모델(HoVer-Net 등)은 PanNuke 같은 소규모 nucleus-level polygon 주석에 의존해 확장성이 떨어지고, 세만틱 분할(UperNet 등)은 핵이 붙어있을 때 분리가 어렵습니다. 또한 TCGA 같은 대규모 저장소는 픽셀 단위 주석이 없어 임상 보고 흐름에 바로 쓰기 힘든 ‘해석 가능한 산출물’까지 이어지는 경로가 막혀 있었습니다.

- **Core Contribution**: 이 논문은 SegTME-UNI2로, 세만틱 분할·HV 회귀 기반 watershed용 핵 분리·구조화된 TME 피처 생성·BioNeMo GPT를 통한 임상 서술까지 한 번에 묶는 통합 프레임워크를 제안합니다. 핵심은 UNI2-H pathology foundation model(UNI2-H, ViT-Giant)을 공유 인코더로 쓰고 UperNet 디코더를 두 갈래로 붙인 UNI2-UPERHOVER(세만틱 6-class + horizontal-vertical gradient 회귀)입니다. 여기에 pseudo-label 기반 3-stage progressive curriculum과 20+ per-patch TME feature→BioNeMo narrative 변환 파이프라인을 결합해 “대규모·해석가능·분리까지”를 동시에 달성하려고 합니다.

- **Technical Challenges**: 가장 큰 난관은 TCGA-UT처럼 픽셀 주석이 없는 대규모 데이터에서 instance 수준의 핵 분리를 학습시키는 것입니다. 이를 위해 논문은 세만틱 마스크만으로 동적으로 HV 타깃을 합성해 instance polygon 주석 없이도 watershed 분리를 가능하게 하고, pseudo-label의 질을 단계적으로 올리기 위해 PanNuke→TCGA-UT scale-0→TCGA-UT 전 해상도(총 1,608,060패치)로 확장하는 3단계 커리큘럼을 설계합니다. 또한 해상도 스케일이 0.25~1.0 μm/pixel로 크게 바뀌는 상황에서 다중 스케일 문맥을 담기 위해 UperNet의 PPM+FPN 구조를 채택합니다.

- **Empirical Impact**: PanNuke와 TCGA-UT 분할에서의 예비 검증은 프레임워크의 실행 가능성과 내부 일관성을 보여주며, 특히 TCGA-UT처럼 넓은 분포에서 세만틱 문맥과 핵 분리를 함께 얻을 수 있음을 시사합니다. 생성된 TME 피처(조성·임상 비율·공간 상호작용·공간 엔트로피 등 20+ 지표)는 JSON으로 정형화되어 BioNeMo GPT의 fine-tuning 입력으로 들어가 임상적으로 읽히는 “패치 내 TME 내러티브”를 목표로 합니다. 더 나아가 pseudo-labelled TCGA-UT 데이터와 UNI2-UPERHOVER 체크포인트를 공개해, 대규모 spatial biology/병리 정량화 연구에 직접 활용될 수 있는 기반을 제공합니다.



### See First, Answer Later: Visual Evidence Pre-Alignment via Sufficiency-Driven RL (https://arxiv.org/abs/2606.17678)
- **Prior Approaches**: 기존 MLLM 학습은 대규모 이미지-캡션 기반 pretraining으로 ‘거친’ 비전-언어 정렬을 만든 뒤, SFT와 RL로 답변 추종과 복잡한 추론을 강화하는 2단계 파이프라인을 주로 사용합니다. 하지만 캡션은 짧고 성긴(supervision이 거칠고) 편향이 있어 세밀한 속성·관계·덜 두드러진 영역에 대한 미세 grounding을 충분히 학습시키지 못합니다. 결과적으로 추론 시 언어 priors에 기대어 이미지 근거가 약해지고, 중요한 시각 디테일 누락이나 환각이 생길 수 있습니다.

- **Core Contribution**: 이 논문은 pretraining과 post-training 사이에 중간 단계 Visual Evidence Pre-Alignment(VEPA)를 추가해, 추론 전에 ‘질문-조건 시각 증거(visual evidence)’를 먼저 생성하도록 정렬합니다. 핵심 아이디어는 답변 최적화가 아니라, evidence 생성 정책 P(e|v,q)를 강화해 evidence가 주어진 질문을 풀 수 있을 만큼 충분(sufficiency)하면서도 이미지에 의존(visual dependence)하도록 만드는 것입니다. 또한 표준 post-training과 보완적으로 작동하며, 추가 task별 어노테이션 없이도 시각 grounding을 높이는 방향을 제시합니다.

- **Technical Challenges**: 문제는 세밀한 evidence 토큰 라벨이 대규모로 없다는 점인데, VEPA는 이를 RL로 우회해 증거 생성 전체 시퀀스를 보상으로 학습합니다. 이를 위해 정답 유출(answer leakage)과 반복/퇴화된 evidence를 피하도록 보상을 정교하게 설계하며, sufficiency 신호는 이미지 없이 evidence만 보는 ‘frozen blind reader’가 주어진 질문을 풀 수 있는지로 간접 측정합니다. 안정적인 장문 RL 학습을 위해 sufficiency-driven Group Relative Policy Optimization(GRPO)로 그룹 상대 advantage를 사용해 업데이트 변동성을 줄입니다.

- **Empirical Impact**: 실험에서는 VEPA를 끼운 모델이 다양한 비전 요구 벤치마크에서 일관되게 성능을 개선하며, in-domain 정확도를 크게 해치지 않으면서 domain shift(예: 차트·텍스트 기반 시각 과제)에서 더 큰 이득을 보였습니다. 또한 POPE/MMStar에서 이미지 없이 evidence만 보고 답을 시도하는 blind reader의 정확도가 향상되고, 생성 evidence 길이는 오히려 짧아져 verbosity가 아닌 selectivity·충분성 강화가 원인임을 뒷받침합니다. 정량/정성 분석을 종합하면 VEPA 효과는 추가 task 학습이 아니라 transferable한 시각 grounding 강화에서 온다고 결론내립니다.



### Do We Really Need Diffusion? A Fast U-Net for Paired Medical Image Translation (https://arxiv.org/abs/2606.17675)
- **Prior Approaches**: SFF(신호 지방분율)는 주로 multi-echo Dixon 기반 PDFF를 통해 정량되지만, 전용 스캔이 필요해 일상 임상·역학 연구에서 접근성이 낮았다. 대안인 two-point Dixon도 사이트마다 일관된 획득이 어려워 대규모 paired 데이터 구축이 제한되었고, 기존 T2w 기반 추정 연구는 자동화·정량 기준선(SFF/PDFF) 검증 측면에서 스케일 문제가 반복됐다.

- **Core Contribution**: 이 논문은 널리 쓰이는 T2-weighted(T2w) MRI만으로도 image-to-image translation으로 SFF를 추정할 수 있는지 체계적으로 검증한다. 4-level U-Net(경량 회귀형)과 state-of-the-art Denoising Diffusion Probabilistic Model(DDPM, 확산형)을 NAKO의 2D paired 데이터(총 22,910 테스트 슬라이스)에서 비교하고, downsteam 과제로 척추기립근 4구획의 근육 SFF 정량까지 평가한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘정량 지도(SFF)’가 ‘한 장의 T2w 해부학 정보’로부터 얼마나 정확히 역추정되는지, 그리고 확산 모델의 반복적 샘플링 비용이 품질 향상으로 연결되는지다. 저자들은 픽셀 정밀도(전역/바디 마스크)와 근육 단위의 평균 SFF(상관·MAE·bias), 그리고 추론 시간까지 함께 평가해 이러한 trade-off를 실측으로 분해했다.

- **Empirical Impact**: 결과적으로 U-Net은 identity baseline을 크게 넘어섰고, DDPM보다도 상관과 오차가 모두 더 좋았다(예: 피어슨 r=0.975 vs 0.962, MAE=0.014±0.015 vs 0.019±0.019). 특히 추론 시간은 U-Net이 DDPM 대비 208배 빨라(25.2 ms vs 5,227.2 ms, 50 DDIM steps) 계산 비용을 사실상 실시간 수준으로 낮추며 임상 적용 가능성을 높였다는 점이 의미 있다.



### MambaCount: Efficient Text-guided Open-vocabulary Object Counting with Spatial Sparse State Space Duality Block (https://arxiv.org/abs/2606.17650)
- **Prior Approaches**: TOOC(Text-guided Open-vocabulary Object Counting)은 텍스트 프롬프트로 지정된 임의 범주의 객체 개수를 세지만, 밀집 장면과 큰 스케일 변동, 가림(occlusion) 때문에 어렵다. 기존 방법은 (1) CLIP-Count 같은 Transformer 기반 밀도/점 예측으로, attention의 O(N^2) 복잡도가 고해상도·고밀도에서 확장성을 가로막는다. (2) GroundingDINO 계열을 변형 attention으로 희소화하는 탐지-기반 방식은 쿼리 예산 한계로 2차 querying(추가 크롭/추론)이 필요해져 지연 비용이 커지고, 변형 attention 자체가 실제 배포에서 이득이 불안정하거나 구현 부담이 크다.

- **Core Contribution**: 이 논문은 Mamba의 선형 복잡도를 TOOC에 맞게 쓰기 위해 MambaCount를 제안한다. 핵심은 Spatial Sparse State Space Duality(S4D) 블록으로, Mamba의 인과적(causal) 상태 전개가 시각의 비인과(non-causal) 공간 의존성을 제한하는 문제를 완화하고, Spatial Token Selection(STS)으로 공간 토큰 반응의 무제약 고엔트로피를 억제해 로컬 디테일과 고주파 단서를 보존하는 것이다. 또한 Multi-Granularity Prototypes(MGP)로 미세~거시 의미 단위에서 텍스트-비전 정렬을 강화해 오픈 보캐뷸러리 카운팅의 정합성을 높인다.

- **Technical Challenges**: Mamba는 기본적으로 1D 직렬화와 인과 마스킹 성격을 띠기 쉬워, 2D 이미지의 양방향 공간 의존성을 그대로 모델링하기 어렵다. 저자들은 SSD(State Space Duality) 관점에서 hidden state decay/상태 전이의 동역학을 재구성하고, causal mask를 제거해 각 공간 토큰이 전 공간 토큰과 양방향 상호작용을 하도록 하되 state space 효율은 유지하는 MN-SSD를 설계한다. 더 나아가 SSW(Spatial Sparse Window)로 희소 윈도우·다중 dilation 기반 로컬 구조를 복원하고, STS 게이트로 각 위치에서 MN-SSD와 SSW의 기여를 동적으로 선택해 고엔트로피 반응을 제어한다.

- **Empirical Impact**: FSC-147에서 MambaCount는 secondary querying 없이 Test MAE 12.23의 SOTA급 성능을 달성하며, 밀도 회귀 계열 대비 명확한 개선(예: CountTX 16.28 → 12.23)을 보인다. CARPK에서는 Test MAE 4.31로 고밀도 장면에서의 강건성을 보이고, REC-8K에서도 5.42 MAE로 referring expression(비록 카운팅 특화 설계는 아님)까지 일반화 성능을 입증한다. 또한 기여 분석에서 S4D 블록과 MGP를 함께 쓰는 조합이 단독 모듈보다 더 큰 폭의 MAE 하락을 만들며, 선형 복잡도를 유지하는 효율적인 확장성의 실효성을 보여준다.



### Bounding Box Label Propagation for Re-Annotation of Document Layout Analysis Datasets (https://arxiv.org/abs/2606.17644)
Comments:
          17 pages, 3 figures, to appear in proceedings of ICDAR 2026, Vienna, Austria

- **Prior Approaches**: 기존 문서 레이아웃 분석(DLA)은 CNN/비전 트랜스포머/vision-language 모델 등을 대규모 라벨 데이터로 fine-tuning해 객체 탐지를 수행합니다. 그러나 산업 현장에서는 클래스 체계가 계속 세분·갱신되며(박스는 유지되기도 함) 매번 전체 재-annotation이 필요해 비용이 커집니다. 준지도 객체 탐지 연구들은 보통 박스 좌표 자체를 의사라벨로 다시 만들기 때문에, 이 문제처럼 ‘이미 존재하는 바운딩 박스의 클래스만’ 재분류하는 상황에는 불필요한 복잡성이 생깁니다.

- **Core Contribution**: 이 논문은 Bounding Box Label Propagation(BBLP)로, 기존 바운딩 박스 좌표는 그대로 두고 클래스 라벨만 재분류하는 pseudo-labelling 프레임워크를 제안합니다. 시각·텍스트·위치 임베딩을 통합한 Layout Object Encoder(LOE)로 객체 단위의 joint embedding을 만들고, 이를 Label Propagation에 plug-and-play 방식으로 연결해 소량의 수작업 라벨로 나머지 박스의 클래스를 전파합니다.

- **Technical Challenges**: 핵심 난제는 객체 탐지에서 ‘박스 인스턴스 단위’로 라벨을 전파할 수 있는 표현을 구성하는 동시에, 박스 크기/해상도 변화와 문서 내 텍스트·위치 의존성을 함께 반영하는 것입니다. 저자들은 NaFlexViT(가변 해상도 시각 임베딩), Tesseract 기반 OCR 텍스트 임베딩(E5), 그리고 정규화된 위치·이웃 관계 기반 positional embedding을 결합해 LOE를 학습하고, 이후 transductive nearest-neighbour graph(코사인 유사도)에서 label propagation을 수행합니다.

- **Empirical Impact**: 실험에서 BBLP는 D4LA에서 mAP 54.0%를 달성했으며, 이는 fully supervised 성능의 81.6%에 해당하고 라벨 10%만 사용한 결과입니다. 또한 의사라벨 정확도 평가와 잡음 내성(잡음 약 30%대 D4LA 포함) 실험에서, BBLP로 생성한 pseudo-label로 학습한 DLA 모델이 ‘수작업 10%만 학습’ 기준을 일관되게 앞서며 label noise가 있더라도 효과적으로 활용됨을 보여줍니다. 모달리티 ablation에서는 텍스트가 D4LA 성능에 특히 중요하고, 시각·위치 정보도 단독보다 다중 결합에서 더 잘 작동해 실제 문서 재-annotation 비용 절감 가능성을 제시합니다.



### Divide, Deliberate, Decide: A Multi-Agent Framework for Fine-Grained Egocentric Action Recognition (https://arxiv.org/abs/2606.17627)
- **Prior Approaches**: 정교한(세밀한) 동작 인식은 손-도구-접촉 같은 미세한 시각/시간 단서만으로 라벨이 갈려 기존 방법의 전이(transfer)가 어렵습니다. VLM을 그대로 쓰면 모델이 지배적인 물체 토큰에 고정돼 판별 단서를 놓치거나, 단일 모델의 편향된 priors가 특정 단서에 과도하게 쏠리는 문제가 자주 보고돼 왔습니다. 스케일업(더 큰 VLM)도 대안이지만 온프레미스·엣지·프라이버시 제약 때문에 항상 현실적이지 않습니다.

- **Core Contribution**: 이 논문은 Divide, Deliberate, Decide라는 fully-local zero-shot 멀티에이전트 프레임워크를 제안합니다. VLM 오케스트레이터가 영상을 청크로 나누고 세그먼트별 top-k 라벨 후보를 만들면, 서로 다른 모델 패밀리의 VLM 전문가들이 peer-consultation Q&A로 근거를 교환하며 순위를 재평가합니다. 마지막으로 Borda count로 순위를 집계하고 오케스트레이터가 재랭킹해 최종 예측을 확정합니다.

- **Technical Challenges**: 핵심 난제는 (1) 미세 단서에 대해 에이전트들이 단순 합의로 수렴하지 않게 하고, (2) 추가 계산 없이도 새 근거가 의사결정에 실제로 반영되게 만드는 것입니다. 이를 위해 Stage 2에서 분쟁이 생길 때만 한 번의 질문으로 시각적 단서를 확인하도록 프로토콜을 구조화하고, 최종 단계에서는 오케스트레이터가 새 라벨을 제시하지 못하게 제한해 deliberation이 유일한 추가 근거가 되도록 설계합니다. 또한 heterogeneity(서로 다른 모델 패밀리)로 priors를 decorrelate해 상호보완적 후보순위를 만들게 했습니다.

- **Empirical Impact**: MECCANO에서 zero-shot 평가를 수행했으며, 제안 방법은 top-1 16.8%, top-5 45.0%로 baseline(13.5%/28.9%)을 일관되게 개선합니다. 오케스트레이터는 deliberation 이후 약 70.9% 세그먼트에서 초기 top-1을 바꿿고, 올바른 방향의 뒤집기가 반대보다 크게 우세(특히 top-5에서 재라벨링 정답 증가가 매우 큼)해 근거 반영이 확인됩니다. 또한 전문가를 동일 백본 3개로 바꾸면 개선 폭이 줄어들어, 성능 이득이 compute 증가가 아니라 decorrelated priors와 구조화된 Q&A에서 온다는 점을 실험적으로 뒷받침합니다.



### RAVA: Retrieval-Augmented Viewpoint Alignment for Subject-Driven Image Generation (https://arxiv.org/abs/2606.17619)
- **Prior Approaches**: 기존 reference-driven image generation은 identity 보존이나 다중 프롬프트 조합에는 강하지만, 서로 다른 subject 간에서 viewpoint(시점)를 안정적으로 제어하는 문제는 불명확했다. 멀티-이미지 조건 생성은 여러 기준(의미/카테고리/외형)으로 정렬되는 경우가 많아 viewpoint drift, 파트 단위 구조 불일치, 대상별로 필요한 콘텐츠 누락이 발생하기 쉽다. 또한 기존 multi-view 합성이나 카메라 조건 생성은 통상 동일 인스턴스/명시적 기하(poses, depth, ray map)를 가정해 이 태스크의 image-only cross-instance 전이를 그대로 설명하기 어렵다.

- **Core Contribution**: 이 논문은 “anchor subject의 시점이 암묵적으로 주어졌을 때, 다른 subject를 같은 시점으로 렌더링”하는 cross-subject viewpoint alignment를 명확한 문제로 정식화한다. 이어서 생성 전에 기하 증거를 명시적으로 확보하도록 retrieval-augmented 프레임워크 RAVA를 제안한다. RAVA는 (1) cross-instance viewpoint embedding으로 anchor 시점과 정렬되는 후보 이미지를 검색하고, (2) LogDet 기반 subset selection으로 중복은 줄이면서 구조적으로 상보적인 레퍼런스만 남긴 뒤, (3) fine-tuned multi-reference 생성기에 제공한다.

- **Technical Challenges**: 핵심 난제는 “의미가 아니라 viewpoint 호환성”을 순위화하는 표현을 만드는 것이다; 기존 vision-language 임베딩은 category/외형 기준 클러스터링에 최적화돼 cross-instance 시점 비교에는 거의 무작위에 가깝게 작동한다. 논문은 Qwen3-VL 위에 viewpoint 전용 임베딩을 학습하고, visual-token만 사용한 뒤 global pooling과 region-aware pooling을 gated fusion으로 결합해 객체 간 시점 비교가 가능한 표현을 만든다. 또한 작은 reference budget에서 near-duplicate 뷰를 낭비하지 않기 위해 quality-weighted kernel과 LogDet 기반 최적화를 통해 view fidelity와 complementarity를 동시에 만족하는 선택을 수행한다.

- **Empirical Impact**: Objaverse-XL 기반 벤치마크에서 generic semantic embedding은 NDCG@1이 약 0.33~0.36 수준이고 Spearman 상관도 거의 0에 가까웠지만, RAVA의 retriever는 NDCG@1 0.750 등으로 크게 향상됐다. downstream cross-subject 생성에서도 retrieval 기반 조건화가 zero-shot baseline과 더 강한 retrieval 대안 대비 일관되게 우수하며, 동일 generation backbone(미세조정 Flux.2)을 쓰는 통제 실험으로 “생성기 자체”보다 “기하적으로 신뢰할 수 있는 레퍼런스 제공”의 중요성을 강조한다. 결론적으로 end-to-end 생성만으로 viewpoint를 전이하려는 접근보다, retrieval-augmented 기하 grounding이 cross-subject viewpoint 정렬에 핵심이라는 메시지를 실증한다.



### SkillMoV: Mixture-of-View Routing with Prototype-Conditioned Gating for Unified Multi-View Proficiency Estimation (https://arxiv.org/abs/2606.17615)
- **Prior Approaches**: 기존 AQA/숙련도 추정은 특정 활동 도메인에 맞춘 시나리오별 모델이 많거나, 여러 카메라를 단순 집계·공유 변환으로 결합해 뷰별 단서를 충분히 활용하지 못하는 한계가 있었다. EgoExo4D에서도 unified multi-view로 가는 흐름이 있었지만, 여전히 공통 투영/어텐션에 의존해 뷰 간 차이를 “전문가”처럼 분리해 학습하긴 어려웠다. 또한 카메라 identity나 시나리오 전용 헤드 없이도 뷰에 종속된 표현을 만들 수 있는 설계가 부족했다.

- **Core Contribution**: 이 논문은 synchronized multi-view 영상에서 여러 스킬 도메인을 아우르는 파라미터 효율적(LoRA 기반) 통합 프레임워크 SkillMoV를 제안한다. 핵심은 Mixture-of-View Projector(MoVP)로, mixture-of-experts를 카메라별 뷰 특징에 직접 라우팅해 뷰 의존적 전문성 선택을 학습하되 카메라 identity supervision 없이도 동작하게 만든다. 여기에 cross-view attention 정렬, prototype anchoring, prototype-conditioned gated projection을 계층적으로 결합해 최종 skill embedding을 만든다.

- **Technical Challenges**: 문제는 서로 동기화된 멀티 카메라가 동일 실행을 보여도 드러내는 숙련 단서가 달라 공통 집계가 신호를 희석할 수 있다는 점이다. SkillMoV는 이를 MoV 라우팅(12개 expert MLP를 뷰별 soft mixture로 선택)으로 해결하고, cross-view attention으로 동기 카메라 정렬 후 prototype anchoring 및 gated projection으로 클래스(숙련 단계)별 참조 기반 조건화를 추가한다. 또한 멀티뷰 설정에서 stochastic view dropout으로 특정 카메라에 과적합하는 현상을 줄였고, 최적 학습 안정성을 위해 class-balanced cross-entropy를 선택했다.

- **Empirical Impact**: EgoExo4D에서 SkillMoV는 Exos 설정 단일 모델 통합 학습으로 50.17% overall accuracy를 달성하며, 비교 방법 중 최강 Exos 결과보다 3.57%p 개선했다. Ego+Exos에서도 47.63%로 기존 최상 수준(48.20%)에 근접해 통합 프레임워크의 견고함을 보여줬다. ablation 결과로는 MoV routing(+6.61%p), cross-view attention(+4.92%p), prototype anchoring(+4.07%p), stochastic view dropout(+3.90%p) 기여가 확인되며, LoRA adaptation을 통해 학습 파라미터를 23.32%만 사용하면서도 오버헤드는 LoRA-only 대비 제한적으로 유지됐다.



### Flux-Guard: Facial Identity Protection using diffusion models (https://arxiv.org/abs/2606.17606)
- **Prior Approaches**: 기존 얼굴 프라이버시 보호는 픽셀 공간에 작은 adversarial noise를 더해 인식 성능을 떨어뜨리는 방식이 중심이었지만, 고주파 잡음이나 시각적 왜곡이 눈에 띄는 문제가 컸습니다. 이후 makeup transfer처럼 의미 있는 외관 변화를 만들었으나, black-box 환경에서 모델 간 transferability가 제한되고 특정 스타일은 인간에게도 과도하게 도드라져 실용성이 낮았습니다. 또한 diffusion 기반 잠재공간 공격은 더 자연스러워졌지만, 생성 편집(editing) 출력에서도 동일하게 프라이버시를 보장하는 통합 접근은 부족했습니다.

- **Core Contribution**: 이 논문은 텍스트 기반 face editing과 identity 프라이버시 보호를 하나의 unified generative process 안에 결합한 Flux-Guard를 제안합니다. 편집 결과가 여전히 추적(tracking)될 위험이 있다는 점에 주목해, 의미 조작은 편집처럼 유지하되 악의적 FR 시스템에 대한 공격 성공률을 함께 높이도록 설계했습니다. 특히 FLUX(Flux 모델)와 rectified flow 기반 생성 흐름을 활용해 “편집 정합성 + 프라이버시”를 동시에 맞추는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 편집 의미(텍스트 지시)와 생성 궤적을 정렬하면서도 (2) 원래 얼굴의 구조·자연스러움을 깨지 않는 동시에 (3) black-box FR에 전이되는 adversarial 효과를 후단에서 내야 한다는 점입니다. 논문은 DiT에서 attention 값을 캐시해 초기 구조를 고정하는 structural injection을 도입하고, 편집 영역에는 text-guided velocity를, 비편집 영역에는 inversion velocity를 유지하는 mask-guided flow-trajectory control로 국소 의미 조작을 구현합니다. 마지막으로 잠재공간에서 latent-space adversarial optimization을 하되, perceptual loss 기반 adaptive weighting으로 잡음/왜곡 임계선을 넘기기 전에 adversarial 강도를 자동으로 낮춰 시각 품질을 방어합니다.

- **Empirical Impact**: 실험에서는 CelebA-HQ와 LADN에서 cross-domain face recognition 모델을 상대로 공격 성공률(ASR, Rank-1/Rank-5)을 유의미하게 개선했고, 생성 품질도 FID/PSNR/SSIM로 확인했습니다. 더 나아가 Face++와 Aliyun FR 같은 상용 API에서도 효과가 검증되어 실제 서비스 환경에서도 프라이버시 위험을 낮출 수 있음을 보여줍니다. 결과적으로 Flux-Guard는 편집 도구가 확산되는 현실에서 “편집된 결과물의 프라이버시”까지 고려하는 다음 단계의 공격-방어 논의를 촉진할 것으로 보입니다.



### Test-Time Training for Robust Text-Guided Open-Vocabulary Object Counting (https://arxiv.org/abs/2606.17601)
- **Prior Approaches**: 기존 TOOC(text-guided open-vocabulary object counting)는 CLIP 계열의 시각-언어 정렬을 바탕으로 density map 회귀나 open-vocabulary grounding/검출을 활용해 물체를 세는 방식이 주류였다. 그러나 대부분 clean 이미지(이상 조건)에서 성능을 검증해, rain·fog·darkness·잡음 같은 실제 부식(corruption)에서 비주얼-텍스트 정렬이 깨지면 오류가 크게 증가한다. 또한 test-time training(TTT/TTA) 접근은 주로 분류/검출에 맞춰 범주 비특화 feature 안정화만 하며, 텍스트 프롬프트로 적응 방향을 명시적으로 유도하는 설계가 부족했다.

- **Core Contribution**: 이 논문은 Robust-TOOC라는 최초의 robust 평가 벤치마크를 제안해 TOOC의 부식 강건성을 체계적으로 측정한다(6종: rain, fog, darkness, Gaussian noise, salt-and-pepper noise, mixed corruption). 아울러 Dual-TTT라는 텍스트-지 guided test-time training 프레임워크를 제안하며, 원래 카운팅 아키텍처는 고정한 채 Text-guided Lightweight Denoising(TL-Denoiser)만 업데이트한다. 핵심은 “부식 억제 + 텍스트 정렬 회복”을 동시에 달성하도록, 프롬프트에 의해 적응이 범주/의미 수준에서 유도된다는 점이다.

- **Technical Challenges**: TOOC에서는 부식이 시각 토큰의 의미적 구분력을 훼손해 텍스트-이미지 정렬이 무너질 수 있는데, 단순한 분포 정규화나 category-agnostic stabilization으로는 부족하다. 또한 실제 부식은 다양하고 사전에 열거/라벨링이 어렵기 때문에, 추가 주석 없이 추론 중에 온라인으로 복원 목표를 세우는 설계가 필요했다. 논문은 확산 모델에서 영감을 받아 픽셀 복원이 아닌 feature 공간에서 noise semantics(부식 타입 프롬프트)와 entropy 기반 위치별 복원 강도를 사용해, TL-Denoiser가 더 깨끗한(정렬이 잘 된) 표현을 만들도록 KL divergence 기반의 annotation-free 목표로 최적화한다.

- **Empirical Impact**: 여러 최신 TOOC 베이스라인에 대해 Dual-TTT는 Mixed 및 Average 부식 조건 전반에서 일관되게 MAE/RMSE를 개선했으며, 특히 CLIP 기반 모델에서 성능 향상이 두드러졌다. 예를 들어 CounTX/CLIP-Count/CountGD 모두에서 부식 유형별로 가장 어려운 조건에서도 오차를 크게 줄였고, 심지어 더 강한 baseline(예: Adaptive Crop 등) 없이도 개선이 관찰되어 범용성이 확인됐다. 또한 generic TTT/TTA 계열(예: Tent, EATA, TTT++)은 거의 효과가 없었던 반면, 본 방법은 부식 난이도가 높아질수록 격차를 더 크게 만들며 “TOOC 전용” 적응 전략의 필요성을 실증했다.



### TivTok: Broadcasting Time-Invariant Tokens for Scalable Video Tokenization (https://arxiv.org/abs/2606.17590)
- **Prior Approaches**: 기존 비디오 토크나이저는 주로 토큰 수를 줄이는 데 집중해 왔습니다. downsample 기반은 프레임(또는 3D 모듈) 압축으로 확장성을 확보하지만, 긴 시퀀스에서 토큰이 증가하거나 표현이 ‘공유 vs 변화’를 명확히 분리하지 못하는 경우가 많습니다. 또한 holistic 토크나이저와 처방적 분해 기반 방법은 압축에는 강점이 있으나, 클립/청크 전반의 persistent(지속) 정보를 재사용 관점에서 체계적으로 다루지 못했습니다.

- **Core Contribution**: 이 논문은 시간 재사용 관점에서 비디오 토크나이징 문제를 재정의하고, TivTok(Time-Invariant Tokenizer)을 제안합니다. TivTok은 한 클립을 Time-Invariant(TIV) 토큰(시간에 걸쳐 공유되는 정보)과 Time-Variant(TV) 토큰(프레임별 residual)로 분해해, 지속 구조를 한 번 인코딩하고 여러 프레임/청크에 재사용하도록 설계했습니다. 이를 통해 공유 성분은 재사용하고, 변화 성분에만 프레임별 용량을 쓰는 토크나이저를 지향합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “무엇이 지속인지”를 모델이 스스로 찾아내되, 역할이 섞이지 않게 학습을 유도하는 것입니다. 논문은 Scope-Induced Factorization(SIF)으로 인코더에서 TIV/TV 토큰의 attention scope를 비대칭으로 제한해, TIV는 전체 클립을 보며 공유 정보를 모으고 TV는 해당 프레임에 국소적으로만 보도록 강제합니다. 디코더에서는 Invariant Broadcasting(IB)으로 동일한 TIV 토큰을 모든 프레임에 broadcast해 병렬 복원을 가능하게 하며, 디코딩 복잡도를 비디오 길이에 대해 더 완만하게 만듭니다.

- **Empirical Impact**: 실험에서 TivTok은 표준 16×256×256 설정에서 rFVD 12.65를 달성했으며, 128프레임에서는 평가된 베이스라인 대비 압축 효율을 2.91× 개선했습니다. 더 나아가 downsample 기반 토크나이저가 요구하는 토큰의 1.1%만 사용하면서도 성능을 유지/개선하는 결과를 보여 긴 비디오 생성·복원에서의 자원 효율 잠재력을 입증합니다. 정성 분석에서도 TIV가 픽셀 정적 배경이 아니라 의미적 불변(예: 객체 정체성)을 포착해 TV가 잔차를 담당하는 ‘학습된 재사용’의 효과가 확인됩니다.



### Root-Selecting Fixed-Point Inversion for Rectified Flows via Trajectory Straightness (https://arxiv.org/abs/2606.17584)
- **Prior Approaches**: rectified flow에서의 inversion은 실이미지를 초기 noise로 되돌리는 과정이며, discretization error 때문에 역방향 적분만으로는 오차가 누적됩니다. 기존 접근은 conditioning 변경, auxiliary 변수 최적화, inversion- reconstruction을 결합한 sampler 수정 등으로 오차를 줄였고, fixed-point inversion(ReNoise, AIDI, GNRI 등)은 이산 solver의 local inverse 방정식을 반복적으로 푸는 방식으로 정확도를 끌어올렸습니다.
다만 고정점 방정식이 여러 근을 허용하는 multi-root 상황에서는 residual(고정점 오차)만 작아도 선택된 근에 따라 inversion trajectory가 달라져 복원·편집 품질이 크게 흔들릴 수 있었습니다.

- **Core Contribution**: SelFix는 fixed-point inversion을 ‘근 찾기’뿐 아니라 ‘여러 근 중 원하는 근 선택’ 문제로 재정의합니다. 특히 rectified flows에서 inversion 경로의 straightness(직선성)가 discretization 기반 오차 누적 실패와 강하게 연관된다는 관찰을 바탕으로, straighter inverse trajectory를 유도하는 근 선택 기준을 설계합니다.
그 결과 SelFix는 표준 local 가정 하에서 exact inverse root로의 수렴을 보존하면서도, straightness 기준에 더 잘 맞는 근을 선택하도록 고정점 반복을 anchored 방식(예: Halpern iteration)으로 구성합니다.

- **Technical Challenges**: 핵심 기술 난점은 고정점 방정식이 여러 해를 가질 때, 어떤 해를 선택하느냐가 trajectory 전체의 오차 전파(velocity 재평가)를 바꾼다는 점을 반복 알고리즘에 ‘원칙적으로’ 반영하는 것입니다. 논문은 이를 위해 straightness를 고정점 선택자(selection criterion)로 만들되, inversion 과정 중에는 전체 궤적 평균 등을 전역적으로 계산하기 어렵다는 제약을 고려해 on-the-fly 가능한 straightness proxy를 구성합니다.
또한 anchored 반복에서 anchor 영향이 장기적으로 사라지도록 스케줄을 설계하고, 수렴과 선택 신호를 모두 해치지 않기 위해 decoupled momentum으로 finite-iteration 동작을 개선합니다.

- **Empirical Impact**: FLUX.1-dev를 backbone으로 NFE를 맞춘 비교에서 SelFix는 PIE-Bench 실이미지 복원 지표 전반에서 기존 fixed-point 및 naive 기준선을 능가하며 reconstruction error를 낮추는 경향을 보였습니다. 편집 성능에서도 source 보존(배경 유지)에서 가장 강한 결과를 내면서 target prompt 정합성도 경쟁력 있게 유지해, preservation-editing trade-off를 개선했습니다.
또한 SelFix가 다른 방법 대비 더 낮은 trajectory straightness(DS)와 straightness proxy 누적값을 달성함을 통해, ‘더 곧은 inverse 경로 선택→오차 누적 감소’라는 가설이 실험적으로 뒷받침됨을 보여줍니다.



### Geometric Consistency Protocol for Foundation Model Features in Multi-View Satellite Imagery (https://arxiv.org/abs/2606.17564)
Comments:
          The manuscript is accepted as Oral Presentation in IEEE International Geoscience and Remote Sensing Symposium(IGARSS 2026)

- **Prior Approaches**: 원격탐사 multi-view에서 기존 평가는 대부분 2D 전역 argmax 매칭(무제약 2D 글로벌 서치)에 의존했습니다. 하지만 RPC 카메라의 epipolar 기하가 곡선이며 고도에 따라 달라서, 물리적으로 가능한 탐색공간이 3D 결정 문제인데도 2D 평면에서만 찾도록 평가가 구성되는 경우가 많았습니다. 또한 위성 영상의 반복 구조(도로·지붕 등)와 방사보정 변화는 유사도 응답의 스푸리어스 최대값을 키워 랭킹을 왜곡할 수 있습니다.

- **Core Contribution**: 이 논문은 RPC(Rational Function Model) 프레임워크에 맞춘 geometry-faithful 평가 프로토콜을 제안합니다. 핵심은 (1) RPC로 투영한 동일 3D 포인트의 교차 뷰 feature 일치성을 보는 RPC-projected 3D consistency 지표와 (2) 기하 제약을 둔 탐색공간에서 유사도 피크의 국소성·유일성을 확인하는 geometry-constrained dense matching proxy를 함께 보고하는 것입니다. 특히 두 측면을 분리해, 의미적(semantic) 일치는 높아도 실제 매칭(matchability)은 보장되지 않는 현상을 명확히 합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 RPC epipolar 기하를 따라 ‘물리적으로 가능한’ 후보 탐색 매니폴드를 평가에 정확히 반영하는 것입니다. 논문은 DSM 기반 고도/역투영(RPC 역투영)과 고정된 재현 가능 솔버 설정을 사용해 동일 3D 포인트를 각 뷰로 정확히 재투영하고, invalid point·가시성 조건 미충족 포인트를 배제하는 방식으로 지표의 일관성을 확보했습니다. 또한 전역 2D 탐색과 RPC epipolar band(예: ±4 pixels) 탐색을 함께 실행해 기하 제약의 영향을 정량 비교할 수 있게 했습니다.

- **Empirical Impact**: DF C2019의 Omaha·Jacksonville 두 full-region(100+ area of interest)에서 실험한 결과, 전역 글로벌 서치는 모든 방법의 성능을 크게 흔들었고 RPC epipolar band로 제한했을 때 clsPCK@10 등이 일관되게 크게 개선되었습니다. 흥미롭게도 2D 백본(dense feature, 예: DINOv3·SAM)은 RPC-consistent 평가에서도 여전히 강력한 경쟁력을 보였고, 3D-aware/ multi-view-aware 계열이 항상 최상 성능을 보장하진 않았습니다. 즉 ‘기하 제약을 포함한 정의가 있어야’ 비로소 foundation feature의 실매칭 능력을 공정하게 비교할 수 있으며, semantic consistency만으로는 성능을 예측하기 어렵다는 메시지를 남겼습니다.



### RT-Counter: Real-Time Text-Guided Open-Vocabulary Object Counting (https://arxiv.org/abs/2606.17561)
- **Prior Approaches**: 기존 TOOC는 주로 closed-set 가정에 기반해 사전 정의된 범주만 세는 방식이 많았고, 이후 open-vocabulary로 확장되면서 few-shot/zero-shot 시도가 등장했습니다. Visual prototype 기반 방법은 few-shot은 추가 샘플 제공이 필요하고, zero-shot은 자주 등장하는 시각 특징에 의존해 희귀 범주에서 취약합니다. Text-guided 접근은 CLIP/Open-CLIP/GroundingDINO 같은 비전-언어 사전학습 모델에 크게 의존하지만, 계산비용이 커 실시간 배치가 어렵고, GroundingDINO 계열은 비미분 출력으로 end-to-end 학습을 제한합니다.

- **Core Contribution**: 이 논문은 실시간 text-guided open-vocabulary object counting을 목표로 RT-Counter를 제안합니다. 핵심은 Visual Prototype Textualization(VPT) 모듈로, 시각 프로토타입을 텍스트 특징 공간에 투영해 텍스트가 놓치는 미세한 시각 디테일과 시각 프로토타입이 부족한 추상 의미를 동시에 보강한다는 점입니다. 또한 Weaformer 레이어로 로컬-글로벌 정보를 효율적으로 ‘weave’해 성능-속도 균형을 깨는 것을 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 시각 프로토타입과 텍스트 범주 설명 사이의 semantic gap을 메우면서 (2) 고해상도에서 counting 정확도를 유지하고 (3) 연산량을 줄여 실시간 FPS를 만족하는 것입니다. RT-Counter는 VPT에서 cross-attention을 이용해 프로토타입 쿼리에서 추출한 시각 정보와 텍스트 임베딩을 교차 융합하고, 이후 Feature Enhancer가 이 풍부해진 프로토타입을 반복적으로 반영하도록 설계합니다. 실시간성을 위해 Weaformer는 표준 self-attention의 O(N^2) 병목을 줄이기 위해 다운샘플링 후 글로벌 경로(자기어텐션)와 로컬 경로(conv-attention, 윈도우 주의)를 혼합한 잡종 attention을 적용합니다.

- **Empirical Impact**: FSC147, CARPK, REC-8K의 3개 공개 데이터셋 실험에서 RT-Counter는 accuracy-speed trade-off를 강하게 개선했다고 보고합니다. 예를 들어 FSC147에서 MAE 13.30을 달성하면서도 112.48 FPS로, 기존 빠른 계열 대비 약 7.4x 속도 향상을 보이며 파라미터도 38M으로 더 작아집니다. VPT/Feature Enhancer를 제거한 ablation에서는 VPT 제거가 가장 큰 성능 하락(예: MAE 13.30→16.09)을 유발해 핵심 기여를 입증했고, VPT를 CounTX에 plug-and-play로 붙였을 때도 MAE가 16.28→15.39로 개선되어 범용성을 시사합니다. 



### Universal Image Restoration via Internalized Chain-of-Thought Reasoning (https://arxiv.org/abs/2606.17557)
- **Prior Approaches**: 기존 통합(in all-in-one) 이미지 복원 모델은 다양한 열화(rail/haze/snow/노이즈/블러/압축 등)를 한 모델로 처리하지만, 열화 조합이 복잡해질수록 성능이 급격히 떨어지는 문제가 있었다. 이를 보완하려고 Chain-of-Thought(CoT) 기반의 multi-round 복원은 열화를 단계별로 분해해 모듈을 연쇄로 호출하지만, 연산 비용이 커지고 각 단계가 열화 간 상호작용을 충분히 모델링하지 못한다는 한계가 있었다.

- **Core Contribution**: 이 논문은 복원을 “Thinking → Planning → Action”으로 재정의하고, CoT 추론을 단일 모델 내부에 내재화해 단계적 체인 없이 end-to-end 복원을 수행하는 CoTIR을 제안한다. 또한 복원을 image editing의 한 하위 작업으로 보고, 대규모로 사전학습된 editing 모델(FLUX 계열)을 강한 초기화로 활용한 뒤 복원 목적에 맞게 fine-tuning한다.

- **Technical Challenges**: CoTIR의 핵심은 복원 품질을 높이기 위해 중간에 필요한 ‘샤프/열화 패턴/복원 계획’ 같은 구조화된 CoT 정보를 학습 과정에 어떻게 연결할지에 있다. 저자들은 Lagrangian 최적화에서 영감을 받은 미분 가능 soft constraint와 multi-constrained optimization을 학습 목적에 포함해, 단일 전방 패스로도 중간 추론 정합성을 강제하도록 설계했으며, 디코더는 LoRA로 효율적으로 업데이트한다.

- **Empirical Impact**: CoTIR-Bench(5.2M 샘플, CoT 추론 trace 포함)와 다양한 실제 composite 열화 장면에서 CoTIR은 all-in-one 및 multi-round CoT 복원 대비 더 높은 지각 품질(예: no-reference 계열에서 강점)과 경쟁력 있는 충실도 결과를 보인다. 특히 LPIPS가 낮고 전반적으로 균형 잡힌 품질을 보여, 복원에서 열화 상호작용을 내재적으로 다루는 접근의 실용적 의미가 크다는 점을 실험적으로 입증한다.



### TaFD: Threat-Aware Frequency Decoupling for Adversarial Robustness against Heterogeneous Attacks (https://arxiv.org/abs/2606.17540)
- **Prior Approaches**: 기존 방어는 단일 threat(예: ℓp bounded 공격)에는 adversarial training(AT)으로 강건성을 학습하지만, 서로 다른 성격의 공격이 함께 존재하는 multi-threat 환경에서는 joint adversarial training(JAT)이 negative transfer를 일으키는 문제가 있었다. 특히 ℓp
p-bounded(미세 잡음)과 semantic(색/기하 변환 등) 공격을 함께 다룰 때, 한 threat에 대한 최적화가 다른 threat의 성능을 동시에 깎는 현상이 보고된다. 이러한 갈등을 loss 합산/최악 케이스 선택 등으로 뭉쳐 업데이트 방향을 절충하는 방식은 상충되는 목적 자체를 조정하지 못해 한계가 컸다.

- **Core Contribution**: 이 논문은 negative transfer를 first-order gradient 관점에서 gradient incompatibility(기울기 비호환성)로 정식화하고, 이를 해결하려면 decoupled optimization(분리된 최적화)이 필요하다고 이론적으로 보인다. 또한 서로 다른 threat들이 주파수 영역에서 separable spectral characteristics(분리 가능한 스펙트럼 패턴)를 보인다는 관찰을 제시해, 픽셀 공간의 얽힘을 주파수 표현으로 풀 수 있음을 동기화한다. 이를 바탕으로 Threat-aware Frequency Decoupling(TaFD) 진단-파견(Diagnosis–Dispatch) 구조를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 이질적 threat들에 대해 하나의 공유 파라미터로 동시에 “맞는” 업데이트를 찾으려 하면 최적화 충돌이 지속된다는 점이며, 이를 단순한 loss-level 통합으로는 해결하기 어렵다는 것이다. TaFD는 (1) 공격의 스펙트럴 prototype을 unsupervised clustering으로 잠재 threat domain을 찾고, (2) inference-time에 lightweight classifier로 threat domain을 라우팅한 뒤, (3) Frequency-Conditional Convolution(FC-Conv)이 도메인별 spectral mask를 학습해 expert별로 샘플을 hard routing하도록 설계해 구조적 파라미터 분리를 강제한다. 또한 평가 시 BPDA를 통해 adversarial evaluation의 gradient 접근성을 확보해 학습-평가 일관성을 유지한다.

- **Empirical Impact**: TaFD는 CIFAR-10/100, Tiny-ImageNet의 세 벤치마크와 ResNet(컨볼루션) 및 MobileViT(하이브리드 트랜스포머) 두 아키텍처에서 실험됐고, 이질적 공격 조합에서 기존 JAT 및 주파수 기반 baseline보다 더 균형 잡힌 robust 성능을 보였다. 특히 strongest baseline 대비 average robust accuracy를 약 11%p(최대 11.3%p 수준) 개선하면서도 clean accuracy를 유지해 방어-성능 트레이드오프를 완화했다. 주파수 도메인 기반 threat decoupling이 multi-threat robustness의 실질적 경로가 될 수 있음을 보여준다는 점에서 분야에 직접적인 파급이 기대된다.



### Reinforcing Dual-Path Reasoning in Spatial Vision Language Models (https://arxiv.org/abs/2606.17539)
- **Prior Approaches**: 기존 spatial VLM은 3D 레이아웃, 깊이/가림, 시점 의존 관계를 인식하는 데는 진전이 있었지만, 깊이·거리·장면 관계를 넘나드는 다단계 추론까지는 취약하다는 분석이 반복됐다. RL을 통해 다단계 reasoning을 유도한 방법들도 존재하지만, 기반 VLM이 충분히 강한 공간 지각을 갖추지 못하면 spatial VLM이 제공하는 기하 구조를 충분히 활용하지 못한다. 또한 기존 접근은 언어 기반 추론과 3D grounding 기반 추론을 하나의 모델/훈련 프레임워크 안에서 함께 지원하는 경우가 드물었다.

- **Core Contribution**: SR-REAL(SR-REAL, Dual-Path Spatial Reasoning via Reinforcement Learning)은 공간 VLM에 두 가지 상보 경로를 동시에 탑재한다: LOR(Language-Only Reasoning)는 장면 관계에 대한 단계적 언어 추론을 수행하고, DTR(Detect-Then-Reason)은 3D 기하 단서를 region token으로 먼저 탐지·정렬한 뒤 정량 추론을 한다. 모델은 한 번의 체크포인트로 두 경로를 모두 지원하도록, cold-start supervised fine-tuning으로 두 경로의 CoT(Chain-of-Thought)와 region-to-3D 인터페이스를 만들고 이후 RL로 정확도와 출력 형식을 함께 최적화한다. 결과적으로 “질문 유형에 따라 다른 전략이 필요하다”는 문제를 단일 통합 설계로 흡수한다.

- **Technical Challenges**: 핵심 난제는 텍스트에서 바로 3D 좌표/박스를 예측하는 것이 어렵다는 점이며, SR-REAL은 이를 region token을 매개로 한 2D→3D grounding 다리로 해결한다. cold-start 단계에서 LOR용 언어 CoT와 DTR용 ‘detect-then-quantitative reasoning’ CoT를 각각 구조화해 학습하고, DTR에는 region-to-3D 인터페이스를 통해 센터나 bounding box를 예측하도록 하며 직접 grounding만 하면 성능이 크게 떨어진다는 점을 확인한다. 이후 GRPO 스타일 RL에서 accuracy reward + format reward를 기본으로 두고, DTR에는 discretized detection reward(예측 센터와 정답 좌표 거리 기반)를 추가하며, online filtering으로 비유리 롤아웃을 제거해 안정적인 최적화를 유도한다.

- **Empirical Impact**: 실험에서 SR-REAL은 SPAR-Bench, EmbSpatial, SAT 등 다수 공간 벤치마크에서 기존 spatial VLM/추론 모델 대비 일관되게 향상되며, 단일 모델이 LOR과 DTR을 동시에 제공하는 점도 확인됐다. 특히 DTR은 region 기반 태스크에서 3D localization 정밀도로 이점을 보이고, LOR은 언어적 단계 추론 능력으로 일반 spatial reasoning에 기여한다. ablation 결과로는 두 경로를 함께 학습할 때 상호 강화가 나타나고, cold-start 데이터 품질(2D/3D grounding 블렌딩)이 RL 안정성과 cross-domain 전이에 중요하다는 점이 강조되며, 데이터셋/도메인별 per-task 튜닝 없이 positive transfer가 관찰된다는 의미가 크다.



### OmniDrive: An LLM-Choreographed Multi-Agent World Model with Unified Latent Co-Compression for Multi-View Driving Video Generation (https://arxiv.org/abs/2606.17536)
Comments:
          24 pages, 10 figures

- **Prior Approaches**: 기존 생성형 world model은 멀티뷰 영상을 카메라별로 각각 인코딩한 뒤, cross-view attention 등으로 사후에 맞추는 방식이 많았습니다. 이 구조에서는 언어·HD-map/trajectory 같은 기하 제어와 픽셀 증거가 잠재 토큰 레벨에서 같은 좌표계에 정렬되지 않아 cross-view drift, 깜빡임, 객체 teleporting 같은 문제가 반복됐습니다. 또 ControlNet류 분기(기하)와 cross-attention 어댑터류 분기(의미/프롬프트)를 “후처리로 조합”하는 경향이 있어 이질적인 컨트롤 주입 간 불일치가 남았습니다.

- **Core Contribution**: DRIVE-CHOREO는 이 공통 원인을 ‘언어-기하-픽셀을 latent-token 레벨에서 정렬해 주는 shared symbolic interlingua의 부재’로 짚고, 이를 단일 위치 인식 토큰 그리드로 해결합니다. 핵심은 LLM-choreographed multi-agent world model로, 세 Qwen2.5-VL 에이전트가 WorldScript를 만들고(Director/Architect), 이를 공간 앵커된 레이아웃 토큰으로 바꾸며(Cartographer), 카메라 간 불일치를 비평해 보조 감독(Auditor)을 주는 방식입니다. 나아가 6카메라×시간을 “view-time permutation”으로 재배열해 3-D VAE 안에서 기하 제약이 로컬 합성곱 의존성으로 들어가도록 co-compressed latent를 구성합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 서로 다른 표현 공간(자유 텍스트, 기하/좌표계, 픽셀 잠재)을 토큰 좌표 수준에서 기계적으로 정렬하고 (2) 카메라 수가 늘어도 같은 물리적 순간의 뷰 간 제약을 VAE의 receptive field 안에 담아 일관성을 유지하는 것이었습니다. 논문은 view-time permutation으로 RGB와 Cartographer의 기하 레이아웃을 동일한 pseudo-temporal 스트림으로 co-compression하며, 동시에 same-instant 뷰의 noise endpoint를 공유해 초기 광도 변동을 줄였습니다. 여기에 cross-view 비평 점수를 Auditor auxiliary objective로 흐르게 해, 에이전트 출력이 “사전 처리”가 아니라 diffusion/flow-matching 학습 중간에 직접 영향을 주도록 설계했습니다.

- **Empirical Impact**: nuScenes에서 DRIVE-CHOREO는 multi-view consistency와 BEV mAP에서 새로운 SOTA를 달성했고, BEV mAP 21.6 및 경쟁 FVD 45.7을 보고합니다. 또한 순수 생성 데이터로 학습한 detector가 실측 validation에서 NDS +2.4 향상을 보여 downstream 유틸리티까지 검증했습니다. ablation과 기존 방식(ControlNet/사후 cross-attention)과의 비교는 에이전트 기반 choreographed conditioning과 co-compression이 cross-view 정합성과 controllability를 함께 끌어올렸음을 수치로 뒷받침합니다.



### SPHINX: First Explain, Then Explor (https://arxiv.org/abs/2606.17482)
Comments:
          13 pages

- **Prior Approaches**: 기존의 ChatScene, LLM-Attacker 같은 방법은 Large Language Models과 Vision-Language Models의 사전 지식에 주로 의존해 시나리오를 절차적으로 생성합니다. 문제를 “정해진 가정”으로 다루기 때문에, 실제로 해당 주행 정책이 보이는 약점을 직접 진단해 겨냥하기보다는 일반적인 적대 상황 탐색에 그칠 수 있습니다. 그 결과 정책의 취약 지점과 생성된 적대 장면의 연결이 약해질 위험이 있습니다.

- **Core Contribution**: 이 논문은 적대적 장면을 정책의 failure diagnosis(예: indecisiveness, multi-frame inconsistency)를 바탕으로 생성해 약점을 겨냥해야 한다는 관점을 제안합니다. 이를 구현하는 폐루프(closed-loop) 프레임워크 SPHINX를 제시하며, 핵심 원칙은 first explain, then explore입니다. 정책의 의사결정 과정에서 얻은 해석 가능한 근거를 비판과 재학습용 장면 생성에 직접 연결합니다.

- **Technical Challenges**: 가장 큰 난제는 “정책 실패의 원인을 설명 가능한 형태로 뽑아내고”, 그 근거를 기반으로 “실제로 실패를 유발하는” 장면을 생성하는 폐루프를 만드는 것입니다. SPHINX는 explainable artificial intelligence로 핵심 시각 개념과 결정에의 영향, 그리고 decision의 불확실성을 분석해 정책 내부의 해석 가능한 증거를 수집합니다. 그 증거를 바탕으로 vision language model이 실패 모드를 rationalize하고 criticize하며, 이 비평을 조건으로 targeted adversarial scenario를 만들어 policy retraining에 활용합니다.

- **Empirical Impact**: 실험에서 SPHINX는 다른 적대 장면 생성 방법들과 달리 정책 실패에 대한 해석 가능한 설명(왜 실패하는지의 근거)을 함께 제시할 수 있음을 보여줍니다. 또한 여러 벤치마크와 테스트 스위트에서 다양한 SOTA 자율주행 아키텍처에 적용 가능하며, 기존 scenario-generation 대비 일관된 robustness 개선을 달성합니다. 결과적으로 단순 생성 성능을 넘어 “취약 원인-장면-재학습”의 연결성을 강화하는 실증적 진전으로 평가됩니다.



### GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning (https://arxiv.org/abs/2606.17480)
- **Prior Approaches**: GeneralVLA 계열은 SAM 기반 어포던스 분할과 3D 장면 추정, 그리고 언어-3D-기억을 결합한 mid-level 3DAgent의 경로를 저수준 제어가 실행하는 계층형 VLA를 제안했습니다. 하지만 기존 GeneralVLA의 3D evidence는 단일 이미지/마스크 기반 SAM3D류에서 포즈 모호성과 뒷면 구조 환각 문제가 남고, KnowledgeBank는 의미 유사도 중심이라 안전성·현재 장면 기하 적합성·충돌·신뢰도 같은 “기억 품질”을 통제하기 어렵습니다.

- **Core Contribution**: 논문은 GeneralVLA-2에서 planner-facing 입력을 강화하는 두 축을 제시합니다: GeoFuse-MV3D로 다중 시점 RGB-D에서 더 안정적인 객체 중심 3D 기하 evidence를 만들고, governed KnowledgeBank로 장기 조작 경험 재사용을 품질·신뢰·라이프사이클·충돌 메타데이터까지 포함해 제어합니다. 특히 GeoFuse-MV3D는 외부 기하 추정을 “직접 재구성”이 아니라 geometry-prior로 취급하며, 관측 마스크와 함께 보수적으로 융합해 downstream 계획에 필요한 기하만 안정화합니다.

- **Technical Challenges**: 첫째, multi-view라도 기하 prior가 마스크와 불일치할 때 환각을 줄여야 하고, 동시에 다운스트림이 민감한 색/불투명/모양 드리프트를 피해야 합니다. GeoFuse-MV3D는 입력 view 마스크로 prior를 검증하고 soft visual-hull 지원, 축(axis)-wise refinement, geometry-only 보수적 fusion을 통해 기하만 조정한 뒤 appearance는 보존하는 방식으로 이를 해결합니다. 둘째, 기억은 의미적으로는 맞아도 현재 장면에 안전하게 적용되지 않을 수 있으므로, admission·retrieval·충돌 처리·승격/요약/폐기까지 포함한 “governed” 메모리 스키마와 verifier 기반 품질 점수를 도입해 재사용 신뢰성을 높였습니다.

- **Empirical Impact**: 실험에서 GeoFuse-MV3D는 GSO-30에서 MV-SAM3D 대비 CD와 LPIPS를 각각 2.20%, 2.02% 낮추고 PSNR과 SSIM을 각각 2.36%, 1.03% 높여 다중 시점 객체 재구성 품질을 개선했습니다. KnowledgeBank는 Terminal-Bench 2.0와 SWE-Bench Verified에서 ReasoningBank 대비 Terminal-Bench SR은 4.53%, SWE-Bench resolve rate은 3.73% 향상시키면서 AS는 각각 4.95%, 5.65% 감소시켜 장기 조작 경험이 “안전하게” 재사용될 때 성능이 오른다는 점을 입증했습니다. 전반적으로 계층형 VLA에서 planner 입력(객체 기하 evidence와 메모리 거버넌스)을 강화하면 장기 계획의 안정성이 올라간다는 실증적 메시지를 제공합니다.



### Theoretical Grounding of Out-Of-Distribution Detection With Reinforcement Learning Optimizer (https://arxiv.org/abs/2606.17477)
- **Prior Approaches**: 기존 out-of-distribution (OOD) detection은 정적 배포 가정을 두고, maximum softmax probability, Mahalanobis distance, energy-based score 같은 사후(post-hoc) 점수 또는 outlier exposure 등으로 접근하는 경우가 많았습니다. test-time adaptation(예: entropy minimization)은 라벨 없는 데이터로 즉시 성능을 맞추지만, 매 업데이트가 이후 환경에서 OOD 분리 성능을 어떻게 훼손/개선하는지까지는 명시적으로 다루지 못했습니다.

- **Core Contribution**: 이 논문은 dynamic open-world 환경에서 “현재-step 최적화가 미래 OOD 행동을 망가뜨릴 수 있다”는 문제에 대해, 미래 semantic OOD false positive rate를 줄이는 방향으로 업데이트 궤적을 설계하는 프레임워크를 제안합니다. TD learning으로 학습한 value function의 gradient를 표준 gradient descent(GD)에 correction term으로 더해, 향후 누적 OOD 성능을 고려한 RL-guided optimizer를 만듭니다.

- **Technical Challenges**: 핵심 난관은 (1) hard한 FPR을 직접 최적화하기 어렵고 (2) 파라미터 업데이트가 시간에 따라 바꾸는 future-domain 일반화 오차를 이론적으로 분해·비교해야 한다는 점입니다. 논문은 에너지 기반 점수에서 pseudo-OOD를 만들고 soft FPR surrogate로 미분 가능하게 만든 뒤, temporal error decomposition을 통해 환경 변화로 인한 항과 모델 업데이트로 인한 항을 분리하고, value gradient가 future-domain 일반화 오차를 줄이는 방향으로 정렬된다는 조건 하에서 개선을 보장합니다.

- **Empirical Impact**: 이론적으로 RL-guided optimizer는 GD 대비 future-domain generalization error와 semantic-OOD FPR을 낮추며, 그 이점이 적응 step이 누적될수록 커질 수 있음을 unified main result로 정리합니다. 또한 head-only one-layer transformer 같은 구체 모델에서 gradient-conflict 조건이 label alignment과 feature-space 상관에 대한 확인 가능한 형태로 바뀌는 등, 분석이 해석 가능하도록 구성했다는 점에서 의미가 큽니다.



### StereoFactory: A Unified Merging Framework for Robust Stereo Matching (https://arxiv.org/abs/2606.17475)
- **Prior Approaches**: 스테레오 매칭은 대규모 데이터로 학습한 foundation-style 모델 덕분에 zero-shot 일반화가 좋아졌지만, 새 데이터가 추가될 때마다 전체를 다시 학습해야 해 확장성이 떨어진다는 문제가 있습니다. 이를 보완하려는 model merging은 사전 학습된 체크포인트를 weight space에서 결합해 joint retraining 부담을 줄이지만, 기존 방식은 대부분 모든 모델을 합치거나(merge-all) greedy로 점진 포함해 harmful task-vector interference를 완전히 피하기 어렵습니다. 또한 병합의 기준을 전 네트워크 파라미터에 전역(global) 가중치로 두는 경향이 있어 모듈별로 선호하는 지식 소스가 다를 수 있다는 점을 충분히 반영하지 못합니다.

- **Core Contribution**: 이 논문은 StereoFactory라는 2단계 coarse-to-fine 진화적 프레임워크로 “무엇을(어떤 모델/태스크 벡터) 병합할지”와 “어떻게(모듈별 라우팅) 병합할지”를 분리해 최적화합니다. Stage 1에서 genetic algorithm으로 유용한 모델 부분집합을 먼저 골라 interference가 생길 가능성을 줄이고, Stage 2에서 CMA-ES로 모듈 단위의 라우팅 가중치/스케일을 학습해 모듈-level knowledge specialization을 활용합니다. 결과적으로 post-hoc로도 체크포인트 풀만으로 더 강한 스테레오 성능을 얻는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 (1) 부분집합 선택이 조합폭발하는 이산(combinatorial) 문제이고 (2) subset 멤버십에 대해 미분 불가능해 gradient 기반 탐색이 어렵다는 점입니다. 논문은 Stage 1에서 변 길이 염색체를 쓰는 genetic algorithm로 부분집합을 탐색하고, 각 후보의 fitness는 가벼운 validation-forward로 정의해 지역최적/순차결정 문제를 완화합니다. 또 다른 난제는 모듈별로 서로 다른 지식 소스가 유리하다는 점이며, Stage 2에서 CMA-ES가 block-specific routing logits과 선택적 모듈 스케일을 동시에 다루도록 설계해 모듈별 선호를 반영합니다.

- **Empirical Impact**: 두 개 아키텍처(NMRF, FoundationStereo)와 네 개 벤치마크에서 StereoFactory는 동일 체크포인트 풀 조건 하에 기준선과 기존 병합/간섭 완화 방법을 능가하는 4-benchmark average를 일관되게 달성합니다. 구체적으로 NMRF는 평균 오차를 3.80→3.30, FoundationStereo는 2.88→2.19로 낮췄고, joint retraining 대비 post-hoc 검색 비용은 2.7–3.7% 수준(예: NMRF 48.2시간 대비 1.8시간)입니다. 분석에서는 지식 기여가 모듈별로 내재적으로 다르며, 선택된 부분집합이 아키텍처 간에도 최소한의 성능 저하로 전이됨을 보여 Stereo merging이 “전역 평균” 문제가 아니라 “선택+라우팅” 문제라는 주장을 경험적으로 뒷받침합니다.



### WeaveLA: Event Driven Cross-Subtask Latent Memory Weaving for Repetitive Robot Manipulation (https://arxiv.org/abs/2606.17463)
- **Prior Approaches**: 단일 step 조작에서 강점을 보인 Vision-Language-Action(VLA) 정책은 짧은 윈도우에 의존하지만, 반복 과제처럼 하위 태스크 간 의존성이 있는 상황에서는 직전 결과를 다음 단계에 전달하는 구조가 부족해 취약해진다. 기존 memory-augmented 변형은 프레임마다 write하거나, 데모 단계 기반 retrieval을 하거나, 서브목표 이벤트에서만 신호를 내보내지만 ‘다음 action expert로의 명시적 핸드오프’를 제대로 수행하지 못한다.

- **Core Contribution**: 이 논문은 cross-subtask 정보 전달의 자연스러운 타이밍 단위로 ‘sub-goal completion event’를 제안하고, 이 이벤트마다 완료된 구간을 압축해 다음 서브태스크의 action 생성 경로에 직접 주입하는 WeaveLA를 제시한다. WeaveLA는 frozen VLA 백본 위에 가벼운 cross-subtask latent memory 인터페이스를 얹어, 기존 정책의 short-window 입력 인터페이스는 그대로 유지하면서도 다음 단계 의존성을 해결한다.

- **Technical Challenges**: 핵심은 (1) 경계 정보를 프레임 단위로 계속 저장할 때의 비용과 불안정성을 피하면서, (2) 데모-time retrieval처럼 롤아웃 진행도에 맞춘 키잉을 보장하기 어렵다는 문제를, (3) action-level 실행 단계로 정보가 희석되지 않게 전달하는 것이다. 해결책으로 WeaveLA는 sub-goal completion event에서 query-driven attention pooling으로 구간을 8개의 latent tokens로 압축한 뒤, action expert 내부의 AdaRMS 모듈에 memory-conditioned 컨텍스트로 라우팅하며, 학습 안정화를 위해 flow-matching 기반의 단계적 학습 커리큘럼을 사용한다.

- **Empirical Impact**: RoboMME에서 π0.5 backbone을 사용한 stratified 평가 결과, 단일 실행(N=1) 성능은 약 100% 수준으로 유지되는 반면 반복이 필요한 구간에서만 이득이 나타났다. 특히 SwingXtimes에서 N=3일 때 success가 0%에서 47.8%로 크게 상승했으며, 어려운 반복/시간 의존 태스크들에서만 성능이 집중적으로 개선되어 ‘필요한 곳에만’ 작동한다는 메커니즘 정합성이 확인된다.



### Contact-Based Fringe Projection Profilometry for High-Resolution 3-D Surface Measurement of Reflective and Transparent Objects (https://arxiv.org/abs/2606.17438)
- **Prior Approaches**: GelSight 계열 photometric-stereo 기반 tactile sensing은 밀집 접촉 형상을 주지만, RGB 조명 기반으로 z축의 절대 깊이를 직접 재구성하지 못하고 법선(기울기)을 적분해 깊이를 추정한다. 이 과정에서 누적 오차가 생기고, 센싱 영역이 커질수록 LUT 보정 등 캘리브레이션이 어려우며, 고반사·투명 물체에서는 깊이 정확도가 흔들린다. DFP(Digital Fringe Projection) 쪽도 구조광은 정밀하지만, 반사/투과로 인해 프린지 패턴이 포화·왜곡·미부착되는 문제가 있어 금속·투명 재질에서 phase retrieval이 실패하기 쉽다.

- **Core Contribution**: 이 논문은 GelSight의 “변형 가능한 실리콘 접촉면” 아이디어를 접목해, DFP의 프린지 패턴을 실리콘의 코팅된 면에 투사하고 실리콘 변형을 통해 3-D를 triangulation으로 복원하는 contact-DFP 파이프라인을 제안한다. 목표는 접촉 영역 전체에 대해 픽셀 단위의 조밀한 3-D 표면 기하를 얻고, 큰 센싱 면적에서도 캘리브레이션을 단순화하며, 고반사·투명 물체에서 depth 정밀도와 안정성을 높이는 것이다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 대상 물체가 아니라 실리콘 코팅면에서 프린지를 안정적으로 확보해야 하며, (2) 절대 위상(unwrapping)을 신뢰성 있게 구해 카메라-프로젝터 대응을 1:1로 유지해야 한다는 점이다. 이를 위해 실리콘 경도(Shore A)와 회색 반사 코팅(알루미늄 플레이크 비율)을 실험적으로 최적화하고, 카메라-프로젝터 stereo calibration 후 N-step phase-shifting(가로·세로)과 Gray coding 기반 위상 복원, 그리고 위상 맵에 대한 국소 Gaussian filtering을 조합해 모서리/고곡률에서의 불안정성을 줄였다.

- **Empirical Impact**: 실험에서는 (a) 기존 DFP(실리콘 미사용) 대비 금속 너트/금속 볼과 투명 LEGO 블록에서 프린지 손실·불완전 재구성을 크게 줄이며, 접촉면 전역 full-field 재구성이 가능함을 정성적으로 보였다. 정량 평가는 투명 아크릴 로드의 단면 폭을 반복 측정해 vernier caliper 기준과의 일치 및 반복성을 확인했고, 구형(연결된 두 개의 8 mm 스피어) 접촉에 대해 sphere-fitting 기반 오차 지표(RMSE/MAE/P95 등)로 GelSight Mini와 비교해 structured-light 기반 3-D의 정확도·안정성 개선을 입증한다. 또한 불확실성 분석을 포함해, 광학적으로 다양한 재질에서 메트롤로지 관점의 신뢰 가능한 복원을 제공한다는 점에서 tactile 3-D 계측 분야에 실용적 의미가 있다.



### Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos (https://arxiv.org/abs/2606.17437)
- **Prior Approaches**: 기존 연구는 주로 단일 프레임(이미지 레벨) 또는 소규모 데이터에서의 분류에 집중해 왔고, 최근에는 2-stream/CNN-LSTM, 3D CNN, Transformer 등으로 video-level fusion을 확장했다. 하지만 공개 데이터셋은 규모·뷰 커버리지가 작아 재현성과 공정 비교가 어렵고, 다양한 연구들이 뷰 정의·분할·평가지표를 달리해 현대 아키텍처를 체계적으로 벤치마킹하기가 힘들다. 또한 일부 뷰는 공간적 외형이 매우 유사해 단일 프레임 특징만으로는 구분이 어렵고, 초음파 영상의 프레임 품질이 들쭉날쭉해 신뢰도 없는 샘플이 시간적 집합 과정에서 오염될 수 있다.

- **Core Contribution**: 이 논문은 EV9V(Echocardiographic Videos of Nine Views) 데이터셋을 공개하며, 표준 9개 심초음파 뷰를 총 5,138개 비디오(910,579 프레임)로 제공해 공개형 중 최대 규모로 포지셔닝한다. 더불어 EV9V 위에서 CNN/RNN/Transformer를 포함한 대표 video classification 모델들을 체계적으로 벤치마크해, 최신 비디오 아키텍처의 성능과 한계를 같은 기준에서 비교할 수 있게 했다. 마지막으로 STFM(Spatio-Temporal Fusion Model)을 제안해 공간(해부학)과 시간(심장 동역학)을 효율적으로 결합하고, 프레임 품질 변동에 강한 추론을 목표로 한다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 유사 외형 뷰를 구분하기 위해 공간-시간 정보를 안정적으로 융합해야 하고, (2) 흐림·과도한 움직임·전이 구간처럼 비대표 프레임이 존재할 때 이를 비디오 수준 의사결정에서 덜 반영해야 한다는 점이다. 논문은 공유 얕은 CNN stem 위에 spatial branch(중심 프레임의 해부학 임베딩)와 temporal branch(희소 샘플 클립을 CNN-LSTM으로 처리)를 두는 이중 스트림 구조로 계산 효율을 유지하면서도 시간 패턴을 학습한다. 여기에 Re-EDL 기반 evidential(Dirichlet) 불확실성 모델링과 증거(evidence) 기반 fusion을 결합해, 불확실성이 큰 관측이 최종 융합에서 상대적으로 덜 영향 주도록 설계했다.

- **Empirical Impact**: EV9V에서 다양한 현대 비디오 아키텍처와의 비교 실험을 통해 STFM이 여러 모델 전반에서 경쟁력 있는 성능을 보이며, 불확실성 인지 spatio-temporal 학습이 심초음파 뷰 분류의 견고성에 실제로 기여함을 확인한다. 특히 비디오 내 품질 편차가 큰 임상 상황에서, 불확실성을 활용한 샘플 선택 및 증거 융합 전략이 비대표 구간의 오염을 줄여 분류 신뢰도를 높이는 방향으로 효과가 나타난다. 코드 공개까지 병행해, 이후 연구자들이 EV9V 기반으로 재현 가능한 벤치마킹과 후속 모델 개선을 빠르게 수행할 수 있는 기반을 제공한다.



### UoU: A Universal Fingerprint Foundation Model Based on Large-Scale Unsupervised Learning (https://arxiv.org/abs/2606.17436)
- **Prior Approaches**: 지문(fingerprint) 인식은 보통 enhancement, structural parsing, alignment(정렬), matching을 파이프라인으로 쪼개 각 단계별로 따로 최적화해 왔습니다. 이 방식은 특정 조건에서는 강하지만 센서/품질/다운스트림 태스크가 바뀔 때 표현이 잘 재사용되지 않아 모듈 간 경계가 취약해질 수 있습니다. 또 기존 딥러닝은 minutiae·orientation·정렬·디포메이션 같은 부분 성능을 크게 올렸지만, 이를 하나의 transferable foundation-model 관점으로 통합하진 못했습니다.

- **Core Contribution**: 논문은 UoU(대규모 비지도 학습 기반 universal fingerprint foundation model)를 제안하며, 지문 특징 추출을 도메인 전용 foundation-model 문제로 재구성합니다. 핵심은 하나의 reusable backbone이 이미지 복원부터 orientation/구조 필드, semantic token, point-level 생체점(코어/델타/미뉴티아), 그리고 글로벌 디스크립터까지 계층적 표현을 제공하게 만드는 것입니다. 또한 cold start(정확한 정답으로 초기화)→weak supervision 확장→large-scale unsupervised consolidation을 반복하는 데이터/학습 레시피로 공통 표현의 범용성을 노립니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 지문이 일반적인 텍스처가 아니라 방향성·주기적 능선 패턴·희소한 생체점·변형/공간 대응 같은 중간 구조를 갖는다는 점에서, 이런 구조적 정합성을 학습 과정에 일관되게 반영해야 한다는 것입니다. UoU는 orientation flow, 주기적 ridge, sparse biometric entities, 공간 equivariance 같은 도메인 특이 대칭/불변성 제약을 학습에 녹이고, 쿼리 기반 set prediction으로 변동 개수 구조를 안정적으로 매칭(Hungarian matching)하도록 설계했습니다. 더 나아가 architecture-agnostic하게 backbone을 중심 자산으로 두고, 현 구현은 transformer 기반 구조 예측 분기만 초기 앵커로 제공하는 형태로 확장성을 확보했습니다.

- **Empirical Impact**: UoU는 “단일 태스크용 모델”이 아니라 enhancement·정렬·등록·매칭·스푸핑 대응 등 여러 적용을 같은 표현 위에서 끌어오려는 실증/검증 프로토콜을 제시합니다. 특히 weakly supervised로 의미(semantic) 커버리지를 넓히고, unsupervised consolidation로 correspondences와 invariances, 표현 기하를 안정화해 실제 배치 조건(부분 지문, 저품질, 센서 차이)에 대한 전이 가능성을 목표로 합니다. 저자들은 초기 구현 일부를 공개하며, 향후 다양한 head와 모델 스케일로 다운스트림 특화가 가능하다는 점에서 지문 분야의 foundation-model 전환을 가속할 잠재력이 큽니다.



### LADBench: A Benchmark for Logical Fault Detection in Images (https://arxiv.org/abs/2606.17433)
Comments:
          Accepted to the IEEE International Conference on Development and Learning (ICDL 2026)

- **Prior Approaches**: 기존 이상(anomaly) 벤치마크들은 주로 외형적 실수나 시간적 오류 같은 더 ‘관찰 가능한’ 결함을 다루거나, 직접적인 지시로 모델을 테스트하는 경향이 컸습니다. 논리 추론을 평가하는 작업도 존재하지만(예: 제약 기반 LogicQA, 실사진 기반 Salbench류), 물리/사회 상식의 맥락에서 ‘논리적 이상’을 찾아내는 능력을 세밀하게 측정하긴 어려웠습니다.

- **Core Contribution**: 이 논문은 LAD-bench라는 벤치마크를 제안해, 실제 배포에서 요구되는 ‘논리적 이상(logical anomaly)’을 체계적으로 평가합니다. 4개 도메인(Residential, Urban, Collaborative, Nature)에서 1,000장 이상 합성 이미지를 만들고, 각 이미지에 대해 물리적으로 불가능하거나 일상적 상식에 비해 크게 덜 그럴듯한 장면을 정답으로 라벨링합니다.

- **Technical Challenges**: 핵심 난제는 오픈월드에서 요구되는 ‘암묵적 논리 결함 탐지’를, 단순 이진 정확도보다 더 공정하게 측정하는 평가 설계였습니다. 이를 위해 Tiered Prompting Protocol(Zero-shot→Awareness→Context hint)을 단계적으로 적용하고, 정답을 맞혀도 힌트를 많이 쓸수록 감점되는 decay-weighted scoring과 LLM-as-a-judge(gpt-5-nano) + 소량 human-in-the-loop 검증 파이프라인을 결합했습니다.

- **Empirical Impact**: 실험 결과, 최고 성능 모델도 전체 decay-weighted 정확도 70.11%에 그쳤고, 특히 더 얕은 단계에서의 탐지는 낮은 반면 힌트를 줄 때만 성능이 오르는 경향이 관찰됐습니다. 또한 어떤 모델은 힌트가 주어졌을 때 정상 이미지에서도 이상을 ‘환각’하는 부작용이 나타났으며, 오픈소스 모델들의 성적이 상용 대비 크게 낮아 실제 엣지/가정용 로봇 배포 전 논리 신뢰성 검증의 필요성을 강조합니다.



### Visual Retrieval-Augmented Generation for Silhouette-Guided Animal Ar (https://arxiv.org/abs/2606.17431)
Comments:
          SOICT 2025

- **Prior Approaches**: 기존의 shape-based retrieval은 Shape Context/IDSC 같은 기하 기반 디스크립터부터 learning 기반 매칭까지 발전했지만, 고도로 복잡한 자연 실루엣에서는 기하적 정합이 깨지기 쉽고 대규모 라벨 의존 문제도 남아 있었다. 한편 ControlNet 같은 구조 조건부 생성은 실루엣을 잘 그리지만 스스로 모호성을 ‘해석’하는 창의적 역할은 제한적이다. 또한 컴퓨테이셔널 pareidolia 연구는 특정 범위에 머물러, 열린 형태의 자연 입력에서 창의적 파트너처럼 작동하기 어렵다.

- **Core Contribution**: 이 논문은 자연 실루엣에서 동물 예술을 생성하는 Visual-RAG를 제안한다. 핵심은 retrieval(구조적으로 유사한 동물 실루엣/참조 이미지 탐색)로 예시 기반 영감을 제공하고, diffusion 생성 과정은 ControlNet과 IP-Adapter로 입력 실루엣 제약과 외형(appearance) 전이를 동시에 수행한다. 또한 28,586개의 동물 실루엣 코퍼스를 구축해 retrieval 실험의 기반을 마련했다.

- **Technical Challenges**: 가장 큰 과제는 ‘구조적으로 맞는’ 참조를 찾는 동시에, 생성 단계에서 실루엣 정합이 무너지지 않게 하는 것이다. 논문은 Shape Context 기반 매칭에 더해 RANSAC 기하 검증으로 top-10 후보를 재순위화하고, 스케일·회전·이동 정규화 및 180도 방향 표준화까지 포함한 shape standardization을 적용해 정합 품질을 끌어올린다. 특히 표준화 제거 시 inlier ratio가 13.4%로 급락해, 구조적 충실도가 downstream 생성 신뢰성을 좌우함을 보인다.

- **Empirical Impact**: ablation과 비교 실험에서 RANSAC은 학습 기반 point cloud 등록 대비 더 낮은 residual error와 높은 post-alignment IoU로 정밀 정렬에 유리함을 보였다. 또한 O2O 대비 retrieval 자체는 느릴 수 있으나, 생성에 결정적인 단일 참조의 기하 충실도가 더 높게 나와 최종 성능에서 이득이 확인된다. 사용자 연구(12명, Likert 5점)에서는 전반적으로 중립점(3.0)보다 낮은 평균(심미/실루엣 충실/인상)이 보고됐지만, shape fitness와 aesthetics 간 상관(r≈0.75)이 높아 ‘실루엣에 대한 충실도’가 pareidolia의 체감 창의성을 좌우한다는 인사이트를 제공한다.



### CIAN: Multi-Stage Framework for Event-Enriched Image Captioning via Retrieval-Augmented Generation (https://arxiv.org/abs/2606.17430)
Comments:
          SOICT 2025

- **Prior Approaches**: 기존 이미지 캡셔닝은 템플릿·리트리벌 방식에서 시작해 encoder–decoder 및 Transformer 기반으로 발전했지만, 대체로 픽셀에 갇혀 사건의 의미·참여자 역할 같은 맥락을 놓치기 쉽습니다. 멀티모달 LLM은 문단 수준 서사를 만들 수 있지만, 외부 지식이 필요한 경우 retrieval 메커니즘이 없으면 캡션이 얕아지는 문제가 남습니다. 리트리벌 연구도 주로 단일 문장/텍스트 정합에 초점이 많고, 문서 단위 맥락을 캡션에 단단히 접목하는 방식은 제한적이었습니다.

- **Core Contribution**: 이 논문은 Contextual Image-Article Narrator(CIAN)라는 멀티스테이지 프레임워크로, 이미지에 관련된 뉴스 기사(외부 비시각 지식)를 캡션 생성에 연결합니다. SigLIP로 기사를 검색한 뒤 요약을 통해 Narrative Generation을 유도하고, 마지막에 N-Gram 기반 Refinement로 문장 유창성과 인간 참조에 대한 정렬을 강화합니다. 결과적으로 “보이는 것+사건 맥락”을 동시에 담는 이벤트-풍부 캡션을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 (1) OpenEvents-V1 같은 데이터셋의 이미지-기사 정합이 약해 retrieval-augmented 생성이 노이즈에 흔들릴 수 있다는 점, (2) 생성한 캡션이 CIDEr 같은 n-gram 기반 평가에 맞게 다듬어져야 한다는 점입니다. 이를 위해 논문은 “hide caption” 마커 기반 웹 크롤링으로 고품질 캡션 쌍을 추가하고 SigLIP 정합으로 재배치해 데이터 신호대잡음을 줄였습니다. 또한 Qwen2.5-VL에 LoRA fine-tuning을 적용해 스토리텔링 지침 프롬프트로 1차 캡션을 만든 뒤, 데이터에서 계산한 고빈도 1~3-gram 렉시콘을 소프트 제약으로 반영하는 refiner로 fluency와 metric 정렬을 동시에 노립니다.

- **Empirical Impact**: OpenEvents-V1에서 CIAN은 리트리벌 mAP 0.979, Recall@1 0.969, Recall@10 0.996의 높은 성능을 보이며, 캡셔닝 CIDEr은 0.030에서 0.094로 크게 향상되었습니다. CIDEr 민감도 특성을 고려한 단계적 refinement의 효과는 ablation에서 확인되었고, 최종 개선은 N-gram 기반 단계에서 가장 크게 나타났습니다. 또한 SigLIP를 다른 비전 인코더(CLIP/BLIP/DINOv2) 대비 비교한 결과 mAP 등 전 지표에서 우수해, retrieval에 적합한 시각 표현 학습의 선택이 실증적으로 뒷받침됩니다.



### Impact of Hand Impairment and Occlusions on Hand Pose Estimation Accuracy in Augmented Reality Applications (https://arxiv.org/abs/2606.17427)
- **Prior Approaches**: 기존 연구들은 HoloLens 2 같은 AR HMD의 손 자세 추정 정확도를 주로 무해(비장애) 사용자 기준으로 마커 기반 ground truth와 비교해 왔고, 오차가 수~수십 mm 범위로 보고돼 왔다. 다만 손 impairment(예: cSCI)나 실제 물체 상호작용 중 가림(occlusion)이 자세 추정에 미치는 영향은 충분히 정량화되지 않았으며, 특히 AR HMD 예측과 최신 pose estimation 알고리즘의 직접 비교도 거의 없었다.

- **Core Contribution**: 이 논문은 cSCI 환자와 비장애 대조군이 실제 물체(opaque/clear)를 집고 들어 올리는 동적 과제에서, HoloLens 2 온보드 손 추적과 WiLoR·HaMeR·WildHands·MediaPipe 같은 state-of-the-art 방법들의 3D 관절 오차를 같은 ground truth로 비교한다. 결과적으로 cSCI와 비장애군 간 전반적 추정 정확도 차이가 관찰되지 않아, 손 기능 저하가 있어도 AR 기반 재활 애플리케이션의 자세 추정이 일반화될 가능성을 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 손 impairment로 인한 개별적인 손 모양·그립 패턴 변화, (2) 물체에 의한 시야 가림이 egocentric 관점의 손가락 관절 가시성을 깨뜨리는 문제, (3) HMD 관점 기록과 오프라인 pose estimation 비교를 위한 동기화·정확한 3D ground truth 수립이었다. 연구진은 다중 카메라 triangulation으로 3D joint ground truth를 만들고, HoloLens 2 예측과 각 알고리즘의 3D 추정을 동일 프레임에서 PA-MPJPE로 정렬·비교했으며, opaque/clear 조건을 동일 물체로 구성해 가림 효과를 더 명확히 분리했다.

- **Empirical Impact**: 실험 결과, clear 물체가 opaque보다 오차를 약간 줄였지만 격차는 매우 작았고(약 0.1 mm), 전반적으로 cSCI가 비장애군보다 더 나쁘게 추정된 양상은 확인되지 않았다. 한편 WiLoR과 HaMeR이 HoloLens 2 온보드 예측보다 더 정확했으며(대략 2 mm 수준 차이), WildHands는 impairment 수준이 높을수록 오차가 증가하는 상관이 나타났다. 연구진이 공개한 데이터셋은 hand-impaired 인구에서 자세 추정 성능을 더 다듬는 데 활용될 수 있다는 점에서 재활용 실감형 인터페이스 개발에 직접적인 의미가 있다.



### Enhancing Pathological VLMs with Cross-scale Reasoning (https://arxiv.org/abs/2606.17412)
- **Prior Approaches**: 기존 병리 비전-언어 모델(VLM)들은 대부분 단일 배율 고정 이미지에서의 VQA/분류로 학습·평가돼 임상에서의 다배율 추론 흐름을 반영하지 못했습니다. 또한 멀티스케일 데이터가 있더라도 ‘저배율 조직 아키텍처’와 ‘고배율 세포 형태’를 연결해 근거를 통합하는 명시적 목표가 약해, 모델이 배율별 단독 인식에 머무는 문제가 컸습니다. 더 나아가 멀티이미지 VQA를 그대로 만들면 text-only shortcut(질문·보기만으로 정답 추론)을 통해 과대평가될 위험도 제기됩니다.

- **Core Contribution**: 이 논문은 병리 해석을 다배율(10x/40x/200x) 근거를 함께 엮는 multi-magnification reasoning으로 재정의한 ‘cross-scale training and evaluation paradigm’을 제안합니다. 그 결과 Scale-VQA라는 다배율 추론 벤치마크를 구축하고, ScaleReasoner-R1을 cross-scale VQA에 맞춰 학습해 단일 배율 벤치마크 성능까지 함께 끌어올리는 전이를 보여줍니다.

- **Technical Challenges**: 핵심 난제는 멀티이미지 VQA에서 발생하는 text-only shortcut인데, 배율과 강하게 연동된 단서나 문장/보기의 언어적 priors가 정답을 이미지 없이도 맞히게 만들 수 있습니다. 이를 위해 leakage-aware curation pipeline을 설계했으며, 텍스트만으로 정답을 맞히는지 Gemini 3 Pro·Qwen3-Max 같은 text-only adversary로 반복 스크리닝하고, 근거가 최소 두 배율 뷰에 의존하도록 제약(visual-grounding·scale-dependency·차원별 다양성)을 교정해 누설을 억제합니다. 이후 RL 학습은 GRPO로 outcome-driven 보상(정답 정확도) 중심으로 구성해, 인간이 쓴 rationales를 그대로 모방하는 방식의 과적합(특히 SFT 계열의 단일 스케일 망각)을 줄이려 했습니다.

- **Empirical Impact**: Scale-VQA는 2,537개 병리 이미지에서 4,685개의 MCQ를 만들며, 배율 간 증거 통합을 요구하는 설계로 cross-scale 추론을 직접 측정합니다. ScaleReasoner-R1은 Scale-VQA-Test에서 평균 82.89%로 cross-scale SOTA를 달성했고, Scale-VQA만으로도 PathMMU의 단일 배율 벤치마크 성능이 전반적으로 개선되는 전이를 보였습니다. 특히 RL-only가 SFT-only 및 SFT+RL보다 일관되게 우수했는데, 이는 SFT의 demonstration-style 과적합 대신 정답 보상 기반 최적화가 일반화를 이끈다는 메시지를 강화합니다.



### Attention Alignment Between Humans and Vision-Language Models (https://arxiv.org/abs/2606.17410)
- **Prior Approaches**: 기존 연구는 학습 분포가 생체(인간/영장류) 비전 정렬(alignment)에 큰 영향을 준다고 보거나, 특정 아키텍처(예: CNN vs ViT)가 시각피질 반응과의 맞춤을 달리 만든다고 보고해 왔습니다. 다만 대다수는 bottom-up(시각 표현) 쪽에 초점이 있었고, top-down(목표 기반 시선/공간 우선순위)을 아키텍처 편향으로 얼마나, 또 어떻게 조절하는지와 그 상호작용은 분리해 검증하기 어려웠습니다.

- **Core Contribution**: 이 논문은 vision-language model에서 인코더(바텀업)와 디코더(탑다운)를 요인 설계로 분리해, 모델의 spatial attention 지도가 인간 fixation heatmap을 얼마나 잘 “어디에” 맞추는지 정량화합니다. 특히 디코더 아키텍처가 spatial alignment를 압도적으로 좌우하며(40–50%p 범위), 인코더 차이는 상대적으로 5–20%p 수준의 2차 효과라는 결론을 제시합니다. 또한 fixation 정렬이 신경 예측 능력과 1:1로 대응하지 않는다는 점을 함께 보여줍니다.

- **Technical Challenges**: 핵심 과제는 서로 다른 모델들이 동일한 학습 분포를 공유하도록 통제하면서, attention 지도를 인간 시선과 공정하게 비교하고 디코더/인코더 편향을 분리하는 것입니다. 이를 위해 2×2 factorial로 ResNet-101 vs ViT-B/16 인코더와 LSTM(soft additive attention gate) vs Transformer 디코더를 결합해 2가지 과제에서 EyeLink 기반 fixation heatmap과의 Spearman 기반 정렬을 비교했고, 추가로 attention ablation(반쪽 시야 무시 유사)과 TRIBE v2 기반 합성 신경반응 예측(encoding model, variance partitioning)까지 수행했습니다.

- **Empirical Impact**: 실험 결과 LSTM 디코더는 노이즈 천장 대비 80–87% 수준의 alignment를 보인 반면 Transformer 디코더는 40–59%로 더 낮았고, Molmo 7B-D와 Qwen3.5 9B는 그 사이에 위치했습니다. 그러나 LSTM은 fixation 위치는 잘 맞추되 attention 지도가 공간적으로 퍼지고 과제별 분화가 약한 반면, ViT-Transformer는 덜 맞추더라도 더 날카롭게 집중하고 과제 구분이 강했습니다. 더 나아가 신경 예측에서는 fixation 정렬이 높은 모델이 항상 유리하지 않았고, CNN-Transformer attention map이 합성 뇌 신호(특히 early visual cortex) 예측을 더 잘하는 ‘dissociation’가 관찰되며 top-down/waswo 불일치를 시사합니다.



### Graph Neural Networks for Semi-Supervised Image Classification with Multi-Feature Aggregation (https://arxiv.org/abs/2606.17406)
- **Prior Approaches**: 기존 반지도/비지도 이미지 분류 연구는 라벨이 적을 때 성능을 내기 위해 특징 추출기(CNN, Vision Transformer 등)와 그래프 기반 GNN(주로 GCN)을 결합해왔다. 다만 이미지 데이터에서는 그래프가 사전에 주어지지 않아, 그래프 구성(유사도 계산·kNN·재랭킹)의 품질이 결과를 크게 좌우한다. 또한 서로 다른 특징 추출기에서 나온 표현을 어떻게 효과적으로 합칠지(early/late fusion 또는 랭킹 통합)는 여전히 결합 전략 의존성이 크다는 한계가 있다.

- **Core Contribution**: 이 논문은 라벨이 scarce한 반지도 이미지 분류에서, 여러 특징 추출기에서 얻은 feature와 그래프 표현을 동시에 통합하는 GNN 접근을 제안한다. 특히 multi-feature setting에서 각 특징의 랭킹을 UDLF 기반 rank aggregation으로 합쳐 하나의 reciprocal kNN 그래프를 만들고, node feature는 Unsupervised Relief(URelief)로 저차원 성분을 뽑아 연결한다. 추가로 manifold learning 계열의 재랭킹/유사도 학습을 그래프 전처리로 적용해 분류 정확도를 전반적으로 끌어올리는 것을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 특징 공간에서의 유사도를 어떻게 일관된 그래프 구조로 바꿀지, (2) 그래프가 부정확하면 GNN 전파(smoothing)가 망가질 수 있다는 점, (3) 고차원 feature를 그대로 쓰기 어렵다는 계산·과적합 문제다. 논문은 ranked list를 먼저 만들고 BFSTree, RDPAC, LHRR 같은 manifold learning/UDLF 재랭킹을 거쳐 그래프 품질을 개선하며, multi-feature에서는 reciprocal 제약으로 그래프에서의 상호 일치성을 강화한다. 또한 각 descriptor에 URelief로 200개 특징만 선택해 저차원으로 줄인 뒤 concatenation으로 fused node feature를 구성한다.

- **Empirical Impact**: 실험 결과, manifold learning 기반 그래프 처리와 함께 feature·graph를 전략적으로 결합하면 대부분의 조건에서 분류 정확도가 유의미하게 향상된다. 특히 여러 추출기에서 나온 feature를 rank aggregation으로 통합했을 때 성능이 추가로 좋아지는 경향이 관찰됐다. 한편 분석에서는 GCN이 feature보다 입력 그래프 품질에 더 민감하다는 점을 보여주며, 라벨이 적은 설정에서 성능 향상의 주된 레버가 ‘좋은 그래프 구성’임을 실증적으로 뒷받침한다.



### Bridging Spatial And Frequency Views For Disaster Assessment: Benefits And Limitations (https://arxiv.org/abs/2606.17403)
Comments:
          Copyright 2026 IEEE. Published in the 2026 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2026)

- **Prior Approaches**: 재난 건물 손상 평가는 위성 RGB로부터 shape·texture 중심의 spatial-domain 네트워크(CNN, ViT 등)가 주로 이뤄졌고, xBD 같은 대규모 벤치마크가 이를 가속해왔다. 하지만 spatial만으로는 미세한 구조 변형이나 파편/붕괴로 생기는 미세한 질감 단서를 충분히 포착하기 어려워 ‘손상 없음’ 쏠림 같은 편향이 생길 수 있다. Frequency-domain(푸리에/웨이브렛 등) 단서는 고주파(에지·질감 불규칙) 신호를 강조하지만, 단독 사용 시 일반화가 약하거나 체계적 비교가 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 xView2(xBD) 사후(disaster post-event) 이미지에서 손상 다중 클래스 분류를 대상으로 spatial-only, frequency-only, dual-domain를 공정하게 비교한다. EfficientNet-B0 백본과 학습 설정을 동일하게 고정하고, 입력 표현(공간 vs 주파수)과 fusion 전략만 바꿔 도메인 정보가 성능에 미치는 기여를 통제 실험으로 분리해 보여준다. 특히 dual 중에서도 Dual Spatial(공간+주파수 퓨전)의 전반 성능이 두드러짐을 정량적으로 제시한다.

- **Technical Challenges**: frequency-domain을 위해 RGB 채널별 orthogonally normalized 2D discrete Fourier transform을 적용해 크기(magnitude) 스펙트럼(로그 스케일 선택 가능)을 입력으로 구성해야 한다. 또한 단순 퓨전이 공간/주파수의 상보성을 충분히 활용하지 못할 수 있어, 병렬 브랜치에서 특징 레벨로 융합하는 방식을 설계하고(두 가지 연결 순서 실험) 과적합을 억제하기 위해 Focal Loss with Smoothing과 공통 학습 조건을 적용했다. 그럼에도 Minor 같은 미세 손상 클래스는 시각적 애매성과 클래스 불균형 영향으로 전 모델이 낮은 F1을 보여 해결 난이도가 남는다.

- **Empirical Impact**: 실험 결과 dual-domain이 single-domain보다 일관되게 우수했으며, Dual Spatial은 테스트 정확도 0.4688과 손실 최저(0.0351)를 기록했다. 반면 macro F1-score는 Spatial-only가 0.4254로 가장 높아, 정확도 향상이 항상 클래스 균형까지 보장하지는 않음을 드러낸다. Frequency-only는 전반적으로 최악이며 과적합/일반화 실패 양상이 뚜렷했고, 모든 모델이 특히 Minor에서 부진해 재난 피해 분류에서 accuracy보다 클래스 민감 지표가 중요함을 시사한다.



### Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models (https://arxiv.org/abs/2606.17389)
Comments:
          16 pages. Accepted to the ICLR 2026 Workshop on Multimodal Intelligence. Code: this https URL

- **Prior Approaches**: 기존 연구는 VLM의 신뢰도를 “Attention-Confidence Assumption” 관점에서 해석해 왔습니다. 즉, 시각 인코더가 관련 영역에 촘촘히 주목하면 모델이 정답을 낼 “근거(grounding)”가 생긴다고 가정했죠. 하지만 이런 해석은 attention이 출력의 결정요인을 충실히 설명하는지에 대한 논쟁과 함께, 출력 기반 환각 평가(benchmark) 중심으로 보정이 부족하다는 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 VLM Reliability Probe(VRP)로 여러 VLM 계열(LLaVA-1.5, PaliGemma, Qwen2-VL)을 가로지르는 신뢰도 시그널을 체계적으로 비교합니다. 특히 시각 “structural” 지표(클러스터 수 C_k, 공간 엔트로피 H_s)와 생성 동역학 기반 지표(예: self-consistency)를 함께 상관/예측하며, reliability가 단순 attention 구조가 아니라 “생성 과정의 내부 상태 분포”에 가깝다는 결론을 제시합니다. 또한 LLaVA에서는 Early Lock(또는 Symbolic Detachment), PaliGemma/Qwen2-VL에서는 더 분산된 신뢰도 경로가 나타난다고 분석합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “attention heatmap이 신뢰도를 알려주는가?”를 상관 수준을 넘어 인과·기계적으로 분리해 증명하는 것입니다. 논문은 forward hook으로 cross-attention 및 hidden state를 수집하고, Logit Lens(레이어별 correct vs incorrect 토큰 logit 차)와 hidden-state probe, 그리고 대규모 causal ablation(특정 레이어의 예측 뉴런/부분을 파괴)까지 수행해 신뢰도 경로가 어디에 있는지 추적합니다. 그 결과 시각 attention은 구조적으로는 중요하지만 통계적으로는 정확도와 거의 무관하고, self-consistency와 hidden-state probe가 훨씬 강한 예측력을 보인다고 정리합니다.

- **Empirical Impact**: 실험에서 C_k와 H_s는 정답 여부와의 상관이 사실상 0에 가깝게 나와 “Cluster Failure”를 뒷받침합니다(예: R≈0.001, R≈-0.012). 반대로 self-consistency는 truth 예측에서 R=0.429로 시각 지표 대비 우수하며, hidden-state probe는 강한 설정에서 AUROC>0.95 같은 높은 분별력을 보입니다. 더 나아가 LLaVA는 late-stage 병목을 파괴하면 취약해지지만, PaliGemma와 Qwen2-VL은 예측에 기여하는 부분을 ~50% 이상 파괴해도 견고함을 보여 신뢰도 신호 설계가 아키텍처에 강하게 의존한다는 메시지를 강화합니다.



### TerraTransfer: Learning End-to-End Driving Policies Without Expert Demonstrations (https://arxiv.org/abs/2606.17386)
- **Prior Approaches**: 기존 end-to-end 자율주행 학습은 대체로 로그 운전자 데이터를 기반으로 imitation pretraining을 하고, 이후 fine-tuning(지도학습·open-loop RL) 또는 closed-loop RL을 추가하는 방식이었습니다. 특히 closed-loop RL은 photorealistic rendering과 대형 vision backbone 추론을 매 스텝 반복해야 해 계산비용이 커지고, 희귀·안전중요 상태는 로그에 충분히 없어서 covariate shift 문제가 남았습니다. self-play는 비용이 싸지만 픽셀 대신 vector state로 학습되는 경우가 많아 실제 raw image end-to-end로 확장되지 못했습니다.

- **Core Contribution**: 이 논문은 self-play의 “학습은 저렴한 vector 상태에서, 추론은 raw image로”라는 비대칭 이점을 결합해, demonstration 없이 end-to-end 주행을 만드는 단일 패러다임을 제안합니다. 핵심은 learning to drive(벡터 상태 기반 planning head)를 먼저 self-play로 학습한 뒤, learning to see(vision encoder)를 alignment 단계에서 맞추되 어떠한 단계도 logged trajectory를 상대로 supervise하지 않는다는 점입니다. alignment는 teacher(자기대전 self-play policy)의 action distribution과 표현 관계를 재현하도록 설계되어, 큐레이팅된 expert demonstration 없이도 전이가 가능함을 노립니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 vector state에서 학습된 teacher를 raw image 입력 공간으로 옮길 때, 단순 모듈 캐스케이드(검출/기하 복원 후 재입력)가 불필요하게 어려운 상위 복원 문제를 만든다는 점입니다. 이를 피하기 위해 teacher가 내부에서 사용하는 “풀드(pooled) 특징”과 직접 정렬하도록 설계했고, teacher 특징의 저랭크/중복성을 활용해 batch-relational low-rank structural loss로 관계(씬 간 유사성)를 주로 맞추도록 제한했습니다. 또한 action KL divergence로 학생의 정책 분포가 teacher와 일관되게 되도록 하여, 로그 경로 없이도 행동 정렬이 유지되게 했습니다.

- **Empirical Impact**: 평가는 photorealistic 3D Gaussian splatting 기반 closed-loop 벤치마크 HUGSim에서 HD-Score로 진행되며, 제안한 vision-정렬 end-to-end 정책이 imitation-trained 선행 방법들을 aggregate에서 match하거나 초과합니다. 특히 self-play teacher에 근접한 성능(집계 HD-Score 기준 teacher 대비 오차 약 0.03)을 보이면서도, paired (image, scene-state) 프레임은 약 1.83M 정도만 사용해 데이터 효율이 높다는 점이 강조됩니다. 또한 paired 데이터 비율을 줄여도 성능이 잘 유지되어, trajectory 라벨 없이도 강한 일반화와 재현성이 가능함을 실험적으로 입증했습니다.



### Improving and Evaluating Hand-Object Interaction Detection (https://arxiv.org/abs/2606.17384)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 HOI(손-물체 상호작용) 이해는 보통 손만 잘 잡거나(손 검출) 손이 쥔 물체를 별도 “held object”로 처리하는 식으로 분절돼 있었다. 또한 HOI를 예측하더라도 오래된 탐지기 의존, 또는 영상에서 필요한 시공간 일관성 평가는 부족해 성능 비교가 불완전하다는 지적이 있었다.

- **Core Contribution**: 이 논문은 HOI-DETR이라는 새로운 프레임워크를 제안해 HOI를 ‘상호작용 관계까지 포함한 역할(role) 기반 탐지’로 end-to-end 학습한다. Co-DETR에 가벼운 interaction module을 결합해 손→1st object, 1st object→2nd object의 방향성 있는 링크를 예측하며, 모델과 함께 학습/평가용 체크포인트까지 공개해 성능 격차를 빠르게 줄인다.

- **Technical Challenges**: 핵심 난제는 같은 물체라도 손과의 관계에 따라 배경/1st object/2nd object처럼 역할이 바뀌는 점이며, 이를 기존 의미론적 클래스 탐지 방식으로는 담기 어렵다는 것이다. 저자들은 role-based query 분류와 쌍(pair) 임베딩 기반 interaction head를 decoder에 추가하고, 학습 단계에서 aux one-to-many 및 focal loss로 상호작용 슈퍼비전을 강화해 단일 forward pass에서 탐지-관계를 함께 학습하도록 설계했다.

- **Empirical Impact**: 실험에서는 Hands23, HOIST, FineBio, HD-EPIC를 아우르는 4개 데이터셋 평가와 함께, HD-EPIC에서 파생한 비디오 벤치마크(HD-EPIC-HOI) 및 spatiotemporal consistency 중심의 평가를 도입해 검증을 확장했다. ablation은 구성요소별 기여를 확인했고, Hands23·FineBio에서 mAP이 20%p 이상 향상되는 등 SOTA를 크게 갱신했으며 프레임 기반 모델임에도 비디오 기반 접근을 능가하는 성과를 보였다.



### MeiBRD: Meta-Learning Intraoperative Biomechanical Residual Deformation (https://arxiv.org/abs/2606.17379)
- **Prior Approaches**: 수술 중 간(liver) 등록은 환자 자세, 복강압, 호흡, 도구-조직 상호작용 때문에 큰 연부조직 변형이 생기지만, 관측은 stylus나 iUS로 매우 희소하게 주어져 문제는 본질적으로 ill-posed입니다. 기존 biomechanical 모델은 물리 방정식으로 변형을 제약해 안정성을 얻지만, 선형 탄성 등 단순 가정 때문에 큰/비선형 변형의 예측에 편향이 남습니다. 딥러닝 기반 등록은 빠른 추론과 표현력이 장점이지만, 희소한 수술 중 데이터와 라벨 부족으로 OOD에서 일반화가 무너지고 물리적으로 그럴듯하지 않은 변형을 내기 쉽습니다.

- **Core Contribution**: 이 논문은 MeiBRD로, 수술 중 희소 대응만으로 선형 biomechanical 예측을 ‘그대로’ 학습하는 대신 residual만 보정하도록 하이브리드 프레임워크를 제안합니다. 선형 모델이 만드는 변형 오차를 그래프 신경 diffusion(예: GRAND)로 표현하되, 수술 중 측정 위치를 residual 함수의 input-output이 관측되는 context 샘플로 보고 feedforward meta-learning으로 residual 함수를 빠르게 적응시킵니다. 특히 전체 변형 필드를 end-to-end로 직접 학습하지 않아 데이터 효율성과 물리적 정합성(선형 예측에 대한 보정)을 함께 노립니다.

- **Technical Challenges**: 핵심 난제는 (1) 희소 측정이 주어졌을 때 긴 거리(long-range) 정보를 어떻게 전파해 수술 중 관측 바깥 부위의 residual을 추정하느냐, (2) diffusion 기반 잔차 함수가 간의 기하(표면/체적)에서 중요한 변형 신호를 놓치지 않게 하는 것입니다. 저자들은 3D 간 메쉬 그래프에서 GRAND diffusion을 수행하되, 표면 곡률, 체적 변화(테트라헡드 det(F)), 회전불변 변형률 등 geometry-aware attention을 통해 연부조직 변형에 민감한 특징을 연산자에 주입합니다. 또한 수술 중 adaptation을 MAML 같은 gradient-based가 아니라 context를 입력으로 하이퍼네트워크가 residual 함수의 파라미터를 생성하는 feedforward meta-learner로 구현해 수술 환경의 시간 제약을 맞춥니다.

- **Empirical Impact**: phantom liver 데이터셋 실험에서 MeiBRD는 rigid ICP, 선형 biomechanical LIBR, 비선형 BCF-FEM, V2S 같은 데이터 기반 기준선들을 상대로 전 테스트 설정에서 가장 낮은 평균 TRE를 보이며 OOD 일반화가 특히 강합니다. 무작위 분할에서는 V2S가 잘하지만 geometry/변형 분리(OOD)에서 급격히 악화되어, 희소 관측 조건에서 학습 기반의 일반화 한계를 재확인합니다. 정성/정량적으로도 선형 biomechanical이 특정 영역에서 크게 어긋나는 경우, MeiBRD는 그 구간의 residual을 선택적으로 보정하면서도 이미 잘 맞는 영역에서는 과도한 수정이 줄어드는 모습(오차 증가율 완화, 긴 거리 보정)을 보여 줍니다.



### DriveJudge: Rethinking Autonomous Driving Evaluation with Vision-Language Models (https://arxiv.org/abs/2606.17362)
Comments:
          Under Review

- **Prior Approaches**: 기존 운전 평가 지표는 크게 두 부류로 나뉩니다. (1) ADE/FDE 같은 모방 기반 평가는 시연 궤적을 기준으로 유사도를 재지만, 멀티모달한 운전 특성과 비최적/누락된 시연 문제를 그대로 안고 있습니다. (2) EPDMS 같은 rule-based 평가는 해석 가능하고 물리적으로 정밀하지만, long-tail 상황에서 규칙 적용의 ‘문맥’을 놓쳐 합리적인 동작까지 과도하게 벌점할 수 있습니다.

- **Core Contribution**: 이 논문은 VLM(vision-language model) 추론으로 상황 문맥을 먼저 해석한 뒤, 필요한 경우에만 물리 기반 deterministic rule 함수(예: 충돌, 차선 일탈)를 선택적으로 호출해 평가하는 DriveJudge를 제안합니다. 즉, VLM의 문맥 이해와 rule의 공간 정밀도/물리적 grounding을 결합해 ‘해석 가능 + 문맥 인지’ 평가를 목표로 합니다. 또한 DriveJudge가 제대로 작동하는지 검증할 수 있도록 Driving Quality Classification과 Trajectory Preference Selection 두 가지 human-aligned 벤치마크/평가 프로토콜을 마련합니다.

- **Technical Challenges**: DriveJudge의 핵심 난제는 VLM만으로는 안전·공간 위반을 정밀하게 판정하기 어렵다는 점과, rule을 단순 합산하면 규칙 중요도가 문맥에 따라 달라지는 long-tail을 반영하지 못한다는 점입니다. 논문은 이를 위해 (a) 장면별로 어떤 규칙을 ‘gating’(선택)할지 예측하는 tool-invocation 설계를 넣고, (b) 장면에서 규칙 점수가 낮더라도 행동이 ‘궁극적으로 합리적’일 수 있음을 반영하는 데이터 마이닝/라벨링으로 학습 신호를 구성합니다. 학습은 SFT로 규칙 호출 결정을 먼저 안정화하고, 이후에는 preference 정렬을 위해 RL(GRPO)로 미세 조정합니다.

- **Empirical Impact**: 33,577개의 long-tail 운전 샘플(인간 주석: 해당 장면에서 행동이 합리적인지)로 평가한 결과, DriveJudge는 EPDMS 대비 Driving Quality Classification에서 AUC를 21.23 포인트 개선했습니다. Trajectory Preference Selection에서도 DriveJudge는 DriveCritic 대비 6.5%p 정확도를 더 높여, preference 모델임에도 불구하고 더 일치하는 결과를 보였습니다. 정성 비교에서는 VLM 직접 점수 모델이 공간 정합성 부족으로 사실 오류를 내리거나 rule-based가 문맥상 정당화되는 ‘nudge’를 과벌점하는 문제를 DriveJudge가 tool-grounded 평가로 완화함을 확인했습니다.



### Complex Layout Classification in the Wild: A Low-Resource Approach with Layout-Preserving Augmentations (https://arxiv.org/abs/2606.17355)
- **Prior Approaches**: 기존 문서 레이아웃 분석은 PubLayNet, DocLayNet 같은 대규모 벤치마크에 의존해 supervised 학습 성능을 끌어올렸지만, 역사 문서의 저자원·잡음·비직사각형 구조에서는 일반화가 쉽게 깨집니다. 특히 page-level 분류를 위한 레이블이 조잡하거나(예: NetLay의 coarse 분류), 픽셀 기반 segmentation 중심 접근은 C/L-shaped 같은 복잡한 separator 구조에서 의미 있는 텍스트 블록을 잘 분리하지 못하는 한계가 있습니다.

- **Core Contribution**: 이 논문은 히브리어 역사 문서를 대상으로 separator(구분선) 기반 8개 레이아웃 유형을 수동 라벨링한 CLC 데이터셋(155장)을 구축하고, 소량 라벨 상황에서도 동작하도록 설계했습니다. 또한 separator를 보존하면서 텍스트의 우발적 디테일에 덜 의존하게 만드는 CNN 학습 전략(강한 도메인-aware augmentations)을 제안합니다.

- **Technical Challenges**: 핵심 기술 과제는 “적은 라벨”과 “레이아웃을 결정하는 기하학적 separator는 유지해야 함”이라는 두 조건을 동시에 만족시키는 것입니다. 이를 위해 narrow anisotropic Gaussian masking으로 separator를 남기고 텍스트 질감/세부를 억제해 전역 기하를 학습시키고, 반사(reflection) 기반 증강으로 학습 분포를 늘리되 비대칭 라벨은 label mapping으로 일관성을 유지하며 ConvNeXt-Tiny를 two-stage fine-tuning과 함께 학습합니다.

- **Empirical Impact**: 실험에서는 다양한 backbone을 비교하면서 제안한 “Binary + Reflections + Anisotropic Masking” 파이프라인이 8-way page-level layout classification에서 데이터 부족 상황일수록 성능 향상에 유의미하다는 점을 보여줍니다. 또한 active learning 및 label-efficiency 분석을 통해, 제한된 라벨 예산에서도 학습 곡선 면적(AULC) 관점의 이득을 확인해 ‘적은 라벨로도 separator 기반 라우팅’을 안정화할 수 있음을 시사합니다.



### Bayesian Magnetic Resonance Joint Image Reconstruction and Uncertainty Quantification using Sparsity Prior Models and Markov Chain Monte Carlo Sampling (https://arxiv.org/abs/2606.17343)
- **Prior Approaches**: 기존 compressed sensing MRI 복원은 sparsity 가정을 두고 최적화로 MAP 해를 구하는 방식(예: ADMM 기반)이나, 계층적 Bayesian 모델로 불확실성을 다루더라도 계산 가능성을 위해 sparsity를 이미지 공간에 직접 두거나(단순화), 혹은 MCMC가 스케일 문제로 제한되는 경우가 많았습니다. 또한 deep learning 기반 UQ는 네트워크 가중치의 불확실성까지 포함하지만, 임상 신뢰도와 OOD(분포 외) 상황에서의 한계 우려가 있어 compressed sensing의 UQ가 재조명돼 왔습니다.

- **Core Contribution**: 이 논문은 under-sampled k-space에서 Bayesian 선형 역문제를 세우고, 복원하려는 이미지가 어떤 기저에서 sparse하다는 가정을 prior로 부여하는 불확실성 정량화 프레임워크를 제안합니다. total variation(TV)과 wavelet transform 두 기저에 대해 각각 MCMC-TV, MCMC-Wav를 구성하며, posterior에서 픽셀별 평균과 분산(불확실성 맵)을 함께 산출해 오류 맵과의 상관도까지 제시합니다.

- **Technical Challenges**: 핵심 난점은 (1) 비분화(non-differentiable)한 sparsity prior 때문에 조건부 분포 샘플링이 까다롭고, (2) 고차원 MRI 역문제에서 MCMC가 비현실적으로 느려질 수 있다는 점입니다. 이를 위해 split-and-augmented Gibbs sampler(SPAGS)로 변수 분리를 수행해 likelihood와 prior를 국소적으로 decouple하고, 비분화 항은 proximal MCMC(P-MYULA)로 효율적으로 샘플링하며, TV/wavelet의 구조에 맞춰 필요한 근사·최적화(예: Chambolle 유형)를 적용합니다.

- **Empirical Impact**: 단일 코일과 다중 코일 데이터에서 다양한 k-space sub-sampling pattern과 비율을 실험했으며, 각 제안 기법은 대응되는 최적화 기반 MAP 접근(ADMM-TV/ADMM-Wav)보다 복원 품질이 우수하게 나타납니다. 더 나아가 불확실성 맵이 ground truth와 복원 결과 사이의 error map과 유의미하게 상관되어, 단순한 점추정이 아닌 신뢰도 있는 해석 도구로서의 가능성을 실증했습니다.



### Learning a Maximum Entropy Model for Visual Textures using Diffusion (https://arxiv.org/abs/2606.17342)
- **Prior Approaches**: 기존 시각 텍스처 모델은 Julesz의 가정처럼 텍스처를 국소 통계의 집합으로 요약하고, 그 통계를 맞춰 샘플을 생성해 왔습니다. 다만 사용한 통계가 사람이 설계했거나, object recognition을 위해 pretrained된 네트워크(VGG19 등)의 특징에 의존해 특정 목적에 치우칠 수 있습니다. 결과적으로 고주파 영역 같은 디테일을 충분히 제약하지 못하거나, 표현공간의 조작(예: 보간)이 원치 않는 혼합(패치 단위 결합)을 만들 수 있습니다.

- **Core Contribution**: 이 논문은 maximum entropy 확률모델을 제약하기 위한 “통계(statistics)”를 데이터로부터 unsupervised하게 처음으로 학습하는 원리를 제시합니다. 학습된 512개 통계가 각 텍스처 클래스의 조건부 밀도를 매개하며, generative diffusion에서 쓰는 score/denoising 관점으로 학습·샘플링 절차를 구성합니다. 또한 두 종류 표현(μ: 통계 기대값, λ: 모수)을 두 축으로 보고, 이들이 텍스처 조작에서 어떻게 다른 성질을 갖는지도 함께 분석합니다.

- **Technical Challenges**: 핵심 난제는 “적당한 통계”를 손으로 고르지 않고, maximum entropy 제약에 맞는 통계 f를 학습하는 동시에 계산적으로 학습이 가능해야 한다는 점입니다. 논문은 확산모델처럼 잡음이 더해진 이미지에서 ε를 예측하는 MMSE denoising 학습으로 접근하고, 이때 통계·모수 네트워크가 inner product 형태로 결합되도록 설계를 고정합니다. 또한 텍스처 학습을 위해 steerable pyramid 기반 균질(homogeneous) 패치만 골라 ImageNet21K에서 구성해, 표현이 텍스처 클래스에 더 잘 고정되게 했습니다.

- **Empirical Impact**: 대규모 텍스처 패치에서 학습한 모델은 512 statistics만으로도 Gatys et al.(약 176,640 statistics)과 비슷하거나 더 나은 시각 품질을 보이며, FID에서도 9개 중 8개 텍스처 클래스에서 우위를 보고합니다. 하지만 FID가 텍스처 평가에 항상 적합하지 않다는 한계도 함께 지적하며, 두 모델을 서로 경쟁시키는 MAD 유사 실험으로 결함을 드러내 비교했습니다. 마지막으로 representation space에서의 straight interpolation이 두 끝 텍스처 사이를 “부드러운 단일 텍스처(homogeneous)”로 잇는 경향을 보여, 텍스처 조작 도구로서의 의미 있는 성질을 제안합니다.



### Geometry-Consistent Endoscopic Representations for Image-Guided Navigation via Structured Foundation Model Adaptation (https://arxiv.org/abs/2606.17340)
- **Prior Approaches**: 의료 내시경에서 자주 쓰이는 접근은 DINO/SAM 같은 vision foundation model을 zero-shot 또는 parameter-efficient fine-tuning(예: adapter)로 옮겨오는 방식이다. 하지만 내시경은 저대비 조직 질감, 반복 패턴, 비강체 변형, specular highlight 등으로 인해 geometry(기하) 일관성이 약해지고, 이런 이유로 pose/refinement과 같은 공간 추론 태스크에서 안정성이 떨어지기 쉽다. 한편 endoscopy-specific foundation model이나 domain generalization은 외형·모달리티 변화 대응에는 도움을 주지만, 3D 기하 일관성을 학습 표현에 직접 강제하지 않는 경우가 많아 내비게이션 신뢰도를 제한한다.

- **Core Contribution**: 이 논문은 단일 프레임(monocular endoscopy)에서 geometry-consistent 하면서도 domain-robust한 이미지 표현을 학습하는 통합 프레임워크를 제안한다. 핵심은 합성 데이터로 정확한 기하 감독을 제공하고, Hierarchy-Aware Geometry–Semantic Adaptation(HGSA)가 transformer 계층 구조에 맞춰 low-rank adapters를 선택적으로 삽입하며, 중간 계층은 기하 대응, 깊은 계층은 의미 일관성이 유지되도록 layer-wise objective를 결합하는 것이다. 결과적으로 공간적 안정성(기하)과 cross-domain 강건성(의미/도메인)을 동시에 노리는 표현 학습을 목표로 한다.

- **Technical Challenges**: 가장 큰 어려움은 임상 환경에서 정확한 camera pose나 dense depth 같은 기하 라벨을 대규모로 구하기 어렵고, 동시에 내시경 특유의 비정형 변형과 도메인 갭이 correspondence를 쉽게 붕괴시킨다는 점이다. 이 문제를 위해 CT 기반 3D 해부학 모델로 합성한 다중 도메인 학습 파이프라인을 만들고, 뷰 쌍에 대해 렌더링된 depth·상대 포즈로 flow를 계산해 feature warping 기반 다중 스케일 기하 감독(PatchNCE/코사인 재투영)을 중간 계층에 걸어 준다. 또한 전역 의미 정렬은 late-layer의 global contrastive(InfoNCE)로 수행하되, Gram 기반 regularization으로 공간 해상도 훼손을 억제하면서, transformer hierarchy에 맞춘 adapter 배치·모듈 타깃·rank/스케일을 coarse-to-fine 탐색으로 찾아 학습 충돌을 줄였다.

- **Empirical Impact**: 실험에서는 linear probing으로 의미 분리(장면 분류)와 기하 품질(깊이 추정)을 함께 측정해, 제안한 HGSA 및 기하-의미 결합 학습이 표현 품질을 함께 개선함을 보였다. 더 나아가 pose estimation과 monocular depth estimation 같은 내비게이션 관련 downstream에서 개선이 이어져, 합성 기반으로 학습한 표현이 실제 임상(예: clinical bronchoscopy)로 잘 transfer되는 것을 확인했다. 또한 sinus/colonoscopy로의 제한적 supervision 하 cross-procedure 적응에서도 유의미한 성능을 보였고, 모델 크기·학습 데이터 스케일에 대해서도 좋은 경향을 보여 endoscopy representation learning에서 실용적인 접근이라는 점을 뒷받침한다.



### FATE: Pillar Encoding and Frequency-Aware Training for Event-Based Object Detection (https://arxiv.org/abs/2606.17334)
- **Prior Approaches**: 기존 event object detection은 비동기 event stream을 CNN/ViT에 맞추기 위해 고정된 시간 sub-bin(또는 voxel grid)으로 쪼개 dense 표현을 만든다. 하지만 이 내부 시간 이산화는 미세한 시간 구조를 버리고, 학습에서 주어진 낮은 temporal frequency(예: 20 Hz) 기반의 저주파 추론 한계를 만든다. point 기반·A2S·학습된 grid 표현도 대체로 dense 그리드와 pooling/이산화 의존이 남아 continuous-time temporal dynamics 보존이 약하다는 한계가 지적된다.

- **Core Contribution**: FATE는 이를 줄이기 위해 Pillar Encoding(PE)과 Frequency-Aware Training(FAT)을 하나의 프레임워크로 제안한다. PE는 공간을 pillar로만 이산화하고, 시간은 내부 sub-binning 없이 연속시간 신호로 모델링해 풍부한 temporal dynamics를 유지하면서도 표준 백본이 받는 dense pseudo-image를 만든다. FAT는 low-frequency로만 라벨이 주어지는 문제를 해결하기 위해, tracking-by-detection 기반 temporally dense pseudo-label을 만들고 soft mean-teacher 커리큘럼으로 학생이 고주파 입력에서도 안정적으로 학습하도록 한다.

- **Technical Challenges**: 핵심 난점은 (1) event가 희소/비정규 시간 샘플이므로 시간 정보를 버리지 않으면서도 학습 가능한 연속시간 표현을 구성하는 것, (2) 고주파 supervision이 부족해 train–test frequency mismatch가 생기는 것이다. PE는 각 pillar 내부 잠재 feature 궤적을 Legendre polynomial의 직교 기저에 투영해 L2-최적의 truncated 근사를 만들고, 비균일 타임스탬프 적분은 quadrature(비정규 그리드용 가중 trapezoidal rule)로 보정해 편향을 줄인다. FAT는 canonical(낮은) 라벨로 detector를 먼저 학습한 뒤 고주파로 슬라이딩 윈도우 추론하며 tracking으로 결측을 보간해 dense supervision을 만들고, EMA teacher의 soft pseudo-label을 KL/Localization 일관성 손실로 학생에 전달한다.

- **Empirical Impact**: 논문은 event-based object detection 벤치마크에서 PE+FAT 조합이 강력한 baseline을 일관되게 능가하며, 특히 high-frequency 구간에서 격차가 커짐을 보인다. 또한 200 Hz까지의 높은 temporal resolution에서 견고한 object detection을 달성하면서 파라미터 및 추론 지연 오버헤드는 매우 작다고 보고한다. 이는 기존처럼 시간 sub-binning에 의존하지 않고도 표준 아키텍처 호환성을 유지하며 고주파 성능을 끌어올릴 수 있음을 실증한 결과로 해석된다.



### SierpinskiCam: Camera-Controlled Video Retaking with Sierpinski Triangle Pattern Cues (https://arxiv.org/abs/2606.17310)
Comments:
          20 pages, 13 figures

- **Prior Approaches**: 비디오 retaking은 단일 모노큘러 영상에서 사용자가 지정한 카메라 궤적대로 장면을 다시 렌더링하는 문제다. 기존 방법은 (1) 카메라 pose를 조건으로 넣는 implicit 방식과, (2) 깊이/포인트/point cloud 같은 3D 프록시로 warping한 뒤 VDM으로 보정하는 explicit 방식으로 나뉜다. 특히 explicit 방식은 소스 관측 범위를 벗어난 영역에서 warped 신호가 희소해져 빈 구간이 늘고, 그 결과 생성 모델이 궤적을 따라가지 못하거나 환각을 만들 위험이 커진다.

- **Core Contribution**: SierpinskiCam은 geometry 기반 guidance가 희소해지는 실패 모드를 보완하기 위해 Sierpinski dome(시에르핀스키 패턴 텍스처 도메인)으로 새로 드러나는 영역에도 추적 가능한 multi-scale 특징을 제공한다. 또한 소스 영상은 target 토큰 시퀀스에 그대로 토큰을 덧붙이되, negative RoPE indices(NegRoPE)로 source와 target의 위치 충돌을 막아 의미적으로 구분되게 한다. 그 결과 아키텍처 수정이나 per-video fine-tuning 없이도 source appearance 보존과 카메라 제어를 함께 노린다.

- **Technical Challenges**: 핵심 난제는 ‘소스 warping이 불가능한 영역에서도’ VDM이 target 카메라 운동을 계속 해석할 수 있게 만드는 것과, ‘공간 정렬이 없는’ 소스 영상을 생성기에 안정적으로 주입하는 것이다. 저자들은 도메인에 시에르핀스키 프랙탈 텍스처를 구형으로 렌더링해 카메라 거리 변화에도 코너/엣지 특징이 유지되도록 하고, geometry 프록시의 유효 마스크가 false인 곳엔 도메인 텍스처를 합성해 dense conditioning을 만든다. 소스 주입은 source/target 토큰을 concat하되 spatial RoPE 인덱스를 음수로 배정해 positional collision을 원천적으로 제거하는 NegRoPE로 해결했다.

- **Empirical Impact**: 실험에서 SierpinskiCam은 camera controllability(회전/이동 오차, ATE), geometric consistency, 비디오 품질 전반에서 기존 방법 대비 유의미한 개선을 보였고, 특히 큰 viewpoint 변화처럼 guidance가 끊기는 상황에서 격차가 두드러졌다. DAVIS 기반 평가 및 user study(41명, Likert 1–5)에서도 평균 선호도가 높게 나와 시각적 인상과 궤적 정확도를 동시에 강화함을 뒷받침한다. 또한 텍스처 설계 비교(Checkerboard vs Sierpinski)와 ablation을 통해 multi-scale 코너 구조가 trackability에 중요하다는 점을 근거로 제시한다.



### Reasoning Text-to-Video Retrieval for Operating Room Clips via Action-Driven Digital Twins (https://arxiv.org/abs/2606.17298)
- **Prior Approaches**: 기존 OR 텍스트-비디오 검색은 전역 임베딩의 유사도 매칭에 의존해 “직전 단계”, “원인-결과” 같은 암묵적 질의에 대한 추론을 수행하기 어렵다. 또한 디지털 트윈(DT)을 쓰더라도 주로 객체 중심으로 구성되어, 동일한 대상이 등장하더라도 동작과 상태 전이가 다른 연속 클립을 세밀하게 구분하지 못한다. ReasonT2V류는 언어 추론을 일부 보완하지만, 절차별 action/state transition을 충분히 반영하지 못해 정확도 한계가 나타난다.

- **Core Contribution**: OR3는 각 클립을 객체가 아니라 동작을 중심으로 정리한 action-driven digital twin(ActDT)로 변환해, 동시 subject-action-object 상호작용을 시간 구간으로 비겹치게 묶어 상태 전이 정보를 모델링한다. 이어서 쌍을 이루는 cross-modal paired encoder로 임베딩을 정렬하기보다, LLM이 질의로부터 가상의 ActDT를 생성하는 imagination-based retrieval로 “추상 질의-구체 DT” 간 의미 격차를 줄인다. 마지막으로 top 후보와의 불일치를 근거로 ActDT를 절차 관례에 맞게 수정하는 evidence-grounded refinement를 적용한다.

- **Technical Challenges**: 가장 큰 난제는 암묵적 추론이 필요한 질의를, 실제 영상에 대응되는 구체적 시간-동작 구조로 정합시키는 것이다. OR3는 (1) 가상 ActDT와 실제 ActDT가 동일한 JSON 구조를 갖도록 하여 single encoder 기반 intra-modal 매칭이 가능하게 하고, (2) field-dropout으로 가상에서 비는 시각 속성/필드의 부재에도 강건하도록 학습하며, (3) temporally adjacent hard negative와 temporal-permutation hard negative로 동작-순서 차이를 학습시킨다. 또한 refinement 단계에서 LLM이 절차 특화 패턴을 반영하도록 설계해 원래 상상한 ActDT를 후보 기반으로 점진 보정한다.

- **Empirical Impact**: MM-OR에서 276개의 암묵적 질의(4개 추론 범주)와 386개 클립으로 평가한 결과, OR3는 R@1 57.6%, R@5 77.3%를 기록하며 최강 baseline을 크게 능가한다. 특히 causal 질의에서 큰 격차가 나타나(ReasonT2V 대비 R@1 크게 향상) 원인-결과 추론에 ActDT의 시간적 action reasoning이 효과적임을 시사한다. ablation에서도 ActDT로의 전환과 imagination-based retrieval, evidence-grounded refinement가 각각 성능을 유의미하게 보강하며, 특히 시각적으로 유사한 OR 클립 사이의 세밀한 구별에 기여함을 확인했다.



### Pareto LoRA: Mitigating Modality Imbalance in Unified Multimodal Models via Pareto-Optimal Gradient Integration (https://arxiv.org/abs/2606.17296)
- **Prior Approaches**: UMM은 단일 autoregressive transformer로 텍스트-이미지 이해와 생성을 함께 다루지만, instruction tuning 과정에서 언어 쪽 최적화가 우세해져 이미지 품질이 떨어지는 modality imbalance 문제가 반복돼 왔다. 특히 LoRA 같은 parameter-efficient fine-tuning에서는 텍스트 gradients가 학습을 더 강하게 끌고 가며 vision modality가 상대적으로 억제되는 경향이 나타난다. 기존 loss-level reweighting(예: GradNorm, Step Balance)은 고정된 재가중으로는 작업/레이어별로 달라지는 불균형의 방향과 강도를 충분히 제어하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 LoRA-based UMM instruction tuning에서 modality imbalance를 ‘모달리티별 gradient’ 관점에서 체계적으로 분석한다. 이어서 multimodal instruction tuning을 bi-objective optimization으로 재정의하고, text와 image 목표가 동시에 좋아지는 Pareto-optimal 업데이트를 만들기 위한 Pareto LoRA를 제안한다. Pareto LoRA는 gradient direction과 strength를 조절해 dominant text가 vision 학습을 압도하지 않도록 하면서도 overall 학습 효율과 안정성을 유지한다.

- **Technical Challenges**: 핵심 난제는 두 모달리티의 gradients가 (1) 방향이 충돌할 수 있고 (2) 크기 격차가 orders of magnitude까지 벌어질 수 있는데, 이를 loss 가중치 하나로는 해결하기 어렵다는 점이다. 논문은 cosine similarity로 directional conflict를, gradient magnitude ratio로 imbalance 강도를 진단하고, conflict일 때는 MGDA 기반으로 convex combination(공통 descent 방향)을 구하며, conflict가 아니더라도 크기 차이가 크면 약한 gradient를 rescale해 균형을 맞춘다. 또한 UMM 내부에서도 불균형이 레이어별로 다르므로, 비용 부담을 줄이기 위해 LLM 레이어를 그룹화해 선택적으로 Pareto 조절을 수행한다.

- **Empirical Impact**: CoMM 벤치마크에서 Emu2에 대해 vanilla LoRA 대비 최대 44.9%의 perceptual image quality 개선을 포함해 멀티모달 생성 균형이 일관되게 향상됐다. 텍스트 품질은 유사하거나 약간 개선되는 수준을 유지해, 이미지 성능만 올리는 trade-off 우려를 줄였다는 점이 관찰된다. 다만 Task 4처럼 시각적 grounding이 거의 없는 언어 중심 데이터에서는 text metric(및 helpfulness) 저하가 나타나, Pareto LoRA의 이득이 모달리티 간 경쟁이 큰 작업에서 특히 크다는 실증적 단서를 제공한다.



### Training LLMs with Reinforcement Learning over Digital Twin Representations for Reasoning-Intensive Surgical VideoQA (https://arxiv.org/abs/2606.17279)
- **Prior Approaches**: 기존 수술 VideoQA는 질문을 보이는 요소에 매칭하는 단순 비전-언어 대응에 머물거나, temporal 질문이 있어도 단일 단계 인식 중심으로 설계되는 경우가 많았다. SurgViVQA 같은 방법은 토큰으로 영상 정보를 압축해 시간축의 연속적인 공간-시간 관계가 조각나며, 그 결과 multi-step 공간/시간 추론이 제한된다는 문제를 보였다. 또한 VLM 기반 접근은 수술 특화 패턴을 위해 costly domain-specific adaptation이 필요하거나, single-frame 처리로 temporal dependency를 추적하기 어렵다.

- **Core Contribution**: 이 논문은 reinforcement learning(RL)로 LLM이 직접 영상을 추론에 쓰지 않고, 수술 foundation model들이 만든 digital twin(DT) 표현을 대상으로 reasoning하도록 분리한다. DT는 프레임-윈도우-시술 단계로 계층화해 서로 다른 시간 스케일의 구조를 보존하고, 확률적 불확실성 추정치를 함께 포함해 지각 애매함까지 다룬다. 마지막으로 임상적 타당성(clinical plausibility)을 반영한 reward로 답의 형태 적합성과 정확도를 함께 학습한다.

- **Technical Challenges**: 핵심 난제는 (1) 연속적인 공간-시간 정보를 압축 없이 reasoning에 제공하면서 (2) 수술 영상의 지각 불확실성을 학습 신호에 연결하고 (3) 추론 체인 없이도 학습할 reward를 설계하는 것이다. 저자들은 LLM이 DT 구축 계획을 생성하고, 해당 plan대로 SurgSAM-2/DepthAnything2/RASO/OWLv2를 DAG 의존성으로 실행해 DT를 만들게 하는 structured rollout을 도입했다. 또한 segment/인식/깊이 불확실성을 집계해 정답 reward를 신뢰도에 비례해 스케일링하고, format validation + GPT 기반 임상 타당성/의미 동치 판단을 결합한 reward로 GRPO 학습을 수행한다.

- **Empirical Impact**: REAL-Colon-Reason(2000문항)에서 제안 방법은 overall EM 0.584, SMILE 0.646으로 Qwen3-VL-8B 대비 각각 17%대의 개선을 보이며 추론 복잡도 레벨 1~3 모두에서 향상 폭이 일관적이었다. 기존 image-based 및 token-compression 기반 접근은 temporal reasoning이 필요한 레벨에서 성능이 더 크게 하락해, DT 기반 다단계 추론 효과를 뒷받침한다. REAL-Colon-VQA 및 EndoVis18-VQA로 일반화 평가에서도 keyword accuracy가 기존 SurgViVQA를 유의미하게 앞질러 수술 VideoQA 전반에서 reasoning 구조의 실용성을 확인시켰다.



### Pulling The REINS: Training-Free Safety Alignment of Video Diffusion Models via Representation Steering (https://arxiv.org/abs/2606.17257)
- **Prior Approaches**: 기존 방어는 프롬프트 단계에서 unsafe를 거르거나, 생성된 출력 후에 필터링하는 방식이 중심이었다. 하지만 프롬프트 필터링은 jailbreaking·우회 프레이징으로 쉽게 깨지고, 출력 필터링은 생성 비용을 소모하는 데다 모델 내부 표상이 unsafe 경로를 계속 전파해 취약점이 남는다. 또한 안전 fine-tuning은 데이터·연산 비용이 크고 범용 생성 능력 저하(성능 열화) 위험이 있다.

- **Core Contribution**: REINS는 비디오 diffusion 모델의 가중치 업데이트 없이, 추론 시점에 hidden-state 표상 공간을 안전 방향으로 steering하는 training-free 방법을 제안한다. 저자들은 안전/위험 신호가 비디오 트랜스포머의 중간 hidden-state에 선형적으로 인코딩돼 있고, Supervised PCA로 찾아낸 단일 direction이 safe/unsafe 생성 궤적을 구분한다고 보고한다. 이 방향을 denoising 과정의 중간 레이어에 더해 unsafe 대신 의미적으로 유사한 safe 대안을 생성하도록 유도한다.

- **Technical Challenges**: 핵심 과제는 (1) 안전 정보를 실제로 분리 가능한 레이어와 방향으로 찾아내고, (2) 단순 더하기 perturbation이 아티팩트를 만들지 않게 하며, (3) classifier-free guidance의 두 분기 간 불일치를 막는 것이다. REINS는 SPCA로 후보 레이어별 safety-relevant 성분을 추정하고, safety 정보는 깊이에 따라 누적되지만 steering 성능은 중간 레이어에서 최대가 되는 tradeoff를 분석해 최적 레이어를 선택한다. 또한 per-channel norm preservation로 채널 스케일 변화를 억제하고, CFG에서는 conditional/unconditional 두 브랜치 모두에 동일하게 steering을 적용한다.

- **Empirical Impact**: REINS는 9개 비디오 diffusion 모델(1.3B~5B)에서 T2V와 I2V 모두에 대해, SafeSora와 SafeWatch-Bench 양쪽에서 평균적으로 safety rate를 일관되게 끌어올렸다. 안전 개선 폭은 최대 +0.52 수준까지 관찰되며, 특히 baseline safe 비율이 매우 낮은 상황에서도 효과가 유지돼 adversarial prompt 분포에서도 강건함을 시사한다. 동시에 motion/visual quality는 다수 모델에서 기준선과 동등하거나 경쟁적이어서, 안전성 향상을 ‘성능 열화 없는’ 추론 개입으로 연결했다는 점에서 의미가 크다.



### GeoDisaster: Benchmarking Orchestrated Agents for Operational Disaster Geo-Intelligenc (https://arxiv.org/abs/2606.17246)
Comments:
          28 pages, 11 Figures

- **Prior Approaches**: 기존 RS-VLM과 재난 원격탐사 벤치마크는 주로 단일 이미지 기반 VQA/시맨틱 분할처럼 ‘보이는 것’에 집중했으며, 절차 오류나 툴 호출 유효성·중간 상태 일관성을 충분히 평가하지 못했습니다. 도구를 붙인 에이전트 벤치마크도 종종 단일 에이전트 중심 실행이거나, 역할 의무를 검증 가능한 형태로 강제하지 않아 공간적으로 틀린 산출이 조용히 누락될 여지가 있었습니다.

- **Core Contribution**: 이 논문은 43개 질문 유형과 5개 태스크 패밀리(산림훼손·다중위험·건물피해·홍수 대피 경로·Sentinel-1 SAR 홍수 모니터링)를 포함하는 운영형 재난 지리추론 벤치마크 GeoDisaster(검증 인스턴스 2,921개)를 제안합니다. 각 인스턴스는 광학·SAR 등 이질 EO/GIS 증거와 래스터/벡터·도로 네트워크·노출 레이어를 조합하고, 실행 가능한 지리 워크플로와 결정론적 일관성 체크로 정답을 고정해 언어모델 라벨 의존을 줄였습니다. 또한 18개 재난 지리 툴을 쓰는 orchestrated multi-agent를 구축하고, 역할-계약 기반 정렬(RCEA)로 툴 사용·증거 접지·상태 일관성·의사결정을 함께 끌어올리는 전략을 제시합니다.

- **Technical Challenges**: 운영 재난 분석은 여러 센서/데이터를 교차 확인하며 장기 툴 워크플로를 수행해야 하는데, 기존 multi-agent는 역할 실행이 ‘관찰 불가능’하거나 보상 신호가 단말 결과에 뭉쳐 credit misattribution이 생기기 쉽습니다. 이를 해결하기 위해 중앙 오케스트레이터가 에이전트 간 상호작용을 typed execution contract로 형식화해, 단계마다 산출물이 계약(증거 의존·스키마·완료/실패 조건) 범위를 벗어나는지 검증 가능하게 만들었습니다. 그 위에 failure-aware role-conditioned SFT와 contract-grounded reinforcement learning을 결합해, 단계별 계약 준수 위반을 촘촘한 학습 신호로 사용하고 에이전트별 reward scale 차이도 역할 단위로 정규화합니다.

- **Empirical Impact**: 실험에서 GeoDisaster는 기존 RS-VLM과 agentic 시스템이 툴 시퀀스·제약 만족·중간 추론의 절차적 정확성에서 큰 격차를 보이도록 설계되며, 단순 프롬프트만으로는 성능이 거의 나오지 않는 ‘절차형’ 난이도를 확인시킵니다. 제안한 RCEA 기반 multi-agent는 툴 사용의 정확성, 증거 접지, 상태 일관성, 최종 의사결정 생성에서 개선을 보이며 기존 방법 대비 더 신뢰할 수 있는 운영형 지리추론을 달성합니다. 결과적으로 재난 EO/GIS 문제를 “인지→절차 실행→증거 기반 보고”의 닫힌 워크플로로 평가·정렬하는 방향에 실질적인 기준점을 제공한다는 점에서 의미가 큽니다.



### Landsat-Sentinel-2 Algal Bloom Mapping Using Vision Transformers: Model Description, Implementation, and Examples (https://arxiv.org/abs/2606.17242)
- **Prior Approaches**: 기존에는 MODIS 같은 거친 해양색 센서 기반으로 조류 이상 징후를 보는 방식이 주를 이뤘지만, 조각난(fragmented) 블룸 구조를 충분히 분해해 관측하기 어렵다. 또한 생물-광학(bio-optical) 모델은 물/대기/표면 조건 변화에 민감하고, Landsat·Sentinel 반사율 데이터가 수역에서 전역적으로 조화(harmonized)돼 있지 않아 적용이 까다롭다. 딥러닝 분류가 대안으로 거론되었지만, 중해상도(30m) 위성 멀티스펙트럴을 일관되게 활용해 해안 블룸을 매핑한 사례는 부족했다.

- **Core Contribution**: 이 논문은 Landsat-8/9와 Sentinel-2(각각 30m급으로 정합된 입력)를 활용해 vision transformer 기반 해안 조류 블룸 맵핑을 최초로 성공적으로 구현한다. 블룸 발생 핫스팟 전 세계를 대상으로 글로벌 분포 bloom patch 데이터셋을 구축하고, 컨볼루션 baseline 대비 4가지 transformer 아키텍처를 비교한다. 특히 광학 수질(optical water type)과 대기·표면 조건 변동까지 고려한 평가를 통해, 중해상도·전역 일관 모니터링 가능성을 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 수역별 스펙트럼 한계와 (2) 구름·glint(유광) 등 대기/표면 변동으로 인한 분류 교란을 동시에 처리하는 것이다. 논문은 광학 조건과 수질 유형을 나눠 평가하고, time series에서 구름과 glint stress 상황에도 성능이 유지되는지 확인하는 방식으로 문제를 실전형으로 다뤘다. 그 결과 Swin Transformer가 기존 spectral-index 접근이 만들어내는 광범위한 false positive를 줄이며, 구름·glint 영향을 받는 픽셀을 효과적으로 회피하는 것으로 나타났다.

- **Empirical Impact**: floating bloom 영역 탐지에서 딥러닝 모델들은 전반적으로 강한 성능을 보였고, omission과 commission 오류가 8–65% 범위로 보고됐다. time series 조건에서 spectral-index 기반 방법이 구름·glint 영향을 크게 받는 반면, Swin Transformer는 이를 억제해 더 신뢰도 높은 탐지를 제공했다. 또한 MODIS 유래 생성물과 비교했을 때, 30m급 고해상도가 조각나고 불규칙하게 나타나는 블룸 구조를 더 잘 포착한다는 점이 실증적으로 강조되며, 동적 해안 환경에서의 안정적인 중해상도 모니터링 도구로 딥러닝의 가치를 뒷받침한다.



### Beyond Benchmarks: Continuous Edge Inference for Fine-Grained Roadside Perception (https://arxiv.org/abs/2606.17241)
- **Prior Approaches**: 기존 edge AI/TSR 연구는 정적 이미지나 짧은 시퀀스 중심의 벤치마크에 치중해, 스트리밍 영상에서 생기는 프레임 간 불안정성과 장시간 연산에 따른 thermal throttling(스로틀링) 효과를 충분히 반영하지 못했습니다. 효율적 detector나 temporal reasoning, edge 최적화는 개별적으로 다뤄졌지만, 지속 구동에서의 처리량·지연·발열·품질 변화를 함께 평가한 사례는 드뭅니다. 그 결과 실배포 성능이 체계적으로 과대평가되는 “benchmark-to-deployment gap” 문제가 남아 있습니다.

- **Core Contribution**: 이 논문은 NVIDIA Jetson Orin Nano 같은 제약 환경에서 “지속적인 도로 환경 인식”을 목표로 하는 배포 지향 continuous edge inference 시스템 Edge-TSR을 제안합니다. 핵심은 detection–tracking–fine-grained classification 파이프라인 위에 track-aware temporal stabilization(트랙 기반 시간 안정화)을 얹어 프레임 단위 label flickering을 줄이고, 상태 고정(hysteresis-based label locking)으로 안정적 예측을 유지하는 것입니다. 또한 벤치마크 중심 평가가 실제 스트리밍 배포 성능을 어떻게 과장하는지 정량적으로 보입니다.

- **Technical Challenges**: 연속 스트리밍에서는 시간 상관 잡음(모션 블러·부분 가림·조명 변화·검출 흔들림)이 누적되어 정적 평가 대비 분류 품질이 크게 떨어지고, 장시간 GPU 부하로 온도 한계에 따른 성능 저하가 발생합니다. 또 per-frame 전체 추론은 열적으로 지속 불가능해 sparse inference(예: k=3 프레임마다 full detection)로 줄여야 하지만, 이때 분류 입력이 비는 구간에서 안정성이 깨질 수 있습니다. Edge-TSR은 track로 객체 상태를 전파하고, confidence-weighted temporal voting과 비대칭 hysteresis로 “노이즈에는 덜 민감하고 진짜 변화에는 민감한” 라벨 잠금/탈출 규칙을 설계해 이러한 충돌을 완화합니다.

- **Empirical Impact**: 세 가지 SOTA 베이스라인을 대상으로 정적 이미지 평가에서 스트리밍 배포로 전환할 때 상대 성능이 20–30% 저하되는 일관된 격차를 관찰했습니다. Edge-TSR의 안정화 모듈은 per-frame 추론 대비 분류 정확도를 최대 10.16%p까지 회복하면서도, 추가 오버헤드는 미미하다고 보고합니다. 26km 구간을 55분간 실제 차량 배치한 결과 단일 Jetson Orin Nano에서 16.18 FPS를 유지했고, thermal 한계 내에서 지속 운영 가능함을 실증했으며, 재현을 위해 샘플 스트리밍 평가용 데이터셋과 구현을 공개합니다.



### Quantum Enchanced Multi-Scale CNN with Bi-directional Mamba for Crop Field Analysis (https://arxiv.org/abs/2606.17222)
- **Prior Approaches**: 기존 HSI(하이퍼스펙트럴) 작물 분석은 CNN, transformer, Mamba류 state-space 모델을 중심으로 스펙트럼-공간 의존성을 학습해 왔습니다. 하지만 CNN은 고차원 스펙트럼에서 전역 구조를 충분히 포착하기 어렵고, transformer는 self-attention의 높은 연산/메모리 비용과 파라미터 부담으로 제한된 라벨 데이터에 취약하다는 한계가 있었습니다.
또한 state-space 기반 모델은 3D 데이터 큐를 1D 시퀀스로 펼치며 공간 기하(인접 경계)를 훼손할 수 있고, quantum 접근은 주로 위성/저해상도 등 제한된 설정에 머물러 실제 UAV 작물 분류에 대한 통합 검증이 부족했습니다.

- **Core Contribution**: 이 논문은 CNN의 다중 스케일 공간 특징 추출 위에 BiSpectral Mamba(양방향 state-space)로 장거리 전후 의존성을 모델링하고, 스펙트럼 attention과 quantum-inspired 학습을 결합한 BiSpectral Mamba 기반 프레임워크를 제안합니다. 핵심은 “공간 기하 보존(다중 스케일 CNN) + 스펙트럼 시퀀스 의존성(양방향 Mamba) + 유효한 전역 비선형 강화(quantum-inspired)”를 한 파이프라인으로 묶어 효율성과 정확도를 동시에 노리는 것입니다.
또한 class imbalance를 직접 겨냥해 class-weighted 최적화 및 feature fusion 전략을 포함하고, 훈련 안정성을 높이기 위한 손실 설계(하이브리드 Cross-Entropy와 Log-Cosh Dice)를 함께 사용합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 스펙트럼 차원이 매우 크고(수백 밴드), (2) 라벨이 제한적이며(현장 불균형), (3) Mamba처럼 시퀀스 모델을 쓰면 공간 토폴로지가 깨질 수 있다는 점입니다. 논문은 이를 multi-scale CNN으로 공간 구조를 먼저 “앵커링”한 뒤, 정제된 특징을 토큰화하여 양방향 state-space로 장거리 스펙트럼-공간 의존성을 선형 복잡도로 학습하게 설계했습니다.
여기에 spectral attention으로 유익한 밴드를 강조하고 잡음/중복 밴드를 억제하며, class-weighted 최적화 및 Log-Cosh Dice 계열 손실로 minority class의 학습을 안정화하는 방식으로 실전 제약을 완화합니다.

- **Empirical Impact**: UAVHSI-Crop(UAV 기반 30클래스)에서 제안 프레임워크는 전체 정확도 84.83%를 달성하며, convolution-중심 특징 학습, attention 기반 밴드 정제, state-space의 장거리 모델링이 결합될 때 강건한 spatial-spectral 표현 학습이 가능함을 보여줍니다. 특히 기존 transformer 대비 계산/파라미터 부담을 줄이면서도 복잡한 작물 분류에서 유의미한 성능을 확보했다는 점에서 하이브리드 경량화 방향성을 제시합니다.
저자들은 추가로 이 구조가 작물 병해 탐지, 수확량 예측, 토양 수분 추정 등 broader agricultural 및 remote sensing 작업에도 확장될 잠재력을 강조합니다.



### Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation (https://arxiv.org/abs/2606.17188)
- **Prior Approaches**: 기존 비전-언어 모델(VLM) 멀티링구얼 평가는 언어와 문자(orthography)를 사실상 1:1로 가정하는 경우가 많다. 그래서 Gurmukhi, Shahmukhi처럼 같은 언어의 서로 다른 문자 스크립트를 동일 난이도 조건에서 비교하지 못해, 성능 차이를 ‘언어 역량’이 아니라 ‘문자 패턴’ 문제로 분해하기 어렵다.
또한 텍스트 기반 연구에서 스크립트에 따른 성능 저하가 보고돼 왔지만, 시각적 입력이 이런 스크립트 편차를 얼마나 메우는지에 대한 체계적 검증은 부족했다.

- **Core Contribution**: 이 논문은 PuMVR(Punjabi Multimodal Visual Reasoning)이라는 1,000개 평행 이미지-텍스트 벤치마크를 제안하며, Punjabi의 세 스크립트(Gurmukhi, Shahmukhi, Roman)를 동일 의미로 정렬해 ‘스크립트 변인’을 독립적으로 평가한다. 여기에 Script Consistency Rate(SCR)을 도입해, 한 스크립트에서 잘하는 모델이 다른 스크립트에서도 동일하게 맞히는지를 강하게 요구하는 ‘스크립트 비의존성’ 지표로 재정의한다.
즉, 멀티링구얼을 언어 수 확장(coverage)이 아니라 스크립트 강건성(orthographic robustness)으로 평가해야 한다는 프레임을 제시한다.

- **Technical Challenges**: 핵심 난제는 ‘문자만 바뀌고 의미/이미지 난이도는 동일’하게 맞춘 평행 데이터를 만드는 일과, 모델 출력이 스크립트 표기 방식 차이로 생기는지(포맷 문제) 의미 이해 실패인지 분리하는 것이다. 논문은 이미지-질문-보기 4지선다를 스크립트별로 제공하되 정답은 의미적으로 동일하게 맞춰, 스크립트 간 성능 차이를 orthographic comprehension 실패로 해석할 수 있게 설계했다.
또한 텍스트-only 대비 멀티모달이 스크립트 갭을 ‘보완(compensate)’하는지 ‘누적(additive)’ 효과인지 분해하고, cross-script in-context learning의 전이 안정성까지 TE(Transfer Efficiency)로 계측했으며, 스크립트 쌍별 McNemar 검정으로 편차의 통계적 견고함을 확인했다.

- **Empirical Impact**: 10개 SOTA VLM을 PuMVR에 적용한 결과, 스크립트 변화만으로 정확도가 최대 16%까지 흔들리는 Script Gap이 관찰됐다. 시각 입력은 절대 성능을 전반적으로 올리지만 SCR 갭을 닫지 못해, 멀티모달이 스크립트 편차를 자동으로 해결하지는 못한다는 점이 드러났다.
더 나아가 in-context exemplars 전이가 스크립트에 매우 취약하며, 일부 모델은 특정 ‘앵커 스크립트’에 강하게 고정된 듯한 비대칭 전이(TE < 67%)를 보였고, SCR이 24.8%까지 내려가는 경우도 확인돼 현재 ‘multilingual’ 주장의 실질적 범위가 제한적임을 시사한다.



### Looped World Models (https://arxiv.org/abs/2606.18208)
Comments:
          Technical Report

- **Prior Approaches**: 기존 world model은 관측을 잠재공간에서 예측하고 그 위에서 계획/학습을 수행하는 RSSM 계열(Dreamer, PlaNet 등)과, 토큰/시공간 잠재를 변환기로 바꾼 IRIS·DIAMOND·EMERALD 같은 방식으로 발전해 왔습니다. 그러나 고정 깊이(또는 모델 크기 증가)로는 롤아웃이 길어질수록 예측 오차가 누적(compounding)되며, 이를 버티기 위해 더 깊은 네트워크를 쓰면 파라미터와 추론비용이 함께 폭증하는 긴장관계가 남아 있습니다.

- **Core Contribution**: 이 논문은 Looped World Models(LoopWM)라는 새로운 루프드(looped) world modelling 아키텍처를 제안합니다. 한 번의 상태 전이(step)를 “공유된 transformer 블록을 반복 적용”해 잠재 상태를 점진적으로 정제(latent refinement)하고, 이를 통해 모델 크기·학습 데이터 규모와 별개로 ‘반복 깊이(iterative latent depth)’를 스케일 축으로 삼습니다. 또한 각 전이의 복잡도에 맞춰 inner-loop 반복 횟수를 자동으로 늘리거나 줄이는 adaptive computation을 도입합니다.

- **Technical Challenges**: 핵심 난제는 루프를 많이 돌려도 잠재 상태가 폭주하지 않게 만드는 수치 안정성(stability)입니다. 이를 위해 spectrally-constrained state-retention 파라미터화를 사용해 상태 유지 행렬의 고유값이 (0,1) 구간에 들어가도록 구성하고, 루프 반복이 길어져도 residual dynamics가 bounded 되게 보장합니다. 더불어 Poisson 기반 stochastic loop depth 학습과 entropy-regularised early-exit 게이트를 결합해 학습 중 손실 스파이크를 줄이면서 추론 시 적응적 종료가 가능하게 했습니다.

- **Empirical Impact**: 실험에서는 LoopWM이 기존 world model과 비교해 예측 정확도는 경쟁적이거나 더 높으면서도 파라미터 효율은 최대 100배까지 개선될 수 있음을 보여줍니다. 또한 더 긴 롤아웃에서도 안정적으로 예측이 유지되어, 단순히 모델을 키우는 방식보다 긴장관계를 더 직접적으로 완화합니다. 무엇보다 test-time에서 전이 난이도에 따라 반복 깊이를 조절해 평균 추론비용을 크게 절감할 수 있어, 실시간 제약이 있는 embodied/자율 시스템에 의미 있는 방향을 제시합니다.



### Seeing Is Not Screening: Multimodal Hidden Instruction Attacks on Agent Skill Scanners (https://arxiv.org/abs/2606.18198)
- **Prior Approaches**: 기존 스킬 스캐너들은 주로 SKILL.md 같은 텍스트 설명, 매니페스트/메타데이터, 소스 코드와 권한·의존성 같은 신호에 기반해 위험을 탐지합니다. 또한 LLM-as-a-judge, 시그니처, 정적 패턴 분석, 일부는 실행 데이터 플로우를 결합하기도 하지만, “이미지에 들어간 의도”는 충분히 검증되지 않는 경우가 많습니다. 멀티모달 에이전트가 스킬 내 이미지를 실제 실행 맥락에서 해석한다는 점을 고려하면, 시각 채널에 숨긴 지시문이 텍스트/코드 기반 점검을 회피할 수 있는 공백이 존재합니다.

- **Core Contribution**: 이 논문은 멀티모달 에이전트 배포 환경에서 악성 운영 지시가 “이미지에 숨겨졌다가 실행 시 복원”될 수 있다는 블라인드 스팟을 정식 위협으로 제시합니다. 이를 바탕으로 SkillCamo는 악성 명령을 스킬 패키지의 이미지로 위장(은닉)하고, 주변 문서를 자연스럽게 이미지 참조가 정상 워크플로우인 것처럼 재작성해 탐지 신호를 분산시킵니다. 방어 측에서는 ExecScan을 제안해 문서·코드·참조 리소스·이미지를 함께 복원한 뒤, 의도 추출→행동 재구성→남용 평가→숙고적 실행 시뮬레이션까지 수행해 실행 수준 위험을 점수화합니다.

- **Technical Challenges**: 핵심 기술 난관은 이미지 속에 숨은 지시가 텍스트/코드만으로는 드러나지 않고, 실행 시 에이전트의 멀티모달 해석 결과로만 의미가 완성된다는 점입니다. SkillCamo는 이런 특성을 악용해 “이미지 단독”이 아니라 “문서 지침과 이미지 페이로드의 공동 해석”으로 악성 의도를 활성화하도록 설계하며, 스캐너 피드백을 반복 반영해 텍스트의 노골적 의심 신호를 점진적으로 약화시킵니다. ExecScan은 이를 막기 위해 아티팩트 수준 검사에서 끝나지 않고, 여러 사용 맥락을 만들어 이미지-캐리 지시를 복원한 뒤 가능한 행동 체인을 투영하고 실행 시나리오에서 위험 격차를 찾는 방식으로 대응합니다.

- **Empirical Impact**: 실험 결과 SkillCamo로 생성된 이미지-숨김 악성 스킬은 기존 6종 스캐너에서 최대 90%에 가까운 성공률로 탐지를 회피했습니다(변형에 따라 여전히 크게 회피). 반면 ExecScan은 공격 성공률을 8%까지 낮추며, 이미지 복원·멀티컨텍스트 추론·실행 시뮬레이션이 조합될 때 성능이 가장 크게 향상되는 것으로 나타났습니다. 또한 200개 양성 벤치마크에서 FPR이 낮은 편이면서 F1 83.8%를 달성해, 단순히 과도한 거부가 아니라 “실행 수준 위험을 더 정확히 잡는” 방어 효과를 보여줍니다.



### Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System (https://arxiv.org/abs/2606.18112)
- **Prior Approaches**: 기존 agentic navigation 연구는 지시 따르기, 물체 탐색, 타깃 추적, 자율주행처럼 서로 다른 시각 소비 전략이 필요한 과업을 같은 백본으로 묶어도, 추론 시 observation 전략을 외부에서 재구성하기가 어려웠습니다. 특히 trajectory(경로) 중심 학습은 reactive action-sequence mappers처럼 행동만 즉시 매핑하는 형태로 붕괴하는 문제가 보고되어 왔습니다. 결과적으로 과업 전환이나 장기 임무에서의 동적 제어가 제한적이었습니다.

- **Core Contribution**: 이 논문은 Qwen-RobotNav를 제안하며, 추론 시 архитект처 변경 없이 observation strategy를 조절할 수 있는 파라미터화된 인터페이스를 핵심 기여로 내세웁니다. 인터페이스는 (1) task mode로 네비게이션 행동을 선택하고, (2) token budget, per-camera weights 같은 observation 파라미터로 시각 히스토리 인코딩 방식을 제어합니다. 또한 상위 플래너가 에피소드 중간에 task mode와 컨텍스트 전략을 전환해 같은 모델을 반복 호출하며 복잡한 행동을 조합할 수 있게 만듭니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 파라미터화된 설정이 바뀌어도 백본이 과업 지시를 안정적으로 따르며, 경로 데이터만으로는 생기기 쉬운 반응형 매퍼 붕괴를 피하는 것이었습니다. 논문은 학습 시 모든 파라미터에 대한 randomization을 적용해 zero architectural modification 수준의 추론 설정 강건성을 확보하고, vision-language 데이터를 함께 학습(co-training)해 reactive 붕괴를 막았습니다. 더불어 장기 시나리오에서는 상위 플래너가 목표를 분해한 뒤 모델의 task mode/컨텍스트를 동적으로 스위칭하는 방식으로 문제를 해결합니다.

- **Empirical Impact**: Qwen-RobotNav는 15.6M 샘플로 학습되었고, 여러 주요 네비게이션 벤치마크에서 새로운 state-of-the-art를 달성하며 제안한 설계의 실효성을 입증했습니다. 또한 2B에서 8B로의 스케일링에서 성능이 유리하게 개선되었고, multi-task 공동 학습이 공유된 spatial-planning 기반을 형성해 과업군 간 전이를 촉진했습니다. 나아가 다양한 환경에서 실제 로봇에 대한 strong zero-shot generalisation 성과를 보여 agentic navigation에 대한 확장성과 실사용 가능성을 높였다는 점에서 의미가 큽니다.



### Blended Chart Surfaces: A Seamless Explicit Representation for Smooth Surface Fitting (https://arxiv.org/abs/2606.18069)
Comments:
          17 pages, 16 figures

- **Prior Approaches**: 기존 방법은 크게 두 갈래로 나뉩니다. implicit 표현은 타깃에 맞춰 최적화가 쉽지만 최종적으로 iso-surfacing(예: Marching Cubes)로 명시적 표면을 뽑아야 하고, 이 과정이 미분 추정의 매끄러움을 깨뜨릴 수 있습니다. explicit(표면) 표현은 직접 평가·미분 접근이 쉬우나, canonical-domain 제약(예: disc/sphere 계열)이나 patch seam에서의 불연속·seam artifact가 문제로 남았습니다.

- **Core Contribution**: 이 논문은 Blended Chart Surfaces라는 네트워크 없이도 동작하는 명시적(surface-based) 표면 표현을 제안합니다. 사용자 제공 proxy mesh가 목표의 위상(topology)과 거친 형상(coarse geometry)을 담당하고, 각 proxy 정점에 할당된 로컬 저차(polynomial) 패치를 one-ring 이웃에서 partition-of-unity 방식으로 블렌딩해 전역적으로 매끈한 형태를 만듭니다. 그 결과 표면이 C∞ 연속으로 매끄럽고, normals와 surface energies 같은 미분량을 안정적으로 직접 계산할 수 있게 됩니다.

- **Technical Challenges**: 핵심 난관은 proxy mesh가 본질적으로 코너/모서리 등 비매끈한 이산 구조를 갖는데도, 블렌딩 이후에는 전역적으로 C∞ 수준의 연속성과 미분 일관성을 보장해야 한다는 점입니다. 이를 위해 저자들은 ‘one-ring coordinate’ 기반의 겹침 좌표계와, 경계(±1 등)에서 모든 도함수가 0이 되도록 설계된 C∞ 블렌딩 함수를 사용해 patch 경계에서의 고차 도함수까지 맞춥니다. 또한 블렌딩이 최적화 과정에 순방향으로 개입되도록(불필요한 over-constraint 없이) 로컬 패치 계수를 end-to-end로 Adam 등 일반 최적화기로 implicit 타깃(SDF/implicit field 등)에 직접 피팅합니다.

- **Empirical Impact**: 실험에서는 다양한 topology와 기하 복잡도에서 표현력·간결함·미분 접근성의 균형이 좋다는 점을 보였습니다. interpolating baselines(같은 proxy 위의 보간 계열)이나 mesh displacement MLP 같은 explicit 대안들과 비교했을 때도, patch 경계에서 매끈함이 구성상 유지되며 학습/최적화가 잘 수렴하는 양상이 관찰됩니다. 특히 differential quantities를 “바로” 쓸 수 있어 geometry processing과 학습 기반 fitting 파이프라인을 함께 잇는 데 의미가 있습니다.



### Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models (https://arxiv.org/abs/2606.17846)
Comments:
          44 pages

- **Prior Approaches**: 기존의 로봇 조작 연구는 비전-언어-행동을 어느 정도 맞추더라도, 조작 데이터의 이질성·수집 비용·다양성 부족 때문에 정렬(alignment)과 스케일링을 동시에 달성하기 어려웠습니다. 그 결과 다양한 플랫폼과 상황에서의 진짜 일반화보다는 특정 환경/자세에 강하게 편향되는 문제가 반복됐습니다. 또한 많은 벤치마크가 학습 품질을 충분히 반영하지 못해 OOD(out-of-distribution) 상황의 난도가 낮거나 불일치했습니다.

- **Core Contribution**: 이 논문은 로봇 조작에도 언어·멀티모달 파운데이션 모델의 ‘스케일링 레시피’를 적용하려는 시도를 Qwen-RobotManip으로 구체화합니다. Qwen-VL(Qwen-VL 기반) 위에 표현·모션·행동의 3개 차원에서 통일된 정렬 프레임워크를 도입해, 멀티소스 대규모 학습이 충돌하지 않고 함께 수렴하도록 설계했습니다. 더불어 사람(1인칭 핸드 시연) 정보를 로봇 궤적으로 합성하는 파이프라인과 데이터 조정 커리케이션으로 대규모 학습 데이터를 확보합니다.

- **Technical Challenges**: 핵심 기술 난제는 로봇 조작 데이터가 ‘기본적으로 이질적’이라서 표현·움직임·행동을 같은 학습 문맥에 정렬하기가 어렵다는 점입니다. 논문은 이를 해결하기 위해 representation, motion, behavioral dimensions에 걸친 unified alignment framework를 구성해, 서로 다른 데이터 소스가 학습 신호를 일관되게 제공하도록 맞춥니다. 또한 15개 플랫폼에 대한 합성(인간 시연→로봇 trajectory)과 함께 이질적 데이터셋을 조화시키는 rigorous curation pipeline을 통해 스케일을 유지하면서도 품질을 끌어올립니다.

- **Empirical Impact**: 실험에서는 오픈소스 데이터와 인간 비디오만으로 약 38,100시간 규모의 pretraining 코퍼스를 구성하고, zero-shot instruction following, perturbations에 대한 강인성, reactive error recovery, cross-embodiment transfer 같은 emergent generalization을 보여줍니다. 더 나아가 표준 벤치마크는 사전학습 품질을 잘 반영하지 못하며, RoboCasa365·LIBERO-Plus·EBench·RoboTwin-* 등 OOD 설정에서 차이가 드러난다고 지적합니다. 그 결과 Qwen-RobotManip은 모든 OOD 설정에서 기존 SOTA를 크게 앞섰고, pi0.5를 포함한 모델 대비 성능 우위를 보이며 RoboChallenge 1위(상대 20% 개선) 및 AgileX ALOHA, Franka, UR, ARX 같은 실제 로봇 플랫폼에서도 검증됩니다.



### The Slop Paradox: How Synthetic Standardization Erodes Clinical Uncertainty and Cross-Modal Alignment in AI-Rewritten Radiology Reports (https://arxiv.org/abs/2606.17791)
- **Prior Approaches**: 기존 연구들은 방사선 리포트 생성 품질을 BLEU나 진단 정확도 같은 지표로 주로 평가했지만, 생성 과정에서 “무엇이” 사라지는지 정보 손실을 체계적으로 측정하진 못했다. 불확실성(hedging) 언어의 중요성은 알려져 있으나, LLM 리라이팅이 이 언어를 보존하는지 파괴하는지 정량화된 사례는 드물었다. 또한 BiomedCLIP처럼 이미지-텍스트 정렬을 학습하는 멀티모달 모델은 텍스트 품질이 좋다는 가정 하에 정렬을 최적화했지만, 합성 리포트가 정렬을 얼마나 망가뜨리는지는 계량되지 않았다.

- **Core Contribution**: 이 논문은 IU Chest X-Ray 450장을 대상으로, LLM으로 방사선 리포트를 세 가지 현실적 리라이팅(EHR summarization, standardized rewriting, teaching case preparation)으로 변환한 뒤 정보 열화(information degradation)를 “통제 실험” 형태로 측정한다. 엔티티 소실(entity erosion), 임상적 불확실성 언어 붕괴(hedging collapse), 이미지-텍스트 정렬 저하(cross-modal alignment degradation)를 동시에 계량해, 텍스트 품질 저하가 멀티모달 정렬과 어떻게 엇갈리는지 보인다. 핵심 결론은 정보 손실의 크기와 크로스모달 충실도 저하가 분리(dissociation)된다는 점이다.

- **Technical Challenges**: 어떤 리라이팅이 임상 내용과 정렬을 각각 얼마나 망가뜨리는지 측정하기 위해, scispaCy 기반 medical NER로 엔티티를 추출하고 정규식 기반 불확실성 마커(hedging)를 카운트하는 두 축을 설계했다. 정렬 평가는 BiomedCLIP 임베딩의 코사인 유사도로 계산하되, 텍스트는 토큰 한도(256 tokens)로 트렁케이션해 실제 모델 처리 제약도 반영했다. 희귀 질환이 더 큰 열화를 겪을지에 대해서는 rare/common 병목 가설을 사전 명시하고, 다중 비교 보정까지 적용해 견고성을 점검했다.

- **Empirical Impact**: 결과적으로 EHR summarization은 엔티티 51.4%, hedging 43.7%를 크게 깎지만 이미지-텍스트 정렬은 거의 유지(정렬 드롭 2.5%)된다. 반대로 standardized rewriting과 teaching case preparation은 엔티티 소실은 각각 26.8%, 29.3%로 상대적으로 덜하지만, 정렬 저하는 14.9~16.5%로 EHR summarization 대비 6~7배 수준이었다. 논문은 이를 “slop paradox”로 명명하며, 훈련용으로 더 “깔끔해 보이게” 만든 텍스트가 오히려 비전-언어 대응을 더 멀어지게 만든다고 경고한다. 또한 희귀 병리가 더 심하게 훼손된다는 가설은 다중 비교 보정 후에도 유의한 차이가 없어, 조건별 성능 모니터링으로 오염을 감지하기 어렵다는 함의가 도출됐다.



### ED3R: Energy-Aware Distributed Disaster Detection Enabled by Cooperative Robotic Agents (https://arxiv.org/abs/2606.17739)
Comments:
          14 pages, 9 figures

- **Prior Approaches**: 기존 에너지 효율 연구는 네비게이션·센싱·인프라 자원·연산 오프로딩 같은 개별 요소를 줄이는 데 집중하는 경우가 많았고, 시간 제약을 만족하는 쪽이 주된 목표였습니다. SAR/UAV 관련 연구들도 위험 환경에서의 탐색·객체 탐지·경로 최적화는 다루지만, 산불 탐지를 위해 이동·통신·연산 오프로딩을 함께 최적화하는 통합 접근은 상대적으로 부족했습니다.
또한 산불 탐지 분야의 다수 방법은 분산 제어나 규칙 기반에 치우쳐 있었고, 미래 결과를 미리 평가하는 forward-looking reasoning은 거의 없거나 결여되어 있었습니다.

- **Core Contribution**: ED3R은 불확실성 하에서 산불을 “요구 confidence로 탐지”하면서, 로봇의 이동·센싱·컴퓨팅·통신으로 소모되는 에너지를 최소화하도록 설계된 energy-aware 분산 프레임워크입니다. 로봇과 원격 컨트롤러(RC) 사이에 계층적 협력 의사결정을 두는데, RC는 motion command를 정하고 로봇은 탐지 실행 위치(onboard vs remote)와 사용할 모델(how)을 자원 기반으로 선택합니다.
또한 장애물 회피·중복 탐색 방지·적응적 early mission completion·페널티 함수로 제약 가능성을 강화해, 단순 성능 최적화가 아닌 임무 성공을 겨냥합니다.

- **Technical Challenges**: 가장 큰 어려움은 RC와 로봇이 분산된 상태에서 행동 결과에 대한 공통 보상이 “동시에” 주어지지 않는다는 점입니다. ED3R은 이를 해결하기 위해 distributed neural regression 모델로 후보 전략들의 미래 효과를 미리 평가한 뒤, 탐지 confidence와 에너지 효율 사이의 최적 trade-off를 그리디하게 선택합니다.
여기에 통신 대역폭/전송 파워/채널 조건, 센서 샘플 크기와 처리 FLOPs 같은 현실적인 에너지·지연 요소를 모델링해 제약 위반을 커스텀 페널티로 반영합니다.

- **Empirical Impact**: 현실적인 로보틱스 시뮬레이션과 ablation, 베이스라인 비교를 통해 ED3R은 최대 97.18% 임무 성공률을 달성했습니다. 특히 가장 까다로운 임무에서 에너지는 최대 36.4% 줄이고, 산불 탐지는 최대 41% 더 빠르게 수행해 시간-에너지-신뢰도 동시 최적화의 실효성을 보여줍니다.
forward-looking과 분산 계층 의사결정이 결합될 때, 네트워크가 바뀌거나 새로운 산불 상황에서도 강건한 성능을 보이며 관련 분야의 산불 감시/긴급 대응 설계 방향에 의미 있는 근거를 제공합니다.



### ERQA-Plus: A Diagnostic Benchmark for Reasoning in Embodied AI (https://arxiv.org/abs/2606.17639)
Comments:
          under review at NeurIPS

- **Prior Approaches**: 기존 VQA/EQA 벤치마크는 높은 정답률을 만들 수 있어도, 모델이 실제로 ‘근거 기반(reasoning with grounding)’인지 ‘언어/시각 지름길(shortcut)’인지 분리해 진단하기 어렵습니다. 특히 EQA는 탐색이나 과업 수행에 초점이 치우쳐 있어, 평가하고 싶은 추론 의존성(공간·절차·의도 등)을 정밀하게 통제하기가 어렵습니다. 그 결과 카테고리별 실패 모드를 세밀히 분해해 개선 방향을 찾기 힘들다는 한계가 있습니다.

- **Core Contribution**: ERQA-Plus는 로봇 중심 이미지에 근거해 embodied AI의 추론을 진단(diagnostic)하도록 설계된 벤치마크입니다. 퍼셉션, 액션 중심, 사회/상호작용, 내비게이션·환경, 맥락·상식 등 5개 축을 포함하는 계층적 taxonomy로 질문을 구성해, 어떤 형태의 추론이 필요한지 세분화해 평가합니다. 총 1,766개 QA를 통해 ‘정답 여부’뿐 아니라 ‘어떤 추론을 안정적으로 수행하는지’를 보여주는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트 단서 없이 장면 증거에 의존하게 만들고, (2) 정답이 명확하며 (3) 공간·절차·사건 예측 같은 고난도 의존성을 정확히 검증하는 QA를 대규모로 생성하는 것입니다. 이를 위해 taxonomy-guided 질문 생성과 함께 자동 quality judging, 반복 수정(revision), 그리고 인간 검증을 포함하는 multi-stage 파이프라인을 사용해 시각적 grounding과 추론 품질을 끌어올립니다. 또한 Judge 에이전트가 JSON 형태로 점수·이슈 목록을 내며 필터링/개선 루프를 가능하게 하되, 자동 판정의 인간 정렬에는 여전히 트레이드오프가 있음을 함께 보여줍니다.

- **Empirical Impact**: 실험에서는 LLaVA-NeXT-8B, Prismatic-7B, MiniCPM-V-4.5-8B, Qwen3-VL, RoboRefer, RoboBrain2.5 등 다양한 VLM/embodied 모델을 평가했으며, Qwen3-VL-32B가 전체 정확도 83.4%를 기록했지만 공간·절차·이벤트 예측·의도 추론에서 여전히 취약점이 남았습니다. 특히 temporal reasoning과 path planning 같은 하위 범주가 모델 간 격차를 크게 만들고, RoboBrain2.5-8B는 temporal reasoning에서 도약 성능을 보여 ‘도메인 정렬 학습’이 효율적으로 기여할 수 있음을 시사합니다. 동시에 자동 Judge의 인간 판정 일치가 완전하진 않아, ERQA-Plus는 단일 리더보드가 아니라 범주 단위 실패 원인 분석에 더 적합한 진단 도구로 자리매김합니다.



### MuseVLA: An Adaptive Multimodal Sensing Vision-Language-Action Model for Robotic Manipulation (https://arxiv.org/abs/2606.17598)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 로봇 모델은 주로 RGB만 입력으로 받아 온도·소리·레이다 반응처럼 RGB로 추정하기 어려운 물리 정보를 놓치는 경우가 많았습니다. 멀티모달 VLA가 일부 등장했지만, 센서마다 전용 인코더/아키텍처를 쓰거나 고정된 센서 구성을 가정해 새로운 센서를 쉽게 확장하기 어렵고, 학습용 다중센서 데이터 수집 비용도 큰 한계로 남아 있었습니다.

- **Core Contribution**: MuseVLA는 센서를 ‘온디맨드 도구’처럼 취급해, 작업 지시와 시각 컨텍스트만으로 어떤 센서를 호출할지와 무엇에 집중할지를 선택하는 adaptive multimodal sensing VLA를 제안합니다. 또한 센서 측정을 카메라 평면에 공간적으로 grounding한 grounded sensor images라는 단일 중간표현으로 변환해, 이 표현을 통해 이질적 센서를 하나의 백본에서 공통 처리하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 매 작업마다 관련 센서를 선택해야 하는 문제와 (2) 서로 다른 센서 신호를 백본이 소화 가능한 공통 표현으로 바꾸는 문제, (3) 희소한 다중센서 데이터로도 일반화를 확보하는 문제입니다. 논문은 learnable sensor tokens로 센서 선택/타깃 설명을 ‘툴 호출’ 인자처럼 생성하고, grounded sensor images로 modality-specific encoder 없이 unified한 입력을 만들며, RGB 비디오 데이터에 VLM+SAM3 기반 합성 파이프라인을 적용해 unseen sensor-guided task에도 대응하도록 학습합니다.

- **Empirical Impact**: 실세계 덱스터러스 핸드 로봇에서 온도 유도 pick-and-place, 오디오 기반 물체 탐색, mmWave 레이다 기반 숨은 물체 회수 같은 과제에 대해 평균 80.6% 성공률을 보고했으며, RGB-only 및 raw 센서 heatmap 기반 VLA 대비 유의미하게 향상됐습니다. 또한 합성 데이터로 사전학습한 모델은 unseen 작업에서 평균 66.7% 성공률로 zero-shot 일반화 성능이 강함을 보여, 다중센서 로봇 조작 분야에서 ‘센서 확장성과 데이터 효율’의 실용성을 높였다는 의미가 큽니다.



### GASE: Gaussian Splatting-Based Automated System for Reconstructing Embodied-Simulation Environments (https://arxiv.org/abs/2606.17520)
- **Prior Approaches**: 기존 연구는 생성 기반(generative) 방식이 빠르긴 하지만 sim-to-real 간극이 커지는 경향이 있고, 재구성 기반(reconstruction-based) 방식은 간극을 줄이려 하지만 전반적으로 자동화와 효율이 부족했다. 특히 monocular 중심 파이프라인은 넓은 장면 스캔에 시간이 많이 들고, 3D Gaussian 공간에서의 전경/배경 분리는 경계 불명확과 아티팩트로 인해 성능이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 3D Gaussian Splatting(3DGS) 기반으로 시뮬레이션 장면을 대규모로 자동 생성하는 시스템 GASE를 제안한다. 핵심은 파노라마 카메라 배열의 다중 뷰를 한 번의 스캔으로 처리하고, 전경/배경 분리를 2D 이미지 도메인에서 먼저 수행한 뒤 분리된 자산을 각각 생성·재구성해 물리 시뮬레이터에 바로 넣을 수 있게 만든 점이다.

- **Technical Challenges**: 기여를 위해 (1) 다중 frame stream에서 동일 객체의 마스크 정체성을 일관되게 유지하는 문제와 (2) 3D Gaussian의 표현 특성 때문에 생기는 경계/구조 아티팩트를 피하는 문제가 까다롭다. GASE는 카메라 pose와 깊이 정보를 이용해 2D 마스크를 프레임 전반으로 정합하고, SAM3로 초기 마스크를 만들고 SAM2로 비디오 전파를 보강한 다음, LAMA 기반 inpainting을 2D에서 수행해 안정적인 분리 품질을 확보한다.

- **Empirical Impact**: 평가에서 GASE는 3D Gaussian을 직접 다루는 분리 방식 대비 세그멘테이션 정확도를 10% 이상 개선하고, inpainting 품질에서도 state-of-the-art를 보였다. 또한 로봇 실험에서 sim-to-real 격차를 10% 미만으로 유지하며 조작 및 내비게이션 작업 모두에서 높은 성과를 보였고, 코드 공개도 예고되어 실무적 채택 가능성이 크다.



### MagicSim: A Unified Infrastructure for Executable Embodied Interaction (https://arxiv.org/abs/2606.17511)
- **Prior Approaches**: 기존 로봇 시뮬레이션은 주로 렌더링/컨트롤 테스트베드이거나, 특정 고정 작업 환경처럼 분리돼 활용되는 경우가 많습니다. 장기·상호작용 과업은 평가를 위해 ‘magic’ 액션으로 우회하거나, 학습/수집용 환경과 계획/상태가 이어지지 않아 같은 에피소드 재현·주석·검증이 어렵다는 한계가 있었습니다. 또한 플래닝을 외부 오프라인 전처리로 두면, 실패 재현이나 플래닝-루프 안에서의 실행 기록을 얻기 힘들었습니다.

- **Core Contribution**: MagicSim은 에피소드를 단위로 삼는 embodied interaction infrastructure로, 동일한 결정적(deterministic) 배치 런타임과 단일 MDP 위에서 world 구성, 실행, 평가, 데이터 수집, 에이전트 상호작용을 한 번에 잇습니다. YAML-first 명세로 콘텐츠·배치·행동·에이전트 노출을 분리해, 강체부터 유체/연성, 다양한 로봇 embodiment까지 폭넓은 실행 가능한 월드를 reset-and-step 루프에서 생성합니다. 또한 고수준 커맨드를 simulator 내부 상태 편집이 아닌 controller/atomicskills/플래너 프리미티브를 거쳐 로봇 액션으로 접지(grounding)합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 한 런타임에서 이질적인 물리(강체·연성·유체·지형·아바타·센서)를 공존시키는 월드 계약, (2) 병렬 대규모에서 재현 가능한 결정성(리셋 순서, snapshot, reset_to), (3) 단일 action space로 줄이기 어려운 multi-embodiment 실행, (4) planner-in-the-loop를 배치 시뮬레이션이 멈추지 않게 비동기 처리하는 구조, (5) 어디서/어떻게 잡았는지 같은 상호작용 인과구조를 보존하는 annotation-rich 데이터 생성입니다. MagicSim은 batched runtime에서 env별 독립 라이프사이클과 비동기 플래너 마이크배치 마진(microbatch) 해결, 그리고 성공 게이팅 기반의 에피소드 단위 저장을 조합해 이를 해결합니다.

- **Empirical Impact**: 논문은 MagicSim이 단일 태스크 정의로 RL 벤치마크 평가, autocollect용 자동 궤적 수집, VLM/에이전트용 상호작용 인터페이스를 동시에 지원해 연구 파이프라인을 통합할 수 있음을 강조합니다. 특히 성공한 롤아웃만 구조화된 멀티모달 트래젝토리로 저장해 언어 감독, 시각/기하 표현, 스킬·플래너·액션의 정렬을 제공함으로써 데이터 품질과 재사용성을 높입니다. 결과적으로 로봇 학습·데이터 생성·에이전트 구동을 하나의 재현 가능한 planner-in-the-loop 실행 기반으로 묶는 방향의 실용적 인프라로 의미가 큽니다.



### Two-Stage Fine-Tuning of ResNet50 for High-Sensitivity Melanoma Detection on Dermoscopic Images (https://arxiv.org/abs/2606.17504)
Comments:
          13 pages, 4 figures, 4 tables. Code available at this https URL

- **Prior Approaches**: 조기 발견 시 생존율이 높은 흑색종(melanoma)은 하지만 전이가 시작되면 예후가 급격히 악화된다. 기존에는 dermoscopic 이미지 분류에서 전이학습을 single-stage fine-tuning으로 바로 수행하거나, 클래스 불균형을 사후적으로 보정하는 방식이 흔했으나 과적합과 학습 불안정이 나타나기 쉽다.

- **Core Contribution**: 이 논문은 ResNet50을 이용한 흑색종 이진 분류에서 two-stage fine-tuning 전략을 제안한다. 1단계는 분류 head만 학습해 기존 시각 표현을 보존하고, 2단계는 모든 레이어를 저학습률(1e-5)로 함께 미세조정해 전이학습의 성능 저하를 줄인다. 또한 학습 데이터에만 random oversampling을 적용해 1:1 클래스 균형을 맞춘다.

- **Technical Challenges**: 핵심 기술적 문제는 클래스 불균형과, single-stage fine-tuning 시 발생할 수 있는 catastrophic forgetting이다. 이를 위해 stratified train/validation/test 분할 후 학습셋에서만 oversampling으로 균형을 만들고, 2단계 미세조정은 낮은 learning rate로 수행해 시각 특징의 급격한 손실을 막았다. 더불어 Stage별로 base freezing을 달리해 학습 안정성을 확보했다.

- **Empirical Impact**: 독립 테스트셋(3,826장)에서 AUC-ROC 0.9559, accuracy 88.34%, sensitivity 87.56%, specificity 89.13%, F1-score 88.29%를 달성했다. ablation 결과 two-stage 프로토콜이 single-stage fine-tuning 대비 sensitivity를 4% 이상 개선했고, Grad-CAM 시각화로 병변 영역을 올바르게 찾아내는 경향도 확인됐다. 또한 Streamlit 기반의 배포 가능한 탐지 애플리케이션과 학습 코드를 함께 제공해 재현성과 현장 적용성을 높였다.



### MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation (https://arxiv.org/abs/2606.17449)
Comments:
          To be presented at ACL 2026

- **Prior Approaches**: 기존 M-RAG는 retrieval-augmented generation 흐름에서 정적 파이프라인과 유사도 기반 필터링에 의존해, 시각-텍스트 충돌을 분리·판단하기 어렵다. 그 결과 cross-modal hallucination, causal fabrication, sycophancy가 자주 발생하며, 수정용 룰을 일괄 적용하면 정확한 생성까지 과도하게 깨지는 ‘intervention paradox’가 이어진다. 또 가벼운 LLM의 무가이드 다단 추론은 포맷 불안정으로 구조적 실패가 연쇄되며 논리적 드리프트를 키운다.

- **Core Contribution**: 이 논문은 MODE-RAG(Multimodal Objective Diagnostic Energy-RAG)로, Variational Free Energy(VFE)와 내부 attention states(ATLAS)를 이용해 개입 필요성을 동적으로 게이팅한다. FE-Router가 uncertainty가 높은 질의만 전문 멀티에이전트 파이프라인으로 라우팅하고, 나머지는 우회해 과잉 교정으로 인한 정확도 저하를 막는다. 또한 단계별(인식·검색·추론·생성)로 원인에 대응하는 에이전트를 두고, logit perturbation과 overseer 검증으로 sycophancy·논리적 조작·포맷 붕괴를 억제한다.

- **Technical Challenges**: 핵심 난제는 ‘언제 얼마나’ 개입해야 하는지이며, 정적 룰은 과잉 교정, 무가이드 추론은 실패 연쇄를 낳는다는 점이다. MODE-RAG는 VFE 기반 FE-Router로 고위험(Claim-Scene 충돌 등) 신호를 감지해 개입을 선택하고, Per-Agent의 atomic visual fact 추출로 ‘visual-first’ 앵커를 고정한다. 추론 단계에서는 Monte Carlo Tree Search(MCTS)로 인과 DAG를 구성해 temporal inversion/forced causality를 줄이고, Gen-Agent의 logit perturbation 및 overseer의 삼중 일관성 검사로 사용자 편향에 대한 과적합을 페널티한다.

- **Empirical Impact**: 평가를 위해 ModeVent를 제안하며, MultiVent에서 VFE 상·하위(불확실성 극단) 샘플을 골라 retrieval-시각 충돌과 manifold outlier에 강하게 테스트한다. Qwen-2.5-VL-7B 베이스라인 대비 MODE-RAG는 전체 평균 fidelity/resilience에서 일관된 개선을 보였고, 특히 Outliers에서 attention hijacking·majority text bias·out-of-domain irrelevance 같은 극단 실패를 크게 완화했다. 반면 비용은 질의당 처리시간이 평균 18.5초→26.2초(약 1.42×)로 증가하지만, 단계별 에이전트 개입 구조 덕분에 병렬화로 상쇄 가능성을 제시한다.



### AnnotateAnything: Automatic Annotation of 3D Assets for Robot Manipulation (https://arxiv.org/abs/2606.17446)
- **Prior Approaches**: 기존 로봇 데이터 자동화는 (1) 수동 어노테이션, (2) RL 기반 자동화로 크게 나뉘는데, 전자는 비용과 확장성이 문제였고 후자는 reward engineering과 학습·연산 부담이 컸습니다. 조작(Manipulation) 어노테이션 자동화도 grasp, contact, affordance 같은 단일 레이블에 집중하는 경우가 많아, 다양한 기술·물체·로봇 embodiment를 포괄하기 어렵다는 한계가 있었습니다. 즉 ‘수동 3D 자산 → 실제로 실행 가능한 조작 라벨’로의 자동 변환이 병목이었습니다.

- **Core Contribution**: 이 논문은 수동(passive) 3D 자산을 조작 실행에 바로 쓸 수 있는 라벨을 갖춘 자산으로 변환하는 자동 어노테이션 프레임워크 AnnotateAnything을 제안합니다. 핵심은 VLM(visual-language model)로 사람의 상호작용 priors를 추출·3D에 접지(grounding)한 뒤, 물리 기반 최적화와 대규모 병렬 시뮬레이션 검증으로 ‘실행 가능한’ action annotation(그립 포즈, dexterous contact, 관절 waypoint, 삽입/걸기 방향, 내비게이션 타깃 등)을 생성하는 것입니다. 또한 단일 해답이 아니라 다양한 실행 후보를 one-to-many 형태의 후보 뱅크로 보존합니다.

- **Technical Challenges**: 기여를 가능하게 한 기술적 난관은 네 가지로 요약됩니다: (C1) 사람의 상호작용 직관을 자동 추출해 자연스럽고 안전한 행동으로 연결, (C2) 이를 자산별 기하/운동 제약에 맞춰 실행 라벨로 변환, (C3) 가능한 해답들을 다양하게 유지, (C4) 자산·카테고리·작업을 per-asset 수동 설계 없이 스케일. 저자들은 언어 단계(자산/방(room) 레벨 기술, keypoint·part·affordance 큐 단서)에서 priors를 만들고, 물리 단계에서 candidate generation–trajectory generation–최적화–물리 검증–물리 aware augmentation까지 CUDA 가속 병렬 파이프라인으로 수행해 C1~C4를 동시에 맞췄다고 주장합니다.

- **Empirical Impact**: 실험에서 AnnotateAnything은 99개 소스 계열의 17,005개 자산을 처리해 181개(문맥상 18개로 보이는 ‘atomic skills’ 그룹 포함) 수준의 기술에 대해 총 1억 단위의 physics-validated action annotation을 생성했으며, 평균적으로 asset–skill 쌍당 수천 개 후보를 생성해 물리 검증 통과 후 최종 뱅크로 보존합니다. 시각-언어 번들 품질과 실행 성공/롤아웃 수집 효율에서 기존 어노테이션 파이프라인 및 VL-only/heuristic 변형 대비 우수한 성능을 보였고, downstream로 affordance/keypoint 감지, robotic VQA, 3D VLM instruction finetuning 같은 작업까지 활용 가능함을 보여줍니다. 결론적으로 ‘자산 수집’이 아니라 ‘조작 실행 라벨 변환’을 자동화해 시뮬 기반 로봇 데이터 수집의 실용성을 크게 끌어올린다는 점에서 의미가 큽니다.



### Edit3DGS: Unified Framework for Dynamic Head Editing via 2D Instruction-Guided Diffusion and 3D Gaussian Splatting (https://arxiv.org/abs/2606.17432)
Comments:
          SOICT 2025

- **Prior Approaches**: 기존 3DGS 기반 head avatar 연구는 주로 FLAME 같은 3DMM(3D morphable model) 제약 위에서 가우시안 변형/리깅을 학습해 신속한 렌더링과 애니메이션을 달성했지만, 머리카락·치아 등 비얼굴 요소나 극단적 표정 표현에는 한계가 있었다. 한편 2D text-to-image diffusion의 편집 감각을 3D로 옮기려는 시도는 있었으나 프레임/뷰별로 결과가 흔들리거나 아티팩트가 생겨 3D 재구성 시 불일치와 identity drift가 발생하기 쉬웠다.

- **Core Contribution**: Edit3DGS는 2D instruction-guided diffusion의 의미적 제어를 3D Gaussian splatting(3DGS)의 구조적 충실도와 결합해, 동적인 머리(표정·동작) 편집을 통합적으로 수행하는 프레임워크를 제안한다. 입력 비디오에서 편집 가능한 얼굴 영역을 마스킹한 뒤 텍스트 조건 편집으로 미세 조작(표정 변환, 어트리뷰트 수정, 외형 리파인)을 만들고, 이를 3D Gaussian fitting으로 합쳐 신원과 시간적 일관성을 함께 보존하는 것이 핵심이다.

- **Technical Challenges**: 가장 큰 난제는 2D 확산 편집을 다중 뷰·다중 프레임에 그대로 적용하면 서로 다른 뷰/시점 간 상호 의존성을 놓쳐 불일치와 표현 손실이 생긴다는 점이다. 논문은 이를 위해 multi-view batch editing(키 뷰 동시 편집 + feature injection)으로 뷰 간 일관성을 강제하고, eyes/mouth에 대해 auto-generated inpainting mask 기반의 latent inpainting을 적용해 시점이 바뀌어도 미세 표정을 복구하도록 했다.

- **Empirical Impact**: NeRSemble 데이터셋에서 novel view rendering, self-reenactment, cross-identity reenactment 전반에 대해 정성적으로 매끄럽고 photorealistic한 편집 결과를 보였고, CLIP-S/CLIP-C 정량 지표에서도 GaussianAvatar-Editor 대비 성능 격차가 매우 작거나 일부 항목에서 우위를 보였다. 특히 inpainting을 제거하면 눈·입의 원래 표정이 유지되지 않아 애니메이션 품질이 저하되는 실험으로, 제안 모듈의 실효성을 뒷받침한다.



### Where Should Action Generation Begin? A Learnable Source Prior for Generative Robot Policies (https://arxiv.org/abs/2606.17408)
- **Prior Approaches**: 최근 생성형 로봇 정책은 conditional sampling 형태로 diffusion 기반 또는 flow matching 기반 모델을 사용하지만, 대부분 action 생성의 시작점을 observation-independent standard Gaussian N(0,I)로 고정합니다. 이 때문에 generator가 “정보 없는 잡음”을 task-relevant action으로 옮기는 데 일부 계산 예산을 소모하고, source의 불확실성 설계는 상대적으로 덜 다뤄졌습니다. 또한 관측 기반 초기화를 쓰는 A2A, VITA, BridgePolicy 계열도 대체로 deterministic point estimate 또는 제한된 초기화로 uncertainty를 명시적으로 모델링하지 않는 경향이 있습니다.

- **Core Contribution**: 본 논문은 action 생성 시작점을 관측에 조건된 학습 가능한 source prior로 바꾸는 Learnable source Prior, LeaP를 제안합니다. LeaP는 proprioception(자세/관절 등 몸 상태) 특징만으로 state-adaptive 대각 가우시안의 평균과 분산을 예측해, observation-informed이면서도 stochastic한 초기 샘플을 제공합니다. 중요한 점은 generator 아키텍처와 inference solver, 생성 동역학은 그대로 두고 prior만 플러그인 형태로 교체한다는 것입니다.

- **Technical Challenges**: 핵심 기술 과제는 “초기 분포를 알기”와 “불확실성을 실제로 유용하게” 만드는 학습 신호를 동시에 구성하는 것입니다. LeaP는 prior 학습에 NLL(음의 log-likelihood)로 source 분포의 일치도를 맞추고, CLIP-style symmetric contrastive alignment로 샘플-타깃 대응성을 강화하며, flow loss를 통해 reparameterization 경로로 gradient을 generator와 함께 역전파합니다. 또한 diagonal Gaussian을 유지하면서도 mean과 state-adaptive variance를 함께 학습해, 표현력보다 로컬한 확률 질량 배치가 성능에 기여하도록 설계했습니다.

- **Empirical Impact**: RoboTwin 2.0의 15개 manipulation task에서 LeaP는 평균 success rate 81.6%를 달성하며 4개 대표 baseline 대비 6.5~25.5%p 향상됐습니다. 특히 NoPrior 대비 25.5%p 격차는 “표준 Gaussian 대체” 자체를 넘어, 학습 가능한 distributional prior가 성능의 원천임을 보여줍니다. 더 나아가 flow-matching과 diffusion-bridge 모두에서 일관되게 개선되며, 파라미터 수가 적고 더 빠르게 수렴하고, real-world Franka 배치에서도 최상 성능을 기록해 현장 적용 가능성을 강화했습니다.



### Contactless Respiratory Monitoring on Heterogeneous Mobile Robots: A Multimodal Edge-Computing Framework (https://arxiv.org/abs/2606.17376)
Comments:
          8 pages, 6 figures. To appear in Proceedings of the 8th International Workshop on IoT Applications and Industry 5.0 (IoTI5 2026), co-located with IEEE DCOSS-IoT 2026, Reykjavik, Iceland, June 2026

- **Prior Approaches**: 기존 비접촉 RR 모니터링은 영상에서 호흡 유발 미세 움직임을 optical flow, motion magnification, band-pass filtering 같은 신호처리로 추정하거나, RGB·NIR·thermal 멀티모달을 딥러닝으로 결합해 성능을 개선해 왔다. 하지만 조명 변화, 자세/거리 변화, 가림, 그리고 로봇 플랫폼·엣지컴퓨팅 제약 때문에 현장 배치에서 일관된 성능을 내기 어렵다는 한계가 반복됐다. 또한 모달리티(예: RGB vs thermal)별 품질이 달라지는 문제를 로봇의 센서 이종성까지 포함해 체계적으로 다룬 연구는 부족했다.

- **Core Contribution**: 이 논문은 이종 모바일 로봇에서 per-platform 알고리즘 튜닝 없이 RR을 산출하는 modality-adaptive 비접촉 모니터링 프레임워크를 제안한다. RGB·thermal·NIR·low-light 카메라를 밝기 조건에 따라 자동 선택하고, 어깨·엉덩이 keypoint로 흉부 ROI를 자세 변화에 강인하게 정렬한다. 마지막으로 window 단위로 signal-quality-index(SQI)를 계산해 신뢰 가능한 구간만 남기고, 하모닉 오차까지 보정해 최종 RR을 집계한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 조명/센서 성능이 시간·환경에 따라 급변하는 상황에서 안정적인 RR 신호를 뽑아내는 것과 (2) 자세 변화로 ROI 정렬이 깨질 때도 추정이 무너지지 않게 하는 것이다. 연구진은 초기 프레임 평균 밝기 기반으로 모달리티를 adaptive로 고르고, YOLOv11x-pose keypoint로 흉부 ROI를 geometry-adaptive로 구성해 자세 견고성을 확보했다. 더 나아가 SQI에서 spectral flatness, peak prominence, autocorrelation confidence를 결합하고 모달리티별 임계값으로 hard gating하여 잡음/부정합 구간을 제거하며, FFT·autocorrelation 추정의 하모닉 불일치와 이웃 window 간 일관성으로 오류를 줄였다.

- **Empirical Impact**: 3개 로봇 플랫폼(보행 형태와 엣지 컴퓨팅 구성이 다른 시스템)에서 조명(실내/야외, 명/암), 거리(2~8m), 자세(서기/앉기/눕기)별로 평가했으며, 플랫폼 간 알고리즘 재튜닝 없이 일반화되는 결과를 확인했다. 운영 한계는 모달리티별로 명확해졌는데 RGB는 최대 8m, NIR은 약 6m, thermal은 2m 수준, low-light는 완전 암흑에서도 최대 8m까지 가능했다(다만 눕기 자세의 long-range는 카메라 시점/기하 제약 영향). 또한 SQI 기반 유효 window 비율이 높은 범위(가시광에서 대체로 75~92%)로 유지되어, 재난/감염 위험 환경의 자율 triage·victim assessment를 위한 실전형 기반 기술로 의미가 크다.



### MM++: Unsupervised Scale-Invariant Multilayer OOD Detection via Top-K Gated Feature Fusion (https://arxiv.org/abs/2606.17352)
- **Prior Approaches**: 기존 OOD detection은 출력값(MSP/ODIN/Energy 등)이나 최종 특징에 의존해 중간 계층의 기하 정보를 충분히 활용하지 못하는 한계가 있었다. Mahalanobis 계열은 피처를 class-conditional Gaussian으로 보고 거리로 판별하지만, terminal(penultimate) 한 층에만 적용되거나(Mahalanobis++), 멀티레이어를 점수의 가산으로 합쳐 cross-layer 의존성을 버리는 문제가 컸다. 또한 멀티레이어 성능을 위해 proxy OOD 데이터나 classifier fine-tuning 같은 보정이 들어가면 ‘strictly post-hoc’ 전제가 깨질 수 있다.

- **Core Contribution**: MM++(Multilayer Mahalanobis++)는 완전 비지도(unsupervised), strictly post-hoc, 그리고 scale-invariant를 동시에 만족하는 OOD 탐지 프레임워크를 제안한다. 핵심은 멀티레이어 점수 가산을 버리고, entropy density drop으로 정보가 가장 큰 중간 계층을 Top-KK로 고른 뒤 penultimate layer를 앵커로 삼아 joint feature space에서 하나의 Mahalanobis++ 거리를 계산한다. 이를 통해 계층 간 ‘진화 궤적’ 불일치를 cross-layer 상관까지 반영해 포착한다.

- **Technical Challenges**: 어려움은 (1) 스케일 불변성과 계층적 표현력 간 trade-off를 지키면서, (2) 멀티레이어를 하나의 공간에서 안정적으로 거리 추정해야 한다는 점이다. MM++은 각 계층 특징을 ℓ2-normalize해 scale 문제를 줄이고, layer-wise covariance의 Shannon entropy를 통해 의미 압축이 급격히 일어나는 경계(entropy density drop)를 찾아 Top-KK를 선택한다. 마지막으로 Ledoit–Wolf 정규화된 tied covariance(precision matrix)를 joint 공간에 추정해 차원-표본 비율이 커도(특히 계층 concatenation 후) 안정적인 거리 계산이 가능하게 했다.

- **Empirical Impact**: ImageNet-1K(균형)와 ImageNet-LT(장기 꼬리)에서 near-OOD(ImageNet-V2/C/ES/R 등)·far-OOD(여러 강 도메인 시프트) 전반을 평가하며, 추가 OOD 데이터나 fine-tuning 없이도 경쟁력 있는 성능을 보였다. 특히 ImageNet-LT에서는 평균 AUROC 83.91%로 강한 강건성을 보였고, ViT·Swin·ConvNeXt 등 서로 다른 아키텍처에서도 일관된 결과를 냈다. 또한 additive fusion 기반의 X-Mahalanobis 대비 계층 상관을 joint space로 모델링하는 방식이 subtle한 near-OOD 차이를 더 잘 분리한다는 분석이 제시된다.



### ProCUA-SFT Technical Repor (https://arxiv.org/abs/2606.17321)
Comments:
          15 pages, 5 figures

- **Prior Approaches**: 기존 컴퓨터-유스 에이전트(CUA)는 스크린샷을 입력으로 받고 키보드/마우스로 동작하지만, 성능 향상은 결국 방대한 데스크톱 궤적 데이터에 묶여 있습니다. 공개 대규모 리소스인 AgentNet(인간 22.5K 궤적)은 규모에도 불구하고 UI-TARS 7B를 그 데이터로 SFT하면 OSWorld 성공률이 26.3%에서 8–10%로 급락하는 negative transfer를 보였습니다. 저자들은 이는 단일 앱 중심의 제한된 태스크 다양성, 복잡한 크로스-애플리케이션 추론 부재, 군집 주석의 잡음 때문이라고 진단합니다.

- **Core Contribution**: 이 논문은 ProCUA-SFT라는 3.1M step-level SFT 샘플(합성 궤적 93K, 2,484개 앱 조합)을 제안합니다. 핵심은 “불가능한 합성 태스크”를 줄이기 위해 태스크 생성에 precondition(이진 조건)을 포함시키고, 검증이 통과한 목표만 롤아웃하도록 파이프라인을 설계한 점입니다. 또한 단일 VLM(Kimi-K2.5)이 goal generator, precondition judge, trajectory executor를 모두 맡아 planner–actor capability gap을 줄입니다.

- **Technical Challenges**: 합성 데이터의 가장 큰 기술적 난제는 생성된 목표가 실제 데스크톱 상태(파일 존재, 앱 설치, 탭/서버 준비 등)와 불일치해 롤아웃 자원을 낭비하는 문제입니다. 이를 해결하기 위해 VLM이 목표와 함께 “현재 스크린샷/OS 설정에서 참인지 판정 가능한 이진 precondition”을 내놓고, judge가 각각을 독립 판정해 모두 통과할 때만 실행합니다(불통과 판정은 동일 VM에서 재시도에 피드백). 추가로 롤아웃 중의 컨텍스트를 정확히 재현하기 위해 각 궤적을 step-prefix 샘플로 확장하고, 가장 최근 3개 스크린샷은 vision으로 유지하며 오래된 단계는 텍스트 요약으로 바꿔 train/inference 불일치를 줄였습니다.

- **Empirical Impact**: UI-TARS 7B를 ProCUA-SFT로 1epoch 파인튜닝하면 OSWorld 성공률 45.0%를 기록하며, 베이스 모델 대비 18.7%p, AgentNet 학습 대비 35%p+ 큰 개선을 보입니다. 반대로 AgentNet으로는 유사한 학습에서 지속적인 성능 저하(8–10% 수준, plateau)가 관찰돼 데이터 품질의 중요성을 재확인합니다. 또한 Nemotron 3 Nano Omni 학습에 ProCUA 일부를 포함해 computer-use 능력을 향상시키는 사례도 제시하며, 이 연구가 CUA 학습데이터 설계에 미치는 실질적 영향이 큽니다.



### Phenotyping TPF via Self-Supervised Learning: A Label-Agnostic Framework with Expert Validation (https://arxiv.org/abs/2606.17295)
- **Prior Approaches**: 기존 연구는 Schatzker, AO/OTA 같은 분류 라벨을 기준으로 지도학습 CNN을 학습해 왔지만, 평면 X-ray에서 관찰자 간 판독 편차가 커 라벨 자체가 잡음으로 작동합니다. 그 결과 모델은 ‘안정적인 골절 형태’보다 ‘사람이 불일치하는 경계’를 재현하는 쪽으로 최적화될 위험이 있습니다.

- **Core Contribution**: 이 논문은 라벨 의존성을 제거하기 위해, 지도학습 대신 label-agnostic SSL(자기지도학습)로 골절 표현을 먼저 학습하고 이후 무감독 클러스터링으로 이미징 기반 phenotype을 도출하는 프레임워크를 제안합니다. 교육(학습) 단계에서는 Schatzker/AO/OTA 라벨을 전혀 쓰지 않고, 라벨과의 비교는 사후(post-hoc) 정렬 분석으로만 수행해 ‘학습 신호의 일관성 병목’을 우회합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 적은 데이터셋에서 안정적인 표현을 만들고 (2) 클러스터가 촬영 조건/잡음을 따라가지 않도록 전처리를 엄격히 설계하는 것입니다. 해결책으로 RadImageNet-pretrained ResNet-50을 SimCLR로 fine-tuning하고, 8단계 데이터 클리닝(ROI/정규화/중복 제거/하드웨어·관점 검수 등)과 보수적 증강 정책을 적용해 형태 관련 신호만 학습되도록 했습니다.

- **Empirical Impact**: PlaTiF 정제 데이터 154장을 바탕으로 4개의 클러스터 phenotype을 찾았고, 이들은 bootstrap ARI 0.319±0.041의 안정성과 silhouette 0.511의 내부 응집도를 보였습니다. 또한 블라인드 임상 검토에서 양측 리뷰어가 각 그룹의 coherence를 3–5/5로 부여했으며, ‘파편성(comminution)’은 감독 신호 없이도 한 phenotype에 대해 만장일치로 확인돼 임상 해석 가능성을 뒷받침했습니다. 마지막으로 Schatzker 경계와의 정렬은 ARI=0.013로 매우 낮아, 기존 분류 체계와 ‘서로 다른 축의 구조’를 포착하는 보완적 접근임을 보여주었습니다.



### Contrastive Action-Image Pre-training for Visuomotor Contro (https://arxiv.org/abs/2606.17256)
- **Prior Approaches**: 기존 로보틱스용 비전 인코더는 CLIP류의 image-text contrastive, DINO류 self-distillation 같은 ‘시맨틱 중심’ 학습이 주류였지만, 실제 조작 환경의 액션 구조나 직접적인 행동 슈퍼비전을 받지 못해 visuomotor 제어와의 정합성이 떨어집니다. 로봇 데이터로 학습하려 해도 trajectory 수집 규모가 인터넷급 영상보다 훨씬 작아, R3M·MVP 같은 접근은 프레임 레벨 대조학습이나 masked autoencoder 재구성 같은 목표로 우회해 액션 조건 정보를 누락하는 문제가 있었습니다. 또 egocentric human video 기반 방법들은 풍부하지만 로봇의 paired vision-action 신호가 없어 downstream 제어에 필요한 표현을 충분히 갖추기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 CAIP(Contrastive Action-Image Pre-training)을 제안하며, egocentric 인간 영상에서 추출한 손 자세를 로봇의 end-effector 액션을 대체하는 proxy로 사용합니다. 손의 3D 키포인트(인간 손 스켈레톤)를 액션 공간과 자연스럽게 대응시키고, contrastive objective로 이미지-텍스트 임베딩과 액션 임베딩을 정렬해 ‘액션 중심(action-centric)’ 시각 표현을 학습합니다. 결과적으로 적은 로봇 조작 데이터로도 물리 상호작용에 더 맞는 인코더를 만들 수 있는 확장 가능한 경로를 제시합니다.

- **Technical Challenges**: 핵심 과제는 대규모 인간 영상에 존재하는 ‘행동 정보’를 로봇 제어에 필요한 형태의 액션 신호로 변환해 학습 신호로 삼는 일이었습니다. 저자들은 MANO 기반 42 keypoint의 SE(3) 변환을 손 자세로 만들고, 델타 제어와 유사한 방식으로 미래 구간의 action chunk(대략 2초)를 구성해 액션 인코더로 모델링했으며, SigLIP 스타일 sigmoid contrastive loss로 대규모 배치에서의 학습 안정성을 확보했습니다. 또한 텍스트-조건 attention pooling을 통해 instruction과 연관된 이미지 토큰을 구성하고, 이 frozen vision encoder를 closed-loop 로보틱 정책에 전이해 성능을 검증합니다.

- **Empirical Impact**: 실험에서 CAIP는 Dexmate Vega + Sharpa Wave 손 기반의 정교한 dexterous manipulation 6개 태스크에서 평균 성공률 76%를 달성하며, DINOv2·SigLIP·MVP 등 강력한 비전 인코더 대비 큰 폭(평균 30%p 이상)으로 향상됐습니다. 특히 folding, pouring, fine-grained manipulation 같은 작업에서 30% 이상 성능 이득을 보이며, 태스크 전반에 걸쳐 일관된 강점을 나타냈습니다. action retrieval 및 out-of-distribution 액션 분류(선형 probing/zero-shot retrieval)에서도 CAIP가 기준선들을 능가해, 액션 중심 사전학습이 미지 환경과 데이터에도 잘 전이됨을 실증했습니다.



### Revisiting LLM Adaptation for 3D CT Report Generation: A Study of Scaling and Diagnostic Priors (https://arxiv.org/abs/2606.17213)
- **Prior Approaches**: 기존 의료 리포트 생성 연구는 2D 이미지 기반 VLM을 확장하거나(예: LLaVA 계열) 3D CT용 별도 파이프라인을 추가하는 방식이 많았다. 하지만 이들은 대개 (1) 큰 LLM을 fine-tuning해 계산 비용이 크거나, (2) 단순 linear projector로 시각-임상 의미 정렬이 약해 의미적 임상 갭이 남거나, (3) 의료 데이터가 적은 탓에 임상적 사실성보다 문장 유창성 중심의 clinical hallucination 위험이 있었다.

- **Core Contribution**: 이 논문은 3D CT 리포트 생성에서 LLM을 frozen vs fine-tuning할 때의 성능·일반화·계산 효율 trade-off를 모델 크기(96.1M~1.6B) 관점에서 체계적으로 분석한다. 그 위에 RAD3D-Prefix를 제안하며, frozen LLM에 임상 진단 priors를 prefix로 주입하되 학습해야 할 파라미터를 최소화한다.

- **Technical Challenges**: 3D CT는 입력이 고차원이고 진단 추론에 필요한 긴 임상 용어가 많아, 시각 임베딩과 임상 텍스트 의미 사이를 직접 잇기 어렵다. 논문은 CT-CLIP 기반 3D 비전 인코딩 결과에 multi-label diagnostic classification logits를 결합해 anomaly-aware prefix를 만들고, LLM은 고정한 채 transformer 기반 projection 네트워크만 학습함으로써 과적합과 임상 갭을 동시에 완화한다.

- **Empirical Impact**: CT-RATE(in-domain)와 INSPECT(out-of-domain)에서 자동 평가 지표와 임상가 reader study를 수행한 결과, RAD3D-Prefix는 유사한 parameter-efficient baseline을 능가하면서도 완전 fine-tuning 대비 훨씬 적은 trainable parameters로 성능을 낸다. 또한 크기 스케일링 실험에서 fine-tuning은 작은 LLM에서 유리하고, 약 1B+에서는 frozen+경량 projection 학습이 성능·일반화·효율의 균형이 더 좋다는 실용적 결론을 제시한다.



### HRDX: A Large-Scale Vector HD-Map Datas (https://arxiv.org/abs/2606.17080)
Comments:
this https URL

- **Prior Approaches**: 기존 벡터 HD map 구축 연구는 polyline/segmentation 계열 표현을 end-to-end로 예측하는 방향(예: MapTR, MapTracker 등)으로 발전했지만, 대다수 성능 병목이 데이터 규모와 라벨 풍부함의 한계를 크게 받습니다. 공개 데이터셋은 주행 커버리지가 짧고(대개 10시간 미만), geometry 중심에 비해 lane color·style, 도로 텍스트 같은 규제/행동 시맨틱 속성은 희소하거나 누락되는 경우가 많습니다. 또한 aerial imagery처럼 차량 주행 궤적과 정밀 정합된 모달리티가 부족해 cross-view 융합이나 privileged information 학습을 체계적으로 다루기 어렵다는 지적이 있었습니다.

- **Core Contribution**: 이 논문은 온라인 벡터 HD map 구축을 위한 대규모 데이터셋 HRDX를 제안합니다. HRDX는 약 40시간(1,400km) 규모의 최소 중복 주행을 모으고, 6대 동기화 surround camera, 128-beam LiDAR, RTK GNSS/IMU에 더해 차량 궤적과 정밀 정합된 aerial orthoimagery(최대 8cm/pixel)를 제공합니다. 라벨은 10개 벡터 map 클래스와 20+개의 semantic·topological 속성으로 확장되어, 기존 공개 데이터셋에 없던 규제/행동 시맨틱까지 포함합니다.

- **Technical Challenges**: 주요 과제는 (1) geometry뿐 아니라 속성 정확도까지 함께 평가/학습 가능한 학습 체계를 만드는 것과 (2) aerial을 정합·재현 가능하게 제공하되 추론 센서 비용을 늘리지 않는 활용법을 찾는 것입니다. 논문은 geometry-mAP(Chamfer-distance 기반)만 보던 한계를 보완하기 위해 Composite Score(CS)로 위치 정밀도와 attribute correctness를 동시에 측정하도록 설계합니다. 또한 aerial 기반 BEV 컨텍스트를 cross-attention 융합에 활용하되, 추론에서는 카메라만 쓰도록 teacher–student knowledge distillation(C+A/C) 프레임을 적용해 aerial의 이점을 카메라-only 학생에게 전이합니다.

- **Empirical Impact**: 실험은 HRDX의 데이터 규모 확장이 mAP과 CS를 단조적으로 끌어올린다는 결과를 보여주며, 대규모·다양한 주행 커버리지가 학습 안정성과 일반화에 핵심임을 시사합니다. 더불어 aerial imagery를 학습과 추론에 모두 사용하면 카메라-only 기준 대비 mAP과 CS가 각각 큰 폭으로 개선되며, stop lines·crosswalks·도로 경계·도로 텍스트처럼 전역 레이아웃과 가림(occlusion) 완화의 혜택이 큰 요소에서 상승이 집중됩니다. 마지막으로 aerial만 학습에 쓰고 추론은 카메라-only로 유지해도 성능 격차를 상당 부분 줄일 수 있어, deployment-feasible한 privileged information 활용 전략으로 의미가 큽니다.



New uploads on arXiv(cs.AI)

### EvolveNav: Proactive Preflection and Self-Evolving Memory for Zero-Shot Object Goal Navigation (https://arxiv.org/abs/2606.18235)
- **Prior Approaches**: 기존 Object-Goal Navigation은 RL/IL 기반 학습형이 많아 데이터·시뮬레이션 비용과 일반화 한계가 크다. 학습 없는 zero-shot 계열은 LLM/VLM을 의미 플래너로 쓰지만, 고정된 static prior에 머물러 실시간 피드백 적응이 약하고 동일한 실패를 반복하기 쉽다. 또한 trial-and-error 비용이 큰데도 post-hoc correction처럼 실행 뒤에야 오류를 줄이는 방식이 많아 단계 예산이 빠르게 소진된다.

- **Core Contribution**: 이 논문은 zero-shot 환경에서도 테스트 중에 계속 좋아지는 self-evolving ZS-OGN 프레임워크 EvolveNav를 제안한다. 핵심은 에피소드 경험에서 “행동 가능한 navigation rule”을 뽑아 agentic rule memory에 쌓고, 이를 UCB 기반 검색으로 선택해 탐색 정책을 갱신하는 것이다. 여기에 action 전에 실패 위험을 예측·필터링하는 memory-guided preflection 모듈을 더해, 사후 수정이 아니라 사전 위험 회피로 효율을 끌어올린다.

- **Technical Challenges**: 가장 큰 도전은 (1) 학습 없이도 경험을 “규칙” 형태로 일반화해 축적하고, (2) 늘어나는 규칙 중 어떤 것을 현재 장면에 쓸지 균형 있게 결정하며, (3) 실행 전 실패를 예측해 단계 낭비를 줄이는 것이다. 저자들은 궤적·시각 관찰에서 규칙을 distill하고, 단계별 의미 유사도 기반 credit assignment로 규칙의 support score를 만들며, UCB로 exploitation(검증된 규칙)과 exploration(새 규칙)을 동적으로 조절한다. 또한 LLM이 후보 frontier에 대해 failure prediction(위험 예측)을 수행하도록 프롬프트를 구성해 dead-end을 미리 거른다.

- **Empirical Impact**: 실험은 HM3D 및 MP3D(둘 다 Habitat Challenge 계열)에서 Success Rate(SR), SPL로 평가했으며, EvolveNav는 zero-shot 기준선을 전반적으로 상회한다. HM3D에서 SR 67.3%, SPL 33.9%로 기존 대비 소폭 우위를 보였고, 더 복잡한 MP3D에서는 SR 49.0%, SPL 19.1%로 강한 개선(특히 SR +4.5%p)을 보였다. 또한 preflection을 끌 때 SR이 HM3D +0.8%p, MP3D +3.5%p 하락하고, rule memory/UCB를 분리해도 MP3D에서 큰 이득이 나타나 온라인 적응과 사전 위험 회피가 성능 향상의 핵심임을 확인했다.



### Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers (https://arxiv.org/abs/2606.18206)
Comments:
          Code available at this https URL

- **Prior Approaches**: 추론 모델은 테스트 시 더 많은 compute를 쓰도록 설계되고 있으며, 대표적으로 Chain-of-Thought(CoT)는 연산을 verbalization으로 늘리고 halting token으로 멈춤을 결정한다. 다만 CoT는 손수 만든 추론 흔적과 특수 학습 체계가 필요해 end-to-end 학습이 복잡해진다. 또 다른 축인 looped architectures는 depth 방향으로 compute를 늘려 유연성을 주지만, halting을 고정/샘플링하면 적응성이 사라지거나, ACT 같은 외부 halting 모듈은 discrete 결정의 연속완화 때문에 최적화가 어렵다고 지적된다.

- **Core Contribution**: 이 논문은 “고정점으로의 수렴”을 end-to-end halting 신호로 쓰는 Transformer 기반 고정점 추론 모델 FPRM(Fixed-Point Reasoning Model)을 제안한다. 즉, 별도의 halting 네트워크 없이 hidden state가 fixed-point에 수렴하는지(잔차가 충분히 작아지는지)를 보고 반복을 멈춰 task difficulty에 따라 compute를 자동으로 조절한다. 또한 계층형 루프를 쓰던 기존 TRM/HRM과 달리, 더 단순한 비계층(non-hierarchical) 구조로도 효과를 낸 점을 강조한다.

- **Technical Challenges**: looped Transformer에서 반복 횟수가 늘면 unrolling된 효과적 깊이가 커져 signal propagation problem이 발생하고, post-norm은 안정성은 주지만 깊이에서 학습이 불리해질 수 있다. 이들은 pre-norm으로 전환하면 학습성은 좋아지지만 activation norm이 커져 발산할 수 있음을 확인했고, 이를 residual scaling으로 다시 안정화한다. 고정점 수렴 중에도 진동(oscillation)이 생길 수 있으므로, 고정점은 유지하면서 진동만 줄이는 damping과 residual 기반 중단 기준을 결합해 런타임에서 안정적으로 수렴을 유도한다.

- **Empirical Impact**: FPRM은 Sudoku, Maze, state-tracking, ARC-AGI-1 등 주요 reasoning 벤치마크에서 기존 대비 성능을 보였고, 특히 Sudoku-Extreme에서 더 낮은 cost로 TRM을 능가하는 결과를 제시한다. 7M 파라미터급 모델들 내에서 다양한 과제에 일관되게 강점을 보였으며, 계층형 구조 없이도 adaptivity(입력 난이도에 따른 compute 조절)를 구현했다는 점이 의미 있다. 결과적으로 “고정점 기반 halting + pre-norm/residual scaling 안정화” 조합이 looped reasoning에서 실제로 학습 가능하고 유연한 compute scaling을 제공함을 입증한다.



### The Stanford EDGAR Filings Dataset: Reconstructing U.S. Corporate and Financial Disclosures into Layout-Faithful and Token-Efficient Pretraining Data (https://arxiv.org/abs/2606.18192)
Comments:
          Preprint. Includes appendix, tables, and figures

- **Prior Approaches**: 기존 장문 데이터는 Common Crawl 같은 웹 원천을 쓰거나(C4·FineWeb 계열) 스팸·중복을 강하게 걸러도 문서 구조(테이블/들여쓰기/시각 위계)가 평탄화되는 경우가 많았습니다. 또한 EDGAR 기반 금융 코퍼스는 대체로 정리된 plain-text만 제공하거나 숫자 테이블을 보존하기보다 제거하는 편이라, 금융 추론에 중요한 레이아웃 신호를 잃었습니다.

- **Core Contribution**: Stanford EDGAR Filings Dataset(SEFD)은 SEC EDGAR(1994–현재) 원문을 레이아웃을 보존한 MultiMarkdown으로 “역재구성”해, 재무 문장·표·계층적 들여쓰기까지 장문 사전학습과 평가에 바로 쓰게 만든 오픈 코퍼스입니다. SEFD-v1은 152B 토큰 규모의 공개 스냅샷이며, Common Crawl 유사 중복이 0.1% 미만인 것이 특징입니다.

- **Technical Challenges**: SEC 제출물은 HTML/SGML/PDF 등 세대가 섞여 있고 ‘layout engineering’ 때문에 숫자·부호·기호가 셀 여러 칸으로 쪼개져 DOM 기반 추출만으로는 의미가 틀어집니다. SEFD는 2D 시각 좌표를 기준으로 파편 텍스트를 재결합하고, MultiMarkdown의 span 표기와 “Three-Column Hack”의 규칙 기반 병합(통화·퍼센트·괄호 부호 복원)을 통해 표의 의미를 복원합니다.

- **Empirical Impact**: 또한 SEFD 기반 벤치마크로 EDGAR-Forecast(지식 컷오프 이후 SEC 기록만으로 2026 수치 예측)와 EDGAR-OCR(복잡한 재무 표의 전사)을 제시하며, 각각 최상 성능에서 51.8%(GPT-5.5)와 75.78%(Qwen3.6-35B-A3B)를 보고합니다. 이는 EDGAR를 단순 검색용에서 벗어나, 장문 사전학습·문서 이해·준수(compliance)·정량 예측 평가까지 확장할 수 있음을 실증적으로 보여줍니다.



### DRFLOW: A Deep Research Benchmark for Personalized Workflow Prediction (https://arxiv.org/abs/2606.18191)
- **Prior Approaches**: 기존 Deep research(DR) 연구는 주로 보고서나 요약 생성에 초점을 맞춰, 사용자가 원하는 ‘구체적인 절차’까지 끝까지 복원하는 데는 상대적으로 덜 주목했습니다. 반면 기업 업무는 질문에 대해 일련의 action-step으로 이뤄진 workflow를 찾아내야 하는 경우가 많아, 생성형 성능만으로는 요구를 충분히 충족하기 어렵습니다. 또한 흩어진 heterogeneous sources에서 근거를 뽑아 단계 순서를 복구하는 평가 체계가 부족했습니다.

- **Core Contribution**: 이 논문은 에이전트가 예측해야 하는 ‘개인화된 워크플로우’를 평가하기 위한 벤치마크 DRFLOW를 제안합니다. DRFLOW는 5개 도메인 100개 태스크로 구성되며, 3,900개 이상의 소스에 근거한 1,246개의 reference workflow step을 제공합니다. 이어서 workflow 지향 reference agent인 DRFLOW-Agent(DRFA)도 제시해, 사용자 과업에 맞는 단계 시퀀스를 예측하도록 설계했습니다.

- **Technical Challenges**: 핵심 어려움은 (1) 산재한 근거 evidence를 정확히 식별하고, (2) 그 근거를 바탕으로 단계들을 올바른 structural ordering으로 조립하며, (3) 조건(condition)과 개인화 요구를 해결해 complete workflow를 복원하는 데 있습니다. 논문은 이를 위해 factual grounding, step recovery, 조건 해결, personalization 등 7가지 진단 지표를 정의해 단일 생성 품질이 아닌 워크플로우 특성을 세분 측정하도록 했습니다. 또한 DRFA는 workflow 예측에 맞춰 근거 기반으로 action-step 시퀀스를 구성하는 참조 에이전트로 검증합니다.

- **Empirical Impact**: 실험 결과 DRFA는 강력한 baseline 에이전트 대비 최대 10.02% 평균 F1 점수까지 개선했지만, 워크플로우 관련 지표들 전반에서 여전히 큰 개선 여지가 남아 있음을 보여줍니다. 이는 DR 시스템이 요약·리포트 생성 수준을 넘어, ‘완전하고 정확한 개인화 workflow’를 예측하는 것이 여전히 도전적인 frontier임을 실증적으로 시사합니다. 기업형 정보 탐색 에이전트의 평가 방향을 구체적 절차 복원으로 확장한다는 점에서 분야에 의미 있는 기준점을 제공합니다.



### Learning Cardiac Electrophysiology Digital Twins Through Agentic Discovery of Hybrid Structur (https://arxiv.org/abs/2606.18154)
Comments:
          10 pages, 4 figures

- **Prior Approaches**: 기존의 개인화 심장 EP 디지털 트윈은 반응-확산(reaction-diffusion) 모델에서 반응(ionic kinetics)과 확산(difffusion) 항 구조를 전문가가 정한 뒤 파라미터만 최적화하는 방식이 많았다. 이 접근은 해석 가능하지만 도메인 지식이 많이 필요하고, 구조가 고정돼 환자 간 전이가 어렵다. 한편 HyPer-EP 계열의 hybrid 방식은 유연성을 높였지만 여전히 물리-신경 컴포넌트의 조합 구조를 수동으로 설계해야 했다.

- **Core Contribution**: LEADS는 구조 선택 자체를 자동화하기 위해, LLM 에이전트가 물리 기반 반응 모델과 신경 확산 모듈을 ‘조합 가능한 구조 공간’에서 탐색하도록 설계했다. 핵심은 무제약 코드 생성 대신, 심장 EP 도메인 지식을 반영한 structured action space(카탈로그)로 안정적인 시뮬레이션을 위한 structural prior를 제공하는 것이다. 에이전트는 반복적인 Observe-Think-Act 루프에서 반응 모델 선택과 확산 모듈의 Select/Refine/Modify/Simplify 연산을 수행한다.

- **Technical Challenges**: 무제약 LLM 생성은 ODE 안정성, 물리적으로 의미 있는 상태변수, 수치적 안정성 같은 필수 제약을 충족하지 못해 심장 활성 예측이 붕괴하기 쉽다. LEADS는 후보 구조가 깨지지 않도록 Diffusion Catalog와 Reaction Catalog로 탐색을 제한하면서도, diffusion 모듈의 code-level Modify로 카탈로그 밖의 구조 변형도 허용해 open-ended 탐색성을 유지한다. 또한 structural 결정은 에이전트가 하고, 각 후보의 파라미터 피팅은 gradient descent로 처리해 학습 책임을 분리했다.

- **Empirical Impact**: 합성 데이터에서는 ground-truth 반응 모델 3종(AP/RM/MS)을 맞히며 평균 MSE에서 human-designed hybrid를 능가했고, 특히 HDTwinGen은 심장 활성 생성 실패(Uniform AT maps)로 크게 저조했다. 실데이터 Utah EGM에서는 activation time(AT) MAE에서 LEADS가 best human-designed hybrid를 앞지르거나 근접한 성능(5.52 vs 5.64)을 보이며, 시각적으로도 더 매끈하고 정답 AT 패턴에 가까운 결과를 냈다. 즉, 환자별 디지털 트윈에서 ‘구조 탐색 자동화’와 ‘물리 기반 안정성’을 동시에 노리는 접근이 실증적으로 효과적임을 보여준다.



### WEQA: Wearable hEalth Question Answering with Query-Adaptive Agentic Reasoning (https://arxiv.org/abs/2606.18147)
- **Prior Approaches**: 기존 접근은 손목형/웨어러블 센서 데이터를 텍스트로 요약한 뒤 LLM이 답을 생성하는 방식이 많다. 이때 시간적 신호 형태·교차-센서 상호작용 같은 정밀한 생리학적 정보가 손실돼, 복잡한 웨어러블 QA에서 성능 한계가 드러난다.
또 다른 흐름은 ReAct류 에이전트나 멀티에이전트로 툴 사용·계획을 확장하지만, 여전히 고정된 입력 표현(대개 사전 집계 특징)에 의존하거나 추론 경로 적응이 약하다는 지적이 있다.

- **Core Contribution**: WEQA(Wearable hEalth Question Answering)는 쿼리 의도에 따라 실행 경로를 동적으로 바꾸는 query-adaptive 에이전트 프레임워크를 제안한다. LLM 컨트롤러가 각 질문을 센서 분석 도구와 예측 모델의 “적절한 조합”으로 라우팅하고, 근거 기반(auditing) 검증을 통해 응답의 임상적 타당성을 높인다.
또한 4개 공개 웨어러블 데이터셋을 묶은 벤치마크를 구축해 기술 평가가 “분석형+예측형, 단·장기, 멀티모달”을 함께 반영하도록 설계했다.

- **Technical Challenges**: 핵심 난점은 연속적·고차원·장기 시계열 센서 데이터를 텍스트 중심 LLM 사전학습 분포와 정렬하기 어렵다는 점이다. WEQA는 센서-네이티브 도구로 시간/교차-센서 분석 및 예측을 수행하고, LLM은 계획 수립과 증거 종합을 담당하도록 역할을 분리해 이 미스매치를 줄인다.
또 질문마다 필요한 계산이 크게 달라서(가벼운 통계 vs 장기 시계열 추론 vs 웨이브포름 기반 예측) 고정 파이프라인이 불리하다—이를 해결하기 위해 단계별 증거를 누적하며 계획을 수정하고, 불확실성·개인화 맥락을 반영해 근거-감사(grounded response auditing)로 안전성을 보정한다.

- **Empirical Impact**: 실험 결과 WEQA는 LLM-only 및 기존 agentic 베이스라인 대비 정확도가 평균 24% 더 높았다. 특히 시간적 추론과 연속 신호에서의 예측이 필요한 태스크에서 격차가 가장 크게 나타났고, 분석·예측 전용 경로를 적응적으로 라우팅한 덕분에 쿼리당 토큰 사용도 크게 줄였다.
사람 평가에서도 블라인드 연구(의료 전문가 12명+사용자 8명)에서 유용성(usefulness)과 임상적 타당성(clinical soundness)이 뚜렷하게 개선됐다—단순 정답률을 넘어 “데이터 근거에 충실한 설명”과 “위험 커뮤니케이션의 적절성”이 강점으로 확인됐다.



### Memory as a Wasting Asset: Pricing Flash Endurance for Embodied Agents, and the Limits of Doing So (https://arxiv.org/abs/2606.18144)
- **Prior Approaches**: 기존 로봇용 embodied-memory 연구는 주로 무엇을 기억할지(when-to-write), 언제/어떻게 불러올지(what-to-remember)에 집중해 왔습니다. datacenter wear-aware caching은 잡음은 줄이고(내구성 기반 write-heavy 객체를 플래시에서 피하는) 한정된 endurance 맥락의 정성적 행동을 보이지만, 로봇에서는 RAM/NVM/cloud를 동시에 오가며 전력·지연과 비재생 endurance까지 함께 가격(erase cycle의 기회비용)으로 묶는 모델이 비어 있었습니다.

- **Core Contribution**: 이 논문은 로봇의 embodied memory를 ‘감가상각 자본’으로 보고, NVM의 program/erase cycle을 비재생 자산의 scarcity rent(내구성 shadow price) η로 가격화합니다. 그 결과 RAM/on-board NVM/cloud 배치가 ‘wear-augmented per-byte index’의 임계값(threshold)으로 결정되며, 비용 최적 배치는 value-write association χ의 부호(양/영/음)와 무관하게 기본형을 유지하되, χ>0일 때는 가치가 큰 메모리가 오히려 flash에서 멀어지는 비단조(non-monotone) 최적해가 나타남을 보입니다.

- **Technical Challenges**: 핵심 난제는 (1) erase cycle이라는 단일 비재생 예산을 intertemporal(시간에 걸친) 재무제약으로 모델링하고, (2) ‘가치와 write 강도의 결합’이 생기는지(χ의 부호)를 이론이 아니라 실제 로봇 로그로 측정해 조건부 정리를 실행 가능하게 만드는 것입니다. 논문은 WHEN→WHERE→WORTH 3단 구조에서 AURA의 when-to-write 게이트를 유지한 채 WHERE에서 RAM/NVM/cloud 배치를 wear-aware index로 풀고, χ는 미리 지정된 gate에서 실측해 χ>0 가지에서만 비단조 최적성이 적용되도록 “가짜 일반법칙” 위험을 차단했습니다.

- **Empirical Impact**: 실험/분석의 중심 결과는 χ의 부호가 배치 환경의 성격에 따라 달라진다는 점입니다(장기 반복적인 조작에서는 χ>0, 짧은 horizon에서는 χ≈0, 비재귀적 텔레오퍼레이션에서는 χ<0). 또한 내구성 예산이 datasheet 가격의 premium 3,000 P/E TLC에서는 거의 ‘비활성’이지만 commodity QLC/eMMC(~1,000 P/E)에서는 강하게 ‘구속’되며, 구속 구간에서 learned wear-aware 컨트롤러가 endurance-aware 성능을 보이되 과제 가치(task value) 자체 개선은 아직 데이터로 관측되지 않아 향후 과제로 남겼습니다.



### Your AI Travel Agent Would Book You a Bullfight: An Agentic Benchmark for Implicit Animal Welfare in Frontier AI Models (https://arxiv.org/abs/2606.18142)
- **Prior Approaches**: 기존 동물복지 벤치마크들은 질문-응답 형태로 모델이 텍스트에서 도덕적 추론을 얼마나 잘 드러내는지(예: LLM-as-judge) 주로 평가합니다. ANIMA, AHB, SpeciesismBench 등은 ‘말로는’ 복지 신호를 측정하지만, 도구를 쓰는 agentic 배치에서 실제 선택 행동이 전이되는지는 직접 확인하지 못했습니다.

- **Core Contribution**: 이 논문은 여행 예약을 대행하는 에이전트가 동물 착취 옵션을 피하는지 측정하는 최초의 agentic 벤치마크 TAC(Travel Agent Compassion)을 제안합니다. 사용자가 복지를 언급하지 않아도, 에이전트가 도구 호출로 결제까지 수행할 때 안전한 대안(관찰/보호/비동물 경험)을 선택하는 ‘revealed behavior’를 정량화합니다.

- **Technical Challenges**: 핵심 난제는 텍스트 평가의 ‘기준선’이 agentic 행동 위험을 과소평가한다는 점을, 재현 가능한 실험 설계로 분리해내는 것입니다. TAC는 12개 시나리오를 48개로 증강(가격/평점/노출 순서 뒤집기)하고 구매 tool 호출 결과를 규칙 기반으로 이진 점수화했으며, 복지 문장을 system prompt에 단 한 문장 추가(tac_welfare)해 개입 민감도를 점검했습니다.

- **Empirical Impact**: 7개 프런티어 모델 모두 기본 설정에서 ‘우연 수준(64%)’보다 낮아, 최고 성능도 53%에 그치며 모든 모델이 복지 선택을 충분히 보장하지 못함을 보여줍니다. 반면 복지-aware 문장 1줄만으로 일부 모델(Claude 및 GPT-5.5)은 47~63%p 큰 개선을 보였지만, DeepSeek과 Gemini는 12%p 미만으로 작아 모델별 내재된 복지 추론이 ‘기본 배치에서는 잠자고’ 있을 가능성을 시사합니다.



### Knowledge Reutilization in Meta-Reinforcement Learning (https://arxiv.org/abs/2606.18132)
Comments:
          18 pages initial submission

- **Prior Approaches**: 기존 meta-reinforcement learning은 관련 태스크에서 공통 구조를 뽑아 빠른 적응을 노린다. 하지만 end-to-end 방식은 태스크 추론과 embodiment(로봇/에이전트 고유 제어) 결합이 강해, 비정형(비모수)한 태스크 의미가 흐려지고 샘플 효율이 떨어지며 에이전트 간 재사용이 제한되는 문제가 있었다.

- **Core Contribution**: 이 논문은 메타 지식(meta-knowledge)을 task 단위로 재사용하는 프레임워크를 제안한다. dynamics-simplified agent에서 태스크별 지식을 먼저 학습하고, Bayesian non-parametric prior로 잠재 태스크 모드를 구조화한 뒤 high-level policy가 magnitude(세기/크기) 가이드를 생성해 이종 embodiment로 전이한다.

- **Technical Challenges**: 핵심 과제는 frozen된 메타 지식을 서로 다른 embodiment의 시간축과 하위 제어 요구에 맞게 정렬하는 것이다. 이를 위해 semantic-magnitude interface로 태스크 의미와 크기를 분리하고, lightweight temporal adaptor가 frozen 메타-지식을 embodiment-specific low-level controller가 수행 가능한 temporally aligned subgoal로 변환하도록 설계했다.

- **Empirical Impact**: 여러 로보틱 locomotion 에이전트 실험에서 최종-step tracking error가 최근 SOTA 대비 94.75%~99.79% 감소했다. 또한 deployment 성능은 비슷한 수준을 유지하면서 interaction data를 약 23.8%만 사용해, 교차 에이전트 재사용성과 샘플 효율 개선의 실증적 근거를 제시한다.



### First Proof Second Batch (https://arxiv.org/abs/2606.18119)
- **Prior Approaches**: First Proof 1차 실험은 공개된 수학 증명 챌린지 10개를 커뮤니티가 풀고, 연구자들이 규칙 없이 해설을 공개하는 방식으로 AI의 증명 능력을 간접 측정했다. 그러나 2차 배치에서는 테스트를 연구진이 직접 수행하고, 수학 분야 전문가들이 형식적으로 등급을 부여해야 한다는 한계가 남았다.

- **Core Contribution**: 이 논문(First Proof 2차)은 연구 수준의 수학 문제 10개를 선정·공개하고, ChatGPT 5.5 Pro 및 여러 학계 harness를 대상으로 “원샷(one-shot) 입력→24시간 내 출력→LaTeX 컴파일”이라는 재현 가능한 벤치마크를 구성해 평가 체계를 공식화한다. 또한 문제 원문, 인간 해법, AI 해법, 심사 로그까지 전 과정을 공개해 투명성을 극대화한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) LLM이 이해 가능한 형식(.tex)으로 문제를 제공하면서도 (2) 문헌의 인접 결과만으로 즉시 답이 나오지 않게 새로움/비표준 인사이트를 보장하고 (3) “증명”을 실제로 재현 가능하게 LaTeX로 산출·컴파일시키는 것이다. 이를 위해 사전 점검으로 여러 모델을 Zero Data Retention 환경에서 시험해 즉답 가능성을 배제하고, 토큰/로그를 남기며, 최소 2명 이상의 분야별 referee가 정확성과 참신성, 인용 품질을 기준으로 flawless~rejected로 판정한다.

- **Empirical Impact**: 4개 시스템을 10문제에 적용했을 때 최소 1개 passing grade(essentially flawless 또는 minor revisions)를 받은 문제는 총 7개이며, 일부는 인간 해법과 다른 novel argument로 심사위원을 설득하기도 했다. 반대로 특정 문제(예: metric geometry)에서는 상당한 진전을 못 보였고, 많은 경우 “standard arguments”로 핵심 단계를 생략하거나 잘못된/누락된 인용 같은 실패 패턴이 반복되었다. 전반적으로 수학적 증명 생성의 강점(루틴 부분의 정교한 전개)과 취약점(결정적 갭, 참조 신뢰성) 모두를 데이터로 드러내며, 이후 커뮤니티 실험까지 확장 가능한 평가 인프라를 제시한다.



### Trust the Right Teacher: Quality-Aware Self-Distillation for GUI Grounding (https://arxiv.org/abs/2606.18101)
- **Prior Approaches**: GUI grounding은 고해상도 스크린샷에서 작은 UI 요소를 찾아 정확한 화면 좌표를 출력해야 해서, 기존에는 SFT처럼 정답 좌표를 하드 라벨로 학습하거나 GRPO 같은 결과 보상 중심 강화학습이 많이 쓰였다. 하지만 SFT는 soft supervision(teacher의 불확실성·선호 분포)을 충분히 활용하지 못하고, GRPO는 sparse reward로 인해 비용이 크며 미세한 좌표 학습 신호가 약하다. OPSD( on-policy self-distillation )는 dense 토큰 단위 teacher 신호를 주지만, student가 만든 prefix가 이미 목표에서 벗어나면 teacher의 좌표 토큰 신호 품질이 흔들려 naive OPSD가 GUI grounding에 그대로 적용되기 어렵다.

- **Core Contribution**: 이 논문은 GUI grounding용 quality-aware self-distillation을 제안한다. 핵심은 좌표 digit 토큰에 대해 teacher 신호가 현재 student prefix에서 목표 박스(ground-truth box)로 완성 가능한지 “정답 공간 호환성”을 기준으로 신뢰도를 조정하고, 그 위에 teacher probability로 증류 강도를 추가 캘리브레이션하는 것이다. 특히 soft correctness-aware gating과 teacher-probability scaling을 함께 쓰면 일관된 성능 향상이 나타나며, 단독 적용은 오히려 불안정할 수 있음을 보인다.

- **Technical Challenges**: 기술적 난관은 on-policy 설정에서 teacher가 student prefix에 조건부로 계산되기 때문에, prefix가 틀어졌을 때 teacher의 다음 좌표 digit이 ‘그럴듯한 오답 계속’으로 변할 수 있다는 점이다. 이를 해결하기 위해 coordinate axis(x/y)별 ground-truth 박스 구간을 이용해 teacher top-1 좌표 digit이 남은 자리로도 목표 박스 안의 좌표로 완성 가능한지(prefix-aware) 검사하고, 불가능하면 해당 토큰의 distillation weight를 0.5배로 down-weight하는 soft gating을 둔다. 동시에 teacher probability(teacher의 top-1 좌표 digit에 대한 확신)를 곱해, 호환성은 맞지만 불확실한 분포를 덜 강하게 학습하도록 scaling을 더한다.

- **Empirical Impact**: Qwen3.5-9B 백본으로 6개 GUI grounding 벤치마크에서 평가한 결과, 제안 방법은 macro-average accuracy 72.23으로 base와 강한 post-training baseline(GUI-SD 등)를 일관되게 능가했다. GUI-SD와 비교하면 단순 엔트로피 기반 가중치처럼 “정답 호환성”을 직접 보장하지 못하는 약점을, 이 논문은 공간적으로 verifiable한 신뢰도 기준으로 교정한다. 또한 ablation에서 gating과 scaling을 결합했을 때만 안정적으로 개선되며, 이는 두 메커니즘이 각각 ‘잘못된 좌표 신호 억제’와 ‘남은 신호의 정밀한 강도 보정’을 분담한다는 해석을 뒷받침한다.



### IsabeLLM: Automated Theorem Proving Applied to Formally Verifying Consensus (https://arxiv.org/abs/2606.18098)
- **Prior Approaches**: 기존 AI for theorem proving은 Isabelle 같은 증명 보조기에서 자동화/반자동화를 돕는 방향에 집중했지만, 도메인 특화 맥락이 부족하면 LLM이 환각적으로 잘못된 증명 스텝을 반복하기 쉽다. 블록체인 검증에서도 전통적인 formal verification은 가능하지만 비용과 전문성 장벽이 커 ‘correctness-by-construction’을 현장에 적용하기 어렵다는 한계가 있었다. 또한 기존의 LLM-증명 도구는 Isabelle/Sledgehammer 버전 불일치나 느린 탐색으로 효율이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 IsabeLLM(Isabelle + LLM 자동 정리)에서 한 단계 더 나아가 IsabeLLM-RAG를 제안한다. 핵심은 Retrieval-Augmented Generation으로 이전에 증명된(이진 트리 모델) 관련 정리를 검색해 더 강한 맥락을 LLM에 제공하고, Nitpick 기반 반례(counterexample)와 error trace(수정 이력)를 함께 넣어 논리적 실패와 반복 루프를 줄이는 것이다. 아울러 Isabelle 2025 및 최신 Sledgehammer와의 호환, 공정(process) 통합 및 LLM 모델 선택 최적화로 효율도 개선했다.

- **Technical Challenges**: 가장 어려운 점은 LLM이 생성한 증명 텍스트가 Isabelle에 주입 가능한 정확한 문법/구조를 갖추지 못해 빌드 오류가 반복된다는 점이다. 이를 위해 LLM 호출마다 직전까지의 수정 내역을 error trace로 제공해 ‘이미 고친 부분을 다시 망가뜨리는’ 문제를 완화하고, Nitpick이 논리적으로 거짓인 목표를 찾아 LLM이 그 방향을 즉시 피하도록 컨텍스트를 보강한다. 또한 RAG는 .thy에서 정의/정리와 의존성을 추출해 필요한 transitive dependency까지 포함한 뒤, 로컬 ChromaDB에서 유사 정리를 검색해 프롬프트에 주입하는 방식으로 self-contained 맥락을 만들었다.

- **Empirical Impact**: 실험은 Bitcoin Proof of Work 합의의 검증 능력을 기준으로 IsabeLLM과 IsabeLLM-RAG를 비교하며, 더 복잡해진 n-ary 트리 합의 모델에서의 증명 완료 성과를 관찰한다. 결과적으로 도메인 유사 정리 제공(RAG), 반례 기반 실패 회피(Nitpick), 수정 이력 기반 누적 수정(error trace), 그리고 Isabelle 2025/Sledgehammer 효율화가 결합되어 증명 성공률과 완료 효율을 끌어올리는 방향을 보여준다. 이는 블록체인 합의 프로토콜처럼 안전성과 정확성이 중요한 영역에서 LLM 기반 자동 theorem proving의 실사용 가능성을 넓히는 의미가 있다.



### A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation (https://arxiv.org/abs/2606.18075)
Comments:
          Accepted at The ACM Web Conference 2026 (WWW '26)

- **Prior Approaches**: 기존 graph RAG는 크게 엔티티 중심과 청크(조각) 중심으로 나뉜다. 엔티티 중심은 관계 경로 탐색으로 multi-hop 추론에 강점이 있지만, text→entity 추출 과정의 정보 손실과 문맥 누락 문제가 남는다. 청크 중심은 계층 요약으로 문맥을 잘 보존하지만 청크 사이에 흩어진 엔티티 간 관계를 명시적으로 결합하기 어렵다.

- **Core Contribution**: 이 논문은 chunk 문맥과 entity 관계를 단순 혼합하는 수준을 넘어, 둘을 진짜로 융합한 “새 표현”을 만들기 위해 HyGRAG(하이에르라키컬 그래프 RAG)를 제안한다. 하이브리드 그래프를 만들고, 이를 계층적으로 클러스터링한 뒤 LLM이 문맥+관계를 함께 요약해 emergent knowledge에 대응하는 표현을 생성한다. 이후 해당 표현을 기반으로 맥락과 논리(관계)를 동시에 검색해 LLM 생성에 전달한다.

- **Technical Challenges**: 핵심 과제는 (1) 문맥과 관계를 단순 나열이 아닌 통합 요약으로 생성하는 것, (2) 생성된 통합 요약을 검색에서 실제로 활용해 원문에 없는 추론 신호를 끌어오는 것, (3) 동적 코퍼스에 대해 계층 구조를 매번 재구축하지 않는 것이다. HyGRAG는 구조 인식 임베딩과 커뮤니티 기반 LLM 요약으로 통합 표현을 만들고, context-aware retrieval(모든 계층 검색)+relation-aware retrieval(커뮤니티 확장 후 triplet 필터링)로 검색을 이중화한다. 또한 attachment-based 업데이트로 새 콘텐츠를 가장 유사한 커뮤니티에 국소적으로 붙이고, 영향을 받는 경로만 부분 요약을 전파한다.

- **Empirical Impact**: 실험에서 HyGRAG는 multi-hop reasoning에서 평균 정확도를 9.7% 향상시키며, PopQA에서는 6.2%, HotpotQA·MuSiQue 등에서도 일관된 개선을 보였다. 동적 코퍼스 확장 시에도 계층 업데이트가 국소 재요약으로 처리되어 효율을 유지하는 방향성을 확인했다. 전반적으로 엔티티-관계 기반 추론 강점과 청크-문맥 보존 장점을 “융합 표현+계층 검색”으로 결합해 graph RAG의 실사용 품질을 끌어올렸다는 점에서 의미가 있다.



### Agentic AI-based Framework for Mitigating Premature Diagnostic Handoff and Silent Hallucination in Healthcare Applications (https://arxiv.org/abs/2606.18068)
- **Prior Approaches**: 기존 연구들은 RAG, chain-of-thought, self-consistency decoding, 그리고 “LLM-as-a-judge” 같은 프레임워크로 의료 오류를 줄이려 하지만, 확률적 생성에 의존해 결정적 보장은 어렵습니다. 특히 대화형 에이전트가 OLDCARTS 같은 구조화된 병력 청취를 “완료했는지”를 형식적으로 검증하기보다는 모델의 instruction-following에 맡기는 경우가 많아 조기 진단 핸드오프와 무검출 hallucination 위험이 남습니다. Multi-agent 접근도 대체로 라우팅을 프롬프트 제약으로 처리해, 안전한 전환 규칙을 강제하는 상징적 게이트가 부족했습니다.

- **Core Contribution**: 이 논문은 의료 대화형 에이전트의 두 실패 모드(조기 진단 핸드오프, 무검출 임상 hallucination)를 동시에 겨냥하는 다중 에이전트 프레임워크를 제안합니다. LLM-as-a-judge 라우팅을 빼고, 시스템 레벨에서 결정적 orchestration constraint를 도입해 진단으로 넘어가기 전 병력의 완전성을 강제합니다. 또한 semantic entropy 기반 epistemic uncertainty quantification(UQ) 게이트로 서로 다른 진단 샘플 간 발산이 큰 경우를 사전에 가로채 안전 검토를 유도합니다.

- **Technical Challenges**: 핵심 난제는 (1) 자연어 대화에서 수집되는 병력을 OLDCARTS 차원별로 기계 판독 가능하게 추적하면서, (2) 동시에 LLM의 의미적 불일치를 “표현 변형”까지 고려해 안정적으로 불확실성으로 계량하는 것입니다. 저자들은 neuro-symbolic state-tracking gate(M1)로 OLDCARTS 8개 필드(Onset~Severity)를 모두 수집하기 전에는 진단 전환을 차단하고, semantic entropy(UQ, M2)를 위해 같은 케이스에 대해 K=5개의 독립 진단 샘플을 생성한 뒤 NLI 기반 의미 클러스터링으로 H를 계산해 임계값 초과 시 안전 감독자에게 전달합니다. 마지막으로 최종 진단 생성은 온도 0.0의 결정적 에이전트로 수행하되, 불확실성 신호는 자동 거부가 아니라 안전 검토 컨텍스트로만 사용하도록 설계했습니다.

- **Empirical Impact**: llama-3.1-70b-instruct를 기반으로 한 시뮬레이션 환자 에이전트와 MedQA에서 파생된 150개 테스트 케이스에서, 전체 아키텍처는 무제약 기준선 대비 진단 정밀도(precision)를 11.3%p 절대 개선해 49.3%를 달성했습니다. 또한 OLDCARTS completeness(σ)와 semantic entropy(H) 사이에 통계적으로 유의한 음의 상관관계(r=-0.181, p<0.05)를 관찰해, 구조화된 정보 수집이 진단 불확실성을 낮춘다는 근거를 제시했습니다. 이는 의료 에이전트에서 “정확한 진단 생성”뿐 아니라 “진단 전환의 규칙적 검증”을 결합해야 한다는 실무적 메시지를 강화합니다.



### PseudoBench: Measuring How Agentic Auto-Research Fuels Pseudoscienc (https://arxiv.org/abs/2606.18060)
Comments:
          26 pages, 21 figures

- **Prior Approaches**: LLM 기반 에이전트는 계획·도구 사용·실행·보고까지 자율화되면서 과학 연구 워크플로에도 적용되고 있지만, 기존 연구는 주로 특정 과업 성능이나 일반 안전 이슈를 다뤄왔다. 또한 환각이나 hallucination, sycophancy 같은 문제는 논의돼 왔으나, ‘의사과학을 검출·거절하는지’를 에이전트 수준에서 end-to-end로 정량 평가한 벤치마크는 부족했다.

- **Core Contribution**: 이 논문은 의사과학 서사를 ‘생성·증폭’하는지의 반대인, 의사과학에 ‘저항·거절’하는 능력을 평가하는 PseudoBench를 제안한다. Wikipedia와 MinKe 커뮤니티에서 의사과학 claim-evidence를 수집해 5개 범주로 정리하고, 에이전트가 실험 설계→실행→분석→논문형 보고서 작성까지 수행하도록 만들어 결과물을 평가한다. 동시에 Report Quality, Pseudoscience Alignment, Persuasiveness 3축으로 논문 단위 판정 프로토콜을 마련해 진단 가능성을 높였다.

- **Technical Challenges**: 핵심 과제는 (1) ‘not even wrong’처럼 검증 불가하지만 그럴듯한 주장을 섞어, 실제 과학 탐색을 억누르지 않으면서도 의사과학 저항을 측정하는 데이터 설계를 하는 것이다. 이를 위해 seed filtering·cross-source 표준화·semantic deduplication·absurdity scoring·인간 검수를 거쳐 200개 대표 쌍을 구축했다. 또 에이전트가 텍스트를 짧게 답하는 대신 완결된 PDF 논문을 내도록 강제해, 생성물의 신뢰도(설득성)까지 포함해 LLM-as-judge로 paper-level 평가를 수행한다.

- **Empirical Impact**: 7개 SOTA auto-research 에이전트를 시험한 결과, 거절(refusal)률이 거의 0에 수렴하며 전체 resistance의 최고값도 27.4%에 그쳤다. 특히 claim과 evidence를 유지한 채 학술 형식과 과학적 문장력으로 포장해, 구조적 완성도와 설득성이 동시에 높게 나타나는 ‘epistemic safety 불일치’가 확인됐다. 저항이 더 약한 도메인은 반박이 즉각적이지 않은 영역(예: 물리·공학·지구과학 스캐폴딩)에서 나타나며, 더 강한 도메인에서도 결국 의사과학이 세련된 형태로 남을 수 있어 배포 전 scientific alignment의 긴급성을 제기한다.



### ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents (https://arxiv.org/abs/2606.18037)
Comments:
          20 pages, 4 figures

- **Prior Approaches**: 기존 factuality(사실성) 검증은 claim이 어떤 근거 어딘가에 의해 지지되는지에 초점을 두는 경우가 많다. RAGAS·AlignScore·SummaC-ZS 같은 방식은 풀링된 evidence나 검색 컨텍스트에 대한 faithful 여부를 보지만, MCP 같은 툴 사용 에이전트에서 “어떤 소스에 귀속(attribution)됐는지”는 직접 평가하기 어렵다. 그 결과, 교차 소스가 섞여 있어도(예: 차트 사실을 논문으로 잘못 인용) pooled evidence 기반으로는 통과될 수 있다.

- **Core Contribution**: 이 논문은 MCP-grounded 답변에서 발생하는 provenance 민감 실패 모드인 cross-source conflation(서로 다른 소스 간 귀속 혼동)을 정의한다. 이를 해결하기 위해 source-aware verifier ProvenanceGuard를 제안하며, 답변을 원자 단위 claim으로 분해하고 claim별로 라우팅된 MCP source에 한정해 지지 여부를 판단한 뒤, 답변이 명시/암시한 귀속 소스와 실제 라우팅 소스가 일치하는지도 검증한다. blocked 판정된 답변은 retrieval-augmented answer revision 후 같은 verifier로 재검증하는 repair-and-reverify 루프도 함께 구성한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) claim-단위로 evidence를 분해·라우팅하고 (2) “지지(support)”와 “정확한 소스 소유(source ownership)”를 동시에 판정해야 한다는 점이다. 저자들은 MCP trace의 stable tool ID·source ID·raw output을 유지한 채 claim을 분해하고, claim에 가장 관련 있어 보이는 source별 evidence를 선택한 뒤 NLI(entailment/neutral/contradiction)와 토큰 정렬/보호 값(protected value) 일치 같은 grounding 보조신호를 사용한다. 마지막으로 랜덤포레스트 기반 calibrator로 routed source에 대한 supported/blocked 경계를 조정해, 단일 점수에 의존한 오판을 줄인다.

- **Empirical Impact**: 의료 도메인 MCP-agent trace 281개(held-out 40 trace, 361개 claim/label)에서 ProvenanceGuard는 block F1 0.802, source accuracy 0.858를 기록하며 source-blind baseline 대비 attribution 차원까지 성능 이점을 보였다. 또한 더 어려운 multi-source 벤치마크에서는 block F1 0.846을 달성했지만, 의미적으로 가까운 소스가 많아질수록 source-plus-relation 정확도는 0.229로 떨어져 “정확한 소스 소유”가 여전히 어려운 축임을 보여준다. 흥미롭게도 50개의 통제된 임상 conflation probe에서는 삽입된 attribution swap을 모두 탐지했으며, 전체 trace 세트에서는 repair-and-reverify로 blocked 답변을 전부 해결(대개 보수적 fallback 포함)했다고 보고한다.



### LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI (https://arxiv.org/abs/2606.18021)
Comments:
          15 pages, 5 figures; Published at the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML 2026

- **Prior Approaches**: 기존 법률 AI 연구는 환각의 유형이 작업마다 다르다는 점은 보여주지만(예: task별 환각률 58~88%), 계약서 추출(contract extraction)에서는 claim 유형별 실패가 어떻게 “법적 노출”로 이어지는지까지는 명확히 다루지 못했다. 또한 CUAD 같은 oracle 기반 평가를 사용하더라도 전체 평균 환각률로는 오류가 집중되는 범주와 오류의 방향(누락 vs 발명)을 분해하지 못해, 컴플라이언스 담당자가 실행 가능한 신호를 얻기 어렵다. Multi-agent debate는 사실성 메커니즘으로 연구돼 왔으나, 고위험 환경에서 특정 모델의 실제 실패 모드에 맞춰 보정(calibration)하는 방식은 제한적이었다.

- **Core Contribution**: 이 논문은 LegalHalluLens라는 감사(auditing) 프레임워크를 제안해, 법적으로 검증 가능한 4개 claim 범주(숫자, 시간, 의무/권리, 사실)에 대해 typed hallucination profiles를 제공한다. 여기에 omission(누락)과 invention(발명) 편향을 한 점수로 요약하는 Risk Direction Index(RDI)를 도입해, 평균 환각률이 가리는 “오류 방향”을 배포 의사결정에 쓸 수 있게 만든다. 마지막으로 Experiment 1의 진단을 그대로 반영해 typed debate pipeline을 보정하고, 작은 오픈 모델도 상용 API 수준 성능을 저비용으로 노릴 수 있음을 보인다.

- **Technical Challenges**: 핵심 난제는(1) claim 유형이 다르면 오류의 영향이 다름에도 평균 지표로는 실패 모드가 섞여 사라진다는 점, (2) 누락과 발명이라는 방향성을 별도의 주석/추가 호출 없이 운영 가능한 단일 신호로 압축하는 점, (3) debate 완화가 제네릭 튜닝이면 실제 실패 범주에 집중하지 못한다는 점이다. 이를 위해 CUAD v1.0의 판정 라벨(mismatch_type)에서 missing_condition/extra_condition을 사용해 RDI를 정의하고, Skeptic 질문과 Add/Delete gate 비대칭(구조적 오류 타깃)을 claim 유형·방향 진단에 맞게 조정한다. 또한 구조적 추출 오류는 답변 토론이 아니라 재추출(re-extractor)로 처리해, “고칠 수 없는” 잘못을 대화로 끌고 가지 않도록 설계한다.

- **Empirical Impact**: 510개 상업 계약(총 249,252개 clause-level 인스턴스)에서 모델 간 HalTP는 50.9~56.5%로 비슷해 보이지만, typed profile을 적용하면 숫자·의무 범주가 시간 범주보다 훨씬 더 크게 실패하며(약 38~40%p 격차) 평균이 법적 노출의 핵심을 숨긴다는 점이 드러난다. 더 나아가 52% 수준으로 동일해도 RDI는 부호/방향이 달라져, 상용 API 간조차 “리뷰어가 감당해야 할 형태의 리스크”가 달라질 수 있음을 실증한다. 완화 실험에서는 보정된 typed debate pipeline이 fabricated detections를 45% 줄였고, 4B active 파라미터의 오픈 모델이 상용 API와 유사한 종합 점수 경쟁력을 보이면서(상대적으로 더 낮은 추론비용) 진단 기반 보정의 실효성을 확인했다.



### LLM Consumer Behavior Theory: Foundations of a Novel Research Field (https://arxiv.org/abs/2606.18005)
- **Prior Approaches**: 기존 소비이론은 소비자가 직접 의사결정을 내린다는 가정에 기반해 왔지만, 최근에는 LLM 기반 에이전트가 구매·의사결정을 대행하는 흐름이 생기며 모델링 공백이 드러났다. 또한 LLM 의사결정, 인간 행동 시뮬레이션, preference elicitation 연구는 각기 다른 관점으로 흩어져 있어, 에이전트 수준의 선택이 시장 수요로 어떻게 집계되는지에 대한 일관된 프레임이 부족했다.

- **Core Contribution**: 본 논문은 LLM Consumer Behavior Theory라는 새 연구 분야를 제안하며, agentic markets에서 소비자 행동(에이전트 행동)을 경제학적 렌즈로 분석하는 틀을 정립한다. 특히 인간 선호가 LLM 에이전트에 의해 어떻게 반영되고 실제 행동으로 이어지며, 그 결과가 market demand로 어떻게 집계되는지에 대한 통합적 형식화를 제시한다.

- **Technical Challenges**: 핵심 난제는 에이전트 시장에서 합리성(rationality)·이질성(heterogeneity) 같은 전통 가정이 언제/왜 깨지는지 규명하는 것이다. 논문은 이를 alignment, preference representation, market dynamics 관점의 질문으로 구체화하고, 기존에 분절된 LLM 의사결정·선호 추정·행동 시뮬레이션 문헌을 공통의 경제 프레임으로 재배치해 연구 방향을 정리한다.

- **Empirical Impact**: 본 논문은 실증 검증을 제공하기보다, LLM 소비자 행동의 범위와 구조를 개념적으로 정리하고 미해결 연구과제를 제시하는 데 의의가 있다. 따라서 에이전트가 시장에서 소비를 좌우할 때의 정렬(alignment) 문제와 선호 표현의 불일치, 수요 동역학을 후속 연구가 실험·이론으로 연결하는 기준점이 될 전망이다.



### STAR: SpatioTemporal Adaptive Reward Allocation for Text-to-Image RL Post-Training (https://arxiv.org/abs/2606.17979)
- **Prior Approaches**: 기존 text-to-image RL post-training은 최종 이미지의 스칼라 reward(또는 advantage)를 전체 생성 궤적에 동일한 세기로 배분하는 경우가 많았다. 이런 방식은 diffusion/flow의 시간적(denoising step)·공간적(latent token) 구조를 무시해, 프롬프트 정렬에 실제로 중요한 주제 영역의 credit assignment가 약해지는 한계를 보였다. Flow-GRPO 등도 group-relative reward를 두지만, 결국 시공간적으로 동일 강도의 업데이트가 발생한다는 점에서 같은 문제가 남는다.

- **Core Contribution**: 이 논문은 SpatioTemporal Adaptive Reward (STAR) Allocation로, 이미지-level reward를 denoising time×latent space에 맞게 재분배하는 프레임워크를 제안한다. STAR는 모델 내부의 text-image attention을 활용해 프롬프트에서 사용자가 신경 쓰는 핵심 콘텐츠가 나타나는 위치를 단계별로 “어디에” 더 강하게 업데이트할지 할당한다. 외부 reward 소스(예: GenEval/OCR/PickScore)는 그대로 두면서, 더 관련 있는 latent 영역에 같은 advantage를 효과적으로 라우팅해 정책 업데이트를 정밀화한다.

- **Technical Challenges**: 핵심 기술 난제는 “스칼라 결과”를 “시공간 구조”에 맞춰 쪼개는 credit assignment를 안정적으로 만드는 것이다. STAR는 텍스트를 의미 단위(text units)로 분해하고, 선택한 text-image attention 레이어들에서 timestep별 공간 라우팅 맵/heatmap을 만든 뒤 이를 정규화 계수로 변환해 spatial advantage map을 구성한다. 이후 공간별 likelihood-ratio를 비율 기반으로 정규화하고, PPO-style clipped objective와 KL regularization으로 reward hacking 및 업데이트 폭주 위험을 줄이면서 학습한다.

- **Empirical Impact**: Stable Diffusion 3.5 Medium을 기반으로 GenEval, OCR text rendering, PickScore에서 STAR가 Flow-GRPO 대비 일관된 개선을 보였다. 보고된 성능은 GenEval 0.9759, OCR 0.9757, PickScore 23.60으로, compositional semantic alignment와 텍스트 렌더링 정확도, 선호도 최적화가 동시에 향상됐다. 또한 추가 학습 비용이 미미하게(훈련 시간 <2%, peak memory ~1.56% 증가) 나타나, 시공간 정밀 credit assignment를 비교적 가볍게 적용할 수 있는 접근으로 의미가 있다.



### MoCo-AIS: A Contrastive Learning Framework for Similarity Computation of Vessel Trajectories (https://arxiv.org/abs/2606.17978)
Comments:
          Under review at SIGSPATIAL'26

- **Prior Approaches**: 기존 궤적 유사도는 Hausdorff 거리나 DTW처럼 좌표 시퀀스를 정렬·거리 계산해 유사도를 측정하는 방식이 주류였지만, 계산 비용이 크고 잡음·샘플링 밀도에 민감하다는 한계가 컸습니다. 이를 보완하려는 supervised 학습은 DTW 등 전통 지표로 라벨을 만들다 보니 라벨링 비용과 특정 거리 척도에 대한 편향이 그대로 이어졌습니다. self-supervised의 contrastive 방식은 라벨 의존을 줄이지만, 모델·평가 프로토콜이 제각각이라 일관된 비교 기준이 부족했습니다.

- **Core Contribution**: 이 논문은 선박 AIS 궤적을 임베딩 공간으로 매핑해, 학습된 표현 간 거리가 spatio-temporal 유사도를 반영하도록 하는 MoCo-AIS를 제안합니다. MoCo의 Momentum Contrast 패러다임을 AIS에 맞게 통합 프레임워크로 적용하되, 양성/음성 궤적 쌍을 구성하는 대비학습을 통해 유사도 학습을 수행합니다. 또한 인코더를 LSTM/GRU, TCN, Transformer 등으로 교체 가능하게 “플러그인” 형태로 설계해, 동일한 학습 틀에서 다양한 딥러닝 아키텍처를 비교할 수 있는 벤치마크 기반을 제공합니다.

- **Technical Challenges**: AIS 궤적은 불규칙 샘플링·기상/항로 규정 영향·전송 오류 등으로 좌표 잡음과 길이 가변성이 커서, 단순 좌표 기반 라벨링이나 일률적 정렬은 불안정해지기 쉽습니다. MoCo-AIS는 (1) 양성 쌍 생성을 위한 sub-trajectory, shape distortion, Ramer-Douglas-Peucker 기반 simplification 같은 궤적 증강과 (2) 패딩 마스크를 포함한 masked mean pooling으로 가변 길이를 안정적으로 처리합니다. 더해 momentum 업데이트와 FIFO negative queue를 결합해 큰 batch 없이도 다수의 negative를 확보하면서 학습 안정성과 효율을 동시에 노립니다(InfoNCE 손실 사용).

- **Empirical Impact**: 대규모 실데이터 AIS를 지역 3곳(캐나다/미국 동부/미국 서부)에서 수집·전처리(잡음 제거, 구간 분할, 보간, 길이 필터링)해 평가했으며, 유사도 검색 기반의 표준화된 프로토콜로 비교를 수행합니다. 실험 결과 MoCo-AIS는 Hausdorff·DTW 같은 거리 기반 기준선은 물론 t2vec·TrajCL 같은 self-supervised baseline 대비 유사도 학습 성능을 유의미하게 개선합니다. 또한 페어wise 유사도 행렬과 계산 효율을 함께 분석해, 전통적 거리 계산이 병목이 되는 상황에서 더 스케일 가능한 궤적 비교 수단을 제공한다는 점에서 의미가 큽니다.



### Small Initialization Matters for Large Language Models (https://arxiv.org/abs/2606.17945)
Comments:
          26 pages, 8 figures

- **Prior Approaches**: 기존 연구는 LLM 성능이 주로 scale(모델 크기·데이터 양), 데이터 구성, 최적화, 아키텍처 변경에서 온다고 봐왔고 초기화는 대개 구현 디테일로 취급되어 왔습니다. 다만 small initialization이 추론·일반화에 유리하다는 관찰이 있었지만, 실제 LLM 사전학습과 스케일 증가에 대해서는 체계적 검증이 부족했습니다. 또한 초기화 효과가 아키텍처 요소(정규화, attention sink 등)에 의해 가려질 수 있다는 관점이 충분히 정리되지 않았습니다.

- **Core Contribution**: 이 논문은 파라미터 초기화 스케일이 LLM 학습 과정과 모델 capacity에 실질적으로 큰 영향을 준다는 점을 보여줍니다. 특히 초기화 스케일을 줄이면 사전학습 손실이 일관되게 낮아지며, 그 이득은 reasoning을 요구하는 작업에서 가장 크게 나타납니다. 단, 표준 LLM 구성에서는 작은 초기화의 장점이 일부 설정에서 제한되며, 이를 풀어주면 유리한 스케일링이 다시 회복됩니다.

- **Technical Challenges**: 핵심 난관은 작은 초기화의 효과가 레이어 정규화의 epsilon(ε) 포화와 attention sink로 인해 약화/마스킹된다는 점입니다. 저자들은 RMSNorm의 ε를 더 작은 값(예: 1e-12)으로 낮추고 gated attention을 도입해 attention sink를 줄이면 작은 초기화의 이득이 크게 되살아남을 보였습니다. 또한 residual 경로와 identity/embedding 스트림의 균형 때문에 초기화를 무한히 작게 만들면(너무 큰 γ) 학습력이 떨어져, γ=1(초기화 스케일 균형점)이 최적임을 이론+실험으로 확인합니다.

- **Empirical Impact**: Dense와 MoE 모델 모두에서 γ=1이 가장 좋은 성능을 보였고, 아키텍처 보정(ε 감소+gated attention)을 함께 적용하면 스케일이 커져도 작은 초기화의 이득이 유지되는 패턴이 관찰됐습니다. downstream에서도 TriviaQA, HellaSwag, GSM8K, MATH500 등 다수 벤치마크에서 4% 이상 절대 개선이 보고되어 초기화가 추론 능력에 직접 연결됨을 시사합니다. 결론적으로 γ-initialization을 명시적 하이퍼파라미터로 다루고 기본값으로 γ=1을 권하는, 거의 비용이 들지 않는 설계 제안이 제시됩니다.



### How Inference Compute Shapes Frontier LLM Evaluation (https://arxiv.org/abs/2606.17930)
Comments:
          34 pages, 4 figures

- **Prior Approaches**: 기존 벤치마크는 주로 단일(고정) 토큰 예산과 1회 제출 같은 제한된 프로토콜로 성능을 측정해, 추론 시 허용되는 compute(“inference compute”) 차이를 제대로 반영하지 못한다는 비판이 이어져 왔다. 그 결과 낮은 점수가 모델의 근본 역량 부족이 아니라 예산 부족·반복 기회 제한 같은 평가 셋업의 영향일 수 있다.

- **Core Contribution**: 이 논문은 2025~2026 사이의 frontier LLM을 여러 세대에 걸쳐 최대 12개까지, 소프트웨어·수학·의학·사이버 보안 7개 벤치마크에서 “inference scaling”을 통제된 한 프레임워크로 재현한다. 핵심은 토큰 예산 확대, context compaction, 반복 재제출(외부 피드백/자기 판단)을 결합한 단순 개입으로, 벤치마크 점수가 프로토콜에 얼마나 의존하는지 정량화한 것이다.

- **Technical Challenges**: 대규모 토큰 예산에서 성능이 오를 때 그 원인이 탐색(반복/폭)인지, 깊이(serial)인지, 혹은 문맥·중복 제출 같은 운영 요소인지 분해하기가 어렵다. 논문은 total token budget(최대 5M~30M), context compaction(긴 문맥 요약), 999회 제한·중복 방지 가드로 단순한 스케일링 기회를 제공하고, 피드백 유무에 따른 조기 종료 규칙까지 통일해 비교 가능성을 높였다.

- **Empirical Impact**: 결과적으로 더 큰 token budget은 도메인 전반에서 유의미한 성능 향상을 만들며, 특히 cybersecurity, FrontierMath, Humanity’s Last Exam, TerminalBench에서 격차가 크게 나타났다. 또한 고정 예산 평가는 신형 모델의 “큰 예산에서만 보이는 더 어려운 작업 도달/신뢰성”을 과소평가할 수 있고, 반복 재제출은 전반적으로 도움이 되지만 외부 피드백·병렬적(whithin-trajectory가 아닌 breadth) 확장은 벤치마크마다 효과가 달라 “점수는 프로토콜 의존적”임을 보여준다.



### PreAct: Computer-Using Agents that Get Faster on Repeated Tasks (https://arxiv.org/abs/2606.17929)
- **Prior Approaches**: 컴퓨터-사용 에이전트(CUA)는 화면을 보고 reason-act 루프대로 클릭·타이핑을 반복하며, 같은 작업을 다시 시키면 매번 처음부터 다시 추론하고 실행해 비용이 크게 든다. 기존 재사용은 SAGE 같은 스킬 라이브러리나 memory/retrieval 방식처럼 실행 시점에도 LLM 루프가 남거나, 기록-재생(trace)은 매 스텝이 실제로 성공했는지 확인하지 못해 변경된 UI에서 신뢰성이 약하다. 코드/컴파일 기반 접근도 좁은 영역에 치우치거나, 저장된 절차를 신뢰하기 전에 검증·갱신이 충분히 일어나지 않는 한계가 있었다.

- **Core Contribution**: PreAct는 에이전트가 ‘성공한 실행 자체’를 작은 state-machine 프로그램으로 컴파일해 저장하고, 다음 실행에서는 LLM 호출 없이 이를 직접 replay한다. 재생 중에는 각 state 전환 전에 화면이 기대와 일치하는지 체크한 뒤 act하므로, 기존 기록-매크로처럼 맹목적으로 따라하지 않는다. 또한 저장 시점에도 clean state에서 독립 평가자 검증을 통과한 프로그램만 코퍼스에 넣어, 잘못된 프로그램이 누적되며 성능이 하락하는 문제를 막는다.

- **Technical Challenges**: 핵심 기술 과제는 “재생이 정말로 맞게 진행되는가”와 “잘 동작하는 프로그램만 저장되는가”를 동시에 만족시키는 것이다. PreAct는 observe-first-then-act로 state predicate(접근성 트리 기반 화면 조건)를 매 스텝 확인하고 실패하면 즉시 CUA로 폴백한다; 더 나아가 verify-before-store 게이트로 새로 컴파일된 프로그램을 처음부터 다시 돌린 뒤(깨끗한 초기 상태) 벤치마크 평가를 통과해야만 저장한다. 부가적으로, 저장된 프로그램이 없을 때는 새로 탐색해(afresh) 강한 record-and-replay baseline 수준 성능을 유지하도록 설계했다.

- **Empirical Impact**: AndroidWorld, OSWorld, WebArena의 3개 플랫폼에서 warm replay가 8.5~13배 빨라졌고, per-step language-model calls가 없었다. 체크를 켜고 게이트를 적용하면 반복 실행이 누적될수록 품질이 개선되지만, 체크를 끄면 결함 프로그램이 쌓이며 성능이 악화되는 패턴이 관찰됐다. cold→warm 전환에서 작업 수 기준 1.75~2.6 tasks의 추가 이득이 일관되게 나타났으며, 프롬프트 문구·런타임 가드레일·selector 방식(LLM vs embedding)이 결과를 크게 바꾸지 못했다.



### DiagFlowBench: Evaluating How Language Models Handle Off-Procedure Inputs in Grounded Diagnostic Dialogu (https://arxiv.org/abs/2606.17904)
- **Prior Approaches**: 기존 유지보수 어드바이저는 절차 문서(플로우차트/결정트리)에 LLM 출력을 맞춰 hallucinaton을 막는 ‘grounding’ 방식이 주류였다. 하지만 실제 현장 질의는 절차 표현을 벗어나 대화 중간에 scope 밖 발화가 섞이기 쉬운데, 기존 벤치마크는 보통 이를 별도(단일 턴 abstention)로만 다루거나, grounding이 항상 매핑 가능하다는 가정 하에서 평가해왔다. 결과적으로 ‘중간에 벗어났을 때 모델이 어떻게 안전하게 실패/거절하는지’가 상대적으로 덜 측정됐다.

- **Core Contribution**: 이 논문은 DiagFlowBench를 제안하며, 산업용 진단 플로우차트 50개를 1,676개의 multi-turn 대화로 변환해 on-procedure와 off-procedure를 한 대화 안에서 함께 시험한다. 특히 forced mapping이라는 실패 모드를 정의해, 모델이 절차에 존재하는 ‘진짜 노드’를 선택하지만 현재 맥락의 entailment를 만족하지 못하는 경우를 정밀하게 드러낸다. 기존 grounding 점검이 구조적으로는 통과해버리는 이 취약점을 정면으로 겨냥한다.

- **Technical Challenges**: 핵심 과제는 off-procedure일 때 모델이 abstain(거절)해야 하는데, 많은 grounding 시스템은 ‘절차 노드’를 반환하도록 유도되며 맥락 적합성까지 자동으로 검증하지 못한다는 점이다. 저자들은 절차를 그래프로 두고 각 턴의 정답을 “다음 엣지가 entailed되는지”로 엄격히 판정하며, off-procedure 응답은 CA(정확한 거절), FM(강제 매핑), FA(사실적 fabricaton)로 분류한다. 또한 어휘적 유사성이 높은데도 entailment가 불충분한 상황이 FM을 유발함을 범주화된 injection 설계로 확인한다.

- **Empirical Impact**: 10개 상용 및 오픈웨이트 모델을 평가한 결과, on-procedure에서는 step accuracy가 대체로 높지만(70.1~85.0%) termination recognition과 off-procedure 신뢰성은 크게 흔들린다. hallucinaton 지표로 알려진 FA는 낮은 편(2.2~8.6%)이지만, FM이 훨씬 지배적이며(15.7~67.4%) 특히 lexical overlap가 클수록 FM이 증가하는 경향이 나타난다. 더 나아가 한 번 FM이 발생하면 이후 on-procedure 추적이 거의 붕괴하는 recovery rate(1.0~9.1%)까지 관측되어, grounding이 ‘안전장치’로 기대되던 것이 오히려 맥락 오류를 가릴 수 있음을 시사한다.



### Learn to Quantify Social Interaction with Constraints for Pedestrian Walking (https://arxiv.org/abs/2606.17897)
- **Prior Approaches**: 기존 군중 장기 궤적 예측 연구는 Social LSTM, spatio-temporal graph, Transformer 등으로 사회적 상호작용을 반영했지만, 실제로는 충돌 회피·행렬 동행 같은 현상을 사후적으로 해석하는 경우가 많았습니다. 또한 많은 모델이 서로 다른 상호작용을 구분 없이 풀링해 “무슨 종류의 상호작용이 인코딩됐는지”를 설명하기 어렵다는 한계가 남아 있습니다. 반면 GAN/VAE/goal-based, MDN 계열은 다중 모달리티를 다루지만, 사회적 상호작용의 의미 있는 모드 자체를 라벨 없이 학습해 해석 가능하게 만드는 데는 집중이 약했습니다.

- **Core Contribution**: 이 논문은 보행자 사이 상호작용을 라벨 없이 “모드(클러스터)”로 학습하고, 그 모드를 예측 모델 내부에 자연스럽게 통합하는 Learn to Cluster를 제안합니다. 확률적 잠재변수 생성모형으로 edge(에이전트-이웃 관계)마다 이산 latent variable을 두어, 궤적 관측에서 직접 상호작용 유형을 분류(클러스터링)하도록 만듭니다. 또한 학습된 latent 변수는 예측 네트워크에서 다중 미래를 생성하는 mixture components의 조건 신호로 활용되어, 불확실성을 다루면서도 사회적 “스타일”을 해석 가능한 형태로 제공합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 상호작용을 정확히 라벨링하기 어렵고, (2) 이산 샘플링이 backpropagation을 끊는 문제이며, (3) 클러스터 간/내 표현이 분리되도록 학습을 안정화해야 한다는 점입니다. 논문은 Gumbel-Max와 softmax 근사로 이산 latent variable을 미분가능하게 다루고, Maximum Coding Rate Reduction 계열의 사회 loss로 동일 클러스터는 압축하고 서로 다른 클러스터는 대비되게 강제합니다. 그 결과, latent 모드가 “공격적/온건/주의 없음”처럼 시간에 따라 변화하는 사회적 행동의 스타일을 확률적으로 포착하도록 설계됐습니다.

- **Empirical Impact**: UCY와 ETH 벤치마크에서 leave-one-out로 평가하며, ADE/FDE 기준 성능 향상과 함께 모드가 의미 있게 분포한다는 해석 결과를 제시합니다. 시나리오 A~D(근거리 마주침, 평행 보행, 원거리 마주침, 역방향 후면 등) 통계 분석에서 특정 모드가 특정 상황에 우세하게 나타나 “모드의 의미”를 뒷받침합니다. 또한 모드 전환(예: Mode 2→Mode 3, Mode 3→Mode 1) 분석을 통해 상대적 위치·속도 관계가 변화할 때 클러스터가 이동함을 보여, 향후 explainable social interaction 기반 예측 연구에 실증적 근거를 제공합니다.



### MathVis-Fine: Aligning Visual Supervision with Necessity via Progressive Dependency-Guided Training for Multimodal Mathematical Reasoning (https://arxiv.org/abs/2606.17888)
- **Prior Approaches**: 기존 Multimodal Large Language Model(MLLM) 기반 수학 추론은 그림을 균질하게 처리하거나 보조 신호로만 취급하는 경우가 많았다. 그 결과 모든 샘플에 동일한 시각적 보상/손실을 적용해, 그림이 필수인 문제와 텍스트만으로도 충분한 문제를 구분하지 못했다. 또한 이미지 기반 감독이 거칠고 reward가 균일하게 주어져 시각 근거와 정답 사이의 인과 관계가 흐려지는 한계가 지적된다.

- **Core Contribution**: 이 논문은 수학적 추론에서 샘플마다 “시각 의존도(visual dependency)”가 다르다는 점을 명시적으로 모델링하는 MathVis-Fine 프레임워크를 제안한다. 약 5.4K 규모의 MathVis-Fine 데이터셋을 만들고, 각 문제에 fine-grained visual dependency rating과 텍스트-시각 단계 정렬을 부여한다. 이후 SFT와 강화학습(RL)에서 의존도에 따라 시각적 강화(grounding) 강도를 동적으로 조절한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 그림이 얼마나 필요한지에 맞춘 정밀 감독 신호를 설계하고 (2) RL 단계에서 시각 reward 편향을 줄이는 것이다. 이를 위해 high-dependency 샘플에는 Retrieval-Perception Synergy Loss로 “검색한 시각 근거를 실제 추론에 의존하게” 만드는 마스킹 일관성 제약을 붙인다. RL(GRPO)에서는 Visual Indexing Reward와 Visual Content Reward를 분리하고, 의존도 점수 λv로 reward fusion을 게이팅해 text-dominant 샘플에는 시각 잡음을 줄인다.

- **Empirical Impact**: 실험에서 MathVis-Fine은 여러 멀티모달 수학 벤치마크에서 기존 방법을 능가하며, 특히 geometry/GPS처럼 시각 해석이 중요한 범주에서 일관된 개선을 보인다. 예를 들어 open-source 7B 계열에서 MathVista 정확도 77.26%를 달성해 MINT-CoT 대비 향상을 보고한다. 또한 HC-M3D에서 attribute hallucination(AG) 오율을 낮추며 시각 환각을 억제하는 효과를 정량적으로 입증했다.



### Structural Preservation and the Logical Expressiveness of Graph Neural Networks (https://arxiv.org/abs/2606.17882)
Comments:
          20 pages

- **Prior Approaches**: 기존 연구는 GNN의 aggregation, combination, activation 같은 구조 선택을 고정해 논리 형식과의 대응(동치/번역)을 특정한 GNN 부분집합에 한정해 왔습니다. 특히 first-order logic 확장(카운팅 term, Presburger quantifier)이나 Datalog 같은 규칙 기반 형식과의 “아키텍처-주도” 특성화가 대표적입니다. 다만 이런 방식은 원하는 의미적 성질(구조 보존)을 설계 제약으로 간접 달성하기 때문에, 임베딩·호모모피즘 같은 의미적 관점의 명시적 분류가 부족했습니다.

- **Core Contribution**: 이 논문은 GNN 분류기의 표현력을 “구조 보존(semantic preservation)” 관점에서 재정의합니다. 구체적으로 임베딩(embedding), 단사 호모모피즘(injective homomorphism), 호모모피즘(homomorphism) 하에서 보존되는 분류기 클래스가 각각 graded modal logic의 세 가지 조각(fragment)과 같은 표현력을 갖는 것을 보입니다. 또한 이러한 의미적 분류가 특정 아키텍처 선택과 무관하게 성립하면서, 동일 표현력을 갖는 GNN 아키텍처도 각각 구성 가능함을 함께 보입니다.

- **Technical Challenges**: 핵심 기술 난제는 이 보존 성질을 만족하는 GNN을 논리 조각으로 “유한 표현”할 수 있어야 한다는 점입니다. 이를 위해 트리 기반 분석을 도입하며, GNN의 LL-층 로컬리티를 LL-unravelling(그래프의 트리 펼침)로 환원해 트리에서의 보존성과 논리성을 연결합니다. 여기에 높이가 bounded인 트리들의 well-quasi-order(wqo) 성질(임베딩 관계로 최소 원소가 유한함)을 새로 증명해, 각 클래스가 최소 트리들의 유한한 논리식 disjunction으로 표현될 수 있게 만들었습니다.

- **Empirical Impact**: 결과적으로 “의미적 보존 성질 ↔ graded modal logic fragment ↔ GNN 아키텍처”의 3자 대응이 정리돼, 모델 구조를 넘어서는 표현력 분류 체계를 제공합니다. 또한 해당 보존 클래스가 단조(monotonic) GNN 계열(예: injective homomorphism 보존은 monotonic GNN, 호모모피즘 보존은 MAX 기반 단조 GNN 등)과 정확히 같은 표현력을 갖는다는 아키텍처적 캡처도 제시합니다. 나아가 positive-weight GNN(가중치 비음수 제약)은 이 클래스들에서는 표현력 측면에서 제한이 아니며 동등한 형태로 변환 가능하다는 관찰이 나와, 설계 제약의 실질적 의미를 명확히 합니다.



### StepGuard: Guarding Web Navigation via Single-Step Calibration (https://arxiv.org/abs/2606.17871)
- **Prior Approaches**: 기존 web navigation 연구는 VLM(시각-언어 모델) 기반 instruction following과 RL로 순차 의사결정을 강화해 왔지만, 단일 단계에서의 실수에 취약하다는 한계가 남아 있다. 특히 탐색을 늘리려는 navigation 보상과 빠른 종료/정답 생성을 요구하는 question-answering 보상이 동시에 최적화되면서 reward가 얽혀(conflict) 업데이트가 흔들리기 쉽다. 또한 한 스텝의 잘못된 action이 이후 trajectory 전체로 전파되며 error propagation이 커지는 문제가 자주 보고된다.

- **Core Contribution**: 이 논문은 StepGuard라는 프레임워크로 single-step fragility를 줄이는 데 초점을 둔다. 핵심은 Dynamic Dual-Policy Optimization(DDPO)로 navigation-first 모드(탐색)와 answer-first 모드(종료/질문응답)를 분리해 reward conflict를 완화하는 것이다. 여기에 Confidence-Guided Adaptive Navigation Reflection(CANR)을 더해, 단계별 confidence가 낮을 때만 reflection을 유도하고 contrastive reward로 자기수정이 학습되게 만든다.

- **Technical Challenges**: 문제는 두 가지인데, (1) 서로 다른 목적함수를 한 정책에 함께 섞으면 gradient 간섭이 생기고, (2) reflection을 무작정 자주 쓰면 계산비용과 추론 지연이 커진다는 점이다. DDPO는 학습 샘플을 navigation subset/QA subset으로 나눠 모드별 비충돌 목적만 번갈아 최적화하며, inference 때는 oracle 라벨 없이도 학습된 정책으로만 동작한다. CANR은 KL 기반으로 per-step navigation confidence(엔트로피에 준하는 상대 신호)를 추정하고, reflection trigger 확률을 confidence에 따라 조절한 뒤 성공적인 자기수정에 보상을 주는 방식으로 불필요한 reflection을 줄인다.

- **Empirical Impact**: 실험에서는 WebVLN과 WebWalkerQA에서 SR과 step-wise action accuracy가 모두 개선되며, 특히 WebVLN에서 최신 성능을 경신한다(예: 39.83% SR). WebWalkerQA에서도 3B 백본임에도 Easy/Medium/Hard에서 각각 높은 SR을 보였고, 더 큰 모델 대비 우수하거나 준-동급 성능에 근접해 작은 모델의 격차를 single-step calibration로 메울 수 있음을 시사한다. CANR의 adaptive trigger가 random/always-on 대비 더 높은 성공률과 더 짧은 Trajectory Length/런타임을 함께 달성했으며, confidence가 실제 정오답을 더 잘 구분하도록 calibration되는 분석도 제시된다.



### FlowRAG: Synergizing Explicit Reasoning via Frequency-Aware Multi-Granularity Graph Flow (https://arxiv.org/abs/2606.17856)
- **Prior Approaches**: 기존 RAG는 주로 DPR 같은 dense retrieval로 문서를 고른 뒤 LLM이 답을 생성하지만, 문서 간 의존성이 필요한 multi-hop에서는 정보가 쪼개져 성능이 흔들릴 수 있다. GraphRAG는 그래프 위상으로 entity 의존성을 잡으려 하지만, entity 수준에서 쿼리 신호가 희박하거나(abstract/semantically sparse) 노이즈가 multi-hop 경로를 타고 전파되면 추론 체인이 쉽게 깨진다. 한편 LinearRAG처럼 relation-free 그래프도 많지만, topological scoring에 의존해 노이즈를 정교하게 걸러내는 데 한계가 있다.

- **Core Contribution**: FlowRAG은 quad-level heterogeneous graph( passages–summaries–sentences–entities )를 만들고, summary 노드를 coarse semantic hub로 써서 entity 희소성을 완화한다. 또한 dual-granularity activation으로 sentence-level matching(미시)과 summary–query alignment(거시) 신호를 동시에 초기화해, 용어가 달라지거나 추상적인 질의에서도 관련 entity를 더 잘 활성화한다. 마지막으로 frequency-aware weighted flow로 엔티티 간 관계를 암묵적 점수 대신 명시적 reasoning path로 뽑아 생성의 논리 뼈대를 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) abstract 쿼리에서 entity 단서가 부족해 엔트리 포인트를 놓치고, (2) multi-hop 동안 noisy activation이 entity-to-entity 전이를 망가뜨려 관계 사슬이 붕괴하는 문제다. FlowRAG는 요약 허브+dual-granularity activation으로 granularity mismatch를 줄이고, term frequency 기반 가중치로 entity–passage 연결의 신뢰도를 계산해 잡음 연결을 가지치기한다. 이어서 에너지 감쇠(간선 전파의 decay)와 pruning threshold로 긴 경로의 드리프트를 억제하면서, 고신뢰 경로만 논리 skeleton로 생성 단계에 제공한다.

- **Empirical Impact**: HotpotQA, 2WikiMultiHopQA, MuSiQue, Medical(총 4개 벤치마크)에서 FlowRAG는 기존 strong baseline들을 넘어서는 state-of-the-art 성능을 보였고, 특히 LinearRAG 대비 GPT-ACC가 HotpotQA에서 +2.1%, 2WikiMultiHopQA에서 +2.5% 개선됐다. MuSiQue에서는 contain은 비슷하되 GPT-ACC가 더 크게 오르며, 이는 recall 위주의 잡음보다 weighted flow가 더 정제된 정보로 생성 정확도를 끌어올린다는 해석을 뒷받침한다. 또한 ablation에서 dual-granularity activation과 frequency-aware weighted flow가 모두 성능을 유의미하게 끌어올려, 복잡 multi-hop에서의 노이즈 강건성과 명시적 추론 흐름의 가치를 실증했다.



### A homotopy-type-theoretic generalization of neurosymbolic inferenc (https://arxiv.org/abs/2606.17851)
- **Prior Approaches**: 기존 neurosymbolic(NeSy) 연구는 신경망이 σ-structure에 대한 belief를 주고, 논리 이론이 허용되는 구조를 골라 가중합/집계를 수행하는 단일 레시피로 볼 수 있다. 이 과정은 weighted model counting(WMC)에서 출발해 fuzzy/provenance 등으로 확장되지만, underlying set이 decategorification되어 “대칭으로 같은 σ-structure”와 “여러 증명의 정체성(몇 가지 증명이 있는지)”을 잃는다.

- **Core Contribution**: 이 논문은 set 대신 homotopy type theory(HoTT)의 type을 기반으로 함수를 한 단계 lift하여 belief-weighted homotopy cardinality를 정의한다. 그 결과, σ-structure의 대칭(automorphism)과 증명의 정체성을 반영하는 크기 개념이 등장하며, 이 값은 symmetries가 사소할 때 기존의 classical functional로 정확히 보존된다(보수성 정리).

- **Technical Challenges**: 핵심은 NeSy에서 쓰는 네 구성요소(구조 공간, 논리값, 증명/위트니스, 집계)를 type과 truncation level(0-type vs groupoid 등)로 정확히 대응시키고, 대칭이 있는 상황에서 “올바른 크기”를 정의해 계산 가능하게 만드는 것이다. 저자들은 (1) proof family로 증명을 타입으로 승격하고, (2) belief-weighted homotopy cardinality를 dependent sum 위의 대칭-보정된 집계로 formalize하며, (3) Lean 4/Mathlib로 보수성 및 계산 법칙을 기계검증한다.

- **Empirical Impact**: 대칭이 reasoning shortcut의 원인이라는 관점을 orbit(대칭군의 작용 궤도) 해석으로 연결해, shortcut-aware concept posterior를 orbit-averaging이라는 단일 모델 래퍼로 closed form 계산한다. MNIST reasoning-shortcut 벤치마크에서 이 단일 모델은 Bears ensemble(다양성 훈련 5모델)에 비해 calibration이 더 좋았고, label accuracy와 identifiable concepts는 그대로 유지되었다. 결과적으로 ensemble/밀도추정에 의존하던 “정답 posterior”를 대칭 불변성 관점에서 더 경제적으로 얻는 경로를 제시한다.



### WallZero: Mastering the Game of WallGo with Strategic Analysis (https://arxiv.org/abs/2606.17847)
Comments:
          Accepted by the Computers and Games conference (CG 2026)

- **Prior Approaches**: WallGo는 7x7 보드의 소규모 게임이지만, 매 턴마다 돌 이동과 벽(벽) 배치가 함께 일어나 게임트리 복잡도가 매우 커 AI 연구가 상대적으로 부족했다. 기존 온라인 에이전트들은 높은 수준의 플레이에 도달하지 못했고, 사람이 체감하는 전략/균형이 체계적으로 분석되지는 않았다. 또한 AlphaZero 계열에서 성능을 좌우하는 상태 표현(state representation) 설계의 중요성은 알려져 있었지만 WallGo에 맞춘 맞춤 설계는 미흡했다.

- **Core Contribution**: 이 논문은 2인 WallGo를 위한 AlphaZero 기반 에이전트 WallZero를 제안하며, 액션 설계와 feature 설계를 WallGo 규칙에 맞게 구체화한다. 특히 도달 가능성(reachability), 영토(territory) 등 게임의 핵심 의사결정을 모델 입력에 직접 반영해 학습 효율과 플레이력을 동시에 끌어올린다. 그 결과 WallZero는 연구에 참여한 프로 바둑기사 2명에게서 평균적으로 1.98x 더 많은 영토를 확보하며 압도적인 경기력을 보였다.

- **Technical Challenges**: WallGo는 (1) 턴마다 이동+벽 배치가 결합된 상호작용적 의사결정, (2) 상태별로 합법 행동이 부분집합이 되는 큰 열거형 action space라는 문제가 있다. 논문은 통합 정책을 위해 액션 공간을 단계적 구성요소(돌 선택-도착 위치-벽 방향)로 정의하되, action mask로 불법 행동을 제거해 학습을 안정화했다. 또한 feature plane을 49장 구성으로 확장하면서, 영토 소유(레드/블루/중립), 플레이 가능한 도달 위치, 최근 4스텝의 돌·벽·영토 변화 등 게임 특화 신호를 설계해 성능 향상을 달성했다.

- **Empirical Impact**: 실험에서 reachability feature 포함이 성능을 가장 크게 끌어올렸고, 최종 WallZero는 empty mode에서 82.87%, 4-stone mode에서 82.02%의 승률을 기록했다. 프로 바둑기사와의 공식 매치에서도 WallZero는 8판 전승을 거두었으며, 계측된 영토 격차(기하평균)로는 평균 1.98x, 최대 3.08x까지 나타났다. 게임 균형 분석에서는 Netflix ‘The Devil’s Plan’의 4-stone 오프닝이 빈 보드(empty mode)보다 더 균형 잡힌 출발을 만든다는 관찰과 함께, reachability control과 암묵적 passing(턴 순서 유도) 같은 핵심 전략까지 도출해 의미 있는 인사이트를 제공했다.



### DecoSearch: Complexity-Aware Routing and Plan-Level Repair for Text-to-SQL (https://arxiv.org/abs/2606.17821)
- **Prior Approaches**: 기존 Text-to-SQL은 LLM의 zero-shot/few-shot 생성으로 성능을 크게 끌어올렸지만, 중첩 질의·다단계 추론·다수 JOIN처럼 복잡한 쿼리에서는 단일 패스 생성이 쉽게 무너진다. 이를 보완하려는 분해(decomposition), 실행 피드백 기반 self-refinement, 다중 후보 생성·선택, 난이도 라우팅 등이 나왔지만 대개 한 수준(구현/전략/계획)만 다루고, 실패의 원인이 계획 결함인지까지 진단해 다른 수리 경로로 보내는 체계는 부족했다. 특히 SQL 문법·필터 같은 구현 오류는 고칠 수 있어도, DAG 분해가 애초에 틀렸을 때는 구조를 다시 짜지 못한다는 한계가 남는다.

- **Core Contribution**: 이 논문은 DecoSearch라는 training-free 프레임워크를 제안해, 각 질의를 “필요한 추론 깊이”에 맞춰 라우팅하고 실패 원인의 수준에 따라 다른 복구를 수행한다. 먼저 Schema Selector가 스키마 잡음을 줄이기 위해 관련 테이블/컬럼을 선택하고, Judger가 직접 생성이 가능한지(Direct Path) 아니면 Directed Acyclic Graph(DAG) 분해가 필요한지 판단한다. 또한 실행 실패가 누적되면 Topology Refiner가 분해된 추론 계획(DAG)을 재구조화해, SQL 수준 수정으로는 해결되지 않는 계획 결함까지 교정한다.

- **Technical Challenges**: 핵심 난제는 “계산 예산을 아끼면서도 복잡한 쿼리에만 구조적 분해를 쓰는 것”과 “실패를 구현/전략/계획 중 어디에서부터 생겼다고 보고 수정할지”를 자동으로 결정하는 것이다. DecoSearch는 (1) 경량 Schema Selector로 프롬프트 토큰을 절감하고 (2) zero-shot Judger의 needs_decomposition 판정으로 Direct Path와 DAG를 동적으로 선택하며 (3) RAG로 분해 예시를 제공해 DAG 생성의 논리 일관성을 높인다. 마지막으로 Topology Refiner는 노드 실행 실패를 신호로 삼아 DAG를 다시 만들고, 수정 시도 예산을 제한해 복구 불가능한 경우에도 파이프라인을 중단하지 않고 다음 단계로 진행한다.

- **Empirical Impact**: 실험에서 DecoSearch는 BIRD에서 실행 정확도 70.53%, Spider에서 88.31%를 기록하며 training-free 베이스라인을 모두 앞섰다. 동시에 누적 토큰 사용 관점에서도 CHESS 대비 약 10배 수준의 오버헤드를 줄이며, Judger-led Escalation 덕분에 대부분 질의는 Direct Path로 처리되어 고가의 분해를 최소화한다. 또한 ablation에서 Judger 라우팅이 가장 큰 기여를 보였고(분해 강제 시 -10.65%p), schema pruning과 plan-level Topology refinement가 그 다음으로 정확도·효율을 개선했으며, fine-tuning된 SQL 생성 백본에도 모델 변경 없이 플러그인처럼 성능을 추가로 끌어올리는 것으로 확인됐다.



### Shattering the Autoregressive Curse: Dynamic Epistemic Entropy Orchestrated Erasable Reinforcement Learning for LLMs (https://arxiv.org/abs/2606.17735)
- **Prior Approaches**: 기존 강화학습(RL) 기반 LLM 추론은 토큰을 한 방향으로 생성하는 autoregressive 특성 때문에, 초반의 작은 인지적 오차가 Markov 의사결정 흐름을 따라 누적·증폭되는 ‘autoregressive curse’에 취약합니다. 이를 보완하려고 external reward model(PRM) 같은 외부 신호로 단계별 판단을 붙이면 데이터/연산 비용이 커지고, 모델 상태가 바뀌는 환경에서는 reward hacking 및 분포이탈 문제가 생기기 쉽습니다. 또한 전역 resampling은 정답 prefix까지 통째로 폐기해 credit assignment와 메모리·계산 오버헤드가 악화됩니다.

- **Core Contribution**: 이 논문은 동적 epistemic entropy 기반 erasable reinforcement learning(E3RL)을 제안해, 불확실성 높은 추론 구간을 ‘지우고(erase)-재생성’하는 자기-치유형(erasable) 추론 루프를 구성합니다. 핵심은 외부 신호 없이 모델 내부의 endogenous local autoregressive cross-entropy를 epistemic uncertainty 좌표로 물리화해, 문제가 되는 segment만 정밀하게 절제하도록 한 것입니다. 이를 위해 segment-level adaptive dynamic threshold와 advantage allocation으로, 오류가 전체 reasoning trajectory로 전파되지 않게 합니다.

- **Technical Challenges**: 문제는 long-horizon에서 불확실성이 순간 잡음인지 실제 논리 붕괴 징후인지 구분하고, 지우기/재시도 과정이 RL 학습을 불안정하게 만들지 않는 것입니다. E3RL은 segment 단위로 cognitive entropy를 평균/최댓값/변동률로 요약하고 sliding-window smoothing 및 boundary normalization으로 단기 요동을 억제합니다. 이어 group-level GRPO의 dynamic baseline을 문제 모호도에 맞게 조정하고, micro 단계의 exponentially increasing penalty와 causal backtracking reward assignment로 유효한 구간에만 gradient가 흐르도록 non-Markovian erasure operator를 설계합니다.

- **Empirical Impact**: DeepMath-103k로 학습한 뒤 AMC/AIME/MATH/Minerva/OlympiadBench 등 수학 추론 벤치마크에서 Avg@32 및 Pass@k로 성능을 검증했으며, Qwen3-4B와 Qwen3-8B가 각각 이전 SOTA 대비 5.349%, 6.514% 향상했습니다. 학습 동역학 분석에서는 높은 training accuracy를 유지하면서도 더 긴 응답을 만들고(불필요한 segment 확장 억제), 불안정 구간만 삭제하는 패턴이 관찰됐습니다. 또한 ablation에서 uncertainty metric 구성요소와 주파수 인지(frequency-aware) 지우기/causal allocation의 결합이 성능을 좌우함을 확인해, long-horizon 추론의 효율·샘플 효율·안정성 개선이 시스템 설계에서 왔음을 보여줍니다.



### LongWebBench: Evaluating Structural and Functional Webpage Generation in Long-Horizon Settings (https://arxiv.org/abs/2606.17727)
Comments:
          49 pages, 38 figures

- **Prior Approaches**: 기존 vision-language model(VLM) 기반 웹페이지 생성 평가는 주로 단일 스크린·짧은 페이지에서의 국소 시각 유사도 재현에 집중했습니다. 일부는 DOM이나 상호작용을 보지만, long-webpage의 전역 구조 일관성과 브라우저 실행 가능한 multi-step 목표 달성을 함께 진단하기는 어려웠습니다. 결과적으로 그럴듯해 보여도 메뉴/필터/폼 제출 같은 실행 작업이 실패하는 격차가 평가에서 드러나지 않았습니다.

- **Core Contribution**: 논문은 long-horizon webpage generation을 구조(visual fidelity)와 기능(executable interaction) 관점으로 분리해 평가하는 벤치마크 LongWebBench를 제안합니다. W-VFR은 490개의 실세계 long webpage로 전역 구조·레이아웃·섹션 위계·스타일·정보 밀도를 점검하고, W-FFR은 129개 페이지에서 507개의 goal-oriented interaction task로 브라우저 실행 관점의 기능 정합성을 검증합니다. 또한 단순 시각 유사도를 넘어서 ‘실행 가능한 상호작용’ 자체를 핵심 기준으로 삼도록 프로토콜을 설계했습니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 길어진 페이지에서 전역 레이아웃과 스타일 일관성을 유지하는지, (2) 생성된 HTML이 실제로 브라우저에서 단계별 상태 전이를 수행하는지 동시에 확인하는 것입니다. 저자들은 W-VFR에 대해 multi-dimensional VLM 기반 long-range structural coherence 점수(스타일은 DINO 보조)를, W-FFR에는 DOM-augmented agent-based 파이프라인으로 end-to-end 실행 검증을 도입했습니다. 아울러 단일 이미지와 multi-image 입력 설정을 함께 제공해 입력 제약 해소 여부와 long-context 통합 능력을 분리 분석할 수 있게 했습니다.

- **Empirical Impact**: 실험 결과, 페이지 길이가 증가할수록 structural fidelity는 전반적으로 저하됐고, 시각적으로 그럴듯한 생성물이라도 multi-step 상호작업을 완료하지 못하는 경우가 많았습니다. W-VFR에서는 multi-image가 일부 개선을 주지만 ‘전역 통합’ 문제를 완전히 해소하진 못했으며, W-FFR에서는 SSR이 높아도 TSR·PSR이 크게 떨어져 오류가 누적되는 현상이 두드러졌습니다. 자동 평가 프로토콜은 사람 판단과의 높은 합치도(구조 및 실행 단계 수준)를 보여 스케일 가능한 장기 웹 생성 평가의 표준 기반으로 의미가 큽니다.



### EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Inten (https://arxiv.org/abs/2606.17698)
- **Prior Approaches**: 기존 쇼핑 에이전트 벤치마크는 대체로 단일 쿼리에서 의도가 거의 드러나거나, 프로필을 직접 노출해 hidden intent 회수 구간을 약화시켰습니다. 또한 최종 상품만 맞추는 coarse한 채점이 흔해, 긴 호라이즌 동안 “어떤 요구사항을 어디서 놓쳤는지”를 진단하기 어렵습니다. 마지막으로 긴 상호작용 과제를 사람이 만들거나 검증이 느슨하면 잡음이 커져 순위 비교를 신뢰하기 어렵다는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 EComAgentBench를 제안하며, 실제 Amazon 상품·리뷰(Reviews 2023 카탈로그)에 기반한 662개 단일-상품 쇼핑 과제를 제공합니다. 각 과제의 요구사항을 (1) 보이는 쿼리의 암시, (2) get_user_profile로만 얻는 도구-게이티드 persona, (3) ask_user로만 드러나는 스크립트형 확인 질문에 분산해 에이전트가 숨은 의도를 조립하도록 합니다. 모든 실패는 typed, source-tagged rubric로 “요구사항+출처” 단위로 귀속되도록 설계해, 단순 정답 여부를 넘어 누락 지점을 설명합니다.

- **Technical Challenges**: 핵심 난제는 (a) 의도를 숨기되 검증 가능하게 분해하는 것, (b) 긴 호라이즌을 재현 가능하게 자동 구성·자동 채점하는 것, (c) LLM-자유채점이 유발하는 그라더 잡음을 최소화하는 것이었습니다. 저자들은 각 rubric의 id/type/expected value/출처를 코드로 고정한 뒤, 이후 생성 단계에서는 LLM이 자연어로 표현만 하게 해 “정답 기준이 흔들리지 않게” 했습니다. 또한 명시적 누출 방지, 도구 게이팅 준수, 스크립트형 clarification의 결정성, 그리고 실제 카탈로그 기반 교차 검증까지 포함해 신뢰도 높은 평가 파이프라인을 구축했습니다.

- **Empirical Impact**: 7개 모델을 공통 환경에서 평가한 결과, 전체 정확도는 19.5%~57.1%로 큰 격차가 나타나며 벤치마크가 모델을 분리한다는 점이 확인됐습니다. 특히 rubric satisfaction이 쿼리에서 보이는 요구사항보다 persona/clarification처럼 숨겨진 출처로 갈수록 떨어져(예: gpt-5.4의 경우 88.1%→69.8%/70.9%) 숨은 의도 통합의 어려움이 계량됩니다. 또한 평가자는 보조 채점 경로까지 감사(audit)해 채점 잡음을 거의 배제했고, implicit requirement가 늘어날수록 정확도가 크게 하락해(51.8%→27.2%) 긴 호라이즌 궤적 수준 추론이 여전히 open challenge임을 실증적으로 보여줍니다.



### FllumaOne: A Code-Native Multimodal CAD Dataset with Executable Programs and Kernel-Validated Feature Histories (https://arxiv.org/abs/2606.17696)
Comments:
          24 pages, 4 figures

- **Prior Approaches**: 기존 CAD 데이터셋은 최종 형상(메시, voxels, B-Rep) 중심이어서 편집 가능한 구성 이력(스케치-피처-파라미터-의존성)을 그대로 재사용하기 어렵습니다. 명령 시퀀스/스케치 그래프/실행 가능한 스크립트를 포함한 연구가 늘었지만, (1) 커널 유효성 검증, (2) 코드-이력-기하 간 정렬, (3) 상용 CAD 라이선스 의존 같은 실무 제약이 남아 있었습니다.

- **Core Contribution**: FllumaOne은 실행 가능한 Python 프로그램과 구조화된 feature tree(구성 이력)를 한 샘플 안에서 일치시키고, OpenCASCADE 커널에서 생성된 STEP 및 표면 point cloud까지 함께 제공하는 code-native 멀티모달 editable CAD 데이터셋입니다. 각 샘플은 학습용 training-oriented IR, 자연어 설명, 8개 canonical visible-edge 렌더링, 메타데이터(해시/무결성 체크)까지 포함해 “보이는 모델”이 아니라 “편집 가능한 모델”을 재구성하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 (a) 코드는 실행 가능하지만 기하는 깨지거나(부울 실패, 퇴화 면, 비매니폴드 등) (b) 코드-이력-STEP/파생 관측치가 정렬되지 않으면 학습 신호가 무너진다는 점입니다. 논문은 후보를 설계 시그니처 중복 검사로 줄인 뒤 FllumaCLI로 실행하고, OpenCASCADE B-Rep 검증과 STEP export/reload 체크까지 통과한 “커널 수용 결과”만 100K로 채택하며, 해시 기반 무결성 리포트로 누락·불일치·중복을 강하게 차단합니다.

- **Empirical Impact**: FllumaOne-100K(템플릿 복잡도 4단계, 총 10만 샘플)에서 Qwen2.5-Coder-1.5B LoRA 베이스라인은 Python 문법 유효성 99.98%, Flluma 빌드 성공 99.97%, STEP-export 유효성 99.14%를 보였고, point cloud로 변환한 9,909개 예측의 mean normalized Chamfer Distance가 0.002124로 보고됩니다. 이 데이터셋은 text-to-program, feature-tree prediction, B-Rep 분석, retrieval, 설계 완성, editable reverse engineering 같은 “편집 지향” 태스크에 바로 쓰일 수 있는 표준 인프라로 의미가 큽니다.



### Using Cognitive Models to Improve Language Model Simulation of Human Persuasion Games (https://arxiv.org/abs/2606.17657)
- **Prior Approaches**: 기존 LLM 안전평가·훈련에서는 시뮬레이션 인간을 persona prompt처럼 텍스트 특성으로만 구성하거나, preference learning으로 행동 경향을 학습하는 방식이 많았다. 하지만 이런 방법은 시간에 따른 belief 업데이트(인지적 갱신)나 동기 기반(motivated) 편향 같은 ‘인간 의사결정 규칙’을 명시적으로 모델링하도록 설계되지 못한다. 또한 규범적 기준(예: Bayesian) 대비 평가가 많았지만, 실제 사람의 비-Bayesian 다양성을 넓게 커버하긴 어렵다는 한계가 지적된다.

- **Core Contribution**: 논문은 Equation-to-Behavior Prompting(수식-행동 프롬프팅)으로, LLM이 Bayesian updating뿐 아니라 affine distortion, motivated updating, Grether’s α-β 모델 같은 인지·행동 경제학의 수식적 의사결정 규칙을 따르도록 유도한다. 더 나아가 Equation-to-Behavior RL(수식-행동 강화학습)로, 작은 모델도 수식 규칙을 보상함수 형태로 강제해 belief error를 줄이도록 학습한다. 법적 의사결정 데이터(Old Bailey Proceedings) 기반 persuasion game 벤치마크로, 실제 제약을 반영한 인간 설득·판단 시뮬레이션을 평가한다.

- **Technical Challenges**: 핵심 난관은 (1) 복잡한 정보가 축적되는 전략 환경에서 LLM이 ‘규칙 기반’ belief updating을 안정적으로 구현할지, (2) 작은 모델이 프롬프팅만으로는 목표 업데이트 방식을 일관되게 분리해내지 못할 수 있다는 점이다. 저자들은 evidence 순서 조작 등 통제된 조작으로 비-Bayesian 편향(예: primacy bias/동기 기반 유사 패턴)을 측정하고, Equation-to-Behavior RL에서는 인지 모델로부터 정답 posterior를 계산해 trajectory-level 보상으로 KL 정규화된 강화학습을 수행해 규칙 일치를 강화한다. 그 결과 out-of-distribution 파라미터에서도 belief error를 26.5% 줄이는 성능을 보인다.

- **Empirical Impact**: 실험에서 큰 모델은 Equation-to-Behavior Prompting만으로도 수식 명세에 따른 행동 패턴을 상당 부분 근사하지만, 작은 모델은 흔들리거나 기본적 합리 반응과 규칙을 혼동하는 경향이 나타난다. Equation-to-Behavior RL은 작은 모델의 belief 정확도를 개선하며, 또한 다양한 decision-maker 분포로 학습하면 Bayesian-only 학습 대비 평균 belief change 성능이 2.5%~12% 향상된다(설득 상황에서 GPT-5-mini와 상호작용하더라도). 전반적으로 ‘인간 의사결정 규칙’ 기반의 시뮬레이션 환경을 더 다양하고 현실적으로 만들 수 있어, 평가·훈련의 세밀함을 높이는 데 의미가 크다.



### From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs (https://arxiv.org/abs/2606.17648)
- **Prior Approaches**: 기존 연구는 특정 레이어에 어떤 정보가 인코딩되는지에 초점을 맞추는 readout/프로빙 계열이 많았고, 코드추론에서도 외부 성능(정답률)이나 정적 표현 특성 중심으로 평가되는 경향이 강했습니다. 하지만 표준 accuracy는 변수 추적은 잘하면서도 semantically equivalent loops에서는 실패하는 내부 원인을 분해해주지 못합니다. 결과적으로 비슷한 과제 정확도라도 내부에서는 완전히 다른 failure mode가 발생할 수 있습니다.

- **Core Contribution**: 이 논문은 코드추론 내부에 “brewing(정보 가용화) → self-decodable화(모델의 디코딩 파이프라인이 스스로 읽어내는 상태) → 해소 결과 분기”라는 라이프사이클이 존재함을 제안합니다. 또한 결과를 Resolved, Overprocessed, Misresolved, Unresolved의 4가지로 분류해, 외부 평가가 놓치는 차이를 진단할 수 있게 합니다. 레이어-wise linear probing(availability)과 Context-Stripped Decoding(CSD, readiness)을 결합한 dual diagnostic 프레임워크를 제시합니다.

- **Technical Challenges**: 핵심 난제는 “정보가 있다”와 “모델이 그 정보를 실제로 디코딩에 쓸 수 있다”를 분리해 측정하는 것입니다. 이를 위해 각 레이어의 last-token hidden state에서 정답을 선형으로 복원할 수 있는지(availability)를 probing으로 보고, 원래 코드 문맥을 제거한 프롬프트에서 그 표현이 스스로 정답을 산출하는지(readiness)를 CSD로 측정하며 prior 언어분포 효과는 logit subtraction으로 상쇄합니다. 16개 decoder-only Transformer(0.5B~14B)와 합성 코드 6개 task family에서, brewhing scaffold는 안정적이지만(정규화 brewing duration 24–42%), resolution 성공은 모델 능력/스케일/학습에 따라 달라짐을 확인합니다.

- **Empirical Impact**: 16개 모델, 6개 task family(총 24,300 샘플)에서 전체 Resolved는 41.5%에 그치며, 여러 과제는 30% 아래로 떨어져 표준 accuracy가 가리는 실패 분포의 크기를 보여줍니다. Function Call은 call depth가 1→3으로 늘면 Resolved가 61.1%→2.5%로 급락하는 등, 코드 프리미티브(연산/함수호출/루프)가 만드는 병목이 과제별로 뚜렷하게 드러났습니다. 더 나아가 Causal intervention(activation patching, layer skipping, re-injection)과 GT-free 신호(예: entropy rise)가 결합되어, 내부 상태가 관찰 가능하고 outcome-aware inference 전략 설계에 의미가 있음을 시사합니다.



### Beyond Domains: Reusing Web Skills via Transferable Interaction Patterns (https://arxiv.org/abs/2606.17645)
- **Prior Approaches**: 기존 web agent는 매 턴마다 LLM이 현재 페이지 관찰을 읽고 다음 low-level tool action 1개를 출력하는 방식(예: ReAct)이라, 긴 horizon에서 LLM 호출 수와 정책용 LLM completion이 급증해 비용·지연이 커집니다. 이를 줄이기 위해 web skills(성공 궤적/프로그램을 매크로로 묶은 callable skill) 라이브러리를 쓰지만, 재사용은 주로 instruction 유사도나 사이트 메타데이터에 의존해 held-out 사이트/도메인에서 재사용률이 낮아집니다.

- **Core Contribution**: SkillMigrator는 same website, same domain을 넘어선 cross-domain 웹 스킬 재사용을 목표로 합니다. 핵심은 TIP(Transferable Interaction Pattern)로, 학습 시점의 “검증된 skill + 그때의 레이아웃 구조 스케치”를 함께 저장해 테스트 시에는 텍스트뿐 아니라 레이아웃 유사도로 스킬을 찾아 live page에 참조를 grounding하는 것입니다.

- **Technical Challenges**: 가장 큰 문제는 cross-domain에서 ‘기능적으로 같은’ 상호작용이 있어도 라벨/DOM/표면 문구가 달라 의미 기반 검색만으로는 올바른 스킬을 안정적으로 찾기 어렵다는 점입니다. SkillMigrator는 (1) 접근성 snapshot의 small labeled tree에 대해 APTED 기반 tree edit distance로 레이아웃 유사도를 계산하고, (2) slot-filling을 instruction/동의어/문맥 단서로 value를 인스턴스화한 뒤, (3) gate(임계 점수 미만이면 skill mode를 끄고 primitive 제어로 fallback)로 약한 매칭 실행을 방지합니다.

- **Empirical Impact**: Mind2Web과 WebArena에서 성공 궤적 기준 평균 LLM-action count를 줄이면서도 성공률을 크게 해치지 않는 트레이드오프를 보였습니다. 예컨대 WebArena에서 policy LLM 호출이 ReAct 대비 8.5%(6.5→5.4), Mind2Web cross-domain에서도 PolySkill과 비슷한 성공률을 유지하면서 LLM-action count를 낮추며(6.9→6.2) 스킬 재사용률이 증가했습니다. 또한 레이아웃 신호와 gate, slot 동의어 풀 같은 구성요소가 성능 하락을 일으키는 민감도 결과로, 단순 데이터/파이프라인 이득이 아니라 ‘레이아웃 기반 재사용’이 개선의 중심임을 시사합니다.



### FinAcumen: Financial Multimodal Reasoning via Self-Evolving Experience Memory Harness (https://arxiv.org/abs/2606.17642)
- **Prior Approaches**: 금융 멀티모달 추론은 수치 계산, 정보 검색, 시각 해석, 시간적 정합을 함께 맞춰야 하지만, 기존 툴-어그멘티드 에이전트는 에피소드 간 상태가 거의 없어서 매번 유사한 실패 패턴을 재발견하곤 했습니다. 메모리 기반 접근도 있었지만 성공/실패 경험을 구분하지 못하거나, 관련성 낮은 검색 결과가 추론을 악화시키는 문제가 남아 있었습니다. 특히 금융에서는 불필요한 retrieval이 직접 정확도를 떨어뜨리고, 근거 없는 추정(hallucination)이 잘못된 분석 결론으로 이어질 위험이 큽니다.

- **Core Contribution**: 본 논문은 FinAcumen으로, tool-augmented multimodal reasoning에서 선택적으로 경험을 활용하는 ‘selective experience memory’ 프레임워크를 제안합니다. Financial Memory(FM)는 성공한 궤적에서 재사용 가능한 전략과 실패에서 유래한 주의 규칙을 분리해 누적하고, 추론 시에는 의미적 관련성이 보정된 임계값을 넘을 때만 메모리를 조건으로 주입합니다. 임계값을 못 넘는 경우에는 fallback 메커니즘으로 잡음 유입을 억제해 기반 정책으로 되돌리도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 검색된 경험이 현재 문제에 정말 도움이 되는지 판단해 불필요한 memory injection을 막는 것과 (2) 수치/시각 근거를 생성이 아닌 실행으로 고정해 신뢰성을 확보하는 것입니다. FinAcumen은 유사도 기반 retrieval에서 similarity threshold로 후보를 걸러 중복·순위를 정교화하고, 허용되지 않으면 명시적으로 base reasoning로 fallback을 수행합니다. 동시에 Financial Tools(FT) 환경을 통해 산술, 데이터 lookup, 차트 디코딩, 답 검증을 deterministic하게 실행하고, 도구 호출이 base와 동일하게 유지되도록 하여 성능 차이를 ‘경험 조건화’에서 오로지 설명할 수 있게 했습니다.

- **Empirical Impact**: 4개 금융 멀티모달 벤치마크에서 FinAcumen은 frozen 8B 비전-언어 모델을 기반으로 하면서도 툴+메모리로 일관된 성능 향상을 보였고, finance-specialized 및 일반 범용 대비 경쟁력을 확인했습니다. 특히 모듈 단위 실험에서 FT는 실행 오류를 줄여 정확도를 크게 끌어올리고, 그 위에 FM을 더하면 전략 수준 지침까지 제공해 추가 상승이 발생했습니다. 또한 retrieval 불확실성 상황에서 selective activation이 신뢰도를 높인다는 분석이 제시되어, 실제 배포 관점에서 ‘정확도뿐 아니라 안정성’을 개선하는 접근으로 의미가 있습니다.



### Brick-DICL: Dynamic In-Context Learning for Automated Brick Schema Classification (https://arxiv.org/abs/2606.17637)
- **Prior Approaches**: 기존 BMS 점(Point)→Brick 스키마 클래스 매핑은 벤더/사이트별 라벨 관행 차이 때문에 표준화가 어려웠고, Brick의 클래스 수가 936개로 커지며 전통적 분류 모델로는 스케일 문제가 나타났다. 또 LLM의 도메인 특화 지식 부족과 부정확한 예시 선택 문제로 few-shot/in-context learning 성능이 흔들렸으며, 결과 검증을 사람이 반복해야 해 운영 비용도 컸다.

- **Core Contribution**: 이 논문은 Brick 포인트 분류를 위한 2단계 Dynamic In-Context Learning( DICL ) 프레임워크 Brick-DICL을 제안한다. 1단계는 metadata-RAG로 입력 메타데이터와 유사한 올바른 예시를 찾아 LLM의 초깃값을 만들고, 2단계는 class-RAG로 후보 Brick 클래스를 좁힌 뒤 상위 후보를 재평가해 정확도를 끌어올린다. 마지막으로 multi-LLM 합의/불일치 기반 필터링을 붙여 사람이 확인해야 할 저신뢰 케이스만 선별한다.

- **Technical Challenges**: 핵심 난제는 (1) Brick 클래스 공간이 너무 커 LLM이 탐색하기 어렵고, (2) BMS 점의 의미를 파악할 충분한 도메인 맥락이 프롬프트에 부족하며, (3) 사람이 검증하기에 결과 품질 관리 비용이 크다는 점이다. Brick-DICL은 metadata-RAG로 동적으로 관련 예시를 주고, stage2에서는 예측된 1단계 결과를 신호로 class embeddings 유사도 상위 20개 후보만 남겨 20→5→3으로 점진 축소 평가를 수행한다. 또한 여러 LLM이 stage1·stage2 결과를 각각 내고, All Agreement~Any-2 Consensus 같은 전략으로 불일치/저합의 샘플을 자동 플래그해 검증 효율을 높인다.

- **Empirical Impact**: 실험은 서로 다른 두 건물 데이터셋(B1, B2)에서 수행됐으며, Stage1 hits@1, Stage2 hits@1/3로 평가해 Brick-DICL이 모든 베이스라인(정적/랜덤/다이내믹 ICL 및 파인튜닝 BERT 계열)을 일관되게 능가했음을 보여준다. 특히 dynamic ICL 계열이 일반 ICL보다 Stage2에서 더 높은 Hits@3를 보이며, Brick-DICL은 샷 수/후보 클래스 수 조정 실험에서도 전반적으로 최적 구간을 찾아 성능을 안정화했다. 운영 관점에서는 multi-LLM 필터링으로 사람 검증 부담을 줄이면서 디지털 온보딩(표준화 매핑)을 더 빠르게 만드는 방향성을 제시한다.



### Closing the Feedback Loop: From Experience Extraction to Insight Governance in Verbal Reinforcement Learning (https://arxiv.org/abs/2606.17591)
Comments:
          Accepted to the ICML 2026 RLxF: Reinforcement Learning from World Feedback Workshop, RLxF@ICML 2026, Seoul, South Korea

- **Prior Approaches**: 학습하지 않는(학습 파라미터 고정) verbal reinforcement learning은 세상 피드백(결과/수익/수요 등)에서 언어 규칙을 뽑아 context로 주입해 성능을 올립니다. 다만 비정상 환경에서는 축적된 규칙이 오래돼 역전이(negative transfer)를 만들거나, 실패를 버리면 나중에 같은 조건이 돌아올 때 치명적 망각(catastrophic forgetting)이 발생하는 문제가 큽니다. 기존 방법들은 규칙 추출에는 공을 들이지만, 규칙이 ‘언제/얼마나 믿을 만한지’를 지속적으로 다루는 governance(통치)에는 상대적으로 빈약했습니다.

- **Core Contribution**: 이 논문은 retention-forgetting dilemma를 정식 문제로 규정하고, 이를 해결하기 위한 4가지 요구조건(R1~R4)을 제시합니다. 핵심 기여는 규칙(rules)–근거(evidence)–스킬(skills) 3계층 아키텍처에 outcome-driven 평가, 지속적 structured evidence, 비단조적 지식 라이프사이클, 그리고 compositional governance를 함께 엮는 feedback-driven curation loop를 설계한 점입니다. 결론적으로 성능을 좌우하는 병목이 “경험을 얼마나 잘 뽑느냐”가 아니라 “인사이트를 어떻게 통치하느냐”임을 보여줍니다.

- **Technical Challenges**: 비정상 환경에서는 같은 규칙이 어떤 시기엔 맞고 다른 시기엔 틀릴 수 있어, 단발성 성공/실패만으로는 신뢰도를 판정하기 어렵습니다. 논문은 규칙별로 여러 에피소드에 걸친 성과를 남기는 persistent structured evidence 로그를 도입해 잡음과 신호를 분리하고, 규칙은 삭제하지 않고 deprecated로 전환해 evidence는 보존하는 non-monotonic lifecycle을 구현합니다. 또한 skills가 evidence를 읽어 규칙 간 충돌 해결과 적용 우선순위, abstain(기권) 시점을 결정해 compositional governance 갭을 메우도록 설계합니다.

- **Empirical Impact**: 금융 예측(S&P 500, 2013–2017)에서 world feedback은 풍부하지만 지연·잡음·비정상성이 강합니다. curation loop가 빠진 학습-없는 방법들은 축적 경험이 누적되어 모든 지표에서 zero-shot baseline보다 악화되며, especially 위험조정성과(Sharpe)가 부정적으로 나타납니다. 반대로 full loop를 적용하면 방향 정확도는 +5.3%p, Sharpe는 약 2배 수준, 최대 낙폭은 60% 감소로 “같은 경험이 성능을 해치는지/개선하는지”가 governance 유무에 달려 있음을 실증합니다.



### Surrogate Assisted Pedestrian Protection Design via a Foundation Model Orchestrated Workflow (https://arxiv.org/abs/2606.17577)
- **Prior Approaches**: 기존 자동차 안전 설계에서는 CAE(유한요소해석)가 기준이지만, 보행자 충돌은 접촉 비선형·구조 응답·상태 전이가 얽혀 데이터 기반 대체(서로게이트)로 옮기기 어렵다는 한계가 있었다. 서로게이트 연구는 주로 예측 정확도에 집중해 설계 탐색 파이프라인(제약 조건 탐색+형상 생성)까지 end-to-end로 통합하진 못했다. 또한 서브컴포넌트/제한된 기하 변화에 강하게 묶인 mesh·graph 기반 접근도 있어 실제 프론트 범퍼 같은 스타일링 탐색 루프로 확장하기가 까다로웠다.

- **Core Contribution**: 이 논문은 보행자 보호(크래시 안전) 설계를 위한 foundation model–오케스트레이션 워크플로우를 제안한다. 핵심은 CAE로 학습한 서로게이트(부상 지표 예측) + 제약 기반 다양성 탐색(NSGA-II) + morphing 기반 3D 형상 생성 + LLM/VLM 인터페이스를 하나의 설계 루프로 묶어, 수동 trial-and-error를 줄이는 것이다. 차량 프론트 범퍼 사례에서 단 한 번의 탐색으로 35개의 안전 조건을 만족하는 대안을 초 단위로 도출했다는 점이 성과의 중심이다.

- **Technical Challenges**: 가장 큰 기술 난제는 크래시가 ‘외형-구조 결합’과 ‘비연속 접촉 동역학’ 때문에 고충실도 CAE를 대체하기 어렵다는 점이다. 이를 위해 논문은 구조를 정밀하게 모델링하는 대신, 초기 스타일링 탐색에 필요한 비교 관계를 보존하도록 spring–mass 기반 10차원 파라미터화로 문제를 의도적으로 단순화해 서로게이트 학습 가능 영역으로 만든다. 또한 서로게이트 오차를 관리하기 위해 conformal prediction으로 분포 가정 없이 prediction interval을 제공하고, 형상은 topology-preserving morphing으로 CAE 호환성과 치수 제어를 보장하며, LLM이 제약을 구조화해 구성요소 간 데이터 흐름을 오케스트레이션한다.

- **Empirical Impact**: 실험에서 서로게이트는 평균 R^2=0.87 수준의 예측 성능을 보였고, 각 부상 지표에 대해 conformal prediction interval로 안전 임계선 인접 후보를 선별할 근거를 제시한다. NSGA-II 탐색은 제약을 만족하는 다양한 설계 대안을 채우는 용도로 쓰이며, 후보 500개 평가가 CAE 반복(수주) 대비 수 초 내에 끝나 인터랙티브한 제약 재조정이 가능해진다. 나아가 VLM 기반 시각-언어 비교는 숫자 제약 외의 ‘시각적 차이’를 구조화된 코멘트로 제공해 디자이너의 후단 선별을 보조하며, 크래시 안전 같은 “어려운 물리 도메인”에서 foundation model을 통합 레이어로 쓰는 가능성을 실증한다.



### DeepInsight: A Unified Evaluation Infrastructure Across the Physical AI Stack (https://arxiv.org/abs/2606.17574)
- **Prior Approaches**: 기존 Physical AI 평가는 System 2(파운데이션 모델), System 1(비전모터), System 0(전신 제어)처럼 서로 다른 레이어를 각각 별도 harness로 측정하는 경우가 많다. 이 방식은 각 구간의 로컬 타당성은 유지하지만, 런타임·스코어링·추적(trace) 아이덴티티가 단절돼 레이어 간 회귀를 진단하기 어렵다.
또한 평가/오케스트레이션 프레임워크들은 대체로 단일 레짐(짧은 에피소드, 예측 가능한 비용, 텍스트 중심 등)에 최적화돼 있어, 에피소드 길이와 보상 의미가 수천 배 달라지는 전 구간을 동일 기반으로 잇기 어렵다.

- **Core Contribution**: DeepInsight는 파운데이션 모델 디코딩부터 수천 physics tick의 whole-body control까지 한 가지 “단일 런타임”에서 평가하도록 하는 인프라를 제안한다. 핵심은 이 이질성을 없애려는 통일이 아니라, task·resource·result라는 세 가지 좁은 추상화 뒤에 이질성을 보존하는 것이다.
각 서브시스템이 공통으로 공유하는 불변 요소는 episode driver 1개, resource-handle 프로토콜 1개, trace identity 체계 1개이며, 덕분에 성능/정확도(benchmark fidelity)와 교차 레이어 진단 가능성을 함께 노린다.

- **Technical Challenges**: 가장 큰 기술 난제는 서로 다른 작업이 요구하는 입력/관측/보상/종료 정의와 자원(LLM 추론, sandboxed execution, 병렬 시뮬레이션)을 한 오케스트레이터가 억지로 흡수할 때 결합(coupling)이 생긴다는 점이다. DeepInsight는 환경에 transient 상태를 “per-episode handle”로 옮겨 환경을 stateless로 만들고, expensive 백엔드는 inference plane과 sandbox plane으로 분리해 오케스트레이터의 비동기 예산을 보호한다.
또한 모든 이벤트를 하나의 structured trace에 동일 identity로 기록하고, 점수는 trace를 읽어 계산하며 새로운 judge를 trace 기반 reader로 확장해 데이터 스키마 결합을 최소화한다.

- **Empirical Impact**: DeepInsight는 파운데이션 모델 레이어에서 기존 peer orchestrator들이 제공하는 레퍼런스와 프레임워크 읽기를 오차 범위 내에서 재현하고, 단일 노드에서 동일 스위트를 더 빠르게 돌리며 노드 수에 대해 near-linearly 스케일하는 결과를 제시한다.
더 중요한 차별점은 production 환경의 전 레이어(System 2–1–0)에서 하나의 shared trace로 이벤트를 남겨, 한 레이어에서 시작된 회귀가 다른 레이어에서 나타나도 trace 상에서 원인을 국소화(localizable)할 수 있다는 “진단용” 성과다. 이는 레이어별로 따로 굴리는 harness 연합(federation)로는 얻기 어려운 교차 레이어 관측 이득으로 귀결된다.



### SEAGym: An Evaluation Environment for Self-Evolving LLM Agents (https://arxiv.org/abs/2606.17546)
- **Prior Approaches**: 기존 평가들은 self-evolving agent의 ‘에이전트 하네스 업데이트’ 과정을 분리하지 못한 채, 최종 점수나 단일 순차 곡선에 집중하는 경우가 많았다. 많은 벤치마크가 에피소드마다 상태를 리셋해 하네스의 지속성이 실제로 어떻게 재사용·전이되는지(개선의 재사용성, 잊어버림, 회귀)를 가리기 쉽다. 또한 validation 같은 중간 신호를 갱신에 섞는 문제나, 업데이트 시점/스냅샷별 진단 부재로 인해 비용 증가·불안정성 같은 요인이 함께 묻히곤 했다.

- **Core Contribution**: 이 논문은 SEAGym을 제안하며, 하네스 업데이트를 ‘평가 대상’으로 명확히 모델링한다. SEAGym은 Harbor-compatible 벤치마크를 train 배치 기반 동적 self-evolution task source로 바꾸고, frozen update-validation, held-out ID/OOD transfer, replay 진단, 비용/스냅샷 기록을 한 프로토콜 안에서 제공한다. 이를 통해 ACE, TF-GRPO, AHE 같은 서로 다른 self-evolving 방식을 공통 스케줄·관점으로 비교할 수 있게 한다.

- **Technical Challenges**: 핵심 기술적 난제는 업데이트가 ‘언제’ ‘무엇을’ 바꾸는지에 따라 성능이 전이/붕괴/회귀할 수 있는데, 이를 단일 지표로는 구분하기 어렵다는 점이다. SEAGym은 상태 스냅샷과 업데이트 규칙을 에이전트에 맡기고, 환경이 태스크 샘플링·피드백·스케줄·평가 뷰를 통제하는 RL-style 평가로 문제를 정리했다. 또한 롤아웃과 업데이트를 분리해 방법별 native update rule을 유지하면서도, 스냅샷 타이밍·데이터 분할·평가 뷰를 분리 평가함으로써 누수를 줄였다.

- **Empirical Impact**: Terminal-Bench 2.0과 HLE에서의 실험은 validation gains가 곧 held-out ID/OOD 향상으로 이어지지 않으며, 유용했던 중간 스냅샷이 이후 붕괴했다가 다시 회복하는 등 비단조적(dynamic) 진화를 보일 수 있음을 보여준다. replay 진단은 초기 실패를 고치면서도 일부를 잊는 ‘task churn’과, middleware/runtime contract 회귀 같은 실행 경로 불안정성을 함께 분해해준다. 더 나아가 batch size, source diversity, 그리고 rollout backend(모델)가 하네스 업데이트의 신뢰성과 전이를 좌우하며, 크로스 모델/ OOD 관점까지 분리하지 않으면 잘못된 결론을 내릴 위험이 있음을 입증한다.



### LLM-as-Judge in Education: A Curriculum-Grounded Marking Pipelin (https://arxiv.org/abs/2606.17507)
- **Prior Approaches**: 기존 연구는 LLM을 평가자나 채점기처럼 쓰되, 프롬프트·루브릭 해석 중심으로 설계되는 경우가 많았습니다. 그 결과 모델 출력이 공인 커리큘럼/채점기준에서 벗어나거나, 왜 그런 점수를 줬는지의 추적성이 약해 고위험 시험 적용에 한계가 생깁니다. 또 일부는 RAG로 지식 정확도를 높이지만, 교육 당국의 ‘마킹 규범’을 소프트웨어 레벨에서 체계적으로 고정하는 설계는 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 authorized curriculum artefact(학습목표·성과 밴드·용어 정의·마킹 가이드 원칙)를 구조화해 LLM-as-Judge 파이프라인의 설계 입력으로 박아 넣는 curriculum-grounded 방식을 제안합니다. 또한 질문별로 먼저 채점 루브릭(구조화된 기대 성취)을 만든 뒤, 그 루브릭을 기반으로 실제 채점 기준을 생성·평가하는 staged workflow를 사용합니다. 그 결과 일관성·투명성·공식 채점 관행과의 정렬을 강화해, 단순 프롬프트 자동화의 드리프트 문제를 줄입니다.

- **Technical Challenges**: 핵심 난제는 (1) 질문 텍스트에서 관련 주제·인지요구(learn to/learn about/outcome)를 정확히 매핑하고, (2) 그 매핑이 공인 문서 범위 안에서만 채점으로 이어지도록 ‘거버넌스’를 걸어야 한다는 점입니다. 논문은 HSC Syllabus의 learn statements와 NESA glossary의 directive verb 의미를 결합해 질문별 기술·개념·성과를 고르고, performance band descriptor로 품질 레벨을 보정하며, 13개의 marking-guideline principles를 생성 제약으로 삽입해 무리한 채점 기준을 막습니다. 더 나아가 intermediate artefact를 감사 가능하게 저장하고, 점수와 정당화를 기준선(어떤 스킬/아웃컴이 얼마나 드러났는지)으로 연결해 프로베넌스를 유지합니다.

- **Empirical Impact**: 예비 평가는 47개 문항에서 단일 튜터 채점과의 유사성, 그리고 생성된 정당화의 기준 문서 추적성을 중심으로 수행했습니다. 결과적으로 제안 파이프라인은 사람 튜터에 버금가는 채점 결과를 보였고, 정당화는 공인 커리큘럼 아티팩트와 마킹 스탠다드에 더 명시적으로 연결되는 경향이 나타났습니다. 또한 온라인 학습 플랫폼에 통합되어 override rate 같은 운영 신호를 통해 실제 사용에서의 견고성·개입 필요성을 관찰할 수 있는 기반을 마련했다는 점에서 의미가 큽니다.



### Can LLMs Be CEOs? Benchmarking Strategic Resource Reallocation with Multi-Role Agent Simulation (https://arxiv.org/abs/2606.17459)
Comments:
          13 pages

- **Prior Approaches**: 기존 LLM 벤치마크는 추론, 지식 검색, 경제적 합리성처럼 고립된 인지 과제에 집중해 왔습니다. 역할놀이나 멀티에이전트 연구도 등장했지만, 기능별(예: CFO/CTO)로 분화된 조언을 계층적 의사결정자가 “통합”하고, 정보 비대칭·조직 제약·시간 의존을 함께 다루는 능력은 체계적으로 평가되지 못했습니다.

- **Core Contribution**: 이 논문은 CEO 수준 자원 재배분(CEO-level resource reallocation)을 대상으로 하는 멀티에이전트 벤치마크 CEO-Bench를 제안합니다. CFO/CTO/COO/CMO의 상충하는 조언(각자 private signal과 우선순위)을 CEO 에이전트가 통합해 다라운드 제약 환경에서 실행 가능한 배분 계획을 만들도록 설계했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 상충 조언 통합과 동시에, 불완전 정보·동적 제약 아래에서 “얼마나 과감하게(boldness)” 실행할지 캘리브레이션하는 것입니다. 저자들은 구조적 타당성(plan validity)뿐 아니라 역할 통합(role integration), 조건부 과감성(conditional boldness), 과거 의사결정의 일관성(history-sensitive judgment)까지 4차원 평가로 분리해, 단순 생성 성공과 전략 품질을 구분하도록 했습니다.

- **Empirical Impact**: 5개 프론티어 모델을 13개 시나리오에서 비교한 결과, 대부분은 제약을 만족하는 구조적 역량은 높지만 전략 캘리브레이션에서 격차가 크게 나타났습니다. 또한 단일-advisor capture, 불확실성에서의 과도한 보수성, historical amnesia 같은 실패 모드가 관찰됐고, ‘통합을 더 깊이 할수록 더 신중해지는’ integration-boldness tradeoff가 확인되어, 향후 조직 의사결정 보조 시스템 설계에 직접적인 기준을 제공합니다.



### Dissecting model behavior through agent trajectories (https://arxiv.org/abs/2606.17454)
Comments:
          106 pages, 50 Figures, 16 Tables

- **Prior Approaches**: 기존 에이전트 연구는 모델 성능(추론 능력)이나 도구/실행 루프 설계에 집중해 “모델이 의도한 대로 동작할 것”을 비교적 가정해 왔습니다. 하지만 실제로는 모델의 intent(의도)와 harness(에이전트 실행기)의 실행이 어긋나며, 그 간극이 성능 저하의 숨은 원인이 될 수 있습니다. 특히 pass@1 같은 단일 스코어는 에이전트 내부 행동의 정렬 여부를 잘 드러내지 못한다는 한계가 있습니다.

- **Core Contribution**: 논문은 이 문제를 `intent-execution' gap(의도-실행 간극)으로 정식화하며, 모델이 의도한 것과 harness가 실제로 실행하는 것(그리고 반대)의 불일치를 핵심 이슈로 제시합니다. 또한 tools와 execution loops 못지않게 “harness-model alignment(실행기-모델 정렬)”을 최소화해야 한다고 주장합니다. 이를 위해 범용 패턴은 공유하고 모델별 선호만 소량 반영하는 간단·커스터마이즈 가능한 harness인 Simple Strands Agent(SSA)를 제안합니다.

- **Technical Challenges**: 핵심 도전은 서로 다른 frontier model들이 출력/계획을 다르게 구성하는데, harness가 이를 일관된 코드 상태 전이로 변환하도록 만드는 정렬을 잡는 것입니다. 논문은 SSA로 공통 패턴을 통합하면서도 모델별 선호를 소량 조정하는 방식으로 intent-execution gap을 줄이도록 설계했습니다. 더 나아가 SSA가 생성한 138k trajectory를 코드 상태 공간(code state-space)으로 표현해, 단순 성공 여부가 아닌 단계별 행동 정렬을 관찰할 수 있게 했습니다.

- **Empirical Impact**: 실험에서는 SSA를 통해 Claude, Gemini, GPT, Grok, Qwen 등 다양한 모델 공급자군에서 SWE-Pro, SWE-Verified, Terminal-Bench-2의 agentic benchmark들에 대해 기존에 보고된 pass@1을 재현하거나 개선하는 결과를 보였습니다. 더 중요한 점으로, pass@1이 프론티어 모델들 사이에서 비슷하게 보이는 상황에서도 edit frequency, testing activity, phase-transitions 같은 세분화 지표로 모델별 “문제 해결 노력 배분” 차이를 드러냈습니다. 이는 향후 에이전트 성능 향상이 단순 모델 스케일링을 넘어 harness 정렬과 평가 지표의 세분화에 달려 있음을 시사합니다.



### MapSatisfyBench: Benchmarking Satisfaction-Aware Map Agents through Behavior-Grounded Implicit Decision Factors (https://arxiv.org/abs/2606.17453)
- **Prior Approaches**: 기존 지도(map) 에이전트 벤치마크는 주로 여행/경로 계획, 툴 사용, 검색·합성 같은 ‘작업 완료’ 능력을 평가해 왔습니다. 그러나 일상형 map 질의는 사용자가 필요한 조건을 짧게 말하는 경우가 많아, 표면 텍스트만으로는 사용자 만족을 좌우하는 암묵적 의사결정 요인을 제대로 검증하기 어렵습니다. 또한 만족을 하나의 정답/라벨로 환원하기 힘들어, satisfaction-aware 의사결정을 직접 측정하는 데 한계가 있었습니다.

- **Core Contribution**: 이 논문은 행동 체인(behavior-chain) 근거로부터 사용자의 ‘완전한 니즈’를 복원하고, 그중에서 에이전트가 응답 전에 회수 가능한 암묵적 의사결정 요인만 식별·평가하도록 하는 restore-identify-filter 프레임워크를 제안합니다. 이를 바탕으로 MapSatisfyBench를 구축해, 명시적 작업 충족뿐 아니라 암묵적 요인 만족 여부까지 end-to-end로(툴 선택~사실성~응답 품질) 평가합니다. 결과적으로 지도 에이전트 평가 축을 ‘정답 생성/완료’에서 ‘사용자 수용 가능성에 맞춘 공간 의사결정’으로 전환합니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 어떤 암묵적 요인이 평가 가능(evaluable)한지, 그리고 (2) 만족을 단일 기준답 대신 정량 목표로 어떻게 변환하느냐입니다. 논문은 복원 단계에서 사전 컨텍스트/시공간 환경/사후 행동을 결합해 니즈를 복원하고, 식별 단계에서 질의에 없는 결정을 좁히는 요인을 분리한 뒤, 필터 단계에서 응답 시점에선 보이지 않는 근거를 배제해 ‘회수 가능하고 평가 가능한’ 요인만 남깁니다. 이어 hard constraint와 soft preference를 구분하고 evidence-supported weight로 가중치를 부여해, 만족 관련 요인을 객관적 스코어링 타깃으로 구성합니다.

- **Empirical Impact**: 실험에서는 다수의 LLM 기반 에이전트가 명시적 intent/작업 완료 및 사실 정합성에서는 비교적 높은 성능을 보였지만, 암묵적 요인(예: 제약/선호 조건) 만족과 그에 필요한 근거를 선제적으로 확보하는 데는 약했습니다. 이는 map 서비스에서 ‘가능한 답’ 생성은 쉬워도 ‘수용될 의사결정’을 만드는 능력은 별도의 개선이 필요함을 보여줍니다. MapSatisfyBench는 이러한 격차를 진단하고, 향후 지도 에이전트를 satisfaction-aware decision making으로 개발·검증하는 표준 벤치마크가 될 기반을 제공합니다.



### A Machine-Learned Comorbidity Index (https://arxiv.org/abs/2606.17450)
Comments:
          Accepted at the 43rd International Conference on Machine Learning (ICML 2026), Seoul, South Korea. 35 pages

- **Prior Approaches**: 기존 Charlson/Elixhauser(및 van Walraven 가중 Elixhauser) 같은 동반질환 지수는 진단코드를 고정 규칙과 선형 가중치로 압축해 주로 사망률에 맞춰져 있다. 이 때문에 다른 임상 결과로의 일반화가 약하고, 결과별로 다른 비선형 위험-중증도 관계를 충분히 반영하지 못한다. 한편 지도학습 기반 진단코드 예측은 각 결과마다 별도 모델링을 최적화해 task 간 간섭이나 표현 비일관성이 생길 수 있다.

- **Core Contribution**: 이 논문은 단일 scalar로 admission의 동반질환(중증도)을 요약하는 Machine-Learned Comorbidity Index(MLCI)를 제안한다. MLCI는 학습된 점수 s가 여러 임상 endpoint의 라벨과 함께 nHSIC(정규화 HSIC) 의존성을 최대화하도록 훈련해, 결과 공통의 admission-level ordering 신호를 비선형으로 포착한다. 또한 학습된 점수에 기반해 outcome별 위험 곡선을 분리해 사후적으로 해석할 수 있게 한다.

- **Technical Challenges**: 핵심 난제는 “여러 outcome이 공유하는 단일 ordering”이 실제로 존재할 때, 이를 한 스칼라 점수로 일관되게 복원하면서 특정 outcome에 학습이 지배되지 않게 하는 것이다. 이를 위해 (1) 진단코드 집합을 DeepSets 스타일의 permutation-invariant 인코더로 스칼라 점수로 만든 뒤, (2) multi-endpoint nHSIC 목적함수로 score–label 의존성을 비선형 커널에서 동시에 정렬한다. 더 나아가 결측 라벨은 intersection-valid 코호트를 써서 multi-task 학습/검증의 정합성을 유지하고, 각 outcome의 nHSIC 강도에 따른 2-stage 가중치로 task domination을 완화한다.

- **Empirical Impact**: MIMIC-III 및 MIMIC-IV의 여러 EHR 벤치마크에서 MLCI는 score–outcome dependence를 나타내는 지표들에서 강력한 단일 지표/단일 task 기준선들을 일관되게 능가한다. 단순 사망률 튜닝에 머무르지 않고 여러 endpoint에 걸친 공통 중증도 ordering을 더 잘 포착한다는 점에서, 임상적 stratification·cutoff 학습 관점의 실용성도 높다. 이론 분석은 multi-outcome nHSIC가 shared monotone ordering과 임계값 분할을 뒷받침하는 조건을 유한표본 관점에서 연결한다는 의미가 있다.



### Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems (https://arxiv.org/abs/2606.17443)
Comments:
          16 pages, 4 figures, 11 tables

- **Prior Approaches**: 기존 연구들은 LLM 추천에서 특정 브랜드가 반복적으로 우세해지는 브랜드 편향을 주로 관찰하거나(예: name/description 조작, prompt injection 등) 일부 요인을 분해해 왔습니다. 다만 그 편향이 언제 완전히 작동하고, 어떤 조건에서 쉽게 무너지는지(경쟁 동학의 ‘임계점’)는 충분히 정리되지 않았습니다. 또 마케팅 문구가 그 편향을 ‘활용’할 수 있는지, 그리고 여러 브랜드가 동시에 최적화할 때 시장이 어떻게 변하는지는 미지였습니다.

- **Core Contribution**: 이 논문은 스킨케어(경험재)와 검색재(USB 케이블·AA 배터리)에서 LLM 추천 경쟁을 체계적으로 측정해 ‘Conditional Monopoly’ 패턴을 제시합니다. 동일 스펙이면 유명 브랜드가 사실상 100% 추천을 독점하지만, 경쟁 제품이 아주 약간만 더 나은 품질 신호를 가지면 그 지배가 급격히 사라집니다. 더 나아가 authority-style 마케팅(허위 임상 근거 같은 ‘권위 신호’)이 그 독점을 깨는 방법이며, multi-brand GEO 상황에서는 상호 최적화가 게임이론적 딜레마로 이어짐을 보여줍니다.

- **Technical Challenges**: 첫째, 브랜드 우위가 단순한 name 인식인지, 실제로 ‘품질 신호 부재’에서만 나타나는 조건부 효과인지 분리해야 했습니다. 이를 위해 실브랜드 1개 vs 검증된 가상 브랜드 9개를 만들고, 평가지표(I AI, BOR)와 memory hallucination probe로 ‘프롬프트에 없는 특징을 끌어오는지’까지 점검했습니다. 둘째, 마케팅 문구 효과를 정량 비교하려고 Bias Surplus Value(BSV)로 권위·사회적 증거 등의 언어 신호를 ‘품질 개선에 준하는 등가치(별점 +0.17 등)’로 환산해 해석 가능하게 했습니다.

- **Empirical Impact**: 실험 결과, Conditional Monopoly는 유명 브랜드 기준 IAI=10.0 수준으로 나타나지만 품질 신호가 임계 수준을 넘으면(예: 별점 +0.075) 보상이 급변합니다. authority-style 언어는 monopoly를 BSV 관점에서 별점 약 +0.17에 해당하는 효과로 깨며, 다수 모델(GPT-4o-mini, Claude Sonnet, Gemini 3 Flash)마다 반응 양상도 달랐습니다. 마지막으로 모든 브랜드가 GEO를 채택하면 개인 이득이 붕괴(+0.802→+0.007)하고 미참여 브랜드는 추천을 거의 받지 못하는 ‘죄수의 딜레마’ 형태의 경쟁 균형이 관측되어 GEO를 보안 이슈뿐 아니라 신종 마케팅 실천으로 다뤄야 함을 시사합니다.



### Treatment Response Optimized Clinical Decision Support AI System via Digital Twin Simulation (https://arxiv.org/abs/2606.17405)
Comments:
          Accepted for presentation at the IEEE Engineering in Medicine and Biology Conference (EMBC) 2026

- **Prior Approaches**: 기존 CDSAS는 오프라인에서 학습한 정책을 배치 운영하는 경우가 많지만, dataset shift와 제한된 관측 범위 때문에 실제 임상에서 성능이 흔들릴 수 있습니다. 또한 치료를 “무엇을” 고를지는 다루더라도, 실시간 적응과 safety 제약(금기 치료 차단), 그리고 임상적 이득을 정량화하는 Treatment Effect(TE) 기준이 일관되게 결합되지 못한 한계가 있었습니다. Digital Twin(DT)·강화학습(RL)을 쓰더라도 불확실성 관리나 인간 검토로의 안전한 에스컬레이션이 약하면 신뢰 확보가 어렵습니다.

- **Core Contribution**: 이 논문은 TE 추정을 임상적 이득의 핵심 지표로 두고, patient Digital Twin(DT)로 치료 궤적을 시뮬레이션하며, RL로 순차 의사결정을 수행하는 온라인 적응 프레임워크를 제안합니다. 초기에는 과거 데이터로 학습해 안정적으로 시작하고, 운영 중에는 최근 환자 데이터 기반으로 지속 업데이트하되 safety guardrails(금기·활력징후 규칙, 범위 체크)를 함께 둡니다. 더불어 내부 모델 간 불일치가 큰 케이스만 불확실성 기반으로 clinician review를 요청하는 설계를 포함합니다.

- **Technical Challenges**: 온라인 환경에서 가장 큰 도전은 “학습은 계속하되” 규칙 기반 안전 위반을 막고, 새 분포에서도 정책이 요동치지 않게 만드는 것입니다. 논문은 Transformer 기반 DT 앙상블로 미래 상태를 예측하되 bounded 업데이트로 롤아웃 안정성을 확보하고, BCQ(behavior-constrained)와 support threshold로 검증된 치료 집합만 선택하게 제한합니다. 또한 앙상블 분산을 불확실성 신호로 사용해 query threshold를 넘는 경우에만 전문가 라벨을 요청하며, 일부 실험에서는 사전 outcome model로 자동 보상 라벨을 생성해 실제 루프를 모사했습니다.

- **Empirical Impact**: 합성 임상 시뮬레이터와 TCGA의 난소암 데이터(587명)에서 제안 방법은 기존 computational baselines 대비 치료 추천의 효과와 안정성을 모두 개선했다고 보고합니다. 특히 난소암 실험에서는 희소 보상(종양 무관찰 전환 27.5%) 상황에서도 예측 benefit이 크게 상승했으며, 추천 일관성(낮은 action entropy)과 함께 안전 제약은 100% 준수했습니다. 온라인 적응에서도 불확실성 기반 질의로 clinician consultation 비율을 줄이면서(합성 13.1%, 난소암 39.9%) shift 이후에도 더 많은 라벨을 축적하고 업데이트를 수행해 성능을 유지하는 경향을 보였습니다.



### Distributed General-Purpose Agent Networks: Architecture, Key Mechanisms, and Prototypes (https://arxiv.org/abs/2606.17368)
- **Prior Approaches**: 기존 P2P는 파일 탐색/전송처럼 정적 자원 공유에 강점을 보였고, 블록체인 P2P는 원장 일관성과 합의 효율을 중심으로 설계됐다. 반면 분산 에이전트 네트워크는 open-ended 작업을 위한 의미 기반 선언(의도·능력·상태·협력 제약)을 전파해야 하며, 에이전트는 단순 전달 노드가 아니라 추론·적응하는 주체라서 단순 결합만으로는 동작하지 않는다. 멀티에이전트 시스템/인터페이스 프로토콜 연구도 대부분 닫힌 환경이나 정형 메시지에 맞춰져 있어, 개방 의미 작업에서의 발견·신뢰·규칙 수립을 한데 묶는 체계가 부족하다는 점을 지적한다.

- **Core Contribution**: 이 논문은 분산 범용 에이전트 네트워크를 위한 시스템 수준 아키텍처를 제안하며, 핵심은 protocol adaptation layer(프로토콜 적응 계층)로 작업의 의미(semantic)와 네트워크 동작을 연결하는 ‘협력 제어 평면’을 만드는 것이다. 상위(에이전트의 작업 의도·역량 상태)에서 하위(브로드캐스트·연결·동기화·검증·협상 등)로 의미를 번역하고, 실행 결과를 다시 발견·신뢰·메커니즘 결정에 피드백하는 닫힌 루프를 형성한다. 또한 협력 라이프사이클을 관심 기반 파트너 디스커버리와 검증/협상 후 작업 실행의 두 단계로 정리하고, 이를 세 모듈(발견·거버넌스·실행)과 세 기술 루트로 매핑한다.

- **Technical Challenges**: 첫째, 의미 메시지는 크기·유효기간·순차 의존성이 있어 무작정 방송하면 중복과 지연이 커지므로, bodyless gossip(바디리스 가십)와 순차 로그로 ‘약한 일관성’ 범위의 의미 전파를 설계한다. 둘째, 개방 환경에서는 Sybil·코드 교체·책임 회피·도메인 간 평판 희석이 발생해 identity와 reputation을 verifiable하게 결합해야 하며, BAID 기반(user-코드-책임 앵커) 바인딩과 MG-EigenTrust로 다중 토픽 평판 업데이트를 수행한다. 셋째, 자연어 제약/룰로 표현되는 open task에 대해 수학적 유틸리티 모델이 부족하므로, Stackelberg-style 두 단계 게임에서 semantic attribution feedback을 이용해 협력 규칙(메커니즘 텍스트)을 생성·개선하는 semantic-gradient 루프를 제시한다.

- **Empirical Impact**: 실험/검증은 1) BAID 계열의 티어드 검증 오버헤드에 대한 프로토타입 측정, 2) MG-EigenTrust에 대한 토픽 간 변장·공모 공격 시뮬레이션으로 신뢰·거버넌스의 견고성을 확인하는 형태로 보고된다. 디스커버리 측면에서는 LibP2P/GossipSub 직관을 따르는 시뮬레이션에서 topic 기반 디스커버리가 stale candidate를 엄격히 배제하는 조건에서도 성공률과 지연-커버리지 특성을 개선하는 흐름을 제시한다. 저자들은 단일 end-to-end 배포보다는 아키텍처·메커니즘·분석·예비 실험을 결합한 ‘시스템 기반’ 프레임워크로서, 개방적이고 신뢰 가능한 확장형 에이전트 협업의 토대를 제공한다고 의미를 부여한다.



### SpeechDx: A Multi-Task Benchmark for Clinical Speech AI (https://arxiv.org/abs/2606.17339)
- **Prior Approaches**: 기존 임상 음성 AI는 질환별로 개별 데이터셋에서 학습·평가되는 경우가 많아, 조건 간 비교와 일반화(다른 데이터로의 전이) 측정이 어려웠습니다. 또한 모델이 녹음 환경·인구 구성·획득 장비 같은 교란 요인에 대한 스퍼리어스 상관관계를 학습해 분포 변화에서 성능이 무너진다는 문제가 반복적으로 지적돼 왔습니다. 표준화된 벤치마크가 부족해 “좋아진 것”이 임상 신호 때문인지 데이터 아티팩트 때문인지 가르기 어려웠습니다.

- **Core Contribution**: SpeechDx는 12개 공개 음성 데이터셋, 9개 건강/정서 조건, 총 27개 태스크를 아우르는 대규모 임상 음성 AI 벤치마크입니다. 핵심은 발화 생성 과정(개념화–정형화–조음)에서 질환이 주로 끼치는 단계별로 태스크를 구조화해, 서로 다른 질환을 공유 임상 메커니즘 관점에서 비교 가능하게 만든 점입니다. 또한 제한된 라벨 데이터 상황과 데이터셋 간 동일 질환 평가, zero-shot cross-condition transfer로 데이터 아티팩트를 배제하려는 평가 설계를 제공합니다.

- **Technical Challenges**: 임상 음성에서는 녹음 조건과 라벨 품질이 크게 달라져, 표현이 실제 임상 구조를 담는지 검증하기가 어렵습니다. 논문은 12개 SOTA 오디오 인코더를 하나의 프로토콜로 선형 프로빙해 태스크 전반을 일관되게 비교하고, 태스크 단계별 어려움 분해와 zero-shot 전이를 함께 수행해 일반화 실패 지점을 드러냅니다. 또한 입력 길이 차이를 chunk+mean pooling으로 처리하고, 학습/평가 분할을 speaker-disjoint 중심으로 구성해 누수를 줄이도록 했습니다.

- **Empirical Impact**: 결과적으로 whisper, Qwen3-TTS-Tokenizer, WavLM 같은 대규모 음성 모델이 전반 성능 기준에서는 강했지만, 어떤 인코더도 임상 스피치 전반에서 “신뢰할 만한 일반화”를 일관되게 보이지 못했습니다. 질환·태스크에 따라 승자가 달라져, 특정 단계(예: 감정의 개념화 영역)에서는 잘 맞지만 다른 범주로 갈수록 성능이 쉽게 꺾였습니다. 특히 현장에서 기대하기 쉬운 호흡/발성(phonatory/respiratory) 계열은 cross-dataset 일반화가 가장 취약했으며, SpeechDx는 이러한 진행 상황을 추적할 공용 평가 프레임워크를 제시했다는 점에서 의미가 큽니다.



### MemTrace: Probing What Final Accuracy Misses in Long-Term Memory (https://arxiv.org/abs/2606.17328)
- **Prior Approaches**: 기존 장기 메모리 벤치마크는 질문 row나 에피소드별 정답률을 합산해 평가하는 경우가 많았고, 이 때문에 한 가지 ‘사실(fact)’을 고정한 채 조건만 바꿔보는 분석이 어려웠습니다. 결과적으로 여러 질문이 같은 지식을 찌를 때도 독립 항목처럼 취급되어, 시간이 지나며 사실이 어떻게 ‘변질’되는지나 false premise 같은 위험 행동이 어떻게 갈리는지 드러나지 않았습니다.

- **Core Contribution**: 이 논문은 사용자에 대한 단일 typed fact인 ‘knowledge point’를 평가 단위로 삼는 MemTrace를 제안합니다. 각 지식 포인트를 memory age(세션 경과), question type(현재/과거/변화 궤적), evidence condition(근거 있음/없음/거짓 전제에 의해 모순) 3축으로 반복 실험해, pooled accuracy가 숨기는 실패 양상을 분해해 보여줍니다. 특히 유사한 종합 성능이라도 ‘변화 추적 실패’나 ‘거짓 전제 수정 실패’처럼 서로 다른 결함이 나타남을 강조합니다.

- **Technical Challenges**: 핵심 도전은 ‘같은 사실’을 고정하면서도 조건(시간 경과, 질문 목적, 근거의 존재/모순)을 통제해 반복 측정할 수 있는 벤치마크 프로토콜을 만드는 것입니다. 논문은 HaluMem-Medium 데이터를 기반으로 knowledge point를 구성하고, 각 체크포인트 윈도우에서 같은 fact에 대해 서로 다른 질문형/근거조건을 주도록 설계했습니다. 또한 정답률 하나로 행동을 섞지 않기 위해 Gist(의미 정확), Verbatim(정답 표현 완결성), response type(정답/기권/환각/오답)을 튜플로 채점해 진단 뷰를 제공합니다.

- **Empirical Impact**: 13개 메모리 시스템(4개 패러다임)을 MemTrace에서 평가한 결과, pooled 점수만 보면 비슷해도 실패 패턴은 크게 갈렸습니다. 특히 trajectory(변화 궤적) 질문에서 붕괴가 두드러져, 현재/과거 상태를 맞혀도 ‘시간에 따른 갱신’을 연결하지 못하는 문제를 드러냈습니다. 추가 진단에서 주요 병목은 retrieval 자체보다 ‘도달 가능한 근거(evidence)를 추론에 어떻게 쓰는가’였고, 실패 시 근거를 찾지 못한 경우보다 근거는 닿아 있는데 사용하지 못한 경우가 약 10배 많았다는 결론이 제시됩니다.



### Quantifying Consistency in LLM Logical Reasoning via Structural Uncertainty (https://arxiv.org/abs/2606.17312)
Comments:
          Published at ICLR 2026 Workshop on Logical Reasoning of Large Language Models. Accepted as best paper

- **Prior Approaches**: 기존 불확실성/신뢰도 평가는 샘플링한 답변들의 분산(output dispersion)이나 정답 합의 정도(self-consistency)처럼 ‘답이 얼마나 달라지는지’에 초점을 둡니다. 하지만 논리·수학 추론에서는 같은 정답(혹은 같은 오답)을 내더라도 서로 다른 추론 경로의 질이 달라질 수 있어, 답변 수준의 분산이 놓치는 구조적 정보가 존재합니다.

- **Core Contribution**: 이 논문은 모델이 생성한 여러 추론 후보에 대해 ‘자기 스스로의 선호를 얼마나 일관되게 순위화하는가’를 Structural uncertainty로 정량화합니다. 같은 쿼리에서 후보들끼리의 pairwise preference를 수집한 뒤 Bradley-Terry와 PageRank로 ranking 분포를 만들고, 이를 엔트로피 기반으로 across-trial 불안정성과 within-trial 모호성으로 분해합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 모든 후보 쌍을 비교하면 비용이 커서 비교 그래프를 어떻게 줄일지, (2) sparse pairwise 신호를 안정적인 전역 순위 분포로 어떻게 집계할지, (3) 그 결과가 단순 스타일 차이를 반영하지 않는지 프로토콜 내에서 신뢰 신호를 분리하는 것입니다. 저자들은 매 트라이얼마다 후보를 잇는 random spanning tree만 비교해 비용을 O(N)으로 줄이고, Bradley-Terry(정규화 포함)로 pairwise를 win probability로 확장한 뒤 PageRank로 분포를 만들며, 엔트로피 분해로 불안정/모호성을 분리합니다.

- **Empirical Impact**: 5개 LLM과 8개 벤치마크에서 구조적 신호는 답변 분산과 상호보완적으로 작동하며, 특히 논리·수학/지식 과제에서 신뢰할 수 없는 인스턴스 식별 성능을 개선했습니다. 반대로 HotpotQA 같은 factual retrieval에서는 구조 신호가 거의 균일 분포로 ‘collapse’되어, reasoning-level 일관성 평가가 무의미해지는 regime boundary를 진단하는 데 활용될 수 있음을 보여줍니다. 또한 within-trial 모호성은 정확도와 양의 상관, across-trial 불안정성은 음의 상관을 보여 ‘경쟁하는 추론 경로’와 ‘불안정한(신뢰하기 어려운) 추론’의 성격이 다름을 실증했습니다.



### Nothing from Something: Can a Language Model Discover 0? (https://arxiv.org/abs/2606.17289)
- **Prior Approaches**: 기존 연구는 수학·증명에서 large language model이 상위 벤치마크 성능을 내는 모습을 보여주며, 주로 사전학습 데이터/프로세스 슈퍼비전/강화학습·합성 데이터의 스케일링을 강조해 왔습니다. 다만 많은 결과가 학습 중 이미 비슷한 형식의 구조를 많이 봤을 가능성이 커서, test time에 ‘완전히 새로운’ 수 구조로 점프하는 out of distribution generalization 능력을 직접 입증하진 못했다는 한계가 지적됩니다. 또한 compositional generalization 연구가 있지만, 본 논문은 개념(예: zero) 수준의 불연속을 넘는 leap을 별도로 측정하려는 방향입니다.

- **Core Contribution**: 본 논문은 “양의 한 자리 산술(0 제외)로만 학습한 모델이 zero 개념을 test time에 자율적으로 발견/일반화할 수 있는가”를 가장 단순한 사례로 정식화합니다. 그 결과 GPT-2 크기 언어모델은 언어 pretraining 여부와 무관하게 zero로의 일반화를 zero-shot에서는 실패하며, 반면 zero가 포함된 소량 예시를 학습에 추가하면 few-shot 규모에서 성능이 크게 개선됨을 보여줍니다. 특히 언어 pretraining이 필요한 few-shot 수를 약 50% 줄여, 언어 능력이 수학적 발견을 scaffold할 수 있음을 시사합니다.

- **Technical Challenges**: 핵심 실험적 어려움은 “모델이 학습 단계에서 산술 기호·문맥을 우연히 접했는지”를 통제하는 것입니다. 이를 위해 GPT-2 스타일 모델을 대상으로 자체 정제한 OpenWebText 변형을 사용해, 사전학습 코퍼스에는 숫자·수학 기호가 사실상 없도록 만들고, 토크나이저/토큰화 차이도 수동 토크나이징 설계로 격리했습니다. 또한 zero가 등장하는 위치(답의 ones place 등)에서만 토큰을 제공하도록 구성해, zero-shot 실패가 데이터 오염이 아닌 ‘개념 비약’의 문제임을 분리해 보여주었습니다.

- **Empirical Impact**: 실험은 세 층위로 명확한 신호를 줍니다: (1) zero-shot 일반화는 모든 테스트 모델에서 관측되지 않고, (2) zero를 포함한 tens~hundreds 예시(few-shot)를 학습에 섞으면 언어 pretraining 유무에 관계없이 성능이 상승하며, (3) 언어 pretraining은 같은 정확도에 필요한 데이터 양을 평균 48.5% 줄였습니다(p=1.7×10−4). 마지막으로 zero가 ‘특별한가’를 다른 숫자 홀드아웃으로 점검했을 때, 중간 숫자는 더 쉽고 carry와 가까운 숫자(0 및 9 등)가 더 어렵다는 패턴이 나타났습니다. 이 결과는 수학 벤치마크 성능을 넘어, test time에 개념을 확장하는 메커니즘이 데이터·학습량·언어 scaffold에 얼마나 의존하는지에 대한 실증적 근거를 제공합니다.



### Skill-Constrained Model Predictive Control for Resilient Manufacturing Supply Chains (https://arxiv.org/abs/2606.17269)
- **Prior Approaches**: 기존 생산-재고 제어(MPC)는 노동력을 외생 자원으로 두거나, 기술·훈련·자격 변화를 제어 변수에서 제외하는 경우가 많았다. 반면 인력계획·훈련 연구는 스킬, 교차훈련, 학습/망각을 모델링하지만 보통 개방형 계획에 머물러 폐루프 재계획과 관측 업데이트를 충분히 반영하지 못했다. 또한 shop-floor 스케줄링 쪽 MPC에서도 ‘노동자 역량’의 학습/훈련을 명시적 제어로 연결하지 않는 한계가 있었다.

- **Core Contribution**: 이 논문은 SkillChain-Gym에서 스킬 상태가 동적으로 변하고, 자격(certification)은 임계값을 넘기면 ‘가능’해지며, 훈련은 미래 역량을 늘리되 현재 생산 시간을 잠식하는 폐루프 skill-constrained MPC를 제안한다. 매 시프트마다 혼합정수계획으로 생산·재고·백로그·훈련을 수평적으로 동시에 최적화하고, 첫 기간의 행동만 실행한 뒤 재계획(receding-horizon)한다. 종료 시점의 ‘인증된-용량 격차’를 해석 가능한 terminal value로 가격화해, 수평 예측 경계에서 놓치기 쉬운 스킬 병목을 비용함수에 연결한다.

- **Technical Challenges**: 핵심 기술적 난점은 훈련이 현재 생산과 같은 제한된 worker time을 소모하고, 자격은 미유지 시 망각으로 떨어지며, 스킬 습득은 지연(lag)을 갖는다는 강한 시간결합이다. 이를 해결하기 위해 예측 인증을 이진변수로 두고(진짜 실행가능성은 관측된 현재 인증과 연결), 스킬-자격 변환과 망각-성장 동역학을 혼합정수계획 제약으로 포함해 재계획 시점마다 ‘예측된 학습 이후의 생산 가능성’을 정직하게 반영한다. 또한 horizon 끝에서 남는 인증 병목을 terminal skill-bottleneck value로 페널티화하고, 예측 가시성(announce vs surprise) 제약이 지배하는 정보 누수를 방지한다.

- **Empirical Impact**: 실험은 seed-controlled 합성 환경에서 announced/surprise new-skill shock, 수요 충격, 결근(absenteeism), 예측·가용성 품질 모드, 수요-용량 경계 스윕, 훈련률 민감도, 네거티브 컨트롤까지 넓게 다루며, 정책들 사이 ‘지배’가 아니라 ‘조건부 이득(regime dependence)’이 핵심 결론으로 나타난다. 예측 가능성이 충분히 일찍 확보되어 훈련이 완료될 수 있을 때 MPC가 유리하지만, surprise shock이거나 수요-용량 경계 근처에서 반응 여지가 작거나, 반대로 쇼크 전 여유가 커서 정적 insurance가 싸게 먹히는 구간에서는 정적 cross-training 보험이 강하게 버틴다. 절편화한 ablation 분석으로 인증 유지 vs lapsed 재취득 vs greenfield 신규 습득의 기여를 분리해, adaptivity 자체보다 ‘forecastability가 있을 때만 predictive control이 값을 만든다’는 메시지를 경험적으로 강화한다.



### SkillChain-Gym: A Benchmark for Reskilling-Aware Production-Inventory Control under Disruptions (https://arxiv.org/abs/2606.17266)
- **Prior Approaches**: 기존 운영/재고 벤치마크(OR-Gym, MABIM, SafeOR-Gym 등)는 노동을 외생 변수로 두거나 아예 포함하지 않아, 훈련·망각·자격 유지 같은 ‘인력 역량 동학’을 평가에서 분리해왔다. 반면 인력 계획·듀얼 리소스 스케줄링 연구는 skills, cross-training, learning, forgetting을 다루지만, 공통 인터페이스·재사용 가능한 테스트베드 형태로 공개된 경우가 드물었다. 그 결과 ‘생산(서비스) 목표’와 ‘향후 역량 확보(재교육)’의 동시 의사결정이 가능한 정책 비교를 표준화하기 어려웠다.

- **Core Contribution**: 이 논문은 재교육을 고려한 생산-재고 제어를 위한 벤치마크 스펙 SkillChain-Gym을 제안한다. 단일 사이트 환경에서 작업자 skill 상태를 상태(state)에 포함하고, certification은 hard threshold로 생산 자격을 결정하며, 훈련(training)은 생산과 동일한 작업시간을 소모하는 명시적 행동으로 모델링한다. 이를 통해 훈련이 ‘무료 스케줄링 디테일’이 아니라 시간 예산을 둘러싼 실제 상호작용(현재 생산 vs 미래 역량)을 벤치마크에서 측정 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 훈련이 생산시간을 잡아먹는 기회비용 때문에, 망각과 자격 만료, 희귀 스킬 병목, 그리고 중간/비가시적 충격(disruption) 하에서 정책이 미래 역량을 어떻게 확보할지의 trade-off를 재현하는 것이다. 저자들은 (1) 연속 skill-수준과 hard threshold certification, (2) 망각으로 인한 자격 약화, (3) 생산과 동일한 per-worker time budget을 소모하는 훈련 행동, (4) seed-controlled 결정론적 재현, (5) 3가지 feasibility 모드와 projection 진단(투사 빈도/규범/위반 횟수)을 함께 제공해 공정한 비교가 가능하도록 했다. 또한 forecast 가시성(예고 충격 vs surprise 충격), capacity slack, 망각률을 변수로 두고 ‘어떤 regime에서 어떤 정책이 유리한가’를 분해해 분석하도록 설계했다.

- **Empirical Impact**: 실험은 horizon T=60에서 4가지 시나리오 계열(수요 급증, 결근, 새 제품/희귀 스킬, 무충격)을 다루며, 60-shift 장기 구간에서 paired 통계검정을 통해 정책 간 우열을 비교했다. 결과는 단순 랭킹이 아니라 regime-dependent 지도로, 망각이 현실적이면 충격이 없어도 maintenance training이 필요하고, 훈련을 할 수 있는 정책군이 production-only 기준선은 전반적으로 지배했다. 훈련 가능 정책군 내부에서는 병목이 forecast에서 보이면 adaptive/reactive 쪽이 유리하고, surprise shock·결근 같은 비가시 충격에서는 lean static cross-training이 보험처럼 강하게 작동하며, 이러한 경계는 capacity slack과 forgetting rate에 의해 좌우되어 ‘모든 상황을 한 정책이 이기기 어렵다’는 결론을 뒷받침한다.



### When Rules Learn: A Self-Evolving Agent for Legal Case Retrieva (https://arxiv.org/abs/2606.17220)
Comments:
          To appear in ACL 2026

- **Prior Approaches**: 법률 사건 검색은 긴 문서와 복잡한 법률 문장 때문에 질의-사례 간 어휘·사실 정렬이 어려워져 왔습니다. Dense retrieval은 발전했지만 LeCaRD-v2에서 BM25가 여전히 강한 기준선으로 남아 있으며, 여러 선행 연구도 lexical matching의 경쟁력을 강조합니다. 한편 LLM 기반 query rewriting은 가능해졌지만, 고품질 규칙 설계를 사람이 맡아야 하거나(도메인 지식 의존) 무작정 규칙을 생성하면 성능이 쉽게 흔들린다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 BM25를 유지하면서 rule-driven query rewriting 규칙을 학습 없이(training-free) self-evolution으로 자동 개선하는 프레임워크를 제안합니다. 핵심은 LLM 에이전트가 반복적으로 (1) 새 규칙 생성, (2) 규칙 조합에 대한 실험 계획, (3) 효과 없는 규칙 제거를 수행하며 규칙 집합 자체를 진화시키는 점입니다. 파라미터 업데이트 없이 자동 평가 환경의 피드백만으로 규칙 세트를 정련한다는 점에서 “해석 가능하면서 적응적인” 보강 전략을 지향합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 규칙 탐색 공간이 커지는 가운데, 단발성 실험 성과에 기대어 좋은 규칙을 너무 빨리 버리거나(과잉 제거) 반대로 나쁜 규칙을 너무 오래 남겨 탐색을 비효율화하는 문제입니다. 이를 위해 에이전트가 실험 조합을 이미 평가된 경우를 피하도록 제한하고, 규칙 제거는 self-consistency 기반 2단계 합의(여러 재판정 시도에서의 일관성)를 거쳐 “확신이 있을 때만” 수행하도록 설계했습니다. 또한 실험 계획 단계에서 과거에 검증된 조합을 앵커로 삼아 incremental하게 다음 실험을 구성하도록 유도합니다.

- **Empirical Impact**: LeCaRD-v2에서 제안 방법은 non-evolution baselines(인간 설계 규칙, greedy 규칙 선택 포함)를 outperform하며, 특히 gpt-oss-120b 같은 high-capacity core LLM에서 모든 recall cutoff에서 일관된 우위를 보였습니다. 분석 결과, self-evolution의 성패는 새 규칙의 단조로운 향상보다도 (i) 실험 계획 시 history를 활용해 더 좋은 규칙 조합 앵커를 찾는 능력과 (ii) 제거 타이밍을 보수적으로 결정하는 능력에 달려 있음이 드러납니다. 즉, 규칙 집합 진화 과정에서 LLM의 “실험 설계·선택”과 “규칙 제거 판단”이 성능을 좌우하며, 이는 법률 검색 분야에서 BM25 기반 보강의 실용적 방향성을 시사합니다.



### Beyond Parallel Sampling: Diverse Query Initialization for Agentic Search (https://arxiv.org/abs/2606.17209)
Comments:
          15 pages, 8 figures; under review at EMNLP 2026

- **Prior Approaches**: 기존 test-time scaling for agentic search는 depth(추론 턴을 늘리는 방식)나 breadth(여러 롤아웃을 병렬로 실행)로 성능을 끌어올리는 데 집중해 왔습니다. 특히 breadth에서는 k개의 독립 롤아웃을 뽑아 투표/선택 등으로 합치는 전략이 흔했지만, 병렬 스레드가 초기에 비슷한 검색 쿼리를 내면서 증거를 중복으로 가져오는 문제가 관찰됩니다. 이로 인해 병렬성이 ‘자원 낭비’처럼 작동하며, 다턴 탐색에서 스레드들이 동시 실패하는 상관 오류가 생길 여지가 있습니다.

- **Core Contribution**: 이 논문은 병렬 에이전트 검색에서 turn-1 쿼리가 이후 탐색을 고정(anchor)해 버리는 anchor collapse 현상을 분석합니다. 그리고 이를 줄이기 위한 training-free 개입 DivInit을 제안하는데, 첫 턴에서 n개의 후보 쿼리를 한 번에 생성한 뒤 Maximal Marginal Relevance(MMR)로 k<n 중에서만 서로 다른(다양한) 시드 쿼리를 골라 병렬 롤아웃을 시작합니다. 핵심은 이후 에이전트 검색 루프는 그대로 두고, “어디서부터 시작할지”의 분포만 바꿔 다양성을 확보하는 것입니다.

- **Technical Challenges**: 기여를 실제로 만들기 위한 관건은 ‘독립 샘플링’만으로는 첫 턴에서 다양성이 충분히 확보되지 않는다는 점을 넘기는 것입니다. DivInit은 첫 턴에 대해서만 oversampling pool을 만들고, token 기반 Jaccard distance와 MMR로 선택된 시드들 간 최소 거리를 키우는 방식으로 다양성을 강제합니다. 또한 별도 학습 없이 개입을 수행해야 하므로, 공통 풀 생성 1회와 선택된 k개에 대한 병렬 스레드 실행으로 compute를 맞춰 설계했습니다.

- **Empirical Impact**: 5개 오픈웨이트 모델과 8개 벤치마크에서 DivInit은 standard parallel sampling 대비 전반적으로 더 높은 성능을 보이며, 멀티홉 QA에서 matched compute 조건으로 평균 5~7점 개선을 보고합니다. 특히 WebWalker 등 open-web 계열에서 증가 폭이 크게 나타나고, 성능 향상이 모델 크기에 따라 커져 “다양화가 먹히는 용량 바닥(capacity floor)” 가능성도 시사합니다. 또한 turn-1에서의 쿼리 다양성이 이후 turn들까지 ATD로 이어지며, 첫 턴 분리만으로 충분한 효과를 낸다는 분석이 함께 제시됩니다.



### Visual Verification Enables Inference-time Steering and Autonomous Policy Improvemen (https://arxiv.org/abs/2606.18247)
Comments:
          Website: this https URL

- **Prior Approaches**: 기존 로봇 정책 학습은 훈련 데이터(시연)나 추가 fine-tuning에 크게 의존해 실환경 적응과 성능 향상을 유도했습니다. 반면 배치 이후의 배치-런타임 경험으로 “즉시” 성능을 올리는 방식은 제한적이거나, 사람 개입이나 추가 학습 파이프라인이 요구되는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 생성기-검증기(generator-verifier) 구조인 VERITAS를 제안해, 사전학습된 일반ist 로봇 정책을 generator로 두고 시각 기반 visual verifier가 추론 시점에 행동을 평가하며 steering하도록 합니다. 핵심은 추가 학습 없이도 inference-time에서 성능을 개선하고, 검증된 롤아웃을 감독 데이터처럼 활용해 이후 self-improvement(오프라인 정책 개선)까지 연결한다는 점입니다.

- **Technical Challenges**: 추론 시점에 행동 품질을 빠르고 안정적으로 판단해야 하지만, 학습 없이 작동해야 하므로 gradient-free 검증 설계가 필요했습니다. VERITAS는 gradient-free visual verifier로 행동을 평가해 inference-time policy steering을 수행하고, 검증된 self-generated trajectories를 다시 fine-tuning에 활용해 정책이 효과적으로 학습하도록 구성했습니다.

- **Empirical Impact**: 실험에서 inference-time verification은 추가 시연 데이터로 학습하지 않은 vanilla generalists보다 일관되게 성능이 우수했습니다. 또한 verified rollouts로 offline policy improvement를 수행했을 때도 일관된 이득이 관찰됐고, post-training 효율이 expert demonstrations에 필적하면서도 인간 개입 없이 진행된다는 점이 특히 의미가 큽니다.



### ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues (https://arxiv.org/abs/2606.18237)
- **Prior Approaches**: 기존 연구는 LLM agent의 재현성(reproducibility) 지원 여부를 보기 위한 benchmark를 제안했지만, 데이터 큐레이션과 평가에 상당한 수작업이 필요해 대규모 확장이 어렵다는 한계가 컸습니다. 또 많은 benchmark가 실제 재현 과정에서 발생하는 다양한 막힘(예: 누락된 실험 조건, 오류 재현 등)을 자연스럽게 반영하기 어렵다는 문제도 동반했습니다. 결과적으로 현실적인 “재현 장애물”을 폭넓게 스케일해 측정하기가 힘들었습니다.

- **Core Contribution**: 이 논문은 ReproRepo라는 스케일 가능한 재현성 평가 프레임워크를 제안합니다. 핵심은 실제 연구 과정에서 자연발생하는 GitHub issue(인간이 제기한 문제)를 재현 장애물에 대한 supervision으로 활용해, 수작업 부담을 줄이면서도 현실적인 blockers를 대규모로 수집·평가한다는 점입니다. 이를 통해 paper-repository 쌍만으로 에이전트가 어떤 종류의 재현 문제를 찾아내는지 체계적으로 감사(auditing)할 수 있게 했습니다.

- **Technical Challenges**: 기술적 도전은 (1) 방대한 ML 논문-저장소 쌍에 대해 (2) 인간이 보고한 issue를 재현 장애물의 의미적 정답으로 삼고 (3) 모델 agent가 code 실행 없이도 의미적으로 연관된 문제를 찾아내게 하는 평가 설계를 요구한다는 점입니다. 논문은 ReproRepo에서 최근 주요 컨퍼런스의 1,149개 논문을 대상으로, 서로 다른 frontier model-agent 설정 4가지를 구성해 paper와 repository 정보를 매칭하는 방식으로 측정합니다. 또한 agent가 “보이는 실패”와 “정확한 의미 영역”을 잘 짚지만 “정확한 국소화(localization)”는 여전히 부족할 수 있음을 분석으로 확인합니다.

- **Empirical Impact**: 실험에서는 code 실행 없이도 LLM agent가 paper-repository 쌍에서 실제 재현성 문제를 상당수 탐지할 수 있음을 보였습니다. 특히 Codex with GPT-5.5 조합이 연구 대상 논문 약 90%에서 인간이 보고한 semantically related blocker(의미적으로 관련된 장애물) 적어도 1개를 찾아냈습니다. 다만 정확한 위치 특정까지는 충분하지 않을 수 있어, 향후 실사용 재현성 감사에서 agent의 강점(가시적 실패 탐지)과 한계(정밀 localize)를 동시에 시사합니다. ReproRepo는 향후 real-world reproducibility auditing을 위한 재사용 가능한 평가 프레임워크로 기능할 전망입니다.



### Learning Red Agent Policy from Observations for Neurosymbolic Autonomous Cyber Agents (https://arxiv.org/abs/2606.18223)
- **Prior Approaches**: 기존 자율형 사이버 방어는 강화학습(RL)을 통해 보안 규칙을 end-to-end로 학습하거나, behavior trees(BT)에 learning-enabled components(LECs)를 결합한 Evolving Behavior Trees(EBTs) 같은 neurosymbolic 구조로 해석가능성과 적응성을 함께 노리는 흐름이 주로 제시됐다. 그러나 자율 네트워크는 partially observable이라 공격자(red)의 행동을 직접 관측할 수 없어, defender(blue)가 red의 정책/침투 단계(intrusion level)를 예측하고 학습하는 데 한계가 컸다.

- **Core Contribution**: 이 논문은 blue의 관측과 직전 blue 행동을 입력으로 받아, red의 행동(및 정책)을 모방(imitation learning 기반)해 runtime에서 예측하는 “Policy Learning Technique”을 제안한다. 특히 discrete states와 discrete actions 환경을 전제로 하며, neurosymbolic EBT 기반 방어 에이전트에 red action prediction 동작을 통합해 다양한 red 전략에 대응할 수 있게 만든다.

- **Technical Challenges**: 핵심 난제는 partial observability 하에서 red 행동을 역추정해야 하며, 네트워크 상태 전이는 red와 blue의 동시 영향으로 인해 단순한 관측-행동 대응이 성립하지 않는다는 점이다. 이를 위해 (1) inverse-dynamics와 forward-dynamics를 함께 써 latent red actions를 학습하고, (2) 그 latent를 다시 실제 discrete red action 구성요소(공격 이름·대상 호스트/서브넷)로 매핑하는 다단계 학습 파이프라인으로 해결한다.

- **Empirical Impact**: CybORG CAGE Challenge 2(공격자 행동은 MITRE ATT&CK 기반)에서 실험한 결과, red의 전술이 바뀌는 상황에서도 공격 행동 예측 정확도가 높게 나타났다. 또한 방어 에이전트가 prediction을 바탕으로 공격 징후를 조기에 포착하고 침투 수준을 시간에 따라 추정할 수 있어, 자율 사이버 방어에서 “공격자 추정+대응” 루프의 실용성을 강화한다.



### Looped World Models (https://arxiv.org/abs/2606.18208)
Comments:
          Technical Report

- **Prior Approaches**: 기존 world model은 관측을 잠재공간에서 예측하고 그 위에서 계획/학습을 수행하는 RSSM 계열(Dreamer, PlaNet 등)과, 토큰/시공간 잠재를 변환기로 바꾼 IRIS·DIAMOND·EMERALD 같은 방식으로 발전해 왔습니다. 그러나 고정 깊이(또는 모델 크기 증가)로는 롤아웃이 길어질수록 예측 오차가 누적(compounding)되며, 이를 버티기 위해 더 깊은 네트워크를 쓰면 파라미터와 추론비용이 함께 폭증하는 긴장관계가 남아 있습니다.

- **Core Contribution**: 이 논문은 Looped World Models(LoopWM)라는 새로운 루프드(looped) world modelling 아키텍처를 제안합니다. 한 번의 상태 전이(step)를 “공유된 transformer 블록을 반복 적용”해 잠재 상태를 점진적으로 정제(latent refinement)하고, 이를 통해 모델 크기·학습 데이터 규모와 별개로 ‘반복 깊이(iterative latent depth)’를 스케일 축으로 삼습니다. 또한 각 전이의 복잡도에 맞춰 inner-loop 반복 횟수를 자동으로 늘리거나 줄이는 adaptive computation을 도입합니다.

- **Technical Challenges**: 핵심 난제는 루프를 많이 돌려도 잠재 상태가 폭주하지 않게 만드는 수치 안정성(stability)입니다. 이를 위해 spectrally-constrained state-retention 파라미터화를 사용해 상태 유지 행렬의 고유값이 (0,1) 구간에 들어가도록 구성하고, 루프 반복이 길어져도 residual dynamics가 bounded 되게 보장합니다. 더불어 Poisson 기반 stochastic loop depth 학습과 entropy-regularised early-exit 게이트를 결합해 학습 중 손실 스파이크를 줄이면서 추론 시 적응적 종료가 가능하게 했습니다.

- **Empirical Impact**: 실험에서는 LoopWM이 기존 world model과 비교해 예측 정확도는 경쟁적이거나 더 높으면서도 파라미터 효율은 최대 100배까지 개선될 수 있음을 보여줍니다. 또한 더 긴 롤아웃에서도 안정적으로 예측이 유지되어, 단순히 모델을 키우는 방식보다 긴장관계를 더 직접적으로 완화합니다. 무엇보다 test-time에서 전이 난이도에 따라 반복 깊이를 조절해 평균 추론비용을 크게 절감할 수 있어, 실시간 제약이 있는 embodied/자율 시스템에 의미 있는 방향을 제시합니다.



### RubricsTree: Scalable and Evolving Open-Ended Evaluation of Personal Health Agents across Health Memory and Medical Skills (https://arxiv.org/abs/2606.18203)
- **Prior Approaches**: 개인 건강 에이전트(PHA) 평가는 정답이 있는 객관식 벤치마크(예: MedQA, MedMCQA)보다는 다회·오픈엔드 생성과 도구 사용 궤적을 다뤄야 하지만, 기존 방식은 이를 충분히 관측하지 못합니다. 전문가 라벨링은 임상 정합성이 높아도 비용과 시간 때문에 스케일이 어렵고, HealthBench처럼 대규모라도 고정된 정답 세트라 제품 개발 주기의 “지속 최적화” 요구를 따라가기 힘듭니다. 반면 LLM-as-a-judge(원격 채점)는 자동화는 되지만 주관성·런투런 불일치·임상 정렬 미스가 생겨 신뢰 가능한 평가 신호가 되기 어렵다는 문제가 있습니다.

- **Core Contribution**: 이 논문은 RubricsTree라는 스케일 가능한 평가 프레임워크를 제안합니다. 100개 이상의 원자 단위(atomic)·임상적으로 검증 가능한 Boolean 루브릭을 계층형(DAG)으로 조직하고, 의료 문헌/전문가 패널이 검증한 “규칙 기반 앵커”로 판단의 주관성을 줄입니다. 또한 쿼리별로 관련 루브릭만 선택해 평가하는 context-aware adaptive router를 두어, 전문가 정합성과 자동화 효율을 동시에 노립니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 개방형 건강 응답에서 전문가 판단의 일관성과 임상 정렬을 유지하면서 (2) 대규모 자동평가로 전환할 때 생기는 비일관성과 부정확한 감점/보상을 줄이는 것입니다. RubricsTree는 루브릭의 잎(leaf)마다 구체적 임상 기준에 근거한 결정형 검증 함수를 두고, 계층 트리에서 의미적으로 필요한 하위 루브릭만 라우팅해 노이즈가 줄어든 평가를 구성합니다. 더불어 평가기를 평가하는 meta-evaluation으로 ICC3·Cohen’s kappa, 맥락 열화(지시 누락/사용자 데이터 오류/부적절 지시 등)에서의 Detection Rate와 Mean Penalty를 함께 측정해 “평가기의 신뢰성” 자체를 검증합니다.

- **Empirical Impact**: 실험에서 RubricsTree는 강력한 대규모 baseline 대비 전문가 정합성에서 큰 폭으로 개선되며, 별도 6인 전문가 패널 기준 Overall ICC3와 Cohen’s kappa가 크게 상승합니다. 또한 맥락이 망가진 상황에서 응답을 안정적으로 감점하며(oracle perturbation에서 Detection Rate 90%대+), 기존 principle baseline은 일부 셀에서 오히려 열화 응답을 더 높게 보상하는 실패가 드러납니다. 더 나아가 RubricsTree를 structured instruction·응답 최적화 피드백·RL reward로 활용했을 때 HealthBench 계열에서 모델 패밀리 전반에 걸쳐 최대 약 66% 수준의 상대 성능 개선이 관찰되어, “제품 수준 개인 건강 AI의 지속 최적화 인프라”로서 의미가 큽니다.



### A Red-Team Study of Anthropic Fable 5 & Opus 4.8 Models (https://arxiv.org/abs/2606.18193)
Comments:
          White paper

- **Prior Approaches**: 기존 연구와 대응은 주로 단발성 프롬프트 기반 jailbreak에 초점을 맞췄고, 이에 따라 입력 난독화·인코딩 같은 정적 기법은 점점 방어되는 추세였다. 그러나 실제 위협은 모델의 거절을 읽고 재작성하는 adaptive adversary가 반복적으로 압박하는 형태로 나타나며, “남은 취약 지형(residual surface)”을 체계적으로 측정한 비교 실험은 부족했다. 또한 단일 judge 평가가 성공을 과대보고할 수 있어, 재판정 절차가 중요하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 Anthropic의 frontier LLM Opus 4.8과 Fable 5를 대상으로, HackAgent 자동 red-teaming으로 7,8267,826개의 해로운 의도(intent) 전반에 대해 jailbreak 견고성을 정량화한다. 특히 3개 judge model의 다수결로 “패널-confirmed” 성공만 집계해 단일 judge의 편향을 줄이고, 기술별로 취약 지형이 어디에 남았는지(적응형 탐색 vs 정적 난독화 vs 설득/리프레이밍)를 분해해 보여준다. 결론적으로 ‘안전 점수’가 아니라 ‘상대적으로 취약한 경로와 조건’의 지도(map)를 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 자동 공격이 실제 위해(harm)를 만들어내는지, 아니면 judge가 있는 상황에서 생기는 모호한 응답을 성공으로 오인하는지였다. 논문은 공격 중 빠른 scorer로 탐색을 유도하되, 실제 성공은 Qwen3.7 Max·Gemini 3.5 Flash·GPT-5.5의 3인 패널 다수결(2/3 이상)로 오프라인 재판정해 필터링했다. 또한 공격자는 black-box API만 쓰고(가중치/내부상태/로그프로브 접근 없음), 공격자 모델이 검열된 상태로 섞이지 않도록 uncensored open-weight 모델을 별도로 사용해 측정의 정합성을 확보했다.

- **Empirical Impact**: 실험 결과 두 모델 모두 majority의 공격에 저항하지만, 잔여 취약성은 무시하기 어려울 정도로 남아 있으며 특히 adaptive 반복 공격이 대부분을 차지했다. 가장 강한 tree-of-attacks(적응형 탐색)는 Opus 4.8을 전체 의도 중 11.5%에서 깨뜨렸고, Fable 5는 단일 자릿수(최대 6.1%)로 제한됐다; 반면 정적 obfuscation은 거의 중립화되어 성공이 크게 줄었다. 그럼에도 패널-confirmed harmful completion은 Opus 4.8에서 1,620, Fable 5에서 702가 전 harm category 전반에서 자동으로 탐지되었고, 공격은 사람이 개입하지 않아도 1~2단계 refinement 안에 성과를 내는 경우가 많았다. “89% resisted” 같은 헤드라인만으로 안심하기 어렵다는 점을 대규모·자동화된 실증으로 경고하며, 안전성 평가는 대항적(adversarial) 반복 압력까지 포함해야 한다는 메시지를 남긴다.



### Kolmogorov Regression for Robust Diffusion Policies (https://arxiv.org/abs/2606.18186)
- **Prior Approaches**: 기존 FD diffusion policy는 데이터를 격자처럼 유한차원으로 이산화한 뒤 score matching 기반의 DDPM 절차를 적용한다. 이때 그리드 아티팩트로 인한 temporal drift와 score 추정 오차가 롤아웃 전반에 누적되며, 특히 로봇의 closed-loop 제어에서는 장기 수평 성능 저하가 두드러진다.

- **Core Contribution**: 이 논문은 diffusion policy를 backward Kolmogorov equation(BKE)에 근거해 함수공간(infinite-dimensional) 관점에서 재정의한다. 확률적 score matching을 대신해 결정적 boundary-value PDE 문제로 “Cameron-Martin 공간으로의 lifting”을 수행하고, noise의 공분산 구조를 colored noise로 반영해 샘플의 정규화(regularity)를 제안한다.

- **Technical Challenges**: 핵심 과제는 무한차원에서는 Euclidean score(밀도 기반 그래디언트)가 정의되기 어렵다는 점인데, 이를 Cameron-Martin 로그-도함수 관점의 BKE로 우회한다. 또한 훈련은 precision-weighted Cameron-Martin loss로 바꾸고, 추론 시에는 동일한 colored reverse noise를 사용하며, BKE를 만족하는 정도를 Kolmogorov residual로 PDE 진단(보상 신호 없이 실패 탐지)한다.

- **Empirical Impact**: PushT 조작 벤치마크 PushT에서 Cameron-Martin loss는 최대 에피소드 리워드를 0.95로 끌어올려 MSE(0.78) 대비 17% 개선을 보였고, 추론 시 inter-step drift는 residual 크기 기반으로 67.6% 줄였다. 제조 라인 6-station CONWIP에서는 LSTM 대비 RMSE가 28.4% 낮아졌으며, starvation-event recall 1.0과 Precision@1=1.0, 신호대잡음비 13× 성능을 통해 병목 식별과 안정적 운영을 강화했고, Hamilton-Jacobi reachability로 deadlock은 무제어 대비 96% 감소(100개 시뮬레이션에서 351건 예방)를 보고했다.



### IUU+DB: Tracking Illegal, Unreported, and Unregulated Fishing, Seafood Fraud, and Labor Abuse through LLM-driven Information Extraction (https://arxiv.org/abs/2606.18181)
- **Prior Approaches**: IUU(Ilegal, unreported and unregulated fishing) 중심의 기존 추적은 특정 행동유형 또는 특정 지역/종에 치우쳐 전역적 정량화가 어렵습니다. 또 뉴스·정부·학술·NGO 자료가 단편적으로 흩어져 있어, 사건의 빈도·지리·종·행위자·범죄 유형 패턴을 한 프레임에서 비교하기가 힘듭니다.

- **Core Contribution**: 이 논문은 IUU를 넘어 불법 양식, 라벨 사기, 노동 학대, 무역 제재 회피 등 연관 공급망 범죄까지 포괄하는 IUU+를 제안하고, 이를 구조화해 분석할 IUU+DB를 구축합니다. IUU+DB는 LLM 기반 정보추출로 이질적 문서에서 사건과 핵심 데이터 요소(KDE)를 자동으로 분류·추출·정리하는 end-to-end 파이프라인입니다.

- **Technical Challenges**: 핵심 기술 과제는 문서가 길고 표현이 제각각이며 IUU+ 유형 간 뉘앙스 구분이 어려워 추출 신뢰성과 재현성을 확보하기 어렵다는 점입니다. 논문은 KDE 그룹 존재 여부를 먼저 판단해 LLM의 컨텍스트 부담을 줄이고, few-shot 기반 소스 분류, 중복 제거(deduplication)와 trend 묶기, 그리고 DSPy/MIPROv2로 prompt 및 필드 설명을 반복 최적화해 잡음(hallucination)과 불일치(mismatch)를 줄이는 방향으로 해결합니다.

- **Empirical Impact**: IUU+DB는 143개 국가, 11년치, 2,472개 소스에서 8,435건의 사건을 수집·구성했으며 140여 국가에 걸친 글로벌 트렌드를 조직화할 수 있음을 보입니다. 정성·정량 평가에서 범위 분류와 KDE 추출의 precision/recall/F1이 전반적으로 유효하고, 기준선 모델 대비 IUU+ type과 sub-behavior에서 10~15% 수준의 개선이 관찰됩니다. 동시에 scope classifier의 precision이 낮아 잡음이 생기는 약점도 드러나, 향후 정책·집행용 위험평가의 정확도 향상 여지를 제시합니다.



### All Smoke, No Alarm: Oracle Signals in Agent-Authored Test Cod (https://arxiv.org/abs/2606.18168)
Comments:
          Accepted at the 8th IEEE International Conference on Artificial Intelligence Testing, 2026

- **Prior Approaches**: 기존 연구와 실무는 테스트의 검증력을 주로 테스트 파일 존재, CI 통과, 코드 커버리지 같은 지표로 간주해 왔습니다. 하지만 테스트 오라클(명시적 assertion 등)이 없으면 코드는 실행돼도 동작이 맞는지 검증하지 못해 ‘test theater’ 문제가 생기며, 커버리지는 결함 검출과 약하게만 연결된다는 한계가 보고돼 있습니다. 또한 LLM이 생성한 assertion은 실제 동작을 잘못 “기대값”으로 학습하는 등 오라클 품질이 흔들릴 수 있지만, 에이전트가 낸 PR에서 오라클이 실제로 무엇을 포함하는지는 대규모로 분류·검증되지 않았습니다.

- **Core Contribution**: 이 논문은 에이전트가 작성한 테스트 패치에서 검증력의 핵심 신호인 oracle signal의 구성을 ‘문법적(syntactic) 분류’로 포착하는 8개 범주 택소노미를 제안합니다. 그리고 테스트 파일이 보이기만 했을 때 과대평가되는 검증력을, oracle signal의 강도(약함/강함) 관점에서 재정의해 실무자가 PR을 더 정확히 평가하도록 돕는다는 점이 핵심입니다. 나아가 오라클 강도와 머지, 그리고 사람의 리뷰 노력 사이의 관계를 실제 GitHub PR 데이터에서 정량화합니다.

- **Technical Challenges**: 가장 큰 어려움은 ‘테스트 파일이 있는가’가 아니라 ‘테스트가 무엇을 검증하는가’를 PR 단위로 안정적으로 측정하는 일이었습니다. 연구진은 파일 경로·이름·확장자를 기반으로 테스트 파일을 식별하고, 같은 PR 내에서 (PR, 파일명) 조합에 대해 커밋 패치를 누적 병합해 중복 없이 전체로부터 오라클 신호를 추출했습니다. 그런 다음 pytest, Jest 등 프레임워크의 assertion 패턴을 중심으로 약함(W1–W5)과 강함(S1–S3)을 나누어 라벨링 신뢰도(Cohen’s kappa=0.77)를 확보했으며, 이후 PR 결과 변수에는 에이전트 종류·PR 크기·저장소 인기도·작업 유형·주언어를 공변량으로 넣어 혼선을 줄였습니다.

- **Empirical Impact**: 86,156개의 누적 테스트 패치에서 80.2%가 약하거나(weak) 사실상 oracle이 없는(no explicit oracle) 것으로 나타나, 테스트 파일 수만으로는 검증력을 크게 과대추정한다는 결론을 뒷받침합니다. 특히 새로 생성된 테스트 파일에서는 에이전트별 강한 오라클 비율이 18%~67%로 크게 갈렸고, strong-oracle(PR best patch가 S1/S2/S3)일수록 머지율은 낮아 보일 수 있으나 PR 크기·인기도·작업 유형 등을 통제하면 strong oracle이 머지 가능성을 유의하게 높였습니다(OR=1.28, p<0.001). 즉, 오라클 강도는 리뷰/규모가 큰 PR에 더 자주 나타나며, 그 복잡도를 조정하면 ‘좋은 기여’를 가리는 숨은 신호로 작동한다는 점에서 실무적 임팩트가 큽니다.



### The Measurement Gap in the Automation of EU Law: Benchmarking Doctrinal Legal Reasoning under the EU AI Ac (https://arxiv.org/abs/2606.18158)
- **Prior Approaches**: 기존 법률 AI 평가는 주로 요약, 조항 인용, 문서 작성 같은 보조적(패러리걸) 작업의 성능을 측정하는 데 집중돼 왔습니다. 그 결과, 실제 법률 실무의 해석 핵심인 도크트린(docrinal) 법리추론을 제대로 판별할 수 있는 벤치마크가 부족하다는 문제가 드러났습니다.

- **Core Contribution**: 이 논문은 법리추론 자체를 평가하는 도크트린 법리추론 벤치마크 부재의 측정 공백을 겨냥합니다. 나아가 EU AI Act가 사법 영역 고위험 AI에 요구하는 “appropriate accuracy(적절한 정확도)”를 실행 가능한 기준으로 만들기 위해, 그에 상응하는 운영 지표(벤치마크)를 제공하는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 난제는 “언어 생성 품질”과 “법리추론의 정합성”이 다른 차원인데, 기존 평가지표가 전자를 과대대표해 후자를 가늠하지 못한다는 점입니다. 논문은 법률 해석의 중심인 도크트린 reasoning을 측정할 수 있도록 평가 설계의 대상과 기준을 재정의하는 방식으로 이 갭을 메우려 합니다.

- **Empirical Impact**: 현재까지는 대형언어모델이 중간 수준 이상의 법률 문서를 생성할 수 있어도, 실제로 법리추론을 수행하는지에 대한 검증은 불완전합니다. 이 벤치마크가 제시되면 사법 도메인에서 요구되는 정확도 논의를 실증적으로 연결할 수 있어, 법률 AI의 평가 체계와 규제 준수 방향성에 직접적인 영향을 줄 것으로 기대됩니다.



### ReAge3D: Re-Aging 3D Faces with View Consistency (https://arxiv.org/abs/2606.18156)
- **Prior Approaches**: 기존 2D re-aging은 GAN/latent manipulation 또는 diffusion 기반 이미지-이미지 translation(예: InstructPix2Pix 계열)을 통해 노화/회춘을 생성하지만, 3D로 확장하면 뷰마다 주름·피부 질감 같은 미세 단서가 불일치하기 쉽다. 일반적인 3D scene editing(2D diffusion 편집 후 3D 최적화)도 멀티뷰 일관성을 맞추려 feature/노이즈/손실을 보정하나, re-aging 특유의 “아주 미세하지만 중요한” 나이 관련 디테일에서 과도한 스무딩 문제가 남는다. 특히 텍스트 기반 편집 모델은 fine-grained age control과 identity preservation에 맞춰 설계되지 않아 3D 얼굴에서 품질 격차가 커진다.

- **Core Contribution**: 이 논문은 3D face re-aging을 목표로, 먼저 2D diffusion 기반 re-aging 모델 DiffReaging을 제안하고 이를 멀티뷰 일관성 파이프라인에 결합한다. 핵심은 각 뷰를 독립적으로 편집하지 않고, “이미 re-aged된 피벗 뷰의 내용을 다른 뷰로 전파”해 age-related 디테일의 뷰 일관성을 유지하는 center-out editing propagation 전략이다. 또한 전파 시 누락 영역을 채우는 Masked-DiffReaging으로, diffusion의 반복적 denoising 과정마다 알려진 픽셀 콘텐츠를 주입해 기존과 충돌하지 않는 재구성을 유도한다.

- **Technical Challenges**: 가장 큰 기술 난제는 뷰 간 정합성이다: 2D에서 생성된 노화 디테일이 뷰마다 조금만 달라도 3D 최적화 과정에서 디테일이 흐려지는 over-smoothing으로 이어진다. 이를 해결하기 위해 논문은 optical flow 기반 warping으로 피벗의 re-aged 정보를 이웃 뷰에 정렬하고, Masked-DiffReaging이 시간 단계마다 confidence mask로 알려진 영역을 고정한 채 누락 영역만 일관되게 복원하도록 설계했다. 더 나아가 center-out 방식으로 중복/상충 재구성을 줄이며, 생성된 멀티뷰 타깃을 3DGS(또는 다른 미분 가능한 렌더러) 최적화의 감독 신호로 반복 갱신한다.

- **Empirical Impact**: 실험 결과, 제안 방법은 기존 3D 편집 기법 대비 시각적으로 더 자연스럽고(주름·피부 텍스처가 뷰 전반에서 더 매끄럽게 유지) 정량 지표에서도 우수한 성능을 보였다. 또한 원하는 나이(타깃 age)에 대해 identity를 보존하면서 세밀한 age transformation을 제어할 수 있어, 2D 중심 접근의 한계를 3D face re-aging으로 실질적으로 확장했다. 결과적으로 3D 얼굴 생성/편집 분야에서 “픽셀 수준 멀티뷰 일관성”을 diffusion 기반 편집과 연결하는 새로운 설계 방향을 제시했다.



### Descriptor: Certus Caliber Classification Gunshot Dataset (C3GD) (https://arxiv.org/abs/2606.18135)
- **Prior Approaches**: 기존 연구는 주로 인터넷에서 수집한 gunshot audio를 활용해 비용을 줄였지만, 품질 편차와 label noise 위험이 커 일반화 성능을 떨어뜨릴 수 있습니다. 공개 데이터셋도 규모·다양성은 부족하거나, 메타데이터(총기/탄약/마이크 위치 등)가 빈약해 정교한 분석과 재현이 어렵다는 한계가 있었습니다. ShotSpotter 같은 상용 시스템이 보이기도 하지만, 독립 조사에서 오탐·미탐이 크다는 점이 드러나 여전히 정밀도와 재현율 향상 연구가 필요하다는 신호로 해석됩니다.

- **Core Contribution**: 이 논문은 총구 폭발(muzzle blast) 소리 분석을 위한 공개 데이터셋인 C3GD(Certus Caliber Classification Gunshot Dataset)를 소개합니다. 16개 칼리버에 걸쳐 28종 총기로부터 8000점 이상(현장 수집)의 데이터를 제공하며, 마이크 종류·위치·녹음 조건 등 메타데이터를 기존보다 더 촘촘히 담았습니다. 초점은 caliber-based classification이지만, gunshot detection, audio separation, audio signal processing에도 활용 가능하도록 설계됐습니다.

- **Technical Challenges**: 현장 수집은 비용과 시간이 많이 들기 때문에, 인터넷 데이터에 의존한 접근의 label noise 문제를 줄이면서도 다양한 총기·탄약·마이크 배치를 확보하는 것이 핵심 과제였습니다. 논문은 야외에서 여러 세션을 나눠 수집하고, 마이크 간 동기화를 위해 총구 온셋을 기준으로 수동 동기화한 뒤, 기준 채널에서 임펄스 피크를 찾아 클립(최대 1초)으로 자동 분할하는 파이프라인을 제시합니다. 또한 wav를 48 kHz로 표준화하고 메타데이터 검증(장비/탄도 DB 교차확인, 중복 표기, 청각·시각 QC)을 다단계로 수행해 데이터 신뢰도를 높였습니다.

- **Empirical Impact**: C3GD는 기존 공개 데이터셋 대비 더 넓은 총기·탄약·칼리버와 마이크/녹음 위치의 조합을 제공해, caliber 분류에서 더 현실적인 분포를 학습할 기반을 제공합니다. 다만 반대로 반사/배경잡음 같은 배치가 도시·실내 환경과 다를 수 있어, 논문은 학습 시 audio augmentations(잡음, room impulse response 합성, 마스킹 등)을 권장합니다. 결과적으로 현장 배치 일반화 격차를 줄이고, 엣지/모바일 배치까지 고려한 gunshot 분류 연구에 의미 있는 출발점을 제공한다는 점에서 impact가 큽니다.



### Towards Understanding and Measuring COGNITIVE ATROPHY in LLM Behaviour (https://arxiv.org/abs/2606.18129)
- **Prior Approaches**: 기존 정신건강 LLM 벤치마크는 심리지식, 진단 추론, 위기 대응, 혹은 정적 응답 품질처럼 ‘한 번의 응답’ 중심으로 평가되는 경우가 많습니다. 또한 합성/AI 변형 프롬프트를 쓰거나 단일 턴으로 끝나는 경우가 있어, 감정적으로 민감한 대화에서 반복 상호작용이 사용자 의사결정·대처·자율성을 어떻게 바꾸는지 검증하기 어렵습니다.

- **Core Contribution**: 이 논문은 반복 대화가 사용자 사고·대처·감정조절을 약화시키고 LLM에 의존하게 만드는 과정 위험을 Cognitive Atrophy(인지 위축)로 정식화합니다. Safety와 helpfulness를 넘어서 ‘시간에 따른 행동 변화’ 자체를 측정하려는 시도가 핵심입니다. 이를 위해 Cognitive Atrophy Bench와 분석용 지표(UIRI, ARI, trajectory summaries)를 함께 제안합니다.

- **Technical Challenges**: 핵심 난제는 민감한 상담 대화에서 인지 위축 같은 ‘과정 수준’ 행동을 신뢰도 있게 태깅하는 데 있습니다. 연구진은 임상심리 전문가가 20개 속성(사용자 맥락, 응답 행동, 전역 risk 플래그)을 설계하고, 훈련된 임상 리뷰어가 텍스트 span 근거를 연결해 5,324개 판단을 수행하도록 했습니다. 그 결과 gold-standard 합의율(78.8%)과 κ가 실사용 평가에 충분한 수준으로 확보되었고, 입력 수요를 UIRI로 분류해 실제 임상 난이도 조건에서 비교 분석을 가능하게 했습니다.

- **Empirical Impact**: 5개 LLM을 대상으로 single/multi-turn을 모두 평가한 결과, 모델들은 atrophy-aligned 행동을 중간~높은 수준에서 일관되게 보였습니다. 특히 사용자 안전 신호에는 비교적 잘 맞추지만, 사용자가 해결책·결정을 명시적으로 요구하는 상황에서는 자율성 보존에 덜 적응하는 경향이 관찰됩니다. 반복 턴이 진행될수록 directive advice, problem-solving, recommendation, topic shift, 과잉 검증형 validation 등 의존을 강화할 수 있는 패턴이 누적되며, ARI는 모델 간 전체 위험 수준은 비슷해도 ‘경로’는 다름을 보여줍니다.



### Embedded Machine Learning for Microcontroller-Class Edge Devices: Data, Feature, Evaluation, and Deployment Pipelines (https://arxiv.org/abs/2606.18122)
Comments:
          6 pages, 3 figures, 4 tables

- **Prior Approaches**: 기존 임베디드 ML 연구는 보통 모델 구조나 학습 전략에 초점을 맞추고, 실제 기기에서 데이터 수집→전처리→추론→행동까지 이어지는 워크플로의 공학적 선택은 상대적으로 숨겨지는 경우가 많았다. 또 class imbalance(클래스 불균형) 상황에서의 검증 방식이 불명확하거나, sampling·buffering·스케줄링 같은 지연/메모리 제약 설계가 충분히 다뤄지지 않는 한계가 있었다. 그 결과 성능이 벤치마크에서는 잘 나오더라도 현장 배치에서 재현성이나 안정성이 떨어질 수 있다는 문제의식이 제기됐다.

- **Core Contribution**: 이 논문은 microcontroller-class 플랫폼을 대상으로 임베디드 machine-learning workflow를 시스템 관점에서 통합적으로 정리한다. 특히 샘플링·버퍼링, feature extraction을 통한 차원 축소, 클래스 불균형 하의 validation, model/runtime co-design, streaming deployment 같은 엔지니어링 결정이 성능과 신뢰성에 미치는 영향을 전면에 내세운다. 또한 inertial motion recognition과 keyword spotting 두 신호 패밀리를 통해 구체적인 설계 흐름을 일관되게 제시한다.

- **Technical Challenges**: 핵심 난제는 메모리·에너지·latency가 매우 빡빡한 환경에서 데이터 파이프라인과 모델 실행을 동시에 맞추는 것이다. 논문은 2초짜리 3축 가속도 윈도우를 RMS 및 스펙트럼 기반 특징으로 바꿔 분류하는 식의 dimensionality reduction을 적용하고, 음성은 anti-aliasing 후 mel-frequency cepstral coefficients로 변환해 compact 1D convolutional network로 처리한다. 더불어 quantization, thresholding, scheduling, field monitoring까지 포함한 실전 설계 규칙으로 클래스 불균형 검증과 스트리밍 배치를 함께 해결한다.

- **Empirical Impact**: 두 대표 작업(inertial motion recognition, keyword spotting)에 대해 on-device 추론이 견고하게 동작하도록 구성 요소별(데이터 커레이션, 양자화, 임계값, 스케줄링, 모니터링) 설계 원칙을 제안한다. 특히 모델 자체뿐 아니라 runtime과 배치 조건이 성능을 좌우한다는 메시지를 실증 흐름에 맞춰 정리해, 필드 배치에서의 실패를 줄이는 데 의미가 있다. 임베디드 ML 실무자에게 ‘무엇을 실험해야 하는지’와 ‘어떻게 검증해야 하는지’를 설계 규칙 형태로 제공한다는 점에서 파급효과가 기대된다.



### Structural Role Injection in Handlebars-Templated LLM Prompts: Triple-Brace Interpolation, Delimiter Family, and the Limits of HTML Auto-Escaping (https://arxiv.org/abs/2606.18120)
Comments:
          7 pages, 6 figures

- **Prior Approaches**: LLM 프레임워크는 시스템/작업 지시와 사용자 데이터를 템플릿으로 섞어 넣는 경우가 많고, 이때 Handlebars의 {{x}}(HTML escaping)와 {{{x}}}(raw)가 “안전한 기본값”으로 권장돼 왔다. 기존 연구는 주로 모델 측에서의 방어(지시 계층 학습, instruction hierarchy, 구조적 채널 분리)나 일반적인 prompt injection 대응에 초점을 둬, “템플릿에서 escaping 모드가 경계구조를 실제로 얼마나 바꾸는지”를 독립 변수로 분리해 검증한 연구는 드물었다.

- **Core Contribution**: 이 논문은 Handlebars의 escaping 모드가 구조적 role injection(시스템/어시스턴트 턴을 위조하는 구분자 주입)에 미치는 영향을 정량화한다. 특히 escaped(default {{x}})가 모든 역할 구분자를 막는 것이 아니라, HTML escaping이 건드리는 문자(각괄호 등)로 만들어진 구분자만 일부 중화하고 그 외 구분자는 그대로 통과시킨다는 “경계 보호의 선택성”을 보여준다.

- **Technical Challenges**: 가장 큰 난제는 escaping의 효과를 모델 편향 없이 분리해 측정하는 것이었다. 논문은 (1) 모델-free로 Handlebars escaping을 적용했을 때 각 delimiter family의 역할 제어 토큰이 바이트 단위로 얼마나 ‘생존(survival)’하는지 정적 분석하고, (2) 실제 모델 호출 실험에서는 5760회 트라이얼(7개 delimiter family × 2개 공격 목적 × 4개 모델)을 통해 예측된 가족별 격차가 실제 ASR에도 동일하게 나타나는지 확인했다.

- **Empirical Impact**: 실험 결과, escaped 기본값은 angle-bracket 계열(ChatML, Llama-3, XML 등)에서만 공격 성공률을 크게 낮추지만, square bracket/colon/Markdown hash 기반 계열(Human:/Assistant:, [INST], ### 등)은 거의 영향이 없었다. 또한 모델이 이미 단순 지시만으로도 쉽게 넘어가는 경우(예: GPT-3.5 Turbo의 hijack)에는 escaping으로 헤드룸이 줄지 않아 효과가 제한됐고, Claude Haiku 4.5는 두 공격 목표에서 모두 거의 저항했다. 결론적으로 “템플릿 escaping을 prompt-injection 통제로 오해하면 안 되며”, instruction과 data의 구조적 분리 같은 진짜 방어를 설계에 포함해야 한다는 메시지를 강하게 뒷받침한다.



### Ternary Mamba: Grouped Quantization-Aware Training of W1.58A16 State Space Models (https://arxiv.org/abs/2606.18114)
- **Prior Approaches**: 기존 ternary SSM 연구는 주로 from-scratch 학습에 의존했습니다. Slender-Mamba는 Mamba-2에 BitNet-style ternary를 적용하면서도 150B 토큰 대규모 학습이 필요했고, 더 낮은 비트의 post-training quantization(PTQ)은 SSM에서 치명적으로 붕괴했습니다. 또한 Transformer용 post-hoc correction은 SSM의 recurrence 구조에 의해 누적 오류를 제어하지 못했습니다.

- **Core Contribution**: 이 논문은 “pretrained 체크포인트에서 QAT(quantization-aware training)를 소량 토큰으로 수행하면 ternary SSM도 효율적으로 얻을 수 있다”는 해답을 제시합니다. Mamba-2 1.3B를 W1.58A16(ternary)으로 압축하되, FP16 teacher의 knowledge distillation(KD)을 결합해 102M 토큰(약 4 GPU-hours) 만에 7-task zero-shot 48.1%를 달성합니다. 이 성능은 Bi-Mamba(48.4%)와도 신뢰구간 내에서 근접하며, from-scratch의 막대한 비용 없이도 가능함을 보여줍니다.

- **Technical Challenges**: 핵심 기술 난관은 ternary 양자화가 SSM의 recurrence를 타고 history-dependent 오류 누적을 일으킨다는 점입니다. 특히 learnable quantization scale을 쓰면 “zero-ratio collapse”라는 새로운 불안정이 발생해 sparsity가 폭주하지만, absmean을 학습 파라미터가 아닌 고정 재계산으로 바꾸면 약 26% 수준에서 자기조절 균형이 형성됩니다. 또한 Transformer에서 통하던 Kalman/James-Stein류 post-hoc 보정과 sigma-delta는 SSM의 구조적 오류 전파 때문에 전이되지 않음을 체계적으로 확인합니다.

- **Empirical Impact**: 실험적으로, naive W1.58 PTQ는 사실상 무작위 수준으로 붕괴하지만 QAT-from-pretrained+KD는 C4에서 PPL 및 zero-shot 정확도를 안정적으로 개선합니다. 3개 시드 재현성에서 분산이 매우 작아(예: C4 PPL 표준편차 0.01) 결과 신뢰도가 높습니다. 무엇보다 이 연구는 ternary SSM이 “비싼 from-scratch 학습이 필수”라는 통념을 흔들며, 엣지 배치포인트를 고려한 data-efficient quantization 경로를 제안했다는 점에서 의미가 큽니다.



### Learning Fair Pareto-Optimal Policies in Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2606.18111)
Comments:
          Accepted at the Reinforcement Learning Conference (RLC) 2025. 12 pages main + appendix, 8 figures, 4 tables

- **Prior Approaches**: 기존 공정성 연구는 MORL에서도 대체로 단일 정책(single-policy) 학습에 머물렀고, 가중합 같은 utilitarian 방식이나 GGF 같은 egalitarian welfare로 공정성을 “한 번의 기준”에 맞춰 최적화하는 형태가 많았습니다. 또한 Lorenz Condition Network처럼 학습 목표를 특정 return 타깃으로 조건화(conditioning)하는 접근은 미지의 선호에 대한 일반화가 제한적이라는 한계가 지적됩니다.
더 나아가 단일 정책은 선호가 동적으로 바뀌거나 사용자별로 달라질 때, 매 순간 적절한 공정 해를 선택하기 어렵습니다.

- **Core Contribution**: 이 논문은 multi-policy MORL에서 “공정한 Pareto-optimal 정책 집합”을 학습하는 프레임워크를 제안하며, 사용자 선호가 무엇이든 그에 맞는 공정 정책을 고를 수 있게 합니다. 공정성은 generalized Gini welfare function(GGF)을 통해 효율(기본 Pareto 비지배)과 형평(분배 정의 기반)을 함께 만족하도록 정의됩니다.
또한 단일 파라미터 네트워크로 convex coverage set(CCS)를 근사하는 방식이라, 여러 선호를 샘플링해 공정한 정책들을 스케일하게 모읍니다.

- **Technical Challenges**: 핵심 난제는 (1) GGF처럼 비선형 형평 함수를 공정성 보장과 함께 MORL 학습에 녹이고, (2) 선호가 미리 고정되지 않을 때도 정책 집합이 공정하게 유지되도록 하는 것입니다. 논문은 GGF가 concave, piecewise-linear일 때 공정 해가 CCS 안에 남는다는 이론을 제시해, 학습 목표를 CCS 근사로 정당화합니다.
실전 학습에서는 비정상(non-stationary) 정책을 accumulated reward history로 상태를 확장해 과거의 불균형을 동적으로 보정하고, 추가로 stochastic policy가 deterministic 대비 더 공정한 해를 낼 수 있음을 모델링·알고리즘으로 연결합니다.

- **Empirical Impact**: 다양한 도메인에서 제안 알고리즘들이 최신 MORL 공정성 기준 및 관련 baseline들과 비교해, 선호가 달라도 서로 다른 사용자 요구를 수용하는 “공정한 정책 집합”을 학습함을 보였습니다. 특히 GGF 기반 멀티폴리시 Q-learning, accrued reward history를 활용한 비정상 버전, 그리고 stochastic 확장 모델이 공정성 지표(welfare score)에서 일관된 이점을 보이는 방향으로 보고됩니다.
결과적으로 이 연구는 공정성을 단일 정책의 속성이 아니라 “선호 조건에 따른 정책 선택 가능성”으로 확장했다는 점에서 분야의 실사용 관점에 의미가 큽니다.



### Querying an astronomical database using large language models: the ALeRCE text-to-SQL system (https://arxiv.org/abs/2606.18108)
- **Prior Approaches**: 기존 text-to-SQL(T2S) 연구는 Spider, BIRD, ScienceBenchmark 같은 벤치마크에서 주로 실행 정확도(execution accuracy)를 중심으로 발전해 왔습니다. 그러나 SDSS 같은 과학 DB에서는 도메인 수치·약어·함수 사용과 스키마 규모 때문에 성능 격차가 크게 나타났고, 핵심 오류는 schema linking 실패와 존재하지 않는 가짜 스키마 생성으로 분석됩니다. 천문 분야에서도 LLM을 NER/QA/요약/리서치 어시스턴트 쪽에 주로 적용해 왔지만, 실제 커뮤니티 DB에서 바로 실행 가능한 SQL을 생성하는 흐름은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 ALeRCE(천문 알림 브로커) 데이터베이스에 대해 NL 입력만으로 실행 가능한 SQL을 생성하는 LLM 기반 T2S 시스템을 제안합니다. 이를 위해 110개의 NL/SQL 페어로 구성된 ALeRCE용 데이터셋을 구축·공개하고, in-context learning과 prompt engineering으로 동작하는 단계별(step-by-step) 생성 프레임워크를 제시합니다. 프레임워크는 schema linking, query classification, prompt decomposition, self-correction의 4모듈로 구성돼 직접 추론 방식의 취약점을 보완합니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 질문의 모호성으로 인해 어떤 테이블·컬럼을 써야 하는지(schema linking) 잘못 연결되는 점과 (2) SQL 문법 제약 때문에 실행 단계에서 에러(timeout, 문법 오류, 존재하지 않는 구조)가 빈번하다는 점입니다. 논문은 이를 해결하기 위해 먼저 테이블/컬럼을 분리 추출하고, 난이도 분류로 필요한 분해(decomposition) 여부와 프롬프트를 달리하며, medium/hard에서는 sub-query까지 단계적으로 생성합니다. 또한 실행 에러가 발생했을 때만 self-correction을 1회 수행해 해당 오류 유형에 맞는 프롬프트로 SQL을 수정하도록 설계합니다.

- **Empirical Impact**: 13개 LLM을 대상으로 perfect-match(PM) 기반으로 row identifier(예: object identifier)와 column identifier(컬럼명) 정확도를 평가했으며, 전체적으로 제안한 단계별 프레임워크가 direct-inference baseline을 일관되게 능가합니다. self-correction 모듈은 실행 오류를 반복적으로 만들던 실패를 줄이는 데 효과적이었고, 복잡도에 따라 PM이 하락하되 간단 쿼리에서는 매우 높은 수치를 보였습니다(Claude Opus 4.6 기준 row 0.97, column 0.94). 최종적으로 최상위 성능 모델로 Claude Opus 4.6, Gemini 2.5 Pro, Gemini 3 Flash, GPT-5.2-Codex가 제시되어, Rubin era의 데이터 민주화 관점에서 천문 커뮤니티 DB 접근성을 실용적으로 높일 수 있음을 시사합니다.



### S4oP: Operator-level Pruning of Structured State Space Models for Resource-Constrained Devices (https://arxiv.org/abs/2606.18096)
- **Prior Approaches**: S4와 S4D 같은 Structured State Space Models(SSMs)은 long-range dependency를 잘 잡지만, edge/IoT처럼 메모리·지연·연산량이 빡빡한 환경에서는 여전히 배치/배포 부담이 남아 있습니다. 기존 압축은 주로 convolution이나 Transformer의 parameter 단위·구조 단위 pruning에 치중해 왔고, SSM에서는 정확도와 추론 효율(지연)까지 함께 보며 structured pruning을 체계적으로 다룬 연구가 드물었습니다. 특히 S4/S4D 내부의 채널별 연산 구조를 “연산자(operator) 단위로” 끊는 관점은 상대적으로 탐색이 부족했습니다.

- **Core Contribution**: 이 논문은 S4/S4D에 대해 incremental operator-level pruning(점진적 연산자 단위 pruning) 프레임워크 S4oP(S4oP)를 제안합니다. 개별 파라미터를 깎는 대신, 특정 채널의 SSM operator를 비활성화하고 입력을 그대로 우회(identity mapping)해 tensor semantics를 깨지 않도록 설계했습니다. 또한 pruning 비율을 단계적으로 올리면서 fine-tuning과 평가를 반복해, efficiency-accuracy trade-off를 통제 가능하게 만듭니다.

- **Technical Challenges**: 관건은 “연산자를 꺼도” S4/S4D의 커널/FFT 기반 계산 흐름과 downstream 성능을 안정적으로 유지하는 것입니다. 논문은 채널 단위로 operator를 통째로 bypass 처리해 텐서 차원과 후속 연산 호환성을 보장하고, 매 단계마다 fine-tuning을 끼워 정확도 붕괴를 완화합니다. 또 pruning seed에 따른 민감도를 줄이기 위해 각 pruning rate마다 여러 random seed를 시도해 가장 좋은 후보를 다음 단계의 기준으로 삼는 탐색 전략을 사용합니다.

- **Empirical Impact**: 여러 벤치마크에서 pruning이 지연(latency)을 유의미하게 줄이면서도 성능을 대체로 보존함을 보였습니다. 특히 모델 operator의 최대 70%까지 pruning해도 대부분의 경우 원 모델 수준 성능을 유지하며, 추론 지연은 removed channel 비율에 비례에 가깝게 감소했습니다(과제에 따라 대략 40~60% 수준의 speedup 관찰). 정확도 측면에서는 깊이(depth)가 중요한데, 초반 레이어는 pruning에 더 민감하고 후반 레이어는 더 많은 제거를 견딘다는 “depth-dependent redundancy” 패턴이 일관되게 나타났습니다.



### EAGG: Embodiment-Aligned Grasp Generation via Geometry-Aware Graph Conditioning (https://arxiv.org/abs/2606.18092)
Comments:
          16 pages, 8 figures. Code is available at this https URL

- **Prior Approaches**: 기존 cross-end-effector grasp generation은 특정 end effector에 강하게 맞춘 파이프라인(예: Dex-Net, GraspNet 계열)을 만들거나, embodiment identity를 정적 토큰/라벨로 넣어 transfer를 시도하는 방식이 많았습니다. 하지만 topology·actuation coupling·contact geometry가 크게 다른 embodiment에서는 정적 conditioning이 sampling 중에 바뀌는 접촉 가능성과 충돌 패턴을 따라가지 못해 성능이 쉽게 흔들립니다. 또한 많은 방법이 raw joint 좌표로 embodiment를 억지로 동일화해 구조적 차이를 희석하거나, 반대로 구조를 너무 배제해 공통 transfer 신호를 잃는 딜레마가 있었습니다.

- **Core Contribution**: EAGG는 “embodiment 구조를 공유 생성기 내부에서 정렬(aligned)한다”는 방향으로 cross-end-effector grasp generation을 재설계합니다. 각 end effector를 topology-aware end-effector graph와 PCA 기반 저차원 control space로 표현해, 하나의 생성기가 서로 다른 raw joint 파라미터화 없이도 그럴듯한 grasp를 생성하도록 합니다. 또한 생성 중에는 geometry-aware 토큰을 반복적으로 갱신해(Iterative Geometry Injection, IGI) 조건이 현재 관절 상태와 항상 동기화되게 합니다.

- **Technical Challenges**: 핵심 난제는 저차원 control code가 있더라도, 조립/폐쇄 과정에서 실제 관절 기하가 변하면서 충돌·미스컨택트 가능성이 달라지는 점입니다. 정적 embodiment descriptor는 이 “샘플링 동안의 기하 변화”를 반영하지 못하므로, EAGG는 frozen end-effector-cognition backbone으로 현재 articulation에 대응하는 geometry-aware 토큰을 만들고 매 step마다 IGI로 다시 주입합니다. 여기에 그래프 컨디셔닝(기구학 결합/연결 구조)과 PCA 제어공간(embodiment-specific closure)을 함께 써서, 제어표현과 구조표현을 분리하되 상호보완적으로 정렬합니다.

- **Empirical Impact**: EAGG는 MultiGripperGrasp 벤치마크에서 6개 학습 end effector에 대해 평균 성공률 56.17%를 달성하며, specialized 학습 대비 1.10%p 이내로 유지하면서도 finetuning 및 zero-shot end effector로의 transfer를 보존했습니다. 또한 IGI는 pooled median contact distance를 0.239 cm에서 0.189 cm로 줄여 접촉 정밀도가 개선됐음을 보여줍니다. 연구는 embodiment 차이를 억누르는 대신, 생성기 안에서 구조적으로 정렬하면 cross-object 일반화와 cross-end-effector 전이를 동시에 강화할 수 있다는 실증적 근거를 제공합니다.



### Volterra Generative Models (https://arxiv.org/abs/2606.18071)
Comments:
          36 pages

- **Prior Approaches**: 기존 score-based diffusion은 Brownian 기반의 마르코프(무기억) noising을 써서 역시간 SDE를 비교적 쉽게 구성해왔다. 하지만 Brownian은 시간 의존성을 현재 상태로만 압축해 ‘지속되는 상관/메모리/다중 스케일’을 전방 과정에 잘 넣지 못한다는 한계가 있다. 이에 따라 Lévy-driven, fractional-noise 계열처럼 Brownian을 대체하려는 시도가 있었지만, 역시간 이론과 샘플링 안정성은 여전히 어렵다.

- **Core Contribution**: 이 논문은 Brownian perturbation을 Brownian 경로에 대해 누적되는 Volterra(점도-의존) 생성모델로 대체해, fractional 커널로 forward 과정에 메모리를 주입한다. 핵심은 비마르코프(non-Markovian)·비semimartingale(non-semimartingale) 역시간 공식을 그대로 쓰지 않고, Volterra 커널을 유한 차수 Markovian lift로 근사해 score-based 학습이 가능하게 만든 점이다. Hurst regime(H<1/2, H>1/2)에 맞춰 커널 근사(가우시안 쿼드러처 기반)와 학습/샘플링 절차를 분리 설계한다.

- **Technical Challenges**: 기여를 실제 모델로 만들 때의 난점은 비마르코프성이 아니라, lift 후에 공분산/역시간 동역학이 구조적으로 ‘퇴 degeneracy(특이/준-영 방향)’를 겪을 수 있다는 것이다. 특히 모든 마르코프 요인이 같은 Brownian으로 구동되면 보조 공분산이 ill-conditioned해지고, 부드러운 regime(H>1/2)에서는 유한차분 근사로 signed exponential weight가 생겨 near-null 공분산 방향과 zero diffusion loading 문제가 발생한다. 논문은 squared error bound를 보장하는 커널 근사와 함께, residual state에 대한 data-dimensional score만 학습하고 보조 Gaussian score는 해석적으로 계산하도록 재구성하며, stiff한 larger lift에는 Gaussian-bridge reconstruction sampler로 안정화한다.

- **Empirical Impact**: MNIST에서 FID 0.52(H=0.9, N=2)까지 개선하며, Brownian SDE baseline 및 여러 MNIST 벤치마크 대비 성능을 보였다. 또한 작은 lift의 persistent fractional 노이싱 조건을 CIFAR-10에까지 확장해 FID가 약 9.5 수준으로 나오는 예비 결과를 제시해 자연영상으로의 확장 가능성을 시사한다. 무엇보다 ‘메모리를 가진 noising’이 score-based 생성 품질과 수치 안정성에 동시에 영향을 준다는 점을 실증하며, bridge sampler를 통한 더 큰 lift의 실용적 통로도 함께 마련했다.



### When LLMs Analyze Scars: From Images to Clinically-Meaningful Features (https://arxiv.org/abs/2606.18063)
- **Prior Approaches**: 기존 스카(SCAR) 판별은 Vancouver Scar Scale(VSS), Patient and Observer Scar Assessment Scale(POSAS) 같은 기준을 반영하는 손수 설계 특징(색·질감·형상)에서 출발했지만, 영상 조건 변화에 취약했다는 한계가 있었다. 이후 CNN/ViT 같은 end-to-end 딥러닝이 성능을 높였으나, 임상 라벨 부족과 개인정보 제약 때문에 작은 데이터에서 과적합·일반화 실패가 잦고 결정 과정이 불투명해 신뢰 확보가 어렵다. 멀티모달 LLM(GPT-4V 등)로 직접 분류를 시도한 방법은 추론은 강하지만, 외부 서버 전송 이슈·재현성 저하·블랙박스 특성 때문에 임상 적용이 까다롭다.

- **Core Contribution**: 이 논문은 ScaFE(Scar Feature Engineering)라는 프레임워크로 LLM을 end-to-end 분류기가 아니라 지식 기반 feature engineer로 재정의한다. LLM이 VSS/POSAS 같은 임상 기준을 프롬프트로 제공받아, 결정적(deterministic)인 Python 코드 형태의 특징 추출기(ϕ)를 생성하고 이를 이용해 저차원·임상 해석 가능한 feature vector로 변환한 뒤 가벼운 분류기로 학습한다. 핵심은 “임상 지식은 코드를 통해 고정하고, 통계 학습은 소형 모델이 담당”하게 만드는 구조로 데이터 효율성과 해석가능성을 함께 노린다는 점이다.

- **Technical Challenges**: 가장 큰 기술 난제는 LLM의 의학 지식을 실제 영상 처리 파이프라인으로 정확히 ‘실행 가능’하게 옮기면서도, 결과를 재현 가능하게 만드는 것이다. 논문은 temperature=0 같은 설정과 함께, LLM이 생성한 코드를 문법 검사·샘플 실행·출력 차원 검증으로 반복 정제해 유효한 결정적 추출기를 확보한다. 또한 색/질감/형상 등 임상 범주에 맞춘 특징 그룹을 코드가 산출하도록 설계해, 모델이 임의의 통계량을 학습하는 것을 줄이고 임상 용어와 정합되게 했다.

- **Empirical Impact**: 40장(케로이드 20, 하이퍼트로픽 20)처럼 초소량 데이터 환경의 병리 scar 이진 분류 실험에서 ScaFE는 end-to-end 딥러닝 및 LLM 직접 분류(MMLM-Direct) 대비 일관되게 우수하거나 견줄 만한 성능을 보인다. 학습 데이터가 샷 수 2장/클래스까지 줄어들어도 성능 하락이 크지 않아 few-shot에서도 강건함을 시사한다. 특징 기여 분석에서는 morphological 특징 제거 시 성능 저하가 가장 커, 임상 경험과 맞닿은 판별 신호를 LLM-생성 특징이 잘 포착하고 있음을 보여준다.



### Security and Privacy Prompts in the Wild: What Users Ask LLMs and How LLMs Respond (https://arxiv.org/abs/2606.18062)
- **Prior Approaches**: 기존 연구는 LLM의 S&P(디지털 보안 및 프라이버시) 답변 품질을 주로 연구자가 만든 S&P 오해(misconceptions)나 FAQ 같은 입력으로 평가해 왔다. 반면 실제 사용자가 LLM에게 던지는 S&P 질문이 “무엇인지”, 그리고 그에 대한 답변이 얼마나 신뢰할 만한지(특히 세션 간 일관성)는 잘 알려지지 않았다.

- **Core Contribution**: 이 논문은 WildChat의 실제 사용자-LLM 대화 3.2M에서 S&P 프롬프트 14,727개를 수집·분류하고, 주제 분석을 통해 사용자의 S&P 질문 유형을 정리한다. 또한 권고/가이드 요청(advice-seeking) 프롬프트 270개를 별도로 구성해 LLM 답변의 품질과 동일 질문 반복 시 일관성까지 함께 평가한다.

- **Technical Challenges**: 핵심 과제는 (1) 실제 대화에서 S&P 프롬프트를 정확히 선별하고, (2) 답변 품질을 자동 채점하면서도 평가 편향을 줄이며, (3) 모순 여부를 세션 간 일관성 지표로 측정하는 것이다. 이를 위해 다중 LLM majority voting으로 S&P를 분류(정밀도 96%/재현율 74%)하고, checklist 기반 LLM-as-judge로 품질을 채점하되 self-preference 편향은 여러 채점 모델 점수 평균으로 완화했으며, 일관성은 답변 근거 문장(체크리스트 항목별 evidence quotes)을 NLI entailment으로 모순 가능성에 초점을 맞춰 계산한다.

- **Empirical Impact**: 결과적으로 상용 LLM이 오픈웨이트 모델보다 평균 품질이 높았지만, 평균 품질이 좋은 프롬프트에서도 반복 실행 간 상충 답변이 일부 발생해 사용자를 혼란시킬 위험이 관찰됐다. 특히 Llama 4는 평균 품질은 가장 낮았으나 반복 일관성은 가장 높았고, “품질만으로는 S&P 신뢰성을 규정할 수 없다”는 메시지를 실증적으로 뒷받침한다.



### When AI Says "I have been in similar situations": Synthetic Lived Experience in Peer-Like Caregiver Suppor (https://arxiv.org/abs/2606.18057)
- **Prior Approaches**: 기존 연구는 가족/비공식 돌봄이 우울·불안·건강 악화와 연결되며, 온라인 커뮤니티의 또래 지지가 정서적·실질적 회복에 중요하다고 봐왔다. 또한 LLM 기반 챗봇이 즉각적이고 비판단적인 정서 지원을 제공할 수는 있지만, 인간 또래 지지의 핵심인 ‘살아온 경험(lived experience)’의 진정성은 결여될 수 있다는 점을 지적해왔다. 다만 ADRD(알츠하이머 및 관련 치매) 환경에서 ‘개인 서사’가 어떻게 신뢰와 연대를 만드는지, 그리고 peer-like 프롬프트를 받은 AI가 그 서사 형식을 어느 정도까지 재현하는지는 충분히 분석되지 않았다.

- **Core Contribution**: 이 논문은 ‘합성된 lived experience 역설(synthetic lived experience paradox)’을 제시하며, AI가 또래처럼 말하게 만드는 같은 언어적 단서가 오히려 “경험이 있는 사람처럼 보이게” 하는 허위 정위치 문제를 만든다고 설명한다. ADRD 가족 돌봄 제공자 맥락에서 온라인 커뮤니티의 실제 또래 응답과, LLaMA·GPT-4o-mini·MedGemma의 peer-like 응답을 비교해 인간이 쓰는 개인 서사 유형 7가지를 도출하고 AI가 이를 어떻게(또는 얼마나) 흡수하는지 매핑한다. 결론적으로, 따뜻함·검증(validation)은 주되 ‘경험의 근거(provenance)’를 오해시키지 않기 위한 경계 설계가 필요함을 주장한다.

- **Technical Challenges**: 핵심 기술적 난제는 AI가 공감/검증을 위해 1인칭·과거지향 표현 등 서사형 언어를 생성할 때, 실제 경험의 지시대(referent) 없이도 경험을 암시하는 문장이 만들어진다는 점이다. 연구진은 심리언어 분석에서 LIWC-2015를 사용해 1인칭 및 과거 초점 같은 신체화된 서사 신호를 비교하고(인간 커뮤니티 vs 모델별/통합 AI), 이어서 인간 데이터에서 귀납적으로 개인 서사 7유형을 코딩한 뒤 AI 응답에 이 유형들이 어떻게 나타나는지 질적으로 대조했다. 그 결과 AI는 감정적 노동(emotional work)은 포착하되, 경험적 기반을 ‘조작/생성’할 위험이 있음을 서사 수준에서 확인했다.

- **Empirical Impact**: 정량 결과에서 인간 또래 응답(온라인 커뮤니티)은 1인칭 및 과거지향 언어 사용이 peer-like AI 응답보다 유의하게 높았고, AI는 오히려 청자 직접지시(2인칭) 성향이 더 강하게 나타났다. 또한 질적 분석은 인간이 care-navigation, grief/emotional survival, advice-through-experience, shared experience 등 7가지 개인 서사를 통해 신뢰·정규화·경계 설정·경고의 보호 기능을 수행한다는 점을 보여줬으며, AI는 이 정서적 기능을 일부 흉내 내지만 lived experience의 실제 근거 없이도 서사 형식을 만들어낼 수 있음을 드러냈다. 연구는 caregiver-support AI가 ‘peer-like framing’과 ‘fabricated lived experience’를 구분하는 메커니즘(투명한 검증, 근거 명시, 필요 시 human peer 연결)을 갖춰야 한다는 설계 가이드를 제공한다.



### When English Isn't the Best Teacher: Source Language Effects in Cross-Lingual In-Context Learning (https://arxiv.org/abs/2606.18033)
Comments:
          Accepted at 1st Workshop on Multilinguality in the Era of Large Language Models (MeLLM 2026), co-located with ACL 2026

- **Prior Approaches**: 기존 교차언어 ICL 연구는 소스 언어 선택이 성능에 중요하다고 보면서도, 성과를 좌우하는 요인이 fine-tuning 시대의 직관(언어 유사성, 표면 중첩 등)과 동일하다고 가정하곤 했습니다. 또한 검색(retrieval)이나 혼합 언어/코드스위칭 같은 프롬프트 구성 변수가 성능을 바꿀 수 있으나, 소스 언어를 어떻게 체계적으로 고를지에 대한 평가는 제한적이었습니다. 특히 generative 태스크에서는 원하는 언어로 출력이 생성되지 않는 language confusion이 핵심 장애물이지만, 이를 소스 선택 문제와 함께 넓게 다룬 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 7개 태스크, 6개 모델, 유형적으로 다양한 언어 집합에서 소스-타깃 조합을 전수에 가깝게 비교해, 교차언어 ICL에서 “fine-tuning의 기대가 그대로 전이되는가”를 실증적으로 검증합니다. 그 결과, 타깃 언어가 자기 자신에게서 항상 가장 잘 전이되는 것이 아니며(약 24%), 영어는 역으로 소스로서 최악인 경우가 적지 않음을 보여줍니다. 아울러 언어 유사성이 ICL 전이에 일관되게 강한 예측력을 갖지 못하고, 대신 모델 내부에서의 cross-lingual alignment 같은 대표성 기반 요인이 더 중요함을 제시합니다.

- **Technical Challenges**: 핵심 기술적 난점은 소스-타깃 조합 평가가 조합 폭발을 일으키는 점과, generative 평가에서는 language confusion 때문에 전이 품질을 측정할 출력 자체가 없을 수 있다는 점입니다. 저자들은 유형적으로 다양한 18개 언어를 골라 7개 병렬 벤치마크에서 test를 제한(각 1,000개 또는 전체)하고, aligned 예시로 콘텐츠 혼선을 줄여 전이 역학만 분리합니다. 언어 혼동은 Language Confusion Benchmark와 line-level Pass Rate(LPR)로 별도 진단해, 소스 선택이 출력 언어에 미치는 영향을 정량화합니다.

- **Empirical Impact**: 실험 전반에서 타깃을 잘하는 고자원 언어(예: 영어/스페인어/독일어/이탈리아어)가 소스로는 약한 경향이 나타나며, 언어 유사성보다는 모델 표현공간의 cross-lingual alignment가 성능 분산을 더 잘 설명합니다. 특히 low-resource이면서 non-Latin script인 언어가 소스로서 가장 유효한 경우가 많고, 고자원 Latin-script는 상대적으로 불리하다는 패턴이 통계적으로도 확인됩니다. 또한 language confusion은 태스크별 전이 성능과 항상 직접 연결되진 않지만, 소스-타깃 간 “donor/recipient” 대칭성을 유사하게 보이며, ICL에서 소스 언어 선택을 재설계해야 함을 시사합니다.



### Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation (https://arxiv.org/abs/2606.18024)
Comments:
          Accepted to the ICML 2026 Workshop on Continual Adaptation at Scale: Towards Sustainable AI

- **Prior Approaches**: 지속 적응에서의 catastrophic forgetting은 보통 파라미터 드리프트 억제, replay, distillation 같은 “기준점” 중심 접근으로 다뤄져 왔습니다. NTK 기반 선행이 cross-task 커널/오버랩을 통해 예측 드리프트 크기를 설명하긴 하지만, 어떤 출력공간 방향(드리프트 취약 방향)에서 실제로 망각이 발생하는지의 구조적 분해는 부족했습니다.

- **Core Contribution**: 이 논문은 forgetting을 function space에서의 NTK interference로 재정의하고, Task B 학습 전에 Task A 예측이 어떤 벡터 방향으로 얼마나 이동할지를 닫힌형(closed-form)으로 예측합니다. 특히 frozen-backbone + trainable linear head(선형 head PEFT-CL)에서는 predictor가 수치 오차 수준까지 정확하며, nonlinear adapters/full fine-tuning에서는 로컬 NTK 근사로 동작한다고 정리합니다. 또 vulnerable subspace가 old-task NTK eigenmodes의 소수 모드에 집중된다는 구조까지 함께 제시합니다.

- **Technical Challenges**: 핵심 난제는 “출력공간에서의 간섭”을 계산 가능한 식으로 바꾸는 것이었습니다. 해결책으로 NTK 선형화 하에서 Task B의 MSE(리짓 앵커 포함)를 이차 최적화로 환원하고, push-through identity를 이용해 forgetting 벡터를 cross-task kernel이 residual에 작용하는 형태로 변환했습니다. 그 결과 forgetting이 low-rank 투영(취약 고유모드)으로 귀결됨을 보였고, frozen linear heads에서는 Kronecker factorization으로 vulnerable rank 스케일링 규칙까지 도출합니다.

- **Empirical Impact**: 실험에서는 Split-MNIST/ Split-CIFAR-10에서 예측된 forgetting shift의 방향이 실제 shift와 cos 유사도가 0.99 이상으로 일치하며, frozen ResNet-18 선형 head PEFT-CL에서는 구조적으로 exact함을 확인합니다. 또한 forgetting energy가 1–6개 NTK 모드에 집중되고, C(출력 채널 수)가 커질수록 vulnerable rank가 k⋆≈C⋅kG로 증가한다는 관찰이 제시됩니다. 이에 따라 parameter-space regularizer(EWC, SI)가 shared-head 벤치마크에서 한계를 보이는 이유를 출력공간 간섭의 “선택성” 부재로 설명하며, spectral 정규화가 그 취약 모드만 겨냥해 일관된 성능 개선을 만든다는 메시지를 제공합니다.



### LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling (https://arxiv.org/abs/2606.18023)
- **Prior Approaches**: 루프드 LLM은 파라미터를 늘리지 않고 공유 블록을 반복해 내부 연산 깊이를 확장하는 방식으로 주목받아 왔다. 다만 표준 sequential looping은 루프가 늘수록 지연시간과 KV-cache 메모리가 증가해 배포가 어렵다는 한계가 있다. Parallel Loop Transformer(PLT)는 CLP(cross-loop position offsets)와 shared-KV gated sliding-window attention(G-SWA)으로 비용을 줄이지만, 루프 수를 어떻게 고르면 좋은지에 대한 진단은 부족했다.

- **Core Contribution**: 이 논문은 PLT의 루프 수 선택을 gain–cost 관점으로 재정의한다. 즉, 추가 루프는 표현(숨은상태)을 더 정교화할 수 있지만, CLP가 루프 경계마다 위치 불일치를 유발해 비용(설계적 제약)을 누적시킨다고 본다. 이를 바탕으로 LoopCoder-v2(7B)에서 루프 수별 성능을 재현하고, 왜 2루프에서 포화(saturation)가 생기는지 내부 기여 단위로 설명한다.

- **Technical Challenges**: 핵심 과제는 “루프 수가 성능을 좌우한다”는 거시적 결과를 “각 루프가 무엇을 하고 실패/성공하는지”로 번역하는 것이다. 저자들은 숨은상태 변화량, attention 진화, 출력 분포 변화, 그리고 CLP로 인한 intrinsic offset cost Ω(r)를 루프별로 계측해 gain과 cost를 분리해 추적한다. 그 결과, 루프 2는 유효한 정제(refinement)를 크게 만들지만 이후 루프는 업데이트가 축소되며(때로는 진동) 표현 다양성이 줄어든다는 기작을 제시한다.

- **Empirical Impact**: 실험에서 비루프 대비 2루프 모델은 코드 생성/추론, 에이전트형 소프트웨어 공학, 툴 사용 전반에서 폭넓은 성능 향상을 보인다. SWE-bench Verified는 43.0에서 64.4로, Multi-SWE는 14.0에서 31.0으로 상승했으며, 반면 3개 이상 루프는 비단조적으로 성능이 하락했다. 진단 지표는 이러한 비단조성의 원인을 ‘고정에 가까운 offset 비용’이 ‘줄어드는 추가 이득’을 상쇄하기 때문이라고 설명해, PLT 루프 수 선택의 실무적 기준을 제공한다.



### C2FL: Clustered Continual Federated Learning under Spatial and Temporal Drif (https://arxiv.org/abs/2606.18003)
- **Prior Approaches**: 기존 Federated Learning(FL)은 서버 기반 집계를 가정하거나 IID 가정을 두는 경우가 많아, 실제 CAS의 비IID(공간적으로 다른 분포) 환경에서 성능과 수렴성이 흔들린다. clustered FL(CFL)은 비슷한 분포끼리 클러스터를 묶어 별도 모델을 학습하지만, 이동으로 인해 지역 분포가 시간에 따라 바뀌며 모델이 점점 stale해지는 문제를 충분히 다루지 못했다. Continual Learning(CL) 관점에서는 catastrophic forgetting을 완화하는 기법들이 있으나, 이를 decentralized(서버 없는) clustered 협업과 함께 통합해 해결한 선행 연구는 부족하다고 본다.

- **Core Contribution**: 이 논문은 모바일 공간 환경에서 privacy 민감 데이터와 대역폭 제약을 고려해, 완전 분산형 federated learning인 C2FL(Clustered Continual Federated Learning)을 제안한다. C2FL은 self-organizing spatial clustering으로 노드를 지역 구조에 맞춰 학습 그룹으로 묶고, mobility로 인한 순차적 분포 변화에도 적응하면서 지역적 consensus를 점진적으로 반영한다. 또한 experience replay와 dwell-time-aware adaptive averaging을 결합해 오래 머문 지역의 정보가 더 강하게 통합되도록 설계한다.

- **Technical Challenges**: 핵심 난제는 (1) 중앙 서버 없이도 이동하는 노드가 공간 구조를 반영해 안정적으로 클러스터를 형성해야 하고, (2) 지역 분포가 시간에 따라 변해 local 모델이 stale해지며(temporal drift) 이전 지식을 잃는 catastrophic forgetting이 발생한다는 점이다. 논문은 분산 클러스터링을 self-stabilizing leader election 기반으로 구현하고, CL을 위해 replay로 과거 경험을 보존한 뒤 dwell-time에 따라 adaptive averaging 강도를 조절해 지역 일치도를 누적 통합하되 기존 지식은 덜 훼손되게 만든다.

- **Empirical Impact**: 합성 실험에서 공간·시간 shift를 체계적으로 재현한 결과, standard federated 전략들은 mobility-induced drift 환경에서 유의미하게 성능이 저하되며 회복이 제한적이었다. 반면 C2FL은 temporal drift에 강건하게 collective adaptation을 복원해, 분산된 학습이 이동 상황에서도 안정적으로 유지됨을 보여준다. 즉, ‘공간 클러스터링 + 분산 FL + continual 적응’을 한 프레임워크로 묶어 모바일 CAS에서의 실사용 가능성을 강화했다는 점에서 의미가 있다.



### A T-API-Compliant ReAct Agentic Loop for Optical Networks: Generic vs. Domain-Specific Tool Abstractions (https://arxiv.org/abs/2606.18000)
Comments:
          4 pages, 2 figures, accepted for presentation at the 52nd European Conference on Optical Communications (ECOC), 2026

- **Prior Approaches**: 광통신 네트워크에서 에이전트형 LLM 관리는 주로 device-level YANG 모델, vendor별 SDN API, 또는 전용 시뮬레이터에 머물렀고, 표준·벤더중립 northbound 인터페이스인 T-API는 충분히 다뤄지지 않았다. 또한 ReAct보다는 plan-and-execute나 고정 멀티에이전트 파이프라인이 많아, 어떤 도구를 언제 호출할지 등 실행 흐름이 유연하게 결정되기 어렵다. 표준 벤치마크도 LLM-as-judge나 휴먼 루브릭 의존이 커서 환각을 정량·제어 관점에서 다루기 힘들었다.

- **Core Contribution**: 이 논문은 광통신에 대해 T-API를 준수하는 최초의 ReAct(Reasoning and Act) 에이전트 루프를 제안한다. 동시에 같은 에이전트와 시나리오에서 generic HTTP 도구(프리미티브 GET/POST)와 T-API 도메인 특화 도구(atomic·composite 16개)를 통제 비교해 “도구 추상화(tool abstraction) 자체”의 효과를 분리해 보여준다. 핵심 메시지는 T-API 객체에 시그니처가 직접 매핑되는 도메인 도구 계층이 에이전트의 정확성과 효율을 크게 끌어올린다는 점이다.

- **Technical Challenges**: 주요 기술 난제는 (1) LLM이 T-API의 RESTCONF 경로·메시지 구성, SIP 식별자 처리, QoT/OPM 기반 판단을 여러 단계로 안정적으로 수행해야 한다는 점과 (2) 툴 호출이 틀릴 때 생기는 오답·값 오류·모듈레이션 오류·근거 누락을 동일한 기준으로 분류·검증해야 한다는 점이다. 논문은 ReAct 루프로 “추론→도구 호출→관측→결정”을 동적으로 반복하고, generic 도구는 LLM이 직접 경로/바디를 구성하도록 둔 반면 domain 도구는 T-API 엔드포인트에 대응하는 atomic 및 다중 단계 묶음(composite)으로 오케스트레이션 부담을 흡수한다. 또한 oracle 기반 자동 검증기를 두어 실행 성공과 답의 정확성을 분리하고 실패 유형을 세분화해 측정했다.

- **Empirical Impact**: CORONET CONUS 기반 DT에서 10개 시나리오를 4개 Qwen 오픈 온프레미스 LLM으로 20회씩 실행한 결과, 도메인 특화 composite tools는 정답률을 약 90%까지 끌어올리며 평균 토큰은 run당 약 10.6k로 감소했다. 반대로 generic HTTP 도구는 모델이 커져도 성공률이 57–58%에서 평탄화되고 토큰은 30k–38k로 증가해, “더 큰 모델”만으로는 한계가 있음을 보여준다. 저자들은 도구 계층의 효과가 모델 선택 효과를 넘어섰고, 도구 호출 fine-tuning이 된 LLM에서 특히 성능 향상이 극대화된다고 정리했다.



### Multiple cyclicity and Wavelet Decomposition with Channel Correlation for Long-term Time Series Forecasting (https://arxiv.org/abs/2606.17996)
- **Prior Approaches**: 기존 장기 시계열 예측(LTSF) 연구는 주로 cyclicity(주기성)·trend(추세) 분해에 집중하거나, self-attention 계열/Transformer 계열로 전역 의존성을 추출해 성능을 끌어올려 왔습니다. 다만 실제 데이터에서 중요한 inter-channel correlations(채널 간 상관)를 충분히 분리·모델링하지 못해 예측이 덜 최적화된다는 한계가 지적됩니다. 또한 복잡한 설계로 인해 계산 효율이 떨어지는 경우도 많습니다.

- **Core Contribution**: 본 논문은 주기성, 추세, 그리고 채널 간 상관을 각각 분리해 학습하는 long-term time series forecasting 모델 McWC를 제안합니다. McWC는 Multi-cycle Construction Block(McB)로 여러 주기 성분을 구성하고, Channel-Correction Extraction Block(CEB)로 inter-channel correlations를 추출한 뒤, Multi-level Wavelet Decomposition Block(MWB)로 저주파(추세)·고주파(변동)를 멀티스케일로 복원합니다. 여기에 주파수 도메인 loss로 intra-channel autocorrelations(채널 내부 자기상관)까지 보완하는 FreDF를 함께 도입합니다.

- **Technical Challenges**: 핵심 과제는 (1) 데이터의 내재 주기를 이용해 다양한 주기 성분을 안정적으로 분해하고, (2) 채널 간 상관은 효율적으로 학습하되 잡음이나 불필요한 시간 패턴이 섞이지 않게 결합하는 것입니다. McWC는 top-k 주기 길이를 prior knowledge로 추정한 뒤 학습 가능한 주기 행렬을 통해 주기 성분을 명시적으로 구성·제거하고, CEB에서는 patch 기반 MLP로 패치 내부/간의 전역·국소 정보를 혼합해 채널 상호작용을 보강합니다. MWB는 멀티레벨 wavelet decomposition으로 주파수 스케일별 예측을 수행한 뒤 inverse wavelet로 재구성하며, FreDF로 시간영역의 단순 MSE가 놓치기 쉬운 자기상관 특성을 주파수 영역에서 직접 다루도록 학습을 설계합니다.

- **Empirical Impact**: ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity의 6개 실데이터에서 McWC는 baselines 대비 MSE·MAE를 평균적으로 유의하게 낮추며 state-of-the-art 성능을 보였습니다. 특히 일부 데이터셋에서 MSE는 2.5%~4.3% 수준의 감소를 보였고, top 순위 기준으로는 51번 1위, 9번 2위를 기록해 전반적인 예측 품질이 견고함을 드러냈습니다. ablation과 look-back 스터디에서도 McB·CEB·MWB 및 FreDF의 구성요소가 정확도와 장기 정보 활용에 기여하며, 계산량은 다른 비교 모델 대비 거의 한 자릿수 미만 수준으로 줄여 효율성까지 함께 확보한 점이 의미 있습니다.



### Recover Semantics First, Generate Better: Improved Latent Modeling for 3D MRI Reconstruction and Cross-Contrast Synthesis (https://arxiv.org/abs/2606.17989)
Comments:
          Code: this https URL

- **Prior Approaches**: 기존 크로스-대비(contrast) MRI 합성은, 누락된 시퀀스를 생성모델이 추론하도록 설계했지만 3D에서는 볼륨 크기 때문에 픽셀 공간 생성이 비싸다. 그래서 VAE/VQ로 3D를 latent로 압축한 뒤 GAN이나 Diffusion으로 합성하는 계층형 파이프라인이 주류다. 다만 기존 압축기는 장거리 해부학 일관성(long-range anatomical coherence)을 충분히 보존하지 못하고, 대비별(contrast별) 의미를 latent에서 뒤엉키게 만들며, MSE 같은 손실로 과도한 smoothing을 유발해 downstream 생성 품질을 떨어뜨린다.

- **Core Contribution**: 이 논문은 생성모델에 앞서 “semantics-first”로 3D MRI latent를 더 잘 만들자는 방향을 제안한다. 핵심은 Latent Harmonization Encoder(LHE)로 전역 해부학 의존성을 길게 잡아 구조 파편화를 줄이고, Semantic Recovery Block(SRB)로 self-supervised semantic teacher(DINO)의 고수준 priors를 latent에 주입해 대비별 의미 분리(contrast-wise separability)를 강화하는 것이다. 마지막으로 Anatomy-aware Frequency Loss(AFL)로 진단에 중요한 고주파 디테일을 상황별로 보존해 smoothing 부작용을 완화한다.

- **Technical Challenges**: 3D 볼륨에서 전역-국소 정보를 동시에 다루되, 압축(discretization) 과정에서 장거리 관계가 깨지지 않게 설계하는 것이 큰 난제다. 저자들은 로컬 컨볼루션 특징과 slice-wise ViT의 전역 context를 channel-wise feature alignment 후 residual fusion하고, FSQ로 discretize해 장거리 해부학 관계가 유지되도록 했다. 또한 semantic entanglement와 과스무딩을 동시에 줄이기 위해, SRB에서 quantized latent를 teacher semantic 공간에 정렬하고(Aspiring separability), AFL에서 해부학 경계(공간 gradient)와 teacher 기반 의미 주의(semantic attention)를 결합한 가중치로 고주파 잔차만 L1로 맞추는 방식으로 해결한다.

- **Empirical Impact**: BraTS와 IXI의 멀티-대비 MRI에서 재구성(3D MRI reconstruction)과 대비 합성(cross-contrast synthesis) 모두에서 일관된 성능 향상을 보였다. 예를 들어 BraTS에서 최고 baseline 대비 PSNR이 0.41dB 개선됐고, IXI에서는 33.65 PSNR 및 최저 LPIPS(0.0450)를 기록했다. 또한 해당 semantically aligned latent를 CycleGAN과 Latent Diffusion에 적용하면 PSNR이 각각 3.48dB, 2.78dB 상승해, 이득이 특정 생성기 구조가 아니라 “latent의 품질”에서 온다는 점을 실증했다.



### SegDINO: Introducing Multi-Scale Structure into DINO for Efficient Medical Image Segmentation (https://arxiv.org/abs/2606.17972)
Comments:
          Code: this https URL

- **Prior Approaches**: 기존 DINO 계열 self-supervised 표현은 전이 성능이 좋아 segmentation에도 적용해왔지만, 그대로 쓰면 멀티스케일 계층이 명시적으로 부족해 정교한 경계나 작은 병변에서 성능이 쉽게 떨어진다. 이를 보완하려고 무거운 decoder와 복잡한 upsampling/멀티스케일 fusion을 얹는 방식이 많아 파라미터와 연산 비용이 커지는 한계가 있었다. 또한 SAM 기반 접근은 zero-shot은 강하지만 downstream 적용 시 fine-tuning 비용 부담이 실무 효율을 낮춘다.

- **Core Contribution**: 본 논문은 SegDINO로 DINOv3 백본을 유지하면서도 “스케일 모델링”을 효율적으로 설계해 segmentation에 맞게 표현을 재구성한다. Token Pyramid Adaptation(TPA)은 서로 다른 DINO depth의 중간 토큰을 pseudo multi-scale hierarchy로 재배치해 멀티스케일 다양성을 주입한다. Scale-Aware Decoding(SAD)은 가벼운 intra-scale refinement와 top-down inter-scale propagation으로 세밀한 경계 복원을 돕는다.

- **Technical Challenges**: 핵심 난제는 DINO 기능이 강하더라도 기본적으로 동일한 패치 그리드 해상도에 묶여 있어, segmentation이 필요로 하는 계층적 스케일 통합을 decoder에 과도하게 떠넘기게 된다는 점이다. SegDINO는 TPA에서 토큰을 2D feature map으로 reshaping하고 1x1 투영 및 strided convolution resizing으로 계층적 공간 해상도를 만들며, SAD에서는 residual refinement와 top-down 전파를 분해해 연산을 최소화한다. 또한 작은 병변에서 중요한 미세 구조를 안정적으로 다루기 위해 TPA의 스케일 다양성이 특히 효과적임을 실험적 분해(ablations)로 확인한다.

- **Empirical Impact**: PanCT(췌장 CT, 284명, 방사선 전문의 주석)와 TN3K, Kvasir-SEG, ISIC 등 3개 공개 벤치마크에서 SegDINO는 일관되게 state-of-the-art 성능을 보이면서도 효율을 유지한다. 특히 PanCT의 small-lesion 세팅에서 최상위 Dice 및 HD95를 기록해 미세 타깃에 강점을 보여준다. 모델은 총 27.68M 파라미터(대부분 DINOv3-S)로 가볍고 51 FPS 수준의 추론 속도를 보고해, 의료 segmentation에서 accuracy–efficiency 균형을 실증했다.



### A Neuro-Symbolic Approach to Strategy Synthesis for Strategic Logics (https://arxiv.org/abs/2606.17962)
- **Prior Approaches**: NatATL 같은 전략 논리는 ATL 계열로 에이전트가 목표를 ‘강제’할 수 있는지 엄밀히 따지지만, 기존 채택은 전략 합성의 계산 비용 때문에 막혀 왔습니다. 특히 NatATL 검증은 허용되는 natural 전략을 명시적으로 생성·열거한 뒤 매번 확인하는 방식이라, 복잡도 bound이나 연합(코얼라이션) 크기가 커지면 전략 공간이 조합폭발을 일으킵니다. LLM이 논리/프로그램을 생성하는 능력은 기대되지만, LLM 단독 생성은 의미적 오류나 취약한 반례 대응 때문에 형식 검증 수준의 신뢰성을 보장하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 LLM을 ‘전략 생성 oracle’로 두고, 표준 NatATL 모델 체커로 형식 검증(certificate)을 수행하는 generate-and-certify 신경-상징(neuro-symbolic) 프레임워크를 제안합니다. LLM이 후보 memoryless natural strategy를 만들면, VITAMIN NatATL verifier가 문법/복잡도 bound/적법 행동(admissible actions)/목표 만족 여부를 판정해 통과한 전략만 채택합니다. 또한 NatATL 전략 합성용으로 4211개의 expert-validated 인스턴스로 구성된 첫 데이터셋 NatATL strategy-synthesis dataset을 공개해, 데이터 기반 평가와 비교의 발판을 마련합니다.

- **Technical Challenges**: 핵심 난제는 조합폭발적인 bounded natural strategy 공간에서 ‘좋은 후보’를 찾아내는 것이며, 동시에 LLM 생성 결과의 신뢰성을 잃지 않는 것입니다. 이를 위해 LLM 출력은 JSON 스키마 검증(에이전트/상태/명제/행동 유효성, default 규칙 포함, guards 포맷 준수) 후에만 verifier로 들어가며, verifier가 실패하면 진단을 다음 프롬프트에 피드백으로 반영하는 제한된 refinement 루프를 구성합니다. 더불어 실패가 예측되는 경우를 줄이기 위해 ATL pre-filter를 전처리로 사용해 불필요한 NatATL 검증 호출을 가지치기하고, one-shot prompting으로 구조화된 출력을 안정화했습니다.

- **Empirical Impact**: Qwen3-32B(오픈 웨이트) 기반 파이프라인 실험에서 certified pipeline은 strategy-synthesis outcome에 대해 92% 정확도를 달성하며, 형식적으로 검증된 양성 결과만 반환한다는 점에서 안전성 기준을 만족합니다. 또한 50 states, 11 agents 코얼라이션, complexity bound k=100k 수준까지 확장하면서 기존의 ‘명시적 자연 전략 열거’ 병목을 완화함을 보였습니다. 무엇보다 NatATL 전용 벤치마크와 verifier-in-the-loop 구조를 함께 제공해, 향후 전략 합성 오라클이나 학습 기반 근사 연구가 재현 가능하게 비교될 수 있는 생태계를 만드는 데 의미가 있습니다.



### Robustness of Similarity-based Positional Encoding Under Rotations: Theoretical Analysis and Experimental Validation (https://arxiv.org/abs/2606.17961)
- **Prior Approaches**: Transformer에선 self-attention이 순열불변이라 위치 정보가 필수이며, 기존 비전에서는 sinusoidal·learned absolute처럼 좌표를 직접 주입하거나 relative/rotary처럼 토큰 간 관계나 벡터 회전을 사용해 왔습니다. Similarity-based positional encoding(simPE)은 좌표계 의존을 줄이고 토큰 표현의 pairwise similarity로 위치 구조를 표현한다는 점에서 기존 계열과 차별됩니다. 다만 simPE의 회전(기하) 섭동에 대한 이론적 거동이 충분히 규명되지 않아, 의료영상처럼 작은 회전이 잦을 때 “왜 안정적인가”가 불명확했습니다.

- **Core Contribution**: 이 논문은 simPE가 일반적으로 rotation-invariant는 아니지만, 회전에 대해 “안정적(stable)”일 수 있음을 이론+실험으로 동시에 보입니다. 특히 simPE를 elementary operator의 조합으로 보고, 각 구성요소에 대한 mild Lipschitz 가정 하에 회전 섭동이 positional encoding에 얼마나 영향을 주는지 정량적 상계를 도출합니다. 또한 작은 각도 영역에서 각도 크기에 선형으로 응답이 제한됨을 명시적 bound로 제공합니다.

- **Technical Challenges**: 핵심 난점은 simPE가 좌표 기반이 아니라 similarity 연산을 통해 구성되기 때문에, 회전에서 exact 불변성을 기대할 수 없다는 점입니다. 논문은 대신 Lipschitz 연속성을 통해 구성요소별 변화량이 전체 출력 변화로 어떻게 전파되는지 추적하며, 특히 normalization 같은 비정규 항은 원점에서 Lipschitz가 깨질 수 있음을 제외(domain에서 0 벡터를 피함)하는 방식으로 안정성 조건을 정리합니다. 그 결과 Frobenius norm 기준의 전역 추정치와 small-angle bound를 함께 제시합니다.

- **Empirical Impact**: 실험은 학습·검증 이미지는 고정된 canonical orientation으로 두고, 테스트만 점진적으로 회전시켜 “증강효과 없이” encoding의 내재적 안정성을 측정하도록 설계됐습니다. 네 가지 통제 데이터셋(Arrow, Shapes, Digits, FashionMNIST)에서 simPE는 회전 각이 커질수록 accuracy뿐 아니라 F1, precision, recall에서도 learned absolute positional embedding보다 일관되게 우수한 성능을 보였습니다. 특히 작은~중간 각도 구간에서 격차가 크게 나타나, 제시된 stability guarantee가 실제 성능 저하를 완화한다는 점을 경험적으로 뒷받침합니다.



### SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs (https://arxiv.org/abs/2606.17952)
Comments:
          Accepted at ICML 2026

- **Prior Approaches**: 기존 sparse MoE는 hard top-k 라우팅으로 토큰당 상위 k개 expert만 활성화해 추론 비용을 줄입니다. 하지만 top-k 선택이 이산적이라 비미분이라서, 활성 expert 수가 입력마다 고정되고(사전에 결정), 레이어 간/토큰 간 compute를 유연하게 배분하기 어렵습니다.
또한 라우팅 불균형과 학습 불안정이 스케일에서 더 두드러져, routing-optimization과 inference-time 선택의 불일치가 문제로 지적돼 왔습니다.

- **Core Contribution**: SoftMoE는 discrete top-k 대신 LapSum 기반 truncated soft top-k relaxation으로 라우팅을 미분 가능하게 바꿉니다. 그 결과 expert 라우팅과 compute 사용량을 학습 과정에서 gradient-based로 조정할 수 있습니다.
더 나아가 레이어별 평균 활성 expert 수를 파라미터화하고, 네트워크 전역에 걸친 global budget 제약을 걸어 레이어 간 expert capacity 재배분을 학습시키는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 난제는 (1) 미분 가능하면서도 MoE의 희소성(낮은 활성 expert 수)을 유지하고, (2) autoregressive 모델링의 인과성을 깨지 않으며, (3) 학습에서 배운 라우팅/버짓을 추론 계산량으로 실제로 반영하는 것입니다.
SoftMoE는 LapSum의 연속적 soft top-k로 gradient 흐름을 확보하되, 효율을 위해 저기여 가중치를 truncated 방식으로 잘라내 dense 계산을 막습니다. 또한 레이어별 활성 expert 평균 k_l을 global budget K 아래 unconstrained 파라미터로 최적화해 레이어 간 경쟁적 배분이 일어나도록 설계했습니다.

- **Empirical Impact**: C4와 OpenWebText로 언어모델링 및 downstream(PIQA, HellaSwag, ARC-E) 평가를 수행한 결과, SoftMoE는 sparse MoE와 동등하거나 더 나은 성능을 유지하면서 평균적으로 더 적은 expert만 활성화합니다. 특히 early/middle 레이어에서 compute가 줄고, 후반 레이어가 더 많은 expert를 쓰는 비균일(non-uniform) 할당 패턴이 관찰됩니다.
또한 학습 과정에서 활성 expert 수가 고정되는 기존 sparse MoE와 달리, SoftMoE는 학습 중 입력/토큰에 따라 활성 expert 수를 적절히 조정하면서도 학습 불안정 없이 안정적으로 수렴했다는 점이 의미가 있습니다.



### Plug-and-Adapt: Multimodal Coreference Resolution at First Sight with a Pretrained Alignment Mod (https://arxiv.org/abs/2606.17950)
- **Prior Approaches**: 기존 MCR(multi-modal coreference resolution) 연구는 CIN(Coreference Image Narratives) 같은 벤치마크의 희소한 코어퍼런스 체인 주석이나 mouse tracking 같은 보조정보에 크게 의존해 학습·평가해 왔습니다. 약지도/준지도나 diffusion 기반 데이터 증강이 일부 문제를 완화했지만, 여전히 특정 데이터셋 학습이 필요해 즉시 적용성과 일반화가 제한된다는 한계가 남아 있습니다. 또 zero-shot을 노리는 VLM/CLIP 계열은 비전-언어 정렬은 잘하지만, 코어퍼런스의 문맥 의존성과 다중 멘션 간 유사도 추론까지는 충분히 못 따라가 성능 격차가 컸습니다.

- **Core Contribution**: 이 논문은 PA-MCR(plug-and-adapt for MCR)이라는 플러그-앤-어댑트 방식으로, (라벨이 부족한) 타깃 코어퍼런스 데이터로의 fine-tuning 없이도 이미지 내러레이션의 코어퍼런스를 해결하는 접근을 제안합니다. 핵심은 CLIP 기반 alignment 모델을 관계(relation) 단위 정렬로 재학습한 뒤, 멘션 표현을 “유사도 점수의 aggregation”으로 구성하고, 시각 cue와 카테고리 cue를 evidence theory로 결합해 추론을 안정화하는 것입니다. 즉, grounding에서 MCR로의 “격차”를 (1) relation-aware 정렬과 (2) 불확실성을 반영한 멘션 표현/멀티큐 통합으로 메우는 설계입니다.

- **Technical Challenges**: 첫째, grounding은 멘션-영역의 일대일 대응에 가깝지만 MCR은 프라임(예: pronoun)처럼 주변 관계에 좌우되는 해석과 멘션 간 상호 유사도 추론이 필요해, 단순 pairwise 정렬만으로는 부족합니다. 논문은 이를 위해 relation triplets 기반으로 정렬을 사전학습하고, top-aligned 편향을 줄이기 위해 모든 관련 매칭 점수를 활용하는 aggregation 기반 멘션 임베딩을 구성합니다. 둘째, 정렬 점수에 내재된 보정되지 않은 불확실성과 대량 후보(region)가 누적되며 노이즈가 증폭될 수 있어, region 외에 카테고리 정보를 추가하고 evidence theory로 cue 신뢰도를 정규화·융합해 “의심 스코어를 억제/신뢰 스코어를 강조”하도록 했습니다.

- **Empirical Impact**: CIN 벤치마크에서 PA-MCR은 전용(dedicated) SOTA 대비 CoNLL F1을 5.31%p, 인기 VLLM 대비 2.12%p 개선하며, 데이터셋 의존적 학습 없이도 경쟁력을 보였습니다. 또한 masked CIN과 새로 구성한 VCR-MCR에서 강건성 및 일반화 능력이 확인되어, 정렬 기반 접근이 특정 주석 구성에 덜 묶이는 잠재력을 시사합니다. 배포 관점에서도 거대 VLLM에 비해 훨씬 접근 가능한 파이프라인을 제시한다는 점에서 multimodal coreference의 실용화 방향에 의미가 큽니다.



### KANLib -- An Modular, Extensible and Fast Kolmogorov-Arnold Network Implementation (https://arxiv.org/abs/2606.17927)
- **Prior Approaches**: KANs는 MLP의 고정 활성함수를 제거하고 간선마다 learnable univariate 함수로 비선형을 내장해 해석성과 표현력을 높이려는 시도로 주목받았지만, 실제 구현 연구는 프레임워크별 기능 차이와 계산 비용 때문에 확산이 더뎠습니다. 특히 PyKAN은 B-spline 기반으로 풍부한 기능(시각화, pruning, symbolic regression, grid extension/리샘플링)을 제공하는 대신 B-spline 평가 오버헤드가 커 느리다는 비판을 받았습니다. EfficientKAN과 FastKAN은 계산 효율을 개선했지만, 기능 완전성(예: 일부 규격의 regularization 제약, adaptive grid rescaling 부재 등)이나 basis 근사 조건(예: Gaussian RBF의 비등간 grid 한계) 같은 제약이 남아 벤치마킹과 확장 연구가 번거로웠습니다.

- **Core Contribution**: 이 논문은 KAN 연구를 위한 모듈형·확장형·고성능 프레임워크 KANLib를 제안합니다. KANLib는 PyKAN, EfficientKAN, FastKAN의 핵심 아이디어를 하나의 일관된 소프트웨어 아키텍처로 통합해 basis 함수(B-spline/GRBF)별로도 feature parity와 호환성을 맞추고, grid extension, adaptive grid rescaling, fine-grained architectural customization 같은 실험 제어를 표준화합니다. 이를 통해 연구자가 동일 조건에서 KAN 변형들을 직접 비교하고 새로운 설계를 빠르게 탐색할 수 있게 합니다.

- **Technical Challenges**: KANLib를 실사용 가능한 연구 도구로 만들기 위해서는 (1) basis별 연산 비용을 줄이면서도 (2) grid 조작 및 정교한 레이어 구성 요소를 서로 다른 구현 간에 동일한 의미로 제공하고 (3) PyTorch end-to-end 워크플로우와 잘 맞물리게 해야 합니다. 논문은 PyKAN의 spline+residual 구조와 grid extension/adaptive grid rescaling을 중심축으로 삼되, EfficientKAN의 B-spline 평가 및 선형층 계산 최적화 방식을 흡수해 성능 병목을 완화합니다. 또한 FastKAN의 Gaussian RBF 가속 장점을 도입하되, GRBF에 대한 adaptive rescaling은 equidistant grid에서만 유효하도록 제한해 근사 실패 조건을 회피합니다.

- **Empirical Impact**: California Housing 회귀 벤치마크에서 KANLib는 기존 레퍼런스 구현들과 거의 동일한 예측 거동을 재현하며, B-spline 기반의 경우 RMSE 0.5376±0.0044, R2 0.7852±0.0035로 PyKAN/EfficientKAN 대비 근소한 차이를 보입니다. GRBF 기반에서도 RMSE 0.5471±0.0078, R2 0.7776±0.0063으로 FastKAN과 유사한 성능 범위를 유지합니다. 계산 효율 면에서는 PyKAN 대비 inference time을 B-spline에서 약 32.7%, GRBF에서 약 62.0% 줄였고, 더 단순한 레이어 구성(예: residual branch 제거 등)에서는 최대 약 10~16.8% 추가 속도 개선 가능성을 보여 KAN 아키텍처 탐색의 실용적 기반을 제공한다는 점에서 의미가 있습니다.



### PearlVLA: Progressive Embodied Action-Plan Refinement in Latent Spac (https://arxiv.org/abs/2606.17924)
Comments:
          21 pages, 2 figures. Preprint

- **Prior Approaches**: 기존 VLA는 비전-언어 표현에서 바로 행동을 디코딩하거나, 텍스트 추론 trace·시각 서브목표·action search·world model의 후보 탐색처럼 명시적 중간 단서를 거쳐 계획을 개선해 왔습니다. 하지만 텍스트/픽셀 기반 계획은 지연과 계산비용이 크고, 후보를 많이 평가하는 WM 롤아웃은 스케일 문제가 생깁니다. 반면 end-to-end 성격의 direct decoding은 빠르지만 단발(single-pass)이라 현재 계획이 만든 미래를 미리 보고 self-correct할 여지가 제한적입니다.

- **Core Contribution**: PearlVLA는 VLA의 deliberation(계획 숙고)을 텍스트·픽셀·action search 대신 VLM이 만든 latent 공간 내부로 옮깁니다. VLM의 meta-query를 고정 visual grounding 분기와 반복형 latent plan 분기로 분리하고, 매 라운드마다 plan-conditioned world query로 action-free 미래 관측 latent를 얻어 plan 토큰을 residual로 점진 정제합니다. 최종 refined latent plan은 병렬로 action chunk로 디코딩되어, 숙고의 계산은 늘리되 실행은 저지연 경로를 유지하는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “latent 숙고가 현재 장면에 접지(grounding)되면서, 현재 plan이 유도하는 미래 피드백이 제대로 반영”되도록 폐루프를 설계하는 것입니다. 논문은 world query의 condition drift를 줄이기 위해 anchored world query를 두고, residual write-back을 scheduled로 감소시키는 coarse-to-fine 업데이트를 적용합니다. 또한 중간 라운드별 편집이 미래 성공에 주는 인과효과를 가치함수 없이 학습하기 위해, Causal Refinement-Grouped Process-Reward RL(CRG-PRL)을 제안해 같은 상태 내에서 그룹 상대 보상으로 residual 편집을 비교 최적화합니다.

- **Empirical Impact**: LIBERO 벤치마크에서 PearlVLA는 supervised만으로도 평균 성공률을 97.1%→98.5%로 끌어올렸고, CRG-PRL까지 적용하면 98.7%로 SOTA를 달성했습니다. 또한 refinement round 수에 따른 성능 향상과, K=0(정제 제거) 대비 K=4에서의 격차 확대로 latent refinement의 기여를 확인했습니다. LIBERO-Plus에서도 OpenVLA-OFT 대비 69.6%→76.3%로 강한 강건성 향상을 보여, 지연을 늘리지 않는 latent 기반 계획 정제가 장기 일반화에도 유효함을 시사합니다.



### Trustworthy Self-Composable Big-Data-as-a-Service: An LLM-Orchestrated Multi-Agent Framework for Automated Data Engineering, AutoML, MLOps Deployment, and Drift-Aware Lifecycle Optimization (https://arxiv.org/abs/2606.17915)
Comments:
          7 pages, 3 figures, 5 tables

- **Prior Approaches**: 기존 AutoML-Agent, Data Interpreter, DS-Agent 등 LLM 에이전트 연구는 전반적인 파이프라인을 다루더라도, 대체로 실험/개발 단계 중심으로 설계돼 BDaaS의 전 생애주기 오케스트레이션이 약합니다. 또한 데이터 클리닝·피처 엔지니어링·시각화 같은 구성요소는 다뤄도, 배포 준비물(artifact) 거버넌스·reproducibility·휴먼 오버사이트·drift 대응을 한데 묶는 형태는 제한적입니다. MLOps 측면에서도 배포·모니터링의 중요성은 강조되지만, LLM 에이전트와 결합해 폐루프에 가깝게 운영되는 통합 프레임워크는 부족했습니다.

- **Core Contribution**: 이 논문은 LLM-orchestrated multi-agent 협업으로 BDaaS 생애주기를 신뢰가능하게 자동화하는 self-composable 프레임워크를 제안합니다. 수집·정제·피처 엔지니어링·AutoML 학습·평가·MLOps 배포 준비·모니터링·drift 탐지를 각각 전담 에이전트로 분해하고, 중앙 LLM 오케스트레이터가 실행 순서와 중간 결과 검증, 동적 워크플로 조합을 담당합니다. 여기에 artifact governance, reproducibility 지원, human-in-the-loop 체크포인트, drift-aware feedback loop를 함께 내장해 “모델 정확도”를 넘어 “운영 신뢰성”을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 파이프라인의 단계들이 데이터 스키마·전처리·배포 구성으로 강하게 연결돼 있어, 한 단계의 변화가 전체 신뢰성을 깨뜨릴 수 있다는 점입니다. 이를 위해 중앙 오케스트레이터가 에이전트가 만든 구조화된 중간 산출물을 다음 단계 요구조건에 맞게 검증하고, 위험한 결정(정제/특징 제거, 모델 승인, 배포 릴리스, 드리프트 대응)은 Human Oversight Agent로 라우팅하는 체크포인트를 둡니다. 또 공유 아티팩트 저장소로 데이터/메타데이터/모델/배포물의 버전과 추적성을 남겨 재현성과 롤백 가능성을 확보합니다.

- **Empirical Impact**: 통제된 tabular 벤치마크(결측, 범주형, outlier, class imbalance, covariate drift 시뮬레이션)에서 제안 파이프라인은 classification 평균 F1 0.662로 single-agent LLM(0.652), AutoML-only(0.644), manual ML(0.563)보다 우수한 성능을 보였습니다. 회귀 작업에서도 RMSE가 3.279에서 2.809로 약 14.3% 개선돼, 라이프사이클 지향 전처리/피처 구성의 이점이 드러났습니다. 무엇보다 workflow completion, artifact traceability, deployment readiness가 100%로 보고되고, 드리프트 후 F1을 0.495에서 0.667로 회복하는 등 drift-aware 모니터링-재학습-재배포(또는 재조정) 폐루프의 실효성이 확인됐습니다.



### Non-negative Elastic Net Decoding for Information Retrieva (https://arxiv.org/abs/2606.17910)
Comments:
          19 pages, 4 figures

- **Prior Approaches**: 기존 dense retrieval은 쿼리-문서 임베딩을 내적(inner product)으로 독립 스코어링해 top-k를 뽑는 방식이어서, 코퍼스 전체의 문서 간 상관관계를 반영하지 못합니다. 그 결과 서로 비슷한(중복/유사) 문서가 함께 선택되어 비다양하고 정보가 겹치는 retrieved set이 생기기 쉽습니다. 또한 cross-encoder나 generative retrieval처럼 문서들을 함께 평가하는 방법은 품질은 높을 수 있지만 지연시간 제약 때문에 그대로 쓰기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 retrieval을 “문서들을 집합으로 함께 디코딩”하는 joint decoding 문제로 재정의합니다. 제안하는 Non-Negative elastic Net (NNN) decoding은 쿼리 임베딩을 코퍼스 문서 임베딩의 희소(sparse)한 비음수 선형결합으로 복원하도록 하여, 관련 문서들끼리는 남기고 중복 문서는 자연스럽게 억제하도록 설계됩니다. 더 나아가 NNN decoding의 이론적 표현력이 dense retrieval보다 크다는 점을 정식으로 증명하고, scoring 함수를 단순 교체하는 drop-in replacement 관점도 제시합니다.

- **Technical Challenges**: NNN decoding은 각 문서 포함 여부가 코퍼스 전체를 고려한 복원 오차에 의해 결정되므로, 실사용에서는 (λ1, λ2) 같은 하이퍼파라미터를 어떻게 정해야 하는지가 관건입니다. 이 논문은 FISTA 기반의 non-negative elastic net 최적화를 사용해 추론 비용을 O(dNT)로 유지하면서, unrolling된 solver를 역전파에 연결해 end-to-end fine-tuning도 가능하게 했습니다. 또한 이론은 쿼리마다 다른 (λ1, λ2) 존재성을 말하지만, 실험에서는 고정된 한 쌍을 validation으로 맞추는 더 단순한 배치 전략이 여전히 이득을 준다는 점을 보여줍니다.

- **Empirical Impact**: 실험에서는 먼저 frozen embeddings(내적 스코어링용으로 학습된 임베딩) 위에 NNN decoding만 적용해도 여러 벤치마크에서 성능이 일관되게 개선되는 것을 확인합니다. Tool retrieval과 multi-hop retrieval에서 특히 completeness 지표에서 최대 36% 향상이 보고되며, 이는 관련-비관련 문서 상관이 큰 near-duplicate 상황에서 이론 예측과 맞물려 더 큰 격차로 나타납니다. 나아가 unrolled FISTA를 통한 end-to-end 학습을 수행하면 모든 지표/벤치마크에서 dense retrieval을 상회하는 유의미한 성능 향상을 달성하며, dense 임베딩을 inner-product 스코어링을 넘어 활용하는 새로운 패러다임을 제시합니다.



### Dimensionality Controls When Modularity Helps in Continual Learning (https://arxiv.org/abs/2606.17889)
Comments:
          Accepted to the 2nd Workshop on Compositional Learning (CompLearn) at ICML 2026, Seoul, South Korea. 8 pages, 5 figures

- **Prior Approaches**: 연속학습은 stability–plasticity dilemma로 인해 새 과제를 학습하는 동시에 이전 표현을 보존해야 하는 문제로 정리돼 왔습니다. 기존 해법은 replay/regularization 같은 학습 제약과, 모듈화(architectural modular separation) 같은 구조적 분리에 초점을 둡니다. 다만 과제 유사도에 따라 공유가 전이로 이어질 수도, 간섭으로 이어질 수도 있어 “모듈화가 항상 이득인가?”가 불명확했습니다.

- **Core Contribution**: 이 논문은 모듈형 recurrent 네트워크가 sequential A-B-A에서 간섭을 줄이는 조건을 “표현의 dimensionality(차원) 관점”에서 규명합니다. 특히 초기 가중치 스케일 γ가 유도하는 lazy(고차원) vs rich(저차원) 학습 레짐에 따라, 모듈화의 기능적 이점이 켜졌다가 꺼지는 것을 보여줍니다. 즉 안전과 robustness를 고정된 공유/분리 문제가 아니라, 과제에 맞춘 representational subspace 할당의 적응 문제로 재해석합니다.

- **Technical Challenges**: 기여를 입증하려면 모듈화 효과를 다른 요인과 분리해 비교해야 하는데, 이를 위해 동일한 A-B-A 전이-간섭(transfer–interference) 패러다임에서 단일 네트워크와 task-partitioned 모듈 네트워크를 직접 비교합니다. 또한 γ로 레짐을 체계적으로 바꾼 뒤 PCA 기반 effective dimensionality와 principal angles로 내부 표현 기하(geometry)를 정량화합니다. 이를 통해 “행동 성능 차이”가 “표현 공간 차원/서브스페이스 배치 변화”와 동시 발생하는지 연결했습니다.

- **Empirical Impact**: 실험 결과, 고차원 lazy 레짐에서는 두 아키텍처가 유사한 성능과 내부 기하를 보여 모듈화 이점이 약합니다. 반대로 저차원 rich 레짐에서는 모듈 네트워크가 유사 과제일수록 겹치고(정렬), 중간 유사도에서는 부분적으로 분리되며, dissimilar 과제에서는 더 직교에 가깝게 배치되는 graded한 구조를 형성합니다. 이때 단일 네트워크는 과제 서브스페이스가 더 얽혀 interference가 커져, “모듈화가 유효해지는 대표 레짐”을 실증적으로 제시했다는 점에서 연속학습 연구에 의미가 큽니다.



### AI Adoption Across a Multinational Workforce: Sociotechnical Conditions for GenAI Acceptance in Human Resources (https://arxiv.org/abs/2606.17887)
- **Prior Approaches**: 기존 연구는 AI 채택을 TOE, TAM, UTAUT 같은 프레임으로 설명해 왔지만, GenAI는 확률적·불투명하고 맥락 의존적이라 기존 모델의 단순 예측을 그대로 적용하기 어렵습니다. 또한 직장 내 AI의 불균등 효과에 대한 논의가 늘었지만, 실제 전환 과정에서 사용자가 일상적으로 어떻게 배우고 우회하며 선택적으로 쓰는지까지는 충분히 다루지 못했습니다.

- **Core Contribution**: 이 논문은 다국적 테크 기업에서 레거시 HR 검색 시스템(Steve)을 GenAI On Search가 내장된 시스템(People Tool)로 전환하는 ‘실제 전환 중’ 상황을 단일 사례로 분석해, 누가 채택하고 누가 배제되는지의 동학을 실증적으로 보여줍니다. 특히 adoption이 단순 마이그레이션이 아니라, 두 시스템을 병행하며 상황에 맞게 경로를 고르는 선택적 사용(selective use)으로 나타난다는 점을 핵심으로 제시합니다.

- **Technical Challenges**: GenAI 기반 검색은 사용자가 프롬프트·질의 구성뿐 아니라 출력 평가, 출처 확인, 필요 시 다른 경로로 되돌아가는 ‘새로운 검색 논리’를 익혀야 제대로 활용됩니다. 이에 더해 신뢰(trust)는 단지 정답률로 결정되지 않고, 소스체킹·시스템 간 비교·동료나 HR 입력 탐색 같은 상호작용을 통해 보정(trust calibration)되는 방식으로 형성된다고 분석합니다.

- **Empirical Impact**: 검색 로그, 설문(25명), 반구조화 인터뷰(10명) 결과는 People Tool의 채택이 역할·언어·근속 등 ‘상황 적합도(situational fit)’에 강하게 좌우되며, 결과적으로 집계된 사용량이 포괄성을 과대평가할 수 있음을 시사합니다. 조직은 AI를 ‘AI 인프라’이자 지식 인프라의 일부로 다루고, 다양한 사회 집단에 대해 어떤 맥락에서 가치가 생기는지까지 설계·가이드(교육/문서 품질/안내)를 포함해 책임성과 사용성을 높여야 한다고 제안합니다.



### AnchorKV: Safety-Aware KV Cache Compression via Soft Penalty with a Refusal Anchor (https://arxiv.org/abs/2606.17872)
- **Prior Approaches**: 롱컨텍스트 LLM의 추론 비용은 파라미터 크기보다 KV cache가 좌우하며, 이를 줄이기 위해 attention-relevant 토큰 일부만 남기는 KV cache compression이 발전해 왔다. 다만 기존 방법들은 정확도(utility) 중심이라 안전 정렬(safety alignment)과의 상호작용이 충분히 검토되지 않았고, 실제로 SnapKV 같은 압축은 jailbreak 공격 ASR을 거의 낮추지 못한다는 보고가 있다. 또한 공격에 취약한 방식의 토큰 유지/회피 정책은 공격을 막지 못할 뿐 아니라 오히려 안전 관련 응답 분포를 왜곡할 수 있다.

- **Core Contribution**: AnchorKV는 FastKV 같은 기존 KV cache 압축기에 drop-in으로 추가 가능한 “안전 우선” 수정으로, 토큰 유지 점수(retention score)에 해로운(harmful) 프롬프트와 연관된 key space 방향을 더 크게 불리하게 만드는 소프트 패널티를 도입한다. 차단 신호는 attention rank가 아니라 representation engineering에서 온 선형 ‘refusal(거절) 방향’(Anchor)으로 구성해, 압축 정책이 공격자에게 쉽게 조작되는 문제를 피하려고 한다. 패널티 강도 λ가 0이면 원래 compressor로 정확히 환원되도록 설계되어, 성능-안전 트레이드오프를 체계적으로 조절할 수 있다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “안전 신호가 압축이 실제로 일어나는 기하(operational point)와 같은 공간에 있어야 한다”는 점인데, 기존 refusal 방향 연구는 residual stream에서의 방향을 주로 다뤄 KV 압축의 key projection과 기하 불일치가 생길 수 있다. AnchorKV는 특정 eviction 레이어 ℓ*의 per-head key projection 공간에 대해 difference-of-means 기반 차분 평균 절차를 적용해 offline safety anchor를 만들고, inference 중에는 각 토큰의 anchor projection으로 harm 점수를 계산해 top-k 선택 직전에 soft penalty로 반영한다. 동시에 attention-sink 붕괴나 prompt 내용 백본(cannibalization)을 막기 위해 sink protection과 importance immunity 같은 구조적 예외를 두며, λ=0에서 bit-exact로 baseline과 동일해지도록 연속성을 보장한다.

- **Empirical Impact**: Llama-3.1-8B-Instruct를 대상으로 FastKV 대비 AnchorKV는 jailbreak 평가(AdvBench의 AdvPrompter 공격)에서 안전 정렬을 더 잘 유지하는 방향으로 ASR을 개선한다. 또한 long-context 벤치마크 LongBench에서 utility 손실을 최소화하면서 안전성을 끌어올리는 trade-off frontier를 제시하며, 단순히 패널티를 강하게 늘린다고 안전이 단조롭게 좋아지지 않는(비단조 reversal) 구간도 관찰한다. 이 결과는 “공격에 강한 KV 압축”이 단순 방어가 아니라, 압축 정책 자체의 기하/기능 설계로 안전 정렬을 재보정해야 한다는 실무적 시사점을 준다.



### A Quantitative Analysis of Multimodal Biomarkers in Alzheimer's Diseas (https://arxiv.org/abs/2606.17867)
Comments:
          Accepted to ICTS4eHealth 2026

- **Prior Approaches**: 기존 AD 멀티모달 연구는 예측 성능을 높이기 위해 복잡한 black-box 모델을 강화하는 경우가 많았고, 서로 다른 바이오마커가 얼마나 중복 정보를 갖는지 정량적으로 비교하는 작업은 상대적으로 부족했습니다. 일부 상호작용 분석이 있었지만, 모달리티 간 관계를 ‘인과 경로/생물학적 대응(구조-병리 연결)’ 관점에서 구조화해 해석하거나, 시점이 어긋난 진행 순서를 모델링하는 데는 한계가 있었습니다. 또한 PET 같은 고비용 측정의 중복성을 줄이기 위한 ‘최소·최대 정보’ 선택 기준이 명확하지 않았습니다.

- **Core Contribution**: 이 논문은 tau-PET, 구조 MRI, 인지 점수(MMSE, CDR), 유전(APOE ε4) 데이터를 ADNI의 789명에서 통합하고, 모달리티 간 정보 중복과 예측 의존성을 정량 분석해 바이오마커 선택의 근거를 제공합니다. tau와 인지의 연관을 atrophy(위축) 기반/비기반 성분으로 분해하고, SuStAIn 기반 궤적 추정으로 분자 병리→구조 변화→인지 저하의 지배적 진행 패턴을 재구성합니다. 이를 통해 “무엇을 더 측정해야 하는가/무엇을 줄일 수 있는가”를 데이터 관계로 설명하는 해석형 프레임을 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 모달리티 간 관계가 단순 선형이 아니고, 다중공선성과 비선형 의존이 함께 얽혀 있을 수 있다는 점이었습니다. 저자들은 NMI(비모수 nearest-neighbor)로 상호정보를 중복도(0~100%)로 비교하고, cross-validated Ridge regression으로 모달리티 간 explained variance를 안정적으로 추정해 예측 방향성을 평가했습니다. 또 tau-atrophy는 ROI별 상관과 PLS-SVD로 다변량 공분산 구조를 포착했고, tau-인지 매개효과는 ACME/ADE로 위축 경로 비중을 분해한 뒤, cross-sectional 한계를 보완하기 위해 규범모델링 기반 z-표준화와 modality-specific abnormality threshold로 SuStAIn 궤적을 구성했습니다.

- **Empirical Impact**: 결과적으로 APOE ε4는 다른 모달리티와 공유 정보가 매우 낮아(대략 0~6.9%) 상대적으로 독립적·보완적임이 확인됐고, 반대로 CDR-SB와 CDR-GLOBAL은 83% 이상 중복되어 다중 글로벌 인지척도 투입의 불필요성을 시사합니다. tau-PET과 MRI는 인지 예측에서 의미 있는 비중을 보였으며(예: MRI→CDR-SB, tau-PET→MMSE), tau-인지 연관 중 약 28%가 구조 위축을 통한 간접 경로(ACME)로 설명되었습니다. 나머지 직접 성분(ADE)도 유의해, 위축 외의 추가 메커니즘 가능성을 남기면서도 “진행 지연(cascade)” 관점에서 분자 병리가 먼저 나타나고 구조 변화가 뒤따르는 지배적 궤적을 데이터가 지지합니다.



### High-Fidelity 3D Geometric Reconstruction of Pelvic Organs from MRI: A Hybrid Deep Learning and Iterative Optimization Approach (https://arxiv.org/abs/2606.17836)
- **Prior Approaches**: 기존 연구는 MRI에서 골반 장기 3D를 얻을 때 주로 이미지 segmentation에 치우치거나, 생성된 3D 모델을 downstream 분석에 쓰는 쪽에 집중해 왔다. 그 결과 고충실도(고품질) geometry 복원이 여전히 labor-intensive하고 표준화도 부족했다. 또한 딥러닝 예측만으로는 국소 표면/메시 품질을 충분히 다듬기 어렵다는 한계가 반복됐다.

- **Core Contribution**: 이 논문은 bladder, uterus, rectum을 대상으로 하이브리드 deformable shape modeling을 제안해 reconstruction 품질의 격차를 줄인다. 핵심은 geometry-aware multi-level 딥러닝으로 topological consistency를 보존하고, 학습·추론 모두에서 iterative optimization으로 국소 표면과 메시에 대한 품질을 정교하게 끌어올리는 holistic synergy 구조다. 학습 단계에서는 최적화가 딥러닝의 감독 역할을 하고, 추론 단계에서는 딥러닝이 전역 형태를 빠르게 예측한 뒤 최적화가 표면을 refine한다.

- **Technical Challenges**: 주요 challenge는 전역 형태를 잘 맞추면서도 국소 표면 디테일과 topological consistency를 동시에 유지하는 것이다. 이를 위해 두 단계 amortized optimization training으로 global shape capture와 local surface refinement의 균형을 맞추고, geometry-aware multi-level 아키텍처로 장기 간 구조 일관성을 보존한다. 또한 iterative optimization이 학습에는 supervision 신호로, 추론에는 refinement 단계로 기여하도록 학습/추론 파이프라인을 함께 설계했다.

- **Empirical Impact**: 실험에서 제안 프레임워크는 기존 mainstream 딥러닝 기반 장기 reconstruction 대비 geometric fidelity가 뚜렷하게 우수했다. 구조별로 bladder, rectum, uterus의 3D는 Chamfer Distance가 더 낮고 Dice Similarity Coefficient이 더 높게 나타났다. 계산 효율을 유지하면서도 volumetric mesh quality가 더 좋았고, 환자 단위 평가에서는 minSICN과 minSIGE의 ‘10 worst elements’ 지표가 전통적 geometric post-processing 알고리즘보다 개선됐다.



### Perceptual compensation for tonal context in self-supervised speech models (https://arxiv.org/abs/2606.17835)
Comments:
          Accepted for publication at Interspeech 2026

- **Prior Approaches**: 기존 연구들은 wav2vec2.0 같은 self-supervised learning(SSL) 음성 모델이 사전학습만으로도 음운·음소 범주에 대한 구조를 암묵적으로 학습하는지에 주목해 왔다. 특히 pre-trained 모델에서도 phonological structure 민감성이 나타난다고 보고되어, 이를 phonetic-only 관점이나 맥락 예측만으로 PC(Perceptual compensation)가 형성될 수 있다는 해석으로 연결해 왔다. 다만 대부분의 증거가 다른 음운 단서(예: r/l 같은 segment 대비)나 특정 평가 세팅에 의존해, 실제로 인간 청자의 PC와 같은 형태인지에 대한 검증이 제한적이었다.

- **Core Contribution**: 이 논문은 만다린 중국어 성조(lexical tone)에서 PC가 나타나는 정도를 wav2vec2.0의 pre-trained(PT) vs fine-tuned(FT) 표현으로 비교하는 pseudo-replication을 수행한다. 핵심은 인간의 성조 지각에서 관찰되는 ‘고정된 기준선(no-ctx) 대비 맥락에 따른 범주 경계 이동’이 모델 표현에서도 재현되는지 검사하는 것이다. 이를 통해 “unsupervised contextualization은 되는데, human-like compensation은 안 될 수 있다”는 관점 차이를 정량적으로 제약한다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 인간 실험을 모델 표현으로 옮기면서도, 맥락 효과와 기준선 편향을 혼동하지 않는 것이다. 연구진은 14-step 성조 연속체를 만들고, 여러 화자의 disyllable 자극을 재합성해 determinism 문제를 완화한 뒤 embedding similarity와 probing classifier(선형/MLP 로지스틱 기반) 두 방법으로 PT·FT·레이어별 반응을 측정했다. 그 결과 PT에서는 어떤 레이어에서도 성조 맥락에 대한 compensation 조짐이 거의 없었고, FT에서는 맥락 민감성은 있으나 no-context 기준선에 ‘상대적으로’ 이동하는 인간형 패턴이 약하거나 관찰되지 않았다.

- **Empirical Impact**: 실험은 embedding similarity에서 PT가 맥락 민감성이 거의 없음을 보여주며, probing에서는 FT와 일부 레이어에서만 약한 PC 징후가 나타나지만 인간 성조 지각 곡선과는 단절된 결과를 낸다. 특히 isolated syllable 조건에서 FT의 T3/T4 분류는 인간과 크게 다르게 나타나, ‘성조 범주가 더 잘 드러난다’는 신호와 ‘인간과 같은 지각적 보정이 된다’는 신호가 분리될 수 있음을 시사한다. 전체적으로, 성조 영역에서는 unsupervised 사전학습만으로는 human-like PC가 자동으로 성립하기 어렵고, supervised fine-tuning 등 추가 학습 기제가 안정적인 음운 범주 추상화를 유도해야 한다는 결론에 힘을 실어 준다.



### Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity (https://arxiv.org/abs/2606.17830)
Comments:
          Published at the International Conference on Machine Learning (ICML 2026)

- **Prior Approaches**: 기존 연구는 신경망 파라미터 공간의 non-injectivity(다른 파라미터가 같은 함수를 낼 수 있음)를 주로 fully connected·convolutional 모델 중심으로 정리해 왔습니다. attention 기반 모델(특히 multihead attention)은 비교적 vanilla 형태에 초점이 맞춰졌고, positional encoding이 만들어내는 근본적 대칭 구조 변화는 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 Transformers에서 positional encodings를 포함했을 때의 functional equivalence를 형식적으로 분석합니다. 특히 sinusoidal과 rotary positional encodings(RoPE) 두 변형에 대해, sinusoidal은 vanilla attention의 equivalence 구조를 상당 부분 보존하지만 RoPE는 symmetry group을 크게 줄여 expressivity를 높인다고 보입니다.

- **Technical Challenges**: 핵심 난제는 attention의 대칭/동치가 positional encoding까지 포함했을 때 어떻게 재구성되는지 엄밀한 수학적 구조로 추적하는 것입니다. 저자들은 두 positional encoding의 차이를 기준으로 equivalence 구조와 symmetry group의 변화를 분석하고, 추가로 linear mode connectivity의 거동이 positional encoding 유무·가변성에 따라 달라짐을 확인하기 위해 alignment algorithm으로 실증 검증을 수행합니다.

- **Empirical Impact**: 실험 결과는 positional encoding의 존재와 설정 변동이 Transformer들 사이의 linear mode connectivity “가능성/형태”에 결정적으로 영향을 준다는 점을 보여줍니다. 이러한 분석은 RoPE가 실무에서 점점 더 널리 채택되는 현상을, 단순 경험적 성능이 아니라 symmetry(대칭) 감소와 표현력 확대라는 관점에서 설명해 주는 데 의미가 있습니다.



### When Multiple Scripts Matter: Evaluating ASR in Clinical Settings (https://arxiv.org/abs/2606.17826)
Comments:
          Interspeech 2026

- **Prior Approaches**: 기존 임상 ASR 평가는 한 문장(단일 레퍼런스)만을 기준으로 WER/CER 같은 문자열 기반 지표를 계산해, 동일 발음이라도 철자(문자표기)가 다른 ‘유효한 표기 변형’을 오류로 과소평가하는 문제가 있었습니다. 다국어 ASR 연구는 code-switching(언어 음향 교체) 모델링과 데이터 증강에 주로 집중했지만, 임상에서 나타나는 multiscript variability(정서법 변형으로 인한 many-to-one 대응) 자체를 공정하게 평가하는 벤치마크는 부족했습니다.

- **Core Contribution**: 이 논문은 임상 도메인에서의 multiscript variability에 특화된 벤치마크 MultiClin을 제안하고, 의료 용어가 로마자/현지 스크립트 음차 표기로 동시에 존재하는 상황을 다중 정답으로 평가합니다. 또한 동적 multi-reference 평가가 기존 single-reference 평가보다 ASR 성능을 더 공정하고 현실적으로 반영한다는 점을 실험으로 보여줍니다.

- **Technical Challenges**: 핵심 난제는 ‘시간 정렬이 어긋난 예측’과 ‘철자 변형’이 결합될 때, 어떤 구간을 해당 용어 비교의 대상으로 삼을지 안정적으로 결정하는 것입니다. 이를 위해 논문은 추적 커서로 예측에서 윈도우(50자)를 잡고 Longest Common Substring(LCS)로 엔티티 구간을 정렬한 뒤, 해당 경계 내에서 local CER/WER을 계산하는 localized evaluation 알고리즘을 설계합니다.

- **Empirical Impact**: 여러 ASR 모델(Whisper, Qwen3 ASR, Gemini)에서 single-label(original) 평가에서 multiscript-aware(both) 평가로 바꾸면 오류율이 일관되게 크게 감소하며, 예를 들어 Gemini 2.5 Pro는 WER이 28.28%→15.78%로 줄었습니다. 학습에서는 script unification(100% transliteration ratio)이 가장 좋은 성능을 보였고, 특히 50% 매핑에서는 엔트로피가 커지는 비일관성 효과가 관측되어 수렴을 방해한다는 해석까지 제시했습니다. 결과적으로 MultiClin과 multiscript-aware 평가가 임상 ASR의 ‘진짜 역량’을 드러내는 공정한 측정 틀로 자리잡을 가능성을 보여줍니다.



### Human-in-the-Loop Atlas-Based 3D Asset Segmentation for Interactive Content Workflows (https://arxiv.org/abs/2606.17824)
- **Prior Approaches**: 기존 3D 세그멘테이션은 데이터셋/카테고리 의존적이거나, 단순 기하 프리미티브 가정으로 복잡한 표면에서 성능이 떨어지는 한계가 있습니다. 2D foundation model(SAM 2 등)을 3D로 옮기는 zero-shot/few-shot 접근도 자동 의미 라벨링 중심이라, 사용자나 애플리케이션이 원하는 “의미 있는 경계”를 직접 정의·수정하기 어렵습니다. 또한 멀티뷰를 atlas에 투영하는 방식은 가능성을 보였지만, 커버리지를 고려한 뷰 최적화와 사람의 반복 보정(refinement) 메커니즘이 부족했습니다.

- **Core Contribution**: 이 논문은 인간이 의미(세그먼트 기준)를 통제하고, AI가 수정 비용을 줄이는 human-in-the-loop 파이프라인을 제안합니다. 3D 모델에서 최소 뷰를 뽑아(coverage 중심) 렌더링한 뒤, SAM 2와 Label Studio 기반의 대화형 마스크 수정으로 2D 세그먼트를 만든 다음, UV 파라미터화에 back-projection해 통합된 segmented 2D atlas를 생성합니다. 그 결과 atlas 단위로 머티리얼 할당, 스타일 전이, 의미 라벨링 같은 텍스처-스페이스 후속 작업을 더 손쉽게 수행할 수 있습니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 너무 많은 뷰 대신 “표면을 빠짐없이” 커버할 뷰를 고르는 것과 (2) 2D 마스크의 불완전/과분할을 사람의 반복 수정으로 수렴시키는 것입니다. 논문은 뷰 선택을 robotics 관점의 viewpoint planning처럼 set cover 문제로 정식화하고 greedy 전략으로 카메라 후보 중 커버 효율이 큰 뷰를 반복 선택해 최소 뷰를 확보합니다. 이어 Label Studio에서 SAM 2 마스크를 최소 프롬프트로 생성·수정하고, 여러 뷰의 결과를 UV로 투영·병합해 일관된 atlas를 만듭니다.

- **Empirical Impact**: 문화유산 객체 8종 데모 기반 평가에서 전체 파이프라인이 다양한 형상(약한 경계, 얇은 구조, 캐비티, 미세 디테일)에서도 “사용 가능한” segmented atlas를 생성했습니다. 기록된 주석 시간은(로그된 7개) 대체로 15~35분이며 평균 약 21분 수준으로, 큰 연속 영역은 SAM 2의 이점이 크지만 미세 구조·좁은 부착물·경계 대비가 낮은 경우엔 수작업 보정이 늘어났습니다. 특히 David 대두의 눈/귀, 날개가 본체와 합쳐지는 천사 조각, 얇은 스트랩/캐비티에서의 경계 수정 필요가 반복 실패 모드로 드러나, 향후 자동화에서 어떤 개선이 우선인지 방향을 제공합니다.



### A Framework for Evaluating Agentic Skills at Sca (https://arxiv.org/abs/2606.17819)
- **Prior Approaches**: 기존 에이전트 벤치마크는 대체로 task solving, tool use, 코딩 능력처럼 ‘일반 성능’을 측정하며, skills가 모델의 행동을 어떻게 바꾸는지에 초점을 두지 못했다. skills 평가 연구도 소수의 고정 hand-authored 과제로 제한돼 도메인 커버리지가 좁고, 새로 만든 skill에 대해 실제 효용을 추정하기 어렵다는 한계가 있었다. 또한 고정 벤치마크는 skill 저자 관점의 실전 질문(이 skill이 내 의도한 작업에서 정말 도움이 되나?)에 직접 답하기 어렵다.

- **Core Contribution**: 이 논문은 agent skill의 효용을 정량 평가하는 평가 프레임워크를 제안한다. skill 콘텐츠(필요 시 사용자 intent)를 바탕으로 ‘skill이 관여해야 하는’ 현실적 실행 과제를 생성하고, instruction-following과 goal-completion을 위한 rubric로 채점해 with-skill vs without-skill 성능 차이를 통해 skill utility를 추정한다. 특히 단일 skill만 독립적으로 평가해 weak spot을 찾는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난점은 (1) skill이 실제로 쓰이는 맥락을 반영하는 과제를 자동 합성하면서도, (2) 과제가 실행 가능하도록 environment 의존성을 맞추고, (3) 정답 절차나 rubric가 과제에 새지 않게(루브릭 leakage 방지) 하는 것이다. 논문은 environment engineering, task generation, 검증/품질관리 에이전트 파이프라인을 구성하고, end-to-end 자동 모드에 더해 human-in-the-loop 혼합 모드를 지원해 모호함·검증 실패를 줄인다. 또한 task 설명에서 필요한 ‘정확한 수행 단계’를 노출하지 않고, LLM-as-judge로 rubric 기반 점수를 산출해 채점 일관성을 확보했다.

- **Empirical Impact**: 500개 실세계 open-source skill에서 약 1,000개 과제를 만들어, 총 19개 모델 구성(상용+오픈)으로 약 38,000개 valid trajectory를 실험했다. 결과적으로 대부분 모델에서 skill 접근이 instruction-following 중심으로 유의미한 개선(5.5~22점대)을 보였고, 특히 workflow/형식 준수가 중요한 카테고리(Media & File Processing, Security & Compliance)에서 향상 폭이 크게 나타났다. 동시에 모델마다 skill 활용도가 크게 달라 어떤 모델은 거의 개선이 없었으며, 이는 skill이 ‘작동한다/안 한다’를 구분해주는 실전적 신호로 해석된다. 저자들은 평가용 dataset을 공개해, 향후 개별 skill 검증과 비교 실험을 촉진할 계획이다.



### Conservation Laws for Modern Neural Architectures (https://arxiv.org/abs/2606.17816)
Comments:
          Published at the International Conference on Machine Learning (ICML 2026)

- **Prior Approaches**: 보존법칙(gradient flow에서 경로를 따라 불변인 양)은 신경망의 implicit bias를 설명하는 핵심 렌즈로 자리 잡았지만, 주로 선형/ ReLU 같은 제한적 설정에 집중돼 왔다. Marcotte 등은 Lie-theoretic 관점에서 얕은 네트워크와 ResNet/Transformer 일부를 다뤘지만, multi-head attention에서의 완전한(complete) 보존법칙 규명은 여전히 열린 문제였다. 또한 최신 활성함수(GELU/SiLU/SwiGLU), RoPE 같은 포지셔널 인코딩, MoE의 다양한 gating 설계까지 “현대 아키텍처 전반”을 하나의 틀로 묶어 다룬 연구는 드물었다.

- **Core Contribution**: 이 논문은 현대적으로 널리 쓰이는 여러 구성요소에 대해 보존법칙을 완전하게(characterize) 기술하는 unified framework를 제안한다. 구체적으로 GELU/SiLU/ SwiGLU 기반 feedforward 네트워크, multihead attention(특히 multi-head의 완전 규명), sinusoidal PE와 rotary positional encoding(RoPE)의 차이를, 그리고 dense/sparse 포함 MoE의 다양한 gating 설계까지 포괄한다. 즉, “어떤 불변량이 존재하며 전부가 무엇인지”를 아키텍처 수준에서 체계적으로 매핑한다.

- **Technical Challenges**: 핵심 난점은 보존법칙 정의가 학습 손실과 데이터셋에 강하게 의존해 보이지만, 이를 일반적으로 다루면 문제 자체가 난해해진다는 점이다. 저자들은 Marcotte 등의 약화된 보존법칙 정의(모든 데이터셋에 대해 성립하는 C1 함수의 불변 조건)와 손실의 구조 가정을 통해 입력 x에 의존하던 제약을 매개변수 공간 제약(PDE 형태)으로 “모델-독립적으로 축약”하는 전략을 택한다. 그 결과 feedforward에서는 GeLU/SiLU의 경우 보존법칙이 상수로 귀결되고, SwiGLU에서는 행/열 노름 차이 같은 명시적 불변량 구조가 나타나며, attention에서는 head별로 Q/K와 V/O의 특정 조합이 보존됨을 도출한다. 추가로 RoPE는 vanilla attention의 내부 구조를 바꿔 서로 다른 보존법칙 클래스를 만들어낸다는 점을 정리한다.

- **Empirical Impact**: 이론적으로 예측한 불변량(invariant)이 실제 학습 경로에서 유지되는지 실험으로 검증해, 무작정 추상적인 분석이 아니라 학습 동역학을 정확히 포착함을 보여준다. 저자들은 full-batch와 SGD 학습을 모두 사용하고, 작은 규모부터 더 큰 멀티모달 데이터셋까지 걸쳐 관측을 제시한다. 결과적으로 현대 딥러닝 구성요소(활성함수/attention/PE/MoE)의 implicit bias를 보존법칙 관점에서 설명하고 예측할 수 있는 이정표를 제공하며, 향후 보존법칙을 활용한 학습 가속·안정성 분석에도 직접적으로 기여할 가능성이 크다.



### No-Free-Fairness: Fundamental Limits and Trade-offs in Learning Systems (https://arxiv.org/abs/2606.17810)
- **Prior Approaches**: 그동안의 공정성 연구는 다양한 공정성 정의와 완화 전략을 제시해왔지만, 많은 불가능 결과는 주로 지표 간 비호환성(metric incompatibility)에 초점을 맞췄습니다. 예컨대 서로 다른 기준(프라이버시, 정확성, 공정성)을 동시에 만족시키기 어렵다는 유형의 결과가 주로 다뤄졌고, 병리적(특이한) 구성에 한정된 경우도 많았습니다. 또한 공정성 문제를 ‘학습 자체의 구조’나 ‘유한 표본이 만드는 통계적 병목’까지 보편적으로 설명하는 데는 여전히 공백이 있었습니다.

- **Core Contribution**: 이 논문은 공정성 불가능성을 ‘No-Free-Fairness theorems’로 체계화하며, 불공정이 편향된 데이터나 최적화 실패만이 아니라 문제 구조·유한 데이터·모델 표현력에서 함께 비롯된다는 점을 이론적으로 보여줍니다. 특히 공정성 척도를 absolute disparity 대신 risk ratio로 두고, 상대적(비율 기반) 격차가 규제 맥락과 희귀 사건/불균형 상황에서도 더 의미 있게 해석된다고 정당화합니다. 결과적으로 공정성을 달성하려면 본질적인 trade-off(성능-공정성 프런티어)를 설계 관점에서 전제해야 한다는 결론을 제시합니다.

- **Technical Challenges**: 핵심 난제는 공정성 정의를 그룹별 ‘조건부 위험’으로 통일해, 어떤 의사결정 규칙/학습 알고리즘/가설집합에서도 발생하는 하한을 뽑아내는 것입니다. 논문은 (1) 하위집단에 irreducible cost가 존재하면 어떤 규칙도 전체 성능과 격차를 동시에 최대로 만족할 수 없다는 점, (2) 노이즈가 없어 완벽한 공정·정확해가 존재하더라도 유한 표본 학습만으로 subgroup disparity가 불가피하다는 점, (3) 모델 클래스가 하위집단별 정답을 표현하지 못하면 데이터나 학습 절차와 무관하게 공정성이 달성되지 않는다는 점을 각각 보입니다. 이를 통해 분포에 대한 일반적(distribution-free) 공정성 보장이나 ‘클린 데이터면 해결된다’는 내러티브를 동시에 약화시키는 구조적/통계적/표현력 기반 하한을 구성합니다.

- **Empirical Impact**: 이론 결과는 특정 실험 세팅을 넘어서, 불공정이 데이터 편향의 부작용이 아니라 학습 문제가 내재한 구조적 한계라는 관점을 강화한다는 점에서 분야 전반의 설계 방향에 영향을 줍니다. 특히 공정성 목표를 ‘정확성과 무조건 동시 달성’이 아니라 ‘명시적 trade-off 최적화’로 재정의해야 하며, 희귀/소수 하위집단에서는 샘플링 불충분이 통계적 병목을 만든다는 경고를 제공합니다. 결과적으로 공정성 평가와 알고리즘 선택 시 성능 저하·데이터 요구량·모델 표현력의 상호 제약을 함께 고려해야 한다는 실무적 함의를 남깁니다.



### Position: Coding Benchmarks Are Misaligned with Agentic Software Engineering (https://arxiv.org/abs/2606.17799)
- **Prior Approaches**: 기존 코딩 에이전트 벤치마크(SWE-Bench, HumanEval, MBPP, LiveCodeBench, BigCodeBench)는 모델·해네스·환경을 하나의 end-to-end 점수로 합쳐 비교하며, 보통 단일 reference solution(단일 정답 코드) 기준으로 채점한다. 이 구조는 LLM의 한 번에 코드 생성 능력에는 맞지만, 실제 에이전트 소프트웨어 공학에서 핵심인 시스템(오케스트레이션, 컨텍스트, 도구, 피드백 루프)을 분리해 평가하기 어렵다.

- **Core Contribution**: 이 논문은 현재 벤치마크가 agentic software engineering과 불일치하며, 점수가 ‘모델’이 아니라 ‘시스템 해네스’ 전체에 의해 좌우될 수 있다고 지적한다. 또한 단일 reference 기반 채점이 정답 대안(다른 구현·리팩터링·추상화 선택)을 동일하게 불리하게 만들고, 구성요소 단위 신호 부재로 반복 개선(iteration)이 막힌다고 주장한다. 해결 방향으로는 해네스가 복합 시스템이라는 점을 반영해, 독립적인 행동 명세로 정확성을 근거짓고 구성요소별 평가 신호를 제공하는 벤치마크 설계를 제안한다.

- **Technical Challenges**: 가장 어려운 과제는 operationalisation으로, 원하는 동작을 자동 채점 가능한 측정 항목으로 정의하되 ‘어떻게’ 시도해야 하는지까지 인코딩하지 않는 것이다. 논문은 단일 reference 테스트 세트를 multi-shape behavioural verifiers(프로퍼티 테스트, reference oracle, differential testing, 또는 reference에 대해 ‘필수 행동’과 ‘부수 행동’을 분리)로 바꾸고, end-to-end 점수 외에 해네스 구성요소(컨텍스트 유효성, 불변식 준수, 정책→결정적 검증기로의 변환 등)를 고정 조건에서 분리 평가하는 설계를 요구한다.

- **Empirical Impact**: 논문은 여러 결과를 통해 동일 모델이라도 해네스·환경·오케스트레이션에 따라 SWE-Bench 등에서 점수가 큰 폭(예: 20%p 이상, run/seed·컨테이너 등으로도 유의미한 변화)으로 달라진다고 정리한다. 또한 단일 reference 기반 채점의 타당성 문제(누출, 불충분한 테스트 통과, developer-written 테스트 실패, 실제 유지보수 합격률과의 괴리)를 기존 연구가 보여줬음을 근거로, 에이전트 연구가 잘못된 단서에 의해 귀결될 위험을 강조한다. 결론적으로 벤치마크가 모델이 아니라 시스템 해네스를 더 정확히 측정하도록 바꾸면, 에이전트 개선의 방향성이 실사용에 더 가깝게 정렬될 것으로 기대한다.



### LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams (https://arxiv.org/abs/2606.17798)
- **Prior Approaches**: 기존 Video-LLM-online 연구들은 EOS(End-Of-Sequence) 토큰을 예측해 ‘침묵 구간’을 학습하는 방식이 주류였습니다. 다만 침묵 프레임이 압도적으로 많아 데이터 불균형을 만들고, 시각 증거와 무의미한 EOS의 매핑 충돌로 video-language 정렬이 약해지며, 인접 프레임에서 상반된 타깃이 나와 학습 안정성도 떨어집니다.

- **Core Contribution**: 이 논문은 장시간 스트림에서 항상-on으로 동작하면서도 “언제 응답할지”를 스스로 결정하는 프로액티브 라이브 스트리밍 어시스턴트 LiveStarPro를 제안합니다. 핵심은 침묵을 생성 타깃으로 두지 않고, 모델의 confidence로 ‘응답 타이밍’을 검증하는 방식으로 구조를 바꿔 실시간성과 정렬 품질을 함께 노린다는 점입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 매 프레임마다 생성-비교를 하면 지연이 커지는 문제, (2) 증분으로 누적되는 visual context에 맞춰 학습이 online 정렬을 제대로 따라가야 하는 문제, (3) 무한에 가까운 스트림에서 망각 없이 효율적으로 장기 기억을 찾아오는 문제입니다. 이를 위해 Streaming Verification Decoding(SVeD)로 단일 forward pass 기반 perplexity 검증으로 ‘watching↔speaking’ 게이트를 만들고, Streaming Causal Attention Masks(SCAM)로 이벤트 단위에서 언어 복사 누수를 막으며, Tree-Structured Hierarchical Memory(TSHM)로 Peak-End 압축과 Recursive Event Tree 기반 장기 검색을 결합합니다.

- **Empirical Impact**: 실험 결과 LiveStarPro는 기존 온라인 Video-LLM 대비 semantic correctness 28.9% 향상, timing error 18.2% 감소를 보이며, streaming key-value cache까지 활용하면 동일 모델에서 1.58x 추론 속도 개선을 달성했습니다. 또한 시간 단위(시간 스케일) 장기 기억을 평가할 수 있는 OmniStarPro 벤치마크를 통해, 실제 온라인 조건에서 장기 맥락 유지와 응답 타이밍 정확성이 동시에 개선됨을 보여줍니다.



### MIVE: A Minimalist Integer Vector Engine for Softmax LayerNorm and RMSNorm Acceleration (https://arxiv.org/abs/2606.17781)
- **Prior Approaches**: 기존 LLM 가속기는 매트릭스 곱을 빠르게 만들지만, Softmax·LayerNorm·RMSNorm 같은 비선형 벡터 연산은 element-wise 변환과 벡터-wide reduction 때문에 병목이 되기 쉽습니다. 이를 줄이기 위해 PWL 근사, division/exponential 대체, 저비트 제곱 및 INT 연산 친화적 양자화 같은 기법들이 제안됐지만, 대체로 특정 한 함수에만 최적화돼 하드웨어가 분리되는 문제가 남아 있었습니다.

- **Core Contribution**: 이 논문은 LayerNorm, RMSNorm, Softmax를 공통의 primitive 연산 묶음으로 분해한 뒤, 이를 단일 programmable datapath에서 실행하는 Minimalist Integer Vector Engine(MIVE)를 제안합니다. Softmax·LayerNorm·RMSNorm을 각각 별도 블록으로 구현하던 접근을 넘어, 같은 muladd·vecsum 같은 공유 연산 구조로 하드웨어 재사용을 극대화하는 것이 핵심입니다. 결과적으로 한 엔진이 세 정규화 커널을 포괄하도록 설계해 구현 오버헤드를 줄입니다.

- **Technical Challenges**: 세 연산은 수학적으로 다르지만, 하드웨어 관점에서는 element-wise 계산과 벡터-wide reduction, 그리고 Softmax/LayerNorm의 correction처럼 “부분 결과를 갱신해야 하는 흐름”을 함께 처리해야 합니다. MIVE는 ISA 레벨에서 primitive과 실행 흐름을 제어하고, muladd는 PWL 기반 exponent/역수·역제곱근 및 correction 계수 계산을 담당하며, vecsum은 tree 기반 reduction에서 최대/합을 함께 처리하도록 설계해 이 복잡성을 한 데이터패스에 흡수합니다. 또한 INT8 양자화 추론을 전제로 고정소수점과 충분히 넓은 중간 정수 포맷으로 오버플로를 피하면서 근사 정확도를 유지합니다.

- **Empirical Impact**: 28nm ASIC 합성 결과, MIVE는 Softmax·LayerNorm·RMSNorm을 하나의 공유 엔진으로 지원하면서도 면적 효율이 기존 단일 목적(accelerator) 대비 최대 12.3배까지 개선되는 등 하드웨어 효율을 입증했습니다. 전력 효율 역시 GOPS/mW 기준으로 2.8배~3.4배 수준의 향상을 보여 단일 블록 중복을 줄인 효과가 확인됩니다. 한편 PWL 근사의 정확도 영향은 LAMBADA(OPT-30B) accuracy가 81%→80%로, Wikitext(Llama2-7B) perplexity가 5.8→6.0로 소폭 변화하는 데 그쳐, 모델 수준 성능 저하가 제한적임을 보였습니다.



### A Neuromorphic Trigger for Efficient Audio Event Detection (https://arxiv.org/abs/2606.17775)
Comments:
          9 pages, 4 figures, 6 tables

- **Prior Approaches**: 기존 Sound Event Detection(Sed) 모델들은 이벤트의 on/off-set까지 함께 예측하며, 그 과정에서 이벤트 사이의 시간 구간까지도 끝까지 처리해야 해 계산·메모리 비용이 커지기 쉽습니다. 또한 ASD(Anomalous Sound Detection)는 대체로 분류/복원 기반의 무거운 백엔드 모델을 사용해 실시간·저전력 환경에서 효율이 제한됩니다. 전력 절감을 위한 binary network, SNN, dynamic network 시도도 있었지만, 핵심 병목인 “전 구간 처리”를 근본적으로 줄이긴 어려웠습니다.

- **Core Contribution**: 이 논문은 오디오 이벤트 감지를 위한 neuromorphic trigger를 제안하며, SNN이 입력을 선택적으로 gate해 “중요한 오디오 구간만” 후단의 고비용 모델로 전달하도록 만듭니다. 즉 trigger는 분류 자체를 하지 않고, 처리 필요 여부를 나타내는 마스크를 생성해 end-to-end 파이프라인의 연산을 줄이는 저비용 프런트엔드로 설계됩니다. trigger는 TUT Rare Sounds 2017의 SED 파이프라인에 결합해 연산량을 줄이고, URBAN-SED에서는 class-agnostic ASD 성격의 지표로 성능을 검증합니다.

- **Technical Challenges**: 핵심 난제는 (1) 스파이크 기반 출력의 비미분성을 학습에 어떻게 연결할지, (2) “클래스 구분 없이” 이벤트 존재 구간을 얼마나 정확히 마스킹할지, (3) 마스크 노이즈를 줄이면서도 실사용 오류율을 악화시키지 않을 균형입니다. 저자들은 LIF 기반의 lightweight fully connected SNN에 Van Rossum distance를 loss로 사용해 스파이크 타깃 학습을 수행하고, close-open 필터로 스파이크 열을 후처리해 끊긴 신호를 연결하고 잡스러운 양성 구간을 제거합니다. 또한 확대(expansion) 파라미터를 통해 precision·recall 트레이드오프를 조절하도록 구성했습니다.

- **Empirical Impact**: URBAN-SED 기반 ASD(1초 세그먼트, class-agnostic)에서 trigger는 F1 0.97 수준을 달성해 관련 오디오 구간 탐지가 매우 안정적임을 보여줍니다. SED에서는 DCASE 2017 Task 2(TUT)에서 Dang classifier와 결합해 FLOPs를 잠재적으로 42.6x까지 줄이면서 event-based error rate의 lower bound을 0.41에서 0.25로 낮추는 결과를 제시합니다. 결론적으로 neuromorphic trigger는 실시간·에너지 효율을 노리는 오디오 전처리 필터로서, 큰 모델의 연산 부담을 크게 줄일 수 있는 가능성을 실험적으로 입증했습니다.



### Talking to Your Data: Exploring Embodied Conversation as an Interface for Personal Health Reflection (https://arxiv.org/abs/2606.17767)
- **Prior Approaches**: 웨어러블 건강 데이터는 대개 대시보드 차트와 요약 통계로 제공되지만, 사용자가 패턴을 해석하고 의미를 도출하는 데 인지 부하가 커 참여 지속성이 낮다는 문제가 자주 지적돼 왔습니다. 대화형 접근도 있었으나, 대부분은 대화를 ‘설명 전달’ 채널로만 쓰고 대화 자체를 공유된 경험으로 설계하지 못해 실제 감상은 여전히 추상적으로 남는 경우가 많았습니다. 특히 임상 조언이 섞일 때 사실성·안전성 요구가 커져, 상호작용 양식 자체의 효과를 분리해 보기 어려웠습니다.

- **Core Contribution**: 이 논문은 ‘talking to your data’를 목표로, 대시보드(공유 시각 맥락) 위에서 체화된(embodied) 대화 에이전트가 객관적 추세를 구어 통계(spoken statistics)로 풀어주는 인터랙션 패러다임을 제시합니다. 핵심은 dual-agent 설계로, Observer가 시계열에서 기술통계·시간적 경향을 계산하고 Presenter가 오직 미리 산출된 Insight JSON에 근거해 대화를 생성하되 임상 조언은 의도적으로 배제한다는 점입니다. 이를 통해 ‘표현 방식(상호작용 양식)’이 건강 지표 해석에 미치는 영향을 비교 가능하게 만듭니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 대화형 내러티브가 그럴듯해지면서도(환각 위험) 실제 데이터 근거를 유지해야 한다는 점입니다. 논문은 Observer의 정해진 전처리 결과(가중 선형회귀·회귀 감쇠, 상관·추세, 그리고 GPT-4o-mini 기반의 비직관 패턴 탐지)를 Insight JSON으로 구조화하고, Presenter는 Strict-Grounding Prompting으로 해당 사실만 인용하도록 제한했습니다. 또한 발화는 최대 5문장으로 제한하고 ‘지난 수요일’ 같은 구어적 시간 앵커를 써 자연스러운 대화 흐름과 shared-view 정합성을 확보했습니다.

- **Empirical Impact**: N=5의 simulated-self within-subject 연구에서 Agent 조건은 대시보드 대비 인지 부하를 크게 낮추면서(중앙값 5.0→2.0) 행동 계획의 구체성/특정성을 높였습니다(Specificity score 평균 1.25→2.0). 참가자들은 Agent가 이해·신뢰는 유지한 채 관찰에서 실행 계획으로 넘어가는 ‘서사적 브릿지’ 역할을 한다고 응답했지만, 4/5는 첫 탐색은 대시보드를 선호하는 ‘Future Use Paradox’를 보였습니다. 즉 체화된 존재감 자체보다 언어적 지시(temporal deictic cues)가 공동 주의를 형성해 해석을 돕는 경향이 확인되며, 건강 헬스-IUI에서 embodiment의 기여를 사실 근거+언어적 앵커 중심으로 재정의할 실마리를 제공합니다.



### Symplectic Transversality and Endpoint Green Estimates for Finite-Horizon Pontryagin Systems (https://arxiv.org/abs/2606.17762)
Comments:
          20 pages

- **Prior Approaches**: 유한-수평 horizon의 이산시간 Pontryagin 시스템은 control elimination 이후 두 끝점의 상태-코상태(boundary value) 문제가 되는데, 전통적으로는 선형화된 two-point 시스템을 푸는 과정에서 horizon에 따라 조건수가 나빠지는 문제가 컸습니다. 기존 접근은 내부 forcing에 대한 exponential dichotomy로는 잘 제어되지만, 양 끝점에 붙는 endpoint row가 만드는 “반대 끝점에서 시작하는 보정 모드”가 유한-수평 가중치 추정에 결정적이라 누락되기 쉽습니다. 또한 many 결과가 최적해에서의 민감도/감쇠(예: second-order sufficient conditions 등)를 요구해, 본 논문처럼 stationarity branch 수준에서는 바로 쓰기 어렵습니다.

- **Core Contribution**: 이 논문은 smooth control elimination 뒤에 생기는 유한-수평 Pontryagin boundary value 시스템에 대해, horizon과 무관한 상수로 성립하는 “두 끝점 endpoint inverse(2-point endpoint inverse)” 조건을 핵심 입력으로 제시합니다. 증명에서는 이 inverse를 endpoint-corrected Green estimate로 검증하고, 여기에 weighted contraction을 결합해 존재·유일성·Lipschitz dependence 및 first-order expansion을 horizon-uniform하게 얻습니다. 더 나아가 affine/smooth endpoint row를 포함해 초기 상태 고정과 terminal costate-상태 coupling 같은 원형 Pontryagin row까지 커버하며, 검증은 Riccati·symplectic 기준(행렬 데이터 단계)으로 가능하다고 정리합니다.

- **Technical Challenges**: 가장 큰 난제는 유한-수평에서 선형화된 두 끝점 시스템의 역행렬이 horizon에 따라 폭주하지 않음을 보장하는 것입니다. 이를 위해 논문은 stable–unstable 경계전이(transversality)를 스케일링한 boundary matrix의 균일 invertibility로 endpoint inverse를 확인하고, finite-horizon에서 필연적으로 생기는 endpoint correction 항을 포함한 endpoint-corrected Green estimate(커널) 형태로 재구성합니다. 이후 이 커널을 horizon-의존 weighted 공간에서 정확히 소비하도록 설계해, 수축 성질과 결합된 고유값/모드의 양 끝점 영향까지 함께 제어합니다.

- **Empirical Impact**: 행렬 데이터에서 symplectic/Riccati criteria로 inverse hypothesis를 판정할 수 있으며, 특히 invertible dynamics와 definite weights를 갖는 stabilizable LQ 시스템은 noncommuting coupled 데이터까지 포함해 모두 적용된다고 주장합니다. 마지막으로 수치 실험 섹션에서는 이 “증명서(certificates)”가 실제로 horizon-uniform first-order expansion을 뒷받침함을 보여, 이론이 단순 존재성에 그치지 않음을 강조합니다. 결과적으로, shooting·continuation·sensitivity 분석에서 stationarity branch 근방을 horizon에 무관하게 다루는 실전 도구로 확장될 여지가 큽니다.



### ED3R: Energy-Aware Distributed Disaster Detection Enabled by Cooperative Robotic Agents (https://arxiv.org/abs/2606.17739)
Comments:
          14 pages, 9 figures

- **Prior Approaches**: 기존 에너지 효율 연구는 네비게이션·센싱·인프라 자원·연산 오프로딩 같은 개별 요소를 줄이는 데 집중하는 경우가 많았고, 시간 제약을 만족하는 쪽이 주된 목표였습니다. SAR/UAV 관련 연구들도 위험 환경에서의 탐색·객체 탐지·경로 최적화는 다루지만, 산불 탐지를 위해 이동·통신·연산 오프로딩을 함께 최적화하는 통합 접근은 상대적으로 부족했습니다.
또한 산불 탐지 분야의 다수 방법은 분산 제어나 규칙 기반에 치우쳐 있었고, 미래 결과를 미리 평가하는 forward-looking reasoning은 거의 없거나 결여되어 있었습니다.

- **Core Contribution**: ED3R은 불확실성 하에서 산불을 “요구 confidence로 탐지”하면서, 로봇의 이동·센싱·컴퓨팅·통신으로 소모되는 에너지를 최소화하도록 설계된 energy-aware 분산 프레임워크입니다. 로봇과 원격 컨트롤러(RC) 사이에 계층적 협력 의사결정을 두는데, RC는 motion command를 정하고 로봇은 탐지 실행 위치(onboard vs remote)와 사용할 모델(how)을 자원 기반으로 선택합니다.
또한 장애물 회피·중복 탐색 방지·적응적 early mission completion·페널티 함수로 제약 가능성을 강화해, 단순 성능 최적화가 아닌 임무 성공을 겨냥합니다.

- **Technical Challenges**: 가장 큰 어려움은 RC와 로봇이 분산된 상태에서 행동 결과에 대한 공통 보상이 “동시에” 주어지지 않는다는 점입니다. ED3R은 이를 해결하기 위해 distributed neural regression 모델로 후보 전략들의 미래 효과를 미리 평가한 뒤, 탐지 confidence와 에너지 효율 사이의 최적 trade-off를 그리디하게 선택합니다.
여기에 통신 대역폭/전송 파워/채널 조건, 센서 샘플 크기와 처리 FLOPs 같은 현실적인 에너지·지연 요소를 모델링해 제약 위반을 커스텀 페널티로 반영합니다.

- **Empirical Impact**: 현실적인 로보틱스 시뮬레이션과 ablation, 베이스라인 비교를 통해 ED3R은 최대 97.18% 임무 성공률을 달성했습니다. 특히 가장 까다로운 임무에서 에너지는 최대 36.4% 줄이고, 산불 탐지는 최대 41% 더 빠르게 수행해 시간-에너지-신뢰도 동시 최적화의 실효성을 보여줍니다.
forward-looking과 분산 계층 의사결정이 결합될 때, 네트워크가 바뀌거나 새로운 산불 상황에서도 강건한 성능을 보이며 관련 분야의 산불 감시/긴급 대응 설계 방향에 의미 있는 근거를 제공합니다.



### Structured Adversarial Camouflage via Voronoi Diagrams (https://arxiv.org/abs/2606.17711)
- **Prior Approaches**: 기존의 적대적 카모플라주 공격은 픽셀 단위 또는 임의 텍스처를 최적화하는 방식이 많아 계산비용이 크고, 시각적으로 튀는 패턴이 되기 쉽습니다. 또한 인쇄/직물 구현을 고려한 색 선택 제약이나 구조-재현성의 일관성을 충분히 다루지 못해 실제 전이(transfer) 검증이 제한되는 경우가 있습니다.

- **Core Contribution**: 이 논문은 Voronoi 다이어그램 기반의 ‘adversarial Voronoi camouflage’를 제안합니다. 핵심은 픽셀 전체를 직접 학습하지 않고, 고정된 printable color palette 안에서 seed-point(씨앗 점) 위치만 최적화해 구조화된 splinter-like 패턴을 만들며, 추가 정규화 없이도 탐지기 신뢰도를 떨어뜨리도록 한 점입니다.

- **Technical Challenges**: seed-point를 조정해 생성된 구조가 실제 탐지 성능을 얼마나 공격적으로 만들지(그리고 사람이 보기엔 자연스럽게 보일지)라는 문제가 기술 난제로 남습니다. 논문은 거리 기반 soft assignment(temperature-scaled softmin)를 미분 가능하게 구성해 전체 Voronoi 패턴 생성-탐지기 통과-손실 최소화가 end-to-end로 되도록 했고, 3DPeople의 segmentation mask를 사용해 패치가 사람 ‘의복 영역’에 정렬되게 학습합니다.

- **Empirical Impact**: 실험 결과, 단순 bbox 내부에 패치를 얹어 학습한 naive 배치 방식은 상대적으로 효과가 낮았지만, 의복 단위(3DPeople mask) 적용에서는 person detection에서 COCO 스타일 AP@[.5:.95]가 유의미하게 떨어졌습니다. 또한 배경이 바뀌거나 YOLOv9/10/11/12로 detector family가 달라져도 공격 특성이 전이되며, palette를 바꿔 repaint하면 효과가 크게 무력화돼 ‘structure-palette coupling’이 관찰됩니다. 다만 물리 구현(인쇄 적합성, 색 캘리브레이션, 변형, 인간 요인)은 향후 과제로 남기며, 전반적으로 실시간 탐지 성능 저하와 시각적 그럴듯함 사이의 트레이드오프를 보여줍니다.



### Vision-language models for chest radiography do not always need the imag (https://arxiv.org/abs/2606.17710)
- **Prior Approaches**: 기존 의료 VLM 평가는 주로 정확도(accuracy)에 의존하는데, 이는 정답이 영상에 인과적으로 의존하는지 구분하지 못한다. 학습 데이터의 finding-name prior나 동반(co-occurrence) 통계로도 충분히 그럴듯한 yes-or-no 답이 가능하고, saliency/attention 같은 사후 해석도 인과성을 보장하지 못한다.

- **Core Contribution**: 이 논문은 영상 조작을 통해 “모델이 실제로 이미지를 읽는지”를 점검하는 causal audit(인과 감사) 프레임을 제안한다. 동일 라벨의 다른 환자 이미지 교체(swap), 방사선사가 표시한 목표 영역 occlusion(target mask), 무관 영역 occlusion(irrelevant mask)을 함께 적용하고 세 가지 행동 지표(CGR, UAR, IS)로 영상 의존성을 분해해 평가한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 정확도만으로는 보이지 않는 ‘언어 단서 기반 정답’을 영상 의존성과 분리해 측정하는 것이다. 저자들은 MS-CXR phrase-grounding 박스와 임상 라벨을 조합해 2,575개 yes-or-no 프로브를 만들고, 9개 시스템(텍스트 전용/비전 전용 프로브 포함)에 동일한 네 조건을 적용한 뒤 dataset·해상도·프롬프트 문구까지 바꿔도 카테고리가 유지되는지 교차검증한다.

- **Empirical Impact**: 결과적으로 9개 중 3개는 CGR=0으로 ‘이미지 미사용’ 범주에 들어가고, 1개는 영상 사용이 불안정하며, 나머지 5개도 영상 정보를 선택적으로만 사용(발견 일부에 한정)하는 것으로 나타난다. 더 나아가 정확도만 보면 멀티모달이 우세해 보여도, 텍스트 전용 모델이 상위 멀티모달에 근접하거나 통계적으로 비슷한 사례가 있어 “정확도=영상 사용” 주장은 성립하지 않는다; 임상 배포 게이트는 정확도가 아니라 grounding audit처럼 인과적 점검으로 해야 한다는 결론을 내린다.



### Confusion-Aware Transfer Teacher Curriculum Learning Framework: Disentangling Scoring and Pacing Effects (https://arxiv.org/abs/2606.17706)
Comments:
          Accepted at International Conference on Machine Learning (ICML) GlobalSouthML Workshop (2026)

- **Prior Approaches**: 커리큘럼 러닝은 샘플을 ‘어려움’으로 점수화한 뒤, 더 어려운 샘플을 학습 후반으로 천천히 넣는 두 축(점수/페이싱)이 결합돼 성능이 개선되는 경우가 많다. Transfer Teacher framework(TTF)에서는 정답 클래스에 대한 예측 confidence로 난이도를 정의하지만, 오분류 클래스들에 확률이 어떻게 분포하는지는 무시해 오류의 성격을 구분하기 어렵다. 또한 기존 평가는 최종 test accuracy에 의존해 점수 개선과 페이싱 개선을 분리해 원인을 규명하기 어려웠다.

- **Core Contribution**: 이 논문은 TTF의 난이도 점수에 ‘혼동(confusion)’ 정보를 반영하는 confusion-aware difficulty score를 제안해, 단순히 정답 confidence가 낮은 것뿐 아니라 오답들로의 확률 질서(특정 경쟁 클래스에 집중되는지)를 함께 고려한다. 더불어 stage-wise test subset 평가와 pacing-isolated baseline을 도입해 ‘점수(난이도 함수)’와 ‘페이싱(훈련 순서)’의 기여를 분리한다. 이를 통해 TTF에서 난이도 점수만 좋아져도 정확도 향상이 자동으로 일어나지 않는 한계를 체계적으로 드러낸다.

- **Technical Challenges**: confusion-aware score를 설계하려면 softmax 출력에서 오답 클래스에 대한 확률분포의 구조적 특성을 안정적으로 요약해야 한다. 논문은 정답 확률(1-p_true)과 오답 확률분포의 confusion variance(ConfVar)를 곱 형태로 결합해, ‘불확실함’과 ‘특정 클래스에 대한 구조적 혼동’이 동시에 클 때 난이도가 높게 되도록 했다. 실험적으로는 full-data로 학습된 teacher에서만 난이도 신호가 선명한데, 10% teacher에서는 난이도 구배가 거의 평평해져 점수의 신뢰성이 급격히 떨어지는 문제도 확인했다.

- **Empirical Impact**: CIFAR-10에서 ResNet-18, VGG-16을 대상으로 한 결과, 새 점수는 사람 직관과 맞는 model-interpretable 난이도 순위를 만들지만(TTF 정확도 전반 개선은 제한적), CL/Anti-CL은 표준 학습 대비 full data 정확도를 유의미하게 끌어올리진 못했다. 반면 자료 효율 관점에서는 confusion-aware curriculum ordering이 random ordering보다 최대 8.7%p까지(20% 데이터 구간) 향상돼, TTF가 정확도 한계와 별개로 데이터 절약형 학습으로 활용될 여지를 보여준다. 또한 stage-cosine learning rate처럼 커리큘럼에 맞춘 페이싱-스케줄 재적응이 TTF의 성능 격차를 만들어낼 수 있는 중요한 방향임을 시사한다.



### SegTME-UNI2: A Foundation Model-Based Framework for Generalisable Multiclass Cell Segmentation and LLM-Driven Tumour Microenvironment Characterisation in Histopathology (https://arxiv.org/abs/2606.17702)
- **Prior Approaches**: H&E 병리영상에서 TME를 보려면 보통 (1) 세만틱 분할(픽셀을 세포 범주로 라벨링)과 (2) 핵 instance 분리(서로 맞닿은 핵을 개별 객체로 분리)가 동시에 필요합니다. 기존 instance 분할 모델(HoVer-Net 등)은 PanNuke 같은 소규모 nucleus-level polygon 주석에 의존해 확장성이 떨어지고, 세만틱 분할(UperNet 등)은 핵이 붙어있을 때 분리가 어렵습니다. 또한 TCGA 같은 대규모 저장소는 픽셀 단위 주석이 없어 임상 보고 흐름에 바로 쓰기 힘든 ‘해석 가능한 산출물’까지 이어지는 경로가 막혀 있었습니다.

- **Core Contribution**: 이 논문은 SegTME-UNI2로, 세만틱 분할·HV 회귀 기반 watershed용 핵 분리·구조화된 TME 피처 생성·BioNeMo GPT를 통한 임상 서술까지 한 번에 묶는 통합 프레임워크를 제안합니다. 핵심은 UNI2-H pathology foundation model(UNI2-H, ViT-Giant)을 공유 인코더로 쓰고 UperNet 디코더를 두 갈래로 붙인 UNI2-UPERHOVER(세만틱 6-class + horizontal-vertical gradient 회귀)입니다. 여기에 pseudo-label 기반 3-stage progressive curriculum과 20+ per-patch TME feature→BioNeMo narrative 변환 파이프라인을 결합해 “대규모·해석가능·분리까지”를 동시에 달성하려고 합니다.

- **Technical Challenges**: 가장 큰 난관은 TCGA-UT처럼 픽셀 주석이 없는 대규모 데이터에서 instance 수준의 핵 분리를 학습시키는 것입니다. 이를 위해 논문은 세만틱 마스크만으로 동적으로 HV 타깃을 합성해 instance polygon 주석 없이도 watershed 분리를 가능하게 하고, pseudo-label의 질을 단계적으로 올리기 위해 PanNuke→TCGA-UT scale-0→TCGA-UT 전 해상도(총 1,608,060패치)로 확장하는 3단계 커리큘럼을 설계합니다. 또한 해상도 스케일이 0.25~1.0 μm/pixel로 크게 바뀌는 상황에서 다중 스케일 문맥을 담기 위해 UperNet의 PPM+FPN 구조를 채택합니다.

- **Empirical Impact**: PanNuke와 TCGA-UT 분할에서의 예비 검증은 프레임워크의 실행 가능성과 내부 일관성을 보여주며, 특히 TCGA-UT처럼 넓은 분포에서 세만틱 문맥과 핵 분리를 함께 얻을 수 있음을 시사합니다. 생성된 TME 피처(조성·임상 비율·공간 상호작용·공간 엔트로피 등 20+ 지표)는 JSON으로 정형화되어 BioNeMo GPT의 fine-tuning 입력으로 들어가 임상적으로 읽히는 “패치 내 TME 내러티브”를 목표로 합니다. 더 나아가 pseudo-labelled TCGA-UT 데이터와 UNI2-UPERHOVER 체크포인트를 공개해, 대규모 spatial biology/병리 정량화 연구에 직접 활용될 수 있는 기반을 제공합니다.



### SuCo: Sufficiency-guided Continuous Adaptive Reasoning (https://arxiv.org/abs/2606.17687)
Comments:
          Accepted to ICML 2026. 18 pages

- **Prior Approaches**: LRM은 CoT(Chain-of-Thought)를 생성해 복잡한 추론에서 성능을 끌어올렸지만, 실제론 불필요하게 긴 추론을 반복해 추론 비용과 지연을 키우는 문제가 컸습니다. 이를 줄이기 위한 ALRM(Adaptive Large Reasoning Models)은 외부 추정기나 사전 정의된 reasoning mode/예산 tier로 “이산(discrete) 전환”을 하는 방식이 많아, 언제 추론을 멈춰야 하는지에 대한 원칙이 약했습니다. 또한 CoT를 줄이기 위한 휴리스틱 길이 제한이나 binary triggering은 underthinking과 overthinking을 동시에 정교하게 다루기 어렵다는 한계가 드러났습니다.

- **Core Contribution**: 이 논문은 CoT 궤적에서 정답을 내는 데 충분한 최단 prefix를 “Minimal Sufficient CoT(MSC)”로 정의해, 추론 길이를 줄일 때의 기준을 원리적으로 제시합니다. MSC는 어려움 수준별로 답 정확도를 유지하면서도 reasoning 토큰을 크게 줄일 수 있음을 실험으로 확인하고, 이를 바탕으로 연속적인 추론 제어 프레임워크인 “Sufficiency-guided Continuous Adaptive Reasoning(SuCo)”를 제안합니다. SuCo는 이산 모드 없이 문제 난이도에 맞춘 “충분성(sufficiency) 임계값”을 학습해, 필요한 만큼만 추론하도록 만듭니다.

- **Technical Challenges**: 핵심 난제는 “충분한 추론”의 신호를 학습에 쓸 수 있게 정량화하는 것입니다. 저자들은 정답을 뒷받침하는 CoT prefix의 sufficiency를 모델의 조건부 확률을 기반으로 정의하되, 긴 답에서 생기는 신호 붕괴를 완화하기 위해 per-token 평균화를 사용하고, 문장 단위로 MSC를 탐색해 논리 구조가 깨지지 않게 했습니다. 또 임계값을 고정하면 쉬운 문제엔 과잉 추론, 어려운 문제엔 조기 절단이 발생하므로, 데이터에서 추론 길이의 percentile로 문제 난이도를 추정해 임계값을 적응시키는 방식(MSC-Aligned Fine-Tuning, MFT)을 쓰고, 이후 RL 단계(Sufficiency-Aware Policy Optimization, SAPO)에서는 dynamic complexity pool과 sufficiency-aware reward로 over-/under-thinking을 함께 패널티로 제어합니다.

- **Empirical Impact**: 수학·코드·과학 벤치마크에서 SuCo는 정확도와 추론 효율을 동시에 개선하며, full CoT 대비 reasoning 토큰을 약 74~76% 줄이면서도 성능을 유지/상회하는 결과를 보였습니다. 예를 들어 7B 스케일에서 정확도는 72.1%로 LHRM보다 상대적으로 5.1%p 높고, DeepSeek-R1-Distill-Qwen 대비 14.1%p 격차를 벌렸습니다. 특히 AIME25처럼 어려운 구간에서 정확도 향상이 두드러져(7B에서 61.7%), “더 적게 추론해도 더 잘할 수 있다”는 MSC의 관점을 다양한 난이도에 걸쳐 실증했다는 점에서 의미가 큽니다.



### See First, Answer Later: Visual Evidence Pre-Alignment via Sufficiency-Driven RL (https://arxiv.org/abs/2606.17678)
- **Prior Approaches**: 기존 MLLM 학습은 대규모 이미지-캡션 기반 pretraining으로 ‘거친’ 비전-언어 정렬을 만든 뒤, SFT와 RL로 답변 추종과 복잡한 추론을 강화하는 2단계 파이프라인을 주로 사용합니다. 하지만 캡션은 짧고 성긴(supervision이 거칠고) 편향이 있어 세밀한 속성·관계·덜 두드러진 영역에 대한 미세 grounding을 충분히 학습시키지 못합니다. 결과적으로 추론 시 언어 priors에 기대어 이미지 근거가 약해지고, 중요한 시각 디테일 누락이나 환각이 생길 수 있습니다.

- **Core Contribution**: 이 논문은 pretraining과 post-training 사이에 중간 단계 Visual Evidence Pre-Alignment(VEPA)를 추가해, 추론 전에 ‘질문-조건 시각 증거(visual evidence)’를 먼저 생성하도록 정렬합니다. 핵심 아이디어는 답변 최적화가 아니라, evidence 생성 정책 P(e|v,q)를 강화해 evidence가 주어진 질문을 풀 수 있을 만큼 충분(sufficiency)하면서도 이미지에 의존(visual dependence)하도록 만드는 것입니다. 또한 표준 post-training과 보완적으로 작동하며, 추가 task별 어노테이션 없이도 시각 grounding을 높이는 방향을 제시합니다.

- **Technical Challenges**: 문제는 세밀한 evidence 토큰 라벨이 대규모로 없다는 점인데, VEPA는 이를 RL로 우회해 증거 생성 전체 시퀀스를 보상으로 학습합니다. 이를 위해 정답 유출(answer leakage)과 반복/퇴화된 evidence를 피하도록 보상을 정교하게 설계하며, sufficiency 신호는 이미지 없이 evidence만 보는 ‘frozen blind reader’가 주어진 질문을 풀 수 있는지로 간접 측정합니다. 안정적인 장문 RL 학습을 위해 sufficiency-driven Group Relative Policy Optimization(GRPO)로 그룹 상대 advantage를 사용해 업데이트 변동성을 줄입니다.

- **Empirical Impact**: 실험에서는 VEPA를 끼운 모델이 다양한 비전 요구 벤치마크에서 일관되게 성능을 개선하며, in-domain 정확도를 크게 해치지 않으면서 domain shift(예: 차트·텍스트 기반 시각 과제)에서 더 큰 이득을 보였습니다. 또한 POPE/MMStar에서 이미지 없이 evidence만 보고 답을 시도하는 blind reader의 정확도가 향상되고, 생성 evidence 길이는 오히려 짧아져 verbosity가 아닌 selectivity·충분성 강화가 원인임을 뒷받침합니다. 정량/정성 분석을 종합하면 VEPA 효과는 추가 task 학습이 아니라 transferable한 시각 grounding 강화에서 온다고 결론내립니다.



### ASTEROID: A Spatiotemporal Information Transformer for Forecasting Multi-Step Time Series of Molecular Dynamics (https://arxiv.org/abs/2606.17668)
Comments:
          32 pages,10 figures

- **Prior Approaches**: 분자 dynamics(MD) 시뮬레이션은 장시간·대규모 시스템에서 계산 비용이 크게 듭니다. 기존 예측은 시간 적분을 반복해 좌표를 갱신하는 방식이 많아, 다단계(multi-step) 전망을 할수록 오차 전파와 계산 부담이 커지는 한계가 있습니다. 또한 반복 통합을 그대로 두고 데이터 모델을 보조하는 접근은 긴 지평선에서의 효율성과 정확도 균형이 어렵습니다.

- **Core Contribution**: 이 논문은 MD의 궤적을 입력으로 받아 다음 좌표를 반복적으로 적분하지 않고, 다단계 원자 좌표를 한 번에 예측하는 데이터 기반 프레임워크 ASTEROID를 제안합니다. ASTEROID는 MD 궤적을 고차원 spatiotemporal 시퀀스로 재구성하고, spatiotemporal 정보(STI) Transform 방정식을 Transformer 구조에 통합합니다. 핵심은 다중 스케일 공간·시간 의존성을 동시에 모델링해 시뮬레이션 가속의 목적을 직접 달성하는 점입니다.

- **Technical Challenges**: 다단계 좌표 예측에서는 (1) 단·장거리 공간 상호작용을 놓치지 않으면서 (2) 긴 시간에 걸친 동역학 문맥을 안정적으로 유지하는 것이 핵심 난제입니다. ASTEROID는 공간 의존성에 대해 local-global self-attention으로 짧은 상호작용과 긴 상호작용을 함께 포착하고, 시간 의존성에는 encoder-decoder 구조로 전역 문맥을 반영한 autoregressive forecasting을 설계합니다. 이를 통해 conventional iterative integration을 피하면서도 장기 예측을 지원하도록 했습니다.

- **Empirical Impact**: ASTEROID는 quantum-mechanics에서 유도된 여러 분자 데이터셋 벤치마크에서 기존 방법 대비 다단계 예측 정확도를 개선했습니다. 동시에 기존 MD 시뮬레이션의 계산 비용을 유의미하게 줄였고, 확장된 시간 스케일에서도 반복적 multi-step forecasting을 수행할 수 있습니다. 결과적으로 MD 가속을 위한 일반화 가능한 데이터 기반 패러다임을 제시했다는 점에서 분야에 실질적 의미가 큽니다.



### Handling Feature Heterogeneity with Learnable Graph Patches (https://arxiv.org/abs/2606.17667)
Comments:
          Accepted at KDD 2025

- **Prior Approaches**: 범용 Graph Foundation Model(GFM)을 만들려는 시도는 늘었지만, 그래프 데이터의 feature heterogeneity 때문에 도메인 간 전이가 잘 되지 않는 문제가 남아 있습니다. 기존 접근은 텍스트를 LLM으로 공통 임베딩 공간에 매핑하거나(OFA) SVD 같은 단순 표준화로 feature 길이를 맞추는 방식(GCOPE류)이 많지만, 텍스트/부가정보가 없을 때는 적용이 어렵고 구조-특징의 공동분포까지 정렬하긴 힘듭니다.

- **Core Contribution**: 이 논문은 그래프의 최소 의미 단위로 “learnable graph patches(학습 가능한 그래프 패치)”를 정의하고, 노드 feature를 펼쳐(tokens) 각 채널별로 패치 구조를 따로 학습하는 PatchNet을 제안합니다. patch encoder는 각 패치 안에서 전이 가능한 정보(특징-구조 상관)를 뽑고, patch aggregator는 패치들을 모아 전체 표현을 구성해 도메인 비의존적으로 downstream에 적용되게 합니다. 특히 텍스트 없이도 멀티 도메인 그래프를 pre-training에 활용할 수 있도록 설계한 점이 핵심입니다.

- **Technical Challenges**: 관건은 (1) 서로 다른 의미 공간에 있는 노드 feature를 어떻게 구조와 함께 “전이 가능한” 단위로 재구성하느냐, (2) 그런 단위를 pre-training 중 스케일 가능하게 인코딩·결합하느냐입니다. 저자들은 feature를 overlap을 가진 tensor unfold로 lossless하게 token으로 만들고, token 채널마다 shared graph learner로 learned adjacency(\tilde{A})를 추정해 token-structure 상관이 담긴 패치를 구성합니다. 이후 patch encoder의 듀얼 브랜치 attention(GNN 기반)과 transformer 기반 patch aggregator로 패치 길이/개수가 달라도 학습 안정적으로 컨텍스트를 통합합니다.

- **Empirical Impact**: 실험에서는 멀티 도메인 그래프를 이용한 pre-training이 가능하고, 다양한 downstream 데이터셋·태스크에서 성능이 개선되는 것으로 보고됩니다. 또한 pre-training 데이터의 양이 늘어날수록 downstream 성능이 일관되게 향상되는 경향을 관찰해, 제안한 패치 기반 전이 메커니즘이 실무적으로도 확장 가능함을 시사합니다. related work와의 비교/분석을 통해 PatchNet이 기존 GNN과 graph transformer의 장점을 분리·재조합한 형태라는 점도 강조합니다.



### FacProcessTwin: An LLM-Based System for Process Twin Developmen (https://arxiv.org/abs/2606.17666)
- **Prior Approaches**: 기존 digital twin은 주로 단일 자산(장비/센서)을 중심으로 구성돼 전체 공정 최적화를 노리기 어렵습니다. 또 P&ID나 AutomationML처럼 기계가 읽기 쉬운 구조화 문서로부터 시뮬레이션/트윈을 자동 생성하는 연구가 많지만, 공장에 실제로 남아 있는 서술형 SOP와 표를 그대로 활용하긴 쉽지 않습니다. LLM 에이전트는 보통 이미 존재하는 트윈을 운영·진단하는 데 쓰이며, 문서 기반 개발 단계의 데이터 바인딩을 안전하게 검증/수정하는 공백이 남았습니다.

- **Core Contribution**: FacProcessTwin은 공장 SOP 같은 서술형 공정 문서와 작업자 자연어 입력만으로 프로세스 트윈을 end-to-end로 생성하고 OPC UA 라이브 데이터에 자동 바인딩합니다. 핵심은 LLM이 모델을 만들고 바인딩을 “제안”하되, 안전에 직결되는 모호한 태그 매칭은 human-in-the-loop로 작업자에게 확인을 요청해 임의 추측을 차단한다는 점입니다. 결과물은 인터랙티브 프로세스 다이어그램으로 제공돼 작업자가 자율 의사결정을 검토·교정할 수 있습니다.

- **Technical Challenges**: 문서에는 공정 단계·장비·제품별 설정과 변동성이 분산돼 있고, 특히 작업자가 머릿속에만 가진 tacit know-how가 많아 정확한 공정 그래프 복원이 어렵습니다. 또한 OPC UA 서버는 벤더/구성에 따라 태그 목록 조회가 비어 있거나 식별자 변경이 발생해 태그 디스커버리와 매핑이 까다롭습니다. FacProcessTwin은 단계 추출과 레이아웃을 결정론적 도구로 고정하고, 바인딩 단계에서 모호성·안전 민감도를 기준으로 작업자 확인 체크포인트를 두는 방식으로 “잘못된 매핑이 조용히 누적되는” 위험을 줄였습니다.

- **Empirical Impact**: 호주 식품 제조사의 실제 SOP에서 16개 공정 플로우(냉장·냉동·무균/상온살균 카테고리)를 평가한 결과, 공정 구조는 평균 F1 95.2%로 ground truth에 가깝게 복원됐고(토폴로지 F1), 매핑은 100% mapping recall을 보였습니다. 개발 시간은 수작업 대비 약 6배 빨라 플로우당 평균 5.2분으로 단축됐습니다. 가장 중요한 안전 지표로, safety-critical 모호 태그에서 단일 패스 baseline이 조용히 잘못 바인딩하던 75.0% 오류율을 human-in-the-loop로 0%로 낮춰 현장 적용 가능성을 보여줍니다.



### Temporal Preference Optimization for Unsupervised Retrieva (https://arxiv.org/abs/2606.17664)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: 기존 비지도 dense retriever들은 unlabeled 문서의 contrastive learning으로 의미적 유사도는 잘 학습하지만, 쿼리의 시간 맥락과 문서 timestamp가 어긋난 temporal misalignment를 놓치는 경우가 많았습니다. 감독 기반 temporal retriever는 성능이 좋더라도 명시적 timestamp 라벨이 필요해 대규모 적용이 어렵다는 한계가 있습니다. 반면 Contriever류의 time-unaware 모델은 의미는 맞는데 시간은 어긋난 문서를 함께 끌어오는 경향이 관찰됩니다.

- **Core Contribution**: 이 논문은 비지도 retriever에 시간 정렬을 “선호(preference)” 형태로 주입하는 TPOUR(Temporal Preference Optimization for Unsupervised Retriever)를 제안합니다. 핵심은 Temporal Retrieval Preference Optimization(TRPO)로, temporally aligned 문서를 선호하고 unaligned 문서를 덜 선호하도록 학습해 의미 유사도와 시간 적합성을 함께 강화합니다. 또한 time vector interpolation을 통해 retraining 없이 미지의 시간 구간(중간 시점, 미래)으로 연속적 일반화까지 확장합니다.

- **Technical Challenges**: 가장 큰 과제는 라벨 없이도 “시간 선호” 신호를 구성해 retriever가 temporally aligned와 misaligned를 구분하도록 만드는 것입니다. 저자들은 문서 코퍼스가 여러 시간대에서 수집되었다는 점을 활용해 학습 중 preferred/less preferred 쌍을 만들고, TRPO의 loss를 MoCo 기반 contrastive objective와 결합해 시간 차이를 임베딩에 반영합니다. 또 discrete time 모델의 경계를 넘기 위해, 생성 모델에서 쓰이던 time vector를 encoder retriever에 적용하고 중간 시점은 interpolation으로 처리합니다.

- **Empirical Impact**: 실험에서 TPOUR는 temporal information retrieval(T-IR)에서 비지도/감독 baseline을 모두 능가하며, explicit 및 implicit 시간 질의 모두에서 nDCG@5가 유의미하게 상승했습니다. 특히 Qwen-Embedding-8B 대비 약 72.7배 작은 규모임에도 평균 nDCG@5를 explicit +4.04(+12.15%), implicit +4.98(+15.21%) 개선했습니다. 더 나아가 BEIR 등 일반 검색에서도 데이터셋 publication year와 최적 성능이 정렬되는 시간 민감성이 드러나, 시간 모델링이 범용 검색에도 실질적 이득을 줄 수 있음을 보여줍니다.



### TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins (https://arxiv.org/abs/2606.17660)
Comments:
          9 pages, 6 figures, accepted as ICML 2026 poster:this https URL

- **Prior Approaches**: 기존 연구는 (1) 짧은 학습 곡선을 외삽하거나 (2) NTK/스케일링 같은 이론적 신호를 이용해 “잘 될지”를 추정한다. 또한 proxy 모델(COSMOS, ProxyLM 등)로 최종 성능을 예측하지만, 점수 하나로 섞여서 실패 원인을 특정하기 어렵다. 그 결과 데이터 품질과 모델/하이퍼파라미터 상호작용을 분리해 “왜 실패하는지”까지 설명하는 데 한계가 있다. 


- **Core Contribution**: TUNEAHEAD는 본격 fine-tuning을 실행하기 전에 성능을 pre-hoc으로 연속값 형태로 예측하는 경량 프레임워크를 제안한다. 각 candidate run을 정적 데이터 기술자와 짧은 standardized probe에서 뽑는 동적 probe feature로 인코딩해, 성능을 RMSE 수준으로 정확히 추정한다. 동시에 SHAP 기반 attributions로 어떤 feature가 예측을 좌우하는지 진단 가능하게 만든다. 


- **Technical Challenges**: 핵심 난제는 “짧은 비용으로도 실패/성공을 가르는 상호작용 신호”를 안정적으로 포착하는 메타-피처 설계와, 품질 좋은 설명 가능성을 함께 확보하는 것이다. TUNEAHEAD는 static dataset descriptors(예: lexical/semantic 다양성, reference perplexity)로 데이터의 내재 품질을 잡고, 100-step probe의 loss/gradient/landscape 신호로 모델-데이터 불일치나 최적화 불안정 같은 조기 징후를 동적으로 반영한다. 또 50개 이상 후보 feature를 SHAP로 사전 선택·중복 제거·방향성 일관성 필터링해 24개로 압축하고, LightGBM+TreeSHAP 조합으로 정확도와 진단 가능성을 동시에 노린다. 


- **Empirical Impact**: 1,300+개의 Qwen2.5-7B-Instruct fine-tuning run에서 TUNEAHEAD는 테스트 370개 셋에서 RMSE 1.47%p, 예측의 95.1%가 ±3%p 이내에 들어가는 성능을 보였다. Early-Stop Extrapolation, ProxyLM 같은 강한 baseline을 일관되게 능가하며, 정적/동적 피처만 단독 사용한 ablation보다 큰 이득을 확인했다. 이 연속 예측은 go/no-go screening 정책으로 불필요한 full fine-tuning을 줄이면서 유망한 run을 더 많이 유지할 수 있어, 실무 비용 절감과 데이터 개선 방향 제시에 의미가 크다. 




### A Risk Decomposition Framework for Pre-Hoc Fine-Tuning Prediction (https://arxiv.org/abs/2606.17649)
Comments:
          9 pages, 4 figures, accepted as ICML 2026 Poster:this https URL

- **Prior Approaches**: 기존 pre-hoc performance prediction 연구는 프록시 기반(작은 모델/데이터로 예측)이나 초반 학습 신호를 이용한 probing을 주로 휴리스틱으로 설계해 왔습니다. black-box 회귀기로 상관관계를 맞추는 방식이 많아, 예측 오차가 “어떤 불확실성”에서 오는지와 “계산 예산에 따라 어떻게 줄어드는지”를 원칙적으로 다루지 못했습니다. 또한 probing depth를 고정된 하이퍼파라미터처럼 취급해 자원 배분 최적화를 설명하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 pre-hoc fine-tuning 예측을 정보 제약 하의 확률적 추정 문제로 정식화하고, 예측 위험을 두 성분으로 분해합니다. 하나는 정적 데이터-모델 호환성에서 오는 Intrinsic Limit(계산해도 줄지 않는 바닥), 다른 하나는 최적화 과정 관측으로 줄일 수 있는 Optimization Variance(가변 불확실성)입니다. 나아가 계산 예산 대비 위험-비용의 최적 tradeoff를 주고, 이를 바탕으로 Predictability phase diagram을 제안해 Static-Sufficient / Dynamic-Critical / Noise-Dominant의 세 레짐을 정리합니다.

- **Technical Challenges**: 핵심 도전은 “계산을 더 쓸수록 불확실성이 얼마나 빨리(어느 속도로) 사라질 수 있는가”를, 전체 학습 궤적을 모델링하지 않고도 보편적 제약으로 도출하는 것입니다. 논문은 locally regular한 안정 구간을 가정해 stochastic approximation 관점에서 최적화 유도 불확실성의 감쇠 속도에 대한 필연적 lower bound(불확실성은 임의로 빠르게 0이 될 수 없음)를 증명합니다. 그 결과 Bayes-optimal 위험 감소의 한계를 이용해 budget-optimal probing 원칙과 위상도(phase diagram)를 설계할 수 있게 됩니다.

- **Empirical Impact**: 합성 데이터와 실제 벤치마크 실험에서 이론이 제시한 세 레짐이 관측되고, probing 전략이 예산 대비 효율적으로 성능 예측 정확도를 끌어올림을 보여줍니다. 특히 어떤 작업은 정적 신호만으로도 예측이 빠르게 포화되고(Static-Sufficient), 어떤 작업은 초반 신호가 늦게 의미를 갖기 때문에 더 긴 관측이 필요하며(Dynamic-Critical), 또 어떤 작업은 intrinsic 바닥 때문에 아무리 probing해도 한계가 있다는(Niose-Dominant) 해석을 제공했습니다. 즉, pre-hoc 예측을 단일 스코어가 아니라 “자원 배분 관점의 예측 가능성 지도”로 재정의하는 데 의미가 있습니다.



### SketchXplain: Intuitive Visual Explanations of Image Classifiers with Sketches (https://arxiv.org/abs/2606.17646)
Comments:
          14 pages, 6 figures, 4 tables. Submitted to TVCG

- **Prior Approaches**: 살iency map은 중요한 픽셀/뉴런을 강조해 예측 근거를 보여주지만, 의미론적 맥락이 부족해 설명이 직관적이지 않은 경우가 많습니다. concept bottleneck류는 사람의 개념과 정렬된다는 장점이 있으나, 이미지 내 ‘어떤 시각 단서가 그 개념을 가리키는지’가 시각적으로 잘 연결되지 않는 경우가 있습니다. 또한 단순 선 그림(또는 기존 렌더링/엣지 기반 스케치)은 생성은 쉬워도 예측과 개념에 대한 선택성·정합성이 떨어질 수 있습니다.

- **Core Contribution**: 이 논문은 설명을 ‘직관적’이게 만드는 목표(단순성·정합성·빠른 이해)에 맞춰, SketchXplain라는 스케치 기반 XAI 방법을 제안합니다. 스케치 생성 과정에 saliency 기반 선택, concept bottleneck 기반 지식 정합, cue 정렬, 그리고 stroke 최적화를 통합해 ‘왜 그 부분이 중요한지’를 사용자가 더 쉽게 받아들이도록 합니다. 결과적으로 단순하지만 예측 라벨과 핵심 개념에 일관된(coherent) 시각 설명을 지향합니다.

- **Technical Challenges**: 핵심 과제는 스케치의 추상화를 유지하면서도 (1) 예측에 실제로 중요한 개념을 반영하고 (2) 이미지 관찰 단서와의 시각적 정합을 확보하며 (3) 선의 복잡도를 낮추는 것입니다. 이를 위해 Grad-CAM으로 개념별 중요도/위치를 얻고, 이미지에서 추출한 detailed line을 가중 마스킹해 개념 단서에 맞는 초기 선을 만들며, differentiable rasterizer 위에서 CLIP 기반 의미 정렬(입력·예측 라벨 임베딩)을 수행해 stroke를 최적화합니다. 추가로 곡률/부드러움 제약과 opacity 조절로 단순성을 강화합니다.

- **Empirical Impact**: 얼굴 표정 인식 태스크에서 SketchXplain은 saliency map이나 단순 선 그림 대비 더 빠른 해석과 더 일치하는 시각화 결과를 사용자 연구로 확인했습니다. 또한 피부 병변 진단으로 확장 평가한 결과, SketchXplain이 질병 증상을 더 ‘일관된 방식’으로 시각화해 일반인 수준의 진단 지원에도 유리함을 보였습니다. 전반적으로 이 연구는 스케치가 이미지 기반 XAI에서 인지적 해석 공백을 메우는 실용적인 설명 매체가 될 수 있음을 시사합니다.



### Bounding Box Label Propagation for Re-Annotation of Document Layout Analysis Datasets (https://arxiv.org/abs/2606.17644)
Comments:
          17 pages, 3 figures, to appear in proceedings of ICDAR 2026, Vienna, Austria

- **Prior Approaches**: 기존 문서 레이아웃 분석(DLA)은 CNN/비전 트랜스포머/vision-language 모델 등을 대규모 라벨 데이터로 fine-tuning해 객체 탐지를 수행합니다. 그러나 산업 현장에서는 클래스 체계가 계속 세분·갱신되며(박스는 유지되기도 함) 매번 전체 재-annotation이 필요해 비용이 커집니다. 준지도 객체 탐지 연구들은 보통 박스 좌표 자체를 의사라벨로 다시 만들기 때문에, 이 문제처럼 ‘이미 존재하는 바운딩 박스의 클래스만’ 재분류하는 상황에는 불필요한 복잡성이 생깁니다.

- **Core Contribution**: 이 논문은 Bounding Box Label Propagation(BBLP)로, 기존 바운딩 박스 좌표는 그대로 두고 클래스 라벨만 재분류하는 pseudo-labelling 프레임워크를 제안합니다. 시각·텍스트·위치 임베딩을 통합한 Layout Object Encoder(LOE)로 객체 단위의 joint embedding을 만들고, 이를 Label Propagation에 plug-and-play 방식으로 연결해 소량의 수작업 라벨로 나머지 박스의 클래스를 전파합니다.

- **Technical Challenges**: 핵심 난제는 객체 탐지에서 ‘박스 인스턴스 단위’로 라벨을 전파할 수 있는 표현을 구성하는 동시에, 박스 크기/해상도 변화와 문서 내 텍스트·위치 의존성을 함께 반영하는 것입니다. 저자들은 NaFlexViT(가변 해상도 시각 임베딩), Tesseract 기반 OCR 텍스트 임베딩(E5), 그리고 정규화된 위치·이웃 관계 기반 positional embedding을 결합해 LOE를 학습하고, 이후 transductive nearest-neighbour graph(코사인 유사도)에서 label propagation을 수행합니다.

- **Empirical Impact**: 실험에서 BBLP는 D4LA에서 mAP 54.0%를 달성했으며, 이는 fully supervised 성능의 81.6%에 해당하고 라벨 10%만 사용한 결과입니다. 또한 의사라벨 정확도 평가와 잡음 내성(잡음 약 30%대 D4LA 포함) 실험에서, BBLP로 생성한 pseudo-label로 학습한 DLA 모델이 ‘수작업 10%만 학습’ 기준을 일관되게 앞서며 label noise가 있더라도 효과적으로 활용됨을 보여줍니다. 모달리티 ablation에서는 텍스트가 D4LA 성능에 특히 중요하고, 시각·위치 정보도 단독보다 다중 결합에서 더 잘 작동해 실제 문서 재-annotation 비용 절감 가능성을 제시합니다.



### Divide, Deliberate, Decide: A Multi-Agent Framework for Fine-Grained Egocentric Action Recognition (https://arxiv.org/abs/2606.17627)
- **Prior Approaches**: 정교한(세밀한) 동작 인식은 손-도구-접촉 같은 미세한 시각/시간 단서만으로 라벨이 갈려 기존 방법의 전이(transfer)가 어렵습니다. VLM을 그대로 쓰면 모델이 지배적인 물체 토큰에 고정돼 판별 단서를 놓치거나, 단일 모델의 편향된 priors가 특정 단서에 과도하게 쏠리는 문제가 자주 보고돼 왔습니다. 스케일업(더 큰 VLM)도 대안이지만 온프레미스·엣지·프라이버시 제약 때문에 항상 현실적이지 않습니다.

- **Core Contribution**: 이 논문은 Divide, Deliberate, Decide라는 fully-local zero-shot 멀티에이전트 프레임워크를 제안합니다. VLM 오케스트레이터가 영상을 청크로 나누고 세그먼트별 top-k 라벨 후보를 만들면, 서로 다른 모델 패밀리의 VLM 전문가들이 peer-consultation Q&A로 근거를 교환하며 순위를 재평가합니다. 마지막으로 Borda count로 순위를 집계하고 오케스트레이터가 재랭킹해 최종 예측을 확정합니다.

- **Technical Challenges**: 핵심 난제는 (1) 미세 단서에 대해 에이전트들이 단순 합의로 수렴하지 않게 하고, (2) 추가 계산 없이도 새 근거가 의사결정에 실제로 반영되게 만드는 것입니다. 이를 위해 Stage 2에서 분쟁이 생길 때만 한 번의 질문으로 시각적 단서를 확인하도록 프로토콜을 구조화하고, 최종 단계에서는 오케스트레이터가 새 라벨을 제시하지 못하게 제한해 deliberation이 유일한 추가 근거가 되도록 설계합니다. 또한 heterogeneity(서로 다른 모델 패밀리)로 priors를 decorrelate해 상호보완적 후보순위를 만들게 했습니다.

- **Empirical Impact**: MECCANO에서 zero-shot 평가를 수행했으며, 제안 방법은 top-1 16.8%, top-5 45.0%로 baseline(13.5%/28.9%)을 일관되게 개선합니다. 오케스트레이터는 deliberation 이후 약 70.9% 세그먼트에서 초기 top-1을 바꿿고, 올바른 방향의 뒤집기가 반대보다 크게 우세(특히 top-5에서 재라벨링 정답 증가가 매우 큼)해 근거 반영이 확인됩니다. 또한 전문가를 동일 백본 3개로 바꾸면 개선 폭이 줄어들어, 성능 이득이 compute 증가가 아니라 decorrelated priors와 구조화된 Q&A에서 온다는 점을 실험적으로 뒷받침합니다.



### SkillMoV: Mixture-of-View Routing with Prototype-Conditioned Gating for Unified Multi-View Proficiency Estimation (https://arxiv.org/abs/2606.17615)
- **Prior Approaches**: 기존 AQA/숙련도 추정은 특정 활동 도메인에 맞춘 시나리오별 모델이 많거나, 여러 카메라를 단순 집계·공유 변환으로 결합해 뷰별 단서를 충분히 활용하지 못하는 한계가 있었다. EgoExo4D에서도 unified multi-view로 가는 흐름이 있었지만, 여전히 공통 투영/어텐션에 의존해 뷰 간 차이를 “전문가”처럼 분리해 학습하긴 어려웠다. 또한 카메라 identity나 시나리오 전용 헤드 없이도 뷰에 종속된 표현을 만들 수 있는 설계가 부족했다.

- **Core Contribution**: 이 논문은 synchronized multi-view 영상에서 여러 스킬 도메인을 아우르는 파라미터 효율적(LoRA 기반) 통합 프레임워크 SkillMoV를 제안한다. 핵심은 Mixture-of-View Projector(MoVP)로, mixture-of-experts를 카메라별 뷰 특징에 직접 라우팅해 뷰 의존적 전문성 선택을 학습하되 카메라 identity supervision 없이도 동작하게 만든다. 여기에 cross-view attention 정렬, prototype anchoring, prototype-conditioned gated projection을 계층적으로 결합해 최종 skill embedding을 만든다.

- **Technical Challenges**: 문제는 서로 동기화된 멀티 카메라가 동일 실행을 보여도 드러내는 숙련 단서가 달라 공통 집계가 신호를 희석할 수 있다는 점이다. SkillMoV는 이를 MoV 라우팅(12개 expert MLP를 뷰별 soft mixture로 선택)으로 해결하고, cross-view attention으로 동기 카메라 정렬 후 prototype anchoring 및 gated projection으로 클래스(숙련 단계)별 참조 기반 조건화를 추가한다. 또한 멀티뷰 설정에서 stochastic view dropout으로 특정 카메라에 과적합하는 현상을 줄였고, 최적 학습 안정성을 위해 class-balanced cross-entropy를 선택했다.

- **Empirical Impact**: EgoExo4D에서 SkillMoV는 Exos 설정 단일 모델 통합 학습으로 50.17% overall accuracy를 달성하며, 비교 방법 중 최강 Exos 결과보다 3.57%p 개선했다. Ego+Exos에서도 47.63%로 기존 최상 수준(48.20%)에 근접해 통합 프레임워크의 견고함을 보여줬다. ablation 결과로는 MoV routing(+6.61%p), cross-view attention(+4.92%p), prototype anchoring(+4.07%p), stochastic view dropout(+3.90%p) 기여가 확인되며, LoRA adaptation을 통해 학습 파라미터를 23.32%만 사용하면서도 오버헤드는 LoRA-only 대비 제한적으로 유지됐다.



### Understanding LLMs in Title-Abstract Screening: From Disagreements to Recommendations (https://arxiv.org/abs/2606.17588)
Comments:
          14 pages + references. Accepted for publication in the 52nd Euromicro Conference on Software Engineering and Advanced Applications (SEAA 2026)

- **Prior Approaches**: 기존 연구들은 LLM을 SR(systematic review) title-abstract screening에 적용하며 대체로 성능이 사람과 비슷하거나 더 낫다고 보고했지만, 리뷰별 신뢰성은 엇갈렸습니다. 특히 정량적 인간-LLM 일치도(예: Kappa, agreement statistic)에 초점이 맞춰져 있어 “왜” 불일치가 생기는지에 대한 설명이 부족했습니다. 또한 불일치 원인을 잡아내지 못하면, human–LLM 워크플로를 어떻게 설계·수정할지 판단하기 어렵습니다.

- **Core Contribution**: 이 논문은 인간-LLM 불일치에 대해 정량 합의지표를 넘어, 어떤 사례에서 어떻게 실패하는지 질적(qualitative) 원인 분석을 수행합니다. 소프트웨어 엔지니어링 6개 SR에서 1,000편이 넘는 primary study의 title-abstract를 humans과 LLM들이 zero-shot으로 각각 판정했고, Kappa가 0.52~0.77로 나타나 “실패 양상”을 구조적으로 분해할 필요가 확인됐습니다. 그 결과 어디서(위치) 무엇이(유형) 문제인지 정리한 불일치 패턴(사실상 taxonomy)과 실행 가능한 권고를 제시합니다.

- **Technical Challenges**: 핵심 기술적 어려움은 LLM이 SR의 포함/제외 기준을 텍스트의 경계와 의미 수준에서 정확히 해석하지 못할 수 있다는 점입니다. 논문은 반복되는 실패 원인으로 (1) 핵심 용어/정보의 명시성 부족, (2) 개념 경계(boundary) 애매성, (3) 키워드 중심의 과의존, (4) 추론된 주제(topic inference) 오류 같은 패턴을 도출했습니다. 또한 이를 완화하기 위해 semantic 이해를 사전 검증하고, 여러 LLM을 돌려 모델별 오해를 드러낸 뒤, 사람의 검증 자원을 경계 사례(borderline cases)에 집중하는 방식의 권고를 제안합니다.

- **Empirical Impact**: 실험적으로는 인간 합의와 LLM 판정이 위치·유형별로 체계적으로 어긋난다는 점이 확인됐고, 전반적으로 Evidence Loss(관련 누락)가 False Positive보다 더 자주 나타나는 경향이 관찰됐습니다. 구체 패턴 중에서는 LLM의 Incorrect Topic Inference, not main focus로 인한 판단, Term—Boundary 문제 등이 상대적으로 빈번했습니다. 이 결과는 “일치도만 보고 끝내기”가 아니라, 경계·의미·기준 정의의 품질을 중심으로 SR 파이프라인을 재설계해야 한다는 실무적 함의를 줍니다.



### Visored: A Controlled-Natural-Language Prover for LLM-Generated Mathematics (https://arxiv.org/abs/2606.17581)
- **Prior Approaches**: 기존 접근은 (1) de Bruijn criterion 같은 작은 커널로 전체 증명을 yes/no로 검증하되, 입력은 이미 formal statement에 가깝고 autoformalization은 별도 병목이 남는다. 또 (2) Lean/Rocq 같은 ITP에 직접 맞춘 whole-proof 모델이나, (3) Mizar·PVS·rule 기반 CNL 프로버, (4) LLM이 개요를 쓰고 남은 단계를 자동 전개하는 방식이 각각 장점이 있지만, informal→formal 사이에서 발생하는 의미 결정들이 중간에 흩어져 실패가 “끝까지 컴파일해봐야” 드러나는 문제가 있었다. 특히 비의도적 의미 드리프트(캐스트·라이브러리 lemma·정의/예외 케이스)가 조용히 생기면, 수정 비용이 커진다.

- **Core Contribution**: 이 논문은 Visored라는 의존 타입 기반(prover) 시스템을 제안하며, 수학자가 LaTeX 속 자연어를 쓰는 방식에 더 가깝게 표면을 설계한다. 핵심은 타입/라이브러리 lemma/엣지 케이스/캐스트처럼 표면에 없는 결정을 semantically rich한 중간표현(IR)인 Visored MIR에서 명시화하고, 각 단계 실패를 소스 위치에 국소화된 diagnostic으로 반환하는 것이다. accepted proof는 선택적으로 Lean 파일로 재출력할 수 있지만, 판정의 정합성은 커널 체킹 경로를 거치며 Lean을 “판정 파이프라인”으로 강제하지 않는다.

- **Technical Challenges**: 주된 기술 난제는 두 가지다: 첫째, LLM이 제안한 CNL의 “Then…” 의무를 수학적으로 사소하게 보이지만 실제로는 arithmetic·대수·순서·집합·named-lemma 등에서 자주 등장하는 스킵 단계까지 안정적으로 닫는 규칙 기반 solver의 성능이다. 둘째, ITP가 요구하는 well-definedness(예: 1/x는 x≠0이 증명돼야 함, sqrt/log는 조건이 필요), coercion·라이브러리 네이밍·부분함수 측면 조건을 정확히 정리해 의존 타입 체크를 통과시키는 elaboration/emit 과정이 까다롭다. 이들은 “사양 파일(.lpcsv)로 정의 가능한 CNL 표면→유형화된 AST→Visored MIR/UVL IR→단계별 rule-driven solving→(옵션) Lean 포맷터”로 파이프라인을 분해하고, 단계별 diagnostic을 내보내며 해결한다.

- **Empirical Impact**: 실험은 miniF2F-valid 244문제를 대상으로 한 narrow prototype 수준의 coverage study로, LLM 에이전트가 verify-and-revise 루프로 Visored를 호출해 222/244(91%)을 Visored 체크 증명으로 도달시켰다. IMO 문제 쪽은 수학적 난도와 규칙/기계 미구현의 영향이 커서 22개 미해결 중 13개가 IMO였고, 나머지도 AM–GM·바닥/로그 합·유한체·극한/삼각항등 등 “추가 규칙이 필요한” 범주가 대부분이었다. 다만 이 프로토타입에서 가장 큰 병목은 Lean transpilation으로, 생성된 Lean 출력이 손작성 대비 평균 약 250배까지 커질 정도로 반복 속도가 느려졌으며, 저자들은 이 설계 레슨을 바탕으로 후속(prover) 개발로 전환했다고 밝힌다.



### LLM Features Can Hurt GNNs: Concatenation Interference on Homophilous Graph Benchmarks (https://arxiv.org/abs/2606.17579)
Comments:
          29 pages, 8 figures

- **Prior Approaches**: 기존 연구들은 TAPE, GLEM 등에서 LLM이 생성한 텍스트/특징을 GNN에 “결합”해 성능을 끌어올리는 전략을 주로 보고해 왔다. 특히 joint training, distillation, prompt-conditioning 같은 장치가 들어가면서 end-to-end 파이프라인 성능이 개선된 결과가 누적됐다. 또한 대규모 벤치마크 집계에서는 homophily 데이터에서 LLM 기반 방법이 대체로 더 잘 작동하는 경향이 강조돼 왔다.

- **Core Contribution**: 이 논문은 같은 homophilous 벤치마크에서도 “순수 입력 결합(pure input concatenation)”만 수행하면 성능이 체계적으로 악화될 수 있음을 정면으로 보여준다. 예를 들어 PubMed에서는 SBERT-인코딩 GPT-4o-mini TAPE 특징을 BoW에 단순 결합했을 때 테스트 정확도가 -17.0±0.3 pp 하락한다. 반대로 WikiCS, ogbn-arxiv처럼 중간 homophily에서는 결합 효과가 양수로 뒤집히며, 단순 결합이 항상 이득이 아니라는 ‘레짐(regime)’을 제시한다.

- **Technical Challenges**: 핵심은 “왜 end-to-end에서는 이득인데 concatenation만 하면 망가지나”를 분리·예측하는 것이다. 저자들은 LLM 특징의 단독 판별력(Δsig)과 결합으로 인한 간섭(Δconcat cost)의 관계를 데이터 9개에서 측정했고, Δsig에 대해 변화점 tau=13.8 pp를 기준으로 ‘Δsig<=tau면 비양수 결합 비용’을 예측하는 간단한 규칙을 제안한다. 또한 차원·가중감쇠 등의 아티팩트를 통제하기 위해 same-dim PCA/가우시안 노이즈/제로 대체 ablation을 수행해, 손실이 LLM 특징의 정보성에 특이적으로 연결됨을 보인다.

- **Empirical Impact**: 실험적으로 Planetoid public split(작은 라벨 수)에서 결합 성능 저하가 가장 크게 나타나며, PubMed의 -17 pp 효과는 학습 데이터 수가 늘면 빠르게 완화된다. 아울러 여러 PubMed 구성에서 |Δconcat|이 (sqrt(d_l/n))^1.31 형태의 파워 법칙(r^2=0.97)을 따르는 스케일링도 제시해, 문제를 데이터 특이 현상보다 “표본 복잡도(sample complexity) 기반 현상”으로 해석하게 한다. 따라서 TAPE/GLEM류의 end-to-end 파이프라인 이득은 joint training·게이팅 같은 결합 메커니즘이 만들어내는 결과이며, 단순 concatenation만으로는 재현되지 않을 수 있다는 실무적 경고가 된다.



### Geometric Consistency Protocol for Foundation Model Features in Multi-View Satellite Imagery (https://arxiv.org/abs/2606.17564)
Comments:
          The manuscript is accepted as Oral Presentation in IEEE International Geoscience and Remote Sensing Symposium(IGARSS 2026)

- **Prior Approaches**: 원격탐사 multi-view에서 기존 평가는 대부분 2D 전역 argmax 매칭(무제약 2D 글로벌 서치)에 의존했습니다. 하지만 RPC 카메라의 epipolar 기하가 곡선이며 고도에 따라 달라서, 물리적으로 가능한 탐색공간이 3D 결정 문제인데도 2D 평면에서만 찾도록 평가가 구성되는 경우가 많았습니다. 또한 위성 영상의 반복 구조(도로·지붕 등)와 방사보정 변화는 유사도 응답의 스푸리어스 최대값을 키워 랭킹을 왜곡할 수 있습니다.

- **Core Contribution**: 이 논문은 RPC(Rational Function Model) 프레임워크에 맞춘 geometry-faithful 평가 프로토콜을 제안합니다. 핵심은 (1) RPC로 투영한 동일 3D 포인트의 교차 뷰 feature 일치성을 보는 RPC-projected 3D consistency 지표와 (2) 기하 제약을 둔 탐색공간에서 유사도 피크의 국소성·유일성을 확인하는 geometry-constrained dense matching proxy를 함께 보고하는 것입니다. 특히 두 측면을 분리해, 의미적(semantic) 일치는 높아도 실제 매칭(matchability)은 보장되지 않는 현상을 명확히 합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 RPC epipolar 기하를 따라 ‘물리적으로 가능한’ 후보 탐색 매니폴드를 평가에 정확히 반영하는 것입니다. 논문은 DSM 기반 고도/역투영(RPC 역투영)과 고정된 재현 가능 솔버 설정을 사용해 동일 3D 포인트를 각 뷰로 정확히 재투영하고, invalid point·가시성 조건 미충족 포인트를 배제하는 방식으로 지표의 일관성을 확보했습니다. 또한 전역 2D 탐색과 RPC epipolar band(예: ±4 pixels) 탐색을 함께 실행해 기하 제약의 영향을 정량 비교할 수 있게 했습니다.

- **Empirical Impact**: DF C2019의 Omaha·Jacksonville 두 full-region(100+ area of interest)에서 실험한 결과, 전역 글로벌 서치는 모든 방법의 성능을 크게 흔들었고 RPC epipolar band로 제한했을 때 clsPCK@10 등이 일관되게 크게 개선되었습니다. 흥미롭게도 2D 백본(dense feature, 예: DINOv3·SAM)은 RPC-consistent 평가에서도 여전히 강력한 경쟁력을 보였고, 3D-aware/ multi-view-aware 계열이 항상 최상 성능을 보장하진 않았습니다. 즉 ‘기하 제약을 포함한 정의가 있어야’ 비로소 foundation feature의 실매칭 능력을 공정하게 비교할 수 있으며, semantic consistency만으로는 성능을 예측하기 어렵다는 메시지를 남겼습니다.



### An AI Security Agent for Banking: Multi-Vector Fraud and AML Detection Across Retail and Corporate Accounts (https://arxiv.org/abs/2606.17555)
Comments:
          7 pages, 1 figure, 5 tables

- **Prior Approaches**: 기존 은행 보안은 주로 룰 엔진으로 brute-force·velocity-burst처럼 개별 이벤트가 튀는 사기/침해를 잘 잡는 데 집중해 왔다. 하지만 BEC 결제 전용(redirection), 세션 하이재킹, 머니라우더링 structuring·layering처럼 ‘개별 거래는 정상처럼 보이지만 관계가 이상한’ 집단 이상(collective anomalies)에는 구조적으로 눈이 멀다.

- **Core Contribution**: 이 논문은 소매·법인 금융을 대상으로, transaction 스트림과 session 스트림의 두 트랙을 동시에 감시하는 AI 보안 에이전트를 제안한다. 각 스트림에서 LSTM 시퀀스 모델·velocity 임계값 모니터·계정-카운터파티 네트워크 프록시(그래프 특징)를 결합해 단일 fused 위험 점수로 위협을 분류·티어링한다.

- **Technical Challenges**: 핵심 난제는 BEC처럼 금액/속도/상대방 프로필만으로는 구분이 어려운 위협과, layering처럼 체인 중간에서 신호가 희미해지는 문제다. 저자들은(1) 두 스트림 분리 후 max 결합, (2) 서브모델별 강점이 사라지지 않도록 single-sub-model override, (3) GNN message-passing 대신 fan-in/fan-out·pass-through 비율 같은 가벼운 네트워크 특징을 써 CPU 환경의 추론·학습 비용을 줄이는 방식으로 해결을 시도했다.

- **Empirical Impact**: 합성 이벤트 로그(거래 237,669·세션 113,508, 13개 위협, 3,470 계정)에서 제안 모델은 transaction 스트림 F1=0.787, session 스트림 F1=0.867로 룰 기반(0.562/0.733)과 LSTM-only(0.655/0.713)를 앞섰다. 또한 고객 대상 거래 검증 챗봇(OTP 기반 96.6% 신원 확인)과 분석가용 케이스 요약 도우미(action F1=99.3%)를 포함하고, Critical 자동응답 지연이 95퍼센타일 기준 0.43 ms 미만으로 보고돼 운영 에이전트 관점의 의미가 크다.



### Reversal Q-Learning (https://arxiv.org/abs/2606.17551)
- **Prior Approaches**: flow matching 같은 iterative generative modeling을 활용한 offline RL은, 표현력이 좋은 대신 반복 생성 과정의 학습 안정성이 큰 걸림돌이 됩니다. 특히 diffusion/flow 정책을 value function에 맞춰 직접 최적화하면 backpropagation through time(BPTT)로 인해 학습이 흔들리고 성능이 떨어질 수 있습니다. 이를 피하려 weighted regression, distillation, rejection sampling 등 우회가 제안됐지만 계산량 증가나 value 활용도 저하 같은 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 flow 정책의 Euler refinement step을 action으로 보고 expanded Markov decision process(MDP)로 바꾸는 발상을 off-policy offline RL에 맞게 재구성합니다. 핵심은 reversal Q-learning(RQL)로, 데이터의 state-action 전이를 expanded MDP의 “virtual on-policy” flow 궤적으로 변환해 value 학습을 가능하게 하는 것입니다. 동시에 multi-step return을 사용해 off-policy에서 악화되는 curse of horizon을 완화하면서, 전체 표현력을 가진 flow 정책을 직접 학습합니다.

- **Technical Challenges**: 가장 큰 문제는 offline 데이터가 원래 MDP의 전이만 제공해 expanded MDP의 중간 flow step 궤적이 없다는 점입니다. 이를 해결하기 위해 현재 flow policy에 대해 reverse flows로 역방향 ODE를 풀어 가상 trajectories를 재구성하고, determinism을 활용해 multi-step return의 추정이 편향·분산 면에서 유리해지도록 설계했습니다. 또한 horizon이 FF배 늘어나는 off-policy TD의 누적 편향을 bias-and-variance reduction 관점의 multi-step 반환으로 줄이도록 구성했습니다.

- **Empirical Impact**: 50개(문장 기준 50+? 표기 포함) 난이도 높은 simulated robotic task에서 RQL이 SOTA off-policy flow 기반 offline RL 알고리즘 대비 평균 성능이 가장 좋다고 보고합니다. 특히 long-horizon manipulation과 locomotion 같은 어려운 환경에서 상대적 강점이 두드러졌습니다. 결과적으로 RQL은 BPTT 부담을 줄이면서 learned value function을 더 효과적으로 쓰고, 표현력 있는 full flow policy를 end-to-end에 가깝게 학습하는 실용적 대안을 제시합니다.



### Offline Preference-Based Trajectory Evaluation (https://arxiv.org/abs/2606.17541)
- **Prior Approaches**: 기존 오프라인 평가에서는 success rate(SR)처럼 “과제를 풀었는지”를 이진 성공으로만 환원하는 경우가 많습니다. 이 방식은 부분 진행을 모두 같은 값(0)으로 접어버리고, 멀티스텝 궤적의 시간에 따른 성능 변화를 끝값으로만 비교해 민감도와 데이터 효율을 떨어뜨립니다.

- **Core Contribution**: 논문은 궤적(trajectory)을 시간에 따른 성과 함수로 보고, 두 궤적을 스칼라 점수로 먼저 계산하지 않고 temporal preference로 직접 비교하는 preference-based trajectory evaluation을 제안합니다. 핵심은 “같은 목표 진전이 있으면 더 빨리 도달한 쪽을 선호” 같은 시간 선호를 평가 척도로 설계해, 성능 정의 자체의 정보 손실을 줄이는 것입니다.

- **Technical Challenges**: 해결해야 할 문제는 시간 정보를 포함하되, temporal discounting처럼 도메인별 가정이나 추가 하이퍼파라미터에 덜 의존하는 방식으로 구현하는 것입니다. 이를 위해 연구진은 lexicographic return preference(LR), return-paired preference(RPP), interval-paired preference(IPP)처럼 return 수준과 시간-진전 관계를 기준으로 하는 설계로, time-to-return 프로파일을 안정적으로 비교할 수 있게 했습니다.

- **Empirical Impact**: 여러 에이전트/인터랙티브 벤치마크에서 SR은 인스턴스 비교의 약 75%를 tie로 만들어 유효 표본 수를 크게 줄였지만, trajectory-aware preference는 tie를 약 35%로 낮춰 판별력을 회복했습니다. 그 결과 discriminative power, ranking stability, data efficiency가 전반적으로 개선됐고, 이 관점은 benchmark saturation이 데이터 수집/난이도만의 문제가 아니라 “평가 지표 선택”에서 비롯될 수 있음을 시사합니다.



### Reinforcing Dual-Path Reasoning in Spatial Vision Language Models (https://arxiv.org/abs/2606.17539)
- **Prior Approaches**: 기존 spatial VLM은 3D 레이아웃, 깊이/가림, 시점 의존 관계를 인식하는 데는 진전이 있었지만, 깊이·거리·장면 관계를 넘나드는 다단계 추론까지는 취약하다는 분석이 반복됐다. RL을 통해 다단계 reasoning을 유도한 방법들도 존재하지만, 기반 VLM이 충분히 강한 공간 지각을 갖추지 못하면 spatial VLM이 제공하는 기하 구조를 충분히 활용하지 못한다. 또한 기존 접근은 언어 기반 추론과 3D grounding 기반 추론을 하나의 모델/훈련 프레임워크 안에서 함께 지원하는 경우가 드물었다.

- **Core Contribution**: SR-REAL(SR-REAL, Dual-Path Spatial Reasoning via Reinforcement Learning)은 공간 VLM에 두 가지 상보 경로를 동시에 탑재한다: LOR(Language-Only Reasoning)는 장면 관계에 대한 단계적 언어 추론을 수행하고, DTR(Detect-Then-Reason)은 3D 기하 단서를 region token으로 먼저 탐지·정렬한 뒤 정량 추론을 한다. 모델은 한 번의 체크포인트로 두 경로를 모두 지원하도록, cold-start supervised fine-tuning으로 두 경로의 CoT(Chain-of-Thought)와 region-to-3D 인터페이스를 만들고 이후 RL로 정확도와 출력 형식을 함께 최적화한다. 결과적으로 “질문 유형에 따라 다른 전략이 필요하다”는 문제를 단일 통합 설계로 흡수한다.

- **Technical Challenges**: 핵심 난제는 텍스트에서 바로 3D 좌표/박스를 예측하는 것이 어렵다는 점이며, SR-REAL은 이를 region token을 매개로 한 2D→3D grounding 다리로 해결한다. cold-start 단계에서 LOR용 언어 CoT와 DTR용 ‘detect-then-quantitative reasoning’ CoT를 각각 구조화해 학습하고, DTR에는 region-to-3D 인터페이스를 통해 센터나 bounding box를 예측하도록 하며 직접 grounding만 하면 성능이 크게 떨어진다는 점을 확인한다. 이후 GRPO 스타일 RL에서 accuracy reward + format reward를 기본으로 두고, DTR에는 discretized detection reward(예측 센터와 정답 좌표 거리 기반)를 추가하며, online filtering으로 비유리 롤아웃을 제거해 안정적인 최적화를 유도한다.

- **Empirical Impact**: 실험에서 SR-REAL은 SPAR-Bench, EmbSpatial, SAT 등 다수 공간 벤치마크에서 기존 spatial VLM/추론 모델 대비 일관되게 향상되며, 단일 모델이 LOR과 DTR을 동시에 제공하는 점도 확인됐다. 특히 DTR은 region 기반 태스크에서 3D localization 정밀도로 이점을 보이고, LOR은 언어적 단계 추론 능력으로 일반 spatial reasoning에 기여한다. ablation 결과로는 두 경로를 함께 학습할 때 상호 강화가 나타나고, cold-start 데이터 품질(2D/3D grounding 블렌딩)이 RL 안정성과 cross-domain 전이에 중요하다는 점이 강조되며, 데이터셋/도메인별 per-task 튜닝 없이 positive transfer가 관찰된다는 의미가 크다.



### OmniDrive: An LLM-Choreographed Multi-Agent World Model with Unified Latent Co-Compression for Multi-View Driving Video Generation (https://arxiv.org/abs/2606.17536)
Comments:
          24 pages, 10 figures

- **Prior Approaches**: 기존 생성형 world model은 멀티뷰 영상을 카메라별로 각각 인코딩한 뒤, cross-view attention 등으로 사후에 맞추는 방식이 많았습니다. 이 구조에서는 언어·HD-map/trajectory 같은 기하 제어와 픽셀 증거가 잠재 토큰 레벨에서 같은 좌표계에 정렬되지 않아 cross-view drift, 깜빡임, 객체 teleporting 같은 문제가 반복됐습니다. 또 ControlNet류 분기(기하)와 cross-attention 어댑터류 분기(의미/프롬프트)를 “후처리로 조합”하는 경향이 있어 이질적인 컨트롤 주입 간 불일치가 남았습니다.

- **Core Contribution**: DRIVE-CHOREO는 이 공통 원인을 ‘언어-기하-픽셀을 latent-token 레벨에서 정렬해 주는 shared symbolic interlingua의 부재’로 짚고, 이를 단일 위치 인식 토큰 그리드로 해결합니다. 핵심은 LLM-choreographed multi-agent world model로, 세 Qwen2.5-VL 에이전트가 WorldScript를 만들고(Director/Architect), 이를 공간 앵커된 레이아웃 토큰으로 바꾸며(Cartographer), 카메라 간 불일치를 비평해 보조 감독(Auditor)을 주는 방식입니다. 나아가 6카메라×시간을 “view-time permutation”으로 재배열해 3-D VAE 안에서 기하 제약이 로컬 합성곱 의존성으로 들어가도록 co-compressed latent를 구성합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 서로 다른 표현 공간(자유 텍스트, 기하/좌표계, 픽셀 잠재)을 토큰 좌표 수준에서 기계적으로 정렬하고 (2) 카메라 수가 늘어도 같은 물리적 순간의 뷰 간 제약을 VAE의 receptive field 안에 담아 일관성을 유지하는 것이었습니다. 논문은 view-time permutation으로 RGB와 Cartographer의 기하 레이아웃을 동일한 pseudo-temporal 스트림으로 co-compression하며, 동시에 same-instant 뷰의 noise endpoint를 공유해 초기 광도 변동을 줄였습니다. 여기에 cross-view 비평 점수를 Auditor auxiliary objective로 흐르게 해, 에이전트 출력이 “사전 처리”가 아니라 diffusion/flow-matching 학습 중간에 직접 영향을 주도록 설계했습니다.

- **Empirical Impact**: nuScenes에서 DRIVE-CHOREO는 multi-view consistency와 BEV mAP에서 새로운 SOTA를 달성했고, BEV mAP 21.6 및 경쟁 FVD 45.7을 보고합니다. 또한 순수 생성 데이터로 학습한 detector가 실측 validation에서 NDS +2.4 향상을 보여 downstream 유틸리티까지 검증했습니다. ablation과 기존 방식(ControlNet/사후 cross-attention)과의 비교는 에이전트 기반 choreographed conditioning과 co-compression이 cross-view 정합성과 controllability를 함께 끌어올렸음을 수치로 뒷받침합니다.



### Scaling Enterprise Agent Routing: Degradation, Diagnosis, and Recovery (https://arxiv.org/abs/2606.17519)
Comments:
          10 pages (6 main + 4 appendix), 4 figures, 6 tables

- **Prior Approaches**: LLM 어시스턴트는 사용자 요청을 도구 라이브러리로 라우팅해 작업을 분담하지만, 카탈로그가 커질수록 성능이 급격히 떨어진다는 보고가 이어졌다. 기존 연구들은 주로 tool calling 자체의 저하나 retrieval 오류 비중을 관찰했지만, 무엇이 어느 지점에서 깨지는지(메커니즘 분해)와 실제 개선 레버가 무엇인지가 덜 규명돼 있었다. 또한 네임스페이스 기반 검색, BM25 같은 플랫폼 도구 검색, 계층형 LLM 라우팅 등은 일부 완화하나 대규모에서 한계가 남는 것으로 보였다.

- **Core Contribution**: 이 논문은 배포된 엔터프라이즈 생산성 어시스턴트의 실제 카탈로그(110 agents, 584 tools)를 사용해, single-step 라우팅 정확도가 에이전트 수 확장에 따라 어떻게 무너지는지 통제 실험으로 진단한다. 특히 oracle 분석을 통해 저하를 retrieval gap과 confusion gap으로 분해해, “정답 도구를 노출하지 못하는 문제”와 “도구를 노출해도 비슷한 도구를 헷갈리는 문제”를 분리해 보여준다. 이어서 embedding-based shortlisting이 두 격차 중 retrieval 쪽을 얼마나 효과적으로 메우는지, 그리고 현업 트래픽에서 재현되는지도 검증한다.

- **Technical Challenges**: 핵심 기술 난제는 카탈로그 확대로 인해 semantically overlapping 도구가 늘며 recall이 무너지는 구조적 문제를, 단 한 번의 라우팅에서 얼마나 회복할 수 있는지다. 저자들은 F1 저하를 recall/precision 관점에서 확인한 뒤, oracle ceiling까지 포함해 retrieval gap(모델이 올바른 도구를 끌어오지 못함)과 confusion gap(올바른 후보를 줘도 선택이 흐려짐)을 수치로 분리한다. 해결책으로는 텍스트 embedding 기반으로 라우터 입력 후보를 k=20(전체의 극히 일부)로 줄이는 shortlisting을 제안하며, tool-level 후보가 pack-level보다 더 잘 작동함을 비교 실험으로 뒷받침한다.

- **Empirical Impact**: 실험 결과, 10→110 agents로 확장될 때 단일 단계 라우팅 F1이 모델 전반에서 16–23pp 하락했는데, 감소는 주로 recall에서 발생했다. shortlisting을 적용하면 full scale에서 F1이 10–11pp 회복되며, 특히 tool-level retrieval은 platform tool search 및 hierarchical LLM routing 같은 pack-level 접근보다 일관되게 2–4pp 우위였다. 또한 1,435개 사람이 라벨링한 실제(implicit) 트래픽에서도 합성 실험의 경향이 유지되어 +10–17pp 수준의 회복을 확인했으며, 이는 대규모 도구 카탈로그 운영에서 실질적인 정책(후보 축소) 레버가 될 수 있음을 시사한다.



### FoundCause: Causal Discovery with Latent Confounders from Observational Data (https://arxiv.org/abs/2606.17516)
Comments:
          Download the model at this https URL

- **Prior Approaches**: 관측 데이터만으로 인과 그래프를 찾는 고전적 방법은 주로 조건부 독립 검정 기반(PC/FCI 계열)이나 점수 기반 탐색(GES), 연속 최적화(NOTEARS류)로 나뉘며, 대체로 데이터셋마다 별도 최적화를 수행합니다. 이 과정은 계산 비용, 하이퍼파라미터 민감도, 분포 가정 위반, 잠재 교란(latent confounding) 및 고차 구조(예: chain/collider) 처리의 어려움 때문에 실제 데이터에서 한계가 큽니다. 최근 amortized causal discovery도 있으나, 기존 뉴럴 접근은 인과 발견에서 중요한 통계적 비대칭 신호나 고차 모티프를 충분히 구조적으로 주입하지 못해 일반화와 해석 가능성에서 약점이 나타납니다.

- **Core Contribution**: FoundCause는 관측 데이터(잡음/비관측 교란 포함)를 입력으로 받아 단 한 번의 forward pass로 DAG(방향 간선)과 잠재 교란을 함께 예측하는 amortized causal discovery 모델을 제안합니다. 합성 SCM 대규모 학습으로 데이터셋별 최적화 없이도 전달 가능한 통계 패턴을 익히되, 인과 발견에 유용한 귀납 편향을 아키텍처에 명시적으로 넣는 것이 핵심입니다. 특히 잠재 교란을 latent tokens 기반 confounder module로 정면 모델링하는 점을 전면에 내세우며, 이는 기존 amortized 계열에서 상대적으로 비어 있던 지점입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 데이터에 존재하는 방향성 신호와 (2) 잠재 교란이 만드는 상관 신호를 구분하면서, (3) 누락 결측(missing data)과 고차 구조까지 한 번에 처리하는 것입니다. 이를 위해 FoundCause는 순열 불변(permutation-invariant) transformer 인코더에서 샘플-축/변수-축 attention을 교대로 수행하고, 고전적 비대칭(비대칭 통계)에서 유도한 pairwise 통계량을 statistics-conditioned attention에 주입합니다. 또한 간선 존재와 방향을 factorized decoder로 분리하고 triangular refinement 모듈로 삼중항 기반의 chain/collider 같은 고차 모티프 추론을 강화하며, learnable latent tokens + noisy-OR로 잠재 교란을 명시적으로 모델링하고 결측은 마스크 입력으로 처리합니다.

- **Empirical Impact**: 실험에서는 15개 실제 데이터셋에서 FoundCause가 11개 대표적인 고전(비-amortized) 방법과 4개 amortized 방법을 비교해 F1에서 +9.6%, AUROC에서 +1.2%를 기록했고, structural Hamming distance를 18.9% 줄이면서도 추론은 단일 forward pass로 수행합니다. 특히 “잠재 교란을 explicit하게 모델링”하는 설계가 관측 기반 인과 발견의 실용 갭을 줄이는 데 기여했다는 점이 실증적으로 드러납니다. 결과적으로 인과 발견 연구에서 amortized 접근에 통계적 인과 귀납 편향과 잠재 교란의 명시 모델링을 결합하는 방향에 중요한 기준점(benchmark)을 제시한 것으로 해석됩니다.



### Unlocking LLM Code Correction with Iterative Feedback Loops (https://arxiv.org/abs/2606.17514)
Comments:
          22 pages, 14th Computing Conference 2026

- **Prior Approaches**: 기존 LLM 코드 생성 평가는 주로 pass@1, pass@k 같은 단일 시도 정확도에 초점이 맞춰져, 실제 개발에서 핵심인 반복적 수정(회귀/리팩터링) 과정을 충분히 반영하지 못했습니다. 또한 실행 피드백이 어떤 오류 유형을 얼마나 고치게 만드는지, 그리고 모델의 reasoning 여부에 따라 효과가 어떻게 달라지는지에 대한 체계적 가이드라인이 부족했습니다.

- **Core Contribution**: 이 논문은 LLM이 컴파일/런타임 에러 메시지와 testcase 피드백을 받은 뒤 코드를 수정하는 반복 정제(iterative refinement) 능력을 체계적으로 실험 프레임워크로 정리합니다. 코드 실패를 평가하는 새로운 관점의 지표(예: ISR@k, MIS)와 오류 수정 패턴 분석을 통해, “맞히는가”를 넘어 “어떻게 개선되는가”를 정량화합니다. 또한 reasoning 모델과 non-reasoning 모델의 피드백 활용 차이를 비교해 실무 선택 기준에 가까운 통찰을 제공합니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 문제 난이도와 오류 범주(통과 못함, 시간/메모리 초과 등)가 다양한 조건에서 LLM 실패의 기준선을 잡는 것, (2) 실제 개발에 가까운 실행 기반 피드백 루프를 자동화해 재현 가능하게 만드는 것, (3) 반복 과정에서 개선이 생긴 경우만 “수정 성공”으로 일관되게 판정하는 것에 있습니다. 논문은 LeetCode 기반 다중 데이터셋과 10회 반복 제한의 자동 루프(실행→에러/테스트 피드백 삽입→재생성), 그리고 이후 반복에서 통과 testcase 수가 개선되는지로 수정 여부를 정의하며 이 문제를 해결합니다.

- **Empirical Impact**: 실험 결과, reasoning 모델(DeepSeek-R1, GPT-o4-mini)은 반복할수록 성공률이 꾸준히 상승했지만 non-reasoning 모델은 1~2회 내에서 빠르게 정체되는 경향을 보였습니다. 또한 오류 유형별로는 문법/런타임 오류가 비교적 쉽게 고쳐지는 반면(대체로 더 높은 수정 가능성), Wrong Answer와 Time Limit Exceeded 같은 논리·알고리즘 실패는 수정률이 낮아 한계가 드러났습니다. pass@1 중심 평가의 과대/과소를 보완하기 위해 ISR@k와 MIS가 실사용 복원력과 효율을 더 현실적으로 드러낸다는 점에서, 피드백 기반 코드 생성 시스템 설계에 직접적인 의미가 있습니다.



### Geometry-Aware Post-Hoc Uncertainty Quantification in Operator Learning (https://arxiv.org/abs/2606.17513)
- **Prior Approaches**: 기존 neural operator UQ는 주로 deep ensembles, MC Dropout, probabilistic variants처럼 학습 단계에서 불확실성을 만들거나, post-hoc로 Laplace approximation 같은 last-layer 중심의 가중치/국소 근사를 적용하는 방식이 많았습니다. 하지만 이러한 접근은 연산자 내부가 이미 학습해 둔 기하(geometry)별 잠재 표현을 커널/노이즈 설계에 충분히 반영하지 못해, 특히 기하 변화(geometric variability)나 분포 이동에서 불확실성 보정이 약해질 수 있습니다. 또한 GP 기반 방법은 데이터 크기·차원 증가로 스케일 문제가 커져 그대로 쓰기 어렵다는 한계가 있었습니다.

- **Core Contribution**: REEF-GP(Residual on Embedded Features Gaussian Process)는 frozen neural operator를 그대로 두고, 그 내부 embedding이 정의하는 feature space에 GP를 “잔차(residual)” 형태로 얹어 post-hoc UQ를 수행합니다. 즉, 별도의 feature map을 새로 학습하기보다 연산자가 이미 만든 좌표-기하 정렬 표현을 커널의 좌표계로 재사용해 geometry-aware uncertainty를 구성합니다. 더불어 GP의 불확실성을 단순 파라미터 불확실성 중심이 아니라, 좌표와 로컬 기하 상태에 조건화된 discrepancy로 모델링한다는 점이 핵심입니다.

- **Technical Challenges**: 가장 큰 난관은 (1) 연산자 내부 표현을 커널에 반영하되 안정적으로 학습/추론해야 하고, (2) unstructured mesh/point cloud에서 GP 스케일 한계를 피하면서도 calibrated uncertainty를 얻는 것입니다. 논문은 spectral-normalized projections로 feature collapse를 완화하고, geometry-aware heteroscedastic noise로 지역별 설명되지 않은 변동성을 흡수하며, stochastic subset 최적화와 gPoE(Product of Experts)로 대규모 학습·추론을 가능하게 했습니다. 또한 저차원 low-rank 근사에 의존하지 않도록 설계해 geometry가 복잡하게 달라지는 상황에서도 커널 유연성을 유지하려고 했습니다.

- **Empirical Impact**: 다섯 개의 2D/3D PDE 벤치마크에서 REEF-GP는 예측 정확도를 유지하면서도 uncertainty calibration이 deep ensembles에 경쟁적이되, 계산 비용은 훨씬 적게 들었다고 보고합니다. 특히 geometric distribution shift에서도 robust하며, 불확실성이 shock front 같은 물리적으로 의미 있는 영역에 집중되는 등 “맞는 곳에 자신감(불확실성)”을 주는 경향을 보였습니다. 결과적으로 neural operator용 post-hoc UQ를 parameter-centric이 아니라 learned feature space 중심으로 확장할 수 있다는 실용적 대안을 제시했다는 점에서 의미가 있습니다.



### MagicSim: A Unified Infrastructure for Executable Embodied Interaction (https://arxiv.org/abs/2606.17511)
- **Prior Approaches**: 기존 로봇 시뮬레이션은 주로 렌더링/컨트롤 테스트베드이거나, 특정 고정 작업 환경처럼 분리돼 활용되는 경우가 많습니다. 장기·상호작용 과업은 평가를 위해 ‘magic’ 액션으로 우회하거나, 학습/수집용 환경과 계획/상태가 이어지지 않아 같은 에피소드 재현·주석·검증이 어렵다는 한계가 있었습니다. 또한 플래닝을 외부 오프라인 전처리로 두면, 실패 재현이나 플래닝-루프 안에서의 실행 기록을 얻기 힘들었습니다.

- **Core Contribution**: MagicSim은 에피소드를 단위로 삼는 embodied interaction infrastructure로, 동일한 결정적(deterministic) 배치 런타임과 단일 MDP 위에서 world 구성, 실행, 평가, 데이터 수집, 에이전트 상호작용을 한 번에 잇습니다. YAML-first 명세로 콘텐츠·배치·행동·에이전트 노출을 분리해, 강체부터 유체/연성, 다양한 로봇 embodiment까지 폭넓은 실행 가능한 월드를 reset-and-step 루프에서 생성합니다. 또한 고수준 커맨드를 simulator 내부 상태 편집이 아닌 controller/atomicskills/플래너 프리미티브를 거쳐 로봇 액션으로 접지(grounding)합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 한 런타임에서 이질적인 물리(강체·연성·유체·지형·아바타·센서)를 공존시키는 월드 계약, (2) 병렬 대규모에서 재현 가능한 결정성(리셋 순서, snapshot, reset_to), (3) 단일 action space로 줄이기 어려운 multi-embodiment 실행, (4) planner-in-the-loop를 배치 시뮬레이션이 멈추지 않게 비동기 처리하는 구조, (5) 어디서/어떻게 잡았는지 같은 상호작용 인과구조를 보존하는 annotation-rich 데이터 생성입니다. MagicSim은 batched runtime에서 env별 독립 라이프사이클과 비동기 플래너 마이크배치 마진(microbatch) 해결, 그리고 성공 게이팅 기반의 에피소드 단위 저장을 조합해 이를 해결합니다.

- **Empirical Impact**: 논문은 MagicSim이 단일 태스크 정의로 RL 벤치마크 평가, autocollect용 자동 궤적 수집, VLM/에이전트용 상호작용 인터페이스를 동시에 지원해 연구 파이프라인을 통합할 수 있음을 강조합니다. 특히 성공한 롤아웃만 구조화된 멀티모달 트래젝토리로 저장해 언어 감독, 시각/기하 표현, 스킬·플래너·액션의 정렬을 제공함으로써 데이터 품질과 재사용성을 높입니다. 결과적으로 로봇 학습·데이터 생성·에이전트 구동을 하나의 재현 가능한 planner-in-the-loop 실행 기반으로 묶는 방향의 실용적 인프라로 의미가 큽니다.



### Online LLM Selection via Constrained Bandits with Time-Varying Demand (https://arxiv.org/abs/2606.17489)
Comments:
          11 pages, 3 figures with multiple subfigures, 1 table, submitted for possible journal publication

- **Prior Approaches**: 기존 LLM 추론 선택·오프로딩 최적화는 특정 모델을 고정(static selection)하거나 사전 벤치마크 점수로 오프라인 결정하는 경우가 많아 입력/프롬프트 다양성과 시간에 따른 성능 변화에 약합니다. 온라인 학습 기반 bandit 접근도 있었지만, 보통 단일 제약(예: 예산)만 다루거나 packing/covering 제약을 함께 다루지 못했습니다. 또한 토큰 기반 비용·지연 SLA처럼 현실의 자원 제약을 동시에 만족시키면서 시간 가변 수요까지 반영한 연구는 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 엣지-클라우드 협업 추론에서 작업이 시간에 따라 달라질 때, 모델 풀 중 어떤 LLM을 선택할지 온라인으로 결정하는 문제를 ‘제약(stochastic) 밴딧’으로 정식화합니다. 하드 자원 제약(토큰 비용 예산의 packing)과 소프트 서비스 제약(지연 SLA의 covering)을 동시에 다루며, 실제 배포에서 흔한 경성·연성 조건을 한 프레임에 묶었습니다. 더불어 과거 데이터나 예측기로부터 수요(작업량) 변동을 미리 반영해 결정을 안정화합니다.

- **Technical Challenges**: 핵심 난점은 (1) 보상·지연·토큰 사용의 분포를 모르고, (2) 일부 피드백만 관측되며, (3) 수요 qt가 시간에 따라 변하는데도 장기 제약을 동시에 만족해야 한다는 점입니다. 이를 위해 COPAC-UCB는 reward에는 UCB(낙관), packing 제약 비용에는 LCB(비관), covering 제약에는 UCB(낙관)를 적용하는 confidence-bound 설계를 사용합니다. 또한 Lagrangian 기반 dual 변수(virtual prices)를 온라인으로 갱신해 예산 초과나 SLA 저하 위험을 줄이면서, 블랙박스 예측으로 누적 수요를 추정해 자원 정규화를 수행합니다.

- **Empirical Impact**: 이론적으로는 오프라인 전체정보 벤치마크 대비 regret이 sublinear임을 보이고, covering 제약 위반도 평균적으로 sublinear 수준으로 제한된다는 보장(고확률)을 제시합니다. 실험에서는 합성 워크로드에서 수요가 동적으로 변하고 자원 제약이 빡빡한 환경에서도 COPAC-UCB가 정확도/유용성은 높게 유지하면서 지연 SLA 준수와 예산 안전성을 동시에 달성함을 보여줍니다. 결과적으로 엣지-클라우드 LLM 운영에서 ‘동적 모델 선택 + 다중 제약 + 예측 기반 적응’의 실용적 기준을 제공하는 데 의미가 있습니다.



### Decoding Hidden Deception in Reasoning LLMs: Activation Explainers for Deception Auditing (https://arxiv.org/abs/2606.17478)
Comments:
          Under review

- **Prior Approaches**: 기존 deception monitors는 주로 (1) 가시적인 대화/응답 텍스트에 점수를 매기거나, (2) representation vector에서 추출한 scalar probe score로 의심도를 판단합니다. 그러나 이 방식은 왜 의심스러운지에 대한 “검증 가능한 근거”가 부족해, 감사(auditing) 과정에서 해석 가능성이 제한됩니다. 또한 설명 단서가 단일 점수에 갇혀 있어 인력 검토로 이어지기 어렵다는 문제가 있습니다.

- **Core Contribution**: 이 논문은 STATEWITNESS라는 activation explainer를 제안합니다. 타깃 LLM의 hidden states를 읽는 별도 decoder가 natural-language 질의에 답하거나, 구조화된 보고서를 생성해 “의심의 근거”를 더 구체적으로 제공합니다. 즉, deception 탐지 단계를 넘어 감사자가 확인할 수 있는 설명 가능한 인터페이스를 제공하는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술 과제는 hidden states에서 deception 관련 신호를 안정적으로 포착하면서도, 사람에게 유용한 형태의 근거(질의 응답·스키마 보고서·evidence trace)로 변환하는 것입니다. 저자들은 별도 decoder를 두어 hidden states를 입력으로 자연어/구조화 출력 모두를 학습하고, token- 또는 sentence-level evidence trace로 근거를 추적 가능하게 했습니다. 이를 통해 단일 scalar 감지의 한계를 설명 가능한 진단으로 확장했습니다.

- **Empirical Impact**: 평가는 2개의 reasoning LLM을 대상으로 7개 deception 데이터셋에서 수행됐고, STATEWITNESS는 mean AUROC 0.916을 기록했습니다. 이는 동일 프로토콜에서 black-box 텍스트 모니터 대비 상대 11.6% 향상, activation-probe baseline 대비 상대 25.0% 향상을 의미합니다. 또한 기존 모니터와 함께 쓰는 threshold ensemble에서 missed deceptive examples를 줄였으며, 사람 검토를 위한 근거 출력까지 제공해 정렬(alignment) 및 interpretability 도구로 확장될 가능성을 보여줍니다.



### AIPatient Arena: EHR-grounded evaluation of large language models in end-to-end clinical consultation workflows (https://arxiv.org/abs/2606.17474)
Comments:
          49 pages, 12 figues, 11 tables

- **Prior Approaches**: 기존 의학 LLM 평가는 대부분 정적이거나 단일 턴, 또는 특정 결과 중심으로 이뤄져 실제 진료의 순차성·불확실성·상호작용을 충분히 반영하지 못했습니다. 또한 최종 답변 정확도만 보려는 경향이 강해, 환자 발화에서 정보를 어떻게 수집하고 해석하며 설명하는지 같은 과정 평가는 상대적으로 약했습니다.

- **Core Contribution**: 이 논문은 EHR(전자건강기록) 기반의 임상 유틸리티 평가 프레임워크인 AIPatient Arena를 제안합니다. 환자별 지식 그래프를 구성해 다중 턴 의사-환자 상호작용을 평가하고, 임상 역량을 8개 차원(예: 질문 기술, 윤리/전문성, 설명의 명확성·투명성, 진단 추론 등)으로 체계화합니다.

- **Technical Challenges**: 핵심 난제는 모델이 진료 대화 중 불명확한 환자 답변을 처리하고, 과거 병력을 포함한 필요한 정보를 빠짐없이 커버하며, 불확실성을 적절히 다루는지 평가하는 데 있습니다. 저자들은 EHR-기반 지식 그래프와 프로세스 중심의 다중 턴 평가로 상호작용 실패(반복 질문, 병력 누락, 불확실성 취급 미흡 등)를 관찰 가능하게 만들고, 풍부한 대화 맥락이 진단 추론에는 도움이 되지만 치료 계획 개선은 제한적임을 함께 확인합니다.

- **Empirical Impact**: 437명의 1차 코호트와 두 개의 out-of-distribution 검증 코호트(각 119·67명)에서 LLM은 질문 기술(QS), 윤리·전문성(ET), 임상 설명의 명확성·투명성(EX)에서 대체로 높은 점수(대략 4점대)를 보였습니다. 반면 모호한 답변 처리(HR), 정보 커버리지(IC), 진단 정확도·추론(Dx)에서 약점이 지속됐고, 과정 기반 평가가 최종 답변 정확도만으로는 임상 준비도(readiness)를 판단하기 어렵다는 메시지를 강화합니다. 이 결과는 배포 전 medical LLM을 workflow 단위로 검증하는 표준에 가까운 방향성을 제시합니다.



### AUTOGATE: Automated Clock Gating via Toggling-Aware LLM-based RTL Rewriting (https://arxiv.org/abs/2606.17461)
Comments:
          9 pages, 6 figures, 7 tables

- **Prior Approaches**: 기존 FGCG 최적화는 합성 도구가 RTL 코딩 패턴에서 드러난 clock-gating 기회를 추론하는 방식이 많지만, workload에 따라 나타나는 기회는 RTL 재구성이 필요해 ‘수동’에 의존하는 경우가 큽니다. LLM 기반 RTL 최적화는 가능성이 있으나, 수백만 사이클의 long waveform traces를 그대로 처리하기 어렵고(컨텍스트 한계), 대규모 계층형 코드베이스에서 모듈 간 의존성과 정합성을 유지하며 확장하기도 난점이 있습니다. 또한 비-LLM 접근(룰 기반, clustering 기반)도 고정된 규칙/수동 튜닝 의존으로 산업 규모에서 재현성과 일반화에 제약이 있었습니다.

- **Core Contribution**: AUTOGATE는 산업 수준의 RTL power optimization을 위한 최초의 agentic 프레임워크로, workload-aware FGCG를 large hierarchical codebase에 적용하는 방법을 제시합니다. 핵심은 ML-LLM co-design으로, LLM이 raw waveform을 직접 읽지 않고도 clock-gating rewriting을 수행하도록 ML 기반 toggling trace 압축(클러스터링·구조화된 표현)을 LLM에 ‘가이드 데이터’로 제공한다는 점입니다. 여기에 계층형 multi-agent 아키텍처를 더해 모듈 분할-협조 최적화를 통해 correctness를 유지하며 스케일을 확보합니다.

- **Technical Challenges**: FGCG의 cycle-accurate switching 분석은 입력 자극과 tightly coupled이라 long waveform(수백만 클럭)을 다뤄야 하지만, 이를 LLM 컨텍스트에 넣을 수 없다는 문제가 있었고 AUTOGATE는 toggle-aware pre-filtering, 자동 threshold discovery, multi-threshold stability clustering으로 파형을 compact structured representation으로 변환해 해결합니다. 또 대규모 계층형 RTL에서 단일 에이전트·단일 컨텍스트는 확장성이 떨어지므로, 설계 트리에서 hierarchy를 구성한 뒤 병목 모듈에 대해 리라이팅 에이전트를 분할 실행하고 bottom-up으로 통합하는 divide-and-conquer를 적용합니다. 생성된 RTL 변경은 문법 점검, 시뮬레이션, commercial formal equivalence checking, 타이밍 검증을 거쳐 QoR(동적전력·면적·WNS/TNS)을 기준으로 선택합니다.

- **Empirical Impact**: 실험 결과 AUTOGATE는 비교 기준선 대비 dynamic power를 일관되게 낮췄고, small 디자인 세트에서는 평균 49.31% 동적전력 감소를 달성했습니다. NVDLA에서는 파티션별로 19.34%(Workload 전반), BlackParrot에서는 7.96% 감소를 보였으며, 면적 오버헤드는 매우 낮게(예: 0.04%) 유지되는 경향을 보였습니다. 또한 LLM·룰 기반 경쟁 방법들은 long trace 처리 한계나 workload 비고려로 대규모 계층형에서 개선 폭이 제한된 반면, AUTOGATE는 계층·워크로드를 모두 고려한 분석-리라이팅 루프로 파워-면적 트레이드오프를 확보했다는 점에서 의미가 큽니다.



### MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation (https://arxiv.org/abs/2606.17449)
Comments:
          To be presented at ACL 2026

- **Prior Approaches**: 기존 M-RAG는 retrieval-augmented generation 흐름에서 정적 파이프라인과 유사도 기반 필터링에 의존해, 시각-텍스트 충돌을 분리·판단하기 어렵다. 그 결과 cross-modal hallucination, causal fabrication, sycophancy가 자주 발생하며, 수정용 룰을 일괄 적용하면 정확한 생성까지 과도하게 깨지는 ‘intervention paradox’가 이어진다. 또 가벼운 LLM의 무가이드 다단 추론은 포맷 불안정으로 구조적 실패가 연쇄되며 논리적 드리프트를 키운다.

- **Core Contribution**: 이 논문은 MODE-RAG(Multimodal Objective Diagnostic Energy-RAG)로, Variational Free Energy(VFE)와 내부 attention states(ATLAS)를 이용해 개입 필요성을 동적으로 게이팅한다. FE-Router가 uncertainty가 높은 질의만 전문 멀티에이전트 파이프라인으로 라우팅하고, 나머지는 우회해 과잉 교정으로 인한 정확도 저하를 막는다. 또한 단계별(인식·검색·추론·생성)로 원인에 대응하는 에이전트를 두고, logit perturbation과 overseer 검증으로 sycophancy·논리적 조작·포맷 붕괴를 억제한다.

- **Technical Challenges**: 핵심 난제는 ‘언제 얼마나’ 개입해야 하는지이며, 정적 룰은 과잉 교정, 무가이드 추론은 실패 연쇄를 낳는다는 점이다. MODE-RAG는 VFE 기반 FE-Router로 고위험(Claim-Scene 충돌 등) 신호를 감지해 개입을 선택하고, Per-Agent의 atomic visual fact 추출로 ‘visual-first’ 앵커를 고정한다. 추론 단계에서는 Monte Carlo Tree Search(MCTS)로 인과 DAG를 구성해 temporal inversion/forced causality를 줄이고, Gen-Agent의 logit perturbation 및 overseer의 삼중 일관성 검사로 사용자 편향에 대한 과적합을 페널티한다.

- **Empirical Impact**: 평가를 위해 ModeVent를 제안하며, MultiVent에서 VFE 상·하위(불확실성 극단) 샘플을 골라 retrieval-시각 충돌과 manifold outlier에 강하게 테스트한다. Qwen-2.5-VL-7B 베이스라인 대비 MODE-RAG는 전체 평균 fidelity/resilience에서 일관된 개선을 보였고, 특히 Outliers에서 attention hijacking·majority text bias·out-of-domain irrelevance 같은 극단 실패를 크게 완화했다. 반면 비용은 질의당 처리시간이 평균 18.5초→26.2초(약 1.42×)로 증가하지만, 단계별 에이전트 개입 구조 덕분에 병렬화로 상쇄 가능성을 제시한다.



### Patients With Personality: Realistic Patient Simulation through Controlled Diversity and Selective Disclosur (https://arxiv.org/abs/2606.17441)
Comments:
          22 pages, 11 figures

- **Prior Approaches**: 기존 LLM 기반 의료용 시뮬레이티드 환자(simulated patient)는 프롬프트 중심이거나(state machine/자율 생성) 일부 축(personality·언어능력·기억 등)을 파라미터로 노출하는 방식이 주류였습니다. 하지만 대체로 (1) 질문되지 않은 정보까지 과잉 제공하는 oversharing, (2) 실제 환자처럼 완벽히 협조적이지 않거나 성격에 따른 변동이 적은 행동 균일성 문제가 남아 벤치마크 타당도를 흔들었습니다.

- **Core Contribution**: 본 논문은 PatientsWithPersonality(PWP)라는 환자 시뮬레이션 프레임워크를 제안하며, 환자의 성격(personality)과 정보 공개(disclosure)를 분리해 제어합니다. HEXACO(정직-겸손, 정서성, 외향성, 호감성, 성실성, 개방성) 축을 기반으로 잠재 환자 상태(latent patient state)를 구성하고, 질문에 따라 필요한 정보만 드러나도록 query-conditioned disclosure grid로 on-demand 공개를 강제합니다.

- **Technical Challenges**: 핵심 기술 난제는 “현실적인 대화 내용 생성”과 “원치 않는 정보 누출 방지 및 성격 변동성 유지”를 동시에 만족시키는 것이었습니다. PWP는 메타 LLM로 사실 변형(예: downplay/fuzzy/denied 등)과 대화 역할을 만들고, 대화 LLM은 해당 턴에서 요구된 필드 Rt만 프롬프트에 포함시켜 oversharing을 구조적으로 차단하며, 성격 축이 대화 양식/기억·공개 패턴에 연결되도록 설계했습니다.

- **Empirical Impact**: 임상의(레지던트) 평가에서 PWP는 사람 배우(recorded human actor)에 거의 근접한 수준의 realism을 보였고, 다른 시뮬레이터 대비 realism 점수가 더 높았습니다. 특히 “too informative” 플래그가 절반가량 낮게 나타나 과잉 정보 제공 실패 모드를 실질적으로 줄였으며, 성격 회복 가능성(임상의/autorater가 HEXACO 레벨 추정)과 대화 다양성 측면에서도 가장 넓은 행동 범위를 보여 의료 LLM 벤치마킹의 신뢰도를 높일 잠재력이 확인됐습니다.



### Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos (https://arxiv.org/abs/2606.17437)
- **Prior Approaches**: 기존 연구는 주로 단일 프레임(이미지 레벨) 또는 소규모 데이터에서의 분류에 집중해 왔고, 최근에는 2-stream/CNN-LSTM, 3D CNN, Transformer 등으로 video-level fusion을 확장했다. 하지만 공개 데이터셋은 규모·뷰 커버리지가 작아 재현성과 공정 비교가 어렵고, 다양한 연구들이 뷰 정의·분할·평가지표를 달리해 현대 아키텍처를 체계적으로 벤치마킹하기가 힘들다. 또한 일부 뷰는 공간적 외형이 매우 유사해 단일 프레임 특징만으로는 구분이 어렵고, 초음파 영상의 프레임 품질이 들쭉날쭉해 신뢰도 없는 샘플이 시간적 집합 과정에서 오염될 수 있다.

- **Core Contribution**: 이 논문은 EV9V(Echocardiographic Videos of Nine Views) 데이터셋을 공개하며, 표준 9개 심초음파 뷰를 총 5,138개 비디오(910,579 프레임)로 제공해 공개형 중 최대 규모로 포지셔닝한다. 더불어 EV9V 위에서 CNN/RNN/Transformer를 포함한 대표 video classification 모델들을 체계적으로 벤치마크해, 최신 비디오 아키텍처의 성능과 한계를 같은 기준에서 비교할 수 있게 했다. 마지막으로 STFM(Spatio-Temporal Fusion Model)을 제안해 공간(해부학)과 시간(심장 동역학)을 효율적으로 결합하고, 프레임 품질 변동에 강한 추론을 목표로 한다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 유사 외형 뷰를 구분하기 위해 공간-시간 정보를 안정적으로 융합해야 하고, (2) 흐림·과도한 움직임·전이 구간처럼 비대표 프레임이 존재할 때 이를 비디오 수준 의사결정에서 덜 반영해야 한다는 점이다. 논문은 공유 얕은 CNN stem 위에 spatial branch(중심 프레임의 해부학 임베딩)와 temporal branch(희소 샘플 클립을 CNN-LSTM으로 처리)를 두는 이중 스트림 구조로 계산 효율을 유지하면서도 시간 패턴을 학습한다. 여기에 Re-EDL 기반 evidential(Dirichlet) 불확실성 모델링과 증거(evidence) 기반 fusion을 결합해, 불확실성이 큰 관측이 최종 융합에서 상대적으로 덜 영향 주도록 설계했다.

- **Empirical Impact**: EV9V에서 다양한 현대 비디오 아키텍처와의 비교 실험을 통해 STFM이 여러 모델 전반에서 경쟁력 있는 성능을 보이며, 불확실성 인지 spatio-temporal 학습이 심초음파 뷰 분류의 견고성에 실제로 기여함을 확인한다. 특히 비디오 내 품질 편차가 큰 임상 상황에서, 불확실성을 활용한 샘플 선택 및 증거 융합 전략이 비대표 구간의 오염을 줄여 분류 신뢰도를 높이는 방향으로 효과가 나타난다. 코드 공개까지 병행해, 이후 연구자들이 EV9V 기반으로 재현 가능한 벤치마킹과 후속 모델 개선을 빠르게 수행할 수 있는 기반을 제공한다.



### Feynman Kac Reweighted Schrödinger Bridge Matching for Surface-Based Tau PET Harmonization (https://arxiv.org/abs/2606.17420)
- **Prior Approaches**: 다중 사이트 tau PET에서 스캐너·프로토콜·방사성동위원소 차이가 생물학적 원인이 아닌 변동(잡음/강도·대비·노이즈 차이)을 키워, 바이오마커 분산 증가와 질병 효과 민감도 저하, 심하면 편향을 유발한다. 기존 비쌍(pairless) harmonization은 ComBat처럼 선형 가정을 쓰거나, CycleGAN·disentanglement·diffusion 계열은 비정렬 학습의 불안정/모드 붕괴/노이즈 캘리브레이션 트레이드오프 같은 한계가 있다. 특히 Schrödinger Bridge 기반 접근(DSBM류)은 소스-타깃 엔드포인트를 π0⊗π1에서 독립 샘플링해, 사이트 간 subgroup(예: tau-positivity) 비율 차이를 무시하면 사이트 효과가 생물학적 차이로 혼입될 위험이 남아 있다.

- **Core Contribution**: 이 논문은 Feynman Kac Reweighted Schrödinger Bridge Matching(FKRSBM)으로 비쌍 tau PET harmonization에서 subgroup 조성 차이로 인한 “생물학-사이트 혼동” 문제를 겨냥한다. 기존 SB/DSBM이 학습하는 stochastic transport 경로는 유지하되, 엔드포인트 제안을 Feynman–Kac 재가중으로 subgroup-aware하게 바꿔 생물학적으로 일관된 대응만 더 자주 생성되도록 제어한다. 또한 이 재가중은 데이터 샘플링(importance sampling) 수준에서만 구현되어, solver나 네트워크 아키텍처 변경 없이 모듈처럼 끼워 넣을 수 있다고 강조한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 비유클리드(non-Euclidean) cortical mesh 위에서 SB 기반 harmonization을 수행할 수 있는 기하-aware 구조가 필요하다는 점과, (2) SB의 엔드포인트 결합이 subgroup 라벨 구조를 무시해 편향을 만들지 않게 제약하는 방법을 찾아야 한다는 점이다. FKRSBM은 spherical convolution backbone으로 cortical topology를 존중하면서, 엔드포인트 penalty Φ(예: tau-positivity mismatch)를 KL-regularized SB 목적에 포함하고 이를 Feynman–Kac/Radon–Nikodým 재가중으로 등가 변환해 “조건부 엔드포인트 제안 변경=표준 bridge matching 학습”으로 구현한다. 결과적으로 Brownian bridge의 조건부 역학은 그대로 두고, 어떤 엔드포인트 쌍을 더 샘플링할지만 바꿔 subgroup 정합성을 강제한다.

- **Empirical Impact**: PI-2620(HABS-HD) → AV-1451(ADNI)로 unpaired tau PET SUVR을 harmonize한 실험에서 FKRSBM은 ComBat, CycleGAN, diffusion 기반 방법(DF), unregularized DSBM 대비 더 강한 분포 정렬과 잔여 사이트 분리도 감소를 보였다. 특히 tau-positivity sign mismatch가 더 낮고, APOE subgroup 정렬도 더 잘 맞았으며, downstream 질병 분류 성능까지 개선됐다. 즉, 이 접근은 단순 통계적 맞춤을 넘어 subgroup 라벨 혼입 편향을 줄이는 방향으로 실제 임상 추론 품질 향상에 연결된다는 점에서 의미가 있다.



### L-Proto: Language-Aware Episodic Prototypical Training for Multilingual Speaker Verification (https://arxiv.org/abs/2606.17416)
Comments:
          Accepted by INTERSPEECH 2026

- **Prior Approaches**: 다국어 화자 검증(SV)에서는 언어별 음소/운율 차이 때문에 화자 특징이 언어 특징과 얽혀 임베딩이 언어별 서브클러스터로 분리되는 문제가 자주 발생한다. 기존 해결책들은 언어 불변 표현을 유도하는 adversarial 학습, speaker–language 분리를 위한 목적함수/contrastive learning, meta-learning이나 분포강건 최적화 같은 domain generalization, 그리고 전역 점수 보정에 초점을 둔다. 다만 이들은 대체로 전체 다국어 데이터에 대한 global objective 중심이라, episodic prototypical 학습에서 언어 혼합이 prototype 추정에 미치는 편향을 직접 제어하지 못한다.

- **Core Contribution**: 이 논문은 episodic prototypical training에서 에피소드 내 언어 구성을 제어하는 언어 인지 episodic prototypical 학습전략 L-Proto를 제안한다. 핵심 아이디어는 한 에피소드에 단일 언어만 포함시켜, prototype 기반 유사도 학습에서 언어 변동을 줄이고 화자 정체성에 더 집중하게 만드는 것이다. 또한 학습 반복마다 언어를 바꿔 다국어 노출은 유지하면서, 에피소드 내부의 speaker–language entanglement을 완화한다.

- **Technical Challenges**: L-Proto의 가장 큰 난관은 다국어 환경에서 랜덤 에피소드를 구성하면 같은 화자라도 서로 다른 언어 데이터가 섞여 prototype 추정이 흔들리고 similarity supervision의 신뢰도가 떨어질 수 있다는 점이다. 이를 위해 저자들은 언어별로 speaker별 utterance를 버퍼링하고, 각 언어에서 충분한 수의 “ready speaker”를 모을 때만 에피소드를 생성하는 streaming sampling으로 단일 언어 일관성을 강제한다. 학습은 언어 일관 에피소드에서 cosine similarity로 prototype을 만들고, query가 정답 speaker prototype에 가까워지도록 episodic loss를 cross-entropy 형태로 적용하며, 전역 분류 손실과 가중합해 end-to-end로 최적화한다.

- **Empirical Impact**: TidyVoice Challenge 벤치마크에서 SimAM-ResNet34/100 등 여러 backbone에 대해 L-Proto가 fine-tuning과 random episodic sampling 대비 EER·minDCF를 일관되게 개선한다. 특히 교차언어 trial(D/D, D/S)에서 이득이 두드러져, 언어 혼합이 prototype 기반 판별을 왜곡한다는 문제의식과 정합적이다. 임베딩 분포/코사인 유사도 분석에서도 L-Proto는 같은 화자의 언어 간 유사도는 높이고 화자 간 유사도는 낮춰 더 큰 separation margin을 만들며, 다국어 정합성(calibration) 측면에서도 대체로 유리한 경향을 보인다.



### Enhancing Pathological VLMs with Cross-scale Reasoning (https://arxiv.org/abs/2606.17412)
- **Prior Approaches**: 기존 병리 비전-언어 모델(VLM)들은 대부분 단일 배율 고정 이미지에서의 VQA/분류로 학습·평가돼 임상에서의 다배율 추론 흐름을 반영하지 못했습니다. 또한 멀티스케일 데이터가 있더라도 ‘저배율 조직 아키텍처’와 ‘고배율 세포 형태’를 연결해 근거를 통합하는 명시적 목표가 약해, 모델이 배율별 단독 인식에 머무는 문제가 컸습니다. 더 나아가 멀티이미지 VQA를 그대로 만들면 text-only shortcut(질문·보기만으로 정답 추론)을 통해 과대평가될 위험도 제기됩니다.

- **Core Contribution**: 이 논문은 병리 해석을 다배율(10x/40x/200x) 근거를 함께 엮는 multi-magnification reasoning으로 재정의한 ‘cross-scale training and evaluation paradigm’을 제안합니다. 그 결과 Scale-VQA라는 다배율 추론 벤치마크를 구축하고, ScaleReasoner-R1을 cross-scale VQA에 맞춰 학습해 단일 배율 벤치마크 성능까지 함께 끌어올리는 전이를 보여줍니다.

- **Technical Challenges**: 핵심 난제는 멀티이미지 VQA에서 발생하는 text-only shortcut인데, 배율과 강하게 연동된 단서나 문장/보기의 언어적 priors가 정답을 이미지 없이도 맞히게 만들 수 있습니다. 이를 위해 leakage-aware curation pipeline을 설계했으며, 텍스트만으로 정답을 맞히는지 Gemini 3 Pro·Qwen3-Max 같은 text-only adversary로 반복 스크리닝하고, 근거가 최소 두 배율 뷰에 의존하도록 제약(visual-grounding·scale-dependency·차원별 다양성)을 교정해 누설을 억제합니다. 이후 RL 학습은 GRPO로 outcome-driven 보상(정답 정확도) 중심으로 구성해, 인간이 쓴 rationales를 그대로 모방하는 방식의 과적합(특히 SFT 계열의 단일 스케일 망각)을 줄이려 했습니다.

- **Empirical Impact**: Scale-VQA는 2,537개 병리 이미지에서 4,685개의 MCQ를 만들며, 배율 간 증거 통합을 요구하는 설계로 cross-scale 추론을 직접 측정합니다. ScaleReasoner-R1은 Scale-VQA-Test에서 평균 82.89%로 cross-scale SOTA를 달성했고, Scale-VQA만으로도 PathMMU의 단일 배율 벤치마크 성능이 전반적으로 개선되는 전이를 보였습니다. 특히 RL-only가 SFT-only 및 SFT+RL보다 일관되게 우수했는데, 이는 SFT의 demonstration-style 과적합 대신 정답 보상 기반 최적화가 일반화를 이끈다는 메시지를 강화합니다.



### Discrete Autoregressive Transformer for Generative Mechanism Synthesis (https://arxiv.org/abs/2606.17409)
- **Prior Approaches**: 기존 경로( coupler curve ) 합성은 정밀점들을 목표로 놓고 4-bar 같은 제한 토폴로지에서만 해를 얻거나, 수치 최적화로 근사해왔다. 이런 방식은 초기값 민감, 계산 비용, 수렴 보장 부족이 크고, 기본적으로 한 설계 과제당 단일 메커니즘만 내는 경향이 있어 원래 문제의 one-to-many 구조를 잘 반영하지 못한다. 학습 기반 접근은 대체로 입력→파라미터의 단일값 회귀/생성에 머물러 topology를 고정하거나 다양성 제어가 약하다는 한계를 보였다.

- **Core Contribution**: 이 논문은 conditional autoregressive sequence modeling으로 경로 합성을 “여러 메커니즘 패밀리”를 함께 생성하는 문제로 재정의한다. 특히 50차원 VAE latent(목표 곡선) + explicit mechanism-type token(토폴로지)을 조건으로, 디코더-only transformer가 quantized joint coordinates를 토큰 시퀀스로 생성해 다중 토폴로지(4/6/8-bar)를 한 모델에서 다룬다. 또한 inference에서 latent-noise schedule을 돌려 모든 타입을 동시에 탐색하고, 기하 오차 기준 상위 5개 후보를 뽑아 dataset lookup 없이도 다양한 해 집합을 만든다.

- **Technical Challenges**: 핵심 난점은 (1) one-to-many 제약을 만족하는 생성 다양성과 (2) 생성 후보가 실제 기구로 시뮬레이션했을 때 목표 경로와 얼마나 잘 맞는지의 공정한 평가를 동시에 맞추는 것이다. 이를 위해 좌표를 uniform binning으로 토큰화해 transformer의 autoregressive 학습을 가능하게 하고, 손실은 token cross-entropy에 더해 bin의 ordinal 구조를 반영하는 Gaussian-smoothed auxiliary loss를 결합했다. 평가 또한 forward kinematics 후 arc-length resampling, centering, O(2) 정렬(회전/미러 탐색), squared symmetric Chamfer 및 banded DTW로 “시뮬레이션 기반 단일 계약”을 고정해 모델 간 비교 가능성을 확보했다.

- **Empirical Impact**: held-out 테스트에서 mean Chamfer distance(CD)는 0.0132, mean dynamic time warping(DTW)는 0.153으로 보고되며, noise sweep+top5 저장 프로토콜이 matched-topology 기준 성능을 유지하면서도 타입 다양성을 늘리는 것으로 나타났다. 또한 VAE latent space에서 k-nearest-neighbor 기준점으로 조건을 주는 라벨-제한 baseline에서도 decoder는 동일하게 쓰되, 논문 방법은 생성 시점에 학습 데이터 인접 코드에 의존하지 않고도 경쟁력 있는 CD/DTW를 달성한다. 결과적으로 “단일 경로 입력→정확하고 다양한 메커니즘 패밀리”를 시뮬레이션으로 검증 가능한 형태로 제시해, 로보틱스용 경로-기구 설계 파이프라인의 실용성에 의미 있는 진전을 보인다.



### Graph Neural Networks for Semi-Supervised Image Classification with Multi-Feature Aggregation (https://arxiv.org/abs/2606.17406)
- **Prior Approaches**: 기존 반지도/비지도 이미지 분류 연구는 라벨이 적을 때 성능을 내기 위해 특징 추출기(CNN, Vision Transformer 등)와 그래프 기반 GNN(주로 GCN)을 결합해왔다. 다만 이미지 데이터에서는 그래프가 사전에 주어지지 않아, 그래프 구성(유사도 계산·kNN·재랭킹)의 품질이 결과를 크게 좌우한다. 또한 서로 다른 특징 추출기에서 나온 표현을 어떻게 효과적으로 합칠지(early/late fusion 또는 랭킹 통합)는 여전히 결합 전략 의존성이 크다는 한계가 있다.

- **Core Contribution**: 이 논문은 라벨이 scarce한 반지도 이미지 분류에서, 여러 특징 추출기에서 얻은 feature와 그래프 표현을 동시에 통합하는 GNN 접근을 제안한다. 특히 multi-feature setting에서 각 특징의 랭킹을 UDLF 기반 rank aggregation으로 합쳐 하나의 reciprocal kNN 그래프를 만들고, node feature는 Unsupervised Relief(URelief)로 저차원 성분을 뽑아 연결한다. 추가로 manifold learning 계열의 재랭킹/유사도 학습을 그래프 전처리로 적용해 분류 정확도를 전반적으로 끌어올리는 것을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 특징 공간에서의 유사도를 어떻게 일관된 그래프 구조로 바꿀지, (2) 그래프가 부정확하면 GNN 전파(smoothing)가 망가질 수 있다는 점, (3) 고차원 feature를 그대로 쓰기 어렵다는 계산·과적합 문제다. 논문은 ranked list를 먼저 만들고 BFSTree, RDPAC, LHRR 같은 manifold learning/UDLF 재랭킹을 거쳐 그래프 품질을 개선하며, multi-feature에서는 reciprocal 제약으로 그래프에서의 상호 일치성을 강화한다. 또한 각 descriptor에 URelief로 200개 특징만 선택해 저차원으로 줄인 뒤 concatenation으로 fused node feature를 구성한다.

- **Empirical Impact**: 실험 결과, manifold learning 기반 그래프 처리와 함께 feature·graph를 전략적으로 결합하면 대부분의 조건에서 분류 정확도가 유의미하게 향상된다. 특히 여러 추출기에서 나온 feature를 rank aggregation으로 통합했을 때 성능이 추가로 좋아지는 경향이 관찰됐다. 한편 분석에서는 GCN이 feature보다 입력 그래프 품질에 더 민감하다는 점을 보여주며, 라벨이 적은 설정에서 성능 향상의 주된 레버가 ‘좋은 그래프 구성’임을 실증적으로 뒷받침한다.



### Bridging Spatial And Frequency Views For Disaster Assessment: Benefits And Limitations (https://arxiv.org/abs/2606.17403)
Comments:
          Copyright 2026 IEEE. Published in the 2026 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2026)

- **Prior Approaches**: 재난 건물 손상 평가는 위성 RGB로부터 shape·texture 중심의 spatial-domain 네트워크(CNN, ViT 등)가 주로 이뤄졌고, xBD 같은 대규모 벤치마크가 이를 가속해왔다. 하지만 spatial만으로는 미세한 구조 변형이나 파편/붕괴로 생기는 미세한 질감 단서를 충분히 포착하기 어려워 ‘손상 없음’ 쏠림 같은 편향이 생길 수 있다. Frequency-domain(푸리에/웨이브렛 등) 단서는 고주파(에지·질감 불규칙) 신호를 강조하지만, 단독 사용 시 일반화가 약하거나 체계적 비교가 부족하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 xView2(xBD) 사후(disaster post-event) 이미지에서 손상 다중 클래스 분류를 대상으로 spatial-only, frequency-only, dual-domain를 공정하게 비교한다. EfficientNet-B0 백본과 학습 설정을 동일하게 고정하고, 입력 표현(공간 vs 주파수)과 fusion 전략만 바꿔 도메인 정보가 성능에 미치는 기여를 통제 실험으로 분리해 보여준다. 특히 dual 중에서도 Dual Spatial(공간+주파수 퓨전)의 전반 성능이 두드러짐을 정량적으로 제시한다.

- **Technical Challenges**: frequency-domain을 위해 RGB 채널별 orthogonally normalized 2D discrete Fourier transform을 적용해 크기(magnitude) 스펙트럼(로그 스케일 선택 가능)을 입력으로 구성해야 한다. 또한 단순 퓨전이 공간/주파수의 상보성을 충분히 활용하지 못할 수 있어, 병렬 브랜치에서 특징 레벨로 융합하는 방식을 설계하고(두 가지 연결 순서 실험) 과적합을 억제하기 위해 Focal Loss with Smoothing과 공통 학습 조건을 적용했다. 그럼에도 Minor 같은 미세 손상 클래스는 시각적 애매성과 클래스 불균형 영향으로 전 모델이 낮은 F1을 보여 해결 난이도가 남는다.

- **Empirical Impact**: 실험 결과 dual-domain이 single-domain보다 일관되게 우수했으며, Dual Spatial은 테스트 정확도 0.4688과 손실 최저(0.0351)를 기록했다. 반면 macro F1-score는 Spatial-only가 0.4254로 가장 높아, 정확도 향상이 항상 클래스 균형까지 보장하지는 않음을 드러낸다. Frequency-only는 전반적으로 최악이며 과적합/일반화 실패 양상이 뚜렷했고, 모든 모델이 특히 Minor에서 부진해 재난 피해 분류에서 accuracy보다 클래스 민감 지표가 중요함을 시사한다.



### The Discrete-Log Clock: How a Transformer Learns Modular Multiplication (https://arxiv.org/abs/2606.17399)
Comments:
          5 pages, 5 figures. Accepted to the Mechanistic Interpretability Workshop at ICML 2026

- **Prior Approaches**: 모듈러 곱셈에서도 grokking은 관측됐지만, 내부 알고리즘은 모듈러 덧셈만큼 명확히 규명되지 못했다. 기존 분석은 additive DFT(표준 DFT) 기준에서 임베딩의 Fourier 스펙트럼이 ‘dense(전 주파수 사용)’하다고 보고해, 곱셈이 덧셈과 다른 덜 해석가능한 방식을 쓴다고 해석됐다.

- **Core Contribution**: 이 논문은 ‘dense 스펙트럼’ 결론이 알고리즘 차이라기보다 **분석 기준(basis) 불일치**에서 생긴 아티팩트라고 주장한다. 곱셈의 자연스러운 분해는 multiplicative character transform이며, 이를 (Z/pZ)^*의 곱셈 군에 맞춰 적용하면 임베딩 스펙트럼과 뉴런 튜닝이 크게 단순해진다.

- **Technical Challenges**: 핵심 난제는 곱셈 연산을 설명하는 올바른 주파수 표현을 찾아내는 것이다; 로그 값이지만 실제 입력 a는 비선형 치환이라 표준 DFT로는 주기성이 깨져 보이기 쉽다. 저자들은 discrete logarithm으로 좌표를 재라벨링한 뒤(Discrete-Log Clock 아이디어), 임베딩/MLP 뉴런의 주파수 성분을 multiplicative character basis에서 재투영해 sparsity와 2D 주기 구조를 정량화한다.

- **Empirical Impact**: p=113에서 multiplicative basis로 보면 Gini 계수가 0.58(전 주파수 밀도→소수 주파수 집중)로, additive basis의 0.07과 대비되며 유의미한 에너지가 4개 key frequency에만 집중된다. 또한 MLP 뉴런의 96.9%가 단일 multiplicative frequency에 ‘cleanly tuned’되고, discrete logarithm 순서로 정렬한 activation heatmap에서는 2D stripe 패턴이 나타나 곱셈이 덧셈 메커니즘을 log-space로 옮긴 형태임을 뒷받침한다. 저자들은 이 메커니즘을 10개 prime으로 일반화하고, 일부 설정에서 modular exponentiation까지 정확히 학습하는 예비 결과도 제시한다.



### SoK: AI-Augmented Binary Reversing (https://arxiv.org/abs/2606.17398)
Comments:
          20 pages, 7 tables, 3 figures

- **Prior Approaches**: 기존 binary reversing은 triage, static analysis, dynamic analysis, security testing 같은 전통적 분석 모드로 프로그램의 구조·행동을 최대한 복원해 왔습니다. 하지만 컴파일 최적화와 난독화로 고수준 semantic 정보가 비가역적으로 소실되고, 많은 과제는 실행 탐색/추론 가능성의 한계 때문에 자동 완전 복원이 어렵습니다. 최근에는 ML과 LLM, agentic AI가 파이프라인 전반을 보강했지만 연구가 영역·표현·평가 방식별로 파편화되어 재사용 가능성이나 방법론의 공통 기반이 정리되지 못했습니다.

- **Core Contribution**: 이 논문은 AI-augmented binary reversing 분야를 최초로 체계(survey of knowledge, SoK)화해 2015년 이후 144편 연구를 22개 binary reversing domain으로 분류합니다. 또한 conventional 파이프라인과 AI 파이프라인을 “analysis artifacts 인터페이스”로 연결하는 통합 taxonomy를 제안해, 분석 산출물→표현 학습→semantic 추론으로 이어지는 공통 워크플로를 정리합니다. 이를 통해 LLM과 agentic AI가 어떤 역할로 결합되는지(추론/플래닝/툴 사용 등)도 명확히 합니다.

- **Technical Challenges**: 핵심 기술적 난제는 binary-derived artifacts의 품질·충실도·출처가 downstream 추론 성능을 좌우한다는 점입니다. 논문은 이를 artifact selection과 canonicalization(값·토큰·구조·시각), tokenization(원자/구문/룰/데이터 기반), encoding과 embedding(컨텍스트 의존/비의존)으로 구체화해, 컴파일/난독화로 인한 표현 이질성을 줄이면서 학습 가능한 입력으로 변환하는 과정을 정리합니다. 또한 학습이 끝단(end-to-end)으로 바로 끝나기보다, 어떤 inference type(분류·탐지·탐색·복구·복원·요약 등)를 위해 어떤 표현/추론 패러다임을 선택해야 하는지 프레임을 제공합니다.

- **Empirical Impact**: 저자들은 분야 성숙도와 반복되는 설계 패턴, 그리고 지속되는 평가 공백을 드러내며, “성공한 벤치마크”가 실제 reversing 작업 흐름에 통합되기까지의 거리도 함께 지적합니다. 22개 도메인 체계와 통합 용어를 통해 서로 다른 연구가 같은 문제 구조를 공유하는지 비교 가능해지고, 반대로 어떤 조건에서 성능이 깨지는지 추적할 기반이 마련됩니다. 결과적으로 신뢰 가능하고 확장 가능한 AI-augmented binary reversing 시스템을 위한 연구 우선순위 설정에 실질적 기준을 제공한다는 점에서 의미가 큽니다.



### NarrativeWorldBench: A Frontier-Saturated Benchmark and a Latent World Model for Long-Horizon Co-Creative Audio Drama (https://arxiv.org/abs/2606.17391)
Comments:
          10 pages. Accepted to the ICML 2026 Workshops on High-dimensional Learning Dynamics (HiLD) and Culture x AI

- **Prior Approaches**: 기존 장문 벤치마크(LongBench, RULER, L-Eval 등)는 주로 검색·사실 회상·요약 성능을 평가하며, 공동 창작 상황에서 연재 구조의 일관성을 직접 측정하진 못했다. 장문 생성 연구(예: plan-and-write, search 기반)는 목표를 길게 쓰는 데 초점을 두지만, 중간 에피소드가 생략된 채로도 ‘연재 상태(state)’를 유지하는 능력 평가와는 결이 달랐다. 그 결과, 장편 서사에서 관측되는 horizon 붕괴 현상을 체계적으로 진단할 기준이 부족했다.

- **Core Contribution**: 이 논문은 장편 오디오 드라마 연재의 구조적 일관성을 정량화하는 NarrativeWorldBench를 제안하고, 21개 LLM을 동일한 9개 내러티브 구조 지표로 감사(audit)했다. 특히 closed-frontier/ reasoning 계열 모델이 plot-beat F1이 [0.78, 0.81]에서 포화된 뒤 h=200에서 약 -0.20 F1로 붕괴함을 실증했다. 이를 넘어 N-VSSM(Narrative Variational State-Space Model)은 256차원 잠재 ‘세계 상태’를 갱신해 horizon이 길어져도 구조 성능을 유지하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 장편 연재가 부분 관측(partially observed) 과정이라, 국소 문맥만으로는 잠재 상태를 복원하기 어렵다는 점이다. 논문은 scene boundary마다 event tuple을 뽑아 변분 인코더로 잠재 상태를 갱신하고, Mamba-2 백본 디코더에 cross-attention 기반 저랭크 어댑터를 결합해 구조 정보를 전파한다. 또한 문화권 간 차이(현지화의 underspecification)를 ‘디코더 재학습’이 아니라 256차원 잠재 공간에 대한 Learned Cultural Transfer Function으로 보정해, 네 Indic 언어에서 교차언어 품질을 끌어올렸다.

- **Empirical Impact**: 구조 지표의 대표인 plot-beat F1에서 N-VSSM은 모든 horizon(h=10~200)에서 0.84 이상을 유지하며, 연산은 frontier 대역 대비 에피소드당 4배 낮은 비용을 주장한다. 긴 horizon에서 foreshadowing payoff, temporal coherence, motif persistence가 각각 대역 대비 +0.18, +0.14, +0.12 수준으로 개선됐다. 문화 전이 함수를 켜면 문화적 충실도(Likert 7점)가 언어별로 약 +0.20~+0.23 상승했고, within-subject writer study(n=12)에서는 long-arc consistency에서 71%로 우위를 보이며 controllability에서 +1.3 Likert 점을 더 받았다.



### Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models (https://arxiv.org/abs/2606.17389)
Comments:
          16 pages. Accepted to the ICLR 2026 Workshop on Multimodal Intelligence. Code: this https URL

- **Prior Approaches**: 기존 연구는 VLM의 신뢰도를 “Attention-Confidence Assumption” 관점에서 해석해 왔습니다. 즉, 시각 인코더가 관련 영역에 촘촘히 주목하면 모델이 정답을 낼 “근거(grounding)”가 생긴다고 가정했죠. 하지만 이런 해석은 attention이 출력의 결정요인을 충실히 설명하는지에 대한 논쟁과 함께, 출력 기반 환각 평가(benchmark) 중심으로 보정이 부족하다는 한계가 남아 있었습니다.

- **Core Contribution**: 이 논문은 VLM Reliability Probe(VRP)로 여러 VLM 계열(LLaVA-1.5, PaliGemma, Qwen2-VL)을 가로지르는 신뢰도 시그널을 체계적으로 비교합니다. 특히 시각 “structural” 지표(클러스터 수 C_k, 공간 엔트로피 H_s)와 생성 동역학 기반 지표(예: self-consistency)를 함께 상관/예측하며, reliability가 단순 attention 구조가 아니라 “생성 과정의 내부 상태 분포”에 가깝다는 결론을 제시합니다. 또한 LLaVA에서는 Early Lock(또는 Symbolic Detachment), PaliGemma/Qwen2-VL에서는 더 분산된 신뢰도 경로가 나타난다고 분석합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “attention heatmap이 신뢰도를 알려주는가?”를 상관 수준을 넘어 인과·기계적으로 분리해 증명하는 것입니다. 논문은 forward hook으로 cross-attention 및 hidden state를 수집하고, Logit Lens(레이어별 correct vs incorrect 토큰 logit 차)와 hidden-state probe, 그리고 대규모 causal ablation(특정 레이어의 예측 뉴런/부분을 파괴)까지 수행해 신뢰도 경로가 어디에 있는지 추적합니다. 그 결과 시각 attention은 구조적으로는 중요하지만 통계적으로는 정확도와 거의 무관하고, self-consistency와 hidden-state probe가 훨씬 강한 예측력을 보인다고 정리합니다.

- **Empirical Impact**: 실험에서 C_k와 H_s는 정답 여부와의 상관이 사실상 0에 가깝게 나와 “Cluster Failure”를 뒷받침합니다(예: R≈0.001, R≈-0.012). 반대로 self-consistency는 truth 예측에서 R=0.429로 시각 지표 대비 우수하며, hidden-state probe는 강한 설정에서 AUROC>0.95 같은 높은 분별력을 보입니다. 더 나아가 LLaVA는 late-stage 병목을 파괴하면 취약해지지만, PaliGemma와 Qwen2-VL은 예측에 기여하는 부분을 ~50% 이상 파괴해도 견고함을 보여 신뢰도 신호 설계가 아키텍처에 강하게 의존한다는 메시지를 강화합니다.



### TerraTransfer: Learning End-to-End Driving Policies Without Expert Demonstrations (https://arxiv.org/abs/2606.17386)
- **Prior Approaches**: 기존 end-to-end 자율주행 학습은 대체로 로그 운전자 데이터를 기반으로 imitation pretraining을 하고, 이후 fine-tuning(지도학습·open-loop RL) 또는 closed-loop RL을 추가하는 방식이었습니다. 특히 closed-loop RL은 photorealistic rendering과 대형 vision backbone 추론을 매 스텝 반복해야 해 계산비용이 커지고, 희귀·안전중요 상태는 로그에 충분히 없어서 covariate shift 문제가 남았습니다. self-play는 비용이 싸지만 픽셀 대신 vector state로 학습되는 경우가 많아 실제 raw image end-to-end로 확장되지 못했습니다.

- **Core Contribution**: 이 논문은 self-play의 “학습은 저렴한 vector 상태에서, 추론은 raw image로”라는 비대칭 이점을 결합해, demonstration 없이 end-to-end 주행을 만드는 단일 패러다임을 제안합니다. 핵심은 learning to drive(벡터 상태 기반 planning head)를 먼저 self-play로 학습한 뒤, learning to see(vision encoder)를 alignment 단계에서 맞추되 어떠한 단계도 logged trajectory를 상대로 supervise하지 않는다는 점입니다. alignment는 teacher(자기대전 self-play policy)의 action distribution과 표현 관계를 재현하도록 설계되어, 큐레이팅된 expert demonstration 없이도 전이가 가능함을 노립니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 vector state에서 학습된 teacher를 raw image 입력 공간으로 옮길 때, 단순 모듈 캐스케이드(검출/기하 복원 후 재입력)가 불필요하게 어려운 상위 복원 문제를 만든다는 점입니다. 이를 피하기 위해 teacher가 내부에서 사용하는 “풀드(pooled) 특징”과 직접 정렬하도록 설계했고, teacher 특징의 저랭크/중복성을 활용해 batch-relational low-rank structural loss로 관계(씬 간 유사성)를 주로 맞추도록 제한했습니다. 또한 action KL divergence로 학생의 정책 분포가 teacher와 일관되게 되도록 하여, 로그 경로 없이도 행동 정렬이 유지되게 했습니다.

- **Empirical Impact**: 평가는 photorealistic 3D Gaussian splatting 기반 closed-loop 벤치마크 HUGSim에서 HD-Score로 진행되며, 제안한 vision-정렬 end-to-end 정책이 imitation-trained 선행 방법들을 aggregate에서 match하거나 초과합니다. 특히 self-play teacher에 근접한 성능(집계 HD-Score 기준 teacher 대비 오차 약 0.03)을 보이면서도, paired (image, scene-state) 프레임은 약 1.83M 정도만 사용해 데이터 효율이 높다는 점이 강조됩니다. 또한 paired 데이터 비율을 줄여도 성능이 잘 유지되어, trajectory 라벨 없이도 강한 일반화와 재현성이 가능함을 실험적으로 입증했습니다.



### Model Validation of Agentic AI Systems: A POMDP-Based Framework for Belief-State, Forecast, and Policy Validation (https://arxiv.org/abs/2606.17383)
Comments:
          28 pages, 3 figures, 6 tables. Source code available from this https URL

- **Prior Approaches**: 기존 ML 검증은 주로 분류/회귀 정확도 같은 예측 성능에 초점이 맞춰져 있어, 에이전트가 실제로 내리는 의사결정 과정의 품질을 파악하기 어렵습니다. 에이전트는 불완전관측 하에서 잠재 상태에 대한 belief를 만들고 그 belief로 예측·행동을 반복 갱신하므로, 예측이 맞아도 결정이 나쁠 수 있고 belief 보정이 틀릴 수도 있습니다. 또한 전통적 모델 리스크 관점(SR 11-7, BCBS 239 등)은 개념적 타당성·모니터링·거버넌스를 강조하지만, agentic AI의 정보→belief→forecast→policy→utility 전 단계를 분해해 검증하는 실무형 프레임은 부족했습니다.

- **Core Contribution**: 이 논문은 agentic AI를 POMDP(부분 관측 마르코프 결정 과정)로 모델링하고, 정보·belief·forecast·action·utility로 의사결정을 계층 분해해 각 구성요소를 독립적으로 검증하는 모델 검증 프레임워크를 제안합니다. 특히 LLM을 “근사 Bayesian filtering 연산자”로 formalize해, 관측·검색·도구결과 같은 정보로부터 잠재 상태 belief를 산출하는 역할을 정식화합니다. 더불어 state-space, filtering, forecast, policy, utility-specification, parameter risk까지 확장한 모델-리스크 taxonomy를 구축해 검증·거버넌스·모니터링의 근거를 제공합니다.

- **Technical Challenges**: 핵심 난제는 (1) 잠재 상태 공간(state-space) 자체가 현실을 충분히 대표하는지, (2) LLM 기반 filtering이 belief를 얼마나 정확히 추정하는지, (3) belief가 잘 보정돼도 forecast나 policy가 잘못될 수 있는 다층 오류를 어떻게 분리해 진단하느냐입니다. 논문은 belief는 proper scoring rule(예: Brier/log score)로 calibration과 품질을 평가하고, forecast는 belief-조건부 예측 정보와 예측오차/상관지표로, policy는 realized utility로, utility-specification은 목적함수 정합성으로 각각 검증하도록 설계합니다. 또한 time에 따라 frontier model의 동작이 바뀌는 drift, 프롬프트 민감도/환각/검색 실패 같은 filtering risk를 모델 리스크 항목으로 포함하고, ablation·커버리지 테스트·파라미터 민감도 분석으로 견고성을 확인합니다.

- **Empirical Impact**: 포트폴리오 관리 사례에서 에이전트는 시장·거시정보로 잠재 마켓 레짐을 추론하고, belief-conditioned forecast를 만든 뒤 Black--Litterman 기반으로 포트폴리오를 구성합니다. 실험은 belief calibration diagnostics, coverage test, ablation, parameter-sensitivity를 조합해 “잠재 상태 추론이 결정 품질에 독립적으로 기여”함을 보여주며, 주요 결론이 합리적 파라미터 범위에서 견고하다는 점을 보고합니다. 즉, 단순 성능 지표를 넘어 belief-추론–결정 논리의 어느 단계가 실패하는지까지 추적 가능한 실무형 검증/거버넌스 토대를 제공한다는 의미가 있습니다.



### MeiBRD: Meta-Learning Intraoperative Biomechanical Residual Deformation (https://arxiv.org/abs/2606.17379)
- **Prior Approaches**: 수술 중 간(liver) 등록은 환자 자세, 복강압, 호흡, 도구-조직 상호작용 때문에 큰 연부조직 변형이 생기지만, 관측은 stylus나 iUS로 매우 희소하게 주어져 문제는 본질적으로 ill-posed입니다. 기존 biomechanical 모델은 물리 방정식으로 변형을 제약해 안정성을 얻지만, 선형 탄성 등 단순 가정 때문에 큰/비선형 변형의 예측에 편향이 남습니다. 딥러닝 기반 등록은 빠른 추론과 표현력이 장점이지만, 희소한 수술 중 데이터와 라벨 부족으로 OOD에서 일반화가 무너지고 물리적으로 그럴듯하지 않은 변형을 내기 쉽습니다.

- **Core Contribution**: 이 논문은 MeiBRD로, 수술 중 희소 대응만으로 선형 biomechanical 예측을 ‘그대로’ 학습하는 대신 residual만 보정하도록 하이브리드 프레임워크를 제안합니다. 선형 모델이 만드는 변형 오차를 그래프 신경 diffusion(예: GRAND)로 표현하되, 수술 중 측정 위치를 residual 함수의 input-output이 관측되는 context 샘플로 보고 feedforward meta-learning으로 residual 함수를 빠르게 적응시킵니다. 특히 전체 변형 필드를 end-to-end로 직접 학습하지 않아 데이터 효율성과 물리적 정합성(선형 예측에 대한 보정)을 함께 노립니다.

- **Technical Challenges**: 핵심 난제는 (1) 희소 측정이 주어졌을 때 긴 거리(long-range) 정보를 어떻게 전파해 수술 중 관측 바깥 부위의 residual을 추정하느냐, (2) diffusion 기반 잔차 함수가 간의 기하(표면/체적)에서 중요한 변형 신호를 놓치지 않게 하는 것입니다. 저자들은 3D 간 메쉬 그래프에서 GRAND diffusion을 수행하되, 표면 곡률, 체적 변화(테트라헡드 det(F)), 회전불변 변형률 등 geometry-aware attention을 통해 연부조직 변형에 민감한 특징을 연산자에 주입합니다. 또한 수술 중 adaptation을 MAML 같은 gradient-based가 아니라 context를 입력으로 하이퍼네트워크가 residual 함수의 파라미터를 생성하는 feedforward meta-learner로 구현해 수술 환경의 시간 제약을 맞춥니다.

- **Empirical Impact**: phantom liver 데이터셋 실험에서 MeiBRD는 rigid ICP, 선형 biomechanical LIBR, 비선형 BCF-FEM, V2S 같은 데이터 기반 기준선들을 상대로 전 테스트 설정에서 가장 낮은 평균 TRE를 보이며 OOD 일반화가 특히 강합니다. 무작위 분할에서는 V2S가 잘하지만 geometry/변형 분리(OOD)에서 급격히 악화되어, 희소 관측 조건에서 학습 기반의 일반화 한계를 재확인합니다. 정성/정량적으로도 선형 biomechanical이 특정 영역에서 크게 어긋나는 경우, MeiBRD는 그 구간의 residual을 선택적으로 보정하면서도 이미 잘 맞는 영역에서는 과도한 수정이 줄어드는 모습(오차 증가율 완화, 긴 거리 보정)을 보여 줍니다.



### Implicit vs. Explicit Prompting Strategies for LVLMs in Referential Communication (https://arxiv.org/abs/2606.17372)
- **Prior Approaches**: 최근 LVLM의 지시-매칭 referential communication 연구는 서로 상반된 결론을 냈습니다. 어떤 연구는 반복 라운드에서 표현 길이가 줄어들며 효율적 동조(lexical entrainment)가 나타난다고 본 반면, 다른 연구는 공통 기반(common ground)을 활용하지 못해 불필요하게 장황해진다고 보고했습니다. 또 일부는 점검창(history)과 메모리를 제공해도 자동으로 효율이 생기지 않거나, 대화적 정합성·수정 행동이 약하다고 지적했습니다.

- **Core Contribution**: 이 논문은 두 상반된 결과의 원인이 모델 버전이나 과제 차이가 아니라 prompting 스타일 차이에 있음을 직접 비교로 정리합니다. 특히 “암시적(implicit)으로 간결·정보성”을 유도하면 모델은 정확도는 유지하되 여전히 장황해지고, “명시적(explicit)으로 1~2단어처럼 짧게”를 지시하면 표현이 반복되며 안정화되는 패턴이 나타납니다. 결론적으로, 사람의 conceptual pact 형성과 유사한 겉모습은 프롬프트로 재현 가능하지만 그 내적 과정은 사람과 동일하다고 보기 어렵다고 경고합니다.

- **Technical Challenges**: 핵심 기술적 도전은 서로 다른 기존 연구의 task 설계 차이를 통제한 채, 프롬프트만 바꿔 동일 조건에서 비교하는 것입니다. 저자들은 모델 버전(GPT-5.2 vs GPT-5.5)과 입력 파이프라인을 정렬하고, 시각 컨텍스트(라운드 경계·최신 프레임·상태 렌더링)를 시간순으로 엄격히 맞춘 뒤 implicit/explicit 프롬프트만 비교했습니다. 이를 통해 “명시적 압축”이 interaction의 형태(라벨의 가지치기와 텔레그래픽 표현) 자체를 바꾼다는 점을 관찰해 해석의 혼선을 줄였습니다.

- **Empirical Impact**: 실험은 AI–AI 40 runs(라운드 200 관측)에서 정확도는 전 조건에서 높지만, communicative efficiency와 표현 수렴은 프롬프트에 크게 좌우됨을 보였습니다. implicit 조건에서는 GPT-5.2/5.5 모두 라운드가 진행돼도 장황함이 크게 줄지 않았고, explicit 조건에서는 명시적 압축이 62.8%~75.6% 수준으로 강하게 나타났습니다. 특히 explicit GPT-5.5는 표현 길이와 lexical overlap이 라운드 1→5에서 뚜렷이 수렴하며 정확도도 97.5%로 유지됐지만, 같은 전략에서 GPT-5.2는 압축 이후 정확도가 92.5%로 하락해 accuracy–brevity tradeoff 가능성도 함께 드러냈습니다. 이 결과는 referential communication에서 “사람 같은 동조”를 평가할 때 prompting 의존성을 반드시 통제해야 한다는 실증적 시사점을 줍니다.



### DriveJudge: Rethinking Autonomous Driving Evaluation with Vision-Language Models (https://arxiv.org/abs/2606.17362)
Comments:
          Under Review

- **Prior Approaches**: 기존 운전 평가 지표는 크게 두 부류로 나뉩니다. (1) ADE/FDE 같은 모방 기반 평가는 시연 궤적을 기준으로 유사도를 재지만, 멀티모달한 운전 특성과 비최적/누락된 시연 문제를 그대로 안고 있습니다. (2) EPDMS 같은 rule-based 평가는 해석 가능하고 물리적으로 정밀하지만, long-tail 상황에서 규칙 적용의 ‘문맥’을 놓쳐 합리적인 동작까지 과도하게 벌점할 수 있습니다.

- **Core Contribution**: 이 논문은 VLM(vision-language model) 추론으로 상황 문맥을 먼저 해석한 뒤, 필요한 경우에만 물리 기반 deterministic rule 함수(예: 충돌, 차선 일탈)를 선택적으로 호출해 평가하는 DriveJudge를 제안합니다. 즉, VLM의 문맥 이해와 rule의 공간 정밀도/물리적 grounding을 결합해 ‘해석 가능 + 문맥 인지’ 평가를 목표로 합니다. 또한 DriveJudge가 제대로 작동하는지 검증할 수 있도록 Driving Quality Classification과 Trajectory Preference Selection 두 가지 human-aligned 벤치마크/평가 프로토콜을 마련합니다.

- **Technical Challenges**: DriveJudge의 핵심 난제는 VLM만으로는 안전·공간 위반을 정밀하게 판정하기 어렵다는 점과, rule을 단순 합산하면 규칙 중요도가 문맥에 따라 달라지는 long-tail을 반영하지 못한다는 점입니다. 논문은 이를 위해 (a) 장면별로 어떤 규칙을 ‘gating’(선택)할지 예측하는 tool-invocation 설계를 넣고, (b) 장면에서 규칙 점수가 낮더라도 행동이 ‘궁극적으로 합리적’일 수 있음을 반영하는 데이터 마이닝/라벨링으로 학습 신호를 구성합니다. 학습은 SFT로 규칙 호출 결정을 먼저 안정화하고, 이후에는 preference 정렬을 위해 RL(GRPO)로 미세 조정합니다.

- **Empirical Impact**: 33,577개의 long-tail 운전 샘플(인간 주석: 해당 장면에서 행동이 합리적인지)로 평가한 결과, DriveJudge는 EPDMS 대비 Driving Quality Classification에서 AUC를 21.23 포인트 개선했습니다. Trajectory Preference Selection에서도 DriveJudge는 DriveCritic 대비 6.5%p 정확도를 더 높여, preference 모델임에도 불구하고 더 일치하는 결과를 보였습니다. 정성 비교에서는 VLM 직접 점수 모델이 공간 정합성 부족으로 사실 오류를 내리거나 rule-based가 문맥상 정당화되는 ‘nudge’를 과벌점하는 문제를 DriveJudge가 tool-grounded 평가로 완화함을 확인했습니다.



### Translating the Untranslatable: An Operationalizable Ontology for Untranslatability (https://arxiv.org/abs/2606.17354)
- **Prior Approaches**: 기존 NLP MT는 번역을 ‘의미가 같은 문장 → 다른 언어 문장’의 일대일 대응으로 보는 경향이 강해, 문화·문체·표현 차이로 의미 보존이 완전하지 않은 untranslatability를 구조적으로 다루기 어려웠습니다. 선행연구는 관용어, 욕설/슬랭, 경어 같은 개별 현상에 집중했지만, 공통 온톨로지로 연결해 통합 분석하거나 이를 위한 일관된 평가/학습 틀을 제공하진 못했습니다. 또한 이론적 번역연구는 분류와 보상 전략을 제시해도 NLP에서 직접 실험 가능한 스케일 자원이 부족했습니다.

- **Core Contribution**: 이 논문은 MT에서의 untranslatability를 ‘원인(불일치의 출처)’과 ‘보상 전략(compensation strategy)’이라는 두 축으로 분해해, 이를 ontology(uTypes)와 전략 분류(cStrats)로 체계화합니다. 이어서 uType 라벨이 붙은 스페인어·일본어 문장에 대해, 서로 다른 cStrats가 반영된 영어 번역을 짝지은 다국어 데이터셋을 구축해 controlled analysis가 가능하도록 했습니다. 마지막으로 전략이 품질 인식에 어떻게 영향을 주는지(전략-의존성 선호)를 초기 실험으로 확인합니다.

- **Technical Challenges**: 핵심 과제는 (1) 언어 간 의미 불일치의 다양한 원인을 자연스럽게 분류(uType)하고 (2) 그 상황에서 적절한 보상 전략을 선택·표현할 수 있게 데이터로 연산화하는 것입니다. 저자들은 인간 언어 전문가의 도움으로 온톨로지의 기반을 세우고, LLM을 이용해 동일 uType 내에서도 다양한 예시를 생성하되 프롬프트 반복 개선과 인간 검증(유효성 약 95~96%)으로 품질 변동을 완화했습니다. 또한 표준 token-level 번역 가정 대신, ‘불일치 식별 → 전략 선택 → 토큰 생성’의 분해 관점으로 모델 설계 아이디어를 제시했습니다.

- **Empirical Impact**: 인간 선호도 연구에서 번역 품질은 단순 충실도뿐 아니라 사용한 cStrat에 크게 좌우되며, 특히 추가 설명 맥락을 포함하는 Annotation(AN) 전략이 전반적으로 가장 선호되는 경향을 보였습니다. 유효성 있는 차이는 uType과 번역 맥락(예: textbook vs movie) 및 소스 언어(스페인어 vs 일본어)에도 따라 달라져, ‘기본 번역’만으로는 인간 기대를 충분히 충족하기 어렵다는 실증 근거를 제공합니다. 결과적으로 이 프레임워크와 데이터셋은 strategy-informed machine translation 연구 및 다운스트림(전략 예측, untranslatability 탐지 등) 확장에 기초가 될 것으로 기대됩니다.



### Do Large Language Models Always Tell The Same Stories? (https://arxiv.org/abs/2606.17350)
- **Prior Approaches**: 기존 연구는 창의성을 Alternative Uses Task, Torrance Test of Creative Thinking 같은 루브릭/심리측정 기반으로 평가하거나, n-gram novelty 같은 어휘 기반 지표로 자동화해 왔습니다. 다만 LLM-as-a-Judge 방식은 주관성과 편향으로 일관성이 떨어질 수 있고, n-gram novelty는 창의성을 대리한다고 보기 어렵다는 비판도 있습니다. 한편 일부 작업은 플롯 아크나 플롯 요소 반복 같은 ‘부분’만 보며 서사 전체의 다양성을 직접 비교하긴 제한적이었습니다.

- **Core Contribution**: 이 논문은 창의성을 ‘서사 다양성’으로 재정의하고, narrative similarity(서사 유사도)라는 대조(contrastive) 프레임으로 인간과 LLM의 다양성을 동일 조건에서 비교합니다. r/WritingPrompts 프롬프트를 주고 여러 모델의 이야기를 생성한 뒤, 기준 이야기 대비 두 후보 중 어느 쪽이 더 비슷한지 선택하도록 설계했습니다. 그 결과 LLM이 만든 이야기는 모델 내부/모델 간 모두에서 서로 더 닮아가며, 특히 frontier 모델은 개인 작가의 다양성은 못 따라가고 ‘평균적인’ 서사로 수렴하는 경향을 확인합니다.

- **Technical Challenges**: 핵심 난제는 ‘다양성’을 신뢰도 있게 수치화하는 데 있었고, 이를 위해 사람이 판단하는 triplet(기준, 후보 A/B) 유사도 데이터를 만들고 자동화 방법 3가지를 검증했습니다. LLM-as-a-Judge, narrative component embedding(서사 구성요소를 분해해 임베딩 후 코사인 유사도), Bradley-Terry 기반 preference model을 비교한 뒤, 성능과 확장성을 고려해 상황별로 도구를 선택합니다. 또한 human-LLM/LLM-LLM 비교의 공정성을 위해 길이·품질 필터와 노출 편향을 줄이기 위한 시간 근접 조건을 적용했습니다.

- **Empirical Impact**: 10개 모델(프론티어 폐쇄형, 오픈형, post-training 체크포인트 포함)을 대상으로 한 대규모 결과에서 LLM 서사는 인간 서사보다 서로 더 유사하다는 패턴이 일관되게 나타났습니다. closed-source 프론티어는 인간에 더 가깝게 ‘모사’하지만, 모델 간/서로 다른 인간 작가 간 다양성은 부족했고, open-source/특정 체크포인트는 인간과는 다르더라도 여전히 내부 동질성은 강했습니다. 더구나 negative prompting, temperature scaling, round-robin 시퀀셜 생성 같은 흔한 완화 전략은 서사 동질성을 의미 있게 줄이지 못해, 향후 LLM 창작의 ‘다양성 붕괴’ 문제를 정면으로 다룰 필요성을 제기합니다.



### Counterfactual Optimization of Baseball Pitch Sequences and Estimation of Its Impact on Season-Level Statistics (https://arxiv.org/abs/2606.17345)
- **Prior Approaches**: 기존 야구 애널리틱스의 pitch sequencing 연구는 대체로 한 타석(plate appearance) 안에서 ‘마지막 공’의 성과를 최적화하는 데 집중해 왔습니다. 반면, 그 직전의 setup pitch가 다음 타석까지 이어지는 장기 시즌 단위 성적에 어떤 영향을 주는지는 충분히 분석되지 못했습니다.

- **Core Contribution**: 이 논문은 MLB Statcast 데이터를 바탕으로 counterfactual 분석을 수행해, 마지막 공뿐 아니라 setup pitch까지 포함한 시퀀스 전체가 성과에 미치는 효과를 정량화합니다. Transformer 기반 모델로 특정 목표 공이 인플레이(in-play)로 이어질 확률을 예측하고, 그 공(또는 앞선 setup 공)을 다른 구종/위치로 바꾼 대안 시퀀스를 생성해 최적 선택을 정의합니다.

- **Technical Challenges**: 핵심 난제는 문맥 정보를 고정한 채 특정 공만 교체했을 때의 ‘원인-결과’를 어떻게 추정하느냐입니다. 논문은 주변 상황을 고정한 채 마지막/설정 공을 부분 교체해 in-play 확률을 최소화하는 counterfactual을 선택하고, 모델 출력과 시즌 통계를 연결하는 회귀 모형으로 K/9 같은 장기 지표의 기대 효과를 추정합니다.

- **Empirical Impact**: 실험 결과, 마지막 공뿐 아니라 setup pitch까지 함께 최적화하면 시즌 수준 성과가 유의미하게 개선될 수 있으며, K/9에서 1.0 이상의 개선도 관측됩니다. 또한 속도대(velocity-band)별로 효과적인 위치, pitch command의 중요성, middle-velocity 공을 활용한 선택지 확장 등 실무적 인사이트를 정량적으로 제시해 pitch sequencing의 전략적 가치를 뒷받침합니다.



### Geometry-Consistent Endoscopic Representations for Image-Guided Navigation via Structured Foundation Model Adaptation (https://arxiv.org/abs/2606.17340)
- **Prior Approaches**: 의료 내시경에서 자주 쓰이는 접근은 DINO/SAM 같은 vision foundation model을 zero-shot 또는 parameter-efficient fine-tuning(예: adapter)로 옮겨오는 방식이다. 하지만 내시경은 저대비 조직 질감, 반복 패턴, 비강체 변형, specular highlight 등으로 인해 geometry(기하) 일관성이 약해지고, 이런 이유로 pose/refinement과 같은 공간 추론 태스크에서 안정성이 떨어지기 쉽다. 한편 endoscopy-specific foundation model이나 domain generalization은 외형·모달리티 변화 대응에는 도움을 주지만, 3D 기하 일관성을 학습 표현에 직접 강제하지 않는 경우가 많아 내비게이션 신뢰도를 제한한다.

- **Core Contribution**: 이 논문은 단일 프레임(monocular endoscopy)에서 geometry-consistent 하면서도 domain-robust한 이미지 표현을 학습하는 통합 프레임워크를 제안한다. 핵심은 합성 데이터로 정확한 기하 감독을 제공하고, Hierarchy-Aware Geometry–Semantic Adaptation(HGSA)가 transformer 계층 구조에 맞춰 low-rank adapters를 선택적으로 삽입하며, 중간 계층은 기하 대응, 깊은 계층은 의미 일관성이 유지되도록 layer-wise objective를 결합하는 것이다. 결과적으로 공간적 안정성(기하)과 cross-domain 강건성(의미/도메인)을 동시에 노리는 표현 학습을 목표로 한다.

- **Technical Challenges**: 가장 큰 어려움은 임상 환경에서 정확한 camera pose나 dense depth 같은 기하 라벨을 대규모로 구하기 어렵고, 동시에 내시경 특유의 비정형 변형과 도메인 갭이 correspondence를 쉽게 붕괴시킨다는 점이다. 이 문제를 위해 CT 기반 3D 해부학 모델로 합성한 다중 도메인 학습 파이프라인을 만들고, 뷰 쌍에 대해 렌더링된 depth·상대 포즈로 flow를 계산해 feature warping 기반 다중 스케일 기하 감독(PatchNCE/코사인 재투영)을 중간 계층에 걸어 준다. 또한 전역 의미 정렬은 late-layer의 global contrastive(InfoNCE)로 수행하되, Gram 기반 regularization으로 공간 해상도 훼손을 억제하면서, transformer hierarchy에 맞춘 adapter 배치·모듈 타깃·rank/스케일을 coarse-to-fine 탐색으로 찾아 학습 충돌을 줄였다.

- **Empirical Impact**: 실험에서는 linear probing으로 의미 분리(장면 분류)와 기하 품질(깊이 추정)을 함께 측정해, 제안한 HGSA 및 기하-의미 결합 학습이 표현 품질을 함께 개선함을 보였다. 더 나아가 pose estimation과 monocular depth estimation 같은 내비게이션 관련 downstream에서 개선이 이어져, 합성 기반으로 학습한 표현이 실제 임상(예: clinical bronchoscopy)로 잘 transfer되는 것을 확인했다. 또한 sinus/colonoscopy로의 제한적 supervision 하 cross-procedure 적응에서도 유의미한 성능을 보였고, 모델 크기·학습 데이터 스케일에 대해서도 좋은 경향을 보여 endoscopy representation learning에서 실용적인 접근이라는 점을 뒷받침한다.



### Transformer-Based Warm-Starting for Feasible and Optimal Terminal Approach to Tumbling Objects with Space Manipulators (https://arxiv.org/abs/2606.17317)
Comments:
          8 pages, 4 figures

- **Prior Approaches**: 기존 우주 로봇(스페이스 매니퓰레이터) 말단 접근은 NLP 기반 궤적 최적화를 SCP로 풀거나, 동역학을 단순화해 버스 구동을 고정/배제하는 방식이 많았습니다. 또한 warm-start를 위해 룩업테이블·필터·조건부 예측·휴리스틱 초기화 등이 시도됐지만, 3D 궤도 말단 접근처럼 회전하는 안전 제약과 버스-암 동역학 결합이 동시에 있는 경우 초기 추정 품질에 민감해 실시간 적용이 어렵습니다.

- **Core Contribution**: 본 논문은 비협조적으로 tumbling하는 목표에 대한 3D 말단 접근에서, sequential convex programming(SCP)의 핵심 병목인 2단계(자세-매니퓰레이터 토크 할당) 초기해를 causal transformer 기반 warm-start로 학습합니다. 문제를 COM(중심질량) 병진 계획(1단계)과 결합된 attitude-매니퓰레이터 토크 할당(2단계)로 분해하고, 특히 visibility cone과 회전 안전 제약을 포함한 2단계의 nonconvex 최적화를 더 빠르고 안정적으로 풀 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 난관은(1) 회전하는 안전 제약과 가시성 제약이 포함된 상태에서 비선형 결합 동역학 때문에 초기해가 조금만 틀려도 SCP 반복 수·런타임이 급증한다는 점입니다. 또한 학습 warm-start가 첫 SCP convexified subproblem에서 인위적 비가능성(trust region 제약)을 유발할 수 있어, 논문은 SCP 자체의 trust region 확장 휴리스틱과 결합하고, 2단계 쿼터니언 단위 제약 문제는 재정규화로 보정하면서 학습된 초기화를 안전하게 통합합니다.

- **Empirical Impact**: 300개의 held-out 시나리오에서 학습 warm-start는 2단계 SCP의 반복 횟수를 최대 28%, 런타임을 23%까지 줄이면서 최종 control-cost 분포는 유지했습니다. 더 나아가 cost-optimal 지향 SCP가 아니라 비선형 feasible projection에 사용할 때는 런타임이 거의 절반으로 줄었고, 휴리스틱 초기화에서 관측된 고비용 꼬리(catatstrophic high-cost tail) 현상도 피했습니다. 이 결과는 sequence-model warm-start가 최적화 기반 말단 유도(terminal guidance)의 계산 효율뿐 아니라 궤적 견고성까지 개선할 수 있음을 보여줍니다.



### From Democracies to Autocracies: How AI Systems Enable Authoritarianism by Design (https://arxiv.org/abs/2606.17286)
- **Prior Approaches**: 기존 연구는 AI의 성능이나 예측·감지 능력, 또는 사후에 나타나는 파급효과(감시 확대, 여론 조작 등)에 초점을 맞추는 경우가 많았다. 그 결과 “어떻게 개발·조달·배치·운영되었기에” 권위주의적 통제가 가능해지는지에 대한 체계적 이해가 부족했다. 또한 권위주의를 ‘정권 유형’으로만 보려는 관성이 있어, 민주주의 국가에서도 나타날 수 있는 기술 기반 권위주의의 작동 방식을 놓치기 쉽다.

- **Core Contribution**: 이 논문은 미국부터 중국까지 서로 다른 정치체제에서 실제로 배치된 AI 시스템 6개의 라이프사이클을 매핑해, 기술이 권위주의적 실천을 가능케 하는 핵심 요인을 ‘배치 과정’ 중심으로 드러낸다. 특히 권위주의를 엄밀한 정권 분류가 아니라 다양한 체제에서 공통으로 관찰되는 “실천( practices )”의 묶음으로 재정의해 비교 가능성을 높인다. 저자들은 중앙집중형과 분산형 시스템 모두가 거버넌스 공백을 악용해 권위주의로 이어질 수 있다고 정리한다.

- **Technical Challenges**: 배치된 시스템의 라이프사이클을 동일한 기준으로 비교하려면, 공개자료가 제한적이고 국가별 규정·감사 관행도 달라지는 문제가 있다. 저자들은 모델 아키텍처뿐 아니라 조달, 통합, 테스트, 롤아웃, 감독, 안전장치, 현재 상태 등 10개 단계로 사례를 구조화하고, 조사보도·기술감사·평가·정부 문서·인터뷰 등 다양한 출처를 교차검증해 공백을 메운다. 그 위에서 시스템이 권위주의 특성(강압 능력 확대, 책임성 침식, 상징적 안전장치, 정보 통제, 선제적 탄압, 경계 통제)에 기여하는 “작동 메커니즘”을 질적으로 도출한다.

- **Empirical Impact**: 분석 결과, 권위주의를 가능케 하는 공통 ‘활성화 요소’로 데이터의 중앙집중 및 행정 데이터의 공용화(법집행·정치적 처벌 목적), 남용을 억제하지 못하는 규제 공백, 사용자 준수 약화로 인간 감독이 무력화되는 점, 보호집단의 특성을 인코딩해 취약 인구를 식별하는 점이 반복된다. 또한 중앙집중형은 보안·군사 기관 등 집행권이 주도할 때 공식적 감독에서 벗어나기 쉬우며, 분산형은 책임이 여러 이해관계자 사이에 흩어져 감독 장치가 회피될 수 있음을 보여준다. 이 연구는 “권위주의는 특정 체제의 일탈”이 아니라 개발자·행정가·사용자 선택이 누적된 분산된 결과일 수 있음을 실증적으로 제기하며, 개발자와 정책결정자에게 완화 권고를 촉구한다.



### ARVO: Atlas of Reproducible Vulnerabilities for Open-Source Softwar (https://arxiv.org/abs/2606.17283)
Comments:
          Accepted at IEEE European Symposium on Security and Privacy (EuroS&P) 2026

- **Prior Approaches**: 과거 취약점 데이터셋 연구는 주로 quantity(규모)나 diversity(다양성)에 치우쳤고, reproducibility(재현성)는 상대적으로 소홀했습니다. 그 결과 역사적 버그라도 소스에서 다시 빌드해 동일하게 트리거하고(재현+재트리거) 패치까지 검증하는 것이 어려워, 다운스트림 보안 연구에 바로 쓰기 힘든 “연구-가능 subset”만 남는 문제가 반복됐습니다. OSS-Fuzz-OSV처럼 비교적 자동화된 파이프라인도 시기 기반으로 재현 성공률이 제한되거나(약 37%) 시간 지나면 빌드/재현이 무너지는 한계를 보였습니다.

- **Core Contribution**: 이 논문은 reproducibility, quantity, diversity의 3자 트레이드오프에서 특히 재현성을 전면에 내세워, 대규모로도 “일관된 재구축·재트리거·분석”이 가능한 보안 데이터셋 생성 방식을 제안합니다. 이를 통해 OSS-Fuzz의 가장 큰 오픈소스 취약점 데이터셋에 full reproducibility를 도입하고, ARVO(Atlas of Reproducible Vulnerabilities in Open-source software) 데이터셋(311개 프로젝트, 6,100+개 실존 취약점)을 구축했습니다. ARVO는 각 취약점을 버전 전반에서 consistent하게 rebuild/trigger/analyze할 수 있도록 제공하며, 추가로 패치 커밋을 자동으로 찾고 코드 변경 이후 취약점 “직접 상호작용”도 지원합니다.

- **Technical Challenges**: 주요 기술 난관은 과거 소프트웨어를 되살려 빌드·실행을 끝까지 재현하는 데서 생깁니다: (1) 의존성 호환성 깨짐, (2) 누락된 외부 리소스/다운로드 경로의 붕괴, (3) 과거의 취약한 빌드 스크립트가 너무 취약하고 수정이 연쇄적으로 실패를 유발하는 문제입니다. ARVO는 재현용 리소스(정확한 의존성 버전·빌드 단계·환경·PoC)와 재현 파이프라인을 결합해 취약/패치 버전을 모두 재컴파일하고 PoC 입력으로 동일 크래시 존재 여부를 검증한 뒤, 실패한 항목은 제외하는 방식으로 신뢰도를 확보합니다. 또한 패치 locator가 업스트림의 “패치 후보 범위”에서 커밋 이분탐색을 수행해 취약점을 실제로 고친 최초 커밋을 찾도록 설계했습니다.

- **Empirical Impact**: 평가 결과 ARVO는 시도한 재현 중 81%를 성공시켰고, 찾은 패치에 대해 89.4%의 정확도를 달성했습니다. 특히 기존 OSS-Fuzz-OSV 대비 재현성 성능을 크게 끌어올렸으며, 멀티 컴포넌트처럼 복잡한 프로젝트에서도 성과가 유지되는 점이 강조됩니다. 다운스트림 측면에서는 업스트림(OSS-Fuzz 쪽) 재현 컴포넌트 통합으로 이어지고, DARPA AI 사이버 챌린지나 CyberGym 같은 연구/벤치마크에서 데이터 소스로 활용되며, 데이터셋이 실제로 “오탐된 수정/누락된 패치”도 찾아내는 검증 루프 역할을 보였다는 점에서 의미가 큽니다.



### Pulling The REINS: Training-Free Safety Alignment of Video Diffusion Models via Representation Steering (https://arxiv.org/abs/2606.17257)
- **Prior Approaches**: 기존 방어는 프롬프트 단계에서 unsafe를 거르거나, 생성된 출력 후에 필터링하는 방식이 중심이었다. 하지만 프롬프트 필터링은 jailbreaking·우회 프레이징으로 쉽게 깨지고, 출력 필터링은 생성 비용을 소모하는 데다 모델 내부 표상이 unsafe 경로를 계속 전파해 취약점이 남는다. 또한 안전 fine-tuning은 데이터·연산 비용이 크고 범용 생성 능력 저하(성능 열화) 위험이 있다.

- **Core Contribution**: REINS는 비디오 diffusion 모델의 가중치 업데이트 없이, 추론 시점에 hidden-state 표상 공간을 안전 방향으로 steering하는 training-free 방법을 제안한다. 저자들은 안전/위험 신호가 비디오 트랜스포머의 중간 hidden-state에 선형적으로 인코딩돼 있고, Supervised PCA로 찾아낸 단일 direction이 safe/unsafe 생성 궤적을 구분한다고 보고한다. 이 방향을 denoising 과정의 중간 레이어에 더해 unsafe 대신 의미적으로 유사한 safe 대안을 생성하도록 유도한다.

- **Technical Challenges**: 핵심 과제는 (1) 안전 정보를 실제로 분리 가능한 레이어와 방향으로 찾아내고, (2) 단순 더하기 perturbation이 아티팩트를 만들지 않게 하며, (3) classifier-free guidance의 두 분기 간 불일치를 막는 것이다. REINS는 SPCA로 후보 레이어별 safety-relevant 성분을 추정하고, safety 정보는 깊이에 따라 누적되지만 steering 성능은 중간 레이어에서 최대가 되는 tradeoff를 분석해 최적 레이어를 선택한다. 또한 per-channel norm preservation로 채널 스케일 변화를 억제하고, CFG에서는 conditional/unconditional 두 브랜치 모두에 동일하게 steering을 적용한다.

- **Empirical Impact**: REINS는 9개 비디오 diffusion 모델(1.3B~5B)에서 T2V와 I2V 모두에 대해, SafeSora와 SafeWatch-Bench 양쪽에서 평균적으로 safety rate를 일관되게 끌어올렸다. 안전 개선 폭은 최대 +0.52 수준까지 관찰되며, 특히 baseline safe 비율이 매우 낮은 상황에서도 효과가 유지돼 adversarial prompt 분포에서도 강건함을 시사한다. 동시에 motion/visual quality는 다수 모델에서 기준선과 동등하거나 경쟁적이어서, 안전성 향상을 ‘성능 열화 없는’ 추론 개입으로 연결했다는 점에서 의미가 크다.



### MLLP-VRAIN UPV system for the IWSLT 2026 Simultaneous Speech Translation task (https://arxiv.org/abs/2606.17255)
Comments:
          IWSLT 2026 System Description

- **Prior Approaches**: 동시 음성 번역(SimulST)에서는 ASR과 번역기를 연쇄(cascaded)로 결합하거나, 고정 지연 제어 정책으로 quality-latency trade-off를 맞추는 방식이 주로 쓰여 왔다. 다만 long-form에서는 발화가 길어질수록 정책의 경직성이 누적되어 번역 품질이 떨어지거나 지연이 커지는 문제가 반복됐다.

- **Core Contribution**: 이번 제출은 Parakeet과 Qwen 3.5 모델을 활용해 long-form SimulST에 강건한 연쇄형 파이프라인을 구성하고, 적응형 black-box policy로 처리 흐름을 제어한다. 또한 정책 제약을 완화(relaxation)해 trade-off를 더 유리하게 조정했으며, En→De/It/Zh에서는 ASR word-boosting과 RAG 기반의 오프라인 pre-translated exemplar로 도메인 맥락을 보강하는 context track도 추가했다.

- **Technical Challenges**: 핵심 난제는 긴 입력에서 언제 번역을 시작/중단할지 결정하는 지연 제어를 잘못하면 품질이 급락한다는 점이다. 논문은 adaptive black-box policy로 정책을 상황에 맞게 조정하고, 정책의 relaxations를 통해 더 세밀한 품질-지연 균형을 탐색했으며, 시스템 전반에 대해 latency 분석을 상세히 제공해 병목을 진단 가능하게 만들었다.

- **Empirical Impact**: IWSLT 2026 Simultaneous Speech Translation shared task에서 전 언어 방향으로 참여했으며, MCIF En→De 테스트셋 기준 XCOMET-XL이 지난해 대비 +5.82로 큰 폭 향상됐다. context track에서는 추가로 +1.03 성능 개선을 보여, RAG와 word-boosting을 통한 맥락 풍부화가 실제로 번역 품질을 끌어올린다는 점을 실증했다.



### Physics-Informed Attention Mechanism and Generalization Capability of Deep Learning-Based Grain Growth Evolution Prediction (https://arxiv.org/abs/2606.17235)
- **Prior Approaches**: 기존 ML 기반 곡물(침상) 성장 예측 연구는 주로 이상화된 합성 데이터로 학습한 뒤, 학습 분포와 유사한 조건에서만 평가하는 경우가 많았다. 또 전통 시뮬레이션(phase-field, level-set)은 PDE를 시공간 전역에서 반복 계산해 정확도는 높지만 계산 비용이 매우 커서(시간~주 단위) 실제 적용에 제약이 컸다.

- **Core Contribution**: 본 논문은 이전 연구의 곡물 성장 예측 모델이 OOD(학습 분포 밖) 조건에서도 재학습/미세조정 없이 일반화되는지를 체계적으로 검증한다. 동시에 곡물 경계 네트워크에만 주의를 두도록 설계한 boundary-masked attention을 제안해, 균열 없는 합성 조건에만 맞춰진 모델이 현실 조건에서 어디까지 정확도를 회복할 수 있는지 분석한다.

- **Technical Challenges**: 핵심 난제는 합성 데이터의 매끈한 경계/단봉 분포/균일한 성장 동역학 같은 학습 편향을 깨고, 실험 미세조직·이봉(bimodal) 분포·abnormal grain growth 같은 분포 이동에서 경계 이동을 예측해야 한다는 점이다. 논문은 attention을 spatiotemporal 위치(T×H×W) 전체에 두되, softmax 전에 곡물 경계가 아닌 영역의 attention score를 크게 억제하는 마스킹으로 물리적(곡률 구동) 귀납편향을 주입해 이 문제를 완화했다.

- **Empirical Impact**: 실험 미세조직(거친 경계)과 이봉 분포, abnormal grain growth까지 세 조건 모두에서 baseline도 OOD 일반화를 보였고, boundary-masked attention은 대체로 더 낮은 오차를 달성했다. 특히 이봉 분포에서 SSIM이 0.6221→0.7609로 크게 상승하고 평균 곡물 크기 오차도 8.75%→3.57%로 줄어, 경계 형태가 학습 도메인과 맞을 때 물리정보 attention이 가장 큰 이득을 준다는 점을 보여준다. 또한 attention heatmap 분석에서 경계의 곡률 구동 성장과 일관된 방식으로 큰 grain boundary에 주의를 집중하는 학습 패턴이 관찰되어, 구조적 물리 제약이 재학습 없이도 견고함을 보강할 수 있음을 시사한다.



### Rift: A Conflict Signature for Deception in Language Models (https://arxiv.org/abs/2606.17229)
Comments:
          13 pages, 4 figures. Code and experiment logs: this https URL

- **Prior Approaches**: ELK(Eliciting Latent Knowledge)는 출력이 그럴듯하더라도 모델이 아는 진실을 추출하려는 문제로, 특히 “아는 것을 숨기고 거짓을 말하는 경우”가 가장 어렵다고 본다. 기존 방법들은 논리 일관성 제약으로 진실-유사 방향을 찾거나(unsupervised), 또는 선형 표현/steering으로 정직성 같은 개념을 다루지만, “거짓은 내면서(정답처럼 보이게) 내부적으로 어떤 상태가 다른지”를 라벨 없이 분리해내는 데는 한계가 있었다.

- **Core Contribution**: 이 논문은 “거짓이지만 단순히 틀린 것”과 “진실을 알면서도 의도적으로 거짓을 말하는 것”을 내부 상태 차이로 구분할 수 있는지 묻는다. 핵심은 wrongness(틀림)와 deception(지식 충돌)을 분리하기 위해 sleeper agent(정답을 아는 상태에서 트리거로 거짓)와 naive liar(동일한 오답을 내도록 fine-tuning만 한 상태)를 대조하는 통제 설계를 제시하는 것이다. 그 결과, 거짓은 출력이 같아도 내부에 “conflict signature”를 남긴다고 주장한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 거짓과 틀림이 행동(출력) 수준에서 동시에 섞여 있어, 단순 분류로는 구분이 불가능하다는 점이다. 논문은 hidden states의 residual rank(압축 불가능성/경쟁 구조)를 층별로 평균내고, 두 응답(정직 vs 거짓) 중 더 높은 conflict 점수를 가진 쪽을 “거짓”으로 라벨 없이 선택하는 per-example 점수 체계를 만든다. 또한 residual rank가 길이/불확실성 같은 다른 요인과 섞이지 않도록 길이-일치 컨트롤, 전략적 자작 deception, concealment(숨기기) 지시까지 포함해 강건성을 확인한다.

- **Empirical Impact**: 실험 결과, deception forward pass는 naive liar 대비 residual rank가 2.1–2.3배 더 높게 나타나며(동일한 오답 출력 조건), GPT-2 계열에서 라벨 없이도 “어느 답이 거짓인지”를 100%로 식별한다. Qwen2.5 및 Phi-3-mini에서도 roleplay 기반 instructed deception이 18/18, 40/40, 34/34로 각 사실에서 거짓을 분리했고, Phi-3에서는 honest 답과 hallucination까지 AUC 1.0 수준으로 완벽 분리했다. 더 나아가 영어로 학습한 deception 검출기가 다른 모델 계열/포맷/언어로도 상당히 전이되며(특히 언어 축에서 교차언어 AUC 1.0), 다만 저자들은 동일 방향을 넣어 “선형으로 거짓을 생성”시키는 causal steering은 성공하지 못해(읽기는 되지만 쓰기는 어려움) 안전 관점에서 탐지-제어의 비대칭성도 함께 시사한다고 정리한다.



### Trust-Aware Multi-Agent Traceability: Confidence-Calibrated Knowledge Graphs for Consistent Software Artifact Managemen (https://arxiv.org/abs/2606.17203)
- **Prior Approaches**: 기존 traceability link recovery는 TF-IDF/LSI 같은 정보검색부터 transformer·LLM 기반, RAG까지 발전했지만 대부분 링크의 “정확도” 중심이며, downstream이 신뢰도를 실제로 소비할 방법은 부족합니다. Knowledge graph는 소프트웨어 아티팩트 관계를 구조화하는 데 유용했지만, regulated traceability에서 요구되는 파이프라인 간 일관성(오류 전파, 신뢰도 불일치)까지 다루는 연구는 드뭅니다. 멀티에이전트 협업(역할 분담)은 주로 코드 생성에 초점이어서, sequential pipeline에서 upstream 판단의 신뢰도가 downstream 행동을 어떻게 바꿔야 하는지에 대한 체계가 약했습니다.

- **Core Contribution**: 이 논문은 trust-aware coordination을 제안하며, 공유 knowledge graph를 “중앙 semantic memory”이자 “에이전트 간 조정 표면”으로 격상합니다. 핵심은 calibrate된 confidence score를 링크 예측 결과가 아니라 파이프라인 신뢰 신호로 사용하고, derivation-time(시딩)과 validation-time(재평가) confidence를 함께 비교해 불일치를 탐지하는 점입니다. 또한 confidence threshold gating, confidence divergence detection, conflict resolution을 하나의 일관성 프로토콜로 묶어 다운스트림의 오류 누적을 줄입니다.

- **Technical Challenges**: 기여를 실현하기 위한 가장 큰 기술적 난제는 (1) 링크 예측을 위한 후보 탐색 폭을 줄이면서도 (2) LLM 판단의 과신을 통제하고 (3) 파이프라인 간 신뢰도 불일치를 “구조화된 그래프 이벤트”로 표현하는 것입니다. 논문은 embedding 기반 retrieval로 후보를 k≈10개 수준으로 축소한 뒤, LLM multi-criteria(요구 커버리지/구현 구체성/용어 정렬/모순 부재)를 통해 2단계 traceability link prediction을 수행합니다. 시딩된 링크는 derivation confidence로 저장하고, validation 단계에서 재점수한 뒤 불일치가 생기면 Conflict node로 materialize하여 사람 검토까지 연결합니다.

- **Empirical Impact**: 자동차 소프트웨어 엔지니어링 사례(6개 서브시스템, 535개 아티팩트)에서 제안 파이프라인은 F1=0.769로 TF-IDF(0.349)와 LLM-only(0.669)를 크게 앞섰고, 후보 사전필터링만으로도 약 13% F1 개선을 보였습니다. 관계 유형별로는 REF_BY에서 보수적으로(오탐 억제), IMP_BY에서 가장 좋은 균형을 보였으며 VER_BY는 용어/표현 불일치 영향으로 개선 여지가 드러났습니다. ablation에서는 confidence calibration과 일관성 제어가 그래프 확장(링크 증가)과 구조적 무결성(충돌·누락 관리) 간 균형에 결정적임을 확인했고, conflict 검출은 튜닝(예: τ=0.5, ε=0.4) 설정에서 10/10 정답 충돌을 false positive 없이 포착했습니다.



### PowerOPD: Stabilizing On-Policy Distillation with Bounded Power Transformation (https://arxiv.org/abs/2606.17199)
- **Prior Approaches**: 온폴리 distillation(OPD)은 학생이 생성한 궤적을 기반으로 teacher–student를 reverse-KL로 맞추며, sparse-reward RL보다 토큰 단위의 밀집 감독을 제공해 SFT와 RL 사이 교량으로 자리잡았다. 다만 실무에서는 full-vocabulary 계산이 비싸서 student-sampled 토큰으로 reverse-KL을 단일 샘플 Monte Carlo로 근사한 vanilla OPD가 사실상 표준이 됐다. 이때 근사는 비용을 줄이지만, 실제 학습에서 샘플 비효율·생성 불안정·성능 갭 같은 병리 현상이 반복적으로 관측된다.

- **Core Contribution**: 이 논문은 vanilla OPD의 문제 원인이 reward인 teacher–student log-probability ratio(로그 비율)가 생성하는 “무한(boundlessness) 보상”의 고분산에 있음을 진단한다. 그리고 log-ratio의 α→0 퇴화(limit)를 포함하되, Box-Cox power transformation으로 보상을 “유계(bounded) + 부호 일관(sign-consistent)” 형태로 설계한 PowerOPD를 제안한다. 즉, transform-then-subtract 구조는 유지하면서도 극단 보상이 기울기 신호를 망가뜨리지 않게 만드는 것이 핵심이다.

- **Technical Challenges**: 핵심 기술 난제는 Monte Carlo 근사로 인해 생기는 드문 토큰·초기 위치 이벤트가 학습 신호를 폭발시킬 때, 방향성(teacher가 더 높으면 양의 보상 등)을 잃지 않으면서 보상을 유계로 제한하는 것이다. 저자들은 post-hoc 보상 안정화(clipping, tanh, z-score 등)가 로그 비율 자체의 구조적 무한성에 의해 이미 왜곡된 뒤에는 효과가 제한적임을 보이고, Box-Cox 기반의 PowerOPD로 처음부터 reward 스케일을 통제한다. 또한 α가 커질수록 정확도와 응답 길이 안정성이 좋아지고, vanilla OPD 대비 gradient norm을 3,000배 이상 줄이는 경향을 보인다.

- **Empirical Impact**: PowerOPD는 수학 추론 벤치마크 6종과 Qwen3 teacher–student 조합 4쌍에서 평가되었으며, vanilla OPD 대비 Avg@8/Pass@8 기준 최대 +6.37/+5.71 향상을 보고한다. 또한 post-hoc 안정화보다도 개선(+3.01/+3.54), full-vocabulary OPD와 비교해도 격차를 더 줄이면서(wall-clock 시간 59.2% 절감, peak GPU 메모리 23.1% 감소) 효율성과 성능을 동시에 얻는다. 더 큰 α는 보상 분포를 유리하게 조절해 모델이 낮게 주는 토큰 학습 비중을 억제하고, teacher나 student가 중요하게 보는 토큰에 학습을 집중시키며 결과적으로 학습 역학을 안정화한다.



### Cluster-Aware Dual-Level Test Specification Generation for Large-Scale Automotive Software Requirements (https://arxiv.org/abs/2606.17197)
- **Prior Approaches**: 기존 NLP4RE(Requirements Engineering) 연구는 요구사항을 요약하거나(clustering+summarization) 임베딩 기반으로 군집화해 정리하는 데 집중해 왔습니다. 그러나 자동차 SRE에서 중요한 ASPICE SWE.6 ‘요구사항별 검증 가능한 테스트 명세’와 ISO 26262의 ‘요구사항 간 의존성/통합 커버리지’를 동시에 만족시키려면, 요구사항을 개별 단위로만 처리하는 방식이 통합 테스트 누락과 중복 테스트를 낳기 쉽습니다. 또 전체 코퍼스를 한 번에 LLM에 넣는 접근은 context-window 한계로 인해 표준 근거와 통합 관점이 끊기는 문제가 있습니다.

- **Core Contribution**: 이 논문은 대규모(수천 요구사항) 환경에서 ASPICE SWE.6 요구를 만족하는 테스트 명세 자동화를 위한 ‘Cluster-then-Summarize’ 파이프라인을 제안합니다. 먼저 요구사항을 문장 임베딩으로 표현한 뒤 UMAP+HDBSCAN로 의미 군집을 만들고, 각 군집을 multi-level map-reduce 요약으로 압축합니다. 그다음 군집 기반 문맥을 LLM 호출에 주입해 요구사항 단위 검증 테스트와 군집 단위 통합(integration) 테스트를 함께 생성합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 수천 개 요구사항을 다룰 때 군집 수/밀도를 자동으로 안정화하고 (2) LLM 호출마다 문맥을 ‘충분히’ 주되 context-window를 넘지 않게 제한하는 것입니다. 이를 위해 min_cluster_size를 normalized Silhouette과 Calinski–Harabasz 점수를 결합한 품질 기준으로 자동 선택하고, HDBSCAN noise를 유사한 클러스터로 재배정해 커버리지를 유지합니다. 또한 nearby-cluster context 메커니즘으로 각 LLM 호출에 제한된 교차-특징 정보를 제공하고, Retrieval-Augmented Generation으로 ISO 26262와 ASPICE 관련 근거를 grounding 하면서 정량 임계값·ASIL 등급 보존을 목표로 합니다.

- **Empirical Impact**: 자동차 요구사항 데이터셋 여러 규모(D1~D7)에서 평가한 결과, 제안 방식은 기준선 대비 통합 테스트 커버리지를 개선하면서 요약 충실도(정량/안전 무결성)도 유지하는 것으로 나타났습니다. 특히 요약 전략 비교에서는 map-reduce가 단일 패스보다 ROUGE-L, BERTScore 및 전문가 평가(완전성/정량 보존)에서 가장 높은 성능을 보였습니다. 또한 클러스터 문맥을 주입한 조건이 semantic boundary(Overlap Error 등) 위반을 줄이고 테스트의 구체성과 다양성을 높여, ‘규모 확장’과 ‘정확한 통합 검증’ 사이의 균형을 실증적으로 보여줍니다.



### Statistical Foundations of LLM-based A/B Testing: A Surrogacy Framework for Human Causal Inferenc (https://arxiv.org/abs/2606.17165)
- **Prior Approaches**: LLM을 A/B 테스트의 대체 참가자로 쓰려는 시도는 빠르고 저렴하게 실험을 돌릴 수 있다는 기대를 바탕으로 빠르게 확산되고 있다. 하지만 기존 연구는 주로 예측/재현 정확도에 초점을 둬, LLM 결과로부터 인간 모집단의 causal effect를 언제 신뢰할 수 있는지의 조건을 명확히 정리하진 못했다. 또한 LLM은 프롬프트·temperature·샘플링 잡음에 민감해, 인과추론에서의 편향과 분산 문제가 쉽게 누적될 수 있다는 한계가 보고되어 왔다.

- **Core Contribution**: 이 논문은 LLM 출력을 인간 결과의 surrogate(대리 지표)로 보고, surrogate-endpoint theory를 LLM 기반 A/B 테스트에 맞게 통계적 틀로 확장한다. 분포 동등성은 어렵다는 전제하에, 인간-LLM 사이의 calibration(보정)과 surrogacy·comparability 조건이 성립하면 LLM에서 추정한 효과가 인간 ATE로 식별됨을 보인다. 조건이 깨질 때는 효과가 부분 식별(partially identified)되며, 사후(과거) 실험 데이터를 통해 surrogacy를 falsify할 수 있는 진단과 worst-case bias 상한도 제시한다.

- **Technical Challenges**: 핵심 난제는 LLM 출력이 인간 결과와 분포가 다를 때도 ATE를 제대로 옮길 수 있는 ‘정확한 통계 조건’이 무엇인지, 그리고 이를 실제 추정 절차에 어떻게 연결하느냐이다. 논문은 인간 데이터(P=0)에서 (X,Y*)→Y 매핑 μ를 학습하고, LLM 데이터(P=1)에서 이를 평가하는 방식으로 식별을 구성하며 comparability로 μ의 관계 안정성을 요구한다. 또 LLM의 확률적 샘플링이 편향과 분산을 동시에 키우지만, 단위당 여러 draw를 평균내는 평균 surrogate로 이를 완화할 수 있음을 이론적으로 정리한다.

- **Empirical Impact**: 시뮬레이션과 실제 데이터(Upworthy headline A/B 테스트 적용)에서 제안한 방법이 이론적 기대에 부합하는 성능을 보이며, 특히 과거 실험 기반 진단의 유용성을 보여준다. 또한 temperature·프롬프트·LLM 선택 같은 설계 변수가 surrogacy 성립성과 추정 품질에 영향을 준다는 점을 실무 권고로 연결한다. 결론적으로, 새로운 개입에 대해서는 LLM surrogate의 타당성을 ‘검증’하기 어렵고 인간 실험이 계속 필수라는 메시지를 통계적으로 강화한다.



### PromptMN: Pseudo Prompting Languag (https://arxiv.org/abs/2606.17164)
Comments:
          32 pages, 2 figures

- **Prior Approaches**: 기존 prompt engineering은 role, 제약, 예시, few-shot 같은 기법을 제안하지만, 실제로는 자유형 prose에 핵심 의도가 묻히기 쉬워 해석이 흔들린다는 한계가 반복적으로 보고됩니다. 특히 agentic 워크플로우나 SDLC처럼 한 번의 오해가 연쇄 실패로 번지면, 문제는 모델 능력보다 컨텍스트 모호성에서 시작되는 경우가 많습니다.

- **Core Contribution**: 이 논문은 %-prefixed typed directives로 역할(%role), 목표(%goal), 요구(%req), 우선순위·제약(%mustnot 등), 계획(%plan/%showplan), 입출력 경계(%in/%out) 등을 자연어에 “주석처럼” 구조화하는 DSL인 PromptMN을 제안합니다. 또한 reverse prompt engineering에서 모델이 원하는 산출물을 PromptMN 형태로 되돌려 작성하게 함으로써, 모델이 추론한 역할·제약·누락 가정을 사람이 먼저 검토할 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작성 순서가 달라도 지시문을 일관되게 해석해야 하고, (2) 과도한 개발자 수준의 문법 부담 없이 리뷰·재사용 가능한 구조를 제공해야 한다는 점입니다. 이를 위해 PromptMN은 의미적 정렬(semantic resolution)로 지시문을 역할 기준으로 재구성해 해석 순서를 안정화하고, 작은 키워드 셋과 블록/구분자(∞\infty …∞\infty, {…}, ;)로 파싱·검토 가능성을 높였습니다.

- **Empirical Impact**: Claude Fable 5, Claude Opus 4.8, Gemini 3.1 Pro, GPT-5.5 등 여러 frontier 모델에서 fine-tuning 없이도 %repeat, 조건, method, prime-checking 같은 복잡 구조를 정확히 해석·실행하는 사례를 보였습니다. 대규모 벤치마크나 사용자 연구는 향후 과제로 남지만, SDLC 시나리오와 Snake game처럼 %showplan/%trace로 초기에 검토 지점을 제공한다는 점에서 사람-모델 협업의 신뢰성을 높일 실용적 방향을 제시합니다.



### Agentic Discovery of Non-Canonical Antimicrobial Peptides with AMPGAN v3 (https://arxiv.org/abs/2606.17127)
Comments:
          Presented at the GenBio Workshop, ICML 2026

- **Prior Approaches**: AMP 발견을 위한 생성 모델은 VAE나 GAN 같은 구조로 서열 유사성과 활성 예측기를 함께 다루며 발전해 왔고, 최근에는 LLM 기반 few-shot/prompt tuning으로 성능이 확대되는 흐름이 있습니다. 다만 (1) 생성된 AMP가 대부분 alpha-helical에 치우치며 프로테아제 분해·구조 불안정 문제가 있고, (2) D-amino acids나 N/C-terminus amidation 같은 화학적 변형을 학습 범위에서 제외하는 경우가 많습니다. 또한 대표적으로 AMPGAN 계열은 conditioning은 가능해도 GAN 학습 불안정(모드 붕괴, 단일 토큰 반복 등)과 L-amino acid 중심의 화학 스코프 한계가 두드러집니다.

- **Core Contribution**: 이 논문은 AMPGAN v3로, 다목표 conditional GAN을 통해 생성 어휘(vocabulary)를 D-amino acids 및 N/C-terminus modifications(예: amidation)까지 확장하는 것을 핵심으로 제시합니다. 동시에 “진짜 peptide처럼 보이는 것”과 “목표 MIC에 부합하는 활성”을 서로 다른 두 판별기(discriminator)로 분리해 학습 안정성을 크게 개선하고, 외부 분류기 기준에서 기존 생성 모델을 능가합니다. 더 나아가 PepCraft라는 multi-agent 프레임워크를 도입해 생성-필터링-검증을 end-to-end 파이프라인으로 묶어 후속 큐레이션 비용을 낮추는 방향을 제시합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 GAN의 학습 불안정성을 유지하면서도 비자연 화학(terminal modifications, D-amino acids)을 포함한 서열을 생성하도록 supervision을 설계하는 것입니다. AMPGAN v3는 adversarial 판별기와 activity-aware 판별기를 분리하고, Dmic 판별기는 실험 MIC 라벨을 가진 real sequence에 대해서만 업데이트하며 generator에는 frozen 상태로 점수를 제공해 안정성을 확보합니다. 또한 Gumbel-Softmax로 이산 서열 샘플링을 미분 가능하게 만들고, 길이/토큰 수준의 복원 손실을 함께 두어 early training의 붕괴를 줄였습니다.

- **Empirical Impact**: 생성 후보 5종을 시험관(in vitro)에서 합성·검증했으며, 그중 2종이 Gram-positive 균에 대해 항균 활성을 보였습니다. 최우수 후보는 B. subtilis에 대해 MIC 8 μg/mL까지 도달했지만, conditioning에 사용한 표적 종(E. coli 또는 S. aureus)에는 오히려 활성이 없어서 “표적 종 조건이 곧 종특이성을 보장하지는 않음”도 함께 관찰됩니다. 아울러 학습 안정성은 30개 random seed 실험에서 AMPGAN v2의 성공률(약 3/30) 대비 AMPGAN v3가 더 높은 유효 출력 비율을 보였고, PepCraft의 우선순위 추천이 in vitro 결과와도 정합성을 보이며 agentic 파이프라인의 실용성을 시사합니다.



### Vibrato Expression Control for Singing Voice Conversion with Improving Independent Contro (https://arxiv.org/abs/2606.17126)
Comments:
          Accepted to IEEE Transactions on Audio, Speech, and Language Processing (TASLP)

- **Prior Approaches**: 기존 singing voice conversion(SVC)은 언어/멜로디 정보는 유지하고 음색·prosody·emotion 등을 바꾸는 방향으로 발전했지만, “표현성”을 위한 Singing Style Conversion(SSC)은 동적 속성들이 pitch·energy·timbre에서 서로 강하게 결합돼 제어가 어렵다는 한계가 컸습니다. SVCC 2025의 베이스라인들은 pitch 스타일을 부분적으로만 다루거나(예: vibrato/멜로디 중심 누락) 음색을 글로벌 수준에서만 분리해 국소적인 스타일(예: vibrato rate, vocal fry)까지 독립 제어하기엔 제약이 남아 있습니다. VibE-SVC는 vibrato를 고주파 F0 contour 예측으로 다루며 진전을 보였지만, pitch-에너지 결합 잔여, rate/extent 분리 미흡, zero-shot 스타일 라벨 부재 같은 문제가 남았습니다.

- **Core Contribution**: VibE-SVC2는 vibrato(피치 스타일)와 phonation(음색 스타일)을 동시에 다루면서도 구조적으로 분리해, pitch 스타일과 timbre 스타일을 각각 독립적으로 제어하는 통합 프레임워크를 제안합니다. 이를 위해 Energy Style Converter로 pitch-energy entanglement의 잔여 누수를 줄이고, Zero-shot Pitch Style Converter로 참조 오디오 없이도(라벨 없이) pitch 스타일을 전이할 수 있게 했습니다. 또한 vibrato extent에 더해 vibrato rate까지 inference 시 독립 제어하도록 temporal scaling 기반 rate scaling을 도입했습니다.

- **Technical Challenges**: 핵심 기술 난점은 vibrato가 단순 F0 진동이 아니라 loudness의 주기적 진동과 동기화되어 pitch와 energy가 함께 얽힌다는 점입니다. 논문은 DWT로 F0의 저주파/고주파를 구조적으로 분해한 뒤, 고주파는 pitch style로, 에너지 contour의 고주파 잔여는 Energy Style Converter가 흡수하도록 설계해 entanglement을 구조적으로 완화합니다. 또 vocal fry처럼 subharmonic 때문에 F0 추정이 깨지면 pitch-jump 아티팩트가 음색 변환까지 망가뜨리므로, ΔΔF0(1차 차분) 기반 Subharmonic Correction(SHC)으로 F0 contour를 보정해 자연성을 확보합니다.

- **Empirical Impact**: VocalSet 및 GTSinger 등에서 객관/주관 평가를 수행해 VibE-SVC2가 기존 방법 대비 pitch·timbre 스타일 정확도와 제어성에서 우수함을 보였다고 보고합니다. 특히 vibrato에서는 extent뿐 아니라 rate까지 독립적으로 조절되는 점이 결과에서 드러나며, vocal fry와 같은 까다로운 phonation에서도 pitch-jump을 완화해 음색 품질이 개선되는 효과가 확인됩니다. 전반적으로 “국소적 동적 singing style”을 라벨 의존 없이(Zero-shot) 다루면서도 서로 간섭을 줄이는 통합 SSC 접근이 관련 분야에 실용적 기준을 제시한다는 의미가 있습니다.



### LineageMark: Multi-user White-box Watermarking for Contribution Tracing in Model Derivation Chains (https://arxiv.org/abs/2606.17123)
Comments:
          14 pages, 2 figures

- **Prior Approaches**: 기존 LLM 워터마킹은 주로 단일 사용자가 한 번 삽입한 뒤(또는 단일 소유권 확인) 출력 행동/생성문 통계로 검증하거나, 특정 내부 파라미터에 신호를 직접 심는 white-box 방식에 집중해 왔습니다. 하지만 파라미터가 반복적으로 drift·편집되는 모델 derivation chain에서는 역사 워터마크가 약화되고, 새 워터마크가 기존 carrier와 겹치며 신호 간 간섭이 커집니다.
또한 key 기반으로 서로 다른 기여자가 독립적으로 “각자 워터마크만” 검증하면서, 동시에 다중 워터마크를 incremental하게 공존시키는 요구를 명시적으로 다루지 못한 경우가 많았습니다.

- **Core Contribution**: 이 논문은 multi-user white-box watermarking을 model derivation chain에 맞춰 정식화하고, (1) lineage preservation, (2) incremental embedding, (3) key-only independent verification을 핵심 요건으로 제시합니다. 이를 만족하도록 LineageMark는 각 워터마크 비트를 “개별 가중치”가 아니라 여러 weight coordinate에 대한 sign-projection statistic으로 인코딩합니다.
또한 안정적인 carrier를 먼저 선별하고, margin-aware 임베딩과 redundant projection을 결합해 이후 파인튜닝/재워터마킹에서도 역사 신호를 더 잘 유지하도록 설계했습니다.

- **Technical Challenges**: derivation chain에서의 가장 큰 문제는 이후 fine-tuning이 워터마크 통계를 바꿔 역사 비트의 부호(sign)를 뒤집거나, 새 삽입이 기존 carrier를 교란해 간섭을 유발한다는 점입니다. LineageMark는 (i) Fisher 기반 functional relevance와 perturbation sensitivity를 이용한 stable carrier 선택, (ii) 키로부터 생성되는 projection group으로 국소 변화의 영향을 통계적으로 흡수, (iii) margin 제약으로 결정 경계로부터의 여유를 확보함으로써 이를 완화합니다.
또한 각 기여자가 private key만으로 detector(투영 검출기)를 재구성해 원본 모델/캘리브레이션 데이터/다른 키 없이 독립 검증이 가능하도록 설계했습니다.

- **Empirical Impact**: 연속 full-parameter fine-tuning으로 구성한 multi-stage derivation chain 실험에서 LineageMark는 역사 워터마크와 새로 삽입된 워터마크 모두에 대해 높은 extraction 정확도를 유지하는 것으로 보고됩니다. 기준 white-box watermarking과 비교해 역사 decay가 낮고 multi-user interference도 덜한 경향을 보였으며, fine-tuning, re-watermarking, quantization, pruning 같은 다양한 수정에도 강건합니다.
결과적으로 오픈 LLM 생태계에서 파생 모델의 기여자 이력 검증과 지식재산권 보호에 바로 활용 가능한 “다중 사용자 계보(라인리지) 워터마킹” 실전 가능성을 높였다는 점에서 의미가 큽니다.



### TrustErase: Auditable Instant Machine Unlearning with Passport-Embedded Representations (https://arxiv.org/abs/2606.17122)
- **Prior Approaches**: 기존 machine unlearning은 retraining/SISA, gradient-based 삭제, distillation 기반 DELETE처럼 추가 최적화 단계가 필요하거나(비용·지연 부담), 혹은 influence functions, L2UL, Boundary Shrink처럼 근사/부분 삭제에 머물러 데이터 가용성이나 증명 가능성에서 제약이 있었습니다. 또한 prompt 기반 Pre-Forgettable Models는 data-free 즉시 삭제가 가능하지만, 배포 후 “정확히 잊었는지”를 구조적으로 보증하는 auditable 인증이 약합니다. 결과적으로 많은 방법이 ‘즉시성·데이터 비의존·검증(감사)·비용’ 중 일부를 포기해왔습니다.

- **Core Contribution**: TrustErase는 passport-embedded 표현(패스포트)을 암호학적 키처럼 다루는 verifiable, data-free unlearning 프레임워크를 제안합니다. LoRA의 파라미터 효율 적응층에 passport를 모듈 단위로 결합하고, 잊을 대상(클래스/데이터셋/조합)에 해당하는 passport를 deactivation(활성 해제)하는 방식으로 retraining·fine-tuning·원본 데이터 접근 없이 즉시 forgetting을 구현합니다. 더 나아가 SVD 기반 hiding으로 passport를 가리되, 권한 있는 감사 기관이 재구성·검증해 “어떤 unlearning이 적용되었는지”를 provably compliance 관점에서 확인할 수 있게 합니다.

- **Technical Challenges**: 핵심 난제는 (1) 특정 데이터 영향만 제거하면서도 (2) 데이터 없이 즉시 동작하게 만들고 (3) 외부에 배포된 모델에서 잊음이 정확히 적용됐는지 구조적으로 감사 가능해야 한다는 점입니다. TrustErase는 shared LoRA 요인 위에 task-specific passport를 고정 주입해 다중 forgetting 구성을 한 모델 공간에 공존시키고, DELETE 계열의 masked-distribution forgetting 목적을 단일 학습 단계에 통합해 teacher 없이도 목표 분포를 유도합니다. 또한 조합(미리 정해지지 않은 forget-set)에 대해 hypernetwork가 atomic passport들을 합성하고, SVD로 passport를 은닉한 뒤 reconstructed passport의 구조/기능이 허용 오차 ϵT 내인지로 인증하는 authority-mediated audit를 설계합니다.

- **Empirical Impact**: MNIST, CIFAR-10, CIFAR-100에서 TrustErase는 data-free 조건에서도 DELETE, L2UL, Boundary Shrink 등 SOTA급 unlearning 성능을 동등 이상으로 달성하며, 잊을 대상에 대한 forgetting 정확도와 retained 성능을 함께 유지합니다. 특히 단일 클래스뿐 아니라 다중 클래스(예: CIFAR-100에서 최대 20클래스 제거)에서도 forgetting 정확도가 0에 수렴하는 구성들이 보고되며, 이는 “train-once, forget-anytime” 패러다임의 실용성을 뒷받침합니다. 더불어 hypernetwork 기반 unseen 조합 일반화와 passport perturbation에 대한 verification robustness 평가까지 제시되어, 규제/감사 요구에 대응하는 accountable unlearning으로서 의미가 큽니다.



### Graph neural networks at war: integrating cybersecurity and drone intelligence in the Israeli-Iranian conflic (https://arxiv.org/abs/2606.17119)
- **Prior Approaches**: 기존 물리-사이버 시스템 보안은 주로 시그니처 기반 탐지나 단일 네트워크 트래픽 분석에 의존해, 침입의 구조적 맥락을 충분히 반영하기 어렵다는 한계가 있었습니다. 드론(무인기) 대응 역시 감지 결과를 기계적으로 제어에 연결하는 방식이 많아, 상황 인지와 군집 협업이 즉각적으로 정교화되지 못했습니다. 또한 그래프 형태의 관계를 학습에 직접 통합하기보다, 별도 모듈로 처리해 end-to-end 대응이 약했습니다.

- **Core Contribution**: 이 논문은 GNN(그래프 신경망)을 활용해 침입 탐지와 드론 응답을 잇는 통합 절차를 제안합니다. 그래프 신경망의 구조적 이해를 바탕으로, 침입 탐지 시스템이 네트워크 구조를 학습해 악성 활동을 식별하고, 그 결과를 드론의 대응 조치로 연결합니다. 즉, 그래프 기반 학습으로 상황 인지(situational awareness)·군집 조정·적응적 기동을 동시에 지원하는 설계를 내놓습니다.

- **Technical Challenges**: 핵심 기술 난제는 동적인 물리-사이버 환경에서 네트워크/행동 관계를 어떤 그래프로 구성하고, 그 표현을 실시간 탐지-응답 파이프라인으로 연결할지입니다. 논문은 침입과 UAV 반응을 유도하는 에뮬레이션 기반 케이스 스터디를 통해, 사이버 공격이 발생했을 때 그래프 기반 학습이 악성 패턴을 포착하도록 모델링했습니다. 또한 성능 비교에서 GraphSAGE 네트워크가 GCN 및 GAT보다 동일 상황에서 더 유리함을 보여, 실제 적용 관점의 안정적 학습을 뒷받침합니다.

- **Empirical Impact**: 에뮬레이션 결과, 탐지 성능은 detection rate 94.2, ROC AUC 평균 0.955로 높게 나타났고 평균 응답 시간은 1.4초로 보고됐습니다. 그래프 기반 학습은 공격 상황에서 드론의 상황 인지와 swarm coordination, adaptive maneuver에 도움이 되었음을 실험적으로 확인했습니다. 비교 실험에서도 GraphSAGE가 GCN/GAT보다 효과적이어서, 동적 cyber-physical system에서 침입 회피와 즉각 대응을 강화하는 접근으로 의미가 큽니다.



### MODE: Modality-Decomposed Expert-Level Mixed-Precision Quantization for MoE Multimodal LLMs (https://arxiv.org/abs/2606.17118)
Comments:
          18 pages, 8 figures

- **Prior Approaches**: 기존 PTQ는 멀티모달에서 텍스트-비전 토큰의 양자화 민감도 차이만 다루거나, MoE에선 expert 중요도를 activation frequency로 추정했지만 MoE-MLLM의 모달리티 불균형과 희소 라우팅 구조를 제대로 결합하지 못했다. 그 결과 expert-level mixed-precision quantization을 MoE-MLLM에 그대로 옮길 때 성능 저하가 커졌다. 특히 글로벌 빈도 집계가 비전 토큰에 의해 지배되면서 텍스트에 중요한 expert가 과소평가되는 문제가 반복됐다.

- **Core Contribution**: 이 논문은 MoE-MLLM용 modality-decomposed expert-level mixed-precision quantization 프레임워크 MODE를 제안한다. MODE는 (1) 텍스트/키 비전 토큰별로 expert selection frequency를 분해·정규화하고, (2) 비전 내부에서 중복( redundant ) 토큰을 걸러 denoised visual frequency를 만들며, (3) 모달리티별 quantization sensitivity도 함께 측정해 중요도를 더 정밀하게 추정한다. 마지막으로 해당 신호를 Integer Linear Programming(ILP)로 넣어 예산 내에서 expert별 bit-width를 최적으로 배분한다.

- **Technical Challenges**: 핵심 기술적 난제는 “빈도 기반 importance”가 MoE-MLLM에서는 모달리티 편향(교차-모달)과 비전 중복 편향(내부-비전) 때문에 신뢰도를 잃는다는 점이다. MODE는 텍스트 토큰과 (SparseVLM 기준으로) key vision tokens(레이어별 attention 기준 상위 20%)를 따로 보고, 각 expert를 후보 비트폭으로 부분 양자화했을 때 KL divergence 기반으로 모달리티별 민감도를 분리 측정해 이 편향을 상쇄한다. 그 결과 expert별 “얼마나 자주 쓰이는가”뿐 아니라 “얼마나 손상이 큰가”까지 묶어 ILP가 bit-width를 결정하게 만든다.

- **Empirical Impact**: 실험에서 MODE는 여러 MoE-MLLM 계열과 10개 멀티모달 벤치마크에서 기존 PTQ 대비 일관되게 우수했으며, W3A16에서 평균 성능 저하를 2.9% 이내로 제한했다. 특히 2-bit 같은 극저비트 설정에서 경쟁 방법 대비 격차를 더 크게 줄이며, rotation-based quantization(예: QuaRot)과도 조합해 추가 개선이 가능함을 보였다. 또한 calibration set을 바꿔도 정확도 변화가 거의 없어 practical deployment 관점에서 신호의 견고성도 확인됐다.



### Probing, Fusion, and Trustworthiness: A Systematic Evaluation of Foundation Model Representations for Multimodal Cancer Analysis (https://arxiv.org/abs/2606.17115)
- **Prior Approaches**: 기존 연구는 한 모달리티별 인코더로 표현을 뽑고, 이후 단일(modality) 혹은 다중(modality) 학습 모듈로 분류를 수행하는 전형적 파이프라인에 집중해 왔다. 또한 의료 Foundation model(FM)은 전이 성능이 좋다고 보고되지만, 공개 벤치마크 중심이라 사전학습 분포와의 근접/중복 가능성 때문에 실제 상용 데이터(산업 수집 분포)에서의 일반화는 상대적으로 덜 다뤄졌다. 다중모달 융합은 concatenation, cross-modal attention, late fusion 등 다양한 전략이 제안됐지만, “융합이 언제 이득인지”에 대한 조건부 분석은 제한적이었다.

- **Core Contribution**: 이 논문은 whole-slide image(H&E WSI)와 transcriptomic profile(omics)을 함께 보유한 두 실사용(상용) 코호트 IH-BC, IH-NSCLC에서 FM 기반 표현의 out-of-distribution(OOD) 전이를 체계적으로 평가한다. 이미지 5개(FM 타일 임베딩)와 omics 3개(UCE, scVI, PCA) 표현을 8개 분류 과제에 대해 unimodal probing으로 먼저 비교하고, 이어서 paired 표현을 기반으로 3가지 이미지-omics fusion이 추가 성능을 주는지 검증한다. 마지막으로 conformal prediction으로 예측 신뢰도(coverage/셋 크기/단일 선택률)를 점검해 임상 보조에 필요한 “불확실성-aware 추론” 가치를 함께 보여준다.

- **Technical Challenges**: 핵심 난제는 (1) 상용/산업 데이터처럼 분포가 바뀐 환경에서 FM 표현이 여전히 유효한지, (2) 융합이 항상 이득이 아니라면 어떤 상황에서 이득/손해가 갈리는지, (3) 높은 stakes 영역에서 단일 점 예측이 얼마나 안전한지 불확실성까지 정량화하는 것이다. 저자들은 frozen representation(사전학습 인코더 고정) 위에 동일 과제 세트를 두고 분류기를 학습해 unimodal vs multimodal의 성능을 공정 비교했으며, 융합은 LateMIL, CONTACT, MCAT 3종을 paired 표현으로 구성해 신호 우세/상호보완 상황을 관찰했다. 또한 split conformal prediction을 적용해 모델 아키텍처와 무관하게 예측 set의 coverage 보장을 확인하고, 실패 케이스에서도 정답이 셋 내에 포함되는 rescue rate로 실사용 관점의 안전망을 제시했다.

- **Empirical Impact**: 실험 결과, 이미지 FM 표현은 OOD에서도 경쟁력 있는 성능을 보이며, 특히 이미지와 omics가 과제에 따라 상호보완적 예측 신호를 제공함이 확인된다. 다만 multimodal fusion은 모든 태스크에서 일관되게 이득이 아니고, 단일 모달이 예측에 지배적일 때는 오히려 성능이 정체되거나 저하될 수 있으며, LateMIL이 비교적 안정적이라는 패턴이 나타난다. conformal prediction 평가에서는 대부분의 다중분류 과제에서 90% 목표 coverage에 부합하거나 초과하며, top-1이 틀린 경우에도 다수 상황에서 true diagnosis가 prediction set 안에서 “복구”되는 비율이 높아 불확실성 기반 임상 보조의 실질적 효용을 강화한다.



### An Evaluation of Data Leakage Risks in Tool-Using LLM Agents in Realistic Scenarios (https://arxiv.org/abs/2606.17114)
- **Prior Approaches**: 기존 에이전트 안전 연구는 주로 prompt injection, jailbreak처럼 적대적 상황에서의 정보 유출(exfiltration)에 초점이 맞춰져 있었다. 반면 실제 업무에서는 요청이 ‘선의(benign)’여도 에이전트가 민감도를 오판하거나 불필요한 정보를 꺼내고 잘못된 수신자에게 전달하는 등 비적대적 데이터 누출이 발생할 수 있다.

- **Core Contribution**: 싱가포르 AI Safety Institute(SG AISI)와 한국 AI Safety Institute(KR AISI)가 에이전트 데이터 누출을 비적대적이지만 현실적인 12개 작업(고객 지원, DevOps, 웹 자동화, 엔터프라이즈/개인 생산성)을 통해 공동 평가했다. 데이터 누출 위험을 lack of data awareness, audience awareness, policy compliance, data minimization, access-boundary awareness의 5유형으로 체계화하고, 두 기관이 독립 환경에서 동일 시나리오를 검증하는 방법론을 제시한다.

- **Technical Challenges**: 핵심 난제는 에이전트의 최종 답이 ‘그럴듯’해도, 실행 궤적(trajectory) 중 어떤 데이터가 조회·전달되었는지 전체 과정을 함께 판정해야 한다는 점이다. 이를 위해 ReAct 루프 기반으로 단계별 tool 호출을 기록하고, user를 별도 LLM으로 시뮬레이션하며, MCP(Model Context Protocol) 도구 환경과 작업별 사실형 rubrics(LLM-judge)를 결합해 ‘스크립트 응답’이 아닌 실제 처리 흐름을 평가한다.

- **Empirical Impact**: 3개 테스트 에이전트 모두 12개 시나리오 전 구간에서 ‘완전 정확+완전 안전’을 달성하지 못했고, 성공한 작업에서도 데이터 핸들링 실패(불필요 정보 접근/부적절 수신자 공개)가 자주 동반된 것으로 나타났다. 또한 claim-action mismatch, simulation-aware 행태, user-simulator role reversal, 자동 채점의 해석 갭이 관찰되어, capability와 데이터 핸들링 안전을 분리 측정해야 한다는 점과 operational data leakage가 1차 안전 이슈임을 실증적으로 강조한다.



### Timestamp-Aware Spatio-Temporal Graph Contrastive Learning for Network Intrusion Detection (https://arxiv.org/abs/2606.17109)
- **Prior Approaches**: 기존 GNN 기반 NIDS는 트래픽 플로우 간 관계(그래프 구조)를 잘 모델링하지만, 대부분 시간을 독립적으로 취급해 공격 패턴이 변하는 상황에서 대응력이 떨어진다는 한계가 있었다. 또한 supervised나 semi-supervised 중심 학습은 미출현 공격에 대한 일반화가 제한되는 문제가 있었다. 일부는 시간 정보를 넣더라도 attention처럼 계산 비용이 큰 설계를 쓰는 경우가 있어 효율이 제약됐다.

- **Core Contribution**: 이 논문은 self-supervised GNN 기반 NIDS 프레임워크를 제안하며, 실제 timestamp를 명시적으로 활용해 시간 의존성을 표현 학습에 반영하는 점이 핵심이다. timestamp 기준으로 temporal graph들을 구성한 뒤 E-GraphSAGE와 LSTM 인코더로 시간 정보와 공간(구조) 의존성을 함께 추출한다. 여기에 multi-view graph contrastive learning으로 시간 연속성, 구조 일관성, 특징 정합성을 동시에 학습하고, gradient-norm 기반 가중치로 대비 손실의 중요도를 자동 조절한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 timestamp의 시간적 연속성과 그래프 구조 정보를 함께 학습하되, time-costly attention 없이도 성능을 유지하는 것이다. 저자들은 temporal graph를 시계열로 분해해 E-GraphSAGE(공간 구조)와 LSTM(시간 흐름)을 결합하는 인코딩으로 이를 해결했다. 또한 temporal/spatial/feature 관점의 contrast를 함께 설계하되, 대비 항의 가중치가 학습을 흔들지 않도록 gradient-norm 기반 adaptive weighting으로 균형을 맞췄다.

- **Empirical Impact**: timestamp가 실제로 포함된 4개 대표 NIDS 데이터셋에서 실험했을 때, 제안 방법은 기존 self-supervised 대비 성능을 유의미하게 개선했다. 동시에 supervised 상태-of-the-art GNN과 비슷한 수준의 성능을 달성하면서도 계산 효율이 높다고 보고한다. 이는 시간 변형 공격을 다루는 NIDS에서 self-supervised GNN의 실용 가능성을 넓히는 결과로 해석된다.



### Models Take Notes at Prefill: KV Cache Can Be Editable and Composab (https://arxiv.org/abs/2606.17107)
- **Prior Approaches**: 기존 prefix caching은 정확히 동일한 prefix에 대해서만 KV를 재사용해, prefix 내부의 토큰 하나가 바뀌면 이후 토큰 전체가 무효화됩니다. 그래서 실무에선 동적(변하는) 내용을 뒤로 몰아 “호이스팅”해야 해서, 중첩 프롬프트나 동적으로 조립되는 컨텍스트에서 제약이 커집니다. 한편 weight editing(예: ROME, MEMIT)이나 fine-tuning은 전역적으로 오염이 생겨 per-request의 mutable state에는 부적합하다는 한계도 제기됩니다.

- **Core Contribution**: 이 논문은 KV 캐시를 단순한 중간활성 저장이 아니라 “메모된 결론(notes)”의 공책으로 재해석하고, 왜 필드 KV만 바꾸면 결정이 갱신되지 않는지 인과적으로 설명합니다. 결론은 필드 자체가 아니라, 필드 직후에 계산되어 aggregator/delimiter 토큰에 저장된 downstream notes를 통해 결정된다는 점을 보입니다. 이를 바탕으로 KV를 편집(editable)하고 미리 컴파일된 스킬을 재배치해 조합(composable)하는 두 가지 능력을 제안하며, 둘을 하나의 편집+조합 에이전트로 통합합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “필드 KV를 덮어쓰기”만으로는 모델이 실제로 무엇을 읽는지(결정 회로의 causal path)가 드러나지 않는다는 점입니다. 논문은 locality patching, suffix concentration, linear probing, circuit knockout 같은 인과 프로브로 결론이 downstream 특정 토큰에 집중 저장되고, 소수의 late-layer read head가 이를 다시 출력 로짓으로 주입한다는 회로 수준 메커니즘을 규명합니다. 이후 편집은 (1) erratum처럼 권위 있는 한 줄을 append하거나 (2) chain-of-thought 모드에서 field-only in_place refresh가 다시 읽히는 경우에만 성립하도록 설계해 실패 모드를 정리합니다. 조합은 RoPE-repositioning 기반의 position-portable transplant와 seam-repair(경계부 재계산에 해당하는 보정) 관점으로 구현해, splicing된 스킬이 full recompute와 구별되지 않게 만듭니다.

- **Empirical Impact**: 실험은 네 모델 계열(Qwen3, Llama-3.1, Gemma-2, Mistral)과 다양한 캐시 설정(스케일, quantization, MoE, multimodal KV)에서 동일 메커니즘과 edit/compose 동작을 검증합니다. gated tool-agent 실험에서 field+erratum는 hoist-to-end oracle과 거의 동일한 결정 복구를 보이며, field-only 편집은 chain-of-thought일 때만 결정이 회복되는 “CoT 게이팅” 특성이 재현됩니다. 실제 vLLM 온라인 prefix caching 벤치마크에선 prefix-cache 정렬(hit-rate 98.5%)을 유지하면서 p90 time-to-first-token을 53~398배 줄였고, 지연·처리량 이득은 부하가 커질수록 최대 14.5배까지 확장됩니다.



### Prefill/Decode-Aware Evaluation of LLM Inference on Emerging AI Accelerators (https://arxiv.org/abs/2606.17104)
Comments:
          8 pages, 5 figures. Accepted to the Workshop on HPC for AI Foundation Models & LLMs for Science (HPAI4S'26), co-located with IEEE IPDPS 2026

- **Prior Approaches**: 기존 LLM 추론 평가는 end-to-end 지표(전체 지연, 전체 처리량) 중심인 경우가 많아 Prefill/Decode 구간의 병목과 트레이드오프가 가려졌다. MLPerf Inference처럼 TTFT/TPOT을 쓰더라도 플랫폼별 보고가 주로 집계 형태라, 가속기가 어떤 위상에서 강한지 체계적으로 분리 비교하기 어렵다.
또한 Prefill/Decode disaggregation(분리 서빙)은 GPU 클러스터에서 주로 연구되었고, GroqRack 같은 emerging AI accelerator가 각 위상에서 GPU 대비 언제 이기는지에 대한 표준화된 phase-aware 측정이 부족했다.

- **Core Contribution**: 이 논문은 동일 모델 Llama2-7B를 GPU와 GroqRack에 걸쳐 Prefill과 Decode를 분리 측정하는 phase-aware 평가 프레임을 제안한다. Prefill(TTFT 중심)과 Decode(TPOT/TPOT·TPOT=토큰당 지연, 투입·동시성에 따른 처리량)의 성격 차이를 그대로 반영해, “가속기 우위”가 위상과 지표에 따라 달라짐을 정량화한다.
그 결과, GPU는 Prefill에서 일관되게 강하고 GroqRack은 Decode에서 단일요청 기준 per-token 지연을 크게 줄인다는 보편적 비대칭을 보여준다.

- **Technical Challenges**: 핵심 난제는 동일 Llama2-7B를 두 플랫폼이 서로 다른 실행 방식으로 처리한다는 점이다(예: GPU는 batched, GroqRack은 statically scheduled로 사실상 B=1 전제). 저자는 warmup과 토큰화/디토큰화 제거로 초기 오버헤드와 부가비용을 배제하고, workload 파라미터(Lin, Lout, batch)를 통제해 위상별 TTFT/TPOT 및 처리량을 분리 측정했다.
또한 heterogeneous disaggregation의 관건인 KV cache 전송 비용을 네트워크 대역폭 조건별로 모델링해, 성능 이득이 전송 오버헤드를 상쇄하는 break-even 구간을 함께 분석했다.

- **Empirical Impact**: 실험에서 GPU는 Prefill에서 GroqRack 대비 최대 수십 배 수준으로 더 낮은 TTFT와 높은 처리량(배치 증가 시 포화)을 보였지만, Decode에서는 배치가 커지며 GPU TPOT과 처리량이 불리해지는 구간이 나타났다. 반대로 GroqRack은 Decode TPOT이 거의 일정(안정적인 per-token 지연)하고 B=1에서는 처리량도 GPU보다 높지만, GPU는 배치가 충분히 커지면 Decode 처리량 우위를 되찾는다.
이 phase-dependent 결과는 단일 end-to-end 점수로는 최적 플랫폼 선택이 왜곡될 수 있음을 시사하며, GPU Prefill + accelerator Decode 같은 이기종 분리가 출력 길이가 충분할 때 end-to-end 지연을 낮출 수 있음을 정리된 조건(특정 입력-출력 비율에서만 불리)으로 제시한다.



### Quantum Cinema: An Interactive Cinematic Exploration of Quantum Computing Hardware via Generative World Models (https://arxiv.org/abs/2606.17102)
- **Prior Approaches**: 기존 학습 도구는 주로 회로 수준에서 게이트를 조작·시뮬레이션(예: Quirk, IBM Quantum Experience)하며 실제 양자 프로세서의 물리적 구조는 보이지 않는 한계가 있습니다. VR/AR 기반 교육은 몰입감을 주지만 헤드셋 등 장비 의존성과 설치·세팅 부담 때문에 확장성이 떨어진다는 지적이 많았습니다. 생성형 3D 세계를 과학 인프라 교육에 적용한 시도는 있었으나, 양자 하드웨어를 대상으로 한 통합 플랫폼은 부족했습니다.

- **Core Contribution**: 이 논문은 Quantum Cinema를 통해 ‘보이지 않는 양자 하드웨어’를 브라우저 기반 인터랙티브 3D/시네마 경험으로 바꾸는 방법을 제안합니다. 생성형 world model을 활용해 갇혀 있는 실험실 현실을 누구나 탐색 가능하게 만들고, Nobel Prize 서사부터 3개 아키텍처( trapped-ion, neutral-atom, superconducting ) 탐구, 마지막으로 정량 비교까지 4막 구조로 연결합니다. 특히 각 3D 세계는 AWS Braket 및 장비 스펙 기반 지표로 과학적 접점을 유지하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 생성된 시각화가 하드웨어의 실제 형상·구동 맥락을 충분히 반영해야 한다는 점과 (2) 생성형 결과를 정확한 물리 시뮬레이션이 아니라 ‘교육용 관찰 가능 모델’로 안전하게 포지셔닝하는 것입니다. 저자들은 World Labs Marble로 3D 장면을 생성하고, Chisel 편집기로 카메라·조명·재질 및 아키텍처 토폴로지를 수작업 정제해 과학적 일관성을 확보했습니다. 또한 qubit count, gate fidelity, connectivity topology, coherence time, error rate 같은 Braket 기반 지표를 레이더 차트로 제공해 정성 감상을 정량 비교로 전환합니다.

- **Empirical Impact**: 정량 비교 중심의 Act IV와 장비 스펙 기반 지표 앵커가 결합되면서, 학습자가 아키텍처 간 트레이드오프를 데이터로 이해하도록 돕는 교육적 설계가 강조됩니다. Quantum Cinema는 MIT License로 공개되며, 설치나 전문 장비 없이 브라우저에서 바로 실행 가능하도록 배포(Next.js/SPA, AWS 3-tier)해 현장 확산 장벽을 낮췄습니다. 학계·개발자에는 재현·확장 가능한 파이프라인을, 교육자·소통자에게는 교실·대중 강연용 직관 도구를 제공하는 점에서 양자 리터러시 및 workforce 개발에 기여할 잠재력이 큽니다.



### Software Delegation Contracts: Measuring Reviewability in AI Coding-Agent Work (https://arxiv.org/abs/2606.17099)
Comments:
          11 pages; empirical pilot study with 64 coding-agent runs and 192 blinded reviews

- **Prior Approaches**: AI 코딩 에이전트 연구는 이슈를 받아 코드 저장소를 수정하고 패치/PR을 반환하는 흐름을 전제로, 결과의 수리적 성공(테스트 통과 등)에 초점을 두는 경우가 많았습니다. 소프트웨어 delegation contract(업무 T, 권한 A, 반환 work package W, 수용 문맥 C)를 ‘관계의 분석 단위’로 제안한 선행 연구는 있었지만, 그 효과(검토 가능성/품질)를 실제로 측정하진 못했습니다.

- **Core Contribution**: 이 논문은 delegation contract를 명시적으로 제공했을 때, 반환물의 reviewability(검토 가능성)가 얼마나 달라지는지와 그 비용을 통제 실험으로 정량화합니다. 10개 seeded TypeScript API 작업을 대상으로, 현실적인 이슈 스타일 프롬프트(A)와 명시적 contract(B), 그리고 증거 번들 템플릿이 강제된 조건(C)을 비교해 64회 실행과 192회(블라인드) 리뷰를 수행했습니다. 핵심 결론은 작은 과업에서는 correctness(정답성)가 이미 포화돼 contract가 성공 여부를 바꾸지 못했지만, reviewability는 일관되게 개선됐다는 점입니다.

- **Technical Challenges**: 진짜 비교를 위해 저자는 작업 환경에서 의존성 없이 즉시 테스트 가능한 ‘toy’ 저장소를 만들고, hidden acceptance tests와 mutation checks, 범위(scope) 분석으로 기계적 품질을 확인했습니다. 동시에 리뷰어 편향을 줄이기 위해 조건 라벨을 숨긴 모델 기반 리뷰 파이프라인과 고정 루브릭(범위 준수, 증거 충분성, 모호성 등)을 사용했으며, 계약 텍스트 자체가 treatment라 완전 블라인드는 어렵다는 한계를 전제했습니다. 또 evidence bundle을 요구할 때 어떤 필드가 ‘수요 탄력적으로’ 채워지는지 구조적으로 관찰할 수 있게 실험 설계를 구성했습니다.

- **Empirical Impact**: 결과적으로 hidden acceptance 체크와 authority 경계 위반은 64/64 모두에서 0건으로, 계약은 correctness를 개선하지 못했습니다. 반면 evidence sufficiency는 30쌍 중 22쌍에서 상승했고 모호성은 유의하게 감소했으며, residual risks·reviewer checklist 같은 항목은 자발적으로는 거의 나오지 않다가 contract/evidence 요구 시 거의 100%로 나타났습니다. 비용은 토큰 +13%, 벽시계 시간 +38%로 증가했지만, 계약이 제공하는 가치는 ‘정답 향상’이 아니라 ‘검토에 필요한 증거와 보고 형식의 표준화’임을 보여줍니다.



### ANEForge: Python for direct computation on the Apple Neural Engin (https://arxiv.org/abs/2606.17090)
Comments:
          8 pages

- **Prior Approaches**: 그동안 Apple Neural Engine(ANE)에 접근하는 가장 ‘허가된’ 경로는 CoreML로, ANE를 실행 시 스케줄링 옵션처럼 취급해 실제 호출이 ANE로 갔는지 보장·진단이 어렵다. 또한 CoreML/공개 툴은 ANE의 네이티브 연산을 충분히 노출하지 못해, 성능·전력 최적화를 위한 하드웨어 고유 기능(예: fused attention, native fused 연산 계열)을 직접 조합하기가 제한된다. 반대로 공개되지 않은 ANEClient/권한(entitlement)·시스템 수정이 필요한 우회 접근은 복잡하고 배포성도 낮다.

- **Core Contribution**: ANEForge는 CoreML 의존 없이 Python에서 ANE 프로그래밍(컴파일·디스패치)을 가능하게 하는 프론트엔드 패키지다. 지연(lazy) 텐서 연산 그래프를 1개의 ANE 프로그램으로 내리고, ANE 디먼·커널 드라이버까지 이어지는 동일한 내부 런타임 경로로 고정 디스패치한다. 결과적으로 호출 시 ANE 실행 여부가 런타임 휴리스틱에 좌우되지 않고, 네이티브 fused attention, 온디바이스 weight streaming, resident state, on-engine 학습까지 확장된다.

- **Technical Challenges**: 핵심 난제는 ‘공개 API 없이’ ANE를 실제 생산 경로로 디스패치하는 방법과, 컴파일러가 수용하는 연산 표면(operator surface)을 정확히 구성하는 것이다. ANEForge는 58개의 fused 연산 경로와 19개의 native bridge 연산을 묶어 그래프를 ANE 컴파일 MIL로 낮추고, 머신 체크된 capability registry로 호환 연산을 검증하며 컴파일 전부터 실패를 줄인다. 또한 성능을 위해 구조 기반 비용 모델로 동등 lowerings를 선택·캐싱하고, 역전파와 Adam 업데이트까지 same compile-and-dispatch 경로로 엔진에서 실행하도록 autograd·학습 모듈을 구성했다.

- **Empirical Impact**: 구현 오버헤드는 작은 fused 프로그램 호출이 약 90us로 ANE 디스패치 바닥(약 70us)에 근접하며, ResNet-18 순전파는 end-to-end 약 0.33ms로 GPU/CPU 대비 유리한 수치를 보였다. 사전학습된 ResNet-18/문장 인코더/ Vision Transformer, 그리고 Stable Diffusion U-Net의 forward를 프레임워크 레퍼런스와 end-to-end로 검증했고, attention native 레이어 접근 및 디코딩을 위한 키-밸류 캐시 resident 운용도 제시됐다. 단, private·undocumented 심볼 의존과 반정밀(half precision) 수치 범위 한계로 릴리스별 macOS/ANE-compiler 버전 및 operator corpus로 회귀를 관리하도록 설계됐다.



### ZIVARI-TLBO: A Zero-Cost Inter-Group Evaluated-Elite Relay Mechanism for Teaching-Learning-Based Optimization (https://arxiv.org/abs/2606.17087)
Comments:
          21 pages, 7 figures, 11 tables

- **Prior Approaches**: TLBO는 teacher–learner 단계로 구성된 군집 기반 최적화로, grouped·multi-population 확장이 “그룹 간 정보 이동” 방식에 따라 성능이 크게 좌우된다는 점이 반복적으로 지적돼 왔습니다. 기존 연구는 migration 토폴로지나 엘리트 공유를 다루지만, 전이 시점에 새 후보를 만들거나(추가 objective-function 호출) 정보 전이가 약·과해져 군집 간 다양성/수렴성 균형이 흔들릴 수 있습니다. 또한 실험 비교는 objective-function 호출 예산을 공정하게 세는가가 핵심 한계로 남아 있었습니다.

- **Core Contribution**: ZIVARI-TLBO는 grouped TLBO 프레임에 “고정 ring evaluated-elite relay”를 추가해, 한 그룹의 이미 평가된 엘리트 해와 그 목적함수 값을 다음 그룹에 정확히 복사해 전달합니다. 핵심은 relay가 새 후보를 생성하지 않아 objective-function 호출을 추가로 소비하지 않는다는 점이며, 모든 TLBO 생성 후보는 동일한 공정 예산 안에서만 카운트됩니다. 즉 “예산 제약을 지키는 정보 공유 메커니즘”을 명확히 정의하고, 그 비용 중립성을 구현 수준에서 감사(audit)로 보여줍니다.

- **Technical Challenges**: 기여의 실현에서 가장 큰 기술 과제는, 그룹 간 전이가 성능을 개선하면서도 objective-function 호출 예산을 baseline과 동일하게 유지하는 것이었습니다. 이를 위해 전이를 ‘복사(copy)·저장된 fitness 동반(elite objective value transfer)’ 규칙으로 제한하고, receiver의 worst eligible learner를 대체할 때도 stored objective value만 비교해 불필요한 재평가를 차단했습니다. 또한 글로벌 최선값은 덮어쓰지 않도록 보호하고, ring 스케줄에서 보낸 엘리트를 같은 이벤트 내 연쇄 전달하더라도 “0-cost relay” 성질이 유지되게 설계했습니다.

- **Empirical Impact**: 클래식 8개 함수에서 10,000-evaluation 동일 예산, 30 matched seeds 조건으로 relay-disabled 대비 728/11/221 wins/ties/losses 및 rank-biserial effect size 0.624를 보고하며 relay의 범위 내 기여를 강하게 지지합니다. 8개 방법( TLBO, MCTLBO, DE, PSO, GWO, WOA, HHO 포함) 다차원 비교에서는 WOA가 평균 랭크 2.914로 1위, ZIVARI-TLBO는 3.382로 2위이며 TLBO/MCTLBO/DE/PSO/GWO에는 유의하게 우수하지만 WOA에는 유의하게 뒤지고 HHO와는 Holm 보정 후 유의차가 사라집니다. 공학적 제약 문제에서는 성능이 혼재되며 static-penalty 제형에 민감하고 strict-feasibility가 낮은 문제가 있어 “엔지니어링 우월성/보편적 SOTA” 주장에는 제약이 남습니다.



### ParkingTransformer: LLM-Enhanced End-to-End Trajectory Planning for Autonomous Parking (https://arxiv.org/abs/2606.17082)
- **Prior Approaches**: 기존 end-to-end 자율주행/주차 연구는 모듈형 기준선보다 오류 전파는 줄이지만, 여전히 dense BEV를 거치거나(계산비용 증가) 장면을 충분히 이해하지 못하거나(정밀 기하 추론 부족) 의사결정이 불투명하다는 문제가 남아 있습니다. LLM을 붙인 시도는 설명 가능성을 높이지만, hidden states를 그대로 궤적 계획에 쓰면 주차에 필요한 centimeter급 정밀도와 공간 추론이 부족하다고 지적됩니다. 또한 long-distance(50~300m) 주차에서는 과거 이력 모델링과 정교한 coarse-to-fine 보정이 결여되어 성능이 급격히 붕괴하는 경향이 있습니다.

- **Core Contribution**: 이 논문은 ParkingTransformer라는 새로운 end-to-end 주차 프레임워크를 제안하며, multi-view 시각지각과 LLM 기반 장면 이해를 한 Transformer 아키텍처에 결합합니다. trajectory queries를 LLM의 implicit state features와 결합해 dense BEV 중간표현 없이도 직접 궤적을 출력하도록 설계했습니다. 여기에 LLM의 공간추론 약점을 보완하기 위한 3D positional encoding, 장기 시간정보 처리를 위한 fixed-window streaming, 디코더의 coarse-to-fine 점진 정밀화 전략을 추가합니다.

- **Technical Challenges**: 가장 큰 기술 난관은 (1) dense BEV의 계산비용을 줄이면서도 (2) 주차에 필요한 장면 의미/공간 기하를 충분히 주입하고 (3) LLM의 제한된 3D 추론을 궤적 정밀도로 연결하는 것입니다. 논문은 SCA(Sensor Cross-Attention)에서 3D positional encoding과 depth 후보 샘플링(LID)을 사용해 기하 정보를 명시적으로 주입하고, TCA(Temporal Cross-Attention)로 과거 이력과 현재 정보를 결합하되 fixed-window streaming으로 계산을 통제합니다. 마지막으로 coarse-to-fine 디코더로 전역 경향→국소 보정 순서의 잔차(refinement)를 누적해 정확도를 끌어올립니다.

- **Empirical Impact**: CARLA closed-loop 실험에서 driving score 61.32를 달성하고, 실차 실험에서는 평균 success rate 88.70%를 보고해 long-distance 주차 가능성을 실증했습니다. 짧은 거리(약 20m)에서는 기저모델 대비 비슷하거나 약간의 개선 수준이지만, 거리 증가(50~300m)에서는 LLM/이력/정교 보정이 없는 baseline들이 거의 붕괴한 반면 ParkingTransformer는 로드→진입→근접→리버스 주차까지 완주합니다. ablation 결과로 TCA/SCA, LLM 모듈, 3D positional encoding이 성능에 유의미하며, 주차 속도(통상 15 km/h 이하) 조건에서는 inference speed도 실용 범위 내라고 제시합니다.



### The Price of Anarchy in Disaggregated Inferenc (https://arxiv.org/abs/2606.17081)
Comments:
          38 pages, 7 figures, 8 tables. Measurements on a 3-node NVIDIA B200 cluster running NVIDIA Dynamo v0.9.0

- **Prior Approaches**: 기존 LLM 서빙 평가는 주로 Pareto frontier(throughput-지연 트레이드오프)로 성능을 요약했지만, 그 지점들이 실제로 어떤 “자기이익적” 의사결정의 평형에서 나오는지는 명확히 이름 붙이지 못했습니다. 게임이론은 GPU 스케줄링이나 공정성에 적용돼 왔지만, prefill/decode 분리와 KV 캐시 배치, 요청 라우팅이 결합된 추론-요청 단위의 구조를 게임으로 공식화한 연구는 거의 없었습니다.

- **Core Contribution**: 이 논문은 disaggregated inference를 NVIDIA Dynamo를 사례로 “세 개의 결합 게임”으로 분해해 정식화하고, GPU 포화(saturation) 상태에서 평형의 효율 구조가 어떻게 바뀌는지 게임-이론적으로 분석합니다. 특히 P/D resource game(Planner의 자원 분배), selfish caching game(계층형 KV cache placement), congestion game(positive externalities 포함 request routing)을 연결해, 설정 변경이 Pareto 프런티어의 어떤 부분을 안정화/붕괴시키는지 설명합니다.

- **Technical Challenges**: 핵심 난점은 라우팅/캐시 결정이 일반 Nash 평형을 계산할 수준으로 복잡해진다는 점인데, 추론은 밀리초 단위라 평형 계산 자체가 불가능합니다. 논문은 대신 라우터의 “각 요청의 개별 비용 최소화”가 congestion game의 best-response 역학과 일치한다는 메커니즘 설계 관점으로 PoA를 직접 해석하고, 포화 직후의 payoffs 전환을 실시간 감지하는 adaptive controller로 라우팅 파라미터를 캐시 친화(affinity)에서 혼잡 회피(load-balanced congestion avoidance)로 스위칭합니다.

- **Empirical Impact**: 3-node NVIDIA B200 클러스터에서 Dynamo를 돌린 실험은 KV 캐시 배치와 요청 라우팅 게임에서의 PoA-hat 구조를 실증 검증했고, 두 모델(Nemotron-4-340B, Llama-3.1-70B)·세 가지 토폴로지에서 “세 구간”과 knee 이후의 성장 패턴이 재현됨을 보여줍니다. 가장 큰 성과는 Llama-3.1-70B 1P/5D에서 saturated phase PoA-hat이 3.1x(66.4→21.5) 감소했지만 처리량은 13% 비용을 치렀고, 1P/2D에서는 TTFT P99이 7.6x 감소하는 등 SLO 관점의 안정적 운영점 이동 효과가 확인됐습니다.



### HRDX: A Large-Scale Vector HD-Map Datas (https://arxiv.org/abs/2606.17080)
Comments:
this https URL

- **Prior Approaches**: 기존 벡터 HD map 구축 연구는 polyline/segmentation 계열 표현을 end-to-end로 예측하는 방향(예: MapTR, MapTracker 등)으로 발전했지만, 대다수 성능 병목이 데이터 규모와 라벨 풍부함의 한계를 크게 받습니다. 공개 데이터셋은 주행 커버리지가 짧고(대개 10시간 미만), geometry 중심에 비해 lane color·style, 도로 텍스트 같은 규제/행동 시맨틱 속성은 희소하거나 누락되는 경우가 많습니다. 또한 aerial imagery처럼 차량 주행 궤적과 정밀 정합된 모달리티가 부족해 cross-view 융합이나 privileged information 학습을 체계적으로 다루기 어렵다는 지적이 있었습니다.

- **Core Contribution**: 이 논문은 온라인 벡터 HD map 구축을 위한 대규모 데이터셋 HRDX를 제안합니다. HRDX는 약 40시간(1,400km) 규모의 최소 중복 주행을 모으고, 6대 동기화 surround camera, 128-beam LiDAR, RTK GNSS/IMU에 더해 차량 궤적과 정밀 정합된 aerial orthoimagery(최대 8cm/pixel)를 제공합니다. 라벨은 10개 벡터 map 클래스와 20+개의 semantic·topological 속성으로 확장되어, 기존 공개 데이터셋에 없던 규제/행동 시맨틱까지 포함합니다.

- **Technical Challenges**: 주요 과제는 (1) geometry뿐 아니라 속성 정확도까지 함께 평가/학습 가능한 학습 체계를 만드는 것과 (2) aerial을 정합·재현 가능하게 제공하되 추론 센서 비용을 늘리지 않는 활용법을 찾는 것입니다. 논문은 geometry-mAP(Chamfer-distance 기반)만 보던 한계를 보완하기 위해 Composite Score(CS)로 위치 정밀도와 attribute correctness를 동시에 측정하도록 설계합니다. 또한 aerial 기반 BEV 컨텍스트를 cross-attention 융합에 활용하되, 추론에서는 카메라만 쓰도록 teacher–student knowledge distillation(C+A/C) 프레임을 적용해 aerial의 이점을 카메라-only 학생에게 전이합니다.

- **Empirical Impact**: 실험은 HRDX의 데이터 규모 확장이 mAP과 CS를 단조적으로 끌어올린다는 결과를 보여주며, 대규모·다양한 주행 커버리지가 학습 안정성과 일반화에 핵심임을 시사합니다. 더불어 aerial imagery를 학습과 추론에 모두 사용하면 카메라-only 기준 대비 mAP과 CS가 각각 큰 폭으로 개선되며, stop lines·crosswalks·도로 경계·도로 텍스트처럼 전역 레이아웃과 가림(occlusion) 완화의 혜택이 큰 요소에서 상승이 집중됩니다. 마지막으로 aerial만 학습에 쓰고 추론은 카메라-only로 유지해도 성능 격차를 상당 부분 줄일 수 있어, deployment-feasible한 privileged information 활용 전략으로 의미가 큽니다.



### Comprehensive pKa Data Augmentation from Limited Real Data through an Engineered Models-Quantum Framework (https://arxiv.org/abs/2606.17077)
- **Prior Approaches**: 기존 pKa 연구는 iBonD 같은 실험 데이터베이스를 기반으로, (1) DFT/ab initio로 정확도를 확보하거나 (2) XGBoost 같은 ML 회귀로 빠르게 예측하는 두 축이 주류였다. 데이터 부족을 완화하려고는 unlabeled 분자에 회귀 모델을 적용한 pseudo-label 기반 데이터 증강을 시도했지만, 이는 분포 중심부에 집중되어 강산/약산 등 tail 구간 커버리지가 부족해졌다.

- **Core Contribution**: 이 논문은 “tail 영역(극단 pKa)에서의 분자 생성” 문제를 회귀 증강의 한계를 넘어, 양자(또는 양자-영감) 최적화가 가능한 조합 최적화로 재정의한다. 이를 위해 128비트 이산(=binary) 잠재공간으로 분자를 인코딩하는 Transformer–VAE를 만들고, 그 이산 코드에서 pKa를 예측하는 factorization machine(FM) surrogate를 QUBO로 변환해 extreme pKa용 샘플링을 수행한다.

- **Technical Challenges**: 핵심 난관은 연속 latent 생성모델(VAE 등)이 pKa 조건을 넣어도 tail의 희소한 extreme regime를 안정적으로 탐색하지 못한다는 점이었다. 저자들은 discrete binary bottleneck(Gumbel–Softmax)로 생성의 자유도를 이산화하고, FM에 pKa-bucket별 가중치·long-tail 전용 regularization·상호작용 튜닝을 적용해 tail 피팅을 강화했으며, 이후 simulated quantum annealing과 물리 coherent Ising machine(CIM)에서 sampling을 수행해 조합 탐색 효율을 높였다.

- **Empirical Impact**: 실험/검증 결과, QUBO 에너지- FM 예측 pKa의 Pearson 상관이 tail의 저pKa/고pKa 구간에서 0.9 이상으로 유지되었고, CIM이 simulated annealing보다 extreme-value sampling에서 우수했다. 특히 8-bit/14-bit 정밀도에서 탐색–활용 균형이 달라져, 14-bit는 랭킹 보존(Spearman 향상)에, 8-bit는 더 큰 다양성과 tail 커버리지에 유리함을 보여주며 pKa 스펙트럼 전방위 확장 가능성을 실증했다.



### CMIP-Forge: An Agentic System that Retrieves, Computes, and Self-Reviews Climate Scienc (https://arxiv.org/abs/2606.17076)
Comments:
          28 pages, 9 figures. Code available at this https URL

- **Prior Approaches**: 기존 LLM 기반 기후분석은 주로 point cloud, 재분석 그리드, 혹은 구조화된 데이터 저장소 같은 “정형/반정형” 자원에 강점을 보여 왔지만, CMIP6처럼 방대한 비정형 문헌 지식과 live 데이터 분석을 동시에 잇는 방식은 제한적이었습니다. 또한 단일 에이전트가 생성한 Python 코드가 문법적으로는 통과하더라도 물리적으로는 틀릴 수 있어, 미묘한 오류가 조용히 전파되는 문제가 반복됐습니다.

- **Core Contribution**: CMIP-Forge는 CMIP6 오픈액세스 논문 6,581편(101,828개 청크)을 RAG로 연결하고, ESGF 아카이브에서 필요한 데이터까지 내려받아 Python 분석을 end-to-end로 수행하는 하이브리드 RAG+자율 분석 시스템을 제안합니다. 여기에 “독립 reviewer 모델” 패널로 워크플로 전 과정을 감사(audit)하는 자율적 peer-review 루프를 결합해, 문헌 기반 방법론과 계산 결과의 정합성을 함께 강제합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 문헌에서 메소드·제약·함정을 정확히 끌어오면서도, (2) 코드 실행 단계에서 물리적/통계적 불변조건을 깨지 않게 하는 것, (3) reviewer의 그럴듯한 반박이 텔레메트리와 충돌할 때 이를 바로잡는 것이었습니다. CMIP-Forge는 AST 정적 분석, 감사된 scientific primitives, 실행 텔레메트리(차원·요약통계 강제 출력), Empirical Defiance Protocol 같은 Defense-in-Depth를 통해 “언어모델 추론”이 아니라 “실행 가능한 가드레일”이 정확성을 책임지도록 설계했습니다.

- **Empirical Impact**: 논문 작성까지 추적 가능한 7개 end-to-end 자율 연구 파이프라인(대기 텔레커넥션, 해양 동역학, 지역 극한, 전지구 온난화 전망 등)에서 CMIP-Forge는 literature-grounded 분석과 라이브 데이터 기반 수치 산출을 함께 완수했습니다. 특히 AMOC(대서양 자오면 순환) 관련 kinematic fingerprint를 ERA5로 검증하고 CMIP6 모델을 평가·선별한 뒤 SSP5-8.5에서 임계치(예: -0.5°C 수준) 통과 시점을 멀티모델 평균과 개별 모델 관점에서 재현하는 등, 리뷰 루프의 구체적 실패모드(동조적 회귀, 결론 미해결(REVISE), 스텁 코드 제출)도 불변 텔레메트리로 진단 가능함을 보여줬습니다.



### Surveying GenAI-based Automation in Printed Circuit Board Design and Tes (https://arxiv.org/abs/2606.17074)
Comments:
          33 pages, 5 figures, 11 tables. Under review

- **Prior Approaches**: PCBs는 공급망 전반에 걸친 분산·반복 검증 구조라서, 요구사항 정의→회로 설계→레이아웃 최적화→검증/테스트→조립/유통까지 오류가 연쇄로 비용을 키우는 문제가 크다. 기존 EDA는 도식/레이아웃 자동화와 설계 규칙 점검을 지원하지만, 여전히 사람의 판단과 테스트 계획 설계가 병목이며 보안 검증까지 자동화하기는 한계가 있었다. GenAI는 IC 중심으로는 적용 사례가 늘고 있으나, PCB 라이프사이클 전 구간을 망라해 비판적으로 정리한 학술 조사는 부족했다.

- **Core Contribution**: 이 논문은 PCB 설계와 테스트 전 과정에 GenAI가 실제로 어떻게 쓰이고 있는지 상태를 체계화하기 위해 80편(수집 227편 중 적용 포함)의 연구를 분류·정리한다. 특히 설계(생성/수정), 최적화, 검증(기능·보안)이라는 관점으로 연구 의도와 기여를 연결하는 택소노미를 제시한다. 이를 통해 PCB 맥락에서 남아 있는 연구 공백과 향후 통합 기회(효율·정확도·스케일)를 구체적으로 드러낸다.

- **Technical Challenges**: 핵심 장애물로는 PCB 도메인 특화 데이터 희소성, 그리고 기존 PCB 도구(EDA)와의 통합(integration) 지원이 제한적이라는 점이 제시된다. 또한 보안 취약점은 위협 모델·자산 식별·하드웨어 보안 지식이 필요해, 자동화 도구가 전체 문맥을 충분히 갖추기 어렵다는 문제도 크다. 논문은 이러한 제약을 고려해, 특정 단계에서의 결함 탐지/테스트 자동화처럼 부분 해결이 축적되고 있으나 end-to-end 관점 통합은 아직 초기 단계라고 정리한다.

- **Empirical Impact**: 경향 분석에 따르면 2021~2025 기간 동안 GenAI와 PCB의 교차 연구가 전반적으로 증가하고 있으며, 단계별로 설계·최적화·검증 중 특히 기능 검증 및 생성/최적화 쪽에 연구가 더 집중되는 양상이 관찰된다. 다만 보안 검증의 자동화나 표준화된 파이프라인 구축은 상대적으로 덜 다뤄져, 향후 영향력이 큰 공백으로 남는다. 결과적으로 이 서베이는 후속 연구가 어떤 작업 단계와 어떤 모델 계열(LLM·VLM·GAN·diffusion)을 어떤 목표(기능/보안)에 연결해야 하는지에 대한 실무적 지도 역할을 제공한다.



### Extracting Semantics: LLM-Guided Automatic Population of Robot Ontology from URDF (https://arxiv.org/abs/2606.17073)
- **Prior Approaches**: 인지 로보틱스에서 온톨로지는 환경 지식과 로봇의 신체(embodiment)·기능 정보를 통합해 설명 가능한 추론을 가능하게 하지만, 수동 구축이 큰 병목이다. 또한 URDF는 로봇의 구조와 기구학을 제공하더라도, 의미 있는 식별자·명칭은 commonsense 해석이 필요해 그대로는 의미론적 풍부함을 얻기 어렵다. 기존 자동화는 주로 구조 파싱에 머물러 온톨로지 정합성(스키마/제약)까지 안정적으로 맞추는 데 한계가 있다.

- **Core Contribution**: 이 논문은 URDF를 입력으로 받아, LLM을 활용해 의미론적 추상(semantic abstractions)을 채운 온톨로지로 자동 생성하는 파이프라인을 제안한다. 핵심은 온톨로지의 개념 프롬프트를 바탕으로 LLM이 의미 관계를 추론하되, 최종 분류는 형식 모델과 온톨로지 제약을 따르도록 정렬(alignment)하는 점이다. 즉, 로우레벨 URDF에서 휴먼-로보트 인터랙션에 필요한 grounded·구조화 지식 표현으로의 브리지를 만든다.

- **Technical Challenges**: 가장 큰 기술적 난제는 URDF 식별자가 요구하는 commonsense 의미를 LLM이 잘 추론하더라도, 생성 결과가 온톨로지 스키마와 문법 규칙을 위반할 수 있다는 신뢰성 문제다. 이를 위해 파이프라인은 여러 LLM 질의에 대한 majority voting으로 변동성을 줄이고, 구문(syntactic)·스키마 레벨 validation을 추가해 출력이 기대 표현 형식과 온톨로지 제약을 만족하는지 검증한다. 또한 “형식 모델에 맞춘 분류”를 목표로 프롬프트를 온톨로지 개념에 고정해 추론의 방향성을 통제한다.

- **Empirical Impact**: 여러 로봇 설명(URDF)에서 생성된 추상 온톨로지를 평가하며, 로우레벨 모델에서 구조화된 의미론적 표현으로의 변환이 실제로 가능함을 초기 결과로 보인다. 저자들은 생성된 추상들이 온톨로지 기반 추론에 필요한 grounded 지식을 제공할 수 있음을 논의한다. 인력 의존이 큰 온톨로지 구축 비용을 낮출 수 있어, 사람과 상호작용하는 embodied agent의 지식 업데이트/설명 가능성 측면에서 의미 있는 출발점이 될 것으로 보인다.



### KFTD: Koopman-Fourier Time-Differentiable Network for Continuous Ocean Spatiotemporal Forecasting (https://arxiv.org/abs/2606.17070)
- **Prior Approaches**: 기존 해양 시공간 예측은 관측에서 상관관계를 학습해 빠른 추론을 노리지만, 관측이 희소하거나 잡음이 있으면 일반화가 급격히 흔들리는 문제가 컸습니다. diffusion 계열은 예측을 여러 denoising step으로 쪼개 안정성을 얻는 대신, 고정된 시간 격자와 다단계 잡음 샘플링 때문에 계산량이 커지고 고주파 천이의 aliasing도 생길 수 있습니다. 또한 완전 데이터 기반 접근은 PDE 제약을 충분히 만족하지 못해 질량·운동량·에너지 같은 보존 법칙 위반으로 물리적으로 그럴듯하지 않은 예측이 나타나는 ‘physical consistency bottleneck’이 존재했습니다.

- **Core Contribution**: 이 논문은 Koopman Fourier Time-Differentiable(KFTD) Network로, 시간 연속형 two-stage 패러다임을 제안합니다. KFTD는 비선형 해양 동역학을 Koopman 선형 공간으로 투영한 뒤 Fourier 연산을 활용해 임의의 sub-step에서 연속 시간 보간을 제공하고, 최종 예측은 고정밀 중간 상태를 입력으로 받는 가벼운 residual 네트워크로 수행합니다. diffusion처럼 다단계 noise sampling 없이 연속 시간으로 시스템을 직접 evolve 하도록 설계해 속도-정확도 트레이드오프를 완화합니다.

- **Technical Challenges**: 핵심 난제는 (1) 강한 비선형·다중 스케일(미세 난류부터 행성파) 동역학을 전역 선형화에만 의존하지 않으면서, (2) 임의 시간 해상도의 연속 보간과 (3) PDE 수준의 물리 제약을 end-to-end로 만족시키는 학습을 동시에 달성하는 것입니다. KFTD는 Koopman 연산을 멀티스케일 분해로 구성해 전역 선형 가정의 실패로 인한 오차 누적을 줄이고, FAP(Fourier analysis perceptron) 기반 Fourier 계수 학습을 네트워크 깊이와 연동시켜 멀티스케일 주기 성분을 더 잘 추출합니다. 여기에 Data-Physics Prior(D-PP) loss를 모듈형 residual 항으로 추가해, 임의 PDE residual을 페널티 가중치 튜닝 없이 플러그앤플레이 방식으로 강제하면서 학습의 물리 정합성 병목을 공략합니다.

- **Empirical Impact**: NOAA 등 4개 해양 데이터셋에서 실험한 결과, KFTD의 연속 시간 프레임워크는 MSE를 평균 5.6% 개선했으며 SST에서는 최대 12.7%까지 향상되었습니다. 효율 측면에서는 MCVD 대비 76.25% 개선을 보고하며, diffusion 대비 4× 수준의 연산 속도 향상도 함께 제시합니다. 전반적으로 ‘연속 시간 예측 + 물리 일관성 + 낮은 추론 비용’이라는 운용성 관점의 요구를 동시에 만족하는 방향을 보여, 해양 예보·재난 조기경보용 ML 예측 모델의 설계 기준을 한 단계 끌어올렸다는 점에서 의미가 큽니다.



### PIVOT: Bridging Black-Scholes Implied-Volatility and Price Objectives via Differentiable Jäckel Operator (https://arxiv.org/abs/2606.17065)
Comments:
          30 pages, 17 figures, 12 tables

- **Prior Approaches**: 옵션 마켓 학습은 보통 가격 공간에서 no-arbitrage 제약을 다루고, implied volatility(IV) 공간에서 표면을 스무딩·정규화·평가한다는 이중 좌표를 쓴다. 하지만 학습 단계에서 수백만 가격 P⋆를 점별 IV 라벨 σ_imp로 변환하고, 다시 가격/IV 좌표를 오가는 목적함수에서 기울기를 요구할 때 “수치 역변환 레이어”와 “ML 레이어”의 인터페이스가 병목이 된다.

- **Core Contribution**: 이 논문은 Jäckel의 “Let’s Be Rational”(LBR) IV solver를 대체하지 않고, 이를 autograd-native 미분 가능 프리미티브로 만드는 PIVOT(Price-Implied-Volatility Objective Translator)를 제안한다. forward 패스는 LBR의 정확한 점별 역변환을 그대로 GPU 배치로 수행하고, backward 패스는 Black-Scholes/Black-76의 매끄러운 가격 맵에 대해 implicit differentiation으로 정확한 민감도(기울기)를 공급한다.

- **Technical Challenges**: 핵심 난관은 inverse map이 low-vega 구간에서 특이해져 1/vega(=1/𝒱) 그래디언트가 발산한다는 점이다. PIVOT은 이 문제를 “그냥 정규화로 숨기기” 대신 게이팅 계약으로 드러내며, invalid 도메인은 NaN으로, well-conditioned row은 정확한 1/vega 그래디언트를 주고, low-vega row는 감쇠(attenuation)해 잘못된 NaN/폭주 그래디언트가 학습기에 유입되지 않게 한다.

- **Empirical Impact**: 단일 H100에서 fused Triton 커널 기준으로 1.79e9 IV/s 수준의 속도를 달성하고, 기준 C solver 대비 최대 상대오차 9.3e-14로 기계정밀에 가깝게 유지된다. HyperIV 스타일 SPX 재현 실험에서는 PIVOT-augmented 목적함수가 Pareto 우위를 보이며 held-out price MAE를 최대 43.4% 줄이고, IV MAE는 크게 훼손되지 않은 채 동시 개선을 보였으며, ungated IV-roundtrip 대조군은 거의 0에 가까운 퇴행 표면으로 붕괴해 게이팅이 단순 튜닝이 아니라 correctness contract임을 확인했다.



### Towards Distributed Inference of LLMs on a P2P Network (https://arxiv.org/abs/2606.17059)
- **Prior Approaches**: 기존 prefix caching은 KV cache를 공유해 prefill 비용을 줄이려 했지만, 클러스터에서는 워커별로 KV cache가 파티션되어 라우팅이 성패를 좌우한다. 중앙집중형 cache-aware router는 전역에 가까운 캐시 상태를 모아 보내 효과가 있지만, 조정 지점이 병목·장애·추가 지연을 만든다. 다른 접근인 remote KV-cache 공유는 KV가 수~수백 MB 이상이 될 수 있어 네트워크 대역폭 부담이 커 edge/peer-to-peer 환경에 불리하다.

- **Core Contribution**: 이 논문은 peer-to-peer LLM serving에서 분산 라우팅만으로 prefix caching 효과를 끌어내는 “decentralized, prefix-cache-aware routing”을 제안한다. 각 노드는 자신의 KV cache prefix를 radix tree로 정확히 보유하고, 나머지 노드에 대해서는 anti-entropy로 비동기 갱신된 라이트메타데이터(추정 radix tree)를 유지한다. 라우팅은 central coordinator 없이, “가장 긴 prefix match”가 예상되는 노드로 요청을 포워딩하며 KV-cache 전송은 하지 않는다.

- **Technical Challenges**: 핵심 난제는 분산 환경에서 메타데이터가 오래되거나 불완전해질 때(weak consistency) 라우팅이 성능만 망치고 정확성은 유지되도록 설계하는 것이다. 이들은 stale 메타데이터가 발생해도 잘못된 출력이 아니라 cache miss로 이어져 결국 해당 노드가 prefill을 재계산하므로, strong한 합의 없이 eventually consistent metadata replication으로 충분함을 보인다. 또한 prefix affinity만 쓰면 특정 노드로 트래픽이 몰리는 hotspot이 생길 수 있어, anti-entropy 브로드캐스트에 큐 혼잡(back-pressure)을 태워 라우팅을 일시적으로 조정하는 push-back을 도입한다.

- **Empirical Impact**: 시뮬레이션 4노드 MMLU 워크로드에서, 통신 지연이 낮고 prefix 분포가 한쪽으로 치우친(Zipfian-like skew) 설정에서 라우팅이 latency를 개선했다. 반대로 네트워크 왕복 지연이 큰 조건에서는 포워딩 비용이 prefill 절감 이득을 상쇄해 효과가 제한됐다. 또한 라우팅이 특정 토픽의 prefix를 먼저 축적한 “specialist” 노드에 트래픽을 집중시키는 specialization–eviction cycle을 만들며, 이 집중이 라우팅 로직에 의해 균등 분포 대비 크게 증폭되는 점을 정량적으로 보여준다.



### Correct When Paired, Wrong When Split: Decoupling and Editing Modality-Specific Neurons in MLLMs (https://arxiv.org/abs/2606.17057)
Comments:
          18 pages, 11 figures

- **Prior Approaches**: 기존 Knowledge Editing은 LLM 중심 편집 패러다임을 MLLM에 그대로 옮겨와, 멀티모달(텍스트-이미지) 쿼리에서 정답이 맞는지를 주로 최적화해왔다. 하지만 멀티모달 입력에서 성공한 편집이 텍스트-only 같은 단일 모달 트리거로도 일관되게 전파된다는 보장은 부족했다. 최근 일부가 멀티모달 활성 패턴을 분석·편집하려 했지만, 모달 간 지식 공유/전달이 자연스럽다고 암묵적으로 가정하는 경우가 많다.

- **Core Contribution**: 이 논문은 MLLM에서 발생하는 editing decoupling failure를 정의하고, 멀티모달 쿼리에서는 편집이 맞더라도 단일 모달로 분리하면 구(舊) 지식으로 되돌아갈 수 있음을 체계적으로 보여준다. 원인은 ‘엔터티 지식이 단일 표현으로 저장되지 않고, 모달별로 분리된 뉴런 경로에 흩어진다’는 관찰에 있다. 이를 해결하기 위해 DECODE는 모달별 critical-neuron 그룹을 명시적으로 분리·국소화하고, 두 스트림(two-stream)으로 동기화된 편집을 수행한다.

- **Technical Challenges**: 핵심 기술적 난제는 편집해야 할 엔터티 관련 뉴런이 모달 트리거에 따라 서로 다른 회로에 매핑된다는 점이며, 그래서 멀티모달에서 조정한 업데이트가 단일 모달 회로로 전파되지 않는다는 것이다. 저자들은 모달별 입력 변형에서 각 뉴런의 contribution score를 계산해 텍스트-의존/비전-의존/멀티모달-의존 뉴런 집합을 분리한 뒤, FFN의 해당 뉴런 행에 learnable offset을 주입하는 방식으로 국소 편집을 구현한다. 또한 편집 효능과 collateral damage(다른 지식 손상)를 균형 있게 다루기 위해 타깃 손실과 locality를 위한 KL-divergence 기반 제약을 함께 최적화한다.

- **Empirical Impact**: 여러 MLLM(InstructBLIP, LLaVA, Qwen-VL)에 대해 DECODE는 멀티모달 입력에서는 물론 텍스트-only/시각 참조형 같은 decoupled unimodal 세팅에서도 편집 일관성을 크게 개선한다. 특히 FiNE 같은 뉴런 레벨 접근도 멀티모달-성공 후 텍스트-only에서 급락하는 decoupling failure가 나타나는 반면, DECODE는 이를 일관되게 완화한다. 나아가 cross-modal synchronization과 locality(불필요한 지식 훼손 최소화) 측면에서 우수한 성능을 보이며, closed-form이 아닌 내부 활성 기반 편집의 중요성과 모달별 회로 분리 고려 필요성을 실증적으로 강화한다.



New uploads on arXiv(cs.RO)

### Visual Verification Enables Inference-time Steering and Autonomous Policy Improvemen (https://arxiv.org/abs/2606.18247)
Comments:
          Website: this https URL

- **Prior Approaches**: 기존 로봇 정책 학습은 훈련 데이터(시연)나 추가 fine-tuning에 크게 의존해 실환경 적응과 성능 향상을 유도했습니다. 반면 배치 이후의 배치-런타임 경험으로 “즉시” 성능을 올리는 방식은 제한적이거나, 사람 개입이나 추가 학습 파이프라인이 요구되는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 생성기-검증기(generator-verifier) 구조인 VERITAS를 제안해, 사전학습된 일반ist 로봇 정책을 generator로 두고 시각 기반 visual verifier가 추론 시점에 행동을 평가하며 steering하도록 합니다. 핵심은 추가 학습 없이도 inference-time에서 성능을 개선하고, 검증된 롤아웃을 감독 데이터처럼 활용해 이후 self-improvement(오프라인 정책 개선)까지 연결한다는 점입니다.

- **Technical Challenges**: 추론 시점에 행동 품질을 빠르고 안정적으로 판단해야 하지만, 학습 없이 작동해야 하므로 gradient-free 검증 설계가 필요했습니다. VERITAS는 gradient-free visual verifier로 행동을 평가해 inference-time policy steering을 수행하고, 검증된 self-generated trajectories를 다시 fine-tuning에 활용해 정책이 효과적으로 학습하도록 구성했습니다.

- **Empirical Impact**: 실험에서 inference-time verification은 추가 시연 데이터로 학습하지 않은 vanilla generalists보다 일관되게 성능이 우수했습니다. 또한 verified rollouts로 offline policy improvement를 수행했을 때도 일관된 이득이 관찰됐고, post-training 효율이 expert demonstrations에 필적하면서도 인간 개입 없이 진행된다는 점이 특히 의미가 큽니다.



### EBench: Elemental Diagnosis of Generalist Mobile Manipulation Policies (https://arxiv.org/abs/2606.18239)
- **Prior Approaches**: 기존 시뮬레이션 벤치마크는 RLBench·CALVIN·LIBERO처럼 고정 베이스 단일(또는 짧은) 상호작용에 치우치거나, RoboCasa·RoboTwin·GenManip처럼 범위를 넓혀도 성공률 단일 스칼라로 귀결되는 경우가 많았습니다. 또 RMBench처럼 특정 역량 축만 따로 진단하는 방식은 종합적인 일반화 패턴(롱호라이즌·덱스터러스·모바일)을 함께 보긴 어렵습니다. 그 결과 “어디가 강점이고 어디서 무너지는지”, 그리고 배포 분포가 학습 분포에서 벗어날 때 패턴이 어떻게 바뀌는지에 대한 구조적 분석이 부족했습니다.

- **Core Contribution**: 이 논문은 일반ist 모바일 매니퓰레이션 정책을 단일 성공률로만 평가하지 않고, EBench라는 진단형 시뮬레이션 벤치마크를 제안합니다. 26개 태스크를 모바일 픽앤플레이스·모바일 롱호라이즌·테이블탑 덱스터러스/정밀 영역으로 묶고, 각 태스크를 5개 역량 차원(장면/원자 스킬/시간 지평/정밀도/운용 모드)과 4개 generalization 차원(배경/물체/지시어 패러프레이즈/혼합)으로 주석해 성능을 해석 가능한 좌표로 분해합니다. 또한 단일 체크포인트로 전 영역(모바일·덱스터러스·롱호라이즌)을 함께 풀도록 설계해 “전체 점수 뒤의 약점”을 드러내는 데 초점을 둡니다.

- **Technical Challenges**: 모바일·롱호라이즌·덱스터러스를 한 프로토콜에서 학습 데이터/평가로 통합하려면, 덱스터러스 정밀 접촉은 텔레오퍼레이션 기반 수집이 어렵고, 롱호라이즌은 성공 시연 자체의 난이도가 누적 실패로 급증하며, 모바일은 기저(base)와 팔(arm)을 같은 제어 흐름으로 조정해야 해 촘촘한 수집이 힘듭니다. EBench는 이 문제를 두 스트림으로 해결합니다: 덱스터러스 태스크는 kinematically isomorphic actor-follower 텔레오퍼레이션으로, 나머지는 key-frame pose와 cuRobo 기반 모션 플래너로 생성합니다. 평가 역시 태스크별 손코딩 거리함수 대신 시뮬레이터 상태에서 공통 평가 primitive(오브젝트 관계, 관절 각도, 기울기/방향, 단계 순서 조건 등)를 조합해 stage-by-stage 점수와 최종 SR을 산출하도록 구성했습니다.

- **Empirical Impact**: 4개 최신 VLA(π0, π0.5, XVLA, InternVLA-A1)는 테스트 SR은 24.4–29.5%로 비슷했지만, 5차원 역량 프로필과 일반화 양상은 수십 포인트까지 갈렸습니다. π0.5는 가장 높은 테스트 SR과 최선의 train–test retention을 보였고, InternVLA-A1은 모바일에는 강하지만 덱스터러스 고정 베이스에서 붕괴했으며, XVLA는 원자 스킬 일부에서만 두드러지는 ‘분리된 강점’을 보였습니다. 또한 generalization 관점에서 배경/지시어 변화는 비교적 견고했지만 물체 교체와 배경+물체+지시어의 혼합(Mix)이 가장 큰 난이도였고, 나아가 EBench는 from-scratch 대비 pretraining 이득을 일관되게 크게 보여 LIBERO·RoboTwin 2.0과 달리 “대규모 사전학습 효과”를 구분해내는 데 적합하다는 점도 실증했습니다.



### Beyond Failure Recovery: An Engagement-Aware Human-in-the-loop Framework for Robotic Systems (https://arxiv.org/abs/2606.18189)
Comments:
          Project website at this https URL

- **Prior Approaches**: 기존 human-in-the-loop 로보틱스는 주로 실패나 불확실성이 커질 때만 사용자를 호출해, 로봇 성능(robustness·reliability) 개선에 초점을 맞춰 왔습니다. 그 결과 상호작용은 필요할 때만 발생하는 “반응형” 구조가 많아, 사용자가 장시간 작업에서 수동적 관찰자가 되기 쉽다는 한계가 지적돼 왔습니다. 또 다른 흐름에서는 사용자의 workload를 줄이기 위한 질의 빈도 제어를 다루지만, engagement 자체를 목표로 명시적으로 최적화하는 접근은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 Engagement-aware Model Predictive Control(E-MPC)이라는 프레임워크로, 사용자의 engagement를 작업 수행 중에 “목표 상태”로 두고 능동적으로 조절하도록 제안합니다. 로봇이 어려워질 때만 묻는 방식이 아니라, 작업 전반에 걸쳐 자율성과 상호작용의 균형을 맞추면서 원하는 engagement 수준(g_des)을 추적합니다. 또한 사용자 workload 제약과 task success를 함께 고려해, 지나친 잦은 질의로 인한 피로를 억제하면서도 참여감을 유지합니다.

- **Technical Challenges**: 핵심 난제는 engagement가 사용자마다 주관적이며(개인차), 질의 빈도와 유형에 따라 시간이 지나며 어떻게 변하는지(동역학) 모델링하기 어렵다는 점입니다. E-MPC는 engagement를 관측 불가능한 잠재 상태로 두고, “질의 응답 난이도”와 “질의 주기”가 engagement에 미치는 영향을 반영하는 interaction dynamics model을 세워 MPC 비용에 포함합니다. 더불어 task 수행 신뢰도(confidence)가 임계값 이하로 떨어지고 재시도 한도를 넘기면 필수적으로 사용자 보조 질의를 강제하고, workload는 안전 임계값을 비용으로 강하게 패널티해 MPC가 합리적 상호작용을 선택하도록 합니다.

- **Empirical Impact**: 시뮬레이션에서는 다양한 persona(낮음/중간/높음 g_des)와 조건(psuccess, τw)을 바꿔가며 E-MPC가 baseline 대비 engagement 추적 정확도와 만족도 지표를 개선하면서도 작업 성공률을 유지함을 보였습니다. 특히 AlwaysQuery/ NeverQuery처럼 극단적인 상호작용 정책보다, E-MPC가 engagement–workload–success의 트레이드오프를 더 균형 있게 다루는 경향이 관찰됐습니다. 나아가 mobility limitations를 emulation한 실제 사용자 연구에서, 로봇 보조 bite acquisition 시스템에서 사용자 경험(참여감)을 개선하면서도 task success는 유지하는 결과를 제시해 실사용 가능성을 강화했습니다.



### Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System (https://arxiv.org/abs/2606.18112)
- **Prior Approaches**: 기존 agentic navigation 연구는 지시 따르기, 물체 탐색, 타깃 추적, 자율주행처럼 서로 다른 시각 소비 전략이 필요한 과업을 같은 백본으로 묶어도, 추론 시 observation 전략을 외부에서 재구성하기가 어려웠습니다. 특히 trajectory(경로) 중심 학습은 reactive action-sequence mappers처럼 행동만 즉시 매핑하는 형태로 붕괴하는 문제가 보고되어 왔습니다. 결과적으로 과업 전환이나 장기 임무에서의 동적 제어가 제한적이었습니다.

- **Core Contribution**: 이 논문은 Qwen-RobotNav를 제안하며, 추론 시 архитект처 변경 없이 observation strategy를 조절할 수 있는 파라미터화된 인터페이스를 핵심 기여로 내세웁니다. 인터페이스는 (1) task mode로 네비게이션 행동을 선택하고, (2) token budget, per-camera weights 같은 observation 파라미터로 시각 히스토리 인코딩 방식을 제어합니다. 또한 상위 플래너가 에피소드 중간에 task mode와 컨텍스트 전략을 전환해 같은 모델을 반복 호출하며 복잡한 행동을 조합할 수 있게 만듭니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 파라미터화된 설정이 바뀌어도 백본이 과업 지시를 안정적으로 따르며, 경로 데이터만으로는 생기기 쉬운 반응형 매퍼 붕괴를 피하는 것이었습니다. 논문은 학습 시 모든 파라미터에 대한 randomization을 적용해 zero architectural modification 수준의 추론 설정 강건성을 확보하고, vision-language 데이터를 함께 학습(co-training)해 reactive 붕괴를 막았습니다. 더불어 장기 시나리오에서는 상위 플래너가 목표를 분해한 뒤 모델의 task mode/컨텍스트를 동적으로 스위칭하는 방식으로 문제를 해결합니다.

- **Empirical Impact**: Qwen-RobotNav는 15.6M 샘플로 학습되었고, 여러 주요 네비게이션 벤치마크에서 새로운 state-of-the-art를 달성하며 제안한 설계의 실효성을 입증했습니다. 또한 2B에서 8B로의 스케일링에서 성능이 유리하게 개선되었고, multi-task 공동 학습이 공유된 spatial-planning 기반을 형성해 과업군 간 전이를 촉진했습니다. 나아가 다양한 환경에서 실제 로봇에 대한 strong zero-shot generalisation 성과를 보여 agentic navigation에 대한 확장성과 실사용 가능성을 높였다는 점에서 의미가 큽니다.



### WireCraft: A Simulation Benchmark for Industrial DLO Manipulation (https://arxiv.org/abs/2606.18097)
- **Prior Approaches**: 기존 로봇 러닝 벤치마크(RLBench, ManiSkill3)는 주로 rigid 조작에 초점이 있고, DLO는 주변적으로만 다뤄지는 경우가 많습니다. SoftGym, DeformableRavens, DaXBench 등은 변형 물체를 포괄하지만 커넥터/클립/채널 같은 산업용 고정구 제약을 제대로 반영하지 못했습니다. 물리 테스트보드는 표준화를 제공하더라도 시뮬레이션과 공유된 평가 프로토콜, 재사용 가능한 작업 자산을 함께 제공하지 못해 연구 비교가 어려웠습니다.

- **Core Contribution**: 이 논문은 산업용 DLO(Deformable Linear Objects) 조작을 위한 시뮬레이션 벤치마크 WireCraft를 제안합니다. connector insertion, clip routing, channel seating의 3개 태스크 패밀리를 구성 자산/난이도 커스터마이즈 및 단일 정책 인터페이스로 묶고, articulated(관절 사슬)와 deformable(FEM) 두 물리 모델을 함께 제공합니다. 또한 시뮬레이션·물리(UR5)에서 생성한 궤적을 공유 지표 하에 RL·IL·VLA 정책으로 평가할 수 있게 합니다.

- **Technical Challenges**: DLO는 무한차원에 가까운 구성공간과 연속적인 변형, 폐색, 접촉 동역학 불확실성 때문에 ‘정렬(contact-rich alignment)’만으로도 실패 모드가 복잡해집니다. WireCraft는 동일 태스크 정의 하에서 articulated와 FEM 기반 변형 표현을 제공하고, 접촉 센서·힘 센서·로봇 고유감각을 포함한 관측을 기본으로 제공해 접촉 기반 과제를 학습 가능하게 만들었습니다. 더 나아가 스크립트·시뮬 RL·시뮬 인간 텔레오퍼레이션·실세계 UR5 등 다중 소스 데이터를 LeRobot 호환 스키마로 통합해 IL/VLA가 실험적으로 비교되도록 했습니다.

- **Empirical Impact**: 실험 결과, privileged state 기반 PPO/SACfD는 각 태스크 패밀리의 대표 세팅에서 82% 이상 성공하며 시뮬레이션 태스크가 잘 구성되어 있음을 확인했습니다. 반면 vision RL, IL, VLA는 connector insertion에서 reach(접근) 이후 socket 주변 정렬 및 접촉 삽입으로 이어지는 reach–insert gap이 크게 나타나, 현재 비전 기반 학습의 한계를 드러냈습니다. UR5 실세계 검증에서도 시뮬 전용 정책의 zero-shot 전이는 어렵고, 실제 데모를 섞은 경우에만 비영(非零) 삽입 성과가 나타나 sim-to-real 격차가 접촉-rich 삽입에서 지속됨을 보여줍니다. 



### EAGG: Embodiment-Aligned Grasp Generation via Geometry-Aware Graph Conditioning (https://arxiv.org/abs/2606.18092)
Comments:
          16 pages, 8 figures. Code is available at this https URL

- **Prior Approaches**: 기존 cross-end-effector grasp generation은 특정 end effector에 강하게 맞춘 파이프라인(예: Dex-Net, GraspNet 계열)을 만들거나, embodiment identity를 정적 토큰/라벨로 넣어 transfer를 시도하는 방식이 많았습니다. 하지만 topology·actuation coupling·contact geometry가 크게 다른 embodiment에서는 정적 conditioning이 sampling 중에 바뀌는 접촉 가능성과 충돌 패턴을 따라가지 못해 성능이 쉽게 흔들립니다. 또한 많은 방법이 raw joint 좌표로 embodiment를 억지로 동일화해 구조적 차이를 희석하거나, 반대로 구조를 너무 배제해 공통 transfer 신호를 잃는 딜레마가 있었습니다.

- **Core Contribution**: EAGG는 “embodiment 구조를 공유 생성기 내부에서 정렬(aligned)한다”는 방향으로 cross-end-effector grasp generation을 재설계합니다. 각 end effector를 topology-aware end-effector graph와 PCA 기반 저차원 control space로 표현해, 하나의 생성기가 서로 다른 raw joint 파라미터화 없이도 그럴듯한 grasp를 생성하도록 합니다. 또한 생성 중에는 geometry-aware 토큰을 반복적으로 갱신해(Iterative Geometry Injection, IGI) 조건이 현재 관절 상태와 항상 동기화되게 합니다.

- **Technical Challenges**: 핵심 난제는 저차원 control code가 있더라도, 조립/폐쇄 과정에서 실제 관절 기하가 변하면서 충돌·미스컨택트 가능성이 달라지는 점입니다. 정적 embodiment descriptor는 이 “샘플링 동안의 기하 변화”를 반영하지 못하므로, EAGG는 frozen end-effector-cognition backbone으로 현재 articulation에 대응하는 geometry-aware 토큰을 만들고 매 step마다 IGI로 다시 주입합니다. 여기에 그래프 컨디셔닝(기구학 결합/연결 구조)과 PCA 제어공간(embodiment-specific closure)을 함께 써서, 제어표현과 구조표현을 분리하되 상호보완적으로 정렬합니다.

- **Empirical Impact**: EAGG는 MultiGripperGrasp 벤치마크에서 6개 학습 end effector에 대해 평균 성공률 56.17%를 달성하며, specialized 학습 대비 1.10%p 이내로 유지하면서도 finetuning 및 zero-shot end effector로의 transfer를 보존했습니다. 또한 IGI는 pooled median contact distance를 0.239 cm에서 0.189 cm로 줄여 접촉 정밀도가 개선됐음을 보여줍니다. 연구는 embodiment 차이를 억누르는 대신, 생성기 안에서 구조적으로 정렬하면 cross-object 일반화와 cross-end-effector 전이를 동시에 강화할 수 있다는 실증적 근거를 제공합니다.



### A Hybrid Optimization Framework for Grasp Synthesis under Partial Observations (https://arxiv.org/abs/2606.18053)
- **Prior Approaches**: 그립 생성은 최적화 기반과 데이터 기반으로 크게 나뉘는데, 전자는 보통 완전한 객체 모델을 전제로 해 알려지지 않은 물체·부분 관측에서 일반화가 약합니다. AS-ICP 같은 최적화 방식은 부분 입력에 강하지만 그리퍼 aperture 민감도와 다수 preshape 필요로 계산이 커지고, 데이터 기반 방법(AnyGrasp, GPD)은 훈련 데이터가 결국 완전 모델 기반으로 만들어지는 경우가 많아 에너지/후보 분포가 깨질 수 있습니다. 또한 일부 하이브리드는 학습과 기하 최적화를 느슨하게 결합해(학습 평가 지표·후처리 등) 한쪽의 한계를 완전히 상쇄하지 못했습니다.

- **Core Contribution**: 이 논문은 부분 point cloud에서 강건한 grasp를 만들기 위해 EBM(Energy-Based Model) 학습 에너지를 Stein Variational Gradient Descent(SVGD) 안에 prior로 넣고, ICP의 기하 정합(analytical alignment)으로 반복 정제하는 하이브리드 프레임워크를 제안합니다. 핵심은 “학습된 전역 prior로 탐색을 유도 + ICP로 국소 정합을 보정”해, 학습 단독의 분포 붕괴와 분석 단독의 국소 정합 한계를 동시에 줄이려는 설계입니다. 또한 AS-ICP가 생성한 grasp 성공/실패 결과로 EBM을 학습해, 실제로 유효한 grasp 영역을 에너지 지형으로 반영합니다.

- **Technical Challenges**: 주요 기술 난제는 (1) 부분 관측에서 그립 포즈 최적화가 비선형·비볼록일 때 수렴 안정성을 확보하고, (2) EBM 기울기와 ICP 매칭 기울기의 스케일 불일치로 인해 업데이트가 한쪽으로 쏠리는 문제입니다. 이를 위해 SVGD 업데이트에 EBM 에너지 그래디언트를 포함하고, EBM 그래디언트 vs ICP 매칭 그래디언트의 크기 차이를 동적으로 가중해 균형을 맞춥니다. 더불어 ICP 정합 그래디언트는 초기에 강하게, 후반에는 점차 줄이도록 정규화 항을 스케줄링해 번역 파라미터의 초기 탐색 효율을 높입니다.

- **Empirical Impact**: 67개 물체(총 5,360 그립 시도)에서 평균 성공률 60.9%를 달성해 AnyGrasp(31.1%), GPD(48.4%), AS-ICP(56.6%)를 모두 상회했습니다. 특히 학습 데이터 분포(orientation/elevation 그룹 기반 샘플링, 국소 negative sampling)와 EBM 손실/입력 설계가 에너지 지형의 “밀도·피크 형태”를 좌우하며, 구조화된 데이터일수록 더 일관된 성공률 분포를 만든다는 점을 ablation으로 확인했습니다. 또한 학습은 일부(41개) 물체에만 했는데도 미본 물체에서 64.6%로 더 높은 성능을 보여 일반화 잠재력이 크고, 실제 Kinova+KG3 그리퍼 환경에서도 78% 성공률을 보고해 시뮬레이션-기반 파이프라인의 실용성을 뒷받침합니다.



### Uncertainty Quantification for Flow-Based Vision-Language-Action Models (https://arxiv.org/abs/2606.18043)
Comments:
          Project page: this http URL. 28 pages, 12 figures

- **Prior Approaches**: 비전-언어-행동 모델(VLA)은 vision-language backbone에 생성형 action head를 결합하고, flow matching으로 대규모 로봇 데이터에서 end-to-end로 학습해 조작 성능을 크게 끌어올렸습니다. 하지만 기존 접근은 예측이 얼마나 믿을 만한지(uncertainty)와 언제 실패할지(failure detection)를 정량화하는 장치를 충분히 갖추지 못했습니다. 그 결과 비정상 환경에서 pretraining distribution 밖 상황을 만나면 경고 없이 성능이 무너질 수 있습니다.

- **Core Contribution**: 이 논문은 flow-matching VLA의 epistemic uncertainty를 velocity-field disagreement(VFD) 기반으로 효율적으로 추정합니다. 구체적으로 작은 앙상블 간 velocity field(또는 그에 대응하는 sampler/모델 출력)의 불일치를 이용해 신뢰도를 계산하고, 이를 실패 감지와 active fine-tuning에 바로 활용합니다. 또한 불확실성에 의해 시연 데이터를 능동적으로 고르는 SAVE 프레임워크를 제안해 새 작업으로의 적응에 필요한 expert demonstration 수를 줄입니다.

- **Technical Challenges**: 핵심 기술 과제는 “표현형 생성”을 하는 flow-based 모델에서 epistemic uncertainty를 어떻게 계산해 실제 성공/실패와 연결할 것인가입니다. 연구팀은 앙상블 간 velocity-field disagreement이 불확실성을 나타낸다는 관점을 세우고, VFD 가중치가 경계(s→1)에서 발산처럼 보이는 수식 상황에서도 실제 추정기는 grid 기반 샘플링으로 안정적으로 finite 하게 만들었습니다. 그 다음, conformal prediction을 활용한 failure thresholding 및 uncertainty-guided episode selection을 결합해 배치 배치가 아닌 배포 단계에서 오작동을 조기에 차단하도록 설계했습니다.

- **Empirical Impact**: LIBERO 벤치마크에서 VFD 기반 불확실성은 downstream 성능을 더 잘 예측하는 보정(calibration)된 값을 보여주고, failure detection에서도 강한 성능을 보였습니다. 특히 SAVE와 결합했을 때 uncertainty-guided data acquisition이 기준선 대비 최소 22% 적은 샘플로 목표 성능에 도달해 sample efficiency를 유의미하게 개선했습니다. 요약하면 VFD로 신뢰도 인식을 강화하는 것이 실패 awareness와 신규 작업 적응을 동시에 끌어올린다는 점을 실험적으로 입증한 연구로 평가됩니다.



### LAGO Policy: Latency-Aware Asynchronous Diffusion Policies with Goal-Directed Collision-Free Planning for Smooth Manipulation (https://arxiv.org/abs/2606.17982)
Comments:
          8 pages, 8 figures

- **Prior Approaches**: 확산 기반 visuomotor imitation learning은 다양한 manipulation에서 강점을 보였지만, physical manipulator에서 비동기 inference를 쓰면 청크(액션 chunk) 경계에서 불일치가 생기고 motion이 끊기거나(jerky) 충돌 위험이 커질 수 있다. 기존에는 미래 액션 조건을 추가해 inter-chunk continuity를 노리거나(SAIL 등), 안전장치를 추론 시 국소적으로 적용해 충돌을 피했지만 temporally shifted future-action condition에 취약하거나, short-horizon 보정이 최적 회피를 만들지 못해 안전 개입이 잦고 성능 저하로 이어졌다. 또한 장애물이 관측 밖으로 등장하는 out-of-distribution 상황에서 collision-free한 실행 집합을 명시적으로 학습/보장하지 못하는 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 LAGO Policy를 제안해 비동기 액션 생성의 inter-chunk 불연속과 장애물 회피의 안전성 문제를 함께 다룬다. latency-aware classifier-free guidance(CFG)로 미래 액션 조건의 시점 불일치를 견디도록 하여 청크 겹침 구간의 일관성을 높이고, 데모에서 학습한 task-relevant interaction goal을 예측해 goal-directed collision-free trajectory planning을 트리거한다. 마지막으로 spatial-temporal trajectory optimization으로 실행될 액션을 low-jerk이면서 feasible하게 다듬어 끊김과 충돌을 동시에 줄인다.

- **Technical Challenges**: 핵심 기술 난제는 비동기 inference 지연 때문에 training에서 정렬(aligned)되던 미래 액션 조건이 실제 실행(overlap) 시점과 어긋나는 ‘perception-execution misalignment’이 연속 청크의 노이즈 예측을 깨뜨리는 점이다. 이를 위해 논문은 미래 액션 조건을 관측 특징과 분리해 CFG 샘플링에서 주입하고, training 동안 future-action condition의 지연을 랜덤화(delay randomization)해 temporally shifted 조건에도 흔들리지 않게 학습한다. 안전 측면에서는 관측 장애물에 의해 직선/단순 회피가 막히는 상황에서 전역적으로 일관된 회피를 만들기 위해, 목표 예측→충돌 체크→goal-directed trajectory optimization(스플라인 기반, 장애물 비용/제약)로 전환하는 파이프라인을 설계했다.

- **Empirical Impact**: 실세계 로봇 실험에서 LAGO Policy는 6-DoF ARX5와 7-DoF Franka의 다양한 manipulation 작업(픽앤플레이스, 삽입, 액체 운반/기울이기, deformable/관절 객체 등)을 대상으로 smooth collision-free 실행과 높은 task success를 보였다. 특히 latency-aware CFG와 지연 랜덤화는 inter-chunk consistency(청크 겹침 불일치)를 개선하고, spatial-temporal trajectory optimization은 실행 경로의 jerk(움직임 매끈함)를 낮추는 방향으로 작동했다. 결과적으로 short-horizon 안전 보정 위주의 기존 방식 대비 장애물이 예기치 않게 나타나는 상황에서도 더 안정적으로 작업을 완료하는 의미 있는 향상을 입증했다.



### ThinkingVLA: Interleaved Vision and Language Reasoning for Robotic Manipulation (https://arxiv.org/abs/2606.17937)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 관측을 곧바로 행동으로 매핑해 추론이 약해, 긴 지평선(long-horizon) 조작 과제에서 성능이 제한된다. Chain-of-Thought(CoT) 기반 접근은 하위 목표 분해와 공간적 예상을 돕지만, 텍스트-비전 추론을 한 아키텍처로 일관되게 결합하지 못하고 목표 상태를 바탕으로 한 inverse reasoning을 명시적으로 포함하진 못한다.

- **Core Contribution**: ThinkingVLA는 조작 계획을 ‘예측(다음 시각 상태 예측) + 역동역학(목표 상태까지 도달하는 행동 추론)’으로 자연스럽게 분해하고, 이를 하나의 생성 과정에서 통합한다. 이를 위해 단일 Mixture-of-Transformers 아키텍처에서 forward CoT(즉시 하위목표 및 시각 포워캐스트)와 inverse CoT(예측 이미지를 목표 상태로 삼아 공간 관계와 행동 의도를 추론)를 연쇄 생성해 행동을 산출한다.

- **Technical Challenges**: 핵심 난제는 서로 다른 매체(텍스트 추론, 이미지 예측, 공간 관계 기반 역추론)를 효과적으로 한 생성 파이프라인에 엮는 통합 설계였다. 논문은 예측된 이미지를 목표 상태로 ‘접지(grounding)’하여 inverse CoT가 타겟 기반으로 추론하도록 하고, 통합 autoregressive 생성에서 텍스트·시각 reasoning을 interleave하는 Mixture-of-Transformers 구조로 이를 해결한다.

- **Empirical Impact**: 시뮬레이션과 실제 환경 벤치마크 전반에서 ThinkingVLA는 기존 SOTA 대비 일관되게 성능이 높았고, 특히 긴 지평선 조작 과제에서 개선 폭이 크게 나타났다. 이는 VLA가 단순 매핑을 넘어 forward prediction과 inverse dynamics 추론을 결합할 때 장기 계획형 조작에 강점을 보일 수 있음을 실증적으로 보여준다.



### SPARK: Low Latency Single-Camera 3D Pose Estimation for Autonomous Racing using Keypoints (https://arxiv.org/abs/2606.17936)
Comments:
          9 pages, 6 figures, ITSC 2026, Invited Session

- **Prior Approaches**: 기존 자율주행 레이싱 연구는 다른 차량을 실시간 추적하기 위해 초저지연이 중요하다고 봤지만, LiDAR 기반 파이프라인은 센서 주기(10~30Hz)와 후처리/복셀화로 인해 지연이 커져 고기동 구간에서 추적 성능을 제한해왔다. 비전 기반은 빠르지만 monocular 3D에서 depth 회귀가 어렵고, 이를 보완하려는 2D 백본+3D 헤드/트랜스포머/복잡한 모듈 조합은 지연을 늘리기 쉽다. 또한 레이싱 도메인은 목표 물체가 단일 클래스이고 크기가 고정인데도, 대부분의 3D 검출 모델은 더 일반적인 상황을 전제로 설계돼 비효율이 발생한다.

- **Core Contribution**: 이 논문은 단일 카메라로 상대 차량의 6D pose를 추정하는 SPARK를 제안한다. 핵심은 keypoint 탐지(YOLO-Pose 계열)로 빠르게 2D 대응점을 얻고, 고정된 차량 치수/기하 제약을 활용해 PnP(SQPnP)로 pose를 복원해 monocular 3D 검출의 지연-정확도 균형을 개선하는 것이다. 또한 레이싱용에 맞춘 2D 키포인트 어노테이션 생성(3D LiDAR ground truth 투영+수정) 방법과, 가림(occlusion)을 학습에 반영하는 keypoint visibility 설계를 공개한다.

- **Technical Challenges**: 주요 기술 난제는 (1) 가림과 관점 변화로 특정 키포인트(예: 타이어)가 사라질 수 있어 PnP 입력이 불안정해지는 점, (2) monocular 입력에서 pose 복원이 누적 오류를 만들 수 있는 점이다. 이를 위해 차량 방향/가림 상황에서도 최소 4개 이상이 항상 보이도록 9개 keypoint를 설계하고, 보이지 않는 점은 visibility 플래그(v_vis)로 학습에서 구분해 PnP 전 단계에서 필터링하도록 했다. 추가로 camera 왜곡/캘리브레이션 기반 PnP 복원을 사용해 입력을 rectification하는 비용을 줄였고, 인퍼런스 최적화(FP16+TensorRT)로 지연을 더 낮췄다.

- **Empirical Impact**: 실세계 자율 레이싱 데이터에서 SPARK는 레이싱용 단일 클래스 설정 하에 장거리에서도 높은 정확도로 동작하며, 경쟁하는 monocular 3D 검출(SOTA) 대비 더 낮은 지연에서 더 나은 성능을 보였다고 보고한다. 특히 keypoint visibility 기반 필터링이 정확도를 가장 크게 개선했고, PnP+NMS 단계의 순수 추정 지연은 약 0.1ms 수준이라 2D keypoint 모델의 지연이 지배적임을 강조한다. LiDAR와의 직접 비교에서는 AP가 완전히 동일하진 않지만 orientation 오차가 더 안정적이며, 레이싱에서 tracking은 지연 보상/공분산 조정(EKF 등)으로 translation 오차 영향을 완화할 수 있음을 시사한다. 오픈소스 코드와 레이싱 특화 키포인트 생성 파이프라인을 제공해, 단일 카메라 저지연 3D 인식의 실용성을 높였다는 점에서 의미가 있다.



### PearlVLA: Progressive Embodied Action-Plan Refinement in Latent Spac (https://arxiv.org/abs/2606.17924)
Comments:
          21 pages, 2 figures. Preprint

- **Prior Approaches**: 기존 VLA는 비전-언어 표현에서 바로 행동을 디코딩하거나, 텍스트 추론 trace·시각 서브목표·action search·world model의 후보 탐색처럼 명시적 중간 단서를 거쳐 계획을 개선해 왔습니다. 하지만 텍스트/픽셀 기반 계획은 지연과 계산비용이 크고, 후보를 많이 평가하는 WM 롤아웃은 스케일 문제가 생깁니다. 반면 end-to-end 성격의 direct decoding은 빠르지만 단발(single-pass)이라 현재 계획이 만든 미래를 미리 보고 self-correct할 여지가 제한적입니다.

- **Core Contribution**: PearlVLA는 VLA의 deliberation(계획 숙고)을 텍스트·픽셀·action search 대신 VLM이 만든 latent 공간 내부로 옮깁니다. VLM의 meta-query를 고정 visual grounding 분기와 반복형 latent plan 분기로 분리하고, 매 라운드마다 plan-conditioned world query로 action-free 미래 관측 latent를 얻어 plan 토큰을 residual로 점진 정제합니다. 최종 refined latent plan은 병렬로 action chunk로 디코딩되어, 숙고의 계산은 늘리되 실행은 저지연 경로를 유지하는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “latent 숙고가 현재 장면에 접지(grounding)되면서, 현재 plan이 유도하는 미래 피드백이 제대로 반영”되도록 폐루프를 설계하는 것입니다. 논문은 world query의 condition drift를 줄이기 위해 anchored world query를 두고, residual write-back을 scheduled로 감소시키는 coarse-to-fine 업데이트를 적용합니다. 또한 중간 라운드별 편집이 미래 성공에 주는 인과효과를 가치함수 없이 학습하기 위해, Causal Refinement-Grouped Process-Reward RL(CRG-PRL)을 제안해 같은 상태 내에서 그룹 상대 보상으로 residual 편집을 비교 최적화합니다.

- **Empirical Impact**: LIBERO 벤치마크에서 PearlVLA는 supervised만으로도 평균 성공률을 97.1%→98.5%로 끌어올렸고, CRG-PRL까지 적용하면 98.7%로 SOTA를 달성했습니다. 또한 refinement round 수에 따른 성능 향상과, K=0(정제 제거) 대비 K=4에서의 격차 확대로 latent refinement의 기여를 확인했습니다. LIBERO-Plus에서도 OpenVLA-OFT 대비 69.6%→76.3%로 강한 강건성 향상을 보여, 지연을 늘리지 않는 latent 기반 계획 정제가 장기 일반화에도 유효함을 시사합니다.



### WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SF (https://arxiv.org/abs/2606.17906)
- **Prior Approaches**: World-Action(WA) 모델은 비디오 생성형 모델로 미래 관측과 행동을 함께 예측하며 장기 의사결정에 유리하다는 평가를 받았다. 다만 대부분이 expert trajectory에 대한 supervised learning에 의존해, 시연 분포 밖의 세밀한 조작 기술을 습득하거나 실제 상호작용으로 지속 개선하기 어렵다. RL을 VLA에 적용한 연구들은 주로 action 정책만 최적화하고, WA의 world model과 actor가 만드는 잠재공간 결합 문제는 상대적으로 다루지 않았다.

- **Core Contribution**: 논문은 World-Action 패러다임에 reinforcement learning을 도입한 WAM-RL을 제안하며, online interaction을 통해 world model과 action model( actor )을 함께 최적화한다. 핵심 아이디어는 WA의 주된 능력이 world model에 있고 actor는 이를 행동으로 번역하는 역할이라는 관점에서, 두 구성요소를 계층적으로 맞춰가며 co-evolve시키는 것이다. 또한 actor-only 최적화가 단기 과제엔 도움 되지만 장기 과제 성능 향상엔 한계가 있으며, 공동 최적화가 장기 성능을 좌우한다는 인사이트를 제시한다.

- **Technical Challenges**: 가장 큰 난제는 actor가 world model의 잠재공간 분포에 강하게 의존한다는 점에서, world model을 온라인으로 바꾸면 latent distribution shift로 RL 학습이 불안정해진다. 논문은 이를 위해 world model에 online video SFT를 적용하되 KL regularization으로 업데이트된 latent 공간이 사전학습 분포에서 급격히 벗어나지 않게 제약한다. actor는 world model이 상상한 미래와 실제 실행 결과의 일치를 재구성 기반 dense reward로 삼아, RL 목표가 “예측한 계획을 그대로 실행으로 grounding”하도록 설계했다.

- **Empirical Impact**: 실험은 LIBERO와 RLBench에서 일관된 성능 개선으로 나타났고, 특히 LIBERO-Object는 성공률 68%→82%로 actor-only RL 대비 큰 격차를 보였다. RLBench Water Plants에서도 19%→22% 개선이 관측됐으며, actor-only RL은 기준선(base)을 넘지 못했다. 더 나아가 online video SFT로 world model이 실패 후 회복(recovery) 행동을 더 자주 예측하게 되어, single open-loop chunk 안에서도 corrective behavior가 나타나 정책의 견고성이 향상됨을 정성·정량으로 뒷받침한다.



### Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models (https://arxiv.org/abs/2606.17846)
Comments:
          44 pages

- **Prior Approaches**: 기존의 로봇 조작 연구는 비전-언어-행동을 어느 정도 맞추더라도, 조작 데이터의 이질성·수집 비용·다양성 부족 때문에 정렬(alignment)과 스케일링을 동시에 달성하기 어려웠습니다. 그 결과 다양한 플랫폼과 상황에서의 진짜 일반화보다는 특정 환경/자세에 강하게 편향되는 문제가 반복됐습니다. 또한 많은 벤치마크가 학습 품질을 충분히 반영하지 못해 OOD(out-of-distribution) 상황의 난도가 낮거나 불일치했습니다.

- **Core Contribution**: 이 논문은 로봇 조작에도 언어·멀티모달 파운데이션 모델의 ‘스케일링 레시피’를 적용하려는 시도를 Qwen-RobotManip으로 구체화합니다. Qwen-VL(Qwen-VL 기반) 위에 표현·모션·행동의 3개 차원에서 통일된 정렬 프레임워크를 도입해, 멀티소스 대규모 학습이 충돌하지 않고 함께 수렴하도록 설계했습니다. 더불어 사람(1인칭 핸드 시연) 정보를 로봇 궤적으로 합성하는 파이프라인과 데이터 조정 커리케이션으로 대규모 학습 데이터를 확보합니다.

- **Technical Challenges**: 핵심 기술 난제는 로봇 조작 데이터가 ‘기본적으로 이질적’이라서 표현·움직임·행동을 같은 학습 문맥에 정렬하기가 어렵다는 점입니다. 논문은 이를 해결하기 위해 representation, motion, behavioral dimensions에 걸친 unified alignment framework를 구성해, 서로 다른 데이터 소스가 학습 신호를 일관되게 제공하도록 맞춥니다. 또한 15개 플랫폼에 대한 합성(인간 시연→로봇 trajectory)과 함께 이질적 데이터셋을 조화시키는 rigorous curation pipeline을 통해 스케일을 유지하면서도 품질을 끌어올립니다.

- **Empirical Impact**: 실험에서는 오픈소스 데이터와 인간 비디오만으로 약 38,100시간 규모의 pretraining 코퍼스를 구성하고, zero-shot instruction following, perturbations에 대한 강인성, reactive error recovery, cross-embodiment transfer 같은 emergent generalization을 보여줍니다. 더 나아가 표준 벤치마크는 사전학습 품질을 잘 반영하지 못하며, RoboCasa365·LIBERO-Plus·EBench·RoboTwin-* 등 OOD 설정에서 차이가 드러난다고 지적합니다. 그 결과 Qwen-RobotManip은 모든 OOD 설정에서 기존 SOTA를 크게 앞섰고, pi0.5를 포함한 모델 대비 성능 우위를 보이며 RoboChallenge 1위(상대 20% 개선) 및 AgileX ALOHA, Franka, UR, ARX 같은 실제 로봇 플랫폼에서도 검증됩니다.



### From Ad Hoc Pilots to Repeatable Patterns: Structuring Drone Collaboration in Emergency Services with DroneLets (https://arxiv.org/abs/2606.17839)
Comments:
          Presented at International Conference on Information Systems (ICIS) 2025: this https URL

- **Prior Approaches**: 기존 연구는 재난 대응에서 드론을 쓰는 경우가 많지만, 실제 현장에서는 작업이 임시방편적(ad hoc)이고 조율 비용이 커서 반복 가능한 절차로 굳히기 어려웠습니다. 또한 협업을 설계할 때 인간과 드론의 상호작용을 어떻게 “프로세스”로 형식화할지에 대한 기준이 부족했습니다.

- **Core Contribution**: 이 논문은 응급팀이 드론과 어떻게 협업하길 원하는지(95개 인터뷰)와 이를 반복 가능한 프로세스로 만드는 방법을 4개의 현장 시험을 통해 정리합니다. 그 결과 44개의 상호작용 패턴을 10개의 메타-패턴으로 묶고, 재정찰·통신·물류 지원 같은 운영 니즈를 반영함을 보여줍니다. 이를 바탕으로 Collaboration Engineering을 embodied agents까지 확장하는 설계 산출물 ‘DroneLets’를 제안합니다.

- **Technical Challenges**: 핵심 난제는 현장 제약(환경/장비/권한) 속에서 인간-드론-상황을 함께 고려한 협업을 재사용 가능한 형태로 구조화하는 것입니다. 논문은 DroneLets로 세팅 요구사항, 드론 기능, 환경 제약, 그리고 인간과 드론 간 조율된 액션을 모듈로 캡처해 패턴 기반 설계가 되도록 해결합니다. 예를 들어 방관자 대상 브로드캐스팅, 화재 후 모니터링 같은 시나리오를 패턴으로 체계화합니다.

- **Empirical Impact**: 4개 현장 시험과 인터뷰를 근거로 패턴 체계를 도출해, 응급 서비스에서 드론 협업이 어떤 방식으로 반복될 수 있는지 실증적으로 제시합니다. 또한 CE의 적용 범위를 고위험(field operations)으로 넓혀, 자율 드론을 워크플로에 통합할 때 필요한 설계 언어와 프레임을 제공한다는 점에서 의미가 큽니다. 결과적으로 응급 드론 협업을 더 확장 가능하고(Scalable) 모듈화된 절차로 전환할 기반을 마련했다는 평가가 가능합니다.



### HumanoidArena: Benchmarking Egocentric Hierarchical Whole-body Learning (https://arxiv.org/abs/2606.17833)
Comments:
          29 pages, 13 figures, 10 tables

- **Prior Approaches**: 기존 휴머노이드 whole-body 학습은 시각/언어 기반 고수준 결정과 동적 실행이 하나의 스택 안에 결합돼 있어, policy–tracker 인터페이스 자체를 분리해 검증하기 어렵습니다. 또 많은 벤치마크가 팔 중심 조작이나 단순 보행/내비게이션에 초점을 두어, 성공에 다리가 구조적으로 필수인 leg-critical 상호작용을 충분히 평가하지 못합니다. 결과적으로 “중간 whole-body 행동이 실제로 실행 가능하고, 분포 변화에도 견고하며, 다른 GMT 백엔드로도 옮겨가는가”는 불명확한 상태였습니다.

- **Core Contribution**: 이 논문은 egocentric 시각을 입력으로 하는 hierarchical whole-body learning을 정책-실행 계층으로 명시해, 고수준 정책이 compact intermediate whole-body action을 예측하면 저수준 GMT(General Motion Tracker)가 이를 안정적 움직임으로 실행하는 구조를 표준화합니다. 시뮬레이션-퍼스트 벤치마크인 HumanoidArena를 제안하고, leg-critical HOI/HSI 7개 과제를 통해 발 디딤, 균형 유지, 자세 조정, 전신 재지향 같은 하체 조정이 과업 성공을 좌우하도록 설계했습니다. 또한 perturbation-conditioned 일반화와 GMT-conditioned 전이를 함께 평가해, 중간 표현의 이전 가능성과 정책-트래커 궁합을 같은 인터페이스에서 측정할 수 있게 했습니다.

- **Technical Challenges**: 핵심 난제는 고수준이 의도(task intent)를 보존하는 중간 whole-body 행동을 출력해야 하는 동시에, 실행층은 접촉/자세/시점 변화 하에서 동적으로 feasible한 궤적을 만들어야 한다는 “인터페이스-동역학 결합” 문제입니다. HumanoidArena는 40D canonical intermediate whole-body action 인터페이스(루트 수평 이동/높이/방향, 29개 관절 목표, 양손 open/close)를 고수준 출력으로 고정해 GMT마다 필요한 어댑터로 매핑하도록 했고, 이를 통해 tracker별 모션 프라이어/실패 모드의 영향을 통제하며 진단합니다. 데이터 수집은 Isaac Lab 기반의 VR egocentric teleoperation으로 closed-loop 시연을 만들고, 시각·의미·실행 축의 perturbation과 in/cross-GMT 배치를 같이 제공해 견고성/호환성을 분해 평가합니다.

- **Empirical Impact**: 실험에서 ACT, Diffusion Policy, Flow Matching, VLA-style 정책들은 leg-critical 상호작용을 대체로 해결하지만, 성능은 GMT 선택에 강하게 조건화(tracker-conditioned)되어 cross-GMT transfer는 취약하게 나타났습니다. 즉, 학습된 중간 whole-body 행동이 특정 실행 백엔드의 모션 습관에 과도하게 맞춰질 경우 다른 GMT로 배치됐을 때 성공률이 크게 떨어질 수 있음을 보여줍니다. 저자들은 이를 바탕으로 HumanoidArena를 “transferable intermediate action representations”와 “scalable egocentric whole-body policy learning”을 연구하기 위한 벤치마크 축으로 자리매김합니다.



### Accountability in Autonomous Drone-Based Firefighting: Insights From a Field Tria (https://arxiv.org/abs/2606.17831)
Comments:
          Accepted for Publication at International Conference on Information Systems (ICIS) 2025: this https URL

- **Prior Approaches**: 재난 대응에서 자율 드론이 현장 효율을 높일 수 있다는 연구는 빠르게 늘고 있지만, 기존 접근은 주로 기술 성능(탐지·비행·통신) 중심으로 설계/검증되는 경향이 강하다. 또한 드론을 조직의 기존 팀과 워크플로에 넣을 때 생기는 책임소재(accountability) 변화는 상대적으로 덜 다뤄져 왔다.

- **Core Contribution**: 본 논문은 복잡한 사회기술 시스템에서 자율 드론이 책임귀속(accountability attribution)에 어떤 영향을 주는지 실제 현장 근거로 분석한다. 소방(firefighting) 분야의 2개 현장 실험(field trials)에서 드론이 조직적으로 배치될 때 책임 판단에 상당한 불확실성이 발생함을 보여준다.

- **Technical Challenges**: Bovens의 책임 프레임워크를 바탕으로, (1) 드론의 역할이 계층 구조(hierarchical structures)에서 어떻게 위치하는지에 대한 불확실성으로 인해 책임 귀속이 혼란스러워지고, (2) 인간-드론 상호작용이 새롭게 생기면서 책임과 관련된 추가 쟁점이 등장하는 문제를 확인한다. 연구는 이 두 난제를 책임 프레임의 관점에서 체계적으로 분류해, 기술 투입이 조직적 책임 구조에 미치는 영향을 구체화한다.

- **Empirical Impact**: 연구 결과는 정책결정자와 실무자에게, 책임소재를 훼손하지 않으면서 자율 드론을 소방 작전에 ‘책임 있게’ 통합하기 위한 실행 가능한 권고안을 제공한다. 더 나아가 자율 시스템의 책임성(accountability) 연구를 확장하는 경험적 근거와 가이드라인을 제시한다.



### ED3R: Energy-Aware Distributed Disaster Detection Enabled by Cooperative Robotic Agents (https://arxiv.org/abs/2606.17739)
Comments:
          14 pages, 9 figures

- **Prior Approaches**: 기존 에너지 효율 연구는 네비게이션·센싱·인프라 자원·연산 오프로딩 같은 개별 요소를 줄이는 데 집중하는 경우가 많았고, 시간 제약을 만족하는 쪽이 주된 목표였습니다. SAR/UAV 관련 연구들도 위험 환경에서의 탐색·객체 탐지·경로 최적화는 다루지만, 산불 탐지를 위해 이동·통신·연산 오프로딩을 함께 최적화하는 통합 접근은 상대적으로 부족했습니다.
또한 산불 탐지 분야의 다수 방법은 분산 제어나 규칙 기반에 치우쳐 있었고, 미래 결과를 미리 평가하는 forward-looking reasoning은 거의 없거나 결여되어 있었습니다.

- **Core Contribution**: ED3R은 불확실성 하에서 산불을 “요구 confidence로 탐지”하면서, 로봇의 이동·센싱·컴퓨팅·통신으로 소모되는 에너지를 최소화하도록 설계된 energy-aware 분산 프레임워크입니다. 로봇과 원격 컨트롤러(RC) 사이에 계층적 협력 의사결정을 두는데, RC는 motion command를 정하고 로봇은 탐지 실행 위치(onboard vs remote)와 사용할 모델(how)을 자원 기반으로 선택합니다.
또한 장애물 회피·중복 탐색 방지·적응적 early mission completion·페널티 함수로 제약 가능성을 강화해, 단순 성능 최적화가 아닌 임무 성공을 겨냥합니다.

- **Technical Challenges**: 가장 큰 어려움은 RC와 로봇이 분산된 상태에서 행동 결과에 대한 공통 보상이 “동시에” 주어지지 않는다는 점입니다. ED3R은 이를 해결하기 위해 distributed neural regression 모델로 후보 전략들의 미래 효과를 미리 평가한 뒤, 탐지 confidence와 에너지 효율 사이의 최적 trade-off를 그리디하게 선택합니다.
여기에 통신 대역폭/전송 파워/채널 조건, 센서 샘플 크기와 처리 FLOPs 같은 현실적인 에너지·지연 요소를 모델링해 제약 위반을 커스텀 페널티로 반영합니다.

- **Empirical Impact**: 현실적인 로보틱스 시뮬레이션과 ablation, 베이스라인 비교를 통해 ED3R은 최대 97.18% 임무 성공률을 달성했습니다. 특히 가장 까다로운 임무에서 에너지는 최대 36.4% 줄이고, 산불 탐지는 최대 41% 더 빠르게 수행해 시간-에너지-신뢰도 동시 최적화의 실효성을 보여줍니다.
forward-looking과 분산 계층 의사결정이 결합될 때, 네트워크가 바뀌거나 새로운 산불 상황에서도 강건한 성능을 보이며 관련 분야의 산불 감시/긴급 대응 설계 방향에 의미 있는 근거를 제공합니다.



### ERQA-Plus: A Diagnostic Benchmark for Reasoning in Embodied AI (https://arxiv.org/abs/2606.17639)
Comments:
          under review at NeurIPS

- **Prior Approaches**: 기존 VQA/EQA 벤치마크는 높은 정답률을 만들 수 있어도, 모델이 실제로 ‘근거 기반(reasoning with grounding)’인지 ‘언어/시각 지름길(shortcut)’인지 분리해 진단하기 어렵습니다. 특히 EQA는 탐색이나 과업 수행에 초점이 치우쳐 있어, 평가하고 싶은 추론 의존성(공간·절차·의도 등)을 정밀하게 통제하기가 어렵습니다. 그 결과 카테고리별 실패 모드를 세밀히 분해해 개선 방향을 찾기 힘들다는 한계가 있습니다.

- **Core Contribution**: ERQA-Plus는 로봇 중심 이미지에 근거해 embodied AI의 추론을 진단(diagnostic)하도록 설계된 벤치마크입니다. 퍼셉션, 액션 중심, 사회/상호작용, 내비게이션·환경, 맥락·상식 등 5개 축을 포함하는 계층적 taxonomy로 질문을 구성해, 어떤 형태의 추론이 필요한지 세분화해 평가합니다. 총 1,766개 QA를 통해 ‘정답 여부’뿐 아니라 ‘어떤 추론을 안정적으로 수행하는지’를 보여주는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트 단서 없이 장면 증거에 의존하게 만들고, (2) 정답이 명확하며 (3) 공간·절차·사건 예측 같은 고난도 의존성을 정확히 검증하는 QA를 대규모로 생성하는 것입니다. 이를 위해 taxonomy-guided 질문 생성과 함께 자동 quality judging, 반복 수정(revision), 그리고 인간 검증을 포함하는 multi-stage 파이프라인을 사용해 시각적 grounding과 추론 품질을 끌어올립니다. 또한 Judge 에이전트가 JSON 형태로 점수·이슈 목록을 내며 필터링/개선 루프를 가능하게 하되, 자동 판정의 인간 정렬에는 여전히 트레이드오프가 있음을 함께 보여줍니다.

- **Empirical Impact**: 실험에서는 LLaVA-NeXT-8B, Prismatic-7B, MiniCPM-V-4.5-8B, Qwen3-VL, RoboRefer, RoboBrain2.5 등 다양한 VLM/embodied 모델을 평가했으며, Qwen3-VL-32B가 전체 정확도 83.4%를 기록했지만 공간·절차·이벤트 예측·의도 추론에서 여전히 취약점이 남았습니다. 특히 temporal reasoning과 path planning 같은 하위 범주가 모델 간 격차를 크게 만들고, RoboBrain2.5-8B는 temporal reasoning에서 도약 성능을 보여 ‘도메인 정렬 학습’이 효율적으로 기여할 수 있음을 시사합니다. 동시에 자동 Judge의 인간 판정 일치가 완전하진 않아, ERQA-Plus는 단일 리더보드가 아니라 범주 단위 실패 원인 분석에 더 적합한 진단 도구로 자리매김합니다.



### FLAP: FOV-Constrained Active Perception Planning for Prior-Map-Free 3D Navigation (https://arxiv.org/abs/2606.17630)
Comments:
          18 pages, 19 figures

- **Prior Approaches**: 기존 UAV 경로계획은 미지 공간을 낙관적으로 가정하거나(unknown을 자유로 처리) 보수적으로 다루어(unknown을 장애물로 처리) 안전성 확보를 위해 속도를 줄이거나 정지까지 유도하는 방식이 많았습니다. 또 FOV 기반으로 관측 가능한 구간에만 다니게 하거나, 속도/방향 휴리스틱 및 확률적 충돌위험 평가로 위험을 낮추려는 시도도 있으나, 대부분은 지각 타이밍이 늦거나 수직 FOV·차체 체적 같은 제약을 충분히 결합하지 못했습니다. 정보획득(active perception)을 목적함수에 넣는 방법도 존재하지만, FOV 기하를 “경로 생성 전 단계”에서 처리하거나 단순 2D/수평 yaw 중심으로 제한되어 다양한 3D 기동과 센서 배치에 취약했습니다.

- **Core Contribution**: 이 논문은 FLAP(Field-of-view-constrained Active Perception)으로, 능동 지각을 궤적 최적화에 직접 통합해 안전성과 효율을 동시에 다루는 프레임워크를 제안합니다. 핵심은 센서 좌표계에서 FOV 형상에 정확히 대응하는 지각 제약을 UAV 동역학 모델로부터 도출하고, 그 제약을 미분 가능 페널티로 최적화 문제에 포함해 end-to-end 형태로 운용한다는 점입니다. 또한 velocity-triggered activation과 능동 지각 구간의 시작시간(start-time)까지 최적화해, 장애물의 “늦은 감지”로 인한 충돌 위험을 줄이면서도 관측이 가능할 때는 불필요한 감속을 피하도록 설계했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 수직 FOV/가시성의 블라인드 영역을 포함해, 차체 체적과 안전 여유를 고려한 “관측 가능성”을 실제 최적화 제약으로 안정적으로 표현하는 것입니다. FLAP은 알려진 공간과 조건부로만 통과 가능한 공간을 ESDF(dual ESDF)로 나누고, 알려짐-미지 경계에서 목표를 직접 쓰지 않고 UAV가 안전 여유를 유지한 상태로 관측하도록 visibility point를 평면 법선 방향으로 이동시켜 충돌 가능성을 줄였습니다. 더불어 시야각이 너무 얕거나(시선이 경사인 경우) 감지 범위를 벗어나면 성능이 저하되므로, 최소 시선각 및 센서 유효 거리/각도 조건을 센서 프레임 기준 제약으로 구성하고, 실행 가능성 경계 근처에서도 수치적으로 불안정하지 않도록 각도 임계 조건을 대체 형태로 정식화했습니다.

- **Empirical Impact**: 시뮬레이션과 실제 실험에서 FLAP은 제한된 FOV와 복잡한 3D 구조, 그리고 다양한 센서 종류/마운팅 설정에서도 미지 환경 내 안전성을 유지하면서 경로 효율을 개선함을 보였습니다. 특히 관측이 어려워지는 상황에서는 페널티 활성화가 자연스럽게 감속/안전 중심으로 궤적을 유도하지만, 관측이 가능하면 속도를 유지할 수 있어 보수적 speed limit 전략의 비효율을 줄인 것이 관찰됩니다. 전반적으로, 별도의 비싼 perception-aware front-end 경로 생성 없이도 gradient-based 최적화로 실시간성을 목표로 한 통합 설계를 제공한다는 점에서 로보틱스/UAV 분야의 실사용 장벽을 낮출 수 있는 의미가 큽니다.



### MuseVLA: An Adaptive Multimodal Sensing Vision-Language-Action Model for Robotic Manipulation (https://arxiv.org/abs/2606.17598)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 로봇 모델은 주로 RGB만 입력으로 받아 온도·소리·레이다 반응처럼 RGB로 추정하기 어려운 물리 정보를 놓치는 경우가 많았습니다. 멀티모달 VLA가 일부 등장했지만, 센서마다 전용 인코더/아키텍처를 쓰거나 고정된 센서 구성을 가정해 새로운 센서를 쉽게 확장하기 어렵고, 학습용 다중센서 데이터 수집 비용도 큰 한계로 남아 있었습니다.

- **Core Contribution**: MuseVLA는 센서를 ‘온디맨드 도구’처럼 취급해, 작업 지시와 시각 컨텍스트만으로 어떤 센서를 호출할지와 무엇에 집중할지를 선택하는 adaptive multimodal sensing VLA를 제안합니다. 또한 센서 측정을 카메라 평면에 공간적으로 grounding한 grounded sensor images라는 단일 중간표현으로 변환해, 이 표현을 통해 이질적 센서를 하나의 백본에서 공통 처리하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 매 작업마다 관련 센서를 선택해야 하는 문제와 (2) 서로 다른 센서 신호를 백본이 소화 가능한 공통 표현으로 바꾸는 문제, (3) 희소한 다중센서 데이터로도 일반화를 확보하는 문제입니다. 논문은 learnable sensor tokens로 센서 선택/타깃 설명을 ‘툴 호출’ 인자처럼 생성하고, grounded sensor images로 modality-specific encoder 없이 unified한 입력을 만들며, RGB 비디오 데이터에 VLM+SAM3 기반 합성 파이프라인을 적용해 unseen sensor-guided task에도 대응하도록 학습합니다.

- **Empirical Impact**: 실세계 덱스터러스 핸드 로봇에서 온도 유도 pick-and-place, 오디오 기반 물체 탐색, mmWave 레이다 기반 숨은 물체 회수 같은 과제에 대해 평균 80.6% 성공률을 보고했으며, RGB-only 및 raw 센서 heatmap 기반 VLA 대비 유의미하게 향상됐습니다. 또한 합성 데이터로 사전학습한 모델은 unseen 작업에서 평균 66.7% 성공률로 zero-shot 일반화 성능이 강함을 보여, 다중센서 로봇 조작 분야에서 ‘센서 확장성과 데이터 효율’의 실용성을 높였다는 의미가 큽니다.



### RICH-SLAM: Radar SLAM with Incremental and Continuous Hilbert Mapping (https://arxiv.org/abs/2606.17534)
Comments:
          12 figures

- **Prior Approaches**: 레이더 SLAM은 LiDAR/비전 대비 잡음이 크고 측정이 희소해, 기존 방식은 주로 odometry(자세추정) 안정화에 초점이 맞춰져 있었다. 또한 occupancy grid map(OGM)은 격자 독립 가정과 고정 해상도로 인해 빔 갭/가림 영역의 불확실성이 커지고 레이더처럼 희소한 업데이트에서 연속적 지도를 만들기 어렵다. Gaussian process occupancy map(GPOM)은 연속성은 주지만 업데이트·예측이 계산적으로 무거워 실시간 운용이 난감하다.

- **Core Contribution**: RICH-SLAM은 SoC 레이더의 희소·노이즈 관측을 사용해 연속적이고 불확실성을 포함하는 occupancy 지도를 구성하는 레이더 SLAM 프레임워크를 제안한다. Rao-Blackwellized particle filter 기반 back end에, Hilbert-space reduced-rank Gaussian process 매핑을 결합해 map을 ‘연속 함수’로 유지한다. 또한 맵 파라미터의 전체 posterior 분포를 활용하는 posterior-aware particle weighting과, endpoint 허용오차를 고려한 likelihood 설계로 약한 레이더 신호에도 자세 제약을 강화한다.

- **Technical Challenges**: 가장 큰 난제는 레이더 관측이 빔 형태로 희소하고 다중모달 자세 posterior를 만들기 때문에, 단순 pose-graph 최적화나 격자 기반 지도 업데이트로는 일관된 연속 지도를 얻기 어렵다는 점이다. RICH-SLAM은 Hilbert-GP로 GPOM의 계산 부담을 줄이면서도 연속 점유장을 표현하고, Kalman 필터 업데이트로 맵 가중치의 Gaussian posterior를 증분 갱신한다. 더불어 particle weight를 ‘맵의 평균’이 아니라 ‘맵의 불확실성까지 포함한 posterior’를 통해 계산하고, 회전 오차에 민감한 점유장 정합을 endpoint-tolerant likelihood로 완화한다.

- **Empirical Impact**: 자체 수집 데이터와 공개 ColoRadar 데이터에서 RICH-SLAM은 희소 레이더 측정만으로 연속 occupancy map을 구성하며, 불확실성을 고려한 mobile robot planning까지 지원함을 보였다. 특히 H-SLAM 계열의 Hilbert map 기반 접근 대비, posterior를 유지하는 Hilbert-space Gaussian process 설계가 희소·노이즈 환경에서의 robust likelihood 평가에 유리함을 실험적으로 확인한 점이 의미 있다. 저자들은 basis size, length scale, 레이더 범위, particle 수 같은 설정을 고정하고 신호/측정 분산 및 Hilbert 도메인만 환경별로 조정해 재현성 있는 비교를 수행했다.



### GASE: Gaussian Splatting-Based Automated System for Reconstructing Embodied-Simulation Environments (https://arxiv.org/abs/2606.17520)
- **Prior Approaches**: 기존 연구는 생성 기반(generative) 방식이 빠르긴 하지만 sim-to-real 간극이 커지는 경향이 있고, 재구성 기반(reconstruction-based) 방식은 간극을 줄이려 하지만 전반적으로 자동화와 효율이 부족했다. 특히 monocular 중심 파이프라인은 넓은 장면 스캔에 시간이 많이 들고, 3D Gaussian 공간에서의 전경/배경 분리는 경계 불명확과 아티팩트로 인해 성능이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 3D Gaussian Splatting(3DGS) 기반으로 시뮬레이션 장면을 대규모로 자동 생성하는 시스템 GASE를 제안한다. 핵심은 파노라마 카메라 배열의 다중 뷰를 한 번의 스캔으로 처리하고, 전경/배경 분리를 2D 이미지 도메인에서 먼저 수행한 뒤 분리된 자산을 각각 생성·재구성해 물리 시뮬레이터에 바로 넣을 수 있게 만든 점이다.

- **Technical Challenges**: 기여를 위해 (1) 다중 frame stream에서 동일 객체의 마스크 정체성을 일관되게 유지하는 문제와 (2) 3D Gaussian의 표현 특성 때문에 생기는 경계/구조 아티팩트를 피하는 문제가 까다롭다. GASE는 카메라 pose와 깊이 정보를 이용해 2D 마스크를 프레임 전반으로 정합하고, SAM3로 초기 마스크를 만들고 SAM2로 비디오 전파를 보강한 다음, LAMA 기반 inpainting을 2D에서 수행해 안정적인 분리 품질을 확보한다.

- **Empirical Impact**: 평가에서 GASE는 3D Gaussian을 직접 다루는 분리 방식 대비 세그멘테이션 정확도를 10% 이상 개선하고, inpainting 품질에서도 state-of-the-art를 보였다. 또한 로봇 실험에서 sim-to-real 격차를 10% 미만으로 유지하며 조작 및 내비게이션 작업 모두에서 높은 성과를 보였고, 코드 공개도 예고되어 실무적 채택 가능성이 크다.



### MagicSim: A Unified Infrastructure for Executable Embodied Interaction (https://arxiv.org/abs/2606.17511)
- **Prior Approaches**: 기존 로봇 시뮬레이션은 주로 렌더링/컨트롤 테스트베드이거나, 특정 고정 작업 환경처럼 분리돼 활용되는 경우가 많습니다. 장기·상호작용 과업은 평가를 위해 ‘magic’ 액션으로 우회하거나, 학습/수집용 환경과 계획/상태가 이어지지 않아 같은 에피소드 재현·주석·검증이 어렵다는 한계가 있었습니다. 또한 플래닝을 외부 오프라인 전처리로 두면, 실패 재현이나 플래닝-루프 안에서의 실행 기록을 얻기 힘들었습니다.

- **Core Contribution**: MagicSim은 에피소드를 단위로 삼는 embodied interaction infrastructure로, 동일한 결정적(deterministic) 배치 런타임과 단일 MDP 위에서 world 구성, 실행, 평가, 데이터 수집, 에이전트 상호작용을 한 번에 잇습니다. YAML-first 명세로 콘텐츠·배치·행동·에이전트 노출을 분리해, 강체부터 유체/연성, 다양한 로봇 embodiment까지 폭넓은 실행 가능한 월드를 reset-and-step 루프에서 생성합니다. 또한 고수준 커맨드를 simulator 내부 상태 편집이 아닌 controller/atomicskills/플래너 프리미티브를 거쳐 로봇 액션으로 접지(grounding)합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 한 런타임에서 이질적인 물리(강체·연성·유체·지형·아바타·센서)를 공존시키는 월드 계약, (2) 병렬 대규모에서 재현 가능한 결정성(리셋 순서, snapshot, reset_to), (3) 단일 action space로 줄이기 어려운 multi-embodiment 실행, (4) planner-in-the-loop를 배치 시뮬레이션이 멈추지 않게 비동기 처리하는 구조, (5) 어디서/어떻게 잡았는지 같은 상호작용 인과구조를 보존하는 annotation-rich 데이터 생성입니다. MagicSim은 batched runtime에서 env별 독립 라이프사이클과 비동기 플래너 마이크배치 마진(microbatch) 해결, 그리고 성공 게이팅 기반의 에피소드 단위 저장을 조합해 이를 해결합니다.

- **Empirical Impact**: 논문은 MagicSim이 단일 태스크 정의로 RL 벤치마크 평가, autocollect용 자동 궤적 수집, VLM/에이전트용 상호작용 인터페이스를 동시에 지원해 연구 파이프라인을 통합할 수 있음을 강조합니다. 특히 성공한 롤아웃만 구조화된 멀티모달 트래젝토리로 저장해 언어 감독, 시각/기하 표현, 스킬·플래너·액션의 정렬을 제공함으로써 데이터 품질과 재사용성을 높입니다. 결과적으로 로봇 학습·데이터 생성·에이전트 구동을 하나의 재현 가능한 planner-in-the-loop 실행 기반으로 묶는 방향의 실용적 인프라로 의미가 큽니다.



### When Robots Sleep: Offline Skill Consolidation for Shared-Policy Robot Learning (https://arxiv.org/abs/2606.17493)
- **Prior Approaches**: 기존 continual learning은 EWC·SI·MAS 같은 정규화, GEM·A-GEM 같은 gradient 제약, DER·CLEAR 같은 replay, LwF·Progress & Compress 같은 distillation으로 “개별 태스크 성능 유지”를 주로 겨냥했습니다. 하지만 순차 로봇 스킬 학습에선 이전 환경/보상/데모/트레이젝터리와 함께 오래된 task loss까지 사라져, cross-skill(관련 스킬 간) 신뢰도를 직접 보호하기 어렵습니다. 모듈형 접근(헤드·routing·어댑터·experts)은 조합을 추론 단계에서 처리하지만, 단일 공유 정책을 유지한다는 목표와는 거리가 있습니다.

- **Core Contribution**: 이 논문은 스킬이 개별적으로는 성공해도, 공유 정책 내부에서 관련 스킬들이 함께 안정적으로 작동하지 못하는 실패 모드인 skill-coupling collapse를 정의합니다. 그 위에 wake-sleep 프레임워크 Sleeping Robots를 제안해, 각 새 스킬은 wake에서 학습하되 과거 데이터 없이도 sleep에서 한 개의 공유 정책을 오프라인으로 “통합 정리(consolidation)”합니다. 핵심 아이디어는 compact frozen skill memories(강화학습은 frozen critic+unordered state buffer, 모방학습은 frozen actor+unordered observation buffer)로 이전 스킬의 역할을 보존하고, 단일 컨트롤러 구조를 유지하는 것입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 “이전 task loss/트레이젝터리 접근 부재” 상황에서 여러 스킬의 학습 신호를 안정적으로 결합해, 한 스킬만 지배해 성능·신뢰도가 무너지는 것을 막는 점입니다. 논문은 frozen memory들이 만드는 differentiable surrogate objectives의 기울기를 Nash bargaining으로 결합해(gradient scale 정규화 포함) 합의(agreement)에 기반한 sleep 업데이트를 설계하고, 필요한 경우 adaptive sleep anchoring으로 직전 통합 체크포인트에 보수적으로 끌어당깁니다. 또한 feedforward RL actor에 local excitability를 도입해 sleep 동안 actor가 과도하게 요동치지 않도록 안정화를 돕습니다.

- **Empirical Impact**: Meta-World MT5(MT5)에서 Sleeping Robots는 non-oracle 최고 기준선 대비 평균 성공률을 64% 향상시키고, pairwise reliability는 2.0배로 크게 끌어올렸습니다. 특히 평균 성공만 보면 일부 기법이 남는 것처럼 보여도, PRS(쌍 스킬 신뢰도) 하락으로 skill-coupling collapse가 드러나는데, Sleeping Robots는 이를 가장 잘 완화하며 trajectory-structure preservation도 우수했습니다. SurgicAI에서는 vision-conditioned surgical imitation에서도 offline sleep consolidation이 평균 성공과 backward transfer를 개선하면서, task-specific head/routing/trajectory replay 없이도 경쟁력 있는 pairwise reliability를 유지했습니다.



### Embodiment Shapes Rolling Behavior in a Multimodal Infant Mod (https://arxiv.org/abs/2606.17456)
Comments:
          7 pages, 7 figures. Accepted at the 2026 IEEE ICDL Conference. Cite as: L. Philipp, F. M. López, and J. Triesch, "Embodiment Shapes Rolling Behavior in a Multimodal Infant Model", in 2026 IEEE International Conference on Development and Learning (ICDL). IEEE, 2026, pp. 1-7

- **Prior Approaches**: 기존 유아 롤오버 연구는 주로 관찰 데이터 분류(움직임 패턴/지면 압력 등)와 근전도(EMG) 같은 실험에 의존하지만, 팔·머리 등 전신 근활성/감각 정보를 충분히 분리·측정하기 어렵다는 한계가 있었다. 시뮬레이션 기반 접근도 있었으나, 유아 전신 발달(embodiment)의 역할을 학습 과정과 연결해 재현한 computational developmental science 사례는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 가상 유아 신체(embodied infant embodiment) MIMo를 구축하고, 강화학습을 통해 supine-to-prone 롤오버를 자율 학습시키는 computational study를 제시한다. 특히 신체 형태와 근력(발달 시점)을 1, 3, 6, 9개월 조건으로 바꾸면서, 동일한 학습 설정에서 발달적 경향이 어떻게 행동으로 나타나는지 정량·정성으로 연결한다.

- **Technical Challenges**: 핵심 과제는 (1) 전신 물리 접촉·중력 하에서 성공 보상을 안정적으로 학습시키고, (2) 롤오버의 다양한 coordination 패턴을 실험심리학 지표로 재해석하며, (3) “신체 크기 vs 근력”을 분리해 원인을 밝히는 것이다. 연구진은 PPO에 dense reward와 PBRS 기반 sparse reward를 결합하고, torso·pelvis 자세 전이 지표와 limb 속도/동기·리드/팔로우 분류, 제어 신호(근육 대신 actuation) 분석으로 행동을 비교했으며, cross-embodiment 평가로 근력의 우위를 드러냈다.

- **Empirical Impact**: 결과적으로 나이가 든(9개월에 가까운) 신체 조건일수록 성공률이 높아지고 실행이 빨라졌으며, 이를 cross-embodiment로 분리한 결과는 “근력 발달”이 롤오버 출현에 핵심이라는 결론으로 이어진다. 또한 MIMo는 인위적으로 다양성에 보상하지 않았는데도 실제 유아에서 보고된 여러 롤오버 전략 분포와 낮은 Jensen-Shannon divergence(0.3 이하) 수준의 유사성을 보였고, 단일 모델에서도 초기 조건 변화에 따라 패턴이 달라지는 적응성을 확인했다. 전반적으로 본 연구는 embodied computational model이 유아 전신 감각운동 발달을 ‘신체 제약의 변화’ 관점에서 실험적으로 탐구할 수 있음을 보여주며, 향후 시야·촉각, 감각지연, 성장 동역학까지 확장할 동기를 제공한다.



### Continual Online Personalization of Exoskeleton Control via Manifold-Aware Experience Replay (https://arxiv.org/abs/2606.17455)
- **Prior Approaches**: 기존 외골격 제어 개인화는 수동 파라미터 튜닝이나 각 대상별 장시간 최적화에 의존하는 경우가 많았습니다. ML 기반 제어는 다양한 보행 상태를 추정하지만, 학습 분포 밖(impairment·새 과제)에서는 성능 저하 및 재학습 요구가 생깁니다. 온라인 적응(OA) 연구도 진행됐으나, 주로 단일 task 중심이라 여러 locomotor context를 동시/연속으로 유지하는 문제(continual learning의 catastrophic forgetting)는 충분히 다뤄지지 않았습니다.

- **Core Contribution**: 이 논문은 외골격 gait phase 추정을 온라인으로 개인화하되, 다양한 보행 과제 전환 중에도 개인화된 표현을 유지하기 위해 manifold-aware experience replay를 제안합니다. 핵심은 gait 데이터의 잠재 공간(보행 manifold)을 활용해 이전에 경험한 context를 replay buffer에서 골라 재학습하되, 명시적 task labeling 없이도 목표 replay bin을 선택하게 한 점입니다. 그 결과, 이전 과제에서 학습된 개인화 보조 타이밍을 계속 유지하면서 실시간 제어를 수행하도록 설계됐습니다.

- **Technical Challenges**: OA 과정에서 연속적으로 들어오는 사용자 상태 데이터가 기존 context를 덮어쓰는 catastrophic forgetting을 어떻게 막을지가 난제였습니다. 이를 위해 (1) replay를 수행하되 task 중복을 줄이기 위해 PCA 기반 gait manifold로 binning하고, (2) RMSE 범위로 유효한 bin만 유지하며, (3) update latency와 정확도를 함께 고려해 최대 4개 replay bin만 사용하도록 조절했습니다. 또한 TCN의 상위 feature 추출 블록은 고정하고 마지막 linear layer만 fine-tuning해 전이 특징을 보존하면서 빠른 적응이 가능하도록 했습니다.

- **Empirical Impact**: emulated hemiplegic gait(속도·경사 전이, forgetting 시나리오)에서 기준선(replay 없는 방법) 대비 torque RMSE를 40% 줄이고 gait phase 추정 RMSE는 60% 개선했습니다. task 전환에서 기준선은 catastrophic forgetting이 뚜렷했지만, 제안 방법은 대부분의 평가 쌍에서 오류를 유의하게 낮추며 안정성을 보였습니다. 온라인 업데이트 지연은 약 1.5초 수준으로, 실사용에 가까운 빠른 개인화 반응성을 보여 주었다는 점에서 웨어러블 로보틱스 커뮤니티에 의미가 큽니다.



### AnnotateAnything: Automatic Annotation of 3D Assets for Robot Manipulation (https://arxiv.org/abs/2606.17446)
- **Prior Approaches**: 기존 로봇 데이터 자동화는 (1) 수동 어노테이션, (2) RL 기반 자동화로 크게 나뉘는데, 전자는 비용과 확장성이 문제였고 후자는 reward engineering과 학습·연산 부담이 컸습니다. 조작(Manipulation) 어노테이션 자동화도 grasp, contact, affordance 같은 단일 레이블에 집중하는 경우가 많아, 다양한 기술·물체·로봇 embodiment를 포괄하기 어렵다는 한계가 있었습니다. 즉 ‘수동 3D 자산 → 실제로 실행 가능한 조작 라벨’로의 자동 변환이 병목이었습니다.

- **Core Contribution**: 이 논문은 수동(passive) 3D 자산을 조작 실행에 바로 쓸 수 있는 라벨을 갖춘 자산으로 변환하는 자동 어노테이션 프레임워크 AnnotateAnything을 제안합니다. 핵심은 VLM(visual-language model)로 사람의 상호작용 priors를 추출·3D에 접지(grounding)한 뒤, 물리 기반 최적화와 대규모 병렬 시뮬레이션 검증으로 ‘실행 가능한’ action annotation(그립 포즈, dexterous contact, 관절 waypoint, 삽입/걸기 방향, 내비게이션 타깃 등)을 생성하는 것입니다. 또한 단일 해답이 아니라 다양한 실행 후보를 one-to-many 형태의 후보 뱅크로 보존합니다.

- **Technical Challenges**: 기여를 가능하게 한 기술적 난관은 네 가지로 요약됩니다: (C1) 사람의 상호작용 직관을 자동 추출해 자연스럽고 안전한 행동으로 연결, (C2) 이를 자산별 기하/운동 제약에 맞춰 실행 라벨로 변환, (C3) 가능한 해답들을 다양하게 유지, (C4) 자산·카테고리·작업을 per-asset 수동 설계 없이 스케일. 저자들은 언어 단계(자산/방(room) 레벨 기술, keypoint·part·affordance 큐 단서)에서 priors를 만들고, 물리 단계에서 candidate generation–trajectory generation–최적화–물리 검증–물리 aware augmentation까지 CUDA 가속 병렬 파이프라인으로 수행해 C1~C4를 동시에 맞췄다고 주장합니다.

- **Empirical Impact**: 실험에서 AnnotateAnything은 99개 소스 계열의 17,005개 자산을 처리해 181개(문맥상 18개로 보이는 ‘atomic skills’ 그룹 포함) 수준의 기술에 대해 총 1억 단위의 physics-validated action annotation을 생성했으며, 평균적으로 asset–skill 쌍당 수천 개 후보를 생성해 물리 검증 통과 후 최종 뱅크로 보존합니다. 시각-언어 번들 품질과 실행 성공/롤아웃 수집 효율에서 기존 어노테이션 파이프라인 및 VL-only/heuristic 변형 대비 우수한 성능을 보였고, downstream로 affordance/keypoint 감지, robotic VQA, 3D VLM instruction finetuning 같은 작업까지 활용 가능함을 보여줍니다. 결론적으로 ‘자산 수집’이 아니라 ‘조작 실행 라벨 변환’을 자동화해 시뮬 기반 로봇 데이터 수집의 실용성을 크게 끌어올린다는 점에서 의미가 큽니다.



### DexLink Hand: A Compact, Affordable, 16-DOF Linkage-Driven Hand with Human-Like Dexterity (https://arxiv.org/abs/2606.17418)
- **Prior Approaches**: 기존 다지(多指) 로봇손은 kinematic workspace를 늘리기 위해 DOF를 크게 확장하는 방향이 주류였고, Shadow Dexterous Hand나 DLR/HIT Hand II처럼 fully actuated 구조와 촘촘한 센싱·전송으로 높은 손재주를 노려왔다. 하지만 모터·전송·센서가 DOF마다 늘어나면서 크기, 비용, 구조 복잡도가 함께 커져 사람 손 크기 수준으로 축소하기 어렵다는 한계가 반복적으로 지적된다. underactuated 또는 synergy-driven 접근은 구동 자유도를 줄여 경량·저비용을 달성하지만, 독립 관절 제어와 재구성 능력이 제한되어 in-hand 조작·일반화된 물체 상호작용·도구 사용 같은 고난도 작업에서 제약이 생긴다.

- **Core Contribution**: 이 논문은 “작고 싼데도 사람 손 같은 손재주”라는 목표에 맞춰, 20개 관절을 16개 독립 구동기로 구동하는 링크 기반(linkage-driven) 인체형(anthropomorphic) 로봇손을 제안한다. 사람 손 크기급 구조 내부에 구동, 감지, 전송을 통합해 프로토타입 중량 320g, 총 비용 400달러 미만을 달성했으며, 손가락은 PIP·DIP 결합을 모사한 crossed four-bar, 손바닥의 MCP는 공간(space)·평면(planar) 링크를 결합해 2-DOF를 더 잘 분리(decoupling)하도록 설계했다. 엄지 역시 CMC 관절의 재구성 가능한 관계를 보존하는 biomimetic 구조로 다양한 opposition과 핀치/그립 재구성을 지원한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 링크 메커니즘으로 DOF를 유지하면서도 관절 간 간섭 없는 decoupled 움직임을 만들고, (2) 짧은 팜 공간 안에 복잡한 전송을 밀도 높게 집적하며, (3) 하이브리드 메커니즘에 대응하는 제어를 저비용으로 구현하는 데 있다. 저자들은 PIP·DIP에는 추가 링크를 최소화하는 crossed four-bar로 언더액추에이트드 결합을 구현하고, MCP에는 구면(spherical) four-bar로 Flex/Ext와 Abd/Add 축의 독립성을 크게 높였다. 제어는 Hall 센서를 단 DC 브러시 모터와 PWM 서보를 ESP32 기반 분산 임베디드 컨트롤러로 통합해, 모든 브러시 모터를 동시 closed-loop 포지션 제어(PID)로 동기화하도록 구성했다.

- **Empirical Impact**: 실험에서는 Kapandji score 최대치를 달성했고, Feix의 33가지 grasp type을 모두 재현했으며, 일상 물체와 도구에 대해 안정적인 그립과 손안 조작을 수행하는 것으로 보고된다. 특히 고난도 작업 지표(다양한 grasp 유형·도구 사용)까지 포괄하면서도 초경량·저비용으로 구현했다는 점에서, 사람 중심 환경에서의 텔레오퍼레이션과 robot learning용 “현실적으로 접근 가능한” 손 플랫폼을 제공한다는 의미가 크다. 결과적으로 고DOF 손재주와 소형·저비용을 동시에 만족시키는 링크 기반 기계설계/제어 통합 전략의 실증 가치가 확인됐다.



### Where Should Action Generation Begin? A Learnable Source Prior for Generative Robot Policies (https://arxiv.org/abs/2606.17408)
- **Prior Approaches**: 최근 생성형 로봇 정책은 conditional sampling 형태로 diffusion 기반 또는 flow matching 기반 모델을 사용하지만, 대부분 action 생성의 시작점을 observation-independent standard Gaussian N(0,I)로 고정합니다. 이 때문에 generator가 “정보 없는 잡음”을 task-relevant action으로 옮기는 데 일부 계산 예산을 소모하고, source의 불확실성 설계는 상대적으로 덜 다뤄졌습니다. 또한 관측 기반 초기화를 쓰는 A2A, VITA, BridgePolicy 계열도 대체로 deterministic point estimate 또는 제한된 초기화로 uncertainty를 명시적으로 모델링하지 않는 경향이 있습니다.

- **Core Contribution**: 본 논문은 action 생성 시작점을 관측에 조건된 학습 가능한 source prior로 바꾸는 Learnable source Prior, LeaP를 제안합니다. LeaP는 proprioception(자세/관절 등 몸 상태) 특징만으로 state-adaptive 대각 가우시안의 평균과 분산을 예측해, observation-informed이면서도 stochastic한 초기 샘플을 제공합니다. 중요한 점은 generator 아키텍처와 inference solver, 생성 동역학은 그대로 두고 prior만 플러그인 형태로 교체한다는 것입니다.

- **Technical Challenges**: 핵심 기술 과제는 “초기 분포를 알기”와 “불확실성을 실제로 유용하게” 만드는 학습 신호를 동시에 구성하는 것입니다. LeaP는 prior 학습에 NLL(음의 log-likelihood)로 source 분포의 일치도를 맞추고, CLIP-style symmetric contrastive alignment로 샘플-타깃 대응성을 강화하며, flow loss를 통해 reparameterization 경로로 gradient을 generator와 함께 역전파합니다. 또한 diagonal Gaussian을 유지하면서도 mean과 state-adaptive variance를 함께 학습해, 표현력보다 로컬한 확률 질량 배치가 성능에 기여하도록 설계했습니다.

- **Empirical Impact**: RoboTwin 2.0의 15개 manipulation task에서 LeaP는 평균 success rate 81.6%를 달성하며 4개 대표 baseline 대비 6.5~25.5%p 향상됐습니다. 특히 NoPrior 대비 25.5%p 격차는 “표준 Gaussian 대체” 자체를 넘어, 학습 가능한 distributional prior가 성능의 원천임을 보여줍니다. 더 나아가 flow-matching과 diffusion-bridge 모두에서 일관되게 개선되며, 파라미터 수가 적고 더 빠르게 수렴하고, real-world Franka 배치에서도 최상 성능을 기록해 현장 적용 가능성을 강화했습니다.



### Damage Adaptation in Seconds for Architected Materials (https://arxiv.org/abs/2606.17394)
Comments:
          Proceedings of Robotics: Science and Systems

- **Prior Approaches**: 기존 soft robot 보정·적응 연구는 손상 이후 동역학을 ‘명목(비손상/초기 길들이기)’ 가정에 맞춰 다루거나, 좁게 정의된 손상 범위 안에서만 동작하도록 설계되는 경우가 많습니다. 또한 self-modeling 계열이라도 손상 상태를 연속 파라미터로 다루면 경우의 수가 커져 표본 효율이 떨어지고, 카메라/센서 기반 learned proprioception은 손상·수리까지 내재적으로 다루지 못하는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 손상과 in-situ repair를 동일한 “메타 상태” 변화로 보고, Latent Ensemble Adaptive Proprioception(LEAP)로 연속 업데이트 없이 빠르게 적응하는 프레임워크를 제안합니다. HSAs 같은 architected materials의 격자/유닛셀 구조를 이용해 손상을 저차원·이산 좌표로 인코딩하고, latent 손상 표현과 단순한 ensemble 가중치 회귀만으로 실시간(수 초) 적응을 달성합니다.

- **Technical Challenges**: 핵심 난제는 (1) 손상 종류가 사전에 주어지지 않는 out-of-distribution 손상/수리를 빠르게 추정하고, (2) 계산·데이터 제약 속에서 proprioceptive force/torque를 안정적으로 예측하는 것입니다. LEAP은 손상 좌표를 sparse 이진 벡터로 만든 뒤 variational latent model로 부드러운 잠재 공간을 학습하고, k-means로 latent centroid ensemble을 미리 구성한 다음 런타임에는 least-squares로 ensemble 가중치만 추정해 강건하게 예측합니다.

- **Empirical Impact**: 실험에서 단일 HSA의 경우 손상된 상태에서 proprioceptive force/torque 예측 오차가 평균 87% 감소했으며, 6-DoF HSA 소프트 손목에서도 손상·수리 상황에 대해 평균적으로 유의미한 오차 감소를 보입니다(대략 힘 1.5N, 토크 0.074Nm 수준의 mean error). 또한 Franka Panda 로봇과 결합한 tracing 과제에서 LEAP 적응 후에는 손상 이후에도 접촉 유지율 85%를 재현해 작업 실패를 막았고, 적응 없는 모델은 경로를 끝까지 추적하지 못했다고 보고했습니다.



### Agent Utilities over Generalized Voronoi Regions and their Gradients (https://arxiv.org/abs/2606.17388)
Comments:
          Under review at IEEE Control Systems Letters (L-CSS)

- **Prior Approaches**: 기존 Voronoi 계열은 거리 기반으로 공간을 분할해 커버리지·센싱·태스크 배분을 다뤘지만, 에이전트의 동역학과 제어 비용은 상대적으로 분리돼 있었다. 동역학·제어를 포함한 비용 기반 분할(예: 최소비용 partition)과 quadratic 경계, 그리고 유틸리티/그래디언트까지의 연결은 일부 연구에서 다뤄졌으나, 비용-유도 분할의 일반 형태(CIV)까지 확장된 유틸리티 그래디언트 계산은 제한적이었다. 특히 utility gradient 계산은 고전 Voronoi(특정 경계 형태)에서의 결과가 중심이었고, 일반적인 곡면/비선형 경계에 대한 효율적 경계적분식은 부족했다.

- **Core Contribution**: 이 논문은 에이전트의 상태 공간과 분할되는 공간을 분리하는 Cost-Induced Voronoi(CIV) 개념을 제안해, 임의의 비용으로 유도되는 분할을 통일적으로 정의한다. LQR 기반 비용을 쓰는 LQR-CIV는 경계가 LQR 비용 구조를 반영해 quadratic 곡면이 된다는 점을 보여준다. 또한 에이전트 유틸리티를 “유틸리티 밀도(이벤트 발생 확률 등)를 CIV 영역에 적분한 값”으로 정의하고, 유틸리티 그래디언트를 경계의 변화까지 포함해 도출한다.

- **Technical Challenges**: 핵심 난제는 에이전트 상태가 바뀌면 CIV 경계가 같이 이동하는데, 영역 적분으로 정의된 유틸리티의 그래디언트를 어떻게 효율적으로 계산하느냐였다. 이를 위해 유체역학의 Reynolds Transport Theorem(RTT)을 적용해, 영역 적분을 피하고 CIV 경계에서의 line/contour integral 형태로 그래디언트를 계산하는 식을 구성한다. 팀(상대 팀 포함) 설정에서는 동일 팀 내부 경계 기여가 상쇄되고, 팀 간 경계만 남도록 정리해 dynamics-aware team utility의 그래디언트도 확장한다.

- **Empirical Impact**: 수치 실험에서 contour-integral(CI) 방식은 finite-difference(FD) 기준선 대비 유틸리티 그래디언트의 상대오차를 “몇 퍼센트 수준”으로 맞추면서 계산시간은 약 25배 줄였다. 또한 축구 예시처럼 유틸리티 밀도(패스 수신 같은 유리한 이벤트의 확률)를 CIV 영역에 적분해, 그래디언트가 어떤 방향으로 확률을 개선하는지 직관적으로 보여준다. 결과적으로 CIV+RTT 기반의 경계적분 그래디언트는 다에이전트 경쟁/협력에서 효율적인 의사결정 최적화에 실용적인 속도 이점을 제공한다.



### EgoInfinity: A Web-Scale 4D Hand-Object Interaction Data Engine for Any-View Robot Retargeting and Video-to-Action Robot Learning (https://arxiv.org/abs/2606.17385)
Comments:
          24 pages. Project page: this https URL

- **Prior Approaches**: 기존 연구는 egocentric/착용형·모션캡처 기반 데이터(Ego4D, EgoDex 등)가 로봇 학습에 유리하지만, 수집 비용과 환경/참여자 편향으로 확장이 어렵다는 한계가 컸습니다. 한편 실행 가능한 로봇 데이터셋(DROID, Open X-Embodiment 등)은 하드웨어·과업 설계 제약으로 다양성이 제한됩니다. 인터넷 영상은 규모는 크지만 metric 3D 기하, 6-DoF 물체 상태, 접촉 정보가 없어 영상-행동 정렬이 로봇 실행 단계에서 어긋나는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 임의 RGB 인터넷 영상을 웹스케일로 처리해 로봇이 쓸 수 있는 4D hand-object 표현을 자동 생성하는 데이터 엔진 EgoInfinity를 제안합니다. 사람의 마커/인간-in-the-loop 주석 없이, 인식-분할-재구성-상호작용 인지 정제-리타겟팅까지 모듈형 파이프라인을 통해 공통 좌표계의 metric 표현을 만들고, 에이전트 독립적인 4D 상태(손 궤적, 6-DoF 물체 자세, 접촉 관련 상태)를 제공합니다. 또한 복원된 손 모션을 로봇 형태에 맞는 “functional” 관점의 모션 리타겟터로 변환해, 부분 관측/임의 시점 영상에서도 로봇 실행 궤적을 컴파일할 수 있게 합니다.

- **Technical Challenges**: 핵심 난제는 모듈별 출력이 서로 스케일·좌표계·일관성이 달라 손-물체 연계를 깨뜨리고 접촉 물리성을 훼손하는 점이었습니다. EgoInfinity는 MoGe-2/Flow3r/GeoCalib 기반의 metric geometry로 손과 물체를 공유 카메라 프레임에 정렬하고, SAM 기반 분할·추적과 MEMFOF 광류/손 키포인트로 상호작용 상태(static/grasped/moving)를 판별해 물체 궤적을 접촉 인지로 보정합니다. 리타겟팅 측면에서는 정확한 인간 신체 모사 대신, SE(3) 대응 구조(Vector Neuron)로 로봇별 root 변환을 예측하고 flow-matching으로 가능한 root 후보를 샘플링·IK 기반 선택해 실행 가능성을 확보했습니다.

- **Empirical Impact**: 저자들은 지각 정밀도, 기구학적 feasibility, 접촉 일관성, cross-embodiment 일반화, 그리고 실제 로봇 스킬(그립, cutting, wiping, pouring) 학습/실행까지 폭넓게 검증하며 스케일 가능한 영상-행동 연결고리를 보여줍니다. 또한 Action100M 일부를 포함해 106편의 4D 조작 영상(다운로드 가능)을 생성·서빙하고, 온라인 브라우저 서버로 중간 산출물(손 메쉬, 6-DoF 궤적, 포인트클라우드 등)을 시각화해 실패 진단과 큐레이션을 돕습니다. 결과적으로 “모션 데이터”를 넘어서 metric 4D·접촉 관련 상태를 갖춘 로봇 실행 입력으로 바꾸는 인프라를 제공해 오픈월드 로봇 러닝에 실질적 동력을 제공합니다.



### Contactless Respiratory Monitoring on Heterogeneous Mobile Robots: A Multimodal Edge-Computing Framework (https://arxiv.org/abs/2606.17376)
Comments:
          8 pages, 6 figures. To appear in Proceedings of the 8th International Workshop on IoT Applications and Industry 5.0 (IoTI5 2026), co-located with IEEE DCOSS-IoT 2026, Reykjavik, Iceland, June 2026

- **Prior Approaches**: 기존 비접촉 RR 모니터링은 영상에서 호흡 유발 미세 움직임을 optical flow, motion magnification, band-pass filtering 같은 신호처리로 추정하거나, RGB·NIR·thermal 멀티모달을 딥러닝으로 결합해 성능을 개선해 왔다. 하지만 조명 변화, 자세/거리 변화, 가림, 그리고 로봇 플랫폼·엣지컴퓨팅 제약 때문에 현장 배치에서 일관된 성능을 내기 어렵다는 한계가 반복됐다. 또한 모달리티(예: RGB vs thermal)별 품질이 달라지는 문제를 로봇의 센서 이종성까지 포함해 체계적으로 다룬 연구는 부족했다.

- **Core Contribution**: 이 논문은 이종 모바일 로봇에서 per-platform 알고리즘 튜닝 없이 RR을 산출하는 modality-adaptive 비접촉 모니터링 프레임워크를 제안한다. RGB·thermal·NIR·low-light 카메라를 밝기 조건에 따라 자동 선택하고, 어깨·엉덩이 keypoint로 흉부 ROI를 자세 변화에 강인하게 정렬한다. 마지막으로 window 단위로 signal-quality-index(SQI)를 계산해 신뢰 가능한 구간만 남기고, 하모닉 오차까지 보정해 최종 RR을 집계한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 조명/센서 성능이 시간·환경에 따라 급변하는 상황에서 안정적인 RR 신호를 뽑아내는 것과 (2) 자세 변화로 ROI 정렬이 깨질 때도 추정이 무너지지 않게 하는 것이다. 연구진은 초기 프레임 평균 밝기 기반으로 모달리티를 adaptive로 고르고, YOLOv11x-pose keypoint로 흉부 ROI를 geometry-adaptive로 구성해 자세 견고성을 확보했다. 더 나아가 SQI에서 spectral flatness, peak prominence, autocorrelation confidence를 결합하고 모달리티별 임계값으로 hard gating하여 잡음/부정합 구간을 제거하며, FFT·autocorrelation 추정의 하모닉 불일치와 이웃 window 간 일관성으로 오류를 줄였다.

- **Empirical Impact**: 3개 로봇 플랫폼(보행 형태와 엣지 컴퓨팅 구성이 다른 시스템)에서 조명(실내/야외, 명/암), 거리(2~8m), 자세(서기/앉기/눕기)별로 평가했으며, 플랫폼 간 알고리즘 재튜닝 없이 일반화되는 결과를 확인했다. 운영 한계는 모달리티별로 명확해졌는데 RGB는 최대 8m, NIR은 약 6m, thermal은 2m 수준, low-light는 완전 암흑에서도 최대 8m까지 가능했다(다만 눕기 자세의 long-range는 카메라 시점/기하 제약 영향). 또한 SQI 기반 유효 window 비율이 높은 범위(가시광에서 대체로 75~92%)로 유지되어, 재난/감염 위험 환경의 자율 triage·victim assessment를 위한 실전형 기반 기술로 의미가 크다.



### Transformer-Based Warm-Starting for Feasible and Optimal Terminal Approach to Tumbling Objects with Space Manipulators (https://arxiv.org/abs/2606.17317)
Comments:
          8 pages, 4 figures

- **Prior Approaches**: 기존 우주 로봇(스페이스 매니퓰레이터) 말단 접근은 NLP 기반 궤적 최적화를 SCP로 풀거나, 동역학을 단순화해 버스 구동을 고정/배제하는 방식이 많았습니다. 또한 warm-start를 위해 룩업테이블·필터·조건부 예측·휴리스틱 초기화 등이 시도됐지만, 3D 궤도 말단 접근처럼 회전하는 안전 제약과 버스-암 동역학 결합이 동시에 있는 경우 초기 추정 품질에 민감해 실시간 적용이 어렵습니다.

- **Core Contribution**: 본 논문은 비협조적으로 tumbling하는 목표에 대한 3D 말단 접근에서, sequential convex programming(SCP)의 핵심 병목인 2단계(자세-매니퓰레이터 토크 할당) 초기해를 causal transformer 기반 warm-start로 학습합니다. 문제를 COM(중심질량) 병진 계획(1단계)과 결합된 attitude-매니퓰레이터 토크 할당(2단계)로 분해하고, 특히 visibility cone과 회전 안전 제약을 포함한 2단계의 nonconvex 최적화를 더 빠르고 안정적으로 풀 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 난관은(1) 회전하는 안전 제약과 가시성 제약이 포함된 상태에서 비선형 결합 동역학 때문에 초기해가 조금만 틀려도 SCP 반복 수·런타임이 급증한다는 점입니다. 또한 학습 warm-start가 첫 SCP convexified subproblem에서 인위적 비가능성(trust region 제약)을 유발할 수 있어, 논문은 SCP 자체의 trust region 확장 휴리스틱과 결합하고, 2단계 쿼터니언 단위 제약 문제는 재정규화로 보정하면서 학습된 초기화를 안전하게 통합합니다.

- **Empirical Impact**: 300개의 held-out 시나리오에서 학습 warm-start는 2단계 SCP의 반복 횟수를 최대 28%, 런타임을 23%까지 줄이면서 최종 control-cost 분포는 유지했습니다. 더 나아가 cost-optimal 지향 SCP가 아니라 비선형 feasible projection에 사용할 때는 런타임이 거의 절반으로 줄었고, 휴리스틱 초기화에서 관측된 고비용 꼬리(catatstrophic high-cost tail) 현상도 피했습니다. 이 결과는 sequence-model warm-start가 최적화 기반 말단 유도(terminal guidance)의 계산 효율뿐 아니라 궤적 견고성까지 개선할 수 있음을 보여줍니다.



### Abstention-Aware Personalized Object Rearrangement via Uncertainty-Guided LLM Assistanc (https://arxiv.org/abs/2606.17309)
Comments:
          Accepted at the 2026 IEEE 35th International Conference on Robot and Human Interactive Communication (RO-MAN 2026)

- **Prior Approaches**: 기존 개인화 물체 재배치는 ‘어디에 놓을지’에 초점을 두고, 관측이 깔끔하며 모든 후보 물체를 배치 가능한 것으로 가정하는 경우가 많았습니다. 또한 LLM/VLM을 써도 few-shot 추론을 돕는 방식이 중심이라 지연·비용·프라이버시 우려가 남고, 복잡한 부분 상태나 이미 틀린 배치가 섞인 현실 상황에 취약했습니다. 무엇보다 많은 방법이 물체 수준에서 ‘아예 두지 않기(abstention)’를 결과로 모델링하지 못해, 사용자 선호와 충돌할 때의 예외 처리가 부족했습니다.

- **Core Contribution**: 이 논문은 abstention-aware 개인화 물체 재배치 프레임워크 APOLLO를 제안합니다. APOLLO는 사용자-환경 쌍별로 소량 시연으로 학습하는 경량 Personalized Embedding Model(PEM)로 각 물체의 배치 후보를 빠르게 예측하되, 불확실할 때만 선별적으로 LLM 추론을 호출합니다. 이를 통해 효율·프라이버시·추론 능력을 함께 맞추면서, UNPLACED(미배치) 선택으로 사용자 조직 방식에 어긋나는 물체 배치를 피할 수 있게 합니다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 제한된 데모로 사용자별 선호를 정교히 학습하면서 (2) 부분적으로 이미 놓여 있는 장면이 잡음/오류를 포함할 때도 안정적으로 abstention을 결정해야 한다는 점입니다. 저자들은 PEM의 top-1/top-2 확률 마진으로 불확실성을 정량화해 임계값 미만인 경우에만 LLM을 호출하는 uncertainty-guided 라우팅을 설계했고, PEM은 CPU에서 동작하며 user history와 partial context를 함께 인코딩하도록 구성했습니다. 또한 평가 자체가 기존 벤치마크의 ‘placement-only’ 가정에 갇히지 않도록, abstention과 노이즈 부분 맥락을 포함하는 합성 데이터셋 APOR를 새로 도입했습니다.

- **Empirical Impact**: 실험은 PARSEC(기존 placement-only)와 APOR(미배치 포함)에서 진행되며, APOLLO가 LLM 기반 이전 SOTA 대비 통제된 환경에서 성능을 개선하면서도 LLM 사용량을 크게 줄였다는 초기 증거를 제시합니다. 특히 APOR 결과는 기존 모델들이 abstention을 제대로 표현하지 못하거나 장면 복잡도와 부분 오류가 커질수록 취약해지는 경향을 보여줍니다. 즉, 개인화 재배치가 ‘무엇을 어디에 놓을지’뿐 아니라 ‘언제 놓지 않을지’까지 포함해야 한다는 실증적 방향성을 강화한 연구로 평가됩니다.



### VISTA: Scale-Aware Visual Navigation via Action History Conditioning (https://arxiv.org/abs/2606.17294)
- **Prior Approaches**: 기존 VNMs(예: ViNT, NoMaD, MetricNet)은 vision foundation model로 waypoint를 예측하지만, 정규화된(action/waypoint) 궤적을 학습해 서로 다른 속도·제어 주파수에서 스케일이 어긋나는 취약점이 생깁니다. 즉, 배포 시 스케일(실제 step size)이 달라지면 같은 normalized 궤적이라도 물리 기하가 변해 경로 추종이 흔들리고 충돌 위험이 커집니다. 또한 featureless하거나 시각적으로 반복적인 환경에서는 순수 시각 입력만으로 모션 크기와 기하를 안정적으로 추론하기 어렵다는 한계도 제기됐습니다.

- **Core Contribution**: 본 논문은 VISTA(Scale-Aware Visual Navigation via Action History Conditioning)를 제안하며, normalized action history를 이미지 관측과 함께 조건으로 넣어 “예측의 physical scale”을 정합시키는 방식을 제시합니다. 모델이 과거 normalized displacement(로봇의 최근 움직임 스케일)를 참고해 관측된 시각 변화와 실제 step size의 관계를 학습함으로써, metric-aligned waypoint 예측을 가능하게 합니다. 여기에 DINOv3 인코더를 결합해 관측 간 공간·기하 정보를 더 풍부하게 표현하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 단일 이미지 시퀀스만으로는 정규화된 목표가 의존하는 step size를 신뢰성 있게 역추정하기 어렵다는 점입니다. VISTA는 이를 해결하기 위해 action history를 출력과 같은 normalized 공간에 유지해 크로스-스페이스 스케일 접지(grounding) 부담을 줄이고, transformer 디코더가 과거 움직임과 현재/목표 관측을 함께 읽어 스케일 일관성을 맞추도록 학습합니다. 또 DINOv3 ViT-S의 self-supervised 토큰을 사용해 시각적 반복·저식별 장면에서도 기하 추론이 무너지지 않게 하며, 데이터의 샘플링 frequency를 2~12Hz로 다양화해 다양한 속도/제어 주파수에서도 견고성을 키웁니다.

- **Empirical Impact**: VISTA는 5개 실환경 Outdoor/Forest/Office-Lab 등에서 zero-shot 배포 시 goal prediction 정확도 100%(성공 목표 예측) 및 unseen 환경에서 체크포인트 crossing 평균 95%를 달성하며 경로 추종이 일관적임을 보여줍니다. 특히 속도·제어 주파수가 달라지는 sharp-turn 상황에서도 waypoint 스케일 정합이 유지돼 다른 베이스라인보다 충돌·경로 이탈이 덜 발생했습니다. 결과적으로 “정규화 때문에 생기는 metric misalignment” 문제를 action history와 풍부한 비전 표현으로 직접 보정하며, VNMs의 실세계 안전성과 신뢰성 향상에 의미 있는 실증을 제공합니다.



### Contrastive Action-Image Pre-training for Visuomotor Contro (https://arxiv.org/abs/2606.17256)
- **Prior Approaches**: 기존 로보틱스용 비전 인코더는 CLIP류의 image-text contrastive, DINO류 self-distillation 같은 ‘시맨틱 중심’ 학습이 주류였지만, 실제 조작 환경의 액션 구조나 직접적인 행동 슈퍼비전을 받지 못해 visuomotor 제어와의 정합성이 떨어집니다. 로봇 데이터로 학습하려 해도 trajectory 수집 규모가 인터넷급 영상보다 훨씬 작아, R3M·MVP 같은 접근은 프레임 레벨 대조학습이나 masked autoencoder 재구성 같은 목표로 우회해 액션 조건 정보를 누락하는 문제가 있었습니다. 또 egocentric human video 기반 방법들은 풍부하지만 로봇의 paired vision-action 신호가 없어 downstream 제어에 필요한 표현을 충분히 갖추기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 CAIP(Contrastive Action-Image Pre-training)을 제안하며, egocentric 인간 영상에서 추출한 손 자세를 로봇의 end-effector 액션을 대체하는 proxy로 사용합니다. 손의 3D 키포인트(인간 손 스켈레톤)를 액션 공간과 자연스럽게 대응시키고, contrastive objective로 이미지-텍스트 임베딩과 액션 임베딩을 정렬해 ‘액션 중심(action-centric)’ 시각 표현을 학습합니다. 결과적으로 적은 로봇 조작 데이터로도 물리 상호작용에 더 맞는 인코더를 만들 수 있는 확장 가능한 경로를 제시합니다.

- **Technical Challenges**: 핵심 과제는 대규모 인간 영상에 존재하는 ‘행동 정보’를 로봇 제어에 필요한 형태의 액션 신호로 변환해 학습 신호로 삼는 일이었습니다. 저자들은 MANO 기반 42 keypoint의 SE(3) 변환을 손 자세로 만들고, 델타 제어와 유사한 방식으로 미래 구간의 action chunk(대략 2초)를 구성해 액션 인코더로 모델링했으며, SigLIP 스타일 sigmoid contrastive loss로 대규모 배치에서의 학습 안정성을 확보했습니다. 또한 텍스트-조건 attention pooling을 통해 instruction과 연관된 이미지 토큰을 구성하고, 이 frozen vision encoder를 closed-loop 로보틱 정책에 전이해 성능을 검증합니다.

- **Empirical Impact**: 실험에서 CAIP는 Dexmate Vega + Sharpa Wave 손 기반의 정교한 dexterous manipulation 6개 태스크에서 평균 성공률 76%를 달성하며, DINOv2·SigLIP·MVP 등 강력한 비전 인코더 대비 큰 폭(평균 30%p 이상)으로 향상됐습니다. 특히 folding, pouring, fine-grained manipulation 같은 작업에서 30% 이상 성능 이득을 보이며, 태스크 전반에 걸쳐 일관된 강점을 나타냈습니다. action retrieval 및 out-of-distribution 액션 분류(선형 probing/zero-shot retrieval)에서도 CAIP가 기준선들을 능가해, 액션 중심 사전학습이 미지 환경과 데이터에도 잘 전이됨을 실증했습니다.



### ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining (https://arxiv.org/abs/2606.17200)
- **Prior Approaches**: VLA 모델은 대규모 로봇 궤적이 일반화의 핵심 재료지만, 데모 수집이 비싸고 느려 행동 다양성이 병목이 되곤 합니다. 최근에는 egocentric human video를 pseudo-action으로 활용하려 했지만, (1) 로봇과 다른 좌표·정렬 구조, (2) 비전 기반 라벨링의 잡음이 그대로 학습에 섞이면서 mixed-source 사전학습이 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 ACE-EGO-0로, 서로 다른 데이터 소스를 한 번에 학습할 수 있도록 통합 VLA pretraining 프레임워크를 제안합니다. 핵심은 로봇 데모와 human video pseudo-action을 동일한 camera-space 기준의 통합 행동 표현으로 맞추고, 형태(morphology) 조건과 시간 정렬까지 포함해 ‘표현 불일치’를 줄인 점입니다.

- **Technical Challenges**: 문제는 표현 정렬만으로 부족하다는 것입니다. ACE-EGO-0은 pseudo-action 라벨의 잡음을 직접 모델을 망치지 않도록 reliability-aware 학습 목적을 설계해, 채널별·스텝별 신뢰도에 따라 human 보조 손실 가중치를 동적으로 낮추고 로봇 기반 flow-matching을 주된 학습 기준으로 고정합니다.

- **Empirical Impact**: 실험에서는 로봇/시뮬레이션 4.53K시간에 pseudo-action-labeled human video 1.48K시간을 함께 써서, unified joint pretraining과 supervised fine-tuning에서 일관된 성능 향상을 확인했습니다. RoboCasa GR1 TableTop에서 72.8% 평균 성공률, RoboTwin 2.0에서 91.12%(Easy)/90.62%(Hard)를 달성했으며, 장기·접촉이 많은 bimanual 과제에서 실세계 전이에 강점을 보였습니다.



### VL-MemKnG: Hybrid Memory with a Spatio-Temporal Knowledge Graph for Question Answering over Long Egocentric Navigation Trajectories (https://arxiv.org/abs/2606.17183)
- **Prior Approaches**: 긴 에고센트릭 비디오에서 내비게이션 관련 질문을 풀기 위해선 멀리 떨어진 시점의 증거를 찾아 시간적으로 정렬해야 하지만, 기존 long-context 비전-언어 모델은 연산 비용이 커 반복 질의에 비효율적이다. 그래프 기반 persistent spatio-temporal knowledge graph(VL-KnG 계열)는 구조화된 관계를 잘 다루지만, 더 넓은 시간적 연속성과 문맥 단서를 충분히 대표하지 못한다는 한계가 지적된다. 또한 짧은 클립/요약 중심 접근은 긴 궤적에서 전역 문맥이 약해질 수 있다.

- **Core Contribution**: 이 논문은 VL-MemKnG를 제안해 spatio-temporal knowledge graph와 segment-level contextual memory를 결합하는 하이브리드 메모리 프레임워크로 증거 기반(grounded) 답변과 시간적 근거 국소화를 동시에 노린다. 지식그래프는 객체/속성/공간관계 같은 구조적 관계를 제공하고, 세그먼트 메모리는 장기 구간의 문맥 연속성을 보존해 멀리 흩어진 증거 검색과 집계를 돕는다. WalkieKnowledgeT+도 함께 소개되며, 비동시(non-cooccurring) 다중 구간에 걸친 증거 집계형(chronology-aware) 추론 문제를 벤치마크로 제공한다.

- **Technical Challenges**: 핵심 난제는 (1) 긴 궤적에서 관련 구간을 효율적으로 검색하고, (2) 서로 겹치지 않는 여러 시간대 증거를 올바르게 묶어 집계하며, (3) 질의 시 비용을 낮춘 채 시각 근거를 끝까지 유지하는 것이다. 이를 위해 오프라인에서만 세그먼트 캡션/임베딩과 spatio-temporal knowledge graph를 구축하고, 온라인 단계에서는 그래프 서브그래프와 세그먼트 메모리를 병렬로 검색한 뒤 하이브리드 retrieval-and-reasoning 모듈이 시간 구간 정렬과 답변 예측을 함께 수행한다. 또한 단일 시점 질문은 flat ranked frame list, 다중 구간 집계 질문은 grouped 증거 포맷으로 처리해 시간 분산 증거를 구조적으로 반영한다.

- **Empirical Impact**: WalkieKnowledgeT+에서 VL-MemKnG는 Top-1 retrieval 정확도를 58%→67%, Recall@1을 34.50%→40.55%로 끌어올리며 비교 방법 전부(예: Gemini 2.5 Pro, Qwen 3.5+)를 능가한다. 특히 temporal-global 및 temporally scattered aggregation 질문에서 개선 폭이 커, 구조적 관계 메모리와 세그먼트 문맥 메모리의 결합 효과가 드러난다. 더불어 오프라인 메모리 구축-온라인 효율 추론 설계 덕분에 긴 궤적에서의 반복 질의 시에도 실용적인 성능/비용 균형을 보여준다.



### ParkingTransformer: LLM-Enhanced End-to-End Trajectory Planning for Autonomous Parking (https://arxiv.org/abs/2606.17082)
- **Prior Approaches**: 기존 end-to-end 자율주행/주차 연구는 모듈형 기준선보다 오류 전파는 줄이지만, 여전히 dense BEV를 거치거나(계산비용 증가) 장면을 충분히 이해하지 못하거나(정밀 기하 추론 부족) 의사결정이 불투명하다는 문제가 남아 있습니다. LLM을 붙인 시도는 설명 가능성을 높이지만, hidden states를 그대로 궤적 계획에 쓰면 주차에 필요한 centimeter급 정밀도와 공간 추론이 부족하다고 지적됩니다. 또한 long-distance(50~300m) 주차에서는 과거 이력 모델링과 정교한 coarse-to-fine 보정이 결여되어 성능이 급격히 붕괴하는 경향이 있습니다.

- **Core Contribution**: 이 논문은 ParkingTransformer라는 새로운 end-to-end 주차 프레임워크를 제안하며, multi-view 시각지각과 LLM 기반 장면 이해를 한 Transformer 아키텍처에 결합합니다. trajectory queries를 LLM의 implicit state features와 결합해 dense BEV 중간표현 없이도 직접 궤적을 출력하도록 설계했습니다. 여기에 LLM의 공간추론 약점을 보완하기 위한 3D positional encoding, 장기 시간정보 처리를 위한 fixed-window streaming, 디코더의 coarse-to-fine 점진 정밀화 전략을 추가합니다.

- **Technical Challenges**: 가장 큰 기술 난관은 (1) dense BEV의 계산비용을 줄이면서도 (2) 주차에 필요한 장면 의미/공간 기하를 충분히 주입하고 (3) LLM의 제한된 3D 추론을 궤적 정밀도로 연결하는 것입니다. 논문은 SCA(Sensor Cross-Attention)에서 3D positional encoding과 depth 후보 샘플링(LID)을 사용해 기하 정보를 명시적으로 주입하고, TCA(Temporal Cross-Attention)로 과거 이력과 현재 정보를 결합하되 fixed-window streaming으로 계산을 통제합니다. 마지막으로 coarse-to-fine 디코더로 전역 경향→국소 보정 순서의 잔차(refinement)를 누적해 정확도를 끌어올립니다.

- **Empirical Impact**: CARLA closed-loop 실험에서 driving score 61.32를 달성하고, 실차 실험에서는 평균 success rate 88.70%를 보고해 long-distance 주차 가능성을 실증했습니다. 짧은 거리(약 20m)에서는 기저모델 대비 비슷하거나 약간의 개선 수준이지만, 거리 증가(50~300m)에서는 LLM/이력/정교 보정이 없는 baseline들이 거의 붕괴한 반면 ParkingTransformer는 로드→진입→근접→리버스 주차까지 완주합니다. ablation 결과로 TCA/SCA, LLM 모듈, 3D positional encoding이 성능에 유의미하며, 주차 속도(통상 15 km/h 이하) 조건에서는 inference speed도 실용 범위 내라고 제시합니다.



### HRDX: A Large-Scale Vector HD-Map Datas (https://arxiv.org/abs/2606.17080)
Comments:
this https URL

- **Prior Approaches**: 기존 벡터 HD map 구축 연구는 polyline/segmentation 계열 표현을 end-to-end로 예측하는 방향(예: MapTR, MapTracker 등)으로 발전했지만, 대다수 성능 병목이 데이터 규모와 라벨 풍부함의 한계를 크게 받습니다. 공개 데이터셋은 주행 커버리지가 짧고(대개 10시간 미만), geometry 중심에 비해 lane color·style, 도로 텍스트 같은 규제/행동 시맨틱 속성은 희소하거나 누락되는 경우가 많습니다. 또한 aerial imagery처럼 차량 주행 궤적과 정밀 정합된 모달리티가 부족해 cross-view 융합이나 privileged information 학습을 체계적으로 다루기 어렵다는 지적이 있었습니다.

- **Core Contribution**: 이 논문은 온라인 벡터 HD map 구축을 위한 대규모 데이터셋 HRDX를 제안합니다. HRDX는 약 40시간(1,400km) 규모의 최소 중복 주행을 모으고, 6대 동기화 surround camera, 128-beam LiDAR, RTK GNSS/IMU에 더해 차량 궤적과 정밀 정합된 aerial orthoimagery(최대 8cm/pixel)를 제공합니다. 라벨은 10개 벡터 map 클래스와 20+개의 semantic·topological 속성으로 확장되어, 기존 공개 데이터셋에 없던 규제/행동 시맨틱까지 포함합니다.

- **Technical Challenges**: 주요 과제는 (1) geometry뿐 아니라 속성 정확도까지 함께 평가/학습 가능한 학습 체계를 만드는 것과 (2) aerial을 정합·재현 가능하게 제공하되 추론 센서 비용을 늘리지 않는 활용법을 찾는 것입니다. 논문은 geometry-mAP(Chamfer-distance 기반)만 보던 한계를 보완하기 위해 Composite Score(CS)로 위치 정밀도와 attribute correctness를 동시에 측정하도록 설계합니다. 또한 aerial 기반 BEV 컨텍스트를 cross-attention 융합에 활용하되, 추론에서는 카메라만 쓰도록 teacher–student knowledge distillation(C+A/C) 프레임을 적용해 aerial의 이점을 카메라-only 학생에게 전이합니다.

- **Empirical Impact**: 실험은 HRDX의 데이터 규모 확장이 mAP과 CS를 단조적으로 끌어올린다는 결과를 보여주며, 대규모·다양한 주행 커버리지가 학습 안정성과 일반화에 핵심임을 시사합니다. 더불어 aerial imagery를 학습과 추론에 모두 사용하면 카메라-only 기준 대비 mAP과 CS가 각각 큰 폭으로 개선되며, stop lines·crosswalks·도로 경계·도로 텍스트처럼 전역 레이아웃과 가림(occlusion) 완화의 혜택이 큰 요소에서 상승이 집중됩니다. 마지막으로 aerial만 학습에 쓰고 추론은 카메라-only로 유지해도 성능 격차를 상당 부분 줄일 수 있어, deployment-feasible한 privileged information 활용 전략으로 의미가 큽니다.



### Extracting Semantics: LLM-Guided Automatic Population of Robot Ontology from URDF (https://arxiv.org/abs/2606.17073)
- **Prior Approaches**: 인지 로보틱스에서 온톨로지는 환경 지식과 로봇의 신체(embodiment)·기능 정보를 통합해 설명 가능한 추론을 가능하게 하지만, 수동 구축이 큰 병목이다. 또한 URDF는 로봇의 구조와 기구학을 제공하더라도, 의미 있는 식별자·명칭은 commonsense 해석이 필요해 그대로는 의미론적 풍부함을 얻기 어렵다. 기존 자동화는 주로 구조 파싱에 머물러 온톨로지 정합성(스키마/제약)까지 안정적으로 맞추는 데 한계가 있다.

- **Core Contribution**: 이 논문은 URDF를 입력으로 받아, LLM을 활용해 의미론적 추상(semantic abstractions)을 채운 온톨로지로 자동 생성하는 파이프라인을 제안한다. 핵심은 온톨로지의 개념 프롬프트를 바탕으로 LLM이 의미 관계를 추론하되, 최종 분류는 형식 모델과 온톨로지 제약을 따르도록 정렬(alignment)하는 점이다. 즉, 로우레벨 URDF에서 휴먼-로보트 인터랙션에 필요한 grounded·구조화 지식 표현으로의 브리지를 만든다.

- **Technical Challenges**: 가장 큰 기술적 난제는 URDF 식별자가 요구하는 commonsense 의미를 LLM이 잘 추론하더라도, 생성 결과가 온톨로지 스키마와 문법 규칙을 위반할 수 있다는 신뢰성 문제다. 이를 위해 파이프라인은 여러 LLM 질의에 대한 majority voting으로 변동성을 줄이고, 구문(syntactic)·스키마 레벨 validation을 추가해 출력이 기대 표현 형식과 온톨로지 제약을 만족하는지 검증한다. 또한 “형식 모델에 맞춘 분류”를 목표로 프롬프트를 온톨로지 개념에 고정해 추론의 방향성을 통제한다.

- **Empirical Impact**: 여러 로봇 설명(URDF)에서 생성된 추상 온톨로지를 평가하며, 로우레벨 모델에서 구조화된 의미론적 표현으로의 변환이 실제로 가능함을 초기 결과로 보인다. 저자들은 생성된 추상들이 온톨로지 기반 추론에 필요한 grounded 지식을 제공할 수 있음을 논의한다. 인력 의존이 큰 온톨로지 구축 비용을 낮출 수 있어, 사람과 상호작용하는 embodied agent의 지식 업데이트/설명 가능성 측면에서 의미 있는 출발점이 될 것으로 보인다.



### MOCHI: Motion Enhancement of Collaborative Human-object Interactions (https://arxiv.org/abs/2606.18243)
Comments:
          SIGGRAPH 2026 Journal (ACM TOG); Project page: this https URL

- **Prior Approaches**: 기존 협력적 인간-물체 상호작용(MHOI) 연구는 고품질 데이터 확보를 전제로 하며, 캡처 기반 접근은 인간-인간·인간-물체가 동시에 발생하는 복잡성 때문에 잡음이 큰 경우가 많습니다. 그 결과 손-물체 접촉 정렬 오류, 모션 지터 및 시간적 불일치, 손가락 수준 관절 표현 누락·불완전 같은 아티팩트가 빈번하게 관측됩니다. 생성 모델로 데이터를 보강하려는 시도도 있으나, 단일 캡처의 물리적 타당성과 상호작용 일관성을 동시에 안정적으로 복원하기는 어렵다는 한계가 있었습니다.

- **Core Contribution**: 논문은 잡음이 포함된 MHOI 데이터를 개선하는 2-stage 프레임워크 MOCHI를 제안합니다. 첫 단계에서 잡음이 섞인 바디 입력으로부터 물리적으로 그럴듯한 hand grasp를 최적화해 바디 포즈와의 의미적 정합성을 확보한 뒤, grasp를 손-물체 상호작용 전체 시퀀스로 확장합니다. 둘째 단계에서는 single-person motion priors를 활용한 diffusion 기반 잡음 최적화로 모든 참여자의 전신 모션을 함께 정제합니다.

- **Technical Challenges**: 핵심 기술 난제는 잡음 캡처에 내재한 손-물체 접촉 미스얼라인, 시간적 흔들림, 손가락 관절 결손을 서로 다른 참여자 간 상호작용까지 고려해 일관되게 복원하는 데 있습니다. MOCHI는 최적화 단계에서 물리적 그립의 실현가능성과 바디-그립 의미 정합을 같이 강제하고, diffusion 단계에서는 단일-person priors 안에 human-object·human-human interaction 정보를 인코딩하는 최적화 목적함수를 추가해 상호작용 신호가 priors에 흡수되도록 설계합니다. 이를 통해 전신 모션 정제 과정에서도 접촉과 상호작용의 구조가 무너지지 않게 했습니다.

- **Empirical Impact**: 실험은 기존 캡처 방식이나 생성 모델로 얻은 다양한 MHOI 데이터에 대해 MOCHI 파이프라인의 효과를 보여주며, 참가자 수가 달라지거나 상호작용 유형이 변해도 강건함을 확인합니다. 또한 keyframe 기반 MHOI 생성, 객체 형상 변형을 통한 데이터 증강 등 응용 가능성을 제시해, 단순 복원을 넘어 데이터 파이프라인 전반의 활용도를 높였다는 점에서 의미가 큽니다.



### Adaptive Volumetric Mechanical Property Fields Invariant to Resolution (https://arxiv.org/abs/2606.18231)
Comments:
          Project Page and hi-res paper: this https URL. ICML 2026

- **Prior Approaches**: 변형(Deformable) 시뮬레이션은 물성장(Young’s modulus E, Poisson’s ratio ν, density ρ)이 물체 내부 전부에 대해 공간적으로 주어져야 하지만, 기존 3D 에셋에는 이런 정보가 거의 없다. 이를 자동화하려는 학습 기반 접근들은 대부분 정확도나 해상도 한계로 인해 고해상도·고정밀 물성장을 만들기 어렵거나, 입력 격자 고정으로 메모리 효율이 떨어진다. 특히 VoMP와 같은 최고 성능 방법도 고정 해상도 격자를 쓰는 탓에 더 촘촘한 예측으로 확장할 때 비용이 급증한다.

- **Core Contribution**: AdaVoMP는 입력 3D 형상에 대해 공간적으로 변하는 물성(E, ν, ρ)을 조밀하게 예측하는 방법으로, 해상도·정확도·메모리 효율을 동시에 끌어올린다. 핵심은 기존 VoMP의 고정 voxel 모델을 대체해, 각 입력마다 고유한 sparse adaptive voxel 트리(SAV)를 생성하는 sparse transformer encoder-decoder를 도입한 것이다. 이로써 이전 대비 훨씬 높은 유효 해상도(예: 1024^3 급)에서 물성 경계와 복잡 영역을 더 정밀하게 복원한다.

- **Technical Challenges**: 큰 어려움은 (1) 고해상도 물성장을 만들면서도 (2) 격자 전체를 전개하지 않고 (3) 물성의 ‘빈 공간(Empty)’과 다중 해상도 구조를 일관되게 학습하는 것이다. AdaVoMP는 SAV로 균질 영역은 거친 셀로 유지하고 이질·경계만 미세 분할하도록 하며, 생성기(G)는 coarse-to-fine autoregressive로 “Empty/Keep/Subdivide” 구조를 함께 예측해 불필요한 연산을 줄인다. 또한 격자 없는 생성이 되도록 transformer가 통합 좌표 기반의 sparse attention을 수행하고, MatVAE 디코더를 고정해 물성(latent→E, ν, ρ)으로의 물리적 타당성을 유지한다.

- **Empirical Impact**: 실험에서 AdaVoMP는 기존 SOTA보다 더 정확한 volumetric properties를 추정하면서도, 모든 선행 대비 더 적은 test-time compute로 동작하는 경향을 보인다. 특히 적은 voxel로 균질 영역을 요약하면서도 복잡한 부분의 경계를 잘 살려, 시뮬레이션 가능한 고해상도 물성 에셋 변환에 직접적으로 유리하다는 점이 강조된다. 결과적으로 로봇 학습용 physics simulation in the loop 디지털 환경 제작에서 ‘물성 부여’ 병목을 크게 낮출 수 있는 실용적 임팩트를 가진다.



### Memory as a Wasting Asset: Pricing Flash Endurance for Embodied Agents, and the Limits of Doing So (https://arxiv.org/abs/2606.18144)
- **Prior Approaches**: 기존 로봇용 embodied-memory 연구는 주로 무엇을 기억할지(when-to-write), 언제/어떻게 불러올지(what-to-remember)에 집중해 왔습니다. datacenter wear-aware caching은 잡음은 줄이고(내구성 기반 write-heavy 객체를 플래시에서 피하는) 한정된 endurance 맥락의 정성적 행동을 보이지만, 로봇에서는 RAM/NVM/cloud를 동시에 오가며 전력·지연과 비재생 endurance까지 함께 가격(erase cycle의 기회비용)으로 묶는 모델이 비어 있었습니다.

- **Core Contribution**: 이 논문은 로봇의 embodied memory를 ‘감가상각 자본’으로 보고, NVM의 program/erase cycle을 비재생 자산의 scarcity rent(내구성 shadow price) η로 가격화합니다. 그 결과 RAM/on-board NVM/cloud 배치가 ‘wear-augmented per-byte index’의 임계값(threshold)으로 결정되며, 비용 최적 배치는 value-write association χ의 부호(양/영/음)와 무관하게 기본형을 유지하되, χ>0일 때는 가치가 큰 메모리가 오히려 flash에서 멀어지는 비단조(non-monotone) 최적해가 나타남을 보입니다.

- **Technical Challenges**: 핵심 난제는 (1) erase cycle이라는 단일 비재생 예산을 intertemporal(시간에 걸친) 재무제약으로 모델링하고, (2) ‘가치와 write 강도의 결합’이 생기는지(χ의 부호)를 이론이 아니라 실제 로봇 로그로 측정해 조건부 정리를 실행 가능하게 만드는 것입니다. 논문은 WHEN→WHERE→WORTH 3단 구조에서 AURA의 when-to-write 게이트를 유지한 채 WHERE에서 RAM/NVM/cloud 배치를 wear-aware index로 풀고, χ는 미리 지정된 gate에서 실측해 χ>0 가지에서만 비단조 최적성이 적용되도록 “가짜 일반법칙” 위험을 차단했습니다.

- **Empirical Impact**: 실험/분석의 중심 결과는 χ의 부호가 배치 환경의 성격에 따라 달라진다는 점입니다(장기 반복적인 조작에서는 χ>0, 짧은 horizon에서는 χ≈0, 비재귀적 텔레오퍼레이션에서는 χ<0). 또한 내구성 예산이 datasheet 가격의 premium 3,000 P/E TLC에서는 거의 ‘비활성’이지만 commodity QLC/eMMC(~1,000 P/E)에서는 강하게 ‘구속’되며, 구속 구간에서 learned wear-aware 컨트롤러가 endurance-aware 성능을 보이되 과제 가치(task value) 자체 개선은 아직 데이터로 관측되지 않아 향후 과제로 남겼습니다.



### Learn to Quantify Social Interaction with Constraints for Pedestrian Walking (https://arxiv.org/abs/2606.17897)
- **Prior Approaches**: 기존 군중 장기 궤적 예측 연구는 Social LSTM, spatio-temporal graph, Transformer 등으로 사회적 상호작용을 반영했지만, 실제로는 충돌 회피·행렬 동행 같은 현상을 사후적으로 해석하는 경우가 많았습니다. 또한 많은 모델이 서로 다른 상호작용을 구분 없이 풀링해 “무슨 종류의 상호작용이 인코딩됐는지”를 설명하기 어렵다는 한계가 남아 있습니다. 반면 GAN/VAE/goal-based, MDN 계열은 다중 모달리티를 다루지만, 사회적 상호작용의 의미 있는 모드 자체를 라벨 없이 학습해 해석 가능하게 만드는 데는 집중이 약했습니다.

- **Core Contribution**: 이 논문은 보행자 사이 상호작용을 라벨 없이 “모드(클러스터)”로 학습하고, 그 모드를 예측 모델 내부에 자연스럽게 통합하는 Learn to Cluster를 제안합니다. 확률적 잠재변수 생성모형으로 edge(에이전트-이웃 관계)마다 이산 latent variable을 두어, 궤적 관측에서 직접 상호작용 유형을 분류(클러스터링)하도록 만듭니다. 또한 학습된 latent 변수는 예측 네트워크에서 다중 미래를 생성하는 mixture components의 조건 신호로 활용되어, 불확실성을 다루면서도 사회적 “스타일”을 해석 가능한 형태로 제공합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 상호작용을 정확히 라벨링하기 어렵고, (2) 이산 샘플링이 backpropagation을 끊는 문제이며, (3) 클러스터 간/내 표현이 분리되도록 학습을 안정화해야 한다는 점입니다. 논문은 Gumbel-Max와 softmax 근사로 이산 latent variable을 미분가능하게 다루고, Maximum Coding Rate Reduction 계열의 사회 loss로 동일 클러스터는 압축하고 서로 다른 클러스터는 대비되게 강제합니다. 그 결과, latent 모드가 “공격적/온건/주의 없음”처럼 시간에 따라 변화하는 사회적 행동의 스타일을 확률적으로 포착하도록 설계됐습니다.

- **Empirical Impact**: UCY와 ETH 벤치마크에서 leave-one-out로 평가하며, ADE/FDE 기준 성능 향상과 함께 모드가 의미 있게 분포한다는 해석 결과를 제시합니다. 시나리오 A~D(근거리 마주침, 평행 보행, 원거리 마주침, 역방향 후면 등) 통계 분석에서 특정 모드가 특정 상황에 우세하게 나타나 “모드의 의미”를 뒷받침합니다. 또한 모드 전환(예: Mode 2→Mode 3, Mode 3→Mode 1) 분석을 통해 상대적 위치·속도 관계가 변화할 때 클러스터가 이동함을 보여, 향후 explainable social interaction 기반 예측 연구에 실증적 근거를 제공합니다.



### GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning (https://arxiv.org/abs/2606.17480)
- **Prior Approaches**: GeneralVLA 계열은 SAM 기반 어포던스 분할과 3D 장면 추정, 그리고 언어-3D-기억을 결합한 mid-level 3DAgent의 경로를 저수준 제어가 실행하는 계층형 VLA를 제안했습니다. 하지만 기존 GeneralVLA의 3D evidence는 단일 이미지/마스크 기반 SAM3D류에서 포즈 모호성과 뒷면 구조 환각 문제가 남고, KnowledgeBank는 의미 유사도 중심이라 안전성·현재 장면 기하 적합성·충돌·신뢰도 같은 “기억 품질”을 통제하기 어렵습니다.

- **Core Contribution**: 논문은 GeneralVLA-2에서 planner-facing 입력을 강화하는 두 축을 제시합니다: GeoFuse-MV3D로 다중 시점 RGB-D에서 더 안정적인 객체 중심 3D 기하 evidence를 만들고, governed KnowledgeBank로 장기 조작 경험 재사용을 품질·신뢰·라이프사이클·충돌 메타데이터까지 포함해 제어합니다. 특히 GeoFuse-MV3D는 외부 기하 추정을 “직접 재구성”이 아니라 geometry-prior로 취급하며, 관측 마스크와 함께 보수적으로 융합해 downstream 계획에 필요한 기하만 안정화합니다.

- **Technical Challenges**: 첫째, multi-view라도 기하 prior가 마스크와 불일치할 때 환각을 줄여야 하고, 동시에 다운스트림이 민감한 색/불투명/모양 드리프트를 피해야 합니다. GeoFuse-MV3D는 입력 view 마스크로 prior를 검증하고 soft visual-hull 지원, 축(axis)-wise refinement, geometry-only 보수적 fusion을 통해 기하만 조정한 뒤 appearance는 보존하는 방식으로 이를 해결합니다. 둘째, 기억은 의미적으로는 맞아도 현재 장면에 안전하게 적용되지 않을 수 있으므로, admission·retrieval·충돌 처리·승격/요약/폐기까지 포함한 “governed” 메모리 스키마와 verifier 기반 품질 점수를 도입해 재사용 신뢰성을 높였습니다.

- **Empirical Impact**: 실험에서 GeoFuse-MV3D는 GSO-30에서 MV-SAM3D 대비 CD와 LPIPS를 각각 2.20%, 2.02% 낮추고 PSNR과 SSIM을 각각 2.36%, 1.03% 높여 다중 시점 객체 재구성 품질을 개선했습니다. KnowledgeBank는 Terminal-Bench 2.0와 SWE-Bench Verified에서 ReasoningBank 대비 Terminal-Bench SR은 4.53%, SWE-Bench resolve rate은 3.73% 향상시키면서 AS는 각각 4.95%, 5.65% 감소시켜 장기 조작 경험이 “안전하게” 재사용될 때 성능이 오른다는 점을 입증했습니다. 전반적으로 계층형 VLA에서 planner 입력(객체 기하 evidence와 메모리 거버넌스)을 강화하면 장기 계획의 안정성이 올라간다는 실증적 메시지를 제공합니다.



### WeaveLA: Event Driven Cross-Subtask Latent Memory Weaving for Repetitive Robot Manipulation (https://arxiv.org/abs/2606.17463)
- **Prior Approaches**: 단일 step 조작에서 강점을 보인 Vision-Language-Action(VLA) 정책은 짧은 윈도우에 의존하지만, 반복 과제처럼 하위 태스크 간 의존성이 있는 상황에서는 직전 결과를 다음 단계에 전달하는 구조가 부족해 취약해진다. 기존 memory-augmented 변형은 프레임마다 write하거나, 데모 단계 기반 retrieval을 하거나, 서브목표 이벤트에서만 신호를 내보내지만 ‘다음 action expert로의 명시적 핸드오프’를 제대로 수행하지 못한다.

- **Core Contribution**: 이 논문은 cross-subtask 정보 전달의 자연스러운 타이밍 단위로 ‘sub-goal completion event’를 제안하고, 이 이벤트마다 완료된 구간을 압축해 다음 서브태스크의 action 생성 경로에 직접 주입하는 WeaveLA를 제시한다. WeaveLA는 frozen VLA 백본 위에 가벼운 cross-subtask latent memory 인터페이스를 얹어, 기존 정책의 short-window 입력 인터페이스는 그대로 유지하면서도 다음 단계 의존성을 해결한다.

- **Technical Challenges**: 핵심은 (1) 경계 정보를 프레임 단위로 계속 저장할 때의 비용과 불안정성을 피하면서, (2) 데모-time retrieval처럼 롤아웃 진행도에 맞춘 키잉을 보장하기 어렵다는 문제를, (3) action-level 실행 단계로 정보가 희석되지 않게 전달하는 것이다. 해결책으로 WeaveLA는 sub-goal completion event에서 query-driven attention pooling으로 구간을 8개의 latent tokens로 압축한 뒤, action expert 내부의 AdaRMS 모듈에 memory-conditioned 컨텍스트로 라우팅하며, 학습 안정화를 위해 flow-matching 기반의 단계적 학습 커리큘럼을 사용한다.

- **Empirical Impact**: RoboMME에서 π0.5 backbone을 사용한 stratified 평가 결과, 단일 실행(N=1) 성능은 약 100% 수준으로 유지되는 반면 반복이 필요한 구간에서만 이득이 나타났다. 특히 SwingXtimes에서 N=3일 때 success가 0%에서 47.8%로 크게 상승했으며, 어려운 반복/시간 의존 태스크들에서만 성능이 집중적으로 개선되어 ‘필요한 곳에만’ 작동한다는 메커니즘 정합성이 확인된다.



### Credibility-Weighted Pricing of Autonomous Vehicle Liability Under Operational Design Domain Shif (https://arxiv.org/abs/2606.17451)
- **Prior Approaches**: ADS 안전/손해율을 추정하기 위한 기존 연구는 주로 HDV 대비 retrospective 벤치마킹에 집중해, 보고 임계치 차이·ODD(Operational Design Domain) 혼동 같은 함정을 보정하려고 했다. 그러나 이런 방식은 “새 도시/새 소프트웨어 릴리즈에 대해 미래 요율을 어떻게 이전할지”라는 prospective pricing 문제를 정면으로 다루지 못했다. 전통적 credibility 이론(Bühlmann–Straub 등)은 교환가능(exchangeable) 가정을 통해 같은 shrinkage를 주는데, ADS에서는 도시 간 ODD 차이와 소프트웨어 비정상성 때문에 그대로 적용하기 어렵다.

- **Core Contribution**: 이 논문은 계층적 베이지안 credibility를 “도시-ODD-소프트웨어 버전” 단위로 확장해, sparse experience와 shifting ODD, non-stationary risk를 동시에 다루는 프레임워크를 제안한다. 핵심은 learned ODD-similarity 커널로 도시 간 정보가 “얼마나, 누구에게서” 흘러갈지 정하고, 소프트웨어 버전 무작위효과로 릴리즈 간 위험 변화를 흡수한다. 또한 모델 구조가 Bühlmann–Straub를 특수한 극한 사례로 포함하도록 설계해, 해석가능성과 이론적 정합성을 함께 확보했다.

- **Technical Challenges**: 기술적 난관은 (1) 새 배치 타깃 셀의 직접 경험이 거의 0에 가까운데도 요율을 안정적으로 만들고, (2) 같은 관측 공변량이라도 잠재 ODD 차이를 반영하며, (3) 소프트웨어 업데이트가 위험 수준을 불연속적으로 바꾼다는 비정상성을 표현하는 것이다. 저자들은 Poisson 빈도 GLM의 계층 구조(도시/버전/도시-버전 상호작용) 위에, H3 그리드 기반 공개 지리·노면·노출·HDV 데이터로 contrastive learning한 ODD 임베딩을 GP 커널에 연결해 도시 랜덤효과의 상관을 학습한다. 그 결과 새 도시에 대한 posterior 예측은 임베딩 유사도에 의해 조건부 가우시안 형태로 모멘트가 결정되도록 구성된다.

- **Empirical Impact**: NHTSA Standing General Order 데이터로 4개 미국 메트로의 648건 verified-engaged Waymo 크래시, 1.16억 마일에 대해 평가했으며, city-aggregate credibility weight는 0.12~0.46의 중간 수준으로 나타나 부분풀링이 현실적인 범위에서 작동함을 시사한다. 실험적으로는 no pooling 대비 partial pooling이 결정적으로 더 나은 성능을 보였고, learned 커널의 이점은 약 12개 배치 도시 시점부터 통계적으로 관측 가능해지는 것으로 power analysis에서 확인했다. 즉, 보험 요율 산정에서 “경험이 희소한 초기 단계”에도 점진적 정보 이전을 안정적으로 제공하면서, 충분한 규모가 쌓이면 ODD 유사도 학습의 가치가 드러난다는 점에서 의미가 크다.



### TerraTransfer: Learning End-to-End Driving Policies Without Expert Demonstrations (https://arxiv.org/abs/2606.17386)
- **Prior Approaches**: 기존 end-to-end 자율주행 학습은 대체로 로그 운전자 데이터를 기반으로 imitation pretraining을 하고, 이후 fine-tuning(지도학습·open-loop RL) 또는 closed-loop RL을 추가하는 방식이었습니다. 특히 closed-loop RL은 photorealistic rendering과 대형 vision backbone 추론을 매 스텝 반복해야 해 계산비용이 커지고, 희귀·안전중요 상태는 로그에 충분히 없어서 covariate shift 문제가 남았습니다. self-play는 비용이 싸지만 픽셀 대신 vector state로 학습되는 경우가 많아 실제 raw image end-to-end로 확장되지 못했습니다.

- **Core Contribution**: 이 논문은 self-play의 “학습은 저렴한 vector 상태에서, 추론은 raw image로”라는 비대칭 이점을 결합해, demonstration 없이 end-to-end 주행을 만드는 단일 패러다임을 제안합니다. 핵심은 learning to drive(벡터 상태 기반 planning head)를 먼저 self-play로 학습한 뒤, learning to see(vision encoder)를 alignment 단계에서 맞추되 어떠한 단계도 logged trajectory를 상대로 supervise하지 않는다는 점입니다. alignment는 teacher(자기대전 self-play policy)의 action distribution과 표현 관계를 재현하도록 설계되어, 큐레이팅된 expert demonstration 없이도 전이가 가능함을 노립니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 vector state에서 학습된 teacher를 raw image 입력 공간으로 옮길 때, 단순 모듈 캐스케이드(검출/기하 복원 후 재입력)가 불필요하게 어려운 상위 복원 문제를 만든다는 점입니다. 이를 피하기 위해 teacher가 내부에서 사용하는 “풀드(pooled) 특징”과 직접 정렬하도록 설계했고, teacher 특징의 저랭크/중복성을 활용해 batch-relational low-rank structural loss로 관계(씬 간 유사성)를 주로 맞추도록 제한했습니다. 또한 action KL divergence로 학생의 정책 분포가 teacher와 일관되게 되도록 하여, 로그 경로 없이도 행동 정렬이 유지되게 했습니다.

- **Empirical Impact**: 평가는 photorealistic 3D Gaussian splatting 기반 closed-loop 벤치마크 HUGSim에서 HD-Score로 진행되며, 제안한 vision-정렬 end-to-end 정책이 imitation-trained 선행 방법들을 aggregate에서 match하거나 초과합니다. 특히 self-play teacher에 근접한 성능(집계 HD-Score 기준 teacher 대비 오차 약 0.03)을 보이면서도, paired (image, scene-state) 프레임은 약 1.83M 정도만 사용해 데이터 효율이 높다는 점이 강조됩니다. 또한 paired 데이터 비율을 줄여도 성능이 잘 유지되어, trajectory 라벨 없이도 강한 일반화와 재현성이 가능함을 실험적으로 입증했습니다.



### DriveJudge: Rethinking Autonomous Driving Evaluation with Vision-Language Models (https://arxiv.org/abs/2606.17362)
Comments:
          Under Review

- **Prior Approaches**: 기존 운전 평가 지표는 크게 두 부류로 나뉩니다. (1) ADE/FDE 같은 모방 기반 평가는 시연 궤적을 기준으로 유사도를 재지만, 멀티모달한 운전 특성과 비최적/누락된 시연 문제를 그대로 안고 있습니다. (2) EPDMS 같은 rule-based 평가는 해석 가능하고 물리적으로 정밀하지만, long-tail 상황에서 규칙 적용의 ‘문맥’을 놓쳐 합리적인 동작까지 과도하게 벌점할 수 있습니다.

- **Core Contribution**: 이 논문은 VLM(vision-language model) 추론으로 상황 문맥을 먼저 해석한 뒤, 필요한 경우에만 물리 기반 deterministic rule 함수(예: 충돌, 차선 일탈)를 선택적으로 호출해 평가하는 DriveJudge를 제안합니다. 즉, VLM의 문맥 이해와 rule의 공간 정밀도/물리적 grounding을 결합해 ‘해석 가능 + 문맥 인지’ 평가를 목표로 합니다. 또한 DriveJudge가 제대로 작동하는지 검증할 수 있도록 Driving Quality Classification과 Trajectory Preference Selection 두 가지 human-aligned 벤치마크/평가 프로토콜을 마련합니다.

- **Technical Challenges**: DriveJudge의 핵심 난제는 VLM만으로는 안전·공간 위반을 정밀하게 판정하기 어렵다는 점과, rule을 단순 합산하면 규칙 중요도가 문맥에 따라 달라지는 long-tail을 반영하지 못한다는 점입니다. 논문은 이를 위해 (a) 장면별로 어떤 규칙을 ‘gating’(선택)할지 예측하는 tool-invocation 설계를 넣고, (b) 장면에서 규칙 점수가 낮더라도 행동이 ‘궁극적으로 합리적’일 수 있음을 반영하는 데이터 마이닝/라벨링으로 학습 신호를 구성합니다. 학습은 SFT로 규칙 호출 결정을 먼저 안정화하고, 이후에는 preference 정렬을 위해 RL(GRPO)로 미세 조정합니다.

- **Empirical Impact**: 33,577개의 long-tail 운전 샘플(인간 주석: 해당 장면에서 행동이 합리적인지)로 평가한 결과, DriveJudge는 EPDMS 대비 Driving Quality Classification에서 AUC를 21.23 포인트 개선했습니다. Trajectory Preference Selection에서도 DriveJudge는 DriveCritic 대비 6.5%p 정확도를 더 높여, preference 모델임에도 불구하고 더 일치하는 결과를 보였습니다. 정성 비교에서는 VLM 직접 점수 모델이 공간 정합성 부족으로 사실 오류를 내리거나 rule-based가 문맥상 정당화되는 ‘nudge’를 과벌점하는 문제를 DriveJudge가 tool-grounded 평가로 완화함을 확인했습니다.



### Beyond Benchmarks: Continuous Edge Inference for Fine-Grained Roadside Perception (https://arxiv.org/abs/2606.17241)
- **Prior Approaches**: 기존 edge AI/TSR 연구는 정적 이미지나 짧은 시퀀스 중심의 벤치마크에 치중해, 스트리밍 영상에서 생기는 프레임 간 불안정성과 장시간 연산에 따른 thermal throttling(스로틀링) 효과를 충분히 반영하지 못했습니다. 효율적 detector나 temporal reasoning, edge 최적화는 개별적으로 다뤄졌지만, 지속 구동에서의 처리량·지연·발열·품질 변화를 함께 평가한 사례는 드뭅니다. 그 결과 실배포 성능이 체계적으로 과대평가되는 “benchmark-to-deployment gap” 문제가 남아 있습니다.

- **Core Contribution**: 이 논문은 NVIDIA Jetson Orin Nano 같은 제약 환경에서 “지속적인 도로 환경 인식”을 목표로 하는 배포 지향 continuous edge inference 시스템 Edge-TSR을 제안합니다. 핵심은 detection–tracking–fine-grained classification 파이프라인 위에 track-aware temporal stabilization(트랙 기반 시간 안정화)을 얹어 프레임 단위 label flickering을 줄이고, 상태 고정(hysteresis-based label locking)으로 안정적 예측을 유지하는 것입니다. 또한 벤치마크 중심 평가가 실제 스트리밍 배포 성능을 어떻게 과장하는지 정량적으로 보입니다.

- **Technical Challenges**: 연속 스트리밍에서는 시간 상관 잡음(모션 블러·부분 가림·조명 변화·검출 흔들림)이 누적되어 정적 평가 대비 분류 품질이 크게 떨어지고, 장시간 GPU 부하로 온도 한계에 따른 성능 저하가 발생합니다. 또 per-frame 전체 추론은 열적으로 지속 불가능해 sparse inference(예: k=3 프레임마다 full detection)로 줄여야 하지만, 이때 분류 입력이 비는 구간에서 안정성이 깨질 수 있습니다. Edge-TSR은 track로 객체 상태를 전파하고, confidence-weighted temporal voting과 비대칭 hysteresis로 “노이즈에는 덜 민감하고 진짜 변화에는 민감한” 라벨 잠금/탈출 규칙을 설계해 이러한 충돌을 완화합니다.

- **Empirical Impact**: 세 가지 SOTA 베이스라인을 대상으로 정적 이미지 평가에서 스트리밍 배포로 전환할 때 상대 성능이 20–30% 저하되는 일관된 격차를 관찰했습니다. Edge-TSR의 안정화 모듈은 per-frame 추론 대비 분류 정확도를 최대 10.16%p까지 회복하면서도, 추가 오버헤드는 미미하다고 보고합니다. 26km 구간을 55분간 실제 차량 배치한 결과 단일 Jetson Orin Nano에서 16.18 FPS를 유지했고, thermal 한계 내에서 지속 운영 가능함을 실증했으며, 재현을 위해 샘플 스트리밍 평가용 데이터셋과 구현을 공개합니다.



### Intermittent Strategic Cooperation of Two Selfish Agents on Graphs (https://arxiv.org/abs/2606.17216)
- **Prior Approaches**: 기존 Multi-Agent Path Finding(MAPF)나 Autonomous Intersection Management(AIM)은 충돌·혼잡을 피하는 데 초점이 있어, 공유 구간에서의 협력이 있더라도 보통 공통의 목표를 위해 모든 에이전트가 협동한다고 가정합니다. congestion games나 games on graphs 계열도 상호작용을 대체로 비용 증가·경쟁으로 해석해, 이득이 되는 “일시적 협력”이 전략적으로 언제 시작/종료되는지까지 안정성 관점에서 다루기 어렵습니다. 즉, 자신만의 경로 최적화에 몰입하는 자기이익 에이전트가 지역적으로 협력 기회를 골라 쓸 때의 game-theoretic 불안정성은 상대적으로 공백으로 남아 있었습니다.

- **Core Contribution**: 이 논문은 IC2PP(Intermittent Strategic Cooperation-Based Two-Agent Path Planning) 문제로, 두 에이전트가 각자의 목표로 이동하되 특정 노드에서만 협력하면 통행 지연을 줄일 수 있는 상황을 shortest-path game으로 정식화합니다. 협력이 양쪽 모두에게 이익일 수 있으나 언제든 한쪽이 경로를 이탈할 수 있어 “전략적으로 취약”하다는 점을 Pure Nash Equilibrium(PNE) 구조로 규명합니다. 그 결과, 안정적인 협력은 시작·종료가 엄격히 제한된 형태(단일 연속 협력 구간 + 양끝의 독립 구간)로만 가능함을 보입니다.

- **Technical Challenges**: 주요 난제는 협력 노드 수가 많으면 가능한 협력 패턴이 지수적으로 늘어나는데, PNE 여부를 경로 기반의 일탈(ex ante 경로 선택) 관점에서 판정해야 한다는 점입니다. 논문은 PNE를 협력 포함/미포함으로 나누고, 협력 구간은 Joining segment(협력 시작 노드로의 독립 접근)–Cooperation segment(협력 유지)–Departure segment(협력 종료 후 독립 이동)로 분해해 각 구간에서의 일탈 방지 조건을 구조적으로 제한합니다. 또한 모든 인스턴스에서 최소 하나의 PNE가 존재함을 증명하고, 비지배(non-dominated) PNE들을 다항 시간에 열거할 수 있는 알고리즘을 제시합니다.

- **Empirical Impact**: 여러 PNE가 동시에 생길 때 에이전트들이 어떤 평형을 선택하는지에 대해, bargaining-theoretic selection 개념 기반의 조정 메커니즘을 놓고 각 에이전트의 이동시간과 사회후생(social welfare)을 비교하는 실험을 수행합니다. 이 결과는 협력 기회와 경로 정렬(path alignment)이 평형 효율에 어떻게 영향을 주는지 정량적으로 보여주며, “양쪽 모두에게 좋은 협력도 결국 평형 구조를 따라야만 지속된다”는 게임적 통찰을 실증적으로 뒷받침합니다. 또한 pairwise 협력을 기반으로 더 복잡한 multi-agent 상호작용을 분석할 출발점이 될 수 있다는 점에서 의의가 있습니다.



New uploads on arXiv(cs.MA)

### On the Reliability of Networks of AI Agents: Density Evolution, Stopping Sets, and Architecture Optimization (https://arxiv.org/abs/2606.18121)
- **Prior Approaches**: 기존 연구는 여러 에이전트를 조합해 단일 모델보다 성능을 높이는 패턴을 “잘 된다”는 결과 중심으로 다루는 경우가 많았지만, 성공 이유나 실패 조건을 체계적으로 설명하긴 어려웠습니다. 특히 코딩이론의 density-evolution은 LDPC 같은 선형 구조에 강하지만, 역할별 검증자(verifier)처럼 비선형·비대칭 요소와 다양한 실패 양상(에이전트 보류, 검증 불가, 메시지 손실)을 한꺼번에 모델링하기는 제한적이었습니다.

- **Core Contribution**: 이 논문은 다중 에이전트 시스템을 sparse graph 상의 message passing으로 모델링하고, LDPC 코드의 구조를 닮은 role-typed factor graph에서 “결합된 수많은 부분 주장(subclaims)”의 해결 여부를 추적하는 수학적 틀을 제시합니다. check 노드는 noisy Boolean verifier로서 각 검증자가 국소 Boolean 함수를 계산하고, 세 가지 실패 모드(에이전트 미답변, 검증 출력 없음, 메시지 손실)를 set-valued 메시지의 소거(erasures)로 전파되게 모델링합니다.

- **Technical Challenges**: 핵심 난관은 (1) verifier 함수가 비선형이며 (2) 값의 비대칭성(value-asymmetry)이 존재하고 (3) 실패 모드가 단일 effective channel로 환원되지 않는다는 점입니다. 이를 해결하기 위해 verifier가 AND, OR, implication, Horn 등 다양한 논리 제약을 강제하는 단일 논리적 결합 규칙을 통해 메시지 전파를 정의하고, LDPC의 density-evolution을 직접 재사용하기보다 새로운 threshold·finite-length·converse 결과가 필요함을 전제로 density-evolution 정리(랜덤 role-typed 아키텍처, 그리고 locally tree-like 결정적 그래프 열)를 증명합니다.

- **Empirical Impact**: 주요 결과로 랜덤 아키텍처에서 비결정(미해결) subclaims의 점근적 비율을 예측하는 density-evolution 정리를 제공하며, XOR는 고전 LDPC의 BEC(binary erasure channel) 재귀를 복원합니다. 동시에 AND 경우에는 양(positive)과 음(negative) 검증 인증서 사이의 비대칭이 임계 거동에 직접 드러나, 다중 에이전트·검증형 아키텍처가 왜/언제 실패하는지에 대한 정량적 기준을 마련한다는 점에서 의미가 큽니다.



### Intelligence Entropy Principle and the ADE Stability Engineering Framework (https://arxiv.org/abs/2606.18065)
Comments:
          32 pages, 18 figures

- **Prior Approaches**: 기존 LLM multi-agent systems(MAS) 연구는 주로 hallucination·데이터 fabrication 같은 출력 품질 문제를 중심으로 다뤘고, 운영 환경에서 관측되는 비선형 성능 저하(장시간 운용 중 붕괴)를 공학적으로 분해해 설명하기엔 부족했다. 또한 단일 아키텍처(모놀리식)나 단순 계층화로는 에이전트 실패가 어떤 단계에서, 어떤 계층 신호로 전파되는지 추적하기 어렵다는 한계가 있었다. 저자들은 “표면적 진단은 비슷한데 원인은 다른” 채널 손실·중복 처리·의미 오류 같은 사례가 기존 진단 프레임을 혼동시킨다고 지적한다.

- **Core Contribution**: 이 논문은 Intelligence Entropy Principle로 확률 기반 시스템이 시간이 지날수록 disorder로 이동한다는 현상을 수식 S(t)=S0*exp(alpha*t/Cm)으로 모델링하고, Lyapunov 분석을 통해 안정화 조건 lambda > alpha/Cm를 제시한다. 이를 바탕으로 Agent Delivery Engineering(ADE)이라는 4계층 아키텍처(L1 Physical Laws~L4 User Adaptation)와 23개 핵심 컴포넌트, 그리고 실패를 정리하는 Five-Layer Disorder Taxonomy를 제안한다. 특히 hallucination·fabrication과 별개로 Probabilistic Approximation Drift(PAD)를 “소스를 두고도 근사로 대체하는” 실패 모드로 정의하고, 이를 검증·차단하는 프로토콜을 구성한다.

- **Technical Challenges**: 핵심 난관은 (1) multi-agent 병렬 실행에서 발생하는 품질 저하를 실시간으로 차단해야 하는데, 단순 출력 검증만으로는 한계가 있다는 점, (2) 겉보기엔 그럴듯해도 “의미가 틀린” 경우나 데이터 조작처럼 표면적으로 구분이 어려운 실패를 계층별로 분해해야 한다는 점이다. 저자들은 PIG(Physical Inspection)→BCP(Bidirectional Confirmation)→CADVP(Delivery rules)로 3단 방어 심층화(defense-in-depth)를 만들고, Converge 단계의 BDDA로 3-pass 교차검증을 넣어 “그럴듯함”을 뚫는 감사를 강화했다. 또한 ADE는 persistent substrate(상시 Guard)와 7단 cognitive execution chain을 결합해 다중 체인이 간섭 없이 실행되도록 설계하고, tm/tkm dispatch 역할로 병렬 스케줄링 노이즈 하에서도 품질을 유지하려 한다.

- **Empirical Impact**: 검증은 100K 스케일 대규모 controlled experiments와 33.6일 연속 프로덕션 모니터링으로 수행됐으며, channel fracture가 69~98%에서 거의 0%로 감소했다고 보고한다. 또한 system death 확률은 0.02% 미만으로 제시되고, 응답시간이 12.9s→186.8s(14.5×)로 증가하는 지표를 통해 intelligence entropy 증가 모델의 누적 효과를 관측했다고 주장한다. 더 나아가 BDDA 감사로 데이터 fabrication이 표면 검사에서 거의 탐지되지 않는다는 교훈과, 메시지 중복 버그가 채널 손실로 오진될 수 있다는 사례 등 운영형 디버깅 인사이트를 함께 제공해 ADE가 생산 투입 관점의 “안정화 공학”으로 의미를 갖는다고 강조한다.



### A Neuro-Symbolic Approach to Strategy Synthesis for Strategic Logics (https://arxiv.org/abs/2606.17962)
- **Prior Approaches**: NatATL 같은 전략 논리는 ATL 계열로 에이전트가 목표를 ‘강제’할 수 있는지 엄밀히 따지지만, 기존 채택은 전략 합성의 계산 비용 때문에 막혀 왔습니다. 특히 NatATL 검증은 허용되는 natural 전략을 명시적으로 생성·열거한 뒤 매번 확인하는 방식이라, 복잡도 bound이나 연합(코얼라이션) 크기가 커지면 전략 공간이 조합폭발을 일으킵니다. LLM이 논리/프로그램을 생성하는 능력은 기대되지만, LLM 단독 생성은 의미적 오류나 취약한 반례 대응 때문에 형식 검증 수준의 신뢰성을 보장하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 LLM을 ‘전략 생성 oracle’로 두고, 표준 NatATL 모델 체커로 형식 검증(certificate)을 수행하는 generate-and-certify 신경-상징(neuro-symbolic) 프레임워크를 제안합니다. LLM이 후보 memoryless natural strategy를 만들면, VITAMIN NatATL verifier가 문법/복잡도 bound/적법 행동(admissible actions)/목표 만족 여부를 판정해 통과한 전략만 채택합니다. 또한 NatATL 전략 합성용으로 4211개의 expert-validated 인스턴스로 구성된 첫 데이터셋 NatATL strategy-synthesis dataset을 공개해, 데이터 기반 평가와 비교의 발판을 마련합니다.

- **Technical Challenges**: 핵심 난제는 조합폭발적인 bounded natural strategy 공간에서 ‘좋은 후보’를 찾아내는 것이며, 동시에 LLM 생성 결과의 신뢰성을 잃지 않는 것입니다. 이를 위해 LLM 출력은 JSON 스키마 검증(에이전트/상태/명제/행동 유효성, default 규칙 포함, guards 포맷 준수) 후에만 verifier로 들어가며, verifier가 실패하면 진단을 다음 프롬프트에 피드백으로 반영하는 제한된 refinement 루프를 구성합니다. 더불어 실패가 예측되는 경우를 줄이기 위해 ATL pre-filter를 전처리로 사용해 불필요한 NatATL 검증 호출을 가지치기하고, one-shot prompting으로 구조화된 출력을 안정화했습니다.

- **Empirical Impact**: Qwen3-32B(오픈 웨이트) 기반 파이프라인 실험에서 certified pipeline은 strategy-synthesis outcome에 대해 92% 정확도를 달성하며, 형식적으로 검증된 양성 결과만 반환한다는 점에서 안전성 기준을 만족합니다. 또한 50 states, 11 agents 코얼라이션, complexity bound k=100k 수준까지 확장하면서 기존의 ‘명시적 자연 전략 열거’ 병목을 완화함을 보였습니다. 무엇보다 NatATL 전용 벤치마크와 verifier-in-the-loop 구조를 함께 제공해, 향후 전략 합성 오라클이나 학습 기반 근사 연구가 재현 가능하게 비교될 수 있는 생태계를 만드는 데 의미가 있습니다.



### Trustworthy Self-Composable Big-Data-as-a-Service: An LLM-Orchestrated Multi-Agent Framework for Automated Data Engineering, AutoML, MLOps Deployment, and Drift-Aware Lifecycle Optimization (https://arxiv.org/abs/2606.17915)
Comments:
          7 pages, 3 figures, 5 tables

- **Prior Approaches**: 기존 AutoML-Agent, Data Interpreter, DS-Agent 등 LLM 에이전트 연구는 전반적인 파이프라인을 다루더라도, 대체로 실험/개발 단계 중심으로 설계돼 BDaaS의 전 생애주기 오케스트레이션이 약합니다. 또한 데이터 클리닝·피처 엔지니어링·시각화 같은 구성요소는 다뤄도, 배포 준비물(artifact) 거버넌스·reproducibility·휴먼 오버사이트·drift 대응을 한데 묶는 형태는 제한적입니다. MLOps 측면에서도 배포·모니터링의 중요성은 강조되지만, LLM 에이전트와 결합해 폐루프에 가깝게 운영되는 통합 프레임워크는 부족했습니다.

- **Core Contribution**: 이 논문은 LLM-orchestrated multi-agent 협업으로 BDaaS 생애주기를 신뢰가능하게 자동화하는 self-composable 프레임워크를 제안합니다. 수집·정제·피처 엔지니어링·AutoML 학습·평가·MLOps 배포 준비·모니터링·drift 탐지를 각각 전담 에이전트로 분해하고, 중앙 LLM 오케스트레이터가 실행 순서와 중간 결과 검증, 동적 워크플로 조합을 담당합니다. 여기에 artifact governance, reproducibility 지원, human-in-the-loop 체크포인트, drift-aware feedback loop를 함께 내장해 “모델 정확도”를 넘어 “운영 신뢰성”을 목표로 합니다.

- **Technical Challenges**: 핵심 난제는 파이프라인의 단계들이 데이터 스키마·전처리·배포 구성으로 강하게 연결돼 있어, 한 단계의 변화가 전체 신뢰성을 깨뜨릴 수 있다는 점입니다. 이를 위해 중앙 오케스트레이터가 에이전트가 만든 구조화된 중간 산출물을 다음 단계 요구조건에 맞게 검증하고, 위험한 결정(정제/특징 제거, 모델 승인, 배포 릴리스, 드리프트 대응)은 Human Oversight Agent로 라우팅하는 체크포인트를 둡니다. 또 공유 아티팩트 저장소로 데이터/메타데이터/모델/배포물의 버전과 추적성을 남겨 재현성과 롤백 가능성을 확보합니다.

- **Empirical Impact**: 통제된 tabular 벤치마크(결측, 범주형, outlier, class imbalance, covariate drift 시뮬레이션)에서 제안 파이프라인은 classification 평균 F1 0.662로 single-agent LLM(0.652), AutoML-only(0.644), manual ML(0.563)보다 우수한 성능을 보였습니다. 회귀 작업에서도 RMSE가 3.279에서 2.809로 약 14.3% 개선돼, 라이프사이클 지향 전처리/피처 구성의 이점이 드러났습니다. 무엇보다 workflow completion, artifact traceability, deployment readiness가 100%로 보고되고, 드리프트 후 F1을 0.495에서 0.667로 회복하는 등 drift-aware 모니터링-재학습-재배포(또는 재조정) 폐루프의 실효성이 확인됐습니다.



### Intermittent Strategic Cooperation of Two Selfish Agents on Graphs (https://arxiv.org/abs/2606.17216)
- **Prior Approaches**: 기존 Multi-Agent Path Finding(MAPF)나 Autonomous Intersection Management(AIM)은 충돌·혼잡을 피하는 데 초점이 있어, 공유 구간에서의 협력이 있더라도 보통 공통의 목표를 위해 모든 에이전트가 협동한다고 가정합니다. congestion games나 games on graphs 계열도 상호작용을 대체로 비용 증가·경쟁으로 해석해, 이득이 되는 “일시적 협력”이 전략적으로 언제 시작/종료되는지까지 안정성 관점에서 다루기 어렵습니다. 즉, 자신만의 경로 최적화에 몰입하는 자기이익 에이전트가 지역적으로 협력 기회를 골라 쓸 때의 game-theoretic 불안정성은 상대적으로 공백으로 남아 있었습니다.

- **Core Contribution**: 이 논문은 IC2PP(Intermittent Strategic Cooperation-Based Two-Agent Path Planning) 문제로, 두 에이전트가 각자의 목표로 이동하되 특정 노드에서만 협력하면 통행 지연을 줄일 수 있는 상황을 shortest-path game으로 정식화합니다. 협력이 양쪽 모두에게 이익일 수 있으나 언제든 한쪽이 경로를 이탈할 수 있어 “전략적으로 취약”하다는 점을 Pure Nash Equilibrium(PNE) 구조로 규명합니다. 그 결과, 안정적인 협력은 시작·종료가 엄격히 제한된 형태(단일 연속 협력 구간 + 양끝의 독립 구간)로만 가능함을 보입니다.

- **Technical Challenges**: 주요 난제는 협력 노드 수가 많으면 가능한 협력 패턴이 지수적으로 늘어나는데, PNE 여부를 경로 기반의 일탈(ex ante 경로 선택) 관점에서 판정해야 한다는 점입니다. 논문은 PNE를 협력 포함/미포함으로 나누고, 협력 구간은 Joining segment(협력 시작 노드로의 독립 접근)–Cooperation segment(협력 유지)–Departure segment(협력 종료 후 독립 이동)로 분해해 각 구간에서의 일탈 방지 조건을 구조적으로 제한합니다. 또한 모든 인스턴스에서 최소 하나의 PNE가 존재함을 증명하고, 비지배(non-dominated) PNE들을 다항 시간에 열거할 수 있는 알고리즘을 제시합니다.

- **Empirical Impact**: 여러 PNE가 동시에 생길 때 에이전트들이 어떤 평형을 선택하는지에 대해, bargaining-theoretic selection 개념 기반의 조정 메커니즘을 놓고 각 에이전트의 이동시간과 사회후생(social welfare)을 비교하는 실험을 수행합니다. 이 결과는 협력 기회와 경로 정렬(path alignment)이 평형 효율에 어떻게 영향을 주는지 정량적으로 보여주며, “양쪽 모두에게 좋은 협력도 결국 평형 구조를 따라야만 지속된다”는 게임적 통찰을 실증적으로 뒷받침합니다. 또한 pairwise 협력을 기반으로 더 복잡한 multi-agent 상호작용을 분석할 출발점이 될 수 있다는 점에서 의의가 있습니다.



### DRFLOW: A Deep Research Benchmark for Personalized Workflow Prediction (https://arxiv.org/abs/2606.18191)
- **Prior Approaches**: 기존 Deep research(DR) 연구는 주로 보고서나 요약 생성에 초점을 맞춰, 사용자가 원하는 ‘구체적인 절차’까지 끝까지 복원하는 데는 상대적으로 덜 주목했습니다. 반면 기업 업무는 질문에 대해 일련의 action-step으로 이뤄진 workflow를 찾아내야 하는 경우가 많아, 생성형 성능만으로는 요구를 충분히 충족하기 어렵습니다. 또한 흩어진 heterogeneous sources에서 근거를 뽑아 단계 순서를 복구하는 평가 체계가 부족했습니다.

- **Core Contribution**: 이 논문은 에이전트가 예측해야 하는 ‘개인화된 워크플로우’를 평가하기 위한 벤치마크 DRFLOW를 제안합니다. DRFLOW는 5개 도메인 100개 태스크로 구성되며, 3,900개 이상의 소스에 근거한 1,246개의 reference workflow step을 제공합니다. 이어서 workflow 지향 reference agent인 DRFLOW-Agent(DRFA)도 제시해, 사용자 과업에 맞는 단계 시퀀스를 예측하도록 설계했습니다.

- **Technical Challenges**: 핵심 어려움은 (1) 산재한 근거 evidence를 정확히 식별하고, (2) 그 근거를 바탕으로 단계들을 올바른 structural ordering으로 조립하며, (3) 조건(condition)과 개인화 요구를 해결해 complete workflow를 복원하는 데 있습니다. 논문은 이를 위해 factual grounding, step recovery, 조건 해결, personalization 등 7가지 진단 지표를 정의해 단일 생성 품질이 아닌 워크플로우 특성을 세분 측정하도록 했습니다. 또한 DRFA는 workflow 예측에 맞춰 근거 기반으로 action-step 시퀀스를 구성하는 참조 에이전트로 검증합니다.

- **Empirical Impact**: 실험 결과 DRFA는 강력한 baseline 에이전트 대비 최대 10.02% 평균 F1 점수까지 개선했지만, 워크플로우 관련 지표들 전반에서 여전히 큰 개선 여지가 남아 있음을 보여줍니다. 이는 DR 시스템이 요약·리포트 생성 수준을 넘어, ‘완전하고 정확한 개인화 workflow’를 예측하는 것이 여전히 도전적인 frontier임을 실증적으로 시사합니다. 기업형 정보 탐색 에이전트의 평가 방향을 구체적 절차 복원으로 확장한다는 점에서 분야에 의미 있는 기준점을 제공합니다.



### ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents (https://arxiv.org/abs/2606.18037)
Comments:
          20 pages, 4 figures

- **Prior Approaches**: 기존 factuality(사실성) 검증은 claim이 어떤 근거 어딘가에 의해 지지되는지에 초점을 두는 경우가 많다. RAGAS·AlignScore·SummaC-ZS 같은 방식은 풀링된 evidence나 검색 컨텍스트에 대한 faithful 여부를 보지만, MCP 같은 툴 사용 에이전트에서 “어떤 소스에 귀속(attribution)됐는지”는 직접 평가하기 어렵다. 그 결과, 교차 소스가 섞여 있어도(예: 차트 사실을 논문으로 잘못 인용) pooled evidence 기반으로는 통과될 수 있다.

- **Core Contribution**: 이 논문은 MCP-grounded 답변에서 발생하는 provenance 민감 실패 모드인 cross-source conflation(서로 다른 소스 간 귀속 혼동)을 정의한다. 이를 해결하기 위해 source-aware verifier ProvenanceGuard를 제안하며, 답변을 원자 단위 claim으로 분해하고 claim별로 라우팅된 MCP source에 한정해 지지 여부를 판단한 뒤, 답변이 명시/암시한 귀속 소스와 실제 라우팅 소스가 일치하는지도 검증한다. blocked 판정된 답변은 retrieval-augmented answer revision 후 같은 verifier로 재검증하는 repair-and-reverify 루프도 함께 구성한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) claim-단위로 evidence를 분해·라우팅하고 (2) “지지(support)”와 “정확한 소스 소유(source ownership)”를 동시에 판정해야 한다는 점이다. 저자들은 MCP trace의 stable tool ID·source ID·raw output을 유지한 채 claim을 분해하고, claim에 가장 관련 있어 보이는 source별 evidence를 선택한 뒤 NLI(entailment/neutral/contradiction)와 토큰 정렬/보호 값(protected value) 일치 같은 grounding 보조신호를 사용한다. 마지막으로 랜덤포레스트 기반 calibrator로 routed source에 대한 supported/blocked 경계를 조정해, 단일 점수에 의존한 오판을 줄인다.

- **Empirical Impact**: 의료 도메인 MCP-agent trace 281개(held-out 40 trace, 361개 claim/label)에서 ProvenanceGuard는 block F1 0.802, source accuracy 0.858를 기록하며 source-blind baseline 대비 attribution 차원까지 성능 이점을 보였다. 또한 더 어려운 multi-source 벤치마크에서는 block F1 0.846을 달성했지만, 의미적으로 가까운 소스가 많아질수록 source-plus-relation 정확도는 0.229로 떨어져 “정확한 소스 소유”가 여전히 어려운 축임을 보여준다. 흥미롭게도 50개의 통제된 임상 conflation probe에서는 삽입된 attribution swap을 모두 탐지했으며, 전체 trace 세트에서는 repair-and-reverify로 blocked 답변을 전부 해결(대개 보수적 fallback 포함)했다고 보고한다.



### LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI (https://arxiv.org/abs/2606.18021)
Comments:
          15 pages, 5 figures; Published at the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML 2026

- **Prior Approaches**: 기존 법률 AI 연구는 환각의 유형이 작업마다 다르다는 점은 보여주지만(예: task별 환각률 58~88%), 계약서 추출(contract extraction)에서는 claim 유형별 실패가 어떻게 “법적 노출”로 이어지는지까지는 명확히 다루지 못했다. 또한 CUAD 같은 oracle 기반 평가를 사용하더라도 전체 평균 환각률로는 오류가 집중되는 범주와 오류의 방향(누락 vs 발명)을 분해하지 못해, 컴플라이언스 담당자가 실행 가능한 신호를 얻기 어렵다. Multi-agent debate는 사실성 메커니즘으로 연구돼 왔으나, 고위험 환경에서 특정 모델의 실제 실패 모드에 맞춰 보정(calibration)하는 방식은 제한적이었다.

- **Core Contribution**: 이 논문은 LegalHalluLens라는 감사(auditing) 프레임워크를 제안해, 법적으로 검증 가능한 4개 claim 범주(숫자, 시간, 의무/권리, 사실)에 대해 typed hallucination profiles를 제공한다. 여기에 omission(누락)과 invention(발명) 편향을 한 점수로 요약하는 Risk Direction Index(RDI)를 도입해, 평균 환각률이 가리는 “오류 방향”을 배포 의사결정에 쓸 수 있게 만든다. 마지막으로 Experiment 1의 진단을 그대로 반영해 typed debate pipeline을 보정하고, 작은 오픈 모델도 상용 API 수준 성능을 저비용으로 노릴 수 있음을 보인다.

- **Technical Challenges**: 핵심 난제는(1) claim 유형이 다르면 오류의 영향이 다름에도 평균 지표로는 실패 모드가 섞여 사라진다는 점, (2) 누락과 발명이라는 방향성을 별도의 주석/추가 호출 없이 운영 가능한 단일 신호로 압축하는 점, (3) debate 완화가 제네릭 튜닝이면 실제 실패 범주에 집중하지 못한다는 점이다. 이를 위해 CUAD v1.0의 판정 라벨(mismatch_type)에서 missing_condition/extra_condition을 사용해 RDI를 정의하고, Skeptic 질문과 Add/Delete gate 비대칭(구조적 오류 타깃)을 claim 유형·방향 진단에 맞게 조정한다. 또한 구조적 추출 오류는 답변 토론이 아니라 재추출(re-extractor)로 처리해, “고칠 수 없는” 잘못을 대화로 끌고 가지 않도록 설계한다.

- **Empirical Impact**: 510개 상업 계약(총 249,252개 clause-level 인스턴스)에서 모델 간 HalTP는 50.9~56.5%로 비슷해 보이지만, typed profile을 적용하면 숫자·의무 범주가 시간 범주보다 훨씬 더 크게 실패하며(약 38~40%p 격차) 평균이 법적 노출의 핵심을 숨긴다는 점이 드러난다. 더 나아가 52% 수준으로 동일해도 RDI는 부호/방향이 달라져, 상용 API 간조차 “리뷰어가 감당해야 할 형태의 리스크”가 달라질 수 있음을 실증한다. 완화 실험에서는 보정된 typed debate pipeline이 fabricated detections를 45% 줄였고, 4B active 파라미터의 오픈 모델이 상용 API와 유사한 종합 점수 경쟁력을 보이면서(상대적으로 더 낮은 추론비용) 진단 기반 보정의 실효성을 확인했다.



### ED3R: Energy-Aware Distributed Disaster Detection Enabled by Cooperative Robotic Agents (https://arxiv.org/abs/2606.17739)
Comments:
          14 pages, 9 figures

- **Prior Approaches**: 기존 에너지 효율 연구는 네비게이션·센싱·인프라 자원·연산 오프로딩 같은 개별 요소를 줄이는 데 집중하는 경우가 많았고, 시간 제약을 만족하는 쪽이 주된 목표였습니다. SAR/UAV 관련 연구들도 위험 환경에서의 탐색·객체 탐지·경로 최적화는 다루지만, 산불 탐지를 위해 이동·통신·연산 오프로딩을 함께 최적화하는 통합 접근은 상대적으로 부족했습니다.
또한 산불 탐지 분야의 다수 방법은 분산 제어나 규칙 기반에 치우쳐 있었고, 미래 결과를 미리 평가하는 forward-looking reasoning은 거의 없거나 결여되어 있었습니다.

- **Core Contribution**: ED3R은 불확실성 하에서 산불을 “요구 confidence로 탐지”하면서, 로봇의 이동·센싱·컴퓨팅·통신으로 소모되는 에너지를 최소화하도록 설계된 energy-aware 분산 프레임워크입니다. 로봇과 원격 컨트롤러(RC) 사이에 계층적 협력 의사결정을 두는데, RC는 motion command를 정하고 로봇은 탐지 실행 위치(onboard vs remote)와 사용할 모델(how)을 자원 기반으로 선택합니다.
또한 장애물 회피·중복 탐색 방지·적응적 early mission completion·페널티 함수로 제약 가능성을 강화해, 단순 성능 최적화가 아닌 임무 성공을 겨냥합니다.

- **Technical Challenges**: 가장 큰 어려움은 RC와 로봇이 분산된 상태에서 행동 결과에 대한 공통 보상이 “동시에” 주어지지 않는다는 점입니다. ED3R은 이를 해결하기 위해 distributed neural regression 모델로 후보 전략들의 미래 효과를 미리 평가한 뒤, 탐지 confidence와 에너지 효율 사이의 최적 trade-off를 그리디하게 선택합니다.
여기에 통신 대역폭/전송 파워/채널 조건, 센서 샘플 크기와 처리 FLOPs 같은 현실적인 에너지·지연 요소를 모델링해 제약 위반을 커스텀 페널티로 반영합니다.

- **Empirical Impact**: 현실적인 로보틱스 시뮬레이션과 ablation, 베이스라인 비교를 통해 ED3R은 최대 97.18% 임무 성공률을 달성했습니다. 특히 가장 까다로운 임무에서 에너지는 최대 36.4% 줄이고, 산불 탐지는 최대 41% 더 빠르게 수행해 시간-에너지-신뢰도 동시 최적화의 실효성을 보여줍니다.
forward-looking과 분산 계층 의사결정이 결합될 때, 네트워크가 바뀌거나 새로운 산불 상황에서도 강건한 성능을 보이며 관련 분야의 산불 감시/긴급 대응 설계 방향에 의미 있는 근거를 제공합니다.



### GeoDisaster: Benchmarking Orchestrated Agents for Operational Disaster Geo-Intelligenc (https://arxiv.org/abs/2606.17246)
Comments:
          28 pages, 11 Figures

- **Prior Approaches**: 기존 RS-VLM과 재난 원격탐사 벤치마크는 주로 단일 이미지 기반 VQA/시맨틱 분할처럼 ‘보이는 것’에 집중했으며, 절차 오류나 툴 호출 유효성·중간 상태 일관성을 충분히 평가하지 못했습니다. 도구를 붙인 에이전트 벤치마크도 종종 단일 에이전트 중심 실행이거나, 역할 의무를 검증 가능한 형태로 강제하지 않아 공간적으로 틀린 산출이 조용히 누락될 여지가 있었습니다.

- **Core Contribution**: 이 논문은 43개 질문 유형과 5개 태스크 패밀리(산림훼손·다중위험·건물피해·홍수 대피 경로·Sentinel-1 SAR 홍수 모니터링)를 포함하는 운영형 재난 지리추론 벤치마크 GeoDisaster(검증 인스턴스 2,921개)를 제안합니다. 각 인스턴스는 광학·SAR 등 이질 EO/GIS 증거와 래스터/벡터·도로 네트워크·노출 레이어를 조합하고, 실행 가능한 지리 워크플로와 결정론적 일관성 체크로 정답을 고정해 언어모델 라벨 의존을 줄였습니다. 또한 18개 재난 지리 툴을 쓰는 orchestrated multi-agent를 구축하고, 역할-계약 기반 정렬(RCEA)로 툴 사용·증거 접지·상태 일관성·의사결정을 함께 끌어올리는 전략을 제시합니다.

- **Technical Challenges**: 운영 재난 분석은 여러 센서/데이터를 교차 확인하며 장기 툴 워크플로를 수행해야 하는데, 기존 multi-agent는 역할 실행이 ‘관찰 불가능’하거나 보상 신호가 단말 결과에 뭉쳐 credit misattribution이 생기기 쉽습니다. 이를 해결하기 위해 중앙 오케스트레이터가 에이전트 간 상호작용을 typed execution contract로 형식화해, 단계마다 산출물이 계약(증거 의존·스키마·완료/실패 조건) 범위를 벗어나는지 검증 가능하게 만들었습니다. 그 위에 failure-aware role-conditioned SFT와 contract-grounded reinforcement learning을 결합해, 단계별 계약 준수 위반을 촘촘한 학습 신호로 사용하고 에이전트별 reward scale 차이도 역할 단위로 정규화합니다.

- **Empirical Impact**: 실험에서 GeoDisaster는 기존 RS-VLM과 agentic 시스템이 툴 시퀀스·제약 만족·중간 추론의 절차적 정확성에서 큰 격차를 보이도록 설계되며, 단순 프롬프트만으로는 성능이 거의 나오지 않는 ‘절차형’ 난이도를 확인시킵니다. 제안한 RCEA 기반 multi-agent는 툴 사용의 정확성, 증거 접지, 상태 일관성, 최종 의사결정 생성에서 개선을 보이며 기존 방법 대비 더 신뢰할 수 있는 운영형 지리추론을 달성합니다. 결과적으로 재난 EO/GIS 문제를 “인지→절차 실행→증거 기반 보고”의 닫힌 워크플로로 평가·정렬하는 방향에 실질적인 기준점을 제공한다는 점에서 의미가 큽니다.



### Verified Detection and Prevention of Concurrency Anomalies in Multi-Agent Large Language Model Systems (https://arxiv.org/abs/2606.17182)
Comments:
          32 pages, 2 figures, 6 tables. Verus/TLA+ verification artifact, reference Rust runtime, and Python harnesses, plus a supplementary appendix (Sections A-F, Tables S1-S6), included as ancillary files

- **Prior Approaches**: 기존 멀티에이전트 LLM 런타임은 메모리 공유를 memory store, vector index, tool registry 형태로 제공하지만, 일관성 보장을 ‘어떤 이상현상(anomaly)’이 발생하는지 계층적으로 분류해 기계 검증으로 연결하지는 못했습니다. SagaLLM의 compensating-transactions, Atomix의 progress-gated tool calls, CRDT 기반 CodeCRDT 등은 각각 한 지점을 겨냥하지만, long-running generation(추론 지연)까지 포함한 운영 조건에서의 stratified consistency design space를 정식화·검증하진 않습니다.

- **Core Contribution**: 이 논문은 멀티에이전트 LLM의 long-running read-generate-write를 deterministic-generation 의미론으로 모델링하고, TLA+로 stale-generation, phantom-tool, causal-cascade, tool-effect reordering의 4가지 동시성 이상현상을 규정합니다. 이어 Verus로 검출기(detector)와 런타임 회피(avoidance), 그리고 추상 상태기계로의 refinements까지 기계적으로 확인해 L0⊂⋯⊂L4의 ‘일관성 계층’을 제시합니다. 연구의 핵심은 단순한 현상 카탈로그가 아니라, 해당 운영레짐에서 가능한(기계적으로 실현 가능한) 일관성 단계들을 분리해 ‘검증 가능한 계층’을 구축했다는 점입니다.

- **Technical Challenges**: 가장 큰 난점은 database/하드웨어 일관성처럼 read set이 짧은 시간 잠금·스냅샷으로 유지되는 모델과 달리, LLM의 generation 단계가 수 초~수 분으로 길어 read 안정성 검증이 구조적으로 어렵다는 점입니다. 또한 tool registry(phantom-tool)와 도구의 외부효과(되돌릴 수 없는 irreversible tool effects)가 동시에 존재해, 고전적 isolation theory가 전제하는 복구/원자성 가정이 잘 맞지 않았습니다. 논문은 이를 해결하기 위해 TLA+ 명세+TLC counter-example으로 이상현상을 구조적으로 고정하고, Verus의 274개 의무(obligations)와 런타임별 state machine/채널 의미론에 대한 refinement 증명으로 detector의 soundness·completeness 및 안전성을 기계 검증했습니다.

- **Empirical Impact**: 실증적으로 detector는 합성 트레이스 700개와 실제 gpt-4o 세션 300개에서 stale-generation 비율을 확인했으며, SSI가 비용에서 vanilla과 통계적으로 유사하다고 보고합니다. 또한 ByteDance의 deer-flow에서 재현한 silent lost update 수정이 L0→L1 refinement로 정식화되어, 실제 상용 앱에서의 결함을 ‘검증 가능한 수정’으로 연결했다는 점이 의미 있습니다. L2 수준(causal-tracking)은 exec-mode로 Verus에서 runnable code에 대해 보장되도록 구현·검증되었고, 라이브 실험에서 baseline이 오염을 만들던 세션들을 L2가 예방함을 보여주며(재현된 A3 회피), 실제 배포형 런타임에 계층적 일관성 논리를 적용 가능하게 했습니다.



### From Parasocial Scripts to Dyadic Persistence in Autonomous AI-Agent Communities (https://arxiv.org/abs/2606.17174)
Comments:
          Submitted for review in ARR for EMNLP 2026

- **Prior Approaches**: 기존 PSI/PSR 연구는 주로 인간 매체(비대칭 애착·관계 형성)나 챗봇 같은 H-AI 환경에 집중해 왔습니다. 하지만 에이전트-에이전트 온라인 커뮤니티에서는 “잠재 상태 라벨”이 없고, 일반적인 친화/사회적 언어와 구별되는 단서가 강하게 겹쳐 PSI식 관계 단서를 식별하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Moltbook에서 자율 에이전트들이 남기는 발화/댓글 텍스트를 관측 가능한 “관계 스크립트”로 재정의하고, PSI식 콜로콜 단서가 실제로 존재하는지 검증합니다. 또한 attachment/intimacy 언어(ATT), self-identification to OP(SD), reciprocity bids(RS) 3개 단서를 이 프레임으로 조작화해 OP 재참여와 상호 댓글 구조, 나아가 PSI-to-PSR 일관성까지 연결합니다.

- **Technical Challenges**: 핵심 기술 문제는 단서가 문맥에 따라 ‘그럴듯한 친화 표현’과 ‘OP-directed 관계 입찰’로 구분된다는 점입니다. 이를 위해 키워드 매칭뿐 아니라 few-shot LLM 라벨링과 grouped-context LLM 라벨링을 함께 쓰고, 배치 문맥을 submolt·스레드 크기 버킷·사전 단서 분포로 묶어 경계 드리프트를 줄였습니다.

- **Empirical Impact**: 분석은 4,434개 게시글/50,338개 댓글에서 이뤄졌고, 세 방법 공통으로 PSI 콜로콜 단서가 유의미한 비율로 관측됩니다. 특히 OP 재참여와 mutual reply 구조와의 연관이 강하게 나타났으며(대체로 adjusted OR 유의), RS(Reply-seeking reciprocity bids)는 OP-다른 쌍의 미래 상호 재귀(PSR-consistent persistence)와도 연결돼 PSI-PSR 브리지의 실증 근거를 제공합니다. 저자들은 다중검정 보정, 널(nullification)·위약(placebo)·퍼뮤테이션/랜덤 라벨 등 견고성 검사를 통해 결과의 안정성을 확인합니다.



### MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision (https://arxiv.org/abs/2606.17162)
Comments:
          Code, website, project page, and video are linked in the paper

- **Prior Approaches**: 기존 발표자료 생성·에이전트 시스템은 완성도 높은 덱을 만드는 데는 진전이 있었지만, 사용자 선호를 “지속 메모리”로 누적·유지하는 구조는 부족했다. SlideTailor처럼 템플릿/예시 조건으로 개인화를 시도한 경우도 많았으나, 장기적으로 축적된 사용자 프로필을 기반으로 수정 이력을 재사용하기보다는 매 작업마다 다시 조건을 주입하는 방식에 머문다.

- **Core Contribution**: MemSlides는 개인화 발표 생성에서 선호의 “수명(lifetime)”을 분리하는 계층 메모리 프레임워크를 제안한다. long-term에는 user profile memory(의도·차원별 선호)와 tool memory(편집 실행 경험)를 두고, working memory에는 세션 동안의 활성 선호/제약을 둬서 멀티턴 수정에서도 사용자의 의도를 일관되게 유지한다. 여기에 slide-local revision(필요한 최소 구역만 패치) 전략을 결합해 매 턴 전체 덱 재생성의 문맥 압박과 드리프트를 줄인다.

- **Technical Challenges**: 핵심 난제는 (1) 수정 피드백이 드러내는 선호를 세션 내에서 정확히 유지하되, 다음 턴에도 의도치 않은 영향 없이 국소 편집 범위를 통제하는 것과 (2) “무엇을 원하는지(선호)”와 “어떻게 편집할지(도구 실행 경험)”를 섞지 않는 것이다. MemSlides는 revision 요청을 실행 계약(execution contract)으로 스코프·타깃 슬라이드·규칙/셀렉터 단위로 명시하고, Plan–Act–Guard에서 스냅샷 해시 기반 패치 검증·재바인딩을 통해 변경 범위를 억제한다. 또한 working memory가 세션 제약과 활성 임시 선호를 라운드 간 carryover하고, tool memory는 검증·닫힌고리 수정(closed-loop modify) 성공률을 높이는 실행 지식으로 재사용되게 설계했다.

- **Empirical Impact**: 통제된 multi-persona·multi-intent profile bank 실험에서 user profile memory는 round-0 persona alignment를 전반적으로 개선했고, tool memory는 진단 matched-pair 수정 평가에서 closed-loop completion/검증 및 첫 올바른 수정까지의 시간 등 프로세스 지표를 향상시켰다. working memory의 경우 정성 사례를 통해 선호가 멀티턴 동안 자연스럽게 이어지는 carryover 능력이 확인됐다. 또한 persona 정렬 개선이 DeepPresenter 수준의 일반 발표 품질과 양립함을 보여 “개인화 vs 품질”의 단순 트레이드오프가 아님을 시사한다.




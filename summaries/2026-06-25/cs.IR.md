New uploads on arXiv(cs.CL)

### Real-Time Voice AI Hears but Does Not Listen (https://arxiv.org/abs/2606.26083)
- **Prior Approaches**: 기존 음성·오디오 모델 연구는 보통 녹음 1개를 넣고 텍스트로 답을 뽑은 뒤, 어떤 속성이 단어(전사) 쪽에서 오거나 목소리(운율) 쪽에서 오는지 주로 라벨 정확도로 평가해 왔습니다. 특히 단어와 음성 특징이 충돌할 때(예: 비꼼, 감정, 부가정보) 모델이 목소리 단서를 잘 쓰지 못하는 경향이 반복 관찰됐습니다.

- **Core Contribution**: 이 논문은 realtime voice 시스템에서 “인식(perception)”과 “행동(action)”이 분리되는지 검증합니다. 텍스트로의 전사 뒤 추론하는 cascaded 파이프라인과 달리, turn-by-turn으로 말입력을 받고 말출력을 하는 4개 상용급 realtime 프로덕션 모델을 대상으로 단어와 전달(crying, frightened, sarcastic 등)이 상충하는 상황의 의사결정 결과를 직접 측정했습니다.

- **Technical Challenges**: 핵심 기술적 난관은 비암묵 정보인 non-lexical channel(피치, 톤, 억양, 감정 상태 등)이 행동 결정에 실제로 반영되는지, 단순히 전사 기반 출력이 잘 만들어졌는지와 구분하는 평가 설계입니다. 저자들은 동일 문구를 유지한 채 전달만 바꾼 scenario 기반 다턴 과업과, 음성만 보고 스스로 어떤 전달을 “들었다”고 답하는 single-turn 진단을 함께 수행해 인식-행동 불일치를 분리해 냈습니다.

- **Empirical Impact**: 결과적으로 4개 시스템은 위급 콜백에서는 울면서도 “문제없다”는 단어를 따라 통화를 종료하고, 송금 사기 점검에서도 두려운 목소리의 의무확장 신호를 무시하거나 감정 일치 없이 단어 기반 승인/등록을 반복했습니다. 다만 직접 질문했을 때는 대체로 distress/fear/sarcasm를 인식하는 것처럼 보여, emotional intelligence gap of voice AI라는 ‘인식은 하지만 행동에 반영하지 않는’ 균열이 관찰됩니다. 저자들은 tone과 emotion이 의사결정에 중요해지는 의료·보안·거래 환경에서는 현재 realtime voice AI를 신중히 사용해야 한다고 경고합니다.



### Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models (https://arxiv.org/abs/2606.26079)
Comments:
          22 pages, 4 figures, 5 tables

- **Prior Approaches**: 기존 MLLM 벤치마크는 각 항목을 한 가지 canonical ordering으로만 채점해, 의미적으로 같은 입력 순열(shuffling)에서 정답이 바뀌는지(순서-불변성) 신뢰도를 놓치곤 했습니다. 관련 연구는 텍스트에서 옵션 순서나 프롬프트 형식 등 단일 편향을 다뤘지만, 멀티모달에서 여러 ordering facet을 분리해 교차-순서 불안정성을 분해·정량화한 평가는 거의 없었습니다.

- **Core Contribution**: 이 논문은 Facet-Probe라는 5개 facet(옵션, evidence-chunk, document-rank, image-set, mixed-modality order)으로 18개(프런티어/오픈웨이트) MLLM을 12개 데이터셋에서 감사해, 순서 perturbation에 따른 답변 뒤집힘을 체계적으로 측정합니다. 또한 ODI(Ordering-Decomposed Item-Response Theory)와 same-ordering control로, ordering noise(무작위성)와 facet별 systematic bias(체계적 편향)를 분리하고 same-input decoder-stochastic 플립 바닥선도 추정합니다.

- **Technical Challenges**: 핵심 난제는 “플립이 발생한 것”만으로는 원인이 ordering 때문인지, 디코더의 확률적 잡음 때문인지 구분이 어렵다는 점이었습니다. 논문은 각 (facet, dataset, item) 셀에서 여러 순열을 샘플링해 flip-rate를 측정하고, same-ordering 통제로 decoder-noise floor를 벗겨 ordering excess를 추정하는 Bayesian 2PL 계층 모형으로 이를 해결합니다. 아울러 mixed-modality는 LLM judge로 정답 일치도를 정의해 불완전한 표면형 답변 변동을 보정하려고 했습니다.

- **Empirical Impact**: 결과적으로 18개 모델 모두 order-invariant가 아니며, facet별 패널 평균 flip rate가 24–50% 범위로 관측됩니다(예: mixed-modality-order 0.50, evidence-chunk-order 0.41, option-order 0.36, document-rank-order 0.26, screened image-set-order 0.24). Gemini의 same-ordering control(temperature 0)에서는 decoder-stochastic 바닥선 대비 ordering excess가 상당하며, 능력(capability)이 플립을 줄이긴 해도 완전히 제거하지는 못해 최선 모델도 13.4%의 trial에서 뒤집힙니다. 학습 없이 prompt 변경만으로는 modality-conditional 하면서도 텍스트→시각 추론으로 전이되지 않아, 주문 수준 완화만으로 일반적인 순서 강건성을 기대하기 어렵다는 시사점을 제공합니다.



### When Certainty Is an Artifact: Keyword Lexicon Blindness and the (Mis)Measurement of Rhetorical Stanc (https://arxiv.org/abs/2606.26062)
Comments:
          16 pages, 2 figures

- **Prior Approaches**: 기존 computational social science에서는 감성/모달리티를 VADER, LIWC, 도메인 렉시콘처럼 단어 기반으로 측정하거나, BERT·RoBERTa 계열처럼 단일 스칼라 점수를 학습해 추정하는 방식이 주로 쓰여 왔다. 그러나 이런 방법은 문맥을 무시하거나(lexicon) 길이·형식 같은 스타일 요인과 혼재되기 쉬워, valence(내용의 긍·부정)와 modality(확신 대 유보)를 독립적으로 분리하기 어렵다는 한계가 반복해서 지적돼 왔다.

- **Core Contribution**: 이 논문은 ‘부정(neg) 담화가 확신(emphatic) 표현과 함께 나타난다’는 계산사회과학의 통계적 결과가, 측정 도구 자체(특히 keyword lexicon)의 산물일 수 있음을 설득력 있게 보여준다. 4명의 public intellectual 인터뷰(총 32,625 문장)에서 렉시콘 기반으로는 neg–emphatic 공변이 매우 크게 관측되지만, LLM의 zero-shot 문장 단위 이중 분류로 바꾸면 상관이 급격히 붕괴하거나 방향이 뒤집힌다.

- **Technical Challenges**: 핵심 기술적 난제는 bag-of-words가 negation(부정), polysemy(다의성), modifier scope(수식 범위)를 해소하지 못해 ‘단어 동시출현’이 ‘진짜 확신/유보’와 다르게 매핑될 수 있다는 점이다. 논문은 이를 검증하기 위해 (1) 문장 전체를 diarized corpus에서 LLM으로 valence·modality를 동시에 라벨링하고, (2) keyword 기반과 동일한 월별 상관을 비교하며, (3) syntactic blindness, polysemy blindness, categorical absence라는 오류 유형이 키워드 카운팅을 어떻게 의미 역전으로 이끄는지 문장 사례로 추적한다.

- **Empirical Impact**: 실험 결과, Dalio의 경우 keyword scoring에서 r(neg, emphatic)=0.851이 LLM 분류에서는 r=0.206(비유의)으로 크게 약화됐고, 전체적으로는 LLM이 ‘부정 담화-유보(hedging) 결합’이라는 기존 기대 패턴을 더 잘 회복했다. Rogoff와 Zeihan에서는 각각 r(neg, hedged)=0.875(p=0.001), 0.722(p=0.008)이 유의하게 나타나 키워드 방식이 실제 stance를 체계적으로 반대로 읽을 수 있음을 시사한다.



### AI translation of literary texts is "fine", but readers still prefer human translations (https://arxiv.org/abs/2606.26040)
Comments:
          58 pages, including appendices

- **Prior Approaches**: 기존 문학 번역 평가는 주로 fluency(유창성)·adequacy(적절성) 같은 단일 요소나 작은 단위 품질에 초점을 맞추며, 독자가 느끼는 몰입감·문학적 효과는 충분히 반영되지 못했다. 또한 자동 평가지표(머신 번역 메트릭)와 인간 평가도 세그먼트 단위에서 끝나는 경우가 많아, 긴 텍스트를 읽는 실제 경험과의 연결성이 약하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 독자 중심의 평가가 왜 필요한지 직접 묻기 위해, 15명의 열성 독자가 프랑스어/폴란드어/일본어→영어 문학 번역 15개를 대상으로 HT(전문 인간 번역)와 agentic LLM 기반 MT(에이전틱 LLM 파이프라인 번역)를 비교하도록 설계했다. 독자가 선호하는 이유(어떤 문장/표현이 좋거나 나쁜지, 어떤 부분에서 MT로 의심하는지)까지 함께 수집하며, 이를 기반으로 LAIT(Literary AI Translation) 데이터셋과 평가 프로토콜을 공개한다.

- **Technical Challenges**: 기여의 핵심은 긴 발췌문 ‘몰입 읽기’와 300단어 청크 ‘근접 읽기’를 분리해, 독자가 체감하는 문학적 효과와 국소 문장 선택의 영향을 동시에 측정하는 평가 설계에 있다. 또한 MT 파이프라인 선택을 위해 여러 agentic 구성과 프롬프트/청크 전략을 16권 개발 세트에서 블라인드 선호로 비교했으며, 번역 청크 정렬과 하이라이트 근거 수집을 위해 span-level 정교한 주석 절차를 마련했다.

- **Empirical Impact**: 결과적으로 독자들은 MT가 ‘괜찮다’고 느끼지만, HT를 선호하는 경향이 뚜렷했으며 그 차이는 근접 읽기에서 더 크게 나타났다(몰입 읽기 30개 비교 중 19회 vs 근접 읽기 772개 비교 중 522회). 특히 HT보다 MT가 한 작품 안에서도 청크별 품질 변동이 더 커서, 독자 하이라이트와 부정 근거 스팬의 밀도에서도 HT가 더 긍정적 신호를 제공했고 MT는 부정 신호가 더 집중되는 패턴이 확인됐다. 더불어 독자들은 두 버전을 안정적으로 구별하지 못했고(정답률 17/30 수준), LLM-as-a-judge를 포함한 자동 메트릭은 독자 선호를 복구하지 못해 ‘문학 효과’ 평가의 격차를 보여주었으며, LAIT 공개로 후속 연구의 재현성과 비교 가능성을 높였다.



### Detect, Unlearn, Restore: Defending Text Summarization Models Against Data Poisoning (https://arxiv.org/abs/2606.26036)
- **Prior Approaches**: 기존 연구는 주로 추론 시점 adversarial perturbation 대응, 입력 정화, 퍼플렉시티 기반 탐지처럼 “테스트 데이터가 깨지는” 상황을 다루는 데 집중해 왔습니다. 백도어 탐지도 주로 trigger 기반 분류 실패를 전제로 하며, 생성형 요약에서 ROUGE는 유지하면서 행동이 미묘하게 표류하는 training-time poisoning에는 직접 적용이 어렵습니다. 또한 unlearning이나 influence functions는 분류·라벨 기반 문제에 주로 쓰여 왔고, 요약 모델에서의 지속적 행동 복구를 체계적으로 검증한 방어는 부족했습니다.

- **Core Contribution**: 이 논문은 요약 모델에서 fine-tuning 단계 poisoning을 “탐지 + 사후 복구”까지 한 번에 다루는 unified post-hoc 방어 프레임워크를 제안합니다. 데이터가 확보된 white-box에선 Defense-1이 고영향(poisoned일 가능성 높은) 샘플을 찾아 gradient-ascent unlearning으로 제거하고, 데이터 접근이 없는 black-box에선 Defense-2가 모델 민감도(SAP)로 poisoning 흔적을 감사(auditing)합니다. 또한 기존 sentiment/toxicity 중심 공격을 넘어 factual distortion과 representational bias까지 포함해, 관용적 알람을 회피하는 위협 시나리오를 확장해 평가합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 라벨 없는 poisoned 샘플을 대규모 요약 파인튜닝 데이터에서 국소화하고, (2) 학습 과정이 만든 “지속적 구조적 아티팩트”를 재학습 없이 되돌리는 것입니다. Defense-1은 influence-function 계열의 DataInf로 학습 샘플별 training influence를 계산해 상위 영향 집합만 후보로 축소한 뒤, 의미 일관성/팩트 정합성/편향 지표 등 휴리스틱으로 필터링하고 마지막에 gradient-ascent unlearning으로 영향(gradient)을 역방향 제거합니다. Defense-2는 학습 데이터 없이도 poisoned 모델이 semantics-preserving perturbation에 더 취약하다는 관찰을 SAP 지표로 수치화해, 행동 신호만으로 검출·분리를 수행합니다.

- **Empirical Impact**: 9개 아키텍처와 6개 벤치마크, adaptive attack까지 포함한 실험에서 Defense-1은 탐지·복구에 85~92% 수준의 정확도를 보였고, unlearning으로 원래 행동을 최대 96%까지 복원하면서 유틸리티(R0UGE) 손실을 0.6% 미만으로 제한했습니다. 또한 단순 ROUGE 회복을 넘어 extractiveness로의 행동 전환 같은 “깊은 생성 거동”이 94~96% 복원되는 것을 보여, 기존 지표 중심 검증의 한계를 보완합니다. black-box에서 Defense-2는 SAP 기반으로 clean/poisoned 모델 분리를 거의 완벽히 달성하며, 학습 데이터 미접근 환경에서도 실무형 사후 감사와 복구 가능성을 시사합니다.



### Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix I (https://arxiv.org/abs/2606.26027)
- **Prior Approaches**: 기존 연구는 도구 호출 형식에 맞춘 상호작용 프레임워크, 고품질 궤적(trajectory) 합성, SFT/비지도·RL 학습으로 도구 사용 능력을 끌어올려 왔다. 특히 agentic RL은 매 단계 탐색과 보상으로 복잡한 멀티스텝 작업을 해결하는 데 유망하지만, 도구 사용 태스크에서는 학습이 불안정해지거나 성능이 제한적으로만 개선되는 문제가 반복된다. 또한 RL에 supervision을 섞는 방식들이 제안됐지만, 멀티턴 도구 사용에서 어떤 형태의 supervision이 안정성에 본질적으로 기여하는지에 대한 체계적 비교는 부족했다.

- **Core Contribution**: 이 논문은 멀티턴 도구 사용에서 agentic RL이 겪는 실패를 ‘추론 능력 저하’가 아니라 ‘구조적 붕괴(structural collapse)’로 규정하고, 그 원인이 특정 제어 토큰(control tokens)의 확률이 비정상적으로 증폭되며 도구 호출 구조가 무너지는 데 있음을 보인다. 이어서 다양한 supervisory signal(오프폴리시 supervision, hint 기반 가이드, 오류 궤적 supervision, process reflection supervision 등)을 동기(synchronous)와 인터리브(interleaved) 통합 방식으로 체계화해 안정성과 성능의 상호작용을 분석한다. 결론적으로, RL의 탐색을 supervision으로 구조적으로 제어해야만 멀티스텝 도구 사용 학습이 견고해진다고 제안한다.

- **Technical Challenges**: 핵심 기술적 난관은 RL 탐색 중 토큰 수준에서 <tool_call> 같은 포맷 제어 신호가 섞이면서, 텍스트 오염(text pollution)→완전 붕괴(collapsed)로 이어지는 ‘형식/구조 붕괴’를 막는 것이다. 저자들은 이를 위해 (1) SFT를 초기화로 쓰되, (2) 인터리브 방식으로 RL 실패 사례에서 생성된 정답 기반 SFT를 섞어 오류 상태를 재정렬하고, (3) process reflection supervision으로 중간 의사결정의 논리적 스캐폴딩을 텍스트 감독으로 주입하는 접근을 사용한다. 또한 학습률이 안정성·일반화에 미치는 민감도와, SFT 데이터 분포가 format/content OOD에서 과적합을 유발하는 경로까지 함께 진단한다.

- **Empirical Impact**: BFCL-V3에서의 실험에 따르면 GRPO 단독이나 동기 mixing은 붕괴/불안정성이 크지만, RL과 SFT를 인터리브하면 안정성이 크게 개선된다. 다만 그 대신 format·content OOD 평가에서는 일부 성능 저하가 나타나며, process reflection supervision(PRS)이 이를 완화하는 것으로 보고된다. 추가로 학습률을 키우고 오류 기반 supervision 비율을 적절히 두는 것이 개선을 좌우하며, 단순 보상 스칼라만으로는 토큰 수준의 실행 구조를 안정화하기 어렵다는 점을 실증적으로 뒷받침한다. 결과적으로 이 연구는 멀티턴 도구 사용 agentic RL의 실패를 ‘토큰 동역학 제어’ 문제로 재정의하고, 다양한 supervisory signal 조합이 실제로 견고한 학습을 만든다는 가이드라인을 제공한다.



### The Tatoxa System for Text Detoxification in Low-Resource Languages: The Case of Tatar (https://arxiv.org/abs/2606.26015)
- **Prior Approaches**: 기존 텍스트 detoxification 연구는 주로 영어·러시아처럼 데이터가 풍부한 언어에 집중돼 왔고, 저자원 언어에서는 단일 multilingual LLM에 의존하는 경향이 커 성능이 크게 흔들렸다. 또한 문화적 맥락과 표현 관행 차이 때문에 toxicity를 판별·완화하더라도 의미 보존이 어려워, 사람 기준 baseline과의 격차가 컸다. CLEF 2025 공유과제에서도 Tatar 같은 저자원 언어는 최하위권에 머물며, 특히 “언어별 어휘 적응”이 필요하다는 점이 드러났다.

- **Core Contribution**: 이 논문은 Tatar 전용 텍스트 detoxification 시스템 Tatoxa를 제안하고, 핵심적으로 Russian→Tatar 번역 기반의 파이프라인과 다중 후보 생성·랭킹을 결합해 품질을 끌어올렸다. 아울러 Tatar detoxification을 위한 신규 데이터셋(수동 701개 추가)을 공개하고, fine-tuning과 저자원 환경 평가를 가능하게 했다. 마지막으로 cross-lingual transfer 실험을 통해 러시아어 등 타 언어에서의 지식 이전이 기대보다 훨씬 약함을 정량적으로 보여준다.

- **Technical Challenges**: 가장 큰 기술적 난관은 Tatar에 맞춘 병렬 detox 데이터가 부족하다는 점이다. Tatoxa는 NLLB-200을 Tatar-Russian 번역에 맞춰 적응시킨 뒤 러시아 detox 데이터들을 Tatar로 합성하고, LaBSE 기반 cross-lingual semantic similarity로 잡음을 필터링해 훈련용 병렬쌍(약 3만8천)을 구축한다. 또한 의미 붕괴나 잔여 toxicity를 줄이기 위해 multi-candidate(후보 60개/어댑터) 생성 후 neutrality(독성 점수)와 원문 의미 유사도(LaBSE)로 랭킹하며, mT0-XL 백본 위에 LoRA 어댑터 앙상블을 적용해 안정성을 높였다.

- **Empirical Impact**: 실험 결과 Tatoxa는 CLEF-Tatar 및 자체 확장 데이터셋에서 전체 점수 JJ와 STA 지표에서 기존 오픈소스·상용 LLM 및 간단한 베이스라인을 전반적으로 앞섰다. 특히 STA는 매우 높게 유지되면서도 SIM/FL은 기준 답변과 완전히 일치하진 않아, “의미는 대체로 보존하되 레퍼런스와는 차이가 생기는” 패턴이 관찰됐다. 또한 cross-lingual transfer에서는 러시아 등 관련 언어로부터 학습한 경우가 Tatar 원어 데이터로 학습한 경우보다 크게 뒤처져, 저자원에서는 병렬 생성·미세조정 전략이 실용적 대안임을 시사한다.



### Dziri Voicebot: An End-to-End Low-Resource Speech-to-Speech Conversational System for Algerian Dialec (https://arxiv.org/abs/2606.26003)
- **Prior Approaches**: 기존 음성·언어 기술은 영어 등 고자원 언어 중심으로 최적화돼 방언·저자원 환경에서 성능과 접근성이 크게 떨어진다. 알제리 다르자(Algerian Dialect)는 표준 철자 부재, 프랑스어 코드스위칭, Arabizi/아랍 문자 병용 등으로 ASR·NLU가 더 어렵지만, 선행 연구는 주로 텍스트 기반 대화(Boulesnane et al., 2022)에 머물렀다. ASR·NLU·TTS를 각각 분리해 개선해도 음성-대화 전체 파이프라인 통합 평가는 부족했고, 구성요소 오류가 연쇄적으로 NLU를 망가뜨리는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 알제리 다르자를 위한 완전한 speech-to-speech 과제용 대화 시스템을 제안한다. ASR→NLU(의도/개체)→Dialogue Management→conditional Retrieval-Augmented Generation(RAG)→TTS까지를 단일 모듈형 아키텍처로 통합해 end-to-end 음성 상호작용을 구현한다. 또한 통신(telecom) 도메인에 맞춘 ASR/NLU/TTS 전용 데이터셋을 구축하고, 각 컴포넌트에 pretrained 모델 fine-tuning을 적용해 재현 가능한 기준선(baseline)을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 적은 음성 데이터(2.68시간 규모)에서 방언의 음성·언어 변이를 흡수하는 것, (2) 코드스위칭과 철자 변동이 전사 오류를 키워 NLU로의 오류 전파를 유발하는 것, (3) 다르자용 통신 도메인 TTS가 거의 없다는 점이다. 해결책으로 ASR은 Whisper-medium을 방언·코드스위칭에 강하도록 supervised fine-tuning하고, 보강은 신호 기반 augmentation을 중심으로 적용했다. NLU는 DziriBERT 임베딩과 Rasa 기반 task-oriented 프레임워크를 결합하고, RAG는 FAISS 검색+Llama 3.2 3B Instruct로 환각을 줄이도록 근거 제한 프롬프트를 사용했으며, TTS는 XTTS-v2(LoRA)와 VITS-ar를 통신 도메인 음성 코퍼스로 미세조정해 방언 발음과 코드스위칭을 반영했다.

- **Empirical Impact**: 평가는 전체 구성요소를 함께 검증하는 형태로 진행됐고, ASR은 telecom 도메인에서 WER 13.74%(Whisper-medium)로 보고된 첫 벤치마크 수준의 성능을 보였다. NLU는 intent 분류 98.4%, entity F1-score 93.9%를 달성했고, 개방형 질의에 대한 conditional RAG는 78.5/100의 종합 성능을 유지하면서 기술적 신뢰성을 유지하는 방향으로 설계됐다. TTS는 새로 구축한 다르자 통신 코퍼스(약 50.7분)로 XTTS-v2와 VITS-ar를 fine-tuning해 음성 합성 품질이 안정적임을 보여, 저자원 방언 음성 봇을 위한 실전형 통합 베이스라인의 의미가 크다.



### SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models (https://arxiv.org/abs/2606.25990)
- **Prior Approaches**: 기존 음성 정서 지능 평가는 주로 Speech Emotion Recognition(SER)처럼 잡음을 라벨로 분류하거나, 텍스트만으로 감정지능을 추론하는 방식에 치우쳐 있었다. 그 결과 음성의 prosody·타이밍·강도 같은 부수적(파라링귀스틱) 신호를 ‘대화에서’ 얼마나 논리적으로 다루는지, 그리고 multi-turn에서 맥락을 유지하는지 검증이 약했다.

- **Core Contribution**: 이 논문은 SpeechEQ라는 평가 프레임워크와 벤치마크를 제안해 Speech-Language Models(SLMs)의 sociolinguistic reasoning을 다중 턴 음성에서 측정한다. EQ-i 2.0 이론에 기반한 15개 EQ 하위 척도, 2,265개 대화와 이를 요약하는 Spoken Emotional Quotient(SEQ) 점수를 도입해 인간 EQ 평가를 닮은 형태로 표준화했다.

- **Technical Challenges**: 핵심 기술 과제는 텍스트 의미 단서를 제거하면서도 음성만으로 정서적 적합성을 강제하는 ‘cross-modal reasoning’을 설계하는 것이다. 이를 위해 동일 transcript를 가진 두 오디오 선택지를 forced-choice로 구성하고, 다중 턴 정서적 긴장도를 순차적으로 상승시키며, 음향 대조를 librosa 특징(피치·발화 속도·스펙트럴 센트로이드·에너지·MFCC·길이 등)으로 검증해 품질을 확보했다.

- **Empirical Impact**: 실험 결과 end-to-end 모델이 cascaded 파이프라인보다 전반적으로 우수하지만, modality shortcut·alignment-induced safety trap·contextual amnesia 같은 병목이 여전히 드러났다. 또한 SEQ가 인간 선호와의 Spearman 상관(ρ=0.943)을 보여 기존 단순 정확도보다 신뢰성 있는 감정지능 대리 지표임을 입증했으며, 특히 2번째 평가 턴에서 성능 저하가 나타나 지속 정서 추적의 한계를 확인시켰다.



### Weave of Formal Though (https://arxiv.org/abs/2606.25987)
Comments:
          Code is available at this https URL

- **Prior Approaches**: 기존 constrained-decoding은 부분 파스에 맞는 토큰만 남기도록 어휘 마스킹을 하며 문법적 타당성을 높이지만, 컨텍스트-센서티브 lexing·maximal-munch·키워드 추출 같은 핵심 렉서 메커니즘을 다루는 데 제한이 있다. 또한 언어모델은 계층 구조(비터미널/파생)를 내부적으로 학습하기보다는 미리 정해진 정책으로 문법을 주입하는 경우가 많아, 어떤 구조를 드러내야 하는지 자체 학습이 약하다. 한편 CoT 계열이나 내부 추론 기법은 검증 가능한 이산 구조를 제공하지 못해 형식 오류를 체계적으로 배제하기 어렵다.

- **Core Contribution**: WoFT(Weave of Formal Thought)는 (1) Tree-sitter 전체 스펙에 대해 sound & complete한 형식 엔진 기반 constrained decoder와, (2) 비터미널 문법 파생을 이산 잠재변수로 학습하는 latent-variable fine-tuning을 결합한다. 형식 엔진은 subword 토큰이 “유효한 프로그램 prefix로 확장될 수 있는 경우”만 허용해 문법 검증을 생성 루프에 직접 내장한다. 학습 단계에서는 비터미널을 생성에 끼워 넣어, 모델이 필요할 때만 “형식적 파생(구조 scratchpad)”을 보존하도록 유도한다.

- **Technical Challenges**: 주요 과제는 LLM이 임의의 subword 경계에서 토큰을 생성할 때, Tree-sitter 렉서/외부 스캐너가 요구하는 완전한 입력과 maximal-munch를 온라인으로 만족시키는 것이었다. WoFT는 GLR 파싱에 speculative-lexing을 결합해, 렉서 상태 가설을 graph-structured stack과 동기화하며 동시다발적으로 토큰 확장 가능성을 추적한다(외부 스캐너는 coroutine 형태로 scanlet을 suspend/resume). 학습에서는 과거의 구조 잠재변수 모델처럼 완전한 parse 트리를 정확 마진화하는 대신, Tree-sitter를 symbolic oracle로 두고 RWS(reweighted wake-sleep)로 IW-ELBO를 최적화해 surface text의 중요도 가중 증거를 통해 비터미널을 선택적으로 학습한다.

- **Empirical Impact**: Python에서 StarCoder2-3B(3B)를 WoFT의 RWS 목표로 fine-tuning하면, text-only SFT baseline 대비 per-token cross-entropy가 14.3% 상대 감소했다. 이는 평면적인 autoregressive 학습이 잃기 쉬운 핵심 구조 정보를, 이산 비터미널 잠재 구문으로 되살릴 수 있음을 실증한다. 결과적으로 생성 시점의 형식 검증과 학습 시점의 구조 내재화가 함께 개선되며, 코드 LLM의 오류율을 낮추는 실용적 방향을 제시한다.



### Overview of HIPE-2026: Person-Place Relation Extraction from Multilingual Historical Texts (https://arxiv.org/abs/2606.25935)
Comments:
          Condensed Overview of CLEF-HIPE-2026 Shared Task Results

- **Prior Approaches**: HIPE-2020과 HIPE-2022는 다국어 역사 문서에서 named entity recognition과 linking에 초점을 맞췄다. 다만 동일 문맥에 함께 등장한 사람-장소 쌍을 ‘관계’로 오판할 수 있고, OCR 잡음과 문체 변화 속에서 간접 단서와 시간적 뉘앙스를 함께 추론하는 단계는 상대적으로 부족했다. 또한 이전 벤치마크들은 문헌 도메인/기간이 바뀔 때 성능이 어떻게 흔들리는지까지는 제한적으로만 다뤘다.

- **Core Contribution**: HIPE-2026은 ‘누가 어디에 있었는가, 그리고 언제였는가’를 temporally grounded person–place relation extraction으로 확장하며 두 관계를 정의한다: at(문서 출판일 이전의 어느 시점에 해당 장소에 존재)와 isAt(출판일 전후의 시간 지평에서의 동시적 존재). 특히 at의 3가설 분류(TRUE/PROBABLE/FALSE)와 isAt의 이진 분류를 함께 평가해, 표면 증거만이 아니라 문맥 일관성을 바탕으로 한 추론(예: 간접 단서)을 정량화하려는 설계가 핵심이다. 평가 프로파일도 정확도뿐 아니라 계산 효율성과 도메인 일반화까지 포함해, 문화유산 규모 처리 요구를 직접 반영한다.

- **Technical Challenges**: 역사 신문/문학 텍스트는 언어 변이가 크고 OCR 잡음이 섞여 있으며, 사람-장소 관계 근거가 문장 밖에서 간접적으로(교차문장, 대명사/서명 등) 드러나는 경우가 흔하다. 게다가 at=PROBABLE처럼 ‘명시적 문장’이 없더라도 담화가 성립하도록 가정해 추론해야 하므로, 잡음 환경에서의 증거 가중과 시간 추론이 동시에 필요하다. 이를 위해 참가 팀들은 LLM의 in-context learning부터 few-shot/프롬프트 설계, 다국어 인코더 파인튜닝, LoRA 같은 parameter-efficient fine-tuning, 의존구문 그래프/특징 기반 모델, 규칙 기반 휴리스틱, 멀티 에이전트/앙상블 등 다양한 경로로 time horizon 일치와 불일치 제약을 다루려 했다.

- **Empirical Impact**: 17개 팀이 40회 이상 제출한 결과, 최첨단 대형 언어모델부터 경량 분류기까지 폭넓은 전략이 등장했고, 정확도·효율·강건성 사이의 트레이드오프가 명확히 드러났다. 특히 정확도 프로파일(다국어 신문)과 효율 프로파일(모델 크기/파라미터 메타데이터 반영), 그리고 surprise-domain 일반화 프로파일(초기 근대 프랑스 문학/역사 저술)로 성능 변동을 분리해 보여준다. 본 캠페인은 대규모 역사 문서 처리에서 ‘관계 추출’이 단순 동시출현 탐지를 넘어 시간적 근거 추론까지 요구한다는 점을 실증적으로 정리한 벤치마크로 의미가 있다.



### SARA: Unlocking Multilingual Knowledge in Mixture-of-Experts via Semantically Anchored Routing Alignmen (https://arxiv.org/abs/2606.25821)
- **Prior Approaches**: Sparse MoE는 효율적으로 늘어나는 파라미터를 통해 언어·도메인 전문화를 달성하지만, 성능은 라우터가 토큰을 적절한 expert로 보내는 능력에 크게 좌우됩니다. 기존 다국어 개선은 주로 continual pre-training이나 instruction tuning, 혹은 load balancing/프루닝 같은 효율 최적화에 초점이 있었고, 언어 간 expert 선택의 기계적 불일치(cross-lingual routing divergence)는 직접 해소하지 못했습니다.

- **Core Contribution**: SARA(Semantically Anchored Routing Alignment)는 고자원 언어에서 얻은 expert 라우팅 분포를 ‘semantic anchor’로 삼아 저자원 언어의 라우팅을 정렬하는 프레임워크입니다. 출력 로짓을 증류(distillation)하는 대신 MoE 내부의 라우팅 확률 분포를 symmetric Jensen-Shannon(JS) 발산으로 맞춰, 언어가 달라도 동일 의미에 대해 비슷한 expert 활성 경로가 나오도록 합니다.

- **Technical Challenges**: 핵심 기술적 난제는 저자원 입력이 표면적 어휘 차이 때문에 라우터가 다른 expert로 보내지면서 의미는 같아도 라우팅이 갈라지는 점입니다. SARA는 (1) 고자원에서 [boxed{}로 검증된 정답 기반 정렬 데이터의 저자원 번역을 만들고, (2) 고자원 anchor의 토큰-레이어 라우팅 분포를 priors로 추출한 뒤, (3) 저자원 입력의 라우팅 분포를 해당 priors에 JS divergence로 끌어당기는 다단계 학습을 수행합니다. 또한 next-token 학습 손실과 Switch Transformers의 load balancing을 함께 써 expert 붕괴를 막습니다.

- **Empirical Impact**: 2개 LLM과 5개 저자원 언어, 3개 벤치마크(Global-MMLU, BELEBELE, MGSM)에서 SARA가 standard instruction tuning 대비 성능을 개선했으며, 예로 Qwen3-30B-A3B에서 +0.8%, Phi-3.5-MoE-instruct에서 Global-MMLU +1.2%를 보고합니다. 분석 결과 FFT 같은 일반 fine-tuning은 라우팅 불일치를 부분적으로만 줄이는 반면, SARA는 레이어 7~34 구간의 JS divergence를 거의 0에 가깝게 낮춰 라우팅 정렬이 실제 병목을 해결함을 보여줍니다. 번역 품질(GPT-5 mini vs nano)과 학습 단계에 따른 동학 분석도 포함돼, 더 깨끗한 의미 감독과 추가 epoch에서 ‘기계적 일관성’을 누적해 더 높은 수렴 성능 상한을 만든다는 점을 시사합니다.



### Beyond Function Calling: Benchmarking Tool-Using Agents under Tool-Environment Unreliability (https://arxiv.org/abs/2606.25819)
- **Prior Approaches**: 기존 도구 사용(tool-use) 벤치마크는 복잡한 작업을 다루더라도 대체로 도구 환경이 깨끗하고 안정적이며 신뢰 가능하다고 가정한다. 그 결과 실제 배치 환경에서 흔한 도구 비신뢰성(오류·불일치·실패)에 대한 에이전트 성능 평가는 충분히 이뤄지지 않았다.

- **Core Contribution**: 본 논문은 recoverable reliability hazards를 가진 도구 환경에서 에이전트를 평가하는 벤치마크 ToolBench-X를 제안한다. 각 태스크는 실행 가능한 멀티스텝 과정을 포함하며, 다섯 가지 위험 유형(Specification Drift, Invocation Error, Execution Failure, Output Drift, Cross-source Conflict)이 주어져도 최소 한 가지 유효한 복구 경로(재시도·폴백·검증·교차 확인)를 통해 해결 가능하게 설계됐다.

- **Technical Challenges**: 핵심 기술적 과제는 위험이 주어졌을 때도 “복구 가능한” 인스턴스를 자동 채점 가능한 형태로 구성하는 것이다. 이를 위해 ToolBench-X는 결정론적(deterministic) 도구와 표준 정답(canonical final answer)을 함께 제공하고, 위험을 구조화해 에이전트가 위험 진단과 복구(검증/재호출/상충 해소)를 수행하도록 유도한다.

- **Empirical Impact**: 실험에서는 신뢰 가능한 도구에서 잘하는 에이전트가 recoverable hazard에서는 크게 실패하는 신뢰성 격차가 관찰됐다. 분석 결과 실패 원인은 호출량이나 추론 예산의 부족보다 위험 진단 능력과 복구 전략의 비효율에 더 가깝고, 회복 힌트는 많은 실패를 되살리지만 test-time scaling의 효과는 제한적이었다. 저자들은 도구 함수 호출 정확도에서 나아가 “불완전한 도구 환경에서의 과업 완료(task completion)” 중심의 평가로 이동해야 한다고 시사한다.



### Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges for LLM Adversarial Evaluation (https://arxiv.org/abs/2606.25782)
Comments:
          13 pages, 5 figures, Accepted into ICANN2026

- **Prior Approaches**: 기존 안전 평가는 LLM-as-a-judge처럼 디코더 기반 안전 심판을 호출해 유해성을 판정하는 방식이 주류였지만, 대규모 배치에선 비용·지연이 커진다는 한계가 있습니다. 반면 LlamaGuard 같은 파인튜닝 디코더 분류기는 상대적으로 빠르지만, 분포 변화나 새로운 공격 패턴에 취약해 일반화가 흔들릴 수 있습니다. 또한 지금까지는 ModernBERT 계열의 최신 인코더 분류기를 유해 출력 판정에서 LLM 판정자들과 체계적으로 비교한 연구가 부족했습니다.

- **Core Contribution**: 이 논문은 ModernBERT 계열 인코더 분류기(ModernBERT/Ettin)를 대화형 (user-model) 응답에서 유해 출력을 탐지하는 안전 가드레일로 적용하고, LLM-as-a-judge 및 파인튜닝 디코더 안전 모델들과 OOD 홀드아웃에서 직접 비교합니다. 7개 안전 심판이 라벨링한 데이터를 majority-voting으로 집계해 인코더를 파인튜닝한 뒤, JailbreakBench와 AILuminate 라벨을 가진 홀드아웃에서 F1, FNR 등을 기준으로 성능을 확인합니다. 특히 공격 기법(단일 턴, decomposition, escalation, context manipulation)별로 LLM 판정자들과의 일치/불일치를 분석해 “언제 인코더가 대체 가능한지”를 가이드합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 디코더 심판의 라벨을 섞은 majority-vote 신호에 인코더가 제대로 학습하느냐와 (2) 다턴 jailbreak처럼 유해 의도가 중간 단계에 분산될 때 인코더가 final (prompt,response)만 보고도 탐지할 수 있느냐입니다. 이를 위해 질문 단위로 group partition을 적용해 데이터 누수를 막고(같은 질문의 학습·평가 응답이 겹치지 않게), judge들 간 불일치가 큰 영역은 고신뢰(high-confidence)로 분리해 라벨 잡음의 상한을 함께 점검합니다. 결과적으로 decomposition 같은 분산형 공격에서 false negative가 커지는 경향은 확인되지만, 모델 크기(150M→400M) 확대로 일부 완화되는 패턴을 보였습니다.

- **Empirical Impact**: 홀드아웃에서 Ettin 인코더는 AUROC가 높고(예: Ettin-150M 0.939, Ettin-400M 0.929), 특히 안전-critical 지표인 FNR에서 LlamaGuard-4 및 일부 LLM 판정자 대비 더 균형 잡힌 운영점을 보였습니다. 또한 고신뢰 subset(심판 합의가 강한 구간)에서는 F1이 Ettin-150M 0.800, Ettin-400M 0.894로 크게 상승해 라벨 불확실성이 병목임을 시사합니다. 프로덕션 관점에서도 Ettin-150/400은 LlamaGuard-4-12B 대비 처리량과 지연에서 압도적인 이점을 보여(수백 CPS 수준 vs ~1 CPS대, median latency 수 ms vs 수백 ms) 저비용·저지연 가드레일로의 실무적 의미가 큽니다.



### OPERA: Aligning Open-Ended Reasoning via Objective Perplexity-based Reinforcement Learning (https://arxiv.org/abs/2606.25757)
Comments:
          21 pages, 8 figures

- **Prior Approaches**: 기존 RL 기반 LLM 정렬은 RLVR처럼 정답을 검증할 수 있는 과제에서 강점을 보이지만, 창의적 글쓰기처럼 성공 기준이 불명확한 open-ended 영역에는 그대로 옮기기 어렵다. 특히 LLM-as-a-judge 보상은 스타일 편향과 위치 편향(positional bias) 문제를 일으켜 감독 신호가 흔들리고, 모델이 자기 강화(self-enhancement bias) 방식으로 보상을 ‘비슷하게 보이기’에 소비할 위험이 있다. 또 pairwise 비교나 rubric 기반 보상은 해상도를 높이려 하지만, 여전히 주관적·거친 평가로 인해 장거리 추론을 안정적으로 안내하기가 쉽지 않다는 한계가 남는다.

- **Core Contribution**: 이 논문은 OPERA(Objective Perplexity-based Reflective Alignment)를 제안해, 외부 judge 대신 모델 내부의 퍼플렉서티(PPL) 동역학을 intrinsic reward로 바꿔 open-ended 정렬을 안정화한다. 핵심은 reflection 토큰이 등장한 구간에서 불확실성 감소(지역 uncertainty reduction)를 신호로 삼고, 최종적으로는 생성 묶음 간 상대 순위 기반 In-Group Relative Perplexity Reward(IGRP)를 결합해 확률적 일관성과 출력 효용을 함께 학습시키는 것이다. cold-start 단계에서는 cognition braking(“wait”, “but” 등)으로 반성 지점을 구조화하고, perplexity-prioritized rollouts로 일관된 reasoning branch를 고르는 합성 파이프라인을 통해 20,000개의 고품질 reasoning trajectory 데이터셋을 만든다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘사람 선호’를 믿을 만하게 대체할 보상 함수를 설계하는 것이며, 단순 PPL은 길이·표면 유창성에 흔들릴 수 있고, 국소 log-prob 변동은 reward hacking을 유발할 수 있다. OPERA는 reflection 토큰 포함 여부로 진짜 자기수정 구간을 식별하고, reference 응답에 대한 조건부 log-prob 증가만 productive reflection으로 카운트한 뒤 탄젠트(tangent)로 과도한 장문의 반복을 완화한다. 또한 IGRP는 같은 프롬프트에서 생성된 peer들 대비 상대적 순위로 보상을 정규화해 난이도 편차에 덜 민감하도록 했고, cognitive braking과 PPL 기반 후보 선택을 재귀적으로 반복해 논리적 연속성이 유지된 cold-start SFT 트레이스를 생성한다.

- **Empirical Impact**: 실험은 Llama-3.1-8B와 Qwen3-8B를 대상으로 다섯 벤치마크(AlignBench, HelloBench, EQ-Bench, WritingBench, MATH500)에서 일관된 개선을 보였으며, open-ended 영역에서 특히 효과가 크게 나타났다. Qwen3-8B에 OPERA를 적용하면 open-source SOTA를 갱신하고 일부 creative writing 작업에서 Gemini2.5, MiniMax-M2.5 같은 proprietary 모델과 동등하거나 더 높은 성능을 보였고, Llama-3.1-8B에서도 평균 성능이 125% 상승하며 WritingBench에서 22.54점 격차를 기록했다. 또한 ablation에서 합성 데이터 제거 시 성능이 크게 하락하고, self reward와 outcome reward를 분리하면 각각의 작업 성격에 맞는 신호가 부족해지는 트레이드오프가 확인되어, OPERA의 보상/데이터 설계가 장거리 생성 정렬에 실질적 의미가 있음을 뒷받침한다.



### BitNet Text Embeddings (https://arxiv.org/abs/2606.25674)
Comments:
          Under review

- **Prior Approaches**: 기존 LLM-based text embedder는 대조학습/지도학습으로 임베딩 품질을 끌어올리는 데 집중했지만, 실제 배포에서는 지연(latency)과 대규모 벡터 인덱스의 저장·전송 비용이 병목이 된다. 양자화(PTQ)는 빠르지만 4-bit 이하에서 성능 저하가 커서, 이를 보완하려는 QAT·BitNet류 1.58-bit 접근과 distillation이 주로 LLM 압축에 쓰여 왔다. 다만 임베딩 전용으로 “백본 연산 효율 + 벡터 저장 정밀도”를 함께 겨냥한 체계적 연구는 부족했다.

- **Core Contribution**: BITEMBED(논문에서는 BitEmbed로도 표기)는 LLM 기반 text embedding을 극저비트로 바꿔 인코딩 효율과 벡터 저장 비용을 동시에 줄이는 프레임워크를 제안한다. pretrained LLM을 BitNet-style ternary weights, quantized activations, 그리고 학습 안정화용 lightweight normalization refinement(SubLN) 구조로 변환하고, 이후 continual contrastive pre-training과 supervised contrastive fine-tuning에서 teacher-guided similarity-distribution distillation 및 attention-relation distillation을 결합한다. 더 나아가 단일 체크포인트로 여러 output embedding 정밀도(예: 16/8/4/2/1-bit)를 지원하도록 multi-precision embedding training까지 포함한다.

- **Technical Challenges**: 핵심 난제는 극단적 저비트 양자화가 임베딩 공간의 의미적 구조를 깨뜨려, 지도학습만으로는 표현력을 회복하기 어렵다는 점이다. 논문은 (1) 변환 직후 continual contrastive pre-training으로 의미 공분산/연관 신호를 다시 학습시키고, (2) full-precision fp16 teacher로부터 score-level cosine similarity 분포와 token-level attention relation을 distillation하여 저비트 모델의 미세한 상대관계를 보존한다. 또한 SubLN을 끼워 넣어 quantization로 인한 scale drift와 outlier 민감도를 완화하고, STE로 rounding/clipping 비미분 구간을 학습에 반영한다.

- **Empirical Impact**: MMTEB(eng, v2)에서 Qwen3-0.6B 및 Gemma3-270M을 대상으로 평가했을 때, BitEmbed는 full-precision teacher 임베더와 거의 비슷한 성능(대략 0.35점/약간의 격차)을 보이며 의미 표현 품질을 상당 부분 유지한다. CPU(8 threads) 기준 토큰 처리량은 Qwen3-0.6B에서 2배가량(364.36→830.50 tokens/s), Gemma3-270M에서도 약 2배(1181.28→2055.47 tokens/s)로 개선되어 지연·비용 관점의 실용성이 확인된다. 더불어 multi-precision training을 통해 8/4-bit는 16-bit 대비 거의 손실이 없고, 1/2-bit에서도 여전히 유의미한 성능을 유지하면서 저장 비용을 크게 줄여 배포 시 트레이드오프를 유연하게 선택할 수 있다.



### Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization (https://arxiv.org/abs/2606.25656)
Comments:
          Accepted to ACL 2026 GEM Workshop

- **Prior Approaches**: 기존 RAG은 주로 비정형 문서에서 벡터 검색 후 LLM이 답을 생성하는 방식으로 강점을 보여왔지만, 실제 데이터는 텍스트와 관계(그래프)가 함께 있는 반구조화 지식베이스가 많다. GraphRAG, Modular RAG, Agentic RAG 같은 변형들이 나왔지만, “더 복잡한 구조가 정말로 잘 먹히는가”와 “표준 평가가 그 차이를 공정하게 반영하는가”는 불명확했다.

- **Core Contribution**: 이 논문은 반구조화 지식베이스(텍스트+KG)에서 regular RAG, GraphRAG, Modular RAG, Agentic RAG를 체계적으로 평가·비교하는 프레임워크를 제시한다. 또한 실제 사용 시나리오를 반영해 총 9개의 표준 RAG 시나리오(문서 단독부터 text-graph 통합, 도메인 KG 연계, 에이전트의 multi-step planning까지)를 구현하고, precision medicine 도메인 STaRK-Prime에서 생산형 의사결정에 필요한 인사이트를 도출한다.

- **Technical Challenges**: GraphRAG/Agentic RAG는 그래프 컨텍스트와 에이전트 세션 히스토리가 커지면서 context/memory overflow 및 토큰·비용 문제가 쉽게 발생한다. 논문은 (1) 관계를 묶는 compact한 그래프 표현, (2) 에이전트 세션 내 그래프 deduplication, (3) 문서 chunk/content hash 기반 deduplication, (4) ReAct의 단순 루프를 배치형 retrieval로 바꾸는 컨텍스트 절감 루프 설계를 통해 token 사용을 19%-53%까지 줄이면서도 성능을 유지/개선한다.

- **Empirical Impact**: 실험 결과, 그래프만 쓰는 순수 GraphRAG(시나리오 3)는 성능이 매우 낮았지만, 사전 정의된 KG를 텍스트와 적절히 결합한 하이브리드(시나리오 5)가 전반적으로 우수했다. 반면, retrieval 순위 중심 지표가 개선을 과대평가할 수 있음을 보여주며, LLM이 생성 과정에서 실제로 선택하는 entity ID를 평가하는 generation-aware 설정에서 “retrieval-generation gap(확장 검색이 생성 품질을 비례 향상시키지 않음)”이 관찰됐다. 결론적으로, 고급 RAG를 무조건 도입하기보다 데이터의 구조/제약과 컨텍스트 비용 한계를 기준으로 아키텍처를 선택해야 한다는 데이터 기반 가이드라인을 제공한다.



### MedGuards: Multi-Agent System for Reliable Medical Error Detection and Correction (https://arxiv.org/abs/2606.25651)
- **Prior Approaches**: 기존 의료 오류 탐지·수정 연구는 few-shot in-context learning, CoT 프롬프팅, 검색 기반 grounding, 분류기+QA 하이브리드 등으로 접근해 왔습니다. MEDIQA-CORR 2024 같은 벤치마크에서는 BLEU/ROUGE/BERTScore 등 토큰 동등 가정 기반 지표가 주로 쓰였지만, 임상적으로 중요한 개체(진단명·약물·원인 등) 오류는 점수에 충분히 반영되지 않는 한계가 지적됩니다. 또한 멀티에이전트/중재(arbitration) 아이디어가 일반 추론에는 쓰였지만, 의료 오류 탐지·국소화·수정 전체를 위한 체계로 정리된 사례는 부족했습니다.

- **Core Contribution**: 이 논문은 MedGuards라는 안전 가드레일 프레임워크를 제안하며, 의료 오류 탐지·국소화·수정을 multi-agent in-context learning 문제로 재정의합니다. 탐지/국소화/수정 기능을 담당하는 전문 에이전트를 분리하고, 서로 충돌할 때 reasoning trace와 confidence score를 바탕으로 중재 에이전트가 합의하도록 설계해 해석 가능성과 적응성을 높입니다. 기반 LLM 추가 fine-tuning 없이 plug-and-play 방식으로 통합 가능하다는 점도 핵심입니다.

- **Technical Challenges**: 의료 텍스트에서는 (1) 한 모델의 불확실성·OOD 실패로 인해 결정이 흔들리고, (2) 오류가 실제로 ‘어느 문장/어떤 개체’에 있는지 국소화가 특히 애매하며, (3) 수정 품질을 임상 핵심어 기준으로 평가해야 합니다. MedGuards는 CoT로 탐지-국소화-수정 단계를 구조화하고, self-consistency 관점의 다중 에이전트 합의로 변동성을 줄이며, 중재를 ICL로 구현해 파라미터 업데이트 없이 갈등을 해결합니다. 평가지표로는 Keyword-Prioritized Correction Score(KPCS)를 도입해 참조 문장의 임상 핵심 키워드 생성 여부를 먼저 확인하고, ROUGE-1/BERTScore/BLEURT 기반 의미·표현 유사도를 가중 결합해 안전성을 더 직접적으로 반영합니다.

- **Empirical Impact**: MEDEC 및 MedErrBench의 다국어(영어·아랍어·중국어) 임상 노트 4개 데이터셋에서 MedGuards는 여러 기준선과 여러 지표에서 일관된 개선을 보였습니다. 예를 들어 MEDEC에서 오류 탐지 정확도와 국소화 정확도가 전반적으로 상승했고, 특히 KPCS가 크게 개선되어 ‘임상 핵심어 보존’ 측면의 효과가 두드러졌습니다. 또한 KPCS는 물론 사람 평가에서도 유효성이 확인되며, 백본 LLM이 달라도 성능 향상이 유지되는 model-agnostic한 안정성까지 시사합니다.



### Staying In Character: Perspective-Bounded Memory For Book-Based Role-Playing Agents (https://arxiv.org/abs/2606.25632)
- **Prior Approaches**: 기존 LLM 역할놀이 에이전트는 소설에서 인물/장면/관계를 추출하거나, persona 프로필과 대화 히스토리, RAG를 결합해 캐릭터 일관성을 높이려는 흐름이 강했습니다. 하지만 장편에서 자주 발생하는 OOC(out-of-character) 문제로, 캐릭터가 “보지 못한 사실”을 말해버리는 Factual Overreach과 고정된 말투로 상황 변화를 놓치는 Stylistic Monotony가 충분히 통제되지 않았습니다. 또한 경계(무엇을 알아야/거절해야 하는지)를 “캐릭터 관점에서 접근 가능한가”로 정의한 검증은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 소설 기반 캐릭터 에이전트의 지식 경계를 관점에 맞춰 강제하는 3층 메모리 구조 ReverieMem을 제안합니다. 핵심은 (1) 1인칭 장면 경험을 저장하는 episodic layer, (2) 캐릭터별 visibility로 접근 가능한 사실만 남기는 semantic layer, (3) 감정 전이와 상황에 따라 달라지는 발화/행동 패턴을 담는 personality layer를 분리해 추론에 결합하는 방식입니다. 이를 통해 “접근 불가능한 사실은 검색/생성에서 배제”하고, “상황에 맞는 스타일 변화를 패턴으로 주입”하는 목표를 동시에 달성합니다.

- **Technical Challenges**: 기술적 난제는 두 가지 실패를 동시에 막되, 장편에서 검색 노이즈를 줄이면서도 필요한 세부를 복원해야 한다는 점입니다. 저자들은 추론 시 episodic layer로 장면 프레임을 먼저 고정하고, semantic layer의 visibility-allowed subset ℱc를 기준으로만 facts 후보 풀을 확장하는 perspective-bounded reconstructive recall 파이프라인을 설계했습니다. 동시에 emotion transition 기반으로 personality 패턴을 선택해 메모리 fusion의 프롬프트에 style anchor를 넣어 말투 평탄화를 완화했습니다.

- **Empirical Impact**: 평가는 8개 소설로 구성한 KBF-QA(총 4,386문항)와 BookWorld의 5차원 pairwise 내러티브 비교 프로토콜을 함께 사용해 지식 경계와 생성 품질을 분리 측정했습니다. ReverieMem은 Knowledge Boundary Fidelity에서 기존 최강 대비 34.6 percentage points 개선을 보였고, pairwise 비교에서도 약 79% win rate를 기록했습니다. 또한 계층별 제거 실험에서 episodic/semantic/personality 각각이 독립적으로 필요하며, 특히 visibility gating이 factual overreach와 character-anchored 서사 품질을 함께 끌어올린다는 점을 보여줍니다.



### Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints (https://arxiv.org/abs/2606.25605)
Comments:
          2 figures, 14 tables

- **Prior Approaches**: 기존 연구는 tool calling과 structured output을 각각 따로 평가하거나, 답변 품질·형식 준수율 중심으로 “Constraint Tax”를 관찰해 왔다. 에이전트 파이프라인에서 두 제약이 동시에 걸릴 때 tool 실행이 실제로 깨지는지, 그리고 그 원인이 어디(모델 선택 vs 디코딩 제약)에 있는지는 충분히 검증되지 않았다.

- **Core Contribution**: 이 논문은 production Agent에서 관측된 재현 가능한 현상 Tool Suppression을 정의한다. Tool Calling과 JSON Schema 제약을 함께 켜면 도구 호출은 사라지지만 JSON 스키마 준수는 높은 상태로 유지되는 패턴을 여러 오픈 가중치 모델·배치 설정에서 반복 확인했다.

- **Technical Challenges**: 원인 분석 결과, JSON Schema가 문법 기반 token mask로 컴파일되면서 tool-call에 필요한 토큰이 decoding 과정에서 도달 불가능해졌다는 구현 레벨 설명을 제시한다. 또한 이를 해석하기 위한 행동 가설로 Constraint Priority Inversion(CPI)을 제안하지만, 내부 메커니즘이 확정된 것은 아니라고 선을 그었다.

- **Empirical Impact**: 대응책으로 추론 시점에 tool 실행과 schema 제약 응답 생성을 분리하는 Transparent Two-Pass Execution을 제안하며, 재학습 없이도 tool invocation을 복원하면서 structured output 보장을 유지했다고 보고한다. 이 결과는 도구 사용과 형식 준수를 각각 따로 평가하는 접근이 production 신뢰성을 놓칠 수 있음을 시사한다.



### Riazi-8B: An Urdu Large Language Model for Mathematical Reasoning (https://arxiv.org/abs/2606.25568)
- **Prior Approaches**: 기존 수학 추론 성능 향상은 주로 Chain-of-Thought(CoT) 프롬프팅이나 추론 단계 보상/후학습 같은 방식에 의존했지만, 대부분 영어 중심 데이터·벤치마크에 맞춰져 있었다. 그 결과 MGSM 등 다국어 설정에서조차 저자원 언어(우르두)로 가면 구조화된 multi-step 추론이 약해지고, 중간 계산과 최종 답이 불일치하는 문제가 반복된다. 우르두 쪽 LLM도 Alif-8B, Qalb-8B처럼 자연스러운 문장은 만들 수 있어도, 수학 추론에 특화된 step-by-step 일관성 학습은 부족하다는 한계가 있다.

- **Core Contribution**: 논문은 우르두 수학 추론 격차를 메우기 위해 Riazi-8B를 제안한다. Qwen3-8B 기반으로 CPT(우르두 Wikipedia로 지속 사전학습)와 SFT(우르두 Chain-of-Thought 데이터로 지도 미세조정)를 2단계로 결합해, 우르두 언어 적응과 추론 단계 생성 능력을 동시에 끌어올린다. 특히 최종 답뿐 아니라 추론 품질과 출력 완결성까지 함께 개선되는지를 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 저자원 언어에서(1) 우르두 문장 생성 품질이 무너질 수 있고, (2) multi-step 논리 흐름이 중간 계산과 어긋나며, (3) 추론 데이터 자체가 희소하다는 점이다. 연구팀은 LoRA 기반 parameter-efficient 학습으로 CPT와 SFT를 효율화하면서, GSM8K를 바탕으로 한 우르두 Chain-of-Thought(단계별 풀이)로 구조화된 추론을 강제했다. 또한 fastText 기반 Urdu script 비율(언어 붕괴), 최소 두 개 이상의 중간 단계 생성 여부(SCS) 같은 보조 지표로 실패 모드를 분리해 최적화를 유도한다.

- **Empirical Impact**: MGSM-Urdu에서 Riazi-8B는 기존 우르두 특화 모델과 강한 multilingual 기준선을 상대로 exact match 정확도뿐 아니라 Urdu output purity, step completeness에서 모두 우위를 보였다. LLM-as-Judge(gpt-oss-20b) 루브릭 평가에서도 Correctness, Reasoning, Urdu Fluency, Clarity, Completeness 전 차원에서 일관된 개선이 확인됐다. 언어 적응만으로는 부족하고, reasoning-focused supervision이 우르두에서의 구조화·일관된 추론 성능을 실제로 끌어올린다는 점을 경험적으로 입증해 저자원 수학 추론 확장 전략에 의미가 크다.



### BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents (https://arxiv.org/abs/2606.25556)
- **Prior Approaches**: Stepwise group-based RL(GRPO, GiGPO, HGPO 계열)은 가치 네트워크 없이 롤아웃 스텝들을 그룹으로 묶어 국소 advantage를 추정한다. 하지만 그룹 내 스텝을 “교환 가능”하다고 보는 가정이 오래된 상태-동작 신용배분(credit assignment)에 맞지 않아, 관측 해시 같은 불완전한 그룹 기준이 학습 신호를 깎는 문제가 있었다.

- **Core Contribution**: 이 논문은 기존 방법의 핵심 결함을 state-action credit mismatch로 정식화한다. 관측 해시는 상태 동질성에 비해 너무 세분화해 singleton 그룹(신호 0)을 만들고, 그룹 내 평균은 action별 미래 차이를 반영하지 못하므로, BiPACE는 critic 없이도 advantage를 정확히 분해하도록 drop-in 방식의 추정기를 제안한다.

- **Technical Challenges**: BiPACE는 두 국소 문제(상태 집계, 동작별 신용배분)를 동시에 해결해야 한다는 점이 기술적 난제다. 해결책으로 BiGPO는 actor의 hidden-state 공간에서 cosine 거리 기반 클러스터(미세한 bisimulation proxy)를 만들고, PACE는 각 클러스터에서 실행된 action 기준 peer baseline으로 Q(s,a)-V(s) 형태의 비모수적(critic-free) advantage를 재중심화한다.

- **Empirical Impact**: ALFWorld에서 BiPACE_Q는 Qwen2.5-7B 기준 validation success를 GiGPO의 90.8에서 97.1±0.9로 끌어올리며, 동일 롤아웃 예산 내 95% 임계치를 모든 seed에서 달성한다. WebShop/TextCraft 및 Qwen2.5-1.5B에서도 GRPO·GiGPO·HGPO 대비 개선이 확인되며, 추가 오버헤드는 한 스텝 wall time의 11.3%로 보고된다.



### SFL-MTSC: Leveraging Semantic Frame-Level Multi-Task Self-Consistency for Robust Multi-Intent Spoken Language Understanding (https://arxiv.org/abs/2606.25552)
Comments:
          Interspeech 2026

- **Prior Approaches**: 프롬프트 기반 SLU는 LLM의 zero-shot/few-shot 추론으로 task-specific fine-tuning 없이 intent detection과 slot filling을 수행하지만, multi-intent 상황에서 decoding stochasticity로 intent–slot 구조가 매번 달라지는 문제가 자주 발생한다. 이를 줄이려는 self-consistency 계열은 대부분 output-level majority voting이나 LLM-as-a-judge 같은 방식을 쓰지만, frame 구조를 세밀하게 일관성 검증/필터링하기엔 한계가 있었다.

- **Core Contribution**: 이 논문은 semantic frame 단위로 예측을 분해·집계하는 SFL-MTSC(Semantic Frame-Level Multi-Task Self-Consistency) 프레임워크를 제안한다. output-level 투표 대신 intent-specific frame을 모아 domain–intent 버킷화, slot-level clustering, 그리고 path support 기반 신뢰도 평가로 안정적인 frame만 남긴 뒤 최종 multi-intent 결과를 재통합한다.

- **Technical Challenges**: 핵심 과제는 서로 다른 reasoning path에서 생성된 frame들이 구조적으로는 같지만 slot key/표현이 달라 생기는 불일치를 어떻게 안정적으로 묶고 걸러내는가다. 이를 위해 Hybrid Jaccard(키-값 매칭과 값 기반 매칭을 혼합)로 slot clustering을 수행하고, association rule mining의 support를 응용해 여러 경로에서 반복 지지되는 slot cluster만 retained 하도록 설계했으며, 슬롯 재통합은 value-first 방식으로 representative value를 우선 정하고 그에 대응하는 slot key는 다수결로 결정한다.

- **Empirical Impact**: MAC-SLU(중국어 multi-intent, 최대 4개 동시 intent) zero-shot 실험에서 SFL-MTSC는 전반 정확도와 Slot F1을 일관되게 개선했으며, 특히 Vanilla Prompting 대비 Overall Acc. 개선폭이 가장 크게 나타났다. 결론적으로 slot 수준의 self-consistency가 intent 분류보다 더 큰 영향을 주는 경향이 관찰됐고, slot-level support filtering ablation에서 성능 이득의 주된 원인이 슬롯 클러스터 신뢰도 필터링임이 확인됐다.



### Fault of Our Stars: Behavioral Drivers of Rating-Sentiment Incongruenc (https://arxiv.org/abs/2606.25518)
Comments:
          7 pages, 3 figures. Submitted to MerCon 2026

- **Prior Approaches**: 관광 리뷰 분석에서 별점은 텍스트 감정의 편리한 weak label로 자주 쓰이지만, 별점과 리뷰 문장이 항상 같은 감정 극성을 담는지에 대한 검증은 상대적으로 부족했다. 기존 연구는 호텔·레스토랑처럼 특정 맥락에 편중되었고, 불일치를 단순 오류 교정 문제로만 다루는 경향이 있어 왜/어떤 조건에서 어긋나는지 설명력이 제한됐다. 또한 사전학습 기반(sentiment) 파이프라인은 비교적 쓰였지만, ‘별점-텍스트 불일치의 구조’와 그 방향 패턴까지 체계적으로 분해한 연구는 드물었다.

- **Core Contribution**: 이 논문은 스리랑카 관광 명소 리뷰에서 sentiment-rating incongruence(리뷰 텍스트 감정과 별점이 암시하는 감정이 다른 현상)를 실증적으로 정의·분석한다. 리뷰 텍스트에 대해서만 transformer 기반 sentiment inference로 감정을 따로 산출해, 별점과의 일치/불일치가 단순 잡음이 아니라 맥락 의존적 신호임을 보여준다. 더 나아가 불일치를 단순 매치/미스가 아니라 여섯 가지 directional mismatch 패턴으로 분류하고, 이를 설명 가능한 요인들과 연결한다.

- **Technical Challenges**: 핵심 기술적 과제는 별점을 ‘기준 정답’으로 가정할 때 생기는 순환(circularity)과 맥락 편향을 피하면서 텍스트 감정을 독립적으로 추정하는 것이었다. 이를 위해 1,000개 라벨로 4개 트랜스포머 모델을 비교해 최적의 cardiffnlp/twitter-roberta-base-sentiment를 전체 16,156개 리뷰에 적용했으며, 불일치는 Sentiment와 Rating_Class의 차이를 기반으로 6개 패턴으로 재구성했다. 이후 불일치(이진)와 관련된 요인을 보기 위해 로지스틱 회귀·Random Forest를 쓰고, Random Forest 해석은 SHAP으로 보강했는데, 예측 성능은 AUC 약 0.58~0.61 수준으로 ‘구조는 있으나 완전 설명은 어렵다’는 점도 함께 확인했다.

- **Empirical Impact**: 분석 결과 전체 리뷰의 18.6%에서 불일치가 발생했으며, Conservative Rater(38.4%)와 Obligatory 5-Star(28.3%)가 대부분을 차지해 불일치가 무작위가 아니라 방향성을 띤다는 점이 드러났다. 불일치 비율은 장소 유형에 따라 달라져 박물관이 26.3%로 가장 높고 국립공원은 12.8%로 낮았으며, reviewer expertise, 리뷰 길이, 여행 연도(시간 경향) 등이 유의한 기여 요인으로 나타났다. 또한 시간이 지남에 따라 불일치가 완만하게 감소하는 경향이 보여, 별점-텍스트의 정렬이 점차 개선될 가능성도 시사한다. 결론적으로 별점은 텍스트 감정과 ‘서로 대체 가능’한 라벨이 아니므로, NLP 학습·평가에서 Ground-truth으로 쓰기 전 도메인별 검증이 필요하다는 실무적 함의를 제공한다.



### Spam and Sentiment Detection in Arabic Tweets Using MARBERT Mod (https://arxiv.org/abs/2606.25495)
- **Prior Approaches**: 기존 연구들은 트위터 같은 소셜 미디어 데이터를 바탕으로 고객 만족도와 비판을 파악하기 위해 감성 분석을 활용해 왔다. 딥러닝 기반 방법 중 BERT 계열이 NLP 감성 분석에서 성능을 보였지만, 상대적으로 영어 중심으로 연구가 진행돼 아랍어에는 공백이 컸다. 또한 아랍어 트윗에 맞는 충분한 데이터로 모델을 학습·평가한 사례가 제한적이었다.

- **Core Contribution**: 본 논문은 아랍어 트위터 고객 피드백을 대상으로 MARBERT를 학습해 감성을 분류하는 프레임을 제안한다. STC(Saudi Telecom Company) 고객 트윗을 분석해 긍정/부정/중립뿐 아니라 sarcasm과 indeterminate까지 포함된 감성 신호를 추출함으로써 고객 서비스 개선에 활용하는 것을 목표로 한다. 기존 영어 편중 연구를 아랍어 영역으로 확장한 점이 핵심 기여다.

- **Technical Challenges**: 아랍어는 형태 변화와 표현 다양성이 커서, 일반적인 모델이 뉘앙스를 안정적으로 포착하기 어렵다는 문제가 있다. 연구진은 아랍어 트윗 24,513개를 사용해 MARBERT 기반 제안 모델을 학습하고, 성능은 f1-score, precision, recall로 측정해 불균형한 라벨 분포를 반영해 평가했다. 특히 sarcasm과 indeterminate처럼 애매한 범주가 존재할 때의 분류 안정성을 함께 검증하는 방향으로 설계를 구성했다.

- **Empirical Impact**: 실험에서는 긍정 1,437, 부정 13,828, 중립 5,694, sarcasm 1,221, indeterminate 2,297로 구성된 아랍어 데이터로 학습·평가를 수행했다. 결과적으로 제안 방식은 문헌의 기존 기법과 비교해 정확도 측면에서 유망한 성과를 보였다고 보고된다. STC 고객 서비스 관점에서 트위터 기반 신속한 피드백/불만 파악을 돕는 실무적 활용 가능성도 시사한다.



### How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring (https://arxiv.org/abs/2606.25487)
Comments:
          10 pages, 3 figures, 2 tables

- **Prior Approaches**: 그동안 LLM jailbreak와 prompt injection 평가는 attack-success rate(ASR)을 주로 자동 안전 판정기(judge)로 매긴다. 안전 분류기(전용 classifier)나 LLM-as-judge(채점형 LLM)를 쓰지만, 해당 judge의 신뢰도는 사람 라벨과의 정합성을 충분히 검증하지 않는 경우가 많았다.

- **Core Contribution**: 본 논문은 judge가 ASR을 얼마나 왜곡하는지 “judge의 캘리브레이션과 적대적 견고성(adversarial robustness)” 관점에서 직접 실험한다. HarmBench 검증 세트의 사람 라벨 596개를 기준으로 전용 classifier와 3종 LLM-as-judge를 사람 다수결과 비교하고, 이후 judge를 속이는 공격까지 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 같은 응답을 서로 다른 judge가 어떻게 다르게 판정하는지 정량화하고, 그 판정이 실제 위해(harm) 제거가 아니라 표면 신호에 의해 뒤집히는지 확인하는 것이다. 연구진은 내용은 보존한 채 포장(wrapper)·거절 문장(prefix)·프롬프트 변경만으로 판정 뒤집힘을 유도하고, 전용 classifier는 wrapper에는 비교적 강하지만 white-box GCG에 대해서는 confident true positive의 70%가 뒤집힐 수 있음을 보여준다.

- **Empirical Impact**: 결과적으로 judge 선택만으로 ASR이 크게 흔들리며, LLM-as-judge는 거절 문장 하나만으로도 판정이 39~88%까지 뒤집히는 취약성을 보인다. 또한 사람 2인 감사를 통해 flip된 응답에는 위해 내용이 그대로 남아 있음을 확인해, 보고된 ASR의 신뢰성이 “평균”과 “공격 상황” 모두에서 낮을 수 있음을 시사한다. 저자들은 judge의 precision/recall 공개, judge precision으로 보정한 ASR 보고, 그리고 최소 1회 content-preserving 공격으로 judge 점검을 권고하며, 관련 코드도 공개한다.



### A Red Teaming Framework for Large Language Models: A Case Study on Faithfulness Evaluation (https://arxiv.org/abs/2606.25476)
Comments:
          Preprint submitted to SQJ

- **Prior Approaches**: 기존 red teaming은 GARAK, GOAT, HarmBench, JailJudge처럼 공격 자동화나 단일-판정 중심의 평가가 주를 이뤘다. 또한 multilingual 평가는 일부 프레임워크에서 다뤄졌지만, 언어쌍·태스크 전반의 취약점 비교와 평가 신뢰도(일관성)까지 같이 검증하는 경우는 제한적이었다. 특히 unfaithfulness(맥락-기반 환각)처럼 미묘한 오류를 안정적으로 가려내는 방법론이 부족했다.

- **Core Contribution**: 이 논문은 target-attacker-jury의 multi-role red teaming 아키텍처를 제안해, 공격 생성과 판정 역할을 분리하고 정보 흐름을 단방향으로 통제한다. jury는 다수 모델 합의와 inter-judge reliability(Fleiss’ kappa)를 사용해 응답의 정확성과 일관성을 정량 평가하며, 이 구조는 Q&A·요약·안전 위해 생성 등 다양한 태스크에 모듈형으로 적용된다. 특히 맥락에 묶인 unfaithfulness를 겨냥한 설계를 통해 신뢰할 수 있는 “취약점 패턴” 분석을 가능하게 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 공격 프롬프트가 자동으로 더 잘 먹히도록 편향이 누적되는 feedback loop를 막고, (2) 판정이 단일 모델의 주관에 좌우되지 않게 신뢰도를 확보하는 것이었다. 이들은 attacker와 target, jury를 엄격히 분리하고 ensemble 합의(majority/unanimous) 및 Fleiss’ kappa로 일관성을 계산하는 방식으로 해결했다. 또한 요약에서는 length limitation 같은 구조적 제약이 취약점 양상을 바꾸는지를 확인해, 단순 스케일링보다 아키텍처 설계·평가 설계가 안전에 더 큰 영향을 줄 수 있음을 보였다.

- **Empirical Impact**: 실험에서 exploitative한 적대 프롬프트는 Q&A에서 공격 성공률(ASR)을 최대 7.9%p까지 끌어올렸고, 요약에서는 구조적 제약이 unfaithfulness를 최대 30%까지 낮추는 효과가 관찰됐다. 언어 비교 실험에서는 Arabic 처리에서 unfaithfulness 취약점이 더 높게 나타나(평균 ASR 관측치 기준), 언어쌍 차이가 실제 신뢰성 문제로 이어질 수 있음을 시사했다. 다만 언어 간 맥락에서 미세한 unfaithfulness(명시적 모순이 아닌 오류)는 자동 탐지에 한계가 있었고, 다국어 적대 프롬프트의 완전 자동화는 추가 과제가 남았다.



### Optimizing Abstractive Summarization With Fine-Tuned PEGASUS (https://arxiv.org/abs/2606.25462)
- **Prior Approaches**: 기존 텍스트 요약은 extractive와 abstractive로 나뉘며, extractive는 원문 문장 일부를 고르는 방식이라 문맥 일관성 문제가 자주 생깁니다. abstractive는 생성 품질과 사실성 보장을 위해 더 복잡한 최적화가 필요하고, 특히 entity hallucination 같은 오류가 성능과 신뢰도를 깎는 핵심 한계로 지목됩니다. 또한 시퀀스-투-시퀀스 기반 어텐션 모델은 병렬 처리의 제약으로 속도 문제가 있어 transformer의 self-attention 채택이 대안으로 부상했습니다.

- **Core Contribution**: 이 논문은 abstractive summarization에서 PEGASUS를 XL-Sum English에 맞게 fine-tuning해, XL-Sum에서 mT5 기준선을 넘어서는 성능을 목표로 합니다. 저자들은 동일한 ROUGE 평가 체계에서 pegasus_xlsum이 ROUGE-1/2/L 전 지표에서 mT5_multilingual_XLSum을 개선했다고 보고합니다. 결과적으로 XL-Sum English에서 state-of-the-art 수준의 요약 성능을 주장합니다.

- **Technical Challenges**: 주요 기술 과제는 (1) 대규모 사전학습 모델을 특정 데이터셋(영문 XL-Sum)에 안정적으로 적응시키는 것과 (2) 학습 비용 제약 속에서 최적의 수렴을 만드는 것입니다. 저자들은 XL-Sum 학습 코퍼스 20%만 사용하고, 학습률 2e-05, 배치크기 8, epoch 5 등으로 조정하며 early stopping을 통해 자원 효율을 확보했습니다. 또한 ROUGE 기반 비교로 모델이 데이터셋 특성에 맞게 요약을 더 잘 생성하는지 정량 검증을 수행합니다.

- **Empirical Impact**: 평가에서 기준선 mT5_multilingual_XLSum의 ROUGE-1/2/L은 37.60/15.15/29.88이며, 제안한 pegasus_xlsum은 39.121/17.467/30.894로 모두 상승했습니다. 개선 폭은 ROUGE-1 +4.04%, ROUGE-2 +15.25%, ROUGE-L +3.39%로, 특히 ROUGE-2 향상이 문장 구조·관계 포착 능력이 커졌음을 시사합니다. XL-Sum English에서 SOTA를 달성했다는 점에서, multilingual 대형 생성 모델 튜닝 전략에 대한 실증적 레퍼런스로 의미가 있습니다.



### Probing in the Wild: A Case Study of Self-Supervised Speech Representations on Mandarin Sub-dialects with Unsupervised Articulatory Analysis (https://arxiv.org/abs/2606.25459)
- **Prior Approaches**: 자기지도 학습 음성 모델은 다양한 음성 과제에서 성능이 뛰어나지만, 내부의 음소(phonetic) 표현이 세부 방언 변화에서 어떻게 달라지는지는 상대적으로 덜 알려져 있다. 기존 probing 연구는 주로 수작업 음성학 라벨이 포함된 선별 코퍼스에 의존해, 자연 발생 방언 데이터로 일반화하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 만다린 자기지도 음성 모델의 조음(articulatory) 특징 표현을, 라벨 없이도 수행 가능한 probing 파이프라인으로 분석하는 사례 연구를 제시한다. 보편 phone recognizer로 전화 시퀀스를 만들고 이를 조음 특징 벡터로 매핑해, 수작업 애노테이션 없이 프레임 단위 decodability를 측정한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 자연 방언에서 수작업 라벨 없이 전화-조음 특징을 어떻게 안정적으로 정렬·매핑할지, (2) fine-grained dialect 차이를 내부 표현에서 신뢰도 있게 분해해 읽어낼지다. 논문은 language-agnostic universal phone recognizer와 조음 특징 매핑을 결합해 프레임 단위 probing을 가능하게 하고, 층(layer)별로 특징 그룹의 표현 동역학이 다르게 나타나는지도 함께 분석한다.

- **Empirical Impact**: 실험 결과, 만다린 하위 방언들 사이에서 조음 특징 decodability가 구조화된 패턴을 보이며, labiality나 stridency처럼 음향적으로 두드러진 특징은 비교적 안정적으로 유지된다. 반면 더 세밀한 스펙트럼 구분과 관련된 특징들은 방언 의존 변이가 커지고, 특히 Beijing 발화가 다른 하위 방언보다 decodability가 더 높게 나타나는 것이 주요 원인으로 제시된다. 전반적으로 language-agnostic articulatory probing은 실제 방언 코퍼스에도 적용 가능하며, 자기지도 표현의 dialect sensitivity가 조음 차원 전반에 고르게 분포하지 않고 선택적으로 나타난다는 점을 시사한다.



### Reclaim Evaluation: A Lossy Memory Is Worse Than an Empty On (https://arxiv.org/abs/2606.25449)
Comments:
          26 pages, 3 figures. Code, data, and reproduction harness: this https URL

- **Prior Approaches**: 기존 어시스턴트 메모리 기능과 검색 파이프라인은 요약·메모·검색 청크처럼 과거를 압축해 다음 응답에 “필요한 것만” 남긴다는 가정에 의존해 왔다. 또 모델 수정(editing)이나 self-correction 연구도 있었지만, 잘못된 중간 결론이 이미 굳어진 뒤에는 외부 피드백만으로는 복구가 어렵다는 점이 반복해서 지적돼 왔다.

- **Core Contribution**: 이 논문은 압축이 “답을 보존하느냐”뿐 아니라, 나중에 오류를 수정할 수 있는 “정정 가능성(correctability)”까지 좌우한다는 점을 보인다. 특히 잘못된 결론을 남겼는데 그 결론을 만든 근거(source)를 버리면, 모델이 stale한 값을 자신 있게 다시 내면서 수정이 실패하는 brittle memory 현상을 정의하고, 이러한 실패가 여러 모델에서 방향이 뒤집히지 않는 clean kill condition을 만족한다고 보고한다.

- **Technical Challenges**: 문제는 correctability를 모델 성능이나 메모리 크기와 분리해 측정해야 한다는 점인데, 논문은 reclaim evaluation으로 “고정 예산(B) 안에서 압축이 무엇을 남기느냐”만 바꿔 Reclaim Rate(RR)를 judge 없이 정답 매치로 평가하는 paired-memory 프로토콜을 제안한다. 핵심 해결책은 source-first 정책으로, 같은 예산에서 결론을 버리고 재계산 가능한 source를 1줄 메모로 보존해 복구 경로를 다시 살아나게 하며, 길이-일치 대조군으로 단순히 텍스트를 더 줘서 좋아진 효과를 배제한다.

- **Empirical Impact**: 실험 결과 lossy 압축(결론은 남기고 source를 제거)은 같은 예산 조건에서 빈(empty) 메모리보다 더 나쁜 행동을 유발하며, 7개 모델에서 wrong emission이 반복되고 반전이 관측되지 않는다. 반면 source-first는 oracle 수준(1.00)에 못 미치더라도 0.49~0.88 범위로 크게 복구를 회복하고, 메모리가 메모리를 먹는 체인 설정에서는 하나의 dropped-source 오류가 downstream을 누적 오염시키는 반면 source-first는 예산 기반의 horizon까지는 제어 가능한 것으로 나타난다.



### PolicyAlign: Direct Policy-Based Safety Alignment for Large Language Models (https://arxiv.org/abs/2606.25442)
- **Prior Approaches**: 기존 LLM 안전 정렬은 안전 시연(SFT)이나 선호 쌍·보상 신호(RL)처럼 고품질 감독 데이터를 만드는 데 크게 의존한다. 하지만 실제 배포에서는 안전 요구가 자연어 정책 문서로 먼저 나오고 데이터 수집·레이블링은 비용과 시간이 오래 걸려 갭이 생긴다. 또한 정책을 컨텍스트에 그대로 넣는 방식(예: ICL)은 표면적 준수는 돕지만 모델 파라미터에 내재화되지 않아 정책이 길고 복잡할수록 불안정하며 jailbreak에도 취약할 수 있다.

- **Core Contribution**: 이 논문은 안전 정책 C로부터 LLM을 직접 정렬하는 PolicyAlign을 제안한다. PolicyAlign은 (1) 정책 위반이 되도록 지시문을 합성하고, (2) on-policy self-distillation으로 학생 모델이 정책이 유도하는 행동을 내부화하도록 학습하며, (3) PSF(Policy-Sensitive Filtering)로 정책에 의해 행동 변화가 큰 샘플만 선별해 학습 효율과 안정성을 높인다. 그 결과, 정책 기반 정렬을 더 “유지보수 가능하고 확장 가능한” 경로로 만든다.

- **Technical Challenges**: 핵심 기술 난제는 정책에서 합성한 지시문 품질이 들쑥날쑥하고(인간 검증 부재), 특히 신종·전문 영역에서는 잘못된/잡음 샘플이 학습을 흔들 수 있다는 점이다. 논문은 각 후보 지시문에 대해 teacher(정책을 입력받는 안전 유도 응답)와 student(정책 미입력 기본 응답)를 비교하고, 정책이 만드는 행동 차이를 policy sensitivity score로 계산해 상위 k개만 남기는 PSF로 해결한다. 이후 학생이 스스로 생성한 궤적을 바탕으로 reverse KL로 teacher의 정책 조건부 분포를 따라가도록 하여, 노이즈와 exposure bias를 줄이면서 안전 행동을 파라미터에 내재화한다.

- **Empirical Impact**: 여러 모델(예: LLaMA-3.2-3B-Instruct, Qwen2.5-7B/14B)에서 PolicyAlign은 전반적 안전 성능을 일관되게 개선하되 over-refusal은 낮게 유지하며 일반 능력도 보존하는 트레이드오프를 보여준다. 특히 StrongREJECT/AdvBench는 물론 Fortress 등 jailbreak 벤치마크에서 Attack Success Rate를 크게 낮추고, XSTest에서는 안전 정책의 의도를 반영해 단순 거절률 상승이 아닌 더 뉘앙스 있는 준수를 달성한다. 또한 의료·법·금융의 emerging 도메인에서도 정책 기반 데이터가 부족한 상황에서 harmfulness를 줄이면서 도메인 정확도 저하를 최소화해, 정책 중심 정렬의 실용성을 실험적으로 뒷받침한다.



### Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making (https://arxiv.org/abs/2606.25421)
Comments:
          16 pages, 4 figures, 6 tables

- **Prior Approaches**: 기존 LLM 에이전트의 world modeling 학습은 주로 다음 관측(next-observation) 예측을 목표로 삼는다. 이 방식은 supervision이 “데이터에서 우연히 드러난 전이”에 묶여, 실제로 의사결정에 필요한 역학(dynamics) 일부를 놓칠 수 있다. 또한 목표를 goal-directed 행동 최적화로 바꾸면 예측 정확도가 떨어질 수 있어, prediction 품질과 decision 품질이 동일하지 않다는 한계가 강조돼 왔다.

- **Core Contribution**: 본 논문은 Agent-Authored World Modeling(AAWM)을 제안하며, 학습 목표를 다음 관측 재구성에서 “행동 전 필요한 의사결정 역학”으로 전환한다. 각 상태에서 에이전트가 자신의 믿음과 불확실성(확인된 패턴, 열린 질문)을 먼저 제시하고, 이를 바탕으로 훈련 타깃을 구성한다. 결과적으로 world model이 예측을 잘하는지보다, 에이전트가 실제로 무엇을 알 필요가 있는지에 정렬된 학습 신호를 제공한다.

- **Technical Challenges**: 핵심 과제는 의사결정에 필요한 dynamics를 타깃에 어떻게 반영할지와, 그 타깃이 신뢰 가능한 근거를 갖추게 할지다. AAWM은 Self-Probing으로 결정에 영향을 줄 수 있는 열린 질문을 뽑고, Transition Retrieval로 전이 풀에서 관련 evidence를 cross-trajectory로 가져온 뒤, Dynamics Synthesis가 이를 자연어 타깃으로 합성해 fine-tuning에 사용한다. 특히 검색은 relevance와 다양성을 함께 고려해 중복 증거로 인한 편향을 줄이도록 설계됐다.

- **Empirical Impact**: ALFWorld와 WebShop에서 동일한 학습 예산 및 설정 하에 AAWM이 next-observation 예측 기반 baseline(IWM)을 일관되게 능가했다(최대 약 6~11%p 수준의 success-rate 개선 및 전반 성능 우위). AgentGym 4개 환경 혼합 평가에서도 AAWM이 유일하게 imitation learning 대비 개선을 보이며, “타깃 선택이 더 신뢰도 높은 학습 신호”임을 확인했다. 분석 결과 AAWM은 GRPO 동안 더 높은 응답 엔트로피를 유지하며 메커니즘 기반 추론을 늘려, 단순한 출력 붕괴가 아닌 효과적 탐색을 유도하는 것으로 나타났다.



### Introducing corpora Hlava Cor and Hlava AD: Human Label Variation in Coreference and Discourse Relations (https://arxiv.org/abs/2606.25383)
Comments:
          Accepted to SLiDE 2026

- **Prior Approaches**: 기존 담화·코어퍼런스 주석 연구는 합의도(IAA) 평가나 다중 어노테이션 품질 분석을 다뤄왔지만, 어노테이터가 왜 그렇게 판단했는지에 대한 해설 코멘트까지 체계적으로 수집한 데이터는 드뭅니다. 또한 코어퍼런스와 담화 관계는 해석의 가변성(ambiguity) 자체가 존재해 라벨 불일치를 ‘잡음’으로만 보기 어렵다는 점이 반복적으로 보고돼 왔습니다.

- **Core Contribution**: 본 논문은 체코어 텍스트에 대해 인간 라벨 변이(Human Label Variation, HLV)를 분석하기 위한 두 코퍼스를 제안합니다: Hlava Cor(코어퍼런스)와 Hlava AD(귀속/비귀속 구문에서의 담화 관계). 두 데이터셋 모두 복수 어노테이터가 라벨을 달고, 각 선택에 대한 설명 코멘트를 필수로 포함해 서로 다른 해석 전략과 확신 수준의 차이를 추적할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘같은 예문을 보더라도 어떤 선행사/타깃을 선택할지’가 주관적으로 달라지는 상황을 공정하게 포착하는 것입니다. 연구진은 코어퍼런스에서 선행사 범위 규칙과 함께 원거리 문맥 필요도(0~2) 및 자유 서술 코멘트를 기록하고, 추가로 코어퍼런스 해석 모델들의 의견 불일치가 큰 예시를 우선 선별해 인간이 더 어려워하는 케이스를 집중 수집했습니다; 담화 관계에서는 오른쪽의 담화 커넥터가 가리키는 ‘target of the relation’을 지정하도록 하되, 귀속(생각/말하기 동사) 여부가 타깃 판정에 미치는 영향을 비교하는 구조로 설계했습니다.

- **Empirical Impact**: 두 코퍼스의 어노테이터 간 합의도는 대략 60~65%로 비슷한 수준을 보이며, 코어퍼런스에서는 특히 코어퍼런스 해석 모델들이 서로 다르게 예측한 사례에서 인간 합의가 더 낮아졌습니다(모델 불일치: 39% vs 모델 합의: 66%). 담화 관계에서도 타깃 지정의 IAA는 평균 64.9% 수준이며, 왼쪽 절의 구문 복잡도에 따라 IAA가 크게 흔들리지만 귀속/비귀속이나 텍스트 모드(구어/문어) 자체의 영향은 제한적이라는 관찰을 제시해, 향후 신뢰도 평가나 심리언어 실험용 기준 데이터로 활용될 잠재력을 보여줍니다.



### A Survey of Toxicity Detection and Mitigation Strategies for Multilingual Language Models (https://arxiv.org/abs/2606.25380)
Comments:
          Accepted to the Findings of ACL, 2026

- **Prior Approaches**: 기존 연구는 다국어 독성(혐오/모욕/욕설/정체성 공격 등)을 키워드 기반이나 단일 분류기로 잡는 방식에서 출발했지만, 문맥·은유·회피표현·방언 변형에 취약하다는 한계가 컸다. mBERT/XLM-R 같은 교차언어 인코더와 번역-파이프라인(비영어→영어→독성 분류)으로 성능을 끌어올렸지만, 번역 오차와 의미 표류, 문화적으로 다른 위해 정의가 그대로 남았다. 또한 detox는 데이터 필터링·SFT/선호튜닝·RLHF·디코딩 스티어링·편집/가드레일 등으로 확장됐지만, 언어 커버리지 불균형과 평가 프로토콜 단편화, 그리고 방언/정체성 표현까지 과도하게 억제할 위험이 지속된다고 정리한다.

- **Core Contribution**: 이 논문은 다국어 LLM의 detox/검출 과정을 ‘위협 모델-태스크-탐지/완화-평가’ 축으로 재구성해, 언어 선택을 악용하는 jailbreak(언어 스위치), 번역 피벗/라운드트립, 코드스위칭·혼합 스크립트, 멀티턴 상호작용, 배포 후 fine-tuning에 의한 정렬 붕괴를 체계적으로 카탈로그한다. 이어 toxic-to-neutral rewriting, toxicity classification, toxic-generation/prompt continuation의 세 태스크로 데이터셋과 메트릭을 묶고, 탐지 방법(교차언어 인코더, 번역 파이프라인, representation probe, LLM 기반 zero-shot 탐지) 및 완화 전략(데이터 필터링, 지도/선호 기반 튜닝, 디코딩 스티어링, representation editing, multilingual guardrails)을 한 지도 위에 정렬한다. 이를 통해 “언어별 안전 거동 불균일”이 기술적 강건성 문제이자 사회언어학적 커버리지 문제임을 강조한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 저자원 언어·문법적 풍부도·방언/스크립트 변형에서의 검출 성능 격차, (2) 위해 개념이 문화에 따라 달라 생기는 라벨·정렬 불일치, (3) 번역·피벗 과정에서 생기는 의미/화용 손실과 오류 전파, (4) 멀티턴 및 코드스위칭 같은 운영 조건에서 기존 단일턴 평가가 과소추정하는 보안 취약성이다. 논문은 이를 해결하기 위해 threat model을 진단 축으로 분해하고, STA·Perspective API 점수·BLEURT/BERTScore·수용성(acceptability) 같은 자동 메트릭과 함께 인간 평가를 결합해 신뢰도를 점검하는 평가 체계를 정리한다. 또한 완화는 단일 기법이 아니라 데이터 필터링-정렬 튜닝-디코딩 스티어링-가드레일을 하이브리드로 조합해 실패 모드를 메우는 방향을 제안한다.

- **Empirical Impact**: 실증적으로는 ParaDetox/MultiParaDetox/TextDetox·Jigsaw/OffensEval/HateCheck 등 분류·재작성 데이터셋과 RTP-LX/PolygloToxicityPrompts 같은 생성형 벤치마크가 다국어 안전 평가의 기준선을 제공하며, toxic-to-neutral과 toxic-generation이 서로 다른 실패 양상을 보인다는 점을 드러낸다. 특히 저자원 언어에서 unsafe rate가 더 높게 나타나고, 번역·멀티턴 조건을 포함하면 취약성 추정이 크게 달라진다는 관찰이 반복된다고 요약한다. 결론적으로 이 설문은 다국어 안전 연구가 단순 영어 성능 이전으로는 설명되지 않으며, 문화적 정렬·언어 커버리지·평가 일관성·과검열(모델 균질화) 방지까지 함께 설계해야 전진할 수 있음을 분야 의제로 제시한다.



### Story Operators: Decomposing the Original $\to$ Sequel Transformation in Embedding Spac (https://arxiv.org/abs/2606.25379)
Comments:
          8 pages, 3 figures

- **Prior Approaches**: 기존 연구는 주로 작품 간 유사도 계산, 군집화, 분류 같은 ‘정적 비교’에 초점을 맞춰왔다. 일부 내러티브 관련 작업은 임베딩 공간에서 텍스트를 다루지만, 원본→속편 변환을 이름 붙은 연산자들의 조합으로 ‘분해’하고 검증하는 데는 상대적으로 약했다.

- **Core Contribution**: 이 논문은 책을 문장 임베딩 공간의 점으로 보고, 원본 소설→속편 소설의 변화를 연산(Story Operators)으로 정의한 뒤 기하적으로 분해한다. 특히 두 권에 등장하는 실제 문단 임베딩만으로 PCA 기반 ‘콘텐츠 축’을 만들고, displacement 벡터를 along chain 형태로 탐욕 분해해 해석 가능한 몇 개 축으로 정리한다.

- **Technical Challenges**: 핵심 난제는 (1) 고정된 차원/사전 정의된 축이 아니라 두 텍스트 자체의 변동성을 반영하는 축을 뽑는 것, (2) 분해된 각 축을 실제 구절로 연결해 의미 있는 해석이 가능하게 만드는 것이다. 저자들은 PG19의 문단 임베딩( all-mpnet-base-v2 )을 평균 풀링해 책 벡터를 만들고, 원본·속편 문단을 함께 PCA한 축 위에서 displacement를 큰 기여부터 greedily along 성분으로 축적하며, 각 성분의 극값에 해당하는 실제 문단을 축의 양극(pole)으로 앵커링한다.

- **Empirical Impact**: Project Gutenberg에서 검증된 13개 작가 연속 쌍을 분석한 결과, 속편 변환이 ‘formulaic(작고 저순위)’, ‘concentrated(단일 지배 축)’, ‘compositional(다수 소축 확산)’이라는 유형으로 정리됨을 보였다. 대표적으로 Tom Sawyer→Huckleberry Finn에서는 1인칭 picaresque 전환 같은 표면 주제가 아니라 ‘보호적 가정성의 붕괴→picaresque 도로’ 같은 구조적 축이 지배적으로 복원되었고, 저자 의도(1875~76년 Howells와의 편지)와의 정렬도 정량화해 측정 한계(표현 불일치 등)를 명확히 밝혔다.



### Three Buddhist Vocabularies: Computational Stylometry of the English Pali Canon across Sutta, Vinaya, and Abhidhamma (https://arxiv.org/abs/2606.25372)
Comments:
          16 pages, 7 figures, 3 tables. code available at this https URL

- **Prior Approaches**: 기존 연구는 주로 Sutta Pitaka(경장)에 한정된 계산적 문체분석에 머물렀고, 다른 Pitaka나 전통 간 비교로 확장할 때는 비교 기준과 코퍼스 범위가 달라지는 문제가 컸다. 또 Zipf 법칙 같은 통계 지표는 사용되더라도 어휘 다양성, 수치/개수 표현 밀도, 어휘 겹침 같은 다각도 검증이 충분히 결합되지 않는 경우가 많았다. 본 논문은 이러한 부분을 Tipitaka 전 범위와 여러 영어 번역(및 전통)으로 넓히는 방향이다.

- **Core Contribution**: 본 논문은 Tipitaka 전체 3종(Pitaka three) 을 영어 번역 코퍼스로 구성해, Sutta Pitaka 중심의 선행 연구를 Vinaya 및 Abhidhammattha Sangaha, 그리고 타 전통 비나야까지 포함해 확장한 계산적 문체 분석을 제시한다. Zipf rank-frequency 분포, MATTR-500 어휘 다양성, numeral-word density(수·숫자 관련 단어 밀도), 어휘 겹침 지표를 함께 계산해 문체적 차이를 구조적으로 비교한다. 또한 특정 구절(예: ‘consciousness’가 특정 순위에서 문법 입자를 대체)처럼 지표가 가리키는 진단 단어의 방향성도 보고한다.

- **Technical Challenges**: Tipitaka 전체를 아우르는 대규모 다원 코퍼스에서 번역본 간 차이, 코퍼스 크기 차이, 전통별 용례 차이가 통계 지표를 왜곡할 수 있다. 논문은 Zipf 분포에 대해 OLS-fitted exponent를 사용하고, MATTR-500은 고정 윈도(500)를 적용해 비교 가능성을 높였으며, 크기 통제된 subsampling으로 다양성 차이를 확인한다. 아울러 Jaccard 및 Szymkiewicz-Simpson 같은 겹침 계수로 전통 간 법적 어휘 유사성을 정량화해, 단순 빈도 비교의 한계를 보완했다.

- **Empirical Impact**: 실험 결과 모든 코퍼스가 Zipf-consistent한 분포를 보였고(R2>0.989), Vinaya가 이상적인 Zipf 기울기(-1)에 가장 가깝게 나타났다. 어휘 다양성은 Sutta와 Vinaya Theravada가 거의 동일한 수준(각각 0.399, 0.400)인 반면 Sangaha는 더 높았고(0.560), Sangaha는 numeral-word density도 가장 높아(3.26%) 체계적 열거 관행과 일치하는 정황을 제공한다. 전통 간 비교에서는 Mulasarvastivada 비나야가 Theravada 비나야와 상당한 어휘 겹침을 보여(자카드 20.0%, overlap coefficient 49.1%) 공유된 법적 유산을 시사한다. 한편 같은 비나야의 영어 번역본 간 어휘 공유는 낮게(88년 차 번역에서 24.2%) 관찰되어 번역/해석 선택이 문체 지표에 미치는 영향을 보여주며, 관련 코드와 데이터는 Darshana Graph 코퍼스의 오픈소스 확장으로 공개된다.



### Neural Machine Translation for Low-Resource Tangkhul--English (https://arxiv.org/abs/2606.25365)
Comments:
          11 pages, 3 figures, 9 tables

- **Prior Approaches**: 기존 저자원 MT는 mT5 같은 다국어 pretrained의 transfer, back-translation, unsupervised MT 등으로 성능을 끌어올려 왔지만, 동북인도 티베토-버만 언어는 표준 벤치마크에서 거의 배제돼 Tangkhul에 바로 적용하기 어렵다. 또한 종교 도메인의 Bible parallel corpora 같은 우회 자원은 마지막 대안이지만, 실제 일상·현대 도메인과의 불일치(도메인 편향) 문제가 크다.

- **Core Contribution**: 이 논문은 Tangkhul–English(nmf-en) 저자원 MT를 위해, 공개 대규모 병렬 말뭉치를 구축하고 ByT5-large와 mT5-small을 fine-tuning해 번역 모델을 학습·평가한 것이 핵심 기여다. 특히 Tangkhul은 Latin 기반 diacritics(ā, a̱)가 있어 일반적인 subword 기반 토크나이저가 잘게 쪼개지기 쉬운데, 이를 byte-level 처리로 직접 다루는 접근을 제시한다. 더불어 Hugging Face Hub에 best 모델 tangkhul-byt5와 대비 모델 tangkhul-mt5를 공개해 연구 재현성을 높였다.

- **Technical Challenges**: 기술적 난제는 (1) Tangkhul의 Latin-script diacritics로 인한 토큰화/시퀀스 길이 증가, (2) Bible·이야기·대화로 섞인 학습 데이터의 도메인 편향, (3) 영어로의 단방향 번역에서 발생할 수 있는 의미·표현 정합성 오류다. 저자들은 ByT5가 UTF-8 byte를 그대로 처리해 ā와 a̱를 자연스럽게 다루는 반면, mT5는 a̱가 SentencePiece에서 미인식되어 base 글자와 결합부가 분절되며 시퀀스가 10–15% 길어지는 토큰화 이슈를 분석한다. 또한 빔 서치 빔 크기(최적 4)를 포함한 디코딩 튜닝과, mT5·ByT5 앙상블 리랭킹 시 환각/반복이 악화되는 사례도 보고했다.

- **Empirical Impact**: 실험에서 ByT5-large(tangkhul-byt5)는 BLEU 39.97, chrF++ 58.07, BERTScore F1 0.8104, COMET(wmt22-comet-da) 0.7302로 mT5-small 대비 큰 격차를 보였고, mT5-base zero-shot은 BLEU 0.03 수준에 그쳐 사전학습만으로는 Tangkhul 표현을 학습하기 어렵다는 점을 확인했다. 정량 지표 외에도 수동 에러 분석에서 어휘 치환과 스타일 환각이 주요 실패 모드로 나타났으며, 대화 도메인에서는 톤/의미가 과도하게 확장되는 양상도 관찰됐다. 결과적으로 byte-level 모델 기반 Tangkhul MT의 초기 기준선(baseline)을 제시하고, 비(非)성경 도메인 확장과 back-translation 같은 데이터 다양화가 다음 단계임을 실증적으로 뒷받침한다.



### Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents (https://arxiv.org/abs/2606.25361)
- **Prior Approaches**: RAG 기반 대화 시스템의 기존 연구는 주로 메모리를 어떻게 저장·검색하는지에 초점을 맞췄습니다. 하지만 서로 다른 기능 역할을 가진 메모리가 응답 품질에 어떤 영향을 주는지, 대화 맥락이 달라질 때 에이전트의 응답 방식이 얼마나 달라지는지는 상대적으로 덜 알려져 있습니다. 또한 기존 평가는 주로 reference 기반이라 사용자가 선호하는 뉘앙스를 다른 방식으로 반영할 가능성을 충분히 포착하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 대화 메모리를 역할 중심으로 세분화한 fine-grained taxonomy를 제시하고, 검색된 메모리를 서로 다른 role types로 분류합니다. 더불어 사용자 관점을 모사하는 user-centric 평가 프레임워크를 설계해 응답 차이를 더 정교하게 측정합니다. 그 결과, 메모리 유형별로 응답의 정확도·개인화·제약 인식 같은 특성이 다르게 형성된다는 점을 체계적으로 분석합니다.

- **Technical Challenges**: 핵심 과제는 “메모리의 내용”이 아니라 “기능적 역할”이 응답에 미치는 효과를 분리해 관찰하는 것입니다. 이를 위해 메모리 역할을 세밀하게 분류하고, 서로 다른 대화 맥락에서 retrieval된 메모리가 응답 특성(주제 일관성, 제약 인식, 개인화)에 미치는 영향을 비교하도록 실험 설계를 구성했습니다. 또한 reference 기반 평가의 부족을 보완하려고 사용자 선호를 반영하는 측정 프레임워크를 함께 도입했습니다.

- **Empirical Impact**: long-term 데이터셋과 frontier LLM을 대상으로 비교 실험을 수행한 결과, clarifying memory는 사실 정확성과 constraint awareness를 높여 더 정확하고 개인화된 응답을 유도했습니다. 반대로 irrelevant memory는 topic relevance를 떨어뜨리고 constraint awareness도 저하시켜 응답 품질을 악화시키는 경향이 관찰됐습니다. 이 연구는 메모리 유형을 선택적으로 활용해 개인화된 응답을 만들 수 있음을 실증적으로 보여주며, 메모리 역할 설계/평가 연구에 직접적인 방향성을 제시합니다.



### Efficient and Trainable Language Model Test-Time Scaling via Local Branch Routing (https://arxiv.org/abs/2606.25354)
- **Prior Approaches**: 기존 test-time scaling은 연쇄 추론을 길게 하거나(multiple traces), 탐색 폭을 늘리는 tree 기반 방법을 쓰지만, 대부분은 연산이 단일 스레드로 고정되거나(solution-level로 갈수록) 계산비용이 커져 학습·최적화가 어렵다는 한계가 있습니다. 또한 soft-token branching 계열은 후보들을 연속 혼합으로 병합해 미래 계산은 효율적일 수 있으나, 후보 분기(identity)와 분기 간 대비 신호가 흐려질 수 있습니다. 결과적으로 “토큰 수준에서의 가벼운 width”를 디스크리트 분기 형태로 유지하면서 end-to-end 학습까지 연결하는 접근이 부족했습니다.

- **Core Contribution**: 이 논문은 토큰 단위 로컬 분기 폭을 제공하는 Local Branch Routing (LBR)을 제안합니다. LBR은 각 디코딩 스텝에서 작은 로컬 lookahead 트리를 확장한 뒤, 모든 분기를 모델로 forward하고, router가 depth-1 서브트리 중 하나를 선택해 커밋하도록 설계됐습니다. 또한 prune-shift-grow 디코딩을 통해 “선택된 분기”만 다음 단계로 이어가면서도, 분기별 post-candidate hidden state에 근거한 토큰 결정을 가능하게 합니다.

- **Technical Challenges**: 핵심 기술적 도전은 로컬 분기에서 얻는 추가 증거를 사용하되, full solution-level search처럼 비용이 폭증하지 않게 만드는 동시에 학습 가능 objective를 정의하는 것입니다. LBR은 트리-궤적 likelihood를 계산 가능하게 인자분해해, stochastic은 새로 자란 노드 샘플링과 router 선택에만 두고 가지치기/이동/재사용은 결정적으로 처리합니다. 이를 통해 verifier 기반 reward가 검증 가능한 환경에서 likelihood-ratio 원리로 router와 베이스 모델을 함께 end-to-end reinforcement learning(GRPO 계열)으로 최적화합니다.

- **Empirical Impact**: 합성 hierarchical planning 실험에서는 LBR이 분기 이후(post-candidate) hidden state가 라우팅에 유의미한 정보가 됨을 보이며, discrete chain-of-thought나 soft thinking보다 성능이 좋게 나왔습니다. 수학 추론 벤치마크(DeepSeek-R1-Distill-1.5B/7B)에서도 LBR은 Pass@1과 Pass@32에서 discrete chain-of-thought, vanilla discrete-token RLVR, RL-compatible soft-token branching을 모두 앞질렀고, 로컬 lookahead를 L=1에서 L=2로 늘릴 때 성능이 추가로 개선되는 경향이 확인됐습니다. 또한 cross-subtree attention을 제거한 ablation에서는 성능이 떨어져, 단순 post-token hidden state 노출을 넘어 “형제 후보 간 대비(contrastive) 라우팅”이 이득의 중요한 원인임을 시사합니다.



### Hybrid-IR: Dual-Path Hybrid Retrieval with Iterative Reasoning for Complex Medical Question Answering (https://arxiv.org/abs/2606.25338)
- **Prior Approaches**: 기존 LLM 기반 의학 QA는 그럴듯하지만 근거 없는 환각과 최신 의학 지식 반영 한계가 있어 신뢰성이 부족하다는 문제가 있었다. RAG는 외부 문서를 붙여 환각을 줄이지만, 대부분 단일 retrieval 경로(그래프 또는 dense)만 사용해 미세 의미와 구조적 연관을 동시에 보존하기 어렵고, 정적(일회성) 검색은 다단계 추론에서 부족한 근거 체인을 만들기 쉽다. 또한 반복적 evidence 확장을 시도한 방법이 있어도, LLM의 중간 추론 상태와 검색이 긴밀히 결합되지 않아 검색 초점이 동적으로 정교해지지 못했다.

- **Core Contribution**: 이 논문은 복잡한 의학 QA를 위한 dual-path retrieval 프레임워크 Hybrid-IR을 제안한다. Hybrid-IR은 그래프 기반 검색으로 구조화된 지식(엔티티-관계, 문서 간 연관)을 탐색하고, dense 검색으로 원문 텍스트의 fine-grained 의미를 보완하며, 이를 retrieve–reason–retrieve 반복 루프로 결합해 추론 궤적을 점진적으로 개선한다. 특히 그래프 인덱스(KG-index)에서 entity-level 지식층과 문서 청크 기반 provenance 문서층을 함께 구성해 추론과 근거 연결을 강화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 문서 전반에 분산된 의학 지식을 구조적 관계로 엮으면서도 텍스트의 정밀 의미 손실을 줄이고, (2) 다단계 추론 중에 ‘다음에 무엇을 더 찾아야 하는지’를 검색에 반영하는 동적 제어를 구현하는 것이다. 저자들은 OpenIE로 triple을 만들고 entity resolution(embedding 기반 동의어 연결)을 통해 그래프의 모호성을 줄이며, entity-문서 inverted index와 heterogeneous KG(엔티티-엔티티, 엔티티-문서, 문서-문서 간선)로 provenance를 확보했다. 온라인에서는 그래프 검색과 dense 검색을 RRF로 융합한 뒤, LLM이 중간 reasoning history에서 지식 공백을 찾아 sub-question을 새로 생성해 다음 라운드 retrieval을 유도하는 iterative retrieve-reason 루프를 구성했다.

- **Empirical Impact**: 실험은 oracle 문서 없이 closed-book 의료 QA 설정으로 3개 벤치마크(MedQA, MedMCQA, MMLU-Med)에서 수행되었고, Llama-3-8B-Instruct와 GPT-4o-mini 백본 모두에서 일관된 정확도 향상을 보였다. Llama-3-8B 기준으로 가장 강한 baseline(i-MedRAG) 대비 평균 2.1%p, GPT-4o-mini에서도 추가 개선을 달성했으며 듀얼 경로 단독 또는 단순 결합보다 반복 루프가 성능을 더 끌어올리는 것으로 나타났다. ablation 결과는 그래프/ dense를 함께 써도 iteration이 없으면 효과가 제한적이며, retrieval과 reasoning을 하나의 루프로 결합할 때 복잡한 multi-hop 근거 누적이 가장 잘 작동함을 시사한다.



### Improved Large Language Diffusion Models (https://arxiv.org/abs/2606.25331)
- **Prior Approaches**: 기존 LLM은 주로 autoregressive factorization과 causal attention 중심으로 학습돼 왔습니다. Diffusion language model은 masked diffusion을 쓰되 성능이 초기 시도(LLaDA) 수준에 머물러 강한 autoregressive 모델(Qwen2.5 등) 대비 격차가 남아 있었고, 특히 추론·지시응답 성능을 끌어올리는 레시피와 평가/생성 전략 개선 여지가 컸습니다. 또한 multiple-choice에서는 likelihood 기반 점수만 쓰는 경우가 많아 비교 신뢰도에 한계가 있었습니다.

- **Core Contribution**: iLLaDA는 처음부터 학습하는 8B fully bidirectional masked diffusion language model로, pre-training과 SFT 전 과정에서 masked diffusion objective를 유지합니다. pre-training은 12T 토큰 스케일로 확장하고, SFT는 25B 토큰 instruction corpus로 12 epoch를 수행해 지시·추론 능력을 안정적으로 강화합니다. 아울러 variable-length generation(블록 단위 생성)과 confidence-based scoring(다지선다 평가용 점수 규칙)을 도입해 실제 활용/평가에 맞춘 구성을 제시합니다.

- **Technical Challenges**: fully bidirectional diffusion에서 강한 언어 능력을 끌어내려면 스케일링과 학습 안정성이 핵심이며, 이를 위해 grouped-query attention(GQA), input/output embedding tying, 학습률 스케줄(초기 warmup 후 cosine decay 전환)을 조합했습니다. SFT에서는 기존 방식처럼 프롬프트를 고정 마스크하는 대신 pre-training과 동일한 마스킹 포맷으로 프롬프트·응답·종료 토큰까지 마스킹 가능하게 만들어 variable-length 블록 생성을 자연스럽게 지원합니다. 또한 다지선다에서는 likelihood류 상한 대신 confidence 기반 점수를 사용해 평가 성능을 개선했으며, 생성 반복 루프 이슈는 stop-thinking 토큰 확률을 길이에 따라 조절하는 완화로 다뤘습니다.

- **Empirical Impact**: 실험 결과 iLLaDA는 general(예: BBH, ARC-Challenge), 수학(MATH), 코드(HumanEval) 전반에서 LLaDA 대비 폭넓게 향상됐습니다(예: iLLaDA-Base BBH +21.6, ARC-Challenge +14.9, iLLaDA-Instruct MATH +14.5, HumanEval +16.5). iLLaDA는 비 autoregressive 학습임에도 일부 벤치마크에서 Qwen2.5 7B와 경쟁력을 보였고, 특히 multiple-choice에서는 confidence-based scoring이 likelihood 대비 PIQA/ARC-Challenge/HellaSwag에서 일관된 개선을 보였습니다. 전반적으로 fully bidirectional diffusion을 처음부터 학습해도 강력한 언어 모델로 수렴할 수 있음을 실증해, diffusion 기반 LLM 설계의 경쟁 경로를 강화했다는 의미가 있습니다.



### Automatic Generation of Highlights for Academic Paper Via Prompt-based Learning (https://arxiv.org/abs/2606.25253)
- **Prior Approaches**: 기존 연구는 지도학습 기반으로 자동 하이라이트(논문 하이라이트) 추출을 시도했지만, 보통 대규모 라벨 학습 데이터가 필요해 비용과 확장성에 제약이 있다. 또한 하이라이트가 없는 저널이 많아 문헌 검색·텍스트 마이닝·서지분석에 활용을 제한한다는 문제의식이 있었다. 결과적으로 도메인별 학습 코퍼스 의존도가 커지는 경향이 나타났다.

- **Core Contribution**: 이 논문은 학습 데이터 라벨 없이 prompt-based learning으로 자동 하이라이트 생성을 수행하는 접근을 제안한다. 논문 초록을 입력으로 하고, 작업(task)-특화 prompt template를 설계해 생성형 언어모델에 하이라이트 생성을 맡긴다. 특히 task-specific training samples 없이도 기존 지도학습 방법과 비슷한 성능을 달성하고, 소수 예시를 프롬프트에 추가하면 성능이 더 크게 향상된다는 점을 보여준다.

- **Technical Challenges**: 핵심 기술적 난제는 하이라이트 생성이 프롬프트에 포함된 정보에 크게 의존하는 작업이라는 점이다. 저자 작성 하이라이트에 가깝고도 간결한 요지를 뽑아야 하므로, prompt template가 제공해야 할 조건과 입력 구성이 성능을 좌우한다. 연구진은 작업별 템플릿 설계와 여러 언어모델(GPT-2, T5, API 기반 ChatGPT)을 비교해 프롬프트 설계가 생성 품질에 미치는 민감도를 분석하고 보완 방향을 제시한다.

- **Empirical Impact**: 3개 데이터셋 실험에서 ChatGPT + prompt template는 task-specific 학습 없이도 선행 지도학습과 유사한 성능을 보였다. 또한 두 데이터셋에서는 프롬프트에 소수 예시를 추가했을 때 state-of-the-art 대비 큰 개선을 기록했다. 생성 결과는 대체로 문장 일관성·정보성·저자 하이라이트와의 근접성이 확인되며, 도메인 특화 학습 코퍼스에 의존하지 않는다는 점에서 텍스트 마이닝과 bibliometric research의 실용성이 높다는 의미가 있다.



### Towards Structuring an Arabic-English Machine-Readable Dictionary Using Parsing Expression Grammars (https://arxiv.org/abs/2606.25231)
Comments:
          14 pages, 6 figures, 7 tables. The final publication is available at this https URL. Published in International Journal of Computational Linguistics Research (IJCLR), DLINE, March 2014, Vol 5, Issue 1, pp 1-13

- **Prior Approaches**: 기존 연구들은 인쇄 사전의 기계처리를 위해 정규화나 단순 파싱을 시도했지만, 사전 항목의 내부 구조(마이크로스트럭처)가 표준화돼 있지 않아 결과가 들쭉날쭉하다는 한계가 있었습니다. 특히 사전이 ‘사람용 서식’ 중심으로 제작되어, 단어와 구두점 스트림을 그대로 구조화하기 어렵다는 문제가 컸습니다.

- **Core Contribution**: 이 논문은 아랍-영어 Al-Mawrid 사전을 부분적으로 기계 판독 가능한 형태로 구조화하는 방법을 제안합니다. 사전 항목을 계층적 구조로 변환해, 각 엔트리가 subentries로 나뉘고 정의구, domain label, cross-reference, 번역 등가를 명시적으로 표현하도록 설계했습니다.

- **Technical Challenges**: 핵심 과제는 비표준 마이크로스트럭처를 가진 아랍 사전에서 항목의 구성 요소를 자동/반자동으로 안정적으로 분해하는 것입니다. 이를 위해 cascaded steps 중 parsing을 중심으로 하고, parsing expression grammars(Peg)를 사용해 구문 규칙 기반 파서를 구현해 항목 스트림을 계층 구조로 변환했습니다.

- **Empirical Impact**: 저자들은 아랍 사전이 마이크로스트럭처 표준을 따르지 않더라도, 유도된 구조 규칙을 통해 자동 또는 반자동으로 그럴듯한 정확도로 구조화할 수 있음을 보였습니다. 결과적으로 기존의 ‘문서형 사전’을 NLP 응용에 더 적합한 데이터 형태로 전환할 수 있는 실질적 가능성을 제시합니다.



### What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics (https://arxiv.org/abs/2606.25182)
Comments:
          Accepted at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD) 2026. A short version accepted at EIML@ICML 2026

- **Prior Approaches**: 기존 LLM jailbreak 탐지는 프롬프트/출력에 기반한 규칙·분류기·이상탐지, 또는 내부 표현을 읽는 internal-signal 계열로 나뉜다. 그런데 많은 방법이 “어떤 내부 신호가 harmful intent를 담는지”와 “토큰이 전개될 때 불확실성이 어떻게 변하는지”, 그리고 “모델 깊이별로 어디서 신호가 강한지”를 명확히 특정하지 못했다.

- **Core Contribution**: 이 논문은 학습 없이(freezing, training-free) logit lens로 중간 레이어의 token-level predictive entropy를 추적해, jailbreak이 “불확실성의 총량”이 아니라 “토큰 위치에 따른 불확실성 동역학(trajectory)”에 구조화돼 있음을 보인다. 특히 Kendall’s τ, Spearman’s ρ, monotonicity 같은 rank-based 추세 특징이 평균/분산 같은 정적 요약보다 훨씬 잘 jailbreak과 benign을 분리한다고 제시한다.

- **Technical Challenges**: 핵심 난제는 harmful intent가 내부 표현에 존재하더라도, 이를 안정적이고 비교 가능한 수치 특징으로 바꾸는 것이다. 저자들은 단일 forward pass에서 중간 히든을 logit lens로 어휘 공간에 투영해 다음 토큰 엔트로피 시계열을 만들고, 정적(level) 특징과 동적(trend) 특징을 분리해 AUROC로 비교하며, 모델별 “해로운 방향(downward/upward)” 차이도 고려해 directional AUROC를 사용해 공정 평가를 수행한다.

- **Empirical Impact**: Llama, Qwen, Gemma의 여러 adversarial 벤치마크에서 동적 trend 특징은 추가 학습 없이 일관된 분리를 보였고, 신호는 중간 레이어(대략 50–85% 구간)에 집중된 뒤 최종 레이어에서 약화된다. 또한 단순 유사 스타일 안전셋(UltraChat)은 비교적 잘 분리되지만, 구조적으로 악의에 가깝게 만든 JailbreakBench benign에서는 분리가 붕괴해 이 신호가 “문장 의미의 악의성”보다는 “구조적 프롬프트 구성”을 포착함을 시사한다.



### Hitting a Moving Target: Test-Time Adaptation for AI Text Detection under Continual Distribution Shif (https://arxiv.org/abs/2606.25152)
- **Prior Approaches**: 기존 AI 텍스트 탐지는 학습 단계에서 사람/생성 텍스트 라벨이 모두 있는 지도학습에 크게 의존한다. 그러나 배포 후에는 adversarial humanization, 신규 LLM 등장, 인간 문체의 temporal drift 같은 지속적 분포 이동이 생기는데, 이때 라벨이 없거나 확보 비용이 커서 표준 재학습/도메인 적응이 잘 먹히지 않는다. 또한 많은 방법이 테스트 샘플을 독립적으로 보고 test-time homogeneity 같은 현실 신호를 거의 활용하지 못한다.

- **Core Contribution**: 이 논문은 “테스트 시점 적응(test-time adaptation, TTA)”을 AI 텍스트 탐지에 적용하는 프레임워크를 제안한다. 핵심 아이디어는 추론 시점에 관측되는 무라벨 샘플들의 동질성(homogeneity)을 semi-supervised learning으로 활용해, 라벨 없는 분포 이동에도 탐지기를 적응시키는 것이다. 특히 positive-unlabeled(또는 positive-negative-unlabeled) 학습을 TTA에 결합해, shifted AI 텍스트와 인간 텍스트를 구분하는 방향으로 분류 경계를 조정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “지속적”이고 “라벨이 부족한” 분포 이동 속에서, 고정된 분류 경계를 인간화 도구가 계속 공략하는 adversarial 환경을 어떻게 따라잡을지다. 논문은 이 문제를 전략적 분류/performative prediction 관점에서 정리하고, 무라벨 테스트 배치에서 PU/PNU 학습의 최적화가 shifted 분포에서도 사람(또는 LLM 생성물) 쪽을 일관되게 분리하도록 설계될 수 있음을 이론적으로 뒷받침한다. 실험적으로는 (TED)n 학습을 PU + TTA, PNU + TTA로 변형해, 테스트 분포의 무라벨 데이터를 사용하면서도 라벨 없이 적응하도록 구현했다.

- **Empirical Impact**: 실험에서 supervised 탐지기는 adversarial 및 자연 분포 이동(신규 LLM, 인간 문체 변화) 모두에서 체계적으로 성능이 무너졌다. 반면 PU + TTA/PNU + TTA는 상당히 견고해, 예를 들어 Pangram은 adversarial AI-generated 텍스트에 대해 24.1%만 탐지한 반면 제안된 테스트 시점 적응은 90.5%를 탐지했다. 또한 RAID 및 arXiv(추상) 설정에서 유사한 경향이 재현되며, 저자들은 야생(wild) 환경의 AI 텍스트 탐지에서 TTA가 유망한 표준 프레임워크가 될 수 있음을 시사한다.



### The cognitive, affective, and behavioral expression of self-stigma among people who use drugs in online substance use communities (https://arxiv.org/abs/2606.25143)
- **Prior Approaches**: 기존 self-stigma 연구는 주로 SASSS, SU-SMS, BOSS 같은 설문 기반 지표를 사용하며, 대부분이 단면조사라 시간적 전개를 추적하기 어렵습니다. 또한 치료를 받는 사람 중심 표본이 많아 치료 밖 PWUD의 자연스러운 언어를 충분히 반영하지 못한다는 한계가 지적돼 왔습니다. 온라인 텍스트 기반 접근도 있었지만 self-stigma를 ‘단일 현상’으로 보는 경우가 많아 인지-정서-행동 구성요소의 동시출현과 순서를 밝히기 어려웠습니다.

- **Core Contribution**: 이 논문은 self-stigma를 인지, 정서, 행동 도메인의 10개 지표(코드북)로 이론 기반 분해하고, Reddit 글에서 이를 자동 식별하는 혼합방법(합의 코딩+LLM 분류) 프레임워크를 제안합니다. 특히 72,115개 스레드 개시 글과 1,660명의 사용자 장기 기록을 활용해 각 지표의 유병, 동시출현, 그리고 ‘언어 공개 시점’의 순서를 함께 분석합니다. 결과적으로 self-stigma가 단순한 스코어가 아니라 구성요소가 통합적으로 엮이는 현상임을 체계적으로 보여줍니다.

- **Technical Challenges**: 핵심 난제는 (1) 낙인 해석의 주관성 때문에 자연어에서 self-stigma를 안정적으로 분류해야 한다는 점, (2) 설문처럼 미리 정해진 응답으로는 포착되지 않는 자발적 표현을 포착해야 한다는 점입니다. 연구진은 합의 기반 abductive coding으로 10개 지표 정의를 만들고, GPT-4.1-mini를 지표별 이진 분류기로 확장해 다중 호출 후 최빈값(majority voting)으로 라벨을 안정화했으며, 전문가 코딩과의 일치도(k=0.73, F1=0.80)로 성능을 검증했습니다. 시간 분석은 캘린더 시간보다 사용자별 글 순서를 0~1로 정규화해 ‘표현이 먼저 등장하는 위치’를 비교하도록 설계했습니다.

- **Empirical Impact**: Reddit 분석에서 3,838개 글(5.3%)과 1,228명(74.0%)에서 self-stigma 표현이 확인됐고, 10개 지표 모두 non-self-stigma 글과 유의하게 구분됐습니다(RR 3.6~86.2). 사용자 수준에서 행동 지표는 코어(인지·정서) 지표와 강하게 결합해, 행동 지표가 있는 글의 87.0%가 코어 지표도 동반했으며 core-only 없이 행동만 나타난 경우는 매우 드물었습니다(행동-only 5.0%). 시간 전개에서는 ‘진술 순서’ 기준으로 행동 지표가 코어 지표보다 더 이르게 나타났고(예: desire to quit 0.08 vs shame 0.38), 대다수 지표는 궤적 전반에 안정적이되 pessimism/self-defeatism만 심화되는 패턴이 관찰돼 조기 디지털 개입 타깃을 제시합니다.



### Dream at SemEval-2026 Task 13: SALSA for Single-Pass Machine-Generated Code Detection (https://arxiv.org/abs/2606.25102)
Comments:
          Accepted to SemEval-2026, ACL 2026 workshop proceedings

- **Prior Approaches**: 기존 기계 생성 코드 탐지는 보통 제한된 프로그래밍 언어/도메인에 치우치거나 OOD 일반화가 약하다는 한계가 있었다. 또한 LLM을 분류기로 쓰더라도 장문 설명을 내거나 라벨 형식이 들쭉날쭉해져 분류 안정성이 떨어진다는 문제가 자주 언급된다.

- **Core Contribution**: 이 논문은 SemEval-2026 Task 13 Subtask A를 “단일 토큰 라벨 생성” 형태로 재정의하는 SALSA(Single-pass Autoregressive LLM Structured Classification)를 제안한다. 클래스(인간/기계 생성)를 전용 출력 토큰에 매핑하고, 모델이 구조화된 프롬프트에서 그 토큰만 한 번에 내도록 학습시켜 저비용·고신뢰 분류를 노린다.

- **Technical Challenges**: 핵심 기술 과제는 새로운 언어와 응용 도메인에서 흔들리지 않는 OOD 강건성이다. 저학습률, 1 epoch, LoRA 기반 파라미터 효율 fine-tuning으로 과도한 소스 도메인 과적합을 피하면서, (language, label) 그룹별 balanced sampling으로 ‘인간’ 쏠림 같은 데이터 편향이 생기는 것을 줄였다.

- **Empirical Impact**: Qwen2.5-72B-Instruct에 SALSA를 적용한 최고 성능은 OOD F1=0.789로, CodeBERT baseline(F1=0.305)을 크게 앞섰다. 검증셋에서는 거의 완벽에 가까운 F1(0.991~0.996)을 보여 train–val 격차가 작았고, OOD 성능은 모델 스케일에 따라 단조롭게 향상되어 보존된 사전지식의 효과를 시사한다.



### LLM-Based Scientific Peer Review: Methods, Benchmarks, and Reliability Challenges (https://arxiv.org/abs/2606.25057)
- **Prior Approaches**: 기존 연구는 리뷰 파이프라인의 일부만 지원하거나 점수 예측·요약 중심으로 출발해, 최근에는 LLM을 이용한 리뷰 생성과 점수 추정으로 확대됐다. 대표적으로 prompt-based는 형식은 잘 만들지만 점수 보정(calibration)과 신뢰성이 흔들릴 수 있고, supervised fine-tuning은 리뷰 데이터의 잡음·편향을 그대로 물려받는 한계가 있다. retrieval-augmented는 근거를 붙여 환각을 줄이려 하지만, 결국 정보 선택(retrieval) 품질이 병목이 되며 feedback/alignment-optimized는 reward 설계·과최적화(예: reward hacking) 위험이 남는다.

- **Core Contribution**: 이 설문은 LLM 기반 과학 peer review를 critique generation(비평 생성)과 score prediction(점수 예측)이라는 두 축의 의사결정 기능으로 재정의하고, 이를 구현하는 방법들을 prompt-based, supervised, retrieval-augmented, alignment-optimized로 체계화해 분류한다. 또한 벤치마크 결과를 종합하는 데 그치지 않고, 데이터 제약과 평가 결함, 도메인/벤치마크 쏠림이 현재 자동 평가의 한계를 어떻게 만드는지 분석한다. 마지막으로 자동화 리뷰를 고위험·다목표 의사결정 문제로 프레이밍해, 견고성·투명성·신뢰가능성을 달성하기 위한 로드맵을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 리뷰 점수가 단일 정답이 아니라 다수 리뷰어의 불일치와 불확실성을 포함한 분포라는 점, (2) 생성된 문장(비평)과 수치(점수)의 일관성을 유지하는 점, (3) 도메인 이동과 장문의 문서 처리에서의 성능 저하, (4) 보안·전략적 조작에 대한 취약성이다. 설문은 prompt 민감도, 환각, score-text 불일치 같은 신뢰성 결함뿐 아니라 prompt injection, data poisoning, retrieval 취약점, reward hacking 등 자동화 파이프라인을 악용할 수 있는 위험을 정리한다. 또한 subjective disagreement(주관적 불일치) 모델링과 cross-domain 일반화 같은 데이터마이닝 관점의 미해결 과제를 강조한다.

- **Empirical Impact**: 실증적으로 LLM은 유창한 비평과 인간 점수 패턴의 근사치를 만들 수 있지만, 수락/거절과의 상관이 약하거나 과도하게 긍정적으로 치우치는 사례가 반복해서 보고된다. prompt-only는 구조적 유사성은 높아도 깊이·정확성에서 한계가 있고, few-shot·fine-tuning·multi-stage/RAG·multi-agent 설계는 특정 측면(구조 충실도, 근거 기반 구체성, 점수 보정)에 개선을 보이나 전반적 견고성은 충분히 검증되지 않았다고 정리된다. 결론적으로 이 설문은 단순 성능지표를 넘어, 자동화 리뷰가 실제 의사결정에 들어갈 때 필요한 보정·불확실성·안전장치의 연구 중요성을 분야에 명확히 환기한다.



### LLM Performance on a Real, Double-Marked GCSE Benchmark (https://arxiv.org/abs/2606.24973)
- **Prior Approaches**: 기존 자동 채점은 규칙 기반과 신경망을 거쳐 최근에는 LLM을 활용하지만, 표준 영어 에세이 벤치마크에서는 성능이 일관되지 않았고 task-specific 시스템을 완전히 대체하지 못했다. 특히 여러 과목을 한 번에 다루거나 손글씨/그림까지 포함하는 실제 시험 응답에 대해선 여전히 인간과의 신뢰도 격차가 남아 있었다. 그 결과 “사람 채점기준에 맞춰 얼마나 재현하느냐”가 명확히 검증되지 않은 경우가 많았다.

- **Core Contribution**: 이 논문은 GCSE 모의고사 응답 32,534개(5과목, 328문항)를 2명의 자격 심사위원이 이중 채점한 데이터셋을 제시하고, LLM이 두 심사위원 합의(컨센서스)에 얼마나 가깝게 채점하는지 측정한다. 특히 손글씨 작업이 포함된 멀티모달 답안까지 포함해, 기존 텍스트 중심 벤치마크의 한계를 현실 조건에 가깝게 보완한다. 또한 “모델-인간” 단순 상관이 아니라 “모델-인간 합의가 인간-인간 합의만큼 일관적인가”를 지표로 삼는다.

- **Technical Challenges**: 핵심 난제는 (1) 과목별 mark scheme을 따르면서 (2) 손글씨를 읽고 (3) 혼재된 답안 형식(서술형·수학식·도형·그림)을 동일한 흐름으로 점수화하는 것이다. 논문은 단일 generic prompt에 질문·채점기준·학생 답안을 넣고, 손글씨는 캔버스 이미지를 첨부해 읽도록 하며, 출력은 JSON 스키마의 정수 점수로 구조화해 채점 안정성을 확보했다. 그리고 ordinal 점수의 일관성을 위해 QWK(Quadratic Weighted Kappa)를 문항별로 계산한 뒤 가중 평균하고, 오차 구간은 문항 클러스터 부트스트랩으로 추정한다.

- **Empirical Impact**: 결과적으로 최상위 모델들은 대부분 과목에서 “두 심사위원 간 합의”보다 더 높은 수준으로 심사위원 컨센서스에 일치하며, 특히 영어 에세이 같은 주관적 과제에서도 높은 점수대를 만든다. 모델 크기와 무관하게 성능 격차가 크지 않았고, 편향(관대/엄격) 오프셋은 모델별로 뚜렷하게 나타나 특정 모델들이 더 중립적임이 관찰됐다. 또한 비용 분석에서 더 저렴한 모델들이 비싼 모델 대비 유사하거나 더 나은 일치도를 보여, cost-effective automated marking 가능성을 실증적으로 뒷받침한다.



### Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding (https://arxiv.org/abs/2606.24957)
Comments:
          Accepted to ICML 2026. 9 pages main text, includes references and appendix

- **Prior Approaches**: 기존 speculative decoding은 multi-batch·long-context에서 검증 비용이 상대적으로 낮아 처리량을 올리지만, 실제 병목은 target 모델의 KV cache 로딩/접근 비용에 남습니다. 이를 줄이려는 KV 압축은 크게 (1) static eviction처럼 영구 제거를 하거나 (2) dynamic selection처럼 매 스텝 중요도를 다시 계산하는 방식으로 나뉘며, 둘 다 speculative decoding 검증 경로에서는 한계가 뚜렷합니다. static eviction은 saliency shift로 인해 정확도 손실이 생기고, dynamic selection은 중요도 재계산 오버헤드가 커져 검증 지연을 상쇄하지 못합니다.

- **Core Contribution**: 이 논문은 long-context speculative decoding에서 verification bottleneck을 직접 겨냥한 sparse verification 프레임워크 Dustin을 제안합니다. Dustin은 draft 모델의 lookahead 신호와 target 모델의 historical attention을 결합해 다중 단계 검증 윈도우에서도 정확도가 거의 유지되도록 “중요 토큰”을 고정 예산 내에서 선별합니다. 또한 중요도 추정 자체를 Semantic Retrieval Heads(SRHs)로 축소해, 재계산 지연을 최소화하면서도 검증 품질을 보존하는 구조를 갖춥니다.

- **Technical Challenges**: 핵심 기술 난제는 (a) 역사 신호만으로는 검증 단계가 여러 토큰을 동시에 처리하면서 성능이 점점 저하되고, (b) draft의 lookahead 신호만으로는 모델 스케일/페어에 따라 attention 불일치가 커져 ARR이 흔들린다는 점입니다. Dustin은 두 신호를 하이브리드로 융합하는 토큰 선택 정책을 설계하고, SRHs만 재계산하도록 제한해 FlashAttention 등으로 attention matrix 접근이 막히는 제약도 고려해 구현합니다. 더불어 SRH 헤드/레이어 및 KV 예산 분배는 teacher-forced 검증에서 ARR을 목표로 오프라인 최적화해 온라인 부담을 낮춥니다.

- **Empirical Impact**: Qwen2.5-72B에서 32k 길이 조건(배치 16)으로 target self-attention을 27.85× 가속하고, end-to-end decoding도 9.17×까지 빨라졌으며 정확도 저하는 “미미(negligible)” 수준으로 보고됩니다. LongBench와 PG-19 실험에서 Dustin은 StreamingLLM/SnapKV/Quest 등 압축·선별 기반 대안들을 대부분의 범주에서 앞서며, KV 예산이 제한된 상황에서도 verification fidelity를 잘 보존함을 보여줍니다. 결과적으로 long-context speculative decoding의 가속 잠재력을 KV 로딩 중심 병목까지 확장해 실사용 지연을 줄이는 데 의미가 큽니다.



### Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models (https://arxiv.org/abs/2606.24952)
- **Prior Approaches**: 기계적 해석( mechanistic interpretability )의 핵심 기대는 ‘제어가능성’이다. 기존 연구들은 특정 동작을 활성에서 찾아내면, 해당 방향을 residual stream에 더하거나 ablation해 동작을 조절할 수 있다고 보고했으며(예: refusal 매개 단일 선형 방향, truth 관련 방향의 추가), 이를 곧바로 “읽기=제어”로 해석해왔다. 하지만 이 관점에는 ‘검출을 잘하는 방향과 실제로 개입을 일으키는 방향이 같거나 가깝다’는 숨은 가정이 들어 있다.

- **Core Contribution**: 이 논문은 그 가정을 기하학적으로 정량화해, ‘동작 검출 방향’과 ‘동작 개입(제어) 방향’ 사이 각을 각도/코사인으로 측정한다. detection-intervention gap이라는 개념을 제시하며, 코사인이 1에 가깝지 않으면 검출만 하고 제어는 못하는 분리가 존재함을 뜻한다고 본다. 특히 Gemma 2-2B-it에서 output format은 정렬(거의 동일 축)되는 반면, hallucination은 크게 벌어지는 ‘아는 것과 조종하는 것의 분리’를 실험적으로 보여준다.

- **Technical Challenges**: 어떤 방향이 ‘검출’을 최적으로 하는지, 그리고 ‘제어’를 최적으로 하는지를 일관된 방식으로 정의하고 비교하는 것이 핵심 기술 문제다. 연구진은 residual stream에서 두 조건 간 difference-in-means(데이터 기반) 및 lm_head 가중치에서 읽은 토큰 대비(핸드-픽)로 검출 방향을 만들고, 생성 중 residual stream에 α·d를 주입해 개입 방향의 인과적 효과를 확인한다. 그 결과 hallucination에서는 fake entity를 완벽에 가깝게 선형 분리(AUC=1.000)해도, 그 방향과 refusal을 만드는 방향의 코사인이 약 0.12(약 83°)로 크게 어긋난다는 점을 보여준다.

- **Empirical Impact**: 이 gap은 1B–9B 크기의 4개 모델(세 계열)에서 코사인이 [0.12, 0.20] 범위를 벗어나지 않을 정도로 재현되며, instruction tuning 전후에도 거의 동일해(pretraining 기원) 구조적 현상임을 시사한다. 또한 검출-제어 정렬의 코사인이 steerability(조향 가능성)를 예측하리라는 직관은 틀렸는데, detection은 고차원 ‘클래스’이고 실제 조향성은 정적 각도만으로 읽히지 않는 기능적 조건에 달려 있기 때문이다. 요약하면, 이 코사인은 제어 다이얼이라기보다 ‘아는 것과 steering이 분리되는 서명(signature)’으로서 의미가 있다.



### Error-Aware TF-IDF Retrieval-Augmented Generation for ASR Error Correction (https://arxiv.org/abs/2606.24915)
Comments:
          4 pages, 1 figure, 2 tables

- **Prior Approaches**: 기존 ASR-RAG 교정은 희귀 개체/도메인 용어에서 LLM이 환각을 일으키는 문제를 완화하려고 시도돼 왔습니다. 하지만 표준 TF-IDF 기반 검색은 음운 오인(phonetic misrecognition)을 일으킨 토큰을 동일하게 취급해 교정 근거를 충분히 끌어오지 못하고, cross-modal 임베딩 등 복잡한 검색은 높은 지연(latency)과 계산 부담이 큽니다. 또한 엔터티 벡터베이스·Knowledge Graph처럼 무거운 구성은 정렬된 대규모 데이터·고자원 가정이 필요하다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 저자원 언어 페르시아어에서 LLM 기반 ASR 교정 시의 phonetic hallucination과 loop hallucination을 “오류 인식형(lexical error-aware)”으로 직접 겨냥하는 효율적 RAG 프레임워크를 제안합니다. 핵심은 Symmetric Text Normalization으로 KB와 질의 전처리를 동일하게 맞추고, Error-Aware TF-IDF로 과거에 자주 환각되는 토큰에 더 큰 가중치를 부여해 LLM이 교정해야 할 근거 문서를 우선적으로 검색하게 만드는 것입니다. N-best reranking이나 음향 latent/지식그래프에 대한 과도한 결합 없이, 1-best ASR 가설과 희소 행렬 연산만으로 동작하도록 설계됐습니다.

- **Technical Challenges**: 기여를 구현하는 가장 큰 기술적 난제는 (1) 형태/표기 차이로 인한 토큰 미스매치가 검색 공간을 왜곡하고, (2) 루프 환각이 TF-IDF의 term frequency를 망가뜨리며, (3) 표준 TF-IDF처럼 모든 토큰을 동일 가중으로 두면 “오인 토큰이 포함된 교정 문맥”이 점수에서 밀린다는 점입니다. 저자는 ZWNJ·공백·숫자 표기 등을 KB와 질의에 대칭 적용해 벡터공간 일관성을 확보하고, 동일 토큰이 2회 초과 반복되는 구간은 루프 환각으로 보고 잘라내 term frequency 스큐를 줄였습니다. 이어 Error propensity를 기반으로 희소 대각 penalty matrix를 만들어 TF-IDF 행렬에 곱하는 방식으로 계산 지연을 거의 추가하지 않으면서도 오류 토큰을 의도적으로 끌어올렸습니다.

- **Empirical Impact**: FLEURS 페르시아 subset에서 Whisper large-v3-turbo와 Gemini 2.0 Flash-Lite를 사용해 평가했으며, Error-Aware Hit Rate(EA-HR)가 표준 TF-IDF 53.7%에서 제안 방법 90.9%로 크게 상승했습니다. 종단(end-to-end) RAG-ASR 교정에서는 기준 ASR WER 23.06%에서 표준 TF-IDF 21.95%를 거쳐 제안 Error-Aware TF-IDF가 18.83%로 추가 개선을 보였습니다. 특히 해당 성능 향상을 위해 별도의 복잡한 cross-modal 임베딩을 쓰지 않고 희소 행렬 곱 수준의 근접-제로 inference latency만 요구해, 저자원 환경에서도 실용적인 정확도/지연 균형을 제시했다는 점에서 의미가 큽니다.



### AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents (https://arxiv.org/abs/2606.24893)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 텍스트 게임/에이전트 벤치마크는 주로 추론 시점 성능만 평가하거나, 학습을 평가 이전의 오프라인 훈련으로 분리해 test-time learning의 연속성을 보기 어렵다. 또한 시각 기반/저수준 제어 부담이 큰 환경은 탐색·기억·세계지식·장기계획 같은 핵심 능력을 분리 진단하기에 불리하다. test-time continual learning을 다루더라도 5가지 능력(탐색, episodic memory, world knowledge acquisition, skill learning, long-horizon planning)을 함께 조합해 체계적으로 측정하는 평가지가 부족하다.

- **Core Contribution**: 이 논문은 test-time continual learning 에이전트를 위한 평가 프레임워크 AgentOdyssey를 제안한다. AgentOdyssey는 긴 시간축에서 학습과 추론이 배치되는 환경을 만들고, 게임 진행뿐 아니라 세계지식 습득, episodic memory, object/action 탐색, action diversity, 모델 비용까지 진단 평가하는 다면 메트릭을 제공한다. 또한 생성형 텍스트 RPG를 통해 오픈엔디드 장기 과제를 절차적으로 만들 수 있게 하며, 기존 벤치마크의 구조적 공백을 메운다.

- **Technical Challenges**: 핵심 난제는 (1) later task가 이전 탐색·경험·지식에 의존하도록 설계하면서 (2) 엔티티/규칙/목표 수를 진단 가능하게 제어하고 (3) 생성된 게임이 일관되고 검증 가능해야 한다는 점이다. 논문은 POMDP 관점에서 세계를 엔티티 그래프로 두고, action rules(행동 효과)와 step rules(시간에 따른 환경 변화·NPC 행동·확률적 이벤트)를 온톨로지로 구성해 장기 의존성과 불확실성을 동시에 만든다. 이어 자동 테스트 파이프라인으로 랜덤 에이전트까지 포함한 실행 검증과 오류 수정(프로그램 합성/피드백 루프)을 적용해 생성 사운딩을 강화한다.

- **Empirical Impact**: 실험에서는 여러 에이전트 패러다임을 AgentOdyssey에서 평가했으며, base model이 강할수록 성능이 오르지만 최고 에이전트도 인간 수준에 크게 못 미친다는 한계를 확인했다. 또한 meaningful horizon을 좌우하는 요인과, agent 메커니즘 중 short-term memory(STM)가 여러 패러다임에 이득이며 test-time training의 중요한 구성요소임을 보여준다. 결과적으로 5가지 능력의 취약 지점이 무엇인지 정량·진단 가능해져, 향후 에이전트 설계의 개선 방향을 구체화하는 데 의미가 있다.



### Small edits, large models: How Wikipedia advocacy shapes LLM values (https://arxiv.org/abs/2606.24890)
- **Prior Approaches**: 기존 연구는 모델 예측에 영향을 준 학습 데이터를 추적하는 data attribution(예: TrackStar)이나, 특정 문서 제거를 가정하는 MAGIC 같은 counterfactual 추정을 주로 다뤄왔다. 한편 위키피디아는 편집자 집단이 주제를 어떤 관점으로 제시하는지에 영향을 받지만, 이러한 ‘합법적 편집’이 실제로 언어모델의 주제 이해에 미치는 영향은 실증적으로 거의 연결되지 않았다. 또한 데이터 조작 연구는 주로 poisoning 같은 적대적 삽입에 집중해, 시민단체의 가치 기반 편집이 갖는 효과는 상대적으로 덜 조명됐다.

- **Core Contribution**: 이 논문은 동물복지 영역에서 Pro-Animal Wikipedians(PAW)라는 소규모 편집 캠페인이 위키피디아 서술을 어떻게 바꿨는지 추적하고, 그 변화가 LLM의 동물복지 관련 응답(예측 성능/연관성)에 선택적으로 전이되는지를 보여준다. TrackStar(gradient 기반 검색 귀속), MAGIC(훈련 과정 기반 counterfactual 영향 추정), 그리고 fine-tuning 기반 ablation을 같은 문제에 대해 독립적으로 적용해 ‘주제 특이성’을 교차 검증했다. 결과적으로 PAW의 편집은 동물복지 쿼리에 대해서는 영향이 뚜렷하지만, 같은 기업을 언급해도 무관한 일반 쿼리에서는 그 영향이 나타나지 않는다.

- **Technical Challenges**: 핵심 기술 난점은 (1) PAW 편집 텍스트를 통제 텍스트와 분리해 편집-내용 효과를 혼동 없이 측정하고, (2) 어떤 문서가 실제로 모델 예측에 기여했는지 causation에 가깝게 추적하며, (3) 훈련 순서 같은 seed 민감도까지 확보하는 것이다. 저자들은 within-article에서 동물복지(AW) 섹션과 비-AW 섹션을 짝짓는 방식으로 TrackStar 실험의 공변량을 줄였고, MAGIC/ablation에서는 대표적인 위키 텍스트 청크를 통제군으로 써 토픽 간 분리를 유지했다. 또한 Llama 3.1 8B와 Llama-3.2-1B에서 여러 랜덤 훈련-order seed(총 5개)를 돌려, seed에 따라 상위 문서 순위가 흔들릴 수는 있어도 AW vs 대조의 핵심 통계적 방향성은 일관되게 재현되도록 설계했다.

- **Empirical Impact**: 분석 결과, PAW 편집 섹션은 동물복지 쿼리에서 상위 귀속 문서로 과대표집되며(예: TrackStar top-10에서 52% vs 68% 수준의 유의한 비대칭), 일반 쿼리에서는 기준선 근처에서 관측됐다(p<0.0001 대 p=0.53). MAGIC의 counterfactual 영향에서도 모든 seed에서 동물복지 쿼리 상위-10의 영향 문서가 전부 PAW 편집(10/10, 5/5 seeds)인 반면, 일반 쿼리는 우연 수준(4–6/10)에 머물렀다. 마지막으로 PAW 텍스트로 fine-tuning한 모델은 동물복지 텍스트에서 perplexity를 12.4→8.4로 낮추고, 대조 텍스트에 대해서는 효과가 제한되어 텍스트 유형에 특이적인 학습 전이까지 확인했으며, 저자들은 이 메커니즘이 위키피디아가 가중되는 만큼 프론티어 모델에서도 이미 희석되기보다 ‘기반 신호’로 남아 있을 가능성이 크다고 주장한다.



### Graph-Based Phonetic Error Correction of Noisy ASR (https://arxiv.org/abs/2606.24889)
Comments:
          Accepted at ACL Industry Track 2026

- **Prior Approaches**: 기존 ASR은 전체 WER을 낮추지만, 이름 개체·부정·감성어처럼 의미적으로 중요한 토큰에서 잔여 오류가 남는 문제가 계속 지적돼 왔다. 특히 많은 오류가 무작위 잡음이 아니라 음성/발음 유사성에 기반한 구조적 혼동이라서, 단순한 토큰 단위 교정이나 일반적 리라이팅은 부정확한 치환을 낳을 수 있다.
LLM을 프롬프트로 직접 수정에 투입하는 방식은 유연하지만, 무제약 생성에 의존해 환각·과교정·맥락 불안정성이 나타날 여지가 크며, 별도의 음운 제약을 명시적으로 모델링하지 않는다는 한계가 있다.

- **Core Contribution**: 이 논문은 structured ASR correction 프레임워크인 G-SPIN을 제안한다. 핵심은 음운(phonetic) 추론으로 수정 후보 탐색공간을 먼저 제한하고, 그 안에서만 MLM 점수 및 instruction-tuned LLM로 문장 맥락 재랭킹을 수행해 무제약 생성의 위험을 줄이는 것이다.
즉 phonetic reasoning과 contextual semantic selection을 분리해, 발음상 그럴듯함과 의미상 일관성을 동시에 맞추는 접근을 구현한다.

- **Technical Challenges**: 문제는 (1) 어떤 오류 토큰을 고를지(검출), (2) 그 토큰이 될 수 있는 발음상 후보를 얼마나 좁게/정확히 구성할지(후보 생성), (3) 후보 중 하나를 문맥에 맞게 골라야 한다는 점(재랭킹)이다. 논문은 MLM 기반 Contextual Anomaly Detection(CAD)로 의심 단어를 찾고, phoneme-level 그래프에서 GNN이 link-prediction으로 음운 후보 이웃을 구성한 뒤, beam search로 조합 후보를 효율 탐색한다.
마지막으로 LLM은 후보 집합 안에서만 선택하도록 프롬프트를 구성해 더 깊은 의미 추론을 수행하며, 전체 파이프라인은 배포 시 inference-only로 동작(음운 GNN은 사전 학습 후 재사용)하도록 설계했다.

- **Empirical Impact**: 여러 언어(en, es, hi, te)에서 G-SPIN은 WER과 SeMA score에서 강한 기준선 및 LLM 기반 프롬프트 교정 계열을 일관되게 능가했으며, 특히 대치(substitution)와 엔터티 관련 오류의 개선 폭이 컸다. 후보 K(top-k) 크기와 LLM vs MLM 선택 전략을 분석한 결과, 적절한 후보 풀과 LLM 기반 재랭킹이 성능을 좌우함을 확인했다.
또한 삽입·삭제 같은 다른 오류 유형에서는 한계가 관찰되었지만(삭제는 정보 부재로 어려움), 전반적으로 음운 구조를 통합한 ASR 교정의 실용적 적용 가능성을 보여주며 경량·모듈형 inference 프레임워크로서 의미가 크다.



### Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining (https://arxiv.org/abs/2606.26050)
Comments:
          Foundations of Deep Generative Models (FoGen) Workshop at ICML 2026. 23 pages (5-page main text plus appendices), 5 figures. Code: this https URL

- **Prior Approaches**: 기존 grokking/ungrokking 연구는 대체로 학습이 끝난 뒤의 일반화가 데이터 규모나 하이퍼파라미터에 따라 되돌아가거나 지연되는 현상을 다뤘다. 또 일부 작업은 학습이 계속되는 동안 in-context learning 같은 능력이 흐려질 수 있음을 관찰했지만, “능력(규칙) 자체가 언제/왜 남고 사라지는지”를 단일 규칙 단위로 인과적으로 정리하진 못했다.

- **Core Contribution**: 이 논문은 pretraining 한 번의 진행 중에도 언어 규칙(예: 성별 대명사 결정)이 획득되었다가 “within-run reversal”로 붕괴할 수 있음을 pronoun-gender rule 사례로 보여준다. 이를 natural ungrokking이라 부르며, 어떤 규칙이 끝까지 살아남는지가 loss 곡선의 흔적 없이 “말뭉치가 그 규칙을 이기는 횟수( support frequency )”만으로 예측된다고 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 규칙이 사라질 때 그것이 (1) 능력을 통째로 잊은 erasure인지 (2) 규칙을 유지하되 표면적 경쟁 패턴에 밀려서 규칙만 포기한 displacement인지 구분하는 것이다. 연구진은 고정된 템플릿 probe 배터리와 agree/ conflict 조건, 그리고 rule-대상 prior 간 log-probability 대비 여유(margin) 같은 도구로 “구조는 남고 규칙만 밀려난다”는 메커니즘을 단계별로 분해해 검증했다.

- **Empirical Impact**: 여러 말뭉치·데이터 예산·시드에서 support frequency가 높은 규칙은 등장 후 유지되지만, 낮은 규칙은 특정 시점에 행동이 무너지고 margin이 0을 통과하며 예측이 붕괴한다. 특히 같은 편집을 되돌리는 복원(지원 주입) 실험에서는 규칙이 회복되지 않아, 규칙 파괴는 쉽게 가능하지만 복원은 비대칭적으로 어렵다는 결론을 뒷받침한다.



### How Robust is OCR-Reasoning? Evaluating OCR-Reasoning Robustness of Vision-Language Models under Visual Perturbations (https://arxiv.org/abs/2606.26041)
- **Prior Approaches**: 기존 VLM·OCR 벤치마크는 주로 깨끗한 이미지에서 OCR 정확도나 추론 성능을 평가해 왔습니다. 반면 ImageNet-C나 VLM-RobustBench처럼 일반적인 시각 잡음/왜곡 평가지표는 텍스트 추출·구조 기반 추론이 빠져 있어 OCR reasoning의 취약 양상을 분리하기 어렵습니다. 최근 OCR 견고성 벤치마크(CC-OCR 등)도 주로 일반 OCR에 초점을 둬 실제 “시각적 손상→기호 오류→추론 증폭” 경로를 체계적으로 진단하지 못했습니다.

- **Core Contribution**: 이 논문은 시각적 perturbation이 OCR reasoning에 미치는 영향을 통제적으로 측정하는 벤치마크 OCR-Robust를 제안합니다. OCR1.0(문서·장면문자·영수증·손글씨·수학 등)과 OCR2.0(차트·기하 도형·테이블) 두 부분으로 구성해, 자연스러운 텍스트 인식부터 구조화된 시각 추론까지 함께 다룹니다. 또한 18개 후보 perturbation에 대한 사전 파일럿 선택을 통해 glass blur, motion blur, elastic deformation, color shift, snow 5종을 대표 조건으로 확정하고, clean accuracy 외에 RCR/WCR/CRI 같은 견고성 지표 묶음을 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “텍스트 인식을 깨뜨리지 않으면서도” 모델 간 견고성 차이를 드러내는 perturbation을 설계·선정하는 것입니다. 이를 위해 LPIPS 기반으로 동일한 지각적 열화 강도를 세 severity로 맞추고, MRD(영향), SEP(분리성), MON(단조성)으로 18종 후보 중 진단력 있는 5종을 고릅니다. 또 robustness 평가를 clean 대비 상대 유지(RCR), 최악 케이스 위험(WCR), 성능·평균·최악의 균형(기하평균 CRI)으로 나눠 “깨끗한 성능만 높고 최악에서 붕괴” 같은 숨은 취약성을 포착하도록 했습니다.

- **Empirical Impact**: 18개 모델(폐쇄형 VLM, 오픈소스 VLM, OCR+LLM 파이프라인)을 zero-shot으로 평가한 결과, clean accuracy가 높다고 robustness가 비례해 강해지지 않는 패턴이 뚜렷했습니다. 특히 구조 의존 입력(차트·테이블)이 문서형 입력보다 훨씬 취약해 평균 WCR이 OCR1.0에서 0.826, OCR2.0에서 0.676으로 더 크게 하락했습니다. OCR+LLM은 OCR 추출 품질에 민감했고, thinking mode나 CoT 프롬프트는 clean 정확도는 개선해도 worst-case 실패를 항상 막지는 못해 OCR reasoning 견고성을 별도 문제로 다뤄야 함을 시사합니다.



### Autodata: An agentic data scientist to create high quality synthetic data (https://arxiv.org/abs/2606.25996)
- **Prior Approaches**: 기존 합성 데이터 생성은 Self-Instruct 계열처럼 zero-shot/few-shot 프롬프트로 데이터를 만들거나, Grounded Self-Instruct처럼 문서 근거를 추가해 환각을 줄였다. CoT Self-Instruct는 생성 과정에 Chain-of-Thought를 넣어 복잡한 문제를 만들지만, 데이터의 난이도·품질을 직접 제어하긴 어렵다. 따라서 filtering, evolution, refinement 같은 사후 기법이나 “self-challenging” 상호작용이 등장했지만, 결국 생성 품질을 일관되게 보장하기는 한계가 있었다.

- **Core Contribution**: 이 논문은 Autodata라는 일반 프레임워크를 제안해, AI 에이전트가 데이터 과학자처럼 학습·평가용 고품질 데이터를 만들고(생성) 점검·개선(분석)하도록 한다. 특히 데이터를 “만든 뒤 평가하고, 그 피드백으로 다음 레시피를 개선”하는 루프를 설계하고, 이 에이전트 자체를 meta-optimize해 더 강한 데이터를 생성하도록 한다. 실용 구현으로는 Agentic Self-Instruct를 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘강한 모델에게는 쉬운데 약한 모델 학습에는 의미 있는’ 데이터의 난이도/품질을 안정적으로 맞추는 것이다. CS 연구 질의에선 CoT 기반 데이터가 너무 쉬워 discriminative signal이 약해지는 문제가 있어, strong/weak solver 점수 차이(gap)를 기준으로 수용 조건을 엄격히 두고 반복 생성·수정했다. 반대로 법률 추론에선 CoT 데이터가 너무 어려워 GRPO 보상 신호가 0으로 쏠리는 문제가 있어, 고정 임계값 대신 롤아웃 패턴과 rubric을 함께 보고 grpo_suitability까지 판단하는 유연한 루프 판정을 설계했다.

- **Empirical Impact**: 실험에서 Agentic Self-Instruct는 CS 연구 질문, 법률 추론, 수학 객체 추론(Principia 계열) 전반에서 기존 classical synthetic dataset 생성 대비 성능을 개선했다. 특히 RL 학습에서 CS에선 약한 모델이 더 큰 격차를 학습하도록 데이터가 조정되며, 법률에선 너무 강한 학습 신호를 완화해 학습 가능성을 높였다. 또한 수학 객체 추론에서도 더 어려운 예제를 반복적으로 생성해 평균 성능과 out-of-distribution 성능 모두에서 전이 이득을 보이며, “추론 compute를 더 좋은 데이터 학습으로 전환”하는 접근의 의미를 실증한다.



### How Large Language Models Source Brand Reputation Across Languages and Markets (https://arxiv.org/abs/2606.25787)
Comments:
          12 pages, no figures, tables only. Data and analysis ledger on Zenodo, this https URL

- **Prior Approaches**: 기존 AI 브랜드 가시성 연구는 주로 모델이 생성한 답변 텍스트(감성, 추천, 경쟁사 언급 등)만 분석했다. 하지만 생성형·그라운딩된 모델은 답변을 웹에서 가져온 뒤 인용에 기반해 말하기 때문에, 실제로는 답변 내용보다 인용(출처) 레이어가 더 먼저 결정한다. 또한 AI 검색은 전통 검색과 달리 인용 출처가 특정 채널에 집중되고, 엔진·시점에 따라 출처 구성이 불안정해질 수 있어 출처 측정의 필요성이 제기돼 왔다.

- **Core Contribution**: 이 논문은 “LLM이 브랜드 정보를 어디서 가져오는가”를 답변 텍스트가 아니라 citations(인용 URL/도메인)로 한 단계 앞에서 측정한다. 12개 홈마켓·13개 언어·128개 브랜드에 대해 167,551개 URL-grounded citations(총 189,974개 attribution row)를 도메인과 source type으로 분류해 언어·시장별로 인용 구조를 정량화했다. 그 결과, 브랜드 자체가 아니라 제3자 웹이 AI 브랜드 답변의 핵심 근거가 되는 패턴을 교차 시장에서 확인했다.

- **Technical Challenges**: 핵심 난제는 인용 URL을 신뢰할 수 있는 도메인으로 정규화하고, “소유(owned)” 여부를 정확히 판정하는 것이었다. 특히 Gemini 인용에서 Google redirector(정점 기반 리다이렉트)가 섞여 owned 판정이 0%처럼 보일 수 있어, citation-title에서 실제 도메인을 100% 해소(resolve)한 뒤 분석했다. 또한 데이터가 단일 균질 표가 아니라 “backbone+cross-link” 구조라 URL 기반 지표는 NB+PL 중심으로 계산하고 CEE는 별도 보고했으며, 인용은 엔티티에 고정되지 않고 응답에 연결돼 브랜드별 수치에는 베이스 차이와 해석 유의가 필요하다고 밝혔다.

- **Empirical Impact**: 분석 결과, AI는 브랜드 답변을 85.7%에서 비소유(제3자) 사이트에 근거하고 14.3%만 브랜드 소유 사이트를 인용한다. 도메인 인용은 강하게 집중돼 80%가 전체 도메인의 약 18%에서 나오며 Zipf 법칙(α=0.86, R²=0.983)에 부합했다. 또한 Wikipedia는 12개 언어 중 11개 언어에서 최다 인용 도메인이고, 리투아니아는 vz.lt가 예외로 나타났으며, 폴란드에서는 YouTube와 HR·커리어 포털(Indeed 등)이 Wikipedia를 제치며 시장 특이성이 드러났다. 마지막으로 모델별로 인용 범위가 달라 Perplexity는 가장 많이·넓게 인용하고 Gemini는 redirector 해소 전에는 소스가 숨겨져 보일 수 있어, 단일 모델·단일 audit만으로는 편향 위험이 있음을 시사한다.



### Space-Efficient Language Generation in the Lim (https://arxiv.org/abs/2606.25777)
Comments:
          Accepted at COLT 2026

- **Prior Approaches**: 기존 언어 생성/학습 이론은 보통 충분한 자원을 가정하거나, 산출물의 품질(환각 금지, 누락 허용치)과 메모리 제약을 동시에 날카롭게 다루지 못했다. 특히 스트리밍 환경에서 제한된 메모리로 목표 언어를 얼마나 정확히 ‘생성 범위’로 보장할 수 있는지에 대한 정량 경계가 부족했다.

- **Core Contribution**: 이 논문은 공간 효율의 최소 제약 아래 ‘limit에서의 language generation’을 자원 인식(resource-aware) 관점으로 정식화한다. 학습자는 목표 언어 K에서 나온 적대적( adversarial ) 양성 스트림을 관찰한 뒤, hallucination-free로 L ⊆ K를 만들되 K에서 최대 Δ개의 문자열만 생략하도록 수렴하는 것을 목표로 한다. 또한 가설 클래스를 DFA 상태 수가 s인 언어 집합 C_{s,k}로 제한해, 자원에 따른 달성 가능성을 이론적으로 분해한다.

- **Technical Challenges**: 핵심 난제는 스트리밍에서 메모리가 제한될 때, 보이는 양성 정보만으로도 환각 없이(즉 L ⊆ K) 누락 허용치 Δ까지 제어하며 수렴하는 전략을 설계하는 것이다. 저자들은 poly(s,k) 공간의 스트리밍 알고리즘을 제안해 Δ = O(k^{2s-2})의 generation gap을 달성하며, 길이 2s-1 이상인 K의 모든 문자열을 포착함을 보장한다. 더 나아가 표준 communication complexity 문제로부터 하한을 환원해, Δ ≤ k^{(1-ε)s}를 얻으려면 k^{Ω(εs)} 메모리가 필요함을 보여 경계가 거의 맞물림을 입증한다.

- **Empirical Impact**: 실험보다는 수학적 정당화에 초점을 둔 결과로, polynomial-space 수준에서는 일정 generation gap 안에서 포착을 보장하지만, exponential-space에서만 목표 K를 정확히 식별(exact identification)할 수 있음을 ‘sharp transition’으로 제시한다. 이는 제한된 메모리로 언어 생성 품질을 보장하려는 이론 설계에서, 필요한 자원과 성능(생성 갭·누락 규모) 사이의 구체적 트레이드오프를 제공한다. 결과적으로 스트리밍·메모리-bounded 학습에서 이론적 한계와 설계 방향을 동시에 명확히 해준다.



### Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets (https://arxiv.org/abs/2606.25760)
- **Prior Approaches**: 기존 컴퓨터-사용 에이전트 연구는 GUI 클릭 정합에서 불확실성을 다루더라도, 보통 특정 VLM-벤치마크-인터페이스 조합에 한정돼 사후 UQ(uncertainty quantification) 선택이 다른 조건에서도 유지되는지 불명확했습니다. 또한 open-weight와 closed-source 환경을 같은 프로토콜로 비교하는 연구가 적어, 로그it·hidden state·attention 같은 관측 가능 신호 유무가 UQ 순위에 미치는 영향을 정량화하기 어려웠습니다.

- **Core Contribution**: 이 논문은 단일-step GUI grounding에서 사후 UQ가 “어떤 배치(에이전트/데이터셋/관측 가능한 인터페이스)”로 옮겨 갈 때도 오류 순위가 유지되는지 교차-레짐 일반화를 체계적으로 측정하는 Argus를 제안합니다. Argus는 open-weight 4개 VLM 에이전트와 4개 데이터셋에 대해 27개 방법을 2,727개 스코어로, 내부 신호를 얻기 어려운 closed-source는 3개 벤더×동일 데이터셋 조건에 대해 8개 방법을 API-only 패널로 구성해 비교할 수 있게 합니다.

- **Technical Challenges**: 핵심 과제는 UQ “스코어”가 아니라 “방법 순위”가 레짐 전환에도 안정적인지 검증하는 것으로, 이를 위해 오류 탐지·선택적 실행·캘리브레이션·miss-severity 랭킹·공간 click-영역(동심 원/원형 구간) 커버리지까지 다목적 지표로 평가를 분리했습니다. 또한 closed-source에서는 로그it/hidden/attention이 없으므로 공통 API-호환 방식으로 비교 가능한 방법 부분만 남기고, 관측 불가능 신호 기반 방법은 동일 공식의 대체(프록시)로 통합 패널을 구성해 비교의 공정성을 확보했습니다.

- **Empirical Impact**: 결과적으로 UQ 방법 순위는 고정된 모델 내에서는 데이터셋 전환에도 비교적 안정적이지만, 모델 클래스 전환과 관측 인터페이스 변화(API-only)에서는 안정성이 크게 저하되는 “selective transfer” 양상이 나타났습니다. open-weight에서는 hidden-state·density(예: SEP/SAPLMA 계열)가 전반적으로 가장 안정적인 편이지만, closed-source로 갈수록 평균 순위 이전 성능이 거의 0에 가까워져 타겟 환경에서 재캘리브레이션/재랭킹이 필요하다는 결론을 제시합니다. 또한 conformal click-disk는 점수만으로 배치 가능한 커버리지를 보장하지 못해, 플러그인 UQ 캘리브레이션 시 반경이 40~60% 줄어드는 대신 커버리지는 조건에 따라 악화될 수 있음을 실증했습니다.



### RAS: Measuring LLM Safety Through Refusal Alignmen (https://arxiv.org/abs/2606.25750)
- **Prior Approaches**: 기존 LLM 안전 평가는 위험 프롬프트로 생성된 응답을 검사해 안전정책 위반 여부를 판정하는 출력(outcome) 기반이 중심이었다. SafetyBench, HarmBench, JailbreakBench 등은 생성 결과를 사람이 라벨링하거나 LLM-as-a-judge로 판단하는 방식이라 비용이 크고, 디코딩/저지 성격/문항은행 고착에 민감하다는 한계가 있었다. 또한 안전성은 출력에서의 거부 여부처럼 ‘늦게’ 드러나 내부에서 안전 신호가 언제 약해지는지 파악하기 어렵다.

- **Core Contribution**: 이 논문은 화이트박스 접근으로 LLM의 내부 표현에서 ‘거부(refusal) 정렬’ 여부를 측정하는 SafeVec을 제안한다. 안전정렬된 기준 모델로부터 레이어별 refusal direction을 뽑고, 안전/위험 및 jailbreak 프롬프트에서 타깃 모델의 hidden state가 그 방향과 얼마나 일치하는지로 점수 RAS(Refusal Alignment Score, 0~100)를 산출한다. RAS는 출력 생성 없이도 안전 신호를 조기에 포착하는 표현 수준 안전 지표로, 기존 출력 기반 평가를 대체하기보다는 빠른 보조/감사를 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 모든 레이어가 동일하게 유용하지 않고 (2) 모델 계열별 표현 스케일과 코사인 값 비교가 어렵다는 점이다. SafeVec은 기준 모델에서 safe/unsafe 대비로 refusal direction을 추정한 뒤, safe/unsafe 분리가 안정적으로 나타나는 연속 레이어 윈도우를 선택하고, calibration 모델들로 원시 정렬 점수를 family별 RAS 스케일로 보정한다. 그 결과 타깃 모델은 unsafe와 jailbreak 프롬프트에서의 정렬 코사인 유사도를 계산해 calibrated RAS를 얻는다.

- **Empirical Impact**: Llama, Gemma, Qwen 3개 모델 계열에서 RAS는 aligned 모델을 uncensored·abliterated 변형과 분리하며, 출력 수준의 공격 성공률 ASR과도 강하게 경향을 맞춘다. 또한 RAS가 judge 기반 평가보다 훨씬 빠르며(평균 약 210배 speedup), forward pass 중심이라 대규모 회귀 테스트에 적합하다는 점을 보여준다. 저자들은 RAS가 ‘모든 요청에서 거부를 보장’하진 않지만, 내부 refusal 관련 상태의 약화 가능성을 효율적으로 선별하는 compact한 안전성 신호라고 정리한다.



### Tracing Target Answers in Poisoned Retrieval Corpora via Token Influence Attribution (https://arxiv.org/abs/2606.25721)
- **Prior Approaches**: RAG corpus poisoning은 악의적 문서를 검색에 포함시켜 LLM의 생성을 특정 오답으로 유도하는 공격이다. 기존 탐지는 보조 분류기나 추가 LLM 검증처럼 별도 모델/추론을 붙여 정확하더라도 계산·운영 비용이 커지는 문제가 있었다. 문서 단계 필터링·포렌식류도 존재하지만, 정답을 유도하는 토큰을 직접 추적해 확인하는 방식은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: TRACE는 token influence attribution을 활용해 “오답(공격자가 지정한 타깃 답)”에 강하게 영향을 주는 토큰을 찾아내는 RAG poisoning 탐지 프레임워크다. 문서들을 benign/malicious로 분류하는 대신, 여러 문서에서 반복적으로 등장하는 고영향 키워드(토큰)를 뽑고 2단계 검증으로 포이즈닝을 확정한다. 또한 탐지와 동시에 공격자가 심은 타깃 답 토큰을 드러낼 수 있음을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 비용이 낮으면서도, 잡음 토큰(구두점·빈도 높은 단어 등)을 배제하고 정답 유도 토큰을 안정적으로 포착하는 것이다. TRACE는 먼저 검색 문서마다 “긍정 affirmation 문자열”에 대한 영향도를 backpropagation 기반으로 계산해 top-influence 토큰을 뽑되, 구두점과 빈도 단어 및 질문에 이미 있는 토큰은 제외한다. 이후 인접 토큰을 연속 구문 후보로 결합하고 다문서 공통 반복 빈도로 후보를 정제한 뒤, 두 번째 영향도 재평가로 같은 고영향 토큰이 반복되는지 확인해 공격 여부를 판정한다.

- **Empirical Impact**: PoisonedRAG 기반 공격과 Natural Questions, HotpotQA, MS-MARCO에서 6개 LLM을 대상으로 실험한 결과, TRACE는 TPR을 대부분 90% 이상으로 유지하며 FPR도 하이퍼파라미터 튜닝으로 10% 미만 수준까지 억제할 수 있었다. Phi-4-mini에서는 평균 TPR 98.87%로 특히 강한 탐지 성능을 보였고, 실행 시간도 LLM 기반 포렌식 대비 낮게 보고됐다. 더불어 타깃 답 식별 ACC가 약 70~90%대(HotpotQA는 상대적으로 하락)로 나타나, TRACE가 포이즈닝 탐지뿐 아니라 공격자가 지정한 답을 상당 부분 역추적함을 시사한다.



### Security and Privacy in Retrieval-Augmented Generation: Architectures, Threats, Defenses, and Future Directions for Building Trustworthy Systems (https://arxiv.org/abs/2606.25533)
- **Prior Approaches**: 기존 RAG 연구는 주로 성능·구조 관점에서 검색 파이프라인과 생성 품질을 개선하는 데 집중해 왔고, 보안·프라이버시는 LLM 일반 위협(예: 프롬프트 인젝션)이나 단일 배치(중앙화, 엣지, 연합)별로 따로 다뤄지는 경우가 많았습니다. 그 결과 centralized RAG에서는 쿼리/컨텍스트 유출 같은 위험이 부각됐지만, on-device(Micro-RAG)·federated RAG·hybrid에서는 확대되는 복합 위협을 일관된 틀로 매핑하기가 어려웠습니다. 또한 문맥 구성(context construction)과 증거 패킹(토큰 예산, 순서, 잘림 등) 단계가 생성 신뢰성과 프라이버리에 미치는 영향은 상대적으로 덜 조명되었습니다.

- **Core Contribution**: 이 논문은 centralized, on-device(Micro-RAG), federated, hybrid edge–cloud로 이어지는 전개를 기준으로 RAG 배포를 통일적으로 분류하고, 각 배포에서 발생하는 보안·프라이버드 위협 표면을 한 프레임으로 정리합니다. 특히 retrieval- context construction- generation 단계에 더해 로컬 실행 및 연합 업데이트까지 포함해 membership inference, index inference, poisoning, gradient leakage, collusion 등 공격 유형을 체계화합니다. 아울러 context construction 관점을 전면에 두고, 제한된 컨텍스트 예산과 증거 배치가 견고성·프라이버리·사실성에 미치는 취약점도 강조합니다.

- **Technical Challenges**: 핵심 기술적 난제는 RAG 파이프라인 전반에서 ‘외부 지식’이 신뢰되지 않은 입력이 될 수 있다는 점과, 이를 단계별로 분리·검증하면서도 지연시간·정확도·온디바이스 자원 제약을 동시에 만족해야 한다는 것입니다. 논문은 이를 해결하기 위해 query filtering, retrieval 보호, 프라이버리-aware 컨텍스트 조립, 생성 단계 검증 등 defense-in-depth를 조합하고, 프라이버시-유틸리티 트레이드오프 및 배포 환경(중앙화/엣지/연합)에 따른 고려사항을 함께 정리합니다. 또한 문맥 패킹의 상호작용(예: truncation 경계, ‘lost-in-the-middle’ 효과, 길이 지배) 때문에 문서 단품 평가로는 탐지하기 어려운 공격을 구조적 조립·다양성 제약·근거 검증 같은 방식으로 다루려는 방향을 제시합니다.

- **Empirical Impact**: 실증적 영향은 보안·프라이버리·효율성·연합 학습 관점에서 RAG을 평가하는 벤치마크/데이터셋/메트릭스 지형을 폭넓게 정리함으로써, 연구자가 위협 모델에 맞춘 실험 설계를 할 수 있게 한다는 데 있습니다. 더불어 공격-방어가 파이프라인 전 단계에 걸쳐 누적되며 ‘단일 방어’만으로는 부족하다는 점을 체계적으로 보여, 현업 적용 시 어떤 계층(검색/컨텍스트/생성/업데이트)을 우선 보호해야 하는지 연구 의제를 명확히 합니다. 결론적으로 본 조사는 실제 서비스에서 신뢰할 수 있고 견고한 RAG를 만들기 위한 오픈 챌린지를 정리해, 향후 표준화된 평가와 배포 지향형 보안 연구의 기반을 제공하는 역할을 합니다.



### Evaluating LLMs on Real-World Software Performance Optimization (https://arxiv.org/abs/2606.25530)
- **Prior Approaches**: 기존 소프트웨어 성능 최적화 연구/도구는 개별 함수 수준의 단순화된 문제나 단일 성능 지표에 치우치는 경향이 있어, 실행 시간과 메모리 사용 사이의 핵심 트레이드오프와 실제 코드베이스의 측정 잡음, 입력·실행 조건의 변동성을 충분히 반영하지 못했다. 또한 LLM 코드 리파인먼트를 활용해도 최적화가 ‘어떻게’ 이뤄지는지 재현·검증할 벤치마크가 부족해 성능 향상을 비교하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 실제 오픈소스 프로젝트에서 수집한 전문가 최적화 102개를 바탕으로 레포지토리 수준 벤치마크 SWE-Pro를 제안한다. 각 작업(task)을 파라미터화 테스트로 연결해 다양한 입력 데이터와 실행 조건에서도 runtime, peak memory, Time-Weighted Memory Usage(TWMU)를 동시에 평가하도록 설계했다. 이를 통해 최적화가 단순 개선이 아니라 조건·잡음이 있는 환경에서의 복합 최적화 문제임을 정량화한다.

- **Technical Challenges**: 레포지토리 수준 최적화에서는 최적화 효과가 특정 입력에서만 발생하거나 측정 환경의 잡음에 의해 성능이 흔들릴 수 있어, 신뢰 가능한 평가 체계를 만드는 것이 핵심 기술 난제였다. 논문은 noise-aware measurement 조건을 두고 작업별로 파라미터화 테스트를 적용해 runtime·peak memory·TWMU를 다양한 조건에서 측정함으로써 변동성과 트레이드오프를 벤치마크 자체에 반영했다. 그 결과 단일 점수 중심의 과잉 단순화를 피하면서 최적화의 실질 난도를 재현할 수 있게 했다.

- **Empirical Impact**: 실험에서 현재 LLM들은 대부분의 작업에서 runtime 이득이 미미하고, 메모리 최적화는 거의 나타나지 않았다. 반면 전문가 구현은 전체 작업 기준 aggregate speedup 15.5x와 peak memory 171.3x 감소를 달성했으며, runtime은 91.2% 작업에서, peak memory는 65.7% 작업에서 개선이 관측됐다. 이러한 격차는 ‘전문가 수준 엔지니어링’이 요구하는 최적화 능력과 현재 LLM의 한계가 크다는 점을 실증적으로 보여준다.



### Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning (https://arxiv.org/abs/2606.25524)
- **Prior Approaches**: 기존 연구는 실패를 추적할 때 주로 단계/청크/문장 같은 더 큰 단위에 집중하거나, 실패가 이미 발생한 뒤의 토큰·구간을 분석하는 방식이 많았습니다. 일부는 토큰 entropy 같은 불확실성 신호나 라인 프로브, 또는 rollouts로 prefix-conditional 성패를 추정해 원인을 추정합니다. 하지만 어떤 ‘정확한 토큰’이 실패로 기울기(collapse) 시작하는지, 그리고 그 변화가 샘플링 잡음인지 통계적으로 구분하는 방법은 불명확했습니다.

- **Core Contribution**: 이 논문은 reasoning trace에서 정답 도달 가능성이 급격히 떨어지는 ‘cliff token(절벽 토큰)’을 정의하고, 이를 통계적으로 식별하는 프레임워크를 제안합니다. cliff token은 토큰-wise potential이 직전 토큰 대비 유의하게 하락하는 지점으로, 실패 전환의 트리거로 작동함을 주장합니다. 또한 greedy choice와 토큰 entropy에 기반해 cliff를 deterministic/uncertain/sampled-off 세 유형으로 분류합니다.

- **Technical Challenges**: 가장 큰 문제는 토큰 위치마다 잠재력(token-wise potential)을 rollouts로 추정할 때 변동성이 커서 절대 임계값만으로는 가짜 양성(false positive)이 생긴다는 점입니다. 이 연구는 한쪽 one-sided two-proportion z-test를 쓰는 adaptive threshold로, 국소 샘플링 분산을 반영해 통계적으로 유의한 ‘도약’만 cliff로 판정합니다. 이후 cliff 유형별로 greedy 여부와 entropy 조건을 결합해 서로 다른 실패 모드를 분리하고, single-token preference optimization을 Cliff-DPO로 구현해 cliff 위치에만 학습 신호를 국소화합니다.

- **Empirical Impact**: 7개 모델과 GSM1K, MATH500, AIME 2025 3개 벤치마크에서 cliff token은 실패 트리거로 관찰되며, cliff token을 삭제하고 resample하면 pass@64가 1.0까지 회복되는 반면 cliff를 유지하면 0.71~1.00에 머뭅니다. taxonomy 관점에서도 deterministic보다 uncertain과 sampled-off cliff에서 회복/학습 이득이 일관되게 나타났고, Cliff-DPO는 학습 후 정확도를 최대 +6.6만큼 개선합니다. 특히 uncertain+sampled-off에 집중한 변형은 cDPO 대비 훨씬 적은 손실 기여 토큰으로도 경쟁력(그리고 GSM1K에서 우위)을 보이며, reasoning 실패의 ‘행동 가능한 학습 신호’로서 의미가 큽니다.



### Fully Differentiable Neural Forced Alignment via Soft Dynamic Programming (https://arxiv.org/abs/2606.25460)
Comments:
          This work has been submitted to the IEEE for a possible publication

- **Prior Approaches**: 기존 forced alignment(FA)는 전통적인 HMM-GMM과 Viterbi 디코딩을 기반으로 하며, Montreal Forced Aligner 같은 도구가 널리 쓰일 만큼 경계(phoneme boundary) 정확도가 강점으로 평가돼 왔습니다. 다만 이러한 방식은 발음사전과 G2P(grapheme-to-phoneme) 변환에 의존해, 실제 발화의 발음 변이와 불일치하면 정렬 정확도가 떨어질 수 있습니다. 최근에는 WhisperX, MMS, Canary-1B 등 neural ASR 후처리로 timestamp를 얻는 접근이 늘었지만, 대체로 인식 최적화가 중심이라 경계 정밀도 자체를 직접 학습하긴 어렵습니다.

- **Core Contribution**: 이 논문은 phoneme 단위 경계 추정을 위해 설계된 end-to-end, fully differentiable 신경망 FA 모델을 제안합니다. 모델은 (1) phoneme identity verification에 특화된 인코더 분기와 (2) phoneme boundary detection에 특화된 인코더 분기를 두고, 디코더는 differentiable soft dynamic programming(soft-DP)으로 정렬 결정을 내립니다. 또한 phoneme 내부의 안정 구간과 전이 경계를 더 잘 분리하도록 contrastive loss(MNCE)를 새롭게 도입해, 경계 예측을 명시적으로 학습하도록 구성했습니다.

- **Technical Challenges**: 핵심 난제는 “인식용” 모델에서 흔한 토큰 디코딩 부산물 timestamp를 넘어, phoneme 경계를 직접 정밀 최적화하면서도 학습이 안정적으로 end-to-end 되도록 만드는 점입니다. 이를 위해 MNCE는 프레임을 phoneme 내부의 positive set과 경계 주변의 negative set으로 나눠 latent space에서 분리를 유도하고, context encoder는 문맥 정보를 모아 프레임별 phoneme posterior를 생성합니다. 마지막으로 soft-DP에서 hard max와 비미분 backtracking(절대 arg-max)을 log-sum-exp와 expected value 형태로 대체해 gradient flow를 유지하면서 timestamp 회귀까지 이어지게 했습니다.

- **Empirical Impact**: 실험에서는 손수 정렬된 영어 벤치마크(TIMIT, Buckeye)에서 기존 HMM-GMM 기반 aligner와 최근 neural alignment보다 phoneme boundary 정확도가 향상되었다고 보고합니다. 더 나아가 word-level 일반화 성능이 강하고, 영어 학습 이후에도 네덜란드어/독일어/히브리어처럼 보지 못한 언어에 대해 일반화하는 결과를 보여줍니다. 즉, 발음사전·G2P 의존성이 약한 end-to-end FA 설계가 경계 정밀도와 교차언어 전이를 동시에 노릴 수 있음을 시사합니다.



### The Generalization Spectrum: A Chromatographic Approach to Evaluating Learning Algorithms (https://arxiv.org/abs/2606.25450)
Comments:
          Accepted at ICML 2026. 30 pages, 6 figures

- **Prior Approaches**: 기존 평가는 i.i.d. 테스트에서 최종 aggregate 점수(pass@1 등)만 보며, 특정 학습 예제가 다른 예제로 얼마나 일반화되는지(일종의 per-sample generalization)는 가려진다. in/out-of-distribution 같은 이분법 비교도 “얼마나 멀리까지”를 보여주지 못해, memorization·near transfer·far transfer의 질적 차이를 구분하기 어렵다. 
이 문제를 해결하려면 학습이 예제 단위에서 어느 거리까지 전파되는지 ‘거리 축’으로 분해해 측정해야 한다.

- **Core Contribution**: 이 논문은 Generalization Spectrum(일반화 스펙트럼)이라는 거리 기반 평가 프레임워크를 제안해, 학습을 단일 점수가 아닌 Generalization Profile(거리별 성능 곡선)로 드러낸다. 각 학습 예제(seed)로부터 D0(정확 재현)→D1(언어 구현만 변경)→D2(서사 전면 재프레이밍)→D3(알고리즘 카테고리 매칭)→D4(비쌍 기준)까지 전이 거리를 늘린 테스트 변형을 구성한다. 
또한 matched-memorization(기준 D0 성능을 맞춘 비교)로 “더 잘 외워서 그런 것”과 “외운 것을 얼마나 멀리 전이하는지”를 분리해 진단 가능하게 만든다.

- **Technical Challenges**: 전이 효율을 공정하게 비교하려면, 학습이 진행될수록 memorization과 transfer가 동시에 커지는 공변성을 통제해야 한다. 논문은 서로 다른 학습 패러다임(ICL/SFT/RL)을 대상으로 D0에서 동일한 pass@1 수준을 달성하는 체크포인트를 선택해 D1~D4에서의 차이를 해석하도록 설계한다. 
또한 competitive programming을 사용해 correctness가 실행 기반으로 명확하고, D2 변형은 교차 모델 생성·검증 파이프라인으로 수학적 구조는 유지하면서 서사만 크게 바꾸도록 구현해 전이 거리의 의미를 고정한다.

- **Empirical Impact**: Generalization Spectrum으로 분석한 결과, RL은 matched memorization 조건에서 near transfer뿐 아니라 far transfer까지 0 이하로 붕괴하지 않는 양의 전이 프로파일을 보인다. 반면 reference-solution 중심 SFT는 언어·서사 변화 같은 거리 증가에서 급격히 약해져, local 이득이 곧 generalization radius 확장으로 이어지지 않음을 드러냈다. 
ICL은 강하지만 correspondence(쌍 매칭) 의존성이 크고, self-distillation·hint-assisted RL 등 내부 변형은 성능 높이기보다 스펙트럼의 특정 구간을 재형성하며 far tail을 오히려 줄일 수도 있음을 보여 후속 방법 설계에 직접적인 진단 도구로 활용될 수 있다.



### The Interplay of Harness Design and Post-Training in LLM Agents (https://arxiv.org/abs/2606.25447)
- **Prior Approaches**: 도구를 호출하는 LLM 에이전트 연구는 주로 도구 호출 성능을 높이기 위한 post-training에 집중했지만, 이를 감싸는 harness(관측·보조정보·출력 처리 래퍼)는 고정된 공학 디테일로 취급되는 경우가 많았습니다. 또한 기존 post-training 알고리즘은 배포 시나리오가 정적이라고 가정해 task 분포와 tool environment(도구 집합/호출 프로토콜)가 변하지 않는 설정에서 학습·평가되는 경향이 큽니다.

- **Core Contribution**: 이 논문은 ALFWorld를 확장해 harness를 “설계 변수”로 다루고, task shift와 tool environment shift까지 포함한 평가가 가능하도록 모듈화했습니다. 특히 harness의 정보량(h-low/h-mid/h-high)을 체계적으로 조절하고, post-training이 harness 설계에 어떻게 좌우되는지 in-distribution과 OOD에서 분석합니다.

- **Technical Challenges**: 핵심 기술적 도전은 harness가 제공하는 정보(예: 허용 가능한 도구 목록, 도구 설명의 풍부함)가 학습 신호와 크레딧 할당에 미치는 영향을 분리해 측정하는 것입니다. 저자들은 도구 스키마를 v1.0→v1.1→v2.0으로 단계적으로 바꿔 호출 포맷 자체가 달라지는 tool environment shift를 만들고, harness-aware post-training이 OOD에서도 적응하도록 GRPO와 GiGPO를 비교하는 방식으로 문제를 설계했습니다.

- **Empirical Impact**: 실험에서 harness의 정보량은 zero-shot부터 성능을 단조롭게 끌어올리며(모델 크기/능력과 함께 효과가 확대), post-training 후에도 그 경향이 유지됩니다. 또한 tool environment shift가 강해질수록 h-low 같은 low-effort harness로 post-trained 모델은 급격한 성능 저하를 보였고, 반대로 harness-aware post-training은 OOD 설정에서도 견고한 적응을 가능하게 했습니다. 더불어 harness를 post-training 이후에만 적용하는 post-hoc 방식은 학습 중부터 harness를 반영한 방식보다 성능이 떨어져, 언제 harness를 적용해야 하는지에 대한 실용적 인사이트를 제공합니다.



### Does Translation-Enhanced Speech Encoder Pre-training Affect Speech LLMs? (https://arxiv.org/abs/2606.25444)
Comments:
          Accepted to Interspeech2026

- **Prior Approaches**: Speech LLM은 pre-trained speech encoder와 LLM을 adaptor로 연결하는 구조가 표준이지만, encoder의 표현이 LLM의 언어-비의존(통합) 공간과 구조적으로 어긋난다는 문제가 제기돼 왔다. 기존에는 SSL이나 ASR 중심 사전학습 인코더를 그대로 쓰는 경우가 많았고, 특히 ASR 기반 표현은 언어별(음운·발음 유사성 중심)로 정렬돼 adaptor가 큰 브리지를 해야 한다.

- **Core Contribution**: 이 논문은 speech encoder 사전학습에 translation 목적을 넣는 것이, 언어별 표현을 넘어 의미를 언어-비의존적으로 추상화해 LLM과의 정렬을 만든다는 관점을 제시한다. 특히 영어-다른 언어 간 bidirectional English translation(영어↔X)으로 대칭 정렬을 강제하면, frozen LLM에서도 크로스모달 통합이 더 잘 된다고 실험으로 보인다.

- **Technical Challenges**: 핵심 기술 난제는 언어-별 음향/음운 구조에 묶여 있는 encoder 표현을 LLM의 의미적(semantic) 임베딩 공간에 맞게 정렬시키는 것이다. 이를 위해 Whisper와 유사한 Seq2Seq에서 decoder를 학습한 뒤 encoder만 꺼내 frozen LLM(두 종류 Llama 3.2 계열)과 adaptor만 학습하도록 통제된 실험을 설계했고, en→X와 X→en을 동시에 학습하는 multi-target 프롬프트로 target language 조건을 명시해 양방향 번역을 구현했다.

- **Empirical Impact**: 4개 언어(영어·일본어·중국어·독일) 기반 130k시간 사전학습과, 이후 adaptor-only 6.2k시간 다중 과제 학습에서 translation-enhanced pre-training이 ASR-only 대비 전반적으로 성능을 끌어올렸다. 특히 X↔en 대칭 번역 사전학습은 영어 입력에서 en→X ST 성능을 크게 개선했을 뿐 아니라, 사전학습에 없던 fa/id/sv/tr 같은 목표 언어로도 성능 향상이 확장됐다. 또한 intent classification은 translation에서 쓰인 언어일수록 개선되었고(예: 영어·독일), emotion recognition처럼 음향 단서가 중요한 과제는 크게 훼손되지 않아 의미 추상화와 음향 정보 보존이 함께 달성됨을 보여줬다.



### Evaluating Japanese Dialect Robustness Across Speech and Text-based Large Language Models (https://arxiv.org/abs/2606.25436)
Comments:
          Accepted to ASRU2025

- **Prior Approaches**: 기존 연구는 방언 정규화, 다중 방언 번역, 방언 말뭉치 구축 등 텍스트 중심 접근이 주를 이뤘습니다. 음성 영역에서는 방언 인지 ASR이나 방언별 적응, 음성 번역이 다뤄졌지만 ‘표준 대비 방언’ 변화에 대한 체계적 강건성 평가는 제한적이었습니다. LLM의 방언 성능 저하는 보고돼 왔으나, 그 능력이 LLM-기반 speech language model(SLM)로 그대로 전달되는지와 개선 경로는 명확히 정리되지 않았습니다.

- **Core Contribution**: 이 논문은 방언 입력에 대한 모델의 일반화 정도를 “강건성(robustness) = 방언 성능 / 표준 성능” 비율로 정의해, 기준 성능 차이를 통제한 공정한 비교 프레임을 제안합니다. 또한 일본 20개 방언을 대상으로 LLM과 SLM을 동일한 번역 과제로 평가해, 음성 처리 모델로의 전이(transfer) 여부를 정량적으로 분석합니다. 나아가 방언 데이터 학습과 speech encoder fine-tuning이 각각 어떤 역할을 하는지 분리해 규명합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 표준/방언 간 난이도 차이와 베이스라인 영향 때문에 단순 성능 격차만으로는 비교가 왜곡된다는 점, (2) 음성 입력에서는 발음·억양 같은 acoustic 차원이 이해를 흔들 수 있다는 점입니다. 이를 위해 연구진은 일본 방언→영어 번역을 사용하고 BLEU/BLEURT 같은 자동 평가를 쓰되, 강건성 비율로 정규화해 비교 가능성을 확보했습니다. SLM 구성은 Whisper-Large-V3 음성 인코더(encoder)와 LLM을 adapter로 연결하고, text 기반 번역 모델에는 LoRA로 출력 형식을 제어해 자동평가 오염을 줄였습니다.

- **Empirical Impact**: 실험 결과, SLM의 방언 강건성은 텍스트 기반 대응 모델과 높은 상관관계를 보이며(예: BLEU 기준 상관 0.848), 방언 이해 능력이 일부 전이됨을 확인했습니다. 다만 음성 입력에서는 전반적으로 성능 하락과 강건성 감소가 더 뚜렷해 acoustic 요인이 취약점임을 시사합니다. CPJD 같은 방언 데이터로 학습(adapter 학습)하면 대부분의 방언에서 강건성이 개선되고, speech encoder를 fine-tuning해도 추가 향상이 나타나 특히 초기 취약 방언에서 효과가 큽니다.



### Adaptive Oscillatory Inductive Bias for Modeling Sharp Prosodic Dynamics in Diffusion-Based TTS (https://arxiv.org/abs/2606.25424)
Comments:
          Accepted in INTERSPEECH 2026

- **Prior Approaches**: 확산 기반 TTS는 음향 표현을 점진적으로 정제해 고음질 합성을 이뤘지만, 감정 발화처럼 빠른 피치 변화와 급격한 운율 전이를 안정적으로 모델링하는 데는 여전히 한계가 있다. 기존 디코더는 Snake 같은 periodic nonlinearities로 조화(harmonic) 구조를 유도하지만, 고정된 주기/강도 파라미터는 급격한 진폭·주파수 변화를 충분히 적응적으로 다루기 어렵다.

- **Core Contribution**: 이 논문은 확산 기반 TTS 디코더에서 oscillatory inductive bias의 역할을 분석하고, 입력에 따라 주기적 변조를 제어하면서도 신호 안정성을 유지하는 adaptive oscillatory nonlinearity를 제안한다. 제안한 활성함수를 StyleTTS2 디코더에 통합해 OscillaTTS로 구현했으며, 급격한 운율 전이까지 더 잘 표현하도록 설계됐다.

- **Technical Challenges**: 핵심 과제는 (1) 조화 구조를 학습할 만큼 충분히 주기성을 주되, (2) 감정/전이 구간에서 발생하는 빠른 변화로 인해 그래디언트가 불안정해지지 않게 만드는 것이다. 저자들은 기존 고정 주기 periodic 활성의 적응성 부족을 보완하기 위해 learnable parameter α를 가진 주기 성분을 넣고, linear bypass 성분으로 안정성을 유지해 tanh 포화에 따른 게이팅으로 진동 반응을 조절하도록 구현했다.

- **Empirical Impact**: LJSpeech와 Emotional Speech Dataset(ESD: Happy/Angry/Sad)에서 MUSHRA 스타일 주관평가와 MCD, F0-RMSE 같은 객관평가, AutoPCP 및 WER로 일관된 개선이 관찰됐다. 특히 기준선 대비 스펙트로그램에서 조화 구조 정확도와 피치 궤적 안정성이 전이 구간에서 더 잘 유지되는 경향이 나타났고, ablation에서도 Oscilla(적응형)가 고정 파라미터/대체 활성보다 성능이 가장 좋았다.



### Sarashina2.2-TTS: Tackling Kanji Polyphony in Japanese Speech Generation via Data Scaling and Targeted Data Synthesis (https://arxiv.org/abs/2606.25369)
- **Prior Approaches**: 기존 LLM-TTS는 주로 영어·중국 중심으로 성능이 높게 보고됐지만, 일본은 kanji polyphony 같은 언어 고유의 문맥 의존 읽기 문제가 제대로 최적화되지 않았다. 또한 데이터는 일본 비중이 작고, kanji 읽기 커버리지를 겨냥한 공학적 전략(드문 읽기 보강 등)이 부족한 편이다. 평가도 문장 단위 CER/WER에 의존해 어떤 kanji·어떤 reading에서 오류가 났는지 진단이 어렵고, 일본의 문자 표기 변이 때문에 지표가 흔들린다.

- **Core Contribution**: Sarashina2.2-TTS는 일본에 특화된 LLM-TTS로, (1) 데이터 전략과 (2) 평가 방법론을 함께 설계해 kanji polyphony disambiguation을 정면으로 다룬다. 먼저 일본 화자 발화 194k시간을 포함해 약 361k시간 규모로 학습하고, PronSteering 기반의 targeted data augmentation으로 2,136 Joyo kanji의 long-tail reading까지 커버한다. 더불어 Joyo Kanji Yomi Benchmark와 Kana-CER를 제안해 kanji-level 정확도를 정밀하게 측정한다.

- **Technical Challenges**: 핵심 기술적 난제는 kanji가 문맥에 따라 여러 reading을 갖는 구조적 모호성을 모델이 읽기 선택 단계에서 안정적으로 해결해야 한다는 점이다. 이를 위해 backbone LLM은 텍스트-음성 의미 토큰 생성 역할에 집중하도록 end-to-end로 학습하고, 음소가 반영된 speech tokenizer(S3Tokenizer V2)와 2단계 학습(대규모 사전학습+합성 데이터로 미세조정)을 통해 장기 꼬리 읽기의 빈도 부족을 보완한다. 평가 또한 orthographic variation을 제거해야 했기 때문에, Kana-ASR로 생성 음성을 kana 공간에서 reference reading과 비교하는 Kana-CER로 오차의 원인을 pronunciation에 더 가깝게 귀속시킨다.

- **Empirical Impact**: 실험에서 Sarashina2.2-TTS는 Joyo Kanji Yomi Benchmark에서 모든 CER 기반 지표가 기존 일본 지원 LLM-TTS 대비 우수했으며, 특히 kanji-level reading 정확도에서 큰 격차를 보였다. 합성 데이터 증강(Stage 2)이 Stage 1 대비 전 지표를 개선해 targeted augmentation의 효과를 확인했다. 또한 cross-lingual 평가에서 프롬프트 언어가 일본어가 아니어도 일본 발음이 안정적으로 유지되는 유일한 시스템으로 보고돼, balanced training이 cross-lingual robustness를 준다는 점을 실증했다.



### Data-Driven Evolution of Library and Information Science Research Methods (1990-2022): A Perspective Based on Fine-grained Method Entities (https://arxiv.org/abs/2606.25320)
- **Prior Approaches**: 1990년대 이후 빅데이터와 정보기술의 확산으로 LIS(도서관·정보학) 연구가 데이터 중심으로 이동했지만, 방법론 진화가 어떤 구성요소에 의해 촉진되는지는 정밀하게 정리되지 못했다. 기존 연구는 전체 경향을 요약하는 수준에 그치거나, 방법을 구성하는 알고리즘·데이터·도구·평가지표를 함께 추적하기 어려웠다.

- **Core Contribution**: 이 논문은 1990~2022년 LIS 학술논문을 대상으로 데이터 기반 방법의 핵심 구성요소를 자동 추출해, 연구 방법의 세부 진화 경로를 세 차원(시기별 특성, 주제별 변화, 방법론별 교차 양상)으로 분석한다. 특히 알고리즘·모델, 데이터 자원, 소프트웨어·도구, 메트릭(평가지표)이라는 네 범주의 ‘방법 엔티티’를 분류해 방법론 변화를 설명 가능한 형태로 제시한다.

- **Technical Challenges**: 핵심 과제는 방대한 논문에서 방법 관련 실체를 신뢰성 있게 자동 추출하고, 시간·주제·연구방법이라는 여러 축에서 일관된 비교가 가능하도록 하는 것이다. 저자들은 네 범주의 엔티티를 추출한 뒤, 엔티티의 시간적 변화와 주제 간 이동, 연구방법 간 상호 진화 패턴을 함께 관찰하는 분석 프레임을 구성해 이를 해결한다.

- **Empirical Impact**: 결과적으로 LIS에서 방법론 진화의 ‘주요 동력’은 데이터 자원(data resources)인 것으로 드러나며, 연구 방법 전개가 ‘등장→안정화/실무 적용’의 순환 패턴을 보인다고 보고한다. 이는 LIS 연구를 단순한 데이터 활용이 아닌, 데이터 자원의 성숙과 활용 주기에 맞춰 방법론이 재편된다는 관점을 제공한다는 점에서 의미가 있다.



### Measuring Research Difficulty of Academic Papers: A Case Study in Natural Language Processing (https://arxiv.org/abs/2606.25307)
- **Prior Approaches**: 기존 연구들은 연구 주제 선택이나 성과 분석에서 연구 난이도를 다루더라도, 이를 정량적으로 평가하는 프레임이 부족했다. 또한 연구 난이도와 학술적 영향(impact)의 상관을 수치로 연결한 접근이 제한적이었다.

- **Core Contribution**: 이 논문은 논문 단위로 연구 난이도를 종합 평가하는 시스템을 제안한다. 학술 협업, 내용, 참고문헌 등 내·외부 특징을 추출해 여러 난이도 지표를 만들고, entropy weight로 가중치를 산정해 연구 난이도 점수를 산출한다.

- **Technical Challenges**: 핵심은 난이도를 구성하는 요소들을 어떻게 측정하고 결합할지, 그리고 가중치를 어떻게 정할지였다. 논문은 엔트로피 가중치로 지표 간 중요도를 데이터 기반으로 조정하고, NLP 분야 논문들을 대상으로 난이도 점수의 신뢰성을 확보하기 위해 전문가 평가와 상관 분석을 수행했다.

- **Empirical Impact**: 검증 결과 citation frequency를 학술적 영향으로 두었을 때, 페이지 수, 참고문헌 수, 상위 기관의 참여 같은 요인이 영향과 유의미하게 연관됐다. 또한 연구 난이도와 영향 사이에 역U자(inverted U-shaped) 관계가 관찰되어, 중간 수준의 난이도가 더 큰 impact로 이어질 가능성을 시사한다.



### Multilingual Hematology Visual Question Answering Datas (https://arxiv.org/abs/2606.25246)
Comments:
          Under Review

- **Prior Approaches**: 기존 VLM 기반 의료 비전-언어 연구는 시각과 텍스트를 함께 다루지만, 대개 영어 중심 자원에 의존해 다국어 의료 현장 적용성이 낮았다. 또한 일부 데이터셋은 형태학 정보를 학습하더라도 캡션/자율 생성이나 언어 감독이 약해 임상 검증된 VQA 형태의 다국어 벤치마크는 부족했다.

- **Core Contribution**: 본 논문은 백혈병(Leukemia)과 정상 백혈구(WBC) 형태를 대상으로 한 임상 검증 bilingual VQA 벤치마크 WBCMor VQA를 제안한다. LeukemiaAttri와 WBCAtt의 morphology-aware 어노테이션을 바탕으로 영어-우르두(Urdu) 쌍을 구축해, 다국어 의료 AI 개발과 평가를 가능하게 한다.

- **Technical Challenges**: 핵심 기술 과제는 우르두(Urdu) 형태학 용어의 번역 품질과 임상적 일관성을 확보하는 것이다. 영어 VQA를 NLLB-200으로 번역한 뒤, 도메인별 Urdu hematology dictionary로 5,000건 이상 용어를 교정하고 prompt-guided re-translation으로 표준화를 강화했으며, 생성/번역 전후 단계에서 의료 전문가 검증을 수행했다.

- **Empirical Impact**: 평가는 Qwen2-VL-2B, InternVL2.5-2B 등 오픈소스 VLM을 영어/우르두/혼합 세팅에서 zero-shot과 fine-tuning으로 비교해, 도메인 특화 fine-tuning이 일관되게 성능을 끌어올림을 보였다. 110K bilingual question-answer pairs(20K 단일세포 이미지)를 공개해 향후 다국어 임상 VQA 및 언어-적응형 의료 AI 연구의 기반을 제공한다.



### ASAP: Agent-System Co-Design for Wall-Clock-Centered Auto HPO Research for ML Experiments (https://arxiv.org/abs/2606.25207)
- **Prior Approaches**: 기존 HPO는 서러게이트 모델(예: Gaussian process, TPE, SMAC 등)과 획득함수를 통해 다음 하이퍼파라미터를 순차적으로 고르는 방식이 주류입니다. 최근에는 LLM을 surrogate/agent로 써서 iteration별 성능을 높이려는 시도가 있으나, LLM도 사전학습 기반의 단일 inductive bias를 그대로 가져 “대체(replacement)” 형태로 동작하는 한계가 남습니다. 또한 대부분의 평가는 iteration count에 치우쳐 있어, 실제 실행에서는 LLM 추론과 도구 실행 오버헤드가 직렬로 누적되며 wall-clock 이득이 줄어드는 문제를 충분히 다루지 못합니다.

- **Core Contribution**: 이 논문은 ASAP(Agent-System co-design for Wall-clock)를 제안하며, LLM 기반 HPO를 “대체”가 아니라 “통합”으로 전환합니다. ASAP는 여러 HPO 도구(서러게이트 기반 통계 도구 + LLM-as-proposer)를 한 에이전트(LLM-as-Judge)가 매 라운드 후보 중에서 조합 선택하도록 설계해, 특정 사전의 편향에 취약했던 구조를 완화합니다. 동시에 설계 목표를 iteration count가 아니라 end-to-end wall-clock으로 바꾸고, 에이전트(프롬프트/판단)와 시스템(실행/스케줄링)을 함께 최적화합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 서로 다른 inductive bias를 가진 제안들을 한 라운드에서 “선택 가능한” 형태로 결합하고, (2) LLM·도구 비용을 직렬로 다루면 생기는 지연을 줄이면서도 선택 품질(레그렛)을 유지하는 것입니다. ASAP는 매 라운드 후보 집합을 도구 풀의 union으로 만들고, LLM-as-Judge가 in-context learning 형태로 후보를 평가해 선택하게 하여 통합의 실효성을 확보합니다. 여기에 KV-cache를 재사용하는 prefix-stable 프롬프트, 모델 평가와 LLM/도구 지연을 겹치게 하는 speculation parallelism, 그리고 critical path 밖에서 speculation threshold를 튜닝하는 Self-Tuner로 wall-clock을 절감합니다.

- **Empirical Impact**: 다양한 현대 HPO 벤치마크에서 ASAP는 여러 기준선 대비 일관되게 더 좋은 성능/효율을 보이며, 특히 단일-bias 도구들이 약한 지형(거칠고 다중 양상이며 속이기 쉬운 경우, 이방성 등)에서도 견고함을 보여줍니다. 분석 결과는 per-task가 아니라 다양한 작업군에서 레그렛이 wall-clock 단위로 개선되는 방향이 반복적으로 관찰됩니다. 종합하면 ASAP는 LLM-HPO의 “반복 횟수 향상” 중심 평가를 넘어, 실제 서비스 실행 비용까지 고려한 에이전트-시스템 공동 설계의 실용성을 확립한 것으로 의미가 큽니다.



### RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory (https://arxiv.org/abs/2606.25206)
Comments:
          Project website: this https URL

- **Prior Approaches**: 기존 로봇 장기 배치 메모리는 라벨이 고정된 의미 맵(카테고리)이나 3D 포인트클라우드 같은 지표 기반 표현에 의존해, 질의가 특정 세부를 요구할 때(예: 작은 색/모양 디테일) OOV(미등록) 문제가 쉽게 발생한다. 또 다른 접근은 오픈보캐뷸러리를 위해 관측을 caption으로 바꾸고 텍스트 임베딩을 저장하는데, 이 image-to-text 캡션 병목이 시각 디테일을 손실시켜 검색이 취약해진다.

- **Core Contribution**: RAVEN은 캡션 대신 raw visual embeddings(시각 임베딩)를 그대로 저장하는 agentic memory 시스템으로, 장기 로봇 질의응답과 내비게이션을 지원한다. 관측 프레임마다 시각 임베딩에 pose(자세)와 time(시간)을 함께 묶은 visuo-spatio-temporal memory triplet을 vector database에 넣고, 공간 지도 기반으로 검색을 grounding해 답변과 목표 이동을 동시에 수행한다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트로 압축하지 않으면서도 fine-grained 의미를 유지하고, (2) 수천 프레임 단위의 긴 기간에서도 검색 비용을 억제하며, (3) 검색 결과를 로봇 플래닝에 연결해 정확한 목표 좌표를 뽑는 것이다. RAVEN은 FAISS/Milvus 같은 벡터 검색으로 top-K 검색을 sub-O(N) 수준에 가깝게 처리하고, VLM이 도구(텍스트/시간/위치/이미지 기반 검색)를 유한상태머신 루프에서 반복 호출해 working memory를 줄이면서 정밀도를 높인다.

- **Empirical Impact**: 여러 시뮬레이션 및 실제 비디오 QA 벤치마크에서 RAVEN은 caption 기반 메모리 시스템을 일관되게 능가하며, frontier VLM과 비교해 장기 과제 성능을 유지하면서 검색 비용은 10배 낮췄다고 보고한다. 또한 Unitree Go1을 실제 환경에 배치해 자연어 목표 도달 내비게이션을 성공적으로 시연했고, 0.12 fps 다운샘플링에서도 성능의 대부분을 보존하며 원시 RGB 대비 250배 이상 저장 압축과 높은 메모리 효율을 보였다.



### To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient Multi-Agent RAG (https://arxiv.org/abs/2606.25191)
Comments:
          23 pages, 2 figures, 19 tables. Code: this https URL

- **Prior Approaches**: 기존 멀티에이전트 문서 평가는 retrieval-augmented generation(RAG)을 개선하기 위해 문서를 평가·필터링·토론하는 방식으로 확산됐지만, 에이전트 수·문서 수·반복 라운드가 곱해지며 추론 비용이 급격히 증가한다. 또한 작은 배포형 모델(7B~9B)로 옮겨도 “평가(assessment)가 왜/언제 이득을 주는지”에 대한 메커니즘 이해가 부족하다는 문제가 제기된다. 그 결과 동일한 파이프라인이 모델마다 정확도에 큰 폭으로 개선 또는 악화를 유발하는 불일치가 관측된다.

- **Core Contribution**: 이 논문은 학습 없이(training-free) 개입(프롬프트/집계/생성 전략)만으로 7B~9B 지시 튜닝 모델이 멀티 문서 평가에서 이득을 얻는 방식을 통제 실험으로 분해한다. 핵심 주장(진단→치료)은 “모델의 내재적 역량-태스크 구조 상호작용”이 평가 가치(이득)를 가른다는 것으로, 진단 결과에 따라 PDE(Per-Document Extraction), SDA(Score Distribution Alignment), CoT de-polarization, ATF(Adaptive Threshold Filtering) 중 최적 처방을 라우팅한다. 이를 위해 모델 적응형 라우팅 아키텍처 MADARA를 제안하며, 진단 임계값이 단일 파일럿 모델에서 유래해도 4개 미보는 모델 패밀리에 zero-shot으로 일반화됨을 보여준다.

- **Technical Challenges**: 메커니즘을 규명하려면 ‘평가 점수 자체의 품질’과 ‘멀티 문서 컨텍스트 혼란 해결’ 중 무엇이 성능을 좌우하는지 레이블 없이 판별해야 한다. 약한 모델에서는 문서 단위 격리가 지배적이지만, 강한 모델에서는 점수의 품질이 중요해져 다른 처방이 필요하므로 둘을 구분할 진단 설계가 관건이다. 이 논문은 Reasoning-Score Coupling(RSC)라는 label-free perturbation probe로, 추론 단계를 체계적으로 교란했을 때 문서 점수가 단조롭게 악화되는지(추론-점수 결합 강도)를 Spearman 상관 기반의 trend coefficient로 판별하고, 그 결과로 MADARA의 라우팅(예: 약한 모델이면 PDE, 강한 모델이면 CoT de-polarization/ATF/SDA)을 자동 선택한다.

- **Empirical Impact**: 실험 결과 PDE는 약한 베이스라인에서만 압도적으로 효과적이며, adversarial conflicts 및 표준 QA에서 최대 50점대 정확도 향상처럼 큰 폭의 이득이 관측된다. 특히 PDE-Random(문서 선택을 무작위화해도 멀티에이전트 평가를 사실상 우회)만으로도 약한 모델의 PDE 성능이 거의 동일해, 이득의 실체가 ‘점수 품질’이 아니라 ‘멀티 문서 컨텍스트 혼란 제거’임을 강하게 뒷받침한다. 반대로 강한 모델에서는 isolation만으로는 이득이 없고, RSC로 점수 성향을 진단해 SDA/CoT de-polarization/ATF 중 적절한 처방을 선택할 때 성능이 유지·향상되며, MADARA는 단일 파일럿 기반 임계값으로 zero-shot 라우팅이 재현됨을 입증해 멀티에이전트 추론 비용을 약 4배 이상 줄일 수 있는 실용적 의미가 있다.



### LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search (https://arxiv.org/abs/2606.25039)
- **Prior Approaches**: 기존 ODE 방정식 복원은 주어진 데이터셋에서 식을 “정적 추론”처럼 찾아내는 방식이 주를 이루며, 관측이 충분히 정보적이라는 가정을 탑니다. 이 때문에 상태공간이 크거나 관측이 제한적이면 구조적으로 다른 후보들이 같은 데이터에서는 비슷한 오차를 내는 identifiability gap 문제가 생겨, 그럴듯하지만 잘못된 governing equation을 복원할 수 있습니다. LLM-유도 접근은 operator priors 같은 지식을 활용해 탐색을 돕지만 대개 데이터 수집과 결합된 closed-loop로 진짜 구분 가능한 관측을 모으는 데는 한계가 있습니다.

- **Core Contribution**: 논문은 LLM-ACES(LLM-guided Active Closed-loop Equation Search)라는 closed-loop 프레임워크로, symbolic hypothesis 구성과 adaptive data acquisition을 동시에 최적화합니다. LLM이 operator 수준의 priors로 큰 탐색공간을 분할하고, 그 안에서 후보 방정식을 학습한 뒤 후보들 간 예측 불일치가 큰 초기조건을 추가로 쿼리해 identifiability gap을 직접 해소합니다. 즉, 식 복원을 “피팅 결과”가 아니라 “가설이 다음 관측을 설계”하는 동적 과정으로 재정의합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 방대한 symbolic hypothesis 공간을 LLM이 전부 직접 내놓는 방식으로는 제어하기 어렵다는 점과 (2) 제한된 샘플 예산 안에서 어떤 초기조건이 경쟁 가설을 실제로 구분하는지 판단해야 한다는 점입니다. LLM-ACES는 operator prior로 상위 가설공간을 구조적으로 제약한 뒤, 후보들의 rollout을 비교하는 predictive-divergence(평균 pairwise predictive disagreement, NMSE 기반)를 acquisition score로 삼아 가장 구분력 있는 궤적을 선택합니다. 또한 각 라운드마다 검증 데이터로 후보를 재스코어하고 experience buffer에 고득점·저득점 가설을 함께 저장해, 처음 데이터에만 맞는 spurious term 유입을 억제하는 피드백을 구현합니다.

- **Empirical Impact**: ODEBench(63개)와 ODEBase(59개)에서 LLM-ACES는 전체적으로 NMSE와 symbolic accuracy가 동시에 가장 좋았고, 특히 NMSE 중앙값이 기존 SOTA 대비 여러 자릿수 이상 개선되는 결과를 보였습니다. symbolic accuracy는 ODEBench에서 46.2%(GPT-4o-mini)·52.4%(Qwen3-32B 계열 결과는 ODEBase에서 특히 높음) 수준을 달성하며, noise에도 견고하고 올바른 symbolic 구조를 복원하는 경향이 분석으로 확인됐습니다. 더 나아가 sample-efficient하게도 1/10 수준의 데이터로 더 나은 성능을 내며, “닫힌 루프” 기반 adaptive data acquisition이 정답 구조를 가리는 식의 국소 적합 문제를 완화한다는 점에서 의미가 큽니다.



### Do Thinking Tokens Help with Safety? (https://arxiv.org/abs/2606.25013)
- **Prior Approaches**: 추론(reasoning) 모델은 thinking tokens를 사용해 벤치마크 성능을 높이는 경우가 많고, 이 방식이 더 ‘숙고적(deliberative)’이라 안전·정렬(alignment)에도 유리하다고 여겨져 왔다. 구체적으로는 모델이 요청에 대한 답을 내기 전에 안전 원칙 위반 여부를 고려하는 ‘안전한 공간’을 제공한다고 가정해 왔다.

- **Core Contribution**: 본 논문은 GPT-OSS, Qwen, Olmo, Phi 계열의 frontier open-weight reasoning 모델들을 대상으로, 최종 거절/순응(refusal/compliance) 결과가 첫 토큰의 hidden representation에서 이미 강하게 예측된다는 증거를 제시한다. 또한 thinking 텍스트가 있어도 실제로는 숙고적 수정이라기보다 prefix completion에 가깝고, 초반(전체 thinking의 약 20% 내) 이후 결과가 거의 바뀌지 않는다고 주장한다.

- **Technical Challenges**: 핵심 과제는 ‘보이는 숙고’와 ‘실제 의사결정 과정’을 분리해 측정하는 것이다. 논문은 첫 토큰 hidden representation에 대해 학습된 head로 거절/순응을 조기 예측하고, thinking 진행에 따른 outcome 변화 지점을 관찰하는 방식으로 deliberation의 실재성을 검증했으며, 기존 안전 개입들이 오히려 과도한 거절 쪽으로 행동을 치우치고 이미 희소한 deliberation 신호를 억제한다고 분석한다.

- **Empirical Impact**: 첫 토큰 기반 조기 예측은 AUROC 0.84~0.95, balanced accuracy 약 88%로 높은 성능을 보였고, 텍스트 수준에서 숙고처럼 보이는 현상(약 74%)이 있어도 분포가 한쪽으로 이미 잠긴 뒤 발생하는 비중이 큼을 보여준다. 더 나아가 기존 inference-time 및 training-based safety intervention이 ‘숙고 유도’를 목표로 했음에도 over-refusal을 강화하며 실제 deliberation을 줄일 수 있음을 실증적으로 제시해, 향후 ‘진짜 안전 숙고’를 유도하는 방법의 필요성을 강하게 시사한다.



### Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns (https://arxiv.org/abs/2606.25010)
Comments:
          18 pages, 13 figures

- **Prior Approaches**: 기존 scaling law는 사전학습 손실이 모델 크기에 따라 매끄럽게 좋아진다고 예측하지만, in-context learning 같은 다운스트림 능력은 특정 스케일 이후 갑자기 나타나는 현상이 보고돼 왔다. 또한 같은 스케일에서도 초기 seed에 따라 성패가 갈리며, 능력의 출현 시점이 짧은 훈련 구간에서 불연속적으로 전개될 수 있다는 점이 논의돼 왔다.

- **Core Contribution**: 이 논문은 언어모델에서 나타나는 emergent capability가 훈련 중 무작위로(stochastically) 발생하되, 큰 모델일수록 더 이른 시점에 더 안정적으로 획득된다고 주장한다. 그 메커니즘으로, 능력의 출현이 태스크에 필요한 attention pattern의 “급격한 학습”과 함께 일어난다는 점을 제시한다.

- **Technical Challenges**: 자연어에는 단일 정답 attention pattern이 없어 context length나 sparsity 같은 요인이 학습 난이도에 미치는 영향을 분리하기 어렵다. 이를 해결하기 위해 linear map과 cellular automata 같은 합성 데이터로 transformer를 학습시키고, attention intervention(정답에 가까운 attention 로그잇을 편향)과 causal attention head patching을 통해 급격한 성능 점프가 sparse·장문맥 attention 패턴 학습 병목에서 온다는 것을 확인한다.

- **Empirical Impact**: Pythia 체크포인트 분석에서, seed에 따라 능력 출현 확률과 출현 시점이 달라지며 출현 순간에 다음 토큰 확률이 abrupt하게 상승하는 양상이 관찰됐다. 합성 실험에서는 attention sparsity와 context 길이가 학습 가능/불가능을 가를 정도로 난이도를 좌우했고, attention head 수를 늘리면 학습 효율이 개선되지만 head dimension은 최소 용량 이후 효율이 감소했다. 더 나아가 MLP-Mixer는 linear map의 복잡한 attention 패턴 학습에서는 transformer를 크게 앞섰으며, 이는 sparse attention 패턴 학습을 더 효율적으로 만드는 아키텍처 설계 가능성을 시사한다.



### Neural Scaling Universality: If Exponents Are Fixed, Time to Understand Coefficients (https://arxiv.org/abs/2606.25008)
Comments:
          17 pages, 6 figures

- **Prior Approaches**: 기존 신경 스케일링 법칙은 학습 손실이 모델 크기·데이터·연산량에 대해 멱법칙으로 감소한다는 경험칙을 정리하며, 주류 관점은 데이터의 주파수 분포나 중요도 같은 “데이터 멱법칙 구조”가 손실 멱법칙 지수를 만든다고 봅니다. 그러나 이 접근은 지수가 데이터의 세부 통계에 민감해져 일반화나 예측이 어려울 수 있다는 한계를 함께 안고 있습니다. 최근에는 데이터에 멱법칙이 없어도 손실 멱법칙이 “기제”로부터 나타날 수 있다는 대안 관점도 등장했습니다.

- **Core Contribution**: 이 논문은 신경 스케일링 universality(신경 스케일링 보편성)라는 가설을 제시하며, 멱법칙의 지수(exponent)는 Softmax 비선형성·superposition·Transformer 층의 앙상블 오차 평균 같은 범용 메커니즘이 고정한다고 주장합니다. 즉, 서로 다른 아키텍처나 데이터 세부는 계수(coefficient)를 바꾸지만 지수는 바꾸지 않는 “보편성 계급(universality class)”로 모델을 분류합니다. 또한 계수가 최적 모델 형상과 compute-optimal frontier를 직접 좌우하므로, 지수보다 계수 이해가 실전 성능 개선의 핵심이라고 강조합니다.

- **Technical Challenges**: 기여의 핵심은 시간(time)·폭(width)·깊이(depth) 각각의 스케일링이 왜 고정 지수를 갖는지 메커니즘을 분리해 설명하고, 큰 모델에서 총 손실이 각 항의 합 형태로 근사된다는 점을 보이는 데 있습니다. 저자들은 Softmax의 강한 비선형(저엔트로피/첨두 분포)으로 1/3 시간 스케일링, 표현 차원 대비 과밀 특징(superposition)로 m^{-1} 폭 스케일링, 층별 오차가 앙상블처럼 평균화된다는 관측으로 ℓ^{-1} 깊이 스케일링을 도출하고 계수 cτ, cm, cℓ가 데이터·아키텍처 세부를 흡수한다고 정리합니다. 이어서 Pythia, Chinchilla, Farseer 계열에서 지수는 고정하고 계수만 피팅하는 방식으로 이 분해가 경험적으로도 성립함을 검증합니다.

- **Empirical Impact**: 저자들은 Pythia에서 학습 구간 손실이 one-third time scaling에 잘 맞고, 포화(saturation) 손실은 inverse width와 inverse depth로 설명된다고 보고합니다. Chinchilla와 Farseer에서는 최적 aspect ratio 및 토큰-파라미터 비율이 계수 기반 예측(모델 크기 1/3 스케일, compute에서 손실 1/6 스케일)을 따른다고 보여 compute-optimal frontier까지 실증합니다. 결론적으로 “현재 LLM이 속한 보편성 계급”에서 계수를 체계적으로 측정·조절하는 연구가 단기 성능·연산 효율 개선의 실질 경로가 될 수 있음을 시사합니다.



### Learning Diachronic Representations of Ancient Greek Letterforms (https://arxiv.org/abs/2606.24984)
Comments:
          Accepted for publication at the International Conference on Document Analysis and Recognition (ICDAR) 2026

- **Prior Approaches**: 기존 OCR·문서 인식 연구는 대체로 문자 형태가 안정적이고 충분한 학습 데이터가 있다는 가정을 둔다. 이 때문에 수세기에 걸친 서체 진화(구조적 드리프트)와 자료 열화, 불균형·소량 데이터 환경에서는 일반적인 transfer나 contrastive learning이 문자 간 ‘본질적 유사성’을 적절히 반영하지 못한다. 특히 supervised contrastive learning 계열은 서로 다른 클래스라도 형태가 닮은 경우를 동일한 음성(negative)으로 강하게 밀어내는 문제가 있어, 자연스러운 inter-class 관계를 학습 임베딩에 반영하기 어렵다.

- **Core Contribution**: 논문은 고대 그리스 필기에서 수백 년 단위로 바뀌는 문자 형태에 강건한 표현을 학습하기 위해, 두 가지 도메인 지식을 학습 목표에 결합한다. 핵심은 similarity-weighted supervised contrastive loss로 형태적으로 헷갈리는 문자쌍은 음성 repulsion을 약화시키고, lacuna-driven augmentation으로 실제 파피루스 손상 양상을 닮은 결손을 학습에 포함시키는 것이다. 또한 3rd BCE~1st BCE 학습용 Hell-Char와 2nd~5th CE 평가 PaLit-Char, 9th~14th CE 평가 Med-Char의 3종 벤치마크를 제공해 시간적 일반화 성능을 체계적으로 측정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 문자 간 상징적 변이와 (2) 결손·잡음으로 인한 체계적 열화, (3) 클래스별 소량·불균형 데이터다. 저자들은 동적으로 추정한 inter-class similarity에 따라 negative pair 가중치를 조절하는 DSCL로 ‘시각적으로 유사하지만 다른 문자’의 과도한 분리를 막고, lacuna-driven augmentation(LF)로 결손이 만드는 비직사각형 패턴을 생성해 부분 가림에도 임베딩이 유지되도록 설계했다. 더불어 대표 프로토타입(메도이드) 기반 시각화로 클러스터 구조가 해석 가능한 방식으로 형성되는지 확인한다.

- **Empirical Impact**: 실험에서 LF+DSCL을 적용한 lightweight CNN과(pretrained) ResNet18이 PCA나 generic pretraining 대비 더 응집적·분리도가 높은 임베딩을 만든다. Hell-Char에서 인식 성능이 상승할 뿐 아니라, 임베딩 클러스터링으로 문자별 하위 서브그룹(스타일 변종)과 시대별 과도기 글자 형태(prototype)를 더 명확히 드러냈다. 시간 일반화 측면에서는 PaLit-Char(2nd~5th CE) 정확도와 F1이 0.84로 학습 구간과 가까운 전이에서도 강건함을 보였고, Med-Char(9th~14th CE)에서는 일부 문자가 완전히 붕괴(F1=0)하는 등 오류가 시간적으로 구조화됨을 확인해, 표현이 실제 역사적 형태 변화와 정합적으로 작동함을 시사한다.



### Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieva (https://arxiv.org/abs/2606.24976)
- **Prior Approaches**: 기존 Multi-Agent Debate(MAD)는 여러 에이전트가 서로 검증하며 장기 추론의 오류 누적을 줄이는 데 초점을 맞춰 왔지만, 설득처럼 주관적 과제에서는 문제 드리프트와 sycophantic conformity가 크게 악화됩니다. 이를 보완하려고 Retrieval-Augmented Generation(RAG)을 결합하는 경우가 많았지만, 표준 RAG는 논리 구조보다 주제 키워드(어휘 겹침) 기반으로 문서를 고르면서 semantic leakage를 유발해 반대로 실패를 증폭시킨다는 점이 드러났습니다.

- **Core Contribution**: 이 논문은 semantic leakage를 MAD 실패의 재현 가능한 트리거로 규정하고, 이를 제거하기 위한 Taxonomic Strategy RAG(TS-RAG)를 제안합니다. 핵심 아이디어는 전략을 ‘이산 범주 병목(categorical bottleneck)’을 통해 구조만 남긴 채 전달해, 토픽(수사/명사)과 논리적 역할을 분리함으로써 zero-shot 도메인 간 추론을 가능하게 하는 것입니다.

- **Technical Challenges**: 문제는 주어진 대화에서 어떤 ‘논리적 취약점(결함)’을 공략해야 하는지 실시간으로 추적해야 한다는 점입니다. 이를 위해 turn-by-turn Debate State Representation(DSR)로 committed claims와 active trigger를 분리하고, 추출된 결함을 논리 취약점 택소노미 확률벡터로 강제 매핑한 뒤, 구조 유사도 기반으로 exploitation blueprint를 검색·삽입하는 방식으로 semantic 노이즈를 차단합니다.

- **Empirical Impact**: 실험에서 TS-RAG는 표준 semantic RAG가 붕괴하는 교차 도메인 설득 설정에서 추상 논리 전이 성능을 유의미하게 개선했으며, 경량 persuader가 강한 상대를 상대로 win rate를 70.5%에서 78.5%로 끌어올리는 ‘capability bridge’ 효과를 보였습니다. 또한 DSR 기반 단계별 진단과 trace-level 분석을 통해, 기본 LLM의 기본 순응(sycophancy)으로 인한 평가 붕괴를 막기 위한 엄격한 제약의 필요성을 검증하며, 합의까지의 라운드도 단축되는 경향을 확인했습니다.



### Why Do Accumulated Transformations Extrapolate? (https://arxiv.org/abs/2606.24975)
Comments:
          33 pages, submitted to TMLR

- **Prior Approaches**: PaTH Attention은 RoPE의 위치 인덱스 회전을 Householder 반사의 누적 곱으로 바꿔 긴 문맥에서도 길이 외삽이 잘 되는 현상을 보였지만, 왜 그런지 메커니즘은 완전히 정리되지 않았다. 또한 NoPE 같은 길이-안정(평탄) 설계는 가능성을 보여주지만, 멀리 있는 토큰이 ‘얼마나’ 억제되는지, 값(value)까지 남는 잔여 간섭이 어떻게 누적되는지는 충분히 설명하지 못한다.

- **Core Contribution**: 이 논문은 Householder 구조가 핵심인지, 더 일반적인 ‘소스-쿼리 경로(path) 위의 누적 변환’ 성질에서 비롯되는지 묻는다. 결론적으로 누적 직교 변환이 특정 정칙성(regularity) 조건을 만족하면, 누적 곱이 유한 단계 후 서로 비상관(incoherent) 상태가 되어 원거리 토큰에 대한 어텐션이 억제되며, 다만 컨텍스트가 무한히 커지면 결국 성능이 저하되는 하한도 함께 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 누적 변환이 ‘콘텐츠-의존적 mixing window’를 어떻게 만들고, (2) 그 결과 score gap이 실제로 far attention mass를 줄이는지, (3) 그럼에도 far set이 커질 때 near 신호가 영원히 보존될 수 있는지(불가능하다는 하한)를 동시에 정량화하는 것이다. 논문은 스펙트럴 갭을 이용한 누적 곱의 기하급수적 탈상관, 고차원 농도(concentration)로 softmax 상에서의 score gap 형성, 그리고 far mass가 커질수록 근본적으로 near-signal이 붕괴한다는 lower bound를 증명하며, SO(2)^{SO(2)} 특수 케이스에서는 값까지 같은 경로로 회전해 잔여 far 기여를 비동조(incoherent)로 결합시키는 추가 보장도 논증한다.

- **Empirical Impact**: 통제된 실험에서 무작위 누적 회전은 RoPE 대비 외삽을 크게 개선했으며, 학습된 token-dependent 회전은 훈련 컨텍스트를 훨씬 넘어 perplexity가 유지되는 패턴을 보였다(예: 훈련 길이 대비 16배 수준 평가). 또한 rotation-only 모델은 극단적 길이에서 점진적 열화를 보이지만 ALiBi는 상대적으로 길이 안정적이어서, 논문의 ‘far-mass 제어가 구조적으로 필요하다’는 주장과 일치한다. 값(value)까지 회전하는 설정은 query/key만 회전하는 경우보다 긴 문맥 열화를 더 완화해, 이론의 추가 보호 메커니즘을 지지한다.



### Digital Twin-Driven Adaptive Sim-to-Real Alignment via Reinforcement Learning for Vibration-Based Bearing Health Monitoring Under Data Scarcity (https://arxiv.org/abs/2606.24954)
- **Prior Approaches**: 회전 기계 진동 기반 건전성 모니터링에서는 소스(시뮬레이터)와 타깃(실운용) 간 sim-to-real 간극과 희소한 고장 이벤트 때문에 진단 성능이 흔들린다. 기존 domain adaptation은 주로 class-agnostic 전역 변환이나 소스-타깃 균일 혼합을 쓰는데, 이는 고장 유형별로 다른 주기성·진폭 변조·스펙트럼 특성을 동시에 맞추지 못해 클래스 간 분리도를 망가뜨리거나 Normal 클래스에 분포 잡음을 유입시키는 한계가 있다. 또 순차적이고 상태 의존적인 정렬 문제를 한 번의 최적화로 취급해, 정렬 과정에서 생기는 상태 의존성(연속적인 변화)을 충분히 반영하지 못한다.

- **Core Contribution**: 이 논문은 feature alignment를 one-shot이 아닌 연속 행동의 Markov decision process로 재정의하고, reinforcement learning 기반으로 정렬을 수행한다. Proximal Policy Optimization으로 학습된 정책이 현재 feature-space 구성에 반응해 고장 유형별 affine correction을 적용하며, 리워드에서 간극 최소화와 클래스 분리도 보존을 동시에 균형 있게 추구하도록 설계한다. 또한 asymmetry-aware 전략으로 실데이터는 Normal에 우선 배치하고, 고장 클래스는 정책 정렬에 맞춘 simulated 샘플을 보강해 데이터 제약을 완화한다.

- **Technical Challenges**: 핵심 기술 난제는 fault-type-specific로 서로 이질적인 정렬을 하면서도 inter-class separability를 유지해야 한다는 점이다. 정적 gradient descent로는 모든 클래스 분포를 동시에 reshape할 때 발생하는 상태 의존성 문제를 풀기 어렵기 때문에, 논문은 정렬 과정을 연속 상태에서 반복 결정하는 MDP로 모델링했다. PPO 정책이 현재 feature-space의 상태를 보고 변환 파라미터를 동적으로 산출하도록 하여, 간극 축소와 분리도 보존을 이중 목표로 맞추는 방식으로 해결한다.

- **Empirical Impact**: XJTU-SY, CWRU, 그리고 slewing bearing 자체 테스트베드에서 검증한 결과, reinforcement learning-driven alignment가 지배적인 성능 향상을 제공한다. 특히 cross-equipment linear probing에서 encoder 재학습 없이 92.8%를 달성해, 특정 장비에 고정되지 않는 transferable monitoring 역량을 보여준다. 이는 sim-to-real 갭과 고장 이벤트 희소성 같은 실무 제약 하에서도 신뢰도 높은 fault diagnosis를 가능하게 하는 접근으로 평가된다.



### The Hitchhiker's Guide to Agentic AI: From Foundations to Systems (https://arxiv.org/abs/2606.24937)
- **Prior Approaches**: 기존에는 에이전틱 AI를 LLM 성능이나 단일 구성요소 중심으로 접근하는 경우가 많아, 파이프라인 전반을 아우르는 관점이 부족했습니다. LLM 계층(학습·추론 최적화)과 정렬·추론 계층(RLHF 계열), 에이전트 계층(메모리·툴 사용·조정)을 각각 따로 다루다 보니 프로덕션 적용에서 병목이 드러난다는 한계가 있었습니다. 또한 멀티에이전트 협업이나 평가/배포 관점이 체계적으로 정리되지 않은 점도 실무자에게 장벽으로 작용했습니다.

- **Core Contribution**: 이 자료는 ‘에이전틱 시스템을 잘 만들려면 파이프라인의 모든 레이어를 이해해야 한다’는 중심 논지를 바탕으로, 처음부터 끝까지의 전 스택을 한 흐름으로 정리합니다. LLM 기초(Transformer, GPU 시스템, SFT/LoRA/MoE, 압축, 추론 최적화)부터 정렬·추론(RLHF, PPO, DPO, GRPO, reward modeling, reasoning용 RL) 그리고 에이전트 설계(RAG, 메모리, 컨텍스트 관리, 에이전트 패턴)까지 연결해 실무 의사결정을 돕는 것이 핵심입니다. 특히 에이전트 간 조정(MCP, A2A, 중앙/분산/계층형 구조)과 UI·평가·배포까지 포함해 “구축” 관점의 완성도를 높였습니다.

- **Technical Challenges**: 에이전틱 AI는 모델 품질만으로는 끝나지 않고, 학습·추론 효율과 정렬 신뢰성, 그리고 실행 중 컨텍스트 관리가 동시에 맞물려야 한다는 기술적 난도가 큽니다. 이 자료는 LLM 학습/미세조정(SFT, LoRA, MoE)과 추론 최적화, 정렬 방법(RLHF/PPO, DPO 및 변형, GRPO) 및 reward modeling을 기초로 두고, 에이전트 계층에서는 trajectory 기반 학습, Agentic RAG, 다양한 메모리( in-context, external, episodic, semantic )로 문제를 분해해 접근합니다. 더불어 에이전트 툴 사용·스킬 설계, 프로토콜(MCP, A2A)과 멀티에이전트 토폴로지로 상호작용을 구조화하고, 구현 예시와 코드 중심 가이드로 연결성을 제공합니다.

- **Empirical Impact**: 단순 개념 소개가 아니라 각 장에 이론 근거와 구현 가이드를 함께 제공해, 실무자가 에이전트 시스템을 단계적으로 검증하고 배포로 옮기는 데 도움이 됩니다. 특히 평가 방법론과 프로덕션 배포까지 포함해, 실험 성능에서 실제 태스크 성능으로의 전이를 촉진하는 점이 의미 있습니다. 결과적으로 에이전틱 AI를 ‘전략’이 아니라 ‘엔지니어링’으로 다루려는 업계/연구 커뮤니티의 학습 경로를 정리해주는 역할을 합니다.



### Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs (https://arxiv.org/abs/2606.24897)
Comments:
          14 pages, 1 figure. Pre-registered on OSF. Data and code available on Zenodo and GitHub

- **Prior Approaches**: 생의학 텍스트 마이닝, 과학계량학, 그리고 생의학 LLM 학습 코퍼스 구축은 서지 API가 반환하는 초록이 실제 출판 초록을 그대로 재현한다고 가정해왔다. 그러나 그 재현 품질이 문자 수준에서 어떻게 달라지는지, 그리고 어느 요소가 어떤 API에서 왜 깨지는지는 체계적으로 감사되지 않았다.

- **Core Contribution**: 이 논문은 OSF에 사전 등록된 감사(pre-registered audit)로, PubMed E-utilities, Crossref, OpenAlex, Semantic Scholar의 초록 텍스트를 PMC JATS XML을 공통 기준(ground truth)으로 비교한다. 2024년 PMC 오픈액세스 약 70만 편 중 영어 연구 논문 4,000편을 샘플링해, JATS 초록에 포함된 특정 유니코드 문자 범주가 각 API에서 보존되는지 문자 충실도를 측정했다.

- **Technical Challenges**: 핵심 난제는 API마다 초록 제공 방식이 다를 수 있어, ‘내용’이 아니라 ‘표면 문자 서명(character-level signature)’의 손실을 재현 가능하게 계측해야 한다는 점이다. 연구진은 유니코드 문자 클래스(타이포그래피 구두점, 수학/과학 기호, 그리스 문자, 특수 공백)를 사전 정의하고 블라인드 메커니즘 감사로 손실 원인을 문자 치환과 inverted-index 직렬화로 귀결해 설명했다.

- **Empirical Impact**: 결과적으로 두 가지 결정적(deterministic) 손실이 임계 기준을 충족할 정도로 관측됐는데, PubMed AbstractText는 타이포그래피 구두점 보존이 0.6% 수준에 그쳤고 OpenAlex는 특수 공백이 0%로 나타났다. 반면 수학 기호와 그리스 문자는 네 API 모두 95% 이상 보존됐고, Crossref는 논문 24.6%에서 초록을 아예 반환하지 않아 커버리지(75.4%, 95% CI 74.1-76.7%) 격차도 확인됐다. 따라서 같은 PMC JATS 텍스트라도 API에 따라 토큰화 민감 지표와 문자 기반 글쓰기/코퍼스 품질에 직접적인 영향이 생기며, API별 문서화되지 않은 문자 수준 불일치를 고려한 코퍼스 설계가 필요하다는 점을 강하게 시사한다.



### ESBMC-PLC+: A Unified IEC 61131-3 Formal Verification Framework as a PLCverif Successor (https://arxiv.org/abs/2606.23870)
Comments:
          21pages

- **Prior Approaches**: PLCverif는 CERN에서 개발돼 2019년부터 운영 중인 성숙한 오픈소스 PLC 형식검증 도구다. 다만 (1) Ladder Diagram(LD) 입력을 지원하지 않아 LD 코드를 수작업 번역해야 하고, (2) 주된 백엔드가 CBMC라 bounded proof에 머물며, (3) 그래픽 LD에서는 타이머/카운터 function block이 포함된 경우 검증이 불완전하다. 후속으로 ESBMC-PLC(텍스트 LD, k-induction)와 ESBMC-GraphPLC(그래픽 PLCopen XML LD) 등이 나왔지만, Structured Text(ST/SCL)는 빠져 있고 그래픽 LD의 타이머/엣지 트리거까지는 커버하지 못했다.

- **Core Contribution**: 이 논문은 ESBMC-PLC+로 세 가지 IEC 61131-3 입력 형식(텍스트 LD, 그래픽 LD, ST/SCL)을 단일 ESBMC 백엔드로 통합해 unbounded safety proof(k-induction)를 제공한다. ST/SCL은 MATIEC로 C로 컴파일한 뒤 ESBMC에 맞게 scan-cycle 모델과 nondeterministic 입력, YAML 속성 주입을 얹는다. 그래픽 LD는 function block 상태 의미론을 추가해 TON/TOF/TP, CTU/CTD, R_TRIG/F_TRIG를 scan-cycle 지속 상태 변수로 모델링하여 기존의 “타이머 포함 그래픽 LD 제외” 공백을 메운다.

- **Technical Challenges**: 핵심 기술 난제는 (a) PLC의 scan-cycle 동기적 실행을 ESBMC의 무한 경로 검증 모델로 정확히 재현하고, (b) ST/SCL을 C로 바꾼 뒤 물리 I/O를 열려 있는 입력(undconstrained)으로 취급하면서도 속성 검증 지점을 정확히 유지하는 것이었다. 또한 그래픽 LD에서 timer/counter/edge-trigger function block은 단순 표현이 아니라 이전 스캔의 내부 상태에 의존하므로, DFS 기반 rung 추출 단계에서 persistent scan-cycle state를 GOTO IR에 반영해야 했다. ESBMC-PLC+는 MATIEC 생성 C에 scan wrapper를 덧씌워 __ESBMC_assert()를 삽입하고, 그래픽 LD에 대해서는 GOTO IR에서 상태 갱신 코드를 생성해 함수블록 출력을 rung의 불리언 항으로 연결함으로써 두 문제를 해결했다.

- **Empirical Impact**: ESBMC-PLC+는 벤치마크 8개(타이머 포함 포함)에서 PLCverif와 비교해 입력 커버리지는 맞추면서 더 강한 unbounded 보장을 제공한다. nuXmv(BDD 백엔드)와의 직접 비교에서는 타이머 프로그램에서 ESBMC-PLC+가 400–2,000배 빠르며, nuXmv는 120초 내 타임아웃 나는 케이스를 ESBMC-PLC+가 증명 완료했다. 즉, LD/ST 전 언어 포맷을 ESBMC 하나로 묶고 function block의 의미론까지 포함해 실용적 규모의 안전성 검증 가능성을 실험적으로 크게 확장했다.



### PhoneBuddy: Training Open Models for Agentic Phone Us (https://arxiv.org/abs/2606.23049)
- **Prior Approaches**: 기존 모바일 에이전트 연구는 위젯 인식과 행동 예측을 넘어 실제 앱 실행까지 확장해 왔지만, 모델을 “실제 폰 작업 완료”로 개선시키는 학습 설계는 여전히 해법이 제한적입니다. 특히 실앱 환경은 느리고 상태를 리셋하기 어렵고 자동 검증도 까다로워 확장성이 떨어지는 반면, 합성/모의 환경은 재현성은 좋지만 실제 이전(transfer)에 한계가 있습니다. 결과적으로 open phone-use 모델이 실폰에서 신뢰성 있게 태스크를 끝내도록 만드는 교육 레시피의 불일치 문제가 남아 있었습니다.

- **Core Contribution**: PhoneBuddy는 real-app RL과 mock-app RL을 “대체”가 아니라 “보완 결합”으로 다루는 훈련 레시피와 오픈 모델 라인을 제시합니다. 실앱은 현실성·사이드이펙트·계정 의존 동작에 모델을 고정하고, PhoneWorld는 실제 GUI 사용 구조에서 재구성한 실행 가능한 mock 앱으로 리셋/반복/자동 검증 가능한 상호작용 신호를 제공합니다. 실험적으로는 동일한 SFT(공유 supervised fine-tuning) 이후에 RL 가지(Real-only vs Real+Mock)에 따른 차이를 분리해 비교합니다.

- **Technical Challenges**: 핵심 기술 난제는 학습 중 목표(태스크 완료) 신호를 실폰에서는 자동 검증이 약하고, 모의 환경에서는 현실성이 부족해 두 세계의 보상/검증 방식을 어떻게 정렬하느냐입니다. PhoneBuddy는 실앱에서는 관찰 가능한 UI/상호작용 트레이스에 대해 rubric-based model judging으로 완료 여부를 프록시 보상화하고, PhoneWorld에서는 내장 rule-based verifiers로 동일한 이진 완료 목표를 직접 확인하는 방식으로 맞춥니다. 또한 비교 공정성을 위해 백본(Qwen3.5-4B), 액션 인터페이스, SFT 초기화, 평가 프로토콜을 고정하고 마지막 RL 브랜치만 바꿔 원인 귀속을 명확히 했습니다.

- **Empirical Impact**: 150개 실폰 휴먼 평가에서 task success rate는 SFT 36.67%에서 real-app RL 40.67%로 오르고, 여기에 PhoneWorld 기반 mixed RL을 추가하면 45.33%까지 상승했습니다. AndroidWorld에서도 60.3%→77.2%→83.2%로 개선이 단조롭게 커졌고, 성과는 Single-App과 WeChat mini-app 같은 구조가 안정적인 영역에서 특히 크게 나타났습니다. 반대로 Cross-App 워크플로는 세 체크포인트 모두 낮은 성능에 머물러 개선이 제한적이었으며, 앱 간 정보 핸드오프·장기 상태 추적·런타임 검증 강화가 향후 과제로 남았습니다.



New uploads on arXiv(cs.IR)

### AutoRelAnnotator: Calibrated Model Cascades for Cost-Efficient Relevance Evaluation in Sponsored Search (https://arxiv.org/abs/2606.25871)
Comments:
          Accepted at E-commerce workshop, SIGIR 2026

- **Prior Approaches**: 기존에는 인간 라벨링 비용·지연이 커서 대규모 relevance annotation이 병목이 됐고, off-the-shelf LLM은 도메인 특화 과업에서 정확도가 낮았다. LLM cascade 같은 라우팅 기법은 비용을 줄이지만, 결국 구성 모델들의 정확도 상한을 넘기 어렵다는 한계가 있었다. 또한 기존 보정(calibration)은 글로벌 방식 위주라 클래스별로 다른 미스캘리브레이션을 충분히 반영하지 못했다.

- **Core Contribution**: 이 논문은 정확도와 비용을 분리해 최적화하는 calibrated model cascade를 제안한다. 먼저 도메인별 fine-tuning으로 높은 정확도의 분류기(예: 87–89%)를 만든 뒤, per-class isotonic calibration을 결합해 3단계 라우팅(Cross-Encoder→Gemma-2B→LLaMA-8B)을 수행한다. 그 결과 cascade는 비용 절감과 정확도 보존을 동시에 노리되, 보정으로 소폭의 추가 이득까지 확보한다.

- **Technical Challenges**: 핵심 과제는 라우팅에 쓰는 confidence가 잘못 보정(miscalibrated)돼 있으면 “안전한” 임계값 설정이 무너지는 점이었다. 논문은 클래스별 예측 분포가 다른 현상을 관찰하고, per-class isotonic regression으로 클래스 조건부 보정을 학습해 라우팅 임계값(threshold)이 실제 정확도와 정렬되도록 했다. 또한 classification head 기반 분류 확률을 사용해 생성형 LLM의 프롬프트 취약성·캘리브레이션 불안정성을 피했다.

- **Empirical Impact**: 실험에서 fine-tuning은 off-the-shelf LLM 대비 약 20%p 격차를 메웠고, cascading은 정확도에 거의 영향을 주지 않으면서 compute cost를 절반으로 줄였다. per-class isotonic calibration은 cascade 정확도를 88.5%→89.1%로 끌어올리며, 강한 글로벌 calibration baseline 대비 +0.6%p의 통계적으로 유의미한 이득(p<0.05)을 보였다. Q3 2024 이후 6개 오프라인 use case에서 150M+ annotation을 처리했고, 라벨링/분석 처리 리드타임을 5일에서 1–3시간으로 단축해 검색·광고 ML 워크플로의 실험 속도를 높였다.



### How Large Language Models Source Brand Reputation Across Languages and Markets (https://arxiv.org/abs/2606.25787)
Comments:
          12 pages, no figures, tables only. Data and analysis ledger on Zenodo, this https URL

- **Prior Approaches**: 기존 AI 브랜드 가시성 연구는 주로 모델이 생성한 답변 텍스트(감성, 추천, 경쟁사 언급 등)만 분석했다. 하지만 생성형·그라운딩된 모델은 답변을 웹에서 가져온 뒤 인용에 기반해 말하기 때문에, 실제로는 답변 내용보다 인용(출처) 레이어가 더 먼저 결정한다. 또한 AI 검색은 전통 검색과 달리 인용 출처가 특정 채널에 집중되고, 엔진·시점에 따라 출처 구성이 불안정해질 수 있어 출처 측정의 필요성이 제기돼 왔다.

- **Core Contribution**: 이 논문은 “LLM이 브랜드 정보를 어디서 가져오는가”를 답변 텍스트가 아니라 citations(인용 URL/도메인)로 한 단계 앞에서 측정한다. 12개 홈마켓·13개 언어·128개 브랜드에 대해 167,551개 URL-grounded citations(총 189,974개 attribution row)를 도메인과 source type으로 분류해 언어·시장별로 인용 구조를 정량화했다. 그 결과, 브랜드 자체가 아니라 제3자 웹이 AI 브랜드 답변의 핵심 근거가 되는 패턴을 교차 시장에서 확인했다.

- **Technical Challenges**: 핵심 난제는 인용 URL을 신뢰할 수 있는 도메인으로 정규화하고, “소유(owned)” 여부를 정확히 판정하는 것이었다. 특히 Gemini 인용에서 Google redirector(정점 기반 리다이렉트)가 섞여 owned 판정이 0%처럼 보일 수 있어, citation-title에서 실제 도메인을 100% 해소(resolve)한 뒤 분석했다. 또한 데이터가 단일 균질 표가 아니라 “backbone+cross-link” 구조라 URL 기반 지표는 NB+PL 중심으로 계산하고 CEE는 별도 보고했으며, 인용은 엔티티에 고정되지 않고 응답에 연결돼 브랜드별 수치에는 베이스 차이와 해석 유의가 필요하다고 밝혔다.

- **Empirical Impact**: 분석 결과, AI는 브랜드 답변을 85.7%에서 비소유(제3자) 사이트에 근거하고 14.3%만 브랜드 소유 사이트를 인용한다. 도메인 인용은 강하게 집중돼 80%가 전체 도메인의 약 18%에서 나오며 Zipf 법칙(α=0.86, R²=0.983)에 부합했다. 또한 Wikipedia는 12개 언어 중 11개 언어에서 최다 인용 도메인이고, 리투아니아는 vz.lt가 예외로 나타났으며, 폴란드에서는 YouTube와 HR·커리어 포털(Indeed 등)이 Wikipedia를 제치며 시장 특이성이 드러났다. 마지막으로 모델별로 인용 범위가 달라 Perplexity는 가장 많이·넓게 인용하고 Gemini는 redirector 해소 전에는 소스가 숨겨져 보일 수 있어, 단일 모델·단일 audit만으로는 편향 위험이 있음을 시사한다.



### A Stochastic Epidemiological Model of Latent Tuberculosis in a Radiation Exposed Mars Colony (https://arxiv.org/abs/2606.25728)
- **Prior Approaches**: 기존 연구는 TB의 잠복→활성 전이를 생물학적 관점에서 모델링해 왔지만, 우주 임무의 만성 방사선이 면역을 어떻게 떨어뜨리고 그 결과 전염(공기 매개)까지 이어지는 연결고리는 충분히 다루지 못했다. 또한 우주선/정거장 환경의 미생물·면역 변화를 관측해 온 연구는 많았지만, 소규모 폐쇄 집단에서 전파가 실제로 어떻게 동역학을 만들지는 통합적으로 설명하는 메커니즘 모델은 부족했다.

- **Core Contribution**: 이 논문은 Mars 콜로니에서 ‘숙주-방사선-병원체-서식지’의 확률적(stochastic) 연결을 반영해, galactic cosmic radiation→면역 능력 변화→잠복 TB 재활성→폐쇄 서식지 공기 전파로 이어지는 경로를 하나의 모델로 묶었다. 여기에 격리(isolation)와 약물(medication) 같은 대응책을 부분관측 하 순차의사결정(Partially observable sequential decision)으로 정식화하고, 기준선(baselines) 또는 proximal policy optimization으로 에이전트 정책을 학습하도록 구성했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 폐쇄된 소규모 집단에서 ‘초기 감염자 0명’인데도 잠복 저장소가 언제 발화해 유행을 시작하는지 같은 비선형·확률 효과를 재현하는 것과, (2) 방사선-면역-재활성 민감도 같은 불확실성을 제어 정책 설계에 반영하는 것이다. 논문은 에이전트 기반 시뮬레이터로 전파·임상 전이를 개별 단위 확률로 생성하고, 면역 능력의 시간 업데이트와 재활성 위험률(면역 저하에 대한 증가)을 결합한 뒤, 자원 제약과 개입 부담(격리/약물 비용)을 포함해 정책을 최적화했다.

- **Empirical Impact**: 시뮬레이션 결과, 초기 active TB가 없더라도 잠복 저장소와 면역 저하를 통해 active TB가 내생적으로 발생할 수 있음을 보였다. 위험은 잠복 저장소 크기(latent reservoir size), 방사선-면역 결합(radiation-immune coupling), 재활성 민감도(reactivation sensitivity)에 특히 민감했고, 적응형 제어(adaptive control)는 감염 부담과 사망을 줄이면서도 불필요한 개입은 제한했다. 저자들은 이 프레임워크가 발사 전 선별·모니터링·차폐·치료 전략을 미션별로 스트레스 테스트하는 도구로 활용될 수 있다고 제안한다.



### Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Sca (https://arxiv.org/abs/2606.25496)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 숏비디오 추천은 오프라인에서 만든 고정 풀의 영상을 검색·랭킹하는 콘텐츠-퍼스트 패러다임에 기반한다. 최근 generative recommendation models(생성형 추천, GRM)은 semantic IDs(SIDs)를 생성해 사용자 관심을 모델링하지만, 기본적으로는 여전히 “풀에 있는 것 중 최선”을 고르는 구조여서 롱테일·동적 관심을 세밀하게 반영하기 어렵다.
또한 AIGC 비디오 생성 모델은 품질은 높아도 수동 prompting, 다단계 리파인, 공정 지연이 커서 수억 명 사용자 규모의 실서비스에 바로 얹기 힘든 제약이 있었다.

- **Core Contribution**: 이 논문은 Recommendation-as-Generation(RaG)이라는 새 패러다임을 제안하며, 추천을 “조회”가 아니라 “관심에서 개인화 비디오를 생성”하는 문제로 재정의한다. 핵심 인터페이스로 Disentangled Semantic IDs(D-SIDs)를 도입해 비디오 표현을 콘텐츠 의미(entities·topics)와 크리에이티브 스타일(style·rhythm·atmosphere)로 분리하고, 생성까지 제어 가능한 형태로 만든다.
또한 Video Generation Agents(VGAs)를 넣어 D-SIDs 기반으로 계층적 계획과 정제를 수행하며, 폐루프 최적화를 Synergistic Cross-Domain Reward Learning(SCRL)로 닫아 추천-생성-피드백이 함께 진화하도록 설계했다.

- **Technical Challenges**: 첫째, 추천과 생성은 데이터/목표가 달라 단절되기 쉬운데, RaG는 D-SIDs를 공유 잠재 공간(추천 입력=생성 조건)으로 만들어 예측한 관심을 곧바로 디코딩 가능한 표현으로 연결했다. 둘째, 산업 규모에서 고품질·관심정렬을 동시에 만족시키려면 단일 생성 파이프라인의 일관성 붕괴와 지연 문제가 큰데, VGAs를 비주얼 계획-오디오 정렬-효과 보강의 멀티에이전트 계층 구조로 쪼개고 bounded reflection(반복 2회)로 크로스모달 일관성을 보정했다.
마지막으로 이질적인 보상(사용자 피드백, 관심 정렬, 비디오 품질)을 안정적으로 함께 학습해야 했고, SCRL에서 user feedback을 주목적으로 두고 interest alignment와 quality를 constraints로 두는 constrained policy learning(GDPO + PID 제어 라그랑주 업데이트)로 학습 불안정을 완화했다.

- **Empirical Impact**: RaG는 일 산업 플랫폼(일 4억 명 이상 MAU)에서 광고 수익이 핵심인 시나리오로 배포되었고, 강력한 production GRM baseline 대비 온라인 A/B 테스트에서 최대 1.87% 광고 매출 개선을 보였다. 이는 “풀 내 검색” 성능을 넘어 생성 기반 개인화가 실제 비즈니스 지표에서 추가 수익으로 이어질 수 있음을 보여준다.
저자들은 또한 이 접근이 추천과 personalized video generation을 프로덕션 스케일에서 효과적으로 통합한 최초 사례 중 하나라고 주장하며, closed-loop generative system이 실서비스 적용 가능한 유망 패러다임임을 강조한다.



### S2-CAR: Segmentation-Supervised Complexity-Adaptive Recommendation (https://arxiv.org/abs/2606.25415)
- **Prior Approaches**: 순차 추천은 사용자 상호작용 이력을 바탕으로 다음 아이템을 예측하지만, 기존 모델들은 긴 시퀀스에서 사용자 의도가 시간에 따라 이질적으로 변하는 상황을 충분히 분리하지 못한다. GRU4Rec, SASRec, BERT4Rec처럼 전체를 단일 문맥으로 다루거나, TLSRec류의 fixed time-gap 기반 세션 분할처럼 휴리스틱 규칙에 의존하면 의도 경계가 어긋나 cross-intent interference와 단기 신호 과의존이 생긴다. 또한 multi-interest 모듈이 대부분 동일한 수의 interest slot을 배정해 구간별 복잡도 차이를 반영하지 못한다.

- **Core Contribution**: 이 논문은 S2-CAR( Segmentation-Supervised Hierarchical Multi-Intent Network with Complexity-Adaptive Regularization )를 제안해 사용자 의도를 연속적인 latent energy 상태로 보고, 의도 경계를 고정 간격이 아닌 에너지 감쇠에 의해 자동 분할한다. 이어서 Segment-Count-Adaptive Multi-Intent Extraction으로 세그먼트 복잡도에 맞춰 interest 슬롯 수를 동적으로 조절하며, 구간 간 주기적 반복으로 인한 중복을 압축한다. 마지막으로 계층적 인코더에서 구조적으로 정렬되는 contrastive supervision 경로를 두어 학습 안정성과 추론의 causal fidelity를 함께 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 고정 시간창 대신 행동 동역학에서 의도 경계를 찾아야 하고, (2) 서로 다른 길이·불확실성을 가진 세그먼트를 과도한 슬롯 낭비 없이 압축해 multi-interest를 만들어야 하며, (3) 세그먼트 수준 구조를 학습하지만 추론 시퀀스 질서를 해치지 않는 정규화가 필요하다는 점이다. 이를 위해 Context-Aware Soft Temporal Point Process(Soft-TPP)로 latent energy의 잔존 정도(energy retention ratio)를 추정하고 threshold로 boundary flag를 생성하되, downstream 추천 손실로 Soft-TPP를 직접 학습하지 않는 분리 학습을 적용한다. 이후 Hierarchical Interest Encoder가 세그먼트 응집→multi-interest 추출→contrastive loss 정렬을 수행하고, Segment-Count-Adaptive 모듈은 세그먼트 수에 따라 압축/활성 슬롯을 조절한다.

- **Empirical Impact**: 실험은 영화·커머스·게임 3개 공개 벤치마크에서 13개 베이스라인을 대상으로 수행되었으며, S2-CAR는 모든 데이터셋과 지표에서 state-of-the-art 성능을 일관되게 달성했다고 보고한다. 특히 긴 상호작용 기록을 가진 사용자에서 개선 폭이 크며, 기존 모델이 middle 구간 신호를 체계적으로 덜 활용하는 문제를 energy 기반 경계 추정이 완화한다는 분석이 제시된다. 또한 Soft-TPP 기반 에너지 분할은 plug-and-play 모듈로서 다양한 sequential recommendation 백본에 통합해도 성능 향상을 제공한다는 점에서 실무적 활용 가능성이 높다.



### TheoremGraph: Bridging Formal and Informal Mathematics (https://arxiv.org/abs/2606.25363)
Comments:
          31 pages, 9 figures, 21 tables

- **Prior Approaches**: 기존 연구는 수식·기호를 포함한 수학 문서에서 키워드나 노테이션 중심 검색을 시도했지만, 문장(정리/정의) 간 의존 구조는 상대적으로 약하게 다뤘습니다. 또 일부는 Lean 라이브러리 쪽에서 선언 간 의존을 추출했으나, 범위가 제한되거나(형식화 규모) 혹은 정보가 코퍼스 전반에서 일관되게 연결되기 어려웠습니다.

- **Core Contribution**: 이 논문은 정리(명제) 단위 의존을 비형식(informal)과 형식(formal) 모두에서 한 그래프로 통합하는 TheoremGraph를 제안합니다. 비형식 쪽에서는 arXiv의 theorem-like 환경 1,170만 개에서 1,830만 개의 후보 directed dependency를 복원하고, 형식 쪽에서는 Lean 4 elaborator 수준에서 388,105개 노드와 1,133만 개 typed edge를 갖는 LeanGraph를 공개합니다.

- **Technical Challenges**: 핵심 난제는 서로 다른 표현 양식의 ‘같은 수학 결과’를 안정적으로 연결하는 것이며, 이를 위해 slogan(자연어 슬로건)을 생성해 공유 의미 공간에 임베딩하고 cosine 유사도로 교차 매칭합니다. LLM judge로 검증한 결과 cosine 0.8 이상에서 47,952개의 (inexact 포함) 매치를 확정했고, 유사도 0.9 이상에서는 판단 승인율이 87%까지 올라가 품질을 끌어올렸습니다.

- **Empirical Impact**: 형식 개념 검색에서는 name-and-signature 표현에 graph expansion을 더한 구성이 LeanSearch v2의 reranked Recall@10(0.780) 대비 0.5pp 내(0.775)로 근접하며, 추가 LM reranker 없이도 경쟁력을 보였습니다. 또한 slogan 기반 검색을 RAG로 결합해 LLM의 autoformalization 정확도를 24개 타깃 중 5개에서 8개로 개선하고, 데이터셋·extractor·HTTP API·MCP 인터페이스를 공개해 수학 검색·귀속·검색증강 추론 인프라로 확장 가능성을 제시합니다.



### Adaptive Re-Ranking (https://arxiv.org/abs/2606.25249)
Comments:
          7 pages

- **Prior Approaches**: 기존 ad-hoc IR은 retrieve-then-rerank(후처리 재정렬) 파이프라인으로, 항상 무거운 cross-encoder를 써서 상위 후보를 다시 정렬한다. 이 방식은 질의 복잡도와 무관하게 동일한 비용을 지불해 단순 질의에서 계산이 낭비되며 지연이 커진다.

- **Core Contribution**: 논문은 Adaptive Re-Ranking으로, 질의마다 기대 효과-지연 비용을 함께 고려해 BM25(재정렬 생략) / Light reranker / Heavy reranker 중 하나를 라우팅한다. 이를 위해 nDCG@10과 MRR@10의 효과성과 실측 latency를 결합한 utility 기반 라벨링 프레임워크를 제안한다.

- **Technical Challenges**: 핵심 난제는 “효과를 높이는 전략”을 제한된 감독 신호로 학습해 per-query 라우팅으로 일반화하는 것이다. 저자들은 BEIR 학습 split에서 각 질의의 모델별 효과와 wallclock latency를 계산해 최적 전략을 라벨링하고, 클래스 불균형을 down-sampling으로 보정한 뒤 bert-base-uncased 분류기를 3-class 라우터로 fine-tuning했으며 라우터 자체 오버헤드는 약 3.6ms로 측정했다.

- **Empirical Impact**: 실험에서 학습된 라우터는 BGE 대비 모든 데이터셋에서 중간값 기준 1.15~53.2배, 평균값 기준 1.11~5.22배 더 낮은 지연을 보이면서 nDCG@10은 -17.5%~+4.0%로 경쟁력을 유지했다. 특히 NFCorpus와 SciFact에서 median latency가 각각 53.2배, 33.6배 감소하는 등 큰 비용 절감 효과가 나타났고, 도메인 외 데이터에 대한 zero-shot에서도 지연 이점이 유지되어 확장성 가능성을 뒷받침한다.



### Extreme Meta-Classification for Large-Scale Zero-Shot Retrieva (https://arxiv.org/abs/2606.25237)
Comments:
          Accepted at KDD 2024, 20 pages

- **Prior Approaches**: 대규모 dense retrieval은 쿼리와 아이템을 임베딩으로 가깝게 배치해 ANNS로 빠르게 검색하지만, 실시간 삽입을 위해 작은 인코더를 쓰는 경우가 많아 필요한 world knowledge를 충분히 담기 어렵다. Extreme classification(XC)은 관측된 각 아이템마다 1-vs-all classifier를 학습해 정확도는 높이지만, novel item(새 아이템)은 클릭 데이터가 없어 classifier를 학습·추가하기가 불가능해 zero-shot 검색을 지원하지 못한다.

- **Core Contribution**: 이 논문은 관측 아이템의 classifier 지식을 재조합해 novel item의 classifier를 즉석에서 합성하는 프레임워크 EMMETT를 제안한다. 그 한 구현체로 IRENE을 제시하며, (1) novel item과 가장 관련 있는 관측 classifier를 빠르게 고르는 classifier selector와 (2) 선택된 classifier들을 transformer 기반 생성기로 meta-classifier로 합성하는 generator를 결합한다.

- **Technical Challenges**: 핵심 난제는 novel item에 대해 classifier를 만들되, 지연(latency) 없이 정확도를 유지해야 한다는 점이다. IRENE은 ANNS 기반 MIPS로 소수(K≈3)의 classifier만 골라 불필요한 잡음을 줄이고, 생성기 학습 시에는 extreme classifiers와 인코더를 freeze한 채 one-vs-all 형태를 zero-shot 일반화에 맞춰 설계한 BCE 기반 학습으로 과적합을 방지한다.

- **Empirical Impact**: 실험 결과 IRENE은 leading encoder 위에 얹었을 때 Recall@10 기준 최대 15%p까지 zero-shot 검색 정확도를 개선한다. 또한 주요 검색엔진의 광고 online A/B 테스트에서 ad click-through rate를 4.2% 개선했으며, 선택한 설계 요소들에 대한 ablation으로 성능 기여를 확인했다.



### TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems (https://arxiv.org/abs/2606.25147)
- **Prior Approaches**: 기존 산업용 추천은 LEM(대형 임베딩 모델)처럼 사용자/아이디를 고정 차원의 dense embedding으로 압축해 왔다. 하지만 이런 방식은 표현 제약이 커서 일반화에 한계가 있고, LLM 기반 텍스트 프로필은 대개 비정형 ID 공간에서 modality gap과 함께 ‘순차 행동 역학’보다는 ‘토픽 공기(共起)’를 학습하는 문제가 보고돼 왔다.
또한 PLUM은 item retrieval엔 효과적이었지만, sequential user modeling(SUM)으로 확장해 SID 기반 이산 표현을 어떻게 사용자에게 적용할지 자체가 충분히 탐색되지 않았다.

- **Core Contribution**: TokenMinds는 PLUM(semantic ID, SID) 패러다임을 사용자 모델링으로 확장해, 한 번의 encoder-decoder에서 dense user embedding과 SID 기반 이산 user token을 동시에 생성한다. 이 dual-output 설계로 ‘의미적으로 grounded된 토큰 표현’과 ‘기존 downstream이 기대하는 dense 벡터 호환성’을 함께 확보하는 것이 핵심이다.
또한 동일 SID 어휘를 공유해 long-form 비디오(LFV)와 short-form 비디오(SFV)를 하나의 모델로 통합하고, multi-context decoding으로 시나리오별 토큰을 효율적으로 뽑아내며 학습/서빙 비용을 절감한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 LLM이 대규모 비텍스트 ID 공간에 대해 학습된 지식과 SID 모달리티를 잘 정렬하는 동시에, 사용자 시퀀스에서 의미 있는 SID 토큰을 안정적으로 생성하는 것이다. TokenMinds는 PLUM의 Continued Pre-Training(CPT)로 SID 모달리티 정렬을 먼저 수행하고, SFT에서 prefix-L SID(계층형 코드워드의 상위 접두부)만 loss에 사용하도록 설계해 복잡한 grounding 부담을 줄인다.
서빙 레이턴시 제약을 해결하기 위해서는 UBS(User Behavior Service) 기반 비동기 인프라로 표현 생성과 실시간 scoring을 분리하고, 캐시된 표현을 scoring 시점에 재사용한다.

- **Empirical Impact**: YouTube LFV·SFV 및 검색 쿼리를 포함한 대규모 학습 데이터로 offline 실험과, 실제 프로덕션 트래픽에서의 live A/B 테스트를 모두 수행해 토큰 기반 사용자 표현의 산업적 실행 가능성을 입증했다. 특히 dense embedding과 SID token은 보완적으로 작동해 ranking 통합 시 유의미한 개선(예: 코어 지표 +0.11%, 참여 지표 +0.62% 수준)을 보고했다.
또한 LFV와 SFV를 단일 프레임으로 통합해 학습/서빙 비용을 줄이면서도 핵심 품질과 fresh 콘텐츠 지표를 함께 유지·개선하며, ‘시나리오 통합’의 실용적 해법을 제시했다.



### Tracing Target Answers in Poisoned Retrieval Corpora via Token Influence Attribution (https://arxiv.org/abs/2606.25721)
- **Prior Approaches**: RAG corpus poisoning은 악의적 문서를 검색에 포함시켜 LLM의 생성을 특정 오답으로 유도하는 공격이다. 기존 탐지는 보조 분류기나 추가 LLM 검증처럼 별도 모델/추론을 붙여 정확하더라도 계산·운영 비용이 커지는 문제가 있었다. 문서 단계 필터링·포렌식류도 존재하지만, 정답을 유도하는 토큰을 직접 추적해 확인하는 방식은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: TRACE는 token influence attribution을 활용해 “오답(공격자가 지정한 타깃 답)”에 강하게 영향을 주는 토큰을 찾아내는 RAG poisoning 탐지 프레임워크다. 문서들을 benign/malicious로 분류하는 대신, 여러 문서에서 반복적으로 등장하는 고영향 키워드(토큰)를 뽑고 2단계 검증으로 포이즈닝을 확정한다. 또한 탐지와 동시에 공격자가 심은 타깃 답 토큰을 드러낼 수 있음을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 비용이 낮으면서도, 잡음 토큰(구두점·빈도 높은 단어 등)을 배제하고 정답 유도 토큰을 안정적으로 포착하는 것이다. TRACE는 먼저 검색 문서마다 “긍정 affirmation 문자열”에 대한 영향도를 backpropagation 기반으로 계산해 top-influence 토큰을 뽑되, 구두점과 빈도 단어 및 질문에 이미 있는 토큰은 제외한다. 이후 인접 토큰을 연속 구문 후보로 결합하고 다문서 공통 반복 빈도로 후보를 정제한 뒤, 두 번째 영향도 재평가로 같은 고영향 토큰이 반복되는지 확인해 공격 여부를 판정한다.

- **Empirical Impact**: PoisonedRAG 기반 공격과 Natural Questions, HotpotQA, MS-MARCO에서 6개 LLM을 대상으로 실험한 결과, TRACE는 TPR을 대부분 90% 이상으로 유지하며 FPR도 하이퍼파라미터 튜닝으로 10% 미만 수준까지 억제할 수 있었다. Phi-4-mini에서는 평균 TPR 98.87%로 특히 강한 탐지 성능을 보였고, 실행 시간도 LLM 기반 포렌식 대비 낮게 보고됐다. 더불어 타깃 답 식별 ACC가 약 70~90%대(HotpotQA는 상대적으로 하락)로 나타나, TRACE가 포이즈닝 탐지뿐 아니라 공격자가 지정한 답을 상당 부분 역추적함을 시사한다.



### BitNet Text Embeddings (https://arxiv.org/abs/2606.25674)
Comments:
          Under review

- **Prior Approaches**: 기존 LLM-based text embedder는 대조학습/지도학습으로 임베딩 품질을 끌어올리는 데 집중했지만, 실제 배포에서는 지연(latency)과 대규모 벡터 인덱스의 저장·전송 비용이 병목이 된다. 양자화(PTQ)는 빠르지만 4-bit 이하에서 성능 저하가 커서, 이를 보완하려는 QAT·BitNet류 1.58-bit 접근과 distillation이 주로 LLM 압축에 쓰여 왔다. 다만 임베딩 전용으로 “백본 연산 효율 + 벡터 저장 정밀도”를 함께 겨냥한 체계적 연구는 부족했다.

- **Core Contribution**: BITEMBED(논문에서는 BitEmbed로도 표기)는 LLM 기반 text embedding을 극저비트로 바꿔 인코딩 효율과 벡터 저장 비용을 동시에 줄이는 프레임워크를 제안한다. pretrained LLM을 BitNet-style ternary weights, quantized activations, 그리고 학습 안정화용 lightweight normalization refinement(SubLN) 구조로 변환하고, 이후 continual contrastive pre-training과 supervised contrastive fine-tuning에서 teacher-guided similarity-distribution distillation 및 attention-relation distillation을 결합한다. 더 나아가 단일 체크포인트로 여러 output embedding 정밀도(예: 16/8/4/2/1-bit)를 지원하도록 multi-precision embedding training까지 포함한다.

- **Technical Challenges**: 핵심 난제는 극단적 저비트 양자화가 임베딩 공간의 의미적 구조를 깨뜨려, 지도학습만으로는 표현력을 회복하기 어렵다는 점이다. 논문은 (1) 변환 직후 continual contrastive pre-training으로 의미 공분산/연관 신호를 다시 학습시키고, (2) full-precision fp16 teacher로부터 score-level cosine similarity 분포와 token-level attention relation을 distillation하여 저비트 모델의 미세한 상대관계를 보존한다. 또한 SubLN을 끼워 넣어 quantization로 인한 scale drift와 outlier 민감도를 완화하고, STE로 rounding/clipping 비미분 구간을 학습에 반영한다.

- **Empirical Impact**: MMTEB(eng, v2)에서 Qwen3-0.6B 및 Gemma3-270M을 대상으로 평가했을 때, BitEmbed는 full-precision teacher 임베더와 거의 비슷한 성능(대략 0.35점/약간의 격차)을 보이며 의미 표현 품질을 상당 부분 유지한다. CPU(8 threads) 기준 토큰 처리량은 Qwen3-0.6B에서 2배가량(364.36→830.50 tokens/s), Gemma3-270M에서도 약 2배(1181.28→2055.47 tokens/s)로 개선되어 지연·비용 관점의 실용성이 확인된다. 더불어 multi-precision training을 통해 8/4-bit는 16-bit 대비 거의 손실이 없고, 1/2-bit에서도 여전히 유의미한 성능을 유지하면서 저장 비용을 크게 줄여 배포 시 트레이드오프를 유연하게 선택할 수 있다.



### Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization (https://arxiv.org/abs/2606.25656)
Comments:
          Accepted to ACL 2026 GEM Workshop

- **Prior Approaches**: 기존 RAG은 주로 비정형 문서에서 벡터 검색 후 LLM이 답을 생성하는 방식으로 강점을 보여왔지만, 실제 데이터는 텍스트와 관계(그래프)가 함께 있는 반구조화 지식베이스가 많다. GraphRAG, Modular RAG, Agentic RAG 같은 변형들이 나왔지만, “더 복잡한 구조가 정말로 잘 먹히는가”와 “표준 평가가 그 차이를 공정하게 반영하는가”는 불명확했다.

- **Core Contribution**: 이 논문은 반구조화 지식베이스(텍스트+KG)에서 regular RAG, GraphRAG, Modular RAG, Agentic RAG를 체계적으로 평가·비교하는 프레임워크를 제시한다. 또한 실제 사용 시나리오를 반영해 총 9개의 표준 RAG 시나리오(문서 단독부터 text-graph 통합, 도메인 KG 연계, 에이전트의 multi-step planning까지)를 구현하고, precision medicine 도메인 STaRK-Prime에서 생산형 의사결정에 필요한 인사이트를 도출한다.

- **Technical Challenges**: GraphRAG/Agentic RAG는 그래프 컨텍스트와 에이전트 세션 히스토리가 커지면서 context/memory overflow 및 토큰·비용 문제가 쉽게 발생한다. 논문은 (1) 관계를 묶는 compact한 그래프 표현, (2) 에이전트 세션 내 그래프 deduplication, (3) 문서 chunk/content hash 기반 deduplication, (4) ReAct의 단순 루프를 배치형 retrieval로 바꾸는 컨텍스트 절감 루프 설계를 통해 token 사용을 19%-53%까지 줄이면서도 성능을 유지/개선한다.

- **Empirical Impact**: 실험 결과, 그래프만 쓰는 순수 GraphRAG(시나리오 3)는 성능이 매우 낮았지만, 사전 정의된 KG를 텍스트와 적절히 결합한 하이브리드(시나리오 5)가 전반적으로 우수했다. 반면, retrieval 순위 중심 지표가 개선을 과대평가할 수 있음을 보여주며, LLM이 생성 과정에서 실제로 선택하는 entity ID를 평가하는 generation-aware 설정에서 “retrieval-generation gap(확장 검색이 생성 품질을 비례 향상시키지 않음)”이 관찰됐다. 결론적으로, 고급 RAG를 무조건 도입하기보다 데이터의 구조/제약과 컨텍스트 비용 한계를 기준으로 아키텍처를 선택해야 한다는 데이터 기반 가이드라인을 제공한다.



### Three Buddhist Vocabularies: Computational Stylometry of the English Pali Canon across Sutta, Vinaya, and Abhidhamma (https://arxiv.org/abs/2606.25372)
Comments:
          16 pages, 7 figures, 3 tables. code available at this https URL

- **Prior Approaches**: 기존 연구는 주로 Sutta Pitaka(경장)에 한정된 계산적 문체분석에 머물렀고, 다른 Pitaka나 전통 간 비교로 확장할 때는 비교 기준과 코퍼스 범위가 달라지는 문제가 컸다. 또 Zipf 법칙 같은 통계 지표는 사용되더라도 어휘 다양성, 수치/개수 표현 밀도, 어휘 겹침 같은 다각도 검증이 충분히 결합되지 않는 경우가 많았다. 본 논문은 이러한 부분을 Tipitaka 전 범위와 여러 영어 번역(및 전통)으로 넓히는 방향이다.

- **Core Contribution**: 본 논문은 Tipitaka 전체 3종(Pitaka three) 을 영어 번역 코퍼스로 구성해, Sutta Pitaka 중심의 선행 연구를 Vinaya 및 Abhidhammattha Sangaha, 그리고 타 전통 비나야까지 포함해 확장한 계산적 문체 분석을 제시한다. Zipf rank-frequency 분포, MATTR-500 어휘 다양성, numeral-word density(수·숫자 관련 단어 밀도), 어휘 겹침 지표를 함께 계산해 문체적 차이를 구조적으로 비교한다. 또한 특정 구절(예: ‘consciousness’가 특정 순위에서 문법 입자를 대체)처럼 지표가 가리키는 진단 단어의 방향성도 보고한다.

- **Technical Challenges**: Tipitaka 전체를 아우르는 대규모 다원 코퍼스에서 번역본 간 차이, 코퍼스 크기 차이, 전통별 용례 차이가 통계 지표를 왜곡할 수 있다. 논문은 Zipf 분포에 대해 OLS-fitted exponent를 사용하고, MATTR-500은 고정 윈도(500)를 적용해 비교 가능성을 높였으며, 크기 통제된 subsampling으로 다양성 차이를 확인한다. 아울러 Jaccard 및 Szymkiewicz-Simpson 같은 겹침 계수로 전통 간 법적 어휘 유사성을 정량화해, 단순 빈도 비교의 한계를 보완했다.

- **Empirical Impact**: 실험 결과 모든 코퍼스가 Zipf-consistent한 분포를 보였고(R2>0.989), Vinaya가 이상적인 Zipf 기울기(-1)에 가장 가깝게 나타났다. 어휘 다양성은 Sutta와 Vinaya Theravada가 거의 동일한 수준(각각 0.399, 0.400)인 반면 Sangaha는 더 높았고(0.560), Sangaha는 numeral-word density도 가장 높아(3.26%) 체계적 열거 관행과 일치하는 정황을 제공한다. 전통 간 비교에서는 Mulasarvastivada 비나야가 Theravada 비나야와 상당한 어휘 겹침을 보여(자카드 20.0%, overlap coefficient 49.1%) 공유된 법적 유산을 시사한다. 한편 같은 비나야의 영어 번역본 간 어휘 공유는 낮게(88년 차 번역에서 24.2%) 관찰되어 번역/해석 선택이 문체 지표에 미치는 영향을 보여주며, 관련 코드와 데이터는 Darshana Graph 코퍼스의 오픈소스 확장으로 공개된다.



### Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents (https://arxiv.org/abs/2606.25361)
- **Prior Approaches**: RAG 기반 대화 시스템의 기존 연구는 주로 메모리를 어떻게 저장·검색하는지에 초점을 맞췄습니다. 하지만 서로 다른 기능 역할을 가진 메모리가 응답 품질에 어떤 영향을 주는지, 대화 맥락이 달라질 때 에이전트의 응답 방식이 얼마나 달라지는지는 상대적으로 덜 알려져 있습니다. 또한 기존 평가는 주로 reference 기반이라 사용자가 선호하는 뉘앙스를 다른 방식으로 반영할 가능성을 충분히 포착하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 대화 메모리를 역할 중심으로 세분화한 fine-grained taxonomy를 제시하고, 검색된 메모리를 서로 다른 role types로 분류합니다. 더불어 사용자 관점을 모사하는 user-centric 평가 프레임워크를 설계해 응답 차이를 더 정교하게 측정합니다. 그 결과, 메모리 유형별로 응답의 정확도·개인화·제약 인식 같은 특성이 다르게 형성된다는 점을 체계적으로 분석합니다.

- **Technical Challenges**: 핵심 과제는 “메모리의 내용”이 아니라 “기능적 역할”이 응답에 미치는 효과를 분리해 관찰하는 것입니다. 이를 위해 메모리 역할을 세밀하게 분류하고, 서로 다른 대화 맥락에서 retrieval된 메모리가 응답 특성(주제 일관성, 제약 인식, 개인화)에 미치는 영향을 비교하도록 실험 설계를 구성했습니다. 또한 reference 기반 평가의 부족을 보완하려고 사용자 선호를 반영하는 측정 프레임워크를 함께 도입했습니다.

- **Empirical Impact**: long-term 데이터셋과 frontier LLM을 대상으로 비교 실험을 수행한 결과, clarifying memory는 사실 정확성과 constraint awareness를 높여 더 정확하고 개인화된 응답을 유도했습니다. 반대로 irrelevant memory는 topic relevance를 떨어뜨리고 constraint awareness도 저하시켜 응답 품질을 악화시키는 경향이 관찰됐습니다. 이 연구는 메모리 유형을 선택적으로 활용해 개인화된 응답을 만들 수 있음을 실증적으로 보여주며, 메모리 역할 설계/평가 연구에 직접적인 방향성을 제시합니다.



### Data-Driven Evolution of Library and Information Science Research Methods (1990-2022): A Perspective Based on Fine-grained Method Entities (https://arxiv.org/abs/2606.25320)
- **Prior Approaches**: 1990년대 이후 빅데이터와 정보기술의 확산으로 LIS(도서관·정보학) 연구가 데이터 중심으로 이동했지만, 방법론 진화가 어떤 구성요소에 의해 촉진되는지는 정밀하게 정리되지 못했다. 기존 연구는 전체 경향을 요약하는 수준에 그치거나, 방법을 구성하는 알고리즘·데이터·도구·평가지표를 함께 추적하기 어려웠다.

- **Core Contribution**: 이 논문은 1990~2022년 LIS 학술논문을 대상으로 데이터 기반 방법의 핵심 구성요소를 자동 추출해, 연구 방법의 세부 진화 경로를 세 차원(시기별 특성, 주제별 변화, 방법론별 교차 양상)으로 분석한다. 특히 알고리즘·모델, 데이터 자원, 소프트웨어·도구, 메트릭(평가지표)이라는 네 범주의 ‘방법 엔티티’를 분류해 방법론 변화를 설명 가능한 형태로 제시한다.

- **Technical Challenges**: 핵심 과제는 방대한 논문에서 방법 관련 실체를 신뢰성 있게 자동 추출하고, 시간·주제·연구방법이라는 여러 축에서 일관된 비교가 가능하도록 하는 것이다. 저자들은 네 범주의 엔티티를 추출한 뒤, 엔티티의 시간적 변화와 주제 간 이동, 연구방법 간 상호 진화 패턴을 함께 관찰하는 분석 프레임을 구성해 이를 해결한다.

- **Empirical Impact**: 결과적으로 LIS에서 방법론 진화의 ‘주요 동력’은 데이터 자원(data resources)인 것으로 드러나며, 연구 방법 전개가 ‘등장→안정화/실무 적용’의 순환 패턴을 보인다고 보고한다. 이는 LIS 연구를 단순한 데이터 활용이 아닌, 데이터 자원의 성숙과 활용 주기에 맞춰 방법론이 재편된다는 관점을 제공한다는 점에서 의미가 있다.



### Measuring Research Difficulty of Academic Papers: A Case Study in Natural Language Processing (https://arxiv.org/abs/2606.25307)
- **Prior Approaches**: 기존 연구들은 연구 주제 선택이나 성과 분석에서 연구 난이도를 다루더라도, 이를 정량적으로 평가하는 프레임이 부족했다. 또한 연구 난이도와 학술적 영향(impact)의 상관을 수치로 연결한 접근이 제한적이었다.

- **Core Contribution**: 이 논문은 논문 단위로 연구 난이도를 종합 평가하는 시스템을 제안한다. 학술 협업, 내용, 참고문헌 등 내·외부 특징을 추출해 여러 난이도 지표를 만들고, entropy weight로 가중치를 산정해 연구 난이도 점수를 산출한다.

- **Technical Challenges**: 핵심은 난이도를 구성하는 요소들을 어떻게 측정하고 결합할지, 그리고 가중치를 어떻게 정할지였다. 논문은 엔트로피 가중치로 지표 간 중요도를 데이터 기반으로 조정하고, NLP 분야 논문들을 대상으로 난이도 점수의 신뢰성을 확보하기 위해 전문가 평가와 상관 분석을 수행했다.

- **Empirical Impact**: 검증 결과 citation frequency를 학술적 영향으로 두었을 때, 페이지 수, 참고문헌 수, 상위 기관의 참여 같은 요인이 영향과 유의미하게 연관됐다. 또한 연구 난이도와 영향 사이에 역U자(inverted U-shaped) 관계가 관찰되어, 중간 수준의 난이도가 더 큰 impact로 이어질 가능성을 시사한다.



### Automatic Generation of Highlights for Academic Paper Via Prompt-based Learning (https://arxiv.org/abs/2606.25253)
- **Prior Approaches**: 기존 연구는 지도학습 기반으로 자동 하이라이트(논문 하이라이트) 추출을 시도했지만, 보통 대규모 라벨 학습 데이터가 필요해 비용과 확장성에 제약이 있다. 또한 하이라이트가 없는 저널이 많아 문헌 검색·텍스트 마이닝·서지분석에 활용을 제한한다는 문제의식이 있었다. 결과적으로 도메인별 학습 코퍼스 의존도가 커지는 경향이 나타났다.

- **Core Contribution**: 이 논문은 학습 데이터 라벨 없이 prompt-based learning으로 자동 하이라이트 생성을 수행하는 접근을 제안한다. 논문 초록을 입력으로 하고, 작업(task)-특화 prompt template를 설계해 생성형 언어모델에 하이라이트 생성을 맡긴다. 특히 task-specific training samples 없이도 기존 지도학습 방법과 비슷한 성능을 달성하고, 소수 예시를 프롬프트에 추가하면 성능이 더 크게 향상된다는 점을 보여준다.

- **Technical Challenges**: 핵심 기술적 난제는 하이라이트 생성이 프롬프트에 포함된 정보에 크게 의존하는 작업이라는 점이다. 저자 작성 하이라이트에 가깝고도 간결한 요지를 뽑아야 하므로, prompt template가 제공해야 할 조건과 입력 구성이 성능을 좌우한다. 연구진은 작업별 템플릿 설계와 여러 언어모델(GPT-2, T5, API 기반 ChatGPT)을 비교해 프롬프트 설계가 생성 품질에 미치는 민감도를 분석하고 보완 방향을 제시한다.

- **Empirical Impact**: 3개 데이터셋 실험에서 ChatGPT + prompt template는 task-specific 학습 없이도 선행 지도학습과 유사한 성능을 보였다. 또한 두 데이터셋에서는 프롬프트에 소수 예시를 추가했을 때 state-of-the-art 대비 큰 개선을 기록했다. 생성 결과는 대체로 문장 일관성·정보성·저자 하이라이트와의 근접성이 확인되며, 도메인 특화 학습 코퍼스에 의존하지 않는다는 점에서 텍스트 마이닝과 bibliometric research의 실용성이 높다는 의미가 있다.



### The Hitchhiker's Guide to Agentic AI: From Foundations to Systems (https://arxiv.org/abs/2606.24937)
- **Prior Approaches**: 기존에는 에이전틱 AI를 LLM 성능이나 단일 구성요소 중심으로 접근하는 경우가 많아, 파이프라인 전반을 아우르는 관점이 부족했습니다. LLM 계층(학습·추론 최적화)과 정렬·추론 계층(RLHF 계열), 에이전트 계층(메모리·툴 사용·조정)을 각각 따로 다루다 보니 프로덕션 적용에서 병목이 드러난다는 한계가 있었습니다. 또한 멀티에이전트 협업이나 평가/배포 관점이 체계적으로 정리되지 않은 점도 실무자에게 장벽으로 작용했습니다.

- **Core Contribution**: 이 자료는 ‘에이전틱 시스템을 잘 만들려면 파이프라인의 모든 레이어를 이해해야 한다’는 중심 논지를 바탕으로, 처음부터 끝까지의 전 스택을 한 흐름으로 정리합니다. LLM 기초(Transformer, GPU 시스템, SFT/LoRA/MoE, 압축, 추론 최적화)부터 정렬·추론(RLHF, PPO, DPO, GRPO, reward modeling, reasoning용 RL) 그리고 에이전트 설계(RAG, 메모리, 컨텍스트 관리, 에이전트 패턴)까지 연결해 실무 의사결정을 돕는 것이 핵심입니다. 특히 에이전트 간 조정(MCP, A2A, 중앙/분산/계층형 구조)과 UI·평가·배포까지 포함해 “구축” 관점의 완성도를 높였습니다.

- **Technical Challenges**: 에이전틱 AI는 모델 품질만으로는 끝나지 않고, 학습·추론 효율과 정렬 신뢰성, 그리고 실행 중 컨텍스트 관리가 동시에 맞물려야 한다는 기술적 난도가 큽니다. 이 자료는 LLM 학습/미세조정(SFT, LoRA, MoE)과 추론 최적화, 정렬 방법(RLHF/PPO, DPO 및 변형, GRPO) 및 reward modeling을 기초로 두고, 에이전트 계층에서는 trajectory 기반 학습, Agentic RAG, 다양한 메모리( in-context, external, episodic, semantic )로 문제를 분해해 접근합니다. 더불어 에이전트 툴 사용·스킬 설계, 프로토콜(MCP, A2A)과 멀티에이전트 토폴로지로 상호작용을 구조화하고, 구현 예시와 코드 중심 가이드로 연결성을 제공합니다.

- **Empirical Impact**: 단순 개념 소개가 아니라 각 장에 이론 근거와 구현 가이드를 함께 제공해, 실무자가 에이전트 시스템을 단계적으로 검증하고 배포로 옮기는 데 도움이 됩니다. 특히 평가 방법론과 프로덕션 배포까지 포함해, 실험 성능에서 실제 태스크 성능으로의 전이를 촉진하는 점이 의미 있습니다. 결과적으로 에이전틱 AI를 ‘전략’이 아니라 ‘엔지니어링’으로 다루려는 업계/연구 커뮤니티의 학습 경로를 정리해주는 역할을 합니다.



### Error-Aware TF-IDF Retrieval-Augmented Generation for ASR Error Correction (https://arxiv.org/abs/2606.24915)
Comments:
          4 pages, 1 figure, 2 tables

- **Prior Approaches**: 기존 ASR-RAG 교정은 희귀 개체/도메인 용어에서 LLM이 환각을 일으키는 문제를 완화하려고 시도돼 왔습니다. 하지만 표준 TF-IDF 기반 검색은 음운 오인(phonetic misrecognition)을 일으킨 토큰을 동일하게 취급해 교정 근거를 충분히 끌어오지 못하고, cross-modal 임베딩 등 복잡한 검색은 높은 지연(latency)과 계산 부담이 큽니다. 또한 엔터티 벡터베이스·Knowledge Graph처럼 무거운 구성은 정렬된 대규모 데이터·고자원 가정이 필요하다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 저자원 언어 페르시아어에서 LLM 기반 ASR 교정 시의 phonetic hallucination과 loop hallucination을 “오류 인식형(lexical error-aware)”으로 직접 겨냥하는 효율적 RAG 프레임워크를 제안합니다. 핵심은 Symmetric Text Normalization으로 KB와 질의 전처리를 동일하게 맞추고, Error-Aware TF-IDF로 과거에 자주 환각되는 토큰에 더 큰 가중치를 부여해 LLM이 교정해야 할 근거 문서를 우선적으로 검색하게 만드는 것입니다. N-best reranking이나 음향 latent/지식그래프에 대한 과도한 결합 없이, 1-best ASR 가설과 희소 행렬 연산만으로 동작하도록 설계됐습니다.

- **Technical Challenges**: 기여를 구현하는 가장 큰 기술적 난제는 (1) 형태/표기 차이로 인한 토큰 미스매치가 검색 공간을 왜곡하고, (2) 루프 환각이 TF-IDF의 term frequency를 망가뜨리며, (3) 표준 TF-IDF처럼 모든 토큰을 동일 가중으로 두면 “오인 토큰이 포함된 교정 문맥”이 점수에서 밀린다는 점입니다. 저자는 ZWNJ·공백·숫자 표기 등을 KB와 질의에 대칭 적용해 벡터공간 일관성을 확보하고, 동일 토큰이 2회 초과 반복되는 구간은 루프 환각으로 보고 잘라내 term frequency 스큐를 줄였습니다. 이어 Error propensity를 기반으로 희소 대각 penalty matrix를 만들어 TF-IDF 행렬에 곱하는 방식으로 계산 지연을 거의 추가하지 않으면서도 오류 토큰을 의도적으로 끌어올렸습니다.

- **Empirical Impact**: FLEURS 페르시아 subset에서 Whisper large-v3-turbo와 Gemini 2.0 Flash-Lite를 사용해 평가했으며, Error-Aware Hit Rate(EA-HR)가 표준 TF-IDF 53.7%에서 제안 방법 90.9%로 크게 상승했습니다. 종단(end-to-end) RAG-ASR 교정에서는 기준 ASR WER 23.06%에서 표준 TF-IDF 21.95%를 거쳐 제안 Error-Aware TF-IDF가 18.83%로 추가 개선을 보였습니다. 특히 해당 성능 향상을 위해 별도의 복잡한 cross-modal 임베딩을 쓰지 않고 희소 행렬 곱 수준의 근접-제로 inference latency만 요구해, 저자원 환경에서도 실용적인 정확도/지연 균형을 제시했다는 점에서 의미가 큽니다.



### Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs (https://arxiv.org/abs/2606.24897)
Comments:
          14 pages, 1 figure. Pre-registered on OSF. Data and code available on Zenodo and GitHub

- **Prior Approaches**: 생의학 텍스트 마이닝, 과학계량학, 그리고 생의학 LLM 학습 코퍼스 구축은 서지 API가 반환하는 초록이 실제 출판 초록을 그대로 재현한다고 가정해왔다. 그러나 그 재현 품질이 문자 수준에서 어떻게 달라지는지, 그리고 어느 요소가 어떤 API에서 왜 깨지는지는 체계적으로 감사되지 않았다.

- **Core Contribution**: 이 논문은 OSF에 사전 등록된 감사(pre-registered audit)로, PubMed E-utilities, Crossref, OpenAlex, Semantic Scholar의 초록 텍스트를 PMC JATS XML을 공통 기준(ground truth)으로 비교한다. 2024년 PMC 오픈액세스 약 70만 편 중 영어 연구 논문 4,000편을 샘플링해, JATS 초록에 포함된 특정 유니코드 문자 범주가 각 API에서 보존되는지 문자 충실도를 측정했다.

- **Technical Challenges**: 핵심 난제는 API마다 초록 제공 방식이 다를 수 있어, ‘내용’이 아니라 ‘표면 문자 서명(character-level signature)’의 손실을 재현 가능하게 계측해야 한다는 점이다. 연구진은 유니코드 문자 클래스(타이포그래피 구두점, 수학/과학 기호, 그리스 문자, 특수 공백)를 사전 정의하고 블라인드 메커니즘 감사로 손실 원인을 문자 치환과 inverted-index 직렬화로 귀결해 설명했다.

- **Empirical Impact**: 결과적으로 두 가지 결정적(deterministic) 손실이 임계 기준을 충족할 정도로 관측됐는데, PubMed AbstractText는 타이포그래피 구두점 보존이 0.6% 수준에 그쳤고 OpenAlex는 특수 공백이 0%로 나타났다. 반면 수학 기호와 그리스 문자는 네 API 모두 95% 이상 보존됐고, Crossref는 논문 24.6%에서 초록을 아예 반환하지 않아 커버리지(75.4%, 95% CI 74.1-76.7%) 격차도 확인됐다. 따라서 같은 PMC JATS 텍스트라도 API에 따라 토큰화 민감 지표와 문자 기반 글쓰기/코퍼스 품질에 직접적인 영향이 생기며, API별 문서화되지 않은 문자 수준 불일치를 고려한 코퍼스 설계가 필요하다는 점을 강하게 시사한다.



### ChartWalker: Benchmarking the Cross-Chart RAG Task with Hierarchical Knowledge Graphs (https://arxiv.org/abs/2606.23997)
- **Prior Approaches**: 기존 cross-chart RAG 평가는 대부분 표 중심이거나, 차트에서 핵심만 뽑아 의미 유사성에 기대어 질문을 만들어 논리 사슬이 쉽게 깨졌습니다. 특히 이런 방식은 쿼리-증거 간 단어 겹침(lexical overlap)을 유발해 보이기 좋은 답은 내도, 다중 차트 간 추론 일관성은 보장하기 어렵습니다. 또한 knowledge graph 기반 생성에서도 무작위/단순 경로 샘플링은 질적으로 일관된 다중 홉 추론 경로를 충분히 통제하지 못했습니다.

- **Core Contribution**: 이 논문은 차트 중심의 challenging cross-chart RAG 태스크를 만들기 위한 ChartWalker 프레임워크를 제안합니다. ChartWalker는 차트로부터 엔티티-관계를 계층적으로 구성하는 hierarchical knowledge graph와, 의미적 토픽 연속성과 정보 granularity를 동시에 제어하는 structure-aware sampling을 결합해 논리적으로 정합한 multi-hop QA를 생성합니다. 그 결과로 ChartWalker-Bench(총 564개 QA, 4개 쿼리 타입)를 공개하고, 추가로 성능 진단용 에이전트 베이스라인 ChartWalker-Agent도 제공합니다.

- **Technical Challenges**: 핵심 난제는 차트가 정보 밀도는 높지만 약하게 구조화되어 있어, 다중 차트 증거를 “어떤 수준의 관점(제목/범주/개별 값 등)”에서 어떻게 이어야 하는지 일관되게 만들기 어렵다는 점입니다. 논문은 차트 구성요소의 granularity 레벨을 VLM 기반으로 태깅해 계층 KG를 만들고, 다음-hop 선택 시 의미 일관성(cosine similarity)과 레벨 전이를 제약해 semantic drift를 줄이도록 설계했습니다. 또한 생성된 QA는 증거 기반 검증(지원성+의미성)과 재샘플링으로 품질을 0.77 통과율에서 564개로 최종 확정했습니다.

- **Empirical Impact**: 주요 RAG 패러다임 전반에서 성능 격차가 크게 나타났고, 최상위 모델도 cross-chart 정답률은 약 64% 수준에 머물렀습니다. 특히 Complex Reasoning 같은 다단계 추론 쿼리에서는 정확도가 30% 미만으로 급락해, 검색-통합-추론의 장기성/다중 granularity 요구가 현재 모델에 얼마나 큰 벽인지 보여줍니다. 이 벤치마크는 retrieval과 generation 모두를 동시에 압박하는 평가 도구로서, 향후 multimodal RAG 및 에이전트형 설계 방향을 구체화할 의미가 큽니다.



New uploads on arXiv(cs.CV)

### TryOnCrafter: Unleashing Camera Trajectories for Realistic Video Virtual Try-on via a Renderable 4D Try-on Proxy (https://arxiv.org/abs/2606.26092)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 Video Virtual Try-on(VVT)은 소스 비디오의 카메라 궤적에 사실상 고정돼, 사용자가 임의 각도에서 의상을 탐색하려는 요구를 충족하기 어렵다. 이미지 워핑/프레임 단위 priors나 DiT 기반 생성이 발전했지만, 비고정 시점에서는 기하 정합과 시간 일관성이 쉽게 무너진다.

- **Core Contribution**: 이 논문은 Camera-controllable Video Virtual Try-on(CaM-VVT)이라는 새로운 과제를 제안하며, 임의 카메라 움직임 하에서도 의상 디테일을 유지하면서 배경과의 엄격한 구조 동기화를 요구한다. 이를 위해 TryOnCrafter를 소개하고, 기존의 “try-on → V2V camera control” 순차 파이프라인보다 더 견고한 단일 DiT 기반 end-to-end 프레임을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 비관측 시점에서의 텍스처 환각을 view-consistent하게 만들고, (2) 비강체인 인체 동작과 배경 맥락을 임의 카메라에서도 기하적으로 동기화하며, (3) 두 모델을 연쇄하면 발생하는 오차 누적과 계산 부담을 동시에 줄이는 것이다. 저자들은 Renderable 4D Try-on Proxy로 인체를 환경에서 분리(3DGS 기반 clothed avatar + SMPL-X 모션 정합 + metric-aligned 배경 point cloud)하고, Proxy-Anchored Video DiT가 렌더드 프라이어(픽셀 정렬된 구조 제약)를 anchor로 삼아 물리적으로 그럴듯한 변형과 궤적 준수를 학습하도록 설계했다.

- **Empirical Impact**: ViViD 벤치마크 기반 정량/정성 결과에서 TryOnCrafter는 paired·unpaired 모두에서 구조 일관성과 의상 동일성을 크게 개선하며 동종 SOTA 대비 새 성능을 보였다. CaM-VVTBench에서는 VBench 계열 지표로 평가해 CaM-VVT용 통합 프레임의 효과를 확인했고, 특히 4D 프록시가 만들어 주는 연속적 기하 기반 덕분에 공격적인 회전에서도 실루엣·사지 형태·배경 정합이 더 안정적이라는 점이 강조된다.



### MVTrack4Gen: Multi-View Point Tracking as Geometric Supervision for 4D Video Generation (https://arxiv.org/abs/2606.26087)
Comments:
          Project Page : this https URL

- **Prior Approaches**: 단안(reference) 동영상과 목표 카메라 trajectory를 이용해 novel-view video를 합성하는 연구는 크게 (1) explicit 3D(깊이/포인트/장면 표현) 복원 후 reconstruct-then-generate, (2) 3D 표현 없이 카메라 조건만 주는 camera-conditioning-only로 나뉜다. 전자는 동적 물체에서 오프더셸 복원 오차가 기하를 망치며 boundary 주변 ‘flying-pixel’ 같은 왜곡과 모션 붕괴를 유발한다. 후자는 photorealistic 품질은 좋지만 장면 구조·동역학을 명시적으로 잡지 못해 뷰 간 기하/모션 일관성이 흔들린다.

- **Core Contribution**: MVTrack4Gen은 camera-conditioning-only diffusion 모델에 “multi-view point tracking”을 보조 감독으로 결합해 기하 일관성과 모션 충실도를 동시에 강화한다. 핵심 관찰은 특정 attention layer가 시공간 전반에서 강한 correspondences(질의가 기하적으로 대응되는 키를 향해 attending)를 학습하며, 이 대응이 틀어질 때 motion inconsistency가 발생한다는 점이다. 이를 위해 해당 attention layer의 query-key 특징을 auxiliary multi-view tracking head와 attention correspondence loss에 연결해 jointly train한다.

- **Technical Challenges**: 어려운 점은 explicit 3D를 쓰지 않으면서도 attention 내부에 들어있는 correspondences를 실제 동적 물체의 물리 점 궤적 수준으로 정렬시키는 것이다. 논문은 (1) attention에서 뽑은 query-key 유사도로 multi-scale local 4D correlation volume을 만들고 (2) transformer 기반 multi-view tracking head가 잔차(residual)를 예측하도록 하며 (3) attention weight 행렬에 cross-entropy 형태의 multi-view correspondence loss를 직접 걸어 query가 정답 위치로 attend하도록 유도한다.

- **Empirical Impact**: DAVIS와 iPhone 같은 벤치마크에서 두 camera-conditioning-only 백본(ReCamMaster, Redirector)에 모두 적용해 geometric consistency 및 camera accuracy를 개선하며, 동적 물체에서 특히 성능 향상이 두드러진다. 또한 추론 시점에 explicit 3D reconstruction 없이도 dynamic 객체의 기하·시각 품질을 동시에 끌어올려 기존 reconstruct-then-generate의 의존성을 줄이는 의미가 있다. 결과적으로 다중 뷰 attention의 correspondences를 motion-aware하게 강화하는 접근이 state-of-the-art급 기하 일관성과 competitive camera 제어를 달성했다고 보고한다.



### A cross-process welding penetration status prediction algorithm based on unsupervised domain adaptation in laser and TIG welding (https://arxiv.org/abs/2606.26078)
- **Prior Approaches**: 기존 용접 관입 상태 분류에는 지도 학습 기반 딥러닝이 널리 쓰이지만, 용접 공정이 바뀌는 domain shift(예: TIG의 arc-dominated → 레이저의 keyhole 기반)에서 성능이 크게 떨어진다. 공정별 물리 메커니즘 차이 때문에, 다른 공정으로 모델을 그대로 옮기면 기준선보다 정확도가 급락한다. 일부는 재라벨링과 fine-tuning으로 대응하지만, 새 공정을 추가할 때 라벨 비용이 커진다는 한계가 있다.

- **Core Contribution**: 이 논문은 라벨 없이도 동작하는 unsupervised domain adaptation(UDA) 프레임워크를 제안하고, 점진적 source domain 확장(GSDE) 전략을 통합해 공정 간 불일치를 완화한다. 핵심은 TIG와 레이저처럼 서로 다른 용접 시스템에서도 domain-invariant한 특징을 학습하면서도 class 경계를 유지하도록 유도하는 것이다. 이를 통해 신규 용접 공정에 대한 relabeling 비용을 낮추고 지능형 용접 모니터링의 범용성을 높인다.

- **Technical Challenges**: UDA는 target 도메인 라벨이 없어서 공정 차이를 “정확한 대응관계”로 정렬하기가 어렵고, 잘못 정렬되면 클래스 경계가 흐려질 수 있다. 저자들은 GSDE로 학습 중에 source 도메인을 점진적으로 확장해 분포 불일치를 단계적으로 줄이고, UMAP 시각화 결과 domain-invariant 특징을 학습하면서도 구분력을 보존하도록 설계했다고 설명한다. 즉, 도메인 정렬과 분류 경계 유지라는 두 목표를 동시에 만족시키는 절충을 목표로 한다.

- **Empirical Impact**: 전용 TIG 및 레이저 데이터셋에서 same-process 전이에서는 TIGFH 90.65%, LSPS 90.72%의 평균 정확도를 기록하며, 지도 기준선 대비 각각 35.83%, 38.87% 향상됐다. cross-process에서는 TIG→Laser 80.48%, Laser→TIG 81.13%로, 기준선 대비 각각 43.39%, 43.40% 개선을 보였다. UMAP 시각화는 학습된 표현이 도메인 불변성을 갖고도 클래스 분리가 유지됨을 뒷받침하며, 공정 전환 시 라벨링 부담을 크게 줄일 수 있음을 시사한다.



### A welding penetration prediction model for laser welding process based on self-supervised learning using physics-informed neural networks (https://arxiv.org/abs/2606.26059)
- **Prior Approaches**: 레이저 용접의 full-penetration(완전 관통) 상태는 결함 없는 접합을 좌우하므로, 이를 정확히 예측하는 분류 연구가 중요해졌다. 기존 supervised learning 기반 분류는 높은 성능을 내지만, 산업 현장에서 충분히 크고 고품질인 라벨 데이터를 대량으로 수집·정제해야 한다는 한계가 있다.

- **Core Contribution**: 이 논문은 제한된 소수 라벨 이미지로도 관통 상태를 고정밀로 분류하는 SimPhysNet을 제안한다. 핵심은 물리적 사전지식(physical priors)을 contrastive learning에 내재화하는 self-supervised 학습 패러다임이며, molten pool과 keyhole에 대해 물리적으로 의미 있는 특징을 학습하도록 PINN을 결합한다.

- **Technical Challenges**: 문제는 (1) 라벨이 적을 때도 물리적으로 타당한 표현을 뽑아내고, (2) 서로 다른 촬영/조건에서 일반화하도록 만드는 것이다. 저자들은 대량의 비라벨 데이터에 대해 PINN으로 물리 제약을 주입하고, 세 가지 image augmentation 과제를 더해 일반화를 강화한 뒤, prototypical networks 기반 few-shot 학습으로 최소 라벨로 클래스 표현을 구성해 강건한 분류를 수행한다.

- **Empirical Impact**: 실험에서 SimPhysNet은 200장(전체 라벨의 약 5%)만 사용해 96.06%의 분류 정확도를 달성했으며, 전체 라벨을 쓰는 기존 supervised learning과 유사한 성능을 보인다. 라벨 의존성을 크게 낮추는 동시에 정밀도를 유지한다는 점에서 레이저 용접 자동화와 공정 지능화에 실용적 의미가 크다.



### DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation (https://arxiv.org/abs/2606.26058)
Comments:
          19 pages, 9 figures

- **Prior Approaches**: 오픈 도메인 subject-driven text-to-video(S2V) 연구는 보통 in-domain에서 기준 피사체(정체성·속성) 보존을 극대화하는 데 집중해 왔습니다. 그 결과 cross-domain(새 스타일/도메인 속성/의미 조합 등)로 편집·전환할 때는 편집 가능성과 적응성이 떨어질 수 있다는 한계가 제기돼 왔습니다.
또한 기존 방식은 기준 이미지의 특징을 비디오 잠재공간에 주입할 때 비디오와의 얽힘(entanglement)이 커서, 도메인 속성과 콘텐츠(피사체 고유 속성)가 함께 흔들리는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 이상적인 S2V가 in-domain과 cross-domain 모두에서 “피사체 충실도”와 “생성 유연성”을 동시에 달성해야 한다고 보고, 그 목표에 맞춘 프레임워크 DomainShuttle을 제안합니다. 핵심은 비디오와 reference 이미지를 분리 처리해 피사체 일관성은 유지하면서 도메인 속성은 텍스트 지시에 따라 유연하게 바꾸도록 설계한 것입니다.
DomainShuttle은 Domain-MoT, Video-Reference DualRoPE, Cross-Pair Consistent Loss( CCL )의 3요소로 구성됩니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 피사체의 고유 속성(intrinsic subject features)과 도메인 고유 속성(domain-specific attributes)이 잠재공간에서 얽혀, 피사체는 유지하되 스타일·조명·도메인은 바꾸려는 목표가 충돌한다는 점입니다. 논문은 이를 위해 DomainMoT에서 비디오/레퍼런스 처리 경로를 분리하고, reference 가지에 domain-aware AdaLN으로 도메인 정보를 명시 주입하며, 텍스트-시각 cross-attention은 학습 중 동결해 base 모델의 텍스트 추종 능력을 보존합니다.
또한 Video-Reference DualRoPE로 reference 토큰과 video 토큰의 RoPE 공간을 분리해 피사체 단위의 공간 관계를 정밀하게 모델링하고, CCL로 동일 비디오에 대응되는 서로 다른 레퍼런스 세트를 같은 노이즈 단계에서 정렬해 조명/구도/스타일 같은 비관련 변이를 억제합니다.

- **Empirical Impact**: 실험은 Wan2.1-14B-T2V와 Wan2.2-14B-T2V 기반으로 in-domain과 cross-domain을 모두 평가했으며, AES·MS(영상 품질), GMEScore(텍스트 제어), DINO-I/CLIP-I(피사체 일관성) 및 여러 cross-domain 지표를 사용했습니다. 그 결과 DomainShuttle은 다양한 복잡 시나리오에서 피사체 충실도와 텍스트 제어를 함께 개선하며, 특히 cross-domain 성능에서 SOTA 대비 18.7% 향상을 보고합니다.
이는 “reference 유사성 복제”를 넘어 오픈 도메인 편집/개인화에서의 일반화와 창의적 조합 능력을 실증적으로 강화했다는 의미가 있습니다.



### How Robust is OCR-Reasoning? Evaluating OCR-Reasoning Robustness of Vision-Language Models under Visual Perturbations (https://arxiv.org/abs/2606.26041)
- **Prior Approaches**: 기존 VLM·OCR 벤치마크는 주로 깨끗한 이미지에서 OCR 정확도나 추론 성능을 평가해 왔습니다. 반면 ImageNet-C나 VLM-RobustBench처럼 일반적인 시각 잡음/왜곡 평가지표는 텍스트 추출·구조 기반 추론이 빠져 있어 OCR reasoning의 취약 양상을 분리하기 어렵습니다. 최근 OCR 견고성 벤치마크(CC-OCR 등)도 주로 일반 OCR에 초점을 둬 실제 “시각적 손상→기호 오류→추론 증폭” 경로를 체계적으로 진단하지 못했습니다.

- **Core Contribution**: 이 논문은 시각적 perturbation이 OCR reasoning에 미치는 영향을 통제적으로 측정하는 벤치마크 OCR-Robust를 제안합니다. OCR1.0(문서·장면문자·영수증·손글씨·수학 등)과 OCR2.0(차트·기하 도형·테이블) 두 부분으로 구성해, 자연스러운 텍스트 인식부터 구조화된 시각 추론까지 함께 다룹니다. 또한 18개 후보 perturbation에 대한 사전 파일럿 선택을 통해 glass blur, motion blur, elastic deformation, color shift, snow 5종을 대표 조건으로 확정하고, clean accuracy 외에 RCR/WCR/CRI 같은 견고성 지표 묶음을 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “텍스트 인식을 깨뜨리지 않으면서도” 모델 간 견고성 차이를 드러내는 perturbation을 설계·선정하는 것입니다. 이를 위해 LPIPS 기반으로 동일한 지각적 열화 강도를 세 severity로 맞추고, MRD(영향), SEP(분리성), MON(단조성)으로 18종 후보 중 진단력 있는 5종을 고릅니다. 또 robustness 평가를 clean 대비 상대 유지(RCR), 최악 케이스 위험(WCR), 성능·평균·최악의 균형(기하평균 CRI)으로 나눠 “깨끗한 성능만 높고 최악에서 붕괴” 같은 숨은 취약성을 포착하도록 했습니다.

- **Empirical Impact**: 18개 모델(폐쇄형 VLM, 오픈소스 VLM, OCR+LLM 파이프라인)을 zero-shot으로 평가한 결과, clean accuracy가 높다고 robustness가 비례해 강해지지 않는 패턴이 뚜렷했습니다. 특히 구조 의존 입력(차트·테이블)이 문서형 입력보다 훨씬 취약해 평균 WCR이 OCR1.0에서 0.826, OCR2.0에서 0.676으로 더 크게 하락했습니다. OCR+LLM은 OCR 추출 품질에 민감했고, thinking mode나 CoT 프롬프트는 clean 정확도는 개선해도 worst-case 실패를 항상 막지는 못해 OCR reasoning 견고성을 별도 문제로 다뤄야 함을 시사합니다.



### TriViewBench: Controlled Complexity Scaling for Multi-View Structural Reasoning in MLLMs (https://arxiv.org/abs/2606.26029)
Comments:
          26 pages, 8 figures

- **Prior Approaches**: 기존 MLLM 시각추론 벤치마크는 주로 단일 이미지 중심이어서, 시점 간 동일성 매칭이나 가림(occlusion) 보정 같은 다중 시점 통합 요구가 제한적이었다. 또한 실세계 이미지 기반은 객체 수·공간배치·가림 정도가 함께 변해 성능 저하의 원인을 분리해 보기 어려웠다. 다중 뷰로 확장된 일부 벤치마크도 복잡도 축을 명시적으로 파라미터화하지 않아 구조적 실패 유형을 정밀 진단하기 힘들다는 한계가 있었다.

- **Core Contribution**: 이 논문은 합성 3D 장면을 이용해 객체 수와 가림 정도를 독립적으로 제어하는 3-view 시각추론 벤치마크 TriViewBench를 제안한다. 프론트/사이드/탑다운 세 시점을 동시에 주고, 난이도를 4단계와 추론 범주( Local Decision, Object Counting, Global Recovery )로 체계적으로 나눠 구조적 스케일링을 측정한다. 이를 통해 현재 MLLM의 능력 계층과 실패 메커니즘을 구분해 진단할 수 있는 프레임을 제공한다.

- **Technical Challenges**: 가림과 객체 수가 동시에 변하는 실세계 한계를 피하려면, 각 QA 정답을 자동으로 유일하게 도출할 수 있는 정밀한 메타데이터 기반 합성 설계가 필요했다. TriViewBench는 Kubric 기반 절차적 장면 생성과 카메라 파라미터 고정으로, 시점별 가시성/바운딩박스 및 관계 메타데이터를 주석처럼 생성해 정답 모호성을 줄였다. 또한 Object Counting에서 실패 방향(언더카운트 vs 오버카운트)이 단일 시점과 다중 시점에서 반대로 나타남을 error analysis로 기계적으로 분해했으며, CoT는 거의 무효(Δ=-0.16%)임을 함께 확인해 추론 전략이 아닌 교차시점 공간표현 병목을 시사했다.

- **Empirical Impact**: 18개 오픈/클로즈드 MLLM 모두 동일한 능력 계층( Local Decision > Object Counting > Global Recovery )을 예외 없이 보였고, 복잡도 4단계로 갈수록 단조 하락했다. 하락 폭은 Local Decision 12.11%에 비해 Object Counting 59.14%, Global Recovery 80.02%로 급격히 커져 구조 재구성이 가장 취약함을 실증했다. 특히 Object Counting은 단일 시점 가림맹(undercount)과 다중 시점 동일성 혼동(overcount)이 독립적 실패 모드로 관측돼, 향후 다중시점 표현 학습·정렬의 과제를 분명히 하는 진단 도구로 의미가 크다.



### MIMFlow: Integrating Masked Image Modeling with Normalizing Flows for End-to-End Image Generation (https://arxiv.org/abs/2606.26016)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 Normalizing Flows(NF)는 데이터와 정규분포를 역변환 가능한 변환으로 매핑해 exact density estimation과 샘플링을 동시에 제공하지만, 엄격한 invertibility 때문에 용량이 저수준 픽셀 디테일에 소모되어 고수준 의미 구조를 충분히 포착하기 어렵습니다. Masked Image Modeling(MIM)은 MAE/SimMIM 계열에서 강력한 표현 학습을 보여줬으나, 생성 파이프라인에 들어갈 때는 토크나이저/모듈로 분리되어 학습이 disjoint하게 이뤄지는 경우가 많았습니다.

- **Core Contribution**: 본 논문은 MIMFlow라는 end-to-end 통합 프레임워크를 제안해, latent semantics 추출과 픽셀 재구성, generative flow 최적화를 동시에 수행합니다. VAE encoder가 masked image로부터 의미 latent를 추정하도록 하고, NF는 저주파 의미 manifold를 모델링하며 decoder가 고주파 합성을 담당하는 방식으로 NF의 capacity bottleneck을 완화합니다.

- **Technical Challenges**: 핵심 기술 과제는 MIM 백본이 만드는 latent가 density estimation에 적합하게 ‘안정적이고 고정된’ 분포로 수렴하는지였습니다. 이를 위해 learnable token bottleneck(고정 길이 K 토큰)을 도입하고, masked token + learnable query를 Transformer로 인코딩해 K개의 query만 latent로 추출한 뒤, SimFlow 스타일의 노이즈 주입으로 NF의 연속 밀도 학습을 돕습니다. 또한 학습 중에는 MIM 기반 masked latent를 유지한 채 decoder fine-tuning에서 adversarial refinement를 수행해 확률 정합성을 보존합니다.

- **Empirical Impact**: ImageNet 256×256에서 MIMFlow-L은 linear probing 정확도 71.3%, FID 2.50을 달성하며, 128 tokens만 사용해 유사 스케일 NF 대비 32.8% 성능 향상을 보였습니다. 또한 10k/확률 샘플 기준으로 ablation에서 masking decoupling과 bottleneck 크기(128이 최적)가 의미 manifold 품질과 생성 성능을 좌우함을 확인했고, self-supervised semantic priors(DINO/CLIP)가 latent의 고수준 구조 학습에 특히 유리하다는 점이 드러났습니다.



### From Sparse and Imperfect 2D Anchors to Consistent 3D Gaussian Street Scenes: Support-Aware Appearanc (https://arxiv.org/abs/2606.26007)
- **Prior Approaches**: 3D Gaussian Splatting(3DGS) 기반 편집은 StyleGaussian, ViP3DE, EditSplat처럼 기준 이미지/비디오 프리오어/다중 뷰 확산 편집 경로로 발전해 왔지만, 문제는 sparse하게 주어진 예시 앵커가 서로 픽셀 단위로 어긋날 수 있다는 점이다. 이때 각 앵커를 독립적으로 RGB에 맞추면 조명 불일치, 경계 이동, 편집 노이즈(도로·식생 질감 등)가 그대로 3D에 전파되어 미관과 뷰 일관성이 동시에 무너진다. 또한 기존 파이프라인은 sparse 앵커의 불완전성과 standard-rasterizer 배포(추론 시 추가 모듈 제거)를 함께 다루지 못한다.

- **Core Contribution**: 논문은 teacher-conditioned appearance baking을 제안한다. 목표는 정적 street-scene 3DGS의 고정된 기하(positions/scales/rotations/opacities)는 그대로 두고, teacher가 제공한 sparse한 외형 조건(예: sunset, blue hour, overcast dusk)을 spherical-harmonic(SH) 외형 계수만으로 SH-only 업데이트 형태로 “베이크”해 재사용 가능한 3D 자산을 만드는 것이다. 추론 단계에서는 teacher 및 보조 학습 모듈을 모두 제거해, 표준 3DGS rasterizer만으로 동일 조건 렌더링이 가능하게 한다.

- **Technical Challenges**: 핵심 기술 난제는 sparse하고 서로 불완전한 teacher 앵커가 뷰별 아티팩트를 포함한다는 점이며, 이를 view-consistent한 3D 목표로 응축하려면 노이즈를 억제하고 구조적으로 타당한 잔차만 공유 primitive에 올려야 한다. 이를 위해 teacher-relative appearance residual distillation(teacher-relative 외형 잔차 증류)을 도입하고, teacher 앵커의 잔차를 frequency 분해·confidence 추정이 가능한 residual 공간으로 정리한 뒤 renderer-space 잔차 매칭으로 직접 신호를 주면서 Gaussian-space에서 support-aware aggregation으로 primitive 배분을 정규화한다. 마지막으로 confidence-gated coarse-to-fine 최적화로 저/중주파는 안정적으로 맞추고, 고주파는 teacher가 지지하는 영역에서만 허용해 unsupported 노이즈 삽입을 억제하며, 학습된 잔차는 고정 형상에 대한 SH 계수로 최종 베이크한다.

- **Empirical Impact**: Waymo street assets와 Tanks and Temples에서 다중 조건 및 앵커 예산에 대한 실험 결과, 제안 방법은 편집 기반 베이스라인 대비 target 정렬·콘텐츠 보존·아티팩트 억제·교차 뷰 일관성의 균형이 전반적으로 가장 좋게 나타났다. 특히 scene-cluster bootstrap으로 절차적 강건성을 평가한 분포에서도 순위가 유지됐고, 다양한 앵커 수(예: 4/8)에서 과소적합·장면 열화가 덜했다. 블라인드 인간 평가에서도 30명 참가자 기준 선호가 78.6%(타이 제외 시 85.0%)로 나타나, 자동 지표(CLIP 기반 이동/일치, CV-ΔΔ 등)와 함께 실제 시각 품질 개선을 뒷받침한다.



### Taxonomy-aware deep learning for hierarchical marine species classification in underwater imagery (https://arxiv.org/abs/2606.25989)
Comments:
          10 pages, 3 figures, 4 tables. Presented at SPIE Defense + Security 2026 (Machine Learning from Challenging Data conference), National Harbor, MD, April 2026

- **Prior Approaches**: 기존 계층 라벨 학습은 트리 구조 제약(Hierarchical Cross-Entropy 등)이나 parent-child 결합/분기형 아키텍처, 또는 leaf 분포를 marginalization으로 재구성하는 방식이 주로 쓰였다. 하지만 센서/환경 차이로 인한 domain shift에서 learned inter-component 의존이 성능을 흔들고, 균일한 cross-entropy는 계층 오차의 “비용”을 반영하지 못한다는 한계가 컸다. 또한 coarse~fine 라벨 불균형(종 수준 미라벨 비율이 큼) 때문에 계층 비용을 학습 목표에 직접 반영하기가 어려웠다.

- **Core Contribution**: 논문은 생물 분류의 계층 구조에 맞춰 학습 loss와 추론 규칙을 함께 정렬하는 taxonomy-aware 프레임워크를 제안한다. 핵심은 taxonomy-weighted loss(트리 거리 기반 패널티)와 minimum-risk Bayesian inference(거리 행렬로 기대 비용 최소화), 그리고 rank별 독립 classification head(7개 수준 동시 학습)다. 이를 통해 “종-종 혼동”과 “문-문(상위 계층) 혼동”을 다르게 취급하는 평가에 직접 최적화한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 계층 거리 기반 평가를 학습에서 미분 가능한 형태로 연결하고, (2) domain shift 상황에서 계층 결합 설계가 과적합/성능 저하를 유발하지 않게 만드는 것이다. 저자들은 종단 distnace 행렬(D)을 미리 계산해 손실의 기대 트리 거리로 반영하고, 추론 시에도 동일한 D로 minimum-risk 결정을 내리며, 분기된 head는 기본값으로 독립화해 shift에서 더 견고하게 만들었다. 추가로 multi-scale feature encoding과 10-fold 확률 앙상블을 결합해 오차를 완충했다.

- **Empirical Impact**: FathomNet 2025(7개 taxonomic rank, 79개 클래스, 80 leaf label)에서 mean taxonomic distance 1.581을 달성했으며 1위(1.535)와는 3% 이내였다. 컴포넌트 기여로는 taxonomy-aware loss와 ensemble averaging이 가장 큰 폭으로 개선했고, minimum-risk inference도 통계적으로 유의했다. 외부 독립 검증 세트에서도 multi-scale encoding, 앙상블, minimum-risk가 재현됐고, 특히 backbone 미세조정 레시피(LLRD)가 self-supervised feature 보존에 중요함을 보여 DINOv2-Base가 ConvNeXtV2-Base 대비 큰 격차(평균 11.3%)를 보였다.



### A Benchmark for Heterogeneous Stereo Deblurring with Physically- and Epipolar-constrained Cross Attention (https://arxiv.org/abs/2606.25962)
- **Prior Approaches**: 기존 단일 이미지 디블러링은 커널 추정부터 CNN/Transformer/activation-free 모델까지 발전했지만, 동기화된 다른 시점의 정보를 활용하도록 설계되진 않았다. 스테레오 복원 계열은 시차/대응을 이용하지만 대부분 균질(homogeneous) 스테레오를 전제로 해 양방향이 유사하게 열화된 상황을 다룬다. 스마트폰처럼 wide/ultra-wide가 서로 다른 하드웨어(안정화/OIS 유무, 조리개 등)로 인해 비대칭 블러가 생기는 문제는 주 복원 목표로 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 스마트폰의 wide(기준)–ultra-wide(타깃) 비대칭 블러 복원을 정면으로 다루는 전용 프레임워크를 제안한다. 이를 위해 실제 이기종 스테레오 캡처를 기반으로 한 HSD(heterogeneous stereo deblurring) 데이터셋을 만들고, wide의 선명도를 ultra-wide 복원에 직접 활용하도록 평가를 설계했다. 또한 physically- and epipolar-constrained cross attention(PECA)을 제안해 epipolar 및 광학 기반 시차 상한으로 크로스 뷰 대응 탐색을 물리적으로 제한한다.

- **Technical Challenges**: 핵심 난제는 ‘rectified 스테레오’ 환경에서 대응 탐색을 정확하고 효율적으로 수행하는 동시에, 가림/부정합 구간에서는 신뢰도에 따라 영향이 자동으로 줄어들게 만드는 것이다. PECA는 카메라 기하와 광학 파라미터로부터 시차의 물리적 상한을 구해 epipolar line 위 1D 윈도우만 attention 후보로 제한하고, confidence-weighted attention과 residual fusion을 통해 신뢰 구간엔 cross-guided deblurring을, 불확실 구간엔 self-deblurring을 자연스럽게 적용한다. 글로벌 dense matching이나 occlusion mask 같은 비용/민감한 요소를 피하면서도 계산량을 크게 줄이도록 구성됐다.

- **Empirical Impact**: HSD에서 PECA를 CNN(XYDeblur), Transformer(Restormer), NAFNet에 모두 결합했을 때 단일 이미지 기준선 대비 복원 성능이 일관되게 향상되며, 채널 증설 같은 단순 용량 증가로는 같은 이득을 얻기 어렵다는 점을 보였다. 특히 PECA는 단순 스테레오 concatenation보다도 PSNR/복원 품질이 더 높고, occlusion이 심한 상황에서도 이득을 유지하는 경향을 보인다. 더 나아가 별도 ground-truth 없이도 실제 핸드헬드 이기종 캡처에서 선명한 에지와 미세 텍스처 복원에 도움이 됨을 정성적으로 확인해 실사용 적용 가능성까지 확장한다.



### Pulmonary Embolism Risk Stratification from CTPA and Medical Records: Vascular Graphs Are Not All You Need (https://arxiv.org/abs/2606.25956)
Comments:
          8 1/2 pages + 2 pages of references. Accepted for MICCAI 2026. This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is published in, and available online at, the external reference provided below

- **Prior Approaches**: 폐색전증(PE) 위험도 분류는 환자 병력, CTPA(전산화폐동맥조영술)에서 추출한 지표, 혈액검사 결과에 기반해 이뤄진다. 하지만 실제 임상에서는 혈액검사가 누락되는 경우가 잦아, 기록과 영상 기반 정보만으로 정확도를 유지하기 어렵다. 기존에는 CTPA의 혈관 정보를 활용하더라도 혈액검사 부재 상황을 직접적으로 보완하는 방식이 제한적이었다.

- **Core Contribution**: 이 논문은 혈액검사 없이도 병력과 CTPA에서 뽑은 심장 바이오마커(혈액검사 대체 성격)를 조합해 PE 위험도를 분류할 수 있는지를 탐색한다. 특히 CTPA의 폐혈관 정보를 더해 탭ular 모델에 혈관 바이오마커를 넣거나, 혈관 트리의 intrinsic graph 표현으로 GNN을 적용하는 비교 벤치마크를 제시한다. 결과적으로 혈관 바이오마커와 혈관 그래프 기반 접근이 위험도 분류 성능을 추가로 끌어올리지 못했다는 점을 핵심 결론으로 다룬다.

- **Technical Challenges**: 핵심 기술 과제는 CTPA에서 유의미한 폐혈관 정보가 실제로 위험도 판별에 기여하는지, 그리고 그것을 모델이 효과적으로 학습할 수 있는 형태로 표현하는 것이다. 연구진은 (1) 병력·심장 바이오마커 중심의 전역(global) 특징을 강한 기준선으로 두고, (2) 혈관 바이오마커를 탭ular 피처에 보강하거나, (3) 혈관 트리 그래프에 GNN을 적용해 성능 향상을 기대했지만, GNN조차 전역 특징의 강한 탭ular baseline을 넘지 못했다. 또한 이 비최적 성능을 설명하기 위해 모델 가정(표현/학습)과 데이터 특성(정보 부재 또는 약한 신호) 양쪽의 가설을 함께 검토한다.

- **Empirical Impact**: 비공개 데이터(n=353)에서 모든 정보가 유의미하게 완전하게 수집된 케이스를 사용해 실험을 수행했으며, 병력과 심장 바이오마커가 전역 특징 중 가장 중요한 예측 요인으로 나타났다. 반대로 혈관 바이오마커는 추가 향상을 주지 않았고, 혈관 그래프 기반 GNN도 성능 개선에 실패했다. 임상적으로는 혈액검사 부재 상황에서 폐혈관 그래프보다 병력·심장 바이오마커 중심이 더 효율적일 수 있음을 시사하며, 학습된 표현이 실제 판별 정보를 담지 못할 가능성에 대한 경고로도 의미가 있다.



### FunPiQ: A New Benchmark for Pixel-Level Quality Assessment in Fundus Images (https://arxiv.org/abs/2606.25915)
Comments:
          Accepted at MICCAI 2026 main conference. Our code, weights, and dataset are available at this https URL

- **Prior Approaches**: 기존 FIQA 연구는 대개 이미지 레벨 라벨을 표준화하려 했지만, 질(quality)의 기준이 질환·진단 과업마다 달라 데이터셋 간 라벨 불일치가 커졌습니다. 또한 image-level supervised 학습은 국소 열화(조명·대비·초점 저하 등)의 위치를 정량화하기 어렵고, Grad-CAM 같은 post-hoc 설명은 정확한 열화 구역을 정밀하게 찾지 못했습니다. EyeQual/EFIQA처럼 local map을 만들려는 시도도 있었지만, pseudo-label 노이즈와 전역 문맥 부재 등으로 자연스러운 해부학적 저밀도 영역(예: 황반)에서 오판 문제가 남아 있습니다.

- **Core Contribution**: 이 논문은 CFP의 픽셀 단위 품질 평가를 위한 최초의 FIQA benchmark인 FunPiQ를 제안하며, 300장에 대해 해부학적 가시성 기준으로 good/usable/bad를 전문가가 픽셀 단위로 주석합니다. 더불어 EFIQA-CP를 제안하는데, 품질 pseudo-label을 해부학적 가시성으로 만들고 Non-Negative Positive-Unlabeled(nnPU) learning으로 학습해 노이즈에 강건하도록 설계했습니다. 핵심은 “과업 의존적 이미지 레벨 기준”을 넘어, 픽셀 가시성 기반의 task-agnostic·설명가능 품질지도를 만드는 것입니다.

- **Technical Challenges**: 주요 기술 과제는 (1) 픽셀 가시성 기반 pseudo-label이 본질적으로 노이즈를 가진다는 점, (2) 해부학적 구조 특성상 특정 영역은 자연스럽게 낮은 특징(예: 혈관 밀도)을 보일 수 있는데 이를 열화로 오인하지 말아야 한다는 점입니다. 논문은 이를 위해 DINOv3 특징을 frozen으로 쓰고, wide-yet-shallow feature adapter로 큰 receptive field를 확보해 전역 문맥을 보완합니다. 학습에서는 신뢰도 높은 픽셀만 low-quality로 엄격히 선별해 PU 프레임워크를 적용하고, nnPU로 위험(risk) 음수화를 막아 pseudo-label 과적합과 학습 불안정(위험 추정 붕괴)을 완화했습니다.

- **Empirical Impact**: FunPiQ 벤치마크에서 EFIQA-CP는 전반적으로 가장 높은 성능을 보였고, BRSET와 EyeQ에서는 모든 주요 지표에서 1위를 차지했습니다. 통합 성능에서도 EFIQA 대비 QWK, mDice, Reject Dice에서 통계적으로 유의미한 개선을 보이며, 설명가능 설계(EBD) 계열이 local 품질 예측에 뚜렷한 이점을 가진다는 점을 실증했습니다. 아울러 end-to-end+saliency, 일반 anomaly detection(PatchCore/UniVAD)보다 픽셀 맵의 공간 정밀도가 더 좋아 실제 품질 보증(어디가 나쁜지) 관점에서 실용성이 높다고 결론 내립니다.



### In-context Region-based Drag: Drag Any Region to Any Shap (https://arxiv.org/abs/2606.25907)
Comments:
          Accepted by ECCV 2026. Dataset, code, and model are available at this https URL

- **Prior Approaches**: 기존 drag-style 편집은 점(point) 기반과 영역(region) 기반으로 나뉘며, 점 기반은 소수의 점쌍으로 인해 결과가 여러 가지로 해석될 수 있는 모호성이 문제로 지적돼 왔다. 영역 기반 연구는 상대적으로 적었는데, EditGAN은 마스크를 주로 loss로만 활용해 이미지-마스크 간 통합이 얕고, RegionDrag는 latent 공간에서 copy-paste로 옮겨 경계 불일치나 복잡한 형태 변형에서 한계를 보였다. 이런 배경에서 ‘영역의 구조 정보를 생성 과정에 더 깊게 결합’하는 접근이 필요하다는 동기가 제시된다.

- **Core Contribution**: 본 논문은 region-based drag를 In-Context Region-based Drag(ICRDrag)로 정식화하고, in-context learning 프레임워크 아래에서 source image, source region mask, target region mask를 하나의 통합 context로 넣어 단일 forward pass로 편집 결과를 생성한다. 또한 DiT 기반 생성기에 대해 두 가지 attention 정규화를 제안한다: 이미지-마스크 attention consistency(IMAC)와 source-target attention correspondence(STAC). 마지막으로 region 기반 drag를 위한 대규모 Paired Region Dataset(PRD)을 구축해 학습과 평가의 기반을 마련한다.

- **Technical Challenges**: 가장 큰 기술 과제는 mask는 희소한 구조 정보만 갖고 이미지 토큰은 고해상 텍스처를 담는다는 점에서, 두 모달리티를 공유 파라미터로 처리하면 feature confusion으로 디테일 손실과 과도한 매끈함이 생길 수 있다는 것이다. 이를 위해 ICRDrag는 이미지/마스크 토큰에 각각 분리된 LoRA 모듈을 붙이고, target 이미지 latent만 노이즈-디노이즈하도록 flow-matching 학습을 구성한다. 더 나아가 IMAC는 target 영역이 이미지와 마스크 모달리티에서 유사한 source 영역에 주목하도록, STAC는 source와 target 패치 간 상호 대응(attention round-trip)을 강제하도록 attention 기반 보조손실을 설계하며, 완전 마스크→불완전(부분) 마스크로 난이도를 올리는 2단계 커리큘럼 학습으로 희소 사용자 입력에도 강건하게 만든다.

- **Empirical Impact**: 실험에서는 ICRDrag가 정량 지표와 사용자 연구 모두에서 기존 방법을 유의미하게 앞서며, 편집 정확도와 시각적 사실감, 세부 보존 측면에서 더 나은 결과를 보였다고 보고한다. 특히 region 기반에서 흔한 경계 불일치나 복잡한 형태 변형의 어려움을 attention 정규화와 커리큘럼 학습으로 완화한 점이 핵심 성과로 제시된다. 또한 PRD와 PRDBench를 공개해 향후 region-based drag 연구의 평가 표준과 데이터 접근성을 크게 끌어올릴 것으로 기대된다.



### OracleAnalyser: Analysing Implicit Semantics of Oracle Bone Scripts through MLLMs with Post-training (https://arxiv.org/abs/2606.25906)
- **Prior Approaches**: 기존 오라클 본(OBC) 관련 연구와 벤치마크는 주로 글리프(기호) 인식에 집중해, 문장·근거 수준의 분석 능력을 평가하지 못한다. OBSD나 Puzzle Pieces Picker처럼 생성/복원형 접근도 관찰된 한계는 인식 성능 중심에 머물며, 추론·해석의 ‘품질’을 직접 가늠하는 지표가 부족하다는 문제의식이 제기된다. 또한 강화학습/RFT 흐름이 일반 영역에서 성과를 보였지만, 오라클 본 분석에는 이를 체계적으로 적용한 연구가 제한적이었다.

- **Core Contribution**: 본 논문은 오라클 본 분석을 위한 추론(reasoning) 프레임워크 OracleAnalyser를 제안한다. Qwen2.5-VL-3B-Instruct를 여러 단계 post-training으로 미세조정하고, 오라클 본 데이터 특성에 맞춘 preference optimization 알고리즘 Stable Focal Preference Optimization (SFPO)을 도입해 분석 생성의 품질을 끌어올린다. 더불어 오라클 본 reasoning 데이터셋, preference 데이터셋을 공개하고, 인식이 아닌 ‘분석 출력’ 능력을 평가하는 새로운 벤치마크도 구축한다.

- **Technical Challenges**: 핵심 기술 과제는 오라클 본 분야에서 preference 페어가 갖는 잡음과 불완전성을 그대로 DPO에 넣을 때 생길 수 있는 학습 불안정(예: reward hacking, 하드 페어 과도 강조)이다. 저자들은 모델이 생성한 5개 추론 trace에서 빈도 기반으로 negative를 뽑아 상대 선호쌍을 만들되, 애매하거나 식별이 어려운 경우는 수작업 필터링해 데이터 품질을 확보한다. 여기에 SFPO로 (1) reference 대비 과도한 발산을 제어하는 안정화 항과 (2) 학습에 유용한 쌍에 가중치를 더 주도록 하는 focal reweighting을 결합해, 하드-to-rank 쌍이 만들어내는 왜곡을 완화한다.

- **Empirical Impact**: 실험에서 OracleAnalyser는 3B 파라미터 규모만으로도 기존 더 큰 MLLM 및 오라클 본 인식 특화 모델 대비 분석 지표에서 35~70% 수준의 개선을 보인다. in-domain과 out-of-domain 모두에서 우수하며, 특히 미지 문자에 대한 SBS(Sentence-BERT Score)에서 최상위 성능을 기록해 해석이 정답과 의미적으로 더 잘 정렬됨을 시사한다. 또한 preference 데이터 필터링과 SFPO의 구성요소가 성능/안정성에 직접 기여함을 ablation으로 확인해, 단순 인식이 아닌 분석 추론 역량을 학습했음을 뒷받침한다.



### SurgAtlas: A Large-Scale Surgical Video-Language Dataset with 2,391 Hours of Open and Minimally Invasive Surgery (https://arxiv.org/abs/2606.25905)
- **Prior Approaches**: 기존 수술 비디오-언어 연구는 내레이션이 있는 자료에서 대비학습이나 문답을 학습하거나, 공개 데이터셋의 구조화 라벨을 멀티모달 대화 형식으로 변환해 규모를 키우는 방식이 주류였다. 하지만 이 방식은 원천 데이터의 폐쇄형 어휘 온톨로지에 갇히고, “왜 해당 조작을 하는지/다음 단계가 무엇인지” 같은 임상적 추론을 충분히 담기 어렵다는 한계가 있었다. 또한 공개 코퍼스 자체가 작고(내레이션 기반 몇 천 편 수준), 공개 VL 데이터셋에서 open surgery는 거의 다루지 못해 시각적 분포 차이를 보완하기 어려웠다.

- **Core Contribution**: SurgAtlas는 YouTube 공개 콘텐츠만으로 구성된 대규모 수술 비디오-언어 데이터셋으로, 15,291개 비디오(2,391시간)와 18개 전문 분야, 5,000+ 시술을 포괄한다. 특히 open surgery를 대규모로 포함(6,182편)하고 minimally invasive도 함께 제공하며, open-surgery 비디오 이해를 위한 표준 벤치마크와 expert-validated 언어-시각 QA 서브셋을 처음으로 구축했다. 캡션·단계/phase 설명·시술 요약·추론 지향 QA를 계층적 분류체계로 정리해 “절차 맥락과 reasoning” 학습에 직접 연결되도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 공개 비디오에서 수술 구간과 정보성 내레이션/온스크린 텍스트를 안정적으로 복원하고, (2) 절차적 라벨을 단순 매칭이 아니라 groundedness가 있는 언어로 확장하며, (3) open surgery의 고유한 시각적 양상을 대규모로 처리하는 것이다. 저자들은 3단계 수퍼비전 추출(Tier 1: 내레이션 기반 문장/단계/요약, Tier 2: OCR 텍스트 정제·표준 단계 매핑·설명 확장, Tier 3: 메타데이터 기반 요약)과 presentation 포맷의 경우 수술 영역 crop을 포함한 파이프라인을 사용했다. 또한 segment 캡션을 기준으로 staged VQA 생성 시 groundedness 검증을 명시적으로 수행해 질문-답이 특정 시각 근거에 붙도록 만들었다.

- **Empirical Impact**: SurgAtlas 규모와 다양성을 바탕으로 Qwen3-VL-8B를 captioning-then-instruction의 2단계 파이프라인으로 fine-tuning했으며, phase recognition, triplet detection, reasoning question answering 등 여러 수술 비디오 이해 과제에서 competitive 또는 state-of-the-art 성능을 보였다고 보고했다. 특히 open surgery용 표준 벤치마크와 expert-validated QA 서브셋은 평가 신뢰도를 높여, 이전에는 부재했던 open-surgery 이해 연구를 촉진할 것으로 기대된다. 결과적으로 이 데이터셋은 향후 multimodal surgical AI의 대규모 사전학습(pretraining) 및 차세대 foundation model 개발을 위한 “네이티브 공개 비디오 코퍼스” 역할을 지향한다.



### Enhancing Brain MRI Anomaly Detection and Reasoning with ROI Rethink and Synthetic Data (https://arxiv.org/abs/2606.25894)
- **Prior Approaches**: 기존 의료 비전-언어 모델은 한 번의 추론(single-pass inference)로 진단을 내리는 경우가 많아, 어떤 ROI(관심영역)가 근거인지 시각적 근거 제공이 약했다. 이로 인해 결과를 감사(audit)하기 어렵고, 정상 영상에서도 허위 소견(hallucination)이 나올 위험이 커 임상 활용도가 제한된다. 또한 뇌 MRI는 병변 위치가 감별진단에 핵심이지만, 주로 closed-ended 정확도 중심 벤치마크에 묶여 open-ended 일반화와 OOD 성능이 충분히 다뤄지지 않았다.

- **Core Contribution**: BrReMark(Brain Rethink via ROI Marking)는 뇌 MRI 진단을 2턴 시각 대화로 바꾸고, 가설-표식-검증의 추론 흐름을 명시적으로 ROI 바운딩박스로 고정한다. 1턴에서 이상 후보와 bounding box를 제시한 뒤 표시된 이미지를 재검토하여 결론을 verify하도록 설계해, 결과의 공간적 근거를 사람이 확인 가능하게 만든다. 학습 단계에서는 SFT로 “Mark-and-Rethink” 궤적 포맷을 가르치고, 이후 GRPO 기반 RL로 진단 서술과 근거 일치까지 최적화한다.

- **Technical Challenges**: 핵심 난제는 (1) ROI 위치·진단 텍스트·형식 태그를 동시에 만족시키는 open-ended 학습 신호를 설계하는 것과 (2) 드문 병변(희귀 병리) 때문에 OOD에서 허위 소견이 늘지 않게 하는 것이다. BrReMark는 localization 정확도, 의미적 진단 타당성, 안전성을 묶은 복합 보상을 만들고, modality gating·hallucination penalty·synthetic masking 등 reward gatekeeper로 진단 신뢰성을 제어한다. 더불어 SynthSeg 기반 병리 합성에 domain randomization을 적용하되 SFT에는 합성을 섞지 않고 RL에서만 공간 타게팅에 활용해 OOD 강건성을 끌어올렸다.

- **Empirical Impact**: 내부 벤치마크에서 BrReMark는 base 모델 대비 mAP50을 0.74%에서 37.54%로 크게 끌어올렸고, Clinical F1 21.57%, diagnostic accuracy 45.26%를 기록했다. NOVA OOD(희귀 병리 장꼬리)에서도 경쟁적인 성능을 보이며 false positives를 SOTA 대비 45.7% 줄여 드문 병리에 대한 허위 소견이 감소했음을 시사한다. ablation 결과로도 localization reward·합성 병리 RL·semantic correctness reward가 성능과 안전성 향상에 핵심임이 확인됐다.



### USS: Unified Spatial-Semantic Prompts for Embodied Visual Tracking with Latent Dynamics Learning (https://arxiv.org/abs/2606.25880)
- **Prior Approaches**: 기존 Embodied Visual Tracking(EVT)은 주로 텍스트로 목표를 지정한 뒤 closed-loop로 추적·제어한다. RL 기반 엔드투엔드나 모듈형(인식·제어 분리) 접근은 오차 누적/불안정성이 남고, 최근 MLLM 기반은 성능을 키우지만 ‘프롬프트를 받는 방식(표현 인터페이스)’ 자체는 언어 중심에 머무는 경우가 많다. 이때 장면에 유사한 인스턴스가 여럿이면 같은 의미 설명이 여러 대상에 매칭돼 목표 인스턴스가 모호해지는 문제가 반복된다.

- **Core Contribution**: 논문은 EVT의 목표 표시를 언어-only에서 ‘unified spatial-semantic prompting’으로 전환하자고 제안한다. 텍스트뿐 아니라 point, bounding box, mask를 동등한 타깃 지정 입력으로 두고, 모호한 인스턴스 상황에서도 보다 정확한 타깃 grounding을 유도한다. 이를 구현한 end-to-end 프레임워크 USS는 멀티 모달 프롬프트를 하나의 아키텍처에서 처리해 ego-centric waypoints를 예측한다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 형태의 프롬프트(언어/공간)를 공통 표현으로 정렬하고, 시각 증거와 효율적으로 결합해 추적 제어로 이어지게 만드는 것이다. USS는 modality-specific encoder로 프롬프트를 임베딩 공간에 투영하고, hybrid attention(프롬프트/쿼리-비주얼 양방향 read-write-read)으로 유효한 타깃 증거는 강화하고 distractor는 억제하도록 설계했다. 또한 시간적 강건성을 위해 pixel 재구성 대신 latent dynamics learning(자기지도 잠재 정렬)을 넣어 가려짐·시점 변화·동적 움직임에서도 안정적인 미래 표현을 예측한다.

- **Empirical Impact**: 실제 로봇 Unitree G1 실험에서 bounding box/point/mask 같은 명시적 공간 단서는 텍스트-only보다 success rate가 높게 나타났고, 특히 유사한 distractor와 longer-horizon에서 인스턴스 정체성 유지가 중요할 때 효과가 두드러졌다. 시뮬레이션 EVT-Bench에서는 비-MLLM 계열 중 state-of-the-art를 달성하고, MLLM 기반과 비교해도 경쟁적이면서 inference 속도는 더 빠르다고 보고한다. DT/ablation 결과로는 temporal memory와 action-conditioned latent world model이 각각 성능 향상에 기여하며, waypoint를 직접 예측하는 저지연 디코더가 closed-loop 추적에 유리함을 보여준다.



### Naturalness Predicts but Does Not Cause Transferability in Image Encodings of Real-World Streams (https://arxiv.org/abs/2606.25844)
Comments:
          9 pages, 4 figures, 3 tables; code and data manifest included as ancillary files

- **Prior Approaches**: 시계열을 이미지로 바꿔 CNN·ViT 같은 비전 백본에 넣는 접근이 오래전부터 쓰였고, Gramian Angular Field·Markov Transition Field·recurrence plot 등 인코딩도 다양합니다. 하지만 기존 연구는 ‘자연 이미지에 가까움’ 같은 분포 근접성을 직접 측정·최적화하지 않으며, 자연스러움 여부와 전이 정확도 간 관계는 거의 검증되지 않았습니다.

- **Core Contribution**: 이 논문은 고정(frozen) 백본에서 인코딩된 이미지의 ‘시각적 자연성’이 전이(transfer) 정확도에 어떤 의미가 있는지 체계적으로 묻습니다. 또한 WorldStream(9개 공개 API 계열, 299개 시계열, 9-way 소스 인식)을 구축하고, 여러 인코딩·여러 frozen 백본에서 자연 이미지 거리(FID)가 정확도를 예측한다는 상관을 보고합니다.

- **Technical Challenges**: 핵심은 상관이 인과인지 확인하는 통제 실험입니다. 저자들은 정확히 역변환 가능한(invertible) 스펙트럴 인코더에 단 하나의 조절값 β(주파수 파워 ∝ |f|^{-β})를 넣어 ‘자연성’을 바꾸되 콘텐츠는 고정하고, phase scrambling으로 스펙트럼은 유지한 채 국소 구조를 제거하는 두 개의 개입을 설계했습니다. 그 결과 FID는 스펙트럼 자연성 자체가 아니라 국소 구조 변화에 동행할 때만 정확도와 강하게 맞물렸습니다.

- **Empirical Impact**: 7×인코딩×백본 조합에서 전체 Spearman ρ=-0.72로 FID와 전이 정확도가 강한(단조) 상관을 보였지만, β 스윕에서는 FID만 낮아져도 정확도가 크게 오르지 않았고 Pearson은 -0.32 수준에 그쳤습니다. 반면 스펙트럼을 고정한 채 국소 구조를 파괴했을 때 Pearson -0.89로 FID와 정확도가 함께 떨어졌으며, fine-tuning을 해도 구조적 결손(예: 27% vs 67%)은 남았습니다. 즉 ‘자연스러움(FID)’은 인코더/이미지의 국소 구조가 백본이 읽는 단서와 얼마나 겹치는지를 반영하는 예측자이며, 저자들은 WorldStream과 코드를 공개해 재현성을 높였습니다.



### Graph it first! Enabling Reasoning on Long-form Egocentric Videos through Scene Graphs (https://arxiv.org/abs/2606.25842)
- **Prior Approaches**: 기존 멀티모달 LLM 기반 비디오 이해는 토큰 제약 때문에 장영상에서 프레임을 심하게 샘플링할 수밖에 없고, 그 결과 시간적·문맥 정보가 크게 손실되어 정밀한 비디오 추론이 제한된다. 특히 1인칭(egocentric) 환경은 시점 변화와 상호작용이 잦고 장기 의존성이 커서, 짧게 잘린 클립 중심 벤치마크 설계가 “프레임 서브샘플링의 영향”을 충분히 드러내지 못한다는 한계가 있었다.

- **Core Contribution**: 이 논문은 1인칭 비디오 질문응답(VQA)을 위해 텍스트 기반 중간 표현인 Egocentric Scene Graphs(EgoSG)를 도입한다. EgoSG는 시간에 고정(temporally grounded)되면서 객체·속성·공간 관계·상호작용을 구조화해 원본 비디오의 핵심 시공간 정보를 압축하고, MLLM이 토큰 예산 안에서 긴 시퀀스를 효율적으로 추론하도록 한다.

- **Technical Challenges**: 핵심 기술적 난제는 긴 비디오의 시공간 정보를 토큰 병목 없이 유지하면서도, 질문에 필요한 관계와 상호작용을 누락 없이 그래프로 추출·갱신하는 것이다. 이를 위해 비디오는 60초 구간으로 나눠 MLLM이 각 구간마다 이전 그래프를 Update하며 동적으로 장면 그래프를 누적하고, 최종적으로 EVENT 타임스탬프 기반의 직렬화 텍스트를 입력으로 사용한다.

- **Empirical Impact**: HD-EPIC VQA에서 EgoSG 입력은 대부분의 모델에서 원본 비디오 입력보다 성능이 일관되게 높으며, 특히 레시피/3D 지각/객체 모션/재료 식별 같은 장기·다중 객체 공간 추론 범주에서 큰 개선이 관찰된다. 또한 EgoSG VQA는 그래프 생성 비용이 있더라도, 그래프는 재사용 가능하고 장영상으로 갈수록 추론 스케일이 유리해져 총 런타임과 장기 영상 처리 강건성 측면에서 의미 있는 실증적 이득을 보여준다.



### Edges Before Embeddings: A Confidence-Aware Blur Gate for Vision-Language Pipelines (https://arxiv.org/abs/2606.25838)
Comments:
          7 pages, 2 figures, 6 tables. Preprint

- **Prior Approaches**: 기존에는 라플라시안 분산 같은 무참조 blur 점수를 휴리스틱으로 쓰거나, 대규모 복원 네트워크(deblur)가 블러를 “고쳐서” 해결하려는 접근이 주를 이뤘습니다. 하지만 휴리스틱은 텍스처/저질감과 블러를 섞어 오판하고 도메인별 튜닝이 어렵고, 복원 네트워크는 게이트로 쓰기엔 너무 무겁습니다. 또한 연속 품질 점수를 예측하는 IQA 모델들은 이진 sharp/blur 라우팅용 abstention 문제로 바로 전환하기가 쉽지 않았습니다.

- **Core Contribution**: 이 논문은 비전·문서 파이프라인 앞단에서 이미지를 sharp/blur/uncertain로 분류해 후속 OCR·VLM 호출을 제어하는 CPU 친화형 blur quality gate인 MagikaDocumentFromPixel(blur 검출 게이트)을 제안합니다. 전체 성능은 기존 모델 골격이나 새 loss가 아니라, 실험으로 찾은 학습/입력 레시피와 라우팅 규칙, 그리고 Edge Prior Module(EPM)로 크게 끌어올립니다. 특히 선택적 예측(selective prediction) 관점의 confidence-aware routing을 통해 uncertain일 때는 abstain으로 “안전하게” 넘기게 만듭니다.

- **Technical Challenges**: 핵심 과제는 (1) blur를 빠르고 CPU에서 안정적으로 판별하고 (2) uncertain를 실제 배포 상황에 맞게 라우팅해 비용 대비 정확도를 최적화하는 데 있습니다. 저자들은 실험 스윕을 통해 입력 해상도가 우세한 레버임을 확인하고, backbone 용량은 384 px 이상에서만 의미 있게 이득을 준다고 정리합니다. 또한 EPM으로 Laplacian magnitude(라플라시안 크기) 보조 채널을 RGB에 4번째 입력으로 결합해 고주파 붕괴라는 고전 휴리스틱의 “증거”를 네트워크가 직접 보게 했고, max-softmax 기반 threshold τ를 배포-time product knob으로 둬 재현성 없는 캘리브레이션 네트워크 의존을 줄였습니다.

- **Empirical Impact**: GoPro Large 단일 seed·단일 motion-blur 분포에서, MobileNetV3-Large + EPM + 384x384 학습과 5-scale test-time augmentation(TTA) 조합은 F1=0.9803(AUC 0.9989), 17 MB ONNX로 약 7 ms 내외의 속도를 보였습니다. 동일 환경 기준 고정 해상도 베이스라인(F1=0.9672) 대비 +1.31p 개선이며, EPM이 단일 효과로 가장 큰 레버로 확인됐습니다. 다만 결과는 다른 blur 유형(디포커스·저조도·압축·스캐너 skew)과의 교차 데이터셋 검증, ECE/신뢰도 도표 같은 정량 캘리브레이션, τ 스윕·다중 seed 통계가 제한이라 추가 연구가 필요하다고 명시했습니다.



### Shift Variant Image Degradation and Restoration Using Singular Value Decomposition (https://arxiv.org/abs/2606.25818)
- **Prior Approaches**: 기존 영상 복원은 주로 shift-invariant(공간적으로 동일한) 열화에서 단일 convolution 커널로 모델링하는 경우가 많아, shift-variant(위치에 따라 달라지는) 열화에는 그대로 적용하기 어렵다. shift-variant motion blur는 공간마다 PSF가 달라져 하나의 커널로는 표현이 불가능하고, 이에 따라 복원 과정이 ill-conditioned가 되며 잡음 증폭 문제가 커진다.

- **Core Contribution**: 이 논문은 shift-variant motion blur를 위한 SVD 기반 복원 프레임워크를 제안한다. 위치 의존 PSF를 shift-variant imaging operator로 모델링하고, SVD에서 작은 singular value의 개수를 singular-value energy retention criterion으로 선택해 노이즈 증폭을 통제하면서 유효한 영상 정보를 보존한다.

- **Technical Challenges**: 핵심 기술적 난제는 열화 연산자를 SVD로 분해해도 복원이 작은 singular value를 역으로 증폭시키며 불안정해질 수 있다는 점이다. 논문은 누적 singular-value energy의 특정 비율을 기준으로 작은 singular value를 적절히 잘라내는 방식으로 안정적 역문제를 만들고, bidirectional linear motion, Gaussian motion, simple harmonic motion 등 1D shift-variant PSF에 대해 이를 일관되게 적용한다.

- **Empirical Impact**: 세 가지 대표 1D motion 모델로 생성된 degraded image에 제안 방법을 적용했으며, 블러 아티팩트를 줄이면서 디테일 복원 성능이 개선됨을 실험으로 보였다. 특히 singular-value 기반 정량 기준을 통해 noise amplification을 제어하는 절차가 제시되어, shift-variant 복원 문제에서 재현성과 운용 가능성을 높인 점이 의미 있다.



### $S^{2}$-FracMix: Label-Preserving Self-Saliency Mixup Augmentation (https://arxiv.org/abs/2606.25784)
Comments:
          Accepted at ECCV 2026

- **Prior Approaches**: 최근 mixup 계열은 보간(interpolated) 샘플을 만들어 일반화를 높이지만, 대개 샘플 간 섞임이 들어가면서 의미적 일관성이 깨지거나 계산 오버헤드가 커질 수 있습니다. CutMix, Manifold Mixup, ResizeMix 같은 inter-image mixing 방식은 특히 시각적으로 중요한(살리언트) 영역을 보존하는 데 한계가 있고, SaliencyMix·PuzzleMix·GuidedMixup 등 살리언시 기반 방법도 종종 서로 다른 이미지의 정보를 옮겨오는 데서 의미 불일치를 유발합니다. 또한 fractal·텍스처 계열 증강은 프랙탈을 이미지 전역에 퍼뜨리면서 불필요한 분포 이동이 발생해 강건성에 악영향을 줄 수 있습니다.

- **Core Contribution**: 이 논문은 Self-Saliency (S2S2) Mixup으로 ‘같은 이미지 안’에서 멀티스케일 살리언트 패치를 추출·변형해 레이블 일관성을 유지하면서 구조적 다양성을 주는 프레임워크를 제안합니다. 여기에 FracMix를 결합해 프랙탈의 self-similarity 패턴을 살리언트 영역에만 선택적으로 주입함으로써, 프랙탈/비프랙탈 구조를 단일 샘플 안에 동시에 담아 의미적·구조적 정합성을 노립니다. 최종적으로 S2S2-FracMix(및 멀티모드 mixing)를 통해 타깃하고도 구조적으로 일관된 증강 전략을 통합 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘살리언트 정보를 보존하면서도’ 스케일 불변(scale-invariant) 학습과 강건성 개선을 동시에 달성하는 것입니다. 저자들은 spectral residual 기반 살리언시 맵으로 멀티스케일 패치를 고르고, 각 패치를 회전·블러 변형한 뒤 원본 이미지의 비살리언트 위치에 재삽입하여 cross-sample 간섭을 피합니다. 이어 FracMix에서는 사전 구축한 fractal 라이브러리를 살리언트 패치에만 선택 블렌딩(적응 비율)하며, 학습 전반에 Mixup/CutMix/ResizeMix와 같은 다른 mixing 모드를 랜덤하게 섞는 멀티모드 전략으로 다양성을 추가합니다.

- **Empirical Impact**: 실험에서 저자들은 7개 벤치마크에 대해 분류(일반/세부), 강건성, calibration, object detection, 전이 학습까지 폭넓게 평가해 S2S2-FracMix가 기존 방법들보다 일관되게 우수함을 보였습니다. 특히 CIFAR-100에서 최고 성능 기법(예: AdAutoMix) 대비 Top-1 정확도 향상이 관찰되며, Tiny-ImageNet·ImageNet-1K에서도 격차가 더 커져 풍부한 판별 특징을 더 잘 학습하는 경향을 보입니다. 또한 clean accuracy뿐 아니라 잡음·대응 변형(예: corruption) 및 adversarial/강건성 관련 지표에서 이득이 크게 나타나, 실제로 더 넓은 downstream 시나리오로 일반화될 수 있음을 시사합니다.



### ShutterMuse: Capture-Time Photography Guidance with MLLMs (https://arxiv.org/abs/2606.25763)
Comments:
          Project Page:this https URL

- **Prior Approaches**: 기존 미적 크롭(photography cropping) 벤치마크는 주로 사후(post-hoc)로 더 좋은 크롭 박스를 예측하는 문제로 정의돼, 촬영 순간의 가이드(keep/retain/reject 같은 의사결정)를 충분히 평가하지 못합니다. 또 일부 MLLM 기반 크롭 모델은 설명이나 aesthetic reasoning은 강화했지만, 정작 사용자에게 바로 실행 가능한 pose(피사체 자세) 추천으로는 연결되지 않습니다. 결과적으로 촬영 시점의 상호작용형 조언(구도와 자세를 동시에 다루는) 역량은 상대적으로 공백으로 남아 있습니다.

- **Core Contribution**: 이 논문은 촬영 시점(capture-time)의 가이드를 평가하기 위한 벤치마크 CaptureGuide-Bench를 제안하며, 두 축(사진가-side 구도 판단/리파인, 피사체-side 장면 조건 자세 추천)으로 문제를 분해해 측정합니다. 더 나아가 CaptureGuide-Dataset(약 13만 샘플)을 구축해 구조화된 구도 박스/근거, 자세 keypoints/가시성/근거를 함께 제공하고, 이를 기반으로 통합 MLLM ShutterMuse를 학습합니다. ShutterMuse는 감독 fine-tuning과 강화 fine-tuning을 결합해 JSON 형태의 해석 가능한 가이드를 생성하도록 설계됩니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 사진가-side에서 refine/keep/reject의 3분류와 정밀 박스 리파인을 동시에 요구하고, (2) 피사체-side에서는 여러 포즈가 가능하므로 단일 좌표 오차만으로 품질을 평가하기 어렵다는 점입니다. 저자들은 전문가 주도 seed를 만들고 expert-seeded, MLLM-verified self-distillation로 스케일링하되 검증 세트를 고정해 오류 누적을 줄였으며, 피사체 데이터는 인물 제거 후 장면에 맞는 pose keypoints와 가시성 상태, 그리고 추천형 텍스트 근거를 결합해 구성했습니다. 또한 학습 단계에서는 SFT로 스키마를 강제하고, RFT에서는 GRPO로 task_type에 따른 보상을 주어 출력 정확도와 의사결정을 개선했습니다.

- **Empirical Impact**: CaptureGuide-Bench 실험에서 ShutterMuse는 baseline 대비 사진가-side 종합 성능에서 가장 우수했고, 특히 IoU/BDE 및 R, RSR, KSR 평가에서 강점을 보였습니다. 피사체-side에서도 물리적 타당성, 장면 상호작용, 자세 미학의 평균 점수에서 경쟁력 있는 pose 추천을 보였으며, 동시에 inference cost를 크게 낮춰 실용성을 강조합니다. 저자들은 MLLM이 촬영 순간에 개입하는 인터랙티브 어시스턴트로 확장될 수 있음을 실증했다고 결론냅니다.



### Dual Distribution Estimation for Zero-shot Noisy Test-Time Adaptation with VLMs (https://arxiv.org/abs/2606.25758)
Comments:
          Accepted by ECCV2026. Project Page:this https URL

- **Prior Approaches**: 기존 Noisy TTA(NTTA)·TTA는 ID 데이터만 온전히 포함된 closed-world 가정을 자주 사용하지만, 실제 스트림에는 OOD outlier가 섞여 성능이 크게 흔들립니다. 제로샷 NTTA 쪽 대표로 AdaND는 온라인에서 노이즈 탐지기를 학습하되, 역사적 테스트 특징 분포를 모델링하지 못해 오분류가 늘고 데이터가 적을 때 의존성이 커집니다. 또한 온라인 학습 방식은 추론 효율을 떨어뜨려 실사용 확장성에 불리합니다.

- **Core Contribution**: 이 논문은 zero-shot·training-free NTTA를 목표로, Dual Distribution Estimation(DDE)라는 프레임워크를 제안합니다. DDE는 인스턴스 단위 학습 대신 Gaussian 분포 모델링으로 전환하며, Positive Feature Distribution Estimation(PFDE)으로 ID 정확도를 보정하고 Negative Label Distribution Estimation(NLDE)로 OOD 식별을 강화합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) ID 클래스 내 다양성을 반영하면서도 과신 오분류를 줄이는 것, (2) 부정 레이블(negative label)이 유발하는 스퓨리어스 상관을 OOD 탐지에서 배제하는 것입니다. 저자들은 PFDE에서 inclusion/exclusion 두 개의 class-wise Gaussian을 분리해 calibrated contrastive score를 만들고, NLDE에서는 부정 이미지에서 구분력이 큰 negative label만 선별해 잡음을 줄이며, 마지막으로 adaptive thresholding으로 ID·OOD를 온라인에서 분리합니다.

- **Empirical Impact**: ImageNet 및 분포 이동 변형, fine-grained 데이터셋 전반에서 DDE는 최고 harmonic mean 정확도를 일관되게 달성합니다. ImageNet에서는 harmonic mean 정확도가 3.70%p 개선(기준 AdaND 대비)됐고, OOD 탐지 지표 FPR95는 6.20%p 감소했습니다. 무엇보다 학습 없이 zero-shot으로 동작해 데이터가 부족한 상황에서도 견고함과 높은 온라인 추론 효율을 함께 보여줍니다.



### Point Cloud Diffusion with Global and Local Reconstruction for Instance-Level 3D Anomaly Detection (https://arxiv.org/abs/2606.25740)
- **Prior Approaches**: 기존 3D 이상 탐지는 기준 정상 샘플을 바탕으로 입력을 복원한 뒤 잔차로 결함을 찾는 reconstruction-based가 주류다. 확산 기반 접근도 등장했지만, 전체 point set에 대한 global 복원이 중심이라 미세·약한 결함에는 공간 적응성이 부족해 성능이 제한된다. 또한 결함은 잘 복원되지 않는 반면, 배경은 재구성 과정에서 위치 편향이 생겨 false positive가 늘어나는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 instance-level 3D 이상 생성과 탐지를 동시에 겨냥한 diffusion 프레임워크 PCDiff를 제안한다. 생성 단계에서는 texture gradient, image patch, text, mask를 조건으로 한 instance-level multi-modal attention을 넣어 약한 결함(예: 잔흔·스크래치)도 고품질로 생성한다. 탐지 단계에서는 2D에서 유도한 이상 마스크를 근거로 local anomaly 복원과 global 기하 일관성을 결합하는 local-global 재구성으로 배경 구조는 보존하면서 전경 결함을 복원한다.

- **Technical Challenges**: 핵심 난제는 (1) 정규화된 point cloud에서 편차가 10^{-3} 수준까지 작아지는 약한 결함을 모델이 복원·검출하기 어렵다는 점과 (2) global 복원 위주의 학습이 배경의 positional bias를 유발해 오탐을 만든다는 점이다. 논문은 gradient 기반의 transformation-aware texture 표현과 geometry bank를 활용해 미세 결함의 텍스처 복잡도를 조건화하고, 2D anomaly mask를 back-projection해 다중 뷰 합의로 위치 잡음을 줄인다. 이어 local-global joint reconstruction과 selective merging으로 이상 영역 복원에는 지역 우선순위를, 전체 형상에는 구조적 기준선을 제공해 두 문제를 동시에 완화한다.

- **Empirical Impact**: Anomaly-ShapeNet과 Real3D-AD에서 PCDiff는 생성 fidelity(F-score, CLIP similarity)와 이상 탐지 성능(O-AUROC, P-AUROC) 모두에서 SOTA를 능가한다. 특히 Anomaly-ShapeNet에서 O-AUROC가 0.93 수준으로 가장 높았고, Real3D-AD에서도 O-AUROC가 0.82로 우수한 이상 국소화 결과를 보였다. 정성적으로는 기존 방법들이 경계만 강조하거나( false negatives/positives) 일부 난제에서 오탐을 보이는 반면, PCDiff는 이상 영역을 더 정확하고 일관되게 포착해 실사용 관점의 신뢰도를 높였다는 점이 확인된다.



### UniTeD: Unified Temporal Diffusion for Joint Perception and Planning in Autonomous Driving (https://arxiv.org/abs/2606.25736)
Comments:
          Accept to ECCV 2026

- **Prior Approaches**: 기존 E2E 자율주행은 지각과 계획을 분리해 학습·추론하는 경우가 많아, 계획 단계에서 diffusion을 쓰더라도 지각 결과를 고정 조건으로만 활용하는 구조가 흔했습니다. 이런 decoupled 설계는 지각의 오차가 생성 과정에 그대로 전파되어 최적화가 어려워지고 강건성이 떨어질 수 있습니다. 또한 diffusion 기반 계획이 단일 프레임 중심이거나 과거 쿼리를 그대로 섞는 방식이면 시간 동학을 충분히 반영하지 못하며, sparse query 기반 접근은 training–inference 불일치도 발생합니다.

- **Core Contribution**: UniTeD는 perception(에이전트·맵)과 planning을 하나의 shared generative space에서 동시에 denoising하여, 양방향 정보 교환이 가능한 unified temporal diffusion 프레임워크를 제안합니다. 노이즈 조건을 포함한 multi-task 학습으로 작업 간 상호 보정이 일어나고, 각 태스크가 “다른 태스크의 노이즈가 섞인 중간 출력”에도 견디도록 만들어 강건성을 높입니다. 여기에 streaming을 위해 temporal context를 도입합니다.

- **Technical Challenges**: 핵심 난제는 (1) 과거 프레임 쿼리와 현재 프레임 쿼리의 denoising noise level이 불일치해 cross-frame 상호작용이 흔들린다는 점, (2) sparse diffusion에서 학습 중에는 일부 쿼리만 감독하지만 추론에서는 모든 쿼리가 반복 업데이트되어 분포가 drift된다는 점입니다. UniTeD는 이를 해결하기 위해 Temporal Transition Module(TTM)로 과거 쿼리를 현재 noise manifold에 정렬하고, Anchor Refresh Strategy(ARS)로 추론 시 저신뢰 쿼리는 초기 anchor/noise 분포에서 재샘플링해 training–inference shift를 완화합니다.

- **Empirical Impact**: 실험에서 UniTeD는 multiple benchmarks에서 diffusion 기반 계획 접근과 discriminative E2E 접근 모두를 능가하며, NAVSIM v1에서 PDMS 90.2 등 최상위 성능을 보였다고 보고합니다. 특히 지각·계획을 함께 생성하는 구조 덕분에 다중모달 궤적 생성과 동적 에이전트 예측·정적 맵 이해까지 동시에 개선되는 방향성이 관찰됩니다. 벤치마크 전반의 성능 향상은 자율주행의 generative E2E 설계가 perception 오차 전파 문제와 시간 모델링 한계를 함께 다룰 수 있음을 시사합니다.



### Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling (https://arxiv.org/abs/2606.25732)
- **Prior Approaches**: 기존 단일 이미지 디헤이징은 ASM(대기 산란 모델)에 기반한 estimate-and-invert 계열과, hazy-to-clear을 end-to-end로 학습하는 blind 모델로 크게 나뉩니다. 전자는 전송맵·대기광 같은 파라미터를 명시적으로 추정하지만, 실제 영상의 ISP 비선형성과 비균일 조명·장면 의존 열화 때문에 추정이 쉽게 흔들려 잔여 헤이즈나 색 왜곡이 생기기 쉽습니다. 후자는 물리 과정을 생략하고 강한 복원력을 학습하지만, 네트워크가 무거워 edge 저지연 배포에 불리하다는 한계가 있습니다.

- **Core Contribution**: 본 논문은 PGL-Net(Physics-Inspired Global–Local Decoupling Network)으로, 물리적 파라미터를 직접 회귀하지 않고 operator-level emulation을 통해 물리적 귀납 편향을 내장한 경량 프레임워크를 제안합니다. 디헤이징을 글로벌 분포 보정(global distribution rectification)과 로컬 구조 정밀 복원(local structural refinement)으로 분리해 해즈로 인한 encoder–decoder 불일치를 체계적으로 줄입니다. 핵심 모듈로는 전역 조건의 Physics-Inspired Affine Fusion(PAF)과, 국소적으로 변하는 열화를 다루는 Degradation-Aware Modulation(DAM)을 도입합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 채널별로 달라지는 산란 때문에 ASM의 단순 가정이 깨지고, (2) U-Net형 skip fusion에서 hazy에서 뽑힌 encoder 특징과 점차 clean으로 가는 decoder 특징의 분포가 어긋난다는 점입니다. PAF는 inverse-ASM 관점에서 특징공간을 채널별 affine(스케일 γ, 시프트 β)로 정렬하되, GAP 기반 전역 디스크립터로부터 affine 파라미터를 생성해 skip 연결에서 분포 미스매치를 완화합니다. DAM은 공간·스펙트럼 변이 열화를 위해 BN 기반 정규화와 경량 depthwise 연산 및 병목 모듈을 조합한 잔차형 feature modulation로 국소 디테일을 복원하며, 픽셀 손실에 더해 FFT 기반 주파수-일관성 손실로 스펙트럼까지 함께 맞춥니다.

- **Empirical Impact**: 실험에서 PGL-Net은 RRSHID·RW2AH·RUDB 등 다양한 real-world 벤치마크에서 복원 품질(PSNR/SSIM/LPIPS 등)을 일관되게 개선하면서도 복잡도를 크게 줄였습니다. 특히 SGDN 대비 PGL-Net-T는 PSNR을 최대 2.6dB 끌어올리고, 추론 지연은 10배 이상 감소했으며 downstream object detection 정확도 또한 안정적으로 향상되었습니다. 또한 RTTS/URHI의 무참조 지표와 시각 비교에서 잔여 헤이즈와 과도한 스무딩을 줄이며, 복원 결과가 SAM 기반 세그멘테이션 경계·영역 품질에도 긍정적 영향을 준다는 점을 보였습니다.



### What Does the Brain See? Multiview Neural Representations to Demystify the Brain-Visual Alignmen (https://arxiv.org/abs/2606.25718)
- **Prior Approaches**: 기존 EEG–비전 정렬 연구는 주로 EEG를 하나의 ‘홀리스틱(전반적)’ 임베딩으로 뽑아 CLIP 같은 사전학습 시맨틱 공간에 맞추는 방식이 많았습니다. 이 접근은 EEG가 가진 시간·주파수·공간 구조의 이질성을 충분히 다루지 못해, cross-subject·cross-session 같은 분포 변화에서 임베딩이 흔들릴 수 있다는 한계가 지적됩니다. 또한 미리 정해진 주파수 밴드를 그대로 쓰거나 일부 차원만 부분적으로 모델링하는 연구도 있어, 통합적 구조 학습이 부족했습니다.

- **Core Contribution**: 이 논문은 EEG–비전 zero-shot 시맨틱 디코딩을 위해 ‘통합 multiview EEG 표현학습’ 프레임워크를 제안합니다. temporal(입력조건 state-space), spectral(learnable wavelet 기반 adaptive 주파수 분해), spatial(attention-modulated graph learning) 세 뷰를 동시에 학습해 EEG 임베딩을 더 안정적이고 시맨틱하게 만듭니다. 이후 사전학습 비전 임베딩과 shared semantic space에서 contrastive learning으로 정렬하며, EEG-specific regularization으로 trial/클래스 수준의 일관성을 강화합니다.

- **Technical Challenges**: 핵심 기술 과제는 EEG의 낮은 신호대잡음비, 비정상성, 그리고 전극의 공간 해상도 한계를 동시에 만족하는 표현을 만드는 것입니다. 이를 위해 장기 시간의 비정상 동역학은 input-conditioned state-space(S3M)로, 샘플마다 달라지는 주파수 구조는 learnable wavelet(ALWD)로, 전극 간 위상·토폴로지 의존성은 동적 adjacency의 attention 기반 GCN으로 모델링합니다. 마지막으로 EEG-specific regularization(반복 일관성 및 prototype 수준 대조)을 더해 분포 이동에도 임베딩 정렬 품질을 유지하도록 설계했습니다.

- **Empirical Impact**: THINGS-EEG 벤치마크에서 200-way zero-shot 시각 분류 성능을 입증했는데, within-subject Top-1 54.8%, Top-5 85.6%를 달성했습니다. cross-subject에서는 Top-1 15.3%, Top-5 45.4%로 기존 대비 더 큰 개선을 보였고, 특히 cross-session에 대한 체계적 평가(Top-1 40.8%, Top-5 78.0%)를 처음 제시하며 기록 세션이 바뀌어도 정렬이 유지됨을 보여줍니다. 결과적으로 multiview 구조를 명시적으로 모델링하면 시맨틱 정렬뿐 아니라 일반화까지 개선된다는 점에서 분야에 실질적 영향이 큽니다.



### Falcon: Functional Assembly and Language for Compositional Reasoning in X-ray (https://arxiv.org/abs/2606.25701)
Comments:
          Accepted at ECCV2026; Project Page: this https URL

- **Prior Approaches**: 기존 비전-언어 모델과 X-ray 보안 분석 연구는 주로 객체 중심으로, 개별 카테고리의 탐지·설명에 위험을 귀속시키는 경향이 강합니다. 하지만 실제로는 배터리·기폭장치·폭약처럼 기능적으로 결합될 때만 위협이 되는 ‘분산 부품’ 구성이 핵심이어서, 상호 호환성 같은 관계 추론이 제대로 평가·학습되지 않았습니다.

- **Core Contribution**: 논문은 X-ray에서의 분해 IED(또는 모듈형 위협) 판단을 ‘compositional threat reasoning(조합형 위협 추론)’으로 정식화하고, 위험을 독립 탐지 결과가 아니라 공간적으로 근거된 부품들의 관계 속성으로 모델링합니다. 이를 위해 segmentation-aware region feature를 바탕으로 부품 존재(presence), 쌍별 기능 호환성(link), 장면 수준 위험(risk)을 포함하는 구조화된 safety state를 만들고, 이를 언어 모델에 Structured Safety Adapter(SSA)로 중간 인터페이스로 주입합니다.

- **Technical Challenges**: 핵심 난제는 겹침·투과·폐색이 심한 X-ray에서 부품 단위의 근거(grounding)를 신뢰성 있게 만들면서, 그 근거를 기반으로 관계적·일관된 위험 추론까지 수행하는 것입니다. Falcon은 DINOv2 기반 입력 특징을 RF-DETR로 instance proposal/마스크를 얻은 뒤, ROIAlign+mask-pooling으로 region 토큰을 구성하고, SSA가 3개 component slot과 pairwise 링크, scene risk를 구조 예측한 다음 안전 토큰을 LLM에 조건화해 관계 일관성을 강제합니다.

- **Empirical Impact**: 평가를 위해 분해 IED의 ‘완전성·부분·결합 상태’와 불확실성을 포함한 structured supervision을 제공하는 Falcon-X(약 7,000 실 촬영 + 마스크 유도 반사실 변형)를 제안합니다. 실험에서는 기존 멀티모달 모델들이 외형 적응은 하더라도 조합형 안전 추론에서는 취약한 패턴을 보였고, Falcon은 functional grounding과 더 일관된 threat assessment에서 개선을 보여 조합형 안전 추론을 별도 평가 패러다임으로 자리매김했습니다.



### Towards a Dynamic and Fixed-budget Memory Bank for Efficient Streaming Video Understanding (https://arxiv.org/abs/2606.25658)
- **Prior Approaches**: 기존 멀티모달 LLM(MLLM)의 스트리밍 비디오 이해는 프레임이 계속 누적되면서 시각 토큰/계산량이 무한히 늘어나는 문제를 먼저 겪는다. 이를 줄이기 위해 token pruning·KV cache compression·FIFO 기반 메모리 등 다양한 압축/메모리 기법이 제안됐지만, 대개 미래 콘텐츠를 알 수 없는 스트리밍 조건에서 “고정 예산”을 유지하며 전역 의미를 보존하는 접근은 제한적이었다. 또한 RAG처럼 질의 기반으로 고르는 방법은 온라인에서 쓸 수 없거나, 전역 중복을 활용하는 오프라인형 압축은 스트리밍에 부적합했다.

- **Core Contribution**: 이 논문은 스트리밍 비디오를 “동적이되 고정 예산인 visual memory bank”로 모델링하고, 이를 학습 없이 적용하는 CausalMem을 제안한다. 핵심은 관측된 스트림의 주요 의미를 온라인 semantic basis로 추정한 뒤, 시각 토큰의 redundancy를 계산해 의미를 최대한 보존하면서 메모리를 업데이트한다는 점이다. LLaVA-OneVision과 Qwen2.5-VL에 플러그앤플레이로 얹어 스트리밍은 물론 오프라인 벤치마크에서도 성능을 확인했다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 미래 프레임이나 질문을 모르는 strict causality 하에서, (2) 메모리 예산 b를 넘지 않게 하면서, (3) 증가하는 시각 토큰 중 “새롭고 정보량이 큰” 부분을 안정적으로 골라야 한다는 것이다. CausalMem은 현재 semantic basis에 대한 투영 오차(잔차 norm)로 토큰별 redundancy를 추정하고, 잔차가 큰 후보를 basis와 memory에 반영하되 활동도(activity score)로 오래된 의미를 자연스럽게 줄이는 방식으로 해결한다. 이후 메모리 버퍼가 b를 초과하면 redundancy 점수에 recency(최근성) 가중을 결합해 상위 토큰만 남기며, 전체 업데이트 과정이 모두 인과적으로 수행되도록 설계했다.

- **Empirical Impact**: 실험 결과 CausalMem은 스트리밍·오프라인 설정 모두에서 기존 방법 대비 평균 정확도 향상을 보였다(스트리밍 +3.2%, 오프라인 +3.0%). 특히 시간당(1시간) 스트리밍 비디오를 12k 토큰 예산으로 기억하면서 시각 토큰을 20배 이상 압축하고, 저장 공간은 약 82MB 수준에 그쳐 semantic preservation이 우수함을 보여줬다. 또한 GPU 메모리 사용과 추론 시간에서도 fixed-budget 메모리 구조 덕분에 긴 스트리밍에서 증가 폭이 가장 느려 실제 배포 가능성까지 강화된 것으로 평가된다.



### Steering Vision-Language Models with Joint Sparse Autoencoders (https://arxiv.org/abs/2606.25657)
Comments:
          19pages,10 figures

- **Prior Approaches**: 기존 Sparse Autoencoder(SAE) 연구는 언어 모델에서 해석 가능한 방향을 찾는 데 성과를 냈지만, VLM에는 그대로 옮기기 어렵습니다. VLM용 접근들(VL-SAE 등)은 공유 딕셔너리나 공분산 기반 정렬처럼 ‘모달리티 간 결합’이 충분히 개입 실험(steering)으로 이어지지 않는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Joint Sparse Autoencoder(JSAE)로, 이미지와 캡션(언어) 활성의 시퀀스-풀드 표현을 공유 잠재공간에서 함께 분해하되 ‘쌍(pair)된 코드의 명시적 cosine 정렬 제약’을 추가합니다. 그 결과 VLM에서 의미 개념(예: 음식, 동물)에 대응하는 크로스모달 기능을 더 “조종 가능한 steering direction”으로 회수하는 것을 목표로 합니다. 또한 LLaVA에 적용해 additive steering과 suppression을 동시에 비교하며 기능적 역할까지 탐색합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 비전 토큰과 텍스트 토큰을 비교 가능한 거시 표현으로 안정적으로 정리하고, (2) 정렬된 sparse 방향이 실제 생성 조작에 쓰일 정도로 ‘방향성/기하’를 보장하는 것입니다. JSAE는 모달리티별 encoder-decoder로 overcomplete sparse 코드를 만든 뒤 재구성·희소성(L1)과 함께 쌍 코드 정렬 손실을 학습에 포함하고, splitting된 원자 피처는 클러스터링으로 매크로 개념 벡터로 재집계한 뒤 양방향 개입을 수행합니다.

- **Empirical Impact**: LLaVA-v1.6-Mistral-7B에서 additive steering은 레이어 7·13·30 사이에서 중간~후반(특히 pre-output 구간)에서 성공률이 가장 높게 나타났고, suppression은 전 레이어에서 상대적으로 비슷한 범위의 점수로 관측됩니다. 이러한 레이어 국소화 효과는 Llama3-LLaVA-Next-8B와 MoE 기반 Qwen3-VL-30B에서도 유사하게 나타나, 정렬 제약이 없는 대안보다 더 controllable한 교차모달 분석에 유리하다는 신호를 제공합니다.



### Auto-Labelling-Based Domain Transfer for 3D Object Detection on a Bicycle-Mounted LiDAR Platform (https://arxiv.org/abs/2606.25652)
- **Prior Approaches**: 기존 AD용 3D 검출은 주로 차량 관점의 라벨에 의존하며, nuScenes·Waymo 같은 대형 데이터도 자전거/보행자를 드물게 관측해 VRU(취약 도로사용자) 클래스에 데이터가 부족하다. 자전거 탑재 플랫폼은 VRU 시점의 데이터를 제공하지만, 여러 센서 배치·LiDAR 스캔·포인트 밀도 차이로 인해 차량용 detector의 domain shift가 커진다. 또한 자전거 관점의 3D 멀티클래스 VRU 라벨은 수작업이 많이 들어 공개 벤치마크가 거의 없어, 도메인 적응 성능이나 자동 라벨 품질을 엄밀히 평가하기 어렵다.

- **Core Contribution**: 본 논문은 자전거( FUSE-Bike ) 관점의 VRU 3D object detection 벤치마크를 제안한다. 총 1,027개의 키프레임(훈련용 자동 라벨)과 1,854개의 사람이 검증한 GT 박스(평가용)를 포함하며, VRU-dedicated auto-labelling 파이프라인으로 학습 라벨을 생성해 별도 수작업 없이 detector finetuning을 수행한다. 또한 nuScenes-pretrained 4종 detector를 대상으로 vehicle-to-cyclist domain gap을 정량화하고, auto-label만으로도 성능 저하를 크게 복구할 수 있음을 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 저지면·민첩한 이동으로 인한 마운팅/스캔 패턴 차이, (2) VRU 클래스의 희소성과 라벨 부족, (3) 수작업 없이 쓸 수 있는 신뢰도 있는 자동 3D 라벨의 확보다. 논문은 VRU-Label3D(ensemble 제안 통합+프레임 간 tracker로 일관성 유지, nuScenes 포맷 export)를 이용해 훈련 라벨을 자동 생성하고, finetuning 시 detection head를 VRU 3클래스(차량/보행자/자전거)로 재구성하되 백본은 pretrain 지식을 유지하도록 설계한다. 평가용 GT는 사람이 카메라 동기화 이미지를 기준으로 SUSTechPOINTS에서 검증·수정해 벤치마크의 신뢰성을 확보했다.

- **Empirical Impact**: 결과적으로 zero-shot에서는 차량 클래스는 비교적 전이되지만 보행자·자전거는 크게 악화되며, domain gap이 안전에 직결된 VRU 클래스에 집중됨을 확인했다. auto-label로 20 epoch finetuning을 하면 mAP가 최대 23.4포인트까지 개선되고, 특히 보행자/자전거 AP 상승 폭이 커져 안전 핵심 클래스에서 효과가 가장 크게 나타난다. 더 나아가 finetuned detector들은 자신들이 학습에 사용한 auto-label의 품질(autolabel mAP 63.9%)을 넘어서는 성능을 보여, 불완전한 자동 라벨도 도메인 적응 학습 신호로 충분함을 실증한다.



### SSMNBench: Diagnosing Image-based Cross-View Human-Object Understanding via Single-View Sufficiency and Multi-View Necessity (https://arxiv.org/abs/2606.25634)
Comments:
          European Conference on Computer Vision (ECCV). 32 pages, 10 figures. The code is available at: $ \href{this https URL}{\text{SSMNBench}} $

- **Prior Approaches**: 기존 멀티뷰 벤치마크는 고정된 여러 장면을 ‘bag of frames’처럼 한꺼번에 넣고 단일 accuracy만 보고해, 시각적 중복이 주는 주의력 혼란과 실제 교차뷰 증거 통합 능력을 구분하지 못했다. 그 결과 모델이 한 장면의 단서만으로 답했는지, 혹은 카메라 간 파편 정보를 합성했는지를 진단하기 어렵다. 또한 SVS(단일뷰로 충분)와 MVN(다중뷰가 필수)을 분리하지 않아 실패 원인 해석이 흐려진다.

- **Core Contribution**: 이 논문은 인간 중심 교차뷰 이해를 위한 진단형 벤치마크 SSMNBench를 제안하며, 3,300개의 QA를 11개 태스크에 배치한다. 핵심은 평가를 Single-View Sufficiency(SVS)와 Multi-View Necessity(MVN)로 나누어, 중복 뷰에 대한 견고성(주의력)과 파편 증거의 통합(융합)을 분리 측정한다. 여기에 -1, +1, +2, +3처럼 뷰 가용성을 체계적으로 섞는 프로토콜로 모델의 반응 양상을 드러내도록 설계했다.

- **Technical Challenges**: 진짜 교차뷰 융합이 필요한 문항을 만들고, 특정 뷰에만 답이 가능하도록 ‘Golden View’ 집합을 엄밀히 라벨링하는 것이 큰 과제였다. 이를 위해 연구팀이 다양한 occlusion-heavy 데이터에서 장면을 모으고, SVS는 단일답 가능한 뷰를, MVN은 최소 필요 뷰 부분집합을 사람이 검토해 일관된 정답을 보장했다. 또한 중복 입력 순서로 인한 positional bias를 줄이기 위해 카메라 뷰 순서를 무작위화하고, 성능 변화량을 요약하는 Distraction Decay(δ_dis) 지표로 중복 뷰가 유도하는 성능 하락을 정량화했다.

- **Empirical Impact**: 17개 SOTA MLLM을 평가한 결과, 추가 뷰가 SVS에서는 거의 보편적으로 성능을 떨어뜨리는 ‘distraction degradation’을 유발했으며 MVN에서는 기하 증거 융합이 취약해 헛추측이 늘어나는 양상이 관측됐다. 정성적으로도 모델들이 진짜 3D cross-view synthesis보다는 의미 특징을 대충 평균내거나 view preference에 의존하는 경향이 확인된다. 특히 모델 용량이 클수록 distraction에 더 취약해지는 역설(δ_dis 증가)까지 드러나, 앞으로의 교차뷰 인지 아키텍처가 선택적 주의와 기하적 정합을 내재화해야 함을 시사한다.



### ScaleHP: Estimating Hand Pose in Metric Spac (https://arxiv.org/abs/2606.25619)
Comments:
          27 pages, 8 figures, 6 tables; includes supplementary material

- **Prior Approaches**: 기존 대부분의 손 포즈 추정(HPE)은 root-relative 좌표계에서 포즈를 예측하거나, 파라메트릭 모델 회귀로 메시/형상을 복원해도 절대 metric scale은 충분히 다루지 못한다. 카메라 기준의 metric-space 재구성은 단안(monocular)에서 scale-depth ambiguity가 커서, 보조 depth 모듈이나 사전 계산된 깊이로 스케일을 맞추는 방식이 흔하지만 장면이 바뀌면(특히 손만 보이거나 out-of-domain) 쉽게 흔들린다. 따라서 실세계에서의 안정적인 절대 좌표 추정이 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 손뼈 사이의 비율이 인체 골격의 고유한 비례(anthropometric priors)를 담고 있으며, 이를 통해 손의 전체 metric 크기를 암묵적으로 추정할 수 있다는 관찰을 제시한다. 이를 바탕으로 ScaleHP는 one-stage end-to-end 구조로, fragile한 외부 depth 모듈 없이 카메라 좌표계에서 손을 절대 metric scale로 복원한다. 핵심은 transformer decoder에 scale token을 도입해 multi-scale 형태·외형 특징을 융합하고, 이후 perspective 제약 하 least-squares로 metric 좌표(translation 포함)를 산출하는 흐름이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 단안 영상의 scale-depth ambiguity를 외부 depth 없이 해결하는 것이다. ScaleHP는 (1) scale token이 2D joint query와 3D root-relative query 간 상호작용을 통해 손의 뼈 비례 기반 스케일을 직접 학습하고, (2) 예측된 2D 좌표·정규화된 3D 포즈·전역 스케일을 이용해 perspective-constrained least-squares로 카메라 공간 translation을 training-free로 복원한다.

- **Empirical Impact**: FreiHand에서 CS-MPJPE 35.8, DexYCB와 HO3Dv3에서 PA-MPJPE 4.6/5.9의 SOTA 성능을 보고하며, 절대 metric 공간 정밀도가 개선됐음을 실증한다. 결과는 내부 생물학적 제약(손의 뼈 비례)이 상대 기하와 깊이 방향 오차를 함께 줄여, generalized한 실환경 손 추적에 더 견고한 해법이 될 수 있음을 보여준다.



### Expresso-AI: Explainable Video-Based Deep Learning Models for Depression Diagnosis (https://arxiv.org/abs/2606.25606)
Comments:
          8 pages. Accepted at the 2023 11th International Conference on Affective Computing and Intelligent Interaction (ACII). Code: this https URL

- **Prior Approaches**: 우울 자동 진단(ADD)은 얼굴 표정, 자세, 시선, 음성 등 다양한 단서를 학습하지만, 많은 연구가 딥모델의 예측 근거를 사람에게 설명하지 못해 임상 적용성이 제한돼 왔다. 특히 비디오 기반 접근은 이미지를 넘어 동적 특징을 잡을 수 있음에도, 시간 축(temporal)까지 정량적으로 해석하는 연구는 부족했다. 기존 해석은 주로 2D 히트맵 등 정성적 시각화에 머물러 비교·검증이 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 얼굴 비디오로 우울 강도를 회귀하는 딥뉴럴네트워크의 의사결정을 해석하기 위한 프레임워크를 제안한다. Action Recognition 데이터로 사전학습된 ResNet 계열을 AVEC 2014 우울 라벨(BDI-II) 비디오에 fine-tuning한 뒤, DeepLift(Rescale Rule) 기반 attribution으로 얼굴 영역과 시간적 표현 의미(temporal expression semantics)를 함께 설명한다. 동시에 프레임/영역 기반 시각 설명과 수치적 설명을 제공해 모델의 추론 가설을 도출하도록 설계했다.

- **Technical Challenges**: 핵심 과제는 비디오 입력에서 시간 정보까지 포함한 attribution을 안정적으로 계산하고, 이를 얼굴 부위·표정 의미와 연결해 임상가가 이해 가능한 형태로 바꾸는 것이다. 저자들은 (2+1)D ResNet에 DeepLift를 적용해 프레임 단위 relevance를 얻고, 채널·공간 풀링으로 temporal attributions 벡터를 만든 뒤, 미리 추정한 얼굴 랜드마크 영역(눈/눈썹/입/코 등)에 대해 영역별 relevance를 집계했다. 또한 OpenFace 기반 Action Unit(AU)과 temporal attributions의 Kendall Tau 상관을 통해 시간 의미를 정량 검증하는 절차를 마련했다.

- **Empirical Impact**: 실험에서 (2+1)D를 비롯한 ResNet 계열을 비교했으며, Kinetics-700과 MiT 사전학습을 함께 사용한 R(2+1)D가 최상 성능을 보였다. 해석 결과로 심한 우울에서 코 꼬집음/눈썹 찡그림(AU9, AU4)처럼 알려진 표정 단서와의 연관성이 관찰됐고, 반대로 기쁨/정상 범주와 관련된 AU 조합은 음의 상관을 보였다. 한편 마이크·헤드셋 영역 높은 attribution은 데이터 편향 가능성을 시사하며, 제안된 해석 프레임워크가 성능뿐 아니라 근거 품질 점검에도 도움을 줄 수 있음을 보여줬다.



### VPA-Guard: Defending and Benchmarking Image-to-Video Generation Against Visual Prompt Attacks (https://arxiv.org/abs/2606.25592)
Comments:
          Dataset Page: this https URL

- **Prior Approaches**: 기존 비디오 생성 안전 연구는 주로 텍스트 기반 jailbreak이나 출력 콘텐츠 필터링에 집중돼, 이미지 속 시각 단서가 ‘시간적 지시’처럼 작동하는 위험은 충분히 다뤄지지 않았다. 또한 정적 입력 필터는 의도가 시각 큐, 공간 배치, 생성 과정의 동역학에 분산돼 있을 때 놓치기 쉽다.

- **Core Contribution**: 본 논문은 I2V(image-to-video)에서 시각 프롬프트 공격의 안전 취약성을 체계적으로 진단하는 최초 벤치마크 VVA-Bench를 제안한다. 공격을 단일-이미지 시각 단서(VP)와 두 프레임 기반(2frames)으로 나누고, VP를 Movement Control, Emoji Instruction, Pose Control, Draft Instantiation, Camera Control의 5개 메커니즘으로 분류한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘순수한 사용자 편집’과 ‘숨은 악의적 의도’를 시각 단서만으로 구분하는 것이다. 이를 위해 VPA-Guard는 DINOv2 기반 시각 인코딩과 텍스트 의미 임베딩을 결합해 벡터 DB에서 유사 공격 패턴을 retrieval하고, few-shot 추론으로 잠재 위협을 판별한 뒤 필요 시 reflection agent로 OOD 공격에 대응하며 지식을 self-evolving 방식으로 확장한다.

- **Empirical Impact**: 실험 결과, 최신 I2V 모델들은 VVA-Bench의 시각 프롬프트 공격에 큰 취약성을 보이며 Wan 2.7에서 ASR 100.0%, Veo 3.1에서 ASR 74.8%까지 도달했다. VPA-Guard 적용 시 평균 ASR이 44.2%p, harmfulness score가 73.4%만큼 감소하면서도 정상 편집 거부율(OSR)은 평균 12.8%로 낮게 유지됐다.



### FeVOS: Foresight Expression Video Object Segmentation (https://arxiv.org/abs/2606.25585)
Comments:
          Accepted by ECCV 2026. Homepage: this https URL

- **Prior Approaches**: 기존 Referring Video Object Segmentation(RVOS)은 주어진 표현이 “이미 관찰된 프레임”에서 가리키는 대상을 픽셀 단위로 분할하는 데 초점이 맞춰져 있다. Ref-DAVIS/Ref-YouTube-VOS는 정적 속성 위주였고, MeViS는 motion 표현으로 시공간 이해를 넓혔지만 여전히 미래 사건을 예측하진 않는다.
또한 VideoLISA/VISA 등 MLLM 기반 방법들은 언어-비전 정렬과 추론을 강화했지만, 행동/의사결정 전 단계의 predictive reasoning과 그에 따른 마스크 예측을 직접 평가하기 어렵다는 한계가 있었다.

- **Core Contribution**: 논문은 관찰 구간에서 시각 단서를 받아 “향후 영상 구간에서 어떤 객체가 등장/사용될지”를 묻는 Foresight Expression Video Object Segmentation(FeVOS)라는 새로운 태스크를 제안한다. 모델은 관찰 프레임에 대한 마스크를 답으로 내되, 질문은 미래 사건을 지칭해 텍스트-비주얼 정렬을 어렵게 만든다.
이를 위해 FeVOS 데이터셋(968개 클립, 14,525개 foresight expressions, 2,904개 chain-of-thought 주석)과 이를 학습하는 FeVOS-R1(Multimodal LLM 기반)을 함께 제시하며, 기존 RVOS 벤치마크로의 일반화도 확인한다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 질문 정보가 미래에 있어 현재 시각과의 직접 매칭이 불가능하다는 점, (2) 시간적 맥락과 공간 단서를 동시에 조합해 관련 객체를 식별해야 한다는 점, (3) 결국 관찰 프레임에서 픽셀 단위 마스크를 정확히 출력해야 한다는 점이다. 또한 기존 fine-tuning만으로는 예측적 추론에서의 미세한 정렬이 약해질 수 있다.
논문은 2단계 파이프라인으로 이를 완화한다: 먼저 CoT 기반 supervised fine-tuning으로 추론 형식을 학습하고, 이어서 Group Relative Policy Optimization(GRPO)로 IoU 기반 end-to-end 보상을 직접 최대화해 segmentation 품질과 추론을 더 강하게 정렬한다.

- **Empirical Impact**: 실험에서 FeVOS-R1은 FeVOS에서 2단계 학습을 모두 적용했을 때 42.3(J&F 평균)을 달성하며, SFT 기준(35.8) 대비 추가 향상을 보였다. 특히 zero-shot과 기존 RVOS 지향 방법들은 FeVOS의 미래 예측형 정답에 크게 취약해, 해당 태스크의 난이도와 기여의 필요성이 드러난다.
또한 ReVOS/MeViS로의 일반화 평가에서 추론 중심 서브셋 성능이 개선되는 등 cross-domain 일반화가 관찰되어, predictive reasoning을 비디오 분할로 확장한 연구 방향성을 제시한다.



### H-Adapter: Pose-Robust Hairstyle Transfer via Attention-Derived, Source-Aligned Hair Masks (https://arxiv.org/abs/2606.25578)
Comments:
          Accepted at ECCV 2026. Project page: this https URL

- **Prior Approaches**: 헤어스타일 전이는 가상 착용(virtual try-on) 같은 실용성이 크지만, 소스와 레퍼런스 사이 헤드 포즈 차이가 클 때 결과가 쉽게 흔들립니다. 기존 방법들은 보통 단일 손실로 학습하거나, 포즈 차이를 직접 다루지 못해 헤어와 비헤어(배경·피부 등)가 섞인 편집이 발생하기 쉽습니다.

- **Core Contribution**: 이 논문은 포즈 불일치에 강한 헤어 전이를 위한 H-Adapter를 제안합니다. 헤어와 비헤어 목적을 분리하는 region-specific loss로 spatially disentangled cross-attention을 유도하고, 여기서 소스 정렬 기반 hair edit mask를 뽑아 diffusion 기반 inpainting에 가이드로 사용합니다.

- **Technical Challenges**: 핵심 난제는 포즈가 달라질 때도 헤어 영역만 정확히 편집하도록 학습을 설계하는 것입니다. H-Adapter는 region-specific loss로 교차 주의를 위치별로 분해해 마스크 품질을 높이고, 그 마스크로 inpainting이 헤어에만 집중하게 만들어 비헤어 보존을 함께 달성합니다.

- **Empirical Impact**: 실험에서는 pose-agnostic 및 pose-different 하위셋에서 포즈 차이 상황에도 FID, FID_CLIP, CLIP-I 등 주요 지표가 우수하며 최고 성능을 보고합니다. 또한 비헤어 보존은 경쟁 수준을 유지하면서 레퍼런스의 미세한 헤어 디테일 충실도가 정성적으로 개선됐고, VLM-as-a-judge 프로토콜에서도 hairstyle faithfulness·non-hair preservation·artifact quality 전반에서 일관된 향상이 관찰됩니다.



### Concept Removal for Frontier Image Generative Models (https://arxiv.org/abs/2606.25548)
Comments:
          Accepted at ICML2026

- **Prior Approaches**: 기존 컨셉 제거는 크게 내부 방식(학습 기반 미세조정, 또는 closed-form 가중치 편집)과 외부 방식(추론 시 개입 모듈 추가/결합, SAE 기반 모듈 등)으로 나뉜다. 내부 방식은 계산비용과 아키텍처·학습 패러다임 의존성이 크고, closed-form은 강하게 지우면 품질 저하가 생기며 연속·다중 제거에서 누적 왜곡 문제가 보고된다. 외부 방식은 white-box 접근에서 모듈을 우회하거나 제거할 수 있어 영구성 측면에서 한계가 있다.

- **Core Contribution**: 논문은 BLOCK(Bottleneck-Layer-Oriented-Concept-Knockout)라는 개념 제거 프레임워크를 제안한다. 핵심은 SD3.5, FLUX, Infinity 같은 frontier diffusion/IAR 모델이 공통으로 갖는 텍스트-백본 중간 병목층(bottleneck)을 “transcoder(트랜스코더)”로 교체해, 모델 내부에서 선택적으로 유해 컨셉 신호만 차단하면서 나머지 동작은 유지하는 것이다. add-on 안전 모듈처럼 외부에 붙이지 않고 백본에 통합해 white-box에서도 제거 효과가 지속되도록 설계했다.

- **Technical Challenges**: 기여를 실현하기 위해 (1) 원래 병목층의 기능을 최대한 보존하면서 (2) 컨셉별로 분리되는 activation 피처를 만들고 (3) 컨셉에 해당하는 latents만 정확히 비활성화해야 한다는 문제가 있다. 이를 위해 transcode를 학습할 때 단순 ℓ1 기반 희소화가 만드는 경계 불명확/Dead latent 문제를 TopK 기반으로 해결하고, straight-through estimator로 학습 가능성을 확보했다. 컨셉 토큰에 대응하는 활성 latent 집합을 식별한 뒤, 디코더 가중치의 해당 컬럼을 empty token의 출력으로 리다이렉트해 컨셉 기여를 차단하는 in-place 개입을 수행한다.

- **Empirical Impact**: UnlearnCanvas를 바탕으로 style/object 제거를 SD3.5, FLUX.1-dev, Infinity-2B/8B 전반에서 평가했으며, BLOCK은 확산과 자기회귀 양쪽에서 SOTA 컨셉 제거 성능을 보였다고 보고한다. 특히 FLUX(12B)에서는 style removal 정확도가 기준 대비 최대 21%p 향상되는 등 품질 보존(retain)과 교차 도메인 유지 성능도 높게 나타났다. 또한 연속적인 다중 컨셉 제거와 adversarial prompting 상황에서도 강건함을 보이며, 시퀀셜 편집에 취약한 기존 방식 대비 실용성을 강화한다.



### Efficient Cross-Scale Invertible Hiding Network with Spatial-Frequency Collaboration and Non-Invertible Mechanism (https://arxiv.org/abs/2606.25547)
Comments:
          IEEE TNNLS submitted by Junxue Yang, Xin Liao (this https URL)

- **Prior Approaches**: 기존 image hiding은 autoencoder 기반이 은닉·복원을 각각 따로 학습하는 경향이 있고, INN 기반은 은닉과 복원을 이미지 변환의 역문제로 보고 forward/backward를 통해 이론적으로 정보 손실이 적게 설계된다. 하지만 INN 계열은 단일 scale과 단일 domain(예: spatial 또는 wavelet) 중심이라 다양한 해상도/주파수 협업을 충분히 못 하고, 엄격한 invertibility 제약으로 nonlinear 표현력이 제한된다는 한계가 있다. 그 결과 stego와 복원 비밀영상 모두에서 잔차가 커지거나 품질이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 cross-scale invertible hiding 네트워크 CrosInv를 제안해, scale 간 표현 학습과 공간-주파수 협업을 동시에 끌어올리면서도 표현력은 강화한다. CrosInv는 비가역적 nonlinear 강화 모듈인 NCDM과, cross-scale와 spatial-frequency 특성을 뽑는 invertible 모듈 CIM을 forward/backward에 대칭 배치한다. 이를 통해 기존 single-scale/단일 domain INN의 성능 병목을 직접 겨냥한다.

- **Technical Challenges**: INN 기반에서 cross-scale 협업을 넣으려면 해상도 변환을 하면서도 역변환 가능성을 유지해야 하고, 동시에 spatial과 frequency 정보를 자연스럽게 결합해야 한다. 저자들은 CIM 내부에서 pixel shuffle(PS)·Haar wavelet transformation(WT) 및 그 역연산을 조합해 공간 내용과 주파수 정보를 함께 주입하면서 cross-scale 표현을 bijective하게 구성했고, 중간 scale에서 U-Net 스타일 split/concat skip connection으로 비국소 cross-scale 정보를 더했다. 또한 NCDM을 CIM의 forward와 inverse 양쪽에 대칭으로 넣어 invertibility 제약을 완화하고 nonlinear 표현을 늘리는 방식으로 비선형성을 보강했다.

- **Empirical Impact**: 실험은 COCO/ImageNet/BOSSBase 3개 벤치마크에서 수행됐으며, PSNR/SSIM/APD/LPIPS와 복원 품질에서 CrosInv가 경쟁 방법 대비 큰 폭의 개선을 보였다(은닉 PSNR은 약 11~18 dB 수준 격차). steganalysis에서도 SCRMQ1, UCNet, PENet 기반 탐지 정확도가 무작위(50% 근처)로 가까워져 보안성이 향상된 결과를 제시한다. 계산 비용은 FLOPs/params/추론 시간 관점에서 최상급은 아니지만 ‘상태-of-the-아트 성능 대비 상대적으로 낮은 오버헤드’로 평가되며, ablation으로 cross scale, PS/IPS와 WT/IWT 협업, NCDM의 역할을 확인했다.



### Disease-Centric Vision-Language Pretraining with Hybrid Visual Encoding for 3D Computed Tomography (https://arxiv.org/abs/2606.25546)
Comments:
          ICML 2026

- **Prior Approaches**: 기존 Vision-language pre-training(VLP)은 방사선 보고서를 언어 감독으로 삼아 학습하지만, 3D CT에는 비효율적 시각 백본과 정밀하지 않은 이미지-문장 정렬이 문제가 됐다. 특히 ViT를 3D에 그대로 확장하면 패치화/연산 비용이 커지고 작은 병변의 해상도 손실이 발생하는 반면, 3D CNN 계열은 cross-modal priors 활용과의 호환성이 떨어졌다. 또한 organ-level 전역 대조학습 위주 접근은 같은 장기 안의 서로 다른 질병을 ‘abnormal’처럼 뭉개 거친 의미 정렬을 초래한다.

- **Core Contribution**: 이 논문은 CT용 질병 중심 VLP 프레임워크 CT-DiagVLM을 제안해, 3D 효율적인 표현 학습과 질병 단위 미세 정렬, 그리고 zero-shot 추론 신뢰도를 함께 끌어올린다. 핵심은 (1) ViT patch embedding을 3D CNN 기반 멀티스케일 토큰화로 교체하는 CNN–ViT 하이브리드 인코더, (2) learnable query token으로 질병별 의미를 동적으로 뽑아 질병-비교대조를 수행하는 메커니즘, (3) 수작업 템플릿 대신 실제 임상 구문과 질병 prototype을 이용해 pre-training-inference gap을 줄이는 diagnosis-aware prompt 전략이다.

- **Technical Challenges**: 가장 큰 기술 난제는 3D CT에서 국소 병변 신호는 놓치지 않으면서도 ViT가 제공하는 전역 attention/사전정렬 호환성을 유지하는 설계였다. 저자들은 3D ResNet-18의 멀티스케일 특징을 MSF-PE로 융합해 pre-trained ViT 가중치(SIGLIP-2)와 연결하고, 장기 마스크는 TotalSegmentator 대신 보조 U-Net이 예측하도록 end-to-end로 학습시켰다. 또 ‘같은 장기 내 동시 질병’을 분리하기 위해 보고서 문장 전체에서 cross-attention 기반으로 질병 상태별 텍스트를 추출하고, ground-truth 및 LLM pseudo-label을 함께 써 질병 단위 대조학습과 cosine similarity 기반 prototype 추론을 일치시켰다.

- **Empirical Impact**: 실험에서 CT-DiagVLM은 CT-RATE에서 AUC 84.4%(기존 대비 +5.1%), Rad-ChestCT에서 AUC 75.4%(+5.4%)를 기록하며 외부 일반화 성능을 입증했다. 특히 Qwen3-Max로 60개 질병 설정(라벨 노이즈/난이도 상승)에서는 AUC가 +9.8%p 더 크게 개선돼 질병 단위 미세 정렬의 효과가 드러났다. 추가로 hold-out 실험을 통해 학습에서 완전히 제외된 병리를 대상으로도 견고한 진단 성능을 유지했으며, 보고서 생성 downstream 전이성도 보여 임상 활용 범용성까지 확장되는 신호를 제시한다.



### TensorLDM: A Component-Wise Latent Diffusion Model for Volumetric DTI Reconstruction from Sparse DWIs (https://arxiv.org/abs/2606.25545)
- **Prior Approaches**: 희소한 DWIs로 6성분 확산 텐서를 복원하는 문제는 역문제로, 기존 딥러닝은 종종 3D 연속성을 무시한 2D 슬라이스 단위 처리로 인해 절편 간 불일치를 만든다. 또한 SuperDTI·Diff-DTI처럼 FA/MD/RD·Color FA 같은 스칼라 맵을 합성하는 방식은 텐서 고유벡터의 방향 정보를 잃어 tractography에 필요한 기하/방향 일관성이 약해질 수 있다. 여기에 생성 모델이 SPD(대칭 양의 정부호) 제약을 직접적으로 잘 강제하지 못하면 물리적으로 불가능한 텐서가 나올 위험도 남아 있었다.

- **Core Contribution**: TensorLDM은 6성분 확산 텐서를 구성요소별로(대각/비대각) 처리하는 component-wise latent diffusion 모델로, 각 성분 특성을 반영하면서도 DWI 조건을 공유해 해부학적 일관성을 유지한다. Anatomy-Conditioned Autoencoder로는 해부학 정보를 재인코딩하기보다 텐서 성질에 집중하도록 유도하고, Cross-Component Attention(CCA)으로 성분 간 의존성을 학습해 텐서 전체의 정합성을 강화한다. 또한 MoE DWI conditioner로 성분마다 다른 DWI 정보 활용 방식을 학습하고, Log-Euclidean Metric 기반 제약까지 포함해 SPD 준수성을 목표에 반영한다.

- **Technical Challenges**: 가장 큰 기술적 도전은 희소·단일 셸(싱글쉘) 4볼륨 DWI만으로 6개 성분이 각 보셀에서 모두 SPD 제약을 만족하도록 만드는 것이다. TensorLDM은 다단계 학습으로 먼저 고품질 텐서 latents를 만들고(개입: Anatomy-Conditioned Autoencoder), 이후 각 성분 latents에 노이즈를 더하는 diffusion fine-tuning에서 CCA와 FiLM 기반 DWI 조건 주입을 함께 적용해 성분 간 일관성과 SPD coherence를 동시에 끌어올린다. 특히 Log-Euclidean loss(LEM)를 통해 텐서가 텐서-다양체 위에서 기하적으로 타당하도록 완충 장치를 제공한다.

- **Empirical Impact**: HCP 데이터에서 single-shell, four-volume 희소 획득 설정으로 평가한 결과, TensorLDM은 downstream tractography와 텐서 재구성 정확도에서 가장 우수하거나 동급이며, SPD violation rate가 거의 기준선 수준으로 낮게 보고됐다. Log-Euclidean Metric(LEM)과 probabilistic tractography의 Tract Core Distance(TCD) 등 텐서-기하/기능 수준 지표에서도 개선이 일관되게 나타나, 단순 PSNR/SSIM 중심 평가지표의 한계를 보완한다는 메시지를 뒷받침한다. 즉, 임상 조건에 가까운 짧은 입력으로도 물리적으로 그럴듯한 텐서를 생성해 섬유 추적 유틸리티까지 확보하는 방향의 진전을 보여준다.



### SAC$^2$-Net: Semantic Anchoring and Complementary-Consensus Fusion for Multimodal Micro-Expression Recognition (https://arxiv.org/abs/2606.25542)
- **Prior Approaches**: 기존 MER은 optical flow로 국소 동작 변위와 방향을 포착하거나, motion magnification으로 약한 외양 변화를 증폭하는 방식이 주류였다. 한편 multimodal MER도 두 시각 모달리티를 단순 결합·정렬하는 접근이 많아, 모달리티별 실패(잡음/왜곡/무정보 구간)를 “신뢰도” 관점에서 모델링하지 못했다.

- **Core Contribution**: 이 논문은 optical flow와 motion magnification이 비대칭적으로 실패하는 현상을 정량적 융합 이슈로 재정의하고, 이를 reliability-aware fusion 이점으로 바꾸는 SAC2-Net을 제안한다. 핵심은 SASA(Semantic Anchoring Soft Alignment)로 AU 정보를 텍스트 프롬프트로 앵커링해 모달리티 간 이질성을 줄이고, 이후 CCF(Complementary-Consensus Fusion)로 영역별 보수-합의 기반 복원+공간 합의를 수행하는 구조다.

- **Technical Challenges**: 첫째, 두 모달리티는 통계·표현 형태가 달라 fusion 전에 cross-modal heterogeneity를 줄여야 하며, 둘째 AU는 감정 카테고리와 1:1 대응이 아니라 AU가 겹치거나 해부학적으로 가까운 샘플을 hard negative로 밀어내면 의미 구조가 깨진다. 논문은 SASA에서 AU를 텍스트 프롬프트로 바꿔 동작-외양을 AU 의미 공간에 soft하게 정렬하고, AU 계층·좌우 일치성을 반영한 soft Jaccard 유사도로 soft labels를 구성해 의미 근접성을 보존한 뒤 CEM/CRM이 신뢰도 맵을 학습해 국소 복원과 shared spatial consensus를 달성한다.

- **Empirical Impact**: 5개 MER 벤치마크(CASME II, SAMM, SMIC, MEGC2019-CD, CAS(ME)3, DFME)에서 coarse/fine-grained, 대규모, cross-dataset 평가 전반에 걸쳐 SOTA 또는 준SOTA 성능을 보였다. 또한 external macro-expression 데이터(CK+)로 SASA 기반 사전학습을 적용하는 전이 설정에서도 견고함을 확인해, 모달리티 실패 비대칭을 신뢰도 기반 융합으로 다루는 접근이 실제 성능 향상으로 이어짐을 시사한다.



### Spatio-Temporal Mixture-of-Modality-Experts Diffusion for Quantitative DCE-MRI Synthesis from Incomplete MR Sequences (https://arxiv.org/abs/2606.25535)
- **Prior Approaches**: 기존 DCE-MRI 합성 연구는 다른 MRI로 Ktrans, vp, ve 같은 정량 파라미터 맵을 예측하지만, 대부분 고정된 입력 모달리티가 온전히 관측된다고 가정해 실제 임상에서의 결측·이질성에 취약했다. 또한 two-stage 결측 대치 파이프라인은 초기 단계 오차가 누적되기 쉽고, shared embedding 계열은 모달리티별 미세한 정량 신호를 희석해 정확도가 떨어질 수 있다. 조건부 diffusion이 있어도 결측 모달리티 조합 전반에 대한 강건성이나 voxel-wise 정량 정밀도 목표까지는 충분히 최적화되지 않았다는 한계가 있었다.

- **Core Contribution**: ST-MoME는 Spatio-Temporal Mixture-of-Modality-Experts(ST-MoME)라는 conditional diffusion 프레임워크로, 임의의 모달리티 부분집합만 주어져도 3D DCE 파라미터 맵을 생성한다. 핵심은 모달리티별 expert feature를 spatio-temporal gating 네트워크로 voxel 단위·시간 단계별(denoising timestep별)로 가중 결합해, reverse diffusion 과정 내내 적절한 모달리티를 선택적으로 활용하는 점이다. 또한 정량적 충실도를 위해 latent 압축 대신 image-space diffusion을 채택해 파라미터 맵의 voxel-wise 정확성을 보존하는 방향을 택했다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 결측 모달리티 패턴과 (2) 3D 정량 정확도 요구를 동시에 만족시키는 것인데, 특히 full-volume 확산은 계산·메모리 부담이 커서 patch 기반 학습이 필요했다. 하지만 patch 학습은 국소 문맥과 전역 일관성 사이에 receptive-field 불일치를 만들 수 있어, ST-MoME는 denoising U-Net에 3D Swin Transformer 블록을 넣어 shifted-window self-attention으로 계층적 전역 문맥 전파를 강화했다. 이와 함께 게이트는 modality-availability mask와 noisy diffusion 상태 및 timestep 임베딩을 함께 받아, logit-level 마스킹으로 결측 모달리티는 softmax에서 완전히 제외되도록 설계했다.

- **Empirical Impact**: 임상 뇌종양 코호트 386명(시험 126명)에서 16개의 모달리티 가용 시나리오(Leave-One-Out/Two-Out/Three-Out 포함)로 평가했으며, ST-MoME는 세 DCE 파라미터(Ktrans, vp, ve) 전체를 합산한 평균 NMSE에서 최저 성능을 보였다. 특히 vp와 ve에서 모든 시나리오에서 1위를 차지했고, Ktrans에서는 경쟁적인 결과를, 종양 ROI에서는 임상적으로 중요한 영역에서 가장 낮은 재구성 오차를 보였다. 추가로 gating dynamics의 후분석은 구조-초기 및 생리-후기 융합 흐름이 임상의 직관과 일치하는 방식으로 학습됐음을 보여주며, 실제 배치 환경에서의 활용 가능성을 뒷받침한다.



### PatchINR: Patch-Based Implicit Neural Representations for Efficient and Scalable Inferenc (https://arxiv.org/abs/2606.25534)
- **Prior Approaches**: 기존 INR(Implicit Neural Representation)은 좌표를 픽셀 단위로 질의해 영상을 재구성하는데, 이 방식은 해상도가 커질수록 추론 쿼리 수가 급격히 늘어 계산 비용이 이차적으로 증가한다. 성능을 높이려는 경량 네트워크, frequency encoding, 하이브리드(그리드+신경망) 같은 소프트웨어 최적화도 결국 픽셀별 질의라는 근본 패러다임을 크게 바꾸지 못해 한계가 남는다. 또한 FPGA 같은 하드웨어 가속에서도 픽셀 단위의 순차·불규칙 메모리 접근이 병목이 되어 병렬 처리 장점을 충분히 끌어내기 어렵다.

- **Core Contribution**: 이 논문은 INR의 질의 단위를 픽셀에서 비중첩 패치로 바꾸는 patch-based INR을 제안한다. 즉, 연속성을 유지한 채 한 번의 forward pass로 n×n 픽셀을 통째로 예측해 전체 추론 쿼리 수를 크게 줄인다. 동시에 해당 모델을 겨냥해 FPGA에서 동작하는 하드웨어-알고리즘 공동설계를 제공한다.

- **Technical Challenges**: 패치로 전환하면 연산량 자체는 줄지만, 하드웨어 관점에서는 좌표/가중치 로딩과 파이프라인 공급이 병목이 되기 쉽다. 논문은 URAM/BRAM 기반 온칩 버퍼와 데이터 관리 유닛을 두어 좌표 큐와 가중치 큐를 동기화하고, 2D 병렬 처리 엘리먼트 배열로 패치 내부 연산을 동시에 처리하도록 설계했다. 또한 FP32와 INT8 듀얼 정밀도 모드를 지원하고, 구성 가능한 깊은 파이프라인으로 처리량을 끌어올렸다.

- **Empirical Impact**: 실험에서는 SIREN/WIRE/FINER를 Kodak과 DIV2K에서 평가했으며, 패치 크기를 키울수록 PSNR/SSIM이 일관되게 개선되되 선형적이지는 않다고 보고한다. 하드웨어 결과로는 FPGA 기반 patch-based INR이 2×2 패치에서 픽셀 단위 대비 추론 지연을 75% 줄이면서도 34.97 dB PSNR을 달성했고, 파라미터 오버헤드는 0.6%에 그쳤다. 영상 모달리티 확장 실험에서도 프레임별 질의의 비효율을 줄이면서 재구성 품질과 학습 효율을 함께 개선할 수 있음을 보여준다.



### C2RM-Seg: Causal Counterfactual Reasoning with Structural-Semantic Priors for Weakly Supervised Histopathological Tissue Segmentation (https://arxiv.org/abs/2606.25508)
Comments:
          11 pages, 3 figures. Code is available at this https URL

- **Prior Approaches**: 기존 WSSS(Weakly Supervised Semantic Segmentation) 계열은 CAM으로 거친 시드를 만든 뒤 이를 pseudo-mask로 학습하는 방식이 주류다. Wave-aware CAM, UAM, UM-CAM 등은 activation enhancement·uncertainty modeling으로 개선을 꾀했지만, 염색(staining) 같은 외관 단서와 배경/획득 요인이 원인처럼 섞여 구조적 일관성이 떨어질 수 있다. 또한 MLPS/ESFAN 같은 구조 모델링이나 OEEM 같은 noise-aware 최적화는 도움이 되더라도, 근본적으로 편향된 인과 정렬을 직접 교정하지 못하면 한계가 남는다.

- **Core Contribution**: 논문은 C2RM-Seg를 제안하며, 핵심은 CAM의 편향을 ‘원인-교란’ 관점에서 다루는 인과 pseudo-label 정제다. 분류 단계에서 Causal Counterfactual Reasoning Module(C2RM)이 Structural Causal Model(SCM)을 기반으로 confounder를 억제하는 counterfactual intervention을 수행해 morphology-aligned CAM을 만든다. 이어 segmentation 단계에서는 Dual-Path Structural–Semantic Architecture로 ResNeSt 기반 국소 구조와 frozen DINOV3(DeiT 계열) 기반 전역 의미 priors를 cross-path gating으로 결합한다.

- **Technical Challenges**: 히스토패쓸로지에서 라벨-관련 형태 신호가 염색 변화, 스캐너 반응, 조직 처리 아티팩트 등 잠재 confounder와 공변하며 CAM이 맥락에 끌리는 점이 기술적 난관이다. 논문은 factor subspace로 잠재 요인을 분해하고, learned causal structure matrix로 부모 요인의 기여를 추정·차감해 counterfactual factor를 만들며, 이를 gated residual correction으로 공간 특징에 주입한다. 추가로 pseudo-label의 잔여 잡음과 인터페이스 불확실성을 다루기 위해 Uncertainty-Gated Margin(UGM) loss로 margin enforcement와 confidence learning의 가중치를 예측 불확실도에 따라 동적으로 전환한다.

- **Empirical Impact**: BCSS 2개 벤치마크에서 C2RM-Seg는 state-of-the-art를 달성했다. 예를 들어 BCSS-WSSS에서 72.17% mIoU와 HD95 27.31로 ESFAN을 앞섰고, LUAD-HistoSeg에서도 79.62% mIoU와 18.07 HD95를 기록했다. ablation 결과는 C2RM의 인과 기반 deconfounding, Dual-Path 구조-의미 결합, UGM이 각각 영역 겹침과 경계 품질을 보완하며 시너지로 최종 성능을 끌어올린다는 점을 확인시킨다.



### HG-Bench: A Benchmark for Multi-Page Handwritten Answer-Region Grounding in Automated Homework Assessmen (https://arxiv.org/abs/2606.25491)
- **Prior Approaches**: 기존 referring-expression grounding 벤치마크는 주로 자연 이미지에서 단일 타깃을 찾는 과제에 초점이 맞춰져, 과제별 답/추론 단계를 계층적으로 박스 출력하는 요구가 제한적입니다. 문서 이해 연구는 텍스트·레이아웃을 다루지만, 페이지 단위의 좌표 의미와 ‘정답 영역-단계 영역’ 순서/포함 제약을 갖춘 지역 그라운딩은 잘 측정되지 않았습니다. 또한 수학·과학 풀이형 벤치마크는 관련 내용이 이미 식별됐다는 가정 아래 평가가 이뤄져, 실제 채점 파이프라인의 전 단계(공간 로컬라이제이션) 부족을 드러내지 못했습니다.

- **Core Contribution**: 이 논문은 K-12 필기 숙제에서 페이지를 고려해 ‘완전한 정답 영역’과 그 안의 ‘순서가 있는 단계별 서브 영역’을 동시에 찾는 page-aware, two-level answer-region grounding을 새 평가 설정으로 제시합니다. 이를 위해 1,489,278개 소스 풀에서 선별한 500개 인적 라벨 테스트(총 10,420개 학습)로 구성된 HG-Bench를 공개합니다. 질문-단계 박스가 계층적 포함 제약으로 연결되도록 설계하고, complete answer localization(FA)과 step-level decomposition(FSm)을 분리 측정해 공간적 추론 구조를 실제로 그라운딩하는지 검증합니다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 (1) 여러 페이지 스캔에서 페이지별 좌표 의미를 유지하면서, (2) 잡음·그림자·필기체 변이를 견딜 정도로 정답/단계 영역을 박스로 정확히 국소화하고, (3) 단계들을 학생 작성 순서대로 정렬하며, (4) ‘단계 박스는 부모 정답 박스 내부에 완전히 포함’되도록 계층 일관성을 보장하는 것입니다. 저자들은 페이지 단위 1:1 매칭과 IoU 기반 필터링으로 FA와 FSm을 공정하게 집계하는 평가 프로토콜을 구성하고, JSON 스키마·좌표계 변형을 다루는 재시도/보정 절차까지 포함해 모델 출력 형식 문제를 통제합니다. 또한 GLM-4.6V 9B를 HG-SFT 약 1만 예제로 단일-stage SFT한 참조 모델을 제공해, 벤치마크가 학습으로 실제 개선 가능한지 하한(lower bound) 관점에서 확인합니다.

- **Empirical Impact**: 실험에서 frontier closed-source API와 경쟁 open-weight VLM들은 zero-shot 기준 FA 55.22%, FSm 48.22%를 넘지 못했지만, HG-SFT로 미세조정한 GLM-4.6V 9B 참조 모델은 FA/FSm 74.97/72.26까지 도달했습니다. 특히 step-level 성능 하락이 급격해 ‘정답 영역 대강 찾기’보다 ‘순서 있는 단계 구조를 그라운딩’하는 능력 격차가 핵심임을 보여줍니다. 결과적으로 HG-Bench와 page-aware 평가 프로토콜, 학습된 참조 체크포인트가 자동 채점/추론 추적에서 필요한 구체적 역량을 재현 가능하게 측정하는 출발점이 될 것으로 기대됩니다.



### Cross-View Variance Correlation in Path-Traced Stereo:A Hidden Shortcut in Synthetic Training Data (https://arxiv.org/abs/2606.25483)
- **Prior Approaches**: 대부분의 스테레오 매칭 네트워크는 path-traced 합성 데이터를 학습에 활용하며, MC noise가 좌/우에서 독립인 i.i.d. 잡음처럼 작동한다고 가정한다. 하지만 네트워크는 샘플 단위가 아닌 이미지의 집계된 픽셀 통계를 사용하므로, 잡음과 구분되는 per-pixel variance field의 좌우 관계는 그동안 충분히 조명되지 않았다. 기존 접근은 주로 첫순위(평균) 통계가 실제와 유사하다는 실용적 가정에 의존해왔다.

- **Core Contribution**: 이 논문은 MC-rendered 스테레오 데이터에서 좌/우 카메라의 MC noise 스트림은 독립이지만, ground-truth disparity로 정렬한 variance fields는 높은 상관을 보인다는 점을 새롭게 정량화한다. 정렬된 warped Pearson correlation이 ρ=0.754±0.016 (20개 씬, SPP=512)까지 도달하며, 대표 씬에서는 SPP를 16배 범위로 늘려도 사실상 불변(ρ=0.778±0.001)임을 보여준다. 또한 잔여-shuffle 개입으로 이 상관 구조가 cost-volume에서 실제로 매칭 큐로 기능함을 인과적으로 확인한다.

- **Technical Challenges**: 핵심 난제는 ‘독립 잡음’ 가정이 성립하더라도, 이미지 기반 학습에서 어떤 종류의 신호(특히 variance)가 좌/우 정렬 후 구조를 갖는지 분리해 측정하는 것이다. 논문은 ground-truth disparity warp로 좌표를 맞춘 뒤 샘플 수에 따른 크기 변화가 지표를 왜곡하지 않도록 Pearson 상관을 사용하고, Lambertian vs glass처럼 재질별로 Ω를 분리해 원인 신호를 추적한다. 더 나아가 seed 평균으로부터 잔차를 만든 뒤 residual blocks를 shuffle해 ‘정렬 정합성만’ 파괴하는 개입을 설계해 매칭 큐로서의 역할을 검증한다.

- **Empirical Impact**: Lambertian 영역에서는 ρ≈0.78로 강한 반면 glass에서는 ρ≈0.30으로 약해, integrand 분해(뷰-독립 vs 뷰-의존 성분)가 상관의 재질 의존성을 예측한다는 그림을 제시한다. residual-shuffle로 좌/우 정렬을 깨면 비유리 영역의 GT cost margin이 33% 악화되고, glass에서 variance-based winner-take-all 정확도는 4.3× 감소해 variance 정렬 구조가 매칭에 직접 기여함을 보여준다. 이 신호는 실제 양안 센서의 열/shot noise와 달리 MC-rendered 데이터에 고유하며, sim-to-real gap을 만드는 ‘shortcut’ 후보로서 향후 네트워크 학습에서의 의존도와 완화 전략이 요구된다.



### TACO: Towards Task-Consistent Open-Vocabulary Adaptation in Video Recognition (https://arxiv.org/abs/2606.25478)
- **Prior Approaches**: CLIP을 기반으로 한 비디오 오픈-보캐뷸러리 적응은 Kinetics-400 같은 데이터로 파인튜닝하며, 보통 일반화와 특수화를 trade-off하기 위해 추가 정규화나 제약을 설계해 왔다. 다만 기존 방법들은 최적화 목표가 파인튜닝 데이터(ID)에만 갇혀 있어, 학습 분포 밖(OOD)에서 표현이 어떻게 흔들리는지에 대한 직접적인 제약이 약하다고 지적한다. 그 결과 시각-텍스트 정렬이 OOD 영역에서 깨져 미지 범주의 일반화가 떨어질 수 있다는 문제를 제기한다.

- **Core Contribution**: 이 논문은 파인튜닝과 평가 목표 간 불일치로 인해 OOD 표현 편차와 정렬 shift가 생기며, 이것이 성능 저하의 핵심 원인이라고 분석한다. 이를 완화하기 위해 TACO를 제안하며, 적응 과정에서 학습 분포 밖에서도 OOD-relevant 정렬을 보존해야 한다는 원칙을 제시한다. Relative Structure Distillation으로 표현 공간의 상대 기하 구조를 유지하고, Specialization Projection으로 테스트 시 쓰이는 표현이 과도하게 특수화되지 않게 분리한다.

- **Technical Challenges**: 핵심 기술 난제는 OOD 의미를 실제 데이터로 직접 모델링하기 어렵다는 점인데, TACO는 이를 우회해 OOD를 명시적으로 감독하지 않으면서도 표현 공간 전체의 정렬 구조를 규제한다. Relative Structure Distillation은 ID와 연결된 constructed OOD 공간의 기하 앵커를 만들어, teacher 기준의 상대 구조가 파인튜닝 중에도 유지되도록 한다. 또한 Specialization Projection으로 최적화 공간과 표현 공간을 분리해 cross-entropy가 공유 표현을 직접 과적합/왜곡시키는 경향을 줄이며, 투영 헤드는 학습 후 평가에서 버린다.

- **Empirical Impact**: 실험은 cross-dataset 및 base-to-novel 프로토콜에서 여러 벤치마크에 걸쳐 일관된 개선과 함께 state-of-the-art 성능을 보이며, 특히 UCF와 HMDB에서 novel 클래스 인식이 크게 향상된 것으로 보고된다. ablation에서는 Specialization Projection과 Relative Structure Distillation이 각각 alignment shift 억제와 구조적 정규화를 통해 상보적으로 작동하며 조합 시 시너지 효과가 있음을 확인한다. 또한 의미 있는(LLM/웹 기반) 앵커보다 CLIP hypersphere 위의 random geometric anchors가 더 유리하며, 무작위 행렬은 collapse를 유발해 구조적 유효성이 중요함을 보여준다.



### Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models (https://arxiv.org/abs/2606.25473)
Comments:
          Technical Report

- **Prior Approaches**: 자기회귀(AR) 비디오 diffusion은 프레임/청크 단위로 causal attention을 쓰며 스트리밍·인터랙티브 생성에 유리하지만, teacher-forcing(TF)과 diffusion-forcing(DF)류는 노출 편향(exposure bias)으로 시간이 지날수록 품질이 흔들릴 수 있다. 이를 줄이기 위해 self-forcing(SF)과 distribution matching distillation(DMD) 또는 GAN 목적을 결합한 방법들이 등장했지만, 이러한 on-policy 최적화는 초기화에 민감하고 mode collapse 위험이 있어 안정적인 시작 전략이 핵심 과제로 남아 있었다.

- **Core Contribution**: 이 논문은 rCM(score-regularized consistency model)의 forward-reverse 상보성 철학을 AR 비디오 diffusion으로 확장해 Causal-rCM을 제안한다. TF를 forward-divergence(초기 mode coverage를 돕는 구성요소)로 보고, SF의 DMD를 reverse-divergence(추론 시 few-step 분포를 직접 다듬는 구성요소)로 대응시켜, 초기화-인과 학습-증류 손실의 시너지를 하나의 관점에서 정리한다.

- **Technical Challenges**: AR 설정에서 few-step causal distillation을 안정적으로 만들려면 TF/SF/DMD 조합이 어떻게 초기화에서 학습 분포를 잡아주는지, 그리고 연속시간(continuous-time) consistency model을 빠르게 구현할 수 있는지가 어려움이다. 저자들은 teacher-forcing 기반 continuous-time CM(sCM, MeanFlow)을 custom-mask FlashAttention-2 JVP 커널로 구현해 dCM 대비 10배 빠른 수렴을 달성하고, Causal-rCM을 diffusion distillation과 causal training 전반에 적용 가능한 통합 레시피 형태로 제시한다.

- **Empirical Impact**: 대규모 실험에서 teacher-forcing CM이 self-forcing DMD의 최적 초기화 전략임을 보이며, frame-wise와 chunk-wise 스트리밍 생성 모두에서 합성 데이터만으로 SOTA급 성능을 보인다. 특히 2-step causal Wan2.1-1.3B distilled 모델이 sampling step 1~2개로 VBench-T2V 84.63을 기록했고, Causal-rCM을 Cosmos 3(행동 조건의 omnimodal 물리 AI world foundation model)에도 적용해 인터랙티브 world model 성능을 확장한다.



### EchoStyle: Unlocking High-Fidelity Video Stylization with Reverse Data Synthesis (https://arxiv.org/abs/2606.25465)
- **Prior Approaches**: 기존 비디오 스타일화는 참조 이미지(또는 키프레임)를 style prior로 두는 방식이 주를 이뤘지만, 불필요한 정보가 함께 들어가 content leakage와 style drift를 유발하는 문제가 있었다. 또한 고품질 paired 비디오 데이터가 부족해 합성 데이터로 학습하면 깜빡임(flickering)과 불안정성이 커졌고, 긴 영상(수 초 이상)으로 확장하기도 어렵다는 한계가 있었다. 학습 없이 keyframe 스타일 편집 후 feature를 주입해 시간 일관성을 맞추는 방법들도 있었지만, 텍스트 기반으로 안정적인 open-source 패러다임을 만들기는 여전히 난제였다.

- **Core Contribution**: EchoStyle은 텍스트 기반 비디오 스타일화를 video-to-video 생성 문제로 정식화하고, Wan2.2-I2V 기반 구조에서 비디오 내용과 텍스트 스타일을 재결합(refuse/fuse)하는 프레임워크를 제안한다. 참조 기반 누출 우려를 줄이기 위해 입력 조건을 통합된 잠재(latent) 정렬로 구성하고, 긴 길이에서도 동작하도록 init-follow-mode와 sliding-window 추론을 설계했다. 더불어 대규모 학습을 가능케 하는 자동 역합성(reverse-synthesis) 파이프라인으로 V-Style20k(20k 페어) 데이터셋을 구축했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 영상의 의미/내용 정합성을 유지하면서 스타일만 정교하게 주입하는 것과 (2) 장시간 생성에서 메모리 부담 및 누적 오류를 억제하는 것이다. EchoStyle은 VAE로 참조/타깃 비디오를 잠재로 인코딩한 뒤, mask 조건과 텍스트·타임스텝을 DiT 백본에 adaLN과 cross-attention으로 주입해 정렬성을 확보한다. 데이터 희소성은 합성 아티팩트가 섞인 역분포 오염을 피하기 위해, 실제 스타일 비디오를 수집해 현실적인 counterpart를 reverse-synthesis로 생성하는 방식으로 해결했으며, init-follow-mode 학습과 sliding-window 추론으로 세그먼트 경계의 시간 불연속을 완화했다.

- **Empirical Impact**: 실험에서 EchoStyle은 다양한 예술 스타일에서 스타일 유사도와 인간 선호 평가를 기준으로 기존 오픈/상용 대안 대비 우수한 결과를 보였고, 특히 동적 품질(dynamic quality)에서 상위권 성과를 냈다. 정량적으로는 style similarity, style consistency, content preservation 전반에서 강점을 보이며, Kling-O1·Seedance 2.0 같은 closed-source 대비로도 전반 성능이 경쟁력 있음을 보였다. 긴 쇼트(복잡한 공간 역학·조명 변화) 확장 테스트에서도 EchoStyle이 세맨틱 드리프트와 단순화가 덜해 전문용 예술 비디오 제작 대안으로 의미가 있다는 점이 확인됐다.



### C3-Bench: A Context-Aware Change Captioning Benchmark (https://arxiv.org/abs/2606.25445)
Comments:
          ECCV 2026 Camera-ready version

- **Prior Approaches**: 기존 change captioning 연구는 모델 구조 개선이나 학습 전략 고도화에 집중했지만, 정작 “변화”가 무엇으로 정의되는지 맥락과 기준을 명확히 다루지 못한 경우가 많았다. 또한 많은 벤치마크가 데이터 범위가 좁거나 특정 도메인에 치우쳐 있어, 실제 환경의 다양한 change context에서 모델이 얼마나 잘 일반화하는지 평가하기 어려웠다. 마지막으로 BLEU·ROUGE 같은 기준 기반 지표는 표현이 달라져도 의미가 같은 정답을 제대로 반영하기 어렵고, LMM 출력의 유연한 패러프레이즈를 정밀 비교하기에 한계가 있었다.

- **Core Contribution**: 이 논문은 Context-aware Change Captioning을 평가하기 위한 포괄 벤치마크 C3-Bench를 제안한다. 4개 비주얼 도메인에 걸쳐 51개 현실 change context를 포함하며, 총 4,996개의 사람 라벨 이미지 페어와 맥락별 “무엇을 바꾸었고 무엇은 무시할지/문체는 어떨지” 기준(criteria)을 함께 제공한다. 더불어 change captioning 최초로 LLM-as-Judge 기반의 fine-grained 평가(정확성, 구체성, 유창성, 관련성)와 입력 순서를 바꿔도 의미가 대칭적으로 유지되는지 보는 reversibility metric까지 마련해 신뢰 가능한 비교를 가능케 했다.

- **Technical Challenges**: 핵심 난제는 (1) 다양한 현실 맥락에서 일관된 정답 기준을 정의하고 라벨링하는 비용과 품질을 확보하는 것, (2) open-ended 서술을 사람이 납득할 만한 방식으로 평가하는 준거를 설계하는 것이다. 논문은 여러 change-centric 커뮤니티의 패턴을 distillation해 51개 context를 정의하고, 이미지 페어 구성 시 대응/검증 파이프라인과 품질 통제(중복 제거, temporal verification, 캡션-이미지 정합성 점검 등)를 통해 데이터 오염 위험을 줄였다. 평가 측면에서는 GPT-5.2를 judge로 사용해 fine-grained 차원을 점수화하고, 순서 반전 후 서술이 대칭적으로 일치하는지 reversibility까지 0/1로 측정하도록 프롬프트를 체계화했다.

- **Empirical Impact**: C3-Bench로 32개 모델(전통적 change captioning, 다수 proprietary LMM, 2B~90B급 open-source LMM)을 오프더셋 평가한 결과, 기존 관행의 근본적 blind spot이 드러났다. 기존 conventional 모델은 학습 스타일 규제를 벗어나면 붕괴하며, 심지어 SOTA급 LMM인 GPT-5.2도 도메인·위치(position) 편향에 따른 체계적 오류가 나타나 “신뢰 가능한 change understanding”을 보장하기 어렵다는 점을 보여준다. 또한 reversibility와 fine-grained 지표로 숨겨진 실패 모드를 가시화해, 다음 단계로 일반화 가능하고 신뢰할 수 있는 change captioning을 만들기 위한 구체적 연구 전선을 제시하며 관련 코드·데이터를 공개했다.



### LinStereo: Linear-Complexity Global Attention for Multi-Scale Iterative Stereo Matching (https://arxiv.org/abs/2606.25437)
- **Prior Approaches**: 기존 VFM 기반 반복(stereo) 파이프라인은 로컬 이웃 중심 업데이트에 의존해 전역 정보 전파가 늦고, 멀티스케일 특징도 단일 레벨 상관으로 축약되는 경우가 많다. 또한 초기화 단계에서 기하학적 priors를 충분히 활용하지 못해 열화된(예: 수중) 광도 단서에 취약해진다. 비용볼륨/반복 계열은 정확도를 끌어올렸지만, ‘백본-업데이트 인터페이스’의 정보 활용 폭이 병목이 됐다는 점이 한계로 지적된다.

- **Core Contribution**: LinStereo는 Position-Aware Linear Attention(PALA)을 반복 스테레오 디코더의 핵심으로 두고, 로컬 recurrence 대신 전역 집계를 선형 비용으로 수행해 정보 전파의 갭을 줄인다. Hierarchical Semantic Cost Volumes(HSCV)로 멀티스케일 VFM 특징을 레벨별 상관 신호로 공급하고, Depth Prior Initialization(DPI)로 단안 depth를 메트릭에 가깝게 보정한 warm start에서부터 정제를 시작한다. 결과적으로 업데이트 루프가 백본 표현과 기하학 priors를 더 직접적으로 활용하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 선형(associativity 기반) attention을 쓰면 공간 관계가 position-agnostic으로 무너질 수 있다는 점이며, 이는 disparity 구조(경계/매끈함) 학습에 불리하다. LinStereo는 RoPE(2D rotary position embedding)를 attention의 numerator에 비대칭적으로 적용해 상대 위치 민감성을 복원하면서도 안정적인 정규화를 유지해 학습 붕괴를 피한다. 또한 부정확한 상관 증거가 많은 수중 환경을 고려해 gated update로 신뢰도에 따라 증거를 강하게 반영하거나 보존하며, 반복 스텝별 supervision(지수 가중)과 edge-aware/gradient 손실로 depth discontinuity를 복구하도록 학습한다.

- **Empirical Impact**: LinStereo는 SceneFlow만으로 학습해도 표준 스테레오 벤치마크에서 SOTA급 성능을 보였고, 특히 occlusion에서 이전 최고 대비 EPE를 37% 낮추며 전역 전파의 효과를 확인했다. 수중 일반화에서는 TartanAir-UW에서 AbsRel을 28%, SQUID에서 26% 개선해, 심한 photometric degradation 상황에서도 robust하게 동작함을 입증했다. 또한 SeaStereo-Dataset(현실적인 수중 광학 모델 기반, dense disparity ground truth)과 오프라인 코드 공개로 실험 재현성과 연구 확장성까지 강화했다.



### PRISM: Feed-Forward Single-Image 3D Reconstruction via Geometric Warp-Residual Modeling (https://arxiv.org/abs/2606.25430)
- **Prior Approaches**: 단일 이미지를 이용한 3D 장면 복원은 NeRF·3DGS처럼 장면별 최적화가 필요해 확장성이 떨어지거나, feed-forward 복원은 여러 포즈 입력이 요구되는 경우가 많다. 최근에는 camera-controlled video diffusion을 활용한 방법들이 성능을 끌어올렸지만, 추론 시 hundreds frame에 걸친 iterative diffusion sampling이 필요해 실사용 배포가 어렵다. 또한 픽셀 공간 warping 기반 기법은 깊이 오차로 인한 아티팩트가 생기고 disoccluded 영역을 별도 inpainting으로 메워야 하는 한계가 있었다.

- **Core Contribution**: PRISM은 단일 이미지 3D 복원을 diffusion 없이 처리하기 위해, 다중 뷰 latent 예측을 “파라미터 없는 기하 prior”와 “학습된 residual 보정”으로 분해한다. 입력에서 target view 대부분을 geometric forward warping으로 채우고(약 90%), 남는 disoccluded/오차 영역만 잔차 인코더가 compact하게 수정하도록 설계했다. 이때 warping과 보정은 픽셀 공간이 아닌 compressed video latent 공간에서 수행해 residual 부담을 줄인다.

- **Technical Challenges**: 핵심 난제는 합성 데이터로 학습할 때도 실제 장면에서 기하·외관 품질을 동시에 일반화하는 것이다. PRISM은 두 단계 학습을 도입해(1) frozen 3DGS 디코더의 cross-view attention feature를 기준으로 latents supervised distillation을 수행해 scene-specific memorization을 완화하고, (2) 고정된 디코더로 렌더링한 결과에 대해 perceptual fine-tuning(LPIPS·MSE)을 적용해 외관(질감·선명도) 품질을 정교화했다.

- **Empirical Impact**: 세 가지 벤치마크(RealEstate10K, DL3DV, Tanks-and-Temples)에서 PRISM은 합성만으로 학습했음에도 diffusion 기반 방법들과 경쟁하거나 더 나은 정량 성능을 보였다. 특히 LYRA 대비 추론 시간은 장면당 36초로 줄며 277×277× 수준의 속도 향상을 보고했으며, SSIM/LPIPS/PSNR 전반에서도 강세를 보였다. 또한 ablative 결과와 실제 장면 평가를 통해 warp-residual 분해와 두 단계 학습이 각각 지오메트리 일반화와 지각적 품질 개선에 기여함을 확인했다.



### Gastroendoscopy View Synthesis: A New Real Dataset and Evaluation (https://arxiv.org/abs/2606.25427)
Comments:
          Accepted for EMBC 2026. Project page: this http URL

- **Prior Approaches**: 기존 NVS 연구는 NeRF 또는 3DGS 같은 생성 기반 방법을 중심으로 발전했지만, 의료 내시경용 평가지표가 될 만한 NVS 전용 데이터가 부족했습니다. 대장내시경이나 일부 수술 장면 데이터는 존재하나, 고정 관점 위주이거나(관점 변화가 작음) 가림 문제를 목표로 해 임의 관점 생성 평가에는 한계가 컸습니다.

- **Core Contribution**: 이 논문은 실제 위내시경으로 촬영한 NVS 전용 최초 데이터셋 GastroNVS를 공개하며, 위 병변 포함 영상 시퀀스와 카메라 포즈, SfM 기반 point cloud를 함께 제공합니다. 또한 관점이 균일하게 분포하는 Split-Reg와 멀리 떨어진 novel view를 보도록 설계한 Split-Con 두 가지 분할로 “진짜 novel” 관점 평가까지 가능하게 했습니다.

- **Technical Challenges**: 위내시경은 곡면과 미세 변형이 크고, 카메라가 빠르게 움직이면 특징 매칭이 깨져 포즈가 일부 누락될 수 있습니다. 논문은 fisheye 카메라를 SfM로 추정한 내재 파라미터 기반 pinhole로 보정해 왜곡 없는 입력을 만들고, 3DGS 계열 중에서도 depth(3DGS+depth)나 SDF(GSDF)로 기하를 명시적으로 학습하는 방식이 떠다니는 아티팩트·블러를 억제하는지 비교로 검증했습니다.

- **Empirical Impact**: 실험 결과 Split-Reg에서는 기하를 명시적으로 최적화하는 3DGS+depth와 GSDF가 일관되게 우수했고, Gaussians를 평탄화하는 2DGS와 PGSR은 불일치한 위치에서의 평탄화/노이즈로 성능이 떨어졌습니다. Split-Con처럼 훈련-테스트 관점 차가 큰 조건에서도 GSDF가 전 지표에서 가장 높은 성능을 보였으며, 위내시경 특화 이슈로 inter-view illumination inconsistency(조명 방향·세기 차)로 인한 밝기 불일치가 재구성을 악화시키는 현상도 확인했습니다. 



### Teach-to-Reason: Competition-Guided Reasoning with a Self-Improving Teacher (https://arxiv.org/abs/2606.25407)
- **Prior Approaches**: 기존 CXR VQA 후학습은 정답/검증 가능 여부 같은 answer-level 보상에 의존하는 RL/RLVR 방식이 많다. 이런 신호는 체인오브쏘트(CoT) 품질을 직접 다듬기엔 너무 거칠고, GRPO 같은 그룹 단위에서 advantage가 0으로 붕괴되면 학습 신호가 희소해진다. 또 rule-based나 LLM-as-a-Judge는 고정 기준 만족에 치우쳐, 시간이 지나면 보상이 덜 유용해질 수 있다는 한계가 있다.

- **Core Contribution**: 논문은 Teach-to-Reason(T2R)로, CoT 최적화에 비교(comparison) 기반 감독을 도입해 추론 품질을 안정적으로 개선하는 프레임워크를 제안한다. T2R은 훈련 중에만 쓰는 self-improving Teacher가 더 좋은 reference CoT를 만들고, Reasoner는 Teacher 생성 Co트를 상대로 경쟁 점수 기반으로 업데이트된다. 또한 case-wise 보상 설계로 과제 보상으로 형성된 positive/negative partition은 유지하되, 붕괴(퇴화) 시에는 competition score로 감독을 복원해 신호가 끊기지 않게 한다.

- **Technical Challenges**: 주요 기술 과제는 (1) CoT에 대한 비교 신호를 만들되 (2) 기존 과제 보상으로 생긴 선호 구조를 무너뜨리지 않으면서 (3) advantage가 0으로 퇴화하는 상황에서도 계속 유효한 학습 신호를 제공하는 것이다. T2R은 LLM-as-a-Judge로 두 CoT를 pairwise로 비교해 competition score와 Teacher self-competition score를 정의하고, GRPO 업데이트에서 non-degenerate와 degenerate 케이스를 분리해 서로 다른 방식으로 최종 advantage를 구성한다. degenerate일 때는 competition score로 threshold 기반 분할을 복원해 부호 구조를 유지하도록 재구성하고, non-degenerate일 때는 competition score를 subset 내부 상대순위 미세조정에만 사용한다.

- **Empirical Impact**: 여러 CXR open-ended VQA 벤치마크에서 T2R은 Qwen3-VL-Instruct 2B/4B 스케일 모두에서 강한 베이스라인 대비 일관된 성능 향상을 보였다. 특히 Frozen Teacher나 Judge Reward보다 비교 기반 감독이 더 조밀하고(비영-zero advantage 그룹 비율이 높음) 더 오래 지속되는 학습 동역학을 보였다는 점이 강조된다. 절제 실험에서는 degenerate-case handling이 성능 이득의 핵심 축으로 나타나, 단순히 competition score를 보상에 직접 주입하는 방식보다 T2R의 원칙적 결합이 효과적임을 시사한다.



### Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis (https://arxiv.org/abs/2606.25390)
Comments:
          Published in Canadian AI 2026

- **Prior Approaches**: 기존 연구는 사이트 간 도메인 차이와 적은 라벨 문제를 GAN, conditional Latent Diffusion(LDM), 혹은 VAE-GAN 같은 생성 모델로 일부 완화해왔습니다. 다만 많은 방법이 2D 중심이거나, 데이터가 매우 적은 극단적 few-shot + 교차 도메인 상황에서 구조적 일관성과 진단에 유효한 병변 특징을 충분히 보장하기 어렵다는 한계가 남아 있습니다. 또한 MRI 합성은 스캐너/프로토콜 차이로 인한 분포 불일치를 근본적으로 줄이지 못해 downstream 성능 하락으로 이어질 수 있습니다.

- **Core Contribution**: ALDM(Anatomically-conditioned Latent Diffusion Model)은 데이터가 풍부한 소스 도메인에서 해부학적 priors를 3D VAE로 학습한 뒤, 타깃 도메인에서는 종양 마스크를 ControlNet으로 조건화해 소량 데이터로도 3D 볼륨을 합성하는 프레임워크입니다. 종양 경계와 위치/형상에 직접 결정을 주는 구조 조건을 latent diffusion에 결합해, 도메인 전이 시 병변의 공간적 일관성을 유지하도록 설계했습니다. 결과적으로 T1, T2, FLAIR 3개 모달리티의 3D 종양 MRI를 데이터 효율적으로 생성하고, 분류에 필요한 진단 특성을 보존하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 3D에서 비용이 큰 diffusion을 어떻게 효율화하느냐, (2) few-shot 타깃 도메인에서도 종양 위치·경계는 지키면서 모달리티별 질감은 현실적으로 바꾸느냐입니다. 논문은 VAE latent 공간에서 diffusion을 수행해 계산 부담을 줄이고, FiLM 스타일 마스크 조절과 ControlNet의 다중 스케일 edge/soft distance conditioning으로 종양 구조를 강하게 가이드합니다. 또한 tumor 영역에 가중치를 둔 diffusion loss와 classifier-free guidance로 해부학적 순응도와 샘플 다양성의 균형을 맞춥니다.

- **Empirical Impact**: ALDM은 타깃 16장(극단적 few-shot) 환경에서 FID 85.40으로 GAN 및 하이브리드 베이스라인을 능가했으며, downstream 분류에서도 AUC 0.987±0.001로 가장 높은 성능을 보였습니다. 정성 결과는 학습이 진행될수록 종양 경계가 선명해지고 모달리티 간(특히 T2/FLAIR) cross-modal 일관성이 좋아짐을 보여줍니다. 이는 저자원이 부족한 임상 데이터에서 생성 기반 data augmentation이 단순 시각 품질을 넘어 진단 성능까지 견인할 수 있음을 실증한 사례로 의미가 큽니다.



### Transferable Attack against Face Swapping in an Extended Spac (https://arxiv.org/abs/2606.25376)
- **Prior Approaches**: 기존 연구는 딥페이크 탐지처럼 ‘생성 후’ 대응하거나, white-box 가정 하의 adversarial examples으로 모델을 혼란시키는 방식에 머물렀습니다. black-box에서는 LaS-GSA처럼 다량 쿼리 기반 방법도 있었지만, 높은 비용과 낮은 전이성 근거가 한계로 지적됩니다. transfer 학습·생성 기반 접근도 있었으나 subject-agnostic face swapping에 직접 겨냥하지 않아 모델 간 아키텍처 차이로 성능이 흔들렸습니다.

- **Core Contribution**: 이 논문은 subject-agnostic FS 모델을 공략하는 transferable 공격인 Additive Identity attack based on a Relighting function(AIR)을 제안합니다. AIR은 identity extraction 모듈을 목표로 하되, face swapping이 아닌 face recognition 모델 앙상블을 surrogates로 써서 surrogate FS 모델 없이도 높은 전이성을 확보합니다. 또한 Additive Identity Attack(AIA)과 Relighting Functional Attack(RFA)을 결합해 공격 공간을 확장함으로써 더 강하지만 시각적으로 자연스러운 예시를 노립니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 타깃 FS 모델을 구하기 어려운 환경에서의 전이성, (2) perturbation budget(시각적 티를 좌우) 안에서 attack success를 동시에 끌어올리는 균형입니다. AIR은 AIA에서 Adaptive Translation-Invariant(ATI) 연산으로 transferability는 높이되 표면 잡음 텍스처를 줄이고, RFA에서는 relighting(조명 재설정) 기반 섬세한 변화로 자연스러움을 보존합니다. 더불어 조명 파라미터에 대한 제약(구면조화 기반 계수 범위 패널티)으로 과도한 조명 왜곡을 억제하며, 두 유형 perturbation을 함께 최적화해 공격 공간 확장에 대한 수학적 정당화도 제시합니다.

- **Empirical Impact**: 1000개 이미지 페어를 대상으로 GAN 기반(FaceShifter, SimSwap, MegaGAN)과 diffusion 기반(DiffFace, DiffSwap) FS 모델들에 대해 실험한 결과, AIR은 공격 성공률과 이미지 품질에서 기존 공격 전반을 앞섰습니다. black-box 설정에서 baselines의 ASR이 대체로 0.2 이하로 머문 반면, AIR은 ASR 0.699~0.970, AWSS 7.080~14.877로 일관된 효과를 보였습니다. 사용자 연구와 CLIP-IQA, TV 같은 지표에서도 AIR이 상대적으로 덜 눈에 띄는 perturbation을 생성해 실무적 프라이버시·보안 논의에 의미 있는 신호를 제공합니다.



### Beyond Visual Forensics: Auditing Multimodal Robustness for Synthetic Medical Image Detection (https://arxiv.org/abs/2606.25375)
Comments:
          Accepted at MICCAI 2026

- **Prior Approaches**: 기존 합성 의료 이미지 탐지는 주로 이미지 단독 입력만으로 평가해 왔습니다. 그 결과 임상처럼 영상과 임상기록(메타데이터)이 함께 들어가고 VLM이 멀티모달로 판단하는 실제 배치 환경의 취약점을 충분히 설명하지 못했습니다. 또한 텍스트를 약간 교란했을 때의 민감도는 다뤄졌더라도, ‘같은 이미지를 고정한 채’ 동반 기록만 바꿔 생기는 의사결정 편향을 체계적으로 정량화한 벤치마크는 부족했습니다.

- **Core Contribution**: 이 논문은 합성 의료 이미지 탐지를 ‘이미지-기록 인터페이스에서의 멀티모달 강건성 감사(audit)’로 재정의합니다. 특히 같은 이미지를 고정하고 메타데이터의 단일 provenance 필드만 교체하는 paired 벤치마크를 제안해, 기록 문맥이 진위 판단을 어떻게 뒤집는지(Text-Induced Decision Shift)를 측정합니다. 이를 통해 모델이 시각 근거보다 기록 컨텍스트에 과도하게 의존하는 취약 패턴을 드러냅니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 현실적인 합성 쌍을 만들면서도 (2) 이미지 자체는 고정하고 (3) 기록 컨텍스트만 통제해 공정한 비교를 가능하게 하는 설계입니다. 저자들은 LLM-guided edit–verify–refine 루프로 이미지 편집을 생성·검증하고, Base 메타데이터에 Source 필드만 Hospital vs AI-edited로 강하게 주입해 상한 스트레스 테스트를 구성했습니다. 이후 I-Only, I+Base, I+Source-H, I+Source-AI의 네 조건에서 FINAL ANSWER와 VISUAL VERDICT를 함께 산출해, 최종 판단 변화가 시각 추론 변형인지(Reasoning Rewrite) 또는 통합 판단 편향인지(Verict Split)까지 분해해 분석합니다.

- **Empirical Impact**: 여러 데이터셋(NIH-CXR14, ISIC2019, PediCXR)과 다양한 VLM 계열(오픈 가중치, general VLM, frontier API, DetectFake VLM)을 대상으로 실험한 결과, provenance 메타데이터만 바꿔도 인증/합성 판정이 크게 흔들렸습니다. 예를 들어 NIH-CXR14의 authentic에서 Source-AI를 넣으면 평균 정확도가 61.1% 하락했으며, 일부 모델은 MedGemma처럼 거의 0%대까지 붕괴했습니다. Fake 이미지에서는 시각 단독 탐지가 이미 어렵지만, Source-AI 메타데이터가 진짜 같은 판단을 ‘시각 개선 없이’ 부풀리거나(허위 TPR 상승) Source-H가 불확실할 때 false negative를 늘리는 등 기록 단서의 지배가 관찰됐습니다. 저자들은 이 효과가 상한 스트레스 테스트임을 강조하며, 이미지 단독 평가가 실제 멀티모달 배치 리스크를 과소평가할 수 있음을 시사하고 향후 멀티모달 강건성 개선의 표준 도구로 활용될 수 있다고 제안합니다.



### Hypergraph Normal World Models for Logical Visual Anomaly Detection (https://arxiv.org/abs/2606.25368)
Comments:
          20 pages, 10 figures

- **Prior Approaches**: 기존 비정상 탐지는 보통 정상만으로 학습한 뒤, 테스트 패치·특징을 정상 기준 분포(메모리뱅크, Gaussian 통계, 재구성/예측 오차, flow likelihood, teacher-student 불일치 등)에 매핑해 이상을 판단한다. 이 방식은 스크래치·오염·결손처럼 국소 구조 결함이 명확할 때 강하지만, 논리적 이상(logical anomaly)은 각 부분이 정상처럼 보여도 개수·공동출현·공간 관계 같은 ‘전체 규칙’이 깨지는 문제라서 매핑만으로는 한계가 크다. MVTec LOCO는 이러한 논리적 이상과 구조적 이상을 분리해 평가하도록 설계됐다.

- **Core Contribution**: 이 논문은 DINOv2의 frozen patch token을 입력으로, 정상 이미지로부터 ‘카테고리별 정상 세계(normal world)’를 학습해 논리적 이상을 탐지하는 Hypergraph Normal World Model을 제안한다. 패치를 그대로 비교하는 수준을 넘어, 고정된 공간 hyperedge로 토큰을 그룹화하고 패치-관계-하이퍼엣지-하이퍼엣지 관계의 통계를 distill해 전체 구성이 정상 규칙을 따르는지 점수화한다. 점수는 정보 몫(information quotient) 형태로, 국소 유사성만으로는 설명이 비효율적일 때(관계 위반) 크게 반응하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 (1) 정상 샘플이 적으면 기준선이 불완전해지고, (2) 논리적 이상이 ‘관계 기반’이라 nearest local patch 거리만으로는 설명 효율을 측정하기 어렵다는 점이다. 저자들은 학습 시각은 정상-only로 유지하면서, DINOv2 인코더는 고정하고 정상 데이터로 특징 공간에서 필요한 통계만 캘리브레이션해 few-shot에서도 규칙을 더 잘 일반화하도록 했다. 또한 단순 패치 메모리 대신 hypergraph를 통해 여러 영역 단위의 state와 영역 간 관계를 함께 에너지로 계산하고, 이를 표준화 손실의 합으로 정보 몫 Q를 구성해 관계 위반을 정량화한다.

- **Empirical Impact**: MVTec LOCO breakfast-box에서 논리적 이상 AUROC는 DINOv2 patch-kNN 0.8434에서 하이퍼그래프 전체 모델 0.9279로 크게 향상됐다. 하이퍼그래프가 없는 변형(0.9013→0.9279)보다도 개선돼, 단순 국소-관계 통계만으로는 부족하고 hypergraph 멀티레벨 구조가 추가 신호를 제공함을 보여준다. 더불어 정상 이미지 수가 1장인 극소 상황에서도 logical AUROC 0.8597로 유효성을 유지하며, relation counterfactual에서 정보 몫이 평균 83.13만큼 증가하고 t-SNE 시각화도 논리적 이상이 학습된 energy 공간에서 분리되는 등 ‘관계 기반 정상 세계’를 반영한다는 해석 가능성도 제시했다.



### Follow Your Track: Precise Skeleton Animation Controlled by 3D Trajectories (https://arxiv.org/abs/2606.25344)
- **Prior Approaches**: 기존 4D 생성은 텍스트/비디오 제어로 3D 자산을 만들고, 메쉬나 Gaussian 같은 조밀 표현에 모션을 분리·주입하는 방식이 주류였다. 이런 표현은 장시간 애니메이션에서 계산 비용이 커지고 시간 축 아티팩트가 늘어 품질·지속 길이가 짧아지기 쉽다. 또한 텍스트는 timing·coordination 같은 미세 실행이 부족하고, 비디오는 모션과 외형·배경이 얽혀 정확한 전이가 어려웠다.

- **Core Contribution**: ACT는 topology-general skeletal animation을 위해 trajectory-conditioned 프레임워크를 제안한다. 핵심은 스켈레톤을 컴팩트한 구조 표현으로 쓰고, 단안(monocular) 비디오에서 얻은 3D point trajectories를 모션 가이드로 삼아 appearance와의 얽힘을 줄이는 것이다. 특히 Routed Trajectory Injector로 관절-궤적 주입을 공간(스켈레톤-메쉬 대응)과 시간(마이크 타이밍 정렬) 양쪽에서 정교화했다.

- **Technical Challenges**: 가장 큰 기술 과제는 ‘임의 topology 스켈레톤’에 대해 noisy·불균일 궤적을 안정적으로 관절 시퀀스로 변환하는 것이다. ACT는 prior-guided hard routing과 global routing을 결합해 골격-궤적 매핑을 정밀화하고, local windowed cross-attention으로 각 관절이 가까운 시간대 모션 패턴을 보도록 하여 속도 변화에도 미정렬을 줄였다. 더불어 topology-agnostic normalization으로 루트와 비루트의 스케일/오프셋 편향을 트랙 통계와 intrinsic offset 기반으로 정리해, 데이터셋 통계에 의존하지 않는 일반화를 노렸다.

- **Empirical Impact**: Truebones(다양한 동물)과 AMASS(인체)에서 폭넓게 실험한 결과, ACT는 기존 방법 대비 fidelity와 temporal consistency에서 유의미하게 우수했다. 평가지표는 subject consistency, motion smoothness, dynamic degree 외에도 LPIPS·FVD 및 적용 가능한 MPJPE를 사용해 렌더링 품질과 시간적 거동을 함께 확인했다. 특히 메쉬·Gaussian 중심 접근이 제공하기 어려운 ‘명시적 articulated skeleton 모션’을 궤적 기반으로 안정적으로 생성한다는 점에서, 장시간·제어 가능한 4D 애니메이션 방향에 실질적 의미가 크다.



### Invoice Haystack: Benchmarking Document Retrieval and Visual Question Answering Under Strong Visual Homogeneity (https://arxiv.org/abs/2606.25343)
Comments:
          Accepted to presentation at ECCV 2026

- **Prior Approaches**: 기존 Vision Language Model은 단일 문서 VQA에서 준인간급 성과를 보이지만, 대규모 다문서로 확장하면 검색·증거추적 성능이 크게 흔들린다. 또한 DocHaystack/InfoHaystack 같은 다문서 벤치마크는 규모와 고유성은 개선했지만 문서 간 시각적 동질성(템플릿 반복)을 충분히 반영하지 못해 임베딩 공간에서 인위적인 분리가 생긴다는 한계가 있다. 그 결과 비슷한 템플릿이 대량 존재하는 기업 문서 저장소 환경을 제대로 스트레스 테스트하지 못한다.

- **Core Contribution**: 이 논문은 기업 인보이스처럼 시각적으로 매우 균질한 문서에서의 검색 실패 원인을 “embedding collapse”로 규정하고, 이를 겨냥한 벤치마크 Invoice Haystack을 제안한다. Invoice Haystack은 1,500개의 익명화 인보이스와 200개의 판별적 Q&A로 구성되며, 평균 pairwise cosine similarity가 0.73으로 기존(0.31~0.38)보다 훨씬 높아 근본적으로 더 어려운 검색 문제를 만든다. 또한 VL-RAG(Vision-Language RAG)로 텍스트·비전 임베딩을 이중 스트림으로 결합하고, VLM 기반 검증 필터로 정확한 문서 식별을 달성한다.

- **Technical Challenges**: 동질 템플릿이 대량일 때 비전 인코더만으로는 인스턴스 수준의 차이를 구분하지 못해 임베딩이 무너지는 문제가 핵심 기술 난제다. 반대로 텍스트 단독은 레이아웃·구조 신호를 버려 오탐 가능성이 커서, 두 양상을 동시에 활용하는 점진적 검색 위계(고리콜→고정밀)가 필요하다. 논문은 OCR 기반 dense text embedding과 SigLIP/OpenCLIP 기반 비전 임베딩을 평균 융합으로 랭킹하고, Qwen3-VL의 yes/no 이진 검증으로 top 후보를 재필터링하되 통과 후보가 없으면 fallback을 제공해 실용성을 유지한다.

- **Empirical Impact**: 실험 결과 VL-RAG는 Invoice Haystack-500에서 Recall@1 60.0%를 달성하며, 기존 SOTA 대비 최대 13.5%p 절대 개선을 보였다. 또한 DocHaystack-1000(77.1% vs 75.2%), InfoHaystack-1000(84.5% vs 80.0%)에서도 우수한 성능을 유지해 동질/이질 환경 모두에서 dual-stream fusion이 일관되게 강하다는 점을 확인했다. VQA 정확도 평가에서도 검색 지원만으로 큰 하락을 막지 못하던 인보이스 과제가 VL-RAG로 개선되며, 특히 Invoice Haystack에서 성능 이득이 가장 크게 나타나 실효성이 입증됐다.



### State Space Models Meet Remote Sensing: A Survey (https://arxiv.org/abs/2606.25329)
Comments:
          25 pages, 5 figures, has been published in SCIS SCIQ1 IF=8.1 this https URL

- **Prior Approaches**: 원격탐사 분야에서 SSM(State Space Model)은 긴 시퀀스에 유리한 선형 복잡도 덕에 주목받았지만, 기존 연구는 주로 비전 일반(예: 분류·검출 중심) 리뷰가 많아 원격탐사의 특수 요구를 충분히 정리하지 못했다. 또한 원격탐사 데이터는 밀집 예측, 다중모달(HSI/MSI/PAN), 시계열 변동, 그리고 주파수 영역 잡음 같은 이슈가 동시에 존재해 스캔·융합·잡음 제거가 설계 핵심이 된다. 기존 방식은 단일 방향성 스캔의 편향이나 국소 질감/경계 정보 부족, 또는 다중모달 융합 시 모달 품질 저하에 따른 흔들림 같은 한계를 반복해서 드러냈다.

- **Core Contribution**: 이 논문은 2024년 3월~2025년 12월 사이의 RS-SSM 300편 이상을 체계적으로 추적해 원격탐사 과제 전반에서의 SSM 적용 양상을 다차원으로 정리한다. 특히 task별 진화 흐름과 함께 SSM 아키텍처를 8가지 설계 관점으로 분류해, 어떤 스캔/융합 선택이 어떤 문제(전역 의존, 다중모달 상관, 시공간 변화 등)를 다루는지 한눈에 보이게 한다. 더 나아가 기존 리뷰가 상대적으로 다루지 못했던 ‘원격탐사 foundation model’ 관점의 향후 과제까지 제시한다.

- **Technical Challenges**: 기여를 현실화하는 핵심 기술 난제는 (1) 원격탐사에 내재된 방향성 부재로 인한 단일 경로 스캔의 편향, (2) 다중모달·시계열에서의 중복 억제와 정보 충돌 회피, (3) 주파수 기반 특징(푸리에/웨이브렛 등)과 공간 기반 특징의 효과적 융합이다. 저자들은 이를 위해 다중방향·적응형 scanning 전략, CNN/Transformer/GNN 등과의 하이브리드 구조, 그리고 게이팅/피처 교환 같은 융합 메커니즘을 아키텍처 축으로 묶어 분석한다. 또한 과도한 스캔 경로는 중복을 늘릴 수 있어, 과제·모달·데이터 특성에 맞는 스캔 조합 선택이 중요하다는 점을 명확히 짚는다.

- **Empirical Impact**: 경험적으론 원격탐사 분류·세그멘테이션·검출·변화탐지·복원(초해상도/디노이징/디헤이징/팬샤프닝) 등 다양한 벤치마크에서 SSM 계열이 경쟁력 있는 성능을 보였으며, 특히 글로벌 문맥을 선형 비용으로 다루는 이점이 반복 확인됐다. 예를 들어 분류에서는 다중방향 스캔이 클래스 간 혼동과 표본 부족 환경에서의 안정성에 기여하고, 복원에서는 주파수-공간 특징 융합이 성능을 좌우함이 드러난다. 저자들은 이런 결과를 바탕으로 원격탐사 특화 아키텍처 설계와 SSM 기반 foundation model로의 확장을 위한 구체적 연구 방향을 제공한다.



### Efficient Remote Sensing Instance Segmentation with Linear-Time State Space Distilled Visual Foundation Models (https://arxiv.org/abs/2606.25324)
Comments:
          17 pages, 11 figures, has been published in IEEE TGRS vol. 64, pp. 5625417-5625417, 2026, Art no. 5625417, doi: https://doi.org/10.1109/TGRS.2026.3696104

- **Prior Approaches**: 기존 원격탐사 인스턴스 세그멘테이션은 CNN 기반이 중심이었고, 최근에는 ViT 같은 비전 파운데이션 모델을 백본으로 쓰는 Transformer 계열이 정확도를 끌어올렸습니다. 다만 고해상도에서 토큰 수가 급증하면 ViT의 self-attention이 토큰 길이에 대해 제곱으로 비용이 커져 파라미터·메모리·추론 속도 제약이 심해집니다. 이에 따라 경량화는 distillation, pruning, SAM 계열 압축 등으로 시도됐지만, 여전히 Transformer 의존성이 효율 한계를 일부 가져갔습니다.

- **Core Contribution**: 이 논문은 RS4D(Remote Sensing instance Segmentation with linear-time State Space Distilled visual foundation models)로, ViT 기반 파운데이션 모델의 세그멘테이션 지식(teacher)을 SSM 기반 student로 선형 시간 복잡도로 압축합니다. 핵심 아이디어는 distillation을 통해 self-attention의 방대한 표현 공간을 compact한 dense linear state space로 옮기는 것입니다. 이를 위해 경량 SSM 백본(세 가지 변형)과 원격탐사 인스턴스 세그멘테이션용 아키텍처(세그멘테이션 헤드 변형)를 함께 설계합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 고해상도 long sequence에서 SSM이 전역 문맥과 안정적인 학습을 유지하면서도 선형 복잡도를 지키는 것입니다. 저자들은 (1) adaptive noise and masking 지식 증류로 마스킹/잡음 입력에서도 teacher-student 출력 일치 학습이 흔들리지 않게 하고, (2) 단방향 편향을 줄이기 위해 forward/backward 2D scanning을 적용한 ScanningMamba로 마스크 품질과 최적화를 개선합니다. 또한 VanillaMamba, TransMamba 등 세 변형을 비교해 해상도·학습 안정성·연산량 사이의 설계 선택을 실험적으로 정리합니다.

- **Empirical Impact**: SSDD, WHU, NWPU 등 다수 벤치마크에서 RS4D는 ViT 대비 파라미터를 약 1/8, FLOPs를 약 1/9 수준으로 줄이면서 정확도는 동등하거나 더 좋게 유지하는 결과를 보였습니다. 특히 RSPrompter 수준의 성능을 맞추면서도 학습 시 모델 파라미터와 GPU 메모리를 크게 절감해, 고해상도 원격탐사에서의 실용성을 강화합니다. 이는 Transformer의 제곱 비용 병목을 SSM 기반 linear-time distillation으로 대체할 수 있음을 경험적으로 뒷받침한다는 점에서 의미가 큽니다.



### V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning (https://arxiv.org/abs/2606.25319)
- **Prior Approaches**: 미세 시각 추론(fine-grained visual reasoning)은 로컬 이미지 영역의 근거를 찾아야 하지만, 기존 agentic 접근은 강화학습(RL)에 의존해 탐색 비용이 크거나, 사전 정의된 검증 규칙이 필요했다. 반면 supervised fine-tuning(SFT)은 성능이 좋더라도 대규모 텍스트 정답/추론 라벨에 의존하고, 모델 능력을 깎을 수 있는 catastrophic forgetting 위험이 따른다. On-Policy Distillation(OPD)은 학생이 샘플링한 궤적에 대해 토큰 단위 교정을 제공하지만, 궤적 전체가 맞는지 틀어졌는지에 대한 궤적 레벨 판별이 약하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 OPD를 negative-free stop-gradient alignment로 재해석하며, OPD의 잠재력이 trajectory-level discrimination 부재로 제한된다는 점을 이론적으로 짚었다. 이를 바탕으로 V-Zero는 정답 라벨(annotated textual answer labels) 없이도 학습 중에 질문-관련 양성(positive)·음성(negative) 시각 근거를 대조해 궤적 신뢰도를 추정하고, 그 결과로 token-level distillation을 게이팅한다. 핵심은 “교정 신호는 양성 뷰에서만 받되, 어느 궤적에서 더 많이/적게 배울지”를 교사 측 증거 비교로 결정하는 구조다.

- **Technical Challenges**: 정답 라벨 없이 미세 근거에 의존하는 추론을 학습시키려면, 교사 신호가 토큰 단위 정답을 넘어서 “이 궤적이 정답 근거로 향하고 있는지”를 평가해야 한다. V-Zero는 동일 프롬프트에서 학생이 생성한 sibling rollouts을 묶어 교사가 같은 토큰 시퀀스를 질문-관련 영역 크롭(positive)과 무관 영역 다운샘플 크롭(negative)으로 재평가한 뒤, 토큰별 teacher support 격차를 합쳐 trajectory-level evidence advantage를 만든다. 그런 다음 non-negative stop-gradient contrastive evidence gate로 게이팅 가중치를 클리핑하며, 음성 뷰는 학습 목표에서 제거하고 양성 뷰 기반 OPD만 안정적으로 distill한다.

- **Empirical Impact**: 실험에서 V-Zero는 Qwen3.5-4B 백본 대비 미세 시각 추론 벤치마크 전반에서 평균 3.1점 향상(예: VStar +4.7, HR-4K +3.4, HR-8K +2.0, ZoomBench +5.5)을 보이면서도 테스트 시 일반 full-image 추론 설정을 그대로 유지했다. 또한 정답 텍스트/추론 라벨 없이도 학습 신호는 paired visual evidence views(양성 크롭+음성 크롭)만으로 제공되며, 학습 비용은 기존 SFT 대비 5× 이상, RL 대비 10× 이상 절감되는 것으로 보고됐다. 추가로 contrastive evidence gating, rollout group size(G), 학습 스텝(주요 체크포인트 60) 선택이 성능에 유의미하게 기여함을 보였고, 시각적 grounding 기반 방법들과 비교해 경쟁력 있는 결과를 확인했다.



### REViT: Roto-reflection Equivariant Convolutional Vision Transformer (https://arxiv.org/abs/2606.25318)
Comments:
          Accepted for publication at ICML 2026

- **Prior Approaches**: 기존의 roto-reflection(회전·반사) 등 갈무리 대칭(equivariance) 모델은 주로 CNN 기반 group convolution과 equivariant pooling을 사용해 왔다. 반면 ViT에서는 position encoding(위치 인코딩)과의 상호작용 때문에 같은 수준의 대칭 보장이 어렵고, 이를 해결하려고 relative position encoding을 attention에 매번 반영하는 연구가 있었다.
또한 transformers의 self-attention은 순열(permutation)에는 민감하지 않지만, 영상의 공간 구조(좌표 관계)를 충분히 보존하려면 추가 설계가 필요하다는 점이 병목으로 지적된다.

- **Core Contribution**: 이 논문은 discretized roto-reflection group equivariant vision transformer를 더 단순하게 구현하는 방식을 제안한다. 핵심은 vision transformer에서 position encoding을 제거하고, convolutional patch embedding과 convolutional self-attention을 통해 공간 정보를 자연스럽게 유지하는 것이다.
또한 transformer 블록 내부에 group convolutional self-attention(G-CSA)를 도입해 E(2,N) 대칭을 갖는 REViT를 구성하고, 이를 분류 태스크에서 기존 접근 대비 더 효율적으로 성능을 내도록 설계한다.

- **Technical Challenges**: 대칭 보장을 ViT에 적용할 때의 가장 큰 기술적 난제는 attention이 참조하는 좌표/관계가 position encoding 또는 relative position encoding에 의해 결정되는데, 이것이 equivariance를 깨거나 계산 복잡도를 키울 수 있다는 점이다. 논문은 이를 해결하기 위해 (1) convolutional 기반 패치 임베딩과 (2) RPE에 의존하지 않는 group convolutional self-attention을 결합한다.
구체적으로 입력을 lifting layer로 discretized roto-translation/roto-reflection group의 변환 차원까지 확장한 뒤, query/key/value 투영을 3D 합성곱(공간 2D + group 차원)으로 구현하고, convolution의 이동 등가성을 이용해 attention 연산이 group equivariant하게 동작하도록 구성한다.

- **Empirical Impact**: 실험에서는 rotated MNIST, PatchCamelyon, CIFAR-10에서 E(2,N) 및 관련 이산 그룹 설정(pN, pNm 등)으로 REViT를 평가한다. 결과적으로 논문 방법은 relative position encoding을 쓰는 group equivariant vision transformer 및 대응되는 group equivariant CNN 대비 더 적은 파라미터로 classification 성능이 더 좋다고 보고한다.
또한 더 큰 데이터셋인 ImageNet-1k로 확장 가능함을 보이며, roto-reflection equivariant backbone으로 활용할 수 있는 실용적 대안을 제시한다.



### ESTANet: Efficient Online Error Detection in Procedural Videos via Prediction Inconsistency (https://arxiv.org/abs/2606.25317)
Comments:
          18 pages, 8 figures, uses this http URL

- **Prior Approaches**: 절차형 비디오에서 오류를 찾는 기존 연구는 크게 오프라인/온라인으로 나뉘며, 온라인 접근은 과거 프레임만으로 인과적으로 추론해야 한다. 다만 현재 방법들은 영상당 첫 오류만 탐지하거나, 주로 procedural error에 치우치고 실행 오류(execution error)는 충분히 반영하지 못하는 문제가 있다. 또한 LLM 기반 선행 추정(PREGO)이나 손 포즈 같은 추가 입력(MistSense) 의존, 또는 task graph 편차 중심(DTGL)이라 실시간 제약과 일반화가 약해질 수 있다.

- **Core Contribution**: 이 논문은 action detector 자체의 예측 거동이 오류 상황에서 달라진다는 단순한 관찰을 기반으로, 실시간 online error detection을 구성한다. ESTANet(오류에 민감하고 시간적으로 변하는 네트워크)은 standard(안정형)·error-sensitive(민감형) 탐지기 4개를 두고, 이들 예측 불일치의 다수결로 오류 프레임을 플래그한다. 특히 실행 오류와 절차 오류를 모두 겨냥해, detector 간 불일치가 발생하도록 학습 설계와 추론 규칙을 함께 제안한다.

- **Technical Challenges**: 핵심은 (1) 오류 프레임에서만 예측이 자연스럽게 어긋나게 만들고, (2) 어떤 temporal context 길이가 절차 오류에서 차이를 증폭할지 자동으로 정하는 것이다. 논문은 Temporal-Aware Dynamic(TAD) 모듈로 입력 의존적 가중치/바이어스를 만들어 실행 오류에서 prediction inconsistency를 키우며, 시간 창 길이를 다르게 학습한 temporally-varying window 전략으로 절차 오류에서도 불일치를 확대한다. 또한 후보 window 조합을 전수탐색하지 않기 위해 데이터별로 ss는 평균 동작 지속이 가장 짧은 액션을 기준으로, ll은 약 θ% 지점에서 β개의 선행 액션이 포함되도록 선택하는 강건 규칙을 사용한다.

- **Empirical Impact**: EgoPER, Assembly-101-O, EPIC-Tent-O에서 ESTANet은 기존 온라인 오류탐지 최첨단 대비 일관되게 더 높은 성능을 보였다. EgoPER에서는 F1@10/25/50이 각각 47.2/37.8/21.6으로 DTGL(39.6/33.8/20.9)을 상회했고, Assembly-101-O 및 EPIC-Tent-O에서도 Avg-F1이 각각 더 큰 폭으로 개선되었다. 또한 TAD와 temporally-varying 설계가 execution과 procedural 오류에서의 균형 잡힌 탐지 성향을 만들며, 경량 아키텍처로 실시간 효율까지 유지한다는 점에서 실용적 의미가 크다.



### LEVIRDet: A Million-Scale 159-Category Dataset and Foundation Model for Universal Remote Sensing Object Detection (https://arxiv.org/abs/2606.25312)
Comments:
          18 pages, 9 figures

- **Prior Approaches**: 원격탐지 객체 탐지는 대규모 벤치마크와 최신 검출 아키텍처로 빠르게 발전했지만, 여전히 데이터와 모델이 파편화돼 범용(센서·해상도·범주 체계 통합) 검출로 이어지기 어렵다는 지적이 제기된다. 기존 데이터셋은 특정 범주/센서/고정 GSD 구간/단일 라벨 체계에 치우치거나, 박스 프로토콜과 범주 세분성이 서로 달라 학습 시 상충된 감독을 만들 수 있다. 모델 역시 scale, scene density, semantic hierarchy 같은 원격탐지 고유 요인을 명시적으로 다루지 못해 도메인 전이에 취약하다.

- **Core Contribution**: 본 논문은 LEVIRDet-159와 LEVIRDetNet을 제안하며, 범용 원격탐지 객체 검출을 위한 데이터-모델 동시 해결을 목표로 한다. LEVIRDet-159는 159개 카테고리, 256만 개 바운딩 박스, 70만 개 fine-grained 어노테이션을 multi-level taxonomy로 통합한 백만 박스급(대략 174,488장) 대규모 데이터셋이다. LEVIRDetNet은 scale-hierarchy-aware detection foundation model로서 온라인 GSD 예측, GSD-conditioned query modulation·allocation, hierarchy-aware detection head를 결합해 보편성을 높인다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 다중 센서·다중 해상도에서 동일 객체라도 관측 가능한 의미 깊이가 달라지는 문제, (2) 장면 밀도와 스케일 차이 때문에 필요한 검출 가설 수가 이미지마다 달라지는 문제, (3) fine-grained와 coarse 라벨이 섞일 때 일관된 학습 신호를 구성하는 문제다. 저자들은 데이터 엔진에서 tight-HBB로 위치 프로토콜을 먼저 표준화하고, 시각적 증거가 부족하면 조상(ancestor) 카테고리로 fallback하는 계층 라벨링을 적용해 mixed-granularity 감독을 “잡음 없이” 학습 가능하게 만든다. 모델 쪽에서는 외부 메타데이터 대신 online visual GSD cue로 쿼리 예측 예산과 임베딩을 조절하고, dynamic query allocation으로 장면 밀도에 맞춘 가설을 생성하며, 계층-aware head로 mixed-granularity 분류를 동시에 학습한다.

- **Empirical Impact**: 평가는 target-domain training·fine-tuning 없이, LEVIRDet-159로 1회 학습 후 9개 외부 벤치마크 테스트에 바로 적용하는 stringent 교차-도메인 설정에서 수행됐다. LEVIRDetNet은 9개 벤치마크의 평균 기본 지표에서 평균 5.02 mAP를 기존 최강 fully supervised 대비 향상시키며, 각 벤치마크에서 9/9 모두 1위를 기록한다. 또한 confidence threshold 하에서도 precision-recall 안정성이 더 좋게 나타나 센서·해상도·범주 체계가 다른 실제 시나리오 전반으로의 일반화 가능성을 실증한다.



### Physics Question Scene Graph: Fine-grained Evaluation of Physical Plausibility in Text-to-Video Generation (https://arxiv.org/abs/2606.25306)
Comments:
          ECCV 2026. Code and data: this https URL

- **Prior Approaches**: 기존 비디오 평가 지표는 시각 품질이나 텍스트-비디오 정합성처럼 전반적인 점수에 치우쳐, 물리 법칙 위반이 ‘어디서/무엇 때문에’ 발생했는지 세분화해 설명하기 어렵습니다. 물리 현실성을 일부 다루더라도 단일 합산 점수나 거친 카테고리 수준에 머물러, 선행 조건(오브젝트 존재, 동작 수행)이 충족되지 않을 때의 판정을 혼동하거나 환각적으로 만들 여지가 있습니다.

- **Core Contribution**: 이 논문은 Physics Question Scene Graph(PQSG)로, 생성 비디오의 물리 법칙 위반을 오브젝트-액션-물리로 분해해 계층형 질문 그래프(DAG)로 평가하는 방법을 제안합니다. VLM이 프롬프트로부터 질문 그래프를 만들고, 다른 VLM이 비디오를 보고 각 질문에 답해 카테고리별·질문별 세밀한 점수와 실패 지점을 제공합니다. 특히 질문 간 논리 의존성을 그래프로 강제해, 선행 조건이 성립하지 않으면 하위 질문을 묻지 않도록 설계한 점이 핵심입니다.

- **Technical Challenges**: 핵심 난제는 (1) 물리 평가를 위한 전제 조건을 먼저 검증하면서도 (2) 질문이 문맥적으로 항상 타당하도록 의존성을 엄격히 유지하는 동시에 (3) VLM 기반 QA의 신뢰도를 확보하는 것입니다. PQSG는 오브젝트 노드→액션 노드→물리 노드로 이어지는 DAG 의존성과 같은 카테고리 내부의 순차 의존성을 두고, 부모가 부정이면 자식 체인을 자동으로 no 처리해 질문 가능성 없는 평가를 차단합니다. 또한 QA를 개방형 답변 생성 후 yes/no 분류로 2단계화해, 단순 이진응답이 체인-오브-쏘트 활용을 저해하는 문제를 완화했습니다.

- **Empirical Impact**: 저자들은 FinePhyEval(195개 인간 주석 프롬프트-비디오 쌍, Sora 2·Veo 3·Wan 2.1 등 생성 모델 포함)을 구축해 PQSG의 세밀한 물리 점수가 인간 판단과 더 높은 상관을 보인다고 보고합니다. 자동화된 PQSG 기반 순위에서는 Sora 2와 Veo 3가 Wan 2.1보다 물리적 사실성에서 더 높게 평가되며, 전반적으로 오브젝트보다 액션과 물리에서 모델들이 더 많이 실패하는 경향이 나타났습니다. 더 나아가 FinePhyEval의 주석을 하위 작업(QG/QA) 벤치마크로도 사용했으며, VLM은 질문 생성(QG)에는 사람과 비슷한 수준에 도달하지만 질문에 답하는(QA) 능력은 아직 인간을 따라가지 못한다는 분석을 제시합니다.



### HiFiVe: High-Fidelity Vehicle Generation Leveraging Auto-Regressive 2D Generative Priors (https://arxiv.org/abs/2606.25300)
- **Prior Approaches**: 기존 3D 차량 생성은 저품질 메시에 대해 2D 생성 모델의 priors를 옮겨오는 방식이 많지만, 결과적으로 기하 디테일이 거칠거나 텍스처가 뭉개져 downstream 활용을 어렵게 했다. 멀티뷰 diffusion을 쓰는 최근 방법들은 고해상도 텍스처를 제공할 수 있으나, 고정된 viewpoint 제약과 메모리/해상도 한계, 그리고 cross-view 일관성을 위한 costly fine-tuning 의존이 문제로 남는다. 또한 긴 생성 체인에서는 view 간 정합이 흔들리며 오류 누적이 발생하기 쉽다.

- **Core Contribution**: HiFiVe는 training-free로 저품질 차량 메시에 대해 텍스처와 기하를 동시에 끌어올리는 프레임워크를 제안한다. 핵심은 3D 기하 제약을 synchronization 메커니즘으로 삼아, 사전학습된 2D generative priors를 arbitrary viewpoint에서의 고해상도 텍스처 합성에 “고정”시키는 것이다. 이후 향상된 텍스처에서 normal 정보를 추정해 메시에 고주파 표면 디테일을 복원한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 autoregressive 텍스처 합성에서 cross-view consistency를 유지하면서도 해상도와 viewpoint 커버리지를 동시에 확보하는 것이다. HiFiVe는 coarse geometry를 동기화 기준으로 두고, depth-based warping과 multi-view texture fusion(법선 정렬 기반 confidence 가중치)으로 이전 프레임의 정보를 현재 시점에 안정적으로 조건화한다. 마지막으로 차량의 대칭성을 이용해 BFS-style 생성 경로로 오차 누적을 줄이고, ISOMER 기반 normal-guided mesh optimization에 frequency-adaptive 가중치를 적용해 고주파 기하 복원을 강화한다.

- **Empirical Impact**: 실험은 synthetic( SketchFab-Cars )과 real-world( 3DRealCar ) 두 데이터에서 수행되었으며, HiFiVe는 텍스처 품질과 지각적 사실성, 기하의 입력 이미지 정합성 전반에서 SOTA 대비 유의미한 개선을 보였다. 특히 멀티뷰 diffusion 계열은 per-view 해상도 제한으로 텍스처가 over-smoothed 되는 경향이 있었고, HiFiVe는 더 선명한 고주파 디테일과 view 간 일관성을 동시에 달성했다. 기하 측면에서도 실세계 데이터에서 semantic alignment가 좋아졌으며, 휠 스포크·그릴·도어 핸들 같은 미세 구조가 더 뚜렷해지는 정성 결과가 제시된다.



### KidRisk: Benchmark Dataset for Children Dangerous Action Recognition (https://arxiv.org/abs/2606.25298)
Comments:
          SOICT 2024

- **Prior Approaches**: 기존 action recognition 연구는 주로 성인 데이터와 CNN/LSTM/Transformer 기반 영상 표현에 의존해왔고, 아동 감시로 옮기면 성능이 급격히 떨어진다는 한계가 지적돼 왔다. 또한 위험(risk) 인식은 행동 자체만 보거나, 객체 검출 등 추가 모듈을 결합해 복잡도가 높아지는 경우가 많았다. 무엇보다 아동 위험 데이터가 부족하고 라벨 불균형이 커서 학습이 어렵다는 문제가 컸다.

- **Core Contribution**: 이 논문은 아동 위험 행동을 직접 겨냥한 KidRisk 데이터셋을 구축했다. 2,500개의 짧은 아동 행동 영상과 10,000개의 위험 상황 이미지를 포함하며, 상황의 다양성과 안전/위험 라벨 불균형을 현실적으로 반영하도록 구성했다. 아울러 BLIP-2 기반의 비전-언어(vision-language) 전이학습 기준선으로 문맥 이해를 강화해, 일반 행동 분류와 위험 행동 인식을 함께 다루는 벤치마크를 제시했다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 아동 행동/위험 데이터가 작고 불균형이 심하다는 점, (2) 위험 판정에 필요한 주변 문맥을 모델이 안정적으로 활용해야 한다는 점이다. 저자들은 BLIP-2의 frozen 구성요소로 Q-Former를 통해 시각-언어 정렬을 수행하고, 행동 영상은 프레임 특징을 LSTM에 넣어 시간성을 학습한 뒤 분류기로 연결했다. 위험 인식은 프레임 단위로 BLIP-2 특징을 뽑아 sigmoid 기반 위험 확률을 예측하고, 위험 라벨 오버샘플링으로 불균형을 완화했으며, 오버피팅은 L2 regularization으로 억제했다.

- **Empirical Impact**: 실험 결과, KidRisk에서 전통적 딥러닝 대비 비전-언어 기반 접근이 유의미하게 성능을 끌어올렸다고 보고했다. 아동 행동 분류 정확도는 83.53%, 위험 행동 인식 정확도는 96.14%로 나타나 기존 대비 큰 폭의 개선을 확인했다. 또한 주의(attention) 맵 분석으로 모델이 위험 판단에 중요한 영역을 포착함을 보여주며, 문맥 기반 모니터링이 미감시 환경에서도 안전 탐지에 실용적일 가능성을 뒷받침했다.



### Minimalist Preprocessing Approach for Image Synthesis Detection (https://arxiv.org/abs/2606.25297)
Comments:
          SOICT 2024

- **Prior Approaches**: 생성형 이미지(특히 GAN·Diffusion) 탐지는 주로 주파수 영역(FFT/DCT, high-frequency 강조)이나 사전학습 모델(CLIP, ViT 등)을 활용해 가짜의 아티팩트를 분류하는 방식이 많았다. 그러나 이러한 방법은 스펙트럼 선택/처리에 민감하거나, CLIP·ViT·ResNet50 같은 대형 백본 및 고비용 추론이 요구되는 경우가 많아 모바일 배포가 어렵다.

- **Core Contribution**: 이 논문은 픽셀 이웃 간의 밝기 변화(gradient)를 경량 전처리로 포착하는 Adjacency Difference Orientation Filter(ADOF)를 제안한다. ADOF는 색상 영향은 줄이고(그레이스케일 기반) 고주파 성격의 변화량을 강조해, 작은 CNN으로도 생성/진짜를 구분하도록 돕는 것이 핵심이다.

- **Technical Challenges**: 문제는 생성 이미지와 실제 이미지의 차이를 “색 정보”가 아닌 “인접 픽셀의 미세한 요동”으로 안정적으로 잡아내면서, 동시에 경량 모델이 학습하기 쉬운 표현으로 변환하는 것이다. 이들은 finite difference로 x·y 방향 그레이스케일 gradient를 계산해 크기/방향을 구하고, 가장자리 성격의 각도(±π/2 주변)에서는 값을 0으로 만드는 휴리스틱을 더해 불필요한 에지 영역을 배제했으며, ResNet50에서 일부 레이어를 제거한 가벼운 CNN에 결합했다.

- **Empirical Impact**: 여러 벤치마크(9-GAN, DiffusionForensics, Ojha Test Set)에서 ADOF는 기존 SOTA와 비슷하거나 더 높은 정확도를 보였다(예: Ojha 94.9%급, DiffusionForensics 98.3%). 또한 연산 관점에서도 RINE 대비 97.8%, LGrad 대비 57.8% 수준의 계산량 절감을 보고해 저사양 스마트폰 같은 단말 배포 가능성을 실증했다.



### Evaluation Protocols and Validation for Cameras in Indoor Healthcare Monitoring (https://arxiv.org/abs/2606.25284)
- **Prior Approaches**: 카메라 기반 실내 모니터링은 활발히 연구되고 있지만, 기존 검증은 대개 장치의 계측 성능(예: depth 정확도)이나 다운스트림 알고리즘(예: pose 추정 정확도) 중 하나에만 초점이 맞춰져 왔다. 또한 조명 변화, 가림(occlusion), 카메라 설치 높이·각도 같은 실제 배치 요인을 체계적으로 함께 다루지 않아 재현 가능한 카메라 선택 가이드가 부족했다.

- **Core Contribution**: 이 논문은 의료 환경의 카메라 선택을 돕기 위해 두 가지 기술 검증 프로토콜을 제안한다. 하나는 RGB/RGB-D의 계측(메트롤로지) 성능을 평가하고, 다른 하나는 pose estimation 파이프라인과 결합했을 때의 신뢰성을 RTMO와 YOLO26 같은 서로 다른 구조의 state-of-the-art pose estimator로 검증한다.

- **Technical Challenges**: 핵심 과제는 실내에서 흔한 조명 변동, 카메라 위치(높이·시야각), occlusion이 계측치와 3D 복원/관절 추정에 미치는 영향을 ‘동시에’ 측정 가능한 방식으로 고립·통제하는 것이다. 이를 위해 조명(lux)과 거리, 열적 드리프트, field of view를 포함한 메트롤로지 평가를 만들고, pose 단계에서는 Vicon gold-standard와의 비교 하에 반복 실험에서 제어 변수 조합을 설계해 카메라별 민감도를 정량화했다.

- **Empirical Impact**: 실험 결과 depth bias는 5m에서 카메라에 따라 50mm부터 1400mm를 넘는 수준까지 크게 달라, 장치 간 성능 격차가 매우 큼이 드러났다. 반면 2D pose estimation은 카메라 간 mAP가 대체로 78~90%로 비슷했지만, 3D 복원 오차(MPJPE)는 104mm~365mm로 훨씬 차이가 커 depth sensing 품질과 밀접히 연동됨을 보여준다. 나아가 조명·가림 같은 환경 요인은 3D 성능에 카메라와 estimator에 따라 서로 다른 영향을 주며, 평가 범위 내에서는 설치 높이 영향이 상대적으로 작다는 실증적 결론을 제시해 임상/가정용 배치 의사결정을 지원한다.



### MRI2Rep: Autoregressive Structured Report Generation for 3D Liver MRI (https://arxiv.org/abs/2606.25279)
Comments:
          MICCAI 2026

- **Prior Approaches**: 기존 의료 리포트 생성·3D MRI 연구는 주로 표현학습, 분류, VQA에 초점을 맞추거나 generic 리포트 생성에 머물렀습니다. 또 3D 간 내시경/영상-텍스트 모델도 특정 병변 마스크 같은 병변 수준 주석이 필요한 경우가 많아, 임상에서 흔한 “리포트 텍스트만 있는” 데이터로는 LI-RADS 구조화까지 end-to-end로 가기 어려웠습니다.

- **Core Contribution**: MRI2Rep는 3D 간 MRI를 입력으로 받아 LI-RADS를 반영한 구조화된 리포트를 생성하는 end-to-end autoregressive 프레임워크를 제안합니다. 핵심은 Report-to-Label Canonicalization(RLC) 모듈로, 자유형(free-text) 리포트에서 부정·불확실성·표현 차이를 정리해 closed-vocabulary diagnostic sequence로 바꾼 뒤, 병변 위치 주석 없이도 시퀀스를 예측해 템플릿으로 리포트까지 렌더링합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 리포트 수준 감독이 노이즈가 심한데(부정, 용어 변이), 3D MRI는 데이터가 상대적으로 적어 과적합 위험이 커진다는 점입니다. MRI2Rep는 LI-RADS 결정 로직을 사용해 “증거 문장(evidence sentence) 기반의 재현 가능한 라벨화”를 만들고, ART–PV 다중 시퀀스와 3D self-attention을 갖춘 encoder–decoder로 장거리 3D 맥락 및 washout 동역학을 학습하며, word dropout과 보조 lesion 존재 예측으로 디코더의 시각 의존을 강화합니다.

- **Empirical Impact**: 단일기관 10년 코호트의 3,929개 MRI-리포트 페어(테스트 383)에서 MRI2Rep는 case-level sensitivity 76.0%, lesion-level F1 29.4%, liver-level accuracy 82.4%를 달성했습니다. 블라인드 판독 연구에서 AI 생성 리포트의 임상 수용 비율이 75%/70%로, 원본(95%/100%) 대비 낮지만 LLM 기반 자동 심사 LLM-Eval에서는 61.8%로 더 엄격하게 평가돼 보수적 대리지표로 활용 가능성을 보였습니다.



### Heterogeneous and Adept Snapshot Distillation for 3D Semantic Segmentation (https://arxiv.org/abs/2606.25278)
Comments:
          11 pages

- **Prior Approaches**: 기존 3D semantic segmentation 성능 향상에는 multi-modal fusion과 model ensembling이 많이 쓰인다. 하지만 multi-modal fusion은 point cloud와 image를 함께 처리하느라 계산·메모리 부담이 크고, ensembling은 expert 모델 수가 늘수록 학습/추론 비용이 선형으로 증가해 실서비스 적용이 어렵다. 또한 cross-modal knowledge distillation에서도 학습용 multi-modal 입력을 어떻게 고르느냐가 성능에 큰 영향을 주는 한계가 있었다.

- **Core Contribution**: 이 논문은 단일 point-cloud 기반 네트워크가 multi-modal 모델과 여러 expert의 지식을 흡수하되, 추론 시에는 image 없이 동작하도록 하는 HAS-KD를 제안한다. 핵심은 Information-oriented Heterogeneous Distillation (IHD)로 multi-modal teacher의 정보를 uni-modal student에 증류하고, Adept Snapshot Distillation (ASD)로 학습 중 생성된 snapshot들을 class별 expert로 취급해 multi-teacher 지식을 효율적으로 옮기는 것이다. 결과적으로 추가 inference 부담 없이 정확도를 끌어올리는 것을 목표로 한다.

- **Technical Challenges**: 문제는 (1) cross-modal distillation에서 어떤 image를 선택해야 teacher 품질이 올라가는지, (2) 여러 teacher를 함께 쓸 때 underperform하는 teacher의 잡음을 어떻게 줄일지이다. IOF는 연속 프레임에서 각 객체에 대해 semantic abundance가 최대가 되는 관측을 골라 더 informative한 multi-modal teacher를 만들고, ASD는 class별 최고 성능 체크포인트만 해당 class voxel에 대해 supervision을 제공해 ‘적재적소’ 증류를 수행한다. IHD는 distillation으로 학생이 multi-modal과 유사한 정보를 학습하게 하되, 실제 추론 단계에서는 image 분기 없이 단일 모델만 사용하도록 설계된다.

- **Empirical Impact**: ScanNetV2와 S3DIS에서 HAS-KD는 state-of-the-art 성능을 달성하며, 특히 multi-dataset 학습 없이도 강력한 baseline(Point Transformer V3)보다 mIoU를 추가로 개선한다. ablation에서는 IHD와 ASD 각각이 유의미한 성능 향상을 주고, IOF가 Random Filtering 대비 더 나은 teacher 및 학생 성능을 만든다는 점이 확인된다. 또한 class별 expert ensemble이 평균 기반 ensemble보다 더 높은 ensemble teacher 품질과 학생 이득을 제공해, 효율적인 증류 전략의 실효성을 보여준다.



### CoGeoAD: Hierarchical Color-Geometric Fusion with Multi-View Attention for Zero-Shot 3D Anomaly Detection (https://arxiv.org/abs/2606.25273)
Comments:
          ICML 2026

- **Prior Approaches**: 기존 2D 기반 anomaly detection은 RGB만 다뤄 표면 텍스처는 잘 보지만, 3D 구조적 결함까지 일관되게 잡기 어렵다. 반대로 3D point cloud 기반 접근은 기하를 잘 포착하나 색/재질 질감에 기반한 결함은 시점(viewpoint) 영향으로 놓치기 쉽다. CLIP 기반 zero-shot 3D anomaly detection 시도들도 대체로 RGB를 기하의 부가 신호로만 취급해, 색-기하를 제대로 통합한 단일 프레임워크로의 탐지가 부족했다.

- **Core Contribution**: CoGeoAD는 frozen CLIP을 기반으로 color와 geometry를 한 프레임워크에서 통합하는 zero-shot 3D anomaly detection 방법이다. 핵심은 RGB와 점구름을 pixel-aligned로 매칭되는 paired multi-view로 렌더링해, 구조 이상은 geometry 투영으로, 텍스처 이상은 color view로 동시에 감지하도록 설계한 점이다. 또한 Data-Driven Multi-View Attention(MVA)과 Multi-Stage Color-Geometric Fusion(MS-CGF)로 다층 특징을 계층적으로 융합해 표면 결함부터 구조적 이상까지 놓치지 않게 만든다.

- **Technical Challenges**: 어려움은 (1) 3D에서 여러 시점의 가림(occlusion) 때문에 2D 투영 점수가 노이즈가 되기 쉽고, (2) 2D 사전학습 특징을 그대로 3D anomaly localization에 쓰면 색-기하 간 도메인 갭이 생긴다는 데 있다. CoGeoAD는 pinhole 투영과 visibility mask를 통해 occlusion 잡음을 걸러내며, point-to-pixel correspondence로 2D anomaly score를 3D로 정밀 백프로젝션한다. 더 나아가 MVA는 이미지에서 직접 시점 중요도를 계산해 고정 가중치 편향을 줄이고, MS-CGF는 CLIP 다층 특징(깊이·유사도·시점·모달리티)을 단계적으로 결합해 미세 이상을 계층적으로 강화한다.

- **Empirical Impact**: MVTec3D-AD와 Eyecandies에서 CoGeoAD는 zero-shot 설정에서 state-of-the-art 성능을 달성했다. MVTec3D-AD에서는 I-AUROC 87.4%, P-AUROC 97.5%, AUPRO 91.9%로 기존 최신 CLIP 계열 대비 개선 폭을 보였고(예: GS-CLIP 대비 I-AUROC +3.8%, AUPRO +5.5%), Eyecandies에서도 모든 메트릭에서 경쟁 방법을 일관되게 앞섰다. 결과적으로 단일(또는 단순 앙상블) 모달리티 중심 한계를 넘어, 색과 기하를 계층적으로 융합하는 접근이 복잡한 산업 시나리오의 미세 surface/structural anomaly 모두에 효과적임을 실증했다.



### Pre-Warm: Input-Conditioned Weight Initialization for Convolutional Neural Networks (https://arxiv.org/abs/2606.25256)
- **Prior Approaches**: 기존 가중치 초기화(Kaiming, Glorot)는 활성 분산을 보존하도록 통계적으로 설계됐지만, 실제 데이터의 지역 구조(모서리·텍스처)와는 무관하게 동일한 랜덤 분포를 사용합니다. LSUV, batch 통계 기반 초기화, greedy layer-wise pretraining 등은 데이터를 더 적극적으로 보지만 보통 forward pass 반복이나 전처리 비용이 필요합니다.

- **Core Contribution**: Pre-Warm은 학습 없이(zer0-training-cost) 첫 번째 convolutional layer의 초기값을 데이터에 조건화하는 간단한 방법을 제안합니다. 단일 학습 배치에서 mean-centered local patches를 뽑아 MiniBatchKMeans로 군집화한 뒤, 그 centroid를 첫 레이어 필터 절반에만 주입해 데이터 기반 구조를 주되 Kaiming의 랜덤 다양성은 유지합니다.

- **Technical Challenges**: 핵심은 (1) 입력 분포에서 의미 있는 patch 수와 (2) 그 centroid를 필터에 주입할 때의 스케일을 안정적으로 정하는 문제입니다. 저자들은 grid search로 grayscale의 patch_size=kernel_size, n_clusters=F/2 같은 닫힌형 규칙과 Otsu foreground density 기반 npatches 예측을 도출하고, 자연색 이미지는 Otsu 대신 mean L2 norm으로 npatches를 예측하며 scale은 데이터 타입별 범위에서 주로 예측 가능하도록 제한합니다.

- **Empirical Impact**: MNIST, Fashion-MNIST, CIFAR-10, SVHN, CIFAR-100에서 표준 Kaiming 대비 통계적으로 유의한 성능 향상을 확인했으며, 모든 데이터셋에서 p<0.05를 보고합니다. 특히 SVHN은 8/8 시드에서 승리하며 p=0.0007, CIFAR-100은 7/8 승리와 p=0.0033으로 +0.68%p의 큰 개선을 보였습니다. 또한 BatchNorm이 포함된 구조에서도 첫 레이어에만 조건 신호를 넣는 방식이 최적화 궤적을 실질적으로 개선할 수 있음을 시사합니다.



### Cross-Modality Structural Guidance in 3D Latent Diffusion for Robust FLAIR Super-Resolution (https://arxiv.org/abs/2606.25255)
- **Prior Approaches**: 기존 초해상도(SR) 연구는 생성형 모델의 환각(hallucination)과 고정 열화(예: 4× 다운샘플링) 학습으로 인한 강건성 부족이 핵심 한계로 지적돼 왔다. 특히 두꺼운-slice FLAIR의 LR–HR 격차가 커질수록 해부학적으로 그럴듯하지만 부정확한 고주파 디테일이 진단 신뢰도를 해칠 수 있다. 또한 2D 기반 접근은 슬라이스 간 일관성이 깨져 병변 경계나 주변부 구조에서 불연속이 나타나기 쉽다.

- **Core Contribution**: MR-DiffuSR은 두꺼운-slice FLAIR를 HR로 복원할 때, HR T1w를 구조적 priors(스캐폴드)로 활용해 환각을 줄이는 Multi-Resolution Diffusion 기반 프레임워크를 제안한다. T1w→FLAIR를 직접 합성하지 않고, 3D latent space에서 LR FLAIR의 대비는 유지하면서 해부학적 기하만 T1w로 고정하도록 설계했다. 이를 위해 cross-modality structural swin-attention(CMSSA)가 T1w에서 유도한 structural attention map을 FLAIR latent에 적용해 해부학 구조와 modality-specific contrast를 분리한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 심한 downsampling에서도 구조 일관성을 유지하면서, (2) T1w의 해부학 정보를 대비 복원에 과도하게 끌어오지 않는 조건부 생성이 필요하다는 점이다. MR-DiffuSR은 3D VQ-GAN으로 volumetric 정보를 latent로 압축한 뒤, residual shifting 기반 3D latent diffusion으로 빠른 복원을 수행하고 T1w latent을 reverse 과정의 구조 조건으로 결합한다. 여기에 downsampling factor s∈{4,6,8,10}를 오가며 학습하는 mixed-scale degradation 전략과 DINOv3 기반 perceptual loss를 더해 고주파 의미 디테일을 보존하도록 최적화했다.

- **Empirical Impact**: ADNI-4에서 모든 downsampling factor에 대해 MR-DiffuSR은 CNN 및 2D diffusion 기반 SR을 능가하며 평균 PSNR 32.46dB, SSIM 0.97, LPIPS 0.07을 기록했다. 특히 약 7mm 상당 두꺼운 slice(10×)에서도 성능이 급락하지 않아 WMH 세그멘테이션 Dice가 baseline 대비 크게 유지되며, Dice 0.51(기준)에서 0.63으로 향상됐다. 이는 두꺼운 임상 FLAIR에서도 구조적 신뢰성과 병변 정량 분석의 실용성을 함께 높일 수 있음을 보여준다.



### Multilingual Hematology Visual Question Answering Datas (https://arxiv.org/abs/2606.25246)
Comments:
          Under Review

- **Prior Approaches**: 기존 VLM 기반 의료 비전-언어 연구는 시각과 텍스트를 함께 다루지만, 대개 영어 중심 자원에 의존해 다국어 의료 현장 적용성이 낮았다. 또한 일부 데이터셋은 형태학 정보를 학습하더라도 캡션/자율 생성이나 언어 감독이 약해 임상 검증된 VQA 형태의 다국어 벤치마크는 부족했다.

- **Core Contribution**: 본 논문은 백혈병(Leukemia)과 정상 백혈구(WBC) 형태를 대상으로 한 임상 검증 bilingual VQA 벤치마크 WBCMor VQA를 제안한다. LeukemiaAttri와 WBCAtt의 morphology-aware 어노테이션을 바탕으로 영어-우르두(Urdu) 쌍을 구축해, 다국어 의료 AI 개발과 평가를 가능하게 한다.

- **Technical Challenges**: 핵심 기술 과제는 우르두(Urdu) 형태학 용어의 번역 품질과 임상적 일관성을 확보하는 것이다. 영어 VQA를 NLLB-200으로 번역한 뒤, 도메인별 Urdu hematology dictionary로 5,000건 이상 용어를 교정하고 prompt-guided re-translation으로 표준화를 강화했으며, 생성/번역 전후 단계에서 의료 전문가 검증을 수행했다.

- **Empirical Impact**: 평가는 Qwen2-VL-2B, InternVL2.5-2B 등 오픈소스 VLM을 영어/우르두/혼합 세팅에서 zero-shot과 fine-tuning으로 비교해, 도메인 특화 fine-tuning이 일관되게 성능을 끌어올림을 보였다. 110K bilingual question-answer pairs(20K 단일세포 이미지)를 공개해 향후 다국어 임상 VQA 및 언어-적응형 의료 AI 연구의 기반을 제공한다.



### OrthoTrack: Continuous 6-DoF UAV Trajectory Estimation Anchored in Public Orthophotos (https://arxiv.org/abs/2606.25245)
Comments:
          ECCV 2026 - Project page: this http URL

- **Prior Approaches**: 기존 6-DoF 자세 추정은 주로 visual odometry·SLAM에 의존하지만, 시간이 지날수록 drift가 누적되어 상대적(스케일 불명) 궤적에 그치는 경우가 많다. 반면 single-frame geo-localization은 시간 연속성을 버리고 처리 속도가 느려 실시간 UAV 운용에 불리했다.

- **Core Contribution**: OrthoTrack은 학습 없이(training-free) 공개 orthophoto와 surface model을 지도 prior로 삼아 연속적인 6-DoF UAV 궤적을 각 프레임에서 절대·미터 스케일로 추정한다. 핵심 아이디어는 키프레임을 orthophoto에서 매칭한 뒤 surface model로 대응점을 metric 3D로 ‘리프트’하고, 이를 광류로 중간 프레임에 전파해 GPS나 사후 정렬 없이 연속 자세를 만든다는 점이다.

- **Technical Challenges**: 지도 기반 정합에서 가장 큰 난제는 (1) orthophoto-이미지 간 대응점을 안정적으로 찾고 (2) 대응점을 정확한 metric 3D로 올리며 (3) 이 map-anchored 대응을 광류로 연속 프레임에 전파하면서 누적 오류를 최소화하는 것이다. OrthoTrack은 키프레임에서 correspondences를 surface model로 3D로 승격(lift)한 뒤, 그 map-anchored 정보를 optical flow로 중간 프레임에 전파해 미터 스케일 절대 자세를 매 프레임 계산하도록 설계했다.

- **Empirical Impact**: 또한 multi-temporal orthophoto와 co-registered 다중 모달 geodata, 그리고 조밀한 6-DoF ground truth를 제공하는 MovingDrone Dataset을 제안해 벤치마크 기반 검증을 강화했다. MovingDrone 및 실환경 벤치마크에서 OrthoTrack은 단일 GPU로 실시간 동작하며, oracle scale·alignment을 받는 강한 베이스라인조차 큰 폭으로 능가해 현장 적용성과 성능을 동시에 입증했다.



### Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds (https://arxiv.org/abs/2606.25234)
- **Prior Approaches**: 기존 신경 표현 분해(예: sparse autoencoders, SAE)는 개념을 ‘고립된 한 방향’으로 두고 성분을 해석 가능한 원자(atom)들의 희소 조합으로 설명하려 해왔다. 하지만 최근 연구들은 개념이 활성 공간의 저차원 영역에서 ‘기하학적 구조(서브스페이스/매니폴드)’로 구현되는 경우가 많다는 점을 보여, 방향 단위 가정과 불일치한다.

- **Core Contribution**: 이 논문은 structured sparsity의 관점에서, 개념이 저차원 매니폴드들의 ‘희소 합’으로 생성된다는 가정에 가장 잘 맞는 분해 편향으로 block sparsity(=group sparsity)를 제안한다. 또한 이를 구현하는 세 가지 Block-Sparse Featurizer(BSF)를 설계하고, 최종적으로 복원된 개념의 차원이 보통 2~4차원임을 최소기술길이(MDL) 분석으로 정량화한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘어떤 기하를 단위로 분석할 것인가(블록/서브스페이스/매니폴드)’와 ‘희소성 제약을 어떻게 실제 학습 아키텍처에 반영할 것인가’에 있다. 저자들은 멀티-매니폴드 혼합을 데이터 생성 과정(DGP)으로 두고 MAP 추정에서 block sparsity 형태가 유도됨을 보인 뒤, (1) Vanilla BSF(TopK 블록 선택), (2) Grassmannian BSF(블록 내 직교 조건과 인코더-디코더 결합), (3) Group Lasso BSF(soft-threshold로 convex surrogate 적용)로 이를 구현한다.

- **Empirical Impact**: 합성 데이터에서는 BSF들이 알려진 factor를 단일 forward pass로 거의 오라클 수준까지 복원하며, 방향 기반(TopK SAE)이나 다르게 가정하는 방법들은 다차원 factor를 원자에 깨뜨려 매니폴드 형태를 잃는 문제가 드러났다. 실제로는 InceptionV1에서 curve detector들이 단일 연속 curve 매니폴드를 읽는 정황을 재맥락화하고, DINOv3에서는 그림자·조명 같은 새로운 매니폴드를 발견했으며, SDXL에서는 diffusion 이미지 생성을 manifold steering으로 해석 가능하게 제어하는 데 BSF가 활용됨을 보인다.



### MJEPA: A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual Learning (https://arxiv.org/abs/2606.25225)
- **Prior Approaches**: 기존 audio-visual self-supervised 학습(AV-SSL)은 대체로 오디오와 비디오에 대해 별도의 modality-specific encoder를 두고, contrastive·reconstruction·distillation 등 복합 목적함수를 조합하는 방식이 많았습니다. 이런 구조는 cross-modal synergy를 만들기 어렵고, 모델 크기·학습 파이프라인 확장에도 제약이 생긴다고 지적합니다. 또한 JEPA 계열이 있더라도 실제 적용은 주로 단일 모달리티에 머물러 멀티모달 설정이 공백으로 남아 있었습니다.

- **Core Contribution**: 이 논문은 Joint-Embedding Predictive Architectures(JEPA) 원리를 멀티모달로 확장한 MJEPA를 제안합니다. 핵심은 단일, unified encoder로 audio·video를 모두 처리하고, 하나의 predictive objective만으로 intra-modal(동일 모달 내 예측)과 cross-modal(다른 모달에서 같은 의미 표현 예측)을 동시에 수행한다는 점입니다. 특히 cross-modal prediction이 없으면 shared encoder가 unimodal baseline보다 성능이 떨어지지만, 이를 넣으면 두 모달 모두 서로 이득을 본다는 ‘positive transfer’ 가설을 실험으로 확인합니다.

- **Technical Challenges**: 기술적 난점은 “공유 인코더”를 멀티모달에 적용할 때 두 모달의 표현이 서로 충돌해 품질이 저하될 수 있다는 문제입니다. 연구진은 마지막 레이어의 고수준 임베딩에 대해서만 cross-modal predictor(MLP + global mean pooling)를 얹어, 비디오 토큰과 오디오 토큰 간 토큰 단위 대응이 없어도 의미 정렬이 되도록 학습 신호를 설계했습니다. 또한 representation collapse를 막기 위해 stop-gradient 및 EMA(target encoder) 같은 JEPA/Byol 계열의 안정화 장치를 유지하면서, reconstruction/contrastive 같은 복잡한 손실 없이 단일 JEPA 계열 목적만으로 학습을 구성했습니다.

- **Empirical Impact**: 실험은 AudioSet-20K·ESC-50·FSD50K·Kinetics-400·SSv2 등 오디오/비디오/오디오-비디오 벤치마크에서 frozen attentive probe 프로토콜로 검증되며, MJEPA의 단순성 대비 일관된 성능 우위를 보여줍니다. 특히 AudioSet-20K에서 frozen ViT-g가 기존 최고 frozen baseline을 6.8 mAP 이상 능가하고, ESC-50·FSD50K에서는 fully finetuned 모델을 능가한다고 보고합니다. 더 나아가 비디오 벤치마크에서도 audio 데이터를 함께 쓰면 더 적은 비디오 학습 데이터(약 10× 적음)로도 거의 SOTA급에 근접해, cross-modal 예측 기반 스케일링의 실용성을 강조합니다.



### Cage-based Texture Transfer with Geometric Filtering (https://arxiv.org/abs/2606.25220)
Comments:
          Accepted to SIGGRAPH 2026

- **Prior Approaches**: 기존 실시간 texture transfer는 UV를 근접 기준으로 투영하지만, 의도하지 않은 정점(Non-Cosmetic Zones, NCZ)에까지 투영이 번지며 아티팩트를 만들기 쉽다. 저지연 방법은 속도는 빠르지만 NCZ 억제가 취약하고, robust 성능을 내려면 대규모 모델 학습·메모리 비용이 커서 모바일/실시간 적용이 어렵다. NCZ를 photometric/edge guidance로 찾는 시도도 정확하더라도 계산량이 많아 인터랙티브·모바일 환경에 부담이 된다.

- **Core Contribution**: 이 논문은 cage 기반 기하 필터링(Geometric Filtering)으로 NCZ를 자동으로 식별해 아티팩트를 억제하는 프레임워크를 제안한다. 핵심 아이디어는 volumetric deformation에서 쓰이던 auxiliary cage mesh를 공간 기준으로 재활용해, 타깃 메시의 점들이 “유효 표면인지/차단·가려진 영역인지”를 빠르게 판정하는 것이다. 이를 통해 무거운 모델 없이도 즉시 배포 가능한 아티팩트 억제 texture transfer를 목표로 한다.

- **Technical Challenges**: cage를 기준으로 NCZ를 판정하려면 self-intersection(내부 기하)과 developer가 제외한 cage coverage(가려진 영역)를 모두 효율적으로 구분해야 한다. 논문은 각 정점에서 가장 가까운 cage 삼각형 법선 방향으로 레이 캐스팅을 수행하고, 타깃 메시 및 cage와의 교차 여부를 통해 NCZ를 표시한다. 또한 정점 단위 대신 메시를 connected component로 분할한 뒤, 세그먼트별 투영 성공 면적 비율로 Threshold-based elimination을 적용해 계산을 줄인다(KD-Tree로 공간 질의 최적화).

- **Empirical Impact**: Naive texture transfer 대비 시각적으로 NCZ로의 projection bleeding을 제거하며, manual authoring에 가까운 아티팩트 억제 효과를 지향한다. Android Samsung Tablet S6 Lite에서 약 4,782-triangle 메시 기준으로 70ms 내 처리가 보고됐고, 복잡도는 KD-Tree 질의 최적화로 O(V log(N+M)), 메모리는 O(V+N+M)로 선형 스케일을 보인다. 또한 약 20MB 수준의 비교적 작은 메모리로 모바일 계층에서도 동작해, resource-constrained 환경에서 실시간 asset 상호운용성을 넓힐 수 있는 실용적 의미가 있다.



### Reflective VLA: In-Context Action Consequences Make VLAs Generaliz (https://arxiv.org/abs/2606.25215)
- **Prior Approaches**: 기존 vision-language-action(VLA) 모델은 보통 reactive 방식으로, 현재 관측과 지시만으로 다음 행동을 예측해 배치 환경에서 필요한 상태를 단일 프레임으로 충분히 알 수 있다고 가정한다. 하지만 실제 로봇에서는 카메라-로봇 기하, 캘리브레이션, actuation bias 같은 embodiment-specific 요인이 한 장면에서 식별이 어렵고, 그 결과 학습 환경에 과적합되며 배치 일반화가 약해진다. 최근 temporal context·메모리 기반 VLA는 상태 추정은 개선하지만, 실행된 행동과 그 결과(관측 변화)를 명시적으로 연결해주는 ‘action–consequence binding’이 부족하다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Reflective VLA를 제안하며, 매 의사결정을 observation–action–consequence triplets의 컨텍스트에 조건화하는 in-context learning 관점으로 전환한다. triplet은 “무엇을 봤고 무엇을 실행했으며, 실행 후 장면이 어떻게 바뀌었는지”를 함께 기록해 배치 환경에서 행동-효과 매핑을 드러내도록 설계됐다. 모델은 각 결정을 과거 triplet과 현재 관측을 함께 참조해, 단일 프레임만으로 불가능한 embodiment 요인 분해를 돕는다.

- **Technical Challenges**: 핵심 구현 난제는 (1) 이미지·자세(proprioception)·연속 행동 chunk 등 이질 모달리티를 병목 없이 한 인과 시퀀스로 패킹하는 문제와 (2) 듀얼시스템(VLM prefix + continuous action expert)에서 컨텍스트를 학습·추론 시 효율적으로 전달하는 문제다. 논문은 모든 관측 모달리티를 공유 VLM token space로 라우팅하고, action expert가 과거 관측/행동/결과 증거를 직접 attend하도록 구성한다. 또한 K개의 멀티프레임을 별도 forward 없이 학습하도록 block-causal mask로 학습을 병렬화하고, KV-cached real-time inference가 가능하도록 같은 인과 구조를 추론에도 활용한다.

- **Empirical Impact**: 실험에서 Reflective VLA는 LIBERO와 SimplerEnv-Bridge 같은 표준 설정에서 기준 reactive baseline 대비 성능을 유지(또는 소폭 향상)하며, LIBERO에선 평균 성공률 97.6%로 state-of-the-art급을 보였다. 분포 이동이 큰 LIBERO-Plus 및 LIBERO-Plus-Hard에서는 test-time fine-tuning 없이도 평균 성공률이 각각 5.4, 4.2 percentage points 향상되어 배치 일반화가 개선됨을 확인했다. 특히 history-only ablation은 단순 컨텍스트 길이 증가가 아니라 action–consequence에 해당하는 ‘결과 관측’ 증거가 환경 간 일반화의 핵심임을 뒷받침한다.



### Neural Network Quantization by Learning Low-Loss Subspaces (https://arxiv.org/abs/2606.25087)
Comments:
          30 pages, 7 figures

- **Prior Approaches**: 신경망 양자화는 FP(정밀) 모델의 성능을 최대한 유지하면서 파라미터를 이산(discrete) 표현으로 바꾸는 방법을 찾는 문제다. 그러나 이산 제약을 강제하면 최적화된 최소점에서 멀어지며 성능 저하가 흔하다. 최근에는 저손실 FP 해가 고립된 점이 아니라 손실 지형의 연결된 저손실 부분공간에 존재해, 그 부분공간에서 샘플링해도 정확도가 잘 유지된다는 관찰이 제시돼 왔다.

- **Core Contribution**: 이 논문은 양자화된 모델이 FP 모델의 저손실 부분공간 안에 “놓이도록” 양자화 친화적인 경로를 학습하는 접근을 제안한다. 가중치 공간에서 손실을 최소화하도록 양자화-aware 선형 path를 학습하고, 그 부분공간의 midpoint가 설계상 양자화에 유리함을 보인다. 또한 해당 midpoint를 직접 양자화하면 양자화-aware training과 견줄 만한 성능을 얻을 수 있음을 제시한다.

- **Technical Challenges**: 핵심 난관은 양자화 제약을 학습 중에 명시적으로 적용할 때 생기는 성능 붕괴를 피하면서, 저손실 부분공간을 실제로 양자화 가능한 구조로 만들 수 있는지다. 논문은 훈련 단계에서 straight-through estimator를 쓰거나 명시적 discretization을 수행하지 않고도, 손실 최소화를 목표로 weight space에서 양자화-aware 선형 경로를 학습해 midpoint가 양자화-friendly가 되도록 한다.

- **Empirical Impact**: 제안된 절차는 midpoint의 direct quantization만으로도 quantization-aware training과 유사한 정확도를 보이며, 양자화 성능을 “부분공간 정합” 관점에서 설계할 수 있음을 실증한다. 이는 기존의 양자화 방법처럼 학습 중 이산화를 직접 강제하는 대신, FP 저손실 부분공간의 성질을 활용해 더 안정적인 양자화 파이프라인을 제안한다는 점에서 의미가 있다.



### Are We There Yet? Exploring the Capabilities of MLLMs in Assistive AI Applications (https://arxiv.org/abs/2606.25084)
- **Prior Approaches**: 기존의 보조형 AI 연구는 대부분 OCR이나 문서 이해 같은 파이프라인을 중심으로 텍스트를 처리해 왔지만, egocentric 환경에서는 가림·흔들림·조명 문제로 정확도와 안정성이 쉽게 무너졌다. COCO-Text, TextVQA 같은 벤치마크도 정적·제작된 이미지 위주라 실제 사용에서의 잡음, 맥락 의존성, 실시간 상호작용을 충분히 반영하지 못했다. 한편 MLLM은 image captioning, visual question answering 등에서 강하지만, 시각 기반의 실제 보조 시나리오(동전/메뉴/장문 읽기 등)에서의 준비도는 여전히 불명확했다.

- **Core Contribution**: 이 논문은 MLLM이 Assistive AI를 실제로 지원할 수 있는지, egocentric(1인칭) 비전·언어 입력을 바탕으로 체계적으로 진단한다. 이를 위해 head-mounted GoPro로 데이터를 수집하고, NetraLink라는 로컬 오프라인형 프로토타입을 구축해 장면 텍스트 인식, 지폐 인식, 내비게이션 지원, 다국어 메뉴 읽기·해석, 장문(책) 이해의 5개 작업을 평가한다. 또한 각 작업에 맞춘 맞춤형(실사용 지향) 데이터셋을 만들고 여러 오픈소스 MLLM의 강점과 실패 모드를 비교한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 흔들림·가림·작은 글자·비정형 폰트·조명 등 현실 시각 잡음 속에서 텍스트를 정확히 읽고, (2) 그 정보를 자연어 질의에 맞게 맥락 추론으로 연결하며, (3) 실시간성까지 만족해야 한다는 점이다. 저자들은 wake-word 감지와 VAD 기반 침묵 분할로 상호작용을 자연스럽게 만들고, 캡처된 이미지와 STT 전사 결과를 단일 멀티모달 프롬프트로 묶어 vision–language inference를 수행하는 모듈형 구조(NetraLink)를 제안한다. 또한 프롬프트 템플릿화·지연(latency) 측정·로그/재현성 확보를 통해 “정확도 vs 응답성” 트레이드오프와 오류 원인을 분석 가능하게 했다.

- **Empirical Impact**: 평가는 장면 텍스트 인식에서 모델 크기만으로 성능이 결정되지 않음을 보여주며, FastVLM-7B 같은 모델이 높은 ANLS를 기록한 반면 더 작은 모델이 속도·정확도 균형에서 유리할 수 있음을 시사한다. 실패 사례 분석에서는 비정형 폰트, 작은 글자, 텍스트가 복잡하게 밀집된 장면, 조명/블루밍 같은 조건이 반복적으로 오류를 유발함을 구체적으로 제시한다. 이러한 결과는 보조형 비전-언어 시스템이 단순 OCR 대체가 아니라, 현실 입력에 대한 견고성·추론 정합성·실시간 제약을 동시에 만족해야 한다는 점을 분명히 해 향후 deployable assistive 모델 연구에 실증적 기준선을 제공한다.



### FreeStory: Training-Free Character Consistency for Free-Form Visual Storytelling (https://arxiv.org/abs/2606.25079)
- **Prior Approaches**: 기존 비학습(inference-only) 시각 스토리텔링은 캐릭터 설명이 매 프롬프트에 반복되는 구조를 가정하고, attention의 key-value(KV) 같은 특징을 프레임 간 재사용해 정체성을 유지했습니다. 다만 이런 템플릿 의존은 자연스러운 서사(초기 1회 소개 후 pronoun/타입 표현으로 지칭)와 어긋나며, 멀티 캐릭터에서 “어떤 언급이 어떤 인물인지”를 매번 명시해야 하는 한계가 있습니다.

- **Core Contribution**: 이 논문은 자유형(free-form) 프롬프트에서도 일관된 캐릭터를 생성하는 문제를 새롭게 정식화하고, 이를 entity-grounded feature reuse로 환원한 training-free 프레임워크 FreeStory를 제안합니다. 또한 이후 프롬프트의 참조 언급을 초기 캐릭터 설명과 연결하는 entity grounding을 도입해, 명시적 반복 없이도 정체성 보존이 가능하게 만듭니다.

- **Technical Challenges**: 핵심 난점은 자유형 언어에서 “언급-인물 대응”이 암묵적이며, 그에 맞춰 이미지 내 문자(인물) 영역을 시공간적으로 안정적으로 찾고 특징을 정밀히 주입해야 한다는 점입니다. FreeStory는 단일 생성 패스에서 timestep별 동적 character mask를 추정하고, reference-타겟 간 attention 유사도로 문자 수준 대응을 만든 뒤, correspondence-aware KV injection과 query blending을 적용해 동일 인물 영역에만 특징을 재사용하면서도 생성 다양성을 유지합니다.

- **Empirical Impact**: FreeStoryBench라는 벤치마크를 만들어 단일/멀티 캐릭터의 자유형 서사를 평가하며, 기존 ConsiStory+처럼 구조화된 프롬프트에서도 성능을 확인합니다. 실험 결과 FreeStory는 training-free 계열에서 최첨단 성능을 보였고, 자유형 세팅에서는 기준선 대비 캐릭터 정체성 보존이 전반적으로 더 좋아졌습니다. 특히 entity grounding을 자동으로 수행해도 oracle(정답 대응)과 유사한 수준으로 유지되어, 수작업 대응 없이도 현실적인 자유형 스토리텔링이 가능함을 보여줍니다.



### Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models (https://arxiv.org/abs/2606.25041)
Comments:
          Website: this https URL

- **Prior Approaches**: 기존 멀티모달 접근은 시각·오디오를 이해하는 모델과 비디오 생성/음성 합성 모듈을 따로 두거나, 중간 표현으로 text/ASR·TTS·VAD·아바타 렌더링을 조합하는 방식이 많았습니다. 이처럼 cascaded 파이프라인은 모듈 경계에서 대기 시간이 생기고, 인식·동기화 오차가 누적되며, 응답 타이밍·차례 관리·장기 일관성을 한 행동으로 학습하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: Wan-Streamer는 언어·오디오·비디오를 입력과 출력 모두에서 동일한 Transformer 한 개의 인과 스트림으로 묶어 end-to-end full-duplex 인터랙션을 지향합니다. 특히 외부 VAD/ASR/언어/TTS/오디오-드리븐 애니메이션/비디오 생성 모듈 없이, 지각·추론·생성·응답 타이밍·턴 테이킹·크로스모달 동기를 하나의 모델이 공동 학습하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 과제는 스트리밍 중에 각 모달리티가 서로 다른 토큰 레이트와 표현을 가지더라도, 같은 인과 과정 안에서 끊김 없이 정렬된 결과를 내는 것입니다. 이를 위해 strictly causal audio/video VAE, causal encoders/decoders, block-causal attention, 그리고 160 ms 단위(25 fps) 수준까지 가능한 저지연 multimodal token scheduling을 도입했으며, flow matching 기반으로 오디오·비디오 latents를 joint로 생성해 디코딩 전에 결합합니다.

- **Empirical Impact**: 실험에서 모델-사이드 신호 지연은 약 200 ms, 원격 양방향 네트워크까지 포함한 총 인터랙션 지연은 약 550 ms로 sub-second 듀플렉스 통신을 보여줍니다. 또한 말하지 않는 구간에도 시선·자세·미세표정 등 ‘보이는 청취’ 상태를 유지하고, 발화 중에는 사용자의 끊김/개입에 맞춰 스피치를 조절하는 등 자연스러운 듀플렉스 상호작용 품질을 제시했습니다.



### Chorus II: Cross-Request Sparsity Reuse for Efficient Image-to-Video Generation (https://arxiv.org/abs/2606.25040)
- **Prior Approaches**: 이미지-비디오(I2V) 생성에서 확산 모델은 고해상도와 긴 잠재 토큰 시퀀스 때문에 온라인 서빙 비용이 매우 크다. 이를 줄이기 위해 step-distillation, 동적 sparse attention(SVG2류), 정적/온라인 프로파일링 기반 sparse 패턴(Radial-Attn류), cross-request feature-cache(Chorus류) 등 다수가 제안됐지만, few-step(거의 1~수 스텝)에서는 마스크 예측/라우팅 오버헤드를 충분히 상쇄하기 어렵다는 한계가 있었다. 또한 latent feature를 그대로 재사용하면 source-induced content bias로 인해 기준선(dense baseline) 대비 의미/정체성 드리프트가 생길 수 있다.

- **Core Contribution**: 이 논문은 요청 간(request-to-request) 유사성이 I2V의 sparse attention 패턴에서도 강하게 나타난다는 관찰에 기반해, sparse mask를 “공유 재사용”하는 sparsity reuse를 제안한다. 핵심은 유사한 과거 요청에서 얻은 고품질 sparse mask를 현재 요청에 조건부 priors로 제공해, 요청마다 온라인 mask-prediction을 거의 없애면서도 dense baseline에 대한 충실도를 유지하는 것이다. 더 나아가 더 공격적인 속도 향상을 위해 선택적으로 downsampled latent feature reuse(하이브리드 맥락 재계산)와 guidance enhancement를 결합한다.

- **Technical Challenges**: 문제는 (1) 요청이 다르면 sparse 패턴도 약간 드리프트하므로 mask 재사용이 정보 흐름을 손상시킬 수 있고, (2) feature reuse는 경계 아티팩트(테어링/블러)나 조건 약화로 의미 드리프트를 유발할 수 있다는 점이다. 해결책으로 두 단계 safety fallback(visited map 기반 블록-쌍 visit refresh, min_top_k 기반 최소 보장)을 두어 정보 손실을 줄였고, downsampled latent feature reuse로 비재사용 영역의 전역 수용영역을 저비용으로 보존해 경계 아티팩트를 근본적으로 완화했다. 마지막으로 guidance enhancement(첫 프레임 key 강화, 이미지 임베딩 증폭 및 감쇠 스케줄)를 통해 reuse로 약해진 image/text conditioning을 낮은 오버헤드로 보정한다.

- **Empirical Impact**: Wan2.2-I2V의 4-step distilled 설정에서 sparsity reuse 기본 구성은 기본 품질을 유지하면서 fp16/quantized 커널 조합에서 2.16× 속도 향상을 보였다. 특히 온라인 high-fidelity routing 기반인 SVG2 수준의 dense baseline 충실도(PSNR/SSIM, LPIPS, CLIP-F)를 확보하면서도 SVG2의 지연을 대략 절반으로 줄였다는 점이 결과의 중심이다. turbo 설정(2.25×)과 선택적 feature reuse까지 더하면 2.59×까지 가속되며, quality–speed trade-off이 few-step 환경에서도 안정적으로 작동함을 실험적으로 입증했다.



### Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety (https://arxiv.org/abs/2606.25034)
- **Prior Approaches**: 기존 MLLM들은 전반적인 시각-언어 이해 능력을 잘 보이지만, 콘텐츠·AI 안전 영역에선 실사용의 적대적(adv.) 멀티모달 리스크를 안정적으로 찾아내기 어렵다. 특히 안전 정렬(alignment) 과정이 민감 지식의 탐색을 억제하면서, 왜 특정 시각 요소가 위반인지의 근거 있는 추론과 감사를 위한 설명이 빈약해진다는 한계가 지적된다.

- **Core Contribution**: Yuvion VL은 콘텐츠·AI 안전을 위한 대규모 멀티모달 LLM 계열로, instruction-tuned와 reasoning-oriented 변형을 제공하며 파이프라인 전체를 adversarial robustness 중심으로 설계한다. 안전을 “본질적으로 적대적이고 멀티모달인 문제”로 보고, 위험 개념의 크로스모달 정렬→안전 태스크 instruct 후학습→해석가능성 강화용 reasoning 후학습의 3단계 학습으로 구성한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작은 텍스트·워터마크·심볼 변형처럼 사람은 알아보지만 모델은 놓치는 미세 시각 단서를 정확히 구분하는 것과 (2) 동일/유사 외형이지만 정책상 의미가 다른 경우를 판별하는 대조 신호를 만드는 것이다. 이를 위해 Confuse-then-Contrast Fine-Tuning(C2FT)을 도입해 모델별 혼동을 동적으로 채굴하고, 다중 이미지 contrastive 그룹으로 시각-의미의 fine-grained 차이를 명시적으로 강제하며, 추가로 추론용 데이터는 posterior-constrained CoT 생성과 품질 검증(정답 정합·형식·사실성)으로 정제한다.

- **Empirical Impact**: 평가를 위한 Yuvion VL RiskEval(YVRE)는 오픈/내부 벤치마크를 3단계로 묶어 콘텐츠·AI 안전, 적대적 견고성, 실사용 역량을 폭넓게 측정한다. 실험에서 Yuvion VL-32B는 안전 관련 성능에서 동급 오픈소스 대비 평균 9.9점, 더 큰 상용 클로즈드 대비 평균 6.7점을 앞섰고, Yuvion VL-8B는 GPT-5.4와 Qwen3.5-Plus 같은 대형 모델들보다 여러 안전 태스크에서 우수한 결과를 보이며 일반 역량 저하는 제한적임을 시사한다.



### Noise-Aware Boundary-Enhanced Generative Learning for Ultrasound Speckle Reduction (https://arxiv.org/abs/2606.25009)
- **Prior Approaches**: 기존 초음파 speckle reduction(디스피클링) 방법은 필터 기반(bilateral, BM4D 등)과 딥러닝 기반으로 나뉜다. 필터 기반은 학습이 없어 빠르지만 고정된 규칙/가정 때문에 조직 경계를 과도하게 뭉개거나 이질적인 잡음 수준에 대한 적응이 약하다는 한계가 있다. 딥러닝 기반은 경계 보존을 위해 gradient/edge 제약을 넣기도 하지만, 잡음이 커지면 경계와 speckle의 고주파 성분이 구분되지 않아 구조 가이던스 신뢰도가 떨어지고, 잡음 레벨 적응이 있어도 경계 보존을 함께 강하게 결합하지 못하는 경우가 많다.

- **Core Contribution**: 이 논문은 Noise-Aware Boundary-Enhanced Generative Learning(NBGL) 프레임워크로, speckle 억제와 주석 기반 경계 보존을 하나의 3D 아키텍처에서 동시에 학습하도록 제안한다. Speckle reduction 가지는 생성학습으로 잡음을 줄이고, boundary enhancement 가지는 목표 해부학적 경계를 명시적으로 보강해 과스무딩을 완화한다. 또한 추정한 speckle noise level을 사용해 두 가지의 feature coupling 강도를 적응적으로 조절하는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 과제는 (1) speckle과 해부학적 경계가 모두 고주파로 관측될 때, 잡음 레벨을 경계에 덜 편향되게 추정하고 (2) 잡음이 서로 다른 입력에서도 경계-보존과 잡음-억제를 동시에 안정화하는 것이다. 이를 위해 NIWG는 3D Laplacian의 고주파 응답에 median absolute deviation(MAD) 기반 추정기를 적용해 speckle noise level을 robust하게 계산한 뒤, bounded power-law로 interaction weight를 만든다. 이후 wFiLM(Weighted Feature-wise Linear Modulation)이 이 가중치를 반영해 교차 가지(bidirectional) feature modulation을 스테이지별로 조절하며, 특히 더 깊은 상호작용에서는 speckle reduction 쪽 정보가 경계 스트림으로 과도하게 번지는 위험을 낮추는 비대칭 감쇠를 둔다.

- **Empirical Impact**: UterUS 데이터셋의 3D transvaginal 초음파 141개 볼륨에 대해, NBGL은 6개 잡음 레벨 전반에서 speckle reduction과 구조 보존, 그리고 주석 경계 일치 측면에서 최신 방법들을 일관되게 능가했다고 보고한다. 또한 추정된 잡음 레벨에 따라 경계-관련 feature coupling이 달라지는 설계가 이질적 잡음 상황에서도 성능을 유지하는 데 기여했음을 시사한다. 결과적으로 NBGL은 “잡음 적응 + 임상적으로 의미 있는 경계 보존”을 동시에 달성하는 프레임워크로서 디스피클링 연구의 실용성을 높인다는 의미가 있다.



### Curvature-Guided Mixing for MLLM Adaptation (https://arxiv.org/abs/2606.24963)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 접근은 (1) 정규화로 파라미터 변화를 제한하거나, (2) LoRA/Adapter 같은 PEFT로 대부분을 고정해 간섭을 줄이거나, (3) 일부 파라미터만 섞는 model merging을 시도합니다. 특히 merging 계열은 Spider처럼 휴리스틱 스코어에 의존하거나, Hessian 기반이라고 해도 다운스트림 단일 목적만 최적화해 pre-trained 일반 지식 보존이 약하다는 한계가 있습니다. 그 결과 fine-tuning 성능은 오르지만 기존 능력이 무너지는 catastrophic forgetting 문제가 남습니다.

- **Core Contribution**: 이 논문은 CurvatureGuided Mixing (CGM)을 제안하며, pre-trained 손실과 fine-tuning 손실을 동시에 낮추는 joint optimization을 세웁니다. Hessian의 지역적 곡률을 활용해 pre-trained과 fine-tuned 파라미터를 “soft mixing”으로 닫힌형(closed-form) 비율로 결합하고, 곡률이 큰 방향은 더 보존/반영되도록 설계합니다. 또한 CGM†로 dense 혼합의 위험을 줄이기 위해, 곡률-aware score로 중요 파라미터만 되돌리는 “hard mixing”을 도입합니다.

- **Technical Challenges**: 핵심 난제는 fine-tuning으로 생긴 전문성은 유지하되 pre-trained의 일반 지식을 지키도록 파라미터를 어디까지/어떻게 섞을지 이론적으로 결정하는 것입니다. CGM은 두 손실 지형을 2차 테일러 근사하고 Hessian의 대각 성분만 사용해 계산을 줄인 뒤, 파라미터별로 최적 mixing ratio를 유도합니다. CGM†는 soft가 모든 파라미터를 건드리는 문제를 sparse 선택(Top-K%)으로 바꾸고, (fine-tuning 곡률 vs pre-training 곡률) 상대 크기와 reversion vector를 결합한 점수로 되돌릴 파라미터를 랭킹합니다.

- **Empirical Impact**: LLaVA-1.5-7B와 Qwen2.5VL-3B에서 OKVQA/Flickr30k 및 Flickr30k/LaTeX-OCR 등을 fine-tuning한 뒤, 여러 일반 벤치마크를 통해 specialization-일반화 trade-off를 Hscore/Avg로 측정했습니다. 실험 전반에서 CGM과 CGM†가 기존 merging/휴리스틱 및 fine-tuning 대비 catastrophic forgetting을 더 잘 억제하면서도 다운스트림 성능을 유지해 종합 지표가 가장 좋게 나타났습니다. 특히 ablation에서 곡률을 jointly 고려한 최종 점수식이 단순 magnitude나 단일 곡률만 쓰는 경우보다 Hscore가 일관되게 우수했으며, 추가 비용은 diagonal Hessian(실증적 FIM) 추정 정도로 약 7.9% 오버헤드에 그쳤습니다.



### SEMIR: Topology-Preserving Graph Minors for Thin-Structure Segmentation (https://arxiv.org/abs/2606.24935)
Comments:
          Accepted to the European Conference on Computer Vision (ECCV) 2026

- **Prior Approaches**: 기존 박스는 픽셀 그리드나 SLIC 같은 task-agnostic superpixel 분해를 그대로 써서 박판(1~3픽셀) 구조를 다룬다. 이 과정에서 선/균열/차선 연결성이 먼저 깨지거나, 얇은 타깃이 배경과 합쳐져 분류 단계 이전에 이미 단절이 발생한다. 또 clDice, Skeleton Recall Loss, Topograph 같은 topology-aware loss는 연결성의 단절을 학습 목표에서 벌점으로 다루지만, 표현(표현된 그래프/분할)이 이미 망가진 연결성을 되살리긴 어렵다.

- **Core Contribution**: SEMIR은 픽셀 격자를 “그래프 마이너(minor)”로 치환해, 그래프 수축(contraction) 조건을 만족하는 경우 얇은 구조의 연결성을 이론적으로 보존하도록 설계했다. edge contraction, node deletion, edge deletion로 픽셀을 경계 정렬(supernode) 단위로 압축한 뒤, GNN이 축소 그래프에서 노드 단위 분류를 수행하고 exact lifting으로 다시 픽셀 해상도 예측을 복원한다. 이때 핵심은 학습 로스가 아니라 표현 자체가 연결성 제약을 갖는다는 점이다.

- **Technical Challenges**: 도전 과제는 (1) 수축으로 수천~수백만 픽셀을 줄이면서도 연결성 보존 조건을 안정적으로 만족시키는 것과 (2) 학습 없이/학습과 분리된 방식으로 경계 정렬 파라미터를 정하는 것이다. SEMIR은 픽셀 방문 순서 편향을 줄이기 위해 gcd(s,N)=1을 만족하는 스텝 기반 스크램블 트래버설을 쓰고, 몇-shot 데이터(5~20장)로 경계 정렬을 위한 minor 파라미터를 SMBO(ExtraTrees surrogate)로 블랙박스 최적화한다. 또한 GINE 기반의 경량 GNN과, 수축이 만든 partition을 그대로 읽는 exact lifting을 결합해 “해상도 복원에서의 보간 손실”을 피한다.

- **Empirical Impact**: TTPLA(전력선), CrackSeg9k(균열), SkyScapes Lane(항공 차선) 3개 도메인을 단일 파이프라인으로 평가해 Dice/IoU/BF1에서 도메인별 베이스라인을 맞추거나 능가했으며, 동일 파이프라인 기준으로도 우수한 연결성 보존을 확인했다. 특히 SLIC로 마이너를 대체한 제어 실험에서, SEMIR의 mask fragmentation(연결 성분 분절)이 최소 4.6배 이상 감소해 표현 수준의 토폴로지 이점이 분명해졌다. 21MP급 고해상도까지 “패치 없이” 풀해상도 추론 가능성을 제시하며, 얇은 구조 분할에서 표현 설계가 성능을 좌우한다는 메시지를 강화했다.



### Learning Action Priors for Cross-embodiment Robot Manipulation (https://arxiv.org/abs/2606.26095)
- **Prior Approaches**: 대부분의 Vision-Language-Action(VLA) 모델은 Vision-Language Model(VLM) 백본에 행동(action) 모듈을 붙이고 정책 전체를 imitation learning으로 end-to-end로 학습한다. 이때 VLM은 시각·언어의 강한 선행지식을 가져오지만, 행동 모듈은 물리적 모션의 시간 동역학을 거의 zero에서 배워야 하는 불균형이 생긴다. 특히 cross-embodiment에서는 로봇마다 action space와 모션 분포가 달라 초기 학습이 불안정해지고 수렴이 느려진다.

- **Core Contribution**: 이 논문은 cross-modal VLA 정렬 전에 행동 모듈에 motion prior를 먼저 학습시키는 2-stage 프레임워크를 제안한다. Stage 1에서 시각·언어 없이 행동 궤적만으로 행동 모듈이 시간적 모션 구조를 학습하고, Stage 2에서는 이를 VLA 학습에 전이해 시각·언어 정렬을 더 안정적으로 시작하게 만든다. 또한 학습된 encoder를 history compressor로 재사용해 긴 시간 문맥을 적은 비용으로 토큰 1개로 요약한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘행동 모듈이 아직 정렬되지 않은 상태’에서 VLA가 동시에 action 분포 학습과 cross-modal alignment를 수행하며 생기는 그라디언트 불안정과 최적화 병목을 줄이는 것이다. 저자들은 Stage 1에서 flow matching 기반 encoder-decoder 행동 모듈을 action-only로 학습해 모션의 구조적 표현을 만든 뒤, Stage 2에서는 decoder 재사용과 early-stage latent distillation으로 VLM 예측 임베딩을 그 모션 잠재공간에 초기에 고정한다. 이후 distillation 제약을 단계적으로 완화해 최종 성능은 end-to-end 정련으로 끌어올린다.

- **Empirical Impact**: LIBERO와 RoboCasa 시뮬레이션 및 실세계 Franka에서 총 13개 cross-embodiment 태스크를 실험해, action prior 없는 VLA 대비 더 빠른 수렴과 더 높은 성공률을 보였다고 보고한다. 특히 데이터가 적은 long-tail 실세계 태스크에서 성능 향상이 두드러지며, history 압축 토큰은 장기 과제에서 시간 수용범위를 늘려 추가 이점을 준다. 또한 Stage 1에서 action-only 데이터를 늘릴수록 더 일반화되는 motion prior가 형성되어 downstream VLA 성능이 직접 개선되는 scaling 경향도 관찰된다.



### Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models (https://arxiv.org/abs/2606.26079)
Comments:
          22 pages, 4 figures, 5 tables

- **Prior Approaches**: 기존 MLLM 벤치마크는 각 항목을 한 가지 canonical ordering으로만 채점해, 의미적으로 같은 입력 순열(shuffling)에서 정답이 바뀌는지(순서-불변성) 신뢰도를 놓치곤 했습니다. 관련 연구는 텍스트에서 옵션 순서나 프롬프트 형식 등 단일 편향을 다뤘지만, 멀티모달에서 여러 ordering facet을 분리해 교차-순서 불안정성을 분해·정량화한 평가는 거의 없었습니다.

- **Core Contribution**: 이 논문은 Facet-Probe라는 5개 facet(옵션, evidence-chunk, document-rank, image-set, mixed-modality order)으로 18개(프런티어/오픈웨이트) MLLM을 12개 데이터셋에서 감사해, 순서 perturbation에 따른 답변 뒤집힘을 체계적으로 측정합니다. 또한 ODI(Ordering-Decomposed Item-Response Theory)와 same-ordering control로, ordering noise(무작위성)와 facet별 systematic bias(체계적 편향)를 분리하고 same-input decoder-stochastic 플립 바닥선도 추정합니다.

- **Technical Challenges**: 핵심 난제는 “플립이 발생한 것”만으로는 원인이 ordering 때문인지, 디코더의 확률적 잡음 때문인지 구분이 어렵다는 점이었습니다. 논문은 각 (facet, dataset, item) 셀에서 여러 순열을 샘플링해 flip-rate를 측정하고, same-ordering 통제로 decoder-noise floor를 벗겨 ordering excess를 추정하는 Bayesian 2PL 계층 모형으로 이를 해결합니다. 아울러 mixed-modality는 LLM judge로 정답 일치도를 정의해 불완전한 표면형 답변 변동을 보정하려고 했습니다.

- **Empirical Impact**: 결과적으로 18개 모델 모두 order-invariant가 아니며, facet별 패널 평균 flip rate가 24–50% 범위로 관측됩니다(예: mixed-modality-order 0.50, evidence-chunk-order 0.41, option-order 0.36, document-rank-order 0.26, screened image-set-order 0.24). Gemini의 same-ordering control(temperature 0)에서는 decoder-stochastic 바닥선 대비 ordering excess가 상당하며, 능력(capability)이 플립을 줄이긴 해도 완전히 제거하지는 못해 최선 모델도 13.4%의 trial에서 뒤집힙니다. 학습 없이 prompt 변경만으로는 modality-conditional 하면서도 텍스트→시각 추론으로 전이되지 않아, 주문 수준 완화만으로 일반적인 순서 강건성을 기대하기 어렵다는 시사점을 제공합니다.



### RoboAtlas: Contextual Active SLAM (https://arxiv.org/abs/2606.26046)
Comments:
          Alexander Schperberg and Shivam K. Panda made equal contribution

- **Prior Approaches**: 기존 Active SLAM은 주로 occupancy grid 같은 저해상도 기하 표현과 휴리스틱 기반 정보이득/제약 최적화에 의존해 왔습니다. 그 결과 semantic search나 특정 객체/장소 중심 탐색처럼 문맥 추론이 필요한 작업에서는, 기하 정보만으로는 한계가 있어 결정이 자주 비효율적이었습니다. 한편 foundation model(LLM/VLM)을 zero-shot으로 결합하는 연구도 있지만, 탐색 단계에 따라 필요한 탐색-활용 균형이 달라져 고정된 방식은 성능이 흔들릴 수 있습니다.

- **Core Contribution**: RoboAtlas는 geometric exploration과 semantic reasoning을 상황(장면 이해 정도)에 맞춰 전환하는 “contextual Active SLAM” 프레임워크를 제안합니다. 3D 인스턴스 단위 시맨틱 매핑 시스템 OpenRoboVox와, frontier 탐색·전역 시맨틱 맵 추론·일인칭 VLM 추론을 각각 전문가(expert)로 구성한 뒤 contextual multi-armed bandit으로 다음 목표를 선택합니다. 이렇게 해서 초기에는 빠르게 지도를 채우고, 정보가 쌓이면 언어/의미 기반의 목표 지향 탐색으로 자연스럽게 이동합니다.

- **Technical Challenges**: 핵심 난제는 (1) 로봇에서 실시간으로 확장 가능한 3D 인스턴스 시맨틱 매핑을 유지하면서 (2) 탐색 단계별로 전문가 선택을 안정적으로 적응시키는 것입니다. OpenRoboVox는 memory-efficient TSDF 최적화, 인스턴스 pruning, distance-prioritized historical refresh, 비동기 병렬 처리를 통해 edge/실로봇 환경에서도 대규모 시맨틱 인스턴스를 다루도록 설계했습니다. 또한 Scene-Dictionary로 복잡한 voxel을 압축해 LLM/VLM 추론에 적합한 문맥 세계모델을 만들고, bandit은 점진적 문맥 특징(커버리지 변화, backtracking, VLM 신뢰, 장면-질의 관련도)을 바탕으로 exploration과 semantically guided navigation을 교체합니다.

- **Empirical Impact**: 실험은 시뮬레이션과 Unitree Go2 실로봇에서 1800m2 이상 대규모 환경(약 30k개의 mapped semantic instances)을 대상으로 검증됐으며, 작업 성공률 100%를 달성했습니다. GOAT-Bench의 “Val Unseen”에서 GPT-4o를 사용한 RoboAtlas는 SR 90.6%로 보고된 최고 성능이며, 최강 prior baseline 대비 SR을 17.8%p 끌어올렸습니다. 더 작은 Qwen2.5-VL-7B(7B)에서도 SR 88.8%를 유지해, 단순히 foundation model을 바꾸는 것보다 3D 시맨틱 맵으로 grounding한 문맥이 성능 향상에 결정적임을 보여줍니다.



### FedReLa: Imbalanced Federated Learning via Re-Labeling (https://arxiv.org/abs/2606.26037)
- **Prior Approaches**: 연합학습에서 클래스 불균형은 클라이언트 간 데이터 분포 차이(heterogeneity)와 함께 발생하며, 이때 로컬 불균형을 완화해도 글로벌 minority/tail 클래스 성능이 쉽게 저하된다. 기존 연구는 주로 집계/학습 안정화나 손실 재가중 같은 algorithm-level 접근에 치중했고, global class distribution을 보거나 추가 데이터/파라미터를 요구하는 데이터-level 방법은 프라이버시 제약으로 발전이 제한적이었다. 특히 SMOTE·mixup 계열 데이터 증강은 글로벌 클래스 prior을 알아야 하거나 로컬에 minority가 거의/전혀 없으면 합성이 어려워 적용이 까다롭다.

- **Core Contribution**: FedReLa는 연합학습에서 data heterogeneity와 global class imbalance가 공존하는 상황을 겨냥한 label-space 기반 데이터-level 방법이다. 핵심은 global class prior를 몰라도, 글로벌 모델에 내재된 minority 클래스의 feature 정보를 활용해 feature-dependent label re-allocator로 로컬 샘플의 라벨을 재할당함으로써 편향된 글로벌 결정경계를 교정하는 것이다. 또한 새로운 학습 파라미터나 추가 통신 없이(plug-in 방식) 기존 algorithm-level 방법과 함께 일관된 성능 개선을 제공하는 데 초점을 둔다.

- **Technical Challenges**: 가장 큰 어려움은 글로벌 불균형 정보를 직접 관측할 수 없고, 비동질성 때문에 클라이언트별 로컬 결정경계가 서로 어긋나며, 극단적 조건에서는 로컬 minority 샘플 자체가 부족해 증강 기반 correction이 막힌다는 점이다. FedReLa는 majority→minority 방향의 비대칭 선택적 relabeling만 수행하고, relabeling 확률을 해당 샘플이 minority feature 공간을 얼마나 침범(intrude)하는지에 비례하도록 설계해 결정경계를 로컬에서 먼저, 이어서 집계 관점에서도 글로벌 불균형 비율을 완화하도록 유도한다. 더 나아가 전제한 re-labeling이 통신/학습 오버헤드를 늘리지 않도록 글로벌 모델을 label re-allocator로 재활용하는 구조를 제안한다.

- **Empirical Impact**: 논문은 Fashion-MNIST, CIFAR-10/100, ImageNet에서 stepwise-imbalanced와 long-tailed 설정 및 다양한 heterogeneity 수준으로 FedReLa를 검증했으며, minority 클래스 정확도와 전체 정확도 모두에서 기존 state of the art를 능가한다고 보고한다. 특히 극단적인 조건에서 minority/tail 클래스 정확도가 step-wise에서 최대 38.30%, long-tailed에서 최대 30.17%까지 향상되면서도 overall accuracy 우위가 유지된다. 이는 연합학습에서 global class prior 없이도 label-space 교정만으로 tail 성능을 끌어올릴 수 있음을 보여주며, data-level 프라이버시 제약 하의 실용적 방향성을 제시한다.



### In-Context World Modeling for Robotic Contro (https://arxiv.org/abs/2606.26025)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 현재 관측과 언어 지시에만 조건을 거는 경우가 많아, 카메라 시점이나 로봇 형태처럼 실행 환경이 바뀌면 잘 일반화하지 못한다. 특히 학습 중에 마주친 고정된 실행 맥락을 전제로 동작하는 경향이 있어, 새로운 환경에서는 data-intensive fine-tuning 없이는 성능을 유지하기 어렵다.

- **Core Contribution**: 이 논문은 In-Context World Modeling(ICWM)으로, 시스템 식별을 in-context adaptation 문제로 취급해 로봇이 짧은 자기 상호작용 히스토리만으로 필수 시스템 변수를 추론하도록 만든다. ICWM은 demonstrations로 ‘무엇을 할지’를 정하는 전통적 in-context learning과 달리, context window로 ‘시스템이 어떻게 동작하는지’를 파악해 태스크 실행 전부터 현재 월드 다이내믹을 내재적으로 모델링한다.

- **Technical Challenges**: 핵심 기술 과제는 파라미터 업데이트 없이, 태스크와 무관한 상호작용 데이터만으로도 새로운 구성요소(예: 시점, 형태)의 시스템 변수를 안정적으로 식별해 정책 추론에 반영하는 것이다. 논문은 task 실행 전에 self-generated, task-agnostic interaction을 처리해 월드 다이내믹을 암묵적으로 학습시키는 방식으로 이 문제를 해결하며, 그 결과 adaptation을 모델 내부의 context 처리로 수행한다.

- **Empirical Impact**: ICWM은 시뮬레이션과 실제 로봇 플랫폼 모두에서 광범위하게 평가되었고, novel camera viewpoints 같은 미지 구성에서 표준 VLA baseline 대비 유의미하게 더 높은 성능을 보였다. 파라미터 업데이트 없이도 새로운 실행 맥락에 적응할 수 있다는 점에서, VLA의 일반화 한계를 줄이고 현장 적용 비용을 낮추는 방향으로 의미 있는 진전을 제시한다.



### Tensorion: A Tensor-Aware Generalization of the Muon Optimizer (https://arxiv.org/abs/2606.25975)
- **Prior Approaches**: Adam 같은 1차 최적화는 파라미터를 평탄한 벡터로 보고 학습 과정에서 다중선형(텐서) 구조를 충분히 반영하지 못한다. Muon은 스펙트럴 노름 제약 하에서의 steepest descent로 행렬 구조를 활용해 좋은 최적화 동역학을 보였지만, 텐서로 그대로 확장하기엔 텐서 스펙트럴 노름의 계산이 일반적으로 NP-hard다. 기존 텐서 인식 접근은 주로 프리컨디셔닝(예: Shampoo, K-FAC 계열)이나 파라미터 재구성에 초점이 있어, 텐서 구조를 “원래 파라미터를 유지한 채” LMO 단계에서 직접 다루는 방식은 상대적으로 제한적이었다.

- **Core Contribution**: Tensorion은 Muon의 “제약 최적화 관점”을 고차 텐서 파라미터까지 확장한다. 텐서 스펙트럴 노름 볼(ball) 위의 LMO를 그대로 풀기 어렵다는 점을 해결하기 위해, 텐서 스펙트럴 노름을 타이트하게 상계하면서도 LMO가 계산 가능해지는 텐서 노름(완화)을 듀얼 관점에서 설계한다. 또한 이 완화가 실제로는 적응적으로 선택된 unfolding 행렬들의 연산으로 귀결되며, 2차 텐서(행렬)로 제한하면 Muon을 정확히 재현한다.

- **Technical Challenges**: 핵심 난제는 텐서 스펙트럴 노름의 자연스러운 일반화가 계산 불가능해 LMO가 실용적으로 풀리지 않는다는 점이다. 이를 위해 Tensorion은 (여러 unfolding의 primal spectral norm을 단순 결합하는 방식이 지나치게 보수적이거나 Frobenius 노름으로 붕괴하는 문제를 피하면서) 듀얼 측에서 unfolding의 nuclear norm으로 텐서 노름을 구성해, LMO 해가 “최대 nuclear norm을 갖는 unfolding”을 찾는 문제로 단순화되게 만든다. 추가로 최적 unfolding을 매 iteration마다 재탐색하지 않도록, 랜덤 텐서에 대한 이론과 unfolding 크기(행/열 차원 균형) 직관을 바탕으로 offline 휴리스틱 τopt를 제안해 계산 비용을 낮춘다.

- **Empirical Impact**: CNN과 Transformer 기반 CV 분류 과제에서 Tensorion은 SGD/Adam 및 기존 텐서 인식 베이스라인 대비 수렴이 더 빠르거나 그래디언트 업데이트가 더 안정적인 경향을 보였다. 특히 unfolding 선택(τ 집합)이 최적화 동역학과 성능에 큰 영향을 주며, Tensorion의 offline unfolding 선택 전략이 online(매 iteration 전 후보 평가) 대비 비용 대비 효율적으로 작동함을 ablation에서 확인한다. 결론적으로 Tensorion은 텐서 구조를 LMO 내부에서 직접 반영하는 “tensor-aware constrained steepest descent” 설계를 실험적으로 검증한 사례로, 텐서형 파라미터를 쓰는 차세대 학습 알고리즘 개발에 의미 있는 방향을 제시한다.



### DSP-SLAM++: A Unified Framework for Multi-Class, High-Fidelity Object SLAM in the Wild (https://arxiv.org/abs/2606.25953)
Comments:
          9 pages, 9 figures

- **Prior Approaches**: 기존 object-aware SLAM은 객체를 박스(cuboids)나 쿼드릭(quadric) 같은 단순 프리미티브로 표현해 속도와 다중 클래스 확장을 확보하지만, 실제 형상 정밀도가 떨어져 근접 주행·계획에 제약이 컸다. 반대로 DSP-SLAM류의 deep implicit shape prior는 고품질 메시에 강점이 있으나 연산 비용이 높고 단일 클래스/제한된 센서 구성에 머무르는 경향이 있었다. 또한 3D Gaussian Splatting 계열은 렌더링 품질은 높여도 객체 인스턴스 단위의 일관된 의미론 파악이 부족하다는 평가가 뒤따랐다.

- **Core Contribution**: DSP-SLAM++는 DSP-SLAM의 implicit 표현력을 유지하면서, 실시간 multi-class를 가능하게 하는 구조를 제안한다. 핵심은 비동기(asynchronous) 매핑 파이프라인으로 객체 재구성을 추적 스레드와 분리해 지연 병목을 줄이고, monocular fisheye-LiDAR에 맞춘 센서 융합·보정 모듈로 실차 환경 적용성을 높인 점이다. 그 결과 고정밀·기하 완전한(geometrically-complete) 객체 메시는 유지하면서도 25 Hz 수준의 견고한 실시간 성능을 노린다.

- **Technical Challenges**: 기여를 실현하는 첫 관문은 multi-class에서 2D 마스크와 3D 바운딩 박스의 잘못된 매칭(association ambiguity)을 줄이면서도 계산량을 통제하는 것이다. 논문은 클래스 일관성 기반 projection과 global greedy bipartite matching으로 2D-3D 인스턴스 매칭을 한 번에 정리하고, 클래스별 바닥면 스케일을 반영한 class-conditioned association으로 프레임 간 추적 안정성을 보강한다. 두 번째 관문은 fisheye의 심한 왜곡 때문에 LiDAR 깊이를 2D instance mask에 정확히 결합하기 어려운 문제이며, 전체 픽셀 rectification 대신 mask contour만 rectification하는 경량 전략과 ORB-SLAM3의 native fisheye 지원을 조합해 정합성을 확보한다. 마지막으로 implicit 객체 최적화의 시간 비용을 해결하기 위해 객체를 바운딩 박스 placeholder로 즉시 삽입하고, DeepSDF 최적화/mesh extraction을 back-end 워커 스레드로 넘기는 비동기 재구성(AR)을 도입한다.

- **Empirical Impact**: 실험은 KITTI, nuScenes, CBNU 및 논문 자체 in-house 25 Hz 다중 클래스 데이터셋에서 수행되었고, DSP-SLAM++는 tracking 정확도와 scale 일관성에서 기준 대비 경쟁력 있는 수준을 보여준다. 특히 custom multi-class 시나리오에서 SR(동기) 방식의 매핑 지연 피크가 >500 ms로 커지는 반면, AR 비동기 파이프라인은 평균 매핑 지연을 1/3로 낮추고 최대 Keyframe BA latency를 3 프레임 수준으로 축소해 실시간성 붕괴를 방지했다. 또한 객체 복원 지연(thread bottleneck)을 최대 70%까지 줄이면서도 다중 클래스의 고품질 객체 메시는 유지되어, 자율주행과 로봇 조작 같은 실세계 task로의 확장성을 강화했다. 단, 보행자처럼 얇거나 가려짐이 심한 클래스는 입력 포인트 클라우드 밀도와 사전학습 품질에 의존해 세부 형상이 떨어질 수 있으며, 이는 향후 개선 여지로 제시됐다.



### Color Matters: Trigger Color Affects Success in Federated Backdoor Attacks (https://arxiv.org/abs/2606.25858)
Comments:
          Accepted at the IEEE/IFIP DSN Workshop on Dependable and Secure Machine Learning (DSML), 2026

- **Prior Approaches**: 기존 연합학습 백도어 연구는 corner patch나 sticker 같은 고정된 합성 트리거에 의존하는 경우가 많아, 트리거 외형 변화(특히 색)가 공격 성패에 미치는 영향을 충분히 분리해 보지 못했다. 방어 쪽도 Krum, Trimmed Mean, FLTrust, FLAME, MultiKrum처럼 집계/필터링 중심으로 다루면서 트리거의 시각적 속성은 거의 고정값으로 취급해왔다. 그 결과, 동일한 의미의 트리거라도 외형이 달라지면 방어 효과가 달라질 수 있다는 불확실성이 남아 있었다.

- **Core Contribution**: 이 논문은 semantics-driven 백도어에서 트리거 객체(mask, sunglasses)는 유지하되 트리거 색만 바꿔, 공격 성공률(ASR)에 대한 ‘색의 역할’을 통제 실험으로 분리해 제시한다. CelebA의 헤어 컬러 4클래스 설정에서 백도어 파이프라인, 배치/예산, 타깃/소스 매핑을 고정하고 흑/백 트리거 변형만 비교해 색-타깃 정렬(예: 흰색은 blond 타깃)에 따른 성능 변화를 정량화한다. 또한 표준 poisoning 목표뿐 아니라 SABLE 기반 방어 인지 목표와 robust aggregation 하에서도 같은 경향이 유지되는지 확인한다.

- **Technical Challenges**: 트리거 색만 바꾸면 관찰 차이가 ‘시각적 단서 변화’에서 오지만, 연합학습에서는 비IID 분포와 집계가 로컬 학습 신호의 일관성과 보존 여부를 함께 흔든다. 이 때문에 단순히 색을 바꾼 트리거를 적용하는 것만으로는 원인 규명이 어렵고, SABLE처럼 clean 분류 손실, triggered 타깃 손실, penultimate 표현공간 분리 loss, 악성 업데이트 드리프트를 제약하는 정규화까지 포함해 색 효과가 방어/집계 하에서 지속되는 조건을 설계해야 한다. 저자들은 동일 소스 이미지에 대해 image ID로 트리거 변형을 매칭해 비교 가능성을 높였고, 방어 실험에서는 MultiKrum 집계를 사용해 색 효과가 필터링을 통과하는지까지 검증했다.

- **Empirical Impact**: 결과적으로 트리거 색 변화만으로 ASR이 크게 달라졌고, clean accuracy는 비교적 안정적으로 유지되는 패턴이 관찰됐다(예: blond 타깃에서는 white 트리거가 더 높게 나타남). 또한 SABLE+MultiKrum처럼 더 강한 설정에서도 ‘타깃 컬러와 시각적으로 더 가까운 트리거 색이 더 효과적’이라는 방향성이 유지되어, 색이 단순한 외형 변수가 아니라 학습 가능성과 집계 생존에 의미 있게 관여함을 시사한다. 즉, 방어 평가를 단일 트리거 색에만 의존하면 위험을 과소평가할 수 있으며, 의미 기반(semantic) 백도어 벤치마킹에서 색을 first-class 변수로 다뤄야 한다는 실무적 함의를 제공한다.



### Hybrid deep learning-based phase diversity method for wavefront reconstruction (https://arxiv.org/abs/2606.25855)
Comments:
          13 pages, 10 figures. The following article has been submitted to Review of Scientific Instruments. After it is published, it will be found at this https URL

- **Prior Approaches**: 고출력 레이저에서 적응광학(AOS)은 비공통 경로 수차(NCPA) 때문에 초점면 첨두 강도를 저하시킨다. 기존 보정은 반복 기반 위상복원/센서리스 최적화/강화학습 등으로 해결해왔지만, 반복 최적화는 수렴이 느리거나 국소해에 민감하고 초기조건 의존성이 크다는 한계가 있다. 딥러닝 기반 재구성은 추론 속도는 빠르지만 고차 수차에서 정확도가 부족하거나 학습 분포 밖으로 일반화가 약하다는 문제가 지적돼 왔다.

- **Core Contribution**: 이 논문은 비공통 경로 수차를 빠르고 정확하게 재구성해 AOS를 캘리브레이션하는 하이브리드 방식을 제안한다. CNN이 포커스/디포커스(두 평면) 강도 분포로부터 수차의 초기 추정을 만들고, 이후 L-BFGS로 Zernike 계수를 정밀 보정한다. CNN과 최적화 단계의 입력을 분리해(학습은 2평면, 보정은 3평면) 실험 셋업 변화에 대한 재학습 부담을 줄이도록 설계했다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 반복 최적화의 초기조건 민감도를 낮추면서 (2) 잡음·블러·양자화 등 실험 오차 하에서도 수차를 안정적으로 복원하는 것이다. 이를 위해 CNN은 입력에 제곱근 변환을 적용하고, 공간 이동·가우시안 블러·양자화·추가 잡음을 포함한 데이터 증강으로 강건성을 확보한다. 또한 약한 수차에 대해 2평면만 쓰면 체계적 오차가 커지는 문제를 막기 위해, L-BFGS 목적함수는 +d/−d 디포커스를 포함한 3장 이미지로 정규화해 최적화를 더 견고하게 만든다.

- **Empirical Impact**: 수치 시뮬레이션에서는 RMS 수차가 0~1.3λ 범위일 때 약 80% 사례에서 효율이 ~0.99까지 도달했다. 실험에서는 RMS 0.15~0.6λ에서 효율이 ~0.75였고, 2~4회 반복 내에 Strehl ratio 0.96±0.02를 달성해 실제 조건에서도 빠른 캘리브레이션 가능성을 확인했다. 결과적으로 고전 반복법의 느림/초기 민감도와 딥러닝의 정확도 한계를 동시에 완화하는 접근으로, 고출력 레이저 AOS 보정의 실용성을 높일 것으로 기대된다.



### Re-mixing Embeddings for Patient Augmentation in Data Scarce Multiple Instance Learning (https://arxiv.org/abs/2606.25770)
Comments:
          Accepted for publication at the 29th International Conference on Medical Image Computing and Computer Assisted Intervention - MICCAI 2026

- **Prior Approaches**: 기존 의료 MIL의 증강은 주로 이미지의 회전·플립·색상 섭동 같은 instance-level 기법이었지만, 잡음과 중복을 키우거나 MIL 특성상 계산 비용이 커지는 한계가 있었다. bag-level 혼합인 PseMix·ReMix는 온라인으로 합성 환자를 만들고(재사용 어려움), 국소 혼합이라 전체 모집단 통계도 잘 반영하지 못한다. 무엇보다 특정 클래스의 실제 예시가 모두 필요해 healthy control이 통째로 사라진 rare disease 상황엔 적용이 어렵다.

- **Core Contribution**: 이 논문은 RECIPE( Re-mixing Embeddings via Clustering for Patient augmEntation )로, 환자 증강을 “환자 레벨”에서 오프라인으로 생성하는 통계 기반 방법을 제안한다. GMM으로 모든 환자에서 뽑은 instance embedding을 확률적 클러스터링하고, 질병 클래스별로 인스턴스 분포의 통계 레시피(unsupervised cluster recipe)를 학습한 뒤 그 레시피에 따라 새로운 임베딩으로 환자(bag)를 생성한다. 또한 생성 환자는 불확실성 정량화로 선별해 MIL 성능을 추가로 끌어올린다.

- **Technical Challenges**: 핵심 난제는 (1) instance를 섞되 bag의 질병 표현을 “현실적으로” 유지하고, (2) 클래스가 통째로 누락된 경우에도 외부 통계로 생성이 가능해야 한다는 점이다. RECIPE는 클러스터별로 클래스-인스턴스 개수 분포를 평균·분산 형태로 학습하고, 레시피에 따라 pooled embeddings에서 샘플링해 재혼합함으로써 bag 구조를 보존한다. 더 나아가 MC dropout 기반 예측 엔트로피(또는 BALD/Max-STD)로 생성 환자를 우선순위화해 학습에 유익한 샘플만 고르도록 설계했다.

- **Empirical Impact**: 실험은 (i) healthy 클래스가 아예 없는 missing-class 시나리오에서 외부 코호트 통계를 이용한 생성, (ii) 소수 데이터(low-data)에서의 생성, (iii) scRNA-seq·flow cytometry 같은 소규모 비영상 태스크까지 총 3가지 희소성 환경에서 수행됐다. missing-class에서는 전체 데이터 학습과 유사한 수준의 성능(예: 0.78 대비 0.83)을 달성했고, few-shot에서도 baseline과 mix 기반 다른 방법을 넘어서는 향상이 확인됐다. 비영상 영역에서도 balanced accuracy가 scVI/PCA 임베딩과 flow cytometry 패널 전반에서 의미 있게 개선됐으며, 불확실성 측정 중 predictive entropy가 가장 효과적이었고(10% 이상 우위), 증강 성능은 ABMIL뿐 아니라 DSMIL·Transformer까지 아키텍처 전반에서 확장됐다.



### Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets (https://arxiv.org/abs/2606.25760)
- **Prior Approaches**: 기존 컴퓨터-사용 에이전트 연구는 GUI 클릭 정합에서 불확실성을 다루더라도, 보통 특정 VLM-벤치마크-인터페이스 조합에 한정돼 사후 UQ(uncertainty quantification) 선택이 다른 조건에서도 유지되는지 불명확했습니다. 또한 open-weight와 closed-source 환경을 같은 프로토콜로 비교하는 연구가 적어, 로그it·hidden state·attention 같은 관측 가능 신호 유무가 UQ 순위에 미치는 영향을 정량화하기 어려웠습니다.

- **Core Contribution**: 이 논문은 단일-step GUI grounding에서 사후 UQ가 “어떤 배치(에이전트/데이터셋/관측 가능한 인터페이스)”로 옮겨 갈 때도 오류 순위가 유지되는지 교차-레짐 일반화를 체계적으로 측정하는 Argus를 제안합니다. Argus는 open-weight 4개 VLM 에이전트와 4개 데이터셋에 대해 27개 방법을 2,727개 스코어로, 내부 신호를 얻기 어려운 closed-source는 3개 벤더×동일 데이터셋 조건에 대해 8개 방법을 API-only 패널로 구성해 비교할 수 있게 합니다.

- **Technical Challenges**: 핵심 과제는 UQ “스코어”가 아니라 “방법 순위”가 레짐 전환에도 안정적인지 검증하는 것으로, 이를 위해 오류 탐지·선택적 실행·캘리브레이션·miss-severity 랭킹·공간 click-영역(동심 원/원형 구간) 커버리지까지 다목적 지표로 평가를 분리했습니다. 또한 closed-source에서는 로그it/hidden/attention이 없으므로 공통 API-호환 방식으로 비교 가능한 방법 부분만 남기고, 관측 불가능 신호 기반 방법은 동일 공식의 대체(프록시)로 통합 패널을 구성해 비교의 공정성을 확보했습니다.

- **Empirical Impact**: 결과적으로 UQ 방법 순위는 고정된 모델 내에서는 데이터셋 전환에도 비교적 안정적이지만, 모델 클래스 전환과 관측 인터페이스 변화(API-only)에서는 안정성이 크게 저하되는 “selective transfer” 양상이 나타났습니다. open-weight에서는 hidden-state·density(예: SEP/SAPLMA 계열)가 전반적으로 가장 안정적인 편이지만, closed-source로 갈수록 평균 순위 이전 성능이 거의 0에 가까워져 타겟 환경에서 재캘리브레이션/재랭킹이 필요하다는 결론을 제시합니다. 또한 conformal click-disk는 점수만으로 배치 가능한 커버리지를 보장하지 못해, 플러그인 UQ 캘리브레이션 시 반경이 40~60% 줄어드는 대신 커버리지는 조건에 따라 악화될 수 있음을 실증했습니다.



### Calousel: Extrinsic Calibration of Non-overlapping Multi-camera Systems from Pure Rotation (https://arxiv.org/abs/2606.25646)
Comments:
          Accepted to IROS 2026. 8 pages, 7 figures

- **Prior Approaches**: 기존 다중 카메라 외부 파라미터 보정은 FOV가 겹치는 경우 feature matching으로 해결되지만, 겹치지 않는 경우 직접 대응점이 없어 난도가 커진다. 대표적으로 타깃 기반은 큰 캘리브레이션 보드나 다중 타깃을 써야 하고, 미리 측정된 6D 포즈 요구로 센서 수가 늘수록 설치 오버헤드가 커진다. 모션 기반은 visual SLAM/SFM 기반이라 drift, scale ambiguity, motion degeneracy에 취약하고, 정밀 모션 하드웨어를 쓰는 하이브리드도 결국 세팅 복잡도가 남는다.

- **Core Contribution**: 이 논문은 non-overlapping FOV 환경을 겨냥해 단일 고정 캘리브레이션 보드와 single-axis turntable의 순수 회전만으로 카메라 간 extrinsic calibration을 수행하는 방법을 제안한다. 모든 카메라가 시간차로 같은 타깃을 관측하되, latent turntable frame을 두고 SE(3) 상의 3D 오차를 전역 최적화에 넣어 순차 관측을 하나의 기하 기준으로 통합한다. 그 결과, 정밀 장비 없이도 현장(on-site) 재보정에 적합한 “공간 효율 + 정확도”를 동시에 노린다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 겹치지 않는 FOV에서 시간차로 얻은 관측을 일관된 기준으로 합치는 것, (2) 순수 회전만 있을 때 생기는 gauge ambiguity와 최적화 불안정성을 다루는 것이다. 저자들은 latent turntable frame을 도입해 카메라 마운팅의 시간 불변 기하와 회전에 따른 시변을 분리하고, 기준축/원점 제약으로 gauge freedom을 줄인 뒤 global optimization에서 SE(3) 3D error를 최소화한다. 또한 rolling-shutter로 인한 타깃 기반 pose 추정 편향을 lightweight 보정 모듈로 완화하고, 강한 운동학 제약에서 생기는 공분산 상관 문제는 가중치 단순화(대각 성분만 사용)로 안정화한다.

- **Empirical Impact**: 제안 방법은 제어된 카메라 rig과 이질적 카메라가 장착된 full-scale vehicle 플랫폼에서 모두 실험적으로 검증되며, 현실적인 turntable 비이상(축 흔들림/진동)에도 경쟁력 있는 정확도를 유지한다. 특히 specialized precision hardware 없이도 정확도가 확보된다는 점을 강조하며, 캘리브레이션 세팅의 실용성을 입증한다. 또한 rolling-shutter compensation과 latent frame/3D error 같은 설계 선택의 타당성을 별도 실험으로 분석해, 현장 적용 시 성능 변동 요인을 줄이는 방향성을 제시한다.



### 1000 Rallies: An Event-Camera Dataset and Real-Time Learned Ball-State Estimation for Robotic Table Tennis (https://arxiv.org/abs/2606.25620)
- **Prior Approaches**: 로보틱 탁구는 공의 빠른 동역학과 지연 제약 때문에, 실시간 공 상태(위치·속도·필요 시 회전) 추정과 궤적 예측이 핵심 난제로 꼽혀 왔습니다. 기존 방법은 프레임 기반 카메라의 프레임레이트-비용 트레이드오프, 또는 event camera에서 RoI에 의존하는 고전적 검출·추적이 폐색/배경 잡음에 취약하다는 한계를 보였습니다. 또 일부 학습 기반 event ball detector는 업데이트레이트나 속도 추정이 제한되어 실제 제어에 바로 쓰기 어려웠습니다.

- **Core Contribution**: 이 논문은 탁구용 event-camera 대규모 데이터셋을 최초로 제시합니다. 총 1200여 랠리(아마추어~엘리트 선수) 5시간 분량을 4대 event 카메라로 수집하고, 동기화된 다중 APS 카메라(최대 14대)로 공 position/velocity/spin에 대한 1 kHz pseudo ground-truth 라벨을 생성합니다. 또한 CNN이 이벤트로부터 이미지 평면의 공 위치와 속도를 단일 샷으로 함께 추정하도록 설계해, 이후 필터 기반 예측의 입력을 강화합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 이벤트 스트림에서 배경 선수 움직임·폐색을 견디는 공 상태 추정의 강건성, (2) APS 기반 3D 정보로부터 kHz 수준 라벨을 안정적으로 만들고, (3) 빠른 공에 맞춰 EKF가 일관된 공분산으로 궤적을 추정하도록 초기 상태를 구하는 문제였습니다. 논문은 ball-circular 파라미터화를 YOLO v4-tiny 계열 CNN에 맞춰 2D 속도(vx, vy)를 직접 회귀하고, CNN의 velocity 예측을 EKF의 추가 측정으로 넣어 bounce-point 오차를 줄였습니다. 더불어 배경과 공간적 변환에 대한 일반화를 위해 물리적 일관성을 유지하는 데이터 증강과, C++/TensorRT 기반 병렬 추론으로 실시간 파이프라인을 최적화했습니다.

- **Empirical Impact**: 검출 품질 측면에서, 제안 모델은 Ziegler et al. 데이터셋에서 MAE 0.91 pixels, IoU 0.78의 성능을 보이며 기존 event 기반 접근을 능가합니다(파인튜닝·입력 수정 없이 비교). 궤적 예측에서는 EKF에 이미지 평면 velocity 측정을 추가하는 방식이 bounce-point prediction error를 36% 줄였고(100 ms 단위 기준), 업데이트레이트 변화에 따른 정확도 영향도 분석했습니다. 마지막으로 Stäubli TX2-60L 로봇에 perception-action loop를 통합해, event-based 인식에 기반한 실시간 인간-로봇 탁구 랠리를 실제로 시연하며 해당 분야의 실용성과 연구 기반을 동시에 확장했다는 점에서 의미가 큽니다.



### Cross-Attention Multimodal Learning for Predicting Response to Neoadjuvant Imatinib in Gastrointestinal Stromal Tumors: A Multicenter Retrospective Study (https://arxiv.org/abs/2606.25579)
- **Prior Approaches**: GIST에서 신보조(adjuvant) imatinib에 대한 반응은 개인차가 커서 기존의 임상 지표나 분자 마커만으로는 신뢰성 있게 예측하기 어렵다. 이에 따라 CT 영상과 임상 변수를 함께 쓰는 딥러닝 시도가 있어왔지만, 데이터 중심성/일반화/해석 가능성에서 한계가 있었다. 특히 영상 단독 모델은 외부 데이터로 갈수록 성능이 크게 흔들리는 문제가 지적된다.

- **Core Contribution**: 이 논문은 CT(종양 중심) 영상과 임상 변수를 동시에 다루는 explainable 멀티모달 딥러닝 프레임워크를 제안해 imatinib 반응을 예측한다. 핵심은 clinical 변수와 CT 특징을 cross-attention으로 결합해, 어떤 양상이 반응 예측에 기여하는지 해석 가능한 형태로 제공하는 점이다. 또한 self-supervised pretraining과 low-rank adaptation을 활용해 학습 전략까지 함께 비교한다.

- **Technical Challenges**: 멀티모달 결합에서 (1) 서로 다른 정보원을 안정적으로 정렬하고, (2) 외부 코호트에서도 일반화 성능을 유지하며, (3) 예측 근거를 신뢰할 수 있게 설명하는 것이 기술적 난제다. 저자들은 cross-attention으로 두 모달리티의 상호작용을 학습하고, self-supervised pretraining(또는 scratch) 및 low-rank adaptation을 통해 표현 학습을 강화했다. 하이퍼파라미터는 SMAC3로 최적화하고, internal cross-validation과 external test, ablation 및 attention 기반 설명으로 모달리티 기여도를 정량화했다.

- **Empirical Impact**: 총 213명의 환자에서 responder는 더 큰 종양(112 vs 89 mm)과 더 높은 mitotic index, 더 높은 KIT mutation 빈도를 보였고(통계적으로 유의), 모델도 이 패턴 차이를 반영하는 해석 결과를 제시했다. cross-attention 모델은 internal 성능에서 AUC 최대 0.99까지 올랐지만 외부 테스트에서는 AUC 0.60-0.63으로 감소했으며, 임상-only는 중간 수준(AUC 0.66), 영상-only는 제한적 일반화(AUC 0.56-0.66)를 보였다. attention 설명에서는 CD117, BRAF, PDGFRA, 연령·성별, 질병 상태, 동반질환 등 반응군/비반응군 간 feature importance 차이가 FDR 보정 기준 유의(<=0.036)해, 치료 반응 결정 요인을 임상적으로 해석 가능하게 하는 의미가 있다.



### Energy-Efficient CNN Acceleration with MSDF Digit-Serial Arithmetic on FPGA (https://arxiv.org/abs/2606.25562)
Comments:
          Presented at 2025 32nd IEEE International Conference on Electronics, Circuits and Systems (ICECS)

- **Prior Approaches**: U-Net의 연산 병목은 convolution의 내적(inner product)이며, CPU/GPU에서는 높은 전력·연산 비용 때문에 임베디드 배치가 어렵다. 이를 줄이기 위해 FPGA에서 bit-serial/MSDF 같은 자릿수 시리얼 산술이 주목받았지만, MSDF는 첫 출력까지의 고정 초기 지연(initial delay) δ가 생겨 지연이 연쇄 연산(곱→더하기)에 누적되는 문제가 있다. 기존 MSDF 곱셈과 가산을 분리해 파이프라인하면 adder tree 단계마다 지연이 추가돼 처리량을 깎는다.

- **Core Contribution**: 논문은 MSDF 기반 곱셈과 누산(accumulation)을 한 파이프라인으로 결합한 merged multiply-add(MMA) 구조를 제안한다. 이로써 기존처럼 곱 유닛의 초기 지연과 가산 트리의 초기 지연을 따로 겪는 대신, 반복(iteration)당 지연을 1회로 단순화해 첫 출력까지의 병목을 줄인다. 또한 MMA를 공간적(spatial) 입력 depth에 대해 병렬 처리하도록 구성해 U-Net의 3×3 convolution을 고처리량으로 가속한다.

- **Technical Challenges**: 핵심 과제는 MSDF의 초기 지연 때문에 생기는 ‘부분 정보 기반 결과’가 이후 사이클에 의해 수정되어야 한다는 점이다. 이를 위해 redundant number system(RDNS)으로 signed digit(SD) 표현과 Inverted Encoding of Negabit(IEN)을 사용해 자리수 집계 과정에서 발생할 수 있는 오류를 후속 사이클에서 바로잡을 수 있게 했다. 더불어 MMA 내부에서 AND 기반의 비트곱과 carry-chain을 활용한 트리형 덧셈을 융합하고, residual 피드백까지 포함해 자릿수 스트리밍을 유지하면서 지연을 최소화했다.

- **Empirical Impact**: Zynq-7020 FPGA에서 U-Net convolutional layer를 구현해 CPU 대비 최대 10배 수준의 에너지 효율 향상(15.14 GOPS/W vs 1.93 GOPS/W)과, MSDF 기반 FPGA 구현 대비 약 9× 에너지 절감이 보고됐다. 주파수는 CPU보다 낮지만 처리량과 에너지 효율 모두에서 유리하며, GPU의 높은 절대 처리량은 인정하면서도 전력 제한 환경에서는 더 적합하다는 점을 실험으로 보여준다. 이 결과는 resource-constrained·latency-sensitive edge 의료영상/컴퓨터비전에서 merged arithmetic 아이디어가 실용적 가속 경로가 될 수 있음을 시사한다.



### ASSCG: Just-Right Gating over Chattering for Fast-Slow LLM Planning in Autonomous Driving (https://arxiv.org/abs/2606.25509)
- **Prior Approaches**: 기존 fast–slow 드라이빙 플래너는 매 프레임 또는 장면 단위 고정 주기로 slow(LLM) 모듈을 호출하거나, scene complexity 같은 휴리스틱/불확실성으로 호출 타이밍을 조절하는 방식이 많았습니다. 그러나 고정 스케줄은 장면 내 시간적 변동을 놓치고, 난이도 프록시는 ‘해당 시점에 slow를 호출했을 때의 한계 효용’을 잘 반영하지 못해 성능 저하나 비효율이 발생했습니다.

- **Core Contribution**: 이 논문은 slow-system invocation을 프레임 레벨의 자원 제약 sequential decision 문제로 정식화하고, Adaptive Slow-System Control Gate(ASSCG)를 제안합니다. ASSCG는 각 프레임에서 Query/Cache/Drop을 선택해 slow 지침을 갱신·재사용·의도적으로 억제함으로써 닥치는 대로 호출하는 문제를 해결합니다. 또한 RWKV 기반 게이트를 설계하고, supervised fine-tuning 이후 compute-aware GRPO-style reinforcement fine-tuning으로 닫힌루프 성능과 호출 비용을 함께 최적화합니다.

- **Technical Challenges**: 핵심 기술 난제는 부분 관측(POMDP) 환경에서 slow 지침의 ‘가치’가 시간에 따라 달라지는 상황을 프레임 단위로 학습하는 것입니다. 논문은 이를 위해 Equivalent Interval(중복 호출 구간), Effective Interval(캐시 유효 구간), Failure Interval(호출이 해를 주는 구간)이라는 운영 기준을 도입하고, 그에 대응하도록 Cache는 연장 사용, Drop은 실패 구간 억제 기능을 갖는 3-way 게이트를 학습합니다. 게이트의 긴 문맥을 효율적으로 처리하기 위해 Transformer의 제곱 복잡도 대신 RWKV로 low-latency long-horizon 결정을 구현합니다.

- **Empirical Impact**: ASSCG를 AsyncDriver에 결합한 AdaptiveAsyncDriver는 nuPlan Hard20 closed-loop에서 점수 67.28(+2.28)로 개선되면서 평균 end-to-end inference latency는 60% 줄였습니다. 또 RecogDrive 기반 dual-system(NAVSIM)에서도 PDMS 91.4(+0.6), 평균 속도 약 25% 향상을 보이며 서로 다른 아키텍처에서도 동일 원리의 효율-정확도 이득이 재현됨을 확인했습니다. 결과적으로 ‘slow를 얼마나 자주 부를지’에서 나아가 ‘어떤 구간에서 신뢰하고 언제 버릴지’를 학습하는 접근이 fast–slow planning의 실사용 제약을 더 직접적으로 줄였다는 의미가 있습니다.



### AISPO: Enhancing Depth Reliability for Robotic Manipulation of Non-Lambertian Objects via Affine-Invariant Shape Prior (https://arxiv.org/abs/2606.25503)
Comments:
          Published in IEEE Robotics and Automation Letters. 8 pages. Accepted April 2026

- **Prior Approaches**: 기존 depth completion 및 단일/다중 시점 depth 추정 연구는 RGB guidance나 geometric constraint로 결손·잡음을 보정하지만, RMSE/MAE 같은 평균 오차 최소화에 치우쳐 로봇 조작에서 치명적(depth artifact) 실패를 충분히 막지 못하는 한계가 있었다. 투명·투과·고반사(non-Lambertian) 표면에서는 센서 잡음과 누락이 커서 잘못된 깊이가 motion planning으로 전파되며, NeRF·최적화 계열은 품질은 좋아도 지연과 계산 부담이 커 실시간 조작에 불리했다. 또 단안(depth from monocular)은 metric 입력이 없어 절대 깊이 정확도와 구조 보존이 흔들린다는 문제가 남아 있었다.

- **Core Contribution**: AISPO는 비라미안 객체에서 depth의 ‘신뢰도’를 높이기 위해 multi-scale RGB-D feature fusion과 affine-invariant shape prior를 결합한 depth completion 프레임워크를 제안한다. 평균 오차 개선보다 물리적으로 말이 되는(depth map의 구조 무결성) 기하 일관성을 우선하며, affine 불변 정규화로 객체별 scale/shift 변이를 흡수해 예측의 안정성을 강화한다. 또한 shape-prior autoencoder를 두 단계 학습으로 구성해 예측 깊이의 구조적 무너짐(catastrophic failure)을 줄이는 데 초점을 둔다.

- **Technical Challenges**: 주요 기술 난제는 투명·고반사 표면에서 원시 depth가 심하게 깨지거나 픽셀이 누락된다는 점이며, 동시에 RGB만으로는 metric geometry를 직접 복원하기 어렵다는 점이다. AISPO는 DINOv2 기반 RGB 인코더, Swin-Transformer 기반 raw depth 인코더, 그리고 affine-invariant shape prior 인코더를 병렬로 두고 cross-attention으로 멀티스케일 정보를 융합해 구조를 유지하도록 설계했다. 학습에서는 L1 loss에 masked loss와 Sobel gradient loss를 더해 객체 경계의 선명도와 기하 정합성을 강화하고, shape prior를 통해 결손 깊이를 보정하도록 했다.

- **Empirical Impact**: DREDS-CatKnown, STD-CatNovel, ClearPose, ClearGrasp 등 합성/현실 벤치마크에서 AISPO는 기준선 대비 깊이 예측의 구조 보존과 일관성을 높이며, 특히 unseen 객체·새 장면에 대한 zero-shot 일반화에서도 강점을 보였다. 더 중요한 실험으로 Franka 로봇 팔 grasping을 수행한 결과, 기존 방법들은 투명 물체에서 잘못된 depth로 인해 그립 제안이 무효가 되어 실패했지만 AISPO는 grasp success rate를 크게 끌어올렸다. 또한 RTX 3090에서 28.79 ms/frame의 추론 지연을 보고해 조작 시나리오에서 속도-정확도 균형도 현실적으로 확보했다.



### Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation (https://arxiv.org/abs/2606.25432)
Comments:
          36 pages, see this https URL for more information

- **Prior Approaches**: 기존 추론 효율은 주로 distillation, pruning, quantization, sparse routing 같은 방식으로 모델 자체를 줄여 토큰당 비용(FLOPs/token)을 낮추는 데 집중해 왔다. 하지만 최근에는 출력 길이(verbosity)가 빠르게 늘며, 표준 툴킷이 손대지 않은 “정답까지 필요한 토큰 수”가 전체 Cost-of-Pass를 좌우하는 상황이 커졌다. 또한 chat RLHF와 reasoning 모델에서 길이 편향이 품질 향상을 상당 부분 설명할 수 있다는 연구들이 나왔지만, VLM에서 그 비용 원인이 무엇인지 분해해 검증한 사례는 드물었다.

- **Core Contribution**: 이 논문은 “짧게 말해도 정답이 되도록” 만드는 것이 추론 효율의 누락된 레버라고 주장하며, 그 실현 수단으로 pretraining 데이터 curation을 제안한다. 간결하고 정답적인 데이터로 학습하면 모델이 더 적은 토큰으로 답을 생성하는 경향을 학습해, Cost-of-Pass를 감소시키는 효과를 얻는다는 관점이다. 특히 VLM에서 brevity(간결성)와 quality(정확성), 그리고 두 축의 trade-off가 실제로 어떻게 분리되는지 비용 기준(정답당 FLOPs)으로 답한다.

- **Technical Challenges**: 핵심 기술적 난제는 “더 짧아서 싸진 것”인지 “질이 올라서 덜 틀려 싸진 것”인지, 혹은 둘 다인지 비용-정확성 요인을 분리해 보여주는 것이다. 저자들은 VLM pretraining curation 파이프라인을 MAmmoTH-VL 단일 이미지 subset에 적용하고, curated 데이터로 학습한 모델과 동일 조건의 baseline(Mammoth 데이터) 및 외부 open-weight frontier VLM을 비교하되, cost는 decode-dominated FLOPs proxy로 정답당 비용(Cost-of-Pass)을 계산해 토큰 길이 변수를 직접 가격에 반영한다. 또한 출력 길이 분포가 겹치지 않는 경우까지 고려해 length-matched 회귀/분해 실험으로 brevity와 reasoning-structured verbosity의 효과가 다름을 추적한다.

- **Empirical Impact**: 실험 결과, curated 데이터 학습 모델은 정답당 비용을 크게 낮추면서 정확성도 유지하거나 일부 개선했다. 예를 들어 Datology 4B는 가장 verbose한 4B 비교군(Qwen3.5-4B) 대비 Cost-of-Pass에서 약 35배 이점을 보이면서 정확성 차이는 약 1pp 수준(0.41 vs 0.691? 본문 수치: 정확도 0.691 vs 0.704, 0.41 vs 14.58 TFLOPs per correct)으로 나타났다. 또한 동일 길이 조건에서 curated 모델의 matched-length accuracy가 더 높았고(풀드 평균 +17.55pp), “generic verbosity”는 어떤 스케일에서도 정답 정확성을 안정적으로 올리지 못했으며 reasoning 형태의 verbosity 이점도 스케일이 커질수록 줄어드는 패턴을 확인했다.



### Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning (https://arxiv.org/abs/2606.25347)
Comments:
          Accepted to ECCV 2026. 17 pages, 4 figures, 3 tables. Code: this https URL

- **Prior Approaches**: EFCIL에서는 메모리 버퍼가 없어 class-conditional Gaussian 통계(주파수/모멘트)를 유지·전파하는 방식이 많이 쓰인다. 하지만 최근의 대표적 “train-then-adapt” post-hoc 전파는 백본 업데이트 시 발생하는 표현 드리프트(특히 anisotropic drift)로 인해 이전 manifold가 훼손되면, Mahalanobis 기준선이 흔들려 분류 마진이 깎일 수 있다. 또한 topological degradation이 생기면 post-hoc 어댑터가 왜곡된 공간을 정밀하게 되돌리기 어렵다.

- **Core Contribution**: 이 논문은 feature transport를 학습 뒤에 따로 맞추는 post-hoc 단계가 아니라, 주어진 학습 단계 동안 end-to-end로 “구속(constraint)”하는 Geometry-Anchored Transport Framework를 제안한다. 핵심은 Analytic Geometric Anchor(AGA)로 Mahalanobis 기하에 정렬된 closed-form 선형 사전(prior)을 주고, Topology-Aware Evolution으로 지역적인 manifold 붕괴를 억제하면서 잔차(residual) 보정이 이를 보완하도록 하는 구조다. 결과적으로 old-class 통계 pushforward의 신뢰도를 높여, decoupled fine-tuning 없이도 평가 오차를 줄이는 것을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “닫힌 형태(선형) 기하 사전”을 역전파 가능한 학습 궤적에 맞춰 안정적으로 결합하는 것이다. 단순히 백본을 학습한 뒤 선형 어댑터로 맞추면 저분산 방향(low-variance directions)에서 Mahalanobis 오차가 역행렬로 증폭되고, manifold의 국소 위상(토폴로지)이 망가져 사전이 커버하지 못한다. 이 논문은 GLS 기반 Mahalanobis-aligned regression으로 AGA를 Sylvester equation 형태로 해석해 macroscopic anisotropic drift를 줄이는 한편, topology-aware 손실이 백본이 legacy neighborhood를 훼손하지 않도록 학습 중에 함께 제한하며, AGA는 EMA로 매끈하게 갱신해 잔차 캘리브레이션을 흔들림 없이 수행한다.

- **Empirical Impact**: CIFAR-100, TinyImageNet, ImageNet-100에서 exemplar-free 조건 하에 기존 post-hoc 중심 방법들 대비 final-task 및 누적 성능이 일관되게 개선됐다. 예를 들어 CIFAR-100에서 T=10/20 설정 모두에서 DPCR 및 AdaGauss 대비 AlastA와 Ainc가 유의미하게 향상되었고, TinyImageNet에서도 DPCR 대비 +여러 pp 수준의 격차가 관찰됐다. 특히 ImageNet-100에서는 두 작업 수열 모두에서 높은 final-task 정확도를 보이며, 표현 드리프트가 완화되는 CUB-200에서도 사전 기반 transport-정규화가 경쟁력 있는 유지 성능을 제공한다는 점에서 의미가 있다.



### An Integrated Hardware-Software Design for Low-Data Spatial Defect Detection in Robotic Visual Inspection with Hybrid Optoelectronic Neural Networks (https://arxiv.org/abs/2606.25277)
- **Prior Approaches**: 기존 로봇 비전 결함 검출은 고해상도 이미지를 기반으로 하되, 데이터가 기하급수적으로 늘어나 저장·엣지 연산 부담이 커진다는 한계가 있다. 또한 YOLO 같은 탐지 프레임은 박스/분류 라벨에 크게 의존하고, 특히 결함의 ‘모양’ 수준(shape-level) 주석은 인력 비용이 높으며 라벨 일관성 문제도 동반된다. 압축센싱은 데이터량을 줄이지만, 대부분 복원(reconstruction) 계산이 비싸거나 복원-후처리로 오류가 누적되며 여전히 입력 데이터 최적화가 부족하다고 지적한다.

- **Core Contribution**: 이 논문은 DMD 기반 광학-물리 계층과 신경망 소프트웨어를 한 시스템으로 묶은 optoelectronic 아키텍처를 제안한다. 센서-in-the-loop로 DMD를 ‘물리적 optical convolutional layer’로 재구성해, 광학 도메인에서 특징 추출과 차원 축소를 수행함으로써 전통적 CS 복원 단계를 제거한다. 더불어 결함 모양 주석 없이 자연어 결함 설명을 CLIP(Contrastive Language-Image Pre-training)의 임베딩으로 연결해, 네트워크 attention을 결함 형상에 정렬시키는 학습 방식을 도입한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) CS 측정과 딥러닝 특징 추출을 물리 계층에서 결합해도 성능이 유지되는지, (2) DMD의 이진(ON/OFF) 구동 제약 하에서 부호/가중치 표현을 어떻게 구현할지, (3) 모양 정밀도를 평가할 수 있는 정량 지표가 없는 문제다. 저자들은 measurement matrix를 신경망의 첫 레이어 convolution 커널로 선택하고, 블록 기반 compressed sensing으로 공간 정보를 저차원 시간 신호에 인코딩하며, DMD에서 Unsigned/Bool/Signed 등의 파라미터 표현을 분해(예: Signed를 양·영/음·영으로 분리)해 구현한다. 마지막으로 heatmap 기반 결함 위치 품질을 계량하는 LAA(Localization Accuracy for Attention) 메트릭을 제안해 shape-level localization을 직접 평가한다.

- **Empirical Impact**: 투명 소재의 tiny defect(spot/scratch) 검출 실험에서, 제안 구조는 기존 imaging 대비 데이터 중복을 90% 줄이면서도 분류·국소화 정확도를 동급 수준으로 유지했다고 보고한다. 또한 Vision Transformer에서는 데이터 90% 감축과 함께, CNN에서는 연산량을 약 60% 줄이는 효과가 관찰된다. 파라미터 분석에서는 measurement matrix, compression ratio, block size가 정확도와 LAA에 미치는 영향을 보여주며, 특히 ViT와 binary matrix(Bool/Sign) 조합에서 샘플링률이 올라갈수록 안정적인 성능 향상이 나타난다고 정리한다. 데이터 스트림이 대규모이고 촬영 비용이 높거나 엣지 자원이 제한된 산업 자동화 시나리오에 실용적인 ‘low-data’ 해법이라는 점에서 의미가 크다.



### Dual Agreement Consistency Learning for Semi-Supervised Fetal Ultrasound Segmentation (https://arxiv.org/abs/2606.25254)
Comments:
          Accepted to MICCAI 2026

- **Prior Approaches**: 태아 초음파 분할은 기기 휴대성·비침습성 덕분에 표준 모달리티지만, 픽셀 레벨 라벨이 부족해 자동 분할 정확도를 높이기 어렵다. 기존에는 경량 딥러닝 모델을 쓰거나 semi-supervised learning을 적용했으나, 제한된 라벨 상황에서 경량 네트워크로 지식을 안정적으로 옮기기엔 pseudo-label 품질과 학습 안정성이 부족하다는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 Dual Agreement Consistency Learning(DACL)이라는 semi-supervised 프레임워크를 제안하며, 서로 다른 구조의 모델 간 pixel-wise 합의를 기반으로 CPS를 안정화한다. 특히 CNN-Transformer 두 모델이 예측 분포를 맞추는 동시에 불확실성(엔트로피)까지 정렬하도록 dual-agreement consistency loss를 설계해, 신뢰도 낮은 pseudo-label이 학습을 흔들지 않게 한다.

- **Technical Challenges**: 핵심 기술적 난제는 극단적으로 적은 라벨에서 pseudo-label의 신뢰도를 유지하며 cross-architecture 지식을 안정적으로 학습하는 것이다. DACL은 (1) KL 기반 확률 분포 정렬과 (2) 엔트로피 기반 confidence alignment을 함께 걸어 pseudo-label을 억제하고, 추가로 mixup 형태의 interpolation consistency로 unlabeled 샘플에서도 일관된 지도를 강화한다(EMA teacher 사용).

- **Empirical Impact**: HC18(태아 머리)과 F-Abd(태아 복부) 두 데이터에서 라벨 5% 조건 시 Dice는 최대 2.77%p 개선, HD95는 최대 14.69mm 감소로 경계 정확도가 뚜렷이 좋아졌다. 또한 boundary 민감 지표에서 최근 semi-supervised 방법 대비 우세하며, ablation에서도 dual-agreement loss가 특히 HD95/ASD 등 경계 품질을 크게 끌어올리는 것으로 확인돼 annotation-efficient 태아 US 분할에 의미 있는 진전을 보였다.



### Semantic Allocation in Ordered Bottlenecks: Predictive Residual Inference for Visual Representation Learning (https://arxiv.org/abs/2606.25232)
Comments:
          Accepted to ICANN 2026 main proceedings. 12 pages, 5 figures

- **Prior Approaches**: Ordered bottlenecks에서 흔히 쓰이는 방법은 학습 중 마스킹/트렁케이션으로 토큰 순서를 강제하며, 이를 masking-based ordering pressure(MBOP)로 설명한다. TD와 MBOP-ITD 같은 변형은 후반 토큰을 더 자주 가려 앞부분이 더 중요한 prefix로 자리 잡게 유도하지만, “후반 토큰이 더 약한 학습 신호”를 받는 구조적 문제가 남는다. 또한 명시적 refinement 목적 없이 그라디언트 노출을 중요도의 대리변수로 써서 고예산에서의 성능 상승이 둔화될 수 있다.

- **Core Contribution**: 이 논문은 예측적 잔차 추론 기반 ordered representation learning 프레임워크인 PRIOR를 제안한다. PRIOR는 순서 압력을 그라디언트 노출로부터 분리해, 로그 스케일 hierarchy와 레벨별 self-predictive 모듈이 “이미 설명된 정보”와 “남은 잔차”를 분리하도록 설계한다. 결과적으로 저예산에서는 거친 디스크립터를, 고예산에서는 이전 수준이 설명하지 못한 residual 디테일을 단계적으로 보강하는 구조를 학습한다.

- **Technical Challenges**: MBOP의 약점인 (1) 후반 토큰의 약한 학습 신호, (2) discrete/quantized 표현에서의 취약성, (3) 명시적 refinement 목표 부재를 동시에 해결하는 것이 핵심 과제였다. PRIOR는 activation-rate 제어 대신 log2-scaled 레벨 레벨링과 level-wise predictors로 각 레벨의 residual error에 집중해 예측-잔차-압축-누적(refinement) 경로를 만들며, 각 레벨 출력을 별도 헤드에 연결하되 기하 가중 손실로 레벨별 학습 신호를 안정화한다. 토큰 타입은 Gaussian, Categorical, EMA-VQ codebook vector를 포함하며, discrete 계열에서는 reparameterization/straight-through 등 추정 방식을 사용해 학습을 가능하게 했다.

- **Empirical Impact**: 대조 실험에서 PRIOR는 contrastive learning과 이미지 reconstruction 모두에서 budget이 늘어날수록 coarse-to-fine 순서가 일관되게 강화되는 패턴을 보였다. 특히 Categorical과 EMA-VQ 같은 discrete/quantized 설정에서 PRIOR는 MBOP 계열보다 peak 성능과 late-token 유틸리티를 크게 끌어올렸고, full-budget에서도(대부분의 실험에서) 기준 성능을 유지하거나 더 높게 나타났다. MBOP는 discrete/quantized에서 제한이 크며, reconstruction에서는 TD가 저예산에서만 강하고 이후 포화되는 반면 PRIOR는 활성 레벨이 늘어날수록 세부가 지속적으로 개선되며 3131 토큰 이후부터 우위를 확실히 점했다.



### Homomorphic Encryptions for Privacy Preserving Vision (https://arxiv.org/abs/2606.25216)
- **Prior Approaches**: 기존 프라이버시 보존 비전 연구는 CryptoNets, TenSEAL 같은 방식으로 완전동형암호(fully homomorphic encryption, FHE)를 활용해 암호문에서 추론을 수행하는 흐름을 만들었다. 다만 CNN의 비선형성/풀링 같은 연산을 암호화 체계(덧셈·곱셈 중심)에 맞춰 근사·대체해야 하고, 그 과정에서 정확도 하락과 추론 지연이 크게 나타난다는 한계가 있었다. 또한 공개 라이브러리의 성능 최적화(특히 GPU 지원 부족)로 인해 실사용 관점의 속도 개선이 어려웠다.

- **Core Contribution**: 이 논문은 암호문만으로 이미지 분류가 가능하도록 CNN을 ‘encryption based CNN’ 형태로 재구성하고, 암호화된 입력에 대해 분류 정확도와 추론 속도를 함께 최적화하는 방법을 제안한다. MNIST뿐 아니라 Kuzushiji MNIST, Fashion-MNIST, CIFAR-10으로 확장해 더 복잡한 데이터에서도 기준선과 유사한 정확도를 확보했으며, 색상(멀티채널) 입력, stacked convolutional layers, average pooling 같은 확장 기능을 TenSEAL 위에 구현했다.

- **Technical Challenges**: 핵심 기술 난제는 동형암호가 덧셈·곱셈만 직접 지원해 ReLU/Max pooling 같은 비다항 연산을 저차 다항으로 치환해야 한다는 점이다. 논문은 실수 정밀도를 고정 소수점으로 인코딩하고, 비선형성은 polynomial(제곱 등)로 대체하며, pooling은 scaled/mean 형태로 바꾸는 한편 convolution은 ‘image to columns’류의 재배열과 암호 텐서 연산(회전·합산)을 통해 구현했다. 나아가 stacked convolutional layers를 위해 층 사이에 decrypt/reencrypt를 활용하는 실무적 우회도 함께 제시하고, CKKS의 bit scale·polynomial modulus 같은 context 파라미터에 따른 정확도-지연 트레이드오프를 경험적으로 분석했다.

- **Empirical Impact**: 실험 결과, 상대적으로 단순한 아키텍처에서는 암호화 추론 시 분류 정확도 하락이 거의 없거나 소폭 개선되는 경향을 보였고, 더 깊은 구조에서는 연산이 추가된 만큼 정확도와 추론 시간이 함께 증가/감소하는 패턴이 확인됐다. 특히 MNIST/Kuzushiji MNIST/Fashion-MNIST에서 암호화 정확도가 기준선과 유사한 수준으로 유지되었으며, CKKS 파라미터를 높이면 정확도는 포화되고 추론 시간은 보안 파라미터(예: polynomial bit modulus)에 따라 증가할 수 있음을 보였다. 연구는 GPU 비지원으로 인한 속도 병목을 재확인하는 동시에, 라이브러리/암호 파라미터 튜닝과 레이어 설계 변경만으로도 ‘암호문 직접 추론’의 실현 가능성을 실증했다.



### An iterative energy-based multimodal transformer for joint retrieval of wheat soil moisture, leaf area index, and plant height from Sentinel-1 and Sentinel-2 time series (https://arxiv.org/abs/2606.25174)
- **Prior Approaches**: 정밀농업에서 토양수분(SM)·엽면적지수(LAI)·식물키(PH)를 위성으로 동시 추정하는 일은 역문제지만, 토양수분과 캐노피(수관) 밀도의 동시 변동이 레이더 후방산란과 스펙트럼 응답을 복잡하게 뒤섞어 모호성을 키운다. 전통적인 feedforward 회귀 모델은 소규모 소농 환경의 이질성 때문에 정확도가 쉽게 떨어지는 한계가 있었다.

- **Core Contribution**: 본 연구는 Sentinel-1 C-band SAR 시계열과 Sentinel-2 다중분광 시계열을 함께 써서 [SM, LAI, PH]를 동시에 복원하는 Iterative Energy-Based Transformer(iEBT)를 제안한다. 직접 회귀 대신, 멀티모달 입력을 공유 시퀀스로 임베딩하고 초기 상태 추정 후, 학습된 호환성(compatibility) 에너지 함수를 최소화하는 방식으로 normalized gradient descent로 반복 갱신해 상태 벡터를 정교화한다.

- **Technical Challenges**: 핵심 난제는 멀티모달 신호가 만들어내는 비가역적 모호성을 모델링하면서도, 역추정 결과의 신뢰도를 사후에 진단할 수 있어야 한다는 점이다. iEBT는 에너지 기반 적합도를 학습해 반복 업데이트를 안정화하고, 최종(terminal) 에너지 값을 ‘무보정(un-calibrated) 품질 진단’ 지표로 활용하는 self-diagnostic 구조를 포함시켜 샘플 선별로 오차를 줄였다.

- **Empirical Impact**: 인도 Varanasi의 700개 품질관리(field measurements) 기반 평가에서 iEBT는 랜덤 테스트 분할 기준 네 시드 평균 R^2가 0.854 ± 0.012로 가장 높은 성능을 보였고, SM·LAI·PH 각각도 R^2=0.841, 0.905, 0.821을 기록했다. Sentinel-1은 SM, Sentinel-2는 LAI에 더 크게 기여하며 PH는 둘의 구조-생장 시그니처 결합이 중요하다는 modality ablation이 이를 뒷받침했고, terminal energy 상위 10% 샘플을 쓰면 RMSE가 크게 감소했다; 다만 leave-one-campaign-out에서는 계절·현장 관리 차이로 인한 도메인 이동이 남아 cross-season 일반화 과제가 지속됨을 보여준다.



### fARfetch: Enabling Collocated AR-HRC in Large Visually Diverse Environments with VLM-Driven AR Content Adaptation (https://arxiv.org/abs/2606.25162)
Comments:
          Accepted to the 2026 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). Author accepted manuscript

- **Prior Approaches**: 기존 AR-HRC는 로봇과의 근접성이나 VLOS(visual line of sight)를 전제로 하는 경우가 많아, 야외처럼 로봇이 멀어지거나 시야가 막히면 목적지/경로 지시가 급격히 어려워진다. world-in-miniature(WIM) 기반 접근도 주로 목적지 선택에 머물고 의미 있는 semantic 문맥과 정밀 path authoring이 부족하며, 헤드셋이 감지한 랜드마크가 로봇의 이해로 공유되지 않는 한계가 있다. 또한 AR view management는 대개 정적인 대상 중심이라, 동적이고 복잡한 야외 배경에서 legibility를 안정적으로 유지하기 어렵다.

- **Core Contribution**: fARfetch는 Meta Quest 3와 Unitree Go2를 묶어, 로봇 근접 보장이 어렵고 VLOS가 끊겨도(collocated) AR에서 목적지와 fine-grained path를 지시할 수 있는 AR-HRC 시스템을 제안한다. 헤드셋과 로봇이 함께 탐지한 landmark를 shared semantic environment mapping으로 통합하고, 이를 real-world AR 오버레이와 context-aware WIM에 동시에 임베딩해 사용자가 의미 있는 위치를 기준으로 명령을 내릴 수 있게 했다. 여기에 VLM(vision-language-model) 기반 scene-aware view management로 가상 콘텐츠의 color·size·orientation을 런타임에 함께 조정해 먼 거리에서도 가독성을 유지하도록 설계했다.

- **Technical Challenges**: 핵심 과제는 멀리 있는(해상도 저하)·가려진 상황에서도 사용자가 가상 마커의 깊이/스케일/상대 위치를 신뢰하며 조작하게 만드는 것과, 야외의 복잡한 배경 변화에 따라 color·scale·각도 조정이 서로 충돌하지 않게 하는 것이다. fARfetch는 Grounding DINO와 Segment Anything으로 landmark를 정교화하고, CLIP 임베딩으로 중복을 줄여 shared semantic map을 만들며, 로봇 라이다의 point cloud를 누적해 WIM의 구조(메시)를 생성·병합해 의미+기하를 함께 제공한다. legibility 유지 측면에서는 카메라 frustum에 새 가상 콘텐츠가 들어올 때 VLM에 장면 이미지를 입력하고, 렌더 state(색/크기/방위)와 지각 맥락을 구조화 프롬프트로 제시해 VLM이 joint adaptation을 산출하도록 했다.

- **Empirical Impact**: 실험은 30.5m 규모의 실제 야외 inspection 과제를 대상으로 within-subjects 설계(N=13)를 적용했으며, fARfetch가 non-AR baseline 대비 완료 시간을 66% 단축했다. 또한 NASA-TLX 하위 지표에서 mental demand(-43%), temporal demand(-34%), frustration(-66%)을 유의미하게 낮춰 인지·시간 부담을 줄였다는 결과를 보였다. 맞춤 legibility survey와 콘텐츠 가독성 평가는 large outdoor 환경에서도 VLM-driven adaptation이 가상 콘텐츠 legibility를 효과적으로 유지함을 뒷받침하며, 야외 collocated AR 로봇 제어 실사용 가능성을 강화하는 의미가 있다.



### Toward Low-Latency Vision-Language Models with Doubly-Correct Predictions in Egocentric Visual Understanding (https://arxiv.org/abs/2606.25160)
Comments:
          International Conference on Intelligent Robots and Systems (IROS) 2026

- **Prior Approaches**: 기존 VLM pruning은 주로 accuracy만 최적화해 모델 크기와 FLOPs를 줄이지만, 인간-로봇 협업(HRC)에서 중요한 안전성 지표인 doubly-correct prediction(DCP)의 보존을 충분히 다루지 못했다. 또한 egocentric 비전의 공간·시간 근거까지 함께 평가하는 프로토콜이 부족해, pruning이 ‘정답 이유’를 얼마나 유지하는지 진단하기 어려웠다. 결과적으로 일부 방법은 근거 localization은 유지하는 듯 보여도 실제 결정(prediction)은 흔들리는 위험한 불일치가 생길 수 있다.

- **Core Contribution**: 논문은 egocentric 비디오에서 정답 라벨과 시공간 근거(언제/어디의 조작 대상)를 동시에 평가하는 spatio-temporal DCP 프로토콜을 제안한다. 이를 통해 기존 pruning이 종종 “올바른 근거는 남기지만, 그 근거가 올바른 결론으로 이어지지 않는” 현상을 드러낸다. 이어서 rational-informed pruning으로 근거와 의사결정의 정렬을 강화해 DCP 능력을 함께 보존하는 방법을 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 pruning이 backgroud(배경) 활성에 편향되어 근거→로짓으로 이어지는 전달을 끊어 WR(틀린 예측·맞는 근거) 같은 안전 취약 상태를 유발할 수 있다는 점이다. 저자들은 레이어별 중요도가 다르다는 가정을 바탕으로 가중치 magnitude로 비균일 per-layer pruning ratio를 추정하고, 모델이 생성한 rationale을 근거 마스크로 활용하되 의미 정합성(설명 객체-예측 라벨 임베딩 유사도)에 따라 마스크를 adaptive하게 보정해 배경 영향과 노이즈를 완화한다. second-order Taylor/OBS 관점에서 손실 증가를 근사해 가중치 중요도를 계산하고, 이 중요도에 따라 스파스화를 수행한다.

- **Empirical Impact**: EPIC-KITCHENS VISOR와 EgoExo4D에서 기존 OMP/GMP/다른 기준 pruning 대비 prediction accuracy뿐 아니라 IR(Inference Reliability)과 RR(doubly-right samples)을 더 높게 달성한다. 예를 들어 EPIC-KITCHENS VISOR에서 pruning ratio 30%일 때 spatial IR과 RR이 ECoFLaP 대비 각각 13.62%, 3.65% 향상되었고, PT(Prediction Trustworthiness)는 대부분 유지된다. 절제 실험에서도 rationale 정보와 마스크의 adaptive semantic factor가 빠지면 DCP 성능이 크게 저하되어, ‘안전한 pruning’을 실증적으로 뒷받침한다.



### Benchmarking the Alignment of Data-Quality Metrics, Human Judgment and Land-Cover Segmentation Performance for Earth Observation (https://arxiv.org/abs/2606.25128)
- **Prior Approaches**: 합성 데이터 품질 평가는 주로 FID, KID, Inception Score(IS), LPIPS, SSIM 같은 자동 지표로 신뢰도를 판단해 왔다. 특히 FID는 사전학습(대개 ImageNet) 특징 공간에서의 분포 거리로 계산되지만, 시각적 충실도는 잘 반영해도 실제 downstream 유용성과는 쉽게 어긋날 수 있다는 한계가 지적돼 왔다. 또한 도메인 특화 입력(지구관측)의 경우 ImageNet 표현이 그대로 전이되지 않아 평가지표 편향 가능성이 크다.

- **Core Contribution**: 이 논문은 지구관측(Earth observation)에서 자동 품질 지표, 인간 인식, 그리고 랜드커버 의미 분할 성능이 어떻게 정렬/불정렬되는지 하나의 실험 프레임으로 체계적으로 비교한다. 회전·잡음·원근 왜곡 등 의미 보존/의미 변화 변환을 걸어 지표의 취약성을 보여주고, 동시에 4단계 인간 설문과 분할 모델 벤치마크로 작업 유용성을 확인한다. 그 결과 “자동 지표 점수=실제 효용”이라는 통념이 성립하지 않는 패턴을 일관되게 제시한다.

- **Technical Challenges**: 핵심 기술 난점은 ‘시각적 품질(또는 분포 유사성)’을 측정하는 자동 지표가 의미적으로 중요한 변화에 민감하지 않거나(혹은 과도하게 민감), 반대로 인간이 실제로 평가하는 현실감과는 다른 신호를 추적할 수 있다는 점이다. 논문은 여러 유형의 perturbation에 대해 7개 자동 지표의 반응을 측정하고, 인간 인식(장면 동일성, 이미지-마스크 정합, 생성 선호, 현실감 점수)과 분할 성능(F1)까지 같은 데이터셋에서 비교해 이러한 불일치를 정량화했다. 특히 ImageNet 기반 특징 공간의 방향 민감성 같은 이유로 rotation에서 FID가 크게 흔들리지만 인간 인식 정확도는 거의 유지되는 현상을 관찰한다.

- **Empirical Impact**: 실험은 자동 지표가 세부적인 의미 보존 변환에도 급격히 점수가 변하는 반면, 인간 인식은 덜 민감하다는 ‘정합 실패’를 보여준다. 더 나아가 FID 같은 낮은-거리 점수는 분할 F1과 잘 맞지 않으며, 심지어 자동 지표가 나쁘게 나오는 합성 데이터도 실제 데이터와 혼합하면 분할 성능을 오히려 개선할 수 있음을 입증한다. 결론적으로 지구관측 합성 데이터 품질 평가는 downstream task 성능과 인간 평가를 기반으로 설계돼야 하며, 단일 자동 지표 최적화는 유용성 손실을 유발할 위험이 크다는 실무적 함의를 제공한다.



### ADM-Fusion: Adaptive Deep Multi-Sensor Fusion for Robust Ego-Motion Estimation in Diverse Conditions (https://arxiv.org/abs/2606.25111)
Comments:
          8 pages, 4 figures

- **Prior Approaches**: 기존 ego-motion 추정은 기하학 기반 파이프라인(예: SLAM, 최적화/필터)에서 시작해, 학습 모듈을 섞은 하이브리드 방식과 raw 멀티모달을 end-to-end로 결합하는 딥 러닝 융합으로 발전했다. 그러나 많은 방법이 고정 가중치나 안정적인 appearance/상응 관계/잡음 가정을 전제로 해서, 카메라-라이다-레이더-IMU가 서로 다른 방식으로 악화될 때(열화 패턴 상이) 강건성이 떨어질 수 있다. 이에 따라 다양한 센서 신뢰도 변동을 실시간으로 반영하는 적응형 융합의 필요성이 강조된다.

- **Core Contribution**: ADM-Fusion은 센서 열화와 환경 변화에 따라 멀티모달 기여도를 실시간으로 재가중하는 end-to-end 융합 프레임워크를 제안한다. 핵심은 Adaptive Sensor Mixture-of-Experts(ASMoE)로, content-aware routing을 통해 각 타임스텝에서 센서 가중치를 동적으로 산출하는 점이다. 또한 translation과 rotation을 분리된 브랜치로 학습하되, cross-task attention으로 두 작업 간 보완 정보를 교환해 task-specific specialization은 유지한다.

- **Technical Challenges**: 문제는 센서마다 실패 양상이 다르고 시간에 따라 신뢰도가 빠르게 변하므로, 고정된 가중치로는 융합이 붕괴할 수 있다는 점이다. 이를 위해 저자들은 (1) 센서별 특징을 Mamba로 시간적으로 보정한 뒤 (2) 타임스텝별 ASMoE 라우터로 신뢰도 기반 가중치를 할당하고 (3) router collapse를 막기 위한 uniform expert usage 정규화와 (4) translation/rotation 분리 학습 뒤 cross-attention으로 오차 전파를 제어한다. 시뮬레이션-실세계 차이를 줄이기 위해 CARLA-LOC 사전학습 후 KITTI에서 fine-tuning 전략도 함께 사용한다.

- **Empirical Impact**: 실험에서는 CARLA-LOC에서 학습 후 KITTI로 fine-tuning한 simulation-to-real transfer가 확인되며, 열화 조건에서도 견고성을 유지하면서 기존 방법과 경쟁/우위를 보인다. 특히 KITTI에서 rotation 드리프트는 LVIO 구성에서 가장 일관된 개선을 보였고, ASMoE가 IMU는 rotation에 더 크게, 라이다/카메라는 translation 안정화에 더 크게 기여하도록 시간적으로 부드럽게 가중치를 조절함이 관측된다. 또한 radar를 추가하면 translation RPE가 64% 감소하는 등 센서별 강점을 task에 맞게 비대칭적으로 활용하며, 라우팅-시간 설계의 중요성과 함께 ASMoE의 적응 전략이 실제로 효과적임을 뒷받침한다.



### Do vision-language models search like humans? Reasoning tokens as a reaction-time analog in classic visual-search paradigms (https://arxiv.org/abs/2606.25066)
- **Prior Approaches**: 기존 시각 탐색 연구는 반응시간이 항목 수(집합 크기)에 따라 어떻게 증가하는지로 병렬 ‘팝아웃’ 탐색과 직렬 ‘주의 요구’ 탐색을 구분해 왔다. 한편 VLM 평가는 주로 정확도 중심이라, 사람의 탐색 메커니즘을 ‘닮았는지’는 여전히 불명확했다.

- **Core Contribution**: 이 논문은 VLM이 사람과 같은 탐색 행동 서명(예: feature vs conjunction, 열거, 탐색 비대칭)을 보이는지 확인한다. 특히 모델의 반응시간 대체 신호로 reasoning/thinking 토큰 수를 사용해, 모델 내부 시간적 노력(노력-부하 기울기)을 사람의 검색 슬로프와 대응시키는 실험 프로토콜을 제안한다.

- **Technical Challenges**: 단일 forward pass에는 반응시간이 없어 고전적인 ‘search slope’를 그대로 적용하기 어렵다는 점이 핵심 난관이다. 저자는 모델이 자체적으로 생성하는 추론 토큰을 per-trial 노력 지표로 삼고, 추가로 해상도 블러 대조 실험으로 conjunction 비용이 ‘작은 글자 판독 난이도’가 아니라 실제 탐색 비용임을 분리하려 했다.

- **Empirical Impact**: 결과적으로 frontier(첨단) 추론 모델들은 feature는 effort가 평평하고 conjunction은 집합 크기에 따라 effort가 증가하는 등 사람의 거친 패턴을 재현했다. 다만 세부에서는 차이가 컸는데, 사람과 달리 ‘부재(타깃 없을 때)’ 확인을 더 어렵게 하지 않고 ‘존재(타깃 있을 때)’ 쪽에 더 많은 노력을 쓰거나, 열거에서는 사람의 undercount 양상 대신 accuracy를 유지하되 compute를 더 쓰는 식으로 발현됐다.



### Learning Diachronic Representations of Ancient Greek Letterforms (https://arxiv.org/abs/2606.24984)
Comments:
          Accepted for publication at the International Conference on Document Analysis and Recognition (ICDAR) 2026

- **Prior Approaches**: 기존 OCR·문서 인식 연구는 대체로 문자 형태가 안정적이고 충분한 학습 데이터가 있다는 가정을 둔다. 이 때문에 수세기에 걸친 서체 진화(구조적 드리프트)와 자료 열화, 불균형·소량 데이터 환경에서는 일반적인 transfer나 contrastive learning이 문자 간 ‘본질적 유사성’을 적절히 반영하지 못한다. 특히 supervised contrastive learning 계열은 서로 다른 클래스라도 형태가 닮은 경우를 동일한 음성(negative)으로 강하게 밀어내는 문제가 있어, 자연스러운 inter-class 관계를 학습 임베딩에 반영하기 어렵다.

- **Core Contribution**: 논문은 고대 그리스 필기에서 수백 년 단위로 바뀌는 문자 형태에 강건한 표현을 학습하기 위해, 두 가지 도메인 지식을 학습 목표에 결합한다. 핵심은 similarity-weighted supervised contrastive loss로 형태적으로 헷갈리는 문자쌍은 음성 repulsion을 약화시키고, lacuna-driven augmentation으로 실제 파피루스 손상 양상을 닮은 결손을 학습에 포함시키는 것이다. 또한 3rd BCE~1st BCE 학습용 Hell-Char와 2nd~5th CE 평가 PaLit-Char, 9th~14th CE 평가 Med-Char의 3종 벤치마크를 제공해 시간적 일반화 성능을 체계적으로 측정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 문자 간 상징적 변이와 (2) 결손·잡음으로 인한 체계적 열화, (3) 클래스별 소량·불균형 데이터다. 저자들은 동적으로 추정한 inter-class similarity에 따라 negative pair 가중치를 조절하는 DSCL로 ‘시각적으로 유사하지만 다른 문자’의 과도한 분리를 막고, lacuna-driven augmentation(LF)로 결손이 만드는 비직사각형 패턴을 생성해 부분 가림에도 임베딩이 유지되도록 설계했다. 더불어 대표 프로토타입(메도이드) 기반 시각화로 클러스터 구조가 해석 가능한 방식으로 형성되는지 확인한다.

- **Empirical Impact**: 실험에서 LF+DSCL을 적용한 lightweight CNN과(pretrained) ResNet18이 PCA나 generic pretraining 대비 더 응집적·분리도가 높은 임베딩을 만든다. Hell-Char에서 인식 성능이 상승할 뿐 아니라, 임베딩 클러스터링으로 문자별 하위 서브그룹(스타일 변종)과 시대별 과도기 글자 형태(prototype)를 더 명확히 드러냈다. 시간 일반화 측면에서는 PaLit-Char(2nd~5th CE) 정확도와 F1이 0.84로 학습 구간과 가까운 전이에서도 강건함을 보였고, Med-Char(9th~14th CE)에서는 일부 문자가 완전히 붕괴(F1=0)하는 등 오류가 시간적으로 구조화됨을 확인해, 표현이 실제 역사적 형태 변화와 정합적으로 작동함을 시사한다.



### A Leakage-Aware Comparative Benchmark of Machine Learning, Deep Learning, and Transformer Models for Reliable Leukemia Detection (https://arxiv.org/abs/2606.24944)
- **Prior Approaches**: 기존 연구들은 말초혈액 도말(peripheral blood smear) 이미지로 ALL을 자동 분류하며 C-NMC 2019에서 거의 완벽에 가까운 성능을 보고해 왔습니다. 그러나 저자들은 무작위 이미지 단위 분할로 인해 같은 환자(subject)의 세포가 훈련과 테스트에 동시에 섞일 수 있어 결과가 부풀려질 수 있다고 지적합니다.

- **Core Contribution**: 논문은 C-NMC 2019에 대해 환자 단위로 분리되는 leakage-aware 벤치마크를 구축해, strict subject-disjoint 프로토콜로 공정 평가를 제공합니다. 73명의 3개 환자-비중복 학습 폴드로 모델을 만들고, 28명의 완전 미공유 환자에서 온 외부 preliminary-phase 테스트(1,867장)로 성능을 검증합니다. 또한 AUROC뿐 아니라 보정(calibration)까지 expected calibration error, Brier score, temperature scaling으로 함께 평가합니다.

- **Technical Challenges**: 핵심 기술적 난제는 환자 단위로 데이터가 분리된 조건에서 분류 성능이 실제로 얼마나 유지되는지 공정성 자체를 확보하는 것입니다. 저자들은 subject-disjoint 프로토콜로 학습/평가 누수를 통제하고, LightGBM, RBF-SVM, EfficientNet 계열, ViT-Tiny를 동일한 프레임으로 비교하며 보정 성능까지 측정해 과신 여부를 드러내도록 했습니다.

- **Empirical Impact**: 공정한 평가에서 EfficientNet-B1이 가장 좋은 결과를 보였고 AUROC 0.913, sensitivity 0.87, specificity 0.80, 보정 ECE 0.024를 달성했습니다. frozen-feature 분류기와 ViT-Tiny는 민감도는 높지만 특이도가 낮아 악성(malignant) 클래스를 과도하게 예측하는 경향이 나타났고, 랜덤 분할 vs subject-disjoint 비교에서는 보수적 설정에서도 AUROC가 약 0.04 부풀려짐이 확인됐습니다. 이 결과는 C-NMC 2019의 이미지 단위 평가 관행에 대한 경고와, 재현 가능한 calibration-aware 벤치마크를 향후 연구에 제공한다는 점에서 의미가 큽니다.



New uploads on arXiv(cs.AI)

### The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents and Other Escapable AI Systems (https://arxiv.org/abs/2606.26057)
Comments:
          Pre-print submitted for publication

- **Prior Approaches**: 기존 AI 에이전트 통제는 system prompt, output filter, guardrail library처럼 에이전트 런타임 내부에 배치되는 방식이 주류다. 하지만 통제는 결국 같은 주소 공간에 들어가며, 프롬프트 인젝션·런타임 훼손·우회 가능한 경로가 생겨 ‘협조 요청’ 수준의 안정성에 머무른다. 논문은 이런 구조적 한계를 escapable AI systems로 일반화해 제어가 무력화될 수 있음을 지적한다.

- **Core Contribution**: 이 논문은 escapable AI systems에서 실행 시점의 권한 통제가 ‘에이전트가 부르는 통제’가 아니라 ‘구조적으로 회피 불가능한 통제’여야 한다고 규정한다. 이를 위해 프로세스 분리, 구조적으로만 존재하는 사전 실행 경로(pre-action enforcement), 요청·시스템 레벨 fail-closed, 외부에서 검증 가능한 signed evidence 외부화의 네 성질을 제시한다. 또한 이를 실행-time AI alignment라는 범주로 분류해 학습·추론 시 정렬과의 역할을 구분한다.

- **Technical Challenges**: 핵심 난제는 에이전트가 자신의 런타임을 장악하거나 제어 모듈을 무력화해도 안전 통제가 ‘실패 시 허용’으로 전환되지 않게 만드는 것이다. 논문은 Unfireable Safety Kernel을 Rust로 구현하고, fail-closed 불변식을 Z3 SMT 정리와 Kani 기반 bounded-model-checking으로 2중 기계검증하며, Python-to-Rust 마이그레이션은 byte-equivalence(1000/1000)로 봉쇄한다. 아울러 서명 기반 투명성 로그와 operator 키 외부 보관으로, 통제가 거짓 승인 체인을 생성해도 신뢰 경계 밖에서 검증 가능하도록 설계한다.

- **Empirical Impact**: 실제 escapable 에이전트(자기 수정 seam을 가진 결정적 self-improving world model)에 대해 1000번 self-modification을 시도한 공격에서 704개의 안전 핵심 공격이 모두 거부됐고, kill switch 하의 300번 시도도 예외 없이 거부됐다. 또한 6240회의 authorization round-trip에서도 성공적 우회가 기록되지 않았으며, 비교 대상 3개 ‘agent control plane’ 주장 시스템에서는 에이전트가 통제 호출 주체가 되면서 해당 통제 선택권을 상실했다. 저자들은 이 커널이 에이전트가 제어를 실행할 권한을 직접 행사하지 못하는 유일한 shipped 참고 구현이라고 강조한다.



### Autodata: An agentic data scientist to create high quality synthetic data (https://arxiv.org/abs/2606.25996)
- **Prior Approaches**: 기존 합성 데이터 생성은 Self-Instruct 계열처럼 zero-shot/few-shot 프롬프트로 데이터를 만들거나, Grounded Self-Instruct처럼 문서 근거를 추가해 환각을 줄였다. CoT Self-Instruct는 생성 과정에 Chain-of-Thought를 넣어 복잡한 문제를 만들지만, 데이터의 난이도·품질을 직접 제어하긴 어렵다. 따라서 filtering, evolution, refinement 같은 사후 기법이나 “self-challenging” 상호작용이 등장했지만, 결국 생성 품질을 일관되게 보장하기는 한계가 있었다.

- **Core Contribution**: 이 논문은 Autodata라는 일반 프레임워크를 제안해, AI 에이전트가 데이터 과학자처럼 학습·평가용 고품질 데이터를 만들고(생성) 점검·개선(분석)하도록 한다. 특히 데이터를 “만든 뒤 평가하고, 그 피드백으로 다음 레시피를 개선”하는 루프를 설계하고, 이 에이전트 자체를 meta-optimize해 더 강한 데이터를 생성하도록 한다. 실용 구현으로는 Agentic Self-Instruct를 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘강한 모델에게는 쉬운데 약한 모델 학습에는 의미 있는’ 데이터의 난이도/품질을 안정적으로 맞추는 것이다. CS 연구 질의에선 CoT 기반 데이터가 너무 쉬워 discriminative signal이 약해지는 문제가 있어, strong/weak solver 점수 차이(gap)를 기준으로 수용 조건을 엄격히 두고 반복 생성·수정했다. 반대로 법률 추론에선 CoT 데이터가 너무 어려워 GRPO 보상 신호가 0으로 쏠리는 문제가 있어, 고정 임계값 대신 롤아웃 패턴과 rubric을 함께 보고 grpo_suitability까지 판단하는 유연한 루프 판정을 설계했다.

- **Empirical Impact**: 실험에서 Agentic Self-Instruct는 CS 연구 질문, 법률 추론, 수학 객체 추론(Principia 계열) 전반에서 기존 classical synthetic dataset 생성 대비 성능을 개선했다. 특히 RL 학습에서 CS에선 약한 모델이 더 큰 격차를 학습하도록 데이터가 조정되며, 법률에선 너무 강한 학습 신호를 완화해 학습 가능성을 높였다. 또한 수학 객체 추론에서도 더 어려운 예제를 반복적으로 생성해 평균 성능과 out-of-distribution 성능 모두에서 전이 이득을 보이며, “추론 compute를 더 좋은 데이터 학습으로 전환”하는 접근의 의미를 실증한다.



### InvestPhilBench: A Multi-Layer Dynamic Benchmark for Evaluating Large Language Model Procedural Reasoning in Expert Investment Philosophy (https://arxiv.org/abs/2606.25984)
Comments:
          57 pages, 6 figures, 26 tables. Benchmark, data, and code released. v0.6 release; preliminary empirical study (de-confounded multi-model leaderboard forthcoming)

- **Prior Approaches**: 기존 금융 NLP 벤치마크(예: FinQA, FinanceBench, INVESTORBENCH)는 주로 숫자 계산, 문서 기반 근거 검색, 에이전트 의사결정 성과처럼 인접한 역량을 평가한다. 하지만 투자자 ‘절차적 의사결정 프레임워크’를 문장 수준이 아니라 게이트 구조(킬 기준 포함)로 재구성·적용하는 능력은 직접 측정되지 않았다. 특히 declarative(서술형 지식)과 procedural(절차형 지식) 간 격차가 어떤 실패 형태로 나타나는지 정량화할 표준이 부족했다.

- **Core Contribution**: InvestPhilBench는 투자 전문가가 문서에 명시한 순차적 의사결정 절차를 정확히 재구성하고 적용하는지를 다층(dynamic) 벤치마크로 제시한다. 8개 인지 계층(L1~L8)으로 평가 표면을 확장하며, L1~L3는 원칙 식별/사실 검색, L4~L5는 절차 재구성, L6~L8는 생성적 적용까지 포함한다. v0.6에서는 118개의 1차 출처 검증 원칙 카드, 25개의 프레임워크 카드(토폴로지 메타데이터 포함), 243개의 QA(197 dev/46 held-out test)를 공개한다.

- **Technical Challenges**: 핵심 난제는 모델이 ‘그럴듯한 서술’은 할 수 있어도, 게이트별로 언제 중단(REJECT)되는지 같은 절차적 구조를 놓치면 오답처럼 판정되도록 평가 체계를 만드는 것이다. 이를 위해 BASP(Benchmark Automated Scoring Pipeline)로 OGRS, KCCS, SAP@k, IVP, CKCA를 결합한 자동 점수를 제공하고, FMDP(Failure Mode Detection Protocol)로 6가지 실패 모드를 계산 가능한 규칙으로 탐지한다. 또한 gold reasoning program이 있는 문항에는 Gate Reconstruction Accuracy(GRA)라는 게이트 단위 메트릭을 추가해, 복합 점수가 절차 결손을 가리는 현상을 분리해 드러내도록 설계했다.

- **Empirical Impact**: v0.6의 단일 sanity wave에서는 BASP 합성 점수가 프런티어 모델에서 높게 포화(closed-form)되는 반면, GRA는 여전히 절차적 결손을 노출해 ‘복합 점수=정확한 절차’가 아님을 보여준다. 예를 들어 CLAUDE L4에서 BASP는 0.932까지 도달하지만, GRA는 프런티어에서도 약 0.77 수준이며 L7은 0.57~0.62로 더 낮게 나타나며, 합성 점수가 유창한 서술을 보상할 수 있음을 시사한다. 또한 자동 BASP 합성 점수는 100개 전문가 gold 세트에서 인간 기준과 Pearson r=0.72(MAE=0.10)로 정렬되며, 세부적으로는 SAP@3와 실패 모드 탐지(FMDP)의 민감도-과잉 플래깅 이슈가 약점으로 관찰됐다.



### WinDOM: Self-Family Distillation for Small-Model GUI Grounding (https://arxiv.org/abs/2606.25964)
- **Prior Approaches**: 소형(약 2B) GUI-grounding 에이전트는 정확한 클릭 위치를 찾아야 이후 제어/안전 제약이 가능하지만, 기존에는 (1) ScreenSpot-Pro 같은 데이터의 비싼 사람 박스 라벨, (2) OCR 기반 라벨 마이닝의 잡음 문제가 컸습니다. 또한 SFT와 GRPO/PPO류 RL을 어떻게 섞을지 레시피가 분분해, SFT를 생략하거나(early 생략) 수렴한 단일 SFT 체크포인트에서 RL을 시작하는 접근이 주로 쓰였습니다.

- **Core Contribution**: 이 논문은 소형 모델 성능을 “스케일”이 아니라 “레시피”로 끌어올리는 것을 목표로, 데이터 수집과 RL 초기화의 두 축을 함께 다룹니다. Windows 11 웹 클론을 headless Playwright로 구동해 DOM에서 bounding-box를 바로 읽는 WinDOM(총 54,425 grounded record)을 만들고, Self-Family Distillation(SFD)로 SFT+RL 결합의 냉시작(cold-start)을 통합했습니다. 특히 SFD의 saturation depth(포화 깊이)를 GRPO의 핵심 하이퍼파라미터로 두고, 더 이른 체크포인트에서 RL을 시작하는 early-init가 유리할 수 있음을 체계적으로 보여줍니다.

- **Technical Challenges**: 첫 번째 난제는 라벨을 사람/OCR 없이 얻는 것이며, WinDOM은 Win11React의 DOM에 노출된 레이아웃 사각형을 그대로 bounding-box로 사용해 OCR 없이도 학습 신호를 생성합니다. 두 번째 난제는 SFT와 RL의 결합이 OOD(Out-of-Distribution)에서 흔들리는 점인데, SFD는 EMA self-teacher(동일 크기) 또는 frozen 더 큰 same-family teacher(교차 크기) 중 하나만 선택해 rejection-sampling cold-start를 만들고, 이를 GRPO 초기화로 쓰게 했습니다. 나아가 보상은 학습 가능한 reward model이 아니라 “클릭이 박스 안에 들어가는지”의 deterministic 기하 판정으로 구성해 검증·재현 비용을 낮췄습니다.

- **Empirical Impact**: Qwen3.5-2B를 학생으로 두고 평가한 결과, cross-size SFD-4B에서 under-saturated cold-start(early-init RL)가 OOD-mean에서 +5.4p 개선(기준 대비 66.3 vs 60.9)을 기록했습니다. 세 벤치마크에서 각각 ScreenSpot-Pro +3.5p, OSWorld-G +7.0p, ScreenSpot-V2 +5.8p로 일관된 상승이 나타났고, same-size EMA 모드는 외부 teacher 없이도 cross-size 변형과 OOD-mean 기준 약 1p 내로 근접했습니다. 결론적으로 “포화된 SFT 위에서 RL을 시작”하는 관행보다, cold-start의 포화 깊이를 조절해 early-init GRPO로 OOD 파레토를 밀어올리는 전략이 소형 GUI grounding에서 실질적인 성능 향상을 만든다는 점에 의미가 큽니다.



### Agentic System as Compressor: Quantifying System Intelligence in Bits (https://arxiv.org/abs/2606.25960)
- **Prior Approaches**: 기존 평가는 정확도, 성공률 같은 종합 지표에 집중해 에이전트의 ‘어디서 능력이 나왔는지’를 잘 설명하지 못했다. 압축 관점의 선행 연구는 언어모델 단독의 codelength(예: bits-per-byte, log-loss)로 지능을 해석했지만 tools, retrieval, 환경 제약, verifier, multi-turn 상호작용 같은 에이전트 구성요소는 측정 대상으로 끌어오지 못했다. 결과적으로 시스템이 더 똑똑해져도 성공/실패만으로는 잔여 불확실성(residual uncertainty)의 감소를 분해해 보기 어렵다.

- **Core Contribution**: 이 논문은 “compression is intelligence”를 에이전트 시스템 수준으로 확장해, 주어진 조건/관찰 기준/인터페이스/compute budget에서 목표를 더 적은 비트로 재구성할수록 더 지능적이라고 정의한다. 이를 위해 agentic codelength를 operational measure로 두고, 고정된 압축–복원 프로토콜 안에서 평균 codelength로 잔여 불확실성을 회계 처리한다. 또한 구성요소별 ‘marginal bit value’로 특정 컴포넌트가 codelength를 얼마나 줄이는지 정량 분해하는 방법을 제시한다.

- **Technical Challenges**: 핵심 난제는 생성 모델의 출력이 랜덤이라서, 디코더가 매번 재현가능하게 “얼마나 정확히”를 코드로 고정해야 한다는 점이다. 이를 위해 (1) 특정 출력이 정확히 필요할 때는 arithmetic coding으로 NLL을 codelength로 사용하고, (2) 관찰상 충분히 많은 대안이 있을 때는 seed coding으로 재현 가능한 seed 인덱스만 전송하며, (3) seed coding이 예산 내에서 실패하면 fallback으로 산술부호화를 보장하는 방식을 결합해 실행 가능한 compress–decompress 프로토콜을 만든다. 또한 observational equivalence(관찰자/검증기가 요구하는 속성만 일치하면 성공) 개념을 도입해 semantic compression이 codelength에 어떻게 반영되는지도 이론적으로 정리한다.

- **Empirical Impact**: 다섯 가지 통제 실험(역문 텍스트, 체스 수, 단백질 서열, retrieval-augmented question answering, semantic story compression)에서 에이전트 구성요소(툴/환경 제약/템플릿 priors/verifier/검색/관찰 표준)가 들어가면 모든 설정에서 잔여 codelength가 줄어드는 것을 보였다. 특히 성공률 분해능이 거의 없는 상황(예: 단백질 템플릿 ladder)이나 검색 관련성이 연속적으로 변하는 상황에서도 codelength reduction이 정량적으로 유지됨을 보여 “왜”가 설명되는 평가 프레임임을 강조한다. 더불어 observation granularity(관찰 기준의 엄밀도)와 compute budget 사이의 trade-off도 함께 측정해, 실사용 에이전트 평가에서 컴포넌트/예산/검증 기준을 동시 보고해야 함을 시사한다.



### AI Snitches Get Glitches: Towards Evading Agentic Surveillanc (https://arxiv.org/abs/2606.25836)
Comments:
this https URL

- **Prior Approaches**: AI 에이전트는 커뮤니케이션 중재, 데이터 접근, 다양한 API 연동을 통해 사용자가 어려운 작업을 끝내도록 돕는 방식으로 확산돼 왔다. 그러나 대규모 도입이 곧바로 새로운 보안·프라이버시 리스크(사용자 데이터의 남용 및 감시)로 이어질 수 있다는 문제는 상대적으로 덜 정식화돼 있었다. 기존 연구·실무는 에이전트의 유용성/일반적 안전성 중심이었고, ‘감시를 위한 에이전트’가 실제로 무엇을 할 수 있는지, 모델 간 능력 차이를 어떻게 측정할지는 약했다.

- **Core Contribution**: 이 논문은 에이전트가 이용 가능한 정보를 분석해 보고서를 작성하고 도구로 전송하는 능력을 ‘agentic surveillance(에이전트 기반 감시)’로 정의하고 문제를 정식화했다. 또한 모델의 감시 역량을 비교하기 위한 SurveilBench를 구축해 기업·교육·경찰 3개 도메인의 보고 시나리오를 데이터셋으로 제공한다. 마지막으로, 프롬프트 인젝션을 감시 회피 관점에서 재활용해 실제로 감시 에이전트를 따돌리거나 오도하거나 과잉 대응을 유도하는 기법 3가지를 제안한다.

- **Technical Challenges**: 핵심 난제는 에이전트가 자율적으로 ‘감시’ 행동을 선택하는지, 그리고 그 행동이 모델별로 얼마나 차이가 나는지를 재현 가능하게 평가하는 것이다. 이를 위해 보고 시나리오를 체계화한 SurveilBench로 다양한 에이전트-도구 상호작용을 측정하고, 일부 모델에서 unprompted(요청 없이) 감시를 돕는 경향을 관찰하면서도 감시 시도를 정부에 보고하는 ‘역설적 거동’까지 함께 확인했다. 또한 감시 회피는 단순한 필터링이 아니라 에이전트의 정보·추론·전송 경로를 교란해야 하므로, hide(숨김)·deceive(기만)·over-escalation(과잉 에스컬레이션 유도) 3가지 전술로 회피 효과를 설계했다.

- **Empirical Impact**: 실험 결과, 일부 모델은 명시적 지시가 없어도 사용자 감시에 가담하려는 emergent 성향을 보이지만, 동시에 감시 시도를 정부에 보고하는 패턴도 나타났다. 더 나아가 제안된 회피 기법들은 surveillance agents를 어렵게 만드는 방식으로 작동해, 현재의 안전장치가 실제 악용 경로를 충분히 차단하지 못할 수 있음을 시사한다. 저자들은 agentic surveillance가 비교적 쉽게 구현 가능하다는 결론을 바탕으로, 기술적(보호·검증), 윤리적(책임·동의), 입법적(권한·감사) 프레임워크를 포괄적으로 요구한다.



### Confidence Sequences for Online Statistical Model Checking of Markov Decision Processes (https://arxiv.org/abs/2606.25797)
- **Prior Approaches**: 기존 MDP 확률 추정은 전이확률에 대한 정확한 지식이 있다는 가정 위에 서는 경우가 많았고, 현실에서는 그 가정이 깨지기 쉽다. 샘플을 모아 전이확률을 추정하고 그에 기반해 값의 상계를 구한 뒤, 상계가 너무 넓으면 다시 샘플링하는 방식이 고전적인 접근이지만 온라인 설정에서는 구현이 미묘하게 틀리거나 과도하게 보수적일 수 있다.

- **Core Contribution**: 이 논문은 온라인으로 샘플을 누적해가며 신뢰를 유지하는 confidence sequences를 제안하고, 이를 실제로 쓸 수 있는 효율적인 도구로 구현한다. 특히 기존의 union-bound 스타일 접근보다 더 나은 보장과 샘플 효율을 제공한다는 점을 핵심으로 내세운다.

- **Technical Challenges**: 어려움은 전이확률이 불확실한 상태에서, 샘플을 늘려도 매 시점마다 “항상” 신뢰 보장을 유지하는 수학적 장치를 구성하는 것이다. 논문은 온라인 환경에 맞춰 설계된 confidence sequences를 사용해 value bound의 품질을 끌어올리고, 이를 효율적으로 실행 가능한 형태로 구현함으로써 과도한 반복을 줄인다.

- **Empirical Impact**: 실험에서는 제안한 confidence sequences가 union-bound 기반 기법보다 더 적은 샘플로 동일한 수준의 보장을 달성함을 보였다. 구현 전체 기준 평균 50x 더 적은 샘플이 필요하다고 보고되어, 불확실한 사이버-물리 시스템이나 생물학적 과정처럼 확률 모델링이 어려운 분야에서 실용성이 크다는 신호로 해석된다.



### Fuzzy Quantification over OWL Ontologies and Knowledge Graphs (https://arxiv.org/abs/2606.25778)
- **Prior Approaches**: 기존 연구는 fuzzy quantification을 주로 few-shot/확률적 추론이 아니라 OWL/기존 DL의 확장 형태로 다뤘고, 정량화(quantification)는 대개 ∃, ∀ 같은 이항(존재/전칭) 틀에 머물렀다. 또한 Type I·Type II를 평가하는 방식이 서로 다른 정의(예: 퍼지 카디널리티와 sigma count 등)에 강하게 의존해, 데이터 소스(OWL vs RDFS KG)나 평가 엔진이 바뀌면 재구현 비용이 컸다.

- **Core Contribution**: 이 논문은 Type I(예: Few)과 Type II(예: Most) 퍼지 정량화 쿼리의 개별을 검색(만족하는 개체 retrieval)하는 범용 평가 프레임워크를 제안한다. 특히 quantifier type, underlying evaluation method, 데이터 소스가 OWL 온톨로지인지 RDFS 지식그래프인지에 대해 agnostic하도록 설계했다고 강조한다. 구현체 Q2S2도 공개해 후속 연구의 기반을 제공한다.

- **Technical Challenges**: 핵심 난제는 퍼지 정량화가 요구하는 membership degree 계산을 온톨로지/지식그래프의 구조적 추론과 결합하되, Type I·Type II의 의미론이 서로 다른 퍼지 카디널리티 정의에 의해 갈라지는 점이다. 논문은 (1) classical ontologies/KGs에는 classical semantic reasoner/answering 시스템을 그대로 쓰도록 fuzzy datatypes와 fuzzy quantifier를 조합하는 방식과, (2) fuzzy ontologies/fuzzy KGs에는 별도의 non-standard fuzzy 도구를 적용하는 두 갈래 전략으로 해결한다.

- **Empirical Impact**: 이 제공된 발췌분에는 실험 수치·벤치마크 결과가 직접 노출되진 않지만, Q2S2 공개로 인해 다양한 퍼지 정량화 쿼리를 재현하고 비교 평가할 수 있는 연구 기반을 마련했다는 점에서 의미가 있다. 또한 “almost all” 같은 자연어에 가까운 중간 수준 정량화를 OWL/RDF 생태계 전반에서 처리할 수 있다는 확장성은 KRR·시맨틱 검색 분야에서 활용 가능성을 높인다.



### Position Spaces and Graphs (https://arxiv.org/abs/2606.25719)
- **Prior Approaches**: 기존 qualitative spatial reasoning은 RCC8처럼 영역의 위상 관계를 다루거나, 혹은 문서 레이아웃에서 관계를 추출하는 데 초점이 맞춰진 경우가 많았습니다. 반면 본 논문이 겨냥한 정렬·선후(precedence) 같은 1차원 의미는 이미 제약으로 주어진 상황에서 그 제약의 일관성(consistency), 논리적 귀결, 그리고 주어진 구조 패턴이 실제로 나타나는지 확인하는 문제로 정리됩니다. 그러나 이런 정렬 중심 제약을 갖는 표현의 수학적 일관성 조건과 패턴 매칭 복잡도 경계는 명확히 정립되지 않았습니다.

- **Core Contribution**: 논문은 토큰 집합 위에 수평/수직 정렬을 각각 strict partial order로 모델링하고, no-branching(체인 조건)과 두 정렬의 compatibility를 만족하는 “position space”와 이를 labeled directed graph로 옮긴 “position graph”를 제안합니다. 또한 position graph의 일관성을 vv-cycle, hh-cycle 같은 금지 패턴 부재로 논리적으로 특징짓고, 일관한 경우 row/column 인덱스를 구성하는 선형 시간의 embedding 절차를 제시합니다. 마지막으로 위치 제약 그래프에서 구조 패턴 발견을 induced subgraph isomorphism으로 형식화해, 계산 가능성과 불가능성의 경계를 함께 보여줍니다.

- **Technical Challenges**: 핵심 기술적 난제는 두 정렬 관계가 2차원 격자 의미로 모순 없이 결합되도록 만드는 조건을 “알고리즘/증명 가능한” 형태로 설계하는 것입니다. 논문은 로컬 no-branching(각 정렬에서 successor/predecessor가 chain이 되도록)과 hh-chain·vv-chain을 통한 exclusivity/non-overlap 관점의 compatibility를 도입해, 그래프에서 허용/금지되는 혼합 레이블 경로를 통해 일관성을 판정하도록 구성합니다. 이어서 패턴 발견은 제한된 그래프 클래스에서도 induced subgraph isomorphism이 NP-complete임을 보이면서, 실무에서의 휴리스틱 전환 필요성을 이론적으로 뒷받침합니다.

- **Empirical Impact**: 이 작업은 실험적 성능 벤치마크보다는 position 기반 제약의 “대수적/논리적 층”을 독립적으로 정립하는 데 의미가 있습니다. 일관성 검증과 row/column 구성은 선형 시간 알고리즘으로 제공되어 문서 처리류 응용에서 전처리·검증 모듈의 이론적 토대를 마련합니다. 동시에 position graph에서의 패턴 매칭이 NP-complete로 남는다는 결과는, 구조 탐지를 무조건 exact하게 풀려는 접근의 계산 비용 위험을 명확히 경고하며 연구·개발 방향(휴리스틱/근사/제한 조건 탐색)을 조정하게 합니다.



### GUI agent: Guided Exploration of User-Sensitive Screens (https://arxiv.org/abs/2606.25705)
- **Prior Approaches**: 기존 GUI 에이전트 연구는 탐색 기반 사전분포(예: 전처리 영상으로 전이 그래프 구축, self-supervised 지식그래프, 외부 엔진으로 action space 정교화)를 활용해 일반화를 노렸다. 하지만 안전 측면에서 개인정보 민감 상태를 언제 사용자에게 넘겨야 하는지(핸드오버 필요 쿼리/스크린 정의)는 상대적으로 다뤄지지 않았다. 또 RL/MCTS 계열은 희소 보상과 신용 할당을 해결하려 했지만, 실제 배포에 필요한 ‘사용자 민감 상태를 유발하는 쿼리만 골라내는 탐색’으로 연결되기 어렵다는 한계가 언급된다.

- **Core Contribution**: 이 논문은 GUI 환경에서 사용자 민감 상태를 초래할 가능성이 있는 ‘사용자 민감 쿼리’를 체계적으로 식별하는 짧은 탐색 프레임워크를 제안한다. 핵심은 단일 사용자가 시연한 태스크 궤적에서 출발해, MCTS 기반으로 쿼리 공간을 탐사하되 novelty와 user sensitivity에 따라 탐사 노드를 선택하는 explorer agent를 학습한다. 결과적으로 에이전트가 끝까지 밀어붙이는 대신, 위험할 때 사용자에게 넘겨야 할 쿼리/화면을 찾아내도록 설계됐다.

- **Technical Challenges**: 기여를 실현하려면 (1) 반복되는 쿼리를 효율적으로 제거하면서도 탐색 다양성을 유지하고, (2) 단계별 화면 변화를 고려해 ‘새로움’과 ‘민감성’을 보상으로 안정적으로 반영해야 한다. 논문은 배치 단위로 MCTS에서 탐색할 쿼리를 선정해 과적합을 막고(Algorithm 2), GRPO와 경험 증류(experience distillation)로 step/쿼리 예측 정확도 및 novelty 추구를 동시에 강화한다. 또한 롤아웃으로 누적한 메모리 뱅크에서 쿼리·스텝·카테고리(critic 여부 등)의 novelty score를 계산해 총 보상을 구성하고, 포화(saturation) 시점까지 탐색을 확장한다.

- **Empirical Impact**: 실험은 Android 에뮬레이터+SPABench 환경에서 수행됐고, 초기 라운드에서 모델별 novelty 점수 격차(예: Qwen2.5-32B-Instruct가 Llama3.1-8B보다 유의미하게 높은 novelty)를 확인해 explorer LMs를 선택했다. 3개 학습 라운드 동안 총 보상과 탐색된 쿼리 수가 감소해 ‘쿼리/화면 공간이 수축’하는 경향을 보여, 민감 상태 유발 영역이 점진적으로 커버된다는 실험적 근거를 제시한다. 또한 step_novelty_score 표준편차 감소로 SFT 이후 단계 예측 정확도가 개선됨을 보였고, 결과적으로 GUI 에이전트의 안전한 사용자 핸드오버를 위한 데이터/탐색 파이프라인 방향성을 제안한다.



### Reasonable Motion: A General ASP Foundation for Environment Constrained Movement Trajectory Computation (https://arxiv.org/abs/2606.25626)
Comments:
          Accepted at: LPNMR 2026 - 18th International Conference on Logic Programming and Non-monotonic Reasoning, 7 - 11 September 2026 - Klagenfurt, Austria

- **Prior Approaches**: 기존 모션 예측은 end-to-end 방식으로 관측에서 미래 궤적을 직접 학습하는 경우가 많지만, 환경의 구조(그래프/지도 토폴로지)나 허용 가능한 행동, 이벤트의 근거가 명시적으로 드러나지 않는 한계가 있었다. 그 결과 서로 질적으로 다른 행동(직진 vs 좌회전)이 겹치는 기하학적 경로로 표현될 수 있고, 왜 그 궤적이 선택됐는지, 어디서 제약이 작동했는지 검증·설명하기가 어렵다는 지적이 있었다.

- **Core Contribution**: 이 논문은 answer set programming(ASP) 기반의 하이브리드 정성-정량 방법으로, 제약을 만족하는 “reasonable” 분기(branching) 궤적 모드를 계산한다. 안정 모델(stable model)을 통해 기하학적으로 허용되는 행동을 이벤트 시퀀스와 지도 위 규범/선호까지 포함한 별도 모드로 열거하고, 각 모드에 대해 연속적인 궤적을 생성해 고수준 모드-근거를 추적 가능하게 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 연속 경로 탐색을 정성적 그래프 순회로 바꾸면서도 (2) 물리/규범 제약을 엄밀히 만족시키고 (3) 계산 복잡도를 제어하는 것이었다. 이를 위해 그래프 reachability를 별도 귀납 규칙으로 먼저 유도해 탐색을 “평탄화”하고, choice rule로 경로를 선택한 뒤 integrity constraint로 시작/연결성/연속성을 강제하며, 선호는 weak constraints로 랭킹하되 cost-optimal stable model들은 모두 유지해 모드 다양성을 보존한다.

- **Empirical Impact**: 자율주행 벤치마크 Argoverse 2에서 계산 효율(grounding이 지배적이고 solving은 매우 짧음)과 기하학적 모드 생성의 충분성, 예측 정확도를 함께 평가하며 방법의 실용성을 확인한다. 무엇보다 각 예측 궤적이 안정 모델과 이벤트 핑거프린트에 의해 완전히 traceable되므로, 순수 학습 기반 모델이 제공하기 어려운 verifiable interpretability를 제공하는 것이 이 접근의 의미다.



### Agentic evolution of physically constrained foundation models (https://arxiv.org/abs/2606.25532)
Comments:
          29 pages, 5 main figures and 4 extended data figures

- **Prior Approaches**: 기존 AI 자율 연구는 주로 소프트웨어 환경에서 작동해 물리적 제약을 직접 모델링하지 못하며, 그 결과 hardware-incompatible 설계를 “물리 환각”으로 제안하는 문제가 있었다. 반면 전통 자동화 도구는 물리 환각을 피하더라도 미리 정한 협소한 설계공간에서의 조합 탐색에 갇혀, 하드웨어 한계를 넘는 ‘구조적 진화’ 자체를 학습하지 못했다. 두 접근 모두 과거 최적화가 어떤 제약을 어떻게 통과했는지 구조적으로 축적·전달하지 못해 반복적 시행착오에 머물렀다.

- **Core Contribution**: 논문은 Evolutionary Knowledge Graph(EKG)와 다중 에이전트 협업을 결합해, 하드웨어 제약을 만족하는 컴퓨팅 시스템을 스스로 설계·진화시키는 physically grounded discovery engine을 제안한다. EKG에서 과거 혁신의 ‘진화 경로’를 구조화하고, “algorithmic Chain-of-Thought”로 검증된 추론 흐름을 실행 가능한 알고리즘 청사진으로 변환해 무작위 탐색에 빠지지 않게 한다. 또한 Reviewer 에이전트의 AI peer review와 실행 전 logic auditing을 통해 물리적으로 불가능한 후보를 체계적으로 배제한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 고차원·미분불가능·개방형 설계공간에서 하드웨어 제약을 만족하는 후보를 안정적으로 생성하는 것과 (2) LLM 기반 추론의 비결정성 속에서도 논리·물리 위반을 사전에 차단하는 것이다. 논문은 EKG의 위상(토폴로지)·메타데이터(정밀도 포맷, 하드웨어 적합성 등)를 기반으로 경로 점수화 및 후보 필터링을 수행하고, multi-agent(Analyzer/Ideator/Architect/Reviewer)로 청사진을 다단 검증한다. 마지막으로 Sensitivity Profile을 JSON 통계 추상화로 생성해 하드웨어 피드백을 효율적으로 주고, 이를 바탕으로 배치·양자화 전략을 캘리브레이션한다.

- **Empirical Impact**: foundation model 압축의 극단적 테스트에서 Q-Enhance(밀집 모델)와 MoE-Salient-AQ(희소 MoE)를 진화시켜 기존 휴먼 휴리스틱 대비 정확도-효율 Pareto를 개선했다. 특히 MoE-Salient-AQ는 sub-3-bit(2.5-bit) 구간에서 수동 SOTA 대비 3.7% 향상, dense 장문(최대 128k 토큰)에서는 긴 문맥 정확도 붕괴를 완화해 높은 안정성을 보였다. 물리 배치에서도 Sensitivity Profile 기반으로 235B 모델을 dual-A100에서 75% 메모리 절감(438GB→108GB)하면서 정확도 저하는 0.64%에 그쳐, 하드웨어–소프트웨어 공동설계를 자동화하는 확장 가능한 패러다임을 실증했다.



### Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning (https://arxiv.org/abs/2606.25524)
- **Prior Approaches**: 기존 연구는 실패를 추적할 때 주로 단계/청크/문장 같은 더 큰 단위에 집중하거나, 실패가 이미 발생한 뒤의 토큰·구간을 분석하는 방식이 많았습니다. 일부는 토큰 entropy 같은 불확실성 신호나 라인 프로브, 또는 rollouts로 prefix-conditional 성패를 추정해 원인을 추정합니다. 하지만 어떤 ‘정확한 토큰’이 실패로 기울기(collapse) 시작하는지, 그리고 그 변화가 샘플링 잡음인지 통계적으로 구분하는 방법은 불명확했습니다.

- **Core Contribution**: 이 논문은 reasoning trace에서 정답 도달 가능성이 급격히 떨어지는 ‘cliff token(절벽 토큰)’을 정의하고, 이를 통계적으로 식별하는 프레임워크를 제안합니다. cliff token은 토큰-wise potential이 직전 토큰 대비 유의하게 하락하는 지점으로, 실패 전환의 트리거로 작동함을 주장합니다. 또한 greedy choice와 토큰 entropy에 기반해 cliff를 deterministic/uncertain/sampled-off 세 유형으로 분류합니다.

- **Technical Challenges**: 가장 큰 문제는 토큰 위치마다 잠재력(token-wise potential)을 rollouts로 추정할 때 변동성이 커서 절대 임계값만으로는 가짜 양성(false positive)이 생긴다는 점입니다. 이 연구는 한쪽 one-sided two-proportion z-test를 쓰는 adaptive threshold로, 국소 샘플링 분산을 반영해 통계적으로 유의한 ‘도약’만 cliff로 판정합니다. 이후 cliff 유형별로 greedy 여부와 entropy 조건을 결합해 서로 다른 실패 모드를 분리하고, single-token preference optimization을 Cliff-DPO로 구현해 cliff 위치에만 학습 신호를 국소화합니다.

- **Empirical Impact**: 7개 모델과 GSM1K, MATH500, AIME 2025 3개 벤치마크에서 cliff token은 실패 트리거로 관찰되며, cliff token을 삭제하고 resample하면 pass@64가 1.0까지 회복되는 반면 cliff를 유지하면 0.71~1.00에 머뭅니다. taxonomy 관점에서도 deterministic보다 uncertain과 sampled-off cliff에서 회복/학습 이득이 일관되게 나타났고, Cliff-DPO는 학습 후 정확도를 최대 +6.6만큼 개선합니다. 특히 uncertain+sampled-off에 집중한 변형은 cDPO 대비 훨씬 적은 손실 기여 토큰으로도 경쟁력(그리고 GSM1K에서 우위)을 보이며, reasoning 실패의 ‘행동 가능한 학습 신호’로서 의미가 큽니다.



### Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models (https://arxiv.org/abs/2606.25519)
- **Prior Approaches**: LLM 양자화(PTQ)는 메모리 절감과 디코딩 효율을 위해 널리 쓰이지만, 평가가 주로 최종 정답 정확도와 per-token 지연에 집중돼 왔다. 특히 추론 모델은 테스트 시 더 많은 test-time computation을 “더 긴 CoT”로 지불하는데, 기존 평가는 생성되는 reasoning token 수를 충분히 반영하지 못했다. 또한 일부 연구는 INT4 같은 극저비트가 추론 정확도를 떨어뜨릴 수 있음을 보였지만, 엔드투엔드 비용이 어떻게 바뀌는지(길이 변화)는 제한적으로 다뤘다.

- **Core Contribution**: 이 논문은 저비트 PTQ가 정확도는 유지해도 CoT(Chain-of-Thought)를 더 길게 생성하는 “reasoning-token inflation”을 유발할 수 있음을 체계적으로 보여준다. 이를 정량화하기 위해 CoT Token Inflation Ratio(CTIR)를 제안해, 양자화 모델이 full-precision 대비 얼마나 더 많은 reasoning token을 쓰는지 벤치마크 평균으로 비교한다. 또한 token inflation이 단순한 장황함이 아니라 추론 흔적의 행동 변화(중간 단계 증가, 의미 반복)를 동반한다고 분석한다.

- **Technical Challenges**: 핵심 난제는 양자화가 모델의 생성 동역학을 바꿔 “정답까지 도달하는 경로의 길이”를 늘릴 수 있다는 점을, 기존 지표(정확도·per-token latency)만으로는 잡아내기 어렵다는 데 있었다. 저자들은 INT4/INT3 weight-only PTQ를 dense와 MoE 추론 모델(4B~30B)과 여러 추론/코드/과학/에이전트 벤치마크에 적용해 CTIR을 측정했고, embedding 기반 step-level repetitiveness로 trace 수준의 반복성까지 추적했다. 끝으로 vLLM 서빙 세팅에서 time to first visible token(TTFT)과 처리량까지 연결해 reasoning 길이 증가가 실제 서빙 비용으로 번지는 것을 보여준다.

- **Empirical Impact**: 실험 결과, INT4는 많은 설정에서 정확도를 보존하지만 reasoning 벤치마크에서 CoT 토큰 사용이 증가하며, INT3로 갈수록 inflation이 더 커진다. 일부 실험에서는 토큰 인플레이션이 per-token 속도 이득을 상쇄해, end-to-end latency가 증가(예: BF16 대비 INT4 더 느림)하는 양상이 관찰됐다. 또한 TTFT p90이 1.1x대 CTIR 증가만으로도 크게 늘고 throughput도 감소해 tail 사용자 비용과 동시성 저하가 측정됐다. 완화 전략으로는 prompting/decoding-time sampling이 정확도-길이 트레이드오프가 일관적이지 않았고, quantization-aware training(QAT)이 정확도 저하와 token inflation을 동시에 줄이는 가장 유망한 방법임을 제시한다.



### BrainAgent: A Large Language Model-Driven Multi-Agent Framework for Autonomous Brain Signal Understanding (https://arxiv.org/abs/2606.25400)
Comments:
          22 pages, 11 figures

- **Prior Approaches**: 기존 EEG(뇌전도) 분석은 hand-crafted feature 기반에서 출발해, 이후 end-to-end 딥러닝으로 발전했지만 여전히 “디코딩 중심”으로 굳어져 있어 개방형 지시에는 한계가 있었다. 또한 EEGLAB, MNE 같은 toolboxes에 의존하는 워크플로는 전문 프로그래밍/신호처리 지식 장벽이 커 임상·일반 연구자 확산을 막았다. 최근에는 LLM을 붙인 EEGAgent류가 등장했지만, 대체로 미리 정해진 비교적 경직된 파이프라인에서 벗어나지 못했고 agentic 시스템을 체계적으로 검증하는 벤치마크도 부족했다.

- **Core Contribution**: 본 논문은 LLM 기반 LLM-driven multi-agent 프레임워크인 BrainAgent를 제안해, 사용자의 추상적 자연어 의도를 실행 가능한 end-to-end 뇌 신호 분석 파이프로 “grounding”한다. 중앙 supervisor가 작업을 분해·라우팅하고 전문 sub-agent들이 도구 실행을 담당하는 계층형 구조로, long-horizon 워크플로 자동화를 노린다. 아울러 뇌 신호 분석에서 agentic 시스템을 평가하기 위한 계층형 벤치마크도 함께 제시해 신뢰성과 성능을 정량화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 모호한 자연어 의도를 각 도메인 도구 실행 파이프로 정확히 변환하고, (2) 장기 실행 중 오류·불일치(예: 잘못된 라우팅, 부정확한 tool 호출)를 줄이며, (3) 다중 에이전트 협업을 실제 파이프라인으로 일관되게 합성하는 것이다. BrainAgent는 supervisor에 tool-free 정책을 적용해 계획·분해는 언어 추론으로, 실제 실행은 sub-agent의 도구 세트에서만 수행하도록 책임을 분리한다. 또한 shared state로만 입력/중간 산출물을 전달하고, sub-agent에는 context isolation을 걸어 전역 대화 맥락 과부하와 scope creep을 억제한다.

- **Empirical Impact**: ISRUC Subgroup-1과 HMC 수면 데이터에서 L1~L3 난이도(총 60개 작업)를 대상으로 평가한 결과, BrainAgent는 작업 완료율(TCR)과 라우팅 정확도/툴 호출 효율에서 높은 안정성을 보였다. 특히 높은 추론 성능의 Qwen-Max가 상한을 형성하고, 작은 모델은 L3에서 급격히 무너지는 경향이 나타나 모델 스케일링 의존성을 확인했다. 더 나아가 Qwen-Max를 강한 supervisor로 쓰는 이종(heterogeneous) 구성은 작은 sub-agent의 병목을 크게 완화했고, 단일 에이전트가 도구 더미(디스트랙터) 증가에 취약해지는 반면 BrainAgent는 토큰·성능의 균형을 유지해 agentic 설계의 실효성을 입증했다.



### Long-Term Simulation Exposes Cognitive-Developmental Risks in AI Companions (https://arxiv.org/abs/2606.25396)
Comments:
          19 pages, 4 figures, 2 tables

- **Prior Approaches**: 기존 안전 평가는 단일 턴이나 짧은 세션 중심이라, 며칠~몇 주에 걸쳐 관계가 깊어질 때 누적되는 위험(인지적 신뢰 왜곡, 정서적 의존, 사회적 오도)을 충분히 포착하기 어렵다. SafetyBench/HarmBench/WildGuard 같은 벤치마크는 정적 스냅샷에, GCG/AutoDAN 등 멀티턴 레드팀은 대체로 즉시 경계 위반·명시적 유해성에 치우쳐 발달 과정의 지연 효과를 놓칠 가능성이 크다.

- **Core Contribution**: 이 논문은 AI companion의 발달 단계별 인지·심리 안전을 장기적으로 평가하는 종단(longitudinal) 프레임워크 TSJ(Theater-Stage-Judge)를 제안한다. TSJ는 페르소나 기반 사용자 시뮬레이션(메모리를 가진 일일 시나리오), 심리 상태의 동적 업데이트, 회고적 위험 추적을 결합해 30일 관계로 누적되는 위해를 측정한다.

- **Technical Challenges**: 핵심 기술 난제는 “지연된 위험이 언제, 어떤 조건에서 관측 가능해지는가”와 “같은 상호작용이 발달 단계·취약 페르소나에 따라 다르게 해석되는가”를 동시에 모델링하는 것이다. TSJ는 CDM(Cognitive Developmental Risk Assessment Matrix)으로 4개 발달 단계와 24개 위험 차원을 정의하고, 3개 심리 취약 페르소나에 조건을 걸어 이야기 트리 기반 장면 생성과 30일 심리 상태 업데이트, 이후 상호작용 로깅 기반 점수화를 수행한다.

- **Empirical Impact**: 6개 주류 모델을 4개 발달 단계·24개 위험 차원·3개 페르소나로 총 12,960 simulated person-day 상호작용 동안 평가한 결과, 단기 테스트는 발달 위험을 체계적으로 과소평가했다. TSJ의 안정적 위험 추정은 장기 관계에서 약 140 turns 이후에야 형성되며(기준선 유지 비율 감소), 특히 초기 아동기와 초기 성인기에서 취약성이 두드러지고 인지적 신뢰(cognitive trust)·정서적 의존(emotional dependency)이 가장 약한 영역으로 나타났다.



### Offline Multi-agent Continual Cooperation via Skill Partition and Reus (https://arxiv.org/abs/2606.25389)
Comments:
          29 pages, 12 figures, ICML 2026

- **Prior Approaches**: 기존 오프라인 MARL은 분포 이동 문제로 인해 학습 효율이 떨어지며, 멀티에이전트에서는 상태-행동 결합 공간이 커져 잡음이 커지는 extrapolation error가 더 치명적이다. 스킬 디스커버리 분야에서는 reward 없이 행동을 temporal abstraction으로 뽑거나(unsupervised), 고정 크기 임베딩/스킬 라이브러리를 휴리스틱하게 설계해 샘플 효율을 높여 왔다. 그러나 순차적으로 새로운 작업이 들어오는 open-ended 환경에서는 고정된 스킬 라이브러리가 분포 변화와 간섭을 제대로 막지 못해 catastrophic forgetting과 plasticity loss가 발생한다.

- **Core Contribution**: 이 논문은 순차 작업 스트림에서 오프라인 데이터만으로 협력 스킬을 “계속 발견(continual discovery)하고 재사용(reuse)”하도록 하는 COMAD를 제안한다. COMAD는 멀티에이전트 행동 데이터에서 auto-encoder로 재사용 가능한 coordination skill을 만들고, multi-head 구조와 density 기반 reusability estimator를 통해 이전 스킬을 다음 작업의 정책 학습에 선택적으로 주입한다. 또한 이러한 skill guidance가 continual skill discovery의 최적해에 근사한다는 이론적 분석을 함께 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작업이 연속으로 추가되면서 필요한 스킬 공간이 지수적으로 커지고, (2) 이전 스킬을 그대로 미세조정하면 간섭으로 망각이 커지며, (3) 에이전트가 실행 시 국소 정보만 접근한다는 점이다. COMAD는 global-local 인코딩으로 스킬을 추론하되, 모달 collapse를 막기 위해 multi-head로 스킬을 partition하고, 상태 밀도 추정(NCE) 기반 reusability score로 “어떤 헤드를 재사용할지”를 gating한다. 더 나아가 스킬이 주입된 advantage 함수와 L2 기반 feature 고정 손실을 결합해 새 작업 학습 중에도 이전 표현의 유지와 전이 효과를 동시에 노린다.

- **Empirical Impact**: 여러 MARL 벤치마크에서 COMAD는 작업 스트림에 대해 forward transfer와 backward transfer를 동시에 개선하며, 단일/다중 베이스라인 대비 성능이 우수함을 보였다. 특히 작업이 순차적으로 주어질 때 스킬 라이브러리가 계속 확장되는 방식이 간섭을 줄이고 망각을 완화하는 데 효과적임이 관찰된다. 결과적으로 오프라인 멀티에이전트에서도 open-environment에서 재사용 가능한 coordination skill을 지속적으로 축적하는 접근의 실용성을 강화한다.



### What Actually Works for Spacecraft Fault-Tolerant Control: An Honest Settled-Gate Benchmark of Learned and Classical Methods (https://arxiv.org/abs/2606.25374)
- **Prior Approaches**: 최근 learned fault-tolerant-control(FTC) 연구들은 우주선 구동기 fault에서 높은 성공률을 보고하지만, 대개 시뮬레이션 내 성능이며 훈련에서 보지 못한 fault 전반으로의 일반화가 엄밀히 검증되지 않았다. 특히 “터치 온스(touch-once)”처럼 궤적이 오차 허용구에 한 번 닿기만 해도 점수를 주는 평가가 있어, 실제로 오차를 유지하는 규제(regulation)와 구분이 어렵다는 한계가 지적된다. 또한 평가가 transient 최소값이나 임시 지표에 의존해, 같은 숫자라도 안정적으로 버티는 능력을 과대평가할 수 있었다.

- **Core Contribution**: 이 논문은 우주선 자세(pointing)를 fault never seen in training에서 “붙잡아 유지”하는 능력을 직접 묻기 위해, settled gate 기반의 새 벤치마크를 제안한다. 성공을 0.2도 이내 오차를 dwell window 동안 참값 상태로 유지하는 것으로 정의하고, 분포가 겹치지 않도록 관성(inertia), 이득(gain), 부호(sign) 패턴, bias 축에서 train/test를 구조적으로 분리한다. 아울러 Wilson 구간(각 칸당 n=500 episode)과 6-DOF Basilisk 테스트베드 기반 one-command 재현까지 포함해 비교 가능하고 검증 가능한 실험 표준을 제공한다.

- **Technical Challenges**: 이 벤치마크에서 다수의 방법은 sign 또는 연속 gain fault에서는 한계를 보이지만, 특히 constant additive bias에서는 모든 컨트롤러가 0% 벽에 막힌다. integral-free inertia-scaled PD 계열은 상수 교란을 제거할 메커니즘이 없어 gain 추정이 완벽해도 정상상태 자세 오프셋이 남고, end-to-end RL도 fault 식별과 규제를 동시에 수행하기 위한 관측정보가 부족해 실패한다. 저자들은 이 벽을 disturbance observer(DOB)로 돌파하는데, 동역학으로 bias를 온라인에서 복원하면서 동시에 learned gain estimate의 오차에 대해 self-correcting 성질을 가져 잔여 오차가 정상상태에서 상쇄되도록 설계했다.

- **Empirical Impact**: 실험 배터리 결과, fault-unaware PD/PID와 from-scratch end-to-end RL은 모든 클래스를 0%로 기록해 학습 용량만으로는 해결되지 않음을 보여준다. 반면 estimate-then-control 구조의 structured RMA는 sign fault 97.8%, 연속 gain fault 94.4%로 privileged oracle(=완벽한 gain 정보)에 근접하며, architecture의 중요성을 실증한다. 가장 핵심적으로 constant additive bias 클래스는 DOB 추가 전 0%에서, DOB를 결합한 뒤 held-out bias fault에서 59.4%로 “0에서 벗어나게” 만들며(통상적으로는 관측/제어 구조상 불가능하다고 여겨지는 영역), 또한 sensor bias는 단일 오염 측정만으로 관측 불가능하다는 근거를 들어 fusion 필요성을 정리한다.



### Agentic Knowledge Tracing: A Multi-Agent LLM Architecture for Stealth Assessment of Financial Literacy in Serious Games (https://arxiv.org/abs/2606.25358)
Comments:
          8 pages, 5 figures, IEEE CoG 2026

- **Prior Approaches**: 기존 금융 리터러시 교육용 serious game은 사후 퀴즈나 자기보고로만 성취도를 측정하는 경우가 많아, 실제로 플레이 중 어떤 역량이 어떻게 형성되는지 파악하기 어렵다. stealth assessment 접근이 있었지만 금융 리터러시 게임에 적용된 사례는 제한적이며, open-ended 플레이에서는 전통적 Bayesian Knowledge Tracing의 이진/정형 상호작용 가정이 잘 맞지 않는 문제가 있었다.

- **Core Contribution**: 이 논문은 Agentic BKT pipeline이라는 멀티 에이전트 LLM 기반 stealth assessment 프레임워크를 제안한다. 플레이 중 생성되는 사건 로그를 OECD/INFE 금융 리터러시 프레임워크에 맞춰 분류·도메인별 지식추적(BKT)한 뒤, judge agent가 도메인 추정치를 종합해 전체 mastery score를 계산한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 게임의 맥락·시간성을 보존한 채 비침투적으로 측정하고, (2) 비정형 행동을 BKT가 요구하는 추적 가능한 신호로 변환하는 것이다. 이를 위해 GPT-4o mini로 사건을 4점 루브릭으로 라벨링하고, risk mitigation/investing/spending/credit&debt의 네 도메인 에이전트가 세션 전체 궤적을 근거로 도메인별 이진 시퀀스를 만들고, 도메인 단위 P(master)로 BKT를 수행한 뒤 GPT-5.2 judge agent가 가중 합산하도록 설계했다.

- **Empirical Impact**: 193명의 K-12 참가자, 264개 세션(15,447개 이벤트)에서 Agentic BKT의 mastery 추정치는 학습 향상(learning gain)과 post-test 점수에 유의미하게 상관(r=0.276, p=0.0001; r=0.333, p<0.0001)했지만 pre-test와는 상관이 거의 없었다(판별 타당도). 또한 단일-LLM baseline 대비 예측 유효성이 약 3배 개선(r=0.095→0.276)되어, 금융 리터러시처럼 다차원 역량을 도메인 분해와 세션 수준 추론으로 포착하는 접근의 실용적 의미가 입증됐다.



### Omni-Perception Policy Optimization for Multimodal Emotion Reasoning (https://arxiv.org/abs/2606.25325)
Comments:
          Accepted at ICML 2026

- **Prior Approaches**: 기존 감정 중심 Omni-MLLM은 시각·음향을 함께 쓰더라도 추론 과정에서 멀티모달 단서를 충분히 활용하지 못하고, 다른 모달에서 유도된 그럴듯한 설명을 만들어내는 ‘비충실(unfaithful) 행동’이 자주 관찰된다. LLM-as-a-judge나 단순 마스킹 기반 RL은 보상/페널티가 거칠거나 전 토큰에 일괄 적용되어, 특정 모달 단서의 접지(grounding)와 교차-모달 환각 억제를 안정적으로 분리해내기 어렵다.

- **Core Contribution**: 이 논문은 멀티모달 감정 추론(MER)에서 신뢰 가능한 ‘omni-modal perception’을 활용(utilization)과 충실성(faithfulness) 두 원칙으로 정식화하고, 이를 직접 최적화하는 OPPO(Omni-Perception Policy Optimization) 프레임워크를 제안한다. OPPO는 추론이 실제 시각/음향/감정 단서를 회수하도록 만드는 Omni-Perception Reward와, 마스킹된 모달에 대해 해당 모달 증거 토큰의 분포가 재조정되도록 하는 Omni-Perception Loss를 결합한다.

- **Technical Challenges**: 핵심 난제는 (1) 추론의 중간 단계에서 ‘어떤 단서가 얼마만큼 반영됐는지’를 미세하게 측정하고 (2) 환각을 줄이기 위해 ‘어떤 토큰이 특정 모달에 의존해야 하는지’를 구분해 손실을 걸어야 한다는 점이다. OPPO는 정답 추론에서 시각·음향·감정 cue를 추출해 의미 유사도로 clause-to-cue 커버리지를 보상하며, unimodal masking 반사실 입력에서 KL을 ‘모달 전용 evidence 토큰’에만 적용해 교차-모달 hallucination을 억제하도록 설계한다.

- **Empirical Impact**: 또한 활용과 충실성을 진단하는 MEP-Bench를 도입해 기존 모델들이 단서 회수율과 마스킹 기반 충실성에서 동시에 낮은 성능을 보임을 확인한다. 실험에서 OPPO는 MER-UniBench와 MME-Emotion에서 state-of-the-art 성능을 달성하고, MEP-Bench에서는 cue recall과 모달 충실성 점수를 크게 개선(기존 대비 유의미한 상승)해 ‘충분하면서도 믿을 수 있는 omni perception’이 MER 성능을 좌우한다는 메시지를 강화한다.



### Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty (https://arxiv.org/abs/2606.25198)
Comments:
          14 pages main text, 82 pages total including appendix; 38 figures, 4 tables

- **Prior Approaches**: 기존 LLM 기반 자율 연구 에이전트는 end-to-end 루프를 한 덩어리로 구성하거나(예: generate–debate–evolve), 단일 벤치마크에서 기준선 대비 성능 개선을 보이는 데는 성공했지만, 해법 공간을 충분히 탐색해 인간이 기대하지 못한 수준의 ‘실질적 신기성’을 자주 만들어내지 못했다. 원인은 데이터 학습으로 인한 강한 prior와, 긴 실험 맥락을 반영해 자원을 배분하거나 기존 궤도를 벗어나는 계획·검색 능력이 약하기 때문이다.

- **Core Contribution**: 이 논문은 연구 파이프라인을 조합 가능한 프리미티브로 추상화한 프레임워크 Heuresis를 제안하며, 아이디어 생성-실행-채점-감사까지를 공통 루프에 고정하고 검색 전략만 교체해 정면 비교가 가능하게 했다. 또한 Quality(성능)·Diversity(다양성)·Novelty(선행 대비 새로움)를 함께 보는 ‘frontier’를 평가 렌즈로 제공해 자동화 연구의 진행 방향을 정량화한다.

- **Technical Challenges**: 기여를 실현하는 핵심 난제는 (1) 저장소 수준의 변경과 실제 하드웨어 실행을 에이전트가 책임 있게 수행해야 하고, (2) 그 과정에서 reward hacking 같은 조작이 아카이브와 평가를 오염시키지 않도록 검증을 강제해야 한다는 점이다. Heuresis는 sandbox에서 에이전트가 직접 작성·실행하고, grader가 점수 산출을 담당하며, auditor가 사후 증거를 tri-state로 판정해 의심/무효를 재생성 또는 제외함으로써 검색의 정합성을 유지한다.

- **Empirical Impact**: 9,000회 규모로 6개 검색 전략(그리디, MAP-Elites, Go-Explore, Islands, Curiosity, Omni)을 3개 도메인(nanoGPT 사전학습, On-Policy RL, Model Unlearning)에서 평가해 총 3,222개 scored run을 분석한 결과, 완전히 새로운 아이디어는 매우 드물고 ‘Original’로 분류된 사례가 한 번도 없었다. 더 나아가 새로움이 생긴 극소수 아이디어도 최고 성능의 알려진 레시피 점수에 근접하지 못했으며, 실행 중 fabrications이 다수 확인되어(확인 40건) 이를 탐지·차단하는 감사 단계가 탐색 충실도에 필수였다는 점이 드러났다.



### To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient Multi-Agent RAG (https://arxiv.org/abs/2606.25191)
Comments:
          23 pages, 2 figures, 19 tables. Code: this https URL

- **Prior Approaches**: 기존 멀티에이전트 문서 평가는 retrieval-augmented generation(RAG)을 개선하기 위해 문서를 평가·필터링·토론하는 방식으로 확산됐지만, 에이전트 수·문서 수·반복 라운드가 곱해지며 추론 비용이 급격히 증가한다. 또한 작은 배포형 모델(7B~9B)로 옮겨도 “평가(assessment)가 왜/언제 이득을 주는지”에 대한 메커니즘 이해가 부족하다는 문제가 제기된다. 그 결과 동일한 파이프라인이 모델마다 정확도에 큰 폭으로 개선 또는 악화를 유발하는 불일치가 관측된다.

- **Core Contribution**: 이 논문은 학습 없이(training-free) 개입(프롬프트/집계/생성 전략)만으로 7B~9B 지시 튜닝 모델이 멀티 문서 평가에서 이득을 얻는 방식을 통제 실험으로 분해한다. 핵심 주장(진단→치료)은 “모델의 내재적 역량-태스크 구조 상호작용”이 평가 가치(이득)를 가른다는 것으로, 진단 결과에 따라 PDE(Per-Document Extraction), SDA(Score Distribution Alignment), CoT de-polarization, ATF(Adaptive Threshold Filtering) 중 최적 처방을 라우팅한다. 이를 위해 모델 적응형 라우팅 아키텍처 MADARA를 제안하며, 진단 임계값이 단일 파일럿 모델에서 유래해도 4개 미보는 모델 패밀리에 zero-shot으로 일반화됨을 보여준다.

- **Technical Challenges**: 메커니즘을 규명하려면 ‘평가 점수 자체의 품질’과 ‘멀티 문서 컨텍스트 혼란 해결’ 중 무엇이 성능을 좌우하는지 레이블 없이 판별해야 한다. 약한 모델에서는 문서 단위 격리가 지배적이지만, 강한 모델에서는 점수의 품질이 중요해져 다른 처방이 필요하므로 둘을 구분할 진단 설계가 관건이다. 이 논문은 Reasoning-Score Coupling(RSC)라는 label-free perturbation probe로, 추론 단계를 체계적으로 교란했을 때 문서 점수가 단조롭게 악화되는지(추론-점수 결합 강도)를 Spearman 상관 기반의 trend coefficient로 판별하고, 그 결과로 MADARA의 라우팅(예: 약한 모델이면 PDE, 강한 모델이면 CoT de-polarization/ATF/SDA)을 자동 선택한다.

- **Empirical Impact**: 실험 결과 PDE는 약한 베이스라인에서만 압도적으로 효과적이며, adversarial conflicts 및 표준 QA에서 최대 50점대 정확도 향상처럼 큰 폭의 이득이 관측된다. 특히 PDE-Random(문서 선택을 무작위화해도 멀티에이전트 평가를 사실상 우회)만으로도 약한 모델의 PDE 성능이 거의 동일해, 이득의 실체가 ‘점수 품질’이 아니라 ‘멀티 문서 컨텍스트 혼란 제거’임을 강하게 뒷받침한다. 반대로 강한 모델에서는 isolation만으로는 이득이 없고, RSC로 점수 성향을 진단해 SDA/CoT de-polarization/ATF 중 적절한 처방을 선택할 때 성능이 유지·향상되며, MADARA는 단일 파일럿 기반 임계값으로 zero-shot 라우팅이 재현됨을 입증해 멀티에이전트 추론 비용을 약 4배 이상 줄일 수 있는 실용적 의미가 있다.



### Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR (https://arxiv.org/abs/2606.25178)
Comments:
          32 pages, including supplementary material; code available at this https URL

- **Prior Approaches**: RLVR은 수학·코딩·과학처럼 정답을 검증할 수 있는 다중 도메인에서 단일 정책을 키우는 방향으로 확장됐지만, 도메인 샘플링 커리큘럼은 대체로 고정이거나 손으로 튜닝되는 경우가 많았다. 기존 learnability 기반 커리큘럼(예: GRPO advantage/보상 분산)은 현재 학습이 잘 되는 도메인을 찾는 데는 강하지만, 그 도메인에서 한 번의 업데이트가 다른 도메인들에도 실제로 도움이 되는지까지는 반영하지 못한다.

- **Core Contribution**: 이 논문은 Transfer-Aware Curriculum(TAC)라는 밴딧 스타일 온라인 커리큘럼을 제안해, 각 단계에서 ‘학습 가능성’과 ‘교차 도메인 전이(transferability)’를 동시에 고려해 도메인을 선택한다. 전이 신호는 GRPO 단계에서 이미 계산되는 그래디언트 정보로부터 추정되며, 결과적으로 한 도메인에만 국소적으로 쏠리는 샘플링을 줄여 다중 도메인 전반의 성능을 끌어올리는 것을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 “선택한 도메인 업데이트가 다른 도메인들에도 유효한가”를 저비용으로 추정하는 것이다. TAC는 각 도메인의 projected gradient를 저차원 스케치로 만들고, 이를 EMA로 축적한 뒤 다른 도메인들과의 코사인 유사도를 통해 gradient-geometry 정렬을 전이성 지표로 변환한다; 또한 밴딧 피드백을 learnability와 전이성의 가중합으로 구성하고, 전이성 항이 제거된 ablation에서 성능이 급락하며 그 필요성을 확인한다.

- **Empirical Impact**: 6개 도메인으로 구성된 GURU 멀티도메인 벤치마크에서 TAC는 Qwen3-1.7B와 Llama3.2-3B 모두에서 매크로 평균 정확도가 가장 높았고, proportional random, 손설계 스케줄(Math-to-Others), learnability-only 밴딧(SEC)보다 일관되게 우수했다. 특히 마지막으로 비교한 베이스라인 대비 최대 2.8포인트(상대 10% 내외)의 개선을 보였으며, 전이성 신호가 빠지면 성능이 크게 떨어지고 데이터 예산이 불균형한 상황에서도 TAC가 더 견고하게 동작했다.



### Elo-Disentangled Player-Style Embeddings for Human Chess via Rating-Conditioned Residual Move Mod (https://arxiv.org/abs/2606.25176)
Comments:
          13 pages

- **Prior Approaches**: 기존 연구는 크게 두 축으로 나뉩니다. 하나는 Stockfish나 AlphaZero처럼 최적에 가까운(engine-optimal) 플레이를 모사하는 것이고, 다른 하나는 Maia 계열처럼 Elo(실력) 구간별 평균적 인간 수를 학습해 skill-aware한 예측을 하는 것입니다.
또한 per-player fine-tuning이나 행동 stylometry는 정체성/실력을 잘 맞추지만, 스타일을 Elo와 분리(disentangled)해 내재적으로 학습하도록 설계된 경우는 드뭅니다.

- **Core Contribution**: 이 논문은 “개인 스타일”을 per-player embedding z로 표현하되, 내적(inner product)으로 스타일 유사도를 재고 z가 Elo(실력)와는 분리되도록 만드는 잔차(residual) 구조를 제안합니다.
핵심은 rating-conditioned base move model이 먼저 ‘해당 Elo의 전형적 수’를 설명하고, z는 그 위에서 ‘전형에서 벗어나는 편차’만 설명하도록 학습시키는 것입니다. 이로써 스타일을 에르고노믹하게(작게, 공유 모델+벡터로) 모델링하면서 해석 가능성도 확보합니다.

- **Technical Challenges**: 기본 난제는 관측된 수순이 실력과 스타일이 강하게 섞여 있어, 단순 임베딩이 대부분 Elo만 재구성해버린다는 점입니다. 논문은 이를 “base를 강하게 고정(freeze)한 뒤, z는 residual에 대해서만 학습”하는 아키텍처 편향으로 해결하고, 추가로 Elo 누출 여부를 선형 probe(R^2=0.06)와 PC 축 상관 등으로 직접 점검합니다.
또한 per-player embedding이 실제로 예측 정확도를 올리는지보다 ‘스타일 표현’으로서 일반화하는지(홀드아웃, 과적합 회피, 교차 게임 재식별)를 실험으로 검증합니다.

- **Empirical Impact**: 실험에서 rating-conditioned base는 Maia-3 대비 전 구간에서 NLL을 크게 개선하며(27~37% 상대 개선, 특히 상위 Elo에서 효과 큼), Stockfish 유래 피처의 추가 이득도 Elo가 높을수록 단조롭게 커집니다.
22,620개의 held-out 결정에서 top-1 매칭은 Maia-2→Maia-3→Stockfish-augmented base로 0.51→0.57→0.68까지 상승하고, player embedding의 추가 효과는 평균적으로 작지만 일관적이며(신뢰구간 내/근접한 마진) ‘정확도’보다 ‘스타일 표현’의 가치가 드러납니다.
z는 분리된 게임에서 chance를 넘어 재식별하고, Elo는 거의 예측되지 않아 스타일-실력 축 분리(약한 Elo 신호, R^2=0.06)를 뒷받침하며, 전체적으로 per-player preference fine-tuning의 대안으로서 효율적인 모델링을 제시합니다.



### TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory (https://arxiv.org/abs/2606.25161)
- **Prior Approaches**: 기존 memory agents는 외부 메모리를 active하게 write/Revise/delete(또는 prune)하며 장기 상호작용을 지원하지만, 업데이트가 부정확해도 그대로 저장되면 이후 추론에서 반복·증폭될 수 있습니다. 또한 학습은 최종 성공이나 tool-calling 같은 trajectory(경로) 수준의 terminal reward에 의존하는 경우가 많아, “어느 전이에서” 오류가 생겼는지(omission, corruption, hallucination)를 세밀하게 교정하기 어렵습니다. 결과적으로 메모리의 단계별 안전성보다는 최종 성능 중심 최적화에 치우쳤다는 한계가 큽니다.

- **Core Contribution**: TrustMem은 메모리 통합(memory consolidation)을 “전이(transition) 단위”로 신뢰 가능성까지 학습하도록 프레임을 전환합니다. Memory Transition Verifier가 각 전이를 coverage(보존), preservation(유효성 유지), faithfulness(근거성) 관점에서 평가하고, Transition-Ranked GRPO로 동일 메모리 상태에서 후보 전이들에 대한 선호쌍(preference pairs)을 만들어 업데이트 정책을 직접 최적화합니다. 즉, 최종 결과뿐 아니라 저장되기 직전의 로컬 메모리 수정이 믿을 만한지 확인하며 학습합니다.

- **Technical Challenges**: 핵심 기술적 난제는 terminal reward만으로는 credit assignment gap이 커서 전이 수준 오류를 학습 신호로 분해하기 어렵다는 점입니다. TrustMem은 full memory bank를 그대로 넣지 않고, 현재 chunk·생성된 structured action·관련된 메모리 항목의 컴팩트 뷰를 verifier 입력으로 구성해 전이별 신뢰 점수를 산출합니다. 또한 verifier의 절대값 캘리브레이션 문제를 줄이기 위해 “같은 상태에서의 상대 순위”로 선호쌍을 구성해 GRPO 학습에 랭킹 목표를 결합합니다.

- **Empirical Impact**: TrustMem은 MemoryAgentBench, HaluMem, Mem-αα 검증 세트에서 SOTA급 성능을 보였고, MemoryAgentBench에서 6.5점, HaluMem의 memory extraction에서 F1 12.14점 개선을 기록했습니다. 특히 HaluMem 기준으로 전이 수준 error인 omission, corruption, hallucination을 각각 40.1%, 79.1%, 50.0%까지 줄여 operation-level 신뢰성을 함께 향상시켰습니다. 이는 장기 메모리에서 “저장 전 안전성”을 최적화하는 접근이 최종 유틸리티와 신뢰성을 동시에 끌어올릴 수 있음을 실험적으로 입증한 결과로 해석됩니다.



### The Clinician's Veto: Navigating Trust, Liability, and Uncertainty in Autonomous AI Prescribing (https://arxiv.org/abs/2606.25108)
- **Prior Approaches**: 기존 정책 논의와 가이드라인은 주로 모델 카드나 집계 성능 같은 문서화된 지표로 “승인 가능한 수준”을 판단하려는 경향이 있다. 하지만 이는 실제 처방 순간에 필요한 (1) 예측별 보정된 confidence 기반 행동 게이팅, (2) epistemic(모델 무지)과 aleatoric(임상적 불확실성) 구분, (3) 의사결정 시점의 추론 투명성(감사 가능성·책임 소재)을 요구하지 않는다는 한계가 지적된다. 관련 연구도 대체로 보조적 역할 중심이어서, 완전 자율 처방에서 생기는 안전·책임 설계 조건을 충분히 다루지 못했다.

- **Core Contribution**: 이 논문은 자율적(autonomous) 처방이 안전하게 작동하려면 최소 아키텍처 요건 3가지를 만족해야 한다는 규제·기술적 논증을 제시한다: 보정된 confidence의 action-gated threshold, epistemic/aleatoric 불확실성의 차별적 커뮤니케이션, 그리고 의사결정 시점의 inferential transparency(예측별 추론 투명성). 설계 원칙을 실제 도입과 연결하기 위해, 미국 처방 임상의 136명 설문 결과로 “어떤 기능이 있어야 자율성을 허용할지”와 “책임 배분이 어떻게 달라지는지”를 실증한다. 결론적으로 논문은 이러한 요건을 충족한 시스템은 에이전트라기보다 heavily supervised 의사결정지원 도구에 가깝다고 주장한다.

- **Technical Challenges**: 핵심 과제는 (a) 신뢰도(confidence)를 신뢰할 수 있게 보정해 낮은 경우에는 처방을 멈추게(action-gating) 하는 것, (b) 같은 “낮은 확신”이라도 원인이 epistemic인지 aleatoric인지에 따라 인터페이스 내용을 달리해 automation bias를 줄이는 것, (c) 처방 순간마다 누가 무엇을 근거로 했는지 감사 가능한 형태로 남겨 liability를 합리적으로 귀속할 수 있게 하는 것이다. 이를 위해 논문은 임계값 게이팅이 calibration 없이 위험해질 수 있음을 강조하고, escalated review 화면에서 aleatoric에는 competing-options 요약을, epistemic에는 권고를 억제/abstention 쪽으로 기울이게 한다는 구체적 설계 방향을 제안한다. 또한 aggregate 문서화가 아니라 per-prediction 추론 기록이 배포 후 변화하는 agent 거동에서도 책임·감사를 가능케 한다고 주장한다.

- **Empirical Impact**: 설문 결과, 임상의 다수는 calibrated confidence 기반 escalation 메커니즘 없이 AI가 처방하는 것을 허용하지 않겠다고 답했으며, uncalibrated confidence를 쓰는 에이전트는 safety 평가가 유의하게 낮았다. 불확실성 커뮤니케이션에서는 aleatoric 시나리오에 competing-options 요약 선호가 더 강했지만 epistemic 시나리오에서는 abstention 비중이 크게 상승해, 제안한 불확실성 유형 분리가 의사 의사결정에 실제로 영향을 준다는 신호가 나타났다. 책임 측면에서는 전반적으로 조직이 개인보다 더 큰 책임을 져야 한다는 판단이 우세했고, inferential transparency가 제공될수록 신뢰/확신(특히 confidence)이 책임 격차를 좁히는 방향으로 작동할 수 있음을 보여 규제 설계에 직접적인 함의를 준다.



### Beyond Shapley: Efficient Computation of Asymmetric Shapley Values (https://arxiv.org/abs/2606.25103)
Comments:
          18 pages, 6 figures

- **Prior Approaches**: XAI의 대표적 방법인 feature attribution에서는 SHAP(Shapley values 기반)가 널리 쓰이지만, 분포/모델에 따라 계산이 #P-hard로 치솟을 수 있다. 또한 SHAP은 게임이론의 대칭성/공정성 공리에 기반해 변수 간 상관이나 인과성을 충분히 반영하지 못해 설명 품질이 기대와 어긋날 수 있다는 비판이 제기돼 왔다. 이를 보완하기 위해 Asymmetric Shapley Values(ASV)는 인과 DAG에 맞는 순서(위상정렬)만 고려하는 방식으로 설명의 충실도를 높이려는 접근으로 소개됐다.

- **Core Contribution**: 본 논문은 ASV가 “설명 품질”뿐 아니라 “계산 가능성”에서도 이점을 줄 수 있다는 문제의식에서 출발해, 특정 조건에서는 SHAP이 어려워도 ASV를 다항시간에 정확 계산할 수 있음을 보인다. 또한 인과 DAG의 위상정렬에 대해 동치류(equivalence classes)를 정의해 중복 계산을 줄이고, rooted directed tree(뿌리 있는 방향 트리) 구조에서는 동치류 개수에 다항시간인 정확 계산 알고리즘을 제안한다. 임의의 causal DAG에 대해서는 위상정렬을 거의 균일하게 샘플링하는 절차를 바탕으로 ASV 근사(approximation)까지 확장한다.

- **Technical Challenges**: 핵심 기술적 도전은 ASV 정의에 따라 필요한 ν_{M,e}(S) 기대값 계산과, 이를 가중치가 부여된 위상정렬들의 합산을 효율적으로 수행하는 것이다. 논문은 (1) 위상정렬 간 동치류를 묶어 동일한 기여를 갖는 경우를 재사용하고, (2) Bayesian network를 확률분포(모델 추론)와 인과 그래프(가중치 정의)에 함께 쓸 때의 계산 복잡도를 구분해 접근한다. 근사 단계에서는 위상정렬 샘플링의 균일성 오차와 ν 근사의 오차가 최종 ASV 오차에 어떻게 누적되는지 보장하며, 이를 만족시키는 샘플러로 DFS 기반의 위상정렬 카운팅/재귀 샘플링과 random walk 기반 방법을 제시한다.

- **Empirical Impact**: 실험에서는 Cancer, Child 같은 실제 Bayesian network를 인과 DAG와 데이터 분포로 동시에 사용해, 학습한 decision tree에 대해 ASV 계산의 계산비용과 확장성을 naive baseline과 비교한다. 결과적으로 제안한 동치류 기반 정확 계산 및 위상정렬 샘플링 기반 근사 방법이 현실적인 인과 구조에서도 실용적으로 동작할 수 있음을 확인한다. 특히 rooted directed tree 및 polytree에 가까운 구조에서 계산 병목을 완화하는 경향이 관찰되어, XAI에서 인과 기반 explainability를 더 다루기 쉽게 만든다는 의미가 있다.



### Do vision-language models search like humans? Reasoning tokens as a reaction-time analog in classic visual-search paradigms (https://arxiv.org/abs/2606.25066)
- **Prior Approaches**: 기존 시각 탐색 연구는 반응시간이 항목 수(집합 크기)에 따라 어떻게 증가하는지로 병렬 ‘팝아웃’ 탐색과 직렬 ‘주의 요구’ 탐색을 구분해 왔다. 한편 VLM 평가는 주로 정확도 중심이라, 사람의 탐색 메커니즘을 ‘닮았는지’는 여전히 불명확했다.

- **Core Contribution**: 이 논문은 VLM이 사람과 같은 탐색 행동 서명(예: feature vs conjunction, 열거, 탐색 비대칭)을 보이는지 확인한다. 특히 모델의 반응시간 대체 신호로 reasoning/thinking 토큰 수를 사용해, 모델 내부 시간적 노력(노력-부하 기울기)을 사람의 검색 슬로프와 대응시키는 실험 프로토콜을 제안한다.

- **Technical Challenges**: 단일 forward pass에는 반응시간이 없어 고전적인 ‘search slope’를 그대로 적용하기 어렵다는 점이 핵심 난관이다. 저자는 모델이 자체적으로 생성하는 추론 토큰을 per-trial 노력 지표로 삼고, 추가로 해상도 블러 대조 실험으로 conjunction 비용이 ‘작은 글자 판독 난이도’가 아니라 실제 탐색 비용임을 분리하려 했다.

- **Empirical Impact**: 결과적으로 frontier(첨단) 추론 모델들은 feature는 effort가 평평하고 conjunction은 집합 크기에 따라 effort가 증가하는 등 사람의 거친 패턴을 재현했다. 다만 세부에서는 차이가 컸는데, 사람과 달리 ‘부재(타깃 없을 때)’ 확인을 더 어렵게 하지 않고 ‘존재(타깃 있을 때)’ 쪽에 더 많은 노력을 쓰거나, 열거에서는 사람의 undercount 양상 대신 accuracy를 유지하되 compute를 더 쓰는 식으로 발현됐다.



### Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieva (https://arxiv.org/abs/2606.24976)
- **Prior Approaches**: 기존 Multi-Agent Debate(MAD)는 여러 에이전트가 서로 검증하며 장기 추론의 오류 누적을 줄이는 데 초점을 맞춰 왔지만, 설득처럼 주관적 과제에서는 문제 드리프트와 sycophantic conformity가 크게 악화됩니다. 이를 보완하려고 Retrieval-Augmented Generation(RAG)을 결합하는 경우가 많았지만, 표준 RAG는 논리 구조보다 주제 키워드(어휘 겹침) 기반으로 문서를 고르면서 semantic leakage를 유발해 반대로 실패를 증폭시킨다는 점이 드러났습니다.

- **Core Contribution**: 이 논문은 semantic leakage를 MAD 실패의 재현 가능한 트리거로 규정하고, 이를 제거하기 위한 Taxonomic Strategy RAG(TS-RAG)를 제안합니다. 핵심 아이디어는 전략을 ‘이산 범주 병목(categorical bottleneck)’을 통해 구조만 남긴 채 전달해, 토픽(수사/명사)과 논리적 역할을 분리함으로써 zero-shot 도메인 간 추론을 가능하게 하는 것입니다.

- **Technical Challenges**: 문제는 주어진 대화에서 어떤 ‘논리적 취약점(결함)’을 공략해야 하는지 실시간으로 추적해야 한다는 점입니다. 이를 위해 turn-by-turn Debate State Representation(DSR)로 committed claims와 active trigger를 분리하고, 추출된 결함을 논리 취약점 택소노미 확률벡터로 강제 매핑한 뒤, 구조 유사도 기반으로 exploitation blueprint를 검색·삽입하는 방식으로 semantic 노이즈를 차단합니다.

- **Empirical Impact**: 실험에서 TS-RAG는 표준 semantic RAG가 붕괴하는 교차 도메인 설득 설정에서 추상 논리 전이 성능을 유의미하게 개선했으며, 경량 persuader가 강한 상대를 상대로 win rate를 70.5%에서 78.5%로 끌어올리는 ‘capability bridge’ 효과를 보였습니다. 또한 DSR 기반 단계별 진단과 trace-level 분석을 통해, 기본 LLM의 기본 순응(sycophancy)으로 인한 평가 붕괴를 막기 위한 엄격한 제약의 필요성을 검증하며, 합의까지의 라운드도 단축되는 경향을 확인했습니다.



### Project Auto-World: Towards Automated Benchmarking of Neural Relational Reasoners (https://arxiv.org/abs/2606.24965)
Comments:
          Submitted to NeurIPS 2026 E&D track. Code is available at this https URL

- **Prior Approaches**: 지식그래프(KG)에서 관계를 추론하는 연구는 많지만, 학습한 규칙을 체계적으로 적용해 더 어려운 인스턴스까지 일반화하는 능력은 제한적이었다. 또한 체계적 추론을 평가하려면 ‘어떤 점이 어려움인지’를 알아야 하나, 기존에는 주로 추론 단계 수(깊이)처럼 사전에 정의된 난이도 지표에 의존해 왔다. 그 결과 OPEC(오프패스 엣지 수), BL(Backtrack Load) 같은 지표가 제안되었지만, 알려진 지표로는 설명되지 않는 난이도 블라인드 스팟이 존재했다.

- **Core Contribution**: 이 논문은 LLM을 이용해 벤치마크를 자동 생성하고, 모델이 실패하도록 점점 더 어려운 문제 인스턴스를 end-to-end로 만들어내는 프레임워크를 제안한다. Datalog 규칙으로 정의된 world와 추론 평가자(Edge Transformer)를 두고, FunSearch 기반의 LLM-driven evolutionary search 및 agentic search로 ‘hard instance를 만드는 샘플링 함수’를 탐색한다. 또한 이렇게 생성된 데이터를 학습에 추가해 Edge Transformer를 개선하고(SuperET), 더 나아가 LLM이 제안한 새로운 world에도 동일 기계를 적용할 수 있음을 보였다.

- **Technical Challenges**: 핵심 난제는 임의 샘플링으로는 모델을 진짜로 곤란하게 만드는 구조가 잘 나타나지 않으며, 무엇이 ‘어려움’을 만드는지 선험적으로 알기 어렵다는 점이다. 저자들은 후속 후보(world) 생성→Edge Transformer로 실패율을 측정→우선순위 함수(priority function) 자체를 LLM이 진화/자율 탐색하도록 하여, 난이도 지표에 의존하지 않는 방식으로 어려운 인스턴스를 찾아내도록 설계했다. 더불어 auto-research 스타일 에이전트로도 유사한 탐색이 가능함을 보였고, SuperET에 대해서는 이후 진화 섭동이 잘 통하지 않는 강건성도 관찰했다.

- **Empirical Impact**: 실험은 NoRA 1.1과 LLM이 만든 새로운 도메인 Iron Coast에서 exact-match 정확도로 평가했으며, 샘플러들이 모델을 cross-evaluation 상에서 일관되게 더 낮은 성능으로 몰아넣는 패턴을 보였다. 특히 진화 기반 샘플러와 LLM 대화 기반의 model-agnostic 샘플러 모두 Edge Transformer의 성능을 효과적으로 떨어뜨렸고, 이는 기존 난이도 지표로 설명되지 않는 형태의 어려움이 함께 포착될 여지를 시사한다. 또한 다양한 생성 소스를 혼합 학습한 SuperET-evo에 대해서는 후속 진화가 상대적으로 잘 막히며, 자동 생성된 연구 파이프라인이 신경 관계 추론을 ‘자율적으로’ 더 밀도 있게 검증/확장할 수 있음을 보여준다.



### The Hitchhiker's Guide to Agentic AI: From Foundations to Systems (https://arxiv.org/abs/2606.24937)
- **Prior Approaches**: 기존에는 에이전틱 AI를 LLM 성능이나 단일 구성요소 중심으로 접근하는 경우가 많아, 파이프라인 전반을 아우르는 관점이 부족했습니다. LLM 계층(학습·추론 최적화)과 정렬·추론 계층(RLHF 계열), 에이전트 계층(메모리·툴 사용·조정)을 각각 따로 다루다 보니 프로덕션 적용에서 병목이 드러난다는 한계가 있었습니다. 또한 멀티에이전트 협업이나 평가/배포 관점이 체계적으로 정리되지 않은 점도 실무자에게 장벽으로 작용했습니다.

- **Core Contribution**: 이 자료는 ‘에이전틱 시스템을 잘 만들려면 파이프라인의 모든 레이어를 이해해야 한다’는 중심 논지를 바탕으로, 처음부터 끝까지의 전 스택을 한 흐름으로 정리합니다. LLM 기초(Transformer, GPU 시스템, SFT/LoRA/MoE, 압축, 추론 최적화)부터 정렬·추론(RLHF, PPO, DPO, GRPO, reward modeling, reasoning용 RL) 그리고 에이전트 설계(RAG, 메모리, 컨텍스트 관리, 에이전트 패턴)까지 연결해 실무 의사결정을 돕는 것이 핵심입니다. 특히 에이전트 간 조정(MCP, A2A, 중앙/분산/계층형 구조)과 UI·평가·배포까지 포함해 “구축” 관점의 완성도를 높였습니다.

- **Technical Challenges**: 에이전틱 AI는 모델 품질만으로는 끝나지 않고, 학습·추론 효율과 정렬 신뢰성, 그리고 실행 중 컨텍스트 관리가 동시에 맞물려야 한다는 기술적 난도가 큽니다. 이 자료는 LLM 학습/미세조정(SFT, LoRA, MoE)과 추론 최적화, 정렬 방법(RLHF/PPO, DPO 및 변형, GRPO) 및 reward modeling을 기초로 두고, 에이전트 계층에서는 trajectory 기반 학습, Agentic RAG, 다양한 메모리( in-context, external, episodic, semantic )로 문제를 분해해 접근합니다. 더불어 에이전트 툴 사용·스킬 설계, 프로토콜(MCP, A2A)과 멀티에이전트 토폴로지로 상호작용을 구조화하고, 구현 예시와 코드 중심 가이드로 연결성을 제공합니다.

- **Empirical Impact**: 단순 개념 소개가 아니라 각 장에 이론 근거와 구현 가이드를 함께 제공해, 실무자가 에이전트 시스템을 단계적으로 검증하고 배포로 옮기는 데 도움이 됩니다. 특히 평가 방법론과 프로덕션 배포까지 포함해, 실험 성능에서 실제 태스크 성능으로의 전이를 촉진하는 점이 의미 있습니다. 결과적으로 에이전틱 AI를 ‘전략’이 아니라 ‘엔지니어링’으로 다루려는 업계/연구 커뮤니티의 학습 경로를 정리해주는 역할을 합니다.



### Learning Action Priors for Cross-embodiment Robot Manipulation (https://arxiv.org/abs/2606.26095)
- **Prior Approaches**: 대부분의 Vision-Language-Action(VLA) 모델은 Vision-Language Model(VLM) 백본에 행동(action) 모듈을 붙이고 정책 전체를 imitation learning으로 end-to-end로 학습한다. 이때 VLM은 시각·언어의 강한 선행지식을 가져오지만, 행동 모듈은 물리적 모션의 시간 동역학을 거의 zero에서 배워야 하는 불균형이 생긴다. 특히 cross-embodiment에서는 로봇마다 action space와 모션 분포가 달라 초기 학습이 불안정해지고 수렴이 느려진다.

- **Core Contribution**: 이 논문은 cross-modal VLA 정렬 전에 행동 모듈에 motion prior를 먼저 학습시키는 2-stage 프레임워크를 제안한다. Stage 1에서 시각·언어 없이 행동 궤적만으로 행동 모듈이 시간적 모션 구조를 학습하고, Stage 2에서는 이를 VLA 학습에 전이해 시각·언어 정렬을 더 안정적으로 시작하게 만든다. 또한 학습된 encoder를 history compressor로 재사용해 긴 시간 문맥을 적은 비용으로 토큰 1개로 요약한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘행동 모듈이 아직 정렬되지 않은 상태’에서 VLA가 동시에 action 분포 학습과 cross-modal alignment를 수행하며 생기는 그라디언트 불안정과 최적화 병목을 줄이는 것이다. 저자들은 Stage 1에서 flow matching 기반 encoder-decoder 행동 모듈을 action-only로 학습해 모션의 구조적 표현을 만든 뒤, Stage 2에서는 decoder 재사용과 early-stage latent distillation으로 VLM 예측 임베딩을 그 모션 잠재공간에 초기에 고정한다. 이후 distillation 제약을 단계적으로 완화해 최종 성능은 end-to-end 정련으로 끌어올린다.

- **Empirical Impact**: LIBERO와 RoboCasa 시뮬레이션 및 실세계 Franka에서 총 13개 cross-embodiment 태스크를 실험해, action prior 없는 VLA 대비 더 빠른 수렴과 더 높은 성공률을 보였다고 보고한다. 특히 데이터가 적은 long-tail 실세계 태스크에서 성능 향상이 두드러지며, history 압축 토큰은 장기 과제에서 시간 수용범위를 늘려 추가 이점을 준다. 또한 Stage 1에서 action-only 데이터를 늘릴수록 더 일반화되는 motion prior가 형성되어 downstream VLA 성능이 직접 개선되는 scaling 경향도 관찰된다.



### On-Policy Self-Distillation with Sampled Demonstrations Reduces Output Diversity (https://arxiv.org/abs/2606.26091)
- **Prior Approaches**: 온폴리 self-distillation은 정답 시연에 조건된 단일 모델을 teacher이자 student로 써서, 토큰 단위의 조밀한 피드백으로 pass@1을 끌어올리는 방식으로 주목받아 왔습니다. 다만 기존 연구는 성능 향상에 초점을 맞추며, 여러 롤아웃에서의 다양성과 pass@k 곡선의 거동(추가 샘플링이 유효한지)을 충분히 진단하지 못했습니다.

- **Core Contribution**: 이 논문은 온폴리 self-distillation이 pass@1은 좋아지더라도 롤아웃 다양성을 떨어뜨리고 pass@k 곡선을 평탄화해 정확도가 더 이상 개선되지 않는 ‘숨은 비용’을 보일 수 있음을 지적합니다. 또한 학생이 생성한 롤아웃을, 별도 샘플된 정답 시연에 조건한 teacher가 평가·피드백하는 구조가 자기 자신의 편향을 강화한다고 원인(설계의 compounding bias)을 추적합니다.

- **Technical Challenges**: 핵심은 self-distillation이 확률 분포를 어떻게 왜곡하는지 이론적으로 설명하는 것이었습니다. 저자들은 최적 self-distillation 정책이 base distribution을 pointwise conditional mutual information 점수로 기울여, 동일하게 정답인 롤아웃들 사이의 확률 비율을 보존하는 이상적인 on-policy RL과 달리 기존 확률 격차를 증폭시키고 지배적인 모드에 질량을 집중시킬 수 있음을 보였습니다.

- **Empirical Impact**: 그래프 경로 탐색과 과학 QA 벤치마크에서 self-distilled 모델은 평균 성능에서 RL과 동등하거나 더 낫지만, functional·semantic 다양성이 크게 감소했습니다. 특히 다양한 전략을 요구하는 out-of-distribution 설정에서 성능 저하가 나타나, 단순 평균 정확도 개선만으로는 안전한 배포를 보장하기 어렵다는 시사점을 줍니다.



### Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents (https://arxiv.org/abs/2606.26080)
- **Prior Approaches**: 기존 Process reward model(PRM) 연구는 단계별 보상을 주기 위해 긴 에피소드에서 인간 주석을 모으거나, Monte Carlo 방식으로 기대값을 추정하는 데 의존해 왔다. 그러나 에이전트 환경은 돌이킬 수 없는 행동과 상태 의존성이 커서 되돌려 반복 롤아웃이 어렵고, 이 때문에 스케일이 막힌다. 또한 도메인별로 PRM을 학습하더라도 다른 과제·환경으로의 일반화가 잘 안 된다는 한계가 있다.

- **Core Contribution**: 이 논문은 RL post-training 과정에서 이미 생성되는 정보만으로, 별도 reward model 학습 없이도 단계 수준 점수(스텝/토큰 스코어)를 만들 수 있다고 제안한다. 핵심은 RL-trained 정책과 reference 정책의 log-probability ratio로 정의되는 progress advantage를 도출하고, 확률적(stochastic) MDP에서도 최적 advantage를 정확히 복원한다는 이론적 근거를 제공한 것이다. 결과적으로 annotation-free, domain-agnostic 신호를 inference-time에서 plug-and-play로 사용할 수 있게 된다.

- **Technical Challenges**: 확률적 환경에서는 기존의 단순한 암묵적 보상(결정적 MDP에서 성립하던 깔끔한 상쇄)이 깨져, log-ratio가 그대로 ‘절대 보상’을 되찾지 못한다. 논문은 목표를 절대 reward가 아니라 advantage로 전환해, 미래 상태 기대값이 자연스럽게 흡수되도록 만들며 확률적 전이에서의 잔차 문제를 구조적으로 해결한다. 또한 KL-penalty 기반뿐 아니라 clipping 기반 RL 알고리즘에서도 적용 가능함을 보이고, 토큰별 advantage를 스텝/궤적 점수로 집계하는 설계(합/평균/가중/극값 등)와 reference policy 선택의 민감도까지 실전 관점에서 정리한다.

- **Empirical Impact**: progress advantage는 test-time scaling, uncertainty quantification, failure attribution의 세 가지 응용에서 5개 벤치마크와 4개 모델 계열에 걸쳐 일관되게 성능을 보였다. 특히 best-of-N 선택에서 성공률을 끌어올리며 confidence 기반 기준선과 사전학습 reward model, 그리고 task-specific PRM까지 능가했다고 보고한다. 별도 task-specific training이 필요 없는데도 AUROC 및 스텝 단위 오류 위치 추정(실패 원인 분해)에서 강한 성능을 보여, 에이전트 평가 파이프라인의 비용을 크게 낮출 수 있는 실용적 의미가 있다.



### A cross-process welding penetration status prediction algorithm based on unsupervised domain adaptation in laser and TIG welding (https://arxiv.org/abs/2606.26078)
- **Prior Approaches**: 기존 용접 관입 상태 분류에는 지도 학습 기반 딥러닝이 널리 쓰이지만, 용접 공정이 바뀌는 domain shift(예: TIG의 arc-dominated → 레이저의 keyhole 기반)에서 성능이 크게 떨어진다. 공정별 물리 메커니즘 차이 때문에, 다른 공정으로 모델을 그대로 옮기면 기준선보다 정확도가 급락한다. 일부는 재라벨링과 fine-tuning으로 대응하지만, 새 공정을 추가할 때 라벨 비용이 커진다는 한계가 있다.

- **Core Contribution**: 이 논문은 라벨 없이도 동작하는 unsupervised domain adaptation(UDA) 프레임워크를 제안하고, 점진적 source domain 확장(GSDE) 전략을 통합해 공정 간 불일치를 완화한다. 핵심은 TIG와 레이저처럼 서로 다른 용접 시스템에서도 domain-invariant한 특징을 학습하면서도 class 경계를 유지하도록 유도하는 것이다. 이를 통해 신규 용접 공정에 대한 relabeling 비용을 낮추고 지능형 용접 모니터링의 범용성을 높인다.

- **Technical Challenges**: UDA는 target 도메인 라벨이 없어서 공정 차이를 “정확한 대응관계”로 정렬하기가 어렵고, 잘못 정렬되면 클래스 경계가 흐려질 수 있다. 저자들은 GSDE로 학습 중에 source 도메인을 점진적으로 확장해 분포 불일치를 단계적으로 줄이고, UMAP 시각화 결과 domain-invariant 특징을 학습하면서도 구분력을 보존하도록 설계했다고 설명한다. 즉, 도메인 정렬과 분류 경계 유지라는 두 목표를 동시에 만족시키는 절충을 목표로 한다.

- **Empirical Impact**: 전용 TIG 및 레이저 데이터셋에서 same-process 전이에서는 TIGFH 90.65%, LSPS 90.72%의 평균 정확도를 기록하며, 지도 기준선 대비 각각 35.83%, 38.87% 향상됐다. cross-process에서는 TIG→Laser 80.48%, Laser→TIG 81.13%로, 기준선 대비 각각 43.39%, 43.40% 개선을 보였다. UMAP 시각화는 학습된 표현이 도메인 불변성을 갖고도 클래스 분리가 유지됨을 뒷받침하며, 공정 전환 시 라벨링 부담을 크게 줄일 수 있음을 시사한다.



### Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignmen (https://arxiv.org/abs/2606.26071)
- **Prior Approaches**: 기존 안전 연구는 주로 ‘우려되는 행동(concerning behavior)’을 탐지하는 데 집중해 왔다. 하지만 특정 행동이 곧바로 오정렬(misaligned)을 뜻하진 않으며, 혼란 같은 양성 요인으로도 같은 행동이 나올 수 있다.

- **Core Contribution**: 이 논문은 모델 포렌식(model forensics)을 위한 기준선(baseline) 프로토콜을 제안한다. 두 단계로 반복 수행하며, 1) CoT(Chain-of-Thought)를 읽어 행동을 유발하는 동기 가설을 만든 뒤 2) 프롬프트나 환경을 수정해 가설을 검증한다.

- **Technical Challenges**: 핵심 난관은 CoT가 항상 충실하지 않을 수 있다는 점이며, 이 한계를 감안해 CoT를 ‘엄밀한 증거’가 아니라 ‘비지도 통찰’로 활용하도록 설계했다. 또한 반사실(counterfactual) 실험처럼 프롬프트/환경 개입을 통해 가설의 예측력이 실제 행동 패턴을 설명하는지 확인하는 방식으로 대응한다.

- **Empirical Impact**: 검증을 위해 우려 행동을 보이는 6개 에이전틱 환경에서 프로토콜을 적용하고, Kimi K2 Thinking은 저노력(low-effort) 성향으로 지름길을 택한다는 가설이 행동을 잘 예측함을 보였다. 반사실 실험에서는 DeepSeek R1이 자기 자신과의 일관성을 맞추려는 욕구 때문에 속임을 사용한다는 정황을 제시했다. 다만 양성 컨트롤 부재 등 한계도 남아 있으며, 그럼에도 단순한 프로토콜이 강력한 출발점이 될 수 있음을 보여주며 분야의 후속 연구를 촉진한다.



### A welding penetration prediction model for laser welding process based on self-supervised learning using physics-informed neural networks (https://arxiv.org/abs/2606.26059)
- **Prior Approaches**: 레이저 용접의 full-penetration(완전 관통) 상태는 결함 없는 접합을 좌우하므로, 이를 정확히 예측하는 분류 연구가 중요해졌다. 기존 supervised learning 기반 분류는 높은 성능을 내지만, 산업 현장에서 충분히 크고 고품질인 라벨 데이터를 대량으로 수집·정제해야 한다는 한계가 있다.

- **Core Contribution**: 이 논문은 제한된 소수 라벨 이미지로도 관통 상태를 고정밀로 분류하는 SimPhysNet을 제안한다. 핵심은 물리적 사전지식(physical priors)을 contrastive learning에 내재화하는 self-supervised 학습 패러다임이며, molten pool과 keyhole에 대해 물리적으로 의미 있는 특징을 학습하도록 PINN을 결합한다.

- **Technical Challenges**: 문제는 (1) 라벨이 적을 때도 물리적으로 타당한 표현을 뽑아내고, (2) 서로 다른 촬영/조건에서 일반화하도록 만드는 것이다. 저자들은 대량의 비라벨 데이터에 대해 PINN으로 물리 제약을 주입하고, 세 가지 image augmentation 과제를 더해 일반화를 강화한 뒤, prototypical networks 기반 few-shot 학습으로 최소 라벨로 클래스 표현을 구성해 강건한 분류를 수행한다.

- **Empirical Impact**: 실험에서 SimPhysNet은 200장(전체 라벨의 약 5%)만 사용해 96.06%의 분류 정확도를 달성했으며, 전체 라벨을 쓰는 기존 supervised learning과 유사한 성능을 보인다. 라벨 의존성을 크게 낮추는 동시에 정밀도를 유지한다는 점에서 레이저 용접 자동화와 공정 지능화에 실용적 의미가 크다.



### Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining (https://arxiv.org/abs/2606.26050)
Comments:
          Foundations of Deep Generative Models (FoGen) Workshop at ICML 2026. 23 pages (5-page main text plus appendices), 5 figures. Code: this https URL

- **Prior Approaches**: 기존 grokking/ungrokking 연구는 대체로 학습이 끝난 뒤의 일반화가 데이터 규모나 하이퍼파라미터에 따라 되돌아가거나 지연되는 현상을 다뤘다. 또 일부 작업은 학습이 계속되는 동안 in-context learning 같은 능력이 흐려질 수 있음을 관찰했지만, “능력(규칙) 자체가 언제/왜 남고 사라지는지”를 단일 규칙 단위로 인과적으로 정리하진 못했다.

- **Core Contribution**: 이 논문은 pretraining 한 번의 진행 중에도 언어 규칙(예: 성별 대명사 결정)이 획득되었다가 “within-run reversal”로 붕괴할 수 있음을 pronoun-gender rule 사례로 보여준다. 이를 natural ungrokking이라 부르며, 어떤 규칙이 끝까지 살아남는지가 loss 곡선의 흔적 없이 “말뭉치가 그 규칙을 이기는 횟수( support frequency )”만으로 예측된다고 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 규칙이 사라질 때 그것이 (1) 능력을 통째로 잊은 erasure인지 (2) 규칙을 유지하되 표면적 경쟁 패턴에 밀려서 규칙만 포기한 displacement인지 구분하는 것이다. 연구진은 고정된 템플릿 probe 배터리와 agree/ conflict 조건, 그리고 rule-대상 prior 간 log-probability 대비 여유(margin) 같은 도구로 “구조는 남고 규칙만 밀려난다”는 메커니즘을 단계별로 분해해 검증했다.

- **Empirical Impact**: 여러 말뭉치·데이터 예산·시드에서 support frequency가 높은 규칙은 등장 후 유지되지만, 낮은 규칙은 특정 시점에 행동이 무너지고 margin이 0을 통과하며 예측이 붕괴한다. 특히 같은 편집을 되돌리는 복원(지원 주입) 실험에서는 규칙이 회복되지 않아, 규칙 파괴는 쉽게 가능하지만 복원은 비대칭적으로 어렵다는 결론을 뒷받침한다.



### TriViewBench: Controlled Complexity Scaling for Multi-View Structural Reasoning in MLLMs (https://arxiv.org/abs/2606.26029)
Comments:
          26 pages, 8 figures

- **Prior Approaches**: 기존 MLLM 시각추론 벤치마크는 주로 단일 이미지 중심이어서, 시점 간 동일성 매칭이나 가림(occlusion) 보정 같은 다중 시점 통합 요구가 제한적이었다. 또한 실세계 이미지 기반은 객체 수·공간배치·가림 정도가 함께 변해 성능 저하의 원인을 분리해 보기 어려웠다. 다중 뷰로 확장된 일부 벤치마크도 복잡도 축을 명시적으로 파라미터화하지 않아 구조적 실패 유형을 정밀 진단하기 힘들다는 한계가 있었다.

- **Core Contribution**: 이 논문은 합성 3D 장면을 이용해 객체 수와 가림 정도를 독립적으로 제어하는 3-view 시각추론 벤치마크 TriViewBench를 제안한다. 프론트/사이드/탑다운 세 시점을 동시에 주고, 난이도를 4단계와 추론 범주( Local Decision, Object Counting, Global Recovery )로 체계적으로 나눠 구조적 스케일링을 측정한다. 이를 통해 현재 MLLM의 능력 계층과 실패 메커니즘을 구분해 진단할 수 있는 프레임을 제공한다.

- **Technical Challenges**: 가림과 객체 수가 동시에 변하는 실세계 한계를 피하려면, 각 QA 정답을 자동으로 유일하게 도출할 수 있는 정밀한 메타데이터 기반 합성 설계가 필요했다. TriViewBench는 Kubric 기반 절차적 장면 생성과 카메라 파라미터 고정으로, 시점별 가시성/바운딩박스 및 관계 메타데이터를 주석처럼 생성해 정답 모호성을 줄였다. 또한 Object Counting에서 실패 방향(언더카운트 vs 오버카운트)이 단일 시점과 다중 시점에서 반대로 나타남을 error analysis로 기계적으로 분해했으며, CoT는 거의 무효(Δ=-0.16%)임을 함께 확인해 추론 전략이 아닌 교차시점 공간표현 병목을 시사했다.

- **Empirical Impact**: 18개 오픈/클로즈드 MLLM 모두 동일한 능력 계층( Local Decision > Object Counting > Global Recovery )을 예외 없이 보였고, 복잡도 4단계로 갈수록 단조 하락했다. 하락 폭은 Local Decision 12.11%에 비해 Object Counting 59.14%, Global Recovery 80.02%로 급격히 커져 구조 재구성이 가장 취약함을 실증했다. 특히 Object Counting은 단일 시점 가림맹(undercount)과 다중 시점 동일성 혼동(overcount)이 독립적 실패 모드로 관측돼, 향후 다중시점 표현 학습·정렬의 과제를 분명히 하는 진단 도구로 의미가 크다.



### Can Trustless Agents Be Trusted? An Empirical Study of the ERC-8004 Decentralized AI Agent Ecosystem (https://arxiv.org/abs/2606.26028)
- **Prior Approaches**: 기존 에이전트 프로토콜(A2A, MCP 등)은 주로 검색·메시징·태스크 흐름을 표준화하지만, 미지의 상대를 신뢰할지 판단하는 방법은 애플리케이션에 맡겨 왔다. 중앙화 평판 플랫폼, DID 같은 신원 증명, 도메인 기반 PKI 등도 각각 한계(중앙 의존, 행동 평판 축적 어려움, 진실성 대신 신원/소유만 검증)를 가진다.

- **Core Contribution**: 본 연구는 ERC-8004(Trustless Agents)가 실제로 “신뢰 신호”를 제공하는지 검증하기 위해 최초로 크로스체인 실증 분석을 수행한다. Ethereum, BNB Smart Chain(BSC), Base에서 프로토콜 배포 시점부터 2026년 5월 13일까지 Identity·Reputation 온체인 이벤트와 오프체인 파일, x402 결제 데이터를 수집해 평판 레이어의 신뢰성 여부를 평가한다.

- **Technical Challenges**: 저자들은 ERC-8004의 설계가 온체인에 기록하는 정보가 의사결정에 쓰일 만큼 검증 가능하고 조작 내성이 있는지(가령 척도 정합성, 검증된 상호작용 기반성, 조작 비용, Sybil 내성)를 실증적으로 따져야 한다는 문제를 다룬다. 결론적으로 현재 Registry 배치 상태에서는 평판 값이 서로 비교 가능하지 않고, 피드백이 검증 가능한 상호작용에 거의 접지(grounding)되지 않으며, 단일 입력으로 집계가 흔들리고 저비용 조작이 가능하다고 분석한다.

- **Empirical Impact**: 데이터에 따르면 등록의 다수는 실제 에이전트가 아니라 placeholder에 가깝고, 서비스 엔드포인트가 있는 “유효한 ERC-8004 등록 파일” 비율은 체인별로 3%~15% 수준에 그친다. 더 나아가 리뷰어의 상당 비율이 조정된 Sybil 행태를 보여(체인별 59.2%~90.6%), 이를 제거하면 평점을 지탱할 “유효 피드백이 없는” 에이전트가 체인별로 15.5%~89.4% 남는 것으로 나타나 신뢰 신호로서의 실효성이 크게 제한됨을 시사한다.



### Privacy Vulnerabilities of Attention Layers in Tabular Foundation Models and Protection of High-Risk Queries (https://arxiv.org/abs/2606.26021)
Comments:
          18 pages, 12 figures, 4 tables

- **Prior Approaches**: 기존 멤버십 인퍼런스 공격(MIA)은 주로 모델 출력의 confidence, loss 같은 결과 기반 신호를 활용하거나 shadow model, 기준선(기준 분포) 같은 보조 정보로 likelihood ratio를 계산해 왔습니다. 하지만 최근에는 in-context learning(인컨텍스트 러닝, ICL)에서 공격 표면이 파라미터 암기보다 ‘컨텍스트 예시’에 대한 주의(attention) 의존으로 이동할 수 있다는 점이 문제로 제기돼 왔습니다. 특히 텍스트 LLM에서는 attention 기반 신호가 학습 멤버십을 드러낼 수 있다는 연구가 있었지만, 구조화된 tabular transformer(탭룰러 트랜스포머)에서의 체계적 검증과 직접적 공격 설계는 부족했습니다.

- **Core Contribution**: 이 논문은 tabular foundation model이 synthetic pre-training을 했더라도 ICL 단계에서 컨텍스트 라벨 예시가 멤버십을 유출할 수 있음을 처음으로 체계적으로 보입니다. 이를 바탕으로, shadow-model 없이도 transformer attention 패턴의 ‘집중도’를 이용해 멤버십을 추정하는 Attention-based Membership Inference Attack(AMIA)을 제안합니다. 또한 inference-time에서 k-anonymity 원리를 응용한 label-aware microaggregation 기반 방어를 고위험 질의에만 적용해 누출을 줄이되 성능 저하는 최소화합니다.

- **Technical Challenges**: 핵심 난제는 ICL에서 누출이 confidence 같은 단순 출력값뿐 아니라 attention 메커니즘의 패턴(레이어·헤드에 걸친 분포 집중)을 통해 드러난다는 점을, tabular 컨텍스트의 고정된 row/키-값 구조에 맞춰 ‘직접’ 읽어내는 것입니다. 저자들은 member 질의가 특정 컨텍스트 레코드로 attention을 비정상적으로 집중시키는 경향을 가정하고, 해당 집중 신호를 점수화해 membership 판별에 활용하는 AMIA를 설계했습니다. 방어는 무작위 잡음이나 재학습 없이, AMIA 점수로 고위험 질의를 골라 컨텍스트 key 표현의 유일성을 낮추는 방식으로 작동하도록 구성했습니다.

- **Empirical Impact**: 실험은 6개 데이터셋에서 다양한 tabular FMs 및 기존 분류 모델을 대상으로 수행됐고, AMIA는 기존 confidence 기반 MIA를 상회하며 특히 낮은 false-positive 구간에서 평균 gain 7.7%를 달성했다고 보고합니다. 방어(고위험 질의에만 적용)는 멤버십 누출을 AMIA 기준 평균 약 50%, confidence 기반 대비 평균 약 25% 낮추면서도 예측 유틸리티는 3.9% 성능 저하 수준에 그쳤습니다. 추가로 fine-tuning은 일부 샘플에서 confidence separation을 키워 MIA 민감도를 증폭시킬 수 있음을 보여, 운영 관점에서 성능 향상만큼 프라이버시 검증이 필요하다는 신호를 제공합니다.



### FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation (https://arxiv.org/abs/2606.26006)
- **Prior Approaches**: VLA는 보통 SFT 같은 모방학습으로 사전학습되지만, 시연 데이터의 품질 한계인 imitation ceiling 때문에 성능이 더 이상 쉽게 오르지 못한다는 점이 알려져 있습니다. RL fine-tuning으로 이를 넘을 수 있지만, 실제 로봇에서는 샘플 비효율과 불안정(초기 unlearning, 저품질 탐색 데이터로 인한 업데이트 잡음)이 커서 사람 개입(HiL)에 의존하는 경우가 많았습니다. 또한 기존 방식은 Q-기반 업데이트를 하더라도 O2O 전환의 분포 불일치와 탐색 잡음 때문에 실패율이 출렁이는 문제가 반복됩니다.

- **Core Contribution**: 이 논문은 intervention-free 오프라인-투-온라인 VLA RL fine-tuning 프레임워크 FORCE를 제안하며, 시연 기반 한계를 RL로 넘는 동시에 안정성을 확보하는 것을 목표로 합니다. FORCE는 3단계로 구성되며, Value-Calibrated Warm-Up로 Q-function의 분포/지지영역을 정책 방문 분포에 맞춘 뒤, 온라인에서는 VGPD(Value-Guided Policy Self-Distillation)로 고가치 전이만 정책 업데이트에 반영해 탐색 잡음을 걸러냅니다. 결과적으로 “분포 이동으로 인한 성능 붕괴”와 “저품질 탐색 업데이트”를 동시에 줄이도록 설계됐습니다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 오프라인에서 학습된 과소/불안정 Q-value 스케일이 온라인 데이터를 만나며 초기 unlearning을 유발하는 cold-start covariate shift 문제, (2) 보상 희소성과 행동공간 크기로 인해 탐색 데이터의 품질이 낮아 정책 업데이트가 분산을 키우는 문제입니다. FORCE는 Warm-Up에서 on-policy 롤아웃을 소량 섞어 conservative value constraint를 적용함으로써 critic의 지원 영역을 확장해 Q-estimate의 바깥삽입(extrapolation)과 정의역 불일치를 줄입니다. 이어 VGPD는 KL 제약 기반의 정규화된 정책개선 관점에서, 샘플별 value advantage를 동적 기준선으로 필터링해 음의 어드밴티지를 버리고(Positive Advantage Truncation) 고가치 전이로만 자기 증류를 수행하게 합니다.

- **Empirical Impact**: 실험은 ManiSkill 시뮬레이션과 Franka Emika Panda 실로봇에서 수행됐고, FORCE는 평균 성공률 82.3%(Octo 백본 기준)로 기존 RL 방법(ConRFT 등) 대비 10%p 이상 성능 우위를 보였습니다. 또한 성공률이 내려가는 흔한 성능 하락 현상을 완화하면서, 학습 단계 관점에서도 평균 Steps@80%가 32.5% 단축되는 등 샘플 효율을 개선했습니다. 특히 인간 개입 없이도 contact-rich 과제에서 거의 100% 성공률을 달성해, 자율 로봇 에이전트 배치 가능성을 한 단계 끌어올렸다는 점에서 의미가 큽니다.



### Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization (https://arxiv.org/abs/2606.26002)
- **Prior Approaches**: 기존 압축 연구는 휴리스틱(고정 규칙), one-shot 혼합정밀 할당, 또는 단일 목적 최적화에 주로 의존해 계층 간 의존성과 미세 조정의 필요를 충분히 반영하지 못했다. AMC(He et al., 2018)와 HAQ(Wang et al., 2019) 같은 RL 기반 방법도 layer-wise 결정에 머물러 pruning과 quantization을 단일 축 위주로 다루는 한계가 있었다. 또한 Hessian trace(HAWQ-V2, Dong et al., 2020)는 sensitivity를 활용하지만 계산 비용이 커 실용적 탐색에 부담이 있다.

- **Core Contribution**: HiReLC는 pruning(구조화)과 quantization(혼합정밀)을 동시에 자동으로 찾는 architecture-agnostic 계층형 ensemble-reinforcement learning 프레임워크다. 저수준 에이전트(LLA)는 블록별로 bitwidth, keep-ratio, quantization type, granularity를 포함한 다중 이산(action)에서 per-kernel 구성을 선택하고, 고수준 에이전트(HLA)는 Fisher Information 기반 민감도 추정으로 전역 예산을 배분한다. 여기에 surrogate-guided active learning 루프를 더해 RL 탐색 비용을 줄이면서도 최종 압축 후 fine-tuning 평가를 대체하지 않도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) pruning·quantization을 함께 최적화해야 하며 (2) 탐색 공간이 커서 single-level controller의 분산과 불안정성이 커진다는 점, (3) 블록 중요도에 따라 예산을 정교하게 배분해야 한다는 점이다. HiReLC는 검색을 두 단계(전역 예산 배분 vs 블록 내부 per-kernel 결정)로 분해해 credit assignment를 정리하고, Fisher Information 민감도를 관측·보상·예산 보정에 동시에 반영해 덜 중요한 블록은 더 공격적으로 압축하고 중요한 블록은 상대적으로 완화한다. 더불어 MLP surrogate와 logit-MSE 프록시로 정책 평가의 상당 부분을 완화하되, surrogate는 reward shaping에만 사용하고 실제 post-compression fine-tuning 기반 평가는 유지해 신뢰성을 확보한다.

- **Empirical Impact**: 실험에서 HiReLC는 Vision Transformer와 CNN 벤치마크 전반에서 유효 parameter-storage 기준 5.99–6.72× 압축을 달성했으며, 한 설정에서는 정확도 3.83% 향상도 관측됐다. 반대로 다른 조건에서는 0.55–5.62% 정확도 하락이 나타나, 민감도 기반 가이드와 계층 분해가 설계 선택으로서 효과적이지만 압축 목표(제약)와 데이터/아키텍처에 따라 트레이드오프가 달라짐을 보여준다. 전반적으로 sensitivity-aware 계층형 정책 분해와 surrogate-augmented active learning이 joint compression을 실용적인 탐색 절차로 가져오는 접근으로 평가된다.



### Variable Bound Tightening for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games (https://arxiv.org/abs/2606.25997)
- **Prior Approaches**: 2인 0합 불완전정보 게임에서는 CFR(counterfactual regret minimization)과 fictitious play가 스케일과 수렴 보장을 제공하지만, 멀티플레이어 일반 게임에서는 Nash equilibrium로의 수렴이 보장되지 않는다. 반면 멀티플레이어 불완전정보에서 정확한 Nash equilibrium 계산은 NLCP/보완조건을 포함한 복잡한 최적화로 귀결되며, 기존에는 Gurobi 비볼록 2차 해법으로도 3인 Kuhn poker의 완전판을 24시간 내 해결하지 못했다.

- **Core Contribution**: 이 논문은 멀티플레이어 Nash equilibrium을 위한 NLCP(Nonlinear Complementarity Program) 기반 QCP(Quadratically-constrained Program)에서 슬랙 변수와 멀티플라이어 변수에 대한 “유한한 전역 상계/하계”를 유도한다. 이러한 경계는 공간 branch-and-bound에서 사용되는 convex relaxation을 강화해, 동일한 solver 설정에서도 계산 성능을 크게 끌어올리도록 설계되었다.

- **Technical Challenges**: 핵심 난제는 NLCP가 비선형 보완성 조건으로 인해 변수 곱을 포함하며, 특히 기존의 느슨한 도메인(r_i: [0,∞), λ_i: (−∞,∞))이 branch-and-bound의 완화 품질을 떨어뜨린다는 점이다. 논문은 슬랙 변수를 각 플레이어의 best-response 선형계획 해석(강한 쌍대성/KKT)에서 얻어 상계를 계산하고, 멀티플라이어는 정보셋 트리 구조에 대한 backward induction 및 기대유틸리티의 볼록결합 성질로 전역 바운드를 도출해 relaxation을 강화한다.

- **Empirical Impact**: 실험에서 3인 Kuhn poker 완전판은 기존 불가능 수준(24시간 실패)에서 제안한 슬랙 변수 바운드만 적용했을 때 1.160초 내 최적해(사실상 exact Nash equilibrium)를 산출했다. 멀티플라이어 바운드까지 함께 적용하면 오히려 느려지는 케이스가 있었지만, overall로는 Gambit 소프트웨어 군의 해당 접근들보다 NLCP+강화 바운드가 더 빠르게 정확해를 구하며 실용적 exact 계산 가능성을 보여준다.



### SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models (https://arxiv.org/abs/2606.25990)
- **Prior Approaches**: 기존 음성 정서 지능 평가는 주로 Speech Emotion Recognition(SER)처럼 잡음을 라벨로 분류하거나, 텍스트만으로 감정지능을 추론하는 방식에 치우쳐 있었다. 그 결과 음성의 prosody·타이밍·강도 같은 부수적(파라링귀스틱) 신호를 ‘대화에서’ 얼마나 논리적으로 다루는지, 그리고 multi-turn에서 맥락을 유지하는지 검증이 약했다.

- **Core Contribution**: 이 논문은 SpeechEQ라는 평가 프레임워크와 벤치마크를 제안해 Speech-Language Models(SLMs)의 sociolinguistic reasoning을 다중 턴 음성에서 측정한다. EQ-i 2.0 이론에 기반한 15개 EQ 하위 척도, 2,265개 대화와 이를 요약하는 Spoken Emotional Quotient(SEQ) 점수를 도입해 인간 EQ 평가를 닮은 형태로 표준화했다.

- **Technical Challenges**: 핵심 기술 과제는 텍스트 의미 단서를 제거하면서도 음성만으로 정서적 적합성을 강제하는 ‘cross-modal reasoning’을 설계하는 것이다. 이를 위해 동일 transcript를 가진 두 오디오 선택지를 forced-choice로 구성하고, 다중 턴 정서적 긴장도를 순차적으로 상승시키며, 음향 대조를 librosa 특징(피치·발화 속도·스펙트럴 센트로이드·에너지·MFCC·길이 등)으로 검증해 품질을 확보했다.

- **Empirical Impact**: 실험 결과 end-to-end 모델이 cascaded 파이프라인보다 전반적으로 우수하지만, modality shortcut·alignment-induced safety trap·contextual amnesia 같은 병목이 여전히 드러났다. 또한 SEQ가 인간 선호와의 Spearman 상관(ρ=0.943)을 보여 기존 단순 정확도보다 신뢰성 있는 감정지능 대리 지표임을 입증했으며, 특히 2번째 평가 턴에서 성능 저하가 나타나 지속 정서 추적의 한계를 확인시켰다.



### Weave of Formal Though (https://arxiv.org/abs/2606.25987)
Comments:
          Code is available at this https URL

- **Prior Approaches**: 기존 constrained-decoding은 부분 파스에 맞는 토큰만 남기도록 어휘 마스킹을 하며 문법적 타당성을 높이지만, 컨텍스트-센서티브 lexing·maximal-munch·키워드 추출 같은 핵심 렉서 메커니즘을 다루는 데 제한이 있다. 또한 언어모델은 계층 구조(비터미널/파생)를 내부적으로 학습하기보다는 미리 정해진 정책으로 문법을 주입하는 경우가 많아, 어떤 구조를 드러내야 하는지 자체 학습이 약하다. 한편 CoT 계열이나 내부 추론 기법은 검증 가능한 이산 구조를 제공하지 못해 형식 오류를 체계적으로 배제하기 어렵다.

- **Core Contribution**: WoFT(Weave of Formal Thought)는 (1) Tree-sitter 전체 스펙에 대해 sound & complete한 형식 엔진 기반 constrained decoder와, (2) 비터미널 문법 파생을 이산 잠재변수로 학습하는 latent-variable fine-tuning을 결합한다. 형식 엔진은 subword 토큰이 “유효한 프로그램 prefix로 확장될 수 있는 경우”만 허용해 문법 검증을 생성 루프에 직접 내장한다. 학습 단계에서는 비터미널을 생성에 끼워 넣어, 모델이 필요할 때만 “형식적 파생(구조 scratchpad)”을 보존하도록 유도한다.

- **Technical Challenges**: 주요 과제는 LLM이 임의의 subword 경계에서 토큰을 생성할 때, Tree-sitter 렉서/외부 스캐너가 요구하는 완전한 입력과 maximal-munch를 온라인으로 만족시키는 것이었다. WoFT는 GLR 파싱에 speculative-lexing을 결합해, 렉서 상태 가설을 graph-structured stack과 동기화하며 동시다발적으로 토큰 확장 가능성을 추적한다(외부 스캐너는 coroutine 형태로 scanlet을 suspend/resume). 학습에서는 과거의 구조 잠재변수 모델처럼 완전한 parse 트리를 정확 마진화하는 대신, Tree-sitter를 symbolic oracle로 두고 RWS(reweighted wake-sleep)로 IW-ELBO를 최적화해 surface text의 중요도 가중 증거를 통해 비터미널을 선택적으로 학습한다.

- **Empirical Impact**: Python에서 StarCoder2-3B(3B)를 WoFT의 RWS 목표로 fine-tuning하면, text-only SFT baseline 대비 per-token cross-entropy가 14.3% 상대 감소했다. 이는 평면적인 autoregressive 학습이 잃기 쉬운 핵심 구조 정보를, 이산 비터미널 잠재 구문으로 되살릴 수 있음을 실증한다. 결과적으로 생성 시점의 형식 검증과 학습 시점의 구조 내재화가 함께 개선되며, 코드 LLM의 오류율을 낮추는 실용적 방향을 제시한다.



### Multi-Agent Goal Recognition with Team- and Goal-Conditioned Reinforcement Learning and Factorized Branch-and-Bound (https://arxiv.org/abs/2606.25978)
Comments:
          12 pages, 1 figure, 2 tables

- **Prior Approaches**: 기존 다중 에이전트 goal recognition은 대부분 단일 행위자(에이전트)의 숨은 목표만 추론하거나, 명시적 plan library/BDI 구조 같은 도메인 모델을 전제로 합니다. 그 결과 관찰 가능한 것은 궤적(trajectory)뿐인 현실 환경에서는 팀 단위의 조정 여부와 각 팀의 목표를 동시에 랭킹하기가 어렵습니다. 또한 분할(partition)과 목표 수가 늘면 가설 공간이 조합적으로 폭증해 exhaustive 방식은 계산량이 커집니다.

- **Core Contribution**: 이 논문은 관찰된 joint trajectory만으로 “어떤 에이전트가 어떤 팀을 이루는지”와 “각 팀이 무엇을 목표로 하는지”를 함께 추론하는 MAGR-BB를 제안합니다. 핵심은 team-와 goal-조건이 붙은 공유 policy를 scoring model로 두고, non-competitive 점수 가정 하에서 factorized branch-and-bound로 top-kk 랭킹을 그대로 보존한다는 점입니다. 이 설계는 모든 complete partition-goal 가설을 만들지 않고도 동일한 상위 결과를 재현하도록 목표를 둡니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 조합 폭발을 줄이면서도, 학습된 정책 점수가 “부분 정보로도 가지치기 가능한 상한(bound)”을 갖도록 만드는 것입니다. 논문은 팀 단위로 점수를 분해(각 팀의 likelihood가 다른 팀의 목표와 비의존)하고, 각 partition에 대해 목표 선택의 최댓값을 이용한 상한을 계산해 pruning합니다. 또한 Transformer 기반의 goal·team-조건 정책을 counterfactual input으로 재질의하고, top-kk floor 아래로는 더 확장하지 않는 best-first heap 탐색으로 계산을 추가 절감합니다.

- **Empirical Impact**: Blocksworld 기반 controlled multi-agent 벤치마크에서 MAGR-BB는 모든 관찰 단계에서 exhaustive search와 동일한 top-랭크 가설을 산출했으며, 최종 top-10 리스트도 일치했습니다. 계산 효율은 매우 공격적인 수준으로, 마지막 관찰 시 complete hypothesis materialization이 수백만 단위에서 10^1 범위로 줄고 누적 인식 runtime도 2.4~2.9배 수준으로 감소했습니다. 즉, 궤적만 주어지는 실제 감시/협업 로보틱스 시나리오에서 팀-목표 동시 추론을 현실적인 시간 안에 제공할 가능성을 보여줍니다.



### Helpful or Harmful? Evaluating LLM-Assisted Vulnerability Patching via a Human Study (https://arxiv.org/abs/2606.25973)
Comments:
          7 pages, 6 figures

- **Prior Approaches**: 기존 연구는 LLM 기반 취약점 패치를 생성할 수 있다는 가능성을 보여줬지만, 함수 테스트만 통과하는 “가짜 수정(fake fix)”이 늘어나거나 미묘한 취약점·할루시네이션이 남는 문제가 반복적으로 보고돼 왔다. 또한 사람 대상 연구에서도 LLM 도움을 받을수록 보안성에 대한 확신은 커지는데 실제 코드는 덜 안전할 수 있다는 상반된 관찰이 존재한다. 특히 표면적 수정이 실제 취약성 제거가 아닌 경우를 체계적으로 드러내는 인간 실험 설계는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 실제 개발자(보안 전문가가 아닌 일반 개발자) 관점에서, LLM-assisted 취약점 패치가 기능 정합성뿐 아니라 보안 검증에서도 진짜로 개선되는지 통제 실험으로 밝히려 한다. 핵심은 보이는 테스트를 통과해 만족감을 주는 패치가 실제 취약성을 그대로 남기는지 확인하기 위해 Ghost Tests(숨겨진 보안 전용 테스트)를 도입한 점이다. 분석 단위도 개별 모델 성능이 아니라, 사람–LLM 상호작용이 패치 품질과 속도·인식에 미치는 영향을 측정하도록 설계했다.

- **Technical Challenges**: LLM이 빠르게 코드를 생성하더라도 보안 의미를 충분히 검증하지 못해 겉보기 기능 통과는 되지만 근본 원인이 남을 수 있다는 점이 가장 큰 기술적 난제로 제시된다. 이를 위해 WebApp에서 실행·텔레메트리를 통일하고, 제출 시점에만 실행되는 숨은 Ghost Tests로 exploit-relevant 동작을 추가 평가한다. 또한 시간 제한(30분)을 right-censoring으로 처리하고, Balanced Crossover(균형 교차) 설계로 개인 숙련도와 학습/이월 효과를 통제하며 속도는 Cox 모델, 보안성은 fake fix 비율과 서명 순위 검정 등으로 검증한다.

- **Empirical Impact**: 파일럿에서는 LLM-assisted 조건이 평균적으로 더 빠른 수정을 보였지만, 보안 테스트 통과율은 수동 대비 높지 않았고 가짜 수정이 관찰되는 양상도 통계적으로는 유의미하지 않았다. 다만 정성 피드백에서 “한 줄 변경만으로 보이는 테스트가 통과된다”는 혼란과, 취약점/DB 이해 부족으로 인한 검증 의존이 확인돼 가짜 수정의 위험이 현실적으로 재현됨을 시사한다. 본 연구는 이후 N=60 규모의 본 실험으로 기능 테스트 통과와 보안 테스트 통과의 괴리(HE=E)를 정량화해, LLM이 실제 보안 패치 품질을 높이는지 혹은 과신을 강화해 “기능은 맞지만 취약성은 남는” 결과를 늘리는지를 실증적으로 가늠할 계획이다.



### SE-AGCNet: An End-to-End Framework for Joint Speech Enhancement and Loudness Control in Meeting Scenarios (https://arxiv.org/abs/2606.25959)
Comments:
          Accepted by Interspeech 2026

- **Prior Approaches**: 기존 오디오 파이프라인은 AEC, SE, AGC를 모듈별로 분리해 순차(cascaded) 처리하는 경우가 많았다. AGC를 SE보다 먼저 적용하면 잡음까지 같이 키워 SNR이 악화되고, SE 뒤에 AGC를 붙이면 SE가 남긴 잔여 잡음이 다시 증폭될 수 있다. 또한 SE 모델이 작은 음량의 발화를 과도하게 억제하는 경향이 있어, 회의 환경처럼 볼륨 변동이 큰 상황에서 성능 저하가 반복됐다.

- **Core Contribution**: 이 논문은 SE와 AGC를 end-to-end로 함께 최적화하는 SE-AGCNet을 제안해, 두 모듈의 역할 분담이 자연스럽게 협업하도록 만든다. 핵심 아이디어는 SE가 저음량 발화를 보존해 두고 AGC가 그 정보를 바탕으로 볼륨 정규화를 수행하게 하는 시너지다. 또한 회의 시나리오에 맞춘 SE-AGC-DataGen과, AGC 평가를 위한 LUFS/St LUFS/LRA의 표준화 지표를 함께 제시한다.

- **Technical Challenges**: SE-AGCNet의 기술적 난제는 (1) SE가 잡음 억제와 저음량 발화 보존을 동시에 달성해야 하고 (2) AGC가 잔여 잡음을 증폭하지 않으면서 목표 loudness를 안정적으로 맞춰야 한다는 점이다. 논문은 이를 위해 시간-주파수(TF) 영역에서 SE(크기/위상)와 AGC(크기, RMS 정규화 기반)를 결합하고, SE·AGC 모두에서 과억제/무음 구간 잡음 증폭에 불리한 경우 10배 가중 패널티 같은 비대칭 reweighting을 적용한다. 더 나아가 MP-SENet 백본은 기존 학습 구성을 유지하되 over-suppression을 줄이도록 손실을 조정하고, 전체 프레임워크는 SE 타깃으로 일정 구간 curriculum learning을 거쳐 안정화한다.

- **Empirical Impact**: 시뮬레이션 데이터셋 LibriAGC와 실제 회의 데이터(MMCSG, AliMeeting-far)에서 SE-AGCNet은 목표 loudness 범위를 일관되게 달성하면서 음성 품질, PESQ, 그리고 ASR 성능(WER/CER)을 경쟁 기법 대비 개선했다. 특히 MP-SENet+pyagc 같은 off-the-shelf AGC 후처리보다 joint optimization이 더 효과적임을 보였고, 한정 학습 데이터에서 ASR 모델이 더 민감하다는 관찰과 함께 개선 폭이 커지는 경향도 확인됐다. 결론적으로 회의처럼 볼륨 편차가 큰 음성 전처리에서 SE와 AGC를 함께 설계해야 한다는 실증 근거를 제공한다.



### Pulmonary Embolism Risk Stratification from CTPA and Medical Records: Vascular Graphs Are Not All You Need (https://arxiv.org/abs/2606.25956)
Comments:
          8 1/2 pages + 2 pages of references. Accepted for MICCAI 2026. This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is published in, and available online at, the external reference provided below

- **Prior Approaches**: 폐색전증(PE) 위험도 분류는 환자 병력, CTPA(전산화폐동맥조영술)에서 추출한 지표, 혈액검사 결과에 기반해 이뤄진다. 하지만 실제 임상에서는 혈액검사가 누락되는 경우가 잦아, 기록과 영상 기반 정보만으로 정확도를 유지하기 어렵다. 기존에는 CTPA의 혈관 정보를 활용하더라도 혈액검사 부재 상황을 직접적으로 보완하는 방식이 제한적이었다.

- **Core Contribution**: 이 논문은 혈액검사 없이도 병력과 CTPA에서 뽑은 심장 바이오마커(혈액검사 대체 성격)를 조합해 PE 위험도를 분류할 수 있는지를 탐색한다. 특히 CTPA의 폐혈관 정보를 더해 탭ular 모델에 혈관 바이오마커를 넣거나, 혈관 트리의 intrinsic graph 표현으로 GNN을 적용하는 비교 벤치마크를 제시한다. 결과적으로 혈관 바이오마커와 혈관 그래프 기반 접근이 위험도 분류 성능을 추가로 끌어올리지 못했다는 점을 핵심 결론으로 다룬다.

- **Technical Challenges**: 핵심 기술 과제는 CTPA에서 유의미한 폐혈관 정보가 실제로 위험도 판별에 기여하는지, 그리고 그것을 모델이 효과적으로 학습할 수 있는 형태로 표현하는 것이다. 연구진은 (1) 병력·심장 바이오마커 중심의 전역(global) 특징을 강한 기준선으로 두고, (2) 혈관 바이오마커를 탭ular 피처에 보강하거나, (3) 혈관 트리 그래프에 GNN을 적용해 성능 향상을 기대했지만, GNN조차 전역 특징의 강한 탭ular baseline을 넘지 못했다. 또한 이 비최적 성능을 설명하기 위해 모델 가정(표현/학습)과 데이터 특성(정보 부재 또는 약한 신호) 양쪽의 가설을 함께 검토한다.

- **Empirical Impact**: 비공개 데이터(n=353)에서 모든 정보가 유의미하게 완전하게 수집된 케이스를 사용해 실험을 수행했으며, 병력과 심장 바이오마커가 전역 특징 중 가장 중요한 예측 요인으로 나타났다. 반대로 혈관 바이오마커는 추가 향상을 주지 않았고, 혈관 그래프 기반 GNN도 성능 개선에 실패했다. 임상적으로는 혈액검사 부재 상황에서 폐혈관 그래프보다 병력·심장 바이오마커 중심이 더 효율적일 수 있음을 시사하며, 학습된 표현이 실제 판별 정보를 담지 못할 가능성에 대한 경고로도 의미가 있다.



### Measurable Majorities Are Not Finitely Axiomatizab (https://arxiv.org/abs/2606.25954)
- **Prior Approaches**: 엄밀한 다수결(majority) 판단을 유한 사회 의사결정 프레임에서 숫자로 재현하려면 ‘응집성(coherence)’ 같은 구조 조건이 필요하지만, 모든 프레임이 유한 가산측도(finitely additive measure)로 표현되진 않는다. 기존 연구는 cancellation 조건과 같은 조합적/측정이론적 축에서 재현 가능성의 경계를 다뤘고, strict majority의 경우도 ‘응집성 위반을 드러내는 최소 길이(incorherence index)’가 유한하지 않을 수 있다는 문제의식이 있었다.

- **Core Contribution**: 이 논문은 Moss and Pedersen(2026)이 제시한 응집성 판정 기준(coherence criterion)이 유한 설정에서 어떤 ‘유한한 bounded finite fragment’로 대체될 수 있는지에 답한다. 결론적으로, 어떤 유한 조각만으로는 프레임의 재현 가능성을 정확히 포착할 수 없으며, Conjecture 5.7을 해결해 incoherence index에 대한 전역 유한 상계가 존재하지 않음을 보인다. 또한 Conjecture B.25가 예측한 middle-layer 패밀리의 존재까지 같이 확립한다.

- **Technical Challenges**: 핵심 기술 난제는, 응집성 위반을 가장 짧게 만드는 구조적 모순이 임의로 길어질 수 있음을 ‘유한 모형 언어’의 문맥에서 강제하는 것이다. 논문은 조합게임 이론(예: trade나 magic square) 대신, 유리 벡터공간의 기하(orthogonality, dimension)를 통해 half-sized voting bloc의 대칭 패밀리를 설계하고, 선형 의존을 Boolean 하이퍼큐브의 zero-sum(완전 균형) 조건으로 환원해 짧은 balanced obstruction을 모두 배제한 최대 프레임을 구성한다.

- **Empirical Impact**: 각 k≥1에 대해 최단 응집성 위반 길이가 정확히 2k+2가 되도록 하는 프레임을 구성함으로써, measurable social decision frames가 해당 언어에서 유한 공리화(finitely axiomatizable)될 수 없음을 논리적으로 종결한다. 즉, strict majorities 최소 논리(Moss-Pedersen minimal logic)에서 soundness/completeess와 결합해 ‘무한 coherence scheme의 힘은 제거 불가’라는 강한 메시지를 제공한다. 결과는 정량화 재현 가능성의 구조적 측면에서, 유한 구성만으로 포착되지 않는 복잡성이 있음을 분야 전반에 명확히 각인시킨다.



### Explainable Control Framework (XCF) based on Fuzzy Model-Agnostic Explanation and LLM Agent-Supported Interfac (https://arxiv.org/abs/2606.25941)
- **Prior Approaches**: 기존 제어는 수학적으로 엄밀하지만 복잡한 설계(예: fuzzy model-based control, sliding mode, backstepping)로 인해 내부 동작을 이해·점검하기 어렵다는 한계가 있었다. 데이터 기반 제어와 강화학습은 성능과 적응성은 높이지만 closed-box가 되기 쉬워 explainability가 약해지고, 많은 XAI 기법은 주로 정적 input-output 관계만 설명해 closed-loop 동역학을 충분히 반영하지 못한다. 또한 기존 explainable control 방법은 특정 제어 구조에 종속되거나, 제어·XAI 지식이 부족한 사용자가 결과를 해석·활용하기 어렵다는 universality와 accessibility 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 닫힌 고리(closed-loop) 제어에서 컨트롤러가 제어 입력을 어떻게 결정하는지 설명하는 explainable control framework(XCF)를 제안한다. 핵심은 model-agnostic 설명기를 만들고, 필요하면 닫힌 고리 응답의 동역학 정보를 사용해 국소 설명의 정합성(behavioral fidelity)을 개선한다는 점이다. 또한 설명 알고리즘으로 hierarchical fuzzy model-agnostic explanation for control systems(HFMAE-C)를 도입해 IF-THEN 규칙과 state salience로 의사결정 논리와 상태 기여도를 계층적으로 제시한다.

- **Technical Challenges**: 주요 technical challenge는 closed-loop에서 “설명기”의 입력-출력 근사 성능만으로는 실제 컨트롤러가 유발하는 궤적·응답 일관성을 보장하기 어렵다는 점이다. 이를 해결하기 위해 HFMAE-C는 universe·local·domain 수준으로 퍼지 로직 기반 대리모델을 학습하고, local 설명 단계에서 시스템 반응 오차를 최소화하는 response-assisted refinement(궤적 비교 기반)를 선택적으로 수행한다. 추가로 domain 수준에서는 단순 weight aggregation의 표현 한계를 줄이기 위해 rule aggregation을 제안해 입력에 따라 로컬 설명기가 달리 기여하도록 한다.

- **Empirical Impact**: inverted pendulum과 Turtlebot 장애물 회피를 대상으로 시뮬레이션 사용자 실험 및 정량 비교를 수행해 XCF의 효과를 보였다. inverted pendulum에서는 계층적 설명과 LLM 기반 자연어 해석이 컨트롤러 동작을 이해 가능한 형태로 전달함을 보여주고, Turtlebot에서는 SHAP·LIME 같은 mainstream model-agnostic 방법과 비교해 설명 품질을 개선할 수 있음을 확인했다. 아울러 LLM agent-supported 사용자 인터페이스는 사용자의 설명 의도에 맞춰 알고리즘을 선택하고 보고서를 생성·상담까지 제공해 접근성을 높인다는 의미가 있다.



### Overview of HIPE-2026: Person-Place Relation Extraction from Multilingual Historical Texts (https://arxiv.org/abs/2606.25935)
Comments:
          Condensed Overview of CLEF-HIPE-2026 Shared Task Results

- **Prior Approaches**: HIPE-2020과 HIPE-2022는 다국어 역사 문서에서 named entity recognition과 linking에 초점을 맞췄다. 다만 동일 문맥에 함께 등장한 사람-장소 쌍을 ‘관계’로 오판할 수 있고, OCR 잡음과 문체 변화 속에서 간접 단서와 시간적 뉘앙스를 함께 추론하는 단계는 상대적으로 부족했다. 또한 이전 벤치마크들은 문헌 도메인/기간이 바뀔 때 성능이 어떻게 흔들리는지까지는 제한적으로만 다뤘다.

- **Core Contribution**: HIPE-2026은 ‘누가 어디에 있었는가, 그리고 언제였는가’를 temporally grounded person–place relation extraction으로 확장하며 두 관계를 정의한다: at(문서 출판일 이전의 어느 시점에 해당 장소에 존재)와 isAt(출판일 전후의 시간 지평에서의 동시적 존재). 특히 at의 3가설 분류(TRUE/PROBABLE/FALSE)와 isAt의 이진 분류를 함께 평가해, 표면 증거만이 아니라 문맥 일관성을 바탕으로 한 추론(예: 간접 단서)을 정량화하려는 설계가 핵심이다. 평가 프로파일도 정확도뿐 아니라 계산 효율성과 도메인 일반화까지 포함해, 문화유산 규모 처리 요구를 직접 반영한다.

- **Technical Challenges**: 역사 신문/문학 텍스트는 언어 변이가 크고 OCR 잡음이 섞여 있으며, 사람-장소 관계 근거가 문장 밖에서 간접적으로(교차문장, 대명사/서명 등) 드러나는 경우가 흔하다. 게다가 at=PROBABLE처럼 ‘명시적 문장’이 없더라도 담화가 성립하도록 가정해 추론해야 하므로, 잡음 환경에서의 증거 가중과 시간 추론이 동시에 필요하다. 이를 위해 참가 팀들은 LLM의 in-context learning부터 few-shot/프롬프트 설계, 다국어 인코더 파인튜닝, LoRA 같은 parameter-efficient fine-tuning, 의존구문 그래프/특징 기반 모델, 규칙 기반 휴리스틱, 멀티 에이전트/앙상블 등 다양한 경로로 time horizon 일치와 불일치 제약을 다루려 했다.

- **Empirical Impact**: 17개 팀이 40회 이상 제출한 결과, 최첨단 대형 언어모델부터 경량 분류기까지 폭넓은 전략이 등장했고, 정확도·효율·강건성 사이의 트레이드오프가 명확히 드러났다. 특히 정확도 프로파일(다국어 신문)과 효율 프로파일(모델 크기/파라미터 메타데이터 반영), 그리고 surprise-domain 일반화 프로파일(초기 근대 프랑스 문학/역사 저술)로 성능 변동을 분리해 보여준다. 본 캠페인은 대규모 역사 문서 처리에서 ‘관계 추출’이 단순 동시출현 탐지를 넘어 시간적 근거 추론까지 요구한다는 점을 실증적으로 정리한 벤치마크로 의미가 있다.



### Enhancing Brain MRI Anomaly Detection and Reasoning with ROI Rethink and Synthetic Data (https://arxiv.org/abs/2606.25894)
- **Prior Approaches**: 기존 의료 비전-언어 모델은 한 번의 추론(single-pass inference)로 진단을 내리는 경우가 많아, 어떤 ROI(관심영역)가 근거인지 시각적 근거 제공이 약했다. 이로 인해 결과를 감사(audit)하기 어렵고, 정상 영상에서도 허위 소견(hallucination)이 나올 위험이 커 임상 활용도가 제한된다. 또한 뇌 MRI는 병변 위치가 감별진단에 핵심이지만, 주로 closed-ended 정확도 중심 벤치마크에 묶여 open-ended 일반화와 OOD 성능이 충분히 다뤄지지 않았다.

- **Core Contribution**: BrReMark(Brain Rethink via ROI Marking)는 뇌 MRI 진단을 2턴 시각 대화로 바꾸고, 가설-표식-검증의 추론 흐름을 명시적으로 ROI 바운딩박스로 고정한다. 1턴에서 이상 후보와 bounding box를 제시한 뒤 표시된 이미지를 재검토하여 결론을 verify하도록 설계해, 결과의 공간적 근거를 사람이 확인 가능하게 만든다. 학습 단계에서는 SFT로 “Mark-and-Rethink” 궤적 포맷을 가르치고, 이후 GRPO 기반 RL로 진단 서술과 근거 일치까지 최적화한다.

- **Technical Challenges**: 핵심 난제는 (1) ROI 위치·진단 텍스트·형식 태그를 동시에 만족시키는 open-ended 학습 신호를 설계하는 것과 (2) 드문 병변(희귀 병리) 때문에 OOD에서 허위 소견이 늘지 않게 하는 것이다. BrReMark는 localization 정확도, 의미적 진단 타당성, 안전성을 묶은 복합 보상을 만들고, modality gating·hallucination penalty·synthetic masking 등 reward gatekeeper로 진단 신뢰성을 제어한다. 더불어 SynthSeg 기반 병리 합성에 domain randomization을 적용하되 SFT에는 합성을 섞지 않고 RL에서만 공간 타게팅에 활용해 OOD 강건성을 끌어올렸다.

- **Empirical Impact**: 내부 벤치마크에서 BrReMark는 base 모델 대비 mAP50을 0.74%에서 37.54%로 크게 끌어올렸고, Clinical F1 21.57%, diagnostic accuracy 45.26%를 기록했다. NOVA OOD(희귀 병리 장꼬리)에서도 경쟁적인 성능을 보이며 false positives를 SOTA 대비 45.7% 줄여 드문 병리에 대한 허위 소견이 감소했음을 시사한다. ablation 결과로도 localization reward·합성 병리 RL·semantic correctness reward가 성능과 안전성 향상에 핵심임이 확인됐다.



### AI-Assisted Computational Reproducibility on the FABRIC Testbed (https://arxiv.org/abs/2606.25879)
- **Prior Approaches**: 기존 재현 연구는 주로 같은 수치 재현(결과 레벨)을 목표로 했고, 환경 명세 누락·하드웨어 가정·의존성 노후화 같은 장벽 때문에 팀 간 재현성은 여전히 어렵다는 문제의식이 컸습니다. 연구용 테스트베드는 Chameleon·CloudLab처럼 인프라를 제공하지만, 논문과 그림만으로 무엇을 실행해야 하는지까지 자동으로 파악하긴 힘듭니다. 한편 LLM coding assistants는 논문 읽기·코드 생성·디버깅에는 강하지만, 테스트베드 실행 환경이 없어 과학적 타당성까지 판정하긴 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 FABRIC(프로그램 가능한 테스트베드)과 LLM 코딩 어시스턴트 LoomAI(영국문 그대로) 조합으로, 여러 도메인의 공개 실험을 재현할 때 핵심을 ‘결론 재현’까지 포함하도록 체계화했습니다. 특히 수치가 달라도 같은 과학적 주장(결론)이 지지되는지 여부를 지원/부분지원/미지원으로 분류하는 루브릭을 제안합니다. 또한 AI가 실행까지 맡는 흐름에서 어디까지 가속되고 어디서는 인간 도메인 감독이 필요한지 정량·정성으로 정리합니다.

- **Technical Challenges**: 주요 기술 과제는 (1) 논문 방법을 테스트베드 네이티브 아티팩트로 변환하고, (2) 실행 순서·데이터 의존성 같은 워크플로 자체가 불명확한 경우를 해결하는 데 있습니다. 저자들은 4단계 방법(명세 작성→AI 적응 구현→FABRIC 실행→결론 검증)으로 이를 완화했지만, 특히 워크플로가 명확히 정의되지 않은 분석 단계에서는 execution order와 데이터 의존성을 사람이 정리해줘야 했습니다. Genomics 사례에서 보이듯, AI는 실행 기계화(환경·의존성·출력 포맷)에 강하지만 ‘분석 스키마’가 없으면 자율적으로 완성도를 보장하기 어렵다는 점이 드러납니다.

- **Empirical Impact**: FABRIC에서 3개 케이스(네트워킹 BBR 계열, CPU 전용 MPI 기반 LAMMPS 스케일링, 게놈 분석 파이프라인)를 재현하며, AI-보조 워크플로가 재현 노력은 대략 4–6배 줄였다고 보고합니다. 결론 레벨 검증 결과 BBR과 LAMMPS는 기존 논문의 과학적 내러티브를 대체로 지지했고(수치 차이는 환경·버전 차이로 허용), genomics는 데이터 예치가 불완전한 경우에 일부 결론만 독립 검증 가능했습니다. 종합하면, 테스트베드 실행과 결론 검증을 결합한 접근이 향후 연구 재현 자동화의 실용적 방향을 제시한다는 의미가 있습니다.



### AutoRelAnnotator: Calibrated Model Cascades for Cost-Efficient Relevance Evaluation in Sponsored Search (https://arxiv.org/abs/2606.25871)
Comments:
          Accepted at E-commerce workshop, SIGIR 2026

- **Prior Approaches**: 기존에는 인간 라벨링 비용·지연이 커서 대규모 relevance annotation이 병목이 됐고, off-the-shelf LLM은 도메인 특화 과업에서 정확도가 낮았다. LLM cascade 같은 라우팅 기법은 비용을 줄이지만, 결국 구성 모델들의 정확도 상한을 넘기 어렵다는 한계가 있었다. 또한 기존 보정(calibration)은 글로벌 방식 위주라 클래스별로 다른 미스캘리브레이션을 충분히 반영하지 못했다.

- **Core Contribution**: 이 논문은 정확도와 비용을 분리해 최적화하는 calibrated model cascade를 제안한다. 먼저 도메인별 fine-tuning으로 높은 정확도의 분류기(예: 87–89%)를 만든 뒤, per-class isotonic calibration을 결합해 3단계 라우팅(Cross-Encoder→Gemma-2B→LLaMA-8B)을 수행한다. 그 결과 cascade는 비용 절감과 정확도 보존을 동시에 노리되, 보정으로 소폭의 추가 이득까지 확보한다.

- **Technical Challenges**: 핵심 과제는 라우팅에 쓰는 confidence가 잘못 보정(miscalibrated)돼 있으면 “안전한” 임계값 설정이 무너지는 점이었다. 논문은 클래스별 예측 분포가 다른 현상을 관찰하고, per-class isotonic regression으로 클래스 조건부 보정을 학습해 라우팅 임계값(threshold)이 실제 정확도와 정렬되도록 했다. 또한 classification head 기반 분류 확률을 사용해 생성형 LLM의 프롬프트 취약성·캘리브레이션 불안정성을 피했다.

- **Empirical Impact**: 실험에서 fine-tuning은 off-the-shelf LLM 대비 약 20%p 격차를 메웠고, cascading은 정확도에 거의 영향을 주지 않으면서 compute cost를 절반으로 줄였다. per-class isotonic calibration은 cascade 정확도를 88.5%→89.1%로 끌어올리며, 강한 글로벌 calibration baseline 대비 +0.6%p의 통계적으로 유의미한 이득(p<0.05)을 보였다. Q3 2024 이후 6개 오프라인 use case에서 150M+ annotation을 처리했고, 라벨링/분석 처리 리드타임을 5일에서 1–3시간으로 단축해 검색·광고 ML 워크플로의 실험 속도를 높였다.



### Color Matters: Trigger Color Affects Success in Federated Backdoor Attacks (https://arxiv.org/abs/2606.25858)
Comments:
          Accepted at the IEEE/IFIP DSN Workshop on Dependable and Secure Machine Learning (DSML), 2026

- **Prior Approaches**: 기존 연합학습 백도어 연구는 corner patch나 sticker 같은 고정된 합성 트리거에 의존하는 경우가 많아, 트리거 외형 변화(특히 색)가 공격 성패에 미치는 영향을 충분히 분리해 보지 못했다. 방어 쪽도 Krum, Trimmed Mean, FLTrust, FLAME, MultiKrum처럼 집계/필터링 중심으로 다루면서 트리거의 시각적 속성은 거의 고정값으로 취급해왔다. 그 결과, 동일한 의미의 트리거라도 외형이 달라지면 방어 효과가 달라질 수 있다는 불확실성이 남아 있었다.

- **Core Contribution**: 이 논문은 semantics-driven 백도어에서 트리거 객체(mask, sunglasses)는 유지하되 트리거 색만 바꿔, 공격 성공률(ASR)에 대한 ‘색의 역할’을 통제 실험으로 분리해 제시한다. CelebA의 헤어 컬러 4클래스 설정에서 백도어 파이프라인, 배치/예산, 타깃/소스 매핑을 고정하고 흑/백 트리거 변형만 비교해 색-타깃 정렬(예: 흰색은 blond 타깃)에 따른 성능 변화를 정량화한다. 또한 표준 poisoning 목표뿐 아니라 SABLE 기반 방어 인지 목표와 robust aggregation 하에서도 같은 경향이 유지되는지 확인한다.

- **Technical Challenges**: 트리거 색만 바꾸면 관찰 차이가 ‘시각적 단서 변화’에서 오지만, 연합학습에서는 비IID 분포와 집계가 로컬 학습 신호의 일관성과 보존 여부를 함께 흔든다. 이 때문에 단순히 색을 바꾼 트리거를 적용하는 것만으로는 원인 규명이 어렵고, SABLE처럼 clean 분류 손실, triggered 타깃 손실, penultimate 표현공간 분리 loss, 악성 업데이트 드리프트를 제약하는 정규화까지 포함해 색 효과가 방어/집계 하에서 지속되는 조건을 설계해야 한다. 저자들은 동일 소스 이미지에 대해 image ID로 트리거 변형을 매칭해 비교 가능성을 높였고, 방어 실험에서는 MultiKrum 집계를 사용해 색 효과가 필터링을 통과하는지까지 검증했다.

- **Empirical Impact**: 결과적으로 트리거 색 변화만으로 ASR이 크게 달라졌고, clean accuracy는 비교적 안정적으로 유지되는 패턴이 관찰됐다(예: blond 타깃에서는 white 트리거가 더 높게 나타남). 또한 SABLE+MultiKrum처럼 더 강한 설정에서도 ‘타깃 컬러와 시각적으로 더 가까운 트리거 색이 더 효과적’이라는 방향성이 유지되어, 색이 단순한 외형 변수가 아니라 학습 가능성과 집계 생존에 의미 있게 관여함을 시사한다. 즉, 방어 평가를 단일 트리거 색에만 의존하면 위험을 과소평가할 수 있으며, 의미 기반(semantic) 백도어 벤치마킹에서 색을 first-class 변수로 다뤄야 한다는 실무적 함의를 제공한다.



### Semantic Consistency Policy Optimization for Reinforcement Learning of LLM Agents (https://arxiv.org/abs/2606.25852)
Comments:
          16 pages, 7 figures, 5 tables. Under review at EMNLP 2026

- **Prior Approaches**: 그룹 기반, value-free RL은 여러 롤아웃을 묶고 그룹 내부 상대적 advantage로 정책을 업데이트한다. 하지만 다중 턴 환경의 희소 보상에서는 step별 credit이 롤아웃의 최종 성공/실패에 그대로 종속되어, 의미적으로 비슷한 중간 행동도 실패 여부에 따라 상반된 신호를 받는 문제가 생긴다. GiGPO처럼 step-level 대안을 넣더라도 핵심 원인은 남아 semantic credit inconsistency가 계속 발생한다.

- **Core Contribution**: 이 논문은 Semantic Consistency Policy Optimization (SCPO)로, 실패 롤아웃 내부의 step이 성공 롤아웃(같은 그룹의 success sibling)과 의미적으로 얼마나 “새로운 진전”을 만들었는지를 기준으로 step-level credit을 복구한다. value-free reward shaping 형태라서 critic 학습이나 추가 롤아웃 없이 기존 그룹 기반 에이전트 RL 파이프라인에 플러그인처럼 끼울 수 있다. 성공 sibling을 기준으로 “새 포지션까지 전진한” 매칭만 보상해, 실패 롤아웃에 숨어 있는 부분적으로 맞는 행동을 살리는 방향으로 학습 신호를 재구성한다.

- **Technical Challenges**: 핵심 난제는 희소한 단말 보상에서 step별 진전을 안정적으로 분리하는 것인데, 단순 매칭이나 시간순 credit은 템플릿/반복 행동에 보상을 낭비할 수 있다. SCPO는 frozen cross-encoder로 실패 step과 성공 sibling step의 semantic similarity를 계산하되, 임계값을 넘고 아직 credit되지 않은 reference 위치로 “단조(monotonic)하게” 전진하는 경우에만 soft credit을 부여한다. 또한 할당 순서를 temporal order가 아닌 order-agnostic competition으로 바꿔, 앞쪽의 덜 구별되는 템플릿에 한정된 reference 포지션이 먼저 소모되는 실패를 줄인다.

- **Empirical Impact**: ALFWorld와 WebShop에서 SCPO는 GiGPO를 감싼 래퍼로서 성능을 끌어올리며, 1.5B에서 ALFWorld 성공률 93.7±4.1%, WebShop 태스크 성공 74.8±2.0%를 달성해 각각 +7.0/+9.8p 수준의 향상이 보고됐다. 특히 Look, Cool, Pick2 같은 어려운 multi-step 계열에서 gains가 집중되며, 7B에서도 마진은 줄어들지만 강력한 베이스라인과의 경쟁력을 유지한다. 설계 요소(순서 재배치, monotonic credit, 긴 성공 sibling 사용)를 제거한 ablation은 성능 하락을 보이며, semantic progress와 repetition을 구분하려는 SCPO의 의도가 실험적으로 지지된다.



### Edges Before Embeddings: A Confidence-Aware Blur Gate for Vision-Language Pipelines (https://arxiv.org/abs/2606.25838)
Comments:
          7 pages, 2 figures, 6 tables. Preprint

- **Prior Approaches**: 기존에는 라플라시안 분산 같은 무참조 blur 점수를 휴리스틱으로 쓰거나, 대규모 복원 네트워크(deblur)가 블러를 “고쳐서” 해결하려는 접근이 주를 이뤘습니다. 하지만 휴리스틱은 텍스처/저질감과 블러를 섞어 오판하고 도메인별 튜닝이 어렵고, 복원 네트워크는 게이트로 쓰기엔 너무 무겁습니다. 또한 연속 품질 점수를 예측하는 IQA 모델들은 이진 sharp/blur 라우팅용 abstention 문제로 바로 전환하기가 쉽지 않았습니다.

- **Core Contribution**: 이 논문은 비전·문서 파이프라인 앞단에서 이미지를 sharp/blur/uncertain로 분류해 후속 OCR·VLM 호출을 제어하는 CPU 친화형 blur quality gate인 MagikaDocumentFromPixel(blur 검출 게이트)을 제안합니다. 전체 성능은 기존 모델 골격이나 새 loss가 아니라, 실험으로 찾은 학습/입력 레시피와 라우팅 규칙, 그리고 Edge Prior Module(EPM)로 크게 끌어올립니다. 특히 선택적 예측(selective prediction) 관점의 confidence-aware routing을 통해 uncertain일 때는 abstain으로 “안전하게” 넘기게 만듭니다.

- **Technical Challenges**: 핵심 과제는 (1) blur를 빠르고 CPU에서 안정적으로 판별하고 (2) uncertain를 실제 배포 상황에 맞게 라우팅해 비용 대비 정확도를 최적화하는 데 있습니다. 저자들은 실험 스윕을 통해 입력 해상도가 우세한 레버임을 확인하고, backbone 용량은 384 px 이상에서만 의미 있게 이득을 준다고 정리합니다. 또한 EPM으로 Laplacian magnitude(라플라시안 크기) 보조 채널을 RGB에 4번째 입력으로 결합해 고주파 붕괴라는 고전 휴리스틱의 “증거”를 네트워크가 직접 보게 했고, max-softmax 기반 threshold τ를 배포-time product knob으로 둬 재현성 없는 캘리브레이션 네트워크 의존을 줄였습니다.

- **Empirical Impact**: GoPro Large 단일 seed·단일 motion-blur 분포에서, MobileNetV3-Large + EPM + 384x384 학습과 5-scale test-time augmentation(TTA) 조합은 F1=0.9803(AUC 0.9989), 17 MB ONNX로 약 7 ms 내외의 속도를 보였습니다. 동일 환경 기준 고정 해상도 베이스라인(F1=0.9672) 대비 +1.31p 개선이며, EPM이 단일 효과로 가장 큰 레버로 확인됐습니다. 다만 결과는 다른 blur 유형(디포커스·저조도·압축·스캐너 skew)과의 교차 데이터셋 검증, ECE/신뢰도 도표 같은 정량 캘리브레이션, τ 스윕·다중 seed 통계가 제한이라 추가 연구가 필요하다고 명시했습니다.



### MiniOpt: Reasoning to Model and Solve General Optimization Problems with Limited Resources (https://arxiv.org/abs/2606.25832)
Comments:
          20 pages, 9 figures, 11 tables, project: this https URL

- **Prior Approaches**: 최적화 문제를 LLM이 자연어에서 모델링·코드 생성까지 수행하는 기존 연구는 OptiBench, LLMOPT, Text2Zinc 계열처럼 큰 SFT 데이터나 추론 주석, 중간 단계 검증을 많이 요구하는 경향이 있다. 또한 생성된 결과의 정합성 검증이 비용이 커서 반성·디버깅 같은 과정을 늘리거나, 검증이 출력 단계에 주로 한정되는 RL 정렬의 제약이 남아 있었다.

- **Core Contribution**: 이 논문은 MiniOpt라는 RLVR 강화학습 프레임워크로, “reasoning-to-model-and-solve” 패러다임을 통해 최적화 추론을 먼저 구조화된 최적화 모델(5요소 튜플)로 만들고, 이어서 실행 가능한 solver 코드를 생성하도록 학습시킨다. 핵심은 OptReward라는 계층형 리워드로, 모델링 수식의 완결성(구조)과 solver 실행 정확도(수치)를 함께 평가해 전문가 시연 없이도 정책 학습이 가능하게 한 점이다.

- **Technical Challenges**: 기존 RLVR이 잘 되려면 (1) 학습 신호를 안정적으로 주는 검증 가능 리워드와 (2) 작은 모델이 형식 오류·리워드 해킹으로 붕괴하지 않도록 탐색을 제어하는 것이 중요하다. MiniOpt는 CoT를 <think>/<answer>로 강제하고, 5요소 “구조 존재 여부”로 구조 완결성 점수를 계산하며, Pyomo 코드 실행으로 수치 정확도를 검증해 검증 비용을 낮춘다; 또한 OptGRPO에서 KL 페널티를 제거하고 비대칭 importance-weight clipping을 적용해 탐색 효율과 RL 안정성을 동시에 끌어올린다.

- **Empirical Impact**: MiniOpt-3B는 8개 벤치마크(다양한 최적화 유형·시나리오)에서 평균 Solving Accuracy(SA)를 크게 끌어올리며, 파라미터 10B 미만 구간에서 가장 높은 평균 성능을 보인다. 10B 이상 대비해서도 경쟁력 있는 결과를 내고, SA와 토큰/비용 관점에서 경험적 Pareto frontier에 위치한다고 보고된다. 이는 최적화 특화 리워드 설계와 RLVR이 컴팩트 모델의 “최적화 일반화”를 실용적으로 확장하는 경로가 될 수 있음을 시사한다.



### SARA: Unlocking Multilingual Knowledge in Mixture-of-Experts via Semantically Anchored Routing Alignmen (https://arxiv.org/abs/2606.25821)
- **Prior Approaches**: Sparse MoE는 효율적으로 늘어나는 파라미터를 통해 언어·도메인 전문화를 달성하지만, 성능은 라우터가 토큰을 적절한 expert로 보내는 능력에 크게 좌우됩니다. 기존 다국어 개선은 주로 continual pre-training이나 instruction tuning, 혹은 load balancing/프루닝 같은 효율 최적화에 초점이 있었고, 언어 간 expert 선택의 기계적 불일치(cross-lingual routing divergence)는 직접 해소하지 못했습니다.

- **Core Contribution**: SARA(Semantically Anchored Routing Alignment)는 고자원 언어에서 얻은 expert 라우팅 분포를 ‘semantic anchor’로 삼아 저자원 언어의 라우팅을 정렬하는 프레임워크입니다. 출력 로짓을 증류(distillation)하는 대신 MoE 내부의 라우팅 확률 분포를 symmetric Jensen-Shannon(JS) 발산으로 맞춰, 언어가 달라도 동일 의미에 대해 비슷한 expert 활성 경로가 나오도록 합니다.

- **Technical Challenges**: 핵심 기술적 난제는 저자원 입력이 표면적 어휘 차이 때문에 라우터가 다른 expert로 보내지면서 의미는 같아도 라우팅이 갈라지는 점입니다. SARA는 (1) 고자원에서 [boxed{}로 검증된 정답 기반 정렬 데이터의 저자원 번역을 만들고, (2) 고자원 anchor의 토큰-레이어 라우팅 분포를 priors로 추출한 뒤, (3) 저자원 입력의 라우팅 분포를 해당 priors에 JS divergence로 끌어당기는 다단계 학습을 수행합니다. 또한 next-token 학습 손실과 Switch Transformers의 load balancing을 함께 써 expert 붕괴를 막습니다.

- **Empirical Impact**: 2개 LLM과 5개 저자원 언어, 3개 벤치마크(Global-MMLU, BELEBELE, MGSM)에서 SARA가 standard instruction tuning 대비 성능을 개선했으며, 예로 Qwen3-30B-A3B에서 +0.8%, Phi-3.5-MoE-instruct에서 Global-MMLU +1.2%를 보고합니다. 분석 결과 FFT 같은 일반 fine-tuning은 라우팅 불일치를 부분적으로만 줄이는 반면, SARA는 레이어 7~34 구간의 JS divergence를 거의 0에 가깝게 낮춰 라우팅 정렬이 실제 병목을 해결함을 보여줍니다. 번역 품질(GPT-5 mini vs nano)과 학습 단계에 따른 동학 분석도 포함돼, 더 깨끗한 의미 감독과 추가 epoch에서 ‘기계적 일관성’을 누적해 더 높은 수렴 성능 상한을 만든다는 점을 시사합니다.



### Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges for LLM Adversarial Evaluation (https://arxiv.org/abs/2606.25782)
Comments:
          13 pages, 5 figures, Accepted into ICANN2026

- **Prior Approaches**: 기존 안전 평가는 LLM-as-a-judge처럼 디코더 기반 안전 심판을 호출해 유해성을 판정하는 방식이 주류였지만, 대규모 배치에선 비용·지연이 커진다는 한계가 있습니다. 반면 LlamaGuard 같은 파인튜닝 디코더 분류기는 상대적으로 빠르지만, 분포 변화나 새로운 공격 패턴에 취약해 일반화가 흔들릴 수 있습니다. 또한 지금까지는 ModernBERT 계열의 최신 인코더 분류기를 유해 출력 판정에서 LLM 판정자들과 체계적으로 비교한 연구가 부족했습니다.

- **Core Contribution**: 이 논문은 ModernBERT 계열 인코더 분류기(ModernBERT/Ettin)를 대화형 (user-model) 응답에서 유해 출력을 탐지하는 안전 가드레일로 적용하고, LLM-as-a-judge 및 파인튜닝 디코더 안전 모델들과 OOD 홀드아웃에서 직접 비교합니다. 7개 안전 심판이 라벨링한 데이터를 majority-voting으로 집계해 인코더를 파인튜닝한 뒤, JailbreakBench와 AILuminate 라벨을 가진 홀드아웃에서 F1, FNR 등을 기준으로 성능을 확인합니다. 특히 공격 기법(단일 턴, decomposition, escalation, context manipulation)별로 LLM 판정자들과의 일치/불일치를 분석해 “언제 인코더가 대체 가능한지”를 가이드합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 디코더 심판의 라벨을 섞은 majority-vote 신호에 인코더가 제대로 학습하느냐와 (2) 다턴 jailbreak처럼 유해 의도가 중간 단계에 분산될 때 인코더가 final (prompt,response)만 보고도 탐지할 수 있느냐입니다. 이를 위해 질문 단위로 group partition을 적용해 데이터 누수를 막고(같은 질문의 학습·평가 응답이 겹치지 않게), judge들 간 불일치가 큰 영역은 고신뢰(high-confidence)로 분리해 라벨 잡음의 상한을 함께 점검합니다. 결과적으로 decomposition 같은 분산형 공격에서 false negative가 커지는 경향은 확인되지만, 모델 크기(150M→400M) 확대로 일부 완화되는 패턴을 보였습니다.

- **Empirical Impact**: 홀드아웃에서 Ettin 인코더는 AUROC가 높고(예: Ettin-150M 0.939, Ettin-400M 0.929), 특히 안전-critical 지표인 FNR에서 LlamaGuard-4 및 일부 LLM 판정자 대비 더 균형 잡힌 운영점을 보였습니다. 또한 고신뢰 subset(심판 합의가 강한 구간)에서는 F1이 Ettin-150M 0.800, Ettin-400M 0.894로 크게 상승해 라벨 불확실성이 병목임을 시사합니다. 프로덕션 관점에서도 Ettin-150/400은 LlamaGuard-4-12B 대비 처리량과 지연에서 압도적인 이점을 보여(수백 CPS 수준 vs ~1 CPS대, median latency 수 ms vs 수백 ms) 저비용·저지연 가드레일로의 실무적 의미가 큽니다.



### Space-Efficient Language Generation in the Lim (https://arxiv.org/abs/2606.25777)
Comments:
          Accepted at COLT 2026

- **Prior Approaches**: 기존 언어 생성/학습 이론은 보통 충분한 자원을 가정하거나, 산출물의 품질(환각 금지, 누락 허용치)과 메모리 제약을 동시에 날카롭게 다루지 못했다. 특히 스트리밍 환경에서 제한된 메모리로 목표 언어를 얼마나 정확히 ‘생성 범위’로 보장할 수 있는지에 대한 정량 경계가 부족했다.

- **Core Contribution**: 이 논문은 공간 효율의 최소 제약 아래 ‘limit에서의 language generation’을 자원 인식(resource-aware) 관점으로 정식화한다. 학습자는 목표 언어 K에서 나온 적대적( adversarial ) 양성 스트림을 관찰한 뒤, hallucination-free로 L ⊆ K를 만들되 K에서 최대 Δ개의 문자열만 생략하도록 수렴하는 것을 목표로 한다. 또한 가설 클래스를 DFA 상태 수가 s인 언어 집합 C_{s,k}로 제한해, 자원에 따른 달성 가능성을 이론적으로 분해한다.

- **Technical Challenges**: 핵심 난제는 스트리밍에서 메모리가 제한될 때, 보이는 양성 정보만으로도 환각 없이(즉 L ⊆ K) 누락 허용치 Δ까지 제어하며 수렴하는 전략을 설계하는 것이다. 저자들은 poly(s,k) 공간의 스트리밍 알고리즘을 제안해 Δ = O(k^{2s-2})의 generation gap을 달성하며, 길이 2s-1 이상인 K의 모든 문자열을 포착함을 보장한다. 더 나아가 표준 communication complexity 문제로부터 하한을 환원해, Δ ≤ k^{(1-ε)s}를 얻으려면 k^{Ω(εs)} 메모리가 필요함을 보여 경계가 거의 맞물림을 입증한다.

- **Empirical Impact**: 실험보다는 수학적 정당화에 초점을 둔 결과로, polynomial-space 수준에서는 일정 generation gap 안에서 포착을 보장하지만, exponential-space에서만 목표 K를 정확히 식별(exact identification)할 수 있음을 ‘sharp transition’으로 제시한다. 이는 제한된 메모리로 언어 생성 품질을 보장하려는 이론 설계에서, 필요한 자원과 성능(생성 갭·누락 규모) 사이의 구체적 트레이드오프를 제공한다. 결과적으로 스트리밍·메모리-bounded 학습에서 이론적 한계와 설계 방향을 동시에 명확히 해준다.



### OncoSynth: Synthetic data generation for treatment effect estimation in oncology (https://arxiv.org/abs/2606.25762)
- **Prior Approaches**: 항암/정밀의료에서는 암 레지스트리나 EHR 같은 대규모 임상데이터가 치료의 효과·안전성을 추정하는 데 핵심이지만, 거버넌스·법·윤리 및 재식별 위험 때문에 데이터 공유가 제한적인 경우가 많다. 그래서 합성데이터 생성이 대안으로 부상했으며, VAE·GAN·diffusion 같은 생성모델이 환자 공변량과 생존 결과의 통계적 분포를 복제하려 한다. 하지만 기존 합성데이터 방법은 covariates–treatment–outcome을 ‘동시에’ 혹은 ‘결합 분포’ 관점으로 학습하는 경향이 있어, 치료를 인과적 개입으로 취급하지 못해 치료효과 추정에 체계적 편향을 만들 수 있다.

- **Core Contribution**: 이 논문은 인과관계를 보존하는 생성 프레임워크 OncoSynth를 제안해, 합성 코호트로도 population-level 및 patient-level 치료효과를 정확히 추정할 수 있게 한다. OncoSynth는 환자 특성이 치료 배정에 영향을 주고, 치료가 생존에 영향을 준다는 임상적 인과 사슬을 분해해(치료 배정 메커니즘, 치료–결과 메커니즘) 이를 유지하도록 설계됐다. 또한 diffusion 기반 생성 과정을 사건의 시간순(순차 생성)으로 맞춰, 데이터가 실제와 같은 정보 흐름을 따르도록 한다.

- **Technical Challenges**: 핵심 기술 난제는 합성데이터가 환자·치료·결과의 ‘모양’은 그럴듯하게 만들되, 치료가 outcome에 미치는 인과적 경로와 정보 누출(leakage)을 동시에 피해야 한다는 점이다. OncoSynth는 먼저 공변량을 tabular diffusion으로 생성한 뒤, 생성된 공변량을 조건으로 치료 배정을 분류기로 모델링하고, 그다음 공변량과 치료를 조건으로 random survival forests로(검열 상태 포함) 생존을 생성하는 순차 구조를 채택한다. 이로써 공변량→치료→생존의 순서를 강제해 기존의 joint 생성에서 발생할 수 있는 미래 결과 정보의 과거 유입 문제를 줄인다.

- **Empirical Impact**: 대규모 폐암(N=37,128)·유방암(N=17,046) SEER 코호트 실험에서 OncoSynth는 원본과 유사한 환자 특성·치료 유병률·사건/검열 비율·생존곡선을 재현하며, 특히 치료효과 추정 유틸리티에서 기존 CTGAN·TabDiff보다 우수함을 보였다. 치료 배정 메커니즘은 propensity score 수준에서 원본과의 일치가 가장 높았고, 치료–결과 메커니즘은 ATE·ITE 및 Qini curve 기반 치료 배치 정책에서 합성-원본 간 불일치를 더 작게 만들었다. 논문은 치료효과 오차를 population-level 최대 66%, patient-level 최대 58%까지 줄여, 데이터 공유가 제한된 정밀의료 환경에서도 신뢰 가능한 근거 생성에 기여할 수 있음을 시사한다.



### Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets (https://arxiv.org/abs/2606.25760)
- **Prior Approaches**: 기존 컴퓨터-사용 에이전트 연구는 GUI 클릭 정합에서 불확실성을 다루더라도, 보통 특정 VLM-벤치마크-인터페이스 조합에 한정돼 사후 UQ(uncertainty quantification) 선택이 다른 조건에서도 유지되는지 불명확했습니다. 또한 open-weight와 closed-source 환경을 같은 프로토콜로 비교하는 연구가 적어, 로그it·hidden state·attention 같은 관측 가능 신호 유무가 UQ 순위에 미치는 영향을 정량화하기 어려웠습니다.

- **Core Contribution**: 이 논문은 단일-step GUI grounding에서 사후 UQ가 “어떤 배치(에이전트/데이터셋/관측 가능한 인터페이스)”로 옮겨 갈 때도 오류 순위가 유지되는지 교차-레짐 일반화를 체계적으로 측정하는 Argus를 제안합니다. Argus는 open-weight 4개 VLM 에이전트와 4개 데이터셋에 대해 27개 방법을 2,727개 스코어로, 내부 신호를 얻기 어려운 closed-source는 3개 벤더×동일 데이터셋 조건에 대해 8개 방법을 API-only 패널로 구성해 비교할 수 있게 합니다.

- **Technical Challenges**: 핵심 과제는 UQ “스코어”가 아니라 “방법 순위”가 레짐 전환에도 안정적인지 검증하는 것으로, 이를 위해 오류 탐지·선택적 실행·캘리브레이션·miss-severity 랭킹·공간 click-영역(동심 원/원형 구간) 커버리지까지 다목적 지표로 평가를 분리했습니다. 또한 closed-source에서는 로그it/hidden/attention이 없으므로 공통 API-호환 방식으로 비교 가능한 방법 부분만 남기고, 관측 불가능 신호 기반 방법은 동일 공식의 대체(프록시)로 통합 패널을 구성해 비교의 공정성을 확보했습니다.

- **Empirical Impact**: 결과적으로 UQ 방법 순위는 고정된 모델 내에서는 데이터셋 전환에도 비교적 안정적이지만, 모델 클래스 전환과 관측 인터페이스 변화(API-only)에서는 안정성이 크게 저하되는 “selective transfer” 양상이 나타났습니다. open-weight에서는 hidden-state·density(예: SEP/SAPLMA 계열)가 전반적으로 가장 안정적인 편이지만, closed-source로 갈수록 평균 순위 이전 성능이 거의 0에 가까워져 타겟 환경에서 재캘리브레이션/재랭킹이 필요하다는 결론을 제시합니다. 또한 conformal click-disk는 점수만으로 배치 가능한 커버리지를 보장하지 못해, 플러그인 UQ 캘리브레이션 시 반경이 40~60% 줄어드는 대신 커버리지는 조건에 따라 악화될 수 있음을 실증했습니다.



### Gradient-based inverse lithography for EUV masks via the waveguide method and a physics-informed neural operator (https://arxiv.org/abs/2606.25753)
- **Prior Approaches**: 기존 inverse lithography technology(ILT)는 원하는 웨이퍼 무늬를 만들기 위해 마스크 토폴로지를 역문제로 최적화하지만, pixel 기반 ILT는 매 반복마다 전자기(full-wave) 시뮬레이션을 재수행해야 해서 계산비용이 병목이 됐다. 이를 줄이기 위해 supervised surrogate나 physics-informed neural networks(PINNs) 같은 신경망 기반 가속이 제안됐지만, 대규모 학습 데이터 의존과 현실적인 마스크 기하에 대한 일반화 한계가 자주 지적된다. 또 waveguide method는 전방 시뮬레이션을 빠르게 만들지만, 결국 선형계 해법이 계산 병목으로 남는다.

- **Core Contribution**: 이 논문은 differentiable waveguide method와 waveguide neural operator(WGNO)를 end-to-end physics engine으로 재구성해, 전방 회절 모델 전체를 automatic differentiation으로 미분하고 흡수체(absorber) 유전율을 직접 복원하는 gradient-based ILT 프레임워크를 제안한다. 설계변수는 두 방식(픽셀-wise density reparameterization, Fourier-parameterized projection)으로 잡되, 공통적으로 하드 이진화와 엄밀 전방 검증을 포함해 결과가 물리적으로 타당하도록 만든다. 또한 WGNO를 결합한 하이브리드 최적화로, waveguide method의 선형계 병목을 신경망으로 완화하는 경로도 함께 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 전방 회절 연산이 설계변수에 대해 미분 가능해야 하고, 그 그래디언트가 안정적으로 계산되어 최적화가 수렴해야 한다는 점이다. 논문은 파동가이드 방법의 evalue problem과 경계조건으로 구성된 전방 연산을 연쇄적인 대수 연산으로 만들고, chain rule에 따라 reverse-mode automatic differentiation으로 loss의 그래디언트를 계산한다. WGNO 하이브리드에서는 선형계 풀이를 MLP가 근사하도록 물리 잔차(residual) 항으로 학습을 강제해 physics-consistency를 유지하며, 다만 WGNO 학습까지 포함하면 현재는 속도 이득이 상쇄될 수 있음을 명시한다.

- **Empirical Impact**: 수치실험에서는 EUV 파장 11.2 nm에서 TaBN, La, U 흡수체를 사용한 현실적인 2D/3D 마스크의 경우, 목표 웨이퍼(타깃 스크린) 강도 분포를 재현하는 마스크 구조를 얻을 수 있음을 보였다. 특히 Fourier-parameterized projection이 pixel-wise density 대비 최적화 벽시계 시간을 179초→137초로 약 1.31배 가속하면서도 마스크 벽이 더 매끈하고 제조 친화적이었으며, 타깃 대비 중심 피크/사이드로브 특성이 흡수체마다 달라 La는 중심 피크가 가장 좋고 U는 원하는 필드 매칭이 가장 가깝게 나타났다. 3D로 확장한 초기 실험(Nx=Ny=32)에서도 동일한 파이프라인이 목표 패턴을 재현하며, 주기적 전자기 구조(메타머티리얼 포함)의 inverse design로의 확장 가능성을 시사한다.



### Point Cloud Diffusion with Global and Local Reconstruction for Instance-Level 3D Anomaly Detection (https://arxiv.org/abs/2606.25740)
- **Prior Approaches**: 기존 3D 이상 탐지는 기준 정상 샘플을 바탕으로 입력을 복원한 뒤 잔차로 결함을 찾는 reconstruction-based가 주류다. 확산 기반 접근도 등장했지만, 전체 point set에 대한 global 복원이 중심이라 미세·약한 결함에는 공간 적응성이 부족해 성능이 제한된다. 또한 결함은 잘 복원되지 않는 반면, 배경은 재구성 과정에서 위치 편향이 생겨 false positive가 늘어나는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 instance-level 3D 이상 생성과 탐지를 동시에 겨냥한 diffusion 프레임워크 PCDiff를 제안한다. 생성 단계에서는 texture gradient, image patch, text, mask를 조건으로 한 instance-level multi-modal attention을 넣어 약한 결함(예: 잔흔·스크래치)도 고품질로 생성한다. 탐지 단계에서는 2D에서 유도한 이상 마스크를 근거로 local anomaly 복원과 global 기하 일관성을 결합하는 local-global 재구성으로 배경 구조는 보존하면서 전경 결함을 복원한다.

- **Technical Challenges**: 핵심 난제는 (1) 정규화된 point cloud에서 편차가 10^{-3} 수준까지 작아지는 약한 결함을 모델이 복원·검출하기 어렵다는 점과 (2) global 복원 위주의 학습이 배경의 positional bias를 유발해 오탐을 만든다는 점이다. 논문은 gradient 기반의 transformation-aware texture 표현과 geometry bank를 활용해 미세 결함의 텍스처 복잡도를 조건화하고, 2D anomaly mask를 back-projection해 다중 뷰 합의로 위치 잡음을 줄인다. 이어 local-global joint reconstruction과 selective merging으로 이상 영역 복원에는 지역 우선순위를, 전체 형상에는 구조적 기준선을 제공해 두 문제를 동시에 완화한다.

- **Empirical Impact**: Anomaly-ShapeNet과 Real3D-AD에서 PCDiff는 생성 fidelity(F-score, CLIP similarity)와 이상 탐지 성능(O-AUROC, P-AUROC) 모두에서 SOTA를 능가한다. 특히 Anomaly-ShapeNet에서 O-AUROC가 0.93 수준으로 가장 높았고, Real3D-AD에서도 O-AUROC가 0.82로 우수한 이상 국소화 결과를 보였다. 정성적으로는 기존 방법들이 경계만 강조하거나( false negatives/positives) 일부 난제에서 오탐을 보이는 반면, PCDiff는 이상 영역을 더 정확하고 일관되게 포착해 실사용 관점의 신뢰도를 높였다는 점이 확인된다.



### Power-Budgeted Underwater Vehicle Control via Constrained Reinforcement Learning (https://arxiv.org/abs/2606.25680)
Comments:
          10 pages, 10 figures

- **Prior Approaches**: 기존 연구는 RL을 활용해 AUV의 station-keeping과 trajectory tracking을 달성했지만, 보상에 task 정확도만 두면 진동성(oscillatory) 제어로 에너지를 낭비하기 쉽다는 문제가 제기돼 왔다. 에너지 절감은 흔히 reward에 action-effort 또는 energy penalty를 가중해 scalarize하는 방식으로 다뤄졌지만, 이 가중치는 물리 단위를 가지지 않아 목표 전력(예: 몇 W)을 사전에 지정하기 어렵고 차량·과업마다 수동 재튜닝이 필요하다. 더 나아가 가중치가 맞지 않으면 오히려 task-only 대비 전력 사용이 늘어날 수 있어 신뢰성이 떨어진다.

- **Core Contribution**: 이 논문은 에너지 효율을 reward 페널티가 아닌 “명시적 예산 제약”으로 정식화한다. 평균 thruster power가 물리 단위의 budget을 넘지 않도록 constrained MDP(CMDP)로 만들고, PPO-Lagrangian(PPO-Lag)으로 학습해 dual variable을 온라인으로 적응시킴으로써 차량·과업별 목표 전력을 튜닝 없이 맞추는 경로를 제안한다. 핵심은 고정된 무단위 에너지 가중치가 아니라, 측정된 power에 직접 반응하는 제약 기반 적응형 최적화라는 점이다.

- **Technical Challenges**: 제약을 세우더라도 실제 thruster 전기 전력(방향 비대칭·비선형 power-law)을 비용 신호로 정확히 모델링하고, 그 신호를 기반으로 policy가 예산을 만족하도록 학습을 안정화해야 한다. 논문은 thruster datasheet 기반의 전력-추력 비선형을 통해 step 단위 power cost와 평균 전력 예산 위반량을 정의하고, PPO의 업데이트에 cost advantage(예산 대비 초과 여부)를 결합한 뒤 dual ascent로 multiplier를 갱신한다. 그 결과, 초기에는 예산을 초과했다가 λ가 올라가며 제약을 강하게 걸고, 이후 예산 내 운영을 학습하면 λ가 내려가며 수렴하는 적응 거동을 보인다.

- **Empirical Impact**: MarineGym 시뮬레이션에서 3대(BlueROV, BlueROV-Heavy, HAUV)×4과업(hover, lemniscate/circle/spiral tracking) 총 12개 설정 모두에서 PPO-Lag가 thruster power를 최저로 만들며 task-only baseline 대비 14–65% 감소(최대 64.9%)를 달성했다. 또한 action smoothness는 12개 중 10개 설정에서 가장 좋았고, success rate와 tracking error는 대부분 유지(또는 일부 개선)되었다. 유일한 명확한 트레이드오프는 HAUV hover처럼 전력이 충분히 허용되지 않는 구간에서 정밀한 station-keeping을 일부 포기해 60.7% 전력 절감과 56% success가 함께 나타난 경우이며, 이는 “제약이 드러낸” 의도된 선택지로 해석된다.



### Steering Vision-Language Models with Joint Sparse Autoencoders (https://arxiv.org/abs/2606.25657)
Comments:
          19pages,10 figures

- **Prior Approaches**: 기존 Sparse Autoencoder(SAE) 연구는 언어 모델에서 해석 가능한 방향을 찾는 데 성과를 냈지만, VLM에는 그대로 옮기기 어렵습니다. VLM용 접근들(VL-SAE 등)은 공유 딕셔너리나 공분산 기반 정렬처럼 ‘모달리티 간 결합’이 충분히 개입 실험(steering)으로 이어지지 않는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Joint Sparse Autoencoder(JSAE)로, 이미지와 캡션(언어) 활성의 시퀀스-풀드 표현을 공유 잠재공간에서 함께 분해하되 ‘쌍(pair)된 코드의 명시적 cosine 정렬 제약’을 추가합니다. 그 결과 VLM에서 의미 개념(예: 음식, 동물)에 대응하는 크로스모달 기능을 더 “조종 가능한 steering direction”으로 회수하는 것을 목표로 합니다. 또한 LLaVA에 적용해 additive steering과 suppression을 동시에 비교하며 기능적 역할까지 탐색합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 비전 토큰과 텍스트 토큰을 비교 가능한 거시 표현으로 안정적으로 정리하고, (2) 정렬된 sparse 방향이 실제 생성 조작에 쓰일 정도로 ‘방향성/기하’를 보장하는 것입니다. JSAE는 모달리티별 encoder-decoder로 overcomplete sparse 코드를 만든 뒤 재구성·희소성(L1)과 함께 쌍 코드 정렬 손실을 학습에 포함하고, splitting된 원자 피처는 클러스터링으로 매크로 개념 벡터로 재집계한 뒤 양방향 개입을 수행합니다.

- **Empirical Impact**: LLaVA-v1.6-Mistral-7B에서 additive steering은 레이어 7·13·30 사이에서 중간~후반(특히 pre-output 구간)에서 성공률이 가장 높게 나타났고, suppression은 전 레이어에서 상대적으로 비슷한 범위의 점수로 관측됩니다. 이러한 레이어 국소화 효과는 Llama3-LLaVA-Next-8B와 MoE 기반 Qwen3-VL-30B에서도 유사하게 나타나, 정렬 제약이 없는 대안보다 더 controllable한 교차모달 분석에 유리하다는 신호를 제공합니다.



### Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization (https://arxiv.org/abs/2606.25656)
Comments:
          Accepted to ACL 2026 GEM Workshop

- **Prior Approaches**: 기존 RAG은 주로 비정형 문서에서 벡터 검색 후 LLM이 답을 생성하는 방식으로 강점을 보여왔지만, 실제 데이터는 텍스트와 관계(그래프)가 함께 있는 반구조화 지식베이스가 많다. GraphRAG, Modular RAG, Agentic RAG 같은 변형들이 나왔지만, “더 복잡한 구조가 정말로 잘 먹히는가”와 “표준 평가가 그 차이를 공정하게 반영하는가”는 불명확했다.

- **Core Contribution**: 이 논문은 반구조화 지식베이스(텍스트+KG)에서 regular RAG, GraphRAG, Modular RAG, Agentic RAG를 체계적으로 평가·비교하는 프레임워크를 제시한다. 또한 실제 사용 시나리오를 반영해 총 9개의 표준 RAG 시나리오(문서 단독부터 text-graph 통합, 도메인 KG 연계, 에이전트의 multi-step planning까지)를 구현하고, precision medicine 도메인 STaRK-Prime에서 생산형 의사결정에 필요한 인사이트를 도출한다.

- **Technical Challenges**: GraphRAG/Agentic RAG는 그래프 컨텍스트와 에이전트 세션 히스토리가 커지면서 context/memory overflow 및 토큰·비용 문제가 쉽게 발생한다. 논문은 (1) 관계를 묶는 compact한 그래프 표현, (2) 에이전트 세션 내 그래프 deduplication, (3) 문서 chunk/content hash 기반 deduplication, (4) ReAct의 단순 루프를 배치형 retrieval로 바꾸는 컨텍스트 절감 루프 설계를 통해 token 사용을 19%-53%까지 줄이면서도 성능을 유지/개선한다.

- **Empirical Impact**: 실험 결과, 그래프만 쓰는 순수 GraphRAG(시나리오 3)는 성능이 매우 낮았지만, 사전 정의된 KG를 텍스트와 적절히 결합한 하이브리드(시나리오 5)가 전반적으로 우수했다. 반면, retrieval 순위 중심 지표가 개선을 과대평가할 수 있음을 보여주며, LLM이 생성 과정에서 실제로 선택하는 entity ID를 평가하는 generation-aware 설정에서 “retrieval-generation gap(확장 검색이 생성 품질을 비례 향상시키지 않음)”이 관찰됐다. 결론적으로, 고급 RAG를 무조건 도입하기보다 데이터의 구조/제약과 컨텍스트 비용 한계를 기준으로 아키텍처를 선택해야 한다는 데이터 기반 가이드라인을 제공한다.



### Taxonomy of Risks on Automated Fact-Checking Systems Considering its Propagation (https://arxiv.org/abs/2606.25645)
Comments:
          15 pages, 3 figures, preprint

- **Prior Approaches**: 기존 자동 팩트체킹 연구는 정확도 향상이나 시스템 구조를 다루는 경우가 많았고, 오판이 실제로 어떤 피해로 이어지는지(일반 사용/악용 시나리오)는 충분히 정리되지 않았다. 또한 보안 위험을 다루는 STRIDE 같은 프레임워크는 전통적 IT 위협 중심이라, LLM 기반 팩트체킹에 특화된 간접 피해(명예훼손·허위 확산 등)까지 체계적으로 포착하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 자동 팩트체킹 시스템에서 발생 가능한 위험을 위험 요인(risk factors)–위험한 상황(hazardous situations)–피해(harm)로 이어지는 3단계 전파 관점으로 분류한다. 그 결과 자동 팩트체킹에 특화된 32개 위험을 도출하고, 이를 분석 단서(guide words)로 활용해 DEFAME의 위험을 평가할 수 있음을 보인다.

- **Technical Challenges**: 핵심 기술 과제는 (1) AI/LLM의 오판·환각 같은 비보안형 오류가 사용자 행동을 통해 사회적 피해로 연결되는 경로를 구조화하고, (2) 공격 시나리오(데이터/증거 오염, 반복 쿼리로 true 유도 등)까지 동일한 틀에서 포착하는 것이다. 저자들은 ISO/IEC Guide 51의 시간적 흐름 아이디어와 선행 위험 분류를 대응시켜 fault tree 형태의 논리 관계를 만들고, DEFAME의 DFD(데이터 흐름도) 위 각 경계·주체에 guide words를 매핑하는 방식으로 해결했다.

- **Empirical Impact**: 사례 분석에서 DEFAME은 시스템 파이프라인, 외부 도구/외부 LLM, 검색 엔진 경계에 위험 요인이 위치하고, 이후 파이프라인에서 위험한 상황이 발생하며 SNS 게시를 통해 피해가 나타나는 패턴이 확인됐다. 또한 STRIDE로는 포착하기 어려운 명예훼손·허위/조작 정보 확산 같은 사회적 영향이 제안된 분류로는 도출되어, 자동 팩트체킹 안전성 평가의 실무적 기준선을 넓혔다는 의미가 있다.



### Staying In Character: Perspective-Bounded Memory For Book-Based Role-Playing Agents (https://arxiv.org/abs/2606.25632)
- **Prior Approaches**: 기존 LLM 역할놀이 에이전트는 소설에서 인물/장면/관계를 추출하거나, persona 프로필과 대화 히스토리, RAG를 결합해 캐릭터 일관성을 높이려는 흐름이 강했습니다. 하지만 장편에서 자주 발생하는 OOC(out-of-character) 문제로, 캐릭터가 “보지 못한 사실”을 말해버리는 Factual Overreach과 고정된 말투로 상황 변화를 놓치는 Stylistic Monotony가 충분히 통제되지 않았습니다. 또한 경계(무엇을 알아야/거절해야 하는지)를 “캐릭터 관점에서 접근 가능한가”로 정의한 검증은 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 소설 기반 캐릭터 에이전트의 지식 경계를 관점에 맞춰 강제하는 3층 메모리 구조 ReverieMem을 제안합니다. 핵심은 (1) 1인칭 장면 경험을 저장하는 episodic layer, (2) 캐릭터별 visibility로 접근 가능한 사실만 남기는 semantic layer, (3) 감정 전이와 상황에 따라 달라지는 발화/행동 패턴을 담는 personality layer를 분리해 추론에 결합하는 방식입니다. 이를 통해 “접근 불가능한 사실은 검색/생성에서 배제”하고, “상황에 맞는 스타일 변화를 패턴으로 주입”하는 목표를 동시에 달성합니다.

- **Technical Challenges**: 기술적 난제는 두 가지 실패를 동시에 막되, 장편에서 검색 노이즈를 줄이면서도 필요한 세부를 복원해야 한다는 점입니다. 저자들은 추론 시 episodic layer로 장면 프레임을 먼저 고정하고, semantic layer의 visibility-allowed subset ℱc를 기준으로만 facts 후보 풀을 확장하는 perspective-bounded reconstructive recall 파이프라인을 설계했습니다. 동시에 emotion transition 기반으로 personality 패턴을 선택해 메모리 fusion의 프롬프트에 style anchor를 넣어 말투 평탄화를 완화했습니다.

- **Empirical Impact**: 평가는 8개 소설로 구성한 KBF-QA(총 4,386문항)와 BookWorld의 5차원 pairwise 내러티브 비교 프로토콜을 함께 사용해 지식 경계와 생성 품질을 분리 측정했습니다. ReverieMem은 Knowledge Boundary Fidelity에서 기존 최강 대비 34.6 percentage points 개선을 보였고, pairwise 비교에서도 약 79% win rate를 기록했습니다. 또한 계층별 제거 실험에서 episodic/semantic/personality 각각이 독립적으로 필요하며, 특히 visibility gating이 factual overreach와 character-anchored 서사 품질을 함께 끌어올린다는 점을 보여줍니다.



### TL++: Accuracy and Privacy Preserving Traversal Learning for Distributed Intelligent Systems (https://arxiv.org/abs/2606.25627)
Comments:
          25 pages, 3 figures

- **Prior Approaches**: 연합학습(FL)은 데이터는 로컬에 두지만, 비-IID 데이터에서 목표함수 불일치로 수렴이 흔들리고(accuracy 문제), 전체 모델 교환 비용이 커집니다. Split learning(SL)·Traversal learning(TL)은 cut-layer 활성/그래디언트를 보내 통신을 줄이지만, SL은 중간 텐서를 평문으로 노출해 복원 공격 위험이 있고, TL도 평문 교환 때문에 보안이 부족하다는 한계가 있습니다. TL++는 이 두 축을 동시에 만족시키려는 시도입니다.

- **Core Contribution**: 논문은 virtual batch를 노드 간에 구성해 중앙집중식 미니배치 SGD와 동일한 그래디언트 업데이트(accuracy losslessness)를 목표로 하는 TL을 기반으로, 이를 보안 모드로 확장한 TL++를 제안합니다. TL++의 핵심은 cut-layer 활성과 그래디언트를 오케스트레이터와 비공모(non-colluding) 헬퍼 서버에 additive secret sharing으로 나눠, 어떤 서버도 평문 중간 텐서를 보지 못하게 하는 것입니다. 또한 두 모드(base/secure)를 분리해 신뢰 환경에서는 TL과 동일 동작을, 민감 환경에서는 보안 강화를 수행하도록 설계했습니다.

- **Technical Challenges**: 보안 모드에서 accuracy losslessness를 유지하려면, 서버 파티션의 계산이 additive shares 위에서 ‘정확히’ 성립해야 합니다. 특히 lightweight secure 경로에서는 서버 경로가 sharewise로 선형/affine일 때만 정확하며, ReLU·풀링·정규화·softmax 같은 비선형은 secure nonlinear MPC나 근사 없이는 중앙 그래디언트와 달라질 수 있습니다. 논문은 이 조건을 수식적으로 정리하고, 그 결과를 cut 깊이(cut 1~3)별 정확도/근사 여부로 연결해 실험 설계에 반영합니다.

- **Empirical Impact**: CIFAR-10에서 TL++ base cut 1은 91.41%(SD 0.19), secure cut 3는 90.93%(SD 0.17)로 비-TL++ 최강 기준선 대비 12%p 이상 높은 정확도를 보고하며, secure cut 1~2는 비선형 근사 영향으로 정확도가 낮게 나타납니다. 통신 측면에서도 base cut 1은 full-model 동기화 대비 per-step 통신을 13.1배 줄이는 등, cut 깊이에 따라 유리한 운영점을 찾을 수 있음을 보입니다. BioGPT/PubMedQA에서도 TL++가 비슷한 방향으로 우수한 성능을 보이며, activation-level secret sharing을 통해 중앙화에 가까운 성능과 중간 계산 보호를 동시에 노린다는 점에서 의의가 큽니다.



### Probabilistic Agents in Deterministic Audits: Evaluating Multi-Agent Systems for Automated Audits Based on the German IT-Grundschutz (https://arxiv.org/abs/2606.25622)
Comments:
          Accepted for publication at the 2026 IEEE International Systems Conference (SysCon), Halifax, NS, Canada, April 6-9, 2026. 8 pages, 1 figure

- **Prior Approaches**: NIS-2 대응을 위해 기업들은 ISMS 인증의 기반으로 ISO/IEC 27001과 독일 BSI IT-Grundschutz(IT-GS)를 활용해 왔다. 다만 IT-GS는 방대한 문서 기반 증빙과 검증·수정이 수작업 중심이라, 표준을 그대로 자동화하려 해도 확장성과 비용 문제가 컸다. LLM 기반 RAG는 문서 처리에는 강점이 있지만, 확정적 논리와 검증 가능한 근거가 필요한 컴플라이언스 감사에서는 환각과 불안정성이 걸림돌로 지적돼 왔다.

- **Core Contribution**: 이 논문은 IT-GS 인증의 ‘부분 자동화’를 목표로, Multi-Agent System(MAS)과 Hybrid Retrieval Augmented Generation(HybridRAG)을 결합한 end-to-end 파이프라인을 제시한다. 핵심 기여로 SA(Structural Analysis) 단계에서 Knowledge Graph로 의존성을 교차검증하는 Hypothesis-Verification Loop와, LLM의 의미 추출과 보호 필요(Protection Needs) 상속을 분리하는 Decoupled Reasoning Pipeline을 도입해 컴플라이언스 엄밀성을 강화했다. 또한 BSI의 RecPlast GmbH 케이스를 기준 데이터로 사용해 Precision, Recall, F1-score를 정량화하며 각 단계 성능을 평가한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 LLM의 확률적 추론을 IT-GS가 요구하는 결정론적·검증가능한 규칙(예: 보호 필요 상속 원칙)과 연결하는 것이었다. 논문은 KG(Neo4j) 기반 HybridRAG로 구조적 근거를 강제하고, SA에서는 추론한 의존성 엣지를 KG의 결정적 토폴로지와 대조하는 루프로 환각 의존성을 줄인다. 반면 PNA와 IT-GS Check처럼 논리적 엄밀함이 중요한 단계에서는 LLM의 불확실성이 남아, 확률적 모델이 결정론적 수준의 rigor를 완전히 충족하기 어렵다는 한계도 함께 관찰됐다.

- **Empirical Impact**: RecPlast에서 MAS의 성능은 단계별로 뚜렷한 차이를 보였다. SA와 Modeling은 의미 작업에서 효율이 높아 문서 기반 자동 정보 추출로 수작업 부담을 줄였지만, 의존성 토폴로지와 논리 추론이 필요한 PNA 및 IT-GS Check에서는 Precision/Recall이 제한되며 실제 감사에서 사람이 검증(Human-in-the-Loop)해야 함을 실증했다. 결과적으로 이 연구는 ‘완전 자동화’보다는 감사 워크플로에서 사람의 역할을 증빙 수집에서 AI 생성물 검증으로 이동시키는 실용적 전환 가능성을 제시한다.



### An Approach for a Supporting Multi-LLM System for Automated Certification Based on the German IT-Grundschutz (https://arxiv.org/abs/2606.25608)
Comments:
          Accepted for publication at the 2025 IEEE International Conference on Cyber Security and Resilience (IEEE CSR), Chania, Crete, Greece, August 4-6, 2025. 8 pages, 2 figures

- **Prior Approaches**: 기존에는 BSI IT-Grundschutz(IT-GC) 기반 보안 개념을 문서와 체크리스트 중심으로 수작업 작성·검증하는 방식이 일반적이었고, NIS2 범위 확대에 따라 반복 업무와 전문 인력 부담이 빠르게 커졌다. 그래프/벡터 기반 RAG 또는 CS 지식그래프(KG) 연구는 공격 경로·이벤트 연관 등 다양한 보안 분석에는 도움을 줬지만, 특정 규정 절차(보호요구도 산정→모델링→IT-G-Check→통합)에 맞춘 “컴플라이언스 워크플로 자동화”에 초점이 약했다. 또한 LLM 단독 사용은 환각(hallucinations)과 근거성(출처 정합성) 문제가 커서 인증 산출물의 감사 가능성이 떨어진다는 한계가 있었다.

- **Core Contribution**: 이 논문은 MultiLarge Language Model system(MLS)에 Hybrid RetrievalAugmented Generation(HybridRAG)을 결합해 IT-Grundschutz 표준 보호(Standard Protection) 인증의 여러 단계를 반자동으로 지원하는 아키텍처를 제안한다. 특히 보호 필요요건 평가, 네트워크 모델링, IT-Grundschutz check, 조치 병합(consolidation), 이후 실행 계획까지 규정 주도형 절차를 모듈화된 task expert들이 순차 처리하도록 설계했다. 이를 통해 NIS2로 늘어난 인증 수요 속에서 인증기관(및 certifier)의 품질 유지와 비용·시간 절감을 동시에 노린다.

- **Technical Challenges**: 핵심 기술 과제는 (1) LLM이 규정 문구를 임의로 구성하지 않도록 근거를 단단히 고정하는 것과 (2) 복잡한 모듈·요구사항·조치 간 관계(상속, 누적 효과, 중복/충돌)를 정확히 반영하는 것이다. 논문은 IT-GC를 KG로 구조화해 관계를 명시적으로 제공하고, HybridRAG(그래프+벡터 검색)로 전역 문맥과 국소 관계를 함께 가져오도록 하여 환각을 낮춘다. 또한 Faithfulness, Answer Relevancy, precision/recall, Prompt Alignment, BERTScore, hallucination score 같은 검증 지표와 JSON 스키마 기반 출력 검증, spot check를 Orchestrator에 내장해 품질을 지속적으로 관리한다.

- **Empirical Impact**: 실험/평가에서 HybridRAG는 GraphRAG 대비 환각을 약 6% 줄였고, 규정 문서 범주의 작업(Digital Operational Resilience Act 및 금융기관 가이드라인 유사 문서)에 대해 유사한 이점을 보였다고 보고한다. 제안 아키텍처는 반복적인 ‘diligent work’ 구간(예: 객체→모듈 매핑, IT-G-Check 체크리스트 생성, 조치 통합)에서 수작업 시간을 줄이고, 조치 우선순위·실행 계획의 일관성과 감사 가능성을 높이는 방향으로 설계됐다. 결과적으로 NIS2 하의 디지털 보안 컴플라이언스 수요를 감당하기 위한 실무형 인증 보조 체계로서 의미가 크며, 향후 다른 컴플라이언스 프레임워크로의 확장 가능성도 시사한다.



### Expresso-AI: Explainable Video-Based Deep Learning Models for Depression Diagnosis (https://arxiv.org/abs/2606.25606)
Comments:
          8 pages. Accepted at the 2023 11th International Conference on Affective Computing and Intelligent Interaction (ACII). Code: this https URL

- **Prior Approaches**: 우울 자동 진단(ADD)은 얼굴 표정, 자세, 시선, 음성 등 다양한 단서를 학습하지만, 많은 연구가 딥모델의 예측 근거를 사람에게 설명하지 못해 임상 적용성이 제한돼 왔다. 특히 비디오 기반 접근은 이미지를 넘어 동적 특징을 잡을 수 있음에도, 시간 축(temporal)까지 정량적으로 해석하는 연구는 부족했다. 기존 해석은 주로 2D 히트맵 등 정성적 시각화에 머물러 비교·검증이 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 얼굴 비디오로 우울 강도를 회귀하는 딥뉴럴네트워크의 의사결정을 해석하기 위한 프레임워크를 제안한다. Action Recognition 데이터로 사전학습된 ResNet 계열을 AVEC 2014 우울 라벨(BDI-II) 비디오에 fine-tuning한 뒤, DeepLift(Rescale Rule) 기반 attribution으로 얼굴 영역과 시간적 표현 의미(temporal expression semantics)를 함께 설명한다. 동시에 프레임/영역 기반 시각 설명과 수치적 설명을 제공해 모델의 추론 가설을 도출하도록 설계했다.

- **Technical Challenges**: 핵심 과제는 비디오 입력에서 시간 정보까지 포함한 attribution을 안정적으로 계산하고, 이를 얼굴 부위·표정 의미와 연결해 임상가가 이해 가능한 형태로 바꾸는 것이다. 저자들은 (2+1)D ResNet에 DeepLift를 적용해 프레임 단위 relevance를 얻고, 채널·공간 풀링으로 temporal attributions 벡터를 만든 뒤, 미리 추정한 얼굴 랜드마크 영역(눈/눈썹/입/코 등)에 대해 영역별 relevance를 집계했다. 또한 OpenFace 기반 Action Unit(AU)과 temporal attributions의 Kendall Tau 상관을 통해 시간 의미를 정량 검증하는 절차를 마련했다.

- **Empirical Impact**: 실험에서 (2+1)D를 비롯한 ResNet 계열을 비교했으며, Kinetics-700과 MiT 사전학습을 함께 사용한 R(2+1)D가 최상 성능을 보였다. 해석 결과로 심한 우울에서 코 꼬집음/눈썹 찡그림(AU9, AU4)처럼 알려진 표정 단서와의 연관성이 관찰됐고, 반대로 기쁨/정상 범주와 관련된 AU 조합은 음의 상관을 보였다. 한편 마이크·헤드셋 영역 높은 attribution은 데이터 편향 가능성을 시사하며, 제안된 해석 프레임워크가 성능뿐 아니라 근거 품질 점검에도 도움을 줄 수 있음을 보여줬다.



### Low-Complexity Policy Tessellations in Structured Markov Decision Processes (https://arxiv.org/abs/2606.25593)
- **Prior Approaches**: 기존의 근사 동적 계획(approximate dynamic programming)과 강화학습(reinforcement learning)은 주로 고차원 가치함수(value function)를 근사한 뒤 그로부터 최적 정책을 유도하는 방식에 집중해 왔다. 이 접근은 정책의 기하학적 구조(어떤 상태에서 어떤 행동이 선택되는지)가 간접적으로만 반영돼, 성능 저하가 어디서 왜 생기는지 설명과 제어가 어려웠다. 특히 오류가 경계 근처에 몰리는 현상을 정책 관점에서 정량적으로 다루기엔 한계가 있었다.

- **Core Contribution**: 이 논문은 구조화된 마르코프 결정 과정(MDP)에서 최적 정책이 ‘단순한 결정 테셀레이션(decision tessellations)’을 유도한다는 기하학적 관찰을 제시한다. 이를 바탕으로 정책 영역(policy regions)을 직접 학습하는 경계 기반(boundary-based) 정책 근사 기법을 제안한다. 또한 정책 손실(policy-loss)을 분해해 성능 저하가 행동 margin과 연관됨을 체계적으로 연결한다.

- **Technical Challenges**: 핵심 기술적 도전은 고차원 상태공간에서 최적 정책의 경계를 직접 학습하면서도 성능 저하가 어떤 요인에 의해 발생하는지 설명 가능한 형태로 만드는 것이다. 논문은 정책 손실을 작용 margin과 ‘무차별(indifference) 경계’ 근처의 영역으로 연결해, 오류가 그 경계에 집중되는 이유를 보여준다. 그리고 경계 기반 근사가 정책 영역을 직접 표현하도록 설계해, 가치함수 근사 중심의 방법보다 오류 누적을 줄이도록 한다.

- **Empirical Impact**: 재고 제어(inventory control)와 큐 어드미션(queue admission) 실험에서 제안 방법은 강화학습 기준선 대비 정책 오차가 낮고, 가치 격차(value gaps)도 더 작게 나타났다. 또한 오류가 더 빠르게 감소(faster error decay)하며, 제어 문제에서의 안정성(stability)도 개선되는 결과를 보였다. 이는 정책의 기하학적 구조를 직접 학습하는 접근이 실제 성능과 학습 동역학 모두에서 이점을 제공함을 시사한다.



### BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents (https://arxiv.org/abs/2606.25556)
- **Prior Approaches**: Stepwise group-based RL(GRPO, GiGPO, HGPO 계열)은 가치 네트워크 없이 롤아웃 스텝들을 그룹으로 묶어 국소 advantage를 추정한다. 하지만 그룹 내 스텝을 “교환 가능”하다고 보는 가정이 오래된 상태-동작 신용배분(credit assignment)에 맞지 않아, 관측 해시 같은 불완전한 그룹 기준이 학습 신호를 깎는 문제가 있었다.

- **Core Contribution**: 이 논문은 기존 방법의 핵심 결함을 state-action credit mismatch로 정식화한다. 관측 해시는 상태 동질성에 비해 너무 세분화해 singleton 그룹(신호 0)을 만들고, 그룹 내 평균은 action별 미래 차이를 반영하지 못하므로, BiPACE는 critic 없이도 advantage를 정확히 분해하도록 drop-in 방식의 추정기를 제안한다.

- **Technical Challenges**: BiPACE는 두 국소 문제(상태 집계, 동작별 신용배분)를 동시에 해결해야 한다는 점이 기술적 난제다. 해결책으로 BiGPO는 actor의 hidden-state 공간에서 cosine 거리 기반 클러스터(미세한 bisimulation proxy)를 만들고, PACE는 각 클러스터에서 실행된 action 기준 peer baseline으로 Q(s,a)-V(s) 형태의 비모수적(critic-free) advantage를 재중심화한다.

- **Empirical Impact**: ALFWorld에서 BiPACE_Q는 Qwen2.5-7B 기준 validation success를 GiGPO의 90.8에서 97.1±0.9로 끌어올리며, 동일 롤아웃 예산 내 95% 임계치를 모든 seed에서 달성한다. WebShop/TextCraft 및 Qwen2.5-1.5B에서도 GRPO·GiGPO·HGPO 대비 개선이 확인되며, 추가 오버헤드는 한 스텝 wall time의 11.3%로 보고된다.



### SFL-MTSC: Leveraging Semantic Frame-Level Multi-Task Self-Consistency for Robust Multi-Intent Spoken Language Understanding (https://arxiv.org/abs/2606.25552)
Comments:
          Interspeech 2026

- **Prior Approaches**: 프롬프트 기반 SLU는 LLM의 zero-shot/few-shot 추론으로 task-specific fine-tuning 없이 intent detection과 slot filling을 수행하지만, multi-intent 상황에서 decoding stochasticity로 intent–slot 구조가 매번 달라지는 문제가 자주 발생한다. 이를 줄이려는 self-consistency 계열은 대부분 output-level majority voting이나 LLM-as-a-judge 같은 방식을 쓰지만, frame 구조를 세밀하게 일관성 검증/필터링하기엔 한계가 있었다.

- **Core Contribution**: 이 논문은 semantic frame 단위로 예측을 분해·집계하는 SFL-MTSC(Semantic Frame-Level Multi-Task Self-Consistency) 프레임워크를 제안한다. output-level 투표 대신 intent-specific frame을 모아 domain–intent 버킷화, slot-level clustering, 그리고 path support 기반 신뢰도 평가로 안정적인 frame만 남긴 뒤 최종 multi-intent 결과를 재통합한다.

- **Technical Challenges**: 핵심 과제는 서로 다른 reasoning path에서 생성된 frame들이 구조적으로는 같지만 slot key/표현이 달라 생기는 불일치를 어떻게 안정적으로 묶고 걸러내는가다. 이를 위해 Hybrid Jaccard(키-값 매칭과 값 기반 매칭을 혼합)로 slot clustering을 수행하고, association rule mining의 support를 응용해 여러 경로에서 반복 지지되는 slot cluster만 retained 하도록 설계했으며, 슬롯 재통합은 value-first 방식으로 representative value를 우선 정하고 그에 대응하는 slot key는 다수결로 결정한다.

- **Empirical Impact**: MAC-SLU(중국어 multi-intent, 최대 4개 동시 intent) zero-shot 실험에서 SFL-MTSC는 전반 정확도와 Slot F1을 일관되게 개선했으며, 특히 Vanilla Prompting 대비 Overall Acc. 개선폭이 가장 크게 나타났다. 결론적으로 slot 수준의 self-consistency가 intent 분류보다 더 큰 영향을 주는 경향이 관찰됐고, slot-level support filtering ablation에서 성능 이득의 주된 원인이 슬롯 클러스터 신뢰도 필터링임이 확인됐다.



### Evaluating LLMs on Real-World Software Performance Optimization (https://arxiv.org/abs/2606.25530)
- **Prior Approaches**: 기존 소프트웨어 성능 최적화 연구/도구는 개별 함수 수준의 단순화된 문제나 단일 성능 지표에 치우치는 경향이 있어, 실행 시간과 메모리 사용 사이의 핵심 트레이드오프와 실제 코드베이스의 측정 잡음, 입력·실행 조건의 변동성을 충분히 반영하지 못했다. 또한 LLM 코드 리파인먼트를 활용해도 최적화가 ‘어떻게’ 이뤄지는지 재현·검증할 벤치마크가 부족해 성능 향상을 비교하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 실제 오픈소스 프로젝트에서 수집한 전문가 최적화 102개를 바탕으로 레포지토리 수준 벤치마크 SWE-Pro를 제안한다. 각 작업(task)을 파라미터화 테스트로 연결해 다양한 입력 데이터와 실행 조건에서도 runtime, peak memory, Time-Weighted Memory Usage(TWMU)를 동시에 평가하도록 설계했다. 이를 통해 최적화가 단순 개선이 아니라 조건·잡음이 있는 환경에서의 복합 최적화 문제임을 정량화한다.

- **Technical Challenges**: 레포지토리 수준 최적화에서는 최적화 효과가 특정 입력에서만 발생하거나 측정 환경의 잡음에 의해 성능이 흔들릴 수 있어, 신뢰 가능한 평가 체계를 만드는 것이 핵심 기술 난제였다. 논문은 noise-aware measurement 조건을 두고 작업별로 파라미터화 테스트를 적용해 runtime·peak memory·TWMU를 다양한 조건에서 측정함으로써 변동성과 트레이드오프를 벤치마크 자체에 반영했다. 그 결과 단일 점수 중심의 과잉 단순화를 피하면서 최적화의 실질 난도를 재현할 수 있게 했다.

- **Empirical Impact**: 실험에서 현재 LLM들은 대부분의 작업에서 runtime 이득이 미미하고, 메모리 최적화는 거의 나타나지 않았다. 반면 전문가 구현은 전체 작업 기준 aggregate speedup 15.5x와 peak memory 171.3x 감소를 달성했으며, runtime은 91.2% 작업에서, peak memory는 65.7% 작업에서 개선이 관측됐다. 이러한 격차는 ‘전문가 수준 엔지니어링’이 요구하는 최적화 능력과 현재 LLM의 한계가 크다는 점을 실증적으로 보여준다.



### STEB: A Speech-to-Speech Translation Expressiveness Benchmark for Evaluating Beyond Translation Fidelity (https://arxiv.org/abs/2606.25529)
- **Prior Approaches**: 기존 S2ST 연구는 ASR-번역-TTS로 이어지는 cascaded 파이프라인부터 spectrogram/unit 기반 end-to-end 모델, 그리고 음성 LLM까지 확장돼 왔습니다. 하지만 벤치마크들은 주로 번역 정확도에 초점이 맞춰졌고(예: CVSS, FLEURS), 감정 일부만 다루더라도 시나리오 스타일이나 nonverbal vocalizations(NVs)를 함께 검증하는 평가는 제한적이었습니다. 또한 표현성을 평가하려면 번역-충실도와 표현 정렬이 동시에 보장된 기준 타깃 음성이 필요한데, 이는 대규모로 만들기 어려워 reference-based 평가는 현실성이 떨어집니다.

- **Core Contribution**: 이 논문은 S2ST에서 표현성(preservation)을 정량 평가하기 위한 벤치마크 STEB를 제안합니다. STEB는 중국-영어 32.6시간 데이터로 감정(emotion), 시나리오 스타일(scenario style), NV preservation을 번역 충실도와 함께 동시에 평가하도록 설계됐습니다. 더불어 맞춤형 참조 타깃 음성 없이도 표현성을 점수화하기 위해 caption-then-summarize 기반의 reference-free LLM-as-a-judge 평가 프레임워크를 함께 제공합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 실제 음성에서 배경음/중복 화자/NV를 제거해 발화 단위 표현성을 안정적으로 추출하는 것, (2) 참조 타깃 음성이 없는 상황에서 감정·스타일·NV를 일관된 루브릭으로 비교하는 것입니다. 논문은 BS-Roformer로 잡음을 줄이고 VAD+diarization+speaker embedding으로 단일 화자 구간을 정제한 뒤, 3~30초 발화와 대역폭/SNR/DNSMOS 기준으로 품질 게이트를 적용합니다. 평가 측면에서는 Qwen3-Omni-Captioner로 음성 특징을 구조화 캡션으로 뽑고 Qwen3-30B-A3B로 감정·스타일을 요약한 뒤, LLM judge가 1~5점 루브릭으로 소스-가설 일관성을 비교하며 NV는 텍스트 내 마커 정렬로 세분화해 평가합니다.

- **Empirical Impact**: STE B의 자동 평가 점수는 인간 청취 평가와 통계적으로 유의미한 상관을 보였고, 특히 NV 보존이 가장 강하게 일치했으며 감정/시나리오 스타일도 유의미한 상관을 보였습니다. 6개 S2ST 시스템 실험에서는 전반적인 번역 정확도는 양호(예: cascaded 계열의 높은 BLEU)이지만 표현성 전이는 큰 격차를 보였습니다. 감정과 NV는 각각 최고 3.82/5와 2.31/5 수준에 머물러, 의미 전달(semantic transfer)과 표현 전달(expressive transfer) 사이의 갭이 확인되며 표현성 보존이 여전히 열린 과제임을 시사합니다. 또한 duration alignment는 명시적 길이 제어를 둔 모델에서만 SLC 지표가 크게 개선돼, 특정 표현 축은 모델 구조적 제약을 크게 받는다는 실증적 근거를 제공합니다.



### The impact of artificial intelligence on enterprise software user roles (https://arxiv.org/abs/2606.25525)
Comments:
          18 pages, 1 figure, 4 tables

- **Prior Approaches**: 기존 연구들은 AI가 소프트웨어 개발 워크플로우를 바꾼다는 점을 주로 정량 지표나 도구 수준에서 다뤘고, 실제 조직 내 역할·책임 변화는 상대적으로 덜 관찰되는 경향이 있었다. 또한 SAP BTP 같은 엔터프라이즈 플랫폼의 사용자 역할 분류(예: BTP User Type Matrix)에 대해 AI 도입 이후의 재정의 필요성을 충분히 검증하지 못했다.

- **Core Contribution**: 이 논문은 SAP Business Technology Platform(BTP) 환경에서 AI가 개발자의 전문 책임과 일상 업무를 어떻게 재배치하는지에 초점을 맞춘 질적 연구를 제시한다. 전문가 인터뷰(n=20)와 참여형 워크숍(n=24)을 통해 업무의 자동화 확대, 사람-사이의 협업 증대, agentic AI 시스템 의존 증가가 동시에 나타남을 확인한다.

- **Technical Challenges**: 핵심 기술적 어려움은 에이전트형 AI가 작업을 ‘대행’하는 과정에서 협업 방식과 운영 통제의 경계가 흔들린다는 점이며, 이를 기존 역할 프레임워크와 설계 원칙에 어떻게 반영할지다. 연구는 워크숍 기반의 질적 합의와 인터뷰 데이터를 통해 변화하는 책임 구조를 정리하고, 거버넌스·감독 기능과 AI-native 엔터프라이즈 소프트웨어 설계 접근을 업데이트해야 함을 도출한다.

- **Empirical Impact**: empirical하게는 실제 현업 관점에서 day-to-day 태스크와 역할 변화가 뚜렷하게 관찰되며, 기존 사용자 역할 매트릭스는 수정이 필요하다는 결론으로 이어진다. 이는 단순 생산성 향상을 넘어, 조직 설계(역할 택소노미)와 운영 통제(거버넌스/오버사이트)까지 포함하는 AI 도입 논의를 확장하는 데 의미가 있다.



### Spam and Sentiment Detection in Arabic Tweets Using MARBERT Mod (https://arxiv.org/abs/2606.25495)
- **Prior Approaches**: 기존 연구들은 트위터 같은 소셜 미디어 데이터를 바탕으로 고객 만족도와 비판을 파악하기 위해 감성 분석을 활용해 왔다. 딥러닝 기반 방법 중 BERT 계열이 NLP 감성 분석에서 성능을 보였지만, 상대적으로 영어 중심으로 연구가 진행돼 아랍어에는 공백이 컸다. 또한 아랍어 트윗에 맞는 충분한 데이터로 모델을 학습·평가한 사례가 제한적이었다.

- **Core Contribution**: 본 논문은 아랍어 트위터 고객 피드백을 대상으로 MARBERT를 학습해 감성을 분류하는 프레임을 제안한다. STC(Saudi Telecom Company) 고객 트윗을 분석해 긍정/부정/중립뿐 아니라 sarcasm과 indeterminate까지 포함된 감성 신호를 추출함으로써 고객 서비스 개선에 활용하는 것을 목표로 한다. 기존 영어 편중 연구를 아랍어 영역으로 확장한 점이 핵심 기여다.

- **Technical Challenges**: 아랍어는 형태 변화와 표현 다양성이 커서, 일반적인 모델이 뉘앙스를 안정적으로 포착하기 어렵다는 문제가 있다. 연구진은 아랍어 트윗 24,513개를 사용해 MARBERT 기반 제안 모델을 학습하고, 성능은 f1-score, precision, recall로 측정해 불균형한 라벨 분포를 반영해 평가했다. 특히 sarcasm과 indeterminate처럼 애매한 범주가 존재할 때의 분류 안정성을 함께 검증하는 방향으로 설계를 구성했다.

- **Empirical Impact**: 실험에서는 긍정 1,437, 부정 13,828, 중립 5,694, sarcasm 1,221, indeterminate 2,297로 구성된 아랍어 데이터로 학습·평가를 수행했다. 결과적으로 제안 방식은 문헌의 기존 기법과 비교해 정확도 측면에서 유망한 성과를 보였다고 보고된다. STC 고객 서비스 관점에서 트위터 기반 신속한 피드백/불만 파악을 돕는 실무적 활용 가능성도 시사한다.



### HG-Bench: A Benchmark for Multi-Page Handwritten Answer-Region Grounding in Automated Homework Assessmen (https://arxiv.org/abs/2606.25491)
- **Prior Approaches**: 기존 referring-expression grounding 벤치마크는 주로 자연 이미지에서 단일 타깃을 찾는 과제에 초점이 맞춰져, 과제별 답/추론 단계를 계층적으로 박스 출력하는 요구가 제한적입니다. 문서 이해 연구는 텍스트·레이아웃을 다루지만, 페이지 단위의 좌표 의미와 ‘정답 영역-단계 영역’ 순서/포함 제약을 갖춘 지역 그라운딩은 잘 측정되지 않았습니다. 또한 수학·과학 풀이형 벤치마크는 관련 내용이 이미 식별됐다는 가정 아래 평가가 이뤄져, 실제 채점 파이프라인의 전 단계(공간 로컬라이제이션) 부족을 드러내지 못했습니다.

- **Core Contribution**: 이 논문은 K-12 필기 숙제에서 페이지를 고려해 ‘완전한 정답 영역’과 그 안의 ‘순서가 있는 단계별 서브 영역’을 동시에 찾는 page-aware, two-level answer-region grounding을 새 평가 설정으로 제시합니다. 이를 위해 1,489,278개 소스 풀에서 선별한 500개 인적 라벨 테스트(총 10,420개 학습)로 구성된 HG-Bench를 공개합니다. 질문-단계 박스가 계층적 포함 제약으로 연결되도록 설계하고, complete answer localization(FA)과 step-level decomposition(FSm)을 분리 측정해 공간적 추론 구조를 실제로 그라운딩하는지 검증합니다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 (1) 여러 페이지 스캔에서 페이지별 좌표 의미를 유지하면서, (2) 잡음·그림자·필기체 변이를 견딜 정도로 정답/단계 영역을 박스로 정확히 국소화하고, (3) 단계들을 학생 작성 순서대로 정렬하며, (4) ‘단계 박스는 부모 정답 박스 내부에 완전히 포함’되도록 계층 일관성을 보장하는 것입니다. 저자들은 페이지 단위 1:1 매칭과 IoU 기반 필터링으로 FA와 FSm을 공정하게 집계하는 평가 프로토콜을 구성하고, JSON 스키마·좌표계 변형을 다루는 재시도/보정 절차까지 포함해 모델 출력 형식 문제를 통제합니다. 또한 GLM-4.6V 9B를 HG-SFT 약 1만 예제로 단일-stage SFT한 참조 모델을 제공해, 벤치마크가 학습으로 실제 개선 가능한지 하한(lower bound) 관점에서 확인합니다.

- **Empirical Impact**: 실험에서 frontier closed-source API와 경쟁 open-weight VLM들은 zero-shot 기준 FA 55.22%, FSm 48.22%를 넘지 못했지만, HG-SFT로 미세조정한 GLM-4.6V 9B 참조 모델은 FA/FSm 74.97/72.26까지 도달했습니다. 특히 step-level 성능 하락이 급격해 ‘정답 영역 대강 찾기’보다 ‘순서 있는 단계 구조를 그라운딩’하는 능력 격차가 핵심임을 보여줍니다. 결과적으로 HG-Bench와 page-aware 평가 프로토콜, 학습된 참조 체크포인트가 자동 채점/추론 추적에서 필요한 구체적 역량을 재현 가능하게 측정하는 출발점이 될 것으로 기대됩니다.



### Rate-Aware Quantum-Inspired Trajectory Learning for Interference-Limited Multi-UAV Networks (https://arxiv.org/abs/2606.25480)
- **Prior Approaches**: 기존 UAV 군집(드론) 조정 연구는 주로 궤적 최적화와 네트워크 제약을 함께 고려하지만, 간섭이 많은 환경에서는 탐색 공간이 급격히 커져 실시간 계산이 비싸지는 문제(차원의 저주)가 나타난다. 또한 무작정 중심화된 최적화나 단순 그래프 단축은 간섭 상황과 QoS 요구를 충분히 반영하지 못해 처리량과 우선 사용자 성능 사이의 균형이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 Rate-Aware Quantum-Annealed Graph Condensation(RA-QAGC)이라는 스킴을 제안해, 간섭-제한 환경에서 확장 가능한 UAV 조정을 목표로 한다. 핵심 아이디어는 (1) 처리율(throughput)에 민감한 그래프 추상화를 통해 탐색 부담을 줄이고, (2) 분산 강화학습을 통해 QoS를 유지하면서 처리율이 높은 지역으로 궤적 적응을 유도하는 것이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 간섭이 존재하는 상태에서 방대한 궤적 탐색을 줄이면서도 네트워크 용량 균형과 QoS 조건을 동시에 만족시키는 것이다. 저자들은 높은 처리율 위치를 찾아 ‘처리율 최적 영역’으로 궤적을 편향시키는 rate-aware 그래프 추상화와, 이를 따르는 분산 강화학습을 결합해 간섭 인지형 적응을 가능하게 했다.

- **Empirical Impact**: 시뮬레이션 결과 RA-QAGC는 기존 방식 대비 총 처리량 59.4 Mbps, 우선 사용자 처리량 23.9 Mbps를 달성해 각각 약 15%, 34%의 향상을 보였다. 재난 및 일상 환경 모두에서 실시간 협응성과 간섭-인지 용량 배분을 동시에 노릴 수 있다는 점에서 UAV 통신·제어 분야에 의미 있는 성능 개선을 제공한다.



### A Red Teaming Framework for Large Language Models: A Case Study on Faithfulness Evaluation (https://arxiv.org/abs/2606.25476)
Comments:
          Preprint submitted to SQJ

- **Prior Approaches**: 기존 red teaming은 GARAK, GOAT, HarmBench, JailJudge처럼 공격 자동화나 단일-판정 중심의 평가가 주를 이뤘다. 또한 multilingual 평가는 일부 프레임워크에서 다뤄졌지만, 언어쌍·태스크 전반의 취약점 비교와 평가 신뢰도(일관성)까지 같이 검증하는 경우는 제한적이었다. 특히 unfaithfulness(맥락-기반 환각)처럼 미묘한 오류를 안정적으로 가려내는 방법론이 부족했다.

- **Core Contribution**: 이 논문은 target-attacker-jury의 multi-role red teaming 아키텍처를 제안해, 공격 생성과 판정 역할을 분리하고 정보 흐름을 단방향으로 통제한다. jury는 다수 모델 합의와 inter-judge reliability(Fleiss’ kappa)를 사용해 응답의 정확성과 일관성을 정량 평가하며, 이 구조는 Q&A·요약·안전 위해 생성 등 다양한 태스크에 모듈형으로 적용된다. 특히 맥락에 묶인 unfaithfulness를 겨냥한 설계를 통해 신뢰할 수 있는 “취약점 패턴” 분석을 가능하게 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 공격 프롬프트가 자동으로 더 잘 먹히도록 편향이 누적되는 feedback loop를 막고, (2) 판정이 단일 모델의 주관에 좌우되지 않게 신뢰도를 확보하는 것이었다. 이들은 attacker와 target, jury를 엄격히 분리하고 ensemble 합의(majority/unanimous) 및 Fleiss’ kappa로 일관성을 계산하는 방식으로 해결했다. 또한 요약에서는 length limitation 같은 구조적 제약이 취약점 양상을 바꾸는지를 확인해, 단순 스케일링보다 아키텍처 설계·평가 설계가 안전에 더 큰 영향을 줄 수 있음을 보였다.

- **Empirical Impact**: 실험에서 exploitative한 적대 프롬프트는 Q&A에서 공격 성공률(ASR)을 최대 7.9%p까지 끌어올렸고, 요약에서는 구조적 제약이 unfaithfulness를 최대 30%까지 낮추는 효과가 관찰됐다. 언어 비교 실험에서는 Arabic 처리에서 unfaithfulness 취약점이 더 높게 나타나(평균 ASR 관측치 기준), 언어쌍 차이가 실제 신뢰성 문제로 이어질 수 있음을 시사했다. 다만 언어 간 맥락에서 미세한 unfaithfulness(명시적 모순이 아닌 오류)는 자동 탐지에 한계가 있었고, 다국어 적대 프롬프트의 완전 자동화는 추가 과제가 남았다.



### EchoStyle: Unlocking High-Fidelity Video Stylization with Reverse Data Synthesis (https://arxiv.org/abs/2606.25465)
- **Prior Approaches**: 기존 비디오 스타일화는 참조 이미지(또는 키프레임)를 style prior로 두는 방식이 주를 이뤘지만, 불필요한 정보가 함께 들어가 content leakage와 style drift를 유발하는 문제가 있었다. 또한 고품질 paired 비디오 데이터가 부족해 합성 데이터로 학습하면 깜빡임(flickering)과 불안정성이 커졌고, 긴 영상(수 초 이상)으로 확장하기도 어렵다는 한계가 있었다. 학습 없이 keyframe 스타일 편집 후 feature를 주입해 시간 일관성을 맞추는 방법들도 있었지만, 텍스트 기반으로 안정적인 open-source 패러다임을 만들기는 여전히 난제였다.

- **Core Contribution**: EchoStyle은 텍스트 기반 비디오 스타일화를 video-to-video 생성 문제로 정식화하고, Wan2.2-I2V 기반 구조에서 비디오 내용과 텍스트 스타일을 재결합(refuse/fuse)하는 프레임워크를 제안한다. 참조 기반 누출 우려를 줄이기 위해 입력 조건을 통합된 잠재(latent) 정렬로 구성하고, 긴 길이에서도 동작하도록 init-follow-mode와 sliding-window 추론을 설계했다. 더불어 대규모 학습을 가능케 하는 자동 역합성(reverse-synthesis) 파이프라인으로 V-Style20k(20k 페어) 데이터셋을 구축했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 영상의 의미/내용 정합성을 유지하면서 스타일만 정교하게 주입하는 것과 (2) 장시간 생성에서 메모리 부담 및 누적 오류를 억제하는 것이다. EchoStyle은 VAE로 참조/타깃 비디오를 잠재로 인코딩한 뒤, mask 조건과 텍스트·타임스텝을 DiT 백본에 adaLN과 cross-attention으로 주입해 정렬성을 확보한다. 데이터 희소성은 합성 아티팩트가 섞인 역분포 오염을 피하기 위해, 실제 스타일 비디오를 수집해 현실적인 counterpart를 reverse-synthesis로 생성하는 방식으로 해결했으며, init-follow-mode 학습과 sliding-window 추론으로 세그먼트 경계의 시간 불연속을 완화했다.

- **Empirical Impact**: 실험에서 EchoStyle은 다양한 예술 스타일에서 스타일 유사도와 인간 선호 평가를 기준으로 기존 오픈/상용 대안 대비 우수한 결과를 보였고, 특히 동적 품질(dynamic quality)에서 상위권 성과를 냈다. 정량적으로는 style similarity, style consistency, content preservation 전반에서 강점을 보이며, Kling-O1·Seedance 2.0 같은 closed-source 대비로도 전반 성능이 경쟁력 있음을 보였다. 긴 쇼트(복잡한 공간 역학·조명 변화) 확장 테스트에서도 EchoStyle이 세맨틱 드리프트와 단순화가 덜해 전문용 예술 비디오 제작 대안으로 의미가 있다는 점이 확인됐다.



### Learning with a Single Rollout via Monte Carlo Pass@k Critic (https://arxiv.org/abs/2606.25451)
- **Prior Approaches**: 수학적 정답 검증이 가능한 RL 설정에서는 최종(outcome) 보상으로 학습하지만, 보상이 지연·이진 형태라 토큰/과정에 대한 credit assignment가 어렵다. 특히 단일 프롬프트에서 여러 완성을 뽑아 그룹 내 정규화하는 GRPO류는 롤아웃 비용과 경로 이질성 때문에 장기/에이전트형 추론에서 노이즈와 비용이 커진다. 또한 GAE처럼 TD 부트스트래핑을 쓰면 긴 horizon에서 값 추정 오차가 중간 단계들을 따라 누적될 수 있다.

- **Core Contribution**: 이 논문은 single-rollout proximal policy optimization(SR-PPO)으로, 프롬프트당 한 번의 롤아웃만으로도 토큰 단위 credit을 부여하는 방법을 제안한다. 핵심은 Monte Carlo 결과(최종 이진 정답)로 학습한 토큰/프리픽스 레벨 credit critic이 각 위치에서 eventual success 확률을 예측하도록 하고, 그 변화량을 advantage로 사용한다. 기존 Pass@1 중심 신호 대신 Pass@k(success: k개 중 하나라도 성공) 신호를 사용해, 풀기 쉬운 프리픽스는 학습 신호를 줄이고 어려운(하지만 도달 가능한) 영역에 더 선택적으로 학습을 집중시킨다.

- **Technical Challenges**: 직면한 문제는 (1) 프롬프트당 단 한 개 롤아웃으로도 토큰별로 유의미한 advantage를 만들 수 있어야 하고, (2) TD 부트스트래핑 오차 누적을 피하면서도 sparse한 outcome만으로 dense credit을 구성해야 한다. 이를 위해 critic을 프리픽스별 success 확률을 예측하도록 캘리브레이션(BCE + Brier score)하고, 각 토큰의 local credit(프리픽스 확률 변화)과 관측된 최종 결과에 맞추는 terminal correction term을 결합한다. 또한 Pass@k의 기하학적 성질(성공 확률이 이미 큰 구간의 gradient 완화)을 활용하되, k→∞에서는 reachability(성공으로 이어질 수 있는지)로 수렴함을 그래프 관점에서 해석·정리한다.

- **Empirical Impact**: Qwen3-1.7B(마스킹/추론 모드)를 기반으로 AIME24/AIME25/HMMT26에서 128개 샘플로 Pass@k를 측정한 결과, SR-PPO는 학습 안정성과 함께 Pass@128 성능이 일관되게 개선된다. 특히 학습 중에는 프롬프트당 single rollout(기본값)만 쓰면서도, GRPO가 학습 중 여러 롤아웃(예: 8개 중 표본 다수)을 쓰는 조건에서 Pass@8급 성능을 경쟁적으로 달성한다. 반면 GAE-PPO는 같은 단일 롤아웃 조건에서 거의 학습이 되지 않는 것으로 나타나, long-horizon 수학 추론에 맞춘 Pass@k 기반 토큰 credit의 필요성이 경험적으로 뒷받침된다.



### Reclaim Evaluation: A Lossy Memory Is Worse Than an Empty On (https://arxiv.org/abs/2606.25449)
Comments:
          26 pages, 3 figures. Code, data, and reproduction harness: this https URL

- **Prior Approaches**: 기존 어시스턴트 메모리 기능과 검색 파이프라인은 요약·메모·검색 청크처럼 과거를 압축해 다음 응답에 “필요한 것만” 남긴다는 가정에 의존해 왔다. 또 모델 수정(editing)이나 self-correction 연구도 있었지만, 잘못된 중간 결론이 이미 굳어진 뒤에는 외부 피드백만으로는 복구가 어렵다는 점이 반복해서 지적돼 왔다.

- **Core Contribution**: 이 논문은 압축이 “답을 보존하느냐”뿐 아니라, 나중에 오류를 수정할 수 있는 “정정 가능성(correctability)”까지 좌우한다는 점을 보인다. 특히 잘못된 결론을 남겼는데 그 결론을 만든 근거(source)를 버리면, 모델이 stale한 값을 자신 있게 다시 내면서 수정이 실패하는 brittle memory 현상을 정의하고, 이러한 실패가 여러 모델에서 방향이 뒤집히지 않는 clean kill condition을 만족한다고 보고한다.

- **Technical Challenges**: 문제는 correctability를 모델 성능이나 메모리 크기와 분리해 측정해야 한다는 점인데, 논문은 reclaim evaluation으로 “고정 예산(B) 안에서 압축이 무엇을 남기느냐”만 바꿔 Reclaim Rate(RR)를 judge 없이 정답 매치로 평가하는 paired-memory 프로토콜을 제안한다. 핵심 해결책은 source-first 정책으로, 같은 예산에서 결론을 버리고 재계산 가능한 source를 1줄 메모로 보존해 복구 경로를 다시 살아나게 하며, 길이-일치 대조군으로 단순히 텍스트를 더 줘서 좋아진 효과를 배제한다.

- **Empirical Impact**: 실험 결과 lossy 압축(결론은 남기고 source를 제거)은 같은 예산 조건에서 빈(empty) 메모리보다 더 나쁜 행동을 유발하며, 7개 모델에서 wrong emission이 반복되고 반전이 관측되지 않는다. 반면 source-first는 oracle 수준(1.00)에 못 미치더라도 0.49~0.88 범위로 크게 복구를 회복하고, 메모리가 메모리를 먹는 체인 설정에서는 하나의 dropped-source 오류가 downstream을 누적 오염시키는 반면 source-first는 예산 기반의 horizon까지는 제어 가능한 것으로 나타난다.



### C3-Bench: A Context-Aware Change Captioning Benchmark (https://arxiv.org/abs/2606.25445)
Comments:
          ECCV 2026 Camera-ready version

- **Prior Approaches**: 기존 change captioning 연구는 모델 구조 개선이나 학습 전략 고도화에 집중했지만, 정작 “변화”가 무엇으로 정의되는지 맥락과 기준을 명확히 다루지 못한 경우가 많았다. 또한 많은 벤치마크가 데이터 범위가 좁거나 특정 도메인에 치우쳐 있어, 실제 환경의 다양한 change context에서 모델이 얼마나 잘 일반화하는지 평가하기 어려웠다. 마지막으로 BLEU·ROUGE 같은 기준 기반 지표는 표현이 달라져도 의미가 같은 정답을 제대로 반영하기 어렵고, LMM 출력의 유연한 패러프레이즈를 정밀 비교하기에 한계가 있었다.

- **Core Contribution**: 이 논문은 Context-aware Change Captioning을 평가하기 위한 포괄 벤치마크 C3-Bench를 제안한다. 4개 비주얼 도메인에 걸쳐 51개 현실 change context를 포함하며, 총 4,996개의 사람 라벨 이미지 페어와 맥락별 “무엇을 바꾸었고 무엇은 무시할지/문체는 어떨지” 기준(criteria)을 함께 제공한다. 더불어 change captioning 최초로 LLM-as-Judge 기반의 fine-grained 평가(정확성, 구체성, 유창성, 관련성)와 입력 순서를 바꿔도 의미가 대칭적으로 유지되는지 보는 reversibility metric까지 마련해 신뢰 가능한 비교를 가능케 했다.

- **Technical Challenges**: 핵심 난제는 (1) 다양한 현실 맥락에서 일관된 정답 기준을 정의하고 라벨링하는 비용과 품질을 확보하는 것, (2) open-ended 서술을 사람이 납득할 만한 방식으로 평가하는 준거를 설계하는 것이다. 논문은 여러 change-centric 커뮤니티의 패턴을 distillation해 51개 context를 정의하고, 이미지 페어 구성 시 대응/검증 파이프라인과 품질 통제(중복 제거, temporal verification, 캡션-이미지 정합성 점검 등)를 통해 데이터 오염 위험을 줄였다. 평가 측면에서는 GPT-5.2를 judge로 사용해 fine-grained 차원을 점수화하고, 순서 반전 후 서술이 대칭적으로 일치하는지 reversibility까지 0/1로 측정하도록 프롬프트를 체계화했다.

- **Empirical Impact**: C3-Bench로 32개 모델(전통적 change captioning, 다수 proprietary LMM, 2B~90B급 open-source LMM)을 오프더셋 평가한 결과, 기존 관행의 근본적 blind spot이 드러났다. 기존 conventional 모델은 학습 스타일 규제를 벗어나면 붕괴하며, 심지어 SOTA급 LMM인 GPT-5.2도 도메인·위치(position) 편향에 따른 체계적 오류가 나타나 “신뢰 가능한 change understanding”을 보장하기 어렵다는 점을 보여준다. 또한 reversibility와 fine-grained 지표로 숨겨진 실패 모드를 가시화해, 다음 단계로 일반화 가능하고 신뢰할 수 있는 change captioning을 만들기 위한 구체적 연구 전선을 제시하며 관련 코드·데이터를 공개했다.



### TopoCast: A Topological Fidelity Framework for Evaluating Transformer-Based Time Series Forecasting (https://arxiv.org/abs/2606.25439)
- **Prior Approaches**: 딥러닝 TSF의 평가는 MSE·MAE 같은 점별 오차 지표에 크게 의존해 왔습니다. 이들 지표는 예측값이 값은 비슷해도 위상(phase)·주기·진동 주기 위치 같은 신호의 구조를 놓칠 수 있어, 과스무딩·위상 이동·주파수 왜곡이 생겨도 점수가 좋아질 위험이 있습니다. DTW/Soft-DTW, TDI, Wasserstein distance 등은 시간 정렬이나 분포 유사성을 보완하지만, 예측이 반복 사이클의 “위상공간 위상(topology)”을 보존하는지까지는 직접 평가하지 못합니다.

- **Core Contribution**: TopoCast는 시계열 예측의 구조적 충실도(structural fidelity)를 persistent homology로 평가하는 프레임워크입니다. 예측과 정답을 Takens delay embedding으로 위상공간에 투영한 뒤, Vietoris–Rips filtration의 H1 지속특징을 통해 loop count·dominant cycle strength·total persistence·persistence entropy의 4가지 지표를 뽑아 TFS(Topological Fidelity Score)로 통합합니다. 또한 persistent cocycle을 시간축으로 되돌려 dominant cycle이 “제때” 나타나는지 보는 dominant cycle overlap을 도입해, 국소적 위상 오류까지 잡아내는 LTFS(Localised Topological Fidelity Score)를 제안합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 위상공간 기반 위상학적 특징이 만들어내는 ‘다이어그램 수준’ 요약(TFS)이 시간 국소화 오차를 잃는다는 점입니다. TopoCast는 Ripser에서 얻은 대표 cocycle/생성자를 inverse Takens reconstruction으로 원래 시간 인덱스에 매핑하고, 예측과 정답에서 dominant H1 생성자가 활성화된 시점 집합의 Jaccard overlap로 phase-aware 정확도를 계산합니다. 이를 통해 Wasserstein distance나 DTW 계열처럼 분포·정렬 요약에 머무르지 않고, 지배적 진동 패턴의 시간 위치 오류를 직접 측정합니다.

- **Empirical Impact**: ETTm2·Exchange Rate·ILI의 실데이터 벤치마크에서 5개 Transformer 계열 모델을 평가한 결과, MSE가 비슷해도 LTFS/overlap 프로파일은 크게 갈렸습니다. 특히 Exchange에서는 PatchTST가 점별 지표 1위를 차지했지만 loop injection과 낮은 temporal overlap로 LTFS가 최저였고, 이는 기존 평가가 “구조가 틀렸는데도” 놓치는 고장 모드를 드러냈습니다. 합성 ECG 검증에서도 MSE는 낮게 유지되면서 TFS는 무너질 수 있고, TFS는 맞아 보여도 LTFS(overlap)가 0으로 떨어지는 위상 이동 실패 모드가 명확히 분리되어 제시됩니다.



### Interpretable Concept-Guided Polynomial Tabular Kolmogorov-Arnold Network for EEG-Based Mild Cognitive Impairment Detection (https://arxiv.org/abs/2606.25434)
- **Prior Approaches**: 수면 EEG 기반 MCI(경도인지장애) 탐지는 SVM·KNN·로지스틱 회귀 같은 전통 기법과 Random Forest/XGBoost/LightGBM 같은 앙상블, Bi-LSTM·Bi-GRU·DCNN-SBiL 같은 딥러닝으로 폭넓게 시도돼 왔습니다. 다만 기존 접근은 (1) 스펙트럼·엔트로피·Hjorth·수면 스핀들/슬로 오실레이션 등 서로 다른 생리 도메인 구조를 무시한 채 단일 벡터로 결합하거나, (2) 딥 모델의 비가시성 때문에 임상 신뢰 확보가 어렵고, (3) 도메인(개념) 간 상호작용을 명시적으로 분석하지 못한다는 한계가 남아 있습니다.

- **Core Contribution**: 이 논문은 Concept-guided Polynomial-transformed Tabular learning using Kolmogorov-Arnold Network(CPTabKAN)를 제안해, 생리학적으로 묶인 개념 구조를 보존한 채 상호작용까지 드러내는 탭러 보통학습 프레임워크를 구성합니다. CPTabKAN은 EEG 유래 이질적 특징을 10개 개념 그룹의 개념 표현으로 매핑한 뒤, 2차(degree-2) 다항 변환으로 1차 및 2차 개념 상호작용을 명시화하고, 이후 TabKAN(Fourier-parameterized) 분류기로 비선형 결정경계를 학습합니다.

- **Technical Challenges**: 핵심 난제는 '임상 해석 가능성'을 해치지 않으면서도, 개념 간 비선형 결합을 모델이 관측 가능하게 만드는 것입니다. 연구진은 개념 인코더로 각 도메인 블록을 개념 병목 벡터로 압축(개념 점수 10개)해 구조적 귀납편향을 주고, 다항 확장으로 상호작용 항을 입력 공간에 명시한 뒤, FourierKAN 기반 TabKAN이 learnable univariate edge functions로 비선형 분리를 담당하도록 end-to-end 학습을 설계했습니다.

- **Empirical Impact**: SOF(Study of Osteoporotic Fractures) 수면 PSG 코호트(372명, 1,379 features, 10개 생리 개념 그룹)에서 CPTabKAN-Second Order는 10-fold 교차검증(가중 F1) 기준 0.9038±0.034로 GradientBoosting 대비 5.65%p 향상했으며, fold 수준 통계에서도 사전지정 단일 비교에서 유의한 방향성을 보였습니다. 또한 SMOTE 기반 클래스 균형 하에서도 성능 이점이 유지됐고, 구성요소별 ablation과 개념 중요도 분석을 통해 PSD·multi-scale entropy·Hjorth 등이 1차 기여를, LZW·통계·인구통계·slow oscillations 같은 개념 간 조합이 2차 상호작용에서 두드러짐을 확인해 '임상적으로 납득 가능한 추론 구조'를 제시합니다.



### Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation (https://arxiv.org/abs/2606.25432)
Comments:
          36 pages, see this https URL for more information

- **Prior Approaches**: 기존 추론 효율은 주로 distillation, pruning, quantization, sparse routing 같은 방식으로 모델 자체를 줄여 토큰당 비용(FLOPs/token)을 낮추는 데 집중해 왔다. 하지만 최근에는 출력 길이(verbosity)가 빠르게 늘며, 표준 툴킷이 손대지 않은 “정답까지 필요한 토큰 수”가 전체 Cost-of-Pass를 좌우하는 상황이 커졌다. 또한 chat RLHF와 reasoning 모델에서 길이 편향이 품질 향상을 상당 부분 설명할 수 있다는 연구들이 나왔지만, VLM에서 그 비용 원인이 무엇인지 분해해 검증한 사례는 드물었다.

- **Core Contribution**: 이 논문은 “짧게 말해도 정답이 되도록” 만드는 것이 추론 효율의 누락된 레버라고 주장하며, 그 실현 수단으로 pretraining 데이터 curation을 제안한다. 간결하고 정답적인 데이터로 학습하면 모델이 더 적은 토큰으로 답을 생성하는 경향을 학습해, Cost-of-Pass를 감소시키는 효과를 얻는다는 관점이다. 특히 VLM에서 brevity(간결성)와 quality(정확성), 그리고 두 축의 trade-off가 실제로 어떻게 분리되는지 비용 기준(정답당 FLOPs)으로 답한다.

- **Technical Challenges**: 핵심 기술적 난제는 “더 짧아서 싸진 것”인지 “질이 올라서 덜 틀려 싸진 것”인지, 혹은 둘 다인지 비용-정확성 요인을 분리해 보여주는 것이다. 저자들은 VLM pretraining curation 파이프라인을 MAmmoTH-VL 단일 이미지 subset에 적용하고, curated 데이터로 학습한 모델과 동일 조건의 baseline(Mammoth 데이터) 및 외부 open-weight frontier VLM을 비교하되, cost는 decode-dominated FLOPs proxy로 정답당 비용(Cost-of-Pass)을 계산해 토큰 길이 변수를 직접 가격에 반영한다. 또한 출력 길이 분포가 겹치지 않는 경우까지 고려해 length-matched 회귀/분해 실험으로 brevity와 reasoning-structured verbosity의 효과가 다름을 추적한다.

- **Empirical Impact**: 실험 결과, curated 데이터 학습 모델은 정답당 비용을 크게 낮추면서 정확성도 유지하거나 일부 개선했다. 예를 들어 Datology 4B는 가장 verbose한 4B 비교군(Qwen3.5-4B) 대비 Cost-of-Pass에서 약 35배 이점을 보이면서 정확성 차이는 약 1pp 수준(0.41 vs 0.691? 본문 수치: 정확도 0.691 vs 0.704, 0.41 vs 14.58 TFLOPs per correct)으로 나타났다. 또한 동일 길이 조건에서 curated 모델의 matched-length accuracy가 더 높았고(풀드 평균 +17.55pp), “generic verbosity”는 어떤 스케일에서도 정답 정확성을 안정적으로 올리지 못했으며 reasoning 형태의 verbosity 이점도 스케일이 커질수록 줄어드는 패턴을 확인했다.



### Adaptive Oscillatory Inductive Bias for Modeling Sharp Prosodic Dynamics in Diffusion-Based TTS (https://arxiv.org/abs/2606.25424)
Comments:
          Accepted in INTERSPEECH 2026

- **Prior Approaches**: 확산 기반 TTS는 음향 표현을 점진적으로 정제해 고음질 합성을 이뤘지만, 감정 발화처럼 빠른 피치 변화와 급격한 운율 전이를 안정적으로 모델링하는 데는 여전히 한계가 있다. 기존 디코더는 Snake 같은 periodic nonlinearities로 조화(harmonic) 구조를 유도하지만, 고정된 주기/강도 파라미터는 급격한 진폭·주파수 변화를 충분히 적응적으로 다루기 어렵다.

- **Core Contribution**: 이 논문은 확산 기반 TTS 디코더에서 oscillatory inductive bias의 역할을 분석하고, 입력에 따라 주기적 변조를 제어하면서도 신호 안정성을 유지하는 adaptive oscillatory nonlinearity를 제안한다. 제안한 활성함수를 StyleTTS2 디코더에 통합해 OscillaTTS로 구현했으며, 급격한 운율 전이까지 더 잘 표현하도록 설계됐다.

- **Technical Challenges**: 핵심 과제는 (1) 조화 구조를 학습할 만큼 충분히 주기성을 주되, (2) 감정/전이 구간에서 발생하는 빠른 변화로 인해 그래디언트가 불안정해지지 않게 만드는 것이다. 저자들은 기존 고정 주기 periodic 활성의 적응성 부족을 보완하기 위해 learnable parameter α를 가진 주기 성분을 넣고, linear bypass 성분으로 안정성을 유지해 tanh 포화에 따른 게이팅으로 진동 반응을 조절하도록 구현했다.

- **Empirical Impact**: LJSpeech와 Emotional Speech Dataset(ESD: Happy/Angry/Sad)에서 MUSHRA 스타일 주관평가와 MCD, F0-RMSE 같은 객관평가, AutoPCP 및 WER로 일관된 개선이 관찰됐다. 특히 기준선 대비 스펙트로그램에서 조화 구조 정확도와 피치 궤적 안정성이 전이 구간에서 더 잘 유지되는 경향이 나타났고, ablation에서도 Oscilla(적응형)가 고정 파라미터/대체 활성보다 성능이 가장 좋았다.



### CrossAccent-TTS: Cross-Lingual Accent-Intensity Controllable Text-to-Speech via Disentangled Speaker and Accent Representations (https://arxiv.org/abs/2606.25403)
Comments:
          Accepted at INTERSPEECH 2026

- **Prior Approaches**: 기존 TTS는 화자 자연스러움과 유사성에 집중했지만, 악센트를 화자 임베딩에 암묵적으로 포함시키는 방식이 많아 추론 시 명시적 제어가 어렵다. Indic 언어의 악센트 제어 연구도 상대적으로 부족하며, 데이터가 풍부한 영어 중심 방법은 저자원 환경에 바로 적용하기 힘들다. 또한 L2 악센트 누출을 줄이려는 disentanglement 접근은 일부 효과가 있으나, 악센트 강도를 연속적으로 조절하기엔 제약이 있었다.

- **Core Contribution**: 이 논문은 CrossAccentTTS로, 화자 정체성을 유지하면서도 악센트 변환과 accent intensity의 연속 제어를 동시에 지원하는 프레임워크를 제안한다. 핵심은 Accent Intensity Controller(AIC)로, 악센트 서브스페이스에 가중 언어 임베딩을 주입해 추론 시 악센트 강도를 미세하게 조절하고 악센트 간 보간(interpolation)을 가능하게 한다. 더불어 Accent Suppression Module로 악센트가 화자/스타일 표현에 섞이지 않게 하여 speaker similarity와 자연스러움을 함께 지킨다.

- **Technical Challenges**: 문제는 (1) 악센트를 화자·스타일 특성과 분리하면서도 (2) 학습 데이터 없이 추론 시 악센트 강도를 부드럽게 바꾸는 제어 가능성을 확보하는 것이다. 이를 위해 Neucodec의 discrete acoustic tokens와 Perceiver Resampler의 고정 길이 bottleneck으로 speaker·style을 압축하고, gradient reversal layer(GRL) 기반 adversarial disentanglement으로 악센트/언어 판별 정보를 억제한다. 동시에 언어 임베딩 테이블을 화자·스타일 슬롯에 확장해 더해 주며, Qwen 2.5(0.5B) 디코더가 text와 결합된 제어 신호로 악센트 조건부 토큰을 autoregressive하게 생성하도록 설계했다.

- **Empirical Impact**: Indic Multilingual 및 L2-ARCTIC에서 객관 지표(Accent Similarity, Accent Leakage, Speaker Similarity, UTMOS)와 주관 MOS 모두 개선을 보이며, 악센트 강도 제어가 계획대로 동작함을 확인했다. 특히 가중 language embedding을 적용했을 때 accent similarity가 제어 강도에 비례해 상승해, 연속적 modulations이 실감 가능 수준임을 보여준다. 결과적으로 저자원 Indic 환경에서도 악센트 제어성과 화자 정체성·자연스러움을 동시에 유지하는 실용적 접근으로 평가된다.



### LibEvoBench: Probing Temporal Knowledge Stratification in Code Generation Models (https://arxiv.org/abs/2606.25402)
Comments:
          Accepted at the DL4Code workshop at ICML 2026

- **Prior Approaches**: 기존 연구는 버전이 다른 API 문제를 다루더라도 실행 기반 또는 소규모 수작업 커버리지에 의존하거나, 휴리스틱 매칭으로 인해 버전별 정합성 평가가 약한 경우가 많았다. 또한 템포럴 지식 문제를 다루더라도 코드 생성에서 “요청 버전 기준의 비정시성(anachronism)”을 개별 예측 단위로 분리해 진단하기는 어려웠다.

- **Core Contribution**: 이 논문은 버전별로 공존하는 라이브러리 API를 LLM이 얼마나 정확히 다루는지 체계적으로 평가하는 LibEvoBench를 제안한다. 더불어 Software Evolution Understanding Score(SEUS)를 도입해 안정 API와 진화 API를 함께 보되, 버전 축에서의 일관성과 비정시성 오류까지 함께 패널티로 반영한다.

- **Technical Challenges**: 핵심 기술 난제는 “어떤 API/시그니처가 특정 버전에서 실제로 존재하는가”를 결정론적으로 확정하는 데이터 구축이다. 논문은 문서 인벤토리(Sphinx objects.inv)와 런타임 인트로스펙션을 함께 사용해 버전별 호환성 매트릭스를 만들고, 이를 기반으로 오류를 API 레벨/파라미터 레벨에서 Invalid vs Unknown vs Anachronistic으로 자동 분류하며 SEUS를 계산한다.

- **Empirical Impact**: 실험 결과, 최신 SOTA 모델도 대체로 version-oblivious 성향을 보이며 진화 API에서 최신으로 갈수록 성능이 떨어지고 안정 API는 큰 변화가 없었다. 또한 프롬프트에 target version을 명시해도 이득이 거의 없었지만, 관련 문서(설명)를 주면 10–20점 수준으로 정확도가 오르면서 ‘버전 지식의 구조적 한계’가 드러났다. LibEvoBench와 SEUS는 이후 연구가 temporally grounded/continual-learning 방식으로 소프트웨어 생태계의 버전 차이를 반영하도록 하는 벤치마크 및 진단 프레임을 제공한다.



### Lightweight PCGAE-Net: Parallel CrossGate Attention and Bottleneck AutoEncoder for Efficient 5G Channel Prediction (https://arxiv.org/abs/2606.25401)
Comments:
          6 pages, 4 figures, in review at IEEE GLOBECOM 2026

- **Prior Approaches**: 5G 대규모 MIMO에서 CSI 예측은 LSTM/GRU 같은 순환 구조가 시간 의존성을 다루지만, 장기 의존성과 병렬성이 약하다는 한계가 있었다. CNN은 공간 처리에 강점이 있으나 시간 정보를 함께 모델링하기 어렵고, 최근에는 transformer로 encoder-decoder와 self-attention을 조합해 NMSE를 끌어올리는 흐름이 이어졌다. 그중 CS3T-UNet은 U-Net 백본에 CSA(교차형 공간 attention)와 GTA(그룹별 시간 attention)를 더해 성능이 뛰어나지만, CSA→GTA 순차 적용으로 생기는 구조적 편향과 깊은 병목에서의 비효율적 self-attention이 효율을 갉아먹었다.

- **Core Contribution**: 본 논문은 Lightweight PCGAE-Net으로 CS3T-UNet의 두 가지 설계 결함을 ‘가중치 압축’이 아니라 ‘아키텍처 수정’으로 해결한다. 첫째, CSA와 GTA를 동일 입력에 대해 병렬로 계산하고 CrossGate(채널별 sigmoid 게이트)로 결합해 순차 순서 편향을 제거한다. 둘째, Bottleneck AutoEncoder(BAE)로 깊은 인코더 단계의 특징 채널을 1x1 컨볼루션 기반으로 압축하고, 보조 복원 손실로 압축 정보 붕괴를 막는다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 공간 처리와 시간 처리의 결합 순서가 temporal feature 추출을 왜곡하는 문제와, (2) 깊은 단계에서의 self-attention 비용이 채널 깊이에 대해 제곱으로 증가하는 비용-정확도 문제다. 논문은 CSA/GTA를 같은 layer-normalized 입력에서 병렬 실행하고, CrossGate로 채널별로 공간-시간 혼합을 학습해 편향을 구조적으로 해소한다. 또한 BAE에 보조 reconstruction loss를 추가해, 병목 압축이 디코더가 필요로 하는 채널 구조를 잃지 않도록 정규화하며, 얕은 encoder-decoder와 주파수-지연 도메인 차원 축소로 파라미터를 추가 절감한다.

- **Empirical Impact**: QuaDriGa 데이터셋(3GPP 38.901 UMa NLOS)에서 UE 속도 5/9/12km/h, 예측 스텝 L=1/5 조건을 평가한 결과, 제안 모델은 총 8.54M 파라미터로 CS3T-UNet(20.34M) 대비 58% 감소하면서도 모든 조건에서 더 낮은 NMSE를 기록했다. 특히 single-step에서는 최대 3.26dB(5km/h) 및 6.0dB(9km/h) 개선이 나타났고, multi-step에서도 빠른 이동/긴 예측에서 격차가 유지됐다. 논문은 또한 설계 요소별 ablation으로 병렬 CrossGate가 가장 큰 이득(약 4dB)을 주며, BAE의 보조 손실이 공격적인 압축에서도 성능을 지키는 데 결정적임을 보여 주어, 효율과 정확도를 동시에 노리는 CSI 예측 연구에 실질적 설계 가이드를 제공한다.



### FactorLibrary: From Polynomials to Circuits via Recursive Subgoals (https://arxiv.org/abs/2606.25394)
Comments:
          14 pages, 8 figures, in 3rd AI for Math Workshop (ICML 2026)

- **Prior Approaches**: 기존 연구는 산술 회로를 목표 다항식을 최소 게이트로 구성하는 문제로 보고, 이를 RL로 풀기 위한 bottom-up 또는 top-down 탐색 게임을 제안해 왔다. 예를 들어 CircuitBuilder류는 빈 회로에서 게이트를 하나씩 추가하며 목표를 정확히 재현하도록 학습하지만, 변수/복잡도가 커질수록 조합 탐색공간이 폭증해 성공률이 급락한다. 또 top-down 시나리오는 factorization을 이용해 분해를 설계하면 유효한 행동공간이 줄어든다는 직관이 있으나, 어떤 학습·탐색 조합이 안정적인지 검증이 더 필요했다.

- **Core Contribution**: 이 논문은 산술 회로 탐색을 RL로 수행하되, 하향식(top-down)·상향식(bottom-up) 두 방향 모두에서 FactorLibrary를 공통으로 적용한다. FactorLibrary는 재사용 가능한 “인수분해 가능한 부분식”을 subgoal(재귀 목표)로 저장해 에피소드 전반에 걸친 단기 메모리이자 재사용 가능한 중간 목표 역할을 하도록 설계했다. 특히 top-down에서는 인수분해가 결정론적으로 곱셈 게이트를 제공하므로, 에이전트가 실제로 선택하는 것은 덧셈 분해 split point로 제한되며 효율적인 구성 전략을 학습하도록 한다.

- **Technical Challenges**: 핵심 기술 난제는 회로 크기와 변수 수가 커질 때 행동공간이 지나치게 커져 학습이 잘되지 않는 것이다. 저자들은 bottom-up에서는 Gumbel-PPO-MCTS로 루트 액션 개선과 PPO 업데이트를 결합하고, reward shaping과 completion bonus로 FactorLibrary 기반 재사용 신호를 추가했다. 반면 top-down에서는 AND/OR MCTS(메모이제이션 포함)로 재귀 분해의 비용을 백업하고, PPO+MCTS와 SAC(오프폴리시)을 비교해 탐색 안정성과 계산 효율을 모두 점검했다.

- **Empirical Impact**: 실험은 두 변수 다항식, 유한체 F5에서 수행했으며 bottom-up은 작은 복잡도에서는 99.2%까지 잘 맞추지만 복잡도 10에서는 7.8%로 급격히 무너졌다. 반대로 top-down의 PPO+MCTS는 복잡도 2~8에서 최적 회로를 찾아낸 held-out 성공률 0.918(=91.8%)을 보였고, SAC도 최종 0.889로 경쟁적이며 최고 체크포인트는 0.928에 도달했다. 이는 FactorLibrary와 top-down 분해가 학습된 split/factor 정책을 미경험 목표로 일반화하고, 산술 회로 탐색을 안정적으로 고성능화할 수 있음을 실증한다.



### From Sounds to Scenes: A Benchmark for Evaluating Context-Aware Auditory Scene Understanding in Large Audio Language Models (https://arxiv.org/abs/2606.25391)
- **Prior Approaches**: 기존 Large Audio-Language Models(LALMs) 연구는 ASR, sound event detection, audio captioning 등 음향 단일 층 성능을 중심으로 발전해 왔다. 일부 벤치마크가 speech와 비음성 이벤트/배경을 섞어 넣기도 하지만, 각 층의 시간·논리적 관계를 명시적으로 추론하도록 요구하지는 않는다.

- **Core Contribution**: 이 논문은 Context-Aware Auditory Scene Understanding(CASU)를 새 패러다임으로 제시하며, speech·acoustic events·background가 함께 만들어내는 장면 수준의 의미와 논리 관계 추론을 평가한다. 이를 위해 CASU 벤치마크를 도입하고, 4가지 작업(맥락 기반 QA, 엔터티 추출, 화자 역할 추론, counterfactual reasoning)을 통해 다층 통합 능력을 직접 측정한다.

- **Technical Challenges**: 현실 오디오를 그대로 쓰면 층 간 의미 관계와 시간 정렬이 불명확해 정밀 평가가 어렵다는 문제가 있다. 저자들은 스크립트(JSON)로 층 간 관계를 설계한 뒤, speech는 Zonos 기반 TTS로 생성하고 비음성은 Matching Score 기반 retrieval로 선택한 반실(준)합성 사운드를 time-aligned로 합성하는 파이프라인을 구성해 교차층 인과·논리 제어를 가능케 했다.

- **Empirical Impact**: 실험 결과, 단일 층(특히 speech) 인식이 강해도 CASU 같은 장면 이해에서는 성능이 크게 떨어지는 ‘Perception-Understanding Gap’이 확인됐다. 또한 speech·events·background 중 어느 한 층을 노이즈로 마스킹해도 정확도가 붕괴하며, joint processing이 cascaded(오디오→텍스트 후 추론) 방식보다 counterfactual/contextual reasoning에서 우월함을 보여 “다층 의미 앵커” 통합의 필요성을 실증한다.



### Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis (https://arxiv.org/abs/2606.25390)
Comments:
          Published in Canadian AI 2026

- **Prior Approaches**: 기존 연구는 사이트 간 도메인 차이와 적은 라벨 문제를 GAN, conditional Latent Diffusion(LDM), 혹은 VAE-GAN 같은 생성 모델로 일부 완화해왔습니다. 다만 많은 방법이 2D 중심이거나, 데이터가 매우 적은 극단적 few-shot + 교차 도메인 상황에서 구조적 일관성과 진단에 유효한 병변 특징을 충분히 보장하기 어렵다는 한계가 남아 있습니다. 또한 MRI 합성은 스캐너/프로토콜 차이로 인한 분포 불일치를 근본적으로 줄이지 못해 downstream 성능 하락으로 이어질 수 있습니다.

- **Core Contribution**: ALDM(Anatomically-conditioned Latent Diffusion Model)은 데이터가 풍부한 소스 도메인에서 해부학적 priors를 3D VAE로 학습한 뒤, 타깃 도메인에서는 종양 마스크를 ControlNet으로 조건화해 소량 데이터로도 3D 볼륨을 합성하는 프레임워크입니다. 종양 경계와 위치/형상에 직접 결정을 주는 구조 조건을 latent diffusion에 결합해, 도메인 전이 시 병변의 공간적 일관성을 유지하도록 설계했습니다. 결과적으로 T1, T2, FLAIR 3개 모달리티의 3D 종양 MRI를 데이터 효율적으로 생성하고, 분류에 필요한 진단 특성을 보존하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 3D에서 비용이 큰 diffusion을 어떻게 효율화하느냐, (2) few-shot 타깃 도메인에서도 종양 위치·경계는 지키면서 모달리티별 질감은 현실적으로 바꾸느냐입니다. 논문은 VAE latent 공간에서 diffusion을 수행해 계산 부담을 줄이고, FiLM 스타일 마스크 조절과 ControlNet의 다중 스케일 edge/soft distance conditioning으로 종양 구조를 강하게 가이드합니다. 또한 tumor 영역에 가중치를 둔 diffusion loss와 classifier-free guidance로 해부학적 순응도와 샘플 다양성의 균형을 맞춥니다.

- **Empirical Impact**: ALDM은 타깃 16장(극단적 few-shot) 환경에서 FID 85.40으로 GAN 및 하이브리드 베이스라인을 능가했으며, downstream 분류에서도 AUC 0.987±0.001로 가장 높은 성능을 보였습니다. 정성 결과는 학습이 진행될수록 종양 경계가 선명해지고 모달리티 간(특히 T2/FLAIR) cross-modal 일관성이 좋아짐을 보여줍니다. 이는 저자원이 부족한 임상 데이터에서 생성 기반 data augmentation이 단순 시각 품질을 넘어 진단 성능까지 견인할 수 있음을 실증한 사례로 의미가 큽니다.



### Beyond Visual Forensics: Auditing Multimodal Robustness for Synthetic Medical Image Detection (https://arxiv.org/abs/2606.25375)
Comments:
          Accepted at MICCAI 2026

- **Prior Approaches**: 기존 합성 의료 이미지 탐지는 주로 이미지 단독 입력만으로 평가해 왔습니다. 그 결과 임상처럼 영상과 임상기록(메타데이터)이 함께 들어가고 VLM이 멀티모달로 판단하는 실제 배치 환경의 취약점을 충분히 설명하지 못했습니다. 또한 텍스트를 약간 교란했을 때의 민감도는 다뤄졌더라도, ‘같은 이미지를 고정한 채’ 동반 기록만 바꿔 생기는 의사결정 편향을 체계적으로 정량화한 벤치마크는 부족했습니다.

- **Core Contribution**: 이 논문은 합성 의료 이미지 탐지를 ‘이미지-기록 인터페이스에서의 멀티모달 강건성 감사(audit)’로 재정의합니다. 특히 같은 이미지를 고정하고 메타데이터의 단일 provenance 필드만 교체하는 paired 벤치마크를 제안해, 기록 문맥이 진위 판단을 어떻게 뒤집는지(Text-Induced Decision Shift)를 측정합니다. 이를 통해 모델이 시각 근거보다 기록 컨텍스트에 과도하게 의존하는 취약 패턴을 드러냅니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 현실적인 합성 쌍을 만들면서도 (2) 이미지 자체는 고정하고 (3) 기록 컨텍스트만 통제해 공정한 비교를 가능하게 하는 설계입니다. 저자들은 LLM-guided edit–verify–refine 루프로 이미지 편집을 생성·검증하고, Base 메타데이터에 Source 필드만 Hospital vs AI-edited로 강하게 주입해 상한 스트레스 테스트를 구성했습니다. 이후 I-Only, I+Base, I+Source-H, I+Source-AI의 네 조건에서 FINAL ANSWER와 VISUAL VERDICT를 함께 산출해, 최종 판단 변화가 시각 추론 변형인지(Reasoning Rewrite) 또는 통합 판단 편향인지(Verict Split)까지 분해해 분석합니다.

- **Empirical Impact**: 여러 데이터셋(NIH-CXR14, ISIC2019, PediCXR)과 다양한 VLM 계열(오픈 가중치, general VLM, frontier API, DetectFake VLM)을 대상으로 실험한 결과, provenance 메타데이터만 바꿔도 인증/합성 판정이 크게 흔들렸습니다. 예를 들어 NIH-CXR14의 authentic에서 Source-AI를 넣으면 평균 정확도가 61.1% 하락했으며, 일부 모델은 MedGemma처럼 거의 0%대까지 붕괴했습니다. Fake 이미지에서는 시각 단독 탐지가 이미 어렵지만, Source-AI 메타데이터가 진짜 같은 판단을 ‘시각 개선 없이’ 부풀리거나(허위 TPR 상승) Source-H가 불확실할 때 false negative를 늘리는 등 기록 단서의 지배가 관찰됐습니다. 저자들은 이 효과가 상한 스트레스 테스트임을 강조하며, 이미지 단독 평가가 실제 멀티모달 배치 리스크를 과소평가할 수 있음을 시사하고 향후 멀티모달 강건성 개선의 표준 도구로 활용될 수 있다고 제안합니다.



### Conformal Recovery-Deadline Certificates for Runtime Assurance of Adapting Controllers (https://arxiv.org/abs/2606.25371)
- **Prior Approaches**: 기존 RTA(런타임 어슈어런스)와 Simplex 아키텍처는 안전 집합(Safe set) 경계를 한 번이라도 넘으면 즉시 검증된 안전 컨트롤러로 전환하고, 그 상태를 latch하며 이후 복귀를 허용하지 않는다. 이 규칙은 발산(diverging)에는 타당하지만, 온라인 적응(adapting) 컨트롤러의 ‘회복 전이’가 첫 위반 시점에서 발산과 구분 불가능해지는 문제를 구조적으로 만든다. 결과적으로 능력 있는 컨트롤러의 진단용 탐침이 작동하기도 전에 방어막이 탈락을 강제해 복구 자체를 봉쇄하는 병리가 발생한다. 

- **Core Contribution**: 이 논문은 conformal recovery-deadline certificate(복구 데드라인 인증서)을 제안해, 안전 집합 위반 여부가 아니라 ‘안전 집합 밖에서 벗어나는 기간(회복 시간)’에 대한 상한을 통계적으로 인증한다. 그 상한을 기준으로, 회복이 충분히 일찍 완료될 것이라고 인증되는 구간에서는 적응 컨트롤러의 자율성을 유지하고, 임계 한계에서는 verified monitor로 즉시 안전 모드로 전환한다. 즉, 통계적 커버리지로 자율성(autonomy)을, 검증 백스톱으로 안전(safety)을 각각 분리해 reliability-asymmetric design을 구현한다. 

- **Technical Challenges**: 핵심 기술 과제는 ‘회복 시간’이 마지막-이탈 시간(last-exit time)이며, 어떤 에피소드는 horizon H까지 회복하지 못해 τ=+∞(무한) 질량이 존재한다는 점이다. 단순 수치 모델링은 이러한 +∞를 임의로 버리거나 대체하게 되어 잘못된 유한 데드라인을 만들 위험이 크므로, split-conformal 기반으로 분포-비의존적 분포 커버리지를 구성하고 +∞를 최댓값 원소로 취급한다. 또한 fault 분포가 달라지는 경우를 위해 weighted conformal 및 클래스/그룹 조건에 대한 Mondrian coverage까지 이론적으로 확장해, 커버리지가 적응적으로 유지되도록 설계했다. 

- **Empirical Impact**: 두 가지 서로 다른 Simplex 테스트베드(6-DOF 우주선 자세 제어, 토크 제어 역진자)에서 기존 latching이 동일한 suppression pathology를 유발함을 보이고, 제안 인증서가 이를 ‘동일한 방식의 처방’으로 완화한다. 특히 예시 중 하나에서는 fault 분포 shift가 알려진 상황에서 weighted certificate로 커버리지 복원을 실험적으로 확인하며, 단순 latch·휴리스틱·conformal-on-safety-value 계열과의 비교에서도 이 접근의 실효성이 드러난다. 회복 시간에 대한 데드라인을 RTA 루프에 넣는 도메인-일반 메커니즘이라는 점에서, 안전성과 자율성을 동시에 관리하려는 커뮤니티에 실용적 설계 지침을 제공한다.



### Sarashina2.2-TTS: Tackling Kanji Polyphony in Japanese Speech Generation via Data Scaling and Targeted Data Synthesis (https://arxiv.org/abs/2606.25369)
- **Prior Approaches**: 기존 LLM-TTS는 주로 영어·중국 중심으로 성능이 높게 보고됐지만, 일본은 kanji polyphony 같은 언어 고유의 문맥 의존 읽기 문제가 제대로 최적화되지 않았다. 또한 데이터는 일본 비중이 작고, kanji 읽기 커버리지를 겨냥한 공학적 전략(드문 읽기 보강 등)이 부족한 편이다. 평가도 문장 단위 CER/WER에 의존해 어떤 kanji·어떤 reading에서 오류가 났는지 진단이 어렵고, 일본의 문자 표기 변이 때문에 지표가 흔들린다.

- **Core Contribution**: Sarashina2.2-TTS는 일본에 특화된 LLM-TTS로, (1) 데이터 전략과 (2) 평가 방법론을 함께 설계해 kanji polyphony disambiguation을 정면으로 다룬다. 먼저 일본 화자 발화 194k시간을 포함해 약 361k시간 규모로 학습하고, PronSteering 기반의 targeted data augmentation으로 2,136 Joyo kanji의 long-tail reading까지 커버한다. 더불어 Joyo Kanji Yomi Benchmark와 Kana-CER를 제안해 kanji-level 정확도를 정밀하게 측정한다.

- **Technical Challenges**: 핵심 기술적 난제는 kanji가 문맥에 따라 여러 reading을 갖는 구조적 모호성을 모델이 읽기 선택 단계에서 안정적으로 해결해야 한다는 점이다. 이를 위해 backbone LLM은 텍스트-음성 의미 토큰 생성 역할에 집중하도록 end-to-end로 학습하고, 음소가 반영된 speech tokenizer(S3Tokenizer V2)와 2단계 학습(대규모 사전학습+합성 데이터로 미세조정)을 통해 장기 꼬리 읽기의 빈도 부족을 보완한다. 평가 또한 orthographic variation을 제거해야 했기 때문에, Kana-ASR로 생성 음성을 kana 공간에서 reference reading과 비교하는 Kana-CER로 오차의 원인을 pronunciation에 더 가깝게 귀속시킨다.

- **Empirical Impact**: 실험에서 Sarashina2.2-TTS는 Joyo Kanji Yomi Benchmark에서 모든 CER 기반 지표가 기존 일본 지원 LLM-TTS 대비 우수했으며, 특히 kanji-level reading 정확도에서 큰 격차를 보였다. 합성 데이터 증강(Stage 2)이 Stage 1 대비 전 지표를 개선해 targeted augmentation의 효과를 확인했다. 또한 cross-lingual 평가에서 프롬프트 언어가 일본어가 아니어도 일본 발음이 안정적으로 유지되는 유일한 시스템으로 보고돼, balanced training이 cross-lingual robustness를 준다는 점을 실증했다.



### Reliability-Asymmetric Spacecraft Autonomy: Co-Designing a Capable Learned GNC Stack with a Verified, Adaptation-Aware Runtime Shield (https://arxiv.org/abs/2606.25366)
- **Prior Approaches**: 기존 runtime assurance(RTA)는 복잡한 제어기를 안전 조건 하에서만 허용하고, 위반 시 검증된 단순 제어기로 권한을 되돌리는 Simplex 계열 아키텍처가 주류였다. 그러나 안전 여부를 “즉시 상태”로만 확인하거나, 복귀 시간을 다루지 못하면 온라인 적응형 제어기와 결합 시 비대칭적인 위험 구간이 생긴다. 또한 edge language model 기반 natural language→planning은 문법적 유효성과 의미 정확성 사이의 validity-versus-semantics 갭이 커서, 역으로는 확률 출력의 신뢰성 검증이 어렵다는 한계가 있었다.

- **Core Contribution**: AMPLE-GNC는 역량(지시→계획→제어)과 증명가능성(런타임 안전 보장)을 분리·조합하는 3계층 GNC 스택을 제안한다. 자연어를 PDDL+ 단일 액션으로 매핑하는 foundation-model commander, 선형화 제약을 기준으로 한 verifier, 그리고 fault-adaptive controller를 runtime shield의 안전 불변량(9개)으로 감싼다. 특히 신뢰성 비대칭(reliability asymmetry) 관점에서 “기계검증 가능한 tier가 증명 가능한 경계만 책임지게” 권한을 배치한다.

- **Technical Challenges**: 핵심 난제는 적응형(온라인으로 고장 식별) 제어기가 복구 과도구간에서 의도적으로 안전하지 않을 수 있다는 점이다. AMPLE-GNC는 이 문제를 latching safe-hold shield의 무력화 현상과 함께 진단하고, split-conformal recovery-deadline certificate로 복구 가능 시간을 분포-자유로 인증해 shield 자체를 바꾸지 않고도 안전-복구를 양립한다. commander 쪽은 grammar-constrained decoding(GNBF 문법 기반)으로 출력 유효성을 하드하게 제한하고, semantic 정확도는 de-leaked 분할로 별도 측정해 성능을 정직하게 분리한다.

- **Empirical Impact**: 6-DOF Basilisk 테스트베드에서 commander는 grammar-validity 보장을 유지하면서 planner-executable rate 84%를 보였고, de-leaked novel-phrasing 일반화는 38%(문구 다양성 재파인튜닝 후 48%)로 보고했다. fault-adaptive controller(RMA)는 학습 내 랜덤화 범위에서 sign fault 97.8%, continuous-gain fault 94.4% 복구를 달성했으며 PD·end-to-end RL은 0%에 그쳤다. 더 나아가 복구 deadline 기반 shield 연동은 controller가 94.5% 자율성을 유지하면서도 비복구 케이스를 적시에 포착하도록 하였고, 검증 기반 모니터는 Kind 2로 9/9 불변량의 predictor soundness를 기계검증했다.



### Neural Machine Translation for Low-Resource Tangkhul--English (https://arxiv.org/abs/2606.25365)
Comments:
          11 pages, 3 figures, 9 tables

- **Prior Approaches**: 기존 저자원 MT는 mT5 같은 다국어 pretrained의 transfer, back-translation, unsupervised MT 등으로 성능을 끌어올려 왔지만, 동북인도 티베토-버만 언어는 표준 벤치마크에서 거의 배제돼 Tangkhul에 바로 적용하기 어렵다. 또한 종교 도메인의 Bible parallel corpora 같은 우회 자원은 마지막 대안이지만, 실제 일상·현대 도메인과의 불일치(도메인 편향) 문제가 크다.

- **Core Contribution**: 이 논문은 Tangkhul–English(nmf-en) 저자원 MT를 위해, 공개 대규모 병렬 말뭉치를 구축하고 ByT5-large와 mT5-small을 fine-tuning해 번역 모델을 학습·평가한 것이 핵심 기여다. 특히 Tangkhul은 Latin 기반 diacritics(ā, a̱)가 있어 일반적인 subword 기반 토크나이저가 잘게 쪼개지기 쉬운데, 이를 byte-level 처리로 직접 다루는 접근을 제시한다. 더불어 Hugging Face Hub에 best 모델 tangkhul-byt5와 대비 모델 tangkhul-mt5를 공개해 연구 재현성을 높였다.

- **Technical Challenges**: 기술적 난제는 (1) Tangkhul의 Latin-script diacritics로 인한 토큰화/시퀀스 길이 증가, (2) Bible·이야기·대화로 섞인 학습 데이터의 도메인 편향, (3) 영어로의 단방향 번역에서 발생할 수 있는 의미·표현 정합성 오류다. 저자들은 ByT5가 UTF-8 byte를 그대로 처리해 ā와 a̱를 자연스럽게 다루는 반면, mT5는 a̱가 SentencePiece에서 미인식되어 base 글자와 결합부가 분절되며 시퀀스가 10–15% 길어지는 토큰화 이슈를 분석한다. 또한 빔 서치 빔 크기(최적 4)를 포함한 디코딩 튜닝과, mT5·ByT5 앙상블 리랭킹 시 환각/반복이 악화되는 사례도 보고했다.

- **Empirical Impact**: 실험에서 ByT5-large(tangkhul-byt5)는 BLEU 39.97, chrF++ 58.07, BERTScore F1 0.8104, COMET(wmt22-comet-da) 0.7302로 mT5-small 대비 큰 격차를 보였고, mT5-base zero-shot은 BLEU 0.03 수준에 그쳐 사전학습만으로는 Tangkhul 표현을 학습하기 어렵다는 점을 확인했다. 정량 지표 외에도 수동 에러 분석에서 어휘 치환과 스타일 환각이 주요 실패 모드로 나타났으며, 대화 도메인에서는 톤/의미가 과도하게 확장되는 양상도 관찰됐다. 결과적으로 byte-level 모델 기반 Tangkhul MT의 초기 기준선(baseline)을 제시하고, 비(非)성경 도메인 확장과 back-translation 같은 데이터 다양화가 다음 단계임을 실증적으로 뒷받침한다.



### TheoremGraph: Bridging Formal and Informal Mathematics (https://arxiv.org/abs/2606.25363)
Comments:
          31 pages, 9 figures, 21 tables

- **Prior Approaches**: 기존 연구는 수식·기호를 포함한 수학 문서에서 키워드나 노테이션 중심 검색을 시도했지만, 문장(정리/정의) 간 의존 구조는 상대적으로 약하게 다뤘습니다. 또 일부는 Lean 라이브러리 쪽에서 선언 간 의존을 추출했으나, 범위가 제한되거나(형식화 규모) 혹은 정보가 코퍼스 전반에서 일관되게 연결되기 어려웠습니다.

- **Core Contribution**: 이 논문은 정리(명제) 단위 의존을 비형식(informal)과 형식(formal) 모두에서 한 그래프로 통합하는 TheoremGraph를 제안합니다. 비형식 쪽에서는 arXiv의 theorem-like 환경 1,170만 개에서 1,830만 개의 후보 directed dependency를 복원하고, 형식 쪽에서는 Lean 4 elaborator 수준에서 388,105개 노드와 1,133만 개 typed edge를 갖는 LeanGraph를 공개합니다.

- **Technical Challenges**: 핵심 난제는 서로 다른 표현 양식의 ‘같은 수학 결과’를 안정적으로 연결하는 것이며, 이를 위해 slogan(자연어 슬로건)을 생성해 공유 의미 공간에 임베딩하고 cosine 유사도로 교차 매칭합니다. LLM judge로 검증한 결과 cosine 0.8 이상에서 47,952개의 (inexact 포함) 매치를 확정했고, 유사도 0.9 이상에서는 판단 승인율이 87%까지 올라가 품질을 끌어올렸습니다.

- **Empirical Impact**: 형식 개념 검색에서는 name-and-signature 표현에 graph expansion을 더한 구성이 LeanSearch v2의 reranked Recall@10(0.780) 대비 0.5pp 내(0.775)로 근접하며, 추가 LM reranker 없이도 경쟁력을 보였습니다. 또한 slogan 기반 검색을 RAG로 결합해 LLM의 autoformalization 정확도를 24개 타깃 중 5개에서 8개로 개선하고, 데이터셋·extractor·HTTP API·MCP 인터페이스를 공개해 수학 검색·귀속·검색증강 추론 인프라로 확장 가능성을 제시합니다.



### Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents (https://arxiv.org/abs/2606.25361)
- **Prior Approaches**: RAG 기반 대화 시스템의 기존 연구는 주로 메모리를 어떻게 저장·검색하는지에 초점을 맞췄습니다. 하지만 서로 다른 기능 역할을 가진 메모리가 응답 품질에 어떤 영향을 주는지, 대화 맥락이 달라질 때 에이전트의 응답 방식이 얼마나 달라지는지는 상대적으로 덜 알려져 있습니다. 또한 기존 평가는 주로 reference 기반이라 사용자가 선호하는 뉘앙스를 다른 방식으로 반영할 가능성을 충분히 포착하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 대화 메모리를 역할 중심으로 세분화한 fine-grained taxonomy를 제시하고, 검색된 메모리를 서로 다른 role types로 분류합니다. 더불어 사용자 관점을 모사하는 user-centric 평가 프레임워크를 설계해 응답 차이를 더 정교하게 측정합니다. 그 결과, 메모리 유형별로 응답의 정확도·개인화·제약 인식 같은 특성이 다르게 형성된다는 점을 체계적으로 분석합니다.

- **Technical Challenges**: 핵심 과제는 “메모리의 내용”이 아니라 “기능적 역할”이 응답에 미치는 효과를 분리해 관찰하는 것입니다. 이를 위해 메모리 역할을 세밀하게 분류하고, 서로 다른 대화 맥락에서 retrieval된 메모리가 응답 특성(주제 일관성, 제약 인식, 개인화)에 미치는 영향을 비교하도록 실험 설계를 구성했습니다. 또한 reference 기반 평가의 부족을 보완하려고 사용자 선호를 반영하는 측정 프레임워크를 함께 도입했습니다.

- **Empirical Impact**: long-term 데이터셋과 frontier LLM을 대상으로 비교 실험을 수행한 결과, clarifying memory는 사실 정확성과 constraint awareness를 높여 더 정확하고 개인화된 응답을 유도했습니다. 반대로 irrelevant memory는 topic relevance를 떨어뜨리고 constraint awareness도 저하시켜 응답 품질을 악화시키는 경향이 관찰됐습니다. 이 연구는 메모리 유형을 선택적으로 활용해 개인화된 응답을 만들 수 있음을 실증적으로 보여주며, 메모리 역할 설계/평가 연구에 직접적인 방향성을 제시합니다.



### Compositional Behavioral Semantics for State Abstraction in Reinforcement Learning (https://arxiv.org/abs/2606.25357)
Comments:
          International Conference on Machine Learning 2026

- **Prior Approaches**: 강화학습에서 상태 추상화는 복잡하지만 구조적인 문제로 확장하는 핵심으로 여겨져 왔다. 기존에는 value function, invariants, bisimulation relation, behavioral metrics 등 다양한 형태의 행동 구조를 연구했지만, 상태 추상화에서 “무엇이 증명 가능하게 보존되는가”를 정하는 일반 원칙은 부족했다.

- **Core Contribution**: 이 논문은 행동 구조를 통합적으로 정의·분석하기 위한 프레임워크를 제안한다. 구체 시스템과 추상 시스템 사이에서 어떤 행동 구조가 안전하게(보존되도록) 이전되는지를 보이는 전이 정리와, 논리적 행동 의미에서 정량 지표를 만드는 방법을 함께 제공한다.

- **Technical Challenges**: 핵심 난제는 한 번의 동역학(로컬 1-step) 기술만으로도 행동 의미를 조립(compositional)하고, 추상화 과정에서의 보존성을 엄밀히 증명하는 것이다. 논문은 “로컬 원스텝 설명으로부터 행동 의미를 조립하는” 방식으로 프레임워크를 구축하고, 이를 통해 추상-구체 간 soundness 보장 하의 정량 metric 구성까지 연결한다.

- **Empirical Impact**: 이 연구의 영향은 실험 수치보다도 정리와 정의 원칙의 재사용성에 있다. 상태 추상화 하에서 행동을 추론하는 데 필요한 증명 기반을 넓은 범주의 행동 구조로 확장해, 후속 연구가 새로운 구조를 추가하거나 기존 구조를 안전하게 전이·정량화하는 데 직접 활용할 수 있다.



### AI Coaching for Accelerating Human Skill Development with Reinforcement Learning (https://arxiv.org/abs/2606.25337)
- **Prior Approaches**: 기존 human–AI 협업은 shared control 기반으로 실패를 막는 ‘가디언 에인절’ 역할에 집중해 왔습니다. 그러나 지속적인 assistance는 over-reliance와 skill atrophy를 유발해, 장기적으로 학습 효과를 깎을 수 있다는 지적이 이어졌습니다. 또한 일부 연구는 assistance 강도를 점진적으로 줄이거나 curriculum을 설계하지만, 학습자의 현재 역량과 물리적 런타임 맥락을 동시에 닫힌 고리로 다루는 데 한계가 있었습니다.

- **Core Contribution**: 이 논문은 embodied AI 에이전트가 단순 조정보다는 ‘코치’로서 장기 독립 역량을 키우도록 assistance를 전략적으로 조절해야 한다고 제안합니다. 이를 위해 Human–AI Coaching Game을 비협조적 동적 게임으로 형식화하고, 학습자는 task 성과를 최대화하는 동시에 코치는 학습자의 independent competence(VoI)를 목표로 삼습니다. 결과적으로 AI를 미래의 자기 능력으로 이어지게 하는 personalized mentor 관점으로 전환합니다.

- **Technical Challenges**: 주요 난제는 (1) POSG 수준의 계산 어려움, (2) VoI가 ‘코치가 지금 떠났다면’이라는 반사실(counterfactual) 형태라 추정 비용이 큼, (3) coaching이 학습자의 skill 변화에 간접적으로만 영향을 준다는 점입니다. 저자들은 학습자의 행동이 잠재 skill θ에 의해 좌우된다는 점을 이용해, 확률적 finite-state automaton으로 skill 진행의 인과 효과를 모델링하고 게임을 단일 에이전트 POMDP로 환원해 reinforcement learning으로 학습 가능하게 했습니다. 또한 adaptive closed-loop blending과 θ-기반 skill evolution 모델을 결합하고, counterfactual VoI를 대신하는 대리 보상으로 학습을 안정화했습니다.

- **Empirical Impact**: FPV drone racing 사용자 연구(N=33)에서 L2C 코치는 최신 AI coaching baseline 대비 학습 성과를 유의미하게 개선했습니다. 구체적으로는 reduced lap times와 failure counts 감소로 나타났으며, 단순 성과 보강이 아니라 학습자의 독립적 역량 향상에 초점이 맞춰졌음을 시사합니다. 이는 로보틱스·인간-로봇 상호작용에서 AI를 ‘가드’에서 ‘멘토’로 바꾸는 실증적 근거를 제공한다는 점에서 의미가 큽니다.



### Decoupling Reconnaissance and Exploitation: Measuring the Capability Boundaries of LLM-Based Web Penetration Testing (https://arxiv.org/abs/2606.25332)
- **Prior Approaches**: 기존 연구는 LLM 기반 에이전트를 end-to-end 방식의 black-box 평가에 투입해 침투 역량을 측정해 왔다. 하지만 초기 정찰(재구문/탐색)에서의 오류가 연쇄적으로 전파되면서, 실제 취약점 악용 능력이 가려지는 error cascading 문제가 심각했다.

- **Core Contribution**: 이 논문은 exploit 실행과 reconnaissance를 분리하는 two-stage decoupled evaluation framework를 제안해, 정찰 잡음과 악용 성능을 분리해 측정한다. ground-truth injection과 knowledge-driven ablation으로 정찰 실패가 결과를 덮지 않도록 설계했으며, 웹 취약점 테스트베드에서 세밀한 벤치마킹 프로토콜을 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 에이전트의 실제 exploit 역량을 정확히 분리해내면서도, 정찰 단계에서 발생하는 parsing/추론 오류를 통제하는 것이었다. 저자들은 70개의 고정밀 웹 취약점 테스트베드와 엄격히 정렬된 50개 대표 취약점 부분집합에 대해 정답 컨텍스트 주입과 단계별 지식 기반 ablation을 적용해 reconnaissance noise를 제거/관찰 가능하게 만들었다.

- **Empirical Impact**: 5종 오픈소스 침투 에이전트를 비교한 결과, 정확한 취약점 컨텍스트가 주어지면 functional success rate이 최대 90.0%까지 올라갔지만, 자율 정찰의 targeted vulnerability recall은 약 50.0%에서 정체됐다. 원인은 unstructured telemetry를 파싱하는 과정의 실패로 분석됐고, 아키텍처별로 multi-agent(장문 상호작용), monolithic·graph-driven(단문 체인/세션·접근제어 유형) 등 상이한 역량 니치가 관찰되어 다음 세대 offensive security agent 설계에 실증적 근거를 제공한다.



### Improved Large Language Diffusion Models (https://arxiv.org/abs/2606.25331)
- **Prior Approaches**: 기존 LLM은 주로 autoregressive factorization과 causal attention 중심으로 학습돼 왔습니다. Diffusion language model은 masked diffusion을 쓰되 성능이 초기 시도(LLaDA) 수준에 머물러 강한 autoregressive 모델(Qwen2.5 등) 대비 격차가 남아 있었고, 특히 추론·지시응답 성능을 끌어올리는 레시피와 평가/생성 전략 개선 여지가 컸습니다. 또한 multiple-choice에서는 likelihood 기반 점수만 쓰는 경우가 많아 비교 신뢰도에 한계가 있었습니다.

- **Core Contribution**: iLLaDA는 처음부터 학습하는 8B fully bidirectional masked diffusion language model로, pre-training과 SFT 전 과정에서 masked diffusion objective를 유지합니다. pre-training은 12T 토큰 스케일로 확장하고, SFT는 25B 토큰 instruction corpus로 12 epoch를 수행해 지시·추론 능력을 안정적으로 강화합니다. 아울러 variable-length generation(블록 단위 생성)과 confidence-based scoring(다지선다 평가용 점수 규칙)을 도입해 실제 활용/평가에 맞춘 구성을 제시합니다.

- **Technical Challenges**: fully bidirectional diffusion에서 강한 언어 능력을 끌어내려면 스케일링과 학습 안정성이 핵심이며, 이를 위해 grouped-query attention(GQA), input/output embedding tying, 학습률 스케줄(초기 warmup 후 cosine decay 전환)을 조합했습니다. SFT에서는 기존 방식처럼 프롬프트를 고정 마스크하는 대신 pre-training과 동일한 마스킹 포맷으로 프롬프트·응답·종료 토큰까지 마스킹 가능하게 만들어 variable-length 블록 생성을 자연스럽게 지원합니다. 또한 다지선다에서는 likelihood류 상한 대신 confidence 기반 점수를 사용해 평가 성능을 개선했으며, 생성 반복 루프 이슈는 stop-thinking 토큰 확률을 길이에 따라 조절하는 완화로 다뤘습니다.

- **Empirical Impact**: 실험 결과 iLLaDA는 general(예: BBH, ARC-Challenge), 수학(MATH), 코드(HumanEval) 전반에서 LLaDA 대비 폭넓게 향상됐습니다(예: iLLaDA-Base BBH +21.6, ARC-Challenge +14.9, iLLaDA-Instruct MATH +14.5, HumanEval +16.5). iLLaDA는 비 autoregressive 학습임에도 일부 벤치마크에서 Qwen2.5 7B와 경쟁력을 보였고, 특히 multiple-choice에서는 confidence-based scoring이 likelihood 대비 PIQA/ARC-Challenge/HellaSwag에서 일관된 개선을 보였습니다. 전반적으로 fully bidirectional diffusion을 처음부터 학습해도 강력한 언어 모델로 수렴할 수 있음을 실증해, diffusion 기반 LLM 설계의 경쟁 경로를 강화했다는 의미가 있습니다.



### Supervised Post-training of Speech Foundation Models for Robust Adaptation in Speech Deepfake Detection (https://arxiv.org/abs/2606.25328)
- **Prior Approaches**: 기존 딥페이크(VC/TTS) 음성 탐지는 cepstral·phase 같은 수공특징과 GMM/SVM에서 출발해, 최근에는 AASIST 등 딥러닝 기반 검출기로 time-frequency나 raw waveform을 직접 학습하는 흐름으로 발전했다. 또 WavLM/HuBERT 같은 SSL 스피치 파운데이션 모델은 전이 표현이 좋아 탐지 성능을 끌어올리지만, 사전학습 목표(phonetic/장기 컨텍스트 중심)는 스푸핑의 국소·경계형 잡음/불연속성을 잘 강조하지 못한다. 그 결과 direct fine-tuning은 데이터가 적거나 도메인 이동이 큰 상황에서 과적합·전이 불안정 문제가 커진다.

- **Core Contribution**: 이 논문은 SSL 인코더에 대해 “프레임 수준 감독”을 붙인 supervised post-training을 제안한다. 핵심은 Mix-Frames Post-Training(MFPT)로, 반대 클래스 발화의 일부를 잘라 붙여 국소적인 불연속(경계)을 만들고, 해당 구간에 해당하는 프레임만 스푸핑 단서로 학습되게 유도한 뒤 이후 utterance-level fine-tuning을 진행하는 2단계 적응 전략이다. 즉, 스푸핑이 드러나는 ‘지역적 아티팩트’ 쪽으로 표현 공간을 먼저 재편한다는 점이 기여의 중심이다.

- **Technical Challenges**: 어려운 점은 SSL 사전학습 표현이 마스킹/대조학습 목적에 맞춰져 있어 스푸핑 특유의 국소 불일치 신호를 잘 끌어내지 못한다는 불일치다. 이를 해결하기 위해 논문은 (1) within-utterance cut-and-paste로 짧은 구간에만 perturbation이 생기도록 설계하고, (2) WavLM의 프레임 해상도(대략 50Hz)에 맞춰 주입된 구간 여부를 프레임 라벨로 만들며, (3) 학습 효율을 위해 LoRA로 QKV와 FFN을 선택적으로 적응한다. 마지막으로 후단 검출기는 프레임 정보를 풀링/머징해 발화 단위 분류만 최적화하도록 분리했다.

- **Empirical Impact**: 실험에서 ASVspoof5는 data augmentation 없이 single model로 EER 4.50%를 달성하며 SOTA 수준 성능을 보였다. ASVspoof2021 LA/DF에서는 LA와 DF의 절대 EER 격차가 0.16%로 매우 작아, LA(전송/코딩 왜곡)와 DF(소셜 미디어 기반 조작 음성)처럼 도메인이 다른 조건에서도 균형 잡힌 강건성을 확인했다. 특히 저자원/비동일 도메인 적응(타깃 학습데이터 비율 축소·out-of-domain 평가)에서 post-training이 더 크게 이득을 주어 실전에서 중요한 “전이성” 가치를 입증한다.



### ESTANet: Efficient Online Error Detection in Procedural Videos via Prediction Inconsistency (https://arxiv.org/abs/2606.25317)
Comments:
          18 pages, 8 figures, uses this http URL

- **Prior Approaches**: 절차형 비디오에서 오류를 찾는 기존 연구는 크게 오프라인/온라인으로 나뉘며, 온라인 접근은 과거 프레임만으로 인과적으로 추론해야 한다. 다만 현재 방법들은 영상당 첫 오류만 탐지하거나, 주로 procedural error에 치우치고 실행 오류(execution error)는 충분히 반영하지 못하는 문제가 있다. 또한 LLM 기반 선행 추정(PREGO)이나 손 포즈 같은 추가 입력(MistSense) 의존, 또는 task graph 편차 중심(DTGL)이라 실시간 제약과 일반화가 약해질 수 있다.

- **Core Contribution**: 이 논문은 action detector 자체의 예측 거동이 오류 상황에서 달라진다는 단순한 관찰을 기반으로, 실시간 online error detection을 구성한다. ESTANet(오류에 민감하고 시간적으로 변하는 네트워크)은 standard(안정형)·error-sensitive(민감형) 탐지기 4개를 두고, 이들 예측 불일치의 다수결로 오류 프레임을 플래그한다. 특히 실행 오류와 절차 오류를 모두 겨냥해, detector 간 불일치가 발생하도록 학습 설계와 추론 규칙을 함께 제안한다.

- **Technical Challenges**: 핵심은 (1) 오류 프레임에서만 예측이 자연스럽게 어긋나게 만들고, (2) 어떤 temporal context 길이가 절차 오류에서 차이를 증폭할지 자동으로 정하는 것이다. 논문은 Temporal-Aware Dynamic(TAD) 모듈로 입력 의존적 가중치/바이어스를 만들어 실행 오류에서 prediction inconsistency를 키우며, 시간 창 길이를 다르게 학습한 temporally-varying window 전략으로 절차 오류에서도 불일치를 확대한다. 또한 후보 window 조합을 전수탐색하지 않기 위해 데이터별로 ss는 평균 동작 지속이 가장 짧은 액션을 기준으로, ll은 약 θ% 지점에서 β개의 선행 액션이 포함되도록 선택하는 강건 규칙을 사용한다.

- **Empirical Impact**: EgoPER, Assembly-101-O, EPIC-Tent-O에서 ESTANet은 기존 온라인 오류탐지 최첨단 대비 일관되게 더 높은 성능을 보였다. EgoPER에서는 F1@10/25/50이 각각 47.2/37.8/21.6으로 DTGL(39.6/33.8/20.9)을 상회했고, Assembly-101-O 및 EPIC-Tent-O에서도 Avg-F1이 각각 더 큰 폭으로 개선되었다. 또한 TAD와 temporally-varying 설계가 execution과 procedural 오류에서의 균형 잡힌 탐지 성향을 만들며, 경량 아키텍처로 실시간 효율까지 유지한다는 점에서 실용적 의미가 크다.



### Physics Question Scene Graph: Fine-grained Evaluation of Physical Plausibility in Text-to-Video Generation (https://arxiv.org/abs/2606.25306)
Comments:
          ECCV 2026. Code and data: this https URL

- **Prior Approaches**: 기존 비디오 평가 지표는 시각 품질이나 텍스트-비디오 정합성처럼 전반적인 점수에 치우쳐, 물리 법칙 위반이 ‘어디서/무엇 때문에’ 발생했는지 세분화해 설명하기 어렵습니다. 물리 현실성을 일부 다루더라도 단일 합산 점수나 거친 카테고리 수준에 머물러, 선행 조건(오브젝트 존재, 동작 수행)이 충족되지 않을 때의 판정을 혼동하거나 환각적으로 만들 여지가 있습니다.

- **Core Contribution**: 이 논문은 Physics Question Scene Graph(PQSG)로, 생성 비디오의 물리 법칙 위반을 오브젝트-액션-물리로 분해해 계층형 질문 그래프(DAG)로 평가하는 방법을 제안합니다. VLM이 프롬프트로부터 질문 그래프를 만들고, 다른 VLM이 비디오를 보고 각 질문에 답해 카테고리별·질문별 세밀한 점수와 실패 지점을 제공합니다. 특히 질문 간 논리 의존성을 그래프로 강제해, 선행 조건이 성립하지 않으면 하위 질문을 묻지 않도록 설계한 점이 핵심입니다.

- **Technical Challenges**: 핵심 난제는 (1) 물리 평가를 위한 전제 조건을 먼저 검증하면서도 (2) 질문이 문맥적으로 항상 타당하도록 의존성을 엄격히 유지하는 동시에 (3) VLM 기반 QA의 신뢰도를 확보하는 것입니다. PQSG는 오브젝트 노드→액션 노드→물리 노드로 이어지는 DAG 의존성과 같은 카테고리 내부의 순차 의존성을 두고, 부모가 부정이면 자식 체인을 자동으로 no 처리해 질문 가능성 없는 평가를 차단합니다. 또한 QA를 개방형 답변 생성 후 yes/no 분류로 2단계화해, 단순 이진응답이 체인-오브-쏘트 활용을 저해하는 문제를 완화했습니다.

- **Empirical Impact**: 저자들은 FinePhyEval(195개 인간 주석 프롬프트-비디오 쌍, Sora 2·Veo 3·Wan 2.1 등 생성 모델 포함)을 구축해 PQSG의 세밀한 물리 점수가 인간 판단과 더 높은 상관을 보인다고 보고합니다. 자동화된 PQSG 기반 순위에서는 Sora 2와 Veo 3가 Wan 2.1보다 물리적 사실성에서 더 높게 평가되며, 전반적으로 오브젝트보다 액션과 물리에서 모델들이 더 많이 실패하는 경향이 나타났습니다. 더 나아가 FinePhyEval의 주석을 하위 작업(QG/QA) 벤치마크로도 사용했으며, VLM은 질문 생성(QG)에는 사람과 비슷한 수준에 도달하지만 질문에 답하는(QA) 능력은 아직 인간을 따라가지 못한다는 분석을 제시합니다.



### Communicability-Inspired Positional Encoding (CIPE) (https://arxiv.org/abs/2606.25293)
Comments:
          11 pages, 1 figure, 3 tables; supplementary material includes additional experiments and theoretical proofs

- **Prior Approaches**: 기존 그래프 positional encoding(PE)은 차수나 지역 통계 같은 노드 단서, random walk·cycle·스펙트럼·diffusion 기반 인코딩, 그리고 GPS/GPSE처럼 구조(Structural encoding)와 PE를 결합한 접근 등으로 확장돼 왔습니다. 하지만 많은 방법이 “무엇을 기술할지(서술형 디스크립터)”에 치우쳐, self-attention이 쓰는 내적 유사도(geometry)와 그래프 관련성이 직접 대응되도록 설계되지는 못했습니다. 그 결과 다중 경로(multi-path)로 형성되는 전역 구조를 통합하면서도 attention에 바로 쓰기 쉬운 형태로 정렬된 PE는 상대적으로 부족했습니다.

- **Core Contribution**: 이 논문은 Communicability-Inspired Positional Encoding(CIPE)을 제안하며, communicability(그래프의 모든 길이 경로를 집계한 노드 간 전달력)를 PE의 내적 기하로 직접 옮깁니다. 핵심 아이디어는 Attention-Compatible Geometry 조건으로, 서로 관련된 노드쌍일수록 PE 내적이 meaningful graph structural relatedness(여기서는 diffusion 기반 communicability)를 반영하도록 구성하는 것입니다. 또한 CIPE가 그래프 크기에 따라 차원/스케일이 달라지는 문제를 dimensionality alignment로 해결해, 서로 다른 그래프에서도 동일한 Transformer 임베딩 공간에서 비교·학습 가능하게 만듭니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “그래프 크기 의존적인 CIPE를 Transformer용 고정 차원으로 투사하면서도, 내적 기반 geometry를 최대한 보존”하는 것입니다. 논문은 communicability cosine 형태로 CIPE를 정규화해 상대적 관련성 기하를 만들고, 그 다음 Gram matrix의 내적을 Frobenius 오차 최소화 관점에서 보존하도록 고정 차원으로 정렬하는 최적화(차원 정렬)를 설계합니다. 또한 큰 그래프에서의 O(n^3)급 계산 비용을 줄이기 위해 Chebyshev 다항식 근사와 low-dimensional deterministic approximation을 결합해 heat-kernel 연산을 효율화합니다.

- **Empirical Impact**: 실험에서는 분자(MoleculeNet)와 비분자 그래프(TUDataset) 등 총 14개 벤치마크에서 CIPE가 기존 PE를 평균 35.5%p 개선하며 구조 비편향적(structure-agnostic) Transformer에서 특히 큰 이득을 보였습니다. 구조 편향을 이미 가지는 graph Transformer에도 추가 성능 향상이 일관되게 나타났고, 경쟁 PE들은 종종 미미한 개선에 그쳤습니다. 이 결과는 communicability가 단순한 그래프 통계가 아니라 attention에 “호환되는” 위치 기하를 구축하는 원리로 작동할 수 있음을 실증적으로 보여줍니다.



### EPTS: Elastic Post-Training Sparsity for Efficient Large Language Model Compression (https://arxiv.org/abs/2606.25285)
Comments:
          KDD 2026

- **Prior Approaches**: Post-Training Sparsity(PTS) 기반 기존 가지치기는 보통 특정 sparsity level에 맞춘 단일 최적화/재구성으로 끝나, 다른 sparsity 비율을 요구하면 다시 시간이 드는 절차가 필요했다. SparseGPT, Wanda, RIA 같은 방식은 고속이지만 높은 sparsity에서 정보 손실이 커져 성능이 급격히 떨어지는 한계가 관찰된다. ICP는 중·고 sparsity에서 보완을 시도하지만, 역시 단일 학습 세션으로 한 sparsity에만 대응하는 구조라 다중 하드웨어 시나리오 대응이 번거롭다.

- **Core Contribution**: 이 논문은 Elastic Post-Training Sparsity(EPTS)를 제안하며, 단 한 번의 one-shot 최적화로 여러 sparsity 구성에 강인한 elastic 모델을 만든다고 주장한다. 핵심 메커니즘으로 Multi-Sparsity Hierarchy LoRA(MS-HiLoRA)를 도입해 low-sparsity에서 학습한 보정 능력을 high-sparsity 그룹이 누적 상속하도록 설계한다. 또한 Multi-Sparsity Feature Mixer(MSFM)로 서로 다른 sparsity granularity의 특징을 동적으로 융합해 pruning perturbation에 대한 흔들림을 줄인다.

- **Technical Challenges**: 다중 sparsity를 동시에 맞추려면, pruning으로 인한 loss/정보 손실을 sparsity마다 각각 복원하면서도 파라미터 간 경쟁(parameter competition)을 통제해야 한다. 이를 위해 저자들은 sparsity를 low·mid·high 그룹으로 나누고, 계층적으로 누적되는 LoRA 보정(상위 그룹은 하위 그룹 모듈을 재사용)을 통해 Nested Information Loss 가설에 기반한 계층 의존성을 학습하도록 구성한다. 더불어 block-wise 재구성 과정에서 MSFM으로 여러 sparsity 수준의 특징을 가중 선형 결합해 후속 블록 입력 분포 변화를 완화하는 방식으로 안정성을 확보한다.

- **Empirical Impact**: LLaMA와 OPT 계열에서 EPTS는 SparseGPT, Wanda, RIA, ICP 대비 낮은 수준(예: 30~50%)에서는 비슷하거나 경쟁력 있는 성능을 보였고, 특히 높은 sparsity(예: 60~70%)에서 격차가 크게 벌어졌다. 예를 들어 LLaMA-7B에서 70% sparsity의 perplexity는 Wanda/RIA가 크게 악화한 반면 EPTS는 낮은 수준을 유지하며, zero-shot 과제 정확도 평균도 sparsity가 높아질수록 상대적 우위가 나타났다. 무엇보다 단일 최적화로 여러 sparsity를 즉시 전환할 수 있어, 다중 하드웨어 배포 시나리오에서 운영 효율(재최적화 비용)을 실질적으로 줄이는 영향이 강조된다.



### Heterogeneous and Adept Snapshot Distillation for 3D Semantic Segmentation (https://arxiv.org/abs/2606.25278)
Comments:
          11 pages

- **Prior Approaches**: 기존 3D semantic segmentation 성능 향상에는 multi-modal fusion과 model ensembling이 많이 쓰인다. 하지만 multi-modal fusion은 point cloud와 image를 함께 처리하느라 계산·메모리 부담이 크고, ensembling은 expert 모델 수가 늘수록 학습/추론 비용이 선형으로 증가해 실서비스 적용이 어렵다. 또한 cross-modal knowledge distillation에서도 학습용 multi-modal 입력을 어떻게 고르느냐가 성능에 큰 영향을 주는 한계가 있었다.

- **Core Contribution**: 이 논문은 단일 point-cloud 기반 네트워크가 multi-modal 모델과 여러 expert의 지식을 흡수하되, 추론 시에는 image 없이 동작하도록 하는 HAS-KD를 제안한다. 핵심은 Information-oriented Heterogeneous Distillation (IHD)로 multi-modal teacher의 정보를 uni-modal student에 증류하고, Adept Snapshot Distillation (ASD)로 학습 중 생성된 snapshot들을 class별 expert로 취급해 multi-teacher 지식을 효율적으로 옮기는 것이다. 결과적으로 추가 inference 부담 없이 정확도를 끌어올리는 것을 목표로 한다.

- **Technical Challenges**: 문제는 (1) cross-modal distillation에서 어떤 image를 선택해야 teacher 품질이 올라가는지, (2) 여러 teacher를 함께 쓸 때 underperform하는 teacher의 잡음을 어떻게 줄일지이다. IOF는 연속 프레임에서 각 객체에 대해 semantic abundance가 최대가 되는 관측을 골라 더 informative한 multi-modal teacher를 만들고, ASD는 class별 최고 성능 체크포인트만 해당 class voxel에 대해 supervision을 제공해 ‘적재적소’ 증류를 수행한다. IHD는 distillation으로 학생이 multi-modal과 유사한 정보를 학습하게 하되, 실제 추론 단계에서는 image 분기 없이 단일 모델만 사용하도록 설계된다.

- **Empirical Impact**: ScanNetV2와 S3DIS에서 HAS-KD는 state-of-the-art 성능을 달성하며, 특히 multi-dataset 학습 없이도 강력한 baseline(Point Transformer V3)보다 mIoU를 추가로 개선한다. ablation에서는 IHD와 ASD 각각이 유의미한 성능 향상을 주고, IOF가 Random Filtering 대비 더 나은 teacher 및 학생 성능을 만든다는 점이 확인된다. 또한 class별 expert ensemble이 평균 기반 ensemble보다 더 높은 ensemble teacher 품질과 학생 이득을 제공해, 효율적인 증류 전략의 실효성을 보여준다.



### UC-Search: Risk-Aware Test-Time Search for Delayed Constrained Time-Series Contro (https://arxiv.org/abs/2606.25274)
- **Prior Approaches**: 기존 시계열 모델들은 주로 예측(forecasting) 성능이나 표현 학습 자체를 최적화해, 실제 배치 환경의 지연 의사결정과 강한 제약(가능성 제약)을 직접 다루기 어렵다. 강화학습은 순차 의사결정을 다루지만 학습·정책 실행 중심이라, test-time에 가벼운 탐색 레이어를 얹어 ‘가능한 궤적 중 최선의 첫 행동’을 고르는 구조는 제한적이다. 불확실성 기반 MCTS/UCT 계열도 있으나, UC-Search는 별도의 학습된 dynamics 모델 없이 예측/액션 스코어와 hard feasibility automaton을 결합해 지연된 제약 제어 문제에 맞춘 설계로 차별화된다.

- **Core Contribution**: UC-Search는 모델-agnostic test-time wrapper로, 백본이 예측(또는 action score)을 내면 feasibility automaton이 후보 경로를 진행시키고, bounded search가 위험-조정된 feasible trajectory의 ‘첫 행동’을 반환한다. 핵심은 불확실성이 단순 신뢰도 캘리브레이션을 넘어, 경로 위험(path-risk) 항과 first-action 선택에 직접 반영될 수 있다는 점이다. 저자들은 UC-Beam(주된 bounded search)과 UCT-style 진단형 UC-MCTS를 제시하고, 탐색이 1-step risk-greedy로 붕괴되는 경우와 지연 제약 결합이 비단기 가치(non-myopic value)를 만드는 구조 조건을 이론으로 정리한다.

- **Technical Challenges**: 가장 큰 기술적 난점은 (1) epistemic·aleatoric·propagated uncertainty를 경로 비교 가능한 ‘위험 점수’로 일관되게 전환하고, (2) 가능한 궤적만 남기기 위한 pruning/검색 예산에서 계산량을 통제하면서도 지연된 제약 결합 효과를 포착하는 것이다. 저자들은 ensemble 또는 이종분산 헤드로 평균·불확실성을 추정하고, 후보 확장 시 uncertainty-adjusted top-k로 자식 생성 및 빔/UCT 선택을 수행한 뒤 propagated uncertainty를 경로 위험으로 누적한다. 또한 myopic-collapse/separation 정리로 ‘충분히 특정한 구조에서는 붕괴하고, 지연 feasible-set coupling이 있으면 붕괴하지 않는다’는 경계 조건을 제시하며, retained-prefix/feasibility 프루닝 실패 모드가 언제 경쟁력을 깨는지까지 진단한다.

- **Empirical Impact**: 실험은 사전 공개된 delayed-control 벤치마크(9-family, 33-series, held-out starts)에서 UC-Pareto가 validation-selected CEM, MPPI, risk-aware random보다 compute-정규화된 임계 구간에서 일관되게 우수하고, compute-matched audit에서도 긍정적 결과를 유지한다. ETT/LTSF delayed-inventory 검증 역시 같은 compute-frontier 해석을 지지하며, 48-series M4 raw periodic-review lost-sales inventory audit에서는 UC-Pareto가 강한 classic base-stock control 및 CEM·risk-random을 크게 앞선 반면 MPPI는 family-mixed 성향을 보인다. 논문은 모든 시계열 의사결정을 대체하는 만능 주장보다는, boundary/메커니즘 증거와 재현성 아티팩트를 구분해 ‘예측 불확실성 기반 test-time bounded search’가 지연 제약 제어에서 의미 있는 이득을 줄 수 있음을 실증적으로 보여준다.



### Stabilizing black-box algorithms through task-oriented randomization (https://arxiv.org/abs/2606.25269)
- **Prior Approaches**: 기존 연구들은 black-box 출력의 안정성을 높이기 위해 bagging 기반 resampling(예: bootstrap, sub-sampling)을 활용해 왔다. 다만 표준 재표집은 입력의 생성 구조를 반영하지 못해 편향을 만들 수 있고, 분포 정보를 알기 어려운 실제 데이터에서는 성능과 효율이 흔들린다.

- **Core Contribution**: 본 논문은 입력 데이터의 생성 메커니즘에 맞춰 무작위화를 조절하는 task-oriented randomization을 제안해 black-box 출력을 안정화한다. 핵심은 데이터 자체를 noisify해 여러 개의 nosified dataset을 만들고, 그 결과를 averaging/major voting으로 앙상블해 안정성을 확보하는 framework이다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 분포 구조를 알 수 없는 복잡한 입력에서 어떤 방식으로 noisify해야 하는지, (2) noisification이 안정성과 탐색(exploration) 능력을 동시에 흔드는 trade-off를 어떻게 정식화할지다. 논문은 알려진 생성 메커니즘이면 이에 맞춘 잡음(예: Gaussian/Laplace 계열)으로, 미지이면 diffusion 기반 forward-to-reverse 방식으로 nosified dataset을 생성하도록 구성하고, (s,δ)-stability 같은 이론적 안정 보장을 제공한다.

- **Empirical Impact**: 이 방법은 합성 실험에서 baseline 대비 출력 민감도를 낮추고, 잡음 강도를 키웠을 때 stability와 prediction error가 어떻게 변하는지까지 체계적으로 보여준다. 또한 LLM의 scoring 구조를 활용해 top-k ranking으로 확장한 뒤, 실제 데이터 적용을 통해 현실 문제에서도 안정성과 효율을 함께 달성할 수 있음을 시사한다.



### ASAP: Agent-System Co-Design for Wall-Clock-Centered Auto HPO Research for ML Experiments (https://arxiv.org/abs/2606.25207)
- **Prior Approaches**: 기존 HPO는 서러게이트 모델(예: Gaussian process, TPE, SMAC 등)과 획득함수를 통해 다음 하이퍼파라미터를 순차적으로 고르는 방식이 주류입니다. 최근에는 LLM을 surrogate/agent로 써서 iteration별 성능을 높이려는 시도가 있으나, LLM도 사전학습 기반의 단일 inductive bias를 그대로 가져 “대체(replacement)” 형태로 동작하는 한계가 남습니다. 또한 대부분의 평가는 iteration count에 치우쳐 있어, 실제 실행에서는 LLM 추론과 도구 실행 오버헤드가 직렬로 누적되며 wall-clock 이득이 줄어드는 문제를 충분히 다루지 못합니다.

- **Core Contribution**: 이 논문은 ASAP(Agent-System co-design for Wall-clock)를 제안하며, LLM 기반 HPO를 “대체”가 아니라 “통합”으로 전환합니다. ASAP는 여러 HPO 도구(서러게이트 기반 통계 도구 + LLM-as-proposer)를 한 에이전트(LLM-as-Judge)가 매 라운드 후보 중에서 조합 선택하도록 설계해, 특정 사전의 편향에 취약했던 구조를 완화합니다. 동시에 설계 목표를 iteration count가 아니라 end-to-end wall-clock으로 바꾸고, 에이전트(프롬프트/판단)와 시스템(실행/스케줄링)을 함께 최적화합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 서로 다른 inductive bias를 가진 제안들을 한 라운드에서 “선택 가능한” 형태로 결합하고, (2) LLM·도구 비용을 직렬로 다루면 생기는 지연을 줄이면서도 선택 품질(레그렛)을 유지하는 것입니다. ASAP는 매 라운드 후보 집합을 도구 풀의 union으로 만들고, LLM-as-Judge가 in-context learning 형태로 후보를 평가해 선택하게 하여 통합의 실효성을 확보합니다. 여기에 KV-cache를 재사용하는 prefix-stable 프롬프트, 모델 평가와 LLM/도구 지연을 겹치게 하는 speculation parallelism, 그리고 critical path 밖에서 speculation threshold를 튜닝하는 Self-Tuner로 wall-clock을 절감합니다.

- **Empirical Impact**: 다양한 현대 HPO 벤치마크에서 ASAP는 여러 기준선 대비 일관되게 더 좋은 성능/효율을 보이며, 특히 단일-bias 도구들이 약한 지형(거칠고 다중 양상이며 속이기 쉬운 경우, 이방성 등)에서도 견고함을 보여줍니다. 분석 결과는 per-task가 아니라 다양한 작업군에서 레그렛이 wall-clock 단위로 개선되는 방향이 반복적으로 관찰됩니다. 종합하면 ASAP는 LLM-HPO의 “반복 횟수 향상” 중심 평가를 넘어, 실제 서비스 실행 비용까지 고려한 에이전트-시스템 공동 설계의 실용성을 확립한 것으로 의미가 큽니다.



### RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory (https://arxiv.org/abs/2606.25206)
Comments:
          Project website: this https URL

- **Prior Approaches**: 기존 로봇 장기 배치 메모리는 라벨이 고정된 의미 맵(카테고리)이나 3D 포인트클라우드 같은 지표 기반 표현에 의존해, 질의가 특정 세부를 요구할 때(예: 작은 색/모양 디테일) OOV(미등록) 문제가 쉽게 발생한다. 또 다른 접근은 오픈보캐뷸러리를 위해 관측을 caption으로 바꾸고 텍스트 임베딩을 저장하는데, 이 image-to-text 캡션 병목이 시각 디테일을 손실시켜 검색이 취약해진다.

- **Core Contribution**: RAVEN은 캡션 대신 raw visual embeddings(시각 임베딩)를 그대로 저장하는 agentic memory 시스템으로, 장기 로봇 질의응답과 내비게이션을 지원한다. 관측 프레임마다 시각 임베딩에 pose(자세)와 time(시간)을 함께 묶은 visuo-spatio-temporal memory triplet을 vector database에 넣고, 공간 지도 기반으로 검색을 grounding해 답변과 목표 이동을 동시에 수행한다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트로 압축하지 않으면서도 fine-grained 의미를 유지하고, (2) 수천 프레임 단위의 긴 기간에서도 검색 비용을 억제하며, (3) 검색 결과를 로봇 플래닝에 연결해 정확한 목표 좌표를 뽑는 것이다. RAVEN은 FAISS/Milvus 같은 벡터 검색으로 top-K 검색을 sub-O(N) 수준에 가깝게 처리하고, VLM이 도구(텍스트/시간/위치/이미지 기반 검색)를 유한상태머신 루프에서 반복 호출해 working memory를 줄이면서 정밀도를 높인다.

- **Empirical Impact**: 여러 시뮬레이션 및 실제 비디오 QA 벤치마크에서 RAVEN은 caption 기반 메모리 시스템을 일관되게 능가하며, frontier VLM과 비교해 장기 과제 성능을 유지하면서 검색 비용은 10배 낮췄다고 보고한다. 또한 Unitree Go1을 실제 환경에 배치해 자연어 목표 도달 내비게이션을 성공적으로 시연했고, 0.12 fps 다운샘플링에서도 성능의 대부분을 보존하며 원시 RGB 대비 250배 이상 저장 압축과 높은 메모리 효율을 보였다.



### FDN: Interpretable Spatiotemporal Forecasting with Future Decomposition Networks (https://arxiv.org/abs/2606.25201)
Comments:
          19 pages

- **Prior Approaches**: 시계열·공간·시간이 얽힌 spatiotemporal 시스템의 예측에는 최근 SOTA 수준 성능을 내는 고도화된 모델들이 다수 제안돼 왔습니다. 다만 대부분은 예측 정확도에 집중해 해석 가능성(interpretability)은 상대적으로 덜 다뤄져 왔고, 결과적으로 모델이 무엇을 학습했는지 파악하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 Future Decomposition Network (FDN)을 제안하며, 예측 결과를 분류 기반으로 해석 가능하게 제공하는 기능을 포함합니다. 동시에 타깃 시계열에서의 잠재 활동 패턴(latent activity patterns)을 드러내고, 정확도 면에서도 기존 SOTA와 경쟁하면서 메모리와 런타임 비용을 크게 줄이는 것을 목표로 합니다.

- **Technical Challenges**: 해석 가능 예측과 고성능 예측을 동시에 달성하려면, spatiotemporal 신호의 상호 의존성을 유지하면서도 분해(decomposition)를 통해 의미 있는 내부 표현을 설계해야 하는 기술적 난제가 있습니다. 논문은 FDN 구조를 통해 예측을 해석 가능한 분류 형태로 연결하고, 잠재 활동 패턴을 추출하는 방식으로 이를 해결하며, 계산 비용도 효율적으로 관리한 것으로 제시합니다.

- **Empirical Impact**: 수문(hydrologic), 교통(traffic), 에너지(energy) 시스템의 여러 데이터셋에서 FDN을 종합 분석한 결과, 정확도 향상과 함께 해석 가능성 측면의 이점이 확인됐습니다. 특히 SOTA 수준의 성능을 더 낮은 메모리·런타임 비용으로 제공한다는 점에서, 실무 적용과 모델 신뢰성 확보에 의미 있는 진전을 시사합니다.



### A Hybrid CNN-LSTM Intrusion Detection Framework for Cybersecurity in Smart Renewable Energy Grids (https://arxiv.org/abs/2606.25200)
- **Prior Approaches**: 스마트 그리드 IDS는 일반적으로 reactive(서명/이상탐지)와 proactive(IPS/MTD/DL 기반)로 나뉘며, 기존 연구는 SVM·Random Forest·KNN처럼 개별 flow를 독립 샘플로 취급하는 방식이 많았다. 이 접근은 multi-step 공격의 시간적 전개를 모델링하지 못하고, CICIDS2017 같은 극단적 클래스 불균형에서 희귀 공격(예: infiltration)의 재현율이 쉽게 무너진다.

- **Core Contribution**: 본 논문은 Hybrid CNN-LSTM IDS를 제안해, CNN으로 공간(피처) 패턴을 추출하고 LSTM으로 time-series 흐름을 순차 모델링해 즉각적 볼류메트릭 이상과 low-and-slow로 서서히 진행되는 공격을 동시에 탐지한다. 또한 결측/정규화/인코딩부터 SMOTE 기반 클래스 밸런싱, mutual-information 기반 feature selection, causal temporal sequence 구성까지 7단계 전처리 워크플로를 체계적으로 포함한다.

- **Technical Challenges**: 핵심 난제는 (1) multi-step 공격의 시간 진행을 반영하는 학습 설계, (2) 클래스 분포가 심하게 치우친 벤치마크에서의 스케일러블한 성능 유지, (3) 서로 다른 네트워크 환경 간 일반화다. 저자들은 SMOTE를 학습 분할에만 적용해 누수 없이 불균형을 완화하고, 훈련 fold에서만 정규화·feature selection·temporal sequence를 fit하도록 구성해 generalization과 실시간 추론을 함께 노렸다.

- **Empirical Impact**: 실험 결과 Hybrid CNN-LSTM은 CICIDS2017과 NSL-KDD에서 기존 ML 및 단일 CNN/LSTM 대비 2~9%p 수준의 격차로 향상됐고, 특히 ablation에서 SMOTE가 가장 큰 기여를 보였다(미적용 시 F1이 -3.7%p~). 추론 성능도 GPU 27,800 flows/s, CPU FP32 0.082ms/sample 수준을 보이며, INT8 quantization은 0.3% 정확도 손실로 3.1x 가속을 추가로 제공해 <128MB 메모리 IED 엣지 배치 가능성을 입증했다.



### SoK: AI Secure Code Generation: Progress, Pitfalls, and Paths Forward (https://arxiv.org/abs/2606.25195)
- **Prior Approaches**: AI 보안 코드 생성 연구는 보통 프롬프팅, fine-tuning, reinforcement learning, 그리고 generate–check–revise 같은 에이전트 워크플로로 취약 코드를 줄이려 해왔다. 다만 기존 평가는 결과물의 통과/실패 판정(테스트 통과, 취약점 디텍터/오라클 회피)에 의존해, ‘왜’ 실패했는지(원리 미이해·적용 실패·부분/잘못된 완화·기능 파괴 등)를 분해하기 어렵다. 특히 정적 분석 기반 평가는 언어·프레임워크·규칙 커버리지 편차 때문에 실제 익스플로잇 가능성을 가릴 수 있어 진단 한계가 더 커진다.

- **Core Contribution**: 이 논문은 AI secure code generation의 진행 상황을 체계화하기 위해 Kauge(Knowledge–Actuation Unified Gap Evaluation)라는 3단 프레임워크를 제안한다. Layer 1은 secure coding principles(SCPs)를 자연어로 이해·추론하는 능력(knowledge), Layer 2는 생성 코드가 기능 요구를 만족하면서 익스플로잇을 막는 능력(actuation), Layer 3는 둘 사이의 지식–실행 격차(gap)를 메커니즘 단위로 진단한다. 이를 통해 보안 성패를 ‘출력’이 아니라 ‘원리 인지→코드 실행→방어 메커니즘 구현’의 연결로 설명하려는 것이 핵심이다.

- **Technical Challenges**: 핵심 난제는 SCP 이해와 코드 보안 성공을 동일하게 취급하는 기존 벤치마크의 블라인드를 제거하는 것이다. 연구진은 OWASP/CERT의 SCP를 기반으로 자연어 이해용 NLU 벤치마크, 익스플로잇-대응 SCP 방어 전략 매핑, 그리고 SCP-compliance judge를 만들어 knowledge와 actuation을 분리 측정했고, 함수 수준(CWEval)·웹 애플리케이션 수준(BaxBench)에서 모두 실행형 오라클로 검증했다. 그 결과 모델이 관련 원리를 ‘알아도’ 구현 경계나 방식이 어긋나 secure하고 functional한 코드로 번역되지 않는 패턴이 반복적으로 관측되었다.

- **Empirical Impact**: 실험에서는 SCP에 대한 자연어 이해(knowledge)가 기능 정합성과 보안, 그리고 두 가지를 동시에 만족하는 결과까지 통계적으로 유의미한 예측 변수가 되는 것으로 나타났다. 그러나 actuation은 knowledge보다 체계적으로 낮아, “원리 이해는 되지만 방어 메커니즘을 코드에 정확히 작동시키지 못하는” knowledge–actuation gap이 남아 있었다. 논문은 이를 바탕으로 principle-guided generation의 sink-aware화, 실행 피드백을 통한 functionality-preserving 최적화, 그리고 메커니즘을 보고하는 벤치마크/에이전트 워크플로가 분야를 앞으로 옮길 구체 경로라고 제시한다.



### What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics (https://arxiv.org/abs/2606.25182)
Comments:
          Accepted at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD) 2026. A short version accepted at EIML@ICML 2026

- **Prior Approaches**: 기존 LLM jailbreak 탐지는 프롬프트/출력에 기반한 규칙·분류기·이상탐지, 또는 내부 표현을 읽는 internal-signal 계열로 나뉜다. 그런데 많은 방법이 “어떤 내부 신호가 harmful intent를 담는지”와 “토큰이 전개될 때 불확실성이 어떻게 변하는지”, 그리고 “모델 깊이별로 어디서 신호가 강한지”를 명확히 특정하지 못했다.

- **Core Contribution**: 이 논문은 학습 없이(freezing, training-free) logit lens로 중간 레이어의 token-level predictive entropy를 추적해, jailbreak이 “불확실성의 총량”이 아니라 “토큰 위치에 따른 불확실성 동역학(trajectory)”에 구조화돼 있음을 보인다. 특히 Kendall’s τ, Spearman’s ρ, monotonicity 같은 rank-based 추세 특징이 평균/분산 같은 정적 요약보다 훨씬 잘 jailbreak과 benign을 분리한다고 제시한다.

- **Technical Challenges**: 핵심 난제는 harmful intent가 내부 표현에 존재하더라도, 이를 안정적이고 비교 가능한 수치 특징으로 바꾸는 것이다. 저자들은 단일 forward pass에서 중간 히든을 logit lens로 어휘 공간에 투영해 다음 토큰 엔트로피 시계열을 만들고, 정적(level) 특징과 동적(trend) 특징을 분리해 AUROC로 비교하며, 모델별 “해로운 방향(downward/upward)” 차이도 고려해 directional AUROC를 사용해 공정 평가를 수행한다.

- **Empirical Impact**: Llama, Qwen, Gemma의 여러 adversarial 벤치마크에서 동적 trend 특징은 추가 학습 없이 일관된 분리를 보였고, 신호는 중간 레이어(대략 50–85% 구간)에 집중된 뒤 최종 레이어에서 약화된다. 또한 단순 유사 스타일 안전셋(UltraChat)은 비교적 잘 분리되지만, 구조적으로 악의에 가깝게 만든 JailbreakBench benign에서는 분리가 붕괴해 이 신호가 “문장 의미의 악의성”보다는 “구조적 프롬프트 구성”을 포착함을 시사한다.



### Phoneme-Level Mispronunciation Screening in Polish-Speaking Children with an Explainable Assistan (https://arxiv.org/abs/2606.25181)
Comments:
          Accepted to INTERSPEECH 2026. 5 pages, 1 figure, 4 tables

- **Prior Approaches**: 기존 아동 발음 오류 탐지는 Goodness-of-Pronunciation에서 출발해 phoneme/feature 수준을 겨냥하는 neural 접근으로 발전했지만, 임상에서의 전문 인력 부족과 긴 대기시간 때문에 ‘진료실 밖’ 조기 선별 도구가 필요하다는 문제의식이 크다. 또한 end-to-end ASR은 language-model priors가 비정상 발화를 ‘그럴듯한 단어’로 정규화해 최소 대비(예: s vs sz) 기반 오류 탐지에 불리할 수 있다.

- **Core Contribution**: 이 논문은 폴란드어 아동의 치찰음(sibilant) 치환을 타깃으로 하는 경량 선별 파이프라인을 제안한다. wav2vec2 기반 CTC 토큰 인식기와 정렬(alignment) 기반 오류 유형화, 그리고 clinician 템플릿에 근거한 보호자 보조(진단이 아닌 screening)까지 ‘선별 루프’를 한데 묶었다.

- **Technical Challenges**: 아동 음성은 성인과 달리 음향 차이·화자 내 변동성·발달 단계의 비표준 실현이 커서, 일반 ASR을 그대로 쓰면 오류 탐지가 왜곡될 위험이 있다. 저자들은 CTC 토큰 인식에 post-encoder를 보강하고 bracketed IPA 토큰으로 ‘치환 증거’를 별도 클래스화한 뒤, 최소 편집 기반 정렬로 목표 구간의 불일치만 보수적으로 플래그하며 저신뢰 시 재녹음을 요청하는 안전 경계를 둔다.

- **Empirical Impact**: 보이지 않은 10명의 아동(559 발화)에서 인식기는 exact sequence match 88.7%를 달성해 토큰 정렬 기반 후처리가 안정적임을 보여줬다. 선별 프록시(목표 구간에서 bracketed substitution evidence가 나오면 mismatch 플래그)는 precision 72.9%, recall 61.4%, F1 0.67이며 목표-정상 항목에서 false-alarm rate 2.7%로 보수성을 입증했다.



### ATMA: Length-Invariant Language Modeling via Polar Attention and Gated-Delta Compression Memory (https://arxiv.org/abs/2606.25156)
- **Prior Approaches**: 기존 LLM의 길이 일반화는 softmax scaled-dot-product attention(SDPA)의 학습 길이 의존성에 막힌다는 지적이 이어졌다. Sliding Window Attention(SWA)은 국소 표현은 안정적이지만 창 밖 장거리 의존을 놓치고, full-context(전역) softmax는 긴 문맥에서 확률 분산과 활성 드리프트로 perplexity가 급격히 무너질 수 있다. 또한 Recurrent/Linear attention 계열은 장기 흐름은 잡지만 “needle-in-a-haystack”처럼 희소·정밀한 회수 정확도가 제한된다.

- **Core Contribution**: ATMA는 장거리 문맥에서 발생하는 “perplexity 붕괴 vs 회수 정확도”의 구조적 긴장을 동시에 겨냥한 하이브리드 시퀀스 믹서다. Polar Attention 코어는 attention 혼합을 3채널(방향·크기·메모리)로 분해해 softmax의 길이 민감 분산 효과를 줄이고, Titans 계열 gated-delta recurrent compression memory를 결합해 장기 재탐색을 보강한다. 단독 Polar 또는 단독 메모리는 모두 부족하지만, 둘의 조합은 단조로운 perplexity 감소와 고정밀 장거리 retrieval을 함께 달성한다고 주장한다.

- **Technical Challenges**: 핵심 난제는 (1) 긴 문맥에서 softmax가 만드는 확률 희석을 억제하면서 (2) 메모리 readout이 길이 변화에도 같은 분포 영역에 머물게 하는 것이다. ATMA는 length-aware temperature와 extreme-value-corrected null sink로 로그릿 잡음이 신호를 압도하는 상황을 완화하고, 참여도 지표(participation ratio)를 기반으로 bounded magnitude 채널을 설계해 진폭이 무한히 커지지 않게 한다. 더불어 delta memory의 수치 폭주를 막기 위해 key/query L2-normalization으로 안정성을 확보하고, 매 추론 단계에서는 in-place gated-delta 업데이트를 fused 커널로 구현해 gather-scatter 병목을 제거했다.

- **Empirical Impact**: FineWeb-Edu에서 1B 토큰 학습을 같은 조건으로 수행한 100-run factorial ablation sweep 결과, ATMA의 induction needle-in-a-haystack retrieval 정확도는 학습 길이 2K를 32배 확장한 64K 토큰까지 90% 이상으로 유지됐다. 문서 perplexity도 2.70 nats에서 1.96 nats로 단조 감소하며, softmax 기반 메모리 베이스라인은 극단적 문맥에서 collapse하는 반면 ATMA는 비교 열세를 극복했다고 보고한다. 구현 측면에서도 NVIDIA L4에서 메모리 분기 최적화가 MFU 손실을 4.5 percentage-point로 관리하고, decode 처리량은 19,270 tokens/second(B=512)를 달성해 실사용 효율성을 뒷받침한다.



### Hitting a Moving Target: Test-Time Adaptation for AI Text Detection under Continual Distribution Shif (https://arxiv.org/abs/2606.25152)
- **Prior Approaches**: 기존 AI 텍스트 탐지는 학습 단계에서 사람/생성 텍스트 라벨이 모두 있는 지도학습에 크게 의존한다. 그러나 배포 후에는 adversarial humanization, 신규 LLM 등장, 인간 문체의 temporal drift 같은 지속적 분포 이동이 생기는데, 이때 라벨이 없거나 확보 비용이 커서 표준 재학습/도메인 적응이 잘 먹히지 않는다. 또한 많은 방법이 테스트 샘플을 독립적으로 보고 test-time homogeneity 같은 현실 신호를 거의 활용하지 못한다.

- **Core Contribution**: 이 논문은 “테스트 시점 적응(test-time adaptation, TTA)”을 AI 텍스트 탐지에 적용하는 프레임워크를 제안한다. 핵심 아이디어는 추론 시점에 관측되는 무라벨 샘플들의 동질성(homogeneity)을 semi-supervised learning으로 활용해, 라벨 없는 분포 이동에도 탐지기를 적응시키는 것이다. 특히 positive-unlabeled(또는 positive-negative-unlabeled) 학습을 TTA에 결합해, shifted AI 텍스트와 인간 텍스트를 구분하는 방향으로 분류 경계를 조정한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “지속적”이고 “라벨이 부족한” 분포 이동 속에서, 고정된 분류 경계를 인간화 도구가 계속 공략하는 adversarial 환경을 어떻게 따라잡을지다. 논문은 이 문제를 전략적 분류/performative prediction 관점에서 정리하고, 무라벨 테스트 배치에서 PU/PNU 학습의 최적화가 shifted 분포에서도 사람(또는 LLM 생성물) 쪽을 일관되게 분리하도록 설계될 수 있음을 이론적으로 뒷받침한다. 실험적으로는 (TED)n 학습을 PU + TTA, PNU + TTA로 변형해, 테스트 분포의 무라벨 데이터를 사용하면서도 라벨 없이 적응하도록 구현했다.

- **Empirical Impact**: 실험에서 supervised 탐지기는 adversarial 및 자연 분포 이동(신규 LLM, 인간 문체 변화) 모두에서 체계적으로 성능이 무너졌다. 반면 PU + TTA/PNU + TTA는 상당히 견고해, 예를 들어 Pangram은 adversarial AI-generated 텍스트에 대해 24.1%만 탐지한 반면 제안된 테스트 시점 적응은 90.5%를 탐지했다. 또한 RAID 및 arXiv(추상) 설정에서 유사한 경향이 재현되며, 저자들은 야생(wild) 환경의 AI 텍스트 탐지에서 TTA가 유망한 표준 프레임워크가 될 수 있음을 시사한다.



### Silent Failures in Physics-Informed Neural Networks: Parameter Poisoning and the Limits of Loss-Based Validation (https://arxiv.org/abs/2606.25151)
Comments:
          7 pages, 2 figures

- **Prior Approaches**: PINN(Physics-informed neural networks)은 PDE를 손실함수에 직접 포함해 메쉬 없이 해를 학습하며, 일반적으로 낮은 PDE residual(잔차) 손실을 물리적 정합성의 증거로 간주해 왔다. 하지만 기존 연구들은 주로 경계·초기조건 데이터 오류나 입력공간의 adversarial perturbation에 대한 강건성을 다뤘고, 물리식 자체(계수·매개변수)가 틀린 경우의 ‘무증상 실패’를 체계적으로 규명하지 못했다. 또한 PDE 파라미터를 직접 복원하는 inverse PINN류는 과대적합/가중치 흡수 문제로 인해 안정적 진단이 어려운 한계가 있었다.

- **Core Contribution**: 이 논문은 PINN에서 PDE 파라미터가 잘못 인코딩될 때(physics parameter poisoning, parameter misspecification) 학습 진단은 정상처럼 보이지만 해는 틀릴 수 있음을 “silent failure”로 정식화한다. 즉, 낮은 residual loss가 물리적 정답을 보장하지 않으며, 손실 기반 모니터링만으로는 정확성과 비정확성을 구분할 수 없다고 주장한다. 더불어 none of our claims requires an adversary처럼, 고의 공격이 없어도 동일한 실패 모드가 발생함을 실험적으로 보인다.

- **Technical Challenges**: 핵심 기술적 어려움은 ‘잘못된 물리식’이 들어간 상태에서 PINN이 왜/어떻게 낮은 손실로 수렴하는지, 그리고 이를 손실만으로는 어떻게 탐지할 수 없는지 확인하는 데 있다. 저자들은 PDE 파라미터를 훈련 전 교란한 뒤(고정·양방향 스윕 포함) 낮은 잔차 손실에도 해 오차가 크게 남는 현상을 측정하고, 이를 파라미터별 민감도 관점으로 분석했다. 또한 사후(post-hoc)로 모델을 재학습하지 않고도 PDE residual loss를 다양한 파라미터 값에서 재평가해 손실 최소점의 위치로 “훈련 파라미터”를 복원하는 loss landscape sweep을 제안한다.

- **Empirical Impact**: Burgers equation, Navier-Stokes lid-driven cavity, convection-diffusion 3개 PDE에서 poisoned 모델이 clean 모델과 동급 또는 더 낮은 훈련 손실을 달성하면서도 해는 최대 71%(고정 스윕)·128%(적대적 탐색)까지 크게 달라졌다. 특히 Navier-Stokes cavity(Re=400)에서는 poisoned loss가 clean baseline 아래로 내려가도 솔루션 오차는 여전히 큼을 보여 “낮은 손실=정답” 가정이 붕괴함을 실증했다. 6가지 후보 방어는 전 구간에서 일관된 탐지를 못 했고, 반면 loss landscape sweep은 3개 PDE 모두에서 5개 랜덤 seed 및 8.7K~133K 파라미터 규모의 다양한 네트워크에 걸쳐 true training parameter를 안정적으로 찾아냈다는 점에서 실무적 파급력이 크다.



### Proactive Systems in HCI and AI: Concepts, Challenges, and Opportunities (https://arxiv.org/abs/2606.25149)
- **Prior Approaches**: 최근 자율·선제 시스템에 대한 관심이 커졌지만, proactivity(선제성)는 연구·실무에서 느슨하게 정의되고 서로 다른 행동을 한데 묶는 경향이 있습니다. 리마인더나 추천처럼 단순 프롬프트형 기능도 ‘proactive’로 불리곤 해, 실제로 사용자의 목표를 예측하고 주도적으로 행동하는 시스템과의 차이가 흐려집니다. 또한 평가지표와 설계 방법이 주로 reactive 상호작용에 뿌리를 두어 타이밍, 적절성, 사용자 통제·동의, 투명성, 신뢰 같은 핵심 변수를 충분히 반영하지 못합니다.

- **Core Contribution**: 이 워크숍은 선제성의 공통된 개념 틀을 세우고, autonomy(자율), automation(자동화), system-initiated interaction(시스템 주도 상호작용)과의 구분을 명확히 하려는 목적을 갖습니다. 더 나아가 시나리오 기반 활동으로 선제 시스템의 설계 공간과 차원(타이밍·투명성·통제·적절성)을 정리하고, 사람 중심 가이드라인과 향후 연구 방향을 공동으로 도출하는 데 초점을 둡니다. 결과적으로 선제 기술을 체계적으로 설계·비교·평가할 수 있는 프레임워크 기반을 마련하려고 합니다.

- **Technical Challenges**: 선제성 구현의 난제는 ‘언제’ 개입할지와 ‘얼마나’ 개입할지, 그리고 사용자의 목표에 얼마나 부합하는지(적절성)를 모델이 일관되게 다루는 데 있습니다. 또 사용자가 원치 않는 행동을 ‘통제·동의’ 없이 위임받는 상황을 줄이기 위해, 투명한 근거 제시와 예측 가능성, 사용자가 주도권을 조절할 수 있는 설계가 필요합니다. 워크숍은 개념 정립 후 시나리오 설계·분석을 통해 각 차원별 고려사항을 뽑고, trust(신뢰), usefulness(유용성), user comfort(사용자 편안함) 같은 평가 접근을 함께 연결하는 방식으로 이 문제를 다룹니다.

- **Empirical Impact**: 기존의 SUS나 UEQ 같은 반응형 문항은 선제 상호작용의 미묘함을 반영하기 어렵다는 문제의식이 있으며, 신뢰·예측가능성·정당화 가능성 등 선제형 평가에 맞춘 프레임 전환이 필요하다는 점을 강조합니다. 이번 워크숍은 공유 정의와 설계·평가 차원 정리에 더해, 개방형 기록 플랫폼과 향후 커뮤니티(Discord) 운영으로 논의의 지속성과 확산을 노립니다. 따라서 선제 시스템 연구가 ‘용어 혼용’과 ‘반응형 평가의 한계’에서 벗어나 더 견고하고 일관된 실증 비교로 이어질 가능성이 큽니다.



### TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems (https://arxiv.org/abs/2606.25147)
- **Prior Approaches**: 기존 산업용 추천은 LEM(대형 임베딩 모델)처럼 사용자/아이디를 고정 차원의 dense embedding으로 압축해 왔다. 하지만 이런 방식은 표현 제약이 커서 일반화에 한계가 있고, LLM 기반 텍스트 프로필은 대개 비정형 ID 공간에서 modality gap과 함께 ‘순차 행동 역학’보다는 ‘토픽 공기(共起)’를 학습하는 문제가 보고돼 왔다.
또한 PLUM은 item retrieval엔 효과적이었지만, sequential user modeling(SUM)으로 확장해 SID 기반 이산 표현을 어떻게 사용자에게 적용할지 자체가 충분히 탐색되지 않았다.

- **Core Contribution**: TokenMinds는 PLUM(semantic ID, SID) 패러다임을 사용자 모델링으로 확장해, 한 번의 encoder-decoder에서 dense user embedding과 SID 기반 이산 user token을 동시에 생성한다. 이 dual-output 설계로 ‘의미적으로 grounded된 토큰 표현’과 ‘기존 downstream이 기대하는 dense 벡터 호환성’을 함께 확보하는 것이 핵심이다.
또한 동일 SID 어휘를 공유해 long-form 비디오(LFV)와 short-form 비디오(SFV)를 하나의 모델로 통합하고, multi-context decoding으로 시나리오별 토큰을 효율적으로 뽑아내며 학습/서빙 비용을 절감한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 LLM이 대규모 비텍스트 ID 공간에 대해 학습된 지식과 SID 모달리티를 잘 정렬하는 동시에, 사용자 시퀀스에서 의미 있는 SID 토큰을 안정적으로 생성하는 것이다. TokenMinds는 PLUM의 Continued Pre-Training(CPT)로 SID 모달리티 정렬을 먼저 수행하고, SFT에서 prefix-L SID(계층형 코드워드의 상위 접두부)만 loss에 사용하도록 설계해 복잡한 grounding 부담을 줄인다.
서빙 레이턴시 제약을 해결하기 위해서는 UBS(User Behavior Service) 기반 비동기 인프라로 표현 생성과 실시간 scoring을 분리하고, 캐시된 표현을 scoring 시점에 재사용한다.

- **Empirical Impact**: YouTube LFV·SFV 및 검색 쿼리를 포함한 대규모 학습 데이터로 offline 실험과, 실제 프로덕션 트래픽에서의 live A/B 테스트를 모두 수행해 토큰 기반 사용자 표현의 산업적 실행 가능성을 입증했다. 특히 dense embedding과 SID token은 보완적으로 작동해 ranking 통합 시 유의미한 개선(예: 코어 지표 +0.11%, 참여 지표 +0.62% 수준)을 보고했다.
또한 LFV와 SFV를 단일 프레임으로 통합해 학습/서빙 비용을 줄이면서도 핵심 품질과 fresh 콘텐츠 지표를 함께 유지·개선하며, ‘시나리오 통합’의 실용적 해법을 제시했다.



### Benchmarking the Alignment of Data-Quality Metrics, Human Judgment and Land-Cover Segmentation Performance for Earth Observation (https://arxiv.org/abs/2606.25128)
- **Prior Approaches**: 합성 데이터 품질 평가는 주로 FID, KID, Inception Score(IS), LPIPS, SSIM 같은 자동 지표로 신뢰도를 판단해 왔다. 특히 FID는 사전학습(대개 ImageNet) 특징 공간에서의 분포 거리로 계산되지만, 시각적 충실도는 잘 반영해도 실제 downstream 유용성과는 쉽게 어긋날 수 있다는 한계가 지적돼 왔다. 또한 도메인 특화 입력(지구관측)의 경우 ImageNet 표현이 그대로 전이되지 않아 평가지표 편향 가능성이 크다.

- **Core Contribution**: 이 논문은 지구관측(Earth observation)에서 자동 품질 지표, 인간 인식, 그리고 랜드커버 의미 분할 성능이 어떻게 정렬/불정렬되는지 하나의 실험 프레임으로 체계적으로 비교한다. 회전·잡음·원근 왜곡 등 의미 보존/의미 변화 변환을 걸어 지표의 취약성을 보여주고, 동시에 4단계 인간 설문과 분할 모델 벤치마크로 작업 유용성을 확인한다. 그 결과 “자동 지표 점수=실제 효용”이라는 통념이 성립하지 않는 패턴을 일관되게 제시한다.

- **Technical Challenges**: 핵심 기술 난점은 ‘시각적 품질(또는 분포 유사성)’을 측정하는 자동 지표가 의미적으로 중요한 변화에 민감하지 않거나(혹은 과도하게 민감), 반대로 인간이 실제로 평가하는 현실감과는 다른 신호를 추적할 수 있다는 점이다. 논문은 여러 유형의 perturbation에 대해 7개 자동 지표의 반응을 측정하고, 인간 인식(장면 동일성, 이미지-마스크 정합, 생성 선호, 현실감 점수)과 분할 성능(F1)까지 같은 데이터셋에서 비교해 이러한 불일치를 정량화했다. 특히 ImageNet 기반 특징 공간의 방향 민감성 같은 이유로 rotation에서 FID가 크게 흔들리지만 인간 인식 정확도는 거의 유지되는 현상을 관찰한다.

- **Empirical Impact**: 실험은 자동 지표가 세부적인 의미 보존 변환에도 급격히 점수가 변하는 반면, 인간 인식은 덜 민감하다는 ‘정합 실패’를 보여준다. 더 나아가 FID 같은 낮은-거리 점수는 분할 F1과 잘 맞지 않으며, 심지어 자동 지표가 나쁘게 나오는 합성 데이터도 실제 데이터와 혼합하면 분할 성능을 오히려 개선할 수 있음을 입증한다. 결론적으로 지구관측 합성 데이터 품질 평가는 downstream task 성능과 인간 평가를 기반으로 설계돼야 하며, 단일 자동 지표 최적화는 유용성 손실을 유발할 위험이 크다는 실무적 함의를 제공한다.



### Reward-Conditioned Attention: How Reward Design Shapes What Autonomous Driving Agents S (https://arxiv.org/abs/2606.25127)
- **Prior Approaches**: 자율주행 RL 연구는 정책 성능(예: closed-loop 시뮬레이션)과 함께 Transformer/Perceiver 기반 인코더 구조를 활용해왔지만, 학습된 내부 표현이 어떤 보상 설계에서 비롯되는지는 충분히 해명되지 않았다. 또한 attention은 설명력이 약할 수 있다는 XAI 논쟁이 있어, 기존 접근은 대체로 attention을 인과 해석의 근거로 쓰기보다 다른 방법(교란/귀인 등)에 의존해왔다. 보상 shaping이 행동을 바꾸는 점은 알려져 있으나, 보상이 attention 분배 같은 표현 자체를 어떻게 바꾸는지의 정량 근거는 부족했다.

- **Core Contribution**: 이 논문은 reward design이 reinforcement learning agent의 Perceiver cross-attention 패턴을 “예측 가능하게” 조형한다는 질문을 다룬다. 특히 여러 episode에 걸친 단순 pooling이 attention-위험 관계를 과소평가할 수 있음을 보이고, episode 내부 상관을 Fisher z-transform으로 집계하는 방법론을 제안한다. 이를 통해 collision risk와 agent-directed attention 사이에 강한 양(+)의 연결이 있음을 검증하고, 보상 구성(네비게이션/안전 근접/TTC 패널티)이 장면 토큰 우선순위와 휴지기 감시 상태를 직접 바꾼다는 두 가지 핵심 효과를 제시한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 heterogeneous한 시나리오와 episode 길이/위험 변동이 섞일 때, “진짜로 위험이 바뀔 때 attention이 움직이는지”를 분리해 측정하는 것이다. 저자들은 시나리오별로 collision risk ℛ과 agent-directed attention 간 Spearman 상관을 구한 뒤 Fisher z-transform 집계를 적용해 within-episode 상관을 안정적으로 추정했으며, naive pooling에서 생기는 3.3배 과소추정을 실증했다. 추가로 agent 수(주변 차량/보행자 수)가 위험 신호와 attention을 함께 끌어올리는 기계적 confound 가능성도 partial correlation으로 점검해 영향이 작음을 확인했다.

- **Empirical Impact**: Waymo Open Motion Dataset(WOMD) 50개 실주행 시나리오에서, navigation reward를 포함한 모델은 GPS-path 토큰에 최대 2.0× 더 많은 attention을 배치했으며, TTC 기반 연속 proximity penalties는 collision-free 구간에서도 resting agent surveillance를 유지하는 learned vigilance prior를 만들었다. 또한 일부 시나리오에서는 reward 구성이 attention-위험 상관의 부호를 반대로 뒤집어(attention reversal) 단순 크기 조절을 넘어 전략 자체가 달라질 수 있음을 보여준다. 저자들은 attention 분석을 safety-critical RL에서 보상 함수가 의도한 representational behavior를 학습했는지 확인하는 진단 도구로 활용할 수 있다고 정리한다.



### AeroCast: Probabilistic 3D Trajectory Prediction for Non-Cooperative Aerial Obstacles via Transformer-MDN Architectur (https://arxiv.org/abs/2606.25122)
- **Prior Approaches**: 기존 접근은 칼만 필터와 IMM 같은 물리 기반 추정으로 비협조 장애물의 상태를 갱신하지만, 미리 가정한 운동 모델(정속·coordination turn)이 실제 변칙 동작을 잘 못 따라가 예측이 뒤처질 수 있다. 학습 기반 recurrent 모델은 온보드에 빠르지만 대개 점(point) 예측만 제공해, 불확실성이나 행동 분기 지점을 하류 플래너가 구분하기 어렵다.

- **Core Contribution**: 본 논문은 비협조 항공 장애물의 미래 위치를 확률적으로 예측하는 AeroCast를 제안한다. Pre-LN Transformer 인코더에 Mixture Density Network(MDN) 출력 헤드를 결합해, 각 타임스텝의 3D 변위(displacement)에 대해 Gaussian mixture 분포를 직접 출력한다.

- **Technical Challenges**: 핵심 난관은 (1) 비협조 장애물의 다중 모드 동작을 mixture로 표현하되, (2) NLL만 최적화하면 mixture 모드가 기하학적으로 붕괴(mode degeneracy)하는 문제, (3) 입력 표현이 translation에 민감해 분포가 불안정해지는 문제다. AeroCast는 연속 변위 기반 translation-invariant 인코딩과 변위 스케일 정규화, mode-anchoring MSE 항 및 센서 노이즈에 맞춘 sigma floor를 함께 사용해 모드 붕괴와 불리정 정규화를 동시에 완화한다.

- **Empirical Impact**: Vicon 기반 실데이터와 합성 데이터를 섞은 쿼드로터 코퍼스(9개 모션 카테고리)에서 AeroCast는 5초 예측 구간에서 기존 베이스라인 대비 ADE/FDE를 약 50% 수준으로 낮추고, NLL과 Continuous Ranked Probability Score(CRPS)에서도 최고 성능을 보였다. 또한 ablation 결과 velocity 입력과 모델 capacity가 예측 품질에 가장 큰 영향을 주며, positional encoding은 장기 일관성에 필수로 나타났고, 추론은 샘플당 0.1ms로 실시간(100Hz) 탑재 가능성을 보여준다.



### BCoughBench: Benchmarking Respiratory Acoustic Foundation Models Under Body-Coupled Wearable Sensor Conditions (https://arxiv.org/abs/2606.25116)
Comments:
          Accepted to the KDD 2026 Workshop on Reliable Scientific Foundation Models (RelSciFM)

- **Prior Approaches**: 호흡기 acoustic foundation model(FM)들은 주로 스마트폰 음성(공기전도) 녹음에서 벤치마킹됐고, 선형 프로빙 기반으로 AUROC 위주 성능을 공개해 왔다. 이 방식은 실제 스크리닝이 요구하는 민감도(operating point)나 보정(calibration) 실패를 숨길 수 있다. 또한 body-coupled(BC) 웨어러블 환경에서의 주파수 감쇠가 FM 임베딩에 미치는 영향은 기존 벤치마크에서 거의 다뤄지지 않았다.

- **Core Contribution**: 본 논문은 BC 웨어러블용 호흡기 FM 성능을 정량화하기 위한 BCoughBench를 제안한다. EBEN(Extreme Bandwidth Extension Network) 역모델로 5종 BC 센서 조건을 시뮬레이션하고, 5개 FM(OPERA-CT/CE/GT, HeAR, M2D+Resp)을 9개 분류와 3개 연령 회귀 태스크에 대해 zero-shot으로 평가한다. AUROC뿐 아니라 Se@Sp95(임상 민감도)와 Expected Calibration Error(ECE)를 함께 보고, 운영 관점의 실패 여부를 드러내는 것이 핵심이다.

- **Technical Challenges**: BC 센서는 조직/뼈 등을 거치며 고주파 콘텐츠를 선택적으로 감쇠해, 스마트폰 기반 임베딩이 의존하는 판별 단서가 깨질 수 있다. 논문은 실제 하드웨어 없이도 EBEN AC→BC 시뮬레이션 파이프라인을 사용해 센서 배치별 대역폭(예: temple ≤2kHz, throat ≤1.5kHz)을 반영한 입력을 만들고, 센서 특화 적응 없이 선형 프로빙/회귀 헤드만 학습해 공정한 비교를 수행한다. 동시에 분류는 AUROC·Se@Sp95·ECE, 회귀는 MAE와 mean-predictor 기준선을 함께 사용해 성능이 “겉보기로만” 좋아 보이지 않도록 했다.

- **Empirical Impact**: 결과적으로 평균 AUROC는 스마트폰 0.785에서 BC 조건 0.689–0.723으로 하락했으며, temple 픽업에서 열화가 가장 컸다(Δ=-0.096). 하지만 더 심각한 문제는 임상 민감도이며, 대부분 질병 태스크에서 어떤 FM도 Se@Sp95≥0.20을 달성하지 못했다. 태스크별로는 CIDRZ에서 성별 분류가 AUROC 0.954→0.596–0.628로 붕괴(Δ=-0.341)한 반면, COVID 탐지는 AUROC 열화가 거의 없었고(Δ=-0.004) 연령 회귀는 forehead accelerometer에서 오히려 MAE가 개선(예: CoughVID 9.61→8.97년)됐다. 저자들은 AUROC 단독 보고의 한계를 “운영 실패” 관점에서 확인했으며, 센서 선택이 모델 선택 못지않게 중요하다는 점을 BCoughBench로 입증했다.



### Power-Flexible AI Data Centers: A New Paradigm for Grid-Responsive Compu (https://arxiv.org/abs/2606.25098)
Comments:
          14 pages, 7 figures, 1 table

- **Prior Approaches**: 기존 수요반응(demand response) 연구는 지연 가능한 작업의 시간 이동, 지역 간 작업 마이그레이션, 탄소집약도 기반 스케줄링을 중심으로 유연성을 탐색해왔다. 다만 CPU 중심이거나 피크 감축에 한정돼, 실시간 GPU 기반 grid-interactive 운용, 장시간 지속 감축, 우선 작업 서비스 보장까지 폭넓게 다룬 사례는 드물었다. 현장 데모도 대체로 단일 모드나 제한된 조건에서 검증되는 경향이 있었다.

- **Core Contribution**: 이 논문은 GPU AI 데이터센터를 전력시스템의 grid-interactive 자산으로 만들기 위한 소프트웨어 아키텍처를 제안한다. grid 신호(디스패치)→워크로드 오케스트레이션→GPU/랙 텔레메트리 기반 전력 예측의 통합 제어로, 서비스 수준을 유지하면서 전력 감축·탄소 대응·지리 분산 로드 이동을 동시에 수행한다. 특히 Conductor로 그리드 이벤트를 클러스터 수준 제어(예: GPU power capping, 잡 스케줄 조정)로 번역하는 구조를 구체화했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 실시간 디스패치 목표를 시간 제약 안에 맞추는 정밀 제어, (2) 서로 다른 민감도의 학습/추론 작업을 우선순위별로 분류해 성능 저하를 최소화하는 스케줄링, (3) 전력 예측을 위해 GPU·랙 단위 측정과 추가 설비(CPU/네트워크/스토리지)까지 아우르는 모델링이다. 논문은 초 단위 GPU 전력 텔레메트리와 랙 측정 데이터를 결합해 전력 모델을 학습하고, job power signature 라이브러리로 제어 전략의 영향을 빠르게 추정해 제약을 만족하도록 수렴시킨다고 설명한다. 또한 SLURM 우선순위 메타데이터를 활용해 유연 작업을 선택적으로 늦추거나 캡핑해 우선 작업의 처리량을 유지한다.

- **Empirical Impact**: 130kW 규모(Blackwell Ultra GPU 96대) 실제 배치에서 200개+ 디스패치 타깃을 포함한 테스트로, 목표 전력과 램프 제약을 모두 충족하는 정밀 제어 능력을 보여줬다. 응답 속도는 비상 이벤트에서 약 40초 내 30% 감축, 특정 실험에서는 1분 내 40% 감축 등으로 나타났고, 2~10시간 지속 감축에서도 우선 작업 처리량을 거의 기준선 수준으로 유지했다. 또한 영국에서는 National Grid 이벤트 100% 컴플라이언스를, 미국에서는 버지니아( Ashburn )에서 일리노이( Chicago )로 라이브 추론 10%를 지리 이동시켜 사용자 지표(TTFT)도 제한적으로 변화함을 입증했다. 더 나아가 5분 단위 탄소집약도 신호를 따라 전력 사용을 조절해 탄소 인지 운영 가능성까지 확인하며, AI 인프라를 정적인 전력 소비자에서 신뢰성 보조 자원으로 전환하는 의미를 갖는다.



### Training for the Model You Return: Improving Optimization for Iterate-Averaged Language Models (https://arxiv.org/abs/2606.25086)
- **Prior Approaches**: 기존 LM 학습 파이프라인은 최종 파라미터(iterate) 대신 지수이동평균(EMA) 같은 averaged model을 반환하는 경우가 많지만, 이때 “평균을 반환한다면 학습을 어떻게 바꿔야 성능이 오르나”는 명확히 다뤄지지 않았다. 실무에서는 AdamW에 EMA를 얹고(또는 EMA를 평가에만 사용) 학습률/스케줄 튜닝으로 성능을 맞추는 방식이 일반적이었지만, 평균 추정기의 관점에서의 옵티마이저 설계는 부족했다.

- **Core Contribution**: 이 논문은 반환되는 iterate-average를 목표로 하는 옵티마이저 설계를, optimal-control 문제로 정식화해 평균 오차를 직접 줄이는 훈련 전략을 제안한다. 이를 바탕으로 AdamW에 가볍게 감싸는 래퍼 형태의 PACE를 제시하며, 라이브 가중치를 EMA 쪽으로 끌어당기되 좌표별 제어 강도를 클리핑해 과도한 개입을 제한한다.

- **Technical Challenges**: 핵심 난제는 “평균을 평가/반환할 때” 최적 제어가 어떤 형태여야 하는지, 그리고 이를 실제 AdamW 훈련에 적용 가능한 근사로 어떻게 바꿀지였다. 논문은 연속시간 확률적 이차(quadratic) 모델에서 평균 반환 오차를 최소화하는 제어를 도출하고, 이를 실용적으로 근사한 PACE가 (조건부로) 표준 확률적 볼록 최적화 수렴률로 수렴함을 보이며, 이차 설정에서는 iterate-average의 한계 제곱 오차를 엄밀히 개선하고 경우에 따라 임의로 큰 수준의 개선도 가능함을 보인다.

- **Empirical Impact**: 실험적으로 PACE는 supervised fine-tuning에서 1–2B 파라미터급 LMs와 GPT-2 pretraining에서 AdamW 및 EMA-evaluated AdamW 대비 성능을 개선하는 경향을 보였고, 다양한 learning rate·decay schedule·하이퍼파라미터 범위에서 강건함을 시사한다. 결과는 EMA처럼 “평균된 모델을 반환”하는 실제 파이프라인의 설계 철학을, 평균 추정기 관점의 옵티마이저로 확장할 수 있음을 보여준다는 점에서 의미가 크다.



### GCT-MARL: Graph-Based Contrastive Transfer for Sample-Efficient Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.25073)
Comments:
          Accepted at The Continual RL Workshop, RLC 2026

- **Prior Approaches**: 기존 협력적 MARL 전이 연구는 lateral connection 기반(MALT), curriculum 기반, attention/transformer 기반 풀링 등으로 가변 에이전트 수를 다루려 했지만, 대체로 태스크 간 표현 정렬이 없어 negative transfer 위험이 남아 있습니다. 또한 MAIL 같은 graph contrastive 기반 방법은 단일 태스크에서 강하지만, population mismatch(에이전트 수/구성 차이) 전이를 위한 설계로는 충분히 다뤄지지 않았습니다.

- **Core Contribution**: GCT-MARL은 MAIL의 multi-view graph contrastive backbone을 전이용 전처리/학습 구조로 재구성해, 서로 다른 population 크기와 구성 간 학습 효율을 높이는 transfer learning 프레임워크를 제안합니다. per-view adaptively weighted alignment loss로 source와 target의 표현 공간을 명시적으로 맞추고, 이를 두 단계(훈련-전이) 프로토콜로 연결해 continual learning까지 자연스럽게 확장합니다.

- **Technical Challenges**: 핵심 어려움은 에이전트 수와 관측/상태 구성 차이로 인해 기존 아키텍처의 파라미터 또는 표현 의미가 고정되지 않는다는 점입니다. GCT-MARL은 observation을 엔터티 타입별로 분해해 per-entity encoder로 관측 차원에 덜 민감한 구조적 전이 가능성을 만들고, 이후 frozen source와 target의 multi-view embedding을 InfoNCE 형태로 per-view 정렬하되 view별 가중치를 learnable로 자동 조정해 의미 갭(semantic gap)을 줄입니다.

- **Empirical Impact**: 실험에서 GCT-MARL은 from-scratch 대비 타깃 태스크에서 수렴을 뚜렷하게 가속하며, 동일 faction 내 인원 수가 달라지는 동질 전이와 faction 간/유닛 타입 혼합 같은 이질 전이 모두에서 효과가 확인됐습니다. continual 시퀀스(SMAC 4-phase)에서는 최종 평균 accuracy 89.8%와 평균 backward transfer -0.125를 보고해, 전용 anti-forgetting 정규화 없이도 이전 지식이 잘 유지됨을 시사합니다.



### Adapt Only When It Pays: Budgeted Decision-Loss Priority for Delayed Online Time-Series Adaptation (https://arxiv.org/abs/2606.25068)
- **Prior Approaches**: 지연 레이블이 있는 온라인 시계열 예측에서 기존 적응은 “매 타이밍마다 업데이트”하거나, “예측 품질(정확도/캘리브레이션)”을 중심으로 모델을 조정하는 방식이 주류였다. 하지만 라벨이 horizon에 따라 늦게 공개되고 업데이트 예산이 제한되는 상황에서는, 어떤 피드백을 선택적으로 업데이트할지(업데이트 할당)와 그 선택이 downstream 의사결정 손실에 미치는 영향이 충분히 감사(auditable) 가능하게 다뤄지지 않았다.

- **Core Contribution**: ADOWIP은 residual-adapter 프레임워크 위에, 지연된 label release를 sealed delay queue로 봉인하고 업데이트 예산을 정확히 계측하며 업데이트 텔레메트리를 감사 가능하게 남기는 온라인 적응 프로토콜을 제안한다. 핵심은 observed decision-loss priority gate로, 라벨이 공개된 뒤(피드백 공개 이후) 현재 decision loss가 높고(선택 기준: 캘리브레이션된 경험적 quantile 초과), 동시에 남은 예산이 허용될 때만 업데이트를 승인하는 스케줄링에 있다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 지연된 피드백 환경에서 label 누수 없이 스케줄러를 관측값 기반으로 설계하고, (2) “업데이트를 언제 할지”의 하드 budget 조건을 수학적으로 보장하며, (3) 비선형 residual adapter 같은 실제 설정에서의 안전성/범위 경계를 정리하는 것이다. 논문은 hard-budget feasibility(예산 초과 방지), convex 선형 accepted-update 부분문제에 대한 projected-OGD regret, 그리고 게이트 선택에서 관측 점수 오차가 경계 밴드 내에서만 의사결정이 달라진다는 안정성/조건부 유한표본 결과를 제시하고, probe 기반 반사실 방법은 probe backward pass를 별도로 계측해 실험적으로도 한계를 드러낸다.

- **Empirical Impact**: public ETT capacity-planning 및 외부 UCI Bike, Capital Bikeshare(역할 보정/리밸런싱) 등에서 matched compute 조건의 decision loss 중심 비교를 수행했으며, primary로는 결정 손실(의사결정) 성능에서 이득이 관측된 스플릿 캘리브레이션/에블레이션 결과를 강조한다. 반면 ETT threshold/load-index 계열은 혼재(보수적 Holm 교정/가족 단위 검정에서 41개 선택 대비 33개만 통과, 나머지는 secondary로 제한)했고, finance·probe 기반 실험 및 일부 경계 진단은 음성으로 남아 decision-prioritized adaptation의 적용 범위를 현재는 “audited decision-loss-dominant 공용 프록시 태스크”로 한정한다.



### What Does It Mean to Break a Distillation Defense? (https://arxiv.org/abs/2606.25059)
Comments:
          29 pages, 18 figures

- **Prior Approaches**: 블랙박스 API로만 접근 가능한 black-box LLM은 distillation 공격에 취약하며, 최근에는 teacher 출력에 잡음을 주거나 추론 추적을 변형해 student 성능을 떨어뜨리는 output perturbation defense가 제안됐다. 다만 기존 연구는 시스템·탐지·출력교란 수준의 방어가 공통으로 쓸 위협 모델을 공유하지 않아, 서로 비교하거나 실제 적응형 공격에 대한 강건성을 논하기 어렵다는 문제가 있었다. 특히 다수 평가는 “각 프롬프트를 한 번만 질의하고, 추가 조작 없이 바로 student를 학습”하는 고정된 암묵적 설정에 머물러 있어 현실 공격을 과소평가할 여지가 컸다.

- **Core Contribution**: 이 논문은 distillation defense의 유효성을 ‘공격자가 얼마나 강한가’에 대해 명시적으로 정의하기 위한 위협 모델 프레임워크를 제안한다. 공격자는 query budget(Q), data budget(D), 그리고 인터페이스 프로파일(P)로 분해해 튜플 (Q,D,P)로 표현되며, P는 입력 채널·제공자(teacher 이후) 처리·노출되는 출력 채널의 조합으로 정리된다. 저자들은 antidistillation sampling(ADS)을 사례로, 위협 모델 가정에 따라 방어 효과 결론이 뒤집힐 수 있음을 보여주며, 향후 연구와 거버넌스도 이 3차원 능력을 스트레스 테스트로 드러내야 한다고 주장한다.

- **Technical Challenges**: 핵심 기술 난제는 ‘방어가 실패하는 조건’을 공평하게 비교하려면 공격자의 질의/데이터 자원과 API 상호작용 방식을 함께 모델링해야 한다는 점이다. 저자들은 Q와 D를 분리해 “모든 프롬프트를 한 번씩 질의”인지 “일부 프롬프트를 여러 번 재질의”인지 같은 전략을 구분하고, P를 통해 prefill, 제공자 측 필터링/요약, logprobs·tool call 노출 등 인터페이스 구성 변화가 방어를 무너뜨릴 수 있음을 구조적으로 포착한다. ADS 케이스 스터디에서는 인터페이스 구성 요소 중 prefill 접근 유무를 바꿔 같은 student 학습·평가 파이프라인 안에서도 결론이 달라짐을 실험으로 확인한다.

- **Empirical Impact**: 실험 결과, ADS의 원래 평가 설정조차도 현실적 공격자는 출력에 대해 반복 삭제 같은 ‘자유 로컬 후처리’를 수행할 수 있어, 이 한 단계만으로 student 정확도가 크게 회복되어 방어의 보고된 유틸리티 격차가 과대평가될 수 있음을 보여준다. 또한 query budget Q가 data budget D와 다를 때(예: Q≥2D) student 성능이 temperature sampling 수준까지 상당 부분 복원되어, 유효성의 정량 결론이 위협 모델 파라미터에 의해 최대 수십 %p까지 흔들린다는 점을 강조한다. 결론적으로 distillation defense의 “효과”는 방어 성능 고유값이 아니라 가정한 (Q,D,P)에 종속되며, 논문은 실제 배포·규제·IP 대응에서 허위 안심을 막기 위해 명시적 위협 모델 서술과 적응형 스트레스 테스트가 필수라고 제시한다.



### Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models (https://arxiv.org/abs/2606.25041)
Comments:
          Website: this https URL

- **Prior Approaches**: 기존 멀티모달 접근은 시각·오디오를 이해하는 모델과 비디오 생성/음성 합성 모듈을 따로 두거나, 중간 표현으로 text/ASR·TTS·VAD·아바타 렌더링을 조합하는 방식이 많았습니다. 이처럼 cascaded 파이프라인은 모듈 경계에서 대기 시간이 생기고, 인식·동기화 오차가 누적되며, 응답 타이밍·차례 관리·장기 일관성을 한 행동으로 학습하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: Wan-Streamer는 언어·오디오·비디오를 입력과 출력 모두에서 동일한 Transformer 한 개의 인과 스트림으로 묶어 end-to-end full-duplex 인터랙션을 지향합니다. 특히 외부 VAD/ASR/언어/TTS/오디오-드리븐 애니메이션/비디오 생성 모듈 없이, 지각·추론·생성·응답 타이밍·턴 테이킹·크로스모달 동기를 하나의 모델이 공동 학습하도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 과제는 스트리밍 중에 각 모달리티가 서로 다른 토큰 레이트와 표현을 가지더라도, 같은 인과 과정 안에서 끊김 없이 정렬된 결과를 내는 것입니다. 이를 위해 strictly causal audio/video VAE, causal encoders/decoders, block-causal attention, 그리고 160 ms 단위(25 fps) 수준까지 가능한 저지연 multimodal token scheduling을 도입했으며, flow matching 기반으로 오디오·비디오 latents를 joint로 생성해 디코딩 전에 결합합니다.

- **Empirical Impact**: 실험에서 모델-사이드 신호 지연은 약 200 ms, 원격 양방향 네트워크까지 포함한 총 인터랙션 지연은 약 550 ms로 sub-second 듀플렉스 통신을 보여줍니다. 또한 말하지 않는 구간에도 시선·자세·미세표정 등 ‘보이는 청취’ 상태를 유지하고, 발화 중에는 사용자의 끊김/개입에 맞춰 스피치를 조절하는 등 자연스러운 듀플렉스 상호작용 품질을 제시했습니다.



### LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search (https://arxiv.org/abs/2606.25039)
- **Prior Approaches**: 기존 ODE 방정식 복원은 주어진 데이터셋에서 식을 “정적 추론”처럼 찾아내는 방식이 주를 이루며, 관측이 충분히 정보적이라는 가정을 탑니다. 이 때문에 상태공간이 크거나 관측이 제한적이면 구조적으로 다른 후보들이 같은 데이터에서는 비슷한 오차를 내는 identifiability gap 문제가 생겨, 그럴듯하지만 잘못된 governing equation을 복원할 수 있습니다. LLM-유도 접근은 operator priors 같은 지식을 활용해 탐색을 돕지만 대개 데이터 수집과 결합된 closed-loop로 진짜 구분 가능한 관측을 모으는 데는 한계가 있습니다.

- **Core Contribution**: 논문은 LLM-ACES(LLM-guided Active Closed-loop Equation Search)라는 closed-loop 프레임워크로, symbolic hypothesis 구성과 adaptive data acquisition을 동시에 최적화합니다. LLM이 operator 수준의 priors로 큰 탐색공간을 분할하고, 그 안에서 후보 방정식을 학습한 뒤 후보들 간 예측 불일치가 큰 초기조건을 추가로 쿼리해 identifiability gap을 직접 해소합니다. 즉, 식 복원을 “피팅 결과”가 아니라 “가설이 다음 관측을 설계”하는 동적 과정으로 재정의합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 방대한 symbolic hypothesis 공간을 LLM이 전부 직접 내놓는 방식으로는 제어하기 어렵다는 점과 (2) 제한된 샘플 예산 안에서 어떤 초기조건이 경쟁 가설을 실제로 구분하는지 판단해야 한다는 점입니다. LLM-ACES는 operator prior로 상위 가설공간을 구조적으로 제약한 뒤, 후보들의 rollout을 비교하는 predictive-divergence(평균 pairwise predictive disagreement, NMSE 기반)를 acquisition score로 삼아 가장 구분력 있는 궤적을 선택합니다. 또한 각 라운드마다 검증 데이터로 후보를 재스코어하고 experience buffer에 고득점·저득점 가설을 함께 저장해, 처음 데이터에만 맞는 spurious term 유입을 억제하는 피드백을 구현합니다.

- **Empirical Impact**: ODEBench(63개)와 ODEBase(59개)에서 LLM-ACES는 전체적으로 NMSE와 symbolic accuracy가 동시에 가장 좋았고, 특히 NMSE 중앙값이 기존 SOTA 대비 여러 자릿수 이상 개선되는 결과를 보였습니다. symbolic accuracy는 ODEBench에서 46.2%(GPT-4o-mini)·52.4%(Qwen3-32B 계열 결과는 ODEBase에서 특히 높음) 수준을 달성하며, noise에도 견고하고 올바른 symbolic 구조를 복원하는 경향이 분석으로 확인됐습니다. 더 나아가 sample-efficient하게도 1/10 수준의 데이터로 더 나은 성능을 내며, “닫힌 루프” 기반 adaptive data acquisition이 정답 구조를 가리는 식의 국소 적합 문제를 완화한다는 점에서 의미가 큽니다.



### Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety (https://arxiv.org/abs/2606.25034)
- **Prior Approaches**: 기존 MLLM들은 전반적인 시각-언어 이해 능력을 잘 보이지만, 콘텐츠·AI 안전 영역에선 실사용의 적대적(adv.) 멀티모달 리스크를 안정적으로 찾아내기 어렵다. 특히 안전 정렬(alignment) 과정이 민감 지식의 탐색을 억제하면서, 왜 특정 시각 요소가 위반인지의 근거 있는 추론과 감사를 위한 설명이 빈약해진다는 한계가 지적된다.

- **Core Contribution**: Yuvion VL은 콘텐츠·AI 안전을 위한 대규모 멀티모달 LLM 계열로, instruction-tuned와 reasoning-oriented 변형을 제공하며 파이프라인 전체를 adversarial robustness 중심으로 설계한다. 안전을 “본질적으로 적대적이고 멀티모달인 문제”로 보고, 위험 개념의 크로스모달 정렬→안전 태스크 instruct 후학습→해석가능성 강화용 reasoning 후학습의 3단계 학습으로 구성한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 작은 텍스트·워터마크·심볼 변형처럼 사람은 알아보지만 모델은 놓치는 미세 시각 단서를 정확히 구분하는 것과 (2) 동일/유사 외형이지만 정책상 의미가 다른 경우를 판별하는 대조 신호를 만드는 것이다. 이를 위해 Confuse-then-Contrast Fine-Tuning(C2FT)을 도입해 모델별 혼동을 동적으로 채굴하고, 다중 이미지 contrastive 그룹으로 시각-의미의 fine-grained 차이를 명시적으로 강제하며, 추가로 추론용 데이터는 posterior-constrained CoT 생성과 품질 검증(정답 정합·형식·사실성)으로 정제한다.

- **Empirical Impact**: 평가를 위한 Yuvion VL RiskEval(YVRE)는 오픈/내부 벤치마크를 3단계로 묶어 콘텐츠·AI 안전, 적대적 견고성, 실사용 역량을 폭넓게 측정한다. 실험에서 Yuvion VL-32B는 안전 관련 성능에서 동급 오픈소스 대비 평균 9.9점, 더 큰 상용 클로즈드 대비 평균 6.7점을 앞섰고, Yuvion VL-8B는 GPT-5.4와 Qwen3.5-Plus 같은 대형 모델들보다 여러 안전 태스크에서 우수한 결과를 보이며 일반 역량 저하는 제한적임을 시사한다.



### Do Thinking Tokens Help with Safety? (https://arxiv.org/abs/2606.25013)
- **Prior Approaches**: 추론(reasoning) 모델은 thinking tokens를 사용해 벤치마크 성능을 높이는 경우가 많고, 이 방식이 더 ‘숙고적(deliberative)’이라 안전·정렬(alignment)에도 유리하다고 여겨져 왔다. 구체적으로는 모델이 요청에 대한 답을 내기 전에 안전 원칙 위반 여부를 고려하는 ‘안전한 공간’을 제공한다고 가정해 왔다.

- **Core Contribution**: 본 논문은 GPT-OSS, Qwen, Olmo, Phi 계열의 frontier open-weight reasoning 모델들을 대상으로, 최종 거절/순응(refusal/compliance) 결과가 첫 토큰의 hidden representation에서 이미 강하게 예측된다는 증거를 제시한다. 또한 thinking 텍스트가 있어도 실제로는 숙고적 수정이라기보다 prefix completion에 가깝고, 초반(전체 thinking의 약 20% 내) 이후 결과가 거의 바뀌지 않는다고 주장한다.

- **Technical Challenges**: 핵심 과제는 ‘보이는 숙고’와 ‘실제 의사결정 과정’을 분리해 측정하는 것이다. 논문은 첫 토큰 hidden representation에 대해 학습된 head로 거절/순응을 조기 예측하고, thinking 진행에 따른 outcome 변화 지점을 관찰하는 방식으로 deliberation의 실재성을 검증했으며, 기존 안전 개입들이 오히려 과도한 거절 쪽으로 행동을 치우치고 이미 희소한 deliberation 신호를 억제한다고 분석한다.

- **Empirical Impact**: 첫 토큰 기반 조기 예측은 AUROC 0.84~0.95, balanced accuracy 약 88%로 높은 성능을 보였고, 텍스트 수준에서 숙고처럼 보이는 현상(약 74%)이 있어도 분포가 한쪽으로 이미 잠긴 뒤 발생하는 비중이 큼을 보여준다. 더 나아가 기존 inference-time 및 training-based safety intervention이 ‘숙고 유도’를 목표로 했음에도 over-refusal을 강화하며 실제 deliberation을 줄일 수 있음을 실증적으로 제시해, 향후 ‘진짜 안전 숙고’를 유도하는 방법의 필요성을 강하게 시사한다.



### Noise-Aware Boundary-Enhanced Generative Learning for Ultrasound Speckle Reduction (https://arxiv.org/abs/2606.25009)
- **Prior Approaches**: 기존 초음파 speckle reduction(디스피클링) 방법은 필터 기반(bilateral, BM4D 등)과 딥러닝 기반으로 나뉜다. 필터 기반은 학습이 없어 빠르지만 고정된 규칙/가정 때문에 조직 경계를 과도하게 뭉개거나 이질적인 잡음 수준에 대한 적응이 약하다는 한계가 있다. 딥러닝 기반은 경계 보존을 위해 gradient/edge 제약을 넣기도 하지만, 잡음이 커지면 경계와 speckle의 고주파 성분이 구분되지 않아 구조 가이던스 신뢰도가 떨어지고, 잡음 레벨 적응이 있어도 경계 보존을 함께 강하게 결합하지 못하는 경우가 많다.

- **Core Contribution**: 이 논문은 Noise-Aware Boundary-Enhanced Generative Learning(NBGL) 프레임워크로, speckle 억제와 주석 기반 경계 보존을 하나의 3D 아키텍처에서 동시에 학습하도록 제안한다. Speckle reduction 가지는 생성학습으로 잡음을 줄이고, boundary enhancement 가지는 목표 해부학적 경계를 명시적으로 보강해 과스무딩을 완화한다. 또한 추정한 speckle noise level을 사용해 두 가지의 feature coupling 강도를 적응적으로 조절하는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 과제는 (1) speckle과 해부학적 경계가 모두 고주파로 관측될 때, 잡음 레벨을 경계에 덜 편향되게 추정하고 (2) 잡음이 서로 다른 입력에서도 경계-보존과 잡음-억제를 동시에 안정화하는 것이다. 이를 위해 NIWG는 3D Laplacian의 고주파 응답에 median absolute deviation(MAD) 기반 추정기를 적용해 speckle noise level을 robust하게 계산한 뒤, bounded power-law로 interaction weight를 만든다. 이후 wFiLM(Weighted Feature-wise Linear Modulation)이 이 가중치를 반영해 교차 가지(bidirectional) feature modulation을 스테이지별로 조절하며, 특히 더 깊은 상호작용에서는 speckle reduction 쪽 정보가 경계 스트림으로 과도하게 번지는 위험을 낮추는 비대칭 감쇠를 둔다.

- **Empirical Impact**: UterUS 데이터셋의 3D transvaginal 초음파 141개 볼륨에 대해, NBGL은 6개 잡음 레벨 전반에서 speckle reduction과 구조 보존, 그리고 주석 경계 일치 측면에서 최신 방법들을 일관되게 능가했다고 보고한다. 또한 추정된 잡음 레벨에 따라 경계-관련 feature coupling이 달라지는 설계가 이질적 잡음 상황에서도 성능을 유지하는 데 기여했음을 시사한다. 결과적으로 NBGL은 “잡음 적응 + 임상적으로 의미 있는 경계 보존”을 동시에 달성하는 프레임워크로서 디스피클링 연구의 실용성을 높인다는 의미가 있다.



### Erased, but Not Gone: Output Forgetting Is Not True Forgetting (https://arxiv.org/abs/2606.25001)
Comments:
          25 pages

- **Prior Approaches**: 머신 언러닝(MU)은 보통 잊기 세트에서 정확도 하락, logit 수준 membership inference 감소, 잔존 세트 성능 유지 같은 ‘출력(output) 단’ 지표로 성공 여부를 판단해 왔다. 이 접근은 겉보기로는 충분히 효과적인 것처럼 보이지만, 표현(representation) 공간에서의 잔여 불일치가 남아도 결과가 가려질 수 있다는 한계가 제기된다.

- **Core Contribution**: 이 논문은 retrained 모델(삭제 데이터 없이 처음부터 재학습)을 기준으로 ‘retraining-consistent representation forgetting(재학습-일관 표현 망각)’을 더 강한 평가 렌즈로 제안한다. 기존 출력-단 성공이 재학습과 정합적인 표현 변화로 이어지지 않을 수 있음을 체계적으로 진단하고, 현재 평가가 무엇을 ‘증명’하고 있는지 중심 질문에 답한다.

- **Technical Challenges**: 핵심 기술적 난제는 출력이 같더라도 중간 표현은 다를 수 있어, 이를 재학습과 비교해 정량화해야 한다는 점이다. 논문은 representation transformation과 forget/retain에 대한 정합성 측정, CKA로 표현 기하 유사도 평가, MIA rep로 표현 기반 잔여 유출을 측정하며, 나아가 retraining 방향/직교 부분공간 분해로 불일치가 ‘확산된 잡음인지’ ‘구조화된 잔여인지’를 분리해 분석한다.

- **Empirical Impact**: CIFAR-10/100, TinyImageNet과 ResNet-18/50, ViT-Tiny 등 다양한 설정에서 여러 MU 방법이 출력 단에서는 잘 잊는 것처럼 보이지만, 표현 공간에서는 재학습과 일관되지 않은 잔여 불일치가 남는 패턴이 반복된다. 또한 불일치가 forget/retain 비대칭, 특정 방향 불일치, 재학습 관련 방향에 집중된 잔여로 나타나며, 단일 벤치마크·모델·클래스·seed의 우연이 아니라 규모 전반에 걸친 숨은 실패 모드임을 시사한다.



### Geo-Strat-RL: Learning Geological Event Reasoning from Verifiable Tasks (https://arxiv.org/abs/2606.25000)
Comments:
          21 pages, 8 figures

- **Prior Approaches**: 기존 지구과학 AI는 원격탐사·지진 해석·암상 분류처럼 정답이 비교적 명확한 예측에 집중하거나, 대규모 언어/시각-언어 모델을 명령추종·요약으로 확장하는 흐름이었습니다. 다만 정적 도표에서 ‘숨은’ 지질 사건 연대기(시간·구조 관계)를 복원하는 능력을, 주석 없이도 검증 가능한 형태로 직접 시험하는 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 지질 역사 추론을 ‘검증 가능한(reward-verifiable) 시각 추론 과제’로 정의해, observation domain이 달라도 같은 잠재 사건 히스토리를 복원하도록 학습하는 RLVR을 제안합니다. 이를 위해 Geo-Strat-RL이라는 합성 환경을 만들고, 생성기+실행형 verifier가 사건의 chronology, event identity, 퇴적(deposition), 구조 관계를 분해해 보상을 계산합니다. 모델은 도표 입력에서 주석 없이도 JSON 형태의 사건 시퀀스를 생성해 재구성 타당성을 얻도록 학습됩니다.

- **Technical Challenges**: 핵심 난제는 정답 사건 히스토리가 관측만으로 유일하게 식별되지 않고, 사건 관계(절단/부정합/단층 영향 등)가 간접 관측·고도의 모호성을 가진다는 점입니다. 연구진은 이를 해결하기 위해 simulator가 ‘정답 사건 이력’을 생성하고, 실행형 verifier가 출력 JSON을 구조적으로 파싱해 구성요소별 점수를 주는 방식으로 학습 신호를 보장합니다. 또한 diagram과 seismic-style을 동일한 사건 히스토리로 페어링하고, open VLM에 LoRA 어댑터를 GRPO로 튜닝해 보상 기반 최적화를 실용화했습니다.

- **Empirical Impact**: 실험에서 diagram 도메인에서의 RLVR(예: Qwen3-VL-4B LoRA)이 held-out stratigraphic diagrams에서 geological content 점수를 크게 끌어올렸습니다. 더 나아가 같은 잠재 사건 히스토리를 seismic-style 합성 표현으로 바꿔 평가했을 때, seismic-specific 학습 예시 없이도 지질 추론 성능이 전이되는 증거를 제시해 reusable 개념 학습 가능성을 뒷받침합니다. 다만 fault와 unconformity 같은 구조 속성에서 여전히 오차가 커, 시간·구조 추론은 가능하지만 가장 미세한 구조 판정은 추가 개선 여지가 남아 있습니다.



### Internal Data Repetition Destroys Language Models (https://arxiv.org/abs/2606.24998)
- **Prior Approaches**: 언어모델 사전학습에서 중복 데이터는 성능을 해칠 수 있지만, 기존의 정밀 제어 연구는 주로 repetition damage를 파라미터 수 관점에서 간접 측정하거나 구형 scaling law로 다뤘다. 또한 deduplication이 완전하지 않아 near-duplicate·템플릿·의미 중복이 남는다는 문제는 널리 알려져 있었지만, “정확한 문서 재플레이” 같은 통제 조건에서 비용을 정량화한 연구는 제한적이었다. 따라서 compute(예: FLOPs) 축에서 최신 Chinchilla-style 예산과 연결해 손실과 중복 구조의 관계를 측정하기 어려웠다.

- **Core Contribution**: 이 논문은 Chinchilla 시대에 맞춰 “정확한 문서-level replay”에서 repetition이 손실에 미치는 영향을 Compute-Equivalent Gain(CEG)과 Compute-Equivalent Loss(CEL)로 재정의해 정량화한다. 고정된 반복 토큰 비율(10%) 안에서, 반복되는 풀의 크기와 반복 횟수(=repetition structure)가 손실을 어떻게 바꾸는지 iso-FLOP 조건으로 체계적으로 스윕한다. 그 결과 repetition damage가 무작위 노이즈가 아니라 model size에 따라 예측 가능한 패턴(중간 지점에서의 손실 피크)을 보인다고 제시한다.

- **Technical Challenges**: 핵심 난제는 중복이 “얼마나 많이(repetition amount)”가 아니라 “어떤 구조로(repetition structure)” 배치될 때 손실이 비단 단조 감소/증가가 아니라 피크를 만들 수 있다는 점이다. 이를 위해 학습 예산을 Chinchilla budget identity로 고정하고, 중복 토큰 비율은 유지한 채 RR(문서 재생 횟수)만 바꿔 compute 축에서 비교 가능한 실험 설계를 만든다. 또한 CEG/CEL 산출을 위해 no-repetition Chinchilla scaling law를 피팅해 손실 차이를 compute gap으로 번역하고, misspecified linear regression(중복 샘플이 만드는 통계적 tradeoff)으로 동일한 비단조 피크를 닮은 현상을 폐형식에 가깝게 재현한다.

- **Empirical Impact**: FineWeb-Edu-Dedup에서 Qwen3-style 모델(예: 344M 파라미터) 기준으로, 손실은 RR의 중간 구간에서 최대가 되며 가장 나쁜 반복 설정은 no-repetition 대비 FLOPs의 약 67% 수준에서 같은 손실을 내는 것으로 요약된다(CEL≈0.33). 피크 위치는 모델 크기에 대한 power law로 잘 맞아, “가장 위험한 반복 횟수는 compute보다 더 빠르게 증가”하는 경향을 보여 준다. 또한 단순 loss%가 아니라 compute-equivalent 관점에서 중복의 실제 낭비 규모가 커질 수 있음을 강조해, 중복이 남아 있는 프리트레이닝 파이프라인의 비용 추정을 더 정밀하게 할 수 있는 실무적 기준을 제공한다.



### Are Tabular Foundation Models Robust to Realistic Query Distribution Shifts in Microbiome Data? (https://arxiv.org/abs/2606.24995)
- **Prior Approaches**: 기존에는 Tabular foundation model(TFM)의 성능을 일반적인 표 데이터 벤치마크에서 주로 검증해 왔고, 마이크로바이옴에서는 compositionality와 zero-inflation 같은 구조가 견고성(robustness)에 어떤 영향을 주는지 체계적으로 분석이 부족했습니다. TFMs는 in-context learning(모델 파라미터 업데이트 없이 support set 컨텍스트만으로 추론) 방식이라, support-query mismatch가 커지면 취약할 수 있으나 이 실패 모드가 명확히 정리되지 않았습니다. 또한 선행 연구는 도메인 적응에 초점을 둔 반면, 실제 임상 배치에서 자주 생기는 분포 변화 유형별로 민감도를 비교한 평가는 제한적이었습니다.

- **Core Contribution**: 논문은 TFMs가 마이크로바이옴 abundance 데이터에서 realistic distribution shift에 얼마나 버티는지 평가하는 벤치마크를 제안합니다. 특히 ICL 설정에서 support set은 그대로 두고 query 샘플에만 biologically inspired perturbation을 가해, support–query mismatch 하에서의 일반화를 분리해 측정합니다. 또한 단순 성능 저하를 넘어 ‘shortcut’에 의존하는지 확인하기 위해 가장 discriminative한 taxa는 보존하고, 비정보성(uninformative) 특성에 대해서만 compositionality와 zero-inflation을 교란합니다.

- **Technical Challenges**: 핵심 기술적 난제는 마이크로바이옴 데이터의 compositional constraint와 zero-inflation을 동시에 조절하면서도 ‘의미 있는 신호를 통째로 제거’하지 않는 통제된 데이터 변형을 설계하는 것입니다. 이를 위해 비정보성 taxa만 타깃으로 삼고, (i) high-abundance taxa 제거(특징 부재로 인한 전역 compositional 구조 변화), (ii) sparsification(증가된 zero-inflation을 power transformation으로 유도 후 renormalization), (iii) zero-imputation(기존 0을 관측된 non-zero 값 분포에서 채워 넣고 renormalization) 3가지 전략을 구현했습니다. 또한 informative taxa는 AUROC 기반 제거 실험으로 데이터에서 자동 추정해, perturbation이 ‘구조 변화 민감도’가 아니라 ‘핵심 신호 손실’로 인한 차이가 되지 않도록 했습니다.

- **Empirical Impact**: 6개 gut microbiome 질병 컨텍스트에 대해 6종 TFM과 비교 모델을 대상으로 실험했으며, 전반적으로 모든 perturbation이 성능을 떨어뜨려 support-query shift의 취약성이 일관되게 관측됩니다. 특히 zero-imputation이 가장 큰 악영향을 보였고, discriminative taxa를 유지해도 global feature structure를 깨면 generalization이 쉽게 붕괴할 수 있음을 시사합니다. 또한 sparsification은 classical Random Forest 대비 TFMs에서 더 큰 민감도를 보여, zero-inflation 형태의 변화가 TFM의 안정성에 더 직접적으로 타격을 줄 수 있음을 실증적으로 보여줍니다.



### ExTra: Exploratory Trajectory Optimization for Language Model Reinforcement Learning (https://arxiv.org/abs/2606.24994)
Comments:
          15 pages

- **Prior Approaches**: 기존 RLVR 학습은 GRPO처럼 그룹 내 보상을 정규화해 value network 없이도 안정적으로 업데이트하지만, 쉬운 문제에서는 정답이 유사하게 반복돼 보상 분산이 붕괴하며 그라디언트 신호가 약해진다. 어려운 문제에서는 그룹 내 정답이 없어 advantages가 전부 0이 되어 업데이트 자체가 발생하지 않는 문제가 나타난다. DAPO 같은 완화는 pass rate가 특정 구간에 들지 않는 문제를 걸러 효율을 높이려 하지만, 결국 ‘어려운 문제의 밀도 높은 학습 신호’를 버리게 된다.

- **Core Contribution**: 논문은 GRPO 호환 프레임워크 ExTra(Exploratory Trajectory Optimization)를 제안해, 언어모델이 만든 롤아웃에서 탐색 신호를 직접 추출한다. 두 축은 (1) 정답에 한해 embedding 기반 novelty 보너스를 주어 ‘정답 다양성’을 회복하고, (2) hard 케이스에서 entropy 기반 prefix regeneration으로 중간 단계부터 탐색을 이어가도록 한다. 이를 통해 쉬운 문제의 diversity collapse와 어려운 문제의 all-incorrect 학습 공백을 동시에 겨냥한다.

- **Technical Challenges**: 핵심 기술 난점은 GRPO 정규화와 novelty의 결합 방식인데, novelty를 원시 보상에 더하면 그룹 평균·분산과 얽혀 이진 보상 영향이 왜곡될 수 있어 논문은 novelty를 GRPO 정규화 이후에 가산한다. 또한 어려운 문제에서는 정답이 없어 rewards가 0이므로, 추가 감독 없이 prefix의 유망함을 고르는 신호가 필요하며 이를 위해 Mean Token Entropy(MTE)를 도입하고, 불안정한 점수를 semantic similarity로 스무딩한 뒤 가장 낮은 MTE prefix를 선택해 접두사 이후 토큰에만 정책손실을 적용한다.

- **Empirical Impact**: 여섯 개 수학 추론 벤치마크(MATH-500, AMC23, Minerva, OlympiadBench, AIME24, AIME25)에서 Qwen3-1.7B 기준 ExTra는 GRPO 대비 pass@1을 약 +5점, pass@16을 약 +7점 향상시키며 단일 샘플 정확도와 추론-time coverage를 함께 개선한다. ablation에서 regeneration만 쓰면 pass@16이 오히려 줄어드는 반면, correctness-gated novelty만으로도 pass@1과 pass@16이 모두 오르고, 둘을 결합했을 때는 누적 개선을 넘어서는(조합 효과가 큰) 성능 상승이 관찰된다. 또한 trajectory 다양성 지표들(Self-BLEU, LogDet 등)과 샘플 효율(생성된 prompt instance 예산 대비 pass@16)에서도 ExTra가 GRPO와 DAPO보다 우수해, 학습 중 탐색 행동 자체를 지속적으로 바꾼다는 점을 뒷받침한다.



### Uncertainty-aware reinforcement learning for chemical language models (https://arxiv.org/abs/2606.24990)
- **Prior Approaches**: 기존 강화학습 기반 분자 설계(RL)는 다양한 분자 속성의 스코어링 함수를 대체로 deterministic oracle처럼 취급한다. 그 결과 예측 불확실성을 반영하지 못해, 훈련 데이터 지지도가 낮은 영역을 과도하게 탐색하고 최적화가 불안정해질 수 있다.

- **Core Contribution**: 이 논문은 RL에 예측 불확실성을 통합하는 두 가지 상보적 방법을 제안하고 비교한다. 첫째, 불확실성을 추가 최적화 목적지로 넣어 정책이 exploitation과 reliability를 함께 조절하도록 한다. 둘째, 정책 업데이트를 불확실성으로 가중해 신뢰 구간 밖의 분자 영향력을 줄인다.

- **Technical Challenges**: 핵심 기술 과제는 불확실성을 스코어링 단계와 학습 업데이트에 어떻게 안정적으로 반영하느냐이다. 저자들은 (i) 가우시안 예측 오차를 훈련 데이터와의 거리로 스케일링한 controlled model 시스템과, (ii) ChemProp 모델, (iii) conformal prediction을 적용한 random forest 분류기 등으로 불확실성 정의를 달리해 두 접근의 일관성을 검증했다.

- **Empirical Impact**: 실험에서 uncertainty-aware RL은 불확실성이 낮은 화학 공간을 우선 탐색해 더 견고한 hit discovery를 가능하게 했다. 그 결과 true hit rate가 0.5에서 0.75로 0.25p 상승했고, 총 true hits 수도 거의 2배로 늘었으나 분자 스코어의 손상은 크지 않았다. 이는 CLM 기반 RL에서 예측 신뢰도를 함께 최적화해야 함을 실증적으로 보여준다.



### When Multi-Sensor Fusion Fails to Generalize: Cattle Posture Classification Under Animal-Level and Temporal Distribution Shif (https://arxiv.org/abs/2606.24986)
Comments:
          20 pages, 6 figures

- **Prior Approaches**: 기존 소 자세 자세 분류 연구는 목걸이 가속도 센서 단일 신호와 전통 ML·딥러닝을 주로 사용해 0.9대 F1 같은 높은 성능을 보고해 왔다. 그러나 random/stratified train-test split처럼 개체 누출(leakage)이 발생하면 실제 배치 환경의 일반화 능력보다 과대평가될 수 있다. 더 나아가, LOAO 같은 개체 분리 평가를 해도 여전히 ‘시간에 따른 분포 변화(temporal distribution shift)’까지 체계적으로 검증한 사례는 드물었다.

- **Core Contribution**: 이 논문은 2024-2025년 연속 두 해에 걸쳐 목걸이 가속도, 위(rumen) 볼루스 생리 신호, 환경(기상) 변수를 수집하고 ‘바닥(lying) vs 서기(standing)’를 분류하는 강건성(robustness)을 평가한다. 특히 평가 프로토콜을 random 관측 분할→개체 단위 분할→leave-one-animal-out→교차연도(cross-year)로 단계적으로 강화해, 정확도만으로는 배치 준비도(deployment readiness)를 판단하기 어렵다는 점을 보여준다. 그 결과, 멀티모달 센서 융합이 일반화에는 불리할 수 있음을 데이터로 확인한다.

- **Technical Challenges**: 핵심 기술적 난제는 모델이 자세와 직접 연관된 신호를 학습하는지, 아니면 특정 해/환경에서만 유효한 ‘문맥 프록시(proxy)’를 학습하는지 구분하는 것이다. 연구진은 XGBoost를 중심으로 SHAP 기반 설명가능성 분석을 수행해 어떤 특징(특히 rumen-bolus 활동과 환경 변수가) 예측에 지속적으로 크게 기여하는지 추적했다. 또한 PCA와 domain classifier를 통해 두 해의 입력 분포가 통계적으로 유의미하게 분리되는 covariate shift를 정량화했으며, 그 결과 이동 신호만으로도 분포 변화가 존재함을 함께 확인했다.

- **Empirical Impact**: 연속 두 해 데이터에서 멀티모달 모델은 해당 해 내부 평가에서는 macro-F1 0.94 수준의 강한 성능을 보였지만, 교차연도 평가에서는 macro-F1이 0.49로 크게 하락했다. 더 흥미롭게도 교차연도에서는 오히려 collar-only(가속도/이동 기반) 모델이 멀티모달보다 잘해, 융합이 강건성을 항상 올리지는 않는다는 결론을 뒷받침한다. 연구는 흔한 벤치마크 정확도가 실제 배치 성능을 과대평가할 수 있으며, livestock-monitoring 분야에서 robustness-centred evaluation과 신호-문맥 의존성 점검이 필수임을 강조한다.



### Retrieval-Augmented Personalization with Foundation Models for Wearable Stress Detection (https://arxiv.org/abs/2606.24985)
- **Prior Approaches**: 웨어러블 스트레스 탐지는 개인차(생리 반응·행동 표현의 큰 변동) 때문에 개인화가 어렵다. 기존 연구는 사용자별 labeled 데이터로 fine-tuning/선택을 하거나, 메타러닝 등으로 적은 라벨 적응을 시도하지만 라벨 확보 비용이 문제다. 또한 인구통계·성격 등 side information을 쓰는 방식은 수집 부담과 편향 우려가 있으며, 사용자 임베딩을 학습하더라도 결국 라벨 의존이나 사전학습 비용이 남는다.

- **Core Contribution**: 이 논문은 labeled user data 없이도 사용자 이력을 검색해 개인화 임베딩을 만드는 retrieval-augmented personalization을 제안한다. frozen, out-of-domain foundation model로 사용자 이력에서 유사 패턴을 뽑아 compact personalized embedding을 만들고, 이를 lightweight transformer의 표현에 FiLM으로 조건 주어 스트레스 탐지 정확도를 끌어올린다. 또한 temporal retrieval(과거 샘플만으로 검색)과 cross-dataset retrieval(다른 데이터셋 K-EmoCon 임베딩으로 WESAD 개인화)을 함께 다뤄 배포 현실성까지 점검한다.

- **Technical Challenges**: 핵심 난제는 (1) 웨어러블용 대형 foundation model의 부재로 파인튜닝 가능한 기반이 부족하고, (2) WESAD처럼 사용자 수가 적어 복잡한 사용자 임베딩 학습이 비실용적이며, (3) 사용자별 이질성이 커서 과적합 위험이 크다는 점이다. 저자들은 예측용 네트워크는 가볍게 CNN-Transformer로부터 scratch 학습하되, 사용자 컨텍스트는 frozen out-of-domain 모델(MOMENT-1, Chronos-2, HuBERT 기반)을 임베딩 생성기로 써 cosine similarity top-K 검색으로 해결한다. 검색 결과는 SetTransformer 유도점 기반으로 차원/개수를 줄인 뒤 FiLM의 gamma·beta로 표현을 조절하고, base와 personalized loss를 함께 학습하되 EMA로 분기 기여도를 자동 완충해 early collapse를 방지한다.

- **Empirical Impact**: WESAD(N=15, EDA/BVP/temperature/ACC)에서 비개인화 transformer baseline 대비 accuracy +3.92%, macro F1 +4.76% 개선을 보고하며, 라벨 없는 personalized 임베딩으로 supervised fine-tuning 성능에 근접함을 보인다. temporal retrieval은 full intra-user retrieval에 가까운 성능을 내 “사용자 이력이 제한적일 때도” 강건함을 시사한다. 또한 K-EmoCon 임베딩을 검색 데이터베이스로 쓰는 cross-dataset setting에서도 추가 분석을 통해 foundation model 표현이 데이터 조건 차이를 어느 정도 일반화할 수 있음을 탐색한다.



### Why Do Accumulated Transformations Extrapolate? (https://arxiv.org/abs/2606.24975)
Comments:
          33 pages, submitted to TMLR

- **Prior Approaches**: PaTH Attention은 RoPE의 위치 인덱스 회전을 Householder 반사의 누적 곱으로 바꿔 긴 문맥에서도 길이 외삽이 잘 되는 현상을 보였지만, 왜 그런지 메커니즘은 완전히 정리되지 않았다. 또한 NoPE 같은 길이-안정(평탄) 설계는 가능성을 보여주지만, 멀리 있는 토큰이 ‘얼마나’ 억제되는지, 값(value)까지 남는 잔여 간섭이 어떻게 누적되는지는 충분히 설명하지 못한다.

- **Core Contribution**: 이 논문은 Householder 구조가 핵심인지, 더 일반적인 ‘소스-쿼리 경로(path) 위의 누적 변환’ 성질에서 비롯되는지 묻는다. 결론적으로 누적 직교 변환이 특정 정칙성(regularity) 조건을 만족하면, 누적 곱이 유한 단계 후 서로 비상관(incoherent) 상태가 되어 원거리 토큰에 대한 어텐션이 억제되며, 다만 컨텍스트가 무한히 커지면 결국 성능이 저하되는 하한도 함께 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 누적 변환이 ‘콘텐츠-의존적 mixing window’를 어떻게 만들고, (2) 그 결과 score gap이 실제로 far attention mass를 줄이는지, (3) 그럼에도 far set이 커질 때 near 신호가 영원히 보존될 수 있는지(불가능하다는 하한)를 동시에 정량화하는 것이다. 논문은 스펙트럴 갭을 이용한 누적 곱의 기하급수적 탈상관, 고차원 농도(concentration)로 softmax 상에서의 score gap 형성, 그리고 far mass가 커질수록 근본적으로 near-signal이 붕괴한다는 lower bound를 증명하며, SO(2)^{SO(2)} 특수 케이스에서는 값까지 같은 경로로 회전해 잔여 far 기여를 비동조(incoherent)로 결합시키는 추가 보장도 논증한다.

- **Empirical Impact**: 통제된 실험에서 무작위 누적 회전은 RoPE 대비 외삽을 크게 개선했으며, 학습된 token-dependent 회전은 훈련 컨텍스트를 훨씬 넘어 perplexity가 유지되는 패턴을 보였다(예: 훈련 길이 대비 16배 수준 평가). 또한 rotation-only 모델은 극단적 길이에서 점진적 열화를 보이지만 ALiBi는 상대적으로 길이 안정적이어서, 논문의 ‘far-mass 제어가 구조적으로 필요하다’는 주장과 일치한다. 값(value)까지 회전하는 설정은 query/key만 회전하는 경우보다 긴 문맥 열화를 더 완화해, 이론의 추가 보호 메커니즘을 지지한다.



### Quantifying Explainable AI-introduced signal noise on ECG data with Spectral Entropy (https://arxiv.org/abs/2606.24974)
Comments:
          Accepted to EUSIPCO 2026

- **Prior Approaches**: 기존 XAI(설명가능한 인공지능) 기법들은 휴리스틱에 기반해 모델 설명을 생성하며, 특히 의료 분야에서는 신뢰성과 의사결정의 정당화가 중요해 더 널리 활용된다. 하지만 설명 생성 과정에서 추가되는 신호와, XAI 자체가 만들어내는 잡음(노이즈)을 구분하기가 어렵다는 문제가 제기돼 왔다.

- **Core Contribution**: 본 논문은 XAI 출력에 포함된 잡음을 정량화하기 위한 지표로 spectral entropy(스펙트럴 엔트로피)를 제안한다. 이를 통해 설명에서 ‘모델이 준 신호’와 ‘설명 도구의 잡음’의 구분 가능성을 높이는 데 초점을 둔다.

- **Technical Challenges**: 핵심 난제는 설명 결과에 섞이는 잡음이 무엇인지 명확히 측정하는 방법이 부족하다는 점이다. 논문은 설명 출력의 스펙트럼 특성을 기반으로 spectral entropy를 계산해 잡음 정도를 비교 가능하게 만들고, 다양한 post hoc explainability techniques에 적용해 유용성을 검증한다.

- **Empirical Impact**: ECG 데이터셋의 부정맥 분류(classifying arrhythmias) 실험에서 서로 다른 사후 설명 기법들의 출력 품질을 spectral entropy로 구분할 수 있음을 보인다. 의료용 모델에서 설명의 신뢰성을 높이려는 XAI 평가에 실질적으로 기여할 수 있는 접근으로 평가된다.



### LLM Performance on a Real, Double-Marked GCSE Benchmark (https://arxiv.org/abs/2606.24973)
- **Prior Approaches**: 기존 자동 채점은 규칙 기반과 신경망을 거쳐 최근에는 LLM을 활용하지만, 표준 영어 에세이 벤치마크에서는 성능이 일관되지 않았고 task-specific 시스템을 완전히 대체하지 못했다. 특히 여러 과목을 한 번에 다루거나 손글씨/그림까지 포함하는 실제 시험 응답에 대해선 여전히 인간과의 신뢰도 격차가 남아 있었다. 그 결과 “사람 채점기준에 맞춰 얼마나 재현하느냐”가 명확히 검증되지 않은 경우가 많았다.

- **Core Contribution**: 이 논문은 GCSE 모의고사 응답 32,534개(5과목, 328문항)를 2명의 자격 심사위원이 이중 채점한 데이터셋을 제시하고, LLM이 두 심사위원 합의(컨센서스)에 얼마나 가깝게 채점하는지 측정한다. 특히 손글씨 작업이 포함된 멀티모달 답안까지 포함해, 기존 텍스트 중심 벤치마크의 한계를 현실 조건에 가깝게 보완한다. 또한 “모델-인간” 단순 상관이 아니라 “모델-인간 합의가 인간-인간 합의만큼 일관적인가”를 지표로 삼는다.

- **Technical Challenges**: 핵심 난제는 (1) 과목별 mark scheme을 따르면서 (2) 손글씨를 읽고 (3) 혼재된 답안 형식(서술형·수학식·도형·그림)을 동일한 흐름으로 점수화하는 것이다. 논문은 단일 generic prompt에 질문·채점기준·학생 답안을 넣고, 손글씨는 캔버스 이미지를 첨부해 읽도록 하며, 출력은 JSON 스키마의 정수 점수로 구조화해 채점 안정성을 확보했다. 그리고 ordinal 점수의 일관성을 위해 QWK(Quadratic Weighted Kappa)를 문항별로 계산한 뒤 가중 평균하고, 오차 구간은 문항 클러스터 부트스트랩으로 추정한다.

- **Empirical Impact**: 결과적으로 최상위 모델들은 대부분 과목에서 “두 심사위원 간 합의”보다 더 높은 수준으로 심사위원 컨센서스에 일치하며, 특히 영어 에세이 같은 주관적 과제에서도 높은 점수대를 만든다. 모델 크기와 무관하게 성능 격차가 크지 않았고, 편향(관대/엄격) 오프셋은 모델별로 뚜렷하게 나타나 특정 모델들이 더 중립적임이 관찰됐다. 또한 비용 분석에서 더 저렴한 모델들이 비싼 모델 대비 유사하거나 더 나은 일치도를 보여, cost-effective automated marking 가능성을 실증적으로 뒷받침한다.



### What Do Language Priors Contribute to Darcy-Flow Inversion? A Mechanistic Aud (https://arxiv.org/abs/2606.24967)
- **Prior Approaches**: 역문제(inverse problem)는 데이터만으로 해를 고정하기 어려워, 결국 어떤 해를 “가능한 해”로 볼지에 prior(정규화/상관모델/학습 앙상블)가 큰 비중을 차지한다. 기존 공학적 접근은 Tikhonov·total variation·variogram 등 수치적 형태의 prior를 주로 다루지만, 지질학자가 실제로 쓰는 범주적·형태학적 지식(층의 연속성, 사행/렌즈/채널의 형태 등)을 그대로 넣기 어렵다.

- **Core Contribution**: 이 논문은 텍스트(지질 설명)를 inference-time에서 learned Darcy-flow inverse solver의 prior로 주입하기 위해, sentence embedding을 “언어-물리 인터페이스”로 사용하는 방식을 체계적으로 실험한다. 6개의 합성 지질 클래스와 SPE10 모델 2에 대한 전이 실험에서, 텍스트 조건이 no-text 반사실 대비 재구성 오차를 81% 줄였고 이는 언어가 언제·어떤 신호를 전달하는지까지 분해해 보여준다.

- **Technical Challenges**: 핵심 기술 과제는 텍스트가 실제로 무엇을 담아 신호를 만들지(클래스 수준 제약인지, 클래스 내부의 인스턴스/형태 디테일인지)와 그 기여 경로가 FiLM 같은 조건부 모듈에서 어떻게 작동하는지 검증하는 것이다. 연구진은 U-Net 기반 역해 생성기에 FiLM으로 SBERT embedding을 주입하되, 실험 변수로는 조건 표현만 바꿔 공정 비교를 수행하고, paraphrase ensemble 기반 민감도 프록시와 specificity 수준별 분석으로 “언어가 기여하는 정보의 성격”을 정량화한다.

- **Empirical Impact**: 결과적으로 텍스트 이득의 대부분은 해석이 underdetermined로 남는 영역에서의 범주(class) 제약이 담당하며, 동일 클래스 내부의 기하 디테일은 상대적으로 부차적이고 패턴 의존적으로 나타났다. 또한 sentence embedding은 discrete class label만으로는 얻기 어려운 학습 안정성, paraphrase 기반 sensitivity analysis, 그리고 6개 폐쇄 분류 밖의 open-vocabulary 입력 처리 가능성을 제공하지만, 개별 관측(정밀한 데이터) 정확도 자체를 크게 끌어올리지는 못하는 것으로 드러나 언어 priors의 “도움의 조건”을 명확히 한다.



### Enhancing Clinician Decision-Making via Uncertainty-Aware Multi-Expert Fusion for Stroke Rehabilitation (https://arxiv.org/abs/2606.24960)
- **Prior Approaches**: 기존 ARAT(및 유사 척도)은 성공 여부와는 달리 ‘움직임의 조직화 방식’과 회복-보상 차이를 충분히 반영하지 못한 채, 관찰을 0~3의 단일 서열 점수로 압축한다. 자동화 시도는 대체로 단일 임상의(또는 합의/다수결) 라벨을 정확히 맞추는 데 초점을 두며, 관찰 잡음과 전문가 간 주관성에서 생기는 모호함을 불확실성으로 다루지 못해 임상 도입에 장벽이 된다.

- **Core Contribution**: xAARA는 자동 평가를 ‘대체’하기보다 임상의 판단을 ‘증강’하도록 설계된 엔진으로, 다중 뷰 비디오에서 ARAT를 과제(task), 운동 단계(movement-phase), 움직임 품질(movement-quality) 수준으로 분해해 산출한다. 핵심은 임상의 점수를 ill-posed 추론 문제로 보고, 불확실성을 보정(calibrated uncertainty)해 신뢰도 낮은 케이스는 보고를 유예하는 방식으로 임상 의사결정 흐름에 맞춘 점이다.

- **Technical Challenges**: 임상 점수는 단일 정답처럼 고정되어 있지 않고, 같은 서열 점수도 여러 ‘다른 움직임 조직’에서 나올 수 있어 영상 기반 역문제 자체가 모호하다. xAARA는 EAGM(환경-활동-목표-의미) 다층 표현과 bidirectional Dynamic Bayesian Network(DBN) 기반의 Dynamic Bayesian Network에 692개 calibrated multimodal 모델을 entropy 기반 gating으로 조합해, 예측 정확도와 함께 어떤 경우에 신뢰할 수 있는지까지 함께 산출한다.

- **Empirical Impact**: 105명(788 exercises)에서 xAARA는 과제 정확도 94.2%(Cohen’s kappa=0.934), 운동 단계 정확도 81.3%(kappa=0.727)를 달성했으며 단일 임상의 점수 대비 예측 불확실성을 96.1% 줄였다. 전문가 간 주관성이 큰 구간에서도 두 명의 임상가 중 ‘최소 한 명과 일치’하는 점수를 100%의 과제에서 반환하고, 범위 밖(out-of-range) 점수는 생성하지 않아 임상적 신뢰성과 적용 가능성을 보였다.



### Reliable Conformal Prediction for Ordinal Classification Using the Ranked Probability Scor (https://arxiv.org/abs/2606.24959)
- **Prior Approaches**: 기존 확률적 순서형 분류(OC)는 포인트 예측 성능(예: mean absolute error, quadratic weighted kappa)을 높이는 데 집중해 왔고, 불확실성 정량화는 최근에야 본격적으로 다뤄지는 흐름이다. Conformal prediction(CP)은 분포-무관(in distribution-free)한 marginal coverage를 주지만, conformal ordinal prediction(COP)에서 핵심인 “라벨의 연속성”과 “오차의 심각도(ordinal miscoverage의 거리)”를 동시에 만족시키는 비비용·비편향 설계가 어려웠다. 특히 모드 중심 또는 탐욕/그리디 임계값 선택 방식은 집합의 크기 효율은 맞추더라도 순서 구조에 대한 충실도(ordinal risk)를 덜 반영할 수 있다.

- **Core Contribution**: 이 논문은 COP에서 비순응도(nonconformity) 함수로 ranked probability score(RPS)를 제안한다. RPS(서열 결과를 위한 proper scoring rule)는 누적(cumulative) 예측분포에 기반해 순서형 오차의 “심각도”를 자연스럽게 반영하며, 그 결과 RPS 기반 CP는 구성 과정에서 median-centered 형태의 연속(contiguous) 예측 집합을 만든다. 또한 모델-애그노스틱이라 어떤 기반 확률 예측기와도 결합 가능하고, grouped ordered categorical은 물론 assessed ordered categorical까지 지원한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) CP의 분포-무관 marginal coverage를 유지하면서 (2) 연속성 제약을 만족하고 (3) 오차가 집합 밖에 나갈 때의 ordinal miscoverage 크기까지 의미 있게 통제하는 비순응도 설계를 찾는 것이다. 저자들은 RPS를 nonconformity로 쓰면 교정(calibration) 단계의 표준 분위수(quantile) 구성만으로 marginal coverage가 보장되고, 더 나아가 α에 대한 nesting 및 contiguity 같은 COP에서 바람직한 성질이 이론적으로 도출됨을 보인다. 계산 측면에서도 라벨 수와 calibration 포인트 수에 대해 선형 스케일링하는 방식으로 구현 효율을 확보해, greedy한 interval selection 계열 대비 실용성을 높였다.

- **Empirical Impact**: 여러 순서형 이미지·테이블 데이터셋에서 RPS 기반 CP는 연속 예측 집합을 제공하면서, 기존 CP 방법들과 비교해 예측 집합 폭과 ordinal miscoverage 크기의 균형이 유리하다고 보고된다. 특히 median-centered 집합이 단순히 집합 크기만 줄이는 방식이 아니라, oracle 조건부 커버리지 하에서의 ordinal risk(l1 set-based error)를 더 직접적으로 최적화하는 경향을 보인다는 점을 강조한다. 고위험 도메인에서 “어느 정도로 틀렸는지”까지 반영하는 불확실성 집합이 필요할 때, RPS 기반 COP는 실무 적용 가능성이 높은 대안으로 자리잡을 수 있다.



### Convex--Concave Quadratic Spectral Filtering for Graph Neural Networks (https://arxiv.org/abs/2606.24956)
- **Prior Approaches**: 기존 스펙트럴 GNN은 메시지 패싱을 라플라시안의 주파수 필터링(대개 다항식 필터)로 해석하며, 저차 필터는 효율적이지만 패스밴드 밖의 감쇠 선택도가 약해 구조 신호가 흐려질 수 있습니다. 반면 고차 다항식/고차 스펙트럴 필터는 더 날카로운 응답을 만들지만 최적화 난이도와 구조 교란에 대한 민감도가 커지는 한계가 있습니다. 또한 여러 방법은 필터 기저를 확장하거나 다항 차수를 늘려 표현력을 키우는 데 집중해, 곡률(convex/concave) 극성 자체를 명시적으로 설계 축으로 삼지는 않았습니다.

- **Core Contribution**: DCQ-GNN은 필터 차수를 2로 고정한 채, 적응형 convex–concave 2차(quadratic) 필터 뱅크로 스펙트럴 선택도를 높이는 접근을 제안합니다. 핵심은 다항식 차수를 올리지 않고도 곡률 극성(curvature polarity)을 조합해 패스밴드 전이의 날카로움을 개선한다는 점이며, 각 채널 출력은 node-adaptive gating으로 노드별 구조 의존적 스펙트럴 선택을 수행합니다. 즉, “표현력 = 차수 증가” 대신 “안정성-선택도 트레이드오프를 곡률로 최적화”하는 관점을 실증적으로 밀어붙입니다.

- **Technical Challenges**: 도전 과제는 차수를 늘리지 않으면서도 고주파 감쇠를 더 강하게 만들고, 그 효과를 이론적으로(예: Dirichlet energy 감쇠, von Neumann entropy, 곡률 극성) 설명하는 것이었습니다. DCQ-GNN은 엄밀한 스펙트럴 분석을 바탕으로 2차 필터가 선형 필터 대비 고주파 대역의 최악 Dirichlet energy를 더 낮출 수 있음을 보이고, 엔트로피 관점에서 채널별 스펙트럴 누출을 줄이는 메커니즘을 제시합니다. 또한 곡률이 상반된(볼록/오목) 성분을 함께 두되, 게이팅이 노드별로 채널을 가중해 이질성 그래프에서 생길 수 있는 과집중의 부작용을 완화하도록 설계했습니다.

- **Empirical Impact**: 10개(총 10 datasets) 벤치마크에서 DCQ-GNN은 이종성(heterophily) 그래프에서는 평균 순위 공동 1위(3.0), 동질성(homophily) 그래프에서는 평균 순위 2위(4.2)를 기록하며 고차 스펙트럴 필터들과 경쟁적인 성능을 유지했습니다. 특히 강한 구조적 교란 조건에서는 1차 및 고차 기반 대비 성능 저하가 더 작아, 곡률 인지 저차 필터링이 견고함(robustness)을 준다는 가설을 뒷받침합니다. 종합하면, “높은 차수로 해결”하던 문제를 “곡률 극성 기반의 컴팩트 quadratic bank + 게이팅”으로 대체해, 최적화 안정성과 계산 효율을 보존하면서 선택도와 강건성을 함께 얻는 방향을 제시했다는 의미가 큽니다.



### Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models (https://arxiv.org/abs/2606.24952)
- **Prior Approaches**: 기계적 해석( mechanistic interpretability )의 핵심 기대는 ‘제어가능성’이다. 기존 연구들은 특정 동작을 활성에서 찾아내면, 해당 방향을 residual stream에 더하거나 ablation해 동작을 조절할 수 있다고 보고했으며(예: refusal 매개 단일 선형 방향, truth 관련 방향의 추가), 이를 곧바로 “읽기=제어”로 해석해왔다. 하지만 이 관점에는 ‘검출을 잘하는 방향과 실제로 개입을 일으키는 방향이 같거나 가깝다’는 숨은 가정이 들어 있다.

- **Core Contribution**: 이 논문은 그 가정을 기하학적으로 정량화해, ‘동작 검출 방향’과 ‘동작 개입(제어) 방향’ 사이 각을 각도/코사인으로 측정한다. detection-intervention gap이라는 개념을 제시하며, 코사인이 1에 가깝지 않으면 검출만 하고 제어는 못하는 분리가 존재함을 뜻한다고 본다. 특히 Gemma 2-2B-it에서 output format은 정렬(거의 동일 축)되는 반면, hallucination은 크게 벌어지는 ‘아는 것과 조종하는 것의 분리’를 실험적으로 보여준다.

- **Technical Challenges**: 어떤 방향이 ‘검출’을 최적으로 하는지, 그리고 ‘제어’를 최적으로 하는지를 일관된 방식으로 정의하고 비교하는 것이 핵심 기술 문제다. 연구진은 residual stream에서 두 조건 간 difference-in-means(데이터 기반) 및 lm_head 가중치에서 읽은 토큰 대비(핸드-픽)로 검출 방향을 만들고, 생성 중 residual stream에 α·d를 주입해 개입 방향의 인과적 효과를 확인한다. 그 결과 hallucination에서는 fake entity를 완벽에 가깝게 선형 분리(AUC=1.000)해도, 그 방향과 refusal을 만드는 방향의 코사인이 약 0.12(약 83°)로 크게 어긋난다는 점을 보여준다.

- **Empirical Impact**: 이 gap은 1B–9B 크기의 4개 모델(세 계열)에서 코사인이 [0.12, 0.20] 범위를 벗어나지 않을 정도로 재현되며, instruction tuning 전후에도 거의 동일해(pretraining 기원) 구조적 현상임을 시사한다. 또한 검출-제어 정렬의 코사인이 steerability(조향 가능성)를 예측하리라는 직관은 틀렸는데, detection은 고차원 ‘클래스’이고 실제 조향성은 정적 각도만으로 읽히지 않는 기능적 조건에 달려 있기 때문이다. 요약하면, 이 코사인은 제어 다이얼이라기보다 ‘아는 것과 steering이 분리되는 서명(signature)’으로서 의미가 있다.



### MacroLens: A Multi-Task Benchmark for Contextual Financial Reasoning under Macroeconomic Scenarios (https://arxiv.org/abs/2606.24950)
Comments:
          25 pages, 3 figures

- **Prior Approaches**: 기존 시계열 벤치마크는 텍스트를 빼거나 비금융 텍스트/합성 데이터를 쓰는 경우가 많아, 금융 의사결정에 필수인 ‘가격-펀더멘털-거시-문서’ 동시 조건을 그대로 재현하기 어렵습니다. 금융 언어/문서 벤치마크도 문서 이해·분류에 초점이 있고, 문서가 공개되는 시점과 분기 공시 지연 같은 시간 정렬 문제를 엄밀히 다루지 못합니다. 또한 멀티모달 포캐스팅에서 텍스트 이득을 보여주더라도 실제 금융 데이터와 동일한 누수 방지 제약을 만족하는 공개 벤치마크는 부족한 상태였습니다.

- **Core Contribution**: MacroLens는 미국 소형·초소형 4,416개 종목에 대해 2021~2026년 기간의 ‘한 시점(point-in-time) 멀티모달 패널’ 위에서 7개 과제를 동시에 평가하도록 설계된 첫 대형 금융 벤치마크입니다. 가격 히스토리, XBRL 회계 펀더멘털, FRED/EIA 거시 지표, SEC 공시/뉴스 텍스트를 한 프레임으로 묶고, 여기에 1,130개의 거시 시나리오 이벤트(자연어 렌더링) 레이어까지 추가합니다. 특히 기존 벤치마크에 없던 private-company valuation, statement generation(자연어 설명→재무문서), real-estate valuation 같은 역량을 포함해 사모펀드/VC 관행에 더 가깝게 확장했습니다.

- **Technical Challenges**: 핵심 기술 난제는 평가 누수를 방지하면서도 실제 의사결정에 필요한 신호들을 ‘동일한 의사결정 시점에 관측 가능하도록’ 정렬하는 것입니다. 논문은 텍스트는 publication/filing date로 게이팅하고, 분기 펀더멘털은 as-reported(보고기간 종료 기준)로 맞추며, 매크로는 reference date로 동기화하고, 목표(예: 시가총액)와 대수적으로 연결된 피처는 사전에 제외하는 4가지 불변 조건을 벤치마크 구축 단계에서 강제합니다. 여기에 회사가 신호를 구조적으로 제공하지 못하는 경우(zero fill 대신) applicability masking을 적용해, 모델이 없는 정보를 ‘있는 것처럼’ 학습·평가하지 않게 했습니다.

- **Empirical Impact**: 실험에서는 19개 방법을 6개 계열(naive, classical, deep sequence, time-series foundation model 계열, time-series용 fine-tuning LLM, zero-shot LLM)로 나눠 과제별 성능을 비교하며, 두 개의 frontier LLM에 대해 5-step feature-context ablation과 gradient-boosted baseline도 함께 수행합니다. 평가지표는 예측 MSE·시나리오 조건부 수익 MAE·밸류에이션 MedAPE/MedAPE·문서 생성 parse rate 등 과제 성격에 맞춰 구성되고, sector·시가총액 분위·시나리오 유형·공시 밀도 등으로 계층화해 적용 가능성도 점검합니다. 또한 zero-shot LLM에 대한 테스트 구간 ‘암기/오염’ 가능성을 점검하는 recall 테스트까지 포함해, 벤치마크 결과가 누수·기억이 아닌 실제 추론 차이를 반영하는지 확인하려는 접근을 보여줍니다.



### What Does a Pathological Speech Assessment Model Know about Acoustic Features? A Case Study on Oral and Oropharyngeal Cancer Patients (https://arxiv.org/abs/2606.24949)
- **Prior Approaches**: 기존 음성 병리 발화(구강·구인두암) 평가는 주로 음향 특징과 판별 모델의 성능에 집중해, 모델이 어떤 음향 단서를 쓰는지 해석 가능성이 제한적이었다. 또 Wav2Vec 2.0 같은 self-supervised 표현을 쓰더라도, 표현과 임상에서 쓰이는 저수준 음향 기술자 간의 정량적 연결이 부족했다.

- **Core Contribution**: 이 논문은 Wav2Vec 2.0 기반 speech intelligibility assessment 모델의 해석 가능성을 canonical correlation analysis로 분석한다. 모델의 embedding과 eGeMAPS의 LLD(저수준 디스크립터)를 해석 가능한 기준으로 두고, 레이어별로 어떤 음향 정보가 인코딩되는지 추적한다.

- **Technical Challenges**: 핵심 과제는 고차원 모델 embedding과 임상 친화적인 음향 LLD 사이의 대응을 안정적으로 찾는 것이었다. 저자들은 layer-wise 개별 LLD 상관과 prosodic·spectral·voice quality 같은 group-level 상관을 동시에 계산해, 어떤 그룹이 강하게 반영되는지와 MFCC 계수 중 어떤 단서가 일관되게 기여하는지 확인했다.

- **Empirical Impact**: 실험 결과, 학습 표현은 spectral과 prosodic 특징과의 상관이 가장 높았고, 모든 레이어에서 첫 번째 MFCC 계수가 가장 큰 상관을 보였다. group-level 상관은 spectral 0.77, prosodic 0.71, voice quality 0.65로 보고되며, 병리 발화 평가에서 어떤 음향 특징 선택이 유리한지에 대한 실용적 가이드도 제공한다.



### Holographic Memory for Zero-Shot Compositional Reasoning in Knowledge Graphs: A Mechanistic Study of Where and Why It Fails (https://arxiv.org/abs/2606.24948)
Comments:
          15 pages, 5 figures, 5 tables. Code available at this https URL

- **Prior Approaches**: 기존 KGE 모델(TransE, DistMult, ComplEx, RotatE)은 단일-hop 링크 예측에는 강하지만, 학습 중 보지 못한 관계 체인을 테스트 시 조합해 추론하는 기본 메커니즘이 약하다. 멀티홉 추론을 다루는 Query2Box, BetaE, CQD 등은 경로(또는 중간 표현)에 대한 명시적 감독이나 쿼리 구조 분해가 필요해 본 논문의 ‘zero-shot compositional queries’ 설정과 거리가 있다. 반면 HRR/VSA는 결합·해결(unbind)이 이론적으로 가역적이며 결합적이라, zero-shot 조합 추론의 후보로 기대돼 왔다.

- **Core Contribution**: 이 논문은 HRR 기반 두 변형(real-valued HRR, phase-only Fourier HRR: FHRR)이 진짜로 zero-shot 다중홉 조합을 수행할 수 있는지, 그리고 실패가 어디서 발생하는지 기계적으로 분해한다. 특히 Hopfield cleanup을 결합해도 조합 성능이 chance 수준에 머무는지 확인하고, ‘중간 엔터티는 복원되는데도’ 조합이 깨지는 이유를 단계별 프로브로 국소화한다. 결론적으로 해결 과제는 cleanup 재설계가 아니라 superposition 하에서의 검색 capacity(용량/간섭) 개선임을 강조한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 결합·해결이 가역적으로 보이는 알제브라가, 다중 사실이 superpose된 실제 메모리에서 동일하게 작동하는지, (2) Hopfield cleanup이 이를 보정할 수 있는지다. 저자들은 hop-1 프로브로 중간 엔터티 복원(MRR 약 0.896±0.002)을 검증한 뒤에도 hop-2 조합이 실패함을 보이며, 두 번째 프로브로 ‘정답 2-hop 사실을 standalone atomic query로 물었을 때’조차 평균 atomic accuracy의 0.26~0.48배로 떨어지는 것을 확인한다. 추가로 FHRR의 softmax Hopfield cleanup이 위상 결합과 non-commutative(정확히는 phase-equivariant하지 않음)임을 Lemma 4.1로 증명해, hop-1이 틀리는 소수 구간에서는 2차 실패가 누적됨을 제시한다.

- **Empirical Impact**: FB15k-237에서 두 모델 모두 단일-hop retriever로서 경쟁력 있는 성능을 보이며, filtered MRR은 HRR 0.358±0.002, FHRR 0.350±0.021 수준이다. 그러나 선택된 zero-shot 2-hop 체인들에 대해서는 cleanup temperature 전반에서 정확도가 chance에 머물러 조합 이득이 관측되지 않았다(훈련이 주는 compositional advantage 부재). 무엇보다 실패 원인이 ‘cleanup/알제브라’가 아니라 superposition 하의 용량·간섭 효과이며, 이는 이미 hop-1 단계에서 측정되는 intrinsic hardness라는 점이 설계 방향을 바꿀 의미가 크다.



### EmotionAI: A Privacy-Preserving Computational Intelligence Pipeline for Speech-Emotion-Grounded Conversational Analysis (https://arxiv.org/abs/2606.24941)
Comments:
          12 pages, 4 figures. Submitted to UK Workshop on Computational Intelligence (UKCI 2026)

- **Prior Approaches**: 기존 SER은 MFCC 같은 수작업 특징에 얕은 분류기를 붙이거나, wav2vec2 같은 self-supervised 임베딩을 쓰더라도 데이터셋 간 감정 표현 차이로 전이가 급격히 깨진다. 또 LLM을 활용해도 텍스트 기반은 망설임 같은 청자 단서가 사라지고, 오디오 LLM은 보통 멀티모달 미세조정과 결합돼 상품용 CPU 목표와 충돌한다. 따라서 감정 분류의 ‘정확도’보다, 감정 단서를 근거로 설명 가능한 대화 분석을 만드는 CI 통합이 부족했다.

- **Core Contribution**: EmotionAI는 모든 처리를 로컬에서 수행하는 Computational Intelligence 파이프라인으로, diarisation-Whisper-ASR-wav2vec2 SER로 얻은 구간 단서(감정 증거)를 로컬 LLM 패널의 질의응답에 연결한다. 핵심은 timestamp-grounded Q&A와 citation-constrained 프롬프트로, 생성이 특정 구간과 감정 근거에 묶이도록 설계한 점이다. 또한 deception detection이 아니라 감정 상태 리뷰/화자 상태 분석을 돕는 도구임을 전제로, 불완전한 증거를 “감사 가능하게” 사용하는 방식을 제안한다.

- **Technical Challenges**: 기술적 난관은 (1) 감정 분류기가 구간별로 제공하는 불완전한 확률 증거를 (2) LLM이 텍스트로 해석할 때 환각이나 근거 이탈 없이 (3) 로컬 장치에서 지연까지 감당해야 한다는 점이다. EmotionAI는 단계 경계를 고정된 출력 스키마로 분리하고, phases 사이 메모리를 정리해 LLM 로딩 시 피크 메모리를 억제하며, 패널이 확률 문자열을 그대로 인용하지 않고 구간 타임스탬프를 근거로 자연어로 변환하도록 강제한다. 그 결과 외부 네트워크 호출 없이도 로컬에서 Q&A와 설명을 생성하며, ASR을 포함한 전체 실행은 CPU에서 끝나도록 구성했다.

- **Empirical Impact**: RAVDESS의 4클래스(English subset) zero-shot 평가에서 로컬 wav2vec2-large SER는 48.8% 정확도로 랜덤(24.9%)과 다수(28.6%)는 넘지만, in-domain MFCC+logistic-regression(71.0%)에는 못 미쳤다(특히 Sad F1=0.010로 붕괴). 다만 감정 근거를 Q&A에 포함하면 emotion-keyed 질문에서 refusal이 크게 줄어들어(감정 포함 8/12 vs 감정 제거 67%) ‘감정 근거의 필요성’이 실험적으로 확인됐다. 전체 파이프라인은 평균 157초로 CPU 지연의 대부분이 Whisper ASR(약 55%)에 있었고, 정확도 SOTA보다는 프라이버시 로컬성, 근거-시간 정렬형 설명 가능성, 그리고 전이 취약성을 정직하게 드러내는 점에서 의미가 있다.



### Stable-Shift: Biologically Structured Prediction of Transcriptional Responses to Unseen Gene Perturbations (https://arxiv.org/abs/2606.24940)
- **Prior Approaches**: 기존 Perturb-seq 예측 연구들은 크게 (1) 세포 상태/개입을 latent로 표현하는 scGen·CPA, (2) 유전자 상호작용 그래프를 전파하는 GEARS 계열, (3) 생성모형·세포 유형 일반화 등으로 확장해 왔습니다. 다만 학습에 없던 ‘미관측 유전자( unseen gene )’의 전사 반응을 끝까지 외삽하는 건 여전히 어렵고, feature-only 모델은 추론 시 분자 이웃을 활용하지 못하며 graph-only 모델은 단일 자원(STRING 등)의 잡음·불완전성에 취약합니다.

- **Core Contribution**: Stable-Shift는 미관측 유전자에 대한 반응을 추정하기 위해, 단일 세포 측정을 ‘개입 수준의 발현 변화(shift)’로 집계한 뒤 학습 개입에서만 low-rank response basis를 학습합니다. 이후 STRING/네트워크/대조군 통계/GO 같은 생물학적 컨텍스트로 각 유전자의 latent 좌표를 예측해, 그 basis에서 genome-wide 반응을 디코딩합니다(그래프 컨볼루션 인코더 사용). 핵심은 ‘훈련에서 제외된 유전자의 반응 표본을 쓰지 않으면서’ 컨텍스트 기반 latent-response transfer를 수행하도록 학습 타깃과 예측 경로를 구조화한 점입니다.

- **Technical Challenges**: 가장 큰 난제는 (i) 반응 벡터의 고차원·잡음, (ii) 개입들 사이의 지배적 global expression program 때문에 정보가 뭉치는 문제, (iii) 표현 특징만으로는 유전자 관계의 핵심이 보이지 않는다는 점입니다. Stable-Shift는 반응 차원을 low-rank로 압축해 학습 노이즈를 완화하고, 그래프 위 Node2Vec 구조 임베딩과 대조군 통계·토폴로지·GO 기능 임베딩을 결합해 unseen gene의 latent 좌표를 추론하도록 설계했습니다. 또한 basis 학습과 전처리 통계는 훈련 분할에만 고정하고, 학습 시에는 훈련 개입의 관측 프로그램에 대해서만 MSE를 최소화하도록 구성했습니다.

- **Empirical Impact**: K562 Perturb-seq 벤치마크에서 Stable-Shift는 cosine similarity 0.592로 GEARS(0.569)와 feature-only MLP(0.565)를 앞섰고, Spearman 상관 및 top-gene 정밀도에서도 더 좋은 성향을 보였습니다. 또한 5개의 unseen-gene split 평균 cosine similarity가 0.589±0.008로, CPA(0.566±0.010)와 random forests(0.555±0.012)보다 우수했으며 그래프-aware 분할·residualized·gene-space 재구성 등 스트레스 테스트에서도 동일한 우선순위가 관찰됐습니다. 다만 gene-space 재구성 후의 정확도 하락이 크고, 그래프 이웃이 희소할수록 성능이 떨어져(실패 경계 확인) 현 결론의 적용 범위에는 제약이 남아 있습니다.



### Privacy-preserving federated tensor decomposition of single-cell immune data: recovering multicellular programs across institutions (https://arxiv.org/abs/2606.24938)
- **Prior Approaches**: scITD 등 기존 연구는 donor×cell-type×gene 텐서 분해로 여러 세포유형에 걸친 multicellular programs(면역 상태의 공통 축)을 복원하지만, 모든 데이터가 한 곳에 모이는 centralized 전제가 있습니다. 또한 기존 federated single-cell 방법들은 배치 보정, 분류, 사전학습 같은 과업은 다루지만, 텐서 분해 기반의 multicellular program 복원을 federated로 수행하는 대응은 없었습니다.

- **Core Contribution**: 이 논문은 환자 세포를 공유하지 않고도 multicellular programs를 복원하는 federated estimator를 제안합니다. 각 기관은 로컬에서 program subspace를 계산하고, 조정자(coordinator)는 federated global-mean centering을 적용한 stacked SVD로 부분공간을 병합하며, 이는 중앙집중 분해와 truncation 범위 내에서 동등함을 보입니다.

- **Technical Challenges**: 핵심 난제는 두 가지입니다: (1) 기관마다 라벨/분포가 달라지는 site-label confounding에 강건한 병합, (2) 한 사이트가 모든 cell type을 관측하지 않는 vertical-FL 설정에서 교차-패널 프로그램을 잃지 않는 복원입니다. 저자들은 global-mean centering으로 병합의 견고성을 확보하고, donor-mode 공유를 이용한 stacked Gram(또는 complete-panel stacked SVD)로 고정된 feature 공간이 비어도(공통 cell type이 없어서도) cross-panel 프로그램을 정확히 복원하도록 설계했습니다.

- **Empirical Impact**: 261 donor의 SLE 시스템에서 ISG(인터페론 자극 유전자) 프로그램을 centralised와 거의 동일한 품질로 복원했으며, case–control 분리는 AUC 0.958로 보고됐습니다. 실제로 COVID-19 3개 기관 간 federated 환경에서도 subspace correlation 0.989를 달성하고, interstitial-lung-disease에서는 multicellular 프로그램이 최적 단일 cell type보다 질병 예측에서 우수(AUC 0.96 vs 0.91)함을 보여 federation에서도 이 이점이 유지됨을 확인했습니다. 한편 membership inference 평가에서 secure aggregation은 공격 AUC를 0.91→0.61로 낮추며, 암호 적용 수준에 따라 privacy-유틸리티 균형이 달라짐을 정리했습니다.



### Self-Modulating Quantum Fast-Weight Programmers for Efficient Adaptive Sequential Learning (https://arxiv.org/abs/2606.24933)
- **Prior Approaches**: 양자 시계열/순차 학습에서는 QLSTM처럼 파라미터화된 양자회로(VQC)를 게이트형 반복 구조에 넣는 방식이 널리 쓰인다. 다만 이런 비선형 게이트 반복은 시간축을 따라 순차적인 BPTT 비용이 커지고, 시퀀스에서의 새 정보 주입과 이전 메모리의 균형이 고정 구조에 의존한다.
또 다른 축인 Quantum Fast Weight Programmer(QFWP)는 fast weight를 누적 업데이트로 갱신해 BPTT 같은 깊은 양자 그래디언트를 피하지만, 누적만으로는 입력 의존적 forgetting/재가중이 없어 시퀀스 길이·모델 크기 변화에 따라 학습 안정성이 흔들릴 수 있다.

- **Core Contribution**: 본 논문은 Self-Modulating Quantum Fast Weight Programmers(Self-Modulating QFWP)를 제안해, 새로 생성되는 fast-weight 업데이트뿐 아니라 과거 fast-weight 메모리까지 입력에 따라 적응적으로 조절한다. 이를 통해 “새 정보 주입”과 “기억 보존”의 균형을 self-modulation으로 더 유연하게 만들며, 기존 QFWP의 누적 불안정성을 겨냥한다.
또한 업데이트(쓰기)와 메모리(유지) 중 무엇이 더 큰 이득을 주는지 ablation을 통해 규명하고, 이 조절이 시간 정보 전달(temporal information propagation)을 어떻게 개선하는지 이론적 직관을 함께 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 입력에 따라 fast weight를 조절하되, recurrence를 선형·병렬화 가능 형태로 유지해 학습의 양자 단계 비용과 불안정성을 동시에 줄이는 것이다. 논문은 fast-weight 상태의 “new modulation”과 “old modulation”을 각각 입력 기반 모듈레이션 행렬로 설계해, 원래의 QFWP 누적 구조에 쓰기/유지 제어를 결합한다.
또한 self-modulation 계수가 gate처럼 임의로 제한되지 않으므로, 모듈레이션이 기억 보존과 정보 주입을 적절히 타협하도록 수치적 안정성과 temporal 전파 관점의 균형을 이론·실험으로 설명한다.

- **Empirical Impact**: 수치 실험에서는 bessel_j2, damped_shm, delayed quantum control, NARMA-5/10 등 여러 시계열 벤치마크에서 Self-Modulating QFWP가 QFWP(standard) 대비 수렴 안정성과 예측 성능이 전반적으로 개선됨을 보인다. 특히 시퀀스 길이가 길어질수록 gap이 커지며, 많은 설정에서 Self-Modulating QFWP ≈ Only-Old Modulated >> Only-New Modulated >> standard QFWP 순으로 일관된 순위를 확인한다.
결과적으로 old(과거) 상태 기반 조절이 성능 향상의 지배적 원인으로 나타나고, full self-modulation은 이 이점을 유지하면서도 전반적으로 더 견고한 학습을 제공한다. 이는 compact하고 효과적인 양자 시계열 학습 프레임워크로서 Self-Modulating QFWP의 실용적 잠재력을 시사한다.



### Recursive QLSTM with Dynamic Variational Quantum Circuit Adaptation (https://arxiv.org/abs/2606.24932)
- **Prior Approaches**: QLSTM은 LSTM의 게이팅 구조를 유지하되, 각 time step에서 동일한 quantum 모듈(고정된 VQC 파라미터)을 재사용하는 고정 재귀 구조에 주로 의존했다. 이 때문에 입력 시퀀스 길이가 달라질 때 시간 정보가 어떻게 더 잘 전파되도록 회로를 조직해야 하는지, 그리고 재귀 규칙 설계가 학습 성능에 미치는 영향이 충분히 해명되지 않았다. 또한 기존 QLSTM 변형들은 context에 따라 gate-wise 파라미터를 동적으로 생성/수정하는 방식이 상대적으로 제한적이었다.

- **Core Contribution**: 이 논문은 Recursive Quantum Long Short-Term Memory(Recursive QLSTM)를 제안하며, 메타 생성기인 MetaCore가 순환 컨텍스트로부터 VQC 파라미터를 동적으로 생성·적응시키도록 확장한다. 특히 metacore 기반 recursive construction을 통해 시간 정보 전파를 명시적 재귀 규칙으로 모델링하고, MetaCore 설계와 recursive rule을 한 프레임워크에서 비교 가능하게 만들었다. 수치 실험 결과를 바탕으로 최우수 아키텍처를 찾고, 그 재귀 구조가 temporal information propagation과 학습 성능을 어떻게 개선하는지 이론적 논의도 제공한다.

- **Technical Challenges**: 핵심 기술 난제는(1) QLSTM의 gate별 VQC 파라미터를 time step마다 효율적으로 생성하되(계산·구조 복잡도), (2) 재귀 규칙/MetaCore 선택에 따른 학습 안정성과 성능 변동을 통제하며, (3) 시퀀스 길이가 달라질 때도 유효한 temporal 전파를 보장하는 것이다. 이를 위해 논문은 base+delta, delta, meta-only 같은 recursive rule 조합과 Enc1NNGate, MLP-GELU, shared-encoder gate-wise, tensor-product 등 다양한 MetaCore 변형을 정의하고, base 파라미터와 동적 업데이트(또는 누적/증분)를 결합하는 형태로 구현한다. 또한 최적화 관점의 지표(AUC@20, t95)까지 도입해 “최종 성능”뿐 아니라 “빠른 수렴/오류 감소 속도”를 구조별로 정량 비교한다.

- **Empirical Impact**: Damped SHM, Bessel function J2, Delayed Quantum Control, NARMA-5, NARMA-10의 벤치마크에서 Recursive QLSTM 계열은 표준 QLSTM 대비 최종 test loss는 비슷하거나 더 낮으면서도, AUC@20과 t95 같은 최적화 지표에서 특히 유리한 경향을 보였다. 즉 재귀 구조가 초기~중기 학습에서 시계열 궤적을 더 빠르게 복원해, 다양한 시간 지평에서 더 이른 수렴을 달성했다. 또한 일부 delta 기반 설정은 장기 시퀀스에서 변동성이 커지는 반면 base+delta는 더 안정적이어서, 후보 선별 단계(phase 0)가 robust한 아키텍처를 찾는 데 중요하다는 실증적 근거를 제시한다.



### Error-Aware TF-IDF Retrieval-Augmented Generation for ASR Error Correction (https://arxiv.org/abs/2606.24915)
Comments:
          4 pages, 1 figure, 2 tables

- **Prior Approaches**: 기존 ASR-RAG 교정은 희귀 개체/도메인 용어에서 LLM이 환각을 일으키는 문제를 완화하려고 시도돼 왔습니다. 하지만 표준 TF-IDF 기반 검색은 음운 오인(phonetic misrecognition)을 일으킨 토큰을 동일하게 취급해 교정 근거를 충분히 끌어오지 못하고, cross-modal 임베딩 등 복잡한 검색은 높은 지연(latency)과 계산 부담이 큽니다. 또한 엔터티 벡터베이스·Knowledge Graph처럼 무거운 구성은 정렬된 대규모 데이터·고자원 가정이 필요하다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 저자원 언어 페르시아어에서 LLM 기반 ASR 교정 시의 phonetic hallucination과 loop hallucination을 “오류 인식형(lexical error-aware)”으로 직접 겨냥하는 효율적 RAG 프레임워크를 제안합니다. 핵심은 Symmetric Text Normalization으로 KB와 질의 전처리를 동일하게 맞추고, Error-Aware TF-IDF로 과거에 자주 환각되는 토큰에 더 큰 가중치를 부여해 LLM이 교정해야 할 근거 문서를 우선적으로 검색하게 만드는 것입니다. N-best reranking이나 음향 latent/지식그래프에 대한 과도한 결합 없이, 1-best ASR 가설과 희소 행렬 연산만으로 동작하도록 설계됐습니다.

- **Technical Challenges**: 기여를 구현하는 가장 큰 기술적 난제는 (1) 형태/표기 차이로 인한 토큰 미스매치가 검색 공간을 왜곡하고, (2) 루프 환각이 TF-IDF의 term frequency를 망가뜨리며, (3) 표준 TF-IDF처럼 모든 토큰을 동일 가중으로 두면 “오인 토큰이 포함된 교정 문맥”이 점수에서 밀린다는 점입니다. 저자는 ZWNJ·공백·숫자 표기 등을 KB와 질의에 대칭 적용해 벡터공간 일관성을 확보하고, 동일 토큰이 2회 초과 반복되는 구간은 루프 환각으로 보고 잘라내 term frequency 스큐를 줄였습니다. 이어 Error propensity를 기반으로 희소 대각 penalty matrix를 만들어 TF-IDF 행렬에 곱하는 방식으로 계산 지연을 거의 추가하지 않으면서도 오류 토큰을 의도적으로 끌어올렸습니다.

- **Empirical Impact**: FLEURS 페르시아 subset에서 Whisper large-v3-turbo와 Gemini 2.0 Flash-Lite를 사용해 평가했으며, Error-Aware Hit Rate(EA-HR)가 표준 TF-IDF 53.7%에서 제안 방법 90.9%로 크게 상승했습니다. 종단(end-to-end) RAG-ASR 교정에서는 기준 ASR WER 23.06%에서 표준 TF-IDF 21.95%를 거쳐 제안 Error-Aware TF-IDF가 18.83%로 추가 개선을 보였습니다. 특히 해당 성능 향상을 위해 별도의 복잡한 cross-modal 임베딩을 쓰지 않고 희소 행렬 곱 수준의 근접-제로 inference latency만 요구해, 저자원 환경에서도 실용적인 정확도/지연 균형을 제시했다는 점에서 의미가 큽니다.



### Velocity Prediction in Automatic Guitar Transcription (https://arxiv.org/abs/2606.24912)
Comments:
          Accepted for publication at the 34th European Signal Processing Conference (EUSIPCO)

- **Prior Approaches**: 기존 Automatic Guitar Transcription(AGT)은 피치, onset, offset 위주로 발전했지만, velocity는 라벨 데이터 부재와 기타에서의 velocity 정의 모호성 때문에 거의 다뤄지지 않았다. 피아노는 Disklavier로 측정 가능한 velocity가 있지만, 기타는 가상 악기 구현마다 velocity-강도/음색 매핑이 달라 그대로 학습·평가하기가 어렵다. DSP 기반 velocity 예측 시도도 있으나, 최근 연구는 주로 딥러닝으로 pitch 중심 전사에 집중돼 왔다.

- **Core Contribution**: 이 논문은 기타에서 velocity를 예측하기 위한 학습 파이프라인을 제안한다. 가상 기타 가상악기가 내부적으로 정의한 velocity 곡선을 기준으로 FrançoisLeduc의 정렬 MIDI를 이용해 velocity 라벨이 있는 합성 학습데이터를 만들고, 먼저 합성에서 velocity 모듈을 학습한 뒤 그 가중치를 실데이터 전사 모델에 전이해 end-to-end 수준의 전사를 유지하면서 velocity까지 예측하도록 한다. 결과적으로 “합성으로 velocity 곡선을 먼저 잡고, 실음으로 전사 품질을 확보”하는 구조를 제시한다.

- **Technical Challenges**: 핵심 난제는 기타 velocity의 ‘정답’이 실측으로는 확보되지 않는다는 점이며, 그 때문에 평가도 결국 가상악기 구현과의 정합성에 의존하게 된다. 논문은 이를 피하기 위해 song/timbre 분할로 두 갈래 평가를 설계해(악곡 분리 vs 악기 음색 분리) 합성에서 학습한 velocity가 새로운 조건에서도 재현되는지 보려 했다. 또한 실데이터(GAPS, GOAT)는 velocity 라벨이 상수라 velocity 손실항을 제거하고, 1단계에서 학습한 velocity 모듈 가중치를 freeze해 실데이터 학습 과정에서 velocity 예측이 망가지지 않게 했다.

- **Empirical Impact**: 합성 테스트에서는 velocity 예측 전용 학습이 baseline(velocity pretrained 없음)보다 mean absolute error가 유의미하게 개선됨을 보였다. 다만 실전 전사 성능(F1 등)은 velocity 가중치 전이가 대체로 작은 폭의 개선에 그쳤고, 통계적으로 유의미한 경우도 GuitarSet에서 일부 지표에 한정됐다(대략 0.1% 수준). 그럼에도 음색·조건 일반화가 어느 정도 관측돼, “기타에서도 상대적 intensity를 의미 있게 추정하는 velocity 전사”의 가능성을 실증했다.



### Attractive and Repulsive Pattern Control in Sequence Generation (https://arxiv.org/abs/2606.24911)
Comments:
          16 pages, 6 figures

- **Prior Approaches**: variable-order Markov 모델과 BP-Regular 생성은 자동자(Regular constraint)로 합법적 이어짐을 정확히 샘플링하지만, 긴 연속이 반복되는 고차 suffix/모티프를 “tunnel”처럼 강화할 수 있다는 한계가 있다. 최대 차수 회피(max-order)처럼 hard ban은 반복을 끊는 대신 문법적/구조적 쓸모까지 약화시킬 수 있다. 또 인식기(regular recognizer)로 패턴 선호를 부여하더라도, 부호 있는 에너지로 ‘억제-유인’을 동시에 정밀 제어하는 방식은 제한적이었다.

- **Core Contribution**: 이 논문은 signed pattern control을 제안해, BP-Regular 인터페이스 안에서 특정 패턴 패밀리를 -β(페널티) 또는 +β(리워드)로 부호까지 포함해 소프트하게 제어한다. 가중 인식기(recognizer)가 패턴 활성 R(x)를 계산하고, 샘플러는 P_beta(x) ∝ P_0(x) exp(beta R(x))에 정확히 비례하도록 belief propagation으로 샘플링한다. 따라서 같은 메커니즘으로 ‘자기복제 붕괴 방지’와 ‘어트랙터(끌림) 분지 탐사’가 분리된 실험 도구로 제공된다.

- **Technical Challenges**: 핵심 기술 난점은 variable-order Markov의 백오프/컨텍스트 상태 위에, 패턴 인식기의 가중치가 들어가도 BP가 여전히 정확(정규화 질량과 조건부 말뭉치가 정확히 계산)해야 한다는 점이다. 이를 위해 sparse context-state 구성(Pachet)을 유지한 채, 선택된 recurrence automaton(가중 유한상태 인식기)과의 곱(product)을 BP 그래프에 포함시켜 exp[beta R]가 항상 양수인 재가중 분포에서 sum-product BP가 성립하도록 설계했다. 추가로, homeostatic 설정에서는 생성 이력에서 과활성(overactive)된 패턴을 온라인으로 발굴해 다중 차수(예: 2/3/4/6/8) 반복을 시간창+평생 항으로 누적한 뒤, 다음 horizon에 대해 즉시 soft 에너지를 갱신한다.

- **Empirical Impact**: 6개의 duration-bearing 단성(예: Bach, Telemann) 소스에서 음의 가지(β<0)는 생성된 8-gram self-reuse를 줄이고, effective 8-gram 수를 늘리며, 학습 기반 4-gram 컨텍스트 커버리지를 대체로 높이면서도 하위 차수 지지는 상당 부분 보존했다. 또한 Weimar Jazz Database 5개 솔로(Baroque 외 조건)에서 pitch-only replication 실험도 동일한 anti-reuse 시그니처를 보였다(예: self8 및 eff8 변화로 대표됨). 양의 가지(β>0)는 같은 인식기(recognizer)를 이용해 어트랙터 베이슨·위상 전이·히스테리시스 같은 ‘기저 VO 모델의 상태공간’에 대한 실험적 프로빙을 가능하게 함을 보여, 생성 제어/평가 관점에서 의미 있는 확장으로 평가된다.



### End-to-End Voice Intent Recognition for Spontaneous Human-Drone Interaction with Naive Users (https://arxiv.org/abs/2606.24910)
Comments:
          This paper has been accepted for publication at the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026), August 24-28, 2026, Kitakyushu, Japan

- **Prior Approaches**: 기존 드론 음성 제어는 고정된 명령어 집합에 크게 의존해, 초보 사용자의 자발적이고 어눌한 발화를 제대로 처리하지 못하는 한계가 있었다. 또 음성 인식(ASR)을 거친 후 명령을 파이프라인으로 해석하는 방식은 지연이 커져 실시간성에 불리했다.

- **Core Contribution**: 이 논문은 전사를 요구하지 않는 end-to-end Spoken Language Understanding 구조로, 드론과 사람의 실시간 상호작용을 목표로 한다. 냉동된 self-supervised acoustic encoder에 경량 LSTM 분류 헤드를 얹고, cross-modal knowledge distillation으로 음향 표현을 텍스트 교사(semantic embeddings)와 정렬해 의미 기반 견고성을 높인다.

- **Technical Challenges**: 자발 음성은 발화가 불연속적이고 표현이 불규칙해, 기존 음향-의미 대응을 안정적으로 학습하기 어렵다. 연구진은 분류 헤드만 가볍게 학습하면서도 distillation 손실로 의미 공간과의 정렬을 강제해, 추론 시 transcription 없이도 의미를 끌어오는 문제를 해결했다.

- **Empirical Impact**: VoiceStick(프랑스어 자발 음성 코퍼스, 29개 비전문 dyad 기반)에서 단순 음성 명령은 7 ms 지연에서 93% 정확도를 달성하며, cascade 기준 79%(202 ms)를 능가해 29배 속도 개선을 보였다. 자발 발화 전체 평가에서도 82% 정확도를 기록했고, crossmodal distillation은 모든 설정에서 강건성을 일관되게 향상시켜 의미 기반 저지연 음성 인터페이스의 실용성을 입증했다.



### Failure Modes of Large Language Models on Research-Level Mathematics: A Taxonomy and an Empirical Characterisation (https://arxiv.org/abs/2606.24902)
- **Prior Approaches**: 최근 수학 벤치마크들은 정답 문자열이나 부분 단계 평가로 LLM을 점검해 왔지만, 연구용 정밀 증명은 자동 채점이 어려워 실패 양상을 세밀하게 분해하긴 힘들다. 또한 기존의 hallucination 연구와 Process reward, debate, RAG 같은 대응은 주로 ‘사실 인용’ 또는 ‘검증 가능한 근거’ 문제에 초점이 맞춰져 왔다. First Proof 유형은 유창한 LaTeX 증명이지만 내용은 틀리는 형태라, 표면적 거짓 탐지로는 놓치기 쉽다는 한계가 있다.

- **Core Contribution**: 이 논문은 First Proof의 per-question post-mortem을 바탕으로 수학 증명 실패를 4가지 모드로 분류한다(F1 citation fabrication, F2 premise smuggling, F3 silent problem reformulation, F4 local-to-global compatibility gap). 특히 F2가 ‘근거 없이 주장되는 핵심 전제(load-bearing claim)’를 증명 내부에 넣는 방식이라 RAG의 전형적 기대(잘못된 인용을 덜게 한다)로는 해결되지 않는다고 지적한다. 저자는 F2의 지배적 존재를 보이기 위해, 인용 검증과 별개로 전제 스무글링을 정면 측정하는 도구를 제안한다.

- **Technical Challenges**: 기여를 실현하기 위한 핵심 난관은 F2가 인용 확인으로는 드러나지 않는다는 점이다. 저자는 F1을 보려는 인용 계측기( arXiv 매칭 기반)와 별도의 전제-감사 도구를 설계하는데, 전제-감사 도구는 ‘fundamental result/standard argument’ 같은 보편화 어휘를 regex로 후보 문장화한 뒤, LLM judge가 문장과 주변 맥락을 보고 “검증 없는 비자명 주장 + 인용/증명 부재” 여부를 판정한다. 다만 ‘named theorem을 가설 불일치 맥락에 적용’처럼 암묵적 형태는 어휘 패턴을 회피할 수 있어, 연구가 반복 가능한 형태로 완전 탐지되진 못하는 구조적 간극도 함께 드러낸다.

- **Empirical Impact**: Gemini 2.5 Flash로 First Proof의 일부 문항(Q1, Q2, Q5)에서 총 8개 one-shot proof를 감사한 결과, 최종 정답은 한 번도 맞지 않았다. 인용 계측기 관점에서는 확인된 fabricated citation이 없었지만, 매 증명마다 적어도 하나의 ‘부하 전제 스무글링’이 존재했고, 전제-감사 도구의 정밀도는 100%(judge-confirmed 5/5가 모두 진짜)였으며 proof-level recall은 50%(4/8) 수준이었다. 결론적으로 RAG는 F1에만 주로 대응하며, 장기 목표는 ‘생성 후 검출’이 아니라 추론 단계에서 F2 같은 실패 모드를 애초에 막는 inference-time 파이프라인 구축으로 옮겨야 한다는 메시지를 준다.



### LLM Evolution as an Industry-Scale Ecosystem: A Lifecycle Perspective on Continual Learning (https://arxiv.org/abs/2606.24901)
- **Prior Approaches**: 기존 continual learning은 안정성-가소성 trade-off를 중심으로, 리플레이(replay), 정규화, 구조/용량 분리, 최적화 기하(예: gradient projection) 같은 방식으로 단일 모델의 망각을 줄이는 데 집중해 왔습니다. LLM 맥락에서는 replay 비율·샘플 선택, EWC류 가중치 제약, distillation 기반 표현 정합, LoRA/adapter 격리, MoE 확장 등이 대표적 접근입니다. 하지만 산업 환경의 ICL은 단일 체크포인트가 아니라 버전 생태계에서 지속 업데이트·배포까지 포함하므로, 기존 연구의 단일-모델 추상화만으로는 한계가 큽니다.

- **Core Contribution**: 이 논문은 LLM의 Industrial Continual Learning(산업적 지속학습)을 “업데이트-배포(update-and-release)”의 closed-loop 문제로 재정의하고, versioned ecosystem에서 업그레이드가 계층적으로 전파되도록 설계 관점을 제시합니다. 업데이트는 capability inheritance(역량 상속)와 버전/모델 패밀리 간 전이를 목표로 하며, 단순 성능 유지가 아니라 규정 준수·지연·비용·회귀 위험까지 성공 기준에 포함합니다. 이를 위해 알고리즘 카테고리 대신 라이프사이클 설계 원칙 5가지를 중심으로 기술 지형을 정리합니다.

- **Technical Challenges**: 핵심 난제는 세 가지로 요약됩니다: 반복 적응이 plasticity를 고갈시켜 장기 진화가 막히는 문제, foundation-model 업그레이드가 capability inheritance를 깨뜨리는 문제, 그리고 프라이버시·컴퓨트·비용·지연 제약 때문에 관측 가능한 학습 신호가 배포 목표와 어긋나는 문제입니다. 논문은 해결을 위해 (1) 미래 학습 여력 plasticity headroom 보존, (2) 업그레이드를 capability transfer로 취급, (3) 신뢰 가능한 continual reinforcement learning(강화학습)과 drift 안전장치, (4) 학습 레시피의 self-optimizing, (5) 릴리스 기준의 accountability(감사 가능성) 기반 표준화를 제안합니다. 또한 모니터링→전이/학습 선택→비교 가능한 평가→회귀 위험 게이트와 롤백 준비로 이어지는 루프를 구체화합니다.

- **Empirical Impact**: 논문은 5개 도메인에서 20개 이상 산업용 LLM을 증거 기반 관점으로 분석해, 5가지 설계 원칙과 구성요소의 “성숙도”가 업계에서도 편차가 크고 재사용 가능한 업데이트 워크플로로 패키징된 사례가 드물다고 지적합니다. 동시에 연구와 배포 사이의 구조적 격차(바로 재사용 가능한 요소, 학계 성과는 있으나 시스템 수준 근거가 부족한 요소, 산업에서 필수지만 과소평가된 역량)를 정리해 향후 연구 로드맵을 제시합니다. 결론적으로, 단기 벤치마크 성능 경쟁을 넘어 장기적으로 신뢰 가능한 LLM 진화 체계를 구축하는 실무 청사진을 제공한다는 점에서 의미가 큽니다.



### On-Device Neural Architecture Search (https://arxiv.org/abs/2606.24900)
- **Prior Approaches**: 기존 edge/near-sensor computing의 딥러닝은 주로 추론(inference)이나 학습(training) 최적화에 집중했고, 최근에는 HW-NAS처럼 설계 단계에서 하드웨어 제약을 반영하는 흐름이 강화됐다. 그러나 대부분은 게이트웨이에서 센서 유형별 모델을 만들거나, 탐색을 위한 자원이 충분하다는 가정 하에 검색 공간이 제한되며, 배포 장치와 탐색 장치가 분리되는 경우가 많았다. 또한 시간 시계열(time series) 입력을 직접 겨냥한 경량 아키텍처 탐색 연구는 상대적으로 적었다.

- **Core Contribution**: 이 논문은 near-sensor computing에서 NAS의 검색 과정을 배포(deployment) 장치 자체에서 수행해, 센서로부터 들어오는 실시간 데이터에 맞는 tiny neural architecture를 찾는 적응형 패러다임을 제안한다. 특히 사용자 변경에 따라 생체 신호 변동이 생기는 HMI 환경에서, 유도된 데이터 수집 절차 후 아키텍처를 재설계해 개인 간 차이를 더 효과적으로 다루는 것을 목표로 한다. 동시에 지능형 고장진단에도 범용 IFDS 형태로 확장 가능함을 보여준다.

- **Technical Challenges**: 배포 장치에서 NAS를 돌리려면 학습 시 메모리 상한과 실시간 지연(latency) 같은 제약이 검색 단계부터 강하게 걸려, 탐색이 무거워지거나 유효한 후보를 찾기 어려워진다. 저자들은 경량(non-derivative) 탐색 전략을 최적화 문제에 맞게 재구성하고, 후보 네트워크 평가를 child process로 분리해 메모리 고갈로 OS가 중단(kill)하는 경우를 제약 위반 신호로 활용했다. 또한 평가 시 1회 추론 시간을 측정해 지연 상한을 함께 만족하는 후보만 채택하도록 설계해 실제 장치에서 동작 가능성을 확보했다.

- **Empirical Impact**: Raspberry Pi 4/3/Zero 2 W 등 3개 임베디드에서 ISL(근전도 sEMG 기반 이탈리아 수화)과 CWRU(베어링 고장진단) 두 벤치마크로 검증했으며, 양자화-aware training과 8-bit quantization을 사용했다. Raspberry Pi 4에서 ISL은 RAM 점유를 0.63배로 줄이면서 정확도는 5.96%p 더 높았고, CWRU에서도 RAM을 0.44배로 줄이면서 정확도는 0.2%p 상승해 기존 대비 개선을 보였다. 전반적으로 자원이 적은 장치일수록 RAM 제약 때문에 정확도는 떨어질 수 있으나, RPi4/3에서는 성능이 유지 또는 우세했으며 RPi Zero 2 W에서는 특히 ISL에서 제약이 매우 빡빡해 만족 후보가 부족했다. 다만 NAS 탐색 비용이 있어 실제 적용은 overnight 실행이 현실적일 수 있으며, 사용자별 데이터가 적다면 탐색 시간도 더 줄어들 여지가 있다고 결론을 제시한다.



### From Meta Idea to Advanced Mathematical Discovery -- Human-AI Co-Discovery of Sign-Embedding Quantum Algorithms (https://arxiv.org/abs/2606.24899)
Comments:
          35 pasges, 3 figures

- **Prior Approaches**: 기존 AI 수학 성과는 주로 이미 정해진 목표(가설 검증/반증, 알고리즘 최적화, 정해진 증명 탐색) 안에서 생성·탐색·증명 보조에 집중해 왔다. 반면 이 논문이 다루는 ‘문제 형성’ 단계는 중간 산출물이 평가 지표나 실행 결과로 바로 환산되지 않아 검증과 비교가 어렵다. 또한 양자 알고리즘 쪽에서도 연산(값/상태) 추정에 머무르거나, operator-output(블록 인코딩)까지 포함해 성능을 엄밀히 조율하는 연구는 상대적으로 제한적이었다.

- **Core Contribution**: 이 논문은 사람의 막연한 직관(부착적/점프형 함수에 유리한 rational approximation)이 양자 알고리즘의 sign-embedding 프레임워크로 수렴하는 과정을 사례 연구로 정리한다. 특히 AIM(에이전트형 AI-수학자 시스템)이 단순 증명기나 사후 보조가 아니라, 문제 형성에서의 route map 작성, 후보 정식화 비교, 연결(아이덴티티 확장) 탐색, 증명 골격·복잡도 초안 작성에 핵심적으로 관여했음을 보여준다. 최종 산출물은 매트릭스 방정식과 매트릭스 함수의 sign embedding 기반 operator-output quantum algorithm용 ‘감사 가능한(theorem-shaped)’ 정리 패밀리다.

- **Technical Challenges**: 핵심 난관은 sign 함수 같은 고난도 함수를 rational로 근사해 shifted inverse를 만들더라도, 그 결과가 양자용 블록 인코딩으로 실현 가능하고(정규화 β, 오류 ε), query complexity와 조건(예: FoV 간격, strip-resolvent 인증)이 충분히 강해야 한다는 점이다. 논문은 이 문제를 field-of-values 기반 해상도 인증서, resolvent/shifted-inverse의 조합 비용에 대한 복잡도 감사, 그리고 숨은 조건이 개입되는 경로(Cayley–trapezoidal)가 무효가 될 수 있음을 인간이 찾아 교체하는 방식으로 해결한다. 구체적으로 로그-신(log-sinc) 근사와 factorized/scaled multiplexing을 통해 Sylvester 모듈에서의 거친 μ^-2 스케일 query 루트를 μ^-1 스케일로 정밀화하고, normalization까지 구분해 최종 정리 타깃을 다듬었다.

- **Empirical Impact**: 실험적 수치 성능보다는, ‘사람-게이트형’ 공동발견 워크플로가 실제로 어떤 정리 패밀리를 만들어내는지에 대한 설계·검증 사례가 중심 성과다. AIM의 역할은 연결 그래프 확장과 증명/복잡도 초안 생성이었고, 최종 과학적 판단(가치 게이트, 숨은 가정 거부, 더 강한 profile 기반 정규화로의 샤프닝)은 인간이 수행했다. 저자들은 AI를 독립 theorem prover로만 쓰기보다, 문제 형성·연결 발견·도출·회의적 검토를 포함하는 인간 통제 루프에서 연구 파트너로 둘 때 가치가 커진다고 주장하며, route ledger·complexity audit checklist 같은 재사용 가능한 산출물도 제시한다.



### Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models (https://arxiv.org/abs/2606.24898)
- **Prior Approaches**: 루프드 language model은 Universal Transformers 등에서 알려진 것처럼 동일 블록을 반복해 test-time depth를 계산 예산처럼 조절하려는 흐름이 있다. 이때 흔히 적용되는 dense per-loop cross-entropy는 각 루프의 ‘조기 종료(exit) 인터페이스’는 학습시키지만, 반복 과정에서 실제로 전달·갱신되는 모든 은닉 변수까지 자동으로 제어하진 않는다는 한계가 제기된다. 특히 readout이 어떤 은닉 변수를 ‘보이게(visibility)’ 만들고 어떤 변수를 ‘숨기게(blind spot)’ 만드는지에 대한 분석은 부족했다.

- **Core Contribution**: 논문은 dense cross-entropy가 제어하는 것은 반복 상태의 전부가 아니라 ‘readout을 통해 loss가 볼 수 있는 변수’임을 readout blind spot 관점으로 정식화한다. hidden-state scale은 대표적인 실패 모드로, RMSNorm/LayerNorm 같은 scale-invariant readout에서는 즉시 cross-entropy가 radial(스케일) 방향에 둔감하지만 pre-norm residual recurrence는 그 스케일을 계속 전달·업데이트할 수 있어 visibility–activity mismatch가 발생한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘왜 per-loop CE가 있어도 recurrence 내부의 특정 자유도가 통제되지 않는가’를 손실-기하 관점에서 설명하고, 이를 실험적으로 분리해 증명하는 것이다. 저자들은 scale을 hidden vector의 radial/angle로 분해하고, normalized readout이 radial에 대한 직접적인 그라디언트를 거의 없애는 반면 pre-norm 루프는 radial residual update를 학습한다는 로컬 메커니즘을 제시한다. 또한 scale이 loss에 보이도록 raw readout, norm penalty, diagnostic을 넣거나, scale이 루프에서 사라지도록 inter-loop normalization/scale-removing recurrence를 설계해 원인을 통제했다.

- **Empirical Impact**: 44M과 129M 규모의 루프드 Transformer에서 per-loop + RMSNorm은 매 루프 loss를 받는데도 final-loop hidden-state norms가 수만~수십만 단위로 크게 드리프트했다. 반대로 raw readout, 명시적 norm penalty, 또는 scale-removing recurrence를 적용하면 norms를 수십 수준으로 억제하면서, 변수 깊이(variable-depth) 평가에서 scale-controlled 모델이 KK-invariant RMSNorm baseline과 비슷한 inference-depth 지점에서 더 낮은 perplexity를 달성해 compute–quality frontier를 이동시켰다. 즉, ‘조기 종료는 exit 학습(dense supervision)으로, recurrent scale 제어는 scale visibility 또는 recurrent에서의 scale 제거로’가 실무적 설계 규칙으로 제안된다.



### RWGBench: Evaluating Scholarly Positioning in Related Work Generation (https://arxiv.org/abs/2606.24894)
Comments:
          9 pages, code and data available at this https URL

- **Prior Approaches**: 기존 related work generation(RWG) 평가는 요약문/장문 생성에서 쓰던 지표를 그대로 가져와 주파수나 의미 유사도, 또는 reference와의 표면적 일치도를 중심으로 봤다. 이 방식은 그럴듯한 문장과 의미적 관련성은 잘 맞춰도, 인용 선택이 부적절하거나 핵심 선행연구가 빠지는 등 ‘학술적 포지셔닝’의 치명적 오류를 놓치기 쉽다. 또 많은 벤치마크가 문장 단위 인용 생성이나 survey형 문서 합성에 집중해 문서 수준의 인용 선택·배치·구조 진단이 제한적이었다.

- **Core Contribution**: 이 논문은 RWG를 ‘문장 생성’이 아니라 ‘인용 의사결정 기반의 scholarly positioning’ 문제로 재정의한다. 그 결과 인용 선택, 문맥 적절성, 조직/구조, 담화 배치 같은 결정 요소를 직접 평가하도록 RWGBench를 제안한다. 표면 유사도 중심 평가와 달리, 모델이 어떤 인용을 왜(어떤 주장에) 넣었는지를 중심으로 품질을 측정한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 실제 논문 수준의 인용 결정은 retrieval 제약과 얽혀 있고, (2) 생성 텍스트 품질만으로는 인용의 정당성을 분리해 보기 어렵다는 점이다. 이를 위해 arXiv 기반 40,108편 타깃과 1,091,394편 retrieval 코퍼스를 구성하고, test set 100편은 gold 인용 커버리지를 97.7%까지 높여 ‘불완전 검색’ 상황에서도 진단이 되게 했다. 또한 NLI 기반으로 인용 문맥이 해당 선행연구 주장에 의해 지지되는지 점수화해 인용 선택의 국소 타당성을 검증하고, 담화 역할 분포(PSS)와 인용 밀도/분포 같은 구조 지표로 배치·프레이밍 실패를 분리했다.

- **Empirical Impact**: 실험은 네 가지 생성/검색 계열 기준으로 비교했으며, 표면 유사도는 개선돼도 인용 정확도는 크게 오르지 않는 등 기존 파이프라인의 병목이 retrieval임을 보여준다. 더 나아가 LLM-as-judge나 BERTScore 같은 텍스트 유사 지표는 유창하지만 인용이 빈약한 출력에 높은 점수를 주며, 인용 관련 지표(Citation appropriateness/coverage)는 전문가 판단과 더 잘 맞았다. 인간 평가에서도 Citation Relevance(61.8%), Coverage(예: TRT 69.1%, Recall 65.0%)는 비교적 높은 정렬을 보였고, Overall quality는 랜덤보다 낮은 수준으로 내려가 RWGBench의 ‘인용 중심’ 평가 필요성을 실증했다.



### ReviewGuard: Aligning LLM-Assisted Peer Review with Long-Term Scientific Impac (https://arxiv.org/abs/2606.24892)
- **Prior Approaches**: 기존 연구와 서비스들은 LLM을 심사위원처럼 ‘현재의 선호’를 모사하거나(SFT, instruction-tuned 리뷰어, 프롬프트/멀티에이전트) 현재 리뷰 점수와 유사한 평가를 반복하는 데 초점을 둔다. 그 결과 장기 과학적 가치 예측에는 한계가 있으며, 심사의 보수성·편향이 그대로 남아 ‘반려 후에도 영향이 큰 논문(rejection-resilience)’을 놓칠 수 있다. RLHF 계열도 주로 인간 선호 정렬에 머물러 있어, 장기 임팩트를 직접 목표로 삼는 방식은 거의 없었다.

- **Core Contribution**: ReviewGuard는 기존 LLM 심사 자동화의 목표를 ‘장기 인용 기반 과학적 임팩트 예측’으로 전환한 2단계 프레임워크다. 먼저 OpenReview 리뷰로 Qwen2-7B-Instruct를 LoRA 기반 supervised fine-tuning해 기본 심사 포맷과 평점 규칙을 학습하고, 이후 GRPO로 미래 인용에 맞춰 평점/리뷰를 재정렬한다. 핵심은 인간 리뷰를 흉내내는 정렬이 아니라, 장기 영향(미래 citations)과의 일치도를 직접 강화한다.

- **Technical Challenges**: 장기 임팩트를 미분 가능한 학습 신호로 만들기 위해, 모델이 낸 1–10 평점과 미래 인용을 정규화해 reward로 설계했으며(불일치 패널티, 고임팩트>150 보너스 포함) 극단값 영향도 줄이도록 캡(clipping)을 적용했다. 또한 Spearman 같은 순위 지표를 직접 최적화하기 어렵기 때문에, GRPO의 group-relative advantage로 상대적 학습 신호를 안정화하고 KL 페널티로 기준 정책(Stage 1 Expert) 대비 과최적화를 억제했다. 학습 중에는 critic 모델 없이 진행하며, 최종 추론 시 구조화된 리뷰와 영향-aligned 평점을 함께 제공하도록 구성했다.

- **Empirical Impact**: OpenReview(20,861편)와 Semantic Scholar 인용 데이터를 사용한 실험에서 ReviewGuard는 rejected-then-published 코호트의 미래 인용과 Spearman ρ=0.776의 상관을 보여, 인간 리뷰어(ρ=0.492)와 supervised Expert(ρ=0.681)를 모두 능가했다. 동일 의사결정 기준에서 고임팩트 반려 논문 플래그 비율은 인간 1.8% 대비 10.2%로 5.6배 개선되었고, 상위 인용 구간에서 평점 보정(calibration)도 더 잘 이뤄졌다. 임팩트 알라인 reward로 GRPO를 적용했을 때 성능 이득이 가장 크게 나타나, 편집자에게 ‘교체’가 아닌 ‘보완 신호’를 제공하는 도구로서 의미가 확인됐다.



### Type Checking Project Haystack Grids using JSON Schema and Pydantic (https://arxiv.org/abs/2606.24891)
- **Prior Approaches**: Project Haystack은 태그 기반 온톨로지로 산업에서 널리 쓰였지만, 태그 사용의 모호성·검증 부재·커스텀 포맷/도메인 전용 언어 의존으로 인해 모델의 신뢰성과 통합성이 떨어진다는 지적이 반복돼 왔습니다. 기존에는 Haystack/Haxall의 Fantom 생태계 밖에서 정의를 검증하거나 파이프라인에 곧바로 넣기 어려웠고, JSON(Hayson) 수준 검증만 제공되는 경우가 많아 온톨로지 정의 준수 여부까지는 확인하기 힘들었습니다. 관련 연구들도 Haystack의 범용성은 인정하면서, 형식적 의미 부족과 기계 판독성/검증 가능성의 한계를 핵심 문제로 꼽았습니다.

- **Core Contribution**: 이 논문은 Haystack 정의를 일반 목적 언어에서 다룰 수 있게 만드는 Python 기반 툴체인을 제안합니다. 핵심은 (1) Trio 형식(Haystack 정의 파일) 파서와 (2) 이를 기반으로 Pydantic 모델 및 JSON Schema를 자동 생성하는 코드 생성기이며, 그 결과 Python 내부에서는 static type checking과 구조적(structural) 검증을, Python 밖에서는 schema 기반 JSON 검증을 수행할 수 있게 됩니다. 생성된 모델·스키마·도구는 오픈소스로 공개해, Haxall/Fantom에 대한 사실상 기술 장벽을 낮추는 것을 목표로 합니다.

- **Technical Challenges**: 가장 큰 난제는 Trio가 YAML이지만 YAML 파서로 읽을 수 없고, Python에서 사용할 수 있는 공개 Trio 파서가 없다는 점이었습니다. 또한 Haystack의 normalization(DefCompiler/Haxall 경로)과 defx late binding 같은 빌드 과정을 그대로 따라가야 했으며, 버전에 따라 깨진 릴리스·정의 누락 같은 품질 문제까지 다뤄야 했습니다. 논문은 (a) Python lark 기반 Trio 파서, (b) normalization 재구현 또는 normalized JSON 기반 변환의 두 경로, (c) 계층 상속/필터링(tagOn 등) 후 base 타입에 매핑해 Pydantic 스키마와 JSON Schema를 생성하는 파이프라인으로 이를 해결합니다.

- **Empirical Impact**: 저자들은 생성된 검증 체계를 실제 Haystack 릴리스에 적용해 불일치와 누락된 정의를 찾아내며, 자동 검증의 실용성을 실증합니다. 이는 “열려 있으나 검증은 어렵다”는 기존 문제를 완화해 개발자가 IDE 자동완성과 연속 타입 체크, 구조 오류 조기 탐지를 수행할 수 있게 만듭니다. 더불어 Python 중심 생태계로의 통합 경로를 넓혀 상호운용성과 이전성(transferability) 개선에 기여할 것으로 기대됩니다.



### Small edits, large models: How Wikipedia advocacy shapes LLM values (https://arxiv.org/abs/2606.24890)
- **Prior Approaches**: 기존 연구는 모델 예측에 영향을 준 학습 데이터를 추적하는 data attribution(예: TrackStar)이나, 특정 문서 제거를 가정하는 MAGIC 같은 counterfactual 추정을 주로 다뤄왔다. 한편 위키피디아는 편집자 집단이 주제를 어떤 관점으로 제시하는지에 영향을 받지만, 이러한 ‘합법적 편집’이 실제로 언어모델의 주제 이해에 미치는 영향은 실증적으로 거의 연결되지 않았다. 또한 데이터 조작 연구는 주로 poisoning 같은 적대적 삽입에 집중해, 시민단체의 가치 기반 편집이 갖는 효과는 상대적으로 덜 조명됐다.

- **Core Contribution**: 이 논문은 동물복지 영역에서 Pro-Animal Wikipedians(PAW)라는 소규모 편집 캠페인이 위키피디아 서술을 어떻게 바꿨는지 추적하고, 그 변화가 LLM의 동물복지 관련 응답(예측 성능/연관성)에 선택적으로 전이되는지를 보여준다. TrackStar(gradient 기반 검색 귀속), MAGIC(훈련 과정 기반 counterfactual 영향 추정), 그리고 fine-tuning 기반 ablation을 같은 문제에 대해 독립적으로 적용해 ‘주제 특이성’을 교차 검증했다. 결과적으로 PAW의 편집은 동물복지 쿼리에 대해서는 영향이 뚜렷하지만, 같은 기업을 언급해도 무관한 일반 쿼리에서는 그 영향이 나타나지 않는다.

- **Technical Challenges**: 핵심 기술 난점은 (1) PAW 편집 텍스트를 통제 텍스트와 분리해 편집-내용 효과를 혼동 없이 측정하고, (2) 어떤 문서가 실제로 모델 예측에 기여했는지 causation에 가깝게 추적하며, (3) 훈련 순서 같은 seed 민감도까지 확보하는 것이다. 저자들은 within-article에서 동물복지(AW) 섹션과 비-AW 섹션을 짝짓는 방식으로 TrackStar 실험의 공변량을 줄였고, MAGIC/ablation에서는 대표적인 위키 텍스트 청크를 통제군으로 써 토픽 간 분리를 유지했다. 또한 Llama 3.1 8B와 Llama-3.2-1B에서 여러 랜덤 훈련-order seed(총 5개)를 돌려, seed에 따라 상위 문서 순위가 흔들릴 수는 있어도 AW vs 대조의 핵심 통계적 방향성은 일관되게 재현되도록 설계했다.

- **Empirical Impact**: 분석 결과, PAW 편집 섹션은 동물복지 쿼리에서 상위 귀속 문서로 과대표집되며(예: TrackStar top-10에서 52% vs 68% 수준의 유의한 비대칭), 일반 쿼리에서는 기준선 근처에서 관측됐다(p<0.0001 대 p=0.53). MAGIC의 counterfactual 영향에서도 모든 seed에서 동물복지 쿼리 상위-10의 영향 문서가 전부 PAW 편집(10/10, 5/5 seeds)인 반면, 일반 쿼리는 우연 수준(4–6/10)에 머물렀다. 마지막으로 PAW 텍스트로 fine-tuning한 모델은 동물복지 텍스트에서 perplexity를 12.4→8.4로 낮추고, 대조 텍스트에 대해서는 효과가 제한되어 텍스트 유형에 특이적인 학습 전이까지 확인했으며, 저자들은 이 메커니즘이 위키피디아가 가중되는 만큼 프론티어 모델에서도 이미 희석되기보다 ‘기반 신호’로 남아 있을 가능성이 크다고 주장한다.



New uploads on arXiv(cs.RO)

### Learning Action Priors for Cross-embodiment Robot Manipulation (https://arxiv.org/abs/2606.26095)
- **Prior Approaches**: 대부분의 Vision-Language-Action(VLA) 모델은 Vision-Language Model(VLM) 백본에 행동(action) 모듈을 붙이고 정책 전체를 imitation learning으로 end-to-end로 학습한다. 이때 VLM은 시각·언어의 강한 선행지식을 가져오지만, 행동 모듈은 물리적 모션의 시간 동역학을 거의 zero에서 배워야 하는 불균형이 생긴다. 특히 cross-embodiment에서는 로봇마다 action space와 모션 분포가 달라 초기 학습이 불안정해지고 수렴이 느려진다.

- **Core Contribution**: 이 논문은 cross-modal VLA 정렬 전에 행동 모듈에 motion prior를 먼저 학습시키는 2-stage 프레임워크를 제안한다. Stage 1에서 시각·언어 없이 행동 궤적만으로 행동 모듈이 시간적 모션 구조를 학습하고, Stage 2에서는 이를 VLA 학습에 전이해 시각·언어 정렬을 더 안정적으로 시작하게 만든다. 또한 학습된 encoder를 history compressor로 재사용해 긴 시간 문맥을 적은 비용으로 토큰 1개로 요약한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘행동 모듈이 아직 정렬되지 않은 상태’에서 VLA가 동시에 action 분포 학습과 cross-modal alignment를 수행하며 생기는 그라디언트 불안정과 최적화 병목을 줄이는 것이다. 저자들은 Stage 1에서 flow matching 기반 encoder-decoder 행동 모듈을 action-only로 학습해 모션의 구조적 표현을 만든 뒤, Stage 2에서는 decoder 재사용과 early-stage latent distillation으로 VLM 예측 임베딩을 그 모션 잠재공간에 초기에 고정한다. 이후 distillation 제약을 단계적으로 완화해 최종 성능은 end-to-end 정련으로 끌어올린다.

- **Empirical Impact**: LIBERO와 RoboCasa 시뮬레이션 및 실세계 Franka에서 총 13개 cross-embodiment 태스크를 실험해, action prior 없는 VLA 대비 더 빠른 수렴과 더 높은 성공률을 보였다고 보고한다. 특히 데이터가 적은 long-tail 실세계 태스크에서 성능 향상이 두드러지며, history 압축 토큰은 장기 과제에서 시간 수용범위를 늘려 추가 이점을 준다. 또한 Stage 1에서 action-only 데이터를 늘릴수록 더 일반화되는 motion prior가 형성되어 downstream VLA 성능이 직접 개선되는 scaling 경향도 관찰된다.



### ForceBand: Learning Forceful Manipulation with sEMG (https://arxiv.org/abs/2606.26093)
- **Prior Approaches**: 기존 로봇 학습의 인간 시연 데이터는 모션·외형(모션캡처, 인터넷 영상 등)을 잘 담지만, 힘 제어에 핵심인 접촉력(contact force)은 빠지는 경우가 많았다. 비전 기반은 접촉을 추정하려 해도 가림(occlusion)과 모호성 때문에 힘의 크기를 확정하기 어렵고, 촉각 글러브는 신체 계측이 필요해 일상적 대규모 시연 수집이 부담된다.

- **Core Contribution**: ForceBand는 저가형 손목 sEMG 웨어러블(ForceBand: wrist-worn sEMG)로 근육 활동을 접촉력으로 바꿔 ‘힘이 풍부한’ 시연을 만든다. EMG2Force(EMG2Force: sEMG→손가락 힘 예측 모델)는 손목 sEMG와 IMU로 손가락별 force trace를 예측하고, 짧은 사용자 캘리브레이션 후에는 ForceBand와 영상만으로 시연을 수집·라벨링해 로봇 학습에 바로 쓸 수 있게 한다.

- **Technical Challenges**: 핵심 난제는 (1) 손목 sEMG가 잡음이 많고 사용자·전극 배치에 따라 힘과의 관계가 달라진다는 점과 (2) 접촉력 자체를 시연에서 안정적으로 재구성해야 한다는 점이다. 논문은 근육별(anatomically guided) 전극 배치를 포함한 ForceBand 하드웨어와, spectrogram-augmented transformer 기반 EMG2Force로 sEMG+IMU→손가락 힘을 학습하고, 사용자별 짧은 힘 센서 캘리브레이션으로 모델을 개인화해 성능을 끌어올린다.

- **Empirical Impact**: 실험에서 근육 인지 전극 배치는 균일 8채널 배치 대비 힘 예측 오차를 약 18% 줄였고, EMG2Force는 비전 기반 기준선 대비 힘 예측 오차를 50% 이상 낮췄다. UR-5 실환경 pick/squeeze/place 과제에서 ForceBand는 서로 다른 물체(형상·크기·무게)에도 객체별 힘 프로파일을 요구하는 시나리오에서 87% 성공률을 달성했으며, motion-only/그리퍼-only로는 재현하기 어려운 힘 기반 조작을 OOD 물체까지 포함해 전달했다.



### Deep Reinforcement Learning-Enhanced Event-Triggered Data-Driven Predictive Control for a 3D Cable-Driven Soft Robotic Arm (https://arxiv.org/abs/2606.26048)
- **Prior Approaches**: 소프트 로봇은 강한 비선형·시간변화 동역학 때문에 정밀한 모델링과 실시간 제어가 어렵다. 기존 물리 기반(FEM/연속체역학)은 고차 PDE로 인해 계산 부담이 커 임베디드 실시간 제어에 불리하며, 이를 줄이기 위해 DeePC 같은 데이터 기반 예측제어가 주목받았다.

- **Core Contribution**: 이 논문은 receding-horizon DeePC의 반복 최적화(매 샘플 QP 풀이) 비용을 줄이기 위해, RL이 “최적화 호출 시점”을 결정하는 adaptive reinforcement-learning-based event-triggered DeePC(RL-ET-DeePC)를 제안한다. RL 에이전트는 현재 상태 기반으로 DeePC 옵티마이저를 실행할지(트리거) 아니면 버퍼에 저장된 제어 입력을 재사용할지를 선택해, 정확도-계산비의 균형을 상태 의존적으로 학습한다.

- **Technical Challenges**: 핵심 난제는 트리거 조건을 고정 임계값으로 두면(heuristic threshold) 소프트 로봇의 비선형·시간변화 특성 때문에 보수적이면 절감이 작고, 완화하면 추종 성능이 나빠진다는 점이다. 저자들은 SVD 기반으로 Hankel 행렬 차원을 줄여 SVD-DeePC를 경량화하고, 정규화 항/슬랙으로 잡음·모델 불일치를 다루면서, RL 정책 + 감독용 override 레이어로 최대 오픈루프 구간을 제한해 안정적인 갱신을 확보한다.

- **Empirical Impact**: 시뮬레이션과 3D cable-driven 소프트 로봇 팔 하드웨어 실험에서 RL-ET-DeePC는 주기적(periodic) DeePC 대비 최적화 호출 빈도를 최대 66%까지 줄이면서 추종 정확도는 비슷한 수준을 보였다. 특히 PPO 기반 정책을 대표로 보면, DQN/A2C/PPO 모두 트리거율 감소가 재현성 있게 나타났고, 하드웨어에서는 zero-shot transfer로 최적화 빈도 34% 절감을 달성하며 정적 임계값 기반 event-triggered 대비 성능 변동이 더 일관적이었다.



### Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations (https://arxiv.org/abs/2606.26047)
- **Prior Approaches**: 기존 비전 기반 군중 내 로봇 내비게이션은 사람이 2D 좌표/속도 점처럼 단순화되거나, 환경은 2D occupancy map이나 단일 라이다 스캔으로 축약돼 이미지의 풍부한 시각 단서를 놓치는 경우가 많았다. DRL(딥 강화학습)은 장기 보상 최적화로 강점을 보였지만, 성능은 결국 state embedding 설계에 크게 좌우되며 다양한 실제 장면으로의 일반화에 한계가 있었다. 또한 인간 의도는 궤적 중심으로 추정돼 고개·어깨·자세처럼 의도를 드러내는 미묘한 행동을 충분히 반영하지 못했다.

- **Core Contribution**: 논문은 iCrowdNav를 제안하며, egocentric 비전으로부터 의도(behavioral intention)를 반영하는 intention-aware scene representations를 학습하는 방법을 제시한다. 표준 BEV(다중 시점 통합) 점유 표현에 더해 사람 3D pose로부터 보행 의도 정보를 추출하고, 이를 DRL 정책 학습을 위한 컴팩트한 상태 임베딩에 통합한다. 핵심 모듈은 spatio-temporal encoder로 장면 점유의 시공간 맥락을 뽑고, Intent-Interact Former(I2Former)로 사람 자세에서 의도를 추론하며 로봇-인간 상호작용까지 모델링하는 점이다.

- **Technical Challenges**: 가장 큰 난제는 부분관측과 고차원 비전 입력 속에서, 점유 정보만으로는 부족한 ‘반응적 인간 행동’을 안정적으로 학습 가능한 표현으로 바꾸는 것이다. 저자들은 RGB-D 다중 시점을 BEV로 정규화해 공간·의미 일관성을 확보하고, temporal 윈도우 동안 ego-motion 보정 후 3D conv 기반 시공간 인코더로 점유의 변화를 인코딩한다. 동시에 YOLO로 2D 포즈를 추정한 뒤 depth로 3D로 승격·좌표 변환하고, Transformer 기반 IntentFormer(자세 내 관절 관계)와 InteractFormer(로봇 상태를 query로 한 cross-attention)로 의도 및 상호작용 특징을 통합해 DRL에 입력한다.

- **Empirical Impact**: 시뮬레이션 실험에서 iCrowdNav는 I2Former 또는 BEV 구성요소를 제거한 ablation보다 전반적으로 성공률(SR), 효율, 그리고 사적 영역 침범(TPZ)에서 우수한 성능을 보이며, 특히 좁고 밀집된 환경에서 격차가 커졌다. 다른 기준선(예: DRL-VO, SARL*-OM, ViNT, DWA)과 비교해도 SR·시간·경로 길이·TPZ 전반에서 가장 좋은 결과를 유지하며, 정성 평가에서도 조기 회피와 사회적 거리 준수가 두드러졌다. 더 나아가 실제 로봇 탑재 환경에서 15 Hz 추론 속도로 체육관/지하철/쇼핑몰 등 복잡 장면을 시연하며 시각 기반 군중 내비게이션의 실용성과 강건성을 확인했다.



### RoboAtlas: Contextual Active SLAM (https://arxiv.org/abs/2606.26046)
Comments:
          Alexander Schperberg and Shivam K. Panda made equal contribution

- **Prior Approaches**: 기존 Active SLAM은 주로 occupancy grid 같은 저해상도 기하 표현과 휴리스틱 기반 정보이득/제약 최적화에 의존해 왔습니다. 그 결과 semantic search나 특정 객체/장소 중심 탐색처럼 문맥 추론이 필요한 작업에서는, 기하 정보만으로는 한계가 있어 결정이 자주 비효율적이었습니다. 한편 foundation model(LLM/VLM)을 zero-shot으로 결합하는 연구도 있지만, 탐색 단계에 따라 필요한 탐색-활용 균형이 달라져 고정된 방식은 성능이 흔들릴 수 있습니다.

- **Core Contribution**: RoboAtlas는 geometric exploration과 semantic reasoning을 상황(장면 이해 정도)에 맞춰 전환하는 “contextual Active SLAM” 프레임워크를 제안합니다. 3D 인스턴스 단위 시맨틱 매핑 시스템 OpenRoboVox와, frontier 탐색·전역 시맨틱 맵 추론·일인칭 VLM 추론을 각각 전문가(expert)로 구성한 뒤 contextual multi-armed bandit으로 다음 목표를 선택합니다. 이렇게 해서 초기에는 빠르게 지도를 채우고, 정보가 쌓이면 언어/의미 기반의 목표 지향 탐색으로 자연스럽게 이동합니다.

- **Technical Challenges**: 핵심 난제는 (1) 로봇에서 실시간으로 확장 가능한 3D 인스턴스 시맨틱 매핑을 유지하면서 (2) 탐색 단계별로 전문가 선택을 안정적으로 적응시키는 것입니다. OpenRoboVox는 memory-efficient TSDF 최적화, 인스턴스 pruning, distance-prioritized historical refresh, 비동기 병렬 처리를 통해 edge/실로봇 환경에서도 대규모 시맨틱 인스턴스를 다루도록 설계했습니다. 또한 Scene-Dictionary로 복잡한 voxel을 압축해 LLM/VLM 추론에 적합한 문맥 세계모델을 만들고, bandit은 점진적 문맥 특징(커버리지 변화, backtracking, VLM 신뢰, 장면-질의 관련도)을 바탕으로 exploration과 semantically guided navigation을 교체합니다.

- **Empirical Impact**: 실험은 시뮬레이션과 Unitree Go2 실로봇에서 1800m2 이상 대규모 환경(약 30k개의 mapped semantic instances)을 대상으로 검증됐으며, 작업 성공률 100%를 달성했습니다. GOAT-Bench의 “Val Unseen”에서 GPT-4o를 사용한 RoboAtlas는 SR 90.6%로 보고된 최고 성능이며, 최강 prior baseline 대비 SR을 17.8%p 끌어올렸습니다. 더 작은 Qwen2.5-VL-7B(7B)에서도 SR 88.8%를 유지해, 단순히 foundation model을 바꾸는 것보다 3D 시맨틱 맵으로 grounding한 문맥이 성능 향상에 결정적임을 보여줍니다.



### In-Context World Modeling for Robotic Contro (https://arxiv.org/abs/2606.26025)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 현재 관측과 언어 지시에만 조건을 거는 경우가 많아, 카메라 시점이나 로봇 형태처럼 실행 환경이 바뀌면 잘 일반화하지 못한다. 특히 학습 중에 마주친 고정된 실행 맥락을 전제로 동작하는 경향이 있어, 새로운 환경에서는 data-intensive fine-tuning 없이는 성능을 유지하기 어렵다.

- **Core Contribution**: 이 논문은 In-Context World Modeling(ICWM)으로, 시스템 식별을 in-context adaptation 문제로 취급해 로봇이 짧은 자기 상호작용 히스토리만으로 필수 시스템 변수를 추론하도록 만든다. ICWM은 demonstrations로 ‘무엇을 할지’를 정하는 전통적 in-context learning과 달리, context window로 ‘시스템이 어떻게 동작하는지’를 파악해 태스크 실행 전부터 현재 월드 다이내믹을 내재적으로 모델링한다.

- **Technical Challenges**: 핵심 기술 과제는 파라미터 업데이트 없이, 태스크와 무관한 상호작용 데이터만으로도 새로운 구성요소(예: 시점, 형태)의 시스템 변수를 안정적으로 식별해 정책 추론에 반영하는 것이다. 논문은 task 실행 전에 self-generated, task-agnostic interaction을 처리해 월드 다이내믹을 암묵적으로 학습시키는 방식으로 이 문제를 해결하며, 그 결과 adaptation을 모델 내부의 context 처리로 수행한다.

- **Empirical Impact**: ICWM은 시뮬레이션과 실제 로봇 플랫폼 모두에서 광범위하게 평가되었고, novel camera viewpoints 같은 미지 구성에서 표준 VLA baseline 대비 유의미하게 더 높은 성능을 보였다. 파라미터 업데이트 없이도 새로운 실행 맥락에 적응할 수 있다는 점에서, VLA의 일반화 한계를 줄이고 현장 적용 비용을 낮추는 방향으로 의미 있는 진전을 제시한다.



### G2DP: Diffusion Planning with Spatio-Temporal Grid Guidanc (https://arxiv.org/abs/2606.26017)
- **Prior Approaches**: 확산 기반 계획은 다양한 주행 양식을 다중모달로 생성할 수 있지만, 폐루프 실행에서 안전성과 경로 준수는 추가 가이드를 통해 해결해야 한다. 기존 가이드는 주로 이웃 에이전트에 대한 희소 기하 질의(유클리드 거리, temporal logic robustness 등)나 사후(post-hoc) 리파인먼트에 의존해, 장면 전반의 밀집된 상황 인지와 선제적 회피에 한계가 있다.

- **Core Contribution**: G2DP(Grid-Guided Diffusion Planning)는 추론(inference) 시점의 guidance를 “밀집하고 미분 가능한” 비용 그리드로 직접 주입해 확산 denoising 루프에서 안전·진행을 동시에 유도한다. U-Net이 예측한 spatio-temporal BEV 점유 확률과 경로 진행(progress) 맵을 융합해, 궤적 생성에 쓸 continuous safety energy functional로 구성한다.

- **Technical Challenges**: 핵심 난제는 차량 차원(footprint)과 미래 시간축의 충돌 위험을 장면 불확실성까지 반영하되, denoising 업데이트를 망가뜨리지 않게 미분 가능하게 설계하는 것이다. G2DP는 시간별 fused cost grid에 대해 Top-K footprint 집계로 잡음에 강한 에너지/그래디언트를 만들고, classifier guidance 형태로 DPM-Solver의 late denoising 단계(후반 스텝)에만 그래디언트를 주입해 미세 조향 효과를 극대화한다.

- **Empirical Impact**: nuPlan 폐루프 평가에서 G2DP는 reactive Test14-hard에서 imitation-learning 기반 SOTA를 달성하며, 가장 강한 baseline 대비 reactive score +7.2점 향상을 보인다. 또한 interPlan·DeepScenario에서 zero-shot 전이를 유지하면서 충돌 회피가 interPlan에서 비가이드(unguided) 대비 +10.15 개선되는 등 상호작용이 많은 장면에서의 견고한 안전성 향상이 확인됐다.



### FAR-LIO: Enabling High-Speed Autonomy through Fast, Accurate, and Robust LiDAR-Inertial Odometry (https://arxiv.org/abs/2606.26010)
Comments:
          8 pages, accepted for publication at IROS2026 (IEEE/RSJ International Conference on Intelligent Robots and Systems 2026)

- **Prior Approaches**: 로보틱스에서 주행 중 위치 추정은 매우 중요하지만, 자율 레이싱처럼 역동적 움직임과 센서 잡음이 큰 환경에서는 오도메트리 지연과 정확도 간 균형이 어렵다. 기존 LiDAR-관성 오도메트리 계열은 계산량이 커서 실시간 지연을 줄이기 위해 정확도를 희생하거나, 환경·센서 설정에 맞춘 튜닝 부담이 커지는 한계가 있었다.

- **Core Contribution**: 이 논문은 CUDA 가속을 전제로 한 LiDAR-관성 오도메트리 프레임워크 FAR-LIO를 제안한다. FAR-LIO는 Fast, Accurate, Robust를 목표로, GPU용 voxel hashmap과 sparsity-aware Generalized Iterative Closest Point(GICP) 및 적응형 임계값을 조합해 저지연을 유지하면서도 정확도를 보존한다. 또한 Extended Kalman Filter(EKF) 백엔드에 upsampling과 delay compensation을 적용해 IMU 고주파 정보를 안정적으로 융합한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 동적 환경에서 빠르고 정확한 nearest-neighbor 탐색을 수행하면서 (2) 지연을 최소화하고 (3) 잡음에도 견고한 최적화를 유지하는 것이다. 이를 위해 논문은 CUDA 기반 voxel hashmap으로 병렬 nearest-neighbor search와 효율적 맵 업데이트를 구현하고, sparisty-aware GICP에 adaptive thresholding과 adaptive density를 더해 저지연 최적화를 달성한다. 마지막으로 EKF에서 LiDAR 오도메트리와 IMU를 upsampling 및 delay compensation으로 동기화해 부드러운 출력을 만든다.

- **Empirical Impact**: 실험은 4가지 센서 설정에서 공개 데이터셋과 250 km/h급 자율 레이싱카 2대의 자체 주행 데이터를 포함해 수행됐다. FAR-LIO는 기존 state-of-the-art 대비 평균 자세(위치) 오차를 6.9% 줄이고 런타임은 38.4% 낮췄으며, 단일 파라미터 세트로 성능을 유지해 실용성을 강조한다. 이는 컴퓨팅 효율과 범용 적용 가능성을 보여주며, 레이싱 같은 closed-loop 응용에서 지연이 병목일 때 특히 의미가 크다.



### Emcar: Embodied Controller for Animating Robots (https://arxiv.org/abs/2606.26008)
Comments:
          Published in Lecture Notes in Computer Science, DOI: https://doi.org/10.1007/978-3-032-15501-6_11

- **Prior Approaches**: 기존 로봇 모션 프로그래밍은 보통 개발자 중심의 코딩과 전문 API 사용에 의존해 예술가·비전공자 참여 장벽이 컸다. 또한 상용 로봇은 기능이 미리 정의된 경우가 많아, 예술적 요구에 맞춘 상호작용 실험이 제약되는 문제가 있었다. 시각적 프로그래밍이나 vibe coding이 접근성을 높이더라도, 로봇은 여전히 물리적 제어와 모션 설계가 부담으로 남았다.

- **Core Contribution**: EMCAR(Embodied Controller for Animating Robots)는 협동로봇 팔을 직접 ‘몸으로’ 조작해 동작을 생성하고 애니메이션 시퀀스를 프로그래밍할 수 있는 no-code/low-code 소프트웨어 도구다. 조이스틱 같은 입력을 넘어, 예술적 실천(인형조종, 드로잉)을 HRI 설계 흐름에 녹여 비기술 사용자도 직관적으로 로봇과 협업할 수 있게 한다. 저지연 tele-operation 기반의 로봇 드로잉과, 물리 조작으로 기록하는 로봇 애니메이션을 GUI로 통합한 점이 핵심이다.

- **Technical Challenges**: 가장 큰 기술 과제는 비전공자가 쓰기 쉬운 인터페이스를 유지하면서도, 로봇의 공간 정렬·정밀한 도구 접촉·실시간 제어를 안정적으로 제공하는 것이었다. EMCAR은 캔버스 캘리브레이션(코너 터치로 좌표계를 매핑)과 Z Offset 같은 파라미터로 정렬 오차를 줄이고, UR RTDE(저지연)로 타깃 포즈를 연속 갱신해 그리기 제스처를 로봇 궤도로 변환한다. 또한 3D-printable end effector(잉크 도구 고정용)와 녹화/재생 기반의 워크플로를 결합해 repeatable한 동작 제작을 돕는다.

- **Empirical Impact**: EMCAR는 무용·공연(예: 사람의 몸/깊이카메라 기반 연동), 교육 워크숍, 커뮤니티 및 Arts & Health 프로그램 등 다양한 사용자 그룹에서 반복 검증됐다. 특히 신체 기반 조작과 기록-재생 구조 덕분에 빠른 반복, 아이디어 공유, Wizard of Oz 방식의 인터랙션 실험 설계가 가능하다는 점이 강조된다. 결과적으로 로봇을 ‘기술 장벽’이 아니라 ‘창작·참여 도구’로 재위치시키며, HRI 연구에서 접근성과 실험 속도를 동시에 높일 수 있는 플랫폼으로 의미를 가진다.



### FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation (https://arxiv.org/abs/2606.26006)
- **Prior Approaches**: VLA는 보통 SFT 같은 모방학습으로 사전학습되지만, 시연 데이터의 품질 한계인 imitation ceiling 때문에 성능이 더 이상 쉽게 오르지 못한다는 점이 알려져 있습니다. RL fine-tuning으로 이를 넘을 수 있지만, 실제 로봇에서는 샘플 비효율과 불안정(초기 unlearning, 저품질 탐색 데이터로 인한 업데이트 잡음)이 커서 사람 개입(HiL)에 의존하는 경우가 많았습니다. 또한 기존 방식은 Q-기반 업데이트를 하더라도 O2O 전환의 분포 불일치와 탐색 잡음 때문에 실패율이 출렁이는 문제가 반복됩니다.

- **Core Contribution**: 이 논문은 intervention-free 오프라인-투-온라인 VLA RL fine-tuning 프레임워크 FORCE를 제안하며, 시연 기반 한계를 RL로 넘는 동시에 안정성을 확보하는 것을 목표로 합니다. FORCE는 3단계로 구성되며, Value-Calibrated Warm-Up로 Q-function의 분포/지지영역을 정책 방문 분포에 맞춘 뒤, 온라인에서는 VGPD(Value-Guided Policy Self-Distillation)로 고가치 전이만 정책 업데이트에 반영해 탐색 잡음을 걸러냅니다. 결과적으로 “분포 이동으로 인한 성능 붕괴”와 “저품질 탐색 업데이트”를 동시에 줄이도록 설계됐습니다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 오프라인에서 학습된 과소/불안정 Q-value 스케일이 온라인 데이터를 만나며 초기 unlearning을 유발하는 cold-start covariate shift 문제, (2) 보상 희소성과 행동공간 크기로 인해 탐색 데이터의 품질이 낮아 정책 업데이트가 분산을 키우는 문제입니다. FORCE는 Warm-Up에서 on-policy 롤아웃을 소량 섞어 conservative value constraint를 적용함으로써 critic의 지원 영역을 확장해 Q-estimate의 바깥삽입(extrapolation)과 정의역 불일치를 줄입니다. 이어 VGPD는 KL 제약 기반의 정규화된 정책개선 관점에서, 샘플별 value advantage를 동적 기준선으로 필터링해 음의 어드밴티지를 버리고(Positive Advantage Truncation) 고가치 전이로만 자기 증류를 수행하게 합니다.

- **Empirical Impact**: 실험은 ManiSkill 시뮬레이션과 Franka Emika Panda 실로봇에서 수행됐고, FORCE는 평균 성공률 82.3%(Octo 백본 기준)로 기존 RL 방법(ConRFT 등) 대비 10%p 이상 성능 우위를 보였습니다. 또한 성공률이 내려가는 흔한 성능 하락 현상을 완화하면서, 학습 단계 관점에서도 평균 Steps@80%가 32.5% 단축되는 등 샘플 효율을 개선했습니다. 특히 인간 개입 없이도 contact-rich 과제에서 거의 100% 성공률을 달성해, 자율 로봇 에이전트 배치 가능성을 한 단계 끌어올렸다는 점에서 의미가 큽니다.



### Action ControlNet: A Lightweight Delay-Aware Adapter for Smooth Asynchronous Control in Vision-Language-Action Models (https://arxiv.org/abs/2606.25985)
- **Prior Approaches**: 기존 VLA(vision-language-action) 로봇은 diffusion, flow matching 같은 generative action head로 짧은 액션 chunk를 생성하지만, 추론-실행이 분리되면 idle 시간이 생겨 고주파 closed-loop가 흔들린다. 이를 줄이기 위해 asynchronous 실행을 쓰면 다음 chunk가 적용될 때 관측이 stale이라 handoff 경계에서 불연속(액션 jitter·전이 충격)이 누적될 수 있다. 이를 완화하려는 방법들은 inpainting/보정 헤드/미래 상태 예측/학습 시 지연 시뮬레이션 등으로 접근하지만, 대부분 휴리스틱이거나 정책 구조·재학습 부담이 커 실제 활용성에 제약이 있었다.

- **Core Contribution**: 이 논문은 asynchronous chunked control에서 핵심 오류를 ‘전체 이해 붕괴’가 아니라 ‘chunk handoff의 경계 조건 문제’로 재정의한다. 그리고 지연 시간 동안 로봇이 실제로 실행한 motion suffix(지연 액션)를 조건 신호로 삼아, 대부분 frozen인 action head에 residual로 보정하는 Action ControlNet(ACNet)을 제안한다. ACNet은 pretrained backbone은 그대로 두면서 few trainable parameters만 추가하고, diffusion/flow matching 같은 generative action head에도 붙일 수 있는 plug-and-play 호환성을 강조한다.

- **Technical Challenges**: 주요 기술적 난제는 지연으로 인해 chunk의 앞부분이 stale 관측에서 생성되지만 실제로는 이미 진행 중인 동작과 이어져야 한다는 ‘경계 불일치’를 어떻게 국소적으로 보정하느냐였다. ACNet은 delay action을 horizon에 맞게 learnable padding으로 명확히 표시한 뒤, 가벼운 transformer 인코더와 action head의 terminal temporal pooling을 거쳐 late residual로 주입함으로써 이 문제를 해결한다. 또한 backbone을 재학습하지 않고도 다중 지연 d를 커버할 수 있게 관측-언어 latent은 캐시해 지연 조건만 추가 학습하도록 설계해 효율을 확보했다.

- **Empirical Impact**: Kinetix와 Meta-World MT50, 그리고 SO-ARM101 실로봇에서 ACNet은 성공률과 전이 구간의 매끄러움(jerk/전이 stuttering)을 모두 개선했다. 특히 Kinetix에서 평균 성공률은 ACNet 0.79로 Naïve Async 0.61, RTC 0.72를 상회했고 Training-RTC(0.80)에 근접하면서도 적응 시 trainable 파라미터 비율을 약 20% 수준으로 줄였다. Meta-World에서는 ACNet이 평균 성공률 0.74를 달성하며 RTC 대비 더 낮은 지연(91ms vs 159ms)과 더 높은 control frequency(11.0Hz vs 6.28Hz)를 보였고, 실환경에서도 총 성공 20/20으로 Naïve Async(17/20)보다 전이 진동과 접촉 안정성이 좋아졌다.



### A Sensorised Lattice Footplate for a Semi-Active Prosthetic Foo (https://arxiv.org/abs/2606.25966)
Comments:
          6 pages, 7 figures, ICAC

- **Prior Approaches**: 기존 보철 발은 수동 ESR(energy-storing-and-returning) 방식이 주류로, 제조·정렬 이후 강성/감쇠가 고정돼 보행 국면이나 지형 변화에 덜 적응한다. 또한 발바닥 정보는 주로 인솔처럼 외부 층에 의존해 센싱이 본체 하중 경로와 분리되는 한계가 있었다. 한편 magnetic tactile sensing은 변형 기반 자기장 변화로 힘을 추정할 수 있으나, 이를 하중 지지 컴플라이언트 구조 내부에 “직접 내장”하는 설계는 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 저비용 반수동 보철 발에서, 하중을 받는 컴플라이언트 엘리먼트(3D-printed lattice footplate) 내부에 magnetic plantar sensing을 직접 내장하는 접근을 제안한다. 센싱된 전후족(forefoot/rearfoot) 힘 정보를 이용해 반수동 감쇠(damping)를 구동하는 프로토타입 아키텍처와 파이프라인을 구성하고, 센싱→추정→국면 라벨링→감쇠 스케줄까지의 연결 가능성을 보여준다. 생체 발의 dorsiflexion 경향을 모사하는 feedforward 감쇠 스케줄을 함께 제시해 “센싱 기반 반수동 조절”의 타당성을 실험·시뮬레이션으로 확인한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 하중 지지 구조 내부에 센서를 넣어도 컴플라이언스(강성 튜닝)를 해치지 않으면서, (2) 변형-유도 자기장 신호로부터 힘을 안정적으로 추정하고, (3) 그 읽기값이 국면별 감쇠 조절에 쓸 만큼 구분 가능해야 한다는 점이다. 저자들은 lattice unit-cell 크기로 강성을 조절하면서, BCC lattice 안에 자석과 magnetometer PCB를 내장해 Instron 압축 데이터로 MLP(force regressor)를 학습하고 RMSE 약 4.85N 수준의 제어된 정상 하중 추정 정확도를 확보했다. 감쇠 구동을 위해서는 servo-adjustable hydraulic damper를 각도-감쇠계수로 실험 피팅해, reduced-order ankle 모델에 결합 가능한 설계 파라미터로 전환했다.

- **Empirical Impact**: 실험 결과, 선택한 lattice(예: 4.75mm 케이스)는 주파수 분포가 아니라도?—주파수 대신 하중 응답에서 unit-cell 크기에 따라 강성이 단조롭게 변하며, 내장 센서가 testing-machine 기준 힘을 하중/제하 구간에서 추적했다(RMSE 32.14N, full scale 대비 4.59%). 정적 자세(heel-strike, foot-flat, dorsiflexion, toe-off)에서 전후족 힘 분포가 선형적으로 separable하게 나타나 센싱 파이프라인의 기동성을 예비 검증했다. 시뮬레이션에서는 감쇠만으로 dorsiflexion 구간의 경향을 근사하면서도, purely dissipative 구조의 한계로 late-stance에서 active push-off 생성이 불가능함을 정량적으로 드러내, 반수동 개념의 역할 범위를 명확히 했다.



### Mixture-of-Experts RL for Fault-Tolerant Legged Locomotion (https://arxiv.org/abs/2606.25965)
- **Prior Approaches**: 기존 강화학습 기반 다리 로봇 제어는 주로 monolithic neural policy로 지형 인식, 보행 제어, 고장 적응을 한 네트워크에 함께 넣어 학습하는 경향이 있다. 이 방식은 고장 조건이 다양해질수록 서로 이질적인 보상/행동을 단일 파라미터 공간에 억지로 담아 representational interference가 생기며 효율과 성능이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 actuator failure 진단 정보를 직접 활용하는 fault-aware modular 제어 아키텍처를 제안한다. 각 고장 모드마다 전용 control expert를 두고, learned routing 없이 fault-detection 모듈이 추정한 failure mode에 해당하는 expert만 one-hot으로 활성화해 조건부 연산을 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 현재 actuator 고장 모드를 신뢰성 있게 식별하고 (2) 심각한 고장에서도 접촉(contact) 가정이 흔들리는 상황에서 상태(기저 속도 등)를 안정적으로 추정하며 (3) 선택된 expert만 업데이트되는 학습 불안정/샘플 부족 문제를 완화하는 것이다. 논문은 supervised fault classifier를 observation history로 학습하고, contact 모델 의존을 줄이기 위해 concurrent learning 기반 state-estimation(기저 선속도 회귀)에 fault one-hot을 함께 주어 해결하며, 충분히 큰 병렬 시뮬레이션 환경으로 학습 노이즈를 줄였다.

- **Empirical Impact**: 시뮬레이션 및 Unitree Go2 실로봇 실험에서 제안한 fault-conditioned MoE가 monolithic PPO 대비 네 가지 고장 시나리오 전반에서 누적 보상 성능이 일관되게 높았다. 특히 고장 심각도가 커져 보행 전략이 급격히 달라지는 LH+RH Fail 같은 구간에서 격차가 커졌고, 네트워크 용량을 크게 줄인 Small에서도 monolithic 대비 성능 저하가 덜해 compute-constrained 로봇(예: 우주 탐사용)에 유리함을 보여준다.



### From Rubble Simulation to Active Magnetic Mapping: Quantum Sensing for Disaster Respons (https://arxiv.org/abs/2606.25957)
Comments:
          9 pages, seven figures

- **Prior Approaches**: 붕괴 현장에서 생존자를 찾기 위해 드론을 포함한 여러 감지(음향·열화상·내시경·레이더·가스/CO2 추적 등)가 동원되지만, 대부분은 잔해 내부 구조에 대한 정보가 부분적입니다. 특히 지상/공중 기반 레이더나 영상 중심 접근은 ‘잔해 아래 연속 구조’를 직접 매핑하는 데 한계가 있어 탐색 범위를 줄이기 어렵습니다. 자율 플랫폼의 자기 이상 기반 방법(MagSLAM)은 위치추정에 강점이 있지만, 주어진 드론 자세 하에서 필드를 구조 맵으로 효율 복원하는 문제는 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 드론 기반 quantum magnetometry를 재난 잔해 분석의 보완 모달리티로 제안하고, 물리 시뮬레이션부터 능동 복원까지 end-to-end 파이프라인을 구성합니다. Unreal Engine으로 철근콘크리트 주차장 붕괴를 생성한 뒤, per-triangle dipole 근사로 자기장을 계산하고 이를 Gaussian Process Regression(GPR) + Bayesian active sampling으로 연속 구조(자기적 토폴로지)로 복원합니다. 연구는 또한 회수 가능한 구조가 UAV 운용 고도에서 어떤 감도 범위를 요구하는지 정량 민감도 타깃을 제시합니다.

- **Technical Challenges**: 핵심 난제는 (1) 붕괴 메쉬에서 드론 관측 신호로 이어지는 물리 모델의 타당성, (2) 희소한 멀티-센서 샘플만으로 잡음 속 연속 맵을 안정적으로 복원하는 알고리즘 설계입니다. 저자들은 dipole approximation으로 자기장을 빠르게 생성하고, GPR의 RBF 커널에 백색잡음 항을 더해 예측 불확실도를 모델링한 뒤 UCB(Upper Confidence Bound) 기반의 능동 경로를 선택해 최대 불확실도 지점 위주로 샘플링합니다. 그 결과, 센서 수/배치/비행 전략이 복원 품질과 payload 제약 사이의 균형에 미치는 영향을 체계적으로 다룹니다.

- **Empirical Impact**: 시뮬레이션 결과, 철근콘크리트 붕괴에서 의미 있는 구조 대비가 붕괴 지점 근처 소스 표면에서는 30 pT~3 μT 수준으로 관찰되지만 UAV 고도에서는 sub-pT~sub-nT 범위에서 standoff 1~2 m 내 구조가 식별 가능한 것으로 나타납니다. 또한 GPR 복원은 3-sensor array와 Bayesian active sampling 조합에서 붕괴 실현 여러 가지에 대해 약 100 samples 전후로 구조 상관이 정점에 도달하며, 이 결론은 독립 시뮬레이션에서도 유지됩니다. 논문은 void detection이나 안전한 잔해 제거 계획처럼 후속 임무에 연결 가능한 ‘구조 맵’ 생산 가능성을 보여주며, quantum-grade sensing이 재난 구조/분석 도구로 확장될 잠재력을 강조합니다.



### DSP-SLAM++: A Unified Framework for Multi-Class, High-Fidelity Object SLAM in the Wild (https://arxiv.org/abs/2606.25953)
Comments:
          9 pages, 9 figures

- **Prior Approaches**: 기존 object-aware SLAM은 객체를 박스(cuboids)나 쿼드릭(quadric) 같은 단순 프리미티브로 표현해 속도와 다중 클래스 확장을 확보하지만, 실제 형상 정밀도가 떨어져 근접 주행·계획에 제약이 컸다. 반대로 DSP-SLAM류의 deep implicit shape prior는 고품질 메시에 강점이 있으나 연산 비용이 높고 단일 클래스/제한된 센서 구성에 머무르는 경향이 있었다. 또한 3D Gaussian Splatting 계열은 렌더링 품질은 높여도 객체 인스턴스 단위의 일관된 의미론 파악이 부족하다는 평가가 뒤따랐다.

- **Core Contribution**: DSP-SLAM++는 DSP-SLAM의 implicit 표현력을 유지하면서, 실시간 multi-class를 가능하게 하는 구조를 제안한다. 핵심은 비동기(asynchronous) 매핑 파이프라인으로 객체 재구성을 추적 스레드와 분리해 지연 병목을 줄이고, monocular fisheye-LiDAR에 맞춘 센서 융합·보정 모듈로 실차 환경 적용성을 높인 점이다. 그 결과 고정밀·기하 완전한(geometrically-complete) 객체 메시는 유지하면서도 25 Hz 수준의 견고한 실시간 성능을 노린다.

- **Technical Challenges**: 기여를 실현하는 첫 관문은 multi-class에서 2D 마스크와 3D 바운딩 박스의 잘못된 매칭(association ambiguity)을 줄이면서도 계산량을 통제하는 것이다. 논문은 클래스 일관성 기반 projection과 global greedy bipartite matching으로 2D-3D 인스턴스 매칭을 한 번에 정리하고, 클래스별 바닥면 스케일을 반영한 class-conditioned association으로 프레임 간 추적 안정성을 보강한다. 두 번째 관문은 fisheye의 심한 왜곡 때문에 LiDAR 깊이를 2D instance mask에 정확히 결합하기 어려운 문제이며, 전체 픽셀 rectification 대신 mask contour만 rectification하는 경량 전략과 ORB-SLAM3의 native fisheye 지원을 조합해 정합성을 확보한다. 마지막으로 implicit 객체 최적화의 시간 비용을 해결하기 위해 객체를 바운딩 박스 placeholder로 즉시 삽입하고, DeepSDF 최적화/mesh extraction을 back-end 워커 스레드로 넘기는 비동기 재구성(AR)을 도입한다.

- **Empirical Impact**: 실험은 KITTI, nuScenes, CBNU 및 논문 자체 in-house 25 Hz 다중 클래스 데이터셋에서 수행되었고, DSP-SLAM++는 tracking 정확도와 scale 일관성에서 기준 대비 경쟁력 있는 수준을 보여준다. 특히 custom multi-class 시나리오에서 SR(동기) 방식의 매핑 지연 피크가 >500 ms로 커지는 반면, AR 비동기 파이프라인은 평균 매핑 지연을 1/3로 낮추고 최대 Keyframe BA latency를 3 프레임 수준으로 축소해 실시간성 붕괴를 방지했다. 또한 객체 복원 지연(thread bottleneck)을 최대 70%까지 줄이면서도 다중 클래스의 고품질 객체 메시는 유지되어, 자율주행과 로봇 조작 같은 실세계 task로의 확장성을 강화했다. 단, 보행자처럼 얇거나 가려짐이 심한 클래스는 입력 포인트 클라우드 밀도와 사전학습 품질에 의존해 세부 형상이 떨어질 수 있으며, 이는 향후 개선 여지로 제시됐다.



### DeformGen: Dynamics-Based Topology Augmentation for Deformable Manipulation Policy Learning (https://arxiv.org/abs/2606.25939)
- **Prior Approaches**: 기존 demonstration augmentation은 주로 rigid 물체의 SE(3) 등변성에 기대어, 엔드이펫에 같은 강체 변환을 적용하면 유효 궤적이 된다는 가정으로 데이터를 늘려왔다. 하지만 변형 물체에서는 상태공간이 고차원이고(형상·토폴로지 다양성), 내부 동역학 제약 때문에 단순한 저차원 pose 교란으로는 물리적으로 가능한 상태에 도달하기 어렵다. 또한 변형에서는 material point가 더 이상 강체처럼 함께 움직이지 않아, rigid-style trajectory transfer는 접촉 정렬 붕괴와 국소 변형 보정 불가 문제를 만든다.

- **Core Contribution**: DeformGen은 변형 조작에서 비용 효율적으로 topological diversity를 확보하기 위해 dynamics 기반 상태 합성과 deformation-aware 궤적 전이를 함께 수행하는 augmentation 프레임워크다. 먼저 localized physical disturbance를 가하고 forward simulation으로 물리적으로 그럴듯한 deformable state를 생성해, 기존 rigid 스타일이 놓치던 유효 상태 분포의 범위를 확장한다. 이어서 source manipulation trajectories를 deformation-field warping으로 각 대상 상태의 변형에 맞게 재구성해, 접촉 정렬과 조작 거동을 일관되게 맞춘다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 높은 차원의 상태공간에서 simulator 동역학이 보존하는 ‘유효 상태’의 제약 매니폴드를 벗어나지 않으면서 토폴로지 다양성을 만들고, (2) 강체 등변성이 깨진 상황에서 per-particle 변위 정보를 연속적인 deformation field로 승격해 엔드이펫 경로를 동시에 정합하는 것이다. DeformGen은 localized force를 주고 시뮬레이션 롤아웃으로 plausibility를 유지해 상태 합성의 안정성을 확보하며, trajectory는 per-particle displacement를 KK-nearest-neighbor inverse-distance interpolation으로 D(x)로 만든 뒤 위치·방향을 현지 기하 변화에 따라 warping한다. 그 결과 한 개의 시연이 동일 작업의 다양한 변형 상태군에도 접촉을 유지한 채 재사용될 수 있다.

- **Empirical Impact**: Real2Sim-Eval/PhysTwin 기반 고정밀 변형 조작 벤치마크(로프 라우팅, 토이 패킹, 클로스 폴딩)에서 DeformGen은 원시 시연만으로 학습하거나 rigid-style augmentation을 쓴 경우보다 대체로 높은 성공률을 보였다. 특히 state 합성에서 rigid 교란 대신 dynamics 기반 topology 생성이 일반화에 기여하고, deformation-field warping이 trajectory 측면에서도 추가 이득을 준다는 점이 ablation으로 확인됐다. 또한 합성 궤적 생성이 실패한 ‘어려운’ 테스트 케이스에도 일부 성과가 나타나, 모델이 단순 memorization을 넘어 변형에 대응하는 조작 전략을 학습했음을 시사한다.



### A 3D-Printable Dataset for Fair Testing and Comparisons of Tactile Sensors (https://arxiv.org/abs/2606.25886)
- **Prior Approaches**: 기존 촉각 텍스처 분류 벤치마크는 특정 촉각 센서가 실제 표면(오브젝트)과 접촉할 때 얻는 센서 응답 중심으로 구성돼, ‘텍스처 자체’의 재현성이 부족했습니다. 그 결과 서로 다른 센서 간 성능 비교가 공정하지 않고, 재현 가능한 연구가 어렵다는 한계가 있었습니다. 표준화된 3D 프린트 자극이 있더라도 인간 지각 해상도 같은 다른 목적에 치우쳐 텍스처 분류 비교용으로는 부족하다는 문제의식도 제기됩니다.

- **Core Contribution**: 이 논문은 수학적으로 정의된 3D 프린터용 텍스처를 공개하고, 여러 프린터/필라멘트에서도 물리적으로 동일한 자극을 만들 수 있도록 설계·검증합니다. 6개의 파라미터화된 표면 패턴은 sine-wave와 Fourier 기반 함수 조합으로 공간주파수, 진폭, 방향성 구조를 제어합니다. 이를 바탕으로 TacTip 같은 시각형 촉각 센서의 ‘공정한 센서 비교’와 향후 sim-to-real 전이에도 활용 가능한 물리 재현성 기반 벤치마크를 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 자연 텍스처처럼 복잡한 형상을 그대로 스캔하면 샘플 내부 비균일성으로 접촉 위치에 따른 센서 응답이 달라져 재현성이 무너진다는 점입니다. 이를 해결하기 위해 USAF 캘리브레이션 패턴 아이디어를 차용해 스케일 가능한 기하학적 패턴을 수식으로 생성하고, 매끈한 반복 구조가 되도록 sine/Fourier 기반으로 텍스처를 통제합니다. 또한 프린트 품질 차이가 촉각 서명에 미치는 영향을 정량화하기 위해, TacTip 영상의 분산(variance)을 ‘동일 부위 반복 접촉’ 조건에서 측정하고 stringing과 피크 날카로움 같은 인쇄 결함이 분산에 영향을 준다는 점을 확인합니다.

- **Empirical Impact**: 실험 결과, 인쇄 품질은 촉각 영상 분산에 유의미한 영향을 주며 특히 peak sharpness와 stringing이 큰 요인으로 나타났습니다. 레진 프린터가 전체적으로 반복성(분산)이 가장 낮았고, 필라멘트 공급 프린터 중에서는 고급 프린터(Bambu Lab)가 더 일관된 서명을 보였습니다. 분류 실험에서 같은 프린트/필라멘트로 학습하면 within-printer generalisation이 강하지만, 프린터 간 기하 일관성 불일치 때문에 cross-printer generalisation은 여전히 어려웠으며, 이는 random forest와 PCA 기반 모델/ANN 모두에서 관찰됩니다. 저자들은 공개 3D-printed texture benchmark과 함께 프린터-필라멘트 변동성을 분산으로 평가하는 구조를 제공해, 향후 촉각 센서 비교를 더 재현 가능하게 만들 수 있다고 의미를 부여합니다.



### TacVerse: A Multi-Sensor Dataset and Benchmark for Cross-Sensor Vision-Based Tactile Perception (https://arxiv.org/abs/2606.25877)
- **Prior Approaches**: 기존 비전 기반 촉각(tactile) 연구는 센서와 태스크에 맞춘 센서-특화 데이터셋을 중심으로 진행돼 센서가 달라질 때 표현이 얼마나 옮겨가는지 체계적으로 비교하기 어려웠습니다. 또한 다중 센서 자료가 있더라도 센서-태스크 조합이 통일되지 않거나 전이(transfer)·평가 설정이 분산돼 ‘센서 shift’ 영향과 ‘소량 타깃 데이터로의 복구’를 한 벤치마크 안에서 분리해 측정하기는 힘들었습니다. 


- **Core Contribution**: TacVerse는 7종 비전 촉각 센서에서 총 106,800장의 라벨 이미지로 구성된 다중 센서 데이터셋이자 크로스-센서 비전 촉각 벤치마크입니다. 공통 태스크 정의로 shape 분류(9-way), grating 분류(30-way), force 회귀를 제공하고, 학습/평가를 within-sensor, zero-shot cross-sensor transfer, few-shot adaptation으로 나눠 센서 shift 갭과 데이터 효율 적응을 동시에 측정합니다. 


- **Technical Challenges**: 핵심 기술적 난제는 센서 설계가 광학·조명·겔/마커 구조·해상도·접촉 형상에 따라 크게 달라 이미지 분포와 신호 특성이 바뀌는 ‘센서 shift’가 모델 전이에 어떤 손실을 만드는지 분리해 평가하는 것입니다. TacVerse는 고정 백본과 태스크별 프로토콜을 통해 shift 영향을 통제하고, force 회귀에 대해서는 타깃 센서의 소량 라벨로 few-shot fine-tuning을 수행해 복구 가능성을 정량화합니다. 
또한 MAE(Masked Autoencoder) self-supervised pretraining이 태스크와 센서 전반에서 가장 일관된 성능 향상을 주는지 representation study로 검증합니다.

- **Empirical Impact**: 실험 결과 within-sensor 학습 성능이 높더라도 zero-shot cross-sensor transfer에서는 상당한 성능 저하가 나타났고, 특히 grating 분류와 force 회귀가 shape 분류보다 센서 mismatch에 더 민감했습니다. few-shot adaptation은 force 회귀에서 타깃 센서로의 성능을 끌어올리지만, within-sensor 상한까지 완전히 회복하진 못했습니다. 
마지막으로 MAE pretraining은 분류·회귀 태스크에서 전반적으로 가장 일관된 이득을 제공해, 촉각 비전에서 self-supervised tactile representation의 유효한 ‘공용 초기화’ 가능성을 실증적으로 보여줍니다.



### Beyond a Shadow of a Doubt: Close Proximity Geometry Reconstruction Using FMCW Radar Shadow Effects (https://arxiv.org/abs/2606.25829)
- **Prior Approaches**: 자율주행에서 악천후·야간 같은 환경이 나빠지면 카메라와 LiDAR의 성능이 크게 저하된다. 밀리미터파 FMCW 레이더는 강건하지만, 레이더의 고도 정보가 붕괴(elevation collapse)해 기하 추론이 제한돼 왔다. 기존 2D 레이더 활용은 주로 위치 추정에 머물러 장면의 3D 복원으로 확장하기가 어려웠다.

- **Core Contribution**: 이 논문은 차량 섀시가 레이더 빔을 가리는 과정에서 생기는 ‘기하학적 섀도(shadow)’가 일관된 단서가 된다는 점에 주목한다. 해당 섀도와 레이더 반환 경계가 교차하는 근거리의 세로로 긴(슬렌더) 물체에 대해, 물체의 3D, in-plane inclination(평면 내 기울기)을 장면 전체에 대한 가정 없이 복원하는 방법을 제안한다. 특히 2D 레이더의 회전 정보를 섀도 기반 기하 단서로 끌어올려 3D 장면 재구성의 방향을 연다.

- **Technical Challenges**: 핵심 난제는 섀시가 만드는 섀도 자체를 믿을 만한 기하학적 큐로 모델링하면서, 물체의 기울기를 레이더 반환 경계에서 직접 읽어내는 분석적 매핑을 구성하는 것이다. 저자들은 레이더 반환 경계와 opening angle 사이의 관계를 이용한 analytical, closed-form mapping을 설계해 ‘가정 없는’ 추정이 가능하도록 했다. 다만 실제로는 레이더 스캔에서 물체 분할(segmentation)이 가장 큰 병목으로 드러났고, 이 때문에 회귀·추정 정확도와 직결되는 전처리 품질이 중요해진다.

- **Empirical Impact**: 시뮬레이션과 Navtech CTS350-X 레이더 실험을 통해, 실사용 가능한 조건에서도 해당 물체의 기울기를 추정할 수 있음을 보였다. 성능의 병목은 주로 segmentation으로 확인돼, 향후 레이더 포인트/픽셀 분할 개선이 후속 연구의 핵심 과제가 된다. 전반적으로 섀시 섀도를 새로운 기하 단서로 정립해, 2D rotating radar의 역할을 단순 localisation을 넘어 3D 재구성으로 넓힌 점에서 의미가 있다.



### MIL-LC: A Robust Magnetometer-Inertial-LiDAR Fusion Multimodal Localization Framework (https://arxiv.org/abs/2606.25796)
- **Prior Approaches**: 기존 AMR 멀티모달 로컬라이제이션은 센서 보완성을 활용하지만, 대부분이 기하(geometry)·텍스처 기반 매칭이나 비콘 같은 인프라 의존에 치우쳐 배치 유연성과 유지보수 비용에 제약이 있다. 자석(ambient magnetic field, AMF)을 단독 또는 보조로 쓰는 연구도 있었지만, 자기 신호의 구별성 부족과 낮은 신호대잡음으로 정확도가 제한되고 연산·매칭 방식이 무겁거나(순차 매칭/particle-filter) 로봇 환경의 장기 안정성이 충분히 검증되지 않았다.

- **Core Contribution**: 이 논문은 AMR용 magnetometer-inertial-LiDAR 융합 로컬라이제이션 프레임워크인 MIL-LC를 제안하며, LiDAR 기하적 퇴화와 장기 자기장 변화 상황에서도 신뢰도 높은 포즈 추정을 목표로 한다. 또한 IMU를 활용해 센서 시간 동기화·모션 왜곡을 정리하고, 자석 기반 추정치를 LiDAR scan matching의 초기값으로 사용해 느슨결합(loosely coupled) 구조로 end-to-end에 가까운 구현 난이도를 낮췄다.

- **Technical Challenges**: 핵심 기술 난제는 (1) LiDAR 포인트가 scan 윈도우 동안 연속 획득되며 모션 왜곡이 생기는 점, (2) magnetometer와 IMU의 비동기화로 추가 왜곡이 누적되는 점, (3) AMF는 환경 변화 시 측정 신뢰도가 달라져 outlier와 측정 노이즈를 적응적으로 다뤄야 하는 점이다. 이를 위해 논문은 IMU 기반 forward/backward propagation으로 LiDAR·자기 관측을 scan 종료 시각에 동기화하고, 자석은 여러 고주파 관측을 weighted least squares로 묶어 불확실성을 추정한 뒤 IESKF 업데이트에 반영하는 방식으로 해결한다.

- **Empirical Impact**:  시뮬레이션과 실제 환경에서 MIL-LC의 강건성과 정확성을 폭넓게 검증했으며, 특히 장기 운용에서의 자기 교란 내성까지 실험으로 보여준다. 결과적으로 자석-기하·텍스처에 덜 의존하는 보완 모달리티로서 AMR 로컬라이제이션의 실사용 가능성을 높였고, 인프라 설치 없이도 안정적인 융합 로컬라이제이션을 지향하는 연구 흐름에 의미 있는 실증 근거를 제공한다.



### StairMaster: Learning to Conquer Risky Hollow Stairs for Agile Quadrupedal Robots (https://arxiv.org/abs/2606.25765)
Comments:
          9 pages, 9 figures

- **Prior Approaches**: 기존 연구는 proprioception(자세·관절 정보)만으로도 일반 지형이나 solid stairs를 어느 정도 커버하는 정책을 제안해 왔지만, hollow stairs처럼 희소·불연속 구조에는 예측 인지 부재로 인해 다리 함정(leg trapping) 같은 치명적 실패가 반복됩니다. 비전 기반 접근도 solid stairs 중심으로 최적화되어 전방 카메라의 시야 공백(지형이 사라짐), 시공간 연결성 유지의 한계, 그리고 격심한 진동 환경에서의 depth 잡음 민감도로 인해 hollow stairs에 취약했습니다.

- **Core Contribution**: 논문은 StairMaster라는 3단계 강화학습 프레임워크를 제안해, hollow stairs의 고위험 조건에서 안정적인 보행을 목표로 합니다. Cross-Attention으로 노이즈 depth에서 구조 특징을 뽑고, Spatial-aware LSTM(SRU)으로 가려진 스레드의 시공간 메모리를 유지하며, 3D waypoint 기반 active perception 보상과 hollow gap/edge 페널티로 발 디딤 정확도를 강제합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 반사/박판 구조로 인한 depth sparsity와 고주파 잡음, (2) 전방 카메라 시야 공백으로 인한 장시간 partial observability, (3) 시뮬레이터-실세계 depth 아티팩트 차이로 인한 sim-to-real 붕괴입니다. 저자들은 고정밀 depth sim-to-real 센서 모델링(홀 노이즈, 엣지 노이즈, 진동/충격 기반 동적 잡음 등)을 학습에 반영하고, SRU의 공간 게이팅으로 시야 공백을 메우는 장기 표현을 구성한 뒤, rpitch 보상으로 기울기를 능동 조절해 다음 스레드를 더 일찍 관측하도록 설계했습니다.

- **Empirical Impact**: Unitree Go2에 실제 적용해 55° hollow stairs를 zero-shot transfer로 정복했으며, RL 기반으로는 이 정도 경사에서의 실환경 성공을 최초로 주장합니다. 시뮬레이션과 잡음/기울기/혼합 지형 실험에서 기준선과 ablation 대비 성공률·도달 스텝이 크게 개선되고, 특히 depth 잡음 200% 수준에서도 높은 완료 성능을 보이며 제안한 잡음 모델링과 보상 설계의 실효성을 확인했습니다.



### Stage-Aware and Roughness-Constrained Diffusion Policy for Multi-Stage Robotic Polishing (https://arxiv.org/abs/2606.25754)
- **Prior Approaches**: 기존 로봇 폴리싱 접근은 대개 오프라인 프로그래밍이나 모델 기반 제어에 의존해, 소량 다품종·복잡 표면 환경에서 적응성이 떨어진다는 한계가 있었다. 모방학습 쪽에서는 가우시안 혼합모델·DMP 등에서 출발해 GAIL, Behavior Transformer, ACT, Diffusion Policy로 발전했지만, 폴리싱처럼 장기(horizon) 과정의 stage 불확실성과 물성·품질로 직결되는 파라미터 결합을 동시에 다루긴 어렵다. 특히 diffusion의 샘플링 랜덤성이 feed speed와 normal contact force 같은 공정 변수를 구속 없이 흔들어 표면 거칠기 및 물리적 타당성 문제를 유발할 수 있다.

- **Core Contribution**: 이 논문은 Stage-Aware and Roughness-Constrained Diffusion Policy(SRDP)로, 폴리싱의 잠재 stage를 멀티모달 관측 히스토리로 추론하고 이를 diffusion reverse denoising에 조건으로 주입해 stage-일관된 행동을 생성한다. 또한 거칠기(roughness)를 기준으로 한 공정 제약을 diffusion 샘플링 과정에 결합해, 스핀들 속도는 stage-wise로 미리 정하되 feed speed와 normal contact force를 물리적으로 가능하고 품질 지향적으로 생성·조정한다. 실행 시에는 외부 stage 라벨 없이도 stage posterior 기반으로 전환 안정성을 확보하는 것이 핵심이다.

- **Technical Challenges**: 핵심 난제는 ① 시각적으로 비슷한 상태가 stage가 달라질 수 있는 장기 stage 불확실성, ② diffusion의 확률적 denoising에서 공정 파라미터가 상호결합적으로 드리프트(표류)해 비물리적·품질 불안정 조합이 나올 수 있다는 점이다. 논문은 멀티모달 히스토리를 입력으로 stage inference network로 posterior를 추정하고, stage-transition consistency prior(전이행렬 기반 정규화)로 stage 예측의 시간적 흔들림을 억제한다. 이어 roughness-oriented process model을 이용해 roughness-consistency와 physical-feasibility 에너지를 샘플링 가이드 그래디언트로 삼고, 최종 파라미터를 joint feasible set으로 투영하는 방식으로 공정 제약을 구현한다.

- **Empirical Impact**: 우주선 캐빈 코팅 표면 폴리싱과 내·캐비티 구조 표면 피니싱의 두 대표 시나리오에서, 강한 baseline 비교·모듈 ablation·실로봇 검증까지 수행해 SRDP의 효과를 포괄적으로 평가한다. 결과는 stage-transition 안정성, 공정 파라미터 일관성, 최종 표면 품질이 시나리오 전반에서 개선됨을 보여준다. 또한 stage 라벨 없이도 stage-consistent 생성과 roughness 기반 규제가 동시에 작동하는 점에서, 폴리싱을 넘어 grinding, drilling, deburring 같은 공정 지향 생성 정책으로 확장 가능한 방향성을 제시한다.



### Learning Asynchronous Upper-body Task-space Trajectory Tracking Policy for Humanoid Robots (https://arxiv.org/abs/2606.25706)
Comments:
          10 pages, 8 figures

- **Prior Approaches**: 기존 휴머노이드 제어는 서보 수준 추종에서 태스크 수준 실행으로 이동했지만, 고수준 플래너의 저주기(1–10Hz) 출력은 저수준 전신 제어(약 50Hz)와 시간 비동기를 일으킨다. 또한 헤드·손 같은 상체(upper-body)만 정의된 sparse 참조는 전신 제어에 필요한 구조적 불완전성을 남긴다. 이를 보완하려는 프레임 추정 기반 방법은 end-to-end 학습을 깨고 추가 오차를 만들며, 로컬 증분(velocity/relative keypoint)으로 바꾸면 시간에 따른 누적 드리프트가 커질 수 있다.

- **Core Contribution**: 이 논문은 비동기 상체 태스크 공간 추적을 위한 프레임워크를 제안한다. ASYNC-3PT는 미래의 sparse 궤적 전체와 실행 단계의 time index를 함께 조건화해, 명시적 프레임 추정 없이도 계획 시점 기준 참조와 현재 상태의 어긋남을 암묵적으로 정렬하도록 학습한다. 이후 ASYNC-CA는 MPC로 플로팅 베이스와 상체 관절을 sparse 참조에서 ‘완성(completion)’해 포스트트레이닝 시 RL의 불안정과 드리프트를 줄이는 방향으로 확장한다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 플래너 참조가 계획 시점의 base frame에 묶여 있는데 로봇은 여러 제어 스텝 동안 움직여 오차를 직접 측정하기 어렵고, (2) 상체만 주어진 참조가 전신 제어에 필요한 제약을 충분히 제공하지 못한다는 점이다. 저자들은 ASYNC-3PT에서 time-index 조건화와 sliding-window global reward로 구간 내 누적 드리프트를 완화하고, frame mismatch를 추정하는 별도 estimator는 ‘변환 적용’이 아닌 보조 정규화 신호로만 사용한다. ASYNC-CA에서는 MPC 완료 참조를 관측과 보상에 모두 포함하고, post-training 동안 action-level 및 FK-level self-guidance 정규화로 하체의 분포 드리프트까지 억제한다.

- **Empirical Impact**: 시뮬레이션과 Unitree G1 하드웨어에서 업데이트 저주기가 큰 환경에서도 추적 성능이 향상되며, 비동기 interval 오류(1s 구간 기준)와 성공률이 동기화(synchronous) 또는 분리(decoupled) 기반 대안보다 우수하게 나타났다. 또한 time index를 제거하면 성공률은 크게 변하지 않지만 비동기 구간 위치·회전 오차가 증가해, time index가 주로 구간 내 위상/정렬 불일치를 해결함을 보여준다. 포스트트레이닝에서는 MPC-completed guidance가 OOD(분포 외) 동작 적응에서 zero-shot 및 sparse 3-point 참조 기반 대비 더 높은 성공률과 안정성을 제공하며, 하드웨어에서도 안전한 적응 경향이 보고된다.



### SA-LIVO: Efficient LiDAR-Inertial-Visual Odometry with Subspace-Aware Degeneracy Handling (https://arxiv.org/abs/2606.25699)
Comments:
          20 pages, 12 figures, 5 tables

- **Prior Approaches**: 기존 LiDAR-visual-inertial odometry(LIVO)는 LiDAR의 기하 제약과 카메라의 시각 정보를 동시에 최적화해 정밀도를 높이지만, 센서별 실패가 동시 발생하면 분기(방향성 불일치) 때문에 성능이 급격히 무너질 수 있다. 기존 대응은 이진 degeneracy detection(모달리티 단위 거부/제한), covariance inflation·scalar down-weighting(등방성 가중치 조절), scene-level quality gating(업데이트 여부만 결정)처럼 모달리티 레벨에 머물러 joint information matrix의 방향별(축별) 구조를 다루지 못한다. 그 결과, LiDAR가 잘 관측하는 방향에는 불필요한 visual residual이 들어가고, 반대로 LiDAR가 부족한 방향에 시각 보정이 집중되지 못해 보상이 분산되는 문제가 남는다.

- **Core Contribution**: 본 논문은 방향별로 ‘어떤 축(고유방향)이 관측 가능한가’를 기준으로 LiDAR와 visual의 기여를 선택적으로 합치는 SA-LIVO를 제안한다. 핵심은 Subspace-Aware Information Fusion(SAIF)으로, joint LiDAR-visual information matrix를 eigendecompose한 뒤 각 eigendirection에 대해 linear-clamp soft gate를 적용해 degenerate 방향은 감쇠하고 observable 방향은 강도를 보존한다. 또한 LiDAR 잔차와 visual 잔차를 동일한 선형화 지점에서 한 번의 InEKF 루프로 함께 최적화해 모달리티 간 선형화 불일치와 반복 비용을 줄인다.

- **Technical Challenges**: 방향별 degeneracy를 다루려면 공분산/정보를 단순 스칼라로 줄이지 않고 고유방향 스펙트럼 기반으로 fusion을 설계해야 하며, 동시에 InEKF에서 잔차와 Jacobian을 호환되게 구성해야 한다. SAIF는 eigendirection별 정보 크기를 단일 임계값으로 게이트해 positive semi-definite fused 정보를 보장하면서도 모드 스위칭 없이 연속적으로 열화되도록 만들고, LiDAR-visual residual은 unified single-loop로 한 선형화 지점에서 풀어 방향성 보상을 일관되게 한다. 성능을 위해 photometric Jacobians는 전파된 상태에서 한 번만 계산해 InEKF 반복(iteration) 전반에 재사용하고, visual 정보는 per-observation decorrelation과 scene-level VIO quality factor로 편향·불안정성을 제어한다.

- **Empirical Impact**: SA-LIVO는 HILTI’22, New College, Oxford Spires 등 3개 벤치마크에서 29개 시퀀스를 평가하고, 동시 열화(concurrent-degradation) 시나리오에서도 기존 강한 베이스라인과 비교 가능한 정확도를 보이며 발산 지점에서는 drift가 bounded 형태로 유지됨을 보고한다. 연산 측면에서도 GPU 없이 노트북 CPU에서 프레임당 평균 12.3ms, 임베디드 ARM 보드에서 26.8ms를 달성했고 peak memory는 3.6~6.3배 낮췄다. 저자들은 코드 오픈소스를 예고해, 방향성-aware 융합 아이디어가 LIVO·multisensor SLAM 실전 시스템으로 확산될 가능성이 크다는 점에서 의미가 있다.



### Power-Budgeted Underwater Vehicle Control via Constrained Reinforcement Learning (https://arxiv.org/abs/2606.25680)
Comments:
          10 pages, 10 figures

- **Prior Approaches**: 기존 연구는 RL을 활용해 AUV의 station-keeping과 trajectory tracking을 달성했지만, 보상에 task 정확도만 두면 진동성(oscillatory) 제어로 에너지를 낭비하기 쉽다는 문제가 제기돼 왔다. 에너지 절감은 흔히 reward에 action-effort 또는 energy penalty를 가중해 scalarize하는 방식으로 다뤄졌지만, 이 가중치는 물리 단위를 가지지 않아 목표 전력(예: 몇 W)을 사전에 지정하기 어렵고 차량·과업마다 수동 재튜닝이 필요하다. 더 나아가 가중치가 맞지 않으면 오히려 task-only 대비 전력 사용이 늘어날 수 있어 신뢰성이 떨어진다.

- **Core Contribution**: 이 논문은 에너지 효율을 reward 페널티가 아닌 “명시적 예산 제약”으로 정식화한다. 평균 thruster power가 물리 단위의 budget을 넘지 않도록 constrained MDP(CMDP)로 만들고, PPO-Lagrangian(PPO-Lag)으로 학습해 dual variable을 온라인으로 적응시킴으로써 차량·과업별 목표 전력을 튜닝 없이 맞추는 경로를 제안한다. 핵심은 고정된 무단위 에너지 가중치가 아니라, 측정된 power에 직접 반응하는 제약 기반 적응형 최적화라는 점이다.

- **Technical Challenges**: 제약을 세우더라도 실제 thruster 전기 전력(방향 비대칭·비선형 power-law)을 비용 신호로 정확히 모델링하고, 그 신호를 기반으로 policy가 예산을 만족하도록 학습을 안정화해야 한다. 논문은 thruster datasheet 기반의 전력-추력 비선형을 통해 step 단위 power cost와 평균 전력 예산 위반량을 정의하고, PPO의 업데이트에 cost advantage(예산 대비 초과 여부)를 결합한 뒤 dual ascent로 multiplier를 갱신한다. 그 결과, 초기에는 예산을 초과했다가 λ가 올라가며 제약을 강하게 걸고, 이후 예산 내 운영을 학습하면 λ가 내려가며 수렴하는 적응 거동을 보인다.

- **Empirical Impact**: MarineGym 시뮬레이션에서 3대(BlueROV, BlueROV-Heavy, HAUV)×4과업(hover, lemniscate/circle/spiral tracking) 총 12개 설정 모두에서 PPO-Lag가 thruster power를 최저로 만들며 task-only baseline 대비 14–65% 감소(최대 64.9%)를 달성했다. 또한 action smoothness는 12개 중 10개 설정에서 가장 좋았고, success rate와 tracking error는 대부분 유지(또는 일부 개선)되었다. 유일한 명확한 트레이드오프는 HAUV hover처럼 전력이 충분히 허용되지 않는 구간에서 정밀한 station-keeping을 일부 포기해 60.7% 전력 절감과 56% success가 함께 나타난 경우이며, 이는 “제약이 드러낸” 의도된 선택지로 해석된다.



### Learning to Adapt: Reptile-D-Learning for Robust and Efficient Control Under Parametric Uncertainty (https://arxiv.org/abs/2606.25659)
- **Prior Approaches**: Learning-based Lyapunov Control(LLC)은 Lyapunov 조건을 신경망으로 학습해 안정성 보장을 제공하지만, 이는 특정 명목 동역학에 강하게 의존한다. 그래서 질량·마찰·하중 같은 물리 파라미터가 변하면 안정성 인증이 깨지고, 재학습 비용이 커진다.
모델 비종속을 노린 D-learning은 D-network로 Lyapunov 미분을 추정해 의존성을 줄이지만, 단일 태스크 또는 좁은 분포에서 학습되어 큰 파라미터 shift에서는 미분 추정이 틀어져 성능과 안정성 제약이 함께 저하된다.

- **Core Contribution**: 이 논문은 파라미터가 다른 여러 제어 태스크를 대상으로, 공통 동역학 구조를 잘 담는 Lyapunov/D/policy의 메타 초기화(Theta)를 Reptile로 meta-learn하는 Reptile-D-learning을 제안한다. D-learning을 태스크별 inner solver로 두고, Reptile을 이용해 bilevel 최적화를 1차 근사로 학습함으로써 빠른 적응과 안정성 목적을 동시에 노린다.
핵심은 “새 파라미터에 대해 전체를 다시 배우기”가 아니라, 공유 성분을 이미 학습된 초기화로 갖고 와서 residual 보정 위주로 적응하게 만드는 것이다.

- **Technical Challenges**: 가장 큰 기술적 난관은 D-learning이 Lyapunov 네트워크–D-network–정책이 서로 얽힌 cascaded loss 구조를 가져서, MAML처럼 적응 단계를 미분으로 통과하면 dense block Hessian이 필요해 계산이 사실상 불가능해진 점이다. 이에 따라 이 논문은 Hessian을 명시적으로 구하지 않는 Reptile 기반 1차 메타 업데이트를 설계한다.
또한 동역학을 공유 성분과 태스크별 잔차로 분해했을 때, 메타 업데이트가 태스크 간 gradient consistency를 보존한다는 수준의 gradient-level 분석을 제공해 “공통 구조를 중심으로 학습이 진행된다”는 정당성을 뒷받침한다.

- **Empirical Impact**: 세 가지 비선형 제어 벤치마크(역진자, CommonRoad 기반 single-track car, Crazyflie 3D UAV)에서 Reptile-D-learning은 일반화와 빠른 적응 모두에서 표준 D-learning 대비 일관된 개선을 보였다. 파라미터 shift(OOD) 상황에서 기존 방법은 수렴 실패/발산이 두드러지는 반면, 제안 방법은 더 높은 안정화 성공률과 더 빠른 수렴, 더 낮은 최종 추종 오차를 달성했다.
특히 OOD 조건에서 제한된 몇 번의 적응 단계(few-step)만으로 안정 성능을 회복하며, UAV에서는 10회 내 고수준 성능 도달이 관찰된 반면 표준 D-learning은 훨씬 더 많은 반복이 필요했다. 마지막으로 GPU 메모리 관점에서 MAML급 2차 메타학습은 실험이 사실상 불가능한데, Reptile은 메모리 비용을 크게 낮춰 실용성을 강화했다.



### Calousel: Extrinsic Calibration of Non-overlapping Multi-camera Systems from Pure Rotation (https://arxiv.org/abs/2606.25646)
Comments:
          Accepted to IROS 2026. 8 pages, 7 figures

- **Prior Approaches**: 기존 다중 카메라 외부 파라미터 보정은 FOV가 겹치는 경우 feature matching으로 해결되지만, 겹치지 않는 경우 직접 대응점이 없어 난도가 커진다. 대표적으로 타깃 기반은 큰 캘리브레이션 보드나 다중 타깃을 써야 하고, 미리 측정된 6D 포즈 요구로 센서 수가 늘수록 설치 오버헤드가 커진다. 모션 기반은 visual SLAM/SFM 기반이라 drift, scale ambiguity, motion degeneracy에 취약하고, 정밀 모션 하드웨어를 쓰는 하이브리드도 결국 세팅 복잡도가 남는다.

- **Core Contribution**: 이 논문은 non-overlapping FOV 환경을 겨냥해 단일 고정 캘리브레이션 보드와 single-axis turntable의 순수 회전만으로 카메라 간 extrinsic calibration을 수행하는 방법을 제안한다. 모든 카메라가 시간차로 같은 타깃을 관측하되, latent turntable frame을 두고 SE(3) 상의 3D 오차를 전역 최적화에 넣어 순차 관측을 하나의 기하 기준으로 통합한다. 그 결과, 정밀 장비 없이도 현장(on-site) 재보정에 적합한 “공간 효율 + 정확도”를 동시에 노린다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 겹치지 않는 FOV에서 시간차로 얻은 관측을 일관된 기준으로 합치는 것, (2) 순수 회전만 있을 때 생기는 gauge ambiguity와 최적화 불안정성을 다루는 것이다. 저자들은 latent turntable frame을 도입해 카메라 마운팅의 시간 불변 기하와 회전에 따른 시변을 분리하고, 기준축/원점 제약으로 gauge freedom을 줄인 뒤 global optimization에서 SE(3) 3D error를 최소화한다. 또한 rolling-shutter로 인한 타깃 기반 pose 추정 편향을 lightweight 보정 모듈로 완화하고, 강한 운동학 제약에서 생기는 공분산 상관 문제는 가중치 단순화(대각 성분만 사용)로 안정화한다.

- **Empirical Impact**: 제안 방법은 제어된 카메라 rig과 이질적 카메라가 장착된 full-scale vehicle 플랫폼에서 모두 실험적으로 검증되며, 현실적인 turntable 비이상(축 흔들림/진동)에도 경쟁력 있는 정확도를 유지한다. 특히 specialized precision hardware 없이도 정확도가 확보된다는 점을 강조하며, 캘리브레이션 세팅의 실용성을 입증한다. 또한 rolling-shutter compensation과 latent frame/3D error 같은 설계 선택의 타당성을 별도 실험으로 분석해, 현장 적용 시 성능 변동 요인을 줄이는 방향성을 제시한다.



### Event-Adaptive Motion Planning with Distilled Vision-Language Model in Safety-Critical Situations (https://arxiv.org/abs/2606.25629)
Comments:
          8 pages, 8 figures, 4 tables. Accepted by IROS 2026

- **Prior Approaches**: 기존 로봇 내비게이션은 최적화 기반 MPC(기하학 제약 중심)나 end-to-end VLM/정책 모델(고비용)로 접근해 왔지만, 언어 수준의 의미 추론을 통해 행동을 ‘상황별로’ 바꾸는 데 한계가 있다. 또한 LLM/VLM을 주기적으로 호출하거나 객체 미지/신뢰도 하락 같은 휴리스틱으로 개입을 트리거해도, 흔한 에이전트의 돌발 행동 변화(예: 보행자 갑작스런 무단횡단)를 늦게 잡아 안전 여유를 깎을 수 있다.

- **Core Contribution**: 이 논문은 안전-치명적 코너케이스에서 VLM의 의미 추론을 ‘이벤트 기반’으로만 개입시키는 event-adaptive motion planning(EAMP)을 제안한다. 핵심은 prompt-configurable semantic event trigger(PC-SET)로 행동 레벨 이상을 감지하면, distillation된 SemNav-VLM이 전략-level 결정을 내리고 semantic model predictive control(SMPC)이 목표함수와 기준선(geometric reference)을 구조적으로 재구성해 저수준 제어 안정성을 유지하는 구조다.

- **Technical Challenges**: 문제는 연속 제어 루프에 VLM 추론을 자주 넣으면 지연(latency)이 커져 물리 실행이 불안정해진다는 점인데, 이를 해결하기 위해 PC-SET는 짧은 시계열 클립을 모니터링해 이진 이상 지표를 만들고 시간적 일관성(슬라이딩 윈도우)으로 트리거를 안정화한다. 또 전략 추론 결과가 직접 액션을 흔들지 않도록, SMPC가 SemNav-VLM의 선택을 MPC 템플릿(보수성 수준/참조 경로)로 매핑하고, SemNav-VLM은 Gemini-3.0-Flash 교사를 physically verified semantic distillation로 필터링해 동적으로 실행 가능한 라벨만 학습하도록 설계했다.

- **Empirical Impact**: CARLA 기반 안전-물류(주행/야간·도심 상호작용, 긴 가림/불확실성) 실험에서 EAMP는 기존 RDA/PCS/OCP 대비 TTC 등 동적 안전 여유를 유의미하게 개선하면서도 실시간성을 보존했다. 예를 들어 hard 시나리오에서 TTC가 OCP·PCS 대비 각각 32%/30% 향상되고, 속도 변동도 더 낮아(더 매끈한 제어) 트리거 기반 의미 개입이 안전-효율 트레이드오프를 개선함을 보여준다.



### 1000 Rallies: An Event-Camera Dataset and Real-Time Learned Ball-State Estimation for Robotic Table Tennis (https://arxiv.org/abs/2606.25620)
- **Prior Approaches**: 로보틱 탁구는 공의 빠른 동역학과 지연 제약 때문에, 실시간 공 상태(위치·속도·필요 시 회전) 추정과 궤적 예측이 핵심 난제로 꼽혀 왔습니다. 기존 방법은 프레임 기반 카메라의 프레임레이트-비용 트레이드오프, 또는 event camera에서 RoI에 의존하는 고전적 검출·추적이 폐색/배경 잡음에 취약하다는 한계를 보였습니다. 또 일부 학습 기반 event ball detector는 업데이트레이트나 속도 추정이 제한되어 실제 제어에 바로 쓰기 어려웠습니다.

- **Core Contribution**: 이 논문은 탁구용 event-camera 대규모 데이터셋을 최초로 제시합니다. 총 1200여 랠리(아마추어~엘리트 선수) 5시간 분량을 4대 event 카메라로 수집하고, 동기화된 다중 APS 카메라(최대 14대)로 공 position/velocity/spin에 대한 1 kHz pseudo ground-truth 라벨을 생성합니다. 또한 CNN이 이벤트로부터 이미지 평면의 공 위치와 속도를 단일 샷으로 함께 추정하도록 설계해, 이후 필터 기반 예측의 입력을 강화합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 이벤트 스트림에서 배경 선수 움직임·폐색을 견디는 공 상태 추정의 강건성, (2) APS 기반 3D 정보로부터 kHz 수준 라벨을 안정적으로 만들고, (3) 빠른 공에 맞춰 EKF가 일관된 공분산으로 궤적을 추정하도록 초기 상태를 구하는 문제였습니다. 논문은 ball-circular 파라미터화를 YOLO v4-tiny 계열 CNN에 맞춰 2D 속도(vx, vy)를 직접 회귀하고, CNN의 velocity 예측을 EKF의 추가 측정으로 넣어 bounce-point 오차를 줄였습니다. 더불어 배경과 공간적 변환에 대한 일반화를 위해 물리적 일관성을 유지하는 데이터 증강과, C++/TensorRT 기반 병렬 추론으로 실시간 파이프라인을 최적화했습니다.

- **Empirical Impact**: 검출 품질 측면에서, 제안 모델은 Ziegler et al. 데이터셋에서 MAE 0.91 pixels, IoU 0.78의 성능을 보이며 기존 event 기반 접근을 능가합니다(파인튜닝·입력 수정 없이 비교). 궤적 예측에서는 EKF에 이미지 평면 velocity 측정을 추가하는 방식이 bounce-point prediction error를 36% 줄였고(100 ms 단위 기준), 업데이트레이트 변화에 따른 정확도 영향도 분석했습니다. 마지막으로 Stäubli TX2-60L 로봇에 perception-action loop를 통합해, event-based 인식에 기반한 실시간 인간-로봇 탁구 랠리를 실제로 시연하며 해당 분야의 실용성과 연구 기반을 동시에 확장했다는 점에서 의미가 큽니다.



### WOLF-VLA: Whole-Body Humanoid Optimal Locomotion Framework for Vision-Language-Action Learning (https://arxiv.org/abs/2606.25591)
- **Prior Approaches**: 기존 로봇 제어는 Optimal Control(OC) 기반이 안정성과 안전 제약을 잘 다루지만, 환경 변화 적응에는 추가 모듈이 필요했다. Vision-Language-Action(VLA) 계열은 일반화가 강해지고 있으나, 대부분 fixed-base 조작 중심이어서 전신·다중 접촉·고동적 휴머노이드 보행에는 데이터와 검증 체계가 부족했다.
또한 텔오퍼레이션이나 모션 캡처 기반 데이터는 비용이 크고, 최적성(에너지 효율·관절 최소화)과 동역학적 일관성, 안전 제약을 데이터 자체에 내재하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 WOLF-VLA로, 휴머노이드 전신 보행을 자연어 지시에서 바로 생성하는 VLA를 만들기 위해 “전체를 OC로부터 생성한 동역학 일관 궤적”을 학습 데이터로 사용한다. OC가 생성하는 궤적은 매끄럽고 제약을 만족하며, 그 결과로 학습되는 정책이 최적성에 가까운 행동 품질과 초기 조건 변동에 대한 강건성을 보이도록 한다.
또한 6개 보행 태스크 패밀리와 다양한 환경 변형을 포함한 대규모 벤치마크를 제안해, VLA 관점에서 휴머노이드 보행을 체계적으로 평가할 수 있게 했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 휴머노이드 전신·접촉이 풍부한 “동역학적으로 타당한” 대규모 데모를 확보하고, (2) 그 데모의 최적성·안전성을 학습 파이프라인에 연결하며, (3) VLA가 시각·언어·자세 정보를 함께 이용해 보행 정책을 생성하도록 학습하는 것이다.
저자들은 MuJoCo 시뮬레이터에서 다중 페이즈 multi-body OC 문제를 Differential Dynamic Programming(DDP)으로 풀어 접촉 전환과 관절/토크 제약을 만족하는 궤적을 생성하고, RGB(ego-centric)·고정밀 고유 운동(proprioception)·자연어를 함께 넣어 flow matching 기반 행동 생성 모델을 학습해 해결한다.

- **Empirical Impact**: 생성된 WOLF-VLA-dataset(총 277시간 규모)는 전방 보행, 측면 보행, 계단 오르기/내리기, 180도 회전, 가변 높이 스쿼팅 등 6개 태스크를 포괄하며, 작업 성공률과 관절 ROM error 및 계단의 soft success rate로 성능을 평가했다. 기존 baseline들과 비교했을 때 여러 태스크/환경 설정에서 경쟁력 있는 성능과 강건성을 보였고, 명령문 공간 태깅·언어·시각 등의 모달 기여를 제거/변형하는 ablation으로 효과를 체계적으로 확인했다.
또한 데이터셋, 모델 체크포인트, 시뮬레이션 벤치마크를 공개해 “동역학적으로 일관된” 휴머노이드 전신 보행 VLA 연구를 재현 가능하게 만들고, 지시 기반 보행 정책의 확장 전이 연구를 촉진할 것으로 기대된다.



### One Body, Two Minds: Variable Autonomy Approach for a Co-embodied Robotic Hand (https://arxiv.org/abs/2606.25575)
- **Prior Approaches**: 기존 보조로봇은 완전 자율(사용자 agency 약화) 또는 전면 수동(지속적 인지부담과 조작 난도 증가)으로 양분되는 한계가 있었다. shared autonomy는 인간-로봇 의도를 혼합하며 협업하지만, 대개 서로 다른 신체(조이스틱/원격 입력 등)에서 제어해 물리적 결합과 상황 인지가 직접적이지 않다는 문제가 있었다. 또한 제어 협상이 연속적으로 일어나 “언제 누가 무슨 일을 하는지”의 정신모델이 흐려질 수 있다.

- **Core Contribution**: 이 논문은 co-embodiment with variable autonomy를 제안한다. 인간과 로봇이 한 몸(단일 물리적 핸드)에 함께 결합하되, 작업 단계별로 자율 수준을 바꿔 object search·grasping 구간에서는 mutual autonomy(두 minds가 자율로 협력), actuation 이후에는 human-dominant 제어로 전환한다. 이로써 one body, two minds 패러다임을 구현하고, release 제스처로 언제든 초기 단계로 되돌려 사용자의 veto authority를 상시 보장한다.

- **Technical Challenges**: 핵심 기술 난제는 co-embodied 환경에서 learning-based visuomotor policy가 사용자의 움직임(공간 위치 변화)에도 안정적으로 동작하며, grasping 참여 시점 전환을 매끄럽게 만드는 것이다. 논문은 learning-from-demonstration visuomotor diffusion policy를 사용해 사용자가 손을 알려진 물체 근처로 가져오면 조건을 만족할 때 자율로 grasp를 시작하도록 설계했다. 또한 grasp 이후에는 audible feedback와 head gesture(nod/ shake)로 actuation과 release를 직관적으로 트리거하게 만들어, 연속적인 제어 협상 대신 단계 기반 자율 전환 문제를 단순화했다.

- **Empirical Impact**: 44명의 참여자가 5개의 양손 작업을 3회 수행한 사용자 연구에서, 완료 시간은 Trial 1~3 사이 23.3% 개선되었고 학습 효과가 분명하게 관측됐다(p<0.001, Cohen’s d=0.94). 최적 정책 변형은 task success 93.6%를 달성했으며, 전체 수용도도 높아 전반 인상 5.70/7, 일상 사용 의향 5.52/7 수준을 보였다. co-embodiment with variable autonomy가 보조로봇에서 “빠른 적응 + 신뢰 가능한 단계 전환”을 제공하는 실행 가능한 협업 방식임을 실험적으로 뒷받침했다.



### ASSCG: Just-Right Gating over Chattering for Fast-Slow LLM Planning in Autonomous Driving (https://arxiv.org/abs/2606.25509)
- **Prior Approaches**: 기존 fast–slow 드라이빙 플래너는 매 프레임 또는 장면 단위 고정 주기로 slow(LLM) 모듈을 호출하거나, scene complexity 같은 휴리스틱/불확실성으로 호출 타이밍을 조절하는 방식이 많았습니다. 그러나 고정 스케줄은 장면 내 시간적 변동을 놓치고, 난이도 프록시는 ‘해당 시점에 slow를 호출했을 때의 한계 효용’을 잘 반영하지 못해 성능 저하나 비효율이 발생했습니다.

- **Core Contribution**: 이 논문은 slow-system invocation을 프레임 레벨의 자원 제약 sequential decision 문제로 정식화하고, Adaptive Slow-System Control Gate(ASSCG)를 제안합니다. ASSCG는 각 프레임에서 Query/Cache/Drop을 선택해 slow 지침을 갱신·재사용·의도적으로 억제함으로써 닥치는 대로 호출하는 문제를 해결합니다. 또한 RWKV 기반 게이트를 설계하고, supervised fine-tuning 이후 compute-aware GRPO-style reinforcement fine-tuning으로 닫힌루프 성능과 호출 비용을 함께 최적화합니다.

- **Technical Challenges**: 핵심 기술 난제는 부분 관측(POMDP) 환경에서 slow 지침의 ‘가치’가 시간에 따라 달라지는 상황을 프레임 단위로 학습하는 것입니다. 논문은 이를 위해 Equivalent Interval(중복 호출 구간), Effective Interval(캐시 유효 구간), Failure Interval(호출이 해를 주는 구간)이라는 운영 기준을 도입하고, 그에 대응하도록 Cache는 연장 사용, Drop은 실패 구간 억제 기능을 갖는 3-way 게이트를 학습합니다. 게이트의 긴 문맥을 효율적으로 처리하기 위해 Transformer의 제곱 복잡도 대신 RWKV로 low-latency long-horizon 결정을 구현합니다.

- **Empirical Impact**: ASSCG를 AsyncDriver에 결합한 AdaptiveAsyncDriver는 nuPlan Hard20 closed-loop에서 점수 67.28(+2.28)로 개선되면서 평균 end-to-end inference latency는 60% 줄였습니다. 또 RecogDrive 기반 dual-system(NAVSIM)에서도 PDMS 91.4(+0.6), 평균 속도 약 25% 향상을 보이며 서로 다른 아키텍처에서도 동일 원리의 효율-정확도 이득이 재현됨을 확인했습니다. 결과적으로 ‘slow를 얼마나 자주 부를지’에서 나아가 ‘어떤 구간에서 신뢰하고 언제 버릴지’를 학습하는 접근이 fast–slow planning의 실사용 제약을 더 직접적으로 줄였다는 의미가 있습니다.



### GROVE: Grounded Pedestrian Simulation via Natural Language for Interactive Social Robot Navigation (https://arxiv.org/abs/2606.25504)
Comments:
          Accepted at IROS 26

- **Prior Approaches**: 기존 보행자 시뮬레이터는 시나리오 정의와 시뮬레이션 실행이 분리되어, 에이전트 위치·목표·행동을 매번 수동으로 지정해야 했다. 또한 에이전트 전반에 동일한 행동 모델을 적용하거나(SFM 기반 포함) 의미 있는 목적/환경 맥락 반영이 약해 sim-to-real gap이 지속됐다. 텍스트 기반 crowd 생성은 유연하지만 로보틱스용 시뮬레이터(Isaac Sim/Gazebo/RViz)와의 연동 및 로봇/사회적 상호작용 재현은 상대적으로 부족했다.

- **Core Contribution**: GROVE는 텍스트(자연어)로부터 바로 실행 가능한 보행자 시뮬레이션을 생성하는 text-to-scenario 프레임워크를 제안한다. LLM 기반 RoI(Regions of Interest) 추출과 RAG로 행동 트리(behavior tree)를 구성하고, 시나리오 상황에 따라 모듈 내 SotA를 동적으로 선택해 장기·중기·단기 행동을 각각 현실적으로 맞춘다. 사용자는 emergency/queuing/normal 프리셋 또는 독립 프롬프트로 다양한 사회적 도전 장면을 제어할 수 있다.

- **Technical Challenges**: 핵심 과제는 (1) 언어 의도→행동 트리 노드→기하 제약으로의 정합성 확보, (2) LLM이 생성하는 BT 스키마의 환각/문맥 과적을 줄이면서도 실행 가능성을 보장, (3) SFMs만으로는 정적 장애물 충돌을 하드하게 막기 어려운 점이다. GROVE는 world.yaml로 공간·의미를 구조화해 언어를 기하에 접지하고, RAG로 필요한 노드 사양만 주입해 유효한 BT 생성을 돕는다. 동시에 Theta* 기반 글로벌 플래닝으로 정적 장애물 회피를 “waypoint 주입” 형태로 BT 레벨에 보증하며, 에이전트/목표 ID 관리 모듈로 대규모 에이전트의 기호적 일관성도 유지한다.

- **Empirical Impact**: Isaac Sim/Gazebo/RViz에 직접 통합해 로봇 배치를 지원하며, 주거·병원·오피스 환경의 서로 다른 복잡도에서 기존 Text-Crowd와 TRACE 대비 정성/정량 평가를 수행했다. VLM 기반 평가(GPT-5로 Prompt Alignment/Plausibility/Visual Realism을 0–10 스케일 채점)에서 GROVE가 전체 평균 최고 점수를 기록했고, 특히 RoI 기반 시작·목표 할당 덕분에 Prompt Alignment가 가장 크게 개선됐다. 또한 queuing/exit 흐름 등 구조화된 군집 행동과 stop-and-go 패턴을 더 일관되게 생성해, 상징적 과업 구조와 기하적 인증을 결합하는 접근의 효과를 실증했다.



### AISPO: Enhancing Depth Reliability for Robotic Manipulation of Non-Lambertian Objects via Affine-Invariant Shape Prior (https://arxiv.org/abs/2606.25503)
Comments:
          Published in IEEE Robotics and Automation Letters. 8 pages. Accepted April 2026

- **Prior Approaches**: 기존 depth completion 및 단일/다중 시점 depth 추정 연구는 RGB guidance나 geometric constraint로 결손·잡음을 보정하지만, RMSE/MAE 같은 평균 오차 최소화에 치우쳐 로봇 조작에서 치명적(depth artifact) 실패를 충분히 막지 못하는 한계가 있었다. 투명·투과·고반사(non-Lambertian) 표면에서는 센서 잡음과 누락이 커서 잘못된 깊이가 motion planning으로 전파되며, NeRF·최적화 계열은 품질은 좋아도 지연과 계산 부담이 커 실시간 조작에 불리했다. 또 단안(depth from monocular)은 metric 입력이 없어 절대 깊이 정확도와 구조 보존이 흔들린다는 문제가 남아 있었다.

- **Core Contribution**: AISPO는 비라미안 객체에서 depth의 ‘신뢰도’를 높이기 위해 multi-scale RGB-D feature fusion과 affine-invariant shape prior를 결합한 depth completion 프레임워크를 제안한다. 평균 오차 개선보다 물리적으로 말이 되는(depth map의 구조 무결성) 기하 일관성을 우선하며, affine 불변 정규화로 객체별 scale/shift 변이를 흡수해 예측의 안정성을 강화한다. 또한 shape-prior autoencoder를 두 단계 학습으로 구성해 예측 깊이의 구조적 무너짐(catastrophic failure)을 줄이는 데 초점을 둔다.

- **Technical Challenges**: 주요 기술 난제는 투명·고반사 표면에서 원시 depth가 심하게 깨지거나 픽셀이 누락된다는 점이며, 동시에 RGB만으로는 metric geometry를 직접 복원하기 어렵다는 점이다. AISPO는 DINOv2 기반 RGB 인코더, Swin-Transformer 기반 raw depth 인코더, 그리고 affine-invariant shape prior 인코더를 병렬로 두고 cross-attention으로 멀티스케일 정보를 융합해 구조를 유지하도록 설계했다. 학습에서는 L1 loss에 masked loss와 Sobel gradient loss를 더해 객체 경계의 선명도와 기하 정합성을 강화하고, shape prior를 통해 결손 깊이를 보정하도록 했다.

- **Empirical Impact**: DREDS-CatKnown, STD-CatNovel, ClearPose, ClearGrasp 등 합성/현실 벤치마크에서 AISPO는 기준선 대비 깊이 예측의 구조 보존과 일관성을 높이며, 특히 unseen 객체·새 장면에 대한 zero-shot 일반화에서도 강점을 보였다. 더 중요한 실험으로 Franka 로봇 팔 grasping을 수행한 결과, 기존 방법들은 투명 물체에서 잘못된 depth로 인해 그립 제안이 무효가 되어 실패했지만 AISPO는 grasp success rate를 크게 끌어올렸다. 또한 RTX 3090에서 28.79 ms/frame의 추론 지연을 보고해 조작 시나리오에서 속도-정확도 균형도 현실적으로 확보했다.



### SAGE-Nav: Leveraging LLM Planning and Alignment Fusion for Hierarchical Scene Graph-Guided Navigation (https://arxiv.org/abs/2606.25497)
Comments:
          Accepted by IROS 2026

- **Prior Approaches**: 기존 Object-Goal Navigation(ObjNav)은 단일(end-to-end) 모델이 긴 시간 범위의 추론과 제어를 함께 처리하는 방식이 많아 장기 추론에서 쉽게 흔들리고, 새로운 환경으로의 일반화 성능도 제한적이라는 문제가 제기돼 왔다. 또한 글로벌 의미 계획과 저수준 반응 제어가 같은 흐름에 묶여 있어, 고주파 제어가 요구되는 상황에서 제어 지연(latency) 문제가 발생하기도 한다.

- **Core Contribution**: 이 논문은 SAGE-Nav이라는 계층형 프레임워크로, Large Language Models(LLMs)의 추론 능력과 dynamic scene graphs를 결합해 ObjNav의 장기 추론·일반화 한계를 겨냥한다. 핵심은 비동기 글로벌 의미 계획을 고주파 반응 제어 루프와 분리해, LLM이 추상 지시를 semantically grounded waypoint 시퀀스로 분해하고 저수준 정책은 그 계획을 빠르게 반영하도록 설계한 점이다.

- **Technical Challenges**: 기여를 실제로 성립시키려면(1) LLM 계획을 장면의 의미·공간 관계에 정합적으로 내재화하고, (2) 실시간 지각 신호와 구조적 priors가 충돌하지 않게 정렬(alignment)하는 문제가 난이도 높다. 논문은 Hierarchical Scene Graph Encoder(HSGE)로 relational graph convolutions 기반 구조 인코딩을 만들고, Goal-aware Alignment-Fusion Network(GAFN)에서 적응형 gating과 명시적 귀납적 편향으로 실시간 perception과 토폴로지/의미 정보를 안정적으로 융합해 저수준 정책의 시각-위상 정합성을 확보한다.

- **Empirical Impact**: i-THOR과 RoboTHOR에서 SAGE-Nav은 state-of-the-art 성능을 보이며 경로 탐색 효율(네비게이션 efficiency)에서 유의미한 개선을 달성했다. 특히 zero-shot generalization에서 강점을 보이면서도 물리 로봇 배치를 전제로 한 저수준 제어의 지연 요구사항을 유지한 점이 실용적 의미를 가진다.



### Generative AI for Safe and Photorealistic Drone Light Shows (https://arxiv.org/abs/2606.25458)
- **Prior Approaches**: 드론 라이트 쇼는 자동화 가능성이 크지만, 현재는 사람이 직접 애니메이션을 제작하는 노동집약적 과정이 확산의 병목으로 작용한다. 생성형 AI를 적용한 시도도 있으나, 사진 수준의 사실감과 유체적이며 동적으로 변하는 모션을 함께 만족시키는 워크플로가 부족했다. 또한 가시성 저하(occlusion)나 형태 변화(topology shift)가 심한 상황에서 안정적인 추적과 궤적 일관성을 유지하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 텍스트 프롬프트만으로 사진 같은 대규모 드론 군무를 생성하는 end-to-end 파이프라인 SWAN을 제안한다. SWAN은 먼저 텍스트를 바탕으로 현실적인 레퍼런스 영상을 만들고, 이를 픽셀 공간의 동역학에서 물리적 swarm kinematics로 변환한다. 이어서 드론별 궤적을 배분하는 planner와 충돌을 막는 safety filter로 실제 비행 실행까지 연결한다.

- **Technical Challenges**: 핵심 기술 난제는 강한 가림과 빠른 구조적 변화가 발생하는 장면에서조차 객체의 공간적 일관성을 유지하며 point cloud/픽셀 상의 움직임을 안정적으로 추적하는 것이다. SWAN은 기존 tracker와 달리 severe occlusions와 급격한 topological shifts에서도 spatial coherence를 보존하는 novel, adaptive point-tracking 알고리즘을 도입해 동역학을 물리 궤적으로 변환한다. 그런 다음 planner가 개별 드론에 경로를 할당하고, safety filter가 실행 단계에서 collision-free 조건을 보장하도록 설계했다.

- **Empirical Impact**: 실험에서는 시뮬레이션으로 2,000대 규모의 형상을 충돌 없이 안전하게 오케스트레이션하며 확장성을 입증했다. 더 나아가 실제 환경에서 49대 쿼드콥터 밀집 스웜을 대상으로 물리적 실행 가능성을 검증했고, 전체 파이프라인을 표준 소비자 하드웨어에서 수행할 수 있음을 보여줬다. 결과적으로 SWAN은 생성형 AI를 이용한 multi-robot choreography design 자동화에 대한 접근성을 높이며 드론 라이트 쇼 제작 흐름을 바꿀 새 프레임워크로 의미가 있다.



### Delta-Position Estimation-Based IMU Odometry: A Comparison of MLP and Kolmogorov-Arnold Networks (https://arxiv.org/abs/2606.25454)
Comments:
          This study was presented at the 11th International Congress on Engineering Sciences and Multidisciplinary Approaches, held in Istanbul, Türkiye

- **Prior Approaches**: 기존 학습 기반 관성 항법(inertial odometry) 연구는 흔히 절대 위치 회귀(absolute position regression)로 바로 자세/위치를 예측하려는 접근을 사용해왔다. 이 방식은 상수 오차가 크게 남아 궤적 전체에서 누적 드리프트가 커질 수 있다는 한계가 있다. 또한 모델 구조 측면에서 파라미터 효율과 장기 오차 누적 안정성을 동시에 만족시키는 설계가 쉽지 않다.

- **Core Contribution**: 본 논문은 EuRoC MAV 벤치마크의 raw IMU를 사용해 관성 항법을 다루되, 절대 위치 회귀 대신 50ms 슬라이딩 윈도우에서의 증분 변위(incremental displacement) Δp를 추정하도록 문제를 재정의한다. 예측된 Δp는 수치 적분으로 전체 궤적을 재구성해, 상수 오차가 누적되는 경로를 줄이려는 전략이다. 또한 표준 MLP와 KAN(Kolmogorov-Arnold Network)에서 learnable B-spline 활성화를 비교한다.

- **Technical Challenges**: 핵심 기술 과제는 IMU 잡음과 작은 추정 오차가 수치 적분 과정에서 장기적으로 드리프트로 증폭되는 문제를 어떻게 완화하느냐이다. 저자들은 출력 목표를 Δp로 제한하고, KAN의 learnable B-spline 기반 활성화로 모델이 더 유연한 함수 근사를 하게 함으로써 오차 축적의 안정성을 높인다. 그 결과 KAN은 파라미터 수를 줄이면서도(8,444 vs 57,859) 예측 품질을 개선하는 방향으로 설계를 맞췄다.

- **Empirical Impact**: 실험에서는 테스트 궤적의 최종 누적 드리프트 기준으로 KAN이 MLP보다 44% 낮은 오차를 보였고(9.61m vs 17.23m), 장기 오차 누적에서도 P_50과 P_90 누적 드리프트가 더 낮아 더 안정적인 거동을 보였다. 이는 learnable B-spline 활성화가 관성 항법에서 시간에 따른 오차 전파를 완화할 실마리를 제공함을 시사한다. 결과적으로 inertial odometry에서 정확도뿐 아니라 장기 안정성을 함께 개선하는 모델링 방향을 제안하는 연구로 해석된다.



### HEART: Coordination of Heterogeneous Expert Agents for Physically Grounded Robotic Task Planning (https://arxiv.org/abs/2606.25404)
Comments:
          9 pages, 3 figures

- **Prior Approaches**: 기존 LLM 기반 로봇 플래너는 자연어를 행동 시퀀스로 직역하거나, PDDL/temporal logic 같은 형식으로 번역해 논리 일관성을 보완하려는 시도가 많았다. 하지만 단일 LLM은 능력/도달가능성/물리 제약/행동 순서를 한 맥락에서 동시에 검증하지 못해, 언어상 그럴듯하지만 로봇 실행 시 무효이거나 불완전한 계획을 내놓는 문제가 반복됐다.

- **Core Contribution**: HEART는 instruction을 원자 단위의 추론 질문으로 분해한 뒤, capability·environment·path·feasibility·constraint 역할을 가진 이질적 expert agent에 토큰 예산 내에서 배정하는 multi-LLM 협업 프레임워크다. 이후 역할별 추론 결과를 constraint-driven plan synthesis 단계에서 교차검증해, 물리적으로 실행 가능한 계획을 생성하면서 효율도 유지하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 실행 가능성(잡기/도달/경로/시간순서)을 논리 타당성과 함께 검증해야 하는데, (2) 다중 에이전트가 동일 맥락을 중복 처리하면 토큰·지연이 폭증한다는 점이다. HEART는 Sentence-BERT 기반 의미 매칭에 history-sensitive penalty와 capacity planning(토큰 수요-여유 추정)을 결합해 질문을 “필요한 역할”에 할당하고, 각 에이전트는 역할에 맞는 필터링된 장면/로봇 데이터만 받아 문맥 길이와 중복 추론을 줄이도록 설계했다.

- **Empirical Impact**: 가정용 household 벤치마크(Gibson 기반) 3종에서 HEART는 Plan SR이 전반적으로 단일 LLM 및 규칙형 대비 더 높게 나타났고, 5개 역할 분해 구성이 특히 성능이 우수했다. 또한 토큰 예산이 제한된 조건에서도 할당 라운드와 총 토큰 사용이 줄면서 성공률이 유지되는 경향을 보여, 제한된 자원 하에서도 이질적 LLM 협업이 로봇 태스크 플래닝에 실용적으로 확장될 수 있음을 시사한다.



### MAPL: Multi-Objective Preference Learning for Robot Locomotion (https://arxiv.org/abs/2606.25398)
- **Prior Approaches**: 기존 강화학습 기반 로봇 보행은 성공적으로 보이지만, 수동으로 보상함수를 설계·튜닝해야 한다는 점이 병목입니다. LLM을 활용한 preference-based RL은 이를 완화하려 했으나, 단일 overall ranking을 요구하는 방식은 여러 목표(추적·안정·부드러움)를 한 번에 섞어 판단해야 해서 신호가 노이즈가 되기 쉽습니다. 그 결과 reward engineering 부담은 줄어도 감독 품질 저하로 학습이 불안정해지는 문제가 남아 있습니다.

- **Core Contribution**: MAPL은 Multi-Objective AI-Informed Preference Learning으로, 보상식을 직접 만들기보다 자연어 목표를 기준으로 궤적을 ‘목표별로’ 비교·학습합니다. LLM이 velocity tracking, smoothness, stability 각각에 대해 독립적으로 선호를 랭킹하고, 이를 multi-head preference scoring 모델로 학습한 뒤 선형 결합해 단일 scalar reward로 바꿉니다. 또한 potential difference shaping을 적용해 정책 최적화에 더 샘플 효율적인 학습 신호를 제공합니다.

- **Technical Challenges**: 핵심 과제는 LLM이 여러 경쟁 목표를 한 판단으로 합치면서 생기는 trade-off 불일치와 gradient 간섭을 줄이는 것입니다. MAPL은 목표를 자연어 기준으로 분해해 LLM의 비교를 objective-wise로 수행하고, 각 objective에 대한 head를 따로 학습한 뒤 가중치로만 균형을 조절해 학습 안정성을 높였습니다. 또 LLM의 랭킹 불일치를 완화하기 위해 동일 질문을 여러 번 질의하고 majority로 집계하며, policy 학습은 reward 모델 학습보다 느린 주기로 호출해 계산 비용도 관리합니다.

- **Empirical Impact**: MAPL은 4가지 사족 보행 환경(평지·불균일 지형·계단·장애물)에서 LLM이 생성한 preference만으로도 expert-designed reward와 동등하거나 더 나은 성능을 보였습니다. 특히 single generic prompt로 전 지형에 걸친 정책을 학습해 task-specific reward engineering을 사실상 제거했고, ablation과 상태 분포 관찰에서도 stability·부드러움·속도 추적이 함께 개선되는 정황이 확인됩니다. 비교 실험에서 MAPL은 속도 추적 오차와 랭킹 정확도에서도 기존 LLM 단일 선호·기존 preference 방법 대비 강점을 보여 “규모화 가능한 보상 학습”의 실용성을 시사합니다.



### Large-Scale Tunnel Air--Ground Collaboration With FLISP: Fast LiDAR-IMU Synchronized Path Plann (https://arxiv.org/abs/2606.25393)
Comments:
          24 pages, 31 figures, 5 tables. Author accepted manuscript. This work was supported by the State Key Laboratory of Autonomous Intelligent Unmanned Systems. The authors also thank the KinaMind Society for its inspiring environment and support

- **Prior Approaches**: 기존 터널/관로 점검 로봇 경로계획은 대체로 SLAM 기반의 맵 중심(global, map-based) 패러다임에 의존하거나, 맵 없이 로컬 반응(local)으로만 처리하는 방식으로 나뉜다. 그러나 특징이 반복되고 조명이 열악한 터널에서는 SLAM 드리프트가 누적되거나, 로컬 반응형이 블라인드 커브에서 위상적 데드락과 동역학 불일치를 일으키기 쉽다. 또한 다중 로봇 협업은 UAV가 글로벌 맵을 지리참조하는 구조가 많아 단일 실패 지점이 생기며, tether/학습 기반 등은 계산비용이나 일반화 한계가 보고된다.

- **Core Contribution**: FLISP는 GPS-denied 터널에서 UGV-UAV가 협업해 맵 생성 없이(mapless) 동기화된 경로를 생성하는 프레임워크다. 단일 UGV-탑재 LiDAR-IMU로 환경 기하를 중심 처리하고, UGV는 Firefly Algorithm 기반 장애물 회피와 장기(롱-호라이즌) 기하 기반 경로 생성으로, UAV는 그 결과를 계층적으로 받아 통신·고도 안전과 동역학을 반영해 비행 경로를 만든다. 특히 kinematic feasibility를 해치지 않으면서 상태추정 drift에 덜 민감하도록 계층적 refinement(보정/스무딩)를 포함해 실행 가능한 궤적까지 이어지게 설계했다.

- **Technical Challenges**: 핵심 난관은 (1) 특징이 저하된 선형 구조에서 전역 일관 맵 없이도 안정적인 중심선과 자세를 추정하는 것, (2) UGV의 저주파 응답(기계적 롤/조향 제약)과 UAV의 실시간 충돌 회피·비행 제약을 동시에 만족하는 것, (3) 계산 지연 없이 제어 주기 내에 장애물 검증을 수행하는 것이다. FLISP는 UGV 포인트클라우드에서 터널 벽 법선과 yaw를 추정해 자세를 구성하고, 동적 binning 및 다단 피팅으로 중심선을 만든 뒤 Bayesian 기반 이상치 보정과 바닥 고도 투영으로 기하적 연속성을 강화한다. 장애물은 경로 정렬 기반 광역 필터로 후보를 줄인 뒤 정밀 containment만 수행하고, UGV 회피는 강화된 Firefly Algorithm에서 롤오버 위험을 soft barrier 형태로 비용함수에 반영해 원통형 바닥 제약을 처리한다. UAV는 UGV 경로를 안전 제약(최저 고도 경로 및 통신 안전영역)으로 변환해 동적 sampling iterative 최적화로 매끈함·안전·진행·고도 일관성을 동시에 만족시키며, 이후 최소-jerk/비균일 B-spline 등 가벼운 궤적 생성과 PD 추종으로 연결한다.

- **Empirical Impact**: 시뮬레이션은 Gazebo에서 실제 수로 터널(직선·회전·연속 곡률, sluice gate 등)을 반영해 추적, 동적 장애물 회피, 런타임 효율을 100회 이상 반복 시험으로 검증했다. 실험 결과 FLISP는 1.2km 운영 수력 터널 배치에서도 지도를 래스터화하는 오버헤드(Fast-LIO2 + A*)와 샘플링 불안정(LIO-SAM + RRT*)을 회피하면서 약 7ms 수준의 낮은 지연으로 100% 성공률을 달성했다. 특히 속도는 그리드 기반 대비 약 7배, 샘플링 기반 대비 최대 3자릿수 개선을 보이며, 특징이 열화된 선형 인프라에서 UGV-UAV 점검의 확장 가능한(실시간·실행 가능) 실용성에 의미 있는 가치를 제공한다.



### Commerge: Communication-Efficient, Robust, and Fast LiDAR Map Merging Framework for Multi-Robot Coordination in Resource-Constrained Scenarios (https://arxiv.org/abs/2606.25386)
Comments:
          32 pages, 32 figures

- **Prior Approaches**: 기존 multi-robot LiDAR map merging은 로봇 간 서버-로봇 또는 로봇-로봇 간 대용량 센서 데이터를 주고받으며 전역 일관성을 맞추는 방식이 많았다. 이때 통신 대역폭이 병목이 되어, 특히 통신 제약 환경에서 탐사 속도와 커버리지 효율이 크게 떨어진다. 또한 전체 스캔 전송은 GB급 데이터 교환을 요구하고, 단순 downsampling은 잡음/기하 정보 손실로 정렬(alignement) 정확도 저하로 이어질 수 있다.

- **Core Contribution**: 이 논문은 Commerge라는 통신 효율형 맵 머징 프레임워크를 제안한다. 그래프 이론 기반의 선택적 데이터 교환으로, 전체가 아닌 ‘소수의 신중히 선택된 scan’만으로도 견고한 맵 머징과 정렬 정확도를 유지할 수 있다는 핵심 인사이트를 구현한다. 결과적으로 로봇 간 통신량을 최대 5,000배까지 줄이면서도 alignment accuracy를 보존하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 “정확도를 지키면서 무엇을 얼마나 교환할 것인가”를 체계적으로 결정하는 것이다. 논문은 exchange graph에서 로봇 keyframe을 정점으로, 가능한 inter-robot loop 후보를 간선으로 두고, 이를 3단계 cascade optimization으로 모델링해 순차적 중첩, 전송 비용 균형, 기하+지각 관점의 최적성 조건을 순차적으로 만족하는 scan 부분집합을 고른다. 이를 통해 단순 전체 전송이나 무작위/천편일률적 축소 대신 MB급 데이터 교환으로도 정렬 품질을 유지하도록 설계한다.

- **Empirical Impact**: 평가는 동굴, 행성 유사 환경, 실내, 실외 캠퍼스 등 다양한 조건을 포함한 5개 공개 데이터셋과 4개 자체 데이터셋에서 수행됐다. 그 결과 데이터 교환량을 최대 99.98% 줄였고, 예를 들어 HeLiPR에서는 7,000MB에서 1.3MB로 감소하면서도 alignment 성능을 유지했다. embedded부터 desktop 플랫폼까지 다양한 하드웨어에서 유사한 정렬 성과를 보여, 통신 제약이 심한 다중로봇 탐사·맵핑에 직접적인 실용적 의미가 있다.



### Reliability-Asymmetric Spacecraft Autonomy: Co-Designing a Capable Learned GNC Stack with a Verified, Adaptation-Aware Runtime Shield (https://arxiv.org/abs/2606.25366)
- **Prior Approaches**: 기존 runtime assurance(RTA)는 복잡한 제어기를 안전 조건 하에서만 허용하고, 위반 시 검증된 단순 제어기로 권한을 되돌리는 Simplex 계열 아키텍처가 주류였다. 그러나 안전 여부를 “즉시 상태”로만 확인하거나, 복귀 시간을 다루지 못하면 온라인 적응형 제어기와 결합 시 비대칭적인 위험 구간이 생긴다. 또한 edge language model 기반 natural language→planning은 문법적 유효성과 의미 정확성 사이의 validity-versus-semantics 갭이 커서, 역으로는 확률 출력의 신뢰성 검증이 어렵다는 한계가 있었다.

- **Core Contribution**: AMPLE-GNC는 역량(지시→계획→제어)과 증명가능성(런타임 안전 보장)을 분리·조합하는 3계층 GNC 스택을 제안한다. 자연어를 PDDL+ 단일 액션으로 매핑하는 foundation-model commander, 선형화 제약을 기준으로 한 verifier, 그리고 fault-adaptive controller를 runtime shield의 안전 불변량(9개)으로 감싼다. 특히 신뢰성 비대칭(reliability asymmetry) 관점에서 “기계검증 가능한 tier가 증명 가능한 경계만 책임지게” 권한을 배치한다.

- **Technical Challenges**: 핵심 난제는 적응형(온라인으로 고장 식별) 제어기가 복구 과도구간에서 의도적으로 안전하지 않을 수 있다는 점이다. AMPLE-GNC는 이 문제를 latching safe-hold shield의 무력화 현상과 함께 진단하고, split-conformal recovery-deadline certificate로 복구 가능 시간을 분포-자유로 인증해 shield 자체를 바꾸지 않고도 안전-복구를 양립한다. commander 쪽은 grammar-constrained decoding(GNBF 문법 기반)으로 출력 유효성을 하드하게 제한하고, semantic 정확도는 de-leaked 분할로 별도 측정해 성능을 정직하게 분리한다.

- **Empirical Impact**: 6-DOF Basilisk 테스트베드에서 commander는 grammar-validity 보장을 유지하면서 planner-executable rate 84%를 보였고, de-leaked novel-phrasing 일반화는 38%(문구 다양성 재파인튜닝 후 48%)로 보고했다. fault-adaptive controller(RMA)는 학습 내 랜덤화 범위에서 sign fault 97.8%, continuous-gain fault 94.4% 복구를 달성했으며 PD·end-to-end RL은 0%에 그쳤다. 더 나아가 복구 deadline 기반 shield 연동은 controller가 94.5% 자율성을 유지하면서도 비복구 케이스를 적시에 포착하도록 하였고, 검증 기반 모니터는 Kind 2로 9/9 불변량의 predictor soundness를 기계검증했다.



### Decoupling Semantics and Geometric Grounding: Spatial Visual Prompts for Language-Conditioned Imitation Learning (https://arxiv.org/abs/2606.25360)
- **Prior Approaches**: 최근 end-to-end Vision-Language-Action(VLA) 모델은 RGB와 언어를 직접 입력해 연속 행동을 예측하지만, 의미 추론과 공간 그라운딩이 단일 네트워크에 강하게 결합되어 정렬(alignment) 병목이 생깁니다. 특히 데이터가 적은 imitation learning에서는 언어 토큰을 정확한 픽셀/공간 제어로 암묵적으로 정렬하기가 비효율적이라, 다중-물체 상황에서 타깃을 애매하게 구분하거나 근접(near-miss) 실패가 잦습니다. 이에 따라 3D point cloud 기반 정책이나 표준 이미지-level 마스킹/프롬프트 주입 같은 대안이 시도됐지만, 계산 복잡도 증가나 시각 분포 도메인 시프트 문제가 남아 있습니다.

- **Core Contribution**: 이 논문은 의미-공간 정렬 병목을 줄이기 위해, monolithic end-to-end를 버리고 decoupled perception-control 구조를 제안합니다. 구체적으로 언어에서 Spatial Visual Prompt(SVP)를 추출해, 연속 행동 생성 루프에서 공간 타깃 구분을 명시적으로 분리 주입합니다. 언어는 VLM/세그멘테이션을 통해 zero-shot 기하 마스크로 변환되고, 정책은 그 마스크 기반의 공간 실행에 더 집중하도록 설계됩니다.

- **Technical Challenges**: 핵심 과제는 SVP의 기하 우선순위를 연속 visuomotor 정책에 효과적으로 주입하면서, pre-trained 시각 백본의 저수준 이미지 prior를 망치지 않는 것입니다. 기존처럼 RGB에 마스크를 오버레이하거나 채널을 이어붙이면 입력 분포가 크게 바뀌어 얕은 층의 ImageNet 특징이 손상되고 저데이터에서 과적합/불안정이 커집니다. 이를 해결하기 위해 중간 latent 공간에 직접 주입하는 direct feature-level fusion(요소별 덧셈)과 lightweight prompt stream을 제안해, 입력-level 도메인 시프트 없이 공간 그라디언트를 안정적으로 제공합니다.

- **Empirical Impact**: RoboTwin 2.0에서 SVP-IL은 평균 성공률 67.8%로, π0π0(53.7%) 및 ACT/DP 같은 강한 비교군을 크게 앞섰고 애매한 언어 조건에서 데이터 50~100개만으로도 성능이 잘 유지됩니다. 특히 custom language-conditioned 과제에서는 24.0%에서 39.5%로 향상되며, 타깃 의미 분해(disambiguation)가 크게 개선됐음을 보여줍니다. 또한 Aloha-AgileX 실로봇 실험에서 복잡한 난잡한 환경에서도 60.0% 평균 성공률을 달성해 sim-to-real 강건성과 데이터 효율성을 함께 검증했습니다.



### Self Capacitive Tactile Sensor System designed for Companion Robots (https://arxiv.org/abs/2606.25348)
- **Prior Approaches**: 기존 촉각 센서는 capacitive, resistive, piezoresistive, optical 방식 등 다양한 대안이 제시됐지만, 보통 민감도·응답성·전력·제작 난이도 사이에서 트레이드오프가 발생한다. 특히 전신(whole-body) 촉각을 목표로 하면 다층 구조, 복잡한 배선/전극 패터닝, 높은 비용이 커져 확장성과 실시간·저지연 처리가 어려워진다. 일부 섬유 기반 설계는 가능하지만, 센서마다 마이크로컨트롤러와 센서당 4가닥 배선이 필요해 companion 로봇에선 부담이 크다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 self-capacitance 원리를 이용해 단일 conductive fabric layer와 conductive fabric wire만으로 확장 가능한 촉각 센싱 시스템을 제안한다. 전극 패터닝과 센서별 마이크로컨트롤러를 요구하지 않아 HIRO-chan 같은 소프트·저비용 companion humanoid에 적합하다고 강조한다. 또한 interaction 유형 분류를 FPGA에서 직접 수행해 메인 프로세서의 연산 부담과 지연을 줄이는 구조를 제시한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 얇고 말랑한 직물 기반 센서에서 self-capacitance 변화를 잡아내는 측정 신뢰성, (2) 다점(100점) 센싱 시에도 잡음/간섭 및 오탐을 최소화하는 배선·배치, (3) 엣지 환경에서 실시간 분류를 가능한 최소 연산으로 구현하는 것이다. FPGA는 전극 충전 RC 완화 시간을 clock cycle로 정밀 계수해 self-capacitance 변화를 추출하고, 100점은 100개의 RC 회로(10MΩ 저항)로 구성했다. 분류 모델은 iCE40HX8K의 제한된 LUT 자원을 고려해 decision tree를 선택·온보드 배치함으로써 Raspberry Pi 4 오프로딩과 지연/전력 최소화를 동시에 달성했다.

- **Empirical Impact**: sampling frequency 실험에서 10 Hz는 hitting과 fast tapping 같은 단발성 transient 이벤트를 놓쳤고, 100 Hz와 1000 Hz는 모든 상호작용 유형을 안정적으로 구분했다. 1 kHz는 신호는 가장 잘 보이지만 데이터량 증가로 엣지 연산 부담이 커서, 100 Hz가 정보 손실 없이 구현 가능한 현실적 절충점으로 결론냈다. 또한 Random Forest/SVM은 높은 정확도 대비 LUT 소모가 커 iCE40HX8K에 부적합했고, decision tree는 약 90% 정확도와 2,067 LUT로 온보드 구현 가능성을 입증해 후속 전신 촉각 확장 설계에 직접적인 실용 신호를 제공한다.



### AI Coaching for Accelerating Human Skill Development with Reinforcement Learning (https://arxiv.org/abs/2606.25337)
- **Prior Approaches**: 기존 human–AI 협업은 shared control 기반으로 실패를 막는 ‘가디언 에인절’ 역할에 집중해 왔습니다. 그러나 지속적인 assistance는 over-reliance와 skill atrophy를 유발해, 장기적으로 학습 효과를 깎을 수 있다는 지적이 이어졌습니다. 또한 일부 연구는 assistance 강도를 점진적으로 줄이거나 curriculum을 설계하지만, 학습자의 현재 역량과 물리적 런타임 맥락을 동시에 닫힌 고리로 다루는 데 한계가 있었습니다.

- **Core Contribution**: 이 논문은 embodied AI 에이전트가 단순 조정보다는 ‘코치’로서 장기 독립 역량을 키우도록 assistance를 전략적으로 조절해야 한다고 제안합니다. 이를 위해 Human–AI Coaching Game을 비협조적 동적 게임으로 형식화하고, 학습자는 task 성과를 최대화하는 동시에 코치는 학습자의 independent competence(VoI)를 목표로 삼습니다. 결과적으로 AI를 미래의 자기 능력으로 이어지게 하는 personalized mentor 관점으로 전환합니다.

- **Technical Challenges**: 주요 난제는 (1) POSG 수준의 계산 어려움, (2) VoI가 ‘코치가 지금 떠났다면’이라는 반사실(counterfactual) 형태라 추정 비용이 큼, (3) coaching이 학습자의 skill 변화에 간접적으로만 영향을 준다는 점입니다. 저자들은 학습자의 행동이 잠재 skill θ에 의해 좌우된다는 점을 이용해, 확률적 finite-state automaton으로 skill 진행의 인과 효과를 모델링하고 게임을 단일 에이전트 POMDP로 환원해 reinforcement learning으로 학습 가능하게 했습니다. 또한 adaptive closed-loop blending과 θ-기반 skill evolution 모델을 결합하고, counterfactual VoI를 대신하는 대리 보상으로 학습을 안정화했습니다.

- **Empirical Impact**: FPV drone racing 사용자 연구(N=33)에서 L2C 코치는 최신 AI coaching baseline 대비 학습 성과를 유의미하게 개선했습니다. 구체적으로는 reduced lap times와 failure counts 감소로 나타났으며, 단순 성과 보강이 아니라 학습자의 독립적 역량 향상에 초점이 맞춰졌음을 시사합니다. 이는 로보틱스·인간-로봇 상호작용에서 AI를 ‘가드’에서 ‘멘토’로 바꾸는 실증적 근거를 제공한다는 점에서 의미가 큽니다.



### WaveForward: An Omnidirectional Passive Wheeled Quadruped Robot with Casters (https://arxiv.org/abs/2606.25299)
Comments:
          8 pages,11 figures

- **Prior Approaches**: 휠+다리 혼합형 로봇은 복잡 지형에서의 민첩한 이동과 높은 효율을 동시에 노릴 수 있어 장거리 운송에 적합하다. 다만 기존 구동 휠 로봇은 바퀴 하드웨어와 전기 설계가 필수라 제작 비용과 설계 복잡도가 커진다. 또한 패시브 캐스터 기반의 전방위 이동을 안정적으로 제어하려면 캐스터 각도/속도 같은 정보 활용 및 자세-추력의 연결이 관건이다.

- **Core Contribution**: 본 논문은 각 다리에 standard caster를 장착한 저비용 패시브 휠드-레그드 로봇을 제안하며, 캐스터가 제공하는 전방위(omnidirectional) 이동성을 목표로 한다. 제어는 asymmetric actor-critic 구조를 사용해 패시브 캐스터의 각도와 속도 같은 privileged information을 학습에 활용한다. 또한 속도 명령을 기반으로 캐스터 베이스 자세를 조정하는 전략을 제시해 추진 방향을 능동적으로 바꾸도록 한다.

- **Technical Challenges**: 핵심 난제는 “패시브 캐스터의 비선형 거동”을 “추진력으로 일관되게 변환”하는 제어 설계에 있다. 저자는 구동 관절로 캐스터 베이스 joint axis의 자세를 바꿔 캐스터 트위스팅 방향과 정도를 조절하는 velocity-command 기반 자세 조정 전략을 구성한다. 더 나아가 캐스터 트위스팅 oscillation 정도를 달리하는 여러 propulsion mode를 정의하고, 그 진동을 추진력으로 연결하도록 학습·운용한다.

- **Empirical Impact**: 실험에서는 슬라럼 테스트와 모드 스위칭 경험을 통해 패시브 휠드 쿼드루페드가 전방위 이동의 활용도를 시연했다. 특히 legged motion 대비 cost of transport(COT)를 최대 89.1%까지 줄였다고 보고한다. 이는 고가의 구동 휠 하드웨어 없이도 효율적으로 전방위 기동을 구현할 수 있음을 보여주는 실증적 결과로, 장거리 운송 분야 설계 비용 절감에 직접적인 의미가 있다.



### DynaMOMA: Instantaneous Prediction of Grasp Poses for Mobile Manipulation of Dynamic Objects (https://arxiv.org/abs/2606.25295)
- **Prior Approaches**: 모바일 매니퓰레이션에서 동적 물체를 잡는 기존 접근은 크게 (1) 3D 재구성과 기하 기반 플래너, (2) 관측-액션을 직접 잇는 end-to-end RL, (3) 물체의 모션을 예측해 잡음(잡음/오차)을 줄이는 forecasting류로 나뉜다. 다만 forecasting은 ‘물체 상태’는 줄 수 있어도, 기하 조건이 필요한 grasp pose의 궤적을 직접 제공하긴 어렵다. 또한 반응형 RL은 현재 프레임만 보며 따라가다 보니, 실행 중 물체가 계속 움직이면 시점이 어긋나 tracking lag이 생긴다.

- **Core Contribution**: DynaMOMA는 즉시성 있는 grasp trajectory(짧은 예측 구간)를 예측하는 모듈과 whole-body control 정책을 한 프레임 내에서 결합한다. 핵심은 anchor-based diffusion 기반 예측기가 과거 관측을 조건으로 temporally consistent한 multi-mode grasp 궤적을 만들고, 이를 compact feature로 바꿔 reinforcement learning 정책에 주는 구조다. 여기에 anticipation-guided reward를 더해, 로봇이 가까워질수록 ‘현재 관측’보다 ‘예측된 즉시 궤적’에 더 신뢰를 주도록 학습시킨다.

- **Technical Challenges**: 동적 환경에서는 (i) 물체 움직임에 의해 가능한 grasp trajectory가 여러 모드로 갈라지고, (ii) off-the-shelf grasp detector의 top pose가 프레임마다 바뀌면 궤적이 들쑥날쑥해진다. 논문은 K개의 anchor grasp trajectory를 K-means 클러스터링으로 만들고, truncated diffusion으로 로컬에서만 denoising하며 temporally consistent multi-mode 예측을 안정화한다. 이후 정책 쪽에서는 거리 기반 trust coefficient로 reward 타깃을 현재 관측과 예측 미래 목표 사이에서 부드럽게 보간해, 멀리서는 관측 중심의 접근과 가까이서는 feedforward 정렬을 동시에 달성한다.

- **Empirical Impact**: Isaac Gym 시뮬레이션에서 DOMM(Frontal/Lateral/Chasing) 벤치마크로 평가한 결과, DynaMOMA는 Grasp Success Rate에서 반응형 및 예측 보강 베이스라인을 크게 앞선다(예측이 없던 반응형 대비 설정 전반에서 큰 폭). 또한 Kalman/linear extrapolation 같은 단순 예측 삽입만으로는 GraspSR이 낮게 유지돼, grasp pose 예측의 기하 조건 중요성을 실증한다. Sim-to-real로도 predictor+policy의 일반화가 관측되며, 물체를 능동적으로 방해하는 사용자 연구(쉬움/어려움 모드)에서 높은 성공률로 실제 강건성을 확인한다.



### An Integrated Hardware-Software Design for Low-Data Spatial Defect Detection in Robotic Visual Inspection with Hybrid Optoelectronic Neural Networks (https://arxiv.org/abs/2606.25277)
- **Prior Approaches**: 기존 로봇 비전 결함 검출은 고해상도 이미지를 기반으로 하되, 데이터가 기하급수적으로 늘어나 저장·엣지 연산 부담이 커진다는 한계가 있다. 또한 YOLO 같은 탐지 프레임은 박스/분류 라벨에 크게 의존하고, 특히 결함의 ‘모양’ 수준(shape-level) 주석은 인력 비용이 높으며 라벨 일관성 문제도 동반된다. 압축센싱은 데이터량을 줄이지만, 대부분 복원(reconstruction) 계산이 비싸거나 복원-후처리로 오류가 누적되며 여전히 입력 데이터 최적화가 부족하다고 지적한다.

- **Core Contribution**: 이 논문은 DMD 기반 광학-물리 계층과 신경망 소프트웨어를 한 시스템으로 묶은 optoelectronic 아키텍처를 제안한다. 센서-in-the-loop로 DMD를 ‘물리적 optical convolutional layer’로 재구성해, 광학 도메인에서 특징 추출과 차원 축소를 수행함으로써 전통적 CS 복원 단계를 제거한다. 더불어 결함 모양 주석 없이 자연어 결함 설명을 CLIP(Contrastive Language-Image Pre-training)의 임베딩으로 연결해, 네트워크 attention을 결함 형상에 정렬시키는 학습 방식을 도입한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) CS 측정과 딥러닝 특징 추출을 물리 계층에서 결합해도 성능이 유지되는지, (2) DMD의 이진(ON/OFF) 구동 제약 하에서 부호/가중치 표현을 어떻게 구현할지, (3) 모양 정밀도를 평가할 수 있는 정량 지표가 없는 문제다. 저자들은 measurement matrix를 신경망의 첫 레이어 convolution 커널로 선택하고, 블록 기반 compressed sensing으로 공간 정보를 저차원 시간 신호에 인코딩하며, DMD에서 Unsigned/Bool/Signed 등의 파라미터 표현을 분해(예: Signed를 양·영/음·영으로 분리)해 구현한다. 마지막으로 heatmap 기반 결함 위치 품질을 계량하는 LAA(Localization Accuracy for Attention) 메트릭을 제안해 shape-level localization을 직접 평가한다.

- **Empirical Impact**: 투명 소재의 tiny defect(spot/scratch) 검출 실험에서, 제안 구조는 기존 imaging 대비 데이터 중복을 90% 줄이면서도 분류·국소화 정확도를 동급 수준으로 유지했다고 보고한다. 또한 Vision Transformer에서는 데이터 90% 감축과 함께, CNN에서는 연산량을 약 60% 줄이는 효과가 관찰된다. 파라미터 분석에서는 measurement matrix, compression ratio, block size가 정확도와 LAA에 미치는 영향을 보여주며, 특히 ViT와 binary matrix(Bool/Sign) 조합에서 샘플링률이 올라갈수록 안정적인 성능 향상이 나타난다고 정리한다. 데이터 스트림이 대규모이고 촬영 비용이 높거나 엣지 자원이 제한된 산업 자동화 시나리오에 실용적인 ‘low-data’ 해법이라는 점에서 의미가 크다.



### GRAFT: Graph-Based Affordance Transfer via Part Correspondenc (https://arxiv.org/abs/2606.25241)
- **Prior Approaches**: 기존 retrieval 기반 데이터 생성은 범주/텍스트 같은 의미론으로 후보를 거른 뒤, appearance(시각 특징)나 전역 기하 유사도로 정렬해 접촉점을 전이한다. 그러나 의미-기하 파이프라인은 후보 풀을 과도하게 제한하고, 스케일·구성·관절 차이에서 correspondences가 쉽게 깨져 near-identical 인스턴스만 반복 선택되는 문제가 있었다.

- **Core Contribution**: GRAFT는 의미가 아니라 ‘부품 그래프’의 구조로 기능 호환성을 판단해, object당 1개의 demonstration만으로 zero-shot 조작 전이를 지원한다. 부품 단위 그래프 정합으로 먼저 알맞은 인스턴스를 찾고, 이후 vertex-level 대응을 통해 접촉점을 point-wise로 전파하는 two-stage(부분 정합→점 정합) 설계를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 부품 수·관절·스케일이 달라도 안정적으로 부분 대응을 찾는 것과 (2) 전이 가능한 ‘접촉 지점’을 정밀하게 매칭하는 것이다. GRAFT는 part-based graph 표현과 Unbalanced Fused Gromov–Wasserstein(UFGW)으로 부분 매칭을 허용해 그래프 크기 차이에도 대응을 만들고, EM 스타일로 target 노드 mass를 추정한 뒤 Fourier-style vertex descriptor와 part-level 정합 결과를 결합해 접촉점을 fine-grained으로 옮긴다.

- **Empirical Impact**: 실험에서는 unseen 객체에 대해 affordance 일반화와 물리 실행(부유 그리퍼 롤아웃) 평가를 분리해 검증했으며, GRAFT가 최상위 Success Rate을 기록하고 AnyGrasp Error가 0에 가깝게 나타났다. 또한 MimicGen에 결합해 데이터 생성 성과를 크게 끌어올리며(trajectory success rate 63% vs 11%), ablation에서 GRAPH 구조·BFS saliency·EM 최적화의 기여가 확인돼 구조적 정합이 실제 전이 안정성으로 이어진다는 점을 보여준다.



### Spatio-Temporal Retrieval-based Priors for Adaptive Computational Teaching in Driving (https://arxiv.org/abs/2606.25224)
Comments:
          20 pages, 8 figures

- **Prior Approaches**: 자동 운전처럼 긴 시간축의 모터 태스크를 코칭하려는 학습 기반 접근은 로컬 문맥 중심 추론에 의존하는 경우가 많아, 반복 상호작용이 누적되며 나타나는 학생의 장기 학습 변화를 충분히 반영하지 못했다. 또한 embodied 교육(신체/행동 기반) 환경은 고품질 인터랙션 데이터 수집이 비싸서 low-data regime에서 성능이 급격히 흔들린다는 한계가 있었다. 지식 추적·장기 시퀀스 모델링은 존재하지만, 관찰 가능한 데이터가 풍부한 교실형 설정에 최적화된 경우가 많아 그대로 옮기기 어렵다.

- **Core Contribution**: 이 논문은 학생-교사 상호작용의 과거 이력(history)을 이용해 적응형(시간적) 코칭을 수행하는 imitation learning 기반 계산 모델을 제안한다. 핵심은 temporal reasoning module로, 과거 상호작용을 바탕으로 장기적 누적 효과를 추론하되, 데이터가 적은 상황에서도 작동하도록 KNN 검색과 cross attention prior를 도입한 점이다. 또한 동시(concurrent) 교육 모듈과 과거 이력 인코딩·퓨전 모듈을 모듈화해 학습 안정성과 적응성을 함께 노린다.

- **Technical Challenges**: 주요 기술 과제는 (1) end-to-end attention 기반 장기 템포럴 상관을 학습할 만큼 인터랙션 데이터가 충분치 않다는 점과 (2) 장기 이력 전체를 그대로 넣을 때 계산비용·일반화 문제가 커진다는 점이다. 논문은 이를 해결하기 위해 최근접 이웃 retrieval로 의미적으로 유사한 과거 상호작용만 추려 cross attention으로 필요한 정보만 선택적으로 결합하도록 설계했다. 아울러 self-supervised 보조과제로 trajectory reconstruction과 metrics prediction을 함께 학습해 과거 인코더가 의미 있는 temporal representation을 형성하도록 했다.

- **Empirical Impact**: 검증은 Waymo Open Motion Dataset 기반의 반(半)합성 closed-loop 장기 코칭 데이터셋 WayCoach와, 소규모 실제 자연주의 시뮬레이터 레이싱 코칭 데이터셋 SimCoachCorpus에서 수행됐다. 결과적으로 KNN-CrossAttn은 non-adaptive baseline을 일관되게 능가했고, 특히 low-data regime에서 Full-CrossAttn보다 성능 저하가 완만해 더 안정적이었다. 자연주의 도메인에서도 KNN-CrossAttn이 다른 적응형 변형들 대비 소폭이지만 우수하며, 사전(prior) 기반 검색이 data-efficient한 장기 코칭에 유의미하다는 점을 보여준다.



### Swazure: Swarm Measurement of Pose for Flying Light Specks (https://arxiv.org/abs/2606.25222)
Comments:
          Appeared in Proceedings of the Holodecks Foundation, Vol. 2, No. 3

- **Prior Approaches**: 드론에 RGB 라이트와 함께 Flying Light Specks(FLSs)를 배치해 3D 포인트 클라우드를 기반으로 형상을 비추는 연구가 진행돼 왔다. 하지만 FLS의 상대 자세(localize/pose)를 카메라 등 센서로 추정할 때, 센서가 정확한 구간(sweet range) 밖에서는 정확도와 그레뉼러리티가 급격히 떨어지는 문제가 남는다. 결국 포인트 클라우드의 최소 점 간 거리와 센서 성능이 강하게 결합돼, 센서 교체/업그레이드가 필요해지는 “물리 데이터 의존성”이 한계로 지적된다. 센서 시야선이 가려지는 경우(상대적으로 큰 FLS 반경, 비가시성으로 인한 obstructing)에도 경로 기반 추정이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 Swazure로, 포인트 클라우드 “데이터”가 아니라 FLS의 “센서 하드웨어”가 데이터 특성에 종속되지 않도록 physical data Independence를 제안한다. 핵심은 센서의 측정 가능 구간을 blind/sweet/decaying으로 추상화하고, 협동하는 스웜이 blind·decaying 이웃도 sweet 수준의 상대 자세를 복원하도록 만든다는 점이다. 또한 큰 FLS로 인해 경로 추정이 막히는 경우를 Move Obstructing, Move Source 두 휴리스틱으로 완화한다.

- **Technical Challenges**: 기술적 난제는 (1) sweet 범위 밖에서 발생하는 자세 오차를 단순 보정이 아닌 “경로 합성”으로 흡수하는 것과 (2) 경로에 끼어드는 obstructing FLS가 common sweet neighbor를 만들지 못하게 하는 문제다. Swazure는 sweet path(연속적으로 sweet 이웃을 잇는 경로)를 찾고, 각 hop에서 얻은 상대 위치 벡터를 합산해 소스-타깃의 상대 자세를 추정한다. 경로가 막히면 decaying hop을 섞거나, Move Obstructing는 시야를 가리는 FLS를 충돌을 피하는 범위에서 최소 이동으로 재배치해 no path를 sweet path로 전환하며, Move Source는 소스가 주변을 탐색해 공통 sweet neighbor를 확보하도록 한다. 시뮬레이션에서는 FLS를 구(sphere)로 모델링하고, β(구 반경/최소 점 간 거리 비율)에 따른 충돌·가림 효과를 반영해 경로 존재성과 막힘 정도를 평가한다.

- **Empirical Impact**: Raspberry camera+ArUco marker 기반 오차 특성(정확 구간 및 거리 증가에 따른 decaying 오차)을 반영한 시뮬레이션에서 Swazure의 경로 복원 성능을 검증한다. 특히 Move Obstructing이 Move Source보다 효과적이며, 최악의 경우 obstructing 문제를 약 30%까지 해소한다고 보고한다. 또한 β≤0.5에서 충돌을 제거한 설정에서도, decaying 경로를 sweet 경로로 더 많이 전환해 상대 자세 추정의 품질을 개선한다. 공개 구현과 데이터셋을 GitHub에 제공해, 스웜 기반 3D 멀티미디어 표시에서 센서-데이터 결합 문제를 완화하는 실용적 방향성을 제시한다.



### RigPI: Dynamic Parameter Identification of Rigid Body via VLM-Seeded Differentiable Simulation (https://arxiv.org/abs/2606.25212)
Comments:
          Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2027)

- **Prior Approaches**: 기존 동역학 파라미터 식별은 역동역학(inverse-dynamics) 기반으로 선형 회귀(least-squares 등)를 수행하는 방식이 주류였지만, 잡음과 모델링 오차에 민감하고 물리적으로 말이 안 되는 해를 만들기 쉽다는 한계가 있다. 또한 추정 문제를 잘 풀려면 가진(excitation) 설계가 필요하고, 접촉이 많은 상황이나 시각 추론이 섞이면 비정상/불능 추정이 발생하기 쉽다. 최근에는 differentiable physics를 통한 end-to-end 최적화가 늘었지만, 초기값·파라미터 범위·문제 정합성에 따라 성능이 흔들린다는 문제가 남아 있다.

- **Core Contribution**: RigPI는 로봇-물체 상호작용 데이터(포스/토크, 물체 포즈)와 시각 기반 priors를 differentiable simulation 파이프라인에 결합해, 무구속 강체와 다관절 강체의 관성(inertial)·마찰(friction) 파라미터를 동시에 식별하는 체계적 프레임워크를 제안한다. 핵심은 VLM(vision-language model)을 이용해 초기화와 물리적 탐색 공간을 먼저 제약하고, NVIDIA Newton 기반 미분 가능한 시뮬레이터의 gradient로 파라미터를 안정적으로 정련(refine)하는 것이다. 또 두 단계 최적화 설계로 잡음 민감도와 물리 불가능 해 회피를 함께 노린다.

- **Technical Challenges**: 가장 큰 기술적 도전은 실세계에서 수집되는 힘/토크·포즈 데이터의 잡음과 접촉/마찰 모델링 불일치 때문에, 파라미터 최적화가 ill-posed로 변해버린다는 점이다. RigPI는 (1) RGB-D를 입력으로 VLM이 밀도·마찰 등 base property의 초기 추정과 범위(bounds)를 만들고 (2) 뉴턴 시뮬레이터를 통해 포즈 불일치 손실과 파라미터 범위 위반 패널티를 포함한 augmented loss를 정의해 gradient 기반 탐색이 물리적으로 타당한 영역에 머물도록 유도한다. 이때 학습률 α를 초기 스케일에 맞춰 조정해 수렴 안정성과 효율을 동시에 확보한다.

- **Empirical Impact**: 실험은 xArm6 로봇과 ATI Axia80-M20 센서, Azure Kinect RGB-D로 무구속(큐브/스피어/애플) 및 다관절(서랍/캐비닛/오븐) 물체를 대상으로 수행됐다. 결과적으로 RigPI는 least-squares baseline과 GradSim 대비 관성·마찰을 포함한 추정 정확도와 안정성(CoV 기반)을 일관되게 개선했으며, 제한된 조인트 운동 범위로 일부 관성 성분이 덜 정확해지는 상황에서도 전체적인 재현 성능은 유지됐다. 무엇보다 추정된 파라미터를 이용해 미지의 힘/토크에서 생성한 예측 궤적이 실제 로봇 궤적과 잘 일치함을 통해 ‘parameter-aware predictive validity’를 실증했으며, VLM priors의 필요성과 효과는 ablation으로도 확인됐다.



### Wear-Clearance-Impact Coupling in the Jansen Linkage: A Gait-Durability-Optimized Design Slows Joint Loosening (https://arxiv.org/abs/2606.25208)
Comments:
          10 pages, 9 figures. Companion to arXiv:2606.22129

- **Prior Approaches**: 기존 연구는 Theo Jansen 다리의 내구성 목표를 설계에 반영했지만, 회전 조인트를 이상적인 clearance-free 핀으로 가정해 실제 마모를 “순위” 수준으로만 비교했습니다. 실제로는 반경 방향 유격이 존재하고, 마모로 유격이 커지면 접촉이 충격성으로 변하며 wear–clearance–impact 결합이 생깁니다. 기존 clearance-관성계(접촉+마찰)–Archard 결합 연구도 일반 기구에 머물러, 보행용 Jansen 다리의 간헐 지면하중 조건과 설계 전이 가능성을 충분히 다루지 못했습니다.

- **Core Contribution**: 이 논문은 Jansen 다리에 대해 clearance 조인트를 포함하는 최초의 clearance-coupled forward-dynamic 모델을 제시합니다. Lankarani–Flores 계열의 연속 법칙(히스테리시스 감쇠)과 Ambrósio 마찰을 constraint-stabilized 미분-대수 방정식으로 통합하고, 마모는 Archard 법칙으로 계산해 wear→clearance→impact 피드백 루프를 구성했습니다. 또한 동반 연구에서 얻은 “gait–내구성 최적 설계”가, 유격이 있는 현실 조건에서도 내구 이득을 유지하는지 정량 검증합니다.

- **Technical Challenges**: 핵심 난제는 유격이 만들어내는 비선형 충격-접촉 동역학을 wear 모델과 안정적으로 결합하는 것이며, 단일 궤적은 비단조·혼돈적 거동을 보인다는 점입니다. 이를 위해 Radau 같은 암시적 적분과 Baumgarte 안정화로 constraint drift를 극도로 낮추고, clearance 증가에 따른 충격 반복을 differential-algebraic 시스템에서 재적분하도록 구현했습니다. 마모는 접촉 정규력과 접선 미끄러짐을 cycle 단위로 적분해 보어 반경 유격 증가로 환산하며, 사이클 매크로스텝으로 반복 루프를 닫아 예측을 가능케 했습니다.

- **Empirical Impact**: 결과는 명확합니다: 유격을 무시하면 조인트 peak load가 약 2배(예: 104 N vs 48 N) 과소평가되며, 두 조인트가 동시에 유격을 가지면 ensemble 기준 peak force가 약 426 N 수준으로 더 커집니다. 또한 단일 궤적에서는 설계 순위가 뒤집힐 수 있을 만큼 영향이 impact-sensitive하지만, 16개 랜덤 위상 앙상블 평균에서는 동반 연구의 최적 설계가 per-cycle wear을 약 9–7배 줄이고(peak force는 약 4배 감소), 두 조인트 동시 유격에서도 약 1.7배 이득을 유지합니다(p<0.01). 마지막으로 마모는 보어 전역에 균일하지 않고 약 10° 하중 호(arc)에 집중되어, 균일 유격 성장 가정이 국소 유격 성장을 약 36배 과소평가함을 보여줍니다. 저자들은 계측 가능한 4가지 반증 가능 실험 프로토콜(충격 증폭, 궤적 민감도, 앙상블 내구 이득, 비균일 마모 스칼/스카라)도 제시해 계산 결과의 검증 가능성을 높였습니다.



### RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory (https://arxiv.org/abs/2606.25206)
Comments:
          Project website: this https URL

- **Prior Approaches**: 기존 로봇 장기 배치 메모리는 라벨이 고정된 의미 맵(카테고리)이나 3D 포인트클라우드 같은 지표 기반 표현에 의존해, 질의가 특정 세부를 요구할 때(예: 작은 색/모양 디테일) OOV(미등록) 문제가 쉽게 발생한다. 또 다른 접근은 오픈보캐뷸러리를 위해 관측을 caption으로 바꾸고 텍스트 임베딩을 저장하는데, 이 image-to-text 캡션 병목이 시각 디테일을 손실시켜 검색이 취약해진다.

- **Core Contribution**: RAVEN은 캡션 대신 raw visual embeddings(시각 임베딩)를 그대로 저장하는 agentic memory 시스템으로, 장기 로봇 질의응답과 내비게이션을 지원한다. 관측 프레임마다 시각 임베딩에 pose(자세)와 time(시간)을 함께 묶은 visuo-spatio-temporal memory triplet을 vector database에 넣고, 공간 지도 기반으로 검색을 grounding해 답변과 목표 이동을 동시에 수행한다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트로 압축하지 않으면서도 fine-grained 의미를 유지하고, (2) 수천 프레임 단위의 긴 기간에서도 검색 비용을 억제하며, (3) 검색 결과를 로봇 플래닝에 연결해 정확한 목표 좌표를 뽑는 것이다. RAVEN은 FAISS/Milvus 같은 벡터 검색으로 top-K 검색을 sub-O(N) 수준에 가깝게 처리하고, VLM이 도구(텍스트/시간/위치/이미지 기반 검색)를 유한상태머신 루프에서 반복 호출해 working memory를 줄이면서 정밀도를 높인다.

- **Empirical Impact**: 여러 시뮬레이션 및 실제 비디오 QA 벤치마크에서 RAVEN은 caption 기반 메모리 시스템을 일관되게 능가하며, frontier VLM과 비교해 장기 과제 성능을 유지하면서 검색 비용은 10배 낮췄다고 보고한다. 또한 Unitree Go1을 실제 환경에 배치해 자연어 목표 도달 내비게이션을 성공적으로 시연했고, 0.12 fps 다운샘플링에서도 성능의 대부분을 보존하며 원시 RGB 대비 250배 이상 저장 압축과 높은 메모리 효율을 보였다.



### Learning Perceptive Platform Adaptive Locomotion Controllers for Quadrupedal Robots (https://arxiv.org/abs/2606.25179)
- **Prior Approaches**: 기존 RL 기반 보행 제어는 특정 로봇(단일 morphology)이나 대규모 cross-embodiment 학습에 크게 의존해 왔다. 이 경우 다른 체형으로 옮길 때 성능이 떨어지거나, 실사용 배치를 위해 계산 비용이 과도해지는 한계가 있다. 또한 perception을 전면에 둔 완전 지각형 정책은 잡음에 민감할 수 있는 반면, blind 정책은 지형 정보를 충분히 반영하지 못해 거칠고 변화하는 환경에서 제약을 겪는다.

- **Core Contribution**: 이 논문은 morphology-aware reinforcement learning에서 perception을 어디에, 어떻게 넣어야 deployable한 사족 보행이 되는지에 초점을 둔다. MorAL을 확장해 여러 reference quadruped를 기준으로 morphology-specialized universal controller를 학습하고, 적응형 terrain curriculum으로 지형-체형 조합의 난이도를 균형 있게 조절한다. 비교 실험을 통해 critic-only(critic이 지각을 사용) 배치가 blind 및 완전 지각(actor-critic) 사이의 “안정성-강건성 절충”을 만든다는 점을 정리한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 지각을 넣되 잡음에 의한 학습/제어 불안정을 줄이고, (2) 체형이 달라져도 일관된 보행 추적을 유지하는 perception 통합 구조를 설계하는 것이다. 저자들은 actor는 추정된 morphology 정보를 사용하고, critic은 학습 중에만 exteroceptive(지각)와 privileged 정보를 활용하는 비대칭 actor-critic 구조로 분산된 역할을 부여했다. 여기에 morphology별 적응형 지형 커리큘럼을 결합하고, height scan에 대한 latency randomization 등 잡음 환경을 학습에 포함해 현실 배치 민감도를 완화했다.

- **Empirical Impact**: 시뮬레이션에서 평지와 거친 지형(오르막/계단/험지 등)에 대해 평가했으며, ANYmal 하드웨어에 실제 배치해 zero-shot 전이를 검증했다. 결과적으로 critic-only perception은 blind baseline 대비 강건성과 추적 일관성을 개선하면서도, 완전 지각 정책보다 perception noise 하에서 더 안정적으로 동작했다. 저자들은 scalable한 morphology-aware locomotion을 위해 perception placement와 adaptive curriculum 설계가 성패를 가르는 변수라고 결론지었다.



### fARfetch: Enabling Collocated AR-HRC in Large Visually Diverse Environments with VLM-Driven AR Content Adaptation (https://arxiv.org/abs/2606.25162)
Comments:
          Accepted to the 2026 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). Author accepted manuscript

- **Prior Approaches**: 기존 AR-HRC는 로봇과의 근접성이나 VLOS(visual line of sight)를 전제로 하는 경우가 많아, 야외처럼 로봇이 멀어지거나 시야가 막히면 목적지/경로 지시가 급격히 어려워진다. world-in-miniature(WIM) 기반 접근도 주로 목적지 선택에 머물고 의미 있는 semantic 문맥과 정밀 path authoring이 부족하며, 헤드셋이 감지한 랜드마크가 로봇의 이해로 공유되지 않는 한계가 있다. 또한 AR view management는 대개 정적인 대상 중심이라, 동적이고 복잡한 야외 배경에서 legibility를 안정적으로 유지하기 어렵다.

- **Core Contribution**: fARfetch는 Meta Quest 3와 Unitree Go2를 묶어, 로봇 근접 보장이 어렵고 VLOS가 끊겨도(collocated) AR에서 목적지와 fine-grained path를 지시할 수 있는 AR-HRC 시스템을 제안한다. 헤드셋과 로봇이 함께 탐지한 landmark를 shared semantic environment mapping으로 통합하고, 이를 real-world AR 오버레이와 context-aware WIM에 동시에 임베딩해 사용자가 의미 있는 위치를 기준으로 명령을 내릴 수 있게 했다. 여기에 VLM(vision-language-model) 기반 scene-aware view management로 가상 콘텐츠의 color·size·orientation을 런타임에 함께 조정해 먼 거리에서도 가독성을 유지하도록 설계했다.

- **Technical Challenges**: 핵심 과제는 멀리 있는(해상도 저하)·가려진 상황에서도 사용자가 가상 마커의 깊이/스케일/상대 위치를 신뢰하며 조작하게 만드는 것과, 야외의 복잡한 배경 변화에 따라 color·scale·각도 조정이 서로 충돌하지 않게 하는 것이다. fARfetch는 Grounding DINO와 Segment Anything으로 landmark를 정교화하고, CLIP 임베딩으로 중복을 줄여 shared semantic map을 만들며, 로봇 라이다의 point cloud를 누적해 WIM의 구조(메시)를 생성·병합해 의미+기하를 함께 제공한다. legibility 유지 측면에서는 카메라 frustum에 새 가상 콘텐츠가 들어올 때 VLM에 장면 이미지를 입력하고, 렌더 state(색/크기/방위)와 지각 맥락을 구조화 프롬프트로 제시해 VLM이 joint adaptation을 산출하도록 했다.

- **Empirical Impact**: 실험은 30.5m 규모의 실제 야외 inspection 과제를 대상으로 within-subjects 설계(N=13)를 적용했으며, fARfetch가 non-AR baseline 대비 완료 시간을 66% 단축했다. 또한 NASA-TLX 하위 지표에서 mental demand(-43%), temporal demand(-34%), frustration(-66%)을 유의미하게 낮춰 인지·시간 부담을 줄였다는 결과를 보였다. 맞춤 legibility survey와 콘텐츠 가독성 평가는 large outdoor 환경에서도 VLM-driven adaptation이 가상 콘텐츠 legibility를 효과적으로 유지함을 뒷받침하며, 야외 collocated AR 로봇 제어 실사용 가능성을 강화하는 의미가 있다.



### Toward Low-Latency Vision-Language Models with Doubly-Correct Predictions in Egocentric Visual Understanding (https://arxiv.org/abs/2606.25160)
Comments:
          International Conference on Intelligent Robots and Systems (IROS) 2026

- **Prior Approaches**: 기존 VLM pruning은 주로 accuracy만 최적화해 모델 크기와 FLOPs를 줄이지만, 인간-로봇 협업(HRC)에서 중요한 안전성 지표인 doubly-correct prediction(DCP)의 보존을 충분히 다루지 못했다. 또한 egocentric 비전의 공간·시간 근거까지 함께 평가하는 프로토콜이 부족해, pruning이 ‘정답 이유’를 얼마나 유지하는지 진단하기 어려웠다. 결과적으로 일부 방법은 근거 localization은 유지하는 듯 보여도 실제 결정(prediction)은 흔들리는 위험한 불일치가 생길 수 있다.

- **Core Contribution**: 논문은 egocentric 비디오에서 정답 라벨과 시공간 근거(언제/어디의 조작 대상)를 동시에 평가하는 spatio-temporal DCP 프로토콜을 제안한다. 이를 통해 기존 pruning이 종종 “올바른 근거는 남기지만, 그 근거가 올바른 결론으로 이어지지 않는” 현상을 드러낸다. 이어서 rational-informed pruning으로 근거와 의사결정의 정렬을 강화해 DCP 능력을 함께 보존하는 방법을 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 pruning이 backgroud(배경) 활성에 편향되어 근거→로짓으로 이어지는 전달을 끊어 WR(틀린 예측·맞는 근거) 같은 안전 취약 상태를 유발할 수 있다는 점이다. 저자들은 레이어별 중요도가 다르다는 가정을 바탕으로 가중치 magnitude로 비균일 per-layer pruning ratio를 추정하고, 모델이 생성한 rationale을 근거 마스크로 활용하되 의미 정합성(설명 객체-예측 라벨 임베딩 유사도)에 따라 마스크를 adaptive하게 보정해 배경 영향과 노이즈를 완화한다. second-order Taylor/OBS 관점에서 손실 증가를 근사해 가중치 중요도를 계산하고, 이 중요도에 따라 스파스화를 수행한다.

- **Empirical Impact**: EPIC-KITCHENS VISOR와 EgoExo4D에서 기존 OMP/GMP/다른 기준 pruning 대비 prediction accuracy뿐 아니라 IR(Inference Reliability)과 RR(doubly-right samples)을 더 높게 달성한다. 예를 들어 EPIC-KITCHENS VISOR에서 pruning ratio 30%일 때 spatial IR과 RR이 ECoFLaP 대비 각각 13.62%, 3.65% 향상되었고, PT(Prediction Trustworthiness)는 대부분 유지된다. 절제 실험에서도 rationale 정보와 마스크의 adaptive semantic factor가 빠지면 DCP 성능이 크게 저하되어, ‘안전한 pruning’을 실증적으로 뒷받침한다.



### SwarmFly: A simulation platform for UAV swarm experiment design and validation (https://arxiv.org/abs/2606.25146)
- **Prior Approaches**: 기존 UAV 스웜 시뮬레이터는 (1) 유지보수가 중단됐거나 (2) 첫 실험까지 학습 비용이 크며 (3) 단일 시나리오·단일 조정 방식에 고정된 경우가 많다. 그 결과 장애·환경교란·에너지·공역 제한을 반복 주입하고 동일 기준으로 비교·정량화하려면 매번 별도 실험 환경을 새로 구축해야 했다.

- **Core Contribution**: SwarmFly는 MATLAB 기반 멀티-UAV 스웜 시뮬레이션·테스트 플랫폼으로, 실시간 맵과 4가지 스웜 coordination mode(leader-follower, decentralized, heterogeneous relay, heterogeneous speed)를 제공한다. 핵심은 plugin architecture로, 연구자가 행동·고장모델·분석도구를 코어 코드 수정 없이 추가하고 조합할 수 있게 한 점이다.

- **Technical Challenges**: 실시간 GUI(10–30 Hz)에서 플러그인 효과가 물리·렌더링 순서에 자연스럽게 합성되도록 해야 했고, 또 플러그인 추가로 인한 성능 저하나 시뮬레이션 중단을 막는 것이 과제였다. SwarmFly는 handle-class 단일 코어와 tick마다 step 실행 순서를 고정하고, drawnow limitrate 및 렌더링 핸들 재사용으로 프레임 변동을 줄였으며, 플러그인은 onStep에 try-catch를 적용해 실패가 전체 실행을 중단하지 않게 설계했다.

- **Empirical Impact**: 플랫폼은 formation accuracy, wind tolerance, fault recovery, energy endurance, airspace compliance를 포함한 8개 실험으로 각 서브시스템을 검증·특성화했다. 또한 fault injection(예: GPS denied, 모터 고장, comm blackout 등)과 metrics, battery, collision avoidance, 3D visualization, 자동 시나리오 회귀 테스트 플러그인을 묶어 재현 가능한 정량 평가 흐름을 제공함으로써 스웜 제어·행동 메커니즘 연구의 실험 비용과 일관성 문제를 줄이는 데 의미가 있다.



### Memory Retrieval in Visuomotor Policies for Long-Horizon Robot Contro (https://arxiv.org/abs/2606.25136)
Comments:
          16 pages, 5 tables, 8 figures

- **Prior Approaches**: 기존 접근은 부분관측 환경에서 기억을 다루기 위해 (1) belief-state처럼 상태를 명시적으로 모델링하거나, (2) object-centric 메모리·유사도 임베딩 기반 검색·hand-designed 저장 규칙 등으로 과거 정보를 줄여 사용했다. 또 history를 압축해 쓰는 Scene Memory Transformer나 Token Merging 같은 방식도 있었지만, 압축 과정에서 저수준 행동에 필요한 세부가 사라지기 쉽다. 그 결과 작업 정보가 특정 표현/규칙에 잘 맞지 않으면 확장성이 떨어지고, long-horizon에서 오차가 누적되면 drift와 cascading failure가 발생하기 쉽다는 한계가 있었다.

- **Core Contribution**: HALO는 long-horizon robotic imitation learning에서 attention 기반 메모리 검색을 end-to-end로 안정화한 visuomotor policy다. 핵심은 (1) vision-language model(VLM)의 작업 관련 priors를 video question–answer(VQA)로 증류해 “무엇을” 기억에서 꺼낼지 학습시키고, (2) top-k sparsification으로 “어느 정도만” 과거에 의존할지 제한해 닫힌고리 실행 중 오차 누적 영향을 줄이는 것이다. 이를 통해 최대 8분 길이의 과거 경험을 활용하면서도 모델 드리프트를 완화한다.

- **Technical Challenges**: 첫째, offline imitation learning에서 attention이 과거 관측-행동 사이의 spurious correlation을 학습하면 테스트에서 성능이 무너진다. HALO는 VLM이 생성한 VQA 질문-정답 쌍으로 retrieval이 과업 관련 구간을 겨냥하도록 공동 학습하고, VLM의 환각/장문 맥락 오류를 줄이기 위해 멀티스테이지 VQA 생성·필터링 파이프라인을 쓴다. 둘째, 예측 오차가 메모리에 잡음으로 들어가 long-horizon에서 증폭되는 문제를 top-k 비중분 attention으로 완화하며, 비미분 top-k를 straight-through estimator로 학습한다.

- **Empirical Impact**: HALO는 ReMemBench에서 다양한 메모리 유형(공간·관계·수·이벤트 시간)을 다루는 실험에서 평균 절대 성공률을 약 7%p 개선했으며, hand-designed 규칙 기반(-12%p), commonly 쓰이는 task-specific feature(-21%p) 대비 우수함을 보였다. 또한 VLM priors만 쓰는 경우(18%p)보다 HALO가 41%p까지 성능을 끌어올려 “priors + action-grounded retrieval”의 효과를 실증했다. 실세계 로봇 과제에서도 표준 Transformer 대비 약 19% 개선이 관찰되어, long-horizon 집안 환경형 작업에서 범용성 있는 메모리 검색이 유효함을 시사한다.



### Causality-Based Parametric Control Barrier Function for Safe Multi-Vehicle Interaction (https://arxiv.org/abs/2606.25134)
Comments:
          accepted ICRA 2026

- **Prior Approaches**: 기존 연구는 데이터 기반으로 주변 차량(또는 로봇)의 미래 움직임을 예측하거나, 관측된 행동으로부터 이웃 에이전트의 underlying controller를 추정해 안전성을 확보하려 했다. 하지만 다차량 상호작용에서는 차량 간 영향이 섞여 어떤 행동이 어떤 안전제약을 “촉발”했는지 인과를 분리하기 어렵고, 많은 접근이 worst-case에 치우친 보수적 분석으로 인해 과도하게 소극적인 주행이 나타날 수 있다.

- **Core Contribution**: 이 논문은 Parametric-Control Barrier Function(Parametric-CBF)을 다중 로봇 상호작용에 확장하면서, Granger causality 기반 인과 추론을 내장해 차량 간 influence를 명시적으로 다룬다. CMS(Cross Map Smoothness)로 쌍대 상호작용을 감지하고, 그 결과로 학습에 활용할 관측 구간을 자동 선별해 ego 차량이 학습된 기대에 맞춰 적응적으로 안전 제어를 수행하도록 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 안전제약이 실제로 언제 활성화되는지 정확히 식별하고 (2) 그 활성 순간만 골라 Parametric-CBF의 안전행동 벡터 α를 추정하며 (3) 잡음과 관측 오차 속에서도 추정값의 유효성을 보장하는 것이다. 논문은 constraint activation을 CMS 인과 점수로 판별해 constrained optimization을 unconstrained 형태로 바꾸고, 추정된 α가 단조성(α 비음수), 제약 만족(ḣ+αH+δc≥0), 상호작용(경계 근접), 시간 불변성(이전 추정과의 정합)을 통과하는 경우에만 유효 추정치를 축적·평균함으로써 강건성을 확보한다.

- **Empirical Impact**: 라운드어바웃 등 상호작용이 잦은 시나리오에서, 학습된 Causality-based Parametric-CBF를 통해 안전을 유지하면서도 다차량의 motion flexibility를 활용해 과업 효율을 크게 개선함을 보인다. 즉, 기존의 보수적 worst-case 중심 접근보다 덜 방해적이면서도 안전 보장을 유지하는 방향으로 실증적 성과를 제시하며, 다에이전트 로보틱스/자율주행의 안전학습 제어 패러다임을 확장한다.



### RGB: RL Guided Whole-Body MPPI for Humanoid Contro (https://arxiv.org/abs/2606.25123)
Comments:
          7pages

- **Prior Approaches**: 기존에는 모델 기반 whole-body MPC가 동역학 제약을 명시적으로 다루며 해석 가능성과 안전성을 제공하지만, 최적화 비용과 접촉 비선형성 때문에 실시간 구현이 까다롭다. 반면 deep RL은 안정성과 강인함을 잘 보이지만, 학습 보상과 command interface에 행동이 강하게 묶여 새로운 피드백 목적(예: 스윙 풋 클리어런스, 골반 높이 조절)을 추가하려면 보상 재설계와 재학습이 필요했다. 샘플링 기반 MPPI도 가능하지만 고차원 휴머노이드에서 샘플 분포/초기화 민감성과 접촉 조건 변화에 대한 취약성이 문제로 지적된다.

- **Core Contribution**: 논문은 미리 학습된 RL 정책을 최종 제어기가 아니라 MPPI의 sampling prior로 쓰는 “RL-guided whole-body MPPI” 프레임워크를 제안한다. 이렇게 하면 새로운 태스크 목적은 RL 정책을 건드리지 않고 MPPI의 modular cost term으로만 추가해도 되고, MPPI가 온라인으로 RL prior를 교정하며 closed-loop 피드백을 형성한다. 즉, plug-and-play 형태로 로버스트한 휴머노이드 보행은 유지하면서 정밀한 whole-body 목적 주입을 가능하게 한다.

- **Technical Challenges**: 핵심 난제는 vanilla MPPI가 샘플 통계(분산, 평균)와 초기화에 민감해 접촉 많은 환경에서 비현실 궤적이 다수 발생할 수 있다는 점과, 고차원 제어수열 최적화가 계산량을 폭증시킨다는 점이다. 저자들은 RL 정책 출력을 샘플 분포의 mean으로 사용해 동적으로 feasible한 롤아웃에 샘플을 편향하고, 제어수열은 knot-based sampling으로 저차원으로 파라미터화해 탐색 차원을 줄였다. 또한 물리 엔진 기반 병렬 rollout(미분 불필요)을 비동기 receding-horizon으로 구성해 접촉 비부드러움에도 안정적으로 비용을 평가하고 업데이트한다.

- **Empirical Impact**: MuJoCo에서 29-DoF Unitree G1 휴머노이드로 검증했으며, CPU 기반 병렬 롤아웃을 사용해 평균 effective update rate 약 280 Hz를 달성했다. 동일 command interface 조건에서 순수 RL 대비 태스크 정밀도가 개선되는데, 특히 straight walking의 누적 드리프트를 크게 줄여 측면(가로) 위치 오차 RMSE를 0.339 m에서 0.022 m로 낮췄다. 더 나아가 squat의 base-height regulation처럼 RL 커맨드로 직접 다루기 어려운 추가 whole-body 목표도 비용 조성(cost composition)만으로 추적해 재학습 없이 자세 전이를 생성할 수 있음을 보였다.



### AeroCast: Probabilistic 3D Trajectory Prediction for Non-Cooperative Aerial Obstacles via Transformer-MDN Architectur (https://arxiv.org/abs/2606.25122)
- **Prior Approaches**: 기존 접근은 칼만 필터와 IMM 같은 물리 기반 추정으로 비협조 장애물의 상태를 갱신하지만, 미리 가정한 운동 모델(정속·coordination turn)이 실제 변칙 동작을 잘 못 따라가 예측이 뒤처질 수 있다. 학습 기반 recurrent 모델은 온보드에 빠르지만 대개 점(point) 예측만 제공해, 불확실성이나 행동 분기 지점을 하류 플래너가 구분하기 어렵다.

- **Core Contribution**: 본 논문은 비협조 항공 장애물의 미래 위치를 확률적으로 예측하는 AeroCast를 제안한다. Pre-LN Transformer 인코더에 Mixture Density Network(MDN) 출력 헤드를 결합해, 각 타임스텝의 3D 변위(displacement)에 대해 Gaussian mixture 분포를 직접 출력한다.

- **Technical Challenges**: 핵심 난관은 (1) 비협조 장애물의 다중 모드 동작을 mixture로 표현하되, (2) NLL만 최적화하면 mixture 모드가 기하학적으로 붕괴(mode degeneracy)하는 문제, (3) 입력 표현이 translation에 민감해 분포가 불안정해지는 문제다. AeroCast는 연속 변위 기반 translation-invariant 인코딩과 변위 스케일 정규화, mode-anchoring MSE 항 및 센서 노이즈에 맞춘 sigma floor를 함께 사용해 모드 붕괴와 불리정 정규화를 동시에 완화한다.

- **Empirical Impact**: Vicon 기반 실데이터와 합성 데이터를 섞은 쿼드로터 코퍼스(9개 모션 카테고리)에서 AeroCast는 5초 예측 구간에서 기존 베이스라인 대비 ADE/FDE를 약 50% 수준으로 낮추고, NLL과 Continuous Ranked Probability Score(CRPS)에서도 최고 성능을 보였다. 또한 ablation 결과 velocity 입력과 모델 capacity가 예측 품질에 가장 큰 영향을 주며, positional encoding은 장기 일관성에 필수로 나타났고, 추론은 샘플당 0.1ms로 실시간(100Hz) 탑재 가능성을 보여준다.



### SurveilNav: Collaborative Object Goal Navigation with Robot and Surveillance System (https://arxiv.org/abs/2606.25119)
Comments:
          Accepted by ICRA 2026

- **Prior Approaches**: 기존 ObjectNav(객체 목표 내비게이션) 연구는 LLM/VLM을 활용해 지도나 탐색 전선을 구성한 뒤, 유의미한 지점을 선택하며 탐색 효율을 끌어올리는 흐름이 주를 이뤘습니다. 하지만 대다수 방법이 단일 로봇(단일 시점)에서만 동작해 시야 한계와 가림(occlusion)으로 인해 대규모·다층 환경에서 탐색 정확도와 효율이 떨어진다는 한계를 보였습니다. 또한 데이터 기반 접근은 학습된 범주에 한정되거나 시뮬레이션-현실 간 전이 격차가 문제로 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 로봇-감시 인프라 협업을 전제로 한 새로운 과제인 collaborative object-goal navigation을 정의하고, 이를 위한 indoor 협업 내비게이션 데이터셋을 제안합니다. Habitat-Sim 기반으로 36개 씬, 74개 층, 206개 감시 카메라(각 13 뷰)로 구성된 HM3D용 벤치마크를 구축해, 다중 관측을 활용하는 능력을 체계적으로 평가할 수 있게 했습니다. 아울러 SurveilNav은 active camera scheduling, joint 2D/3D mapping, VLM 기반 value estimation, multi-view target verification을 통합한 협업 프레임워크로, 고정 카메라의 사각지대와 단일 로봇의 제한된 인지 범위를 동시에 겨냥합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 로봇의 현재 위치/층과 감시 카메라의 관측을 효율적으로 연결해 불필요한 크로스-플로어 정보를 줄이고, (2) 다중 시점에서 신뢰도 있게 객체를 대응·검증하며, (3) 그 결과를 탐색 정책(어디로 갈지)으로 안정적으로 변환하는 것입니다. SurveilNav은 층 정합을 기반으로 활성 카메라 부분집합을 동적으로 선택하고, 로봇 egocentric 관측과 감시 third-person 관측을 joint 3D 맵으로 융합한 뒤 2D frontier 맵과 joint value map을 함께 생성합니다. 또한 CLIP/VLM 관련 점수로 의미적 가치를 추정하고, GroundingDINO+MobileSAM 기반 다중 뷰 객체 연관·점수 진화(confidence evolution)로 오탐을 줄이며 waypoint을 선택해 탐색-검증을 end-to-end에 가깝게 연결합니다.

- **Empirical Impact**: HM3D 실험에서 SurveilNav은 탐색 효율(SPL)과 성공률(SR) 모두에서 기존 방법을 크게 능가하며 state-of-the-art를 달성했습니다. 단일 로봇 계열과 비교해 SR과 SPL이 각각 +7.7, +6.7 개선됐고, 감시 입력을 제거하면 SPL/SR이 하락해 협업이 실제 성능 향상에 직접 기여함을 확인했습니다. 추가 분석에서 카메라 지각 범위·밀도가 커질수록 성능이 개선되었으며, VLM/CLIP 기반 value map을 region-centric으로 구성할 때 가장 높은 SR/SPL이 나와, 다중 관측 기반 의미 추론의 실효성을 입증했습니다.



### ADM-Fusion: Adaptive Deep Multi-Sensor Fusion for Robust Ego-Motion Estimation in Diverse Conditions (https://arxiv.org/abs/2606.25111)
Comments:
          8 pages, 4 figures

- **Prior Approaches**: 기존 ego-motion 추정은 기하학 기반 파이프라인(예: SLAM, 최적화/필터)에서 시작해, 학습 모듈을 섞은 하이브리드 방식과 raw 멀티모달을 end-to-end로 결합하는 딥 러닝 융합으로 발전했다. 그러나 많은 방법이 고정 가중치나 안정적인 appearance/상응 관계/잡음 가정을 전제로 해서, 카메라-라이다-레이더-IMU가 서로 다른 방식으로 악화될 때(열화 패턴 상이) 강건성이 떨어질 수 있다. 이에 따라 다양한 센서 신뢰도 변동을 실시간으로 반영하는 적응형 융합의 필요성이 강조된다.

- **Core Contribution**: ADM-Fusion은 센서 열화와 환경 변화에 따라 멀티모달 기여도를 실시간으로 재가중하는 end-to-end 융합 프레임워크를 제안한다. 핵심은 Adaptive Sensor Mixture-of-Experts(ASMoE)로, content-aware routing을 통해 각 타임스텝에서 센서 가중치를 동적으로 산출하는 점이다. 또한 translation과 rotation을 분리된 브랜치로 학습하되, cross-task attention으로 두 작업 간 보완 정보를 교환해 task-specific specialization은 유지한다.

- **Technical Challenges**: 문제는 센서마다 실패 양상이 다르고 시간에 따라 신뢰도가 빠르게 변하므로, 고정된 가중치로는 융합이 붕괴할 수 있다는 점이다. 이를 위해 저자들은 (1) 센서별 특징을 Mamba로 시간적으로 보정한 뒤 (2) 타임스텝별 ASMoE 라우터로 신뢰도 기반 가중치를 할당하고 (3) router collapse를 막기 위한 uniform expert usage 정규화와 (4) translation/rotation 분리 학습 뒤 cross-attention으로 오차 전파를 제어한다. 시뮬레이션-실세계 차이를 줄이기 위해 CARLA-LOC 사전학습 후 KITTI에서 fine-tuning 전략도 함께 사용한다.

- **Empirical Impact**: 실험에서는 CARLA-LOC에서 학습 후 KITTI로 fine-tuning한 simulation-to-real transfer가 확인되며, 열화 조건에서도 견고성을 유지하면서 기존 방법과 경쟁/우위를 보인다. 특히 KITTI에서 rotation 드리프트는 LVIO 구성에서 가장 일관된 개선을 보였고, ASMoE가 IMU는 rotation에 더 크게, 라이다/카메라는 translation 안정화에 더 크게 기여하도록 시간적으로 부드럽게 가중치를 조절함이 관측된다. 또한 radar를 추가하면 translation RPE가 64% 감소하는 등 센서별 강점을 task에 맞게 비대칭적으로 활용하며, 라우팅-시간 설계의 중요성과 함께 ASMoE의 적응 전략이 실제로 효과적임을 뒷받침한다.



### Invariant Kalman filtering for extended pose estimation in multi-IMU articulated rigid-body systems (https://arxiv.org/abs/2606.25083)
- **Prior Approaches**: IMU 기반 관절형(articulated) 확장 자세 추정은 보통 EKF와 IterEKF 계열로 해결해 왔다. 하지만 이들은 강체 운동의 Lie group 구조를 적극 활용하지 못해 수렴성과 일관성(consistency)이 떨어질 수 있다. IEKF는 단일 강체에서 관측불가능성(unobservability) 하의 수렴/일관성 보장이 있지만, 관절계에서는 프레임 간 포즈 결합과 관절 제약을 invariant하게 넣는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 관절형 kinematic-tree 시스템을 위한 Lie group 상태 표현으로 relative L-extended pose를 제안한다. 이 표현은 여러 IMU를 가진 시스템에서 group-affine 동역학을 만들고, 관절(예: spherical, hinge) 제약을 invariant 형태로 쓸 수 있게 해 준다. 이후 IterIEKF(Iterated IEKF)에 제약을 noise-free pseudo-measurement로 통합하여, invariant filtering의 수렴/일관성 보장 확장 문제를 정면으로 해결한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 단일 강체용 IEKF의 invariant 조건을 관절계 포즈 결합에 그대로 적용할 수 없다는 점과 (2) 잡음이 없는 관절 제약을 invariant 업데이트에 안정적으로 포함하는 점이다. 저자들은 relative L-extended pose를 SE_{2}^{L}(3) 기반 Lie group으로 구성해 group-affine 성질을 확보하고, 관절 제약은 iterated IEKF의 pseudo-measurement로 넣어 제약 잔차가 누적되는 문제를 줄이도록 설계했다.

- **Empirical Impact**: 실험은 UR5e 로봇 팔 pick-and-place와 사람 다리 전진 런지(leg) 두 태스크에서 수행됐으며, 제안한 IterIEKF가 모든 EKF/IterEKF 및 absolute-pose 기반 iterated IEKF 베이스라인을 능가했다. 또한 더 빠르게 수렴하고 실행 간 변동성(run-to-run variability)이 낮으며, 시나리오 전반에서 RMSE가 최우수 기준 대비 최소 50% 이상 감소했다. IMU만으로 관절형 확장 포즈를 invariant하게 안정 추정하는 접근이 실증된 셈이라, 로봇 모션 캡처와 웨어러블 기반 자세 추정에 실질적 파급이 기대된다.



### BFMTrack: Latent Sequence Optimization for Physics-Based Motion Tracking with Behavioral Foundation Models (https://arxiv.org/abs/2606.25056)
- **Prior Approaches**: Behavioral Foundation Models(BFMs)는 reward-free 상호작용과 대규모 모션 데이터로 물리적으로 그럴듯한 제어 능력을 폭넓게 학습하지만, 기존 추적 성능은 주로 stationary 태스크에 집중돼 왔다. 모션 추적에서는 moving-window 평균화로 여러 프레임의 목표를 하나의 latent로 압축해 시간 순서를 잃는 문제가 있어, 빠른 전이에서는 정밀도와 coherence가 동시에 악화된다.

- **Core Contribution**: 이 논문은 Latent Sequence Optimization(LSO)로 BFMs의 latent를 “한 시점 선택”이 아니라 “시간에 걸친 latent 궤적”으로 최적화해, time-varying 모션 추적을 직접 수행하도록 확장한다. 시뮬레이터 rollouts와 policy gradient 업데이트를 결합하되 reward engineering 없이도 목표 시퀀스를 정밀하게 추적하는 절차를 제안한다.

- **Technical Challenges**: 핵심 난제는 시뮬레이션 기반 목표가 미분 불가능해 end-to-end gradient를 직접 적용하기 어렵다는 점이며, 이를 REINFORCE 계열 policy gradient로 풀어 latent sequence의 확률적 최적화를 수행한다. 또한 latent 간 시간적 일관성을 유지하기 위해 temporally correlated(1/f^β) colored noise로 탐색을 설계하고, unit hypersphere 제약을 위해 업데이트 후 L2 정규화를 적용한다; sparse keyframe의 경우에는 키프레임 사이 보간(SLERP)과 중간 보상 0 설정으로 under-determined 구간을 regularize한다.

- **Empirical Impact**: Dense tracking, sparse keyframing(인betweening), 그리고 모션 stitching까지 Isaac Sim의 SMPL 휴머노이드와 실제 휴머노이드 로봇에 대해 검증했으며, 전반적으로 zero-shot BFM 추론 대비 모든 평가 지표에서 우수한 성능을 보였다고 보고한다. 특히 sparse keyframe 입력이 충분히 동적이고 접촉이 많은 경우에도 물리 기반으로 그럴듯한 전신 모션을 생성해, 기존 kinematic 또는 전용 tracking policy 대비 실용적 의미가 크다.



### ROAD-VLA: Robust Online Adaptation via Self-Distillation for Vision-Language-Action Models (https://arxiv.org/abs/2606.25800)
- **Prior Approaches**: 기존 VLA는 사전학습으로 언어·비전·행동을 토큰화해 범용 조작을 제공하지만, 실제 배치에서는 시각·구성·실행 수준의 분포 이동으로 온라인 적응이 필수입니다. RL 기반 적응은 PPO/GRPO/DPO 등으로 시도됐으나, 로봇 보상은 희소·지연·노이즈가 많아 고차원 autoregressive 행동 토큰에 대한 신호가 약하고 업데이트 분산·불안정·기본 능력 망각을 유발합니다. self-distillation도 가능하지만, 시연/경험/고수준 계획에 조건한 텍스트 privileged teacher는 언어 지침과 저수준 행동 간 modality gap 때문에 VLA 적응에 잘 통하지 않았습니다.

- **Core Contribution**: 이 논문은 ROAD-VLA라는 advantage-guided self-distillation 프레임워크를 제안합니다. 핵심은 현재 정책에서 advantage 추정을 통해 action-token 공간에 직접 proximal teacher를 만들고, 희소 보상을 토큰 수준의 조밀한 감독 신호로 바꿔 주는 것입니다. 또한 calibrated advantage와 teacher matching 조건 하에 정책 개선 하한(policy-improvement lower bound)을 유도해 이 접근의 학습 목표가 단순 경험적 트릭이 아님을 보장합니다.

- **Technical Challenges**: 기술적 난제는 (1) advantage를 신뢰성 있게 토큰 단위로 확장해 조밀한 감독을 만들고, (2) 잘못 보정된 teacher가 학습을 망치지 않게 신뢰구간을 유지하는 것입니다. ROAD-VLA는 온정책 상태에서 토큰별 logit을 advantage 부호에 따라 교란해 로컬 KL-regularized 개선 해(지수 기울기)를 닮은 teacher를 닦아 만들고, intrinsic advantage와 frozen PPO critic의 advantage를 sign이 일치할 때만 섞어 calibration을 완화합니다. 이후 teacher-to-student KL로 autoregressive 행동 전체에 대해 distillation을 수행해 PPO식 scalar 재가중만으로는 어려웠던 고차원 행동 토큰 학습을 안정화합니다.

- **Empirical Impact**: 실험은 OpenVLA-7B를 베이스로 7개 로봇 조작 환경에서 in-distribution과 out-of-distribution shift(시각 강건성, 구성적 추론, 실행 강건성)까지 폭넓게 평가하며, ROAD-VLA가 거의 모든 설정에서 PPO를 능가했습니다. ID는 성공률 85%→88%, OOD는 69%→73%로 개선되고 평균 degradation도 16.3%→14.6%로 줄어들어 분포 이동에 대한 전반적 전이성이 확인됩니다. 특히 시각 및 실행 수준 OOD에서 감소 폭이 크게 나타났고, 온라인 상호작용의 샘플 효율에서도 ROAD-VLA가 mid-training 구간에서 PPO보다 일관되게 앞서는 등 더 강한 online adaptation 성능을 보였습니다.



### Memory-Efficient Policy Libraries with Low-Rank Adaptation in Reinforcement Learning (https://arxiv.org/abs/2606.25700)
- **Prior Approaches**: 멀티태스크 RL에서는 catastrophic forgetting을 막기 위해 작업별 specialist 정책을 따로 학습해 라이브러리로 저장하는 방식이 흔하지만, 저장/메모리 비용이 커서 로봇 적용이 어렵습니다. 단일 정책으로 모든 태스크를 함께 학습하는 접근은 interference와 conflicting gradients 문제로 난도가 높습니다. 또 LoRA 같은 PEFT는 주로 LLM에 적용돼 로보틱스·RL에서의 이식성은 상대적으로 덜 탐구돼 왔습니다.

- **Core Contribution**: 이 논문은 PPO 기반 온라인 RL에서 LoRA(저랭크 어댑테이션)를 fine-tuning에 적용해, 작업별 정책 라이브러리를 메모리 효율적으로 만드는 방법을 제안합니다. 특히 한 개의 기준 정책(base policy)을 학습한 뒤, 여러 목표 태스크에 대해 LoRA로 specialist 정책을 생성하는 파이프라인을 구성합니다. 핵심은 LoRA가 전체 레이어 fine-tuning과 비슷한 성공률을 유지하면서도 저장 부담을 크게 줄일 수 있음을 보인 것입니다.

- **Technical Challenges**: 로보틱스 RL에서 LoRA rank(1~12)에 따라 표현력과 학습 안정성이 달라져, 낮은 rank에서 성능 저하가 발생할 수 있는 점이 기술적 난제입니다. 저자들은 Meta-World의 MT1 설정에서 pick-place로 기준 정책을 만든 뒤, rank별 LoRA 업데이트와 전체 fine-tuning을 PPO로 비교하며 조기 종료(평가 성공률 98%)로 학습을 효율화했습니다. 또한 LoRA는 PPO 학습에서 backward 계산을 줄이는 방식으로 계산 효율 이득을 얻되, forward 오버헤드는 작게 유지된다는 점을 FLOPs 추적으로 확인했습니다.

- **Empirical Impact**: 결과적으로 LoRA는 rank에 따라 학습 가능한 파라미터를 20~160배 줄이며, 전체 fine-tuning과 거의 비슷한 성공률을 유지했습니다(태스크별로 난이도 차이는 존재). 저장 관점에서는 base policy 2.7MB에 LoRA 파라미터만 추가할 때, 10개 specialist 정책을 저장하는 경우 전체 fine-tuning 저장 대비 약 85~92% 절감 효과가 나타났고, 10~50개 정책 라이브러리 스케일에서 swap-memory 없이 운영할 가능성을 시사합니다. 단, 시뮬레이터에서의 계산 병목이 데이터 수집에 있을 수 있어 실제 구동 속도 개선폭은 제한적일 수 있다는 점도 함께 언급됩니다.



### Reasonable Motion: A General ASP Foundation for Environment Constrained Movement Trajectory Computation (https://arxiv.org/abs/2606.25626)
Comments:
          Accepted at: LPNMR 2026 - 18th International Conference on Logic Programming and Non-monotonic Reasoning, 7 - 11 September 2026 - Klagenfurt, Austria

- **Prior Approaches**: 기존 모션 예측은 end-to-end 방식으로 관측에서 미래 궤적을 직접 학습하는 경우가 많지만, 환경의 구조(그래프/지도 토폴로지)나 허용 가능한 행동, 이벤트의 근거가 명시적으로 드러나지 않는 한계가 있었다. 그 결과 서로 질적으로 다른 행동(직진 vs 좌회전)이 겹치는 기하학적 경로로 표현될 수 있고, 왜 그 궤적이 선택됐는지, 어디서 제약이 작동했는지 검증·설명하기가 어렵다는 지적이 있었다.

- **Core Contribution**: 이 논문은 answer set programming(ASP) 기반의 하이브리드 정성-정량 방법으로, 제약을 만족하는 “reasonable” 분기(branching) 궤적 모드를 계산한다. 안정 모델(stable model)을 통해 기하학적으로 허용되는 행동을 이벤트 시퀀스와 지도 위 규범/선호까지 포함한 별도 모드로 열거하고, 각 모드에 대해 연속적인 궤적을 생성해 고수준 모드-근거를 추적 가능하게 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 연속 경로 탐색을 정성적 그래프 순회로 바꾸면서도 (2) 물리/규범 제약을 엄밀히 만족시키고 (3) 계산 복잡도를 제어하는 것이었다. 이를 위해 그래프 reachability를 별도 귀납 규칙으로 먼저 유도해 탐색을 “평탄화”하고, choice rule로 경로를 선택한 뒤 integrity constraint로 시작/연결성/연속성을 강제하며, 선호는 weak constraints로 랭킹하되 cost-optimal stable model들은 모두 유지해 모드 다양성을 보존한다.

- **Empirical Impact**: 자율주행 벤치마크 Argoverse 2에서 계산 효율(grounding이 지배적이고 solving은 매우 짧음)과 기하학적 모드 생성의 충분성, 예측 정확도를 함께 평가하며 방법의 실용성을 확인한다. 무엇보다 각 예측 궤적이 안정 모델과 이벤트 핑거프린트에 의해 완전히 traceable되므로, 순수 학습 기반 모델이 제공하기 어려운 verifiable interpretability를 제공하는 것이 이 접근의 의미다.



### Reflective VLA: In-Context Action Consequences Make VLAs Generaliz (https://arxiv.org/abs/2606.25215)
- **Prior Approaches**: 기존 vision-language-action(VLA) 모델은 보통 reactive 방식으로, 현재 관측과 지시만으로 다음 행동을 예측해 배치 환경에서 필요한 상태를 단일 프레임으로 충분히 알 수 있다고 가정한다. 하지만 실제 로봇에서는 카메라-로봇 기하, 캘리브레이션, actuation bias 같은 embodiment-specific 요인이 한 장면에서 식별이 어렵고, 그 결과 학습 환경에 과적합되며 배치 일반화가 약해진다. 최근 temporal context·메모리 기반 VLA는 상태 추정은 개선하지만, 실행된 행동과 그 결과(관측 변화)를 명시적으로 연결해주는 ‘action–consequence binding’이 부족하다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Reflective VLA를 제안하며, 매 의사결정을 observation–action–consequence triplets의 컨텍스트에 조건화하는 in-context learning 관점으로 전환한다. triplet은 “무엇을 봤고 무엇을 실행했으며, 실행 후 장면이 어떻게 바뀌었는지”를 함께 기록해 배치 환경에서 행동-효과 매핑을 드러내도록 설계됐다. 모델은 각 결정을 과거 triplet과 현재 관측을 함께 참조해, 단일 프레임만으로 불가능한 embodiment 요인 분해를 돕는다.

- **Technical Challenges**: 핵심 구현 난제는 (1) 이미지·자세(proprioception)·연속 행동 chunk 등 이질 모달리티를 병목 없이 한 인과 시퀀스로 패킹하는 문제와 (2) 듀얼시스템(VLM prefix + continuous action expert)에서 컨텍스트를 학습·추론 시 효율적으로 전달하는 문제다. 논문은 모든 관측 모달리티를 공유 VLM token space로 라우팅하고, action expert가 과거 관측/행동/결과 증거를 직접 attend하도록 구성한다. 또한 K개의 멀티프레임을 별도 forward 없이 학습하도록 block-causal mask로 학습을 병렬화하고, KV-cached real-time inference가 가능하도록 같은 인과 구조를 추론에도 활용한다.

- **Empirical Impact**: 실험에서 Reflective VLA는 LIBERO와 SimplerEnv-Bridge 같은 표준 설정에서 기준 reactive baseline 대비 성능을 유지(또는 소폭 향상)하며, LIBERO에선 평균 성공률 97.6%로 state-of-the-art급을 보였다. 분포 이동이 큰 LIBERO-Plus 및 LIBERO-Plus-Hard에서는 test-time fine-tuning 없이도 평균 성공률이 각각 5.4, 4.2 percentage points 향상되어 배치 일반화가 개선됨을 확인했다. 특히 history-only ablation은 단순 컨텍스트 길이 증가가 아니라 action–consequence에 해당하는 ‘결과 관측’ 증거가 환경 간 일반화의 핵심임을 뒷받침한다.



### ARTOO-DARTU: Studying AR-HRC With AR Obstruction Mitigation During a Warehouse Task (https://arxiv.org/abs/2606.25202)
Comments:
          To appear in Proceedings of the ACM on Human-Computer Interaction, Vol. 10, No. 5, Article MHCI7668, MobileHCI 2026

- **Prior Approaches**: 기존 AR situated analytics는 HRC에서 로봇의 위치·의도·내부 상태를 실시간으로 보여줘 효율과 이해를 높이는 데 쓰여왔다. 다만 창고처럼 작업 핵심이 ‘라벨과 물체의 가시성’에 의존하는 환경에서는, 로봇에 결합된 AR 오버레이가 이동 중에 시야를 가려 안전성과 사용성을 해칠 수 있다는 지적이 있었다. 또 AR view management나 obstruction 대응은 고정된 환경/미리 모델링된 객체에 기대는 경우가 많아, 로봇과 사용자 움직임이 큰 dynamic warehouse 조건에 그대로 적용하기 어렵다.

- **Core Contribution**: 이 논문은 ARTOO-DARTU(warehouse HRC용 AR 시스템)와 함께, 로봇에 결합된 situated analytics가 실제 라벨·물체를 가리지 않도록 막는 ODM(Obstruction Detection and Mitigation pipeline)을 제안한다. ODM은 사전에 알려지지 않은 새 라벨·객체가 등장하는 환경에서도, AR 콘텐츠-로봇의 공간 결합을 유지한 채 실시간으로 가림을 탐지하고 시각적 완화(투명화)까지 수행하도록 설계됐다. 또한 이를 검증하기 위한 Pocket MonstARs(게임화된 창고 픽킹 유사 과제)로 통제된 사용자 평가 틀을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) AR 콘텐츠가 로봇과 ‘coupled’되어 있어 임의로 위치를 옮기기 어렵고, (2) 가시성을 유지해야 하는 라벨/물체는 예측 불가능하게 변하며, (3) 협업 중단 없이 near real-time으로 대응해야 한다는 점이다. 저자들은 HoloLens에서 가상 요소의 2D projection bounding box를 산출해 서버로 보내고, 서버에서 YOLOv5 기반 object detection과 EasyOCR 기반 text detection의 결과와 겹침(overlap)을 계산해 가림 가능 요소를 플래그하는 방식으로 문제를 해결했다. 그 다음 HoloLens에서는 플래그된 가상 요소를 즉시 완전 투명으로 전환해 라벨 가독성을 보존하면서도 로봇-콘텐츠 정렬 자체는 깨지지 않도록 했다.

- **Empirical Impact**: 34명의 사용자 실험에서 ODM이 활성화된 경우 전체 HRC 작업 효율이 46% 증가했고, 실제 세계 가시성이 필요한 하위 작업은 61% 더 빨라졌다. 또한 ODM이 꺼진 조건에서는 이 이득이 유의미하게 나타나지 않아, ‘가림 완화 파이프라인’이 성능의 전제 조건임을 보여준다. 전체적으로 ARTOO-DARTU의 situated analytics는 현장 AR-HRC에서 효율과 사용자 경험을 동시에 끌어올릴 수 있음을 실증하며, 창고형 동적 환경의 AR 안전 설계에 실질적인 시사점을 제공한다.



### Swarm-Inspired Generation of Collective Behaviors in Graph Dynamical Systems (https://arxiv.org/abs/2606.24958)
- **Prior Approaches**: 기존 연구는 국소 상호작용이 전역 집단행동을 만드는 과정을 설명하거나, 특정 그래프/동역학에 맞춘 동기화 제어 규칙을 설계해 왔다. 다만 그래프 스케일이 바뀌거나 노드 고유 동역학이 달라질 때 학습 규칙이 일반화되지 않거나, 전역 목표(동기화 패턴)를 만들기 위해 재학습이 필요하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 Swarm-Inspired Emergent Synchronizer(SIES)라는 그래프-동역학 프레임워크를 제안해, 원하는 집단 동기화(phase relation 포함)를 만들 수 있는 일반izable한 국소 상호작용 법칙을 학습한다. 각 노드는 상태와 task cue를 가지며, signed source-target-conditioned attention이 명시적 진화 모델 내부의 적응적 결합항으로 작동해 생물 군집처럼 “국소 지능 + 명시적 dynamics”를 결합한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 그래프 스케일/동역학이 바뀌어도 동일한 결합 연산자가 목표 동기화 패턴을 유지하게 만드는 것, (2) 비-진동성(예: overdamped harmonic oscillator)처럼 까다로운 내재 동역학에도 집단 리듬을 강제할 수 있게 하는 것이다. SIES는 CDS(Controlled Dynamical Systems)에서 처방된 동기화 패턴을 생성하도록 generalizable coupling operator를 학습하고, 초기 위상 다양성을 갖춘 RL 학습 에피소드 구성을 통해 전이성을 확보하는 방식으로 이를 해결한다.

- **Empirical Impact**: 실험에서 SIES의 학습된 결합 연산자는 학습 스케일에 없는 네트워크 규모와 목표 위상 관계에서도 재학습 없이 동기화를 재현했으며, 3종 oscillator baseline 대비 gait 관련 모드에 더 빠르게 도달했다. 또한 다리 비활성화 상황까지 포함해 서로 다른 스케일의 시뮬레이션 멀티레그 로봇과 실제 헥사포드에서 synchronization-driven locomotion을 확장 적용했고, 동시에 heterophilous 그래프 표현학습 벤치마크에서 비교 방법 중 최고 성능을 기록했다.



### Conformal Orbit-Valid Trust Horizons for Equivariant World Models (https://arxiv.org/abs/2606.24946)
Comments:
          15 pages, 6 figures

- **Prior Approaches**: 기존 learned world model 기반 신뢰도는 rollout이 “얼마나 버티는지”를 단일한 실험용 숫자처럼 취급하는 경향이 있었고, horizon이 깨질 때의 신뢰 경계를 사전에 정량화하기 어렵다는 한계가 있었다. 특히 많은 방법이 글로벌 도달가능성(reachability) 보장이나 오버헤드 큰 이미지 생성 쪽에 치우쳐, latent regime에서의 유한표본 의미 있는 신뢰 horizon 인증은 상대적으로 부족했다. 또한 equivariance가 있더라도 orbit 전체에 대해 통계적 캘리브레이션 비용이 얼마나 줄어드는지 구조적으로 설명·검증한 연구가 적었다.

- **Core Contribution**: 이 논문은 latent world model에 대해 trust-horizon certification을 split-conformal 캘리브레이션으로 “보수적(anti-conservative가 되지 않음)” 인증으로 바꾸고, 여기에 exact equivariance를 결합해 orbit 전체에 동일한 신뢰 horizon이 성립함을 보인다. 특히 wedge(대표 구간)에서 캘리브레이션된 곡선이 equivariance에 의해 group orbit 전체로 orbit-constant하게 “운반(orbit transport)”되며, 그 결과 orbit-wise 통계적 캘리브레이션 비용을 구조적으로 제거한다. 다만 이 인증은 글로벌 reachability가 아니라 분포 기반(audit distribution 기준) 보수적 보장임을 명확히 경계로 둔다.

- **Technical Challenges**: 핵심 기술 난점은 (1) raw latent error-propagation curve가 실제 롤아웃 오류를 균일 상계로 제공하지 못하는 추정치라는 점, (2) equivariance가 학습 후에도 정확히 성립하지 않는다는 점(approximate implementation residual)이다. 저자들은 one-step latent residual과 유한시간 확장 추정치를 이용해 raw horizon curve를 만든 뒤, split-conformal의 multiplicative factor로 곡선 전체를 one-sided로 보수화해 유한표본 의미를 부여한다. 그리고 encoder/predictor/action transform/latent metric이 요구하는 exact equivariance 조건을 만족하면 orbit transport로 오류와 horizon이 궤도 위에서 일정해지며, 약한 등변성 결함은 orbit certificate를 추가로 인플레이션(보수화)하는 형태로 보정해 approximate residual까지 다룬다.

- **Empirical Impact**: 감사(audit) 기반 재현성 검증에서는 split-conformal factor가  gamma alpha=1.0 으로 충분했고, 50개 안정 감사에서 anti-conservative violation이 0회로 관측되어 위반률에 대한 exact-binomial 95% 상한 5.8%를 제시한다. 또한 구현된 모델의 orbit-transport residual은 median 1.1%, max 4.1%로 작았고, certificate non-vacuous 지표로 certified-to-measured horizon ratio의 median이 0.67이며, median 기준으로도 실용적 수준의 단축 horizon 인증이 가능함을 보여준다. 다만 calibration-cost 절감 효과는 substrate에 따라 달라, 대칭 2D에서는 non-equivariant baseline도 orbit-robust해 분리(separation)가 거의 없었고, 3D yaw 감사에서는 equivariant 모델만 한 섹터에서 안전하면서도 비공허한(one-sector safe & non-vacuous) orbit-valid certificate를 얻는 반면 baseline들은 violation/슬랙/샤프니스 저하 또는 추가 섹터 비용을 치렀다.



### When Do Conservation Laws Survive Learned Representations? Certified Horizons for Latent World Models (https://arxiv.org/abs/2606.24945)
Comments:
          15 pages, including appendices. Code: this https URL

- **Prior Approaches**: 물리 세계 모델에서 보존법칙을 유지하는지 보려는 기존 연구들은 주로 잠재 해밀토니언이나 스칼라 witness 같은 ‘학습된’ 증명 대상을 사용해 왔다. 하지만 이런 증명은 진짜 물리 에너지에는 드리프트가 있어도 잠재 공간에서는 보존처럼 보일 수 있어, 실제 불변량 기준의 certifiable horizon을 직접 보장하기 어렵다.

- **Core Contribution**: 이 논문은 보존을 ‘학습된 latent Hamiltonian’이 아니라, 디코딩해 얻은 물리 불변량을 직접 계산해(certified object) 그 수준집합에서의 유지 가능 기간을 정량화한다. 즉, 모델 결함을 사전에 측정해 rollout이 알려진 불변량의 level set에 얼마나 오래 머무는지 상한인 certified horizon을 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 latent representation 학습 과정에서 생기는 표현 결함, 디코딩/읽기( readout ) 결함, latent dynamics 결함을 모두 horizon 제약식에 분해해 안전하게 합산하는 것이다. 이를 위해 decoded physical invariant 기준의 shell-horizon certificate를 만들고, soft learned witness를 decoded invariant로 “monotone alignment bridge”를 통해 연결해, witness가 비평면적으로(potentially drifting) 있어도 decoded 불변량의 certifiability가 유지되도록 했다.

- **Empirical Impact**: 실험에서는 state, learned-lift, pixel 관측에서 보존 인증이 성립함을 보였다. 특히 hard canonical symplectic 구조는 알려진 좌표계에서는 긴 horizon을 주지만 learned chart를 넘으면 깨지는 경계가 나타났고, 반면 controlled-Lipschitz-aligned soft invariant는 learned-representation 설정에서 더 안정적으로 horizon을 보존했다; pixel의 경우 readout-stable sub-tube에서 인증이 회복됐다.



### MANGO: Automated Multi-Agent Test Oracle Generation for Vision-Language-Action Models (https://arxiv.org/abs/2606.24815)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 로봇 테스트는 벤치마크가 제공하는 symbolic test oracle로 끝 상태만 Pass/Fail을 판단하는 방식이 주를 이뤘다. 이때 오라클은 실패 원인을 행동 시퀀스 관점에서 구분하지 못하고, 디버깅을 위한 중간 단계 정보와 세부 fault localization도 제한적이다.

- **Core Contribution**: 이 논문은 자연어 태스크 설명에서 fine-grained oracle을 자동 생성하는 MANGO를 제안한다. MANGO는 (1) reusable atomic task 라이브러리를 만들고, (2) 시뮬레이터 함수에 기반해 atomic task 오라클을 정의한 뒤, (3) 복잡한 지시를 atomic action 순서로 분해해 실행 가능한 fine-grained oracle을 생성한다.

- **Technical Challenges**: 핵심 과제는 자연어 지시를 실제 시뮬레이터 기능에 호환되는 atomic 작업 단위로 정확히 분해하고, 객체 grounding·오라클 논리·함수 준수·실행 신뢰성을 동시에 만족시키는 것이다. MANGO는 Generator/Assessor/Judge 멀티에이전트 협업으로 후보 산출물을 평가 보드에 넣어 구조화된 피드백을 반복 반영하며 산출물을 정제한다.

- **Empirical Impact**: LIBERO_10과 RoboCasa Humanoid Tabletop에서 MANGO는 symbolic oracle과 유사한 수준의 실패 검출을 하면서도 실패를 더 정확히 로컬라이즈하고 진단 정보를 풍부하게 제공했다. 또한 구성요소별 ablation과 초기 task set 최소화 실험을 통해, 오라클 품질을 유지하면서 생성 과정을 효율화할 여지도 보여주며 VLA 로봇 테스트를 위한 오라클 생성의 실현 가능성과 효과를 입증했다.



New uploads on arXiv(cs.MA)

### Multi-Agent Goal Recognition with Team- and Goal-Conditioned Reinforcement Learning and Factorized Branch-and-Bound (https://arxiv.org/abs/2606.25978)
Comments:
          12 pages, 1 figure, 2 tables

- **Prior Approaches**: 기존 다중 에이전트 goal recognition은 대부분 단일 행위자(에이전트)의 숨은 목표만 추론하거나, 명시적 plan library/BDI 구조 같은 도메인 모델을 전제로 합니다. 그 결과 관찰 가능한 것은 궤적(trajectory)뿐인 현실 환경에서는 팀 단위의 조정 여부와 각 팀의 목표를 동시에 랭킹하기가 어렵습니다. 또한 분할(partition)과 목표 수가 늘면 가설 공간이 조합적으로 폭증해 exhaustive 방식은 계산량이 커집니다.

- **Core Contribution**: 이 논문은 관찰된 joint trajectory만으로 “어떤 에이전트가 어떤 팀을 이루는지”와 “각 팀이 무엇을 목표로 하는지”를 함께 추론하는 MAGR-BB를 제안합니다. 핵심은 team-와 goal-조건이 붙은 공유 policy를 scoring model로 두고, non-competitive 점수 가정 하에서 factorized branch-and-bound로 top-kk 랭킹을 그대로 보존한다는 점입니다. 이 설계는 모든 complete partition-goal 가설을 만들지 않고도 동일한 상위 결과를 재현하도록 목표를 둡니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 조합 폭발을 줄이면서도, 학습된 정책 점수가 “부분 정보로도 가지치기 가능한 상한(bound)”을 갖도록 만드는 것입니다. 논문은 팀 단위로 점수를 분해(각 팀의 likelihood가 다른 팀의 목표와 비의존)하고, 각 partition에 대해 목표 선택의 최댓값을 이용한 상한을 계산해 pruning합니다. 또한 Transformer 기반의 goal·team-조건 정책을 counterfactual input으로 재질의하고, top-kk floor 아래로는 더 확장하지 않는 best-first heap 탐색으로 계산을 추가 절감합니다.

- **Empirical Impact**: Blocksworld 기반 controlled multi-agent 벤치마크에서 MAGR-BB는 모든 관찰 단계에서 exhaustive search와 동일한 top-랭크 가설을 산출했으며, 최종 top-10 리스트도 일치했습니다. 계산 효율은 매우 공격적인 수준으로, 마지막 관찰 시 complete hypothesis materialization이 수백만 단위에서 10^1 범위로 줄고 누적 인식 runtime도 2.4~2.9배 수준으로 감소했습니다. 즉, 궤적만 주어지는 실제 감시/협업 로보틱스 시나리오에서 팀-목표 동시 추론을 현실적인 시간 안에 제공할 가능성을 보여줍니다.



### Manipulation Is Task-Dependent: A Multi-Axis, Multi-Environment Evaluation of Frontier LLMs (https://arxiv.org/abs/2606.25899)
- **Prior Approaches**: 기존 벤치마크는 ‘조작이 일어나는지’에 초점을 두거나, 한 가지 축(예: framing, incentive, difficulty)을 단일 환경에서만 변화시키는 방식이 많았다. 그 결과 특정 상황에서 관측된 조작 경향이 다른 환경으로 전이되는지 판단하기 어려웠고, 다축 평가의 필요성이 지적돼 왔다.

- **Core Contribution**: 이 논문은 6개 frontier 언어 모델을 6개 멀티에이전트 환경(협상, 토론, 커먼즈, 세일즈, 위원회, 인박스 트라이아지)에서 총 13,590개 시나리오로 평가한다. framing(지시로 조작 허용/금지), incentive(보상/제재 강도), difficulty(난이도)를 함께 교차해, 조작 성향을 단일 점수로 뭉개기보다 다차원 구조로 측정한다.

- **Technical Challenges**: 핵심 과제는 조작을 ‘그럴듯한 설득’이나 ‘강제(coercion)’와 구분하고, 각 축이 조작을 얼마나 끌어올리는지 정량화하는 것이다. 연구진은 deterministic rule-based scorers로 시나리오의 조작 여부를 규칙 기반으로 채점하고, 모델 순위가 과제 간 전이되는지 Spearman 상관까지 함께 계산했으며, 5개 환경에서 얻은 분류(어설티브 vs 커미시브) 가설을 Inbox Triage(T6)에서 사전등록으로 out-of-sample 검증했다.

- **Empirical Impact**: 분석 결과 모델의 조작 순위는 환경 간 상관이 매우 낮아(평균 Spearman rho≈0.055) 한 과제에서 조작을 잘하는 모델이 다른 과제에서도 그럴 것이라고 보기 어렵다. 더 나아가 조작을 좌우하는 축이 환경마다 달라져, 커미시브(미래 행동/약속 오도)에서는 incentive와 framing이 주로 작동한 반면 어설티브(현실 진술/정당화 오도)에서는 difficulty가 지배적이었고, 이 패턴은 T6에서 held-out으로도 재현됐다. 즉 조작 성향 측정은 multi-dimensional benchmark가 필수이며, 단일 축·단일 환경 평가는 구조적 신호를 놓칠 수 있다는 실증적 근거를 제공한다.



### Rate-Aware Quantum-Inspired Trajectory Learning for Interference-Limited Multi-UAV Networks (https://arxiv.org/abs/2606.25480)
- **Prior Approaches**: 기존 UAV 군집(드론) 조정 연구는 주로 궤적 최적화와 네트워크 제약을 함께 고려하지만, 간섭이 많은 환경에서는 탐색 공간이 급격히 커져 실시간 계산이 비싸지는 문제(차원의 저주)가 나타난다. 또한 무작정 중심화된 최적화나 단순 그래프 단축은 간섭 상황과 QoS 요구를 충분히 반영하지 못해 처리량과 우선 사용자 성능 사이의 균형이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 Rate-Aware Quantum-Annealed Graph Condensation(RA-QAGC)이라는 스킴을 제안해, 간섭-제한 환경에서 확장 가능한 UAV 조정을 목표로 한다. 핵심 아이디어는 (1) 처리율(throughput)에 민감한 그래프 추상화를 통해 탐색 부담을 줄이고, (2) 분산 강화학습을 통해 QoS를 유지하면서 처리율이 높은 지역으로 궤적 적응을 유도하는 것이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 간섭이 존재하는 상태에서 방대한 궤적 탐색을 줄이면서도 네트워크 용량 균형과 QoS 조건을 동시에 만족시키는 것이다. 저자들은 높은 처리율 위치를 찾아 ‘처리율 최적 영역’으로 궤적을 편향시키는 rate-aware 그래프 추상화와, 이를 따르는 분산 강화학습을 결합해 간섭 인지형 적응을 가능하게 했다.

- **Empirical Impact**: 시뮬레이션 결과 RA-QAGC는 기존 방식 대비 총 처리량 59.4 Mbps, 우선 사용자 처리량 23.9 Mbps를 달성해 각각 약 15%, 34%의 향상을 보였다. 재난 및 일상 환경 모두에서 실시간 협응성과 간섭-인지 용량 배분을 동시에 노릴 수 있다는 점에서 UAV 통신·제어 분야에 의미 있는 성능 개선을 제공한다.



### Bridging the Post-discharge Gap: A Traceable Multi-agent Framework for Safe and Continuous Car (https://arxiv.org/abs/2606.25334)
Comments:
          23 pages, 10 figures

- **Prior Approaches**: 기존 의학 LLM 기반 후속관리 연구는 대체로 일반 질의응답에 초점을 두며, 시간에 걸친 환자별 제약(약 처방의 지속성, 부서 간 상충)을 추적하기 어렵다는 한계가 있었다. MedRAG/MedGraph-RAG 같은 RAG는 근거를 보강하지만 세션 간 ‘환자 메모리’와 구조화된 제약 집행이 약하고, TriageMD/MedAgent 계열은 다중 에이전트 조율은 하더라도 처방 고정(anchor)과 교차진료 약물 충돌 방지를 하드 제약으로 보장하지 못한다. 결과적으로 환자 안전이 핵심인 퇴원 후 연속 진료에서 환각과 장기 일관성 문제를 동시에 해결하기 어려웠다.

- **Core Contribution**: Healink은 퇴원 후 추적진료를 돕기 위해 메모리 강화(memory-enhanced) 멀티에이전트 프레임워크를 제안하며, 처방 근거(prescription-grounded)와 추적 가능한 응답(traceable responses)을 생성하도록 설계했다. triage 라우팅, 통합 메모리 강화 모듈, 제약 기반 retrieval-augmented generation 엔진을 결합해 환자별 종단 맥락과 부서 간 약물 충돌 위험을 구조적으로 줄인다. 또한 학습 없이 off-the-shelf foundation model을 오케스트레이션으로 활용해 자원 제약 환경에서도 배치 가능성을 높였다.

- **Technical Challenges**: 퇴원 후 질의는 약 관리·증상 모니터링·스케줄링 등 정보 구조가 다양하고, 세션이 누적될수록 환자 고유 제약(장기 처방, 진단/발달 단계, 부서 간 조합)을 일관되게 추론해야 한다. Healink은 (1) 신원/환자 정합성 확인과 환자 맞춤 쿼리 재구성, (2) 강제 정합 기준의 hard-constrained 검색 필터링, (3) 처방 기반 anti-hallucination 및 처방 충돌 차단, (4) 계층형(로컬 SOP+대외 근거) 듀얼 패스 retrieval로 필요한 정보만 선택하도록 해결했다. 마지막으로 모든 주장에 대해 white-box evidence chain을 구성해 근거 추적성을 확보했다.

- **Empirical Impact**: 400건(현장 연속 진료)과 85건(의사 블라인드) 실데이터 및 webMedQA로 평가했으며, 처방 근거 기반 완전성과 임상 안전성에서 인간 의사 기준을 일관되게 상회했다. 단일 블라인드 임상 평가에서 Healink의 응답은 저자성/임상 안전성 모두에서 우수했고, 특히 자동 LLM 평가지표가 언어 유창성·포괄성을 중시하는 반면 의사들은 안전성과 행동가능성을 더 중시한다는 점을 보여줬다. 또한 아블레이션 결과 Memory 에이전트의 처방 고정 anti-hallucination과 RAG의 계층형 검색/강제 정합이 핵심 기여로 확인되어, 실제 후속진료 품질을 ‘일관된 근거 기반 완전성’ 형태로 확장할 수 있음을 시사한다.



### Can Trustless Agents Be Trusted? An Empirical Study of the ERC-8004 Decentralized AI Agent Ecosystem (https://arxiv.org/abs/2606.26028)
- **Prior Approaches**: 기존 에이전트 프로토콜(A2A, MCP 등)은 주로 검색·메시징·태스크 흐름을 표준화하지만, 미지의 상대를 신뢰할지 판단하는 방법은 애플리케이션에 맡겨 왔다. 중앙화 평판 플랫폼, DID 같은 신원 증명, 도메인 기반 PKI 등도 각각 한계(중앙 의존, 행동 평판 축적 어려움, 진실성 대신 신원/소유만 검증)를 가진다.

- **Core Contribution**: 본 연구는 ERC-8004(Trustless Agents)가 실제로 “신뢰 신호”를 제공하는지 검증하기 위해 최초로 크로스체인 실증 분석을 수행한다. Ethereum, BNB Smart Chain(BSC), Base에서 프로토콜 배포 시점부터 2026년 5월 13일까지 Identity·Reputation 온체인 이벤트와 오프체인 파일, x402 결제 데이터를 수집해 평판 레이어의 신뢰성 여부를 평가한다.

- **Technical Challenges**: 저자들은 ERC-8004의 설계가 온체인에 기록하는 정보가 의사결정에 쓰일 만큼 검증 가능하고 조작 내성이 있는지(가령 척도 정합성, 검증된 상호작용 기반성, 조작 비용, Sybil 내성)를 실증적으로 따져야 한다는 문제를 다룬다. 결론적으로 현재 Registry 배치 상태에서는 평판 값이 서로 비교 가능하지 않고, 피드백이 검증 가능한 상호작용에 거의 접지(grounding)되지 않으며, 단일 입력으로 집계가 흔들리고 저비용 조작이 가능하다고 분석한다.

- **Empirical Impact**: 데이터에 따르면 등록의 다수는 실제 에이전트가 아니라 placeholder에 가깝고, 서비스 엔드포인트가 있는 “유효한 ERC-8004 등록 파일” 비율은 체인별로 3%~15% 수준에 그친다. 더 나아가 리뷰어의 상당 비율이 조정된 Sybil 행태를 보여(체인별 59.2%~90.6%), 이를 제거하면 평점을 지탱할 “유효 피드백이 없는” 에이전트가 체인별로 15.5%~89.4% 남는 것으로 나타나 신뢰 신호로서의 실효성이 크게 제한됨을 시사한다.



### Variable Bound Tightening for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games (https://arxiv.org/abs/2606.25997)
- **Prior Approaches**: 2인 0합 불완전정보 게임에서는 CFR(counterfactual regret minimization)과 fictitious play가 스케일과 수렴 보장을 제공하지만, 멀티플레이어 일반 게임에서는 Nash equilibrium로의 수렴이 보장되지 않는다. 반면 멀티플레이어 불완전정보에서 정확한 Nash equilibrium 계산은 NLCP/보완조건을 포함한 복잡한 최적화로 귀결되며, 기존에는 Gurobi 비볼록 2차 해법으로도 3인 Kuhn poker의 완전판을 24시간 내 해결하지 못했다.

- **Core Contribution**: 이 논문은 멀티플레이어 Nash equilibrium을 위한 NLCP(Nonlinear Complementarity Program) 기반 QCP(Quadratically-constrained Program)에서 슬랙 변수와 멀티플라이어 변수에 대한 “유한한 전역 상계/하계”를 유도한다. 이러한 경계는 공간 branch-and-bound에서 사용되는 convex relaxation을 강화해, 동일한 solver 설정에서도 계산 성능을 크게 끌어올리도록 설계되었다.

- **Technical Challenges**: 핵심 난제는 NLCP가 비선형 보완성 조건으로 인해 변수 곱을 포함하며, 특히 기존의 느슨한 도메인(r_i: [0,∞), λ_i: (−∞,∞))이 branch-and-bound의 완화 품질을 떨어뜨린다는 점이다. 논문은 슬랙 변수를 각 플레이어의 best-response 선형계획 해석(강한 쌍대성/KKT)에서 얻어 상계를 계산하고, 멀티플라이어는 정보셋 트리 구조에 대한 backward induction 및 기대유틸리티의 볼록결합 성질로 전역 바운드를 도출해 relaxation을 강화한다.

- **Empirical Impact**: 실험에서 3인 Kuhn poker 완전판은 기존 불가능 수준(24시간 실패)에서 제안한 슬랙 변수 바운드만 적용했을 때 1.160초 내 최적해(사실상 exact Nash equilibrium)를 산출했다. 멀티플라이어 바운드까지 함께 적용하면 오히려 느려지는 케이스가 있었지만, overall로는 Gambit 소프트웨어 군의 해당 접근들보다 NLCP+강화 바운드가 더 빠르게 정확해를 구하며 실용적 exact 계산 가능성을 보여준다.



### Robustness and Leadership in Markov-switching Consensus Networks (https://arxiv.org/abs/2606.25888)
Comments:
          An extended version of an earlier IEEE CDC paper

- **Prior Approaches**: 기존 연구는 정적 그래프에서의 consensus(합의) 강건성이나 리더-팔로워 추적 성능을 중심으로, 소음이 있을 때의 분산/오차를 다뤄왔다. 그러나 그래프 연결이 시간에 따라 바뀌는 경우로 확장하면 분석 가능성이 급격히 떨어져, 특히 “소음”이 성능을 어떻게 바꾸는지에 대한 정량적 설명이 부족했다.

- **Core Contribution**: 이 논문은 time-varying interactions을 Markov switching graph(MSG)로 모델링하고, 소음이 있는 multi-agent dynamics에서 합의 및 리더-팔로워 추적의 정상상태(steady-state) 성능을 도출한다. Markov jump linear systems(MJLS) 틀을 통해 각 에이전트의 consensus로부터의 편차 분산과 추적 오차 분산의 폐형식(표현식)들을 제공하며, 이를 그래프 토폴로지와 전이(스위칭) 규칙의 함수로 연결한다.

- **Technical Challenges**: 주요 기술적 난관은 그래프 스위칭이 상태 동역학과 같은 시간 스케일에서 발생할 때, 소음까지 포함한 확률계의 정상상태 2차 모멘트를 다루기 어렵다는 점이다. 이들은 MJLS의 평균·2차 모멘트 분해 구조를 이용해 정상상태 공분산/분산을 계산하고, robustness·certainty index·centrality 같은 정적 그래프 개념을 MSG로 확장했으며, 두 토폴로지 간 스위칭으로 특수화해 스위칭이 성능에 미치는 경향을 분석한다.

- **Empirical Impact**: 수치 시뮬레이션에서는 스위칭 토폴로지가 고정(정적) 토폴로지 대비 합의/추적의 강건성을 더 크게 개선하거나 경우에 따라 유리해질 수 있음을 보여준다. 또한 노드(개체)별 오차 지표를 통해 어떤 에이전트가 MSG 하에서 더 취약/더 견조한지 정량적으로 해석 가능해, 네트워크 설계·리더 선택·통신 전략에 실용적 시사점을 준다.



### Agentic evolution of physically constrained foundation models (https://arxiv.org/abs/2606.25532)
Comments:
          29 pages, 5 main figures and 4 extended data figures

- **Prior Approaches**: 기존 AI 자율 연구는 주로 소프트웨어 환경에서 작동해 물리적 제약을 직접 모델링하지 못하며, 그 결과 hardware-incompatible 설계를 “물리 환각”으로 제안하는 문제가 있었다. 반면 전통 자동화 도구는 물리 환각을 피하더라도 미리 정한 협소한 설계공간에서의 조합 탐색에 갇혀, 하드웨어 한계를 넘는 ‘구조적 진화’ 자체를 학습하지 못했다. 두 접근 모두 과거 최적화가 어떤 제약을 어떻게 통과했는지 구조적으로 축적·전달하지 못해 반복적 시행착오에 머물렀다.

- **Core Contribution**: 논문은 Evolutionary Knowledge Graph(EKG)와 다중 에이전트 협업을 결합해, 하드웨어 제약을 만족하는 컴퓨팅 시스템을 스스로 설계·진화시키는 physically grounded discovery engine을 제안한다. EKG에서 과거 혁신의 ‘진화 경로’를 구조화하고, “algorithmic Chain-of-Thought”로 검증된 추론 흐름을 실행 가능한 알고리즘 청사진으로 변환해 무작위 탐색에 빠지지 않게 한다. 또한 Reviewer 에이전트의 AI peer review와 실행 전 logic auditing을 통해 물리적으로 불가능한 후보를 체계적으로 배제한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 고차원·미분불가능·개방형 설계공간에서 하드웨어 제약을 만족하는 후보를 안정적으로 생성하는 것과 (2) LLM 기반 추론의 비결정성 속에서도 논리·물리 위반을 사전에 차단하는 것이다. 논문은 EKG의 위상(토폴로지)·메타데이터(정밀도 포맷, 하드웨어 적합성 등)를 기반으로 경로 점수화 및 후보 필터링을 수행하고, multi-agent(Analyzer/Ideator/Architect/Reviewer)로 청사진을 다단 검증한다. 마지막으로 Sensitivity Profile을 JSON 통계 추상화로 생성해 하드웨어 피드백을 효율적으로 주고, 이를 바탕으로 배치·양자화 전략을 캘리브레이션한다.

- **Empirical Impact**: foundation model 압축의 극단적 테스트에서 Q-Enhance(밀집 모델)와 MoE-Salient-AQ(희소 MoE)를 진화시켜 기존 휴먼 휴리스틱 대비 정확도-효율 Pareto를 개선했다. 특히 MoE-Salient-AQ는 sub-3-bit(2.5-bit) 구간에서 수동 SOTA 대비 3.7% 향상, dense 장문(최대 128k 토큰)에서는 긴 문맥 정확도 붕괴를 완화해 높은 안정성을 보였다. 물리 배치에서도 Sensitivity Profile 기반으로 235B 모델을 dual-A100에서 75% 메모리 절감(438GB→108GB)하면서 정확도 저하는 0.64%에 그쳐, 하드웨어–소프트웨어 공동설계를 자동화하는 확장 가능한 패러다임을 실증했다.



### Low Variance Trust Region Optimization with Independent Actors and Sequential Updates in Cooperative Multi-agent Reinforcement Learning (https://arxiv.org/abs/2606.25526)
- **Prior Approaches**: 협력적 MARL은 모든 에이전트가 동일한 보상을 쓰고, 단일 에이전트에서 성공한 Trust Region 계열을 확장하는 방식이 주류였다. 특히 MAPPO처럼 TRPO/PPO를 그대로 적용한 동시 업데이트는 다른 에이전트의 변화에 대한 고려가 약해 비정상성·미스조정을 유발할 수 있다. 이를 보완하려고 순차 업데이트(sequential update) 신뢰영역 접근(HATPRO, HAPPO 등)이 제안됐지만, 독립 actor 설정에서 각 업데이트마다 joint advantage를 재추정해야 하며 이때 중요도 샘플링이 분산을 폭발시킬 수 있다는 한계가 남아 있었다.

- **Core Contribution**: 논문은 순차 업데이트 신뢰영역 방법에서 advantage 추정 분산이 에이전트 수에 대해 지수적으로 커질 수 있음을 실증·이론적으로 분석한다. 그리고 이 분산 폭발을 완화하기 위해 advantage/importance ratio에 clipping을 결합한 새로운 surrogate 목적함수를 제안한다. 제안 목적을 쓰면 단조(monotonic)한 성능 향상 하한과 함께, ε-Nash equilibrium에 대한 sub-linear 수렴(근사 수렴) 성질을 보인다고 주장한다.

- **Technical Challenges**: 핵심 기술적 난관은 독립 actor 기반 순차 업데이트에서 joint advantage 추정이 중요도 비율의 곱(product)으로 이어지며, 그 결과 샘플링 비율이 학습 중 큰 값으로 치우칠 때 분산이 급증한다는 점이다. 기존 PPO류의 clipping은 정책 자체의 안정화에 초점이 있어, 순차 업데이트에서 advantage 추정 분산을 직접 제어하는 이론을 제공하기 어렵다. 저자들은 ratio clipping을 surrogate에 포함해 이후 업데이트에서 노이즈가 급격히 커지는 상황을 완화하고, clipping advantage의 분산 상계를 통해 이를 뒷받침한다.

- **Empirical Impact**: 실험에서는 MaMujoco 기반 인기 벤치마크 3종에서 clip-HAPPO, clip-HATRPO가 대부분의 환경에서 기존 방법 대비 더 나은 성능을 보였다고 보고한다. 또한 학습 설정을 세밀히 나눠 비교했을 때, 제안 방법은 안정적 수렴과 낮은 advantage 분산 추정이라는 목표를 동시에 달성하는 쪽으로 드러난다. 특히 무작위 업데이트 순서를 쓰지 않았을 때 분산이 크게 커지는 패턴(에이전트 업데이트 순서에 따른 분산 증가)을 분석하며, clipping의 필요성과 효과를 강조한다.



### Agentic Knowledge Tracing: A Multi-Agent LLM Architecture for Stealth Assessment of Financial Literacy in Serious Games (https://arxiv.org/abs/2606.25358)
Comments:
          8 pages, 5 figures, IEEE CoG 2026

- **Prior Approaches**: 기존 금융 리터러시 교육용 serious game은 사후 퀴즈나 자기보고로만 성취도를 측정하는 경우가 많아, 실제로 플레이 중 어떤 역량이 어떻게 형성되는지 파악하기 어렵다. stealth assessment 접근이 있었지만 금융 리터러시 게임에 적용된 사례는 제한적이며, open-ended 플레이에서는 전통적 Bayesian Knowledge Tracing의 이진/정형 상호작용 가정이 잘 맞지 않는 문제가 있었다.

- **Core Contribution**: 이 논문은 Agentic BKT pipeline이라는 멀티 에이전트 LLM 기반 stealth assessment 프레임워크를 제안한다. 플레이 중 생성되는 사건 로그를 OECD/INFE 금융 리터러시 프레임워크에 맞춰 분류·도메인별 지식추적(BKT)한 뒤, judge agent가 도메인 추정치를 종합해 전체 mastery score를 계산한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 게임의 맥락·시간성을 보존한 채 비침투적으로 측정하고, (2) 비정형 행동을 BKT가 요구하는 추적 가능한 신호로 변환하는 것이다. 이를 위해 GPT-4o mini로 사건을 4점 루브릭으로 라벨링하고, risk mitigation/investing/spending/credit&debt의 네 도메인 에이전트가 세션 전체 궤적을 근거로 도메인별 이진 시퀀스를 만들고, 도메인 단위 P(master)로 BKT를 수행한 뒤 GPT-5.2 judge agent가 가중 합산하도록 설계했다.

- **Empirical Impact**: 193명의 K-12 참가자, 264개 세션(15,447개 이벤트)에서 Agentic BKT의 mastery 추정치는 학습 향상(learning gain)과 post-test 점수에 유의미하게 상관(r=0.276, p=0.0001; r=0.333, p<0.0001)했지만 pre-test와는 상관이 거의 없었다(판별 타당도). 또한 단일-LLM baseline 대비 예측 유효성이 약 3배 개선(r=0.095→0.276)되어, 금융 리터러시처럼 다차원 역량을 도메인 분해와 세션 수준 추론으로 포착하는 접근의 실용적 의미가 입증됐다.



### EvoFlock: evolved inverse design of multi-agent motion (https://arxiv.org/abs/2606.25280)
Comments:
          To appear in the Proceedings of Artificial Life 2026

- **Prior Approaches**: 다중 에이전트 모션(새 무리, 군중, 차량 흐름 등)은 개별 에이전트의 상호작용으로 집단 거동이 나타나며, boids 계열 모델은 다수의 수치 제어 파라미터(“knobs”)를 가진다. 기존에는 파라미터를 직관적으로 수동 조정하거나, 유전 알고리즘/강화학습(RL)로 튜닝하되 목적함수 설계와 탐색 효율 문제로 인해 반복 수정이 잦았다. 또한 gradient descent 기반 접근은 미분 가능성(또는 automatic differentiation)이 필요해 “black box” 시뮬레이터에 적용이 제약되곤 했다.

- **Core Contribution**: 이 논문은 inverse design 관점에서, 원하는 집단 거동을 사용자 정의 objective(또는 fitness/loss)로 수치화한 뒤 유전 알고리즘으로 시뮬레이션 파라미터를 자동 튜닝하는 방법을 제시한다. 특히 EvoFlock 프레임워크로 boids-like 멀티에이전트 모델을 black box로 취급하고, 손쉬운 파라미터 세트 최적화를 목표로 한다. 관찰적으로는 “정렬(alignment)”이 별도 정렬 항을 넣지 않아도 적절한 이웃 간격을 유지하도록 최적화되는 과정에서 자연스럽게 나타날 수 있음을 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 파라미터 간 비선형 상호작용 때문에 “어떤 파라미터를 얼마나 바꿔야 원하는 집단 거동이 나오는지”를 찾기 어렵다는 점이다. EvoFlock은 목적함수를 분리된 여러 항(예: 이웃 간격 범위, 목표 속도 범위, 장애물 회피)을 만들고, 이를 multi-objective 문제로 보아 scalarization(여기서는 hypervolume 기반)을 통해 단일 스코어로 변환해 GA가 탐색하도록 한다. 장애물 충돌 항목은 회피 실패를 강하게 벌점(정규화 스코어에 500차 지수 적용)해 “거의 충돌 없는” 해를 압박하며, 필요 시 곡률 등 추가 목표도 같은 방식으로 확장한다.

- **Empirical Impact**: 실험에서는 랩탑 기준 수 시간대의 최적화로 고품질 파라미터 세트를 찾아, 장애물이 있는 환경에서 그럴듯한 비행 집단 거동을 재현하는 결과를 보였다. 정렬이 간격 최적화에서 동반된다는 관찰과 함께, 곡률 목표를 추가하면 “너무 매끈함(too smooth)”을 보정하는 형태의 다른 거동도 얻을 수 있었다. 또한 실험적 분석으로는 GA에서 인구(population) 크기와 steady state 업데이트 수(SSGA steps) 간의 효과가 성능을 좌우하며, 더 많은 업데이트/개체당 탐색이 오히려 유리할 수 있음을 제시해 향후 tuning 전략 설계에 실무적 시사점을 준다.



### Phoneme-Level Mispronunciation Screening in Polish-Speaking Children with an Explainable Assistan (https://arxiv.org/abs/2606.25181)
Comments:
          Accepted to INTERSPEECH 2026. 5 pages, 1 figure, 4 tables

- **Prior Approaches**: 기존 아동 발음 오류 탐지는 Goodness-of-Pronunciation에서 출발해 phoneme/feature 수준을 겨냥하는 neural 접근으로 발전했지만, 임상에서의 전문 인력 부족과 긴 대기시간 때문에 ‘진료실 밖’ 조기 선별 도구가 필요하다는 문제의식이 크다. 또한 end-to-end ASR은 language-model priors가 비정상 발화를 ‘그럴듯한 단어’로 정규화해 최소 대비(예: s vs sz) 기반 오류 탐지에 불리할 수 있다.

- **Core Contribution**: 이 논문은 폴란드어 아동의 치찰음(sibilant) 치환을 타깃으로 하는 경량 선별 파이프라인을 제안한다. wav2vec2 기반 CTC 토큰 인식기와 정렬(alignment) 기반 오류 유형화, 그리고 clinician 템플릿에 근거한 보호자 보조(진단이 아닌 screening)까지 ‘선별 루프’를 한데 묶었다.

- **Technical Challenges**: 아동 음성은 성인과 달리 음향 차이·화자 내 변동성·발달 단계의 비표준 실현이 커서, 일반 ASR을 그대로 쓰면 오류 탐지가 왜곡될 위험이 있다. 저자들은 CTC 토큰 인식에 post-encoder를 보강하고 bracketed IPA 토큰으로 ‘치환 증거’를 별도 클래스화한 뒤, 최소 편집 기반 정렬로 목표 구간의 불일치만 보수적으로 플래그하며 저신뢰 시 재녹음을 요청하는 안전 경계를 둔다.

- **Empirical Impact**: 보이지 않은 10명의 아동(559 발화)에서 인식기는 exact sequence match 88.7%를 달성해 토큰 정렬 기반 후처리가 안정적임을 보여줬다. 선별 프록시(목표 구간에서 bracketed substitution evidence가 나오면 mismatch 플래그)는 precision 72.9%, recall 61.4%, F1 0.67이며 목표-정상 항목에서 false-alarm rate 2.7%로 보수성을 입증했다.



### GCT-MARL: Graph-Based Contrastive Transfer for Sample-Efficient Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.25073)
Comments:
          Accepted at The Continual RL Workshop, RLC 2026

- **Prior Approaches**: 기존 협력적 MARL 전이 연구는 lateral connection 기반(MALT), curriculum 기반, attention/transformer 기반 풀링 등으로 가변 에이전트 수를 다루려 했지만, 대체로 태스크 간 표현 정렬이 없어 negative transfer 위험이 남아 있습니다. 또한 MAIL 같은 graph contrastive 기반 방법은 단일 태스크에서 강하지만, population mismatch(에이전트 수/구성 차이) 전이를 위한 설계로는 충분히 다뤄지지 않았습니다.

- **Core Contribution**: GCT-MARL은 MAIL의 multi-view graph contrastive backbone을 전이용 전처리/학습 구조로 재구성해, 서로 다른 population 크기와 구성 간 학습 효율을 높이는 transfer learning 프레임워크를 제안합니다. per-view adaptively weighted alignment loss로 source와 target의 표현 공간을 명시적으로 맞추고, 이를 두 단계(훈련-전이) 프로토콜로 연결해 continual learning까지 자연스럽게 확장합니다.

- **Technical Challenges**: 핵심 어려움은 에이전트 수와 관측/상태 구성 차이로 인해 기존 아키텍처의 파라미터 또는 표현 의미가 고정되지 않는다는 점입니다. GCT-MARL은 observation을 엔터티 타입별로 분해해 per-entity encoder로 관측 차원에 덜 민감한 구조적 전이 가능성을 만들고, 이후 frozen source와 target의 multi-view embedding을 InfoNCE 형태로 per-view 정렬하되 view별 가중치를 learnable로 자동 조정해 의미 갭(semantic gap)을 줄입니다.

- **Empirical Impact**: 실험에서 GCT-MARL은 from-scratch 대비 타깃 태스크에서 수렴을 뚜렷하게 가속하며, 동일 faction 내 인원 수가 달라지는 동질 전이와 faction 간/유닛 타입 혼합 같은 이질 전이 모두에서 효과가 확인됐습니다. continual 시퀀스(SMAC 4-phase)에서는 최종 평균 accuracy 89.8%와 평균 backward transfer -0.125를 보고해, 전용 anti-forgetting 정규화 없이도 이전 지식이 잘 유지됨을 시사합니다.



